---
source_pdf: SpatialVLA.pdf
paper_sha256: d63e7346ca0f74282a9a575c05ea025318d40c72883874a2de1841a90d0e791e
processed_at: '2026-08-12T09:47:24-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SpatialVLA

Andrej，我换个画风，不堆公式不堆表，直接讲"这帮人到底干了啥、为啥这么干、为啥 work"。

---

## 一句话总结

**现在所有 VLA model 都是"半瞎"的 — 它们看的是 2D 平面照片，输出的是 256 个等宽桶子离散化的 action，两端都丢了 3D 空间结构。SpatialVLA 给 observation 端塞了个 3D 坐标"补丁"，给 action 端按真实数据分布重新划格子，结果用 3.5B 参数打爆了 55B 参数的 RT-2-X。**

---

## 它在解决什么"直觉性"问题？

你想想人怎么抓杯子：
- 你**不是**看一张 2D 照片就开抓
- 你脑子里有个 **3D 空间地图**，知道杯子在你左前方 30cm、桌面以上 10cm
- 你的动作是个**完整的 3D 向量**（往左前下方伸手），而不是 7 个独立维度各走各的

但现在的 VLA 模型：
- 看到的只是 224×224 的 RGB pixel
- 输出 action 是把 x, y, z, roll, pitch, yaw, grip **各切成 256 个等宽的桶**，然后独立预测 7 个 token

这有两个大问题：

### 问题 1：观察是 2D 的

OpenVLA、RT-2、π0 都只看 RGB image。你给它换个 lighting、换张桌布、相机挪个角度，它就懵了 — 因为它没有"物体在 3D 空间哪里"的概念，它学的全是 2D texture pattern。

### 问题 2：动作离散化太粗暴

robot 的动作分布**极度不均匀**。paper 里 Fig. 13 画了一张散点图，几乎所有数据点都挤在中心椭球里，2-sigma 之外几乎没有数据。你给它 256 个等宽桶，99% 的桶永远是空的，真正高频的小动作反而分辨率不够。

这就像你给一个中文 NLP model 做分词，把字符均匀切分成 256 个 ASCII 区间 — 完全没利用 token 出现频率。

---

## SpatialVLA 的两招

### 第一招：Ego3D Position Encoding（给 2D feature "长出 3D 维度"）

**人话**：对每个 image patch，算出它对应的 3D 坐标，把这个坐标编码后加到 feature 上。

具体怎么干：
1. 用 ZoeDepth（一个深度估计 model）预测每个 pixel 的 depth
2. 用相机内参 back-project 回 3D 坐标 — 注意是**相机坐标系**，不是 robot 坐标系
3. 每个像素的 3D 坐标过 sinusoidal encoding + MLP，变成一个 embedding
4. **直接加**到 SigLIP 输出的 2D feature 上

为啥是"ego"centric？因为不同 robot 相机装的位置不一样（有的装机械臂上，有的装旁边支架上），如果用 robot base 坐标系，每个 robot 都得标定。用**相机自己的坐标系**，不管啥 robot，只要给它 RGB + 内参 + 深度，输出就是统一的 egocentric 表征。

**一个反直觉的发现**：用 ZoeDepth 预测的深度比用 RealSense 实际测的深度效果还好（72.7% vs 70.5%）。为啥？因为 sensor depth 有噪声、有空洞、有边缘伪影，而 ZoeDepth 是个 prior-based 平滑深度，给的是"合理的空间结构"而不是"精确的毫米测量"。token-based model 要的是前者。

### 第二招：Adaptive Action Grids（按数据分布划格子）

**人话**：别再用 256 个等宽桶了，按数据实际分布来分 — 数据多的地方分细一点，数据少的地方分粗一点。

具体怎么做：
1. 把 7D action 拆成三组：translation (x,y,z)、rotation (roll,pitch,yaw)、gripper
2. translation 转成 polar coordinates $(\phi, \theta, r)$ — 解耦方向和距离
3. 在整个 dataset 上 fit Gaussian 分布
4. 用 Gaussian 的 inverse CDF（PPF）**等概率**划分 grid — 每个格子的数据量一样多
5. translation 划成 $16 \times 32 \times 8 = 4096$ 格（方向更细，距离粗）
6. rotation 划成 $16 \times 16 \times 16 = 4096$ 格
7. gripper 就 2 格（开/合）
8. 总共 8194 个 action token

**关键**：预测时**一次只生成 3 个 token**（trans / rot / grip 各一个），而不是 OpenVLA 的 7 个。autoregressive decode 是 $O(L^2)$，token 数减半，推理快近 4 倍，在 RTX 4090 上能跑到 20Hz。

这相当于把 3D action 空间看成一个 codebook，每格是一个 "spatial word"。模型学的是 "3D 空间哪个格子里的动作该干啥"，这就直接把空间结构编进了 vocab。

---

## Post-training 那一招也很妙

当你 fine-tune 一个新 robot 时，它的工作空间和动作分布可能不一样。SpatialVLA 的做法：
1. 在新数据上重新 fit Gaussian
2. 重新划格子（新格子位置可能跟旧的不重合）
3. 新格子的 embedding 用**三线性插值**从旧格子初始化 — 把旧 codebook 当成一个 3D 体积场，新格子是采样点

这比 LoRA 直接学低秩矩阵聪明多了 — pre-trained 的 spatial embedding 已经编码了"3D 空间某位置附近该做什么动作"的知识，新 robot 只是这个空间的另一个采样者，插值传递既快又准。

---

## 为什么这招特别 work？我看出的几条直觉

### 1. 3 tokens vs 7 tokens — 推理加速是硬通货

autoregressive 是 $O(L^2)$ attention + $O(L)$ KV cache。token 从 7 降到 3，推理速度差不多 4×。这把 VLA 推到 20Hz 可部署区间，**这是工程上最值钱的贡献**。

### 2. Gaussian partition 比 uniform bins 提升惊人

Ablation 里 Pick Coke Visual Matching 从 linear 256 bins 的 19% 飙到 adaptive 8194 的 70.7% — **+51.7% 的 gap**。这说明过去 OpenVLA / RT-2 的 action tokenization 是个隐性瓶颈，大家都在视觉 encoder / LLM backbone 上卷，但 action 那端一直没人动过。

### 3. Ego3D 在环境变化下特别值钱

Variant Aggregation 测的就是 lighting/texture/camera pose 变化。加 Ego3D 后 Pick Coke 从 68.9% 升到 81.6% — 因为 3D 坐标告诉你"杯子还是那个杯子，只是看起来不一样"，model 不再被 2D appearance 变化骗到。

### 4. Frozen text embedding — 别丢了世界知识

PaliGemma 的 text embedding 在几十亿图文对上学了 language grounding，如果你让它跟着 robot 数据漂移，就会 catastrophic forgetting。冻住它，只 train spatial embedding + vision encoder + LLM backbone — instruction following 能力保留得很好。

---

## 它的短板（也是 follow-up 机会）

### 1. 单帧观察 → 长 horizon 拉胯

LIBERO-Long 任务表现一般，因为只看当前帧 + 历史预测 tokens，没有显式 memory 机制。π0 用 diffusion chunks、RDT 用 1.2B diffusion model 处理这个更好。

### 2. Gaussian 假设会塌缩

如果 robot 大部分时间只在一个轴上动（比如推抽屉只往一个方向推），Gaussian 会塌缩成扁饼，其他轴的分辨率丢失。paper 自己承认了，建议未来用 VAE latent space 或 kernel density estimation。

### 3. 3 tokens 对 humanoid / bimanual 不够

单臂 7D 用 3 tokens 够了。双臂 14D？humanoid 30+ DOF？3 tokens 表达力不够。paper 在 Limitations 提到未来要 "share action grids among embodiments like dexterous hands, bimanual arms, and legs" — 这是个开放问题。

### 4. ZoeDepth 是 frozen 的

它是个 generic depth prior，没在 manipulation 数据上 fine-tune 过。一个 robot-specific depth model 可能更好，但成本高。

---

## 它让我联想到的东西

- **NeRF 的 positional encoding**：Ego3D encoding 本质上就是 NeRF-style 的 $\gamma(x)$ 编码用在 robot policy 上
- **3D Gaussian Splatting**：Action grid 的 trilinear interpolation init 跟 3DGS 用 sparse 点初始化 density field 有点神似
- **VQ-VAE codebook**：8194 个 action token embedding 就是个 codebook，但 codebook 的几何位置是已知固定的（由 grid 划分决定），所以可以插值
- **FAST (Pertsch 2025)**：同期的 action tokenization 工作，用 VQ 学 codebook，没显式利用 spatial 结构 — SpatialVLA 用结构化 grid + Gaussian partition 是显式 prior 的方式
- **CogACT**：用 diffusion action head 完全连续输出 — 是 adaptive grids 的另一个极端，混合两者可能是方向
- **HPT (Karpathy 你老板那个组)**：用 stem module 对齐异构 robot 的 representation，思路类似但更偏 "learn 对齐"，SpatialVLA 是 "geometry 对齐"

---

## 我的整体判断

这篇 paper 最大的贡献**不是某个单点突破，而是指出 VLA 在 spatial 维度上一直欠债**：

- Observation 端：所有人都用 2D image，没人认真加 3D
- Action 端：所有人都用 256 uniform bins，没人想过数据分布

SpatialVLA 把这两块一起补了，结果用 3.5B 参数在 SimplerEnv 上 zero-shot 打败了 55B 参数的 RT-2-X，在 LIBERO-Spatial 上拿到 88.2%。**Ego3D encoding 和 Adaptive grids 这两个 trick，大概率会被未来 VLA 工作当 default 组件用**。

它的局限是单帧 + Gaussian 假设 + 3 tokens 表达力天花板 — 这些都是清晰的 follow-up 入口。如果你要 follow up，我会赌方向是：
1. 把 action grid 换成 implicit latent space + equivariant MLP（解决 Gaussian 塌缩 + SO(3) 结构）
2. 加 multi-view epipolar encoding（解决单帧限制）
3. diffusion head 替代 autoregressive decode（解决长 horizon）
4. 扩展到 bimanual / humanoid（解决 3 tokens 表达力）

---

## 一些链接

- SpatialVLA Project: https://spatialvla.github.io
- PaliGemma 2: https://arxiv.org/abs/2412.03555
- Open X-Embodiment: https://robotics-transformer-x.github.io
- RH20T dataset: https://rh20t.github.io
- OpenVLA: https://github.com/openvla/openvla
- π0: https://arxiv.org/abs/2410.24164
- Octo: https://octo-models.github.io
- SimplerEnv: https://simpler-env.github.io
- LIBERO: https://lifelong-robot-learning.github.io/libero
- ZoeDepth: https://arxiv.org/abs/2302.12288
- 3D-LLM: https://3dllm.github.io
- 3D-VLA: https://arxiv.org/abs/2403.09631
- TraceVLA: https://arxiv.org/abs/2412.10345
- FAST: https://arxiv.org/abs/2501.09747
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu
- HPT: https://arxiv.org/abs/2409.20537

---

**最后一句大白话**：SpatialVLA 说"别再把 VLA model 当纯 2D image classifier 训了，robot 活在 3D 世界里，observation 要带 3D 坐标、action 要按数据分布分格子、跨 robot 要靠几何插值传递知识"。三句话听起来都直觉到不行，但合起来在 OXE 这种异构大数据上 pretrain 出来，3.5B 打 55B。

---

# SpatialVLA: 给 VLA 装上 3D Spatial 眼睛与 Action 的 Spatial 化重新定义

Andrej，这篇 paper 我读得很兴奋，因为它恰好打在我最近一直在思考的痛点上：**所有 VLA model 都在 2D pixel 平面里"假装"理解物理世界，action 又用粗鲁的 256 bins 线性离散化**。SpatialVLA 同时攻破了这两端 — observation 端注入 egocentric 3D position encoding，action 端用 Gaussian-driven adaptive action grids 替代 uniform bins，并且把 7D action 压成 3 tokens，推理速度比 OpenVLA 快一大截。它的 zero-shot 性能甚至用 3.5B 参数超过了 55B 参数的 RT-2-X，在 LIBERO-Spatial 上拿到 88.2% 成功率。下面我把每个设计拆开讲，重点 build 你的 intuition。

---

## 1. 核心问题与 motivation

先看一下它要解决的两个根本 heterogeneity 问题：

| 问题 | 具体表现 | SpatialVLA 的解决思路 |
|------|---------|---------------------|
| Observation 不对齐 | 不同 robot 的 camera 装在不同位置（wrist vs third-person），3D 观测空间不统一 | Ego3D Position Encoding：以相机自身为坐标系原点，免去 robot-camera 标定 |
| Action 不对齐 | 不同 robot 自由度、controller、workspace 都不同，连续 7D action 难以跨机器人共享 | Adaptive Action Grids：按数据分布自适应划分 spatial grids，并允许 post-training 时 re-discretize |

**关键直觉**：人脑里有 "cognitive map"（Tolman 1948 的经典概念，paper 引用了 [63]），我们在 manipulation 时本能地构建结构化空间表征，并且把物体对齐到一个直觉化的 workspace。SpatialVLA 想给 VLA 模型装上类似的"空间感"。我注意到 paper 在 Related Work 里把 3D-LLM、LLaVA-3D、LEO、3D-VLA 都列出来 [24, 10, 25, 69] 作对比，批评它们都忽略了 **action 端的 3D spatial 特性** — 这一点我觉得是这篇 paper 站位最准的地方。

Reference:
- Project page: https://spatialvla.github.io
- Open X-Embodiment: https://robotics-transformer-x.github.io
- π0 (Physical Intelligence): https://arxiv.org/abs/2410.24164

---

## 2. Ego3D Position Encoding：让 2D visual feature "长出" 3D 维度

### 2.1 架构总览（Fig. 2 解析）

SpatialVLA 的 pipeline 是：

```
RGB image o_t ──┬──> SigLIP encoder ───> 2D semantic feature X ∈ R^(d×h×w)
                │                              │
                └──> ZoeDepth ───> depth D ───> back-projection π^(-1) ──> 3D position P ∈ R^(3×h×w)
                                                                          │
                                                              sinusoidal γ(·) + MLP
                                                                          │
                                                                       P' ∈ R^(d×h×w)
                                                                          │
                                                              O_3d = X + P'  (公式 2)
                                                                          │
                                                                          v
                                            Language instruction L ──> Gemma2 LLM
                                                                          │
                                                                          v
                                                            autoregressive predict
                                                                          │
                                                                          v
                                            3 spatial action tokens (trans/rot/grip)
                                                                          │
                                                              de-tokenize ──> continuous action A_t
```

### 2.2 公式 (2) 的变量详解

$$\mathbf{O}_{3d} = \mathbf{X} + \mathbf{P'} = \mathbf{X} + \text{MLP}(\gamma(\mathbf{P}))$$

- $\mathbf{X} \in \mathbb{R}^{d \times h \times w}$：SigLIP 输出的 2D semantic feature。$d=1152$（SigLIP 维度），$h=w=224/14=16$（SigLIP 的 patch size 是 14）
- $\mathbf{P} \in \mathbb{R}^{3 \times h \times w}$：每个 patch 中心像素在 egocentric 3D 坐标系下的 $(x,y,z)$ 坐标，通过相机内参 + ZoeDepth 的 depth map back-projection 得来
- $\gamma(\cdot)$：sinusoidal positional encoding 函数，类似 NeRF 的位置编码（paper 没说具体 L，但我推测是 10 个 frequency，所以输出维度是 $3 \times (2L+1) \times 1$ 那种结构，最终拼成 204 维，因为 Tab. VII 里 Ego3DPositionEmbeddingMLP 的 input shape 是 `(B, 204)`）
- $\text{MLP}(\cdot)$：Linear → LayerNorm → ReLU → Linear，把 204 维映射到 1152 维（Tab. VII 中明确给出）
- $\mathbf{P'} \in \mathbb{R}^{d \times h \times w}$：编码后的 3D 位置 embedding，与 2D feature 同维度
- **相加** $\mathbf{X} + \mathbf{P'}$：用加法而不是 concatenation，这保留了原 2D semantic feature 的全部维度用于 LLM 处理，同时把 3D 信号"调制"上去 — 类似 ViT 中的 positional embedding 加法范式

### 2.3 为什么是 "Ego"-centric？

这点是整个设计的灵魂。**Ego3D 是相机坐标系下的 3D 坐标**，不经过任何 robot-camera 标定。这意味着：

- WidowX 第三人称相机 → 装在固定支架，ego frame 就是支架坐标系
- Franka 外部相机 → 同理
- wrist camera → ego frame 跟着机械臂末端走

任何 robot，只要给它 RGB + 内参 + 深度（预测或测量），都能得到一致的 egocentric 表征。**这本质上是把"机器人异构"问题甩给了"相机统一"问题** — 而所有 robot 都能产生相机视角下的 RGB。我觉得这是 paper 最聪明的 trick 之一，比 3D-LLM 那种需要 scene-level 3D 重建 + 5cm voxel grid 的做法轻量得多。

### 2.4 为什么用 ZoeDepth 而不是 sensor depth？

这点在 Appendix C Q3 给出了反直觉的答案：
- OXE + RH20T 中 28 个数据集都没有 sensor depth，所以 pretraining 必须用 ZoeDepth 预测
- 但即使在有 sensor depth 的 fine-tuning 阶段，ZoeDepth 反而更好（72.7% vs 70.5%）
- 原因：sensor depth 噪声大、有空洞、有边缘伪影；ZoeDepth 是 prior-based 平滑深度，更适合 VLA 这种 token-based 学习
- Tab. IX 给出对比：w/o depth 45.4%，sensor depth 70.5%，Zoe depth 72.7% — Zoe 比 no-depth 高了 27.3%，这是巨大 gap

**我的 intuition**：ZoeDepth 起到了一个 "spatial prior regularizer" 的作用，把训练时的 noisy depth 信号平滑掉了。这有点像 NeRF 训练时用 COLMAP pseudo-depth 比用 ARKit raw depth 更稳定 — prior-based depth 给的是 "合理的空间结构" 而不是 "精确的毫米测量"，正好是 token-based model 想要的。

Reference:
- ZoeDepth: https://arxiv.org/abs/2302.12288
- SigLIP: https://arxiv.org/abs/2303.15343
- PaliGemma 2 (backbone): https://arxiv.org/abs/2412.03555

---

## 3. Adaptive Action Grids：把 7D 连续 action 重写成 3 个 spatial tokens

这是 paper 最有创意的部分，我觉得它解决了 OpenVLA / RT-2 一直被诟病的 action tokenization 问题。

### 3.1 传统做法的痛点

RT-1 / RT-2 / OpenVLA 都是把每个 action 维度（7 维：x, y, z, roll, pitch, yaw, grip）**独立地**用 256 个 uniform bins 离散化，于是每步预测 7 个 tokens。

问题：
1. **Uniform bins 浪费分辨率**：robot 动作高度集中在小范围，paper 的 Fig. 13 显示几乎所有数据点都聚在 2-sigma 椭球内。Uniform 256 bins 把大量分辨率留给"机器人永远不会去的动作"，导致精细动作区分辨率不够
2. **7 tokens 太多**：autoregressive decode 慢，每多一个 token 翻一倍 latency
3. **维度独立建模丢失空间结构**：x、y、z 各自分桶，丢了"一个 3D translation 是一个整体"的几何意义

### 3.2 SpatialVLA 的三招

**第一招：拆解 action 语义**

$$\mathbf{a} = \{\mathbf{a}_{trans}, \mathbf{a}_{rot}, \mathbf{a}_{grip}\}$$

- $\mathbf{a}_{trans} = \{x, y, z\}$：平移
- $\mathbf{a}_{rot} = \{roll, pitch, yaw\}$：旋转
- $\mathbf{a}_{grip} = \{grip\}$：夹爪开合（2 个离散 token）

**第二招：translation 转 polar coordinates**

把 $(x, y, z)$ 转成 $(\phi, \theta, r)$，**方向** $(\phi, \theta)$ 与**距离** $r$ 解耦。这是个关键的几何先验：在 manipulation 里，方向比距离更重要且更难学 — 方向需要球面均匀分辨率，而距离大部分是小步长。

**第三招：按 Gaussian 概率等分 grid**

公式 (4)：

$$a_2, \ldots, a_M = \arg\min_{a_2, \ldots, a_M} \sum_{i=1}^{M} \left| \int_{a_i}^{a_{i+1}} f(x) dx - \frac{1}{M} \right|$$

- $f(x)$：该 action 维度的 Gaussian PDF $\mathcal{N}(\mu^a, \Sigma^a)$
- $M$：grid 数量
- 目标：找 $M-1$ 个分点，使每个区间 $[a_i, a_{i+1})$ 的积分概率都是 $1/M$
- 这等价于 Gaussian 的 inverse CDF 在均匀分位点上的取值 — 即 PPF (Percent Point Function)，Algorithm 1 里就是用 `scipy.stats.norm.ppf` 实现

### 3.3 Grid 分辨率设计

Tab. VII 和 Appendix B 明确给出：
- $\theta \in [0, \pi]$，16 bins
- $\phi \in [-\pi, \pi]$，32 bins（方向更精细）
- $r \in [0, 3]$，8 bins（距离粗一点就行）
- $M_{trans} = 16 \times 32 \times 8 = 4096$

- $roll, pitch, yaw \in [-1, 1]$，各 16 bins
- $M_{rot} = 16 \times 16 \times 16 = 4096$
- $M_{grip} = 2$

**总 vocab size $V = 4096 + 4096 + 2 = 8194$**

但关键在于：**一次 action prediction 只生成 3 个 tokens**（一个 trans token + 一个 rot token + 一个 grip token），而不是 OpenVLA 的 7 个。这是为什么 paper 标榜 "fewer tokens per action"，inference 在 RTX 4090 上能到 ~20Hz。

### 3.4 与 NeRF 编码 / VQ-VAE 的联想

我读到这里时立刻联想到两件事：
1. **NeRF 的 positional encoding + Gaussian spatial density**：把连续空间编码成离散 vocab，本质是 spatial 信号到 token 信号的对齐
2. **VQ-VAE 的 codebook**：8194 个 action token embeddings $\mathbf{E}_a \in \mathbb{R}^{d \times V}$ 就是一个 codebook，每个 code 对应 3D 空间中的一个 cell。和 VQ-VAE 不同的是，这里 codebook 的几何位置是已知且固定的（由 grid 划分决定），所以可以 trilinear interpolation

### 3.5 公式 (5) 的 embedding 结构

$$\mathbf{E}_a = \{\mathbf{E}_{trans}, \mathbf{E}_{rot}, \mathbf{E}_{grip}\}$$

- $\mathbf{E}_{trans} \in \mathbb{R}^{d \times M_{trans}} = \mathbb{R}^{2304 \times 4096}$
- $\mathbf{E}_{rot} \in \mathbb{R}^{d \times M_{rot}} = \mathbb{R}^{2304 \times 4096}$
- $\mathbf{E}_{grip} \in \mathbb{R}^{d \times 2}$
- 这些 embedding 与 LLM 的 text embedding 共享参数空间（linearization 后拼到 128k vocab 上）

**Intuition**：每个 action token 的 embedding 在训练中学到的，是"这一格 3D 空间内的动作应该具备的语义 + 空间特征"。因为它和 text embedding 在同一空间，所以 LM 可以自然地用语言-空间对齐的方式来 generate action。

---

## 4. Post-training：Spatial Embedding Adaption

### 4.1 痛点与 motivation

OpenVLA fine-tune 一个新 robot 的做法是 LoRA + 重新学 action head。问题是新 robot 的动作分布可能完全不一样（比如 Franka 比 WidowX 工作空间大、动作幅度大），原来的 4096 grid 可能不匹配。

SpatialVLA 的方案：

1. 在新数据集上重新 fit Gaussian $\mathcal{N}(\mu_{new}, \Sigma_{new})$
2. 重新划 grid $\mathbf{G}_{new}$
3. 新 token embedding $\mathbf{E}_{\mathfrak{a}_{new}}$ 用 trilinear interpolation 从旧 grid 初始化

### 4.2 公式 (6) 详解

$$\mathbf{e}_{\mathfrak{a}_{new}}^{i} = \sum_{j=1}^{K} w_j \mathbf{e}^j$$

- $i$：新 grid 中第 $i$ 个 cell
- $(\phi_{new}^i, \theta_{new}^i, r_{new}^i)$：新 grid 第 $i$ 个 cell 的 centroid 3D 坐标
- $\mathbf{G}^{adj} = \{\mathbf{G}^1, \ldots, \mathbf{G}^K\}$：在旧 grid 中找到的相邻 $K$ 个 cell（$K=8$ 对 3D trilinear）
- $\mathbf{e}^j \in \mathbb{R}^d$：相邻 cell 在 pre-trained model 中的 action token embedding
- $w_j$：归一化的距离权重（trilinear interpolation 的标准权重）

**这本质上就是把 3D 空间里的 embedding 当成一个 trilinear 插值的 3D 体积场**。新 robot 的 grid 是这个场在不同位置的采样点，embedding 通过插值"传递"过来。

### 4.3 这为什么有效（Fig. 8 解析）

Fig. 8 显示了 cross-sectional features 可视化：
- 左图：pre-trained grid 的 embedding 在某个切面上呈现清晰的 spatial 结构
- 中图：直接 fine-tune 后的 embedding 出现"重构断层"
- 右图：用 Gaussian adaption 初始化后，结构平滑过渡，模型从更合理的起点开始学习

Tab. V 的数字佐证：
- LIBERO-Spatial: full params 77.7% → LoRA 83.6% → +Gaussian adaption **88.2%**（+4.6%）
- LIBERO-Object: 73.3% → 84.8% → **89.9%**（+5.1%）
- LIBERO-Goal: 78.5% → 76.4% → 78.6%
- LIBERO-Long: 50.1% → **55.5%**（+5.4%）

**直觉**：这是个很漂亮的 transfer learning 设计 — pre-trained 的 spatial embedding 已经编码了 "3D 空间中某个位置附近该做什么动作" 的知识，新 robot 只是这个空间的另一个采样者。比起 LoRA 把所有知识塞进低秩矩阵，这种 explicit 的几何对齐效率高得多。

---

## 5. Pre-training 与 Deployment 细节

### 5.1 数据混合（Tab. VII + Fig. 9）

1.1M real robot episodes，主要组成：
- Bridge 15.34%
- Fractal 14.71%
- DROID 11.66%
- BC-Z 8.64%
- Kuka 7.06%
- RH20T 5.67%（在 OpenVLA 基础上额外加的）
- Stanford Hydra 5.15%
- Language Table 5.06%
- 其他 20 多个数据集补齐

Paper 提到两个 data mixture 上的小工程技巧：
1. **FMB 数据集降权**：发现比例过高导致 robot "右偏"
2. **DROID 后期剔除**：第二个 stage 把 DROID 移除，再训 40k steps，accuracy 从 90% 提升到 95%（Fig. 12 显示 loss 在 stage 2 进一步下降）— 这呼应了 OpenVLA 的发现

### 5.2 训练超参

- 64 张 A100，10 天
- Batch size 2048（这非常大，paper 也用 ZeRO stage 1 配合 DeepSpeed）
- AdamW, lr = 2e-5, linear schedule, warmup ratio 0.005
- PaliGemma 2 作为 backbone
- **关键**：text token embedding 冻结，只 train spatial embedding + vision encoder + LLM backbone
  - 原因：保留 VLM 的世界知识，避免 catastrophic forgetting
  - Ablation Tab. IV setting #9 验证：冻 LLM embedding 后 Pick Coke Can visual matching 从 50.7% → 70.7%

### 5.3 Inference 配置

- 输入：224×224 RGB（单帧，单第三人称相机）
- 输出：T=4 chunk，12 tokens（4 步 × 3 tokens/步）
- ensemble actions 后执行，再 predict 下一个 chunk
- 8.5GB 显存，~20Hz on RTX 4090
- 数据增强：random crop + color jitter（对 SimplerEnv / WidowX zero-shot 至关重要）

---

## 6. 实验结果详解

### 6.1 SimplerEnv - Google Robot (Tab. I)

| Model | Visual Matching Avg | Variant Aggregation Avg |
|-------|---------------------|-------------------------|
| RT-2-X (55B) | 60.7% | 64.3% |
| OpenVLA | 27.7% | 39.8% |
| RoboVLM (fine-tune) | 63.4% | 51.3% |
| π0 (BF16) | 70.1% | — |
| **SpatialVLA (zero-shot)** | **71.9%** | **68.8%** |
| **SpatialVLA (fine-tune)** | **75.1%** | **70.7%** |

**亮点**：3.5B 参数打败 55B 的 RT-2-X，并且在 Variant Aggregation 上几乎 +5% — Variant Aggregation 测试的就是 lighting/texture/camera pose 变化下的鲁棒性，这正是 Ego3D encoding 应该最受益的场景。

### 6.2 SimplerEnv - WidowX (Tab. II)

| Model | Overall Average |
|-------|-----------------|
| OpenVLA | 1.0% |
| RoboVLM (fine-tune) | 31.3% |
| Octo-Small | 30.0% |
| **SpatialVLA (zero-shot)** | **34.4%** |
| **SpatialVLA (fine-tune)** | **42.7%** |

**亮点**：Fine-tune 后 "Put Eggplant in Yellow Basket" 拿到 **100%** — 这个数据点让我印象深刻，说明 model 在合适 adaption 后是可以达到接近完美的 in-domain performance 的。

### 6.3 LIBERO (Tab. III)

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| Diffusion Policy from scratch | 78.3% | 92.5% | 68.3% | 50.5% | 72.4% |
| Octo fine-tuned | 78.9% | 85.7% | 84.6% | 51.1% | 75.1% |
| OpenVLA fine-tuned | 84.7% | 88.4% | 79.2% | 53.7% | 76.5% |
| TraceVLA fine-tuned | 84.6% | 85.2% | 75.1% | 54.1% | 74.8% |
| **SpatialVLA fine-tuned** | **88.2%** | **89.9%** | 78.6% | **55.5%** | **78.1%** |

**亮点**：LIBERO-Spatial 上 88.2% — 比 OpenVLA 高 3.5%，是所有方法最高。这个 suite 专门测试 object layout 变化的泛化，正好印证了 Ego3D encoding 学到了 spatial reasoning 而不是死记位置。LIBERO-Long 较弱是 model 没有显式历史信息机制的局限。

### 6.4 Franka 真机（Fig. 6）

- Single Task: SpatialVLA 82% vs Diffusion Policy 81% vs OpenVLA 72% vs Octo 54%
- Instruction Following: **SpatialVLA 比 OpenVLA +12%**，Diffusion Policy 仅 26%
- Multi-task: SpatialVLA 57%，OpenVLA 42%，Octo 41%

**Intuition**：当多任务、需要 instruction grounding 时，VLA 的预训练知识 + spatial encoding 优势放大。Diffusion Policy 在 multi-task 上几乎崩了，因为没有语言先验。

### 6.5 Spatial Understanding（Tab. XV）

最让我兴奋的一组数字 — 这些任务专门设计有 spatial layout 变化：

| Task | OpenVLA | SpatialVLA |
|------|---------|------------|
| Place Plush Toy Closest to Robot on Car | 45.45% | **72.72%** |
| Put Green Cup on Pink Cloth (Stove, low) | 27.27% | **72.72%** |
| Put Green Cup on Pink Cloth (Sink, high) | 45.45% | **81.81%** |
| Put Carrot in Plate (elevated on pan) | 54.54% | **63.63%** |

OpenVLA 在高低位置变化、3D 关系变化下都掉得很惨。SpatialVLA 在 "Place Plush Toy Closest to Robot" 这种需要 3D 近远判断的任务上比 OpenVLA 高 27% — 这是 Ego3D encoding 价值的最直接证明。

---

## 7. Ablation 深度解读（Tab. IV）

### 7.1 Action 离散化方式对比

| # | Setting | Pick Coke (VA / VM) |
|---|---------|---------------------|
| 1 | SpatialVLA (adaptive 8194) | 81.6% / 70.7% |
| 2 | Linear 256 bins | 40.7% / 19.0% |
| 3 | Uniform distribution split | 77.9% / 28.0% |
| 8 | w/o ego3d | 68.9% / 70.3% |
| 9 | w/o freeze LLM embed | 70.2% / 50.7% |

**Setting 1 vs 2**: Adaptive 8194 vs Linear 256 — 在 variant aggregation 上 +40.9%，在 visual matching 上 +51.7%。这数字是整篇 paper 最震撼的对比之一。**Linear 256 bins 这个传统设计实际上是 OpenVLA / RT-2 性能的瓶颈**。

**Setting 1 vs 3**: 用 uniform 分布而非真实 Gaussian 分布划分 — visual matching 从 70.7% 掉到 28.0%。说明问题不在于"分多少 bin"，而在于"分在哪儿" — 必须按真实数据分布来分。

### 7.2 Grid 分辨率 ablation

| Resolution | Pick Coke (VM) | Move Near (VM) | Put Eggplant (Success) |
|------------|---------------|----------------|------------------------|
| 1026 | 67.3% | 54.2% | 54.2% |
| 4610 | 68.0% | 79.2% | 75.0% |
| 6166 | 74.0% | 79.2% | 87.5% |
| **8194** | **70.7%** | **85.4%** | **87.5%** |

- 1026 → 8194：Move Near 提升 +31.2%，Put Eggplant +33.3%
- 但 6166 → 8194 在 Pick Coke 上反而略降（74.0 → 70.7）
- **Intuition**：分辨率在 6166-8194 之间是 saturation 区间。paper 在 Appendix C Q1 也讨论了 — 再加分辨率会带来 128k vocab 的参数膨胀，trade-off 不划算。这给未来 humanoid robot / 双臂 / 灵巧手留了空间（paper 在 Limitation 提到）

### 7.3 Ego3D 的影响（Tab. IV setting 8）

- w/o ego3d: Variant Aggregation Pick Coke 81.6 → 68.9，Move Near 79.2 → 66.7
- **降 ~13%** — variant aggregation 测的就是环境变化，正好是 ego3D 应该帮的地方
- Appendix Tab. IX 进一步验证：w/o depth 45.4%，sensor depth 70.5%，Zoe depth 72.7%

### 7.4 Freeze LLM Embedding（Tab. IV setting 9）

- 不冻：Pick Coke Visual Matching 50.7%，冻了 70.7%
- **直觉**：VLM 的 text embedding 已经在几十亿图文对里学到 language grounding，如果让它跟随 robot 数据漂移，会丢失 language 理解力。这跟 LLaMA fine-tuning 时经常冻结 embedding 是一个道理。

---

## 8. 与其他 VLA / 3D Foundation Models 的比较

| Model | Observation | Action Tokenization | Cross-embodiment | Spatial Understanding |
|-------|-------------|---------------------|------------------|------------------------|
| RT-1 [6] | 2D image | 256 bins × 7 tokens | ✗ | ✗ |
| RT-2 [7] | 2D image + VLM | 256 bins × 7 tokens | partial | ✗ |
| OpenVLA [30] | 2D image + VLM | 256 bins × 7 tokens | ✓ (OXE) | ✗ |
| Octo [48] | 2D image | Diffusion / token | ✓ (OXE) | ✗ |
| π0 [5] | 2D image + VLM | Flow matching continuous | ✓ | ✗ |
| 3D-VLA [69] | 3D point cloud | token | ✗ | ✓ (但只 generate world state，不直接 action) |
| LEO [25] | 2D + 3D | token | ✗ | ✓ (但 action 是 navigation 类) |
| **SpatialVLA** | **2D + Ego3D** | **Adaptive grids × 3 tokens** | **✓ (OXE+RH20T)** | **✓** |

**SpatialVLA 的独特性**：第一个把 3D spatial encoding 和 spatial action grids 同时引入 generalist VLA 的工作，并且在 OXE+RH20T 这种真异构数据上 pretrain。3D-VLA 和 LEO 都做了 3D，但它们更偏 world modeling / navigation，没碰 cross-embodiment manipulation 的问题。

Reference:
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0: https://arxiv.org/abs/2410.24164
- Octo: https://octo-models.github.io
- 3D-VLA: https://arxiv.org/abs/2403.09631
- TraceVLA: https://arxiv.org/abs/2412.10345
- HPT: https://arxiv.org/abs/2409.20537

---

## 9. 我对这篇 paper 的几点直觉判断

### 9.1 它做对了什么

1. **3 tokens 而非 7 tokens** — 这是个被低估的工程突破。autoregressive decode 是 $O(L^2)$ attention + $O(L)$ token generation，token 数减半意味着 inference 加速近 4×。这把 VLA 推到 20Hz 的可部署区间。

2. **Gaussian-driven grid** 比 uniform bins 强这么多让我惊讶 — +40% 的 gap 说明过去 VLA 工作 在 action 表征上欠了一大块。

3. **Ego3D 用预测深度反而比 sensor 深度好** — 这是个反直觉但重要发现，说明 prior-based 几何信息比 noise 测量值更适合 token-based 学习。

4. **Spatial Embedding Adaption 用 trilinear interpolation** — 把 codebook 看成 3D 体积场来插值，简单但有效。这给后续 VLA fine-tune 提供了一个新范式。

### 9.2 它的潜在局限（我自己读出来的）

1. **单帧观察**：paper 只用当前帧 + 历史预测 tokens，没有显式 memory 机制。所以 LIBERO-Long 表现一般 — 长时序任务会丢上下文。这点 π0 / RDT-1B 用 diffusion chunks 做得更好。

2. **Gaussian 假设**：paper Limitations 里坦诚承认了 — 在单轴运动场景下 Gaussian 会塌缩，导致其他轴丢失分辨率。Kernel Density Estimation 或者 VAE 隐空间映射可能是下一步。

3. **ZoeDepth 是 frozen**：8.6% 参数 + 0.06s/step 开销，但它是 generic depth prior，没在 manipulation 数据上 fine-tune。一个 robot-specific depth model 可能更好，但成本更高。

4. **3 tokens 对 7+ DOF robot 不够**：bimanual 或 humanoid 有 14+ DOF，3 tokens 表达力可能不足。Paper 提到未来要 "share action grids among embodiments like dexterous hands, bimanual arms, and legs"，这是个开放问题。

### 9.3 它让我想到的相关工作联想

- **NeRF 的 spatial encoding**：Ego3D encoding 在某种意义上是 "NeRF-style position encoding 用在 robot policy 上"，把每个 patch 的 3D 坐标 $\gamma(P)$ 注入到 feature 中
- **3D Gaussian Splatting**：Action grid 的 trilinear interpolation init 跟 3DGS 用 sparse点初始化 density field 有点神似
- **FAST (Pertsch 2025) [51]**：同期的 action tokenization 工作，用 VQ 来学习 codebook，没显式利用 spatial 结构。SpatialVLA 用结构化 grid + Gaussian partition 是显式 prior 的方式
- **CogACT [31]**：用 diffusion action head，连续输出 — 是 adaptive grids 的另一个极端（完全连续）。混合两者可能是有意思的方向
- **HPT [65]**：用 heterogeneous pre-trained transformers 学共享 representation，思路类似但更偏向 "stem module 对齐"，SpatialVLA 是 "spatial geometry 对齐"，两者可以结合

### 9.4 我会怎么 follow up

如果我来做下一步，会尝试：
1. 把 action grid 换成 **implicit latent space**（VAE 编码到连续空间）+ Gaussian prior — 解决 Gaussian 塌缩问题
2. 把 Ego3D 换成 **multi-view epipolar encoding** — 加入时序 stereo 信息
3. 用 **diffusion head** 替代 autoregressive decode — 解决长 horizon 问题（π0 / RDT 已证明）
4. 在 grid embedding 上加 **rotational equivariance**（SO(3) equivariant MLP）— 现在 roll/pitch/yaw 是独立分桶，丢了 SO(3) 结构

---

## 10. 一些可能用得上的链接

- SpatialVLA Project: https://spatialvla.github.io
- PaliGemma 2 (backbone): https://arxiv.org/abs/2412.03555
- Open X-Embodiment: https://robotics-transformer-x.github.io
- RH20T dataset: https://rh20t.github.io
- OpenVLA: https://github.com/openvla/openvla
- π0 (Physical Intelligence): https://arxiv.org/abs/2410.24164
- Octo: https://octo-models.github.io
- SimplerEnv benchmark: https://simpler-env.github.io
- LIBERO benchmark: https://lifelong-robot-learning.github.io/libero
- ZoeDepth: https://arxiv.org/abs/2302.12288
- 3D-LLM: https://3dllm.github.io
- 3D-VLA: https://arxiv.org/abs/2403.09631
- TraceVLA: https://arxiv.org/abs/2412.10345
- FAST (Pertsch et al.): https://arxiv.org/abs/2501.09747
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu

---

## TL;DR for Intuition

SpatialVLA 的核心论断是：**VLA 模型的 spatial intelligence 缺失，原因在两端 — observation 是 2D 的，action 是 1D uniform bins 的，都丢了 3D 几何结构**。它的两招：

1. **Ego3D Position Encoding**：用预测 depth 把每个 image patch 映射到相机坐标系下的 3D 点，再 sinusoidal+MLP 编码加到 2D feature 上 — 让 SigLIP feature 平面"长出" 3D 维度。这一招让 model 在 lighting/texture/camera pose 变化下泛化提升 +13%。

2. **Adaptive Action Grids**：把 7D action 拆成 trans/rot/grip 三组，trans 转 polar 坐标，按数据 Gaussian 分布等概率划分 grid（而非 uniform bins），并用 3D grid 的 token embedding 共享 text vocab。一次预测只需 3 tokens，推理 4× 加速。这一招让 Pick Coke 从 19% (linear 256) 飙到 70.7% (adaptive 8194)。

再加一个 **Spatial Embedding Adaption**：fine-tune 时重新拟合 Gaussian，trilinear interpolate 旧 embedding 初始化新 grid — 把 codebook 当成可插值的 3D 体积场，跨 embodiment transfer 既快又准。

读完我的总体感觉：这篇 paper 在 VLA 的 spatial 维度上明显推进了一步，**Ego3D encoding 和 Adaptive grids 都是被低估的设计**，未来 VLA 工作大概率会把这两个 trick 作为 default 组件。它的局限在于单帧 + Gaussian 假设 + 3 tokens 表达力，这些都是 follow-up 工作的明确入口。
