---
source_pdf: JoyAI-RA 0.1 A Foundation Model for.pdf
paper_sha256: 9e81be0069d1b83b676ccba9d66eb3d28b74e3c518ae9c8fe6b576c148881c00
processed_at: '2026-08-05T10:57:30-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej, 咱们抛开学术黑话, 用最直白的人话来拆解 JoyAI-RA 这篇 paper。 这套系统的本质, 就是在教一个 robot 大脑怎么从"看人类干活"过渡到"自己用机械臂干活", 并且干得还很灵活。

Project page: https://joyai-ra.github.io/

---

## 1. 核心痛点: 为什么 robot 总是学不好?

Open-world robotic autonomy 卡在两个地方, 且互相纠缠:
1. **Data 多样性不够**: 遥操作 collect robot data 太贵了。 你让 robot 抓个杯子, 数据好搞; 你让它整理房间, 收集数据的成本就爆炸。
2. **Embodiment gap (身体形态差异)**: 人类有五根灵活的手指, robot 可能就是个两指夹爪。 你让 robot 直接模仿人类的手部动作, 维度都对不上。

JoyAI-RA 的解题思路非常粗暴且有效: 既然单靠 robot 数据不够, 那我就把 **Web 数据 + 人类第一视角视频 + 仿真数据 + 真实 robot 数据** 全搅在一起, 喂给同一个模型。 为了解决身体形态不一样的问题, 它搞了一个 **"统一动作空间"**, 相当于搞了个世界语, 让所有身体形态的动作都能在一个坐标系里表达。

---

## 2. 数据大乱炖: 四道菜的营养搭配

这四源数据各有各的用处, 缺一不可:

*   **Multi-Modal Web Data**: 喂给模型大量的图文 QA、空间推理数据 (比如 Cambrian-10M, RefSpatial)。 这相当于给模型上"通识课", 让它先搞懂"什么叫苹果", "什么叫左边", 建立视觉和语言的连接。
*   **EgoLive (自研人类第一视角视频)**: 这篇 paper 的精髓。 用 60 FPS 的相机挂在人头上, 拍人干各种活 (1969 种物体, 1796 种动作, 涵盖家庭、零售、物流)。 关键在于他们做了 **per-frame (逐帧) 的 subtask 标注**。 机器不仅看到人在动, 还知道这一帧在"倒水", 下一帧在"放杯子"。 拍完之后, 用 pipeline 估出人手的 3D 轨迹, 然后映射 (retarget) 到 ALOHA、Fourier、Agibot G1 等 robot 的关节上。
*   **Simulation Data**: 仿真环境里生成的轨迹数据。 仿真数据的好处是便宜、可控, 想要多少有多少, 给模型提供基础的"动作感"。
*   **Real-Robot Data**: 包含开源的 Open X-Embodiment、AgiBot-World, 以及他们自家的 JDAgibot。 真实 robot 数据最贵, 但无可替代, 因为只有真实硬件才能捕捉到接触时的摩擦力、传感器的噪声。

---

## 3. 统一动作空间: 怎么让不同的身体说同一种语言?

跨 embodiment 最头疼的是: ALOHA 有 14 个自由度, Agibot G1 有几十个, 人类手更复杂。 怎么放在一起训练?

### 3.1 Camera-Frame End-Effector Representation (相机坐标系)
Paper 做了一个极其关键的决定: 把所有的动作 (action) 和状态 (proprioceptive state) 都转换到 **相机坐标系** 下表示, 放弃传统的机器人基座坐标系 (base frame)。

直觉上: 假设你头上戴个 GoPro。 你往前走一步, 在相机坐标系里, 就是画面里的东西往后退。 不管你是双足机器人、轮式机器人还是单臂机器人, 只要相机看到的画面变化是一样的, 你的 action vector 就是一样的。 这样一来, 视觉输入 (image) 和动作输出 (action) 就完全在同一个视角下对齐了, 极大降低了模型的学习难度。

### 3.2 Unified Action Dimensionality (定长向量 + Masking)
模型定义了一个固定长度的超级大向量, 涵盖了你能想到的所有部位:
`[左臂关节, 右臂关节, 左灵巧手, 右灵巧手, 左夹爪, 右夹爪...]`

如果当前训练的数据是单臂 robot, 那么这个大向量里关于左臂、灵巧手的维度全部置为 0, 并且在算 loss 的时候用 mask 把它们遮掉 (loss 为 0, 不回传梯度)。 

这样, 从单臂夹爪到双臂灵巧手, 都能用同一套网络结构和同一套表征来训练, 实现了真正的 cross-embodiment。

---

## 4. 模型架构: 大脑和小脑的分工

JoyAI-RA 架构上分两半, 模拟人类的"大脑"和"小脑脊髓":

*   **VLM (大脑)**: 负责看图、读指令, 输出空间感知和语义表征 $z_t$。 它告诉你"耳机在桌上, 需要挂到支架上"。
*   **Perceiver Action Expert (小脑)**: 拿到 VLM 的表征 $z_t$ 后, 生成连续的底层控制指令 $a_{t:t+H}$ (action chunk)。 用的是 Perceiver 架构, 通过 cross-attention 高效融合多视角图像和动作历史。

这里最精彩的是 action expert 用了 **Flow Matching (流匹配)** 框架来生成连续动作。

### Flow Matching 公式拆解 (Build Intuition)

Flow matching 的直觉: 把生成动作的过程看作是从"一团纯噪声"慢慢流向"标准答案"的过程。 模型要学的就是预测这个流场里每一点的"水流方向" (velocity)。

Step 1: 构造噪声动作轨迹
$$ a_{1:H}^{\tau, \omega} = \tau \cdot a_{1:H} + (1 - \tau) \cdot \omega $$
变量解释:
*   $a_{1:H}$: Ground-truth 的真实动作序列 (长度为 $H$)。
*   $\omega \sim \mathcal{N}(0, I)$: 从标准高斯分布采样的纯噪声。
*   $\tau \in [0, 1]$: 时间步系数。 $\tau=0$ 时全是噪声, $\tau=1$ 时全是真实动作。
*   $a_{1:H}^{\tau, \omega}$: 噪声和真实动作混合后的结果。

Step 2: 模型预测 velocity
$$ v^{\text{out}} = f_\theta(z_t, a_{t:t+H}^0, \tau) $$
模型 $f_\theta$ 吃进 VLM 表征 $z_t$、当前的动作噪声状态以及时间步 $\tau$, 预测出下一步的流动方向。

Step 3: Loss 计算
$$ \mathcal{L}_{\text{flow}} = \mathbb{E}\left[\|\omega - a_{1:H} - f_\theta^a(a_{1:H}^{\tau, \omega})\|^2\right] $$
模型预测的方向, 应该无限接近于从当前噪声点指向真实动作点的向量 ($\omega - a_{1:H}$)。

### Perceiver 内部的小细节: AdaLN
在 Perceiver 的每一层, 有个叫 **AdaLN (Adaptive Layer Norm)** 机制:
$$ \tilde{z}_t = \text{AdaLN}_z(z_t, \tau) $$
直觉: $\tau$ 表示去噪进行到了哪一步。 早期 ($\tau \to 0$, 全是噪声) 时, VLM 的表征应该提供宏观的规划; 晚期 ($\tau \to 1$, 接近真实动作) 时, 应该提供微观的修正。 AdaLN 让 VLM 的特征根据去噪阶段 $\tau$ 动态调整 scale 和 shift, 实现了"宏观微观自适应"。

---

## 5. 三阶段训练法: 从通识到专才

模型训练分三步走, 逐步从泛化走向专精:

**Stage 1: VLM Co-Pretraining (通识教育)**
混合 Web 数据 + 人类视频 + 离散化动作 (FAST tokens), 搞大杂烩预训练。 Loss 是标准的自回归 Next-token Prediction:
$$ \mathcal{L} = -\sum_{j=1}^{n-1} M_j \log p_\theta(y_{j+1} | x_{1:j}) $$
(变量: $x$ 是输入, $y$ 是输出 token, $M_j$ 是 mask 掩码)

**Stage 2: VLA Co-Pretraining (动作专项训练)**
加入 flow matching loss, 引入仿真轨迹 + 真实 robot + 映射后的人类视频 (全在统一动作空间里)。 此时模型一边要继续做 VQA, 一边要预测连续动作。 这里的 loss 是两个加权和:
$$ \mathcal{L} = \alpha \cdot \mathcal{L}_{\text{auto-reg}} + \mathcal{L}_{\text{flow}} $$
($\alpha$ 是控制比例的超参数)

**Stage 3: Post-Training (目标 robot 专精)**
扔掉所有的 VQA loss 和其他数据源, 只用目标 robot (比如 Agibot G1) 自己的数据, 只优化 flow matching loss, 让模型彻底适应目标 hardware。

---

## 6. 实验里的黄金 Insight (Ablation 大揭秘)

这篇 paper 最有价值的是它的 ablation study, 揭示了 VLA 模型 scaling 的一些反直觉现象:

### Insight 1: 用少量窄域 robot 数据预训练, 比不预训练还差!
在 RoboTwin 2.0 benchmark 上:
*   No Pretraining (从零开始): 81.64%
*   JDAgibot Only (只用自家的真实 robot 数据预训练): 77.62% **(跌了!)**

直觉解释: 如果你只给模型看一种特定 robot 干特定活的数据, 它会过拟合到这个 narrow distribution 里, 丢失了对物理世界泛化的能力。 这说明 **Data Diversity (数据多样性) 远比 Data Relevance (数据相关性) 重要**。

### Insight 2: 人类视频数据有 "Scaling Threshold" (规模门槛)
*   JDAgibot + 10% EgoLive (只用一点点人类视频): 81.40% (没啥用)
*   JDAgibot + Full EgoLive (用全量人类视频): 87.42% (暴涨)

人类视频带来的先验知识, 喂一点点是不够的, 必须达到一定的规模, 模型才能从中学到通用的物理交互规律。 这跟 LLM pretraining 里的 "涌现能力" 非常像。

### Insight 3: VLM 和 VLA 两阶段预训练是互补的
*   只做 VLM Co-pretraining: 87.84%
*   只做 VLA Co-pretraining: 87.42%
*   两个都做: 90.48%

虽然单看都在 87% 左右, 但两个一起做能直接干到 90%。 这说明 Stage 1 学的"看懂世界"和 Stage 2 学的"跨身体运动"在表征层面是互补的, 不会互相覆盖。

### Insight 4: In-domain 数据搞不好会帮倒忙
在真实 Agibot benchmark 上, 如果加入的 in-domain (同领域) 人类视频与评测任务的环境、语义高度匹配 (比如 Headphones 任务), 成绩大幅提升。 但如果 in-domain 视频的环境跟评测任务 mismatch (比如 Food Scraps 任务), 成绩反而下降。 引入了冲突的监督信号。

---

## 7. 看看战果: SOTA 表现

在 RoboTwin 2.0 仿真测试里:
JoyAI-RA Easy 模式 90.48%, Hard 模式 89.28%。 最恐怖的是 Easy 到 Hard 只掉了 1.2%, 而 π0.5 掉了 6%。 说明面对未见过的场景随机化, JoyAI-RA 的抗干扰能力极强。

在 RoboCasa GR1 (24 个长序列任务) 上:
平均成功率 63.2%, 碾压了 GR00T N1.6 (47.6%) 和 ABot-M0 (58.3%)。 特别是在 "CanToDrawerClose" 这种长序列操作上, 领先前 SOTA 16 个点。 这全靠 EgoLive 里的 per-frame subtask 标注带来的时序推理能力。

在真实 Agibot G1 上:
6 个任务平均成功率从 π0.5 的 0.62 提升到 0.74。 需要精准语义识别和放置的任务 (如挂耳机、包装药品) 提升最大。 但在多步长序列任务 (如烤牛角包) 上, π0.5 依然有优势, 说明 JoyAI-RA 的 action expert 在超长时序规划上还有提升空间。

---

## 8. 总结 (Takeaway)

JoyAI-RA 的成功说明了 VLA 模型发展的一个核心趋势: **工程配方比算法花样更重要**。 它并没有发明什么惊世骇俗的新架构, Perceiver 和 Flow Matching 都是现成的。 它真正做对的是:
1. 把 camera frame 作为统一语义空间, 脱离了 robot base 的束缚。
2. 大量引入 60 FPS 带细粒度标注的人类第一视角视频, 并验证了其 scaling threshold 效应。
3. 证明了 multi-source heterogeneous pretraining (多源异构预训练) 中, diversity 压倒一切。

未来 VLA 的竞争, 可能会从"谁有更大的 transformer"变成"谁能以更低的成本 retarget 更多模态的物理数据 (人类视频、动物视频、仿真数据) 到统一 action space 里"。

参考链接:
*   JoyAI-RA Project: https://joyai-ra.github.io/
*   EgoDex (对比数据集): https://arxiv.org/abs/2505.11709
*   π0 (Flow Matching for VLA): https://arxiv.org/abs/2410.24164
*   Perceiver 架构: https://arxiv.org/abs/2103.03206
*   FAST (离散动作 token): https://arxiv.org/abs/2501.09747
*   RoboTwin 2.0 (Benchmark): https://arxiv.org/abs/2506.18088

---

# JoyAI-RA 0.1: VLA Foundation Model 深度解析

Andrej, 这篇 paper 是 Joy Future Academy (京东旗下) 的工作, 定位是 **generalizable robotic manipulation** 的 VLA foundation model。 我会从直觉出发, 把每层拆开讲。

Project page: https://joyai-ra.github.io/

---

## 1. 核心动机: 为什么需要 multi-source heterogeneous pretraining

Robotic autonomy 的瓶颈有两层, 且互相耦合:

1. **Data diversity 不足**: 现有 robot datasets (RT-1, Open X-Embodiment) 在 task coverage 和 long-tail interaction 上覆盖不够。 Teleoperation 太贵。
2. **Embodiment gap**: 跨 embodiment 的行为知识 transfer 困难。 Human hand 跟 robot gripper/dexterous hand 差异巨大。

JoyAI-RA 的核心 hypothesis: 单一 source 解决不了这两个问题。 需要 **four complementary data sources** + **explicit action-space unification** + **multi-level co-pretraining**, 才能 bridge embodiment gap。

直觉上: 这跟 LLM 的 pretraining recipe 思路一致 — code/math/multilingual 互补才能 build 通用 prior。 这里换成了 web VQA / human video / sim trajectory / real-robot 四路。

---

## 2. 数据构建: 四源互补

### 2.1 Multi-Modal Web Data
提供 **semantic + linguistic prior**:
- Cambrian-10M [34] (vision-centric multi-modal)
- RefSpatial [45] (spatial referring with reasoning)
- Galaxea [21]
- Cosmos-Reason1-SFT [1] (physical common sense)

这层没有 executable trajectory, 主要 build visual grounding 和 language-conditioned perception。

参考: https://arxiv.org/abs/2411.15438 (Cambrian-10M), https://arxiv.org/abs/2506.04308 (RoboRefer)

### 2.2 EgoLive (in-house egocentric human data)
这是这篇 paper 的亮点之一:

| 维度 | 数值 |
|---|---|
| Frame rate | 60 FPS |
| Object categories | 1,969 |
| Action categories | 1,796 |
| Total tasks | 10,000+ (household 3,779 + retail 3,686 + logistics 2,518) |
| Annotation | per-frame subtask + episode-level description |
| Target embodiments | ALOHA [42], Fourier [14], Agibot G1 |

关键 design choice: **operation-centric textual annotation** — 每帧都有 subtask 标注, 而非仅 episode-level。 这给 temporally grounded action decomposition 提供监督。

Hand-pose 估计后 retarget 到不同 robot morphology。 Retargeting 链路:
```
RGB video (60 FPS) 
  → hand-pose estimation (in-house pipeline) 
  → 3D hand trajectory 
  → retarget to robot joint space (ALOHA / Fourier / Agibot G1)
  → unified action token in camera frame
```

### 2.3 Simulation Data
- InternData-A1 [33]: high-fidelity synthetic, multi-embodiment
- GenieSim3.0-Dataset [40]
- InternData M1 [11]

作用: scalable, controllable action supervision。 给跨 embodiment 的 action diversity。

参考: https://arxiv.org/abs/2511.16651

### 2.4 Real-Robot Data
- Open X-Embodiment [28] (聚合多 embodiment)
- AgiBot-World [8]
- Galaxea Open-World Dataset [21]
- **JDAgibot** (in-house self-collected)

Real-robot 的不可替代价值: 捕捉 **contact uncertainty**, **sensing noise**, **hardware constraints**。 这些 sim 学不到。

---

## 3. Unified Action Space: 跨 embodiment 的关键

### 3.1 Camera-Frame End-Effector Representation

把 6-DoF end-effector pose 分解:
- 3D translation vector (相机坐标系下的位移)
- 3D axis-angle rotation vector (旋转)

为什么 camera frame? 两个 benefit:

1. **Consistent physical semantics**: 同一个 action vector 在不同 robot base placement 下表达相同 spatial displacement。 Robot-specific base frame 和 joint config 不再耦合。
2. **Visual-action alignment**: image input 和 predicted action 共享 viewpoint。 VLM 的视觉 grounded 表征和 action prediction 对齐。

这个设计跟 OpenVLA / π0 的关键区别: π0 用 base-frame end-effector delta, JoyAI-RA 用 **camera-frame absolute** (或 delta, paper 没明说, 但从 formula 看更像 absolute pose 的 chunk)。

### 3.2 Unified Action Dimensionality

定义一个 **fixed-length** action vector, 覆盖所有 actuator groups:

```
[left_arm_joints, right_arm_joints, 
 left_dexterous_hand, right_dexterous_hand, 
 left_gripper, right_gripper, 
 base_mobile, ...]
```

每个 embodiment 只 fill 自己的 dimension, 缺失的 dimension 在 loss 里 **mask out**:

$$
L_{\text{masked}} = \sum_i m_i \cdot \| \hat{a}_i - a_i \|^2
$$

其中 $m_i \in \{0, 1\}$ 是 mask, $i$ 索引 action dimension。

这样从 single-arm gripper 到 bimanual dexterous-hand 系统可以在同一 fixed-dim representation 下共同训练。 这是 cross-embodiment 的核心 trick。

直觉: 类似于 multi-task learning 里的 task-specific head + shared backbone, 但这里 head 是统一的, mask 控制哪个 task 用哪些 dim。

---

## 4. Model Architecture: VLM + Perceiver-based Action Expert

### 4.1 整体设计

两个 module:
- **VLM backbone**: vision-language understanding, 产出 spatially grounded multimodal representation $z_t$
- **Perception-Action Expert**: 基于 Perceiver [20] architecture, 通过 latent bottleneck 做 multi-modal fusion, 输出 continuous action chunk

直觉: VLM 像 "大脑", 处理 high-level semantics; Perceiver action expert 像 "小脑+脊髓", 处理 low-level continuous control。 二者解耦。

Perceiver 的优势: cross-attention 到 latent array, 复杂度 $O(N \cdot M)$ 而非 $O(N^2)$, 适合长 action chunk + 多视角 image tokens。

### 4.2 Joint Modeling

JoyAI-RA 联合建模两类输出:
- **Tokenized text** $\ell$ (e.g. subtask description): autoregressive
- **Continuous action chunk** $a_{t:t+H}$: flow matching

执行顺序: 先 autoregressive 生成 high-level semantic description, 然后 condition on 这个 intermediate representation 生成 action chunk。

这跟 Motus [4] 的 Mixture-of-Transformer 思路相似, 但 JoyAI-RA 用 Perceiver 做连续 action 生成。

### 4.3 Flow Matching Formulation (核心公式)

Action expert 是 **conditional velocity field predictor** under flow-matching framework。

**Step 1: Latent sequence 构造**

$$
a_{t:t+H}^0 = \text{Concat}\big(\phi_s(s_t), f_{\text{future}}, \phi_a(\tilde{a}_{t:t+H}, \tau)\big)
$$

变量解释:
- $s_t$: 当前 **proprioceptive state** (robot joint position / pose)
- $\phi_s(\cdot)$: state encoder, 把 proprio 投影到 latent
- $f_{\text{future}} \in \mathbb{R}^{K \times d}$: **learnable future tokens**, 即一组 trainable query embeddings, 作为未来 action step 的 placeholder。 类似 Perceiver 的 latent array 或 DETR 的 object queries
- $\tilde{a}_{t:t+H}$: **noisy action trajectory**, 通过 interpolation 得到 (见下方)
- $\tau \in [0, 1]$: flow-matching timestep
- $\phi_a(\cdot, \tau)$: action encoder, 同时编码 noisy action 和 timestep
- 上标 $0$ 表示 "initial latent before Perceiver layers"

**Step 2: Noisy action interpolation**

$$
a_{1:H}^{\tau, \omega} = \tau \cdot a_{1:H} + (1 - \tau) \cdot \omega
$$

变量:
- $a_{1:H}$: ground-truth action chunk
- $\omega \sim \mathcal{N}(0, I)$: Gaussian noise
- $\tau \in [0, 1]$: $\tau = 0$ 时全是 noise, $\tau = 1$ 时全是 GT action

这跟 π0 的 flow matching formulation 一致, 是 Rectified Flow / Stochastic Interpolant 的标准做法。

**Step 3: Velocity field prediction**

$$
v_{t:t+H}^{\text{out}} = f_\theta(z_t, a_{t:t+H}^0, \tau)
$$

变量:
- $z_t$: VLM 输出的 visual-language representation
- $a_{t:t+H}^0$: 上面构造的 latent sequence
- $f_\theta$: Perceiver-based expert

Training target: 让 $f_\theta$ 预测 **velocity** $\omega - a_{1:H}$ (从 noise 指向 GT 的向量场)。 MSE loss:

$$
L_{\text{flow}} = \mathbb{E}_{\mathcal{D}, \tau, \omega}\left[\|\omega - a_{1:H} - f_\theta^a(a_{1:H}^{\tau, \omega})\|^2\right]
$$

### 4.4 Perceiver Layer 细节

每层做两件事:

**(1) Timestep-adaptive normalization**:

$$
\tilde{z}_t = \text{AdaLN}_z(z_t, \tau)
$$

AdaLN (Adaptive Layer Norm) 让 visual-language stream 根据 denoising timestep $\tau$ 调制。 直觉: 早期 timestep ($\tau \to 0$, action 接近 noise), VLM 应该提供 coarse planning; 晚期 timestep ($\tau \to 1$, action 接近 GT), VLM 应该提供 fine-grained correction。 AdaLN 学到这种 dynamic modulation。

AdaLN 公式 (DiT 风格):
$$
\tilde{z} = \gamma(\tau) \cdot \text{LN}(z) + \beta(\tau)
$$
其中 $\gamma, \beta$ 是 timestep-conditioned MLP 输出的 scale 和 shift。

**(2) Residual cross-attention**:

$$
h_{t:t+H}' = h_{t:t+H} + \text{MHA}\big(Q = h_{t:t+H}, \ K, V = [h_{t:t+H}; \tilde{z}_t]\big)
$$

变量:
- $h_{t:t+H}$: 当前 action latent sequence (query)
- $\tilde{z}_t$: timestep-modulated VLM representation (key/value 的一部分)
- $[h_{t:t+H}; \tilde{z}_t]$: concatenation, 让 action latent 既能 self-attend 又能 attend 到 VLM context

直觉: action latent 作为 query, 主动从 (action history + VLM context) 里拉信息。 Perceiver 的 latent bottleneck 模式。

**(3) Residual feed-forward**:

$$
h_{t:t+H}^{\text{out}} = h_{t:t+H}' + \text{MLP}(h_{t:t+H}')
$$

标准 Transformer block。

最后从 refined latent 解码出 velocity prediction $v_{t:t+H}^{\text{out}}$。

---

## 5. Three-Stage Training Recipe

### Stage 1: VLM Co-Pretraining

$$
\mathcal{L}_{\text{VLM Co-Pretraining}}(\theta) = \mathbb{E}_{(x, y) \sim \mathcal{D}}\left[-\sum_{j=1}^{n-1} M_j \log p_\theta(y_{j+1} | x_{1:j})\right] \quad (1)
$$

变量:
- $x_{1:n}$: input tokens (image + text instruction)
- $y_{1:n}$: output tokens (VLM response + discrete FAST [29] action tokens)
- $M_j$: loss mask, 控制哪些 token 参与 loss
- $\mathcal{D}$: VLM pretraining dataset mixture

四类数据 mix:
1. General VQA (保持基础视觉理解)
2. Embodied VQA (含 point / bbox / trajectory 输出, 强化 spatial reasoning)
3. Cross-embodiment action data (discrete FAST token 形式, 为后续 continuous action 打基础)
4. Human video data (拓展 visual input 和 action distribution)

关键 insight: action 这阶段是 **discretized** (FAST tokens), 不是 continuous。 这跟 π0 直接上 continuous flow matching 不同。 JoyAI-RA 先 discrete 再 continuous, 两阶段细化。

FAST 参考: https://arxiv.org/abs/2501.09747

### Stage 2: VLA Co-Pretraining

引入 action expert, loss 加上 flow matching:

$$
\mathcal{L}_{\text{VLA Co-Pretraining}}(\theta) = \alpha \cdot \mathbb{E}\left[-\sum_{j=1}^{n-1} M_j \log p_\theta(y_{j+1} | x_{1:j})\right] + \mathbb{E}_{\mathcal{D}, \tau, \omega}\left[\|\omega - a_{1:H} - f_\theta^a(a_{1:H}^{\tau, \omega})\|^2\right] \quad (2)
$$

变量:
- $\alpha$: loss multiplier, 平衡 autoregressive 和 flow matching
- 第一项: 保留 VLM 能力 (避免 catastrophic forgetting)
- 第二项: flow matching loss 训练 action expert

这阶段数据: General VQA + Embodied VQA + sim trajectory + real-robot demo + retargeted human demo, 全部 unified action space。

### Stage 3: Post-Training on Target Robot

仅用 target robot data, 只优化 flow matching:

$$
\mathcal{L}_{\text{Post-Training}}(\theta) = \mathbb{E}_{\mathcal{D}_{\text{target}}, \tau, \omega}\left[\|\omega - a_{1:H} - f_\theta^a(a_{1:H}^{\tau, \omega})\|^2\right] \quad (3)
$$

- $\mathcal{D}_{\text{target}}$: target robot 或 target sim benchmark 数据
- 丢弃 autoregressive loss, end-to-end fine-tune 全部参数

直觉: 前两 stage build general prior, 这一 stage 做 domain adaptation 到具体 deployment robot。

---

## 6. 实验数据详解

### 6.1 RoboTwin 2.0 [9]

Setup: 2,500 clean demos + 25,000 randomized demos (500/task × 50 tasks)。 Randomization 含 background / clutter / table height / lighting。

| Method | Easy (%) | Hard (%) |
|---|---|---|
| π0 [5] | 65.92 | 58.40 |
| π0.5 [30] | 82.74 | 76.76 |
| Motus [4] | 88.66 | 87.02 |
| LingBot-VLA [38] | 88.56 | 86.68 |
| **JoyAI-RA** | **90.48** | **89.28** |

Easy-Hard gap: JoyAI-RA 仅 1.20% 跌幅, π0 是 7.52%, π0.5 是 5.98%。 说明 multi-source pretraining 学到的 prior 对 distribution shift robust。

部分任务 100%: Adjust Bottle, Grab Roller, Place Empty Cup。 

弱项: Hanging Mug (Easy 31%, Hard 28%)。 这是所有方法都弱的 task, 可能 task 本身定义困难。

### 6.2 RoboCasa GR1 Tabletop [27]

6-DoF dexterous hand benchmark, 24 tasks, 50 rollouts/task。

| Method | Avg Success (%) |
|---|---|
| GR00T-N1.6 [27] | 47.6 |
| Qwen3PI [10] | 43.9 |
| TwinBrainVLA [41] | 54.6 |
| DualCoT-VLA [44] | 55.1 |
| ABot-M0 [39] | 58.3 |
| Being-H0.7 [3] | 49.2 |
| **JoyAI-RA** | **63.2** |

显著提升的 task (vs 前最佳):
- CanToDrawerClose: +16.0 (90 vs 74)
- MilkToMicrowaveClose: +24.0 (84 vs 60)
- TrayToPot: +18.0 (88 vs 70)

这些是 long-horizon compositional task, 说明 EgoLive 的 hierarchical annotation 有效。

弱项: PlacematToTieredshelf (14%), TrayToTieredshelf (24%)。 Tieredshelf 类全部方法都弱, 可能 task 设置过难。

### 6.3 Real-World Agibot Benchmark (G1 humanoid)

5 场景 × 6 task × 20 trials:

| Task | π0.5 | JoyAI-RA |
|---|---|---|
| Headphones | — | 高 |
| Remedy | — | 高 |
| Mouse | — | 中 |
| Food Scraps | — | 弱 |
| Cup | 高 | 中 |
| Croissant | 高 | 中 |
| **Average** | **0.62** | **0.74** |

观察:
- Headphones / Remedy 涉及 accurate target recognition + final placement, JoyAI-RA 大幅领先 → semantic grounding 强
- Cup / Croissant 涉及 long-horizon multi-step, π0.5 反而更好 → 说明 action expert 的 temporal reasoning 还有提升空间
- Food Scraps 所有人都难 → bimanual coordination 是 universal challenge

---

## 7. 关键 Ablation 分析

### 7.1 EgoLive Data Scaling

RoboTwin 2.0 上, VLA pretraining only (无 VLM-only stage):

| Setting | Success (%) |
|---|---|
| No Pretraining | 81.64 |
| JDAgibot Only | 77.62 |
| EgoLive(10%) + JDAgibot | 81.40 |
| EgoLive(Full) + JDAgibot | **87.42** |

重要 insight: 
1. JDAgibot Only (77.62%) **低于** No Pretraining (81.64%)。 说明小规模 robot-only pretraining 可能引入 narrow prior, 反而 hurt generalization。 这是 data diversity 比 data relevance 更重要的证据。
2. EgoLive 10% (81.40%) 还没显示出 benefit, Full EgoLive (87.42%) 才显著。 说明 human video 的 benefit 有 scaling threshold, 跟 LLM pretraining 的 "scale matters" 直觉一致。

EgoLive 贡献最大的 task 类别 (Figure 6):
- Spatial & Stacking (Stack Blocks Three, Stack Bowls Three): +40%+ 提升
- Complex Interaction (Open Microwave): 显著
- Fine-grained Placement (Place Mouse Pad, Place Cans Plasticbox, Place Fan): 显著

### 7.2 Co-Pretraining Framework 各 stage 贡献

| Training Paradigm | Setting | Success (%) |
|---|---|---|
| Baseline (VLA Post-Training only) | — | 81.28 |
| Only VLM Co-Pretraining | w/ human data | 87.84 |
| Only VLA Co-Pretraining | w/ human data | 87.42 |
| VLM + VLA Co-Pretraining | w/ human data | **90.48** |
| VLM + VLA Co-Pretraining | w/o sim data (Stage 2) | 89.10 |
| VLM + VLA Co-Pretraining | w/ sim data (Stage 2) | 90.24 |

Insight:
1. VLM Co-Pretraining 单独 +6.56%, VLA Co-Pretraining 单独 +6.14%, 二者结合 +9.20% (不是简单相加, 说明有 overlap 也有 complementarity)
2. Sim data 在 Stage 2 贡献 +1.14%。 看似小, 但 sim 提供 cross-embodiment action diversity, 对 unseen morphology 泛化关键

### 7.3 EgoLive vs EgoDex [18] 对比

| Setting | Success (%) |
|---|---|
| Baseline | 81.28 |
| + JDAgibot only | 79.20 |
| EgoDex only | 86.88 |
| EgoLive only | 87.16 |
| EgoLive + EgoDex + JDAgibot | **89.30** |

语义分布对比 (Figure 8): EgoLive 在 nouns / verbs / adjectives 上都 **heavier-tailed**, 即 long-tail 覆盖更广。 t-SNE 显示 EgoLive 占据更大且更 continuous 的 semantic region。

互补性: EgoDex 偏 toy-like manipulation, EgoLive 偏 long-horizon real-world interaction。 Combined 后 +1.86% over EgoLive alone。

### 7.4 In-Domain EgoLive Ablation

AgiBot real-world benchmark:

| Variant | Avg |
|---|---|
| Baseline (含 in-domain EgoLive) | 0.74 |
| w/o In-Domain | 较低 |

提升最大: Remedy, Headphones (语义对齐强的 task)
反而退步: Mouse, Food Scraps (in-domain video 与 evaluation 场景 distribution mismatch, 引入 conflicting supervision)

重要 warning: in-domain data 不是万能, distribution mismatch 时会 hurt。 这是 data curation 的重要信号。

---

## 8. 跟 Related Work 的关键区别

### 8.1 vs π0 / π0.5 [5, 30]
- π0: 单一 continuous flow matching, 依赖大规模 real-robot data scaling
- JoyAI-RA: **structured multi-stage** recipe, 先 discrete FAST token, 再 continuous flow。 加 Perceiver-based action expert 解耦 VLM 和 control。

### 8.2 vs OpenVLA [23]
- OpenVLA: pure autoregressive, discrete action token
- JoyAI-RA: hybrid (text autoregressive + action flow matching)

### 8.3 vs GR00T N1 [27]
- GR00T N1: NVIDIA humanoid foundation model
- JoyAI-RA: 强调 multi-source heterogeneity + action-space unification 跨 morphology

### 8.4 vs Motus [4]
- Motus: Mixture-of-Transformer 联合 understanding / video generation / action
- JoyAI-RA: Perceiver cross-attention 做 fusion, 更轻量

### 8.5 vs ABot-M0 [39]
- ABot-M0: action manifold learning + geometry-aware perception
- JoyAI-RA: flow matching + action masking, 更朴素的 cross-embodiment 方案

参考链接:
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- GR00T N1: https://arxiv.org/abs/2503.14734
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- AgiBot-World: https://arxiv.org/abs/2503.06669
- EgoDex: https://arxiv.org/abs/2505.11709
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- RoboCasa: https://arxiv.org/abs/2406.02523
- Motus: https://arxiv.org/abs/2512.13030
- Perceiver: https://arxiv.org/abs/2103.03206

---

## 9. 直觉总结 (Build Intuition)

把 JoyAI-RA 想成三层 pipeline:

```
[Layer 1: Semantic Brain]      VLM Co-Pretraining
   web data + embodied VQA + human video + discrete FAST action
   → 学 "看懂世界 + 描述任务 + 粗粒度 action token"
   
[Layer 2: Cross-Embodiment Motor]   VLA Co-Pretraining  
   sim trajectory + real-robot + retargeted human
   unified action space + Perceiver action expert + flow matching
   → 学 "在 camera frame 下生成连续 action chunk, 跨 morphology"
   
[Layer 3: Deployment Specialist]   Post-Training
   target robot data only
   → 学 "在这个具体 robot 上的精细执行"
```

三个核心 design pillar:
1. **Data heterogeneity**: 4 源互补, web 给 semantic, human 给 behavioral diversity + scale, sim 给 controllable action supervision, real-robot 给 deployment realism
2. **Action-space unification**: camera-frame end-effector + fixed-dim vector + masking。 把 morphology 差异吸收到 representation 层
3. **Multi-stage training**: progressive specialization, 避免 catastrophic forgetting, 保留 cross-source transfer

最值得记的实验 finding:
- **JDAgibot Only < No Pretraining** → 小规模 narrow domain data 反而 hurt, diversity 才是 key
- **EgoLive 10% ≈ No Pretraining < EgoLive Full** → human video benefit 有 scaling threshold
- **VLM + VLA Co-Pretraining > 任何单 stage** → 两个 stage 互补, 不冗余

最值得关注的 weakness:
- Hanging Mug 类任务全方法都低 (≈30%), task 定义本身可能有问题, 或需要不同 inductive bias
- Long-horizon multi-step real-robot (Cup, Croissant) 仍弱于 π0.5, action expert 的 temporal reasoning 需要加强
- In-domain data 在 distribution mismatch 时 hurt, 说明 data curation 仍是 open problem

---

## 10. 我会问作者的问题

1. Unified action space 里, base mobile robot 和 bimanual dexterous hand 在同一 vector, masking 比例如何? 会不会 high-DOF system (dexterous hand) dominate gradient?
2. AdaLN 的 $\gamma(\tau), \beta(\tau)$ 是用 sinusoidal embedding 还是 MLP? 不同 denoising stage 学到的 modulation pattern 是什么?
3. Future tokens $f_{\text{future}}$ 数量 $K$ 怎么定? $H$ (action horizon) 是固定还是 variable?
4. Camera-frame action 在多 camera setup 下如何选? 还是 fused 成 single virtual camera?
5. EgoLive 60 FPS 但 robot control 通常 10-20 Hz, retargeting 时如何 downsample? 会不会丢失 fast dynamics?
6. Discrete FAST (Stage 1) 到 continuous flow matching (Stage 2) 的 transition, action token 的离散化是否成为 bottleneck?

整体看, 这篇 paper 的 contribution 主要在 **engineering recipe** (multi-source + multi-stage + unified action space) 而非新算法。 Perceiver + flow matching 都是已有组件。 真正的 insight 来自 ablation: **diversity > relevance**, **scaling threshold for human video**, **two-stage co-pretraining complementary**。 这些 finding 对未来 VLA scaling law 的研究有指导意义。
