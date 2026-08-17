---
source_pdf: One-Policy-Fits-All Geometry-Aware Action Latents for.pdf
paper_sha256: d5ec167e7ee431e5230ab8da13fbaa5ed10f8987fa9a11472e4db3e6a323c40d
processed_at: '2026-08-05T23:49:15-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，那我换个讲法，就当咱俩在白板前聊。

## 一句话总结

**OPFA 让 11 种完全不同的手（从两指夹爪到五指灵巧手）的数据可以混在一起 joint train 一个 policy，而且新来一只手只要 8 条 demo 就能 work。**

听起来很神奇，对吧？因为灵巧手跟夹爪的 action space 根本对不上——一个是 $\mathbb{R}^{16}$，一个是 $\mathbb{R}^7$，你怎么把它们 batch 到一起？

---

## 痛点在哪

讲 cross-embodiment 之前先想想 LLM 为什么能 scale。文本里所有 token 都来自同一个 vocabulary，不同语言也能用 BPE 对齐 subword。所以你把英文、法文、代码、数学全堆进去训，模型反而越来越聪明。

robot 没这个命。Inspire Hand 的 action vector 长这样：

```
[thumb_yaw, thumb_mcp, thumb_ip, index_mcp, index_pip, ..., pinky_dip]  # 16 维
```

Robotiq-2F 的 action vector 长这样：

```
[arm_x, arm_y, arm_z, arm_roll, arm_pitch, arm_yaw, gripper_open]  # 7 维
```

这两个 vector 语义上毫无对应关系，维度也对不上。你 naively concat 一个 batch 喂给 policy network，model 完全懵。

之前 people 怎么办的？三种方案，每种都有坑：

**方案 A：每个 embodiment 一个 decoder head**（Octo [11]、DexVLA [12] 这类）
就像你给每个员工配一个专属翻译。问题是 Inspire Hand 的 decoder 只能看 Inspire 的数据，学不到 XHand 数据里的知识。few-shot 场景更惨——你只有 8 条 Inspire demo，decoder 严重 overfit。

**方案 B：统一成 end-effector pose**（RT-1 [31]、RDT-1B [7]）
把所有 action 都抽象成 "末端在哪里 + gripper 开关"。对夹爪 OK，对灵巧手等于把 16 维信息压成 1 维，手型信息全丢了。这是 "降维到最笨的 embodiment"，灵巧手的 expressive power 全废。

**方案 C：retarget 到人手 MANO model**（CrossDex [38]、VideoDex [37]）
把所有 robot hand 的动作先映射到人手 model，再学 policy。问题：Inspire Hand、Allegro、Shadow Hand 跟人手几何差异很大，强行 retarget 会引入 action conflict——本来 Inspire 能抓的姿势，retarget 后变成不可达。

OPFA 的切入点：**这三条路都不走，找第四条**。

---

## OPFA 的核心 insight

关键观察：**所有 end-effector 在任意时刻的 "形状" 都可以用一个 3D point cloud 表示**。

你给我 Inspire 的 16 个 joint angle，我做 forward kinematics 算出每个 fingertip、每个指节、palm 上的点在 3D 空间中的坐标，得到一个 point cloud。你给我 Robotiq 的 7 维 action，我同样做 FK 得到另一个 point cloud。

point cloud 是 embodiment-agnostic 的——它就是个 3D 几何对象，跟 "这只手有几个 motor" 没关系。3D vision 网络 (KPConv、PointNet++、Transformer) 天生能处理变长 point cloud。

所以 OPFA 的思路是：**把 "预测 joint angle" 这件事，换成 "预测一个几何 latent"，然后再从这个 latent 解出具体 joint**。

policy 不再直接输出 $\mathbf{a} \in \mathbb{R}^{d_m}$，而是输出 $\mathbf{z} \in \mathbb{R}^{d_{latent}}$，这个 $\mathbf{z}$ 是一个统一的、跨 embodiment 的几何 representation。维度恒定，能 batch，能 joint train。

decoder 再把 $\mathbf{z}$ 翻译成具体 hardware 能执行的 joint command。

---

## 整个 pipeline 走一遍

OPFA 是 two-stage。

### Stage 1: 训一个 geometry autoencoder

这一步完全 self-supervised，不需要任何人工标注：

```python
for each embodiment m in {Inspire, XHand, Leap, Allegro, Robotiq, ...}:
    for random joint config a^m in reachable space:
        P = forward_kinematics(m, a^m)   # 算 3D 点云
        save (a^m, P) into training set
```

你只需要每个 hand 的 URDF 文件，就能生成无限多 (joint angle, point cloud) pair。这一步超 cheap，所以能 scale 到 11 个 embodiment。

然后训一个标准 encoder-decoder：

- **Encoder $f_\theta$**：吃 point cloud $\mathcal{P}$，吐 latent $\mathbf{z} \in \mathbb{R}^{d_{latent}}$
- **Decoder $g_\psi$**：吃 latent $\mathbf{z}$，吐 joint angle $\hat{\mathbf{a}}^m \in \mathbb{R}^{d_m}$
- **Loss**：$\|\hat{\mathbf{a}}^m - \mathbf{a}^m\|_2^2$ (RMSE)

这个 autoencoder 跟 VAE 的区别：没有 KL term，纯 reconstruction。目的也不是 sampling，是让 latent $\mathbf{z}$ 成为跨 embodiment 对齐的几何接口。

### Stage 2: 插到下游 policy 里

OPFA 是 base-policy-agnostic，paper 用 DP3 [17] (https://arxiv.org/abs/2403.03954) 当 backbone。DP3 是个在 3D point cloud 上跑 diffusion 的 visuomotor policy。

集成就改两个地方：

1. observation 里的 proprioceptive state $\mathbf{s_t^m}$（原本是 joint angle）替换成 $\mathcal{G}(\mathbf{s_t^m})$（当前状态的 GaLR encoding）。policy 看到的不是 "我的 thumb yaw 是 0.3 rad"，而是 "我的手现在长这个几何样子"。

2. action prediction target 从 $\mathbf{a_t^m}$ 替换成 $\mathcal{G}(\mathbf{a_t^m})$。diffusion 在 latent space denoise，不直接预测 joint。

inference 时：diffusion 输出 $\hat{\mathbf{z}}_t$ → frozen decoder → 具体 embodiment 的 joint angle → 发给 hardware。

公式 (5) 把这个 composition 写清楚：

$$\mathcal{G} = f_\theta \circ f_{FK}^m$$

意思是 $\mathcal{G}$ 这个函数把 joint angle 变成 GaLR：先做 FK 变 point cloud，再用 encoder 抽 latent。

公式 (6) 定义 policy：

$$\pi_m: \mathbf{o_t^m} \mapsto \mathcal{G}(\mathbf{a_t^m}) \in \mathbb{R}^{d_{latent}}$$

policy 学的是 observation → 目标 GaLR 的映射，不是 observation → 具体 joint。

---

## Encoder 长啥样

这里有几个有意思的设计选择。

### Multi-scale 下采样

原始 point cloud 太密（每个 hand 几千个点），直接处理冗余又慢。OPFA 做三层下采样：

- $\mathcal{P}$：原始 dense 点云
- $\tilde{\mathcal{P}}$：第一层下采样（中等密度）
- $\hat{\mathcal{P}}$：superpoints（最稀疏，几百个点）

满足 $|\hat{\mathcal{P}}| < |\tilde{\mathcal{P}}| < |\mathcal{P}|$。

这个 multi-scale 是为了在不同尺度抓不同信息：fine scale 抓 fingertip 这种细结构，coarse scale 抓 palm 整体形状。

### KPConv 抽 local feature

local feature 用 KPConv [14]（Thomas et al. ICCV 2019, https://arxiv.org/abs/1904.08868）。公式 (1) 是它的核心：

$$g(\mathbf{x}) = \sum_i \sum_{k<K} h(\mathbf{y_i}, \tilde{\mathbf{x_k}}) \mathbf{W_k} f_{\mathbf{x_i}}$$

变量含义：
- $\mathbf{x}$：query point（中心）
- $\mathbf{x_i}$：$\mathbf{x}$ 邻域内的 neighbor
- $\mathbf{y_i} = \mathbf{x_i} - \mathbf{x}$：neighbor 相对中心的坐标
- $\tilde{\mathbf{x_k}}$：第 $k$ 个 kernel point（$K$ 个 kernel points 散布在球内）
- $\mathbf{W_k} \in \mathbb{R}^{D_{in} \times D_{out}}$：第 $k$ 个 kernel point 的权重矩阵
- $f_{\mathbf{x_i}}$：neighbor $\mathbf{x_i}$ 的 input feature
- $h(\cdot, \cdot)$：proximity function

公式 (2) 定义 $h$：

$$h(\mathbf{y_i}, \tilde{\mathbf{x_k}}) = \max\left(0, 1 - \frac{\|\mathbf{y_i} - \tilde{\mathbf{x_k}}\|}{\sigma}\right)$$

$\sigma$ 控制 kernel point 影响半径。neighbor 离 kernel point 越近权重越大，距离超过 $\sigma$ 直接归零。

**Intuition**：KPConv 就是 3D 版的 conv2d。2D conv 用方形 grid kernel，KPConv 用一组在球内离散分布的 kernel points（几何上类似 ball packing）。每个 kernel point 关联一个 weight matrix，output 是邻居 features 按到各个 kernel point 的距离加权求和。好处是 kernel point 位置可学习，能 adapt 局部几何结构。

### Geometric Transformer 抓 global feature + 双重位置编码

KPConv 抽完 multi-scale local feature，得到一组 superpoint features $\{f_p\}_{p \in \hat{\mathcal{P}}}$。接下来在 superpoint 层级做 transformer cross-attention 抓 global gesture。

但直接做 attention 有个坑——**positional ambiguity**。cross-embodiment 时，不同 hand 的 superpoint 在物理空间分布完全不同。Inspire 的 thumb 在某个 3D 位置，Allegro 的 thumb 在另一个 3D 位置，光看坐标 transformer 学不到 "这两个是同一个语义部位"。

OPFA 的 trick 是**双重位置编码**：

**Coordinate embedding $r^p$**：naive 的 3D 坐标，告诉 transformer 这个 superpoint 在物理空间哪里。

**Semantic embedding $r^s$**：给每个 superpoint $p$ 分配一个 2D 语义 index：

$$\pi(p) = (u_p, v_p)$$

- $u_p \in \{0, 1, 2, 3, 4, 5\}$：finger index（palm=0, thumb=1, index=2, middle=3, ring=4, little=5）
- $v_p \in \mathbb{Z}_{\geq 0}$：指节 index（从 base 到 tip 递增）

投影到 feature space：

$$r^s = \mathbf{s_p} \mathbf{W^S} \in \mathbb{R}^{d_t}$$

$\mathbf{W^S} \in \mathbb{R}^{2 \times d_t}$ 是投影矩阵，$d_t$ 是 superpoint feature 维度。

公式 (3) 加起来送进 transformer：

$$\tilde{f}_p = \text{Transformer}(f_p, r^p + r^s)$$

**这个 semantic embedding 是整个 paper 最聪明的 trick**。考虑 Inspire 和 Allegro 都有 thumb + 4 个 finger，几何尺寸天差地别，但语义结构同构。只看坐标，transformer 看到的是两组无关的 3D 点；加上 semantic embedding，transformer 知道 "Inspire 的 thumb segment 0" 和 "Allegro 的 thumb segment 0" 语义对应，attention 能 cross-embodiment 工作。这是 OPFA 能让 transformer 参数跨 embodiment 共享的关键。

最后公式 (4) 做 global average pooling 得到 GaLR：

$$z = \frac{1}{|\hat{\mathcal{P}}|} \sum_{p \in \hat{\mathcal{P}}} \tilde{f}_p$$

所有 embodiment 的 GaLR 都是 $\mathbb{R}^{d_{latent}}$ 维，**action space 维度被强制对齐了**。

参考：Geometric Transformer 原文 Qin et al. CVPR 2022, https://arxiv.org/abs/2202.05661

---

## Decoder 的设计：Universal Hand Model

这是 paper 第二个关键贡献。

之前方案 A 给每个 embodiment 单独 decoder，OPFA 用**一个 unified decoder** 处理所有 embodiment。怎么做到的？

构造一个 **hypothetical universal hand model $\mathcal{H}$**——一个超集，包含所有可能 end-effector 的所有 physical joint：

- thumb yaw, thumb base flexion, thumb MCP, thumb IP
- index yaw (abduction), index MCP, index PIP, index DIP
- middle/ring/little finger 同理
- palm DOF, wrist roll/pitch/yaw
- gripper open/close (for parallel jaw)
- ...

Decoder $g_\psi$ 一次性输出 $\hat{\Theta} \in \mathbb{R}^{|\mathcal{H}|}$（universal hand 的所有 joint）。对具体 embodiment $m$，从 $\hat{\Theta}$ 中 **select** 出它实际有的 joints，得到 $\hat{\Theta}^m \in \mathbb{R}^{d_m}$，算 RMSE loss。

**Intuition**：这等于把 "不同 embodiment 不同维度 decoder" 转成 "同一个高维 decoder + mask"。参数共享最大化，few-shot 时不会因为某 embodiment 数据少导致 decoder head underfit。

这个 idea 跟 NLP 里 BPE vocabulary、speech 里统一 phoneme set、pose estimation 里 SMPL-X (https://smpl-x.is.tue.mpg.de/) 的设计哲学一致：**用一个 superset 顶一个统一接口**。SMPL-X 把人体、手、脸全压进一个 parametric model，OPFA 把所有 robot hand 压进一个 universal joint superset。

---

## 看几个实验数字感受一下

### Spatial generalization (Table I)

setup：Inspire 和 XHand 各 72 demos，但数据在 workspace 不同区域。测试时让 A 在 B 见过的区域执行。

| Setting | Inspire (Kettle) | Inspire (Banana) | XHand (Bucket) |
|---|---|---|---|
| w/o Co-train | 30.0 | 10.0 | 1.0 |
| Naive Co-train | 57.0 | 83.0 | 33.0 |
| **OPFA** | **83.0** | **98.0** | **94.0** |

w/o Co-train 几乎全崩——没见过的区域当然不会。Naive Co-train 有点帮助但不稳定。OPFA 全面碾压。

这个 spatial generalization 表明 OPFA 在 **wrist-level trajectory** 这层共享了知识。不同 hand 抓同一个 kettle，手腕运动是同构的，OPFA 通过 GaLR 把这个同构性提取出来了。

### Few-shot learning (Figure 4)

setup：Inspire 72 demos + XHand $n$ demos co-train，$n \in \{1, 2, 4, 8, 12, 18, 36\}$。

关键数据点：**$n=8$ 时 Inspire 已经接近 72-demos 的 full-train 性能**。

更有意思的是 Naive Co-train 出现 **training inhibition**——随着 demo 数增加，性能反而下降，甚至低于 w/o Co-train baseline。

**Intuition**：Naive Co-train 用 embodiment-specific decoder，XHand 只有 8 demos 时 XHand decoder 严重 overfit，同时 shared encoder 被 noisy 信号干扰，Inspire 性能跟着下降。OPFA 没这个问题，因为 unified decoder 强制 cross-embodiment 参数共享，8 demos 也受益于 Inspire 的 72 demos 学到的 decoder 权重。

Figure 5 在 9 个 embodiment 上做实验，每个只 8 demos，co-train 72 个总 demos。OPFA 平均 62.1%，Leap Hand 达 93%，比 Naive Co-train 高 20%+。

对照 RT-X (https://robotics-transform-x.github.io/) 的 cross-embodiment effort，OPFA 在灵巧手上的 few-shot 效率明显更高。

### Real-world (Table II)

7 个真实任务，覆盖：
- Pick&place: Basket, Mango
- Contact-intensive deformable: Tissue, Pot (pouring)
- Long-horizon: Broom (sweeping), Drawer (multi-step)
- Fine-motor: Syringe

最 striking 的数字是 **Pot (pouring) 任务**：Naive Co-train 在 Leap Hand 上只有 30%，比 w/o Co-train 的 40% 还差。OPFA 达到 80%。这直接验证 paper 的核心 claim——结构差异大的 embodiment 做 Naive co-train 会有 action conflict，必须用 geometry-aware 表示才能避免。

**Syringe** (precision) 上 Leap Hand 100%、XHand 100%。说明 GaLR 不只学粗粒度 grasp，fine-motor control 的连续控制信号也能 cross-embodiment 共享。

Robotiq-2F (gripper) 在 Basket 任务上从 30% 提到 90%。说明 framework 不只对 dexterous hand 有效，gripper 也受益于 cross-embodiment co-training with hands。这是双向的知识流动。

---

## 横向对比一下定位

### vs Octo (https://octo-models.github.io/)
Octo 用 standardized end-effector space 做 pretrain，fine-tune 时用 modular adapter 适配新 action space。问题：每个新 embodiment 还是要 fine-tune adapter，不是真正 "one-policy"。OPFA 的 unified decoder 不需要任何 per-embodiment tuning。

### vs RDT-1B (https://arxiv.org/abs/2410.07864)
RDT-1B 把所有 action 统一成 "Physically Interpretable Unified Action Space" (end-effector pose + gripper state)。问题：灵巧手的高 DoF 信息全丢，降维到 1 DoF gripper。OPFA 保留全部 DoF 通过 point cloud latent。

### vs CrossDex (https://arxiv.org/abs/2410.02479)
CrossDex 用 MANO human hand model 做中间桥梁。问题：robot hand 跟人手几何差异大时 retargeting 引入 conflict。OPFA 直接学 robot hand 的 point cloud 表示，不需要人手 prior。

### vs Latent Action Diffusion (Bauer et al. https://arxiv.org/abs/2506.14608)
最接近的工作，也在 latent space 做 diffusion，但每个 embodiment 单独 decoder。OPFA 的 unified decoder 是关键差异。

### vs π0 (https://arxiv.org/abs/2410.24164)
π0 是 Physical Intelligence 的 VLA flow model，主要在 gripper 类 embodiment 上做 scaling。OPFA 的 geometry-aware 思路可以 complement π0，让 π0 扩展到灵巧手。

---

## 跨领域联想

### 跟 NLP tokenizer 的类比
OPFA 的 universal hand model $\mathcal{H}$ + select-mask 机制，本质上是 **continuous 版的 BPE vocabulary**。NLP 用离散 token vocabulary 跨语言共享 subword，OPFA 用 universal joint superset 跨 embodiment 共享 sub-action。

### 跟 SMPL-X 的类比
SMPL-X (https://smpl-x.is.tue.mpg.de/) 是 human body + hand + face 的统一 parametric model。OPFA 的 universal hand model 是 robot hand 的类比，但更激进——它统一 gripper + dexterous hand 这种结构完全不同的 embodiment。

### 跟 VAE 的类比
OPFA 的 encoder-decoder 结构很像 VAE，但有几个关键差异：
1. 输入是 point cloud（geometry），不是 image
2. 没有 KL regularization，纯 reconstruction loss
3. latent space 不是为生成，是为 cross-embodiment alignment

### 跟 Differentiable Simulation 的联系
$f_{FK}^m$ 是 forward kinematics，OPFA 在 Stage 1 把它当 fixed operator。如果换成 differentiable FK + differentiable simulation，整个 framework 可以扩展到 RL setting——policy 在 latent space 学，gradient 通过 decoder 和 FK 回传到 encoder。这是个很自然的 follow-up 方向。

### 跟 Contrastive Learning 的可能结合
当前 OPFA 用 reconstruction loss 训 GaLR。可以加 contrastive loss：同一个 grasp pose 在不同 embodiment 下的 point cloud 应该 map 到接近的 latent。这种 cross-embodiment contrastive 可能学到更 semantic 的 representation，类似 CLIP 在 image-text 上的做法。

### 跟 LLM scaling law 的连接
OPFA 的真正野心是**让 robot learning 也能享受 scaling law**。LLM 之所以能 scale，是因为所有数据可以拼进一个 batch。OPFA 通过 geometry 统一 embodiment，让 robot data 也能拼进一个 batch。如果这条路走通，未来可能有 robot foundation model，跨 embodiment、跨 task、跨 scene 一起训。

---

## 局限性

### Two-stage 训练的耦合
Stage 1 的 encoder-decoder freeze 后插到 Stage 2。如果 Stage 1 学到的 latent 没保留 task-relevant 细节，Stage 2 无法纠正。改进方向是 end-to-end joint training，让 reconstruction loss 和 policy loss 一起优化。但这样需要 differentiable FK 和 careful loss balancing。

### Point cloud 表示的损失
从 joint angle → FK → point cloud → latent → joint angle 是个 bottleneck。point cloud 是离散采样，可能丢失 fine-grained joint correlation。改进方向是用更密集的点云或隐式表示（NeRF/SDF）替代。

### 缺少 contact 和 tactile 信息
Inspire Hand 有 tactile 版本，但 paper 里 GaLR 只用 kinematic point cloud，没用到 tactile signal。把 tactile 融入 GaLR 是个 obvious extension。

### 仅在 manipulation 验证
Framework 理论上可以扩展到 locomotion（不同 legged robot），但 paper 没验证。把 OPFA 用到 cross-embodiment locomotion 是个 future work 方向。

### Universal hand model 的设计
当前 $\mathcal{H}$ 是 manually designed 的 superset。如果 embodiment 数量继续扩展（上百个），这个 set 设计会变复杂。改进是 **learned universal joint set**，从数据中自动发现 semantic joint categories。

---

## 我的总评

OPFA 是一个 **结构清晰、insight 明确、实验扎实** 的工作。核心贡献是把 cross-embodiment 问题转成几何问题——**用 point cloud 作为 embodiment-agnostic 的中间表示**。

技术上几个值得记住的点：
1. **GaLR = 3D point cloud encoder + transformer + GAP**，简单但有效
2. **Semantic positional embedding** 是 cross-embodiment transformer 的 key trick
3. **Universal hand model + select-mask** 是替代 per-embodiment decoder 的优雅方案
4. **Two-stage 解耦** 让 GaLR 可以 scale 到任意多 embodiment 而不需要 task-specific 数据

从 research direction 看，OPFA 代表 robot learning 中一种**通过几何抽象统一 embodiment**的范式。这条路线跟"通过语言统一 task" (VLA, RT-2, π0) 和"通过演示接口统一 hardware" (UMI) 是 complement 的。未来很可能融合：用 LLM 提供 task semantic，用 UMI 类接口收集 data，用 OPFA 类方法统一 embodiment，最后训一个真正 universal 的 robot foundation model。

这 paper 让我想到你之前在 Tesla 讲过的 "AI 是 software 2.0"——OPFA 这类工作本质上是在把 cross-embodiment 这种 implicit 的、靠 engineer 手动 retarget 的 knowledge，变成 explicit 的、learnable 的 geometry representation。这就是把 software 1.0 的 hard-coded 映射，变成 software 2.0 的 learned representation。

参考链接汇总：
- OPFA Project: https://mujc2021.github.io/opfa/
- DP3: https://arxiv.org/abs/2403.03954
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- KPConv: https://arxiv.org/abs/1904.08868
- Geometric Transformer: https://arxiv.org/abs/2202.05661
- Octo: https://arxiv.org/abs/2405.12213
- RDT-1B: https://arxiv.org/abs/2410.07864
- CrossDex: https://arxiv.org/abs/2410.02479
- UMI: https://arxiv.org/abs/2402.10329
- Open X-Embodiment: https://robotics-transform-x.github.io/
- π0: https://arxiv.org/abs/2410.24164
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- Latent Action Diffusion: https://arxiv.org/abs/2506.14608

---

# OPFA: One-Policy-Fits-All 深度技术解读

这篇 paper 是 Shanghai AI Lab + Tsinghua + CUHK 的工作，本质上是想解决一个困扰 robot learning 很久的问题：**不同 end-effector (gripper, dexterous hand) 的数据能不能放一起 joint training？** 这个问题在 LLM 时代显得格外刺眼——文本和视觉可以堆 trillion token 训练一个大模型，但 robot data 因为 embodiment 耦合而被碎片化成上百个 "小数据集孤岛"。OPFA 给出了一个几何驱动的解决方案。Project page: https://mujc2021.github.io/opfa/

---

## 1. 问题本质的 intuition

先讲清楚 cross-embodiment 难在哪。假设你有 Inspire Hand (16 DoF)、XHand (15 DoF)、Robotiq-2F (1 DoF) 三种末端执行器，要 pick 一个 spray bottle。

**Action space 不一致**：Inspire 的 action 是 $\mathbb{R}^{16}$ 的 joint angle 向量，XHand 是 $\mathbb{R}^{15}$，Robotiq 是 $\mathbb{R}^{7}$ (6 DoF arm + 1 DoF gripper)。你没办法把它们 concat 成一个 batch 喂给同一个 action head。

**几何结构不一致**：Inspire 是 5 指人形手，Leap 是 4 指欠驱动手，Robotiq 是 2 指 parallel jaw。它们 "抓同一个 spray" 的动作语义完全不同。

现有方案的困境：
- **Embodiment-specific decoder** (Octo [11], DexVLA [12])：每个 hand 一个 head。问题是 decoder $g_{\psi_m}$ 只能看 $\mathcal{D}_m$ 的数据，few-shot 时严重 overfit，shared representation 学不到 cross-embodiment 的几何共性。
- **统一 action space** (RT-1 [31], RT-2 [32], RDT-1B [7])：把 action 全部抽象成 end-effector pose + gripper open/close。这对 parallel-jaw gripper 有效，但灵巧手的高 DoF 信息全丢了，本质上是 "降维到最简 embodiment"，浪费了灵巧手的表达能力。
- **MANO retargeting** (CrossDex [38], VideoDex [37])：把所有 hand 先 retarget 到 MANO human hand model 再学策略。问题在于 Inspire 和 Allegro 几何上跟人手差很远，强行 align 会引入 action conflict。

OPFA 的核心 insight：**所有 end-effector 的几何状态都可以表示成 point cloud**。point cloud 是 embodiment-agnostic 的几何对象，3D point cloud 处理网络可以天然地跨 embodiment 共享参数。把 "预测 action" 这件事转成 "预测一个 latent 几何表示"，再用 unified decoder 反解到具体 joint angle。

---

## 2. 整体架构 (Figure 2 解析)

OPFA 是一个 **two-stage** pipeline，这一点很关键，因为它把 "学几何表示" 和 "学策略" 解耦了，类似 VAE pretrain + downstream 的思路。

### Stage 1: 自监督学习 GaLR (Geometry-Aware Latent Representation)

数据生成是 **完全自动化、annotation-free** 的：

```
For each embodiment m:
    Sample joint configurations a^m ∈ J^m  (random reachable states)
    P^m = f_FK^m(a^m)  # forward kinematics → point cloud
    Save (a^m, P^m) pairs
```

这一步只用了 URDF + FK，没有人工标注。这是它能 scale 到 11 个 embodiment 的关键——你需要的就是每个 hand 的 URDF 文件。

然后训 encoder-decoder：
- Encoder $f_\theta$: point cloud $\mathcal{P}$ → latent $\mathbf{z} \in \mathbb{R}^{d_{latent}}$
- Decoder $g_\psi$: latent $\mathbf{z}$ → joint angles $\hat{\mathbf{a}}^m \in \mathbb{R}^{d_m}$
- Loss: RMSE between $\hat{\mathbf{a}}^m$ 和 ground truth $\mathbf{a}^m$

### Stage 2: 集成进 downstream policy (DP3 backbone)

把 Stage 1 的 encoder-decoder 冻结，插到 DP3 [17] (https://arxiv.org/abs/2403.03954) 里：
- observation 中的 proprioceptive state $\mathbf{s_t^m}$ 用 $\mathcal{G}(\mathbf{s_t^m})$ 替换
- action prediction target 从 joint angle 改成 GaLR $\mathbf{z}$
- 训 diffusion policy 在 latent space 做 denoise
- inference 时用 pretrained decoder 把 $\mathbf{z}$ 反解成具体 embodiment 的 joint angle

公式 (5) 和 (6) 是这个映射的形式化：

$$\mathcal{G} = f_\theta \circ f_{FK}^m$$
$$\pi_m: \mathbf{o_t^m} \mapsto \mathcal{G}(\mathbf{a_t^m}) \in \mathbb{R}^{d_{latent}}$$

注意 $f_\theta \circ f_{FK}^m$ 是个复合函数：先做 FK 把 joint angle 变成 point cloud，再用 vision encoder 抽 latent。这个 composition 的巧妙之处在于：**policy 输出和 "目标几何状态" 对应，而不是和 joint angle 直接对应**。policy 看到的是 "我希望末端执行器呈现什么空间形状"，decoder 负责把这个几何意图翻译成具体 hardware 能执行的 joint 命令。

---

## 3. Encoder 细节：KPConv + Geometric Transformer

这是 paper 里技术含量最高的部分，值得逐行展开。

### 3.1 Multi-scale point cloud 下采样

原始 dense point cloud $\mathcal{P} \in \mathbb{R}^{|\mathcal{P}| \times 3}$ 经过三次下采样得到：
- $\tilde{\mathcal{P}}$：第一层下采样（中等密度）
- $\hat{\mathcal{P}}$：superpoints（最稀疏）

满足 $|\hat{\mathcal{P}}| < |\tilde{\mathcal{P}}| < |\mathcal{P}|$。这个 multi-scale 结构是借鉴 PointNet++ 和 KPConv 原文的设计，目的是在不同尺度捕捉局部几何特征——比如 finger tip 这种细结构在 fine scale 抽，palm 整体形状在 coarse scale 抽。

### 3.2 KPConv (Kernel Point Convolution)

公式 (1) 是 KPConv 的核心：

$$g(\mathbf{x}) = \sum_i \sum_{k<K} h(\mathbf{y_i}, \tilde{\mathbf{x_k}}) \mathbf{W_k} f_{\mathbf{x_i}}$$

变量含义：
- $\mathbf{x}$：query point (中心点)
- $\mathbf{x_i}$：$\mathbf{x}$ 邻域内的 neighboring point，$i$ 是 neighbor 索引
- $\mathbf{y_i} = \mathbf{x_i} - \mathbf{x} \in B_r^3$：相对坐标，$B_r^3 = \{\mathbf{y} \in \mathbb{R}^3 | \|\mathbf{y}\| \leq r\}$ 是半径 $r$ 的球
- $\tilde{\mathbf{x_k}}$：$k$-th kernel point，$k < K$，$K$ 是 kernel point 数量（KPConv 原文默认 $K=15$）
- $\mathbf{W_k} \in \mathbb{R}^{D_{in} \times D_{out}}$：$k$-th kernel point 的 weight matrix
- $f_{\mathbf{x_i}} \in \mathbb{R}^{D_{in}}$：input feature at $\mathbf{x_i}$
- $h(\cdot, \cdot)$：proximity function

公式 (2) 是 proximity function 的实现：

$$h(\mathbf{y_i}, \tilde{\mathbf{x_k}}) = \max\left(0, 1 - \frac{\|\mathbf{y_i} - \tilde{\mathbf{x_k}}\|}{\sigma}\right)$$

- $\sigma$：控制 kernel point 影响半径的超参
- 这是一个 **linear decay + ReLU truncation**：neighbor 离 kernel point 越近，权重越大；距离超过 $\sigma$ 就置 0

**Intuition**：KPConv 可以理解为 "3D 版的 conv2d"。2D conv 用方形 grid kernel，KPConv 用一组在球内离散分布的 kernel points（几何上类似球填充），每个 kernel point 关联一个 weight matrix。公式 (1) 就是把所有 neighbor 的 feature 按它们到各个 kernel point 的距离加权求和。这种设计的优势是 kernel point 位置可学习，能 adapt 到不同局部几何结构。

参考：KPConv 原文 Thomas et al. ICCV 2019 https://arxiv.org/abs/1904.08868

### 3.3 Geometric Transformer + 双重位置编码

KPConv 抽完 multi-scale local feature 后，得到一组 superpoint features $\{f_p\}_{p \in \hat{\mathcal{P}}}$。接下来在 superpoint 层级做 cross-attention 捕捉 **global gesture information**。

直接对 superpoint 做 attention 会遇到 **positional ambiguity** 问题：OPFA 跨 embodiment，不同 hand 的 superpoint 在物理空间分布完全不同，单纯用坐标做位置编码会让 transformer 学不到稳定的结构关系。

OPFA 的解法是 **双重 positional embedding**：

**Coordinate positional embedding $r^p$**：naive 的 3D 坐标编码，告诉 transformer 这个 superpoint 在物理空间哪里。

**Semantic positional embedding $r^s$**：这是 paper 的一个 key insight。给每个 superpoint $p$ 分配一个 2D 语义 index：

$$\pi(p) = (u_p, v_p)$$

- $u_p \in \{0, 1, 2, 3, 4, 5\}$：finger-level index
  - $u_p = 0$: palm
  - $u_p = 1$: thumb
  - $u_p = 2$: index finger
  - $u_p = 3$: middle finger
  - $u_p = 4$: ring finger
  - $u_p = 5$: little finger
- $v_p \in \mathbb{Z}_{\geq 0}$：segment-level index along finger（从 base 到 tip 递增）

形成 2D index vector $\mathbf{s_p} = [u_p, v_p]^\top \in \mathbb{R}^2$，线性投影成 embedding：

$$r^s = \mathbf{s_p} \mathbf{W^S} \in \mathbb{R}^{d_t}$$

- $\mathbf{W^S} \in \mathbb{R}^{2 \times d_t}$：projection matrix
- $d_t$：superpoint feature dimension

公式 (3) 把两种 embedding 加起来送进 transformer：

$$\tilde{f}_p = \text{Transformer}(f_p, r^p + r^s), \quad p \in \hat{\mathcal{P}}$$

**为什么 semantic embedding 重要**：考虑 Inspire Hand 和 Allegro Hand，它们都有 thumb + 4 个 finger，几何尺寸完全不同，但语义结构同构（都是 "thumb 在内侧，4 指在外侧排开"）。如果只用坐标编码，transformer 看到的就是两组完全无关的 3D 点云；加上 semantic embedding 后，transformer 知道 "Inspire 的 thumb segment 0" 和 "Allegro 的 thumb segment 0" 在语义上是对应的，attention 可以 cross-embodiment 工作。这是 OPFA 能跨 embodiment 共享 transformer 参数的关键 trick。

最后公式 (4) 做 global average pooling 得到 GaLR：

$$z = \frac{1}{|\hat{\mathcal{P}}|} \sum_{p \in \hat{\mathcal{P}}} \tilde{f}_p$$

所有 embodiment 的 GaLR 都是 $\mathbb{R}^{d_{latent}}$ 维，**action space 维度被强制对齐了**。

参考：Geometric Transformer 原文 Qin et al. CVPR 2022 https://arxiv.org/abs/2202.05661

---

## 4. Unified Decoder with Latent Retargeting

这是另一个关键技术贡献，解决 "如何从一个统一 latent 解出不同 embodiment 的 joint"。

### 4.1 Universal Hand Model $\mathcal{H}$

OPFA 构造一个 **hypothetical universal hand model** $\mathcal{H}$，包含所有可能 end-effector 的所有 physical joint：

- thumb yaw, thumb base flexion, thumb MCP, thumb IP
- index finger yaw (abduction), index MCP, index PIP, index DIP
- middle/ring/little finger 同理
- palm DOF, wrist roll/pitch/yaw
- gripper open/close (for parallel jaw)

这个 universal hand 是一个 **超集**——比如 Robotiq-2F 只有 "gripper open/close" 这一个 joint，那它对应 $\mathcal{H}$ 中那个 joint 的子集；Inspire Hand 用 $\mathcal{H}$ 中所有 finger joints；Shadow Hand 用最多 joints。

### 4.2 Select-then-decode

Decoder $g_\psi$ 一次性预测 $\hat{\Theta} \in \mathbb{R}^{|\mathcal{H}|}$（universal hand 的所有 joint）。对每个具体 embodiment $m$，从 $\hat{\Theta}$ 中 **select** 出它实际有的 joints，得到 $\hat{\Theta}^m \in \mathbb{R}^{d_m}$。

Loss 是 RMSE：

$$\mathcal{L} = \|\hat{\Theta}^m - \mathbf{a}^m\|_2^2$$

**Intuition**：这个设计把 "不同 embodiment 不同维度 decoder" 转成 "同一个高维 decoder + mask"。参数共享最大化，few-shot 时不会因为某个 embodiment 数据少导致 decoder head underfit。同时 universal joint set 提供了隐式的 **semantic alignment**——不同 embodiment 的 "thumb flexion" 共享同一个 latent dimension，自然 encourage encoder 学到对齐的 representation。

这个 idea 跟 NLP 里 BPE vocabulary、speech 里统一 phoneme set、以及 pose estimation 里 SMPL-X (https://smpl-x.is.tue.mpg.de/) 的设计哲学一致：**用一个 superset 顶一个统一接口**。

---

## 5. 跟 downstream policy (DP3) 集成

OPFA 是 base-policy-agnostic，paper 中用 DP3 [17] (Ze et al., https://arxiv.org/abs/2403.03954) 做实现。DP3 本身是一个在 3D point cloud 上做 diffusion 的 visuomotor policy。

集成方式：

1. **State 修改**：原 DP3 的 state $\mathbf{s_t^m}$ (joint angle) 替换成 $\mathcal{G}(\mathbf{s_t^m})$ (GaLR encoding of current state)。policy 看到的是 "当前末端执行器的几何状态 latent"，不是 raw joint angle。

2. **Action 修改**：原 DP3 的 action target $\mathbf{a_t^m}$ 替换成 $\mathcal{G}(\mathbf{a_t^m})$ (GaLR encoding of target state)。diffusion 在 GaLR 空间做 denoise。

3. **Inference**：diffusion 输出 predicted GaLR $\hat{\mathbf{z}}_t$，用 pretrained decoder $g_\psi$ 反解成具体 embodiment 的 joint angle $\hat{\Theta}^m$，发给 hardware。

这个设计的好处：**整个 policy 网络参数是 cross-embodiment 共享的**，没有任何 embodiment-specific 的 head。所有 cross-embodiment 信息都通过 GaLR 这个统一接口流动。

---

## 6. 实验结果分析

### 6.1 Embodiment 覆盖

11 个 end-effector，覆盖从 1 DoF 到 ~20 DoF：
- Real world: XHand, Inspire Hand (tactile), Robotiq-2F-85, Leap Hand
- Sim only: UMI, Robotiq-3F, Allegro, Shadow Hand, Ability Hand, Schunk SVH, Inspire Hand (non-tactile)

这个覆盖范围在 cross-embodiment paper 里属于 SOTA 的，对比 CrossDex (https://arxiv.org/abs/2410.02479) 主要做 5 个 hand，DexVLA 主要做 gripper 类。

### 6.2 Spatial Generalization (Table I)

实验设置：每个 embodiment 72 demos，但数据分布在 workspace 不同区域。测试时让 embodiment A 在 embodiment B 见过的区域执行。

| Setting | Inspire (Kettle) | Inspire (Bucket) | Inspire (Banana) | XHand (Bucket) | XHand (Banana) |
|---|---|---|---|---|---|
| w/o Co-train | 30.0 | 3.0 | 10.0 | 1.0 | 5.0 |
| Naive Co-train | 57.0 | 50.0 | 83.0 | 33.0 | 30.0 |
| **OPFA** | **83.0** | **75.0** | **98.0** | **94.0** | **67.0** |

Inspire Hand 在 Kettle 任务上从 30% → 83%，提升 53%。这种 spatial generalization 表明 OPFA 在 **wrist-level trajectory** 这一层共享了知识——不同 hand 抓同一个 kettle 在手腕轨迹上是同构的，OPFA 通过 GaLR 把这个同构性提取出来了。

### 6.3 Object Generalization (Table I 最后一列)

更挑战的 setup：Inspire Hand 只见过抓 can，XHand 只见过抓 spray，co-train 后 cross-test。

| Setting | Inspire (spray) | XHand (can) |
|---|---|---|
| w/o Co-train | 1.0 | 41.0 |
| Naive Co-train | 57.0 | 53.0 |
| **OPFA** | **83.0** | **71.0** |

OPFA 比 Naive Co-train 高 26% (Inspire) 和 18% (XHand)。这个结果很有意思——object generalization 比 spatial generalization 更难，但 OPFA 仍然显著领先，说明 GaLR 不只学到 wrist trajectory 共性，连 grasp pattern 的部分共性也学到了。

### 6.4 Few-shot Learning (Figure 4, 5)

实验设置：Inspire Hand 72 demos + XHand $n$ demos co-train，$n \in \{1, 2, 4, 8, 12, 18, 36\}$。

关键数据点：
- **$n=8$ 时 Inspire Hand 已经接近 72-demos 的 full-train 性能**
- Naive Co-train 出现 **training inhibition**：随着 demo 数增加，性能反而下降，甚至低于 w/o Co-train baseline

这个 training inhibition 现象的 intuition：Naive Co-train 用 embodiment-specific decoder，当 XHand 只有 8 demos 时，XHand decoder 严重 overfit；同时 shared encoder 被 XHand 的 noisy 8-demo 信号干扰，导致 Inspire 性能也下降。OPFA 没有这个问题，因为 unified decoder 强制 cross-embodiment 参数共享，8 demos 也能受益于 Inspire 的 72 demos 学到的 decoder 权重。

Figure 5 在 9 个 embodiment 上做实验，每个 embodiment 只 8 demos，co-train 72 个总 demos。OPFA 平均 62.1%，Leap Hand 达到 93%，比 Naive Co-train 高 20%+。这是非常强 few-shot 结果，对照 RT-X (https://robotics-transform-x.github.io/) 的 cross-embodiment effort，OPFA 在灵巧手上的 few-shot 效率明显更高。

### 6.5 Real-world 实验 (Table II)

7 个真实任务，覆盖：
- Pick&place: Basket, Mango
- Contact-intensive deformable: Tissue, Pot (pouring)
- Long-horizon: Broom (sweeping), Drawer (multi-step)
- Fine-motor: Syringe

亮点：
- **Pot (pouring) 任务**：Naive Co-train 在 Leap Hand 上只有 30%，比 w/o Co-train 的 40% 还差。OPFA 达到 80%。这个结果直接验证了 paper 的核心 claim——结构差异大的 embodiment 做 Naive co-train 会有 action conflict，必须用 geometry-aware 表示才能避免。
- **Syringe (precision)**：Leap Hand 100%, XHand 100%。说明 GaLR 不只学粗粒度 grasp，fine-motor control 的连续控制信号也能 cross-embodiment 共享。
- **Robotiq-2F (gripper)**：在 Basket 任务上从 30% (w/o Co-train) 提到 90% (OPFA)。说明这个 framework 不只对 dexterous hand 有效，gripper 也受益于 cross-embodiment co-training with hands。

---

## 7. 跟相关工作的对比定位

### 7.1 vs Octo (https://octo-models.github.io/)
Octo 用 standardized end-effector space 做 pretrain，然后 fine-tune 时用 modular adapter 适配新 action space。问题：每个新 embodiment 还是要 fine-tune adapter，不是真正 "one-policy"。OPFA 的 unified decoder 不需要任何 per-embodiment tuning。

### 7.2 vs RDT-1B (https://arxiv.org/abs/2410.07864)
RDT-1B 把所有 action 统一成 "Physically Interpretable Unified Action Space" (end-effector pose + gripper state)。问题：把灵巧手降维成 1 DoF gripper，丢失高 DoF 信息。OPFA 保留全部 DoF 通过 point cloud latent。

### 7.3 vs CrossDex (https://arxiv.org/abs/2410.02479)
CrossDex 用 MANO human hand model 做中间桥梁。问题：robot hand 跟人手几何差异大时 retargeting 引入 conflict。OPFA 直接学 robot hand 的 point cloud 表示，不需要人手 prior。

### 7.4 vs Latent Action Diffusion (Bauer et al. https://arxiv.org/abs/2506.14608)
这个是最接近的工作，也是在 latent space 做 diffusion。但它每个 embodiment 训一个 decoder。OPFA 的 unified decoder 是关键差异。

### 7.5 vs UMI (https://arxiv.org/abs/2402.10329)
UMI 是 data collection interface (gripper + camera)，跟 OPFA 解决的问题不同，但 UMI 在 paper 里作为 11 个 embodiment 之一被测试，说明 OPFA 能容纳这种特殊 hardware。

### 7.6 vs π0 (https://arxiv.org/abs/2410.24164)
π0 是 Physical Intelligence 的 VLA flow model，主要在 gripper 类 embodiment 上做 scaling。OPFA 的 geometry-aware 思路可以 complement π0，让 π0 也能扩展到灵巧手。

---

## 8. 跟其他领域的 conceptual 联系

### 8.1 跟 NLP tokenizer 的类比
OPFA 的 universal hand model $\mathcal{H}$ + select-mask 机制，本质上是 **continuous 版的 BPE vocabulary**。NLP 用离散 token vocabulary 跨语言共享 subword，OPFA 用 universal joint superset 跨 embodiment 共享 sub-action。

### 8.2 跟 SMPL/SMPL-X 的类比
SMPL-X (https://smpl-x.is.tue.mpg.de/) 是 human body + hand + face 的统一 parametric model，所有 human motion 都可以表示成 SMPL-X 参数。OPFA 的 universal hand model 是 robot hand 的类比，但更激进——它不只是统一同类 embodiment，而是统一 gripper + dexterous hand 这种结构完全不同的 embodiment。

### 8.3 跟 VAE 的类比
OPFA 的 encoder-decoder 结构很像 VAE，但有几个关键差异：
1. 输入是 point cloud（geometry），不是 image
2. 没有 KL regularization，纯 reconstruction loss
3. latent space 不是为生成，是为 cross-embodiment alignment

### 8.4 跟 Differentiable Simulation 的联系
$f_{FK}^m$ 是 forward kinematics，OPFA 在 Stage 1 把它当成 fixed operator。如果换成 differentiable FK + differentiable simulation，整个 framework 可以扩展到 RL setting，policy 在 latent space 学，gradient 通过 decoder 和 FK 回传到 encoder。这是一个很自然的 follow-up 方向。

### 8.5 跟 Contrastive Learning 的可能结合
当前 OPFA 用 reconstruction loss 训 GaLR。可以加 contrastive loss：同一个 grasp pose 在不同 embodiment 下的 point cloud 应该 map 到接近的 latent。这种 cross-embodiment contrastive 可能学到更 semantic 的 representation，类似 CLIP 在 image-text 上的做法。

---

## 9. 局限性和可能的改进

### 9.1 2-stage 训练的耦合问题
Stage 1 的 encoder-decoder 是 freeze 后插到 Stage 2。如果 Stage 1 学到的 latent 没有保留 task-relevant 的细节，Stage 2 无法纠正。一个改进方向是 **end-to-end joint training**：把 reconstruction loss 和 policy loss 一起优化，让 GaLR 既保留几何信息又对 task 有用。但这样需要 differentiable FK 和 careful loss balancing。

### 9.2 Point cloud 表示的损失
从 joint angle → FK → point cloud → latent → joint angle，这是一个 **bottleneck**。point cloud 是离散采样，可能丢失 fine-grained joint correlation。一个改进是用更密集的点云或者隐式表示（NeRF/SDF）替代。

### 9.3 缺少 contact 和 tactile 信息
Inspire Hand 有 tactile 版本，但 paper 里 GaLR 只用 kinematic point cloud，没用到 tactile signal。把 tactile 融入 GaLR 是个 obvious extension。

### 9.4 仅在 manipulation 验证
Framework 理论上可以扩展到 locomotion (不同 legged robot)，但 paper 没验证。把 OPFA 用到 cross-embodiment locomotion 是个 future work 方向，跟 UniPi、Cross-embodiment RL literature 接轨。

### 9.5 Universal hand model 的设计
当前 $\mathcal{H}$ 是 manually designed 的 superset。如果 embodiment 数量继续扩展（比如上百个），这个 set 设计会变复杂。一个改进是 **learned universal joint set**，从数据中自动发现 semantic joint categories。

---

## 10. 我的总评

OPFA 是一个 **结构清晰、insight 明确、实验扎实** 的工作。它最核心的贡献是把 cross-embodiment 这个问题转成了一个几何问题——**用 point cloud 作为 embodiment-agnostic 的中间表示**，然后用统一的 encoder + universal decoder 实现参数共享。

技术上有几个亮点值得记住：
1. **GaLR = 3D point cloud encoder + transformer + GAP**，简单但有效
2. **Semantic positional embedding** 是 cross-embodiment transformer 的 key trick
3. **Universal hand model + select-mask** 是替代 per-embodiment decoder 的优雅方案
4. **Two-stage 解耦** 让 GaLR 可以 scale 到任意多 embodiment 而不需要 task-specific 数据

从 research direction 看，OPFA 代表了 robot learning 中一种 **"通过几何抽象统一 embodiment"** 的范式。这条路线跟 "通过语言统一 task" (VLA, RT-2, π0) 和 "通过演示接口统一 hardware" (UMI) 是 complement 的，未来很可能融合：用 LLM 提供 task semantic，用 UMI 类接口收集 data，用 OPFA 类方法统一 embodiment，最后训一个真正 universal 的 robot foundation model。

参考链接汇总：
- Project page: https://mujc2021.github.io/opfa/
- DP3: https://arxiv.org/abs/2403.03954
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- KPConv: https://arxiv.org/abs/1904.08868
- Geometric Transformer: https://arxiv.org/abs/2202.05661
- Octo: https://arxiv.org/abs/2405.12213
- RDT-1B: https://arxiv.org/abs/2410.07864
- CrossDex: https://arxiv.org/abs/2410.02479
- UMI: https://arxiv.org/abs/2402.10329
- Open X-Embodiment: https://robotics-transform-x.github.io/
- π0: https://arxiv.org/abs/2410.24164
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- Latent Action Diffusion: https://arxiv.org/abs/2506.14608
