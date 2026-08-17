---
source_pdf: LESSMIMIC Long-Horizon Humanoid Interaction with Unified Distance Field
  Representations.pdf
paper_sha256: f38df8e3a2acc02a79da3c685bf404dea82d069cb2c7a7b458dd7fcb0a34f575
processed_at: '2026-08-05T14:27:10-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 LESSMIMIC

好，前面那版太学术了，我换种讲法。

---

## 这帮人到底解决了什么问题？

想象你教一个 humanoid robot 干活：推椅子、捡箱子、搬东西、坐下。现在的做法基本两条路：

**第一条路**：给 robot 看一段 human 的 MoCap 动作，让它跟着模仿。问题在于——这段动作是在某个特定大小的箱子上录的。你换个 23cm 的小箱子，或者 60cm 的大圆柱，robot 就懵了。因为它学到的是"在这个尺寸的箱子前，joint 角度应该是这些"，它没学到"我跟这个东西的几何关系应该长什么样"。

更麻烦的是，一旦 robot 稍微偏离 reference trajectory，系统就判定为 tracking failure。你想让它 adaptively 调整？不行，reference 说了算。

**第二条路**：干脆不要 reference，直接用 RL 从头学。灵活了，但每个 task 要单独 design reward、单独训 policy。push 一个 policy，pickup 另一个 policy，它们之间没法 seamless 拼起来做 long-horizon 任务。

LESSMIMIC 说：**这两条路都搞错了重点**。问题不在 policy 架构，不在 reward design，问题在 **representation**——你用什么语言告诉 policy "现在 interaction 进行到哪一步了"。

---

## 他们找到的 representation：Distance Field

Distance Field 听起来很 fancy，其实概念特别朴素。

想象空间里每个点都有一个数值，这个数值 = 它离最近 object surface 有多远。就这么简单。你站在 1 米外，值是 1；贴近 surface，值接近 0。

但关键的 magic 在 gradient。DF 的 gradient 指向 surface 的法线方向——也就是说，**在任何位置你都能知道"surface 朝哪边"**。这个信息太有价值了，因为它告诉你：我现在是 face 着 surface、还是侧着、还是背对着。

point cloud 给你一堆离散点，你不知道点之间空隙的几何意义；voxel 给你格子，丢了 gradient；neural implicit 有 gradient 但 query 太慢。DF 同时满足 continuous、differentiable、query 极快三个条件。

---

## 最核心的那个公式，用大白话讲

paper 里公式 (1) 是整篇文章的灵魂：

$$
\mathbf{v}_t^{\mathrm{norm}} = (\mathbf{v}_t \cdot \nabla \Phi(\mathbf{x}_t)) \nabla \Phi(\mathbf{x}_t), \quad \mathbf{v}_t^{\mathrm{tan}} = \mathbf{v}_t - \mathbf{v}_t^{\mathrm{norm}}
$$

翻译成人话：你的 hand 在以某个速度 $\mathbf{v}_t$ 运动。把这个速度拆成两部分——

**一部分是朝 surface 法线方向的**（$\mathbf{v}_t^{\mathrm{norm}}$）：你在 approach 这个 object，或者在 push 它、squeeze 它。这一部分的 magnitude 大，说明你正在使劲往 surface 上压。

**另一部分是沿 surface 切面的**（$\mathbf{v}_t^{\mathrm{tan}}$）：你在 surface 上滑动、traverse。这一部分告诉你正在沿表面 flow。

为什么这个拆解这么 powerful？因为**不管 object 是什么形状、什么大小，"我正在以 5cm/s 朝法线方向压 + 10cm/s 沿切面滑"这个描述的语义是一样的**。

换个大箱子？法线方向变了，但"我在 approach + slide"这件事没变。换个球？同理。representation 自动 shape-agnostic、scale-agnostic。

这就是为什么 LESSMIMIC 能 generalize 到从没见过的 object——它根本不关心 object 长什么样，它只关心"我跟 surface 的局部几何关系"。

---

## 训练分三步，每步解决一个 specific 问题

### 第一步：让 policy 大致学会怎么动

你不能直接从零开始 RL，sample efficiency 太差。所以他们先用一个 teacher policy（ResMimic，一个能 track reference motion 的 policy）生成 physically valid 的轨迹数据，然后用 behavior cloning 教 student policy。

**关键 trick**：student 看不到 reference motion。它只看三个东西——自己的 proprioception、一个 sparse 的 root trajectory command（"往那边走"这种粗略指令）、以及 DF encode 出来的 interaction latent $z_t$。

teacher 有 reference motion 这种 privileged information，student 没有。这是 deliberate 的信息不对称——逼 student 从一开始就学会靠 DF 信号做决策，而不是靠 reference 作弊。

用 DAgger 而不是普通 BC，因为普通 BC 有 covariate shift 问题：student 一旦偏离 teacher 的轨迹分布，就没见过那种 state，直接崩。DAgger 让 student 自己 rollout，teacher 在每个 state 上给 label，这样 student 学到的是"我自己会陷入的 state 怎么 recover"。

### 第二步：逼 policy 真正理解几何，而不是死记硬背

第一步训出来的 policy 有个问题：它在训练集的 object 上 work，但本质上是 memorize 了 specific 的 joint trajectory。你换个 geometry 它就懵——因为它没学到"为什么这样动是对的"。

第二步解决这个。做法很暴力也很 effective：**把 object 的 scale、shape、物理属性全部 randomize**，然后在 randomized environment 里 RL fine-tune。

但问题来了——reference motion 在新 geometry 上无效了（你没法用 1.0× box 的 reference 去指导 1.6× cylinder 的 interaction），task-specific reward 又会破坏 unified representation 的初衷。怎么办？

**他们发明了 AIP（Adversarial Interaction Prior）**。思路借自 AMP（Adversarial Motion Prior，Xue Bin Peng 那篇）——AMP 用一个 discriminator 判断"这个 motion 看起来像不像人"，作为 reward 信号。

AIP 做了一个关键 modification：**discriminator 的输入不是 full robot state，而是 DF interaction latent $z_t$**。

这是什么意思？discriminator 学到的不是"valid 的 joint configuration 长什么样"，而是"valid 的 interaction 几何 signature 长什么样"——approach、contact、release 这个过程的 DF pattern。

所以 policy 可以为了 match 一个新 geometry 的 object，synthesiz 出从没见过的 pose，只要这个 pose 产生的 $z_t$ signature 是 discriminator 认可的。这就是 generalization 的 mechanism——supervision 在 geometry 层面，不在 pose 层面。

reward 总共三部分：
- $r_{\text{task}}$：follow root trajectory command
- $r_{\text{interact}}$：AIP discriminator 给的分数（interaction signature 对不对）
- $r_{\text{style}}$：AMP discriminator 给的分数（motion 看起来自不自然）

两个 discriminator 分工：一个管几何、一个管姿态。

### 第三步：从 MoCap 迁移到 depth camera

前两步训出来的 policy 依赖 MoCap 提供 object 的全局信息——lab 里有 MoCap 设备能精确知道 object 在哪。real world 部署哪有 MoCap？

所以第三步做 distillation：训一个 vision-based student policy，输入换成 egocentric depth camera 的画面。visual encoder 学着从 depth image 推出 latent $z_t$——**相当于让 network 自己学会从 depth 估计 DF**。

用 DAgger-style distillation，teacher（前两步训好的 policy）frozen，student 自己 rollout、teacher 给 action label。配合大量 domain randomization（camera jitter、depth noise、physical property randomization）做 sim-to-real。

---

## 实验里最 striking 的几个结果

### 1. Scale generalization 碾压 baseline

训练只在 1.0× scale 上做。测试 0.4× 到 1.6×。

PickUp 任务在 0.4×（15cm 的小箱子）：
- HDMI（reference-based）：0%
- ResMimic（reference-based）：0%
- PhysHSI（reference-free）：23%
- **LESSMIMIC：63%**

在 1.6×（60cm 大圆柱）：
- HDMI：1.7%
- **LESSMIMIC：94%**

reference-based 方法在 extreme scale 直接崩，因为 reference motion 假设的尺寸跟现实差太远。LESSMIMIC 因为 condition 在 local DF 上，对它来说大小箱子的 $z_t$ signature 在 approach 阶段几乎一样。

### 2. Long-horizon composition 是 emergent 的

随机拼 N 个 task（push → pickup → carry → sit → ...），一个 policy 一口气执行，中间不 reset。

| N | LESSMIMIC | 所有 ablation |
|---|---|---|
| 5 | 61.7% | <25% |
| 10 | 38.1% | ~0% |
| 40 | 2.1% | 0% |

所有去掉单个 component 的 ablation 在 N≥10 全部 collapse 到 0%。只有 full model 能撑到 40。

为什么？因为没有 explicit task scheduler，policy 只看 $z_t$。当 $z_t$ 的 pattern 从 "push signature" 自然演化到 "pickup signature"，policy 自动 transition。这是 representation 设计带来的 emergent behavior，不是 engineered 出来的。

这种思路在 hierarchical RL 里很难做——Options framework、FeUdal Networks 都需要 explicit sub-policy boundary。LESSMIMIC 通过 unified observation 绕过了这个难题。

参考：[Options Framework](https://arxiv.org/abs/1606.05296), [FeUdal Networks](https://arxiv.org/abs/1703.01161)

### 3. Real world 真的 work

物理 humanoid 上跑：
- PickUp 22cm 箱子：MoCap 版 10/10，Vision 版 8/10
- PickUp 60cm 箱子：MoCap 版 8/10，Vision 版 7/10
- SitStand 12cm 椅子：8/10
- SitStand 46cm 椅子：10/10

vision 版本在 SitStand 上没评估，因为 depth camera 看不到背后 pelvis 接触——这是 egocentric observation 的固有 limitation。

---

## Ablation 告诉我们什么

去掉每个 component 看影响：

- **去掉 AIP**：PickUp 从 100% 掉到 23%。AIP 是 geometric generalization 的核心 mechanism。
- **去掉 geometry randomization**：scale 外直接 0%。randomization 是 generalization 的 data 基础。
- **去掉 RL fine-tune**：只有 BC 不够，BC 学不到 geometric rule。
- **去掉 Transformer 换 MLP**：PickUp 和 Carry 直接 0%。temporal dependency 太强，MLP 的 fixed receptive field 撑不住。
- **去掉 synthetic physicalization**（用 raw MoCap 替代 ResMimic 生成的数据）：contact-rich 任务退化严重。raw retargeted MoCap 违反 physics，student 学到不可行的 motion。

AIP + randomization 是两根支柱，缺一不可。AIP 提供 supervision signal 告诉 policy "什么是 valid interaction"，randomization 提供 data distribution 让 policy 见过各种 geometry。

---

## 这篇文章的 bigger picture

从更高视角看，LESSMIMIC 代表了 humanoid research 里一个重要的 representation shift：

**从 motion-centric 到 geometry-centric**。

reference motion 是 pose-level 的 representation——它编码"在时刻 t，joint i 应该在角度 θ"。这种 representation 天然 overfit to specific geometry。

DF 是 relation-level 的 representation——它编码"我跟 surface 的几何关系当前处于什么状态"。这种 representation 天然 generalize across geometry。

这个 shift 跟 computer vision 里从 template matching 到 feature learning 的演进很像，也跟 NLP 里从 rule-based 到 representation learning 的演进类似——**找对 representation，问题就解决了一半**。

跟现在几个 hot direction 的关系：
- VLA（Vision-Language-Action）模型如 [OpenVLA](https://openvla.github.io/)、[RT-2](https://robotics-transformer2.github.io/) 用 language 作为 task specification，但缺少 contact-rich interaction 所需的 fine-grained geometric signal。DF 可以作为 complement。
- World model 如 [DreamerV3](https://arxiv.org/abs/2301.04104) 学环境的 latent dynamics，DF 可以作为 world model 里 object representation 的 inductive bias。
- Humanoid foundation model 如 [BFM-Zero](https://humanoid-bfm.github.io/) 用 unsupervised RL 训 general control foundation，DF-based interaction representation 可以作为它的 task specification interface。

---

## 局限性，他们自己承认的

1. **只支持 rigid object**。articulated object（抽屉、门）和 deformable object（衣服、软体）的 DF 没法用 single rigid field 表达。需要 articulated SDF 或 deformable SDF。
2. **Partial observability**。SitStand 的 back-side contact 在 egocentric depth 下看不见。需要 multi-view camera 或 active perception。
3. **Vision 版 long-horizon 退化严重**。N=5 只有 15.9%，N≥15 全 0。depth → DF 的 implicit estimation 在长序列上 accumulate error。可能需要更 strong 的 geometric foundation model 像 [FoundationPose](https://nvlabs.github.io/FoundationPose/) 或 [DUSt3R](https://dust3r.europe.naverlabs.com/) 来提供更准的 online DF estimate。
4. **Dynamic object 的 DF**。object 在被 manipulate 时自己在动，DF anchored 在 object frame 意味着需要实时 track object pose。对快速运动 object 可能需要 velocity-aware DF。

---

## 一句话总结

LESSMIMIC 告诉我们：**humanoid interaction 的关键不是教它怎么动，而是给它一个能跨几何 generalize 的 representation 让它自己学会怎么动**。Distance Field 恰好是这个 representation——continuous、differentiable、shape-invariant、query 极快。配上 AIP 这种 geometry-level supervision 和三阶段 decoupled pipeline，一个 policy 就能 generalize across shape/scale、compose 40 个 task、还能从 failure 里 recover。

Project page: https://lessmimic.github.io

如果你想 build 更深的 intuition，我建议去看他们 project page 上的 video——看 DF signal 在 interaction 过程中怎么演化的，比看公式直观一百倍。Fig. 4 那张 sit-down 过程的 DF distance/gradient 曲线也值得仔细看，它就是把"坐下"这个动作翻译成了几何语言。

---

# LESSMIMIC: 用 Distance Field 统一 Humanoid Interaction 的 Representation

Andrej，这篇 paper 的核心 thesis 非常 clean：**interaction 的本质是 local geometric relationship，不是 absolute motion trajectory**。一旦你把这个 representation 找对了，generalization across shape/scale、long-horizon skill composition、failure recovery 这些原本被当作 competing objectives 的东西，自然从一个 policy 里 emergent 出来。下面我从 intuition、formulation、architecture、training pipeline、实验数据几个层面把它拆开。

---

## 1. 核心问题与 Insight

### 1.1 现有 humanoid interaction 的两大 camp 都跛脚

| Camp | 代表方法 | 牺牲了什么 |
|---|---|---|
| Reference-based | HDMI [51], ResMimic [62], CLONE [26], VisualMimic [56] | 把 object geometry 和 reference trajectory 绑死，policy 变成 "geometric specialist"；任何偏离 reference 的 deviation 都被 penalize 为 tracking failure，丢了 maneuverability |
| Reference-free | PhysHSI [50], Op3-Soccer [10] | 拿到 flexibility，但只能靠 task-specific reward engineering，policy 之间无法 compose |

LESSMIMIC 的 claim 是：**Distance Field (DF) 是同时满足 (i) reference-free inference、(ii) geometric generalization、(iii) long-horizon composition 的 unified representation**。

### 1.2 为什么是 DF？建立你的 intuition

考虑一个 hand 正在 grasp 一个 box vs. 一个 cylinder vs. 一个 soccer ball。在 global coordinate 下，这三者的"正确 joint configuration"完全不同；但在 hand 局部，"hand 距 surface 多远"、"surface 法线指向哪"、"hand 沿法线方向的 approach velocity 多大"、"hand 沿切面 sliding velocity 多大"——这些量在不同 object 之间是 **invariant** 的。这就是 DF 提供的 abstraction：

- $\Phi(\mathbf{x}_t)$：space 中任意点到最近 surface 的距离 → continuous, differentiable
- $\nabla \Phi(\mathbf{x}_t)$：gradient = surface normal（即使在 contact 中也 well-defined）
- Query cost 几乎为零，适合 high-frequency control loop（这点 SDF/neural implicit 做不到）

**对比其它 representation 的 trade-off**：

| Representation | 表达力 | 计算成本 | 是否有 gradient info |
|---|---|---|---|
| Voxel / Occupancy [16, 20] | explicit geometry | 高 memory/compute | ❌ 离散化丢失 gradient |
| Point cloud [3, 28, 63] | 详细 surface | 中 | ❌ 无连续距离 |
| Mesh [11, 36, 46] | 详细 surface | 中 | ❌ 无连续距离 |
| Neural implicit [4, 41, 47, 55] | expressive continuous | 高 inference latency | ✅ 但太慢 |
| **Unsigned Distance Field** | 连续可微 | 极低 query cost | ✅ analytical gradient |

注意：因为 humanoid-object interaction 中间没有 interpenetration（不像 grasping 内部），**unsigned** DF 的距离幅值 + 局部 gradient 已经是 sufficient geometric description，不需要 signed 的内部/外部区分——这是 paper 的一个务实简化。

---

## 2. DF-based Interaction Representation 公式精解

### 2.1 Velocity Decomposition（公式 1，核心中的核心）

设 link（比如 hand 或 pelvis）位置 $\mathbf{x}_t \in \mathbb{R}^3$，线速度 $\mathbf{v}_t \in \mathbb{R}^3$。$\Phi: \mathbb{R}^3 \to \mathbb{R}$ 是 unsigned distance field，$\nabla \Phi(\mathbf{x}_t)$ 是 $\mathbf{x}_t$ 在 surface 上的 projection 点处的 gradient（即 local surface normal，单位化）。

$$
\mathbf{v}_t^{\mathrm{norm}} = (\mathbf{v}_t \cdot \nabla \Phi(\mathbf{x}_t)) \nabla \Phi(\mathbf{x}_t), \quad \mathbf{v}_t^{\mathrm{tan}} = \mathbf{v}_t - \mathbf{v}_t^{\mathrm{norm}}
$$

**变量含义逐项解读**：

- $\mathbf{v}_t \cdot \nabla \Phi(\mathbf{x}_t)$：标量，是 velocity 在 surface normal 方向上的投影长度。物理意义 = approach rate（朝 surface 推进的速率），如果是负数表示在远离。
- $(\mathbf{v}_t \cdot \nabla \Phi) \nabla \Phi$：把这个标量重新乘回单位 normal，得到 **法向速度向量**。它捕获"interaction intensity relative to surface"——你在 squeeze、push、还是 release？
- $\mathbf{v}_t^{\mathrm{tan}} = \mathbf{v}_t - \mathbf{v}_t^{\mathrm{norm}}$：减去法向分量后剩下的就是切向分量，对应 **sliding / surface traversal**——你在 surface 上 flow 的方向与速率。

**Intuition**：这一步本质上把 global velocity 投影到一个 **object-surface-aligned local frame** 里。无论 object 是 23cm 的 box 还是 60cm 的 cylinder，"hand 沿 surface 法线推 5cm/s + 沿切面滑 10cm/s" 这个描述在两种几何下都意味着同样的 interaction 语义。这就是 shape/scale invariance 的来源。

类比一下：在 classic mechanics 里你做 polar/cylindrical coordinates 分解也是类似思路——把一个复杂运动投影到 problem-symmetry-aligned 的 basis 上，问题就 simplify 了。这里 DF gradient 充当的是 **data-driven, geometry-adaptive 的 local basis**。

### 2.2 Per-link Tuple 与 Temporal Window（公式 2）

$$
\mathbf{u}_t = [\Phi(\mathbf{x}_t), \nabla \Phi(\mathbf{x}_t), \mathbf{v}_t^{\mathrm{norm}}, \mathbf{v}_t^{\mathrm{tan}}], \quad I_t = \{\mathbf{u}_{t-l+1}, \ldots, \mathbf{u}_t\}
$$

- $\mathbf{u}_t$：单个 link 在时刻 $t$ 的 4-tuple 几何特征（distance scalar + gradient 3D vector + normal velocity 3D vector + tangential velocity 3D vector）
- $l$：temporal window length，收集最近 $l$ 步的 $\mathbf{u}$ 序列
- $I_t$：interaction representation，是一个 length-$l$ 的时序特征序列

**为什么需要 temporal window**？因为单帧的 $\mathbf{u}_t$ 只能告诉你"当前在哪、当前速度方向"，但 interaction 是一个动态过程——你正在 approach、还是 contact、还是 release，必须看 evolution。Fig. 4 很直观：sit down 过程中 mean DF distance 单调下降、gradient magnitude 上升，到了 Sit 阶段 plateau，再 Stand 时反向——这条曲线的 **shape** 就是 interaction phase 的 signature。

### 2.3 VAE Encoding 到 Latent $z_t$

$I_t$ 经过 VAE encoder 压成 compact latent $z_t$。两个目的：

1. **Sensor noise smoothing**：DF 测量来自 MoCap 或 depth camera，有噪声；VAE bottleneck 起到 denoising autoencoder 作用。
2. **Fixed-dim input**：不管 $l$ 多长，policy 输入维度恒定，简化架构。

这点让我想到 early vision 里 VAE 作为 latent state 的经典 trick（World Models, Ha & Schmidhuber 2018；PlaNet, Hafner et al. 2019）——都是用 VAE 把 high-dim observation 压成 policy 友好的 compact state。这里用在几何信号上，逻辑完全一致。

参考：[World Models](https://worldmodels.github.io), [PlaNet](https://planetrl.github.io)

---

## 3. 三阶段 Training Pipeline 架构解析

整个 pipeline 是个 teacher-student 结构 + adversarial post-training + visual distillation，每个 stage 解决一个 specific bottleneck。

### Stage 1: Interaction Skill Pre-Training（BC + DAgger）

**Teacher**: $\pi_{\mathrm{mimic}}$ = ResMimic [62]，一个 motion tracking + residual 的 policy。它能访问 privileged observation $O_{\mathrm{mimic}}$（包括 full-body reference motions），用来 generate physically valid 轨迹。注意 retargeted MoCap data 在 raw 形式下经常违反 physics 约束，所以需要 ResMimic 这一层 "physicalization"。

**Student**: $\pi_{\mathrm{base}}$ 只看 $O_{\mathrm{base}} = [o_{\mathrm{prop}}, c_t^{\mathrm{root}}, z_t]$，**完全没有 reference motion**。这里 $o_{\mathrm{prop}}$ 是 proprioception（joint DoF pos/vel），$c_t^{\mathrm{root}}$ 是 sparse root trajectory command（注意是 sparse！不是 full-body trajectory），$z_t$ 是 DF latent。

**Loss（公式 3）**：

$$
\mathcal{L}_{\mathrm{BC}} = \mathbb{E}_{s \sim \pi_{\mathrm{base}}}\left[\|\pi_{\mathrm{base}}(o_{\mathrm{base}}) - \pi_{\mathrm{mimic}}(o_{\mathrm{mimic}})\|_2^2\right]
$$

注意 expectation 是 $s \sim \pi_{\mathrm{base}}$，不是 $s \sim \pi_{\mathrm{mimic}}$——这是 **DAgger** [42] 的关键。学生自己 rollout，老师在每个 state 上给 corrective label。这避免了 covariate shift：如果只在老师分布上训练，学生一旦 slight 偏离就雪崩。DAgger 让学生在自己会陷入的 state 上学到 recovery。

参考：[DAgger paper](https://www.cs.cmu.edu/~sross1/publications/Ross-Gordon-Bagnell-dagger.pdf)

### Stage 2: Discriminative Post-Training（RL + AIP）

这一 stage 是 paper 最有 idea 的部分。问题：BC 训出来的 $\pi_{\mathrm{base}}$ 在 fixed object set 上 work，但它实际上 **memorize 了 specific kinematic trajectories**，并没有学到 geometric rule。怎么逼它学 rule？

**做法**：在 procedurally randomized environment 里 fine-tune，object scale/shape/物理属性全部 random。关键是不再用 motion-tracking reward（因为 reference motions 是 fixed geometry 下采集的，对新几何无效），改用 **Adversarial Interaction Prior (AIP)**。

**AIP Discriminator Loss（公式 4）**：

$$
\mathcal{L}_D = \mathbb{E}_{z \sim \mathcal{B}_{\mathrm{ref}}}[(D(z) - 1)^2] + \mathbb{E}_{z \sim \pi}[(D(z) + 1)^2]
$$

- $\mathcal{B}_{\mathrm{ref}}$：reference interaction buffer，存的是 Stage 1 中 teacher 在 fixed object 上产生的 "valid interaction" 的 $z_t$ 样本
- $z \sim \pi$：当前 policy 在 randomized object 上 rollout 产生的 $z_t$
- $D(z) \to 1$：判为 reference-like（valid interaction pattern）
- $D(z) \to -1$：判为 generated（可能 invalid）
- 用 least-squares GAN 而非 vanilla GAN cross-entropy，因为 LSGAN 的 gradient 更稳定（Mao et al. 2017）

**核心 insight**：discriminator 输入是 $z_t$（geometric interaction latent），**不是 full robot state**。这意味着 discriminator 学到的是 "valid 的 approach-contact-release 几何 signature 长什么样"，而不是 "valid 的 joint configuration 长什么样"。

这跟 AMP [39] 的区别至关重要：

| | AMP (Adversarial Motion Prior) | AIP (Adversarial Interaction Prior) |
|---|---|---|
| Discriminator input | full robot state $s_t$（joint pos/vel） | DF interaction latent $z_t$ |
| Regularizes | motion naturalness（步态、姿态） | interaction validity（geometric 接触模式） |
| 泛化性 | limited to demonstrated kinematic templates | 跨 shape/scale 都适用，因为 z 是 geometry-invariant |

**Composite Reward**：

$$
r_t = r_{\mathrm{task}} + \lambda_i r_{\mathrm{interact}} + \lambda_s r_{\mathrm{style}}
$$

各项具体形式（公式 5-7 + Tab. A2）：

$$
r_{\mathrm{task}}(\mathbf{x}_t, \mathbf{c}_t) = -\|\mathbf{x}_t^{\mathrm{root}} - \mathbf{c}_t^{\mathrm{root}}\|_2
$$

变量：$\mathbf{x}_t^{\mathrm{root}}$ 是 humanoid root 当前位置，$\mathbf{c}_t^{\mathrm{root}}$ 是 commanded root 目标位置。这个 reward **不需要 reference motion**，只要 follow 一个 sparse 的 root trajectory command——这是 reference-free 的关键。

$$
r_{\mathrm{interact}}(z_t) = \max(0, 1 - 0.25(D(z_t) - 1)^2)
$$

变量：$D(z_t)$ 是 discriminator 对当前 interaction latent 的打分。当 $D(z_t) \to 1$（reference-like），reward $\to 1$；偏离越远 reward 越低，clip 在 0。系数 0.25 是 scaling factor 控制奖励的 sensitivity。

$$
r_{\mathrm{style}}(s_t) = \max(0, 1 - 0.25(D_{\mathrm{AMP}}(s_t) - 1)^2)
$$

变量：$D_{\mathrm{AMP}}$ 是 standard AMP discriminator on full state $s_t$，确保 motion 看起来 natural（不要乱扭）。

**Intuition**：$r_{\mathrm{interact}}$ 管"interaction 的几何 signature 对不对"，$r_{\mathrm{style}}$ 管"motion 的 kinematic 看起来像不像人"。两个 discriminator 分工：一个 geometry-level、一个 pose-level。policy 可以为了 match 一个新 geometry 的 interaction signature 而合成 novel pose，只要这个 pose 在 motion style 上仍然 plausible。

完整 reward 表见 Tab. A2，还有 Action Reg（$\|\Delta a_t\|^2$，weight 5.0）、Termination（-10.0）、Joint Limit（-5.0）、Object Tracking（$\exp(-\|x_t^{\mathrm{obj}} - \tilde{x}_t^{\mathrm{obj}}\|^2/\sigma^2)$，weight 1.0）。

### Stage 3: Visual-Motor Policy Distillation

Stage 2 出来的 $\pi_{\mathrm{full}}$ 依赖 MoCap 提供 object 信息，real-world 部署困难。Stage 3 把它 distill 成 $\pi_{\mathrm{vis}}$，只用 egocentric depth。

**Student observation**：$o_{\mathrm{vis}} = [o_{\mathrm{prop}}, c_t, S_t]$，其中 $S_t$ 是 egocentric depth frame history。

**Visual encoder** $E_\phi$ 把 $S_t$ map 成 latent $z_t$，**实质上是在学习从 depth 恢复出 Stage 2 里 explicitly 提供的 DF geometric cues**。这是一个 implicit DF estimation 的过程——很有意思，相当于让 network 自己从 depth 学会"surface 距离 + normal 方向"。

**Distillation Loss（公式 8）**：

$$
\mathcal{L}_{\mathrm{distill}} = \mathbb{E}_{s \sim \pi_{\mathrm{vis}}}\left[\|\pi_{\mathrm{vis}}(o_{\mathrm{vis}}) - \pi_{\mathrm{full}}(o_{\mathrm{base}})\|_2^2\right]
$$

- $\pi_{\mathrm{full}}$ frozen 作为 teacher
- $\pi_{\mathrm{vis}}$ rollout 收集 trajectory（DAgger-style，避免 covariate shift）
- 每个遇到的 state 上 query teacher 拿 action label
- Domain randomization 很重：camera extrinsic jitter、depth quantization、dropout、additive sensor noise、physical property randomization

**Intuition**：这等价于让 visual encoder 学一个 "DF regressor from depth" 的 implicit function，但 supervised by end-to-end action matching 而非 explicit DF supervision。这种 "通过 distillation 隐式学几何先验" 的 pattern 在 robot learning 里反复出现（e.g., [Learning to See before Learning to Act](https://arxiv.org/abs/1910.07087), [Deep Visual Foresight](https://arxiv.org/abs/1610.01265)）。

---

## 4. 架构图深度解析（Fig. 3）

四个 subfigure 串起整个 pipeline：

**(a) DF Feature Extraction**：
- 输入：MoCap or egocentric depth → object geometry
- 对每个 task-relevant link（hand, pelvis 等）计算 per-link DF tuple $\mathbf{u}_t$
- Velocity decomposition + 时序累积成 $I_t$
- VAE encode 成 $z_t$

**(b) Pre-Training**：
- Teacher $\pi_{\mathrm{mimic}}$（ResMimic）拿 reference motion + residual 学会 physically valid interaction
- Student $\pi_{\mathrm{base}}$ BC + DAgger 学，observation 只用 $[o_{\mathrm{prop}}, c_t^{\mathrm{root}}, z_t]$

**(c) Post-Training**：
- 同一个 policy 架构
- 环境 procedural randomize（object scale/shape/physics）
- AIP discriminator 监督 $z_t$，AMP discriminator 监督 $s_t$
- RL fine-tune 出 $\pi_{\mathrm{full}}$

**(d) Visual Distillation**：
- $\pi_{\mathrm{full}}$ 作为 frozen teacher
- $E_\phi$ 学 depth → latent
- DAgger distill 出 $\pi_{\mathrm{vis}}$

**关键设计 choice**：三个 stage 用 **同一个 Transformer policy 架构**，只换 training objective / supervision signal / environment config。这让 weight 可以 seamless carry over，不需要 architecture surgery。

---

## 5. 实验数据深度解读

### 5.1 Object Scale Generalization（Tab. II）

训练在 scale 1.0×，测试 0.4×–1.6×。最 striking 的几个点：

**PickUp 0.4×（15cm³ box）**：
- HDMI: 0.0%
- ResMimic: 0.0%
- VisualMimic: 0.0%
- PhysHSI: 23.1%
- **LESSMIMIC (MoCap): 63.0%**
- **LESSMIMIC (Vision): 63.7%**

reference-based 方法在 extreme scale 完全崩，因为 reference motion 假设的是 1.0× box 的尺寸，小 box 要求的 reach/grasp pattern 完全不同。LESSMIMIC 因为 condition 在 local DF 上，对它来说"小 box"和"大 box"的 $z_t$ signature 在 approach 阶段几乎一样——只是 distance scalar scale 不同，gradient 方向 + velocity decomposition 不变。

**PickUp 1.6×（60cm diameter cylinder）**：
- HDMI: 1.7%
- ResMimic: 63.0%
- **LESSMIMIC: 94.0% / 93.0%**

注意 reference-based 的 ResMimic 在 1.4× 还能 99.7%，到 1.6× 掉到 63%——可见 reference motion 有一定 tolerance，但 extreme deviation 就崩。

**Push 1.0× contact rate**：
- LESSMIMIC (MoCap): 51.3% （并不是最高，HDMI 97.3%）
- 但在 1.4×：HDMI 掉到 10.6%，LESSMIMIC 仍 2.0%（虽然也不高）

这里要注意 Push 任务 LESSMIMIC 整体不如 PickUp/SitStand，可能因为 Push 是 sustained bimanual contact，对 friction 估计敏感，DF 提供的几何信号不足以完全 capture contact dynamics。

### 5.2 Ablation Study 解读（Tab. II 下半部分）

每个 ablation 去掉一个 component：

| Ablation | PickUp 1.0× | SitStand 1.0× | Carry | 解释 |
|---|---|---|---|---|
| Ours - AIP | 23.3% | 98.3% | 0.0% | AIP 是 geometric generalization 的主要 mechanism，去掉后 PickUp 大幅退化 |
| Ours - Syn. | 99.3% | 30.0% | 66.5% | 去掉 synthetic physicalization（用 raw MoCap 替代 ResMimic 生成数据），contact-rich 任务退化严重 |
| Ours - Rand. | 0.0% | 81.3% | 0.0% | 没有几何 randomization，scale 外完全 overfit |
| Ours - RL | 31.7% | 64.7% | 5.3% | 只有 BC 没有 RL fine-tune，BC 不足以 bridge 几何 gap |
| Ours - Trans. | 0.0% | 77.7% | 0.0% | MLP 替代 Transformer，PickUp/Carry 完全失败——temporal dependency 太强 |

**Build intuition**：AIP 和 geometry randomization 是 generalization 的两根支柱——AIP 提供 supervision signal 让 policy 知道"什么是 valid interaction signature"，randomization 提供 data distribution 让 policy 见过各种 geometry 组合。两者缺一不可。Transformer 是必要的，因为 multi-skill interaction 的 temporal context 跨度很大，MLP 的 fixed receptive field 不够。

### 5.3 Long-Horizon Skill Composition（Tab. III）

这是 paper 最 impressive 的结果。随机组合 N 个 heterogeneous task（push → pickup → carry → sit-stand → ...），单个 policy 一次执行，无 environment reset。

| Method | N=5 | N=10 | N=15 | N=25 | N=40 |
|---|---|---|---|---|---|
| Ours - AIP | 5.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| Ours - Syn. | 22.1% | 4.9% | 1.0% | 0.0% | 0.0% |
| Ours - Rand. | 1.9% | 0.0% | 0.0% | 0.0% | 0.0% |
| Ours - RL | 3.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| Ours - Trans. | 1.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Ours (MoCap)** | **61.7%** | **38.1%** | **23.5%** | **9.0%** | **2.1%** |
| Ours (Vision) | 15.9% | 2.5% | 0.0% | 0.0% | 0.0% |

注意所有 ablation 在 N≥10 都 collapse 到 0%，**只有 full model 能撑到 N=40 还有 2.1%**。这说明 long-horizon composition 不是某一个 component 的功劳，是整个 representation + pipeline 协同的 emergent property。

**为什么 unified DF representation 能做 long-horizon composition？** Intuition：因为没有 explicit task sequencing，policy 只看 $z_t$——当 $z_t$ 的 signature 从 "push pattern" 自然演化到 "pickup pattern"，policy 自动 transition。这就像一个 continuous state machine，state 之间的 transition 由 geometric context 驱动，而不是 external scheduler。这种 emergent composition 在 hierarchical RL 里很难做出来（e.g., Options framework, FuN 等），因为 sub-policy 切换需要 explicit boundary。这里通过 representation 设计绕过了这个问题。

参考：[Options Framework](https://arxiv.org/abs/1606.05296), [FeUdal Networks](https://arxiv.org/abs/1703.01161)

### 5.4 Real-World Deployment（Tab. IV）

物理 humanoid 上验证：
- PickUp 22cm³：MoCap 10/10，Vision 8/10
- PickUp 60cm³：MoCap 8/10，Vision 7/10
- SitStand 12cm：MoCap 8/10
- SitStand 46cm：MoCap 10/10

Root tracking accuracy（$R_{\mathrm{acc}}$）在 75-94% 之间，sim-to-real gap 主要体现在 tracking 精度下降，但 discrete success rate 仍然高。注意 SitStand Vision 没评估，因为 depth camera 看不到背后 pelvis 接触——这是一个 representation limitation，back-side contact 在 egocentric 视角下 unobservable。

---

## 6. 跟相关工作的 positioning

### 6.1 Distance Field 在 Robotics 的历史脉络

DF/SDF 的概念来自 Osher & Sethian 1988 [35] 的 level set methods，原本是 PDE 数值方法。在 graphics 里广泛用于 surface representation、collision detection。在 robotics 里的应用路径：

- **Motion Planning**：[Kinodynamic RRT with SDF](https://arxiv.org/abs/1404.3895), ContactSDF [55]
- **Grasping**：[Dex-Net](https://berkeleyautomation.github.io/dex-net/) 用 SDF 做 grasp quality 评估
- **Manipulation**：[Neural Collision Fields](https://arxiv.org/abs/2308.14156) 用 SDF 做 differentiable collision

LESSMIMIC 的 contribution 是把它用作 **policy observation**，而不仅仅是 planner 内部的几何工具。这是 representation 的"使用层级"提升。

### 6.2 跟 Humanoid Loco-manipulation 大 family 的关系

近两年 humanoid whole-body control 爆发，几个代表性方向：

- **OmniH2O [12]** / **HumanPlus [9]**：human teleoperation → humanoid shadowing，reference-based
- **BeyondMimic [27]**：用 guided diffusion 从 motion tracking 扩展到 versatile control
- **Exbody2 [18]**：expressive whole-body control
- **BFM-Zero [24]**：unsupervised RL 训 behavioral foundation model for humanoid
- **AMO [23]**：adaptive motion optimization for hyper-dexterous control
- **PhysHSI [50]**：reference-free，但是 task-specific
- **CLONE [26]**：closed-loop teleoperation for long-horizon
- **LeverB [53]**：vision-language instruction conditioned humanoid control

LESSMIMIC 的差异化：**reference-free at inference + task-unified observation + long-horizon composition** 三者同时满足。Tab. I 很清楚地 position 了这一点。

参考：[HumanPlus](https://humanplus.github.io/), [OmniH2O](https://omni-h2o.github.io/), [BeyondMimic](https://arxiv.org/abs/2508.08241), [Exbody2](https://arxiv.org/abs/2502.10348), [PhysHSI](https://arxiv.org/abs/2510.11072), [CLONE](https://arxiv.org/abs/2506.04288)

### 6.3 跟 Adversarial Prior Methods 的关系

- **AMP [39]**：adversarial motion prior，discriminator on full state
- **CALM [48]**：conditional adversarial latent models
- **Mimickit [37]**：general framework for motion imitation

AIP 是 AMP 的"interaction-level"变种——把 discriminator input 从 motion state 换成 interaction latent。这是个很 elegant 的 idea transfer。

---

## 7. Limitations 与 Future Direction

Paper 自己提到：

1. **Articulated / deformable objects**：当前 DF 假设 rigid object，对 articulated（抽屉、门）或 deformable（衣物、软体）需要 extended representation。可能的方向：[Neural SDF for Articulated Objects](https://arxiv.org/abs/2306.12916), [DefSDF](https://arxiv.org/abs/2205.07912)
2. **Partial observability**：SitStand 的 back-side contact 在 egocentric depth 下 unobservable。需要 multi-view or active perception。
3. **Vision variant long-horizon 退化严重**：N=5 只有 15.9%，N≥15 全 0。说明 depth-only 的 geometric recovery 还不够 robust 长序列。可能的改进：[FoundationPose](https://nvlabs.github.io/FoundationPose/), [DUSt3R](https://dust3r.europe.naverlabs.com/) 类几何基础模型提供更准的 DF estimate。

我自己的额外思考：

4. **DF 来源的 sim-to-real gap**：simulation 里 DF 是 analytic 的（知道 object mesh），real-world 上要么用 depth 估 SDF（噪声大），要么用 MoCap（限定 lab setup）。Stage 3 的 distillation 缓解但没彻底解决。可以考虑用 [neural implicit reconstruction](https://github.com/NVlabs/nvdiffrec) 实时建 DF map。
5. **Dynamic DF for moving objects**：当前 DF anchored 在 object，但 object 在被 manipulate 时本身在动。公式 (1) 里 $\mathbf{x}_t$ 是 link 在 world frame 的位置，$\Phi$ 在 object frame——这意味着 DF representation 隐含了 object pose tracking。对快速运动 object（比如抛接）可能需要 velocity-aware DF。
6. **Long-horizon 的 credit assignment**：N=40 success 2.1% 看似低，但考虑到没有任何 task boundary signal、无 reset、无 explicit memory，这已经是 striking 的 emergent behavior。引入 [Transformer-XL](https://arxiv.org/abs/1901.02860) style 的 recurrence 或者 external memory 可能进一步 push。

---

## 8. 给你的 Intuition 总结

把这个 paper 浓缩成几句话：

1. **Representation 决定 generalization ceiling**。reference motion 是 pose-level representation，generalize 不了 geometry；DF 是 local-geometry-level representation，天然 shape/scale invariant。
2. **Velocity decomposition into normal/tangential 是 micro-foundation**。它把 "global velocity" 翻译成 "interaction semantics"（approach vs slide），这是 contact-rich manipulation 的 language。
3. **AIP 把 supervision 从 pose-level 抬到 interaction-signature-level**。这让 policy 可以 synthesize novel pose for novel geometry，只要 interaction signature 仍然 valid。
4. **Long-horizon composition 是 emergent 的，不是 engineered 的**。unified DF observation 让 task transition 由 geometric context 自然驱动，不需要 scheduler。
5. **三阶段 pipeline 是 decoupling 的艺术**：BC 学初始化，RL 学 generalize，distillation 学 deploy。每阶段解决一个 bottleneck，不混淆。

这跟最近 VLA / world model / humanoid 几条主线都有交集——DF 这个 representation 其实可以视为一种 **task-agnostic geometric foundation**，未来跟 vision-language model 结合（[LeverB](https://arxiv.org/abs/2506.13751) 已经在做 V-L conditioned humanoid）可能产生更强大的组合。

Project page: https://lessmimic.github.io

希望能 build 起你的 intuition。如果你对其中某个 component 想深挖（比如 AIP 跟 diffusion policy 的关系、或者 DF 在不同 link set 上的 ablation），可以继续聊。
