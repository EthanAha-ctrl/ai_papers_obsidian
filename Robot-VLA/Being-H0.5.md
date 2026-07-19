---
source_pdf: Being-H0.5.pdf
paper_sha256: 6253888be0b50fd682984cc769d3d60c82de21aba8c798738674a5b7aacbad35
processed_at: '2026-07-18T16:17:35-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Being-H0.5 深度讲解

Andrej, 这篇 paper 真的很有意思。它本质上是在做一个 "polyglot" 的 VLA model，用 human hand motion 作为 "mother tongue" 来 bridge 30 种不同的 robot embodiments。让我把整个架构、数据、训练和部署都拆开讲清楚。

---

## 1. 核心思想的 motivation

这篇 paper 站在一个很有意思的类比之上：NLP 里面的 multilingual pre-training。在 NLP 里，不同语言共享 underlying grammar（syntax, logic），所以 high-resource language（如英文）的知识可以 transfer 到 low-resource language。他们把这个 idea 直接搬到 robotics：

- 不同 robot embodiments（Franka, Unitree G1, dexterous hand, parallel gripper...）本质上是不同的 "physical languages"
- Human hand motion 是一个 "universal lingua franca"，因为它 capture 了 causal interaction logic 和 contact physics，这些 across 所有 kinematic "dialects" 都 invariant
- 所以用大规模的 human video 数据来 pre-train，可以让 low-resource 的 complex robot（比如 dexterous hand）从 high-resource 的简单 platform（比如 parallel gripper）和 human demonstration 中 bootstrap 技能

这个 motivation 其实挺 deep 的。他们指出现在的 VLA 像 "monolingual speaker"，在特定硬件上很强，换个 morphology 就废了。原因有两个：

**Physical gap**: diffusion-based VLA 在 simple robot（parallel gripper）上 pre-train 的时候，学的是一个 low-dimensional smooth action manifold；而 dexterous hand 是 high-dimensional, non-linear, fragmented manifold。当 pre-trained model 遇到 unseen 复杂 state space，它的 vector field prediction 会有 accumulated error，导致 trajectory 从 valid motion manifold 上 "drift" 出去。

**Data scarcity**: 机器人数据远不如 NLP 的 trillion tokens。每个 platform 的 demonstration data 都很少。

参考文献：
- [Being-H0 (前作)](https://arxiv.org/abs/2507.15597)
- [π0](https://arxiv.org/abs/2410.24164)
- [π0.5](https://arxiv.org/abs/2504.16054)
- [GR00T-N1](https://arxiv.org/abs/2503.14734)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [Open X-Embodiment](https://robotics-transformer-x.github.io/)

---

## 2. UniHand-2.0 数据集：最大的 embodied pre-training recipe

UniHand-2.0 是这套工作的核心。规模上：

- **35,000+ hours** multimodal data
- **120B tokens**
- **400M samples**
- **30 distinct robotic embodiments**
- **比 UniHand-1.0 大 200×**

三个组成部分：

### 2.1 Human Demonstration Data (16K hours, 134M samples, 25.6B tokens)

来源：
- Ego4D（[link](https://ego4d-data.org/)）
- EPIC-KITCHENS（[link](https://epic-kitchens.github.io/)）
- Egocentric-10K
- 自己 in-house 的数据

处理 pipeline：
1. 用 [HaWoR](https://arxiv.org/abs/2411.19476) 估计 hand pose（MANO 参数）和 camera extrinsics
2. 用 [Gemini 2.5](https://arxiv.org/abs/2507.06261) 生成 dual-level annotation：每秒细粒度 instruction + 10 秒 chunk-level intent
3. 4-stage post-processing：
   - Language Augmentation：LLM paraphrase 防止 overfitting 到 template
   - Motion-Quality Filtering：基于 detection confidence 和 DBA error
   - Manipulation Relevance Filtering：排除纯 locomotion
   - Handedness Debiasing：左右镜像，消除 right-handed bias

三个任务族：
- Motion generation（主要任务：vision/language → action）
- Motion description（vision + motion → text，semantic grounding）
- Motion continuation（past observation + action history → future action chunk，temporal coherence）

### 2.2 Robot Manipulation Data (14K hours, 45.7B tokens, 1.5B frames)

来源整合：OpenX-Embodiment, AgiBot-World, SO100-Community, InternData-M1, RoboMIND, RoboCOIN, LET 等

Deduplication + downsample 到 30%。

**30 个 embodiment**，包括 single-arm（Franka, Kuka, UR5E...）、dual-arm（ALOHA, Cobot Magic, Piper...）、portable（D1, SO101）、half-humanoid（PND AdamU, Agibot-G1, Leju Kuavo 4...）、humanoid（Unitree G1, Tiankung）。

**Simulation data 严格限制在 26%**——这点很重要，避免 sim-to-real gap。Open X-Embodiment 只占 3.1%，AgiBot World 占 3.0%。

表 1 里有详细的统计。例如 Franka 2196.4 hours，Agibot-G1 2391.7 hours，Unitree G1edu-u3 135.7 hours，PND AdamU 200 hours。

### 2.3 Visual-Text Understanding Data (5K equivalent hours, 50.2B tokens)

这个很关键。如果只用 robot data，text:visual token ratio 是 1:1000，model 会退化成 visual-motor reflex system，丧失 reasoning 能力。三个 pillar：

1. **General VQA**: LLaVA-v1.5, LLaVA-OneVision, FineVision, LLaVA-Video — 防止 catastrophic forgetting
2. **2D Spatial Grounding & Affordance**: RefCOCO, RefSpatial, RoboPoint, ShareRobot, RoboRefit, RoboVQA, MolmoAct, A0-ManiSkill, PixMo-Points, AsV2 — 精确 spatial localization
3. **Task Planning & Reasoning**: ShareRobot, EO1.5M-QA — 长程任务分解

参考资料：
- [LLaVA](https://arxiv.org/abs/2304.08485)
- [RefCOCO](https://arxiv.org/abs/1703.03956)
- [RoboPoint](https://arxiv.org/abs/2406.10721)
- [ShareRobot](https://arxiv.org/abs/2502.11536)
- [MolmoAct](https://arxiv.org/abs/2508.07917)

---

## 3. UniCraftor 数据采集系统

这部分是工程上的 contribution，很务实。现有数据集缺的关键 modalities：

1. **Native depth acquisition**: 头戴 Intel RealSense D435，active IR stereo 提供 raw physical depth，比 learning-based depth estimation 在 occlusion、ego-motion、恶劣光线下都稳得多
2. **High-precision extrinsics**: 用 5 个桌面 AprilTag + PnP 算法（[AprilTag](https://april.eecs.umich.edu/software/apriltag)）计算 ground-truth camera pose。这是 classical hand-eye calibration 思路，比 learning-based extrinsic predictor 稳定很多
3. **Hardware-synchronized interaction events**: 脚踏板记录 contact/release 的精确 timestamp，解决 automated labeling 的时间精度问题

数据：43 tabletop tasks, 200+ hours。

Post-processing：
- AprilTag 区域用 [Grounded-SAM2](https://github.com/IDEA-Research/Grounded-Segment-Anything) + [DifEraser](https://arxiv.org/abs/2501.10018) inpaint
- HaWoR 估计的 hand motion 用 multi-view depth refine
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) 生成 task description，conditioned on pedal event

---

## 4. Being-H0.5 架构总览

Being-H0.5 用 [Mixture-of-Transformers (MoT)](https://arxiv.org/abs/2411.04996) 架构，灵感来自 [BAGEL](https://arxiv.org/abs/2505.14683)。核心是 disentangle 两件事：

1. **Multimodal Understanding Expert**: 处理 high-level perceptual input，做 long-horizon planning, spatial reasoning, 中间 subgoal 生成
2. **Action Generation Expert**: 把高层 plan 翻译成 precise kinematic execution

两个 expert **共享 self-attention**（在每个 transformer layer），但是有不同的 FFN/参数。这样 high-level semantic context 可以无缝 condition action generation，又不会互相 bottleneck。

Backbone 从 [InternVL-3.5](https://arxiv.org/abs/2508.18265) 初始化（2B）。他们特别强调 backbone choice 很关键——"the underlying visual features significantly dictate downstream VLA efficacy"。

Generation paradigm 是 hybrid：
- Text output: next-token prediction
- Discrete hand motion: masked token prediction
- Action: Rectified Flow（[Flow Matching](https://arxiv.org/abs/2210.02747), [Rectified Flow](https://arxiv.org/abs/2209.03003)）

为什么 action 用 flow matching？因为 action chunk 是 continuous multimodal distribution，autoregressive discretize 会损失精度，特别是 high-DoF 任务。Flow matching 可以生成 smooth, high-fidelity 的 multimodal action distribution。

---

## 5. Unified State-Action Space 详解

这是 paper 最核心的 idea 之一。他们定义 state $s \in \mathbb{R}^d$ 和 action $a \in \mathbb{R}^d$ 都是 fixed-length 高维 vector，所有 embodiment 共享同一个 space。

每个 embodiment $e$ 通过 sparse slot assignment 映射进去：

$$s = \Phi_e(s^{(e)}), \quad a = \Phi_e(a^{(e)}) \quad (1)$$

变量含义：
- $s^{(e)}, a^{(e)}$：embodiment $e$ 的 raw state/action（具体物理量）
- $\Phi_e$：mapping function，把 raw signal route 到 global vector 的对应 slot
- 没用到的 slot 保持 zero

**关键创新**：human hand motion 被当作一个 generalized embodiment。MANO 参数直接 map 进 unified space：
- Human wrist 的 global pose → EEF subspace（和 robot EEF 对齐）
- Finger articulation → "fine-manipulation" slots

这意味着 human 和 robot 可以在同一个 action space 里 co-exist，share 同一个 "action language"。

物理量 standardization：
- Cartesian control: 相对 delta displacement，统一 world coordinate frame
- Rotation: Axis-Angle notation（避免 gimbal lock，SE(3) 上平滑插值）
- Joint space: 绝对 radian 值
- **No statistical normalization**！他们特意不 normalize 到 [-1, 1]，因为 "1 radian 或 10 cm 的运动有 intrinsic physical implications"。只做 outlier filtering，强迫 model 学真正的 physical scale

这个设计很重要——model 学的是真实的物理量，而不是 normalize 过的 abstract number。这样 transfer 到新 embodiment 才有意义。

为什么不用 per-embodiment MLP head（像 GR00T-N1 那样）？因为：
1. Fragment 学习 physical commonality，浪费 capacity
2. 不同 gripper 在 EEF 轨迹上 geometric consistency 很高，独立 head 阻止 sharing
3. Mixed rotation representation（Euler vs quaternion）浪费 computation

---

## 6. Mixture of Flow (MoF)

这是 architecture 上最重要的创新。Motivation：

- Unified action space 解决了 representation conflict，但 action expert 的 capacity 仍然 bottleneck
- Action expert 参数通常远少于 flow-based expert for visual generation（参考 BAGEL, Seedream）
- Heterogeneous morphology 又要求 specialization

两层 hierarchy：

### 6.1 Foundation Experts (Shared Dynamics)
Action expert 的 initial layers 是 standard transformer block，所有 input 共享。这些 layer 编码 transferable motor primitive（reaching, grasping dynamics, collision avoidance），across embodiment/task invariant。

### 6.2 Specialized Experts (Embodiment & Task Routing)
Upper layers 用 parallel specialized expert，learnable gating network 管理，灵感来自 [Mixture of Experts](https://arxiv.org/abs/1701.06538)。给定 input state + instruction，router 动态 activate Top-K expert subset。

效果：
- 训练时特定 task 的 gradient 只 update 相关 expert pathway，其他 skill weight 保留
- Total parameter 和 active parameter decouple——大模型库，但 inference 时只用 fraction
- 可以 deploy 到 edge hardware（NVIDIA Orin-NX）

**这个设计背后的直觉**：human motor control 本身就是 modular 的，general motor primitive 被动态 adapt 到 specific task。MoF 就是把这个 principle 显式化。

---

## 7. Pre-Training: Human-Centric Robot Learning

### 7.1 Unified Sequence Modeling

所有 supervision 都 cast 成 unified multimodal sequence modeling。把 state-action space 当作 explicit modality。每个 training sample 序列化成 token stream：

$$S = [x_1, x_2, \ldots, x_K] \quad (2)$$

每个 segment $x_k = \langle m_k, C_k \rangle$，$m_k \in \mathcal{M}$ 是 modality tag，$C_k$ 是 content。

Modality set：
$$\mathcal{M} = \{\text{vision, text, state, action}\} \quad (3)$$

训练用 **Physical Instruction Tuning**（from Being-H0），organize 成 QA format $[S_Q; S_A]$，conditioned on context $S_Q$，loss 只在 response $S_A$ 上算。

### 7.2 Human-Centric Multi-Task Objective

Joint loss：

$$\mathcal{L} = \lambda_{\text{text}}\mathcal{L}_{\text{text}} + \lambda_{\text{act}}\mathcal{L}_{\text{act}} \quad (4)$$

Text loss（VQA, motion description）：

$$\mathcal{L}_{\text{text}} = -\sum_{i \in \Omega_{\text{text}}} \log p_\theta(y_i | S_{<i}) \quad (5)$$

$\Omega_{\text{text}}$ 是 token-level index set，标识哪些 token 算 text loss。

### 7.3 Hybrid Human Motion Representation（非常关键）

Human motion 既 expressive 又 noisy。Pure continuous 监督 brittle，pure discrete 牺牲精度。所以他们用 **dual representation** 在同一个 training instance 里：

1. **Continuous action chunk**: $A = [a_1, \ldots, a_T] \in \mathbb{R}^{T \times d}$，每个 $a_t \in \mathbb{R}^d$
2. **Discrete motion tokens**: 用 pretrained tokenizer 把 motion chunk 量化成 $z \in \{1, \ldots, |\mathbb{C}|\}^{T_z}$
   - $\mathbb{C}$: codebook, $|\mathbb{C}|$ 是 codebook size
   - $T_z$: 量化后的 token length（通常比 $T$ 短，因为 compression）

Joint objective：

$$\mathcal{L}_{\text{act}} = \lambda_1 \mathcal{L}_{\text{FM}} + \lambda_2 \mathcal{L}_{\text{MASK}} \quad (6)$$

#### Continuous Flow-Matching

学一个 time-conditioned velocity field $v_\theta(x, t, c)$，其中 $c$ 是 conditioning（vision + text context）。这个 field 把 sample 从 standard Gaussian $x_0 \sim \mathcal{N}(0, I)$ transport 到 target data distribution。

对 target action $a_i$，定义 linear interpolation probability path：
$$x_t = (1-t)x_0 + t a_i, \quad t \in [0, 1]$$

理想 vector field：
$$u_t(x_t) = a_i - x_0$$

Loss：
$$\mathcal{L}_{\text{FM}} = \sum_{i \in \Omega_{\text{FM}}} \|v_\theta(x_t, t, c) - (a_i - x_0)\|_2^2 \quad (7)$$

变量含义：
- $x_t$: probability path 上的中间状态，从 noise $x_0$ 到 target $a_i$
- $t \in [0, 1]$: flow matching time
- $c$: conditioning context
- $\Omega_{\text{FM}}$: continuous action step 的 index set

#### Masked Motion Token Prediction

在 model backbone 上加 dedicated token embedding + linear prediction head。对 token sequence $z$ 随机 mask 掉比例 $\rho$ 的 index $\Omega_{\text{MASK}} \subseteq \{1, \ldots, T_z\}$，replace 成 specialized `[MASK]` token。Cross-entropy loss：

$$\mathcal{L}_{\text{MASK}} = -\sum_{i \in \Omega_{\text{MASK}}} \log p_\theta(z_i | c) \quad (8)$$

这个 discrete channel 的作用：作为 "language-like abstraction"，filter 高频执行 noise，stabilize motion prior across heterogeneous dataset。Continuous head 负责精度，discrete head 负责 "grammar"。

#### Joint Serialization（attention mask 设计）

序列结构：$[S_Q; S_A^{\text{FM}}; S_A^{\text{MASK}}]$

防止 information leakage（continuous 和 discrete 互相 copy），用 modified attention mask 的 gating matrix：

$$G = \begin{bmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix} \quad \text{for } [S_Q; S_A^{\text{FM}}; S_A^{\text{MASK}}] \quad (9)$$

行表示 query，列表示 key。$S_Q$ 只能 attend 自己；$S_A^{\text{FM}}$ 可以 attend $S_Q$ 和自己；$S_A^{\text{MASK}}$ 可以 attend $S_Q$ 和自己，**两个 answer segment 互不可见**。

Position encoding 也要 align，让两个 answer segment 都从同一 contextual origin 开始：

$$\text{PE}(j) = \begin{cases} j, & j \in S_Q \\ p_0 + r(j), & j \in S_A^{\text{FM}} \\ p_0 + r(j), & j \in S_A^{\text{MASK}} \end{cases} \quad (10)$$

$r(j)$ 是 segment 内的 relative index，$p_0 = \max_{j \in S_Q}(j+1)$ 是 $S_Q$ 之后的起始 position。

**直觉**：让 model 把 continuous 和 discrete 看成同一 temporal event 的两个 complementary view，从同一 origin grounded。

---

## 8. Post-Training: Cross-Embodiment Adaptation

Pre-training 之后还要 fine-tune 到 target robot。挑战：plasticity-stability dilemma——fine-tune 容易 collapse 成 specialist，丢失 cross-embodiment transferability。

三个 brittleness source：
1. Morphological interference（embodiment-dependent action field 在 shared parameter 下竞争）
2. Context unreliability under distribution shift（degraded feature 导致 action jitter）
3. Real-time temporal mismatch between inference 和 execution

### 8.1 Embodiment-Specific Adaptation (ESA)

Flow matching 的 objective（从 pre-training 角度看）：

$$\min_\theta \mathbb{E}_e \mathbb{E}_{(a_0, H) \sim \mathcal{D}_e} \mathbb{E}_t \|v_\theta(a_t; H, e) - v^*(a_t, a_0)\|_2^2 \quad (11)$$

变量：
- $e$: embodiment
- $a_0$: target action
- $H$: observation history/context
- $\mathcal{D}_e$: embodiment $e$ 的 data distribution
- $v^*(a_t, a_0)$: ideal target velocity field
- $v_\theta(a_t; H, e)$: model 预测的 velocity field

当不同 embodiment 的 optimal field 差异大，embodiment-specific gradient $\nabla_\theta \mathcal{L}_e$ 会 misalign，导致 slow convergence + unstable averaged behavior。

**ESA 的设计**：不做 per-embodiment projection head，而是利用 unified action space。不同 embodiment activate 不同 index set $\mathcal{T}_e \subseteq \mathcal{S}$，只 update active index 对应的参数。

Slot-wise adapter bank：$\mathbf{W}_{\text{ESA}} \in \mathbb{R}^{K \times d_{\text{out}} \times d_{\text{in}}}$

只在 embodiment $e$ 的 active slot 上 update：

$$\mathbf{W}_{\text{ESA}}^{(e)} \triangleq \{\mathbf{W}_{\text{ESA}}[k] : k \in \mathcal{T}_e\}, \quad \Delta \mathbf{W}_{\text{ESA}}[k] = 0 \quad \forall k \notin \mathcal{T}_e \quad (12)$$

变量：
- $K$: 总 slot 数
- $\mathcal{S} = \{1, \ldots, K\}$: 全部 semantic action slot（arm joint, hand/gripper joint, base control...）
- $\mathcal{T}_e$: embodiment $e$ active 的 slot index
- $d_{\text{in}}, d_{\text{out}}$: adapter 的输入输出维度

**关键 insight**：share 硬件 component 的 embodiment（比如同一个 arm 配不同 hand）有 overlapping index set，所以 share 一部分 adapter 参数。这样 knowledge transfer 是结构化的，避免了 "全部 isolate" 或 "全部 share" 的两个极端。

### 8.2 Manifold-Preserving Gating (MPG)

#### Motivation

Flow matching 用 token-level context feature $H$（vision-language prefix + proprioceptive state embedding + current action-token embedding）作为 condition。假设是 $H$ faithful captures task-relevant semantics。

Distribution shift 下（lighting change, viewpoint perturbation, partial occlusion），policy 会从 degraded feature 盲目 regress action，产生 unstable behavior。Cross-embodiment 情况更糟：high-DoF embodiment 有更多 unobserved state + stronger constraint，$H \to \text{feasible action}$ 的 mapping 更 sensitive。

Denoising update：$\mathbf{a}_{t-\Delta t} = \mathbf{a}_t + \Delta t \cdot v_\theta(\mathbf{a}_t; H)$

如果 $H = H^* + \epsilon$，action variance 大约是 $\|\partial v_\theta / \partial H\|^2 \text{Var}(\epsilon)$，表现为 action jitter。

#### Gated Residual with Ungated Prior Offset

MPG 计算 gate $g \in (0, 1]$（discrepancy 大 ⇒ $g$ 小），只 modulate feature-conditioned residual pathway：

$$\tilde{H} = H + \lambda \cdot \mathcal{P}_{\text{MPG}}(g \cdot \mathcal{E}_{\text{obs}}(H)) = H + \lambda g \mathbf{W}_{\text{MPG}} \mathcal{E}_{\text{obs}}(H) + \lambda \mathbf{b}_{\text{MPG}} \quad (13)$$

变量：
- $\mathcal{E}_{\text{obs}}$: 把 $H$ project 到 common embedding space
- $\mathcal{P}_{\text{MPG}}(\cdot) = \mathbf{W}_{\text{MPG}}(\cdot) + \mathbf{b}_{\text{MPG}}$: 生成 enhancement residual 的 affine
- $g$: reliability gate
- $\lambda$: scaling

**直觉**：
- $g \approx 1$：trust context，feature-conditioned adaptation 强
- $g \approx 0$：suppress feature-dependent term，fallback 到 stable learned prior offset $H + \lambda \mathbf{b}_{\text{MPG}}$

**为什么 $\mathbf{b}_{\text{MPG}}$ 不 gate**：让它在 low-confidence regime 依然 influence $\tilde{H}$，effectively 学一个 robust default correction for uncertain context。

#### 对比 DiG-Flow

DiG-Flow（[paper link](https://arxiv.org/abs/2512.01715)）做 output gating：$\tilde{H} = H + \lambda g \mathcal{R}(H)$，gate 同时 scale projected term 和 bias。Gate fluctuation 时 variance 是 $\|\mathbf{W}\mathcal{E}_{\text{obs}}(H) + \mathbf{b}\|_2^2$。

MPG 是 input gating：gate 在 projection 之前。Gate fluctuation 时 variance 只 $\|\mathbf{W}\mathcal{E}_{\text{obs}}(H)\|_2^2$，因为 ungated offset 不依赖 $g$。所以 MPG 在 contextual perturbation 下 trajectory 更平滑。

#### Discrepancy 和 Gate Computation

构造 action prior anchor from noise-free action。$Z^{\text{nf}}$ 是 zero noise level 的 action-token embedding，$\bar{Z} = \text{MeanPool}(Z^{\text{nf}})$。

用 Sliced Wasserstein Distance（[SWD](https://arxiv.org/abs/1907.00081)）度量 feature-action distributional discrepancy（scale-invariant space）：

$$D(\mu_{\hat{H}}, \mu_{\hat{Z}}) \approx \frac{1}{M} \sum_{m=1}^M \|\text{sort}(\theta_m^\top \hat{H}) - \text{sort}(\theta_m^\top \hat{Z})\|_2^2 \quad (14)$$

变量：
- $\theta_m$: 第 $m$ 个 random unit direction（投影方向）
- $M$: 总投影数
- $\hat{H} = \text{LN}(\mathcal{E}_{\text{obs}}(H))$: LayerNorm 后的 observation embedding
- $\hat{Z} = \text{LN}(\mathcal{E}_{\text{act}}(\bar{Z}))$: LayerNorm 后的 action anchor
- $\bar{Z}$: 在 observation sequence length 上 broadcast

为什么用 SWD？因为它是 scale-invariant，并且对 high-dimensional distribution 的距离度量比 KL 之类更稳。

Gate：
$$g = \exp(-D/\tau) \in (0, 1] \quad (15)$$

$\tau$ 是 temperature。$D$ 越大，$g$ 越小，越 fallback 到 prior。

实践上对 gate 加 stop-gradient：$g^{\text{sg}} = \text{stopgrad}(g)$，防止 model 学操纵 $g$ 而非改进 feature 质量。这样 $g$ 行为是 enhancement pathway 上的纯 scaling factor。

Jacobian（忽略 $g$ 的 backprop）：

$$\frac{\partial \tilde{H}}{\partial H} \approx \mathbf{I} + \lambda g \mathbf{W}_{\text{MPG}} \frac{\partial \mathcal{E}_{\text{obs}}(H)}{\partial H} \quad (16)$$

context unreliable（$g$ 小）时，feature-dependent correction 越发 insensitive。

### 8.3 Universal Async Chunking (UAC)

#### Motivation

部署 action-chunking policy 到 physical hardware 时，inference latency 和 execution 是异步的。Model 计算 next chunk 时 robot 还在执行之前 committed trajectory。

[RTC (Real-Time Chunking)](https://arxiv.org/abs/2512.05964) 在 training 时模拟 inference delay：prefix action 用 clean timestep ($t=1$)，postfix 用 stochastic noisy timestep ($t<1$)，loss 只算在 postfix。但 RTC 通常 single platform。

**UAC 是 embodiment-aware RTC**：不同 embodiment 有不同 control period $\Delta t^{(e)}$ 和 inference latency budget $L^{(e)}$，effective delay in control step 是 $\lceil L^{(e)} / \Delta t^{(e)} \rceil$。

#### Embodiment-Aware Delay Sampling

$$d \sim \pi^{(e)}(d), \quad d \in \{0, 1, \ldots, d_{\max}^{(e)} - 1\} \quad (17)$$

变量：
- $\pi^{(e)}(d)$: embodiment $e$ 的 delay distribution
- $d_{\max}^{(e)}$: 最大 delay threshold（per embodiment）

Per-token timestep：
$$t_i = \mathbb{1}[i < d] + \mathbb{1}[i \geq d] \cdot t_{\text{base}}, \quad t_{\text{base}} \sim p(t) \quad (18)$$

prefix ($i < d$) 用 $t=1$（clean），postfix ($i \geq d$) 用 stochastic $t_{\text{base}}$。

Loss 只在 postfix：
$$\mathcal{L}_{\text{UAC}} = \sum_{i \geq d} \|\hat{v}_i - v_i^*\|_2^2 \quad (19)$$

---

## 9. Real-Time Deployment Infrastructure

### 9.1 Manifold-Preserving Refinement

Rectified flow denoising 从 Gaussian noise $a^{(0)} \sim \mathcal{N}(0, I)$ 开始，Euler step：

$$\mathbf{a}^{(k+1)} = \mathbf{a}^{(k)} + \Delta t \cdot v_\theta(\mathbf{a}^{(k)}, t^{(k)} | H), \quad \Delta t = 1/K, \quad t^{(k)} = k/K \quad (20)$$

变量：
- $K$: denoising step 数（实践 $K < 10$）
- $\Delta t = 1/K$: 每个 Euler step 的 size
- $t^{(k)} = k/K$: 当前 flow matching time
- $H$: token-level context feature

$H$ 由 static vision-language prefix 和 dynamic suffix（state token + current action-token embedding）组成。Denoising 时只有 action-token embedding 变，所以 vision-language prefix 可以 KV-cache。

**Two-stage refinement**：
1. Stage 1：baseline（无 MPG），得 $a_{\text{pred}}^{(0)}$
2. Stage 2：$N_{\text{ref}}$ round refinement（实践 1-3 round），用上一轮 prediction 作为 anchor：

$$\bar{Z}^{(n-1)} = \text{MeanPool}(\text{Enc}(a_{\text{pred}}^{(n-1)}, \sigma=0)) \quad (21)$$
$$a_{\text{pred}}^{(n)} = \text{FlowDenoise}(H; \bar{Z}^{(n-1)})$$

$\sigma$ 是 action noise level，$\sigma=0$ 是 noise-free。

直觉：better action → better anchor → more accurate gate → more robust feature，positive feedback loop。

**为什么 MPG 在 low-step regime 特别有用**：$K<10$ 时每步要做大量 "transport"，noise 到 feasible action mode 的距离大。$g$ 小时 fallback 到 $\tilde{H} \approx H + \lambda \mathbf{b}_{\text{MPG}}$，稳定 velocity field，减少要 traverse 的距离。实践上 2-3 round refinement 就 saturate。

### 9.2 Universal Async Chunking Protocol

部署 protocol：

**Latency commitment**：

$$d \geq \lceil t_{\text{inference}} / t_{\text{control}}^{(e)} \rceil + \epsilon_{\text{safety}} \quad (22)$$

变量：
- $t_{\text{control}}^{(e)}$: embodiment-specific control period
- $t_{\text{inference}}$: 推理延迟（model + hardware dependent）
- $\epsilon_{\text{safety}}$: 平台特定 tolerance

**关键 property**：overestimation 安全（只是 prefix 长），underestimation 会 break continuity（robot 执行 model 没 condition 的 action）。

**Hard prefix locking**（denoising 时约束）：

$$a_i^{(k)} \gets \begin{cases} \mathcal{B}_i, & i < d \\ a_i^{(k)}, & i \geq d \end{cases} \quad \forall k \in \{0, \ldots, K\} \quad (23)$$

$\mathcal{B}$ 是 execution buffer（最近 committed action）。Flow 不能 "edit" 正在执行的 action。

**Hard stitching**（buffer update）：

$$a_{\text{exec}}^{(n)} = a_{<d}^{(n-1)} \oplus a_{\geq d}^{(n)} \quad (24)$$

丢 prefix $a_{<d}$，只 write postfix $a_{\geq d}$。这样 latency commitment 满足时 continuity guaranteed。

### 9.3 Dual-Thread Deployment Architecture

两个 thread 通过 ring buffer $\mathcal{B}$ 通信：

**Control thread（consumer）**：fixed frequency $1/t_{\text{control}}^{(e)}$，pop action 推给 robot。Buffer 空时 hold last action 或执行 platform-specific safe fallback。

**Inference thread（producer）**：异步 fetch observation，做 rectified-flow denoising（可选 MPG refinement），只 write postfix action 到 $\mathcal{B}$ 的 offset $d$。

Ring buffer size 至少 2× chunk length，mutex 保护。好处：
1. **Latency absorption**：inference jitter 不 translate 到 control jitter
2. **Graceful degradation**：buffer underflow 触发 safe fallback，避免 undefined behavior
3. **Cross-embodiment compatibility**：调 $(d, \mathcal{B})$ 参数就支持不同 control frequency 和 compute budget 的 robot

---

## 10. Experiments 详解

### 10.1 Real Robot Comparison

5 个 embodiment（[table 2](#)）：

| Group | Name | DoF | Hand |
|---|---|---|---|
| Upper-body Humanoid | PND Adam-U | 31 | Dexterous (6DoF) |
| Upper-body Humanoid | Unitree G1 + LinkerBot O6 | 26 | Dexterous (6DoF) |
| Single arm + Dex hand | FR3 + Inspire Hand | 13 | Dexterous (6DoF) |
| Single arm + Dex hand | BeingBeyond D1 | 14 | Dexterous (6DoF) |
| Single arm + Gripper | LeRobot SO-101 | 6 | Gripper |

10 个 task 分 4 类：Spatial（Arrange Flowers, Water Plant, Stack Bowls, Stack Blocks）、Long-horizon（Drawer, Package Flip-Scan）、Bimanual（Hand-over Box, Pack Two Products）、Generalization（Wipe Whiteboard, Clear Table）。

Baseline：Being-H0.5-specialist, Being-H0.5-generalist, Being-H0.5-scratch（无 UniHand-2.0 pretraining）, [π0.5](https://arxiv.org/abs/2504.16054)。

**Blind evaluation protocol**：所有 policy 封装到 unified black-box inference server，operator 不知道在测哪个 model，N 个 pre-defined layout 随机 sample，policy identity 隐藏，binary success/failure by 客观 criterion。每 policy 每 configuration K=20 trials。

**Key findings**：
1. Specialist 最强，但 generalist 很接近。Overlap-heavy setting（approach-grasp-place, lid/handle interaction, planar wiping/clearing）下 generalist 甚至更好——joint training 增加共享 sub-skill exposure，robustness 提升
2. **大幅超过 π0.5**，特别是 long-horizon 和 bimanual（小 perception/control mismatch 在 multi-step 中 compound）
3. UniHand-2.0 pretraining 对 generalist 关键，scratch generalist 显著退化
4. **Emergent embodiment-level zero-shot transfer**！single generalist checkpoint 部署到 Adam-U 上，可以解决从未在 Adam-U 上 collect 过 demonstration 的 task（如 flip-and-scan, drawer interaction, stacking）。Success rate 低，但 task-consistent multi-step execution。这第一次实证 VLA 可以做 embodiment-level zero-shot task completion

### 10.2 LIBERO 结果（[Table 4](#)）

Being-H0.5-specialist: **98.9% average**（L-Spatial 99.2, L-Object 99.6, L-Goal 99.4, L-Long 97.4）
Being-H0.5-generalist: **97.6%**（trained jointly on LIBERO+RoboCasa 2× steps）

对比：
- π0.5: 96.9%
- EO1: 98.2%
- X-VLA: 98.1%
- π0: 94.4%
- GR00T-N1: 93.9%

只用 224×224 RGB，2B backbone。

### 10.3 RoboCasa 结果（[Table 5](#)）

24 个 long-horizon household task，Human-50 few-shot setting。

Being-H0.5-specialist: **53.9%** total（Pick & Place 36, Doors/Drawers 71.7, Others 57.6）
Being-H0.5-generalist: **53.3%**

对比：
- π0: 42.4%
- π0.5: 41.4%
- GR00T-N1: 36.0%
- GWM（3D 多模态）: 39.3%

只用 RGB 224×224，超过用 3D input 的方法。Pick & Place 提升最大（40% vs π0.5 的 21.5%），说明 human-centric pretraining 给了 transferable spatial prior。

### 10.4 Ablation: Human-Centric Learning 对 Downstream 的影响

LIBERO 5-shot 限制 10K step 训练。比较 native VLM init vs human-centric pretrain init，对各种 frozen component 组合：

**Key finding**：当 Und+ViT frozen 时，pretrain 给 +25.8% / +41.6%（LIBERO-Long）的 boost。这说明 long-horizon 任务的 temporal consistency 和 intent-grounding **几乎完全** 来自 pre-trained VLM 里的 "world knowledge"。

Full FT 时 marginal utility 下降，部分任务（L-Object）出现 negative transfer（-4.4% Multi-task Full FT）。说明简单 object-centric task 上，aggressive joint fine-tune 可能 overwrite 专门的 pretrain feature。

### 10.5 Action Expert Plasticity

Freeze 0-7 layer：性能几乎不变（>80% success）
Freeze >14 layer：性能急剧下降
全 freeze：<20%

所以 optimal：freeze semantic backbone（Und+ViT）+ 保持 Action Expert 全 trainable。

### 10.6 Masked Motion Token Prediction 的 ablation

用 Mean Wrist Displacement Similarity (MWDS) 评估——因为 unified action space 是 delta representation，MPJPE 不 informative。

| Method | Lab ↑ | Wild ↑ |
|---|---|---|
| Hybrid (Ours) | 0.33 | 0.20 |
| w/o $\mathcal{L}_{\text{MASK}}$ | 0.35 | 0.28 |

等等，这表里 "Hybrid" 反而更低？仔细看 paper 我觉得可能搞反了，但按 paper 写的是 Hybrid Lab 0.33 vs no-MASK 0.35。Paper 的解读是 "removing masked prediction leads to a clear drop in MWDS"——可能 paper 的解读有笔误，或者更高数字代表更差。这里 MWDS 用 cosine similarity，paper 写的是 "averages the cosine similarity between predicted and ground-truth wrist displacement vectors"。Cosine similarity 越高越好，所以 0.35 > 0.33 表示 w/o MASK 更好...这看起来是 paper 笔误。可能数据写反了，或者 metric 解释有歧义。建议读者下载原始 paper 复核。

### 10.7 MPG + UAC 的 ablation

去掉 MPG + UAC，long-horizon 和 bimanual 下降最严重（execution delay + unreliable context 让 compounding error 放大）。Spatial / generalization 受影响小（execution horizon 短，compounding 少）。

---

## 11. 我的 Intuition 总结

这篇 paper 在做几件相互 reinforce 的事：

**1. Human as universal manipulator template**
这个 metaphor 很 powerful。Human hand 是 high-DoF、expressive 的 universal embodiment。Robot 各种种 morphology 都是它的 "projection" 或 "subset"。把 MANO 参数 map 到 unified space 的 "fine-manipulation" slot，相当于 human 是 "superset"，自然能 transfer 到 subset。这跟 NLP 里用 high-resource language 学 grammar 再 transfer 到 low-resource 是同一个 principle。

**2. Unified action space without statistical normalization**
保留 raw physical magnitude，强迫 model 学真实物理量。这是很关键的 design choice——如果 normalize 到 [-1, 1]，1 cm 和 10 cm 在 normalized space 里可能都接近 0.1，model 就无法学到 "10 cm 是 large motion" 这种 physical intuition。Slot-based sparse assignment 让 shared morphology component 共享 adapter，isolated component 独立——这个介于 "all share" 和 "all isolate" 之间的 middle ground 很 sensible。

**3. Hybrid continuous + discrete motion representation**
Continuous flow-matching 给精度，discrete masked token 给 "grammar" 级别的 abstraction。Attention mask 设计让两个 answer segment 互不可见但共享 context——这有点像 multi-task learning 里 shared encoder + task-specific head，但这里 head 之间也不互相干扰。Stop-gradient on gate 防止 degenerate solution，是很 standard 但容易被忽略的 trick。

**4. Mixture of Flow = Mixture of Experts for action generation**
Shared lower layers 学 motor primitive（reaching, grasping dynamics），specialized upper layers 学 embodiment/task-specific dynamics。Active parameter 远小于 total parameter，所以 deploy 到 edge device 可行。这跟 MoE in LLM 的 motivation 一样：scalable capacity without linear compute overhead。

**5. MPG: graceful degradation under distribution shift**
Input gating（gate 在 projection 之前）而不是 output gating，让 ungated bias $\mathbf{b}_{\text{MPG}}$ 在 low-confidence regime 依然 influence 输出，提供 stable fallback。SWD 做 distributional discrepancy 比 KL 更适合 high-dim embedding space，scale-invariant 也重要（不同 embodiment 的 action scale 不同）。

**6. UAC: cross-embodiment async consistency**
传统 RTC 只为单一 platform 设计。UAC 让 delay distribution 和 max delay 都 per-embodiment，single checkpoint 可以适应不同 control frequency（10 Hz tabletop 到 50 Hz humanoid）。Hard prefix locking 防止 flow denoising "edit" 正在执行的动作；hard stitching 保证 continuity。这个 dual-thread + ring buffer 架构在 real-time system 里是 standard pattern，但和 training-time UAC coupled 后很 elegant。

**7. Emergent zero-shot transfer**
这是最 exciting 的 finding。Single generalist checkpoint 在 Adam-U 上能做从未训过的 task（flip-and-scan, drawer, stacking），虽然 success rate 低。这说明 unified action space + cross-embodiment joint training 让 model 学到了 task abstraction（不是 embodiment-specific action convention），可以 compositional generalize 到新 morphology。这跟 LLM 的 emergent ability 类似——scale 和 diversity 到一定程度会出现 unexpected capability。

**潜在的问题 / 我会问的问题**：

1. UniHand-2.0 中 simulation data 26%——这个数字怎么定的？sensitivity analysis？
2. Hybrid Motion Representation 里，continuous 和 discrete 的 weight $\lambda_1, \lambda_2$ 怎么 tune？
3. MPG 里 $\tau$ (gate temperature) 怎么 schedule？是否 per-embodiment？
4. UAC 里 $\pi^{(e)}(d)$ 具体是什么 distribution？Uniform? Geometric?
5. Zero-shot transfer 在哪些 task-embodiment pair 上 work，哪些不 work？Failure mode 是什么？
6. 30 个 embodiment 的数据极度不均衡（Franka 2196 hours vs Cobotta 10.6 hours），如何 handle imbalance？
7. Pre-training 1000 GPU-hour recipe 的具体 breakdown？FLOPs？carbon footprint？
8. InternVL-3.5 作为 backbone——有没有 ablation 比较其他 VLM（Qwen2.5-VL, LLaVA-OneVision）？

参考资源：
- [Being-H0.5 项目主页](https://research.beingbeyond.com/being-h05)
- [Being-H0 前作](https://arxiv.org/abs/2507.15597)
- [π0](https://arxiv.org/abs/2410.24164)
- [π0.5](https://arxiv.org/abs/2504.16054)
- [GR00T-N1](https://arxiv.org/abs/2503.14734)
- [OpenVLA](https://openvla.github.io/)
- [Open X-Embodiment](https://robotics-transformer-x.github.io/)
- [LIBERO benchmark](https://libero-project.github.io/)
- [RoboCasa](https://robocasa.ai/)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Ego4D](https://ego4d-data.org/)
- [EPIC-KITCHENS](https://epic-kitchens.github.io/)
- [MANO hand model](https://mano.is.tue.mpg.de/)
- [HaWoR](https://hawor24.github.io/)
- [InternVL-3.5](https://github.com/OpenGVLab/InternVL)
- [Mixture-of-Transformers](https://arxiv.org/abs/2411.04996)
- [BAGEL](https://github.com/ByteDance-Seed/BAGEL)
- [DiG-Flow](https://arxiv.org/abs/2512.01715)
- [RTC](https://arxiv.org/abs/2512.05964)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [DROID dataset](https://droid-dataset.github.io/)
- [UMI](https://universal-manipulation-interface.github.io/)
- [AprilTag](https://april.eecs.umich.edu/software/apriltag)
- [Grounded-SAM2](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [Gemini 2.5](https://arxiv.org/abs/2507.06261)
- [DINOv2](https://dinov2.metademolab.com/)
- [Mixture of Experts](https://arxiv.org/abs/1701.06538)
- [Sliced Wasserstein Distance](https://arxiv.org/abs/1907.00081)
- [RoboPoint](https://arxiv.org/abs/2406.10721)
- [ShareRobot](https://arxiv.org/abs/2502.11536)
- [MolmoAct](https://arxiv.org/abs/2508.07917)
- [EgoVLA](https://arxiv.org/abs/2507.12440)
- [SO-101](https://github.com/huggingface/lerobot)
- [Unitree G1](https://www.unitree.com/g1)
- [X-VLA](https://arxiv.org/abs/2510.10274)
- [InternVLA-M1](https://arxiv.org/abs/2510.13778)
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
- [CogACT](https://arxiv.org/abs/2411.19650)
- [RDT-1B](https://arxiv.org/abs/2410.07864)
- [AgiBot World](https://agibot-world.com/)
- [RoboMIND](https://arxiv.org/abs/2412.13877)

---

整体上，Being-H0.5 是个非常 solid 的工作。它把 human-centric learning（用大规模 human video 作为 universal prior）、unified action space（物理量级 slot-based representation）、MoF（action expert 的 sparse routing）、MPG（input gating for robust flow denoising）、UAC（cross-embodiment async consistency）几个东西打包成一个 coherent 的 system。最 exciting 的 empirical 发现是 emergent embodiment-level zero-shot transfer——这暗示了 cross-embodiment VLA scaling 的一个 promising direction，跟 LLM 在 scale 到一定程度出现 emergent ability 的故事 arc 很像。

值得 follow 的几个未来方向：(1) 进一步 scale UniHand 到更多小时 + 更多 embodiment；(2) 加入 tactile sensing（UniCraftor 已经留好接口）；(3) 把 backbone 换到更大的 VLM（如 InternVL-3.5-8B/78B）测试 scaling law；(4) 探索更 fine-grained 的 zero-shot transfer failure mode，理解什么 task structure 更容易 transfer。

最后，他们承诺 open-source 模型权重、训练 pipeline、simulation 脚本、real-world deployment infrastructure 和 1000 GPU-hour pre-training recipe，这对社区是巨大贡献。
