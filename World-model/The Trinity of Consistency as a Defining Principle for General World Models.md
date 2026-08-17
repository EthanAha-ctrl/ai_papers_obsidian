---
source_pdf: The Trinity of Consistency as a Defining Principle for General World Models.pdf
paper_sha256: 74b8bb72c8f0afdab8d019e66da66c8ae053fcfa3d0efebf3e32acf4834bfe69
processed_at: '2026-08-12T15:00:03-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话聊透 The Trinity of Consistency 这篇 Paper

Andrej，用最直白的话讲，这篇 paper 的核心吐槽点非常精准：当前像 Sora 这样的 video generation model，本质上只是个“高级画师”。它通过海量数据记住了像素分布的统计规律，能画出让你觉得“哇，这像真的一样”的视频。然而，一旦你让它去处理需要严格物理逻辑的任务，比如物体被遮挡后的 permanence（持久性），或者多视角的 3D 结构一致性，它就开始幻觉满天飞，表现得像个不懂物理常识的盲人画手。

作者提出，真正配得上 General World Model 称号的系统，必须同时拿捏三个维度的 consistency。这就构成了所谓的 **Trinity of Consistency** 框架。

## 1. Trinity 拆解：World Model 的三块拼图

### Modal Consistency (语义接口)
这就好比给 world model 装上一个能听懂多语言的大脑。模型得把 text, image, audio 这些异质信息映射到同一个 semantic space 里。背后的数学基础是 Platonic Representation Hypothesis，假设存在一个真实的物理世界 latent space $\mathcal{Z}_{world}$，text 和 image 都只是它向低维空间的 projection。
$$x_{img} = \mathcal{P}_{img}(\mathcal{Z}_{world}), \quad x_{txt} = \mathcal{P}_{txt}(\mathcal{Z}_{world})$$
这里 $\mathcal{P}_{img}$ 和 $\mathcal{P}_{txt}$ 是投影算子，$x_{img}$ 是图像观测值，$x_{txt}$ 是文本观测值。
问题在于，text 是高度抽象的离散符号逻辑，image 是充满高频细节的连续像素流。强行用 Hypersphere 假设去对齐，就会产生 modality gap。作者在架构演进里力推 MM-DiT，核心就是给 text 和 image 维护独立的 weight sets $W_{txt}, W_{img}$，只在 attention 计算时交换数据，让 Hessian matrix 呈现 block-diagonal 结构，从而大幅降低梯度冲突。

### Spatial Consistency (几何基石)
模型得在 3D 空间里有常识。不能换个视角，狗的鼻子就长到眼睛上面了，这就是经典的 Janus Problem。作者把这个问题抽象为在 spatiotemporal manifold $\mathcal{M} \subseteq \mathbb{R}^3 \times \mathbb{R}^+$ 上解逆微分方程。
最核心的物理法则是 Radiative Transfer Equation (RTE)：
$$\underbrace{d \cdot \nabla L(x,d)}_{\text{Transport}} = \underbrace{-\sigma(x)L(x,d)}_{\text{Absorption}} + \underbrace{\sigma(x)c(x,d)}_{\text{Emission}}$$
这里 $L(x,d)$ 是位置 $x$ 沿方向 $d$ 的 radiance，$\sigma(x)$ 是体积密度，$c(x,d)$ 是依赖于视角的颜色发射。
NeRF 靠沿着 ray 做 Riemann sum 积分硬算，3D Gaussian Splatting (3DGS) 则把它离散化成一组 Lagrangian primitive $\mathcal{G}_i(\mu, \Sigma, \alpha, SH)$。这里 $\mu$ 是 3D 均值位置，$\Sigma$ 是协方差矩阵，$\alpha$ 是不透明度，$SH$ 是球谐函数系数。3DGS 的 Forward Push 机制让梯度 $\frac{\partial \mathcal{L}}{\partial \mu}$ 能直接稀疏地反向传播到几何参数上，这比 NeRF 的 Backward Pull 高效太多。

### Temporal Consistency (因果引擎)
视频不仅是帧的平滑过渡，更要遵循因果律。水杯必须先掉落，然后碎裂，绝对不能凭空恢复。作者强调了 Video Consistency Distance (VCD) 这个指标，用来抓高频闪烁：
$$\mathcal{L}_{\text{VCD}}(\hat{V}) = \mathbb{E}_t\left[\|\mathcal{F}_t(\phi(\hat{v}_t)) - \mathcal{F}_t(\phi(\hat{v}_{t-1}))\|^2_{\text{High-Pass}}\right]$$
这里 $\hat{V}$ 是生成的视频，$\phi(\cdot)$ 是 CLIP 特征提取器，$\mathcal{F}_t$ 是沿时间轴的短时傅里叶变换 (STFT)，$\|\cdot\|^2_{\text{High-Pass}}$ 是高通滤波后的范数。物理世界里的运动在频域上应当是连续的，如果生成模型出现 temporal inconsistency，在 high-frequency band 就会有剧烈的能量波动。

## 2. 架构演进的 Intuition：从画皮到画骨

早期的 video model 用 temporal inflation（比如 AnimateDiff），冻结 2D U-Net，只在中间插入 1D temporal attention。这就像给静态画做微小的弹性形变，一旦遇到大视角旋转或新内容出现，立刻原形毕露。

后来大家搞 Discrete Autoregressive (AR) 建模，把视频变成 token 序列，用 causal mask 预测。数学上是 next-token prediction：
$$\mathcal{L}_{AR} = -\mathbb{E}\left[\sum \log P(s_t | s_{<t})\right]$$
这里 $s_t$ 是 timestep $t$ 的 token，$s_{<t}$ 是历史 tokens。
问题在于，如果每一步预测的算子 Lipschitz constant $L > 1$，长序列的误差就会以 $L^T$ 指数级放大。序列越长，画面崩得越惨。

现在的 SOTA 路线是 Continuous Flow Matching (FM) 结合 Native 3D DiT。FM 的 loss 是回归 velocity field：
$$\mathcal{L}_{FM} = \mathbb{E}\left[\|v_\theta(x_t) - (x_1 - x_0)\|^2\right]$$
这里 $v_\theta$ 是网络预测的速度场，$x_1$ 是数据样本，$x_0$ 是噪声样本，$x_t = (1-t)x_0 + tx_1$ 是线性插值。
Rectified Flow 证明了这种直线轨迹的 Lipschitz constant $L \approx 1$，误差累积变成了线性增长 $T \cdot \bar{\epsilon}_{step}$。配上 3D-RoPE 机制算全序列 attention，模型终于开始能 capture 长距离的物理相互作用了。

## 3. Modal 与 Spatial 的融合：让语言操控几何

怎么让 text 真正控制 3D 几何？作者提到了 Score Distillation Sampling (SDS) 的核心逻辑：
$$\nabla_\theta \mathcal{L}_{SDS} = \mathbb{E}_{t,\epsilon}\left[w_{guidance}(\epsilon_\theta(z_t, t) - \epsilon) \frac{\partial x_{img}}{\partial \theta}\right]$$
这里 $w_{guidance}$ 是权重系数，$\epsilon_\theta$ 是冻结的 2D diffusion model 预测的噪声，$\frac{\partial x_{img}}{\partial \theta}$ 是可微渲染器的 Jacobian 矩阵，$\theta$ 是 3D 场参数。
本质上就是让 2D 图像模型计算出的语义残差，通过 Jacobian 矩阵反向传播，直接去雕刻 3D 几何参数。这就打通了 Semantic 和 Geometry 的壁垒。

## 4. CoW-Bench 揭露了什么真相

作者搞了个 CoW-Bench，专门去测这三个维度的 consistency。评估方式极其硬核：把视频抽 4 帧排成 2x2 的网格，用 16 个 atomic checks 去逐项打分（0-2分）。

结果发现了一个非常普遍的现象，叫 **Constraint Backoff（约束退让）**。
当模型遇到复杂的 prompt，比如“把那只罕见的蓝色材质的猫放在红色的箱子上”，模型发现直接精确渲染太难，就会悄悄把要求降级，给你画一只普通的猫，或者普通的箱子。视觉上看不出毛病，打分也很高，但在逻辑上完全违背了 prompt 的硬性约束。

在 Maze-2D（2D迷宫）任务里更明显。模型能画出非常逼真的迷宫墙（spatial consistency 高分），但那个移动的物体根本到不了终点，甚至走着走着就瞬移了（temporal-spatial consistency 崩盘）。这证明目前的 model 只是一个 high-fidelity pixel interpolator，它脑子里没有一个 persistent world state（持续的世界状态）。

## 5. Karpathy 视角的联想

你之前讲过 system 1 和 system 2 的区别。现在的 video model 就是纯纯的 system 1，靠直觉画画，缺乏深思熟虑的 system 2 reasoning。这篇 paper 提出未来的终局是 **Prompt-as-Action** 范式，实际上就是在呼唤把 Unified Multimodal Model (UMM) 的 system 2 reasoning 能力，直接编译成 spatiotemporal dynamic simulation。

关于 Yann LeCun 力推的 JEPA。JEPA 是在 latent space 里做 prediction，避开了像素生成的困难。这篇 paper 里把它归类为 "Vector-as-Action" paradigm，指出它虽然避开了高频像素干扰，但缺乏对物理方程 explicit 的约束，interaction 机制完全不透明。而 Genie 系列属于 "Key-as-Action"，虽然有了离散的动作空间，但太受限，缺乏语义表达力。

这篇 paper 其实给出了一个非常清晰的信号：单纯堆参数和堆数据，只能让画师画得更像，永远成不了物理学家。要实现真正的 World Simulator，必须在架构层面把 Modal, Spatial, Temporal 的 consistency 强制耦合，把 Hamiltonian、守恒定律作为 differentiable operator 写进 loss function 里。从画皮走向画骨，这是通往 AGI 物理世界建模的必经之路。

Reference: [Sora Technical Report](https://openai.com/research/video-generation-models-as-world-simulators), [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987), [3DGS](https://arxiv.org/abs/2308.14737), [JEPA](https://openreview.net/pdf?id=BZ5a1r-kVsf)

---

# The Trinity of Consistency: World Models 的定义原则深度解析

## 1. 核心论点：Trinity of Consistency 框架

这篇 paper 的核心贡献是提出了一个原则性的理论框架，定义了什么是 General World Model。作者认为，一个真正的 World Model 必须同时满足三个正交又协同的 consistency 约束：

**Modal Consistency** (Semantic Interface) - 将 text, image, tactile 等 heterogeneous information 对齐到 unified semantic space

**Spatial Consistency** (Geometric Basis) - 构建符合 geometry, occlusion, object permanence 的 3D-aware representation

**Temporal Consistency** (Causal Engine) - 遵循 physical laws 和 causal logic 的 temporal evolution

这个框架的关键 insight 在于：现有的 video generation models 如 Sora 虽然能生成视觉上 plausible 的内容，但本质上仍是 "naive physicists"，会产生 structural hallucinations 和 causality violations。真正的 World Model 需要这三个维度协同涌现，而非孤立优化。

Reference: [Sora Technical Report](https://openai.com/research/video-generation-models-as-world-simulators), [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)

## 2. Modal Consistency 的理论基础与架构演进

### 2.1 Platonic Representation Hypothesis 与 Modality Gap

Modal Consistency 的理论基础来自 Platonic Representation Hypothesis [Huh et al., 2024]。形式化定义如下：

存在一个客观的 latent physical state space $\mathcal{Z}_{world}$，images 和 text 都是这个高维实体到不同低维子空间的 projections：
$$x_{img} = \mathcal{P}_{img}(\mathcal{Z}_{world}), \quad x_{txt} = \mathcal{P}_{txt}(\mathcal{Z}_{world})$$

核心问题是这是一个 ill-posed problem - visual projection $\mathcal{P}_{img}$ 保留了 high-frequency physical entropy，而 textual projection $\mathcal{P}_{txt}$ 高度抽象化为 discrete symbolic logic。这种 **Entropy Asymmetry** 是 direct alignment 的根本障碍。

**Hypersphere Hypothesis** (CLIP 类方法) 假设 features 均匀分布在 unit hypersphere $\mathbb{S}^{d-1}$ 上。但 Liang et al. [NeurIPS 2022] 指出了 **cone effect**：joint optimization 导致 visual 和 textual embeddings 坍缩为两个狭窄且分离的 conical regions，破坏了 isotropy。

这里的关键数学 insight 是 **manifold nonisomorphism**：
- Visual data: continuous, dense low-dimensional manifold
- Linguistic data: sparse, discrete clustering structure

两者 intrinsic dimensionality 和 data density 的根本差异，使得 perfect isometric alignment 成为 ill-posed problem。

Reference: [CLIP](https://arxiv.org/abs/2103.00020), [Mind the Gap](https://arxiv.org/abs/2208.02074)

### 2.2 Discrete AR vs. Continuous Flow Matching 的本质对比

Table 2 给出了两种 generative manifold mechanism 的对比：

**Discrete Autoregressive (AR):**
$$\mathcal{L}_{AR} = -\mathbb{E}\left[\sum \log P(s_t | s_{<t})\right]$$

其中 $s_t$ 是 timestep $t$ 的 discrete token，$s_{<t}$ 是历史 tokens。误差传播特性是 **Exponential**：若 operator 的 Lipschitz constant $L > 1$，初始量化误差 $\epsilon_0$ 经过 $T$ 步累积后：
$$\|\delta_T\| \approx L^T \|\epsilon_0\|$$

这解释了为什么 AR 模型在生成长序列时尾部经常出现 structural collapse。

**Continuous Flow Matching (FM):**
$$\mathcal{L}_{FM} = \mathbb{E}\left[\|v_\theta(x_t) - (x_1 - x_0)\|^2\right]$$

其中 $v_\theta(x_t)$ 是网络预测的 velocity field，$x_1$ 是 data sample，$x_0$ 是 noise sample，$x_t = (1-t)x_0 + tx_1$ 是 linear interpolation。

Rectified Flow 证明了 Reflow 操作将 transport trajectory 修正为近乎直线，对应 Lipschitz constant $L \approx 1$，误差累积变为 **linear growth**：
$$\|\delta_T\| \approx T \cdot \bar{\epsilon}_{step}$$

这允许 FM 在极少 steps 内生成 high-fidelity samples，同时完美保持 latent space 的 continuous semantic manifold。

Reference: [Rectified Flow](https://arxiv.org/abs/2209.03003), [Flow Matching](https://arxiv.org/abs/2210.02747)

### 2.3 MM-DiT 的 Orthogonal Decoupling 架构

架构演进经历了三个阶段：

**(1) Dual-Tower** (CLIP, ALIGN)：独立 encoders 处理 images 和 text，通过 contrastive learning 投影到 shared hypersphere。缺陷：缺乏 deep fine-grained interaction。

**(2) Connector-based** (Flamingo, BLIP-2)：frozen visual encoder + learnable bridge modules (Q-Former, Perceiver Resampler)。但这造成了 information asymmetry - LLaVA 中的 linear projection $W_{proj}$ 本质上是 low-rank compressor，在优化时 suppresses 了 high-frequency components。

**(3) MM-DiT** (Stable Diffusion 3.5, Emu3)：**weight decoupling** 策略，为 text 和 image 维护独立的 weight sets $W_{txt}, W_{img}$，仅在 attention operations 时交换数据。

关键的数学 insight 在于 Hessian matrix 的 block-diagonal 结构：
$$H_{total} \approx \begin{bmatrix} H_{txt} & 0 \\ 0 & H_{img} \end{bmatrix}, \quad \text{s.t.} \frac{\partial^2 \mathcal{L}}{\partial W_{txt} \partial W_{img}} \approx 0$$

其中 $H_{total}$ 是 joint Hessian matrix，$W_{txt/img}$ 是 modality-specific parameters。这种结构有效隔离了 modality-specific curvature，使 gradient updates 趋于 orthogonal。Empirical data 显示 gradient conflict rate 从 AR paradigms 的 >50% 降至约 30%。

Reference: [Stable Diffusion 3](https://arxiv.org/abs/2403.03206), [Emu3](https://arxiv.org/abs/2409.18869)

### 2.4 RL-based Intent Alignment

Traditional MLE 只捕获 pixel statistical correlations，缺乏 explicit supervision。PhyGDPO 引入了 physics-aware VLM feedback，核心 loss function：

$$\mathcal{L}_{\text{Phy-DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(v_w)}{\pi_{ref}(v_w)} - \beta \log \frac{\pi_\theta(v_l)}{\pi_{ref}(v_l)} + \alpha \Delta \text{PhysScore}\right)\right]$$

变量解释：
- $\beta$: KL divergence penalty coefficient，控制偏离 reference policy $\pi_{ref}$ 的程度
- $\pi_\theta$: 当前 policy
- $\pi_{ref}$: reference policy (frozen)
- $v_w, v_l$: winning 和 losing video samples
- $\alpha$: 物理约束权重
- $\Delta\text{PhysScore}$: 物理合规性分数差异

AR 模型面临 discrete space non-differentiability 和 temporal error accumulation 的双重挑战。AR-GRPO 提出的 factorized optimization：

$$\mathcal{L}_{AR-RL} = \underbrace{\mathbb{E}_\pi[R(x)]}_{\text{Alignment Gain}} - \beta \underbrace{D_{KL}(\pi \| \pi_{ref})}_{\text{Temporal Smoothing}}$$

- $R(x)$: reward function (CLIP 或 VQA feedback)
- $\beta$: KL divergence regularization coefficient
- $D_{KL}$: KL divergence

这个设计的关键在于 alignment term 指导 token selection 符合 semantic intent，KL divergence constraint 强制 policy 保持在 pretrained language manifold 内，防止 "Language Collapse"。

Reference: [DPO](https://arxiv.org/abs/2305.18290), [VideoDPO](https://arxiv.org/abs/2403.18431)

## 3. Spatial Consistency：从 2D Proxy 到 3D Primitives

### 3.1 Geometric Decomposition

Spatial Consistency 被分解为两个层次：

**Micro-level: Local Neighborhood Topological Consistency**
对应 Lipschitz Condition：对于 manifold $\mathcal{M}$ 上任意相邻两点，其物理属性差异被 Euclidean distance 严格线性约束。IGR [Gropp et al., ICML 2020] 利用 Eikonal equation 约束 gradient norm。

**Macro-level: Global Geometric Consistency**
要求满足 Epipolar Equivariance：从不同 viewpoints $v_a, v_b$ 观察同一物体时：
$$\mathbf{x}_b^\top \mathbf{F}_{ab} \mathbf{x}_a = 0$$

其中 $\mathbf{x}_a, \mathbf{x}_b$ 是对应点的 homogeneous coordinates，$\mathbf{F}_{ab}$ 是 fundamental matrix。违反此约束导致 **Janus Problem** - 不同 viewpoints 产生 incompatible object geometries。

Reference: [Multiple View Geometry](https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/)

### 3.2 Theoretical Formulation: 三大物理定律

作者将 spatial consistency 形式化为 spatiotemporal manifold $\mathcal{M} \subseteq \mathbb{R}^3 \times \mathbb{R}^+$ 上的 coupled Inverse Differential Problems：

**(1) Physical Rendering: Radiative Transfer Equation (RTE)**
$$\underbrace{d \cdot \nabla L(x,d)}_{\text{Transport}} = \underbrace{-\sigma(x)L(x,d)}_{\text{Absorption}} + \underbrace{\sigma(x)c(x,d)}_{\text{Emission}}$$

变量：
- $L(x,d)$: 位置 $x$ 沿方向 $d$ 的 radiance
- $\sigma(x)$: 位置 $x$ 的 volume density
- $c(x,d)$: view-dependent color emission
- $d$: ray direction

NeRF 通过 dense Riemann summation 近似求解；3DGS 通过 Lagrangian Gaussian basis functions 转化为 analytical rasterization。

**(2) Generative Evolution: Stochastic Differential Equation (SDE)**
$$d\Phi_t = f(\Phi_t, t)dt + g(t)dw$$

- $f(\cdot)$: deterministic drift term，governing semantic evolution
- $g(t)$: diffusion coefficient
- $w$: standard Wiener process

当 $g(t) = 0$ 时退化为 ODE，即 Flow Matching。

**(3) Motion Law: Lagrangian Transport**
$$\frac{d\mathbf{x}}{dt} = \mathbf{v}(\mathbf{x}, t), \quad \text{s.t.} \frac{D\Phi}{Dt} = 0$$

- $\mathbf{v}$: velocity field
- $\frac{D\Phi}{Dt}$: material derivative
- $\Phi$: conserved feature

Material derivative = 0 意味着 feature $\Phi$ 随 fluid 运动保持守恒，这是 explicit primitive paradigm 中 particle tracking 的数学基础。

Reference: [NeRF](https://arxiv.org/abs/2003.08934), [3DGS](https://arxiv.org/abs/2308.14737)

### 3.3 2D Proxy Manifold 的根本局限

早期 video prediction 方法 (ConvLSTM, PredRNN) 试图将高维 physical state field $\Phi$ 的 evolutionary operator $\mathcal{F}_{3D}: SE(3) \times \mathbb{R}^3 \to \mathbb{R}^3$ 简化为 2D image manifold 上的映射 $\mathcal{F}_\theta: \mathbb{R}^{H\times W} \to \mathbb{R}^{H\times W}$。

这引入了 **Domain Mismatch**：convolution operations $W * I$ 只有 Translation Equivariance，缺乏对 3D rotation group $SO(3)$ 的感知能力。当 camera 进行 large viewpoint transformations 时，模型无法 decouple extrinsic camera motion 和 intrinsic object deformation，产生 non-physical Non-rigid Distortion。

PhyDNet [Guen & Thome, CVPR 2020] 试图通过 Moment Matching 将 convolution kernel weights 近似为 PDE finite difference operators：
$$\frac{\partial \mathcal{H}}{\partial t} \approx \sum_k c_k \frac{\partial^k \mathcal{H}}{\partial x^k} \implies \text{Filter Weights} \xrightarrow{\text{Moment}} \text{Finite Difference Stencils}$$

但一旦 occlusion 导致 depth mutations，optical flow field 变得 non-differentiable，PDE constraints 立即失效。

### 3.4 Implicit Continuous Fields: NeRF 谱系

NeRF [Mildenhall et al., ECCV 2020] 用 MLP 参数化 continuous coordinate mapping $F_\Theta: (x,d) \mapsto (c,\sigma)$，通过 differentiable Volume Rendering Integral 连接 3D field 与 2D observations。

**Anti-aliasing 机制：Mip-NeRF**
$$\gamma(\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \mathbb{E}_{\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})}[\gamma(\mathbf{x})] \approx \sin(\boldsymbol{\mu}) \circ \exp\left(-\frac{1}{2}\text{diag}(\boldsymbol{\Sigma})\right)$$

- $\boldsymbol{\mu}$: conical frustum 的 mean vector
- $\boldsymbol{\Sigma}$: covariance matrix
- $\circ$: element-wise product
- $\gamma(\cdot)$: positional encoding

指数衰减项 $\exp(-\boldsymbol{\Sigma})$ 本质是 **Adaptive Low-pass Filter**。当 sampling cone radius 增大 (variance $\boldsymbol{\Sigma}$ 大，对应 distant views 或 low-resolution regions)，high-frequency features 被指数抑制。这是 Fourier transforms 中 Uncertainty Principle 的体现：spatial localization 越宽，frequency bandwidth 越窄。

**Surface Extraction: SDF 与 Eikonal Constraint**
$$\mathcal{L}_{geo} = \mathbb{E}_x [(\|\nabla f(x)\|_2 - 1)^2]$$

- $f(x)$: signed distance function
- $\nabla f$: gradient of SDF
- $\|\nabla f\|_2 = 1$: 强制 gradient norm 恒为 1

这确保 zero-level set $\mathcal{S} = \{x | f(x) = 0\}$ 收敛到 smooth, closed manifold surface。

Reference: [Mip-NeRF](https://arxiv.org/abs/2103.13415), [NeuS](https://arxiv.org/abs/2106.01120)

### 3.5 Explicit Lagrangian Primitives: 3DGS 革命

3D Gaussian Splatting [Kerbl et al., 2023] 将 scene 表示为 anisotropic Gaussian primitives 集合 $\Phi = \{\mathcal{G}_i(\mu, \Sigma, \alpha, SH)\}_{i=1}^M$：

- $\mu$: 3D position (mean)
- $\Sigma$: 3D covariance matrix
- $\alpha$: opacity
- $SH$: spherical harmonics coefficients (view-dependent color)

**Rasterization Pipeline**:
1. Frustum Culling 与 Projection: $\Sigma^{2D} = JW\Sigma^{3D}W^TJ^T$，其中 $W$ 是 viewing transformation，$J$ 是 Jacobian of projective transformation
2. Tiled radix sort: 复杂度 $O(N \cdot k)$，computational bottleneck

**Integral Duality**: NeRF 是 Backward Pull (ray marching)，容易 gradient masking ($\partial C / \partial \sigma_{far} \approx 0$)；3DGS 是 Forward Push，explicit sparsity 允许 error gradients $\frac{\partial \mathcal{L}}{\partial \mu}$ 绕过 MLP 直接 sparsely backpropagate 到 geometric parameters。

**Adaptive Density Control** (AMR variant):
- $\|\nabla \mathcal{L}\| > \tau, \|\Sigma\| < \epsilon$: underfitting → Clone
- $\|\nabla \mathcal{L}\| > \tau, \|\Sigma\| > \epsilon$: overfitting → Split

**4D Evolution 的三条路径**:

**(i) Lagrangian Particle Tracking** (PhysGaussian):
$$\mu(t) = \mu_0 + \int v(\tau) d\tau$$
嵌入 continuum mechanics equations $\rho \ddot{\mathbf{x}} = \nabla \cdot \boldsymbol{\sigma} + \mathbf{g}$

**(ii) Eulerian Tensor Decomposition** (4D-GS):
$$\mathcal{T}(x,y,z,t) \approx \sum_{r=1}^R \mathbf{u}_r(x) \circ \mathbf{v}_r(y) \circ \mathbf{w}_r(z) \circ \mathbf{h}_r(t)$$

- $\circ$: outer product
- $R$: tensor rank
- $\mathbf{u}_r, \mathbf{v}_r, \mathbf{w}_r, \mathbf{h}_r$: 沿各维度的 factor vectors

将 storage complexity 从 $O(N^4)$ 优化到 $O(N^2)$。

**(iii) Canonical Deformation** (Deformable-GS):
$$\boldsymbol{\mu}_t = \boldsymbol{\mu}_0 + \Delta_\mu(\boldsymbol{\mu}_0, t), \quad \boldsymbol{\Sigma}_t = f(\boldsymbol{\Sigma}_0, \Delta_r(\boldsymbol{\mu}_0, t))$$

- $\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0$: canonical space 中的 mean 和 covariance
- $\Delta_\mu$: 预测的 position offset
- $\Delta_r$: rotation update

利用 MLP 的 spectral bias 有效 capture high-frequency motion fields。

Reference: [3DGS](https://repo.akm.cv/), [4D-GS](https://arxiv.org/abs/2402.07307)

## 4. Temporal Consistency: 从 Frequency Stability 到 Causal Reasoning

### 4.1 Video Consistency Distance (VCD)

传统 FVD 主要评估 spatial feature distributions，难以检测 temporal high-frequency Flickering。VCD 设计为：

$$\mathcal{L}_{\text{VCD}}(\hat{V}) = \mathbb{E}_t\left[\|\mathcal{F}_t(\phi(\hat{v}_t)) - \mathcal{F}_t(\phi(\hat{v}_{t-1}))\|^2_{\text{High-Pass}}\right]$$

- $\hat{V}$: generated video
- $\phi(\cdot)$: feature extractor (CLIP Image Encoder)
- $\mathcal{F}_t$: Short-Time Fourier Transform (STFT) along time axis
- $\|\cdot\|^2_{\text{High-Pass}}$: high-pass filtered norm

物理意义：real-world motion features 应在 frequency domain 具有连续性，而 generative models 的 temporal inconsistencies (texture flickering) 在 high-frequency band 表现为显著 energy fluctuations。

### 4.2 Temporal Inflation 的理论边界

AnimateDiff 等 Temporal Inflation paradigm 采用 "Spatial Freeze, Temporal Insertion" 策略，将 video joint distribution $p(x_{1:T})$ 简化为 first-order Markov chain。

这种 ELBO relaxation 忽略了 high-order dependencies $p(x_t | x_{<t-1})$，导致 KL divergence term 在 long sequences 上显著增加。Empirical 表现：当生成帧数 $T > 16$ 时，初始帧的 identity features 被独立 noise injection 逐渐稀释 (Semantic Drift)。

Frequency analysis 显示 temporal attention 在 (B·HW) 维度独立操作，在 high-frequency band (>15Hz) 出现显著 energy loss，视觉上表现为 nonphysical texture flickering。

FastInit 提出了 learning-based noise initialization：训练 lightweight inversion network 基于前帧 spatiotemporal features 直接预测 optimal initial noise，但这仍受限于 2D topological anchor。

### 4.3 Native 3D DiT 与 Full Spatiotemporal Isomorphism

Sora, HunyuanVideo 等建立了 Native 3D DiT paradigm：

**(i) Full Sequence Joint Attention**:
$$\text{Attn} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + \mathcal{M}\right)V$$

通过 3D-RoPE 在 full sequence $N = T \times H \times W$ 上计算 joint attention。Empirical studies (Physics-IQ) 表明，decomposition architectures 难以 approximate Navier-Stokes equations 中的 convective terms 和 long-range correlations。

**(ii) Manifold Diffeomorphism**: Flow Matching 对应 manifold 上的 diffeomorphism，从 Gaussian noise smooth recovery minute texture details，消除 discretization 导致的 edge flickering。

**Computational Evolution**:
- Video-TTT: Test-Time Training 压缩历史 context 到网络 weights，实现 $O(N)$ linear complexity
- Pyramid Flow: 层级 decoupling，computational cost 降低 5-10x
- TeaCache: 利用相邻 timesteps feature outputs 的 Pearson correlation > 0.98，training-free dynamic caching，2-3x 加速

### 4.4 Causal Reasoning: Chain-of-Frame

从 visual fluency 到 event causality 的跃迁。Video-CoT, Video Espresso 提出 Chain-of-Frame paradigm，将 video generation 分解为 keyframe planning 和 intermediate frame synthesis：

$$\text{Plan}(c_{txt}) \to \{k_1, k_2, \ldots, k_n\} \to \text{Interpolate}(k_i, k_{i+1})$$

这强制 model 先确定 causal nodes，再 generate visual process。Think Sound 将此扩展到 auditory modality，通过 aligning underlying causal graph structures across modalities，enforce logical self-consistency throughout full spatiotemporal span。

Reference: [Sora](https://openai.com/research/video-generation-models-as-world-simulators), [HunyuanVideo](https://arxiv.org/abs/2412.03630)

## 5. Modal-Spatial Consistency 融合：四条技术路径

### 5.1 Pixel Space Manipulation

核心假设：大规模 2D projection data $\{x_{img,i}\}_{i=1}^N$ 足以覆盖高维 3D manifold $\mathcal{M}_{world}$ 的 topological structure。

**Instruction-Driven Editing 的 Gradient Decoupling**:
$$\nabla_\theta \mathcal{L} = \underbrace{\nabla_{\theta_{base}} \mathcal{L}_{prior}}_{\approx 0 (\text{Frozen})} + \underbrace{\nabla_{\theta_{adapter}} \mathcal{L}_{edit}}_{\text{Semantics}}$$

- $\theta_{base}$: 冻结的 base model parameters
- $\theta_{adapter}$: 可训练的 side-network parameters

**Attention Injection as Gating**:
$$\text{Attn}_{edit}(Q,K,V) \leftarrow \alpha \cdot \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right) + (1-\alpha) \cdot M_{attn}^{src}$$

- $M_{attn}^{src}$: 源图像保留的 attention map
- $\alpha$: injection strength coefficient

**Joint Modeling Breaking Information Bottleneck**:
$$\mathcal{L}_{joint} = -\sum_t \log p(x_{img,t} | x_{img,<t}, x^{txt}) - \sum_j \log p(x_j^{txt} | x_{img}, x_{<j}^{txt})$$

- $x_{img,t}$: step $t$ 的 image tokens
- $x^{txt}$: text tokens
- 全双工 information flow capture "a cat on a table" 中的 pixel-level spatial constraints

### 5.2 View Space Mapping

引入 3D geometric information 作为 structured condition variables $\tau = \{P_t, \mathcal{D}\}$：

$$\mathcal{L}_{view} = \mathbb{E}_{z,t,c,\tau,\epsilon}\left[\|\epsilon - \epsilon_\theta(z_t, t, c, \tau)\|_2^2\right] + \lambda \mathcal{R}_{consist}$$

- $z_t$: timestep $t$ 的 noisy latent
- $\epsilon_\theta$: conditioned on geometry $\tau$ 的 noise prediction network
- $\mathcal{R}_{consist}$: multi-view consistency regularization

**Structured Sparse Attention** (MVDream, SyncDreamer):
$$\text{Attn}(Q_i, K_j, V_j) \propto \exp\left(\frac{Q_i K_j^T}{\sqrt{d}} + \mathcal{M}_{epi}(i,j)\right)$$

$\mathcal{M}_{epi}(i,j)$: 来自 epipolar constraints 的 geometric bias。强制不同 views 的 tokens 仅与其 geometrically corresponding epipolar line regions 交互，将 geometric hard constraints 转化为 attention 中的 soft inductive bias。

### 5.3 Volume Space Representation

**Score Distillation Sampling (SDS)**:
$$\nabla_\theta \mathcal{L}_{SDS} = \mathbb{E}_{t,\epsilon}\left[w_{guidance}(\epsilon_\theta(z_t, t) - \epsilon) \frac{\partial x_{img}}{\partial \theta}\right]$$

- $w_{guidance}$: weighting factor
- $\epsilon_\theta$: frozen diffusion model 预测的 noise
- $\frac{\partial x_{img}}{\partial \theta}$: differentiable renderer $g$ 的 Jacobian matrix

2D model 计算的 semantic residual 通过 Jacobian matrix backpropagate 到 3D geometry。

**Video Manifold as Dynamic 3D Prior**:
$$p(x_{img,novel} | x_{img,ref}) \approx p_{video}(x_{img,t+1} | x_{img,t}, motion\_cond)$$

Video Diffusion Models 的 temporal correlation ≈ spatial consistency。SV3D 利用 temporal attention layer 作为 soft epipolar constraint。

### 5.4 RL for Modal-Spatial Alignment

R-DPO 提出 sub-manifold optimization under spatial masks：
$$\mathcal{L}_{R-DPO} = -\sum_k \mathbb{E}_{(x_w, x_l) \sim B_k}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(x_{img,w}^k | c_k)}{\pi_{ref}(x_{img,w}^k | c_k)} - \beta \log \frac{\pi_\theta(x_{img,l}^k | c_k)}{\pi_{ref}(x_{img,l}^k | c_k)}\right)\right]$$

- $B_k$: region $k$ 的 local preference dataset
- $x_w, x_l$: winning 和 losing image crops
- $\sigma$: sigmoid function

Reference: [SDS/DreamFusion](https://arxiv.org/abs/2209.14988), [MVDream](https://arxiv.org/abs/2308.16512)

## 6. CoW-Bench: 评估 World Models 的新基准

### 6.1 任务设计

CoW-Bench 围绕三个 core consistencies 及其 pairwise integration 设计了 6 个 task categories，18 个 sub-tasks：

| Category | Sub-task I (Atomic) | Sub-task II (Dynamic) | Sub-task III (Constraint) |
|----------|-------------------|----------------------|------------------------|
| **M** (Modal) | Style/Material transfer | Fine-grained control | Multi-constraint composition |
| **S** (Spatial) | Planar layout | Hierarchical occlusion | Multi-view 3D structure |
| **T** (Temporal) | Worldline persistence | Rule-guided evolution | Ordered stage transitions |
| **M×S** | Semantic planar binding | Semantic hierarchy control | Semantic 3D view consistency |
| **M×T** | Long-horizon anchoring | Attribute dynamics alignment | Triggered event compliance |
| **S×T** | Planar maze trajectory | Occlusion dynamics | 3D loop navigation coherence |

关键设计原则是 **Single-Consistency Variable Control Protocol**: 每个 sub-task 只允许与 target consistency 直接相关的 variables 变化，其他 confounding factors (entity count, background complexity, camera movement, motion magnitude, occlusion conditions) 明确约束。

### 6.2 Atomic Decomposition 评估框架

**16 个 Atomic Checks (A1-A16)**:

| ID | Atomic Check | Operational Definition |
|----|-------------|------------------------|
| A1 | Identity lock | Target entity 不变，无 identity swap/duplication/replacement |
| A2 | Attribute binding | Key attributes 保持绑定到同一 entity |
| A3 | Constraint non-relaxation | Specified constraints 不被 weakened 或 substituted |
| A4 | Evidence clarity | 支持每个 constraint judgment 的 evidence 清晰明确 |
| A5 | Mutual exclusivity | 互斥 properties 不共现于同一 target |
| A6 | Locality of change | 变化限制在 designated region/attribute |
| A7 | Non-target invariance | Non-target entities/regions 保持稳定 |
| A8 | No spurious additions | 无 instruction 之外的额外 entities |
| A9 | Set completeness | Required entities 形成完整 set with correct cardinalities |
| A10 | Relation correctness | Specified relations/actions 被满足，无 role swapping |
| A11 | Multi-constraint coverage | Multiple constraints 联合满足，无 selective omission |
| A12 | Worldline stability | Output 描绘单一 consistent world，无 frame-wise reinitialization |
| A13 | Temporal continuity | 变化平滑，无 abrupt jumps 或 oscillatory backtracking |
| A14 | Stage structure | Discrete stages 可识别且按正确顺序出现 |
| A15 | Occlusion & layering | Depth ordering 和 occlusion 正确且 non-contradictory |
| A16 | 3D geometric coherence | Multi-view outputs 作为单一 3D scene 的 projections 一致 |

**Scoring Scale (0-2)**:
- 0: 明确 violation 或 failure
- 1: partial fulfillment 但有 ambiguity, deviation, 或 unclear evidence
- 2: 清晰, 稳定, 无争议的 fulfillment

**2×2 Grid Temporal Sampling Protocol**: 从 video sequence 中 uniformly sample 4 frames，按 chronological order 排列为 2×2 grid (left-to-right, top-to-bottom)。Evaluator 必须 frame-by-frame 分析，reducing selective observation bias。

### 6.3 主要实验结果分析

Table 9 的 main results 揭示了几个关键 insight：

**(1) Temporal control 是瓶颈，而非 coherence**
T-WL (worldline persistence) 对多数 video models 都很高 (Sora: 9.32)，但 T-Rule 和 T-Stage-Order 在不同模型间差异显著。这支持了 CoW-Bench 的核心论点：world models 需要 constraint satisfaction over time，而非仅 smoothness。

**(2) Spatial consistency 在 single-view 3D 较强，cross-view anchoring 仍破裂**
多数 top models 在 S-3D 上 >9.0 (Nano Banana Pro: 9.61)，但 TS-Maze-2D 和 time-space settings 显著降低。这表明 local geometric plausibility 比 maintaining globally anchored spatial structure under motion 容易。

**(3) Fusion tasks 揭示真正的 world-model gap**
最强 separation 出现在 cross-consistency families (MT, MS, TS)。Leading models 在 MT-PropKeep 接近 ceiling，但在 MT-PropChange 和 TS-Maze-2D degrade。Nano Banana Pro 在 TS-Maze-2D 仅 4.46，表明即使 per-frame fidelity excellent，global world-state maintenance 仍未解决。

**(4) Constraint Backoff 现象普遍**
Table 10 的 Backoff column 揭示：models 倾向于将 unusual 或 strict constraints 替换为更 common defaults。Closed-source image models 此行为略轻 (~1.6-1.8)，但仍然 non-trivial。

### 6.4 Cross-Axis Consistency 深度分析

**Modal-Space Consistency** (Figure 35):
- **Act-align 和 Pos-rel 是主导瓶颈**：即使 geometry-leaning cues 高分 (Id-stab, Scene)，constraint 仍可能 attached to wrong object
- **Neg-rel 和 Excl. 通常强于 Pos-rel**：models 更可靠地 avoid forbidden configurations，而非 enforce exact required placement
- **Sem-MV tail-risk**：weaker models 在 multi-view stability anchors 上 catastrophic failure

**Modal-Time Consistency** (Figure 36):
- **Anchoring 通常强**，variance 集中在 weakest systems
- **Attr-Dyn 是主要瓶颈**：Env-stab 高，但 Follow 和 Rate 显著低，表明 models 保持 scene stable 但不可靠地 execute instructed evolution
- **Trigger Event**：Pre-hold 较强，但 Trigger 和 Post-comp degrade，event 在正确时间不 salient，post-trigger state 不 persist

**Time-Space Consistency** (Figure 37):
- **Maze-2D 是最 sharp discriminator**：Legal 和 Struct-stab 可能 moderate，但 Start/Goal 和 Correct 接近零，表明 core failure 是无法 maintain single identifiable trajectory from correct anchor to correct goal
- **Occlusion-under-motion 较成熟**：errors 集中在 Occ-move 和 Parallax (depth-layer updates)
- **3D Loop Navigation**：geometry 可以 frame-by-frame plausible，但 sequence 无法解释为单一 3D scene 沿 continuous camera path 遍历

## 7. 深层 Intuition: 从 Generator 到 World Simulator 的跃迁

### 7.1 Action Space 范式演进 (Figure 50)

论文结论部分提出了 World Model paradigms 按 interactive action space expressiveness 的演进：

**Vector-as-Action** (JEPA): latent space prediction，interaction mechanisms opaque，缺乏 semantic interpretability

**Key-as-Action** (Genie series): 引入 limited interactivity，但 confined to narrow, discrete, predefined action spaces

**Prompt-as-Action** (forward-looking): UMMs with modal consistency + video generation models with spatial-temporal consistency unified。Equipped with internal semantic compiler，interpret high-dimensional natural-language prompts 并 translate 为 universal spatiotemporal simulations adhering to Trinity of Consistency。

### 7.2 Constraint Backoff 的结构根源

论文的关键 insight：constraint backoff 不仅是 insufficient training data 的后果，而是 current models represent interaction 的 structural artifact。当 action space 不可解释 (uninterpretable) 或 rigidly predefined 时，models 缺乏 expressive capacity 来 ground semantic commitments in physical dynamics。在这种 constraints 下，consistency violations 不是 accidental errors，而是 almost inevitable outcomes。

这要求 **paradigm shift in interaction formalization**，而非单纯的 scale expansion。

### 7.3 Physical Authenticity 的 Differentiability 问题

现有 diffusion 和 AR architectures 以 pixel-level 或 token-level likelihood maximization 为最高目标，导致 generation results 落入 visual plausibility trap：
- Rigid bodies 无 support 悬浮
- Fluid momentum 不守恒
- Elastic coefficients 随 gestures 漂移

Future challenge：如何 embed Hamiltonians, conservation laws, 或 differential equations 作为 soft constraints 甚至 differentiable operators 进入 loss function，forcing network 从 painting the skin 到 painting the bones。

### 7.4 Long-term Causal Chains 的 Butterfly Effect Brittleness

当前 spatiotemporal attention 只能 maintain short-range memory for tens of seconds。一旦进入 hour-day scale，object identity consistency 和 event logic 因 error accumulation 雪崩式失败。

提议的 hierarchical implicit dynamics:
- **Macro level**: abstract causality via symbolic narratives 或 scene graphs
- **Meso level**: sparse 4D representations 压缩 event nodes
- **Micro level**: high-dimensional attention 完成 texture details

实现 multi-clock mechanism: slow variable fidelity + fast variable sampling。

### 7.5 从 Physical Simulator 到 General World OS

最终 vision: World Model 不应止于 physical sandbox，应成为容纳 autonomous agents evolution 和 gaming 的 Matrix：
- **Multi-agent gaming**: modeling social causality，deducing Nash Equilibrium dynamics under 不同 policies $\pi_i$ 的交互
- **GUI agents cross-domain generalization**: 从 3D physical world 扩展到 2D digital environments，理解 screen layouts 的 functional semantics 和 API calls 的 state transition logic $P(S_{t+1}^{screen} | S_t^{screen}, a_{ui})$

这标志着 world model 从 pure physical simulator 进化为 General World OS，encompassing both physical 和 digital attributes。

## 8. 个人 Intuition 与联想

### 8.1 与 LeCun JEPA 的对比

LeCun 的 JEPA [2022] 采用 non-generative prediction paradigm：$Pred(Enc(x_t), z) \approx Enc(x_{t+1})$，在 abstract representation space model state transitions，提供 denoised, efficient planning space。这与论文中提到的 "Vector-as-Action" paradigm 对应。

Trinity of Consistency 框架的优势在于提供了更细粒度的 decomposable analysis framework，而 JEPA 更强调 representation learning 的 abstract level。两者本质上是 complementary：JEPA 提供了 abstract planning space，Trinity of Consistency 提供了 verification 的 dimensions。

Reference: [LeCun JEPA](https://openreview.net/pdf?id=BZ5a1r-kVsf)

### 8.2 与 Genie 系列的关系

Google DeepMind 的 Genie 1/2/3 通过 unsupervised Latent Action Model 从 massive unlabeled videos 中 decouple discrete action tokens，将 probabilistic modeling 从 $P(x_{future} | x_{past})$ 转向 controlled state transitions $P(s_{t+1} | s_t, a_t)$。

这代表了论文中 "Key-as-Action" paradigm。Genie 的 limitation 在于 action space 的 predefined discrete nature，限制了 semantic expressiveness。论文提出的 "Prompt-as-Action" paradigm 通过 UMMs 的 internal semantic compiler 突破此限制。

Reference: [Genie](https://arxiv.org/abs/2401.15402), [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)

### 8.3 与 DreamerV3 的互补性

DreamerV3 [Hafner et al., 2024] 与 V-JEPA 类似，采用 non-generative prediction paradigm，在 abstract representation space model state transitions。这与论文中 "Vector-as-Action" paradigm 对应。

DreamerV3 的 advantage 在于 long-horizon planning 的 stability，但 disadvantage 在于缺乏 explicit physical constraints verification。Trinity of Consistency 框架可以作为 DreamerV3 的 evaluation 和 verification framework。

Reference: [DreamerV3](https://arxiv.org/abs/2301.04104)

### 8.4 与 NVIDIA Cosmos 的关系

NVIDIA Cosmos [2025] embeds rigid/fluid dynamics equations into Transformer masks 以实现 industrial-grade physical simulation。这代表了论文中 explicit geometric anchoring 与 temporal consistency 的深度 integration 尝试。

Cosmos 的 approach 体现了论文中提到的 "embedding differential equations as differentiable operators" 的 future direction，但仍限于 specific physics types (rigid/fluid)，缺乏 general causal reasoning capability。

Reference: [Cosmos](https://arxiv.org/abs/2501.03511)

### 8.5 Autoregressive-Diffusion Hybrid 的数学本质

论文中 Autoregressive-Diffusion Hybrid Model 的 joint probability density：
$$p_\theta(x_{img,1:T}, z_{1:T} | c) = \prod_{t=1}^T \underbrace{p_{AR}(z_t | z_{<t}, x_{img,<t}, c)}_{\text{Causal Temporal Dynamics}} \cdot \underbrace{p_{Diff}(x_{img,t} | z_t, x_{img,<t}, c)}_{\text{Conditionally Denoised Rendering}}$$

- $x_{img,1:T}$: generated multi-modal sequence (Video/Audio)
- $z_t$: noisy latent 或 intermediate features
- $p_{AR}$: low-dimensional causal modeling
- $p_{Diff}$: high-fidelity conditional denoising distribution

这是 causal logic 与 high-quality generation 的 joint probability density construction，超越了 simple probability superposition。Show-o, HybridVLA 等工作展示了此范式在 Embodied AI 领域的潜力。

### 8.6 Counterfactual Reasoning 的缺失

论文指出 current benchmarks 缺乏 causal probes for active intervention。True world cognition 必须经历 intervener mode 的 test：
> "If the support is removed at this moment, how will the object trajectory $\tau$ change?"

这要求 parameterized interventions 的 evaluation interface，验证 model 是否 constructed structured causal graph，还是仅 performing pixel-level probabilistic completion。

Judea Pearl 的 causal hierarchy (association → intervention → counterfactual) 在 world models 评估中尚未充分体现。CoW-Bench 是朝此方向的重要一步，但仍局限于 observational evaluation。

Reference: [Pearl Causality](https://www.cambridge.org/core/books/causality/)

## 9. 总结：Trinity of Consistency 的深层意义

这篇 paper 的核心 conviction 简单而不妥协：**consistency 不是一个 world model 的 optional attribute，而是其存在的 criterion**。一个产生 visually compelling pixels 但 fails to maintain cross-dimensional consistency 的 system，无论 scale 如何，本质上仍是 texture synthesizer 而非 world simulator。

Trinity of Consistency 不仅仅是一个 analytical framework，它划定了 boundary - 一个 paradigmatic divide：生成 resemble the world 的 images 与构造 understand the world 的 models。

对于 AGI 的 pursuit，这提供了清晰的技术 roadmap：
1. **Short-term**: 在现有 MM-DiT 架构中加强 cross-consistency 的 joint training
2. **Mid-term**: 发展 Prompt-as-Action paradigm，实现 natural language 到 universal spatiotemporal simulations 的 semantic compilation
3. **Long-term**: 构建 General World OS，encompassing physical 和 digital attributes，支持 multi-agent gaming 和 cross-domain generalization

CoW-Bench 作为评估工具，operationalize consistency 为 strict constraint-satisfaction problem，暴露了 pervasive constraint-backoff phenomenon，为区分 visual mimicry 和 genuine physical simulation 提供了必要的 diagnostic resolution。

Reference: [CoW-Bench Paper](https://arxiv.org/), [Sora](https://openai.com/research/video-generation-models-as-world-simulators), [Genie 3](https://deepmind.google/technologies/genie-3/), [3DGS](https://arxiv.org/abs/2308.14737), [NeRF](https://arxiv.org/abs/2003.08934)

---

这篇 paper 为 world models 领域建立了 principled theoretical framework，并通过 CoW-Bench 提供了 rigorous evaluation suite。它不仅是 technical survey，更是对未来 AGI 路径的 visionary statement：从 passive observer 到 proactive simulator 的跃迁，需要的是 architectural unification 而非 component refinement。Trinity of Consistency 正是这个 unification 的 guiding principle。
