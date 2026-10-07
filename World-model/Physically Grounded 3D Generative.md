---
source_pdf: Physically Grounded 3D Generative.pdf
paper_sha256: f625e6f66b8397a9193698bfd96042e108d005962bd32ea5de7afa0ab3006cce
processed_at: '2026-08-06T03:26:57-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

## 一句话总结

机器人抓东西的时候,手把物体挡住了大半,光靠摄像头根本猜不出物体长啥样。这篇 paper 说:别只靠眼睛,用手本身的感觉——手在哪、手指碰到了哪——来补上眼睛看不见的部分,而且要把这些"物理事实"当成硬约束塞进 3D 生成模型里,不让模型瞎猜出违反物理的形状。

---

## 为什么这事难

你想想机器人抓个杯子。从头顶摄像头看下去,杯子被五根手指盖住了一大半,可能只剩个把手露在外面。你让模型从这一小截把手还原整个杯子,模型会怎么办?它只能靠以前见过的杯子形状去猜。猜出来的东西可能看着像杯子,但往机器人手上一放——穿过去了,杯子跟手指交叉在一起。这种重建对控制完全没用,你不能拿一个穿透手的"杯子"去规划下一步动作。

问题本质是:单张图片到 3D 形状,信息量根本不够,是 ill-posed 的。纯视觉的生成 prior 给你一个"看起来合理"的答案,但"看起来合理"和"物理上可行"是两码事。

---

## 关键 insight

作者抓住一个特别朴素的事实:**机器人在抓东西的时候,手本身就在不断产生关于物体的物理信息**。

- 手的关节角度(proprioception)告诉你手长什么样、摆在哪。你知道手在哪,就知道物体至少不能在手占据的空间里。
- 手指尖的触觉告诉你手指碰到了物体表面。碰到的那几个点,就是物体表面必须经过的位置。

这些信息恰好出现在视觉缺失的地方——手挡住的区域。视觉告诉你"露出来的部分长这样",触觉告诉你"藏起来的部分表面必须经过这个点",手的几何告诉你"藏起来的部分不能在我身体里"。三种信息互补,把原本 under-constrained 的问题变成了 over-constrained。

---

## 怎么把三种信息对齐到一起

这是整个 paper 最聪明的地方。作者选了一个统一的 3D 坐标系:一个 $64^3$ 的体素网格,z 轴对准摄像头方向。所有东西都放进这个网格里:

- 物体用 SDF(signed distance field)表示——每个体素存一个数,正数表示在物体外面多远,负数表示在物体里面多远,零就是表面。
- 手也用 SDF 表示,放进同一个网格。手在哪,网格里对应位置就是负的。
- 触觉接触点也标到这个网格里,哪个体素有接触就标 1。

三个模态在同一个 $64^3$ 网格里对齐了,几何关系天然成立。物体的 SDF 在手 SDF 的负区域里不能也是负的(否则穿透),在接触点处应该接近零(表面经过这里)。这两个约束用 SDF 表达出来特别干净。

---

## 生成模型怎么搭

整体架构继承自 TRELLIS 那篇 paper,分两步:

**第一步:训一个 VAE 压缩 SDF。** $64^3$ 的 SDF 太大了,直接在上面跑 diffusion 很贵。先用一个 3D 卷积 autoencoder 把它压到一个小的 spatial latent,再从这个 latent 解码回 SDF。训练时除了重建误差,还加了两个几何约束:Eikonal 约束保证 SDF 梯度的模长是 1(这样它真的是距离场,不是随便什么标量场),还有表面附近的 normal 一致性约束。

**第二步:在这个 latent space 里跑 flow matching。** flow matching 跟 diffusion 类似,都是从噪声到数据的生成过程,但轨迹是直线,采样更快。模型的条件输入是:DINOv2 编码的图像证据(visible RGB × object mask)、object mask、hand mask、手和触觉的融合 latent。

预训练用干净的单物体图像,学一个通用的形状先验。微调时用模拟的抓取场景,物体被手遮挡,这时加入 proprioception 和 touch 条件,同时加上物理约束 loss。

---

## 物理约束怎么加进去

两个 loss,一个推一个拉:

**Non-interpenetration loss:不让你穿透手。** 物体 SDF 在手内部的体素处,如果物体 SDF 也是负的(物体也在这里),就 penalize。这里有个工程细节:直接用 SDF 值的话,穿透越深梯度越大,训练会爆炸。作者用一个 tanh saturation 把 SDF 在 0.1 处截断,深穿透不再给更大的惩罚,梯度保持稳定。

**Contact-consistency loss:表面必须经过接触点。** 触觉标出接触的体素处,物体 SDF 的绝对值应该接近零(表面在这里)。这个 loss 就是接触点处 $|SDF|$ 的平均值。

还有个时间加权 $w(t) = (1-t)^2$:flow matching 早期 $t$ 大,latent 全是噪声,decode 出来的 SDF 也是垃圾,这时候加物理 loss 没意义反而有害。到后期 $t$ 小了,形状逐渐成形,物理约束才开始发力。

---

## 推理时的物理引导

训练时加 loss 是一回事,推理时还能再加一层。作者在 flow matching 的采样循环里插了一个"修正项"。

标准采样每步是:当前 latent 减去时间步长乘以模型预测的速度场。作者改成:当前 latent 减去时间步长乘以(模型预测速度 + 修正项 $\theta_k$)。

这个 $\theta_k$ 怎么来的?每一步把当前 latent decode 成 SDF,算物理 loss 的梯度,用 EMA 平滑一下累积起来,作为修正项。方向很直觉:梯度告诉你往哪个方向调整 latent 能减少穿透、让表面更接近接触点,那就往那个方向推一点。

有几个 stabilization trick:梯度归一化防失控,trust region 限制修正项大小别盖过模型自己的速度场(否则 prior 被破坏,形状会变奇怪),还有可选的投影防止接触能量增加。paper 里 Figure A3 展示了采样轨迹:前几步形状太乱,修正没效果;到中后期形状稳定了,修正项把表面拉向接触点、推离手指内部,形状逐渐变得物理合理。

---

## 实验说了什么

**模拟实验**,用 YCB 数据集 36 个物体,按遮挡程度分 5 档:

跟纯视觉的 Amodal3R 比,全面碾压,尤其重遮挡时差距巨大。Amodal3R 在 80-100% 遮挡时 Chamfer Distance 是 0.439,本文是 0.109,差四倍。

跟 SAM3D 比(SAM3D 用了更大的模型和更多的数据),轻遮挡时两者差不多,SAM3D 甚至偶尔略胜。但遮挡一重,本文明显领先,因为物理约束开始发挥作用。

**Pose 估计**:本文方法不需要 depth,靠 proprioception + touch 拿到 ADD-S@0.1 = 0.73,SAM3D 用 ground truth depth 才 0.658。原因是物体重建在手的坐标系里,pose 是隐式副产品。

**Ablation**:去掉 touch 只留 proprioception,CD 从 0.033 退到 0.085;再去掉 proprioception 只留 vision,退到 0.142。两个模态贡献量级相当,缺一不可。

**触觉噪声**:3mm 扰动几乎无影响,5mm 才有轻微退化。因为 $64^3$ 网格在典型场景下体素大小就是几个毫米,小扰动被量化吸收了。

**真机实验**:在 iCub 人形机器人上跑,用了一个训练时没见过的五指手,装了 XELA 磁触觉传感器。结果还是赢过 SAM3D,证明能跨 embodiment 泛化。但 calibration 误差会导致手和物体对不齐,Figure A2 展示了这个问题。

---

## 这篇 paper 好在哪

我觉得最优雅的设计是那个 camera-aligned 的 $64^3$ 统一网格。所有模态都在同一个几何空间里,对齐不需要额外的网络去学,几何关系天然成立。SDF 作为表示让物理约束变得数学上干净:穿透就是两个 SDF 在同一位置都是负的,接触就是物体 SDF 在接触点处是零。

把物理约束同时用在训练 loss 和推理引导上,两者用同一个能量函数,一致性高。而且推理引导不需要额外网络,直接 decode 算梯度就行,计算开销很小。

真机实验不是摆设,真的在没见过的手上跑通了,说明方法不依赖具体手形态,靠的是通用的 SDF 编码 + 前向运动学。

---

## 局限和我想的问题

$64^3$ 太粗了,叉子的齿、剪刀的刃这种细结构直接丢了,Stage B 也只能部分恢复。用多分辨率或者 octree 应该能解决,但实现复杂度会上去。

依赖准确的 calibration。camera-hand 外参和 forward kinematics 有误差,手和物体就对不齐,所有物理约束都会带偏。真机实验里这个问题已经出现了。也许可以用触觉残差反过来 refine calibration,但 paper 没做。

只用 binary touch 浪费了 XELA 传感器的力信息。如果有力的方向和大小,可以推断摩擦、柔度,甚至推演动力学,但现在只用了"碰到/没碰到"。

两阶段还是 sub-optimal,Stage A 给 SDF,Stage B 再 refine,中间有信息损失。端到端可微应该更好,但作者保留 modular 设计是为了能 plug 进不同 refinement pipeline,这个 trade-off 合理。

单视图是真机操作的硬约束,但多视图肯定会进一步降 ambiguity。如果头上有多个 camera,或者眼动相机,这个方法应该能直接扩展,只要每个 view 都做同样的 camera-aligned grid。

---

## 我的直觉总结

这篇 paper 给我的最大启发是:generative model 在机器人场景里,光有视觉 prior 不够,必须 grounded in 物理。物理信息不一定要从物理引擎来,机器人自身的 proprioception 和 touch 就是现成的物理证据,而且它们恰好出现在视觉失效的地方。把这些证据变成 SDF 上的数学约束,既能在训练时 regularize 生成分布,又能在推理时 guide 采样方向,一个框架两种用法,数学上很统一。

这跟 physical AI 的大方向完全一致——生成模型不能只追求"看起来像",必须"物理上对"才能用于 control。这篇 paper 在 perception 端把这件事做实了。

---

# Physically Grounded 3D Generative Reconstruction under Hand Occlusion — 深度技术剖析

这篇 paper 来自 Istituto Italiano di Tecnologia (IIT) 的 Humanoid Sensing & Perception lab, 第一作者 Gabriele Caddeo 长期做 vision-tactile fusion(Sim2Surf, FeelAnyForce)。核心思想是把 **proprioception + multi-contact touch** 作为几何约束注入到一个 **flow-matching 3D generative model** 里, 解决机器人 in-hand manipulation 时单视图 RGB 重建的 ill-posedness 问题。和 Amodal3R、SAM3D 这类纯 vision 的 generative 3D pipeline 相比, 它走的是 "physics-grounded generation" 路线, 与 DSO (https://arxiv.org/abs/2410.06124) → 我们这条线一脉相承。

下面我从 representation、Stage-A 架构、physics injection、inference-time guidance、实验数据几个层面拆给你 build intuition。

---

## 1. 核心问题与设计直觉

机器人抓取时 hand 严重遮挡 object, 单视图 RGB → 3D 是一个 fundamental under-constrained 问题。Vision-only generative prior 会输出 "看起来 plausible 但物理上不可用" 的形状: 穿透 hand、scale drift、contact 区域缺失。作者抓住一个关键 insight: **manipulation 本身就在产生物理证据** — proprioception 给 hand 的 posed geometry, touch 给 surface 必须经过的点集, non-interpenetration 给 object 不能进入的体积。这些 cue 恰好出现在 vision 缺失的地方。

这跟 tactile-only 重建(如 TouchSDF https://ieeexplore.ieee.org/document/10517503)正好互补: touch 是 local 的, vision 是 global 的, 两者在 occluded region 形成 "global shape prior + local surface constraint" 的互补结构。

---

## 2. Representation: Pose-Aware Camera-Aligned SDF

这是整篇 paper 最关键的 design choice, 直接决定了后续多模态融合能不能 "对齐"。

### 2.1 为什么是 SDF 而不是 occupancy / radiance field

作者选 **signed distance field** $S_o : \Omega \to \mathbb{R}$, 在 $\Omega = [-1, 1]^3$ 上离散到 $R^3$ 网格($R=64$)。zero level set $\{\mathbf{x} : S_o(\mathbf{x}) = 0\}$ 定义 surface。

SDF 相比 occupancy 的优势在 paper 里通过两个 regularizer 体现:
- **Eikonal**: $\|\nabla S(\mathbf{x})\|_2 \approx 1$ almost everywhere, 这让 SDF 是真正的距离场, 而不是任意标量场。这个性质使得 contact loss $\mathcal{L}_C = |S_o(\mathbf{x})|$ at contact voxels 直接具有 "到 surface 的几何距离" 的物理含义。
- **可微分**: physics guidance 需要 $\nabla_{x_k} E(x_k)$, SDF 的梯度场天然提供 surface normal $\nabla S / \|\nabla S\|$, 这是非-interpenetration 推力方向的天然来源。

### 2.2 Camera-aligned grid 的精妙之处

**Grid +z 轴 = camera viewing direction**, grid 旋转作为 metadata 存储。这看似只是一个坐标系约定, 实际有三个重要作用:

1. **2D-3D 像素一致性**: 把 SDF 沿 grid z 轴 projection 与 conditioning image 对齐, 不需要额外的视角变换网络, 直接 geometrically consistent。
2. **Pose-aware data augmentation**: 公式 (1) 的 in-grid similarity augmentation
   $$S_o^{\text{aug}}(\mathbf{x}) = s_{aug} \cdot S_o\left(\frac{\mathbf{R}_g^\top(\mathbf{x} - \mathbf{t})}{s_{aug}}\right)$$
   这里 $\mathbf{R}_g \in SO(3)$ 是 grid 旋转, $\mathbf{t} \in \mathbb{R}^3$ 是 grid 内平移, $s_{aug} \in [0.5, 1.0]$ 是 downscaling factor。这个 augmentation 模拟了 grasping scene 中 object 不一定居中、不一定 tightly fit to grid 的真实情况 — 在 grasp scene 里 object 和 hand 共享同一个 grid, object 的 size/location 由 real-world scale 和 hand-object relative pose 决定。这一步对应 supplementary A.1 的 "pose-consistent sprite placement", 把 rendered RGBA sprite 按 grid 中的 bounding box 重新 paste 到 conditioning canvas 上, 保证 image 和 SDF 在像素级对齐。
3. **隐式 pose 估计**: 因为 object 重建在 hand 的同一 3D domain, pose 被 implicitly recover, 不需要单独的 pose head。Table 2 显示 ADD-S@0.1 = 0.73 vs SAM3D 的 0.658, 就来自这个隐式约束。

### 2.3 Non-watertight mesh 处理

用 Dual Octree Graph (Wang et al., SIGGRAPH 2022, https://arxiv.org/abs/2110.00189) 处理 non-watertight mesh 的 SDF 计算 — 这是真实物体(如 fork、clamp)必须面对的问题, 不能假设 watertight。

---

## 3. Stage A: Structure-VAE + Conditional Flow Transformer

整体架构继承 TRELLIS (Xiang et al., 2024, https://arxiv.org/abs/2412.01506), 但做了三处关键改造。

### 3.1 Structure-VAE: 学一个 SDF 的 latent space

Encoder $E_\phi$ 输出 diagonal Gaussian:
$$(\boldsymbol{\mu}, \log \boldsymbol{\sigma}^2) = E_\phi(S), \quad \mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

latent grid $\mathbf{z} \in \mathbb{R}^{C \times R' \times R' \times R'}$, 注意这是 **spatial latent**(像 VQ-GAN / latent diffusion 那样保留 spatial 结构), 而非全局 vector latent。Decoder $D_\theta$ 重建 SDF: $\widehat{S} = D_\theta(\mathbf{z})$。

**VAE Loss (Eq. 7)**:
$$\mathcal{L}_{\text{VAE}} = \lambda_{\text{L1}} \mathcal{L}_{\text{L1}} + \lambda_{\text{eik}} \mathcal{L}_{\text{eik}} + \lambda_{\text{n}} \mathcal{L}_{\text{n}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}}$$

各项直觉:
- $\mathcal{L}_{\text{L1}} = \frac{1}{|\Omega|}\sum_{\mathbf{x}} |\widehat{S}(\mathbf{x}) - S(\mathbf{x})|$: 用 L1 而非 L2, 对 SDF 这种有尖锐 zero-crossing 的场更鲁棒。
- $\mathcal{L}_{\text{eik}} = \mathbb{E}_{\mathbf{x}}(\|\nabla \widehat{S}(\mathbf{x})\|_2 - 1)^2$: 强制 Eikonal 方程, 保证 SDF 的几何意义。用 3×3×3 finite-difference stencil 实现, **只在 interior mask 上算**(drop one-voxel rim)避免 padding 边界伪影。
- $\mathcal{L}_{\text{n}} = \mathbb{E}_{\mathbf{x}: |S(\mathbf{x})| < 2h}\left(1 - \frac{\nabla \widehat{S}}{\|\nabla \widehat{S}\|} \cdot \frac{\nabla S}{\|\nabla S\|}\right)$: **near-surface normal consistency**, 只在 $|S| < 2h$ 的 narrow band 内约束($h$ 是 bandwidth half-width), 把"算力"集中在真正定义 surface 的 region。这是个很常用的 NeuS / VolSDF 风格 trick。

### 3.2 Flow Transformer 预训练

用 **flow matching** (Lipman et al., https://arxiv.org/abs/2210.02747) 而非 DDPM, 主要是为了 sampling 速度快(直线 trajectory vs 弯曲的 diffusion trajectory)。

Flow matching 训练目标 (Eq. 8):
$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_0, t, \epsilon}\left[\left\|f_\theta(x_t, \alpha t; I) - ((1-\sigma_{\min})\epsilon - x_0)\right\|_2^2\right]$$

变量含义:
- $x_0 \in \mathbb{R}^{C \times R' \times R' \times R'}$: 目标 latent grid(从 Structure-VAE encoder 得到)
- $t \in [0, 1]$: flow matching time, $t=0$ 是 clean, $t=1$ 是 noise
- $\alpha$: 时间 rescale 因子, $t \to \tau = \alpha t$ before timestep embedding
- $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: 噪声
- $x_t$: 线性插值 $x_t = (1-t)x_0 + t\epsilon$, 但用 $\sigma_{\min}$ 保留一点 minimum noise 防止 $t=0$ 退化
- $f_\theta$: denoiser, 预测 velocity field $v = (1-\sigma_{\min})\epsilon - x_0$(target velocity)
- $I$: conditioning image, 通过 DINOv2 (Oquab et al., https://arxiv.org/abs/2304.07193) 编码

预训练只在 vision-only clean object images 上做, 学一个 strong shape prior。

### 3.3 微调: 多模态条件融合

这是这篇 paper 的核心创新。grasp scene 提供 $(I, M_o, M_h)$, 微调时 conditioning 包括:

1. **Visible RGB evidence**: $I_o = I \odot M_o$ — 把 object mask 外的像素清零, 只保留 visible 部分。同时把 $(M_o, M_h)$ 两个 mask 也喂进去, 让 model 知道哪里是 occluded。这跟 Amodal3R (https://arxiv.org/abs/2503.13439) 的设计一致。

2. **Proprioception → Hand latent**: 
   - Forward kinematics 得到 hand mesh $\mathcal{M}_h$, rasterize 成 SDF grid $S_h$
   - 用 **同一个 Structure-VAE encoder** 编码 $S_h$ 得到 $x_{0,h} = E_\phi(S_h)$ — 这是一个非常聪明的 reuse, 让 hand 和 object 在同一个 latent space 里, 几何对齐自然成立。

3. **Touch representation** $T = [C, D]$:
   - $C \in \{0,1\}^{R^3}$: binary contact-occupancy grid, sparse
   - $D \in \mathbb{R}^{R^3}$: 每个 voxel 到最近 contact voxel 的距离, dense smooth field
   
   这个双通道设计很关键。$C$ 本身太 sparse, 直接喂 CNN 学不到东西; $D$ 把 sparse contact "diffuse" 成 dense field, 让 contact 信息能 propagate 到邻近 voxel, 帮助 learning。这本质是一个 **geodesic distance transform** 思路, 跟 ShapeFormer、PVD 的 chamfer-distance field 类似。

4. **Fusion (Eq. 9)**:
   $$\widetilde{x}_{0,h} = \text{Conv}_{1\times1\times1}\left([x_{0,h}, g_\psi(T)]\right)$$
   $g_\psi$ 是 lightweight 3D CNN, 把 $T$ 编码成 feature volume, 与 hand latent $x_{0,h}$ 在 channel 维 concat, 用 1×1×1 conv 投影回原 channel 数。当 touch disable 时 $g_\psi(T) \to 0$, $\widetilde{x}_{0,h} \to x_{0,h}$, 这个 **residual-like design** 让 ablation 很干净。

5. **Cross-attention 注入**: $\widetilde{x}_{0,h}$ patchify 成 tokens, 通过额外的 cross-attention stream 喂给 denoiser, 与 image-based conditioning 并列。这是 multi-stream cross-attention, 一个 stream 给 vision, 一个 stream 给 hand+touch。

---

## 4. Physics-Aware Losses: 把物理约束变成可微目标

### 4.1 Non-Interpenetration Loss $\mathcal{L}_{\text{NI}}$

最关键的 trick 是 **smooth saturation**:
$$\psi_\tau(s) = \tau \tanh(\text{ReLU}(s)/\tau), \quad \tau = 0.1$$

为什么需要 saturation? 如果直接用 $-\widehat{S}_o$ 当 "interior mass", 当 object 深深穿透 hand 时 $\widehat{S}_o \to -\infty$, gradient 会爆炸。$\psi_\tau$ 把 $-S$ 在 $[0, \tau]$ 内线性增长, 之后 saturate 到 $\tau$, gradient 保持 bounded。

定义:
- $A_o(\mathbf{x}) = \psi_\tau(-\widehat{S}_o(\mathbf{x}))$: object 的 saturated interior field
- $M_h(\mathbf{x}) = \mathbf{1}[\psi_\tau(-S_h(\mathbf{x})) > 0]$: hand 体积的 binary mask

Non-interpenetration loss (Eq. 12):
$$\mathcal{L}_{\text{NI}} = \frac{1}{B}\sum_{b=1}^B \frac{\sum_{\mathbf{x}} A_{o,b}(\mathbf{x}) M_{h,b}(\mathbf{x})}{\max(1, \sum_{\mathbf{x}} M_{h,b}(\mathbf{x}))}$$

分母用 $\max(1, \cdot)$ 防 zero-division(hand 完全在 grid 外时), 整体是 "object 在 hand 内的平均质量"。

### 4.2 Contact-Consistency Loss $\mathcal{L}_{\text{C}}$

$$\mathcal{L}_{\text{C}} = \frac{1}{B}\sum_{b=1}^B \frac{\sum_{\mathbf{x}} C_b(\mathbf{x})|\widehat{S}_{o,b}(\mathbf{x})|}{\max(1, \sum_{\mathbf{x}} C_b(\mathbf{x}))}$$

直觉: contact voxel $C(\mathbf{x}) = 1$ 处, surface 必须经过, 所以 $|\widehat{S}_o(\mathbf{x})| \to 0$。因为 SDF 的几何含义, $|S_o|$ 就是到 surface 的距离, 这个 loss 直接说 "contact 点到预测 surface 的距离要小"。

### 4.3 Time-weighted Aggregation

$$w(t) = (1-t)^2$$

这是个非常关键的 design。在大 $t$(高噪声)时, decoder 输出的 SDF 几乎是 noise, 此时 physics loss gradient 是垃圾。$w(t) = (1-t)^2$ 在 $t \to 1$ 时压到 0, 在 $t \to 0$(clean)时给满权重。这跟 classifier-free guidance 里训练时 dropout condition 类似 — 在生成不同阶段施加不同强度的约束。

### 4.4 总微调目标 (Eq. 14)

$$\mathcal{L} = \mathcal{L}_{\text{FM}} + \lambda_{\text{NI}} \mathcal{L}_{\text{NI}} + \lambda_{\text{C}} \mathcal{L}_{\text{C}}$$

$\lambda_{\text{NI}}$ warmup, $\lambda_{\text{C}}$ fixed。warmup 的原因: 一开始 model 还没学好 shape prior, 直接 penalize interpenetration 会 push 出 weird shapes; 等 prior 稳定后再加 physics constraint。

---

## 5. Inference-Time Physical Guidance: 训练时学 prior, 推理时加约束

这是 paper 最 mathematically elegant 的部分, 借鉴了 "Training-Free Guided Flow Matching with Optimal Control" (Wang et al., https://arxiv.org/abs/2410.18070)。

### 5.1 Modified Euler Update (Eq. 15)

标准 flow matching sampling: $x_{k+1} = x_k - \Delta t_k \cdot v_\theta(x_k, t_k; \text{cond})$

本文加了一个 **additive control term** $\theta_k$:
$$x_{k+1} = x_k - \Delta t_k \cdot (v_\theta(x_k, t_k; \text{cond}) + \theta_k)$$

### 5.2 Physics Energy & Gradient

定义 physics energy (Eq. 16):
$$E(x_k) = \lambda_{\text{NI}} \mathcal{L}_{\text{NI}}(\widehat{S}_o, S_h) + \lambda_{\text{C}} \mathcal{L}_{\text{C}}(\widehat{S}_o, T)$$

每个 step:
1. 用 frozen Structure-VAE decoder 把当前 latent $x_k$ decode 成 $\widehat{S}_o$
2. 计算 $g_k = \nabla_{x_k} E(x_k)$ — 通过 SDF 的 chain rule 反传到 latent space
3. EMA 更新 control term (Eq. 17):
   $$\theta_{k+1} = \beta \theta_k + \eta g_k, \quad \beta \in (0, 1), \eta > 0$$

EMA 的作用: 单步 gradient 可能 noisy/oscillatory, EMA 平滑 control term, 相当于一个 momentum-based guidance。

### 5.3 Stabilization Heuristics

- **Gradient normalization**: 防止 $g_k$ 量级失控
- **Trust-region on $\|\theta_k\|$ relative to $\|v_\theta\|$**: control term 不能 dominate learned velocity, 否则 prior 被覆盖, 输出会偏离 natural shape distribution
- **Optional projection to prevent increasing contact energy**: 只允许减少 contact energy 的更新方向

Fig. A3 展示了一个典型 trajectory: 早期($k<5$) SDF 太 noisy, guidance 效果有限; 中后期 guidance 把 surface 拉到 contact 点、推出 hand 内部。这跟 classifier guidance 在 diffusion 里的 "前期 shape 大体定, 后期 refine detail" 一致。

---

## 6. Stage B: Texture & Refinement

Stage B 主要是个 integration demo, 证明 Stage A 的 metric-scale 物理一致输出能 plug 进现有 refinement pipeline。用 SLat (Structured Latents from TRELLIS, https://arxiv.org/abs/2412.01506), 在 camera-aligned canonical grid 里 decode, 再用已知 similarity transform 映回 world/camera frame。Conditioning 同样用 $I_o = I \odot M_o$ 和 $M_h$。

也可以替换成 SAM3D (https://arxiv.org/abs/2511.16624) 的 refinement stage。这种 modular 设计让 Stage A 可以独立改进, 而不绑死某个 refinement 方法。

---

## 7. 实验数据深度解析

### 7.1 主结果 Table 1 (3D Reconstruction)

按 occlusion bin $B_1$–$B_5$ 分层评估(0–20%, 20–40%, ..., 80–100%):

| Method | Bin | CD↓ | NC↑ | F@0.02↑ | Voxel-IoU↑ | EMD↓ |
|---|---|---|---|---|---|---|
| Amodal3R | All | 0.188 | 0.669 | 0.162 | 0.339 | 0.314 |
| SAM3D | All | 0.039 | 0.803 | 0.228 | 0.504 | 0.202 |
| **Ours** | **All** | **0.033** | **0.844** | **0.189** | **0.586** | **0.184** |

关键观察:
- **Amodal3R 在 heavy occlusion 下崩溃**: $B_5$ 的 CD = 0.439, NC 跌到 0.602。这是 vision-only occlusion-aware 方法在 80-100% occlusion 下的 fundamental limit。
- **vs SAM3D**: SAM3D 用更大模型 + 更大数据, 在 low occlusion ($B_1$) 下 F@0.02 = 0.265 略胜我们的 0.215。这是合理的 — 极轻遮挡下 vision prior 已经足够强, physics constraint 反而可能略带 bias。但 **从 $B_3$ 开始我们全面领先**, $B_5$ 时 SAM3D CD = 0.153 vs 我们 0.109, 差距 ~30%。
- **Voxel-IoU 0.586 vs SAM3D 0.504**: 这个 metric 对 volume overlap敏感, 我们的优势来自 non-interpenetration constraint 防止 object "膨胀"进 hand 体积。

### 7.2 Pose Estimation Table 2

| Method | 3D IoU↑ | ICP-Rot↓ | ADD-S↓ | ADD-S@0.1↑ |
|---|---|---|---|---|
| SAM3D | 0.406 | 18.875 | 0.095 | 0.658 |
| **Ours** | **0.530** | **10.0** | **0.07** | **0.73** |

ADD-S@0.1 = 0.73 vs 0.658, ICP-Rot 减半(18.9° → 10.0°), 这是显著提升。pose 估计的精度直接来自 "object 重建在 hand 的同一 domain" — 不需要单独配准, pose 是 by-product。

注意 SAM3D 这里用的是 **ground-truth depth** 算 point map 才能拿到 0.658; 我们不用 depth, 用 proprioception + touch 就超过它。如果 SAM3D 用 noisy depth, 性能会更差。这说明在真实机器人场景(没有 GT depth)我们的优势更大。

### 7.3 Sensing Ablation Table A1

| Variant | CD↓ | NC↑ | Voxel-IoU↑ |
|---|---|---|---|
| Vision-Only | 0.142 | 0.696 | 0.357 |
| No-Touch(+proprioception) | 0.085 | 0.760 | 0.430 |
| Full(+touch) | 0.033 | 0.844 | 0.586 |

CD: 0.142 → 0.085 → 0.033, **proprioception 贡献最大(0.057), touch 额外贡献 0.052**。两者量级相当, 互补性强。proprioception 提供 global constraint(知道 hand 在哪), touch 提供 local surface constraint(知道 surface 经过哪), 缺一不可。

### 7.4 Tactile Noise Robustness Table A2

3mm perturbation: CD 从 0.033 → 0.033 (negligible)
5mm perturbation: CD 从 0.033 → 0.036 (small degradation)

这个 robustness 来自 **grid 量化**: $R=64$ over $[-1,1]^3$ 给 voxel size ~3cm/64 ≈ 0.5mm 在 normalized space, 但映射到 typical scene(几十 cm)后 voxel size 约 3-5mm, 所以 3mm 噪声被 quantization 吸收。5mm 才开始 affect 多个 voxel。这是个 favorable property — 不需要 touch 极高精度。

### 7.5 Real-World Table 3

| Method | CD↓ | NC↑ | Voxel-IoU↑ |
|---|---|---|---|
| Amodal3R | 0.252 | 0.501 | 0.352 |
| SAM3D | 0.037 | 0.888 | 0.510 |
| Ours | **0.035** | **0.903** | **0.648** |

注意我们在 real-world 上 Voxel-IoU = 0.648 **甚至高于** simulation 的 0.586。这有点 surprising, 作者没解释, 我推测是:
1. Real-world test set 只有 5 个 object(sugar_box, mustard_bottle, strawberry, pear, bleach_cleanser), 都是 relatively simple shape, 比 YCB 36 object 平均简单。
2. Real tactile 给的真实 contact cue 可能比 simulated 更 informative(真实 sensor 有 force 量级, 模拟只有 binary)。
3. 但 calibration noise 会带来 hand-object misalignment, Fig. A2 展示了这个问题, Fig. 4 第三行有 artifact。

---

## 8. 关键 Insights & Connections

### 8.1 与 Amodal3R / TRELLIS 的关系

Amodal3R 是同一 architecture 但 vision-only, 在 Table 1 全面落后。说明在 heavy occlusion 下, **vision prior + mask 不够, 必须有 active physical sensing**。这呼应了 embodied AI 里 "perception for action vs action for perception" 的双向性 — manipulation 既消费 perception 又产生 perception cue。

### 8.2 与 Diffusion Guidance 的关系

Inference-time physical guidance 跟 **classifier guidance** (Dhariwal & Nichol, https://arxiv.org/abs/2105.05233) 和 **classifier-free guidance** (Ho & Salimans, https://arxiv.org/abs/2207.12598) 是同源思想, 但有两个关键差异:
1. 不是训一个额外 classifier, 而是用 **decoder + physics loss** 直接算 gradient, 不引入额外网络。
2. EMA + trust-region stabilization 是为 flow matching 特有的(velocity field 比 noise prediction 更敏感)。

跟 **DDS / SDEdit** 类方法比, 这里不需要额外 forward pass, 直接在 latent space 加 control term, computational overhead 极小。

### 8.3 与 Tactile-Only 重建的关系

TouchSDF (https://ieeexplore.ieee.org/document/10517503)、Snap-It-Tap-It-Splat-It (https://arxiv.org/abs/2411.17671) 等纯 touch 方法是 local 探索, 需要 active exploration policy。本文方法用 **single frame**, 不需要探索, 因为 vision 提供了 global shape prior。这是 vision-tactile fusion 的 sweet spot — vision 给 rough shape, touch 在 occluded region refine。

### 8.4 与 FoundationPose 的关系

FoundationPose (Wen et al., https://arxiv.org/abs/2312.08344) 是 vision-based 6DoF pose estimation 的 SOTA, 但需要 CAD model 或 few-shot reference images。本文方法 **不需要 object CAD**, 因为 generative prior 从大数据集学来。trade-off 是精度可能不如 model-based, 但 generalization 更强。Table 2 没跟 FoundationPose 比, 是个未来工作方向。

### 8.5 Cross-Embodiment Transfer

训练用 Barrett(3 finger)、ShadowHand(5 finger)、Allegro(4 finger) 三种 hand, real-world 用 iCub humanoid 的 5-finger hand with XELA sensors。**没在 test hand 上训练过**。这是 cross-embodiment generalization 的强 evidence, 因为 hand conditioning 是通过 **general SDF encoder + forward kinematics**, 不依赖具体 hand morphology。

### 8.6 Limitations & My Thoughts

1. **Fixed $R=64$ grid**: 对 thin structures(fork tine, scissor blade)会丢失。作者提到 Stage A 损失的 detail 在 Stage B 只能 partial recover。可以用 multi-resolution(像 XCube, https://arxiv.org/abs/2310.16757)或 octree-based representation。
2. **Calibration noise sensitive**: Fig. A2 显示 calibration error 直接导致 hand-object misalignment, 这是所有 proprioception-based 方法的通病。可以学一个 calibration refinement head, 或者用 tactile-residual 来 jointly refine calibration。
3. **Binary touch only**: XELA sensor 其实能给 force vector, 本文只用 binary contact。如果能用 force magnitude + direction, 可以推断 friction、compliance, 直接喂 dynamic model。
4. **Two-stage 离散**: Stage A 给 SDF, Stage B 再 refine。端到端可微应该更好, 但作者保留 modular 是为了 plug-and-play。
5. **Single view**: 多视图(眼动 + 多 camera)会进一步降 ambiguity, 但 single egocentric view 是 manipulation 时的真实约束。

---

## 9. 我对这篇 paper 的总体评价

这是一个 **well-executed 的 "physical AI × generative 3D" crossover paper**, 几个亮点:

- **Representation design 极其 coherent**: camera-aligned SDF + spatial latent + Eikonal regularizer 三个组件互相 reinforce, 让 multi-modal fusion 在几何上 naturally 成立。
- **Physics injection 双管齐下**: training-time loss + inference-time guidance, 两者用同一个 energy function, 一致性高。
- **Real-world 验证扎实**: 不只是 simulation demo, 真的在 iCub humanoid 上跑通, 用 unseen end-effector + real XELA tactile sensor。
- **Ablation 干净**: sensing modality 逐个加, noise level 测试, 都做得很 transparent。

可改进空间主要在 **resolution scalability**(64³ 太粗)、**end-to-end optimization**(两 stage 还是 sub-optimal)、**force-rich tactile**(只用 binary 浪费了 XELA 的能力)。但作为 v1, 已经是一个很强的 baseline, 给后续 "physical generative 3D for manipulation" 立了个标杆。

跟你的 lecture 里讲的 "physical AI" 方向高度一致 — generative model 不能只学 visual prior, 必须 grounded in physical constraint 才能用于 control。这篇 paper 把 proprioception 和 touch 作为 physical constraint 注入 generative 3D, 正是 physical AI 在 perception 端的具体实例化。

---

## 参考链接

- TRELLIS (Structured 3D Latents): https://arxiv.org/abs/2412.01506
- Amodal3R: https://arxiv.org/abs/2503.13439
- SAM3D: https://arxiv.org/abs/2511.16624
- Flow Matching: https://arxiv.org/abs/2210.02747
- DINOv2: https://arxiv.org/abs/2304.07193
- Training-Free Guided Flow Matching with Optimal Control: https://arxiv.org/abs/2410.18070
- YCB Benchmark: https://ieeexplore.ieee.org/document/7250872
- Google Scanned Objects: https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-objects/
- D(R,O)-Grasp: https://arxiv.org/abs/2410.01702
- FoundationPose: https://arxiv.org/abs/2312.08344
- TouchSDF: https://ieeexplore.ieee.org/document/10517503
- Snap-It-Tap-It-Splat-It: https://arxiv.org/abs/2411.17671
- DSO (physical soundness): https://arxiv.org/abs/2410.06124
- XCube (sparse voxel hierarchies): https://arxiv.org/abs/2310.16757
- Classifier Guidance: https://arxiv.org/abs/2105.05233
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598
- Dual Octree Graph: https://arxiv.org/abs/2110.00189
