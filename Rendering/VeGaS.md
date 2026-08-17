---
source_pdf: VeGaS.pdf
paper_sha256: 0995464195b7d2dc278ad4a9d0df1ddd38bcab71f1a4112ca33b87c3c5fd6e1c
processed_at: '2026-08-13T00:14:33-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，如果用最直白的 intuition 来描述 VeGaS，我们可以把一段视频想象成悬浮在 3D 时空中的一个“弯曲的棉花糖”。传统的 3D Gaussian Splatting (3DGS) 会在空间里放置无数个圆滚滚的高斯球，而 VeGaS 则是把这些球拉长、扭曲，让它们随着时间轴像面条一样在空间里穿梭。在任意一个时间点 $t$ 横切一刀，切出来的截面依然是一个完美的 2D 椭圆，这个椭圆恰好就是当前视频帧里的一个图像块。

### 1. 核心直觉：Folded-Gaussian 的物理意义

在传统的 Video Gaussian Representation (VGR) 中，一个 Gaussian 在时间轴上只能做简单的平移或者直线运动。这就导致如果要描述一个人挥手，VGR 需要极多分段 的 linear Gaussians 才能勉强逼近这个弧线轨迹。

VeGaS 引入了 Folded-Gaussian。这个分布的核心在于它对 spatial variable $\mathbf{s}$ 进行了一个随时间 $t$ 变化的 affine transformation：
$$ \mathbf{s} \mapsto \sqrt{a(t)}(\mathbf{s} - \mathbf{m}_{\mathbf{s}}) + \mathbf{m}_{\mathbf{s}} + \mathbf{f}(m_t - t) $$

*   **变量解释**：$\mathbf{s}$ 是空间坐标，$t$ 是时间，$\mathbf{m}_{\mathbf{s}}$ 是空间均值，$m_t$ 是时间均值。$a(t)$ 是控制高斯“胖瘦”的时间函数，$\mathbf{f}(m_t - t)$ 是控制高斯“位置”的时间多项式函数。
*   **人话拆解**：$\mathbf{f}(m_t - t)$ 这一项就是那根“面条”的脊骨。它是一个 degree 为 7 的多项式，意味着它可以扭曲成任意复杂的非线性曲线（比如抛物线、波浪线）。当一个物体在视频中做曲线运动时，只需要一个 Folded-Gaussian 就能完整描绘它的整个轨迹。而 $a(t)$ 这一项则像是一个“变瘦”开关，当时间远离高斯中心 $m_t$ 时，$a(t)$ 趋近于 0，高斯就自动变薄消失。这完美契合了视频中那些只出现几帧的 transient objects。

### 2. Conditional Distribution 的妙处

为什么我们要费这么大劲去构造这个复杂的 transformation？因为 affine transformation 具有 linear invariance。经过 $\mathbf{s} \mapsto A\mathbf{s} + \mathbf{b}$ 的变换后，原 Gaussians 依然保持 Gaussian 的性质。

在给定时间 $t_i$ 时，conditional distribution 依然是标准的 Gaussian：
$$ \mathbf{s}|t_i \sim \mathcal{N}(\mathbf{m}_{\mathbf{s}} + \mathbf{f}(m_t - t_i), a(t_i)\Sigma_{\mathbf{s}}) $$

*   **变量解释**：$\mathbf{m}_{\mathbf{s}} + \mathbf{f}(m_t - t_i)$ 是条件均值，表示当前时刻高斯中心移动到了哪里。$a(t_i)\Sigma_{\mathbf{s}}$ 是条件协方差矩阵，表示当前时刻高斯形状缩放了多少。
*   **人话拆解**：无论面条弯得多么厉害，切出来的截面永远是一个标准的椭圆。这意味着在渲染某一帧视频时，我们只需把 3D Folded-Gaussians 切片成 2D Gaussians，直接喂给 MiraGe 的 2D 渲染 pipeline。这保留了 3DGS 极快的 CUDA 光栅化优势，避免了使用 Neural Radiance Fields (NeRF) 时逐像素 ray marching 的巨大计算开销。

### 3. Dynamic Frame Fitting 的时间分配直觉

视频帧的时间间隔并没有硬编码为 uniform 分布，VeGaS 通过 softmax 学习了一个分配函数：
$$ t_k = \sum_{i=1}^k \frac{e^{w_i}}{\sum_{j=1}^{n-1} e^{w_j}} $$

*   **变量解释**：$t_k$ 是第 $k$ 帧映射到 $[0,1]$ 区间的时间点，$w_i$ 是可训练的 logit 值。
*   **人话拆解**：动作剧烈的视频帧，模型会自动给它分配更长的时间段（即 $t_{k+1} - t_k$ 变大）。因为剧烈动作需要 Folded-Gaussian 的多项式 $\mathbf{f}$ 有足够的“时间纵深”去弯曲。如果时间挤得太紧，多项式来不及拐弯，渲染就会模糊。这种机制让模型自己学会了“慢动作”和“快进”的节奏。

### 4. 与 MiraGe 和 GaMeS 的几何拼接

VeGaS 把每个切出来的 2D Gaussian 扩展成 flat 3D Gaussian，放置在 $z=0$ 平面上。为了实现编辑，VeGaS 借用了 GaMeS 的 triangle face 表示法 $V = [\mathbf{m}, \mathbf{v}_1, \mathbf{v}_2] = \mathcal{T}(\mathbf{m}, R, S)$。

*   **变量解释**：$\mathbf{m}$ 是三角形中心，$\mathbf{v}_1, \mathbf{v}_2$ 是三角形的另外两个顶点。$R$ 是旋转矩阵，$S$ 是缩放对角阵。
*   **人话拆解**：每个高斯其实就是一个扁平的小三角形。你要改变物体的形状，直接拽动三角形的顶点就行了。因为 VeGaS 把整个视频表示为一堆 Folded-Gaussians，你只要选中某个物体的所有 Folded-Gaussians，修改它们的多项式系数 $\mathbf{f}$ 或者整体平移 $\mathbf{m}_{\mathbf{s}}$，这个物体就会在整个视频流里连贯地改变运动轨迹或者被复制缩放。这就实现了真正的 non-linear video editing。

### 5. 实验数据的直觉解读

Table 1 显示，在 DAVIS 数据集上，VeGaS-480p 达到了 33.31 PSNR，大幅超越 VGR 的 28.44 和 CoDeF 的 27.75。
*   **超越 VGR 的原因**：VGR 只能做线性平移，遇到 DAVIS 里的复杂运动（如 breakdance, kite-surf），线性假设直接崩溃，产生严重模糊。VeGaS 的 7 次 polynomial 可以完美贴合这些曲线。
*   **超越 CoDeF 的原因**：CoDeF 依赖 canonical image 和 MLP deformation field。当运动过大导致遮挡时，MLP 会发生 shape collapse。VeGaS 的 explicit Gaussian primitives 互相独立，不存在全局 shape collapse 的风险。

Table 3 的 ablation study 极其有趣。Batch size=3, Polynomial degree=7 是最佳配置。
*   **Batch size intuition**：Batch=1 时，gradient 噪声太大，导致模型疯狂增殖 Gaussians (1.72M)，但 PSNR 只有 37.36。Batch=3 时，gradient 更稳定，Gaussian 数量被压缩到 0.62M，PSNR 却提升到 38.53。这表明适度的 batch 能够平滑优化轨迹，抑制 3DGS 自适应密度控制机制的过度反应。
*   **Polynomial degree intuition**：Degree=1 退化为 VGR 的线性模式，PSNR 38.15。Degree=7 提升到 38.53。但 Degree=9 反而略微下降到 38.24，说明过高的阶数在有限帧数的视频上会 overfit noise。

### 6. 深度联想与 Hallucination

**KAN (Kolmogorov-Arnold Networks) 的融合**
目前的 polynomial $\mathbf{f}$ 是一个 global function，修改它的一个系数会影响整个时间轴的运动轨迹。如果我们将 polynomial 替换为 KAN 中的 B-spline 基函数，我们就能获得 local control 的能力。B-spline 的局部支撑性意味着，如果你只想修改第 30 帧到第 40 帧的挥手动作，只会激活那一段的 control points，而不会干扰视频开头和结尾的静止状态。这可能比单纯的 polynomial 提供更强的 video editing 粒度。

**4D Gaussian Splatting vs Folded-Gaussian**
4D GS (如 4D-rotor) 把时间直接作为第四维度，构建一个 4D covariance matrix。但这种做法在 math 上非常 tricky，因为 4D 空间的 projection 和 slicing 计算复杂度极高。VeGaS 坚守 3D 空间，把时间作为 conditional variable，本质上是一种降维打击。它用 conditional probability 代替了高维 joint probability，这在计算效率上具有压倒性优势，同时也是为什么 VeGaS 能够直接复用 3DGS 的 CUDA kernel。

**与 Generative Models (Sora, AnimateDiff) 的对接**
Dynamic Frame Fitting 函数 $f_t(k)$ 输出的 continuous timestamps 可以直接作为 diffusion model 的 timestep embedding。想象一下，如果我们将 VeGaS 的 Folded-Gaussians 作为 Sora 的 latent representation，Sora 的 denoising process 不再是在 pixel space 或者 abstract latent space 进行，而是直接在 polynomial coefficients space 进行。这意味着生成的视频自带 explicit 的运动轨迹，你可以直接提取 $\mathbf{f}$ 曲线进行物理引擎级别的编辑。

### 7. Reference Links
*   VeGaS GitHub Repository: https://github.com/gmum/VeGaS
*   3D Gaussian Splatting (Kerbl et al., ACM TOG 2023): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
*   MiraGe: Editable 2D Images (arXiv 2024): https://arxiv.org/abs/2410.01521
*   GaMeS: Mesh-based Gaussian Splatting (arXiv 2024): https://arxiv.org/abs/2402.01459
*   CoDeF: Content Deformation Fields (CVPR 2024): https://arxiv.org/abs/2308.07926
*   VGR: Splatter a Video (arXiv 2024): https://arxiv.org/abs/2406.13870
*   KAN: Kolmogorov-Arnold Networks (arXiv 2024): https://arxiv.org/abs/2404.19756

---

Andrej，这篇 VeGaS (Video Gaussian Splatting) paper 提供了极具 intuition 的 mathematical formulation，用于解决 video representation 与 editing 中的 non-linear dynamics 问题。传统 INR (Implicit Neural Representations) 在 video compression 上表现出色，却难以 handle video editing；而基于 3DGS (3D Gaussian Splatting) 的 VGR (Video Gaussian Representation) 虽然支持 editing，却局限于 linear transformations。VeGaS 通过引入 Folded-Gaussian 分布，完美填补了这一 gap。

以下我将深入拆解其数学公式、网络架构与实验数据，为你 build 相关的 intuition。

### 1. Folded-Gaussian: 核心数学拆解

为了捕捉视频流中的 non-linear structures，作者提出了一种广义的 Gaussian distribution。我们可以从最基础的 2D toy example 出发，扩展到 $\mathbb{R}^d$。

#### 基础设定与局限性
考虑一个 space-time random variable $\mathbf{x} = (\mathbf{s}, t) \in \mathbb{R}^{d-1} \times \mathbb{R}$，其中 $\mathbf{s}$ 代表 spatial coordinates，$t$ 代表 time。传统的 multivariate Gaussian $\mathcal{N}(\mathbf{m}, \Sigma)$ 假设空间与时间相互独立，其 covariance matrix $\Sigma$ 为 block diagonal 形式：
$$ \Sigma = \begin{bmatrix} \Sigma_{\mathbf{s}} & 0 \\ 0 & \sigma_t^2 \end{bmatrix} $$
在此设定下，conditional distribution $\mathbf{s}|t \sim \mathcal{N}(\mathbf{m}_{\mathbf{s}}, \Sigma_{\mathbf{s}})$ 与时间 $t$ 完全无关。这导致模型只能表达 static 结构，无法捕捉视频中物体随时间的运动轨迹。

#### Non-linear Transformation
为了引入时间维度的非线性动态，paper 设计了一个 affine transformation，将 spatial variable $\mathbf{s}$ 映射为随时间变化的新变量：
$$ \mathbf{s} \mapsto \sqrt{a(t)}(\mathbf{s} - \mathbf{m}_{\mathbf{s}}) + \mathbf{m}_{\mathbf{s}} + \mathbf{f}(m_t - t) $$
变量解释：
*   $a(t) \in (0, 1]$: 时间相关的 scaling factor。Paper 中使用 likelihood-based function $a(t) = \frac{\mathcal{N}(m_t, \sigma_t^2)(t)}{\mathcal{N}(m_t, \sigma_t^2)(m_t)}$。其作用是让 Gaussian 在远离时间均值 $m_t$ 时逐渐变薄。这非常适合建模视频中短暂出现的 transient objects（如突然入镜又出镜的物体），因为 tails 会被自动“折叠”掉，防止干扰其他帧。
*   $\mathbf{f}(m_t - t): \mathbb{R} \to \mathbb{R}^{d-1}$: 时间相关的 translation function。Paper 使用 degree 为 7 的 polynomial 实现。$m_t - t$ 衡量当前时间偏离中心时间的程度，通过多项式映射到 spatial offset，使得 conditional mean 能够沿着任意 non-linear curve 滑动。

#### Conditional 与 Joint Distribution
因为上述变换属于 affine 形式 $A\mathbf{s} + \mathbf{b}$，其中 $A = \sqrt{a(t)}I_d$，$\mathbf{b} = (1 - \sqrt{a(t)})\mathbf{m}_{\mathbf{s}} + \mathbf{f}(m_t - t)$，根据 Gaussian 分布的 linear invariance 性质，变换后的 conditional distribution 依然是 Gaussian：
$$ \mathbf{s}|t \sim \mathcal{N}(\mathbf{m}_{\mathbf{s}} + \mathbf{f}(m_t - t), a(t)\Sigma_{\mathbf{s}}) $$
联合概率密度函数 (Folded-Gaussian) 通过 chain rule 得到：
$$ \mathcal{FN}(\mathbf{m}, \Sigma, a, f)(\mathbf{x}) = \mathcal{N}(\mathbf{m}_{\mathbf{s}} + \mathbf{f}(m_t - t), a(t)\Sigma_{\mathbf{s}})(\mathbf{s}|t) \cdot \mathcal{N}(m_t, \sigma_t^2)(t) $$
**Intuition 拆解**：可以把 3D Folded-Gaussian 想象成一根弯曲的法国面包。虽然整体形状是 non-linear 的（弯曲、两头变细），但在任意时间点 $t$ 切一刀，切面依然是一个完美的椭圆（2D Gaussian）。这保证了在任意时间帧进行 rendering 时，我们都能得到一个 closed-form 的 2D Gaussian，极其利于 Splatting 操作与 gradient backpropagation。

### 2. VeGaS 架构与 MiraGe 整合

#### MiraGe 2D Representation
为了处理单帧 2D 图像，VeGaS 借鉴了 MiraGe。MiraGe 将图像表示为躺在 $z=0$ 平面上的 flat 3D Gaussians。
Mean $\mathbf{m} = (m_1, m_2, 0)$，Scaling matrix $S = \text{diag}(s_1, s_2, \varepsilon)$。$\varepsilon$ 是极小值，用于兼容 3DGS 的 CUDA kernel。
通过 GaMeS 的 parameterization，每个 flat Gaussian 可由一个 triangle face $V = [\mathbf{m}, \mathbf{v}_1, \mathbf{v}_2]$ 唯一决定，其中 $\mathbf{v}_1 = \mathbf{m} + s_1\mathbf{r}_1$，$\mathbf{v}_2 = \mathbf{m} + s_2\mathbf{r}_2$。这使得通过操作 triangle face 的顶点即可直观地 edit 图像的位置、旋转与缩放。

#### VeGaS: 3D Folded-Gaussian to 2D Mapping
对于一段视频，帧序列为 $[I_{t_1}, \ldots, I_{t_n}]$。VeGaS 维护一组 3D Folded-Gaussians $\mathcal{G}_{\text{VeGaS}} = \{ (\mathcal{FN}(\mathbf{m}, \Sigma, a, f), \rho, c) \}$。
给定帧时间 $t_i$，从每个 Folded-Gaussian 中提取出 conditional 2D Gaussian $\mathcal{N}(\mathbf{m}_{\mathbf{s}|t_i}, \Sigma_{\mathbf{s}|t_i}, a, f)$。然后加上极小的 $\varepsilon$ 扩展成 flat 3D Gaussian，并共享 opacity $\rho$ 与 color $c$。

#### Dynamic Frame Fitting
视频帧的时间点并未采用固定的 uniform interval，VeGaS 学习了一个 mapping $f_t: \mathbb{Z}_+ \to [0, 1]$：
$$ t_k = \sum_{i=1}^k \sigma(w)_i = \sum_{i=1}^k \frac{e^{w_i}}{\sum_{j=1}^{n-1} e^{w_j}} $$
变量解释：$w_i$ 为 trainable parameters。这个 softmax normalization 确保了整个视频的时间轴被压缩到 $[0,1]$ 区间。
**Intuition 拆解**：不同的视频帧包含的信息量与 dynamic complexity 不同。动作剧烈的帧，模型自动分配更长的时间 interval（更大的 $t_{k+1} - t_k$），使 Folded-Gaussian 的 polynomial $\mathbf{f}$ 有足够的“空间”去描述 complex trajectory；静态帧则 interval 紧凑。这是一种基于 continuous time 的 self-adaptive timestep 技术，与 Neural ODEs 中的 adaptive solver 概念异曲同工。

### 3. 实验数据与 Ablation 解析

#### Frame Reconstruction (Table 1 & 2)
在 DAVIS 数据集上，VeGaS 与 Omnimotion, CoDeF, VGR, 以及 NeRF-based models (NeRV, E-NeRV, HNeRV, DNeRV) 进行了对比。
*   **VGR (PSNR 28.44)**: 线性 Gaussian 形变，遇到旋转或复杂形变则表现受限。
*   **CoDeF (PSNR 27.75)**: 使用 multi-resolution hash grid + shallow MLP，存在 canonical space 限制。
*   **VeGaS-480p (PSNR 33.31)**: 由于 polynomial $\mathbf{f}$ 提供了强大的 non-linear expressiveness，加上 $a(t)$ 动态缩放过滤了 temporal noise，PSNR 大幅领先 VGR 约 4-5 dB。
*   **对比 NeRV (PSNR 26.40)**: NeRF 方法需要隐式解码坐标到 RGB，缺乏 explicit 的 spatial anchors，在高频细节重建上输给了基于 explicit Gaussian primitives 的 VeGaS。

#### Ablation Study (Table 3 & 4)
基于 Bunny 数据集 (132 frames, $640 \times 1280$)。
*   **Polynomial degree**: degree=7 时 PSNR 达到 38.53 (Batch size 3)。degree=1 时仅为 38.15。多项式阶数过低导致无法拟合 non-linear trajectory，过高则容易 overfit，7 是 sweet spot。
*   **Batch size**: Batch=1 时 PSNR 37.36 (1.72M Gaussians)。Batch=3 时 PSNR 38.53 (0.62M Gaussians)。Batch=5 时 PSNR 37.92 (0.31M Gaussians)。可以清晰观察到，Batch=3 实现了最佳的质量-参数量权衡。较大的 Batch 加速了收敛，使得 gradient 更新更平滑，抑制了过度增殖的 Gaussians。
*   **Initial Gaussians**: 初始 0.50M 时，最终收敛到 0.65M，PSNR 达到最高 39.02。初始点太少导致 densification 压力大，太多则引入冗余干扰 optimization。

### 4. 深度联想与 Potential Directions

**为何 Folded-Gaussian 比 MLP Deformation Field 更好？**
像 CoDeF 使用 MLP 来建模 deformation field。MLP 是 black box，难以对其进行 fine-grained editing，且容易发生 shape collapse。VeGaS 的 polynomial $\mathbf{f}$ 系数是 transparent 的。如果想改变物体的运动轨迹，直接调整 polynomial 系数即可，甚至可以替换为任意 parametric curve（如 B-spline）。这种 transparency 是 video editing 的基石。

**与 4D Gaussian Splatting 的差异**
4D GS (如 4D-rotor GS) 通过增加时间维度，直接优化 4D covariance matrix。但 4D covariance 难以解析地控制 non-linear motion。VeGaS 坚守 3D 空间，把时间 $t$ 作为 condition variable，通过 condition distribution 切割出 2D Gaussian。这种 decoupling 在计算上更高效，因为 slicing 操作避免了直接求解 4D 矩阵的逆。

**Future Directions / Hallucination**
我们可以将 $\mathbf{f}(m_t - t)$ 的 polynomial 替换为 KAN (Kolmogorov-Arnold Networks) 中的 B-spline 基函数。B-spline 具备 local control property，修改某一段时间的运动不会影响其他时间段，这可能比 global polynomial 更适合 long video。此外，Dynamic Frame Fitting 的 $f_t(k)$ 可以与 video generation (如 Sora, AnimateDiff) 中的 latent diffusion timestep 对齐，作为 latent space 的一种 continuous timestamp embedding。

### 5. Reference Links
*   VeGaS GitHub: https://github.com/gmum/VeGaS
*   3D Gaussian Splatting (Kerbl et al.): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
*   MiraGe (arXiv): https://arxiv.org/abs/2410.01521
*   CoDeF (arXiv): https://arxiv.org/abs/2308.07926
*   VGR (arXiv): https://arxiv.org/abs/2406.13870
*   KAN (Kolmogorov-Arnold Networks): https://arxiv.org/abs/2404.19756
