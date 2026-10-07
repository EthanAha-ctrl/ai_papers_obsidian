---
source_pdf: Choreographing a World of Dynamic Objects.pdf
paper_sha256: 00df959280afc61d79f879aa67732ae671e3d15dabc8b84fa4fd53f05ce8f1c6
processed_at: '2026-08-03T15:36:59-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，没问题，我把上面那些复杂的公式和推导扔掉，咱们用纯直觉和人话把这篇 paper 的精髓再过一遍。

说白了，这篇 paper 核心就干一件事：**给一张静态的 3D 场景图（几个 mesh 拼起来的），加上一句 text prompt（比如“一个人把盘子放进微波炉并关门”），让它自动生成一段多物体互动的 4D 动画。**

这事儿为什么难？我给你梳理一下他们的思考链路：

### 1. 为什么不直接训个 end-to-end 模型？
因为你没数据。现在的 4D 数据集（比如 Objaverse）几乎都是单个物体的形变（比如人动一动、狗跑一跑），**根本没有“A 和 B 交互”的 4D 数据**。你想让模型学“人开微波炉”，数据集里连微波炉和手的 4D 互动样本都没几个。所以 end-to-end 训练（像 AnimateAnyMesh 那样）在多物体交互场景下直接歇菜。

### 2. 曲线救国：让 2D 视频模型当“动作指导”
既然没 4D 数据，那有什么？我们有海量 2D 视频数据，以及基于这些数据训练出来的强大的 2D 视频生成模型（比如 Wan 2.2）。这些模型脑子里装满了“世界是怎么运转的”的先验知识。

CHORD 的核心思路就是 **Score Distillation Sampling (SDS)** 的视频版：我渲染一段当前 3D 场景的 2D 视频，加点噪声，喂给视频模型，然后问它：“哥们，根据这句 text，你觉得这视频下一步该怎么变？”视频模型吐给我一个梯度信号，我拿着这个信号去微调我 3D 场景的参数。循环往复，直到 3D 动画在视频模型眼里看起来是“合理”的。

### 3. 坑 1：现代视频模型的“底层架构”变了
DreamFusion 时代的 SDS 是针对 DDPM 架构推导的，但现在先进的视频模型都换成了 **Rectified Flow (RF)** 架构。这两种模型加噪的方式和预测的目标都不一样，你直接套老公式，梯度算出来是错的。

所以作者重新推导了一套针对 RF 的 SDS 公式。更关键的是，他们发现了一个经验现象：**物体的大幅度运动，只有在高噪声水平下才能被模型“逼”出来**。如果你按常规做法随机采样噪声，大部分算出来的梯度都是没啥用的微调，物体根本动不起来。

**填坑方法**：他们搞了个 importance sampling，专门挑那些权重大的高噪声点来采样，然后把它做成一个 annealing schedule（退火策略）。训练初期用高噪声，逼着物体先学会大幅度的“粗动作”；训练后期降噪声，再去抠细节。

### 4. 坑 2：4D 参数太自由，模型一学就崩
就算梯度对了，4D deformation梯度对了，4D deformation

---

# CHORD: Choreographing a World of Dynamic Objects 深度解读

Andrej, 这篇 paper 我读得非常仔细，下面我把整个 pipeline、关键数学推导、架构设计 motivation、实验数据，以及我自己的 intuition 都尽量详细地展开。paper 主页在 https://yanzhelyu.github.io/chord 。

---

## 1. Big Picture: 这篇 paper 真正在解决什么问题

这个工作核心要解决的问题可以一句话概括：**给定一个静态 3D scene（多 object mesh 拼起来的 snapshot）+ 一句 text prompt（比如 "a man lowers the head of the lamp with his hand"），自动生成 scene-level 的 4D motion，并且 motion 要 consistent across multiple interacting objects**。

为什么这件事难？可以从三个 dimension 看：

1. **Data scarcity**：现有的 4D dataset（比如 Objaverse、Objaverse-XL、AnimMate [14]）几乎都是 single-object deformation，scene-level multi-object interaction 数据几乎没有。end-to-end 学习 [73] 因此被限制在 humanoid/单 object 范畴。

2. **Representation instability**：4D deformation field 是一个 spatially high-dimensional 且 temporally ill-regularized 的优化目标，特别是在只用 noisy SDS gradient 当 supervision 的情况下，很容易 collapse 到零运动或者 artifact。

3. **Distillation 与 modern video model 的 architecture gap**：原始 SDS [53] 是为 DDPM-style diffusion model 推导的，而 modern video generative model（Wan 2.2 [65]、SD3 [16]、Sora 等）基本都换成了 **Rectified Flow (RF)** [44]。RF 的 forward process、velocity target 都和 DDPM 不同，直接套用 SDS 公式会出错。

CHORD 的 contribution 就是同时 attack 这三件事：
- 用 video generative model 当 "choreographer"，绕过 4D data scarcity；
- 设计 hierarchical 4D representation（spatial coarse-to-fine control points + temporal Fenwick tree）稳定优化；
- 重新推导 RF-SDS，提出 W-RFSDS + annealing noise schedule。

下游还 demo 了 robot manipulation（dense object flow 作为 manipulation policy guidance）。

---

## 2. Background: SDS 的本质

为了 build intuition，先把 DreamFusion [53] 的 SDS 讲清楚，因为这是 CHORD 的起点。

SDS 的本质想法：你有一个 3D representation 参数 θ，渲染出 image z = g(θ)，然后把这个 image 加噪喂给一个 pretrained 2D diffusion model，让 diffusion model "告诉你" 这个 image 应该往哪个方向改才更符合 text y。

原始 SDS 公式 (Eq. 1)：

$$
\nabla_{\theta} \mathcal{L}_{\mathrm{SDS}}(\theta; \mathbf{z}, \mathbf{y}) = \mathbb{E}_{\tau, \epsilon}\left[ w(\tau)\left( \hat{\epsilon}(\mathbf{z}_\tau; \tau, \mathbf{y}) - \epsilon \right) \frac{\partial \mathbf{z}}{\partial \theta} \right]
$$

变量说明：
- θ：3D asset 参数（这里就是 4D representation 参数）
- z：从当前 3D asset 渲染出来的 image (or video)
- τ：noise level，uniformly sampled from (0,1)
- ε ~ N(0, I)：随机加的 Gaussian noise
- z_τ = √(1-τ)·z + √τ·ε (DDPM convention)：noisy image
- ε̂(z_τ; τ, y)：diffusion model 预测的 noise（given text y）
- w(τ)：weighting function，控制不同 noise level 的贡献
- ∂z/∂θ：从 image space 到 parameter space 的 Jacobian（通过 differentiable rendering）

intuition：ε̂ - ε 是 "model 认为的 noise" 和 "实际加的 noise" 的差，如果 image 已经 perfect，这两个应该相等，gradient ≈ 0。如果不相等，说明 image 不符合 model 的 prior，gradient 推 θ 去减小这个差。

为什么忽略 ∂ε̂/∂z？因为 backprop through 整个 diffusion U-Net 太贵，SDS 用一个 "stop gradient" 的近似，只保留通过 ∂z/∂θ 的 chain rule。这正是 SDS 被 Vincent et al. 批评为 "not a true gradient" 的地方，但实践上 work。

4D-SDS 就是把这个 idea 从 image 扩到 video：渲染一个 video，加 noise，喂给 video diffusion model，回传 gradient 给 4D representation。

---

## 3. RF-SDS 推导：从 DDPM 到 Rectified Flow

这是这篇 paper 的第一个关键 technical contribution。我详细推导一下。

### 3.1 Rectified Flow 的 training loss

RF 模型 [44, 16] 用 linear interpolation 而不是 DDPM 的 √(1-τ) 形式：

$$
\mathbf{z}_\tau = (1-\tau)\mathbf{z} + \tau \epsilon
$$

变量：
- z：clean image/video latent
- τ ∈ [0,1]：noise level，注意这里 τ=0 是 clean，τ=1 是 pure noise
- ε ~ N(0, I)
- z_τ：linearly interpolated noisy latent

velocity target 是 (ε - z)（从 z 指向 ε 的方向）。

RF training loss (Eq. 14)：

$$
\mathcal{L}_{\mathrm{RF}}(\theta; \mathbf{z}, \mathbf{y}) = \mathbb{E}_{\tau \sim \mathcal{U}(0,1), \epsilon}\left[ w(\tau) \left\| \hat{v}(\mathbf{z}_\tau; \tau, \mathbf{y}) - (\epsilon - \mathbf{z}) \right\|^2 \right]
$$

变量：
- ŵ(z_τ; τ, y)：RF model 预测的 velocity
- (ε - z)：true velocity（朝向 noise 的方向）
- w(τ)：training schedule 的 weight

### 3.2 对 z 求梯度 (Eq. 15)

对 z 求导（注意 z 同时出现在 z_τ 和 (ε-z) 中）：

$$
\nabla_{\mathbf{z}} \mathcal{L}_{\mathrm{RF}} = \mathbb{E}_{\tau, \epsilon}\left[ w(\tau) \left( \hat{v}(\mathbf{z}_\tau; \tau, \mathbf{y}) - (\epsilon - \mathbf{z}) \right) \left( \frac{\partial \hat{v}(\mathbf{z}_\tau; \tau, \mathbf{y})}{\partial \mathbf{z}} + I \right) \right]
$$

第二项里的 ∂v̂/∂z 是通过 RF model 的 backprop，第一项的 +I 来自 d/dz[ε - z] = -I，乘以外面的负号得到 +I。

### 3.3 SDS 风格近似 (Eq. 16 / Eq. 2)

按 SDS 的做法，丢掉 ∂v̂/∂z 那一项（太贵），用 chain rule 链回 θ：

$$
\nabla_{\theta} \mathcal{L}_{\mathrm{RFSDS}}(\theta; \mathbf{z}, \mathbf{y}) = \mathbb{E}_{\tau, \epsilon}\left[ w(\tau) \left( \hat{v}(\mathbf{z}_\tau; \tau, \mathbf{y}) - \epsilon + \mathbf{z} \right) \frac{\partial \mathbf{z}}{\partial \theta} \right]
$$

注意这里 paper 写作 "ε - z" 改成了 "-ε + z"，跟原始 (Eq. 2) 一致。变量含义：
- v̂(z_τ; τ, y)：RF model 预测的 velocity
- (ε - z)：target velocity
- (v̂ - ε + z)：prediction minus target，类似 ε̂ - ε 的角色
- ∂z/∂θ：differentiable rendering 的 Jacobian

### 3.4 W-RFSDS：核心创新 (Eq. 3)

paper 的关键 empirical observation：**deformations 主要在 high noise level τ 时才生成**——只有当 substantial noise added 时，RF model 才会输出能驱动 substantial motion 的 velocity prediction。低 τ 时 gradient 太小、太局部。

如果 uniform 采样 τ，w(τ) 加权之后 effective gradient 被低 τ 的"无效" samples 主导。paper 的 trick：把 w(τ) "absorbed" 进采样分布本身。

定义：

$$
\hat{w}(\tau) = \frac{w(\tau)}{\int_{-\infty}^{\infty} w(\tau) d\tau}
$$

这是 w(τ) 的 normalized 形式，可以当成 probability density function。然后把 τ 的采样从 uniform 改成 ŵ(τ)。

新的 W-RFSDS gradient (Eq. 3)：

$$
\nabla_{\theta} \mathcal{L}_{\mathrm{W\text{-}RFSDS}}(\theta; \mathbf{z}, \mathbf{y}) = \mathbb{E}_{\tau \sim \hat{w}(\tau), \epsilon}\left[ \left( \hat{v}(\mathbf{z}_\tau; \tau, \mathbf{y}) - \epsilon + \mathbf{z} \right) \frac{\partial \mathbf{z}}{\partial \theta} \right]
$$

为什么 w(τ) 消失了？这用的是 importance sampling 的 identity：

$$
\mathbb{E}_{\tau \sim \mathcal{U}}\left[ w(\tau) f(\tau) \right] = \mathbb{E}_{\tau \sim \hat{w}(\tau)}\left[ f(\tau) \right]
$$

因为 ŵ 是 w 的归一化。所以 W-RFSDS 跟 RFSDS 在 expectation 上等价，但 variance 更低，因为高 w(τ) 的 τ 被更频繁采样。

paper 说 "ensures invariance of the expectation of gradients"——意思是这个变换不改变 expected gradient 的方向，只改变 variance。

### 3.5 Annealing Schedule (Eq. 4)

光有 importance sampling 还不够，paper 还要 deterministic 的 annealing。定义 CDF：

$$
h(\tau) = \int_{-\infty}^{\tau} \hat{w}(t) dt
$$

在第 i 步（总 I 步）选 τ_i：

$$
h(\tau_i) = 1 - \frac{i}{I+1}
$$

变量：
- i：当前 iteration index (0 to I)
- I：total iterations（这里 2000）
- τ_i：第 i 步的 noise level

直觉：i=0 时 h(τ_0)=1，对应 τ_0 趋近 1（高 noise）；i=I 时 h(τ_I)=1/(I+1) → 0，对应 τ_I 趋近 0（低 noise）。τ 单调下降，实现 coarse-to-fine 优化 schedule。

这个 schedule 配合 spatial coarse-to-fine control points（后面讲）形成完整 curriculum：高 τ 学 coarse motion，低 τ 学 fine detail。Ablation (Figure 8) 显示去掉这个 schedule 后 laptop 会 "float" 在桌子上，因为低 τ 时 model 缺乏足够 gradient 把 laptop 真正"压下去"。

---

## 4. Hierarchical 4D Representation：paper 的第二个核心

W-RFSDS gradient 信号其实挺 noisy 的，直接优化高维 4D deformation field 会爆炸。CHORD 的解决方案是把 representation 的 dimension 降下来，并且 hierarchical 化——spatial 和 temporal 两个方向都做。

### 4.1 Canonical Geometry: 3D-GS

输入是 N 个 mesh，先转成 3D Gaussian Splatting (3D-GS) [31]：S = {G_i}_{i=1}^N。每个 G_i 是一组 Gaussians，每个 Gaussian 参数化为 (μ, q, S, C, o)：

- μ：mean (位置)
- q：quaternion (旋转)
- S：scaling matrix
- C：color (spherical harmonics)
- o：opacity

为什么用 3D-GS 而不是直接 mesh？因为 3D-GS 的 differentiable rendering 是 dense、smooth gradient，而 mesh rendering 涉及 rasterization 的 discontinuity，gradient 难算。3D-GS 这一层是 gradient "桥梁"。

训练完后，deformation 可以直接 transfer 回 mesh：把 mesh vertex 位置当 μ 喂进 deformation 公式就行。

3D-GS 实现 paper 用的是 gsplat [80] (https://github.com/rerun-io/gsplat 之类)。

### 4.2 Spatial Hierarchy: Bi-Level Control Points

每个 object i 在 time t 的 deformation field T_i^t，用一组 control points 表示。每个 control point k：

- p_k ∈ R³：mean position（固定）
- Σ_k ∈ R^{3x3}：covariance，决定 influence radius
- (R_k^t, T_k^t) ∈ SE(3)：time-varying rigid transformation
- r_k^t ∈ R⁴：R_k^t 的 quaternion 表示

对一个 Gaussian (μ, q, S, C, o)，找它 K 个最近 control points N，用 linear blend skinning (LBS) 计算 deformation。

#### 4.2.1 Gaussian mean 的 deformation (Eq. 5)

$$
\mu^t = \sum_{k \in \mathcal{N}} \beta_k \left( R_k^t (\mu - \mathbf{p}_k) + \mathbf{p}_k + T_k^t \right)
$$

变量解析：
- μ：Gaussian 原始 mean
- p_k：control point k 的 position
- R_k^t：control point k 在 time t 的 rotation matrix
- T_k^t：control point k 在 time t 的 translation
- β_k：blending weight (Eq. 7)
- (μ - p_k)：把 Gaussian 平移到 control point 局部坐标系
- R_k^t(...)：在 control point 局部旋转
- + p_k + T_k^t：先回到世界坐标原点，再加 control point 的 translation
- Σ_k β_k(...)：LBS 加权混合

intuition：这就是 standard LBS，每个 control point 想把 Gaussian "拽" 到一个 transformed 位置，最后加权平均。权重 β_k 决定哪个 control point 说了算。

#### 4.2.2 Gaussian rotation 的 deformation (Eq. 6)

$$
\mathbf{q}^t = \left( \sum_{k \in \mathcal{N}} \beta_k r_k^t \right) \otimes \mathbf{q}
$$

变量：
- q：Gaussian 原始 rotation (quaternion)
- r_k^t：control point k 在 time t 的 rotation quaternion
- ⊗：quaternion product
- (Σ β_k r_k^t)：weighted average quaternion（注意这不是 normalized 的，要做 norm 之后才能用）
- q^t：deformed quaternion

intuition：quaternion 的加权平均然后用 quaternion product 应用 deformation。这里有个细节：sum 后未必是 unit quaternion，需要 normalize。

#### 4.2.3 Blending weight (Eq. 7)

$$
\beta_k = \frac{\hat{\beta}_k}{\sum_{l \in \mathcal{N}} \hat{\beta}_l}, \quad \hat{\beta}_k = \exp\left( -\frac{1}{2} (\mu - p_k) \Sigma_k^{-1} (\mu - p_k)^T \right)
$$

这就是 multivariate Gaussian kernel：
- (μ - p_k) Σ_k^{-1} (μ - p_k)^T：Mahalanobis distance squared
- exp(-1/2 × ...)：标准 Gaussian PDF kernel
- Σ_k^{-1}：inverse covariance，control point 越远 influence 衰减越快
- 归一化保证 Σ β_k = 1

#### 4.2.4 Bi-Level Coarse-to-Fine (Eq. 8, 9)

coarse control points 先训（在 high τ 时），fine control points 后加（low τ 时）。fine control points 加的是 residual deformation：

$$
\mu_{\mathrm{final}}^t = \Delta \mu^t + \mu^t
$$

$$
\mathbf{q}_{\mathrm{final}}^t = \Delta \mathbf{q}^t \otimes \mathbf{q}^t
$$

变量：
- μ^t, q^t：coarse layer 输出
- Δμ^t, Δq^t：fine layer 输出（用同样公式算，但只在低 τ 时启用）

为什么 bi-level 关键？Ablation (Figure 9) 显示：
- 去掉 fine control points：抓握这种细节 motion 没法表达，比如 "抓耳机" 的精细手指 motion。
- 从一开始就只用 fine control points：高 τ 时的 noisy gradient 会让 fine points 乱跑，造成 distortions。
- 去掉 coarse control points（只用 fine）：distortions 严重，因为 fine points 缺乏全局结构先验。

这个设计是从 SC-GS [27] 借来的，CHORD 把它从 spatial-only 推到 4D 时空。

### 4.3 Temporal Hierarchy: Fenwick Tree

这是 paper 最 elegant 的设计，我重点讲。

#### 4.3.1 Naive approach 的问题

最直接的 temporal parameterization：对每个 control point k，独立地存 (R_k^0, T_k^0), (R_k^1, T_k^1), ..., (R_k^T, T_k^T)，每帧一个 deformation。

问题：所有 deformation 初始化为 0（identity），第 0 帧 frozen。后面 frame 的 deformation 必须从 0 开始学，远离第 0 帧的 frame 越难学，因为 W-RFSDS gradient 在每帧是独立的、noisy 的，没有"积累"。

paper 的观察：temporally 邻近的 frame 应该 share 参数，因为 motion 在时间上是 locally smooth 的。

#### 4.3.2 Fenwick Tree (Binary Indexed Tree)

Fenwick tree [18] (Peter M. Fenwick, 1994, https://en.wikipedia.org/wiki/Fenwick_tree) 是经典 algorithm design 里的 data structure，原本用于 prefix sum query。CHORD 把它拿来存 deformation。

对每个 control point k，存一组 nodes F_k = {(r_k^{[j]}, T_k^{[j]})}_{j=1}^T。每个 node [j] 编码一段 frame range 的 cumulative deformation。

Fenwick tree 的 key property：query frame t 时，会返回一个特定的 node set BIT(t)，这些 node 的 cumulative sum 给出 frame t 的 deformation。邻近 frame 的 BIT(t) 和 BIT(t+1) 共享很多 nodes，自然就 enforces 了 temporal coherence。

具体公式 (Eq. 10, 11)：

$$
T_k^t = \sum_{j \in \mathrm{BIT}(t)} T_k^{[j]}
$$

$$
r_k^t = \mathrm{norm}\left( \sum_{j \in \mathrm{BIT}(t)} r_k^{[j]} \right)
$$

变量：
- T_k^t, r_k^t：control point k 在 frame t 的 translation/rotation
- BIT(t)：Fenwick query 返回的 node set
- T_k^{[j]}, r_k^{[j]}：Fenwick tree 的 node values
- norm(·)：normalize quaternion 到 unit length

intuition (paper Figure 4)：假设有 8 帧。
- node [6] 编码 frame 5-6 的累积 deformation
- query frame 6: BIT(6) = {[6], [4]}（4 编码 1-4，6 编码 5-6）
- query frame 7: BIT(7) = {[7], [4]}（7 编码 frame 7）
- frame 6 和 frame 7 共享 node [4]，所以它们 "share parameters through overlapping intervals"

更具体的 Fenwick indexing：node index j 的"管辖范围"是 (j - lowbit(j), j]，lowbit(j) = j & -j。query 时一直 j -= lowbit(j) 直到 j=0。这是 Fenwick 1994 paper 里的经典定义。

为什么这比 RNN 或 temporal convolution 更适合 SDS 优化？
- RNN/conv 假设有"全局时间模型"，但 SDS gradient 是 per-frame noise，不一定 smooth。
- Fenwick tree 是 explicit 的 parameter sharing，gradient 直接通过 sum 链回每个 node，optimization landscape 更 friendly。
- long-horizon：早期 frame 的 deformation 自然被很多后续 frame 的 query 引用，所以 early frame 会被反复监督，late frame 通过新增 node 学到 residual。

Ablation (Figure 9 top)：去掉 Fenwick tree，late frame 出现 severe artifacts，因为 late frame 的 gradient 没法 backprop 到 early frame 的 "good initialization"。

paper 还有一个细节：在 iteration 100 时，把 frame 30 之后的所有 deformation reinitialize 成 frame 30 的 deformation（"split training schedule"），进一步稳定 long-horizon 学习。

#### 4.3.3 一个 Python-style pseudo 代码

为了 build intuition，我把 Fenwick query 的逻辑写出来：

```python
def fenwick_query(nodes, t):  # t is 1-indexed frame
    result = identity()
    while t > 0:
        result = result.compose(nodes[t])
        t -= t & (-t)  # lowbit strip
    return result

def fenwick_update(nodes, t, delta):  # add delta to nodes affecting frame t
    while t <= T:
        nodes[t] = nodes[t].compose(delta)
        t += t & (-t)
```

在 CHORD 的场景，"update" 是通过 gradient descent 优化 node values，"query" 是 forward pass 计算 frame t 的 deformation。

---

## 5. Regularization (Eq. 12, 13)

光有 hierarchical representation 还不够，paper 加了两个 reg：

### 5.1 Temporal Regularization (Eq. 12)

渲染一个 3D flow map video F：把每个 Gaussian 的 color attribute 换成 μ_i^t - μ_i^{t+1}（连续两帧 mean 的差）。然后 render 出来就是 per-pixel 的 3D flow。

$$
\mathcal{L}_{\mathrm{temp}} = \sum_t \sum_{\mathbf{p}} \| F_{\mathbf{p}}^t \|_2^2
$$

变量：
- F_p^t：pixel p, time t 的 rendered 3D flow
- 内层 sum 遍历所有 pixel

直觉：这个 loss penalize 任何 motion。听起来 weird——我们不是想要 motion 吗？

intuition：这个 reg 不是要 suppress motion，而是要 suppress "flicker"——那些 W-RFSDS gradient 因为 noise 导致的 spurious per-frame motion。理想 motion 应该是 smooth trajectory，但 flicker 会让相邻 frame 的 flow 在 magnitude 上波动很大，被 L2 norm 强烈 penalize。

Ablation (Figure 10)：去掉 temporal reg，cat 的 tail 会突然 "appear"（其实是因为某些 frame 的 Gaussian 跑到视野内，造成 flicker）。

weight 是 decay 的，从 9.6 衰减到 1.6，早期 motion 小 reg 大，晚期 motion 大 reg 小。

### 5.2 Spatial Regularization: ARAP (Eq. 13)

ARAP (As-Rigid-As-Possible, Sorkine & Alexa 2007, https://www.cs.cmu.edu/~kmcrane/Projects/AsRigidAsPossible/ 等) 是经典 geometry processing 方法。CHORD 用它鼓励 deformation 局部 rigid。

构造：
- 对每个 object i，先算 SDF φ_i(x)
- 在 voxel grid V_s 上提取近表面 voxel centers：S_i = {x | |φ_i(x)| ≤ τ, x ∈ V_s}，要求 |S_i| ≈ 7500
- τ 这里是 SDF threshold，跟 noise level τ 是不同含义（paper 复用了符号，slightly confusing）

ARAP loss (Eq. 13)：

$$
\mathcal{L}_{\mathrm{ARAP}} = \sum_{i, t, \mathbf{x} \in \mathcal{S}_i, \mathbf{y} \in \mathcal{N}_\mathbf{x}} \left\| \mathbf{x} - \mathbf{y} - \hat{R}_\mathbf{x}(\mathbf{x}^t - \mathbf{y}^t) \right\|_2^2
$$

变量：
- x, y：S_i 中两个邻近点（y ∈ N_x 是 x 的 10 nearest neighbors）
- x^t, y^t：deformed 位置（用 Eq. 5 算）
- R̂_x：在 x 处估计的 local rotation matrix（通过 SVD of neighbors' displacement）
- (x - y)：原始 relative position
- R̂_x(x^t - y^t)：用 estimated rotation 旋转后的 deformed relative position

直觉：如果 deformation 是 rigid 的，那 R̂_x 应该精确地把 (x - y) 映射到 (x^t - y^t)，loss 为 0。non-rigid deformation 会有残差，被 penalize。

这个 reg 鼓励 deformation 局部 rigid，penalize stretch/skew 这类 non-rigid 变形，但允许大块 rigid transform（因为 R̂ 可以是任意 rotation）。这跟 bi-level control points 的设计 compatible：coarse level 主导 rigid motion，fine level 才允许 non-rigid。

Ablation (Figure 10)：去掉 spatial reg，object 出现 distortions（比如局部拉伸）。

weight decay：3000 → 300，权重很大说明 ARAP loss 的 per-term magnitude 很小（点对距离的平方），需要大 weight。

---

## 6. 完整 Algorithm Flow

我把整个 training loop 写一下：

```
Input: N meshes, text prompt y
Output: 4D deformation parameters

1. Convert meshes to 3D-GS: S = {G_i}_{i=1}^N
2. Initialize control points via FPS + K-means on voxel centers
3. Initialize Fenwick tree nodes for each control point to identity
4. For i = 0 to I=2000:
    a. Compute τ_i via annealing schedule (Eq. 4)
    b. Compute CFG scale, regularization weights (decayed)
    c. Sample camera pose
    d. Forward pass:
       - For t in 0..T-1:
         - For each Gaussian: query Fenwick tree for (R_k^t, T_k^t)
         - Apply LBS (Eq. 5,6) [or +fine (Eq. 8,9) if τ_i small]
         - Render image at frame t
       - Stack frames -> video z
       - Render flow map video F (for temporal reg)
       - Compute ARAP on sampled point cloud (for spatial reg)
    e. Compute W-RFSDS gradient (Eq. 3):
       - ε ~ N(0,I), z_τ = (1-τ_i)z + τ_i ε
       - v̂ = Wan2.2(z_τ; τ_i, y)  (no backprop through model)
       - gradient = (v̂ - ε + z) * ∂z/∂θ
    f. Compute L_temp (Eq. 12), L_ARAP (Eq. 13)
    g. Update θ (deformation params, control point scales) via Adam
    h. At i=100: reinit frame>30 deformation to frame=30 value
5. Transfer Gaussian deformation back to mesh vertices
```

Training detail (from appendix)：
- Wan 2.2 (14B) image-to-video
- Resolution 832×464
- 41 frames deformation sequence
- 2,000 iterations, batch 4
- ~20 hours on NVIDIA H200 GPU
- LR: deformation 0.006→0.00006, scale 0.006→0.00006, rotation 0.003→0.00003
- CFG scale 25→12
- Temporal reg weight 9.6→1.6
- Spatial reg weight 3000→300
- voxel size 通过 binary search 使得 |S_i| ≈ 7500

---

## 7. Experiments 深入分析

### 7.1 Baselines 对比

四个 baseline 代表四类方法：

1. **Animate3D [30]** (https://arxiv.org/abs/2403.12113) — multi-view video diffusion + 4D reconstruction
2. **AnimateAnyMesh [73]** (https://arxiv.org/abs/2506.09982) — feed-forward 4D foundation model，直接预测 mesh deformation，用 RF backbone
3. **MotionDreamer [64]** (https://openaccess.thecvf.com/content/CVPR2025/papers/Uzolas_MotionDreamer_Exploring_Semantic_Video_Diffusion_Features_for_Zero-Shot_3D_CVPR_2025_paper.pdf) — 先用 video diffusion 生成参考 video，然后通过 diffusion feature matching 把 mesh animate 到 video
4. **TrajectoryCrafter [84]** (https://arxiv.org/abs/2505.02702 等) — 重新定向 monocular video 的 camera trajectory，再做 4D reconstruction

这四个 baseline 对应不同 paradigm：
- Animate3D, TrajectoryCrafter: "generate video then reconstruct 4D"
- AnimateAnyMesh: "feed-forward 4D model"
- MotionDreamer: "feature matching against generated video"
- CHORD: "distill from video model directly"

### 7.2 Test scenes

6 个 scene：
1. "A man petting a dog"
2. "A cat stepping on a cushion"
3. "A sealion nudging a ball"
4. "A block falling on a trampoline"
5. "Two men shaking hands"
6. "A robot picking up a block"

涵盖 rigid/articulated/deformable 不同类型，多 object 交互。

### 7.3 Quantitative (Table 1)

| Method | Alignment ↑ | Realism ↑ | SA↑ | PC↑ |
|---|---|---|---|---|
| Animate3D | 0.34% | 0.51% | 3.83 | 3.42 |
| AnimateAnyMesh | 1.01% | 0.51% | 3.50 | **4.50** |
| MotionDreamer (DC) | 0.51% | 0.84% | 3.42 | 4.08 |
| MotionDreamer (Wan) | 0.84% | 0.34% | 3.50 | 3.83 |
| TrajectoryCrafter | 9.60% | 10.44% | 4.17 | 3.83 |
| **CHORD** | **87.71%** | **87.37%** | **4.33** | 4.25 |

变量：
- Alignment: user study 中认为此方法 best aligns with prompt 的比例
- Realism: user study 中认为此方法 best motion realism 的比例
- SA (Semantic Adherence): VideoPhy-2 [5] 自动评测 semantic 一致性
- PC (Physical Commonsense): VideoPhy-2 评测 physical 合理性

值得注意的点：
- CHORD user study 几乎垄断（87%+），但 baseline 全部 < 11%。
- AnimateAnyMesh 的 PC 4.50 反而最高——但 paper 指出这是因为它的常见 failure mode 是 object 保持静态（不动就 physical reasonable 但不 follow prompt）。这是一个很有意思的 metric 缺陷：PC 只评 physical 不评 alignment，"什么都不动" 得高分。
- CHORD SA 4.33 > TrajectoryCrafter 4.17，PC 4.25 < AnimateAnyMesh 4.50，整体 second in PC。

### 7.4 Additional Single-Object Comparison (Table 3)

Single-object mesh animation 上对比 5 个 prompt (chest closing, lamp lowering, scissors, tiger sitting, tiger walking)，50 participants user study：

| | Animate3D | AnimateAnyMesh | MD (DC) | MD (Wan) | TC | Ours |
|---|---|---|---|---|---|---|
| Alignment (raw avg) | 1.6 | 0.4 | 1.6 | 0.4 | 1.2 | 44.8 |
| Realism (raw avg) | 2.8 | 0.8 | 2.6 | 0.2 | 1.6 | 42.0 |

CHORD 89.6% alignment preference, 84% realism preference。在 single-object setting 上优势依然压倒性。

### 7.5 Ablation 深度解读

#### Ablation 1: Noise Level Sampling (Figure 8)

去掉 W-RFSDS 的 ŵ(τ) 采样，换成 uniform 采样 + w(τ) 加权，结果是 laptop "漂浮" 在 table 上方。intuition：uniform 采样下，高 τ (能驱动大 motion) 被 under-sample，低 τ (只能 fine-tune) 被过度采样，于是 model 没法学到 "把 laptop 真正放下" 的 substantial vertical motion。

#### Ablation 2: 4D Representation (Figure 9)

四个变体：
- Full: Fenwick + coarse + fine
- No Fenwick: late frame artifact 严重
- No fine: detailed motion (抓耳机) 丢失
- No coarse: distortions

intuition：每个 component 对应不同 scale 的 motion——coarse 处理 large rigid transform，fine 处理 local detail，Fenwick 处理 long-horizon coherence。三者各司其职，缺一不可。

#### Ablation 3: Regularization (Figure 10)

- No temporal: tail sudden appearance (flicker)
- No spatial: distortions

这两个 reg 是 SDS 优化稳定的"安全带"。

---

## 8. Applications

### 8.1 Long-Horizon Motion

用 last frame 当下一次 generation 的 input scene，autoregressive 生成更 long sequence。Figure 1 demo 了一个 4-action sequence。简单但 effective，主要靠 representation 的 locality 保证 frame-to-frame 一致性。

### 8.2 Real-World Scanned Object Animation

因为 distill 的 video model 是在 real-world video 上训练的，generated motion 自然具有 real-world 物理特性，所以 scanned object 也能 animate 而不需要 sim-real gap 处理。Figure 6 demo "A man closing the lid of a laptop"。

### 8.3 Robot Manipulation（这是我觉得最 exciting 的部分）

CHORD 的 dense object flow 可以直接当 robot manipulation guidance：

1. 用 AnyGrasp [17] (https://github.com/graspnet/anygrasp_sdk) 提议 grasp pose
2. robot 要么 grasp object 要么 move to a pushing pose（offset from grasp pose）
3. 用 rigid attachment forward model：end-effector 的 relative transformation 也 apply 到 object 的 initial points
4. motion planner (Pyroki [32], https://pyroki.github.io/) solve 一系列 end-effector pose，最小化：
   - transformed points vs. dense flow alignment
   - reachability
   - pose smoothness

这个 setup 处理三类 object：
- Rigid (第一行)
- Articulated (第二行)
- Deformable (第三、四行：fabric folding)

特别 impressive 的是 deformable object (fabric) 的 manipulation——传统 method 很难处理 deformable 因为 state space 太大，CHORD 直接给 dense flow 作为 guidance bypass 这个问题。

---

## 9. Limitations & Failure Cases

paper 列了三类 failure：

### 9.1 Video Model Limitation (Figure 12 第一行)

如果 video model 本身没法 sample 出符合 prompt 的 video，那 W-RFSDS gradient 就是 misleading 的。比如某些复杂 physics interaction，video model 自己都搞不定。

### 9.2 Newly Appearing Objects (Figure 12 第二行)

CHORD 只能 deform 初始 scene 里的 geometry，如果 prompt 要求 "倒水" 但场景里没有 water mesh，就没法生成。这是 fundamental limitation——pure deformation-based representation 没法 handle topology change or instantiation of new objects。

可能的 future work：在 optimization 过程中引入 geometry generation module（比如 score-based 3D generation）。

### 9.3 Training Cost

20 GPU-hours per scene on H200。瓶颈是 backprop through VAE [33]——paper 指出大部分 time 花在 VAE gradient 上。未来方向：avoid VAE backprop，因为目标只是 motion 不是 appearance。

这个 insight 跟最近一些 paper 一致，比如 4D Gaussian Splatting 系列都在 attack VAE/UNet backprop 的成本。一种可能：latent SDS (在 latent space 直接监督) 或 score-jacobian chaining (https://arxiv.org/abs/2402.08016) 类似的 trick。

---

## 10. 我自己的 Intuition 和 Thoughts

我想 share 几点个人思考：

### 10.1 SDS 在 RF 上的正确推导

paper 里 RF-SDS 的推导值得仔细看，因为现在 Sora、SD3、Wan 都是 RF-based，未来 4D generation 工作 almost surely 都要用 RF-SDS 而不是 DDPM-SDS。这个 derivation 应该会成为后续工作的 "standard reference"。

W-RFSDS 的 importance sampling trick 也很 elegant。其实这个 trick 在 DDPM-SDS 也能用——DDPM 也有 w(τ)，可以归一化采样。但 RF 的 forward process (linear interpolation) 让 importance sampling 的效果更明显，因为 velocity prediction 的 magnitude 随 τ 的变化更敏感。

参考 DreamTime [26] (https://arxiv.org/abs/2306.01417) 之前已经探讨过 noise schedule 对 SDS 的重要性，CHORD 把这个 idea 推到 RF 上。

### 10.2 Fenwick Tree 的"重新发现"

Fenwick tree 1994 年的 algorithm design data structure 被 "repurpose" 到 4D representation，这是 cross-domain idea transfer 的好例子。Fenwick tree 的核心 property "overlapping interval sharing" 跟 temporal coherence 是天然 fit。

我觉得这个 idea 可以推广：
- Multi-resolution temporal model：Fenwick tree 本身就是 hierarchical 的，low-bit-clear 操作把 frame 划分成不同 length 的 interval。这跟 wavelet 或 hierarchical temporal model 等价。
- 可能可以应用到 video generation 的 latent temporal modeling 上。

### 10.3 Hierarchical Control Points vs. Neural Deformation Field

CHORD 选了 control points 而不是 MLP deformation field（NeRFies [50] 那种），核心 trade-off：
- Control points: explicit, low-dim, gradient friendly, but less expressive
- MLP: high expressiveness, but optimization landscape 极度复杂，noisy SDS gradient 下容易 collapse

这个 trade-off 在 SDS-based 4D generation 普遍存在。Birth and Death of a Rose [21] (https://birth-death-of-a-rose.github.io/) 也是用 explicit control point 类似思路。SC-GS [27] 是 CHORD 的 spatial 设计来源。

### 10.4 SDS 的"假梯度"问题

SDS 严格意义上不是真正的 likelihood gradient（因为 stop-gradient 了 ∂ε̂/∂z 那项）。这个争议从 DreamFusion 一发表就有。CHORD 沿用这个 trick，paper appendix B 明确说 "we omit the term that backpropagates through the RF model"。

未来的方向可能是 Variational Score Distillation (VSD, https://arxiv.org/abs/2311.17904) 那种更 principled 的方法，但 VSD 需要 training 一个 auxiliary diffusion model，cost 高。CHORD 的 simplicity 反而是优点。

### 10.5 跟 Robotics 的 Connection

CHORD 的 robot manipulation demo 是我觉得最有 downstream value 的部分。dense object flow 作为 manipulation guidance 的 idea 可以推广：
- Imitation learning：用 CHORD 生成的 4D motion 当 demonstration data
- RL reward：用 CHORD 生成的 flow 作为 reward shaping signal
- Affordance learning：从 CHORD 生成的多 object interaction 里学 affordance

参考 Yang et al. "Tracking and Reconstructing Hand-Object Interactions" (https://handtracker.cs.brown.edu/) 等 HOI 工作的思路。

### 10.6 Multi-Object Interaction 的稀缺性

paper 提到一个 deep insight：现有 4D dataset 几乎都是 single-object，所以 end-to-end learning 方法 (AnimateAnyMesh) 在 multi-object setting 上完全失效。这暴露了一个 fundamental data problem。

CHORD 的解决方案是 distill from video model——video model 训练数据里有大量 multi-object interaction 视频，implicit knowledge 被 distill 到 4D 里。这是绕过 data scarcity 的 clever 方案。

未来可能：用 CHORD 反过来 generate synthetic 4D multi-object interaction dataset，再 fine-tune 一个 end-to-end 模型。这是 synthetic data bootstrapping 的循环。

### 10.7 跟 World Model 的关联

CHORD 本质上是在 build 一个 "mini world model"——给定 scene state 和 action description，predict next 4D state。这跟 Sora-style world model (https://openai.com/research/sora-system-card) 在 spirit 上类似，只是 CHORD 是 explicit 4D representation，Sora 是 implicit 2D video。

未来 convergence 方向：在 4D representation 上做 video model 的 grounding。CHORD 的 distillation pipeline 是这个方向的雏形。

参考 Genie 2 (https://deepmind.google/discover/blog/introducing-genie-2-a-large-scale-foundation-world-model/)、V-JEPA 2 (https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks-agentic-data/) 等近期 world model 工作。

### 10.8 关于"4D Foundation Model"

AnimateAnyMesh 提出 4D foundation model 概念，但 data 限制太大。CHORD 的范式其实指向另一种 "foundation" ——**用 2D video foundation model 当 4D generation 的 teacher**。这种 distillation-based foundation 可能比 end-to-end 4D foundation 更 scalable。

类比：LLM 时代之前，大家都想训 NLU foundation model，最后 GPT 用 next-token prediction 把一切统一了。4D 的"GPT moment" 可能也是某个 simple objective 加上 massive 2D supervision。

### 10.9 公式细节中的微妙之处

我注意到 Eq. 6 的 quaternion blending：

$$
\mathbf{q}^t = \left( \sum_{k \in \mathcal{N}} \beta_k r_k^t \right) \otimes \mathbf{q}
$$

这里 sum 是 unnormalized quaternion sum，然后用 ⊗ 应用到原 q。但 sum 可能不是 unit quaternion，Eq. 11 在 Fenwick query 里显式 normalize 了，但 Eq. 6 没显式 normalize。这可能是个隐含的 normalize step（实际实现里应该 normalize 了），或者 paper 略写。Quaternion 加权平均是经典 hard problem (https://www.geometrictools.com/Documentation/Quaternions.pdf)，naive linear average 不是最优。Future work 可能用 log-map averaging。

### 10.10 关于 Annealing Schedule 的细节

Eq. 4：h(τ_i) = 1 - i/(I+1)。

i=0 时 h(τ_0) = 1，对应 τ_0 = +∞ 的 CDF 值，也就是 τ_0 → max noise。i=I 时 h(τ_I) = 1/(I+1) ≈ 0，对应 τ_I → 0（接近 clean）。

paper 用 annealing 而不是 stochastic 采样，可能是为了 deterministic curriculum。 stochastic sampling 在 importance sampling 下也 work，但 annealing 在 iteration 早期就 push substantial motion 形成，后期 fine-tune，避免了 stochastic 采样可能让 motion "oscillate" 在不同 scale 之间。

### 10.11 关于 Implementation 的细节

- 41 frames deformation：典型的 short clip length，跟 Wan 2.2 训练视频长度一致
- 2000 iterations × batch 4 = 8000 gradient steps
- 20 GPU-hours on H200：单 scene 成本，跟 4D-fy [4] (https://sherwinbahmani.github.io/4dfy/) 等类似工作的 cost 相当
- VAE backprop 占大头——这个 cost bottleneck 是 community 共识

### 10.12 对 Paper 写作的小观察

paper 写得很清晰，method section 结构是：
- 3.1 SDS prelude
- 3.2 RF-SDS + W-RFSDS
- 3.3 Hierarchical 4D representation
- 3.4 Regularization

这个结构先讲 supervision signal 再讲 representation 再讲 reg，逻辑顺畅。Appendix B 把 RF-SDS derivation 完整给出，做得比较规范。

---

## 11. 相关参考链接汇总

为了方便你进一步探索，我把相关参考整理如下：

**核心方法相关：**
- CHORD project page: https://yanzhelyu.github.io/chord
- DreamFusion (SDS 原始 paper): https://dreamfusion3d.github.io/
- Rectified Flow: https://arxiv.org/abs/2209.03003
- Stable Diffusion 3 (RF architecture): https://arxiv.org/abs/2403.03206
- Wan 2.2 video model: https://arxiv.org/abs/2503.20314
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- gsplat library: https://github.com/rerun-io/gsplat or https://gsplat.github.io/
- SC-GS (sparse-control GS): https://yihua7.github.io/SC-GS/
- Fenwick tree (Wikipedia): https://en.wikipedia.org/wiki/Fenwick_tree
- Fenwick 1994 paper: https://www.sciencedirect.com/science/article/pii/0167642394900106

**4D Generation 相关：**
- 4D-Fy: https://sherwinbahmani.github.io/4dfy/
- Birth and Death of a Rose: https://birth-death-of-a-rose.github.io/
- TC4D: https://tc4d.github.io/
- Animate3D: https://arxiv.org/abs/2403.12113
- AnimateAnyMesh: https://arxiv.org/abs/2506.09982
- MotionDreamer: https://openaccess.thecvf.com/content/CVPR2025/papers/Uzolas_MotionDreamer_Exploring_Semantic_Video_Diffusion_Features_for_Zero-Shot_3D_CVPR_2025_paper.pdf
- TrajectoryCrafter: https://arxiv.org/abs/2505.02702
- Cat4D: https://cat-4d.github.io/
- Shape of Motion: https://shapeofmotion.github.io/
- DreamScene4D: https://dreamscene4d.github.io/

**Robotics & Manipulation：**
- AnyGrasp: https://github.com/graspnet/anygrasp_sdk
- Pyroki (kinematic optimization): https://pyroki.github.io/
- DiffRenderRobot (Differentiable Robot Rendering): https://arxiv.org/abs/2410.13851

**Diffusion/SDS Theory：**
- VSD (Variational Score Distillation): https://huang-yiwen.com/wp-content/uploads/2023/12/VSD.pdf
- DreamTime: https://arxiv.org/abs/2306.01417
- Score Jacobian Chaining: https://arxiv.org/abs/2402.08016
- Classifier-free Guidance: https://arxiv.org/abs/2207.12598

**Geometry Processing：**
- ARAP: https://www.cs.cmu.edu/~kmcrane/Projects/AsRigidAsPossible/
- Neural Jacobian Fields: https://neuraljacobianfields.github.io/

**World Models (相关 inspiration)：**
- Genie 2: https://deepmind.google/discover/blog/introducing-genie-2-a-large-scale-foundation-world-model/
- V-JEPA 2: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks-agentic-data/
- Sora: https://openai.com/research/video-generation-models-as-world-simulators

**Datasets：**
- Objaverse: https://objaverse.allenai.org/
- Objaverse-XL: https://objaverse.allenai.org/objaverse-xl
- AnimMate: https://anymate.github.io/
- AMASS: https://amass.is.tue.mpg.de/

---

## 12. 总结

CHORD 在我看来是一个 "elegant synthesis" paper：把 RF-SDS、coarse-to-fine control points、Fenwick tree、ARAP 这几个 component 拼起来，形成了一个 robust 的 multi-object 4D generation pipeline。每个 component 单独看都不算全新，但组合在一起 work 得很好的这个 fact 本身就是 contribution。

最重要的 takeaways：
1. **RF-SDS 的正确推导** — 后续 4D generation 工作的必备 building block
2. **W-RFSDS importance sampling** — 把 noise schedule 优化从 hyper-parameter 变成 principled 设计
3. **Fenwick tree 当 temporal representation** — 一个 brilliant cross-domain idea transfer
4. **Hierarchical control points** — coarse-to-fine 跟 annealing schedule 天然 pair
5. **Robot manipulation application** — dense object flow 作为 policy guidance，把 4D generation 跟 robotics 桥接起来

我自己的 guess 是后续工作会沿几个方向走：
- **Avoid VAE backprop** 来 reduce training cost（这是 paper 自己提的 future work）
- **Handle newly appearing objects** 通过 hybrid deformation + generation
- **End-to-end fine-tune** video model for 4D-aware generation（让 video model 直接输出 multi-view consistent video，然后做 4D reconstruction）
- **Real-time 4D generation** 通过 amortized inference（一旦 distill 出来一个 good 4D prior，可以 train 一个 feed-forward network）

如果你对哪个具体 component 想更深入讨论（比如 Fenwick tree 的 gradient flow、RF vs DDPM 的 SDS 推导细节、或者 robot manipulation 那个 motion planner 的具体 objective），可以告诉我，我再展开。
