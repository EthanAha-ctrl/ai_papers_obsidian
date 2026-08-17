---
source_pdf: UniK3D.pdf
paper_sha256: 8b3016bdf88ee91588a55e6b2e140249e53a67f65b5e927a1c76ce61cf3ee71a
processed_at: '2026-08-12T19:59:12-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# UniK3D 人话版

## 一句话总结

以前的单目深度估计方法都假设"相机是针孔的"，所以鱼眼、全景图这种大视野相机直接废掉。UniK3D 把整个问题搬到球面上来做，用一组数学基函数（球谐函数）来表示任意相机的光线方向，从而一个模型搞定从针孔到 360° 全景的所有相机。

paper: https://arxiv.org/abs/2411.15643
code: https://github.com/lpiccinelli-eth/unik3d

---

## 问题到底出在哪

先讲一个最朴素的道理。你拿手机拍一张照片，照片里一个人离你 3 米。网络看到这个人的像素尺寸，要推算距离。针孔相机下，物体在图像上的大小 ∝ 真实大小 / 深度（depth，也就是 z 轴距离）。这个关系很干净，所以网络好学。

但你换成一个 200° 的鱼眼镜头，事情就乱了。同一个 3D 点，在鱼眼图上的位置和针孔图完全不一样。更糟的是，当 FoV 超过 180°，光线开始往相机后面跑。这时候 depth 会变成负数、变成无穷大，disparity = 1/depth 直接爆炸，log(depth) 直接未定义。**数学上就根本没法表达**，跟模型多大、数据多少都没关系。

以前的方法比如 UniDepth [60]、Metric3D [85]、DepthPro [9]，本质上都先预测针孔相机参数（focal length 之类的），再用这些参数算每根光线方向。pinhole 参数本身只能描述直线投影的窄视野相机。你喂再多鱼眼数据进去，它学出来的还是 pinhole 的近似——这是 representation 层面的硬限制。

UniK3D 的核心 insight：**别预测针孔参数了，直接预测每根光线在球面上的方向**。

参考 UniDepth: https://arxiv.org/abs/2403.18913

---

## 为什么要用球面

想象你在相机中心，往四面八方看。你的视线方向可以画在以你为中心的单位球面上：方向用两个角度描述——polar angle θ（你和光轴的夹角）和 azimuthal angle φ（绕光轴旋转的角度）。

球面坐标天然覆盖所有方向。180°、360° 都没有死角。每个像素 (u,v) 对应球面上一个点 (θ, φ)，也就是一根光线。再加一个距离 r（从相机中心到 3D 点的欧氏距离），就能把这个 3D 点完全定下来：

$$\text{3D点} = r \cdot (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$$

这里 r 是 radial distance，不是传统的 depth。差别很大：

- depth d = r × cos(θ)：当 θ 接近 90°（光线几乎平行于图像平面），cos(θ) 接近 0，d 也接近 0。要表示一个 10 米远的点，depth 得从一个极小的数变化，梯度爆炸，网络根本学不动
- radial distance r：直接就是距离，和 θ 无关。网络看到图像上物体的角尺寸，就知道 1/r 大概是多少。这个关系在任何相机模型下都成立

paper 里管这个叫 "disentanglement of camera and scene geometry"。说人话：**用 r 代替 depth，把"物体在图上多大"和"离相机多远"这件事解耦得更干净**。用 depth 的话，图上物体尺寸同时依赖 r 和 θ，网络不知道是该调大 r 还是调小 θ。

---

## 怎么表示任意相机：球谐函数

球面上每个像素都有一根光线方向 (θ, φ)。最暴力的做法是每个像素直接预测一个方向，但这完全没有 inductive bias，数据少了根本学不动。

UniK3D 的做法是用球谐函数（Spherical Harmonics, SH）作为 basis。球谐函数是球面上的正交基，类比于圆周上的 Fourier 级数。任何定义在球面上的平滑函数都能展开成 SH 的线性组合。

具体来说，网络预测 15 个标量系数 H_lm，然后：

$$\text{光线方向} = \sum_{l=1}^{3} \sum_{m=-l}^{l} H_{lm} \cdot B_{lm}(\theta, \phi)$$

B_lm 是预定义的球谐基函数（Legendre 多项式），l 是 degree（1, 2, 3），m 是 order（-l 到 l）。3rd degree 去掉常数项后正好 3+5+7=15 个系数。

为什么 15 个就够？因为真实相机的光线方向随像素变化是平滑的，没有高频振荡。低阶 SH 就能很好近似。更高阶反而会 overfit 噪声。

再加 3 个 domain 参数（主点位置 cx, cy + 水平 FoV），总共 18 个数就能描述任何相机的光线分布。极度 compact。

为什么 SH 比其他方案好？paper 的 Table 4 做了对比：

- **Pinhole 模型**：Pano 上 F_A 只有 24.6，因为 pinhole 本质上表示不了 360°
- **Zernike 多项式**：这是光学里描述镜头像差用的，定义在单位圆盘上，inherently 是平面的，搞不了球面/全景。Pano F_A = 31.8
- **Non-parametric**（每像素预测一个 ray）：Pano F_A = 51.7，还行但不够好，因为纯 data-driven 对长尾泛化差
- **SH**：Pano F_A = 58.6，最好。既有 inductive bias（连续、可微、稀疏），又不假设特定相机模型

---

## 一个被低估的问题：FoV 收缩

paper 在 preliminary study 里发现一个诡异现象：训练数据里明明有各种 FoV 包括全景，但网络预测出来的 FoV 还是会塌缩到窄视野。原因很简单——公开数据集 90% 以上是针孔小视野图像，网络在回归时会偏向最常见的模式，忽略长尾。

简单重采样数据不管用，因为这会破坏场景多样性。UniK3D 的解决方案非常优雅——**用分位数回归损失（quantile regression loss）**：

$$L = \alpha \cdot |\hat{\theta} - \theta^*| \text{ (当 } \hat{\theta} > \theta^*) + (1-\alpha) \cdot |\hat{\theta} - \theta^*| \text{ (当 } \hat{\theta} \leq \theta^*)$$

说人话：如果网络预测的 polar angle 比真实值小（低估 FoV），和比真实值大（高估 FoV），给不同的惩罚权重。

paper 用 α=0.7。这意味着网络被推向预测第 70 百分位的 θ 值，systematically 倾向于预测更大的 angle，从而对抗小 FoV 的 bias。这个方法只需要搜一个 1D 超参，完全不改变数据分布，非常干净。

对于 azimuthal angle φ，用 α=0.5（标准 MAE），因为方位角相对主点是对称的，没有收缩偏差。

---

## 另一个隐藏瓶颈：Camera Conditioning

paper 发现第二个微妙问题：即使你给网络 ground-truth 的相机光线方向，网络也用不好这个信息。原因是 camera 信息和几何特征纠缠在一起，网络会走捷径，忽略 camera 输入。

UniK3D 用了 4 个 trick 来强制网络 listen to camera：

**Trick 1: 固定的 sinusoidal encoding**
相机光线在喂给 radial network 之前，经过一个固定的（不学习的）正弦编码。这样 camera 信息以结构化的形式进入网络，网络没法通过 learnable encoding 把它 shortcut 掉。

**Trick 2: Curriculum learning**
训练初期给 radial module 喂 GT 相机参数，后期逐渐过渡到预测的相机参数。概率公式：

$$p_{GT}(s) = 1 - \tanh(s / 10^5)$$

s 是训练步数。前 10 万步几乎全用 GT，让网络先学会"在完美相机条件下做 radial 预测"，再过渡到真实推理场景。

**Trick 3: Stop-gradient**
喂给 radial module 的 camera 信息是 detach 的。radial 的 loss 不会 backprop 到 angular module。这防止两个模块耦合——否则 radial 会通过 gradient "调整" angular 的输出，导致 angular 不再 faithful 于真实相机几何。

**Trick 4: 禁用 LayerScale**
LayerScale 是 CaiT [68] 引入的，给 residual block 输出乘一个可学习的标量。paper 发现在 cross-attention 里，LayerScale 会学到一个接近 0 的值，effectively 把 camera conditioning 关掉。禁用后网络被迫真正利用 camera 信息。

参考 CaiT: https://arxiv.org/abs/2103.17239

---

## 整体架构串一遍

输入一张图 H×W×3：

**Step 1: Encoder**
ViT backbone（DINO 预训练），输出 4 个分辨率的 dense features 和 4 个 class tokens。

**Step 2: Angular Module**
4 个 class tokens → project 成 18 个 token（3 个 domain + 15 个 SH 系数）→ 2 层 Transformer Encoder → 18 个标量。前 3 个定义 domain（HFoV, cx, cy），后 15 个是 SH 系数。用公式重建 dense ray map C ∈ R^(H×W×2)。

**Step 3: Radial Module**
Dense features + sine-encoded camera rays C → 4 层 Transformer Decoder（cross-attention，query 是 features，key/value 是 camera rays）→ FPN 上采样 → 输出 log-radius R_log ∈ R^(H×W) 和 log-confidence。

**Step 4: 组装输出**
C || R → spherical-to-Cartesian → point cloud O ∈ R^(H×W×3)。

**Loss**: angular loss（asymmetric, α=0.7 for θ）+ radial L1 loss（log 空间）+ confidence loss。

整个模型 ViT-L 版本 358M 参数，推理 88ms（518×518 input），比 UniDepth 快（146ms），比 DepthPro 快一个数量级（808ms）。

---

## 实验结果有多炸裂

Table 1 的 zero-shot 评测，13 个数据集分 4 个 domain：

**大视野（L.FoV, 120°-180°）**：
- UniDepth: F_A = 16.9
- DepthPro: F_A = 26.1
- UniK3D-Large: F_A = 71.6

提升 175%。其他方法在这个 domain 基本不可用，UniK3D 能 reasonable 重建。

**全景（Pano, 360°）**：
- UniDepth: F_A = 2.0
- DepthPro: F_A = 1.9
- UniK3D-Large: F_A = 80.2

其他方法完全失败，UniK3D 能用。相机光线精度 ρ_A 从 1.9 提升到 57.1，30 倍。

**小视野（S.FoV, 针孔）**：
- UniDepth: δ1^SSI = 94.9
- UniK3D-Large: δ1^SSI = 96.1

UniK3D 在传统 domain 也是最好。universal design 没有牺牲 specialized performance。

Table 2 更有意思：在 Stanford-2D3D（全景数据集）上，UniK3D 用 2% Matterport3D 采样训练，zero-shot 测试，超过了专门为 equirectangular 训练的 BiFuse++ 和 UniFuse。证明 general framework 在 specialized domain 也能赢。

参考 BiFuse++: https://arxiv.org/abs/2111.00179

---

## Ablation 告诉我们什么

**Data alone 不够**：加 distortion data 对 Pano 改善有限（pinhole model 下 5.9 → 3.0，反而变差），但换 representation（SH + radius）能从 5.9 跳到 53.8。Representation 比 data 更根本。

**SH + radius 必须协同**：单独换 radius（pinhole model）Pano 只从 5.9 → 10.1。单独换 SH（depth output）Pano 只从 5.9 → 10.9。两个一起换，Pano 从 5.9 → 53.8。这是 synergy。

**Conditioning design 很重要**：Add/Cat vs cross-attention，Pano 上 42.5 vs 58.6。Simple broadcasting 会被网络忽略，需要 selective retrieval。

---

## 我的直觉总结

**直觉 1: Representation 是天花板**
当 representation 有 inherent limitation，data 灌再多也突破不了。Pinhole 假设就是天花板，再多鱼眼数据也学不出真鱼眼。

**直觉 2: 解耦需要多管齐下**
解耦 camera 和 scene geometry 不只是改 output space。UniK3D 用了 spherical output + SH camera + stop-gradient + static encoding + asymmetric loss，至少 5 个机制协同。少一个都掉点。

**直觉 3: Inductive bias 在中等数据量下是资产**
SH 的连续性/可微/稀疏让 8M 样本就能泛化到全景。Non-parametric 方法需要 far more data。

**直觉 4: Conditioning 是隐藏瓶颈**
大家都关注 depth network capacity，但怎么让网络真正"听"camera 信息这个事被低估了。Table 8 显示 conditioning design 能造成 42.5 vs 58.6 的差距。

**直觉 5: Asymmetric loss 处理长尾很优雅**
与其重采样数据破坏场景多样性，不如在 loss 层面做分位数回归。1D 超参搜索，不改变数据分布。

---

## 局限和可能的改进

1. **数据多样性仍是瓶颈**：SH 理论上支持任何相机，但训练数据 panoramic/fisheye 比例小
2. **Camera augmentation 不够真实**：softmax splatting 生成的畸变图像有部分不 realistic
3. **Confidence 在 OOD 下不可靠**：regression task 通病

可能的改进方向：
- 用 diffusion model 生成 realistic 畸变图像
- 提高 SH degree（5th degree, 35 系数）处理 extreme distortion
- Adaptive degree 让网络自己决定用多少阶
- Multi-view consistency 自监督

---

## 相关工作链接

- UniK3D paper: https://arxiv.org/abs/2411.15643
- UniDepth (前作): https://arxiv.org/abs/2403.18913
- Depth Anything: https://arxiv.org/abs/2401.10891
- Depth Anything v2: https://arxiv.org/abs/2406.09414
- Metric3D: https://arxiv.org/abs/2307.10964
- Metric3D v2: https://arxiv.org/abs/2404.15506
- DepthPro: https://arxiv.org/abs/2410.02073
- MASt3R: https://arxiv.org/abs/2406.09756
- DUSt3R: https://arxiv.org/abs/2312.14132
- ZoeDepth: https://arxiv.org/abs/2302.12288
- DINOv2: https://arxiv.org/abs/2304.07193
- Spherical Harmonics: https://en.wikipedia.org/wiki/Spherical_harmonics
- Quantile regression: https://en.wikipedia.org/wiki/Quantile_regression
- CaiT (LayerScale): https://arxiv.org/abs/2103.17239

---

这篇 paper 最漂亮的地方在于，它把一个看似 engineering 的问题（支持多种相机）通过 spherical representation + SH basis 提升到了一个 principled 的几何框架。radial distance vs depth、SH 作为 camera-agnostic 表示、asymmetric loss 对抗收缩——每个 design choice 都有清晰的几何/统计直觉。最终一个 unified framework 同时达到 S.FoV、L.FoV、Pano 的 SotA，说明这个框架抓住了问题本质。

---

# UniK3D: Universal Camera Monocular 3D Estimation 深度解析

## 1. 大图景：这篇 paper 到底在解决什么问题

现有的 monocular metric depth estimation (MMDE) 方法几乎都隐含一个假设：**pinhole camera model**。这个假设在 FoV < 90° 的标准相机下还算合理，但当你遇到 fisheye（120°-180°）、panoramic（360° equirectangular）这些相机时，整个 pipeline 就崩了。问题出在两个层面：

**层面一：输出表示的数学 ill-posedness**

传统方法用 depth（z 轴距离）或者 disparity 作为输出。对于 pinhole 相机，depth d 和 image 上像素坐标 (u, v) 的关系是：

$$d = \frac{f \cdot Z}{...}$$

但当 FoV 超过 180° 时，沿着 optical axis 反方向的点，其 depth 会变成负数或者无穷大。 disparity = 1/d 在 d→0 时爆炸， log-depth = log(d) 在 d≤0 时直接未定义。这是**数学上就根本没法表示**的问题，跟模型能力无关。

**层面二：camera modeling 的限制**

即使像 UniDepth [60] 那样把 camera prediction 和 depth estimation 解耦，它还是先预测 pinhole 参数（focal length, principal point），然后把这些参数 encode 成 rays。问题在于：pinhole 模型本身就只能表示 FoV < 180° 的直线投影相机。你给它再多 fisheye 数据，它学出来的也是 pinhole 的近似，本质上是 wrong inductive bias。

UniK3D 的核心 insight 是：**彻底抛弃 parametric camera model，直接在 spherical space 里建模 rays**。这样从 pinhole 到 panoramic 就是同一个表示空间里的不同点，模型可以无缝泛化。

paper: https://arxiv.org/abs/2411.15643
code: https://github.com/lpiccinelli-eth/unik3d

---

## 2. 核心 insight：为什么 spherical representation

### 2.1 Output space 的选择：radial distance vs depth

这是整个方法最关键的 insight，需要仔细拆解。

假设有一个 3D 点 P，在 spherical coordinates 下表示为 $(r, \theta, \phi)$，其中：
- $r$ = radial distance（点到 camera center 的欧氏距离）
- $\theta$ = polar angle（与 optical axis 的夹角，从 z 轴量起）
- $\phi$ = azimuthal angle（在 xy-plane 上的方位角）

在 Cartesian 下，$z = r \cos\theta$，所以传统的 depth $d = z = r\cos\theta$。

现在考虑 image 上一个 object 的投影大小。对于 pinhole 相机，object 在 image plane 上的尺寸 $\propto \frac{\text{object size}}{d}$，这里 $d$ 是 depth。但这个关系**只在 pinhole 假设下成立**。更一般地，对于任意相机模型，object 在 image 上的投影尺寸是 $\frac{\text{object size}}{r}$ 的 univocal function（这里 r 是 radial distance），**与 θ 无关**。

为什么？因为从 camera center 出发，沿着一根 ray 看出去，object 的 angular size 取决于 object 距离 camera center 有多远，也就是 $r$。而 $\theta$（ray 的方向）只影响 object 投影到 image 的哪个位置，不影响投影尺寸。这是一个**几何不变量**：无论相机是什么投影模型，angular size $\propto 1/r$。

反观 depth $d = r\cos\theta$：object 在 image 上的尺寸同时依赖于 $r$ 和 $\theta$。这意味着同一个 image projection 可以对应不同的 $(r, \theta)$ 组合（只要 $r\cos\theta$ 相同）。**网络在 decode 时遇到了 ambiguity**：它看到 image 上一个 object 的尺寸，无法唯一确定这是大的 $r$ + 小的 $\theta$ 还是小的 $r$ + 大的 $\theta$。

用 $r$ 作为 output 就把这个 ambiguity 消除了。这就是 paper 里说的 "disentanglement of camera and scene geometry"。

**数值稳定性**：当 $\theta \to 90°$（即接近 xy-plane），$\cos\theta \to 0$，$d = r\cos\theta \to 0$。这意味着要表示一个 $r$ 很大的点（比如 10 米外），depth 要从一个很小的数变化，gradient $\frac{\partial d}{\partial \theta} = -r\sin\theta$ 在 $\theta = 90°$ 时达到 $-r$，极大。网络在这个区域极难训练。用 $r$ 直接就没有这个问题。

### 2.2 Spherical 框架对 panoramic 的天然支持

当 FoV = 360°（panoramic），所有方向的 ray 都要被覆盖。spherical coordinates $(\theta, \phi)$ 在 $\theta \in [0, \pi], \phi \in [0, 2\pi)$ 下完整覆盖整个 unit sphere $S^2$。paper 里写的是 $\mathbb{S}^3$，这里我理解是指 3D 空间中的单位球面（严格说应该是 $S^2$，可能 paper 笔误，或者他们用 $\mathbb{S}^3$ 表示 3D sphere space 概念）。

从 spherical 表示到 Cartesian point cloud 的转换是 bijective 的：

$$\mathbf{O}(u,v) = r(u,v) \cdot \begin{pmatrix} \sin\theta(u,v)\cos\phi(u,v) \\ \sin\theta(u,v)\sin\phi(u,v) \\ \cos\theta(u,v) \end{pmatrix}$$

每个像素 $(u,v)$ 对应一根 ray $(\theta, \phi)$ 和一个距离 $r$，直接 unproject 成 3D 点。这个框架对任何 FoV 都成立，180°、360° 都没区别。

---

## 3. Camera Representation：Spherical Harmonics Basis

这是第二个核心创新。让我详细拆解。

### 3.1 传统方法的问题

UniDepth [60] 的做法是：预测 pinhole 参数 $(f_x, f_y, c_x, c_y)$ → 对每个像素 $(u,v)$ 计算 ray direction → 把 ray direction 用 SH basis encode 成高维特征 → 用这个特征 condition depth network。

这里有个根本矛盾：**pinhole 参数本身就限制了能表达的相机类型**。即使后面的 SH encoding 再灵活，输入端的 pinhole 假设已经把信息 bottleneck 住了。给一个 fisheye 图像，网络预测出来的 pinhole 参数是某种 "best fit pinhole"，但这个 fit 本身就是有损的。

### 3.2 UniK3D 的做法：直接预测 SH 系数

UniK3D 跳过 pinhole 参数这一步，**直接预测 SH 系数**，然后通过 inverse SH transform 重建 rays。

公式 (1)：

$$\mathbf{C} = \mathcal{F}_{B}^{-1}\{\mathbf{H}\} = \sum_{l=0}^{L} \sum_{m=-l}^{l} \mathbf{H}_{lm} \mathbf{B}_{lm}(\theta, \phi)$$

变量解释：
- $\mathbf{C}$：reconstructed angular field，即每个像素对应的 ray direction $(\theta, \phi)$，shape 是 $\mathbb{R}^{H \times W \times 2}$（分别对应 θ 和 φ 两个 channel）
- $\mathcal{F}_{B}^{-1}$：从 coefficient space 到 angular space 的 inverse transform
- $\mathbf{H}$：predicted coefficients tensor，网络直接输出这个
- $\mathbf{H}_{lm}$：degree $l$、order $m$ 对应的 SH 系数，是一个标量
- $\mathbf{B}_{lm}(\theta, \phi)$：SH basis functions，即 Legendre polynomials 的组合
- $l$：degree（0, 1, 2, 3，对应不同的 angular frequency）
- $m$：order（$-l \leq m \leq l$，在 degree $l$ 下有 $2l+1$ 个 order）

### 3.3 为什么 SH 是好的 basis

Spherical Harmonics 是 unit sphere 上的 orthonormal basis，类比于 Fourier series 在 circle 上的角色。任何定义在 sphere 上的 square-integrable 函数都可以展开成 SH 的线性组合。

对于 camera rays 来说：
- **Continuity**: SH 是连续函数的线性组合，保证了 predicted rays 是空间连续的（相邻像素的 ray 方向不会突变）
- **Differentiability**: SH 是解析的，gradient 可以解析计算，利于反向传播
- **Sparsity**: 低频的 camera distortion 用低 degree 就能很好地近似。3rd degree 去掉 $l=0$ 的 constant component 后，剩下 $\sum_{l=1}^{3}(2l+1) = 3 + 5 + 7 = 15$ 个系数。这 15 个系数就足以表示绝大多数真实相机的 ray 分布
- **Generality**: SH 不假设任何特定的 camera model，pinhole、fisheye、panoramic 都是 SH 空间里的不同点

### 3.4 Domain 参数：如何定义 SH 的作用域

SH basis 的 domain 由 4 个参数定义：
- **Principal point (pole)**: 2 个参数 $(c_x, c_y)$，是 SH 展开的 "center"，对应广义的 principal point
- **Horizontal FoV (HFoV)**: 1 个参数，定义 azimuthal $\phi$ 的范围
- **Vertical FoV (VFoV)**: 1 个参数，但 paper 假设 square pixels，所以 $\text{VFoV} = \text{HFoV} \times \frac{H}{W}$，直接从 HFoV 推导

所以总共需要预测 3 个 domain 参数 + 15 个 SH 系数 = 18 个 token。

具体计算：
$$\text{HFoV} = 2\pi \cdot \sigma(\mathbf{T}_0)$$

$$c_x = \frac{\sigma(\mathbf{T}_1) \cdot W}{2}, \quad c_y = \frac{\sigma(\mathbf{T}_2) \cdot H}{2}$$

其中 $\sigma$ 是 sigmoid，$\mathbf{T}_0, \mathbf{T}_1, \mathbf{T}_2$ 是 network 预测的前 3 个 token。sigmoid 保证输出在合理范围（HFoV ∈ (0, 2π)，principal point 在 image 范围内）。

### 3.5 与其他 camera model 的对比

paper 的 Table 4 ablation 比较了 4 种 camera model：

| Model | S.FoV F_A | Pano F_A |
|-------|-----------|----------|
| Pinhole | 55.5 | 24.6 |
| Zernike | 56.6 | 31.8 |
| Non-Parametric | 56.4 | 51.7 |
| SH | 57.3 | 58.6 |

观察：
- **Pinhole** 在 Pano 上彻底失败（24.6），因为 pinhole 本质上无法表示 360°
- **Zernike polynomial** 是镜头 aberration 的经典表示，但它是定义在 unit disk 上的，inherently planar，无法表示 spherical/equirectangular geometry。所以 Pano 表现也差
- **Non-parametric**（每像素预测一个 ray）在 Pano 上中等（51.7），但在 L.FoV 上差。原因是纯 data-driven，对 distribution tails（L.FoV, Pano）泛化差，需要大量数据
- **SH** 在所有 domain 上都是最好或接近最好的，因为它既有 basis 的 inductive bias（continuity, sparsity），又不假设特定 camera model

---

## 4. Distribution Contraction：一个被低估但关键的问题

### 4.1 问题是什么

paper 在 preliminary study 里发现一个现象：即使训练数据包含各种 FoV（包括 panoramic），网络预测出来的 FoV 仍然会 **contract 到一个窄的 range**，倾向于预测 small-FoV。这是因为训练数据中 small-FoV pinhole images 占绝大多数（几乎所有公开 dataset 都是 pinhole 的），网络在 regression 时会 regress 到 most frequent mode，忽略 distribution 的 tails。

简单的 data rebalancing 不管用，因为：
- 改变采样概率会破坏 3D scene 的 diversity
- 跨多个 dataset 调整采样极其复杂
- 本质问题是 **angular distribution 的长尾**，不是场景分布的长尾

### 4.2 Asymmetric Angular Loss

公式 (2)：

$$\mathcal{L}_{\mathrm{AA}}^{\alpha}(\hat{\theta}, \theta^{*}) = \alpha \sum_{\hat{\theta} > \theta^{*}} |\hat{\theta} - \theta^{*}| + (1-\alpha) \sum_{\hat{\theta} \leq \theta^{*}} |\hat{\theta} - \theta^{*}|$$

变量：
- $\alpha \in [0,1]$：target quantile，控制 over/under-estimation 的权重
- $\hat{\theta}$：predicted polar angle
- $\theta^{*}$：ground-truth polar angle

这是一个 **pinball loss / quantile regression loss**。当 $\hat{\theta} > \theta^*$（over-estimation，预测的 angle 比 GT 大，即预测的 FoV 更大），loss 权重是 $\alpha$。当 $\hat{\theta} \leq \theta^*$（under-estimation），权重是 $1-\alpha$。

关键：$\alpha = 0.7$ for $\theta$（polar angle）。这意味着网络如果**低估** polar angle（$\hat{\theta} \leq \theta^*$，即预测的 FoV 比 GT 小），loss 权重是 $1 - 0.7 = 0.3$，**惩罚更轻**。相反，如果高估 polar angle（$\hat{\theta} > \theta^*$，预测 FoV 比 GT 大），loss 权重是 $0.7$，**惩罚更重**。

等等，这个逻辑反了？让我再想想。如果训练数据 skewed 到 small FoV，网络倾向于 predict 小 angle，也就是 under-estimate。我们想要 push 网络 predict 更大的 angle，应该 **惩罚 under-estimation 更重**，即 $\alpha < 0.5$。

但 paper 用 $\alpha = 0.7 > 0.5$，这是在惩罚 over-estimation 更重。Hmm。

让我重新理解 quantile regression 的 semantics。Quantile regression at quantile $\alpha$ 会 regress 到第 $\alpha$ 个 quantile。如果 $\alpha = 0.7$，网络会预测一个**比 median 更大**的值（第 70 percentile），也就是**倾向于高估**。

具体来说，pinball loss 的最优解是 conditional quantile。当 $\alpha = 0.7$，loss minimum 在 $\hat{\theta} = $ 第 70 percentile of $\theta^*$。所以网络被 push 到 predict 更大的 angle，这正是我们想要的——对抗 small-FoV 的 bias，让网络愿意 predict 大 FoV。

所以 $\alpha = 0.7$ 的含义是：让网络预测 polar angle 的第 70 percentile，systematically 高估一点，避免 contract 到 small FoV。这与直觉一致。

对于 azimuthal angle $\phi$，paper 用 $\alpha = 0.5$（standard MAE），因为 azimuth 相对于 principal point 是 symmetric 的，没有 contraction bias。

最终 angular loss 公式 (3)：

$$\mathcal{L}_{\mathrm{A}}(\hat{\mathbf{C}}, \mathbf{C}^{*}) = \beta \mathcal{L}_{\mathrm{AA}}^{0.7}(\hat{\theta}, \theta^{*}) + (1-\beta) \mathcal{L}_{\mathrm{AA}}^{0.5}(\hat{\phi}, \phi^{*})$$

$\beta = 0.75$，polar angle 的权重更高，因为 polar angle 直接决定 FoV，是 contraction 的主要维度。

### 4.3 Camera Conditioning 的增强

paper 发现另一个 subtle 的问题：即使给网络 GT camera rays 作为 input，网络也**不能有效利用**这些信息来 condition radial prediction。原因是 weak conditioning：网络把 camera 信息和 geometric features 纠缠在一起，没有真正 disentangle。

解决方案有 4 个 components：

**(1) Static (non-learnable) encoding of camera rays**

camera rays $\mathbf{C}$ 在喂给 Radial Module 之前，先经过一个固定的 sinusoidal encoding（类似 NeRF 的 positional encoding），不学习。这样 camera 信息以 explicit 的结构化形式进入网络，网络不能通过 learnable 的 encoding 来 "shortcut" 掉 camera 信息。

**(2) Curriculum learning**

GT camera 喂给 Radial Module 的概率：

$$p_{\text{GT}}(s) = 1 - \tanh\left(\frac{s}{10^5}\right)$$

其中 $s$ 是当前 optimization step。训练初期 $s$ 小，$\tanh$ 接近 0，$p_{\text{GT}} \approx 1$，几乎全用 GT camera。随着 $s$ 增大，$p_{\text{GT}} \to 0$，过渡到用 predicted camera。

这给网络一个 "warm start"：先学会在 perfect camera 条件下 predict radial，再学会在 noisy predicted camera 条件下 robust。

**(3) Stop-gradient on camera output**

喂给 Radial Module 的 camera 信息是 **detached** 的（`stop-gradient`）。这意味着 Radial Module 的 loss 不会 backprop 到 Angular Module。目的是防止 Radial Module 通过 feedback 来 "调整" Angular Module 的输出，导致两个 module 耦合，conditioning 失效。Angular Module 只通过自己的 angular loss 训练。

**(4) Disable LayerScale in cross-attention**

LayerScale [68] 是 CaiT 引入的，给 residual block 的 output 乘一个 learnable scalar。paper 发现在 cross-attention layer 里，LayerScale 会学到一个接近 0 的 scalar，**effectively 把 camera conditioning 关掉**，变成 shortcut。禁用 LayerScale 后，cross-attention 被迫真正利用 camera 信息。

### 4.4 Gradient scaling

paper 提到一个经验观察：camera-induced gradient 对 encoder weights 的 magnitude 大约是 radial-induced gradient 的 10x。如果不处理，encoder 会被 angular loss dominate。解决方案是把从 Angular Module 流向 class tokens 的 gradient **乘以 0.1**。这平衡了两个 task 的学习。

---

## 5. Architecture 细节

让我把整个 pipeline 串起来（参考 Figure 2）：

### 5.1 Encoder

- ViT backbone（Small/Base/Large），初始化用 DINO-pretrained weights
- 移除最后 3 层（pooling, FC, softmax）
- 从 last 4 layers 提取 dense features $\mathbf{F} \in \mathbb{R}^{h \times w \times C \times 4}$ 和 class tokens $\mathbf{T}$
- $(h, w) = (H/14, W/14)$，即 patch size 14
- features 和 class tokens 分别经过 LayerNorm + linear projection 到统一 channel dimension（Large: 512, Base: 384, Small: 256）
- **关键**：不同分辨率的 features 和 class tokens 用**不同的** normalization 和 projection 权重，不 share

### 5.2 Angular Module

输入：4 个 class tokens（来自 last 4 layers）
处理流程：
1. Project 到 dimension 3D, 3D, 5D, 7D（分别对应 1st, 2nd, 3rd degree SH 的 token 数，但这里数字有点奇怪，让我算一下：1st degree 3 个 + domain 3 个 = 6？2nd degree 5 个？3rd degree 7 个？加起来 3+3+5+7=18，对，总共 18 个 token）
2. 实际上 paper 说的是 chunks of size 3, 3, 5, 7 based on channel dimension d。第一个 3 是 domain tokens，后面 3+5+7=15 是 SH 系数（1st degree 3 个，2nd degree 5 个，3rd degree 7 个）
3. 18 个 token 经过 2 层 Transformer Encoder（8 heads, MLP hidden dim $4C$, GELU, residual connections）
4. 每个 token project 到 scalar
5. 前 3 个 token → domain parameters (HFoV, $c_x$, $c_y$)
6. 后 15 个 token → SH coefficients $\mathbf{H}_{lm}$
7. 根据 domain parameters 计算 SH basis $\mathbf{B}_{lm}(\theta, \phi)$
8. 公式 (1) 的 inverse transform：$\mathbf{C} = \sum \mathbf{H}_{lm} \mathbf{B}_{lm}$，得到 dense ray map $\mathbf{C} \in \mathbb{R}^{H \times W \times 2}$

### 5.3 Radial Module

输入：dense features $\mathbf{F}$（4 个分辨率）+ sine-encoded camera rays $\mathbf{C}$
处理流程：
1. 4 个并行的 Transformer Decoder layers，每个对应一个分辨率，1 head
2. Cross-attention：query 是 dense features $\mathbf{F}$，key/value 是 sine-encoded $\mathbf{C}$
3. **没有 LayerScale**，只有 residual connection
4. Conditioned features 经过 FPN-style 处理：
   - 最深 features → 2 个 ResNet blocks → bilinear upsample → project halve channels
   - 与上一层 features 融合（上一层也 project + 2x2 transposed conv upsample）
   - 重复直到所有 4 个分辨率用完
5. 最终 upsample 到 input resolution $H \times W$
6. Project 到 single channel → $\mathbf{R}_{\log}$（log-radius）
7. 另一个相同架构但 separate weights 的 head → $\Sigma_{\log}$（log-confidence）
8. Element-wise exponentiation 得到 $\mathbf{R}$ 和 $\Sigma$

### 5.4 最终输出

Concatenate camera 和 radial：$\mathbf{O} = \mathbf{C} \| \mathbf{R}$
Spherical-to-Cartesian transform 得到 point cloud $\mathbf{O} \in \mathbb{R}^{H \times W \times 3}$

### 5.5 复杂度

Table 7 的数据（ViT-L backbone，518×518 input）：

| Method | Latency (ms) | Params (M) |
|--------|-------------|------------|
| DepthAnything v2 | 78.1 | 334.7 |
| UniDepth | 146.4 | 347.0 |
| DepthPro | 808.1 | 952.0 |
| UniK3D | 88.4 | 358.8 |
| - Radial Module | 21.9 | 38.2 |
| - Angular Module | 3.1 | 12.1 |

UniK3D 比 UniDepth 快很多（88 vs 146 ms），只比 DepthAnything v2 慢 10ms。Angular Module 只增加 3.1ms 和 12M params，非常轻量。DepthPro 巨慢（808ms）且参数巨大（952M），是因为它用 1536×1536 input + multi-resolution 架构。

---

## 6. Losses 和 Optimization

### 6.1 三个 loss 的组合

$$\mathcal{L} = \mathcal{L}_{\mathrm{A}} + \eta \mathcal{L}_{\mathrm{rad}} + \gamma \mathcal{L}_{\mathrm{conf}}$$

- $\mathcal{L}_{\mathrm{A}}$：angular loss，公式 (3)，$\beta = 0.75$
- $\mathcal{L}_{\mathrm{rad}} = \|\hat{\mathbf{R}}_{\log} - \mathbf{R}_{\log}^{*}\|_1$：radial L1 loss（在 log space）
- $\mathcal{L}_{\mathrm{conf}} = \| |\hat{\mathbf{R}}_{\log} - \mathbf{R}_{\log}^{*}| - \Sigma \|_1$：confidence loss，让 $\Sigma$ 预测 detached 的 radial error
- $\eta = 2$，$\gamma = 0.1$

radial loss 权重 2.0 是 angular 的 2 倍，说明 radial（scene geometry）是主任务，angular 是辅助但必要的。

**为什么用 log-space for radial**：log-space 让网络在近处和远处都有 reasonable gradient。如果直接预测 $r$，近处的 error 在 loss 里占比极小（因为 $r$ 本身小），网络会忽略近处。log-space 下，$\log(r)$ 的 error 在近处远处 scale 类似。

**Confidence 的设计**：$\Sigma$ 被训练去预测 detached radial error $|\hat{\mathbf{R}}_{\log} - \mathbf{R}_{\log}^{*}|$。detach 防止 confidence loss 影响 radial prediction。confidence 主要用于 downstream tasks（比如 fusion, uncertainty-aware planning）。

### 6.2 Optimization 细节

- Optimizer: AdamW, $\beta_1 = 0.9, \beta_2 = 0.999$
- Initial LR: $5 \times 10^{-5}$，encoder LR 是 1/10（$5 \times 10^{-6}$）
- Weight decay: 0.1
- LR scheduler: Cosine annealing to 1/10，从 30% training 开始
- 250k iterations，batch size 128
- 16x NVIDIA 4090，6 天
- EMA: 0.9995
- 16-bit float

### 6.3 Data augmentation

- 几何：random resize, random crop, aspect ratio ∈ [1:2, 2:1], random zoom [0.5, 2.0], random translation [-0.05, 0.05]
- 光度：color jitter (80%, intensity [0, 0.5]), gamma (80%, [0.5, 1.5]), greyscale (20%), Gaussian blur (20%, sigma [0.1, 2.0])
- **Camera augmentation**：用 pinhole 图像 + 预测 depth 来 simulate distorted cameras。具体流程：unproject 2D depth 到 3D point cloud → project 到 random distorted camera model (EUCM, Fisheye624, Kannala-Brandt) → 计算 deformation field → softmax splatting warp image。只在 10k steps 后启用（需要 decent depth prediction）

---

## 7. 实验结果解读

### 7.1 主结果（Table 1）

这是 zero-shot evaluation on 13 datasets，分 4 个 domain：

| | S.FoV | L.FoV | Pano |
|---|---|---|---|
| | δ1^SSI / F_A / ρ_A | δ1^SSI / F_A / ρ_A | δ1^SSI / F_A / ρ_A |
| UniDepth | 94.9 / 59.0 / 85.0 | 68.6 / 16.9 / 19.8 | 33.0 / 2.0 / 1.7 |
| DepthPro | 87.4 / 56.0 / 79.6 | 64.5 / 26.1 / 32.1 | 31.8 / 1.9 / 1.9 |
| **UniK3D-Large** | **96.1 / 68.1 / 89.4** | **91.2 / 71.6 / 81.9** | **81.4 / 80.2 / 57.1** |

关键观察：

**L.FoV 上 UniK3D 的巨大优势**：F_A 从 26.1（DepthPro）到 71.6，提升 **175%**。这说明 UniK3D 的 spherical + SH 设计在 wide-FoV 下是真的一枝独秀。UniDepth 在 L.FoV 上 F_A 只有 16.9，因为它本质还是 pinhole，在 large FoV 下 pinhole assumption 彻底失效。

**Pano 上的绝对优势**：UniK3D F_A = 80.2，其他方法都在 1-10 范围。这意味着其他方法在 panoramic 上**基本上完全失败**，而 UniK3D 能 reasonable reconstruct。ρ_A 从 1.9（DepthPro）到 57.1，camera ray 预测精度提升 30x。

**S.FoV 上没有 trade-off**：UniK3D 在传统 pinhole domain 仍然是 best（δ1^SSI 96.1 vs UniDepth 94.9）。这说明 universal design 没有牺牲 specialized performance。

### 7.2 与 equirectangular-specialized 方法比较（Table 2）

| Method | δ1 | A.Rel |
|---|---|---|
| BiFuse++ | 91.4 | 10.7 |
| UniFuse | 91.3 | 9.42 |
| UniK3D | 96.8 | 8.01 |

UniK3D 用 2% Matterport3D 采样训练，在 Stanford-2D3D 上 zero-shot 测试，超过了专门为 equirectangular 设计训练的 BiFuse++/UniFuse。这证明 UniK3D 的 general framework 在 specialized domain 也能赢。

### 7.3 Fine-tuning 结果（Table 16, 17）

在 KITTI 上 fine-tune 后：

| Method | δ1 | A.Rel | RMS |
|---|---|---|---|
| Metric3Dv2 | 98.5 | 4.40 | 1.99 |
| DepthAnything v2 | 98.3 | 4.50 | 1.86 |
| UniK3D | 99.0 | 3.69 | 1.68 |

A.Rel 3.69 是非常强的数字。这说明 UniK3D 的 pretrain 提供了好的 initialization，fine-tune 到 specific domain 也能达到 SotA。

---

## 8. Ablation 的关键发现

### 8.1 Data 的影响（Table 3）

比较 training data 有无 distorted cameras：
- 加 distortion data 在 S.FoV_Dist 上 F_A 从 31.7 → 40.4（Pinhole model）
- 加 distortion data 在 Pano 上改善有限（5.9 → 3.0，反而变差），因为 log-depth 表示本身就不适合 panoramic

这说明：**data alone 解决不了根本问题**，需要 representation 的改变。

### 8.2 Camera Model 的影响（Table 4）

SH > Zernike > Non-Parametric > Pinhole（在 L.FoV 和 Pano 上）

在 S.FoV 上差别不大（55-57），因为这些 camera model 在 small FoV 下都能 well-approximate pinhole。差别体现在 tails：SH 在 Pano 上 F_A 58.6，Pinhole 只有 24.6。

### 8.3 Output Representation 的影响（Table 5）

比较 depth vs radius 作为 output 第三维：

| Model | Output | Pano F_A | L.FoV F_A |
|---|---|---|---|
| Pinhole | depth | 5.9 | 44.9 |
| Pinhole | radius | 10.1 | 44.4 |
| SH | depth | 10.9 | 48.5 |
| SH | radius | 53.8 | 51.8 |

关键发现：**radius 只在 SH camera model 下才有效**。Pinhole + radius 在 Pano 上只从 5.9 → 10.1，因为 pinhole camera 本身无法表示 panoramic rays，output representation 改了也没用。SH + radius 才是 synergy，从 10.9 → 53.8，5x 提升。

这验证了 paper 的核心 thesis：**camera representation 和 output representation 需要协同设计**。单独改一个不够。

### 8.4 组件的影响（Table 6）

| L_AA | Cond | S.FoV_Dist F_A | L.FoV F_A | Pano F_A |
|---|---|---|---|---|
| ✗ | ✗ | 35.0 | 51.8 | 53.8 |
| ✓ | ✗ | 39.5 | 52.9 | 56.1 |
| ✓ | ✓ | 44.6 | 53.5 | 58.6 |

- Asymmetric loss 单独：S.FoV_Dist +4.5, Pano +2.3
- Conditioning 单独（在 L_AA 基础上）：S.FoV_Dist +5.1, L.FoV +0.6, Pano +2.5
- 两者有 synergy

### 8.5 Camera conditioning design（Table 8）

| Cond | S.FoV_Dist F_A | Pano F_A |
|---|---|---|
| Add | 26.3 | 42.5 |
| Cat | 28.7 | 42.3 |
| Prompt (cross-attn) | 44.6 | 58.6 |

Add 和 Cat 远远不如 cross-attention based conditioning（"Prompt"）。Add 在 Pano 上只有 42.5，Prompt 是 58.6。这说明 camera 信息需要被 **selectively retrieved**，simple broadcasting 会让网络忽略它。

---

## 9. 一些值得深挖的 design choice

### 9.1 为什么 stop-gradient on camera

这个设计很微妙。如果 Radial Module 的 loss 能 backprop 到 Angular Module，会发生什么？

想象一个 failure mode：Radial Module 发现某个区域的 radial prediction 不准，它会尝试通过 gradient "调整" Angular Module 的 ray prediction，让 ray 方向变一变来弥补 radial 的误差。这会导致 Angular Module 的输出不再 faithful 到真实 camera geometry，而是变成一个 "为 radial loss 服务的" 畸形 ray map。

Stop-gradient 切断这条路，强迫 Angular Module 只通过 angular loss 学习真实 camera，Radial Module 必须在给定（detached）camera 条件下尽量做好 radial。这保证了解耦。

### 9.2 为什么 3rd degree SH 够用

SH 的 degree 决定了能表示的 angular frequency。3rd degree 最高能表示 $l=3$ 的 angular variation，对应 3 个 "振荡周期" 在 sphere 上。

对于真实相机，ray direction 随像素的变化是 smooth 的（没有高频振荡），3rd degree 足够。更高 degree 会 overfit 噪声，且增加参数。15 个系数 + 4 个 domain = 19 个参数，极度 compact。

### 9.3 Curriculum learning 的 schedule

$p_{\text{GT}}(s) = 1 - \tanh(s/10^5)$

这个 schedule 的特点：
- $s = 0$：$p = 1$，全 GT
- $s = 10^5$：$p = 1 - \tanh(1) \approx 0.24$，主要 predicted
- $s = 2 \times 10^5$：$p \approx 0.004$，几乎全 predicted
- $s = 2.5 \times 10^5$（训练结束）：$p \approx 0.0009$

整个 training 250k steps，所以前 ~100k steps 主要是 GT camera，后 150k steps 过渡到 predicted camera。这给 Radial Module 足够时间学会利用 camera 信息，再过渡到 real inference scenario。

---

## 10. 我的 intuition 总结

让我试着提炼几个 take-away intuition：

**Intuition 1: Representation 比 Data 更重要**

Table 3 和 Table 5 的对比说明，单纯加 distorted camera data 对 Pano 改善有限（Pinhole model），但换 representation（SH + radius）就能从 5.9 跳到 53.8。当你的 representation 有 inherent limitation 时，data 灌再多也突破不了。

**Intuition 2: Disentanglement 需要多个 mechanism 协同**

解耦 camera 和 scene geometry 不是只改 output space 就够的。UniK3D 用了至少 5 个 mechanism：
1. Spherical output (radial vs depth)
2. SH camera representation（无 parametric assumption）
3. Stop-gradient（防 feedback coupling）
4. Static encoding（防 learnable shortcut）
5. Asymmetric loss（防 distribution contraction）

少了任何一个，性能都会显著下降。

**Intuition 3: Inductive bias 在 small data regime 是 asset**

SH 的 continuity/differentiability/sparsity inductive bias 让 UniK3D 用相对适中的数据量（8M samples）就能泛化到 panoramic，而非 parametric 方法（每像素一个 ray）需要 far more data。

**Intuition 4: Camera conditioning 是 monocular 3D 的隐藏瓶颈**

很多方法 focus 在 depth network 的 capacity，但 UniK3D 的 Table 8 显示，conditioning design（Add vs Prompt）在 Pano 上能造成 42.5 vs 58.6 的巨大差距。如何让网络真正 "listen to" camera 信息是一个被低估的问题。

**Intuition 5: Asymmetric loss 是处理 long-tail 的优雅方法**

与其 rebalancing data（破坏 scene diversity），不如在 loss 层面做 quantile regression。$\alpha$ 是一个 1D 超参，search space 极小，且不改变 data distribution。这是 robust statistics 思想在 deep learning 里的干净应用。

---

## 11. 限制和可能的改进方向

paper 自己提到一些 limitations：

1. **Data diversity 仍是瓶颈**：虽然 SH representation 理论上支持任何 camera，但 training data 里 panoramic/fisheye 比例小，影响了 tail performance
2. **Camera augmentation 不够 realistic**：softmax splatting 生成的 distorted images 有一部分 unrealistic，因为采样的是 camera parameters 而非 rays
3. **Confidence prediction 在 OOD 下 unreliable**：这是 regression task 的通病

可能的改进方向（我的 speculation）：
- 用 diffusion model 来 generate realistic distorted images，而非 geometric warping
- 把 SH degree 提高（比如 5th degree，35 个系数），可能在 extreme distortion 下更准
- Adaptive degree：让网络自己决定用多少 degree，根据 input complexity
- Joint training with multi-view geometric consistency，提供额外的 self-supervision signal

---

## 12. 相关工作的 web links

- UniDepth (前作): https://arxiv.org/abs/2403.18913
- Depth Anything: https://arxiv.org/abs/2401.10891
- Depth Anything v2: https://arxiv.org/abs/2406.09414
- Metric3D: https://arxiv.org/abs/2307.10964
- Metric3D v2: https://arxiv.org/abs/2404.15506
- DepthPro: https://arxiv.org/abs/2410.02073
- MASt3R: https://arxiv.org/abs/2406.09756
- DUSt3R: https://arxiv.org/abs/2312.14132
- ZoeDepth: https://arxiv.org/abs/2302.12288
- DINOv2: https://arxiv.org/abs/2304.07193
- Spherical Harmonics (经典参考): https://en.wikipedia.org/wiki/Spherical_harmonics
- Quantile regression / Pinball loss: https://en.wikipedia.org/wiki/Quantile_regression
- LayerScale (CaiT): https://arxiv.org/abs/2103.17239

---

这篇 paper 我觉得最漂亮的地方在于它把一个看似 engineering 的问题（支持多种 camera）通过 **spherical representation + SH basis** 提升到了一个 principled 的几何框架。radial distance vs depth 的选择、SH 作为 camera-agnostic representation、asymmetric loss 对抗 contraction——每一个 design choice 都有清晰的几何/统计 intuition 支撑。最后能在一个 unified framework 下同时达到 S.FoV, L.FoV, Pano 的 SotA，说明这个框架抓住了问题的本质。

希望这个 walkthrough 帮你建立了对 UniK3D 的 intuition。
