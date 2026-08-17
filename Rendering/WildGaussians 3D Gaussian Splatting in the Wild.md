---
source_pdf: WildGaussians 3D Gaussian Splatting in the Wild.pdf
paper_sha256: dc016e38fa1d4357223789f139f0e915bb3a3ddcf9cf705ac91d890d467bb0c4
processed_at: '2026-08-13T04:34:36-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 WildGaussians

## 一句话版本

3DGS原来只能处理"干净"的实验室照片,WildGaussians让它能处理像Flickr上随便拍的那种"脏"照片——有路人遮挡、白天黑夜曝光差异大、还有天空背景变化的问题,而且渲染速度不掉。

## 为什么需要这篇论文

想象你是一个游客去Brandenburg Gate拍了一堆照片,传到网上。有人想用这些照片reconstruct一个3D模型。问题来了:

- **你白天去,他晚上去**:同一个建筑,照片看起来完全不一样,exposure差好几档
- **你拍照时有个路人挡着**:这位路人在下一张照片就走了,但3DGS不知道,会以为这个路人就是建筑的一部分
- **天空**:白天蓝天,晚上黑天,3DGS根本不会在天空区域初始化任何Gaussian

3DGS原版遇到这些情况就崩了。NeRF其实能处理(NeRF-W早就做了),因为NeRF有shared parameters,学错了能"慢慢修正"。但3DGS一旦prune掉了某个Gaussian,就永远没了。所以需要重新设计。

## 三个核心问题与解决方案

### 问题1:Appearance changes (外观变化)

**难点**:白天和晚上的同一面墙,pixel value完全不同。3DGS的每个Gaussian存了一个固定color,没法同时表示白天和晚上。

**NeRF-W的做法**:每张照片配一个appearance embedding向量,喂给MLP,让MLP输出对应这张照片的颜色。

**3DGS如果照搬**:每个Gaussian都要过一遍MLP才能得到颜色,渲染慢得要死。3DGS的核心卖点就是real-time,这个不能丢。

**WildGaussians的思路**:
- 每张照片一个embedding $\mathbf{e}_j$ (32维)——表示这张照片的"全局氛围" (白天/晚上/阴天)
- 每个Gaussian一个embedding $\mathbf{g}_i$ (24维)——表示这个Gaussian的"局部特性" (这盏灯晚上会亮,那面墙反射不一样)
- 一个小MLP接收这两个embedding + Gaussian的基础颜色 $\bar{c}_i$,输出一组affine transform参数 $\gamma$ (scale) 和 $\beta$ (bias)
- 最终颜色 $\tilde{c}_i = \gamma \cdot \hat{c}_i(\mathbf{r}) + \beta$

说白了,就是给每个Gaussian的颜色做一个"调色",这个调色由"照片的氛围"和"这个Gaussian本身的特性"共同决定。

**公式详解**:
$$(\beta, \gamma) = f(\mathbf{e}_j, \mathbf{g}_i, \bar{c}_i)$$
- $f$:2层hidden layer,每层128维,ReLU activation的小MLP
- 输入是三个向量concat在一起
- 输出3组 $(\beta_k, \gamma_k)$,对应RGB三个channel

$$\tilde{c}_i = \gamma \cdot \hat{c}_i(\mathbf{r}) + \beta$$
- $\hat{c}_i(\mathbf{r})$:原始的view-dependent color (通过SH计算)
- $\gamma$:scale (gain),控制亮度缩放
- $\beta$:bias (offset),控制颜色偏移
- 这是per-channel的affine transform

**关键技巧:baking**
训练完之后,如果只需要渲染某一个appearance (比如白天的),可以预先把所有Gaussian的 $(\beta, \gamma)$ 算出来,直接更新它们的SH coefficients,变成标准3DGS。这样inference速度和原版3DGS一样快。这是对比concurrent works (GS-W, SWAG) 的关键优势——后者每个frame都要跑额外的网络。

**为什么用affine不用更复杂的transform?**
Affine只能做per-channel的linear scale和shift,看起来很弱。但实践中:
- Exposure difference就是大致linear scaling
- White balance差异就是per-channel的不同scale
- 局部效果 (某盏灯亮起)由per-Gaussian embedding $\mathbf{g}_i$ 补充
- Affine参数少,容易优化,容易bake

**Fourier features初始化的妙处**
如果 $\mathbf{g}_i$ 随机初始化,空间上相邻的两个Gaussian可能学到完全不同的embedding,导致appearance field不连续,渲染时会有artifacts。

WildGaussians用Fourier features初始化 $\mathbf{g}_i$:
- 先把Gaussian的中心位置 $\mu_i$ 归一化到 $[0,1]$
- 然后计算 $\sin(\pi p_k 2^m)$ 和 $\cos(\pi p_k 2^m)$,其中 $k \in \{1,2,3\}$ 是坐标,$m \in \{1,2,3,4\}$ 是频率
- 维度 $3 \times 2 \times 4 = 24$,正好匹配embedding size

这样的话,空间上相近的Gaussian,它们的 $\mathbf{g}_i$ 自然接近,保证了appearance field的空间连续性。这其实是一种implicit smoothness prior。

### 问题2:Occlusions (遮挡)

**难点**:训练照片里有行人、车辆,这些是transient objects,不应该被reconstruct进3D scene。但3DGS看到这些像素的color和预测不符,会产生大gradient,触发densification,长出一堆错误的Gaussians。

**NeRF-W / NeRF On-the-go的做法**:每个pixel预测一个uncertainty $\sigma$,loss除以 $\sigma^2$,这样高uncertainty的pixel对loss贡献小。

经典uncertainty loss:
$$\mathcal{L}_u = \frac{\|\tilde{C} - C\|_2^2}{2\sigma^2} + \log\sigma + \text{const}$$
- $\tilde{C}$:predicted color
- $C$:ground truth color
- $\sigma$:predicted uncertainty
- 第一项:weighted MSE,$\sigma$ 越大weight越小
- 第二项:防止 $\sigma \to \infty$ 的正则

**问题**:这个loss在appearance changes下会fail。想象一张白天照片和一张夜晚照片,预测的pixel color和GT差异很大,不是因为遮挡,而是因为appearance没学好。MSE loss会错误地认为这些pixel都是"high uncertainty",从而ignore掉整个background,导致appearance永远学不好。这是个chicken-and-egg problem。

**WildGaussians的思路**:用DINO v2 features来定义"semantic similarity",而不是pixel-level color similarity。

DINO是self-supervised ViT,它的features对lighting、color、texture变化是invariant的。同一面墙,白天和晚上,DINO features会很接近;但如果是个遮挡的路人,DINO features会和background完全不同。

**DINO cosine similarity loss**:
$$\mathcal{L}_{dino}(\tilde{D}, D) = \min\left(1, 2 - \frac{2\tilde{D}\cdot D}{\|\tilde{D}\|_2 \|D\|_2}\right)$$
- $\tilde{D}$:rendered image的DINO features (per 14x14 patch)
- $D$:training image的DINO features (per 14x14 patch)
- 中间项是 $2 \cos(\tilde{D}, D)$
- 当cosine similarity = 1 (完全匹配):loss = 0
- 当cosine similarity < 0.5:loss saturate at 1

**Combined uncertainty loss**:
$$\mathcal{L}_{uncertainty} = \frac{\mathcal{L}_{dino}(\tilde{D}, D)}{2\sigma^2} + \lambda_{prior}\log\sigma$$
- $\lambda_{prior} = 0.5$
- 只更新uncertainty predictor,不backprop到rendering pipeline

**Uncertainty predictor架构**:极其简单,就是一个affine transform + softplus,作用在DINO features上。DINO features是pre-trained的,不更新。

**Binary mask trick (关键创新)**:
NeRF-W直接用 $1/(2\sigma^2)$ 作为weight去乘loss,但在3DGS里这样会出问题。因为3DGS的densification是基于gradient magnitude的threshold,continuous weighting会让gradient分布扭曲。

WildGaussians改成binary mask:
$$M = \mathbb{1}\left(\frac{1}{2\sigma^2} > 1\right)$$
- 当 $\sigma < 1/\sqrt{2} \approx 0.707$:mask = 1 (trust this pixel)
- 否则:mask = 0 (ignore this pixel)

Masked loss:
$$\mathcal{L}_{color-masked} = \lambda_{dssim} M \mathrm{DSSIM}(\hat{C}, C) + (1-\lambda_{dssim}) M \|\tilde{C} - C\|_1$$

**Intuition**:Binary mask保证gradient scaling最多是1,不会扭曲densification的gradient统计。同时"ignore"比"downweight"更aggressive,occlusion handling更干净。

### 问题3:Sky modeling

**难点**:SfM (Structure-from-Motion) 通常不会在天空区域生成任何3D points,因为天空没有feature points可以match。所以3DGS初始化时天空区域是空的。

**WildGaussians的方案**:
1. 计算scene radius $r_s$ (用input points的 $L^\infty$ norm的97%分位数)
2. 在距离scene center $10 r_s$ 处放一个sphere
3. 用Fibonacci sphere sampling在这个sphere上生成100,000个均匀分布的点
4. 把这些点投影到所有training cameras,保留至少被一个camera看到的点
5. 加入initial point cloud,opacity设为1.0 (其他点opacity=0.1)

Fibonacci sphere sampling用golden ratio生成spiral pattern,比random sampling均匀得多。这样天空就有了Gaussians,配合appearance modeling,可以学习day/night sky的差异。

## Decoupled loss的intuition

WildGaussians有一个很巧妙的设计:DSSIM和L1分别用于不同的image。

$$\mathcal{L}_{color} = \lambda_{dssim}\mathrm{DSSIM}(\hat{C}, C) + (1-\lambda_{dssim})\|\tilde{C} - C\|_1$$

- $\hat{C}$:**没有**appearance modeling的rendering (原始SH color)
- $\tilde{C}$:**有**appearance modeling的rendering (affine toned color)
- $C$:training image

**为什么这么分?**
- DSSIM关注结构、感知相似度,对appearance变化robust。用它监督没有toning的图,让网络学geometry,不被appearance干扰。
- L1关注pixel-wise颜色准确性,用它监督toning后的图,让网络学appearance correction。

这样geometry学习和appearance学习解耦,各司其职,训练更稳定。

## 实验数据解读

### NeRF On-the-go dataset (Table 1)

这个dataset主要是occlusion,几乎没有appearance changes。

| Method | GPU hrs | FPS | Avg PSNR |
|--------|---------|-----|----------|
| NeRF On-the-go | 43 | <1 | 21.71 |
| 3DGS | 0.35 | 116 | 19.30 |
| Mip-Splatting | 0.18 | 82 | 19.12 |
| GOF | 0.41 | 43 | 19.24 |
| GS-W | 0.55 | 71 | 19.56 |
| **WildGaussians** | **0.50** | **108** | **22.15** |

**关键观察**:
1. WildGaussians比NeRF On-the-go高0.44 dB PSNR,但training time短86倍
2. FPS 108接近原版3DGS的116,说明uncertainty modeling几乎不影响渲染速度
3. 在high occlusion场景,比3DGS高4 dB,说明uncertainty modeling效果显著

### Photo Tourism dataset (Table 2)

这个dataset既有appearance changes又有occlusions,是真正的"wild"。

| Method | Brandenburg PSNR | Sacre Coeur | Trevi Fountain |
|--------|------------------|-------------|-----------------|
| NeRF-W-re | 24.17 | 19.20 | 18.97 |
| Ha-NeRF | 24.04 | 20.02 | 20.18 |
| K-Planes | 25.49 | 20.61 | 22.67 |
| RefinedFields | 26.64 | 22.26 | 23.42 |
| 3DGS | 19.37 | 17.44 | 17.58 |
| GS-W | 23.51 | 19.39 | 20.06 |
| SWAG | 26.33 | 21.16 | 23.10 |
| **WildGaussians** | **27.77** | **22.56** | **23.63** |

**关键观察**:
1. 3DGS原版在Photo Tourism上严重退化 (Brandenburg 19.37 vs NeRF-W-re 24.17),因为appearance changes让3DGS grow出错误的Gaussians
2. WildGaussians在所有scene上都SOTA
3. 相比SWAG (concurrent work),rendering速度快7倍 (117 vs 15 FPS),因为可以bake appearance

### Ablation study (Table 3)

| Setting | Photo Tourism PSNR | On-the-go high PSNR |
|---------|---------------------|----------------------|
| Full method | 24.63 | 23.03 |
| w/o uncertainty | 24.32 | 20.27 |
| w/o appearance | 18.47 | 22.80 |

**关键insights**:
- **Appearance modeling在Photo Tourism上至关重要** (去掉掉6 dB)
- **Uncertainty modeling在高occlusion上至关重要** (去掉掉2.7 dB)
- **低occlusion下uncertainty可以省略**
- **Appearance modeling在低occlusion场景几乎无害**

### Extended ablation (Table 4) 的几个重要发现

- **w/o Gaussian embeddings** (只有per-image):Brandenburg 27.77 → 25.18,证明per-Gaussian embedding对local appearance (lamps, shadows) 必要
- **VastGaussian appearance modeling**:只在small appearance差异下work,large appearance变化会有artifacts
- **MSSIM uncertainty**:Trevi Fountain 23.63 → 21.20,验证DINO-based uncertainty的robustness
- **Explicit Mask R-CNN masks**:27.77 → 26.87,learned uncertainty比explicit segmentation更好,因为能处理soft boundaries

## Appearance embedding的可视化

论文Figure 8的t-SNE投影显示:
- Day images聚类在一起
- Night images聚类在一起
- Day和night明显分离

这说明appearance embedding学到了语义有意义的representation,不是random noise。

Figure 6的appearance interpolation显示:从白天到晚上的embedding插值,lamps逐渐出现,transition smooth。这说明embedding space是一个continuous manifold,不是离散的pockets。

Figure 7显示固定一个appearance embedding,移动camera,rendering保持multiview consistency——这是3DGS的explicit geometry advantage。

## 一些更深的思考

### 为什么3DGS比NeRF更难处理in-the-wild?

NeRF的MLP有shared parameters,如果某个region学错了,可以通过其他view的supervision来修正。3DGS的每个Gaussian是独立的,如果被prune了,就永远没了。而且3DGS的densification是基于gradient的,appearance changes会产生虚假的gradient,触发错误的densification。

所以WildGaussians的设计哲学是:在appearance changes发生之前,先用uncertainty mask屏蔽掉那些有问题的pixels,避免虚假gradient污染densification。这是一个"预防"策略,而非"修正"策略。

### Binary mask vs continuous weighting的trade-off

Binary mask更aggressive,会完全ignore某些pixels。Continuous weighting更soft,但可能"半信半疑"地学习,效果不如clean ignore。在3DGS的context下,binary mask还有一个额外好处:不破坏gradient statistics,保证densification stable。

### Per-Gaussian embedding的expressiveness

24维的 $\mathbf{g}_i$ 看起来很小,但配合MLP可以表达相当复杂的local appearance。每个Gaussian的embedding可以理解为"这个Gaussian在不同appearance下的行为模式"。Fourier features init保证了空间连续性,使得appearance field是smooth的。

### Baking的实用性

Baking是WildGaussians的killer feature。在AR/VR应用中,通常只需要render一个固定appearance (比如白天的),baking之后就是标准3DGS,可以deploy到mobile设备。这是NeRF-W和concurrent works做不到的。

### DINO features的limitation

DINO是patch-wise (14x14) features,分辨率有限。对于small occluders (比如远处的小人),可能落在同一个patch里,DINO features被average掉,uncertainty预测不准。这是一个fundamental limitation,paper没讨论。

### 与diffusion prior结合的未来方向

Paper的limitation部分提到:当某个region被occluder遮挡太多,uncertainty无法recover几何。一个future direction是用pre-trained diffusion model作为prior,在high uncertainty区域inpaint几何。这是当前3DGS研究的热门方向。

## 总结:WildGaussians的设计哲学

1. **保留3DGS的explicit representation优势**,只通过最小限度的附加capacity获取appearance modeling能力
2. **Bake-able**:训练时的appearance modeling可以"折叠"回标准3DGS,inference时无额外开销
3. **DINO-based uncertainty**:利用self-supervised features的appearance invariance,解决MSE/DSSIM在appearance changes下的failure
4. **Binary mask**:与3DGS的discrete densification nature匹配,不破坏gradient statistics
5. **Fourier features init**:保证appearance field的空间连续性
6. **Decoupled DSSIM/L1 loss**:geometry和appearance学习分离,训练stable

这套组合让3DGS第一次在Photo Tourism这种highly unconstrained dataset上达到SOTA,且保持real-time rendering。这是3DGS走向实际应用的重要一步。

## References

- WildGaussians project: https://wild-gaussians.github.io/
- 3DGS original: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- 3DGS code: https://github.com/graphdeco-inria/gaussian-splatting
- Mip-Splatting: https://niujinshuchong.github.io/mip-splatting/
- NeRF-W: https://nerf-w.github.io/
- NeRF On-the-go: https://weining-ren.github.io/nerf-on-the-go/
- DINOv2: https://dinov2.metaseishin.io/
- Photo Tourism dataset: http://phototour.cs.washington.edu/
- Splatfacto-W: https://github.com/nerfstudio-project/nerfstudio
- SWAG: https://arxiv.org/abs/2403.10404
- GS-W: https://arxiv.org/abs/2403.15704
- Scaffold-GS: https://city-super.github.io/scaffold-gs/
- AbsGS: https://github.com/MarcWang5566/AbsGS
- NerfBaselines: https://jkulhanek.com/nerfbaselines
- Fourier Features: https://github.com/tancik/fourier-feature-networks
- EWA Splatting: https://www.cs.umd.edu/~zwicker/publications/EWA-Splatting-TVCG2002.pdf
- RobustNeRF: https://robustnerf.github.io/
- SpotLessSplats: https://spotlesssplats.github.io/
- VastGaussian: https://arxiv.org/abs/2402.04364
- Adam optimizer: https://arxiv.org/abs/1412.6980

---

# WildGaussians: 3D Gaussian Splatting in the Wild 深度技术讲解

## 1. 论文核心问题与motivation

这篇paper要解决的核心问题:把3D Gaussian Splatting (3DGS) 从受控实验室环境扩展到 "in-the-wild" 数据。In-the-wild 特指crowd-sourced photo collections (例如Photo Tourism dataset中来自Flickr的照片),包含三类disturbances:

- **Appearance changes**:不同时间、季节、曝光、天气、昼夜差异
- **Occlusions / transient objects**:行人、车辆、临时物体
- **Sky modeling**:天空在appearance变化下差异巨大

3DGS本身struggle在这些场景的核心原因:explicit representation + 每个Gaussian独立存储color,导致缺乏shared parameters。NeRF的MLP共享参数,可以在appearance变化下"recover from early mistakes";3DGS的densification和pruning过程一旦误删几何,就irreversible。这是一个关键intuition:training dynamics不同导致失败模式不同。

## 2. 3DGS Preliminaries 公式详解

回顾3DGS的基础:

### 2.1 2D投影
3D Gaussian $\mathcal{G}_i$ 由 mean $\mu_i$、covariance $\Sigma_i$、opacity $\alpha_i$、view-dependent color (SH coefficients) 描述。

2D covariance $\Sigma'_i$ 在image space上的投影:
$$\Sigma'_i = (J W \Sigma_i W^T J^T)_{1:2,1:2}$$

- $W \in \mathbb{R}^{4 \times 4}$:viewing transformation (world-to-camera-to-clip)
- $J$:Jacobian of affine approximation of projective transformation (从3D EWA splatting [Zwicker 2001]来的)
- $(\cdot)_{1:2,1:2}$:取前两行两列,因为只关心2D image plane的covariance

### 2.2 Alpha compositing
对每个pixel,color $\hat{C}$ 由front-to-back排序的Gaussians累加得到:
$$\hat{C} = \sum_i \alpha_i \hat{c}_i(\mathbf{r})$$
其中 $\alpha_i = \exp\left(-\frac{1}{2}(x-\mu'_i)^T (\Sigma'_i)^{-1} (x-\mu'_i)\right)$ 是基于2D Gaussian在pixel $x$ 处的evaluation,$\hat{c}_i(\mathbf{r})$ 是view-dependent color via SH,$\mathbf{r}$ 是ray direction。

### 2.3 Training loss
$$\mathcal{L}_{3DGS} = \lambda_{dssim}\mathrm{DSSIM}(\hat{C}, C) + (1-\lambda_{dssim})\|\hat{C} - C\|_1$$
- $\lambda_{dssim}=0.2$:weighting
- DSSIM捕捉结构相似度,L1捕捉pixel-wise颜色

### 2.4 Adaptive density control
3DGS有两个key operations:
- **Pruning**:移除opacity $\alpha_i$ 小或3D size过大的Gaussians
- **Cloning/splitting**:对2D mean梯度 $|\nabla_{\mu'_i}|$ 大的Gaussians进行densification

这里有个微妙点:论文特别提到用 **absolute gradient accumulation** [AbsGS, Gaussian Opacity Fields] 代替actual gradients累积,这是因为 signed gradients可能在多视角下cancel,absolute保留magnitude用于densification判断。

## 3. Appearance Modeling 深度解析

### 3.1 设计思路

NeRF-W [Martin-Brualla 2021] 的做法:per-image appearance embedding $\mathbf{a}_j$ 输入到MLP,conditioning radiance field output。Urban Radiance Fields [Rematas 2022] 的做法:从image embedding单独预测affine transform。

WildGaussians的设计关键差异:**affine transformation模型 + per-Gaussian embedding**。理由是局部appearance变化(local appearance changes)如夜晚scene某盏灯亮起,无法用global affine transform解释,需要per-Gaussian capacity。

### 3.2 公式详解

**输入**:
- $\mathbf{e}_j \in \mathbb{R}^{32}$:per-image appearance embedding (32维)
- $\mathbf{g}_i \in \mathbb{R}^{24}$:per-Gaussian appearance embedding (24维)
- $\bar{c}_i$:0-th order SH coefficient (base color) for Gaussian $i$

**MLP**:
$$(\beta, \gamma) = f(\mathbf{e}_j, \mathbf{g}_i, \bar{c}_i)$$
- $f$:2 hidden layers of 128 units,ReLU activation
- 输出 $(\beta, \gamma) = \{(\beta_k, \gamma_k)\}_{k=1}^{3}$ per color channel

**Toned color**:
$$\tilde{c}_i = \gamma \cdot \hat{c}_i(\mathbf{r}) + \beta$$
- $\gamma \in \mathbb{R}^3$:scale (gain)
- $\beta \in \mathbb{R}^3$:bias (offset)
- $\hat{c}_i(\mathbf{r})$:原始view-dependent SH color

这是一个per-channel的affine color transform。

### 3.3 MLP output的prior scaling trick

为稳定训练,论文对MLP最后一层 raw output $(\hat\beta, \hat\gamma)$ 做一个 reparametrization:
- $\beta_k = 0.01 \hat\beta_k$
- $\gamma_k = 0.01 \hat\gamma_k + 1$

含义:初始化时让 $\gamma \approx 1$ (identity scale)、$\beta \approx 0$ (zero bias),且learning rate等效缩小100x。这避免了early training phase中affine transform不稳定干扰SH coefficient学习。这种prior trick非常类似于BN residual init 或者 NeRF的positional encoding warmup。

### 3.4 Per-Gaussian embedding初始化:Fourier features

**问题**:random初始化 $\mathbf{g}_i$ 会导致缺乏locality bias,相近的Gaussians学到不同的embedding,generalization差。

**方案**:用Fourier features [Tancik 2020, Vaswani 2017] 初始化:
1. 把input point cloud centered后归一化到 $[0, 1]$ (用 $L^\infty$ norm的0.97分位数)
2. 对normalized point $p$ 计算:
$$\mathbf{g}_i = \text{concat}\left[\sin(\pi p_k 2^m), \cos(\pi p_k 2^m)\right]_{k=1,2,3; m=1,\dots,4}$$
- $p_k$:第 $k$ 个coordinate (x,y,z)
- $m$:frequency band index,从1到4
- 总维数:$3 \times 2 \times 4 = 24$,正好匹配embedding大小

这是个非常优雅的设计:Fourier features自带locality prior,space上相近的Gaussians的 $\mathbf{g}_i$ 自然接近,使得appearance learning有spatial smoothness。

### 3.5 Decoupled DSSIM/L1 loss

这是一个重要的intuition point。论文将DSSIM和L1分别用于不同image:
$$\mathcal{L}_{color} = \lambda_{dssim}\mathrm{DSSIM}(\hat{C}, C) + (1-\lambda_{dssim})\|\tilde{C} - C\|_1$$

- $\hat{C}$:**rasterized image WITHOUT appearance modeling**(原始SH color)
- $\tilde{C}$:**rasterized image WITH appearance toning**(affine transformed color)

**Intuition**:DSSIM关注结构/感知相似度,对appearance变化robust。把DSSIM用于没有toning的图,让网络学geometry和structure,而不被appearance干扰;L1则专门学习appearance correction (toned image vs GT)。这样geometry学习和appearance学习解耦,各司其职。

### 3.6 Test-time appearance optimization

对于unseen image:
- $\mathbf{e}_j$ 初始化为zero
- 用Adam优化128步,learning rate 0.1
- 其他参数 (Gaussians, MLP, $\mathbf{g}_i$)固定

这是一个非常快速的test-time adaptation,只优化32维向量。

### 3.7 "Baking" appearance回标准3DGS

如果只需要在单一种appearance下render,test-time可以pre-compute所有Gaussians的 $(\beta_i, \gamma_i)$,直接更新SH coefficients,变成标准3DGS。这样rendering speed与原始3DGS完全一致。这是一个非常实用的design choice,也是对比GS-W、SWAG等concurrent works的优势——后者依赖reference image的CNN features,inference慢。

## 4. Uncertainty Modeling 深度解析

### 4.1 NeRF-W / NeRF On-the-go的uncertainty loss

经典formulation用aleatoric uncertainty (Kendall & Gal):
$$\mathcal{L}_u = \frac{\|\tilde{C} - C\|_2^2}{2\sigma^2} + \log\sigma + \frac{\log 2\pi}{2}$$

- $\tilde{C}$:predicted color
- $C$:GT color
- $\sigma$:predicted uncertainty (per-pixel variance)
- 第一项:weighted MSE,uncertainty大时降低loss weight
- 第二项:regularizer防止 $\sigma \to \infty$ (entropy maximization)
- 第三项:constant,无影响

NeRF On-the-go把MSE替换为DSSIM variant,理由是DSSIM更robust。

### 4.2 关键insight:MSE/DSSIM都fail under appearance changes

论文Figure 3的核心point:在appearance变化剧烈的场景 (白天vs夜晚),background本身pixel-wise就差异大,MSE/DSSIM会把background也误判为high uncertainty,从而忽略background的learning。这导致appearance modeling学不到正确参数,陷入chicken-and-egg问题。

### 4.3 DINO-based uncertainty loss

**DINO v2 features** [Oquab 2024] 是self-supervised ViT-S/14提取的patch features,每张图被切成14x14的patches,每个patch对应一个feature vector。DINO features本身是appearance-invariant的semantic representation,对lighting、color、texture变化robust。

**DINO cosine similarity loss**:
$$\mathcal{L}_{dino}(\tilde{D}, D) = \min\left(1, 2 - \frac{2\tilde{D}\cdot D}{\|\tilde{D}\|_2 \|D\|_2}\right)$$

- $\tilde{D}$:predicted image的DINO features (per patch)
- $D$:training image的DINO features (per patch)
- 中间项 $\frac{2\tilde{D}\cdot D}{\|\tilde{D}\|_2 \|D\|_2} = 2 \cos(\tilde{D}, D)$:scaled cosine similarity
- 当cosine similarity = 1:loss = $\min(1, 0) = 0$ (perfect)
- 当cosine similarity < 0.5:$2 - 2\cos < 1$,loss saturate at 1

**Combined uncertainty loss**:
$$\mathcal{L}_{uncertainty} = \frac{\mathcal{L}_{dino}(\tilde{D}, D)}{2\sigma^2} + \lambda_{prior}\log\sigma$$
- $\sigma$:per-patch uncertainty
- $\lambda_{prior}=0.5$:log prior weight
- 第一项:weighted by uncertainty
- 关键:dino loss不backprop到rendering pipeline,只更新uncertainty predictor (一个简单的affine transform + softplus)

### 4.4 Uncertainty predictor架构

非常简单:
1. 从training image提取DINO v2 ViT-S/14 features (image resize到max 350)
2. Trainable affine transform: $W \in \mathbb{R}^{d_{dino}\times 1}$,$b \in \mathbb{R}$
3. softplus activation确保非负
4. Bilinear upsample 14x14 patches 到原图分辨率
5. Clip 到 $[0.1, \infty)$ 确保最小weight

### 4.5 Binary mask trick (关键创新)

**问题**:如果像NeRF-W一样直接用uncertainty weighting $1/(2\sigma^2)$ 去weight per-pixel loss,会破坏3DGS的densification。

**原因**:3DGS的densification用的是 **absolute gradient的累加**。如果uncertainty weighting乘到loss上,有些pixel weight小、有些大,absolute gradient分布被扭曲,导致hyperparameter-sensitive且unstable。

**解决方案**:把uncertainty转换成binary mask:
$$M = \mathbb{1}\left(\frac{1}{2\sigma^2} > 1\right)$$

- 当 $\sigma < 1/\sqrt{2} \approx 0.707$ 时,mask=1 (trust)
- 否则mask=0 (ignore)

Masked loss:
$$\mathcal{L}_{color-masked} = \lambda_{dssim} M \mathrm{DSSIM}(\hat{C}, C) + (1-\lambda_{dssim}) M \|\tilde{C} - C\|_1$$

**Intuition**:用binary mask保证gradient scaling最多是1,densification算法的gradient统计不被扭曲。同时,直接ignore pixels比weighting更aggressive,occlusion handling更干净。

### 4.6 Opacity reset后的训练策略

3DGS的opacity定期reset (每3000步) 防止local minima,但reset后短时间内rendering会corrupted (alpha不准确)。如果在这时继续训练uncertainty,uncertainty会学到错误的"high uncertainty"状态。

**解决方案**:opacity reset后的500 iterations内禁用uncertainty training。这是一个细节但重要的engineering trick。

## 5. Sky Handling

**问题**:SfM通常不在天空生成points,导致天空区域无Gaussian。

**方案**:
1. 计算 scene radius $r_s$ = input points $L^\infty$ norm的97%分位数
2. 在距scene center $10 r_s$ 处放置sphere
3. 用Fibonacci sphere sampling生成100,000个均匀分布的点
4. 投影到所有training cameras,保留可见点
5. 加入initial point cloud,opacity=1.0 (其他points opacity=0.1)

Fibonacci sphere sampling用golden ratio生成spiral pattern,均匀度比random sampling好。这种sky modeling与appearance modeling联合,可以学习day/night sky差异。

## 6. 实验结果详细分析

### 6.1 NeRF On-the-go dataset (Table 1)

数据集特点:6个sequence,occlusion ratio从5% (low) 到30% (high),基本无illumination变化。

| Method | GPU hrs | FPS | Low (PSNR) | Medium (PSNR) | High (PSNR) | Avg (PSNR) |
|--------|---------|-----|------------|----------------|-------------|------------|
| NeRF On-the-go | 43 | <1 | 20.63 | 22.31 | 22.19 | 21.71 |
| 3DGS | 0.35 | 116 | 19.68 | 19.19 | 19.03 | 19.30 |
| Mip-Splatting | 0.18 | 82* | 20.15 | 19.12 | 18.10 | 19.12 |
| GOF | 0.41 | 43* | 20.54 | 19.39 | 17.81 | 19.24 |
| GS-W | 0.55 | 71* | 18.67 | 21.50 | 18.52 | 19.56 |
| **WildGaussians** | **0.50** | **108** | **20.62** | **22.80** | **23.03** | **22.15** |

**关键观察**:
1. WildGaussians在high occlusion场景比3DGS高4 dB PSNR
2. Training time (0.5 hrs) 是NeRF On-the-go (43 hrs)的1/86
3. Rendering speed 108 FPS接近原始3DGS的116 FPS,比GS-W的71快
4. 在low occlusion场景,3DGS本身已经不错 (得益于SfM点云prior),WildGaussians的uncertainty modeling提升不大
5. 高occlusion下,3DGS derivatives (Mip-Splatting, GOF) 退化严重 (17.81 / 18.10)

### 6.2 Photo Tourism dataset (Table 2)

数据集特点:Brandenburg Gate、Sacre Coeur、Trevi Fountain, occlusion ratio平均3.5%,appearance变化剧烈 (昼夜、季节)。

| Method | Brandenburg PSNR | Sacre Coeur | Trevi Fountain |
|--------|------------------|-------------|-----------------|
| NeRF | 18.90 | 15.60 | 16.14 |
| NeRF-W-re | 24.17 | 19.20 | 18.97 |
| Ha-NeRF | 24.04 | 20.02 | 20.18 |
| K-Planes | 25.49 | 20.61 | 22.67 |
| RefinedFields | 26.64 | 22.26 | 23.42 |
| 3DGS | 19.37 | 17.44 | 17.58 |
| GS-W | 23.51 | 19.39 | 20.06 |
| SWAG | 26.33 | 21.16 | 23.10 |
| **WildGaussians** | **27.77** | **22.56** | **23.63** |

**关键观察**:
1. WildGaussians在所有三个scenes上都是SOTA
2. 3DGS在Photo Tourism上严重退化 (19.37 vs NeRF-W-re的24.17),原因是appearance变化导致3DGS "grow unnecessary Gaussians to explain higher gradients",模型被扭曲
3. 相比SWAG (concurrent work),WildGaussians快7倍rendering (117 vs 15 FPS),因为可以bake appearance
4. Training time 7.2 hrs虽然比3DGS长,但比NeRF-W-re的164 hrs短22倍

### 6.3 Ablation study (Table 3)

| Setting | MipNeRF360 (0% occ) PSNR | Photo Tourism (3.5%) PSNR | On-the-go low (5%) | On-the-go high (26%) |
|---------|---------------------------|-----------------------------|--------------------|-----------------------|
| Full method | 23.73 | 24.63 | 20.62 | 23.03 |
| w/o uncert. | 23.71 | 24.32 | 20.53 | 20.27 |
| w/o app. | 23.31 | 18.47 | 20.80 | 22.80 |

**关键insights**:
1. **Appearance modeling在Photo Tourism上至关重要** (24.63 → 18.47, drop 6 dB)
2. **Uncertainty modeling在高occlusion上至关重要** (23.03 → 20.27, drop 2.7 dB)
3. **低occlusion下uncertainty可以省略** (20.62 → 20.53,基本无差)
4. **Appearance modeling在低occlusion场景几乎无害** (可以一直开着)
5. 在clean dataset (MipNeRF360 bicycle) 上,appearance modeling还带来0.4 dB提升——可能是因为learned affine能够correct一些ambient illumination差异

### 6.4 Extended ablation (Table 4) 关键发现

- **w/o Fourier features**: 轻微下降,Brandenburg Gate 27.77 → 27.33,验证locality prior重要性
- **VastGaussian appearance modeling**: 只在small appearance差异下work,larger appearance变化会留visible artifacts
- **w/o Gaussian embeddings (只有per-image)**: 大幅下降 (Brandenburg 27.77 → 25.18),证明per-Gaussian embedding对local appearance (lamps, shadows) 必要
- **MSSIM uncertainty**: 在Trevi Fountain上明显退化 (23.63 → 21.20),验证DINO-based uncertainty在appearance变化下的robustness
- **Explicit masks (Mask R-CNN)**: 性能稍差 (27.77 → 26.87),说明learned uncertainty可以更细腻地处理soft occlusion boundaries

### 6.5 Appearance embedding可视化 (Figure 8)

t-SNE投影显示appearance embeddings天然聚类:
- Day images聚一起
- Night images聚一起
- Day和night明显分离

这说明per-image embedding学到了语义有意义的appearance representation,而不是random noise。Figure 6的appearance interpolation显示lights逐渐出现,说明embedding space是smooth、continuous的manifold。

## 7. 与Concurrent Works对比

- **SWAG** [Dahmani 2024]:用external hash-grid implicit field存储appearance,rendering需要额外网络forward pass,慢
- **Scaffold-GS** [Lu 2024]:类似hash-grid approach
- **GS-W** [Zhang 2024]:用CNN features of reference image,需要conditioning,NeRF-W protocol下evaluation
- **Splatfacto-W** [Xu 2024]:用类似appearance MLP输出SH (而非affine transform),思路相近但affine design更简洁
- **VastGaussian** [Lin 2024]:用CNN post-process,无法处理large appearance changes
- **SpotLessSplats** [Sabour 2024]:只处理occlusions,不处理appearance
- **RobustGaussian** [Darmon 2024]:类似只处理occlusions

WildGaussians的优势:simple architecture、bake-able到标准3DGS、同时处理appearance + occlusions、DINO-based uncertainty robust to appearance changes。

## 8. 我的Intuition总结与延伸思考

### 8.1 为什么affine transform够用?

Affine color transform $\gamma \cdot c + \beta$ 只能做per-channel的linear scale和shift,理论上无法建模gamma correction、color cast、复杂tone mapping。但实践中够用是因为:
1. 大部分appearance变化是exposure difference (linear scaling)
2. White balance差异表现为per-channel scale
3. Local illumination (lamps)通过per-Gaussian $\mathbf{g}_i$ 建模,弥补global affine的不足
4. Affine参数少,容易优化,bake-able

### 8.2 为什么binary mask比continuous weighting好?

3DGS的densification是基于absolute gradient magnitude的threshold,本质上是discrete decision。Continuous weighting会让gradient分布扭曲,threshold变得hyperparameter-sensitive。Binary mask保持gradient统计,只需选择trust/ignore,与densification的discrete nature匹配。

### 8.3 为什么DINO features比MSE/DSSIM robust?

DINO是contrastive self-supervised学习,features天然invariant toappearance nuisance factors (lighting, color jitter)。MSE/DSSIM是pixel-space metrics,直接sensitive tocolor difference。在appearance变化下,DINO能区分"semantic content change (occluder)" vs "appearance-only change (lighting)"。

### 8.4 与diffusion prior结合的可能性

论文limitation提到:当区域被occluder遮挡过多,uncertainty无法recover几何。一个future direction是用预训练diffusion model (e.g., Stable Diffusion)作为prior,在uncertainty高的区域inpaint。这可以refine几何细节。

### 8.5 与Mip-Splatting的关系

WildGaussians直接基于Mip-Splatting [Yu 2024] implementation,Mip-Splatting解决了3DGS的aliasing问题。WildGaussians继承了这个优势。同时,AbsGS的absolute gradient fix也被纳入,这是当前3DGS改进的state-of-the-art baseline。

### 8.6 关于occlusion建模的哲学

NeRF On-the-go和WildGaussians都选择"ignore occluders"策略,而非"model occluders"。这是合理的,因为occluders (行人、车辆)本身没有多视角一致性,modeling它们反而破坏geometry。但有些应用 (e.g., 4D reconstruction) 需要model dynamic objects,这需要不同的formulation (e.g., EmerNeRF, D2NeRF)。

### 8.7 关于test-time optimization的实用性

Per-image embedding test-time optimization需要128步Adam,lr=0.1。这非常快 (大概几秒)。但需要GT image作为supervision。对于纯novel view synthesis (无GT)的场景,需要extrapolate embedding from nearest training images,这可能不够robust。这是该方法在real-time AR/VR应用的潜在限制。

### 8.8 Fourier features初始化的深层意义

Random init的 $\mathbf{g}_i$ 在高维空间无locality,相近的Gaussians可能学到完全不同的embedding,导致appearance field不smooth。Fourier features init把3D position encode到频率空间,自然保证nearby Gaussians有similar embedding。这其实是一种implicit smoothness prior。如果不这么做,appearance interpolation会出artifacts (Figure 6的smooth transition无法实现)。

### 8.9 与NeRF appearance modeling的本质差异

NeRF-W的appearance embedding直接concat到MLP input,影响整个网络的output。3DGS的explicit representation如果模仿这种做法,需要per-Gaussian forward pass一个MLP,推理慢。WildGaussians的affine transform只在color层面change,不修改geometry,且可以bake。这是一个非常巧妙的工程trade-off:既享受了appearance modeling的能力,又保留了3DGS的explicit advantage。

### 8.10 关于per-Gaussian embedding维度的选择

论文选24维,与Fourier features init的 $3 \times 2 \times 4 = 24$ 一致。如果用更高维,expressiveness提升但参数量增加 (3DGS典型有1M-10M Gaussians,每增一维embedding就增几MB显存)。24维是一个合理的trade-off。

## 9. 参考资源

- 论文project page: https://wild-gaussians.github.io/
- 3DGS原始论文: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- 3DGS code: https://github.com/graphdeco-inria/gaussian-splatting
- Mip-Splatting: https://niujinshuchong.github.io/mip-splatting/
- NeRF-W: https://nerf-w.github.io/
- NeRF On-the-go: https://weining-ren.github.io/nerf-on-the-go/
- DINOv2: https://dinov2.metaseishin.io/ , https://github.com/facebookresearch/dinov2
- Photo Tourism dataset: http://phototour.cs.washington.edu/
- Splatfacto-W (nerfstudio): https://github.com/nerfstudio-project/nerfstudio
- SWAG: https://arxiv.org/abs/2403.10404
- GS-W: https://arxiv.org/abs/2403.15704
- Scaffold-GS: https://city-super.github.io/scaffold-gs/
- AbsGS: https://github.com/MarcWang5566/AbsGS
- NerfBaselines: https://jkulhanek.com/nerfbaselines
- RobustNeRF: https://robustnerf.github.io/
- SpotLessSplats: https://spotlesssplats.github.io/
- EWA Splatting (Zwicker 2001): https://www.cs.umd.edu/~zwicker/publications/EWA-Splatting-TVCG2002.pdf
- Fourier Features (Tancik 2020): https://github.com/tancik/fourier-feature-networks
- Adam optimizer: https://arxiv.org/abs/1412.6980

## 10. 总结

WildGaussians是3DGS家族中第一个在Photo Tourism这种highly unconstrained dataset上达到SOTA且保持real-time rendering的方法。其核心design choices:
1. **Per-image + per-Gaussian appearance embedding** + small MLP → affine color transform (bake-able)
2. **DINO-based uncertainty loss** → robust to appearance changes
3. **Binary mask** → 不破坏3DGS的densification gradient statistics
4. **Fourier features init** → spatially smooth appearance field
5. **Decoupled DSSIM/L1 loss** → geometry与appearance学习分离
6. **Sky sphere initialization** → handling天空appearance变化
7. **Opacity reset后的uncertainty training pause** → engineering stability

这套组合让3DGS能够handle NeRF-W曾经主导的unconstrained photo collections场景,且速度上比NeRF-W快几个数量级。其设计哲学:"保留3DGS的explicit representation优势,通过最小限度的附加capacity (small MLP + low-dim embeddings) 获取appearance modeling能力,且保证可以'bake'回原始3DGS"。这是一个工程和理论的优雅平衡。
