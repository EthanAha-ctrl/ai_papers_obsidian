---
source_pdf: Diffusion Models in 3D Vision A Survey.pdf
paper_sha256: 1b8a212245043262d5788dc1e590f491bc4df640105fd17b72e828c5057ed4bf
processed_at: '2026-08-03T21:46:27-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Diffusion Models in 3D Vision Survey

好的，让我换个方式，就像咱俩在咖啡厅聊天那样，把这篇paper的精髓用直觉性的语言给你过一遍。

## 先说这paper到底在干嘛

过去几年，diffusion models在2D图像生成上大杀器，Stable Diffusion、DALL-E这些大家都在玩。然后有人就开始琢磨：**能不能用这套把戏来做3D的东西？** 比如给我一个"椅子"的描述，我直接给你生出一个3D椅子模型；给我一张正面照，我给你补全整个3D物体。

这篇survey就是把这个方向所有的工作都梳理了一遍，从底层数学原理，到各种3D任务的具体做法，到数据集、评估指标，最后聊了聊还有哪些坑没填。

参考: [Stable Diffusion](https://stability.ai/news/stable-diffusion-public-release) | [DALL-E 3](https://openai.com/dall-e-3)

---

## Diffusion Model到底是个啥玩意

### 核心直觉

想象你有一张干净的照片，然后你一点点往上面加noise，加个1000步，最后变成完全的random noise——这叫**forward process**。啥都看不出来了，全是雪花点。

然后你训练一个neural network，让它学会**反过来走**：给它一张纯noise，它一步步把noise去掉，最后还原出一张有意义的图像——这叫**reverse process**。

训练的时候，你给它看"加了noise的图片"和"原始图片"，让它学会预测"加了什么noise"。推理的时候，你给它纯random noise，它就能给你生成全新的、合理的图片。

这个就是**score-based generative modeling**的核心思想。

### 数学原理拆解

#### Forward Process

公式长这样：

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

人话翻译：
- $x_{t-1}$：上一步的图像
- $x_t$：当前步的图像
- $\beta_t$：这一步加多少noise，通常很小，比如0.001到0.02
- $\sqrt{1-\beta_t}$：把原图缩小一点点
- $\beta_t I$：加上一点点Gaussian noise

所以每一步就是：**把原图稍微缩小一点 + 加点noise**。重复1000次，就变成纯noise了。

**关键trick**：你不需要一步步算，有个closed-form公式可以直接从原图 $x_0$ 跳到任意步骤 $x_t$：

$$x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon$$

人话：
- $\alpha_t = \prod_{s=1}^t (1-\beta_s)$，是个累积乘积
- $\sqrt{\alpha_t}$：原始signal保留多少
- $\sqrt{1-\alpha_t}$：noise占多少
- $\epsilon$：从 $\mathcal{N}(0, I)$ 采样的random noise

当 $t \to T$，$\alpha_t \to 0$，signal全没了，就剩noise了。

#### Reverse Process

反向过程就是训练一个network $\epsilon_\theta(x_t, t)$ 来预测noise：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{1-\beta_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\alpha_t}} \epsilon_\theta(x_t, t)\right)$$

人话：
- 给你一个noisy image $x_t$ 和当前步骤 $t$
- network预测出noise $\epsilon_\theta$
- 然后从 $x_t$ 里把noise减掉一点点
- 得到稍微干净一点的 $x_{t-1}$
- 重复这个过程，最后得到 $x_0$

#### Training Loss

训练目标简单得惊人：

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

人话：**就是MSE loss**。network预测的noise和真实加的noise之间的平方差。就这么简单。

### 三种Formulation

论文讲了三种formulation，但其实都是一回事的不同视角：

**1. DDPM (Ho et al., 2020)**：discrete time steps，最经典
- [Original DDPM paper](https://arxiv.org/abs/2006.11239)

**2. Score-Based Generative Models (Song & Ermon, 2020)**：explicitly估计score function $\nabla_x \log p(x)$
- [Score-based paper](https://arxiv.org/abs/1907.05600)

**3. Score SDE (Song et al., 2021)**：continuous time，用SDE描述
- [Score SDE paper](https://arxiv.org/abs/2011.13456)

三种数学上等价，实践中DDPM用得最多。

---

## 3D Vision有啥不一样

### 3D数据长啥样

3D数据跟2D图像完全不一样，有几种主流表示方式：

#### Explicit Representations（显式表示）

**Point Clouds**：一堆3D点，每个点有(x,y,z)坐标
- LiDAR直接输出的就是这个
- 缺点：没有拓扑信息，不知道哪些点连在一起

**Voxel Grids**：把3D空间切成小方块
- 像Minecraft那种
- 优点：规则结构，可以用3D CNN
- 缺点：分辨率高了，memory爆炸（cubic增长）

**Meshes**：vertices + edges + faces
- 计算机图形学最爱
- 优点：rendering效率高
- 缺点：拓扑不规则，处理起来麻烦

#### Implicit Representations（隐式表示）

**SDF (Signed Distance Field)**：每个点返回到surface的距离
- 内部是负，surface是0，外部是正
- 连续光滑，适合做shape blending

**NeRF (Neural Radiance Field)**：用neural network表示场景
- 输入一个3D点和viewing direction，输出color和density
- [NeRF paper](https://arxiv.org/abs/2003.08934)

**3D Gaussian Splatting**：每个点用一个Gaussian表示
- 2023年的大热点
- 可以real-time rendering
- [3DGS paper](https://arxiv.org/abs/2308.14737)

### 3D Vision的三大痛点

1. **Occlusion**：物体被挡住了，sensor看不见
2. **Varying Point Density**：LiDAR打在物体上，近的地方点密，远的地方点稀
3. **Noise and Outliers**：sensor不精确，有反射干扰

这恰恰是diffusion models的强项——它们能处理uncertainty，生成多个plausible solution。

---

## 具体任务深度拆解

### 1. Unconditional 3D Generation

**目标**：直接从random noise生成3D shape

**演进路径**：

**早期做法（RenderDiffusion, Anciukevičius et al., 2023）**：
- 用2D diffusion model
- 每个denoising step都render一下中间的3D表示
- 通过inductive bias强制3D consistency
- [RenderDiffusion](https://arxiv.org/abs/2306.16728)

**Point-Voxel Diffusion (Zhou et al., 2021)**：
- 用point-voxel混合表示
- 直接在3D空间做diffusion
- [PVD paper](https://arxiv.org/abs/2104.03670)

**LION (Vahdat et al., 2022)** - 最有影响力的工作之一：
- Hierarchical VAE + 双diffusion models
- 一个diffusion生成global shape vector
- 另一个diffusion生成point cloud结构
- 在latent space做diffusion，效率高很多
- [LION paper](https://arxiv.org/abs/2210.06678)

**TIGER (Ren et al., 2024)** - 很有意思的insight：
- 发现attention和convolution在不同阶段作用不同
- Early stages：global attention定义整体shape
- Later stages：local convolution细化surface details
- 用可学习的mask来自适应融合
- [TIGER paper](https://arxiv.org/abs/2403.06169)

**Implicit representation的玩法**：

**Diffusion-SDF (Shim et al., 2023)**：
1. 先生成低分辨率SDF
2. 再用diffusion做super-resolution
3. Marching Cubes提取mesh
- [Diffusion-SDF](https://arxiv.org/abs/2303.11908)

### 2. Image-to-3D Generation

**目标**：给一张2D图，生成3D model

这是最practical的任务，也是最有挑战的，因为2D到3D是ill-posed problem。

#### 第一类：用2D diffusion作为prior

**DreamFusion (Poole et al., 2023)** - 这个工作是里程碑：

核心思想叫**Score Distillation Sampling (SDS)**：

1. 你有个NeRF表示3D scene
2. 从不同viewpoint render出2D image
3. 把2D image扔给pre-trained 2D diffusion model
4. 用diffusion model的score function来guide NeRF的优化

数学上：
$$\nabla_\theta \mathcal{L}_{SDS} = \mathbb{E}_{t, \epsilon}\left[w(t)(\epsilon_\theta(I_t, t) - \epsilon) \frac{\partial I}{\partial \theta}\right]$$

人话：
- $I_t$：rendered image加了noise
- $\epsilon_\theta(I_t, t)$：diffusion model预测的noise
- $\epsilon_\theta(I_t, t) - \epsilon$：prediction residual，告诉你image哪里"不对"
- $\frac{\partial I}{\partial \theta}$：rendering的Jacobian，告诉你怎么改3D参数能让image更好
- $w(t)$：weighting function

**神奇之处**：你完全不需要3D training data！只用2D diffusion model就能生成3D content。

- [DreamFusion paper](https://arxiv.org/abs/2209.14988)

#### 第二类：Multi-view generation

**One-2-3-45 (Liu et al., 2024)**：
1. 用2D diffusion生成多视角图像
2. 用cost-volume reconstruction重建3D mesh
3. 45秒生成一个3D model
- [One-2-3-45](https://arxiv.org/abs/2306.00614)

**Wonder3D (Long et al., 2024)**：
- Cross-domain attention保证multi-view consistency
- Geometry-aware normal fusion
- [Wonder3D](https://arxiv.org/abs/2310.15028)

**InstantMesh (Xu et al., 2024)**：
- Multi-view diffusion + Large Reconstruction Model
- 直接predict mesh，质量很高
- [InstantMesh](https://arxiv.org/abs/2404.07191)

#### 第三类：3D Gaussian Splatting方法

**LGM (Tang et al., 2024)**：
- Multi-view Gaussian features
- Asymmetric U-Net
- 快速生成3D model
- [LGM](https://arxiv.org/abs/2402.05054)

### 3. Text-to-3D Generation

**目标**：给文字描述，生成3D model

#### Point-E (OpenAI, Nichol et al., 2022)

OpenAI的工作，dual-stage：
1. Text → Image (用text-to-image diffusion)
2. Image → Point cloud (conditional diffusion)
- [Point-E](https://arxiv.org/abs/2212.08751)

#### Fantasia3D (Chen et al., 2023)

分离geometry和appearance：
- Text features分别guide shape和surface properties
- 这样可以分别控制形状和纹理
- [Fantasia3D](https://arxiv.org/abs/2303.13873)

#### ProlificDreamer (Wang et al., 2024)

改进SDS，提出**Variational Score Distillation (VSD)**：
- 把3D parameters当作probabilistic distributions
- 而不是point estimates
- 大幅提升quality和diversity
- [ProlificDreamer](https://arxiv.org/abs/2305.13913)

#### Janus Problem

这个是text-to-3D的经典问题：模型经常生成multi-face的怪物，因为2D diffusion model不知道3D consistency。

**解决方案**：
- 4D-fy (Bahmani et al., 2024)：alternating static initialization + variational SDS
- [4D-fy](https://arxiv.org/abs/2311.18484)

### 4. Texture Generation

**目标**：给3D mesh生成纹理

#### TexFusion (Cao et al., 2023)

- Text-guided
- Latent diffusion model
- Iteratively denoise 2D rendered views
- Aggregate到unified texture map
- [TexFusion](https://arxiv.org/abs/2306.09964)

#### Multi-view consistency是核心难点

**TexPainter (Zhang et al., 2024)**：
- 用pre-trained DDIM
- 每个denoising step解码到color space
- Latent code optimization对齐multi-view
- [TexPainter](https://arxiv.org/abs/2406.04224)

### 5. Human Avatar Generation

#### 头部生成

**RODIN (Wang et al., 2023)**：
- NeRF based
- 3D feature maps投影到2D planes
- 3D-aware convolutions
- [RODIN](https://arxiv.org/abs/2212.03604)

#### 全身生成

**PrimDiffusion (Chen et al., 2023)**：
- Volumetric primitives
- 每个primitive编码radiance + motion
- 大幅简化computation
- [PrimDiffusion](https://arxiv.org/abs/2312.07672)

#### 动态motion

**PhysDiff (Yuan et al., 2023)**：
- 把physics simulator嵌入diffusion
- Denoised motions投影到physically plausible trajectories
- [PhysDiff](https://arxiv.org/abs/2212.02991)

### 6. Scene Generation

#### Image-guided

**LucidDreamer (Chung et al., 2023)**：
- "Dreaming"：geometric image generation + 3D projection
- "Alignment"：point cloud integration
- [LucidDreamer](https://arxiv.org/abs/2311.13384)

#### Text-guided

**SceneScape (Fridman et al., 2024)**：
- Text-to-image + depth prediction
- 3D-consistent video generation
- [SceneScape](https://arxiv.org/abs/2312.08285)

#### Scene graph-based

**GraphDreamer (Gao et al., 2024)**：
- 用scene graph解决attribute conflicts
- [GraphDreamer](https://arxiv.org/abs/2403.01923)

### 7. Novel View Synthesis

**Zero123 (Liu et al., 2023)** - 很有影响力的工作：
- Single image → multi-view
- 用synthetic data训练
- [Zero123](https://arxiv.org/abs/2303.11328)

**SyncDreamer (Liu et al., 2024)**：
- 3D-aware feature attention
- Joint probability modeling
- 保证multi-view consistency
- [SyncDreamer](https://arxiv.org/abs/2310.15900)

### 8. Depth Estimation

**Marigold (Ke et al., 2024)** - SOTA方法：
- 基于Stable Diffusion
- 只fine-tune U-Net的denoising部分
- 保留pre-trained visual priors
- [Marigold](https://arxiv.org/abs/2312.02145)

**EcoDepth (Patni et al., 2024)**：
- 用pre-trained Vision Transformers
- 提取global contextual features
- [EcoDepth](https://arxiv.org/abs/2404.02286)

---

## 3D vs 2D Diffusion的关键差异

| 维度 | 2D | 3D |
|------|----|----|
| **数据量** | 海量（互联网照片） | 稀缺（需要3D扫描） |
| **数据结构** | 规则pixel grid | point cloud/voxel/mesh，各不相同 |
| **计算量** | 高但可接受 | 爆炸式增长（volumetric rendering） |
| **Topology** | 固定grid topology | 可变拓扑，需要order-invariant处理 |
| **Memory** | 图像级别 | 3D volume，cubic增长 |

---

## 核心数据集

### Object Datasets

- **ShapeNet**：51K synthetic CAD models，老牌baseline
  - [ShapeNet](https://shapenet.org/)
  
- **Objaverse**：800K models，目前最大
  - [Objaverse](https://objaverse.allenai.org/)

- **CO3D**：19K real scanned objects
  - [CO3D](https://github.com/facebookresearch/co3d)

### Human Datasets

- **Renderpeople**：40K synthetic human models
  - [Renderpeople](https://renderpeople.com/)

- **AMASS**：11K motion capture数据
  - [AMASS](https://amass.is.tue.mpg.de/)

### Scene Datasets

- **ScanNet**：1.5K indoor scenes
  - [ScanNet](http://www.scan-net.org/)

- **KITTI**：autonomous driving经典数据
  - [KITTI](http://www.cvlibs.net/datasets/kitti/)

---

## 评估指标

### Distance Metrics

- **CD (Chamfer Distance)**：两组点集之间的平均最近邻距离
  $$CD(S_1, S_2) = \frac{1}{|S_1|}\sum_{x \in S_1}\min_{y \in S_2}\|x-y\|^2 + \frac{1}{|S_2|}\sum_{y \in S_2}\min_{x \in S_1}\|x-y\|^2$$
  
- **EMD (Earth Mover's Distance)**：把一个分布transform成另一个的minimum cost

### Distribution Metrics

- **FID (Fréchet Inception Distance)**：在feature space比较distribution
- **KID (Kernel Inception Distance)**：基于kernel method

### Quality Metrics

- **PSNR**：peak signal-to-noise ratio
- **SSIM**：structural similarity

---

## 当前的主要挑战

### 1. Computational Efficiency

**问题**：3D diffusion太慢了
- 2D diffusion已经很慢（生成一张图要几秒到几十秒）
- 3D的维度更高，rendering更复杂
- 训练time和inference time都成问题

**可能方向**：
- 减少diffusion steps（[Consistency Models](https://arxiv.org/abs/2303.01469)）
- 更高效的architecture
- Latent space diffusion

### 2. 数据稀缺

**问题**：
- 2D有互联网级的海量数据
- 3D数据需要专业设备采集，成本高
- 导致3D model泛化能力差

**可能方向**：
- 用2D pre-trained model迁移到3D
- Synthetic data generation
- Self-supervised pretraining

### 3. Multimodal Fusion

**问题**：如何有效融合image + text + 3D geometry？

**可能方向**：
- Cross-attention mechanisms
- Unified embeddings
- Joint training frameworks

### 4. Fine-grained Control

**问题**：用户想精确控制生成的shape、texture、pose

**可能方向**：
- Interpretable latent spaces
- Conditional generation
- Attribute disentanglement

### 5. Dynamic Scenes

**问题**：现有方法主要处理static scenes，dynamic的很少

**可能方向**：
- Spatiotemporal diffusion
- 4D representation (3D + time)
- Physics-aware generation

### 6. Physical Plausibility

**问题**：生成的3D content可能物理上不合理

**可能方向**：
- 加入physics constraints
- Collision detection
- Material properties modeling

---

## 我的核心Insights

### Insight 1: 为什么Diffusion适合3D

3D data的三大痛点——occlusion、sparse sampling、noise——恰好是diffusion models的强项。Diffusion是probabilistic的，能generate多个plausible solution，而不是只给一个deterministic answer。这在处理ill-posed 3D reconstruction问题时非常关键。

### Insight 2: 2D-3D Bridge的核心作用

几乎所有的3D diffusion方法都在做一件事：**用2D diffusion作为bridge**。因为2D diffusion已经被海量数据训练得很好了，我们要想办法把它的knowledge迁移到3D。

SDS (Score Distillation Sampling)是最优雅的bridge，它通过differentiable rendering让2D diffusion的score function直接guide 3D optimization。

### Insight 3: Representation选择决定architecture

你选什么3D representation，决定了你整个pipeline的设计：
- Point cloud → 需要order-invariant network
- Voxel grid → 可以用3D CNN但memory爆炸
- Mesh → 处理topology麻烦
- SDF → 连续但decoding难
- Gaussian Splatting → 新兴方向，潜力大

### Insight 4: Two-stage是主流

大多数practical方法都是two-stage：
1. **Coarse stage**：快速生成rough 3D structure
2. **Fine stage**：refine details

比如Make-It-3D先做NeRF coarse geometry，再做textured point cloud refinement。这种coarse-to-fine策略在3D中特别重要，因为3D的search space太大，直接一步到位很难。

### Insight 5: 3D Gaussian Splatting是game changer

2023年3DGS出来后，很多方法都开始用它替代NeRF，because：
- Real-time rendering
- Explicit representation更好control
- Training更快
- Memory efficiency更高

### Insight 6: Janus Problem还没完全解决

Text-to-3D最常见的bug：生成多脸怪物。因为2D diffusion model对每个view独立生成，不知道整个3D object的consistency。虽然4D-fy等方法用alternating optimization缓解了，但根本解决还需要新的architecture design。

### Insight 7: Physics integration是未来方向

现在的3D diffusion模型生成的物体可能物理上不合理——比如椅子悬空、桌子腿穿透。未来需要把physics simulation嵌入diffusion process，像PhysDiff做的那样。

### Insight 8: Pretraining是关键

2D vision的成功很大程度上靠pretraining（ImageNet, CLIP, LAION等）。3D vision需要类似的large-scale pretraining。Objaverse (800K models)是个好的开始，但还远远不够。

---

## 未来方向的预测

1. **3D Foundation Models**：类似CLIP/SAM的3D version，能处理所有3D任务的unified model

2. **Real-time 3D Generation**：现在生成一个3D model要几分钟，未来需要几秒钟

3. **Video-to-4D**：从video生成动态3D场景

4. **Interactive Editing**：用户可以用自然语言或sketch实时编辑3D content

5. **Physics-aware Generation**：生成的物体满足物理约束，可以直接用于simulation

6. **Cross-modal Transfer**：更好地利用2D knowledge，减少对3D data的依赖

---

## 推荐的Reading List

### 必读论文

1. [DDPM](https://arxiv.org/abs/2006.11239) - Diffusion model基础
2. [Score SDE](https://arxiv.org/abs/2011.13456) - 连续时间formulation
3. [DreamFusion](https://arxiv.org/abs/2209.14988) - SDS开创性工作
4. [Point-E](https://arxiv.org/abs/2212.08751) - OpenAI的text-to-3D
5. [LION](https://arxiv.org/abs/2210.06678) - Latent diffusion for 3D
6. [3D Gaussian Splatting](https://arxiv.org/abs/2308.14737) - 新representation
7. [InstantMesh](https://arxiv.org/abs/2404.07191) - SOTA image-to-3D
8. [ProlificDreamer](https://arxiv.org/abs/2305.13913) - VSD改进
9. [Consistency Models](https://arxiv.org/abs/2303.01469) - 加速diffusion
10. [Zero123](https://arxiv.org/abs/2303.11328) - Novel view synthesis

### Datasets

- [ShapeNet](https://shapenet.org/)
- [Objaverse](https://objaverse.allenai.org/)
- [ScanNet](http://www.scan-net.org/)
- [KITTI](http://www.cvlibs.net/datasets/kitti/)

### Code Repositories

- [Awesome 3D Diffusion](https://github.com/topics/3d-diffusion)
- [DreamFusion PyTorch Implementation](https://github.com/ashawkey/stable-dreamfusion)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

---

总结一句：**3D diffusion是个fast-moving field，核心挑战在于如何用2D的knowledge来guide 3D generation，同时处理好3D data的scarcity和complexity。未来的突破点很可能在3D foundation models + physics-aware generation。**

---

# Diffusion Models in 3D Vision: A Survey - 深度技术解析

## 论文整体概览

这篇survey由Zhen Wang等人撰写，系统性地回顾了diffusion models在3D vision领域的应用。论文的核心贡献在于将原本为2D generative tasks设计的diffusion models扩展到3D data的复杂场景中，涵盖3D object generation、shape completion、point cloud reconstruction、scene construction等多个子任务。

论文的核心论点是：diffusion models的probabilistic nature非常适合处理3D data中inherent的uncertainty和ambiguity，because真实世界的3D data经常包含occlusions、missing regions、sensor noise等imperfections，而diffusion models能够iteratively refine outputs并且generate multiple plausible solutions。

参考链接：[arXiv survey on Diffusion Models](https://arxiv.org/abs/2209.02646) | [Ho et al. DDPM original paper](https://arxiv.org/abs/2006.11239) | [Song et al. Score SDE](https://arxiv.org/abs/2011.13456)

---

## 2. Diffusion Model Basics - 数学原理详解

### 2.1 Forward Diffusion Process

Forward process的核心思想是通过Markov chain逐步将clean data $x_0$ perturb成pure Gaussian noise。给定数据分布 $q(x_0)$，forward process定义为：

$$q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})$$

这里：
- $x_0$：原始clean data sample
- $x_t$：第t步的noisy sample
- $T$：total diffusion steps
- $q(x_t|x_{t-1})$：transition kernel

每个transition step的具体形式：

$$q(x_t|x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I\right)$$

变量解释：
- $\beta_t \in (0,1)$：pre-defined noise schedule，控制第t步的noise intensity
- $I$：identity matrix
- $\sqrt{1-\beta_t}$：scaling factor，保持variance stability
- $\beta_t$：variance of added Gaussian noise

**关键insight**：通过reparameterization trick和Gaussian distribution的product rule，我们可以得到closed-form的cumulative transition：

$$q(x_t|x_0) = \mathcal{N}\left(x_t; \sqrt{\alpha_t} x_0, (1-\alpha_t) I\right)$$

其中 $\alpha_t := \prod_{s=1}^{t}(1-\beta_s)$，这个累积乘积确保 $\lim_{t \to T} \alpha_t \approx 0$，使得 $q(x_T) \to \mathcal{N}(0, I)$。

**Intuition building**：这个closed-form非常重要，because它允许我们直接从 $x_0$ 采样任意步骤 $t$ 的 $x_t$，而不需要iteratively执行所有中间步骤。具体采样公式：

$$x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 2.2 Reverse Diffusion Process

Reverse process的目标是从pure noise $x_T \sim \mathcal{N}(0, I)$ 开始，iteratively denoise并reconstruct clean data。learned transition distribution：

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)$$

每个reverse step：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I\right)$$

- $\mu_\theta(x_t, t)$：neural network预测的mean
- $\sigma_t^2$：variance，通常fixed或为 $\beta_t$ 的deterministic function

**Key reparameterization**（Ho et al. 2020）：直接predict noise $\epsilon$ 而不是mean：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{1-\beta_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\alpha_t}} \epsilon_\theta(x_t, t)\right)$$

这个reparameterization的好处是将data reconstruction task与noise estimation task解耦，training stability更好。

### 2.3 Score Matching与Probability Density

Score function的定义：

$$\nabla_{x_t} \log p_\theta(x_t) = \mathbb{E}_{q(x_t|x_0)}\left[\frac{\sqrt{\alpha_t} x_0 - x_t}{1-\alpha_t}\right] = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\alpha_t}}$$

这里 $\nabla_{x_t}$ 表示对 $x_t$ 求gradient，$\log p_\theta(x_t)$ 是log probability density。

**Score Matching Loss**：

$$\mathcal{L}_{SM} = \mathbb{E}_{q(x_t|x_0)}\left[\frac{\lambda(t)}{2}\left\|\epsilon_\theta - \frac{x_t - \sqrt{\alpha_t} x_0}{\sqrt{1-\alpha_t}}\right\|_2^2\right]$$

其中 $\lambda(t) = (1-\alpha_t)/\alpha_t$ 是weighting function，compensate不同diffusion step的variance scale差异。

### 2.4 Overall Denoising Loss

最终simplified training objective：

$$\mathcal{L}_{Denoise} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

- $t$：uniformly sampled from $\{1, ..., T\}$
- $x_0$：从training data采样
- $\epsilon$：从 $\mathcal{N}(0, I)$ 采样的noise
- $x_t$：通过forward process计算得到

**Intuition**：这个loss本质上是让neural network学习predict added noise，which implicitly captures underlying score function of data distribution。Minimizing这个loss等价于maximizing variational lower bound of log-likelihood。

### 2.5 Diffusion Taxonomy

论文将diffusion models分为三类：

#### 2.5.1 DDPMs (Denoising Diffusion Probabilistic Models)
- Discrete-time Markov chain
- Forward: $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$
- Reverse: learned transition kernel $p_\theta(x_{t-1}|x_t)$
- 训练objective: $\mathbb{E}_{t, x_0, \epsilon}[\lambda(t)\|\epsilon - \epsilon_\theta(x_t, t)\|_2^2]$

#### 2.5.2 SGMs (Score-Based Generative Models)
- Explicitly estimate score function $\nabla_x \log p(x)$
- Noise-conditional score network $s_\theta(x_t, t)$
- 训练: $\mathbb{E}_{t, x_0, \epsilon}[\lambda(t)\|\epsilon + \sigma_t s_\theta(x_t, t)\|^2]$
- Sampling: annealed Langevin dynamics
  $$x_t^{(i+1)} \leftarrow x_t^{(i)} + \frac{s_t}{2} s_\theta(x_t^{(i)}, t) + \sqrt{s_t} \epsilon^{(i)}$$

#### 2.5.3 Score SDEs (Stochastic Differential Equations)
Continuous-time formulation：

Forward SDE:
$$dx = f(x, t)dt + g(t)dw$$

- $f(x, t)$：drift function（deterministic motion）
- $g(t)$：diffusion coefficient（stochastic variation）
- $dw$：Wiener process (Brownian motion)

Reverse-time SDE (Anderson's result):
$$dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$$

Probability flow ODE (deterministic equivalent):
$$dx = \left[f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log q_t(x)\right]dt$$

**Important connection**：DDPM对应的SDE是 $dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)}dw$，SGM对应 $dx = \sqrt{\frac{d[\sigma(t)^2]}{dt}}dw$。

参考：[Song et al. Score SDE paper](https://arxiv.org/abs/2011.13456) | [DDPM by Ho et al.](https://arxiv.org/abs/2006.11239)

---

## 3. 3D Vision Fundamentals

### 3.1 3D Representations分类

论文将3D representations分为三大类，这是理解后续3D diffusion tasks的基础：

#### 3.1.1 2D Representations for 3D

**Depth Map**: 每个pixel value对应camera到surface的距离。
- 优点：efficient，与2D image processing兼容
- 缺点：只能表示visible surfaces，occluded areas缺失
- 生成方式：stereo vision, structured light, ToF cameras

**Multi-View Images**: 多视角2D images，利用parallax恢复geometry。
- 优点：保留real-world textures
- 缺点：需要复杂算法进行depth estimation和3D reconstruction

#### 3.1.2 Explicit 3D Representations

**Point Clouds**: 3D空间中的sampled points，每个point有 $(x, y, z)$ coordinates及optional attributes（color, normal, intensity）。
- LiDAR, structured light scanners的直接output
- 缺点：lack topological information，需要转换为mesh或voxel

**Voxel Grids**: 将3D空间discretize为cubic units。
- 优点：regular structure，适合3D CNNs
- 缺点：cubic growth with resolution，memory intensive
- 解决方案：octrees（hierarchical representation）

**Meshes**: vertices, edges, faces的interconnected结构。
- 优点：efficient rendering，minimal data for complex surfaces
- 应用：computer graphics, 3D modeling, gaming, 3D printing

#### 3.1.3 Implicit 3D Representations

**Neural Implicit Representations**: 用neural networks表示continuous functions，可在任意3D point query。

**Occupancy Fields**: binary function判断point是否被surface occupied。

**Signed Distance Fields (SDFs)**:
- Inside object: negative values
- Surface: zero
- Outside: positive values
- 优点：smooth, continuous，适合shape blending, deformation

**3D Gaussian Splatting**: 每个point关联一个Gaussian blob，probabilistic representation。
- 优点：modeling uncertainty, handling sparse/noisy data
- 应用：volumetric rendering, scene reconstruction

### 3.2 3D Vision的Challenges

1. **Occlusion**: objects被其他objects遮挡，导致incomplete data
2. **Varying Point Density**: sensor距离、角度、反射特性导致non-uniform sampling
3. **Noise and Outliers**: sensor precision limitations, environmental factors

---

## 4. 3D Diffusion Generation Tasks - 详细分析

### 4.1 Unconditional 3D Generation

**目标**：从random noise直接生成diverse 3D structures，without explicit inputs。

**Key methods evolution**：

1. **RenderDiffusion (Anciukevičius et al., 2023)**:
   - 使用2D diffusion models
   - 在每个denoising step rendering intermediate 3D representations
   - 通过inductive biases enforce 3D consistency
   - 仅依赖2D supervision

2. **Point-Voxel Diffusion (Zhou et al., 2021)**:
   - Probabilistic framework
   - Point-voxel hybrid representations
   - Reverse diffusion process transforms noisy point clouds → structured 3D shapes

3. **TIGER (Ren et al., 2024)** - Time-varying denoising model:
   - **Key insight**: attention和convolution在diffusion中的dynamic roles
   - Early stages: global attention mechanisms dominate → define overall shapes
   - Later stages: local convolutions → refine surface details
   - 使用optimizable masks adaptively fuse features

4. **Point-Diff (Zheng et al., 2024)**:
   - Pre-training strategy
   - Point cloud recovery as conditional generation task
   - Recurrent uniform sampling

5. **DiT-3D (Mo et al., 2023)**:
   - Hybrid 2D-3D approach
   - Voxelize point clouds
   - Fine-tune pre-trained 2D models for 3D synthesis

**Implicit representation approaches**:

- **Diffusion-SDF (Shim et al., 2023)**: 
  1. Generate low-resolution SDFs
  2. Apply diffusion-based super-resolution
  3. Direct mesh extraction via Marching Cubes

- **Chou et al., 2023**: Modulate SDFs into diffusion-trained latent vectors

- **DiffRF (Müller et al., 2023)**: Generate explicit voxel-grid RFs using 3D DDPMs

- **Triplane Diffusion (Shue et al., 2023)**: 
  - Project 3D scenes → 2D triplanes
  - Leverage 2D diffusion models
  - Synthesize neural RFs efficiently

**Latent space compression**:

- **3D-LDM (Nam et al., 2022)**: Auto-decoders on SDFs → compact latent space
- **LION (Vahdat et al., 2022)**: Hierarchical VAEs with dual diffusion models
  - One for global shape vectors
  - Another for point cloud structures
- **3DShape2VecSet (Zhang et al., 2023a)**: Transformer-friendly latent with Radial Basis Functions

**3D Completion tasks**:

- **Lyu et al., 2022**: Two-stage coarse-to-fine diffusion
  1. Generate rough shape
  2. Improve details

- **Kasten et al., 2024**: Integrate semantic guidance from text-to-image diffusion

- **LiDAR completion (Nunes et al., 2024)**: 
  - **Key innovation**: Perturb individual points locally rather than globally
  - Preserve scene structure while learning details

参考：[Point-E by OpenAI](https://arxiv.org/abs/2212.08751) | [LION: Latent Point Diffusion](https://arxiv.org/abs/2210.06678)

### 4.2 Image-to-3D Generation

**核心挑战**：从sparse 2D supervision重建3D，需要resolve inherent ambiguities。

**Methodology分类**：

#### A. Diffusion priors + geometric constraints

**SJC (Wang et al., 2023a) - Score Jacobian Chaining**:
- Chain gradients from diffusion model with differentiable Jacobian renderer matrix
- Translate 2D image gradients → 3D asset updates
- **Advantage**: 3D generation without 3D training data

**NeuralLift-360 (Xu et al., 2023a)**:
- CLIP-guided sampling
- Merge diffusion priors with reference images
- Scale-invariant depth supervision
- Reduce reliance on multi-view depth consistency

**Viewset Diffusion (Szymanowicz et al., 2023)**:
- Train diffusion on multi-view 2D data
- Embed bijective mapping between viewsets ↔ 3D models
- Ensure geometric alignment within denoising network

**MeshDiffusion (Liu et al., 2023b)**:
- Deformable tetrahedral grids represent 3D meshes
- Bypass topology irregularities
- Score-based diffusion on parametric mesh representations

#### B. Multi-view hypothesis generation

**RealFusion (Melas-Kyriazi et al., 2023)**:
- Optimize text prompts to guide diffusion
- Synthesize novel views
- Unsupervised 3D reconstruction of arbitrary objects

**One-2-3-45 (Liu et al., 2024c)**:
1. Generate multi-view images via 2D diffusion
2. Apply cost-volume-based neural surface reconstruction

**EfficientDreamer (Hu et al., 2024)**:
- Produce four orthogonal views from text prompts
- Impose geometric priors

**Wonder3D (Long et al., 2024)**:
- Cross-domain attention
- Geometry-aware normal fusion
- Enhanced multi-view consistency

**InstantMesh (Xu et al., 2024a)**:
- Multi-view diffusion + Large Reconstruction Model
- Directly predict meshes from consistent multi-view inputs
- Accelerate high-quality mesh generation

#### C. Two-stage approaches

**Make-It-3D (Tang et al., 2023)**:
1. Optimize NeRF using diffusion priors + reference image constraints
2. Refine into textured point cloud with high-fidelity textures

**LAS-Diffusion (Zheng et al., 2023)**:
1. Occupancy-diffusion stage → low-resolution shape shells
2. SDF-diffusion → refine geometric details
- View-aware local attention aligns 2D image patches with 3D voxel features

**Magic123 (Qian et al., 2023)**:
- NeRF-based coarse geometry → memory-efficient mesh optimization
- Balance exploration (2D diffusion) vs exploitation (3D priors)

#### D. 3D Gaussian Splatting approaches

**LGM (Tang et al., 2024a)**:
- Multi-view Gaussian features for differentiable rendering
- Integrate with asymmetric U-Net
- Rapid 3D model generation from single images

**GRM (Xu et al., 2024b)**:
- Transformer-based architecture
- Map pixels → 3D Gaussian parameters
- Re-project to form dense scene representations

参考：[DreamFusion](https://arxiv.org/abs/2209.14988) | [Magic3D](https://arxiv.org/abs/2211.10440) | [InstantMesh](https://arxiv.org/abs/2404.07191)

### 4.3 Text-to-3D Generation

**核心**：将textual descriptions转换为3D models，连接language semantics与3D structural representations。

**Evolution路径**：

**早期dual-stage (Point-E, Nichol et al., 2022a)**:
1. Text-to-image diffusion → single view 2D representation
2. Secondary conditional diffusion → 3D point cloud

**Patch-based autoencoder (Li et al., 2023b)**:
- Encode truncated SDFs into localized Gaussian latent spaces
- Voxelized diffusion model
- Model inter-patch relationships + global structural coherence

**Latent-NeRF (Metzer et al., 2023)**:
- Operate entirely in compressed latent space
- Bypass RGB space
- Reduce repeated image encoding during guidance steps

**3D consistency enforcement (Seo et al., 2024)**:
1. Generate preliminary 3D point clouds from text
2. Project into multi-view depth maps
3. Feed depth maps into diffusion process
4. Viewpoint-aware optimization

**Geometry-appearance separation**:

**Fantasia3D (Chen et al., 2023b)**:
- Separate geometry and appearance modeling
- Text features independently guide shape formation + surface properties

**Sequential pipeline (Wu et al., 2023b)**:
1. Diffusion model generates geometric sketch
2. Colorization based on established shape
- Ensure structure-texture alignment

**3D Gaussian Splatting innovations**:

**GSGEN (Chen et al., 2024d)**:
- Two-phase optimization:
  1. Construct coarse 3D structure
  2. Refine details via compactness-driven densification

**4D dynamic (Ling et al., 2024)**:
- Dynamic 3D Gaussian mixtures + deformation fields
- Represent 4D motion
- Integrate text-to-image, text-to-video, multi-view diffusion in score distillation

**GaussianDreamer (Yi et al., 2024)**:
- 3D diffusion prior seeds geometry
- 2D diffusion iteratively improves shape + appearance

**Score Distillation Sampling (SDS) improvements**:

**ProlificDreamer (Wang et al., 2024a)**:
- Reimagine SDS as variational process
- Treat 3D parameters as probabilistic distributions
- Align multi-view renderings with 2D diffusion priors

**4D-fy (Bahmani et al., 2024)**:
- Hybrid strategy: static scene initialization + variational SDS-driven refinement
- Mitigate "Janus problem" artifacts

**Time-Prioritized SDS (Huang et al., 2023)**:
- Use decay function to prioritize critical denoising steps
- Accelerate convergence + improve quality

**DreamFusion (Poole et al., 2023)**:
- Bypass 3D supervision entirely
- Optimize NeRF through 2D diffusion-guided loss functions
- Distill 3D consistency from multi-view image projections

**Direct2.5 (Lu et al., 2024)**:
- Fine-tune pre-trained 2D diffusion → multi-view 2.5D framework
- Directly capture 3D structural distributions
- Preserve diversity and generalization of 2D models

### 4.4 Texture Generation

**目标**：mapping realistic surface details onto 3D meshes，preserve geometric accuracy。

**Key methods**:

**TexFusion (Cao et al., 2023)**:
- Text-guided 3D texture synthesis
- Latent diffusion model + 3D-consistent sampling strategy
- Iteratively denoise 2D rendered views
- Aggregate predictions into unified latent texture map

**Point-UV Diffusion (Yu et al., 2023)**:
- Two-stage coarse-to-fine framework
- UV mapping + diffusion models
- Seamless texture generation

**TexOct (Li et al., 2024)**:
- Octree-based diffusion model
- Efficiently represent surface-sampled point clouds
- Resolve occlusion and sparse sampling artifacts

**Multi-view consistency solutions**:

**TexPainter (Zhang et al., 2024b)**:
- Decode noise-free latent states from pre-trained diffusion model (DDIM)
- Decode to color space at each denoising step
- Latent code optimization to align rendered views

**One-2-3-45++ (Liu et al., 2024b)**:
1. Fine-tune 2D diffusion → generate six consistent viewpoints
2. Coarse-to-fine 3D diffusion → predict textured meshes

**Lighting integration**:

**DreamMat (Zhang et al., 2024d)**:
- Train under specific lighting conditions
- Produce shadow-free PBR materials
- Ensure consistency with geometry

**Paint3D (Zeng et al., 2024)**:
- Lighting-agnostic textures
- Pre-trained 2D diffusion + UV completion-upscaling pipeline

**Depth-aware methods**:

**Text2Tex (Chen et al., 2023a)**:
- Depth-aware diffusion model
- Synthesize high-resolution partial textures from text prompts
- Progressively refine local textures via viewpoints

**TEXTure (Richardson et al., 2023)**:
- Depth-to-image diffusion with dynamic trimap segmentation
- Seamless multi-view textures

### 4.5 Human Avatar Generation

**Focus areas**: head generation, full-body models, dynamic motion synthesis.

**Head generation**:

**RODIN (Wang et al., 2023b)**:
- Neural Radiance Fields
- Project 3D feature maps onto 2D planes
- 3D-aware convolutions preserve spatial relationships

**AvatarCraft (Jiang et al., 2023)**:
- Text-guided diffusion
- Parameterized control over body shape and pose

**Morphable Diffusion (Chen et al., 2024b)**:
- Deformation diffusion model
- Synthesize 3D meshes conditioned on identity + expression parameters
- Multi-view consistency constraints

**DiffusionGAN3D (Lei et al., 2024)**:
- Integrate pre-trained 3D generators with text-to-image diffusion
- Relative Distance Loss + adaptive triplanes
- Enhanced diversity

**Full-body generation**:

**PrimDiffusion (Chen et al., 2023c)**:
- Volumetric primitives (small 3D units encoding radiance + motion)
- Streamline computation
- Handle intricate body topology

**Chupa (Kim et al., 2023a)**:
- Decompose into 2D normal map generation + 3D reconstruction
- Multi-view consistency + reduced computational overhead

**DINAR (Svitov et al., 2023)**:
- Neural textures + SMPL-X parametric model
- Cartoon-style avatars
- Efficient texture recovery via latent diffusion

**SITH (Ho et al., 2024)**:
- Predict occluded back views using diffusion
- Reconstruct complete meshes from single view

**Dynamic motion**:

**AnimateMe (Gerogiannis et al., 2024)**:
- Graph neural networks as denoising diffusion models in mesh space
- Temporal coherence in facial animations

**InterDiff (Xu et al., 2023b)**:
- Interaction diffusion + physics-based corrections
- Generate long-term 3D human-object interaction sequences

**PhysDiff (Yuan et al., 2023)**:
- Integrate physics simulators into diffusion pipeline
- Project denoised motions onto physically plausible trajectories
- Motion imitation

### 4.6 Scene Generation

**三种主要approach**：

#### Image-guided methods

**NeuralField-LDM (Kim et al., 2023b)**:
- Scene auto-encoder encode image-pose pairs → neural fields
- Latent compression + hierarchical diffusion modeling

**Diffusionerf (Wynn & Turmukhambetov, 2023)**:
- DDPM trained on synthetic RGBD patches
- Regularize scene priors in sparse-view NeRF scenarios

**LucidDreamer (Chung et al., 2023)**:
- Iteratively refine 3D scenes
- "Dreaming": geometric image generation + 3D projection
- "Alignment": point cloud integration

**BlockFusion (Wu et al., 2024)**:
- Compress 3D planes into latent space via VAE
- Scalable diffusion-based scene synthesis

**LiDAR Diffusion (Ran et al., 2024)**:
- Enhance geometric authenticity using prior knowledge
- Realistic sensor data generation

#### Text-guided methods

**SceneScape (Fridman et al., 2024)**:
- Text-to-image models + depth prediction
- 3D-consistent video generation

**Locally Conditioned Diffusion (Po & Wetzstein, 2024)**:
- Localized scene control via bounding boxes + text prompts

**Text2NeRF (Zhang et al., 2024c)**:
- Text-to-image diffusion + NeRF
- Progressive refinement + depth alignment
- Photorealistic multi-view scenes

#### Scene graph-based methods

**GraphDreamer (Gao et al., 2024)**:
- Resolve attribute conflicts in text-to-3D conversion
- Use graph information for structured semantic relationships

**DiffuScene (Tang et al., 2024b)**:
- Synthesize unordered object collections with denoised attributes

**EchoScene (Zhai et al., 2024)**:
- Dual-branch architecture
- Graph-convolutional information exchange during denoising
- Global consistency between scene graph nodes

### 4.7 3D Editing and Manipulation

**Key approaches**:

**DATID-3D (Kim & Chun, 2023)**:
- Text-to-image diffusion + adversarial training
- Transform pre-trained 3D generator samples → diversified target images
- Maintain text prompt diversity

**Sketch-based editing**:

**SKED (Mikaeili et al., 2023)** & **SketchDream (Liu et al., 2024a)**:
- Sketch guidance + text prompts
- Geometric reasoning for coarse positioning
- Generative models for detail refinement
- Multi-view sketches + depth-aware diffusion
- Specialized loss functions for 3D consistency

**Volumetric editing**:

**Vox-E (Sella et al., 2023)**:
- Volumetric regularization losses in 3D space
- Maintain global structural correlations during voxel-based editing

**DreamEditor (Zhuang et al., 2023)**:
- Grid-based neural fields
- Semantic-aware diffusion models
- Localized editing

**GaussianEditor (Chen et al., 2024c)**:
- Gaussian semantic tracing
- Hierarchical splatting constraints
- Enhanced control

**Identity-preserving editing**:

**HeadSculpt (Han et al., 2024)**:
- Coarse-to-fine workflow
- Landmark-based ControlNet + text inversion
- Score-blending techniques for facial identity preservation

**3D Paintbrush (Decatur et al., 2024)**:
- Localized stylization via neural texture maps
- Cascaded score distillation
- Precise adherence to surface geometry

### 4.8 Novel View Synthesis

**核心挑战**：generate consistent multi-perspective scene representations from limited inputs。

**Hybrid 2D-3D frameworks**:

**NeRDi (Deng et al., 2023)**:
- Optimize neural radiance fields
- Minimize diffusion-based distribution losses across rendered views

**HOLODIFFUSION (Karnewar et al., 2023)**:
- Train 3D UNet using explicit-implicit feature grids
- 2D supervision for spatial consistency

**Multi-view consistency via geometric constraints**:

**Pose-guided diffusion (Tseng et al., 2023)**:
- Epipolar attention layers guided by camera poses
- Feature correspondences between input and novel views
- Particularly effective for large camera motions

**DiM (Watson et al., 2023)**:
- Stochastic conditioning
- Cross-attention UNet variants
- Improve 3D consistency during image generation

**SyncDreamer (Liu et al., 2024d)**:
- 3D-aware feature attention
- Joint probability modeling
- Synchronized multi-view synthesis

**Sparse input scenarios**:

**SparseFusion (Zhou & Tulsiani, 2023)**:
- View-conditioned latent diffusion models
- Optimize neural 3D representations from segmented images

**Zero123 (Liu et al., 2023a)** & **Zero123++ (Shi et al., 2023)**:
- Single-image reconstruction through synthetic dataset training
- Multi-view joint distribution modeling
- Scaled reference attention + modified noise schedules

### 4.9 Depth Estimation

**核心**：从2D visual inputs预测3D scene geometry。

**Synthetic data generation approaches**:

**Atlantis (Zhang et al., 2024a)** - Depth-2-Underwater ControlNet:
- Transform terrestrial depth maps → realistic underwater scenes
- Preserve geometric accuracy using stable diffusion

**Tosi et al., 2024**:
- Text-to-image diffusion with depth-aware control
- Generate challenging synthetic environments with depth annotations

**Marigold (Ke et al., 2024)**:
- Stable diffusion-based framework
- State-of-the-art monocular depth estimation
- Synthetic data fine-tuning
- Retain pre-trained visual priors by modifying only denoising U-Net

**Pre-training methods**:

**EcoDepth (Patni et al., 2024)**:
- Incorporate global contextual features of pre-trained Vision Transformers
- Improve depth prediction accuracy

**DDVM (Saxena et al., 2024)**:
- Self-supervised pre-training in image translation tasks
- Subsequent supervised RGB-D fine-tuning

**Architectural innovations**:

**DDP (Ji et al., 2023)**:
- Multitask conditional diffusion framework
- Competitive benchmark performance without task-specific modifications

**DADP (Li et al., 2023a)**:
- Integrate noise and depth predictors
- Enhance structural consistency in sparsely annotated autonomous driving scenarios

**DiffusionDepth (Duan et al., 2024)**:
- Self-refinement approach
- Learn to reverse diffusion processes from refined depth predictions to random distributions
- Address sparse ground-truth limitations

---

## 5. Datasets and Metrics

### 5.1 Key 3D Datasets

| Category | Dataset | Samples | Type | Source |
|----------|---------|---------|------|--------|
| **Object** | ShapeNet | 51K | Mesh | Synth |
| | Objaverse | 800K | Mesh | Synth |
| | CO3D | 19K | MultiView | Real |
| | ModelNet40 | 12K | Mesh | Synth |
| | Cap3D | 785K | Mesh | Synth |
| **Human** | Renderpeople | 40K | Mesh | Synth |
| | THuman 2.0 | 500 | RGBD | Real |
| | UESTC | 25.6K | RGBD | Real |
| | HumanML3D | 15K | Mesh | Synth |
| | AMASS | 11K | Mesh | Synth |
| **Scene** | Realestate10K | 750K | MultiView | Real |
| | ScanNet | 1.5K | RGBD, PC | Real |
| | KITTI | 389 | RGBD, PC | Real |
| | nuScenes | 1K | PC | Real |
| | Waymo | 2K | PC | Real |

### 5.2 Evaluation Metrics分类

**Distance Metrics**:
- **CD (Chamfer Distance)**: average nearest neighbor distance between point sets
- **EMD (Earth Mover's Distance)**: minimum conversion cost between distributions
- **TMD**: total difference between distributions in multi-dimensional space
- **LFD (Light Field Distance)**: differences in light fields from different viewpoints

**Coverage Metrics**:
- **COV**: proportion of generated samples covering test samples
- **1-NNA**: mixing degree between test and generated samples based on nearest neighbors

**Distribution Metrics**:
- **FID**: distribution similarity through mean and covariance in feature space
- **KID**: distribution distance based on kernel functions
- **FVD**: extends FID for video frame features
- **MMD**: distance via optimal matching of elements

**Quality Metrics**:
- **PSNR**: signal fidelity based on peak signal-to-noise ratio
- **SSIM**: image similarity through structure, brightness, contrast
- **VQ**: comprehensive aesthetic and perceptual quality
- **PQ**: quality based on human perception

**Error Metrics** (for depth estimation):
- **REL**: mean absolute relative error
- **SqRel**: mean squared relative error
- **RMSE**: root mean squared error
- **RMSElog**: RMSE on logarithmic scale

参考：[ShapeNet](https://shapenet.org/) | [Objaverse](https://objaverse.allenai.org/) | [ScanNet](http://www.scan-net.org/) | [KITTI](http://www.cvlibs.net/datasets/kitti/)

---

## 6. Limitations and Future Directions

### 6.1 Computational Efficiency

**Problem**: 
- Diffusion models需要large number of iterations
- High-dimensional 3D data → memory和computation requirements scale rapidly
- Training time + inference speed都受影响

**Potential solutions**:
- Optimize number of diffusion steps
- Explore more efficient architectures
- Consistency models, rectified flow等加速方法

### 6.2 Multimodal Fusion

**Challenge**: 
- Effectively integrating 2D images, 3D geometry, textual descriptions
- Single-modality tasks已成功，multimodal fusion仍有improvement space

**Future directions**:
- Cross-attention mechanisms
- Embeddings capturing 2D-3D-text relationships
- Unified architectures handling multiple modalities

### 6.3 Large-scale Pretraining and Transfer Learning

**Current state**: 
- 2D vision有large-scale pre-training成功案例
- 3D diffusion modelsunderexplored due to data scarcity + computational demands

**Future research**:
- Transfer learning from 2D pre-trained models to 3D tasks
- Understanding transferability across 2D ↔ 3D domains
- Robust pre-training strategies

### 6.4 Interpretability and Fine-grained Control

**Limitation**: 
- Lack of interpretability over generation process
- Users需要control shape, texture, pose等specific aspects

**Solutions**:
- Interpretable latent spaces
- Conditional generation mechanisms
- Disentanglement of various attributes
- Explainability tools

### 6.5 Complex and Dynamic Scenes

**Current limitation**: 
- Most models针对static objects/scenes
- Dynamic environments (moving objects, changing lighting, large-scale outdoor)未fully addressed

**Future work**:
- Encoding temporal information
- Spatiotemporal diffusion processes
- Hybrid models combining static understanding + temporal dynamics

### 6.6 Physical Constraints

**Motivation**: 
- Generate physically realistic and plausible 3D content
- Real-world applications需要physical accuracy

**Approaches**:
- Conservation laws, collision detection, material properties
- Physics-based priors or loss functions
- Geometric consistency enforcement
- Simulate gravity, fluid dynamics等real-world interactions

参考：[Consistency Models](https://arxiv.org/abs/2303.01469) | [Rectified Flow](https://arxiv.org/abs/2209.03003)

---

## Key Insights and Intuition Building

### 1. 为什么Diffusion Models适合3D Vision?

**Probabilistic nature**: 3D data often contains ambiguities, missing regions, sensor noise。Diffusion models operate probabilistically，能generate multiple plausible solutions，adapt to incomplete/sparse input。

**Iterative refinement**: 通过iterative denoising process，excel in scenarios requiring robustness to occlusions or partial data，like reconstructing full 3D shapes from limited views。

**Score function estimation**: Score function $\nabla_x \log p(x)$ 指向probability density增加最快的方向，在high-dimensional 3D space中提供稳定的gradient guidance。

### 2. 3D vs 2D Diffusion的关键差异

| Aspect | 2D Diffusion | 3D Diffusion |
|--------|--------------|--------------|
| **Data scale** | Vast (photos, videos) | Limited, labor-intensive curation |
| **Structural complexity** | Regular grid (pixels) | Meshes, voxels, point clouds (specialized architectures) |
| **Computational demands** | High but manageable | Dramatically escalated (volumetric rendering) |
| **Topology handling** | Grid topology | Variable topology, order-invariant processing needed |

### 3. 3D Representation Choice的影响

| Representation | Advantages for Diffusion | Challenges |
|---------------|--------------------------|------------|
| **Point Clouds** | Direct sensor output, sparse | Lack topology, order-invariant processing needed |
| **Voxel Grids** | Regular structure, CNN compatible | Cubic memory growth |
| **Meshes** | Efficient rendering, minimal data | Topology irregularities |
| **Neural Implicit (SDF)** | Continuous, smooth | Decoding to explicit geometry challenging |
| **3D Gaussian Splatting** | Probabilistic, handles uncertainty | Emerging, less mature |

### 4. Training Objective的数学直觉

**Forward process的meaning**：
$$x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon$$

- $\sqrt{\alpha_t}$：signal retention factor，随t递减
- $\sqrt{1-\alpha_t}$：noise injection factor，随t递增
- 当 $\alpha_t \to 0$：pure noise
- 当 $\alpha_t \to 1$：original signal

**Reverse process的learning**：
Neural network学习 $\epsilon_\theta(x_t, t)$ 预测noise，which implicitly learns：

$$\nabla_{x_t} \log p(x_t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\alpha_t}}$$

这个score function指向data distribution的高密度区域，指导reverse process生成realistic samples。

### 5. Score Distillation Sampling (SDS)的核心思想

**DreamFusion的key insight**: 
对于3D representation $\theta$ (e.g., NeRF)，rendered image $I = R(\theta, c)$，diffusion model提供prior：

$$\nabla_\theta \mathcal{L}_{SDS} = \mathbb{E}_{t, \epsilon}\left[w(t)(\epsilon_\theta(I_t, t) - \epsilon) \frac{\partial I}{\partial \theta}\right]$$

- $w(t)$：weighting function
- $\epsilon_\theta(I_t, t) - \epsilon$：noise prediction residual
- $\frac{\partial I}{\partial \theta}$：rendering的Jacobian

这个gradient通过differentiable rendering backpropagate到3D parameters，允许2D diffusion priors指导3D optimization。

---

## 总结与展望

这篇survey系统地梳理了diffusion models在3D vision中的applications，从mathematical foundations到specific tasks，从datasets到evaluation metrics。论文的核心贡献在于：

1. **Comprehensive categorization**: 覆盖unconditional generation, image-to-3D, text-to-3D, texture generation, human avatar, scene generation, 3D editing, novel view synthesis, depth estimation等9大task categories

2. **Technical depth**: 详细讲解DDPM, SGM, Score SDE三种formulations的mathematical principles和connections

3. **Practical guidance**: 总结datasets, metrics, challenges, future directions

4. **Future roadmap**: Identifying computational efficiency, multimodal fusion, large-scale pretraining, interpretability, dynamic scenes, physical constraints等key research directions

对于building intuition而言，核心要点是：diffusion models的probabilistic framework和iterative refinement机制天然适合处理3D data的uncertainty和complexity，而3D representations的选择（explicit vs implicit, point cloud vs voxel vs mesh）直接决定了architecture design和computational efficiency。未来的research将focus on efficiency optimization, multimodal fusion, physical constraints integration等方向。

**Key references for deep dive**:
- [DDPM Original Paper](https://arxiv.org/abs/2006.11239)
- [Score SDE](https://arxiv.org/abs/2011.13456)  
- [DreamFusion](https://arxiv.org/abs/2209.14988)
- [Point-E](https://arxiv.org/abs/2212.08751)
- [3D Gaussian Splatting](https://arxiv.org/abs/2308.14737)
- [InstantMesh](https://arxiv.org/abs/2404.07191)
- [Objaverse Dataset](https://objaverse.allenai.org/)
- [Consistency Models](https://arxiv.org/abs/2303.01469)
