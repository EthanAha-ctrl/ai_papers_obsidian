---
source_pdf: TERRA EXPLORABLE NATIVE 3D WORLD MODEL.pdf
paper_sha256: 1394a507363d427a75504bce76980760c5b4ca258156a9e9749fbb5b9669ab21
processed_at: '2026-08-12T13:34:53-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Terra 用大白话讲：直接在 3D 空间搭积木的 World Model

Andrej，我们把这篇 paper 揉碎了用最直白的话再过一遍。为了 build 你的 intuition，我们可以把 world model 的生成过程比作“造一个可以走进去的房子”。

现有的 world model 大多是 2D（比如 Sora）或者 2.5D（比如 Cosmos、Prometheus）。2D model 就像是在房子的每一面墙外面画一幅极其逼真的画，你站在外面看觉得很震撼，可一旦你想推门进去，从不同的视角看，就会发现墙角对不上，家具会变形。2.5D model 试图解决这个问题，它在画画的同时顺便量一下墙离你有多远（depth），并且记录你站的位置（pose）。但因为画面、深度和位置是深度耦合的，网络很难在 data-driven 的过程中完美学会怎么把这些东西拼成一个无缝的 3D 空间，多视角看的时候依然会有几何漂移。

Terra 的思路极其干脆：我们不要在 2D 画面上修修补补了，直接在 3D 空间里“搭积木”。它生成的是一堆带有语义信息的 3D 点（point latents），然后再把这些点变成可以渲染的 3D Gaussian primitives。因为底下的物理实体就是同一套 3D 积木，所以无论你用什么视角去看，通过 rasterizer 投影出来的画面在几何上绝对是一致的，彻底绕开了 2D/2.5D model 学习透视变换约束的痛苦。

---

## 1. 怎么把现实世界变成“积木”：P2G-VAE

现实世界扫描出来的点云动辄上百万个点，直接拿去生成计算量爆炸。所以 Terra 先搞了一个 Variational Autoencoder (VAE)，叫 P2G-VAE。它的任务是把这上百万个点压缩成大概 5000 个“精华积木”（point latents），然后又能把这些积木还原回去，变成可以渲染的 3D Gaussian。

这里有个极其巧妙的工程设计：**Robust Position Perturbation**。
传统的 VAE 为了让 latent space 规整，会用 KL loss 把所有的特征拉成正态分布。可是对于 3D 点来说，它的坐标本身就是极其重要的几何信息！如果你把一个原本在墙上的点的坐标也强行拉成正态分布，这个点就飞到半空中了，局部几何结构全毁了。

Terra 的做法是给坐标加上一点微小的高斯噪声，特征还是照常做 KL 正则化。公式如下：
$$ p = \hat{p} + \boldsymbol{n}, \quad \boldsymbol{f} \sim \mathcal{N}(\text{mean}(\hat{\boldsymbol{f}}), \text{diag}(\text{var}(\hat{\boldsymbol{f}}))) $$
变量解释：
- $\hat{p}$ 和 $\hat{\boldsymbol{f}}$ 分别是原本的位置和特征。
- $\boldsymbol{n} \sim \mathcal{N}(\boldsymbol{0}, \sigma^2 \boldsymbol{I}_3)$ 是加在位置上的小扰动。
- $p$ 和 $\boldsymbol{f}$ 是进入 bottleneck 的最终表示。

**Intuition**: 为什么要故意给位置加噪？因为生成模型生成的点必然带有误差。如果在训练 VAE decoder 的时候，它见过的输入都是完美无瑕的，那生成时遇到稍微有点偏差的点，它就不会渲染了。加扰动相当于给 decoder 打疫苗，让它对位置噪声免疫，后期生成质量大幅提升（实验中 P-FID 从 15.28 降到 8.79）。

Decoder 端把 5000 个点变回几十万个 Gaussian 时，用了 Adaptive Upsampling。每个点像 DETR 里的 object query 一样，生出 $K$ 个子点，预测相对父点的位移和特征残差，慢慢把结构填补完整。

---

## 2. 怎么凭空“捏”出这些积木：SPFlow

有了 P2G-VAE 压缩出的 5000 个点，我们怎么让模型凭空生成符合真实房间分布的点集呢？Terra 用了 Flow Matching（比 diffusion 更直截了当的生成模型）。

公式很直白：
$$ \boldsymbol{P}_t = t\boldsymbol{P} + (1-t)\boldsymbol{N} $$
变量解释：
- $\boldsymbol{P}$ 是干净的 point latents，$\boldsymbol{N}$ 是纯随机噪声。
- $t \in [0,1]$ 是时间步，$t=0$ 时全是噪声，$t=1$ 时全是真实数据。
- 模型的任务就是预测速度向量 $\boldsymbol{V}$，把噪声沿着直线推到真实数据那里。

但是这里有个巨大的坑：**点云是无序的**。
在图像生成里，左上角的噪声变成左上角的像素，这是天经地义的。但在点云里，如果你硬规定序列里的第 1 个噪声点必须去生成房间最右边的那个真实点，那这个噪声点要跨越整个房间去“搬家”，生成的轨迹极其弯曲，模型根本收敛不了。

Terra 提出了神级 trick：**Distance-aware Trajectory Smoothing**。
既然让左边的噪声去生成右边的点不合理，那我们在训练前，先用算法把离得最近的噪声点和真实点配对好不就行了？
$$ \mathcal{M}^* = \arg\min_{\mathcal{M}} \sum_{m=1}^M \| \boldsymbol{p}^{(m)} - \boldsymbol{N}_{\mathcal{M}_m, :3} \|^2 $$
变量解释：
- $\boldsymbol{p}^{(m)}$ 是第 $m$ 个真实点的位置。
- $\boldsymbol{N}_{\mathcal{M}_m, :3}$ 是分配给它的噪声点的前 3 维坐标。
- $\mathcal{M}^*$ 是一个排列矩阵，目标是让所有配对点的距离平方和最小。

这其实就是一个经典的 Linear Assignment Problem (LAP)，用 Jonker-Volgenant algorithm 算法一步就能解出。把配对搞定后，每个噪声点只需要在原地附近稍微挪一挪就能变成真实点，flow matching 的轨迹瞬间拉直。这个 trick 在实验里起到了定海神针的作用，去掉它 P-FID 直接从 8.79 爆炸到 24.84。

---

## 3. 怎么实现“探险”：Progressive Exploration

World model 的终极目标是能探索。Terra 把探索变成了 point latent 空间里的“outpainting”（向外补全）。

过程是这样的：
1. 先无条件生成一个初始的小房间 $S_0$。
2. 选一个探索方向，把当前已知房间在这个方向边缘的一部分点作为条件 $C_1$。
3. 模型根据这些条件，生成下一块区域的点 $S_1$。
4. 把新点拼接到老点上，继续往前走。

公式表达为：
$$ S_0 = g(\emptyset, C_0; \boldsymbol{\theta}), \quad S_i = g(\mathbb{S}_{i-1}, C_i; \boldsymbol{\theta}), \quad \mathbb{S}_i = \{S_0, S_1, \ldots, S_i\} $$
变量解释：
- $\mathbb{S}_{i-1}$ 是截至 $i-1$ 步所有已知的场景块。
- $C_i$ 是从已知块中切出来的条件。
- $g(\cdot, \cdot; \boldsymbol{\theta})$ 是参数为 $\boldsymbol{\theta}$ 的生成网络。

**Intuition**: 因为底层的 representation 就是 3D 空间里的点，所以拼接历史信息根本不需要像 video model 那样搞复杂的 temporal memory。直接把上一步的点云和这一步的条件点云在空间上拼起来就行，这种 spatial 演化方式天然适合无尽的场景扩展。

---

## 4. 实验数据说明了什么？

我们看 Table 2 的无条件生成数据：
- **Terra (Point)**: P-FID 8.79, P-KID 1.745%
- **Prometheus (RGBD)**: P-FID 32.35, P-KID 12.481%
- **Trellis (3D Grid)**: P-FID 19.62, P-KID 7.658%

在几何质量（P-FID, P-KID）上，Terra 把基于 2.5D 和基于 Voxel 的方法按在地上摩擦。因为 point latents 极其擅长描绘场景的几何分布。
但是在纹理质量（FID, KID）上，Terra 稍微逊色于 Prometheus（307.2 vs 263.3）。这是因为 Prometheus 背靠 Stable Diffusion 这种巨无霸级别的 2D 图像先验，而 Terra 是纯 native 3D 从零开始学的，没有看过海量互联网图片。这是一个极其坦诚的 trade-off。

---

## 5. 我的直觉与疯狂联想

1. **Representation 即归纳偏置**: Terra 用 1513 个 ScanNet 场景就训出了几何 SOTA 的模型，吊打用海量数据喂出来的 2.5D model。这再次证明，选对 representation（point latents + 3DGS）能省下几个数量级的算力。这与 LeCun 提倡的 JEPA 架构（V-JEPA 2）在精神上有点神似：在紧凑的 latent space 里做预测和演化，避免了生成高维像素带来的巨大浪费。只不过 Terra 的 latent space 是显式的 3D 空间。
2. **OT 配对是无序数据生成的胜负手**: Distance-aware Trajectory Smoothing 这个设计让我联想到 P2P-Bridge。对于 point cloud 这种无序集合，如果不做 Optimal Transport 意义上的配对，生成模型就是在做无用功。后续如果 scene scale 扩大到百万级点，可能需要用 Sinkhorn 算法的近似来实现快速 OT 配对。
3. **Spatial Exploration vs Temporal Prediction**: 我们一直把 world model 等同于 video prediction（如 Sora、Genie 3），陷入 temporal consistency 的泥潭。Terra 把探索定义为 spatial outpainting，瞬间避开了长视频漂移的问题。如果把这个思路用到 robotics 上，机器人的 memory 就是一堆不断拼接的 3D point latents，而不是一段越来越长的视频帧。
4. **2D Prior 的蒸馏**: Terra 最大的软肋是没有 2D diffusion 的纹理先验。我立刻联想到 DUSt3R / MASt3R 或者 ViewCrafter，它们巧妙地利用了 2D 模型。如果下一代 Terra 能把 ControlNet 里的 2D feature 通过 cross-attention 注入到 3DGS 的 rasterizer 渲染分支中，那 FID 指标绝对能直接拉平 Prometheus，实现几何与纹理的双重碾压。

相关参考链接：
- Terra 的核心基础 3D Gaussian Splatting: https://inria.github.io/gaussian-splatting/
- Point Transformer V3 架构: https://github.com/Pointcept/PointTransformerV3
- Flow Matching 原理: https://arxiv.org/abs/2210.02747
- 经典 Latent Point Diffusion (LION): https://nvidia-research-labs.github.io/lion/
- 视频世界模型 Genie 3: https://deepmind.google/discover/blog/genie-3/
- V-JEPA 2 (预测世界模型): https://ai.meta.com/blog/v-jepa-2/
- 2.5D 视角生成 ViewCrafter: https://viewcrafter.github.io/
- 3D 几何理解 DUSt3R: https://github.com/naver/dust3r

---

# Terra: Native 3D World Model 深度解读

Andrej，这篇论文非常对你最近关注的 world model / simulator 路线胃口。我尽量把它讲透，并且把相关的联想（LION、Trellis、Cosmos、V-JEPA 2、ViewCrafter、OccWorld 等）都串起来 build intuition。

---

## 1. 核心动机：为什么 pixel-aligned 表示是死胡同

World model 的本质问题在 representation。当前主流有三派：

**2D world model（Sora、Genie 3、Voyager）**：纯 video prediction，学一个 $p(x_{t+1} | x_{\le t})$。3D consistency 完全靠 training data 隐式习得，结果是 free-viewpoint 一旦跨视角大就崩。

**2.5D world model（Prometheus、Cosmos、Aether、ViewCrafter）**：联合生成 RGB $R$、depth $D$、pose $T$，希望三者满足 reprojection constraint。论文里 Eq.(3) 把它写得很清楚：

$$\boldsymbol{R}^{(n)}|_{\boldsymbol{x}^{(n)}} = \boldsymbol{R}^{(m)}|_{\boldsymbol{x}^{(m)}}, \quad d^{(n)}\boldsymbol{x}^{(n)} = \boldsymbol{T}^{(n)}\boldsymbol{x}, \quad n,m = 1, 2, \ldots, N$$

变量解释：
- $\boldsymbol{x}$：3D 空间中某个可见点的坐标
- $\boldsymbol{x}^{(n)}$：该点在 $n$ 视角图像上的 2D 像素坐标
- $d^{(n)}$：该点在 $n$ 视角的 depth
- $\boldsymbol{T}^{(n)}$：$n$ 视角的相机外参（投影矩阵）
- $\boldsymbol{R}^{(n)}|_{\boldsymbol{x}^{(n)}}$：在 $n$ 视角图像上 $\boldsymbol{x}^{(n)}$ 处采样得到的 RGB 值

这个公式物理上没问题，但作为 learning objective 极其糟糕：perspective projection 把 $R, D, T$ 三个量深度耦合，网络要 data-driven 地学会 "如果同一个 3D 点投影到两个视角，像素颜色要一致"——这是结构性 constraint，靠 implicit learning 永远学不干净，所以才有多视角漂移、depth flickering。

**Native 3D（Terra）**：直接放弃 Eq.(2)(3) 的形式，把 $S_i$ 写成 point latents $\boldsymbol{P}_i \in \mathbb{R}^{M_i \times (3+D)}$。3D consistency 不再是 constraint，而是 representation 本身的属性——任何 viewpoint 渲染都来自同一组 3D Gaussians，几何上自动一致。

> Intuition: pixel-aligned 是在 2D 层面拼 3D 拼图，native 3D 是直接在 3D 空间建模再投影。前者把 perspective transform 的耦合塞进 network，后者把 perspective transform 留给 rasterizer 这种确定性的、可微的算子。

参考链接：
- 3D Gaussian Splatting 原文: https://inria.github.io/gaussian-splatting/
- ViewCrafter: https://viewcrafter.github.io/
- Cosmos (NVIDIA): https://github.com/nvidia-cosmos/cosmos-world-foundation-model
- V-JEPA 2: https://ai.meta.com/blog/v-jepa-2/

---

## 2. 整体架构：两个核心模块

```
Input RGB Point Cloud Q  →  P2G-VAE Encoder  →  Point Latents P  →  SPFlow (flow matching)  →  Point Latents P'  →  P2G-VAE Decoder  →  3D Gaussians  →  Rasterize
                                                                                                        ↑
                                                                                          condition (cropping/uniform)
```

P2G-VAE 解决 "怎么把 1M 点压成 ~5K latents 且能解码回可渲染的 Gaussians"；SPFlow 解决 "怎么从噪声采样出符合 scene distribution 的 latents"。Explorable world model 就是 SPFlow 在 conditional setting 下做 outpainting。

---

## 3. P2G-VAE：把 point cloud 变成 Gaussian 的 VAE

### 3.1 Backbone 选择

基于 PTv3 (Point Transformer v3, Wu et al. 2024) 做主架构，移除 residual connections。原因我推测：VAE 的 encoder-decoder 是一个 funnel，residual 会把 high-frequency 几何细节 shortcut 进 bottleneck，破坏 latent space 的 compactness。

- Encoder：3 次 stride-2 downsampling，1M → 5K 点
- Decoder：3 次 upsampling，K = 7, 3, 3（即 5K → 35K → 105K → 315K 点）

PTv3 的核心是 serialization-based attention（基于 Hilbert / Morton curve 把 3D point 序列化后做 local window attention），比 PTv2 快几十倍，特别适合这种百万级点的场景。

参考链接:
- PTv3: https://github.com/Pointcept/PointTransformerV3

### 3.2 Robust Position Perturbation（关键创新）

传统 VAE 的 KL loss 把 latent feature 拉向 $\mathcal{N}(0, I)$，但 point latents 的 position $\hat{\boldsymbol{p}} \in \mathbb{R}^3$ 本身就是几何信息！把 position 也正则成高斯噪声会摧毁局部性（locality）——一个在墙上的点被 push 到半空。

Terra 的解法在 Eq.(4)：

$$\boldsymbol{P} = [(\boldsymbol{p}^{(m)} \in \mathbb{R}^3, \boldsymbol{f}^{(m)} \in \mathbb{R}^D)|_{m=1}^M], \quad \boldsymbol{p} = \hat{\boldsymbol{p}} + \boldsymbol{n}, \quad \boldsymbol{f} \sim \mathcal{N}(\text{mean}(\hat{\boldsymbol{f}}), \text{diag}(\text{var}(\hat{\boldsymbol{f}})))$$

变量解释：
- $\boldsymbol{P}$：point latents 集合
- $\boldsymbol{p}^{(m)}, \boldsymbol{f}^{(m)}$：第 $m$ 个 latent 的 position 和 feature
- $\hat{\boldsymbol{p}}, \hat{\boldsymbol{f}}$：VAE bottleneck 入口处的原始 position / feature
- $\boldsymbol{n} \sim \mathcal{N}(\boldsymbol{0}, \sigma^2 \boldsymbol{I}_3)$：预定义的高斯噪声
- $\sigma$：噪声强度超参

**Intuition**：position 不做 KL regularization，只加小幅 Gaussian perturbation（数据增广意义上的 perturbation）；feature 仍然做 KL，把 $\boldsymbol{f}$ 拉向 $\mathcal{N}(\text{mean}, \text{diag}(\text{var}))$。这样 latent space 既平滑（feature 端），又保留几何 locality（position 端）。

更进一步，这种 perturbation 让 decoder 训练时见过 "稍微抖动的 position"，对 generative model 后期输出的 noisy position 更鲁棒——这一点在 ablation 里被验证：去掉 perturbation 后 P-FID 从 8.79 飙到 15.28，几乎翻倍，但 reconstruction PSNR 反而提升（20.487 vs 19.742）。这是一个典型的 reconstruction-generation trade-off：让 VAE "难一点"，generation 才 "稳一点"。和 VQ-VAE 用 EMA codebook、Stable Diffusion 用 KL 系数调小是同一种设计哲学。

### 3.3 Adaptive Upsampling and Refinement

Decoder 要从 5K 点恢复到 315K Gaussians，传统做法是 KNN duplication + mask trimming（XCube、Can3Tok 那套）。Terra 改成 query-based upsampling，Eq.(5)：

$$\hat{\boldsymbol{q}}^{(k)}|_{k=1}^K = \text{ups}(\boldsymbol{f}, \boldsymbol{q}^{(k)}|_{k=1}^K), \quad \boldsymbol{p}^{(k)} = \boldsymbol{p} + \text{disp}(\hat{\boldsymbol{q}}^{(k)}), \quad \boldsymbol{f}^{(k)} = \boldsymbol{f} + \text{resf}(\hat{\boldsymbol{q}}^{(k)})$$

变量解释：
- $\boldsymbol{q}^{(k)}|_{k=1}^K$：$K$ 个 learnable query（类似 DETR 的 object query）
- $\text{ups}(\cdot)$：point-query 交互模块（cross-attention）
- $\text{disp}(\cdot)$：预测子点相对父点的 displacement
- $\text{resf}(\cdot)$：预测子点相对父点的 feature residual

每个父点 $(\boldsymbol{p}, \boldsymbol{f})$ 被拆成 $K$ 个子点，子点位置 = 父点 + displacement，子点 feature = 父点 + residual。这本质上是把 "如何 densify" 学进 network 而不是 rule-based——比如墙面的点该往法向方向 spread，地毯的点该在平面内填充，这些策略从 data 学。

Refinement module 进一步：$\boldsymbol{p}' = \boldsymbol{p} + \text{refine}(\boldsymbol{f})$，用 feature 再修正一次 position。两次坐标修正（upsample + refine）类似 coarse-to-fine refinement，类似 P2P-Bridge 的扩散 bridge 思路，但这里是在 VAE decoder 内部一次前向完成。

### 3.4 Comprehensive Regularizations（Loss 设计）

Eq.(6) 是个 cocktail：

$$L_{vae} = L_{l2} + \lambda_1 L_{ssim} + \lambda_2 L_{lpips} + \lambda_3 L_{cham} + \lambda_4 L_{norm} + \lambda_5 L_{rank} + \lambda_6 L_{color} + \lambda_7 L_{kl}$$

每项的 intuition：
- $L_{l2}, L_{ssim}, L_{lpips}$：标准 differentiable rendering loss（来自 3DGS 训练）
- $L_{cham}$：Chamfer distance，between input point cloud 和 intermediate upsample/refine 输出，提供 explicit geometry supervision，避免位移 prediction 漂掉
- $L_{norm}$：Gaussian 法向正则，避免 splat 乱转
- $L_{rank}$：effective rank regularization（Hyung et al. 2024），限制 3D Gaussian 的 scale rotation matrix 的秩，防止变成 needle-like 退化
- $L_{color}$：**这是论文亮点**，把每个 Gaussian 的颜色直接和 input point cloud 中 nearest point 的颜色对齐，bypass rasterization。为什么友好？因为 rasterize 是 non-differentiable 的 picking 算子 + differentiable 的 alpha blending，前者靠 forward+backward surrogate 走，梯度质量本身就一般；直接 NN match 颜色是 L2 级 clean gradient。
- $L_{kl}$：feature 的 KL，position 不参与

Ablation 显示 $L_{color}$ 去掉后 PSNR 掉 0.16、P-FID 掉 1.82，是单项最显著的正则之一。

参考链接:
- 3DGS effective rank regularization: https://arxiv.org/abs/2406.12272
- P2P-Bridge: https://arxiv.org/abs/2408.15065

---

## 4. SPFlow：在 point latent 空间做 flow matching

### 4.1 为什么用 flow matching 而不是 diffusion

Flow matching (Lipman et al. 2022) 是 diffusion 的"直化"版本：直接学一个 vector field 把 noise distribution $\mathcal{N}(0, I)$ 传输到 data distribution $\mathcal{P}$，轨迹是 straight line（在 optimal transport 意义下）。优点：训练 objective 简单、sampling 步数少、ODE 求解器比 SDE 稳。

Eq.(7)(8) 是标准 rectified flow 公式：

$$\boldsymbol{P}_t = t\boldsymbol{P} + (1-t)\boldsymbol{N}, \quad \boldsymbol{V} = \mathcal{F}(\boldsymbol{P}_t, t; \phi)$$

$$L_{flow} = \mathbb{E}_{t \sim \mathcal{U}[0,1], \boldsymbol{P} \sim \mathcal{P}, \boldsymbol{N} \sim \mathcal{N}(\boldsymbol{0}, I)} \| \mathcal{F}(\boldsymbol{P}_t, t; \phi) - (\boldsymbol{P} - \boldsymbol{N}) \|^2$$

变量解释：
- $\boldsymbol{P} \in \mathbb{R}^{M \times (3+D)}$：clean point latents（含 position + feature）
- $\boldsymbol{N} \in \mathbb{R}^{M \times (3+D)}$：噪声
- $t \in [0,1]$：time schedule，$t=0$ 全噪声，$t=1$ 全数据
- $\boldsymbol{V}$：velocity vector（flow 的速度场）
- $\mathcal{F}(\cdot, \cdot; \phi)$：3D sparse convolution UNet（OA-CNNs backbone）
- $\boldsymbol{P} - \boldsymbol{N}$：target velocity（从 noise 指向 data 的直线方向）

关键：**position 和 feature 联合加噪、联合 denoise**。这点 LION（Vahdat et al. 2022）是分开做两个 diffusion 的，Terra 是 concat 起来一起走。论文的 motivation 是 "geometry 和 texture 互补，相互 enhance"——这个 intuition 我个人觉得对，但论文没给 ablation 证明 "联合 vs 分开" 的差距。

参考链接:
- Flow Matching: https://arxiv.org/abs/2210.02747
- LION: https://nvidia-research-labs.github.io/lion/
- OA-CNNs: https://arxiv.org/abs/2405.12921

### 4.2 Distance-aware Trajectory Smoothing（另一个关键创新）

这是整篇论文最优雅的 trick。在 grid-based latent diffusion 里，noise sample 和 data sample 按 grid index 一一对应即可——pixel (i,j) 的噪声对应 pixel (i,j) 的 data，transport 距离短、trajectory 直。

但 point latents 是无序的、unstructured 的集合！如果按 sequence index 1-to-1 配对，可能左边墙上点的 target 是右边地板上的噪声，velocity vector 跨越整个房间，ODE 轨迹极度弯曲、收敛慢、生成质量差。

Eq.(9) 把这个写成 linear assignment problem (LAP)：

$$\mathcal{M}^* = \arg\min_{\mathcal{M}} \sum_{m=1}^M \| \boldsymbol{p}^{(m)} - \boldsymbol{N}_{\mathcal{M}_m, :3} \|^2, \quad \mathcal{M} = \text{reorder}([1, 2, \ldots, M])$$

变量解释：
- $\mathcal{M}$：一个 permutation，表示 noise sample 和 point latent 的配对方式
- $\boldsymbol{p}^{(m)}$：第 $m$ 个 point latent 的 position
- $\boldsymbol{N}_{\mathcal{M}_m, :3}$：被分配给第 $m$ 个 point latent 的 noise sample 的前 3 维（position 部分）
- $\mathcal{M}^*$：最优 permutation，最小化所有 point-noise pair 的距离平方和

求解用 Jonker-Volgenant algorithm（1987，经典 LAP 求解器，$O(M^3)$ 但常数小）。这个操作只在 training 的数据准备阶段做一次配对，不在每一步 training 内做——一旦 fixed 配对，每次取 batch 时按这个 permutation 重新 order noise 即可。

Ablation 数据极其漂亮：去掉 trajectory smoothing，P-FID 从 8.79 暴涨到 24.84，P-KID 从 1.745% 到 11.387%。基本是 magnitude 级别的差距。这是 "small trick, huge impact" 的典范。

> Intuition: 把 optimal transport 的"分配问题"前置到 data preparation，让 flow matching 拟合的 vector field 在空间上 short-range、smooth、可插值。这和 rectified flow 的 "cut corners" 思想、以及 P2P-Bridge 用 diffusion bridge 在 point cloud 之间学 short path 是一脉相承的。

参考链接:
- P2P-Bridge: https://arxiv.org/abs/2408.15065
- Not-so-optimal transport flows (Hui et al. 2025): https://arxiv.org/abs/2502.12456
- Jonker-Volgenant 原文: https://link.springer.com/article/10.1007/BF02278710

### 4.3 Simple Conditioning Mechanism

为了支持 explorable world model，Terra 设计了三种 mask 条件：
1. **Cropping**：随机切一个 connected 3D 区域作为 condition，让模型想象 unknown region（类似 image outpainting）
2. **Uniform sampling**：在整个 scene 上稀疏采样若干 latents 作为 condition，让模型 refine known region（类似 image inpainting / SR）
3. **Combination**：先 crop 再 uniform sample，模拟 RGBD 局部观测

注入方式极其 simple：把 conditional latents 和 noisy latents 在 sequence 维度 concat，diffusion 过程中 condition 固定不动。和 InstructPix2Pix / ControlNet 把 condition 走 cross-attention 比起来，这种 "concat + freeze" 更像 LLM 的 in-context learning，简单粗暴但有效。

三阶段训练：
- Stage 1: Reconstruction（训 P2G-VAE，36K iter）
- Stage 2: Unconditional generative pretrain（训 SPFlow，100K iter）
- Stage 3: Masked conditional generation（fine-tune SPFlow，40K iter）

Stage 2 → Stage 3 是经典 pretrain-then-finetune 范式，Stage 2 学 scene 分布的 prior，Stage 3 学 "给定 partial context 怎么补全"。

---

## 5. Explorable World Model 的形式化

Eq.(1) 给出了 progressive exploration 的递归定义：

$$S_0 = g(\emptyset, C_0; \boldsymbol{\theta}), \quad S_i = g(\mathbb{S}_{i-1}, C_i; \boldsymbol{\theta}), \quad \mathbb{S}_i = \{S_0, S_1, \ldots, S_i\}$$

变量解释：
- $S_i$：第 $i$ 步探索生成的 scene chunk（一组 point latents）
- $\mathbb{S}_{i-1}$：截至 $i-1$ 步所有已知 chunk 的集合
- $C_i$：第 $i$ 步的 conditional signal（探索方向、目标区域等）
- $g(\cdot, \cdot; \boldsymbol{\theta})$：参数为 $\boldsymbol{\theta}$ 的生成模型
- $S_0 = g(\emptyset, C_0; \boldsymbol{\theta})$：初始 scene，从空条件生成

具体操作：
1. 用 SPFlow 生成第一个 chunk $S_0$（unconditional 或 text-conditioned，论文没明确说有 text）
2. 选一个 exploration direction，把 $S_0$ 朝该方向 crop 出 connected region 作为 condition $C_1$
3. SPFlow 生成 $S_1$，concat 到 $\mathbb{S}_0$
4. 重复

因为 point latents 是 spatial sparse 的，concat 历史 chunk 就是直接 position-feature 拼接，不需要像 video frame 那样维护 temporal memory。这和 Genie 3 / V-JEPA 2 在 video latent 空间做 autoregressive 是同样的精神，但 representation 不一样。

参考链接:
- Genie 3: https://deepmind.google/discover/blog/genie-3/
- OccWorld (3D occupancy world model): https://github.com/wzzheng/OccWorld

---

## 6. 实验数据解读

### 6.1 Reconstruction（Table 1）

| Method | Input | PSNR↑ | SSIM↑ | LPIPS↓ | Abs.Rel↓ | RMSE↓ | δ1↑ |
|---|---|---|---|---|---|---|---|
| PixelSplat | RGB | 18.165 | 0.686 | 0.493 | 0.094 | 0.287 | 0.832 |
| MVSplat | RGB | 17.126 | 0.621 | 0.552 | 0.139 | 0.326 | 0.824 |
| Prometheus | RGBD | 17.279 | 0.644 | **0.448** | 0.087 | 0.251 | 0.901 |
| Can3Tok* | Gaussians | 19.578 | 0.733 | 0.514 | 0.031 | 0.151 | 0.973 |
| **Terra** | RGB PC | **19.742** | **0.753** | 0.530 | **0.026** | **0.137** | **0.978** |

几个关键解读：
- Terra 输入只是 colored point cloud（unproject RGBD），但 PSNR 超过 Can3Tok（用 offline 重构的高质量 Gaussians 输入）。这说明 P2G-VAE 本身就是极强的 scene reconstructor。
- LPIPS 输给 Prometheus，因为 Prometheus 有 2D diffusion pretrain（Stable Diffusion 级别的 image prior），perceptual quality 强；Terra 完全 native 3D 训练，没有这种 prior。这是个 systemic gap，作者诚实承认。
- Depth metrics (Abs.Rel, RMSE, δ1) 大幅领先，说明 Terra 的几何质量是真的好，符合 native 3D 的预期。

### 6.2 Unconditional Generation（Table 2）

| Method | Repr. | P-FID↓ | P-KID%↓ | FID↓ | KID%↓ |
|---|---|---|---|---|---|
| Prometheus | RGBD | 32.35 | 12.481 | **263.3** | **10.726** |
| Trellis | 3D Grid | 19.62 | 7.658 | 361.4 | 23.748 |
| **Terra** | Point | **8.79** | **1.745** | 307.2 | 18.919 |

- **Geometry（P-FID, P-KID）**：Terra 是 Trellis 的 2 倍好，是 Prometheus 的 3-7 倍好。说明 point latents 学 scene 几何分布的能力远超 voxel grid 和 RGBD pixel-aligned。
- **Texture（FID, KID）**：Terra 输 Prometheus，输 2D diffusion prior。
- Trellis 是 Xiang et al. 2025 的 structured voxel latents，做 object generation 很好，但放到 scene 级别 voxel 分辨率不够、纹理也差。这佐证了 Terra 论文 title 里 "native 3D + explorable + scene-level + rendering-compatible" 同时打满的稀缺性。

### 6.3 Image-Conditioned Generation（Table 2 右半）

| Method | CD↓ | EMD↓ | FID↓ | KID%↓ |
|---|---|---|---|---|
| Prometheus | 0.374 | 0.531 | **208.3** | **12.387** |
| Trellis | 0.405 | 0.589 | 314.9 | 24.713 |
| **Terra** | **0.217** | **0.474** | 262.4 | 20.283 |

CD (Chamfer Distance) 和 EMD (Earth Mover's Distance) 是 point cloud 之间距离的标准 metric。Terra 的 CD 是 Prometheus 的 58%，几何一致性碾压。

### 6.4 Ablation（Table 3）

最重要的两个发现：
1. **Robust Position Perturbation**：reconstruction PSNR 20.487（更好），但 P-FID 15.28（差近 2 倍）。典型 recon-gen trade-off，perturbation 是为 generation 服务的。
2. **Distance-aware Trajectory Smoothing**：去掉后 P-FID 24.84（差 3 倍），P-KID 11.387%（差 7 倍）。这是整个 generative training 收敛的关键。

### 6.5 数据规模

- ScanNet v2：1513 scenes，958 train / 243 val
- 训练 P2G-VAE 36K iter，SPFlow 100K + 40K iter
- 输入 crop 大小 2.4×2.4 m²
- Single forward pass 即可渲染整个 scene，vs ViewCrafter 多步迭代

数据量其实很小（958 scenes），但效果已经 SOTA，说明 point latent representation 的 sample efficiency 高。这也呼应了 Karpathy 你自己常说的 "representation 决定了 data efficiency"。

---

## 7. 相关联想 & 与其他路线对比

### 7.1 vs LION (Vahdat 2022)
LION 是 latent point diffusion 的开山之作，但只做 object-level shape generation（无 texture），用两个独立 diffusion（shape latent + appearance latent）。Terra 是 scene-level、joint diffusion、output 是可渲染 3DGS。LION 的思想被 Terra 继承并大幅扩展。

### 7.2 vs Trellis (Xiang 2025)
Trellis 用 structured voxel latents（稀疏 3D grid），object-level 优秀，scene-level 受限于 voxel 分辨率。Terra 用 point latents，分辨率随 surface adaptive，scene 友好。

### 7.3 vs OccWorld (Zheng 2024a)
OccWorld 也是 3D occupancy world model，做自动驾驶场景的 4D prediction，但 occupancy 是 coarse voxel grid，不可渲染，不能做 photorealistic 输出。Terra 走的是 fine-grained + rendering-compatible 路线。

### 7.4 vs V-JEPA 2 (LeCun / Meta 2025)
V-JEPA 2 在 video latent space 做 joint-embedding predictive architecture，是 non-generative、predictive world model。Terra 是 generative、explicit 3D 的 world model。两者代表了 LeCun 的 JEPA 路线和 generative model 路线的分野。Terra 的 point latents 实际上很接近 JEPA 哲学里的 "non-generative latent predictive space"，只不过 Terra 在 latent space 仍然用 flow matching 做 generative sampling。

### 7.5 vs Cosmos (NVIDIA 2025)
Cosmos 是 NVIDIA 的 world foundation model，2.5D RGBD 路线，scale 巨大（10M+ hours video）。Terra 只在 ScanNet 1513 scenes 上训练就 SOTA，说明 native 3D 的 inductive bias 在 small data regime 下碾压靠 scale 撑起来的 2.5D。

### 7.6 vs Genie 3 (DeepMind 2024)
Genie 3 是 video world model，real-time interactive，photorealism 强。但它的 3D consistency 来自 latent dynamics model 的隐式学习，不保证。Terra 用 representation 强制 3D consistency，代价是不实时、scale 小。

### 7.7 vs Wonderworld (Yu 2025)
Wonderworld 从 single image 生成 3D scene，是 image-conditioned 3D generation 但单步。Terra 是 progressive exploration，多步 outpainting，可以无限扩展。

### 7.8 vs GaussianAnything (Lan 2025)
GaussianAnything 也是 point cloud latent diffusion + 3DGS，但 object-level。Terra 是 scene-level + explorable。两者在 VAE 设计上有相似思路。

参考链接:
- Trellis: https://trellis3d.github.io/
- OccWorld: https://github.com/wzzheng/OccWorld
- Wonderworld: https://wonderworld-2024.github.io/
- GaussianAnything: https://llng.net/GaussianAnything/

---

## 8. 局限性 & 我的思考

论文没明确写 limitations section（很可惜），从 paper 内容能推断出几个：

1. **Texture quality vs 2D diffusion prior**：FID 输给 Prometheus 是 systemic 的，因为没借力 Stable Diffusion 这类 2D prior。一个直接的 fix 是把 2D diffusion 的 feature 注入 SPFlow decoder，类似 ControlNet 注入 3DGS 渲染的 image。
2. **Scale**：只在 ScanNet 1513 scenes 上训，泛化到 outdoor / unbounded scene 未知。ScanNet 是室内、texture 相对简单。能不能 scale 到 Objaverse-XL 级 + HyperSim + KITTI 还需要后续工作。
3. **Dynamic world**：当前 Terra 是 static scene world model，没有 time dimension。Explorable 指 spatial exploration 不是 temporal evolution。要做真正 4D world model 还要在 point latents 上加 time index，类似 XCube 的 sparse voxel hierarchy 加 time。
4. **Point latent 数量自适应**：$M_i$ 根据复杂度变化，但论文没说怎么动态决定 $M_i$。可能现在是 heuristic crop，未来需要 learnable point budget。
5. **LAP 求解成本**：Jonker-Volgenant 是 $O(M^3)$，$M \approx 5000$ 时单次 ~125e9 ops，预处理慢但一次性。如果 scene scale 到 1M latents，得换成 auction algorithm 或 Sinkhorn 的近似 OT。
6. **No text conditioning**：当前是 unconditional + image conditional，没有 text-to-scene。这是走向 consumer-grade tool 的必经之路。
7. **2D diffusion prior integration**：能不能 distill SD/FLUX 的 prior 进 native 3D latent？这是把 FID 拉平 Prometheus 的关键。
8. **Long-horizon exploration drift**：progressive generation 多步后，early chunks 和 late chunks 之间是否 drift？类似 video generation 的 long-range temporal drift。论文 Figure 7 显示 5 步还 coherent，但 50 步呢？100 步呢？这是个 critical scaling question。

---

## 9. 总结：Terra 给我的几个核心 intuition

1. **Representation 决定 consistency**：3D consistency 不是 loss 约束出来的，是 representation 内秉的。Point latents + 3DGS rasterizer 让 consistency 成为 free property。
2. **VAE 设计要为 generation 服务，不只是 reconstruction**：robust position perturbation 主动降低 reconstruction 性能来换 generation 鲁棒性，这种 recon-gen trade-off 在 latent generative model 里反复出现（VQ-VAE 的 EMA、SD 的 KL 系数、tokenizer 的 codebook size）。
3. **Optimal transport 配对在 unstructured data 上极重要**：grid-based data 自带 spatial index 做 OT 配对，point cloud 没有，必须显式求解 LAP。这个 trick 应该会成为后续 point latent generative model 的标配。
4. **Scene-level native 3D generation 是可行的**：之前 native 3D generative 一直困在 object-level（LION、Trellis、GaussianAnything），Terra 证明 scene-level 也能 work，关键是 VAE 的 downsample-upsample 设计要适合 scene scale。
5. **Progressive outpainting 是 spatial world model 的合理形式**：把 world evolution 重新定义为 spatial 而非 temporal，避免了 video prediction 的 temporal consistency 难题。这个 framing 我觉得很适合 robotics / embodied AI 的场景重建。
6. **Small data + strong inductive bias 能赢 big data + weak bias**：958 scenes 训出的 native 3D 模型在 geometry 上击败 Cosmos / Prometheus 这类大数据 2.5D 模型。representation engineering 仍是高 ROI 方向。

---

## Reference Links 汇总

- **Terra paper**: (论文本身，arXiv 待发)
- **3D Gaussian Splatting**: https://inria.github.io/gaussian-splatting/
- **PTv3**: https://github.com/Pointcept/PointTransformerV3
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **LION**: https://nvidia-research-labs.github.io/lion/
- **Trellis**: https://trellis3d.github.io/
- **Cosmos**: https://github.com/nvidia-cosmos/cosmos-world-foundation-model
- **V-JEPA 2**: https://ai.meta.com/blog/v-jepa-2/
- **Genie 3**: https://deepmind.google/discover/blog/genie-3/
- **ViewCrafter**: https://viewcrafter.github.io/
- **OccWorld**: https://github.com/wzzheng/OccWorld
- **P2P-Bridge**: https://arxiv.org/abs/2408.15065
- **GaussianAnything**: https://llng.net/GaussianAnything/
- **Wonderworld**: https://wonderworld-2024.github.io/
- **ScanNet v2**: http://www.scan-net.org/
- **OA-CNNs**: https://arxiv.org/abs/2405.12921
- **Not-so-optimal transport flows**: https://arxiv.org/abs/2502.12456
- **Jonker-Volgenant LAP**: https://link.springer.com/article/10.1007/BF02278710
- **Effective rank reg for 3DGS**: https://arxiv.org/abs/2406.12272

希望这个讲解对你的 intuition building 有帮助。如果你想再深挖某一块（比如 flow matching 在 unstructured data 上的 OT 配对 vs Sinkhorn 的对比，或者 P2G-VAE decoder 的 query upsampling 和 DETR object query 的关系），我可以再展开。
