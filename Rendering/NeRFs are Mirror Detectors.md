---
source_pdf: NeRFs are Mirror Detectors.pdf
paper_sha256: 88f70a0388e58c025eb44f3c425bdeb41e096a841310468803176b33973f0b13
processed_at: '2026-08-05T22:12:06-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# NeRFs are Mirror Detectors — 人话版本

## 一句话版本

NeRF 看到镜子就会 "翻车"，翻车的地方恰好就是镜子，作者把这个翻车信号捡起来当镜子探测器，再反过来用显式的 3D primitive 把镜面建出来，一路端到端训下去，最后不用任何人工标注的 mirror mask 就能重建镜面反射。

---

## 故事的开头：NeRF 怎么被镜子搞糊涂的

想象你拍一个房间，房间里挂着一面镜子。你拍 100 张不同角度的照片，喂给 NeRF，让它学一个 3D scene。NeRF 的核心假设是：**空间中每个点 $x$ 的颜色 $c$ 只和 position $x$ 以及 view direction $d$ 有关**——也就是 $c(x, d)$ 是个 well-defined 的 5D function。

这个假设对 diffuse surface 完美成立：墙上一点从任何角度看都是那个颜色，view-dependent 的变化只是高光那种细节。

但镜子打破这个假设。镜面点 $x_m$ 看到的 "颜色"，其实是 **另一个场景点 $y$ 沿反射方向传过来的光**。当你移动相机，镜子里同一 pixel 对应的 $y$ 完全变了——你从左边照镜子看到左边的房间，从右边照看到右边的房间。镜面点的 "颜色" 实际上 depends on 整个 scene 的 radiance field，远不止 $x_m$ 自己。

NeRF 不知道这回事，它把镜子当成 **一扇窗户**——以为镜后真的有一个 scene 在那里。于是它就在镜子背后那块空荡荡的空间里硬生生"捏"出一个 virtual scene 来 fit 所有视角的 reflection。问题是：

- 不同视角的 reflection 不一致，NeRF 只能把它们"平均"掉
- 平均的结果就是镜面区域 render 出来糊糊的，不 sharp
- 镜面的 geometry 也跑偏了——depth 跑到镜子背后的虚拟空间去了
- 整个 scene representation 是 inconsistent 的

这就是 paper Section 1 说的 "NeRF interprets the mirror's reflection as an independent virtual scene, providing the illusion of viewing the respective content through a window"。

参考：原 NeRF paper https://arxiv.org/abs/2003.08934

---

## 核心洞察：NeRF 的 "翻车" 本身就是镜子信号

作者发现一个特别简洁的事实：**标准 NeRF 在镜面区域一定会留下 photometric artifact**——render 出来的颜色和 ground truth 对不上、blurry、structure 不 sharp。而 diffuse 区域 NeRF 已经收敛得很好。

这个观察听起来平凡，但细细想其实很 powerful：

- 如果一个区域 NeRF 学得好，说明这个区域满足 view-consistency 假设，那它不是镜子
- 如果一个区域 NeRF 学得烂，说明这个区域违反 view-consistency，那它要么是镜子、要么是 NeRF 整体没收敛好的区域（比如 textureless）

所以现在的问题变成：**怎么把 "镜子的烂" 和 "其他原因的烂" 区分开？**

这就是 paper Section 4.1 的核心。作者用两个信号组合：

### 信号一：SSIM (Structural Similarity)

$$\mathrm{SSIM}(r) = \frac{(2\mu\mu^* + c_1)(2\tilde{\sigma} + c_2)}{(\mu^2 + (\mu^*)^2 + c_1)(\sigma^2 + (\sigma^*)^2 + c_2)}$$

变量解释（这次讲人话）：
- $\mu$ 是 rendered image 在 pixel $r$ 周围一个小 window 的平均亮度
- $\mu^*$ 是 ground truth image 同位置的 window 平均亮度
- $\sigma$、$\sigma^*$ 是各自的方差（local contrast）
- $\tilde{\sigma}$ 是两幅图 local window 的 covariance（结构相似性）
- $c_1$、$c_2$ 是防止除零的小常数

SSIM 范围 $[-1, 1]$，1 表示完全一样。作者用 $1 - \mathrm{SSIM}$ 翻成 "dissimilarity"，高分 = "结构对不上"。

为什么不用 MSE？因为 MSE 抓不到 "structure loss"——NeRF 在镜面区域不是颜色错了，而是 **把多个视角的不同 reflection 平均成一个糊糊的低频 texture**。这种 "blurriness" 在 MSE 上可能差距不大，但在 SSIM 上很显眼（因为 local contrast 被抹平了）。

参考 SSIM：https://www.cns.nyu.edu/~lcv/ssim/

### 信号二：Depth Variance

$$V(r) = \sum_{k=1}^{K} T_k \alpha_k (t_k - D(r))^2$$

变量：
- $T_k$ 是 ray 走到 $t_k$ 之前的 transmittance（"还剩多少光没被吸收"）
- $\alpha_k$ 是 $t_k$ 处的 absorption probability
- $t_k$ 是第 $k$ 个 sample point 的 depth
- $D(r)$ 是 expected depth

$V(r)$ 就是 "absorption position 沿 ray 的方差"。**方差大 = NeRF 对这个 pixel 的 depth 没把握，probability mass 沿 ray 摊得很开**。方差小 = depth 锐利确定。

为什么这个 crucial？因为 **textureless 区域也会有 high SSIM error**——NeRF 在大白墙上也学不好。但大白墙的 $V(r)$ 大（depth 不确定），而镜面区域的 $V(r)$ 小（NeRF 学到了镜面表面的一个 sharp depth，只是 appearance 学不好）。

### 两个信号组合：score

$$s(r) = \frac{1 - \mathrm{SSIM}(r)}{2} \, e^{-cV(r)}$$

变量：
- $\frac{1 - \mathrm{SSIM}(r)}{2}$ 把 dissimilarity 归一化到 $[0,1]$
- $e^{-cV(r)}$ 是个 exponential decay weighting，$V(r)$ 大时整个 score 被压低
- $c \in \mathbb{R}$ 是控制衰减速度的 hyperparameter

直觉：**高 score = 结构对不上 AND depth 很确定**。

这个设计妙就妙在 $V(r)$ 充当 "否决票"——它把 "NeRF 整体没学好"（depth 也糊）的区域排除了，只留下 "NeRF 在这一块 geometry 上有把握、但 appearance 死活对不上" 的区域。后者几乎只能是镜子（或类似的高反射 surface）。

这一步对应 paper Figure 2：你能看到 depth variance 图、SSIM 图、最终 score 图——score 高的 region 恰好就是镜面。

---

## 从 score 到 3D primitive：把镜子找出来

现在每个 pixel 有一个 score，threshold 一下得到一组 high-score pixels。每个 pixel 对应一条 ray，ray 有 expected depth $D(r)$，于是可以 unproject 成 3D point：

$$x_{\text{mirror}} = o + D(r) \cdot d$$

变量：
- $o$ 是 camera origin
- $d$ 是 ray direction
- $D(r)$ 是该 ray 的 expected absorption depth

所有 high-score pixels unproject 完，得到一个 point cloud。这个 point cloud 大致落在镜面 surface 上。

### 法向量估计

每个 point 用 hybrid radius + kNN 取邻居（应对 point cloud 密度不均），然后做 PCA，最小特征值对应的 eigenvector 就是 normal 方向。但 normal 有朝向歧义（朝里还是朝外），作者用 "该 point 来自的 camera position" 解决——normal 应该朝向 camera 那一边。

### Primitive 拟合

作者假设镜面是 plane 或 cylinder 这种简单 primitive。步骤：

1. **k-means 聚类**：把 point cloud 切成 $k$ 个 cluster（$k$ 是预设镜子数量）
2. **RANSAC 拟合**：每个 cluster 用 RANSAC 拟合预设类型的 primitive（先拟 unbounded shape，比如 infinite plane；再算 oriented bounding box 限定范围）
3. **质量评估**：用三个 metric 判断 fit 好不好：
   - Inlier ratio（多少点落在 surface 上）
   - 平均到 unbounded shape 的距离
   - 预测 normal 和 point cloud 估计 normal 的相似度

每个 primitive 用一组参数 $\theta$ 描述——plane 就是法向量 $n$ + offset $b$ + bounding box size；cylinder 是 axis 方向 + radius + height。

参考 RANSAC：https://en.wikipedia.org/wiki/Random_sample_consensus

---

## 第二阶段训练：让 NeRF 知道镜子是镜子

现在有了一组 primitive 当 mirror geometry 的初始猜测。接下来要训一个 reflection-aware 的 NeRF，并且让 primitive 参数 $\theta$ 也能被优化。

### Reflection-aware rendering

每个 camera ray march 进 volume，递归检查是否 intersect 任一 mirror primitive。如果命中：

$$d_{\text{refl}} = d - 2(d \cdot n)n$$

变量：
- $d$ 是入射方向
- $n$ 是镜面 normal
- $d_{\text{refl}}$ 是反射方向

ray 在 surface 点 reflection 之后继续 march，直到 bounce limit 停。最终 pixel 的 color 是反射路径上累积的 radiance。

这一步基于 TraM-NeRF 的工作：https://arxiv.org/abs/2404.17370

### 微分 mirror 参数：绕开 volume rendering

直接把 volume rendering 的 gradient 传到 $\theta$ 上不稳定——path tracing 的 gradient 噪声大，因为 sampling 是随机的、ray-march 的 alpha compositing 对 surface 参数的依赖是 indirect 的。

作者用一个 trick：**渲染一个 differentiable antialiased mask**。对每个 pixel frustum（不是一条 ray，而是一个 cone），算它和 mirror surface 的相交面积比例 $m \in [0,1]$。然后：

$$C_{\text{pixel}} = (1 - m) \cdot C_{\text{primary}} + m \cdot C_{\text{reflected}}$$

变量：
- $C_{\text{primary}}$ 是 ray 没 mirror 时的 normal NeRF render
- $C_{\text{reflected}}$ 是反射 ray 的 render
- $m$ 是 frustum 和 mirror 相交的面积比例

$m$ 用 differentiable rasterizer（nvdiffrast, Laine et al. 2020）算，对 $\theta$ 是可微的。这样 image loss $\|C_{\text{pixel}} - C^*\|$ 的 gradient 通过 $m$ 直接传到 $\theta$，绕开了 volume rendering 那条 noisy path。

参考 nvdiffrast：https://github.com/NVlabs/nvdiffrast

### p-norm schedule：处理 sharp edge vs smooth color 的 tradeoff

这是 paper 里我个人觉得最有 general 价值的工程 trick。损失函数：

$$\mathcal{L}_I' = \frac{1}{|R|} \sum_{r \in R} \| C(r) - C^*(r) \|_{p(\tau)}^{p(\tau)}$$

变量：
- $C(r)$、$C^*(r)$ 是 rendered 和 ground truth color
- $p(\tau)$ 是 iteration $\tau$ 的 piecewise linear function，从 2 → 1 → 2

具体 schedule：

| 阶段 | iteration 范围 | $p$ | 目的 |
|---|---|---|---|
| 初始 | $\tau < \tau_{\text{init}}$ | 2 (MSE) | 先让 NeRF 整体收敛到一个 reasonable baseline |
| 中段 | $\tau_{\text{init}} < \tau < \tau_{\text{inc}}$ | 1 (L1) | 鼓励 sharp edges，让 mirror boundary 锐利 |
| 末段 | $\tau_{\text{inc}} < \tau < \tau_{\text{std}}$ | 1 → 2 线性 | 去掉 L1 造成的 piecewise-constant color plateau |

**为什么 L1 鼓励 sharp edges？**

L1 norm 是 sparsity-inducing。对一组 residual，L1 倾向于把大部分 residual 推到 0、保留少数大 residual；L2 倾向于所有 residual 均摊小。映射到 image：L2 优化出一个 "blurry averaging"（多个候选 appearance 取均值），L1 优化出一个 "sharp selection"（保留一个 dominant appearance）。

镜面边界就是这样——L2 会让 mirror edge 周围几个 pixel 是 "half mirror, half wall" 的 blend，糊掉；L1 会强迫 pixel 们 "选边站"，要么是 mirror 要么是 wall，边界 sharp。

**为什么 L1 又会卡在 local constant color？**

L1 的 gradient 在 residual → 0 时是 $\text{sign}(\text{residual})$，常数 magnitude 1，不像 L2 的 $2 \cdot \text{residual}$ 会随 residual 变小而变小。所以当大部分 pixel 已经对上、剩小区域需要精细化时，gradient 信号不够局部化，优化会卡在 piecewise constant 的解——mirror 内部出现 flat color block。

所以最后切回 L2 把 plateau 平滑掉。这个 schedule 在 paper Figure 3 有可视化，非常直观。

参考 L1 vs L2 in image reconstruction：https://en.wikipedia.org/wiki/Total_variation_denoising

---

## 实验结果：不输带人工标注的方法

数据集：TraM-NeRF 数据集子集，7 个 synthetic + 2 个 real-world scene。Metric：PSNR / SSIM / LPIPS。

### Synthetic Scene，全图指标

| Method | 用 mirror prior? | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| NeRF | ✗ | 26.53 | 0.819 | 0.370 |
| Mip-NeRF 360 | ✗ | 26.05 | 0.775 | 0.413 |
| Ref-NeRF | ✗ | 25.67 | 0.784 | 0.418 |
| MS-NeRF | ✗ | 29.13 | 0.836 | 0.377 |
| Mirror-NeRF | ✓ 人工标注 | 21.49 | 0.712 | 0.526 |
| TraM-NeRF | ✓ 人工标注 | 31.46 | 0.875 | 0.285 |
| **Ours** | ✗ 无标注 | **30.30** | **0.875** | **0.263** |

注意 LPIPS 上我们 **0.263 比 TraM-NeRF 0.285 还好**——尽管 TraM-NeRF 用了 manual mask。这个挺 surprising 的，作者的解释隐含是：自动检测的 mirror region 边界可能比人工标注更精确一点（人工标注有 labeling noise）。

### Synthetic Scene，仅镜面区域

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| NeRF | 28.55 | 0.945 | 0.070 |
| Mip-NeRF 360 | 28.96 | 0.946 | 0.069 |
| Ref-NeRF | 29.01 | 0.950 | 0.070 |
| MS-NeRF | 33.48 | 0.965 | 0.060 |
| Mirror-NeRF | 25.82 | 0.933 | 0.081 |
| TraM-NeRF | 38.23 | 0.982 | 0.034 |
| **Ours** | **35.94** | **0.978** | **0.037** |

只看镜面区域，我们 35.94 dB vs TraM-NeRF 38.23 dB，差 2.3 dB。这就是 "no prior" vs "perfect manual prior" 的 gap。这 2.3 dB 的 gap 来源主要是 stage (b) 的 detection 不完美——point cloud 可能漏一些 mirror 区域、或者边界拟合不精确。

Real-world 上差距更小（我们 32.20 vs TraM-NeRF 34.00，差 1.8 dB），可能因为真实场景的 manual mask 也有 noise， TraM-NeRF 没占尽 prior 优势。

---

## Limitations（作者老实讲了）

1. **Detection depends on NeRF failure**：如果镜子背后的 scene 没被 capture（比如镜子挂在墙上、墙后没拍），NeRF 会把镜后空荡荡的空间当 virtual scene 来 fit，反而 "假装成功"，于是 score 不高、镜面检测不到。这种场景里 proxy geometry 凑合用，但下游 geometry task（autonomous navigation、object tracking）就废了。
2. **Only primary-ray mirrors**：antialiased mask 只对第一次 surface interaction 算。多次 bounce（镜子里看到另一面镜子）的 frustum-surface intersection area 没做 differentiable antialiasing。
3. **Need to know primitive type & count**：k-means 的 $k$ 和 primitive 类型（plane? cylinder?）是 hyperparameter，得人工设。作者建议用 Efficient RANSAC (Schnabel et al. 2007) 之类自动 shape detection。
4. **Failure mode**：如果 stage (b) 出错、mirror 初始化位置错（paper Figure 6 演示），stage (c) 的优化会把 mirror 缩到 0 或推到 scene 后面，退化到 Mip-NeRF 360 水平——graceful degradation，不会爆炸，但 mirror 没了。
5. **Real-world 只验证 planar**：cylinder 在 synthetic 上 demo 过（Figure 7，PSNR 35.27 vs MS-NeRF 33.55），但 real-world 上其他 primitive 没验证。

参考 Efficient RANSAC：https://cg.cs.uni-bonn.de/en/publications/efficient-ransac-for-point-cloud-shape-detection/

---

## 一些更广的 intuition 和联想

### 1. "Failure as signal" 这个 pattern

这个 paper 让我最兴奋的是它代表了一个更 general 的 meta-pattern：**model 的失败模式本身就是 signal**。NeRF 在镜面上的失败不是 bug，是 feature——它告诉你 "这里有 view-inconsistent 的东西"。

这种思路在很多领域出现过：
- **Active learning**：用 model uncertainty 选样本去标注
- **Outlier detection**：用 model residual 找异常点
- **GAN debugging**：discriminator 找 generator 的 weakness
- **Adversarial examples**：model 失败的位置揭示 decision boundary 的几何结构
- **Classical CV**：optical flow 失败的地方往往是 occlusion boundary

NeRF 的 implicit assumption 是 view-consistency + static scene + Lambertian-ish。任何违反这些假设的 surface（mirror、glass、dynamic object、caustics）都会在 NeRF 训练中留下 photometric artifact。把这些 artifact 系统性地提取成 detection signal，可以泛化到很多场景。

参考 Active Learning: https://en.wikipedia.org/wiki/Active_learning_(machine_learning)

### 2. L1 vs L2 这个 trick 的根

L1 鼓励 sparsity、sharp edges，L2 鼓励 smooth averaging，这是 signal processing 里几十年的老故事：
- **Compressed sensing**：L1 让 signal 在某 basis 下 sparse
- **Total variation denoising**：L1 of gradient 保留 edge
- **LASSO**：L1 regularization 做 feature selection
- **Robust statistics**：L1 对 outlier 更 robust

在 differentiable rendering 里这个老 trick 也能复活，挺有意思。p-norm schedule 本质上是 "前期让模型收敛，中期切 L1 出 sharp edge，后期切回 L2 平滑"——half-quadratic splitting 那类老 CV 算法的现代 differentiable simulation 版本。

参考 Total Variation Denoising: https://en.wikipedia.org/wiki/Total_variation_denoising
参考 LASSO: https://en.wikipedia.org/wiki/Lasso_(statistics)

### 3. 为什么 SSIM × depth variance 这个组合 work

我反复想这个 score 设计，觉得它 hit 了一个特别好的 decomposition：

$$s(r) = \underbrace{\frac{1 - \mathrm{SSIM}(r)}{2}}_{\text{appearance failure}} \cdot \underbrace{e^{-cV(r)}}_{\text{geometry confidence}}$$

- appearance failure 高 = "NeRF 在这块 appearance 上学得烂"
- geometry confidence 高 = "NeRF 在这块 geometry 上学得有把握"
- 两个都高 = "烂得有把握" = "NeRF 知道 surface 在哪、但死活学不出 appearance" = 镜子

这个 decomposition 之所以 work，是因为 NeRF 的两个 output channel（color、density）在镜面上的表现是 **decoupled** 的——density 还能学到镜面表面的 sharp distribution（因为多视角的镜面位置是一致的），但 color 学不出（因为多视角的 reflection 不一致）。

把这个 decoupling 显式量化、再 threshold，就是 detection signal。这种 "decoupled channel failure" 的思路在别的 model 上也可能用——比如 GAN 的 generator 在某些 channel 上 mode collapse、另一些 channel 学得好；或者 diffusion model 在某些 frequency band 上学得好、另一些学不好。

### 4. 跟 3D Gaussian Splatting 的平行方向

最近 3D Gaussian Splatting (Kerbl et al. 2023) 也遇到镜面问题。有几个 follow-up：
- **MirrorGaussian** (Liu et al. 2024): https://arxiv.org/abs/2405.11921
- **Mirror-3DGS** (Meng et al. 2024): https://arxiv.org/abs/2404.01168

这些都是用 "two sets of Gaussians + reflection" 的思路，但都 **rely on manual mirror mask**。这篇 paper 的 "自动 detection" 思路完全可以移植过去——训一个 vanilla 3DGS，在失败区域做 SSIM + depth variance 分析，反投影拟合 primitive，再 joint optimize。如果有人做出来，应该是个不错的工作。

参考 3DGS: https://arxiv.org/abs/2308.14737

### 5. 跟 NeRF in the Wild / NeRF-W 的关系

NeRF-W (Martin-Brualla et al. CVPR 2021) 处理 "in-the-wild" photo collection——不同光照、不同 crowd、不同 exposure。它用 per-image appearance embedding + uncertainty 来 absorb multi-view inconsistency。

镜子也是一种 multi-view inconsistency，但和 NeRF-W 处理的那种本质不同：
- NeRF-W 的 inconsistency 是 **per-image global**（光照变化、人群出现）——所以 per-image embedding 能吸收
- 镜子的 inconsistency 是 **per-pixel local + view-dependent**——必须用 geometric reasoning（反射定律）来处理

参考 NeRF-W: https://arxiv.org/abs/2008.02268

### 6. 跟 Volume Rendering 经典理论的关系

这个 paper 用的是经典 alpha compositing：

$$C(r) = \sum_{k=1}^{K} T_k \alpha_k c_k$$

这个公式来自 Max 1995 (https://ieeexplore.ieee.org/document/385007)，原本是 scientific volume rendering 用的。NeRF 借用了它，但 assumption 是 "each particle emits color independently"。镜子违反这个——镜子 "粒子" 的颜色取决于反射方向上的其他粒子。

要从根上解决，得用更复杂的 light transport——比如 path tracing、Bidirectional Path Tracing、Metropolis Light Transport。NeRF->NeMf、Mirror-NeRF 都尝试了，但都 expensive。这篇 paper 用 "explicit mirror geometry + single-bounce reflection" 是个 pragmatic 折中——便宜、能 work，但只覆盖 mirror 这种简单 case。

参考 Path Tracing: https://en.wikipedia.org/wiki/Path_tracing

### 7. 一个可能的 follow-up：把 "failure signal" 用到更广的 view-inconsistent surface

我想看这样的工作：

- **Glass / transparent surface**：玻璃既是反射又是透射，NeRF 在玻璃上的失败模式可能和镜子不同（mixed reflection + transmission）。可以设计一个 score = reflection_score + transmission_score + depth_confidence，去做 glass detection
- **Caustics**：caustics 是光线聚焦形成的高光 pattern，NeRF 学不好 caustics，因为它们是 view-dependent 且 geometry 在 3D 空间外（在 surface 上）
- **Dynamic reflection**：镜子里的 dynamic object（比如镜子里有人走过）会让 NeRF 完全崩溃，但崩溃的 pattern 可能可以做 motion detection
- **Anisotropic specular**：拉丝金属、CD 表面这种 anisotropic BRDF，NeRF 也学不好，failure pattern 可能用来 infer material type

这些都基于同一个 meta-pattern：**NeRF 失败 = assumption violation = 该区域有物理 special**。

---

## 一句话 takeaway

把 NeRF 在镜面上的 photometric failure 用 "SSIM dissimilarity × depth confidence" 量化成 score map，反投影成 point cloud，RANSAC 拟合成 explicit primitive，作为 differentiable mirror geometry 再 joint optimize 整个 NeRF——用一个 p-norm schedule (2→1→2) 处理 sharp edge 和 smooth color 之间的 tradeoff，最终在不需要任何 mirror mask annotation 的情况下，PSNR/SSIM 接近带 manual prior 的 TraM-NeRF，LPIPS 甚至更好。

核心 insight 是：**NeRF 翻车的地方就是镜子，把翻车信号捡起来当 detector**。

GitHub repo: https://github.com/vc-bonn/nerfs-are-mirror-detectors

---

# NeRFs are Mirror Detectors — 技术讲解

## 1. Big Picture / TL;DR

这篇 paper 的核心 thesis 一句话能说完：**NeRF 训练失败的地方恰好就是镜子**。镜子表面在多视角下是 multi-view inconsistent 的（每个 viewpoint 看到的 reflection 都不同），标准 NeRF 在这部分区域会留下 photometric artifact —— rendered color 对不上 ground truth。其他 diffuse 区域 NeRF 已经收敛得很好。所以 NeRF 本身就是一个 mirror detector，作者把这个"failure signal"捡起来当 detection prior，再用 explicit 3D primitive（plane / cylinder）建模镜面，做一个可微 ray reflection 的二阶段联合优化，最终摆脱了所有 prior method 对 human-annotated mirror mask 的依赖。

GitHub repo: https://github.com/vc-bonn/nerfs-are-mirror-detectors

相关工作链接：
- 原 NeRF: https://arxiv.org/abs/2003.08934
- Mip-NeRF 360: https://arxiv.org/abs/2111.12027
- Ref-NeRF: https://arxiv.org/abs/2111.13458
- MS-NeRF: https://arxiv.org/abs/2304.02418
- NeRFReN: https://arxiv.org/abs/2111.12027
- Mirror-NeRF: https://arxiv.org/abs/2305.14841
- TraM-NeRF: https://arxiv.org/abs/2404.17370
- SSIM 原始 paper: https://www.cns.nyu.edu/~lcv/ssim/

## 2. 问题为什么难 — Mirror 对 NeRF 的本质挑战

标准 NeRF 假设 scene 是 view-consistent 的：从任何视角看同一个 3D point $x$，它的 outgoing radiance $c(x, d)$ 只取决于 position $x$ 和 view direction $d$。NeRF 的 MLP $F_\theta(x, d) \to (c, \sigma)$ 是有 well-defined 的 representation 的。

镜子打破这个假设。镜面点 $x_m$ 上看到的 radiance 取决于另一处场景点 $y$ 在镜面反射方向上的 radiance。当 camera 移动时，对应到 $x_m$ 上的 image pixel 完全来自不同的 $y$ —— 你在镜子里看到的是另一个相机角度的"虚拟场景"。NeRF 把这当成"window into a virtual scene"，于是把镜面后面的 empty space 当成 proxy geometry，硬生造一个能在多视角下"差不多对得上"的 radiance field，结果是：

- mirror geometry 错（depth 跑到镜面背后）
- mirror appearance 模糊（一个 5D MLP 想把多个视角的不同 reflection 压成一个 view-dependent color）
- 整个 scene 表征 inconsistent

## 3. Pipeline 三阶段（对应 Figure 1）

### Stage (a): Standard NeRF + Depth Reprojection Loss

先训一个标准 NeRF，但加一个 depth reprojection loss（来自 SPARF, Truong et al. CVPR 2023），解决 textureless / far-away 区域 depth 学不好的问题。

Standard volume rendering（公式 1）：

$$
C(r) = \sum_{k=1}^{K} T_k \alpha_k c_k
$$

变量解释：
- $r$：camera ray，$r(t) = o + td$，$o$ 是 camera origin，$d$ 是 ray direction
- $t_k$：沿 ray 的第 $k$ 个 sample point 的 depth
- $\sigma_k = \sigma(r(t_k)) \in [0,1]$：density（volume opacity）at $t_k$
- $\delta_k = t_{k+1} - t_k$：相邻 sample 间隔
- $\alpha_k = 1 - e^{-\sigma_k \delta_k}$：ray 在 $t_k$ 处"hit a particle"的概率（discrete alpha compositing）
- $T_k = \exp(-\sum_{j=1}^{k-1} \sigma_j \delta_j)$：transmittance，ray 从起点走到 $t_k$ 之前没被吸收的概率
- $c_k = c(r(t_k))$：radiance at $t_k$
- $C(r) \in [0,1]^3$：最终 rendered RGB

Expected depth（公式 3）：

$$
D(r) = \sum_{k=1}^{K} T_k \alpha_k t_k
$$

这是 ray 的 expected absorption position，几何上对应"被吸收到 pixel 上的 3D 点"。

Depth reprojection loss（公式 4）：

$$
\mathcal{L}_D(R) = \frac{1}{|R|} \sum_{r \in R} w_{ij} |D(r_j) - z_j(D(r), r)|^2
$$

变量：
- $i$：当前 ray $r$ 来自的 camera index
- $j$：从训练集里 uniform 随机抽的另一个 camera index
- $r_j$：从 camera $j$ 的 origin 出发、经过 absorption 点 $o + D(r)d$ 的那条 ray
- $z_j(D(r), r)$：absorption point 在 camera $j$ 坐标系下的 depth
- $w_{ij} = \frac{1}{w_{\max}} |\phi_{ij}| \|o_i - o_j\|_2$：weight，$|\phi_{ij}|$ 是 camera $i,j$ 光轴夹角，$\|o_i - o_j\|_2$ 是 camera origin 距离
- $w_{\max}$：normalize 用

直觉：如果 NeRF 学的 depth 是 geometrically correct 的，那么同一个 3D absorption point 从另一个 camera 看应该有 consistent depth。$w_{ij}$ 起作用的地方是：当两个 camera 离得很远、角度差很大，它们的 ray 可能观察不同表面点（occlusion），就降低这个 cross-camera consistency 项的权重。这个 loss 把 diffuse 区域的 depth 拉直，但 mirror 区域还是不行——因为 mirror 的"apparent depth"本来就是错的（虚拟场景在镜后）。

### Stage (b): Mirror Detection via SSIM + Depth Variance

这是 paper 的核心 insight。训完 stage (a) 后，对每个 training image pixel，算一个 score $s(r)$，高 score = 这 pixel 大概率属于镜面。

**SSIM**（公式 5）—— 衡量 local structural similarity：

$$
\mathrm{SSIM}(r) = \frac{(2\mu\mu^* + c_1)(2\tilde{\sigma} + c_2)}{(\mu^2 + (\mu^*)^2 + c_1)(\sigma^2 + (\sigma^*)^2 + c_2)}
$$

变量：
- $\mu, \mu^*$：在 rendered image 和 ground truth image 上、以 ray $r$ 对应 pixel 为中心的 local window 的均值
- $\sigma, \sigma^*$：相应方差
- $\tilde{\sigma}$：两幅图像 local window 之间的 covariance
- $c_1, c_2$：small constants 防止除零，按 Wang et al. 2004 设

为什么用 SSIM 而不是 per-pixel MSE？因为 mirror 区域的 failure 表现为 "blurriness / structure loss"，SSIM 抓的是 local structure（luminance、contrast、structure 三项乘起来），比 squared error 更敏感地捕捉到 "NeRF 把镜面反射平均成一个糊掉的低频 texture" 这种 artifact。

**Depth variance**（公式 6）：

$$
V(r) = \sum_{k=1}^{K} T_k \alpha_k (t_k - D(r))^2
$$

这是 absorption position 沿 ray 的 variance。$V(r)$ 大表示 NeRF 对 depth 不确定（probability mass 沿 ray 摊得很开）；$V(r)$ 小表示 depth 锐利、确定。

**Final score**（公式 7）：

$$
s(r) = \frac{1 - \mathrm{SSIM}(r)}{2} \, e^{-cV(r)}
$$

变量：
- $1 - \mathrm{SSIM}(r)$：把 "similarity" 翻成 "dissimilarity"（高分 = 不像）
- 除以 2 把范围 normalize 到 $[0, 1]$（SSIM $\in [-1, 1]$）
- $e^{-cV(r)}$：depth variance 大时把这个 weight 压低
- $c \in \mathbb{R}$：控制衰减速度的 hyperparameter

直觉：mirror 区域应该满足两个条件**同时**成立—— (1) appearance 学不好 (high $1 - \mathrm{SSIM}$)；(2) geometry 学得挺准 (low $V$)。第二个条件 crucial，因为 textureless diffuse 区域也会有 high SSIM error，但那种地方 $V$ 也大（depth 不确定），所以 $e^{-cV}$ 会把它压掉。这个 $V$ 作为"否决票"的设计非常巧妙 —— 把"NeRF 整体没学好"和"NeRF 只在镜面上没学好"区分开。

**Primitive Fitting**：threshold score 得到一组 high-score pixels，unproject 到 3D（用 $D(r)$），得 point cloud。估 normal（hybrid radius + kNN 取邻居，再 PCA）。然后：
1. 用 k-means 把 point cloud 切成 $k$ 个 cluster（$k$ 是预设镜子数量）
2. 每个 cluster 用 RANSAC 拟合 primitive（plane / cylinder / 等等），先拟合 unbounded shape（如 infinite plane），再算 oriented bounding box 限定范围
3. 用三个 metric 评估 fit 质量：inlier ratio、average shortest distance to unbounded shape、predicted normal vs estimated normal 的相似度

每个 primitive 用一组参数 $\theta$ 描述（plane 是法向量 + offset + size，cylinder 是 axis + radius + height 之类）。

### Stage (c): Joint Optimization with Differentiable Mirror Geometry

这一步基于 TraM-NeRF 的 reflection-aware rendering，但作者把它做成 differentiable w.r.t. $\theta$，并加上 antialiased mask blending 和 p-norm schedule。

**Reflection-aware rendering**：每个 camera ray，递归检查是否 intersect 任一 primitive；命中则在 surface 处反射，按 ideal reflection law $d_{\text{refl}} = d - 2(d \cdot n)n$ 算新方向，继续 ray march；达到 bounce limit 停。

**Differentiable mask blending**：直接把 ray-march 的 color 往 primitive $\theta$ 上传 gradient 不稳定（path tracing 的 gradient 噪声大）。作者绕过 Eq. (1)，改用 differentiable antialiased mask：对每个 pixel frustum，算它和 mirror surface 的相交面积比例 $m \in [0,1]$，然后

$$
C_{\text{pixel}} = (1 - m) \cdot C_{\text{primary}} + m \cdot C_{\text{reflected}}
$$

$C_{\text{primary}}$ 是 normal NeRF 渲的（假设没镜子），$C_{\text{reflected}}$ 是反射 ray 渲的。$m$ 是 differentiable w.r.t. $\theta$（用 nvdiffrast 之类的 differentiable rasterizer，Laine et al. 2020）。这样 image loss 的 gradient 通过 $m$ 直接传到 $\theta$，避开了 volume rendering 那条 noisy path。

**p-norm schedule**（公式 8）—— 这是我觉得 paper 里最 insightful 的工程细节：

$$
\mathcal{L}_I' = \frac{1}{|R|} \sum_{r \in R} \| C(r) - C^*(r) \|_{p(\tau)}^{p(\tau)}
$$

$p(\tau)$ 是 iteration $\tau$ 的 piecewise linear function：
- $\tau < \tau_{\text{init}}$：$p = 2$（MSE，先把整个 NeRF 拉到一个 reasonable baseline）
- $\tau_{\text{init}} < \tau < \tau_{\text{inc}}$：$p = 1$（L1 loss，鼓励 sharp edges，让 mirror boundary 锐利出来 —— mirror 区域和非 mirror 区域差异最大化）
- $\tau_{\text{inc}} < \tau < \tau_{\text{std}}$：$p$ 线性从 1 升回 2（去掉 L1 造成的 piecewise-constant color plateau）

为什么 L1 鼓励 sharp edges？L1 norm 是 sparsity-inducing：对于一组 residual，L1 倾向于把大部分 residual 推到 0、保留少数大 residual，而 L2 倾向于把所有 residual 均摊小。映射到 image 上，L2 优化的解是 "blurry averaging"（多个候选 appearance 取均值），L1 优化的解是 "sharp selection"（保留一个 dominant appearance）。

为什么 L1 又会卡在 local constant color？L1 的 gradient 在 residual $\to 0$ 时是 sign(residual)，constant magnitude 1（不是 2·residual 那样随 residual 变小而变小），所以当大部分 pixel 已经对上、剩小区域需要精细化时，gradient 信号不够局部化，优化会卡在 piecewise constant 解。所以最后要切回 L2 把 plateau 平滑掉。

这个 $p(\tau)$ schedule 在 Figure 3 有可视化 —— L2 一直训下去镜子边缘糊；L1 一直训镜子内部出现 flat color block；schedule 之后边缘锐利且内部 smooth。

## 4. 实验结果（Table 1）

数据集：TraM-NeRF 数据集的子集，7 个 synthetic + 2 个 real-world scene，含一个或多个 planar mirror。Metric：PSNR / SSIM / LPIPS。

**Full Images（Synthetic）**：

| Method | Prior | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| NeRF | ✗ | 26.53 | 0.819 | 0.370 |
| Mip-NeRF 360 | ✗ | 26.05 | 0.775 | 0.413 |
| Ref-NeRF | ✗ | 25.67 | 0.784 | 0.418 |
| MS-NeRF | ✗ | 29.13 | 0.836 | 0.377 |
| Mirror-NeRF | ✓ | 21.49 | 0.712 | 0.526 |
| TraM-NeRF | ✓ | 31.46 | 0.875 | 0.285 |
| **Ours** | ✗ | **30.30** | **0.875** | **0.263** |

注意 LPIPS 上我们 0.263 比 TraM-NeRF 0.285 还好，尽管 TraM-NeRF 用了 manual mask。这说明自动检测的 mirror region 反而可能比人工标注更"precise"一点 —— 边界更准确。

**Mirror Regions only（Synthetic）**：

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| NeRF | 28.55 | 0.945 | 0.070 |
| Mip-NeRF 360 | 28.96 | 0.946 | 0.069 |
| Ref-NeRF | 29.01 | 0.950 | 0.070 |
| MS-NeRF | 33.48 | 0.965 | 0.060 |
| Mirror-NeRF | 25.82 | 0.933 | 0.081 |
| TraM-NeRF | 38.23 | 0.982 | 0.034 |
| **Ours** | **35.94** | **0.978** | **0.037** |

只看镜面区域，我们 35.94 dB 离 TraM-NeRF 38.23 dB 大约 2.3 dB 差距 —— 这 2.3 dB 就是 "no prior" vs "perfect manual prior" 的 gap，未来工作显然可以缩小。但已经远超所有 baseline method。

Real-world 上我们 32.20 dB 比 TraM-NeRF 34.00 dB 差 1.8 dB，差距比 synthetic 小一点，说明真实场景里 manual prior 也不那么完美（mask 边界本来就有 labeling noise）。

## 5. Limitations（作者也老老实实讲了）

1. **Detection depends on NeRF failure**：如果镜子背后的场景没被 capture（比如镜子挂在墙上，墙后面没拍），NeRF 会把镜后空间当作虚拟场景来重建，反而能"成功"拟合，于是 score 不高、镜面检测不到。这种场景里几何上用 proxy 也凑合，但下游 geometry task（autonomous navigation、object tracking）就废了。
2. **Only primary-ray mirrors**：antialiased mask 只对第一次 surface interaction 算。多次 bounce（mirror 看镜子里另一面镜子）的 frustum-surface intersection area 没做 differentiable antialiasing。
3. **Need to know primitive type & count**：k-means 的 $k$ 和 primitive 类型是 hyperparameter，得人工设。作者说可以用 Efficient RANSAC (Schnabel et al. 2007) 之类自动检测。
4. **Failure mode**：如果 stage (b) 出错、mirror 初始化位置错（Figure 6），stage (c) 的优化会把 mirror 缩到 0 或推到 scene 后面，退化到 Mip-NeRF 360 水平 —— graceful degradation，不会爆炸，但 mirror 没了。
5. **Real-world 上只验证 planar**：cylinder 在 synthetic 上 demo 过（Figure 7，PSNR 35.27 vs MS-NeRF 33.55），但 real-world 其他 primitive 没验证。

## 6. 我的 intuition / takeaway

这个 paper 让我想到一个更 general 的 meta-pattern：**"失败模式作为信号"**。NeRF 在镜面上的失败并不是 bug，是 feature —— 它告诉你"这里有 view-inconsistent 的东西"。把这种 inconsistency 显式提取出来作为 supervision signal，比 hand-craft 一个 mirror prior 优雅得多。

类比一下：
- Ref-NeRF 的 reflected direction conditioning 是"教 NeRF 什么是 specular"的硬塞 prior
- NeRFReN 的 transmitted/reflected decomposition 是"假设一个生成模型"
- TraM-NeRF 是"假设我知道镜面在哪，做物理正确的 rendering"
- 这篇 paper 是"让 NeRF 自己告诉我镜面在哪"

我比较看好这种"self-discovering"路线 —— 当 NeRF 在某区域 loss 高，往往是因为该区域违反了 NeRF 的某个 implicit assumption（view-consistency, static scene, Lambertian-ish）。把这些 assumption 的 violation 当 detection signal，可以泛化到很多场景：transparent surface (glass)、dynamic scene (motion)、light field (caustics) 等等。Score function 怎么设计才是关键 —— 这篇用 SSIM × depth variance，是 photometric inconsistency × geometric certainty 的组合，generalize 时要想清楚对应的两个因子。

另一个 takeaway：p-norm schedule 这个 trick 其实更 general。任何 sharp-edge / piecewise-smooth 的 inverse problem 都可能用到 —— sparsity prior 早期 + smoothness prior 后期。可以联想到 L1-TV denoising、super-resolution 里的 half-quadratic splitting 之类 classic CV 技术，看来 differentiable rendering 里这些老 trick 也能复活。

## 7. 一些值得 follow 的链接

- TraM-NeRF (基础方法): https://arxiv.org/abs/2404.17370
- MS-NeRF (主要 baseline): https://openaccess.thecvf.com/content/CVPR2023/papers/Yin_Multi-Space_Neural_Radiance_Fields_CVPR_2023_paper
- Ref-NeRF: https://arxiv.org/abs/2111.13458
- Mip-NeRF 360: https://arxiv.org/abs/2111.12027
- SPARF (depth reprojection loss): https://arxiv.org/abs/2301.02511
- nvdiffrast (differentiable rasterization for mask): https://github.com/NVlabs/nvdiffrast
- SSIM: https://en.wikipedia.org/wiki/Structural_similarity
- Efficient RANSAC (shape detection for future work): https://cg.cs.uni-bonn.de/en/publications/efficient-ransac-for-point-cloud-shape-detection/
- 3D Gaussian Splatting (相关 parallel direction): https://arxiv.org/abs/2308.14737

## 8. 一句话总结

把 NeRF 在镜面上的 photometric failure 用 SSIM + depth variance 量化成 score map，反投影成 point cloud，RANSAC 拟合成 explicit primitive，作为 differentiable mirror geometry 再 joint optimize 整个 NeRF —— 用一个 p-norm schedule 解决 L2-blur 和 L1-plateau 之间的 tradeoff，最终在不需要任何 mirror mask annotation 的情况下，逼近带 manual prior 的 TraM-NeRF 的镜面重建质量。
