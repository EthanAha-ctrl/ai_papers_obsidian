---
source_pdf: PanSplat.pdf
paper_sha256: 4b0e535f65a356111fe4eedd20f779d699a8e8737e0b93dc831e8cc6a7d9fb03
processed_at: '2026-08-06T02:02:05-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PanSplat 用人话说

## 这篇paper想干啥

你拿一个360°相机，在房间里拍两张全景照片（相隔1米左右），然后想生成中间任意视角的图——比如VR里你往前走一步看到的画面。这事叫 **wide-baseline panorama novel view synthesis**。

以前要么慢得要死（PanoGRF一张图23.8秒，VR没法用），要么分辨率太低（512×1024，戴VR头显一眼就糊）。PanSplat第一个做到：**4K分辨率 + 0.34秒推理 + 单卡A100能训练**。

paper: https://arxiv.org/abs/2412.05290
code: https://github.com/chengzhag/Pansplat

---

## 为什么这事难

三个目标互相打架：

1. **4K分辨率**：VR头显贴脸看，低分辨率立刻露馅
2. **实时推理**：NeRF类方法慢成狗，没法用
3. **显存要小**：4K全景图一张就4096×2048×3=25MB，feature map、cost volume、Gaussian参数全堆显存会爆

MVSplat [16]（目前perspective feed-forward 3DGS的SOTA）在512×1024就OOM了。PanoGRF [17]（panorama NeRF SOTA）一张图23.8秒。都没法用。

---

## PanSplat的三个核心insight

我把paper拆成三件事，每一件单独看都不算breakthrough，但**组合起来**才让4K成为可能。

### Insight 1: 全景图是球面，不是拉伸的矩形

Equirectangular projection 把球面摊成矩形：
$$\theta = \frac{x}{W} \cdot 2\pi, \quad \phi = \frac{y}{H} \cdot \pi - \frac{\pi}{2}$$

变量解释：
- $x, y$：图像像素坐标
- $W, H$：图像宽高（4K下 $W=4096, H=2048$）
- $\theta$：经度，$[0, 2\pi)$
- $\phi$：纬度，$[-\pi/2, \pi/2]$

问题在球面面积元素：
$$dA = R^2 \cos\phi \, d\theta \, d\phi$$

靠近poles（$\phi \to \pm\pi/2$）时 $\cos\phi \to 0$，但图像上每个像素对应的 $d\theta \, d\phi$ 是固定的。**结果：poles处像素严重过采样**——很多像素描述同一个极小的球面区域。

MVSplat用pixel-aligned Gaussian（每个像素一个Gaussian），poles处就有海量冗余Gaussian互相重叠。Fig.2左下角可视化得很清楚，poles处Gaussian挤成一团。

**PanSplat的解法：Fibonacci lattice**

Fibonacci lattice是在球面上quasi-uniform采样的经典方法。第$j$个Gaussian在图像上的坐标：

$$
(x_j, y_j) = \left( j \cdot \phi \mod 1, \frac{j}{n-1} \right)
$$

变量解释：
- $j \in [0, n-1]$：Gaussian的index
- $\phi = \frac{1+\sqrt{5}}{2}$：黄金比例≈1.618（无理数）
- $j \cdot \phi \mod 1$：取$j \cdot \phi$的小数部分。因为$\phi$是无理数，小数部分会均匀密布$[0,1)$（Weyl等分布定理）
- $y_j = j/(n-1)$：纬度方向线性均匀分布
- $n = \lfloor W^2/\pi \rfloor$：Gaussian总数，让equator附近密度与像素密度匹配

Intuition：golden ratio让$x_j$序列在$[0,1)$上有最优的low-discrepancy性质——每加一个点就填最大空隙。线性的$y_j$配上$1/\cos\phi$的面积权重，自然就让球面上分布均匀。

Ablation（Table 3）：
- Base（pixel-aligned）：1049K Gaussians, WS-PSNR 27.07
- +Fibo：668K Gaussians（减少36.34%），WS-PSNR 27.86（反而升0.79 dB）

**这就是免费午餐**——单纯改采样策略，Gaussian少了三分之一，质量还升了。

参考：
- Fibonacci Lattice可视化: https://observablehq.com/@meetamit/fibonacci-lattices
- Roberts 2020更优分布: https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/

---

### Insight 2: 多尺度Gaussian Pyramid，但要有residual引导

光有Fibonacci还不够，paper又加了3D Gaussian Pyramid（3DGP），4个level（$l=0,1,2,3$），每层Gaussian数：

$$
n^l = \left\lfloor \frac{W^2}{2^l \cdot \pi} \right\rfloor
$$

变量解释：
- $l=0$：最细层，$n^0 = \lfloor W^2/\pi \rfloor$，捕获high-frequency details
- $l=3$：最粗层，$n^3 = \lfloor W^2/(8\pi) \rfloor$，捕获global structure
- $2^l$：每升一层，Gaussian数减半

**但单纯堆金字塔没用**——paper在Sec. E的ablation发现网络会"lazy"，最粗两层输出无意义Gaussian。所以加**residual design**：

每个level $l$的Gaussian head输入包括：
1. 当前level的cost volume（resize过）
2. 当前level的input image
3. 当前level的feature map $F_i^l$
4. **上一level的feature map** $\tilde{F}_i^{l+1}$（upsample过的）

Gaussian head预测的是**残差**，从粗到细累积修正。这强制了coarse-to-fine的依赖。

Ablation（Table E.1）：
- w/o 3DGP residual: WS-PSNR 28.14（-0.67 dB）
- w/o Hierarchical CV: WS-PSNR 26.95（-1.86 dB）
- Full: 28.81

Fig. E.1可视化：Full model在最粗level也能渲染出可识别房间结构，w/o residual版本最粗level渲染出噪声。

Intuition：神经网络没有明确引导就会偷懒，residual connection强制信息从粗到细流动。

---

### Insight 3: Geometry和Appearance分辨率decouple + Deferred Backprop

这是让4K训练成为可能的**工程核心**。

**关键观察：图像质量更依赖texture resolution，而非geometry resolution**

人眼看scene时，纹理糊了立刻能看出来，geometry略糙感知不强。所以：

- **Hierarchical cost volume只在512×1024跑**（geometry低分辨率足够）
- **Gaussian heads在全分辨率跑**（appearance需要全分辨率纹理）
- 中间通过upsample衔接

Fig.6显示：MVSplat在512×1024就OOM，PanSplat靠这个decoupling能跑到768×1536。但4K还是不够。

**Two-step Deferred Backpropagation**：

```
Forward pass（关闭auto-diff）:
  Cost volume + Gaussian heads → Gaussians → 渲染全景图 → 算image loss
  缓存 ∂L/∂I（pixel gradients）

Backward Step 1（开启auto-diff）:
  重新渲染全景图，一次一个cubemap face
  每个face: 反向传播 ∂L/∂face → ∂L/∂Gaussians（累积）

Backward Step 2（开启auto-diff）:
  重新生成Gaussian参数，一次一个tile
  每个tile: 反向传播 ∂L/∂Gaussians → ∂L/∂网络参数（累积）
```

为什么能这么做：
1. Image loss是pixel-wise的，pixel gradients一次forward就够
2. Gaussian rendering是local operation（每个像素只受附近Gaussian影响），face by face梯度能正确累积
3. Gaussian head是local CNN（3×3卷积），tile by tile梯度也能累积——但tile边界要padding 3 pixels避免discontinuity

Fig. G.1内存数据：
- 无deferred BP: 4K OOM在80GB A100
- 4 tiles deferred BP: 支持2048×4096训练
- 16 tiles deferred BP: 4K推理能在24GB RTX 3090跑

Intuition：deferred BP本质是"存储所有中间结果"换成"重新计算+梯度累积"，用compute换memory。4K这种极端case下compute比memory便宜。

参考：
- GGRT [43]: https://arxiv.org/abs/2408.15282
- ARF [81]: https://arxiv.org/abs/2108.09089

---

## Hierarchical Spherical Cost Volume

这块借鉴CasMVS [25]和PanoGRF的spherical projection，适配panorama。

**Feature Pyramid**:
- FPN [46]提取4级feature pyramid $\{F_i^l\}_{l=0}^{L-1}$，通道数$C^l \in \{128, 96, 64, 32\}$
- 最粗level（$l=3$）加Swin Transformer [48] with cross-view attention，让两个输入panorama交换信息（wide-baseline matching的关键）
- 用xFormers [36]加速

**Cost Volume Construction**:
1. 对reference view $i$，采样$D=128$个inverse depth候选$d \in [d_{min}, d_{max}]$
2. 把另一view $1-i$的feature warp到reference view坐标系（spherical projection [17, 42]）
3. 计算correlation: $C_i^3 = \langle F_i^3, \text{warp}(F_{1-i}^3) \rangle$（dot product）
4. 2D U-Net refine，融合monocular depth features [33]
5. Softmax over depth dimension → depth probability $p(d)$
6. 加权平均：$D_i^3 = \sum_d d \cdot p(d)$

**Hierarchical Refinement**（$l=3 \to l=2 \to l=1$，跳过$l=0$）：
1. $D_i^{l+1}$upsample到level $l$
2. 在$D_i^{l+1}$附近小范围$(d_{max}-d_{min})/2^{3-l}$搜索$D/2^{3-l}$个候选
3. 独立2D U-Net refine，加$D_i^{l+1}$作为contextual input

跳过$l=0$是为了在相似内存预算下达到MVSplat的2× depth分辨率。Trade-off：纹理全分辨率，几何1/2分辨率够用。

参考：
- CasMVS: https://arxiv.org/abs/1912.06378
- MVSplat: https://arxiv.org/abs/2403.07607
- PanoGRF: https://arxiv.org/abs/2310.17613
- Swin Transformer: https://arxiv.org/abs/2103.14030
- FPN: https://arxiv.org/abs/1612.03144

---

## Cubemap Renderer

为什么不直接在equirectangular上splatting？因为3DGS原生CUDA renderer是perspective camera的。equirectangular上rasterization在poles处有严重distortion。

PanSplat方案：
1. 同一位置放6个90° FOV perspective camera，朝6个cubemap face方向
2. 用原生3DGS CUDA renderer渲染6张face图
3. Differentiable grid sampling + bilinear interpolation拼成equirectangular panorama

工程细节：cube face边缘要pad 4个相邻face的边缘像素，保证bilinear interpolation在接缝处有正确邻居。

为什么cubemap而非直接equirectangular splatting：
1. 复用现有3DGS CUDA renderer，避免重写
2. **Sequential face rendering让4K渲染的中间结果不必同时驻留显存**——这是two-step deferred backprop的基础

参考：
- 3DGS CUDA renderer: https://github.com/graphdeco-inria/gaussian-splatting

---

## Deferred Blending: 真实数据的moving objects

真实数据（360Loc [30], Insta360）有moving objects（行人、相机操作者、车辆），两个输入view的depth不一致，合并Gaussians会出现ghosting artifacts。

解决方案：**不合并两个view的Gaussians，分别渲染再blend**：

$$
I = \frac{d_1 \tilde{I}_0 + d_0 \tilde{I}_1}{d_0 + d_1}
$$

变量解释：
- $\tilde{I}_i$：只用view $i$的Gaussians在target view渲染的图
- $d_i$：target view到input view $i$的距离
- $I$：final blended image

Intuition：target view离view 0近时（$d_0$小），公式中$\tilde{I}_0$权重$\frac{d_1}{d_0+d_1} \to 1$，所以靠近view 0时几乎只用view 0渲染，避免view 1里的moving object干扰。target在中间时两者权重相等，平滑过渡。

Table 2提升：
- 360Loc WS-PSNR: 24.96 → 27.35（+2.39 dB）
- Insta360 WS-PSNR: 24.43 → 25.68（+1.24 dB）

Frame distance小的时候（target离某个input view很近）提升最大（Fig. F.1）——正好对应deferred blending设计目的。

参考：
- IBRNet [66]: https://arxiv.org/abs/2102.13090
- 360Loc dataset: https://arxiv.org/abs/2310.19029

---

## 实验数据深度分析

### Table 1: Synthetic datasets

| Method | 1.0m WS-PSNR | 1.5m | 2.0m | Replica | Residential |
|--------|--------------|------|------|---------|-------------|
| S-NeRF [49] | 15.25 | 14.16 | 13.13 | 16.10 | 22.47 |
| OmniSyn [42] | 22.90 | 20.31 | 18.91 | 23.17 | — |
| IBRNet [66] | 25.72 | 21.69 | 20.04 | 22.65 | 22.47 |
| NeuRay [47] | 24.92 | 21.92 | 19.85 | 25.90 | 22.38 |
| PanoGRF [17] | 27.12 | 23.38 | 20.96 | 29.22 | 31.03 |
| MVSplat [16] | 28.19 | 21.82 | 13.31 | 30.54 | 31.21 |
| **PanSplat** | **28.81** | **24.09** | 20.56 | 30.78 | 30.97 |

观察：
1. **Wide baseline退化**：MVSplat在2.0m崩到13.31（和S-NeRF一个量级），原因是MVSplat为perspective设计，cost volume在wide baseline下匹配失败。PanSplat的hierarchical cost volume让wide baseline还能保持20.56
2. **1.0m**：PanSplat比MVSplat高0.62 dB，比PanoGRF高1.69 dB
3. **2.0m**：PanSplat比PanoGRF略低0.40 dB，但比MVSplat高7.25 dB
4. **Replica & Residential**：泛化能力相近，说明model在unseen scene structure上泛化好

### Table 2: Real data

| Dataset | Method | PSNR | WS-PSNR | SSIM | LPIPS |
|---------|--------|------|---------|------|-------|
| 360Loc | MVSplat | 24.13 | 24.67 | 0.823 | 0.170 |
| 360Loc | PanSplat | 24.96 | 25.58 | 0.833 | 0.159 |
| 360Loc | PanSplat+BL | **27.35** | **28.14** | **0.860** | **0.127** |
| Insta360 | MVSplat | 20.93 | 23.24 | 0.786 | 0.227 |
| Insta360 | PanSplat | 21.92 | 24.43 | 0.813 | 0.211 |
| Insta360 | PanSplat+BL | **23.36** | **25.68** | **0.822** | **0.183** |

Deferred blending带来~2-3 dB提升，印证moving objects是真实数据的主要artifact来源。

### Table 3: 主Ablation

| Setup | #Gaussians | WS-PSNR | SSIM | LPIPS |
|-------|------------|---------|------|-------|
| Base | 1049K (100%) | 27.07 | 0.895 | 0.127 |
| +Fibo | 668K (63.67%) | 27.86 | 0.906 | 0.116 |
| +3DGP (Full) | 887K (84.55%) | 28.81 | 0.931 | 0.091 |

Intuition：
- Base → +Fibo：Gaussian减36%，质量升0.79 dB
- +Fibo → +3DGP：Gaussian加33%，质量再升0.95 dB
- 最终Gaussian数887K，仍比Base少15.45%，质量高1.74 dB

### Table E.1: Design Ablation

| Setup | WS-PSNR |
|-------|---------|
| w/o Mono depth | 28.84 |
| w/o 3DGP residual | 28.14 (-0.67) |
| w/o Hierarchical CV | 26.95 (-1.86) |
| w/o First 3 GHs | 28.05 (-0.76) |
| Full | 28.81 |

观察：
- Mono depth贡献微弱（0.03 dB）
- Hierarchical CV是最重要单一设计（-1.86 dB），没有它cost volume在wide baseline上完全失败
- 3DGP residual强制coarse-to-fine，没有它最粗level学不出来
- First 3 GHs（只用最细level）掉0.76 dB，多尺度确实有用

### Table D.1: Narrow baseline

| Method | 0.2m WS-PSNR | 0.5m WS-PSNR |
|--------|--------------|--------------|
| PanoGRF | **34.29** | 31.41 |
| MVSplat | 32.93 | 31.55 |
| PanSplat | 33.92 | **32.46** |

在0.2m窄baseline PanoGRF反而最好。Intuition：PanoGRF用dense cost volume+双目+单目融合，窄baseline匹配容易，dense sampling充分利用信息；PanSplat跳过最细cost volume层级，窄baseline下精度上限受限于这个设计选择。但0.5m以上PanSplat全面领先，wide baseline才是实际场景的真实需求。

### 速度

- PanoGRF: 23.8s/image（NeRF volumetric rendering）
- PanSplat: 0.34s/image（0.32s forward + 0.02s 3DGS rasterize）
- **70× 加速**

这是3DGS相比NeRF的天然优势——rasterization vs volumetric sampling的本质差异。

---

## Training Loss详解

### Synthetic data（有GT depth）

**Depth loss**：
$$
\mathcal{L}_{depth} = \sum_{i=0,1} \sum_{l=1}^{3} \gamma^{l-1} \left\| D_i^l - \hat{D}_i^l \right\|_1
$$

变量：
- $i \in \{0,1\}$：两个input views
- $l \in \{1,2,3\}$：三个cost volume层级
- $\gamma = 0.9$：衰减因子，$l=1$时$\gamma^0=1$（最细最重），$l=3$时$\gamma^2=0.81$
- $D_i^l$：level $l$的预测depth
- $\hat{D}_i^l$：GT depth下采样到level $l$

**RGB loss**：
$$
\mathcal{L}_{rgb} = \| I - \hat{I} \|_2 + \lambda \cdot \text{LPIPS}(I, \hat{I})
$$

变量：
- $I$：渲染图，$\hat{I}$：GT图
- $\| \cdot \|_2$：L2范数（MSE）
- $\lambda = 0.1$：LPIPS权重
- LPIPS [83]：perceptual loss用VGG features

总loss：$\mathcal{L}_{synthetic} = \alpha \mathcal{L}_{depth} + \mathcal{L}_{rgb}$，$\alpha = 0.05$

### Real data（无GT depth）

用auxiliary Gaussian heads在每个cost volume level单独渲染+监督：

$$
\mathcal{L}_{real} = \sum_{l=1}^{3} \gamma^{l-1} \mathcal{L}_{rgb}(I^l, \hat{I}) + \mathcal{L}_{rgb}(I, \hat{I})
$$

变量：
- $I^l$：从level $l$的auxiliary head渲染的panorama
- $I$：主Gaussian head渲染的panorama
- 第一项：supervise每个level的cost volume通过可微分渲染
- 第二项：supervise最终输出

这是self-supervised depth learning via 3DGS的思路（[16, 67]也用），把depth supervision转换成image supervision。Auxiliary heads只有2个CNN层（很轻），不和主head共享residual design。

参考：
- LPIPS: https://arxiv.org/abs/1801.03924
- MVSplat: https://arxiv.org/abs/2403.07607
- LatentSplat [67]: https://arxiv.org/abs/2403.16292

---

## 4K Training Pipeline

训练schedule很informative：
1. **Stage 1**: Matterport3D @ height 256, batch 6, 10 epochs
2. **Stage 2**: fine-tune @ height 512, batch 2, 5 epochs
3. **Stage 3**: 4K Matterport3D fine-tune, progressive height 1024→2048, 每stage 3 epochs
4. **Real fine-tune**: 360Loc @ height 512→1024→2048, iterations 65K/26K/13K

1024和2048阶段启用deferred BP，分别用4 tiles和16 tiles，batch size 3和1。

关键trick：fine-tune真实数据时**冻结hierarchical cost volume**，只调Gaussian heads——避免没有GT depth时cost volume漂移，同时让Gaussian heads适应真实图像纹理分布。

---

## 整体intuition总结

PanSplat成功可以归结为三件事协同：

1. **Geometry/Appearance Decoupling**: geometry在512×1024跑，appearance在4K跑。Intuition：纹理高频，几何低频——人眼看scene不会因为geometry略糙而感觉差，但纹理糊立刻能看出来。

2. **Spherical-aware Sampling**: Fibonacci lattice + spherical cost volume。把panorama当作球面而非拉伸的矩形处理。Intuition：如果承认panorama是球面的，那所有pixel-aligned操作都有bias——Fibonacci lattice是消除这个bias的最简洁方式。

3. **Deferred Computation**: two-step deferred backpropagation + cubemap renderer + tiled Gaussian heads。让4K的中间张量不必同时驻留显存。Intuition：deferred BP本质是把"存储所有中间结果"替换成"重新计算+梯度累积"，用compute换memory，4K这种极端case下compute比memory便宜。

三点层层递进——单做任何一个都不够。比如只做resolution decoupling，PanSplat在768×1536就OOM；加上deferred BP才能到4K。这是paper"组合拳"的价值。

但paper也坦白说deferred BP让training复杂度上升，工程实现难度大。这是这类work的固有trade-off。

---

## 关键Limitations

Paper自己承认：**不支持动态场景**。Future work可以融合motion-aware representations（如dynamic 3DGS, 4DGS）。

我额外想到的几个limitation：
1. **Wide baseline退化**：2.0m时PanSplat比PanoGRF略低，hierarchical设计在极宽baseline下还有改进空间
2. **GT depth依赖**：合成数据训练需要GT depth，限制可用数据集
3. **Cubemap renderer边缘**：虽然在cubemap接缝处padding，但极端high-frequency区域可能仍有artifact
4. **Inference tiling overhead**：tiled operation增加overhead，4K实时VR可能还是吃力

---

## 与相关工作的broader思考

### vs. MVSplat
- **Geometry部分**：MVSplat用单层cost volume（128 candidates），PanSplat用3层hierarchical（128+64+32）
- **Gaussian placement**：MVSplat pixel-aligned，PanSplat Fibonacci
- **Multi-scale**：MVSplat单尺度，PanSplat pyramid
- **Memory**：MVSplat 512×1024就OOM，PanSplat 4K训练

PanSplat等于把MVSplat的idea重新设计成panorama-aware + scalable。

### vs. PanoGRF
- **质量**：PanSplat高1.69 dB (1.0m)
- **速度**：PanSplat快70×
- **Resolution**：PanoGRF最多512×1024，PanSplat 4K
- **Architecture**：PanoGRF还在NeRF框架，PanSplat转3DGS

### vs. Splatter-360 [18]
Concurrent work，也做panoramic 3DGS，但"does not address the unique challenges of high-resolution on real-world datasets"。PanSplat的deferred BP和4K真实数据generalization是核心优势。

参考：
- Splatter-360: https://arxiv.org/abs/2412.06250
- HiSplat [62]: https://arxiv.org/abs/2410.06245
- pixelSplat: https://github.com/dcharatan/pixelsplat

---

## 可能的follow-up方向

1. **Dynamic scene**：加入motion field，每个Gaussian学一个velocity（4DGS思路）
2. **More than 2 views**：当前只2 view，扩展到N views需要新的cost volume设计
3. **End-to-end joint training with depth estimation**：当前fine-tune真实数据要冻结cost volume，能否end-to-end？
4. **Latent space 3DGS**：在latent space而非像素空间做splatting，进一步省内存
5. **Adaptive Gaussian density**：Fibonacci是均匀的，能否根据scene复杂度自适应分布？
6. **Distortion-aware perceptual loss**：LPIPS为perspective设计，panorama上有distortion bias
7. **Nerf-friendly cubemap splatting**：当前cubemap渲染接缝可能有artifact，能否用spherical harmonics basis直接在球面splat？

---

## Final thoughts

PanSplat是典型"engineering-heavy"的paper，三个核心创新（Fibonacci lattice, hierarchical cost volume + 3DGP, two-step deferred BP）每一个单独看都不算breakthrough，但组合在一起让4K panoramic NVS第一次在单GPU上可行。这是"system-level innovation"而非"algorithm-level innovation"。

PanoGRF留下的23.8s/image的NeRF ceiling被PanSplat用70×加速打破，意味着VR panorama tour这种应用第一次有了"real-time + high-res"两个同时满足的可能性。后续如果有人做dynamic scene extension，整个VR tour pipeline就基本可用。

一句话概括PanSplat的核心idea：**承认panorama是球面，承认4K显存装不下，于是用spherical Fibonacci lattice让Gaussian均匀、用hierarchical cost volume让geometry精准、用deferred backprop让training可行**。三件事环环相扣，缺一不可。

---

**主要references**:
- PanSplat code: https://github.com/chengzhag/PanSplat
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- MVSplat: https://github.com/donydchen/mvsplat
- PanoGRF: https://github.com/chenchen-111/PanoGRF
- 3DGS ECCV 2024 (Kerbl et al.): https://arxiv.org/abs/2308.14737
- CasMVS: https://arxiv.org/abs/1912.06378
- Fibonacci Lattice: https://observablehq.com/@meetamit/fibonacci-lattices
- Swin Transformer: https://arxiv.org/abs/2103.14030
- FPN: https://arxiv.org/abs/1612.03144
- 360Loc dataset: https://arxiv.org/abs/2310.19029
- Matterport3D: https://arxiv.org/abs/1709.06158
- LPIPS: https://arxiv.org/abs/1801.03924
- xFormers: https://github.com/facebookresearch/xformers
- Splatter-360: https://arxiv.org/abs/2412.06250
- HiSplat: https://arxiv.org/abs/2410.06245
- GGRT: https://arxiv.org/abs/2408.15282
- IBRNet: https://arxiv.org/abs/2102.13090
- LatentSplat: https://arxiv.org/abs/2403.16292
- pixelSplat: https://github.com/dcharatan/pixelsplat
- DepthSplat [73]: https://arxiv.org/abs/2410.13862
- UniMatch [71]: https://arxiv.org/abs/2309.15131
- UniFuse [33]: https://arxiv.org/abs/2103.00182

---

# PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting 深度解析

## 1. 核心问题与动机

PanSplat 要解决的问题非常具体：从两张 **wide-baseline** 的 posed 全景图（panorama）合成新的视角。这件事的难点在于三个相互冲突的目标：

1. **Resolution**：VR 应用需要至少 4K (2048×4096)，因为人眼在 HMD 里靠得很近，低分辨率会立刻看到 pixel
2. **Inference speed**：NeRF 类方法（PanoGRF [17] 单图 23.8s）根本不能用于 VR
3. **Memory**：4K 全景图的 feature map、cost volume、Gaussian 参数同时驻留 GPU 显存会爆掉

PanoGRF 用 NeRF + volumetric sampling 的方式做 4K 完全做不到；MVSplat [16] 是 feed-forward 3DGS，但在 512×1024 分辨率下就 OOM。PanSplat 的核心 insight 是 **decouple geometry 和 appearance 的分辨率**，配合 **two-step deferred backpropagation**，让 4K 训练在单卡 A100 上可行，推理能在 24GB 的 RTX 3090 上跑。

Paper 链接：https://arxiv.org/abs/2412.05290 (推算)
Code: https://github.com/chengzhag/PanSplat

---

## 2. Equirectangular Geometry 的难点（intuition）

这是理解整篇 paper 的关键 intuition。Equirectangular projection 是把球面映射到矩形：

$$
\theta = \frac{x}{W} \cdot 2\pi, \quad \phi = \frac{y}{H} \cdot \pi - \frac{\pi}{2}
$$

其中 $x \in [0, W)$ 是水平像素坐标，$y \in [0, H)$ 是垂直像素坐标，$\theta$ 是经度，$\phi$ 是纬度。

问题在于球面上的面积元素：

$$
dA = R^2 \cos\phi \, d\theta \, d\phi
$$

靠近 poles ($\phi \to \pm\pi/2$) 时 $\cos\phi \to 0$，但 equirectangular 图上每个像素对应的 $d\theta \, d\phi$ 是固定的。所以 **poles 处像素密度被严重过采样**——很多像素描述同一个很小的球面区域。

MVSplat 之类的方法采用 **pixel-aligned Gaussian**（每个像素对应一个 Gaussian），结果就是 poles 处有海量冗余 Gaussian 互相重叠，浪费表达能力和算力。在 Fig. 2 的左下角可视化得很清楚。

---

## 3. Fibonacci Lattice Gaussian：核心创新之一

### 3.1 数学构造

Fibonacci lattice（也叫 Fibonacci sphere）是在球面上 quasi-uniform 采样点的经典方法。设要采 $n$ 个点，第 $j$ 个点 ($j=0,1,\dots,n-1$) 在 equirectangular 图上的坐标是：

$$
(x_j, y_j) = \left( \frac{j \cdot \phi \mod 1}{1}, \frac{j}{n-1} \right)
$$

其中 $\phi = \frac{1+\sqrt{5}}{2}$ 是 **golden ratio**（黄金比例，约 1.618）。

这里我详细解释一下每个符号：
- $j$：Gaussian 的 index，从 $0$ 到 $n-1$
- $x_j$：水平方向归一化坐标，$j \cdot \phi \mod 1$ 取小数部分，因为 $\phi$ 是无理数，$j \cdot \phi$ 的小数部分会均匀密布 $[0,1)$，这是 Weyl 等分布定理保证的
- $y_j$：垂直方向归一化坐标，$j/(n-1)$ 线性从 0 到 1（即从北极到南极均匀分布）
- $\mod 1$：取小数部分操作
- $n = \lfloor W^2/\pi \rfloor$：让 Gaussian 在 equator 附近的密度大致等同于像素密度（因为球面面积 $\approx 4\pi R^2$，而 equirectangular 矩形面积 $\propto W \cdot H = W \cdot W/2 = W^2/2$，所以 $W^2/\pi$ 让两者匹配）

### 3.2 Intuition

为什么 Fibonacci lattice 比 pixel-aligned 好？关键在于 golden ratio 让 $x_j$ 序列在 $[0,1)$ 上有最优的 low-discrepancy 性质——每新增一个点，它会填在前面的"最大空隙"里。这种均匀性正好对应球面上的面积权重 $\cos\phi$：Fibonacci lattice 通过线性的 $y_j$ 让每条纬线上点的密度自然适配 $1/\cos\phi$ 的需求。

Ablation 显示 +Fibo 单独把 Gaussian 数从 1049K 减到 668K（**减少 36.34%**），而 WS-PSNR 反而从 27.07 升到 27.86。这是 "免费午餐"——单纯改变采样策略就同时省算力又提质量。

参考链接：
- Fibonacci Lattices 可视化：https://observablehq.com/@meetamit/fibonacci-lattices
- Roberts (2020) "How to evenly distribute points on a sphere more effectively than the canonical Fibonacci Lattice": https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/

---

## 4. Spherical 3D Gaussian Pyramid：多尺度表示

### 4.1 层级定义

设 $L=4$ 个层级 ($l=0,1,2,3$)，每层 Gaussian 数量：

$$
n^l = \left\lfloor \frac{W^2}{2^l \cdot \pi} \right\rfloor
$$

- $l=0$（最细）：$n^0 = \lfloor W^2/\pi \rfloor$，最密集，捕获 high-frequency details
- $l=3$（最粗）：$n^3 = \lfloor W^2/(8\pi) \rfloor$，最稀疏，捕获 global structure
- 每升一层，Gaussian 数量大致减半

### 4.2 Residual Design（关键）

单纯有金字塔不行——paper 在 Sec. E 的 ablation 发现网络会 "lazy" 不利用多尺度信息。所以引入 **residual connection between adjacent levels**：

在 level $l$ 的 Gaussian head 里，输入是：
1. 当前 level 的 cost volume $\tilde{C}_i^1$（resize 过的）
2. 当前 level 的 input image $I_i$
3. 当前 level 的 feature map $F_i^l$
4. **up-sampled feature map from level $l+1$**: $\tilde{F}_i^{l+1}$ (up-sampled)

Gaussian head 预测的是 **残差**（residual），相当于从粗到细累积修正。这强制了 coarse-to-fine 的依赖，避免最粗 level 输出无意义的 Gaussian。

Ablation 显示 w/o 3DGP residual 让 WS-PSNR 从 28.81 跌到 28.14（-0.67 dB），而 w/o Hierarchical CV 跌到 26.95（-1.86 dB），证明 residual 和 hierarchical 是 **互相配合**的，单独任何一个都达不到 SOTA。

可视化在 Fig. E.1 中：Full model 在最粗 level 也能渲染出可识别的房间结构，w/o residual 的版本最粗 level 渲出来是噪声。

---

## 5. Hierarchical Spherical Cost Volume

这一块借鉴 Cascade Cost Volume (CasMVS [25]) 和 PanoGRF 的 spherical projection，但适配 panorama。

### 5.1 Feature Pyramid

- **FPN [46]** backbone 提取 4 级 feature pyramid $\{F_i^l\}_{l=0}^{L-1}$，通道数 $C^l \in \{128, 96, 64, 32\}$（粗到细）
- **Swin Transformer [48]** 加在 FPN 的最粗 level（l=3），带 **cross-view attention** 让两个输入 panorama 之间交换信息，这对 wide-baseline matching 至关重要
- 用 xFormers [36] 加速

### 5.2 Cost Volume Construction

在 $l=3$（最粗）初始化 cost volume：

1. 对 reference view $i$，采样 $D$ 个 inverse depth 候选 $d \in [d_{min}, d_{max}]$
2. 把另一个 view $1-i$ 的 feature warp 到 reference view 的坐标系（用 spherical projection [17, 42]）
3. 计算 correlation：$C_i^3 = \langle F_i^3, \text{warp}(F_{1-i}^3) \rangle$ (dot product)
4. 用 2D U-Net refine，融合 monocular depth features [33] 作为先验
5. softmax over depth dimension 得到 depth probability $p(d)$
6. 加权平均：$D_i^3 = \sum_d d \cdot p(d)$

### 5.3 Hierarchical Refinement

从 $l=3$ → $l=2$ → $l=1$（跳过 $l=0$）：

1. 把 $D_i^{l+1}$ up-sample 到 level $l$
2. 在 $D_i^{l+1}$ 附近的小范围内（$(d_{max}-d_{min})/2^{3-l}$）搜索 $D/2^{3-l}$ 个候选
3. 每个 level 用独立的 2D U-Net refine，加上 $D_i^{l+1}$ 作为 contextual input

跳过 $l=0$ 是为了在相似内存预算下达到 MVSplat 的 2× depth 分辨率。这种 trade-off 的 intuition 是：纹理需要全分辨率（appearance head 全分辨率），几何在 1/2 分辨率就足够。

公式里的变量：
- $D$：最粗 level 的 depth candidates 数（paper 设为 128）
- $D/2^{3-l}$：level $l$ 的 depth candidates 数（$l=3$ 时 128, $l=2$ 时 64, $l=1$ 时 32）
- $D_i^l$：level $l$ 的 depth prediction

参考：
- CasMVS: https://arxiv.org/abs/1912.06378
- MVSplat: https://arxiv.org/abs/2403.07607

---

## 6. Gaussian Heads 与 Cubemap Renderer

### 6.1 Gaussian Head 结构

每层一个轻量 CNN（3 个 3×3 卷积），输出 feature map $\tilde{F}_i^l$。然后基于 Fibonacci lattice 的坐标 $(x_j, y_j)$ 双线性插值出每个 Gaussian 的 feature vector，再用 FC 层预测参数：

$$
(\mu_i^l, \alpha_i^l, \Sigma_i^l, c_i^l) = \text{FC}(\text{interp}(\tilde{F}_i^l, x_j, y_j))
$$

参数含义：
- $\mu_i^l \in \mathbb{R}^3$：Gaussian center 3D 坐标，由 image-plane 坐标 $(x_j, y_j)$ + predicted depth unproject 得到
- $\alpha_i^l$：opacity，过 sigmoid
- $\Sigma_i^l$：covariance，由 scaling $s \in [s_{min}, s_{max}]$ 和 quaternion 组成。scaling 乘以 pixel size 让 Gaussian 大小与该位置像素大小匹配
- $c_i^l$：color，用 spherical harmonics 系数

### 6.2 Cubemap Renderer

为什么不直接 splat 到 equirectangular？因为 3DGS 原生 CUDA renderer 是基于 perspective camera 的。直接在 equirectangular 上做 rasterization 在 poles 处会有严重 distortion。

PanSplat 的方案：
1. 在同一位置放 6 个 90° FOV 的 perspective camera，分别朝 6 个 cubemap face 方向
2. 用原生 3DGS CUDA renderer 渲染 6 张 face 图
3. 用 differentiable grid sampling + bilinear interpolation 拼成 equirectangular panorama

关键工程细节：cube face 的边缘要 pad 4 个相邻 face 的边缘像素，保证 bilinear interpolation 在接缝处有正确邻居。

为什么 cubemap 而非直接 equirectangular splatting？两个原因：
1. 复用现有 3DGS CUDA renderer，避免重写
2. **sequential face rendering** 让 4K 渲染的中间结果不必同时驻留显存——这是 two-step deferred backpropagation 的基础

---

## 7. Two-step Deferred Backpropagation：内存关键

这是 paper 最有工程价值的创新，让 4K 训练成为可能。核心 insight 来自观察：**图像质量更依赖 texture resolution 而非 geometry resolution**。

### 7.1 Resolution Decoupling

- Hierarchical cost volume 只在 512×1024 跑（geometry 在低分辨率足够）
- Gaussian heads 在全分辨率跑（appearance 需要全分辨率纹理）
- 中间通过 up-sample 衔接

这让 MVSplat 在 512×1024 OOM 时，PanSplat 还能跑到 768×1536（Fig. 6）。但 4K 还是不够。

### 7.2 Forward + Cache + Two Backward

```
Forward Pass (auto-diff OFF):
  Cost volume + Gaussian heads → Gaussians → render full panorama → image loss
  Cache ∂L/∂I on the image (gradients on pixels)

Backward Step 1 (auto-diff ON):
  Re-render panorama FACE BY FACE (6 cubemap faces)
  For each face: backprop ∂L/∂face → ∂L/∂Gaussians (accumulate)

Backward Step 2 (auto-diff ON):
  Re-generate Gaussians TILE BY TILE (N×N tiles of Gaussian head inputs)
  For each tile: backprop ∂L/∂Gaussians → ∂L/∂Network params (accumulate)
```

为什么能这么做？因为：
1. Image loss 是对 pixels 求的，pixel gradients 只需要一次 forward 计算
2. Gaussian rendering 对 Gaussian 参数是 local operation（每个像素只受附近 Gaussian 影响），所以 face by face 渲染时梯度能正确累积
3. Gaussian head 是 local CNN operation（3×3 卷积），所以 tile by tile 时梯度也能正确累积——但需要在 tile 之间 padding 3 pixels 以避免边界 discontinuity

### 7.3 内存收益

Fig. G.1 显示：
- 无 deferred BP：4K 直接 OOM 在 80GB A100
- 4 tiles deferred BP：支持到 2048×4096
- 16 tiles deferred BP：训练显存进一步降低，4K inference 也能在 24GB RTX 3090 跑

注意 w/o Fibo 在 inference 时 1792×3584 就 OOM，但 PanSplat Full 还能撑——Fibo 帮忙减少了 Gaussian 数量，间接降低了渲染时的中间张量大小。

### 7.4 Tiled Operation 细节（工程 trick）

直接 split tile 会因为卷积 zero-padding 导致 tile 边界 discontinuity。PanSplat 的方案：
1. 先把 input pad 3 pixels（左右 wrap-around copy 保持球形连续性，上下 zero pad）
2. Tile 区域扩大 3 pixels，相邻 tile 之间有 3-pixel overlap
3. Crop output tile 到原大小后拼接

这样得到的 output 与 non-tiled 完全一致。

类似思路在 [43, 81] 也有，但 PanSplat 是第一个把它系统化到 panorama Gaussian splatting 的。

参考：
- GGRT [43]: https://arxiv.org/abs/2408.15282
- ARF [81]: https://arxiv.org/abs/2108.09089

---

## 8. Deferred Blending：处理动态物体

真实数据（360Loc, Insta360）里有 moving objects（行人、相机操作者、车辆），两个输入 view 的 depth 不一致，合并 Gaussians 会出现 ghosting artifacts。

解决方案：**不合并两个 view 的 Gaussians**，而是分别渲染：

$$
I = \frac{d_1 \tilde{I}_0 + d_0 \tilde{I}_1}{d_0 + d_1}
$$

变量解释：
- $\tilde{I}_i$：只用 view $i$ 的 Gaussians 在 target view 渲染的图
- $d_i$：target view 到 input view $i$ 的距离
- $I$：final blended image

Intuition：当 target view 离 view 0 近时（$d_0$ 小），公式中 $\tilde{I}_0$ 的权重 $\frac{d_1}{d_0+d_1} \to 1$，所以靠近 view 0 时几乎只用 view 0 渲染；这避免了 view 1 里的 moving object 干扰。当 target 在两 view 中间时，两者权重相等，blending 平滑过渡。

Table 2 显示在真实数据上提升明显：
- 360Loc WS-PSNR: 24.96 → 27.35 (+2.39 dB)
- Insta360 WS-PSNR: 24.43 → 25.68 (+1.24 dB)

在 frame distance 小的时候（即 target 离某个 input view 很近）提升最大（Fig. F.1）——正好对应 deferred blending 设计的目的。

参考 IBRNet [66] 也有类似 idea：https://arxiv.org/abs/2102.13090

---

## 9. Training Loss 详解

### 9.1 Synthetic data（有 GT depth）

**Depth loss**:

$$
\mathcal{L}_{depth} = \sum_{i=0,1} \sum_{l=1}^{3} \gamma^{l-1} \left\| D_i^l - \hat{D}_i^l \right\|_1
$$

变量：
- $i \in \{0,1\}$：两个 input views
- $l \in \{1,2,3\}$：三个 cost volume 层级（不含 $l=0$ 因为最细 level 不构造 cost volume）
- $\gamma = 0.9$：衰减因子，$l=1$ 时 $\gamma^0 = 1$（最细最重），$l=3$ 时 $\gamma^2 = 0.81$（最粗最轻）
- $D_i^l$：level $l$ 的预测 depth
- $\hat{D}_i^l$：GT depth 下采样到 level $l$

**RGB loss**:

$$
\mathcal{L}_{rgb} = \| I - \hat{I} \|_2 + \lambda \cdot \text{LPIPS}(I, \hat{I})
$$

变量：
- $I$：渲染图
- $\hat{I}$：GT 图
- $\| \cdot \|_2$：L2 范数（MSE）
- $\lambda = 0.1$：LPIPS 权重
- LPIPS [83]：perceptual loss 用 VGG features

总 loss：$\mathcal{L}_{synthetic} = \alpha \mathcal{L}_{depth} + \mathcal{L}_{rgb}$，$\alpha = 0.05$（depth loss 权重低，主要靠 RGB loss）。

### 9.2 Real data（无 GT depth）

用 **auxiliary Gaussian heads** 在每个 cost volume level 单独渲染 + 监督：

$$
\mathcal{L}_{real} = \sum_{l=1}^{3} \gamma^{l-1} \mathcal{L}_{rgb}(I^l, \hat{I}) + \mathcal{L}_{rgb}(I, \hat{I})
$$

变量：
- $I^l$：从 level $l$ 的 auxiliary head 渲染的 panorama
- $I$：主 Gaussian head 渲染的 panorama
- 第一项：supervise 每个 level 的 cost volume 通过可微分渲染
- 第二项：supervise 最终输出

这是 self-supervised depth learning via 3DGS 的思路（[16, 67] 也用），把 depth supervision 转换成 image supervision。Auxiliary heads 只有 2 个 CNN 层（很轻），不和主 head 共享 residual design。

---

## 10. 实验结果深度分析

### 10.1 Table 1：Synthetic 上的对比

| Method | 1.0m WS-PSNR | 1.5m | 2.0m | Replica | Residential |
|--------|--------------|------|------|---------|-------------|
| S-NeRF [49] | 15.25 | 14.16 | 13.13 | 16.10 | 22.47 |
| OmniSyn [42] | 22.90 | 20.31 | 18.91 | 23.17 | — |
| IBRNet [66] | 25.72 | 21.69 | 20.04 | 22.65 | 22.47 |
| NeuRay [47] | 24.92 | 21.92 | 19.85 | 25.90 | 22.38 |
| PanoGRF [17] | 27.12 | 23.38 | 20.96 | 29.22 | 31.03 |
| MVSplat [16] | 28.19 | 21.82 | 13.31 | 30.54 | 31.21 |
| **PanSplat** | **28.81** | **24.09** | 20.56 | 30.78 | 30.97 |

关键观察：
1. **Wide baseline 退化**：MVSplat 在 2.0m 时崩到 13.31（和 S-NeRF 一个量级），原因是 MVSplat 是为 perspective 设计，cost volume 在 wide baseline 下匹配失败；PanSplat 的 hierarchical cost volume 让 wide baseline 还能保持 20.56
2. **1.0m**：PanSplat 比 MVSplat 高 0.62 dB，比 PanoGRF 高 1.69 dB
3. **2.0m**：PanSplat 比 PanoGRF 略低 0.40 dB，但比 MVSplat 高 7.25 dB
4. **Replica & Residential**：泛化能力相近，PanSplat 在 Replica 略胜，Residential 略输（但仍可竞争）——说明 model 在 unseen scene structure 上泛化好

### 10.2 Table 2：Real data 对比

| Dataset | Method | PSNR | WS-PSNR | SSIM | LPIPS |
|---------|--------|------|---------|------|-------|
| 360Loc | MVSplat | 24.13 | 24.67 | 0.823 | 0.170 |
| 360Loc | PanSplat | 24.96 | 25.58 | 0.833 | 0.159 |
| 360Loc | PanSplat+BL | **27.35** | **28.14** | **0.860** | **0.127** |
| Insta360 | MVSplat | 20.93 | 23.24 | 0.786 | 0.227 |
| Insta360 | PanSplat | 21.92 | 24.43 | 0.813 | 0.211 |
| Insta360 | PanSplat+BL | **23.36** | **25.68** | **0.822** | **0.183** |

Deferred blending 带来 ~2-3 dB 的提升，这是巨大的——印证了 moving objects 是真实数据的主要 artifact 来源，deferred blending 巧妙绕过。

### 10.3 Table 3：主 Ablation

| Setup | #Gaussians | WS-PSNR | SSIM | LPIPS |
|-------|------------|---------|------|-------|
| Base | 1049K (100%) | 27.07 | 0.895 | 0.127 |
| +Fibo | 668K (63.67%) | 27.86 | 0.906 | 0.116 |
| +3DGP (Full) | 887K (84.55%) | 28.81 | 0.931 | 0.091 |

Intuition：
- Base → +Fibo：Gaussian 减 36%，质量升 0.79 dB
- +Fibo → +3DGP：Gaussian 加 33%，质量再升 0.95 dB
- 最终 Gaussian 数 887K，仍比 Base 少 15.45%，质量高 1.74 dB

### 10.4 Table E.1：Design Ablation

| Setup | WS-PSNR |
|-------|---------|
| w/o Mono depth | 28.84 |
| w/o 3DGP residual | 28.14 (-0.67) |
| w/o Hierarchical CV | 26.95 (-1.86) |
| w/o First 3 GHs | 28.05 (-0.76) |
| Full | 28.81 |

观察：
- Mono depth 贡献微弱（0.03 dB），但保留以追求极致性能
- Hierarchical CV 是最重要的单一设计（-1.86 dB），因为没有它 cost volume 在 wide baseline 上完全失败
- 3DGP residual 强制 coarse-to-fine，没有它最粗 level 学不出来
- First 3 GHs（即只用最细 level）掉 0.76 dB，说明多尺度确实有用，但 hierarchical CV 的几何准确性更基础

### 10.5 Table D.1：Narrow baseline 对比

| Method | 0.2m WS-PSNR | 0.5m WS-PSNR |
|--------|--------------|--------------|
| PanoGRF | **34.29** | 31.41 |
| MVSplat | 32.93 | 31.55 |
| PanSplat | 33.92 | **32.46** |

在 0.2m 窄 baseline PanoGRF 反而最好。Intuition：PanoGRF 用 dense cost volume + 双目 + 单目融合，在窄 baseline 时匹配容易，dense sampling 充分利用信息；PanSplat 跳过最细 cost volume 层级，在窄 baseline 下精度上限受限于这个设计选择。但 0.5m 以上 PanSplat 全面领先，wide baseline 才是实际场景（VR tour）的真实需求。

### 10.6 Speed Comparison

- PanoGRF: 23.8s/image（NeRF volumetric rendering）
- PanSplat: 0.34s/image（0.32s forward + 0.02s 3DGS rasterize）
- **70× 加速**

这是 3DGS 相比 NeRF 的天然优势——rasterization vs volumetric sampling 的本质差异。

---

## 11. 网络架构细节（Supp. Sec. A）

补充一些关键数字：
- **FPN + Swin**：Swin 6 个 Transformer block，每个 block = self-attention + cross-view attention
- **Feature pyramid channels**: $C^l \in \{128, 96, 64, 32\}$（粗→细）
- **2D U-Net**: 3 个 level，channels $\{128, 64, 32\}$，bottleneck 处加 cross-view attention
- **Depth candidates**: $D \in \{128, 64, 32\}$（粗→细）
- **Gaussian head**: 3 个 3×3 conv，stride 1
- **Covariance**：scaling = normalized vector $s \in [s_{min}, s_{max}]$ × pixel size，quaternion 表示 rotation
- **Color**：spherical harmonics

Scaling 设计的 intuition：Gaussian 大小与对应位置的像素大小匹配，避免不同 latitude 处 Gaussian 看起来"一样大"造成的渲染问题。

---

## 12. 4K Training Pipeline（Supp. Sec. C）

训练 schedule 很 informative：
1. **Stage 1**：Matterport3D @ height 256，batch 6，10 epochs
2. **Stage 2**：fine-tune @ height 512，batch 2，5 epochs
3. **Stage 3**：4K Matterport3D fine-tune，progressive height 1024 → 2048，每个 stage 3 epochs
4. **Real fine-tune**：360Loc @ height 512 → 1024 → 2048，iterations 65K/26K/13K

1024 和 2048 阶段启用 deferred BP，分别用 4 tiles 和 16 tiles，batch size 3 和 1。

关键 trick：fine-tune 真实数据时**冻结 hierarchical cost volume**，只调 Gaussian heads——这避免了在没有 GT depth 时 cost volume 漂移，同时让 Gaussian heads 适应真实图像的纹理分布。

---

## 13. 关键 Limitations

Paper 自己承认：**不支持动态场景**（moving objects 是单帧画面中的问题，但 scene-level 的动态变化未处理）。这指向 future work 可以融合 motion-aware representations（如 dynamic 3DGS, 4DGS）。

我额外想到的几个 limitation：
1. **Wide baseline 退化**：2.0m 时 PanSplat 比 PanoGRF 略低，说明 hierarchical 设计在极宽 baseline 下还有改进空间
2. **GT depth 依赖**：合成数据训练需要 GT depth，限制了可用数据集
3. **Cubemap renderer 边缘**：虽然在 cubemap 接缝处做了 padding，但极端 high-frequency 区域可能仍有 artifact
4. **Inference tiling**：虽然支持，但 tiled operation 增加了 overhead，4K 实时 VR 可能还是吃力

---

## 14. 与相关工作的 broader 思考

### 14.1 vs. MVSplat

MVSplat 是 SOTA 的 perspective feed-forward 3DGS 方法。PanSplat 与其对比：
- **Geometry 部分**：MVSplat 用单层 cost volume（128 depth candidates），PanSplat 用 3 层 hierarchical（128+64+32）
- **Gaussian placement**：MVSplat pixel-aligned，PanSplat Fibonacci
- **Multi-scale**：MVSplat 单尺度，PanSplat pyramid
- **Memory**：MVSplat 512×1024 就 OOM，PanSplat 4K 训练

PanSplat 等于把 MVSplat 的 idea 重新设计成 panorama-aware + scalable。

### 14.2 vs. PanoGRF

PanoGRF 是 panoramic NeRF 的 SOTA：
- **质量**：PanSplat 高 1.69 dB (1.0m)
- **速度**：PanSplat 快 70×
- **Resolution**：PanoGRF 最多 512×1024，PanSplat 4K
- **Architecture**：PanoGRF 还在 NeRF 框架（volumetric sampling），PanSplat 转 3DGS

### 14.3 vs. Splatter-360 [18]

Concurrent work，也做 panoramic 3DGS，但 paper 指出 "does not address the unique challenges of high-resolution on real-world datasets"。PanSplat 的 deferred BP 和 4K 真实数据 generalization 是它的核心优势。

参考链接：
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- MVSplat: https://github.com/donydchen/mvsplat
- PanoGRF: https://github.com/chenchen-111/PanoGRF
- pixelSplat: https://github.com/dcharatan/pixelsplat
- Splatter-360: https://arxiv.org/abs/2412.06250

### 14.4 Hierarchical Gaussian 这条线

HiSplat [62] 是 concurrent work，也探索 hierarchical 3DGS，但 PanSplat 强调 "does not address the unique challenges of high-resolution or panoramic formats"。这是 "general idea + specific engineering for panorama" 的组合拳。

### 14.5 Fibonacci Lattice 在 graphics 里的应用

Fibonacci sphere 在 rendering、sampling、distribution 里有悠久历史。代表性 reference:
- Frisch & Hanebeck [24]: https://ieeexplore.ieee.org/document/9464239
- Roberts 2020: https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/

PanSplat 把这个 idea 用到 3DGS Gaussian placement 上是新颖的工程组合。

---

## 15. 我自己的整体 intuition 总结

PanSplat 的成功可以归结为三件事的协同：

1. **Geometry/Appearance Decoupling**：geometry 在 512×1024 跑，appearance 在 4K 跑。这是 single A100 训练 4K 的根本前提。Intuition：纹理高频，几何低频——人眼看 scene 不会因为 geometry 略糙而感觉差，但纹理糊立刻能看出来。

2. **Spherical-aware Sampling**：Fibonacci lattice + spherical cost volume。把 panorama 当作球面而非拉伸的矩形来处理。Intuition：如果我们承认 panorama 是球面的，那所有 pixel-aligned 操作都是有 bias 的——Fibonacci lattice 是消除这个 bias 的最简洁方式。

3. **Deferred Computation**：two-step deferred backpropagation + cubemap renderer + tiled Gaussian heads。这三件事配合才能让 4K 的中间张量不必同时驻留显存。Intuition：deferred BP 本质上是把 "存储所有中间结果" 替换成 "重新计算 + 梯度累积"，用 compute 换 memory，在 4K 这种极端 case 下 compute 还是比 memory 便宜。

这三点层层递进——单做任何一个都不够。比如只做 resolution decoupling，PanSplat 在 768×1536 就 OOM 了；加上 deferred BP 才能到 4K。这就是 paper 的"组合拳"价值。

但 paper 也坦白说，这种 deferred BP 让 training 复杂度上升，工程实现难度大。这是这类 work 的固有 trade-off。

---

## 16. 可能的 follow-up 方向

1. **Dynamic scene**：加入 motion field，每个 Gaussian 学一个 velocity（4DGS 思路）
2. **More than 2 views**：当前只 2 view，扩展到 N views 需要新的 cost volume 设计
3. **End-to-end joint training with depth estimation**：当前 fine-tune 真实数据要冻结 cost volume，能否 end-to-end？
4. **Latent space 3DGS**：在 latent space 而非像素空间做 splatting，进一步省内存
5. **Adaptive Gaussian density**：Fibonacci 是均匀的，能否根据 scene 复杂度自适应分布？
6. **Distortion-aware perceptual loss**：LPIPS 是为 perspective 设计，panorama 上有 distortion bias，可以设计 sphere-aware perceptual loss
7. **Nerf-friendly cubemap splatting**：当前 cubemap 渲染接缝可能有 artifact，能否用 spherical harmonics basis 直接在球面 splat？

---

## 17. Final thoughts

PanSplat 是一篇非常 "engineering-heavy" 的 paper，三个核心创新（Fibonacci lattice, hierarchical cost volume + 3DGP, two-step deferred BP）每一个单独看都不算 breakthrough，但组合在一起让 4K panoramic NVS 第一次在单 GPU 上可行。这是典型的 "system-level innovation" 而非 "algorithm-level innovation"。

PanoGRF 留下的 23.8s/image 的 NeRF ceiling 被 PanSplat 用 70× 加速打破了，这意味着 VR panorama tour 这种应用第一次有了"real-time + high-res"两个同时满足的可能性。后续如果有人做 dynamic scene extension，整个 VR tour pipeline 就基本可用了。

如果让我用一句话概括 PanSplat 的核心 idea：**承认 panorama 是球面，承认 4K 显存装不下，于是用 spherical Fibonacci lattice 让 Gaussian 均匀、用 hierarchical cost volume 让 geometry 精准、用 deferred backprop 让 training 可行**。三件事环环相扣，缺一不可。

---

**主要 references**:
- PanSplat code: https://github.com/chengzhag/PanSplat
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- MVSplat: https://github.com/donydchen/mvsplat
- PanoGRF: https://github.com/chenchen-111/PanoGRF
- 3DGS ECCV 2024 (Kerbl et al.): https://arxiv.org/abs/2308.14737
- CasMVS: https://arxiv.org/abs/1912.06378
- Fibonacci Lattice: https://observablehq.com/@meetamit/fibonacci-lattices
- Swin Transformer: https://arxiv.org/abs/2103.14030
- FPN: https://arxiv.org/abs/1612.03144
- 360Loc dataset: https://arxiv.org/abs/2310.19029
- Matterport3D: https://arxiv.org/abs/1709.06158
- LPIPS: https://arxiv.org/abs/1801.03924
- xFormers: https://github.com/facebookresearch/xformers
- Splatter-360: https://arxiv.org/abs/2412.06250
- HiSplat: https://arxiv.org/abs/2410.06245
- GGRT: https://arxiv.org/abs/2408.15282
