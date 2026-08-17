---
source_pdf: SPLATFORMER.pdf
paper_sha256: fc6db5f3b8a278ee3050655981995c2a79f9e02cb24bcf905ed7d1974a1d7ff2
processed_at: '2026-08-12T10:07:26-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 SplatFormer

## 这篇 paper 在干嘛

3D Gaussian Splatting（3DGS）牛吧，渲染快、效果炸。但有个尴尬问题没人愿意提：**你只在平地绕着雕像拍一圈，训练出来的模型，从顶上看就崩了**。满屏的尖刺、飞絮、破洞。

为啥呢？因为 3DGS 优化的时候只盯着训练视角的 pixel loss。每个 Gaussian splat 只需要"在训练视角下看起来对"就行。所以它学会了**糊弄训练视角**——用又扁又长的椭圆 splat 贴在表面侧面，从前面看是一片平滑，从顶上俯视就变成一根根刺。

这个问题叫 **OOD-NVS**（out-of-distribution novel view synthesis）。简单说就是：训练视角和测试视角分布差太远，3DGS 没见过这种角度，直接破功。

## SplatFormer 的思路

作者的 idea 简单粗暴又优雅：

> 你不是训练出了一个有毛病的 3DGS 吗？那我再训一个 transformer，**专门帮你修这个毛病**，一次前向搞定。

具体流程：
1. 先用老方法优化出 initial 3DGS（有 artifacts 那个）
2. 把每个 Gaussian splat 当成一个"点"，splat 的属性（位置、大小、不透明度、颜色 SH、旋转）当成这个点的 feature
3. 扔进一个 Point Transformer（基于 PTv3），让 attention 机制去看每个 splat 周围的邻居
4. Transformer 输出每个 splat 属性的"修正量"（residual），加回去得到 refined 3DGS
5. 用这个 refined 版本去渲染 OOD 视角，artifact 就没了

**一个 forward pass 108ms**，不用 per-scene 重新优化，直接修好。

## 为什么这样 work

几个关键 design choice，每个都有 intuition：

### 1. 为什么用 Point Transformer 而不是 CNN 或 MLP

3DGS 的 splats 就是一堆**杂乱无章、分布不均的 3D 点**——表面密、空气稀。这不是 grid，不是 image，是 point cloud。Point Transformer 的 attention 机制正好擅长处理这种 irregular 结构，每个 splat 动态地看周围邻居，"你这个 splat 周围都是合理的 surface，但你突出来一根刺，不对，给我压回去"。

### 2. 为什么学 residual 而不是直接学完整的 splat

因为大部分 splat 其实是对的，只有少数是 artifact。**直接预测容易把好的也搞坏**，学 residual 等于告诉网络"默认啥也别改，只在需要的地方动一下"。再加上 zero-init 最后一层，网络刚开始训练时输出 = 输入，是 identity，训练超级稳定。

实验数据：residual vs direct，PSNR 差 **1.7 dB**，差距很明显。

### 3. 为什么用 2D rendering loss 而不是直接监督 3D 属性

这个 ablation 很有意思。作者试过用"完美 3DGS"做 3D label 直接监督，结果**3D loss 能 minimize，但 2D 渲染质量不涨**。为什么？神经网络的 spectral bias——学低频容易学高频难。3D attribute 小误差在 splatting 渲染时被放大，照样出 artifact。

所以最终目标是什么？**是 2D 图好看**，那就直接 optimize 2D loss，别整虚的。

### 4. 为什么能 generalize 到没见过的数据

SplatFormer 只在 synthetic 数据（ShapeNet + Objaverse）上训练，但能迁移到 GSO 真实扫描物体 + iPhone 拍的真实场景，PSNR 照样涨。说明它学到的不是"这是个椅子"这种 object prior，而是"3D splat 应该长成什么样才合理"这种**抽象的几何合理性 prior**。这个就很强了，因为意味着可以 scale。

## 结果有多炸

ShapeNet-OOD 上：3DGS 是 20.21 dB，SplatFormer 是 **27.98 dB**，提升 **7.77 dB**。这什么概念，一般 NVS paper 提升 0.5 dB 就能开香槟了。

对比其他方法：
- 2DGS（加几何正则）：23.52
- SplatFields（Deep Image Prior 风格）：23.15
- LaRa（feed-forward 4 view）：20.94
- 用 DiffBIR 做 2D denoising 再 retrain 3DGS：25.16

SplatFormer 都碾压。**2D denoising 路线输得很彻底**——单帧好看但 multi-view 不一致，retrain 3DGS 也救不回来。3D space 直接 refine 才是正解。

## 我看完的几个 takeaway

1. **OOD-NVS 这个 problem setting 本身就值得单独拎出来研究**，之前大家都在搞 sparse view 或者 in-distribution interpolation，这个"训练视角偏一边，测试视角偏另一边"的真实场景被忽视了。

2. **Refinement 比 from-scratch prediction 容易得多**。feed-forward 模型（LaRa、PixelSplat）卡在 4 个 view 的 memory 限制上，而 SplatFormer 绕开这个限制——先让 3DGS optimization 把 32 个 view 信息压进 splats，再用 transformer refine。这是 division of labor 的胜利。

3. **Point Transformer 在 3DGS 上第一次 work 这件事本身意义大**。之前 3DGS refinement 都靠 hand-crafted 正则（2DGS 的 normal consistency、SplatFields 的 DIP），现在终于有人证明了 learned prior 能 work，且能 scale。

4. **Limitations 也很诚实**：高频细节还是不如 MipNeRF360（虽然 MipNeRF360 训练 7 小时、渲染 <1fps，不实用）；unbounded scene 的 foreground 会被 normalization 缩小；只学了低仰角 input 的 prior，反过来不行。这些其实都是下一步的方向。

5. **更深层的隐喻**：这玩意儿很像 LLM 里"先 draft 再 refine"的 inference-time scaling 思路。与其让 3DGS optimization 自己 struggle 找全局最优，不如先快速生成 draft，再让 transformer 做一次 refinement pass。这个 pattern 可能会成为 3D 重建 pipeline 的标准组件，就像 ResNet 里的 skip connection 一样自然。

总之这篇 paper 最让我舒服的地方是它的**简洁**：train a transformer to refine 3DGS, supervised by rendering loss。没有花里胡哨的 trick，但 first-principles 上完全正确。

---

# SplatFormer 深度解析：用 Point Transformer 修复 3DGS 的 OOD 视角伪影

Karpathy 你好，这篇 paper 解决的问题非常 fundamental 且之前大家都避而不谈：3DGS 在训练视角分布之外（OOD）的视角下渲染质量严重退化。我把它拆成「问题定义 → 方法直觉 → 架构细节 → 公式推导 → 实验数据 → 我的 critique」来讲，build 一下你的 intuition。

---

## 1. 核心问题：为什么 3DGS 在 OOD 视角会崩

### 1.1 Setup 描述

想象在 museum 拍雕像，用户绕物体走一圈，相机 elevation 大致接近水平（低仰角），azimuth 覆盖 360°。但 AR/VR 应用里用户想从俯视角看（high elevation），此时新视角偏离训练分布很大。

Paper 定义了 **OOD-NVS** 这个 task：
- Input views: $N_\text{in} = 32$ 张，elevation $\phi \in (0, \phi_\text{max})$，$\phi_\text{max} \in \{10°, 20°\}$
- OOD test views: $N_\text{out} = 9$ 张，elevation $\phi_\text{ood} \geq 70°$（俯视）

这与 prior 的 Sparse NVS（只给 4 张 input）和 Nerfbusters（视角分布类似但只是 unobserved region）都不同——OOD-NVS 是大视角偏差，但物体其实在低仰角是被 dense 覆盖的。

### 1.2 3DGS 为何崩塌：Intuition

3DGS 的每个 splat $\mathcal{G}_k$ 是一个 3D 高斯，参数化为：
- Mean position $\mathbf{p}_k \in \mathbb{R}^{3\times 1}$：3D 中心
- Opacity $\alpha_k \in [0,1]$
- Spherical harmonics $\mathbf{a}_k \in \mathbb{R}^S$：view-dependent color $\mathbf{c}_k$
- Scale $\mathbf{s}_k \in \mathbb{R}^3$ + Rotation $\mathbf{q}_k \in \mathbb{R}^4$（quaternion）→ 协方差 $\Sigma_k \in \mathbb{R}^{3\times 3}$

**Key observation**：3DGS 优化时只是 minimize 输入 view 的 photometric loss，没有任何机制约束 splats 在没见过的视角下也合理。结果就是：

1. **Elongated splats**：当一个 surface 只在某一侧被看到，3DGS 会用沿观察方向拉伸的薄 splat 来"糊"像素。从 input view 看是平的，但从侧视/俯视看就是一根尖刺（spike）。
2. **Floaters / unordered geometry**：优化早期生成的 floaters 在 input view 上恰好被其他 splats 遮挡，loss 看不见它们，但 OOD 视角下它们就暴露了。
3. **Surface undersampling**：低仰角 input 看不到顶面，3DGS 不会主动"补"顶面 splats，俯视就直接看穿。

Fig. 2 的曲线很说明问题：随 elevation 增大，PSNR 从 ~24 跌到 ~15，跌出悬崖。

---

## 2. SplatFormer 方法的核心 Intuition

Paper 提出三个 design principle：
1. **Leverage generic priors** from large-scale datasets（不要 hardcode geometry constraints）
2. **3D consistency** in renderings（不要在 2D space 上 denoise）
3. **Fully utilize** 所有 input view 的几何信息（不要像 feed-forward 模型只支持 4 个 view）

### 2.1 高层架构（Fig. 3）

Pipeline：
```
Input Images → 3DGS optimization (Sec.3) → Initial 3DGS (biased)
                                              ↓
                                    SplatFormer (point transformer)
                                              ↓
                                      Refined 3DGS
                                              ↓
                                    Render OOD views
```

关键点是 **feed-forward refinement**：SplatFormer 是一个 pre-trained network，对每个新场景只需要一次 forward pass（108ms，900MB GPU），不需要 per-scene optimization。

### 2.2 公式 (4)–(6) 数学描述

Encoder：
$$\{\mathbf{v}_k\}_{k=1}^K = f_\theta(\{\mathcal{G}_k\}_{k=1}^K)$$

其中：
- $\{\mathcal{G}_k\}_{k=1}^K$：初始 Gaussian splat 集合，$K$ 是 splat 数量
- $\mathbf{v}_k \in \mathbb{R}^V$：每个 splat 的 abstract feature，$V=96$
- $f_\theta$：PTv3-based encoder，参数 $\theta$

Decoder（residual prediction）：
$$\{\Delta \mathcal{G}_k = (\Delta \mathbf{p}_k, \Delta \mathbf{s}_k, \Delta \boldsymbol{\alpha}_k, \Delta \mathbf{q}_k, \Delta \mathbf{a}_k)\}_{k=1}^K = g_\theta(\{\mathcal{G}_k, \mathbf{v}_k\}_{k=1}^K)$$

输出 5 个 residual：position、scale、opacity、quaternion（rotation）、SH coefficients。

Refinement：
$$\{\mathcal{G}'_k\}_{k=1}^K = \{\mathcal{G}_k + \Delta \mathcal{G}_k\}_{k=1}^K$$

**Intuition**：residual learning 比 direct prediction 好得多（ablation Tab.4: 23.06 vs 21.36），因为大部分 splats 已经基本正确，只有少数需要修正。Zero-init final layer 让初始输出 = 输入，训练初期就稳定。

---

## 3. 架构细节

### 3.1 Point Transformer Encoder (PTv3 based)

PTv3 ([Wu et al., 2024](https://github.com/PointTransformerV3/PTv3)) 的核心思想：用 space-filling curve（如 Hilbert curve）serialize 3D points 成 1D 序列，然后用 1D attention 在序列局部窗口内做 attention，避免 KD-tree 的不可微问题。

SplatFormer 的具体配置：
```
Embedding (MLP) → 5 down-pooling stages → 4 up-pooling stages → V=96 features
Down stages: attention blocks (2,2,2,6,2), hidden (64,96,128,256,512)
Up stages:   attention blocks (2,2,2,2),  hidden (256,128,96,96)
Grid resolution: 384, pooling strides: (1,2,2,2)
```

每个 stage = LayerNorm + Multi-head Attention + MLP（标准 transformer block）。Grid pooling 把空间划分成 voxel，每个 voxel 内做 pooling 减少点数。

**为什么用 point transformer 而不是 sparse conv 或 MLP**：
- PointNet/MLP：每个点独立处理，无法建模 splat 之间的空间关系
- Sparse convolution（Minkowski）：感受野固定，对 OOD 这种需要"理解整个物体形状"的任务不灵活
- Attention：动态感受野，可以根据局部几何自动聚合邻域信息

### 3.2 Feature Decoder

5 个独立 MLP branches，分别预测 5 种 residual：
```
Each branch: 4 linear layers, hidden=512, ReLU (last layer: Tanh)
Tanh on residual mean: normalize to [0,1] (matches normalized positions)
Final layer weights & biases: zero-initialized
```

**为何分 5 个 branch 而不是统一 MLP**：每种 attribute 的 scale 和语义差异大（position vs opacity vs SH），独立 branch 让模型可以学习不同的 refinement 策略。

### 3.3 总参数量与计算

- ~50M parameters
- 单次 forward: 108ms（RTX 4090），900MB GPU
- 单场景 input Gaussians: 70k–100k
- 最多支持 4M Gaussians（取决于空间分布的 entropy）

---

## 4. 训练数据与 Objective

### 4.1 数据集构建

- ShapeNet: 33k objects, diffuse lighting
- Objaverse 1.0: 48k objects, specular + shadows
- 对每个 object 渲染 32 张低仰角 view + 5 张俯视 view
- 对每张低仰角 view 用 gsplat ([Ye et al. 2024](https://github.com/nerfstudio-project/gsplat)) 优化初始 3DGS（10k steps，~3 分钟/场景）
- 总共 ~3000 GPU hours（RTX 2080Ti）

### 4.2 Loss

Eq. 7:
$$\mathcal{L}_\text{SplatFormer} = \mathcal{L}_1 + \mathcal{L}_\text{LPIPS}$$

每次 iteration 渲染 4 张 target images：70% OOD views + 30% input views。这个 mix 很关键——纯 OOD 训练会让 in-distribution 性能下降，纯 in-distribution 学不到 OOD 泛化。

### 4.3 为什么 2D supervision 而不是 3D supervision（Fig. C.2）

Paper 做了个 ablation：用 optimal 3DGS（用全部 56 个 view 训练）作为 3D label，直接 supervise 3D attribute 的 L1 误差。结果：3D loss 能 minimize，但 2D rendering 质量不提升！

**Intuition**：Neural network 有 spectral bias ([Rahaman et al., 2019](https://arxiv.org/abs/1806.08734))，学低频容易、学高频难。即使 3D attribute 的 L1 误差很小，剩余的高频误差在 rendering 时被 volume splatting 放大，仍会产生可见 artifacts。2D supervision 直接 optimize rendering 结果，更符合下游目标。

---

## 5. 实验数据深度分析

### 5.1 Main Results (Table 1)

ShapeNet-OOD（$\phi_\text{ood} \geq 70°$）：

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 3DGS | 20.21 | 0.763 | 0.242 |
| 2DGS | 23.52 | 0.863 | 0.188 |
| SplatFields | 23.15 | 0.850 | 0.185 |
| LaRa | 20.94 | 0.839 | 0.222 |
| **SplatFormer** | **27.98** | **0.920** | **0.136** |

PSNR 提升 +7.77 dB，这是巨大 gap。Objaverse-OOD：23.06 vs 19.24 (3DGS)。

### 5.2 Cross-dataset Generalization (Table 2)

SplatFormer 只在 Objaverse 上训练，测试 GSO 和 real-world iPhone 数据：
- GSO-OOD: 25.01 PSNR（3DGS 21.78，+3.23 dB）
- Real-World-OOD: 24.33 PSNR（3DGS 23.83，+0.50 dB）

这说明学到的是 generic splat refinement prior，不是 object-specific prior。

### 5.3 2D vs 3D Refinement (Table 3)

对比 DiffBIR（state-of-the-art 2D image restoration）：
- 3DGS: 20.21
- DiffBIR stage1: 24.81
- DiffBIR stage2 + retrain 3DGS: 25.16
- SplatFormer: 28.09

**Intuition**：2D denoising 在单帧上看着不错，但 multi-view 不一致。即使 retrain 3DGS 也只能 partially 修复。3D refinement 直接在 splat 空间操作，本质上保证了 multi-view consistency。

### 5.4 Ablation (Table 4)

Objaverse-OOD:
- PTv3 + Residual: 23.06（best）
- Minkowski + Residual: 22.67
- PTv3 + Direct: 21.36

PTv3 比 Minkowski 略好（+0.39 dB），但 Residual vs Direct 差距更大（+1.70 dB）。说明 residual learning 是关键 design choice。

### 5.5 Geometry Evaluation (Table D.1)

虽然 paper 主要 focus NVS，但顺便也改善了 geometry：
- Depth MAE: 3DGS 6.70e-4 → SplatFormer 4.05e-4
- Normal MAE: 3DGS 0.239 → SplatFormer 0.214

### 5.6 各种 elevation 评估 (Table E.1)

在 GSO-OOD 上测了 $\phi \in [20°, 90°]$ 和 $R \in [0.2, 1.0]$：
- 高仰角 ($\phi=90°$) 3DGS 21.66 → SplatFormer 25.03（+3.37 dB）
- 极端 close-up ($R=0.2$) 也改善（+1 dB across elevations）

---

## 6. Limitations（paper 自己承认的）

1. **High-frequency texture**：still 不如 MipNeRF360 的细节（Fig. G.1），因为 PTv3 的 grid pooling 限制了 high-freq 表达
2. **Real-world generalization**：训练数据全是 synthetic，specular/复杂光照处理不够
3. **Diverse input distribution**：如果训练时是低仰角 input，测试时反着给高仰角 input + 低仰角 test，效果有限（Tab. H.1: +0.21 dB only）。说明 prior 是 input distribution-specific 的
4. **Unbounded scenes**：MVImgNet 上做实验（Tab. H.2），能去掉 floaters 但 foreground 几何 refine 困难——因为 normalization 把 foreground 缩小了

---

## 7. 我的深度 Critique 与联想

### 7.1 这篇 paper 真正解决的是什么

表面上：OOD-NVS 的 artifacts。本质上：**3DGS 的 over-fitting to training ray distribution**。这和 NeRF 时代的 "ray-dependent density" 问题同源，但 3DGS 因为 splat 是 explicit primitive，比 NeRF 的 implicit density field 更脆弱——一个 stretched splat 从前面看是好的，从侧面就是尖刺。

### 7.2 为什么 point transformer 在这里 work

Point cloud 的核心挑战是 **unordered + irregularly sampled**。3DGS 优化后产生的 splats 完美符合这两个特点：数量从 0 到几百万，分布极不均匀（surface 上密集，empty space 稀疏）。PTv3 的 serialization + grid pooling 正好处理这个：

- Hilbert curve serialization：保留空间局部性，让 1D attention 等效于 local 3D attention
- Grid pooling：在 voxel 内聚合，自动处理密度差异
- Multi-stage U-Net：hierarchical receptive field，能看 local 细节也能看 global 结构

### 7.3 与其他 generative refinement 思路的对比

**Diffusion-based refinement** (e.g., [ReconFusion](https://reconfusion.github.io/), [CAT3D](https://cat3d.github.io/))：用 2D diffusion 先验补全 unseen region。优点是能 hallucinate，缺点是 multi-view 不一致 + 速度慢。SplatFormer 反其道而行——**不 hallucinate，只 refine 已有的 splats**。这适合 input 已经 dense 覆盖大部分区域的场景（museum statue），不适合真正 sparse input。

**Feed-forward 2D-to-3D** (e.g., [LaRa](https://anpei.info/lara-web/), [PixelSplat](https://pixelsplat.github.io/))：从 2D image 直接预测 3D primitives。Memory 限制只能 4 个 view。SplatFormer 把"3DGS optimization"当作 preprocessing，绕开了 memory 限制，能用 32 个 view 的所有信息。这是个聪明的 division of labor。

### 7.4 这套 idea 可以延伸的方向

1. **2DGS refinement**：paper 自己提到，2DGS 在 OOD-NVS 上比 3DGS 强（PSNR 23.52 vs 20.21 in ShapeNet），但还有空间。把 SplatFormer 改造支持 2DGS 输入（增加 normal residual output）应该能进一步提升。
2. **Dynamic scenes**：4DGS 也有 OOD 问题，把 SplatFormer 扩展到时序维度，每帧 refine。
3. **In-the-wild unbounded**：MVImgNet 实验显示 foreground 缩小问题。可以引入 foreground-background 分割（Segment3D 已经有了），分别 refine。
4. **Generative SplatFormer**：现在的 model 只能 refine，不能 add/remove splats。如果加入 set transformation（类似 DETR 的 object queries），就能动态增删 splats，可能解决 high-frequency detail 限制。
5. **Self-supervised 真实数据训练**：用 multi-view video（如 [CO3D](https://github.com/facebookresearch/co3d)）+ next-view prediction self-supervision，绕开需要 ground-truth OOD views 的限制。

### 7.5 与 LLM/Vision Transformer 的类比

这个工作很像 **LLM 的 post-hoc refinement**：先有个 "draft"（initial 3DGS），再用 transformer 做 "refinement"（SplatFormer）。对应到 LLM：
- Initial 3DGS ≈ draft token（cheap, fast, but biased）
- SplatFormer ≈ LLM verifier/refiner（expensive, but corrects errors）

这让人想到 reasoning model 的 inference-time scaling 思路：与其让 3DGS optimization 自己 struggle，不如先快速生成 draft，再 refined。这其实是个 general pattern。

### 7.6 工程上的启发

1. **Residual learning + zero-init 是 invariant-preserving 的关键**：训练初期网络是 identity，不会破坏已经 work 的部分。
2. **2D supervision > 3D supervision for 3D tasks**：因为最终 metric 是 2D 的，直接 optimize end metric。这是 NeRF/3DGS 工作的常见 trick，但 paper 用 ablation 清楚证明了。
3. **Dataset engineering 是核心**：3000 GPU hours 的 dataset curation，但这笔投资换来的是 generic prior。和 LLM 的数据 scaling 思路一致。

### 7.7 几个我好奇但 paper 没说的点

1. **OOD view 的 camera distance**：训练时是 R=1，但 Tab. E.1 测试了 R=0.2–1.0 都能 work，说明 prior 在 distance 维度上 generalize 得意外地好。
2. **Splat count scaling**：训练时 cap 100k，测试时能到 4M。这种 100x 外推能 work 暗示 attention 的 spatial locality 起了关键作用——每个 splat 只看局部邻居，不依赖全局 splat count。
3. **Failure mode 的根源**：MVImgNet foreground 缩小问题——归一化 + downpooling 让小物体信息丢失。这可能启发 **adaptive pooling**：在 pooling 时根据 local point density 保留信息。

---

## 8. 关键 Reference Links

- **SplatFormer project page**: https://sergeyprokudin.github.io/splatformer/
- **3DGS original paper**: https://repo.as-3dgaussiansplatting.cc/  
- **PTv3**: https://github.com/PointTransformerV3/PTv3
- **gsplat toolbox**: https://github.com/nerfstudio-project/gsplat
- **ShapeNet**: https://shapenet.org/
- **Objaverse**: https://objaverse.allenai.org/
- **GSO**: https://app.gazebosim.com/GoogleResearch/fuel_google_scanned_objects
- **MipNeRF360**: https://jonbarron.info/mipnerf360/
- **2DGS**: https://surh.github.io/2d-gaussian-splatting/
- **SplatFields**: https://markomih.github.io/SplatFields/
- **LaRa**: https://anpei.info/lara-web/
- **Nerfbusters**: https://hanzolab.github.io/nerfbusters/
- **DiffBIR**: https://github.com/XPixelLabs/DiffBIR
- **SyncDreamer**: https://liuyuan-peng.github.io/SyncDreamer/
- **EscherNet**: https://eschernet.github.io/
- **InstantNGP**: https://nvlabs.github.io/instant-ngp/
- **Spectral bias paper**: https://arxiv.org/abs/1806.08734

---

## 9. 总结 Intuition

1. **3DGS 在 OOD 视角的崩塌**根源于 splat primitives 的 over-fitting：optimization 只看到训练 ray，splat 形状被这些 ray "塑形"，从其他角度看就崩。
2. **Point transformer 是合适的 refine 工具**因为 3DGS 本质是 irregular 3D point set with attributes，attention 能动态聚合局部信息修正 artifacts。
3. **2D supervision > 3D supervision** 因为下游 metric 是 2D，spectral bias 让 small 3D error 在 rendering 时放大。
4. **Residual learning + zero-init** 保证 refinement 是 "small perturbation"，不会 destroy 已有的好部分。
5. **学到的 prior 是 generic 的**，能跨 dataset（synthetic → real）和跨 object category generalization，说明 SplatFormer 学到的是"什么是合理的 3D splat distribution"这种抽象知识，而不是物体 specific 的知识。

整个工作最打动我的地方是它的 simplicity：**train a transformer to refine 3DGS, supervised by rendering loss**。这个 setup 干净、可扩展、且 first-principles 正确。未来 3DGS-based scene reconstruction pipeline 里加一个 SplatFormer-style refinement step，可能就像图像分类 pipeline 里加一个 batch norm 一样自然。
