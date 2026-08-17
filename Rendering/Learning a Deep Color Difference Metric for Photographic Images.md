---
source_pdf: Learning a Deep Color Difference Metric for Photographic Images.pdf
paper_sha256: f1ae28caa3d80956e2c730077778c1284e8e05771e18fd48eccc2321c29a46d9
processed_at: '2026-08-05T12:38:55-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 CD-Flow

## 这 paper 到底干嘛的

你拍两张同一场景的照片，比如用 iPhone 和小米各拍一张，颜色看起来有点不一样。你想用数字量化"这俩颜色差多少"——这就是 color difference (CD)。

这事听起来简单，其实巨难。工业界用了几十年的公式（CIELAB, CIEDE2000）都是上世纪 70-80 年代拿**纯色小方块**让人眼看、然后拟合出来的。但你拿它去算**真实照片**的 CD，跟人眼感受对不上。

## 为什么老方法不行

CIEDE2000 的做法：把每个 RGB pixel 转成 LAB 三个数，然后两个 pixel 算 3D Euclidean distance，最后整张图取平均。

问题：
- **纯色方块没 structure**，照片有 texture、edge、context
- **Pixel-by-pixel 对 alignment 极度敏感**——两张照片哪怕差 2 个 pixel 的平移，每个 pixel 对不上，算出来全是噪声
- 人眼不是这么看颜色的——人脑的 V1 里有两类神经元，一类管大片颜色，一类管颜色 texture/boundary，**颜色和形状在视觉皮层里是绑在一起处理的**

## CD-Flow 的核心 idea

一句话：**学一个 RGB 到另一个 space 的可逆变换，在这个新 space 里直接算 Euclidean distance 当 CD**。

为什么用 normalizing flow？三个理由：

**1. Bijective（双向可逆）意味着 information 不丢**
RGB 图像有 $H \times W \times 3$ 个数，latent space 也是同样维度。所有信息都保留，只是重新排列组合了一下。这保证了它是**数学上真正的 metric**——满足 triangle inequality，满足 identity of indiscernibles。

这跟 LPIPS 不一样。LPIPS 用 VGG feature，VGG 是 classifier，把 image 压到一个 task-specific subspace，信息是丢掉的，所以不是 proper metric。

**2. Multi-scale 模拟视觉皮层**
Flow 有 K=6 个 scale。每个 scale 先 squeeze（把 2×2 spatial patch 折叠到 channel 维度，相当于扩大 receptive field），然后做几步 flow 处理，再 split 成两半——一半当这个 scale 的 output，另一半继续进下一级 squeeze。

直觉：
- **Coarse scale** = 大 receptive field = 响应整片颜色（像 V1 的 single-opponent cells）
- **Fine scale** = 小 receptive field = 响应 texture/boundary（像 V1 的 double-opponent cells）

而且 coarse scale 天然对 translation/rotation 鲁棒——你平移一张图，全局颜色统计几乎不变。

**3. Multi-scale loss 强制每个 scale 都能预测 CD**
$$\ell = \sum_{k=1}^{K} \|\Delta E_k - \Delta V\|_p$$

每个 scale 都自己预测一遍 human rating $\Delta V$。这让 coarse scale 不能偷懒——它必须自己 capture 足够信息来预测 CD，不能靠 fine scale rescue。

结果就是 coarse scale 学到 robust 的颜色统计特征，对 geometric distortion 天然不敏感。这就是为什么 CD-Flow 在 misaligned image pair 上比 CIEDE2000 好一大截。

## 一个关键的工程细节

光用 CD loss 训练，affine coupling layer 的 scale 会塌到 0——因为 scale=0 时输出跟输入无关，网络可以走捷径 fit target。但 scale=0 之后 inverse 就除以零爆炸了。

所以加了个 NLL loss（normalizing flow 的标准训练 loss）当 regularizer。NLL 里有 $-\log|\det J|$ 这一项，惩罚 Jacobian 塌缩，强制保持 bijectivity。

直觉：**CD loss 负责"有用"，NLL loss 负责"不塌"**。两个一起才能 work。

## 结果如何

在 SPCD dataset（最大的 photographic CD dataset）上：
- CD-Flow: STRESS=18.47
- CD-Net（之前的 SOTA，也是 DNN）: 21.43
- CIEDE2000: 31.44
- LPIPS: 64.15（彻底崩了，因为 LPIPS 是为 structure/quality 设计的，不是为 color diff）

加了 5% translation / 3° rotation / 1.05x zoom 后：
- CD-Flow: 20.1
- CD-Net: 22.5
- CIEDE2000: 28.0

泛化到纯色方块（CIEDE2000 的训练集）：
- CIEDE2000: 28.98（意料之中，它就是 fit 这些的）
- CD-Flow: 35.06（虽然没见过方块，但合理）
- CD-Net: 38.87（overfit 到 image-specific）

泛化到 TID2013 的 color distortion：
- CD-Flow: 14.11（最好）
- DISTS: 15.24
- LPIPS: 15.42

## 为什么这个工作有意思

它不是"又一个 DNN beat 传统公式"的故事。它的 architecture design 里**每个选择都对应一个明确的设计目标**：

| 设计 | 对应性质 |
|---|---|
| Normalizing flow (bijective) | 数学上 proper metric |
| Multi-scale squeeze | 模拟 V1 的 single/double-opponent cells |
| Autoregressive factorization | Coarse-to-fine，像视觉皮层 hierarchical processing |
| Multi-scale loss | 每个 scale 独立 robust，几何扰动不敏感 |
| NLL regularizer | 防止 bijectivity 崩塌 |

这种"每个 architectural choice 都有 reason"的工作，在 deep learning 里其实挺少见的。大部分 paper 就是"我们堆了 ResNet + attention + 一些 trick，效果更好了"，这篇不是。

**最深的 intuition**：好的 perceptual metric 需要 information-preserving 的 transform。Surjective transform（VGG、分类器 backbone）丢信息，适合 task-specific 评估但不适合 general metric。Bijective transform（normalizing flow）保信息，reparametrize 但不丢东西，所以更接近真正的"感知均匀坐标系"。

这其实呼应了 color science 的老梦想——CIELAB 当年就是想要一个"perceptually uniform"的坐标系，只是 handcrafted 的 uniformity 不够好。CD-Flow 用 data-driven 的方式重新实现这个梦想，而且 extend 到了 photographic images。

---

# Learning a Deep Color Difference Metric for Photographic Images - 深度解析

Hey Andrej! 这篇 paper 是 Kede Ma 组（City University of Hong Kong）的工作，第一作者 Haoyu Chen，核心想法非常优雅：**用 normalizing flow 学一个 RGB 到 latent space 的 bijective coordinate transform，再用 Euclidean distance 作为 color difference (CD) metric**。下面我从 motivation、架构、数学性质、实验直觉几个层面来 build your intuition。

---

## 1. 核心问题的 motivation

传统 CD metrics（CIELAB, CIE94, CIEDE2000 等）都是 handcrafted formula，基于 uniform color patches（比如 MacAdam ellipses、BFD-P、Leeds、Witt、RIT-DuPont 这些 dataset）做 psychophysical 拟合的。

问题在于：
- **Color patches ≠ photographic images**：patches 没有 spatial structure，没有 texture，没有 boundary
- **Pixel-by-pixel CD 计算对 misalignment 极度敏感**：同一场景两个相机拍的照片，会有 parallax、translation，pixel-wise 算 CIEDE2000 就完全乱掉
- **CIELAB uniformity 不理想**：即使对 patches 也不够 uniform（Luo & Rigg 1986 chromaticity discrimination ellipses 已经显示这个问题）

Vision science 的现代 view（Shapley & Hawken 2011, Vision Research）：V1 里有 **single-opponent cells**（响应大面积颜色，类似 coarse scale）和 **double-opponent cells**（响应 color patterns, textures, boundaries，类似 fine scale）。Color 和 form 在 visual cortical processing 中 inextricably linked，不分模块处理。

这就给了 paper 一个生物学的 hook：**multi-scale 处理对应不同 size 的 receptive fields，对应 single/double-opponent cells 的不同响应特性**。

---

## 2. 四个 desirable properties 的设计

这是 paper 最精妙的地方 —— 用一个 architecture 同时满足四个看似冲突的性质：

| Property | 如何实现 |
|---|---|
| ① Color-form inextricable interaction | Multi-scale flow + squeezing (trade space for channel，等价于扩大 receptive field) |
| ② Proper metric (数学上) | Bijective transform $f$ + Euclidean distance $\Rightarrow$ 满足 non-negativity, symmetry, identity of indiscernibles, triangle inequality |
| ③ Accurate to human perception | 在 SPCD dataset 上端到端监督学习 |
| ④ Robust to geometric distortion | Multi-scale autoregressive：coarser scale 先算，finer scale 条件于 coarser；coarse scale 天然容忍 translation/dilation/rotation |

特别是 property ②：CD-Net（Wang et al. 2023, TPAMI）虽然也是 DNN-based，但它的 feature transform 可能是 surjective 的，导致不是真正的 metric（identity of indiscernibles 可能不成立，triangle inequality 也可能挂掉）。Normalizing flow 的 bijectivity 干净地解决了这一点。

**为什么 bijectivity 保证 proper metric?**

设 $f: \mathcal{X} \to \mathcal{Z}$ 是 bijection，定义 $d(x, y) = \|f(x) - f(y)\|_2$。那么：
- $d(x,y) \geq 0$，且 $d(x,y)=0 \Leftrightarrow f(x)=f(y) \Leftrightarrow x=y$（identity of indiscernibles，靠 injectivity）
- $d(x,y) = d(y,x)$（Euclidean 对称）
- $d(x,z) \leq d(x,y) + d(y,z)$（Euclidean 的 triangle inequality 通过 $f$ 拉回）

这是非常干净的设计，比 LPIPS（不是 metric，因为靠 VGG feature，可能不满足 triangle inequality）理论性质好得多。LPIPS 论文里其实作者也提到这一点，只是 DISTS 论文 follow-up 才认真讨论。

参考：[LPIPS paper](https://arxiv.org/abs/1801.03924), [DISTS paper](https://arxiv.org/abs/2004.07728)

---

## 3. Architecture 详解

### 3.1 整体结构（multi-scale autoregressive normalizing flow）

Figure 1 描述：K=6 scales，每个 scale 处理 + split（最后一个 scale 不 split）。设 $z_0 = x$ 是输入 RGB 图像，在 k-th scale：

$$z_{2(k-1)} \xrightarrow{\text{squeeze + flow steps + split}} z_{2k-1} \oplus z_{2k}$$

$z_{2k}$ 进入下一级 scale 处理，$z_{2k-1}$ 作为这个 scale 的输出 feature（用于 CD 计算）。最后一级只产生 $z_{2K-1}$，不 split。

最终的 latent representation 是 $\mathbf{z} = \{z_1, z_3, z_5, \dots, z_{2K-1}\}$。

**Autoregressive 概率密度因子化**：

$$p(\mathbf{z}) = \prod_{k=1}^{K-1} p(z_{2k-1} | z_{2k}) \cdot p(z_{2K-1}) \tag{1}$$

第二个等号成立是因为 bijectivity：$\{z_{\geq (2k+1)}\}$ 和 $z_{2k}$ 是一一对应的（细 scale 决定粗 scale，反之亦然，因为是 bijection）。$p(z_{2k-1}|z_{2k})$ 用 conditionally independent Gaussians 建模，mean 和 diagonal covariance 由一个 tiny NN 从 $z_{2k}$ 算出来。$p(z_{2K-1})$ 是 unconditional Gaussian，参数直接 backprop 学。

**Intuition**：这种 factorization 模拟了视觉皮层的 hierarchical processing —— coarse feature 给 fine feature 提供上下文（top-down），fine feature 又 refine coarse feature（bottom-up）。这在 Bayesian 视觉感知里也有对应（prediction error coding）。

### 3.2 单个 flow step 三个操作

Figure 2：每个 flow step = Actnorm → Invertible 1×1 Conv → Affine Coupling。

**① Actnorm**（替代 batchnorm，因为小 batch size 时 batchnorm 退化）：

$$z' = s \odot z + t \tag{2}$$

其中 $s, t \in \mathbb{R}^c$ 是 per-channel learnable scale 和 bias，$\odot$ 是 element-wise 乘法。给定输入 $z \in \mathbb{R}^{c \times h \times w}$，log-determinant 是 $h \cdot w \cdot \log|\det(\text{diag}(s))|$（因为对每个 spatial position 都做了同一个 channel-wise 仿射，Jacobian 是 block-diagonal，每个 block 是 $\text{diag}(s)$，共 $h \cdot w$ 个 block）。

**② Invertible 1×1 Convolution**（Glow 的核心 trick，Kingma & Dhariwal 2018）：

$$z' = Wz \tag{3}$$

$W \in \mathbb{R}^{c \times c}$ 是 learnable weight matrix，对每个 spatial position 应用同一个线性变换（mixing channels）。Log-determinant = $h \cdot w \cdot \log|\det(W)|$。

**为什么需要这一步？** Affine coupling 只对一半的 channel 做变换，另一半原封不动。如果连续多个 coupling step 不加 permutation，那么始终原封不动的那一半 channel 永远不会被动到，表达力受限。RealNVP 用 fixed random permutation 来解决，Glow 用 learnable 1×1 conv 更灵活（等价于在 channel 维度做 PCA-like 线性变换，所以叫 "linear transform in PCA"）。

参考：[Glow paper](https://arxiv.org/abs/1807.03039), [RealNVP paper](https://arxiv.org/abs/1605.08803)

**③ Affine Coupling**（RealNVP 的核心）：

输入 $z \in \mathbb{R}^D$ 沿 channel 维 split 成两半 $z_{1:d}$ 和 $z_{d+1:D}$（$d = D/2$）。前一半原封不动，后一半做仿射变换，仿射的参数 $s(\cdot), t(\cdot)$ 由前一半通过 NN 算出：

$$z'_{1:d} = z_{1:d}$$
$$z'_{d+1:D} = z_{d+1:D} \odot e^{s(z_{1:d})} + t(z_{1:d}) \tag{4}$$

**为什么用 $e^{s}$ 而不是 $s$？** 因为要保证 scale 是正的（这是 bijectivity 的关键 —— 若允许 scale 为负或零，会出现 multi-valued inverse）。用 $e^s$ 自动保证 scale > 0。

**Jacobian 是下三角阵**：
$$J = \begin{bmatrix} I_d & 0 \\ \frac{\partial z'_{d+1:D}}{\partial z_{1:d}} & \text{diag}(e^{s(z_{1:d})}) \end{bmatrix}$$

$\det(J) = \prod_i e^{s_i(z_{1:d})} = e^{\sum_i s_i(z_{1:d})}$，所以 log-determinant 就是 $\sum_i s_i(z_{1:d})$ —— 这是 normalizing flow 用 coupling layer 的核心计算优势（不用做 O(D^3) 的 determinant 计算，只要 O(D) 求和）。

**Inverse**（解析可求）：
$$z_{1:d} = z'_{1:d}$$
$$z_{d+1:D} = (z'_{d+1:D} - t(z'_{1:d})) \odot e^{-s(z'_{1:d})} \tag{5}$$

注意 $s, t$ 这两个函数本身**不需要是可逆的**，因为 inverse 时直接用 forward 的 $s, t$ 作用在 $z'_{1:d}$ 上即可（而 $z'_{1:d} = z_{1:d}$，所以信息无损）。这就是 coupling layer 的妙处。

**Intuition**：这是 "easy forward, easy inverse" 的设计 —— 仿射变换很 expressive，但 inverse 不需要 invert 任何 NN，只需要"反向"应用仿射即可。

### 3.3 Squeezing Operation（用 space 换 channel）

$$\text{squeeze}: s \times s \times c \to \frac{s}{2} \times \frac{s}{2} \times 4c$$

把 2×2 spatial patch 拆到 channel 维度（类似 pixel shuffle 的逆操作 / space-to-depth）。

**为什么这个对应 single/double-opponent cells？**
- Squeeze 之前：channel 数少，spatial 大 → receptive field 小 → 类似 double-opponent cells（响应 texture, boundary）
- Squeeze 之后：channel 数多，spatial 小 → 等效 receptive field 大 → 类似 single-opponent cells（响应大面积颜色）

但 paper 里的描述是相反的：coarser scale 上 local CD 是大区域 color（single-opponent），finer scale 上 local CD 响应 texture/boundary（double-opponent）。这对应于 SQUEEZE 之后 spatial 变小（等效 receptive field 变大），但 squeeze 操作本身又把 spatial info 塞到 channel 里，等下一层 flow 步骤能 access 那些信息。

这里其实有点 tricky。让我重新理解：multi-scale 处理流程是 `image → squeeze → flow → split → (coarse half 进入下个 scale)`。Coarse half（被 split 保留下来用于 CD 计算）实际上是低 spatial resolution 的，channel 维度比较多。但 split 是沿 channel 切的，所以切完之后 coarse half 的 spatial 和 fine half 一样，只是 channel 数减半。

更精确的 intuition：每个 scale 在该 spatial resolution 上做了 L=8 步 flow + channel mixing，然后 split 一半进入下一级（继续 squeeze，spatial 再小一半）。所以：
- $z_1$ 在原始分辨率 $s \times s$
- $z_3$ 在 $s/2 \times s/2$
- $z_5$ 在 $s/4 \times s/4$
- ...
- $z_{2K-1}$ 在 $s/2^{K-1} \times s/2^{K-1}$，最 coarse

CD 在每个 scale 独立算（公式 9），最后 multi-scale 加和（公式 8）。Coarse scale 对应 large receptive field 的特征，对 translation 鲁棒，对应 single-opponent；fine scale 对应 small receptive field 的特征，对 texture 敏感，对应 double-opponent。这是论文 Figure 3 显示的 emergent property —— 没有显式监督 local CD map，但自然涌现出这种 single/double-opponent 的分工。

参考：[RealNVP squeezing](https://arxiv.org/abs/1605.08803)

---

## 4. CD Distance 和 Loss

### 4.1 单尺度 CD

$$\Delta E(\mathbf{x}, \mathbf{y}) = \sqrt{\frac{(f(\mathbf{x}) - f(\mathbf{y}))^T (f(\mathbf{x}) - f(\mathbf{y}))}{D}} \tag{6}$$

$D$ 是 latent space 的总维度，相当于做一个 normalized Euclidean distance（除以 $D$ 等价于 mean squared difference 然后开方）。这等价于 $\ell_2$ norm per dimension，符合 CIELAB $\Delta E$ 的传统定义（CIELAB $\Delta E$ 也是 $\sqrt{\Delta L^2 + \Delta a^2 + \Delta b^2}$，3 维 Euclidean）。

### 4.2 Multi-scale CD

$$\Delta E_k(\mathbf{x}, \mathbf{y}) = \sqrt{\frac{(f_{k:}(\mathbf{x}) - f_{k:}(\mathbf{y}))^T(f_{k:}(\mathbf{x}) - f_{k:}(\mathbf{y}))}{D_k}} \tag{9}$$

其中 $f_{k:}(\mathbf{x}) = [z_{2k-1}^T, z_{2k+1}^T, \dots, z_{2K-1}^T]^T$ 是从 k-th scale 到 final scale 的所有 feature 拼起来，$D_k$ 是这些 feature 的总维度。

**关键设计**：$\Delta E_1 = \Delta E$（用全部 scale），$\Delta E_K$ 只用最后一个 scale。

$$\ell_{\text{ms}}(\mathbf{x}, \mathbf{y}) = \sum_{k=1}^{K} \|\Delta E_k(\mathbf{x}, \mathbf{y}) - \Delta V(\mathbf{x}, \mathbf{y})\|_p \tag{8}$$

**为什么 multi-scale？** 每个 $\Delta E_k$ 都试图预测同一个 human rating $\Delta V$，但是用不同 scale 的 feature。这等价于让每个 scale 单独都能预测 CD，从而鼓励 coarse scale（geometric 鲁棒）也能 hold its own。这就是 property ④ 的来源 —— 不是靠 data augmentation 加扰动，而是靠 architecture 内在的 multi-scale 监督。

### 4.3 Negative Log-Likelihood Loss（防止 bijectivity 崩塌）

这是 paper 一个很重要的工程细节。光用 CD loss 训练，affine coupling 的 scale factor 会趋近于 0（因为 scale=0 时 $z'_{d+1:D} = t(z_{1:d})$，输出与输入无关，CD loss 可以让 NN fit target 而不管输入）。一旦 scale=0，inverse 公式 (5) 里要除以 $e^s$，就 explode 了（Behrmann et al. 2021 ICML 称之为 "exploding inverse problem"）。

为了防止这个，加上 NLL loss：

$$\ell_{\text{nl}}(\mathbf{x}) = -\log p_{\mathcal{X}}(\mathbf{x}) = -\log p_{\mathcal{Z}}(f(\mathbf{x})) - \log\left|\det\left(\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}}\right)\right| \tag{10}$$

由 change of variables formula：$p_{\mathcal{X}}(\mathbf{x}) = p_{\mathcal{Z}}(f(\mathbf{x})) \cdot |\det J_f|$。NLL 就是让模型对真实图像 assign 高概率，这要求 Jacobian 不要塌缩（要保留信息）。

**Total loss**：

$$\ell(B) = \frac{1}{|B|}\sum_{(\mathbf{x},\mathbf{y}) \in B} \left[\ell_{\text{ms}}(\mathbf{x}, \mathbf{y}) + \lambda (\ell_{\text{nl}}(\mathbf{x}) + \ell_{\text{nl}}(\mathbf{y}))\right] \tag{11}$$

$\lambda$ 是 trade-off 系数。NLL 起到 regularizer 的作用，强制 flow 保持 bijective。这是 DNN-based CD metric 的一个 overlooked 的问题 —— 如果只是用 NN 当 feature extractor，特征空间可能 collapse，metric 性质就破坏了。

参考：[Understanding and mitigating exploding inverses in invertible neural networks](https://arxiv.org/abs/2001.09468)

---

## 5. 实验结果分析

### 5.1 主实验 SPCD（Table 1）

SPCD 是 Wang et al. 2023 TPAMI 提出来的 dataset：15,335 张图，1,000 个 scene，30,000 pairs，每 pair 20 个 human ratings。10,005 pairs 是非完美对齐的（real-world photography 不可避免有 misalignment）。

**CD-Flow 在所有指标上最好**：
- All pairs: STRESS=18.473, PLCC=0.871, SRCC=0.865
- vs CD-Net: STRESS=21.431（提升约 14%）
- vs CIEDE2000: STRESS=31.439（提升约 41%）
- vs LPIPS: STRESS=64.145（LPIPS 在 CD 任务上完全崩溃，因为它针对 quality 优化，不是针对 color-only diff）

**值得注意的几个点**：

1. **CIE94 vs CIEDE2000 vs CIELAB 在 natural image 上几乎没区别**。CIE94 STRESS=34.326，CIEDE2000 STRESS=31.439，CIELAB STRESS=31.872，差距很小。这说明 handcrafted 对 patch 的优化迁移不到 image 上。

2. **LPIPS, DISTS 即使在 SPCD 上 retrain，效果也不如 CD-Flow**。LPIPS 依赖 VGG feature，VGG 是 ImageNet 训练的，feature 主要 capture structure/texture，对 color appearance diff 不敏感。DISTS 也是。这是 paper 一个 strong claim：**general-purpose IQA model 不适合 CD task**。

3. **Non-perfectly aligned pairs（10,005 pairs）上 CD-Flow 优势更明显**：
   - CD-Flow: STRESS=21.374
   - CD-Net: STRESS=22.543
   - CIEDE2000: STRESS=30.347
   - LPIPS: STRESS=53.132（崩溃）
   
   这正验证了 property ④（geometric robustness）。

### 5.2 几何扰动实验（Table 2）

人为加 translation（5% pixels）、rotation（3°）、dilation（1.05x zoom）。

CD-Flow 在所有扰动下都最好或并列最好：
- Translation: CD-Flow 19.311 vs CD-Net 19.825（差不多）
- Rotation: CD-Flow 20.139 vs CD-Net 22.463（CD-Flow 明显好）
- Dilation: CD-Flow 21.352 vs CD-Net 21.704（差不多）

CIELAB-based 方法在扰动下崩得很厉害（CIEDE2000 translation 28.035 → 28.035，dilation 29.928，PLCC 从 0.825 掉到 0.566）。

**为什么 CD-Flow 对 rotation 也鲁棒？** 这是 multi-scale + flow 的 emergent property —— coarse scale 的 feature 主要是颜色统计（rot-invariant 的整体颜色），fine scale 即使 mis-aligned，对 coarse 的 contribution 不大。Plus 1×1 conv 在 channel 维度做 mixing，是 rotation-invariant 的。Squeeze 也是 2×2 块操作，对小幅 rotation 鲁棒。

### 5.3 泛化到 color patches（Table 3, COM dataset）

COM dataset 包含 BFD-P, Leeds, Witt, RIT-DuPont 四个 patch dataset。

- CIEDE2000: COM 整体 STRESS=28.979（最好，意料之中，CIEDE2000 就是 fit 这些 patches 的）
- CD-Flow: COM 整体 STRESS=35.061（比 CIEDE2000 差但比 CIELAB 的 45.202 好）
- CD-Net: STRESS=38.872（最差，说明 CD-Net 在 patches 上 overfit 到 image-specific features）

**有意思**：CD-Flow 比 CD-Net 在 patches 上 generalize 得更好（35.061 vs 38.872），这说明 normalizing flow 的 bijective 约束带来某种 inductive bias，让 latent space 更"均匀"，对 patch 也能合理处理。

### 5.4 TID2013 泛化（Table 4）

TID2013 里有三种 color-related distortion：quantization noise, color quantization with dither, chromatic aberration。

- CD-Flow: STRESS=14.110（最好）
- DISTS: STRESS=15.235（第二）
- LPIPS: STRESS=15.420
- CD-Net: STRESS=15.962
- CIEDE2000: STRESS=18.203

这里 CD-Flow 比 DISTS 和 LPIPS 略好，而 LPIPS/DISTS 在 TID2013 上 pre-train 时见过类似 distortion。这是 strong evidence that bijective flow 学到的 representation 对 color distortion general 性很强。

### 5.5 Ablation（Table 5, 6）

**Number of flow steps L**：L=2 → L=16，STRESS 从 21.633 → 17.792，单调下降但收益递减。L=8 是 default（trade-off 计算 cost）。

**Number of scales K**：K=2 → K=8，STRESS 从 24.686 → 18.473（K=6 时）→ 18.524（K=8 时反弹）。

**K=8 反弹的原因**：作者解释说输入分辨率 768×768，K=8 时 coarsest scale 是 768/2^7 = 6×6 像素，再 squeeze 就接近"全局平均颜色"，"less biologically plausible and less practically meaningful"。这是一个有趣的 observation —— multi-scale 不能太 multi，最终 scale 不能退化成 global average color。

---

## 6. 个人 commentary & 关联思考

### 6.1 跟 LPIPS / DISTS 的根本区别

LPIPS 用 VGG feature 然后 linear weight，本质上是个 surjective transform（VGG 是 classifier，丢掉很多对 color 敏感但对 classification 无用的信息）。DISTS 也是基于 VGG 但加上 texture similarity。两者都是为 **image quality / structure** 设计的，不是为 **color appearance** 设计的。

CD-Flow 用 normalizing flow，**bijective 意味着信息无损**，latent space 维度 = input 维度。所有 color info 都保留，只是 reparametrize。这是为什么它对 color diff 敏感而对 structure inductive 弱。

**这给一个更 general 的 intuition**：metric 的性质由 feature transform 的 injectivity 决定。Surjective transform 投影到 task-relevant subspace（适合 task-specific quality，但破坏 metric 性质），bijective transform 保留全部信息（适合 general perceptual metric）。

### 6.2 跟 Normalizing Flow 的常见用法区别

NF 一般是做 generative modeling（生成 samples）。这里 paper 用 NF 做 **feature transform for metric learning**，没有 sampling，只是利用了 NF 的 bijectivity + tractable Jacobian + 可学的 expressive transform。

这其实是个很自然的 idea，但少见。一个相关的工作是 ResNet-based perceptual loss，但那些是 surjective 的。Bijective flow 用于 metric 的好处是 metric 性质可以严格证明。

类似 idea 在 speaker verification、face recognition 里有出现过（用 flow 学 embedding），但 IQA 领域少见。

### 6.3 跟 Vision Science 的对应

Paper 提到 single-opponent vs double-opponent cells，并 claim Figure 3 的 multi-scale CD map emergent 表现出这种分工。这其实是一个 weak claim —— paper 没有显式监督 local CD map，只是看 forward pass 中每个 spatial position 的 latent distance。

但这个对应是合理的：
- Coarse scale feature 由 NLL 训练成 Gaussian（mean 集中在 color statistics），响应 large area color → single-opponent-like
- Fine scale feature 是 high-frequency 信号，响应 texture, edge → double-opponent-like

如果作者进一步可视化 first-layer flow 的 receptive field，会更 convince。一个可能的 follow-up 是 inspect affine coupling 的 $s(\cdot), t(\cdot)$ NN 学到什么 spatial pattern。

### 6.4 Multi-scale loss 的妙处

公式 (8) 让每个 scale 都能独立预测 $\Delta V$。这等价于 **deep supervision**（像 deeply-supervised Net），但每层 supervise 同一个 target。

这种设计的好处是： coarse scale 不会被 fine scale "rescue"，必须自己 capture enough info to predict CD。所以 coarse scale 学到 color-appearance-level 的特征，对 geometric 鲁棒。

跟 U-Net 的 skip connection 不一样 —— U-Net 的 skip 是 information flow，不是 supervision。这里 multi-scale 是 supervision 同时也是 feature aggregation。

### 6.5 NLL loss 的双关作用

公式 (10) 的 NLL loss 一方面是 normalizing flow 训练的标准做法（让 latent space 接近 Gaussian），另一方面是 **bijectivity 的 regularizer**。

NLL = $-\log p_{\mathcal{Z}}(f(x)) - \log|\det J|$。第一项鼓励 $f(x)$ 落在 Gaussian 高概率区，第二项鼓励 Jacobian 不塌缩（保持 volume，即保持信息）。后者直接对应 bijectivity 的保持。

如果只有 CD loss，模型可以走捷径：把 affine coupling 的 scale 设成 0，那么 $z'_{d+1:D} = t(z_{1:d})$，相当于 $z_{d+1:D}$ 的信息被丢弃（replaced by 一个固定值），CD loss 还能 fit。但 inverse 会爆炸。NLL 的 $-\log|\det J|$ 项惩罚这种 collapse。

这个 insight 对所有 invertible NN 的 metric learning 都适用。

### 6.6 Generalization 的根本原因

CD-Flow 在 TID2013 和 color patches 上 generalize 得比 CD-Net 好，根本原因可能是：

1. **Bijective constraint 限制了 overfitting** —— feature space 维度等于 input space，不能压到一个低维 task-specific manifold。
2. **Gaussian prior 在 latent space 强制平滑** —— latent 不会 over-concentrate 在 training data 的某些方向。
3. **Multi-scale supervision 让每个 scale 都 general** —— 不会 overfit 到 fine-scale texture idiosyncrasies。

这三个加起来让 CD-Flow 更接近一个 "principled" 的 metric。

---

## 7. Potential extensions & open questions

1. **为什么 NLL 是必要的？** 能否用其他 bijectivity regularizer（比如 Lipschitz constraint、spectral normalization on coupling layers）替代？NLL 一个 issue 是它假设 latent 是 Gaussian，但 CD loss 想要的是 metric structure，两者可能矛盾。$\lambda$ 的 trade-off 可能 sensitive。

2. **Squeezing 4c channels 后 channel 数很快爆炸**：$c \to 4c \to 16c \to \dots$，K=6 时 channel 数到 $4^5 \cdot 3 \approx 3000$。Split 减半但仍然大。这限制了输入分辨率和 K。Wavelet Flow（Yu et al. 2020）可能能 scale 到更高分辨率。

3. **和 deep perceptual metric 比较的理论分析**：CD-Flow 是 proper metric，LPIPS/DISTS 不是。能否证明 proper metric 在 optimization 中更 stable？Ding et al. 2021 IJCV 有讨论这点（reference [11]），但只针对 MSE、SSIM、LPIPS 这种 simple case，没涵盖 flow-based metric。

4. **跟 JND (just-noticeable difference) 的关系**：Butteraugli 和 FLIP 是 JND metric，针对 visibility threshold。CD-Flow 针对的是 suprathreshold CD（human rating 是 magnitude，不是 threshold）。两个 task 不完全一样。能否用同一 framework unification？

5. **Color space 的 inductive bias**：paper 直接用 RGB input，没有先做 XYZ → LAB 之类的 color space transform。但 normalizing flow 的 expressive 足够学到这种 transform 吗？从实验看 yes（性能比 CIELAB 好），但 inspect 学到的 first-layer features 是否对应 opponent color coding 会很有意思。

6. **Cross-camera robustness**：SPCD 是 smartphone photography dataset，pairs 来自不同手机。如果扩展到 cross-illuminant（同一 scene 不同 lighting），metric 还 work 吗？这关系到 color constancy 和 illumination-invariant representation。

7. **Latent space 的 perceptual uniformity 验证**：paper 说 "empirically verify the perceptual uniformity of the learned color image representation from multiple aspects"，但 abstract/section 里没明确给。能否像 CIEDE2000 那样画 MacAdam ellipses 看看 latent space 是否真的 uniform？这是 color science community 最关心的。

---

## 8. 主要参考链接

- **CD-Flow GitHub**: [https://github.com/haoychen3/CD-Flow](https://github.com/haoychen3/CD-Flow)
- **SPCD dataset / CD-Net paper (Wang et al. 2023 TPAMI)**: [https://arxiv.org/abs/2307.07689](https://arxiv.org/abs/2307.07689)
- **RealNVP**: [https://arxiv.org/abs/1605.08803](https://arxiv.org/abs/1605.08803)
- **Glow (invertible 1×1 conv)**: [https://arxiv.org/abs/1807.03039](https://arxiv.org/abs/1807.03039)
- **LPIPS**: [https://arxiv.org/abs/1801.03924](https://arxiv.org/abs/1801.03924)
- **DISTS**: [https://arxiv.org/abs/2004.07728](https://arxiv.org/abs/2004.07728)
- **Exploding inverses in invertible NN (Behrmann et al. 2021)**: [https://arxiv.org/abs/2001.09468](https://arxiv.org/abs/2001.09468)
- **Shapley & Hawken 2011 Vision Research (color in cortex)**: [https://pubmed.ncbi.nlm.nih.gov/21277891/](https://pubmed.ncbi.nlm.nih.gov/21277891/)
- **CIEDE2000 (Luo, Cui, Rigg 2001)**: [https://onlinelibrary.wiley.com/doi/abs/10.1002/col.1049](https://onlinelibrary.wiley.com/doi/abs/10.1002/col.1049)
- **FLIP**: [https://research.nvidia.com/publication/2020-06_FLIP](https://research.nvidia.com/publication/2020-06_FLIP)
- **Normalizing flows review (Papamakarios et al. 2021)**: [https://arxiv.org/abs/1912.02762](https://arxiv.org/abs/1912.02762)
- **Butteraugli (Guetzli)**: [https://arxiv.org/abs/1703.04421](https://arxiv.org/abs/1703.04421)
- **Kede Ma's lab page**: [http://www.kedema.org/](http://www.kedema.org/)

---

## 9. 总结直觉

如果让我用一句话 capture CD-Flow 的 essence：**"用 bijective normalizing flow 学一个 reparametrization，让 RGB 图像在 latent space 上变成'感知均匀'的坐标系，然后用 Euclidean 距离测 CD —— bijectivity 保证是 proper metric，multi-scale autoregressive 结构自然对应 V1 的 single/double-opponent cells，并对几何扰动鲁棒。"**

它跟传统 CIEDE2000 的关系，类似 BERT 跟 handcrafted NLP features 的关系 —— data-driven 取代 handcrafted formula，但 architecture design 仍然包含 strong inductive bias（multi-scale 对应 visual cortex, bijectivity 对应 metric property, Gaussian prior 对应 perceptual uniformity）。

对 photographic images 来说，这是第一个同时满足 (i) biologically plausible, (ii) 数学 proper metric, (iii) accurate, (iv) robust to geometric distortion 的 CD metric。挺 elegant 的工作。
