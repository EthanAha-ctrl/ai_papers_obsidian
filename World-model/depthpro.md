---
source_pdf: depthpro.pdf
paper_sha256: 1108db233e3082286a522961a869f1c24bbf0f7b967aa85dc346809c294ac9af
processed_at: '2026-08-03T19:56:52-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Depth Pro：如何让单目深度估计又快、又准、又清晰

如果让我给这篇 Apple 的 paper 一句话总结：**以前的 monocular depth estimation model 总是在做 trade-off，要么准但糊，要么细但慢得要命，要么 metric 但吃相机参数；Depth Pro 把这些 trade-off 全打碎了，用一套非常工程化的组合拳，同时做到了 metric、sharp 和 sub-second。**

为了 build your intuition，我们一层一层拆开看。

---

## 1. 痛点在哪？为什么需要 Depth Pro？

想象你在做 novel view synthesis（单图转 3D 视角），你需要一张 depth map。
- 如果用 Marigold（diffusion model），它画得很细，连头发丝都能画出来，但它需要 50 步去噪，跑一次要 4 秒多。而且它只能预测 relative depth，没有绝对尺度，你不知道场景里的人到底是 1.6 米还是 3 米高。
- 如果用 Metric3D，它能给你绝对尺度的 metric depth，但是它要求你必须提供相机的 focal length（焦距）。如果是网上的随便一张图，没有 EXIF 信息，它就歇菜了。
- 如果用 PatchFusion，它确实分辨率高、细节多，但它把图切成无数块去算，跑一张图要 84 秒，根本没法做交互应用。

Apple 的要求非常苛刻：
1. **Metric**：必须带绝对物理尺度，不依赖相机内参。
2. **Sharp**：2.25 MP 分辨率，头发丝、毛皮边缘得清晰，消除 view synthesis 里的 "flying pixels"。
3. **Fast**：0.3 秒出结果，支持实时交互。

怎么做到的？主要靠四个核心 trick。

---

## 2. 核心架构直觉：多尺度放大镜 + 全局宏观视角

传统的 ViT（Vision Transformer）有个致命弱点：self-attention 的计算复杂度是 $O(N^2)$（$N$ 是 token 数）。如果直接在 1536×1536 的大图上跑 ViT，显存直接爆炸，计算量是 384×384 的 256 倍。

Depth Pro 的架构思路（Figure 3）非常精妙：**不要去魔改 ViT 内部结构，直接拿现成的 plain ViT-L DINOv2 当黑盒，改变你的使用方式。**

它是这样工作的：
1. 输入图固定 resize 到 1536×1536。
2. **Patch Encoder**：像拿放大镜一样，把 1536×1536 的图切成 35 个 384×384 的小 patch，有些重叠。重叠是为了避免边缘出现接缝。所有 patch 丢给同一个 ViT 处理，这逼着 ViT 学到 scale-invariant 的特征。
3. **Image Encoder**：同时把整张图 downsample 到 384×384，单独喂给另一个 ViT。这个 encoder 负责看全局，回答"这是什么场景、整体纵深大概多少"。
4. **Merge**：35 个 patch 的特征怎么拼回大图？用 Voronoi partition（泰森多边形）。以每个 patch 的中心点为种子，离哪个种子近的像素就归哪个 patch，这样重叠区域的特征自然平滑过渡，没有 seam。

**Intuition**：Patch encoder 提供高频局部细节（"这根头发在哪"），Image encoder 提供低频全局上下文（"这是个大场景还是小房间"），两者一结合，既看得远又看得细。而且因为 plain ViT 可以直接替换成未来更好的 backbone（比如 DINOv3），工程友好度极高。

---

## 3. 数学与 Loss：为什么算 1/d 而不是 d？

网络实际预测的不是 depth $d$，而是 **canonical inverse depth** $C$。最终 metric depth 的换算公式是：

$$D_m = \frac{f_{px}}{w \cdot C}$$

变量解释：
- $D_m$：最终的 metric depth（绝对物理距离，单位米）
- $f_{px}$：focal length in pixels（像素单位的焦距）
- $w$：image width in pixels
- $C$：网络预测的 canonical inverse depth

**为什么要算 $1/d$？**
看 Table 10 的实验数据，在 0-1m 的极近处，预测 inverse depth 的 $\delta_1$ 是 0.730，预测 depth 只有 0.657。
因为 $1/d$ 在 $d$ 很小时会把误差放大。近处物体的微小误差对视觉体验极其敏感，远处物体差几米无所谓。预测 $1/d$ 天然让网络把注意力集中在近处，这对 novel view synthesis 是最完美的表示。

**Loss 设计**
只用 L1 loss 会出现 blurry edges，这是 L1 的通病（倾向取 median）。为了逼出 sharp boundaries，引入了多尺度 gradient loss：

$$\mathcal{L}_{*,p,M}(C, \hat{C}) = \frac{1}{M} \sum_j^M \frac{1}{N_j} \sum_i^{N_j} |\nabla_* C_i^j - \nabla_* \hat{C}_i^j|^p$$

变量解释：
- $\nabla_*$：空间导数算子，可以是 Scharr (S) 算一阶梯度，或 Laplacian (L) 算二阶梯度
- $p$：范数阶数（1 或 2）
- $M$：scales 的数量（论文用 6 个尺度）
- $j$：尺度索引，每个尺度做高斯模糊并降采样一半
- $N_j$：尺度 $j$ 上的像素总数

直觉上，就是在 6 个由清晰到模糊的分辨率上，强制网络的输出在边缘跳变处和 GT 的一阶、二阶导数完全一致。一阶抓边缘，二阶抓角点和极值。

---

## 4. 反向训练策略：先 Real，后 Synthetic

这是这篇 paper 最反直觉但最 brilliant 的点。

在 sim2real 领域，大家通常是先在 synthetic data（完美 GT）上 pretrain，再到 real data 上 finetune。
Depth Pro 偏不这样，它倒过来：

- **Stage 1**：在所有 real + synthetic 数据上训练。只算 L1 loss，加一点 synthetic data 上的 gradient loss。
- **Stage 2**：**只用 synthetic 数据**继续训练。加上所有的 gradient loss（一阶、二阶）。

为什么？因为 real data 的 depth GT 在物体边缘处通常是错的、缺失的、misaligned 的。如果在 Stage 2 继续用 real data 算 gradient loss，网络会被边缘的噪声带坏，学不出 sharp edge。Synthetic data 的边缘是完美像素级的，所以 Stage 2 专心利用 synthetic 的完美边缘来"磨快"网络。

Table 13 证明了这一点：
- Depth Pro（先 real 后 synthetic）：Hypersim F1 = 0.465
- 传统策略（先 synthetic 后 real）：F1 = 0.095，直接崩溃。

---

## 5. Focal Length Estimation：凭空猜相机参数

既然要 zero-shot metric depth，但用户不给你 EXIF，怎么办？自己估 focal length。

Depth Pro 训练了一个轻量级的 head 专门估 horizontal angular field-of-view。它的架构（对应 Figure 3）也很直觉：
它并行接收两路 feature：
1. 主 depth network 中冻结的 frozen features（已经包含了场景深度几何信息）。
2. 一个单独训练的 ViT image encoder 输出的 task-specific features（专门为了估焦距学的特征）。

两路特征 concat 后，过一个 3 层 conv head（kernels: 3,3,6; strides: 2,2,1），输出一个标量。
训练时只用 L2 loss，并且一定要在 depth network 训完之后再单独训这个 head，避免两个目标互相打架。

效果在 Table 3 里极其亮眼：在 PPR10K 数据集上，估计误差 <25% 的图像占 64.6%，而之前的 SOTA（SPEC）只有 34.6%，直接碾压。

---

## 6. 评价指标创新：拿抠图数据集来评 Depth

要证明自己的边缘很 sharp，怎么量化？以前大家看深度图觉得"好像挺糊的"，但拿不出数据。

Depth Pro 提出了一个基于相邻像素 depth 比例的 occluding contour 指标：

$$c_d(i,j) = \left\lceil \frac{d(j)}{d(i)} > \left(1 + \frac{t}{100}\right) \right\rceil$$

如果像素 $j$ 的深度比 $i$ 大 $t\%$，就判定 $i$ 到 $j$ 之间存在一条 occluding contour。$t$ 取 5 到 25，算加权的 F1 score。

最天才的地方在于：没有 GT depth 怎么办？拿 **image matting** 和 **segmentation** 的数据集来用！
如果 mask 上 $i$ 是前景，$j$ 是背景，那它们之间肯定有 contour。于是 Depth Pro 拿来做头发抠图的 AM-2k 数据集、做人像抠图的 P3M-10k，全部变成 depth 边缘评测集，直接解决了 depth GT 无法精确标注头发丝的历史难题。

---

## 7. 实验结果碾压局

看 Table 5 的 runtime 比较，直观感受一下"降维打击"：

| Method | Params | Native Res | tHD (ms) ↓ |
| :--- | :--- | :--- | :--- |
| Marigold | 949M | 768×768 | 4433.6 |
| Metric3D v2 | 1.378G | 616×1064 | 1299.7 |
| PatchFusion | 203M | Original | 84029.9 |
| **Depth Pro** | **504M** | **1536×1536** | **341.3** |

Depth Pro 的原生输出分辨率是 2.25 MP，是 Marigold 的近 4 倍，是 Metric3D v2 的 3.5 倍。但在 V100 上只要 0.34 秒。比 Marigold 快 13 倍，比 PatchFusion 快 246 倍，比 Metric3D v2 快 4 倍，参数量还只有 Metric3D v2 的三分之一。

看 Table 2 的边界锐度 F1：
- Depth Pro 在 AM-2k 上的 Recall 是 0.173，Metric3D v2 只有 0.024，Depth Anything v2 只有 0.107。细节捕捉能力翻了数倍。

---

## 8. Build Intuition 总结

1. **架构取舍**：plain ViT + 多尺度 patch 切分，比去魔改 Swin、Perceiver 更优雅，因为 backbone 进步的红利你可以白嫖。
2. **数据策略**：当 real data 的某些属性（如边缘 GT）不可靠时，利用 synthetic data 完美的属性做针对性微调。这打破常规 sim2real 的刻板印象。
3. **评价体系**：用 matting mask 替代 depth GT 来评价边界，这是一个极其聪明的 cross-task 借力。
4. **工程目标驱动**：fixed resolution 看似限制灵活性，实则保证了恒定显存和极速推理，这是 foundation model 走向产品化的必经之路。

Depth Pro 用最工程化的方式，解决了 long-standing 的 metric depth 估计问题，是一个典型的"不只是堆模型，而是解构问题并设计针对性机制"的优秀范例。

*References:*
- [Depth Pro arXiv Paper](https://arxiv.org/abs/2410.02073)
- [Depth Pro GitHub Repo](https://github.com/apple/ml-depth-pro)
- [DINOv2 Backbone](https://arxiv.org/abs/2304.07193)
- [Marigold: Diffusion-based Depth](https://arxiv.org/abs/2312.02145)
- [PatchFusion](https://arxiv.org/abs/2312.08179)
- [Metric3D v2](https://arxiv.org/abs/2312.06505)

---

# Depth Pro：Apple 的 Zero-Shot Metric Monocular Depth Foundation Model 深度解析

## 1. 这篇 Paper 要解决什么问题

Apple 这篇 paper 的核心目标是打造一个 monocular depth estimation 的 foundation model，同时满足三个此前被 trade-off 的需求：

1. **Metric depth with absolute scale**：之前的工作如 MiDaS、Marigold 只能预测 relative depth（scale-and-shift invariant），导致下游应用（如 novel view synthesis 中"从这个角度、63mm 处合成视图"这种需求）无法实现。
2. **Sharp boundaries at high resolution**：单像素级精确勾勒 hair、fur、vegetation 等 thin structures，消除 "flying pixels"。
3. **Sub-second latency**：交互式应用要求 <1 秒。Marigold 用 diffusion 50 步去噪要 4-5 秒；PatchFusion 要 84 秒。

Depth Pro 的 headline 数字：**2.25-megapixel 输出，V100 上 0.3 秒**，且**不需要任何 camera intrinsics 输入**。

GitHub repo: https://github.com/apple/ml-depth-pro

---

## 2. 网络架构详解（对应 Figure 3）

### 2.1 核心设计哲学：Multi-Scale Patch-Based ViT

Depth Pro 没有去改造 ViT 内部结构（如 Swin 的 shifted window、Perceiver 的 cross-attention），而是直接复用 plain ViT-L DINOv2 作为 backbone，通过 **multi-scale patch-based 应用方式** 拼出 high-resolution 输出。

这种设计有两个 motivation：

**Motivation A**：plain ViT 的 self-attention 复杂度是 $O(N^2)$，其中 $N$ 是 token 数。如果直接在 1536×1536 上跑 ViT，token 数会是 384×384 输入的 16 倍，attention 矩阵计算量是 256 倍（quartic in image dimension）。Patch-based 处理把 attention 限制在每个 patch 内部，complexity 线性叠加。

**Motivation B**：可以直接 swap in 任何 future ViT pretrained backbone（如 DINOv3、SigLIP v2），无需重新设计架构和 retrain。

### 2.2 具体数据流（Figure 3 解析）

输入图像先 downsample 到固定的 **1536×1536**（为什么固定？避免 OOM、保证 constant runtime）。

然后从多个 scale 上提取 patch：

- **Patch encoder**（ViT-L DINOv2，权重跨 scale 共享）：
  - Scale 1（最细）：384×384 patches，**25 个**，overlapping 以避免 seam
  - Scale 2：384×384 patches，**9 个**，overlapping
  - 更粗 scale：...共 **35 个 patches**
  - 所有 patch concatenate 到 batch dimension 上，一次性 forward

- **Image encoder**（独立 ViT-L DINOv2）：处理整个 384×384 downsampled image，提供 **global context**，anchor patch predictions

每个 384×384 patch 经过 ViT 后输出 **24×24 feature map**（384/16=24，patch size 16）。

Finest scale 还额外提取 **intermediate features**（Features 1 & 2 in Fig. 3），共 25+25=50 个 feature patches，用于捕捉 finer-grained details。

### 2.3 Merge Operation（Appendix C.1）

多个 overlapping patches 怎么 merge 成 single feature map？用 **Voronoi partition**：

- 把每个 patch 的中心作为 seed
- 生成 Voronoi 图，每个 feature map 像素归到最近的 patch seed
- 每个 patch 只保留被 Voronoi cell 覆盖的区域

这样在 patch 边界处不会出现 seam artifact。Overlapping patches 保证 receptive field 部分覆盖邻居，提供 smooth transition。

### 2.4 Decoder

用 DPT decoder（Ranftl et al., 2021）的结构，把 multi-scale features upsample 并 fuse 成 final depth map。Decoder 通道数 256。

### 2.5 关键设计选择的 Intuition

**为什么 fixed 1536×1536**：1536 = 4 × 384，所以 4×4=16 个 non-overlapping patch 即可覆盖；加 overlapping 后变 25 个。Fixed resolution 让 runtime 完全可预测，且避免 large image 上的 OOM。

**为什么 image encoder 独立**：patch encoder 看到的是局部 384×384，缺 global context（"这是一张室内还是室外？场景的 scale 是什么？"）。Image encoder 提供 global anchor，相当于"告诉你这张图大概是怎样的场景"，patch encoder 负责"在这个 context 下，这段 hair 的 depth 边界在哪"。

**为什么 patch encoder 跨 scale 共享权重**：作者直觉认为这能让 encoder 学到 **scale-invariant representation**——同一个物体在不同 scale 下应该有类似的 feature pattern。

---

## 3. Loss Functions 与公式详解

### 3.1 Canonical Inverse Depth 表示

网络预测的是 **canonical inverse depth** $\check{C} = \check{f}(I)$，然后通过 focal length 转换为 metric depth：

$$D_m = \frac{f_{px}}{w \cdot C}$$

变量解释：
- $D_m$：metric depth（米）
- $f_{px}$：focal length in pixels
- $w$：image width in pixels
- $C$：canonical inverse depth

这个公式来自 Metric3D (Yin et al., 2023) 的 canonical camera 思路。**Canonical** 意味着把所有图像"归一化"到一个 reference focal length，避免训练时不同相机的 scale 歧义。

**为什么 inverse depth 而非 depth**：Table 10 的 ablation 给出了答案：
- Inverse-depth: 0-1m 范围 $\delta_1$ = 0.730
- Log-depth: 0-1m 范围 $\delta_1$ = 0.700
- Depth: 0-1m 范围 $\delta_1$ = 0.657

Inverse depth 把近处物体的误差放大（因为 1/d 对小 d 敏感），这正好对 novel view synthesis 有利——近处物体的精度对 3D 效果更关键。

### 3.2 主损失 $\mathcal{L}_{MAE}$（Eq. 1）

$$\mathcal{L}_{MAE}(\hat{C}, C) = \frac{1}{N} \sum_i^N |\hat{C}_i - C_i|$$

变量：
- $\hat{C}$：ground-truth canonical inverse depth
- $C$：predicted canonical inverse depth
- $N$：pixel 数
- $i$：pixel index

对 real-world datasets，会 **discard top 20% error pixels per image**（trimmed loss）。这是因为 real depth maps 在 boundary 处常有 mismatch/missing data，直接用 L1 会被这些噪声主导。Trim 后训练更稳定。

对 non-metric datasets，predictions 和 GT 都先用 **median absolute deviation from median**（MiDaS 的 SSI normalization）归一化，再算 loss。

### 3.3 Multi-Scale Derivative Loss（Eq. 2）

$$\mathcal{L}_{*,p,M}(C, \hat{C}) = \frac{1}{M} \sum_j^M \frac{1}{N_j} \sum_i^{N_j} |\nabla_* C_i^j - \nabla_* \hat{C}_i^j|^p$$

变量：
- $\nabla_*$：spatial derivative operator，* 可以是 Scharr (S) 或 Laplacian (L)
- $p$：error norm，1 或 2
- $M$：scales 数量（论文用 6）
- $j$：scale index，每 scale blur + downsample by factor 2
- $N_j$：scale $j$ 上的 pixel 数

三个 shorthand：
- $\mathcal{L}_{MAGE} = \mathcal{L}_{S,1,6}$：Mean Absolute Gradient Error，一阶 Scharr + L1
- $\mathcal{L}_{MALE} = \mathcal{L}_{L,1,6}$：Mean Absolute Laplace Error，二阶 Laplacian + L1
- $\mathcal{L}_{MSGE} = \mathcal{L}_{S,2,6}$：Mean Squared Gradient Error，一阶 Scharr + L2

**为什么需要 derivative loss**：单纯 L1 loss 会产生 blurry edges（这是 L1 的 known artifact，倾向预测 median 值）。Gradient loss 直接惩罚 edge 位置的预测错误，迫使网络学到 sharp transitions。

**Multi-scale 的意义**：单一 scale 的 gradient loss 只关注最高频边界，但 6 个 scale 让网络在不同模糊度上都有 sharp prediction——这间接训练网络"知道 boundary 在哪"，即使最终输出在 finest scale。

### 3.4 Training Curriculum

两阶段策略，**反传统**：

**Stage 1**（250 epochs）：
- 用所有 datasets（real + synthetic）
- Losses：$\mathcal{L}_{MAE}$ on metric datasets，$\mathcal{L}_{SSI-MAE}$ on non-metric
- 加上 **$\mathcal{L}_{SSI-MAGE}$ on synthetic datasets only**（scale-shift invariant gradient loss）
- 为什么 synthetic only：real depth 在 boundary 处 noisy，gradient loss 会被噪声主导

**Stage 2**（100 epochs）：
- **只用 synthetic datasets**（Hypersim, Tartanair, Synscapes, Urbansyn, Dynamic Replica, Bedlam, IRS, Virtual Kitti2, SAIL-VOS-3D）
- Losses：$\mathcal{L}_{MAE} + \mathcal{L}_{MSE} + \mathcal{L}_{MAGE} + \mathcal{L}_{MALE} + \mathcal{L}_{MSGE}$
- 目的：sharpen boundaries

**反传统在哪**：传统 pipeline 是"先 synthetic pretrain → real finetune"（sim2real），因为 real 数据更真实。Depth Pro 反过来"先 real → 再 synthetic"，因为 synthetic 有 pixel-accurate GT，能提供 sharp boundary supervision，而 real 数据的 boundary GT 不可靠会污染 sharpness。

Table 13 的 ablation 证实了这一点：
- 3A（Ours，real→synthetic）：Hypersim F1 = 0.465
- 3B（single stage）：F1 = 0.478（略高，但 metric depth 略差）
- 3C（synthetic→real，传统）：F1 = 0.095，$\delta_1$ = 0.5（崩了）

3C 崩溃的原因：synthetic 学到的 sharp features 被 real 的 noisy boundary 磨平。

### 3.5 Stage 1 Loss 选择 Ablation（Table 11）

- 1A：只 MAE
- 1B：MAE + gradient on synthetic only
- 1C（Ours）：MAE + SSI-MAGE on synthetic only
- 1D：MAE + gradient on all datasets

1C 在 Apolloscape 上 F1=0.442，远好于 1A 的 0.221。1D 用所有 datasets 算 gradient 反而比 1B 只用 synthetic 差，因为 real 噪声干扰。

---

## 4. Boundary Evaluation Metrics 详解

这是 paper 的重要 contribution，因为 prior benchmarks 很少量化 boundary sharpness。

### 4.1 Occluding Contour 定义

对于两个相邻 pixels $i, j$，定义 occluding contour：

$$c_d(i,j) = \left\lceil \frac{d(j)}{d(i)} > \left(1 + \frac{t}{100}\right) \right\rceil$$

变量：
- $d(i)$, $d(j)$：pixels $i, j$ 的 depth
- $t$：threshold percentage（论文取 5 到 25）
- $\lceil \cdot \rceil$：Iverson bracket，条件成立为 1，否则为 0

**Intuition**：如果 pixel $j$ 的 depth 比 pixel $i$ 大 $t\%$ 以上，意味着 $i$ 是前景，$j$ 是背景，$i \to j$ 之间存在 occluding contour。

### 4.2 Precision 和 Recall（Eq. 3）

$$P(t) = \frac{\sum_{i,j \in N(i)} c_d(i,j) \wedge c_{\hat{d}}(i,j)}{\sum_{i,j \in N(i)} c_d(i,j)}$$

$$R(t) = \frac{\sum_{i,j \in N(i)} c_d(i,j) \wedge c_{\hat{d}}(i,j)}{\sum_{i,j \in N(i)} c_{\hat{d}}(i,j)}$$

变量：
- $N(i)$：pixel $i$ 的邻居集合
- $c_d$：从 GT depth $d$ 推出的 contour
- $c_{\hat{d}}$：从 predicted depth $\hat{d}$ 推出的 contour
- $\wedge$：logical AND

**Precision** = 预测的 contour 中有多少是对的（避免假阳，即避免在平坦区域预测出 fake boundary）
**Recall** = GT contour 中有多少被预测到了（避免假阴，即不要漏掉真实边界）

### 4.3 多阈值加权 F1

阈值 $t$ 从 $t_{min}=5$ 线性到 $t_{max}=25$，权重偏向高阈值（更大的 depth jump 更重要）。最终 F1 = weighted average。

### 4.4 用 Matting/Segmentation 数据集扩展 Evaluation

关键创新：没有 GT depth 也能 eval boundary！

对于 binary mask $b$（来自 matting/saliency/segmentation dataset）：

$$c_b(i,j) = b(i) \wedge \neg b(j)$$

意思是：如果 pixel $i$ 是前景（mask=1），pixel $j$ 是背景（mask=0），则 $i \to j$ 之间存在 contour。

这样可以用 **AM-2k**（hair matting）、**P3M-10k**（portrait matting）、**DIS-5k**（salient object segmentation）这些 high-quality annotation datasets 来评估 depth boundary——即使它们没有 GT depth。

**Caveat**：binary mask 标注的是 whole object，所以只能算 Recall（覆盖多少 GT contour），不能算 Precision（因为 mask 没标出 object 内部的 contour，predicted internal contour 会被误判为 false positive）。

### 4.5 NMS 抑制 blurry edges

为惩罚 blurry predicted edges（即一个真实 contour 被预测成 3-4 个 pixels 宽的"渐变带"），在 $c_{\hat{d}}$ 的 connected components 内做 non-maximum suppression。

---

## 5. Focal Length Estimation

### 5.1 为什么需要这个 Head

Metric depth 需要 focal length：$D_m = f_{px} / (wC)$。如果输入图没有 EXIF（如截图、网络图），怎么获得 $f_{px}$？

之前的工作要么假设 intrinsics 已知（Metric3D, ZeroDepth），要么用 separate network 估计（UniDepth, SPEC）。Depth Pro 把 focal length estimation 嵌入到同一个 model 中。

### 5.2 架构

Focal length head 包含两个 feature source：

1. **Frozen features from depth network**：从 depth estimation network 的 intermediate features 中取（已 trained，包含 scene layout 信息）
2. **Separate ViT image encoder**：task-specific encoder，专门为 focal length 训练

两个 feature streams concat 后过一个 **3-layer conv head**：
- Conv kernel sizes: 3, 3, 6
- Strides: 2, 2, 1
- Channels: 128 → 64 → 32 → 1
- ReLU between layers
- 输出：单张图的 horizontal angular FOV

Loss：$\mathcal{L}_2$ on FOV。

### 5.3 Training 时机

**Depth network 训练完之后再单独训练 focal length head**。理由：

1. 避免 depth 和 focal length loss 的 balancing 问题（joint training 容易 conflict）
2. 可以用不同 datasets：depth network 用某些 narrow-domain single-camera datasets（如 KITTI），但 focal length head 不应该用这些（会过拟合到固定 focal length）；focal length head 可以用 large-scale 有 EXIF 但无 depth 的 datasets

### 5.4 Ablation（Table 14）

- Encoder for depth only（frozen depth features + small head）：$\delta_{25\%} = 60.0$
- Encoder for focal length only（separate ViT from scratch）：74.4
- Encoder for depth + refinement network：63.6
- **Parallel encoders（Ours）**：78.2

Parallel 最好，说明需要 depth features + task-specific features 都不可或缺。

---

## 6. 实验结果分析

### 6.1 Zero-Shot Metric Depth（Table 1）

测试 datasets：Booster, ETH3D, Middlebury, NuScenes, Sintel, Sun-RGBD——这些都没有被任何对比方法用于 training。

$\delta_1$ metric：predicted 和 GT depth 误差在 25% 以内的 pixel 占比。

| Method | Booster | ETH3D | Middlebury | NuScenes | Sintel | Sun-RGBD | Avg Rank↓ |
|---|---|---|---|---|---|---|---|
| DepthAnything v2 | 59.5 | 36.3 | 37.2 | 17.7 | 5.9 | 72.4 | 5.8 |
| Metric3D v2 | 39.4 | 87.7 | 29.9 | 82.6 | 38.3 | 75.6 | 3.7 |
| UniDepth | 27.6 | 25.3 | 31.9 | 83.6 | 16.5 | 95.8 | 4.2 |
| **Depth Pro** | **46.6** | **41.5** | **60.5** | **49.1** | **40.0** | **89.0** | **2.5** |

Depth Pro 在 Booster、Middlebury、Sintel 上拿了第一，平均排名 2.5 最佳。

**重要 caveat**：Depth Anything v2 和 Metric3D v2 用了 per-domain crop size 或 domain-specific models（标记为灰色），违反了 strict zero-shot。Depth Pro 不需要这些 trick。

### 6.2 Zero-Shot Boundary Accuracy（Table 2）

这是 Depth Pro 的杀手锏。F1 score for Sintel/Spring/iBims（有 GT depth），Recall for AM-2k/P3M-10k/DIS-5k（matting/segmentation datasets）。

| Method | Sintel F1↑ | Spring F1↑ | iBims F1↑ | AMR↑ | P3M R↑ | DIS R↑ |
|---|---|---|---|---|---|---|
| PatchFusion | 0.312 | 0.032 | 0.134 | 0.061 | 0.109 | 0.068 |
| DepthAnything v2 | 0.228 | 0.056 | 0.111 | 0.107 | 0.131 | 0.056 |
| Marigold | 0.068 | 0.032 | 0.149 | 0.064 | 0.101 | 0.049 |
| **Depth Pro** | **0.409** | **0.079** | **0.176** | **0.173** | **0.168** | **0.077** |

Depth Pro 在所有 6 个 datasets 上都最好，比 diffusion-based Marigold 和 tile-based PatchFusion 都强。

### 6.3 Runtime（Table 5）

V100 GPU，batch=1：

| Method | Params | Native Resolution | tHD (ms) |
|---|---|---|---|
| DPT | 123M | 384×384 (0.15 MP) | 30.6 |
| Marigold | 949M | 768×768 (0.59 MP) | 4433.6 |
| Metric3D v2 | 1.378G | 616×1064 (0.66 MP) | 1299.7 |
| PatchFusion | 203M | Original | 84029.9 |
| ZeroDepth | 233M | Original | 8795.7 |
| **Depth Pro** | **504M** | **1536×1536 (2.36 MP)** | **341.3** |

Depth Pro 的 native resolution 是 Metric3D v2 的 3.5 倍，但 runtime 只是其 1/4，参数量也是其 1/2.7。PatchFusion 慢到 84 秒是因为 tile-based pipeline 需要 per-tile inference + 多步融合。

### 6.4 Focal Length Estimation（Table 3）

测试 datasets：DDDP, FiveK, PPR10K, RAISE, SPAQ, ZOOM。指标 $\delta_{25\%}$ 和 $\delta_{50\%}$：relative error 小于 25%/50% 的图像占比。

| Method | DDDP $\delta_{25\%}$ | FiveK $\delta_{25\%}$ | PPR10K $\delta_{25\%}$ | RAISE $\delta_{25\%}$ | SPAQ $\delta_{25\%}$ | ZOOM $\delta_{25\%}$ |
|---|---|---|---|---|---|---|
| UniDepth | 6.8 | 24.8 | 13.8 | 35.4 | 44.2 | 20.4 |
| SPEC | 14.6 | 30.2 | 34.6 | 49.2 | 50.0 | 23.2 |
| im2pcl | 7.3 | 28.0 | 24.2 | 51.8 | 26.6 | 22.4 |
| **Depth Pro** | **66.9** | **74.2** | **64.6** | **84.2** | **68.4** | **69.8** |

Depth Pro 在所有 datasets 上的 $\delta_{25\%}$ 都超过 60%，第二好的 SPEC 最好才 50%。这是数量级提升。

PPR10K 上 Depth Pro 64.6% vs SPEC 34.6%，差 30 个百分点。

---

## 7. Ablation Studies 关键洞察

### 7.1 Native Output Resolution（Table 7）

把 GT depth 下采样到不同 resolution 再上采样回原 resolution，看 metric 退化：

| Output Resolution | Log10↓ | AbsRel↓ | F1↑ |
|---|---|---|---|
| 1536×1536 (Depth Pro) | 0.019 | 0.004 | 0.311 |
| 768×768 (Marigold) | 0.048 | 0.010 | 0.131 |
| 518×518 (Depth Anything v2) | 0.084 | 0.016 | 0.065 |
| 384×384 (DPT) | 0.123 | 0.024 | 0.044 |

**Resolution 翻倍，F1 提升约 3 倍**。但作者强调这只是上界——实际 method 即使在 high resolution 上也未必达到，因为还需要 architecture 和 loss 配合。

### 7.2 Backbone Comparison（Table 8）

在 5 个 RGB-D datasets 上训练，评估 Booster, Hypersim, Middlebury, NYUv2：

- **ViT-L DINOv2**：AbsRel=0.040, Log10=0.129（最好）
- ViT-L MAE：AbsRel=0.041, Log10=0.150
- ViT-L BeiTv2：AbsRel=0.042, Log10=0.134
- ViT-L CLIP：AbsRel=0.057（差）
- ConvNext-XXL：AbsRel=0.075（差）
- SegAnything ViT-L：AbsRel=0.087（差）

DINOv2 显著领先，self-supervised pretraining 在 dense prediction 上比 CLIP、supervised ImageNet 都强。

### 7.3 High-Resolution Architecture Comparison（Table 9）

所有模型在 1536×1536 上跑：

| Method | Latency | NYUv2 $\delta_1$ | iBims F1 | DIS R |
|---|---|---|---|---|
| ConvNext-XXL | 304ms | 68.0 | 0.134 | 0.031 |
| SegAnything ViT-L | 349ms | 53.2 | 0.140 | 0.051 |
| SWINv2-L | 272ms | 58.4 | 0.117 | 0.028 |
| ViT-L DINOv2 (naive upscale) | 392ms | 96.5 | 0.161 | 0.065 |
| **Depth Pro** | **341ms** | 96.1 | **0.177** | **0.080** |

Depth Pro 比 naive ViT-L DINOv2 upscale 还快（341 vs 392ms），boundary recall 高 23%。说明 multi-scale patch-based 比直接 upscale ViT 更高效。

---

## 8. 与 Prior Work 的关系

### 8.1 MiDaS 系列（Ranftl et al., 2022, 2021）

MiDaS 引入 SSI-invariant loss 和 multi-dataset training，但只做 relative depth，输出 resolution 低（384×384）。Depth Pro 继承了 SSI 思路但扩展到 metric + high resolution。

Paper: https://arxiv.org/abs/2103.13413

### 8.2 Metric3D / Metric3D v2（Yin et al., 2023; Hu et al., 2024）

Metric3D 引入 canonical camera transformation：用 focal length 把图像 warp 到 canonical space，预测 canonical depth，再 warp 回来。Depth Pro 借用了 $D_m = f_{px}/(wC)$ 公式。

但 Metric3D v2 用 surface normal 作 auxiliary output，参数量 1.378G，是 Depth Pro 的 2.7 倍，runtime 是 4 倍。

Paper: https://arxiv.org/abs/2307.10984

### 8.3 Marigold（Ke et al., 2024）

把 Stable Diffusion finetune 用于 relative depth estimation，质量高但慢（4.4 秒，50 步 diffusion）。Depth Pro 在 boundary accuracy 上超过 Marigold，runtime 快 13 倍。

Paper: https://arxiv.org/abs/2312.02145

### 8.4 PatchFusion（Li et al., 2024a）

BoostingDepth 的 end-to-end 版本，tile-based + image-adaptive patch sampling。Runtime 慢（84 秒），且是 metric depth 但需要 intrinsics。

Paper: https://arxiv.org/abs/2312.08179

### 8.5 UniDepth（Piccinelli et al., 2024）

第一个不需要 intrinsics 的 universal metric depth model，用 camera embedding in spherical space。Depth Pro 在 focal length estimation 上大幅超过 UniDepth。

Paper: https://arxiv.org/abs/2403.18913

### 8.6 Depth Anything v1/v2（Yang et al., 2024a, 2024b）

DA 用 1.5M unlabeled images self-supervised pretrain，v2 用 600M synthetic data。Metric depth 是 domain-specific（indoor/outdoor 分别），不算 strict zero-shot。Boundary 不如 Depth Pro sharp。

Paper: https://arxiv.org/abs/2406.09414

---

## 9. Limitations

作者承认：
- **Translucent surfaces**：玻璃、水等半透明物体，pixelwise depth 定义本身 ill-posed
- **Volumetric scattering**：雾、烟等参与介质的 depth 也 ambiguous

这两点其实是 monocular depth 的根本问题，Depth Pro 没有特别处理。

---

## 10. 个人 Build Intuition 的关键 Takeaways

1. **Patch-based ViT 比 ViT 改造更优**：不要动 ViT 内部，而是改 application pattern。这让 backbone 升级成本几乎为零。
2. **Canonical inverse depth 是 metric depth 的关键 representation**：解决 multi-camera scale ambiguity，同时近处物体精度更高（对 view synthesis 友好）。
3. **反传统 curriculum（real → synthetic）**：当 real data GT 在某些方面 noisy 时，应该先用 real 学 generalization，再用 synthetic 学 sharpness。这个 insight 可推广到其他 dense prediction 任务。
4. **Boundary evaluation 可以无需 depth GT**：用 matting/segmentation mask 推 occluding contour，绕过 depth GT 缺失问题。这是一个很有价值的 benchmark 思路。
5. **Focal length estimation 需要独立 encoder**：joint training 会 conflict，parallel encoder + frozen depth features 是最佳配置。
6. **Multi-scale derivative loss 比 L1 loss 更能 sharpen boundary**：但要在 synthetic data 上算 gradient（real data boundary noisy）。
7. **Resolution 是 boundary accuracy 的必要非充分条件**：resolution 翻倍 F1 翻 3 倍是上界，实际还需要 architecture + loss 配合。

---

## References

- Depth Pro GitHub: https://github.com/apple/ml-depth-pro
- Depth Pro arXiv: https://arxiv.org/abs/2410.02073
- DPT (Ranftl et al., 2021): https://arxiv.org/abs/2103.13413
- MiDaS (Ranftl et al., 2022): https://arxiv.org/abs/1907.01341
- Metric3D (Yin et al., 2023): https://arxiv.org/abs/2307.10984
- Metric3D v2 (Hu et al., 2024): https://arxiv.org/abs/2312.06505
- Marigold (Ke et al., 2024): https://arxiv.org/abs/2312.02145
- PatchFusion (Li et al., 2024a): https://arxiv.org/abs/2312.08179
- UniDepth (Piccinelli et al., 2024): https://arxiv.org/abs/2403.18913
- Depth Anything v2 (Yang et al., 2024b): https://arxiv.org/abs/2406.09414
- DINOv2 (Oquab et al., 2024): https://arxiv.org/abs/2304.07193
- ZeroDepth (Guizilini et al., 2023): https://arxiv.org/abs/2306.17253
- ZoeDepth (Bhat et al., 2023): https://arxiv.org/abs/2302.12284
- SPEC (Kocabas et al., 2021): https://arxiv.org/abs/2106.12073
- iBims (Koch et al., 2018): https://arxiv.org/abs/1812.04200
- AM-2k (Li et al., 2022a): https://arxiv.org/abs/2207.05031
- DIS-5k (Qin et al., 2022): https://arxiv.org/abs/2203.03041
- BokehMe (Peng et al., 2022a): https://arxiv.org/abs/2112.01607
- ControlNet (Zhang et al., 2023b): https://arxiv.org/abs/2302.05543
- BoostingDepth (Miangoleh et al., 2021): https://arxiv.org/abs/2108.08873
- SMD-Nets (Tosi et al., 2021): https://arxiv.org/abs/2104.03266
