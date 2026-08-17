---
source_pdf: Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object
  Removal.pdf
paper_sha256: 9c4a7bab9c5dda8794546fb5fc8cf92c48c34ff3da9b47d014d159cdcff208ef
processed_at: '2026-08-05T12:35:26-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 GScream

好的 Andrej，我换个人话版本，少绕弯子。

## 这篇 paper 到底在干嘛

你有一堆多视角照片拍了一个场景，里面有个碍眼的东西（比如一盏灯、一张桌子），你想把它从 3D 场景里"抠掉"，露出后面被挡住的部分。

这事 image in-painting 早就能干——2D 上填个洞嘛。但 3D 里难，**因为**你得保证从任何角度看，抠掉的那个洞都填得合理、连贯、没破绽。

以前大家用 NeRF 干这事，但 NeRF 太慢了，训一个场景要好几个小时。3DGS 快得多，理论上更合适。**但是**直接把 3DGS 套上去，会发现两个坑：

## 两个坑

**坑一：Gaussian 会"飘"**

3DGS 就是拿一堆小椭球去拟合一堆图。正常情况下每个视角都有真实照片盯着，椭球们乖乖待在该待的位置。**但是**你要 remove 的那个区域，只有一张 in-painted 的参考图，其他视角啥监督都没有。那些椭球就放飞自我了——飘到半空中、乱七八糟、render 出来全是洞。你看 Figure 5(a) 红框里那些飘着的 blob，就是没管住的结果。

**坑二：texture 接不上**

你只 in-paint 了一张参考图。从那个参考视角看还行，换个角度看，mask 边界处直接裂开——sharp boundary、black holes、texture 对不上。Figure 6(b) 红箭头指的那些黑洞就是这问题。

## GScream 的解法

就两招，简单粗暴但有效：

### 第一招：拿 depth 当拐杖

既然 RGB 监督不够，那就再加个 depth 监督。用 Marigold（一个 diffusion-based 的单目深度估计器）给每个视角估一张 depth map，然后逼 3DGS render 出来的 depth 跟它对齐。

有个小细节：Marigold 给的是 relative depth（相对深度），3DGS 给的是 metric depth（绝对深度），数值范围对不上。所以每次算 loss 之前，先做个 least-squares 拟合，把 scale 和 shift 调好，再比较。这个过程每个 iteration 都做，叫 "online alignment"。

效果很直接——那些飘着的椭球被 depth 拉回地面了。Figure 5(b) 里 blob 们老老实实待在草地和灌木丛的位置。

还有个 mask 加权的小设计：参考视角（in-painted 那张）全图都监督，**因为**整张图都是合理的；其他视角只监督 mask 外面，**因为** mask 内是 object 的 depth，不能用。

### 第二招：让旁边的 texture "传染"过去

这是最 clever 的部分。

2D in-painting 只管了一张图，其他视角的 mask 区域没人管。那怎么办？让 mask **外面**的 3D Gaussian features 主动"传"到 mask **里面**去。

具体做法：在每个视角上，sample 一个 patch（同时覆盖 mask 内外），把 patch 内的 3D anchors 分成两组——inside 和 surrounding。然后让这两组 features 互相做 cross-attention：

- inside 的 features 去 query surrounding 的 features → 把外面的纹理信息"拉"进来
- surrounding 的 features 也去 query inside 的 features → 让边界两边互相 align

这样做了之后，mask 区域的 features 不是凭空生成的，是从旁边"借"来的，自然就跟周围连贯。而且这是在 3D anchor feature 层面做的，render 到任何视角都 consistent。

**为什么**非要用 Scaffold-GS 而不是 vanilla 3DGS？**因为** vanilla 3DGS 有几百万个独立 Gaussian，每个都做 cross-attention 算不动。Scaffold-GS 把 Gaussians 组织在 anchor 周围，anchor 数量少得多，cross-attention 才算得动。这是工程上的必要选择。

## 效果怎么样

定量上：
- PSNR、SSIM、LPIPS、FID 全面持平或超过 SPIn-NeRF 和 OR-NeRF
- 训练时间 1.2 小时，比 SPIn-NeRF（3h）快 2.5 倍，比 OR-NeRF（6h）快 5 倍

定性上：
- depth supervision 把飘着的 blob 拉回来了
- cross-attention 把 mask 边界的裂缝和黑洞填上了

有个有意思的细节：SPIn-NeRF 和 OR-NeRF 训练时都用了 LPIPS loss，GScream **没用**，但 LPIPS metric 反而更好。说明 3D feature propagation 本身就能改善 perceptual quality，不需要显式 perceptual loss 来凑。

## 一句话总结

3DGS 做 object removal 的核心问题是 mask 区域没人监督。解法就两步：depth 监督把 geometry 撑住，cross-attention 把旁边的 texture 借过来。都在 3D 空间做，天然 multi-view consistent，比在 2D 上逐视角 in-painting 靠谱得多。

---

# GScream: 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal

好的 Andrej，这篇 paper 我仔细读完了，整体思路其实非常清晰。让我从 first principle 出发帮你 build intuition。

## 1. 问题的本质：为什么 object removal 在 3DGS 上 hard

先理清一下 task formulation。给定 N 张 posed multi-view images $\{I_i\}_{i=0}^{N}$，以及对应的 binary masks $\{M_i\}$ 标注要 remove 的 object。目标输出是一个 3D Gaussian representation，render 出来的新视角里 object 消失，并且 background 是几何 + texture consistent 的。

这件事在 NeRF 上已经有 SPIn-NeRF、OR-NeRF、View-Sub 等工作，但都 suffer from NeRF 的固有缺陷：训练慢、渲染慢。3DGS 本来应该是天然更合适的选择（explicit representation、real-time rendering、fast optimization），**但是**直接把 3DGS 套上去会暴露两个 3DGS 在 object removal 场景下的 specifically 痛点：

**痛点 1：Geometry Discreteness**
3DGS 用 millions of independent Gaussian primitives 拟合 RGB，standard loss 只有 photometric loss，对 underlying geometry 几乎没有强约束。这导致 Gaussian blobs 的 3D position 可以"漂浮"——只要渲染出来的 2D RGB 对就行。在 normal novel view synthesis 里这无所谓，**因为**所有视角都有 supervision。但是在 object removal 场景下，mask 区域内**没有** ground truth RGB（除了那一张 in-painted reference image），Gaussians 在 removal region 内的 3D 位置完全 unconstrained，会飘到空气中、出现 holes、geometry 完全 wrong。这就是 Figure 5(a) 红框里展示的现象。

**痛点 2：Texture Incoherence Across Views**
2D in-painting model（LaMa / Stable Diffusion）只给了一张 reference view 的 in-painted image。其他视角的 mask 区域要靠 3D representation 自己 generate。直接 fit 这一张 reference view 会导致：从 reference view 看 OK，从其他视角看 mask 边界处出现 sharp boundary、texture gap、black holes（Figure 6 Scene-1 (b) 红箭头）。

GScream 的核心 thesis：**不能只靠 2D prior 监督，要在 3D space 里 explicitly enforce geometry consistency 和 feature propagation**。

---

## 2. 方法架构总览

GScream 的 pipeline 用 Scaffold-GS 作为 base model（不是 vanilla 3DGS）。这个选择很关键，后面会讲为什么。

两个核心 module：
1. **Monocular Depth Guided Training** —— 解决 geometry
2. **Cross-Attention Feature Regularization** —— 解决 texture coherence

整个流程：
- 选 view 0 作为 reference view
- 用 2D in-painting（LaMa 或 SD）生成 $\bar{I}_0$（reference view 的 in-painted 版本）
- 用 Marigold 对所有 views 估 monocular depth $\mathcal{D} = \{D_i\}$
- 用 SfM points 初始化 Scaffold-GS anchors
- 训练时：color loss + depth loss + TV loss + cross-attention feature regularization
- 最后渲染监督 total loss

---

## 3. 为什么选 Scaffold-GS 而非 vanilla 3DGS

这点 paper 里讲得比较轻描淡写，但其实是工程上很重要的设计。vanilla 3DGS 每个 Gaussian blob 独立存 $\mu, S, R, c, \alpha$，densify 之后能到 millions 个，每个都要做 cross-attention 计算量爆炸。

Scaffold-GS 的核心 idea：scene 用一组 sparse **anchors** 组织，每个 anchor 带 learnable feature embedding，周围的 Gaussian attributes（$\mu$ offset, scale, rotation, color feature, opacity）都由 decoder 从 anchor feature 解码出来。densify 在 anchor level 而非 Gaussian level。

这意味着 GScream 的 cross-attention 只需要在 anchor feature 上做，anchor 数量远小于 Gaussian 数量，计算可行。这是整个 cross-attention 设计能成立的前提条件。

Scaffold-GS paper: https://arxiv.org/abs/2312.00109

---

## 4. Component 1: Monocular Depth Guided Training

### 4.1 核心公式

**Depth rendering**（Equation 5）：
$$\hat{D} = \sum_{k=1}^{K} t_k \alpha_k \prod_{j=1}^{k-1}(1-\alpha_j)$$

变量解释：
- $K$：ray 上的 sampling points 数量
- $t_k$：第 $k$ 个 Gaussian 的 mean $\mu_k$ 在 camera coordinate system 下的 z-coordinate（即沿 camera 光轴的深度值）
- $\alpha_k$：第 $k$ 个 Gaussian 的 alpha-blending weight，由 projected 2D Gaussian evaluated 在 pixel 处的值乘以 opacity 得到
- $\prod_{j=1}^{k-1}(1-\alpha_j)$：transmittance，前面的 Gaussians 挡住的比例
- 上标 $k-1$、下标 $j=1$：表示从 ray 入口到当前 point 之前所有 Gaussians 的累积遮挡

这跟 color rendering（Equation 2）形式完全一样，只是把 color $c_k$ 换成了 depth $t_k$。这是 3DGS 的 standard depth rendering，没什么 trick。

**Weighted depth loss**（Equation 3）：
$$\mathcal{L}_{\text{depth}} = \frac{1}{HW} \sum M_i' \| (w\hat{D}_i + q) - D_i \|$$

变量：
- $H, W$：image height, width，$HW$ 做 normalization
- $M_i'$：per-view 的 weighted mask，见 Equation 4
- $w, q$：scale 和 shift 参数，online alignment 用的，因为 monocular depth 是 relative depth 不是 metric depth
- $\hat{D}_i$：从 3DGS render 出来的 depth
- $D_i$：Marigold 估出来的 monocular depth

**Mask weighting**（Equation 4）：
$$M_i' = \begin{cases} \lambda_1 M_i + \lambda_2(1-M_i), & \text{if } i=0 \\ \lambda_3(1-M_i), & \text{if } i \neq 0 \end{cases}$$

这里很关键，需要仔细解读：
- $i=0$ 是 reference view（已经 in-painted 过的 $\bar{I}_0$）
  - mask 内（$M_i=1$，即原本是 object 的区域，现在被 in-painted 填上了）：weight $\lambda_1$
  - mask 外（$1-M_i$，background，原本就在）：weight $\lambda_2$
  - **所以 reference view 全图都被监督**，包括 in-painted 区域，因为 $\bar{I}_0$ 是合理的 ground truth
- $i \neq 0$ 是其他 views
  - 只监督 mask 外（$\lambda_3(1-M_i)$），因为 mask 内是 object，depth 是 object 的 depth，**不能**用来监督 removal 后的 background

$\lambda_1, \lambda_2, \lambda_3$ 是平衡这些 region 监督强度的超参数。

### 4.2 Online Scale-Shift Alignment

Marigold 给的是 relative depth（值域 normalized 到 [0,1] 之类），3DGS render 出来的是 metric depth（取决于 SfM 的 scale）。两者数值 scale 完全不匹配。

GScream 用 least-squares 求解 $w, q$，把 rendered depth $\hat{D}$ 对齐到 monocular depth $D$：
$$\min_{w,q} \sum (w\hat{D} + q - D)^2$$

这跟 MonoSDF 的做法一样（reference [43], https://arxiv.org/abs/2206.00665）。注意 $w, q$ 是 per-image 的，每个 view 单独算，**因为** Marigold 在不同 view 上的 scale 也不一定一致。

paper 没明确说 alignment 是每个 iteration 都做还是每隔几步做一次。从 "online" 这个词推测应该是 training 过程中持续更新，类似 MonoSDF 的做法——每个 iteration 重新 least-squares fit 一次 $w, q$，然后算 loss，再 backprop 到 3DGS。

### 4.3 Total Variation Loss（Equation 6）

$$\mathcal{L}_{\text{tv}} = \frac{1}{N} \sum M_i' \| \nabla((w\hat{D}_i + q) - D_i) \|$$

$\nabla$ 是 spatial gradient（image 横纵方向的差分）。这个 loss 强制 aligned rendered depth 和 monocular depth 的**差分图**平滑，避免 depth supervision 在空间上跳变。在 mask 边界处尤其重要，**因为** mask 内外 depth 来自不同 source（mask 内是 in-painted reference 的 depth，mask 外是真实 Marigold depth），容易不连续。

### 4.4 Color Loss（Equation 7）

$$\mathcal{L}_{\text{color}} = \frac{1}{HW} \sum M_i' \left( (1-\lambda_{\text{ssim}})\|\hat{C}_i - I_i\| + \lambda_{\text{ssim}} \text{SSIM}(\hat{C}_i, I_i) \right)$$

standard L1 + SSIM loss，乘以 $M_i'$ 做加权。$\lambda_{\text{ssim}}$ 平衡 L1 和 SSIM。

注意这里 reference view 用的是 $\bar{I}_0$（in-painted 后的），其他 views 用原 $I_i$。

### 4.5 Total Loss（Equation 8）

$$\mathcal{L}_{\text{total}} = \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{tv}} \mathcal{L}_{\text{tv}} + \mathcal{L}_{\text{color}}$$

$\lambda_{\text{depth}}, \lambda_{\text{tv}}$ 是 loss weight 超参数，color loss 权重默认为 1。

### 4.6 Intuition

整个 depth guidance 的核心 intuition：3DGS 本身的 photometric loss 对 geometry 是 underconstrained 的，**因为** Gaussian 可以在 depth 方向上自由滑动而不影响 2D RGB rendering（只要 density 够）。加 monocular depth supervision 相当于在 geometry 维度加了一个 anchor，把 floating Gaussians 拉回到合理的 depth 上。在 object removal 场景下这尤其 critical，**因为** mask 区域内 RGB supervision 几乎为零（只有一张 reference），完全靠 depth 把 geometry 撑起来。

---

## 5. Component 2: Cross-Attention Feature Regularization

这是 paper 最 novel 的部分。核心 idea：**让 visible region 的 anchor features 主动 propagate 到 in-painted region**，避免只靠 2D prior 导致的 view-inconsistency。

### 5.1 3D Gaussian Sampling

对每个 view $i$：
1. 在 2D image 上 sample 一个 patch，这个 patch **同时覆盖** mask 内和外
2. 把所有 3D anchor 的 center 投影到这个 view
3. 找出投影落在 patch 内的 anchors
4. 根据投影是否落在 2D mask 内，分成两组：
   - $f_{\text{in}}$：投影在 mask 内的 anchors 的 features（in-painted region）
   - $f_{\text{sur}}$：投影在 mask 外的 anchors 的 features（surrounding / visible region）

这里用的是 2D mask back-projection 来分组，**而不是** depth-based back-projection。paper 里说 "we believe that our approach based on 2D mask back-projection is sufficient"。这个选择其实有 limitation：2D mask 在一个 view 内的 back-projection 对应 3D 空间的一个 frustum，可能包含一些**实际不属于** "object 后方" 的 anchors。但是因为 Scaffold-GS anchor 比较稀疏，加上 cross-attention 本身是 soft 信息传播，这个问题在实践中影响不大。

### 5.2 Bidirectional Cross-Attention（Equation 9）

$$\hat{f}_{\text{in}} = \text{Attention}(Q=f_{\text{in}}, K=f_{\text{sur}}, V=f_{\text{sur}})$$
$$\hat{f}_{\text{sur}} = \text{Attention}(Q=f_{\text{sur}}, K=f_{\text{in}}, V=f_{\text{in}})$$

Standard scaled dot-product attention：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

变量：
- $Q, K, V$：query, key, value matrices
- $d_k$：token length（feature dimension），$\sqrt{d_k}$ 是 standard scaling 防止 dot product 过大
- $K^T$：$K$ 的转置，$(Q K^T)_{ij}$ 是 query $i$ 和 key $j$ 的相似度

**Bidirectional** 的设计很巧妙：
- $\hat{f}_{\text{in}}$ 的更新：in-painted region 的 anchor features 去 query surrounding region 的 features，把 surrounding 的纹理信息"拉"进来
- $\hat{f}_{\text{sur}}$ 的更新：surrounding region 的 anchor features 也去 query in-painted region 的 features

paper 说这两组 attention module **共享参数**（"two sets of shared-parameter cross-attention modules"），但没说清楚是同一个 module 跑两次还是两个 module tied weights。从描述看更像前者——同一个 attention module，跑两次，分别以 in→sur 和 sur→in 的方向。

### 5.3 为什么 Bidirectional 而非 Unidirectional

单向（只 $\hat{f}_{\text{in}}$）会让 in-painted region 的 features 被强行对齐到 surrounding，但 surrounding region 的 features 没有被 push 去 "接受" in-painted region 的信息，可能造成 boundary 处依然 sharp。

双向的话，**两边** anchor features 都会被 update，gradient 也 flow 到 surrounding region 的 anchors 上，让它们也"知道" in-painted region 的存在。这样在 mask boundary 处，两边的 features 会互相 align，texture transition 更 smooth。

### 5.4 后续流程

更新后的 $\hat{f}_{\text{in}}, \hat{f}_{\text{sur}}$ 赋回对应的 anchors，然后这些 anchors 走正常的 Scaffold-GS pipeline：neural blob growing（从 anchor feature decode 出 Gaussian attributes）→ differentiable rendering → 监督 $\mathcal{L}_{\text{total}}$。

gradient 通过 cross-attention 流回到 anchors 的 features 上，包括 visible region 的 anchors。这是关键——visible region 的 anchors **也**会被 update，让它们的 features 与 in-painted region 的 features 更 consistent。

### 5.5 Intuition

把 cross-attention 想象成一种 3D 空间内的 "feature smoothing" operator。2D in-painting 只解决了 reference view 的 mask 区域 texture，3D space 里其他视角的 mask 区域**没有**直接 supervision。Cross-attention 通过让 visible anchors 和 in-painted anchors 互相 attend，相当于在 3D feature space 里做了一次 interpolation / propagation，把 visible region 的 texture 信息扩散到 in-painted region，并且这种扩散是 view-consistent 的（**因为**在 3D anchor 上做，render 到任何视角都一致）。

这跟 image domain 的 2D in-painting 有本质区别：2D in-painting 是 per-pixel 的，每个 view 独立 in-paint 会有 multi-view inconsistency；3D feature propagation 是在 representation 层面做的，一旦 features consistent，所有视角 render 出来都 consistent。

---

## 6. 实验数据深度解读

### 6.1 主实验（Table 1）

| Method | PSNR | masked-PSNR | SSIM | masked-SSIM | LPIPS | masked-LPIPS | FID | Train Time |
|---|---|---|---|---|---|---|---|---|
| SPIn-NeRF | 20.18 | 15.80 | 0.46 | 0.21 | 0.47 | 0.58 | 58.78 | ~3.0h |
| OR-NeRF | 20.32 | 15.74 | 0.54 | 0.21 | 0.35 | 0.56 | 38.69 | ~6.0h |
| View-Sub | - | - | - | - | - | 0.45* | - | - |
| **GScream** | **20.49** | **15.84** | **0.58** | **0.21** | **0.28** | 0.54 | **36.72** | **~1.2h** |

几个 observation：
- PSNR 全图只提升 0.17-0.31 dB，但 masked-PSNR 提升更明显（vs SPIn-NeRF +0.04, vs OR-NeRF +0.10），说明提升主要在 mask 区域
- SSIM 提升最大（0.46→0.58 vs SPIn-NeRF），structural consistency 改善显著
- LPIPS 全图从 0.47 降到 0.28，**这是最大的相对提升**，说明 perceptual quality 改善明显
- FID 36.72 最低，feature distribution 最接近真实图像
- Training time ~1.2h，相比 SPIn-NeRF (3h) 快 ~2.5x，相比 OR-NeRF (6h) 快 ~5x

注意 paper 提到 SPIn-NeRF 和 OR-NeRF 都用了 patch-based LPIPS loss，GScream **没有**用 LPIPS loss 做 training，但 LPIPS metric 仍然更好。这是很强的证据说明 3D feature propagation 本身就改善了 perceptual quality，不需要显式 perceptual loss。

masked-SSIM 三者都是 0.21，没提升。可能 SSIM 在 mask 区域内对 structural pattern 敏感，in-painted 区域的 structure 跟 ground truth 本来就有差异。

### 6.2 Ablation（Table 2）

| Variant | PSNR | masked-PSNR | SSIM | masked-SSIM | LPIPS | masked-LPIPS |
|---|---|---|---|---|---|---|
| w/o Cross-Attn & Mono-Depth | 20.12 | 14.87 | 0.58 | 0.19 | 0.26 | 0.56 |
| w/o Cross-Attn | 20.47 | 15.63 | 0.58 | 0.20 | 0.26 | 0.50 |
| Full GScream | 20.49 | 15.84 | 0.58 | 0.21 | 0.28 | 0.54 |

关键观察：
- **Mono-Depth 的贡献**：从 "w/o Cross-Attn & Mono-Depth" 到 "w/o Cross-Attn"（即加上 depth），masked-PSNR 从 14.87 跳到 15.63（+0.76），masked-SSIM 从 0.19 到 0.20。**这是最大的一笔提升**，说明 depth supervision 是 dominant factor。
- **Cross-Attention 的贡献**：从 "w/o Cross-Attn" 到 "Full"，masked-PSNR 从 15.63 到 15.84（+0.21），masked-SSIM 从 0.20 到 0.21。提升相对小，但定性上（Figure 6）能看到 cross-attention 消除了 boundary 和 holes。
- **奇怪的点**：加 cross-attention 后全图 LPIPS 从 0.26 升到 0.28（变差了），masked-LPIPS 从 0.50 升到 0.54（也变差了）。paper 承认了这点 "a marginal reduction in the LPIPS metric"，但定性上 cross-attention 明显更好。这说明 LPIPS 在这里**不是**一个好的 metric——LPIPS 基于 VGG features，可能更偏好 sharp texture，而 cross-attention 做的是 feature smoothing，让 boundary 更 soft，反而让 LPIPS 变差。但人眼看 cross-attention 的结果明显更 natural。

### 6.3 Depth Estimator Ablation（Figure 8）

比较 Midas vs Marigold 作为 depth prior source：
- Midas 在 red fence 处 depth 不连续 → GScream 学出的 texture 也不连续
- Marigold depth 更连续 → GScream texture 更连续

这说明 depth guidance 的 quality 直接决定 final result 的 quality，整个 pipeline 对 depth estimator 很 sensitive。Marigold 基于 diffusion model，比 Midas 的 transformer-based 估计更 robust。

Marigold paper: https://arxiv.org/abs/2312.04561

### 6.4 2D In-painting Model Ablation（Figure 9）

LaMa vs Stable Diffusion 作为 reference view in-painting：
- LaMa 把 sink 和 indentation 都 remove 了
- SD 只 remove sink，保留了 indentation
- 两者作为 reference 都能产生 reasonable GScream 结果

paper 的结论：reference in-painting 的具体方法**不重要**，只要 reference 合理，GScream 能 generate 3D consistent 结果。这其实是个 strong claim，说明 GScream 对 2D prior 的依赖是 robust 的。

---

## 7. 与 GaussianEditor 的对比（Figure 7）

GaussianEditor 是 general 3DGS editing framework，也 support object removal，但用的是 2D diffusion prior 直接 guide Hierarchical Gaussian Splatting (HGS) 更新。

GScream 在 object removal 这个 specific task 上明显更好，**因为** GaussianEditor 没有 explicit 的 geometry completion 和 3D feature propagation 机制，纯靠 2D prior fit 出来的结果在 mask 区域容易 unrealistic。

GaussianEditor paper: https://arxiv.org/abs/2311.14521

---

## 8. 一些联想和潜在 limitation

### 8.1 为什么不直接用 metric depth

Marigold 给的是 relative depth，需要 online alignment。如果用 metric depth estimator（比如 Zoedepth、Metric3D），理论上可以省掉 $w, q$ 的 alignment 步骤。但 metric depth estimator 在 in-the-wild 场景上 accuracy 不如 relative depth estimator 稳定，所以这个 trade-off 是合理的。

Metric3D paper: https://arxiv.org/abs/2307.10984

### 8.2 Cross-Attention 的计算开销

paper 没明确报 cross-attention 增加的训练时间。从 "训练 1.2h" 包含 cross-attention 来看，overhead 应该可控。但理论上：
- 每个 view 都要 sample patch + project anchors + cross-attention
- Anchor 数量取决于 Scaffold-GS densify 程度
- Patch 数量、大小影响 token 数量

如果场景复杂、anchor 多、patch 大，cross-attention 的 $O(n^2 d)$ 复杂度会成为 bottleneck。可以用 linear attention 或 FlashAttention 优化。

### 8.3 对 large removed object 的处理

paper 的实验都是中小型 object removal（lamp, table, fence section, sink）。如果 object 很大（比如整面墙、整栋楼），in-painted region 占 image 很大比例，surrounding region 能提供的信息有限，cross-attention 能 propagate 的内容也有限。这种 extreme case 可能需要 generative prior（diffusion model）更深度参与，而不只是做 reference view in-painting。

### 8.4 跟 3D-aware diffusion inpainting 的关系

最近有一类工作（比如 InpaintNeRF360, https://arxiv.org/abs/2305.15094）用 text-guided diffusion 直接 generate multi-view consistent in-painted images。GScream 走的是另一条路：2D in-painting 只做一次 reference，剩下的靠 3D representation 自己 + depth + cross-attention 搞定。两条路线的 trade-off：
- GScream 路线：快，对 2D prior 依赖低，但 large removal 时生成能力受限
- Diffusion-guided 路线：生成能力强，但慢，且 multi-view consistency 仍是 challenge

### 8.5 跟 SuGaR / 2DGS 的关系

GScream 用 Scaffold-GS 作为 base。但其实 geometry consistency 的问题在 vanilla 3DGS 上更严重。如果用 2DGS（https://arxiv.org/abs/2401.10291）这种 geometry-aware 的 variant，可能 depth supervision 的需求会降低。2DGS 用 surface normal 和 depth distortion loss，本身 geometry 就更准。这是一个未探索的 direction。

### 8.6 Depth supervision 跟 MonoSDF / VolSDF 的渊源

GScream 的 online scale-shift alignment 直接借鉴自 MonoSDF（reference [43]）。MonoSDF 在 NeRF/SDF 上用 monocular normal + depth 做 supervision，证明这些 cheap 2D priors 能显著提升 3D reconstruction geometry。GScream 把这个 idea 迁移到 3DGS object removal 上，思路一致。

MonoSDF: https://arxiv.org/abs/2206.00665

### 8.7 Cross-Attention 跟 Feature 3DGS 的关系

最近有一些工作把 neural features attach 到 Gaussians 上做 downstream task（feature 3DGS）。GScream 的 cross-attention 本质上是在 anchor feature space 做 propagation，可以看作 feature 3DGS 的一种应用。如果结合 LERF / Feature 3DGS 的 CLIP features，可能能做 semantics-aware object removal（比如 "remove the chair but keep the cushion pattern on floor")。

LERF: https://lerf.io/

---

## 9. 总结：GScream 的核心 contribution

1. **第一个**把 3DGS 专门用于 object removal 的工作（GaussianEditor 是 general editing，不是 specific for removal）
2. **Monocular depth guidance + online alignment**：解决 3DGS geometry underconstrained 的老问题，在 removal 场景下尤其 critical
3. **Bidirectional cross-attention on anchor features**：利用 Scaffold-GS 的 anchor 结构，在 3D feature space 做 visible→in-painted 的信息传播，**不依赖** per-view 2D in-painting，从根本上保证 multi-view consistency
4. **训练效率**：1.2h vs NeRF-based 方法的 3-6h

核心 intuition 一句话总结：3DGS object removal 的本质问题是 mask 区域内 supervision 不足，solution 是 (a) 用 monocular depth 把 geometry 维度的 supervision 补上，(b) 用 cross-attention 把 visible region 的 texture feature 主动 propagate 到 mask region，两者都在 3D space 做，inherently multi-view consistent。

---

## Reference Links

- GScream Project Page: https://w-ted.github.io/publications/gscream
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Scaffold-GS: https://arxiv.org/abs/2312.00109
- SPIn-NeRF: https://arxiv.org/abs/2305.04312
- OR-NeRF: https://arxiv.org/abs/2305.10503
- Marigold: https://arxiv.org/abs/2312.04561
- MonoSDF: https://arxiv.org/abs/2206.00665
- GaussianEditor: https://arxiv.org/abs/2311.14521
- LaMa: https://arxiv.org/abs/2109.07092
- 2DGS: https://arxiv.org/abs/2401.10291
- Metric3D: https://arxiv.org/abs/2307.10984
- InpaintNeRF360: https://arxiv.org/abs/2305.15094
- LERF: https://lerf.io/

希望这个 walk-through 帮你 build 起对 GScream 的 intuition，Andrej。如果哪个 component 你想再深挖（比如 cross-attention 的具体实现细节、Scaffold-GS decoder 结构、或者跟其他 object removal 方法的更细对比），告诉我，我可以展开讲。
