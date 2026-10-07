---
source_pdf: GeoQuery Geometry-Query Diffusion for Sparse-View Reconstruction.pdf
paper_sha256: a3ac0f9bfbdf53158eb6fd4da0bd3e3d293a637ae0e4aa6d9505ca113eb28c73
processed_at: '2026-08-04T21:34:39-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 GeoQuery

## 一句话版

这 paper 发现一个 bug: **用烂图当 query 去 clean 图里找信息, 你只会找到更多烂的东西**。修法很简单: 用几何信息算出"应该去哪儿找", 别让烂图自己决定。

---

## 这个 bug 到底是什么

假设你有这么个 pipeline: 用 3 个 view 训了个 3DGS, 渲出来的图全是 floaters、结构崩坏。你拿 [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) 当 refiner 想修它, 同时把一个 clean training view 喂进去当 reference。

之前的做法 ([DIFIX3D+](https://github.com/WuJianzhe/DIFIX3D-Plus), [ReconFusion](https://reconfusion.github.io/)) 是这么干的: 把 target 和 reference 的 feature 拼起来, 跑个 self-attention:

$$F^{out} = \text{Softmax}\left(\frac{Q^t [K^t; K^r]^\top}{\sqrt{d}}\right) [V^t; V^r]$$

翻译成人话: target 图说"我这一块需要 reference 里某些信息", attention 帮它从 reference 里挑出来。

**问题**: target 图是烂的。floaters 让 target 的 query feature $Q^t = W_Q F^t$ 变成了一个**语义模糊、乱七八糟的向量**。这个向量在 reference 的 feature space 里找最近邻, 找到的几乎都是错的东西——可能仅仅因为 reference 某块区域的 low-level texture 跟 artifact 有 surface 相似, attention 就把它匹配过来了。

然后这个错的内容被写回 target, target 变得更烂, 下一次 refine 更烂。这就是个 **vicious cycle** (论文原话)。

---

## 为什么这个 bug 之前没人发现

我猜是因为这个 bug 表现得很隐蔽。你看 [Table 3](https://arxiv.org/) 那个 region-level 分析:

| Region | 3DGS | DIFIX3D+ | GeoQuery |
|---|---|---|---|
| Low-error ($e \leq 30$) | 25.82 | **25.07 (-0.75)** | 26.19 (+0.37) |
| High-error ($e > 30$) | 11.16 | 13.16 (+2.00) | 15.19 (+4.03) |

DIFIX3D+ 在烂区域 +2.00 dB, 看起来"修好了一部分"。但在**原本好的区域 -0.75 dB**, 整体 PSNR 还是被拉低了。你只看平均 PSNR, 这个现象就藏起来了。

为啥好区域也会被搞坏? attention 是全局操作, softmax 共享 normalization。烂区域的 corrupted query 在 reference 里乱抢, 会**偷走**本该分给好区域的 attention 权重。好区域的 query 本来想 retrieve reference 的正确位置, 结果发现 attention mass 被烂 query 抢走了, 只能凑合 retrieve 错的。

这跟 transformer 里"一个 bad token 污染整个 sequence"是同一个道理。

---

## 怎么修: 把 query 换成干净的东西

论文的核心 trick 在 [Eq. 6](https://arxiv.org/):

$$F^{r \rightarrow t}(\mathbf{u}_t) = M_{t \leftarrow r}(\mathbf{u}_t) \odot \text{Sample}(F^r, C_{t \leftarrow r}(\mathbf{u}_t))$$

变量解释:
- $\mathbf{u}_t \in \mathbb{R}^2$: target 图里某个像素的坐标
- $C_{t \leftarrow r}(\mathbf{u}_t) \in \mathbb{R}^2$: 用几何算出来的"target 像素 $\mathbf{u}_t$ 在 reference 里对应哪个像素"
- $F^r \in \mathbb{R}^{H/l \times W/l \times d}$: reference 在 UNet 某 layer 的 feature
- $\text{Sample}(\cdot, \cdot)$: bilinear interpolation, 按坐标去 $F^r$ 里采
- $M_{t \leftarrow r} \in \{0,1\}$: 这个 target 像素在 reference 里能不能看到 (occlusion / 出 FOV)
- $\odot$: mask 掉看不见的地方

**人话**: 不让 target 自己说"我要找什么", 让几何提前算好"target 的像素 $\mathbf{u}_t$ 对应 reference 的像素 $C_{t \leftarrow r}(\mathbf{u}_t)$", 然后把 reference 在那个位置的 feature 直接 warp 过来当 query。

这个 warp 过来的 feature 叫 **proxy feature $F^{r \rightarrow t}$**, 它在结构上跟 target 对齐, 内容上却是 clean reference 的。

---

## $C_{t \leftarrow r}$ 怎么算的

[Eq. 5](https://arxiv.org/) 给的两步:

**Step 1**: reference 像素 $\mathbf{u}_r$ 加上 metric depth $D^r(\mathbf{u}_r)$ back-project 到 3D:
$$\mathbf{x} = \pi^{-1}(\mathbf{u}_r, D^r(\mathbf{u}_r), \mathbf{K}_r)$$

实质就是 $\mathbf{x} = D^r(\mathbf{u}_r) \cdot \mathbf{K}_r^{-1} [\mathbf{u}_r; 1]$, 把 2D 像素 + depth 还原成 3D 点。

**Step 2**: 把这个 3D 点 project 到 target view:
$$\mathbf{u}_t = \pi(\mathbf{K}_t \mathbf{T}_t \mathbf{T}_r^{-1} \mathbf{x})$$

变量:
- $\mathbf{T}_t, \mathbf{T}_r \in \mathbb{R}^{4 \times 4}$: target / reference 的 world-to-camera pose
- $\mathbf{T}_r^{-1}$: reference camera-to-world, $\mathbf{T}_t \mathbf{T}_r^{-1}$: reference-to-target relative pose
- $\mathbf{K}_t \in \mathbb{R}^{3 \times 3}$: target 内参
- $\pi(\cdot)$: 透视除法, $[x'; y'; z'] \to (x'/z', y'/z')$

**Step 3**: 把整个 reference coordinate map **forward splat** 到 target view (用 [Softmax Splatting](https://arxiv.org/abs/2004.03865) 处理 collision), 得到稠密的 $C_{t \leftarrow r} \in \mathbb{R}^{H \times W \times 2}$ 和 visibility mask $M_{t \leftarrow r}$。

为啥用 forward splatting 而不是 backward warping? **因为 target 的 depth 是烂的** (artifact 污染), backward warping 要先估 target depth。Forward splatting 让 geometry 严格从 clean reference 流向 target, 单向的, 干净的。

---

## 还加了个 local window

光有 proxy query 还不够, 论文还加了 [Eq. 7-8](https://arxiv.org/):

$$K_\Delta(\mathbf{u}_t) = \text{Sample}(W_K F^r, C_{t \leftarrow r}(\mathbf{u}_t) + \Delta)$$
$$V_\Delta(\mathbf{u}_t) = \text{Sample}(W_V F^r, C_{t \leftarrow r}(\mathbf{u}_t) + \Delta)$$
$$F_{geo}^t(\mathbf{u}_t) = \sum_{\Delta \in \Omega} \text{Softmax}_\Delta\left(\frac{\langle Q(\mathbf{u}_t), K_\Delta(\mathbf{u}_t)\rangle}{\sqrt{d}}\right) V_\Delta(\mathbf{u}_t)$$

变量:
- $\Delta \in \Omega$: 以 $C_{t \leftarrow r}(\mathbf{u}_t)$ 为中心的 $k \times k$ 窗口里的偏移
- $Q(\mathbf{u}_t) = W_Q F^{r \rightarrow t}(\mathbf{u}_t)$: proxy query
- $W_K, W_V \in \mathbb{R}^{d \times d}$: 标准 K/V 投影矩阵
- $\sqrt{d}$: scaled attention 的常规缩放

人话: 不在整张 reference 图里找匹配, 只在几何对应点周围 $k \times k$ 小窗口里找。

三个理由:
1. **Depth 估计有误差**, $C_{t \leftarrow r}$ 给的点可能偏几个像素。Local window 让 attention 在附近搜一下, 容忍这个误差。
2. **Disocclusion 处**, forward splatting 处理不了 sub-pixel 精度, local window 给 spatial robustness。
3. **防止 spurious match**。即使 query 是 clean proxy, 全图 attention 仍可能 retrieve 到语义相似但位置不对的区域 (一张图里有 5 个窗户, 你只想要对应那个)。

[Fig. 8](https://arxiv.org/) ablation: $k=3$ 最优, $k=1$ 太死, $k \geq 5$ 又开始退化。典型 bias-variance tradeoff:
- $k$ 小: 几何 prior 太硬, 容不下 depth noise
- $k$ 大: 退化回 global attention, 几何约束失效

复杂度从 $O(N^2)$ 降到 $O(Nk^2)$, 这是能在 21GB A100 上跑通的实际原因。1024×576 feature map, $k=3$ 时计算量降了 $\sim 57000\times$。

---

## Adaptive Gate: 该用几何时就用, 不该用就 fallback

[Eq. 9](https://arxiv.org/):

$$w = \sigma(\text{MLP}([F^t, F_{geo}^t]))$$
$$F^t(\mathbf{u}_t) \gets (1 - w(\mathbf{u}_t)) \odot F^t(\mathbf{u}_t) + w(\mathbf{u}_t) \odot F_{geo}^t(\mathbf{u}_t)$$

变量:
- $[\cdot, \cdot]$: channel concat, $2d$ 维
- MLP: 2 层, 输出每 spatial location 一个标量
- $\sigma$: sigmoid, $w \in (0,1)^{H/l \times W/l}$
- $\gets$: in-place 更新

这个 gate 是 graceful degradation 用的:
- $M_{t \leftarrow r}(\mathbf{u}_t) = 0$ (occlusion / 出 FOV / depth 失败): $F^{r \rightarrow t} = 0$, $F_{geo}^t$ 输出无意义, MLP 学到 $w \to 0$, fall back 到原 $F^t$
- Correspondence 强: $w \to 1$, 充分用 GCA

人话: **几何准的地方听几何的, 几何失败的地方听 diffusion 的**, 让模型自己决定权重。

---

## 训练 loss

[Eq. 10-11](https://arxiv.org/):

$$\mathcal{L}_{gram} = \frac{1}{L}\sum_{l=1}^{L} \beta_l \|G_l(\hat{I}) - G_l(I)\|_2$$
$$G_l(I) = \phi_l(I)^\top \phi_l(I)$$
$$\mathcal{L} = \lambda_{recon}\mathcal{L}_{recon} + \lambda_{lpips}\mathcal{L}_{lpips} + \lambda_{gram}\mathcal{L}_{gram}$$

变量:
- $\phi_l(\cdot)$: VGG-16 第 $l$ 层 feature, shape $C_l \times H_l \times W_l$
- $G_l(I) \in \mathbb{R}^{C_l \times C_l}$: Gram matrix, 把 $\phi_l(I)$ flatten 成 $C_l \times (H_l W_l)$ 再自乘, 抓 channel correlation (= style)
- $\beta_l$: 浅层小深层大的权重
- $\lambda_{recon}, \lambda_{lpips}, \lambda_{gram}$: 三个 loss 的权重

三个 loss 互补: pixel loss 给 anchor, LPIPS 给 semantic structure, Gram 给 sharp texture。这个组合来自 [FILM](https://arxiv.org/abs/2207.08014) 的 video frame interpolation, 借过来用在 diffusion refinement 上。

---

## 实验数据怎么读

### Table 1: 单纯的 artifact removal

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|---|---|---|---|---|
| DIFIX3D+ (w/o ref) | 18.26 | 0.493 | 0.388 | 21.04 |
| DIFIX3D+ | 18.79 | 0.529 | 0.348 | 12.83 |
| GeoQuery | **19.88** | **0.566** | **0.314** | **10.20** |

对比解读:
- w/o ref → w/ ref: +0.53 PSNR, -8.21 FID。reference 本身有用
- DIFIX3D+ → GeoQuery: +1.09 PSNR, -2.63 FID。**几何引导比"加 reference"本身收益大**

FID 减半 (21→10) 说明 distribution-level realism 大涨, 跟"消除 hallucination"的论断对得上。

### Table 2: 不同 view 数下

| | 3-view | 6-view | 9-view |
|---|---|---|---|
| DIFIX3D (Mip-NeRF360) | 14.15 | 16.14 | 17.54 |
| GeoQuery (Mip-NeRF360) | **15.07 (+0.92)** | **16.93 (+0.79)** | **18.22 (+0.68)** |

3-view regime 增益最大 (+0.92 dB), 这是方法 designed for 的 hard case。View 越多 3DGS 越干净, query contamination 越弱, GeoQuery 边际收益越小。完全符合预期。

### Table 4: 核心消融

| SA | GCA/R | GCA/P | AF | PSNR | FID |
|---|---|---|---|---|---|
| ✓ | | | | 18.79 | 12.83 |
| ✓ | ✓ | | ✓ | 19.42 | 11.60 |
| ✓ | | ✓ | | 19.57 | 11.11 |
| ✓ | | ✓ | ✓ | **19.88** | **10.20** |

(GCA/R: query 用 corrupted rendering; GCA/P: query 用 proxy; AF: adaptive fusion)

逐行读:
- Row1→Row2: 加 GCA + local window, query 还是用烂 rendering 的, +0.63 PSNR。**光 local window 约束就有点用, 但不够**
- Row1→Row3: 把 query 换成 proxy, 没 AF, +0.78 PSNR。**Proxy query 贡献最大**
- Row3→Row4: 再加 AF, +0.31 PSNR。AF 是稳定补强

**Row2→Row4 是最 sharp 的对比**: 两种 setup 都有 AF + local window, 唯一区别是 query source。rendering query → proxy query, +0.46 PSNR。这就是 paper 的核心 claim: **query 的来源比 attention 范围更重要**。

---

## 跟 epipolar attention 的对比

[Epipolar attention](https://arxiv.org/abs/2312.12337) (pixelSplat, MVSplat, [DepthSplat](https://arxiv.org/abs/2411.09525)) 也是把 cross-view attention 用几何约束, 约束在 epipolar line 上。

[Table 6](https://arxiv.org/) 的对比:

| Method | PSNR | FID |
|---|---|---|
| Epipolar Attn only | 19.21 | 12.16 |
| Proxy + Epipolar Attn | 19.73 | 11.05 |
| Proxy + GCA | **19.88** | **10.20** |

关键差异: epipolar attention 还是拿 target feature 当 query, 只是把 key/value 限制在 epipolar line 上 → **依然受 query contamination**。

加 proxy query 比加 epipolar 约束增益大得多 (+0.42 → +1.0 PSNR)。**再次证明 query source 是关键**。

---

## 一些设计选择的 deeper thinking

### 为啥不直接 warp reference RGB 到 target?

那就是 classical IBR (image-based rendering, e.g. [IBRNet](https://arxiv.org/abs/2102.13090))。问题:
- Depth 误差导致 ghosting
- Occlusion boundary 撕裂
- 没有 generative completion 能力

GCA 在 feature 空间做 soft lookup (softmax 权重处理 depth 不确定性), 然后走 diffusion decoder 保留生成能力。本质上是 **"geometric anchor + generative synthesis"** 的 hybrid。

### Proxy feature 当 query 本质上是在 reference 上做 self-attention

[Eq. 8](https://arxiv.org/): query $W_Q F^{r \rightarrow t}$ 来自 reference, key $W_K F^r$ / value $W_V F^r$ 也来自 reference。**这是 reference feature 上 self-attention, 但 spatial index 由 target 决定**。

语义上等价于: "在 reference 的 local neighborhood 内, 找到跟 proxy 最匹配的 token"。这其实在**做 sub-pixel accurate correspondence refinement**——depth 给的对应点是 approximate 的, attention 在小窗口里能微调到更准的位置。

这跟 [Deformable DETR](https://arxiv.org/abs/2010.04159) 的 deformable attention 思路同源: 给个 reference point, 周围采样 offset。GCA 是 deformable attention 的 special case, reference point = geometric correspondence, offset 限制在 $k \times k$ grid。

---

## 局限性 (论文自己承认的)

[Section 6](https://arxiv.org/):

1. **Textureless region**: MVS depth 失败 → correspondence 不准 → GCA 退化
2. **Specular surface**: depth 估计和几何 prior 都崩
3. **Extreme viewpoint**: reference 完全看不到的区域, $M_{t \leftarrow r} = 0$, GCA fall back 到 global

实际意思是: **GeoQuery 的收益与 depth 质量强正相关**。换上更强的 depth estimator ([Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3) 已经是当下 SOTA, 未来还会有更好的), 性能会继续涨。

---

## 一个抽象的 take-away

把 attention 看成 content-addressable memory:
- **Query** = address
- **Key** = 索引 label
- **Value** = 实际内容

任何 attention-based retrieval 都假设 **address 是 trustworthy 的**。如果 address source 不可靠 (input corrupted / noisy / out-of-distribution), 整个 retrieval 系统进入 garbage-in-garbage-out 的 positive feedback loop。

GeoQuery 给的解决方案是: **当 default address source 不可靠时, 用一个 deterministic 的 modality 替代它**。这里有 metric depth + camera pose 提供 deterministic address, reference feature 做 content, diffusion decoder 做生成。

这个 design pattern 其实挺通用的。RAG 系统里 query embedding 不可靠时, 用 metadata (ID, timestamp, geo) 替代 retrieval。Autonomous driving 里 sensor noise 大时, 用 HD map 的 geometric prior 锚定 detection。Medical imaging 里用 anatomical atlas 锚定 segmentation。

**"Address source 必须比 content source 更可靠"** 这个原则, 比 GeoQuery 本身更值得记住。

---

## Reference 链接汇总

**这篇 paper & 关联代码**:
- Paper PDF: 待 ACM 2026 正式发布, 当前版本可从作者 [Rawmantic AI](https://github.com/Rawmantic-AI) 或 [UESTC](https://github.com/UESTC) 实验室页找
- 最 close 的 baseline: [DIFIX3D+ (Wu et al. CVPR 2025)](https://github.com/WuJianzhe/DIFIX3D-Plus)

**Backbone 与组件**:
- Diffusion backbone: [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)
- Depth estimator: [Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3) | [MVSFormer++](https://arxiv.org/abs/2401.11673)
- Softmax splatting: [Niklaus & Liu CVPR 2020](https://arxiv.org/abs/2004.03865)
- Deformable attention (概念关联): [Deformable DETR](https://arxiv.org/abs/2010.04159)

**Dataset**:
- [DL3DV-10K](https://github.com/YangLiu2022/DL3DV-10K)
- [Mip-NeRF360](https://jonbarron.info/mipnerf360/)

**Geometric attention 相关工作**:
- [pixelSplat](https://arxiv.org/abs/2312.12337)
- [MVSplat](https://arxiv.org/abs/2403.07607)
- [DepthSplat](https://arxiv.org/abs/2411.09525)

**Render-and-refine 一族**:
- [ReconFusion (Wu et al. CVPR 2024)](https://reconfusion.github.io/)
- [3DGS-Enhancer (Liu et al. NeurIPS 2024)](https://arxiv.org/abs/2410.16284)
- [GenFusion (Wu et al. CVPR 2025)](https://genfusion.github.io/)
- [GSFixer (Yin et al. 2025)](https://arxiv.org/abs/2508.09667)

**Sparse-view 3DGS regularization**:
- [FSGS](https://arxiv.org/abs/2402.04307)
- [DNGaussian](https://arxiv.org/abs/2403.06912)
- [DropGaussian](https://arxiv.org/abs/2412.02029)
- [NexusGS (epipolar depth prior)](https://arxiv.org/abs/2411.16751)

**Loss 设计**:
- [FILM (frame interpolation, Gram loss 来源)](https://arxiv.org/abs/2207.08014)

---

# GeoQuery: Geometry-Query Diffusion for Sparse-View Reconstruction 深度解析

## 1. 核心 Insight: 这篇 paper 真正在解决什么问题

这篇 paper 的核心贡献, 与其说是 "提出一个新模块", 不如说是 **诊断出了一个被前人忽视的 failure mode**, 然后**对症下药**。

### 1.1 "Render-and-Refine" 范式的隐性 assumption

最近的 sparse-view 3DGS pipeline (DIFIX3D+ [Wu et al. 2025b](https://github.com/WuJianzhe/DIFIX3D-Plus), ReconFusion [Wu et al. 2024](https://reconfusion.github.io/), 3DGS-Enhancer [Liu et al. 2024b](https://arxiv.org/abs/2410.16284), GenFusion [Wu et al. 2025a](https://genfusion.github.io/)) 都遵循一个共同 pattern:

```
sparse-view 3DGS  →  artifact-prone rendering ̃I_t  →  diffusion refiner  →  Î_t  →  pseudo-GT  →  refine 3DGS
```

在这个 loop 中, diffusion refiner 需要从 clean reference view $I^r$ 借信息。**所有前人方法都默认一个关键 assumption: target view 的 query feature 是"足够好"的**——它至少能正确地告诉你 "我要 retrieve 什么"。

这就是 multi-view self-attention 的 standard recipe (来自 MVDream [Shi et al. 2023b](https://arxiv.org/abs/2308.16512), Zero123++ [Shi et al. 2023a](https://arxiv.org/abs/2310.15110), One-2-3-45++ [Liu et al. 2024a](https://one-2-3-45.github.io/)):

$$F^{concat} = [F^t; F^r], \quad F^{out} = \text{Softmax}\left(\frac{Q^t (K^t; K^r)^\top}{\sqrt{d}}\right)(V^t; V^r)$$

query 来自 $F^t$, key/value 来自 concat 后的 target+reference。Attention map 自由选择从 target 自身或 reference 处取信息。

### 1.2 Query Contamination 的本质

**Query contamination 的本质是 "corrupted address lookup"**。可以把 cross-view attention 看成一个 content-addressable memory:
- **Query $Q^t$** = "我要查询的语义地址"
- **Key $K^r$** = "reference 中每个 token 的语义标签"
- **Value $V^r$** = "reference 中存储的实际内容"

当 $\tilde{I}^t$ 含 floaters / structural collapse / blur, query embedding $Q^t = W_Q F^t$ 不再代表 "这个像素应该对应 reference 中的哪一部分", 而是被 artifacts 污染成一个**"看起来像 artifact 但又不是真实语义"的混合向量**。这个 vector 在 reference 中找到的最近邻, 几乎必然是语义错配的——它可能 retrieve 到一个完全无关的区域, 仅仅因为 reference 那块区域的 low-level texture 与 artifact 有某种 spurious 相似性。

这就形成一个 **positive feedback loop** (vicious cycle, 论文用语):

$$\text{artifact} \xrightarrow{Q^t \text{ corrupted}} \text{wrong retrieval} \xrightarrow{\text{hallucination}} \text{more artifact} \xrightarrow{\cdots}$$

**Table 3 的 region-level 分析是这篇 paper 最 sharp 的实验证据**: DIFIX3D+ 在 low-error region (3DGS 渲染本来就好的地方) 的 PSNR **反而下降 0.75 dB** (从 25.82 → 25.07)。这个结果乍看反直觉——为什么 refiner 会把"已经好的地方"搞坏?

直觉解释: query contamination 不是局部现象, **它是一个全局的 retrieval bias**。即使在 low-error region, query feature 仍然参与了 attention 与所有 reference tokens 的 inner product 计算。High-error region 的 corrupted queries 会污染共享的 softmax normalization (因为它从 reference "掠夺"了本应分配给 low-error region 的 attention mass), 导致 low-error region 的 retrieval 也被扰动。这跟 transformer 中 "一个 bad token 污染整个 sequence" 的现象本质相同。

而 GeoQuery 在两个 region 都获得正收益 (low-error +0.37, high-error +4.03), 说明它把"全局 retrieval"做对了。

## 2. 方法的技术解剖

### 2.1 Geometric Correspondence: 用 deterministic address 替代 semantic address

论文的核心 trick 在 Equation (5)–(6):

**Step 1: Back-project reference pixel to 3D**
$$\mathbf{x} = \pi^{-1}(\mathbf{u}_r, D^r(\mathbf{u}_r), \mathbf{K}_r)$$

变量含义:
- $\mathbf{u}_r \in \mathbb{R}^2$: reference view 中某像素坐标 $(u, v)$
- $D^r(\mathbf{u}_r) \in \mathbb{R}^+$: reference 在该像素处的 metric depth (来自 [MVSFormer++](https://arxiv.org/abs/2401.11673) 或 [Depth Anything v3](https://arxiv.org/abs/2511.10647))
- $\mathbf{K}_r \in \mathbb{R}^{3\times 3}$: reference 相机内参
- $\pi^{-1}$: back-projection, 实质是 $\mathbf{x} = D^r(\mathbf{u}_r) \cdot \mathbf{K}_r^{-1} [\mathbf{u}_r; 1]$

**Step 2: Project 3D point to target view**
$$\mathbf{u}_t = \pi(\mathbf{K}_t \mathbf{T}_t \mathbf{T}_r^{-1} \mathbf{x})$$

变量含义:
- $\mathbf{T}_t, \mathbf{T}_r \in \mathbb{R}^{4\times 4}$: target / reference 相机外参 (world-to-camera)
- $\mathbf{T}_r^{-1}$: reference camera-to-world, 把 $\mathbf{x}$ (在 reference camera 坐标系下) 转回 world
- $\mathbf{T}_t \mathbf{T}_r^{-1}$: reference-to-target relative pose
- $\pi(\cdot)$: perspective projection $\mathbf{u}_t = (x'/z', y'/z')$ where $[x'; y'; z'] = \mathbf{K}_t \mathbf{T}_t \mathbf{T}_r^{-1} \mathbf{x}$

**Step 3: Forward splatting 得到稠密 correspondence field**

这里有一个**很 deliberate 的设计选择**: 用 **forward splatting** (从 reference 拍到 target), 而不是 backward warping (从 target 拉取 reference)。

直觉: target view $\tilde{I}^t$ 是 corrupted 的, 如果在 target 上做 depth 估计并 back-project, 这个 depth 也会被 artifact 污染。Reference 是 clean training view, 它的 depth (来自 MVS) 更可靠。所以**几何先验必须从 reference 流向 target**, 这与 information flow 方向一致。

Forward splatting 之后:
- $C_{t \leftarrow r} \in \mathbb{R}^{H \times W \times 2}$: target 每个像素对应 reference 中哪个像素
- $M_{t \leftarrow r} \in \{0,1\}^{H \times W}$: 该像素是否被 reference 看到 (occlusion 处理)

这里 forward splatting 必然有 collision (多个 reference 像素映射到同一 target 值) 和 hole (target 像素没被任何 reference 像素覆盖)。Softmax splatting [Niklaus & Liu 2020](https://arxiv.org/abs/2004.03865) 用 z-buffer 加权可以缓解, hole 则由 $M_{t \leftarrow r} = 0$ 标记。

### 2.2 Geometry-Indexed Proxy Features (Equation 6)

$$F^{r \rightarrow t}(\mathbf{u}_t) = M_{t \leftarrow r}(\mathbf{u}_t) \odot \text{Sample}(F^r, C_{t \leftarrow r}(\mathbf{u}_t))$$

变量解析:
- $F^r \in \mathbb{R}^{H/l \times W/l \times d}$: reference view 在 UNet 某 layer 的 feature map, $l$ 是 downsampling factor, $d$ 是 channel
- $C_{t \leftarrow r}$ 被 downsample 到 feature resolution
- $\text{Sample}(\cdot, \cdot)$: bilinear interpolation, 在 $F^r$ 上以 $C_{t \leftarrow r}(\mathbf{u}_t)$ 为坐标采样
- $M_{t \leftarrow r} \in \{0, 1\}$: binary mask, occlusion / out-of-FOV 处为 0
- $\odot$: element-wise product (broadcasting 到 channel 维)

**这个操作的语义是 "把 reference feature warp 到 target 的几何骨架上"**。proxy feature 在结构上与 target 对齐 (因为 $C_{t \leftarrow r}$ 描述的就是 target→reference 的几何映射), 但内容是 clean reference 的。

**关键 trick**: 把 $F^{r \rightarrow t}$ 作为 **query $Q$** (而不是 key/value) 喂给 attention。这是对 "query contamination" 的直接 fix: 用 clean feature 充当 query, 让 attention 在 reference 的局部 neighborhood 内做精细化检索。

### 2.3 Local $k \times k$ Window Attention (Equation 7–8)

$$K_\Delta(\mathbf{u}_t) = \text{Sample}(W_K F^r, C_{t \leftarrow r}(\mathbf{u}_t) + \Delta)$$
$$V_\Delta(\mathbf{u}_t) = \text{Sample}(W_V F^r, C_{t \leftarrow r}(\mathbf{u}_t) + \Delta)$$
$$F_{geo}^t(\mathbf{u}_t) = \sum_{\Delta \in \Omega} \text{Softmax}_\Delta\left(\frac{\langle Q(\mathbf{u}_t), K_\Delta(\mathbf{u}_t)\rangle}{\sqrt{d}}\right) V_\Delta(\mathbf{u}_t)$$

变量:
- $\Delta \in \Omega$: 在以 $C_{t \leftarrow r}(\mathbf{u}_t)$ 为中心的 $k \times k$ 窗口内的偏移量, $\Omega = \{-\lfloor k/2\rfloor, \dots, \lfloor k/2\rfloor\}^2$
- $W_K, W_V \in \mathbb{R}^{d \times d}$: 标准 attention 的 key/value projection
- $Q(\mathbf{u}_t) = W_Q F^{r \rightarrow t}(\mathbf{u}_t)$: proxy query (注意 query 来自 proxy, 不是 target)
- $\langle \cdot, \cdot \rangle$: inner product
- $\sqrt{d}$: standard scaled dot-product attention scaling

**为什么需要 local window?** 三个理由:

1. **Depth estimation 误差**: metric depth 不是完美的, MVSFormer++ 在 sparse-view 下误差更大。$C_{t \leftarrow r}$ 给的对应点可能有几个 pixel 的偏移。Local window 让 attention 在对应点附近"搜索", 容忍这种几何误差。
2. **Disocclusion 与非刚性**: forward splatting 处理不了 occlusion boundary 处的 sub-pixel 误差, local window 提供 spatial robustness。
3. **防止 spurious long-range match**: 即使 query 是 clean proxy, 在整个 reference feature map 上做 global attention 仍可能 retrieve 到 remote 但语义相似的区域 (e.g. 多个相似的窗户)。Local constraint 把 attention 限制在几何确定的 neighborhood, 抑制 spurious match。

**Fig. 8 的 ablation 非常 informative**: $k=3$ 最优, $k=1$ (只取 proxy 点本身) 太弱, $k \geq 5$ 又开始退化。这个 U 形曲线非常符合 "bias-variance tradeoff":
- $k$ 小: 几何 prior 太死, 不能容忍 depth 误差
- $k$ 大: 又回到 global attention 的语义检索模式, 失去几何约束

**复杂度**: $O(N^2) \to O(Nk^2)$, 对 1024×576 的 feature map, $k=3$ 时 reduction factor 是 $\sim 57000\times$。这是 paper 在 21GB A100 上能跑通的实际原因。

### 2.4 Adaptive Fusion Gate (Equation 9)

$$w = \sigma(\text{MLP}([F^t, F_{geo}^t]))$$
$$F^t(\mathbf{u}_t) \gets (1 - w(\mathbf{u}_t)) \odot F^t(\mathbf{u}_t) + w(\mathbf{u}_t) \odot F_{geo}^t(\mathbf{u}_t)$$

变量:
- $[\cdot, \cdot]$: channel-wise concat
- $\text{MLP}$: 2-layer MLP, 输入 $2d$ channel, 输出 1 channel per spatial location
- $\sigma$: sigmoid, 输出 $w \in (0,1)^{H/l \times W/l}$
- $\gets$: in-place update of $F^t$

**这是一个 graceful degradation 设计**:
- 当 $M_{t \leftarrow r}(\mathbf{u}_t) = 0$ (occlusion / correspondence failure): $F^{r \rightarrow t}(\mathbf{u}_t) = 0$ → $F_{geo}^t(\mathbf{u}_t)$ 输出接近 0 (因为 attention 的 V 输入也依赖 proxy 的几何引导, 实际中 occlusion 区域 softmax 会散开但 V 还是 reference 局部, MLP 学会把 $w \to 0$) → $F^t \gets F^t$, fall back 到 global self-attention branch
- 当 correspondence 强: $F_{geo}^t$ 提供 clean geometric evidence, MLP 学到 $w \to 1$, 充分利用 GCA

这与 paper 在 Discussion 部分的论断完全一致: "When reference projections are invalid or occluded, the validity mask disables local retrieval, and the adaptive fusion falls back to the global branch for semantic completion."

### 2.5 Training Objective (Equation 10–11)

$$\mathcal{L}_{gram} = \frac{1}{L}\sum_{l=1}^{L} \beta_l \|G_l(\hat{I}) - G_l(I)\|_2$$
$$G_l(I) = \phi_l(I)^\top \phi_l(I)$$
$$\mathcal{L} = \lambda_{recon}\mathcal{L}_{recon} + \lambda_{lpips}\mathcal{L}_{lpips} + \lambda_{gram}\mathcal{L}_{gram}$$

变量:
- $\phi_l(\cdot)$: VGG-16 第 $l$ 层 feature extractor, 输出 $C_l \times H_l \times W_l$
- $G_l(I) \in \mathbb{R}^{C_l \times C_l}$: Gram matrix = $\phi_l(I)$ flatten 到 $C_l \times (H_l W_l)$ 再 self-multiply, 捕获 channel-wise correlation (= style)
- $\beta_l$: layer-specific weight (浅层小, 深层大)
- $\lambda_{recon}, \lambda_{lpips}, \lambda_{gram}$: 各 loss 权重

为什么用 Gram loss? 这是 [FILM](https://arxiv.org/abs/2207.08014) (Reda et al. 2022) 借来的 trick, 用于 video frame interpolation, 现在用在 diffusion refinement。Gram loss 关注 texture style 而非具体 content 位置, 与 $\mathcal{L}_{lpips}$ (perceptual) 和 $\mathcal{L}_{recon}$ (pixel) 互补。**目的**: pixel loss 给 anchor, LPIPS 给 semantic structure, Gram 给 sharp texture。三者结合避免 diffusion output 过度 smooth。

## 3. 与相关方法的联系和区分

### 3.1 vs. Epipolar Attention (pixelSplat [Charatan et al. 2024](https://arxiv.org/abs/2312.12337), MVSplat [Chen et al. 2024](https://arxiv.org/abs/2403.07607), DepthSplat [Xu et al. 2025a](https://arxiv.org/abs/2411.09525))

Epipolar attention 把 cross-view attention 限制在 epipolar line 上 (用已知 pose 计算)。这是**几何约束 attention 的一种 form**。

**关键区别**:
- Epipolar attention 仍然用 target feature 作 query, 在 epipolar line 上做 content-based retrieval → **仍然受 query contamination 影响**
- GCA 用 proxy feature (来自 reference) 作 query, 在 correspondence 点的 local window 上检索 → **绕过 query contamination**

**Table 6 的实验对比是核心**: 
| Method | PSNR | FID |
|---|---|---|
| Epipolar Attn only | 19.21 | 12.16 |
| Proxy + Epipolar Attn | 19.73 | 11.05 |
| Proxy + GCA (full) | 19.88 | 10.20 |

**仅用 epipolar 约束** → +0.42 PSNR over baseline。**加 proxy query** → +1.0 PSNR。**加 GCA window** → 再 +0.15 PSNR。 

Proxy query 的贡献最大, 这正是 paper 的核心 insight: query source 比 attention 范围更重要。

### 3.2 vs. ReconFusion / DIFIX3D+

ReconFusion 和 DIFIX3D+ 都是 image diffusion-based refiner, 但都用 standard multi-view self-attention。GeoQuery 本质上是 DIFIX3D+ 的"几何增强版"——在同一个 backbone (SD-Turbo [Sauer et al. 2024](https://arxiv.org/abs/2403.12006)) 上加 GCA module, 训练 pipeline 几乎不变。**这让 GeoQuery 的 ablation 极其干净**, 因为它在与 DIFIX3D+ 几乎完全相同的 setup 下获得 +1.09 dB PSNR。

### 3.3 vs. Video Diffusion 方法 (3DGS-Enhancer, GenFusion, GSFixer [Yin et al. 2025](https://arxiv.org/abs/2508.09667))

Video diffusion 方法本质上是把 cross-view consistency 委托给 3D U-Net 的 temporal attention。这等价于隐式学了一个 "video prior", 但这个 prior 没有显式几何 ground truth。当 3DGS 渲染出的 view sequence 有 artifacts, temporal attention 仍然会被污染。

GeoQuery 选择"显式几何 + image diffusion"路线, 用 metric depth 提供 hard geometric anchor, 代价是依赖 MVS 的质量。

### 3.4 与 RAG (Retrieval Augmented Generation) 的类比

如果把 cross-view attention 类比为 RAG, 那么:
- **Standard multi-view self-attention** = 用 query 自身的 embedding 去 retrieve 文档。Query 如果 garbage, retrieval 就 garbage。这正是 RAG 系统的 "noisy query problem"。
- **GCA** = 用 deterministic metadata (e.g. document ID, timestamp, geo-location) 替代 embedding-based retrieval, 然后在 retrieved 文档附近做 semantic refinement。这相当于 "hybrid retrieval" (BM25 + dense) 中的 BM25 部分。

这个类比可以帮助记忆: **proxy query 是"几何 BM25", global self-attention 是"dense retrieval"**, 两者通过 adaptive gate 加权融合。

## 4. 实验结果的关键解读

### 4.1 Table 1: Artifact Removal (DL3DV)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|---|---|---|---|---|
| DIFIX3D+ (w/o ref) | 18.26 | 0.493 | 0.388 | 21.04 |
| DIFIX3D+ | 18.79 | 0.529 | 0.348 | 12.83 |
| GeoQuery | **19.88** | **0.566** | **0.314** | **10.20** |

观察:
- "w/o ref" DIFIX3D+ → "with ref" DIFIX3D+: +0.53 PSNR, -8.21 FID。说明 reference 的确帮助, 但增益有限
- DIFIX3D+ → GeoQuery: +1.09 PSNR, -2.63 FID。**几何引导比"加 reference"本身还重要**

FID 从 21.04 → 10.20 的减半, 说明 distribution-level 的 realism 提升巨大。FID 对 structural hallucination 非常敏感 (hallucination 产生 fake modes), 这与"消除 query contamination"的论断一致。

### 4.2 Table 2: Sparse-view Reconstruction 跨 view 数

最 informative 的对比是 3-view regime:
- Mip-NeRF360: DIFIX3D 14.15 → GeoQuery **15.07 (+0.92 dB)**
- DL3DV: DIFIX3D 15.20 → GeoQuery **15.98 (+0.78 dB)**

3-view 是 paper 提到的 "extreme sparsity" 设定, 也是方法 designed for 的 hard case。在 9-view 时增益缩小到 +0.25-0.68 dB, 这是符合预期的: view 越多, 3DGS 渲染 artifact 越少, query contamination 越弱, GeoQuery 的边际收益越小。

### 4.3 Table 3: Region-level (最 sharp 的诊断)

| Region | 3DGS | DIFIX3D+ | GeoQuery |
|---|---|---|---|
| Low-error ($e \leq 30$) | 25.82 | 25.07 (-0.75) | **26.19 (+0.37)** |
| High-error ($e > 30$) | 11.16 | 13.16 (+2.00) | **15.19 (+4.03)** |

Threshold $\tau = 30$ 大约对应 PSNR=20 dB 的等高线 (因为 error 是某种 per-pixel metric, 与 PSNR 的逆关系大致是 PSNR $\sim -10\log(e^2/255^2)$)。

**DIFIX3D+ 在 low-error 区域 -0.75 dB** 是一个"smoking gun"——它证实了 query contamination 不仅没修好 high-error 区域, 还把本来好的区域搞坏了。GeoQuery 在两个区域都正向, 说明它真正解决了 retrieval 错位问题。

### 4.4 Table 4: Ablation (核心 ablation)

| SA | GCA/R | GCA/P | AF | PSNR | FID |
|---|---|---|---|---|---|
| ✓ | | | | 18.79 | 12.83 (DIFIX3D+) |
| ✓ | ✓ | | ✓ | 19.42 | 11.60 |
| ✓ | | ✓ | | 19.57 | 11.11 |
| ✓ | | ✓ | ✓ | **19.88** | **10.20** |

(GCA/R: query 仍来自 corrupted rendering; GCA/P: query 来自 proxy; AF: adaptive fusion)

**关键观察**:
- Row 1 → Row 2 (加 GCA 但用 rendering query): +0.63 PSNR。**仅靠 local window 约束本身就能改善**, 但提升有限
- Row 1 → Row 3 (用 proxy query 但 w/o adaptive fusion): +0.78 PSNR。**Proxy query 的提升大于 local window**
- Row 3 → Row 4 (加 adaptive fusion): +0.31 PSNR。AF 不是 huge boost 但稳定改善
- Row 2 → Row 4 (rendering query → proxy query, 都加 AF): +0.46 PSNR。再次确认 proxy 是核心

**这是 paper 的 core argument 的实验支柱**: 你可以加 local window, 可以加 adaptive gate, 但**如果你不把 query 换成 proxy, 你就解决不了 contamination**。

## 5. 设计选择的 deeper thinking

### 5.1 为什么用 forward splatting 而不是 backward warping?

Backward warping 的标准做法: 在 target 上 query depth $D^t$, back-project 到 3D, project 到 reference, 采样 reference。但这要求 $D^t$ 可靠, 而 $D^t$ 来自 corrupted $\tilde{I}^t$。

Forward splatting 完全绕开 target depth, **几何信号严格单向从 clean reference 流向 target**。代价是 forward splatting 有 hole 问题 (某些 target 像素没有 reference 像素映射过来), 但 $M_{t \leftarrow r}$ 显式标记这些 hole, 让 adaptive gate 决定 fallback。

### 5.2 Proxy feature 与"clean reference feature 之间的 attention"

仔细看 Equation 8: query 是 $W_Q F^{r \rightarrow t}$ (proxy, 来自 reference), key/value 也是 $W_K F^r, W_V F^r$ (reference)。**这是 reference feature 上的 self-attention, 但 spatially indexed by correspondence**。

等价于: "在 reference 的 local neighborhood 内, 找到与 proxy 最匹配的 tokens"。这其实在做 **sub-pixel accurate correspondence refinement**——depth 给的对应点是 approximate 的, attention 在 local window 内可以微调到更准确的对应。

这让我联想到 [Deformable Attention](https://arxiv.org/abs/2010.04159) 的思路: 给定 reference point, 在周围 4D offset 上采样。GCA 是 deformable attention 的 special case, 其中 reference point = geometric correspondence, offsets 限制在 $k \times k$ grid。

### 5.3 为什么不直接 warp reference pixel color, 而要 feature-level attention?

直接 warp reference RGB 到 target 就是 classical view synthesis (e.g. [IBRNet](https://arxiv.org/abs/2102.13090))。问题:
- Occlusion boundary 处会撕裂
- Depth 误差导致 ghosting
- 没有 generative completion 能力

Feature-level attention 让 diffusion model 在 reference feature 上做 "soft lookup", 用 softmax 权重处理 depth 不确定性。最终 output 仍走 diffusion decoder, 保留生成能力。

### 5.4 失败模式 (paper 自己承认的 limitation)

Paper 在 Section 6 提到:
1. **Textureless region**: MVS depth 失败 → correspondence 不准 → GCA 退化
2. **Specular surface**: depth 估计与几何先验都失败
3. **Extreme viewpoint disparity**: reference 完全看不到的区域, $M_{t \leftarrow r} = 0$, GCA 整个 fall back

这告诉我们: **GeoQuery 的 gain 与 depth quality 强正相关**。如果换上更好的 depth estimator (paper 用 Depth Anything v3), 性能可能继续涨。这也指向 future work: 让 GCA 与 depth estimation joint-train, 让 depth 在 artifact 区域也能鲁棒。

## 6. 更宏观的 intuition

### 6.1 Generative prior vs. Geometric prior 的 "调和"

3DGS 重建领域的两大流派:
- **Geometric prior**: depth regularization (DNGaussian [Li et al. 2024](https://arxiv.org/abs/2403.06912)), epipolar constraint (NexusGS [Zheng et al. 2025](https://arxiv.org/abs/2411.16751)), MVS prior
- **Generative prior**: diffusion-based refiner (DIFIX3D+, ReconFusion, GenFusion)

两者一直互相"skeptical": geometric 方法说"diffusion hallucinates inconsistent 3D", generative 方法说"geometric 在 sparse-view 下 under-constrained"。

GeoQuery 的立场: **两者本质不冲突, 而是应该被 hierarchical 地组织**:
- Geometric prior 提供 "retrieval address" (where to look)
- Generative prior 提供 "content synthesis" (what to render)

这是一个**很有移植价值的范式**。可以想象在 RAG、autonomous driving (geometric prior from LiDAR + generative prior for occluded regions)、medical imaging (anatomical prior + diffusion) 等领域都有类似的 design pattern。

### 6.2 "Address vs. Content" 的抽象

从更抽象的层面看, attention 机制的"query" 是一个 address, "key/value" 是一个 content store。任何 attention-based retrieval 系统都隐含一个 assumption: **address 是 trustworthy 的**。

当一个系统中有部分 input 不可靠 (artifacts, occlusion, noise), 直接用这些 input 生成 query 是危险的。解决方案有两种:
1. **Replace address source**: 用另一个 modality 的 deterministic signal 替代 (GeoQuery 的选择)
2. **Robustify the address**: 用 iterative refinement / EM-like approach 让 query 逐步 clean up (类似 AlphaFold 的 iterative attention)

GeoQuery 选择 1, 因为 geometric cue 是 deterministic 且可获得的。在缺少 deterministic modality 的场景, 选择 2 更合适。

### 6.3 关于为什么这种 architecture 没有更早出现

我个人的 hypothesis: **"render and refine" paradigm 本身比较新** (ReconFusion 2024, DIFIX3D 2024, DIFIX3D+ 2025)。早期 NeRF + diffusion (e.g. DreamFusion) 是 SDS-based, 不涉及 cross-view attention, 自然没有这个问题。

只有当 community 转向 "用 diffusion 做 post-hoc refiner" + "用 multi-view self-attention 跨视图同步" 这两个组合, query contamination 才显现。这是一个新 paradigm 的 secondary failure mode, 需要 maturity 才能诊断。Paper [Wu et al. 2025b] DIFIX3D+ 在 self-attention 上做了一些设计但没有击中要害, GeoQuery 是顺着这条线继续往下的 logical step。

## 7. 一些可以追问的 future work

读完 paper 我会想知道:

1. **GCA 是否可以 self-distill**? 用 GeoQuery 输出的 clean image 训练一个"corrupted-query-tolerant" self-attention, 让模型逐步摆脱对 proxy 的依赖?
2. **能不能端到端 learn depth + GCA**? 现在 depth 是 frozen 预计算, 但 depth error 直接影响 GCA。Joint train 能让 depth 适配下游需求。
3. **GCA 能否用于 video diffusion** (3DGS-Enhancer 范式)? Video diffusion 的 temporal attention 同样有 query contamination, 可以用相邻帧的 optical flow 做 proxy。
4. **GCA 在 large-baseline setting** (e.g. 360° 重建中两个相隔很远的 view) 的表现? Paper 在 Mip-NeRF360 上实验了, 但没有专门分析 baseline angle vs. gain。
5. **能否用 learned correspondence (e.g. LoFTR, [GMFlow](https://arxiv.org/abs/2111.13630)) 替代 depth-based correspondence**? 这可能让方法 generalize 到 dynamic scene。

## 8. 总结性 intuition

如果让我用一句话概括 GeoQuery 的贡献:

> **"Don't ask a corrupted image to tell you where to look in a clean image. Use geometry to figure out where to look, then let attention refine the details."**

这句话背后是一个 deep design principle: **在一个 retrieval pipeline 中, address 和 content 应该由不同 source 提供, 且 address 的 source 必须比 content 的 source 更可靠**。当 address source 不可靠时, 整个 retrieval 系统会进入 "garbage in, garbage out" 的 positive feedback loop, 而 break 这个 loop 的最 surgical 的方法就是 replace address source。

GeoQuery 用 geometric prior 替代 corrupted target feature 作为 attention 的 address source, 同时保留 diffusion model 作为 content synthesizer, 在 sparse-view 3DGS refinement 这个具体 task 上验证了该 principle 的有效性。

---

**References & Resources**:
- Paper: GeoQuery (to appear, ACM 2026)
- Closest baseline: [DIFIX3D+](https://github.com/WuJianzhe/DIFIX3D-Plus), [ReconFusion](https://reconfusion.github.io/)
- Backbone: [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)
- Depth estimator: [Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3), [MVSFormer++](https://arxiv.org/abs/2401.11673)
- Datasets: [DL3DV-10K](https://github.com/YangLiu2022/DL3DV-10K), [Mip-NeRF360](https://jonbarron.info/mipnerf360/)
- Related geometric attention: [pixelSplat](https://arxiv.org/abs/2312.12337), [MVSplat](https://arxiv.org/abs/2403.07607), [DepthSplat](https://arxiv.org/abs/2411.09525)
- Deformable attention (conceptual related): [Deformable DETR](https://arxiv.org/abs/2010.04159)
- Softmax splatting: [Niklaus & Liu 2020](https://arxiv.org/abs/2004.03865)
- Video diffusion for 3DGS: [3DGS-Enhancer](https://arxiv.org/abs/2410.16284), [GenFusion](https://genfusion.github.io/), [GSFixer](https://arxiv.org/abs/2508.09667)
- Sparse-view 3DGS regularization: [FSGS](https://arxiv.org/abs/2402.04307), [DNGaussian](https://arxiv.org/abs/2403.06912), [DropGaussian](https://arxiv.org/abs/2412.02029)
