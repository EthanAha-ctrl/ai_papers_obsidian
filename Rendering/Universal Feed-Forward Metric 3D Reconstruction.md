---
source_pdf: Universal Feed-Forward Metric 3D Reconstruction.pdf
paper_sha256: d85c86e28fa2679f726d050d85b3f7743ba93f8321b933dcbee78eccd3ff26d4
processed_at: '2026-08-12T20:15:43-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MapAnything 用人话讲

## 一句话版本

以前做 3D reconstruction 要针对不同任务训不同的 model，MapAnything 训一个 model 搞定 12+ 种任务，而且能吃各种 input（image、camera intrinsics、poses、depth 都行，给啥用啥），输出还是 metric scale 的。

---

## 问题出在哪

先说清楚为啥要搞这个。

传统 feed-forward 3D reconstruction 有个麻烦事：**representation 设计得不好**。

DUSt3R 和 MASt3R 预测的是 pointmap——就是每个 pixel 对应的 3D 点坐标。这个东西把 camera intrinsics、extrinsics、scene geometry 全部揉在一起了。后果是：

1. 你没法把已知的 camera intrinsics 喂进去当 input，因为 intrinsics 已经被 entangle 在 pointmap 里了
2. 多 view 之间有 redundant prediction，VGGT 要预测两套东西（pointmaps 一套，cameras+depth 一套）
3. Metric scale 没法搞，因为 pointmap 本身是 up-to-scale 的

这就像你写代码把所有变量都塞在一个 struct 里，想单独改一个字段都不行。

---

## 核心招数：Factorize

MapAnything 的核心 insight 就一个词：**factorize**。

把 3D scene 表示拆成四个独立的量：

```
3D Scene = Rays × Depth × Pose × Scale
```

具体说：

- **Rays $R_i$**：每个 pixel 的 ray direction，就是 camera 每个像素看出去的方向。这玩意儿纯粹由 camera intrinsics 决定，跟场景内容无关，跟 scale 也无关
- **Depth $\tilde{D}_i$**：每个 pixel 沿 ray 的深度，但是在 up-to-scale space 里预测的
- **Pose $\tilde{P}_i$**：camera 的 rotation + translation，translation 也是 up-to-scale 的
- **Scale $m$**：一个标量，把 up-to-scale 的东西 upgrade 成 metric 的

最终的 metric 3D 点：

$$X_i^{\text{metric}} = m \cdot (O_i \cdot (R_i \cdot \tilde{D}_i) + \tilde{T}_i)$$

拆开看：
1. $R_i \cdot \tilde{D}_i$ = local pointmap（在 camera 坐标系下的 3D 点）
2. $O_i \cdot (\cdot) + \tilde{T}_i$ = 变换到 world frame（view 1 的坐标系）
3. $m \cdot (\cdot)$ = 放大到 metric scale

**Intuition**：这跟 classic SfM 的分解是对应的。SfM 里你也是先做 calibration（→ rays）、再 triangulation（→ depth）、再 pose estimation（→ pose）、再 scale recovery（→ scale）。MapAnything 只是把这套流程 bake 进了 transformer。

**为啥这么拆好？**

因为每个量的性质不一样：
- Rays 跟 scale 无关，所以可以单独预测，loss 也不需要 normalize
- Depth 在 up-to-scale space 预测，这样 monocular 数据（没有 metric scale GT）也能拿来训练
- Pose 的 rotation 跟 scale 无关，translation 跟 scale 有关，所以拆开
- Scale 是个标量，用单独的 token 预测，可以从 visual cues 学 scale prior

---

## 架构怎么搭的

### Encoder 阶段

**Image encoder**：用 DINOv2 ViT-G，取第 24 层的 patch features。为啥用 DINOv2 不用 DUSt3R 自己的 encoder？实验发现 DINOv2 的 semantic features 对 multi-view correspondence 帮助很大，而且 fine-tune 后 generalization 更好。

**Geometric input encoder**：
- Dense 的东西（rays, depth）：shallow conv + pixel unshuffle，project 到跟 image features 同样的维度
- Global 的东西（quaternion, translation direction, depth scale, pose scale）：4-layer MLP with GeLU

然后所有 encoding 做 layer norm → sum → layer norm，得到每个 view 的 token sequence。

### 关键 trick：Scale Token

在 N 个 view 的 patch tokens 后面 append 一个 learnable scale token。这个 token 跟所有 view tokens 一起进 transformer，做 attention，最后通过 MLP 输出 metric scale $m$。

**Intuition**：这个 scale token 就像 CLS token 一样，它 attend 所有 views 的信息，然后输出一个 scene-wide 的 scale。当 input 里有 metric pose/depth 时，它可以从这些信息里 "读出" scale；当只有 image input 时，它从 visual cues（人多大、车多大）guess scale。

### Multi-view Transformer

16 层 alternating attention transformer，用 DINOv2 ViT-G 最后 16 层初始化。

**Alternating attention** 是啥？就是在 within-view self-attention 和 cross-view attention 之间交替。

为啥不直接全 global attention？Table S.2b 的 ablation 很清楚：
- Alternating: rel=0.29, τ=31.8
- Global: rel=0.53, τ=19.7

差了一大截。Intuition 是：within-view attention 先让每个 view 自己消化 local 信息（类似单图理解），cross-view attention 再做 correspondence。不是所有 token pairs 都需要 cross-view 信息交换，先 local 后 global 更高效。

### Decoder

三路输出：
1. **DPT head**：解码 N view 的 patch tokens → 每个 view 的 dense outputs（rays, depth, mask, confidence）
2. **Pose head**：average pooling + conv → quaternion + up-to-scale translation
3. **Scale head**：scale token → 2-layer MLP with ReLU → metric scale $m$（exponentially scaled，因为 scene scale 变化大）

---

## Loss 怎么设的

这部分有几个重要的 design choice。

### Scale-invariant normalization

因为 depth 和 pointmap 是 up-to-scale 的，要先 normalize 再算 loss：

$$\hat{z} = \frac{\|(\hat{X}_i[V_i])_{i=1}^N\|}{\sum_{i=1}^N V_i}, \quad \tilde{z} = \frac{\|(\tilde{X}_i[V_i])_{i=1}^N\|}{\sum_{i=1}^N V_i}$$

其中 $V_i$ 是 GT validity mask，$\hat{z}$ 是 GT 的 scale，$\tilde{z}$ 是 prediction 的 scale。

**关键 trick**：算 metric scale loss 的时候，用 stop-gradient：
$$z^{\text{metric}} = m \cdot \text{sg}(\tilde{z})$$

为啥？防止 scale loss 的梯度污染 geometry prediction。你只想让 $m$ 去拟合 scale，不想让 geometry prediction 为了迎合 scale loss 而变形。

### Log-space loss

对 depth 和 pointmap，在 log space 算 loss：
$$f_{\log}(\mathbf{x}) = \frac{\mathbf{x}}{\|\mathbf{x}\|} \cdot \log(1 + \|\mathbf{x}\|)$$

- $\mathbf{x}/\|\mathbf{x}\|$：方向信息
- $\log(1 + \|\mathbf{x}\|)$：magnitude 压缩

为啥要 log？Depth 值从 0.1m 到 100m 都有，直接 L2 loss 会让远处的点 dominate。Table S.2a 的 ablation：
- 有 log: rel=0.29, τ=31.8
- 没 log: rel=0.39, τ=27.3

差很多。

### Rotation loss 的细节

Quaternion 有个 double cover 问题：$Q$ 和 $-Q$ 表示同一个 rotation。所以 loss 要取 min：
$$\mathcal{L}_{\text{rot}} = \sum_{i=1}^N \min(\|\hat{Q}_i - Q_i\|, \|-\hat{Q}_i - Q_i\|)$$

### 各 loss 权重

$$\mathcal{L} = 10 \cdot \mathcal{L}_{\text{pointmap}} + \mathcal{L}_{\text{rays}} + \mathcal{L}_{\text{rot}} + \mathcal{L}_{\text{translation}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{lpm}} + \mathcal{L}_{\text{scale}} + \mathcal{L}_{\text{normal}} + \mathcal{L}_{\text{GM}} + 0.1 \cdot \mathcal{L}_{\text{mask}}$$

Global pointmap loss 权重 10（最重要），mask loss 权重 0.1（辅助）。

Normal loss 和 gradient matching loss 只在 synthetic data 上用，因为 real data 的 geometry 太 noisy。

所有 regression loss 用 Barron 的 adaptive robust loss（$c=0.05, \alpha=0.5$）处理 outliers，还排除 top 5% per-pixel loss values。

---

## 训练策略

### Input probability-based training

这是让一个 model 支持 64 种 input combination 的关键。

训练时每个 batch 随机决定给不给 geometric input：
- Overall probability: 0.9（90% 的 batch 有 geometric input）
- 每个 factor（rays, depth, pose）：0.5 的概率给
- 如果 depth 给了，50% 概率给 dense depth，50% 概率给 90% sparsified depth
- Per-view probability: 0.95（允许部分 view 有 info，部分没有）
- 0.05 概率不给 metric scale factor（即使 GT 是 metric 的，训练 robustness）

**Intuition**：这就像 data augmentation，但是在 input modality 层面做。模型学到的是 "有这些 info 就用，没有就靠 image 推断" 的能力。

### Two-stage curriculum

- Stage 1（6 天）：4-2 views，batch 768-1536
- Stage 2（4 天）：24-2 views，batch 128-1536，10× lower LR

先少 view 学 basic correspondence，再多 view 学 long-range consistency。Figure S.2 显示 Stage 1 训完的 model（只见过 4 views）已经能 generalize 到 100 views。

### Covisibility-based sampling

预先计算所有 image pair 的 covisibility（基于 GT depth+pose 的 reprojection error），训练时用 25% threshold 做 random walk，保证采到的 views 形成 connected graph。

为啥？避免 disjoint image sets 做 input——transformer 没法在不相关的 views 间建 correspondence。

---

## 实验结果怎么看

### Multi-view reconstruction（Figure 4）

在 ETH3D、ScanNet++、TartanAirV2 上，2 到 100 views：
- Images only 就 SOTA
- 加 auxiliary input 进一步提升
- VGGT 在 ~50 views 后开始退化，MapAnything 更 stable

### Two-view（Table 2）

跟 Pow3R（唯一另一个支持 geometric priors 的方法）比：

| Input | Method | Scale rel | Points rel | AUC |
|-------|--------|----------|-----------|-----|
| Images only | VGGT | - | 0.20 | 34.2 |
| Images only | MapAnything | 0.13 | 0.08 | 56.0 |
| Imgs+K+Pose+Depth | Pow3R | 0.03 | 0.03 | 81.3 |
| Imgs+K+Pose+Depth | MapAnything | 0.01 | 0.02 | 94.8 |

MapAnything 几乎全面碾压，尤其 pose estimation（AUC 94.8 vs 81.3）。

### Single-view calibration（Table 3）

没专门训练 single image，但还是 SOTA：
- MapAnything: 1.06°
- AnyCalib: 2.01°
- VGGT: 4.00°

这说明 factored ray representation 有很好的 generalization——rays 本质就是 calibration 的参数化。

### Ablation 的关键发现

Table 5a 显示 factored representation（RDP & Scale）是最优的：
- Images only: τ=40.7（vs Local PM+Pose 的 33.2）
- Images+K+Pose: τ=57.8（vs 53.5）

Table 5b 显示 universal training 比 expert training 还好：
- Images only: universal τ=40.7 vs expert 31.8
- 一个 model 顶三个 expert model

**Intuition**：multi-task training 让模型学到更 general 的 representation，不同 input configuration 之间互相 regularize。

---

## 跟其他方法的区别

| 方法 | Representation | Input | Scale | Post-processing |
|------|---------------|-------|-------|-----------------|
| DUSt3R | Coupled pointmap | Images only | Up-to-scale | 需要优化 |
| MASt3R | Coupled pointmap | Images only | Up-to-scale | 需要优化 |
| VGGT | Redundant pointmap+depth | Images only | Up-to-scale | 无 |
| FASt3R | Redundant pointmap | Images only | Up-to-scale | 无 |
| π3 | Decoupled PM+pose | Images only | Up-to-scale | 无 |
| Pow3R | Coupled pointmap | Images+priors | 部分 | 需要优化 |
| **MapAnything** | **Factored (R,D,P,S)** | **Images+任意 priors** | **Metric** | **无** |

MapAnything 的优势：
1. Factored representation 避免 redundancy
2. 支持 heterogeneous inputs（给啥用啥）
3. 直接输出 metric scale
4. 任意 number of views
5. 支持 generic central camera（不限于 pinhole）

---

## 我的理解

### Factored representation 为啥 work？

这其实是把 classic vision geometry 的 decomposition principle 重新 inject 到 transformer 里。

Classic SfM 的步骤：
1. Feature matching → cross-view attention
2. Calibration → ray prediction
3. Triangulation → $R_i \cdot \tilde{D}_i$
4. Pose estimation → pose head
5. Scale recovery → scale token

每个步骤都有对应的 geometric meaning，network 学到的 representation 是 interpretable 的。这跟直接预测 pointmap 不同，pointmap 把所有东西揉在一起，network 需要自己 figure out 怎么 disentangle。

### Scale token 为啥 work？

Monocular reconstruction 本质上 up-to-scale（Gauge freedom）。但 real-world 需要 metric scale。

MapAnything 把 scale 作为一个单独的 token，让 network 从 visual cues 学 scale prior。比如：
- 人的高度 ~1.7m
- 车的长度 ~4.5m
- 门的高度 ~2m

当有 metric pose/depth input 时，scale token 可以 attend 到这些信息，实现 scale 的 "grounding"。

当只有 image 时，scale token 从学到的 prior 里 guess。Table 4 显示 KITTI（outdoor driving）的 metric scale estimation 还不错（rel=8.48），但 ScanNet（indoor）较差（rel=31.12），论文说可能是 benchmark dataset quality 问题。

### Universal training 为啥比 expert training 好？

Table 5b 的结果有点 surprising：universal model 在 images-only 上 τ=40.7，比 expert model 的 31.8 好 9 个点。

我的理解：不同 input configuration 之间互相 regularize。比如 images-only 的 model 可以从 pose-conditioned 的训练中学到更好的 multi-view consistency，因为 pose 信息提供了 "correspondence supervision" 的信号。

这跟 multi-task learning 的 general insight 一致：相关任务一起训练能学到更好的 shared representation。

### Limitations

论文承认的：
1. 不 model input noise/uncertainty
2. 不支持 only-camera-no-image 的 view
3. Scalability 受 pixel-to-token one-to-one 限制
4. 不支持 dynamic scene

我觉得还有个 fundamental limitation：**metric scale estimation 在纯 image input 时还是不够 robust**。Table 4 显示 ScanNet 上 rel=31.12，这说明 indoor scene 的 scale prior 还没学好。可能需要更多 metric scale 数据，或者引入其他 scale cue（如 object detection）。

---

## Reference

- [MapAnything paper](https://arxiv.org/)（待 release）
- [VGGT](https://vgg-t.github.io/)
- [DUSt3R](https://github.com/naver/dust3r)
- [MASt3R](https://github.com/naver/mast3r)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [Pow3R](https://arxiv.org/abs/2504.07991)
- [FASt3R](https://arxiv.org/abs/2501.13928)
- [π3](https://arxiv.org/abs/2507.13347)
- [MASt3R-SfM](https://github.com/naver/mast3r-sfm)
- [MASt3R-SLAM](https://arxiv.org/abs/2502.19197)
- [MoGe-2](https://arxiv.org/abs/2504.07054)
- [UniDepthV2](https://github.com/piccinelli flask/UniDepth)
- [Metric3D v2](https://github.com/YvanYin/Metric3D)
- [Depth Pro](https://github.com/apple/ml-depth-pro)
- [AnyCalib](https://arxiv.org/abs/2506.19088)
- [ScanNet++](https://github.com/scannetpp/scannetpp)
- [TartanAir](https://theairlab.org/tartanair)
- [ETH3D](https://eth3d.net/)
- [BlendedMVS](https://github.com/YoYo000/BlendedMVS)
- [DL3DV-10K](https://github.com/DL3DV-10K/DL3DV-10K)
- [Robust MVD Benchmark](https://github.com/labc-mfs/RobustMVD)
- [Adaptive Robust Loss](https://github.com/jonbarron/robust_loss_pytorch)

---

总结一句话：MapAnything 通过 factored representation（rays + depth + pose + scale）把 3D reconstruction 的不同物理量解耦，让一个 transformer model 能处理 12+ 种 task configuration，支持 heterogeneous inputs，直接输出 metric scale，训练效率还比 expert model 高。核心 insight 是把 classic vision geometry 的 decomposition wisdom inject 到 modern transformer 里。

---

# MapAnything: Universal Feed-Forward Metric 3D Reconstruction 深度解析

## 一、核心 insight：为什么需要 Factored Representation？

传统 feed-forward 3D reconstruction 方法（如 DUSt3R, MASt3R, VGGT）直接预测 pointmaps，这种方式存在几个 fundamental 问题：

1. **Coupled representation**：pointmaps 把 camera intrinsics、extrinsics 和 scene geometry 全部 entangle 在一起，导致无法灵活接受部分已知信息作为 input
2. **Redundancy**：VGGT 需要两个 branch 分别预测 pointmaps 和 cameras+depth，FASt3R 需要为每个 view 预测 redundant pointmaps
3. **Scale ambiguity**：纯 pointmap 表示无法直接处理 metric scale，因为 monocular reconstruction 本身是 up-to-scale 的

MapAnything 的 key insight 是把 scene geometry **factorize** 成四个解耦的组件：

$$\text{MapAnything}(\hat{\mathcal{I}}, [\hat{\mathcal{R}}, \hat{\mathcal{Q}}, \hat{\mathcal{T}}, \hat{\mathcal{D}}]) = \{m, (R_i, \tilde{D}_i, \tilde{P}_i)_{i=1}^N\}$$

其中：
- $m \in \mathbb{R}$：global metric scaling factor（单个标量，scene-wide）
- $R_i \in \mathbb{R}^{3 \times H \times W}$：per-view local ray directions（normalized to unit length）
- $\tilde{D}_i \in \mathbb{R}^{1 \times H \times W}$：per-view ray depths in up-to-scale space（tilde 表示 up-to-scale）
- $\tilde{P}_i \in \mathbb{R}^{4 \times 4}$：per-view camera pose in frame of image $\hat{I}_1$

这种 factorization 的妙处在于：
- **Ray directions $R_i$** 是纯 intrinsics 的函数，与 scene scale 完全无关
- **Depth $\tilde{D}_i$** 在 up-to-scale space 中预测，scale 由单独的 $m$ 处理
- **Pose $\tilde{P}_i$** 的 rotation 与 scale 无关，translation 在 up-to-scale space 中预测
- 这四个量在物理上自然解耦，可以分别用不同的 head 预测

最终的 metric reconstruction 通过简单组合得到：
$$\tilde{L}_i = R_i \cdot \tilde{D}_i \quad \text{(local pointmap)}$$
$$\tilde{X}_i = O_i \cdot \tilde{L}_i + \tilde{T}_i \quad \text{(world frame pointmap)}$$
$$X_i^{\text{metric}} = m \cdot \tilde{X}_i \quad \text{(metric reconstruction)}$$

其中 $O_i$ 是从 quaternion $Q_i$ 转换得到的 rotation matrix。

**Intuition**：这种分解相当于把 3D reconstruction 问题分解为"形状"(shape via rays + depth) + "位姿" (pose via quaternion + translation) + "尺度" (scale via single scalar) 三个子问题，每个子问题都有合适的 invariance/equivariance 性质。这与 classic SfM 中分别处理 calibration, pose estimation, triangulation 的思路是相通的，但用 end-to-end 方式联合训练。

---

## 二、Architecture 详解

### 2.1 整体 Pipeline

参考 Figure 2，整个 pipeline 分为三个阶段：

**Stage 1: Encoding**
- Image encoder: DINOv2 ViT-G（取第 24 层 normalized patch features），输出 $F_I \in \mathbb{R}^{1536 \times H/14 \times W/14}$
- Dense geometric encoder: shallow conv encoder + pixel unshuffle (size 14)，处理 ray directions 和 normalized ray depths，输出 $F_R, F_D \in \mathbb{R}^{1536 \times H/14 \times W/14}$
- Global geometric encoder: 4-layer MLP with GeLU，处理 rotations (quaternions), translation directions, depth scales, pose scales，输出 $F_Q, F_T, F_{\hat{z}_d}, F_{\hat{z}_p} \in \mathbb{R}^{1536}$

**Stage 2: Fusion**
- 所有 encodings 经过 layer norm → sum → layer norm
- 添加 constant reference view embedding 到 view 1 的 patch tokens（区分 reference frame）
- Append 一个 learnable scale token 到 N 个 view patch tokens 的末尾
- 输入 16-layer alternating-attention transformer（24 heads, 1536 dim, MLP ratio 4）

**Stage 3: Decoding**
- DPT head 解码 N-view patch tokens → N 个 dense per-view outputs（rays, depths, masks, confidence）
- Average pooling-based conv pose head → quaternions + up-to-scale translations
- 2-layer MLP with ReLU → metric scale $m$（exponentially scaled 因为 scene scale 变化大）

### 2.2 关键设计决策

**为什么用 DINOv2 而非 DUSt3R encoder 或 CroCov2？**
论文在实验中发现 DINOv2 在 downstream performance、convergence speed、generalization（尤其 fine-tune 后）方面最优。DINOv2 ViT-G 提供了 1536 维的 rich semantic features，这些 features 对 multi-view correspondence 和 geometry understanding 都很有帮助。

**为什么 multi-view transformer 也用 DINOv2 初始化？**
论文发现用 DINOv2 ViT-G 的最后 16 层初始化 multi-view transformer 比 random init 的 ViT-B 收敛快得多，最终性能也更好（参考 Figure S.3 的对比）。

**为什么不用 RoPE？**
DINOv2 的 patch-level positional encoding 已经足够，RoPE 会在每个 attention layer 中引入不必要的 bias（因为 RoPE 原本是设计给 LLM 的 sequence modeling）。

**Alternating Attention 的作用**
参考 VGGT [67] 的设计，alternating attention 在 within-view self-attention 和 cross-view attention 之间交替。Table S.2b 显示 alternating attention 显著优于 global attention with view PE（FASt3R 用的方案）：
- Alternating: rel=0.29, τ=31.8
- Global w/ View PE: rel=0.53, τ=19.7

**Intuition**：within-view attention 让每个 view 先聚合 local 信息（类似单图理解），cross-view attention 再做 correspondence。这种分解比直接 global attention 更高效，因为不是所有 token pairs 都需要 cross-view 信息交换。

### 2.3 Geometric Input Encoding 的巧妙之处

为了让模型接受 heterogeneous inputs（部分 view 有 geometric info，部分没有），论文设计了以下 factorization：

**Depth factorization**:
$$\hat{z}_{d_i} \in \mathbb{R}^+ \quad \text{(per-view average depth)}$$
$$\hat{D}_i / \hat{z}_{d_i} \quad \text{(normalized ray depths)}$$

**Pose factorization**:
- Rotation 单独编码（因为 rotation 与 scale 无关）
- Translation 分解为：
  - Pose scale: $\hat{z}_p = \frac{1}{|S_t|} \sum_{i \in S_t} \|\hat{T}_i\|$（所有有 translation 的 view 共享同一个 pose scale）
  - Normalized translation: $\hat{T}_i / \hat{z}_p$

**Scale handling**:
- 只有当提供的 pose/depth 是 metric 时才使用 scale 信息作为 input
- Scale 值很大且变化剧烈，所以用 log-transform 后再 encode

**为什么 depth 和 pose 的 normalization 要 decouple？**
因为训练时我们不假设 depth 和 pose 总是一起提供。但在 training objective 中，predicted depth 和 pose 是一起 normalize 的（为了 multi-view consistency）。

---

## 三、Loss Function 深度解析

### 3.1 Scale-Invariant Normalization

对于 up-to-scale predictions，需要先 normalize 再计算 loss。论文用 GT validity masks $V_i$ 计算：

$$\hat{z} = \|(\hat{X}_i[V_i])_{i=1}^N\| / \sum_{i=1}^N V_i \quad \text{(GT scale)}$$
$$\tilde{z} = \|(\tilde{X}_i[V_i])_{i=1}^N\| / \sum_{i=1}^N V_i \quad \text{(prediction scale)}$$

**关键技巧**：为了防止 scale loss 的梯度污染 geometry prediction，用 detached scale：
$$z^{\text{metric}} = m \cdot \text{sg}(\tilde{z})$$
其中 $\text{sg}$ 表示 stop-gradient。

### 3.2 各个 Loss 项

**Ray loss**（scale-invariant by nature）：
$$\mathcal{L}_{\text{rays}} = \sum_{i=1}^N \|\hat{R}_i - R_i\|$$

**Rotation loss**（处理 quaternion 的 double cover）：
$$\mathcal{L}_{\text{rot}} = \sum_{i=1}^N \min(\|\hat{Q}_i - Q_i\|, \|-\hat{Q}_i - Q_i\|)$$
因为 $Q$ 和 $-Q$ 表示同一个 rotation，需要取 min。

**Translation loss**（scale-invariant）：
$$\mathcal{L}_{\text{translation}} = \sum_{i=1}^N \|\hat{T}_i/\hat{z} - \tilde{T}_i/\tilde{z}\|$$

**Log-space loss**（关键！）：
$$f_{\log}: \mathbf{x} \to (\mathbf{x}/\|\mathbf{x}\|) \cdot \log(1 + \|\mathbf{x}\|)$$
- 第一个因子 $\mathbf{x}/\|\mathbf{x}\|$ 保留方向信息
- 第二个因子 $\log(1 + \|\mathbf{x}\|)$ 压缩 magnitude
- 这种 log-space 处理对 depth 和 pointmap 极其重要，因为 depth 值在 0.1m 到 100m+ 之间变化很大
- Table S.2a 显示 no log loss 会让 rel 从 0.29 退化到 0.39

**Depth loss**:
$$\mathcal{L}_{\text{depth}} = \sum_{i=1}^N \|f_{\log}(\hat{D}_i/\hat{z}) - f_{\log}(\tilde{D}_i/\tilde{z})\|$$

**Local pointmap loss**:
$$\mathcal{L}_{\text{lpm}} = \sum_{i=1}^N \|f_{\log}(\hat{L}_i/\hat{z}) - f_{\log}(\tilde{L}_i/\tilde{z})\|$$

**Global pointmap loss**（confidence-weighted，类似 DUSt3R）：
$$\mathcal{L}_{\text{pointmap}} = \sum_{i=1}^N (C_i \|f_{\log}(\hat{X}_i/\hat{z}) - f_{\log}(\tilde{X}_i/\tilde{z})\| - \alpha \log(C_i))$$
- $C_i$ 是 predicted confidence
- $-\alpha \log(C_i)$ 防止 confidence 坍塌为 0

**Scale loss**:
$$\mathcal{L}_{\text{scale}} = \|f_{\log}(\hat{z}) - f_{\log}(z^{\text{metric}})\|$$

**Auxiliary losses**（只在 synthetic data 上）：
- Normal loss $\mathcal{L}_{\text{normal}}$ on local pointmaps
- Multi-scale gradient matching $\mathcal{L}_{\text{GM}}$ on log z-depth
- 这些 loss 在 real data 上不用，因为 real geometry 噪声大

**Mask loss**:
$$\mathcal{L}_{\text{mask}} = \text{BCE}(\text{predicted non-ambiguous mask}, \text{GT})$$

**Total loss**:
$$\mathcal{L} = 10 \cdot \mathcal{L}_{\text{pointmap}} + \mathcal{L}_{\text{rays}} + \mathcal{L}_{\text{rot}} + \mathcal{L}_{\text{translation}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{lpm}} + \mathcal{L}_{\text{scale}} + \mathcal{L}_{\text{normal}} + \mathcal{L}_{\text{GM}} + 0.1 \cdot \mathcal{L}_{\text{mask}}$$

- Global pointmap loss 权重 10（最重要）
- Mask loss 权重 0.1（辅助）
- 所有 regression loss 用 adaptive robust loss（$c=0.05, \alpha=0.5$）处理 outliers
- 排除 top 5% per-pixel loss values 避免 imperfect GT 的影响

---

## 四、Universal Training Strategy

### 4.1 Input Probability-based Training

为了一个 model 支持多种 input configuration，论文用 probability-based sampling：

- Overall geometric input probability: 0.9
- 每个 factorization（rays, depth, pose）的 input probability: 0.5
- 当 depth 被选中时，dense 和 90% sparsified depth 各 50% 概率
- Per-view input probability: 0.95（允许部分 view 有 geometric info）
- 0.05 概率不提供 metric scale factor 给 metric GT 数据集（robustness）

这种设计支持 $2^6 = 64$ 种 input combinations。

### 4.2 Multi-View Sampling

对每个 dataset 预先计算所有 image pairs 的 covisibility（基于 GT depth + pose 的 reprojection error）。训练时用 covisibility threshold 25% 做 random walk sampling，得到 single connected component graph。

**为什么需要 connected component？**
避免 disjoint image sets 作为 input，因为 transformer 无法在不相关 views 之间建立 correspondence。

### 4.3 Two-Stage Curriculum

- **Stage 1** (6 days): 4-2 views，effective batch 768-1536
- **Stage 2** (4 days): 24-2 views，effective batch 128-1536，10× lower LR

**Intuition**：先在小 number of views 上学习 basic geometry correspondence，再 extend 到更多 views。这种 curriculum 让模型在 4 views 训练后就能 generalize 到 100 views（Figure S.2）。

### 4.4 数据集与 License

13 个 datasets 分为两个 model variant：
- **Apache 2.0 model**: 6 datasets（BlendedMVS, Mapillary Planet-Scale Depth, ScanNet++ v2, Spring, TartanAirV2-WB, UnrealStereo4K）
- **CC BY-NC 4.0 model**: 额外 7 datasets（Aria Synthetic Environments, DL3DV-10K, Dynamic Replica, MegaDepth, MVS-Synth, ParallelDomain-4D, SAIL-VOS 3D）

MPSD 原本是 monocular metric depth dataset，论文额外 acquire 了 pose 和 camera info，构建了 ~72K scenes 的 multi-view metric dataset，并 open-source 了 metadata。

---

## 五、实验结果分析

### 5.1 Multi-View Dense Reconstruction (Figure 4)

在 ETH3D, ScanNet++ v2, TartanAirV2-WB 上，从 2 到 100 views 测试：
- **Images only**: MapAnything 在 rel, τ, ATE_RMSE, AUC@5, err° 上都 SOTA
- **Images + auxiliary inputs**: 性能显著进一步提升
- 与 VGGT 相比，MapAnything 在大 number of views 时优势更明显（VGGT 在 ~50 views 后退化）

### 5.2 Two-View Reconstruction (Table 2)

关键对比：MapAnything vs Pow3R（唯一另一个支持 scene priors 的 feed-forward 方法）：

| Input | Method | Scale rel↓ | Points rel↓ | τ↑ | ATE↓ | AUC↑ |
|-------|--------|-----------|------------|-----|------|------|
| Images | VGGT | - | 0.20 | 43.2 | 0.07 | 34.2 |
| Images | MapAnything | 0.13 | 0.08 | 57.5 | 0.02 | 56.0 |
| Imgs+Intrinsics+Poses+Depth | Pow3R | 0.03 | 0.03 | 90.1 | 0.01 | 81.3 |
| Imgs+Intrinsics+Poses+Depth | MapAnything | 0.01 | 0.02 | 82.0 | 0.00 | 94.8 |

MapAnything 在几乎所有配置下都优于 Pow3R，尤其 pose estimation（AUC 94.8 vs 81.3）。

### 5.3 Single-View Calibration (Table 3)

尽管没有专门训练 single-image 输入，MapAnything 仍达到 SOTA：
- MapAnything: avg 1.06°
- AnyCalib: 2.01°
- MoGe-2: 1.95°
- VGGT: 4.00°

这证明了 factored ray representation 的 generalization 能力——rays 本质上就是 camera calibration 的参数化。

### 5.4 Depth Estimation (Table 4)

在 Robust-MVD benchmark 上：
- **Single-view metric**: MapAnything 在 KITTI 上 rel=8.48（vs Metric3DV2 的 8.70），但在 ScanNet 上较差（rel=31.12）
- 论文承认 ScanNet 上的 metric scale estimation sub-optimal，可能因为 benchmark dataset quality 问题
- **Multi-view with alignment**: MapAnything rel=4.04 on KITTI，接近 π3 的 3.09

### 5.5 Ablation Studies (Table 5)

**Scene Representation ablation**:
- RDP & Scale（论文方案）最优
- RDP without scale: metric scale 无法准确预测
- Local PM + Pose: 没有 factored representation 的 baseline
- LPMP & Scale: 一种中间方案

**Expert vs Universal Training**:
- Universal training 在 images-only 上 τ=40.7 vs expert 的 31.8（显著更好）
- 在 images+intrinsics+poses 上 universal 略差（rel 0.05 vs 0.03）
- **结论**：multi-task training 是 highly efficient 的，一个 universal model 相当于多个 bespoke models

---

## 六、与 Concurrent Work 的对比

### 6.1 vs VGGT [67]
- VGGT 预测 redundant pointmaps 和 cameras+depth（两个 branch）
- VGGT 用第一个 frame 作为 reference coordinate，π3 [74] 通过 fine-tune 移除这个 redundancy
- MapAnything 直接用 factored representation 避免 redundancy
- MapAnything 支持 heterogeneous inputs，VGGT 只支持 images

### 6.2 vs DUSt3R/MASt3R [72, 29]
- DUSt3R/MASt3R 预测 coupled pointmap representation
- 需要昂贵的 post-processing 和 symmetric inference
- MapAnything 直接预测 decoupled quantities，无需 post-processing

### 6.3 vs FASt3R [78]
- FASt3R 用 positional encoding 处理 long-sequence inference
- FASt3R 预测 redundant pointmaps，dense geometry 受 pose estimation 影响
- MapAnything 的 factored representation 避免 this coupling

### 6.4 vs Pow3R [23]
- Pow3R 是第一个支持 geometric priors 作为 input 的方法
- 但 Pow3R 只支持 2-view pinhole camera with single focal length
- Pow3R 无法 condition on metric scale
- MapAnything 支持任意 number of views 和 generic central projection camera model

### 6.5 vs π3 [74]
- π3 fine-tune VGGT 预测 up-to-scale decoupled local pointmaps 和 global pose
- 但 π3 没有单独的 metric scale factor
- Table 5a 显示这种 design 是 sub-optimal 的

---

## 七、Limitations 与 Future Directions

论文承认的 limitations：
1. 不显式 model geometric inputs 的 noise/uncertainty
2. 不支持只有 camera 没有 image 的 view（如 novel view synthesis 的 target view）
3. Iterative inference 和 test-time compute scaling 未探索
4. Multi-modal fusion 在 input 前 完成，可以探索更高效的直接 input 方式
5. Scalability 受 pixel-to-output one-to-one mapping 限制
6. 不处理 dynamic motion 和 scene flow

---

## 八、Intuition Building：为什么 Factored Representation 这么 powerful？

让我用 classic SfM 的视角来理解：

Classic SfM 分解为：
1. **Feature detection & matching**（对应 cross-view attention）
2. **Two-view pose estimation**（对应 pose head）
3. **Camera calibration**（对应 ray prediction）
4. **Triangulation**（对应 $R_i \cdot \tilde{D}_i$）
5. **Bundle adjustment**（对应 end-to-end training 的 multi-view consistency loss）
6. **Scale recovery**（对应 metric scale factor $m$）

MapAnything 把这些步骤 "baked into" 一个 transformer，但保留了每一步的 geometric structure：
- Rays $R_i$ ↔ calibration
- Depth $\tilde{D}_i$ ↔ triangulated point distance
- Pose $\tilde{P}_i$ ↔ camera extrinsics
- Scale $m$ ↔ metric scale recovery

这种 correspondence 让 network 学到的 representation 是 geometrically interpretable 的，也使得 heterogeneous inputs 可以自然地 inject 到对应的位置。

**Scale factor $m$ 的妙处**：
- Monocular reconstruction 本质上 up-to-scale（Gauge freedom）
- 但 real-world 应用需要 metric scale
- 传统方法用 known camera height、GPS、IMU 等 recovery scale
- MapAnything 把 scale 作为单独的 learnable token，让 network 从 visual cues（人物大小、车辆尺寸等）学习 scale prior
- 当有 metric pose/depth input 时，scale token 可以 attend 到这些信息，实现 scale 的 "grounding"

---

## 九、Reference Links

- **MapAnything**: 待 release（Apache 2.0 + CC BY-NC 4.0）
- **VGGT**: https://vgg-t.github.io/ - Visual Geometry Grounded Transformer
- **DUSt3R**: https://github.com/naver/dust3r - Geometric 3D Vision Made Easy
- **MASt3R**: https://github.com/naver/mast3r - Grounding Image Matching in 3D
- **DINOv2**: https://github.com/facebookresearch/dinov2 - Learning Robust Visual Features
- **DPT**: https://arxiv.org/abs/2103.13413 - Vision Transformers for Dense Prediction
- **Pow3R**: https://arxiv.org/abs/2504.07991 - Empowering Unconstrained 3D Reconstruction
- **FASt3R**: https://arxiv.org/abs/2501.13928 - Fast 3D Reconstruction
- **π3**: https://arxiv.org/abs/2507.13347 - Scalable Permutation-equivariant Visual Geometry
- **MASt3R-SfM**: https://github.com/naver/mast3r-sfm - Fully-integrated SfM
- **MASt3R-SLAM**: https://arxiv.org/abs/2502.19197 - Real-time Dense SLAM
- **MoGe-2**: https://arxiv.org/abs/2504.07054 - Monocular Geometry with Metric Scale
- **UniDepthV2**: https://github.com/piccinelli flask/UniDepth - Universal Monocular Metric Depth
- **Metric3D v2**: https://github.com/YvanYin/Metric3D - Versatile Monocular Geometric Foundation Model
- **Depth Pro**: https://github.com/apple/ml-depth-pro - Sharp Monocular Metric Depth
- **AnyCalib**: https://arxiv.org/abs/2506.19088 - Model-agnostic Single-view Calibration
- **ScanNet++**: https://github.com/scannetpp/scannetpp - High-fidelity Indoor 3D Scenes
- **TartanAir**: https://theairlab.org/tartanair - Visual SLAM Dataset
- **ETH3D**: https://eth3d.net/ - Multi-view Stereo Benchmark
- **BlendedMVS**: https://github.com/YoYo000/BlendedMVS - Large-scale MVS Dataset
- **DL3DV-10K**: https://github.com/DL3DV-10K/DL3DV-10K - Large-scale 3D Vision Dataset
- **Mapillary Planet-Scale Depth**: https://arxiv.org/abs/2011.12859 - Planet-scale Depth Dataset
- **Robust MVD**: https://github.com/labc-mfs/RobustMVD - Robust Multi-view Depth Benchmark
- **DeMoN**: https://github.com/lmb-freiburg/demon - Depth and Motion Network
- **DeepV2D**: https://github.com/princeton-vl/DeepV2D - Video to Depth
- **MultiMAE**: https://github.com/EPFL-VILAB/MultiMAE - Multi-modal Multi-task Masked Autoencoders
- **Adaptive Robust Loss**: https://github.com/jonbarron/robust_loss_pytorch - General & Adaptive Robust Loss

---

## 十、Potential Extensions & Speculations

基于论文的 limitations 和我的理解，几个有潜力的 future direction：

1. **Uncertainty-aware inputs**：当前 geometric inputs 被当作 deterministic，可以引入 probabilistic encoding（如 Gaussian tokens），让模型 attend 时考虑 input uncertainty。

2. **Test-time compute scaling**：类似 LLM 的 chain-of-thought，可以 iterative refinement。MASt3R-SLAM 已经展示了 iterative inference 的潜力。

3. **Token compression for large scenes**：当前 pixel-to-token one-to-one 限制 scalability。可以引入 hierarchical token merging（类似 ToMe）或 latent scene representation。

4. **Dynamic scene support**：Any4D [26] 已经探索 4D reconstruction，可以把 scene flow 作为额外 factorization 加入。

5. **Neural radiance field integration**：MapAnything 的 factored outputs 可以直接 initialize NeRF 或 3D Gaussian Splatting，实现 reconstruction + rendering 的 unified pipeline。

6. **Language grounding**：Scale token $m$ 可以 condition on text prompts（"这个场景是室内/室外"），利用 LLM 的 common sense 提升 scale estimation。

7. **Self-supervised scale learning**：当前 metric scale 依赖 metric GT datasets。可以用 video temporal consistency + known object categories 实现 weakly-supervised scale learning。

---

希望这个 deep dive 帮助你 build intuition 关于 MapAnything 的核心 design choices。Factored representation 是这篇论文最重要的 insight——它把 classic vision geometry 的 decomposition principle 重新 inject 到 modern transformer architecture 中，既保留了 geometric interpretability，又获得了 end-to-end training 的优势。这种 "old wisdom + new architecture" 的 combination 模式在 foundation model 时代值得特别关注。
