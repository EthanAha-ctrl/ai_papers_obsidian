---
source_pdf: VGGT.pdf
paper_sha256: 5d7ab44d129172f39ef5918feb0014c4d041fd0e92f995d4dbbd7327a6e91d52
processed_at: '2026-08-13T00:26:36-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VGGT 人话版：一个 transformer 吃下照片吐出 3D 世界

## 一句话总结

你给它几张到几百张照片，它一次性告诉你：每张照片的相机在哪、朝向哪、深度图长啥样、3D 点云长啥样、点怎么在图之间 track。全程一个 forward pass，不到 1 秒。而且比那些跑 10 秒 Bundle Adjustment 的老方法还要准。

项目地址：https://github.com/facebookresearch/vggt
Demo：https://huggingface.co/spaces/facebook/vggt

---

## 为什么这件事以前没人做成

3D reconstruction 这个领域，几十年来的套路是 SfM pipeline。你拿 COLMAP 跑一下，它在内部做的事情大致是：feature detection → matching → triangulation → bundle adjustment。Bundle Adjustment 就是一个巨大的 non-linear least squares，反复调整 camera poses 和 3D points，让所有照片的 reprojection error 最小。整个流程慢，而且每个环节都要单独 tune。

深度学习进来之后，大家逐步替换 pipeline 的各个零件。SuperPoint 换掉 feature detector，SuperGlue / LightGlue 换掉 matcher，MVSNet 换掉 dense reconstruction。但是 BA 这个东西一直是座山，因为它是全局优化，很难用纯 feed-forward 替代。

DUSt3R (https://github.com/naver/dust3r) 做了一件很聪明的事：它让网络直接预测两张图的 point map，所谓 point map 就是每个 pixel 对应的 3D 坐标。这个表示是 over-parameterized 的，但好处是 camera 参数可以从 point map 反推（通过 PnP）。DUSt3R 的问题在于它只能处理两张图，多图的话需要 pairwise 预测完再做一个昂贵的 global alignment，后处理比前向还慢。

MASt3R (https://github.com/naver/mast3r) 是 DUSt3R 的升级版，加了 matching head，但还是 pairwise。

VGGT 的 insight 很直接：既然 pairwise 限制了我们，那就直接做 N-wise。既然 global alignment 慢，那就让网络一次性把 alignment 学进 weights 里。用大数据和大模型硬怼。

---

## 架构到底怎么设计的

### 输入侧：DINOv2 patchify

每张图先过 DINOv2 (https://github.com/facebookresearch/dinov2) 被切成 14×14 patches，每个 patch 变成一个 token。DINOv2 已经在海量无标注图像上自监督训练过，它的 features 自带很强的 semantic 和 geometric 信息，用作者的话说，这让早期训练非常稳定，对 hyperparameter 不敏感。

所以 VGGT 其实是站在 DINOv2 的肩膀上。这个细节很关键，说明 VGGT 的成功不完全来自架构，也来自 good initialization。

### 主体：Alternating Attention，这是论文最核心的设计

一个标准 transformer 把所有 tokens 放一起做 self-attention 就行了，但是 VGGT 做了一个小改动：交替使用 frame-wise 和 global attention。

具体来说，24 层 block，每层先做 frame-wise self-attention，再做 global self-attention。

- **Frame-wise self-attention**：每张图的 K 个 tokens 自己内部 attend。相当于每张图独立做一次 ViT 的 self-attention。
- **Global self-attention**：所有 N 张图的所有 tokens 一起 attend。相当于 N×K 个 tokens 的全连接。

为什么交替？直觉是：global attention 让图之间交换信息，做跨视图的几何推理（类似 MVS 的 cost volume 但更 soft）；frame-wise attention 让每张图的 tokens 在被全局信息 "污染" 之后，回炉重新 ground 到自己 frame 的 context。

Ablation 在 Table 5 里很清楚：

| Architecture | Overall↓ |
|---|---|
| Cross-Attention | 1.061 |
| Global Self-Attention Only | 0.827 |
| **Alternating-Attention** | **0.709** |

只用 global attention 明显比交替差。这个 "回炉" 步骤非常关键。intuition 上，它像一种 structural normalization，防止每张图的 token identity 在深层网络里被 average 掉。

### Camera token 与 Register token 的不对称

每张图除了 image tokens，还加 1 个 camera token 和 4 个 register tokens。

关键设计：第一张图的 camera token 和 register tokens 是一组 learnable parameters $\bar{\mathbf{t}}^{\mathbf{g}}, \bar{\mathbf{t}}^R$，其他所有图共享另一组 $\bar{\bar{\mathbf{t}}}^{\mathbf{g}}, \bar{\bar{\mathbf{t}}}^R$。

这打破 permutation equivariance，让网络知道哪张图是 reference。因为所有 3D 输出都 anchored 在第一张图的相机坐标系。

Register tokens 借鉴 Darcet et al. 的工作 (https://arxiv.org/abs/2309.16588)，他们发现 ViT 高层 feature map 会有 high-norm artifact tokens，其实是网络把 global information 塞到某些 token 里。VGGT 主动给网络几个 "垃圾桶" tokens 专门存这些信息，output 时直接丢掉，避免污染 image tokens。

---

## 输出头：四个 head 同时预测

### Camera head

4 层 self-attention + 1 层 linear，输入 camera tokens，输出 9 维向量 $[\mathbf{q}, \mathbf{t}, \mathbf{f}]$：
- $\mathbf{q} \in \mathbb{R}^4$：rotation quaternion，4 维单位四元数表示 SO(3)
- $\mathbf{t} \in \mathbb{R}^3$：translation vector
- $\mathbf{f} \in \mathbb{R}^2$：field of view，假设 principal point 在图像中心

第一张图的 extrinsics 强制为单位变换，$\mathbf{q}_1 = [0,0,0,1]$，$\mathbf{t}_1 = [0,0,0]$。

### DPT head (dense prediction)

DPT (https://github.com/isl-org/DPT) 把 transformer tokens 上采样回 full resolution feature map，然后输出：
- Depth map $D_i \in \mathbb{R}^{H \times W}$
- Point map $P_i \in \mathbb{R}^{3 \times H \times W}$（3D 坐标在第一张图坐标系下）
- Tracking features $T_i \in \mathbb{R}^{C \times H \times W}$
- Aleatoric uncertainty $\Sigma_i^D, \Sigma_i^P \in \mathbb{R}_+^{H \times W}$

特别注意，作者从 backbone 的第 4、11、17、23 个 block 都抽 tokens 给 DPT，类似 U-Net 的 multi-scale skip connection。

### Tracking head (CoTracker2)

直接复用 CoTracker2 (https://github.com/facebookresearch/co-tracker) 架构。输入是 DPT head 输出的 tracking features $T_i$，query 点在 query 图上 bilinear 采样 feature，与其他图做 correlation，再过 self-attention 输出对应点。

关键：这个 tracker 不假设 temporal order，可以处理任意图像集合，不只是视频序列。

---

## Over-complete prediction 的妙处

Point map $P_i$、depth map $D_i$、camera $\mathbf{g}_i$ 三者数学上不独立。知道 depth 和 camera 就能反投影得到 point map，知道 point map 就能 PnP 得到 camera。

但是 VGGT 故意三个一起预测，而且三个都加 supervision。Table 6 的 ablation：

| L_cam | L_depth | L_track | Overall↓ |
|---|---|---|---|
| × | √ | √ | 0.834 |
| √ | × | √ | 0.727 |
| √ | √ | × | 0.790 |
| √ | √ | √ | **0.709** |

去掉哪个都变差。直觉：这相当于在 loss 里加入 implicit consistency constraint，强迫三个 head 学到 geometrically consistent 的表示。

更骚的是，推理时作者发现：用 depth head 和 camera head 反投影得到的 point map，比直接用 point map head 输出还准（Table 3，0.677 vs 0.709）。

intuition：分解复杂任务为简单子任务，每个子任务更容易学。Joint training 享受多任务 regularize，但 inference 时用 composition 而非 direct head，避开 direct head 的 difficulty。这是一种 "train heavy, infer smart" 的策略。

---

## 训练 loss 细节

总 loss：

$$\mathcal{L} = \mathcal{L}_{\text{camera}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{pmap}} + \lambda \mathcal{L}_{\text{track}}$$

$\lambda = 0.05$，camera / depth / pmap 三者数值范围接近，不需要额外权重。

### Depth loss 带 aleatoric uncertainty

$$\mathcal{L}_{\text{depth}} = \sum_{i=1}^N \left[\|\Sigma_i^D \odot (\hat{D}_i - D_i)\| + \|\Sigma_i^D \odot (\nabla \hat{D}_i - \nabla D_i)\| - \alpha \log \Sigma_i^D\right]$$

变量解释：
- $\Sigma_i^D$：网络预测的 per-pixel uncertainty，正值
- $\odot$：channel-broadcast 逐元素乘
- $\nabla$：spatial gradient，即图像 x/y 方向差分
- $\alpha$：regularization 系数

直觉：网络在难预测的 pixel 上输出大 $\Sigma$，loss 自动 down-weight 那部分，相当于网络自己说 "这块我不确定"。$-\alpha \log \Sigma$ 项防止网络 trivially 把 $\Sigma$ 推到无穷大。Gradient term 让 depth edges 更 sharp，这是 monocular depth estimation 的标配。

Point map loss 形式对称，只是换成 $\Sigma_i^P$。

### Ground truth 归一化的小心思

所有 3D 量归一化到第一张图坐标系，然后计算 point map 中所有 3D 点到原点的平均欧氏距离 $s$，用 $s$ 归一化 translations, point maps, depth maps。

DUSt3R 的做法：对网络 output 也应用这个归一化。
VGGT 的做法：只对 GT 归一化，强迫网络自己学。

作者在 Discussion 里说，对 prediction 做归一化不仅不必要，反而引入训练不稳定。让网络从数据中学到 canonical scale，更 robust。

---

## 实验数据看几个最关键的

### Camera pose on CO3Dv2 + RealEstate10K

| Method | Re10K | CO3Dv2 | Time |
|---|---|---|---|
| DUSt3R + global alignment | 48.0 | 66.5 | ~7s |
| MASt3R + global alignment | 67.7 | 76.7 | ~7s |
| VGGSfM v2 + BA | 78.9 | 83.4 | ~10s |
| **VGGT (feed-forward)** | **85.3** | **88.2** | **~0.2s** |
| **VGGT + BA** | **93.5** | **91.8** | ~1.8s |

VGGT feed-forward 模式比所有需要 post-optimization 的方法都快且准。加 BA 后还能进一步提升，但只要 1.8s，而 VGGSfM v2 要 10s。因为 VGGT 直接输出高质量 point / depth maps 作为 BA 的初始化，跳过 triangulation 和 iterative refinement。

### DTU MVS

| Method | Known GT Cam | Overall↓ |
|---|---|---|
| GeoMVSNet | √ | 0.295 |
| DUSt3R | × | 1.741 |
| **VGGT** | **×** | **0.382** |

VGGT 在不知道 GT camera 的情况下接近知道 GT camera 的 GeoMVSNet，远超 DUSt3R。

### Runtime

100 帧只要 3.12 秒，21GB GPU memory。200 帧 8.75 秒，40GB。Flash Attention v3 加持。这个 scalability 已经能处理大部分实际场景。

---

## 为什么 feed-forward 能干过 iterative optimization

这个问题其实挺深刻的。Bundle Adjustment 是在给定 visual observations 下做 MAP estimation，理论上最优。为什么 VGGT 一个 forward pass 反而更准？

我的理解是：

1. **数据驱动的 prior**：BA 假设的是 Gaussian noise model，但真实图像的 noise model 远比这个复杂。VGGT 从百万级 3D 标注数据中学到了更准确的 "visual feature → 3D structure" 映射，这个 prior 比 BA 的 hand-crafted noise model 更强。

2. **Global reasoning vs local optimization**：BA 是 iterative refinement，容易陷入 local minimum，特别是 bad initialization 时。VGGT 一次 forward 就是 global reasoning，attention 机制让它能在所有 tokens 之间做全连接的信息融合，本质上更 robust。

3. **Feature learning 的力量**：传统 pipeline 用 hand-crafted features (SIFT) 或轻量 learned features (SuperPoint)，VGGT 直接用 DINOv2 这种 billion-scale 预训练的 features，信息量完全不同。

4. **Multi-task supervision**：同时监督 camera、depth、point map、track 四个任务，互相 regularize。BA 只优化 reprojection error 一个目标。

类比：AlphaGo 用 policy/value network 替代 MCTS 的大量 rollout；ChatGPT 用 next-token prediction 替代符号推理；VGGT 用 feed-forward transformer 替代 BA。都是 "用海量数据训练的大模型" 替代 "手工设计的优化算法" 的范式转移。

---

## 下游任务证明 features 很强

### Novel View Synthesis

在 GSO 数据集上 fine-tune VGGT for NVS，与 LVSM (https://arxiv.org/abs/2410.17242) 对比：

| Method | Known Input Cam | Train Data | PSNR↑ |
|---|---|---|---|
| LVSM | √ | full | 31.71 |
| VGGT-NVS | × | 20% | 30.41 |

VGGT 不知道 input camera，训练数据只有 LVSM 的 20%，依然接近 LVSM。说明 VGGT 的 features 包含了很强的 geometry 信息，能 transfer 到 NVS。

### Dynamic Point Tracking

用 VGGT backbone 替换 CoTracker2 的 backbone，在 TAP-Vid (https://tapvid.github.io/) 上 fine-tune：

| Method | RGB-S AJ↑ | RGB-S $\delta_{\text{avg}}^{\text{vis}}$↑ |
|---|---|---|
| CoTracker | 67.4 | 78.9 |
| **CoTracker + VGGT** | **72.1** | **84.0** |

VGGT 不是为动态场景训练的，但它的 features transfer 过去之后显著提升 CoTracker 性能。这验证了 "通用 backbone + specialized head" 的设计模式。

---

## 一些可能的延伸方向

1. **Video SfM with temporal encoding**：当前对 input 是 permutation equivariant（除第一帧），视频场景可以加 temporal positional encoding 引入 smoothness prior。

2. **Self-supervised pre-training**：当前依赖大量 GT 3D annotation。作者在 Discussion 提到 differentiable BA 可以作为无监督信号，但训练慢 4 倍。这是一个 promising direction。

3. **3D Gaussian Splatting output**：当前输出 point map。可以扩展输出 3D Gaussian primitives (位置 + covariance + color + opacity)，直接喂给 Gaussian Splatting renderer。这与 GS-LRM (https://arxiv.org/abs/2404.19702) 思路结合。

4. **Long video reconstruction**：100-200 帧已经 OK，千帧级别需要 sliding window + global consistency。借鉴 StreamPETR 那种 memory mechanism 可能可行。

5. **Robotics perception backbone**：VGGT 的 features 包含丰富 3D 信息，比 CLIP features 更适合 robotics。可以替代 CLIP 作为 VLA (Vision-Language-Action) model 的 visual encoder。

6. **Joint with diffusion models**：VGGT 给 geometry，diffusion 给 texture / generation。可以做 3D-aware image generation / editing。

7. **Multi-modal fusion**：Architecture 很容易扩展，patchify 阶段加 modality-specific tokenizer 即可融合 LiDAR / IMU / depth sensor。

8. **Real-time SLAM**：当前 100 帧 3 秒，已经接近 real-time。配合 streaming inference 和 memory mechanism，可能做成 SLAM 系统。

9. **Non-rigid reconstruction**：当前 limitation 是大幅 non-rigid deformation 失败。可以在 training data 里加更多 dynamic scene 数据，或者引入 deformation field 表示。

10. **Fisheye / panoramic support**：当前只支持 perspective camera。patchify 阶段换成 fisheye-aware tokenizer，或者用 spherical projection 处理 panoramic。

---

## 跟相关工作的关系网

- **DUSt3R / MASt3R**：pairwise predecessor，VGGT 是 N-wise 进化版，去掉 global alignment post-processing。
- **VGGSfM**：用 differentiable BA 做 end-to-end SfM，VGGT 选择不用 BA in training，而是用大量监督数据 implicit 学几何。
- **DINOv2**：VGGT 的 patchifier，提供 strong initialization。
- **CoTracker2**：VGGT 直接复用作为 tracking head。
- **DPT**：VGGT 的 dense prediction head。
- **Vision Transformer Registers**：VGGT 用 register tokens 处理 ViT artifact。
- **FlashAttention v3**：让 VGGT 能 scale 到 200 帧。
- **LRM / GS-LRM**：single-image-to-3D 的 transformer，VGGT 是更通用的 N-image-to-3D。
- **LVSM**：novel view synthesis 的大 transformer，VGGT fine-tune 后接近其性能。
- **Fast3R**：同期工作，也做 multi-view feed-forward reconstruction，VGGT 性能更高（CO3Dv2 AUC@30: 88.2 vs 82.5）。

---

## 一句话直觉

VGGT 告诉我们：3D vision 不需要 pipeline，不需要 optimization post-processing，不需要 specialized architecture。你需要的就是一个足够大的 transformer，加上 Alternating Attention 这个 minimal inductive bias，用海量 3D 标注数据训练。剩下的，网络自己会学到几何推理。

这跟 GPT 替代手工规则系统、CLIP 替代手工 feature engineering、AlphaGo 替代 MCTS rollout 是同一个故事。3D vision 终于也迎来了它的 "大模型时刻"。

参考链接汇总：
- VGGT 代码: https://github.com/facebookresearch/vggt
- VGGT demo: https://huggingface.co/spaces/facebook/vggt
- VGGT 项目页: https://vgg-t.github.io/
- DINOv2: https://github.com/facebookresearch/dinov2
- DUSt3R: https://github.com/naver/dust3r
- MASt3R: https://github.com/naver/mast3r
- VGGSfM: https://github.com/facebookresearch/vggsfm
- CoTracker: https://github.com/facebookresearch/co-tracker
- DPT: https://github.com/isl-org/DPT
- FlashAttention: https://github.com/Dao-AILab/flashAttention
- TAP-Vid benchmark: https://tapvid.github.io/
- ViT Registers paper: https://arxiv.org/abs/2309.16588
- LVSM: https://arxiv.org/abs/2410.17242

---

# VGGT: Visual Geometry Grounded Transformer 深度解析

## 核心思想与定位

VGGT 来自 Oxford VGG 和 Meta AI，由 Jianyuan Wang 等人于 2025 年提出。它的核心 insight 极其简洁：**用一个 1.2B 参数的标准 transformer，在单次 forward pass 中，从 1 张到数百张图像一次性预测所有 3D 属性**（camera intrinsics/extrinsics, depth maps, point maps, 3D point tracks），耗时不到 1 秒，性能甚至超过依赖 Bundle Adjustment 后处理的传统方法。

这是 3D vision 领域的一个 paradigm shift。传统 pipeline 是 SfM (COLMAP) 那种 "geometric optimization + learning components" 的拼装；DUSt3R/MASt3R 把 pairwise reconstruction 做成端到端但必须配 global alignment；VGGSfM 把 BA 变成 differentiable，但仍然慢。VGGT 直接把 geometry optimization "烧进" network weights 里，让一个 feed-forward 大模型接管一切。

参考链接:
- 论文 arXiv: https://arxiv.org/abs/2503.11651
- 项目主页: https://vgg-t.github.io/
- 代码: https://github.com/facebookresearch/vggt
- Hugging Face demo: https://huggingface.co/spaces/facebook/vggt

---

## 1. 问题形式化

给定 N 张 RGB 图像 $(I_i)_{i=1}^N$，其中 $I_i \in \mathbb{R}^{3 \times H \times W}$，VGGT 学一个映射：

$$f\left((I_i)_{i=1}^N\right) = (\mathbf{g}_i, D_i, P_i, T_i)_{i=1}^N \tag{1}$$

变量含义：
- $\mathbf{g}_i \in \mathbb{R}^9$：第 $i$ 张图的相机参数（intrinsics + extrinsics）
- $D_i \in \mathbb{R}^{H \times W}$：depth map
- $P_i \in \mathbb{R}^{3 \times H \times W}$：viewpoint-invariant point map
- $T_i \in \mathbb{R}^{C \times H \times W}$：用于 tracking 的 dense features

**相机参数化**：$\mathbf{g} = [\mathbf{q}, \mathbf{t}, \mathbf{f}]$
- $\mathbf{q} \in \mathbb{R}^4$：rotation quaternion（单位四元数表示旋转）
- $\mathbf{t} \in \mathbb{R}^3$：translation vector
- $\mathbf{f} \in \mathbb{R}^2$：field of view (FoV)
- 假设 principal point 在图像中心

**关键设计 - Viewpoint Invariance**：所有 $P_i(\mathbf{y})$ 都定义在第一张相机 $\mathbf{g}_1$ 的坐标系下（即 world reference frame），这与 DUSt3R 一致。第一张图的 extrinsics 强制设为单位变换：$\mathbf{q}_1 = [0,0,0,1]$，$\mathbf{t}_1 = [0,0,0]$。

**Over-complete predictions 的巧妙设计**：camera params、depth maps、point maps 在数学上不是独立的（PnP 能从 point map 反推 camera，depth + camera 能反推 point map）。但作者故意让网络同时预测所有这些 redundant 量，因为 multi-task supervision 互相 regularize，比单一任务效果更好（Table 6 验证）。

---

## 2. 架构核心：Alternating-Attention (AA) Transformer

这是论文最重要的 architectural insight。架构图（Figure 2）：

```
Input Images (N 张)
      ↓
   DINOv2 patchify → 每张图 K 个 tokens
      ↓
+ Camera tokens (每张图 1 个) + Register tokens (每张图 4 个)
      ↓
┌─────────────────────────────────────┐
│  24 层 Alternating Attention blocks:  │
│  ┌─────────────────────────────┐    │
│  │ Frame-wise Self-Attention   │    │ ← 每张图独立 attend
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Global Self-Attention       │    │ ← 跨所有图 attend
│  └─────────────────────────────┘    │
│                × 24                   │
└─────────────────────────────────────┘
      ↓
   ┌──────┴──────┐
   ↓             ↓
Camera head    DPT head
(4×SA + Linear) (depth, point map, tracking features)
```

### 为什么 Alternating Attention 工作？

**Frame-wise self-attention**：每个 frame 内部的 $K$ 个 tokens 互相 attend，相当于一个 ViT 在单张图上做 self-attention。它做的事情是：
- 在单帧内聚合 spatial context
- 把 camera token 和 register token "匹配" 到对应的 image tokens（这是为什么 camera token 后来变成 frame-specific 的关键）
- 相当于让每张图先 "独立消化" 自己的信息

**Global self-attention**：跨所有 $N \times K$ 个 tokens 一起 attend，让网络做跨视图的 triangulation reasoning。这是替代传统 MVS cost volume 的机制 - cost volume 本质上是在每对图之间做 correlation，global attention 是 soft 的、全连接的 cost volume。

**为什么交替而不是只用 global？** Table 5 的 ablation 给出答案：

| Architecture | Acc.↓ | Comp.↓ | Overall↓ |
|---|---|---|---|
| Cross-Attention | 1.287 | 0.835 | 1.061 |
| Global Self-Attention Only | 1.032 | 0.621 | 0.827 |
| **Alternating-Attention** | **0.901** | **0.518** | **0.709** |

intuition: 只用 global attention 时，每张图的 token 在层与层之间没有 "回炉重炼" 的机会，信息全部被 mixed up，失去了 frame-level 的 identity。Alternating 的 frame-wise 步骤相当于一个 "normalization" - 让每个 token 在被 global context 污染之前，先重新 ground 回自己的 frame。这有点像 LayerNorm 的作用，但是 structural 的。

**为什么不用 cross-attention？** Cross-attention 让每张图 attend 到其他所有图，参数量爆炸，且论文提到 "preliminary experiments consistently showed cross-attention underperforms self-attention"。我推测原因：cross-attention 容易让网络陷入 "找 correspondence" 的局部模式，而 self-attention 是 "我把我的 context 拿出来给大家看，大家一起 refine" 的更对称的形式，更易优化。

### Camera Token 与 Register Token 的不对称设计

这是 architecture 的另一精髓：

- 第一张图的 camera token $\mathbf{t}_1^{\mathbf{g}} := \bar{\mathbf{t}}^{\mathbf{g}}$ 和 register tokens $\mathbf{t}_1^R := \bar{\mathbf{t}}^R$ 是一组 learnable parameters
- 其他所有图 $\mathbf{t}_i^{\mathbf{g}} := \bar{\bar{\mathbf{t}}}^{\mathbf{g}}, \mathbf{t}_i^R := \bar{\bar{\mathbf{t}}}^R$（$i \in [2, N]$）共享另一组 learnable parameters

这打破 permutation equivariance，让网络知道哪张图是 reference。Output 的 3D 量都 anchored 在第一张图的坐标系上。

**Register tokens** 借鉴 Darcet et al. (Vision Transformers Need Registers, arXiv:2309.16588) 的发现：ViT 高层 feature map 上会出现 high-norm artifact tokens，这些 tokens 实际上被网络用作 "global information buffer"。VGGT 主动加 register tokens 给网络 "垃圾桶"，防止它污染 image tokens。Output 时 register tokens 直接丢弃。

### Backbone 配置

- 24 个 attention blocks（每个含 1 个 frame-wise + 1 个 global attention layer）
- Feature dim 1024，16 heads（ViT-L 规模）
- QKNorm + LayerScale（init=0.01）稳定训练
- DINOv2-Large 做 patchification（14×14 patches）
- 1.2B 总参数

**为什么用 DINOv2 patchify 而不是从头训 conv？** 作者在 Discussion 中提到：DINOv2 提供更稳定的早期训练，对 learning rate / momentum 等 hyperparameter 不敏感。这其实是用 SSL 预训练的 "geometry-aware features" 作为 warm start，让网络不用从头学 low-level vision。

---

## 3. 预测头设计

### Camera Head
4 个 self-attention layers + 1 个 linear layer，输入是 camera tokens $(\hat{\mathbf{t}}_i^{\mathbf{g}})_{i=1}^N$，输出 $\hat{\mathbf{g}}^i \in \mathbb{R}^9$。Lightweight，仅占 backbone 5% runtime 和 2% GPU memory。

### Dense Prediction Head (DPT)
DPT (Dense Prediction Transformer, Ranftl et al. ICCV 2021) 把 transformer tokens 上采样到 full resolution feature maps $F_i \in \mathbb{R}^{C'' \times H \times W}$，然后用 3×3 conv 输出：
- Depth map $D_i$
- Point map $P_i$
- Tracking features $T_i \in \mathbb{R}^{C \times H \times W}$
- Aleatoric uncertainty $\Sigma_i^D, \Sigma_i^P \in \mathbb{R}_+^{H \times W}$

特别地，作者从第 4、11、17、23 个 block 都抽 tokens 给 DPT 做 multi-scale fusion（类似 U-Net skip connection 的思路）。

### Tracking Head (CoTracker2)
基于 CoTracker2 (Karaev et al. ECCV 2024)：
1. 在 query 图 $I_q$ 上 bilinear 采样 query point $\mathbf{y}_j$ 处的 feature
2. 与其他图 $T_i$ 做 correlation 得到 correlation maps
3. 通过 self-attention 处理 correlation maps 输出对应 2D 点 $\hat{\mathbf{y}}_{j,i}$

注意：tracking head **不假设 temporal order**，可以处理任意图像集合，不仅仅是视频。这使得 VGGT 能处理 unordered photo collection 的 tracking。

---

## 4. 训练目标 - 多任务 loss 详解

总损失：

$$\mathcal{L} = \mathcal{L}_{\text{camera}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{pmap}} + \lambda \mathcal{L}_{\text{track}} \tag{2}$$

其中 $\lambda = 0.05$。作者发现 camera/depth/pmap loss 数值范围相近，不需要额外权重。

### Camera Loss

$$\mathcal{L}_{\text{camera}} = \sum_{i=1}^N \|\hat{\mathbf{g}}_i - \mathbf{g}_i\|_\epsilon$$

$\|\cdot\|_\epsilon$ 是 Huber loss（smooth L1），对 outlier 鲁棒。直接回归 9D 参数向量，包括 quaternion（虽然不是单位约束，但实测 OK）。

### Depth Loss (with aleatoric uncertainty)

$$\mathcal{L}_{\text{depth}} = \sum_{i=1}^N \left[\|\Sigma_i^D \odot (\hat{D}_i - D_i)\| + \|\Sigma_i^D \odot (\nabla \hat{D}_i - \nabla D_i)\| - \alpha \log \Sigma_i^D\right]$$

变量解释：
- $\Sigma_i^D \in \mathbb{R}_+^{H \times W}$：网络预测的 per-pixel uncertainty
- $\odot$：channel-broadcast element-wise product
- $\nabla$：spatial gradient operator（图像 x/y 方向的差分）
- $\alpha$：log-uncertainty regularization 系数
- $-\alpha \log \Sigma_i^D$ 项防止网络把 $\Sigma_i^D$ 推向无穷大来 trivially minimize 前两项

这是 Kendall & Gal (NeurIPS 2017) 的 aleatoric uncertainty loss。直觉：网络在难预测的 pixel 上输出大 $\Sigma$，loss 自动 down-weight 那部分。Gradient term 是 monocular depth estimation 的常见技巧，让 depth edges 更 sharp。

### Point Map Loss

形式与 depth loss 完全对称，只是换成 point map uncertainty $\Sigma_i^P$。

### Tracking Loss

$$\mathcal{L}_{\text{track}} = \sum_{j=1}^M \sum_{i=1}^N \|\mathbf{y}_{j,i} - \hat{\mathbf{y}}_{j,i}\|$$

$\mathbf{y}_{j,i}$ 是 query point $\mathbf{y}_j$ 在图 $I_i$ 中的 ground truth 对应点。还加了 visibility binary cross-entropy loss。

### Ground Truth 坐标归一化

这是一个关键 trick，区别于 DUSt3R：
1. 表达所有量在 $\mathbf{g}_1$ 坐标系下
2. 计算 point map $P$ 中所有 3D 点到原点的平均欧氏距离 $s$
3. 用 $s$ 归一化 $\mathbf{t}$、$P$、$D$

**DUSt3R 的做法**：对网络 output 也应用归一化。
**VGGT 的做法**：只对 ground truth 归一化，**强迫网络自己学习** 这个 normalization scale。

作者的 insight：对 prediction 做归一化既不必要也无益，反而引入训练不稳定。让网络从数据中学到 canonical scale，是一种 "implicit regularization"。

### Training 配置

- AdamW optimizer，160K iterations
- Cosine LR scheduler，peak LR = 2e-4，warmup 8K iter
- 每个 batch 随机采 2-24 帧某场景，总 batch 含 48 frames
- 图像 resize 到 max dim = 518
- Color jitter + Gaussian blur + grayscale augmentation，per-frame 独立
- 64× A100 GPU，9 天
- bfloat16 + gradient checkpointing
- Gradient clip threshold = 1.0

### 训练数据

17 个数据集，覆盖 indoor/outdoor, synthetic/real：
Co3Dv2, BlendMVS, DL3DV, MegaDepth, Kubric, WildRGB, ScanNet, Hyper-Sim, Mapillary, Habitat, Replica, MVS-Synth, PointOdyssey, Virtual KITTI, Aria Synthetic, Aria Digital Twin, Objaverse-like synthetic

---

## 5. 关键实验结果

### 5.1 Camera Pose Estimation (Table 1)

在 CO3Dv2 和 RealEstate10K 上，每场景随机 10 帧评估 AUC@30：

| Method | Re10K AUC@30↑ | CO3Dv2 AUC@30↑ | Time |
|---|---|---|---|
| COLMAP+SPSG | 45.2 | 25.3 | ~15s |
| DUSt3R | 48.0 | 66.5 | ~7s |
| MASt3R | 67.7 | 76.7 | ~7s |
| VGGSfM v2 | 78.9 | 83.4 | ~10s |
| Fast3R (concurrent) | 72.7 | 82.5 | ~0.2s |
| **VGGT (feed-forward)** | **85.3** | **88.2** | **~0.2s** |
| **VGGT + BA** | **93.5** | **91.8** | ~1.8s |

VGGT feed-forward 模式比所有需要 post-optimization 的方法都快，且准确度更高。加 BA 后还能进一步提升，但只有 1.8s（VGGSfM v2 要 10s）。原因是 VGGT 直接输出高质量的 point/depth maps，可以直接作为 BA 的初始化，跳过了 triangulation 和 iterative refinement。

### 5.2 Dense MVS on DTU (Table 2)

VGGT 是除 DUSt3R 外唯一不知道 GT camera 的方法：

| Method | Known GT Cam | Acc.↓ | Comp.↓ | Overall↓ |
|---|---|---|---|---|
| GeoMVSNet | √ | 0.331 | 0.259 | 0.295 |
| DUSt3R | × | 2.677 | 0.805 | 1.741 |
| **VGGT** | **×** | **0.389** | **0.374** | **0.382** |

VGGT 在没有 GT camera 的情况下接近知道 GT camera 的 GeoMVSNet 性能，比 DUSt3R 好非常多。

### 5.3 Point Map Estimation on ETH3D (Table 3)

| Method | Acc.↓ | Comp.↓ | Overall↓ | Time |
|---|---|---|---|---|
| DUSt3R | 1.167 | 0.842 | 1.005 | ~7s |
| MASt3R | 0.968 | 0.684 | 0.826 | ~9s |
| VGGT (Point head) | 0.901 | 0.518 | 0.709 | ~0.2s |
| **VGGT (Depth + Cam)** | **0.873** | **0.482** | **0.677** | ~0.2s |

**关键发现**：直接用 point map head 不如 "用 depth head + camera head 反投影" 准确。直觉：分解复杂任务为子任务，每个子任务更易学。Joint training 让 depth 和 camera head 互相 regularize，但推理时用 composition 而非 direct head，享受 multi-task benefit 的同时避免直接 head 的 difficulty。

### 5.4 Two-View Matching on ScanNet (Table 4)

VGGT 的 tracking head 并非为两视图匹配设计，但仍然 SOTA：

| Method | AUC@5↑ | AUC@10↑ | AUC@20↑ |
|---|---|---|---|
| SuperGlue | 16.2 | 33.8 | 51.8 |
| LoFTR | 22.1 | 40.8 | 57.6 |
| Roma | 31.8 | 53.4 | 70.9 |
| **VGGT** | **33.9** | **55.2** | **73.4** |

### 5.5 IMC Phototourism (Table 10)

| Method | Test Opt | AUC@3° | AUC@10° | Runtime |
|---|---|---|---|---|
| VGGSfMv2 | √ | 59.32 | 76.82 | ~10s |
| DUSt3R | √ | 13.46 | 35.62 | ~7s |
| MASt3R | √ | 30.25 | 57.42 | ~9s |
| VGGT | × | 39.23 | 71.26 | 0.2s |
| **VGGT + BA** | √ | **66.37** | **84.91** | **1.8s** |

VGGT + BA 在 CVPR'24 IMC Challenge camera pose estimation 上达到 SOTA。

### 5.6 多任务 Ablation (Table 6)

| W/ L_camera | W/ L_depth | W/ L_track | Overall↓ |
|---|---|---|---|
| × | √ | √ | 0.834 |
| √ | × | √ | 0.727 |
| √ | √ | × | 0.790 |
| **√** | **√** | **√** | **0.709** |

Camera loss 对 point map accuracy 贡献最大（去掉后 0.709→0.834），depth loss 贡献较小，track loss 中等。说明让网络显式预测 camera 是关键，因为它 forced 网络理解 multi-view geometry 的 reference frame 关系。

### 5.7 下游任务：Novel View Synthesis (Table 7)

在 GSO 数据集上 fine-tune VGGT for NVS，与 LVSM 对比：

| Method | Known Input Cam | Train Data | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|
| GS-LRM | √ | full | 29.59 | 0.944 | 0.051 |
| LVSM | √ | full | 31.71 | 0.957 | 0.027 |
| VGGT-NVS* | × | 20% | 30.41 | 0.949 | 0.033 |

VGGT 不知道 input camera 参数，且训练数据只有 LVSM 的 20%，仍接近 LVSM 性能。

### 5.8 下游任务：Dynamic Point Tracking (Table 8)

用 VGGT 预训练 backbone 替换 CoTracker 的 backbone，在 TAP-Vid 上 fine-tune：

| Method | Kinetics AJ | RGB-S AJ | DAVIS AJ |
|---|---|---|---|
| CoTracker | 49.6 | 67.4 | 61.8 |
| **CoTracker + VGGT** | **57.2** | **72.1** | **64.7** |

$\delta_{\text{avg}}^{\text{vis}}$ 在 RGB-S 上从 78.9 提升到 84.0。即便 VGGT 不是为 dynamic scene 训练的，它的 features 仍能 transfer 到动态任务。

### 5.9 Runtime/Memory (Table 9)

| Frames | 1 | 2 | 4 | 8 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| Time (s) | 0.04 | 0.05 | 0.07 | 0.11 | 0.14 | 0.31 | 1.04 | 3.12 | 8.75 |
| Mem (GB) | 1.88 | 2.07 | 2.45 | 3.23 | 3.63 | 5.58 | 11.41 | 21.15 | 40.63 |

H100 + Flash Attention v3，336×518 分辨率。100 帧只要 3 秒 21GB。

---

## 6. 一些 Intuition 与思考

### 6.1 为什么 feed-forward 大模型能替代 iterative optimization？

传统 SfM 的 BA 是一个 non-linear least squares，通过反复 triangulation + re-projection error minimization 找到 MAP 估计。VGGT 的 strategy 是：用海量带 GT 3D annotation 的数据，让 transformer 在 forward pass 中**记住** "看到这些 visual features + 这些 camera configurations，对应的 3D 结构应该长这样"。这是从 algorithm 到 pattern matching 的转变。

类比：AlphaGo 用 policy/value network 替代 MCTS 的大量 rollout；ChatGPT 用 next-token prediction 替代符号推理。VGGT 是 3D vision 的同种范式转移。

### 6.2 Over-complete prediction 的 regularization 效果

Point map $P_i$、depth $D_i$、camera $\mathbf{g}_i$ 之间有 closed-form 关系：$P_i(\mathbf{y}) = \pi^{-1}(\mathbf{y}, D_i(\mathbf{y}), \mathbf{g}_i)$。同时监督这三个相当于在 loss 中加入了一个 implicit consistency constraint。如果网络预测的 depth 和 camera 反投影出来的 point 与 point head 输出不一致，loss 都会惩罚。这强迫三个 head 学到 geometrically consistent 的表示。

### 6.3 Alternating Attention 的计算效率

Global self-attention 在 $N \times K$ tokens 上是 $O((NK)^2)$ 复杂度。Frame-wise 是 $O(N \cdot K^2)$。对于 $K \approx 400$ patches, $N=100$ images：
- 纯 global: $(40000)^2 = 1.6 \times 10^9$ per layer
- 交替: $100 \cdot 400^2 + 40000^2 = 1.76 \times 10^9$ per layer

复杂度其实差不多，但 frame-wise 部分更容易 parallelize，且 memory 友好（不需要存全 attention matrix）。Flash Attention 进一步优化。

### 6.4 与 Fast3R 的对比

Fast3R (Yang et al. arXiv:2501.13928) 是同期工作，也用 single forward pass 处理多图。VGGT 在 CO3Dv2 AUC@30 上 88.2 vs Fast3R 82.5。差异可能源于：
- VGGT 有 tracking head，多任务 supervision
- VGGT 用 DINOv2 patchify（Fast3R 用 conv）
- VGGT 的 Alternating Attention（Fast3R 用纯 global）

### 6.5 Limitations

论文承认：
- 不支持 fisheye / panoramic 图像
- 极端 input rotations 性能下降
- 大幅 non-rigid deformation 失败

但这些 limitation 可以通过 fine-tune 目标数据集轻松解决，这是 feed-forward 模型相比 test-time optimization 的优势。

---

## 7. 与其他工作的脉络

VGGT 处于以下研究线的交汇点：

1. **大模型替代 pipeline**：CLIP, DINOv2, GPT 等证明 "大数据 + 大模型 + 简单架构" 能超越 task-specific 设计。VGGT 把这个理念搬到 3D vision。

2. **DUSt3R/MASt3R 的进化**：DUSt3R (CVPR 2024, https://dust3r.europe.naverlabs.com/) 引入 point map 作为 over-parameterized 3D 表示；MASt3R 加了 matching head。VGGT 把 pairwise 扩展到 N-wise，去掉了 global alignment post-processing。

3. **VGGSfM 的 differentiable BA**：VGGSfM (CVPR 2024, https://vggsfm.github.io/) 让 BA 可微，端到端训练。VGGT 选择直接不使用 BA in training，而是用大量监督数据让网络 implicitly 学到几何推理。

4. **LRM/GS-LRM 的 single-image-to-3D**：LRM (ICLR 2024) 用 transformer 从单图重建 3D。VGGT 是更通用的 N-image-to-3D。

5. **CoTracker 的 tracking**：VGGT 直接 reuse CoTracker2 architecture 作为 tracking head，验证了 "通用 backbone + specialized head" 的设计模式。

6. **Vision Transformer 的 register tokens**：Darcet et al. 发现 ViT 的高层 artifact tokens，VGGT 主动用 register tokens 处理这个问题。

7. **Plücker rays for view synthesis**：在 NVS 下游任务中，VGGT 用 Plücker rays 编码 target viewpoint，这是 LVSM (Jin et al. arXiv:2410.17242) 的方法。

---

## 8. 一些可能的拓展思考

1. **Video SfM with temporal prior**：VGGT 现在对 input 是 permutation equivariant（除第一帧），对于视频可以加 positional encoding 引入 temporal smoothness prior。

2. **Self-supervised pre-training**：当前依赖大量 GT 3D annotation。可以用 differentiable BA 或 photometric loss 做 self-supervision，作者在 Discussion 中提到这是 promising direction。

3. **Multi-modal extension**：camera + LiDAR / IMU fusion。Architecture 很容易扩展，只要在 patchify 阶段加 modality-specific tokenizer。

4. **Dense 3D Gaussians as output**：当前输出 point map。可以扩展输出 3D Gaussian primitives (位置 + covariance + color)，直接用于 Gaussian Splatting 渲染。这与 GS-LRM 思路相似。

5. **Long video reconstruction**：100-200 帧已经 OK，但千帧级别视频需要 sliding window + global consistency 维护。可以借鉴 StreamPETR 那种 memory mechanism。

6. **Geometry-aware pre-training for robotics**：VGGT 的 features 可以作为机器人 perception backbone，比 CLIP features 包含更多 3D 信息。

7. **Joint with generative models**：VGGT 给出 geometry, generative model 给出 texture / novel views。可以做 3D-aware image generation / editing。

---

## 9. 总结

VGGT 的核心贡献是把 3D vision 从 "modular pipeline + optimization" 推进到 "single large feed-forward model"。它的成功验证了几个关键假设：
- 标准 transformer 架构（带 Alternating Attention 这个 minimal inductive bias）足够表达 3D reasoning
- Multi-task joint prediction 优于 single-task specialization
- Over-complete prediction 的 implicit regularization 比 explicit constraint 更有效
- Feed-forward 可以达到甚至超过 iterative optimization 的精度

这是 3D vision 的 "GPT moment" - 一个统一的大模型在多个任务上达到 SOTA，且作为 feature backbone 在下游任务上展现强 transfer 能力。代码和模型已开源，预计会催生大量 follow-up 工作。

参考资源：
- 官方代码: https://github.com/facebookresearch/vggt
- 在线 demo: https://huggingface.co/spaces/facebook/vggt
- Project page: https://vgg-t.github.io/
- DINOv2: https://github.com/facebookresearch/dinov2
- DUSt3R: https://github.com/naver/dust3r
- MASt3R: https://github.com/naver/mast3r
- VGGSfM: https://github.com/facebookresearch/vggsfm
- CoTracker: https://github.com/facebookresearch/co-tracker
- DPT: https://github.com/isl-org/DPT
- FlashAttention v3: https://github.com/Dao-AILab/flashAttention
