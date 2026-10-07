---
source_pdf: GenFusion.pdf
paper_sha256: 18c2966ac7aac471aa7bf72370dd71b9bb534ea90a3284bbb7ea0fa9209a3ff6
processed_at: '2026-08-04T14:28:46-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GenFusion 的人话版

Andrej，好，我换个方式讲。假设我们坐在咖啡馆，你在白板上画图，我在旁边问问题，我们一步步把这个 paper 啃下来。

---

## 故事的起点：一个尴尬的 gap

先说为什么需要这个工作。

3D 这个领域有两个大方向：

**Reconstruction 派**（NeRF、3DGS）：你拿相机绕着物体拍一圈，算法从多视角图像反推 3D 场景。拍得越多越准，但问题是——你总得拍得到。没拍到的角落，算法只能瞎猜，猜出来的东西叫 artifact：floaters（飘在空中的噪点）、needle-like Gaussians（刺猬一样的尖刺）、black holes（直接空洞）。

**Generation 派**（DreamFusion、ZeroNVS）：你给一张图甚至一段文字，模型凭空生成 3D 内容。不需要多视角拍摄，但问题是——生成的东西几何上经常不对，细节糊，一致性差。

这两派中间有个 gap：

- Reconstruction 要 dense view，给少了就崩
- Generation 给一张图就行，但质量离 dense reconstruction 差很远

**GenFusion 想做的事情**：能不能让这两派互相帮忙？reconstruction 拍不到的地方，让 generation 来补；generation 生成的内容，反过来又帮 reconstruction 把空缺填上。

听起来简单，但怎么具体实现？

---

## 核心比喻：一个会修图的助手

想象你是一个 3D 建模师，正在用 2DGS 重建一个场景。你拍了 3 张图，重建出来的 3D 模型在某些角度看还行，换个角度一看——全是破洞和噪点。

这时候你叫来一个助手，这个助手特别擅长"看懂破图然后脑补出干净的图"。你把你渲染出来的破图给他，他还你一张干净的图。你拿这张干净图当新的训练数据，继续优化你的 3D 模型。模型变好了，渲染出来的破图没那么破了，助手再修一次，又更好了……

这就是 GenFusion 的 cyclic fusion——一个 **reconstruction 和 generation 互相喂饭** 的闭环。

那个"助手"就是一个 video diffusion model，基于 DynamiCrafter [1] 改的。为什么用 video 不用 image？因为你要修的是一串连续视角的图，image diffusion 修出来的每张图之间不一致，拼起来会闪烁。video diffusion 天然保证 temporal consistency。

参考：https://doubtiu.github.io/DynamiCrafter.github.io/

---

## 第一个聪明点：怎么训练这个助手？

这是 paper 里我觉得最优雅的设计。

问题：你想训练一个 diffusion model，输入是"有 artifact 的渲染图"，输出是"干净图"。那你需要大量这样的 (artifact, clean) pair。怎么造？

**朴素方案**：拿一个 dense view 的场景，用一部分 view 做 reconstruction，剩下的 view 做 test，渲染出 artifact 图，和 ground truth 配对。

问题在于：dense view 场景你 reconstruction 时本来视角覆盖就够，test view 离 train view 很近，渲染出来根本没多少 artifact。模型学不到"真正破的图怎么修"。

**GenFusion 的方案：masked reconstruction**

把每张输入图切成 4 块（左上、右上、左下、右下，每块 H/2 × W/2）。对每个场景，只保留其中一块，**mask 掉 75% 的像素**。然后用 2DGS 重建。

直觉上这相当于什么？相当于你的相机突然变成了一个 narrow FoV 的长焦镜头，只能看到画面的一小块。你绕着场景走一圈，但每次只看到四分之一的画面。这种极端 sparse 的角度覆盖，reconstruction 出来必然全是破洞和 artifact。

然后你沿原始完整轨迹（full FoV）重新渲染，那些被 mask 掉的区域自然就是黑的、破的、有 floaters 的。这正好 mimics 了真实 far-field rendering 的 artifact pattern。

**关键细节**：mask 是 per-scene 固定的，不是 per-view 随机的。如果 per-view 随机 mask，就变成 dropout 了，reconstruction 还是能凑合 fit，不会产生 systematic artifact。per-scene 固定 mask 才能逼出"真正欠观测"的 artifact。

这个 trick 一石二鸟：
1. 生成的训练数据里 artifact pattern 和真实 inference 时的 artifact 一致
2. 被 mask 的区域是黑的，强迫 diffusion 学会 outpainting（往外画），而不只是 inpainting（往里补）

训练数据用了 DL3DV-10K [2]，10510 个真实场景视频，规模够大。每个场景跑 7000 步 2DGS 重建，渲染 960×540 分辨率的 RGB-D video pair。

参考：https://dl3dv-10k.github.io/DL3DV-10K.github.io/

---

## 第二个聪明点：给 diffusion 喂 depth

光修 RGB 不够，你修出来的图几何上得对。GenFusion 把 RGB VAE 换成了 RGB-D VAE（来自 LDM3D [3]）。

这个改动看似简单，效果显著。Table 1 的消融：

| Config | FID↓ |
|--------|------|
| RGB VAE, 16 frames, 512×320 | 26.16 |
| RGB-D VAE, 16 frames, 512×320 | 25.40 |
| RGB-D VAE, 48 frames, 512×320 | 29.35（反而变差） |
| RGB-D VAE, 16 frames, 960×512 | **22.55** |

RGB-D 比 RGB 好，即使 backbone 原本是在 RGB latent 上预训练的。为什么？因为 depth 给了 diffusion 一个 explicit geometry anchor，它不需要从 RGB 反推 3D 结构，condition → output 的 mapping 变简单了。

48 frames 反而变差，这个有点反直觉。我猜原因是：长序列 temporal consistency 更难学，训练数据有限，长序列容易 overfit 或者 inconsistency 累积。16 frames 是个 sweet spot。

高分辨率提升大（26→22 FID），因为 artifact（比如 needle-like Gaussian）是 high-frequency 的，低分辨率下细节糊掉了，模型学不到精细修复。

参考：https://github.com/IntelLabs/theia-iso

---

## 第三个聪明点：cyclic fusion 怎么转起来

现在助手训练好了，怎么用它来优化一个新的、没见过的场景？

```
循环开始：
  1. 用当前的 2DGS 模型，沿新轨迹渲染 RGB-D video（可能有 artifact）
  2. 喂给冻结的 video diffusion，得到干净的 RGB-D video
  3. 把这些干净图加进训练集
  4. 继续优化 2DGS（用原始 input view + 生成的 view 一起监督）
  5. 回到 1
```

这是一个 positive feedback loop：2DGS 越好 → diffusion 输入的 artifact 越少 → 生成质量越高 → 监督信号越准 → 2DGS 更好。

但这里有个工程问题：**轨迹怎么采？**

paper 说这是最 critical 的 component。两类轨迹交替用：

- **Interpolation 轨迹**：在相邻 input view 之间插值，修小 artifact
- **Spiral 轨迹**：绕所有 input 相机做一个螺旋路径，模拟 far-field，覆盖欠观测角度

纯 interpolation 只能修小毛病；纯 spiral 离 input 太远，diffusion 容易 hallucinate。两者交替提供"先近后远、渐进扩展"的信号。

---

## 第四个聪明点：warm-up 和 annealing

不能一上来就让 diffusion 监督 2DGS。因为初始 2DGS 渲染出来全是黑的和噪点，diffusion 输入太烂，生成质量不可靠。如果拿这种生成图当监督，会带偏优化。

所以前 1000 步只做 warm-up：纯用 input view 优化 2DGS，让它先建立一个 reasonable 的 initial geometry。

1000 步之后开始引入 generation loss，但权重 λ 不是突变的，用 sin 曲线 annealing：

$$\lambda(k) = 1.0 \cdot \sin\left(\frac{k - K_{start}}{K_{end} - K_{start}} \cdot \pi\right)$$

- $k$：当前 iteration
- $K_{start}$：generation loss 开始的 iteration
- $K_{end}$：结束的 iteration
- sin 曲线在 $[K_{start}, K_{end}]$ 内从 0 → 1 → 0

前半段 λ 逐渐增大，让 generation prior 逐步介入；后半段 λ 减小回 0，让 2DGS 最终在 input supervision 上 polish，避免 overfit 到 generative hallucination。

这个 schedule 的 intuition：generation prior 是"拐杖"，帮你走过 artifact 最严重的阶段；最后阶段要把拐杖扔掉，让真实数据主导。

---

## 第五个聪明点：sparsity-aware densification

原版 3DGS 的 densification 机制：每 100 步看一次每个 Gaussian 的 view-space position gradient，超阈值就 clone/split，然后 reset gradient。

这在 dense view 下 work，因为每个 Gaussian 大部分时间都被观测到，gradient 累积稳定。

但 sparse view 下不行：某个 Gaussian 可能只在 5 步里被观测到（100 步里只有 5 步），gradient 累积极少，但它在那 5 步里 gradient 很大，应该被 densify。原版机制会漏掉它。

GenFusion 的修改：
1. **Disable gradient reset**：保留全历史 gradient
2. **加 visibility count threshold**：gradient 超阈值 **且** visibility count > minimal 才 densify

这个改动单独贡献 +0.47 PSNR（supp Table 1，15.34 → 15.81）。看起来不大，但在 artifact 修复这种 marginal gain 很难挤的任务里，已经很显著了。

---

## 第六个聪明点：Content Expansion

遇到大片黑区（完全没观测的区域），光靠 photometric loss 不够，因为周围没有 Gaussian 可以 clone。需要 **主动添加新点**。

判定公式：

$$T < \tau_T \quad \text{or} \quad |D - \hat{D}| > \tau_D$$

- $T$：cumulative opacity，$T$ 小说明这个像素基本没被 Gaussian 覆盖（黑区）
- $D$：2DGS 渲染的 depth
- $\hat{D}$：diffusion 生成的 depth（经过 scale-shift alignment）
- $\tau_T, \tau_D$：阈值

满足条件的像素，把 diffusion 生成的 RGB-D **back-project** 成 3D 点，直接加进 Gaussian 点云。相当于让 diffusion model "虚构" 3D content，然后在后续 iterations 中用 photometric loss refine 这些点的颜色、大小、opacity。

这里有个隐性挑战：monocular depth 是 metric-ambiguous 的（只有相对深度，没有绝对尺度），所以 $\hat{D}$ 需要和 $D$ 做 scale-and-shift alignment。paper 用 ScaleAndShiftInvariant loss [4]，在 unbounded scene 上这个 alignment 可能失败，是 limitation 之一。

参考：https://github.com/cleinc/bts

---

## 实验结果怎么说

### View Interpolation（Mip-NeRF360，稀疏视角重建）

Table 2 的关键数字：

| Method | 3-view | 6-view | 9-view |
|--------|--------|--------|--------|
| 3DGS | 13.06 | 14.96 | 16.79 |
| FSGS | 14.17 | 16.12 | 17.94 |
| ReconFusion | 15.50 | 16.93 | 18.19 |
| **GenFusion** | 15.29 | 17.16 | **18.36** |

9-view 上 GenFusion 达到 18.36，**首次让 3DGS-based 方法在 Mip-NeRF360 sparse view 上追平 SOTA NeRF**。这很重要，因为 3DGS 在 sparse view 下 notoriously 难训，容易 floaters。

3-view 上 GenFusion (15.29) 略输 ReconFusion (15.50)。我猜原因是：极端 sparse view 下 artifact 太严重，video diffusion 在这种 garbage input 上容易 hallucinate；ReconFusion 用 PixelNeRF-style feedforward prior 更稳定。

### View Extrapolation（DL3DV + TnT，masked 评估）

这是 GenFusion 真正拉开差距的地方：

| Dataset | 3DGS | 2DGS | FSGS | GenFusion |
|---------|------|------|------|-----------|
| DL3DV 1/2 fps | 17.22 | 16.56 | 18.25 | **20.47** |
| TnT 1/2 fps | 15.95 | 15.46 | 16.72 | **17.45** |

DL3DV 上 +2.2 PSNR over 次优，这个 gap 在 NVS 领域是 huge 的。说明 generative prior 在 far-field extrapolation 上确实是 game-changer。

### Ablation 里有趣的发现

Supp Table 1：

| Component | PSNR |
|-----------|------|
| 2DGS baseline | 13.87 |
| + train view monocular depth | 13.89（几乎无 gain） |
| + sample view RGB | 15.33（+1.46，最大贡献） |
| + sample view depth | 15.34（几乎无 gain） |
| + sparsity-aware densification | 15.81（+0.47） |

**有意思的是**：train view 上的 monocular depth 监督定量上几乎无效，但 supp Figure 1 显示它定性上能减少 floaters。说明 PSNR/LPIPS 对 floaters 这种 artifact 不敏感（floaters 面积小但视觉上很扎眼）。这是 NVS 评估的老问题——定量指标和 human perception 不完全对齐。

sample view RGB regularization 贡献最大（+1.46），因为这是从 generated view 引入全新 supervision，等于"无中生有"地扩展了训练集。

---

## 和 SDS 的对比

DreamFusion 的 SDS（Score Distillation Sampling）[5] 思路上有相似之处——都用 diffusion prior 监督 3D 优化。但有几个关键差异：

| | SDS | GenFusion |
|---|-----|-----------|
| Prior 来源 | 2D image diffusion | Video diffusion |
| 监督形式 | Score gradient（间接） | Direct pixel L2 loss（直接） |
| Temporal consistency | 无 | 有 |
| Geometry awareness | 无 | RGB-D VAE + depth conditioning |
| 优化稳定性 | 差（mode collapse 常见） | 好（L2 信号 clean） |
| Inference cost | 低（只算 score） | 高（full denoising pass） |

SDS 的 gradient 是 noise-conditional score，方差大，容易 collapse 到 mode-seeking。GenFusion 直接拿生成图做 L2 监督，gradient 信号 clean 稳定，但代价是每次 cyclic iteration 都要跑完整 DDIM 25 steps。

这是 stability vs speed 的 trade-off，paper 选了 stability。每 scene 总训练时间约 40 分钟，比 baseline 2DGS 慢 3-4x。

参考：https://dreamfusion3d.github.io/

---

## 我觉得最值得记住的三点

**1. Masked reconstruction 这个 trick 的双重身份**

它既是训练数据生成方法，又是 evaluation protocol。训练时用它造 (artifact, clean) pair，评估时用它模拟 far-field rendering。这种 train/eval consistency 是 method 设计的高水准——避免了 train/test distribution shift 导致的 metric 误导。

很多 paper 的 evaluation protocol 和训练 setup 不一致，导致 metric 好看但实际泛化差。GenFusion 这个设计从源头杜绝了这个问题。

**2. Video 作为 reconstruction 和 generation 的 bridge**

为什么是 video 而不是 image 或 3D？

- Image：缺 temporal consistency，拼起来闪烁
- 3D（比如直接在 Gaussian space 做 diffusion）：太难，没有预训练 prior 可用
- Video：有 temporal consistency，DynamiCrafter 已经有强大预训练 prior，只需 minimal adaptation

video 是当前 sweet spot——既有 2D diffusion 的成熟 prior，又能通过多帧隐式约束 3D consistency。

**3. Closed-loop 的哲学**

GenFusion 的 cyclic fusion 本质上是 EM-like 的 alternating optimization：
- E-step：用当前 3DGS 渲染，喂给 diffusion 生成 clean view（相当于 E-step 估计 latent）
- M-step：用生成的 clean view 监督 3DGS 优化（相当于 M-step 更新参数）

这种 closed-loop 在 ML 里很常见（k-means、EM、self-training），GenFusion 把它用到 3D vision 里，让 reconstruction 和 generation 互相 bootstrap。这个 pattern 可以推广到很多"两个互补模型互相 teaching"的场景。

---

## 局限和可能的改进

paper 自己承认：

1. **慢**：每 1000 iter 调一次 video diffusion，每 scene 40 分钟
2. **大 invisible region 模糊**：多个 video fragment 生成不一致时，cyclic fusion 会 average 它们导致 blur
3. **Depth alignment 误差**：monocular depth 在 unbounded scene 上 scale-shift alignment 可能失败

**我会想到的改进方向**：

- **Cross-fragment consistency loss**：cyclic fusion 里加一个约束，让不同 fragment 生成的 overlapping 区域 photometric 一致
- **Test-time distillation**：把 video diffusion 蒸馏成 3D-aware feedforward 模型，减少 inference cost
- **3DGS-native diffusion**：直接在 Gaussian parameter space 做 diffusion，避免 RGB-D latent 的 2D bias（这是更激进的方向，但没有预训练 prior 可用）
- **Coarse-to-fine CFG schedule**：早期高 CFG 强 prior 修大 artifact，后期低 CFG 保留 detail

---

## 一句话总结

GenFusion 让 generative prior 通过 video 这个媒介成为 reconstruction 的可微 supervisor，同时让 reconstruction 的 artifact 成为 generation 的 condition signal，两者形成 closed-loop 互相 bootstrap，弥合了 dense reconstruction 和 sparse generation 之间的 conditioning gap。

核心贡献不是某个单一技术，而是把 masked reconstruction、video diffusion、cyclic fusion 这三件事串起来形成 coherent pipeline 的系统工程。

**主要参考**：
- GenFusion project: https://genfusion.sibowu.com
- DynamiCrafter: https://doubtiu.github.io/DynamiCrafter.github.io/
- LDM3D: https://github.com/IntelLabs/theia-iso
- 2DGS: https://buaa-pal.github.io/2DGS/
- DL3DV-10K: https://dl3dv-10k.github.io/DL3DV-10K.github.io/
- ReconFusion: https://reconfusion.github.io/
- DreamFusion (SDS): https://dreamfusion3d.github.io/
- ViewCrafter (类似思路): https://viewcrafter.github.io/
- CAT3D (multi-view diffusion): https://cat3d.github.io/

---

# GenFusion 深度解析：Closing the Loop between Reconstruction and Generation

Andrej，这篇 paper 的核心 intuition 非常优雅：3D reconstruction 和 3D generation 两个领域存在 conditioning gap，前者依赖 dense view coverage，后者仅在 single/no view 下工作。GenFusion 通过 video diffusion 把这两端桥接起来，形成一个 cyclic feedback loop。让我从 motivation、架构、训练目标、cyclic fusion 四个层面展开，并补充一些 paper 没明说但很重要的细节。

---

## 1. Motivation：Viewpoint Saturation 的本质问题

paper 第 3 段已经点明：从 multi-view image 优化 NeRF/3DGS 本质上是 **ill-posed inverse problem**，无限多个 photo-consistent 解释都能 fit 输入图像。这导致：

- **Floaters**：close-camera regions 因为 sample 点多，gradient 大，容易"伪造"view-dependent effects
- **Background collapse**：unobserved regions 退化成 needle-like artifacts
- **Feedforward 方法 saturation**：PixelNeRF / MVSplat 类方法在 4-8 张图以上 saturate，因为 cost volume aggregation 架构上无法 scale

GenFusion 的关键洞察是：与其用 sparsity / smoothness 这种 hand-crafted regularizer 去约束 ill-posed 解空间，不如用 **learned generative prior** 直接提供 far-field 的 supervision signal，并且通过 video 的形式保证 temporal consistency。这个思路和 ReconFusion (CVPR 2024) [1] 一脉相承，但 ReconFusion 用 PixelNeRF-style diffusion 做 per-frame sample loss，缺乏 temporal coherence；GenFusion 改用 video diffusion 同时修复一组 frames，consistency 自然获得。

参考链接：
- ReconFusion: https://reconfusion.github.io/
- Nerfbusters (artifact 分析): https://github.com/ethanweber/nerfbusters
- ViewCrafter (类似思路，point cloud based): https://viewcrafter.github.io/

---

## 2. 核心架构：Reconstruction-driven Video Diffusion

### 2.1 Masked 3D Reconstruction —— 关键的数据生成 trick

这是 paper 里最聪明的设计之一。问题在于：怎么训练一个能修复 far-field artifact 的 diffusion model？直接用 dense view reconstruction 后再 render test view 作为 artifact 数据，问题是 sampled views 通常已经 fully cover 场景，target view 只在 interpolation 区间，模型学不到 extrapolation。

**Masked reconstruction 方案**：

1. 将每张输入图按 spatial 切成 4 个 non-overlapping patches: top-left / top-right / bottom-left / bottom-right (每块 H/2 × W/2)
2. 对每个 scene 随机选一个 patch (例如 top-left)，**mask 掉其余 75% pixels**
3. 用 2DGS [2] 对这个 patch-only 序列做 reconstruction
4. 然后沿原始完整轨迹 (full FoV) 重新 render，得到 artifact-prone RGB-D video
5. 这个 artifact-prone video 和原始 capture video 组成 training pair

**直觉**：masking 75% 等价于模拟一个 narrow FoV 相机，reconstruction 时角度覆盖极窄 → 渲染 full view 时必然在 mask 区域出现 holes / needle artifacts / floaters，这正好 mimics 真实 far-field rendering 的 artifact pattern。同时 mask 区域是 black 的，强迫 diffusion 学会 outpainting 而非仅 inpainting。

注意一个细节：mask 在 **per-scene 而非 per-view** 粒度上固定，强制 reconstruction 过程有 limited view coverage。这点很关键，如果 per-view 随机 mask 就退化成 dropout-style augmentation，无法制造 systematic artifact。

参考：
- 2DGS: https://buaa-pal.github.io/2DGS/
- DL3DV-10K dataset: https://github.com/DL3DV-10K/DL3DV-10K

### 2.2 Video Diffusion 架构

基于 DynamiCrafter [3] (image-to-animation LDM) 改造：

```
Input: artifact-prone RGB-D video (4×T×H×W)
       + reference image (CLIP embedding)
       
        ┌──────────────────────────────────────┐
        │  RGB-D VAE Encoder (from LDM3D [4])  │
        │  E: 4×T×H×W → 4×T×(H/8)×(W/8)        │
        └──────────────────────────────────────┘
                      │
                      ▼  concat with per-frame noise z_t
        ┌──────────────────────────────────────┐
        │  3D-UNet Denoiser ε_θ                │
        │  condition: t, CLIP(ref)             │
        │  + artifact latent as sequence cond  │
        └──────────────────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────────────────┐
        │  RGB-D VAE Decoder D                 │
        │  → clean RGB-D video                 │
        └──────────────────────────────────────┘
```

**关键改动**：

1. **RGB-D VAE 替换 RGB VAE**：使用 LDM3D 预训练的 VAE，将 4-channel (RGB+D) 编码到 latent。这个改动允许 geometry 信息进入 diffusion，且不破坏 DynamiCrafter 已学到的 RGB prior（因为 LDM3D VAE 在 RGB 上的 distribution 与原始 RGB VAE 相近）。

2. **Sequence conditioning**：artifact-prone RGB-D video 编码后 **concat 到 per-frame initial noise** 上 (channel-wise)，而非仅作为 cross-attention key/value。这样 rich visual detail 直接进入 denoiser 主路径。

3. **CLIP global conditioning**：从输入序列中 sample 一个 reference frame (训练时随机，推理时选 nearest input view 到 target trajectory)，过 CLIP image encoder 得到 global scene description。这提供 high-level 语义 guidance。

### 2.3 训练目标

公式 (3)：

$$\mathcal{L} = \mathbb{E}_{\mathcal{E}(x), c, \epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|_2^2 \right]$$

变量解释：
- $x$：ground truth RGB-D video
- $\mathcal{E}(x)$：VAE encoder 输出的 clean latent $z_0$
- $t \in \{1, \ldots, T\}$：diffusion timestep
- $\epsilon \sim \mathcal{N}(0, 1)$：标准 Gaussian noise
- $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$：forward process 加噪，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ 是 cumulative product of noise schedule
- $c$：conditioning tuple = (artifact-prone RGB-D latent, CLIP feature)
- $\epsilon_\theta$：可学习 denoising network (3D-UNet with temporal attention)

这是标准 DDPM objective，但 $c$ 中包含 reconstruction artifact，模型必须 learn "从 artifact 推断 clean content" 的 conditional distribution。

**Classifier-free guidance**：训练时 10% 概率 dropout conditioning image，推理时 DDIM 25 steps，CFG scale = 3.2。这个 scale 比较大，说明 artifact conditioning 信号较强需要平衡 generative prior。

### 2.4 VAE 设计消融 (Table 1)

| Config | FID↓ |
|--------|------|
| RGB VAE, 16 frames, 512×320 | 26.16 |
| RGB-D VAE, 16 frames, 512×320 | 25.40 |
| RGB-D VAE, 48 frames, 512×320 | 29.35 |
| RGB-D VAE, 16 frames, 960×512 | **22.55** |

**直觉解读**：
- RGB-D > RGB（即使 backbone 预训练在 RGB latent 上）：因为 depth 提供 explicit geometry constraint，让 diffusion 不需要从 RGB 推断 3D 结构，等于降低了 condition → output 的 mapping 复杂度。
- 48 frames 反而变差：temporal consistency 在长序列上 harder，且训练数据有限，长序列 overfit 风险大。
- 高分辨率显著提升：artifact pattern (如 needle-like Gaussian) 是 high-frequency 的，低分辨率下细节丢失导致模型学不到精细修复。

---

## 3. Cyclic Fusion：闭环优化

这是 paper 的另一个核心创新。把 video diffusion 和 3DGS 优化耦合起来：

### 3.1 整体流程

```
For iteration k = 1, 2, ..., N:
    if k < 1000:  # warm-up
        optimize 2DGS with L_recon only (input views)
    else if k % 1000 == 0:
        1. Sample novel trajectories (interpolation + spiral)
        2. Render RGB-D video from current 2DGS
        3. Feed to frozen video diffusion → clean RGB-D video
        4. Add to supervision set
        5. Sparsity-aware densification (if needed)
    Optimize 2DGS with L = L_recon + λ(k) * L_gen
```

公式 (2) 是 high-level objective：

$$\arg\min_\theta \mathbb{E} \left[ \left\| G_\phi(\hat{I}_{k+1} | R_\theta(\tilde{I}_k)) - R_\theta(\hat{I}_{k+1}) \right\|_2 \right]$$

- $k$：cyclic iteration index
- $\hat{I}_{k+1}$：next-iteration target view
- $R_\theta(\tilde{I}_k)$：current reconstruction rendering
- $G_\phi(\cdot | \cdot)$：frozen video diffusion model
- $R_\theta(\hat{I}_{k+1})$：3DGS 在新视角的渲染

直觉：让 3DGS rendering 逼近 diffusion 生成的 "假 GT"，而 diffusion 的输入又是 3DGS 当前 (artifact) 的状态。两者形成 positive feedback loop——3DGS 越好，diffusion input artifact 越少，生成质量越高；反过来 diffusion 生成越准，3DGS supervision 越强。

### 3.2 Trajectory Sampling

paper 强调这是 **most critical component**。两类轨迹：

1. **View interpolation**：在相邻 input views 之间插值，确保 input view 邻域的几何精度
2. **Spiral / spherical path**：across 所有 input camera poses，模拟 far-field rendering，覆盖 under-observed angles

这个组合很关键：纯 interpolation 只能修小 artifact，纯 spiral 距离 input 太远会导致 diffusion hallucinate。两者交替提供渐进的 "expand-and-refine" 信号。

### 3.3 Sparsity-aware Densification

原版 3DGS densification 依赖 view-space position gradient 的 running average，每 K=100 steps reset。在 sparse view 下 Gaussian visibility count 极低，gradient 累积不稳定，导致 densification 决策错误。

GenFusion 的修改：
- **Disable gradient reset**：保留全历史 gradient
- **Visibility count threshold**：只有当 gradient 超阈值 **且** visibility count > minimal 才加入 densification list
- Densification 频率：每 100 iterations

这个修改在 supp Table 1 中验证，sparsity-aware densification 单独贡献 +0.47 PSNR (15.34 → 15.81)，是 ablation 中第二大 gain。

### 3.4 Content Expansion：Unreliable Depth 判定

公式 (4)：

$$T < \tau_T \quad \text{or} \quad |D - \hat{D}| > \tau_D$$

- $T \in [0, 1]$：cumulative opacity (transmittance 的补数)，反映该像素是否被任何 Gaussian 覆盖
- $D$：当前 2DGS 渲染的 depth
- $\hat{D}$：video diffusion 生成 (并 alignment 后) 的 depth
- $\tau_T, \tau_D$：thresholds

满足任一条件的像素视为 unreliable，从 generated RGB-D **back-project** 出新 Gaussian 点加入场景。这相当于让 diffusion model "虚构" 3D content，然后用 photometric loss 在后续 iterations 中 refine 这些点的 attributes。

**Alignment**：$\hat{D}$ 需要 scale-and-shift align 到 $D$，因为 monocular depth 是 metric-ambiguous 的，paper 用 ScaleAndShiftInvariant loss [5]。

### 3.5 Loss Function

公式 (5)：

$$\mathscr{L} = \mathscr{L}_{recon} + \lambda \mathscr{L}_{gen}$$

其中：

$$\mathscr{L}_{recon} = \lambda_{l_1} \mathcal{L}_{l1} + \lambda_{SSIM} \mathcal{L}_{SSIM} + \lambda_{mono} \mathcal{L}_{mono}$$

- $\mathcal{L}_{l1}$：L1 pixel loss between rendering 和 input
- $\mathcal{L}_{SSIM}$：structural similarity loss
- $\mathcal{L}_{mono}$：scale-invariant depth loss [6] between rendered depth $\hat{D}$ 和 monocular depth $D$

$\mathcal{L}_{gen}$ 结构相同但 applied 到 generated views。

### 3.6 Warm-up Annealing

公式 (6)：

$$\lambda(k) = 1.0 \cdot \sin\left(\frac{k - K_{start}}{K_{end} - K_{start}} \cdot \pi\right)$$

- $k$：current iteration
- $K_{start}$：generation loss 开始的 iteration (paper 中是 1000)
- $K_{end}$：generation loss 结束的 iteration

**直觉**：sin 曲线在 $[K_{start}, K_{end}]$ 内从 0 → 1 → 0，避免 generation loss 突然引入导致 optimization 不稳定。前 1000 iter 只用 input view warm-up，让 2DGS 先建立一个 reasonable 的 initial geometry，否则 diffusion input 全是 black/noise，生成质量不可靠。后期 annealing 回 0，让 model 最终在 input supervision 上 polish，避免 overfit 到 generative hallucination。

---

## 4. 实验数据分析

### 4.1 View Interpolation (Table 2, Mip-NeRF360)

| Method | 3-view PSNR | 6-view | 9-view | Avg |
|--------|------------|--------|--------|-----|
| Zip-NeRF | 12.77 | 13.61 | 14.30 | 13.56 |
| ReconFusion | 15.50 | 16.93 | 18.19 | 16.87 |
| 3DGS | 13.06 | 14.96 | 16.79 | 14.94 |
| FSGS | 14.17 | 16.12 | 17.94 | 16.08 |
| **GenFusion** | **15.29** | **17.16** | **18.36** | **16.93** |

**重要观察**：GenFusion 在 9-view 上 18.36，**首次让 3DGS-based 方法在 Mip-NeRF360 sparse view 上达到 SOTA NeRF 水平**。这很关键因为 3DGS 通常在 sparse view 上比 NeRF 难训得多（容易 floaters）。

3-view GenFusion (15.29) 略低于 ReconFusion (15.50)，因为 ReconFusion 用 PixelNeRF-style feedforward prior 在 extreme sparse 下更稳定，而 GenFusion 的 video diffusion 在 artifact 极严重时可能 hallucinate。

### 4.2 View Extrapolation (Table 3, DL3DV + TnT)

这是 GenFusion 真正的强项，masked reconstruction 评估协议：

| Dataset | 3DGS | 2DGS | FSGS | GenFusion |
|---------|------|------|------|-----------|
| DL3DV 1/2 fps | 17.22 | 16.56 | 18.25 | **20.47** |
| TnT 1/2 fps | 15.95 | 15.46 | 16.72 | **17.45** |
| DL3DV 1/4 fps | 16.90 | 16.02 | 17.83 | **20.01** |
| TnT 1/4 fps | 14.75 | 14.38 | 16.04 | **16.29** |

**DL3DV 上 +2.2 PSNR** over FSGS（次优），这个 gap 在 NVS 领域是 huge 的。说明 generative prior 在 far-field extrapolation 上确实是 game-changer。

### 4.3 Ablation (Supp Table 1)

| Component | PSNR |
|-----------|------|
| 2DGS baseline | 13.87 |
| + train view monocular depth | 13.89 (几乎无 gain) |
| + sample view RGB | 15.33 (+1.46) |
| + sample view depth | 15.34 (+0.01) |
| + sparsity-aware densification | 15.81 (+0.47) |

**有趣发现**：
1. Train view 上的 monocular depth 监督几乎无效（仅 +0.02 PSNR），但 supp Figure 1 显示它能减少 floating artifacts——定量指标敏感度低，定性改善明显。
2. Sample view RGB regularization 贡献最大 (+1.46)，因为这是从 generated view 引入新 supervision。
3. Sample view depth 几乎无额外 gain，但定性上提升几何精度。
4. Sparsity-aware densification +0.47，是工程上不可忽视的 trick。

---

## 5. 局限与未解决问题

paper Discussion 部分承认：

1. **Inference 慢**：每 1000 iter 调一次 video diffusion (DDIM 25 steps)，每 scene 总训练时间约 40 分钟。比 baseline 2DGS 慢 3-4x。
2. **大 invisible region 模糊**：当多个 video fragment 生成内容不一致时，cyclic fusion 会 average 它们导致 blur。这是 video diffusion 3D consistency 的根本问题——每个 fragment 独立生成，跨 fragment 没有 explicit 3D constraint。
3. **Depth alignment 误差**：monocular depth 和 3DGS depth 的 scale-and-shift alignment 在 unbounded scene 上可能失败，导致 back-projected Gaussian 位置错误。

**我会想到的改进方向**：
- Multi-fragment consistency loss：在 cyclic fusion 中加 cross-fragment photometric constraint
- Test-time distillation：把 video diffusion 蒸馏成 3D-aware feedforward model，减少 inference cost
- 3DGS-native diffusion：直接在 Gaussian parameter space 做 diffusion，避免 RGB-D latent 的 2D bias
- Coarse-to-fine CFG schedule：早期高 CFG（强 prior 修复大 artifact），后期低 CFG（保留 detail）

参考类似工作：
- CAT3D (multi-view diffusion): https://cat3d.github.io/
- ViewCrafter (point cloud conditioning): https://viewcrafter.github.io/
- 3DGS-Enhancer (concurrent work): https://github.com/PKU-YuanGroup/3DGS-Enhancer

---

## 6. 总结：Intuition Building

GenFusion 的核心 insight 用一句话总结：**让 generative prior 通过 video 这个媒介成为 reconstruction 的可微 supervisor，同时让 reconstruction 的 artifact 成为 generation 的 condition signal**。这个双向闭环的本质是：

- Diffusion model 学到 $p(I_{clean} | I_{artifact})$ 的 conditional distribution
- 3DGS 优化目标变成 $\min_\theta \| R_\theta - G_\phi(R_\theta) \|$
- Gradient 通过 $R_\theta$ (differentiable renderer) 流回 Gaussian 参数，diffusion 部分 frozen

这种 "render-then-diffuse-then-supervise" 的范式在思想上和 SDS (Score Distillation Sampling) [7] 有相似之处，但有几个关键差异：

| | SDS | GenFusion |
|---|-----|-----------|
| Prior 来源 | 2D image diffusion | Video diffusion |
| Supervision 形式 | Score gradient | Direct pixel loss on generated video |
| Temporal consistency | 无 (per-frame) | 有 (video) |
| Geometry awareness | 无 | RGB-D VAE + depth conditioning |
| 优化稳定性 | 差 (mode collapse) | 好 (direct L2 supervision) |

SDS 的梯度是 noise-conditional score，方差大、容易 collapse；GenFusion 直接用生成图像做 L2 监督，gradient 信号 clean，但代价是 inference 时需要 full denoising pass。这是一个 trade-off，paper 选择 stability over speed。

**最值得记住的一点**：masked reconstruction 这个 trick 不仅用于数据生成，还复用为 evaluation protocol。这种"训练-评估一致性"是 method 设计的高水准，避免了 train/test distribution shift 导致的 metric 误导。

希望这个解析能帮你 build intuition。如果深入讨论某个具体模块（比如 video diffusion 的 condition mechanism 或 sparsity-aware densification 的实现），可以继续聊。

**主要 reference links**：
- GenFusion project: https://genfusion.sibowu.com
- DynamiCrafter: https://doubtiu.github.io/DynamiCrafter.github.io/
- LDM3D: https://github.com/IntelLabs/theia-iso
- 2DGS: https://buaa-pal.github.io/2DGS/
- DL3DV-10K: https://dl3dv-10k.github.io/DL3DV-10K.github.io/
- Mip-NeRF360: https://jonbarron.info/mipnerf360/
- ReconFusion: https://reconfusion.github.io/
- Tanks and Temples: https://www.tanksandtemples.org/
- Score Distillation (DreamFusion): https://dreamfusion3d.github.io/
