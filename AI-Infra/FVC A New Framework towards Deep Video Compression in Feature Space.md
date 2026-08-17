---
source_pdf: FVC A New Framework towards Deep Video Compression in Feature Space.pdf
paper_sha256: f4c8efe170979799da89d90b3005cf6b770358af86f985206093c793fb0a3515
processed_at: '2026-08-04T11:38:13-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我用人话给你讲讲 FVC 这篇 paper 到底在干啥。

---

## 一句话总结

以前的深度视频压缩都在**像素层面**做运动估计和补偿，FVC 说：咱们全部搬到 feature space 去做吧，效果会好很多。

就这么简单的一个 insight，但背后的逻辑链条很 elegant。

---

## 视频压缩到底在干嘛？

你有一串视频帧 $X_1, X_2, ..., X_t$。相邻帧之间大部分内容是重复的，比如一个人走路，背景基本不变，只是人挪了位置。视频压缩的核心就是：

- **Motion Estimation**: 算出来这个人从上一帧到这一帧挪了多少（叫 optical flow）
- **Motion Compensation**: 根据这个 flow，把上一帧"扭"过来，得到一个 prediction
- **Residual Coding**: prediction 不完美，把"预测错了的那部分"压缩传输
- **Entropy Coding**: 把上面这些信息变成比特流

接收端拿到的就是：flow + residual bits → 反推 → 重建出当前帧。

---

## 以前的方法（DVC 那一套）有什么问题？

DVC (CVPR 2019) 是第一个 end-to-end 的深度视频压缩框架，思路很好，但所有操作都在 pixel space 做。问题出在哪？

**问题 1：Pixel-level optical flow 在复杂运动下很不准**

想象一匹马在跑，马的腿部肌肉在变形，鬃毛在飘，背景的尘土在飞。这种 non-rigid motion 用 optical flow 估计，每个像素都要给一个 $(u, v)$ 位移向量。但实际中这种 dense per-pixel flow 非常 noisy，尤其在物体边界、遮挡区域。

**问题 2：Pixel-level warping 会引入 artifacts**

拿到 flow 之后，你要把上一帧"扭"过来。这个 warping 操作本质上是 bilinear interpolation，会在高频细节处产生模糊和 ghosting。越模糊，residual 越大，要传的 bits 越多。

**问题 3：Pixel-level residual 难压缩**

如果 prediction 质量差，residual 里会含有大量 high-frequency structure，这玩意儿比原始图像还难压缩。

这三个问题加在一起，就是 pixel-space paradigm 的 bottleneck。

---

## FVC 的核心 Insight

Deep features 在 image restoration、video super-resolution 这些任务上已经被验证 representation power 远超 raw pixels。那为什么不把 ME / MC / Residual Coding 全部放到 feature space 去做？

Feature space 是什么概念？就是你把原始图像 $X_t$ 通过一个 conv 网络提取成 $F_t$，这个 $F_t$ 的 spatial size 是原图的 1/2（stride=2），但 channel 数变多了。它编码的是 semantic structure 而非 raw pixel values。

在 feature space 做操作有什么好处？

- Feature 经过 conv 已经 abstract 过了，noise-robust
- Motion 在 feature space 的表达更 compact（semantic-level 的 motion 而非 pixel-level）
- Residual 在 feature space 更 sparse，更好压缩
- 所有模块共享同一个 representation，gradient flow 更顺畅

---

## Deformable Convolution 是怎么回事？

这是 FVC 最关键的创新。我用最直觉的方式讲讲。

**传统卷积**：一个 $3 \times 3$ 的 kernel 在 feature map 上滑动，每次在 9 个固定位置（regular grid）上采样，加权求和。

**Deformable Conv**：同样是 $3 \times 3$ kernel，但每个采样位置可以"漂移"。网络学一组 offset $\Delta p_n$，告诉 kernel "别在 regular grid 上采，去这几个位置采"。

这和 optical flow warping 有什么区别？

Optical flow warping：给每个像素一个 $(u, v)$，然后做 bilinear warp。这是一个**per-pixel translation field** 的假设。它假设每个像素是独立平移的。但在 non-rigid motion 下，这个假设太强了。

Deformable Conv：给每个 kernel 位置一个 offset。一个 $3 \times 3$ kernel 只有 9 个 offsets。它控制的是**kernel 的采样形状**，让 kernel 可以根据 local motion pattern 自适应地扭曲自己的形状。

直觉上理解：optical flow 是"告诉你每个像素往哪挪"，deformable conv 是"告诉 kernel 该从哪些位置采样来融合"。后者更 flexible，也更 robust，因为它是 kernel-level 的 motion 而非 pixel-level 的 motion。

FVC 把 feature map 分成 G=8 个 channel groups，每个 group 共享一组 offsets。这样既减少了 offset map 的 channel 数，也让同组 channel 的 motion pattern 一致。

---

## FVC 的 Pipeline 走一遍

假设现在要压缩第 $t$ 帧。

**Step 1: Feature Extraction**

把当前帧 $X_t$ 和上一帧的重建帧 $\hat{X}_{t-1}$ 各自通过一个 conv 网络（stride=2 + resblocks），提取成 feature $F_t$ 和 $F_{t-1}^{ref}$。

**Step 2: Deformable Compensation（核心模块）**

这个模块干三件事：

1. **Motion Estimation**：把 $F_t$ 和 $F_{t-1}^{ref}$ concatenate 起来，通过一个轻量级两层 conv 网络，输出 offset map $O_t$。这个 $O_t$ 就是 deformable conv 需要的 offsets。

2. **Motion Compression**：offset map $O_t$ 也要传到 decoder 端，所以用 auto-encoder 风格的网络压缩它（encoder → quantize → decoder），得到 reconstructed offset $\hat{O}_t$。

3. **Motion Compensation**：用 $\hat{O}_t$ 作为 offsets，对 $F_{t-1}^{ref}$ 做 deformable convolution，输出 warped feature，再用两层 conv refine 一下，得到 predicted feature $\bar{F}_t$。

**Step 3: Residual Compression**

算 residual feature $R_t = F_t - \bar{F}_t$，用另一个 auto-encoder 压缩，得到 $\hat{R}_t$。

把 $\hat{R}_t$ 加回 $\bar{F}_t$，得到 initial reconstructed feature $\tilde{F}_t$。

**Step 4: Multi-frame Feature Fusion**

FVC 用 3 个历史帧的 features $F_{t-1}^{ref}, F_{t-2}^{ref}, F_{t-3}^{ref}$。对每个历史帧，先用 deformable compensation 得到 predicted feature，然后用 non-local attention 机制让 $\tilde{F}_t$ 和这些 predicted features 做注意力交互。

Non-local attention 怎么做的？对 $\tilde{F}_t$ 上的每个位置 $(i,j)$，拿一个小 patch 去 reference feature 上对应位置的 patch 做 similarity 计算，softmax 得到 attention weight，加权求和。

最后把 4 个 refined features（3 个历史帧 + 自己）concat 起来，通过一个 conv 层融合，residual add 回 $\tilde{F}_t$，得到最终 $\hat{F}_t$。

**Step 5: Frame Reconstruction**

$\hat{F}_t$ 通过 deconv 网络（resblocks + deconv with stride=2），上采样回 pixel space，得到重建帧 $\hat{X}_t$。

---

## Loss Function

标准 RD trade-off：

$$RD = R_o + R_r + \lambda \cdot d(X_t, \hat{X}_t)$$

- $R_o$：编码 offset map 的 bits
- $R_r$：编码 residual feature 的 bits
- $\lambda$：控制 rate-distortion trade-off，取值 $\{256, 512, 1024, 2048\}$，对应不同 quality level
- $d$：MSE 或 MS-SSIM

训练时用 uniform noise 模拟量化（可微分），推理时直接 round。

---

## 实验结果怎么说

**BDBR（相对于 H.265 的 bit-rate saving）**

在 UVG 数据集上，FVC 比 H.265 省 28.71% bit-rate，而之前最好的方法 HU_ECCV20 只省 13.27%。这个 margin 很大。

在 HEVC 各 class 上，FVC 也全面领先。最夸张的是 HEVC Class C，之前的方法有的还比 H.265 差（DVC +20.65%），FVC 能省 14.18%。

**Ablation Study 的关键发现**

论文做了很细致的 ablation，最有意思的发现是：

- Motion 在 feature space 操作 vs pixel space 操作：**差 1.2 dB**
- Residual 在 feature space vs pixel space：**差 0.3 dB**
- Multi-frame fusion module：**贡献 0.5 dB**
- Non-local attention：**贡献 0.2 dB**

这说明什么？**motion compensation 从 pixel space 搬到 feature space，是最大的 performance gain 来源**。这正好验证了 FVC 的核心 hypothesis：pixel-level optical flow + warping 是 bottleneck。

**Deformable Compensation 单独评估**

在 HEVC Class C 上，只看 motion prediction 质量（不算 residual）：

- DVC*（用相同的压缩网络，但 pixel-level flow）：PSNR 24.85 dB @ 0.051 bpp
- FVC：PSNR 26.60 dB @ 0.017 bpp

FVC 用 1/3 的 motion bits 换得 1.75 dB 更高的 prediction quality。这个结果非常 striking，说明 feature-space deformable compensation 的 coding efficiency 远超 pixel-space optical flow。

---

## 为什么这个工作 Important？

从你的视角看，FVC 的 significance 在于：

**1. Paradigm shift**

以前的 deep video compression 是"用神经网络替换 H.264/H.265 的各个模块"。FVC 是"重新思考在哪个 space 做操作"。这个 abstraction level 更高。

**2. Deformable Conv 进入 video compression**

Deformable conv 在 object detection、video super-resolution 里已经用了好几年，但没人把它用到 video compression。FVC 是第一个，而且效果很好。这说明 video compression community 和 video restoration community 的 ideas 可以 cross-pollinate。

**3. Feature-space coding 的 empirical evidence**

之前有人做 feature-level residual（Feng et al. CVPR 2020 workshop），但没人把所有操作都放 feature space。FVC 用实验证明了 feature-space coding 全面优于 pixel-space coding，这给后续工作提供了 strong baseline。

**4. 和 generative compression 的 connection**

FVC 的 feature extraction = encoder，frame reconstruction = decoder，这本质上是 VAE 结构。Feature space 就是 latent space。这和 VQ-VAE、diffusion-based compression 的 trends 一致。FVC 可以看作 VAE-based video compression + deformable temporal prediction。

---

## 代码实现细节

论文用 PyTorch 实现，在单张 2080Ti 上训练。两阶段训练：

- Stage 1：不用 multi-frame fusion，train 2M steps，lr=5e-5
- Stage 2：加入 multi-frame fusion，先 train 400K steps lr=5e-5，再 100K steps lr=5e-6

参数量约 26M，大部分在 compression network 里。推理时间 1080p 一帧 548ms，其中 multi-frame fusion 占 347ms（论文说未来要加速这块）。

数据集用 Vimeo-90K 训练，测试在 HEVC/UVG/VTL/MCL-JCV 四个 benchmark 上。

---

## 我的一些 Intuition

读完这篇 paper 我的几个 takeaway：

**Feature space 是更好的 compression working space**。这不是 FVC 独有的 insight，但 FVC 在 video coding 上验证了它。未来如果做 generative compression（比如用 diffusion model 压视频），大概率也会在 latent space 操作。

**Deformable conv 比 optical flow 更适合 coding**。Optical flow 的 per-pixel translation 假设太强。Deformable conv 的 kernel-level motion 更 compact 也更 robust。这个 insight 可能在 video generation、video prediction 里也有借鉴价值。

**Multi-frame fusion + non-local attention 是 temporal context 的关键**。Single reference frame 的 prediction 能力有限，利用 long-range temporal redundancy 能显著提升。这和 video restoration 的多帧聚合思路一致。

**End-to-end optimization 的威力**。传统 codec 的各个模块是 hand-crafted 且 isolated 的，深度方法让所有模块共享 representation 和 gradient，这比 isolated optimization 强得多。

**Future directions**：FVC 的 multi-frame fusion 模块太慢（347ms），这是个 bottleneck。如果能用更轻量的 attention（比如 local window self-attention）替代 non-local attention，速度会好很多。另外，如果用 transformer-based entropy model 替代 hyperprior，rate estimation 会更准确。

---

参考链接：
- FVC paper: https://arxiv.org/abs/2105.09600
- DVC: https://arxiv.org/abs/1812.00101  
- Deformable Conv: https://arxiv.org/abs/1703.06211
- Deformable Conv v2: https://arxiv.org/abs/1811.11168
- EDVR (deformable conv for video restoration): https://arxiv.org/abs/1905.02716
- Hyperprior image compression (Minnen et al.): https://arxiv.org/abs/1809.02736
- Scale-Space Flow (EA_CVPR20): https://arxiv.org/abs/2006.06242

希望这个版本更直觉一些。FVC 的核心其实就一句话：**把所有 video coding 操作从 pixel space 搬到 feature space，用 deformable conv 替代 optical flow warping**。剩下的 engineering detail 都是在 make this idea work。

---

# FVC: Feature-Space Video Coding 深度解析

## 1. 核心直觉: 为什么要把视频压缩搬到 Feature Space?

传统 hybrid video coding framework (H.264/H.265) 的核心 pipeline 包含四个关键操作:
- **Motion Estimation (ME)**: 在 pixel space 估计 optical flow
- **Motion Compensation (MC)**: 用 optical flow warp 参考帧得到预测帧
- **Residual Coding**: 压缩当前帧和预测帧之间的差异
- **Entropy Coding**: 比特流生成

DVC (Lu et al., CVPR 2019) 等深度视频压缩方法把这个 hybrid framework 用 neural network 替换，但**所有操作仍停留在 pixel space**。FVC 的核心 insight 是: pixel-level operations 存在三个 fundamental limitations:

1. **Pixel-level optical flow estimation 在 non-rigid motion 下不准确** — 例如人体运动、烟雾、水波这种复杂非刚体运动
2. **Pixel-level motion compensation 引入 artifacts** — bilinear/spline warping 会模糊高频细节
3. **Pixel-level residual 难压缩** — 当 prediction 不准时，residual 含有大量高频结构信息

FVC 的 key idea: 既然 deep features 在 image restoration、video super-resolution 等任务上都有更强的 representation power，何不把 ME / MC / Residual Coding 全部放到 feature space 操作? 这就是 FVC (Feature-space Video Coding) 的核心思想。

参考链接:
- DVC 原始论文: https://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_DVC_An_End-To-End_Deep_Video_Compression_Framework_CVPR_2019_paper.pdf
- Deformable Conv 原始论文: https://arxiv.org/abs/1703.06211

---

## 2. 整体 Architecture 解析

FVC 的 pipeline (Figure 1):

```
Input frame X_t ──► Feature Extraction ──► F_t (feature)
                                              │
Previous recon frame X̂_{t-1} ──► Feature Extraction ──► F_{t-1}^ref
                                              │
                    ┌─────────────────────────┘
                    ▼
        ┌──────────────────────────┐
        │ Deformable Compensation   │ ──► Predicted feature F̄_t
        │  (ME + MC in feature space)│
        └──────────────────────────┘
                    │
                    ▼
       R_t = F_t - F̄_t  ──► Residual Compression ──► R̂_t
                                              │
                    ┌─────────────────────────┘
                    ▼
              F̃_t = F̄_t + R̂_t (initial recon)
                    │
                    ▼
        ┌──────────────────────────┐
        │ Multi-frame Feature Fusion│ ──► F̂_t
        │ (NLA over F_{t-1}^ref,     │
        │  F_{t-2}^ref, F_{t-3}^ref)│
        └──────────────────────────┘
                    │
                    ▼
        Frame Reconstruction ──► X̂_t
```

**关键 insight**: 所有 redundancy reduction (motion + residual) 都发生在 downsampled feature space (stride=2，所以 spatial size 是原图的 1/2)，然后再用 deconv 上采样回 pixel space。这意味着 compression 的"工作空间"是更紧凑的语义空间，而非像素空间。

---

## 3. Deformable Compensation Module — 数学细节

这是 FVC 最核心的创新点。把 motion estimation 和 motion compensation 都放到 feature space，用 deformable convolution 替代 optical flow warping。

### 3.1 Deformable Convolution 公式

标准 2D convolution 在 location $p_0$ 上的输出:

$$
y(p_0) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0 + p_n)
$$

其中:
- $p_0$: 输出 feature map 上的 spatial location, $p_0 \in \mathbb{Z}^2$
- $p_n$: 在 kernel $\mathcal{R} = \{-1, 0, 1\}^2$ 上的偏移 (regular grid)
- $w(p_n)$: 该位置的 convolution weight
- $x(p_0 + p_n)$: 输入 feature map 上对应位置的值

Deformable convolution 引入 learnable offset $\Delta p_n$:

$$
y(p_0) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0 + p_n + \Delta p_n)
$$

其中 $\Delta p_n \in \mathbb{R}^2$ 是从 offset map 中采样得到的 fractional offset。由于 $p_0 + p_n + \Delta p_n$ 通常落在 fractional position，使用 bilinear interpolation 计算 $x$:

$$
x(p) = \sum_{q \in \mathcal{N}(p)} x(q) \cdot \max(0, 1 - |p_x - q_x|) \cdot \max(0, 1 - |p_y - q_y|)
$$

其中 $\mathcal{N}(p)$ 是 $p$ 周围 4 个最近整数像素点。

### 3.2 为什么 Deformable Conv 比 Optical Flow Warping 更强?

**Optical flow warping** 假设所有像素的运动场是 dense 的 per-pixel flow $(u, v)$, warp 操作:

$$
\hat{X}(x, y) = X(x + u(x,y), y + v(x,y))
$$

这有一个隐含假设: **motion 是 smooth、locally translational 的**。在 non-rigid motion (如肌肉变形、烟雾) 下，每个像素独立 flow 估计会噪声大、不准确。

**Deformable convolution** 的思路不同:
- 不是给每个像素一个 flow vector，而是给每个 kernel 位置一个 offset
- $3 \times 3$ kernel 有 9 个 offsets，每个 offset 控制一个采样位置
- **不同 channel group 共享 offset** (FVC 用 $G=8$ groups, 每个 group 共享同一组 offsets)

这意味着: 一个 $3 \times 3$ kernel 不是在 regular grid 上采样输入，而是通过 9 个 learnable offsets 决定从哪里采样。这给了网络**更强的 spatial adaptation capability** — kernel 可以根据 local motion pattern 自适应地"扭曲"自己的采样形状。

### 3.3 FVC 的 Offset Map 结构

FVC 中 reference feature 被分成 $G=8$ 个 channel groups, 每个 group 共享一组 offsets。对 $3 \times 3$ kernel, 每个位置需要 9 个 offsets × 2 (x 和 y 方向) = 18 个值。所以 offset map 的 channel 数为:

$$
C_{offset} = G \times 2 \times 9 = 8 \times 2 \times 9 = 144
$$

但 FVC 实际使用的是 72 个 channel (Figure 4 注释说 "2 directions of offset map"), 因为在 deformable conv 实现中，groups 之间的 offset 也可能共享。总 offset map 数量论文提到 "72 offset maps"。

### 3.4 Deformable Compensation Pipeline (Figure 3)

1. **Motion Estimation**: 两层 conv 网络输入 $F_{t-1}^{ref}$ 和 $F_t$ (concatenate), 输出 offset map $O_t$
2. **Motion Compression**: auto-encoder style 网络压缩 $O_t$:
   - Encoder: Resblocks → latent
   - Quantize (训练时加 uniform noise, 推理时 round)
   - Decoder: Resblocks → reconstructed $\hat{O}_t$
3. **Motion Compensation**: deformable conv 用 $F_{t-1}^{ref}$ 作为输入, $\hat{O}_t$ 作为 offsets → 输出 warped feature → 两层 conv refine → $\bar{F}_t$

---

## 4. Multi-frame Feature Fusion Module — Non-local Attention

### 4.1 Motivation

只用一帧参考容易 error propagation。FVC 利用 3 个 previous reconstructed frames 的 features $F_{t-1}^{ref}, F_{t-2}^{ref}, F_{t-3}^{ref}$，通过 non-local attention 融合。

### 4.2 Non-local Attention 数学形式

对于当前帧 initial reconstructed feature $\tilde{F}_t$ 的位置 $(i, j)$, feature vector $f_{i,j} \in \mathbb{R}^{1 \times 1 \times c}$, 在参考帧 $\bar{F}_{t-k}^{ref}$ 上找 collocated patch $f_{i,j}^{ref} \in \mathbb{R}^{p \times p \times c}$ (typically $p=3$ or $5$)。

Attention weight 通过 channel-wise convolution 计算:

$$
a_{i,j}^{(m,n)} = \text{softmax}_{(m,n) \in p \times p}\left( \text{Conv}_{c \to 1}(f_{i,j} \oplus f_{i,j}^{ref}(m,n)) \right)
$$

其中:
- $\oplus$ 表示 concatenation
- $\text{Conv}_{c \to 1}$: 沿 channel 方向做卷积, 输出单 channel scalar
- softmax 在 patch 内 spatial 维度上归一化

Refined feature:
$$
\hat{f}_{i,j}^{ref} = \sum_{(m,n)} a_{i,j}^{(m,n)} \cdot f_{i,j}^{ref}(m,n)
$$

### 4.3 Self-attention on $\tilde{F}_t$ 本身

论文提到 "we also refine $\tilde{F}_t$ itself by using the non-local attention mechanism based on the so-called self-attention mechanism"。也就是说 $\tilde{F}_t$ 自己对自己做 non-local attention，挖掘 spatial long-range dependency。

### 4.4 Final Fusion

最后 concat 四个 refined features ($\hat{F}_{t-3}^{ref}, \hat{F}_{t-2}^{ref}, \hat{F}_{t-1}^{ref}, \hat{F}_t^{ref}$), 通过 conv 层融合, **residual add 回 $\tilde{F}_t$**:

$$
\hat{F}_t = \tilde{F}_t + \text{Conv}(\text{Concat}[\hat{F}_{t-3}^{ref}, \hat{F}_{t-2}^{ref}, \hat{F}_{t-1}^{ref}, \hat{F}_t^{ref}])
$$

这种 residual 设计保证 fusion module 至少不会比 input 差，类似 U-Net 的 skip connection 思想。

---

## 5. Rate-Distortion Optimization

### 5.1 Loss Function

FVC 优化标准 RD trade-off:

$$
RD = R + \lambda D = R_o + R_r + \lambda \cdot d(X_t, \hat{X}_t)
$$

变量含义:
- $R_o$: 编码 offset map $O_t$ 所需的 bits
- $R_r$: 编码 residual feature $R_t$ 所需的 bits
- $\lambda$: Lagrange multiplier, 控制 rate-distortion trade-off, 取值 $\{256, 512, 1024, 2048\}$
- $d(\cdot, \cdot)$: distortion, 可以是 MSE 或 MS-SSIM
- $X_t, \hat{X}_t$: 原始 frame 和重建 frame

### 5.2 Bit Estimation

训练阶段用 hyperprior entropy model (来自 Minnen et al., NeurIPS 2018) 估计 bits:

$$
R_o = -\log_2 p(\hat{O}_t | z_o), \quad R_r = -\log_2 p(\hat{R}_t | z_r)
$$

其中 $z_o, z_r$ 是 hyperprior 的 latent variables，用于 capture spatial dependencies。FVC 没有用 auto-regressive context model (因 computation cost)。

### 5.3 Quantization

训练时用 uniform noise 模拟量化:
$$
\tilde{y} = y + U(-0.5, 0.5)
$$

推理时直接 round:
$$
\hat{y} = \text{round}(y)
$$

这是 Ballé et al. (ICLR 2018) 提出的可微分量化技巧。

---

## 6. 实验数据表解读

### 6.1 BDBR 结果 (Table 1)

BDBR (Bjontegaard Delta Bit-Rate) 表示相对于 anchor (H.265) 的 bit-rate saving。负值表示省比特率。

| Dataset | DVC | LU_ECCV20 | HU_ECCV20 | **FVC** |
|---------|-----|-----------|-----------|---------|
| HEVC Class B | +2.97 | -15.92 | -14.91 | **-23.75** |
| HEVC Class C | +20.65 | -3.78 | +1.76 | **-14.18** |
| HEVC Class D | +14.08 | -8.29 | -1.77 | **-18.39** |
| UVG | +8.45 | -7.34 | -13.27 | **-28.71** |
| VTL | -10.92 | -16.85 | -20.17 | **-28.10** |
| MCL-JCV | +13.94 | +4.75 | -13.71 | **-22.48** |

在 UVG (1080p 高帧率视频) 上 FVC 比 H.265 省 28.71% bit-rate，比次优的 HU_ECCV20 多省 15.44%。这是相当大的 margin。

### 6.2 Ablation Study (Figure 7)

在 HEVC Class D 上:
- **FVC (完整模型)**: 最佳
- **FVC w/o NLA**: 下降 ~0.2 dB → non-local attention 贡献 0.2 dB
- **FVC-basic (w/o MFF)**: 在 0.3 bpp 时下降 0.5 dB → multi-frame fusion 贡献 0.5 dB
- **FVC-basic (FS-motion & PS-residual)**: 比 FVC-basic 低 0.3 dB @ 0.38 bpp → feature-space residual 比 pixel-space residual 多 0.3 dB
- **FVC-basic (PS-motion & PS-residual)**: 比 FS-motion & PS-residual 低 1.2 dB @ 0.4 bpp → feature-space motion 多 1.2 dB

**重要 insight**: motion 在 feature space 操作带来的增益 (1.2 dB) 远大于 residual 在 feature space 操作的增益 (0.3 dB)。这说明 pixel-space optical flow + warping 是 bottleneck，换成 deformable conv 后 benefit 最大。

### 6.3 Deformable Compensation 单独评估 (Figure 8)

在 HEVC Class C 上, 仅比较 motion prediction 质量:
- DVC*: PSNR = 24.85 dB @ 0.051 bpp (估计值)
- FVC: PSNR = 26.60 dB @ 0.017 bpp

FVC 用 **1/3 的 motion bits** 换得 **1.75 dB 更高 PSNR** 的 prediction。这是非常惊人的结果，说明 feature-space deformable compensation **大幅提升 motion coding efficiency**。

### 6.4 Running Time

| Method | Time (1080p) | BDBR (Class D) | Params |
|--------|------------|---------------|--------|
| DVC | 460ms | +14.08% | - |
| DVC* | 709ms | +4.31% | - |
| FVC-basic | 201ms | -7.09% | ~24M (compression nets) |
| FVC | 548ms | **-18.39%** | ~26M (total) |

FVC 的 multi-frame fusion 模块花了 347ms, 这是 future work 的加速目标。但 FVC-basic 已经比 DVC/DVC* 都更快且更好。

---

## 7. 关键 Insight 总结 — 给 Karpathy 的 Intuition

1. **Working in feature space 是 inductive bias 的 shift**: pixel space 是 entropy 最大的表示，feature space 经过 conv 已经把 high-frequency details 编码为 semantic structure, 在这个空间做 motion/warping/residual coding 天然 noise-robust

2. **Deformable conv 比 optical flow 更适合 video coding**: optical flow 假设 per-pixel translation field, deformable conv 让 kernel shape 自适应 motion pattern, 这对 non-rigid motion 鲁棒得多。这个 insight 在 video super-resolution (TDAN, EDVR) 中已经被验证，FVC 是第一个把它带到 video compression 的工作

3. **Multi-frame fusion + non-local attention 是 temporal context 的关键**: video coding 不只是 pair-wise prediction, 利用 long-range temporal redundancy 能显著提升 — 类似 video restoration 中 multi-frame aggregation 的思想

4. **压缩系统在 feature space 中所有组件协同**: pixel-space approach 是 isolated module optimization, feature-space approach 让所有 module 共享 representation, gradient flow 更顺畅, end-to-end optimization 更有效

5. **未来方向**: 
   - 用 latent diffusion model 做 residual coding (类似 Stable Diffusion 思路)
   - Transformer-based entropy model 替代 hyperprior
   - 神经 codec 的 implicit neural representation (NeRF-style)
   - 多帧 deformable compensation 的 temporal consistency

参考阅读:
- FVC 论文: https://arxiv.org/abs/2105.09600
- DVC: https://arxiv.org/abs/1812.00101
- Deformable Conv v2: https://arxiv.org/abs/1811.11168
- EDVR (deformable conv for video restoration): https://arxiv.org/abs/1905.02716
- Scale-Space Flow (EA_CVPR20): https://arxiv.org/abs/2006.06242
- Content Adaptive DVC (LU_ECCV20): https://arxiv.org/abs/2003.11242
- Hyperprior image compression: https://arxiv.org/abs/1809.02736

---

## 8. 从 Karpathy 视角的延伸思考

这个 paper 的 architecture 本质上是把 classical video coding 的"block matching + DCT"范式，完全用 **learnable feature representations + deformable sampling** 替换。如果我们把这件事抽象一下:

```
Classical:      X_t = MC(X_{t-1}, flow) + DCT_residual(Q)
FVC:            F_t = DeformConv(F_{t-1}^ref, offset) + AutoEnc_residual(Q)
                X_t = Decoder(F_t)
```

FVC 的 feature space 实际上是 latent variable model 的 latent space, compression 在 latent space 中进行, 这正是 VAE-based generative compression 的思路。FVC 可以看作 **VAE-based video compression + deformable temporal prediction** 的组合。

如果再大胆一点联想, 这和 VQ-VAE、Neural Compression 的 trends 一致:
- **神经压缩的本质**: 找到信息的最小 sufficient statistics, 在 latent space 操作
- FVC 的 feature extraction = encoder, frame reconstruction = decoder, 这构成 VAE 结构
- Deformable compensation = temporal prior (类似 VAE 中的 learned prior)
- Multi-frame fusion = amortized inference over temporal context

未来的 neural video codec 很可能走向: **diffusion-based generative prior + feature-space motion compensation + transformer entropy model**, FVC 在这条 path 上提供了 feature-space coding 的关键 empirical evidence。

希望这些细节和 intuition 对你构建 mental model 有帮助! 整体上 FVC 是 deep video compression 领域一个 critical 的 paradigm shift paper, 值得仔细咀嚼其 feature-space 的设计选择。
