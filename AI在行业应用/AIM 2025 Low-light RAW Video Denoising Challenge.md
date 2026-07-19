---
source_pdf: AIM 2025 Low-light RAW Video Denoising Challenge.pdf
paper_sha256: 05de93bfe75fe382cced4f4bca1f3eab5004f1a722633e51c257c254aefb0548
processed_at: '2026-07-18T06:19:16-07:00'
target_folder: AI在行业应用
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AIM 2025 Low-light RAW Video Denoising Challenge 深度解析

Andrej, 这篇 paper 是 MSU 团队组织的 AIM 2025 Low-light RAW Video Denoising Challenge 的官方报告,核心 contribution 是构建了一个 multi-sensor、multi-condition 的 realistic benchmark,并汇集了 4 个 team 的方法对比。我会从 dataset 设计、noise 物理、各 team 方法细节、实验结论这几个维度展开,目标是帮你 build 出对这个领域当前 SOTA 的 intuition。

---

## 1. 为什么需要一个新的 dataset?——Existing benchmarks 的痛点

Low-light RAW video denoising 长期缺乏 fair benchmark,核心原因有二:

**(a) Ground truth 难以获取**: 与 single-image denoising 不同, video 中 scene 有 motion, 你无法简单地用 long-exposure 一张当作 GT, 因为 motion blur 会污染 reference。Existing 方案分两类:
- Object-motion datasets (e.g., SMD [56], SID-Video [11]) — noisy/clean pairs 之间 alignment 较差
- Beam-splitter / mechatronic rigs (e.g., BVI-LowLight [4], SID-Dark [23, 51]) — 只能产生 low/normal-light 对,不是真正 denoise 任务所需的 high-SNR reference

**(b) Synthetic noise 的 transfer gap**: 很多工作 [2, 7, 26, 58] 用 heteroscedastic Gaussian (signal-dependent) 或 Noise Flow 仿真,但真实 sensor 还存在 row noise、fixed pattern noise、quantization、clipping 等结构化 artifact,合成-真实之间存在 gap。YOND [22] 这篇就明确指出 SIDD-style multi-frame averaging 在 low-light 下会留下残差。

本文的解法: 用 **motorized rail + stop-motion capture** 解决 alignment, 用 **200~500 帧 burst averaging** 解决 GT SNR。这是 SIDD [1] protocol 的 video extension。

参考链接:
- SIDD: https://www.eecs.yorku.ca/~kamel/sidd/
- BVI-LowLight: https://arxiv.org/abs/2402.01970
- YOND: https://arxiv.org/abs/2506.03645

---

## 2. Dataset capture protocol 的关键设计

### 2.1 Stop-motion rail 采集

Setup 见 Figure 2: smartphone 固定在 motorized linear translation stage 上, NVIDIA Jetson Nano 同步控制 stage 位移和 shutter 触发, Pani [13] app + libsoftwareasync [5] 实现远程 RAW burst 采集。

**核心 trick**: 不是真正拍视频, 而是模拟 video。流程是 "move → capture → move → capture → ...", 这样每一帧的位置精确可控, 后续可以在这个位置上 burst 拍 200/500 帧用来 averaging GT。这是 alignment 的物理保证。

### 2.2 Frame-rate emulation

要让数据反映真实视频的 motion magnitude, paper 把 exposure time 和 frame-to-frame displacement 耦合:
- 1/24 s: 每帧位移 5x
- 1/60 s: 中等位移
- 1/120 s: 最小位移

这是合理的, 因为长 exposure 对应低 FPS, 同一物体在 frame 间的物理位移更大。

### 2.3 ISO setting 矩阵 (Table 1)

ISO 的设计目的是让不同 (lux, exposure) 条件下输出 brightness 大致一致, 这样 test 的时候比较的是 pure denoise 能力, 而不是 brightness normalization:

| Illuminance | 1/24 s | 1/60 s | 1/120 s |
|---|---|---|---|
| 10 lx | 800 | 2000 | 4000 |
| 5 lx | 1250 | 3125 | 6250 |
| 1 lx | 2000 | 5000 | 10000 |

直觉上, exposure 减半 → ISO 加倍, lux 减半 → ISO 加倍, 两者相乘保持 photon count 大致守恒 (实际上 sensor 满阱和 read noise 会让线性偏离, 但作为 protocol 足够 fair)。

### 2.4 Sensor 覆盖 (Table 2)

5 个手机 (Samsung Z Fold4, Pixel 5a, Pixel 7 Pro, Samsung S20, POCO X3 Pro) 共 14 个 camera module: wide / ultrawide / telephoto / front 都有, 覆盖 different sensor size、pixel pitch、CFA。注意 Z Fold4 Wide max ISO 只到 1600, Pixel 7 Pro Telephoto max ISO 1143, 这两个 sensor 在最 extreme 条件下会 saturate 输出, 这也是 cross-sensor 表现差距的来源之一。

### 2.5 GT 质量与 SNR upper-bound

Burst averaging 的 SNR 提升 √N:
$$\text{SNR}_{\text{avg}} = \sqrt{N} \cdot \text{SNR}_{\text{single}}$$

Train/val 用 200 帧 → SNR 提升 ~23 dB; test 用 500 帧 → 提升 ~27 dB。在 1 lx / 1/120 s 这种 extreme 条件下, 单帧 SNR 可能只有 5~10 dB, GT 残留 noise 约 -15 dB, 这对评价 PSNR 48+ 的 method 已经有 non-negligible bias。VMCL-ISP 的 Figure 12 明确展示了 GT 里的 residual noise 和 defective pixels。

---

## 3. Noise 物理回顾 (build intuition)

RAW domain 的 noise 主要由两部分组成:

$$y = x + \eta, \quad \eta \sim \mathcal{N}(0, \sigma^2(x))$$

其中 signal-dependent variance 经典建模为:

$$\sigma^2(x) = D \cdot x + R$$

- $x$: 真实 photon count (linear RAW)
- $D$: dark current / shot noise 系数, 正比于 gain (ISO)
- $R$: read noise variance, 也随 ISO 变化

更精细的模型 (Wei et al. [54]) 还会加上 row noise、banding、FPN、quantization 和 clipping:

$$\sigma^2_{\text{total}}(x) = D \cdot x + R + \sigma^2_{\text{row}} + \sigma^2_{\text{FPN}}$$

这也是为什么 challenge 强调 signal-dependent 和 sensor-specific——一个在 Pixel 7 Pro Wide 上训练的 model, 直接迁移到 Z Fold4 Telephoto 会有 drop, 因为 $D$、$R$、row noise 的相对权重不同。

Variance-Stabilizing Transform (VST) 的目的就是把这个 heteroscedastic noise 变成 homoscedastic, 经典 Anscombe transform:

$$f(x) = 2\sqrt{x + \tfrac{3}{8}}$$

让变换后方差近似为 1, 然后用 AWGN denoiser 处理。YOND 的 EM-VST 是对 sensor-specific 噪声参数做 expectation matching, 让同一 VST 在不同 sensor 上都逼近 homoscedastic。

参考:
- Hasinoff "Photon, Poisson noise": https://link.springer.com/referenceworkentry/10.1007/978-0-387-31439-6_482
- Wei et al. (ECCV 2020): https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_A_Physics-Based_Noise_Formation_Model_for_Extreme_Low-Light_Raw_Denoising_CVPR_2020_paper

---

## 4. Challenge results 总览 (Table 3)

| Method | Multi-frame | PSNR ↑ | SSIM ↑ | Rank |
|---|---|---|---|---|
| SNU-ISPL (DarkVRAI) | ✓ | 48.32 | 0.9879 | 1 |
| XJAI | ✓ | 48.19 | 0.9865 | 2 |
| AxeraAI (NAFNet single) | ✗ | 46.52 | 0.9814 | 3 |
| VMCL-ISP (YOND) | ✗ | 45.76 | 0.9682 | 4 |
| UNet w/ Attn baseline | ✓ | 45.72 | 0.9797 | — |
| UNet baseline | ✗ | 43.52 | 0.9691 | — |
| Noisy | — | 36.06 | 0.8093 | — |

**Key intuition**: 
- Noisy → best: +12 dB, 这就是 denoise 任务的可压缩空间
- Single-frame SOTA (NAFNet) → best multi-frame: +1.8 dB, 这是 temporal 信息能贡献的 gain
- Multi-frame baseline (UNet+Attn) 已经能 beat single-frame SOTA, 说明 even naive temporal fusion 也有用
- 两 top 方法 gap 仅 0.13 dB, 但都甩开第 3 名 1.7 dB——说明 alignment + capture-condition-aware 这两件事做对了, multi-frame 的 ceiling 还很高

---

## 5. 方法细节深挖

### 5.1 SNU-ISPL: DarkVRAI (Rank 1)

整体是 two-stage: **Frame Alignment** + **Denoising**, 每个阶段都用 capture condition (sensor, lux, fps) 做 conditioning, 每个 stage 前置 BOSS (Burst-Order Selective Scan) module。

#### 5.1.1 BOSS module (基于 MambaIR [25])

State Space Model 的连续形式:
$$h'(t) = A h(t) + B x(t), \quad y(t) = C h(t)$$

离散化 (zero-order hold):
$$\bar{A} = e^{\Delta A}, \quad \bar{B} = (\Delta A)^{-1}(e^{\Delta A} - I) \cdot \Delta B$$
$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t$$

变量含义:
- $h_t \in \mathbb{R}^{N}$: hidden state at time $t$, $N$ 是 state dimension
- $x_t \in \mathbb{R}^{C}$: input feature at frame $t$
- $A \in \mathbb{R}^{N\times N}$: state transition matrix (history decay)
- $B \in \mathbb{R}^{N\times C}$: input projection
- $C \in \mathbb{R}^{C\times N}$: output projection
- $\Delta$: discretization step

**Selective scan** 让 $B$、$C$、$\Delta$ input-dependent:
$$B_t = \text{Linear}(x_t), \quad C_t = \text{Linear}(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}(x_t))$$

这样模型可以根据当前 frame 内容动态决定 forget 多少 history。BOSS 把这个机制用在 burst 的 temporal 维度上——10 帧 frame 按顺序扫, 长程依赖靠 state 累积。

直觉上, Mamba 在 1D sequence 上的优势是 linear complexity + 数据相关遗忘门, 用在 video temporal axis 上比 self-attention 更省显存, 比 ConvLSTM 更 expressive。

参考 MambaIR: https://arxiv.org/abs/2402.15648

#### 5.1.2 Frame Alignment (基于 Burstormer [19])

输入: 10 frame RAW, Bayer-packed 后每帧 4-channel (R, Gr, Gb, B)。Alignment module 输出 aligned features $F_i \in \mathbb{R}^{C\times H\times W}$ with $C=48$。

Burstormer 的核心是 multi-scale reference-based deformable attention: 每个 frame 的 patch 和 reference frame 的 patch 做相似度匹配, 通过 deformable offset 采样对齐。DarkVRAI 在每个 encoder 前加 BOSS, 把 temporal context 注入到 alignment 之前——这样 alignment 时 network 已经"知道"前后帧大概在哪里。

参考 Burstormer: https://openaccess.thecvf.com/content/CVPR2023/papers/Dudhane_Burstormer_Burst_Image_Restoration_and_Enhancement_Transformer_CVPR_2023_paper

#### 5.1.3 Denoising (NAFBlocks)

NAFBlock 的两个核心设计:

**SimpleGate** (替代 GELU/ReLU):
$$X \in \mathbb{R}^{2C\cdot H\cdot W} \to [X_1, X_2], \quad X_1, X_2 \in \mathbb{R}^{C\cdot H\cdot W}$$
$$\text{out} = X_1 \odot X_2$$

直觉: 把 channel 一半做"门控信号"乘到另一半上, 信息流类似 GLU 但省掉 activation function。

**Simplified Channel Attention (SCA)**:
$$s = \text{pool}(X) \in \mathbb{R}^{C\times 1\times 1}, \quad \text{out} = X \cdot s$$

去掉了 Conv 在 spatial 上的 1x1, 只留 channel attention, 因为 NAFNet 发现 restoration 任务里 channel 维度的 reweight 就足够。

NAFBlock 数量: encoder 4-4-4, bottleneck 8, decoder 4-4-4 (对称 U-Net 结构)。

#### 5.1.4 Capture-Condition Conditioning via Adaptive LayerNorm

Standard LayerNorm:
$$\text{LN}(z) = \gamma \odot \frac{z - \mu}{\sigma} + \beta$$

Adaptive LayerNorm (Peebles & Xie [44], DiT 用的):
$$\text{adaLN}(z, c) = (1 + \gamma(c)) \odot \frac{z - \mu}{\sigma} + \beta(c)$$

其中 $c$ 是 capture condition 的 embedding (sensor / lux / fps 三者 one-hot 后 concat, 再过一个 MLP 得到 trainable vector)。$\gamma(c)$ 和 $\beta(c)$ 通过两个独立 MLP 从 condition vector 生成。

直觉: 不同 capture condition 下 noise 的统计特性不同, model 需要根据 condition 调整 feature 的 normalize scale 和 shift——例如 1 lx/1/120s 时 noise 大, $\gamma$ 应该让 activation 更 "compressed", 1/24s/10lx 时 $\gamma$ 应该放大细节。

参考 Adaptive LayerNorm: https://arxiv.org/abs/2212.09748

#### 5.1.5 Training

- Patch size 256, stride 192
- Loss: $L = \|x - \hat{x}\|_1 + \lambda \cdot \text{MS-SSIM}(x, \hat{x})$
- Adam ($\beta_1=0.9, \beta_2=0.999$), lr $2\cdot 10^{-4}$ → cosine decay to $10^{-6}$
- 300K iterations, batch 4, 单张 RTX 3090

注意: 论文明确说 "without further augmentation to preserve the Bayer pattern"——意思是不能 random flip / rotation, 因为 Bayer CFA 的 spatial 邻接关系会破坏。

---

### 5.2 XJAI: Hierarchical Restormer (Rank 2)

#### 5.2.1 Pipeline (Figure 9)

9 帧输入 (注意是 9 不是 10, 因为 frame 9 是 target, 训练时用额外 20 个 noise realization 之一替换 target 增加鲁棒性), 通过 overlapped patch embedding → 4-level U-Net (Restormer blocks) → progressive refinement → residual + input → output。

**Patch embedding**: 3×3 conv, stride 1, overlapped——避免 non-overlapping 的 blocking artifact。

**Multi-scale**: 用 PixelUnshuffle 做 downsampling。PixelUnshuffle 是 PixelShuffle 的逆:
$$\text{PixelUnshuffle}(X)_{c \cdot r^2 + (i \cdot r + j), \, h, \, w} = X_{c, \, r\cdot h + i, \, r\cdot w + j}$$
其中 $r$ 是 downsample factor (这里 $r=2$), $X \in \mathbb{R}^{C\times rH \times rW}$ → 输出 $\mathbb{R}^{r^2 C \times H \times W}$。本质上是 space-to-depth reshape, 不丢失信息, 比 strided conv 更 lossless。

**Restormer block** 配置: [4, 6, 6, 8] across levels, heads [1, 2, 4, 8]。

Restormer 的两个关键 module:

**MDTA (Multi-Dconv Head Transposed Attention)**:
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
$$A = \text{softmax}\left(\frac{Q^{\top} K}{\sqrt{d}}\right), \quad \text{out} = V A$$

注意是 $Q^\top K$, 不是 $QK^\top$——attention 在 channel 维度算, 不是 spatial 维度。复杂度 $O(C^2 \cdot HW)$, 对高分辨率图友好。还要 $Q$、$K$、$V$ 之前有 3×3 depthwise conv 注入 local context。

**GDFN (Gated-Dconv Feed-Forward Network)**:
$$\text{out} = \text{Conv}\left((\text{GELU}(\text{Conv}_{1}(X))) \odot \text{Conv}_{2}(X)\right)$$

用 gating 替代 plain FFN, 增强非线性表达, 同样带 depthwise conv。

参考 Restormer: https://arxiv.org/abs/2111.09881

#### 5.2.2 Training trick: 随机噪声替换

训练时 9 帧中第 9 帧 (target) 以 1/20 概率被替换成 challenge 提供的另外 20 个 noisy realization 之一 (extra_noisy_09_00 ... extra_noisy_09_19)。这是一个 self-augmentation——给 model 看到同一 GT 下的不同 noise 样本, 提升 robustness。

AdamW, lr $5\cdot 10^{-4}$, Charbonnier loss $L = \sqrt{\|x-\hat{x}\|^2 + \epsilon^2}$ ($\epsilon$ 是 smoothing term, 通常 1e-3), cosine annealing, 80K iters。

Charbonnier 相比 L1 在零点附近可导, gradient 更平滑, 收敛更稳。

---

### 5.3 AxeraAI: NAFNet single-frame (Rank 3)

Edge device 友好导向的 baseline exploration。三种训练方式对比:

| Approach | 描述 | 结果 |
|---|---|---|
| Supervised single-frame | L1 loss, GT 配对 | 最好 |
| Self-supervised (N2N) | 用两个 noisy frame 互相监督 | 略差 |
| Multi-frame finetuning | 在 single-frame 基础上加多帧 fusion | 没有提升 |

**重要 observation**: AxeraAI 尝试 multi-frame finetune 后发现比 single-frame 还差, 这跟 top 2 team 的结论相反。可能原因: NAFNet 架构本身没有 explicit alignment module, 直接 concat 多帧会让 noise 也 concat, 反而干扰 network。这反过来说明 multi-frame 不是"输入多帧 + 大网络"就行, 必须 alignment-aware。

Training: Adam, lr 1e-4, batch 8, patch $256\times 256\times 4$ (4 channels 表示 Bayer-packed), 2000 epochs, 每 epoch 固定采 80 个 instance。单 RTX 4090, ~10h。

参考 NAFNet: https://arxiv.org/abs/2204.04676

---

### 5.4 VMCL-ISP: YOND (Rank 4)

Pipeline: **CNE → EM-VST → SNR-Net (Restormer-based)**。

#### 5.4.1 Coarse-to-fine Noise Estimation (CNE)

从 color chart patch 估计每个 sensor × condition 的 noise 参数 $\theta = (D, R, \sigma_{\text{row}}, ...)$。Coarse-to-fine 指先用整张图估粗值, 再用残差 refine。

#### 5.4.2 EM-VST

期望匹配的 VST: 让不同 sensor 经过同一个 transform 后, noise 方差都 match 到 1。直观上是对 Anscombe 的一种 sensor-aware 校准:
$$f(x; \theta) = 2\sqrt{D \cdot x + R + \tfrac{3}{8} D^2}$$

让变换后方差稳定在 1, 与 sensor 无关。

#### 5.4.3 SNR-Net (Figure 11)

Restormer backbone + guidance branch。Guidance branch 输入是 SNR map (从 noise estimate 和 signal 计算的局部 SNR), 用来告诉主分支哪里可信、哪里需要 aggressive denoise。

#### 5.4.4 训练数据

不用 challenge 数据训主网络, 用 LSDIR [35] 大数据集通过 unprocessing [7] 合成 pseudo-RAW + clipped Gaussian noise。然后用 challenge 的 color chart scene 估参数, 应用到 test scene。这是典型 blind denoising 思路。

#### 5.4.5 对 dataset 的批判

VMCL-ISP 给出了一个有价值的诊断 (Figure 12): challenge 的 GT 在 low-light 下仍然有 residual noise、defective pixels、black level 错误导致的 color bias。他们指出 data 被 clip 了负值, 导致 black level 估计困难。这是对 SIDD-style averaging 协议的实质性批评, 后续 challenge 应该会改进。

参考 YOND: https://arxiv.org/abs/2506.03645
参考 PMN: https://ieeexplore.ieee.org/document/10136126

---

## 6. Cross-sensor & cross-condition 分析 (Table 4, Figure 5)

### 6.1 Per-sensor PSNR (Table 4 关键数字)

挑几个有意思的:
- Pixel 7 Pro Telephoto (sensor 极端, max ISO 1143): SNU-ISPL 53.00, XJAI 53.30, 但 AxeraAI 52.51, VMCL-ISP 52.48——四方法都很高, 因为这个 sensor ISO 上限低, 实际 noise 不算大, 不同方法差距小
- Pixel 5a Wide: SNU-ISPL 43.16, XJAI 43.01, AxeraAI 41.19——single-frame 差 2 dB, 这 sensor 上 multi-frame 优势明显
- Z Fold4 Telephoto (max ISO 10000): SNU-ISPL 46.15, XJAI 47.60, AxeraAI 46.04——XJAI 反超, 说明它的 hierarchical transformer 对 extreme noise 更鲁棒
- POCO X3 Pro Wide: SNU-ISPL 45.53, XJAI 47.01——XJAI 显著好, 可能 POCO 这颗 sensor 的 noise 统计 XJAI 的 inductive bias 更匹配

直觉: single-frame (AxeraAI) 在 ISO 受限的 sensor 上能匹敌 multi-frame, 但 ISO 上限高的 sensor (extreme noise) 上明显劣于 multi-frame。

### 6.2 Per-condition (Figure 5)

- **Bright (10 lx, 1/24 s)**: 所有方法 PSNR 接近, multi-frame gain < 0.5 dB。SNR 已经高, 单帧就能 denoise 干净, temporal 帮不上忙。
- **Mid (5 lx, 1/60 s 等)**: gap 开始拉大, multi-frame 领先 1~1.5 dB
- **Extreme (1 lx, 1/120 s)**: top multi-frame 比 best single-frame 高 **3 dB 以上**。这是 SNR 极低时, 单帧 spatial denoise 会 oversmooth 或留 noise, 而 multi-frame 能从多帧里"投票"出真实 signal。

这个 3 dB 的数字非常关键——它告诉你 multi-frame 在 hardest case 上几乎能再造一倍 SNR ($10^{3/10} \approx 2$)。

---

## 7. 关于 evaluation 的注意点

### 7.1 PSNR / SSIM on linear mosaicked RAW

PSNR:
$$\text{PSNR} = 10 \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$

MAX=1 (因为 RAW 已经 normalize 到 [0,1])。MSE 直接在 4-channel Bayer-packed domain 算。

SSIM:
$$\text{SSIM}(x,y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

- $\mu_x, \mu_y$: local mean of $x$ and $y$
- $\sigma_x^2, \sigma_y^2, \sigma_{xy}$: local variance and covariance
- $C_1 = (0.01 L)^2, C_2 = (0.03 L)^2$: stabilizer constants, $L=1$ here

注意: paper 用 MS-SSIM [52], multi-scale 5 levels, 但 evaluation 报告的是标准 SSIM。

### 7.2 Ranking = mean of per-metric ranks

不是 mean PSNR/SSIM, 是 rank 的平均。这样如果 SSIM 和 PSNR 顺序不一致, 折中。SNU-ISPL PSNR 第 1 + SSIM 第 1 = 平均 1, XJAI 是 2+2=2, 完全 tie-break 清晰。

### 7.3 Submit 格式

Center 1024×1024 crop, preserve Bayer pattern。这避免了边界 artifact 影响 metric, 也限制了 test 区域为 image center (一般 lens center optical quality 最好)。

---

## 8. Baseline 的价值

UNet (Noise2Noise) 和 UNet+Attn 是 organizer 自己跑的 baseline:

- **UNet single-frame**: 43.52 dB / 0.9691, 比 AxeraAI NAFNet 低 3 dB。说明网络架构 (NAFNet 的 SimpleGate + SCA) 在 single-frame 设置下确实比 vanilla UNet 强很多。
- **UNet + channel-attention 多帧 fusion**: 45.72 dB / 0.9797, 比 single-frame UNet 高 2.2 dB, 跟 top multi-frame SOTA 差 2.6 dB。说明 alignment-aware 设计 (Burstormer-based) + conditioning 比朴素 concat 多帧 fusion 强得多。

参考 Noise2Noise: https://arxiv.org/abs/1803.04189

---

## 9. 我 (作为一个 reader) 的几点 take-aways

1. **Multi-frame gain 在 extreme low-light 是实在的 3 dB**: 这不是边际改进, 是接近翻倍 SNR。任何 production low-light pipeline 都应该考虑 burst fusion。

2. **Alignment 是 multi-frame 的瓶颈**: AxeraAI 直接 concat 多帧失败, 说明 alignment-aware (Burstormer, deformable attn) 是必备组件。

3. **Capture-condition conditioning 是新方向**: DarkVRAI 用 sensor / lux / fps 通过 adaLN 注入, 在 cross-sensor / cross-condition generalization 上确实稳。这暗示 future 的 denoise model 应该是 condition-aware 的, 而非 fixed。

4. **Mamba 在 video temporal axis 上 work**: BOSS module 用 selective scan 替代 temporal attention, 在 10 帧 burst 上线性复杂度优势不大, 但更可能是 inductive bias 更匹配 sequential nature of burst。

5. **GT quality 是 future bottleneck**: VMCL-ISP 指出 SIDD-style averaging 在 extreme low-light 下 GT 仍有 noise, PSNR 48+ 的数字可能被 GT noise cap 住。Future dataset 可能需要更长 burst (1000+ 帧) 或 beam-splitter 真正 high-SNR。

6. **Single-frame NAFNet 仍是非常强的 baseline**: 在 SNR 不极端的场景下能逼近 multi-frame, 这对资源受限的 edge deployment 是好消息。

7. **数据协议的精细之处**: stop-motion rail + ISO normalization + frame-rate emulation 这套组合拳, 让 dataset 既 alignment-precise 又 physically realistic, 是值得借鉴的 dataset engineering 范式。

---

## 10. 后续值得 follow 的工作

- **YOND [22]** (VMCL-ISP 主方法): blind RAW denoising, 跳出 camera-specific data 依赖
- **PMN [21]** (Feng et al., TPAMI 2024): data quality 对 denoise learnability 的影响
- **MambaIR [25]**: state space model for image restoration, BOSS 的源头
- **Burstormer [19]**: burst restoration 的 deformable attention 范式
- **Restormer [59]**: transposed attention 高分辨率 restoration 的标准 backbone
- **NAFNet [12]**: simple baseline 范式, 启发后续大量 activation-free 设计
- **RVideFormer [57]**: RAW video denoising 专属 transformer
- **BVI-LowLight [4]**: 真实 low-light video benchmark, beam-splitter rig
- **Dancing in the Dark [23]**: mechatronic alignment 的 low-light video dataset

paper 本身的官方页面应该在 AIM 2025 workshop 上, 关联 challenges 链接在 https://www.cvlai.net/aim/2025/ 。

如果你 (Andrej) 想进一步 build intuition on 这个领域的 next step, 我会建议关注三个方向: (1) implicit neural representation 替代显式 alignment; (2) diffusion-based RAW denoise (用 score matching 处理 signal-dependent noise); (3) cross-sensor generalization via noise-equivariant embedding (类似 domain adaptation 思路, 让 model 学到 sensor-invariant feature)。
