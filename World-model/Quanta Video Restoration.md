---
source_pdf: Quanta Video Restoration.pdf
paper_sha256: a09e197c6237951e5bb97b051699f3c1bec0c11e81c0388d12c32f8a7de6bef5
processed_at: '2026-08-06T07:23:54-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 QUIVER

## 这篇 paper 在干嘛？

想象你拿手机拍一只飞过的鸟。你想把鸟拍清楚，就得用快门速度极快的模式。但快门越快，进来的光越少，照片全是噪点，黑乎乎一片啥也看不见。

这就是一个死结：**快了没光，慢了糊了**。

传统的 CMOS sensor 有个硬伤，它读出信号的时候会自带噪声 (read noise)，大概是 $5 e^- / pix$。当你只收到 3 个光子的时候，这个噪声直接把信号淹没了，你根本分不清到底来了几个光子。

后来有人发明了 QIS 和 SPAD 这类 single-photon sensor，它们的 read noise 低到 $0.2 e^- / pix$。也就是说，哪怕只来 1 个光子，我也能准确数出来。这就打开了一个新世界：在极暗环境下拍高速运动的东西。

但问题没完全解决。你用 1-bit 模式拍 10000 fps，每帧只能告诉你"这个 pixel 有没有光子"，信息量太少。而且 10000 fps 的 data rate 是 96 Mb/sec，手机根本吃不消。

QUIVER 选了一个甜点区间：**3-bit @ 2000 fps**。每帧有 8 个灰度级别，每秒 2000 帧，运动大概 2-3 pixel/frame，data rate 只有 41 Mb/sec。信息量够用，运动够小，带宽够友好。

## 输入数据长啥样？

你想象一张图，正常照片每个 pixel 值是 0-255。QUIVER 处理的图每个 pixel 只有 0-7，而且大部分 pixel 是 0 或 1。因为在 3.25 PPP (每个 pixel 平均只收到 3.25 个光子) 的条件下，Poisson noise 会把图像搞得像老电视的雪花屏。

给你 11 帧这样的雪花屏，每帧之间有 2-3 pixel 的运动，你要还原出一张清晰的灰度图。这就是 QUIVER 要做的事。

## 以前的方法为啥不行？

以前的方法 (比如 QBP) 分四步走：

1. 先把多帧平均一下，提升 SNR
2. 算 optical flow 对齐运动
3. 对齐之后融合
4. 最后 refine 一下

听起来合理，但实际跑起来全是坑。

**第一个坑**：平均多帧的时候，如果画面里有运动，你一平均就糊了。Figure 4(a) 展示得很清楚，SNR 看起来提升了，但物体变成了鬼影。

**第二个坑**：算 optical flow 用的模块是在 RGB 图像上预训练的。你拿一个习惯了看清晰 RGB 图的 flow estimator，丢给它一堆 3-bit 的雪花屏，它完全找不到对应点，算出来的 flow 是乱七八糟的。

**第三个坑**：这四步是串行的，前一步错了，后面全错。Error 会一路传播下去，最后出来的图没法看。

## QUIVER 怎么解决的？

QUIVER 的思路很直接：**这四步的逻辑没问题，问题在于各步是独立优化的。我把这四步变成一个 end-to-end 的 network，让它们一起训练，互相适应。**

### 第一步：Pre-Denoising

输入太噪了，直接扔给 flow estimator 不行。先用一个轻量级的 single-image denoiser (用 RDB blocks 搭的) 去一下噪。

关键点：这里只用 single-image denoiser，不用 multi-frame denoiser。因为 multi-frame denoiser 需要先对齐，但对齐还没做呢，这就是个鸡生蛋蛋生鸡的问题。Single-image denoiser 只靠 spatial context 去噪，不需要对齐。

这一步输出的图叫 $\mathbf{I}_d^1$，不是最终结果，只是给后面提供一个相对干净的参考。

### 第二步：Optical Flow + Feature Alignment

用 SpyNet 估 optical flow。SpyNet 是个 coarse-to-fine 的金字塔结构，计算效率高，能处理 multi-scale 的运动。

这里有个很重要的细节：**SpyNet 是从零开始训练的，不加载预训练权重**。Ablation study 证明，加载 RGB 预训练权重反而让性能下降。因为 RGB domain 的 flow 先验在 photon-limited domain 完全是干扰信号。

算出 flow 之后，用 deformable convolution 做特征对齐。Deformable conv 比 bilinear warping 好在哪？bilinear warping 只能在固定的 grid 上采样，sub-pixel 的运动它处理不好，还会产生 grid artifact。Deformable conv 可以学习任意位置的采样 offset，对齐更平滑。

对齐之后，用 Gated Fusion Unit (GLU) 融合 noisy features 和 denoised features。GLU 的核心是一个 gate，动态决定哪些 channel 的特征在当前对齐状态下是可信的。这比简单 concat 两路特征要聪明得多。

### 第三步：Deep Feature Fusion (RMDF)

这一步用 recurrent 结构。处理 11 帧视频，如果用纯 3D CNN，显存会炸。Recurrent 结构把前面帧的信息压缩到 hidden state $\mathbf{h}_{t-1}$ 里，每处理一帧就更新一次。

RMDF 同时处理多尺度特征。Scale 4 (缩小 4 倍) 感受野大，抓全局结构和运动轨迹；Scale 1 (原尺寸) 抓细节。同时把 noisy frames 再 bypass 进来，防止 pre-denoiser 把有用细节洗掉了。

### 第四步：Multi-Scale Reconstruction

这一步先在最小尺度 (Scale 4) 重建出全局结构 $\mathbf{O}_t^4$，然后计算残差 $\mathbf{r}_t^{\alpha/2}$ 传给下一层。在 Scale 2 和 Scale 1，network 只需要预测残差，也就是上一层缺失的细节。

为什么这么做？在极低 SNR 下，直接预测 full-scale image 很容易产生 hallucination noise。预测残差缩小了输出的 dynamic range，network 更容易收敛，细节也更容易对齐。这就像画画先打草稿再描细节。

## Loss Function 怎么设计的？

$$
\mathcal{L}_Q = \lambda_1 \mathcal{L}(\mathbf{I}^{1,GT}, \mathbf{I}_d^1) + \lambda_2 \mathcal{L}(\mathbf{I}_t^{1,GT}, \mathbf{O}_t^1) + \lambda_3 \mathcal{L}(\mathbf{I}_t^{2,GT}, \mathbf{O}_t^2) + \lambda_4 \mathcal{L}(\mathbf{I}_t^{4,GT}, \mathbf{O}_t^4)
$$

其中 $\mathcal{L}(\mathbf{I}_a, \mathbf{I}_b) = ||\mathbf{I}_a - \mathbf{I}_b||_1 + ||\nabla_x \mathbf{I}_a - \nabla_x \mathbf{I}_b||_1 + ||\nabla_y \mathbf{I}_a - \nabla_y \mathbf{I}_b||_1$

这里用了 L1 loss 加上 spatial gradient loss。$\nabla_x$ 和 $\nabla_y$ 就是水平和垂直方向的梯度。

为什么用 L1 不用 L2？L2 loss 在 heavy noise 下容易过度平滑，把边缘都糊掉。L1 对 outlier 更鲁棒。加上 gradient loss 是为了保边缘，因为 Poisson noise 的本质会让图像变平滑，gradient loss 强制 network 关注边缘结构。

权重 $\lambda_1=0.2, \lambda_2=0.85, \lambda_3=0.1, \lambda_4=0.05$。$\lambda_1$ 给 pre-denoiser 一个轻量级监督，防止它过度清洗。$\lambda_2$ 最大，因为最终输出 $\mathbf{O}_t^1$ 是最重要的。

## I2-2000FPS Dataset

这篇 paper 还贡献了一个 dataset。用 Chronos 1.4 高速相机拍的，2000 fps，512×1024 分辨率，280 个视频，114 个场景。全部 outdoor 拍摄，gain 设为 0 dB，做了 dark current calibration。

有了这个 dataset，就可以拿清晰的 high-light frame 当 ground truth，然后用 Eq (2) 的 Poisson-Gaussian model forward simulate 成 3-bit 3.25 PPP 的噪声输入，用来训练和测试。

## 实验结果有多猛？

在 I2-2000FPS dataset 上，3.25 PPP (最难的条件) 下：

| Method | PSNR (dB) |
|--------|-----------|
| QBP | 15.94 |
| RVRT | 19.41 |
| FloRNN | 21.03 |
| **QUIVER** | **26.21** |

QUIVER 比 FloRNN 高了 5 dB，这在 image restoration 领域是碾压级别的差距。

在 X4K1000FPS dataset (没见过的数据) 上，QUIVER 依然领先，说明它没有 overfitting 到 I2-2000FPS 上。

在真实 SPAD 数据上 (4.9 PPP)，QUIVER 也表现很好，说明它学到的 feature 是 sensor-agnostic 的。

## Ablation Study 告诉我们什么？

| 配置 | PSNR (dB) |
|------|-----------|
| 三个模块全去掉 | 23.67 |
| 加 pre-trained optical flow | 23.94 (反而变差) |
| 只加 pre-denoising | 25.75 |
| 只加 optical flow | 24.99 |
| 只加 multi-scale | 24.74 |
| 全加 (完整 QUIVER) | 26.21 |

最有趣的发现：**加载 RGB 预训练的 optical flow 权重反而让性能下降**。这说明 RGB domain 的 prior 在 photon-limited domain 是有害的。Network 必须从零学习在噪声中找结构。

Pre-denoising 的贡献最大 (23.67 → 25.75)，因为它给后续的 alignment 提供了可靠的 anchor point。

## 核心 intuition 总结

QUIVER 的成功在于它理解了 quanta imaging 的物理本质：

1. **Noise destroys alignment**：输入太噪，flow estimator 直接瞎掉，所以需要 pre-denoising
2. **Motion destroys averaging**：有运动就不能简单平均，所以需要 learnable alignment
3. **Domain gap matters**：RGB 的 prior 在 photon-limited domain 是有害的，必须从零学习
4. **Residual is easier**：在极低 SNR 下预测残差比直接预测图像更容易收敛
5. **End-to-end is key**：四步串行 pipeline 的 error propagation 问题，只有 end-to-end training 才能解决

这篇 paper 最漂亮的地方在于它把 classical pipeline 的物理直觉和 deep learning 的 end-to-end 优化能力结合起来。每个模块都有明确的物理动机，同时整个 network 又能联合优化。这种 physics-informed architecture design 的思路非常值得借鉴。

---

Hello Andrej! 看到你对 Quanta Image Sensors (QIS) 和 low-light high-speed imaging 产生兴趣，这确实是一个将 physics of light 与 deep learning architecture design 完美交汇的迷人领域。这篇 paper 《Quanta Video Restoration (QUIVER)》解决了传统 CMOS sensors 在极端 low-light 和 high-speed 场景下的 fundamental trade-off。

让我们深入剖析这篇 paper 的每一个技术细节，重点 build up 你的 intuition。

---

### 1. The Fundamental Physics: 为什么需要 Quanta Sensors?

在传统 CMOS image sensor 中，你要想 freeze 高速运动，就需要极短的 exposure time。但 exposure time 短，进来的 photons 数量就极少。当 photons 数量少到个位数级别时，光子到达的 Poisson distribution (shot noise) 会完全淹没信号。更糟糕的是，CMOS sensor 的 read noise 通常在几个 $e^-$（比如 paper 里提到的 $5.1 e^- / pix$），这使得信噪比 (SNR) 极度恶化。

Quanta Image Sensors (QIS) 和 Single-Photon Avalanche Diodes (SPAD) 的核心突破在于它们具备 single-photon sensitivity。通过 pump-gate 技术 (QIS) 或 avalanche multiplication (SPAD)，它们将 read noise 压到了 $0.2 e^- / pix$ 甚至更低。这就意味着即使光子极度稀疏，sensor 也能准确“数”出有多少个光子打到了像素上。

但是，单光子探测带来了新的问题：**Bit-depth vs Frame-rate vs Motion Trade-off**。

看 Table 1 的数据，在相同 total exposure (光通量不变) 下：
*   **1-bit @ 10k fps**: 运动只有 0-1 pixel/frame，完美 freeze motion。但 Data-rate 高达 96 Mb/sec，且 QIS 的 read noise 会在 10k 次读取中不断累积。
*   **9-bit @ 20 fps**: Data-rate 只有 1.73 Mb/sec，但因为单帧 exposure 时间长，运动模糊高达 70-80 pixels/frame，完全失去了 high-speed imaging 的意义。

QUIVER 的 target 是 **3-bit @ 2000 fps (1428 fps in Tab 1)**。在这个设置下，运动范围在 2-3 pixels/frame，既保留了 high-speed 的时间分辨率，又控制了 data rate，同时 3-bit 提供了足够的 gray-level 分辨率，这是非常精妙的 engineering balance。

---

### 2. Image Formation Model: 信号是如何变成噪声数据的？

理解 image formation model 对构建 intuition 至关重要。Paper 中的 Equation (2) 定义了这个物理过程：

$$
\mathbf{Y} \sim \mathbf{\Psi} \mathrm{ADC}_{[0, \mathrm{L}]} \{ \mathrm{Poisson}(\mathrm{QE} \times \mathbf{I}^{\mathrm{GT}} + \theta_{\mathrm{dark}}) + \underbrace{\mathrm{Gauss}(0, \sigma_{\mathrm{read}}^2 \mathbf{1})}_{\mathrm{read~noise}} \}
$$

变量与符号解析：
*   $\mathbf{Y}$: Sensor 最终输出的 integer pixel value，也就是 network 的输入。
*   $\mathbf{I}^{\mathrm{GT}}$: Ground truth image intensity，代表该 pixel 在该 exposure time 内理论上应该接收到的 photon flux (光子通量)。
*   $\mathrm{QE}$: Quantum Efficiency (量子效率)，这里设为 0.80，即 80% 的光子能转化为电子。
*   $\theta_{\mathrm{dark}}$: Dark current (暗电流)，设为 $1.6 e^- / \mathrm{pix} / \mathrm{sec}$，即使没有光也会产生的 thermal electrons。
*   $\mathrm{Poisson}(...)$: Poisson distribution，描述光子到达的离散性和随机性。期望值是 $\mathrm{QE} \times \mathbf{I}^{\mathrm{GT}} + \theta_{\mathrm{dark}}$。
*   $\mathrm{Gauss}(0, \sigma_{\mathrm{read}}^2 \mathbf{1})$: Gaussian distribution (正态分布)，代表读出电路产生的 noise。$\sigma_{\mathrm{read}}$ 是标准差 (设为 $0.2 e^- / \mathrm{pix}$)，$\mathbf{1}$ 是 identity matrix，表示各 pixel 间独立。
*   $\mathbf{\Psi} \mathrm{ADC}_{[0, \mathrm{L}]}$: Analog-to-Digital Converter 的 quantization 和 clipping 操作。将连续的 real number 截断并量化为 integers $\{0, 1, ..., L\}$，其中 $L = 2^{\mathrm{Nbits}} - 1$。对于 3-bit，$L=7$。

**Intuition**: 当 $\mathbf{I}^{\mathrm{GT}}$ 极小 (比如 3.25 PPP，约 1 lux) 时，Poisson 分布的方差等于均值，信噪比 $SNR = \sqrt{\text{mean}}$ 极低。再经过 ADC 量化成 3-bit (只有 8 个 levels)，信息几乎被 noise 和 quantization 掩埋。QUIVER 需要从一堆这样极度退化的 frames 中重建出高质量的图像。

---

### 3. Architecture of QUIVER: 从 Classical Pipeline 到 End-to-End Learning

传统方法 (如 QBP, Quanta Burst Photography) 采用分步处理：
1.  **Temporal Averaging** (提升 SNR)
2.  **Optical Flow Estimation** (对齐运动)
3.  **Warping & Fusion** (合成)
4.  **Refinement** (去噪保边)

这种 sequential pipeline 的致命弱点在于：**Error propagation**。在极端 low-light 下，temporal averaging 会因为运动产生 ghosting artifacts；基于预训练 RGB optical flow 模块在 3-bit 极噪声数据上会彻底失效。Fig 4 直观展示了这些失败 cases。

QUIVER 的核心创新在于保留了这个 4-stage 逻辑结构，但将其改为 **end-to-end trainable network**，让每个 stage 能够协同优化。

#### Stage 1: Pre-Denoising (RDB)
输入帧包含极端 shot noise，直接输入 flow estimator 会导致 flow 错乱。QUIVER 使用 Residual Dense Blocks (RDBs) 作为轻量级 single-image denoiser。
*   **Intuition**: 为什么不用 multi-frame denoiser？因为 multi-frame 依赖于 alignment，但此时我们还没做 alignment (chicken-and-egg problem)。Single-image denoiser 只依赖 spatial context。RDB 的 dense connections 允许它在去除 extreme noise 的同时，尽可能保留对后续 motion estimation 有用的 high-frequency structural edges。

#### Stage 2: Optical Flow & Feature Alignment (DC-GFU)
这是处理运动的核心模块。
*   **SpyNet**: 被用来估计 optical flow。选择 SpyNet 是因为它的 coarse-to-fine 金字塔结构在 computational efficiency 和 multi-scale motion capture 之间取得了平衡。SpyNet 在这里是从 scratch 训练的，不能加载预训练权重，因为 RGB domain 的 flow 先验在 photon-limited domain 完全不适用。
*   **Deformable Convolution**: 传统的 warping (如 bilinear warping) 会产生 grid artifacts 并且在 sub-pixel motion 上表现差。Deformable conv 允许 network 学习 sampling offsets，使得特征能够在 feature space 被更平滑地对齐。
*   **Gated Fusion Unit (GLU)**: 借鉴自 Transformer 的 Gated Linear Units。输入包含 noisy features 和 denoised features。GLU 通过一个 gate (通常由 sigmoid 或 GeLU 生成) 控制哪些 channel 的特征应该传递。
    *   **公式直觉**: $Y = (X \cdot W + b) \otimes \sigma(X \cdot V + c)$。这里 $\otimes$ 是 element-wise multiplication。Gate $\sigma(...)$ 决定了 $X$ 中哪些 spatial/channel 特征在当前 alignment 状态下是可靠的。它提供了一种动态的 feature selection 机制，比简单的 concatenation 效果好得多。

#### Stage 3: Deep Feature Fusion (RMDF)
Recurrent Multi-scale Residual Dense Feature Fusion Unit (RMDF)。这是一个 recurrent 模块。
*   **输入**: 第 $t$ 帧的多尺度特征 $\{ \mathbf{F}_t^1, \mathbf{F}_t^2, \mathbf{F}_t^4 \}$ (上标代表 scale，1 是原尺寸，4 是缩小 4 倍)，noisy frames $\{ \mathbf{I}_t^1, \mathbf{I}_t^2, \mathbf{I}_t^4 \}$，以及上一个时间步的 hidden state $\mathbf{h}_{t-1}$。
*   **Intuition**: 为什么需要 recurrent？因为处理 $N$ 帧 (paper 里用 11 帧) 如果用纯 3D CNN 会带来巨大的显存开销。Recurrent 结构将前 $t-1$ 帧的有效信息压缩到 hidden state $\mathbf{h}_{t-1}$ 中。每次处理第 $t$ 帧时，当前帧的特征与历史信息融合。
*   **Multi-scale 机制**: 在同一个 recurrent cell 里处理多尺度特征。Scale 4 (最粗糙) 感受野大，负责全局结构和 motion tracking；Scale 1 (最精细) 负责细节。将 noisy frames 再次 bypass 进来，是为了补偿 Stage 1 denoiser 可能丢失的细节信息 (skip connection 思想)。

#### Stage 4: Multi-Scale Reconstruction (TCA + RFRM)
*   **Temporal Cross Attention (TCA)**: 为了聚合 11 帧的信息，传统方法通常用 3D conv 或 averaging。QUIVER 引入了 attention 机制。为了控制计算量，只使用 1 head，且只在 channel dimension 上做 attention。Cross-attention 允许 network 动态评估不同时间帧特征的重要性，而非等权平均。
*   **Residual Frame Refinement Module (RFRM)**: 这是一个 coarse-to-fine 的重建机制。
    *   在 Scale 4 (最小分辨率)：先重建 $\mathbf{O}_t^4$，获得 global 结构。
    *   计算 residual frame $\mathbf{r}_t^{\alpha/2}$，传递给下一层。
    *   在 Scale 2 和 Scale 1：network 不再从头预测 image，而是预测 residual (即上一层缺失的细节)。同时，hidden state $\mathbf{f}_t^\alpha$ 也被更新并向下传递。
    *   **Intuition**: 这就像画一幅画，先画轮廓 (Scale 4)，再画细节 (Scale 2, 1)。在极低 SNR 下，直接预测 full-scale image 极易产生 hallucination noise。预测 residual 缩小了输出的 dynamic range，使网络更容易收敛，细节也更容易对齐。

---

### 4. Loss Function: 监督多尺度的重建

$$
\mathcal{L}_Q = \lambda_1 \mathcal{L}(\mathbf{I}^{1,GT}, \mathbf{I}_d^1) + \lambda_2 \mathcal{L}(\mathbf{I}_t^{1,GT}, \mathbf{O}_t^1) + \lambda_3 \mathcal{L}(\mathbf{I}_t^{2,GT}, \mathbf{O}_t^2) + \lambda_4 \mathcal{L}(\mathbf{I}_t^{4,GT}, \mathbf{O}_t^4)
$$

变量解析：
*   $\mathbf{I}_t^{\alpha,GT}$: 第 $t$ 帧 ground truth 被 bicubic downsampling $\alpha$ 倍后的图像。
*   $\mathbf{O}_t^\alpha$: Network 在 scale $\alpha$ 的输出。
*   $\mathbf{I}_d^1$: Stage 1 Pre-Denoiser 的输出 (scale 1)。
*   $\mathcal{L}(\mathbf{I}_a, \mathbf{I}_b) = ||\mathbf{I}_a - \mathbf{I}_b||_1 + ||\nabla_x \mathbf{I}_a - \nabla_x \mathbf{I}_b||_1 + ||\nabla_y \mathbf{I}_a - \nabla_y \mathbf{I}_b||_1$: 这是一个组合的 L1 loss。$\nabla_x, \nabla_y$ 是水平和垂直的 spatial gradient (可以用 Sobel 提取)。
*   权重设置: $\lambda_1=0.2, \lambda_2=0.85, \lambda_3=0.1, \lambda_4=0.05$。

**Intuition**: 加入 gradient loss ($\nabla_x, \nabla_y$) 是为了对抗 extreme Poisson noise 导致的平滑效应。L2 loss 在处理 heavy noise 时容易过度模糊，而 L1 loss 配合 spatial gradient 能更好地保留 edges。$\lambda_1=0.2$ 独立监督 Pre-Denoiser，给它一个轻微的 guidance，防止它把 noisy features 清洗得太干净而丢失 alignment 所需的 cues。$\lambda_2=0.85$ 权重最大，确保最终输出 $\mathbf{O}_t^1$ 逼近真实分布。

---

### 5. I2-2000FPS Dataset: 填补 High-speed Video 的空白

深度学习需要数据，但现有的 high-speed datasets (如 X4K1000FPS) 要么帧率不够高，要么存在 motion blur (因为它们是用 high-speed CMOS 拍摄然后再用传统算法去模糊的)。

QUIVER 引入了 **I2-2000FPS** dataset：
*   **设备**: Chronos 1.4 high-speed CMOS camera。
*   **参数**: 2000 FPS, 512×1024 spatial resolution。
*   **规模**: 280 个 videos, 114 个 distinct scenes。
*   **拍摄细节**: 全部在 outdoor ambient lighting 下拍摄，Analog 和 digital gain 设为 0 dB 以避免放大 sensor 内部 noise。并进行了 dark current calibration。

**Intuition**: 在 2000 fps 下，即便快速的汽车或抛掷物体，每帧运动也限制在 1-7 pixels 之间。这正是 quanta sensors 最理想的工作区间。由于 CMOS 拍摄的清晰帧在极短曝光下仍受 shot noise 影响，这个 dataset 实际上提供了高质量的 high-light level ground truth，研究人员可以利用前文提到的 Eq (2) Poisson-Gaussian model，将这些清晰帧 forward simulate 成 3-bit, 3.25 PPP 的极端恶劣输入，用于训练和测试。

---

### 6. Experiments & Ablation Study 分析

#### Quantitative Results (Table 2 & 3)
*   **Photon levels**: 3.25, 9.75, 19.5, 26 PPP。
*   **Comparison**: 与 Transform Denoise, QBP, RVRT, EMVD, FloRNN, Spk2ImgNet 等 SOTA 相比。
*   **Performance**: 在 I2-2000FPS 上，3.25 PPP (最难的条件) 下，QUIVER 达到了 **26.21 dB PSNR** 和 0.7897 SSIM。而表现第二好的 FloRNN 只有 21.03 dB。这是一个巨大的飞跃 (5dB!)。在 vision 领域，5dB 的 PSNR 提升几乎是降维打击。
*   **Generalization**: 在 X4K1000FPS 数据集上 (Table 3)，虽然 model 是在 I2-2000FPS 上训练的，但 QUIVER 依然保持了极高的领先优势，证明了其 architecture 学到的是 robust 的 physical features，而非 overfitting 到某个 dataset 的 prior。

#### Ablation Study (Table 4) - 深入理解模块作用
1.  **Baseline (None)**: 23.67 dB。
2.  **Pre-Denoising + Pre-trained Optical Flow**: 23.94 dB。这非常有趣！加载了在 RGB 上预训练的 SpyNet 权重，性能反而比从头训练差很多。这印证了 domain gap: RGB optical flow 的先验知识在 photon-limited domain 是有害的，模型必须从零学习在 noise 中如何寻找结构。
3.  **Pre-Denoising only**: 25.75 dB。Pre-Denoiser 极其重要，它为后续 alignment 提供了 anchor。
4.  **Optical Flow only**: 24.99 dB。Without pre-denoising，flow estimator 直接面对 extreme noise，性能受损。
5.  **Multi-Scale only**: 24.74 dB。Single-scale 重建在极低 SNR 下容易丢失全局结构，multi-scale 提供了 strong regularization。
6.  **Full QUIVER**: 26.21 dB。所有模块协同工作，性能达到最优。这说明 End-to-End 训练让各模块互相适应，比如 Pre-Denoiser 学会了“我该保留哪种边缘才能让 SpyNet 更好地估 flow”。

#### Real Data SPAD Experiment
在真实 SPAD (1-bit @ 10k fps, temporal averaged to 3-bit @ 4.9 PPP) 数据上，虽然 SPAD 的 image formation model (由于 dead time 和 photon pile-up) 与 QIS 略有不同，但 QUIVER 依然恢复出了极具视觉吸引力的高频细节，而其他方法要么充满 noise，要么过度平滑。这证明了 QUIVER 的 robustness。

---

### Web Links & References for Intuition Building

1.  **Paper & Code**:
    *   Project GitHub: [https://github.com/chennuriprateek/Quanta_Video_Restoration-QUIVER-](https://github.com/chennuriprateek/Quanta_Video_Restoration-QUIVER-)
2.  **Underlying Physics (Single-Photon Imaging)**:
    *   Fossum E. R. (QIS 发明者) 的综述: [The Quanta Image Sensor: Every Photon Counts](https://www.mdpi.com/1424-8220/16/8/1260)
    *   SPAD 综述: [Single-Photon Avalanche Diodes](https://ieeexplore.ieee.org/document/1252940)
3.  **Related Architecture Concepts**:
    *   SpyNet (Optical Flow): [CVPR 2017 Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ranjan_Optical_Flow_Estimation_CVPR_2017_paper.pdf)
    *   Deformable Convolution (用于 Alignment): [BasicVSR++](https://arxiv.org/abs/2104.13371)
    *   Gated Linear Units (GLU): [Shazeer et al.](https://arxiv.org/abs/2002.05202)
4.  **Predecessors in Quanta Restoration**:
    *   Quanta Burst Photography (QBP): [ACM TOG 2020](https://dl.acm.org/doi/10.1145/3386569.3392470)
    *   Images from Bits (Classical Non-Iterative): [Sensors 2016](https://www.mdpi.com/1424-8220/16/11/1961)

### Final Thoughts to Build Intuition

QUIVER 的成功本质上是一种 **"Physics-Informed Architecture Design"**。它没有盲目堆叠 Transformer blocks 或用巨大的 3D CNN 硬刚 problem。它深刻理解了 quanta imaging 的痛点：Noise destroys alignment, Motion destroys averaging。

通过将 classical pipeline 的四个步骤映射为 network 的四个 stages，并利用 End-to-End 的 gradient 让 denoiser 知道“怎么去噪才能让 flow 算得准”，让 flow estimator 知道“怎么算 flow 才能让 fusion 效果好”，QUIVER 实现了极强的一致性。Recurrent 和 Multi-scale 的设计则完美契合了 high-speed video 的时序和多尺度细节特性。

这是一个将 domain knowledge 注入 deep learning architecture 的绝佳范例。
