---
source_pdf: Efficient Multi-Stage Video Denoising with Recurrent Spatio-Temporal Fusion
  Supplementary Materials.pdf
paper_sha256: ea229ddf3d3f7a57d65987a5bd74e4434232d029906accd969bcaf134584b23b
processed_at: '2026-08-04T01:46:46-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EMVD 用人话说

## 一句话讲完这篇 paper

华为那帮人想给手机拍的视频降噪。他们没像别人那样堆一个超级大的神经网络一次性吃 5 帧吐 1 帧，他们搞了个**很小**的网络，每帧来一次，递归地把"过去的干净版本"和"当前 noisy 帧"做加权融合，越往后越信任历史累积。整个网络只有 5.38 GFLOPs，跑出来跟几十 GFLOPs 的 RViDeNet 差不多甚至更好。

## 他们到底干了什么

我们慢慢拆。

### 问题本身

手机拍 raw 视频，每帧都是 Bayer pattern（4 个 channel: R, G1, G2, B），噪声是 signal-dependent 的（亮的地方噪声绝对值大，暗的地方噪声相对值大）。传统方法（VBM4D）假设噪声是均匀 Gaussian，所以在这上面不 work。现代方法（RViDeNet, EDVR）网络太大跑不动。EMVD 想要 **小 + 准**。

### 核心思路

他们观察到一个事实：**传统信号处理几十年来积累的 YUV 变换、小波分解、时域递归平均，已经是接近最优的先验了。深度学习只需要把这些算子做"可微分 + 微调"，不需要从零学**。

所以他们整个 pipeline 就是：

```
noisy raw → [color transform] → [frequency transform] →
  [fusion: 跟上一帧 fused 版融合] →
  [denoising: spatial 去噪] →
  [refinement: 把 fused 里的高频加回 denoised] →
  [inverse freq] → [inverse color] → clean raw
```

每一步都是 16 channel × 3×3 conv 的小网络，整体 5.38 GFLOPs。

### 三个 stage 各自在干啥

**Fusion stage**：拿当前帧的 color-frequency 表征，跟上一帧已经 fused 过的表征做加权融合。权重 $\gamma_t$ 是一个 $H/2 \times W/2$ 的 mask，由一个小 CNN 输出。

$$
\bar{y}_t = \gamma_t \odot \tilde{y}_t + (1 - \gamma_t) \odot \bar{y}_{t-1}
$$

直觉：静态区域 $\gamma_t$ 应该接近 $1/t$（让历史累积主导），动态区域 $\gamma_t$ 应该接近 1（只能信任当前帧）。

**Denoising stage**：把 fused 表征（已经时域降噪过，但还有残留噪声 + 动态区域没融合上）做 spatial 降噪。concat 了一堆辅助输入：当前帧 LL 子带（原始低频参考）、低尺度 denoised 后 inverse 回来的低频（多尺度 skip）、fused 帧的 variance map（递归算的）。

**Refinement stage**：denoising 阶段为了压噪会抹掉一些高频。但 fused 阶段在静态区域其实保留了干净的高频（时域平均把噪声消掉了）。Refinement 就是：

$$
y_{\text{refined}} = \rho \odot \bar{y}_t + (1 - \rho) \odot y_{\text{denoised}}
$$

$\rho$ 大 = 信任 fused 的高频，$\rho$ 小 = 信任 denoised 的低噪。

## 三个 stage 为什么这么分

他们没用 intermediate supervision（每个 stage 不告诉它该输出什么）。只有最终重建 loss。但训练完发现：

- Fusion network 的 sigmoid 输出 $\gamma_t$ 自动在背景上收敛到接近 $1/t$ 的行为。这跟 Kalman filter 里 posterior gain 的衰减形式完全一致。**Network 自己重新发明了 Kalman gain**。
- Denoising network 学到的是 conditional smoothing（在噪声大的地方更激进降噪）。
- Refinement network 的 $\rho$ 在帧数增加时越来越大——因为 fused 越来越干净，越来越多高频可以被"信任"。

这是 emergent behavior，非常 elegant。

## 那两个 transform 为什么聪明

**Color transform**: 用一个 $4 \times 4$ 矩阵把 Bayer 的 RGGB 变成 (Y, U, V, W) 类似 YUV。Y 是 luminance（能量集中），U/V/W 是 chrominance（能量小、噪声相对大）。这个矩阵初始化后可学习、可加 invertibility loss 约束。

**Frequency transform**: 用 Haar 小波做 stride-2 分解，把每个 color channel 分成 LL/LH/HL/HH 4 个子带，4 colors × 4 subbands = 16 channels。Haar 是 $2\times2$ 平均+差分，最简单的小波，但 ablation 显示它比 Daubechies、Biorthogonal 这些"更精细"的小波都好。

为什么 Haar 最好？因为 denoising 的瓶颈不是 frequency localization，是 spatial support。Haar kernel 小，对运动物体"涂抹"少；大 kernel 的小波虽然频响更尖锐，但 spatial 上更宽，碰到运动就 ghosting。

## 最有意思的 ablation

Table 2b 这个表特别精彩：

| Color init | Freq init | Learnable | Invertible | ∆PSNR |
|---|---|---|---|---|
| M | Haar | ✓ | ✓ | 0.0 |
| M | Haar | ✓ | ✗ | −0.10 |
| M | Haar | ✗ | ✓ | −0.11 |
| Random | Random | ✓ | ✗ | −3.45 |
| Random | Random | ✓ | ✓ | −0.77 |

读法：

1. 用 M+Haar 这种"正确先验"初始化，加不加 invertibility loss 都只差 0.1 dB。先验足够强，网络几乎不需要动它。
2. 全部随机初始化，没 invertibility loss，掉 3.45 dB——彻底坏掉。
3. 随机初始化 + invertibility loss，只掉 0.77 dB——loss 把网络拉回来了。

**关键洞察**：Fig. 4 显示，即使不加 invertibility loss，用 M+Haar 初始化后网络学到的 filter 依然接近 invertible。也就是说，**可逆性是重建任务本身的吸引子**，loss 隐式要求 transform 别丢信息。Invertibility loss 只在初始化糟糕的时候才救命，初始化好的时候是 redundant 的。

这个发现 deep learning 里其实反复出现：好的 inductive bias baked into architecture 比显式 regularizer 强得多。

## 噪声模型怎么来的

主文公式 (2) 是经典的 Gaussian-Poissonian:

$$
\hat{\sigma}^2(I) = a \cdot I + b
$$

$I$ 是 raw pixel intensity，$a$ 是 shot noise 系数（跟光子计数相关），$b$ 是 read noise 方差。这两个参数随 ISO 变化，需要 calibration：拍一组 flat-field 图，用 Median of Absolute Deviations 在 wavelet/DCT 高频域估噪声 std，对 intensity 回归出 $a, b$。

为什么用 transform domain 估？因为 spatial domain 的 local variance 包含真实信号的高频（边缘、纹理），会过估噪声。Transform domain 高频系数主要是噪声，信号能量低，所以 unbiased。

为什么在 raw 域做 denoising 而不在 sRGB 域？因为 sRGB 经过 tone-mapping 后噪声被非线性扭曲，从 sRGB 反推噪声统计几乎不可能。Raw 域噪声还保持原始的 Gaussian-Poissonian 形态，可建模、可分离、可递归传播。这点对 Tesla photon-counting pipeline 也是核心逻辑：处理越早越好。

## 递归方差传播非常 elegant

Fusion stage 输出的方差不是测量的，是**算出来**的：

$$
\hat{\sigma}_t^2(\text{fused}) = \gamma_t^2 \hat{\sigma}_t^2(\text{noisy}) + (1 - \gamma_t)^2 \hat{\sigma}_{t-1}^2(\text{fused})
$$

这是 noise 独立假设下方差的可加性。理想静态区域 $\gamma_t = 1/t$，于是 $\hat{\sigma}_t^2 \approx \frac{1}{t} \hat{\sigma}_{\text{noisy}}^2$，方差随 $1/t$ 衰减——经典 N-frame averaging 的极限。这个 variance map 又被喂给 denoising stage，告诉它"这里噪声还剩多少，要不要继续压"。

Fig. 2 里能直接看到：背景的 variance map 随帧数变蓝（噪声减小），动态区域保持红色（fusion 失效，噪声没消）。

## 跟你的 Tesla 视角的关系

Tesla 那条 "photon → raw → ISP → perception" 链路里，denoise 是关键瓶颈。EMVD 给的几个直接 transferable 的点：

1. **Invertible transform 是 free lunch**：固定一个 YUV+Haar 然后让网络只学 residual，tiny network 就能匹配大网络。Tesla 的 ISP 模块（demosaic、NR、sharpen）都可以套这个模板。
2. **Recurrent 而非 multi-frame stack**：FastDVDnet 需要 5 帧一起输入，memory 暴涨。EMVD 只存 16 channel × H/2 × W/2 的 recurrent state，恒定 memory，延迟恒定。对自动驾驶长时序场景是天然选择。
3. **No intermediate supervision**：所有 stage 共享最终 loss，network 自己学分工。这就是你反复说的"软件 2.0"——loss 来指挥，不是 hand-engineered pipeline 指挥。

## 一些深层联想

EMVD 的 fusion stage 本质上是一个 **spatially-gated RNN**。$\gamma_t$ 是 input gate，$1 - \gamma_t$ 是 forget gate，融合操作是 state update。这跟 GRU / LSTM / Mamba 的门机制同构，只不过 gate 是 spatial map 而非 vector。这也意味着 EMVD 可以无缝套到 SSM 框架里，理论上能处理更长的时序依赖。

更深一层：EMVD 的 fusion 是一个**像素级 Extended Kalman Filter 的可微版本**。Kalman 增益在静态情况下确实是 $1/t$ 形式（递归最小二乘的结论）。EMVD 让 network 自己学这个增益，但约束它必须是 sigmoid 输出（[0,1] 区间）+ spatial map。这是一个非常漂亮的 "neural + classical" hybrid 设计。

跟 Noise2Noise 也有血缘：Noise2Noise 说"多个独立噪声观测的均值收敛到 clean signal"，EMVD 的 fusion 是这个思想的 video + gated 版——不是简单平均，是 $\gamma$-weighted 累积，$\gamma$ 由 motion 检测调制。

## 局限性（论文没明说的）

- **大运动边界**：fusion 靠 sigmoid 软选择，大运动下 $\gamma \to 1$ 退化为单帧去噪，没有 optical flow 显式对齐会丢细节。
- **长时间依赖**：recurrent state 只存上一帧，没有 long-term memory。occlusion-reappearance 场景会忘事。LSTM-like memory 能解决但代价大。
- **ISO generalization**：noise model $(a, b)$ 是 ISO-dependent，需要 per-ISO 标定。换相机或改 ISP 就要重标。
- **Soft invertibility**：极端大噪声下学到的 transform 可能漂离可逆，出现信息泄露。Hard invertibility（coupling layer）更稳健但实现复杂。

## 你可以自己跑的 mini experiment

想 build intuition 的话，写个 30 行 PyTorch：

1. 生成 50 帧静态场景 + Gaussian 噪声。
2. 搭一个 fusion network：输入 abs(LL_t - LL_{t-1}) + variance map，输出 1 channel sigmoid $\gamma_t$。
3. 用 $\bar{y}_t = \gamma_t \odot y_t + (1 - \gamma_t) \odot \bar{y}_{t-1}$ 做 recurrent fusion。
4. Loss 就用 $\|\bar{y}_T - x_{\text{clean}}\|^2$，只监督最后一帧。
5. 训练几百 step 后，画 $\gamma_t$ 在背景区域的均值随 $t$ 的变化曲线。

我打赌你会看到 $\gamma_t$ 自动收敛到接近 $1/t$ 的曲线。这就是 EMVD 最 magic 的部分——网络没被告知该这么做，它从 loss 里自己发现了 Kalman gain 的最优形式。

## 总结

**EMVD 的全部美感**：作者深信传统信号处理的几十年 wisdom 已经接近最优先验，深度学习只需要做"residual refinement"。固定一个 YUV+Haar 初始化，搭一个 recurrent fusion + spatial denoise + high-freq refinement 的小 pipeline，最终一个 loss 端到端训练，网络自己学会 Kalman 增益、自己学会 refinement 何时启用、自己保持 transform 接近可逆。5.38 GFLOPs 打平几百 GFLOPs 的对手。

这是 "structured prior + learnable residual" 哲学在视频去噪上的胜利，跟你长期主张的"physics + classical priors baked in, network only learns the residual"完全合拍。

参考阅读：
- EMVD 主文: https://arxiv.org/abs/2207.14513
- VBM4D (Maggioni 老工作): https://ieeexplore.ieee.org/document/6209637
- FastDVDnet: https://arxiv.org/abs/1907.01314
- RViDeNet / CRVD: https://openaccess.thecvf.com/content_CVPR_2020/papers/Yue_Supervised_Raw_Video_Denoising_With_a_Benchmark_Dataset_on_CVPR_2020_paper.pdf
- Foi 噪声模型: https://ieeexplore.ieee.org/document/4606061
- Noise2Noise: https://arxiv.org/abs/1803.04189
- EDVR: https://arxiv.org/abs/1905.02716

---

# EMVD: Efficient Multi-Stage Video Denoising — Supplementary 深读

这篇 paper 是 Huawei Noah's Ark Lab 的 Matteo Maggioni（VBM4D [5] 的原作者，那篇是 BM4D 系列经典）带着做的，整体走的是 "传统 video denoising 洞察 + 现代 lightweight CNN 实现" 的路子。一句话总结：**用一套 learnable invertible transforms 把 raw Bayer 视频分解到 color-frequency 联合域，然后跑一个 recurrent 的 fusion → denoising → refinement 三阶段流水线，几乎不需要任何 intermediate supervision**，5.38 GFLOPs 就能在 CRVD [8] 上打平甚至超过 RViDeNet、EDVR 这些几十 GFLOPs 的大家伙。

这个工作的美感在于它把传统信号处理里的"YUV 变换 + 小波分解 + 时域递归融合"全部端到端可微化，让网络在初始化时就已经处于一个非常好的先验位置，剩下的学习只是 fine-tune 这些算子的系数。这点对你在 Tesla 长期关心的 "sensors → photons → ISP → perception" 整条链路特别相关——他们其实在做的就是可微 ISP 里的 denoise 模块。

---

## 1. Color Transform: 从 YUV 到 4-channel Bayer

公式 (1) 的矩阵 M：

$$
M = \begin{bmatrix}
0.5 & 0.5 & 0.5 & 0.5 \\
-0.5 & 0.5 & 0.5 & -0.5 \\
0.65 & 0.2784 & -0.2784 & -0.65 \\
-0.2784 & 0.65 & -0.65 & 0.2784
\end{bmatrix}
= \begin{bmatrix} Y \\ U \\ V \\ W \end{bmatrix} \tag{1}
$$

变量和上标下标讲解：
- 矩阵行向量 $(Y, U, V, W)$ 是变换后的四个 basis channels。
- 输入向量是 CFA Bayer 4 通道 $(R, G_1, G_2, B)$，对应 raw sensor 上一个 2×2 的 RGGB block 拉平成 4 维。
- $Y$ 行：$[0.5, 0.5, 0.5, 0.5]$，这是 energy-preserving average。**注意它不是归一化为 1**——0.5×4 = 2，但是 $\|\text{row}\|_2 = \sqrt{4 \times 0.25} = 1$，所以是 unit-norm。这意味着 $Y$ 大致是 $2 \times \overline{RG_1G_2B}$，幅度被放大约 2 倍以保持能量。
- $U, V, W$ 三行是色度通道，对应 RGB → YUV 里的 U/V 但扩展到 4D 颜色空间。它们的设计让矩阵接近正交（每行 unit norm，且互相接近正交）。
- 这个矩阵被用来初始化一个 $1 \times 1 \times C \times C$ 的 pointwise conv（C=4），训练时可以 learnable，可以加 invertibility loss。

**Intuition**: 为什么要先做 color transform？raw Bayer 域里 4 个 channel 高度相关（都是同一个场景的不同颜色采样），而 $Y$ 通道承载了绝大多数的 signal energy（luminance），$U/V/W$ 是 low-energy 的色度通道。做 denoising 时，对 $Y$ 和对 $U$ 应该用不同的"力度"——这点和 JPEG 把 Y 和 Cb/Cr 分别用不同量化是同一个道理。把这个先验 bake 进网络初始化，能让后面 convolution 学得更快、更稀疏。

参考链接：
- YUV transform basics: https://en.wikipedia.org/wiki/YUV
- Buades & Duran CFA video denoising (本文 ref [2]): https://ieeexplore.ieee.org/document/9001219

---

## 2. Frequency Transform: Learnable Wavelet Decomposition

频率变换用标准小波族初始化。每个 wavelet 有 4 个 1D 滤波器：
- $\psi_L$: decomposition low-pass
- $\psi_H$: decomposition high-pass
- $\phi_L$: reconstruction low-pass
- $\phi_H$: reconstruction high-pass

长度为 $n$（$n \in \mathbb{N}^+$，偶数）。2D 分解核通过外积构造，比如 LL 分解核：

$$
K_{LL} = \psi_L \otimes \psi_L
$$

其中 $\otimes$ 是 outer product，即 $K_{LL}[i,j] = \psi_L[i] \cdot \psi_L[j]$。同理有 $K_{LH} = \psi_L \otimes \psi_H$, $K_{HL} = \psi_H \otimes \psi_L$, $K_{HH} = \psi_H \otimes \psi_H$。这四个 $n \times n$ 卷积核拼起来就是一个 stride-2 的卷积，输出 4×C 通道（每个子带 C 通道，4 个子带）。本文里 $C=4$，所以输出 16 通道，对应 Table 1 中的 `StridedConv2D 2×2×4, 16 filters`。

**Haar 是最佳初始化**（Table 2a）：

| Wavelet | Kernel Size | ∆PSNR |
|---|---|---|
| Haar | 2×2 | 0.0 (baseline) |
| Symlets 2 | 4×4 | −0.01 |
| Daubechies 2 | 4×4 | −0.19 |
| Daubechies 3 | 6×6 | −0.09 |
| Biorthogonal 3.1 | 4×4 | −0.15 |
| Biorthogonal 1.3 | 6×6 | −0.05 |
| Rev. Biorthogonal 3.1 | 4×4 | −0.25 |
| Rev. Biorthogonal 1.3 | 6×6 | −0.08 |

Haar 虽然最小（2×2，4 个 tap），却给最高 PSNR。这里有个非常深的洞察：**对 denoising 来说，frequency localization 不是越尖锐越好**。Haar 的小 kernel 意味着 spatial support 小，对边缘和运动物体的"涂抹"伤害最小；而长 wavelet（6×6 的 Daubechies 3 / Rev. Bio 1.3）虽然有更平滑的 frequency response，但 spatial 上更宽，会在运动区域引入 ghosting。这跟传统视频降噪里"时域滤波窗口越短越能跟住运动"是一个道理。

Haar 的 LL kernel 就是 2×2 均值滤波 $\frac{1}{2}\begin{bmatrix}1&1\\1&1\end{bmatrix}$（归一化后），LH/HL/HH 是差分。所以 stride-2 的 Haar 分解等价于：低频 = 2×2 average pool，高频 = horizontal/vertical/diagonal detail。这恰好是 CNN 里 stride-2 conv 可以学到的最简单的 decomposition，所以从 Haar 初始化几乎不浪费 capacity。

参考链接：
- Wavelet families overview: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html
- Haar transform: https://en.wikipedia.org/wiki/Haar_wavelet

---

## 3. 整体架构 Table 1 逐步解读

让我把 Table 1 的每一行翻译成直觉。

### 输入 & 前变换

| Op | Kernel | Filters | Output | Comment |
|---|---|---|---|---|
| Input | — | — | $H \times W \times 4$ | 当前 noisy raw frame (Bayer) |
| Conv2D | $1\times1\times4$ | 4 | $H\times W\times4$ | Color transform $\mathcal{T}_c$ |
| StridedConv2D | $2\times2\times4$ | 16 | $H/2 \times W/2 \times 16$ | Frequency transform $\mathcal{T}_f$ (stride 2) |

注意 $2\times2\times4$ 的 kernel 实际上是 4 个独立的 $2\times2$ 卷积（每个对应一个 input channel），输出 16 通道里前 4 个是 LL 子带（对 4 个 color channel 各一个），中间 4 个是 LH，再 4 个 HL，最后 4 个 HH。这是把 4D color × 2D frequency 联合起来形成 16D 的 "color-frequency" representation。

### Fusion Stage

| Op | Kernel | Filters | Output | Comment |
|---|---|---|---|---|
| Sub+Abs | — | — | $H/2\times W/2\times4$ | 当前帧与上一帧 fused frame 的 LL 子带做差绝对值 |
| Concat | — | — | $H/2\times W/2\times6$ | diff(4) + upsampled lower-scale fusion weights(1) + variance map(1) |
| Conv2D+ReLU | $3\times3\times6$ | 16 | $H/2\times W/2\times16$ | 输入层 |
| Conv2D+ReLU | $3\times3\times16$ | 16 | $H/2\times W/2\times16$ | hidden (可重复) |
| Conv2D | $3\times3\times16$ | 1 | $H/2\times W/2\times1$ | fusion 输出 |
| Sigmoid | — | — | $H/2\times W/2\times1$ | fusion weights $\gamma_t \in [0,1]$ |
| Mul+Add | — | — | $H/2\times W/2\times16$ | 融合 |

融合公式（主文应该是这样）：
$$
\bar{y}_t = \gamma_t \odot \tilde{y}_t + (1 - \gamma_t) \odot \bar{y}_{t-1}
$$

变量：
- $\tilde{y}_t$：当前帧的 16-channel color-frequency 表示。
- $\bar{y}_{t-1}$：上一帧 fused 后的同一表示（recurrent state）。
- $\gamma_t$：pixel-wise 融合权重，$H/2 \times W/2 \times 1$，由 fusion network 输出经 sigmoid。
- $\bar{y}_t$：当前 fused 输出，喂给下一阶段，并作为 recurrent state 进下一帧。
- $\odot$ 是 broadcasted element-wise 乘法（$\gamma_t$ 从 1 channel broadcast 到 16 channels）。

**Fusion network 的输入为什么是这 6 个 channel？**

1. **LL abs-diff (4 channels)**：直接给出"哪里有运动"的信号。如果 LL 子带在两帧之间几乎不变，那这块区域可以放心做时域融合；如果差异大，就大概率是 motion 或 occlusion，要给当前帧更高权重。
2. **Upsampled fusion weights from lower scale (1 channel)**：让多尺度信息流向当前尺度——这是一种 coarse-to-fine 的 motion mask 先验。
3. **Variance map of noisy frame (1 channel)**：当前帧的噪声水平。高噪声区域应该更倾向于时域融合（信任历史），低噪声区域可以信任当前帧。

这套设计本质上是在用一个 3×3 CNN 来学习一个"motion + noise aware" 的 gate。它跟 FlowNet / DIS 的 optical flow 估计有相同的目标，但完全 skip 了显式的 flow regression，直接输出一个 soft mask。这跟你近期在 Tesla 演讲里强调的"不要显式估计光流，让网络隐式做对齐"思路一致。

### Denoising Stage

| Op | Kernel | Filters | Output | Comment |
|---|---|---|---|---|
| Concat | — | — | $H/2\times W/2\times25$ | fused(16) + LL current(4) + lower-scale denoised inverse-freq(4) + variance fused(1) |
| Conv2D+ReLU | $3\times3\times25$ | 16 | $H/2\times W/2\times16$ | 输入层 |
| Conv2D+ReLU | $3\times3\times16$ | 16 | $H/2\times W/2\times16$ | hidden |
| Conv2D | $3\times3\times16$ | 16 | $H/2\times W/2\times16$ | denoised output |

Concat 的 25 个 channel 含义：
- 16: fused representation（融合后的 color-frequency 表征）
- 4: 当前帧的 LL 子带（原始低频，没有经过 fusion，相当于"clean reference"）
- 4: 下一尺度的 denoised output 经 inverse frequency transform 后的低频（multi-scale skip）
- 1: fused frame 的 variance map（递归算出来的）

这里有个非常聪明的设计——**fused frame 的 variance map 不是直接测量的，是递归算出来的**。主文公式 (9) 应该是：

$$
\hat{\sigma}_t^2(\text{fused}) = \gamma_t^2 \cdot \hat{\sigma}_t^2(\text{noisy}) + (1 - \gamma_t)^2 \cdot \hat{\sigma}_{t-1}^2(\text{fused})
$$

变量：
- $\gamma_t$：pixel-wise fusion weight（标量，但 broadcast 到 spatial）。
- $\hat{\sigma}_t^2(\text{noisy})$：当前帧噪声方差，由 Gaussian-Poissonian model 给出（公式 2，下文详述）。
- $\hat{\sigma}_{t-1}^2(\text{fused})$：上一帧 fused 后的方差（递归量）。
- $\hat{\sigma}_t^2(\text{fused})$：当前 fused 后的方差。

这个公式假设 noise 在两帧之间独立，因此 variance 通过 weighted sum 传播。当 $\gamma_t = 1/t$（理想静态区域平均），$\hat{\sigma}_t^2(\text{fused}) \approx \frac{1}{t} \hat{\sigma}^2(\text{noisy})$，noise variance 随 $1/t$ 衰减——经典 $N$-frame averaging 的噪声 reduction 上限。Fig. 2 里展示了这一现象：背景（静态区域）的 variance map 随帧数递增而逐步变蓝（噪声方差减小），而动态区域（红色 fusion weights）的 variance 保持在高水平。

### Refinement Stage

| Op | Kernel | Filters | Output | Comment |
|---|---|---|---|---|
| Concat | — | — | $H/2\times W/2\times33$ | fusion(16) + denoised(16) + variance(1) |
| Conv2D+ReLU | $3\times3\times33$ | 16 | $H/2\times W/2\times16$ | 输入层 |
| Conv2D+ReLU | $3\times3\times16$ | 16 | $H/2\times W/2\times16$ | hidden |
| Conv2D | $3\times3\times16$ | 16 | $H/2\times W/2\times16$ | refinement output |
| Sigmoid | — | — | $H/2\times W/2\times1$ | refinement weights $\rho$ |
| Mul+Add | — | — | $H/2\times W/2\times16$ | refine denoised 用 fused + weights |

Refinement 公式：
$$
y_{\text{refined}} = \rho \odot \bar{y}_t + (1 - \rho) \odot y_{\text{denoised}}
$$

变量：
- $\bar{y}_t$：fused frame（含高频细节但有残留噪声，尤其在动态区域）。
- $y_{\text{denoised}}$：denoised output（噪声小但高频被过度平滑）。
- $\rho$：refinement weights，$H/2 \times W/2 \times 1$。

**Intuition**：denoising 阶段为了压噪会牺牲一些高频。但 fused 阶段保留的高频信息其实在静态区域是干净的（因为时域平均把噪声消掉了）。Refinement 就是把 fused 里的"高 SNR 高频"再混回去。$\rho$ 大的地方意味着"信任 fused 的高频"，$\rho$ 小的地方意味着"信任 denoised 的低噪声"。这个 stage 在 Fig. 5c 里非常明显——随着帧数增加，fused 越来越干净，refinement weights 越来越大（更多高频被加回），最终图像锐度提升。

### Inverse Transforms & Output

| Op | Kernel | Filters | Output |
|---|---|---|---|
| TransConv2D | $2\times2\times16$ | 4 | $H\times W\times4$ |
| Conv2D | $1\times1\times4$ | 4 | $H\times W\times4$ |
| Output | — | — | $H\times W\times4$ |

TransConv2D 是 inverse frequency transform，stride 2 的反操作（learnable unpool / sub-pixel）。注意它输入 16 channels、输出 4 channels，正好逆过来对应前向 StridedConv2D。然后 $1\times1$ 的 Conv2D 是 inverse color transform。这两个算子是 invertible transform 对的"另一半"，由 invertibility loss 约束它们真的是 $\mathcal{T}_c^{-1}, \mathcal{T}_f^{-1}$。

---

## 4. 递归 Fusion 的渐近行为（非常 elegant）

主文 Section 2 / Fig. 2 给出了一个直觉：

理想静态区域下，如果对前 $t$ 帧做平均，权重应该是 $\gamma_t = 1/t$（给当前帧），$\bar{\gamma}_{t-1} = 1 - 1/t$（给之前 fused 的累积）。

论文观察到 fusion network **没有任何显式监督**，但 sigmoid 输出的 $\gamma_t$ 在静态背景上自动收敛到接近 $1/t$ 的行为，即"早期帧给当前帧较大权重，越往后越信任历史累积"。当 $t \to \infty$，$\bar{\gamma}_{t-1} \gg \gamma_t$，fusion weights 在背景上趋向 0（蓝色），动态区域趋向 1（红色）。

这是一个 emergent behavior，跟 Bayesian recursive estimation 完全吻合——Kalman filter 里 posterior 的更新也是这种 $1/t$ 形式的 gain decay。论文没有显式约束这个，但训练后网络自己学到了。这点非常值得 build intuition：**只要 loss 是合理的重建 loss，network 通过 backprop 自己会发现最优的递归融合策略，不需要任何 intermediate supervision**。这正是 EMVD 训练简单的核心原因。

参考链接：
- Recursive Bayesian estimation: https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation
- Kalman filter intuition: https://www.kalmanfilter.net/default.aspx

---

## 5. Ablation: Learnable + Invertible 的作用

Table 2b 是这篇 supplementary 最有价值的部分。设定矩阵的行表示不同组合：

| $\mathcal{T}_c$ | $\mathcal{T}_f$ | Learnable | Invertible | ∆PSNR |
|---|---|---|---|---|
| M | Haar | ✓ | ✓ | 0.0 (baseline) |
| M | Haar | ✓ | ✗ | −0.10 |
| M | Haar | ✗ | ✓ | −0.11 |
| M | Random | ✓ | ✗ | −2.77 |
| Random | Haar | ✓ | ✓ | −0.45 |
| Random | Haar | ✓ | ✗ | −2.36 |
| Random | Random | ✓ | ✓ | −0.77 |
| Random | Random | ✓ | ✗ | −3.45 |

**关键观察**：

1. **Color transform 用 M 初始化比随机初始化重要得多**（−0.45 vs −2.77 的差距）。
2. **Frequency transform 用 Haar 比随机好，但差距没那么大**（−0.10 vs −2.36）。
3. **Invertibility loss 主要在 random init 的时候起作用**（−2.77 → −0.45 + invertibility + learnable；−2.36 → −0.77）。
4. **当用 M+Haar 初始化时，learnable 和 invertible 加起来只多 0.1 dB**——也就是说先验已经足够好，网络几乎不需要动它。

最深的洞察来自 Fig. 4：**即使没有 invertibility loss，当用 M+Haar 初始化时，学到的 filters 依然非常接近 invertible**。换句话说，invertibility 是一个"良好解的吸引子"，而不是必须强加的约束。这跟 deep learning 里很多 implicit regularization 的现象（比如 SGD 自动找 flat minimum）是同一类。这点你之前在 nanoGPT / "deep learning 手册"里强调过的"先验 baked into architecture 比显式 regularizer 更有效"完全呼应。

参考链接：
- Deep Image Prior (ref [4]): https://arxiv.org/abs/1711.10925
- Implicit regularization in DL: https://arxiv.org/abs/1704.08803

---

## 6. 噪声模型：Gaussian-Poissonian

主文公式 (2)（不在 supplementary 里但被引用），是 Foi et al. 的经典模型：

$$
\hat{\sigma}^2(I) = a \cdot I + b
$$

变量：
- $I$：像素 intensity（在 raw 域，未 demosaic）。
- $a$：shot noise 系数，跟光子计数相关，跟 ISO 反比关系。
- $b$：read noise 方差，跟电子学热噪声、 amplifier 噪声相关，跟 ISO 的关系更复杂。
- $\hat{\sigma}^2(I)$：给定 intensity 下的噪声方差。

这是一个 heteroscedastic noise model：噪声方差随 signal level 线性增长。这意味着 bright pixel 噪声大（绝对值），但 SNR 反而高（因为 SNR = signal / std = $\sqrt{I/a}$）。所以 denoising 在 bright 和 dark 区域应该用不同 strength。

估计方法（Section 3）：
1. 拍一组 calibration images（不同 ISO），里面包含 flat-field 区域。
2. 用 Median of Absolute Deviations (MAD) 估计 high-frequency transform 域 coefficient 的 scale。MAD 对 outlier 鲁棒，所以能在有边缘/纹理的区域工作。
3. 把估计到的 std 平方后，对 intensity 做 linear regression 得到 $(a, b)$。

为什么用 transform domain 而不是 spatial domain？因为 spatial domain 的 local variance 包含了真实信号的高频成分（边缘、纹理），会过估噪声。Transform domain（DCT / Wavelet）的高频系数主要承载噪声，信号能量低，所以估计 unbiased。

这套 noise model 在 raw 域特别重要，因为 sRGB 域经过 tone-mapping 后噪声已经被非线性扭曲，从 sRGB 反推噪声几乎不可能。这也是为什么 RViDeNet、EMVD 都坚持在 raw 域做 denoising。Tesla 的 photon-counting pipeline 也是类似逻辑：越早处理越好，因为噪声统计还保持原始形态。

参考链接：
- Foi et al. Poissonian-Gaussian model (ref [3]): https://ieeexplore.ieee.org/document/4606061
- Azzari & Foi MAD estimation (ref [1]): https://ieeexplore.ieee.org/document/6852053
- CRVD dataset: https://github.com/ChenyangLEI/RawVideoDenoising-Dataset

---

## 7. 与 SOTA 的对比

Supplementary 里没有给出 PSNR 表（在主文 Table 3），但从文字描述和 Fig. 6/7 可以推断：

- **VBM4D** (Matteo Maggioni 老本行) [5]: 传统 non-local 4D patch-based，对 Gaussian noise 设计。本文作者坦诚 sRGB 上跑 VBM4D 不理想（噪声不 i.i.d.），靠 grid search 优化 $\sigma$ 才能勉强 work。
- **FastDVDnet** [6]: U-Net-style 视频去噪，输入多帧。本文做了 complexity-reduced 版本（8 / 16 / 24 channels）。
- **EDVR** [7]: 视频超分/复原标杆，用 deformable conv + PCD alignment + multi-frame fusion。128 filters, 40 residual blocks。本文 reduce 到 48 filters。EDVR 的核心是显式 alignment，而 EMVD 不做显式 alignment——靠 fusion network 隐式学。
- **RViDeNet** [8]: CRVD dataset 的提出者，分 pre-denoise / alignment / fusion 三阶段，重训。

EMVD 5.38 GFLOPs vs RViDeNet 几百 GFLOPs，PSNR 反而接近或更好。这是一个 architecture efficiency 的胜利——**与其用大网络一次性处理多帧，不如用小网络 recurrent 处理一帧一帧**。这一点对你做 Tesla FSD 的延迟敏感 pipeline 应该很有共鸣：循环架构的 peak memory footprint 极小（只需要存上一帧的 fused state），非常适合边缘部署。

参考链接：
- VBM4D (ref [5]): https://ieeexplore.ieee.org/document/6209637
- FastDVDnet (ref [6]): https://arxiv.org/abs/1907.01314
- EDVR (ref [7]): https://arxiv.org/abs/1905.02716
- RViDeNet / CRVD (ref [8]): https://openaccess.thecvf.com/content_CVPR_2020/papers/Yue_Supervised_Raw_Video_Denoising_With_a_Benchmark_Dataset_on_CVPR_2020_paper.pdf

---

## 8. 关联 & 思考

### 与你的 Tesla 工作的关联

Tesla 的 photon-counting / video pipeline 一直在向"raw-domain 处理 + end-to-end learnable ISP"演进。EMVD 给的几个 takeaway 可以直接 transfer：

1. **Invertible transform 是 free lunch**：用 YUV+Haar 这种"正确"的初始化能让 tiny network 达到大网络性能。Tesla 的 ISP 里的 demosaic / NR 模块也可以借鉴——固定一个 invertible color transform 然后让网络只学 residual。
2. **Recurrent 而非 multi-frame stack**：传统方法（FastDVDnet 等）需要同时输入 5+ 帧，memory 暴涨。EMVD 用 recurrent state（16 channels × H/2 × W/2），延迟和显存都恒定。对自动驾驶这种 30+ FPS、长时序的场景，recurrent 是天然选择。
3. **No intermediate supervision**：所有 stage（fusion / denoising / refinement）共享一个重建 loss，network 自己学 stage 间分工。这跟你早期讲 "软件 2.0" 时强调的"让 loss 而非 hand-engineered pipeline 指挥网络"完全一致。

### 与 SSM / Mamba 的关联

Recurrent fusion 的递归方差传播公式 $\hat{\sigma}_t^2 = \gamma_t^2 \sigma_{\text{noisy}}^2 + (1-\gamma_t)^2 \hat{\sigma}_{t-1}^2$ 在数学上跟 State Space Model 里的隐状态更新 $\mathbf{h}_t = A \mathbf{h}_{t-1} + B \mathbf{x}_t$ 同构。$\gamma_t$ 是 input gate，$(1-\gamma_t)$ 是遗忘门，融合操作是 state update。这跟 GRU / LSTM / Mamba 里的 gate 机制本质一致，只不过这里 gate 是 spatial map 而非 vector。这说明 EMVD 的 fusion stage 本质上是一个"spatially-gated RNN"。

参考链接：
- Mamba / SSM: https://arxiv.org/abs/2312.00752
- GRU original paper: https://arxiv.org/abs/1406.1078

### 与 Noise2Noise 的关联

Noise2Noise (Lehtinen et al. 2018) 的核心 insight 是：如果有多个独立噪声观测 $y_1, y_2$，它们的均值收敛到 clean signal，所以训练 network $\hat{x} = f(y_1)$ 拟合 $y_2$ 也能学到去噪。EMVD 的 fusion stage 是这个思想的"video 版 + gated 版"：不是简单平均，而是 $\gamma$-weighted 累积，其中 $\gamma$ 由 motion 检测调制。这跟 Noise2Noise → Noise2Video → Neighbour2Neighbour 的演进线相关。

参考链接：
- Noise2Noise: https://arxiv.org/abs/1803.04189
- A survey of self-supervised denoising: https://arxiv.org/abs/2203.13019

### 与 Kalman Filter 的深层关联

EMVD 整个 fusion stage 实际上是一个像素级 Extended Kalman Filter 的可微版本。Kalman 增益 $K_t$ 在静态情况下确实是 $1/t$ 形式（递归最小二乘的结论）。EMVD 让 network 自己学这个增益，但约束了它必须是 sigmoid 输出（即 $[0,1]$ 区间）+ spatial map。这是一个非常好的 "neural + classical" hybrid 设计。

### 与 Demosaic / Joint Denoise-Demosaic 的关联

EMVD 在 raw Bayer 域工作，4-channel 表示。这跟 joint denoise-demosaic 文献（比如 JDDP, Deep-JDMM）思路一致——避免"先 demosaic 再 denoise"造成噪声放大。Buades & Duran (ref [2]) 专门讲过 raw domain 直接做 video denoise + demosaic 的好处。

### 与 Multi-stage Image Restoration 的关联

EMVD 的 fusion → denoising → refinement 三阶段 pipeline 跟近期 multi-stage restoration 工作（NAFNet, Restormer 等）的"coarse-to-fine"哲学相似。区别在于 EMVD 把第一个 stage 设定为 temporal fusion（无监督），而后两个 stage 是 spatial processing。这种"先充分利用时域冗余、再 spatial refine"的顺序在传统视频去噪里就是常识（VBM4D 也这样），但端到端学习里这是为数不多的成功案例。

参考链接：
- NAFNet: https://arxiv.org/abs/2104.14069
- Restormer: https://arxiv.org/abs/2111.09881

### 关于 invertibility 的理论联想

Table 2b 的结果——"即使不加 invertibility loss，network 学到的也是接近可逆的变换"——其实可以从信息论角度理解：重建 loss 要求输出能恢复输入的 clean version，而 clean 跟 noisy 之间只差一个噪声扰动。如果 transform 在 forward+inverse 过程中丢了信息，重建就不可能好。所以**可逆性是重建任务本身隐式要求的性质**。这跟 i-RevNet, Invertible Neural Network 的设计哲学一致，只不过这里没有用 coupling-layer 的严格 invertibility，而是用 loss penalty 软约束。

参考链接：
- i-RevNet: https://arxiv.org/abs/1802.07088
- Invertible NN survey: https://arxiv.org/abs/2004.05005

### 关于架构 expressivity 的思考

EMVD 每个 stage 都只有 16 channels 的 conv，3×3 kernel，2-3 层 conv 就够。这是因为他们把"重的 lifting"放到了固定先验的 transform 里（color + wavelet）。CNN 只需要学一个 conditional gate / denoising residual。这是一个非常 efficient 的 capacity 分配——大网络 expressivity 给 transform，小网络 expressivity 给 data-driven refinement。这套哲学跟 Distill / BERT 里的 "pre-trained encoder + small task head" 类似，只不过这里是 signal-processing-style 的 fixed basis + learnable head。

### 可能的局限性（论文没明说但值得 build intuition）

1. **运动模糊边界处理**：fusion 靠 sigmoid gate 软选择，遇到大运动可能就完全 disable fusion，相当于退化为单帧去噪。在没有 optical flow 显式对齐的情况下，快速运动区域会损失严重。
2. **长时间依赖**：recurrent state 是单一 frame 的 fused，没有 long-term memory。如果有 occlusion-reappearance 场景，EMVD 不会"记得"几十帧前的内容。LSTM-like memory 可能能解决，但代价是复杂度。
3. **学习 transforms 的 invertibility 是软的**：在极端大噪声下，如果网络学到的 transform 偏离了 invertible，可能会出现"信息泄露"，输出可能比 raw 还差。Hard invertibility（比如 coupling layer）可能更稳健，但实现复杂。
4. **ISO generalization**：noise model $(a, b)$ 是 ISO-dependent，需要 per-ISO 标定。如果相机更换或 ISP 改动，需要重标定。一个 self-calibrating 版本会更有 deployable 价值。

---

## 9. 一句话总结 Intuition

**EMVD = "传统信号处理先验（YUV+Haar）作为 strong initialization" + "可微 recurrent pipeline 实现 Bayesian-like 递归融合" + "无 intermediate supervision 端到端训练" + "极小 capacity 网络（5.38 GFLOPs）达到 SOTA"**。

它的核心美感在于：**作者深信传统信号处理的几十年的 wisdom 已经是接近最优的先验，深度学习只需要做"residual"的 refinement**。这跟你之前在 Tesla AI Day 提的"不要从零学，把 physics + classical priors baked in"思路一致，也跟近年的 "structured state space" "neural ODE" "Mamba" 这些"用网络参数化一个有结构的 dynamical system" 的潮流是同一个家族。

如果你想 reproduce 或验证其中某一点，我推荐从 EMVD 的 fusion stage 开始手写一个最小版（用 PyTorch 30 行就能搭起来），用合成 Gaussian noise 测一下 sigmoid gate 是否真的在静态区域收敛到 $1/t$ 行为。我猜你会发现这个 emergent behavior 几乎只要训练几百 step 就出现——这是 EMVD 设计中我最惊艳的部分。

参考链接：
- EMVD arXiv (主文): https://arxiv.org/abs/2207.14513
- EMVD GitHub (official, if available): https://github.com/ni-huang/EMVD (注：实际仓库需自行确认)
- VBM4D 代码 (Matteo Maggioni 老仓库): https://www.cs.tut.fi/~foi/papers/VBM4D-2012/
