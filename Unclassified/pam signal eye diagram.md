因为 PAM (Pulse Amplitude Modulation) signal 是 high-speed digital communication 中最核心的 modulation format，并且 Eye Diagram 是评估 Signal Integrity (SI) 的最直观工具，所以理解 PAM signal eye diagram 对于构建高速互联 intuition 至关重要。

### 1. 第一性原理

从第一性原理出发，Eye Diagram 的本质是 **时间域波形的叠加**。

假设我们有一个连续的 PAM signal $v(t)$，我们以 Symbol rate 的周期 $T_{sym}$ (即一个 Unit Interval, UI) 为窗口，将 $v(t)$ 切分成无数段，并且将它们在同一个二维坐标系上叠加显示。

数学上，Eye Diagram 可以表示为：
$$ E(t) = \bigcup_{k=-\infty}^{\infty} v(t + k \cdot T_{sym}) \quad \text{for} \quad 0 \le t < T_{sym} $$
其中：
*   $E(t)$ 是 Eye diagram 的重叠波形集合
*   $v(t)$ 是原始的 PAM signal voltage waveform
*   $k$ 是 symbol 的 index (上标或下标，表示第 $k$ 个 transmitted symbol)
*   $T_{sym}$ 是 symbol period (一个 UI 的时间长度)

因为 NRZ (Non-Return-to-Zero, 即 PAM2) 只有 2 个电平，所以叠加后只会形成 **1 个 "eye"**。但是 PAM-N signal 有 $N$ 个 amplitude levels，所以叠加后会形成 **$N-1$ 个 "eye"**。如果 是 PAM4，那么就会有 3 个 eye；如果是 PAM8，就会有 7 个 eye。

### 2. PAM-N Eye Diagram 架构图解析

为了 build your intuition，我们可以解构 PAM4 的 3-eye 架构：

*   **Vertical Axis (Voltage):** PAM4 有 4 个 nominal levels：$V_0, V_1, V_2, V_3$。
    *   $V_0$: Lowest level (通常代表 bits `00`)
    *   $V_1$: Second level (通常代表 bits `01`)
    *   $V_2$: Third level (通常代表 bits `10`)
    *   $V_3$: Top level (通常代表 bits `11`)
*   **Eye Opening:** 3 个 eye 分别位于：
    *   Eye 0: $V_0$ 和 $V_1$ 之间
    *   Eye 1: $V_1$ 和 $V_2$ 之间 (Middle eye)
    *   Eye 2: $V_2$ 和 $V_3$ 之间

由于 Transmitter 的 linearity 限制 和 Channel 的 insertion loss，这 3 个 eye 通常 **不是等高的**。Middle eye 往往最小，因为 Transition time 的非对称性 导致中间电平的 margin 被压缩。

### 3. 关键参数与公式

#### 3.1 Eye Height (EH) 与 Eye Width (EW)
Eye Height 决定了 vertical noise margin，Eye Width 决定了 horizontal timing margin。

对于 PAM-N 的第 $m$ 个 eye，其 Eye Height 定义为：
$$ EH_m = \min(V_{m}) - \max(V_{m-1}) - 3 \sigma_{noise, m} $$
其中：
*   $EH_m$ 是第 $m$ 个 eye 的 eye height (下标 $m$ 表示从下往上第 $m$ 个 eye)
*   $\min(V_{m})$ 是 level $m$ 的最低电压值 (由 ISI 和 jitter 导致)
*   $\max(V_{m-1})$ 是 level $m-1$ 的最高电压值
*   $\sigma_{noise, m}$ 是该 eye 区域内的 RMS noise

Eye Width 的计算类似，通常基于 Crossing point 的时间分布：
$$ EW = T_{sym} - 2 \cdot \sigma_{jitter} \cdot \text{Factor} $$

#### 3.2 SNR Penalty (First Principles Analysis)
从 Shannon 信道容量 和能量效率的角度，PAM-N 相比 PAM2 付出了 SNR penalty。

假设满量程电压为 $V_{FS}$，PAM-N 的 equally-spaced level spacing 为：
$$ \Delta V = \frac{V_{FS}}{N-1} $$
而 NRZ (PAM2) 的 spacing 是 $V_{FS}$。因此，PAM-N 的 noise margin 缩小了 $(N-1)$ 倍。

对应的 SNR penalty 为：
$$ SNR_{penalty} = 20 \log_{10}(N-1) \quad \text{[dB]} $$
对于 PAM4，$N=4$，penalty 为 $20 \log_10(3) \approx 9.5$ dB。
虽然 PAM4 的 baud rate 减半，bandwidth requirement 降低了，但是它需要额外的 9.5 dB SNR 来维持相同的 BER，这就解释了为什么 PAM4 eye diagram 看起来比 NRZ 更 "closed"，并且更脆弱。

#### 3.3 Symbol Error Rate (SER) 与 Bit Error Rate (BER)
在 Additive White Gaussian Noise (AWGN) channel 下，PAM-N 的 SER 近似为：
$$ SER \approx \frac{2(N-1)}{N} Q\left( \sqrt{\frac{3 \log_2 N}{N^2 - 1} \cdot SNR_{symbol}} \right) $$
其中：
*   $N$ 是 PAM 的 order (e.g., 4 for PAM4)
*   $Q(\cdot)$ 是 Q-function (标准正态分布的右尾概率)
*   $SNR_{symbol}$ 是每个 symbol 的 signal-to-noise ratio
*   分子中的 $\log_2 N$ 将 symbol energy 转换为 bit energy
*   分母中的 $N^2 - 1$ 反映了 multi-level 带来的 average power normalization

### 4. 导致 Eye Closure 的物理机制

因为 Channel 是 bandwidth-limited 的，所以 Signal 会受到多种 impairments：

*   **ISI (Inter-Symbol Interference):** Channel 的 impulse response $h(t)$ 具有长长的 tail。当 previous symbols 的 tail 拖尾到 current symbol 时，会导致 level 变化，eye 变窄。
*   **Jitter:** 包括 Deterministic Jitter (DJ, 由 ISI、crosstalk 引起) 和 Random Jitter (RJ, 由 thermal noise 引起)。Jitter 导致 crossing points 变宽，EW 缩小。
*   **Crosstalk (XTALK):** 在 differential pairs 中，FEXT (Far-End Crosstalk) 和 NEXT (Near-End Crosstalk) 会直接叠加在 signal 上，破坏 eye 的 vertical opening。
*   **Non-linearity:** DAC/ADC 的 INL/DNL，或者 Tx driver 的 compression，导致 PAM levels 不等距，某些 eye (特别是 middle eye) 被严重压缩。

### 5. 实验数据表参考：PAM4 vs NRZ at 56 GBaud

| Parameter | NRZ @ 56 GBaud | PAM4 @ 56 GBaud (56 GBd) | Unit | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Data Rate | 56 | 112 | Gbps | PAM4 carries 2 bits/symbol |
| Number of Eyes | 1 | 3 | - | $N-1$ rule |
| Nominal Spacing | $V_{FS}$ | $V_{FS}/3$ | V | PAM4 margin is 1/3 of NRZ |
| SNR Penalty | 0 | ~9.5 | dB | Calculated via $20\log_{10}(3)$ |
| Typical BER Target | $10^{-12}$ | $10^{-6}$ (pre-FEC) | - | PAM4 relies on FEC (RS-FEC/KP4) |
| Eye Height (Typ.) | ~300 | ~80 | mV | Post-channel (e.g., 30dB loss) |

### 6. Equalization 与 Eye Opening

因为 Channel loss 通常会导致 eye 完全闭合，所以 Receiver 必须使用 Equalization。

*   **CTLE (Continuous Time Linear Equalizer):** High-pass filter，补偿 Channel 的 low-pass 特性。
    *   Transfer function: $H_{CTLE}(f) = \frac{g_m}{1 + j f / f_p} \cdot (1 + j f / f_z)$
    *   $f_z$ 是 zero frequency，提供 high-frequency boost
    *   $f_p$ 是 pole frequency
*   **DFE (Decision Feedback Equalizer):** Non-linear equalizer，消除 post-cursor ISI without noise amplification。
    *   $y_n = x_n + \sum_{k=1}^{K} c_k \cdot \hat{x}_{n-k}$
    *   $y_n$ 是 equalized sample
    *   $c_k$ 是第 $k$ 阶 DFE tap coefficient
    *   $\hat{x}_{n-k}$ 是 previously decided symbol

通过 FFE + CTLE + DFE 的联合作用，原本闭合的 PAM eye diagram 可以重新被 "open" 出来，从而让 Sampler 能够正确判决。

### 7. 扩展联想

*   **PAM6 / PAM8:** 为了追求更高的 data rate (如 224G SerDes)，industry 正在探索 PAM6 (3 bits/symbol) 甚至 PAM8 (4 bits/symbol)。PAM6 有 5 个 eyes，SNR penalty 进一步增加，对 DSP equalization 的要求极高。
*   **ENOB (Effective Number of Bits):** PAM signal 本质上是 quantized analog signal。ADC-based receiver 的 ENOB 直接决定了能分辨多少个 PAM levels。
*   **MLSE (Maximum Likelihood Sequence Estimation):** 对于 severely closed eye，传统的 threshold decision 失效，需要使用 Viterbi algorithm 在 trellis diagram 上寻找最可能的 sequence，这跳出了单纯 "open eye" 的范畴，直接对 closed eye 进行 sequence detection。
*   **CEI-112G-VSR/MR/LR:** OIF 定义的不同 reach 标准，其对 eye diagram 的 mask 要求不同。VSR (Very Short Reach) 的 eye mask 最宽松，LR (Long Reach) 最严格。

### Web Links for Reference

1.  **Keysight Technologies - PAM4 Fundamentals:** [Understanding PAM4 and Its Test Challenges](https://www.keysight.com/us/en/assets/7018-06035/application-notes/5992-1662.pdf) (Detailed breakdown of PAM4 eye structures and measurement methodologies)
2.  **Teledyne LeCroy - PAM4 Eye Analysis:** [PAM4 Signal Integrity and Eye Diagram Analysis](https://teledynelecroy.com/doc/understanding-pam4-signals) (Visual explanation of 3-eye diagrams and BER contours)
3.  **IEEE 802.3bs (200G/400G Ethernet):** [IEEE P802.3bs Task Force](https://www.ieee802.org/3/bs/) (Official standard defining PAM4 eye mask templates and transmitter specifications)
4.  **Rambus - PAM4 Signaling:** [PAM4 Signaling for High-Speed Memory and SerDes](https://www.rambus.com/blogs/pam4-signaling/) (Good first-principles explanation of SNR penalty and DSP requirements)