---
source_pdf: Partial coherence enhances parallelized.pdf
paper_sha256: 5b622b72f105aa72c9d1b36eac6b0e7917b61ddf01cdc2e8a59e35865da95624
processed_at: '2026-08-06T02:22:24-07:00'
target_folder: AI芯片
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话总结

**用"不太纯净"的光，反而能让光计算跑得更快更多，还能省硬件。**

## 1. 背景：光计算现在怎么玩的

光计算芯片做矩阵乘法，基本套路是：**把数据变成光的强度，把权重变成光的衰减，光在芯片里走一圈，自然就完成了乘加运算。**

但有个烦人的问题——

**如果用激光（完全相干光）**，光太"纯净"了，纯净到两束光碰到一起会互相干涉。你把同一束激光分给 N 个输入通道，它们在输出端汇合时，会因为相位差产生明暗条纹，信号剧烈抖动，根本没法算。

**传统解法**：给每个输入通道分配一个**不同颜色（波长）的光**。N 个通道要 N 种颜色。颜色不够用了，芯片就做不大。这叫 WDM（波分复用）。

打个比方：这就像开会时，每个人都得用自己专属频道对讲机，不然就串台。人一多，频道就不够分了。

## 2. 这篇 paper 的核心 trick

作者问了个反常识的问题：**凭什么一定要用激光？用稍微"脏"一点的光不行吗？**

"脏"的光指的是**部分相干光**——不是纯净的单色激光，而是有一定带宽的宽带光，比如 0.8 纳米宽。

这种光有个特性：**它的相干长度很短**（大概半毫米），超出这个距离，两束同源的光就"认不出对方了"，碰在一起不干涉。

**关键操作**：在相邻输入通道之间接一段 1 米长的光纤（远超半毫米的相干长度），这样三路光走到汇合点时已经互相"陌生"，不会干涉。

结果就是：**同一个颜色的光可以同时分给所有 N 个输入通道，不用再每种通道配一个颜色。**

继续用开会的比方：不用给每人配专属频道了，大家都用同一个频道，但给每个人戴个"延迟耳机"，让他们说话错开时间，就不会串台了。

## 3. 为什么这是个大事

**省了多少？**

假设你要同时处理 P 个任务，每个任务输入维度是 N：
- 传统做法（激光）：需要 P × N 种颜色
- 新做法（部分相干光）：只要 P 种颜色

省了 (N-1) × P 倍。N 和 P 越大，省得越多。

而且，传统做法里 N 受限于**光器件能支持的波长窗口**（通常就几十纳米）。新做法里输入带宽不随 N 增长，**芯片可以做得更大**。

**代价是什么？**

部分相干光来自放大器的自发辐射噪声（ASE），本质上比激光"吵"。信噪比在高功率时会饱和，不像激光那样线性提升。**但在芯片实际工作的低功率区（微瓦到毫瓦级），信噪比够用。** 作者实测了眼图，4 纳米带宽以上就很清晰。

## 4. 两个实验验证

**实验一：帕金森病人步态识别（3×3 芯片）**

用 10 个病人的步态数据，做卷积提取特征，再用 CNN 分类。
- CPU 算的准确率：92.7%
- 传统激光光子芯片：92.2%
- 新的部分相干光芯片：92.2%

**准确率一样，但部分相干光只用了 2 个波长，激光方案要 6 个。**

**实验二：MNIST 手写数字识别（9×3 芯片，高速版）**

用硅光芯片集成电吸收调制器（EAM），跑 2 GSa/s，总算力 0.108 TOPS。
- CPU 理论准确率：95.0%
- 部分相干光芯片：92.4%
- 做 4 次平均后：93.9%

稍微低于理论值，因为 ASE 噪声，但 averaging 能补回来。

## 5. 还有几个工程细节

**负权重怎么办？** 光强只能正，但卷积核有负值（比如 Sobel 算子有 -1）。作者用了**差分测量法**：测四次（正常、全零、只设权重、只设输入），后三次做参考，减掉 unwanted 项。简单粗暴但有效。硬件上可以用平衡光电探测器实现，不用真测四次。

**1 米光纤是外接的**，没法集成在芯片上。作者给了个方案：用多个独立的 ASE 光源，它们本来就互相不相干，不需要延迟线。每个光源驱动几十个通道，比每个通道一个激光器便宜多了。

**理论极限**：用 40 纳米带宽的 ASE 光源，10 个光载波，每个 4 纳米带宽，数据加载率 30 GSa/s，系统能跑到 **16.2 TOPS**。当然这是估算，没实测。

## 6. 这篇 paper 真正的价值

不是 0.108 TOPS 这个数字（跟数字芯片比很普通），也不是帕金森识别这个应用（数据集太小）。

**价值在于打破了一个思维定式**——光计算社区默认"越相干越好"，这篇 paper 说"在 crossbar 架构下，降低相干性反而能提升并行度"。

这是个 regime 判断：
- 小规模、高信噪比需求 → 用激光
- 大规模、追求并行吞吐 → 用部分相干光

类似当年从"晶体管越多越好"到"多核并行更好"的转向。

## 7. 我觉得的不足

1. **4-bit 权重精度**——相变材料 GST 能做的精度有限，跟数字芯片的 8-bit 有差距
2. **光纤延迟线是外接的**——全集成还需要更多工程
3. **差分测量要算四次**——proof of concept 阶段可以，实际产品得硬件化
4. **0.108 TOPS / 1 TOPS/W**——跟数字 ASIC 比有数量级差距，作者自己也承认 contribution 是架构不是绝对数字
5. **任务都太小**——MNIST 和步态识别都是 toy task，没在 ImageNet 或更大规模上验证

但总的来说，这是个概念上很干净的 insight，把"相干性"从"always better"重新定位成"depends on regime"，对光计算社区有启发意义。

---

参考链接：
- 原文：https://doi.org/10.1038/s41586-024-07590-y
- 前作 PCM tensor core：https://doi.org/10.1038/s41586-020-03070-1
- Coherent nanophotonic circuit：https://doi.org/10.1038/nphoton.2017.93
- Broadcast-and-weight：https://doi.org/10.1109/JLT.2014.2345652
- 帕金森步态数据集：https://doi.org/10.13026/C24H3N

---

# Partial Coherence Enhances Parallelized Photonic Computing - 详细讲解

Paper link: https://doi.org/10.1038/s41586-024-07590-y

## 1. 核心 Insight - 这篇 paper 在打破什么"常识"

传统 photonic computing 的 guiding principle 一直认为 **越 coherent 越好**。无论是 Shen et al. 的 coherent nanophotonic circuits (Nature Photonics 2017, https://doi.org/10.1038/nphoton.2017.93)、Tait et al. 的 broadcast-and-weight protocol (JLT 2014, https://doi.org/10.1109/JLT.2014.2345652)、还是 Feldmann et al. 的 phase-change-material photonic tensor core (Nature 2021, https://doi.org/10.1038/s41586-020-03070-1)，全都依赖 coherent light source。

这篇 paper 的核心 claim 是：对于 photonic convolutional processing 这种 **不依赖 interference 来做 computation** 的架构（crossbar array 用 amplitude modulation 做 weighting），**降低 coherence 反而能 boost parallelism**。这个反直觉的结论来自一个很简单的物理事实——

> 当你把同一个 coherent wavelength 分到 N 个 input channel 然后在 bus waveguide 里 combine 时，phase fluctuation $\Delta\varphi$ 会导致 intensity fluctuation $|E + Ee^{i\Delta\varphi}|^2$ sinuisoidally 振荡。所以传统方案必须给每个 channel 分配一个**独立的 wavelength**，即 N 维 input vector 要消耗 N 个 optical band。

而 partially coherent light 的 coherence length $L_c \propto 1/\Delta\omega$ 很短（0.8nm 带宽时只有 ~550µm），如果给相邻 input channel 之间引入 1m 的 fibre delay（远大于 $L_c$），那么 N 个 channel 之间就变成 mutually incoherent，combine 时不再 interference。这样**同一个 wavelength 可以同时 feed N 个 input channel**，N-fold parallelism enhancement。

## 2. Coherence 的物理 - 从 first principles 构建 intuition

### 2.1 单个 unit cell 的 intensity fluctuation

考虑 Fig. 1a 里的 unit cell：light 等分成两 arm，做 multiplication 后在 common bus 里 sum。两 arm 之间的 phase difference 记为 $\Delta\varphi$。

- **Coherent source** ($\mathcal{E} = e^{i\omega_0 t}$, 单频): 输出 intensity 为 $|E + Ee^{i\Delta\varphi}|^2 = 2|E|^2(1 + \cos\Delta\varphi)$，sinusoidally 随 $\Delta\varphi$ 变化。任何微小的 phase jitter 都会被 translate 成 intensity noise。

- **Idealized incoherent source** (覆盖整个 frequency range): 所有 frequency 分量的 interference term 平均掉，output 与 $\Delta\varphi$ 无关。但 WDM 不能用，因为没剩 bandwidth。

- **Partially coherent source** (Gaussian spectrum $\mathrm{Gauss}(\omega|\omega_0, \Delta\omega)$): 这是 sweet spot。Output intensity 对 $\Delta\varphi$ 的依赖随 phase difference 增大而 progressively decay。具体来说，degree of coherence 服从 Wiener-Khinchin theorem —— spectral density 的 Fourier transform 就是 temporal coherence function。

### 2.2 Coherence length 公式

Degree of coherence 定义为 interference visibility:

$$\gamma = \frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min}}$$

其中 $I_{\max}, I_{\min}$ 是 interference fringe 的 peak 和 valley。Coherence length 定义为 $\gamma$ 降到 0.5 时的 path difference。

实验测得（Fig. 2c, d）coherence length 与 optical bandwidth 成反比：

$$L_c \cdot \Delta\omega \approx \text{const}$$

这是经典的 Fourier transform limit。具体数值：
- 0.8 nm bandwidth (C34 channel): $L_c \approx 550\,\mu\text{m}$
- 2.0 nm: ~220 µm
- 4.0 nm: ~110 µm
- 8.0 nm: ~55 µm
- 16.0 nm: ~28 µm

只要 path difference $\gg L_c$，两 beam 就 effectively incoherent。

### 2.3 SNR tradeoff

这里有个重要的 trade-off（Fig. 2e）。Partially coherent light 来自 EDFA 的 ASE (amplified spontaneous emission)，是 stochastic process，inherently 比 coherent laser 的 shot noise 大。Paper 里测得：

$$\text{SNR}_{\text{partial}}(P) \approx \text{SNR}_{\text{saturated}} \propto \frac{\Delta\omega_{\text{optical}}}{\Delta\omega_{\text{electrical}}}$$

也就是说 SNR 随 optical power 增大而 saturate，不再像 coherent light 那样 linearly 提升。但在 integrated photonics 实际工作区间（0.1 µW 到 0.1 mW），partially coherent 的 SNR 没有显著低于 coherent light。这个区间分析是 paper 的关键 justification —— 不是在所有场景都赢，而是在 photonic computing 实际工作的 power regime 下够用。

Eye diagram（Fig. 2f, 2 GHz, 0.05 mW）也验证了：0.8nm 时眼图模糊，4.0nm 以上 eye opening 清晰。所以 paper 在 MNIST 实验里用 8.0 nm bandwidth，不是随便选的。

## 3. 系统架构 - 两种 Photonic Tensor Core

### 3.1 Photonic Memory Tensor Core (3×3, for Parkinson gait)

Fig. 3a 是 chip 的 optical image。Crossbar array 用 **phase-change material (GST, Ge₂Sb₂Te₅) + ITO capping** 做非挥发 weight memory。Weight 通过 pump-probe scheme 写入：用 high-power pulse 把 GST 从 amorphous (low transmission) 切到 crystalline (high transmission)，中间态通过 partial crystallization 实现 4-bit granularity。

Weight 到 transmission 的 mapping：

$$T = w \cdot \frac{T_{\max} - T_{\min}}{2} + \frac{T_{\max} + T_{\min}}{2}$$

- $w \in [-1, 1]$ 是 target weight
- $T_{\max}, T_{\min}$ 是 GST 在 fully crystalline / fully amorphous 时的 transmission
- $T \in [T_{\min}, T_{\max}]$ 是物理可测的 transmission level

实验测得 $T_{\max} - T_{\min} > 20\%$，足够做 4-bit operation。

**核心 trick: 1m 的 fibre delay**（Fig. 3b）。相邻 input channel 之间加 1m 的 fibre，远大于 0.8nm 部分相干光的 550µm coherence length。这样三路 light 在 bus 里 combine 时，phase fluctuation 被 de-correlate，intensity 稳定（Fig. 3d）。对比 coherent light 在同样 setup 下有剧烈 fluctuation（Fig. 3c）。

### 3.2 Photonic EAM Tensor Core (9×3, for MNIST, high-speed)

这个 chip 用 **IMEC iSiPP50G silicon photonics platform**，集成 EAM (electro-absorption modulator) 做 weight，集成 photodetector 做 readout。Fig. 5a, 5b。

- Input: 9 个 input grating coupler，每个有 EAM 做 input data encoding
- Crossbar: 9×3 的 EAM array 做 weight
- Output: 3 个 photodetector (with TIA)
- Driver: Xilinx RFSoC ZCU216 FPGA，16 DACs at 2 GSa/s

Data loading rate 是 2 GSa/s per channel，9 个 channel 总共 18 GSa/s，对应 0.108 TOPS。Energy efficiency 估算 1 TOPS/W。这个数字跟 electrical ASIC 比其实不算惊艳，但 paper 的 contribution 是 architecture，不是绝对数字。

## 4. 非负 transmission 到负 weight 的 mapping - 这个细节很重要

Photonic 系统的 output 是 intensity，必然 non-negative。但 convolution kernel 比如 Sobel filter $[1, 0, -1]$ 是有负值的。Paper 在 Methods 里给了一个 elegant 的四步 measurement + post-processing scheme（公式 1-5）：

设 $P_i = x_i(P_{\max} - P_{\min}) + P_{\min}$（input encoding），$T_i = w_i\frac{T_{\max} - T_{\min}}{2} + \frac{T_{\max} + T_{\min}}{2}$（weight encoding）。

Step (d): 测 $\sum_i P_i T_i$ —— 这是正常 computation，含 4 项交叉乘积。
Step (e): $x=0, w=0$ → 测 $\sum_i P_{\min}\frac{T_{\max}+T_{\min}}{2}$ —— 纯 DC offset，一次。
Step (f): $x=0$, 设 target $w$ → 测 weight-only 项，每个 kernel 一次。
Step (g): 设 target $x$, $w=0$ → 测 input-only 项，每个 input vector 一次。

最终：

$$\text{Result} = (1) - (3) - (4) + (2) = (P_{\max} - P_{\min})\frac{T_{\max} - T_{\min}}{2}\sum_i x_i w_i$$

这个 differential measurement 把所有 unwanted 项 (DC, $x$-only, $w$-only) 都 cancel 掉，剩下纯粹的 $\sum_i x_i w_i$。

**intuition**: 这是 analog computing 经典技巧——你做不出负值，那就用 "positive reference + differential" 把负值 extract 出来。Paper 提到用 balanced photodetection 可以 hardware 实现，避免 doubling。这是和 Wright et al. (Nature 2022, https://doi.org/10.1038/s41586-022-04414-0) 那种 deep physical neural network 类似的思路。

## 5. 实验结果 - Parkinson Gait + MNIST

### 5.1 Parkinson Gait Classification

数据集：PhysioNet 的 "Gait in Parkinson's Disease"（https://doi.org/10.13026/C24H3N），10 个病人，每人 50 个 gait pulse，1.2s duration，下采样到 31 个 time point，interval 0.04s。

Kernel 是 3 个手工设计的 1×3 filter:
- $[1, 1, -1]^T$: right-edge extraction
- $[1, -1, 1]^T$: peak suppression  
- $[-1, 1, 1]^T$: left-edge extraction

这些是 signal processing 里很经典的差分 kernel，专门用来 detect edge / peak。

Parallelism 实证：用两个 wavelength（C34 0.8nm 和 C33 0.8nm）同时处理两个病人的 gait signal，节省了 4 个 wavelength（传统 coherent 系统要 6 个 wavelength）。

CNN 架构（Fig. 4d）：Conv(3×1×3) → ReLU → Flatten(87) → FC(10) → Softmax。
- 不用 conv layer: 84.4%
- CPU conv: 92.7%
- Coherent photonic conv: 92.2%
- Partially coherent photonic conv: 92.2%

误差分布 Gaussian，mean 和 std 接近（Fig. 4c）。**关键是 partially coherent 在 accuracy 上几乎没损失**，但 wavelength 消耗少 $(N-1)\times P$ 倍。

### 5.2 MNIST Handwritten Digits

用 Sobel $G_x$ 和 $G_y$ 做 edge detection。Sobel 是经典 3×3 kernel:
$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

Photonic EAM tensor core 9×3 正好 encode 一个 3×3 kernel flatten 成 9 维。

Convolution accuracy:
- Without averaging: normalized std 0.094, CNN accuracy 92.4%
- 4-point averaging: normalized std 0.049, CNN accuracy 93.9%
- CPU baseline: 95.0%

Averaging 把 ASE 的 stochastic noise 平均掉，符合 $\sigma_n \propto 1/\sqrt{N}$ 规律。

## 6. Scalability 分析 - paper 的真正贡献

### 6.1 Wavelength 消耗对比

设 photonic tensor core 维度 $N \times M$，parallelism 为 $P$（同时处理 $P$ 个 input vector）。

- **Coherent**: $P \times N$ 个 wavelength（每个 input vector 需要 $N$ 个独立 wavelength 避免 interference）
- **Partially coherent**: $P$ 个 wavelength（每个 input vector 共用 1 个 wavelength，靠 path-length de-correlation 避免 interference）

节省比例：$(N-1) \times P$。对于大 $N$ 和大 $P$，这是 game-changer。

### 6.2 上限估计

Paper 在 Discussion 里给了 scalability 上限：silicon nitride-on-silicon platform propagation loss 0.4 dB/cm，假设最长 delay line 损耗 < 3 dB，单个 4nm ASE source 可以 support ~59 个 input channel。

要 scale 更大，用 **array of independent ASE sources** 工作在同一 wavelength band。这些 source 互相 incoherent，不需要 delay line（因为本来就是 different source）。每个 ASE source 驱动几十个 channel。这比每个 channel 一个 laser 的方案经济得多。

理论上限：30 GSa/s data loading × 10 个 optical carriers × (40nm ASE band / 4nm per carrier) = **16.2 TOPS** 系统处理速度。

### 6.3 SNR 的 absolute upper bound

Paper Supplementary 里也承认 partially coherent 在 small scale 上 SNR 不如 coherent。临界点是当：
- $N$ 小，$P$ 小：coherent 赢（高 SNR，少量 wavelength 够用）
- $N$ 大，$P$ 大：partially coherent 赢（parallelism 优势压倒 SNR 劣势）

这是个 regime 问题，不是 universal claim。

## 7. 与相关工作的 context

这篇 paper 的 intellectual lineage 我觉得可以这样理解：

1. **Feldmann et al. Nature 2021** (https://doi.org/10.1038/s41586-020-03070-1) — 提出 phase-change photonic tensor core，用 frequency comb 做 WDM。每个 input 一个 wavelength，瓶颈是 comb 的 spectral range 限制了 N。

2. **Shen et al. Nature Photonics 2017** (https://doi.org/10.1038/nphoton.2017.93) — coherent MZI mesh，要大量 phase shifter 精确 control。 scalability 受 thermal crosstalk 限制。

3. **Tait et al. broadcast-and-weight** (https://doi.org/10.1109/JLT.2014.2345652) — MRR-based，每个 wavelength 对应一个 MRR，N 个 input 要 N 个 MRR 精确 tune。

4. **本 paper** — 把"避免 channel 间 interference"这个**问题**，从"spectral separation"换成"temporal incoherence"。这其实是个 reframe，不是新物理，但很 powerful。

5. **后续可能延伸** — 我想到几个方向：
   - 跟 Hafiz et al. 的 analog self-learning hardware (Scientific Reports 2024, https://doi.org/10.1038/s41598-024-53249-0) 结合，做 in-situ training
   - 跟 Wright et al. Nature 2022 deep physical neural networks (https://doi.org/10.1038/s41586-022-04414-0) 思路结合，在 partially coherent regime 做 forward/backward propagation
   - SLED 替代 EDFA ASE，提升 SNR。Mehta et al. IEEE PTL 2023 (https://doi.org/10.1109/LPT.2023.3248285) 有 heterogeneously integrated III-V/silicon SLED 可用
   - 跟 microcomb (Bai et al. Nature Communications 2023, https://doi.org/10.1038/s41467-022-35513-8) 对比时，本方案优势是不需要 comb 的 strict pump control

## 8. 我的批评性思考

几个我觉得可以质疑的点：

1. **4-bit weight 精度** — GST 的 non-volatile multi-level 在文献里普遍也就 4-5 bit，跟 digital NN 的 8-bit INT8 比有 gap。Paper 没特别讨论 bit-precision 对 CNN accuracy 的影响。

2. **Fibre delay line 是 off-chip** — Fig. 3b 里 1m 的 fibre 是外接的。如果要 fully integrated，on-chip delay line 1m 在 SiN 上要 ~25cm spiral，0.4 dB/cm × 25cm = 10 dB loss，太多。Paper 在 Supplementary 里讨论用多个 ASE source 解决，但这又增加了 source 数量。

3. **Post-processing 的 latency** — 公式 (1)-(5) 的 differential measurement 要 4 次测量 + software post-processing。这是 proof-of-concept 限制，但实际加速器要 hardware 实现 balanced photodetection。

4. **0.108 TOPS / 1 TOPS/W** — 这两个数字跟 digital ASIC 比（比如 TSMC 7nm 的 100+ TOPS/W）有数量级差距。Paper 自己说 contribution 是 architecture 不是绝对数字，但 reviewer 肯定会问。

5. **Gait + MNIST 是 toy task** — ImageNet 级别 or LLM-scale 才能 prove scalability claim。Paper 说理论上 16.2 TOPS，但没实测。

但总体上，这是一个 conceptually clean 的 insight，把 coherence 从"always better"变成"depends on regime"。这种 paradigm shift 类似于 analogue vs digital、Von Neumann vs in-memory computing 的 reframe。对 photonic computing 社区是 valuable 的一课。

---

**总结一句**: 这篇 paper 的核心贡献是用 partially coherent light 把 photonic tensor core 的 wavelength consumption 从 $O(NP)$ 降到 $O(P)$，scability 不再被 spectral window 限制。代价是 SNR（在 ASE source 上）和需要 delay line。Regime 是 large N、large P 时赢，small scale 时 coherent 仍然好。

主要参考链接：
- Paper: https://doi.org/10.1038/s41586-024-07590-y
- Feldmann PCM tensor core: https://doi.org/10.1038/s41586-020-03070-1
- Shen coherent nanophotonic: https://doi.org/10.1038/nphoton.2017.93
- Tait broadcast-and-weight: https://doi.org/10.1109/JLT.2014.2345652
- Wright deep PNN: https://doi.org/10.1038/s41586-022-04414-0
- PhysioNet Gait PD: https://doi.org/10.13026/C24H3N
- MNIST: https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
