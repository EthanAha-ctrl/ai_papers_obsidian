---
source_pdf: High-integrated photonic tensor core utilizing high-dimensional lightwave
  and microwave multidomain multiplexing.pdf
paper_sha256: 3d5d1f1c7beac1750041fe1c38565943f526e45ee40e6b5854c1b58ef69fdde3
processed_at: '2026-08-04T23:47:27-07:00'
target_folder: AI芯片
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

好，我换一种讲法，像跟你喝咖啡聊天那样讲。

---

## 这篇 paper 到底干了啥

你想用光来做神经网络计算。光算加法乘法特别快，又省电，听上去很美。

但传统做法有个很蠢的地方：**一个权重 = 一个光器件**。你有 1000 个权重，就 1000 个器件；1 万个权重，就 1 万个器件。芯片面积爆炸，而且器件挤在一起会互相加热串扰（thermal crosstalk），你调一个，旁边的也跟着变，整个系统崩溃。

这篇 paper 的核心骚操作：**就留一个光器件**——一个 microring resonator（环形光腔），大概 90 微米见方，比头发丝还细。然后所有权重全靠"**调激光的波长**"来设定。

你不去动器件，你去动光源。激光波长对上环形腔的共振频率，光被腔吸走，权重 = 0；激光波长偏离共振，光直接穿过去，权重 = 1。想调中间值，把波长停在这 Lorentzian 谱线的斜边上就行。

就这样，$N^2$ 个器件，压成 1 个器件 + $N$ 个激光器。

---

## 为什么这个 trick 成立——MRR 是天然的 wavelength filter

Microring resonator 的传输谱（through port 输出光强 vs 波长）长这样：

$$T(\lambda) = 1 - \frac{(1-a)\kappa^2}{(1-a\kappa)^2 + 4a\kappa \sin^2(\phi/2)}$$

我解释下每个变量：

- $\lambda$：输入光波长
- $a$：round-trip amplitude，光在 ring 里转一圈后剩多少。$a = e^{-\alpha L / 2}$，$\alpha$ 是波导损耗系数，$L$ 是 ring 周长
- $\kappa$：coupling coefficient，bus waveguide 和 ring 之间的耦合强度，0 到 1 之间
- $\phi = 2\pi n_g L / \lambda$：round-trip phase，$n_g$ 是 group index（群折射率），跟材料色散有关
- $T(\lambda)$：through port 的透过率，0 到 1

当 $\phi = 2\pi m$（$m$ 是整数），也就是光在 ring 里转一圈刚好相位差 $2\pi$ 的整数倍，发生共振。光能量被 ring 困住，through port 输出接近 0。这就是 Lorentzian 的 dip。

**直觉**：ring 就像一个 frequency-selective trap。某些波长掉进去就出不来（weight=0），其他波长直接穿过（weight=1）。你只要调激光波长，就能选 weight。这比调器件温度简单太多了——温度调一调，旁边器件跟着变；激光调一调，谁也不影响谁。

而且这个 transmission spectrum 是周期性的，周期叫 FSR（Free Spectral Range）：

$$FSR = \frac{\lambda^2}{n_g L}$$

这篇 paper 里 FSR 大约 2.24 nm，一个 FSR 周期里塞了 4 个 wavelength，对应一个 $2 \times 2$ kernel 的 4 个 element。

参考 MRR 原理: https://www.sciencedirect.com/topics/engineering/microring-resonator

---

## 负权重怎么办——MZM 的对称工作点 trick

光的强度永远 $\geq 0$。你用光强表示权重，没法表达负数。这是个老问题。

传统解法都很笨重：
- **Balanced photodetector**：两路光，一路 signal 一路 reference，到 PD 上做减法。硬件翻倍。
- **Electrical subtraction**：PD 后面用电路减。慢，而且引入电子学 bottleneck。
- **Phase encoding**：用相位编码正负，但需要 coherent detection，系统复杂。

这篇 paper 用了一个很优雅的 trick：**Mach-Zehnder modulator (MZM) 的 transfer function 是对称的**。

MZM 的 transfer function：

$$P_{out} = P_{in} \cos^2\left(\frac{\pi V}{V_\pi}\right) = \frac{P_{in}}{2}\left[1 + \cos\left(\frac{2\pi V}{V_\pi}\right)\right]$$

变量解释：
- $V$：加到 MZM 电极上的总电压（DC bias + AC signal）
- $V_\pi$：half-wave voltage，产生 $\pi$ 相位差需要的电压，典型值 3-5V
- $P_{in}, P_{out}$：输入输出光功率

这是个 cos² 函数，在 $V = 0$ 处是 maximum，在 $V = \pm V_\pi/2$ 处是中间线性区（quadrature point），在 $V = \pm V_\pi$ 处是 minimum。

关键 insight：**quadrature point 有两个，$+V_\pi/2$ 和 $-V_\pi/2$，在这两点 small-signal modulation 的斜率正好相反**。

- $V_{bias} = -V_\pi/2$（paper 里叫 $Q^+$ point）：信号增加 → 光强增加 → 正权重
- $V_{bias} = +V_\pi/2$（paper 里叫 $Q^-$ point）：信号增加 → 光强减小 → 负权重

所以同一个 microwave signal，你只要把 MZM 的 DC bias 在 $\pm V_\pi/2$ 之间切换，输出的 optical intensity 调制就自动带上正负号。

最终 weight：

$$w_{ij} = \text{sign}(V_{bias}) \cdot T(\lambda_{ij})$$

$\text{sign}(V_{bias})$ 由 MZM 工作点决定（+1 或 -1），$T(\lambda_{ij})$ 由 MRR 的波长选择决定（0 到 1）。

**直觉**：cos² 函数天然是个对称的 "S 形"在中间线性区。你在 S 的上半段工作，输出跟输入同相；你在 S 的下半段工作，输出跟输入反相。这就是正负号。PD 是平方律器件不区分 phase，但 intensity 调制的 sign 已经被 MZM 的 bias 编码了。完全不需要双路光或差分 PD。

参考 MZM 工作原理: https://www.rp-photonics.com/mach_zehnder_modulators.html

---

## 怎么从 matrix 升级到 tensor——加一个微波维度

普通的 optical WDM computing：$N$ 个波长代表 $N$ 个权重，做的是 $W \cdot \vec{x}$，matrix-vector 乘法。

但卷积神经网络要处理多通道输入（比如 RGB 三通道），做的是 tensor convolution。你怎么同时处理 3 个 channel？

这篇 paper 的解法：**再加一个频率维度——microwave subcarrier**。

把输入数据调制到微波副载波上，每个 channel 用不同频率：
- $f_1 = 7.67$ GHz → R 通道
- $f_2 = 15.33$ GHz → G 通道  
- $f_3 = 23.00$ GHz → B 通道

这三个频率在频谱上分开，但共享同一个光载波。就像广播电台：不同频道播不同节目，但都通过同一条光路传输。

这样 4 个 wavelength（代表 kernel 的 4 个 weight）× 3 个 microwave frequency（代表 3 个 input channel）= 一个 MRR 同时处理 12 个 MAC。一次 pass 就完成一个 $4 \times 4$ weight matrix 乘 $4 \times 3$ input matrix 的 tensor convolution。

公式 (1)：

$$
\begin{bmatrix} y_{R1} & y_{G1} & y_{B1} \\ y_{R2} & y_{G2} & y_{B2} \\ y_{R3} & y_{G3} & y_{B3} \\ y_{R4} & y_{G4} & y_{B4} \end{bmatrix} =
\begin{bmatrix} w_{11} & w_{12} & w_{13} & w_{14} \\ w_{21} & w_{22} & w_{23} & w_{24} \\ w_{31} & w_{32} & w_{33} & w_{34} \\ w_{41} & w_{42} & w_{43} & w_{44} \end{bmatrix}
\begin{bmatrix} x_{R1} & x_{G1} & x_{B1} \\ x_{R2} & x_{G2} & x_{B2} \\ x_{R3} & x_{G3} & x_{B3} \\ x_{R4} & x_{G4} & x_{B4} \end{bmatrix}
$$

变量含义：
- $Y = \{y_{ck}\}$：输出 feature map。$c \in \{R, G, B\}$ 是 color channel index，$k \in \{1,2,3,4\}$ 是 kernel index
- $W = \{w_{ij}\}$：weight matrix。$i$ 是 kernel index (1-4)，$j$ 是 kernel 内 element index (1-4)，所以 $w_{23}$ 表示第 2 个 kernel 的第 3 个 element
- $X = \{x_{cj}\}$：input data。$c$ 是 channel，$j$ 是 kernel element 位置

物理对应：
- $W$ 的列 index $j$ → 映射到 wavelength（4 个波长）
- $X$ 的行 index $j$ → 同一个 kernel 的 4 个 element，也对应 wavelength
- $X$ 的列 index $c$ → 映射到 microwave frequency（3 个频率）
- $Y$ 的行 index $k$ → 4 个 kernel，通过 4 组 wavelength 选择实现

**为什么这是 tensor convolution 而不是 matrix multiplication**：形式上 $Y = WX$ 是矩阵乘，但语义上 $W$ 的每一行是一个 kernel 在空间展开，$X$ 的每一列是一个 channel 在 kernel element 维度展开。这就是 im2col + matmul 的光子版本。在软件里我们用 `im2col` 把卷积转成矩阵乘法，这里 paper 用数据预处理那一步把 28×28 图像 flatten 成 729×4 matrix，本质一样。

参考 im2col 原理: https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine-learning/deep-learning/convolutional-neural-networks/im2col

---

## 数据怎么流——从 RGB 图到 feature map

我走一遍完整 pipeline，你感受下数据是怎么变形的。

### 第 1 步：图像预处理

输入 86×86 RGB 图。拆成 3 个 86×86 monochrome matrix（R、G、B 各一个）。

用 $2 \times 2$ kernel 做 convolution，stride=1：
- 输出尺寸 = (86 - 2) / 1 + 1 = 85
- 每次滑动覆盖 4 个 pixel，总滑动次数 = 85 × 85 = 7225

把每个 monochrome matrix 重排成 **7225 行 × 4 列** 的 matrix：
- 行 = 7225 次 kernel slide
- 列 = kernel 的 4 个 element

### 第 2 步：时序展开

7225 × 4 matrix 按列拆成 4 个长度 7225 的 vector。这 4 个 vector 通过 AWG 的 4 个 channel 依次输出（time-division multiplexing）。

每个 vector 同时承载 3 个 microwave subcarrier（R/G/B 三个频率），所以一个 AWG channel 的 RF spectrum 是 3 个载波叠加。

### 第 3 步：光域计算

1. 4 个 laser 发 4 个波长（1555.48, 1557.74, 1559.97, 1562.25 nm）
2. 每个波长进一个 MZM，被对应 AWG channel 的 analog signal 调制
3. 4 个调制后的光通过 coupler 合并成一路
4. 进 MRR——每个波长按 Lorentzian 衰减实现权重乘法
5. SOA 补光损（MRR 和 coupler 都损耗了光）
6. PD 做光电转换，同时把同一波长上不同时间点的光强累加（这是加法操作）
7. Oscilloscope 采样波形

### 第 4 步：后处理

PD 输出的电信号里有 3 个 microwave subcarrier 叠加。用 3 个 local oscillator（7.67, 15.33, 23 GHz）做 down-conversion（混频到 baseband），再过 low-pass filter，就分离出 R/G/B 三个 channel 的 feature map。

这就是完整的数据流。

---

## 计算密度怎么算出来的

Paper 里的算法：

$$\text{Computing speed} = 30.67 \, \text{GBaud} \times 8 = 245.33 \, \text{GOPS}$$

$$\text{Computing density} = \frac{245.33 \, \text{GOPS}}{7207.50 \, \mu m^2} = 34.04 \, \text{TOPS/mm}^2$$

变量含义：
- 30.67 GBaud：每秒 30.67 G 个 symbol，每个 symbol 对应一次 MAC
- × 8：每个周期 8 个 operation。这 8 怎么来的——4 个 kernel 同时算 + 2 个时间点（或者 4 个 kernel + 2 个 modulation state），paper 说的含糊，需要看 supplementary。最合理的解释是 4 kernel × 2 polarity（MZM 的 ±Vπ/2 两个工作点代表两种符号）
- 7207.50 μm²：MRR 的面积，93 μm × 77.5 μm

**对比 Nvidia H100**：2.43 TOPS/mm²。这个 chip 是 H100 的 14 倍 density。

为什么差这么多？因为 electronic transistor 受 physical scaling limit（nm node 到了 3-5 nm 接近极限），interconnect RC delay 限制时钟频率，clock distribution network 占大量面积。Photonic device 不受这些 scaling law 约束，吞吐由 bandwidth × parallelism 决定，单器件就能跑 30+ GHz。

参考 H100 spec: https://resources.nvidia.com/en-us-datacenter-overview-resources/hpc-ai-datasheet

---

## 这套方案真正的商业价值——干掉高速 DA/AD

光计算 chip 长期有个尴尬：**光计算核心本身又快又省电，但前后端的 DA/AD 转换器又慢又耗电**。

一个 30 GHz 的 DAC 功耗几瓦到十几瓦，ADC 更夸张。Power 全被前后端吃了，光核心那点 fJ/OP 的优势被淹没。

这篇 paper 的微波副载波 trick 直接解这个问题：

**用 K 个低速 DA/AD 通道替代 1 个高速 DA/AD 通道**。

比如 30.67 GBaud 总速率，分到 3 个 subcarrier，每路 10 GBaud。10 GSa/s 的 DAC 功耗远低于 30 GSa/s 的 DAC。Paper 里说实际可以做到 ≤ 5 GSa/s，因为 demux 可以用 passive electrical mixer + filter 实现（模拟域处理，不需要 ADC）。

这是把 OFDM 通信里的 subcarrier 思路搬到 optical computing。通信领域这套玩了几十年了，非常成熟。

**直觉**：与其造一个超快但超贵的 DAC，不如造 3 个慢一点但便宜的 DAC，用频率把它们分开。在通信里这叫 frequency division multiplexing (FDM)，1960 年代就在用了。这篇 paper 把 FDM 从通信搬到计算，解决了一个长期困扰光计算落地的工程瓶颈。

参考 OFDM 在光通信中的应用: https://ieeexplore.ieee.org/document/4276468

---

## MNIST 实验——96.41% 准确率怎么来的

### 架构

```
28×28 input → Optical Conv (4 个 2×2 kernel) → ReLU → FC(E) → 10 logits
```

4 个 kernel 是 hand-crafted 的 edge detector：

$$K_1 = \begin{bmatrix} -1 & -1 \\ 1 & 1 \end{bmatrix}, \quad K_2 = \begin{bmatrix} 1 & 1 \\ -1 & -1 \end{bmatrix}, \quad K_3 = \begin{bmatrix} -1 & 1 \\ -1 & 1 \end{bmatrix}, \quad K_4 = \begin{bmatrix} 1 & -1 \\ 1 & -1 \end{bmatrix}$$

直觉上这就是 Sobel edge detector 的简化版：检测上、下、左、右四个方向的边缘。

- 28×28 输入 → stride=1, padding=0 → 输出 27×27 = 729 个 position
- flatten 成 729×4 matrix（4 列对应 kernel 的 4 个 element）
- 光域做完 conv → ReLU 激活 → electrical FC layer → 10 类输出

### 训练

- 60000 train + 10000 test
- 250 epochs
- Cross-entropy loss
- 实验精度 96.41% vs 理论 96.79%，0.38% gap 来自 optical noise（SOA 的 ASE noise、PD 的 shot noise、laser 的 RIN）

96.41% 这个数对单 conv layer 的浅网络是合理的。LeNet-1（1 conv + 1 FC）在 MNIST 上也就 96-97%。真正强的 LeNet-5 能到 99.2%，但要 2 个 conv layer + 2 个 FC layer。

参考 LeNet 在 MNIST 上的性能: http://yann.lecun.com/exdb/mnist/

---

## 跟其他方案对比

Paper 里 Table 1 的关键数据：

| 方案 | 平台 | MNIST Acc | Wavelength 数 | Data rate | 能耗/OP | Precision | Density |
|------|------|-----------|---------------|-----------|---------|-----------|---------|
| MZI mesh (Shen 2017) | Si | 76.70% | 1 | / | 15 fJ | 5-bit | 1.12 |
| MRR array (Filipovich 2022) | Si | 97.41% | 4 | 10 GBaud | 0.2 pJ | 4-bit | 5.78 |
| MRR (Bai 2023) | Si | 96.60% | 4 | 17 GBaud | 0.42 pJ | / | 1.04 |
| PCM (Feldmann 2021) | SiN | 95.30% | 36 | 2 GBaud | 2.5 pJ | 7-bit | 0.20 |
| MMI (Meng 2023) | SiN | 92.17% | 4 | 16.6 GBaud | 2.42 pJ | 5-bit | 25.48 |
| **Nvidia H100** | Si | / | / | 1.4 GBaud | 0.35 pJ | 8-bit | 2.43 |
| **This work** | Si | **96.41%** | 4 | **30.67 GBaud** | 3.75 pJ | 5-bit | **34.04** |

**密度上这篇碾压所有方案**，34.04 TOPS/mm² 比 MMI 的 25.48 还高 30%，比 H100 高 14 倍。

但 **能耗上这篇是最差的**，3.75 pJ/OP，比 H100 的 0.35 pJ/OP 高一个数量级。原因：
1. 4 个 external laser 功耗（每个 ~50-100 mW）
2. 4 个 MZM 的 driver 功耗
3. SOA 功耗
4. AWG 和高速 oscilloscope 是 lab 设备，功耗巨大

**精度只有 5-bit**，比 PCM 的 7-bit 和 H100 的 8-bit 都低。这限制了它的应用场景——做 inference 可以，做 training 难。

**kernel 是 hand-crafted 的，不是 learned**。这是这篇 paper 最大的局限：weight 由 laser wavelength 决定，laser 是 off-chip 的，gradient 没法 backprop 到 wavelength detuning。所以 optical layer 是 fixed feature extractor，training 只在 electrical FC layer。

参考对比工作的链接：
- MZI: https://www.nature.com/articles/nphoton.2017.93
- PCM: https://www.nature.com/articles/s41586-020-03070-1
- MMI: https://www.nature.com/articles/s41467-023-38602-w
- Microcomb: https://www.nature.com/articles/s41467-022-36773-7

---

## 这篇 paper 没说清楚的几个问题

### Q1: 8 个 operation 怎么来的？

Paper 说 "8 operations in one period"，但没明确解释。最合理的猜测是 4 个 kernel × 2 个 modulation polarity（MZM ±Vπ/2 两个工作点）。但也可能是 4 wavelength × 2 时间点。需要查 Supplementary Note 才能确认。

### Q2: 5-bit precision 怎么实现的？

Lorentzian 的斜边是连续的，理论上可以做任意精度。但 paper 用的是 binary weight（0 或 1），5-bit 可能来自多次测量平均 + laser wavelength tuning 的分辨率。laser tuning resolution 是 1 MHz，对应波长分辨率 ~10 fm，这足够高精度。

但 paper 没展示 weight 精度 vs 计算精度的实验，这块不够严谨。

### Q3: Scalability 真能 scale 吗？

增加 wavelength 数量：
- 优点：更多 kernel 并行，density 进一步提升
- 缺点：每个 wavelength 要一个 laser，成本和功耗线性增长
- FSR 内 channel 数受 Lorentzian linewidth 限制，估计 8-16 个 wavelength 是上限

Multicore 复制：
- Paper 在 Supplementary Note 7 讨论了
- 类似 GPU 的 multi-SM 架构，每个 SM 是一个 MRR
- 但 wavelength routing 复杂，多 core 之间 wavelength 冲突要处理

### Q4: 能不能 in-situ training？

当前不行。Weight 是 laser wavelength 决定的，gradient 无法穿过 wavelength detuning 反传。

可能的解法：
- **Differentiable photonic simulation**：用 photonic simulator 做 forward，gradient 通过 simulator 反传
- **In-situ training via SOA nonlinear**：利用 SOA 的 gain saturation 实现本地梯度计算（Wright 2022 Nature 的思路）
- **Hybrid training**：optical layer 做 fixed feature extraction，electronic layer 做 training（这篇 paper 用的就是这个）

参考 Wright 2022 in-situ training: https://www.nature.com/articles/s41586-021-04223-4

---

## 这篇 paper 的真正贡献——intuition 层面

如果让我总结这篇 paper 最值得记住的 idea：

**"More MACs 不需要 more devices，需要 more dimensions on one device."**

传统 optical computing 的 scaling law 是 $devices \sim O(N^2)$，这篇 paper 改写成 $devices \sim O(1) + wavelengths \sim O(N) + microwave\_freq \sim O(K)$。

四个维度正交叠在一个 MRR 上：
1. **Wavelength**：encode weight magnitude（通过 Lorentzian）
2. **Microwave frequency**：encode input channel（通过 FDM）
3. **Time**：encode spatial position（通过 TDM）
4. **MZM bias**：encode weight sign（通过 transfer function 对称性）

这就是为什么一个 90 微米见方的器件能做到 34 TOPS/mm²——它利用了光的所有可调制维度。

**人话总结**：以前光计算芯片像盖大楼，越多房间（器件）越好。这篇 paper 说，其实一个房间就够了，你只要在这个房间里同时开多盏灯（不同波长）、放多个广播（不同微波频率）、在不同时间点干活（时序），一个房间能干 100 个房间的活。

---

## 可能的下一步演进

1. **On-chip tunable laser**（Komljenovic 2015, ref 39）：把 external laser 换成 on-chip laser array，整个系统集成度再上一个台阶
2. **Microcomb source**（Bai 2023, ref 36）：用一个 microcomb 产生几十个 wavelength，替代多个 laser，成本和功耗大降
3. **Coherent detection**：当前是 direct detection，intensity only。用 coherent + LO 可以做 phase encoding，precision 大幅提升
4. **In-situ training**：这是光计算 chip 真正落地的关键。Wright 2022 展示了可能路径，但要搬到 MRR 架构上还有大量工程问题
5. **ImageNet scale demo**：当前只做了 MNIST。要在 ImageNet 上跑 ResNet 才有说服力
6. **Transformer attention**：CNN conv 只是 tensor operation 的一种。Attention 的 QKV matmul 也能用这套架构做吗？理论上可以，但 softmax 和 layer norm 这种非线性操作要回到电子域

参考 microcomb optical computing: https://www.nature.com/articles/s41467-022-36773-7

参考 on-chip tunable laser: https://ieeexplore.ieee.org/document/7080294

---

## 最后吐槽两句

这篇 paper 的 demo 距离实用还远。96.41% MNIST 在 2025 年是个非常弱的 demo。真正的杀手 app 应该是 ImageNet 上的 ResNet-50 inference，或者 transformer 的 attention 加速。

而且能耗 3.75 pJ/OP 比 H100 的 0.35 pJ/OP 差 10 倍。Density 高不代表 energy efficient。这篇 paper 的卖点是 footprint 小，适合端侧 edge device。但端侧 device 对能耗敏感，这又矛盾了。

不过从 research 角度看，**single device + multi-dimension multiplexing** 这个 paradigm 是真正有 insight 的。它告诉光计算社区：别再堆器件了，去挖光的物理维度。光有 wavelength、phase、polarization、mode、time、frequency 这些维度，每个维度都可以 encode 信息。你把它们都利用起来，一个器件顶过去一万个器件。

这是光计算相对于电子计算的真正优势——电子 transistor 基本只有 voltage 和 charge 两个自由度，光有十几个。这篇 paper 只用了 wavelength + microwave freq + time 三个，还剩 polarization、mode、OAM、phase 等没用。未来如果把所有维度都榨干，density 还能再提一个数量级。

这才是这篇 paper 给我们的真正启示：**光的 parallelism 是多维度的，don't waste it**。

参考 multi-dimensional optical computing review: https://www.nature.com/articles/s41586-020-03063-0

---

# 这篇 paper 的核心直觉

让我从直觉出发，把这篇 paper 的关键 idea 拆开讲。

---

## 1. 一句话概括

这篇文章提出用一个 **single microring resonator (MRR)**，借助 **wavelength + time + microwave frequency** 三个维度同时 multiplexing，实现 **tensor convolution**（不是 matrix convolution），并且在硅光平台上做到了 **34.04 TOPS/mm²** 的 computing density——比 MZI mesh、PCM array、MMI 等方案高一个数量级。

直觉上，传统 optical computing 的瓶颈是 "more MACs = more devices = bigger chip"。这篇 paper 的逻辑是 **"more MACs = more dimensions on ONE device"**——把 N² 个 MRR 压成 1 个 MRR，靠维度复用而不是器件堆叠来 scale。

---

## 2. 为什么传统方案遇到瓶颈

### 2.1 MZI mesh (Shen et al., Nature Photonics 2017)

SVD decomposition: $W = U \Sigma V^\dagger$，需要 $N \times N$ 的 MZI mesh 实现 $N \times N$ matrix。
- 优点：coherent、可训练
- 缺点：器件数 $\sim O(N^2)$，cascaded loss 累积，calibration 复杂

### 2.2 MRR array (Xu et al., Nature 2021; Filipovich 2022)

每个 weight 一个 MRR，靠 thermo-optic effect 调谐。
- 问题：thermal crosstalk → 必须拉大 MRR 间距 → footprint 膨胀
- substrate hollowing、复杂反馈电路 → fabrication 复杂

### 2.3 PCM (Feldmann et al., Nature 2021)

用 phase-change material 做 non-volatile weight。
- 优点：存算一体
- 缺点：weight precision 受 crystallization 控制，reconfiguration 慢，36 wavelengths 才到 0.20 TOPS/mm²

### 2.4 MMI (Meng et al., Nature Comm 2023)

用 multimode interference 自成像做卷积，4 wavelengths 已经做到 25.48 TOPS/mm²——是这篇文章之前的 record。但 mode coupling 限制了进一步 scale。

这篇文章的 insight：**抛弃 array，回到单个 resonator，把所有信息塞进它的 frequency response**。

参考链接：
- MZI mesh: https://www.nature.com/articles/nphoton.2017.93
- PCM tensor core: https://www.nature.com/articles/s41586-020-03070-1
- MMI conv: https://www.nature.com/articles/s41467-023-38602-w
- MRR 11 TOPS: https://www.nature.com/articles/s41586-020-03070-1

---

## 3. MRR 作为 weight 的物理直觉

### 3.1 Lorentzian transmission

Microring resonator 的传输谱是 Lorentzian：

$$T(\lambda) = 1 - \frac{(1-a)\kappa^2}{(1-a\kappa)^2 + 4a\kappa \sin^2(\phi/2)}$$

- $a$：round-trip amplitude ($a = e^{-\alpha L/2}$, $\alpha$ 是 waveguide loss，$L$ 是 ring 周长)
- $\kappa$：coupling coefficient (bus 和 ring 之间的耦合强度)
- $\phi = 2\pi n_g L / \lambda$：round-trip phase，$n_g$ 是 group index，$\lambda$ 是 wavelength

当 $\lambda = \lambda_{res}$（共振）时，光能量几乎全被 ring 困住，through port 输出接近 0 → weight = "0"。

当 $\lambda$ 远离 $\lambda_{res}$（off-resonance）时，光直接 through，几乎不损耗 → weight = "1"。

**核心直觉**：MRR 是一个天然的 **wavelength-selective attenuator**。你不需要去改变 MRR，只需要改变激光波长就能调权重。这就是这篇 paper 的关键 trick：**weight 写在 laser wavelength 上，不写在 device 上**。

### 3.2 FSR (Free Spectral Range) 复用

MRR 的 transmission spectrum 是周期性的，FSR = $\lambda^2 / (n_g L)$。

这篇文章里 FSR 内放 4 个 wavelengths，对应一个 kernel 的 4 个 elements。每个 wavelength 通过独立 laser 调谐到 on/off resonance 实现 weight = 0/1。如果需要 intermediate weight，把 wavelength 调到 Lorentzian 的斜边即可（虽然 paper 主用 binary weight，5-bit precision 来自多次测量）。

**为什么是 thermally tuned-free**：传统 MRR array 要 thermo-optic 调 ring 的 $n_g$ 来移动 $\lambda_{res}$。这里完全不动 ring，只调 laser——避免了 thermal crosstalk。整个 chip 只用一个 TEC (thermoelectric cooler) 维持恒温。

---

## 4. Microwave subcarrier：tensor 而不是 matrix 的关键

### 4.1 从 matrix-vector 到 matrix-matrix

传统 WDM optical computing：用 $N$ 个 wavelengths 代表 $N$ 个 weights，做的是 $W \cdot x$（matrix-vector）。

要扩展到 tensor（multi-channel input），需要额外维度。这篇文章引入 **microwave frequency domain**：

- 3 个 microwave subcarrier $f_1, f_2, f_3$ = 7.67, 15.33, 23 GHz 分别对应 R/G/B 三个 channel
- 4 个 wavelengths $\lambda_1, \lambda_5, \lambda_9, \lambda_{13}$ 对应一个 kernel 的 4 个 weights

这样 $4 \times 4$ weight matrix $W$ 和 $4 \times 3$ input matrix $X$ 一次性完成乘法，输出 $4 \times 3$ feature map $Y$。这就是 tensor convolution。

### 4.2 公式 (1) 解析

$$
Y = W \cdot X
$$

$$
\begin{bmatrix} y_{R1} & y_{G1} & y_{B1} \\ y_{R2} & y_{G2} & y_{B2} \\ y_{R3} & y_{G3} & y_{B3} \\ y_{R4} & y_{G4} & y_{B4} \end{bmatrix} =
\begin{bmatrix} w_{11} & w_{12} & w_{13} & w_{14} \\ w_{21} & w_{22} & w_{23} & w_{24} \\ w_{31} & w_{32} & w_{33} & w_{34} \\ w_{41} & w_{42} & w_{43} & w_{44} \end{bmatrix}
\begin{bmatrix} x_{R1} & x_{G1} & x_{B1} \\ x_{R2} & x_{G2} & x_{B2} \\ x_{R3} & x_{G3} & x_{B3} \\ x_{R4} & x_{G4} & x_{B4} \end{bmatrix}
$$

变量含义：
- $Y \in \mathbb{R}^{4 \times 3}$：输出 feature map。下标 $R/G/B$ 表示 color channel，$1..4$ 表示 4 个 kernel slide 位置（4 个 kernel）
- $W \in \mathbb{R}^{4 \times 4}$：4 个 $2 \times 2$ kernel flatten 成的 weight matrix。$w_{ij}$ 中 $i$ 是 kernel index (1-4)，$j$ 是 kernel 内 element index (1-4)
- $X \in \mathbb{R}^{4 \times 3}$：input data。行 index 是 kernel element (1-4)，列 index 是 R/G/B channel

**关键 insight**：列 index（color channel）映射到 microwave frequency，行 index（kernel element）映射到 wavelength。两个维度正交，所以一个 MRR 同时处理所有 12 个 MAC 操作。这就是为什么单 MRR 能做到 tensor 而不是被 matrix 限制。

---

## 5. 负权重怎么实现——MZM quadrature bias trick

### 5.1 问题

Optical intensity 永远 $\geq 0$。直接用 intensity 表示 weight 没法表达负数。传统方案：
- **Balanced photodetector** (Xu 2021)：两路光相减
- **Electrical subtraction** (Meng 2022/2023)：PD 后电路减
- **Phase differential** (Xu 2021 Light Sci Appl)：用 phase 编码

### 5.2 这篇 paper 的 trick

用 MZM 的 **transfer function**：

$$P_{out} = P_{in} \cos^2\left(\frac{\pi V}{V_\pi}\right) = \frac{P_{in}}{2}\left[1 + \cos\left(\frac{2\pi V}{V_\pi}\right)\right]$$

- $V$：施加的 bias + signal 电压
- $V_\pi$：half-wave voltage，产生 $\pi$ phase shift 所需的 DC 电压

如果 small-signal modulation，工作点设在 quadrature point（cos 曲线线性区中点）：

- **Q⁺ point**：$V_{bias} = -V_\pi/2$，transfer function 斜率为正 → $\Delta P_{out} \propto +\Delta V_{sig}$
- **Q⁻ point**：$V_{bias} = +V_\pi/2$，transfer function 斜率为负 → $\Delta P_{out} \propto -\Delta V_{sig}$

也就是说，**同一个 microwave signal 调到 MZM，bias 不同就能输出正或负的 optical intensity 调制**。这就把 weight 的 sign 写到了 MZM 的 bias 上，weight 的 magnitude 写到了 laser wavelength 上。

$$w_{ij} = \text{sign}(V_{bias}) \cdot T(\lambda_{ij})$$

其中 $T(\lambda)$ 是 MRR 的传输率，$\text{sign}$ 由 MZM 决定。

**直觉**：MZM 的 transfer function 在 $\pm V_\pi/2$ 处对称，所以同一个 AC signal 偏置在不同象限会产生 180° 相位差的光强调制。PD 是平方律器件不区分相位，但调制幅度一正一负，等效于 weight sign 翻转。这是非常优雅的设计——不需要双路光，不需要额外 PD。

---

## 6. 数据预处理 pipeline（构建计算直觉）

### 6.1 RGB image → tensor 编码

输入 86×86 RGB image → 拆成 3 个 86×86 monochrome matrix (R, G, B)。

每个 monochrome matrix 用 2×2 kernel 做 convolution，stride=1：
- output size = (86-2)/1 + 1 = 85
- 总滑动次数 = 85 × 85 = 7225

所以每个 channel 被重排成 **7225 × 4** 的 matrix：
- 行 = 7225 次 kernel slide
- 列 = kernel 的 4 个 element (2×2)

### 6.2 时序展开

7225 × 4 matrix 按列拆成 4 个长度 7225 的 vector。这 4 个 vector 通过 4 个 AWG channel 时序输出（time-division multiplexing）。

每个 vector 同时承载 3 个 microwave subcarrier（对应 R/G/B），所以一个 AWG channel 的 RF spectrum 是 3 个载波叠加。

### 6.3 整体数据流

1. AWG 输出 4 通道 analog signal（每通道包含 R/G/B 三个 subcarrier）
2. 4 个 laser $\lambda_1, \lambda_5, \lambda_9, \lambda_{13}$ 进 4 个 MZM，分别被调制
3. 4 个被调制的光在 coupler 合并
4. 进 MRR，按 wavelength 各自加权
5. SOA 补偿 loss
6. PD 做光电转换 + summation（同一 wavelength 的 4 个时间点积分）
7. Oscilloscope 采样
8. 数字后处理：3 路 down-conversion (7.67/15.33/23 GHz) + low-pass filter → 恢复 3 个 channel feature map

---

## 7. Computing density 计算细节

**单个 convolution 周期**包含 8 个 MAC：
- 一个 $2 \times 2$ kernel 滑过 input 一次 = 4 个 multiply + 1 个 add = 4 MAC
- 但因为 3 个 channel (R/G/B) 同时算（microwave multiplexing），实际 4 × 3 = 12 MAC per kernel per position
- 但 paper 里只算 4 个 kernel 同时算 = 8 ops（这里有点微妙，应该是 4 kernel × 2 模式简化）

Paper 公式：

$$\text{Computing speed} = 30.67 \, \text{GBaud} \times 8 = 245.33 \, \text{GOPS}$$

$$\text{Computing density} = \frac{245.33 \, \text{GOPS}}{7207.50 \, \mu m^2} = 34.04 \, \text{TOPS/mm}^2$$

- 30.67 GBaud：每秒 30.67 G 个 symbol，每个 symbol 是一次 MAC
- $\times 8$：因为 4 wavelengths + 2 microwave subcarriers（or 4 kernel + 2 polarization？paper 解释含糊，应该是 4 个 kernel 同时 + 2 个时间点，需要看 supplementary）

**对比表数据**（重排一下）：

| Type | Platform | MNIST Acc | Wavelengths | Data rate (GBaud) | Energy/OP | Precision | Density (TOPS/mm²) |
|------|----------|-----------|-------------|-------------------|-----------|-----------|---------------------|
| MZI (Shen 2017) | Si | 76.70% | 1 | / | 15 fJ | 5-bit | 1.12 |
| MRR (Filipovich 2022) | Si | 97.41% | 4 | 10 | 0.2 pJ | 4-bit | 5.78 |
| MRR (Bai 2023) | Si | 96.60% | 4 | 17 | 0.42 pJ | / | 1.04 |
| PCM (Feldmann 2021) | SiN | 95.30% | 36 | 2 | 2.5 pJ | 7-bit | 0.20 |
| MMI (Meng 2023) | SiN | 92.17% | 4 | 16.6 | 2.42 pJ | 5-bit | 25.48 |
| **Nvidia H100** | Si | / | / | 1.4 | 0.35 pJ | 8-bit | 2.43 |
| **This work** | Si | **96.41%** | 4 | **30.67** | 3.75 pJ | 5-bit | **34.04** |

注意：**energy efficiency 这篇是 3.75 pJ/OP，反而是最差的**——主要因为 SOA + 多个 laser + AWG 功耗。但 density 极高，适合 footprint-limited 场景。

参考 Nvidia H100 specs: https://resources.nvidia.com/en-us-datacenter-overview-resources/hpc-ai-datasheet

---

## 8. MNIST 实验细节

### 8.1 架构

```
28×28 input → Optical Conv (4× 2×2 kernels, stride?) → ReLU → FC(E) → 10 logits
```

- Optical conv layer：4 个 kernel，分别是
  $\begin{bmatrix}-1 & -1 \\ 1 & 1\end{bmatrix}$, $\begin{bmatrix}1 & 1 \\ -1 & -1\end{bmatrix}$, $\begin{bmatrix}-1 & 1 \\ -1 & 1\end{bmatrix}$, $\begin{bmatrix}1 & -1 \\ 1 & -1\end{bmatrix}$
  这 4 个 kernel 显然是 **edge detectors**（上、下、左、右 edge）
- 28×28 → flatten 成 729 × 4 matrix（应该是 stride=2，output = 14×14 = 196，×4 kernel = 784；或 729 = 27×27，stride=1）

实际上 $729 = 27^2$，所以是 stride=1，padding=0，output = (28-2)+1 = 27 → 27×27 = 729 positions。

### 8.2 训练

- 60000 train + 10000 test
- 250 epochs
- 损失：cross-entropy
- 实验精度 96.41% vs 理论 96.79% → 0.38% gap，来自 optical noise / SOA ASE / PD shot noise

### 8.3 为什么只用 1 个 conv layer 就到 96.41%

这是经典 LeNet-style 浅网络，但 optical conv 的 kernel 是 hand-crafted edge detector（不是 learned）。FC layer 在 electrical domain 做 classification。理论上 LeNet-1（1 conv + 1 FC）也能到 ~96-97%。

参考 LeNet 性能: http://yann.lecun.com/exdb/mnist/

---

## 9. 实验细节里值得注意的点

### 9.1 Wavelength 选择

- 1555.48 nm, 1557.74 nm, 1559.97 nm, 1562.25 nm
- 间距约 2.24 nm（对应 FSR 内 channel spacing）
- 都在 MRR transmission 的 flat-top（off-resonance 端）以保证 weight magnitude = "1"
- 对 Lorentzian 的斜边区域代表 intermediate weight

### 9.2 Microwave subcarrier frequency

- $f_1 = 7.67$ GHz (R channel)
- $f_2 = 15.33$ GHz (G channel)
- $f_3 = 23.00$ GHz (B channel)
- 等间距 7.67 GHz → 整数倍，便于同步和 down-conversion
- Data rate per subcarrier = 1.92 GBaud (彩色图像 demo)
- MNIST 时升到 30.67 GBaud（grey-scale，只需 1 channel）

### 9.3 器件参数

- MRR: 93 μm × 77.5 μm = 7207.5 μm², insertion loss < 0.1 dB
- Edge coupler loss: 1.5 dB/facet
- 200mm wafer, 90 nm lithography (这里 paper 写 90nm，应该是 220nm SOI + 90nm 某个工艺细节，标准 IME AMF 流程)
- TEC 控温
- FSR = 2.24 nm channel spacing, flatness 1.51 dB
- Laser: IDPHOTONICS CoBrite-DX，1 MHz tuning resolution
- AWG: Keysight M8196A, 92 GSa/s
- Modulator: iXblue MX-LN-40 (MZM, $V_\pi$ 大约 3-5V)
- PD: Finisar XPDV2150R, 50 GHz BW
- OSC: Keysight UXR0402A, 256 GSa/s

---

## 10. 这套方案的真正价值——避免 high-speed DA/AD

### 10.1 现有光计算芯片的 power wall

Feldmann (PCM) 2021 在 Nature 文章里指出，optical accelerator 的瓶颈往往不在光计算本身，而是高速 DA/AD 转换：
- 30+ GHz DAC 功耗几瓦到十几瓦
- 30+ GHz ADC 更夸张
- Power 被前端后端吃光，光计算 core 本身那点 fJ/OP 优势被淹没

### 10.2 Microwave subcarrier 的 trick

这篇 paper 的核心商业价值：**用 $K$ 个低速 DA/AD 通道替代 1 个高速 DA/AD 通道**。

例如 30.67 GBaud 总速率，分到 3 个 subcarrier，每路 10 GBaud → 用 10 GSa/s DAC（功耗低、成本低）。

理论上 paper 说 ≤5 GSa/s 就够（因为可以用 passive electrical mixer + filter 做 demux，不需要高速 ADC）。这是把 OFDM 通信里的 subcarrier 思路搬到 optical computing。

参考 OFDM 在 optical 的应用: https://ieeexplore.ieee.org/document/4276468

### 10.3 不足与潜在 issue

- **3.75 pJ/OP 能耗偏高**：MZM 驱动 + 4 个 laser + SOA + 高速 PD，整套 lab bench 功耗远超 chip 本身
- **5-bit precision**：weight 限制在 5-bit，比 PCM 的 7-bit、H100 的 8-bit 都低
- **Demo 浅**：只验证 MNIST，没 ImageNet/CIFAR
- **Kernel 是 hand-crafted**：4 个固定 edge detector，没有 training backprop 到 optical layer（这正是 Lightelligence 的 Yichen Shen 作者团队可能想下一步做）
- **Scalability**：增加 wavelength 数量会增加 laser 成本；FSR 内 channel 数受限于 Lorentzian 的 sharpness

---

## 11. 与同类工作的 deeper 对比

### 11.1 vs. Dong et al. 2023 (Nature Photonics, ref 21)

Dong 的工作也是 microwave subcarrier + MRR，但用 MRR **array** 做高维 tensor。这篇 paper 的进步是用 **single MRR + 多 wavelength** 替代 array，把 footprint 压到底。

参考: https://www.nature.com/articles/s41566-023-01296-5

### 11.2 vs. Lightelligence (Shen et al., Nature Photonics 2017)

Yichen Shen 是这篇 paper 的 co-author（ affiliation 6: Lightelligence Group）。他的 MZI mesh 工作是光计算的里程碑，但器件数 $\sim N^2$。这篇 paper 用 MRR 的多波长实现 $\sim 1$ device，是路线上的根本转变。

参考: https://www.nature.com/articles/nphoton.2017.93

### 11.3 vs. Chen et al. 2023 Nature (ACAM chip, ref 29)

Chen 2023 的 all-analog photoelectronic chip 也做高速 vision task，但用的是 analog electronic + photonic 混合，不是 pure photonic tensor。这篇 paper 走的是相反方向：把 electronic 功能尽量压到 photonic。

参考: https://www.nature.com/articles/s41586-023-06658-6

---

## 12. 公式 (1) 的更深解读：为什么这是 "tensor convolution" 而不是 "matrix multiplication"

形式上 Eq.(1) 就是 matrix-matrix 乘法 $Y = WX$。但作者叫它 tensor convolution，因为：

- $W$ 的每一行是 **一个 kernel 在不同空间位置的展开**
- $X$ 的每一列是 **一个 input channel 在不同 kernel element 上的展开**
- $Y$ 的每个元素 $y_{c,k}$ = kernel $k$ 与 channel $c$ 的 spatial convolution 在某位置的值

直觉：**普通的 matrix mult 是 $A \times B$，tensor convolution 是把 $A$ 拆成空间-通道双索引，$B$ 拆成 kernel-空间双索引，然后 batch multiply**。

这正是深度学习里 `Conv2d` 在底层被实现为 `im2col + matmul` 的光学版本——`im2col` 是数据预处理那一步（把 28×28 转成 729×4 matrix）。

参考 im2col: https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine-learning/deep-learning/convolutional-neural-networks/im2col

---

## 13. 直觉总结

这篇文章的 **core trick 链条**：

1. **Single device replaces array** → 解决 thermal crosstalk + footprint
2. **Wavelength encodes weight magnitude** → 利用 MRR Lorentzian 自然做 attenuation，无需 thermo-optic
3. **MZM bias encodes weight sign** → 利用 cos² transfer function 的对称性，免平衡 PD
4. **Microwave subcarrier encodes input channel** → 把 matrix mult 升级为 tensor conv，同时低速化 DA/AD
5. **Time-division handles spatial sliding** → 完成 im2col 的等价展开

四个维度（wavelength, microwave freq, time, MZM bias）正交叠在一个 MRR 上，所以 $O(N^2)$ device 变成 $O(1)$ device + $O(N)$ lasers。

**为什么 H100 是 2.43 TOPS/mm²，这个 chip 是 34.04**：因为 electronic transistor 的物理尺寸下限（~nm node）+ interconnect RC delay + clock distribution 限制了 density。Photonic device 不受同样 scaling law 约束，单器件吞吐由 BW × parallelism 决定。

**为什么这篇没用 backprop training optical layer**：因为 weight 由 laser wavelength 决定，laser 是 off-chip 的，没法 end-to-end 反传梯度（梯度无法穿过 wavelength detuning）。所以 optical layer 是 fixed feature extractor，training 只在 FC layer。这是当前光计算芯片的普遍局限，也是下一步要解决的——in-situ training (Wright 2022 Nature) 或 differentiable photonic simulation。

参考 Wright 2022: https://www.nature.com/articles/s41586-021-04223-4

---

## 14. 可能的延伸方向

- **On-chip tunable laser integration** (ref 39 Komljenovic)：把 4 个 external laser 换成 on-chip tunable laser array，进一步缩小系统
- **Wavemeter for precise calibration** (ref 40 Yao patent)：解决 laser wavelength drift
- **Multicore OTPU**：复制 MRR + wavelength routing，做大规模 parallel（类似 GPU 的 SM）
- **Comb source** (Bai 2023 microcomb, ref 36)：用 microcomb 替代 multi-laser，一个 comb 产生几十个 wavelengths，密度再上一个数量级
- **Coherent detection**：当前是 direct detection，intensity only。如果用 coherent detection + LO，可以做 phase encoding，weight precision 大幅提升
- **In-situ training**：用 SOA 的 gain saturation 或 MRR 的 nonlinear effect 实现本地梯度计算

参考 microcomb optical computing: https://www.nature.com/articles/s41467-022-36773-7

---

## 15. 总结表格

| 维度 | 传统 MRR array | MZI mesh | PCM | This work (single MRR + microwave) |
|------|---------------|----------|-----|-----------------------------------|
| 器件数 | $O(N^2)$ | $O(N^2)$ | $O(N^2)$ | $O(1)$ |
| Weight 机制 | Thermo-optic | Phase shifter | Phase transition | Wavelength tuning |
| Sign 处理 | Balanced PD | Phase | / | MZM quadrature bias |
| 多通道 | WDM | Spatial | WDM | WDM + microwave FDM |
| Thermal crosstalk | 严重 | 中等 | 无 | 无 |
| Computing density | 1-6 TOPS/mm² | 1 TOPS/mm² | 0.2 TOPS/mm² | **34 TOPS/mm²** |
| DA/AD 要求 | 高速 | 高速 | 高速 | 低速多通道 |
| Energy/OP | 0.2-2.5 pJ | 15 fJ | 2.5 pJ | 3.75 pJ |
| MNIST acc | 96-97% | 76% | 95% | 96.41% |
| In-situ training | 可行 | 可行 | 困难 | 困难 |

---

## 关键参考链接汇总

- **本 paper (Light Sci & Appl 2025)**: https://doi.org/10.1038/s41377-024-01706-9
- **Shen MZI mesh (Nature Photonics 2017)**: https://www.nature.com/articles/nphoton.2017.93
- **Feldmann PCM (Nature 2021)**: https://www.nature.com/articles/s41586-020-03070-1
- **Xu 11 TOPS (Nature 2021)**: https://www.nature.com/articles/s41586-020-03070-1
- **Dong higher-dim photonic tensor (Nature Photonics 2023)**: https://www.nature.com/articles/s41566-023-01296-5
- **Meng MMI (Nature Comm 2023)**: https://www.nature.com/articles/s41467-023-38602-w
- **Bai microcomb (Nature Comm 2023)**: https://www.nature.com/articles/s41467-022-36773-7
- **Chen ACAM chip (Nature 2023)**: https://www.nature.com/articles/s41586-023-06658-6
- **Wright in-situ training (Nature 2022)**: https://www.nature.com/articles/s41586-021-04223-4
- **Wetzstein review (Nature 2020)**: https://www.nature.com/articles/s41586-020-03063-0
- **MNIST database**: http://yann.lecun.com/exdb/mnist/
- **LeNet original**: https://ieeexplore.ieee.org/document/726791

这篇 paper 的优雅之处在于：**它没有发明新物理，只是把 MRR 的 frequency response、MZM 的 transfer function、WDM、OFDM 这几个成熟技术拼在一起，用极简的 single-device 架构实现 tensor operation**。这正是 Karpathy 你常说的 "simple things that scale" 的精神——只是这里 scale 的维度从 parameter count 变成了 optical multiplexing dimension。
