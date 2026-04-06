**Allan variance (AVAR)**，也称为 **two-sample variance (双样本方差)**，是衡量 **clocks**（时钟）、**oscillators**（振荡器）和 **amplifiers**（放大器）**frequency stability**（频率稳定性）的统计量。它由 **David W. Allan** 在1966年提出。

**Allan deviation (ADEV)**，也称为 **sigma-tau**，是 Allan variance 的平方根。

想象两个相邻的时间间隔测量：

```
参考时钟测量间隔 τ
待测时钟在这两个间隔内分别前进 τy 和 τy'
```

如果时钟非常稳定，那么 y - y′ 的值应该很小。重复这个过程多次，(y - y′)² 的平均值就等于 **2倍的 Allan variance**。

**公式解释**：
```
σ²y(τ) = ½⟨(y - y′)²⟩
```

其中：
- σ²y(τ) = Allan variance
- y = 第一个时间间隔内的平均相对频率
- y′ = 第二个相邻时间间隔内的平均相对频率  
- τ = observation time（观测时间/采样周期）
- ⟨...⟩ = time average（时间平均）或 expectation operator（期望算子）

---

## 二、数学定义与公式详解

### 2.1 Fractional Frequency（分数频率）

首先定义 **fractional frequency（分数频率）**：

```
y(t) = [ν(t) - ν0] / ν0 = Δν(t)/ν0
```

变量说明：
- y(t) = fractional frequency at time t
- ν(t) = actual frequency（实际频率）
- ν0 = nominal frequency（标称频率/额定频率）
- Δν(t) = frequency deviation（频率偏差）

### 2.2 Average Fractional Frequency（平均分数频率）

```
ȳk = (1/τ) ∫ₖᵗ⁽ᵏ⁺¹⁾ᵗ y(t)dt
```

或者：
```
ȳk = [x(tₖ₊τ) - x(tₖ)] / τ
```

变量说明：
- ȳk = k-th average fractional frequency（第k个平均分数频率）
- τ = observation period（观测周期）
- x(t) = time error function（时间误差函数）
- tₖ = k-th time sample point（第k个采样时间点）

### 2.3 M-sample Variance（M样本方差）

```
σ²y(M,T,τ) = 1/(M-1) Σᵢ₌₁ᴹ⁻¹ (ȳi - μ)²
```

其中：
- M = number of samples（样本数量）
- T = time between measurements（测量间隔时间）
- τ = observation time per sample（每个样本的观测时间）
- μ = mean of the average fractional frequency（平均分数频率的均值）
- ȳi = i-th average fractional frequency sample（第i个平均分数频率样本）

### 2.4 Allan Variance 精确定义

```
σ²y(τ) = ½⟨(ȳk₊₁ - ȳk)²⟩
```

或者用 time error 表示：

```
σ²y(τ) = ½⟨[x(tₖ₊₂τ) - 2x(tₖ₊τ) + x(tₖ)]² / τ²⟩
```

条件：T = τ（即无 dead time 测量）

### 2.5 Allan Deviation

```
σy(τ) = √[σ²y(τ)]
```

---

## 三、Oscillator Model（振荡器模型）

### 3.1 基本模型

```
V(t) = V₀ sin[Φ(t)]
```

分解为：
```
Φ(t) = 2πν₀t + φ(t)
```

变量：
- V₀ = signal amplitude（信号幅度）
- Φ(t) = total phase（总相位）
- ν₀ = nominal frequency（标称频率）
- φ(t) = phase fluctuation（相位波动）

### 3.2 Angular Frequency（角频率）

```
ω₀ = 2πν₀
```

其中 ω₀ = nominal angular frequency（标称角频率），单位：rad/s

### 3.3 Time Error Function（时间误差函数）

```
x(t) = Φ(t)/(2πν₀) - t = φ(t)/(2πν₀)
```

物理意义：expected nominal time 与 actual normal time 之间的差值

---

## 四、Power-law Noise（幂律噪声）详解

Allan variance 的核心价值在于能够区分不同的噪声类型。以下是**五种典型噪声类型**：

### 4.1 Noise Power Spectral Density（噪声功率谱密度）

**Phase noise PSD**（相位噪声功率谱）：
```
Sφ(f) = hα f^α
```

**Frequency noise PSD**（频率噪声功率谱）：
```
Sy(f) = (f/ν₀)² Sφ(f) = hβ f^β = hα f^(α+2)
```

变量：
- Sφ(f) = phase noise spectral density
- Sy(f) = frequency noise spectral density
- f = frequency offset（频偏）
- hα = power coefficient（功率系数）
- α = phase noise exponent（相位噪声指数）
- β = frequency noise exponent（频率噪声指数）

关系：β = α + 2

### 4.2 五种噪声类型详表

| **Noise Type** | **Phase Noise Slope (α)** | **Frequency Noise Slope (β)** | **μ (Allan Variance)** | **σy(τ) Dependence on τ** |
|---------------|--------------------------|------------------------------|------------------------|---------------------------|
| **White PM** (WPM) | 0 | +2 | 1 | τ^(-1) |
| **Flicker PM** (FPM) | -1 | +1 | 1 | τ^(-1) |
| **White FM** (WFM) | -2 | 0 | -1 | τ^(-1/2) |
| **Flicker FM** (FFM) | -3 | -1 | -2 | τ^0 (constant) |
| **Random Walk FM** (RWFM) | -4 | -2 | -2 | τ^(+1/2) |

其中 Allan variance 可以表示为：
```
σ²y(τ) = Kα τ^μ
```

### 4.3 各噪声类型深入解析

#### **(1) White Phase Modulation (WPM)**
- **物理起源**：thermal noise（热噪声）、shot noise（散粒噪声）
- **特点**：相位噪声是平坦的（Sφ(f) ∝ f⁰）
- **Allan variance**：σy(τ) ∝ τ^(-1)
- **物理直觉**：短时间相位波动大，长时间平均后快速衰减

#### **(2) Flicker Phase Modulation (FPM)**
- **物理起源**：半导体中的 flicker noise（1/f噪声）
- **特点**：Sφ(f) ∝ f^(-1)
- **Allan variance**：σy(τ) ∝ τ^(-1) [与WPM无法区分！]
- **需用 Modified Allan Variance 区分**

#### **(3) White Frequency Modulation (WFM)**
- **物理起源**：放大器的白噪声
- **特点**：Sy(f) ∝ f⁰（平坦的频率噪声）
- **Allan variance**：σy(τ) ∝ τ^(-1/2)
- **物理意义**：频率随机游走的白噪声驱动

#### **(4) Flicker Frequency Modulation (FFM)**
- **物理起源**：晶体振荡器的 flicker noise
- **特点**：Sy(f) ∝ f^(-1)，也称为 "pink noise"
- **Allan variance**：σy(τ) ∝ τ⁰（常数）
- **物理意义**：频率的慢速波动，不随平均时间改善

#### **(5) Random Walk Frequency Modulation (RWFM)**
- **物理起源**：aging effects（老化效应）、环境温度变化
- **特点**：Sy(f) ∝ f^(-2)，也称为 "brown noise"
- **Allan variance**：σy(τ) ∝ τ^(+1/2)
- **物理意义**：频率执行 random walk（随机游走），随时间恶化

---

## 五、α–μ Mapping（指数映射表）

| **α** | **β** | **μ** | **Noise Type** |
|-------|-------|-------|---------------|
| -2 | -4 | 1 | White PM |
| -1 | -3 | 1 | Flicker PM |
| 0 | -2 | -1 | White FM |
| 1 | -1 | -2 | Flicker FM |
| 2 | 0 | -2 | Random Walk FM |

公式推导关系：
```
β = α + 2
```

---

## 六、Estimators（估计器）详解

### 6.1 Fixed τ Estimator（固定τ估计器）

```
σ²y(τ) = 1/(2M) Σᵢ₌₁ᴹ⁻¹ (ȳi₊₁ - ȳi)²
```

变量：
- M = number of samples
- ȳi = i-th average fractional frequency

### 6.2 Non-overlapped Variable τ Estimator（非重叠变τ估计器）

```
σ²y(nτ₀) = 1/[2(M-n)] Σᵢ₌₁ᴹ⁻ⁿ (ȳi₊ₙ - ȳi)²
```

其中：
- τ₀ = basic measurement period
- n = integer multiplier
- τ = nτ₀ = desired observation time

**缺点**：只使用了 1/n 的可用样本，数据利用率低。

### 6.3 Overlapped Variable τ Estimator（重叠变τ估计器）⭐

```
σ²y(nτ₀) = 1/[2n²(M-n)] Σᵢ₌₁ᴹ⁻ⁿ [Σⱼ₌₀ⁿ⁻¹ (ȳiⱼ₊ₙ₊₁ - ȳiⱼ₊ₙ)]²
```

或者更简洁的形式：

```
σ²y(τ) = 1/[2n²(N-2n)] Σᵢ₌₁ᴺ⁻²ⁿ⁺¹ (xᵢ₊₂ₙ - 2xᵢ₊ₙ + xᵢ)² / τ²
```

**优势**：
- 样本利用率提高 √n 倍
- 更好的统计置信度
- IEEE、ITU-T、ETSI 标准推荐使用

### 6.4 Modified Allan Variance (MVAR / MAVAR)

**目的**：区分 White PM 和 Flicker PM

```
Mod σ²y(τ) = 1/[2n⁴M] Σᵢ₌₁ᴹ⁻³ⁿ⁺¹ [Σⱼ₌₀²ⁿ⁻¹ (-1)ʲ C(2n,j) xᵢⱼ₊ₙ]² / τ²
```

其中：
- C(2n,j) = binomial coefficient
- n = bandwidth reduction factor

**MDEV 响应**：
- White PM: σy(τ) ∝ τ^(-3/2)
- Flicker PM: σy(τ) ∝ τ^(-1)

### 6.5 Time Deviation (TDEV)

```
TDEV(τ) = τ √[Mod σ²y(τ)/3]
```

或者：
```
σₓ(τ) = TDEV(τ)/√3 = τ √[Mod σ²y(τ)/9]
```

**物理意义**：时间误差的标准差

---

## 七、Bias Functions（偏置函数）

### 7.1 B₁ Bias Function（B₁偏置函数）

关联 M-sample variance 与 2-sample variance：

```
B₁(N,r,μ) = [N²/(N²-1)] × [Σₖ₌₁ᴺ⁻¹ (N-k)/N × [(k+r-1)^μ⁺² - 2k^μ⁺² + (k-r-1)^μ⁺²] / r²]
```

变量：
- N = number of samples
- r = T/τ (ratio of measurement time to observation time)
- μ = noise exponent

### 7.2 B₂ Bias Function（B₂偏置函数）

处理 dead time 的影响：

```
B₂(r,μ) = [(r-1)² / (r²μ²)] × [r^(μ+2) - 1] / [(μ+1)(μ+2)]
```

### 7.3 B₃ Bias Function（B₃偏置函数）

处理 concatenated samples（串联样本）的偏置：

```
B₃(M,N,r,μ) = Σᵢ₌₁ᴺ⁻¹ Σⱼ₌₁ᴺ⁻¹ w(i,j,M,r,μ)
```

---

## 八、置信区间与自由度

### 8.1 Chi-square Distribution（卡方分布）

```
df × s²/σ² ∼ χ²(df)
```

其中：
- df = degrees of freedom（自由度）
- s² = sample variance（样本方差）
- σ² = true variance（真实方差）

**90% 置信区间**：

```
[df × s² / χ²₀.₉₅(df)] ≤ σ² ≤ [df × s² / χ²₀.₀₅(df)]
```

### 8.2 Effective Degrees of Freedom（有效自由度）

| **Noise Type** | **Degrees of Freedom** |
|---------------|------------------------|
| WPM | df ≈ (N+1)(N-2)/[3(N-τ/T)] |
| FPM | df ≈ exp[ln(N/τ) - 0.577] |
| WFM | df ≈ [(N+1)(N-2)/[3(N-τ/T)]] × [τ/T/(N-τ/T)] |
| FFM | df ≈ [N(N-1)(N-2)/2] × [(τ/T)²/(N-τ/T)²] |
| RWFM | df ≈ [3N(N+1)(N-2)]/[4(N-τ/T)³] |

---

## 九、Measurement Issues（测量问题）

### 9.1 Dead Time（死时间）

**定义**：两次测量之间仪器无法观测信号的时间

**影响**：
- 引入系统偏置
- 降低数据利用率

**解决方案**：
- Zero-dead-time counters（零死时间计数器）
- Bias function correction（偏置函数校正）

### 9.2 Bandwidth Limits（带宽限制）

**Nyquist rate**：
```
fₙᵧ = 1/(2T)
```

其中 T = sample period（采样周期）

**High corner frequency（高截止频率）**：
```
fH = 1/(2τ₀)
```

### 9.3 Linear Drift（线性漂移）

**频率响应**：
```
ν(t) = ν₀ + a·t
```

其中 a = frequency drift rate（频率漂移率）

**Allan variance 响应**：
```
σ²y(τ) = (a²τ²)/3
```

---

## 十、实验数据与典型曲线

### 10.1 典型 Allan Deviation 曲线特征

```
Log σy(τ)
  │
  │     WPM/FPM区域
  │    ┌──────────┐
  │   /            \  WFM区域
  │  /              \  FFM区域
  │ /                \_______
  │/                         \
  └─────────────────────────────→ Log τ
   短τ        中τ           长τ
```

### 10.2 实际测量数据示例

| **τ (s)** | **σy(τ)** | **Noise Type** |
|-----------|-----------|----------------|
| 10⁻³ | 1×10⁻⁹ | White PM |
| 10⁻² | 3×10⁻¹⁰ | Flicker PM |
| 10⁻¹ | 1×10⁻¹⁰ | White FM |
| 1 | 1×10⁻¹⁰ | Flicker FM |
| 10 | 3×10⁻¹⁰ | Random Walk FM |
| 100 | 1×10⁻⁹ | Frequency Drift |

### 10.3 典型振荡器性能

**Crystal Oscillator (OCXO)**：
- τ = 1s: σy ≈ 1×10⁻¹¹
- τ = 10s: σy ≈ 3×10⁻¹²

**Atomic Clock (Cs)**：
- τ = 1s: σy ≈ 5×10⁻¹²
- τ = 10⁴s: σy ≈ 1×10⁻¹³

**Hydrogen Maser**：
- τ = 1s: σy ≈ 2×10⁻¹³
- τ = 10³s: σy ≈ 5×10⁻¹⁵

---

## 十一、应用领域

### 11.1 Timekeeping（计时）
- Atomic clocks（原子钟）
- Crystal oscillators（晶体振荡器）
- Frequency-stabilized lasers（频率稳定激光器）

### 11.2 Navigation（导航）
- GPS/GLONASS/Galileo
- Inertial navigation systems

### 11.3 Telecommunications（电信）
- Network synchronization（网络同步）
- Precision Time Protocol (PTP)
- Synchronous Digital Hierarchy (SDH)

### 11.4 Sensors（传感器）
- Fiber optic gyroscopes（光纤陀螺仪）
- MEMS gyroscopes（MEMS陀螺仪）
- Accelerometers（加速度计）

---

## 十二、Filter Properties（滤波器特性）

### 12.1 Transfer Function（传递函数）

在频域中，Allan variance 可以表示为：

```
σ²y(τ) = ∫₀^∞ |H(f)|² Sy(f) df
```

其中传递函数：

```
|H(f)|² = 2 sin²(πfτ) / (πfτ)² = [2 sin(πfτ) / (πfτ)]²
```

### 12.2 物理解释

该传递函数类似于一个 band-pass filter（带通滤波器）：
- 低频截止：f ≈ 1/τ
- 高频截止：f ≈ 1/τ₀

---

## 十三、重要参考资源

### 13.1 关键论文

1. **David W. Allan**, "Statistics of Atomic Frequency Standards", *Proc. IEEE*, 1966
2. **D. B. Leeson**, "A Simple Model of Feedback Oscillator Noise Spectrum", *Proc. IEEE*, 1966
3. **J. A. Barnes**, "Atomic Timekeeping and the Statistics of Precision Signal Generators"

### 13.2 标准文档

- **IEEE Standard 1139**: Standard Definitions of Physical Quantities for Fundamental Frequency and Time Metrology
- **ITU-T Recommendation G.813**: Timing characteristics of SDH equipment slave clocks

### 13.3 软件工具

- **Stable32**: [William Riley](https://www.wriley.com/)
- **AllanTools**: Python library - [GitHub](https://github.com/aewallin/allantools)
- **Allanvar**: R package for sensor error characterization

### 13.4 在线资源

- [NIST Handbook of Frequency Stability Analysis](https://tf.nist.gov/general/pdf/2220.pdf)
- [David W. Allan's Overview](http://www.allanstime.com/AllanVariance/)
- [UFFC Frequency Control Teaching Resources](https://uffc.org/frequency-control/teaching-resources/)

---

## 十四、深入物理直觉构建

### 14.1 为什么需要 Allan Variance？

传统 **standard deviation（标准差）** 在处理 non-stationary noise（非平稳噪声）时会 **发散**，因为：

对于 flicker noise 和 random walk noise：

```
lim(N→∞) σ²(N) → ∞
```

而 Allan variance 通过 **differencing（差分）** 操作：
- 消除低频漂移
- 对特定噪声类型收敛
- 提供 time-scale dependent（时间尺度依赖）的稳定性度量

### 14.2 二阶差分的物理意义

```
x(t+2τ) - 2x(t+τ) + x(t)
```

这是 **second difference（二阶差分）**，相当于：
- 两次求导的离散近似
- 抑制线性趋势（一阶差分抑制常数项）
- 对噪声的" curvature（曲率）"敏感

### 14.3 Log-log 图的斜率含义

在 log(σy) vs log(τ) 图中：
- **斜率 -1**: WPM/FPM (短时间噪声主导)
- **斜率 -1/2**: WFM (随机游走)
- **斜率 0**: FFM (flicker corner)
- **斜率 +1/2**: RWFM (长期漂移)
- **斜率 +1**: Systematic drift (系统漂移)

---

## 十五、高级主题

### 15.1 Hadamard Variance

**定义**：使用三样本差分

```
Hσ²y(τ) = 1/[6(M-2)] Σᵢ₌₁ᴹ⁻² (ȳi₊₂ - 2ȳi₊₁ + ȳi)²
```

**优势**：
- 对线性漂移不敏感
- 更好的噪声分离能力

### 15.2 Total Variance

扩展测量数据，通过 wrapping（循环）增加有效样本数。

### 15.3 Theovariance

由 **Theodore Walter** 提出，优化使用所有可用数据。

---

## 总结

Allan variance 是一个强大的 **time-domain stability analysis（时域稳定性分析）** 工具，其核心价值在于：

1. **噪声类型识别**：区分五种不同的幂律噪声
2. **尺度依赖性**：揭示不同时间尺度上的稳定性特征
3. **收敛性**：对传统统计工具发散的噪声类型仍能收敛
4. **实用性**：广泛应用于精密计时、导航、电信等领域

通过分析 Allan deviation 的 **log-log 曲线斜率**，工程师可以快速诊断振荡器的噪声特性并优化系统设计。

---

**参考文献**：
- [Allan Variance - Wikipedia](https://en.wikipedia.org/wiki/Allan_variance)
- [NIST Special Publication 1065](https://tf.nist.gov/general/pdf/2220.pdf)
- [IEEE Standard 1139-2008](https://standards.ieee.org/)