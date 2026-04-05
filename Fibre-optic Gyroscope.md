**FOG (Fibre-optic Gyroscope)** 是一种基于 **Sagnac effect** 的惯性传感器，
**Sagnac interferometer (Sagnac干涉仪)**，其工作原理如下：

```
[Light Source] → [Beam Splitter] → ┌─────────────────┐
                                     │  Fibre Coil    │
                                     │  (多匝光纤)     │
                                     └─────────────────┘
                                              ↓
                                 [Interference Detector]
                                              ↓
                                 [Angular Velocity Output]
```

当系统旋转时，**clockwise beam (顺时针光束)** 和 **anticlockwise beam (逆时针光束)** 经历不同的 **optical path length (光程)**，产生 **differential phase shift (相位差)**。

## 2. Sagnac Effect 数学原理详解

### 2.1 基本相位差公式

**Sagnac phase shift (Sagnac相位差)** 的精确表达式为：

$$\Delta \phi = \frac{8\pi A \Omega}{\lambda c}$$
- $\Delta \phi$: differential phase shift (相位差)，单位：radians
- $A$: effective area (有效面积)，单位：m²
- $\Omega$: angular velocity (角速度)，单位：rad/s
- $\lambda$: optical wavelength (光波长)，单位：m
- $c$: speed of light in vacuum (真空光速)，≈ 3×10⁸ m/s

### 2.2 考虑折射率的修正公式

当考虑 **refractive index of fibre (光纤折射率)** $n$ 时：

$$\Delta \phi = \frac{8\pi A \Omega n}{\lambda c}$$

### 2.3 多匝光纤的有效面积

对于多匝光纤线圈：

$$A_{eff} = N \cdot A_{loop}$$
- $A_{eff}$: effective area (有效面积)
- $N$: number of turns (匝数)
- $A_{loop}$: geometric area of single loop (单匝几何面积)

因此，多匝线圈的相位差公式变为：

$$\Delta \phi = \frac{8\pi N A_{loop} \Omega}{\lambda c}$$

### 2.4 光纤长度与直径的关系

给定 **fibre length (光纤长度)** $L$ 和 **coil diameter (线圈直径)** $D$：

$$N = \frac{L}{\pi D}$$

代入后得到：

$$\Delta \phi = \frac{8\pi L D \Omega}{\lambda c \cdot \pi D} = \frac{8L D \Omega}{\lambda c}$$

## 3. 详细架构设计

### 3.1 IFOG (Interferometric FOG) 架构

```
┌─────────────────────────────────────────────────────────┐
│                    IFOG Architecture                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   [Laser Diode]                                         │
│        │                                                │
│        ↓                                                │
│   [Optical Isolator] (防止反射)                         │
│        │                                                │
│        ↓                                                │
│   [Coupler 1]                                          │
│        │                                                │
│        ├─→ [Phase Modulator] → [Fibre Coil] ←────┐     │
│        │              (CW direction)              │     │
│        │                                         │     │
│        │              (CCW direction)            │     │
│        └─────────────────────────────────────────┘     │
│        │                                                │
│        ↓                                                │
│   [Coupler 2]                                          │
│        │                                                │
│        ↓                                                │
│   [Photodetector]                                       │
│        │                                                │
│        ↓                                                │
│   [Signal Processing]                                  │
│        │                                                │
│        ↓                                                │
│   [Angular Velocity Output]                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 RFOG (Resonator FOG) 架构

**RFOG** 使用 **resonant cavity (谐振腔)**，其 **finesse (精细度)** $F$ 定义为：

$$F = \frac{\sqrt{R}}{1-R}$$

- $R$: reflectivity of cavity mirrors (腔镜反射率)

**Resonance frequency shift (谐振频率偏移)** 与角速度的关系：

$$\Delta f = \frac{4A\Omega}{\lambda P}$$

**其中：**
- $P$: perimeter of resonator (谐振腔周长)

## 4. 性能参数与公式

### 4.1 Scale Factor (标度因子)

**Scale factor** 表示相位差与角速度之间的比例关系：

$$SF = \frac{\Delta \phi}{\Omega} = \frac{8\pi A_{eff}}{\lambda c}$$

### 4.2 Bias Stability (偏置稳定性)

**Bias stability** 通常用 **angle random walk (ARW)** 表示：

$$ARW = \frac{\sigma_{bias}}{\sqrt{\tau}}$$

**其中：**
- $\sigma_{bias}$: bias noise standard deviation (偏置噪声标准差)
- $\tau$: integration time (积分时间)

**典型值：**
- High-performance FOG: 0.001°/√h
- Medium-performance FOG: 0.01°/√h
- Low-performance FOG: 0.1°/√h

### 4.3 Dynamic Range (动态范围)

$$DR = 20 \log_{10}\left(\frac{\Omega_{max}}{\Omega_{min}}\right)$$

**典型值：** 120-140 dB

## 5. 两种工作模式详解

### 5.1 Open-loop Configuration (开环配置)

**输出信号** 直接由干涉强度决定：

$$I_{out} = I_0 \left[1 + \cos(\Delta \phi + \phi_m)\right]$$

**其中：**
- $I_0$: input light intensity (输入光强)
- $\phi_m$: modulation phase (调制相位)

**优点：**
- Simpler design (设计简单)
- Lower power consumption (功耗低)

**缺点：**
- Limited dynamic range (动态范围有限)
- Nonlinear response (响应非线性)

### 5.2 Closed-loop Configuration (闭环配置)

使用 **feedback mechanism (反馈机制)** 将相位差维持在零附近：

$$\phi_{fb} = -\Delta \phi$$

**输出** 是反馈信号：

$$\Omega_{out} = \frac{\phi_{fb}}{SF}$$

**优点：**
- Wider dynamic range (动态范围宽)
- Better linearity (线性度好)
- Improved stability (稳定性提高)

## 6. 误差源与补偿技术

### 6.1 主要误差源

| Error Type (误差类型) | Physical Origin (物理起源) | Compensation Method (补偿方法) |
|---------------------|-------------------------|------------------------------|
| **Shupe effect** | Thermal gradients in fibre (光纤温度梯度) | Symmetric winding (对称绕制) |
| **Kerr effect** | Optical nonlinearities (光学非线性) | Low-coherence source (低相干源) |
| **Rayleigh backscattering** | Impurities in fibre (光纤杂质) | Phase modulation (相位调制) |
| **Polarization fading** | Polarization drift (偏振漂移) | Polarization-maintaining fibre (保偏光纤) |

### 6.2 Shupe Effect 详解

**Shupe effect** 是由 **non-reciprocal phase shift (非互易相位偏移)** 引起的：

$$\Delta \phi_{Shupe} = \frac{2\pi n}{\lambda c} \int_{0}^{L} \frac{dn}{dT}(z) \cdot \Delta T(z) \cdot \Omega(z) \, dz$$

**其中：**
- $n$: refractive index (折射率)
- $\frac{dn}{dT}$: thermo-optic coefficient (热光系数)
- $\Delta T(z)$: temperature change along fibre (沿光纤温度变化)
- $\Omega(z)$: rotation rate profile (旋转速率分布)

**Quadrupolar winding (四极绕法)** 补偿技术：

```
Layer 1:  ──────────────→
Layer 2:  ←──────────────
Layer 3:  ←──────────────
Layer 4:  ──────────────→
```

### 6.3 Kerr Effect 补偿

**Kerr-induced phase shift**:

$$\Delta \phi_{Kerr} = \frac{2\pi}{\lambda} n_2 L (P_{CW} - P_{CCW})$$

**其中：**
- $n_2$: nonlinear refractive index (非线性折射率)
- $P_{CW}, P_{CCW}$: clockwise and counter-clockwise powers (顺时针和逆时针功率)

**解决方案：** 使用 **broadband source (宽带光源)** 降低相干性。

## 7. 应用场景与技术对比

### 7.1 FOG vs 其他陀螺仪技术

| Parameter (参数) | FOG | Ring Laser Gyroscope (RLG) | MEMS Gyroscope |
|-----------------|-----|---------------------------|----------------|
| **Bias Stability (偏置稳定性)** | 0.001-0.1°/h | 0.001-0.01°/h | 1-100°/h |
| **Scale Factor Stability (标度因子稳定性)** | 1-10 ppm | 1-5 ppm | 100-1000 ppm |
| **Dynamic Range (动态范围)** | ±1000°/s | ±1000°/s | ±500°/s |
| **Startup Time (启动时间)** | <1s | 1-10s | <0.1s |
| **Lifetime (寿命)** | 10-20 years | 5-10 years | 5-10 years |
| **Cost (成本)** | High | High | Low |

## 8. 先进技术发展

### 8.1 Hollow-core Fibre FOG

**Hollow-core fibre (空芯光纤)** 可以减少环境敏感度：

$$\Delta \phi_{HC} = \frac{8\pi A \Omega}{\lambda c} \cdot \frac{n_{eff}}{n_{vacuum}}$$

**其中：**
- $n_{eff} \approx 1$: 有效折射率接近真空

### 8.2 Quantum FOG

利用 **quantum entanglement (量子纠缠)** 提高精度：

$$\Delta \phi_{quantum} = \frac{\Delta \phi_{classical}}{N}$$

**其中：**
- $N$: number of entangled photons (纠缠光子数)

1. **Sagnac Effect Explanation:**
   https://en.wikipedia.org/wiki/Sagnac_effect

2. **Ring Laser Gyroscope (for comparison):**
   https://en.wikipedia.org/wiki/Ring_laser_gyroscope

3. **NASA Space Applications of FOG:**
   https://ntrs.nasa.gov/api/citations/20150012300/downloads/20150012300.pdf

4. **FOG Tutorial by KVH Industries:**
   https://www.kvh.com/Products-and-Solutions/Fiber-Optic-Gyroscopes.aspx

5. **IEEE Xplore FOG Papers:**
   https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=fiber+optic+gyroscope

6. **SPIE Digital Library on FOG:**
   https://www.spiedigitallibrary.org/