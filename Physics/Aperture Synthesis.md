**合成孔径** 是一种 **interferometry** 技术，它将多个 **telescopes** 收集到的信号进行混合，产生与整个望远镜集合尺寸相当的角分辨率的图像。
在每个分离和方向上，**interferometer** 的波瓣模式产生一个输出，该输出是被观测物体亮度空间分布的 **Fourier transform** 的一个分量。图像（或"map"）从这些测量中产生。

**关键公式**：
```
I(θ, φ) = ∫∫ V(u, v) · e^(2πi(ux + vy)) dudv
```

其中：
- **I(θ, φ)** = 源的强度分布（图像域）
- **V(u, v)** = 复可见度函数（**UV plane** 测量值）
- **u, v** = 空间频率坐标，对应于投影的 **baseline** 向量
- **x, y** = 天空坐标（方向余弦）
- **θ, φ** = 源的角度坐标

**UV plane** 是合成孔径中的核心概念，它代表空间频率域。每个 **baseline** 在 **UV plane** 中对应一个采样点。
## 二、 Baseline 计算

**Baseline** 定义为从 **radio source** 看到的任意两个望远镜之间的投影分离。

**基线数量公式**：
```
nb = (n² - n) / 2 = C(n, 2)
```

**变量说明**：
- **nb** = 独立 **baseline** 的数量
- **n** = **array** 中望远镜的数量
- **C(n, 2)** = 组合数，从n个元素中选2个

**具体例子**：
- **Very Large Array (VLA)**: n = 27 → nb = 351 条独立基线
- 6个 **optical telescopes**: n = 6 → nb = 15 条基线

**更详细的基线向量表示**：
```
B⃗ij = B⃗j - B⃗i
```
其中：
- **B⃗ij** = 望远镜i和j之间的基线向量
- **B⃗i**, **B⃗j** = 各望远镜在参考系中的位置向量

**投影基线（UV坐标）**：
```
(u, v) = (B⃗ij · x̂ / λ, B⃗ij · ŷ / λ)
```
其中：
- **λ** = 观测波长
- **x̂**, **ŷ** = **UV plane** 的单位向量

参考链接：
- [ALMA - Baseline Coordinates](https://almascience.nao.ac.jp/documents-and-tools/cycle-10/alma-technical-handbook/viewing-chapter/5)

## 三、 地球旋转合成孔径

大多数 **radio frequency** 合成孔径 **interferometers** 利用地球的旋转来增加观测中包含的不同 **baselines** 的数量。

**几何原理**：

```
B⃗proj(t) = R(t) · B⃗physical
```

其中：
- **B⃗proj(t)** = 在时间t的投影基线向量
- **R(t)** = 随时间变化的旋转矩阵（由于地球自转）
- **B⃗physical** = 地面上固定的物理基线向量

**旋转矩阵**（简化版）：
```
R(t) = [cos(ωt)  -sin(ωt)  0
        sin(ωt)   cos(ωt)  0
         0         0       1]
```

其中 **ω** = 地球自转角速度 ≈ 7.292 × 10⁻⁵ rad/s

**UV 轨迹方程**：
```
u(t) = Bx/λ · [sin(Ha) · sin(Dec) + cos(Ha) · cos(Dec) · sin(Lat)]
v(t) = By/λ · cos(Dec) + (Bz/λ) · sin(Dec)
```

其中：
- **Ha** = 时角（随时间变化）
- **Dec** = 源的赤纬
- **Lat** = 观测站的纬度
- **Bx, By, Bz** = 基线在东-北-上坐标系中的分量

**历史背景**：地球旋转的使用在1950年的论文"A preliminary survey of the radio stars in the Northern Hemisphere"中详细讨论。

参考链接：
- [ATNF - Earth Rotation Synthesis](https://www.atnf.csiro.au/people/Matt.Whiting/SynthesisImaging/synthesis1.html)
- [Hartmann - Aperture Synthesis](http://www.aoc.nrao.edu/events/syna/1/hartmann_slides.pdf)

## 四、 测量要求与技术挑战

### 测量条件
**合成孔径** 只有在每台望远镜都能测量入射信号的 **amplitude** 和 **phase** 时才可能。

**可见度函数**：
```
V(u, v) = ∫∫ I(l, m) · e^(-2πi(ul + vm)) dldm
```

其中：
- **I(l, m)** = 源的亮度分布（方向余弦坐标）
- **l, m** = 天空平面的方向余弦

### Radio 与 Optical 的技术差异

**Radio Frequencies**：
- 可以通过电子设备直接测量 **phase**
- 相关处理在 **software** 中完成
- 自1950年代以来成功使用

**Optical Frequencies**：
- **electromagnetic field** 无法直接测量
- 必须通过敏感的光学传播并进行光学干涉
- 需要精确的 **optical delay** 和 **atmospheric wavefront aberration** 修正
- 这是一项非常苛刻的技术，直到1990年代才成为可能

**相位稳定性要求**：
```
Δφ << 2π · (ΔL/λ)
```

其中：
- **Δφ** = 相位误差
- **ΔL** = 光程差误差
- **λ** = 观测波长

对于 **optical interferometry** (λ ≈ 500 nm)，需要纳米级的路径精度。

**大气相位延迟修正**：
```
Δφ_atm = 2π/λ · (n - 1) · h · sec(z)
```

其中：
- **n** = 大气折射率
- **h** = 大气有效高度
- **z** = 天顶角

参考链接：
- [VLTI - Optical Interferometry](https://www.eso.org/sci/facilities/paranal/instruments/vlti/)
- [Keck Interferometer](https://keckobservatory.org/keck-interferometer/)

## 五、 信号处理与图像重建

### Fourier Transform 重建

理想情况下，完全采样的 **Fourier transform** 包含的信息等同于孔径直径等于最大 **baseline** 的传统望远镜的图像。

**直接逆变换**：
```
I_dirty(x, y) = Σ Σ V(ui, vj) · e^(2πi(uix + vjy)) · W(ui, vj)
```

其中：
- **I_dirty** = "脏图"（未经清洗的重建图像）
- **W(u, v)** = 采样权重函数

### 点扩散函数 (Point Spread Function, PSF)

```
PSF(x, y) = Σ Σ e^(2πi(uix + vjy)) · W(ui, vj)
```

**脏图** 与 **PSF** 的关系：
```
I_dirty = I_true ⨂ PSF + N
```

其中：
- **I_true** = 真实图像
- **⨂** = 卷积运算符
- **N** = 噪声项

### 非线性去卷积算法

由于完全采样通常不可行，开发了多种算法：

**CLEAN Algorithm**（Högbom 1974）：
```
1. I_iter = I_dirty
2. while max(|I_iter|) > threshold:
   a. (x_max, y_max) = argmax(|I_iter|)
   b. I_iter(x_max, y_max) -= γ · PSF
   c. I_clean(x_max, y_max) += γ · I_iter(x_max, y_max)
3. I_final = conv(I_clean, beam) + residual
```

其中：
- **γ** = 循环增益（通常0.1-0.3）
- **beam** = 拟合的 **dirty beam**

**Maximum Entropy Method (MEM)**：

目标函数：
```
Q = -Σ I_true · log(I_true / m) + α · χ²
```

约束条件：
```
χ² = Σ (V_observed - V_model)² / σ²
```

其中：
- **m** = 先验模型（通常为平坦或平滑）
- **α** = 正则化参数
- **σ** = 测量不确定度

参考链接：
- [NRAO - CLEAN Algorithm](https://www.cv.nrao.edu/~abridle/deconvol/node7.html)
- [CASAguide - Imaging](https://casa.nrao.edu/casadocs/casa-5.1.0/imaging/synthesis-imaging)

## 六、 现代 Radio Camera 概念

为了避免计算瓶颈，像 **Deep Synoptic Array** 这样的望远镜使用大量随机分布的天线几乎完全采样 **UV plane**。

**密集采样优势**：
- **PSF** 接近理想 δ 函数
- 允许简单的重建算法
- 图像可以实时计算

**采样密度**：
```
N_ant = 2000 → nb = (2000² - 2000)/2 = 1,999,000 基线
```

**点源响应**：
```
PSF_ideal(x, y) ≈ δ(x, y) ⊗ B(x, y)
```

其中 **B(x, y)** 是单个天线的波束图案。

参考链接：
- [Deep Synoptic Array](https://deepsynopticarray.org/)
- [HERA - Hydrogen Epoch of Reionization Array](https://reionization.org/)

## 七、 历史发展

### 关键时间线

| 年代 | 事件 | 贡献者 |
|------|------|--------|
| 1946 | **Aperture synthesis** 概念首次提出 | Ruby Payne-Scott, Joseph Pawsey |
| 1950 | 地球旋转方法讨论 | Cambridge University |
| 1950s-60s | **One-Mile Telescope** 开发 | Martin Ryle 团队 |
| 1960s-70s | 计算机支持 **Fourier transform** 反演 | Titan 等计算机 |
| 1970s | **Ryle Telescope** (5 km 有效孔径) | Mullard Radio Astronomy Observatory |
| 1990s | **Optical interferometry** 技术突破 | VLTI, Keck 等项目 |
| 2019 | **Event Horizon Telescope** 首张黑洞图像 | EHT Collaboration |

### Nobel Prize 相关

**Martin Ryle** 和 **Tony Hewish** 因其在开发 **radio interferometry** 方面的贡献共同获得诺贝尔奖。

**Very Long Baseline Interferometry (VLBI)** 发展：
```
B_max 可达数千公里
θ_min ≈ λ / B_max
```

对于 **EHT**：
- **λ** ≈ 1.3 mm
- **B_max** ≈ 10,000 km
- **θ_min** ≈ 20 μas（微角秒）

参考链接：
- [Nobel Prize 1974](https://www.nobelprize.org/prizes/physics/1974/summary/)
- [EHT Project](https://eventhorizontelescope.org/)

## 八、 具体系统示例

### Very Large Array (VLA)

**配置**：
- 27 个 25 米天线
- 四种配置：A (最大), B, C, D (最小)
- 最大 **baseline**: 36 km (A配置)
- 频率范围: 1-50 GHz

**灵敏度公式**：
```
ΔS = SEFD / (√(N_b · Δν · τ))
```

其中：
- **SEFD** = 系统等效通量密度
- **N_b** = 基线数量
- **Δν** = 带宽
- **τ** = 积分时间

### Atacama Large Millimeter Array (ALMA)

**配置**：
- 66 个天线（12 米和 7 米）
- 最大 **baseline**: 16 km
- 频率范围: 31-950 GHz
- 高海拔: 5000 米

**UV 覆盖优化**：
```
D_uv = Σ W(u, v) · δ(u - u_i, v - v_j)
```

权重函数：
```
W(u, v) = 1/σ²(u, v)
```

参考链接：
- [VLA Documentation](https://science.nrao.edu/facilities/vla/docs/manuals)
- [ALMA Science Portal](https://almascience.nao.ac.jp/)

## 九、 与 Synthetic Aperture Radar (SAR) 的区别

虽然两者都使用 "synthetic aperture" 一词，但它们在技术和历史上是独立的：

**Aperture Synthesis (天文)**：
- 基于 **interferometry**
- 多个同时工作的接收器
- 测量相干 **phase**

**SAR (雷达)**：
- 基于 **Doppler technique**
- 单个移动平台
- 利用平台运动合成孔径

**SAR 方位向分辨率**：
```
ρ_a = L / 2
```

其中：
- **ρ_a** = 方位向分辨率
- **L** = 合成孔径长度

**SAR 距离向分辨率**：
```
ρ_r = c / (2 · B)
```

其中：
- **ρ_r** = 距离向分辨率
- **c** = 光速
- **B** = 信号带宽

参考链接：
- [SAR Principles](https://www.eoportal.org/web/eoportal/satellite-missions/content/-/article/sar-principles)
- [NASA SAR Data](https://search.asf.alaska.edu/)

## 十、 高级主题：UV 覆盖与图像质量

### UV 覆盖准则

**Nyquist 采样**：
```
Δu ≤ 1/(2 · θ_FOV)
```

其中：
- **Δu** = UV 采样间隔
- **θ_FOV** = 视场

**UV 覆盖评估**：
```
C_uv = Σ Σ (2π · r · dr · dθ)
```

其中：
- **r** = UV 平面径向距离
- **θ** = UV 平面方位角

### 图像质量指标

**Beam Fitting**：
```
θ_b = k · λ / B_eff
```

其中：
- **θ_b** = 束宽（分辨率）
- **k** = 形状因子（通常 1.1-1.2）
- **B_eff** = 有效基线

**动态范围**：
```
DR = I_peak / σ_rms
```

其中：
- **DR** = 动态范围
- **I_peak** = 峰值强度
- **σ_rms** = 残差噪声

参考链接：
- [NRAO - UV Coverage](https://www.cv.nrao.edu/~abridle/phc/hc3.htm)
- [CASA - Imaging Guide](https://casaguides.nrao.edu/)

---

**总结**：合成孔径通过多望远镜干涉和 **Fourier transform** 原理，实现了相当于单一大孔径望远镜的角分辨率。从 1940 年代的概念提出到现代的 **Event Horizon Telescope**，这项技术在天文学领域产生了深远的影响。关键技术包括精确的 **phase** 测量、**UV plane** 采样优化、以及复杂的图像重建算法。