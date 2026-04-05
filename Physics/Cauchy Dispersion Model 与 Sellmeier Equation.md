# Cauchy Dispersion Model 与 Sellmeier Equation 详解

## 一、第一性原理：什么是 Optical Dispersion？

当 light 在 medium 中传播时，refractive index $n$ 不是常数，而随 wavelength $\lambda$ 变化。这种**现象**称为 **dispersion**。

从**微观第一性原理**理解：
- Light 的 oscillating electric field $\vec{E}(t) = \vec{E}_0 e^{-i\omega t}$ 驱动 medium 中的 bound electrons 做 forced oscillation
- 电子的响应可用 **Lorentz oscillator model** 描述：
$$m_e \frac{d^2x}{dt^2} + m_e\gamma\frac{dx}{dt} + m_e\omega_0^2 x = -eE(t)$$

其中：
- $m_e$ = electron mass
- $x$ = electron displacement from equilibrium
- $\gamma$ = damping coefficient (阻尼系数)
- $\omega_0$ = natural resonance frequency (固有共振频率)
- $e$ = electron charge
- $\omega = \frac{2\pi c}{\lambda}$ = light angular frequency

这个方程的解给出 **electric susceptibility** $\chi(\omega)$，进而得到 **dielectric function** $\varepsilon(\omega) = 1 + \chi(\omega)$，最终 refractive index $n = \sqrt{\varepsilon}$。

---

## 二、Cauchy Dispersion Model

### 2.1 历史背景

**Augustin-Louis Cauchy** 在 1836 年提出这个经验公式，早于对原子结构的理解。

### 2.2 数学形式

**Cauchy's equation**：

$$n(\lambda) = A + \frac{B}{\lambda^2} + \frac{C}{\lambda^4} + \frac{D}{\lambda^6} + \cdots$$

更常见的形式：

$$n(\lambda) = A + \frac{B}{\lambda^2} + \frac{C}{\lambda^4}$$

**变量说明**：
- $n(\lambda)$ = refractive index at wavelength $\lambda$
- $\lambda$ = wavelength of light (通常以 μm 或 nm 为单位)
- $A, B, C, D, \ldots$ = **Cauchy coefficients**，是 material-specific empirical constants
- $A$ 通常接近 material 的 static refractive index
- $B$ 与 electronic polarizability 相关
- $C, D, \ldots$ 是高阶修正项

### 2.3 为什么是 $\lambda^{-2}$ 的幂级数？

从 **Lorentz oscillator model** 出发，在 **far from resonance** (远离共振) 的区域：

$$\varepsilon(\omega) = 1 + \sum_j \frac{N_j e^2}{\varepsilon_0 m_e} \frac{1}{\omega_j^2 - \omega^2 - i\gamma_j\omega}$$

当 $\omega \ll \omega_j$ (即波长远离 absorption edge) 时，做 Taylor expansion：

$$\frac{1}{\omega_j^2 - \omega^2} = \frac{1}{\omega_j^2} \cdot \frac{1}{1 - (\omega/\omega_j)^2} \approx \frac{1}{\omega_j^2}\left(1 + \frac{\omega^2}{\omega_j^2} + \frac{\omega^4}{\omega_j^4} + \cdots\right)$$

由于 $\omega \propto \lambda^{-1}$，所以：

$$n^2(\lambda) \approx A_0 + A_1\lambda^2 + A_2\lambda^4 + \cdots$$

等等，这给出的是 $\lambda^2$ 的展开，与 Cauchy form 不同。实际上 Cauchy form 更适用于 **UV-absorbing materials** 在 visible region，此时 $\omega$ 接近 resonance 但在透明窗口。

### 2.4 适用范围

| 条件 | 说明 |
|------|------|
| **透明区域** | Material 在所考虑 wavelength 范围内没有 absorption |
| **远离 resonance** | Wavelength 远离 material 的 absorption bands |
| **可见光区域** | 对于大多数 optical glasses，visible spectrum (400-700 nm) |
| **有限 wavelength range** | 通常不超过 ~100 nm 的 range |

### 2.5 Cauchy Coefficients 示例数据

| Material | A | B (μm²) | C (μm⁴) |
|----------|---|---------|---------|
| **BK7 glass** | 1.5046 | 0.00420 | 0.0 |
| **Fused silica** | 1.4580 | 0.00354 | 0.0 |
| **SF11 glass** | 1.7370 | 0.01340 | 0.0 |
| **Diamond** | 2.3728 | 0.01000 | 0.0 |

### 2.6 Cauchy Model 的局限性

1. **无法处理 absorption region**：当 $\lambda$ 接近 absorption edge 时，$n(\lambda)$ 会 diverge
2. **无法描述 anomalous dispersion**：在 absorption band 附近，$dn/d\lambda < 0$ (normal dispersion 变成 anomalous)
3. **适用 wavelength 范围有限**：需要针对不同 range 拟合不同的 coefficients
4. **缺乏物理意义**：Coefficients A, B, C 纯粹是 empirical fitting parameters

---

## 三、Sellmeier Equation

### 3.1 历史背景

**Wolfgang Sellmeier** 在 1872 年基于 **resonance absorption theory** 提出这个方程。它有更明确的物理基础。

### 3.2 数学形式

**Sellmeier equation** 的标准形式：

$$n^2(\lambda) = 1 + \sum_{j=1}^{m} \frac{B_j \lambda^2}{\lambda^2 - C_j}$$

或者写成：

$$n^2(\lambda) = 1 + \sum_{j=1}^{m} \frac{B_j \lambda^2}{\lambda^2 - \lambda_j^2}$$

**变量说明**：
- $n(\lambda)$ = refractive index
- $\lambda$ = wavelength (通常以 μm 为单位)
- $B_j$ = **Sellmeier B-coefficients** (dimensionless)，与 oscillator strength 相关
- $C_j$ 或 $\lambda_j^2$ = **Sellmeier C-coefficients** (单位 μm²)，与 resonance wavelength squared 相关
- $m$ = number of oscillators considered (通常取 2-3)

**物理意义**：
- 每个 term $\frac{B_j \lambda^2}{\lambda^2 - \lambda_j^2}$ 对应一个 electronic or ionic resonance
- $B_j \propto N_j f_j$，其中 $N_j$ 是 oscillator density，$f_j$ 是 oscillator strength
- $\lambda_j$ 是 resonance wavelength (对应于 absorption peak)

### 3.3 与 Lorentz Oscillator Model 的关系

从 Lorentz model，假设无 damping ($\gamma = 0$)：

$$\varepsilon(\omega) = n^2 = 1 + \sum_j \frac{\omega_p,j^2}{\omega_j^2 - \omega^2}$$

其中 $\omega_p,j^2 = \frac{N_j e^2}{\varepsilon_0 m_e f_j}$ 是 plasma frequency squared。

将 $\omega = \frac{2\pi c}{\lambda}$ 和 $\omega_j = \frac{2\pi c}{\lambda_j}$ 代入：

$$n^2 = 1 + \sum_j \frac{\omega_p,j^2}{\omega_j^2 - \omega^2} = 1 + \sum_j \frac{\omega_p,j^2}{\omega_j^2} \cdot \frac{1}{1 - (\omega/\omega_j)^2}$$

$$= 1 + \sum_j \frac{\omega_p,j^2}{\omega_j^2} \cdot \frac{1}{1 - (\lambda_j/\lambda)^2} = 1 + \sum_j \frac{\omega_p,j^2 \lambda^2}{\omega_j^2 (\lambda^2 - \lambda_j^2)}$$

$$= 1 + \sum_j \frac{\omega_p,j^2 \lambda^2 \cdot \lambda_j^2}{\omega_j^2 \lambda_j^2 (\lambda^2 - \lambda_j^2)} = 1 + \sum_j \frac{\omega_p,j^2}{\omega_j^2} \cdot \frac{\lambda^2}{\lambda^2 - \lambda_j^2}$$

定义 $B_j = \frac{\omega_p,j^2}{\omega_j^2}$，就得到 Sellmeier form。

### 3.4 Sellmeier Coefficients 示例

**Fused Silica (SiO₂)** - 3-term Sellmeier:
$$n^2 = 1 + \frac{0.6961663\lambda^2}{\lambda^2 - 0.0684043^2} + \frac{0.4079426\lambda^2}{\lambda^2 - 0.1162414^2} + \frac{0.8974794\lambda^2}{\lambda^2 - 9.896161^2}$$

| j | B_j | C_j = λ_j² (μm²) | λ_j (μm) | Physical origin |
|---|-----|------------------|----------|-----------------|
| 1 | 0.6961663 | 0.0046791 | 0.0684 | UV electronic transition |
| 2 | 0.4079426 | 0.013512 | 0.1162 | UV electronic transition |
| 3 | 0.8974794 | 97.934 | 9.896 | IR vibrational mode |

**BK7 Optical Glass**:
$$n^2 = 1 + \frac{1.03961212\lambda^2}{\lambda^2 - 0.00600069867} + \frac{0.231792344\lambda^2}{\lambda^2 - 0.0200179144} + \frac{1.01046945\lambda^2}{\lambda^2 - 103.560653}$$

### 3.5 Sellmeier Model 的优势

1. **Wider wavelength range**：可覆盖从 UV 到 IR 的 broad spectrum
2. **Physical meaning**：Coefficients 与 actual resonance wavelengths 对应
3. **Smooth extrapolation**：在 measured data 范围外行为合理
4. **Handles multiple resonances**：通过增加 terms 可以处理多个 absorption bands

### 3.6 Sellmeier 的局限性

1. **在 resonance wavelength 处发散**：当 $\lambda = \lambda_j$ 时，分母为零
2. **忽略 damping**：真实材料有 absorption，需要 complex refractive index
3. **需要 accurate coefficients**：实验测量需要覆盖足够宽的 wavelength range

---

## 四、Extended Sellmeier Models

### 4.1 Sellmeier with Absorption

引入 **extinction coefficient** $k(\lambda)$：

$$\tilde{n}(\lambda) = n(\lambda) + ik(\lambda)$$

Complex Sellmeier form：

$$\tilde{n}^2(\lambda) = \varepsilon_1 + i\varepsilon_2 = 1 + \sum_j \frac{B_j \lambda^2}{\lambda^2 - \lambda_j^2 + i\Gamma_j \lambda}$$

其中 $\Gamma_j$ 是 damping parameter。

### 4.2 Temperature-Dependent Sellmeier

**Thermo-optic coefficient** $\frac{dn}{dT}$ 导致：

$$n^2(\lambda, T) = 1 + \sum_j \frac{B_j(T) \lambda^2}{\lambda^2 - C_j(T)}$$

通常用 linear approximation：

$$B_j(T) = B_j(T_0) + \frac{dB_j}{dT}(T - T_0)$$

### 4.3 Sellmeier for Anisotropic Materials

对于 **uniaxial crystals**，需要两个 Sellmeier equations：

$$n_o^2(\lambda) = 1 + \sum_j \frac{B_{o,j} \lambda^2}{\lambda^2 - C_{o,j}} \quad \text{(ordinary ray)}$$

$$n_e^2(\lambda) = 1 + \sum_j \frac{B_{e,j} \lambda^2}{\lambda^2 - C_{e,j}} \quad \text{(extraordinary ray)}$$

例如 **Calcite (CaCO₃)**：
- $n_o \approx 1.658$ at 589 nm
- $n_e \approx 1.486$ at 589 nm

---

## 五、Cauchy vs Sellmeier 对比

| Feature | Cauchy Model | Sellmeier Model |
|---------|--------------|-----------------|
| **Mathematical form** | $n = A + B/\lambda^2 + C/\lambda^4$ | $n^2 = 1 + \sum B_j\lambda^2/(\lambda^2-C_j)$ |
| **Physical basis** | Empirical | Semi-empirical (resonance model) |
| **Accuracy** | Good for limited range | Better for wider range |
| **Wavelength range** | ~100 nm | Can span UV-visible-IR |
| **Parameters** | Usually 2-3 | Usually 3-6 (per oscillator) |
| **Physical interpretation** | Coefficients lack meaning | $B_j$, $C_j$ relate to resonances |
| **Near absorption edge** | Fails (diverges) | Better (but still needs damping term) |
| **Computational simplicity** | Very simple | Moderate |
| **Use case** | Quick approximation, thin films | Optical design, precision applications |

---

## 六、在 Optical Lens Design 中的应用

### 6.1 Chromatic Aberration Correction

Lens design 中需要 minimize **chromatic aberration**：

$$\delta f = f \cdot \frac{dn}{d\lambda} \cdot \Delta\lambda$$

使用 accurate dispersion model 可以：
1. **Ray tracing** at multiple wavelengths
2. **Achromatic doublet design**：选择两种 glass with complementary dispersion
3. **Apochromatic design**：三片或更多 glasses

### 6.2 Abbe Number

**Abbe number** $V_d$ 定义为：

$$V_d = \frac{n_d - 1}{n_F - n_C}$$

其中：
- $n_d$ = refractive index at $\lambda = 587.6$ nm (He d-line)
- $n_F$ = refractive index at $\lambda = 486.1$ nm (H F-line)
- $n_C$ = refractive index at $\lambda = 656.3$ nm (H C-line)

High $V_d$ = low dispersion (good for crown glass)
Low $V_d$ = high dispersion (flint glass)

### 6.3 Partial Dispersion

**Relative partial dispersion**：

$$P_{g,F} = \frac{n_g - n_F}{n_F - n_C}$$

其中 $n_g$ is at $\lambda = 435.8$ nm (Hg g-line)。

This is crucial for **apochromatic lens** design.

---

## 七、其他 Dispersion Models

### 7.1 Schott Glass Dispersion Formula

**Schott** uses a polynomial form：

$$n^2 = A_0 + A_1\lambda^2 + A_2\lambda^{-2} + A_3\lambda^{-4} + A_4\lambda^{-6} + A_5\lambda^{-8}$$

6 coefficients，适用于 365-1014 nm range。

### 7.2 Hartmann Formula

$$n(\lambda) = A + \frac{B}{(\lambda - C)^{1.2}}$$

Historical formula，现已较少使用。

### 7.3 Herzberger Formula

$$n = A + BL + CL^2 + D\lambda^2 + E\lambda^4 + \cdots$$

其中 $L = \frac{1}{\lambda^2 - 0.028}$ (0.028 is a fixed constant)

### 7.4 Cornu Equation

$$n = n_0 + \frac{A}{\lambda - \lambda_0}$$

Used near a single resonance.

---

## 八、实验测量方法

### 8.1 Minimum Deviation Method (Prism)

使用 **spectrometer** 测量 prism 的 minimum deviation angle $\delta_m$：

$$n = \frac{\sin\left(\frac{A + \delta_m}{2}\right)}{\sin(A/2)}$$

其中 $A$ 是 prism apex angle。

### 8.2 Ellipsometry

测量 **reflectance** 和 **phase change**，直接得到 complex refractive index $\tilde{n} = n + ik$。

### 8.3 Interferometry

通过 **Fabry-Pérot interferometer** 或 **Mach-Zehnder interferometer** 测量 optical path length difference。

---

## 九、Web References

1. **Refractive Index Database** - https://refractiveindex.info/
   - Comprehensive database of Sellmeier coefficients for thousands of materials

2. **Schott Optical Glass Data** - https://www.schott.com/en-us/products/optical-glass
   - Technical datasheets with dispersion formulas

3. **SPIE Optipedia - Dispersion** - https://spie.org/publications/fg05_p03-05_dispersion
   - Tutorial on dispersion theory

4. **RP Photonics Encyclopedia** - https://www.rp-photonics.com/sellmeier_formula.html
   - Detailed explanation of Sellmeier equation

5. **NIST Refractive Index** - https://physics.nist.gov/PhysRefData/Handbook/Tables/silicatable2.htm
   - Fused silica refractive index data

6. **Thorlabs Optical Material Properties** - https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=6973

---

## 十、总结：构建你的 Intuition

### 核心要点：

1. **Dispersion 来自 light 与 matter 的 resonant interaction**：当 light frequency 接近 material 的 electronic 或 vibrational resonance 时，折射率急剧变化。

2. **Cauchy 是 Taylor expansion 近似**：在远离 resonance 的透明区域，可以用简单的 $\lambda^{-2}$ 幂级数近似。

3. **Sellmeier 是 physical model 的简化版**：保留了 resonance structure，能描述更宽的 wavelength range。

4. **两个 model 都不是万能的**：
   - 有 absorption 时需要 complex refractive index
   - 有 temperature dependence 时需要 thermal corrections
   - 有 anisotropy 时需要多个 equations

5. **Lens design 的关键**：准确的 dispersion model 允许精确 ray tracing，从而 optimize lens performance across wavelengths。