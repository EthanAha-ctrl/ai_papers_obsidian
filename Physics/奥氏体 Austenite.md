# Austenite 的全面解析

## 1. 基本定义与晶体结构

**Austenite**（奥氏体）是 steel 和其他 alloys 中的一种 solid solution phase，本质上是由 **γ-Fe（Gamma Iron）** 作为 solvent，碳或其他元素作为 solute 形成的 **interstitial solid solution**。

### Crystal Structure 的第一性原理分析

```
┌─────────────────────────────────────────────────────────┐
│           Austenite 的 FCC 晶体结构示意                   │
│                                                         │
│              ● ──────── ● ──────── ●                    │
│             /│         /│         /│                    │
│            ● │─────── ● │─────── ● │                    │
│           /│ │       /│ │       /│ │                    │
│          ● │ │─────● │ │─────● │ │                    │
│          ││ │      ││ │      ││ │                      │
│          ││ ●──────││ ●──────││ ●                      │
│          ││/       ││/       ││/                       │
│          ●─────────●─────────●                         │
│          │/        │/        │/                        │
│          ●─────────●─────────●                         │
│                                                         │
│   ● = Fe atoms (corner + face centers)                  │
│   ○ = Carbon atoms (octahedral interstitial sites)     │
│                                                         │
│   Lattice parameter: a ≈ 3.57 Å (at 915°C)             │
└─────────────────────────────────────────────────────────┘
```

**关键参数**：

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Lattice constant | a | 3.57-3.68 | Å |
| Atomic packing factor | APF | 0.74 | - |
| Coordination number | CN | 12 | - |
| Atoms per unit cell | n | 4 | - |
| Octahedral interstitial sites | - | 4 | per cell |
| Tetrahedral interstitial sites | - | 8 | per cell |

### 第一性原理推导 FCC 结构稳定性

**Gibbs Free Energy** 决定 phase stability：

$$G = H - TS$$

其中：
- $G$ = Gibbs free energy (J/mol)
- $H$ = Enthalpy (J/mol)
- $T$ = Absolute temperature (K)
- $S$ = Entropy (J/mol·K)

对于 Austenite 的形成，需要：

$$\Delta G^{\alpha \rightarrow \gamma} = G^{\gamma} - G^{\alpha} < 0$$

其中：
- $\alpha$ = BCC ferrite
- $\gamma$ = FCC austenite

---

## 2. Phase Diagram 与形成条件

### Fe-C Phase Diagram 关键区域

```
Temperature (°C)
    │
1600├───────────────────────────────────── Liquid
    │                    ╲
1500├                     ╲
    │                      ╲
1400├                       ╲─────── δ-Ferrite + Liquid
    │                                ╲
1300├                                 ╲
    │                                  ╲
1200├───────────────────────────────────╲─── Austenite + Liquid
    │                                     ╲
1100├                                      ╲
    │                                       ╲
1000├────────────────────────────────────────╲─── Austenite
    │         γ (Austenite)                    │
 900├─────────────────────────────────────────┤ A₃ line
    │                     ╲                    │
 800├                      ╲         A₁ line ──┼─── Eutectoid (727°C)
    │                       ╲                   │
 727├────────────────────────╲──────────────────┤
    │   α+Ferrite   │ α+Fe₃C  │    α+Fe₃C      │
    │    region     │(Pearlite)│   (Tempered)  │
 600├───────────────┼─────────┼────────────────┤
    │               │         │                │
    └───────────────┼─────────┼────────────────┼───► %C
                  0.022      0.77            2.14    6.67
                  (Eutectoid) (Eutectoid)  (Eutectic)(Fe₃C)
```

**关键点详解**：

| Critical Point | Temperature | Carbon Content | Significance |
|----------------|-------------|----------------|--------------|
| A₁ (Eutectoid) | 727°C | 0.77 wt% C | α → γ transformation start |
| A₃ | 912°C (pure Fe) | 0-0.77 wt% C | Upper boundary of γ field |
| A₄ | 1394°C (pure Fe) | - | γ → δ transformation |
| E (Eutectoid) | 727°C | 0.77 wt% C | Maximum C in pearlite |
| S (Maximum solubility) | 1147°C | 2.14 wt% C | Max C in austenite |

---

## 3. Carbon Solubility 的数学模型

### Carbon 在 Austenite 中的溶解度

**Solubility equation** (经验公式)：

$$C_{\gamma}^{max} = 2.14 - 0.009(T - 1147)$$

其中：
- $C_{\gamma}^{max}$ = Maximum carbon solubility in austenite (wt%)
- $T$ = Temperature (°C)

或者更精确的 **Arrhenius-type equation**：

$$C_{eq} = C_0 \exp\left(-\frac{Q}{RT}\right)$$

其中：
- $C_{eq}$ = Equilibrium carbon concentration
- $C_0$ = Pre-exponential constant
- $Q$ = Activation energy for dissolution (J/mol)
- $R$ = Gas constant = 8.314 J/(mol·K)
- $T$ = Absolute temperature (K)

### Carbon 的 Diffusion 在 Austenite 中

**Diffusion coefficient**：

$$D_C^{\gamma} = D_0 \exp\left(-\frac{Q_D}{RT}\right)$$

其中：
- $D_C^{\gamma}$ = Diffusion coefficient of carbon in austenite (m²/s)
- $D_0$ = Frequency factor ≈ 0.23 cm²/s = 2.3×10⁻⁵ m²/s
- $Q_D$ = Activation energy for diffusion ≈ 148,000 J/mol

**数值对比**：

| Phase | D₀ (m²/s) | Q (kJ/mol) | D at 1000°C (m²/s) |
|-------|-----------|------------|---------------------|
| C in γ-Fe (Austenite) | 2.3×10⁻⁵ | 148 | ~1.5×10⁻¹¹ |
| C in α-Fe (Ferrite) | 1.0×10⁻⁵ | 80 | ~1.5×10⁻⁹ |
| N in γ-Fe | 1.0×10⁻⁵ | 137 | ~5.0×10⁻¹² |

**关键洞察**：Carbon 在 Ferrite 中的 diffusion 比 Austenite 快约 **100倍**！

---

## 4. Thermodynamics 深度解析

### Magnetic Contribution to Phase Stability

Austenite 是 **paramagnetic**（顺磁性），而 Ferrite 是 **ferromagnetic**（铁磁性）below Curie temperature。

**Magnetic free energy contribution**：

$$G^{mag} = RT\ln(\beta + 1) \cdot f\left(\frac{T}{T_C}\right)$$

其中：
- $\beta$ = Magnetic moment (Bohr magnetons)
- $T_C$ = Curie temperature (K)
- $f(\tau)$ = Function defined by Inden's model

对于 Fe：
- $T_C^{α} \approx 770°C$ (Ferrite)
- $T_C^{γ} \approx 0$ (Austenite 是 paramagnetic)

### Regular Solution Model

Austenite 中 Fe-C system 可用 **Regular Solution Model** 描述：

$$G^{\gamma} = (1-x_C)G_{Fe}^{\gamma} + x_C G_C^{\gamma} + RT[x_C \ln x_C + (1-x_C)\ln(1-x_C)] + x_C(1-x_C)L_{FeC}^{\gamma}$$

其中：
- $x_C$ = Mole fraction of carbon
- $G_{Fe}^{\gamma}$ = Gibbs energy of pure γ-Fe
- $G_C^{\gamma}$ = Gibbs energy of carbon in austenite
- $L_{FeC}^{\gamma}$ = Interaction parameter

**Interaction parameter 的物理意义**：

$$L_{FeC}^{\gamma} = A + B \cdot T$$

实验拟合值：
- $A \approx -83,000$ J/mol
- $B \approx 25$ J/(mol·K)

负值表示 Fe-C interaction 是 **attractive**！

---

## 5. Crystallography 详细分析

### Octahedral vs Tetrahedral Interstitial Sites

```
┌──────────────────────────────────────────────────────────────┐
│           FCC 中的 Interstitial Sites                         │
│                                                              │
│   Octahedral Site (碳原子实际占据的位置)                        │
│   ┌─────────────┐                                            │
│   │     ●       │  位置: Face centers + Edge centers        │
│   │   ／ │ ＼    │  数量: (12 edges × 1/4) + (1 center) = 4  │
│   │ ●   ○   ●  │  Radius ratio: r_c/r_Fe ≈ 0.414           │
│   │   ＼ │ ／    │                                            │
│   │     ●       │  Hole radius: r_o = 0.414 × r_Fe          │
│   └─────────────┘                                            │
│                                                              │
│   Tetrahedral Site (碳原子不常占据)                            │
│   ┌─────────────┐                                            │
│   │      ●      │  位置: Body positions                      │
│   │     ／│＼    │  数量: 8 (inside unit cell)               │
│   │    ● ○ ●   │  Radius ratio: r_c/r_Fe ≈ 0.225           │
│   │     ＼│／    │                                            │
│   │      ●      │  Hole radius: r_t = 0.225 × r_Fe          │
│   └─────────────┘                                            │
│                                                              │
│   r_Fe = atomic radius of Fe ≈ 1.24 Å                       │
│   r_C = covalent radius of C ≈ 0.77 Å                       │
│   Carbon prefers octahedral sites due to larger hole size   │
└──────────────────────────────────────────────────────────────┘
```

### Lattice Distortion by Carbon

碳原子占据 octahedral site 会造成 **tetragonal distortion**：

$$\frac{\Delta a}{a_0} \approx 0.038 \times x_C$$

其中：
- $\Delta a$ = Change in lattice parameter
- $a_0$ = Lattice parameter of pure γ-Fe
- $x_C$ = Carbon concentration (atomic fraction)

**实验数据**：

| C content (wt%) | a (Å) at 900°C |
|-----------------|----------------|
| 0.0 | 3.656 |
| 0.2 | 3.662 |
| 0.4 | 3.668 |
| 0.6 | 3.674 |
| 0.8 | 3.680 |

---

## 6. Phase Transformation Kinetics

### Nucleation and Growth of Austenite

从 Pearlite → Austenite transformation：

**Nucleation rate** (Classical Nucleation Theory)：

$$\dot{N} = N_0 \exp\left(-\frac{Q_D}{RT}\right) \exp\left(-\frac{\Delta G^*}{kT}\right)$$

其中：
- $\dot{N}$ = Nucleation rate (nuclei/m³·s)
- $N_0$ = Pre-exponential factor
- $\Delta G^*$ = Critical nucleation barrier

**Critical nucleus radius**：

$$r^* = \frac{2\gamma_{\alpha\gamma}}{\Delta G_v}$$

其中：
- $r^*$ = Critical radius (m)
- $\gamma_{\alpha\gamma}$ = Interface energy between ferrite and austenite (J/m²)
- $\Delta G_v$ = Volume free energy change (J/m³)

**Growth rate** (Diffusion-controlled)：

$$v = \frac{D}{r} \cdot \frac{C_{eq}^{\gamma} - C_{eq}^{\alpha}}{C^{\gamma} - C^{\alpha}}$$

其中：
- $v$ = Growth velocity (m/s)
- $D$ = Diffusion coefficient
- $r$ = Characteristic diffusion distance

### Time-Temperature-Transformation (TTT) Diagram

```
Temperature (°C)
    │
 900├──────●───────────────────────────────────── A₃
    │      │╲
 850├──────┼─╲───────────────────────────────────
    │      │  ╲
 800├──────┼───╲─────────────────────────────────
    │      │    ╲           Start of transformation
 750├──────┼─────╲─────●────●────●───────────────
    │      │      ╲   ╱      ╲    ╲
 700├──────┼───────╲─●────────╲────╲───────────── A₁
    │      │        ╱          ╲    ╲
 650├──────┼───────●────────────╲────╲───────────
    │      │      ╱              ╲    ╲          50% transformation
 600├──────┼─────●────────────────╲────╲─────────
    │      │    ╱                  ╲    ╲
 550├──────┼───●────────────────────╲────╲───────
    │      │  ╱                      ╲    ╲      Finish
 500├──────┼─●────────────────────────╲────●─────
    │      │╱                          ╲  ╱
    └──────┼────────────────────────────●──────► time (s)
           │          1    10   100   1000  10000
           │
           └──► Austenitization region
```

---

## 7. Austenite in Different Steel Types

### Austenitic Stainless Steel

**Typical composition**：

| Element | Range (wt%) | Function |
|---------|-------------|----------|
| Cr | 17-20 | Corrosion resistance |
| Ni | 8-12 | Austenite stabilizer |
| C | <0.08 | Prevent carbide formation |
| Mn | ≤2.0 | Deoxidizer |
| Si | ≤1.0 | Deoxidizer |

**Schaeffler Diagram** (预测 microstructure)：

$$\text{Ni}_{eq} = \text{Ni} + 30\text{C} + 0.5\text{Mn} + 0.3\text{Cu}$$

$$\text{Cr}_{eq} = \text{Cr} + 1.5\text{Si} + \text{Mo} + 0.5\text{Nb}$$

其中：
- $\text{Ni}_{eq}$ = Nickel equivalent
- $\text{Cr}_{eq}$ = Chromium equivalent

**Phase boundaries**：

| Region | Ni_eq / Cr_eq | Microstructure |
|--------|---------------|----------------|
| Martensite | < 0.4 | M |
| M + γ | 0.4 - 0.7 | M + Austenite |
| Austenite | > 0.7 | γ |
| γ + Ferrite | 0.6 - 0.9 | γ + α |

### TRIP Steel (Transformation Induced Plasticity)

**机制**：Austenite → Martensite transformation under strain

**数学描述** (应变诱导相变动力学)：

$$f_{\alpha'} = 1 - \exp\left[-\beta \cdot (\epsilon - \epsilon_0)^n\right]$$

其中：
- $f_{\alpha'}$ = Volume fraction of martensite
- $\epsilon$ = True strain
- $\epsilon_0$ = Critical strain for transformation
- $\beta, n$ = Material constants

**典型 retained austenite 含量**：

| Steel Type | RA Content | Carbon in RA |
|------------|------------|--------------|
| TRIP 780 | 8-12% | 1.2-1.5% |
| TRIP 980 | 5-10% | 1.4-1.8% |
| Q&P 980 | 5-8% | 1.5-2.0% |

---

## 8. Stacking Fault Energy (SFE) 的关键作用

### SFE 决定变形机制

**Stacking Fault Energy**：

$$\gamma_{SFE} = 2\rho \Delta G^{fcc-hcp} + 2\sigma$$

其中：
- $\gamma_{SFE}$ = Stacking fault energy (mJ/m²)
- $\rho$ = Atomic density on {111} plane = $4/\sqrt{3}a^2$ (atoms/m²)
- $\Delta G^{fcc-hcp}$ = Free energy difference between FCC and HCP
- $\sigma$ = Interface energy (typically ~10 mJ/m²)

**SFE 与变形机制的关系**：

| SFE Range (mJ/m²) | Deformation Mechanism | Material Example |
|-------------------|----------------------|------------------|
| < 10 | ε-martensite formation | High-Mn TWIP |
| 10-20 | Mechanical twinning | TWIP steels |
| 20-35 | TWIP + TRIP transition | Medium-Mn steels |
| 35-50 | Planar slip | Austenitic SS |
| > 50 | Wavy slip | High-Ni steels |

### SFE 计算的 Thermodynamic Model

$$\gamma_{SFE} = 2\rho \left[\Delta G^{chem} + \Delta G^{mag} + \Delta G^{el}\right] + 2\sigma$$

其中：
- $\Delta G^{chem}$ = Chemical contribution
- $\Delta G^{mag}$ = Magnetic contribution  
- $\Delta G^{el}$ = Elastic contribution

**Composition effect on SFE**：

| Element | dSFE/dC (mJ/m² per wt%) |
|---------|--------------------------|
| C | +10 to +15 |
| Ni | +2 to +3 |
| Mn | -0.5 to +0.5 (nonlinear) |
| Si | -3 to -5 |
| Al | +5 to +8 |
| Cr | -1 to -2 |

---

## 9. Grain Size Effects

### Hall-Petch Relationship for Austenite

$$\sigma_y = \sigma_0 + k_y d^{-1/2}$$

其中：
- $\sigma_y$ = Yield strength (MPa)
- $\sigma_0$ = Friction stress (MPa)
- $k_y$ = Hall-Petch coefficient (MPa·mm¹/²)
- $d$ = Grain diameter (mm)

**典型值 for Austenitic Steel**：

| Parameter | Value | Unit |
|-----------|-------|------|
| σ₀ | 50-100 | MPa |
| k_y | 300-500 | MPa·μm¹/² |

### Grain Growth Kinetics

**Normal grain growth**：

$$d^n - d_0^n = Kt$$

其中：
- $d$ = Grain size at time t
- $d_0$ = Initial grain size
- $n$ = Grain growth exponent (typically 2-3)
- $K$ = Temperature-dependent constant
- $t$ = Time

**Temperature dependence**：

$$K = K_0 \exp\left(-\frac{Q_g}{RT}\right)$$

其中：
- $Q_g$ = Activation energy for grain growth (J/mol)
- $Q_g^{austenite} \approx 200-300$ kJ/mol

---

## 10. Martensite Start Temperature (M_s)

### Empirical Equations for M_s

**Andrews equation** (广泛使用)：

$$M_s (°C) = 539 - 423C - 30.4Mn - 17.7Ni - 12.1Cr - 7.5Mo - 10Cu$$

其中元素符号代表 wt% 含量。

**更精确的公式** (考虑 Si 和 Co)：

$$M_s (°C) = 561 - 474C - 33Mn - 17Ni - 17Cr - 21Mo - 10Si - 10Co$$

**适用于 high-alloy steels**：

$$M_s (K) = 834 - 461C - 35.4Mn - 23.8Ni - 20.5Cr - 17.7Mo - 12.7Si$$

### Effect of Austenite Grain Size on M_s

$$M_s(d) = M_s(\infty) + K_{AGS} \cdot d^{-1/2}$$

其中：
- $M_s(d)$ = M_s for grain size d
- $M_s(\infty)$ = M_s for infinite grain size
- $K_{AGS}$ = Grain size coefficient

**实验数据**：

| Grain Size (μm) | M_s (°C) | ΔM_s |
|-----------------|----------|------|
| 10 | 380 | +15 |
| 25 | 370 | +5 |
| 50 | 367 | +2 |
| 100 | 365 | 0 (reference) |

---

## 11. 实际应用中的关键考量

### Heat Treatment Windows

| Process | Temperature Range | Purpose |
|---------|-------------------|---------|
| Normalizing | A₃ + 50-80°C | Refine grain structure |
| Full Annealing | A₃ + 30-50°C | Soften for machining |
| Hardening | A₃ + 30-50°C (hypoeutectoid) | Form martensite |
| Austempering | Ms + 50-100°C (isothermal) | Form bainite |
| Solution Annealing | 1000-1150°C (SS) | Dissolve carbides |

### Critical Heating and Cooling Rates

**Critical cooling rate** (避免 pearlite formation)：

$$v_{crit} = \frac{T_{A_1} - T_{nose}}{t_{nose}}$$

其中：
- $v_{crit}$ = Critical cooling rate (°C/s)
- $T_{nose}$ = Temperature at TTT diagram nose
- $t_{nose}$ = Time at TTT diagram nose

**典型值**：

| Steel Grade | Critical Cooling Rate |
|-------------|----------------------|
| 1045 (0.45%C) | 200-400 °C/s |
| 4140 (Alloy steel) | 10-30 °C/s |
| D2 (Tool steel) | 1-5 °C/s |

---

## 12. References

1. **Krauss, G.** (2015). *Steels: Processing, Structure, and Performance*. ASM International.
   - https://www.asminternational.org/home/-/journal_content/56/10192/06681G/PUBLICATION

2. **Bhadeshia, H.K.D.H., & Honeycombe, R.W.K.** (2017). *Steels: Microstructure and Properties*. Butterworth-Heinemann.
   - https://www.sciencedirect.com/book/9780081002704/steels

3. **Thermodynamic Calculations - Thermo-Calc Software**
   - https://www.thermocalc.com/

4. **Fe-C Phase Diagram Database**
   - https://matthey.com/knowledge/phase-diagrams/

5. **Stacking Fault Energy Database - NIST**
   - https://trc.nist.gov/metals.html

6. **Grain Growth Modeling**
   - https://www.sciencedirect.com/topics/materials-science/grain-growth

7. **TRIP Steel Mechanism Review**
   - https://www.sciencedirect.com/science/article/pii/S0079642516300046

8. **Martensite Transformation Kinetics**
   - https://www.sciencedirect.com/topics/materials-science/martensite-start-temperature

---

## 核心直觉总结

| Aspect                    | Key Insight                                                             |
| ------------------------- | ----------------------------------------------------------------------- |
| **Crystal Structure**     | FCC 的 74% packing efficiency 使 carbon atoms 能够占据 octahedral sites       |
| **Temperature Stability** | Austenite 只在高温稳定，因为 ΔG = H - TS 中 entropy term 主导                       |
| **Carbon Solubility**     | FCC 比 BCC 多 100倍的碳溶解度，因为 interstitial sites 更大                          |
| **Diffusion**             | Carbon 在 austenite 中的 diffusion 比 ferrite 慢 100倍                        |
| **SFE Control**           | SFE 决定 austenite 的变形模式：low SFE → twinning; high SFE → dislocation glide |
| **Alloy Design**          | Ni、Mn、C 是 austenite stabilizers；Cr、Mo、Si 是 ferrite stabilizers          |

# Martensite 的全面解析

## 1. 基本定义与本质

**Martensite** 是通过 **Diffusionless, Shear Transformation**（无扩散切变相变）从 Austenite 快速冷却形成的亚稳态相。以德国冶金学家 **Adolf Martens** 命名。

### 第一性原理理解

```
┌─────────────────────────────────────────────────────────────┐
│         Martensite 的本质特征                                 │
│                                                             │
│   Austenite (FCC)          →          Martensite (BCT/BCC)  │
│                                                             │
│        ● ─── ●                        ● ─── ●               │
│       /│    /│                       │     │               │
│      ● │   ● │   Shear              ●     ●                │
│     /│ │  /│ │  ────────→          │     │                │
│    ● │ │ ● │ │                     ●     ●                 │
│    ││ │  ││ │                      │     │                │
│    ││ ●  ││ ●                       ● ─── ●                │
│    ││/   ││/                                                │
│    ●──── ●                                                  │
│                                                             │
│   Feature 1: 无扩散 (Diffusionless)                         │
│   Feature 2: 切变机制                        │
│   Feature 3: 晶格对应                     │
│   Feature 4: 体积膨胀                         │
│   Feature 5: 高密度位错                           │
└─────────────────────────────────────────────────────────────┘
```

### 核心区别：Diffusional vs Diffusionless Transformation

| Aspect | Diffusional (Pearlite) | Diffusionless (Martensite) |
|--------|------------------------|----------------------------|
| Atom movement | Long-range diffusion | Cooperative shuffle |
| Composition change | Yes (partitioning) | No (composition invariant) |
| Interface velocity | D-dependent, slow | Sound velocity, fast |
| Surface relief | None | Tilting, invariant plane strain |
| Time dependence | Time-dependent | Athermal (time-independent) |

---

## 2. Crystal Structure 详细解析

### Body-Centered Tetragonal (BCT) Structure

Martensite 的 **Tetragonality** 由碳含量决定：

```
┌─────────────────────────────────────────────────────────────┐
│           BCT Martensite 晶胞                                │
│                                                             │
│                    c (tetragonal axis)                      │
│                    ↑                                        │
│                    │                                        │
│              ●─────●─────●                                  │
│             /│     │     │                                  │
│            ● │─────●─────│───→ a                           │
│           /│ │     │     │                                  │
│          ● │ │─────●─────│                                  │
│          ││ │     │     │                                   │
│          ││ ●─────●─────●                                   │
│          ││/      │                                          │
│          ●───────●                                           │
│                                                             │
│   c/a ratio = 1 + 0.045 × (wt% C)                          │
│                                                             │
│   Carbon positions: z ≈ 0.28 (distorted octahedral sites)  │
└─────────────────────────────────────────────────────────────┘
```

### Bain Correspondence Model

**Bain Strain** 描述 FCC → BCT 转变的纯应变部分：

$$\mathbf{B} = \begin{pmatrix} \eta_1 & 0 & 0 \\ 0 & \eta_2 & 0 \\ 0 & 0 & \eta_3 \end{pmatrix}$$

其中：
- $\eta_1 = \eta_2 = \frac{a_{BCT}}{a_{FCC}/\sqrt{2}}$ (contraction in x, y)
- $\eta_3 = \frac{c_{BCT}}{a_{FCC}}$ (expansion in z)

**数值计算**：

| C (wt%) | a (Å) | c (Å) | c/a | $\eta_1, \eta_2$ | $\eta_3$ |
|---------|-------|-------|-----|------------------|----------|
| 0.0 | 2.866 | 2.866 | 1.000 | 0.849 | 1.143 |
| 0.4 | 2.858 | 2.935 | 1.027 | 0.847 | 1.170 |
| 0.8 | 2.850 | 3.005 | 1.054 | 0.844 | 1.198 |
| 1.2 | 2.843 | 3.075 | 1.081 | 0.842 | 1.226 |

### Lattice Parameters vs Carbon Content

**经验公式**：

$$a_{\alpha'} = a_{\alpha} - 0.015 \times (\text{wt\% C}) \quad (\text{Å})$$

$$c_{\alpha'} = a_{\alpha} + 0.118 \times (\text{wt\% C}) \quad (\text{Å})$$

其中：
- $a_{\alpha'}$ = Martensite a-parameter
- $c_{\alpha'}$ = Martensite c-parameter
- $a_{\alpha} = 2.866$ Å (pure α-Fe lattice parameter)

**Tetragonality ratio**：

$$\frac{c}{a} = 1 + 0.045 \times (\text{wt\% C})$$

---

## 3. Crystallographic Orientation Relationships

### 三大经典关系

**1. Kurdjumov-Sachs (K-S) Relationship**

$$(111)_\gamma \parallel (011)_{\alpha'}$$

$$[10\bar{1}]_\gamma \parallel [11\bar{1}]_{\alpha'}$$

**24 variants** 存在。

**2. Nishiyama-Wassermann (N-W) Relationship**

$$(111)_\gamma \parallel (011)_{\alpha'}$$

$$[1\bar{1}0]_\gamma \parallel [100]_{\alpha'}$$

**12 variants** 存在。

**3. Greninger-Troiano (G-T) Relationship**

$$(111)_\gamma \parallel (011)_{\alpha'}$$

旋转约 2.5° from N-W

### Orientation Relationship 的数学表达

```
┌─────────────────────────────────────────────────────────────┐
│         Stereographic Projection 表示 OR                     │
│                                                             │
│                    [001]γ                                   │
│                       │                                     │
│                       │                                     │
│              (010)γ───●───(100)γ                            │
│                      /│\                                    │
│                     / │ \                                   │
│                    /  │  \                                  │
│          (011)α'──●───●───●──(101)α'                       │
│                  /    │    \                                │
│                 /     │     \                               │
│                /      │      \                              │
│          [100]α'─────●─────[010]α'                          │
│                       │                                     │
│                       │                                     │
│                    [001]α'                                  │
│                                                             │
│   K-S:  (111)γ || (011)α', [101]γ || [111]α'              │
│   N-W:  (111)γ || (011)α', [110]γ || [100]α'              │
│   G-T:  Rotated ~2.5° from N-W                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Phenomenological Theory of Martensite Transformation (PTMT)

### Crystallographic Theory 的核心方程

**Total transformation strain**：

$$\mathbf{F} = \mathbf{R} \cdot \mathbf{B} \cdot \mathbf{P}$$

其中：
- $\mathbf{F}$ = Total deformation gradient tensor
- $\mathbf{R}$ = Rigid body rotation (正交矩阵)
- $\mathbf{B}$ = Bain strain (对角矩阵)
- $\mathbf{P}$ = Simple shear (lattice invariant shear)

### Invariant Plane Strain (IPS) Condition

为了满足 habit plane 的存在，需要：

$$\det(\mathbf{F} - \mathbf{I}) = 0$$

或者更精确地：

$$\mathbf{F} = \mathbf{I} + \mathbf{b}_0 \otimes \mathbf{n}_0$$

其中：
- $\mathbf{b}_0$ = Shape strain direction
- $\mathbf{n}_0$ = Habit plane normal

### Shape Strain Parameters

**Magnitude of shape strain**：

$$m_0 = \sqrt{g_1^2 + g_2^2 + g_3^2 + 2g_1g_2\cos\phi + 2g_2g_3\cos\psi + 2g_3g_1\cos\theta}$$

其中：
- $g_1, g_2, g_3$ = Principal strains
- $\phi, \psi, \theta$ = Angles between principal axes

**典型值**：

| Steel Type | $m_0$ | Dilatational component | Shear component |
|------------|-------|------------------------|-----------------|
| Fe-C (0.8%C) | 0.19 | 0.03 | 0.19 |
| Fe-Ni-C | 0.22 | 0.04 | 0.21 |
| Fe-Ni (29%Ni) | 0.20 | 0.02 | 0.20 |

---

## 5. Morphology: Lath vs Plate Martensite

### 临界碳含量决定的形态

```
┌─────────────────────────────────────────────────────────────┐
│         Martensite Morphology vs Carbon Content             │
│                                                             │
│   Volume Fraction of Plate Martensite                       │
│   1.0 │                                            ●────────│
│       │                                      ●─────          │
│   0.8 │                                ●────                 │
│       │                          ●─────                      │
│   0.6 │                    ●─────                            │
│       │              ●─────                                  │
│   0.4 │        ●─────                                        │
│       │  ●─────                                              │
│   0.2 ●────                                                  │
│       │                                                      │
│   0.0 ├──────┼──────┼──────┼──────┼──────┼──────┼───────────│
│       0.0    0.3    0.6    0.8    1.0    1.2    1.4          │
│                      Carbon Content (wt%)                    │
│                                                             │
│   < 0.6 wt% C:  Lath Martensite                             │
│   > 1.0 wt% C:  Plate Martensite                            │
│   0.6-1.0 wt%:  Mixed morphology                            │
└─────────────────────────────────────────────────────────────┘
```

### Lath Martensite 特征

**结构参数**：

| Parameter | Typical Value | Unit |
|-----------|---------------|------|
| Lath width | 0.1-0.5 | μm |
| Lath length | 5-20 | μm |
| Packet size | 10-50 | μm |
| Block width | 1-5 | μm |
| Dislocation density | 10¹⁴-10¹⁵ | m⁻² |

**Hierarchy 结构**：

```
Austenite Grain
    │
    ├── Packet (same habit plane variants)
    │       │
    │       ├── Block (same Bain variant)
    │       │       │
    │       │       └── Lath (sub-block structure)
    │       │               │
    │       │               └── Dislocation cells/tangles
    │       │
    │       └── Block
    │
    └── Packet
```

### Plate Martensite 特征

**结构参数**：

| Parameter | Typical Value | Unit |
|-----------|---------------|------|
| Plate thickness | 0.1-1 | μm |
| Plate length | 10-50 | μm |
| Midrib thickness | ~0.1 | μm |
| Twin spacing | 5-20 | nm |
| Dislocation density | 10¹³-10¹⁴ | m⁻² |

**Transformation sequence**：

1. First-formed plates span entire austenite grain
2. Later plates constrained by prior plates
3. "Geometrically necessary" pattern develops
4. Retained austenite trapped between plates

---

## 6. Thermodynamics of Martensite Formation

### Chemical Free Energy Change

**Driving force for transformation**：

$$\Delta G^{\gamma \rightarrow \alpha'} = G^{\alpha'} - G^{\gamma}$$

At $M_s$ temperature:

$$\Delta G_{M_s}^{\gamma \rightarrow \alpha'} = \Delta G_{chem} + \Delta G_{non-chem} = 0$$

**Non-chemical contributions**：

$$\Delta G_{non-chem} = W_{irr} + W_{rev}$$

其中：
- $W_{irr}$ = Irreversible work (plastic deformation)
- $W_{rev}$ = Reversible work (elastic strain)

### Critical Driving Force

$$|\Delta G_{M_s}| = W_{fr} + \Delta G_{surf} + \Delta G_{strain}$$

其中：
- $W_{fr}$ = Frictional work for interface motion
- $\Delta G_{surf}$ = Surface energy of martensite plate
- $\Delta G_{strain}$ = Elastic strain energy

**典型值**：

| Material | $|\Delta G_{M_s}|$ (J/mol) |
|----------|---------------------------|
| Fe-C (0.4%C) | ~1200 |
| Fe-C (0.8%C) | ~1100 |
| Fe-Ni (28%Ni) | ~1500 |
| Fe-Ni-C | ~1300 |

### Effect of Applied Stress

**Patel-Cohen Equation**：

$$\Delta M_s = \frac{\sigma \cdot g \cdot \cos\phi}{\Delta S^{\gamma \rightarrow \alpha'}}$$

其中：
- $\Delta M_s$ = Change in M_s temperature (K)
- $\sigma$ = Applied stress (MPa)
- $g$ = Transformation shear strain magnitude
- $\phi$ = Angle between stress axis and habit plane normal
- $\Delta S^{\gamma \rightarrow \alpha'}$ = Entropy change of transformation

---

## 7. Kinetics of Martensite Transformation

### Athermal Transformation Kinetics

**Koistinen-Marburger Equation**：

$$f = 1 - \exp\left[-\alpha_m(M_s - T_q)\right]$$

其中：
- $f$ = Volume fraction of martensite
- $\alpha_m$ = Rate parameter (typically 0.011-0.015 K⁻¹)
- $T_q$ = Quenching temperature (K or °C)

**适用条件**：
- Cooling rate > critical cooling rate
- Below M_s temperature
- Isothermal holding not considered

### Isothermal Martensite Formation

**Avrami-type kinetics**：

$$f = 1 - \exp\left(-\beta \cdot t^n\right)$$

其中：
- $\beta$ = Temperature-dependent rate constant
- $t$ = Time
- $n$ = Avrami exponent (typically 1-2 for martensite)

**Time-temperature-transformation for martensite**：

$$\dot{f} = f_0 \exp\left(-\frac{Q}{RT}\right) \cdot (1-f)$$

### Burst Phenomenon

**Autocatalytic nucleation**：

$$\frac{df}{dT} = -\alpha_m(1-f) - \beta_m f(1-f)$$

其中：
- $\alpha_m$ = Normal rate parameter
- $\beta_m$ = Autocatalytic coefficient

**Burst transformation 条件**：
- High carbon content (>1.2%)
- Large austenite grain size
- Applied stress/stress concentration

---

## 8. Dislocation Structure in Martensite

### Lath Martensite 中的位错结构

**Dislocation density estimation**：

$$\rho_d = \frac{K \cdot \epsilon}{b \cdot d}$$

其中：
- $\rho_d$ = Dislocation density (m⁻²)
- $K$ = Geometric factor (~10)
- $\epsilon$ = Transformation strain
- $b$ = Burgers vector magnitude (~2.5×10⁻¹⁰ m)
- $d$ = Lath width (m)

**Experimental data**：

| Carbon Content | $\rho_d$ (m⁻²) | $\rho_t$ (m⁻²) | Dominant defect |
|----------------|----------------|----------------|-----------------|
| 0.1% C | 8×10¹⁴ | <10¹² | Dislocations |
| 0.4% C | 1.2×10¹⁵ | 5×10¹¹ | Dislocations |
| 0.8% C | 1.5×10¹⁵ | 2×10¹³ | Mixed |
| 1.2% C | 5×10¹⁴ | 1×10¹⁴ | Twins |

### Twinning in Plate Martensite

**Twinning system**: {112}$_{\alpha'}$ 〈111〉$_{\alpha'}$

**Twin density**：

$$\rho_{twin} = \frac{1}{t_{twin}}$$

其中：
- $\rho_{twin}$ = Twin density (m⁻¹)
- $t_{twin}$ = Average twin spacing (m)

**Twinning vs Slip transition criterion**：

$$\frac{\tau_{slip}}{\tau_{twin}} < 1 \Rightarrow \text{Slip dominant}$$

$$\frac{\tau_{slip}}{\tau_{twin}} > 1 \Rightarrow \text{Twinning dominant}$$

其中：
- $\tau_{slip}$ = Critical resolved shear stress for slip
- $\tau_{twin}$ = Critical resolved shear stress for twinning

---

## 9. Mechanical Properties

### Strength of Martensite

**Yield strength contributions**：

$$\sigma_y = \sigma_0 + \sigma_{ss} + \sigma_{disl} + \sigma_{ppt} + \sigma_{gs}$$

其中：
- $\sigma_0$ = Lattice friction stress (~50 MPa for Fe)
- $\sigma_{ss}$ = Solid solution strengthening
- $\sigma_{disl}$ = Dislocation strengthening
- $\sigma_{ppt}$ = Precipitation strengthening
- $\sigma_{gs}$ = Grain size strengthening

### Dislocation Strengthening (Taylor Hardening)

$$\sigma_{disl} = M \alpha G b \sqrt{\rho_d}$$

其中：
- $M$ = Taylor factor (≈2.75 for BCC polycrystals)
- $\alpha$ = Constant (0.15-0.5)
- $G$ = Shear modulus (≈80 GPa for Fe)
- $b$ = Burgers vector magnitude
- $\rho_d$ = Dislocation density

**计算示例**：

For $\rho_d = 10^{15}$ m⁻²:

$$\sigma_{disl} = 2.75 \times 0.3 \times 80 \times 10^9 \times 2.5 \times 10^{-10} \times \sqrt{10^{15}}$$

$$\sigma_{disl} \approx 660 \text{ MPa}$$

### Solid Solution Strengthening by Carbon

**Interstitial strengthening**：

$$\sigma_{ss} = k_C \cdot C^{1/2}$$

其中：
- $k_C$ = Strengthening coefficient (~2000 MPa/wt%$^{1/2}$)
- $C$ = Carbon content (wt%)

**Fleischer model** for interstitials:

$$\tau_{ss} = \frac{G \epsilon_s^{3/2} c^{1/2}}{A}$$

其中：
- $\epsilon_s$ = Size misfit parameter
- $c$ = Atomic fraction of solute
- $A$ = Constant (~30)

### Carbon Content vs Hardness

**Empirical relationship**：

$$HRC = 20 + 60 \times (\text{wt\% C})^{1/2}$$

**Experimental data**：

| C (wt%) | As-quenched HRC | $\sigma_{UTS}$ (MPa) | Elongation (%) |
|---------|-----------------|----------------------|----------------|
| 0.2 | 35-40 | 1200-1400 | 5-10 |
| 0.4 | 50-55 | 1800-2200 | 2-5 |
| 0.6 | 58-62 | 2400-2800 | 1-3 |
| 0.8 | 62-65 | 2800-3200 | <2 |
| 1.0 | 64-67 | 3000-3500 | <1 |

---

## 10. Ms Temperature Prediction

### Empirical Equations

**Andrews Equation (Linear)**:

$$M_s (°C) = 539 - 423C - 30.4Mn - 17.7Ni - 12.1Cr - 7.5Mo - 10Cu$$

**Payson-Savage Equation**:

$$M_s (°C) = 497 - 317C - 33Mn - 28Cr - 17Ni - 11Si - 11Mo$$

**Liu Equation (Nonlinear)**:

$$M_s (°C) = 545 - 450C(1 - 0.2C) - 30Mn - 18Ni - 12Cr - 8Mo - 5Si$$

**Effect of Austenite Grain Size**:

$$M_s(d) = M_s(\infty) + K_{AGS} \cdot d^{-1/2}$$

其中：
- $d$ = Grain diameter (m)
- $K_{AGS}$ ≈ 5-15 MPa·m$^{1/2}$

### Effect of Prior Processing

| Processing Factor | Effect on M_s | Mechanism |
|-------------------|---------------|-----------|
| Austenite grain refinement | Slight increase | Reduced nucleation barrier |
| Prior deformation (ausforming) | Increase | Defect-assisted nucleation |
| Austenite strengthening | Decrease | Increased resistance to shear |
| Applied tensile stress | Increase | Mechanical driving force |
| Applied compressive stress | Decrease | Opposes dilatation |

---

## 11. Retained Austenite

### Stability of Retained Austenite

**Reasons for retention**:

1. **Chemical stabilization**: High carbon/nickel content
2. **Mechanical stabilization**: Constrained by surrounding martensite
3. **Size effect**: Thin films between martensite plates
4. **Incomplete transformation**: Stopped above M_f

**Volume fraction estimation**:

$$V_\gamma = V_{\gamma0} \cdot \exp\left[-\alpha_m(M_s - T_q)\right]$$

### Carbon Enrichment in Retained Austenite

**Mass balance**:

$$C_\gamma = \frac{C_0 - f \cdot C_{\alpha'}}{1 - f}$$

其中：
- $C_0$ = Initial carbon content
- $f$ = Martensite fraction
- $C_\gamma$ = Carbon in retained austenite
- $C_{\alpha'}$ = Carbon in martensite

**Experimental observations**:

| Steel | Initial C | Final C in RA | RA Volume Fraction |
|-------|-----------|---------------|-------------------|
| 0.4%C | 0.40% | 0.8-1.2% | 2-5% |
| 0.8%C | 0.80% | 1.2-1.6% | 8-15% |
| TRIP steel | 0.2% | 1.0-1.5% | 5-12% |
| Q&P steel | 0.3% | 1.5-2.0% | 5-10% |

---

## 12. Tempering of Martensite

### Tempering Stages

```
┌─────────────────────────────────────────────────────────────┐
│            Tempering Stages and Microstructural Changes     │
│                                                             │
│ Temperature (°C)                                            │
│                                                             │
│ 700├───────────────────────────────────────────────────────│
│    │         Stage 4: Spheroidization                       │
│ 600├───────────────────────────────────────────────────────│
│    │         Stage 3: Carbide transformation               │
│ 500├───────────────────────────────────────────────────────│
│    │                ε → θ                                  │
│ 400├───────────────────────────────────────────────────────│
│    │         Stage 2: Retained austenite decomposition     │
│ 300├───────────────────────────────────────────────────────│
│    │         Stage 1: Carbon clustering & ε-carbide        │
│ 200├───────────────────────────────────────────────────────│
│    │         precipitation                                 │
│ 100├───────────────────────────────────────────────────────│
│    │         As-quenched martensite                        │
│   0├───────────────────────────────────────────────────────│
│    └───────────────────────────────────────────────────────│
│        Time →                                               │
└─────────────────────────────────────────────────────────────┘
```

### Tempering Kinetics

**Hollomon-Jaffe Parameter**:

$$P = T(C + \log t)$$

其中：
- $P$ = Tempering parameter
- $T$ = Absolute temperature (K)
- $t$ = Time (hours)
- $C$ = Material constant (~20 for steels)

**Hardness prediction**:

$$H = H_0 - k \cdot \log(t) \cdot \exp\left(-\frac{Q_T}{RT}\right)$$

其中：
- $H$ = Hardness after tempering
- $H_0$ = As-quenched hardness
- $k$ = Constant
- $Q_T$ = Activation energy for tempering

### Stage 1: Carbon Segregation and ε-Carbide

**Reaction**:

$$\alpha'_{(supersaturated)} \rightarrow \alpha_{(C-deficient)} + \varepsilon\text{-carbide}$$

**Temperature range**: 100-250°C

**ε-carbide (Fe₂.₄C)**:
- Hexagonal structure
- Composition: Fe₂.₄C to Fe₃C
- Forms as fine needles/plates

### Stage 2: Retained Austenite Decomposition

**Reaction**:

$$\gamma_{retained} \rightarrow \alpha + \theta\text{-carbide (cementite)}$$

**Temperature range**: 200-350°C

**Mechanism**: Diffusional decomposition to bainite-like product

### Stage 3: Cementite Formation

**Reaction**:

$$\varepsilon\text{-carbide} + \alpha \rightarrow \theta\text{-carbide} + \alpha$$

**Temperature range**: 250-400°C

**Cementite (Fe₃C)**:
- Orthorhombic structure
- a = 4.524 Å, b = 5.088 Å, c = 6.741 Å
- Forms initially at lath boundaries, then spheroidizes

### Stage 4: Coarsening and Recovery

**Temperature range**: 400-700°C

**Coarsening kinetics (Ostwald ripening)**:

$$r^3 - r_0^3 = K \cdot t$$

其中：
- $r$ = Average particle radius at time t
- $r_0$ = Initial particle radius
- $K$ = Coarsening rate constant

$$K = \frac{8\gamma V_m D C_\infty}{9RT}$$

其中：
- $\gamma$ = Interface energy
- $V_m$ = Molar volume
- $D$ = Diffusion coefficient
- $C_\infty$ = Equilibrium solubility

---

## 13. Special Martensitic Transformations

### Shape Memory Alloys (SMA)

**Thermoelastic martensite**:

$$|\Delta G_{non-chem}| < |\Delta G_{chem}|$$

**Conditions for thermoelasticity**:

1. Small transformation strain
2. Low frictional resistance
3. Reversible interface motion
4. Fine, self-accommodating variants

**Nitinol (NiTi)**:

| Property | Value |
|----------|-------|
| M_s | 40-80°C |
| A_s | 70-100°C |
| Transformation strain | 6-8% |
| Recovery stress | 500-800 MPa |

### TRIP Effect in Steels

**Transformation-Induced Plasticity**:

$$\varepsilon_{total} = \varepsilon_{elastic} + \varepsilon_{plastic} + \varepsilon_{TRIP}$$

**Strain-induced transformation**:

$$\frac{df_{\alpha'}}{d\varepsilon} = A \cdot (1 - f_{\alpha'}) \cdot \left(\frac{\sigma}{\sigma_0}\right)^n$$

其中：
- $f_{\alpha'}$ = Martensite fraction
- $\sigma$ = Applied stress
- $\sigma_0$ = Reference stress
- $A, n$ = Material constants

---

## 14. Martensite in Different Steel Systems

### Tool Steels

**High-carbon, high-alloy martensite**:

| Steel | C | Cr | Mo | V | HRC (as-quenched) |
|-------|---|----|----|---|-------------------|
| D2 | 1.5% | 12% | 0.8% | 0.8% | 62-65 |
| M2 | 0.85% | 4% | 5% | 2% | 64-66 |
| A2 | 1.0% | 5% | 1% | 0.2% | 60-62 |

**Secondary hardening**:

$$\Delta H = k \cdot f_{ppt}^{1/2} \cdot r^{-1}$$

其中：
- $f_{ppt}$ = Volume fraction of precipitates
- $r$ = Precipitate radius

### Martensitic Stainless Steels

**Composition design**:

| Grade | C | Cr | Mo | Ni |
|-------|---|----|----|----|
| 420 | 0.15% | 13% | - | - |
| 440C | 1.0% | 17% | 0.5% | - |
| 422 | 0.2% | 12% | 1% | 0.5% |

### Maraging Steels

**Low-carbon martensite + precipitation hardening**:

**Strengthening mechanism**:

$$\sigma_y = \sigma_{Fe-Ni} + \sigma_{ppt}$$

| Grade | Ni | Co | Mo | Ti | $\sigma_y$ (MPa) |
|-------|----|----|----|----|------------------|
| Maraging 250 | 18% | 8% | 5% | 0.4% | 1700 |
| Maraging 300 | 18% | 9% | 5% | 0.6% | 2000 |
| Maraging 350 | 18% | 12% | 4% | 1.5% | 2400 |

---

## 15. Defects and Failure Modes

### Quench Cracking

**Causes**:

1. Thermal stress: $\sigma_{th} = E \alpha \Delta T$
2. Transformation stress: Volume expansion (~4%)
3. Stress concentrations at notches/corners

**Cracking criterion**:

$$\sigma_{max} > \sigma_f$$

其中：
- $\sigma_{max}$ = Maximum tensile stress
- $\sigma_f$ = Fracture stress of martensite

**Factors reducing cracking susceptibility**:

| Factor | Effect |
|--------|--------|
| Lower carbon content | Reduced tetragonality strain |
| Uniform section thickness | Reduced thermal gradients |
| Alloy additions (Mo, V) | Improved hardenability, lower critical cooling rate |
| Martempering | Reduced thermal shock |

### Hydrogen Embrittlement

**Martensite susceptibility**:

$$K_{IH} = K_{IC} - \alpha \cdot C_H$$

其中：
- $K_{IH}$ = Threshold stress intensity with hydrogen
- $K_{IC}$ = Fracture toughness without hydrogen
- $C_H$ = Hydrogen concentration
- $\alpha$ = Empirical constant

---

## 16. Modern Characterization Techniques

### X-ray Diffraction Analysis

**Retained austenite measurement**:

$$V_\gamma = \frac{1}{1 + \frac{R_{\alpha'}}{R_\gamma} \cdot \frac{I_{\alpha'}}{I_\gamma}}$$

其中：
- $I_\gamma$ = Integrated intensity of austenite peak
- $I_{\alpha'}$ = Integrated intensity of martensite peak
- $R_\gamma, R_{\alpha'}$ = Calculated R-factors

**Tetragonality measurement**:

$$\frac{c}{a} = \frac{2d_{002}}{d_{200} + d_{020}}$$

### EBSD Analysis

**Dislocation density from KAM (Kernel Average Misorientation)**:

$$\rho_d \approx \frac{2\theta_{KAM}}{b \cdot u}$$

其中：
- $\theta_{KAM}$ = Average KAM angle
- $u$ = Step size

---

## 17. References

1. **Krauss, G.** (2015). *Steels: Processing, Structure, and Performance*. ASM International.
   - https://www.asminternational.org/home/-/journal_content/56/10192/06681G/PUBLICATION

2. **Bhadeshia, H.K.D.H.** *Martensite Transformation*.
   - https://www.phase-trans.msm.cam.ac.uk/teaching/alloys/martensite.html

3. **Wayman, C.M.** (1983). *Crystallography and Crystal Defects*. Addison-Wesley.
   - https://www.sciencedirect.com/topics/materials-science/crystallographic-theory-martensite

4. **Olson, G.B., & Cohen, M.** (1975). *Kinetics of strain-induced martensitic nucleation*. Metallurgical Transactions A.
   - https://link.springer.com/article/10.1007/BF02672253

5. **Patel, J.R., & Cohen, M.** (1953). *Criterion for the action of applied stress in the martensitic transformation*. Acta Metallurgica.
   - https://www.sciencedirect.com/science/article/abs/pii/0001616053902830

6. **Kelly, P.M.** (2003). *Crystallography of Martensite Transformations in Steels*.
   - https://www.sciencedirect.com/topics/materials-science/kurdjumov-sachs-relationship

7. **Speich, G.R., & Leslie, W.C.** (1972). *Tempering of Steel*. Metallurgical Transactions.
   - https://link.springer.com/article/10.1007/BF02643218

8. **Furuhara, T.** *Martensite Crystallography*.
   - https://www.nims.go.jp/mmc/research/report/4-5.pdf

---

## 核心直觉总结

| Aspect             | Key Insight                                                                                        |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| **Nature**         | Diffusionless shear transformation → atomic positions change cooperatively, not independently      |
| **Speed**          | Interface velocity approaches sound speed (~1000 m/s) because no diffusion is needed               |
| **Structure**      | BCT tetragonality comes from carbon trapped in octahedral sites, stretching one axis               |
| **Strength**       | High strength from: solid solution (C) + dislocations + fine lath structure + internal stress      |
| **Morphology**     | Low C → dislocation laths; High C → twinned plates (twinning accommodates higher strain)           |
| **Thermodynamics** | Chemical driving force must overcome strain energy + friction work                                 |
| **Prediction**     | M_s controlled by composition (stabilizes/destabilizes austenite) and processing history           |
| **Tempering**      | Gradual relaxation: C leaves solution → carbides form → dislocations recover → ductility increases |