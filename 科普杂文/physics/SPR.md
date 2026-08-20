# Surface Plasmon Resonance (SPR) 技术详解

## 一、基本概念与物理原理

### 1.1 什么是 Surface Plasmon

**Surface Plasmon** 是一种在 **metal-dielectric interface**（金属-介质界面）上传播的 **collective oscillation**（集体振荡）of **conduction electrons**（传导电子）。当 **electromagnetic wave** 与金属表面的 **free electrons** 发生 **coherent oscillation** 时，就会产生 **surface plasmon polariton (SPP)**。

### 1.2 SPR 的物理本质

SPR 现象基于以下关键物理过程：

1. **Evanescent Wave Generation**: 当光从 **high refractive index medium**（高折射率介质）入射到 **low refractive index medium**（低折射率介质）的界面，且入射角大于 **critical angle** 时，会发生 **total internal reflection (TIR)**，同时在界面处产生 **evanescent wave**（倏逝波）。

2. **Momentum Matching**: 当 **evanescent wave** 的 **wave vector** 与 **surface plasmon** 的 wave vector 相匹配时，发生 **resonance coupling**。

---

## 二、数学描述与公式详解

### 2.1 Wave Vector of Evanescent Wave

对于 **Kretschmann configuration**（最常用的 SPR 配置），evanescent wave 的波矢量为：

$$k_{ev} = k_0 n_{glass} \sin\theta$$

其中：
- $k_0 = \frac{2\pi}{\lambda}$：真空中的波矢量（$\lambda$ 为入射光波长）
- $n_{glass}$：玻璃棱镜的 **refractive index**
- $\theta$：入射角

### 2.2 Surface Plasmon Wave Vector

Surface plasmon 在金属-介质界面的波矢量为：

$$k_{sp} = k_0 \sqrt{\frac{\varepsilon_m \varepsilon_d}{\varepsilon_m + \varepsilon_d}}$$

其中：
- $\varepsilon_m$：金属的 **complex dielectric constant**（复介电常数），$\varepsilon_m = \varepsilon_m' + i\varepsilon_m''$
  - $\varepsilon_m'$：实部，代表 **polarization** 能力
  - $\varepsilon_m''$：虚部，代表 **absorption loss**
- $\varepsilon_d$：介质的 **dielectric constant**（通常为正实数）

### 2.3 Resonance Condition

当 $k_{ev} = k_{sp}$ 时，发生 SPR：

$$n_{glass} \sin\theta_{res} = \sqrt{\frac{\varepsilon_m \varepsilon_d}{\varepsilon_m + \varepsilon_d}}$$

$\theta_{res}$ 称为 **resonance angle**（共振角）。

### 2.4 Reflectance 公式（Fresnel Equations 扩展）

对于多层薄膜系统的反射率计算：

$$R_p = \left| \frac{r_{01} + r_{12}e^{2ik_{z1}d_1}}{1 + r_{01}r_{12}e^{2ik_{z1}d_1}} \right|^2$$

其中：
- $r_{ij}$：界面 $i-j$ 的 **Fresnel reflection coefficient** for p-polarized light
- $k_{zi}$：第 $i$ 层中波矢量的 $z$ 分量
- $d_1$：金属薄膜厚度

**Fresnel coefficient** for p-polarization：

$$r_{ij}^p = \frac{\varepsilon_j k_{zi} - \varepsilon_i k_{zj}}{\varepsilon_j k_{zi} + \varepsilon_i k_{zj}}$$

### 2.5 Penetration Depth

Evanescent wave 在介质中的 **penetration depth**：

$$\delta_d = \frac{\lambda}{2\pi} \sqrt{\frac{\varepsilon_m' + \varepsilon_d}{-\varepsilon_d^2}}$$

典型值约为 **100-300 nm**，这决定了 SPR 的 **sensing depth**。

---

## 三、实验装置与架构

### 3.1 Kretschmann Configuration（最常用）

```
                    Photodetector
                         ↑
                         |
    Prism (High n)    ◄──┘ Reflected Light
    ┌─────────────┐
    │             │
    │      ↓      │──── Metal Film (Au/Ag ~50nm)
    │   Incident  │
    │    Light    │──── Sample/Analyte
    │   (p-pol)   │
    └─────────────┘
```

**关键组件**：

| 组件 | 材料/规格 | 功能 |
|------|-----------|------|
| **Prism** | SF10 glass, n≈1.7-1.8 | 提供 high momentum for coupling |
| **Metal Film** | Gold (40-50nm) 或 Silver (40-50nm) | 支持 surface plasmon |
| **Light Source** | LED (670-980nm) 或 Laser | 提供 monochromatic 或 polychromatic illumination |
| **Detector** | Photodiode array 或 CCD | 检测 reflectivity 变化 |

### 3.2 Otto Configuration

适用于 **thick metal samples** 或 **非透明金属**：

```
         Prism
    ┌──────────┐
    │     ↓    │
    │  Light   │
    └──────────┘
         ↓
    [Air Gap ~1μm]
         ↓
    ════════════════  Metal Surface
```

**特点**：通过调节 **air gap** 来控制 coupling strength。

### 3.3 SPR 仪器架构图

```
┌────────────────────────────────────────────────────────────┐
│                    SPR Instrument                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   ┌─────────┐    ┌─────────┐    ┌──────────────────┐      │
│   │  LED/   │    │ Polarizer│    │    Prism        │      │
│   │ Laser   │───►│ (p-pol) │───►│   Coupler      │      │
│   └─────────┘    └─────────┘    └────────┬─────────┘      │
│                                           │                │
│                                           ▼                │
│                              ┌──────────────────────┐     │
│                              │   Sensor Chip        │     │
│                              │  ┌────────────────┐  │     │
│                              │  │ Glass substrate│  │     │
│                              │  ├────────────────┤  │     │
│                              │  │  Au film (50nm)│  │     │
│                              │  ├────────────────┤  │     │
│                              │  │ Ligand layer   │◄─┼─────┼── Analyte
│                              │  └────────────────┘  │     │
│                              └──────────┬───────────┘     │
│                                         │                  │
│                                         ▼                  │
│                              ┌──────────────────────┐     │
│                              │  Microfluidics       │     │
│                              │  (Sample delivery)   │     │
│                              └──────────────────────┘     │
│                                                            │
│   ┌──────────────────────────────────────────────────┐    │
│   │           Detection & Analysis                    │    │
│   │  ┌─────────────┐    ┌────────────────────────┐  │    │
│   │  │ Photodiode  │───►│ Data Processing Unit   │  │    │
│   │  │ Array       │    │ (Resonance angle/RU)   │  │    │
│   │  └─────────────┘    └────────────────────────┘  │    │
│   └──────────────────────────────────────────────────┘    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 四、Sensing Mechanism 与工作原理

### 4.1 Resonance Angle Shift

当 **analyte** 在 **sensor surface** 上发生 **binding** 时，局部 **refractive index** 发生变化：

$$\Delta n = \left(\frac{dn}{dc}\right) \cdot \Delta c$$

其中：
- $\frac{dn}{dc}$：**refractive index increment**（通常 ~0.18-0.19 mL/g for proteins）
- $\Delta c$：表面 **concentration change**

这导致 **resonance angle shift**：

$$\Delta\theta_{res} = S \cdot \Delta n$$

其中 $S$ 为 **sensitivity**（单位：°/RIU）。

### 4.2 Response Units (RU)

标准 **Biacore** 系统定义：

$$1 \text{ RU} = 10^{-6} \text{ RIU} \approx 1 \text{ pg/mm}^2$$

典型实验响应范围：**100-10,000 RU**

### 4.3 Sensitivity 计算

对于 **angular interrogation** 模式：

$$S_\theta = \frac{d\theta_{res}}{dn_s} = \frac{k_0}{k_{ev}} \cdot \frac{|\varepsilon_m'|}{\sqrt{-\varepsilon_m'}} \cdot \frac{1}{\sqrt{\varepsilon_m' + \varepsilon_d}}$$

对于 **wavelength interrogation** 模式：

$$S_\lambda = \frac{d\lambda_{res}}{dn_s} = \frac{\lambda}{n_{eff}} \cdot \frac{\partial n_{eff}}{\partial n_s}$$

---

## 五、Kinetics Analysis（动力学分析）

### 5.1 Langmuir Binding Model

最简单的 **1:1 binding model**：

**Association phase**：
$$\frac{dR}{dt} = k_a C (R_{max} - R) - k_d R$$

**Dissociation phase**（$C=0$）：
$$\frac{dR}{dt} = -k_d R$$

解析解：
- Association: $R(t) = R_{eq}(1 - e^{-(k_a C + k_d)t})$
- Dissociation: $R(t) = R_0 e^{-k_d t}$

其中：
- $R$：SPR response at time $t$
- $k_a$：**association rate constant** (M⁻¹s⁻¹)
- $k_d$：**dissociation rate constant** (s⁻¹)
- $C$：analyte concentration
- $R_{max}$：maximum binding capacity
- $R_{eq} = \frac{k_a C R_{max}}{k_a C + k_d}$：equilibrium response

### 5.2 Equilibrium Dissociation Constant

$$K_D = \frac{k_d}{k_a} = \frac{[A][L]}{[AL]}$$

单位为 **Molar (M)**，数值越小表示 **affinity** 越强。

### 5.3 Complex Binding Models

| Model | Equation | 应用场景 |
|-------|----------|----------|
| **Two-state** | $A + L \rightleftharpoons AL \rightleftharpoons AL^*$ | Conformational change |
| **Bivalent** | $A + L \rightleftharpoons AL + L \rightleftharpoons AL_2$ | Antibody binding |
| **Heterogeneous** | $A + L_1 \rightleftharpoons AL_1$; $A + L_2 \rightleftharpoons AL_2$ | Multiple binding sites |

---

## 六、Sensor Chip 类型与 Surface Chemistry

### 6.1 Commercial Sensor Chips

| Chip Type | Surface Coating | 应用 |
|-----------|-----------------|------|
| **CM5** | Carboxymethyl dextran | 通用，protein immobilization |
| **NTA** | Nitrilotriacetic acid | His-tagged proteins |
| **SA** | Streptavidin | Biotinylated molecules |
| **C1** | Flat carboxylate | Small molecules |
| **L1** | Lipid-capturing | Membrane proteins |
| **HPA** | Hydrophobic | Lipid bilayer formation |

### 6.2 Immobilization Chemistry

**Amine Coupling**（最常用）：

```
Step 1: Activation
COOH-dextran + EDC/NHS → NHS-ester

Step 2: Coupling  
NHS-ester + NH₂-protein → Amide bond

Step 3: Deactivation
Remaining NHS-ester + Ethanolamine → Blocked
```

**化学方程式**：
$$\text{-COOH} + \text{EDC} \xrightarrow{\text{NHS}} \text{-COO-NHS} \xrightarrow{\text{-NH}_2} \text{-CONH-} + \text{NHS}$$

### 6.3 Dextran Matrix Effect

**Carboxymethyl dextran** 的三维结构：
- 厚度：~100-200 nm
- 延伸了 **sensing volume**
- 提供 **high binding capacity**
- 但可能导致 **mass transport limitation**

---

## 七、实验数据解读

### 7.1 典型 Sensorgram

```
Response (RU)
    │
    │     ┌───────┐
    │    /│       │\
    │   / │       │ \
    │  /  │       │  \_______
    │ /   │       │          \
    │/    │       │           \
    └────┴──┴─────┴───────────┴───► Time
         ↑  ↑     ↑           ↑
         │  │     │           │
      Baseline  Association  Dissociation  Regeneration
      (buffer)  (analyte)    (buffer)      (acid/base)
```

### 7.2 常见问题与 Artifacts

| Artifact | 原因 | 解决方案 |
|----------|------|----------|
| **Bulk shift** | Buffer refractive index change | Reference subtraction |
| **Mass transport** | Diffusion limited | Lower ligand density, higher flow rate |
| **Non-specific binding** | Non-specific interactions | Add surfactant, optimize buffer |
| **Drift** | Temperature fluctuation | Temperature stabilization |
| **Refolding** | Harsh regeneration | Milder regeneration conditions |

### 7.3 Data Quality Indicators

```
Good Data Characteristics:
├── S/N ratio > 10
├── Baseline stability < 1 RU/min
├── Reproducibility (CV < 5%)
├── χ² of fit < 10% of Rmax
└── Residuals randomly distributed
```

---

## 八、SPR 的变体与新技术

### 8.1 Surface Plasmon Resonance Imaging (SPRi)

**原理**：使用 **CCD camera** 对整个 sensor surface 进行 **real-time imaging**。

**优势**：
- **High-throughput** screening
- 同时监测 **多个 spots**
- 可视化 **binding distribution**

### 8.2 Localized Surface Plasmon Resonance (LSPR)

基于 **metal nanoparticles** (Au, Ag) 的 SPR：

$$\lambda_{max} \propto \varepsilon_m + 2\varepsilon_d$$

**特点**：
- 不需要 prism coupling
- 便携式设备
- Sensitivity 相对较低
- Penetration depth ~10-30 nm

### 8.3 Long-Range Surface Plasmon Resonance (LRSPR)

使用 **thin metal film** 夹在 **symmetric dielectric layers** 之间：

**优势**：
- 更 narrow resonance dip
- 更 high sensitivity
- 更 deep penetration depth (~μm)

### 8.4 Fiber Optic SPR

将 **metal coating** 直接镀在 **optical fiber** 上：

$$S_{FO-SPR} = \frac{\Delta\lambda_{res}}{\Delta n}$$

**应用**：Remote sensing, in-situ monitoring

---

## 九、应用领域详解

### 9.1 Drug Discovery

| 应用 | 描述 |
|------|------|
| **Hit Validation** | 确认 small molecule 与 target 的 binding |
| **Lead Optimization** | 比较 binding affinity of analogs |
| **ADME Studies** | Protein-drug interaction studies |
| **Fragment-based Screening** | Low MW compounds (<300 Da) |

### 9.2 Protein-Protein Interactions

```
典型实验流程:
1. Ligand immobilization (Protein A)
2. Analyte injection (Protein B)
3. Binding kinetics measurement
4. Affinity determination
5. Competition/inhibition studies
```

### 9.3 Biomolecular Interaction Analysis 表

| Interaction Type | K_D Range | 典型 Examples |
|------------------|-----------|---------------|
| **Very strong** | pM - nM | Antibody-antigen, biotin-streptavidin |
| **Strong** | nM - μM | Enzyme-inhibitor, receptor-ligand |
| **Moderate** | μM - mM | Protein-carbohydrate |
| **Weak** | mM | Fragment screening |

---

## 十、SPR 与其他技术的比较

### 10.1 技术对比表

| Technique | Label-free | Kinetics | Throughput | Sensitivity | Sample Req. |
|-----------|------------|----------|------------|-------------|-------------|
| **SPR** | ✓ | ✓ | Medium | High | Low |
| **ITC** | ✓ | ✗ | Low | Medium | High |
| **ELISA** | ✗ | ✗ | High | High | Low |
| **BLI** | ✓ | ✓ | High | Medium | Low |
| **MST** | ✓ | ✗ | Medium | Medium | Low |
| **QCM** | ✓ | ✓ | Low | Medium | Medium |

### 10.2 BLI vs SPR

**Bio-Layer Interferometry (BLI)**：

原理：
$$\lambda = 2n \cdot d$$

$n$：film refractive index, $d$：thickness

| 特性 | SPR | BLI |
|------|-----|-----|
| **Configuration** | Prism-based | Fiber optic |
| **Sample consumption** | Higher | Lower |
| **Throughput** | 1-4 channels | 8-96 channels |
| **Data quality** | Higher | Good |
| **Cost per run** | Higher | Lower |

---

## 十一、技术参数优化指南

### 11.1 Metal Film 优化

**Gold vs Silver**：

| Parameter | Gold | Silver |
|-----------|------|--------|
| **Resonance sharpness** | Moderate | Sharper |
| **Chemical stability** | Excellent | Poor (oxidizes) |
| **Biocompatibility** | Excellent | Good |
| **Cost** | Higher | Lower |
| **Sensitivity** | Moderate | Higher |

**Optimal Thickness**：

$$d_{opt} = \frac{\lambda}{4\pi} \cdot \frac{1}{\sqrt{|\varepsilon_m'|}}$$

对于 Gold at 760nm：$d_{opt} \approx 47-50$ nm

### 11.2 Flow Rate 优化

**Mass Transport Limited Condition**：

$$k_m = \frac{D}{h} \cdot \sqrt[3]{\frac{4Q}{\pi D L W^2}}$$

其中：
- $D$：diffusion coefficient
- $h$：flow cell height
- $Q$：flow rate
- $L$：flow cell length
- $W$：flow cell width

**Rule of Thumb**：
- 对于 $k_a > 10^6$ M⁻¹s⁻¹：使用高 flow rate (>50 μL/min)
- 避免 **mass transport limitation**

### 11.3 Temperature Control

温度影响：
$$\frac{dn}{dT} \approx -10^{-4} \text{ /°C (water)}$$

温度稳定性要求：**< ±0.01°C**

---

## 十二、数学推导深入

### 12.1 Drude Model for Metal Dielectric Function

金属的 dielectric function：

$$\varepsilon_m(\omega) = 1 - \frac{\omega_p^2}{\omega^2 + i\gamma\omega}$$

其中：
- $\omega_p = \sqrt{\frac{n_e e^2}{m_e \varepsilon_0}}$：**plasma frequency**
- $\gamma$：**damping constant**
- $n_e$：electron density
- $m_e$：effective electron mass

对于 Gold：
- $\omega_p \approx 9$ eV
- $\gamma \approx 0.07$ eV

### 12.2 Dispersion Relation

Surface plasmon 的 **dispersion relation**：

$$k_{sp}(\omega) = \frac{\omega}{c} \sqrt{\frac{\varepsilon_m(\omega)\varepsilon_d}{\varepsilon_m(\omega) + \varepsilon_d}}$$

**Light line**：
$$k_{light} = \frac{\omega}{c} n_{glass}$$

SPR coupling 需要 $k_{sp} = k_{light}$，这只能通过 **prism coupling** 或 **grating coupling** 实现。

### 12.3 Figure of Merit (FOM)

评估 SPR sensor 性能的关键指标：

$$FOM = \frac{S}{FWHM}$$

其中：
- $S$：sensitivity
- $FWHM$：resonance peak 的 **full width at half maximum**

---

## 十三、实验设计实例

### 13.1 抗原-抗体相互作用实验

**实验参数**：

| Parameter | Value |
|-----------|-------|
| **Ligand** | Anti-BSA antibody |
| **Analyte** | BSA |
| **Immobilization** | Amine coupling |
| **Running buffer** | HBS-EP (pH 7.4) |
| **Flow rate** | 30 μL/min |
| **Analyte concentrations** | 0.78 - 100 nM (2-fold dilution) |
| **Contact time** | 180 s |
| **Dissociation time** | 600 s |
| **Regeneration** | 10 mM Glycine-HCl pH 2.0 |

**Expected Results**：
- $k_a$: $10^4 - 10^6$ M⁻¹s⁻¹
- $k_d$: $10^{-4} - 10^{-2}$ s⁻¹
- $K_D$: nM range

### 13.2 Small Molecule Screening

**挑战**：Small molecule 引起的 RU 变化很小

**解决方案**：
1. **High ligand density immobilization**
2. **Competition assay format**
3. **Signal amplification strategies**

**公式**：

$$R_{small} = \frac{MW_{analyte}}{MW_{ligand}} \cdot R_{max} \cdot \frac{[A]}{K_D + [A]}$$

---

## 十四、References 与 Web Links

### 学术资源

1. **SPR 原理经典文献**：
   - Kretschmann, E., & Raether, H. (1968). *Zeitschrift für Naturforschung A*, 23(12), 2135-2136.
   - Link: https://www.degruyter.com/document/doi/10.1515/zna-1968-1247/html

2. **Biacore 技术手册**：
   - GE Healthcare Life Sciences SPR Handbook
   - Link: https://www.cytivalifesciences.com/en/us/solutions/protein-research/surface-plasmon-resonance

3. **SPR 教程**：
   - Cornell University SPR Tutorial
   - Link: https://www.nbic.org/resources/tutorials/spr-tutorial/

4. **Review Article**：
   - Schuck, P. (1997). *Annual Review of Biophysics and Biomolecular Structure*, 26, 541-566.
   - Link: https://www.annualreviews.org/doi/10.1146/annurev.biophys.26.1.541

### 商业资源

5. **Biacore (Cytiva)**：
   - https://www.cytivalifesciences.com/en/us/solutions/protein-research/surface-plasmon-resonance

6. **Biacore T200 Technical Note**：
   - Link: https://www.cytivalifesciences.com/GLOBAL/en/product/biacore-t200

7. **Reichert SPR**：
   - https://www.reichertai.com/spr/

8. **Nicoya OpenSPR**：
   - https://nicoyalife.com/surface-plasmon-resonance/

### 教学资源

9. **SPR Animation & Tutorial**：
   - https://www.biacore.com/lifesciences/technology/spr_technology/index.html

10. **MIT OpenCourseWare - Biomaterials**：
    - https://ocw.mit.edu/courses/biological-engineering/

11. **SPR Imaging Tutorial**：
    - https://www.horiba.com/usa/scientific/products/surface-plasmon-resonance-imaging-spri/

### 数据分析软件

12. **BIAevaluation Software**：
    - https://www.cytivalifesciences.com/en/us/solutions/protein-research/surface-plasmon-resonance/data-analysis

13. **Scrubber (BioLogic Software)**：
    - https://www.biologic.com.au/scrubber.html

---

## 十五、Intuition Building 总结

### 核心 Intuition Points：

1. **Momentum Matching**: SPR 本质上是光子与 surface electron oscillation 的 momentum 匹配过程。

2. **Evanescent Sensing**: SPR 只 "看到" 表面 ~200-300 nm 范围内的变化，这是其 specificity 的来源。

3. **Binding = RI Change = Angle Shift**: 任何引起 surface refractive index 变化的过程都会导致 resonance angle shift。

4. **Kinetics from Shape**: Sensorgram 的形状包含了 association 和 dissociation 的 kinetic 信息。

5. **Affinity from Equilibrium**: Steady-state response 与 affinity 直接相关。

6. **Mass Transport Matters**: 扩散限制了观察到的 binding rate，需要实验设计来避免。

7. **Surface Chemistry is Critical**: Ligand immobilization 的质量决定了实验的成功与否。

---

如果您需要更深入了解某个特定方面，例如 **specific applications**、**advanced data analysis**、或 **troubleshooting specific experiments**，请告诉我！