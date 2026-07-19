---
source_pdf: A molecule with half-Möbius topology.pdf
paper_sha256: 331154a51b0d0dbfb9b21e40ff67c4ec97dc5a93029703c2271339095b1a0dc2
processed_at: '2026-07-17T20:39:26-07:00'
target_folder: Physics
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# A molecule with half-Möbius topology - 深度技术解析

## 1. 核心发现概述

这篇paper报道了一个真正拓扑非平凡的分子体系，C₁₃Cl₂，在NaCl(111)/Au(111)表面上通过atom manipulation合成。它的π轨道基底绕环一圈扭转90°，对应一种全新的 **half-Möbius topology** ($GML^{±1}_4$)。这与传统的Hückel ($GML^0_2$/$GML^0_4$，无扭转) 和Möbius ($GML^1_2$/$GML^2_4$，扭转180°) 都不同。

paper链接: [Science paper](https://www.science.org/doi/10.1126/science.adq6304) (具体DOI需要查询)；Zenodo数据: [10.5281/zenodo.15495263](https://zenodo.org/records/15495263)

## 2. Generalized Möbius-Listing (GML) 拓扑分类

### 2.1 GML体定义

$GML^n_m$ body：cross section有m-fold对称性，一圈circumnavigation中扭转 $(n/m) \times 360°$。符号约定：$n>0$右旋，$n<0$左旋。

| GML body | 扭转角 | edge数 | 回到起点需要的circumnavigation次数 | 拓扑类型 | 实例 |
|---|---|---|---|---|---|
| $GML^0_2$ | 0° | 2 | 1 | Hückel | benzene |
| $GML^0_4$ | 0° | 2 | 1 | Hückel (两个分离π系统) | cyclo[18]carbon |
| $GML^1_2$ | 180° | 1 | 2 | Möbius | Herges' Möbius annulene |
| $GML^2_4$ | 180° | 2 | 2 | Möbius (两个正交) | 假想的螺旋cyclocarbon |
| $GML^{±1}_4$ | ±90° | 1 | 4 | **half-Möbius** (本文) | C₁₃Cl₂ |

参考: [Herges Chem. Rev. 2006](https://pubs.acs.org/doi/10.1021/cr0505465)；[Tavkhelidze GML原论文](https://www.ams.org/mathscinet) (Proceedings of Ukrainian Mathematical Congress 2011)

### 2.2 直觉构建

关键洞察是**edge数和回环次数**：在Hückel $GML^0_2$ 中，绕一圈回到起点；在Möbius $GML^1_2$ 中，绕一圈到"对面"，绕两圈才回到起点；half-Möbius $GML^{±1}_4$ 只有1个edge，需要绕4圈。这本质上对应不同周期的边界条件。

## 3. C₁₃Cl₂ 分子结构与电子拓扑

### 3.1 结构特点

13个碳组成的环 + 2个Cl（在C1和C7）：
- 11个sp-hybridized carbons（C2-C6, C8-C13）— 2-coordinate
- 2个sp²-hybridized carbons（C1, C7）— 3-coordinate，与Cl成键
- **6-bond segment** (C2-C6, 偶数键) — 倾向cumulenic
- **7-bond segment** (C8-C13, 奇数键) — 倾向polyynic（有BLA）

这种**奇偶不匹配**是half-Möbius拓扑稳定的根本原因。

### 3.2 sp vs sp² hybridization的关键区别

**sp²系统** (benzene, Möbius hydrocarbons)：每个C有3个邻居，p轨道方向由几何决定 — 拓扑**structurally encoded**。

**sp系统** (cyclocarbons)：每个C只有2个邻居，两个正交p轨道方向**不被几何固定** — 拓扑**electronic structure决定**。这正是为何cyclocarbon家族特别适合工程化非平凡电子拓扑。

参考: [Garner, Hoffmann, Solomon, ACS Cent. Sci. 2018](https://pubs.acs.org/doi/10.1021/acscentsci.8b00017) — 关于coarctate Möbius orbital basis

### 3.3 even/odd segment的竞争

根据Garner et al.：
- **even-bond cumulenic chain**：能量最小在$\varphi = 90°$，Möbius topology
- **odd-bond cumulenic chain**：能量最小在$\varphi = 0°$，Hückel topology

在C₁₃Cl₂中，两个segment分别倾向于90°和0°，**竞争结果**是平衡扭转角 $\varphi \approx 24°$（on NaCl）。这是结构稳定half-Möbius而非完全Möbius的原因。

## 4. Tight-Binding Hamiltonian (Eq. 1-15)

### 4.1 Hamiltonian形式

$$\hat{H}_{nn} = -\sum_{\langle i,j \rangle}^N \sum_{\langle k,l \rangle}^2 t_{kl} c_{ik}^\dagger c_{jl}$$

变量解释：
- $\langle i,j \rangle$: 对相邻原子 i, j 求和
- $\langle k,l \rangle$: 对每个原子的两个正交 p 轨道 k, l 求和（标记 $p_z$ 和 $p_{xy}$）
- $c_{ik}^\dagger$: 在原子 i 的 k 轨道上产生电子的算符
- $t_{kl}$: 相邻原子 k, l 轨道之间的hopping

### 4.2 Hopping依赖角度 (Eq. 2)

$$t_{kl} = t_0 \cos\phi_{ijkl}$$

其中 $\phi_{ijkl}$ 是相邻原子 i 上 k 轨道和 j 上 l 轨道之间的角度。

### 4.3 Helical basis (Eq. 5-6)

在helical basis中，N个原子累计2θ扭转，所以：

$$\phi_{ij11} = \phi_{ij22} = \frac{2\theta}{N-1}$$

$$\phi_{ij12} = \phi_{ij21} = \frac{\pi}{2} - \frac{2\theta}{N-1}$$

**直觉**：每个相邻原子对之间累积 $2\theta/(N-1)$ 的旋转，N-1个间距累加为 $2\theta$，这是从一端到另一端的总扭转。

### 4.4 Next-nearest-neighbor coupling (Eq. 11-13)

为了让π共轭**穿透sp²中心**，加入次近邻耦合：

$$\hat{H}_{nnn} = -\sum_{\langle k,l\rangle}^2 t_{6,8kl} c_{6k}^\dagger c_{8l} - \sum_{\langle k,l\rangle}^2 t_{13,2kl} c_{13k}^\dagger c_{2l}$$

C6-C8 跨过 C7 (sp²)，C13-C2 跨过 C1 (sp²)。这模拟**hyperconjugation (σ-π mixing)**，是稳定Möbius拓扑的关键。

### 4.5 Bending term (Eq. 15)

$$\hat{H} = \hat{H}_{nn}(t_0, \delta, \theta) + \hat{H}_{nnn}(t_{nnn}, \theta) + 2 \cdot \frac{1}{2} k_{\text{bend}} \sin^2\theta$$

- 最后一项：sp² 碳（C1, C7）的out-of-plane弯曲能量，乘以2因为有2个sp²中心
- $k_{\text{bend}}$: 弯曲力常数
- $\delta$: BLA 参数，模拟7-bond segment的polyynic character

### 4.6 关键结果 (Fig. S3)

参数集 $\delta = 0.1$, $k_{\text{bend}} = t_0$, $t_{nnn} = 0.1t_0$ 给出与CASPT2相符的singlet最小（$\theta \approx 8-15°$），而triplet最小在 $\theta = 0°$（planar Hückel）。

## 5. 螺旋 Pseudo-Jahn-Teller 效应

### 5.1 经典类比：cyclobutadiene

Cyclobutadiene的 $D_{4h} \to D_{2h}$ 畸变是教科书级别的pseudo-Jahn-Teller effect。在 $D_{4h}$ 高对称下，frontier orbitals简并，反芳香性极强；通过 $D_{2h}$ 畸变解除简并，反芳香性缓解。

### 5.2 C₁₃Cl₂的螺旋版本

paper提出helical analogue：
- **未畸变**：planar triplet geometry下，$^1 2_{@triplet}$ 是diradicaloid，dominant configuration $^1|11\rangle$（两个frontier orbitals各1个电子，反平行自旋）
- **畸变**：Cs → C₁ 对称破缺，解除两个frontier orbitals (mirror-symmetric，相反helicity) 的简并
- **结果**：其中一个closed-shell configuration $|20\rangle$ 或 $|02\rangle$ 占主导

| 配置 | $^1|11\rangle$ | $|20\rangle$ | $|02\rangle$ |
|---|---|---|---|
| 占主导的态 | planar singlet (diradicaloid) | $^1 2$-P | $^1 2$-M |
| 几何 | $C_s$ | $C_1$ (右旋) | $C_1$ (左旋) |
| 能量降低 | — | 0.26 eV | 0.26 eV |

参考: [Bersuker Chem. Rev. 2021](https://pubs.acs.org/doi/10.1021/acs.chemrev.0c00947)；[Garner et al. PCCP 2022](https://pubs.rsc.org/en/content/articlelanding/2022/cp/d2cp02309a)

### 5.3 几何-电子相互因果

**关键**: 几何畸变**不来自**简单bond-angle strain，而是**来自**电子结构的拓扑切换需求。triplet geometry下，singlet仍保持ground state，但作为open-shell diradicaloid。通过**结构扭曲**，singlet可以变成closed-shell，能量降低。

NICS(2)zz变化：
- $^3 2$ (planar triplet): +6.2 ppm (anti-aromatic)
- $^1 2_{@triplet}$ (planar singlet): +3.3 ppm (anti-aromatic部分缓解)
- $^1 2$-M / $^1 2$-P (扭曲singlet): -3.7 ppm (弱芳香或弱反芳香)

## 6. Half-Möbius的Berry Phase

### 6.1 边界条件的pseudospinor表示 (Eq. 34-35)

每原子的两个正交p轨道可写成pseudospinor:

$$|\psi\rangle = \binom{\psi_A}{\psi_B}$$

一次circumnavigation的边界条件:

$$U_C = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix} = i\sigma_y$$

这是Bloch sphere上绕y轴的旋转。

**计算**:
- $U_C^2 = -I$ → 两次circumnavigation后波函数变号
- $U_C^4 = I$ → 四次circumnavigation后周期性恢复

### 6.2 与Aharonov-Bohm flux的类比 (Eq. 36-37)

Möbius graphene strip的tight-binding (Guo et al. 2009):

$$\hat{H} = -t \sum_{j=1}^N e^{i\frac{\gamma}{N}} (c_j^\dagger c_{j+1})$$

其中 $\gamma$ 是每次hopping的phase shift。一圈累积总phase $\gamma$，类比于有效Aharonov-Bohm flux：

$$\gamma = \frac{2\pi\Phi}{\Phi_0}$$

其中 $\Phi = B\pi R^2$ 是磁通量, $\Phi_0 = h/e$ 是Dirac flux quantum。

| 拓扑 | $\gamma$ | 等效磁通 $\Phi/\Phi_0$ |
|---|---|---|
| Hückel $GML^0_2$ | 0 | 0 |
| Möbius $GML^1_2$ | $\pi$ | 1/2 |
| Half-Möbius $GML^1_4$ | $\pi/2$ (名义) | 1/4 |

**重要caveat**: 对half-Möbius，interference effect只在**两次完整circumnavigation后**才可观察，所以"Berry phase = π/2"是一种nominal写法，实际可观察的是两次circumnavigation累积π。

参考: [Berry 1984 Proc. R. Soc. A](https://royalsocietypublishing.org/doi/10.1098/rspa.1984.0023)；[Guo et al. PRB 2009](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.80.195310)；[Wang et al. Nat. Photon. 2023](https://www.nature.com/articles/s41566-022-01116-6) — 光学Möbius strip微腔中实验观测Berry相位

### 6.3 轨道角动量量子化 (Eq. 24-33)

粒子在环上的Schrödinger方程:

$$\hat{H}\psi(\phi) = -\frac{\hbar^2}{2Mr^2}\frac{d^2\psi(\phi)}{d\phi^2} = E\psi(\phi)$$

- $\phi$: 方位角
- $M$: 粒子质量
- $r$: 环半径
- $E$: 能量本征值

解：$\psi_n(\phi) = \frac{1}{\sqrt{2\pi}} e^{\pm in\phi}$，能量 $E_n = \frac{n^2\hbar^2}{2Mr^2}$，角动量 $L_z = \pm n\hbar$。

**Hückel** ($2\pi$周期): $n_H = 0, \pm 1, \pm 2, ...$ (整数)
**Möbius** ($4\pi$周期): $n = n_H/2 = 0, \pm 1/2, \pm 1, ...$ (整数 + 半整数)
**Half-Möbius** ($8\pi$周期): $n_{8\pi} = n_H/4 = 0, \pm 1/4, \pm 1/2, \pm 3/4, ...$ (整数 + 半整数 + 四分之一整数)

**直觉**: 拓扑约束下的边界条件周期越长，允许的角动量量子化越细，可以理解为pseudo-quadruple-valued wavefunction。

## 7. 实验方法 - AFM与STM

### 7.1 实验设置

- Home-built combined STM/AFM at T = 5K, UHV
- qPlus sensor, frequency-modulation mode, oscillation amplitude A = 0.5 Å
- CO-functionalized tip
- Bilayer NaCl on Au(111) 作为绝缘衬底
- 前驱体1 (decachlorofluorene, $C_{13}Cl_{10}$) 通过热升华沉积在冷(<10K) Au表面

参考: [Gross et al. Science 2009](https://www.science.org/doi/10.1126/science.1176210) — CO-tip AFM首次单分子结构解析；[Giessibl APL 1998](https://doi.org/10.1063/1.122910) — qPlus sensor原论文

### 7.2 关键实验观察

#### A. AFM成像几何（Fig. 4）

在不同tip-height offset $\Delta z$下采集AFM数据：
- AFM-far: 仅显示整体形状
- AFM-intermediate: 显示out-of-plane distortion ("up"/"down")
- AFM-close: 原子级键分辨

实验观察到的非平面、手性几何与CASPT2计算（θ₁=15.2°, θ₂=9.5°, φ=24.7° on NaCl）极好吻合。

#### B. Switching动力学 (Fig. 5)

I(t)测量显示三个电流plateau：
- **High current**: $^1 2$-M
- **Low current**: $^1 2$-P
- **Intermediate current**: $^3 2$

**关键发现**: 切换阈值 $|V| > 210$ mV，与计算 $^1 2$ vs $^3 2$ 能量差（~0.44 eV on NaCl，但gas phase更小，0.2-0.3 eV量级）相符。

#### C. Switching机制分析 (Fig. S21)

log-log plot of inverse lifetime vs. current:
- $k(^1 2$-M$) = 1.4$, $k(^1 2$-P$) = 1.5$ → 隧穿电子诱导（1-2电子过程）
- $k(^3 2) = 0.2$ → **不主要**由隧穿电子触发，自发衰变寿命~0.1 s

这告诉我们singlet ↔ triplet切换的物理机制是不同的：singlet态受电流影响大，triplet态本质是亚稳态自发衰变。

#### D. STM成像轨道密度 (Fig. 6F)

$^1 2$-M在NIR (V = 1.0V)的STM图像显示**helical LUMO density**，与CASPT2计算的Dyson orbital完美吻合 — 这是half-Möbius拓扑最直接的实验证据。

相比之下，$^3 2$的STM图像（PIR at V = -0.54V）显示**非螺旋、out-of-plane**轨道密度，对应 $GML^0_4$ Hückel拓扑。

## 8. Quantum计算 - SqDRIFT算法

### 8.1 动机

C₁₃Cl₂是强多参考体系（diradicaloid character），传统CI方法处理大active space困难。SqDRIFT (sample-based quantum diagonalization的quantum版) 利用量子硬件作为采样机器。

参考: [Piccinelli et al. arXiv:2508.02578 (2025)](https://arxiv.org/abs/2508.02578)；[Campbell PRL 2019](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.070503) — qDRIFT协议；[Motta et al. Chem. Sci. 2023](https://pubs.rsc.org/en/content/articlelanding/2023/sc/d3sc02539a)

### 8.2 算法架构

**Step 1**: 用CASSCF(12,12)获得初始molecular orbitals
**Step 2**: Hamiltonian映射到量子比特 (Jordan-Wigner mapping, Eq. 17, 18)

$$|\psi\rangle = e^{-iHt}|\psi_0\rangle, \quad H = \sum_i c_i h_i$$

$$e^{-iHt} \approx \prod_k e^{-ih_kt}$$

**关键**: qDRIFT randomization — 每次只采样部分Hamiltonian terms $h_k$，按 $|c_k|^2$ 概率采样。

**Step 3**: 量子硬件执行电路，收集samples
**Step 4**: 经典对角化在sampled subspace中
**Step 5**: ext-SqDRIFT — 在选定的 $|c_i| > \eta$ determinants上做CISD扩展

### 8.3 实际参数

- Active space: **32 electrons in 36 orbitals** (neutral) / 33 electrons in 36 orbitals (anion)
- Jordan-Wigner mapping: **72 qubits**
- 222,814 fermionic second-quantization operators
- 每个分子: 5, 10, 15 excitations × t = 1, 2, 3 a.u. × 500 randomizations = 4500 circuits
- 每circuit采样1024次 → ~4.6M samples/molecule
- 硬件: IBM Heron processor **ibm_kingston** (156 qubits, 2-qubit gate error 2.1×10⁻³, readout error 8.5×10⁻³)
- 经典后处理: subspace up to 20M determinants, 对角化需~10 min on 196 cores

### 8.4 Dyson Orbital计算 (Eq. 21-23)

Dyson orbital是两个不同N-electron wavefunctions之间的overlap:

$$\psi_{Dyson}(\mathbf{r}) = \sum_\mu d_\mu \phi_\mu(\mathbf{r}), \quad d_\mu = \langle a_\mu^\dagger \Psi_{neutral} | \Psi_{anion} \rangle$$

- $a_\mu^\dagger$: 在μ molecular orbital上的creation operator
- $|\Psi_{neutral}\rangle$, $|\Psi_{anion}\rangle$: SqDRIFT得到的correlated wavefunctions
- $d_\mu$: 通过Slater determinant overlap计算

**直觉**: Dyson orbital对应"在添加或移除一个电子时变化的那部分波函数"，可直接与STM测量的orbital density对比。

### 8.5 关键结果

SqDRIFT计算的Dyson orbital (Fig. 3D left) 与经典CASSCF(12,12) (right) 及CASPT2 (Fig. S9A) 一致，说明**12 electrons active space已足够描述half-Möbius拓扑本质**。这是重要validation — 拓扑性质主要来自frontier orbitals附近的相关。

参考: [IBM Quantum Heron](https://www.ibm.com/quantum/blog/ibm-quantum-heron) — 处理器架构；[Barison et al. Quantum Sci. Technol. 2025](https://iopscience.iop.org/article/10.1088/2058-9565/adb9e4) — ext-SqDRIFT

## 9. 物理意义与展望

### 9.1 拓扑切换的可控性

不同于之前Möbius分子中拓扑由结构编码，C₁₃Cl₂的half-Möbius拓扑由**两个monovalent Cl substituent**这一小扰动决定。这允许：
- 切换half-Möbius ↔ Hückel（singlet ↔ triplet）
- 切换handedness ($^1 2$-M ↔ $^1 2$-P)

这种**可控拓扑切换**为研究topology-driven quasiparticle properties打开大门。

### 9.2 推测性质

paper推测：
1. **大磁场响应**: 类似Möbius graphene strip (Guo 2009)，half-Möbius分子可能显示异常磁响应
2. **大分子内电流磁场**: 螺旋frontier orbitals + ring currents → 大磁场 (Bro-Jørgensen et al. JACS Au 2025)
3. **Spin-orbit coupling**: singlet (helical) vs triplet (non-helical) orbital character不同，导致SOC ~2.3 cm⁻¹ → potential **spin-momentum locking** (Gotlieb et al. Science 2018)
4. **Orbital angular momentum**: $L_z$ 可取integer, half-integer, **quarter-integer** values，pseudo-quadruple-valued wavefunction

### 9.3 概念性展望

- 推广到更复杂分子网络with braiding of connectivity and topology
- 可能的switchable braiding topologies
- 与fragile topological insulators (Muechler et al. 2014) 的联系
- 与persistent currents in mesoscopic rings (Levy et al. PRL 1990) 的类比

参考: [Muechler et al. PRB 2014](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.245142)；[Bro-Jørgensen et al. JACS Au 2025](https://pubs.acs.org/doi/10.1021/jacsau.5c00446)；[Gotlieb et al. Science 2018](https://www.science.org/doi/10.1126/science.aap9418)

## 10. 总结与直觉要点

1. **奇偶segment不匹配** → 内禀拓扑倾向竞争 → 稳定half-Möbius而非完全Möbius或完全Hückel
2. **Helical pseudo-Jahn-Teller**: 电子结构驱动几何畸变，类比cyclobutadiene的 $D_{4h} \to D_{2h}$ 但在螺旋自由度上
3. **Berry phase名义π/2，但只在两圈后可观** → 表观8π周期边界条件
4. **sp hybridization是关键**: p轨道方向不被几何固定，允许electronic structure决定topology
5. **Quantum hardware validation**: 32 electrons active space确认了12 electrons active space的sufficiency，为未来研究更大topologically non-trivial systems铺路
6. **拓扑作为可切换property**: 小扰动 (Cl vs H) 控制大拓扑类别，为quantum device设计提供新思路

**最深层的直觉**: 在这个分子中，**electronic topology和geometric chirality是coupled的**，通过pseudo-Jahn-Teller机制相互决定。我们看到的不是"几何决定拓扑"或"拓扑决定几何"，而是**两者共同emerges from minimizing electronic energy under topological constraint**。这是为什么half-Möbius是一种**electronic topology**而非structural topology — 这也是它能被small perturbation切换的根本原因。

参考资源:
- [Cyclo[18]carbon by Kaiser et al. Science 2019](https://www.science.org/doi/10.1126/science.aay1914)
- [Cyclo[13]carbon by Albrecht et al. Science 2024](https://www.science.org/doi/10.1126/science.adq6304)
- [Möbius carbon nanobelt by Segawa et al. Nat. Synth. 2022](https://www.nature.com/articles/s44160-022-00124-0)
- [Triply twisted Möbius by Fan et al. Nat. Synth. 2023](https://www.nature.com/articles/s44160-023-00277-5)
- [IBM Research Zurich - Leo Gross group](https://www.zurich.ibm.com/stories/atomic-scale-imaging)
- [OpenMolcas](https://www.openmolcas.org/)
