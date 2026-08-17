---
source_pdf: Designing a Magnetic Endoscope for In Vivo Contact-Based Tissue.pdf
paper_sha256: 97dc754a5dd75cfa3f1626c9dc26bd3ee06c840ad51c2bcec506e119f2743e85
processed_at: '2026-08-03T19:59:31-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲一遍

## 一句话说清楚

他们做了个**磁控内窥镜**, 长得像个**橄榄球+圆筒的混血**, 塞进猪大肠里, 靠外面的机械臂举着大磁铁吸着它走。最妙的地方: 这个形状本身让它能在肠壁上**自动滚转**, 把之前磁控一直搞不定的"roll自由度"给找回来了。

## 为啥需要这玩意

传统colonoscopy你知道的, 一根管子硬捅进去, 病人痛苦, 要打sedation, 还可能捅穿。后来有人搞**胶囊内窥镜** — 吞颗带camera的pill, 但不能控制方向。

所以中间路线是 **magnetic flexible endoscope (MFE)**: 一根细管子, 头部有magnet, 外面用robot arm举着大magnet吸它, 像遥控车一样在肠道里开。好处是front-driven, 不捅, 病人舒服。

但有个**物理死结**: magnetic控制最多只能做到5个DoF。为啥? 因为torque公式 $\boldsymbol{\tau} = \mathbf{m_I} \times \mathbf{B_E}$ — torque永远垂直于magnet方向, 所以**绕磁铁自身轴的转动, 外面怎么搞都没法直接产生**。这个"lost DoF"就是roll。

没roll, 你想做surface scanning (比如贴着肠壁扫一圈做ultrasound biopsy) 就做不到 — sensor永远朝一个方向。

## 核心trick

**别想着直接actuate roll, 让形状自己逼出roll**。

他们找了个叫 **oloid** 的几何形状。这玩意是1929年一个瑞士人Paul Schatz发现的, 像两个互相垂直的圆弧挤出来的convex hull, 表面全是smooth curve, 没有尖角。

Oloid最骚的性质: 你让它pitch和yaw来回摆, 它在平面上rolling时**会自然产生roll运动**。Center of mass还始终保持一个高度, 不会上下颠簸。这是个纯几何效应 — developable surface, 每条generator line依次接地。

类比一下: 想象你拿个橄榄球, 不是圆球那种, 是那种两头尖中间圆的, 你让它前后左右晃, 它会自己滚转。Oloid就是这个效应的数学理想型。

所以只要外部magnet能控制pitch + yaw, oloid头自己在tissue上蹭蹭就roll了。**不耗额外电, 不加motor, 不增重量**, 纯靠geometry。

## 为啥选oloid不选sphericon

Sphericon是oloid的表亲, 也能做类似的事。但paper里给了三点理由:

1. **Oloid光滑没尖角**, sphericon有4个vertex — 进大肠, 尖角就是事故
2. **同等截面, oloid内部空间大50%** — 要塞magnet core, camera, insufflation tube, 这些都是圆柱形, oloid更友好
3. **30° vs 45° 倾角** — oloid更"温和", rolling motion更平滑

## 但不能直接用完整oloid

完整oloid空间利用率太低 — bounding box里只占50% volume。装下所有components的话, 整个内窥镜会大到**10倍于普通圆柱形内窥镜**, 进不了人肠。

所以做了个 **hybrid**: 一半oloid + 一半cylinder。Cylinder那部分装magnet core和camera, oloid那部分管roll recovery。Volume ratio升到0.63, roll range还有180°, 整体尺寸20×20×35mm, 跟现有MFE差不多。

## 控制怎么搞

外面是个 **KUKA 7-DoF机械臂**, 末端挂着个大cylindrical permanent magnet (EPM)。内窥镜头部有个小magnet (IPM)。

数学上: 把magnetic dipole-dipole的force/torque model做**线性化** (Jacobian), 每个时间步重算一次。用户joystick输入想要的force/torque, PID给error correction, pseudoinverse Jacobian把magnetic command转成EPM的position+orientation命令, 再用一次pseudoinverse Jacobian把EPM命令转成KUKA 7个关节的速度。

注意EPM绕自己磁化轴旋转**没影响** — 这就是公式(6)里skew-symmetric matrix在干的事, 它自动把那个useless DoF过滤掉。

## Contact sensor

为了验证"sweep时确实贴着tissue", 头部加了8个capacitive电极, 2×4排布, 占40平方毫米。Capacitive的好处是薄 (0.1mm), 不需要压力, flexible PCB直接贴上。采样50Hz, 分4个quarter输出binary contact/no-contact。

## In vivo实验

39kg猪, general anesthesia, enema清肠, OME从rectum插入20cm (pig colon那段是直的, 适合测试)。

两种motion:
- **Sweep**: 上半圆surface扫一遍, 100° roll range
- **Roll**: ±50°纯rolling

结果: 100° sweep里, **74%的时间4个pad全接触**, 单个pad平均92%接触率。最弱的是对角线 (pad 0 & 3), 最强是同侧相邻 (pad 0 & 1) — 符合直觉, rolling时对角线先lift off。

## 还有哪些坑没填

1. **Open-loop控制**: joystick手动开, 没用contact feedback做闭环 — 这是下一步, 能把74%提到95%+
2. **100° not 360°**: 真要360° sweep做µUS的话, 这range还不够
3. **Tissue damage没量化**: 滚100°在肠壁上摩擦, mucosal stress多少没测
4. **弯曲段没测**: 直colon能work, 脾曲肝曲那种bend处行为未知
5. **Subsurface sensor还没真正集成**: 只测了contact sensor, 真正的µUS / Terahertz / OCT还要后续工作

## 最终intuition

这个paper的核心哲学: **actuator不够, geometry来凑**。

传统思路是"我要6个DoF就加6个motor", 这里是"我有5个DoF + 一个特殊形状, 凑出第6个"。代价是geometry变复杂, space utilization下降。收益是zero power, zero extra failure mode, mechanical simplicity。

这种**underactuated + geometric coupling**思路在robotics里其实一直都在 (diff drive, falling cat reorientation, snake locomotion), 但在medical device领域用oloid做magnetic endoscope的roll recovery, 是头一次。

Pietro Valdastri这个组在magnetic endoscopy上做了很多年, 从Science Robotics到Nature Machine Intelligence都有。这篇IROS是他们在clinical translation路上又往前蹭了一小步 — 没特别fancy, 但很扎实, 是个clever的engineering trick。

---

想深挖具体哪个部分可以再聊 — 比如oloid的generator line怎么参数化, Jacobian pseudoinverse的数值稳定性, 或者capacitive sensing的电路细节。

---

# Paper深度解读: Magnetic Endoscope with Developable Oloid Roller

## 1. Core Problem — 磁操纵内窥镜的"Lost DoF"

这篇paper的核心问题非常specific: 在magnetic manipulation的endoscope系统中, **永远只能达到5-DoF**, 而不是完整的6-DoF。原因在物理本质上:

公式(4) $\boldsymbol{\tau} = \mathbf{m_I} \times \mathbf{B_E}$ 给出了IPM受到的torque。这里 $\mathbf{m_I} \in \mathbb{R}^3$ 是IPM的magnetic dipole moment, $\mathbf{B_E} \in \mathbb{R}^3$ 是EPM在IPM位置产生的magnetic field。

关键观察: torque **永远垂直于** $\mathbf{m_I}$, 所以**绕着 $\mathbf{m_I}$ 自身方向(磁化轴)的rotation永远无法由外部磁场产生**。这就是"lost DoF" — 即 **roll about magnetization axis**。

这个limitation对很多应用是致命的, 因为:
- biopsy需要精确的tool orientation
- µUS / OCT / Terahertz这些subsurface sensing modality需要**contact-based scanning**, 即sensor head需要贴着tissue进行rotational sweep
- 没有roll DoF, 就无法扫描到目标区域的不同切片

Paper链接: [IEEE IROS 2025](https://doi.org/10.1109/iros60139.2025.11247414) | [White Rose Repository](https://eprints.whiterose.ac.uk/id/eprint/237138/)

## 2. Key Insight — 用几何形状"被动"恢复roll DoF

最聪明的想法: **不靠额外的actuator去直接产生roll, 而是设计一个特殊的形状, 使得pitch和yaw的coupling运动通过rolling contact在tissue上诱导出roll motion**。这是一种典型的 **underactuated robotics** 思路 — 通过geometry + constraint去recover一个"丢失"的DoF。

类比于:
- **Nonholonomic car**: 不能侧向滑动, 但通过特定的forward+steering组合可以到达任何configuration
- **Falling cat**: 自由下落中cat没有external torque, 但通过deformation改变moment of inertia, 可以reorient
- **Locomotion via shape change**: Amoeba-like robots通过shape change获得net displacement

这里用的是 **developable surface** 的特殊性质。Developable surface是Gaussian curvature为零的surface, 可以无拉伸地展开成平面。这类surface在rolling时, **每条generator line依次与ground接触**, 整个process的center of mass保持恒定高度, 不需要任何额外energy去lift物体。

## 3. Oloid几何详解

### 3.1 Oloid vs Sphericon

Oloid由Paul Schatz在1929年发现, 它是两个**互相垂直的相同圆弧的convex hull**。Sphericon是David Hirsch在1980年发明的related shape, 由两个互相垂直的**半圆**的convex hull形成。

Table I的几何比较:

| Property | Oloid | Sphericon |
|---|---|---|
| Angle of inclination | 30° | 45° |
| Generator length | $\sqrt{3}r$ | $\sqrt{2}r$ |
| Vertices | **0** | 4 |
| Edges | 2 | 2 |
| Internal volume | $3.0524r^3$ | $2.0943r^3$ |
| Length | $3r$ | $2r$ |
| Cross-section | $2r^2$ | $2r^2$ |
| Overall volume (incl. bounding box) | $5.9921r^3$ | $3.9921r^3$ |
| Volume ratio (to bounding box) | 0.509 | 0.525 |

变量定义: $r$ 是两个forming circles的radius。

### 3.2 为什么选Oloid而不是Sphericon

作者给出了三个关键理由:
1. **0 vertices**: Sphericon有4个尖点, 对soft tissue是危险的; Oloid完全光滑(smooth), 适合clinical use
2. **Cross-section相同情况下, oloid内部volume更大**: $3.0524r^3$ vs $2.0943r^3$, 差不多50% more volume。这对**embedding cylindrical components**(magnet core, camera, irrigation tube)是关键优势
3. **Endoscope的rigid length限制**: 临床accept的rigid length是30mm (upper GI) 或60mm (lower GI)。同一个cross-section, oloid更长($3r$), 提供更多internal space

参考: [Oloid - Wikipedia](https://en.wikipedia.org/wiki/Oloid) | [Sphericon - Wikipedia](https://en.wikipedia.org/wiki/Sphericon)

### 3.3 Oloid的参数化与"Oloidicity"度量

Oloid由arc length parameter $t$ 离散化, $t \in [-\frac{2\pi}{3}, \frac{2\pi}{3}]$ (注意这里论文公式(2)中是写成 $\frac{2\pi}{3} \leq t \leq \frac{2\pi}{3}$, 显然是typo, 应该是 $-\frac{2\pi}{3} \leq t \leq \frac{2\pi}{3}$, 总跨度 $\frac{4\pi}{3}$)。

每个 $t$ 对应两个forming circles上的一对points $(P_1(t), P_2(t))$，连线即为generator line。Full oloid的generator length始终是 $\sqrt{3} r$ (这也是为什么它和sphericon $\sqrt{2} r$ 不同)。

**Oloidicity** metric的提出是为了量化"hybrid design偏离完整oloid的程度"。

公式(1): $g(t) = \frac{\text{Length of generator line at } t}{r\sqrt{3}}$

- 完整oloid: $g(t) = 1, \forall t$
- 部分oloid: $0 \leq g(t) < 1$

公式(2): $\text{Oloidicity} = \frac{\int_{-2\pi/3}^{2\pi/3} \int_{-2\pi/3}^{2\pi/3} g(t_1, t_2) \, dt_1 \, dt_2}{(4\pi/3)^2}$

Intuition: 这是把"完整度函数"在generator parameter space上做**积分平均**, 然后用full oloid对应的理想积分做归一化。本质上等价于问: **"如果我们的shape把所有generator line都恢复到full length, 表面积会变成多少倍?"**

注意: 严格说这里是2D integral over $(t_1, t_2)$ plane, 因为hybrid design可能在不同方向有不同的truncation。但full oloid其实只需要1D parameter $t$ (因为两个圆上的点是1-1对应的), 所以这个2D积分有点over-parameterized — 可能是作者为了generalize到更复杂的hybrid形状(两圆的truncation范围不同)而保留的flexibility。

## 4. Magnetic Manipulation数学推导

这部分是paper最数学密集的地方, 我把每一步都展开讲清楚。

### 4.1 Magnetic Force & Torque模型

公式(3)(4):
$$\mathbf{f} = (\mathbf{m_I} \cdot \nabla) \mathbf{B_E}$$
$$\boldsymbol{\tau} = \mathbf{m_I} \times \mathbf{B_E}$$

这里 $\mathbf{B_E}$ 是EPM在IPM处的field。$\mathbf{m_I}$ 是IPM的dipole moment, $\nabla$ 是spatial gradient operator (3D)。

- $\mathbf{f}$: magnetic force, 来自field gradient (dipole在non-uniform field中受力)
- $\boldsymbol{\tau}$: magnetic torque, 来自dipole与field的对齐 (alignment effect, 让dipole moment与field平行)

### 4.2 Robotic Arm Kinematics

KUKA LBR有7-DoF, 关节变量 $\mathbf{q} \in \mathbb{R}^7$。

公式(5): $\begin{bmatrix} \dot{\mathbf{p_E}} \\ \boldsymbol{\omega_E} \end{bmatrix} = J_R(\mathbf{q}) \dot{\mathbf{q}}$

- $\dot{\mathbf{p_E}} \in \mathbb{R}^3$: EPM的linear velocity
- $\boldsymbol{\omega_E} \in \mathbb{R}^3$: EPM的angular velocity
- $J_R(\mathbf{q}) \in \mathbb{R}^{6 \times 7}$: robot的geometric Jacobian (不是kinematic Jacobian, 是standard 6×7 manipulator Jacobian)

### 4.3 EPM Jacobian与磁场对称性

公式(6): $\begin{bmatrix} \dot{\mathbf{p_E}} \\ \dot{\mathbf{m_E}} \end{bmatrix} = \begin{bmatrix} \mathbb{I_3} & \mathbb{O_3} \\ \mathbb{O_3} & S(\hat{\mathbf{m_E}})^T \end{bmatrix} J_R(\mathbf{q}) \dot{\mathbf{q}} = J_E(\mathbf{q}) \dot{\mathbf{q}}$

关键insight: EPM本身是一个permanent magnet, 它的magnetic dipole方向$\hat{\mathbf{m_E}}$ 是EPM坐标系中的一个固定方向。当robot关节运动导致EPM旋转时, $\hat{\mathbf{m_E}}$ 在world frame里也会跟着旋转。

矩阵分析:
- $\mathbb{I_3}$, $\mathbb{O_3}$: 3×3 identity / zero matrix
- $S(\hat{\mathbf{m_E}})^T$: skew-symmetric matrix的transpose

为什么用skew-symmetric? 因为对于任意向量 $\mathbf{v}$, $S(\hat{\mathbf{m}})\mathbf{v} = \hat{\mathbf{m}} \times \mathbf{v}$。这里 $S(\hat{\mathbf{m}})^T \boldsymbol{\omega}$ 给出的是 **绕$\hat{\mathbf{m}}$轴以外的旋转对$\hat{\mathbf{m}}$变化的贡献**。

更具体地, $\dot{\hat{\mathbf{m}}} = \boldsymbol{\omega} \times \hat{\mathbf{m}} = -S(\hat{\mathbf{m}}) \boldsymbol{\omega} = S(\hat{\mathbf{m}})^T \boldsymbol{\omega}$ — 因为 $S^T = -S$ for skew-symmetric。

但这里有个 **subtle点**: 绕$\hat{\mathbf{m}}$本身的旋转**不改变$\hat{\mathbf{m}}$的方向**。所以这一项自动把"lost DoF"排除了 — EPM绕自己磁化轴的旋转不会改变field distribution, 也不会被robot需要做。

### 4.4 Linearized Dipole-Dipole Interaction

公式(7)(8):
$$\begin{bmatrix} \dot{\mathbf{f}} \\ \dot{\boldsymbol{\tau}} \end{bmatrix} = \begin{bmatrix} \frac{\partial \mathbf{F_m}}{\partial \mathbf{p}} & \frac{\partial \mathbf{F_m}}{\partial \hat{m}_E} & \frac{\partial \mathbf{F_m}}{\partial \hat{m}_I} \\ \frac{\partial \boldsymbol{\tau_m}}{\partial \mathbf{p}} & \frac{\partial \boldsymbol{\tau_m}}{\partial \mathbf{m_E}} & \frac{\partial \boldsymbol{\tau_m}}{\partial \mathbf{m_I}} \end{bmatrix} \begin{bmatrix} \dot{\mathbf{p}} \\ \dot{\hat{\mathbf{m}}_E} \\ \dot{\hat{\mathbf{m}}_I} \end{bmatrix} = J_f(\mathbf{p}, \mathbf{m_E}, \mathbf{m_I}) \begin{bmatrix} \dot{\mathbf{p}} \\ \dot{\hat{\mathbf{m}}_E} \\ \dot{\hat{\mathbf{m}}_I} \end{bmatrix}$$

变量: $\mathbf{p} = \mathbf{p_I} - \mathbf{p_E}$ 是IPM相对EPM的position。

这是dipole-dipole force/torque model的**一阶Taylor展开**。背后假设是: 在每个时间步的局部, 系统behavior可以用linear model近似。这种linearization在每个control time step都重新计算一次(assumption保持locally valid)。

公式(9): 假设IPM pose不变, 简化为:
$$\begin{bmatrix} \dot{\mathbf{f}} \\ \dot{\boldsymbol{\tau}} \end{bmatrix} = J_f(\mathbf{p}, \mathbf{m_E}, \mathbf{m_I}) \begin{bmatrix} \dot{\mathbf{p}}_E \\ \dot{\mathbf{m}}_E \end{bmatrix}$$

### 4.5 控制律与Inverse Kinematics

公式(10): $\begin{bmatrix} \dot{\mathbf{p}}_E \\ \dot{\mathbf{m}}_E \end{bmatrix} = J_f^\dagger \, \text{pid}\left(\begin{bmatrix} \dot{\mathbf{f}} \\ \dot{\boldsymbol{\tau}} \end{bmatrix}\right)$

- $J_f^\dagger$: Jacobian的**pseudoinverse** (使用weighted/damped least squares, 参考[Martin et al. Nature Machine Intelligence 2020](https://www.nature.com/articles/s42256-020-00231-9))
- $\text{pid}(\cdot)$: PID controller的输出, based on force/torque error

公式(11): 最终关节速度
$$\dot{\mathbf{q}} = J^\dagger \mathbf{W_a} \begin{bmatrix} \dot{\mathbf{p_E}} \\ \dot{\mathbf{m_E}} \end{bmatrix}$$

- $J^\dagger \in \mathbb{R}^{7 \times 6}$: robot Jacobian的pseudoinverse
- $\mathbf{W_a} \in \mathbb{R}^{6 \times 6}$: weighting matrix (allows prioritization ofposition vs orientation control, 也可加joint limit avoidance)

整个pipeline:
1. User joystick input → desired force/torque
2. PID → target $\dot{\mathbf{f}}, \dot{\boldsymbol{\tau}}$
3. $J_f^\dagger$ → required EPM motion $\dot{\mathbf{p_E}}, \dot{\mathbf{m_E}}$
4. $J^\dagger \mathbf{W_a}$ → joint velocities $\dot{\mathbf{q}} \in \mathbb{R}^7$
5. Robot controller (内部IK) → joint commands

这里有个**redundancy**值得注意: 7-DoF arm控制6-DoF EPM pose, 有1维null space — 可以用于obstacle avoidance或joint limit avoidance (标准redundant manipulator control)。

## 5. Hybrid Design — 在几何与组件集成之间的权衡

### 5.1 Design Constraints

- 整体尺寸: $20 \times 20 \times 40$ mm (与cylindrical MFE [21](https://www.liebertpub.com/doi/abs/10.1089/soro.2023.0182)相当)
- 必须包含: magnet core, localization system, white-light imaging (WLI), insufflation, irrigation, camera cleaning
- Smooth edges, waterproof
- Sensor-agnostic ("hot swappable")
- 最少25° range in roll/yaw/pitch

### 5.2 候选Design对比 (Figure 4)

| Design | Volume Ratio | Roll Range | Oloidicity |
|---|---|---|---|
| Full oloid | 0.50 | max | 1.0 |
| Half oloid (180°) | 0.53 | 180° | <1.0 |
| **Hybrid (oloid + cylinder)** | **0.63** | **180°** | <1.0 |
| Full cylinder | 0.79 | 0° (无roll) | — |

Intuition: 
- **Volume ratio** = shape volume / bounding box volume
- 越接近1, 内部空间utilization越高, 但roll range越受限
- Hybrid是个sweet spot: 还能保留180° roll (远超25°最低要求), 同时volume ratio接近cylinder的60%+ (内部组件, 尤其cylindrical的magnet core, 可以fit进 cylinder part)

另一个关键数据: "为了容纳一个cylindrical component of diameter $d$, full oloid需要 ~10×, half oloid ~5×, hybrid ~3× larger than the cylinder"。Hybrid的优势巨大, 因为endoscope内部主要是cylindrical things (magnet core, camera module, insufflation tube)。

### 5.3 最终OME (Oloid Magnetic Endoscope)

尺寸: **20 × 20 × 35 mm** (note: 长度35mm而非40mm, 说明实际可以做得更紧凑)

组件布局 (Figure 5):
- **Front view (A)**: camera, LED, irrigation, ICC (camera cleaning), insufflation
- **Contact sensor PCB (B)**: 上方中心位置, 平整面积最大
- **Side view (C)**: clip mechanism让top half可hot-swap; magnet core封装在bottom half

制造: FormLabs Form3 resin 3D printer, 三部分打印 (top half + 两个bottom half)。底部用high viscosity cyanoacrylate (Permabond 4c40)密封防水, 通风孔帮助内部空气在密封时排出。

## 6. Contact Sensor

### 6.1 物理设计

- 4 × 10 mm² 面积 (4×10, 不是4mm × 10mm — 是4×10 = 40mm², 但具体长宽我re-read 是 "4 x 10 mm²", 应该理解为4 × 10 mm = 40 mm²的矩形area)
- **8 electrodes in 2×4 pattern**
- Capacitive sensing原理: 电极与tissue接触时, dielectric property变化引起capacitance变化
- Flexible PCB: 10mm wide, 0.1mm thick, 26mm long
- 采样率: **50 Hz**
- 输出: 4个quarters的binary contact/no-contact

参考: [Alagi et al. 2016 Haptics Symposium](https://ieeexplore.ieee.org/document/7463193)

### 6.2 为什么用capacitive而不是resistive或optical

- 体积小 (0.1mm厚度) — 适合endoscope内部空间紧张的应用
- 不需要直接接触力 — 避免损伤tissue
- 容易做flexible PCB — 适应curved surface

## 7. In Vivo Experimental Results

### 7.1 实验设置

- **Porcine model**: 39 kg large white female pig (猪的GI anatomy与人类相似)
- Anesthesia下进行
- UK Home Office license: **PF5151DAF** (合规于Animal (Scientific Procedures) Act 1986)
- Enema后插入, 进入colon约20cm (这段是直的, 适合实验)
- Open-loop joystick控制 (注意: **没有用contact sensor feedback做closed-loop**, 这是个future work方向)

### 7.2 实验1: Sweeping Motion (Figure 6A)

- 上半部分colon表面sweep
- Roll range: **100°**
- 5次重复, 分析了其中2次
- **74% of motion, 所有4个pads都保持contact**
- **单个pad平均contact rate: 92%**

### 7.3 实验2: Rolling Motion (Figure 6B)

- 纯rolling, **±50° (共100°)** 范围
- 5次重复

### 7.4 Contact Pattern Analysis (Figure 7)

| Pad pair | Correlation | Note |
|---|---|---|
| pads 0 & 1 (left side) | **strongest** | 在motion中常同时contact |
| pads 0 & 3 | **weakest** | 对角线最难maintain contact |

Intuition: 0&3是**对角线**关系(top-left ↔ bottom-right), 在rolling时, 这条对角线最先脱离surface。0&1是**横向相邻**, 在rolling过程中可以协同contact。

### 7.5 失败模式分析

74%成功率意味着26%的时间有至少一个pad失去contact。最可能的failure mode:
- **对角线接触**: 当endoscope稍微tilted, 对角线pad容易lift off
- **Colon wall的不平整**: 实际colon不是平面, 有haustral folds
- **Open-loop控制**: 没有contact feedback做correction, 容易drift

## 8. My Intuition & 延伸思考

### 8.1 这个方法本质上是"几何约束 vs 主动控制"的trade-off

传统approach: 加一个motor去直接驱动roll → 增加 complexity, weight, power consumption, failure mode  
Oloid approach: 牺牲一些几何紧凑性 (volume ratio从0.79降到0.63), 换取 **passive roll recovery** via geometry → **zero extra power, zero extra actuator**

这个design philosophy在minimalist robotics里很常见, 比如:
- [HAMR (Harvard Ambulatory MicroRobot)](https://www.seas.harvard.edu/microrobotics/) 用PCB-based的high-frequency actuation
- [Origami robots](https://www.science.org/doi/10.1126/scirobotics.aah3628) 用folding获得 locomotion
- [Tensegrity robots](https://en.wikipedia.org/wiki/Tensegrity_robotics) 用structural properties实现impact resistance

### 8.2 Oloid的"Roto-Translation"性质

Oloid有个非常神奇的几何性质: 当它在一个平面上rolling时, 它做的是 **meandering motion** — 即同时具有pitch和yaw的oscillation, 但center of mass保持恒定height (无vertical bobbing)。这种性质由其 **generator line依次接触surface** 保证。

类比:
- **Sphere**: roll时center是恒定的, 但每个方向的rolling都等价 — 没有方向偏好, 但也不允许"独特"的rolling mode
- **Cylinder**: 一个方向rolling容易, 但跨方向困难
- **Oloid**: pitch + yaw的coupling → roll, 实际上是一种"symmetry breaking" — shape的non-trivial geometry强制了specific motion mode

### 8.3 Developable Surfaces家族

Paper提到oloid属于"developable roller family"。这个family包括:
- **Oloid** (Schatz, 1929)
- **Sphericon** (Hirsch, 1980)
- **Two-circle roller** (more general)
- **Tetrahedric roller** (convex hull of two perpendicular triangles)
- **H-oloid** / 其他变体

这些都有共同性质: developable surface, rolling时center of mass恒高, 但motion path不同。理论上可以设计特定motion trajectory的roller。

参考: [Dirnbock & Stachel, "The Development of the Oloid"](https://www.mathematische-nachrichten.de/en) | [Developable surface - Wikipedia](https://en.wikipedia.org/wiki/Developable_surface)

### 8.4 Underactuation与DiffDrive类比

Oloid-based roll recovery其实是一种 **underactuated locomotion**。DiffDrive robot只有两个轮子motor, 但通过左右轮速度差可以实现rotation — 这是"用已有DoF产生新DoF"。

Oloid case: 已有5-DoF (3 translation + pitch + yaw), 通过geometric coupling产生第6个DoF (roll)。需要 **tissue contact** 作为constraint — 这就是为什么这个方法需要contact-based sensing作为motivation, 它本质上是个 **constraint-mediated locomotion**。

如果endoscope悬空 (no contact), oloid roll机制失效 — 这是这个方法的关键limitation, 也是为什么必须做in vivo而不是只在bench test。

### 8.5 与Limbless Locomotion的联系

Oloid rolling很像limbless locomotion研究里的 **body shape undulation**。Snake robot通过periodic body wave获得forward motion; 这里oloid通过pitch+yaw oscillation获得roll。两者都是 **non-holonomic, geometry-mediated locomotion**。

相关工作:
- [Choset et al. CMU Snake Robots](https://www.cs.cmu.edu/~biorobotics/)
- [Hatton & Choset, "Geometric Motion Planning"](https://www.cs.cmu.edu/~jchasety/)

### 8.6 临床应用前景

Paper最后提到future directions:
1. **µUS / Terahertz / OCT integration** for "virtual biopsy" — 即不取组织就能histology-level诊断, 减少 actual biopsy次数
2. **Endoscopic Submucosal Dissection (ESD)** — 一种高难度therapeutic procedure, 需要precise control, roll recovery是关键
3. **Closed-loop control** with contact feedback — 把74%提到95%+

ESD特别值得讨论: 这是处理early GI cancer的标准方法, 但用传统endoscope难度极高 (学习曲线陡, complication率高)。Magnetic endoscope + roll recovery可以democratize这个procedure, 降低training门槛。

参考: [Norton et al. Science Robotics 2019](https://www.science.org/doi/10.1126/scirobotics.aav7725) | [Martin et al. Nature Machine Intelligence 2020](https://www.nature.com/articles/s42256-020-00231-9)

### 8.7 一些concerns / open questions

1. **Tissue damage**: 100° roll下, endoscope在colon wall上摩擦可能造成mucosal damage。Paper说oloid是smooth的(0 vertices), 但仍未quantify mucosal stress。
2. **Bench test vs In vivo gap**: 实际colon wall是compliant, 不是rigid plane。Oloid rolling的几何假设是平面, 在curved surface上behavior可能不同。
3. **Roll范围限制**: 100°而不是360°意味着不能做完全rotation scan。对一些imaging modality可能不够 (例如µUS往往需要360° rotation)。
4. **Tether vs Untethered**: Paper说也适用于capsule endoscope, 但capsule没有tether, 不能提供 insufflation, irrigation, 也没有camera cable — 实际可搭载的sensor更有限。
5. **Long colon navigation**: 直段20cm可以work, 但colon有弯曲(脾曲、肝曲), 在bend处oloid mechanism行为未知。

### 8.8 数学延伸: Why 30° Inclination?

Table I中oloid的angle of inclination是30°, sphericon是45°。这背后的数学:

Oloid由两个**互相垂直**的圆弧构成, generator line连接圆上对应arc-length的points。当圆是unit circle, arc length $t$ 对应chord angle $\theta = 2t$ on circle, generator line与垂直轴的夹角是 $\arctan(\frac{1}{\sqrt{3}}) = 30°$ — 这正是oloid的inclination angle。

Sphericon由于是半圆 + 半圆垂直, 其几何稍简单: 半圆的弦与对角线呈45°。

这种几何细节其实控制了 **rolling frequency**: angle越小, 每个pitch/yaw cycle产生的roll angle越小, 但更平滑; angle越大, motion更"激进"。30°对endoscope的smoother motion是合适的。

## 9. 总结

| 维度 | 关键take-away |
|---|---|
| **Core problem** | Magnetic manipulation loses roll DoF around magnetization axis |
| **Solution** | Oloid developable roller geometry passively recovers roll via pitch+yaw coupling |
| **Geometry** | 0 vertices (smooth), $3.0524r^3$ internal volume, $3r$ length, 30° inclination |
| **Hybrid design** | Oloid + cylinder, volume ratio 0.63, 180° roll range, 20×20×35mm |
| **Magnetic control** | Dipole-dipole linearization, $J_f^\dagger$ pseudoinverse, 7-DoF KUKA arm |
| **In vivo result** | 100° sweep, 74% contact maintenance, 92% per-pad average |
| **Limitation** | Open-loop, 100° max roll, requires tissue contact, not yet clinically validated |
| **Future** | Closed-loop, µUS/THz integration, ESD application |

这是一篇典型的"用geometry bypass actuation redundancy"的paper。核心insight是: **如果机械设计可以包含differential geometric structure, 就可以用更少的actuator达到相同的task space coverage**。这个思路在robotics里有广泛应用, 但在medical robotics领域用oloid形状做roll recovery是novel的。期待看到他们后续把sensor真正integrate + closed-loop contact control的版本。

---

**Relevant links for further reading:**

- [Paper (IEEE Xplore)](https://doi.org/10.1109/iros60139.2025.11247414)
- [White Rose Repository Preprint](https://eprints.whiterose.ac.uk/id/eprint/237138/)
- [Valdastri Lab at Vanderbilt](https://engineering.vanderbilt.edu/bio/pietro-valdastri)
- [Martin et al. Nature Machine Intelligence 2020](https://www.nature.com/articles/s42256-020-00231-9)
- [Norton et al. Science Robotics 2019](https://www.science.org/doi/10.1126/scirobotics.aav7725)
- [Oloid - Wikipedia](https://en.wikipedia.org/wiki/Oloid)
- [Sphericon - Wikipedia](https://en.wikipedia.org/wiki/Sphericon)
- [Dirnbock & Stachel - Oloid Development paper](https://link.springer.com/article/10.1007/s00006-006-0116-8)
- [Alagi et al. Capacitive Tactile Sensor](https://ieeexplore.ieee.org/document/7463193)
- [Kim et al. Vine Robots Magnetic Actuation](https://www.liebertpub.com/doi/abs/10.1089/soro.2023.0182)
- [Barducci et al. - Gut Fundamentals for Capsule Engineers](https://doi.org/10.1088/2516-1091/abab4c)
- [Taddese et al. - Real-time Pose Estimation](https://doi.org/10.1177/0278364918779132)
