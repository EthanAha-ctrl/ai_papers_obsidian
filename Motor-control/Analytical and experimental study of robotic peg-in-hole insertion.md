---
source_pdf: Analytical and experimental study of robotic peg-in-hole insertion.pdf
paper_sha256: a68890251e48cbcf38c4143550813a49b89ea1064bc2b4d5a8755b558df2e6cf
processed_at: '2026-08-18T00:57:18-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 1989 年的 paper

## 一句话版本

把一个圆柱 peg 插进一个圆柱 hole 里这件事，看起来弱智，其实是工业 assembly 里最经典的"看着容易做着难"问题 — 这篇 paper 说：**你别去买那个贵的 RCC 装置了，把 robot arm 摆到一个特殊的姿态（singular configuration 附近），robot 自己就变"软"了，能起到 RCC 的作用，insertion force 能小 4 倍**。

## 为什么 peg-in-hole 难

想象你拿一根笔要插进一个比笔稍微粗一点的笔筒里。如果你手完全刚性、笔和笔筒位置完全对准，那直接推进去就行。但现实是：

- 你的手会有 **lateral error** $\epsilon_0$（笔尖横向偏了点）
- 你的手会有 **angular error** $\theta_0$（笔歪了点）
- 笔筒和笔之间 **clearance** $R - r$ 很小（比如 0.05 mm）

这三个误差组合起来，如果你硬推，笔就**卡死**（jamming）。卡死的原因是：接触点的 friction force 方向不对，你越推越卡，摩擦把笔"焊"在 hole 里。

## 1970-80 年代的解法：RCC device

Draper Lab 的人（Watson, Drake, Simunovic, Whitney）在 70 年代发明了一个叫 **RCC (Remote Center Compliance)** 的小机械装置 [RCC patent](https://patents.google.com/patent/US4116417A)，装在 robot wrist 上。它的本质就是：让 wrist 在 lateral direction 和 rotational direction 上都"软"（有 compliance），但是在 axial direction 上"硬"。

这样当 peg 接触 chamfer 时，lateral 反力把 peg 自动往中心推；当 peg two-point contact 时，torque 自动把 peg 转正。RCC 像一个"六维弹簧"，自动吸收误差。

问题是：RCC **贵**。大部分工业 robot 不装 RCC，也不用复杂的 force feedback control，就是硬怼。

## 本文的 idea

Yilun Jia 这位 NJIT 的硕士生想：**robot arm 本身的关节就有柔性**（joint 的 gear、bearing、belt 都有微小变形），这些柔性反映到 end-effector 上就是一个 compliance matrix $C$。这个 $C$ 不是固定的，**它随 robot configuration 变化**。

那能不能不装 RCC，直接挑一个 configuration，让 end-effector 在 peg frame 下的 lateral compliance $c_{22}$ 特别大、rotational compliance $c_{33}$ 也大？这样 robot 自己就变成 RCC 了。

答案是可以 — 在 **kinematic singularity** 附近（关节共线、Jacobian 退化），end-effector 在某些 direction 上"软得像棉花"，compliance 趋于无穷大。

## 四个 phase 的故事

论文把 insertion 拆成四个阶段：

### Phase 1: Approach
peg 还没碰到 hole，纯粹位移。

### Phase 2: Chamfer crossing
peg 碰到 hole 入口的 chamfer（那个斜边），沿斜面往下滑。物理上发生的事：chamfer 的斜面给 peg 一个 **lateral force $F_y$**，把 peg 往 hole 中心拉。这一阶段主要消除 **lateral error $\epsilon_0$**。

公式核心：
$$F_x = \frac{(X_h - X_{hI})/\tan\alpha}{-c_{21} + c_{22}/K_f - c_{23}r}$$

- $X_h - X_{hI}$：peg 沿 chamfer 下滑的距离
- $\alpha$：chamfer 倾角
- $K_f = (\cos\alpha + \mu\sin\alpha)/(\sin\alpha - \mu\cos\alpha)$：chamfer 的力传动比，friction $\mu$ 越大 $K_f$ 越大（chamfer 自锁）
- $c_{22}$：lateral compliance，**越大越好** — 这样单位 $F_y$ 能让 peg 横移更多，更容易吸 error
- $c_{23}, c_{21}$：cross-coupling terms，普通 robot 这些不为零，RCC 设为零

### Phase 3: One-point contact
peg 已经滑过 chamfer，进入 hole 内部，**只有一角接触 hole 的 rim**。此时 peg 同时在 translate 和 rotate。

几何关系：
$$\delta Y = \epsilon_0 - cR + \ell\theta$$

- $\epsilon_0$：初始 lateral error
- $cR = R - r$：clearance
- $\ell$：当前 insertion depth
- $\theta$：当前 peg tilt

直觉：peg tip 当前位置 = 初始偏移 - clearance（被 hole wall 限制的允许位移）+ depth×tilt（越深 angular error 累积越大）。

force 公式（case 1）：
$$F_x = \frac{cR - \epsilon_0 - \ell\theta_0}{[(2c_{23} - c_{33}\ell)\ell - c_{22}]/\mu + c_{21} + c_{32}r - c_{31}\ell - c_{33}\ell r}$$

注意分母里的 $c_{33}\ell$ 项 — **$\ell$ 越大，分母绝对值越大，$F_x$ 越大**。也就是说：**peg 越插越深，one-point contact 阶段的 force 上升越快**。

### Phase 4: Two-point contact
peg 的两端同时接触 hole 内壁。这一阶段 peg 几何上被卡住，必须靠 **torque** 把 angular error $\theta_0$ 转正。

torque 平衡：
$$M_z = \lambda r F_x - \mu r(1+\lambda)F_y, \quad \lambda = \frac{\ell}{2\mu r}$$

- $\lambda$：depth-to-friction-length ratio，$\lambda$ 大意味着 peg 已经插入很深
- 物理直觉：$\lambda$ 大时，两 contact point 之间 leverage 长，但是 torque 需求也大

**Two-point contact 是最可怕的阶段**，force 急剧上升。论文 Fig. 4.2 里能清楚看到：one-point 阶段 force 平缓，一到 two-point 阶段（$L = 16$ mm）force 垂直跳上去。

## Critical point $\ell_{12}$：为什么越大越好

从 one-point 到 two-point 的转换深度叫 **critical point** $\ell_{12}$，是 quadratic equation 的根：
$$\zeta\ell^2 + \eta\ell + \xi = 0$$

$\zeta, \eta, \xi$ 都是 $c_{ij}, \epsilon_0, \theta_0, cR, \mu, r$ 的函数。

**$\ell_{12}$ 越大，one-point contact 阶段持续越久，insertion force 一直保持小值**。一旦 $\ell_{12}$ 小，很快就进入 two-point 阶段，force 爆炸。

所以设计 insertion 的核心目标：**让 $\ell_{12}$ 尽量大**。

## Jamming 的物理直觉

Jamming 就是 peg 卡死。Whitney 给的判据：在 $(F_x, F_y, M_z)$ force space 里画一个 parallelogram，force combination 落在 parallelogram 内就不 jamming，落外面就 jamming。

但这个判据工程上没用 — 你又不能直接控制 force combination。本文的 contribution 是把这个 force-space 判据转换成 parameter-space 判据：
$$AF_x \ge B$$

$A, B$ 都是用 $c_{ij}, \epsilon_0, \theta_0, cR, \mu, r, \ell$ 算出来的常数。这样工程师就能直接根据 robot configuration、几何参数、误差来算"我推多大力才不会卡死"。

## 实验结果：四个 configuration 对比

Yilun 在 IBM 7540 SCARA robot 上测了四种 configuration：

| Config | $\theta_2$ | $\theta_3$ | $c_{22}$ (m/N) | 最大 $F_x$ |
|--------|-----------|-----------|----------------|------------|
| 1 | 0° | 0° | 1.2877e-4 | 最小 |
| 2 | 0° | -90° | 8.1108e-5 | **4× Config 1** |
| 3 | 45° | -45° | 1.0660e-4 | 中等 |
| 4 | 90° | 0° | 9.9990e-5 | 中等 |

**Config 1 的 $\theta_2 = \theta_3 = 0$** 意味着 SCARA arm 的两个关节都"伸直了"，这是 **singular configuration**。此时 end-effector 在 lateral direction 上几乎没有任何 resistance（关节共线，一推就横向滑），compliance $c_{22}$ 最大。

结果就是 Config 1 的 insertion force 比 Config 2 小 4 倍。

这就是本文最 punchy 的 finding：**你不用花一分钱买 RCC，只要把 robot 摆对了姿态**。

## 参数敏感性

论文还测了几个参数的影响：

### Clearance $R - r$
从 Table 3：
- $R-r = 0$（零 clearance）：$F_{x,max} = 348$ oz（98 N），$\ell_{12} = 0$（一进 hole 就 two-point）
- $R-r = 0.12$ mm：$F_{x,max} = 21.7$ oz（6 N），$\ell_{12} = 28.7$ mm

clearance 翻 0.12 mm 这么一点，force 降 16 倍。这就是 precision assembly 的本质矛盾 — tight clearance 加工贵且装配难。

### Angular error $\theta_0$
从 Table 4：
- $\theta_0 = 0$：$F_{x,max} = 62$ oz
- $\theta_0 = 1°$：$F_{x,max} = 419$ oz（7 倍）
- $\theta_0 = 2°$：$F_{x,max} = 880$ oz（14 倍）

**angular error 是最致命的**。因为 angular error 让 peg 一开始就歪，$\ell_{12}$ 急剧变小（从 12 mm 降到 2 mm），立刻进入 two-point contact 阶段。

### Lateral error $\epsilon_0$
从 Table 5：
- $\epsilon_0 = 0.25$ mm：$\ell_{12} = 47.6$ mm > 35 mm 总深度，**永远不 two-point contact**
- $\epsilon_0 = 1$ mm：$F_{x,max} = 62$ oz
- $\epsilon_0 = 2$ mm：$F_{x,max} = 153$ oz（2.5 倍）

lateral error 相对 friendly，因为 chamfer crossing 阶段就能把它吸收掉。**$\epsilon_0$ 只要小于 chamfer width $W$，就能被 chamfer geometrically 纠正**。

## Peg radius $r$ 不重要？

这是反直觉的发现：peg 半径从 2.35 mm 到 18.35 mm 变化，$\ell_{12}$ 和 $F_{x,max}$ 几乎不变（Table 2）。

直觉解释：peg 越粗，contact point 离 peg axis 越远，单位 $F_y$ 产生的 moment 越大；但同时 unit moment 产生的 angular deflection 也按比例变化。两者 scale 一致，所以 cancel out。

不过这个结论只在 rigid body 假设下成立。如果 peg 是细长杆，自己会弯曲 buckling，size 就关键了 — 这是论文没覆盖的。

## 这篇 paper 对今天还有什么用

### 1. Singular configuration exploitation
这是本文最 valuable 的 insight：**Jacobian 退化处，end-effector 自动变软**。这后来在 variable stiffness actuation [Vanderborght 2013](https://ieeexplore.ieee.org/document/6600745) 和 Cartesian impedance control [Hogan 1985](https://www.sciencedirect.com/science/article/pii/0022243X85900099) 里都有 follow-up。

### 2. Phase-based reasoning
把复杂 contact 问题按 phase 分解（chamfer / one-point / two-point），每个 phase 有自己的几何和力学，这种思路今天在 [model-based RL for assembly](https://arxiv.org/abs/2012.01993) 里仍然有启发：分别给每个 phase 设计 subtask reward。

### 3. Analytical prior for learning
今天的 insertion 多用 force-torque feedback + RL [OpenAI Rubik's cube](https://openai.com/research/solving-rubiks-cube)，但纯 learning sample inefficient。本文这类 analytical model 可以做 **model-based prior**，让 RL 只学 residual — 类似 [residual policy learning](https://arxiv.org/abs/1812.06298) 的思路。

### 4. Software-defined RCC
当年要装一个机械 RCC，今天可以装一个 F/T sensor + 跑 admittance control loop [ ATI F/T sensor](https://www.ati-ia.com/products-ft/ft-sensors.aspx)，软件里实现任意 compliance matrix $C$。相当于把硬件 RCC 升级为 software-defined，而且可以 configuration-dependent、task-dependent 动态切换。

## 一些思考

这篇 paper 让我想到一个更广的 principle：**很多看起来需要硬件解决的问题，其实可以靠软件+几何 insight 绕过去**。

RCC device 本质上是一个被动的"智能弹簧" — 它通过机械结构实现了 force-deflection 的特定解耦关系。Jia 这篇 paper 说：等等，robot arm 本身的 joint flexibility 就构成了一个 compliance matrix，只要我挑对 configuration，就能 mimic RCC 的行为。

这个思路在 robotics 里反复出现：
- [Mimic RCC](https://ieeexplore.ieee.org/document/126785)：用被动机械实现 force guidance
- [Active compliance](https://link.springer.com/article/10.1007/BF00430691)：用 force sensor + position control 主动实现 compliance
- [Learning compliance](https://arxiv.org/abs/2007.04637)：用 RL 学到 task-appropriate compliance

每一代人都用自己时代的工具重新发现这个 insight。Jia 用的是 analytical mechanics + IBM 7540 SCARA robot，今天我们用 GPU + RL，但底层物理是一样的：**contact、friction、geometry、compliance** 这四样东西，决定了所有 assembly task 的命运。

## 一句话总结

这篇 1989 年的硕士论文告诉我们：**robotic assembly 不一定要花大钱买精密装置，理解了 compliance、geometry、friction 之间的耦合关系，挑对 robot 姿态，用 robot 本身的"软"就能解决大部分 insertion 问题**。这种 "intelligence over hardware" 的思路，永远有用。

参考资源：
- Whitney 1982 经典 paper: https://doi.org/10.1115/1.3139694
- RCC patent: https://patents.google.com/patent/US4116417A  
- IBM SCARA robot 历史: https://www.computerhistory.org/revolution/artificial-intelligence-robotics/6/328
- Modern force-guided assembly 综述: https://ieeexplore.ieee.org/document/126785
- Variable Stiffness Actuation: https://ieeexplore.ieee.org/document/6600745
- Hogan impedance control: https://www.sciencedirect.com/science/article/pii/0022243X85900099

---

# Robotic Peg-in-hole Insertion 论文详解

## 1. 论文背景与动机

这篇是 Yilun Jia 在 1989 年完成的硕士论文，导师是 NJIT (New Jersey Institute of Technology) 的 Ming C. Leu 教授。核心 idea 是研究 **robotic peg-in-hole insertion** 中的 insertion force 与 moment 如何受到 manipulator compliance 的影响，关键 insight 是 **不依赖 RCC (Remote Center Compliance) device**，仅靠选择合适的 manipulator configuration 来改善 insertion 条件。

Peg-in-hole insertion 是工业 assembly 的基础操作，涵盖 plug insertion、pin insertion、force fits、fastener installation、IC assembly 等场景。这项工作可以追溯到 1970 年代 Draper Laboratory 的研究 [Whitney 1982](https://asmedigitalcollection.asme.org/dscsystems/article-abstract/104/1/65/413358), 以及 Watson、Drake、Simunovic 发明的 RCC device [Drake 1977](https://www.sciencedirect.com/science/article/pii/0368192381900465)。

作者观察到一个实际事实：**大部分工业 robot 既不使用 RCC device 也不使用 feedback control algorithm**，原因在于 cost。于是研究路径是：能不能利用 manipulator 本身的 mechanical compliance（来源于 joint compliance，且随 configuration 变化）来辅助 insertion。

## 2. 核心物理 insight

### 2.1 四个 phase

Insertion 过程分为四个 phase：
1. **Approach** — peg 接近 hole，尚无接触
2. **Chamfer crossing** — peg 接触 chamfer，沿 chamfer 斜面下滑，**lateral error ε₀ 被几何地吸收**
3. **One-point contact** — peg 接触 hole rim 的单一接触点，peg 同时 translate 与 rotate
4. **Two-point contact** — peg 接触 hole 内对侧壁，形成两个接触点，**angular error θ₀ 通过 torque 被吸收**

关键 insight：chamfer crossing 和 one-point contact 主要修正 lateral error；two-point contact 主要修正 angular error。这正好对应 RCC 的设计原理 — 不同 phase 需要不同的 compliance 方向。

### 2.2 两种 failure mode

- **Wedging**：peg 被 contact force 压紧，无法退出也无法前进，与 part deformability 有关。本文假设 rigid body，不讨论 wedging。
- **Jamming**：insertion force vector 偏离 hole axis 太远，contact reaction force 无法将 peg 转正，peg 卡死。

Whitney 的经典 jamming 判据（以 force/torque 表达）：
$$\mu |F_y| \le |F_x|$$
$$|M_z/r + \mu(1+\lambda) F_y| \le \lambda |F_x|$$

其中 $\mu$ 是 friction coefficient，$r$ 是 peg radius，$\lambda = \ell/(2\mu r)$，$\ell$ 是 insertion depth。这两个不等式在 $(F_x, F_y, M_z)$ 空间中形成一个 parallelogram 的"safe zone"（Fig. 2.7），任何位于 parallelogram 内的 force combination 保证不 jamming。

本文的 **创新点** 是将这个抽象的 force-space 判据转换成包含 insertion parameters（depth、compliance、几何、error）的判据，使得工程师可以直接根据可观测参数判断 jamming risk。

## 3. 数学模型详解

### 3.1 关键变量约定

- $(X, Y, Z)$：peg coordinate frame，原点在 peg tip face 的几何中心
- $(X_h, Y_h, Z_h)$：hole coordinate frame
- $\epsilon_0$：初始 **lateral error**（peg tip 与 hole axis 的横向偏移）
- $\theta_0$：初始 **angular error**（peg axis 与 hole axis 的夹角）
- $r$：peg radius
- $R$：hole radius
- $c = (R-r)/R$：clearance ratio
- $\alpha$：chamfer angle
- $W$：chamfer width
- $\mu$：friction coefficient
- $\ell$：insertion depth（从 contact point 量到 peg tip 的距离）
- $F_x, F_y, M_z$：robot actuator 提供的 insertion force（沿 Z 方向的轴向力记作 $F_x$ 在本论文中作为主驱动力，实际是 insertion direction）、lateral force 与 moment
- $\delta S = [\delta X, \delta Y, \delta\theta]^T$：peg tip 的 deflection vector
- $C$：3×3 symmetric **effective compliance matrix**，描述 manipulator 在 peg frame 下的柔性

注意：本文用 $F_x$ 表示 insertion direction 的力（即沿 hole axis 的推进力），这是 planar model 中 X 方向约定，与传统 Z 方向 insertion 略有不同 — 在 Fig. 2.2 中可以看到 X 方向沿 hole axis。

### 3.2 Compliance matrix 与 deflection 关系

核心方程（10a）：
$$\begin{bmatrix} \delta X \\ \delta Y \\ \delta\theta \end{bmatrix} = -\begin{bmatrix} c_{11} & c_{12} & c_{13} \\ c_{21} & c_{22} & c_{23} \\ c_{31} & c_{32} & c_{33} \end{bmatrix}\begin{bmatrix} F_x \\ F_y \\ M_z \end{bmatrix}$$

- $c_{11}$：unit $F_x$ 产生的 $\delta X$（轴向 compliance）
- $c_{22}$：unit $F_y$ 产生的 $\delta Y$（lateral compliance）— RCC 设计就是要让 $c_{22}$ 大
- $c_{33}$：unit $M_z$ 产生的 $\delta\theta$（rotational compliance）— two-point contact 需要这个大
- $c_{23} = c_{32}$：force/moment 耦合项，**是 RCC vs. 普通 robot 的核心区别**
- $c_{12}, c_{13}, c_{31}$ 等：cross-coupling terms，普通 manipulator 不为零，RCC device 设计成这些为零

**关键 insight**：RCC device 的 compliance matrix 在 peg frame 下是 diagonal（解耦），普通 manipulator 是 full symmetric matrix，所有 $c_{ij}$ 都不为零，导致 $F_x$ 推进时会产生 $\delta Y$ 偏移、产生 $\delta\theta$ 倾斜，反之亦然。这种耦合**有时候是好事，有时候是坏事**，取决于 configuration。

### 3.3 Chamfer crossing 的力学

接触 chamfer 时，normal force $f_N$ 分解为 $f_x$（轴向）和 $f_y$（侧向）：

$$f_x = f_N(\cos\alpha + \mu\sin\alpha)$$
$$f_y = f_N(\sin\alpha - \mu\cos\alpha)$$

其中 $\cos\alpha + \mu\sin\alpha$ 来自 normal+friction 在 X 方向的投影，$\sin\alpha - \mu\cos\alpha$ 是 Y 方向投影（friction 方向与运动方向相反，故取负号）。

由此推出 insertion force 间的耦合常数：
$$K_f = \frac{\cos\alpha + \mu\sin\alpha}{\sin\alpha - \mu\cos\alpha}$$

物理意义：$K_f$ 是 chamfer 上"轴向推力 vs. 侧向反力"的传动比。$\mu$ 增大时 $K_f$ 增大，意味着 chamfer 自锁效应增强，需要更大的 $F_x$ 才能产生足够的 $F_y$ 把 peg 拉入 chamfer 中心。

最终 insertion force（公式 11）：
$$F_x = \frac{(X_h - X_{hI})/\tan\alpha}{-c_{21} + c_{22}/K_f - c_{23} r}$$

- 分子 $(X_h - X_{hI})/\tan\alpha$ 是 peg 沿 chamfer 下滑对应的 lateral deflection
- 分母 $-c_{21} + c_{22}/K_f - c_{23} r$ 是 effective lateral compliance 在 chamfer constraint 下的修正

**直觉**：分母越大，单位 $F_x$ 产生的 deflection 越大，所需 $F_x$ 越小。所以选 manipulator configuration 使 $c_{22}/K_f - c_{23} r$ 大、$c_{21}$ 小是 chamfer crossing 阶段的设计目标。

### 3.4 One-point contact (Case 1) 力学

Fig. 2.4a 的几何：peg 上角接触 hole 近侧 rim。从 force balance：

$$F_x = f_{1x}, \quad F_y = -f_{1y}, \quad M_z = f_{1x} r + f_{1y}\ell$$

friction constraint $f_{1x}/f_{1y} = \mu$ 给出：
$$F_y = -F_x/\mu, \quad M_z = F_x r - F_y\ell$$

lateral deflection（公式 17）：
$$\delta Y = \epsilon_0 - cR + \ell\theta$$

物理意义：peg tip 当前 lateral position = 初始 error $\epsilon_0$ - clearance $cR$（向中心一侧的允许偏移）+ $\ell\theta$（angular error 引入的、随 depth 增大的额外偏移）。

通过消元最终得到 $F_x$（公式 22）：
$$F_x = \frac{cR - \epsilon_0 - \ell\theta_0}{[(2c_{23} - c_{33}\ell)\ell - c_{22}]/\mu + c_{21} + c_{32}r - c_{31}\ell - c_{33}\ell r}$$

**分母是 compliance 与几何的复杂耦合项**。关键观察：分母里 $c_{31}\ell$ 与 $c_{33}\ell r$ 随 $\ell$ 线性增长 — 也就是说，**当 peg 插入越深，one-point contact 阶段的 effective stiffness 越大，force 上升越快**。这就是为什么 two-point contact 越晚发生越好（critical point $\ell_{12}$ 越大越好）。

### 3.5 Two-point contact 力学

Fig. 2.5：peg 上角与下角同时接触 hole 内壁。两个 contact point，分别距 tip 为 $\ell$ 与 0，peg radius 为 $r$。

Force balance：
$$F_x = f_{1x} + f_{2x}$$
$$F_y = -f_{1y} + f_{2y}$$
$$M_z = f_{1x} r + f_{1y}\ell - f_{2x} r$$

通过 friction constraint 与几何约束解出（公式 29）：
$$M_z = \lambda r F_x - \mu r(1+\lambda) F_y, \quad \lambda = \frac{\ell}{2\mu r}$$

**直觉**：$\lambda$ 是 depth-to-friction-length ratio。$\lambda$ 大意味着 peg 已经插入很深，两接触点之间 leverage 长，单位 $F_y$ 产生的 corrective torque 大 — two-point contact 在深度大时反而更容易 correct angular error。但 $\lambda$ 大也意味着 $M_z$ 的力臂长，需要 $M_z$ 更大才能转动 peg。

Deflection 约束：
$$\delta Y = \epsilon_0 + cR, \quad \delta\theta = 2cR/\ell - \theta_0$$

物理意义：two-point contact 强制 peg lateral position 等于 $\epsilon_0 + cR$（即贴住一侧 hole wall），peg 的 tilt 被几何强制为 $2cR/\ell$（两端必须同时接触），初始 angular error $\theta_0$ 在 deflection 中体现为 $\delta\theta$。当 $\ell$ 大时 $2cR/\ell$ 很小，peg 几乎平行 hole axis，这符合直觉。

最终 $F_x, F_y$ 由公式 (33), (34) 给出，分母是关于 $c_{ij}, \lambda, \mu, r$ 的复杂 determinant — 本质是 $2\times2$ linear system 的解。

### 3.6 Critical point of phase transition

One-point contact 与 two-point contact 之间的转换点 $\ell_{12}$ 通过求解 quadratic equation (37)：
$$\zeta\ell^2 + \eta\ell + \xi = 0$$

其中 $\zeta, \eta, \xi$ 都是 $c_{ij}, \epsilon_0, \theta_0, cR, \mu, r$ 的函数。求解给出 $\ell_{12}$（one→two 转换深度）和 $\ell_{21}$（two→one 反向转换深度，通常意味着 peg 已经深到 $2cR/\ell$ 不足以维持两接触点）。

**直觉**：$\ell_{12}$ 越大，one-point contact 持续越久，insertion force 在更长时间内保持小值；$\ell_{12}$ 越小，越早进入 two-point contact 阶段，force 急剧上升。所以 **增大 $\ell_{12}$ 是设计目标**。

## 4. Jamming avoidance 的修正判据

Whitney 原始判据在 force space 中，但工程上需要参数化判据。本文推导出（公式 52）：
$$AF_x \ge B$$

$A$ 与 $B$ 是关于 compliance $c_{ij}$、几何 $cR, r$、initial error $\epsilon_0, \theta_0$、friction $\mu$、depth $\ell$ 的复杂函数。这个判据的意义：**给定一组 insertion 参数，可以直接计算所需的最小 $F_x$ 以避免 jamming**。

更深的 insight：由于 $A, B$ 都依赖 $c_{ij}$，**调整 manipulator configuration 改变 $c_{ij}$，可以在给定 $F_x$ 容量下扩大 safe insertion parameter region**。

## 5. 实验 setup

### 5.1 主要硬件

- **IBM 7540 robot**：SCARA-type 3-DOF arm，AML/Entry 编程 [IBM 7540 manual](https://www.scribd.com/document/IBM-7540-Manual)（这类 retro 文档现已罕见）
- **LORD F/T 75/250 Force/Torque Sensing System**：6-axis F/T sensor，range ±75 lb / ±250 lb-in，输出 raw strain gauge 经 preprocessor A/D 转换后由 controller 转为 Cartesian force/torque [Lord F/T manual](https://www.ati-ia.com/products-ft/ft-sensors.aspx) — Lord 的 F/T 业务后被 ATI Industrial Automation 收购
- **Keyence LC-2010 laser displacement meter**：4 mm range, 0.001 mm precision [Keyence laser displacement](https://www.keyence.com/products/measure/laser-1d/) — 这种 micron 级 displacement 测量是 compliance 测量的关键
- **Slot set 与 working table**：slot 而非 round hole 设计，便于 peg 在 slot 内自由滑动设定 error；working table 提供 2 translation + 1 rotation，用于对准与设置 error

### 5.2 Compliance measurement 方法

将 laser sensor 置于 peg tip 附近，通过 working table 手轮施加 $F_y$，测量产生的 $\Delta Y_1$，得到 $c_{22} = \Delta Y_1 / F_y$。例如论文给出 $F_y = 40$ oz = 11.121 N，$\Delta Y_1 = 1.112$ mm，得：
$$c_{22} = \frac{1.112 \times 10^{-3}}{11.121} = 9.9990 \times 10^{-5} \text{ m/N}$$

测 $c_{33}$ 时，在 peg tip 加一根 rod，施加 $F_y$ 产生 moment $L_1 F_y$，再用已知的 $c_{32}$ 反推 $c_{33}$。这种"叠加施加"的间接测量是 compliance 测量的标准技巧。

### 5.3 测量得到的 compliance matrix (Table 1)

四种 configuration 下测得的 compliance 元素值（部分摘录）：

| Config | $\theta_2$ | $\theta_3$ | $c_{22}$ (m/N) | $c_{32}$ (1/N) | $c_{33}$ (1/Nm) |
|--------|-----------|-----------|----------------|----------------|-----------------|
| 1 | 0° | 0° | 1.2877e-4 | 8.254e-4 | 8.078e-3 |
| 2 | 0° | -90° | 8.1108e-5 | 6.5046e-4 | 6.699e-3 |
| 3 | 45° | -45° | 1.0660e-4 | 8.1313e-4 | 8.1346e-3 |
| 4 | 90° | 0° | 9.9990e-5 | 8.0130e-4 | 7.5320e-3 |

**关键观察**：Config 1 ($\theta_2 = \theta_3 = 0$) 的 $c_{22}$ 最大 (1.2877e-4)，意味着 lateral compliance 最大 — 这是 **singular configuration** 附近的特性，关节共线使 end-effector 在 lateral direction 上最"软"。

## 6. 实验结果与 insight

### 6.1 Manipulator configuration 的影响 (Section 4.1)

四种 configuration 的 insertion force 实验与理论曲线对比如 Fig. 4.2-4.5。Fig. 4.6 综合显示：

- **Config 2 ($\theta_2=0, \theta_3=-90°$)**：最大 $F_x$ 约为 Config 1 的 **4 倍**
- **Config 1 ($\theta_2=\theta_3=0$)**：force 最小，且 $\ell_{12}$ 最大
- 理论预测与实验曲线高度吻合

**核心 insight**：在接近 **singular configuration**（关节共线，Jacobian 退化）附近作业，end-effector 的 lateral compliance 自动增大，**robot 本身就成了一个天然的 RCC**。这是本文最重要的实践贡献。

这一发现后来被很多研究 follow up：[Schimmels & Peshkin 1992](https://ieeexplore.ieee.org/document/126785) 的 "Admittance Matrix design for force-guided assembly" 进一步推广了通过 configuration selection 实现 artificial compliance 的方法。

### 6.2 Peg 尺寸的影响 (Section 4.2)

实验与理论都表明：**peg 半径 $r$ 对 insertion force 几乎无影响**（Fig. 4.8, 4.9，曲线接近水平）。

直觉解释：force 公式中 $r$ 同时出现在分子（如 $M_z = F_x r$）与分母（如 compliance coupling），相互抵消。物理上：peg 越粗，contact point 离 peg axis 越远，但 unit $F_y$ 产生的 angular deflection 也按比例变化，两者 scale 量一致。

不过注意：这个结论仅在 **rigid body assumption** 下成立。如果 peg 本身有 compliance（长细比大），peg 像柱一样 buckling，size 会显著影响 — 这在 [Whitney 1982](https://asmedigitalcollection.asme.org/dscsystems/article-abstract/104/1/65/413358) 中有讨论。

### 6.3 Clearance 的影响 (Section 4.3)

Fig. 4.10 与 Fig. 4.11-4.12 的数据（Table 3）：

| $R-r$ (mm) | $\ell_{12}$ (mm) | $F_{x,max}$ (oz) | $M_{z,max}$ (oz-in) |
|------------|------------------|-------------------|---------------------|
| 0.00 | 0.0 | 348.1 | 763.2 |
| 0.03 | 7.4 | 91.9 | 624.6 |
| 0.06 | 14.7 | 57.2 | 486.0 |
| 0.09 | 21.7 | 39.0 | 347.4 |
| 0.12 | 28.7 | 21.7 | 208.7 |

**Clearance 越大，$\ell_{12}$ 越大，$F_{x,max}$ 越小**。$c=0$ 时 $\ell_{12}=0$ — 即 peg 一进入 hole 就立刻 two-point contact，force 爆炸式上升至 348 oz（约 98 N）。

这是经典的 precision assembly 难题：tight clearance 制造精度高但装配困难。工业上常用 [selective assembly](https://en.wikipedia.org/wiki/Selective_assembly) — 通过测量分组匹配 peg-hole pair 来人为放大 effective clearance。

### 6.4 Angular error 的影响 (Section 4.4.1)

Table 4 数据：

| $\theta_0$ (deg) | $\ell_{12}$ (mm) | $F_{x,max}$ (oz) |
|------------------|-------------------|-------------------|
| 0.00 | 12.3 | 62.2 |
| 0.25 | 8.0 | 136.8 |
| 0.50 | 5.9 | 221.6 |
| 1.00 | 3.9 | 418.7 |
| 1.50 | 2.9 | 640.7 |
| 2.00 | 2.3 | 879.9 |

**Angular error 对 $\ell_{12}$ 与 $F_x$ 的影响是高度非线性的**。从 0° 到 0.25°，$\ell_{12}$ 减少 35%，但 $F_x$ 翻倍以上；从 0° 到 2°，$F_x$ 增大 14 倍。

直觉：angular error 直接决定 two-point contact 发生得多早，$\ell_{12} \propto 1/\theta_0$ 近似成立。$\theta_0$ 是最致命的 error 形式。

### 6.5 Lateral error 的影响 (Section 4.4.2)

Table 5 数据：

| $\epsilon_0$ (mm) | $\ell_{12}$ (mm) | $F_{x,max}$ (oz) |
|-------------------|-------------------|-------------------|
| 0.25 | 47.6 | 1.3 |
| 0.50 | 24.0 | 16.6 |
| 1.00 | 12.3 | 62.2 |
| 1.50 | 8.3 | 107.9 |
| 2.00 | 6.2 | 153.5 |

$\epsilon_0$ 的影响相比 $\theta_0$ 更线性。但 $\epsilon_0 = 0.25$ mm 时 $\ell_{12} = 47.6$ mm > 总 insertion depth 35 mm，意味着 **永远不进入 two-point contact**，insertion 极轻 — 这就是 chamfer 设计的精髓。

## 7. 综合 insight 与现代联系

### 7.1 Singular configuration 的 exploitation

本文最重要的 insight 是 **在 manipulator kinematic singularity 附近作业，end-effector compliance 自动放大**。这相当于利用 Jacobian 退化（$\det(J) \to 0$）使得 Cartesian stiffness $K = J^{-T} K_j J^{-1} \to 0$（$K_j$ 是 joint stiffness）。

这一思路在后来发展成两个方向：
1. **Variable Stiffness Actuation (VSA)**：通过 antagonistic tendon 或气动 muscle 主动调节 joint stiffness [Vanderborght et al. 2013](https://ieeexplore.ieee.org/document/6600745)
2. **Cartesian impedance control**：通过 Jacobian 在线调节 Cartesian compliance [Hogan 1985](https://www.sciencedirect.com/science/article/pii/0022243X85900099)

### 7.2 与 modern learning-based assembly 的联系

今天 industrial robot insertion 多采用 RL + force-torque feedback，例如 [OpenAI Rubik's cube](https://openai.com/research/solving-rubiks-cube) 与 [NVIDIA Isaac Lab](https://developer.nvidia.com/isaac/sim) 的 insertion tasks。但本文揭示的几个 principle 仍然适用：

1. **Singular configuration 附近 force 容易发散** — modern controller 同样需要考虑 manipulability 退化
2. **Angular error 比 lateral error 难纠正** — RL policy 通常对 angular perturbation 更敏感
3. **Chamfer + compliance 的 phase-based geometry** 仍可启发 reward shaping：分别对 chamfer crossing、one-point、two-point 三个 phase 设计 subtask reward

### 7.3 与 Whitney 经典工作的关系

[Whitney 1982](https://asmedigitalcollection.asme.org/dscsystems/article-abstract/104/1/65/413358) 给出了 RCC device 的完整设计与 jamming 理论，但 RCC device 是 **附加在 wrist 上的机械装置**，compliance 在 peg frame 下 diagonal。本文去掉了 RCC，让 compliance 来自 manipulator joint 本身，导致 $C$ matrix 在 peg frame 下 full symmetric，分析必须处理 cross-coupling $c_{23}, c_{31}$ 等 — 这是本文的技术增量。

Modern implementation 可以用 [force-torque sensor + admittance control](https://ieeexplore.ieee.org/document/8793788) 主动模拟任意 $C$ matrix，相当于 software-defined RCC。

## 8. 论文局限与延伸

1. **Rigid body 假设**：peg、hole、gripper 都假设为 rigid。若 peg 细长（如 IC 引脚），peg 自身弯曲会显著改变 phase transition — 见 [De Fazio & Whitney 1987](https://www.sciencedirect.com/science/article/pii/0007850687900252)。
2. **Planar model**：3D insertion 有 $\epsilon_x, \epsilon_y, \theta_x, \theta_y$ 四个 error，本文只考虑一个 $\epsilon_0$ 与一个 $\theta_0$。3D 情形需要 6×6 compliance matrix [Whitney 1982 extended](https://link.springer.com/chapter/10.1007/978-1-4615-9745-3_2)。
3. **Static 分析**：忽略 dynamics，假设 insertion speed 恒定且惯性可忽略。高速 insertion 时 [vibration-aided assembly](https://www.sciencedirect.com/science/article/pii/S0020750397000913) 能显著降摩擦，本文最后一段也提及。
4. **Single insertion direction**：未讨论 [peg transfer](https://ieeexplore.ieee.org/document/8793567)（peg 在 hole 间转移）或 [multiple peg-in-hole](https://link.springer.com/article/10.1007/s10514-020-09920-y) 等扩展场景。

## 9. 关键公式 cheat sheet

| Phase | $F_x$ formula | 关键 compliance 项 |
|-------|---------------|---------------------|
| Chamfer crossing | $F_x = \frac{(X_h-X_{hI})/\tan\alpha}{-c_{21} + c_{22}/K_f - c_{23}r}$ | $c_{22}$ 越大越好 |
| One-point (case 1) | $F_x = \frac{cR - \epsilon_0 - \ell\theta_0}{[(2c_{23}-c_{33}\ell)\ell - c_{22}]/\mu + c_{21} + c_{32}r - c_{31}\ell - c_{33}\ell r}$ | $c_{33}$ 在 $\ell$ 大时主导 |
| Two-point | $F_x = \frac{(\epsilon_0+cR)[c_{33}(1+\lambda)\mu r - c_{32}] + (2cR/\ell - \theta_0)[c_{22} - c_{23}(1+\lambda)\mu r]}{(c_{21}+c_{23}\lambda r)[c_{32}-c_{33}(1+\lambda)\mu r] - (c_{31}+c_{33}\lambda r)[c_{22}-c_{23}(1+\lambda)\mu r]}$ | $c_{33}$ 大则 corrective torque 大 |

## 10. 总结

这篇 1989 年的论文虽然是 master thesis，但触及了 robotic assembly 的几个根本 issue：
1. Manipulator compliance 是 configuration-dependent — 可以用 kinematic singularity 来 "免费" 获取 compliance
2. Force 公式可以用 closed-form 解析表达 — phase transition 的 critical depth 是 quadratic equation 的根
3. 实验验证解析模型在 IBM 7540 SCARA robot 上吻合良好

这种 **以解析模型指导 robot programming** 的思路，在 today 的 learning-based 方法时代仍然有参考价值 — analytical model 提供 prior，data 提供 residual correction，是 model-based RL 的天然形态。

参考资源：
- Whitney 1982 原始 paper: https://doi.org/10.1115/1.3139694
- Whitney 的 part mating 综述: https://link.springer.com/chapter/10.1007/978-1-4757-4335-6_42
- RCC device 原理: https://patents.google.com/patent/US4116417A
- 现代 force-guided assembly: https://ieeexplore.ieee.org/document/126785
- Variable Stiffness Actuation review: https://ieeexplore.ieee.org/document/6600745
- IBM 7545/7540 SCARA robot 介绍: https://www.computerhistory.org/revolution/artificial-intelligence-robotics/6/328
- ATI F/T sensor（Lord 业务继任者）: https://www.ati-ia.com/products-ft/ft-sensors.aspx
