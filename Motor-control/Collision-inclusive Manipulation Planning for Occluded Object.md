---
source_pdf: Collision-inclusive Manipulation Planning for Occluded Object.pdf
paper_sha256: bba0b857211a192b71873b88e653ee3ceb819c6cc43cf336f4ee442f63001c01
processed_at: '2026-08-03T16:34:04-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

## 一句话总结

**让 robot 抓那些"藏起来"的东西** —— 比如平躺在桌上的书、被邻居挤住的书架上的书 —— 这些场景下传统的"不要碰到任何东西"的规划方法直接死掉,因为根本不存在一条不碰东西的路。这篇 paper 的核心想法是:**别怕碰,贴着东西滑进去,撞一下没关系,撞错了再来一次**。

---

## 一、问题到底是啥

想象三个场景:

**场景 A**: 一本书平摊在桌上。你想用两指 gripper 把它捏起来。但 gripper 的手指得插到书底下那个 1-2 毫米的缝隙里。从空中直接戳下去? 你不知道缝在哪 (perception 误差),戳偏了就废了。

**场景 B**: 书架上书塞得很紧,你想抽中间那本。手指得从两侧的 gap 分别塞进去。同问题: gap 的位置你看不准。

**场景 C**: 桌上一堆东西挤在一起,你想抓其中一个。同问题。

传统 motion planner 的做法是:规划一条从起点到 grasp pose 的路径,**约束是路径上每一点都不能碰任何东西**。但这几个场景里,grasp pose 本身就在物体深处,任何路径都会穿模,planner 直接说"无解"。

这篇 paper 说: 那就让它穿模,我用一种"软"的控制方式 (impedance control) 让 robot 在真实执行时撞上去能扛得住,而且撞上去之后环境会"引导"手指滑到正确位置。

---

## 二、核心 Intuition: 人怎么干这事

你闭着眼睛想在书架上抽一本书,你会怎么做?

**不会**做的: 凭记忆瞄准 gap 的位置,直接从空中戳过去。因为你的位置记忆有误差,大概率戳偏。

**会**做的: 把手指贴在书的侧面,沿着表面慢慢往里 slide。当手指碰到 gap 的时候,物理上手指会"掉进" gap 里,因为 gap 处没有阻力。这就是 environment 帮你 funnel (漏斗) 到正确位置。

这里有两个 key insight:

1. **Environmental constraint 是朋友,不是敌人**。桌面、书表面这些 contact 把你的手指 motion 限制在一个低维空间 (只能沿表面滑,不能穿表面),你的 perception uncertainty 反而被消除了。
2. **Compliance (柔顺) 是必须的**。如果你硬戳,撞到书侧面 robot 就卡死或者损坏东西。如果你是"软"的 (impedance control 让 gripper 像弹簧一样),撞上去能 slide 一下、退一下、换条路继续。

但 compliance 引入新问题: robot 软了之后,撞上东西到底会怎么动,很难精确预测 (dynamics 不准)。所以单次执行可能 fail。Paper 的解法: **fail 了就再来一次**。因为 framework 不会大幅改变 environment (只是 slide 一下),retry 是在同一个 "funnel" 里探索,总有某次会成功。这就像 lock-picking——你 rake pick 来回刮,总有某次 pin 会 set。

---

## 三、整个 Framework 的结构 (Algorithm 1)

```
观察物体 pose (用 FoundationPose)
↓
while 没成功 and 次数 < 10:
    采样一个 pre-grasp 起点 X_0 (在目标附近随机采)
    ↓
    规划一条 geometric path (两步走,见下文)
    ↓
    把 path 转成 impedance control 指令序列 ξ_0..ξ_{T-1}
    ↓
    让 robot 执行这串 ξ
    ↓
    如果 robot 成功到达 grasp pose → 关 gripper → 抓住了?
        yes → return 成功
        no  → 换个 X_0 再来
```

注意这是 **open-loop**: 一串 ξ 算好之后直接发给 robot,中途不根据 feedback 修改 ξ。如果中途 robot 偏离 path 太远,直接 abort 这次 trial,换个起点重试。

---

## 四、几何表示 (Section IV-A)

要让 planner 能算"手指离物体表面有多远",需要一个可微分 (differentiable) 的几何表示。Paper 用两套:

### Robot gripper: Point cloud
Gripper mesh 上采 1000 个 surface points (蓝色) + 50 个 volume points (红色)。
- **Surface points** 用来估 contact force (力发生在表面)
- **Volume points** 用来算 collision cost (体积内的穿透才算真正碰撞)

为什么分开? Surface points 密了计算贵,而 collision cost 只要"有没有碰"的宏观信号,用稀疏的 volume points 够了。Force 估计需要表面精确,所以用密 surface points。

### Objects: Signed Distance Field (SDF)
每个物体在一个 3D grid (resolution Δ=1cm) 上存 signed distance。每个 voxel 存一个数:
- 正值: 离表面这么远 (free space)
- 0: 在表面上
- 负值: 穿透表面这么深

SDF 的好处: **query O(1), 可微分**。给任意点 p, 通过 trilinear interpolation 得到 $\phi(p)$。这对后面 gradient-based optimization 必不可少。

Eq. (3) 把 base frame 下的点 p 转到 object body frame:
$$\phi_i(\mathbf{p}) = \mathrm{SDF}_i\left((X^i)^{-1} \cdot \mathbf{p}\right)$$

- $X^i \in SE(3)$: object i 在 base frame 中的 pose
- $(X^i)^{-1} \cdot \mathbf{p}$: 把 p 从 base frame 变到 object body frame
- $\mathrm{SDF}_i(\cdot)$: 查这个 object 的 SDF

Eq. (4) 取所有物体中最近的 signed distance,作为"到环境的距离":
$$\phi(\mathbf{p}) = \min_{i \in \{1, ..., M\}} \phi_i(\mathbf{p})$$

SDF 是 robotics / graphics 的标准工具,类似工作:
- DeepSDF: https://github.com/facebookresearch/DeepSDF
- NeuralSDF: https://github.com/zekunhao1995/NeuralSDF

---

## 五、Path Planning: 两步走 (Section IV-B)

这是 paper 的核心算法。分两步,理由后讲。

### Step 1: 只规划 fingertip 位置,用 A* (Eq. 5)

暂时不管 gripper 朝向,只规划 fingertip 这个点 (task frame 定义在 fingertip 上) 在 3D 空间的轨迹。优化目标:

$$\min_{\mathbf{p}_1, ..., \mathbf{p}_T} \quad w \cdot \sum_{t=1}^{T} \phi(\mathbf{p}_t)^2 + \sum_{t=0}^{T-1} \|\mathbf{p}_t - \mathbf{p}_{t+1}\|^2 \quad \text{(5a)}$$
$$\text{s.t.} \quad \phi(\mathbf{p}_t) \leq -\delta_g, \quad \forall t = 1, ..., T \quad \text{(5b)}$$

变量解释:
- $\mathbf{p}_t \in \mathbb{R}^3$: fingertip 在第 t 个 waypoint 的位置 (优化变量)
- $T = 20$: 路径上 waypoint 数量
- $w = 1/\Delta^2 = 1/\text{(1cm)}^2$: 两个 term 之间的权重
- $\phi(\mathbf{p}_t)^2$: SDF 的平方,**鼓励 fingertip 贴着物体表面** (SDF=0 就在表面)
- $\|\mathbf{p}_t - \mathbf{p}_{t+1}\|^2$: 平滑项,鼓励路径短
- $\delta_g \geq 0$: 允许 fingertip **最多穿透环境 $\delta_g$ 这么深**。约束 (5b) $\phi(\mathbf{p}_t) \leq -\delta_g$ 意思是 fingertip 至少穿透 $\delta_g$ 这么深 (SDF 负值表示穿透)

**人话翻译**: 第一项说"贴着表面走",第二项说"路径别绕远",约束说"别戳太深"。

**为什么贴表面?** 这就是前面说的 "盲眼滑进 gap" 策略。手指贴着 book 表面 slide,碰到 gap 就掉进去。如果直接从空中戳,perception 误差让你大概率戳偏。

A* 实现 (Algorithm 2): workspace 离散化成 3D grid,违反约束 (5b) 的 voxel 当 obstacle。Cost $c(\mathbf{p}, \mathbf{p}') = w \cdot \phi(\mathbf{p}')^2 + \|\mathbf{p} - \mathbf{p}'\|^2$。Heuristic $h = \|\mathbf{p} - \mathbf{p}'\|^2$,admissible (低估真实 cost,保证 A* 找到最优)。

如果找不到 path,就增大 $\delta_g$ (允许更深穿透),重试。这个 trick 很重要:因为 perception 有误差,物体可能被估计成穿透桌面,fingertip-to-table gap 算出来是负的,必须允许 $\delta_g$ 增大才能找到路。

### Step 2: Refine 完整 SE(3) pose,最小化 gripper body collision (Eq. 6-8)

Step 1 只管 fingertip 一个点,但 gripper 是有大小的,gripper base、另一只 finger 都可能撞东西。Step 2 在 SE(3) 上 refine 完整 pose。

Eq. (6) 是 CHOMP-style collision cost:

$$c(\phi(\mathbf{p})) = \begin{cases} 
-\phi(\mathbf{p}) + \frac{1}{2}\varepsilon & \phi(\mathbf{p}) < 0 \\
\frac{1}{2\varepsilon}(\phi(\mathbf{p}) - \varepsilon)^2 & 0 \leq \phi(\mathbf{p}) < \varepsilon \\
0 & \text{otherwise}
\end{cases}$$

变量:
- $\phi(\mathbf{p})$: signed distance (负=穿透,正=free space)
- $\varepsilon = 1$ cm: 安全 margin,collision cost 在距离 $\varepsilon$ 内开始非零

形状解释:
- $\phi < 0$ (已穿透): cost 线性增长,越深 cost 越大
- $0 \leq \phi < \varepsilon$ (在 margin 内,快碰了): cost 二次平滑上升
- $\phi \geq \varepsilon$ (远离): cost = 0

为什么有 smooth ramp 在 $[0, \varepsilon]$? 因为 gradient-based optimizer 在 $\phi = 0$ 处需要连续可导,否则会数值爆炸。CHOMP 原始 paper: https://www.ri.cmu.edu/pub_files/2009/5/icra09-chomp.pdf

Eq. (7): 总 collision cost 是 gripper 内 50 个 volume points 对所有物体的 cost 之和:
$$C(X_t) = \sum_{i=1}^{M} \sum_{\mathbf{p} \in \mathcal{P}^v} c(\phi_i(X_t \cdot \mathbf{p}))$$

- $X_t \cdot \mathbf{p}$: 把 task frame 下的点 p 变到 base frame
- $\mathcal{P}^v$: gripper 体积内的 50 个采样点

Eq. (8): refinement 优化
$$\min_{\widetilde{X}_1, ..., \widetilde{X}_T} \quad \sum_{t=1}^{T} C(\widetilde{X}_t) + \sum_{t=0}^{T-1} d(\widetilde{X}_t, \widetilde{X}_{t+1}) \quad \text{(8a)}$$
$$\text{s.t.} \quad \widetilde{X}_T = X^g \cdot X^* \quad \text{(8b)}$$
$$\|\widetilde{\mathbf{p}}_t - \mathbf{p}_t\| \leq \delta_p, \quad \forall t \quad \text{(8c)}$$

- $\widetilde{X}_t \in SE(3)$: 优化变量,完整 pose (position + orientation)
- $d(\cdot, \cdot)$: SE(3) distance (位置 L2 + 朝向角度)
- $X^g \cdot X^*$: 目标 grasp pose 在 base frame 下的表示 ($X^g$ 是 object pose,$X^*$ 是 grasp 相对 object 的 pose)
- $\widetilde{\mathbf{p}}_t$: $\widetilde{X}_t$ 的位置部分
- $\mathbf{p}_t$: Step 1 找到的位置路径
- $\delta_p$: 允许的偏离

**约束 (8c) 的关键意义**: 让 fingertip 不偏离 Step 1 找的"贴表面"路径太远。约束 (8b) 让末端正好到 grasp pose。

### 为什么要两步,不直接在 SE(3) 上一次优化?

直觉: SE(3) 是 6 维,直接优化很容易 stuck 在 local minimum。比如 optimizer 可能让 gripper 转 90 度,让 gripper body 远离物体——但这样 fingertip 也跑到别处去了,完全没"贴表面 slide"的效果。

两步走是 coarse-to-fine: 先在低维 (3D 位置) 找到"贴表面"的关键路径骨架,然后在高维 (6D SE(3)) refine,但被这个骨架 anchor 住,不能跑偏。这是 motion planning 里常见的 pattern,和 trajectory optimization 里 "warm start" 一个道理。

---

## 六、生成 Impedance Control 指令 (Section IV-C)

有了 geometric path $\widetilde{X}_1, ..., \widetilde{X}_T$,现在要把它转成 robot 能执行的指令。但这里有个 chicken-and-egg 问题: 这条 path 是在"穿透物体"的虚拟几何空间里规划的,真实执行时 robot 会撞东西,撞了之后怎么动我们不知道。所以不能直接把 path 当 reference 发给 position controller——那会撞坏东西或者卡死。

**解法**: 用 **Cartesian impedance control**。这个 controller 不追求"精确到这一帧",它追求"软软地往那个方向推"。具体: 在 task frame 上模拟一个虚拟弹簧阻尼系统,reference pose $\boldsymbol{\xi}$ 是弹簧的"自然长度",gripper 当前 pose $\mathbf{x}_t$ 是弹簧当前长度。controller 算出虚拟力 $\mathbf{f}_t = K(\boldsymbol{\xi} - \mathbf{x}_t) - D\dot{\mathbf{x}}_t$,把这个力送给 robot 的 force/torque controller。

### Impedance control 基础公式 (Eq. 1-2)

虚拟弹簧阻尼方程:
$$\mathbf{f}_t = K(\boldsymbol{\xi} - \mathbf{x}_t) - D\dot{\mathbf{x}}_t$$

- $\boldsymbol{\xi} \in SE(3)$: **reference equilibrium pose**,这是我们优化的控制指令
- $\mathbf{x}_t \in \mathbb{R}^6$: 当前 task frame pose (position + RPY)
- $K \in \mathbb{R}^{6\times 6}$: stiffness 矩阵 (对角)
- $D \in \mathbb{R}^{6\times 6}$: damping 矩阵 (对角)
- $\mathbf{f}_t \in \mathbb{R}^6$: 虚拟 wrench (3D force + 3D torque)

Robot joint-space dynamics:
$$M(\mathbf{q}_t)\ddot{\mathbf{q}}_t + C(\mathbf{q}_t, \dot{\mathbf{q}}_t) = J(\mathbf{q}_t)^\top (\mathbf{f}_t + \mathbf{f}_{ext})$$

- $\mathbf{q}_t \in \mathbb{R}^N$: joint angles (Franka N=7)
- $M(\mathbf{q}_t)$: $N \times N$ mass matrix
- $C(\mathbf{q}_t, \dot{\mathbf{q}}_t)$: Coriolis + centrifugal 项
- $J(\mathbf{q}_t) \in \mathbb{R}^{6\times N}$: Jacobian
- $\mathbf{f}_{ext} \in \mathbb{R}^6$: 外部 contact wrench

参考 Hogan 1985 (impedance control 鼻祖): http://asmedigitalcollection.asme.org/dscc1985/proceedings-abstract/DSCC1985-v1/80271/229391

### 转 task space + 慢动作近似 (Eq. 9-10)

把 joint space dynamics 通过 Jacobian 转到 task space:

$$\Lambda(\mathbf{q}_t)(\ddot{\mathbf{x}}_t - \dot{J}\dot{\mathbf{q}}_t) = \mathbf{f}_t + \mathbf{f}_{ext} \quad \text{(9)}$$

- $\Lambda(\mathbf{q}_t) = J^{-\top} M J^{-1}$: **task space inertia matrix** (operational space dynamics)

慢动作假设: $\dot{J}\dot{\mathbf{q}}$ 项可以忽略 (motion 慢时 Jacobian 变化率小),得到:

$$\ddot{\mathbf{x}}_t \approx \Lambda(\mathbf{q}_t)^{-1} (\mathbf{f}_t + \mathbf{f}_{ext}) \quad \text{(10)}$$

这是 Khatib 1987 的 operational space dynamics 简化版: https://khatib.stanford.edu/publications/Khatib_1987_RA.pdf

**假设合理吗?** Contact-rich slow manipulation 场景下,robot 速度慢,确实 $\dot{J}\dot{q}$ 很小。但高速 motion (swing-up, 抛物体) 下这个近似会 break。Paper 的 task 都是慢的,所以 OK。

### 估 external wrench $\mathbf{f}_{ext}$ (Eq. 11-12)

这是 paper 最聪明的设计。Robot 在执行 geometric path 时,会"穿透"环境 (因为 path 允许 $\delta_g$ 穿透)。这种穿透在真实世界对应 contact force。Paper 用一个非常粗糙的 model 估算这个 force:

对每个 surface point $\mathbf{p} \in \mathcal{P}^s$:
$$\mathbf{f}(\mathbf{p}) = \mathbb{1}_{\{\phi(\widetilde{X}_t \cdot \mathbf{p}) < 0\}} \cdot k \cdot \phi(\widetilde{X}_t \cdot \mathbf{p}) \cdot \mathbf{n} \quad \text{(11)}$$

变量:
- $\mathbb{1}_{\{\phi < 0\}}$: 只对穿透的点算 (没穿透没 contact)
- $k$: virtual stiffness hyperparameter
- $\phi(\widetilde{X}_t \cdot \mathbf{p})$: signed distance (负值=穿透深度)
- $\mathbf{n} \in \mathbb{R}^3$: gripper 表面在 point p 处的外法线

注意: $\phi$ 是负的,所以 $k \cdot \phi \cdot \mathbf{n}$ 是沿外法线**反向**的力——环境把 gripper 往外推。这物理上对。

Eq. (12) 总 wrench 是所有穿透点的 force 和 torque 平均:
$$\mathbf{f}_{ext}(\widetilde{X}_t) = \frac{1}{Z} \begin{pmatrix} \widetilde{R}_t \cdot \sum_{\mathbf{p} \in \mathcal{P}^s} \mathbf{f}(\mathbf{p}) \\ \widetilde{R}_t \cdot \sum_{\mathbf{p} \in \mathcal{P}^s} \mathbf{p} \times \mathbf{f}(\mathbf{p}) \end{pmatrix}$$

- $Z = \sum \mathbb{1}_{\{\phi < 0\}}$: 穿透点数量 (归一化用)
- $\widetilde{R}_t \in \mathbb{R}^{3\times 3}$: $\widetilde{X}_t$ 的旋转部分,把力从 task frame 转到 base frame
- 上半: 合力 (3D)
- 下半: 合力矩 (cross product 给 moment arm)

**这个 estimate 多粗糙?** 非常粗糙。没有 friction,没有 contact patch 的真实形状,没有 deformation。但 paper 说: 不需要准,只要给个方向感,让后面的 optimization 知道大概要 push 多远、push 哪个方向来 compensate 掉 contact。最后 impedance control 的 compliance 会吸收残余误差。

这是 **coarse model + robust controller** 哲学,classic control 里很常见,在 manipulation planning 里用得不多——大部分 planning 都假设 model 准。

### Forward dynamics simulation $\Pi$ (Eq. 13-14)

把所有 piece 组装成 $\Pi(\mathbf{x}_t, \dot{\mathbf{x}}_t, \boldsymbol{\xi}_t) \to (\mathbf{x}_{t+1}, \dot{\mathbf{x}}_{t+1})$:

$$\mathbf{x}_\tau = \mathbf{x}_t + \int_t^\tau \dot{\mathbf{x}}_s \, ds \quad \text{(14a 上)}$$
$$\dot{\mathbf{x}}_\tau = \dot{\mathbf{x}}_t + \int_t^\tau \ddot{\mathbf{x}}_s \, ds \quad \text{(14a 下)}$$
$$\ddot{\mathbf{x}}_\tau = \Lambda^{-1}(\mathbf{f}_\tau - \mathbf{f}_{ext}(\widetilde{X}_t)) \quad \text{(14b)}$$
$$\mathbf{f}_\tau = K(\boldsymbol{\xi}_t - \mathbf{x}_\tau) - D\dot{\mathbf{x}}_\tau \quad \text{(14c)}$$

数值积分用 2ms 时间步。$D$ 设为 critical damping $D = 2\sqrt{K}$,避免 oscillation。Critical damping 是 control theory 标配,见 https://en.wikipedia.org/wiki/Damping#Damping_ratio。

### 优化 $\boldsymbol{\xi}$ 序列 (Eq. 15)

最终优化:

$$\min_{\boldsymbol{\xi}_0, ..., \boldsymbol{\xi}_{T-1}} \quad \sum_{t=1}^{T} d(\widetilde{X}_t, X_t) \quad \text{(15a)}$$
$$\text{s.t.} \quad \|\dot{\mathbf{x}}_0\|^2 = 0, \quad \|\dot{\mathbf{x}}_T\|^2 = 0 \quad \text{(15b)}$$
$$\|\dot{\mathbf{x}}_t\|^2 \leq \delta_v, \quad \forall t = 1, ..., T-1 \quad \text{(15c)}$$
$$\mathbf{x}_{t+1}, \dot{\mathbf{x}}_{t+1} = \Pi(\mathbf{x}_t, \dot{\mathbf{x}}_t, \boldsymbol{\xi}_t) \quad \text{(15d)}$$

- $\boldsymbol{\xi}_t$: 优化变量,第 t 个 reference pose
- $X_t$: dynamics $\Pi$ 下 predicted 的 pose
- $\widetilde{X}_t$: Section IV-B 规划的 geometric path
- $d(\cdot, \cdot)$: SE(3) distance
- $\delta_v$: 速度上界

约束:
- (15b): 起止速度 0 (rest-to-rest)
- (15c): 中间速度 bounded (safety)
- (15d): dynamics 约束,把 $\boldsymbol{\xi}_t$ 和 $\mathbf{x}_{t+1}$ 耦合

**目标 (15a) 的意义**: 让 "在 approximate dynamics 下预测出来的轨迹" 尽可能贴 "geometric path 规划的 desired 轨迹"。换句话说,我们要找一串 $\boldsymbol{\xi}$,使得即使有 contact force (用粗糙 model 估的),robot 也能跟着 desired path 走。

求解用 CasADi + IPOPT (interior point method),参考:
- CasADi: https://web.casadi.org/
- IPOPT: https://coin-or.github.io/Ipopt/

---

## 七、Manipulation Funnel by Repetitions (Section III-B)

这部分 intuition 很漂亮,但数学不重。Mason 1986 的 manipulation funnel: 把 initial state 的 uncertainty 集合通过 constrained motion "漏斗" 到确定的 final state。经典例子: 把零件放到漏斗里,不管零件初始姿态如何,漏斗壁会把零件引导到唯一出口。

Paper 把这个概念 **"degenerate"** 了 (Fig. 2)。原本 funnel 是空间概念——一次 motion 把不确定性收敛。这里 funnel neck 是 "flat" 的——单次 motion 可能找不到出口 (因为 perception / dynamics 不准)。但 framework 不会大幅改变 environment,所以可以 retry。每次 retry 在 funnel 内随机探索,总有某次找到出口。

数学上,这是把 funnel 从 "空间维度" 扩展到 "时间维度" (repetitions)。和 Monte Carlo methods、stochastic search 是一类思路。

参考 Mason 1986: https://www.cs.cmu.edu/~mma/mason-pubs/Mason86.pdf

---

## 八、实验

### Setup
- Robot: Franka Emika (7-DoF),单臂 + 双臂
- Vision: FoundationPose (6D pose estimation) + CNOS (segmentation),RTX 3060
- 9 个 YCB 物体: Book, Cleanser Bottle, Mustard Bottle, Cracker Box, Timer, Clamp, Scissors, Bowl, Plate
- 每 trial 最多 10 reps,每物体 10 trials
- 路径长度 T=20,SDF resolution Δ=1cm,ε=1cm,δ_g 从 0 开始步长 1mm

参考:
- YCB dataset: https://www.ycbbenchmarks.com/
- FoundationPose: https://foundationpose.github.io/
- CNOS: https://github.com/nv-nguyen/CNOS

### Experiment A: 单物体在桌上 (Fig. 7 表格)

**单臂结果**:
| Metric | 值 |
|---|---|
| Overall success | 90/90 (100%) |
| Average reps to success | 1.4 ± 0.9 |
| Average total solve time | 3.60 ± 2.63 秒 |

**双臂结果** (一只 arm 抓,另一只挡):
| Metric | 值 |
|---|---|
| Overall success | 90/90 (100%) |
| Average reps | 1.6 ± 1.2 |
| Average solve time | 4.58 ± 3.66 秒 |

**有意思的细节**:
- **Cracker Box** 单臂需要 2.7 reps,因为 box 厚度接近 gripper 最大开口,要求 orientation 对准。而且 box 是空的容易 deform,真实形状和 mesh 不一致。
- **Bowl** 单臂 1.0 reps,但双臂需要 3.3 reps,因为 bowl 有 narrow rim 在 mesh 里没建模,增加 unmodeled contact。
- **Solve time variance 大** (std 2.63s),因为 perception error 导致物体被估计穿透桌面,A* 要增大 δ_g 多次重试,有时一次 A* 跑 6+ 秒。
- **A* 部分通常 < 0.5s**,Refine (CHOMP-style) 通常 < 1.5s,Impd (生成 ξ) 通常 2s 左右。大部分时间花在 Impd 优化上 (CasADi + IPOPT 求解)。

### Experiment B: 紧密堆叠多物体 (Fig. 8)

三个场景:
1. Tray 里抓 target book (单臂)
2. Shelf 中抓 target book (单臂)
3. Table 上叠放抓 target book (双臂)

这里要 **同时插入两个 finger** 进两个 gap,更难。所有场景 100% success,average reps 略增 (因为更复杂 contact geometry)。Solve time 4-7 秒。

### Experiment C: Baseline 对比 (Fig. 9)

Baseline: Wirnshofer 2018 (belief space planning for assembly, https://ieeexplore.ieee.org/document/8460987)。这个 baseline 也用 contact 引导 gripper,但它的 dynamics model 更精确,所以对 modeling inaccuracy 更敏感。

只测 3 个简单 case,5 trials each (因为 baseline 容易 fail 导致物体或另一只 arm 损坏)。每 trial 给 120 秒预算,允许 baseline 在预算内 replan + retry。

Baseline 的问题:
- **更 sensitive to modeling inaccuracies**: 因为它 trust model,paper 的 approximate dynamics (Eq. 14) 让它经常算错
- **Random sampling-based motion generation 效率低**: 通常 > 20 秒找解
- **Execution failures 多**: finger 插不进 book gap,clamp 抓错位置

---

## 九、为什么这个 Framework Works —— 三个 Compensating Components

这篇 paper 的 elegance 在于: 把一个 hard problem 拆成三个 uncertainty,每个 component 处理一种。

### Component 1: Collision-inclusive path planning
**解决的问题**: 传统 collision-free planner 在 occluded grasping 下无解。
**做法**: 允许 path 穿透 environment (Eq. 5b 的 $\delta_g$),通过 soft cost 而非 hard constraint 处理 collision。
**引入的 uncertainty**: 真实执行时会有 contact force,dynamics 不确定。

### Component 2: Impedance control
**解决的问题**: Collision 引入的 dynamics uncertainty。
**做法**: 用 compliance 吸收 contact force——撞上去 gripper 退一下,不会硬刚坏东西。
**引入的 uncertainty**: 单次执行可能因为 perception / model 不准,fail 掉。

### Component 3: Manipulation funnel via repetitions
**解决的问题**: 单次 fail 的概率。
**做法**: Retry,每次换个起点。因为 framework 不大幅改变 environment,所有 retry 都在同一个 "funnel" 内探索,总有某次掉进出口。
**核心 insight**: Contact-rich task 的失败不破坏 environment,这是 retry 可行的前提。

如果只有一个 component,系统会卡住:
- 只有 collision-inclusive planning,没 impedance control → 撞坏东西
- 只有 impedance control,没 planning → 撞上去乱动,方向不对
- 只有 retry,没 planning + control → 瞎试,效率低

三个组合起来,systematic exploit environment + absorb uncertainty + retry。

---

## 十、深入 Intuition: 几个 Subtle Design Choices

### 1. 为什么用 SDF,不用 mesh collision checking?

Mesh-mesh collision checking (e.g., GJK algorithm, https://en.wikipedia.org/wiki/Gilbert%E2%80%93Johnson%E2%80%93Keerthi_distance_algorithm) 给的是 binary yes/no,不可微。Optimization 需要 gradient,所以需要 smooth distance field。SDF 是最简单选择。其他选项: occupancy grid (不可微)、neural SDF (可微但慢)。

### 2. 为什么 Step 1 用 A*,不用 gradient descent?

A* 是 discrete graph search,**保证找到全局最优** (given admissible heuristic)。Gradient descent 在 3D 位置空间可能 stuck 在 local minimum (比如 fingertip 被困在某个角落)。Paper 的 Eq. (5) 是 non-convex optimization,A* 给出全局解。

但 A* 只能在 discrete grid 上跑,所以 Step 2 用 gradient-based refinement 把 path smooth 化、调整 orientation。

### 3. 为什么 ξ optimization 用 approximate dynamics 当 constraint,不当 objective?

如果 dynamics 当 objective (minimize 跟踪误差),optimizer 可能让 ξ 离 path 很远来 "欺骗" dynamics。当 constraint (15d),dynamics 是硬约束——optimizer 必须找的 ξ 在 dynamics 下确实产生接近 desired path 的轨迹。这是 differentiable simulation + trajectory optimization 的标准套路。

### 4. 为什么不直接做 closed-loop impedance control with feedback?

Paper 是 open-loop 发 ξ。可以想象 closed-loop 版本: 用 force/torque sensor 估真实 $\mathbf{f}_{ext}$,在线 replan ξ。但 paper 没做,可能因为:
- replan 时间 4 秒,太慢,做不到 in-loop
- perception 是 vision-based,不能实时更新 object pose
- 用 retry 代替 closed-loop,简单且足够好

这暗示一个 **future direction**: 用 RL 学一个 reactive policy 来在线调整 ξ,可能减少 reps,提高效率。

### 5. Repetitions vs. RL 的对比

Repetitions 本质是 **open-loop planning + 简单 retry**。RL 学一个 policy 可以 **closed-loop reactive**。Paper 的 baseline 是 model-based planner (Wirnshofer),没和 RL 比。可能 future work: 把这个 framework 当 model-based baseline,学一个 RL policy 来 beat 它。Paper 的 100% success rate + 1.4 reps avg 是一个 hard baseline。

RL 方向的相关工作:
- Learning contact-rich manipulation: https://arxiv.org/abs/2110.04768
- Reinforcement learning for dexterous manipulation: https://dexterous-learning.github.io/

---

## 十一、Limitations 我自己猜的

1. **慢动作假设**: Eq. (10) 忽略 $\dot{J}\dot{q}$,dynamic motion 下 break。但 task 都是慢的,OK。
2. **Object 不能大幅移动**: 这是 framework 前提。如果物体 loosely packed,抓 book 时旁边 book 会倒,funnel collapse。
3. **Geometric model 必需**: 需要 gripper mesh + object mesh。对 unknown objects 要先 reconstruct。对软体、透明物体仍有挑战。
4. **Open-loop ξ**: 没在线 feedback 调整。如果中途 contact 状态严重偏离 plan,只能 abort + retry。可以想象 closed-loop 版本,用 F/T sensor 在线 replan。
5. **Single fingertip task frame**: 对 dual-finger asymmetric 任务,task frame 在一个 fingertip,另一个 finger 靠 gripper symmetric geometry "顺带" 进去。对真正 asymmetric 任务需要 extension。
6. **K_max = 10 的选择**: 如果某些极端场景需要 50 reps,paper 的 framework 可能不实用。但实验中 average 1.4 reps,所以 10 上限很宽松。

---

## 十二、Related Work 脉络

如果要做 follow-up,这些是必读:

- **CHOMP** (Ratliff 2009): paper 的 Eq. (6) 直接用 — https://www.ri.cmu.edu/pub_files/2009/5/icra09-chomp.pdf
- **Mason manipulation funnel**: funnel 概念源头 — https://www.cs.cmu.edu/~mma/mason-pubs/Mason86.pdf
- **Eppner & Brock environmental constraints in grasping**: 同思想 grasping 应用 — https://www.researchgate.net/publication/224528889
- **Khatib operational space dynamics**: task space dynamics — https://khatib.stanford.edu/publications/Khatib_1987_RA.pdf
- **Hogan impedance control**: impedance control 奠基 — http://asmedigitalcollection.asme.org/dscc1985/proceedings-abstract/DSCC1985-v1/80271/229391
- **FoundationPose**: 6D pose estimator — https://foundationpose.github.io/
- **Wirnshofer 2018**: baseline — https://ieeexplore.ieee.org/document/8460987
- **Suomalainen 2022 survey**: contact-rich manipulation survey — https://arxiv.org/abs/2202.10181
- **DiffHand**: differentiable dynamics for manipulation — https://diffhand.github.io/
- **DeepSDF**: learned SDF — https://github.com/facebookresearch/DeepSDF

---

## 十三、最最人话总结

这篇 paper 做的事: 让 robot 用"贴着滑"的方式抓藏起来的东西。

具体三招:
1. **规划时允许穿模**: 规划一条"贴着物体表面"的路径,允许穿透一点点。这比传统"不能碰任何东西"的规划更合理,因为 occluded grasping 本来就要碰。
2. **执行时软软的**: 用 impedance control,robot 像弹簧,撞上去能退,不会硬刚坏东西。但需要估一下撞上会多大力,paper 用一个很粗糙的 model 估,够用就行,剩下误差 compliance 吸收。
3. **撞错了再来一次**: 因为 robot 不会大幅改变 environment,retry 是在同一个"漏斗"里探索,平均 1.4 次就成功。

这套 framework 在 9 个 YCB 物体上 100% success,平均 1.4 reps,solve time 3.6 秒。对做 learning-based manipulation 的人来说,这是一个 **strong, interpretable model-based baseline**——任何 RL / IL 方法要 claim 学到 contact-rich policy,都得先 beat 它。而且 framework 的 modular breakdown (perception → planning → control → retry) 是 debugging learning policy 时很好的参考结构。

---

# Collision-inclusive Manipulation Planning for Occluded Object Grasping via Compliant Robot Motions — 深度技术讲解

## 一、Paper 的核心 Intuition

这篇 paper 解决一个非常具体但很有意思的问题: **occluded object grasping**。想象一本书躺在桌面上、或者被其他书紧紧夹在书架上, gripper 想要 grasp 这本书, 但是所有可行的 grasp pose 都被环境遮挡了。传统 collision-free planner 在这种场景下直接 fail, 因为任何一条到 grasp pose 的路径都会穿透 obstacle。

作者的核心 insight 是: **不要把 collision 当 bug, 要把它当 feature**。Human 在这种场景下怎么做事? 我们会让手指贴着桌面 slide 进 book 下面的 gap, 或者贴着旁边的书 slide 进书架的缝隙。这种 "compliant, contact-rich, sliding" 的 motion 利用了 **environmental constraints** —— 接触表面把 finger 的 motion 限制在一个低维 manifold 上, perception uncertainty (你不知道 gap 到底在哪) 被 contact 时的物理 constraint 给 "funnel" 掉了。

但是这里有个 chicken-and-egg 问题: compliance 引入了更多 uncertainty (robot 在 contact 下怎么动很难精确预测), 同时又靠 contact 来减少 uncertainty (environment 把 finger 引导到正确位置)。paper 的方法就是在这两者之间做 trade-off, 用 **roughly modeled collisions + Cartesian impedance control + task repetitions** 三个工具组合起来。

---

## 二、整体框架 (Algorithm 1)

整体 loop 很简单:

```
while k < K_max:
    observe object poses (用 FoundationPose)
    sample pre-grasp pose X_0 around (X^g · X*)
    plan geometric path via A* + CHOMP refinement
    generate impedance controls ξ_0..ξ_{T-1} via optimization
    move to X_0
    if execute(ξ) succeeds: return true
    else: try again with new X_0
```

关键的概念是 **degenerate manipulation funnel** (Fig. 2)。Mason 在 1986 年提出 manipulation funnel 的概念: 把 initial configuration 的不确定性集合, 通过一系列 constrained motions, "漏斗" 到一个确定的 final state。这里作者把这个概念扩展了: 一次 execution 可能因为 perception / dynamics 的误差 fail, 但是因为 framework 不会大幅改变 environment configuration (只是 slide 一下), 所以可以 retry。每次 retry 相当于在 funnel 内部 explore, 最后总有某一次会 "掉进" 正确的 exit (stable grasp)。

Intuition 上, 这就像 lock-picking: 你不知道 pin 的精确位置, 但是你用 tension wrench 保持轻微 torque, 然后 rake pick 来回刮, 总有一次 pin 会 set。Environment contact 就是那个 tension wrench 提供的 constraint, repetitions 就是 rake 的来回 motion。

Reference: Mason, "Mechanics of Robotic Manipulation", 1986 — https://www.cs.cmu.edu/~mma/mason-pubs/Mason86.pdf

---

## 三、几何表示 (Section IV-A)

### Robot gripper: Point cloud
Gripper mesh 上采样 1000 个 surface points (blue) + 50 个 volume points (red)。Surface points 用来估算 contact force (因为 force 发生在 surface), volume points 用来做 collision cost (因为体积内的 penetration 才是真正的 collision)。

为什么这样分? 因为 surface points 太密会导致 collision cost 计算昂贵, 而 volume points 太少会漏掉 collision。Surface points 用于 Eq. (11) 的 force estimation, volume points 用于 Eq. (7) 的 collision cost。

### Objects: Signed Distance Field (SDF)
每个 object 在一个 3D grid 上离散化 (resolution Δ = 1cm), 每个 voxel 存 signed distance 到 object surface 的值。SDF 的好处是 **differentiable** 且 query O(1), 这对后面 gradient-based optimization 必不可少。

Eq. (3) 定义 φ_i(p): 把 base frame 下的点 p 用 object pose X^i 的逆变换转到 object body frame, 然后查 SDF_i:

$$\phi_i(\mathbf{p}) = \mathrm{SDF}_i\left((X^i)^{-1} \cdot \mathbf{p}\right)$$

这里 $X^i \in SE(3)$ 是 object i 在 workspace (robot base frame) 中的 pose, $(X^i)^{-1} \cdot \mathbf{p}$ 把 p 从 base frame 变换到 object body frame。

Eq. (4) 定义整个环境的 SDF: 取到所有 object 的最小 signed distance:
$$\phi(\mathbf{p}) = \min_{i \in \{1, ..., M\}} \phi_i(\mathbf{p})$$

SDF 是机器人 manipulation / motion planning 里非常基础的 representation, 类似的工作还有:
- DiffHand: https://diffhand.github.io/
- Neural SDF (DeepSDF): https://github.com/facebookresearch/DeepSDF

---

## 四、Geometric Path Planning (Section IV-B)

这是 paper 的核心算法部分, 分两步:

### Step 1: Positional path-finding via A* (Eq. 5)

只规划 fingertip (task frame) 的 **位置**, 暂时不管 orientation。优化目标:

$$\min_{\mathbf{p}_1, ..., \mathbf{p}_T} \quad w \cdot \sum_{t=1}^{T} \phi(\mathbf{p}_t)^2 + \sum_{t=0}^{T-1} \|\mathbf{p}_t - \mathbf{p}_{t+1}\|^2 \tag{5a}$$
$$\text{s.t.} \quad \phi(\mathbf{p}_t) + \delta_g \leq 0, \quad \forall t = 1, ..., T \tag{5b}$$

变量解释:
- $\mathbf{p}_t \in \mathbb{R}^3$: fingertip 在第 t 个 waypoint 的位置
- $T = 20$: 路径 waypoint 数量
- $w = 1/\Delta^2$: trade-off factor, $\Delta$ 是 SDF grid resolution (1cm), 所以 w = 1 cm⁻²
- $\phi(\mathbf{p}_t)^2$: 鼓励 fingertip **靠近** environmental surface (SDF = 0 表示在 surface 上)
- $\|\mathbf{p}_t - \mathbf{p}_{t+1}\|^2$: smoothing term, 鼓励路径短
- $\delta_g \geq 0$: 允许 fingertip 穿透 environment 的最大深度 (单位 cm), constraint (5b) 等价于 $\phi(\mathbf{p}_t) \leq -\delta_g$, 即 fingertip 至少穿透 $\delta_g$ 这么深

Intuition: 第一项让 fingertip "贴着" 物体表面走 (SDF ≈ 0 附近), 这样可以利用 contact 来获得 constraint。第二项让路径短。Constraint 防止 fingertip 无脑穿透物体。

**为什么这么设计?** 想象一下你盲着眼睛想把手指塞进两本书之间的 gap。如果你的策略是直接从空中戳过去, perception error 会导致你大概率戳偏。但如果你让手指先贴着书表面 slide, 书表面会 "捕获" 你的手指, 把它 funnel 到 gap 里。这就是第一项要做的事情: 主动让 fingertip 靠近 surface。

A* 实现细节 (Algorithm 2): workspace 离散化为 3D grid (resolution Δ), voxels 违反 constraint (5b) 的当成 obstacle。Cost $c(\mathbf{p}, \mathbf{p}') = w \cdot \phi(\mathbf{p}')^2 + \|\mathbf{p} - \mathbf{p}'\|^2$; heuristic $h(\mathbf{p}, \mathbf{p}') = \|\mathbf{p} - \mathbf{p}'\|^2$ (admissible, 因为它 underestimate 真实 cost)。如果找不到 path, 就增加 $\delta_g$ (允许更深 penetration) 重试。

这里有个工程上的 trick: 因为 perception 有误差, object 可能被检测成穿透 table, 导致 fingertip-to-table gap 是负值。这种情况下, 必须允许 $\delta_g$ 增大才能找到 path。这就是 Algorithm 2 的 iterative $\delta_g$ 增加策略的目的。

### Step 2: Path refinement via collision minimization (Eq. 6-8)

Step 1 只规划了 fingertip 位置, 没有规划 gripper orientation。这一步要 refine 出完整的 SE(3) pose $\widetilde{X}_t$, 同时:
1. 保持 fingertip 位置接近 Step 1 找到的路径
2. 最小化 gripper 其他部分 (gripper base, 另一个 finger) 与环境的 collision

Eq. (6) 是 CHOMP-style collision cost (reference: https://www.ri.cmu.edu/pub_files/2009/5/icra09-chomp.pdf):

$$c(\phi(\mathbf{p})) = \begin{cases} -\phi(\mathbf{p}) + \frac{1}{2}\varepsilon & \phi(\mathbf{p}) < 0 \\ \frac{1}{2\varepsilon}(\phi(\mathbf{p}) - \varepsilon)^2 & 0 \leq \phi(\mathbf{p}) < \varepsilon \\ 0 & \text{otherwise} \end{cases}$$

变量解释:
- $\phi(\mathbf{p})$: 点 p 到环境的 signed distance (负 = penetration, 正 = free space)
- $\varepsilon = 1$ cm: safety margin, 决定 collision cost 在多远的距离开始 non-zero

这个 cost function 的形状:
- $\phi < 0$ (penetrating): cost 线性增长, 越深 cost 越大
- $0 \leq \phi < \varepsilon$ (在 margin 内): cost 是二次的, smooth ramp
- $\phi \geq \varepsilon$ (远离): cost = 0

这个 smooth ramp 设计是为了让 gradient-based optimizer 在 $\phi = 0$ 处不会 discontinuity。

Eq. (7): 总 collision cost 是所有 volume points 和所有 objects 的 cost 之和:
$$C(X_t) = \sum_{i=1}^{M} \sum_{\mathbf{p} \in \mathcal{P}^v} c(\phi_i(X_t \cdot \mathbf{p}))$$

其中 $X_t \cdot \mathbf{p}$ 把 task frame 下的点 $\mathbf{p}$ 变换到 base frame, $\mathcal{P}^v$ 是 gripper volume 内的采样点 (50 个)。

Eq. (8) 是 refinement 优化:
$$\min_{\widetilde{X}_1, ..., \widetilde{X}_T} \quad \sum_{t=1}^{T} C(\widetilde{X}_t) + \sum_{t=0}^{T-1} d(\widetilde{X}_t, \widetilde{X}_{t+1}) \tag{8a}$$
$$\text{s.t.} \quad \widetilde{X}_T = X^g \cdot X^* \tag{8b}$$
$$\|\widetilde{\mathbf{p}}_t - \mathbf{p}_t\| \leq \delta_p, \quad \forall t \in \{1, ..., T\} \tag{8c}$$

变量:
- $\widetilde{X}_t \in SE(3)$: 优化变量, 每个 waypoint 的完整 pose
- $d(\cdot, \cdot)$: SE(3) 上的 distance metric (通常 position L2 + orientation angle)
- $X^g \cdot X^*$: target grasp pose (在 base frame 下)
- $\widetilde{\mathbf{p}}_t$: $\widetilde{X}_t$ 的 positional component
- $\mathbf{p}_t$: Step 1 找到的 positional path
- $\delta_p$: 允许的偏差

约束 (8c) 是关键: 它保证 refinement 不会让 fingertip 偏离 Step 1 找到的 "constraint-exploiting" 路径太远。约束 (8b) 保证末端 pose 是目标 grasp。

---

## 五、Impedance Control Generation (Section IV-C)

有了 geometric path $\widetilde{X}_1, ..., \widetilde{X}_T$, 现在要生成一系列 **Cartesian impedance control commands** $\xi_0, ..., \xi_{T-1}$, 让 robot 在真实物理下 (有 contact) 仍然能跟随这条 path。

### Impedance control 基础 (Eq. 1-2)

Cartesian impedance control 在 task frame 上模拟一个 virtual spring-damper:
$$\mathbf{f}_t = K(\boldsymbol{\xi} - \mathbf{x}_t) - D\dot{\mathbf{x}}_t$$

变量:
- $\boldsymbol{\xi} \in SE(3)$: reference equilibrium pose (这是 control input, 我们要优化的东西)
- $\mathbf{x}_t \in \mathbb{R}^6$: 当前 task frame pose (position + roll-pitch-yaw)
- $K, D \in \mathbb{R}^{6 \times 6}$: stiffness 和 damping 矩阵 (diagonal)
- $\mathbf{f}_t \in \mathbb{R}^6$: virtual wrench (force + torque) 作用在 task frame

Robot dynamics (Eq. 2):
$$M(\mathbf{q}_t)\ddot{\mathbf{q}}_t + C(\mathbf{q}_t, \dot{\mathbf{q}}_t) = J(\mathbf{q}_t)^\top (\mathbf{f}_t + \mathbf{f}_{ext})$$

变量:
- $\mathbf{q}_t \in \mathbb{R}^N$: joint angles (N = 7 for Franka)
- $M(\mathbf{q}_t)$: joint space mass matrix $N \times N$
- $C(\mathbf{q}_t, \dot{\mathbf{q}}_t)$: Coriolis + centrifugal terms
- $J(\mathbf{q}_t) \in \mathbb{R}^{6 \times N}$: Jacobian
- $\mathbf{f}_{ext}$: 外部 contact wrench

### Approximate forward dynamics (Eq. 9-14)

把 joint space dynamics 转到 task space, 用 quasi-static 假设 ($\dot{J}\dot{q}$ 忽略, 因为 motion 慢):

$$\ddot{\mathbf{x}}_t \approx \Lambda(\mathbf{q}_t)^{-1} (\mathbf{f}_t + \mathbf{f}_{ext}) \tag{10}$$

其中 $\Lambda(\mathbf{q}_t) = J^{-\top} M J^{-1}$ 是 task-space inertia (operational space dynamics)。

Reference: Khatib 1987 "A Unified Approach for Motion and Force Control of Robot Manipulators" — https://khatib.stanford.edu/publications/Khatib_1987_RA.pdf

### External wrench estimation (Eq. 11-12)

这是 paper 的一个关键设计: **从 geometric path 上的 virtual penetration 估算 contact force**。

Eq. (11): 每个 surface point $\mathbf{p} \in \mathcal{P}^s$ 上的 local force:
$$\mathbf{f}(\mathbf{p}) = \mathbb{1}_{\{\phi(\widetilde{X}_t \cdot \mathbf{p}) < 0\}} \cdot k \cdot \phi(\widetilde{X}_t \cdot \mathbf{p}) \cdot \mathbf{n}$$

变量:
- $\mathbb{1}_{\{\phi < 0\}}$: indicator function, 只对 penetrating 的点生效
- $k$: virtual stiffness hyperparameter
- $\phi(\widetilde{X}_t \cdot \mathbf{p})$: signed distance, **负值** 表示 penetration depth
- $\mathbf{n} \in \mathbb{R}^3$: outward normal of gripper surface at point p

注意这里 $\phi$ 是负的, 所以 $k \cdot \phi \cdot \mathbf{n}$ 是沿 surface normal **向内** 的 force (环境 push gripper 出去)。

Eq. (12): 总 external wrench 是所有 penetrating points 的 force / torque 平均:
$$\mathbf{f}_{ext}(\widetilde{X}_t) = \frac{1}{Z} \begin{pmatrix} \widetilde{R}_t \cdot \sum_{\mathbf{p} \in \mathcal{P}^s} \mathbf{f}(\mathbf{p}) \\ \widetilde{R}_t \cdot \sum_{\mathbf{p} \in \mathcal{P}^s} \mathbf{p} \times \mathbf{f}(\mathbf{p}) \end{pmatrix}$$

变量:
- $Z = \sum \mathbb{1}_{\{\phi < 0\}}$: penetrating points 数量 (归一化)
- $\widetilde{R}_t \in \mathbb{R}^{3 \times 3}$: $\widetilde{X}_t$ 的 rotation 部分, 把 force 从 task frame 转到 base frame
- 上半部分: total force
- 下半部分: total torque (cross product 给出 moment arm)

**Intuition**: 这里的 $\mathbf{f}_{ext}$ 是一个 **非常粗糙** 的 estimate。它不需要精确, 因为 impedance control 本身就是 design 来 tolerate uncertainty 的。这个 estimate 只是给 optimization 一个 "方向感", 让生成的 ξ 知道大概要 push 多远、往什么方向 push 才能 compensate 掉 contact force。

### Forward dynamics simulation (Eq. 13-14)

把所有 piece 组装成 forward dynamics function $\Pi$:

给定 $(\mathbf{x}_t, \dot{\mathbf{x}}_t, \boldsymbol{\xi}_t)$, 通过积分得到 $(\mathbf{x}_{t+1}, \dot{\mathbf{x}}_{t+1})$:

$$\mathbf{x}_\tau = \mathbf{x}_t + \int_t^\tau \dot{\mathbf{x}}_s \, ds, \quad \dot{\mathbf{x}}_\tau = \dot{\mathbf{x}}_t + \int_t^\tau \ddot{\mathbf{x}}_s \, ds \tag{14a}$$
$$\ddot{\mathbf{x}}_\tau = \Lambda^{-1}(\mathbf{f}_\tau - \mathbf{f}_{ext}(\widetilde{X}_t)) \tag{14b}$$
$$\mathbf{f}_\tau = K(\boldsymbol{\xi}_t - \mathbf{x}_\tau) - D\dot{\mathbf{x}}_\tau \tag{14c}$$

数值积分用 2 ms 时间步, damping 设为 critical damping $D = 2\sqrt{K}$。Critical damping 是 control theory 里避免 oscillation 的标准选择。

### Optimization for ξ (Eq. 15)

最终, 优化问题:

$$\min_{\boldsymbol{\xi}_0, ..., \boldsymbol{\xi}_{T-1}} \quad \sum_{t=1}^{T} d(\widetilde{X}_t, X_t) \tag{15a}$$
$$\text{s.t.} \quad \|\dot{\mathbf{x}}_0\|^2 = 0, \quad \|\dot{\mathbf{x}}_T\|^2 = 0 \tag{15b}$$
$$\|\dot{\mathbf{x}}_t\|^2 \leq \delta_v, \quad \forall t = 1, ..., T-1 \tag{15c}$$
$$\mathbf{x}_{t+1}, \dot{\mathbf{x}}_{t+1} = \Pi(\mathbf{x}_t, \dot{\mathbf{x}}_t, \boldsymbol{\xi}_t), \quad \forall t = 0, ..., T-1 \tag{15d}$$

变量:
- $\boldsymbol{\xi}_t \in \mathbb{R}^6$: 第 t 个 waypoint 的 impedance reference pose (优化变量)
- $X_t$: 在 dynamics Π 下 predicted 的 pose
- $\widetilde{X}_t$: Section IV-B 规划的 desired geometric path
- $d(\cdot, \cdot)$: SE(3) distance
- $\delta_v$: velocity bound

约束:
- (15b): start 和 end 速度为 0
- (15c): 中间速度 bounded, 保证 safety
- (15d): dynamics 约束, 把 $\boldsymbol{\xi}_t$ 和 $\mathbf{x}_{t+1}$ 耦合起来

**关键 insight**: 这里把 approximate dynamics 当 **heuristic constraint** 用, 而不是 ground truth。真实 robot 有 perception error, 有 unmodeled friction, 有 mesh approximation error, 所以 Π 是不准的。但只要 Π 大致对, optimization 找到的 ξ 就会大致 correct, 然后 impedance control 的 compliance 会吸收掉残余的 error。这是一个 **model-based planning + model-free compliance** 的 hybrid 设计哲学。

Reference: Hogan 1985 "Impedance Control: An Approach to Manipulation" — http://asmedigitalcollection.asme.org/dscc1985/proceedings-abstract/DSCC1985-v1/80271/229391

---

## 六、实验结果

### Setup
- Robot: Franka Emika (7-DoF), 单臂和双臂两种 setup
- Vision: FoundationPose (6D pose estimation) + CNOS (segmentation), 单 RTX 3060
- Optimizer: CasADi + IPOPT (interior point method)
- 路径长度 T = 20
- SDF resolution Δ = 1 cm, ε = 1 cm
- δ_g 起始 0 mm, 步长 1 mm

Reference: 
- FoundationPose: https://foundationpose.github.io/
- CasADi: https://web.casadi.org/
- IPOPT: https://coin-or.github.io/Ipopt/

### Experiment A: Single object on table (Fig. 7 table)

9 个物体 (来自 YCB dataset, https://www.ycbbenchmarks.com/):
- Book, Cleanser Bottle, Mustard Bottle, Cracker Box, Timer, Clamp, Scissors, Bowl, Plate

每个物体 10 trials, 每 trial 最多 10 repetitions。

**单臂 setup** 结果:
| Metric | 值 |
|---|---|
| Overall success | 90 / 90 (100%) |
| Average reps to success | 1.4 ± 0.9 |
| Average total solve time | 3.60 ± 2.63 秒 |

**双臂 setup** 结果 (一个 arm 抓, 另一个 arm 挡):
| Metric | 值 |
|---|---|
| Overall success | 90 / 90 (100%) |
| Average reps | 1.6 ± 1.2 |
| Average solve time | 4.58 ± 3.66 秒 |

**关键 observations**:
1. **Cracker Box 和 Bowl 需要更多 reps** (2.7 和 1.0 reps 单臂)。原因: 它们的 thickness 接近 gripper 最大开口宽度, 要求 gripper orientation 精确对齐才能 insert。Cracker box 是空的容易 deform, bowl 有 narrow rim 在 mesh 里没建模。
2. **Solve time variance 大**, 因为 perception error 导致 object 被估计穿透 table, A* 要增大 δ_g 多次重试。
3. **双臂 setup 时间略长**, 但 success rate 仍然 100%。

### Experiment B: Tightly stacked objects (Fig. 8)

三个场景:
1. Tray 中抓 target book (单臂)
2. Shelf 中抓 target book (单臂)
3. Table 上叠放抓 target book (双臂)

这里要 **同时插入两个 finger** 进两个 gap, 难度更高。所有场景 100% success, average reps 略增 (因为更复杂的 contact geometry)。Solve time 4-7 秒, 仍然实用。

### Experiment C: Baseline comparison (Fig. 9)

Baseline 选的是 Wirnshofer 2018 (belief space planning for assembly)。作者只测了 3 个简单 case, 5 trials each, 因为 baseline 容易 fail 导致物体或另一只 arm 损坏。

Baseline 的问题:
1. 更 sensitive to modeling inaccuracy (因为它的 dynamics model 更精确, 但是 paper 用的是 approximate dynamics)
2. Random sampling-based motion generation 效率低, 通常 > 20 秒
3. Execution failure 多: finger 插不进 book gap, clamp 抓错位置

Reference: Wirnshofer et al. 2018 — https://ieeexplore.ieee.org/document/8460987

---

## 七、深度 Intuition: 为什么这个 framework works

1. **Collision-inclusive 的本质**: 传统 motion planning 把 collision 当 hard constraint, 这在 occluded grasping 场景下直接 fail (没有 collision-free path)。这篇 paper 把 collision 当 **soft cost**, 允许 path 穿透 environment, 但通过 impedance control 在执行时吸收这些 collision。

2. **Environmental constraint exploitation**: Human 在 blind grasping 场景下本能做的事是 "find a surface and slide along it"。Paper 用 Eq. (5a) 的 $\phi(\mathbf{p}_t)^2$ 项 explicitly 鼓励 fingertip 贴着 surface 走。这把 perception uncertainty (不知道 gap 精确位置) 转化为 contact constraint (surface 把 finger 引到 gap 里)。

3. **Manipulation funnel via repetitions**: 单次 execution 因为 uncertainty 可能 fail, 但因为 framework 不大幅改变 environment, retry 是 "exploit 同一个 funnel"。这是把 Mason 的 funnel idea 从空间维度扩展到时间维度 (repetitions)。

4. **Approximate dynamics as heuristic**: 这是 paper 最聪明的 design 之一。如果你想要精确 dynamics, 你需要精确 friction model, 精确 contact model, 这在真实世界很难。Paper 用一个粗糙的 virtual penetration force model (Eq. 11), 然后 **rely on impedance control 的 compliance 来 absorb residual error**。这种 "coarse model + robust controller" 哲学在 classic control 里很常见, 但在 manipulation planning 里用得不多。

5. **Two-stage path planning 的妙处**: 
   - Stage 1 (A* on positional): 在低维空间 (3D position) 找一个 rough guide path, 鼓励 surface contact
   - Stage 2 (CHOMP refinement on SE(3)): 在高维空间 refine, 最小化 gripper body collision, 但被 Stage 1 的 path "anchor" 住

   这种 coarse-to-fine 在 motion planning 里是好 practice, 因为直接在高维空间做 constraint-exploiting optimization 容易 stuck 在 local minimum (gripper 可能 rotate 到 weird orientation 来 "偷懒" 减小 collision cost)。

---

## 八、Limitations 和潜在延伸

Paper 自己没有详细讨论 limitation, 但从 design 看出:

1. **Slow motion 假设**: Eq. (10) 忽略 $\dot{J}\dot{q}$ 项, 这在 dynamic motion (比如 swing-up, 抛物体) 下会 break。但 contact-rich slow manipulation 这个 assumption 一般成立。

2. **Object configuration 不能大幅改变**: 这是 framework 的前提。如果 objects 是 loosely packed, 抓 book 的时候其他 book 会倒, 整个 funnel 就 collapse 了。

3. **Geometric model 必需**: 需要 gripper mesh + object mesh。对 unknown objects, 需要先 reconstruct (e.g., 用 FoundationPose 或 neural SDF)。但对完全 novel 物体 (e.g., 软体, 透明物体) 仍然有挑战。

4. **Open-loop impedance control**: ξ sequence 是 open-loop commanded, 没有 feedback 调整 ξ。如果中间 contact 状态严重偏离 plan, 只能 early terminate (Algorithm 1 的 EXECUTE 函数) 然后 retry。可以想象一个 **closed-loop 版本**, 用 force/torque feedback 在线 replan ξ。

5. **Single fingertip task frame**: 论文把 task frame 定义在 single fingertip。对 dual-finger insertion (Experiment B), 其实 task frame 还是在一个 fingertip, 另一个 finger 靠 gripper 的 symmetric geometry "顺带" 进去。对 asymmetric dual-finger 任务可能需要 extension。

---

## 九、相关工作脉络

- **CHOMP** (Ratliff 2009): gradient-based motion planning with smooth collision cost, paper 的 Eq. (6) 直接用 — https://www.ri.cmu.edu/pub_files/2009/5/icra09-chomp.pdf
- **Mason's manipulation funnel**: funnel idea 的源头 — https://www.cs.cmu.edu/~mma/mason-pubs/Mason86.pdf
- **Eppner & Brock "Environmental constraints in grasping"**: 同一思想在 grasping 上的应用 — https://www.researchgate.net/publication/224528889
- **Khatib operational space dynamics**: task space dynamics formulation — https://khatib.stanford.edu/publications/Khatib_1987_RA.pdf
- **Hogan impedance control**: impedance control 的奠基性工作 — Hogan 1985
- **FoundationPose**: 6D pose estimator — https://foundationpose.github.io/
- **Wirnshofer 2018**: baseline, belief space planning for assembly — https://ieeexplore.ieee.org/document/8460987
- **CHOMP variant + contact-rich**: 像是 Suomalainen 2022 survey "A survey of robot manipulation in contact" — https://arxiv.org/abs/2202.10181

---

## 十、总结

这篇 paper 的 elegance 在于: 它把一个看似 hard 的问题 (occluded grasping in tight space) 分解为三个 mutually compensating 的 components:
1. **Collision-inclusive path planning** 解决 "没有 collision-free path" 的问题
2. **Impedance control** 解决 "collision 引入的 uncertainty" 问题
3. **Manipulation funnel via repetitions** 解决 "单次 execution 因为 uncertainty fail" 的问题

每个 component 单独都不是全新 idea, 但组合在一起对 occluded grasping 这个 specific problem 形成了一个 clean, interpretable, effective 的 framework。这种 design 思路 — 把 uncertainty 拆开, 每一层用一个 tool 处理一种 uncertainty — 是 robotics 里很值得学习的 pattern。

对做 imitation learning / RL 的读者, 这篇 paper 提供了一个好的 "model-based baseline" 视角: 当你想要 learn contact-rich policy 时, 一个 hand-designed 的 model-based framework 已经能达到 100% success rate on 9 个 YCB objects, average 1.4 reps。这意味着任何 learning-based 方法都要至少 beat 这个 baseline, 才能 claim 学到了有意义的东西。而且 model-based framework 的 interpretable breakdown (perception → planning → control → retry) 是 debugging learning policy 时很好的 reference。
