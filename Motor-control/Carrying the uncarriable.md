---
source_pdf: Carrying the uncarriable.pdf
paper_sha256: ff25ffa90cec6cda878cb1f6b7639369015419c09cc8a7cd7cef4669d227575a
processed_at: '2026-08-03T15:00:39-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话说清楚这 paper 干啥

想象你和朋友搬一个大柜子，你推一边，朋友推另一边，你们**不用说话**也能配合得很好 —— 因为你推的时候，柜子是 rigid 的，力量会"告诉"另一边的人该往哪走。这就是传统 admittance controller 干的事，靠 **force 信号** 传递意图。

但现在换个场景：你拿一根**软绳子**挂个箱子，绳子是松的。你往前走，绳子没绷紧之前，另一头的人**完全感觉不到**你在动。Force 信号失效了。传统方法就傻眼了。

这 paper 的核心 idea 特别简单：**既然 force 传不过去，那就用眼睛看！** 装个 MoCap 测人手在动，直接告诉机器人"人在动，你跟着动"。然后用一个公式自动判断"什么时候 force 可信、什么时候该用 MoCap"。

就这么个事。

---

## 这 paper 解决了哪三个痛点

### 痛点 1: Object 太大太重，一个 robot 抬不动

12kg 的箱子加 1.1 米宽，一个 Franka Panda arm 根本 grasp 不住。所以要 multi-robot。

### 痛点 2: Object 是 deformable 的，haptic 信号失效

用 lifting straps（吊带）挂箱子，strap 松弛时 force 根本传不过去。传统 admittance controller 等于瞎了。

### 痛点 3: 多 robot 之间不想搞复杂通信

如果两个 robot 之间要高速通信协调，framework 就复杂了，工业部署也麻烦。所以希望 **decentralized**，每个 robot 自己跟 human 交互，不需要 robot-to-robot 通信。

---

## 核心公式 (Eq. 2) 到底在算啥

$$
\alpha^{R_i}(t) = 1 - \frac{\int_{t_c - W_l}^{t_c} \| \mathbf{v}_{adm}^{R_i}(t) \| \, dt}{\int_{t_c - W_l}^{t_c} \| \mathbf{v}_h(t) \| \, dt + \epsilon}
$$

翻译成人话：

分子 = 过去 0.5 秒里，robot 通过 force 自己"想动"多少
分母 = 过去 0.5 秒里，人的手实际动了多少

如果两者差不多（分子 ≈ 分母），说明 force 信号工作正常，object 是 rigid 的，**α ≈ 0**，trust force。

如果分子远小于分母（force 传不过来，但人手在动），说明 object 是 deformable 的，**α ≈ 1**，trust MoCap。

这个 ratio 本质上就是 **"haptic channel 的传输效率"** 的度量。非常 elegant 的物理直觉。

---

## Eq. (3) 为啥是加性的，不是 convex combination

$$
\mathbf{v}_d^{R_i}(t) = \mathbf{v}_{adm}^{R_i}(t) + \alpha^{R_i}(t)\, \mathbf{v}_h(t)
$$

你可能会问：为啥不是 $v_d = (1-\alpha) v_{adm} + \alpha v_h$？这样更"对称"啊？

仔细想想就明白了：

| Object | $v_{adm}$ | $\alpha$ | $v_d$ |
|---|---|---|---|
| Rigid | ≈ $v_h$ | ≈ 0 | ≈ $v_{adm}$ ≈ $v_h$ ✓ |
| Deformable | ≈ 0 | ≈ 1 | ≈ $v_h$ ✓ |

Rigid 情况下 $v_{adm}$ 已经等于 $v_h$ 了（force 完整传递），所以加不加 $v_h$ 都一样。Deformable 情况下 $v_{adm}$ 是 0，加上 $\alpha \cdot v_h$ 就是 $v_h$。这两种情况都 work。

**关键 trick**：这种 asymmetric 设计避免了 rigid 情况下 MoCap 的 noise 污染 reference（因为 α≈0 把它压住了）。如果用 symmetric convex combination，rigid 情况下 MoCap noise 会被放大。这是个很 subtle 但 smart 的 design choice。

---

## 实验设计为啥很 clever

实验路径有三个 sub-movements：backwards、sideways、down-up。

为啥 down-up 最 challenging？因为 strap 在 vertical motion 时**永远松弛**（重力让 strap 一直被拉直，但 box 上下移动时 strap 长度不变所以不会绷紧）—— force 完全传不过去。

而 backwards/sideways 时，strap 至少在 horizontal direction 会逐渐拉紧，force 能部分传递。

所以这个 path 设计**故意包含了三种 deformability regime**：
- Backwards：partial deformability
- Sideways：partial deformability  
- Down-up：extreme deformability（admittance 完全失效）

这就是为啥 Fig. 4(b) 里 down-up 阶段机器人完全不动 —— baseline 直接 fail 了。而 ACI 因为 fallback 到 MoCap，依然能跟踪。

---

## 为啥选 Kairos + MOCA 这种异构组合

这是这 paper 的另一个亮点：**framework 不挑 robot**。

- **MOCA**：Robotnik omni-base + Franka Panda (7-DoF **torque-controlled**)
- **Kairos**：Robotnik omni-base + UR16e (6-DoF **position-controlled**)

一个用 torque controller，一个用 position controller。一个用 inverse dynamics (Eq. 5-6)，一个用 HQP/CLIK (Eq. 7)。完全不同的 control paradigm。

但 ACI 输出的只是 desired pose 和 twist（运动学层面），和 robot 内部用什么 controller 解耦。所以这个 framework 能 plug 进任何 robot —— 这点对工业应用很重要，因为工厂里啥 robot 都有。

---

## 关键结果 Fig. 6 在说什么

Fig. 6(a) 显示：
- 用 strap 搬箱子时 α 接近 1（deformable，trust MoCap）
- 直接 grasp 柜子时 α 接近 0（rigid，trust admittance）

这说明公式 (2) **自动识别了 deformability**，不需要事先告诉系统"这是 deformable object"。这是 framework 的核心价值 —— **deformation-agnostic**。

Fig. 6(b) 更重要：
- 用 strap 搬箱子时，admittance baseline 需要人施加更大的 force 才能驱动 robot
- ACI 显著降低了所需 force

这是 **ergonomic improvement** —— 减轻工人的 physical load。工业场景里这意味着减少腰背损伤，巨大的实际价值。

---

## 和 ML 的联系（Karpathy 应该感兴趣的部分）

这个 α 公式本质上就是一个 **hand-crafted gating mechanism**，结构上和 GRU 的 update gate 几乎同构：

GRU: $h_t = (1-z) \odot h_{t-1} + z \odot \tilde{h}_t$
ACI: $v_d = v_{adm} + \alpha \cdot v_h$

两者都是用 gating coefficient 决定两个 information source 的 mix ratio。区别是：
- GRU 的 gate 是 learned
- ACI 的 gate 是 physics-informed hand-crafted

这给我们一个 thought：如果把这个 framework 放到 end-to-end learning pipeline 里，α 可以是 NN 输出，input 是历史 force/torque + MoCap + vision + ...，可能更 robust。但 hand-crafted 版本已经 work 得很好，说明 **physics prior 在 narrow domain 仍然很 powerful**，不一定要 learn。

另一个联系：这个 framework 本质是 **sensor fusion with reliability estimation**。和 autonomous driving 里的 radar+lidar+camera fusion 是一类问题。confidence estimation 不一定要 learned，sensor 之间的 agreement/disagreement 本身就是 reliability 的 signal。

---

## 我觉得这 paper 的核心贡献

1. **公式 (2) 是核心创新**：用一个 sliding window velocity ratio 自动推断 deformability，这是 physics-informed 但 elegant 的 design。
2. **Asymmetric gating (Eq. 3)**：避免了 rigid 情况下 MoCap noise 污染，subtle 但重要。
3. **Decentralized + heterogeneous**：每个 robot 独立运行 ACI，robot 之间不通信，但通过 human 间接协调。这是 elegant 的 system design。
4. **Real-world validation**：12kg 箱子加 strap，不是 toy experiment，是工业级别的 challenge。

---

## 这 paper 的 limitation

1. **只有 2 个 subjects**：statistical power 不够，但 qualitative 趋势很清楚。
2. **MoCap 依赖**：Xsens IMU 有 drift，长时间任务可能累积 error。换 vision-based MoCap（OptiTrack/Vicon）可以解决但需要 external infrastructure。
3. **α 是 scalar**：各向同性 deformability 假设。实际 strap 在不同方向 deformability 不同，应该用 $\alpha \in \mathbb{R}^3$ per-axis。这是 obvious next step。
4. **N=2 robots**：scaling 到 N>2 没做。理论支持但实际 emergent dynamics 未知。
5. **没有 obstacle avoidance**：作者自己承认，工业部署必须加。

---

## 一句话总结

**这 paper 用一个简单 velocity ratio 公式自动判断 object 是不是 deformable，并相应地在 force-based admittance controller 和 MoCap-based tracking 之间做 soft gating，让多个异构 robot 不用互相通信就能和人一起搬运又大又重又软的物体。**

核心 insight：**当 force channel 失效时，用 vision channel 做 fallback**，并用 sensor agreement 自动判断何时该 fallback。这个思路在 robotics 之外（autonomous driving、sensor fusion）也通用。

---

## References

- Paper video: https://youtu.be/Q3sA6YzTaaE
- HRII Lab (Ajoudani): https://hrii.iit.it/
- Franka Panda: https://www.franka.de/
- UR16e: https://www.universal-robots.com/products/ur16-robot/
- Robotnik SUMMIT-XL STEEL: https://robotnik.es/robotics-robots-mobile-robots/summit-xl-steel-en/
- Pisa/IIT SoftHand: https://www.qbrobotics.com/products/qbsofthand-research/
- Xsens MVN: https://www.movella.com/products/motion-capture/xsens-mvn-analyze
- Khatib 1995 Operational Space Formulation: https://journals.sagepub.com/doi/10.1177/027836499501400102
- Sirintuna et al. IROS 2022 (前期工作): https://hrii.iit.it/publications/

如果你想再深挖某个点（比如为啥用 admittance 不用 impedance、HQP 怎么解、SoftHand 的 underactuation 怎么 grasp 不规则物体、或这个 framework 怎么用 RL 替换 hand-crafted α），告诉我哪个方向，我再展开。

---

# Carrying the Uncarriable: 一篇关于 deformation-agnostic 的 human-multi-robot 协作搬运 paper 详解

这篇 paper 由意大利 IIT (Istituto Italiano di Tecnologia) 的 Human-Robot Interfaces and Physical Interaction (HRII) lab 的 Arash Ajoudani group 发表（Sirintuna, Ozdamar, Ajoudani），是其 IROS 2022 工作 [2] 的延伸。配套 demo video 在 https://youtu.be/Q3sA6YzTaaE 。HRII lab 主页：https://hrii.iit.it/ 。

---

## 1. 核心问题 (Problem statement)

这篇 paper 解决一个很具体的 pHRI (physical Human-Robot Interaction) 痛点：

- 传统的 co-manipulation 文献假设 jointly-held object 是 **rigid** —— 这样 wrench (force + moment) 才能从 human 端经由 object **完整 rigid transformation** 传到 robot end-effector。
- 一旦 object 是 **deformable**（比如用 forklift lifting straps 挂着的 bulky box），haptic channel 就**部分失效**：human 推/拉的力量无法通过 slack 的 strap 传递到 robot 端，admittance/impedance controller 就抓瞎 —— 必须等 strap 被拉紧到一定程度才能感知到 wrench，造成响应延迟甚至完全失响应。
- 另外一个大件物体（"uncarriable"）单 robot payload 不够，需要 multi-robot team，但**多 robot 之间又希望 decentralized，不要求严格 inter-robot communication**。

这篇 paper 提出的解法叫 **Adaptive Collaborative Interface (ACI)**：把 **haptic information**（F/T sensor 测的 force）和 **exteroceptive motion information**（MoCap 测的 human hand velocity）两种信号通过一个 **adaptive index α** 融合起来，α 自动反映 object 的 deformability 程度。deformable 时 α→1，主要 trust MoCap；rigid 时 α→0，主要 trust admittance force loop。

---

## 2. 系统架构（Fig. 2 解析）

整张 high-level scheme 是：

```
        Human  ─── MoCap (Xsens) ─── v_h(t) ──┐
                                              │
   Object ─ F/T sensor ─ F_H^{R_i}(t) ──► [ Admittance Controller ]
                                                      │
                                                      ▼ v_adm^{R_i}(t)
                                              ┌──────┴───────────┐
                                              │  Reference       │
                                              │  Generator       │ ◄── α^{R_i}(t)
                                              │  (per robot R_i) │
                                              └──────┬───────────┘
                                                     ▼ x_d^{R_i}, ẋ_d^{R_i}
                                          ┌──────────────────────────┐
                                          │ Whole-body Loco-Manip.   │
                                          │ Controller (per robot)   │
                                          └──────────────────────────┘
```

每个 robot 独立运行一个 ACI 实例，所以是 **decentralized**。Robot 之间没有显式 communication，全靠 human 作为 "shared medium" 间接协调 —— 这是这个 framework 的另一个 elegant 性质。

---

## 3. 公式逐个拆解

### 3.1 Eq. (1) Admittance Controller

$$
V_{adm}^{R_i}(s) = \frac{F_H^{R_i}(s)}{M_{adm}^{R_i} s + D_{adm}^{R_i}}
$$

变量与上标/下标说明：
- $V_{adm}^{R_i}(s) \in \mathbb{R}^3$：第 i 个 robot 的 **admittance reference translational velocity** 在 Laplace 域的表示。上标 $R_i$ 标识 robot id，下标 $adm$ = admittance-derived。
- $F_H^{R_i}(s) \in \mathbb{R}^3$：第 i 个 robot end-effector 上 F/T sensor 测得的 force（来自经由 object 传来的 human 施加的力），Laplace 域。
- $M_{adm}^{R_i} \in \mathbb{R}^{3\times3}$：desired virtual mass matrix（实验里 $diag\{4,4,4\}$），决定惯性 "感觉"。
- $D_{adm}^{R_i} \in \mathbb{R}^{3\times3}$：desired virtual damping matrix（实验里 $diag\{45,45,45\}$），决定速度衰减。
- $s$：Laplace variable。
- $i$：robot id（1 = MOCA, 2 = Kairos）。

**Intuition**：这就是经典的 1 阶 mass-damper admittance law —— force 输入 -> velocity 输出。是一个低通滤波器，截止频率 $\omega_c = D_{adm}/M_{adm}$，实验里大约 11.25 rad/s ≈ 1.8 Hz。这意味着 admittance loop 对高于 ~2Hz 的扰动不响应，避免 oscillation。

### 3.2 Eq. (2) Adaptive Index α（核心创新点）

$$
\alpha^{R_i}(t) = 1 - \frac{\int_{t_c - W_l}^{t_c} \| \mathbf{v}_{adm}^{R_i}(t) \| \, dt}{\int_{t_c - W_l}^{t_c} \| \mathbf{v}_h(t) \| \, dt + \epsilon}
$$

变量：
- $\alpha^{R_i}(t) \in [0,1]$：第 i 个 robot 的 adaptive index。下标 R_i 表示这是 per-robot 的（因为每个 robot 感受到的 force 不一样，deformability 的"局部感知"可能不同 —— 这是 framework 灵活性的来源）。
- $t_c$：current time。
- $W_l$：sliding time window length（实验里 0.5s）。
- $\epsilon$：small constant 防止除零（典型的 numerical stability trick）。
- 分子：sliding window 内 admittance reference velocity 的 magnitude 累积积分。
- 分母：sliding window 内 human hand velocity magnitude 的累积积分（+ ε）。

**这个公式的 intuition 极其重要**：分子 / 分母 实际上是一个 "energy ratio" / "displacement ratio"，回答了 "在最近 0.5 秒里，robot 通过 haptic feedback 自己产生 motion 的量 / human 实际产生 motion 的量"。这个 ratio 反映了 haptic channel 的"传导效率"：
- 当 object **rigid**：human 的 force 经由 object 完整传递 → $v_{adm} \approx v_h$ → ratio ≈ 1 → $\alpha \to 0$（haptic feedback 已经足够，不需要 fallback）。
- 当 object **highly deformable**：force 几乎传不过来 → $v_{adm} \approx 0$ → ratio ≈ 0 → $\alpha \to 1$（haptic 信息失效，必须 trust MoCap）。
- 当 object **partially deformable**：中间值，两种信号按比例 blend。

我作为 reader 的第一反应：这个公式像极了 ML 里 **GRU 的 update gate** 或者一个 **soft attention/gating mechanism** —— 都是 $\alpha \cdot \text{source}_A + (1-\alpha) \cdot \text{source}_B$ 的形式，权重由一个信号 ratio 决定。区别是这个 gating 是 **hand-crafted**（基于物理直觉），不是 learned。这个其实是相当 elegant 的 physics-informed gating。如果你把它放在不同的 framework 里看，它在做 **sensor fusion with confidence-weighted gating**，confidence 由传感器一致性推断。

为什么 per-robot 单独算 α？因为 deformable object 在不同 robot 端的 coupling 可能不同（比如 strap 一边紧一边松），这给了 framework 更细的 adaptivity。

### 3.3 Eq. (3) Reference Generator 的最终 velocity

$$
\mathbf{v}_d^{R_i}(t) = \mathbf{v}_{adm}^{R_i}(t) + \alpha^{R_i}(t)\, \mathbf{v}_h(t)
$$

注意这是**加性**的，不是 convex combination（没有 $1-\alpha$ 在第一项上）。这有点 surprising，仔细想其实是合理的：

| Object 情况 | $v_{adm}$ | $\alpha$ | $v_d$ | 信号主导 |
|---|---|---|---|---|
| Highly deformable | $\approx 0$ | $\approx 1$ | $\approx v_h$ | MoCap 主导 |
| Non-deformable (rigid) | $\approx v_h$ | $\approx 0$ | $\approx v_{adm} \approx v_h$ | Admittance 主导（也等于 v_h） |
| Partially deformable | $v_{adm}$ | $\alpha(t)$ | $v_{adm} + \alpha v_h$ | Blend |

**关键 intuition**：rigid 情况下 $v_{adm} \approx v_h$（因为 force 完整传递），所以两种信号一致，$v_d \approx v_h$。这种设计相当于一个 **asymmetric gating**：admittance 信号作为 "基础"（无脑加进去），MoCap 信号只在 deformability 高时才"补救"。这避免了 rigid 情况下 v_h 噪声直接污染 v_d（因为此时 α≈0）。作者在 Table I 里把这个表清晰列出，是一个非常好的 sanity check。

接下来 pose 通过积分得到：$\mathbf{x}_d^{R_i}(t) = \int_0^t \dot{\mathbf{x}}_d^{R_i}(t)\, dt$，其中 $\dot{\mathbf{x}}_d^{R_i}(t) = [\mathbf{v}_d^{R_i}(t)^T, \mathbf{0}^T]^T$。注意 angular velocity 部分 zero（twist 的 rotation 部分不主动 setpoint），意味着 robot 只跟踪 translational motion，不主动施加 rotation —— 这对于搬抬重物是合理的（rotation 应由 human 决定，robot 跟随）。

### 3.4 Eq. (4) MOCA 的 Whole-Body Decoupled Dynamics

$$
\begin{bmatrix} M_b & 0 \\ 0 & M_a(q_a) \end{bmatrix} \begin{bmatrix} \ddot{q}_b \\ \ddot{q}_a \end{bmatrix} + \begin{bmatrix} D_b & 0 \\ 0 & C_a(q_a, \dot{q}_a) \end{bmatrix} \begin{bmatrix} \dot{q}_b \\ \dot{q}_a \end{bmatrix} + \begin{bmatrix} 0 \\ g_a(q_a) \end{bmatrix} = \begin{bmatrix} \tau_b \\ \tau_a \end{bmatrix} + \begin{bmatrix} \tau_{b,ext} \\ \tau_{a,ext} \end{bmatrix}
$$

变量说明：
- $M_b \in \mathbb{R}^{n_b \times n_b}$：mobile base 的 virtual inertia（这里 $n_b = 3$，对应 omni-base 的 $x, y, \theta$）。
- $M_a(q_a) \in \mathbb{R}^{n_a \times n_a}$：arm 的 inertia matrix（依赖 joint config，$n_a = 7$ for Panda）。
- $D_b \in \mathbb{R}^{n_b \times n_b}$：base 的 virtual damping。
- $C_a(q_a, \dot{q}_a) \in \mathbb{R}^{n_a}$：arm 的 Coriolis + centrifugal 项。
- $g_a(q_a) \in \mathbb{R}^{n_a}$：arm 的 gravity vector。
- $q_b, \dot{q}_b, \ddot{q}_b$：base 的 joint position/velocity/acceleration（对于 omni-base 实际是 wheel-equivalent velocity 的 representation）。
- $q_a, \dot{q}_a, \ddot{q}_a$：arm 的 joint position/velocity/acceleration。
- $\tau_b, \tau_a$：commanded joint torques（对于 base 是 virtual torque, omni-base 通常 velocity-controlled 所以这个是抽象层）。
- $\tau_{b,ext}, \tau_{a,ext}$：external torques（含 human/object 反作用力）。

**Intuition**：这是一个 **block-diagonal decoupling** —— base 和 arm 在 dynamics 层面分开处理。Base 用 virtual inertia + damping 形式（其实在底层是 velocity controlled, 这里把 base 抽象成"等效"二阶系统），arm 用 full rigid body dynamics。这种 decoupling 在 mobile manipulator 控制里是常见 simplification，避免了 base-arm coupling 动力学的复杂处理。Base 那块的 damping $D_b$（实验里是 $10 M_b$）相当于一个 $10\text{ rad/s}$ 的截止频率，确保 base 运动平滑、不 jerky。

### 3.5 Eq. (5) & Eq. (6) Prioritized Weighted Inverse Dynamics

$$
\min_{\tau_c} \frac{1}{2} \|\tau_c - \tau_0\|_W^2 \quad \text{s.t.} \quad \bar{J}^T \tau_c = F
$$

解：
$$
\tau_c = W^{-1} M^{-1} J^T \Lambda_W \Lambda^{-1} F + (I - W^{-1} M^{-1} J^T \Lambda_W J M^{-1}) \tau_0
$$

变量：
- $\tau_c$：优化变量，全 robot 的 commanded joint torques（向量长度 $n_a + n_b = 10$ for MOCA）。
- $\tau_0$：null-space desired torque（用于让 robot 维持 default pose）。
- $W$：positive definite weighting matrix（控制 base 和 arm 的"任务分配偏好"）。论文里 $W = H^T M^{-1} H$，$H$ 是对角 tuning matrix —— 这就是 "weighted" 二字的来源。
- $\bar{J}^T = (J M^{-1} J^T)^{-1} J M^{-1}$：dynamically consistent pseudo-inverse（Khatib 1995，见 https://I-cite）。
- $F = D_d(\dot{x}_d - \dot{x}) + K_d(x_d - x)$：Cartesian impedance 的 operational force。
- $\Lambda = (J M^{-1} J^T)^{-1}$：Cartesian inertia。
- $\Lambda_W = J^{-T} M W M J^{-1}$：weighted Cartesian inertia。

**Intuition**：这是 Oussama Khatib 的 **Operational Space Formulation** (1995, https://journals.sagepub.com/doi/10.1177/027836499501400102) 的 weighted extension。核心思路：
- 在 joint torque 空间里，无数个 $\tau_c$ 能产生同一个 Cartesian force $F$（redundancy）。
- 用 weighting matrix $W$ 决定这个 redundancy 怎么分配 —— 大 $W$ 对应的 joint 会被 "penalize"，更倾向于用其他 joint。
- $\tau_0$ 项让 robot 在不破坏 Cartesian task 的前提下尽量靠近 default posture（保护 arm joint limit）。
- 第一项（task torque）+ 第二项（null-space projection）= 标准 projection-based redundancy resolution 公式。

**为什么 "weighted"？** 实验里 $H = I$，但 $W$ 可以根据 task 调整 base vs. arm 的"贡献偏好"。比如，如果想要 arm 更多承担 motion（精度高），可以让 arm 对应的 $W$ 元素小（penalty 小）；反之想要 base 多动（payload 大、运动范围广），就让 base 的 $W$ 小。这篇 paper 用 $H = I$ 是 simplification，但 framework 留了这个 knob。前期工作 Lamon et al. 2020 提到不同 $H$ 实现 different mobility modes。这个思路其实和 Loco-Manipulation 的现代 controller design 一脉相承 —— see https://ieeexplore.ieee.org/document/9197113 。

### 3.6 Eq. (7) Kairos 的 HQP / Closed-Loop IK

$$
\mathcal{L}_1 = \|\dot{x}_d + K(x_d - x) - J\dot{q}\|_{W_1}^2 + \|k\dot{q}\|_{W_2}^2
$$

变量：
- $\dot{q} \in \mathbb{R}^{n_a+n_b}$：优化变量，全 robot joint velocities（$n_a = 6$ for UR16e, $n_b = 3$ → 长度 9）。
- $J \in \mathbb{R}^{6 \times 9}$：whole-body Jacobian。
- $x \in \mathbb{R}^6$：current end-effector pose。
- $K \in \mathbb{R}^{6 \times 6}$：反馈 gain matrix（实验里 $diag\{0.1, 0.1, 0.1, 0.01, 0.01, 0.01\}$，注意 rotation 部分比 translation 小 10x，因为 UR16e position-controlled 对 rotation 跟踪弱）。
- $W_1, W_2$：任务项和 regularizer 的权重。
- $k > 0$：damping factor（动态，按 manipulability index 调）。

第二任务：$\mathcal{L}_2 = \|q_0 - q\|_{W_3}^2$，让 arm 维持 default config。$W_3 = diag\{\mathbf{0}_{n_b}, \mathbf{1}_{n_a}\}$ —— 只对 arm 加 constraint，对 base 不约束（base 自由分配 motion）。

**Intuition**：这是 **Closed-Loop Inverse Kinematics (CLIK)** + **Hierarchical QP**。区别于 MOCA 的 inverse dynamics（torque level），Kairos 是 position-controlled arm，所以直接在 velocity level 做 IK。公式里的 $\dot{x}_d + K(x_d - x)$ 是 **feedforward + feedback** —— $\dot{x}_d$ 是 desired velocity 的前馈，$K(x_d - x)$ 是 pose error 的 feedback。这种结构在 task-space tracking 控制中叫 "velocity-level reference" —— see Siciliano et al. 的 robotics textbook。

Damping factor $k$ 是经典 **DLS (Damped Least Squares)** trick，见 Wampler 1986, Chiaverini et al. 1992。当 manipulability 低（arm 接近 singularity）时，$k$ 增大避免 joint velocity 爆炸。这是 IK 数值稳定性的标准做法 —— see https://ieeexplore.ieee.org/document/1088303 。

### 3.7 Eq. (8) Alignment Metric

$$
D_{AM}^{*} = \frac{\int_{t_s}^{t_e} \| R(t)^* \| \, dt}{t_e - t_s}
$$

变量：
- $D_{AM}^*$：alignment metric。上标 $*$ 是 $x$, $y$, $z$ 三个分量（per-axis）。
- $R(t)$：human 和 robot end-effector 之间的相对位置 difference vector（current vs. initial alignment）。
- $t_s, t_e$：实验起始和终止时间。
- $r_{cee}, r_{chh}$：current end-effector position, current human hand position。
- $r_{see}, r_{shh}$：starting end-effector position, starting human hand position。

具体来说 $R(t)$ 定义为：当前 human 和 robot 的相对位置 - 起始 human 和 robot 的相对位置。如果 robot 完美跟随 human 保持初始 relative geometry，$R(t) = 0$，$D_{AM} = 0$（ideal）。任何偏离都会让 $D_{AM}$ 增大。

**Intuition**：这个 metric 本质是 **时间平均的 relative pose drift**。它是为 deformable object 设计的 —— 因为 deformable 时不能直接比 absolute pose，但可以比 human-robot 的 relative geometry 是否稳定。这个 metric design 很 elegant。

---

## 4. Robotic Platforms 详解

### 4.1 MOCA (MObile Collaborative robotic Assistant)

- **Mobile base**：Robotnik SUMMIT-XL STEEL，omni-directional（4 wheel mec 或类似，全向运动，3-DoF: $x, y, \theta$）。Link: https://robotnik.es/robotics-robots-mobile-robots/summit-xl/
- **Arm**：Franka Emika Panda，7-DoF torque-controlled。这是我最熟的 arm 之一，Franka Control Interface (FCI) 1kHz torque loop，每 joint 都有 torque sensor。Link: https://www.franka.de/
- **Hand**：Pisa/IIT SoftHand (underactuated, 5 fingers, 1 motor adaptive grasp)。Link: https://www.qbrobotics.com/products/qbsofthand-research/
- **F/T sensor**：在 arm flange 和 SoftHand 之间，额外加一个 6-axis F/T sensor（应该是 ATI 或类似）。这是为了精确测量 human 通过 object 传来的 wrench，不依赖 Franka 内部 torque 估计。

### 4.2 Kairos

- **Mobile base**：同样 Robotnik SUMMIT-XL STEEL。
- **Arm**：Universal Robots UR16e，6-DoF **position-controlled**，16kg payload。UR 系列是 position-controlled，不是 torque-controlled —— 这就解释了为什么 Kairos 用 HQP/CLIK 而不是 inverse dynamics。Link: https://www.universal-robots.com/products/ur16-robot/
- **Hand**：Pisa/IIT SoftHand。
- **F/T sensor**：同样附加在 flange。

### 4.3 MoCap

**Xsens MVN**，17 IMUs，全身穿戴。Inertial-based MoCap。Link: https://www.movella.com/products/motion-capture/xsens-mvn-analyze

- 优点：不需要 camera、indoor 完全可用、抗 occlusion。
- 缺点：drift（需要 calibration）、对 ferromagnetic 环境敏感。
- 测的是 human hand velocity $v_h(t)$。
- Paper 说可以替换成 vision-based MoCap（如 OptiTrack, Vicon），framework 是 MoCap-agnostic 的。

---

## 5. 实验设计

### 5.1 两个场景

**Scenario A: Bulky Box on Forklift Moving Straps (FMS)**
- 12 kg box，$110 \times 90 \times 120$ cm
- 商用 lifting straps，交叉挂在 human shoulder 和 robot SoftHand 上
- 这是 **deformable** scenario：strap 不紧绷时 force 传不过去
- Box 本身 rigid，但 strap 是 compliant coupling

**Scenario B: Rigid Closet**
- 6 kg wooden closet，$80 \times 30 \times 170$ cm
- 直接 SoftHand grasp
- 这是 **rigid** scenario：force 完整传递

### 5.2 Path 设计

三个 sub-movements：
1. **Backwards**：~120 cm
2. **Sideways**：~80 cm
3. **Down-up**：~20 cm 下降 + 20 cm 上升

这个 path 设计很 careful —— 每个 sub-movement 对 strap 的 deformation 影响不同：
- Backwards / Sideways：strap 沿 horizontal 方向会被拉伸，能传部分 force
- Down-up：strap slack 增加（vertical motion 时 strap 不会绷紧），几乎完全传不动 force —— 这是最 challenging 的部分

### 5.3 Baseline

只用 admittance controller（不 fusion MoCap），即 Eq. (1) 直接输出作为 desired velocity。

### 5.4 Controller 参数

| 参数 | 值 |
|---|---|
| $M_{adm}$ | $diag\{4,4,4\}$ |
| $D_{adm}$ | $diag\{45,45,45\}$ |
| $W_l$ | 0.5 s |
| MOCA $K_d$ | $diag\{200,200,200,30,30,30\}$（trans 200, rot 30，rot 弱 stiffness 让 rotation 跟随 human） |
| MOCA $D_d$ | $2\xi K_d^{1/2}, \xi = 0.7$ (critical damping-ish) |
| MOCA $M_b$ | $diag\{105, 105, 210\}$ (z 更高，避免 base 跳) |
| MOCA $D_b$ | $10 M_b$ |
| MOCA $K_0$ | $diag\{50 \cdot \mathbf{1}_{10}\}$ (null-space joint stiffness) |
| Kairos $K$ | $diag\{0.1, 0.1, 0.1, 0.01, 0.01, 0.01\}$ |
| Kairos $W_1$ | $100 \cdot diag\{10, 10, 10, 5, 5, 5\}$ |
| Kairos $W_2$ | $diag\{\mathbf{10}_{n_b}, \mathbf{0.5}_{n_a}\}$ (base joint velocity penalty 10，arm joint velocity penalty 0.5 → base 多动 arm 少动) |
| Kairos $W_3$ | $diag\{\mathbf{0}_{n_b}, \mathbf{1}_{n_a}\}$ (只对 arm 加 posture constraint) |

注意 Kairos $W_2$ 的设计很巧妙 —— base penalty 大于 arm → solver 倾向用 base（base 大 motion），arm penalty 小 → arm 可以做 fine motion 补偿 base 跟踪误差。这对应"base 负责粗运动，arm 负责精细补偿"的 loco-manipulation design pattern。

---

## 6. 关键结果分析

### 6.1 Fig. 4 - 时序数据

**Fig. 4(a) ACI 控制器（FMS 场景）**：
- 顶部图：α 随时间变化。两个 robot 的 α 都接近 1，但有小 dip —— dip 出现在 $v_h \approx v_{adm} \approx 0$ 的时刻（短暂静止），此时 ratio 计算 noise，α 短暂降到接近 0。这是公式 (2) 的一个 numerical edge case，实际中不影响 task 因为此时 motion 也为零。
- 中下部图：$v_{ee}$ (实际 end-effector velocity)、$v_{adm}$、$v_h$ 的时序。$v_{adm}$ 始终接近 0（force 传不过来），但 $v_{ee}$ 紧跟 $v_h$ —— 说明 ACI 通过 MoCap fallback 成功跟踪 human。

**Fig. 4(b) Admittance baseline（FMS 场景）**：
- backwards 和 sideways sub-movement 开始时，robot 有明显延迟（灰色箭头标出）—— human 必须先拉 strap 拉 to 紧 才能 transmit force。
- down-up 阶段 robot 完全不动（灰色圆圈）—— strap 在 vertical motion 下永远 slack，force 永远传不过去。

这是一个 very visual 的"failure mode"，正好印证了 paper 的核心 motivation。

### 6.2 Fig. 5 - Alignment Metric 量化

三条 bar chart 分别对应 $D_{AM}^x$, $D_{AM}^y$, $D_{AM}^z$：
- 黄色（R1=MOCA）和紫色（R2=Kairos）
- 虚线 = admittance baseline, 实线 = ACI

观察：
- ACI 的 $D_{AM}$ 在所有 axis 都显著低于 admittance baseline。
- $D_{AM}^z$ 的差距最大 —— 因为 z 方向（down-up）正是 deformability 最严重、admittance 最失效的场景。

### 6.3 Fig. 6 - α 与 Force 统计

**Fig. 6(a)**：
- FMS scenario（FMS bars）的 α 远高于 Closet scenario（C bars）。这符合理论预期：deformable strap → 高 α → trust MoCap。
- Closet scenario 的 α 接近 0（rigid object → haptic 完整 → admittance 主导）。

**Fig. 6(b)**：
- Closet scenario 下两个 controller 的 force 幅值差不多（rigid object → force 自然传递，不需要 ACI）。
- FMS scenario 下，admittance baseline 需要更高的 force 才能驱动 robot（human 必须 harder pull strap）—— 而 ACI 因为 fallback 到 MoCap，所需 force 显著低。

这是 ergonomic 角度的关键 benefit —— 减少 human 的 physical load。

---

## 7. 与相关工作的关系

### 7.1 Admittance/Impedance Control 历史

经典 pHRI 文献链 [3]-[7]：Ikeura & Inooka 1995 variable impedance，Duchaine & Gosselin 2007/2009 velocity-based variable impedance。这些都是 rigid-object 假设。

### 7.2 Deformable Object Co-manipulation

- Maeda et al. 2001 [9]：hybrid impedance + visual feedback，用 minimum jerk model。但 minimum jerk model 对复杂协作不合适（Miossec & Kheddar [10] 指出）。
- Kruse et al. 2015 [11]：专门针对布料，依赖 visual detection of deformed areas。
- DelPreto & Rus 2019 [12]：EMG-based co-lifting，但 1-D。

这篇 paper 的差异化：**通用 deformability-agnostic**，不限于特定 object 类型。

### 7.3 Multi-Robot Collaboration

- Wang & Schwager 2016 "FORCE-ANTS" [13]：force-amplifying n-robot transport，no inter-robot communication。Link: https://journals.sagepub.com/doi/10.1177/0278364915584694 
- Elwin et al. 2023 "mocobot" [14]：Omnid mobile collaborative robot，human-multi-robot teaming。Link: https://ieeexplore.ieee.org/document/9900499

这两篇都需要 rigid connection。本 paper 第一次把 multi-robot + deformable object + human 合作同时解决。

### 7.4 整个 framework 的 lineage

这是 Ajoudani group 一系列工作的延伸：
- Wu et al. 2019 [16]：MOCA teleoperation interface
- Wu et al. 2021 [17]：weighted whole-body Cartesian impedance for MOCA
- Lamon et al. 2020 [19]：mixed case palletizing with weighted formulation
- Sirintuna, Giammarino, Ajoudani 2022 [2]：IROS 论文，单 robot 版本
- 这篇：multi-robot 扩展

前期 IROS 2022 paper 链接（虽然具体 IEEE 链接我没有，但能在 IIT HRII publications 页找到）：https://hrii.iit.it/publications/

---

## 8. 我对这篇 paper 的 criticism / open questions

尽管 framework 设计 elegant，仍有几点可以挑：

1. **α 是 hand-crafted formula**：能否用 learned gating (e.g., small NN 或 RL policy) 替代？作者用 energy ratio 作为 deformability proxy 是 physics-informed 但 suboptimal —— 比如 strap 在某些 direction deformable 而其他 direction rigid 的情况（anisotropic deformability）需要 per-axis α，公式 (2) 的 scalar α 不能直接处理。扩展到 $\alpha \in \mathbb{R}^3$ 是 trivial 的下一步。
2. **MoCap 是 global position 信号**：deformable 时 MoCap 给的是 hand position，robot 没有直接的 proprioceptive sense of object state。如果 hand position 测量有偏（Xsens IMU drift），整个 system 会有 bias。
3. **没有 obstacle avoidance**：作者承认，未来工作要加 base obstacle avoidance —— 这对 industrial deployment 是 critical 的。
4. **Heterogeneity 仅 2 robots**：scaling 到 N robots 的实验没做，虽然 framework 理论上支持 N 个。在 N>2 时，object 的 distributed compliance behavior 可能 emergent 出现意想不到的 dynamics。
5. **α saturation 在 0**：但没说上界 saturation 在 1，应该也 sat。论文里说 $[0,1]$，但公式 (2) 在某些 noisy 情况下可能 >1（当 $v_{adm}$ > $v_h$，比如 robot 外部被推）。需要 saturate。
6. **只测 2 个 subjects**：statistical power 低。但 qualitative claims 在 alignment metric 里清晰可见。

### 8.1 与 ML / modern robotics 的联想

- 公式 (2) 实际上是个 **self-supervised confidence estimator**：用 sensor A (haptic) 和 sensor B (MoCap) 的 agreement/disagreement 推断 sensor A 的可靠性。这在 ML 里是 **sensor reliability estimation / multi-modal sensor fusion** 的核心问题。类比：autonomous driving 里 radar + camera disagreement 用来 detect sensor failure。
- 公式 (3) 是一种 **gated residual** —— 类似 Highway Network (Srivastava et al. 2015) 或 GRU 的 gating。highway network: $y = g \cdot H(x) + (1-g) \cdot x$，和这个公式 (3) 的结构 reverse 是不是看起来很熟？
- 这个 framework 的 design philosophy —— **physics-informed gating + fallback to exteroception** —— 在 robot learning 里很值得借鉴。特别是当我们做 imitation learning 时，proprioception 可能 noisy 或 delayed，类似的 fallback design 能 robustify policy。

### 8.2 与 Loco-Manipulation 现代工作的联系

近年的 loco-manipulation（如 Boston Dynamics Atlas 整体规划、Stanford Mobile ALOHA、CMU 这里没列）越来越关心 whole-body coordination。这篇 paper 的 weighted inverse dynamics formulation 在 mobile manipulator 上是经典做法，但放到 humanoid 上（如 Atlas）需要额外的 leg dynamics 耦合。一个 potential extension 是把这个 ACI framework 拓到 humanoid loco-manipulation —— 同样 deformable object 场景对人形机器人更常见。

### 8.3 与 Sim-to-Real 的潜在联系

这篇 paper 的 hand-crafted α 在 deformable object 上 works，但 deformable object 的 physics simulation 在 sim-to-real pipeline 里是个 hard problem（soft body simulation）。ACI 这种 "ignore physics of object, only use haptic+vision fusion" 的思路恰好 bypass 了 sim-to-real gap 的某一部分 —— 因为它把 deformable object 的 physics 当作黑盒，只关心 sensory fusion。这对 future sim-to-real training of co-manipulation policies 有启发意义。

---

## 9. 总结

这篇 paper 的核心 contribution 简洁且 actionable：

> **一个 per-robot 的 adaptive index α 自动推断 object 的 deformability 并在 haptic-only admittance controller 和 vision-based MoCap tracking 之间 soft-gate，从而 decentralized、heterogeneous multi-robot 协作搬运 unwieldy / deformable object 成为可能。**

公式 beauty：
- Eq. (2) 是 elegant 的 physics-informed gating（一个 ratio 捕获 deformability）。
- Eq. (3) 是 asymmetric gating（admittance + α·MoCap，rigid 情况下两者一致不影响，deformable 情况下 fallback 到 MoCap）。
- 整个 framework 是 modular 的：admittance law 可换，MoCap 可换，robot controller 可换（demonstrated by MOCA torque + Kairos position 双平台）。

主要 takeaway 给我做 ML/robotics 的人：
- Hand-crafted physics-informed gating 在 narrow domain 上仍然很 powerful，不要无脑 learn。
- Multi-modal sensor fusion 的 confidence estimation 不一定要 learned，可以用 sensor 间的 ratio。
- Decentralized multi-robot control 在有 shared medium (here: human) 的情况下可能比 explicit communication 简单。

---

## 关键 references (web links)

- **Paper video**: https://youtu.be/Q3sA6YzTaaE
- **HRII Lab (Ajoudani)**: https://hrii.iit.it/
- **Franka Emika Panda**: https://www.franka.de/
- **UR16e**: https://www.universal-robots.com/products/ur16-robot/
- **Robotnik SUMMIT-XL STEEL**: https://robotnik.es/robotics-robots-mobile-robots/summit-xl-steel-en/
- **Pisa/IIT SoftHand (qbrobotics)**: https://www.qbrobotics.com/products/qbsofthand-research/
- **Xsens MVN**: https://www.movella.com/products/motion-capture/xsens-mvn-analyze
- **Khatib 1995 Operational Space Formulation**: https://journals.sagepub.com/doi/10.1177/027836499501400102
- **MOCA whole-body (Wu et al. 2021)**: https://ieeexplore.ieee.org/document/9362431
- **FORCE-ANTS (Wang & Schwager 2016)**: https://journals.sagepub.com/doi/10.1177/0278364915584694
- **mocobot (Elwin et al. 2023)**: https://ieeexplore.ieee.org/document/9900499
- **Wampler 1986 DLS**: https://ieeexplore.ieee.org/document/1088303
- **Sirintuna et al. IROS 2022 (前期工作)**: 在 IEEE XplSearch "Sirintuna Ajoudani IROS 2022" 应能找到。

如果你想继续深挖某个 sub-component（比如 weighted inverse dynamics 的 derivation、HQP 的 dual problem、SoftHand 在 deformable object grasp 上的角色、或这个 framework 怎么用 RL/IL 替换 hand-crafted gating），可以告诉我，我可以深入讲。
