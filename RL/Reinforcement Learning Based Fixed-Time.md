---
source_pdf: Reinforcement Learning Based Fixed-Time.pdf
paper_sha256: 8121a734ba57847ed561437b4f8dcbd394098aa520a76a4c6eb5d7bc1cf6f5fe
processed_at: '2026-08-11T22:15:51-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

## 一句话总结

这篇 paper 讲的是怎么让一个**不知道自己参数的机械臂**, 在**有干扰、有扭矩上限**的情况下, **保证在固定时间内**精确跟踪一条指定轨迹, 而且这个"固定时间"跟机械臂一开始偏离多少无关。

## 这个问题为什么难

机械臂的 dynamics 长这样:

$$M(\pmb q)\ddot{\pmb q} + C(\pmb q,\dot{\pmb q})\dot{\pmb q} + \pmb g(\pmb q) = \pmb\tau + \pmb d$$

左边三项分别是: inertia (惯性力)、Coriolis (科氏力/离心力)、gravity (重力)。右边是 motor torque 加 external disturbance。

Real-world 的麻烦在于:
1. 你**不知道** $M, C, \pmb g$ 的精确值 (mass distribution、friction、load 变化)
2. Motor 有**扭矩上限** (e.g., Kinova JACO2 一个关节最多输出 ±3 N·m)
3. 有**外部干扰** (人推一下、风、摩擦变化)
4. 你希望它**快速**收敛到目标轨迹, 而且收敛时间**可预测**

工业场景里, "可预测"特别重要。如果机械臂收敛时间依赖于初始偏差, 那 assembly line 上你就不知道每个 cycle 要等多久, throughput 无法 guarantee。

## 为什么之前的 finite-time 方法不够用

Finite-time control (Bhat & Bernstein 1999, https://doi.org/10.1137/S0363012997325329) 已经能让 error 在有限时间内收敛到 0。但有个 trap: 收敛时间 $T$ 依赖初始误差 $e(0)$。

举个例子, finite-time sliding mode 用 $\dot s = -k\text{sign}(s)$, 收敛时间 $T = |s(0)|/k$。如果 $s(0) = 10$, $k=1$, 要 10 秒; 如果 $s(0) = 100$, 要 100 秒。

在机械臂场景下, 初始偏差可能从 0.01 rad 到 3 rad 都有 (取决于操作员把它放在哪), 你没办法 design 一个 $k$ 同时 cover 所有情况 —— $k$ 太小 large error 收敛慢, $k$ 太大 torque 爆炸 (还会触发 saturation)。

## Fixed-time 的核心 insight

Polyakov 2012 (https://doi.org/10.1109/TAC.2012.2180608) 提出的 fixed-time stability 想法非常 elegant:

在 sliding mode reaching law 里**同时用两个幂次项**, 一个小于 1, 一个大于 1:

$$\dot s = -(\alpha |s|^p + \beta |s|^g)^k$$

where $0 < pk < 1 < gk$.

**为什么这样做 work**:

- 当 $|s|$ 大 (e.g., 100), $|s|^g$ ($g>1$) 这个项**爆炸式**拉大 gradient, state 飞速冲向 0 附近, 这阶段主要由 $g$ 项主导
- 当 $|s|$ 小 (e.g., 0.01), $|s|^p$ ($p<1$) 这个项反而比 $|s|$ 本身大 (因为 $0.01^{0.7} \approx 0.05 > 0.01$), 提供"放大"效应, 让 state 快速终结到 0, 这阶段由 $p$ 项主导
- 两个 regime 加起来, 总时间上界 $T \le \frac{1}{\alpha^k(1-pk)} + \frac{1}{\beta^k(gk-1)}$, **完全与 $s(0)$ 无关**

Intuition 上, 这像是给控制系统装了"两档变速箱": 高速档处理 large error, 低速档处理 small error, 整个 process 速度都很快。

Paper 里的 sliding surface 进一步把这个 trick 用了**两层**:

**第一层**: reaching law $\text{sig}^{v_4}(\sigma_1\text{sig}^{v_2}(\pmb s) + \sigma_2\text{sig}^{v_3}(\pmb s))$, where $v_2 v_4 < 1 < v_3 v_4$, 保证 sliding surface $\pmb s$ 在 fixed time 收敛

**第二层**: sliding surface 自身 $K(\pmb e_1) = (\alpha|e_1|^{p-1/(kv_1)} + \beta|e_1|^{g-1/(kv_1)})^{kv_1}$, where $\frac{1}{v_1} < pk < 1 < gk$, 保证一旦 $\pmb s = 0$, tracking error $\pmb e_1$ 也在 fixed time 收敛

两层 cascade, 总时间 $T_{max} = T_1 + T_2$, 都与初始 state 无关。这是 paper 的核心理论 contribution。

## 为什么还需要 RL

Fixed-time control 的 standard 设计要求**知道 system dynamics** (至少知道 nominal $M_0, C_0, \pmb g_0$)。Paper 里确实用了 nominal model 做 feedback linearization (公式 23 里前乘 $M_0$)。

但 nominal model 和真实 model 之间的 gap —— paper 里叫 $\pmb l(t)$ —— 是 state-dependent 未知非线性。传统 robust control 用 large gain 或 sign term 吞掉它, 代价是 chattering + conservative。

Paper 的想法: 用 **RBF neural network 在线学习** $\pmb l(t)$。RBF NN 是 universal approximator (Cybenko 1989, https://doi.org/10.1016/0893-6080(89)90040-9), 给定足够多 hidden nodes 可以逼近任何连续函数到任意精度。

但单纯做 function approximation 还不够 smart。Paper 用了 **actor-critic structure** (Vamvoudakis & Lewis 2010, https://doi.org/10.1016/j.automatica.2010.02.018):

- **Actor NN**: 输出 control action (这里用来逼近 uncertainty $\pmb l$, 然后通过 controller 抵消它)
- **Critic NN**: 估计 cost-to-go $I(t) = \int_t^\infty e^{-(\iota-t)/\psi}\varphi(\iota)d\iota$, 也就是从当前 state 开始到无穷远的累计 cost (tracking error + control effort 的 discounted sum)

Actor 根据 Critic 的估值调整自己的策略, Critic 根据实际 observed cost 修正自己的估值。这就是 RL 里 policy evaluation + policy improvement 的 online 版本。

**为什么要这么搞**:

- 纯 function approximation (e.g., 纯 RBF adaptive control, 像 [34]) 只是让 uncertainty 估计越来越准, 没有"什么是好控制"的概念
- Actor-critic 显式优化 cost function $I(t)$, 控制策略是在线 optimal 的 (在 NN 表达能力范围内)
- Cost function 里既有 tracking error ($\pmb e^T D \pmb e$) 又有 control effort ($\pmb\tau^T R \pmb\tau$), 自动 trade-off 性能和能耗

Critic 的 updating law (公式 17) 本质是 gradient descent on Bellman residual $\gamma(t) = \varphi(t) - \frac{1}{\psi}\hat I + \dot{\hat I}$。这个 $\gamma$ 是 continuous-time TD error, 当 $\gamma \to 0$ 时 NN 估值满足 Bellman equation, 即达到 optimal value function。

## 为什么还要 anti-windup

Actuator saturation 是 real robot 上的硬约束。Kinova JACO2 一个关节最大扭矩 ±3 N·m, 你 controller 算出 10 N·m, 实际只给 3, 剩下 7 被"截掉"了。

这个截掉的部分 $\pmb\tau_\Delta = \pmb\tau - \pmb\tau_0$ 会引起两个问题:

1. **Controller 不知道自己被截了**, 继续按 nominal 算, integral term 不断累积 ("windup"), 等饱和过去后 system 会 overshoot 甚至失稳
2. **Lyapunov analysis 中断**, 因为 $\pmb\tau \neq \pmb\tau_0$, 你之前证明的 $\dot V < 0$ 不再成立

Paper 的 anti-windup compensator (公式 24):

$$\dot{\pmb\zeta} = -K_\zeta\pmb\zeta - \text{sig}^{v_4}(\sigma_3\text{sig}^{v_2}(\pmb\zeta) + \sigma_4\text{sig}^{v_3}(\pmb\zeta)) + \pmb s + \pmb\tau_\Delta$$

$\pmb\zeta$ 是个 internal state, 实时跟踪 $\pmb\tau_\Delta$ (饱和 residual), 然后通过 controller (公式 23) 里 $\pmb\zeta$ 项 inject 回去, 抵消饱和效应。Compensator 自己也是 fixed-time stable (用同样的双幂次 trick), 保证 compensation 不会 lag。

直觉上, 这像个"误差检测器 + 补偿器": 检测到实际扭矩和命令扭矩的 gap, 立刻在 controller 里加一个 counter-term, 让 closed-loop dynamics 看起来好像没饱和一样。

## 整体架构怎么串起来

我画一下 dataflow:

```
[Desired trajectory x_d]
            │
            ▼
[Tracking error e_1, e_2] ───────► [Sliding surface s = K(e_1)e_1 + sig^v1(e_2)]
            │                                          │
            ▼                                          ▼
       [Actor NN] ──── f_NN (逼近 uncertainty l) ───► [RL-FTNTSM Controller (23)]
            ▲                                          │
            │                                          ▼
       [Critic NN] ◄── Bellman residual γ(t)      [τ_0 nominal control]
            ▲                                          │
            │                                          ▼
       [Cost φ(t)] ◄──────────────────────────── [Actuator saturation]
                                                       │
                                                       ▼
                                                 [Actual torque τ]
                                                       │
                                                       ▼
                                              [Robot dynamics]
                                                       │
                                  ┌────────────────────┘
                                  ▼
                            [State x_1, x_2]
                                  │
                                  ▼
                            [Tracking error] (闭环)
```

Anti-windup compensator 在 saturation 那个节点工作, 检测 $\pmb\tau - \pmb\tau_0$, 把 $\pmb\zeta$ inject 回 controller。

## 实验结果说了什么

**Two-link 仿真**: 跟 ESO-NTSM (Sun & Liu 2019, https://doi.org/10.1016/j.ymssp.2019.106374) 对比, ITAE 从 0.4177 降到 0.1015 (joint 1), 降低 75%。这个 improvement 主要来自 fixed-time 收敛 —— ESO-NTSM 是 finite-time, 在 large initial error 时收敛慢, 拖累了 ITAE (time-weighted, 后期 error 权重大)。

**Kinova 实验**: 跟 RBF-NTSM (Tran & Kang 2016, https://doi.org/10.1007/s12541-016-0113-5) 对比, 周期轨迹和指数轨迹 steady-state accuracy 都明显更好。这说明 RL 学习到的 control policy 比 fixed-gain robust control 更精细。

**Robustness test**: disturbance 从 0.5 N·m 加到 3-4 N·m (6 倍), tracking error 仍然在 fixed time 收敛。说明 robust term $\bar d\text{sign}(\pmb s)$ + NN approximation 共同 handle large disturbance。

**NN weight 有界**: 图14 显示 weight norm 始终在预设范围内, 验证 projection algorithm (公式 18, 31) 有效, 这是 Lyapunov stability proof 成立的前提。

## 我的思考与联想

读完这篇, 我有几个 directions 联想:

### 1. 跟 Model Predictive Control (MPC) 的对比

MPC 也显式优化 cost function, 而且可以 handle constraints (包括 saturation)。但 MPC 需要求解 online optimization, computational cost 高, 在机械臂这种快 dynamics 上 real-time 有点难。Actor-critic RL 是 offline training (或 online gradient descent) + online forward pass, 比 MPC 轻量。

不过 MPC 的 advantage 是可以显式 handle state constraints (joint limits, workspace), 这篇 paper 完全没考虑 safety。如果机械臂在 human-robot collaboration 场景下, 加上 barrier function 或者 safety filter (e.g., Control Barrier Functions, Ames et al 2019, https://doi.org/10.1146/annurev-control-053018-023743) 会很 natural extension。

### 2. 跟 Deep RL 的对比

这篇用的 RBF NN 只有 4 个 hidden nodes ($r=4$), 表达能力有限。Modern Deep RL (PPO, SAC, TD3) 用几十万参数的 deep network, 表达能力强得多。但 deep RL 在 control theory 意义上没有 stability guarantee, 而 RBF + projection + Lyapunov design 可以严格证明 fixed-time convergence。

这个 trade-off (expressiveness vs theoretical guarantee) 是 current RL + control 融合的核心 tension。一个有意思的方向是 Lyapunov Neural Networks (Chang et al 2019, https://doi.org/10.1109/CDC.2019.9029653), 学习 Lyapunov function 同时保证 stability。

### 3. Fixed-time 在 robot 之外的 application

Fixed-time control 在 spacecraft attitude control (Jiang et al 2016, https://doi.org/10.1109/TCST.2016.2549947)、hypersonic missile (Basin et al 2018, https://doi.org/10.1109/TIE.2017.2733479)、multi-agent consensus (Zuo et al 2018, https://doi.org/10.1109/TAC.2017.2723000) 都有应用。共同点是: 这些 system 都要求**严格的 timing guarantee**, 而且初始条件不可预测。

### 4. Anti-windup 的更 general 形式

这篇用的 anti-windup 是 dynamic compensator + fixed-time reaching law。更 general 的 anti-windup framework 是 AWBT (Anti-Windup Bumpless Transfer, Astrom & Rundqwist 1989, https://doi.org/10.1016/0005-1098(89)90124-6), 把 anti-windup 看成 regulator 自身的 modification。那篇 framework 下可以更灵活地 design compensation, 可能可以跟 RL 进一步结合 (用 NN 学习 optimal anti-windup gain)。

### 5. Sample efficiency 的隐忧

Actor-critic 用 gradient descent 在线学习, learning rate $\delta_a, \delta_c$ 是 constant。但 RBF NN 在 training 早期 approximation error 大, fixed-time controller 又强制 fast convergence, 可能 force system 进入 NN 还没学好的 region。Paper 没讨论这个 transient phase 的 robustness。Sim2real gap (Kinova 实验跟仿真参数可能不完全 match) 也可能影响 NN 学习。这其实是 RL 用于 real robot 的普遍问题 (Lange et al 2012, https://doi.org/10.1162/NECO_a_00304)。

### 6. 两个 mathematical trick 的通用性

这篇的两个核心 trick —— **双幂次 fixed-time** 和 **RBF NN + projection** —— 在其他 control problem 上也可以直接 plug-in:

- 多智能体 consensus: 把每个 agent 看成一个"关节", 共同跟踪 leader trajectory
- Process control: chemical reactor 的 temperature control, 时间常数大, fixed-time 保证 batch processing throughput
- Power system: frequency regulation, 跟踪 nominal frequency, 扰动是 load fluctuation

RBF + projection 的 stability guarantee 思路也适用于 any basis function (polynomial, spline, 甚至 deep ReLU network 加 spectral normalization), 通用性很强。

### 7. 跟我熟悉的 deep learning 的 connection

RBF NN 本质上是个 kernel method, 跟 SVM、Gaussian Process 有 family resemblance。Critic 的 Bellman residual minimization 跟 TD-learning 在 deep RL 里的 target network update 同源 (Mnih et al 2015, https://doi.org/10.1038/nature14236)。区别是 deep RL 用 stochastic gradient + replay buffer, 这篇用 deterministic gradient + online update。

如果把这个 framework 搬到 deep RL: critic network 用 deep MLP, actor network 也用 deep MLP, projection algorithm 换成 weight clipping 或 spectral normalization, 学习算法换成 experience replay + mini-batch。理论上可以做, 但 Lyapunov stability proof 会变得非常 tricky (deep ReLU 不 Lipschitz, V 不容易证明 positive definite)。这可能是一个 future direction —— **Deep RL + Lyapunov-based stability guarantee**。

---

总结一下, 这篇 paper 是 control theory + RL 的一次 carefully engineered 融合, 每个组件 (fixed-time, NTSM, actor-critic, anti-windup) 都有 solid prior art, 但组合在一起需要**非常细致的 Lyapunov analysis** (Appendix 里那串推导)。读完后我的最大 takeaway 是: **control theory 的"双幂次 trick"在数学上极其简洁, 几乎免费就拿到了与初始状态无关的收敛时间上界, 这对 real-time control 系统设计是巨大的工程价值**。

---

# Reinforcement Learning Based Fixed-Time Trajectory Tracking Control 深度技术讲解

## 一、Paper 整体定位与动机

这篇 paper 发表于 IEEE Transactions on Neural Networks and Learning Systems (TNNLS) 2021, 作者 Shengjie Cao, Liang Sun, Jingjing Jiang, Zongyu Zuo。DOI: https://doi.org/10.1109/TNNLS.2021.3116713

这篇工作本质上把**三个独立成熟的 control 理论方向**进行了 tight coupling:
1. **Fixed-time stability** (Polyakov 2012, https://doi.org/10.1109/TAC.2012.2180608)
2. **Actor-Critic RL with RBF neural network approximation** (Vamvoudakis & Lewis 2010, https://doi.org/10.1016/j.automatica.2010.02.018)
3. **Non-singular terminal sliding mode control** (NTSM) 配合 anti-windup saturation compensation

核心 motivation: 现有 RL-based robotic control 方法都局限于 **finite-time framework**, 而 finite-time 收敛时间依赖于系统初始状态, 在 robotic manipulator 这种 state 横跨 large range 的场景下不可接受。Fixed-time framework 把收敛时间上界与初始状态解耦, 与 RL 的 model-free / model-partial 特性结合, 可以在 unknown dynamics + actuator saturation 条件下实现 guaranteed convergence time。

---

## 二、Problem Formulation 数学结构

### 2.1 机械臂 Euler-Lagrange Dynamics

$$M(\pmb q)\ddot{\pmb q} + C(\pmb q,\dot{\pmb q})\dot{\pmb q} + \pmb g(\pmb q) = \pmb\tau(t) + \pmb d(t) \tag{1}$$

变量含义:
- $\pmb q \in \mathbb{R}^n$: joint position vector, n 是机械臂自由度 (两连杆仿真 n=2, Kinova 实验取第3关节 n=1)
- $M(\pmb q) \in \mathbb{R}^{n\times n}$: inertia matrix, symmetric positive definite
- $C(\pmb q,\dot{\pmb q}) \in \mathbb{R}^{n\times n}$: Coriolis and centrifugal matrix
- $\pmb g(\pmb q) \in \mathbb{R}^n$: gravity torque vector
- $\pmb\tau(t)$: control input (joint torque)
- $\pmb d(t)$: external disturbance torque

### 2.2 Model Decomposition (Nominal + Uncertainty)

$$M = M_0 + M_\Delta, \quad C = C_0 + C_\Delta, \quad \pmb g = \pmb g_0 + \pmb g_\Delta$$

下标 0 表示 nominal known part, 下标 Δ 表示 unknown bounded uncertainty。这种 decomposition 是 robust/adaptive control 的标准 trick, 让 controller 可以基于 nominal model 设计, 用 NN / robust term 处理 Δ。

### 2.3 状态空间形式与 Tracking Error

定义 $\pmb x_1 = \pmb q, \pmb x_2 = \dot{\pmb q}$, 跟踪误差 $\pmb e_1 = \pmb x_1 - \pmb x_d, \pmb e_2 = \dot{\pmb x}_1 - \dot{\pmb x}_d$:

$$\dot{\pmb e}_1 = \pmb e_2, \quad \dot{\pmb e}_2 = \pmb l(t) + M_0^{-1}\pmb\tau + M^{-1}\pmb d - \ddot{\pmb x}_d \tag{5}$$

其中建模不确定性 $\pmb l(t) = M^{-1}[-C\dot{\pmb q} - \pmb g(\pmb q)] + \bar M_\Delta \pmb\tau$, $\bar M_\Delta = M^{-1} - M_0^{-1}$。这个 $\pmb l(t)$ 是 actor NN 要逼近的目标函数, 它 state-dependent (依赖于 q, q̇, τ)。

### 2.4 Input Saturation 约束

$$\tau_{Li} \le \tau_i \le \tau_{Hi}, \quad \tau_i = \begin{cases} \tau_{Hi} & \tau_i \ge \tau_{Hi} \\ \tau_{0i} & \tau_{Li} \le \tau_{0i} < \tau_{Hi} \\ \tau_{Li} & \tau_i < \tau_{Li} \end{cases}$$

作者把实际 control input 分解为 $\pmb\tau = \pmb\tau_0 + \pmb\tau_\Delta$, 其中 $\pmb\tau_0$ 是 nominal design part, $\pmb\tau_\Delta$ 是 saturation-induced residual (bounded by Assumption 1)。这是 anti-windup 设计的关键, 让 saturation effect 在 Lyapunov analysis 中体现为 bounded disturbance-like 项。

---

## 三、Actor-Critic RL Architecture 详细解析

### 3.1 Long-term Cost Function (Bellman-like)

$$I(t) = \int_t^\infty e^{-(\iota-t)/\psi}\varphi(\iota)d\iota \tag{13}$$

$$\varphi(t) = (\pmb x_1 - \pmb x_d)^T D (\pmb x_1 - \pmb x_d) + \pmb\tau^T R \pmb\tau \tag{14}$$

- $\psi > 0$: discount factor, 决定 future cost 衰减速率
- $D \in \mathbb{R}^{n\times n}, R \in \mathbb{R}^{n\times n}$: positive definite weighting matrices, $D$ penalizes tracking error, $R$ penalizes control effort
- $I(t)$ 即 cost-to-go function, 对应 optimal control 中的 value function

这是 continuous-time RL 的标准形式, 类似 Hamilton-Jacobi-Bellman equation 的 value function, 不同之处在于这里通过 NN 在线逼近, 而非解析求解 HJB。

### 3.2 Critic NN 设计

Critic 逼近 cost-to-go:
$$I = \pmb w_c^{*T}\pmb\sigma_c(\pmb e_1) + \varepsilon_c, \quad \hat I = \hat{\pmb w}_c^T\pmb\sigma_c(\pmb e_1)$$

- $\pmb w_c^* \in \mathbb{R}^r$: optimal weight vector, r 是 RBF hidden layer 节点数 (paper 中 r=4)
- $\pmb\sigma_c(\pmb e_1)$: Gaussian RBF basis function vector
- $\varepsilon_c$: approximation residual

RBF 函数形式:
$$\sigma_i(Z) = \exp\left[-\frac{(Z-\pmb\mu_i)^T(Z-\pmb\mu_i)}{\chi_k^2}\right]$$
- $\pmb\mu_i$: 第 i 个 receptive field 中心
- $\chi_k$: Gaussian width

**Bellman residual error** (即时序差分 TD error):
$$\gamma(t) = \varphi(t) - \frac{1}{\psi}\hat I(t) + \dot{\hat I}(t) \tag{15}$$

这个方程是 continuous-time Bellman equation $\hat I = \int_t^\infty e^{-(\iota-t)/\psi}\varphi d\iota$ 的微分形式: 对 $I$ 求 time derivative 得 $\dot I = \frac{1}{\psi}I - \varphi$, 即 $\varphi - \frac{1}{\psi}I + \dot I = 0$。NN 估计产生 residual $\gamma$。

**Gradient descent updating law**:
$$\dot{\hat{\pmb w}}_c = -\delta_c \gamma\pmb\Lambda, \quad \pmb\Lambda = -\frac{\pmb\sigma_c}{\psi} + \nabla\pmb\sigma_c \dot{\pmb e}_1 \tag{17}$$

- $\delta_c > 0$: critic learning rate
- $\pmb\Lambda$ 中第一项来自 $-\frac{1}{\psi}\hat I$ 对 $\hat{\pmb w}_c$ 的梯度, 第二项来自 $\dot{\hat I} = \frac{\partial \hat I}{\partial \pmb e_1}\dot{\pmb e}_1$ 的链式法则

Remark 2 的关键 intuition: 把 $\dot{\pmb e}_1$ 引入 critic NN input, 使网络能更准确逼近 $I(t)$ 的空间结构, 因为 $I$ 本质上是 state-dependent value function。

### 3.3 Projection-based Updating Law

为保证 NN 权重有界 (Lyapunov 稳定性证明必需), 作者用 projection algorithm:

$$\dot{\pmb w}_c = \begin{cases} -\delta_c\pmb\rho_c & \|\hat{\pmb w}_c\| \le \|\bar{\pmb w}_c\| \text{ 或边界处且向外} \\ -\delta_c\pmb\rho_c + \delta_c\pmb\xi_c & \|\hat{\pmb w}_c\| = \|\bar{\pmb w}_c\|, \hat{\pmb w}_c^T\pmb\rho_c \le 0 \end{cases}$$

- $\pmb\xi_c = \frac{\hat{\pmb w}_c^T\pmb\rho_c}{\|\hat{\pmb w}_c\|^2}\hat{\pmb w}_c$: projection 项, 在权重边界上把 updating 方向投影回允许集
- $\bar{\pmb w}_c$: 预设的权重上界

Theorem 1 用 Lyapunov function $V_c = \frac{1}{2\delta_c}\hat{\pmb w}_c^T\hat{\pmb w}_c$ 证明 projection 保证 $\|\hat{\pmb w}_c\| \le \|\bar{\pmb w}_c\|$ 总成立。这种有界性是后续 fixed-time analysis 的 foundation。

### 3.4 Actor NN 与 FTNTSM Controller

#### 3.4.1 Novel Non-singular Fixed-time Fast Terminal Sliding Mode Surface

$$\pmb s = K(\pmb e_1)\pmb e_1 + \text{sig}^{v_1}(\pmb e_2) \tag{19}$$

$$K(\pmb e_1) = \text{diag}\{k_{e_{11}}, k_{e_{12}}, \ldots, k_{e_{1n}}\}$$

$$k_{e_{1i}} = \left(\alpha|e_{1i}|^{p-1/(kv_1)} + \beta|e_{1i}|^{g-1/(kv_1)}\right)^{kv_1} \tag{20}$$

参数约束:
- $\alpha > 0, \beta > 0$: 双幂次项系数
- $k > 1, v_1 > 1$: 决定非线性度
- $gk > 1$ 且 $\frac{1}{v_1} < pk < 1$: 关键约束, 保证 fixed-time 收敛

**Intuition**: 传统 terminal sliding mode 用单一幂次 $\text{sig}^v(\pmb e_1) + \pmb e_2$, 收敛速度在 large error 时慢。这里 $K(\pmb e_1)$ 是 state-varying gain, 当 $|e_{1i}|$ 大时, $\alpha|e_{1i}|^{p-1/(kv_1)}$ 项 (因 p<1, p-1/(kv_1) 可能更小) 主导, gain 增大; 当 $|e_{1i}|$ 小时, $\beta|e_{1i}|^{g-1/(kv_1)}$ 项 (g>1) 主导, 保证 fast terminal 收敛。

双幂次 $\alpha + \beta$ 结构是 fixed-time control 的标志性设计 (Polyakov 2012): 在 $|e|$ 大时 $|e|^p$ ($p<1$) 项主导加快收敛, 在 $|e|$ 小时 $|e|^g$ ($g>1$) 项主导快速终结, 两个 regime 都有 fast convergence, 整体 upper bound 与初始 state 无关。

#### 3.4.2 Time Derivative of Sliding Surface

$$\dot{\pmb s} = \tilde K(\pmb e_1)\pmb e_2 + K(\pmb e_1)\pmb e_2 + v_1 E_2 \dot{\pmb e}_2 \tag{21}$$

其中:
- $E_2 = \text{diag}\{|e_{21}|^{v_1-1}, \ldots, |e_{2n}|^{v_1-1}\}$: 来自 $\frac{d}{dt}\text{sig}^{v_1}(\pmb e_2) = v_1 E_2 \dot{\pmb e}_2$
- $\tilde K(\pmb e_1)$: 注意 Remark 4 强调 $\tilde K \neq \frac{d}{dt}K$, 是单独计算的辅助矩阵 (公式22)

#### 3.4.3 RL-FTNTSM Controller

$$\pmb\tau_0 = M_0\left[-\frac{1}{v_1}(\tilde K + K)\text{sig}^{2-v_1}(\pmb e_2) - \frac{1}{v_1}\text{sig}^{1-v_1}(\pmb e_2)\big(\text{sig}^{v_4}(\sigma_1\text{sig}^{v_2}(\pmb s) + \sigma_2\text{sig}^{v_3}(\pmb s)) + \pmb\zeta + K_s\pmb s\big) + \ddot{\pmb x}_d - \pmb f_{NN} - \bar d\text{sign}(\pmb s)\right] \tag{23}$$

让我逐项解释这个复杂表达式的每一项 physical meaning:

1. **$-\frac{1}{v_1}(\tilde K + K)\text{sig}^{2-v_1}(\pmb e_2)$**: 抵消 sliding surface 自身 dynamics 中的非线性项, 这是 feedback linearization 的标准思路
2. **$-\frac{1}{v_1}\text{sig}^{1-v_1}(\pmb e_2)(\cdot)$**: terminal sliding mode reaching law, 用 sig 函数避免 singularity
3. **$\text{sig}^{v_4}(\sigma_1\text{sig}^{v_2}(\pmb s) + \sigma_2\text{sig}^{v_3}(\pmb s))$**: **双幂次 fixed-time reaching law**, $\sigma_1, \sigma_2 > 0$, $v_2 v_4 < 1, 1 < v_3 v_4 < v_4$, 这个结构保证 $\dot{\pmb s}$ 在 $\pmb s$ 大和小两种 regime 都 fast converge
4. **$\pmb\zeta$**: anti-windup compensator state, 实时抵消 saturation residual $\pmb\tau_\Delta$
5. **$K_s\pmb s$**: linear stabilizing term, 提供 robust margin (在 Lyapunov analysis 中吞掉 NN 逼近误差和 disturbance)
6. **$\ddot{\pmb x}_d$**: feedforward, 抵消 desired acceleration
7. **$\pmb f_{NN}$**: actor NN 输出, 逼近建模不确定性 $\pmb l(t)$
8. **$\bar d\text{sign}(\pmb s)$**: robust sign term, 吞掉 bounded external disturbance
9. **$M_0$ 前乘**: 用 nominal inertia matrix inverse-feedback-linearize, 把 system nominal dynamics 转为 double integrator

#### 3.4.4 Actor NN 结构

$$\pmb f_{NN} = \begin{bmatrix} \hat{\pmb w}_{a1}^T\pmb\sigma_{a1}(\pmb z_{a1}) \\ \hat{\pmb w}_{a2}^T\pmb\sigma_{a2}(\pmb z_{a2}) \\ \vdots \\ \hat{\pmb w}_{an}^T\pmb\sigma_{an}(\pmb z_{an}) \end{bmatrix}$$

- 每个关节有独立的 actor NN, weight vector $\hat{\pmb w}_{ai} \in \mathbb{R}^m$
- input $\pmb z_{ai} = [e_{1i}, e_{2i}]^T$: 只用该关节的 tracking error, 不用其他关节 state, 这种 decoupled 结构简化了 weight 学习
- 共享 RBF Gaussian basis function

Learning error:
$$\mu_a = \sum_{i=1}^n \tilde{\pmb w}_{ai}^T\pmb\sigma_{ai}(\pmb z_{ai}), \quad \tilde{\pmb w}_{ai} = \pmb w_{ai}^* - \hat{\pmb w}_{ai}$$

Actor error signal:
$$e_a = \mu_a + k_I(\hat I(t) - I_d(t)), \quad I_d = 0$$

- $k_I > 0$: critic 信号注入 actor 学习的 gain
- $I_d = 0$: 理想 cost-to-go 是 0 (perfect tracking, zero control effort)

**关键 insight**: 这个 $e_a$ 把 actor learning 与 critic valuation 强耦合。Critic 估计的 cost-to-go 直接参与 actor 的 gradient, 实现 true actor-critic structure。

#### 3.4.5 Actor Updating Law

$$E_a = \ln(\cosh e_a)$$

$$\dot{\hat{\pmb w}}_{ai} = -\delta_a\tanh(\mu_a + k_I\hat I)\pmb\sigma_{ai} \tag{29}$$

- 用 $\ln(\cosh\cdot)$ 作为 loss (smooth approximation of $|\cdot|$), gradient 是 $\tanh$
- $\tanh$ 比 $\text{sign}$ 平滑, 减少 chattering
- $\delta_a > 0$: actor learning rate

因 $\mu_a$ 不可用 (含 $\pmb w_{ai}^*$), 用 modified version:
$$\dot{\hat{\pmb w}}_{ai} = -\delta_a\tanh\left(\sum_{i=1}^n\hat{\pmb w}_{ai}^T\pmb\sigma_{ai} + k_I\hat I\right)\pmb\sigma_{ai} \tag{30}$$

用当前 NN 估计代替真实 $\mu_a$, 这是 actor-critic 在线学习常见处理。

---

## 四、Anti-Windup Compensator

$$\dot{\pmb\zeta} = -K_\zeta\pmb\zeta - \text{sig}^{v_4}(\sigma_3\text{sig}^{v_2}(\pmb\zeta) + \sigma_4\text{sig}^{v_3}(\pmb\zeta)) + \pmb s + \pmb\tau_\Delta \tag{24}$$

- $K_\zeta$: positive definite damping matrix
- $\sigma_3, \sigma_4 > 0$: 双幂次 fixed-time gain (与 sliding mode 同构)
- $\pmb s$: sliding surface 信号注入
- $\pmb\tau_\Delta$: saturation residual (实际中通过 $\pmb\tau - \pmb\tau_0$ 计算)

**intuition**: 当 actuator 饱和时, $\pmb\tau_\Delta \neq 0$, compensator 接收这个 residual 并通过 $\pmb\zeta$ 注入 controller (公式23中的 $\pmb\zeta$ 项), 抵消 saturation 带来的 deviation。$-K_\zeta\pmb\zeta$ 防止 compensator state 漂移, fixed-time reaching law 保证 $\pmb\zeta$ 自身在固定时间收敛。这是 paper contribution 3 的核心。

---

## 五、Stability Analysis 深度解析 (Appendix)

### 5.1 Composite Lyapunov Function

$$V(t) = \frac{1}{2}\pmb s^T\pmb s + \frac{1}{2}\pmb\zeta^T\pmb\zeta + \frac{1}{2\delta_c}\tilde{\pmb w}_c^T\tilde{\pmb w}_c + \frac{1}{2\delta_a}\sum_{i=1}^n\tilde{\pmb w}_{ai}^T\tilde{\pmb w}_{ai} \tag{39}$$

四项分别对应: sliding surface energy, anti-windup state energy, critic NN weight error energy, actor NN weight error energy。这是 adaptive/learning control 标准 composite Lyapunov 设计。

### 5.2 Closed-loop Sliding Dynamics

代入 controller 后:
$$\dot{\pmb s} = -\text{sig}^{v_4}(\sigma_1\text{sig}^{v_2}(\pmb s) + \sigma_2\text{sig}^{v_3}(\pmb s)) - \pmb\zeta - K_s\pmb s + v_1 E_2[M_0^{-1}\pmb\tau_\Delta + \tilde{\pmb f} + M_0^{-1}\pmb d - \bar d\text{sign}(\pmb s)] \tag{40}$$

其中 $\tilde{\pmb f}$ 是 actor NN 逼近误差 (Eq.41)。

### 5.3 Young's Inequality 关键技巧

$$v_1\pmb s^T E_2 M_0^{-1}\pmb\tau_\Delta \le \frac{v_1}{2}\pmb s^T K_M\pmb s + \frac{v_1}{2}\pmb\tau_\Delta^T\pmb\tau_\Delta$$

其中 $K_M = E_2 M_0^{-1}M_0^{-1}E_2$。

Young's inequality $ab \le \frac{a^2}{2} + \frac{b^2}{2}$ 是 control theory 中处理 cross-term 的标准工具, 把 disturbance/uncertainty 相关 cross-term 分离为 quadratic in state + constant term, 后者被 Lyapunov function 中其他 negative terms 吞掉。

### 5.4 Fixed-time Convergence via Lemma 2

通过一系列代数操作, 最终得到:
$$\dot V(t) \le -(3n+1)^{1-v_4}\left(\sigma_5 V^{(1+v_2v_4)/2v_4} + \sigma_6 V^{(1+v_3v_4)/2v_4}\right)^{v_4} + \Omega_2 \tag{48}$$

这正是 Lemma 2 中的形式 $\dot V \le -(\alpha V^c + \beta V^q)^k + \epsilon$。

**Fixed-time 收敛时间上界**:
$$T_1 = \frac{2}{(3n+1)^{1-v_4}\sigma_5^{v_4}\theta^{v_4}(1-v_2v_4)} + \frac{2}{(3n+1)^{1-v_4}\sigma_6^{v_4}\theta^{v_4}(v_3v_4-1)}$$

$$T_2 = \frac{1}{\alpha^k\theta^k(1-pk)} + \frac{1}{\beta^k\theta^k(gk-1)}$$

$$T \le T_{max} = T_1 + T_2 \tag{35}$$

**Crucial observation**: $T_1$ 只依赖于控制参数 $(\sigma_5, \sigma_6, v_2, v_3, v_4, \theta, n)$, $T_2$ 只依赖于 $(\alpha, \beta, p, g, k, \theta)$, 完全不依赖初始状态 $\pmb x_1(0), \pmb x_2(0)$。这是 fixed-time control 相对 finite-time control 的核心优势。

### 5.5 Two-stage Convergence 逻辑

1. **Stage 1**: sliding surface $\pmb s \to \bar s$ (小 neighborhood of 0), 时间 $T_1$, 通过 reaching law 的 fixed-time 性质
2. **Stage 2**: sliding surface 上, $\pmb e_1 \to \bar e_1$, $\pmb e_2 \to \bar e_2$, 时间 $T_2$, 通过 $K(\pmb e_1)$ 双幂次项的 fixed-time 性质

Total $T = T_1 + T_2$, 这种 cascade 结构是 sliding mode control 的标志性分析模式。

---

## 六、Simulation 与 Experimental Validation

### 6.1 Two-link Planar Manipulator 仿真

**物理参数**:
- $m_1 = 2.00$ kg, $m_2 = 0.85$ kg (link masses)
- $l_1 = 0.35$ m, $l_2 = 0.31$ m (link lengths)
- $I_i = \frac{1}{4}m_i l_i^2$ (moment of inertia, uniform rod assumption)
- $g = 9.8$ m/s²
- Disturbance: $\pmb d(t) = [0.5(\sin(0.1t)+1), 0.5(\cos(0.1t)+1)]^T$ N·m

**Controller 参数**:
- Sliding mode: $\alpha = 1.5, \beta = 1.5, p = 0.7, g = 10/9, k = 1.2, v_1 = 1.2$
- Reaching law: $v_2 = 1/3, v_3 = 3/4, v_4 = 2, \sigma_1 = \sigma_2 = \sqrt 2$
- Robust: $K_s = \text{diag}\{100, 100\}, K_\zeta = \text{diag}\{1, 1\}, \bar d = 2.2$
- NN: $r = 4$ (Gaussian nodes), RBF for both actor and critic
- Saturation: $\tau_{Li} = -5, \tau_{Hi} = 5$ N·m

**Desired trajectory**: $\pmb x_d = [0.1(\sin(0.5t)+\cos(0.5t)), 0.1\sin(t)+\cos(t)]^T$ rad

### 6.2 Performance Comparison Table

| 指标 | Error (rad) | Controller (23) | Controller (36) ESO-NTSM |
|------|------------|----------------|--------------------------|
| IAE | $e_1$ joint 1 | 0.1952 | 0.2688 |
| IAE | $e_1$ joint 2 | 0.1150 | 0.2671 |
| ITAE | $e_1$ joint 1 | 0.1015 | 0.4177 |
| ITAE | $e_1$ joint 2 | 0.0772 | 0.4187 |

- **IAE** = $\int_0^t(|e_1(\tau)| + |e_2(\tau)|)d\tau$: 累积绝对误差, 衡量 overall tracking 精度
- **ITAE** = $\int_0^t \tau(|e_1(\tau)| + |e_2(\tau)|)d\tau$: 时间加权 ATE, 强调稳态误差

Proposed controller (23) 在 IAE 上降低约 30-57%, ITAE 上降低约 75-81%。ITAE 改善更显著, 说明 fixed-time 收敛确实带来稳态精度提升。

### 6.3 Kinova JACO2 实验

硬件: Kinova JACO2 6-DOF 轻型机械臂 (图2), 通过 Matlab 直接控制 torque mode, 取第3关节做实验。

**两种 desired trajectory**:
1. **Exponential**: $x_d = 2.3 - \frac{7}{5}e^{-0.5t} + \frac{7}{20}e^{-2t}$
2. **Periodic**: $x_d = 1.3 + 0.3\sin(t)$

**Saturation**: $\tau_{Li} = -2, \tau_{Hi} = 2$ (exponential), $\tau_{Li} = -3, \tau_{Hi} = 3$ (periodic)

对比 controller (38) (Tran & Kang 2016, RBF NN + NTSM), 实验 result 显示 controller (23) 在 steady-state accuracy 上明显优于 (38)。

### 6.4 Robustness Test

把 disturbance 增大到 $\pmb d(t) = [3(\sin(0.1t)+1), 4(\cos(0.1t)+1)]^T$ N·m (原 6 倍), 仿真 (图11-13) 显示 tracking error 仍在固定时间收敛到 small neighborhood, 验证 robustness。图14 显示 NN weight norm 始终有界, 验证 projection algorithm 有效性。

---

## 七、Parameter Sensitivity 分析

Remark 6 给出 parameter tuning heuristic:

| 参数 | 作用 | 调大影响 | 调小影响 |
|------|------|---------|---------|
| $\alpha, \beta$ | 双幂次 gain | 缩短 $T_2$, 但 torque 更剧烈 | 延长收敛时间 |
| $p, g, k$ | 幂次 | (受约束 $pk<1, gk>1$) | 缩短 $T_2$, 但 chattering 增大, 稳态精度降 |
| $v_1$ | sliding 非线性度 | torque 变化剧烈 | singularity 风险 |
| $v_2, v_3, v_4$ | reaching 非线性 | (受约束) | torque chattering 剧烈 |
| $\sigma_1, \sigma_2, \sigma_3, \sigma_4$ | reaching gain | state/torque chattering 增大 | 收敛慢 |
| $K_s, K_\zeta$ | 线性 stabilizing gain | excite 高频 dynamics | robust margin 不够 |
| $\bar d$ | robust sign term | 必须 > disturbance 幅值 | 无法吞掉 disturbance |

实验观察 (图9, 10) 验证 $\alpha$ 增大缩短收敛时间。

---

## 八、与 Related Work 对比

### 8.1 vs. Finite-time RL Control ([29]-[32])

| 维度 | Finite-time RL | This paper (Fixed-time RL) |
|------|--------------|--------------------------|
| 收敛时间 | 依赖初始状态 | 上界 $T_{max}$ 独立于初始状态 |
| Large initial error | 收敛慢 | 收敛快 (双幂次 $|e|^p$, $p<1$) |
| Small initial error | 收敛快 | 收敛快 (双幂次 $|e|^g$, $g>1$) |
| Theoretical guarantee | 渐近 | 显式时间上界 |

### 8.2 vs. ESO-based NTSM (Sun & Liu 2019, [5])

[5] 用 Extended State Observer (ESO) 估计 modeling uncertainty $\pmb l(t)$, 这里用 actor NN。ESO 是 model-based observer, 收敛速度受 observer gain 限制; actor NN 是 function approximator, 在线学习不确定性的 spatial structure, 长期来看更 efficient。

仿真数据显示 ESO-NTSM ITAE 是 RL-FTNTSM 的 ~5 倍, 证实 RL 学习优势。

### 8.3 vs. Saturated Backstepping with NN (Liu et al 2016, [34])

[34] 用 NN approximation + saturated adaptive backstepping, 没有 optimization 概念, 忽略 cost function。本文 actor-critic 结构显式优化 cost-to-go, control strategy 是 optimal 的 (在 NN approximation 范围内)。

### 8.4 vs. Adaptive Fixed-time Convergent Observer (Basin et al 2020, [19])

[19] 关注 disturbance 估计, 不增加 controller gain。本文 NN 直接逼近 uncertainty $\pmb l(t)$ 而非只 estimate disturbance, 结构上更 general (因为 $\pmb l$ 还包含 model uncertainty)。

---

## 九、Potential Weakness 与 Open Questions

1. **NN approximation error 上界 $k_{\varepsilon i}$ 假设**: 公式 (41) 后假设 $\varepsilon_i \le k_{\varepsilon i}$, 但 RBF NN 在 training 早期的 approximation error 可能较大。是否在 transient phase 保证 $\varepsilon_i$ 小? 论文没讨论。
2. **RBF 中心 $\pmb\mu_i$ 和 width $\chi_k$ 是预设固定**: 不是 adaptive。如果 uncertainty 在 state space 中分布不均匀, 固定 RBF 可能 inefficient。可以考虑用 deep NN 或 adaptive RBF (Sanner & Slotine style)。
3. **Weight projection 上界 $\bar{\pmb w}$ 选择**: 没给出系统化方法, 实际需 trial-and-error (Remark 7 提到 $K_s$ 也需 trial-and-error)。
4. **Computational complexity**: 每个关节有独立 actor NN (m weights) + 共享 critic NN (r weights), 在线计算 RBF + gradient, 在 6-DOF 机械臂上可能挑战实时性 (50 Hz 或更高 control rate)。
5. **Actuator saturation 处理**: anti-windup compensator 假设 $\pmb\tau_\Delta$ 可计算 (通过 $\pmb\tau - \pmb\tau_0$), 实际中如果 actuator 内部 saturation 不直接 measurable (e.g., 通过 PWM limit 间接), 这个信号可能 noisy 或 delayed。
6. **No safety consideration**: 没有讨论 state constraint (joint limit, workspace), 在 real robot 上需要 barrier function 或 reference governor 配合。
7. **Sample efficiency vs convergence time trade-off**: fixed-time framework 强制 fast convergence, 可能迫使 NN 在 transient phase 学习不充分, 导致 weight 进入了次优 region, 之后难以 escape。

---

## 十、关键 Reference 与延伸阅读

1. **Polyakov 2012** (Fixed-time 原始论文): https://doi.org/10.1109/TAC.2012.2180608 - fixed-time stability 的 foundational paper, Lemma 2 即源自此
2. **Vamvoudakis & Lewis 2010** (Actor-critic online optimal control): https://doi.org/10.1016/j.automatica.2010.02.018 - continuous-time actor-critic RL 的经典 framework
3. **Bhat & Bernstein 1999** (Finite-time stability): https://doi.org/10.1137/S0363012997325329 - finite-time Lyapunov stability 理论基础
4. **Sun & Liu 2019** (ESO-NTSM 对比基线): https://doi.org/10.1016/j.ymssp.2019.106374 - paper 中 controller (36) 的来源
5. **Tran & Kang 2016** (RBF-NTSM 实验对比基线): https://doi.org/10.1007/s12541-016-0113-5 - paper 中 controller (38) 的来源
6. **He et al 2020** (Flexible manipulator RL): https://doi.org/10.1109/TSMC.2020.2975232 - single-link flexible RL 控制, 类似 framework
7. **Zuo & Tie 2014** (Lemma 1 来源): https://doi.org/10.1080/00207179.2013.822566 - multi-agent finite-time consensus, Lemma 1 的 inequality 出处
8. **Huang & Jia 2017** (NFFTSM): https://doi.org/10.1049/iet-cta.2017.0477 - non-singular fast fixed-time sliding mode, paper 中 sliding surface (19) 的灵感来源

---

## 十一、整体 Summary

这篇 paper 的核心 contribution 在于把 **RL-based adaptive control** 从 finite-time framework 推进到 fixed-time framework, 并显式处理 actuator saturation。技术上, fixed-time property 来自两层双幂次设计:

1. **Reaching law** $\text{sig}^{v_4}(\sigma_1\text{sig}^{v_2}(\pmb s) + \sigma_2\text{sig}^{v_3}(\pmb s))$ 保证 sliding surface $\pmb s$ 在 fixed time收敛
2. **Sliding surface** $K(\pmb e_1) = (\alpha|e_1|^p + \beta|e_1|^g)^k$ 保证 sliding phase 也在 fixed time收敛

两层 cascade 的总时间 $T_{max} = T_1 + T_2$ 完全由设计参数决定。Actor-Critic RL 在线逼近 modeling uncertainty 并优化 cost-to-go, projection algorithm 保证 NN weight 有界, anti-windup compensator 实时抵消 saturation residual。

实验上, 与 ESO-NTSM 和 RBF-NTSM 对比都显示明显改善, 验证 RL + fixed-time + anti-windup 三者结合的协同效应。Future work 方向包括 complex flexible structures (paper Section V 提及), 个人认为还可以扩展到 multi-agent manipulator coordination 和 human-robot collaboration 场景 (where fixed-time guarantee 对 safety 至关重要)。
