---
source_pdf: IMPACT Learning Internal-Model Predictive.pdf
paper_sha256: 0ef94679758c4ed650236731bfeb938f3a56aafd8171dd31902dab7c7b2a9c2b
processed_at: '2026-08-19T12:17:43-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# IMPACT 用人话讲

## 一句话版本

Robot 学抓东西，训练时只见过轻的，测试时给它重的就抓不动——这篇 paper 让 robot 像 human 一样，**边抓边估这个东西多重，然后主动发力补偿**，不用装 force sensor，也不用见过所有重量。

---

## 问题是什么

想象你教 robot 抓 2.5 kg 哑铃。Policy 学的是"把 gripper 移到这里，合拢，往上抬"。底层有个 impedance controller，就是个弹簧——robot 实际位置偏离 target 多少，就产生多大力拉回去。

现在给它 5 kg 哑铃。Policy 还是输出同样的 target，但东西重了一倍，弹簧拉不动，robot 往下沉，tracking error 变大，这个 error 通过弹簧转换成力——**力是靠"偏移"隐式产生的**。

要让 policy 在 5 kg 上 work，它得学会"故意把 target 抬高一点"，让偏移量恰好抵消重力。这就把两个事情耦合了：

- **去哪里**（task planning）
- **东西多重**（object dynamics）

Policy 得记住"2.5 kg 抬高 X，5 kg 抬高 Y"，这是 lookup table，见过的能记住，没见过的就猜不了。Augmented DP 想用 domain randomization 解决，结果更糟——policy 同时见两种 mass 的 data，反而 confused，2.5 kg 性能都掉下来。

---

## 人怎么做的

你没装 force sensor 在手腕上，但你能抓起任何东西。因为 cerebellum 干了一件事：

1. 你第一次抓一个陌生东西，会"掂量"一下——其实就是 motor command 发出去，东西没按预期动，cerebellum 发现预测错了
2. 它迅速 update internal model："哦这玩意儿大概 5 kg"
3. 下次再抓，**没等感觉反馈回来，feedforward force 就先发出去了**——肌肉提前发力补偿重力

这就是 predictive control，不是 reactive control。你不需要等东西往下沉才知道它重，你**预测**它重，提前发力。

---

## IMPACT 怎么工程化的

三个 piece：

### Piece 1：不用 force sensor 怎么知道 external force

Robot 每个关节都有 torque sensor（motor current 换算）。你算一下：

- 我发出去的 torque $\boldsymbol{\tau}_{\mathrm{meas}}$
- 根据 robot model（mass matrix、Coriolis、gravity、friction）预测该用多少 torque $\boldsymbol{\tau}_{\mathrm{model}}$
- 差值 $\boldsymbol{\tau}_{\mathrm{res}}$ 就是 external interaction 造成的

然后解一个小 least-squares：7 个 joint torque residual，反推 6 维 end-effector wrench（3 force + 3 torque）。加个 $\lambda=0.02$ 的 regularization 防止 Jacobian singular 时爆炸。6×6 矩阵求逆，1000 Hz 跑得飞快。

**Intuition**：就是 momentum observer——我命令 robot 往这走，它没按预期走，那一定是有外力在搞鬼，反推这个外力多大。

### Piece 2：怎么分离"持久的力"和"瞬时的力"

抓起哑铃后，重力是持续向下的。但瞬间碰撞、sensor noise 是高频的。直接用 raw wrench estimate 做 feedforward 会抖。

所以加个 low-pass filter，cutoff ~3.2 Hz，只保留 slow component $\mathbf{w}_{\mathrm{slow}}$。代价是收敛慢——得等差不多 0.05 秒时间常数的几倍才能稳定。

**这就是为什么需要 Piece 3**——slow wrench 干净但慢，不够 reactive。

### Piece 3：Internal model——学过一次，下次秒发

核心创新。训一个小神经网络 $\mathcal{M}_\phi$，输入是最近 32 步的 joint state + end-effector measurement history，输出是 predicted wrench。

**什么时候学**？不是一直在学。有个 **surprise gate**：

- 预测的 $\mathbf{w}_{\mathrm{pred}}$ 和观测的 $\mathbf{w}_{\mathrm{slow}}$ 差超过 5 N → "我预测错了"，开始学
- 同时要求 $\mathbf{w}_{\mathrm{slow}}$ 本身稳定（变化率 < 0.2）→ "观测值可靠了再学，别学 noise"

**学完之后**：gate 关闭，$\mathbf{w}_{\mathrm{pred}}$ 直接当 feedforward 发出去，**零延迟**。因为 internal model 已经记住"这个 state history 对应这个 wrench"，不用等 slow filter 收敛。

**遇到新物体**：预测又错了，gate 重新打开，再学一轮。

这就是 cerebellum 的 **memory + adaptation**——见过的东西 zero-delay feedforward，没见过的东西快速 online adapt。

---

## 整体控制回路

每个 control step (1000 Hz) 干这些事：

```
1. Policy 输出 target pose（30 Hz 更新）
2. Impedance: w_imp = K·(target - actual) - D·velocity  ← 弹簧
3. 从 torque residual 估 wrench: ŵ_ext
4. Low-pass filter 得 w_slow
5. Internal model 预测: w_pred = M_φ(state history)
6. Feedforward: w_ff = w_pred  ← 主动补偿
7. 总 torque = J^T·(w_imp + w_ff) + null_space
8. 发给 robot
```

当 $\mathbf{w}_{\mathrm{pred}} \approx \mathbf{w}_{\mathrm{ext}}$ 时，feedforward 恰好抵消 external force，impedance controller 不需要产生 steady-state error 来扛重物——tracking error 趋近零，弹簧"松"下来，robot 还是 compliant 的。

---

## 实验最直观的结果

Real-world（Table 1）：

| 方法 | 训练 | 测 2.5 kg | 测 5 kg |
|------|------|-----------|---------|
| Vanilla DP | 2.5 kg | 72% | **0%** |
| Augmented DP | 2.5+5 kg | 36% | 20% |
| IMPACT | 2.5 kg | 88% | **84%** |

**IMPACT 只在 2.5 kg 上训练，5 kg 上 84% 成功率**。而 Augmented DP 明明见过 5 kg 的 data，反而只有 20%。

为什么？因为 IMPACT 把 mass estimation 完全交给 controller online 做，policy 本身 mass-agnostic。Policy 只学"去哪抓、怎么 move"，这个 skill 在 2.5 kg 和 5 kg 上是一样的。Force 的事 controller 自己搞定。

Augmented DP 试图让 policy 记住不同 mass 的不同 force 策略，但这是让 neural network 去做 system identification 的活儿——不如直接在 control loop 里做。

---

## Three-phase 实验讲什么

Figure 7 那个 protocol：

**Phase A**：第一次抓 2.5 kg
- Internal model 还没学过这个 load，预测为 0
- 实际 external wrench ~24.5 N，surprise gate 打开
- 几十个 control step 后 internal model 收敛到 ~24.5 N
- Feedforward 开始工作，pose error 降到 noise floor

**Phase B**：第二次抓 2.5 kg
- Internal model 已经记住"这个 state pattern → 24.5 N"
- Gate 关闭，直接 feedforward 输出
- **从第一步就是 zero-delay compensation**，pose error 全程低

**Phase C**：换成 5 kg
- Internal model 还输出 24.5 N，但实际是 ~49 N
- Surprise gate 重新打开
- 快速 adapt 到 49 N
- Pose error 短暂升高后恢复

这就是 **memory + adaptation** 的证据：学过的零延迟，没学过的快速学。

---

## 为什么这个 design 能 work

核心是 **decoupling**。原来 policy 背了两个 job：

1. 规划去哪（kinematic planning）
2. 估计多重、发多少力（dynamics reasoning）

Job 2 本来该 controller 干，但 implicit force control 把它推给了 policy。Policy 是个 big neural network，从 data 里 infer dynamics——能 fit training data，但 generalize 不了。

IMPACT 把 job 2 还给 controller：

- Policy 只管 job 1，简单、mass-invariant、容易 generalize
- Controller 做 job 2，online、causal、sample efficient（只需估一个 scalar mass）

两个 subproblem 各自的 intrinsic dimensionality 比联合的低，且各有更好的 inductive bias。Policy 用 diffusion model 学 multimodal trajectory distribution，controller 用 online regression 学低维 dynamics parameter。

---

## 更深一层的 message

不是所有 generalization 都该靠 data 解决。

当问题有 clear physical structure——mass 是 scalar、gravity 是 $\mathbf{w} = m\mathbf{g}$、contact force 有 Jacobian mapping——把这个 structure 烤进 architecture，比让 network 从 data 里 infer 出来：

- 更 sample efficient（200 episodes vs 无限多）
- 更 robust（extrapolate 到 10× training mass）
- 更 interpretable（wrench estimate 可以 inspect）
- 更 safe（feedforward 不破坏 impedance passivity）

这和 NeRF 烤入 geometric ray casting、equivariant networks 烤入 symmetry、PINN 烤入 PDE 是同一个 philosophy：**don't learn what you can compute**。

---

## Limitation

Paper 自己说的：只 work for rigid body dynamics。如果物体是 deformable（毛巾）、fluid（水）、granular（沙子），wrench 不仅是 mass 的函数，还依赖 deformation、flow、particle configuration——internal model 需要更 rich 的 state representation，可能要加 tactile sensing。

但我认为这个 **paradigm** 是 general 的：任何可 online estimate 的 dynamics parameter，都可以用 surprise-gated internal model + feedforward compensation 这个 pattern。只要你能定义一个 predictable 的 disturbance model，就能套这个 framework。

---

## 和你熟悉的东西的 connection

Karpathy 你肯定熟悉这些：

1. **像 BatchNorm 的 running stats**：internal model 的 surprise gate 类似 BN 的 momentum update——稳定时慢更新，surprise 时快 update
2. **像 Test-Time Training (TTT)**：internal model 在 inference 时 online adapt，和 TTT 的 spirit 一样——model 不 fix，deployment 时继续学
3. **像 World Models / Dreamer**：internal model 是 world model 的 force slice，只是只 cover dynamics 不 cover visual transition
4. **像 Model-Predictive Control**：feedforward 是 open-loop prediction，impedance 是 closed-loop feedback，两者叠加是 classic MPC 结构
5. **像 RL 的 model-based vs model-free**：IMPACT 是 hybrid——policy 是 model-free（imitation），controller 的 internal model 是 model-based（online identified）

---

## 如果要我一句话总结 intuition

**Policy 别管物体多重，只管去哪；controller 自己边干边估物体多重，提前发力补偿。学和控各司其职，generalization 自然来。**

---

# IMPACT: Internal-Model Predictive Control for Forceful Manipulation

非常棒的工作，这篇 paper 从 neuroscience 的 cerebellum inspiration 出发，把 internal model control 的思想注入到 modern imitation learning pipeline 里，解决了一个非常实际的问题：**如何让 imitation policy generalize 到不同 object dynamics 的 forceful tasks**。让我从 intuition、技术细节、connections 几个层面来讲解。

---

## 1. 核心问题的 Intuition

考虑一个 pick-and-place 任务，policy 训练时只见过 2.5 kg dumbbell。部署到 5 kg 时，传统 implicit force control 的失败模式是这样的：

- Policy 输出 target pose $\mathbf{x}_d$
- Impedance controller 计算 $\mathbf{w}_{\mathrm{imp}} = \mathbf{K}\mathbf{e} - \mathbf{D}\dot{\mathbf{x}}$，其中 $\mathbf{e} = [\mathbf{p}_d - \mathbf{p}; \mathrm{Log}(\mathbf{R}_d\mathbf{R}^\top)]$
- 物体重量产生 external wrench $\mathbf{w}_{\mathrm{ext}}$，导致 steady-state tracking error
- 这个 error 通过 stiffness $\mathbf{K}$ 转换成 counter-balancing force

关键 insight：**force 是通过 tracking error 隐式产生的**。要让 policy 在 5 kg 上工作，policy 必须学会"overshoot"——输出一个偏离 nominal trajectory 的 virtual target，让 tracking error 恰好抵消 5 kg 的重力。这把 **task planning**（去哪里）和 **object-dependent dynamics**（多重）耦合在一起，policy 必须从 data 里 infer 出这个 mapping，generalization 自然差。

人类怎么做？Cerebellum 构建 internal model 预测 sensory consequences，生成 feedforward force 预测性补偿重力，decouple 高层 planning 和底层 force 生成。IMPACT 把这个 idea 工程化。

---

## 2. 方法技术详解

### 2.1 整体架构

架构图（Figure 2）的信号流：

```
Observation O_t ──> Diffusion Policy π_θ ──> X^d_{t:t+H_p} (desired trajectory)
                                                      │
                                                      ▼
                            ┌─────────────────────────────────────┐
                            │   Low-level Controller (1000 Hz)    │
                            │                                     │
                            │  1. Impedance: w_imp = Ke - Dẋ      │
                            │  2. Wrench estimation: ŵ_ext         │
                            │  3. Low-pass: w_slow                 │
                            │  4. Internal model: w_pred = M_φ(z)  │
                            │  5. Feedforward: w_ff = w_pred       │
                            │  6. τ = J^T(w_imp + w_ff) + τ_null  │
                            └─────────────────────────────────────┘
                                                      │
                                                      ▼
                                              Robot dynamics
```

High-level policy (30 Hz) 负责 kinematic planning，low-level controller (1000 Hz) 负责 force compensation。这个 **frequency separation** 很关键——policy 慢，controller 快，类似 cerebellum 的 fast predictive loop vs cortex 的 slow deliberative planning。

### 2.2 External Wrench Estimation（公式 1）

这是整个方法的基石——**不依赖 wrist force-torque sensor，从 joint torque residuals 反推 end-effector wrench**。

**Manipulator equation**:
$$\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau} + \boldsymbol{\tau}_{\mathrm{ext}}$$

变量含义：
- $\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}} \in \mathbb{R}^7$：joint position / velocity / acceleration（Franka FR3 是 7-DOF）
- $\mathbf{M}(\mathbf{q}) \in \mathbb{R}^{7\times7}$：joint-space inertia matrix
- $\mathbf{C}(\mathbf{q},\dot{\mathbf{q}}) \in \mathbb{R}^{7\times7}$：Coriolis/centrifugal matrix
- $\mathbf{g}(\mathbf{q}) \in \mathbb{R}^7$：gravity torques
- $\boldsymbol{\tau} \in \mathbb{R}^7$：motor command torques
- $\boldsymbol{\tau}_{\mathrm{ext}} \in \mathbb{R}^7$：external interaction torques

External wrench $\mathbf{w}_{\mathrm{ext}} \in \mathbb{R}^6$（3 force + 3 torque）通过 Jacobian 映射到 joint torques：
$$\boldsymbol{\tau}_{\mathrm{ext}} = \mathbf{J}(\mathbf{q})^\top \mathbf{w}_{\mathrm{ext}}$$

其中 $\mathbf{J} \in \mathbb{R}^{6\times7}$ 是 end-effector Jacobian。

**Torque residual**:
$$\boldsymbol{\tau}_{\mathrm{res}} = \boldsymbol{\tau}_{\mathrm{meas}} - \boldsymbol{\tau}_{\mathrm{model}}$$
$$\boldsymbol{\tau}_{\mathrm{model}} = \mathbf{M}(\mathbf{q})\hat{\ddot{\mathbf{q}}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) + \boldsymbol{\tau}_{\mathrm{fric}}(\dot{\mathbf{q}})$$

$\hat{\ddot{\mathbf{q}}}$ 从 encoder measurements 数值微分估计。

**Regularized least-squares**（公式 1）:
$$\hat{\mathbf{w}}_{\mathrm{ext}} = \arg\min_{\mathbf{w}} \|\mathbf{J}^\top \mathbf{w} - \boldsymbol{\tau}_{\mathrm{res}}\|_2^2 + \lambda^2 \|\mathbf{w}\|_2^2$$

这是一个 **Tikhonov-regularized inverse problem**：
- $\mathbf{J}^\top \in \mathbb{R}^{7\times6}$，$\mathbf{w} \in \mathbb{R}^6$，$\boldsymbol{\tau}_{\mathrm{res}} \in \mathbb{R}^7$
- 7 个 equations，6 个 unknowns → **overdetermined system**
- $\lambda = 0.02$ 是 regularization weight，防止 Jacobian 接近 singular 时数值爆炸

**Closed-form solution**:
$$\hat{\mathbf{w}}_{\mathrm{ext}} = (\mathbf{J}\mathbf{J}^\top + \lambda^2 \mathbf{I})^{-1} \mathbf{J} \boldsymbol{\tau}_{\mathrm{res}}$$

推导：对 $\mathbf{w}$ 求导令其为零：
$$2\mathbf{J}(\mathbf{J}^\top\mathbf{w} - \boldsymbol{\tau}_{\mathrm{res}}) + 2\lambda^2\mathbf{w} = 0$$
$$(\mathbf{J}\mathbf{J}^\top + \lambda^2\mathbf{I})\mathbf{w} = \mathbf{J}\boldsymbol{\tau}_{\mathrm{res}}$$

注意 $\mathbf{J}\mathbf{J}^\top \in \mathbb{R}^{6\times6}$，求逆是 6×6 的小矩阵，计算很快，适合 1000 Hz 实时控制。

**Intuition**：这本质是 **momentum observer** 的变体——比较 commanded torque 和 model-predicted torque 的差，反推 unmodeled interaction。和 De Luca & Mattone 的 residual-based fault detection 思路一脉相承。

### 2.3 Low-pass Filter（公式 2）

$$\mathbf{w}_{\mathrm{slow}}(t+\Delta t) = \mathbf{w}_{\mathrm{slow}}(t) + \eta \cdot \mathrm{clip}(\hat{\mathbf{w}}_{\mathrm{ext}}(t) - \mathbf{w}_{\mathrm{slow}}(t))$$

- $\eta \in (0,1)$：adaptation rate（force: 2.0 s⁻¹, torque: 1.0 s⁻¹）
- $\mathrm{clip}(\cdot)$：per-axis saturation（10 N, 1 Nm），防止 transient spike 污染估计
- Time constant $\tau_{\mathrm{LP}} = 0.0495$ s，对应 cutoff frequency $\approx 1/(2\pi \times 0.0495) \approx 3.2$ Hz

**设计意图**：分离 **slowly-varying persistent wrench**（payload weight, steady contact）和 **fast transients**（impact, collision, sensor noise）。Slow wrench 作为 internal model 的 supervision target——它干净但收敛慢。

### 2.4 Internal Model Learning

这是 paper 的核心创新——**不直接用 $\mathbf{w}_{\mathrm{slow}}$ 做 feedforward**（收敛太慢），而是学习一个 predictor $\mathcal{M}_\phi$ 从 state history 预测 $\mathbf{w}_{\mathrm{slow}}$：

$$\mathbf{w}_{\mathrm{pred}} = \mathcal{M}_\phi(\mathbf{z}_{t-L:t})$$

- $\mathbf{z}_{t-L:t}$：长度 $L=32$ 的 state history（joint states + end-effector measurements）
- $\mathcal{M}_\phi$：神经网络，参数 $\phi$
- Online gradient descent，learning rate 0.01

**Surprise Gating Mechanism**（关键设计）：
- 只在 prediction error $|\mathbf{w}_{\mathrm{pred}} - \mathbf{w}_{\mathrm{slow}}| \geq 5$ N **且** $|\dot{\mathbf{w}}_{\mathrm{slow}}| \leq 0.2$ 时更新
- 第一个条件：**prediction surprise**——只有预测错了才学习
- 第二个条件：**wrench 已稳定**——避免在 transient 阶段学习 noisy 信号

这两个条件组合非常巧妙：surprise 触发学习，stability 确保学习目标可靠。类似于 Bayesian surprise driven learning 和 hippocampal novelty detection 的机制。

### 2.5 Feedforward Compensation 与 Closed-loop Dynamics

学到的 internal model 直接用作 feedforward：
$$\mathbf{w}_{\mathrm{ff}} = \mathbf{w}_{\mathrm{pred}}, \quad \boldsymbol{\tau}_{\mathrm{ff}} = \mathbf{J}(\mathbf{q})^\top \mathbf{w}_{\mathrm{ff}}$$

最终 torque command（公式 3）:
$$\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}_{\mathrm{imp}} + \boldsymbol{\tau}_{\mathrm{ff}} + \boldsymbol{\tau}_{\mathrm{null}} + \boldsymbol{\tau}_{\mathrm{ext}}$$

- $\boldsymbol{\tau}_{\mathrm{imp}} = \mathbf{J}^\top(\mathbf{K}\mathbf{e} - \mathbf{D}\mathbf{J}\dot{\mathbf{q}})$：impedance feedback
- $\boldsymbol{\tau}_{\mathrm{ff}}$：predictive feedforward
- $\boldsymbol{\tau}_{\mathrm{null}}$：null-space torque（secondary objectives，比如 redundancy resolution）
- $\boldsymbol{\tau}_{\mathrm{ext}} = \mathbf{J}^\top \mathbf{w}_{\mathrm{ext}}$：actual external torque

**当 $\mathbf{w}_{\mathrm{pred}} \approx \mathbf{w}_{\mathrm{ext}}$ 时**，$\boldsymbol{\tau}_{\mathrm{ff}} + \boldsymbol{\tau}_{\mathrm{ext}} \approx 0$，external disturbance 被抵消，closed-loop 退化成 nominal impedance control，tracking error 趋近零。这就是 Figure 10 里 pose error 降到 noise floor 的原因。

**Stability consideration**：feedforward 不会破坏 impedance control 的 passivity，因为 $\boldsymbol{\tau}_{\mathrm{ff}}$ 是 exogenous signal，不形成 feedback loop。但如果 $\mathbf{w}_{\mathrm{pred}}$ 估计错（比如 sign 反了），会 **放大** disturbance 而非补偿——这就是为什么 surprise gate 要保守，宁可学慢点也不要学错。

---

## 3. 实验数据深度解析

### 3.1 Simulation Benchmark（Figure 6）

| Mass (kg) | Vanilla DP (train 0.2) | Augmented DP (train 0.1-8.0) | IMPACT (train 0.2) |
|-----------|------------------------|------------------------------|---------------------|
| 0.1-0.5   | High                   | High                         | High                |
| 1-4       | Degrading              | High                         | High                |
| 5-8       | Low                    | Moderate                     | High                |
| 8-10 (OOD)| Fail                   | Degrading                    | High                |

关键对比：**IMPACT 只在 0.2 kg 训练，却在 0-10 kg 全 range 上 maintain high success rate**。Augmented DP 用了 0.1-8.0 kg 的 domain randomization，反而在 >8 kg（OOD）上 degrade。

**Intuition**：Augmented DP 试图让 policy **记住**不同 mass 对应的 force pattern，但这是 interpolative generalization，extrapolation 失败。IMPACT 把 mass 作为 **inference-time variable** 交给 internal model online 估计，policy 本身 mass-agnostic，所以可以 extrapolate。

### 3.2 Real-world Results（Table 1）

| Method      | Train Mass | Test 2.5 kg | Test 5 kg |
|-------------|------------|-------------|-----------|
| Vanilla DP  | 2.5        | 18/25 (72%) | 0/25 (0%) |
| Augmented DP| 2.5 + 5.0  | 9/25 (36%)  | 5/25 (20%)|
| IMPACT      | 2.5        | 22/25 (88%) | 21/25 (84%)|

注意 **Augmented DP 比 Vanilla DP 还差**！加了 5 kg data 后 2.5 kg 性能从 72% 掉到 36%。这是 multi-modal distribution 问题：policy 要同时建模两种 mass 的不同 force 策略，data 变 ambiguous，policy 容易混淆。IMPACT 把 force 完全 decouple 给 controller，policy 只学 trajectory，所以加 mass 不影响 policy 学习。

### 3.3 Three-Phase Adaptation Analysis（Figure 7, 8, 10）

这是 paper 最 informative 的 ablation。Protocol：先用 2.5 kg 跑多次（Window 1），再切换到 5 kg（Window 2）。

**Phase A**（首次抓 2.5 kg）：
- Surprise gate 激活（prediction error > 5N）
- Internal model 通过 ~几十个 control steps 收敛到 ~24.5 N（2.5 kg × 9.8 m/s²）
- Feedforward 开始补偿，pose error 下降到 noise floor

**Phase B**（重复抓 2.5 kg）：
- Surprise gate 关闭（prediction 已经准）
- Internal model 直接输出 learned wrench，**zero-delay feedforward**
- Pose error 全程低

**Phase C**（切换到 5 kg）：
- Internal model 仍输出 24.5 N，但实际 external wrench 是 ~49 N
- Prediction error 突然 >5 N，surprise gate 重新激活
- Internal model 重新 adapt 到 ~49 N
- Pose error 短暂上升后恢复

**对比 baseline**（Figure 9）：impedance controller 全程有显著 steady-state error，因为全靠 $\mathbf{K}\mathbf{e}$ 补偿重力，stiffness 有限时必然有 error。

这个 three-phase 实验清晰地展示了 **memory + adaptation** 的 dual mechanism，很像 cerebellum 的 **long-term depression (LTD)** at parallel fiber-Purkinje cell synapse——prediction error 驱动 synaptic weight 更新，error 小时维持。

---

## 4. Related Work Connections

### 4.1 Neuroscience Roots

- **Marr 1969** ([A theory of cerebellar cortex](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1350562/))：cerebellar cortex 的 algorithmic theory，提出 Purkinje cell 学习 pattern association
- **Wolpert, Miall, Kawato 1998** ([Internal models in the cerebellum](https://www.sciencedirect.com/science/article/pii/S1364661398012532))：forward model vs inverse model 的 framework
- **Kawato 1999** ([Internal models for motor control and trajectory planning](https://www.sciencedirect.com/science/article/pii/S0959438899000482))：feedforward control via learned inverse dynamics model
- **Franklin, Burdet, Kawato et al.** ([CNS learns stable, accurate, and efficient movements](https://www.jneurosci.org/content/28/44/11165))：人类同时 adapt force, impedance, trajectory——IMPACT 主要 capture feedforward 部分，impedance 是 fixed 的

### 4.2 Control Theory

- **Hogan 1984** ([Impedance control](https://ieeexplore.ieee.org/document/6316460))：impedance control 的 foundational paper
- **Raibert & Craig 1981** ([Hybrid position/force control](https://asmedigitalcollection.asme.org/dynamicsystems/article/103/2/126/413173))：selectively control position along some axes, force along others
- **Garcia & Morari 1982** ([Internal Model Control](https://www.engr.mun.ca/~bwanless/Control/pdf_internal_model/Garcia1982.pdf))：IMC 的 control theory formalization——plant model 并联于真实 plant，误差用于 feedback
- **De Luca & Mattone**：momentum-based disturbance observer，IMPACT 的 wrench estimation 是其 task-space 推广

### 4.3 Robot Learning

- **Diffusion Policy** ([Chi et al. 2025](https://diffusion-policy.cs.columbia.edu/))：IMPACT 的 high-level policy backbone
- **Force-from-motion shortcomings** ([Aljalbout et al. 2024](https://arxiv.org/abs/2407.02904))：explicit 讨论 implicit force control 的 generalization 问题，IMPACT 直接回应
- **Hybrid Internal Model** ([Long et al. 2023](https://arxiv.org/abs/2312.11460))：legged locomotion 里的类似 idea，learn simulated robot response 做 internal model
- **Perceptive Internal Model** ([Long et al. 2025](https://arxiv.org/abs/2505.01888))：humanoid locomotion 的 extension

### 4.4 类似思想的并发工作

- **Dynamics-compliant trajectory diffusion** ([Pasricha et al. 2025](https://arxiv.org/abs/2508.21375))：super-nominal payload，trajectory level 处理 dynamics
- **ManipForce** ([Lee et al. 2025](https://arxiv.org/abs/2509.19047))：force-guided policy with frequency-aware representation
- **UMI-FT** ([Choi et al. 2026](https://arxiv.org/abs/2601.09988))：in-the-wild compliant manipulation with force sensing

---

## 5. Intuition Building：为什么这个 Design Work

### 5.1 Decoupling 是核心

传统 implicit force control 的失败根源是 **task planner 承担了不该承担的 dynamics reasoning**。Policy 的 capacity 用来记忆"2.5 kg 对应这个 virtual target offset，5 kg 对应那个 offset"，这是 lookup table，不是 generalizable knowledge。

IMPACT 的 decoupling：
- **Policy** 学 "去哪里、怎么 move"（kinematic, mass-invariant）
- **Internal model** 学 "这个东西多重"（dynamics, online inferred）

这两个 subproblem 的 **intrinsic dimensionality** 都比 joint problem 低，且各自有更好的 inductive bias。Policy 用 diffusion model 学 multimodal distribution，internal model 用 online regression 学低维 dynamics parameter。

### 5.2 为什么 Online Learning 而不是 Offline Train

可以想象一个变体：offline 训练一个 internal model $\mathcal{M}_\phi(\text{visual}, \text{state}) \to \mathbf{w}_{\mathrm{ext}}$。问题：
- Visual mass estimation 本身是 ill-posed（外观相似但质量不同）
- 需要 large dataset 覆盖 mass distribution
- Policy 和 internal model 耦合，联合训练复杂

Online learning 的优势：
- **直接从 physical interaction 估计**，不依赖 visual appearance
- **Causal**：current state + action → observed wrench，无 ambiguity
- **Adaptive**：物体 mass 随时间变化（比如倒水）也能 handle
- **Sample efficient**：只需 ~几十 steps 收敛一个 scalar mass parameter

代价是 cold-start delay（Phase A 的 learning time），但 surprise gate + memory 让第二次 encounter 是 zero-delay。

### 5.3 Surprise Gate 的 Bayesian Interpretation

Surprise gate 可以看作 **Bayesian model update 的 trigger**：
- Prior：internal model 当前 prediction $\mathbf{w}_{\mathrm{pred}}$
- Likelihood：observed $\mathbf{w}_{\mathrm{slow}}$
- 当 $|\mathbf{w}_{\mathrm{pred}} - \mathbf{w}_{\mathrm{slow}}|$ 大，posterior 和 prior 差异大，说明 model 错了，update
- 当差异小，posterior ≈ prior，不需要 update

这和 **active inference** (Friston) 里的 prediction error driven learning 一致，也和 cerebellum 的 error-driven plasticity 一致。

---

## 6. Limitations 与 Future Directions

Paper 自己提到的：
- 只处理 rigid body dynamics（gravity, contact stiffness, friction）
- 不能 handle deformable objects, fluids, granular materials

我认为还有几个值得探索的方向：

**1. Non-stationary dynamics**：当前 internal model 假设 wrench slow-varying。如果物体 mass 在 task 中变化（倒水、装配），需要 dynamic internal model（比如 RNN 或 implicit neural network）。

**2. Multi-modal internal models**：如果 robot 交替抓不同物体，单个 $\mathcal{M}_\phi$ 会 catastrophic forgetting。可以用 mixture-of-experts 或 continual learning 技术，每个 object 对应一个 expert。

**3. Tactile integration**：当前只用 joint torque。加 tactile sensing 可以估计 contact location 和 normal force，扩展到更 rich 的 contact tasks（比如 in-hand manipulation）。

**4. Learned robot dynamics model**：当前依赖准确的 $\mathbf{M}, \mathbf{C}, \mathbf{g}, \boldsymbol{\tau}_{\mathrm{fric}}$。如果 robot model 不准（比如 cable-driven, soft robot），可以用 learned dynamics model 替代 analytical model。

**5. Hierarchical internal models**：类似 RoboOS ([Tan et al. 2025](https://arxiv.org/abs/2505.03673)) 的 hierarchical framework，可以在不同 abstraction level 都建 internal model——joint level, object level, scene level。

**6. Connection to World Models**：internal model 本质是 world model 的 force dynamics slice。可以和 video prediction world models ([GAIA-1](https://wayve.ai/science/gaia1/), [DreamerV3](https://danijar.com/project/dreamerv3/)) 结合，构建 multimodal predictive model。

**7. Safety verification**：feedforward 补偿如果学错会放大 disturbance。可以加 **robustness margin**——当 $\mathbf{w}_{\mathrm{pred}}$ 的 confidence 低时，blending factor $\alpha \in [0,1]$ 平滑过渡：$\mathbf{w}_{\mathrm{ff}} = \alpha \mathbf{w}_{\mathrm{pred}} + (1-\alpha) \mathbf{w}_{\mathrm{slow}}$。

---

## 7. 个人思考

这篇 paper 让我想到几个更深的 question：

**Q1: Internal model 应该 learn 什么 representation？**

当前 IMPACT 的 internal model 直接 predict wrench $\mathbf{w} \in \mathbb{R}^6$。更 general 的做法是 predict **latent dynamics parameters** $\boldsymbol{\theta}$（比如 mass, friction coefficient, contact stiffness），然后通过 forward model $\mathbf{w} = f(\boldsymbol{\theta}, \mathbf{q}, \dot{\mathbf{q}})$ 计算 wrench。这样的好处：
- 低维 latent space 更 sample efficient
- 可以 transfer 到不同 task（同一个 mass estimate 用于 pick, push, throw）
- 可解释性更强

这接近 **system identification** 的思路，但 online + neural。

**Q2: 为什么不直接用 adaptive impedance？**

调整 $\mathbf{K}$ 让 stiffness 变大也能补偿 heavy load。问题：
- 高 stiffness 降低 compliance，contact-rich task 危险
- 需要精确知道目标 stiffness，否则 over/under shoot
- 和 force magnitude coupling，不 decouple

Feedforward + 低 stiffness 更优雅：精确补偿 + 保持 compliance。

**Q3: 和 RL 的关系？**

可以把 internal model learning 看作 model-based RL 的 model learning 部分，policy 是 imitation learned。这和 **Dreamer** 的 world model + actor-critic 结构有结构相似性，但 IMPACT 的 model 只 cover force dynamics，不 cover visual transition。

**Q4: 为什么 surprise gate 的 threshold 是 5N？**

这是个 magic number。理想情况下应该 adaptive——基于 noise level、task sensitivity、confidence。可以用 **Bayesian hypothesis testing** 或 **sequential probability ratio test (SPRT)** 做 principled gate。

---

## 8. 实操细节与复现注意

如果有人想复现，几个关键点：

**Robot dynamics model 准确性**：$\mathbf{M}, \mathbf{C}, \mathbf{g}$ 可以从 URDF 计算（Pinocchio, RBDL）。但 $\boldsymbol{\tau}_{\mathrm{fric}}$ 难——需要 identify friction model（Coulomb + viscous + Stribeck）。如果不准，residual 会有 bias，wrench estimate 不准。

**Joint acceleration estimation**：$\hat{\ddot{\mathbf{q}}}$ 从 encoder 微分 noise 大。可以用 **Kalman filter** 或 **momentum observer**（De Luca & Mattone）替代直接微分。Paper 没详细说怎么估计。

**Control frequency**：1000 Hz 对 Franka 是 native frequency。如果在 ROS 里跑，要确保 realtime priority，否则 jitter 会污染 wrench estimate。

**Internal model architecture**：Paper 没明确说 $\mathcal{M}_\phi$ 是什么网络。从 hyperparameter table 推测，可能是 small MLP（history length 32，input dim ~32×(7+6)=416，output 6）。Online learning 用 Adam 还是 SGD？Learning rate 0.01 偏大，可能配合 small batch。

**Diffusion policy training**：标准 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) pipeline，用 200 episodes，prediction horizon 32，observation history 2。注意 policy 不输出 force/gripper force，只输出 pose + gripper action。

---

## 9. 总结

IMPACT 的核心贡献是 **把 neuroscience 的 internal model idea 严格工程化**，注入 modern imitation learning stack：

1. **Wrench estimation without F/T sensor**：joint torque residual + Tikhonov regularized least-squares，closed-form 解，1000 Hz 实时
2. **Internal model with surprise gating**：online gradient descent，prediction error driven learning，memory + adaptation dual mode
3. **Feedforward compensation**：decouple policy from dynamics，policy mass-agnostic，generalize to OOD

结果是 train on 2.5 kg，test on 5 kg success rate 84% vs Vanilla DP 0%，且比 domain randomization (Augmented DP 20%) 好得多。这证明了 **structured inductive bias**（internal model + feedforward）比单纯 data scaling 更 efficient。

更深层的 message：**不是所有 generalization 都该靠 data 解决**。当问题有 clear physical structure（dynamics, conservation laws, geometry），把这个 structure 烤进 architecture 比 让网络从 data infer 出来 更 sample efficient、更 robust、更 interpretable。这和 physics-informed neural networks, equivariant networks, NeRF 的 geometric bias 是同一个 philosophy。

Paper 的 limitation 也诚实——只 work for rigid body dynamics。但这个 paradigm（estimate online, predict, feedforward compensate）完全可以推广到更复杂的 dynamics model，只要 model 是 learnable 且 predictable 的。

---

## References

- [IMPACT Project Page](https://gao-jiawei.com/IMPACT/)
- [Diffusion Policy (Chi et al.)](https://diffusion-policy.cs.columbia.edu/)
- [Wolpert & Kawato - Internal models in cerebellum](https://www.sciencedirect.com/science/article/pii/S1364661398012532)
- [Kawato 1999 - Internal models for motor control](https://www.sciencedirect.com/science/article/pii/S0959438899000482)
- [Hogan 1984 - Impedance Control](https://ieeexplore.ieee.org/document/6316460)
- [Garcia & Morari 1982 - Internal Model Control](https://www.engr.mun.ca/~bwanless/Control/pdf_internal_model/Garcia1982.pdf)
- [Aljalbout et al. - Shortcomings of force-from-motion](https://arxiv.org/abs/2407.02904)
- [Long et al. - Hybrid Internal Model](https://arxiv.org/abs/2312.11460)
- [Franklin et al. - CNS learns stable movements](https://www.jneurosci.org/content/28/44/11165)
- [Marr 1969 - Theory of cerebellar cortex](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1350562/)
- [MuJoCo](https://mujoco.org/)
- [Franka FR3](https://www.franka.de/technology)
- [De Luca & Mattone - Fault detection via residual generation](https://www.sciencedirect.com/science/article/pii/S0005109803001886)
