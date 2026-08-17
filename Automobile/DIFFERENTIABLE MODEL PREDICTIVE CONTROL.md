---
source_pdf: DIFFERENTIABLE MODEL PREDICTIVE CONTROL.pdf
paper_sha256: 50cb568fcd88b39aa1011c6304481d2e5b0cd4bee427b6dae9725f9ef40fd912
processed_at: '2026-08-03T21:14:01-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DiffMPC 人话版 — 跟 Karpathy 喝咖啡聊的那种

## 一句话版本

MPC 这种"看着前方 T 步做规划"的控制算法，本来要顺着时间一步一步算（像读一本书必须一页一页翻），GPU 帮不上忙。这篇 paper 换了个算法：把"顺着算"改成"猜-检查-修正"的迭代法，每次迭代里所有时间步可以同时算。于是 GPU 终于能开干，speedup 4-7 倍，batch size 可以拉到 256，最后真的让 Toyota Supra 在水坑里漂移都不甩尾。

---

## 1. 先讲清楚 MPC 是啥，为啥要 differentiable

MPC (Model Predictive Control) 是控制界的老炮：每隔一段时间，它看着当前 state，往前预测 T 步，解一个 optimization 问题（"未来 T 步里我该踩多少油门、打多少方向盘，能让车走得最贴合 reference 又不费劲"），然后只执行第一步，下个 tick 重新规划。像开车时眼睛一直看前面 3 秒的路，不断修正。

它强在 **structure**：dynamics constraint、actuator limit、cost function 全都是 explicit 的物理知识。这跟纯 neural network policy "state in, action out" 完全不同——MPC 是"白盒 + 物理先验"。

Differentiable MPC 的 idea：把 MPC 包成一个 NN layer，让它能 backprop。这样你就能学 cost weights、学 vehicle parameters、甚至把 NN 嵌进 cost/dynamics 里，用 SGD 端到端训练。物理知识当 inductive bias，数据用来 tune 剩下的。

类比：MPC 像 transformer 里 "attention 结构是 hand-crafted inductive bias"，trainable weights 是 NN 学的。DiffMPC 让这个组合能在 GPU 上 scale。

---

## 2. 痛点在哪

Differentiable MPC 不是新概念，mpc.pytorch (Amos 2018)、trajax 都做过。但它们在 GPU 上一直跑不快。

为啥？因为传统 OCP solver 是 **Riccati recursion**：

> 从时间 T 往前扫到时间 0，每一步用上一步的 cost-to-go 算下一步。

这个 recursion 严格 sequential：$t$ 时刻依赖 $t+1$ 时刻的结果。GPU 上 batch 32 个 trajectory，每个 trajectory 内部仍然串行。GPU 几千个 core 大量 idle，kernel launch overhead 还在累积。这就是为啥 trajax 在 GPU 上比 CPU 几乎没加速（Table 3：forward 0.95s vs 0.93s，backward 1.83s vs 1.90s）。

类比：你想用 GPU 并行算 Fibonacci 数列。每项依赖前两项，根本没法并行——core 再多也救不了。

---

## 3. 核心 insight：换算法别硬刚

Riccati 是 block-tridiagonal system 的 **direct solver**（精确解）。但 tridiagonal system 也可以用 **iterative solver**（迭代逼近）。

PCG (Preconditioned Conjugate Gradient) 就是这种 iterative solver。它的核心 op 是 sparse matrix-vector multiply——而 mat-vec 可以在时间轴上完全并行（每个 $t$ 的 block 同时算）。

直觉：
- Riccati 像"从头读到尾"，必须一页一页翻
- PCG 像"猜个答案，检查所有章节同时哪里不对，修正，再猜，再检查"

每次 iteration 内部 GPU 可以 fully parallel over $t$。iteration 数量取决于 condition number，preconditioner 把它压下来就 OK。

但 preconditioner 本身也得 parallel-friendly。你不能用 Riccati 当 preconditioner（那就退化回 sequential 了）。Bu & Plancher 2024 提出的 **symmetric stair preconditioner** 解决了这个问题：

> 把时间轴分成 pairs (0,1), (2,3), ..., 每个 pair 内部 dense solve（小规模，快），pair 之间保留 tridiagonal coupling。所有 pair 同时算。

类比 multigrid：coarse grid capture 全局耦合，fine grid capture 局部误差。但 implementation 上更简单——直接基于 block 结构。

---

## 4. 几个 elegant 的设计点

### 4.1 Forward 和 backward 用同一个 KKT matrix

Forward pass 解 $K \cdot (z, \lambda) = (-b, d)$。  
Backward pass 解 $K \cdot (\tilde z, \tilde \lambda) = (\nabla_z \ell, 0)$。

**同一个 K**，只是 RHS 不同。所以 forward pass 算出来的 $G, H, S, \Phi^{-1}$ 全部 cache 给 backward 用。backward 只需要重新跑 PCG iteration（可以 warm-start 用 forward 的 $\lambda$）+ 自动微分的 chain rule。

mpc.pytorch/trajax 没法这样 reuse——它们的 forward 和 backward 是不同的 Riccati recursion。这就是为啥 DiffMPC 的 backward pass speedup 比 forward 还大（Table 3：DiffMPC backward 0.32s，trajax 1.83s，6× speedup）。

### 4.2 iLQR-style Gauss-Newton：偷懒但够用

完整 SQP 应该用 Lagrangian 的 Hessian $\nabla^2 c + \nabla^2(\lambda^\top f)$。但 paper 跟 Amos 2018 一样，只用 $\nabla^2 c$，忽略 dynamics curvature。

为啥 OK？
1. dynamics Hessian 不一定 PSD，会让 QP non-convex
2. 二阶 dynamics derivative 在 JAX 里能 autodiff 但 memory 翻倍
3. Gauss-Newton SQP 在 constraint residual 小时依然 superlinear convergence
4. RL/IL 用 SGD 训练，本来就不需要 exact gradient

代价：gradient 精度略低。但 paper 的实验显示对 RL/IL 完全够用。

### 4.3 三层 warm-starting

1. **SQP 外层**：上一个 SQP iteration 的 solution 作为下一个的 initial guess
2. **PCG 内层**：上一次 PCG 的 $\lambda$ 作为下一次 PCG 的 initial guess
3. **MPC replanning**：下一个 time step 的 OCP 用 shifted previous solution 作为 init

第三点对 real-time MPC 特别重要。Riccati 方法每次都要 cold start factorize，DiffMPC 可以 warm start，replan 频率高时优势明显。

附录 Fig.9 显示 tolerance $10^{-4}$ 时 warm-start 给 9-11% speedup。$10^{-12}$ 时只有 4%（PCG 收敛快了，warm-start 帮助小）。

### 4.4 Line search 也 parallel

每次 SQP iteration 解完 QP 后要做 line search 选 step size $\alpha$。候选 $\alpha \in \{1.0, 0.7, 0.3, 0.1, 0.01\}$，**全部 parallel evaluate** merit function，选最大 acceptable step。

小细节但很有 GPU 精神：任何"试几个 candidate"的操作都该并行。

---

## 5. Toyota Supra 漂移 — 真车验证

### 5.1 为啥 drifting 是 hard case

Drifting 时 tire force saturated，vehicle 在 unstable regime 操作。$\beta$ (sideslip angle，车头方向与速度方向的夹角) 可能 30-60 度。任何小扰动（比如水坑让 $\mu$ 从 1.0 突然降到 0.6）会迅速放大成 spin-out。

Baseline MPC 手动 tuned，nominal dry condition 下表现很好。但 puddle 一来就崩。

### 5.2 学什么

$\theta = (\text{diag}(Q), \text{diag}(R), \mu_f, \mu_r, C_f, C_r, I_w) \in \mathbb{R}_{>0}^{14}$

14 个参数：cost diagonal + 物理参数。Log-normalize 保持 positivity。

RL reward: $R(x) = -((r - r_{\text{ref}})^2 + 0.3 \cdot e^{-50 \beta^2})$

第二项很巧妙：$e^{-50 \beta^2}$ 在 $\beta = 0$ 时最大（=1），前面加负号变成 penalty。即鼓励 $\beta \neq 0$（必须漂移）。但 $\beta$ 大时 penalty saturate，避免 unbounded。

作者特别强调：**没这个 sideslip reward，学到的 policy 会"作弊"——稳稳 drive on path 但根本不 drift**。因为稳定不漂移的 reward 比 unstable 漂移高。这是 unstable regime 训练的常见 reward shaping 难题。

### 5.3 Domain randomization

每次 rollout sample 32 个 environment，randomize $\mu, C, I_w$、initial position、puddle 位置和长度。Puddle 里 $\mu \to 0.6$。

注意 puddle 位置 $s_{\text{puddle}}$ 是 signed：因为 drift 时车 sideways，前轮可能先入水也可能后轮先入水，这俩情况 dynamics 完全不同。

### 5.4 训练配置

- MPC horizon $T = 35$ (3.5s look-ahead)
- Episode $H = 200$ steps (4s)
- Batch $B = 32$
- 1 SQP iteration per step (real-time iteration scheme)
- 20ms control delay in simulator（match real vehicle）
- 1000 gradient steps, 14 hours on RTX 4090

关键：**batch $\geq 32$ 且 episode $\geq 100$ 是必须的**。Drifting 是高风险 regime，需要大量 sample 覆盖各种 failure mode。这正是 DiffMPC GPU speedup 的价值——让 large batch training tractable。

### 5.5 学到了啥

Fig.10 显示：
- Rear tire friction $\mu_r$: **-13%**（asymmetric，只 rear 降，front 不降！）
- Sideslip cost $Q_{33}$: **-58%**
- 其他参数变化小

物理直觉：学到的 policy 让 MPC "以为" rear tire grip 更差，因此 actuate 更保守（lower torque, lower wheel speed）。同时降低 sideslip tracking 严格度，允许 $\beta$ 偏离 reference 一点，换取对 disturbance 的 margin。

这是典型 **robust control trade-off**：sacrifice nominal performance 换 robustness。Baseline 在 nominal 下更贴 reference、更高 $\beta$，但 puddle 一来就崩。Learned policy 在 nominal 下"佛系"一点，但 puddle 里稳。

### 5.6 结果

仿真 20 个 randomized rollout：
- Baseline: 70% success
- Learned: **100% success**

真车 Toyota Supra：
- Figure-8 with puddle in first turn: baseline spin-out, learned 完成
- Donut with puddle: baseline spin-out, learned robustly drift through（Fig.7, 8）
- **Donut 没在训练中出现！**只训了 figure-8。Sim-to-real + figure-8 → donut 的 transfer 完全靠 MPC structure

为啥能 transfer？MPC 的 reference tracking error $e, \Delta\phi$ 定义与 trajectory shape 无关。学到的 cost weights 和 physical parameters 是 trajectory-agnostic。如果是 pure NN policy (MLP state→action)，figure-8 训练几乎不可能 transfer 到 donut。

---

## 6. 数据里几个有意思的点

### 6.1 Theseus 慢得离谱

Theseus 是 general-purpose NLLS solver。用 Wan et al. 2024 的做法，把 dynamics rollout 塞进 objective function（不显式处理 time-sparsity）。CPU 上 80s+，GPU 上 6s。比所有其他方法都慢一个数量级。

教训：general-purpose differentiable solver 在结构化问题上打不过 structure-exploiting solver。

### 6.2 GPU 上 mpc.pytorch/trajax 几乎没 speedup

Table 3：trajax forward CPU 0.93s vs GPU 0.95s。甚至 slightly 慢。原因就是 Riccati recursion 的 sequential dependency + GPU kernel launch overhead。

mpc.pytorch GPU 上甚至比 CPU 慢（forward 1.91 vs 1.44）。作者为了公平比较还改了原版代码消除一个 serial list comprehension，把 backward 从 136s 降到 4.5s。说明 prior 工具在 GPU scaling 上确实有 bug 级 issue。

### 6.3 DiffMPC 在 CPU 上比 trajax 慢

CPU 上 DiffMPC forward 1.33s, trajax 0.93s。Backward 2.70s vs 1.90s。

Expected trade-off：PCG iteration 比 direct Riccati 多 overhead。CPU 单核串行 Riccati 反而 efficient。Paper 明确说 DiffMPC 是 GPU-tailored，CPU 部署应该用 trajax。

### 6.4 Nonlinear problem 上 speedup 更大

Table 4 nonlinear attitude stabilization ($n_x=3, n_u=3, T=25, B=16$)：
- trajax backward: 505 ms
- DiffMPC backward: 69 ms
- **7.3× speedup**

比 linear problem 的 4× 更大。猜测因为 nonlinear SQP 需要多次 iteration，DiffMPC 的 warm-start 在多次 iteration 间累积优势。

---

## 7. 我觉得这篇 paper 对 ML 社区的意义

### 7.1 Algorithmic innovation > ML novelty

这篇 paper 没提新 NN architecture、没提新 loss function、没提新 RL algorithm。纯 algorithmic innovation（PCG + stair preconditioner 替代 Riccati）+ 系统工程（JAX 实现、warm-start、cache reuse）。

但它的价值很大：让 differentiable optimization 这个 toolbox 在 GPU 上 practical。这种 "infra paper" 对 ML 社区重要——就像 FlashAttention 不提新 attention 架构，但让 long-context training practical。

### 7.2 Structure + Learning 是 ML 的未来

Karpathy 你自己讲过 "Software 2.0 + Software 1.0 hybrid"。DiffMPC 就是这个方向的范例：

- **Structure (Software 1.0)**: dynamics constraint、actuator limit、cost structure、SQP solver logic——全是 hand-crafted 物理知识
- **Learning (Software 2.0)**: cost weights、物理参数、NN-encoded cost/dynamics——data-driven tuning

这种 hybrid 比 pure NN policy sample efficiency 高、safety 好、interpretable、可 transfer。但需要 differentiable solver 把两者 stitch 起来。DiffMPC 就是这样的 stitch layer。

### 7.3 MPC as test-time compute

类比 LLM 的 chain-of-thought——MPC solve 是 policy 的 "thinking time"。DiffMPC 让 "more thinking" tractable，可以想象 adaptive horizon（easy state 短 horizon, hard state 长 horizon）。

这与你在 tweet 讨论的 "inference compute scaling" 是 control 领域的对应。

### 7.4 Real-world validation > toy benchmark

Toyota Supra 真车漂移 + water puddle 不是 toy。这是 limit handling 的 hard case，sim-to-real 一步到位。这种 application paper 比 MuJoCo benchmark 刷 SOTA 有说服力得多。

---

## 8. Future work 我觉得有意思的几个方向

### 8.1 加入 inequality constraints

现在 DiffMPC 只处理 equality constraints。actuator limit 之类 inequality 只能 penalty method（不精确、tune $\mu$ 麻烦）。

Augmented Lagrangian 或 interior point 是 natural extension。Frey et al. 2025 (https://arxiv.org/abs/2505.01353) 在 CPU 上做了，搬到 GPU + PCG 是 obvious next step。

挑战：active set 变化时 gradient 不连续。需要 smoothing 或 path-following。

### 8.2 Contact dynamics

Legged robot locomotion、dexterous manipulation 本质是 contact（complementarity constraints）。Differentiable complementarity 是开放问题（CALIPSO 部分处理但 CPU only）。

如果 DiffMPC 能扩展到 complementarity，机器人社区会疯抢。

### 8.3 NN in cost / dynamics

Paper 没实验 $c^\theta$ 或 $f^\theta$ 是 NN 的情况。但理论上完全支持（$\theta$ 可以是 NN weights）。

可以想象：
- Learned cost: NN 从 demonstration 学"什么是好驾驶"
- Learned residual dynamics: $f^\theta = f^{\text{physics}} + \text{NN}_\theta(x, u)$
- 与 differentiable physics simulation (Brax, MuJoCo MJX) 结合做 model-based RL

### 8.4 World model + MPC (Dreamer-style)

Dreamer 在 latent space plan。如果 world model 是 differentiable 且 structured，可以用 DiffMPC 在 latent space plan，gradient flow back 到 world model training。

### 8.5 Differentiable system identification at scale

把 $\theta$ 设为全部物理参数，用 IL/RL 端到端 identify vehicle dynamics。Paper 已经 partial 做了 (learn $\mu, C, I_w$)，可以 push to full system ID with fleets of vehicles sharing gradient (federated differentiable ID)。

### 8.6 MPC as projection in diffusion

类比 constrained diffusion sampling——每个 denoising step 后用 MPC project 到 dynamics-feasible manifold。Diffusion model 学 trajectory distribution，MPC 当 constraint projection。

### 8.7 Continuous-time limit

OCP 的 time-sparsity 在 $\Delta t \to 0$ 时变成 ODE 的 causal structure。理论上可以 unify DiffMPC 与 neural ODE training (adjoint method 是 IFT 在 continuous time 的极限)。

---

## 9. 最直觉的类比总结

把 DiffMPC 想成 "FlashAttention for differentiable optimal control"：

- FlashAttention 没提新 attention，但让 long-context training practical
- DiffMPC 没提新 control algorithm，但让 differentiable MPC on GPU practical
- 两者都是 algorithmic reorganization to expose parallelism
- 两者都 unlock 了一批之前 impossible 的 application

对 Karpathy 你可能的 angle：这篇 paper 是 "structured inductive bias + differentiable solver + GPU scaling" 三者结合的范例。MPC 不是 NN 的替代，是 NN 的 complement——MPC 提供 physics-informed structure，NN 提供 data-driven tuning。DiffMPC 让这个 combination 在 GPU 上 efficient enough for real RL training。

未来 direction 太多了，期待看到更多 application 级 paper 用这个 tooling。

---

希望这人话版 build up 出来的 intuition 对你有用。要继续聊哪块都行——stair preconditioner 具体怎么 derive、vehicle dynamics 的 linearization 细节、RL training dynamics 的分析、或者跟其他不同iable optimization 工具的对比。

---

# Differentiable Model Predictive Control on the GPU — 深度解读

## 1. 论文核心 idea 的一句话直觉

把 optimal control problem (OCP) 的 sparse-in-time KKT 系统的求解，从 sequential Riccati recursion 换成 **PCG with symmetric stair preconditioner**，从而把 time axis $t = 0, \ldots, T$ 上的依赖解开，让 GPU 的几千个 core 真正并行干活。然后整套 forward + backward 都 JAX 化，可 warm-start，可 end-to-end differentiate，最终在 Toyota Supra 上让 RL 学出来的 MPC policy 在 water puddle 里也能稳稳 drift。

Paper: https://github.com/ToyotaResearchInstitute/diffmpc  
Video: https://youtu.be/r42iJBw-L4E  
Related MPCGPU (forward-only 前作): https://arxiv.org/abs/2403.05232

---

## 2. 为什么这件事重要 (build intuition first)

Differentiable optimization as a layer 是 Amos & Kolter 的 OptNet (https://arxiv.org/abs/1703.00443) 以来 ML 与 control 结合的关键 bridge。把 MPC 嵌入 neural network 的好处是 **physics-informed inductive bias**：

- constraint structure（dynamics、actuator limit）作为 hard prior
- cost structure（tracking、effort）作为 soft prior  
- 学习的只是 parameters $\theta$，sample efficiency 比 black-box NN policy 高得多

但 bottleneck 一直很尴尬：传统 OCP solver (iLQR, Riccati) 本质是沿 time axis 的 sequential scan。GPU 上跑 iLQR，batch 32 个 trajectory 时，每个 trajectory 内部依然串行 over $t$，thousands of cores 大量 idle。这就是为什么 mpc.pytorch (https://arxiv.org/abs/1810.13400) 和 trajax (https://github.com/google/trajax) 在 GPU 上相对 CPU 加速比并不高。

DiffMPC 的关键 insight：OCP 的 Schur complement $S$ 是 block-tridiagonal，这个结构允许我们用 **iterative method (PCG)** 替代 **direct factorization (Riccati)**。iterative method 的 core op 是 sparse mat-vec，可以 fully parallel over $t$；只要 preconditioner 也是 parallel-friendly 的（symmetric stair 满足），整个 solve 就能在 GPU 上 scale。

---

## 3. 数学骨架：从 OCP 到可微 linear system

### 3.1 OCP 形式

$$
\min_{z=(x,u)} \sum_{t=0}^T c_t^{x,\theta}(x_t) + \sum_{t=0}^{T-1} c_t^{u,\theta}(u_t) \quad \text{s.t.} \quad f_t^\theta(x_{t+1}, x_t, u_t) = 0, \quad x_0 = x_s^\theta
$$

变量解释：
- $z = (x, u)$：optimization variables，包含 state trajectory $\{x_0, \ldots, x_T\}$ 和 control trajectory $\{u_0, \ldots, u_{T-1}\}$
- $t \in \{0, \ldots, T\}$：discrete time index，$T$ 是 prediction horizon
- $c_t^{x,\theta}(\cdot)$：state cost at time $t$，由 parameters $\theta$ parameterize
- $c_t^{u,\theta}(\cdot)$：control cost at time $t$
- $f_t^\theta(\cdot)$：equality constraint，通常是 discretized dynamics $x_{t+1} = f(x_t, u_t)$ 写成 implicit form $f_t^\theta(x_{t+1}, x_t, u_t) = 0$
- $x_s^\theta$：initial state，也可以 depend on $\theta$（比如 encoder 输出）
- $\theta$：我们要 differentiate w.r.t. 的参数，可以是 NN weights、物理参数、cost weights

### 3.2 Differentiable optimization 的 IFT 原理

考虑 generic equality-constrained problem $P$：
$$
z = \arg\min_{z \in \mathbb{R}^n} f(z, \theta) \quad \text{s.t.} \quad g(z, \theta) = 0
$$

KKT conditions (这里只有 equality，简化为 stationarity + primal feasibility)：
$$
F(z, \lambda, \theta) := \begin{bmatrix} \nabla_z f(z, \theta) + \lambda^\top \nabla_z g(z, \theta) \\ g(z, \theta) \end{bmatrix} = 0
$$

其中：
- $\lambda \in \mathbb{R}^q$：Lagrange multiplier (dual variable)
- $f$：cost scalar function
- $g: \mathbb{R}^n \to \mathbb{R}^q$：equality constraint vector function
- Lagrangian $L(z, \lambda, \theta) := f(z, \theta) + \lambda^\top g(z, \theta)$

**Implicit Function Theorem**: 对 $F(z(\theta), \lambda(\theta), \theta) = 0$ 全微分：
$$
\frac{dF}{d\theta} = \frac{\partial F}{\partial w} \frac{\partial w}{\partial \theta} + \frac{\partial F}{\partial \theta} = 0 \implies \frac{\partial w}{\partial \theta} = -\left[\frac{\partial F}{\partial w}\right]^{-1} \frac{\partial F}{\partial \theta}
$$

其中 $w = (z, \lambda)$，KKT matrix：
$$
\frac{\partial F}{\partial w} = \begin{bmatrix} \nabla_{zz} L & \nabla_z g^\top \\ \nabla_z g & 0 \end{bmatrix} \in \mathbb{R}^{(n+q) \times (n+q)}
$$

**Reverse-mode (VJP)**: 给定 scalar loss $\ell: \mathbb{R}^n \to \mathbb{R}$，gradient w.r.t. $\theta$ 是
$$
\frac{\partial \ell}{\partial \theta}^\top = -\frac{\partial F}{\partial \theta}^\top \left[\frac{\partial F}{\partial w}\right]^{-1} \begin{bmatrix} \frac{\partial \ell}{\partial z} \\ 0 \end{bmatrix}
$$

只需要解 **一个** $(n+q) \times (n+q)$ linear system（RHS 是 $\frac{\partial \ell}{\partial z}$ padded with zero）。

**Forward-mode (JVP)** 要解 $p$ 个 linear system（$p = \dim(\theta)$），当 $\theta$ 维度高时远慢于 VJP。这与 backprop vs forward-mode AD 是完全一样的 trade-off。

### 3.3 SQP forward pass

对 non-convex OCP，用 Sequential Quadratic Programming。每次 iteration 在当前 guess $(x, u)$ 处 quadraticize cost、linearize dynamics：

$$
\min_{z} \sum_{t=0}^T \frac{1}{2} x_t^\top Q_t x_t + q_t^\top x_t + \sum_{t=0}^{T-1} \frac{1}{2} u_t^\top R_t u_t + r_t^\top u_t \quad \text{s.t.} \quad A_t^+ x_{t+1} + A_t x_t + B_t u_t = C_t, \quad x_0 = x_s
$$

矩阵定义（全部在当前 linearization point 处 evaluate，可 parallel over $t$ 和 batch）：
- $Q_t = \nabla^2 c_t^{x,\theta}(x_t)$, $q_t = \nabla c_t^{x,\theta}(x_t)$: state cost 的 Hessian/gradient
- $R_t = \nabla^2 c_t^{u,\theta}(u_t)$, $r_t = \nabla c_t^{u,\theta}(u_t)$: control cost 同理
- $A_t^+ = \nabla_{x_{t+1}} f_t^\theta$: dynamics 对 next state 的 Jacobian
- $A_t = \nabla_{x_t} f_t^\theta$: dynamics 对 current state 的 Jacobian  
- $B_t = \nabla_{u_t} f_t^\theta$: dynamics 对 control 的 Jacobian
- $C_t = A_t^+ x_{t+1} + A_t x_t + B_t u_t - f_t^\theta$: linearization residual (用 Newton step 处理 $f \neq 0$)

**Key design choice (iLQR-style Gauss-Newton)**: cost Hessian 用 $\nabla_z^2 c$ 而不是完整的 Lagrangian Hessian $\nabla_z^2 L = \nabla_z^2 c + \nabla_z^2 (\lambda^\top f)$，即忽略 dynamics curvature。这让 $(Q, R)$ 容易保持 PD（额外 project 到 PD cone）。代价是 gradient 精度略低，但实现简单、不需要二阶 dynamics derivatives。

### 3.4 KKT system 与 Schur complement 的 block-tridiagonal 结构

QP 的 KKT matrix：
$$
\frac{\partial F}{\partial w} = \begin{bmatrix} G & H^\top \\ H & 0 \end{bmatrix}
$$

- $G = \text{blockdiag}(Q_0, R_0, Q_1, R_1, \ldots, Q_T)$: 完全 block diagonal，每个 block 独立可逆
- $H$: block-bidiagonal，因为每个 dynamics constraint $f_t^\theta$ 只 link $x_t, u_t, x_{t+1}$

**Schur complement** 求解：
$$
\begin{bmatrix} z \\ \lambda \end{bmatrix} = \begin{bmatrix} G & H^\top \\ H & 0 \end{bmatrix}^{-1} \begin{bmatrix} -b \\ d \end{bmatrix}
$$

等价于：
$$
S := -H G^{-1} H^\top, \quad \gamma := d + H G^{-1} b
$$
$$
S \lambda = \gamma, \quad z = -G^{-1}(b + H^\top \lambda)
$$

**结构直觉**: $G^{-1}$ block-diagonal，$H$ block-bidiagonal $\Rightarrow$ $S = -H G^{-1} H^\top$ 是 **block-tridiagonal**。这就是 OCP 的 sparse-in-time 结构最关键的体现：每个 $\lambda_t$ 只与 $\lambda_{t-1}, \lambda_{t+1}$ 直接耦合。

### 3.5 Symmetric stair preconditioner

PCG 解 $S \lambda = \gamma$ 的 convergence rate 取决于 $S$ 的 condition number。DiffMPC 用 Bu & Plancher (https://arxiv.org/abs/2403.05232, ICRA 2024) 的 symmetric stair preconditioner：

$$
\Phi^{-1} = \begin{bmatrix}
Q_0 & -Q_0 \phi_0^\top \chi_0^{-1} \\
-\chi_0^{-1} \phi_0 Q_0 & \chi_0^{-1} & -\chi_0^{-1} \phi_1^\top \chi_1^{-1} \\
& -\chi_1^{-1} \phi_1 \chi_0^{-1} & \chi_1^{-1} & \ddots \\
& & \ddots & \ddots
\end{bmatrix}
$$

其中：
- $\chi_t = A_t Q_t^{-1} A_t^\top + B_t R_t^{-1} B_t^\top + A_t^+ Q_{t+1}^{-1} A_t^{+\top}$: 对角 block（自身耦合）
- $\phi_t = A_t Q_t^{-1} A_{t-1}^{+\top}$: 上/下对角 block（time coupling），约定 $A_{-1}^+ = I$

**为什么这个 preconditioner 是 GPU-friendly 的？** 

直觉：Riccati recursion 像是"从头到尾扫一遍"，是 strict sequential。Symmetric stair 把 time axis 分成 pairs (0,1), (2,3), ...，每个 pair 内部 block-inverse 一次性算完（每个 pair 内部小规模 dense solve），pair 之间保留 tridiagonal coupling 让 preconditioner 仍然 informative。所有 pair 的 inverse 可以 **parallel over $t$** 计算，PCG 主循环里的 mat-vec $S p$ 和 $\Phi^{-1} r$ 也都 parallel over $t$。

这与 multigrid 有精神上的相似：用 coarser representation 来 capture global coupling，用 local solves 来 capture high-frequency errors，但 implementation 上更简单——直接基于 block 结构。

### 3.6 Backward pass 的"免费午餐"

backward pass 要解：
$$
\begin{bmatrix} \tilde z \\ \tilde \lambda \end{bmatrix} = \begin{bmatrix} G & H^\top \\ H & 0 \end{bmatrix}^{-1} \begin{bmatrix} \frac{\partial \ell}{\partial z} \\ 0 \end{bmatrix}
$$

观察：**KKT matrix 与 forward pass 完全相同**，只是 RHS 不同（$(-b, d) \to (\frac{\partial \ell}{\partial z}, 0)$）。所以：
- $G, H, S, \Phi^{-1}$ 全部 reuse forward pass 的计算
- 只需重新跑 PCG iteration，且可以 warm-start（用 forward pass 最后一次 PCG 的 $\lambda$ 作为 initial guess）

最终 state/control recovery (parallel over $t$)：
$$
x_t = -Q_t^{-1}(q_t + A_{t-1}^{+\top} \lambda_t + A_t^\top \lambda_{t+1})
$$
$$
u_t = -R_t^{-1}(r_t + B_t^\top \lambda_{t+1})
$$

每个 $t$ 完全独立，GPU 上 `vmap`/`pmap` 直接 batch。

### 3.7 Sensitivity 最终公式

把 $\tilde z, \tilde \lambda$ 代回 IFT 公式：
$$
\frac{\partial \ell}{\partial \theta}^\top = -\frac{\partial F}{\partial \theta}^\top \begin{bmatrix} \tilde z \\ \tilde \lambda \end{bmatrix}
$$

其中 $\frac{\partial F}{\partial \theta}$ 包含 cost gradient $\nabla_\theta(\nabla_z c^\theta)$, dynamics gradient $\nabla_\theta(\nabla_z f^\theta)$, initial state gradient $\nabla_\theta x_s^\theta$ 等，全部可以 parallel over $t$ 计算（autodiff 自动做）。

---

## 4. Algorithm 整体架构

```
┌──────────────────────────────────────────────────────────┐
│ Forward pass (SQP loop)                                  │
│  while not converged:                                    │
│    1. Evaluate (Q, R, q, r, A, B, C) parallel over t   │
│    2. Form Schur S = -H G^-1 H^T  + γ                   │
│    3. PCG solve S λ = γ   (warm-start, parallel over t) │
│    4. Recover (x^+, u^+) via Eq.11 parallel over t      │
│    5. Line search parallel over candidate α             │
│  Cache: G, H, S, Φ^{-1}, λ                               │
├──────────────────────────────────────────────────────────┤
│ Backward pass (single IFT)                               │
│  1. Form RHS (ℓ_z, 0)                                    │
│  2. PCG solve S λ̃ = γ̃   (reuse S, Φ^-1, warm-start)   │
│  3. Recover z̃ via Eq.11                                   │
│  4. Compute ℓ_θ = -F_θ^T [z̃; λ̃]                         │
└──────────────────────────────────────────────────────────┘
```

几个关键 design choices：

**Line search**: merit function $\varphi(x, u) = \sum_t (c_t^x + c_t^u + \mu \|f_t\|_1)$，penalty parameter $\mu$ 用 Nocedal & Wright Eq.18.33 计算。在 candidate step sizes $\{1.0, 0.7, 0.3, 0.1, 0.01\}$ 上 **parallel evaluate**，取最大 acceptable step。

**Warm-starting**: 三层 warm-start
1. SQP 外层用上一次 SQP 的 solution 作为 initial guess
2. PCG 内层用上一次 PCG 的 $\lambda$ 作为 initial guess
3. MPC replanning 时，下一个 time step 的 OCP warm-start with shifted previous solution

第三点对 real-time MPC 特别重要，但 Riccati-based 方法做不到（必须 cold start 每次 factorize）。DiffMPC 在 $\epsilon = 10^{-4}$ tolerance 下 warm-start 给 9-11% speedup（附录 Fig. 9）。

---

## 5. 性能数据深度解读

Table 3 (Problem 1: $n_x=8, n_u=4, T=40, H=50, B=64$)：

| Method | GPU | Forward (s) | Backward (s) |
|---|---|---|---|
| Theseus | × | 49.35 | 81.47 |
| Theseus | ✓ | 4.39 | 6.19 |
| mpc.pytorch | × | 1.44 | 3.37 |
| mpc.pytorch | ✓ | 1.91 | 4.46 |
| trajax | × | 0.93 | 1.90 |
| trajax | ✓ | 0.95 | 1.83 |
| **DiffMPC** | × | 1.33 | 2.70 |
| **DiffMPC** | ✓ | **0.22** | **0.32** |

几个 insight：

**1. Theseus 慢**: 它是 general-purpose NLLS solver，没利用 OCP 的 time-sparsity，把 dynamics rollout 塞进 objective function (Wan et al. 2024 的做法)。即使 GPU 上也慢，因为它处理的是 dense matrix。

**2. mpc.pytorch / trajax 在 GPU 上几乎没有 speedup**: forward 1.91 vs CPU 1.44，甚至变慢。原因就是 Riccati recursion 的 sequential dependency。GPU 上每个 kernel launch 有 overhead，serial dependency 让 overhead 累积。

**3. DiffMPC 在 CPU 上比 trajax 慢**: 0.22 vs 0.93 在 GPU，但 CPU 上 1.33 vs 0.93。这是 expected trade-off——PCG iteration 比 direct Riccati factorization 多了 overhead，CPU 单核串行 Riccati 反而 efficient。论文也明确说 DiffMPC 是 GPU-tailored，CPU 上应该用 trajax。

**4. Backward pass 接近 forward pass 时间**: DiffMPC backward 0.32 vs forward 0.22。这是 IFT 的优美性质——只需 reuse KKT matrix 解一个新 RHS，加上自动微分的 overhead。其他方法 backward 普遍比 forward 慢 1.5-2x，因为它们的 backward 没法 reuse forward 的 factorization。

Nonlinear attitude stabilization (Table 4, $n_x = 3, n_u = 3, T = 25, B = 16$):
- trajax backward: 505 ms
- DiffMPC backward: 69 ms
- **7.3× speedup**

IL cart-pole benchmark (Fig. 4): DiffMPC 大约 2× training speedup over trajax，loss 收敛行为相当。

---

## 6. Toyota Supra Drifting 应用

### 6.1 任务设定

Drifting 是 limit handling 的极端 case：tire force saturated, vehicle 在 unstable regime 操作，small perturbation 会迅速 amplify 成 spin-out。Water puddle 让 $\mu$ 突然 drop，相当于 step disturbance。Baseline MPC（手动 tuned）在 puddle 里 spin-out。

### 6.2 Vehicle model

Dynamic bicycle model + Fiala brush tire (Svendenius 2007)：

State $x = (r, v, \beta, \omega_r, \Delta\phi, e, s) \in \mathbb{R}^7$:
- $r$: yaw rate (rad/s)
- $v$: longitudinal velocity (m/s)
- $\beta$: sideslip angle (rad) — vehicle longitudinal axis 与 velocity vector 的夹角
- $\omega_r$: rear wheel speed (rad/s)
- $\Delta\phi$: heading tracking error
- $e$: lateral tracking error
- $s$: path distance along reference

Control $u = (\delta, \tau)$:
- $\delta$: steering angle (rad)
- $\tau$: engine torque (Nm)

Dynamics parameters: $(a, b, I_z, m, r_w, I_w, C_f, C_r, \mu_f, \mu_r) \in \mathbb{R}_{>0}^{10}$
- $(a, b)$: CG 到 front/rear axle 距离
- $I_z$: yaw moment of inertia
- $m$: vehicle mass
- $r_w$: wheel radius
- $I_w$: wheel moment of inertia
- $C_f, C_r$: front/rear cornering stiffness
- $\mu_f, \mu_r$: front/rear tire friction coefficients

Simulator 用 Dormand-Prince integrator (Diffrax, https://arxiv.org/abs/2202.02429) 高精度积分；MPC 内部用 trapezoidal rule 简单积分（$\Delta t = 0.1$s）。这个 mismatch 是故意的——simulator 比 MPC model 更准，让 RL 学到 robust to model error 的 policy。

### 6.3 Learnable parameters

$$
\theta = (\text{diag}(Q), \text{diag}(R), \mu_f, \mu_r, C_f, C_r, I_w) \in \mathbb{R}_{>0}^{14}
$$

14 个参数：7 个 cost diagonal + 5 个 physical + 2 个 tire。Log-normalized for training（保持 positivity）。

Cost 用 control rate 而不是 control 本身：
$$
c_t^{u,\theta}(\dot u_t) = \frac{1}{2} \dot u_t^\top R \dot u_t, \quad \dot u_t = (u_{t+1} - u_t)/\Delta t
$$

通过 augment state $x \gets (x, u)$，control input 变成 $\dot u$。这是 racing/drift MPC 的标准 trick，避免 steering/torque jitter。

### 6.4 RL reward 设计

$$
R(x) = -\left((r - r_{\text{ref}})^2 + \lambda e^{-\gamma \beta^2}\right)
$$

- 第一项: yaw rate tracking
- 第二项: sideslip reward，$\lambda = 0.3, \gamma = 50$
  - $e^{-\gamma \beta^2}$ 是 peaked at $\beta = 0$ 的 bell-shaped function，前面加负号 → penalize $\beta$ 接近 0
  - 即鼓励 non-zero sideslip（drifting 的标志）
  - 但 $\beta$ 很大时 penalty 也 saturate，避免 unbounded growth

**作者特别强调**: 如果没有 sideslip reward term，学到的 policy 会"作弊"——稳稳地 drive on reference path 但根本不 drift。这是 reward shaping 在 unstable regime 训练的常见 issue。

### 6.5 Domain randomization

每次 rollout，从 uniform distribution sample 32 个 environment:
- $\mu_f \in [0.94, 1.04]$, $\mu_r \in [0.85, 0.95]$
- $C_f \in [52, 56] \times 10^3$, $C_r \in [200, 240] \times 10^3$
- $I_w \in [8, 14]$
- $s_0 \in [185, 700]$: initial position along path
- $\ell_{\text{puddle}} \in [0, 5]$: puddle length
- $s_{\text{puddle}} \in [-1, 1]$: relative puddle entry (front vs rear wheel)

**Puddle effect**: $\mu \to 0.6$ on puddle。Drifting 时车体 sideways，前轮可能后入 puddle，所以 $s_{\text{puddle}}$ 是 signed。

### 6.6 训练配置

- MPC horizon $T = 35$（3.5s look-ahead）
- Episode length $H = 200$（4s rollout）
- Batch size $B = 32$
- Single SQP iteration per step（real-time iteration scheme，Gros et al. 2020）
- 20ms control delay in simulator（match real vehicle latency）
- 1000 gradient steps, 14 hours on RTX 4090

作者特别强调：**large batch + long episode 是必须的**。Batch < 32 或 episode < 100 steps 训练不稳定。因为 drifting 是高风险 regime，需要大量 samples 覆盖各种 failure mode 才能学到 robust policy。这恰好是 DiffMPC GPU speedup 的价值——让 large batch training tractable。

### 6.7 学到的参数变化（Fig. 10）

- Rear tire friction $\mu_r$: **-13%**（asymmetric！只 rear 降，front 不降）
- Sideslip cost $Q_{33}$: **-58%**（大幅降低 sideslip tracking penalty）
- 其他参数变化较小

**物理直觉**: 学到的 policy 让 MPC "以为" rear tire grip 更差，因此 actuate 更保守（lower torque, lower wheel speed）。同时降低 sideslip tracking 严格度，允许 $\beta$ 偏离 reference 一点，换取对 disturbance 的 margin。这是**保守的 robust controller**：在 nominal 下不极致 push limit，但遇到 puddle 时有 buffer。

Baseline policy 在 nominal 下表现更好（更贴 reference、更高 $\beta$），但 puddle 一来就 spin-out。Learned policy sacrifice nominal performance 换 robustness——典型 robust control trade-off。

### 6.8 实验结果

仿真 (20 randomized rollouts):
- Baseline: 70% success rate
- Learned: **100% success rate**

真车 Toyota Supra:
- Figure-8 with puddle in first turn: baseline spin-out, learned 完成
- Donut with puddle: baseline spin-out, learned robustly drift through (Fig. 7, 8)
- **关键**: donut trajectory 没在训练中出现！只有 figure-8 训练。Sim-to-real + figure-8 → donut 的 transfer 完全靠 MPC 的 inductive bias（reference tracking 的结构）

---

## 7. Intuition 给 Karpathy 的几个关键点

### 7.1 为什么 PCG + stair preconditioner 击败 Riccati？

Riccati recursion 的本质是 **block Thomas algorithm**，是 block-tridiagonal system 的 direct solver。它的优势是 numerical stability 和 exact solution（在 fixed point 处），劣势是 strict sequential dependency over $t$。

PCG 是 iterative method，convergence rate $\propto \sqrt{\kappa(S)}$ where $\kappa$ 是 condition number。 preconditioner $\Phi^{-1}$ 把 effective condition number 降下来。每 iteration 的 mat-vec $S p$ 和 $\Phi^{-1} r$ 都 parallel over $t$。

**Trade-off**: 当 $T$ 小（< 20）、batch 小（< 16）时，PCG iteration overhead 大于 Riccati 的 sequential cost。当 $T$ 大、batch 大时，PCG 的 parallelism 占优。GPU 上几乎总是后者。CPU 上前者。

这与 deep learning 里 "small matmul 用 CPU, large matmul 用 GPU" 是同一个道理。

### 7.2 Forward = Backward 的 reuse 之美

KKT matrix $\frac{\partial F}{\partial w}$ 在 forward 和 backward 完全相同。这意味着 forward pass 的所有 expensive precomputation（building $G, H, S, \Phi^{-1}$）都可以 cache 给 backward 用。backward 只需重新做 PCG iteration（不同 RHS）+ 自动微分的 chain rule overhead。

mpc.pytorch/trajax 的 Riccati forward 和 backward 是不同的 recursion（forward 是 cost-to-go, backward 是 gradient propagation），不能 reuse。这是 DiffMPC 在 backward pass 上 speedup 更大的原因。

### 7.3 为什么忽略 dynamics curvature (iLQR-style) 可接受？

完整的 SQP 应该用 $\nabla_z^2 L = \nabla_z^2 c + \nabla_z^2(\lambda^\top f)$，但论文（和 Amos 2018, Frey 2025 等）都用 Gauss-Newton approximation $\nabla_z^2 c$ only。

直觉解释：
1. **PD 保证**: $\nabla_z^2(\lambda^\top f)$ 不一定 PSD，会让 QP non-convex。Gauss-Newton 总是 PSD（project 后 PD）。
2. **计算成本**: 二阶 dynamics derivatives 在 JAX 里虽然能 autodiff，但 memory 和 compute 翻倍。
3. **Local convergence rate**: Gauss-Newton SQP 在 constraint residual 小时依然 superlinear convergence，对 MPC real-time iteration scheme（1 iteration）足够。
4. **Gradient accuracy**: 论文承认 gradient 精度下降，但 RL/IL 训练对 exact gradient 不敏感（SGD 本身 noisy）。

### 7.4 Sim-to-real transfer 的关键

Baseline MPC 在 nominal conditions 下手动 tuned 到极致，遇到 puddle 就崩。Learned policy 训练时只看 figure-8 + puddle，deploy 到 donut + puddle 直接 work。

为什么能 transfer？
1. **MPC 结构本身是 transferable 的**: reference tracking error $e, \Delta\phi$ 的定义与 trajectory shape 无关。学到的 cost weights 和 physical parameters 是 trajectory-agnostic。
2. **Domain randomization 起作用**: 学到的不是"figure-8 上怎么开"，而是"在各种 $\mu, C$ 下怎么 robustly drift"。这 generalize 到 donut。
3. **物理参数 $\mu_r$ 下调**: 这是 implicit robust control——MPC 内部 model 假设更差 grip，actuate 更保守。这种保守策略在任意 trajectory 上都 robust。

如果是 pure NN policy（e.g., MLP from state to action），figure-8 训练几乎不可能 transfer 到 donut，更别说 sim-to-real。MPC 的 inductive bias 是关键。

### 7.5 Limitations 的诚实评估

1. **Inequality constraints**: 只能 penalty method。Interior point 或 augmented Lagrangian 需要 active set 管理，gradient 在 active set 切换时不连续，differentiation 麻烦。CALIPSO (https://arxiv.org/abs/2210.05306) 和 Frey et al. 2025 (https://arxiv.org/abs/2505.01353) 处理了这个问题但 CPU only。

2. **CPU 上慢**: JAX 在 CPU 上的 dispatch overhead 大，PCG iteration 数比 Riccati direct solve 多。如果 deployment 是 CPU-only embedded system，DiffMPC 不合适。

3. **Hyperparameter tuning**: PCG tolerance、max iter 等 solver hyperparameters 不可微。这其实是个开放问题——how to differentiate through solver convergence choice。

4. **Initialization sensitivity**: SQP 对 initial guess 敏感。Poor init 会导致 divergence。论文 future work 提到 robust initialization pipeline。

---

## 8. 与相关工作的关系图谱

### 8.1 Direct ancestors

- **MPCGPU** (Adabag et al. 2024, https://arxiv.org/abs/2403.05232): DiffMPC 的 forward pass 直接基于 MPCGPU。MPCGPU 是 forward-only（不可微），DiffMPC 加了 IFT backward pass + JAX 重写 + RL 应用。
- **Symmetric stair preconditioner** (Bu & Plancher 2024, ICRA): preconditioner 的理论基础。
- **Differentiable MPC** (Amos et al. 2018, https://arxiv.org/abs/1810.13400): mpc.pytorch，第一个 widely-used differentiable MPC，iLQR-based，PyTorch。
- **OptNet** (Amos & Kolter 2017, https://arxiv.org/abs/1703.00443): differentiable QP as NN layer 的开山之作，IFT 原理。
- **Implicit differentiation** (Blondel et al. 2022, https://arxiv.org/abs/2105.15183): modular IFT 框架，DiffMPC 的 backward pass 思路来源。

### 8.2 Concurrent / alternative approaches

- **Frey et al. 2025** (https://arxiv.org/abs/2505.01353): differentiable NMPC with augmented Lagrangian, full second-order, CPU only, acados-style。
- **Bambade et al. 2024** (https://arxiv.org/abs/2302.08568): augmented Lagrangian differentiable QP, CPU。
- **CALIPSO** (Howell et al. 2022, https://arxiv.org/abs/2210.05306): differentiable solver with conic + complementarity constraints, CPU。
- **CuClarabel** (Chen et al. 2024, https://arxiv.org/abs/2412.19027): GPU conic solver, non-differentiable。
- **ReLU-QP** (Bishop et al. 2024): GPU QP solver for MPC, non-differentiable。
- **cusadi** (Jeon et al. 2024, https://arxiv.org/abs/2401.16574): GPU parallelization framework for symbolic optimal control, non-differentiable。

DiffMPC 在这个 landscape 中的独特位置：**GPU-friendly + differentiable + OCP-structure-exploiting**。其他 method 通常只满足两个。

### 8.3 Application-side related

- **DiffStack** (Karkus et al. 2023, https://arxiv.org/abs/2211.14842): differentiable autonomous driving stack。
- **Djeumou et al. 2024** (https://arxiv.org/abs/2401.10539): physics-informed conditional diffusion model for drifting。同一团队（TRI）的之前工作。
- **Davydov et al. 2025**: active information gathering for limit handling。同一团队。
- **Lew et al. 2025**: risk-averse MPC for racing in adverse conditions。同一团队。

可以看出 TRI 在 limit handling + differentiable control 这个方向有 sustained effort，DiffMPC 是 tooling infra paper，其他是 application paper。

---

## 9. 我对 future work 的几个联想

### 9.1 Differentiable inequality constraints via interior point

Interior point method 的 KKT system 与 equality-only case 结构相似（只是多了 slack variables 和 barrier term）。理论上可以把 PCG + stair preconditioner 推广到 interior point。挑战在于：
- Barrier parameter $\mu$ 减小时 condition number 爆炸
- Central path 跟踪需要 careful step size selection
- Active set 变化时 gradient 不连续（需要 smoothing）

参考: https://arxiv.org/abs/2505.01353 (Frey et al.) 在 CPU 上做了这件事，搬到 GPU 是 natural next step。

### 9.2 Contact-rich manipulation

Contact dynamics 通常 modeled as complementarity constraints ($0 \leq \lambda \perp \phi(x) \geq 0$)。Differentiable complementarity 是开放问题（CALIPSO 部分处理）。如果 DiffMPC 能扩展到 complementarity，可以应用于 legged robot locomotion、dexterous manipulation。

### 9.3 Neural network in cost / dynamics

$ c_t^\theta(x_t)$ 或 $f_t^\theta$ 可以是 NN。DiffMPC 已经支持（$\theta$ 可以是 NN weights），但 paper 没实验这个。可以想象：
- Learned cost: NN encodes "what does good driving look like" from demonstrations
- Learned residual dynamics: $f_t^\theta = f_t^{\text{physics}} + \text{NN}_\theta(x_t, u_t)$
- 这与 differentiable physics simulation (Brax, MuJoCo MJX) 结合可以做 model-based RL 的端到端学习

### 9.4 MPC as test-time compute

类比 LLM 的 chain-of-thought——MPC solve 是 policy 的"thinking time"。DiffMPC 的 GPU efficiency 让 "more thinking" tractable。可以想象 adaptive horizon: easy state 用 short horizon, hard state 用 long horizon, 用 NN learned scheduler 决定。

### 9.5 Diffusion + MPC

两个方向：
- **Diffusion 生成 reference, MPC track**: diffusion model 学 trajectory distribution, MPC 作为 projection 到 feasible set。
- **MPC as projection in diffusion**: 类似 constraint projection in guided diffusion, 每个 denoising step 后用 MPC project 到 dynamics-feasible manifold。

### 9.6 World model + MPC (Dreamer-style)

Dreamer (https://arxiv.org/abs/1912.01603) 在 latent space 做 planning。如果 world model 是 differentiable 且 structured (e.g., latent ODE), 可以用 DiffMPC 在 latent space plan, gradient flow back to world model training。

### 9.7 Multi-agent MPC

多车 racing / platooning / traffic 的 OCP 是 coupled multi-agent。time-sparsity 结构变成更复杂的 graph sparsity (agent-time bipartite graph)。PCG + graph-based preconditioner (e.g., overlapping Schwarz) 是可能的 extension。

### 9.8 Differentiable system identification at scale

把 $\theta$ 设为全部物理参数 ($m, I_z, C_f, C_r, \mu_f, \mu_r, \ldots$), 用 IL/RL 端到端 identify vehicle dynamics。Paper 已经 partial 做了 (learn $\mu, C, I_w$), 但可以 push to full system ID with fleets of vehicles sharing gradient information (federated differentiable ID)。

### 9.9 Risk-sensitive differentiable MPC

把 expected cost 换成 CVaR 或 exponential risk:
$$
\text{CVaR}_\alpha[c] = \inf_t \left\{ t + \frac{1}{1-\alpha} \mathbb{E}[(c - t)^+] \right\}
$$
需要 differentiable risk measure + sampling-based approximation。这对 safety-critical applications (autonomous driving, humanoid robot) 重要。

### 9.10 Connection to neural ODE / continuous-depth

OCP 的时间离散化与 neural ODE 的 continuous-depth 是对偶视角。DiffMPC 的 time-sparsity 结构在 continuous time 极限下变成 ODE 的 causal structure。理论上可以 unify DiffMPC 与 neural ODE training (e.g., adjoint method 是 IFT 在 continuous time 的极限)。

---

## 10. 实现层面的几个细节

### 10.1 JAX 选择

JAX 的 functional style (no side effect, jit-friendly) 对 differentiable optimization 特别合适：
- `jax.vmap` 自动 batch over problem instances
- `jax.jit` 编译整个 SQP loop
- `jax.grad` 通过 IFT custom_vjp rule 实现 backward
- `jax.checkpoint` 控制 memory (rematerialize forward in backward if needed)

对比 PyTorch:
- mpc.pytorch 用 PyTorch, autograd 自动 differentiate through iLQR iterations
- 但 PyTorch 的 dynamic graph + Python GIL 让 batch scaling 不如 JAX XLA

### 10.2 Float64 必要性

论文用 float64。原因：
- KKT matrix 在 near-convergence 时 ill-conditioned
- PCG 对 float32 rounding error 累积敏感
- 硬件 deployment 需要可重复的 numerical behavior

JAX 默认 float32，需要 `jax.config.update("jax_enable_x64", True)`。这会显著降低 GPU throughput (A100 上 float64 throughput 是 float32 的 1/2 到 1/4)。

### 10.3 OSQP for deployment

训练用 DiffMPC (PCG), deployment on vehicle 用 OSQP (https://osqp.org/, Stellato et al. 2020)。OSQP 是 ADMM-based QP solver, CPU-friendly, real-time capable。两个 solver 在 QP level 应该给出接近的 solution (都在 tolerance 内), 但内部 numerical path 不同——这是部署时需要注意的 subtle issue。

### 10.4 SQP iteration count

Training 用 1 SQP iteration (real-time iteration scheme), deployment 也是 1 iteration。这是 deliberate choice:
- 多 iteration 在训练时增加 compute cost 线性
- Deployment 上多 iteration 可能 violate real-time budget
- Real-time iteration 的理论保证 (Gros et al. 2020) 在 1 iteration 也 hold

但 1 iteration 对 initial guess quality 敏感, warm-starting 因此特别重要。

---

## 11. 总结

DiffMPC 是 differentiable optimal control 领域的一篇重要 infra paper。核心贡献是 **algorithmic** (PCG + stair preconditioner 替代 Riccati) 而非 ML novelty。它的价值在于：

1. **打开 GPU scaling**: 让 differentiable MPC 不再是 batch < 16 的 toy, 可以 scale 到 batch 256+, 让 RL training tractable
2. **Reusable infrastructure**: JAX-based, 可嵌入任意 ML pipeline, code 开源
3. **Real-world validation**: Toyota Supra drifting 是 limit handling 的 hard case, 不是 toy benchmark
4. **Forward/backward reuse**: IFT 的优美性质被 fully exploited, backward 几乎 free

对 Karpathy 的 ML 视角: 这篇 paper 是 "structured inductive bias + differentiable solver + GPU scaling" 三者结合的范例。MPC 不是 NN 的替代, 是 NN 的 complement——MPC 提供 physics-informed structure, NN 提供 data-driven tuning。DiffMPC 让这个 combination 在 GPU 上 efficient enough for real RL training。

对未来 research: inequality constraints、contact dynamics、multi-agent、risk-sensitive、continuous-depth 都是 natural extensions。Tool 已备, applications 待开发。

---

## 12. Reference 汇总

**Paper & Code**:
- DiffMPC paper: https://github.com/ToyotaResearchInstitute/diffmpc  
- Video: https://youtu.be/r42iJBw-L4E  
- MPCGPU (forward-only 前作): https://arxiv.org/abs/2403.05232  
- Bu & Plancher (stair preconditioner): https://arxiv.org/abs/2403.05232 (ICRA 2024)

**Differentiable optimization foundations**:
- OptNet (Amos & Kolter 2017): https://arxiv.org/abs/1703.00443  
- Differentiable MPC (Amos et al. 2018): https://arxiv.org/abs/1810.13400  
- mpc.pytorch: https://github.com/locuslab/mpc.pytorch  
- trajax: https://github.com/google/trajax  
- Theseus: https://github.com/facebookresearch/theseus  
- Blondel et al. implicit differentiation: https://arxiv.org/abs/2105.15183  
- Gould et al. 2016: https://arxiv.org/abs/1607.05447

**Other differentiable MPC**:
- Frey et al. 2025 (differentiable NMPC): https://arxiv.org/abs/2505.01353  
- Bambade et al. 2024 (augmented Lagrangian): https://arxiv.org/abs/2302.08568  
- CALIPSO: https://arxiv.org/abs/2210.05306  
- DiffStack: https://arxiv.org/abs/2211.14842

**GPU QP solvers (non-differentiable)**:
- CuClarabel: https://arxiv.org/abs/2412.19027  
- cusadi: https://arxiv.org/abs/2401.16574  
- OSQP: https://osqp.org/

**Vehicle dynamics & drifting**:
- Svendenius tire model: Lund thesis 2007  
- Goh & Gerdes drifting: https://ieeexplore.ieee.org/document/7535475  
- Lew et al. risk-averse racing: https://arxiv.org/abs/2502.08200  
- Djeumou et al. drifting diffusion: https://arxiv.org/abs/2401.10539

**JAX ecosystem**:
- JAX: https://github.com/google/jax  
- Diffrax (ODE solver): https://arxiv.org/abs/2202.02429  
- Dreamer (world model + planning): https://arxiv.org/abs/1912.01603

希望这个 build up 出来的 intuition 对你有帮助。如果想 deep dive 任何 part（比如 stair preconditioner 的具体 derivation、vehicle dynamics 的 linearization、RL training dynamics），可以继续聊。
