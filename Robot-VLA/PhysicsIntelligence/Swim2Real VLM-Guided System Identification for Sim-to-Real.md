---
source_pdf: Swim2Real VLM-Guided System Identification for Sim-to-Real.pdf
paper_sha256: 6588412b145cf233d81e65de9836f3faac5473abbe7acac6c9c54b1a42cf56d0
processed_at: '2026-08-12T11:41:38-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Swim2Real

## 一句话版本

你有一条机器鱼，想在电脑里先模拟它、训练它怎么游，再扔到真水里跑。但模拟器永远不准。这篇 paper 的招很直接：拍真鱼游的视频，拍模拟鱼游的视频，让 Gemini 同时看两段视频，让它说"你这模拟鱼尾巴摆得太僵了，高频下根本摆不动，说明 damping 太大"。然后它给你一版新参数，你试一下，好用了留着，不好用退回去。就这样来回 40 次，模拟器就调准了。训练出来的 RL policy 放到真鱼上游得比传统方法远 12%。

---

## 为什么这事儿难

你做一个 rigid robot（比如四足狗），sim-to-real 已经有成熟套路了：domain randomization 把参数随机采样训一大堆 policy，总有一个能 robust 到 reality。但这条 fish 是 **soft body + water**，两样都是非线性的，耦合在一起就是 chaos。

具体三个 pain point：

**1. Parameter landscape 是 chaotic 的**
tail 是 nitinol rod 做的连续柔体，水里受 fluid force。你改一个 stiffness，整个 tail 的 bending mode 全变。gradient 信息没用，因为你挪一步，landscape 就变了。传统 optimizer 在这种地方容易卡住或者崩。

**2. Simulator 天生就不准**
paper 用的 MuJoCo + stateless ellipsoid fluid model，5 个 fluid coefficients 近似水的阻力。这是简化模型，不是真 CFD。所以不管你怎么调，sim 和 real 之间有个 persistent gap，domain randomization 只能 "train around" 这个 gap，不能 close 它。

**3. 水下实验难复现**
水流、浮力、tendon 摩擦全难控制。你跑一次和跑下一次，marker trajectory 都不一样。这让 scalar error signal 噪声很大。

---

## 以前怎么搞，为什么不行

这同一个平台上一篇 paper [12] 用了 **手动三阶段**：

- FFT analysis 调 stiffness
- Grid search 调 motor geometry  
- BayesOpt 调 fluid coefficients

工程师要自己决定：哪几个参数一组、每组用什么方法、什么顺序跑。这是 human-in-the-loop system id，不 scalable。换个 robot 就得重新设计这套流程。

**为什么 black-box optimizer（BayesOpt, CMA-ES）也不够好**：它们把 simulator 当黑盒，只 consume 一个 scalar（L2 marker error），丢掉了 video comparison 里的 rich 信息。你给 BayesOpt 看 "tail bends too sharply at 1.5Hz"，它听不懂，它只会看一个数字变大还是变小。

---

## Swim2Real 的核心 idea

**VLM 能做 physical reasoning，这是 scalar loss 做不到的**。

VLM 看完 sim 和 real 的视频，能说出这种话：

> "Simulation exhibits critical failure at high frequencies (≥1.25 Hz), where tail motion is severely attenuated, leading to near-complete loss of thrust. This 'locking up' behavior points to excessive damping forces that become dominant at high angular velocities. Conversely, at low frequencies, simulated tail appears too flexible, over-swinging compared to real fish."

然后它 decrease fluid drag + increase hinge stiffness。这一步就把 error 从 168.8 mm 干到 56.5 mm。一个 round。

**scalar L2 loss 永远给不出这种 frequency-resolved diagnosis**。它只能说 "error is 168.8mm"，不能告诉你 why。

---

## 算法本身，其实很朴素

```
初始化 θ_best (random)
for 40 次预算:
    VLM 看视频 + numerical data + history → 提一版 θ'
    Δ = θ' - θ_best
    试三个 step size: β^0=1.0, β^1=0.5, β^2=0.25
    哪个让 L2 error 降了就用哪个
    都没用就 reject，告诉 VLM 下一轮别这么提了
```

就这么多。没有神经网络，没有 RL，没有 fancy architecture。**VLM 给方向，line search 给步长**。

为什么 line search 这么关键？因为 VLM 擅长 diagnose "tail bends too sharply → 应该 reduce stiffness"，但不擅长说 "reduce 多少"。它经常一减减多了，从 168.8 直接到 250。Line search 让它先试 full step，不行再 half，再 quarter，总有一个 step size 刚好。

数据上：78 个 round 里，33 个 accepted。其中 39% 用 full step，33% 用 half，27% 需要 quarter。**61% 的成功 update 如果没有 line search 就会被 reject**。Accept rate 从 14% 涨到 42%，triple 了。

---

## VLM 到底在看什么

每次 round 喂给 Gemini 2.5 Pro 的东西：

**Video**：
- 8 个频率下 sim 和 real 的 skeleton overlay（sim 蓝，real 红，并排）
- Worst-matching frequency 的 real video（让它 focus 最大 discrepancy）

**Numerical**：
- 每个 marker 每个频率的 L2 error
- 每个 marker 的 velocity
- 当前参数 vector
- 参数 bounds
- History：之前提过什么、被 reject 没

**Semantic labels**：
- "hingeStiffness 3 controls restoring torque at joint 3"

这点很关键。VLM 不是瞎猜，它有 **parameter semantics**，能把 visual observation（"joint 3 弯太多"）映射到 specific parameter（hingeStiffness_3）。没有这个 grounding，VLM 提的参数就是 random。

---

## 16 维参数空间，长啥样

| Group | Count | 干啥的 |
|---|---|---|
| Fluid coefficients | 5 | blunt drag, slender drag, angular drag, Kutta lift, Magnus lift |
| Motor arm length | 1 | crank-slider 的 lever arm |
| Hinge stiffness | 5 | 5 个 joint 各自的 restoring torque |
| Hinge damping | 5 | 5 个 joint 各自的 velocity-proportional damping |

这 16 个全部同时调，没有 hand-designed 分组。之前的 paper [12] 只调 9 个还分三阶段。这是真 scaling。

---

## 结果，重点不是 mean，是 reliability

| Method | Error (mm) | Worst seed | Velocity MAE | RL swim distance |
|---|---|---|---|---|
| Random | 82.0 ± 34.3 | 141.8 | 21.3 mm/s | 6.0 m |
| CMA-ES | 112.7 ± 83.9 | 254.2 | 14.5 mm/s | 4.0 m |
| BayesOpt | 52.4 ± 2.1 | 55.6 | 12.9 mm/s | 6.8 m |
| **Swim2Real** | **51.3 ± 1.2** | **53.2** | **7.4 mm/s** | **7.6 m** |

你看 mean 只差 1.1mm，在 seed variance 内。但真正 telling 的是：

- **Variance**: ±1.2 vs ±2.1，Swim2Real tight 一倍
- **Worst seed**: 53.2 vs 55.6，更稳
- **Velocity MAE**: 7.4 vs 12.9，差 43%。Velocity 不在 calibration objective 里，是独立 test
- **RL transfer**: 7.6 vs 6.8 m，远 12%
- **CMA-ES 崩了**: 16D 空间 + budget 40 → 只能跑 3 代，covariance 还没收敛就 budget 耗光，2/5 seeds 直接炸

**CMA-ES catastrophic failure 的本质**：population size = $4 + \lfloor 3 \ln 16 \rfloor = 12$，budget 40，只能跑 3 generations。CMA-ES 要足够代数来 estimate covariance evolution path。chaotic landscape + high dim + tiny budget = covariance estimate 还没收敛就结束了。

---

## 50mm error floor 从哪来

Per-marker error 有 head-to-tail gradient：
- Head markers (M0-M2): 23 mm
- Tail markers (M6-M8): 80 mm  
- Tail tip M8: 101 mm

**这是 simulator fidelity limit，不是 calibration algorithm limit**。tail 是 continuum soft body，你用 5 个 rigid hinge 近似它，approximation error 沿 kinematic chain 累积。任何 marker-based calibration 都撞这堵墙。要降得换更高 fidelity 的 fluid model 或者 per-segment fluid coefficients。

---

## VLM 的两种失败模式

58% 的 round 被 reject。两种主要失败：

**1. Fluid coefficients vs joint stiffness 分不清**
两者都影响 tail amplitude，但机制不同：
- Fluid drag: external hydrodynamic force
- Joint stiffness: internal restoring torque

VLM 经常同时调两个，互相 cancel。这是 **parameter identifiability** 问题——video observation 不足以 disambiguate 两种机制。

**2. Motor arm length 没 visual signature**
这个参数控制 crank-slider 内部的 lever arm，但在 skeleton overlay 里看不到。VLM 只能从整体 thrust mismatch 间接猜，本质是 informed guess。

---

## Real deployment 的 caveat

Target-reaching policy 训练完，motor commands 50Hz open-loop 跑到真鱼上。Simulation 里能精准 reach target，真鱼上有个 consistent **leftward arc**。

两个 hypothesis：
- Open-loop 无 state feedback，policy 无法 correct drift
- Hardware steering bias：tendon 和 routing hole 之间摩擦不对称

**关键 insight**：calibration 在 local body frame 里做（去掉 global position 和 heading），所以 steering bias **对任何 marker-based calibration 都是 invisible 的**。这是 local-frame calibration 的 fundamental limitation。要解得 closed-loop + onboard state estimation。

---

## 我觉得这 paper 真正的 contribution

不是 "VLM 调参比 BayesOpt 好 1.1mm"——mean 差异在 noise 内。真正 contribution 是三件事：

**1. VLM = visual gradient**

传统优化：$\nabla \mathcal{L}$ 给 direction，line search 给 step size。
Swim2Real：VLM 给 direction（通过 physical reasoning），line search 给 step size。

这个 analogy 很 powerful。Differentiable simulation 需要可微 backend，stateless ellipsoid model 用不了。VLM 绕过了这个限制——它不需要 $\nabla \mathcal{L}$ 存在，只需要能 visualize 和 reason about discrepancy。本质上是把 VLM 当 **non-differentiable simulator 的 gradient oracle**。

**2. Reliability > peak performance**

Engineering deployment 不需要偶尔跑出 50mm 的神级参数，需要每次都稳定在 51±1mm。Swim2Real 5 seeds 全在 3mm 范围内，zero outliers。这是 VLM + line search 的真正价值——不是更聪明，是更稳。

**3. Out-of-objective generalization**

Calibration objective 是 marker L2 error，但 Swim2Real 在 velocity（不在 objective 里）也赢了 43%。说明 VLM 调出来的参数 **physically meaningful**，不是 overfit marker trajectory。BayesOpt 能 minimize L2 到 52mm，但它的参数组合可能是物理上无意义的（stiffness 过低 compensating damping 过高），导致 thrust dynamics 不对。VLM 的 physical reasoning 避免了这种 spurious minimum。

---

## 我的几个联想

**跟 Eureka/DrEureka 的本质区别**：Eureka 是 LLM 写 reward function，DrEureka 是 LLM 写 domain randomization distribution，都是 "train around the gap"。Swim2Real 是 "close the gap"。这是 system identification vs domain randomization 的经典 dichotomy，但用 VLM 代替手工 design。

**跟 differentiable simulation 的关系**：gradSim [19] 那条线需要可微 backend，soft fish 用 differentiable sim [20, 21] 也做过。但 stateless ellipsoid model 用不了。VLM 是 "gradient-free differentiable simulator"——它不通过 chain rule 给 gradient，通过 visual comparison 给 direction。

**Scaling 的想象空间**：这篇是 16D。如果到 100D（比如 full humanoid soft robot），GP surrogate scale 不上去（BayesOpt 的 kernel matrix 爆掉），CMA-ES population size 要爆炸。VLM 的优势是 cost 跟 dimension 弱相关——它看视频推理，不管你参数多少维。这可能是 high-dim system id 的出路。

**VLM 作为 physics reasoner 的范式**：现在 VLM 大部分用在 task planning、code generation 这种 high-level。这篇把它拉到 low-level parameter estimation。物理推理这件事，GPT-4V 和 Gemini 已经够用了，不需要 AGI。这是个 useful 的 niche。

---

## 总结

一句话：**把 VLM 当 visual gradient，配 line search 调 simulator 参数，比 black-box optimizer 更稳、out-of-objective generalization 更好、能 zero-shot deploy 到真硬件上**。算法朴素，但 idea 可能是个新范式——用 foundation model 的物理直觉代替手工 cost function design。

参考链接：
- Paper predecessor Vid2Sid: https://arxiv.org/abs/2602.19359  
- Platform paper: https://arxiv.org/abs/2602.23283
- Eureka (LLM reward design): https://arxiv.org/abs/2310.12931
- DrEureka (LLM domain randomization): https://arxiv.org/abs/2407.13235
- gradSim (differentiable sim): https://arxiv.org/abs/2104.02646
- BayesSim (Bayesian sys id): https://arxiv.org/abs/1906.04233
- Soft fish differentiable sim: https://arxiv.org/abs/2208.08357
- Learned residual physics soft robots: https://arxiv.org/abs/2405.04328
- MuJoCo: https://mujoco.org/
- SAC (RL algorithm): https://arxiv.org/abs/1801.01290

---

# Swim2Real: 用VLM做Sim-to-Real System Identification

## 一、Core Insight —— 为什么这件事值得做

Soft aquatic robot的sim-to-real特别难，原因有threefold：

1. **Nonlinear fluid-structure coupling**：tendon-driven fish的tail是continuum soft body + hydrodynamics，parameter landscape是chaotic的，gradient信息不可靠。
2. **Simplified fluid model**：paper用的是stateless ellipsoid model（5个fluid coefficients）而非CFD，所以sim inherent有个persistent sim-to-real gap，domain randomization只能"train around"而非"close"。
3. **Controlled aquatic experiments难复现**：水扰动、tendon摩擦、bouyancy都难控制。

Prior work [12] on this exact platform需要manual 3-stage decomposition：FFT analysis for stiffness → grid search for motor geometry → BayesOpt for fluid coefficients。Practitioner要决定which params to group, which method per group, which order。这是"human-in-the-loop system id"，不scalable。

**Swim2Real的thesis**：black-box optimizer（BayesOpt, CMA-ES）只consume scalar error signal，丢掉了video comparison里rich的visual + physical structure。VLM（Gemini 2.5 Pro）能做physical reasoning——它能说"tail bends too sharply at high frequency, suggesting excessive damping"——这是scalar L2 loss无法提供的interpretable diagnostic。VLM本质上扮演了**"visual gradient"**的角色：给direction，line search给magnitude。这直接类比gradient-based optimization里∇L给direction，backtracking line search找step size。

Paper这里的核心claim其实微妙：最终L2 marker error Swim2Real (51.3 mm) vs BayesOpt (52.4 mm)差异在variance内。真正的win在于：
- **Reliability**：5 seeds全在3mm范围内，zero outliers；BayesOpt variance是其2倍
- **Out-of-objective generalization**：forward velocity MAE 7.4 vs 12.9 mm/s（velocity不在calibration objective里）
- **Downstream RL transfer**：swim 12% farther than BayesOpt policies

这说明VLM调出来的params是physically meaningful的，不只是overfit marker trajectory。

参考链接：
- Vid2Sid (predecessor, same authors): https://arxiv.org/abs/2602.19359
- Prior platform paper (Michelis et al.): https://arxiv.org/abs/2602.23283
- Eureka (LLM for reward design): https://arxiv.org/abs/2310.12931
- DrEureka (LLM for DR): https://arxiv.org/abs/2407.13235

---

## 二、Problem Formulation 细节

### Objective (Equation 1)

$$
\theta^* = \arg\min_{\theta \in \Theta} \frac{1}{|\mathcal{F}|} \sum_{f \in \mathcal{F}} \frac{1}{M T_f} \sum_{m=1}^{M} \sum_{t=1}^{T_f} \|\mathbf{p}_m^{\mathrm{s}}(t, f, \theta) - \mathbf{p}_m^{\mathrm{r}}(t, f)\|_2
$$

变量含义：
- $\theta \in \mathbb{R}^d$：16维parameter vector（d=16）
- $\Theta \subset \mathbb{R}^d$：bounded parameter space（bounds见表I）
- $\mathcal{F}$：actuation frequencies集合 = {0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25} Hz, $|\mathcal{F}|=8$
- $M=9$：body markers数（M0 head到M8 tail tip，共11个tracked但用9个算error）
- $T_f$：frequency $f$ 下的timesteps
- $\mathbf{p}_m^{\mathrm{s}}, \mathbf{p}_m^{\mathrm{r}} \in \mathbb{R}^2$：sim和real的marker 2D位置（overhead camera）
- $\|\cdot\|_2$：$\mathbb{R}^2$上的Euclidean norm

这个objective是**per-marker, per-timestep, per-frequency的L2 distance的average**。关键设计：
- Trajectories先rotate到fish local frame，去掉global position和heading（否则不同位置/朝向的swim没法比）
- Mean over frequencies让optimizer不偏向某个特定gait
- 一次evaluate要跑$|\mathcal{F}|=8$次MuJoCo simulation，这是dominant cost

### Parameter Space (Table I)

| Group | Count | Bounds | Unit |
|---|---|---|---|
| Fluid coefficients | 5 | [0, 10] | dimensionless |
| Motor arm length | 1 | [0.01, 0.06] | m |
| Hinge stiffness (per joint) | 5 | [0.1, 5.0] | N·m/rad |
| Hinge damping (per joint) | 5 | [0.0, 2.0] | N·m·s/rad |

**Fluid coefficients**对应stateless ellipsoid model的5个力：
- Blunt drag（head-on resistance）
- Slender drag（沿body的friction）
- Angular drag（rotation resistance）
- Kutta lift（尾涡lifting）
- Magnus lift（旋转lift）

**Motor arm length**控制crank-slider tendon mechanism的lever arm。这里有个关键的subtlety：这个parameter在skeleton overlay video里**没有直接visual signature**——VLM只能通过整体thrust mismatch来guess，这是后面提到的failure mode之一。

**Per-joint stiffness/damping**：tail被discretize成5个rigid segments，用hinge joints连接。每个hinge有stiffness $k$（restoring torque, $\tau = -k\theta$）和damping $b$（$\tau = -b\dot{\theta}$）。

---

## 三、Algorithm Architecture

### Pipeline整体（Fig. 1 + Fig. 2）

```
Stage 1: VLM-Guided Calibration (40 sim evals budget)
─────────────────────────────────────────────────
  RandomInit θ_best ∈ Θ
  L_best = Evaluate(θ_best)
  H = ∅  (history)
  while B > 0:
    V = RenderOverlay(θ_best)  # sim vs real skeleton video
    θ' = QueryVLM(V, θ_best, H)  # VLM proposes direction
    Δ = θ' - θ_best
    accepted = false
    for k = 0,1,2:  # backtracking line search
      θ_k = Clip(θ_best + β^k · Δ, Θ)
      L = Evaluate(θ_k); B -= 1
      if L < L_best:
        θ_best, L_best = θ_k, L
        accepted = true; break
    Append (Δ, β^k, accepted) to H
  return θ_best

Stage 2: RL Training + Deployment
─────────────────────────────────────────────────
  MuJoCo simulator instantiated with θ_best
  SAC training (5M steps, 3 seeds)
  Motor commands deployed open-loop at 50 Hz on hardware
```

### VLM Input Design

每次iteration VLM（Gemini 2.5 Pro）拿到两路information：

**Video input**：
- Side-by-side skeleton overlay at **each** of 8 frequencies（sim blue, real red markers）
- Real video at **worst-matching** frequency（让VLM focus on最大discrepancy）

**Numerical data**：
- Per-marker, per-frequency L2 errors
- Per-marker velocities
- Current parameter vector
- Parameter bounds
- History of previous proposals + outcomes（accepted/rejected, β^k）
- Parameter semantics（e.g., "hingeStiffness 3 controls restoring torque at joint 3"）

这个design很关键：VLM不是只看video猜参数，它有**structured numerical context + semantic labels + history**。Semantic labels让VLM能把visual observation（"tail bends too sharply at joint 3"）映射到specific parameter（hingeStiffness_3）。History让VLM能avoid重复mistakes和做course correction。

### Backtracking Line Search (Equation 2)

$$
\theta_k = \theta_{\mathrm{best}} + \beta^k \cdot \Delta, \quad k = 0, 1, \ldots, K-1
$$

变量：
- $\theta_{\mathrm{best}}$：当前best params
- $\theta'$：VLM proposal
- $\Delta = \theta' - \theta_{\mathrm{best}}$：VLM-proposed update direction
- $\beta \in (0,1)$：decay factor = 0.5
- $K=3$：max steps to try
- $\theta_k$：第k个candidate

Accept first $k$ where $\mathcal{L}(\theta_k) < \mathcal{L}(\theta_{\mathrm{best}})$。如果都不improve，round rejected，反馈给VLM下一轮。

**为什么这个work**：VLM擅长diagnose direction（"tail bends too sharply → reduce stiffness"）但不擅长quantify magnitude（"reduce by how much?"）。Line search geometrically halve step size，cover VLM的magnitude overestimation。

经典numerical optimization的backtracking line search通常用Armijo condition：
$$
L(\theta + \alpha p) \leq L(\theta) + c_1 \alpha \nabla L(\theta)^T p
$$
这里simplification：不需要sufficient decrease condition，只要strict improvement（$L(\theta_k) < L_{\mathrm{best}}$）。原因：VLM给的"direction"不是真正的gradient，没法计算expected decrease slope。

### Ablation数据解读（Section V-C）

78个VLM rounds（5 seeds × ~15.6 rounds/run）里line search接受了33个。其中：
- **39.4%** 用full step（$\beta^0 = 1.0$）
- **33.3%** 用half step（$\beta^1 = 0.5$）
- **27.3%** 需要quarter step（$\beta^2 = 0.25$）

→ **61%的成功updates会被no-line-search版本reject**。这直接量化了line search的价值。

每个VLM call平均cost 2.5 sim evaluations（line search overhead），但accept rate从14% → 42%（3×），所以**per accepted update的成本从7.2降到5.9 evaluations**。这是个rare的win-win：既提高reliability又降低unit cost。

---

## 四、实验结果深度解析

### Table II 核心数据

| Method | Best (mm) ↓ | Worst (mm) ↓ | AUC (mm) ↓ |
|---|---|---|---|
| Random | 82.0 ± 34.3 | 141.8 | 129.3 ± 26.6 |
| CMA-ES | 112.7 ± 83.9 | 254.2 | 156.4 ± 91.7 |
| BayesOpt | 52.4 ± 2.1 | 55.6 | 94.2 ± 17.6 |
| **Swim2Real** | **51.3 ± 1.2** | **53.2** | **85.9 ± 30.9** |
| w/o line search | 52.4 ± 2.8 | 57.3 | — |
| w/ Gemini 3.1 | 54.6 ± 4.2 | 62.0 | — |
| Warm start | 91.1 ± 49.5 | 177.4 | — |

**关键intuitions**：

1. **CMA-ES catastrophic failure的root cause**：16D空间，population size $4 + \lfloor 3\ln(16)\rfloor \approx 12$，budget 40 → 只能跑~3 generations。CMA-ES需要足够generations来estimate covariance matrix evolution path。在chaotic landscape + high-dim + tiny budget下，covariance estimate还没收敛就budget耗尽。2/5 seeds collapse到123和254 mm。

2. **BayesOpt vs Swim2Real的真正差异不是mean**：mean差1.1mm在seed variance内。差异在：
   - **Worst case**: 55.6 vs 53.2 mm（Swim2Real更稳）
   - **Variance**: ±2.1 vs ±1.2 mm（Swim2Real tight 2×）
   - **AUC**: 94.2 vs 85.9 mm（Swim2Real converges更快更稳）
   - **Out-of-objective velocity MAE**: 12.9 vs 7.4 mm/s（43% lower，half！）

3. **AUC variance的paradox**：Swim2Real AUC variance (±30.9) 比 BayesOpt (±17.6) 大。这是因为不同seed的VLM convergence speed不同，但final accuracy高度consistent。BayesOpt的GP surrogate在mid-optimization收敛快但plateau也早；Swim2Real line search允许steady refinement across所有phases。

4. **Line search ablation**: w/o line search 52.4±2.8 mm vs with 51.3±1.2 mm。Mean只差1.1mm但**worst case从53.2跳到57.3，variance翻倍**。Line search主要是reliability improver而非accuracy improver。

5. **Gemini 3.1 vs 2.5 Pro**: 54.6 vs 52.4 mm（no-line-search config）。差异2.2mm在variance内，说明method不critically depend on specific VLM version。这是portability的重要证据。

6. **Warm start ablation**: 91.1±49.5 mm，worst 177.4。一个VLM call + 剩下random search。证明iterative VLM feedback是essential，不是单次VLM "magic"。

### Velocity Match (Fig. 6) —— Out-of-Distribution Test

Forward velocity = head marker position沿swim axis的linear fit slope。这个metric**不在calibration objective里**，是独立的simulator quality test。

| Method | Velocity MAE (mm/s) |
|---|---|
| Random | 21.3 |
| CMA-ES | 14.5 |
| BayesOpt | 12.9 |
| **Swim2Real** | **7.4** |

Swim2Real比BayesOpt好**43%**。Gap在high frequencies（>1.5 Hz）尤其大，这是thrust-producing dynamics最critical的区域。这解释了为什么downstream RL性能更好：swimming靠thrust，不靠marker shape matching。

### VLM Verbatim Diagnosis (Section V-C)

第0个seed第一个accepted round的VLM输出：

> "The simulation exhibits a critical failure at high frequencies (≥1.25 Hz), where the tail motion is severely attenuated, leading to a near-complete loss of thrust [...]. This 'locking up' behavior points to excessive damping forces that become dominant at high angular velocities. Conversely, at low frequencies, the simulated tail appears too flexible, over-swinging compared to the real fish."

VLM基于此**decreased fluid drag + increased hinge stiffness**。Proposal在$\beta^0=1.0$ accepted，error从168.8 → 56.5 mm。

这就是VLM作为"physics reasoner"的价值：它不只是minimize scalar，它reasons about **frequency-dependent regime**——high freq damping dominant vs low freq flexibility。Scalar L2 loss看不到这种frequency-resolved mechanism，只能blindly tune。

### Downstream RL (Section V-E)

| Method | Forward swim distance (3 seeds) |
|---|---|
| CMA-ES | 4.0 ± 0.6 m |
| Random | 6.0 ± 0.0 m |
| BayesOpt | 6.8 ± 0.1 m |
| **Swim2Real** | **7.6 ± 0.0 m** |

Swim2Real比BayesOpt远**12%**，比CMA-ES远**90%**。Ranking和calibration accuracy单调一致，说明L2 calibration metric确实capture了real simulator fidelity差异。

### RL Setup细节

**Forward swimming**:
- Observation (52-dim): 11 markers pos + vel (44) + head pose (4) + motor state (4)
- Action: $a \in \mathbb{R}$（continuous motor acceleration）
- Reward: $r = -x_{\mathrm{head}}$（x轴backward，negate reward forward progress）
- Episode: 2s

**Target reaching**:
- Observation (22-dim): motor state + tail joint angles/vel + goal vector in body frame
- 3-stage curriculum: target region 1×1m → 3×4m, success threshold 10cm → 5cm
- Reward (Equation 3):
$$
r = -d(\mathbf{p}_{\mathrm{fish}}, \mathbf{p}_{\mathrm{goal}}) - 0.5\|a\| + 300 \cdot \mathbf{1}[d < 0.05\,\mathrm{m}]
$$
  - $d(\mathbf{p}_{\mathrm{fish}}, \mathbf{p}_{\mathrm{goal}})$: fish到goal的Euclidean距离
  - $\|a\|$: action magnitude（鼓励smooth control）
  - $\mathbf{1}[d < 0.05]$: success indicator，5cm内奖励300

**SAC hyperparams**: lr $2 \times 10^{-3}$, batch 256, $\gamma = 0.99$, $\tau = 0.1$, 50K exploration steps, [256, 256] MLP, 5M training steps, 3 seeds per condition.

### Hardware (Fig. 3)

- 0.6 m length, 1.5 kg total
- Raspberry Pi Zero 2 W onboard
- Dynamixel XW-540-T140-R motor (waterproof)
- U2D2 controller
- 3S LiPo + DC-DC converter
- Super-elastic nitinol rods（compliant tail核心）
- 5个3D-printed spine segments
- **Antagonistic tendons crossing at tail midpoint** → bio-inspired S-bend（这是设计精髓：单motor通过crank-slider + 交叉tendon产生S-shape，模仿真实鱼carangiform swimming）
- Dual-output crank-slider mechanism

**Data collection**:
- 11 markers, CSRT tracker
- 2160×3840 overhead camera, 60 fps
- 2m × 3m pool

---

## 五、Limitations & 50mm Error Floor

### Error来源分析（Section VI-A）

Per-marker error有head-to-tail gradient：
- Head markers (M0–M2): **23 mm** average
- Tail markers (M6–M8): **80 mm** average
- Tail tip (M8): **101 mm** alone

**Root cause**: 5个rigid hinge joints近似continuum bending tail，approximation error沿kinematic chain累积。这是**simulator fidelity limit**，不是calibration algorithm limit。任何marker-based calibration都撞这个floor。要降低需要：
- Per-segment fluid coefficients（5 joints共享5个fluid coefficients，实际可能不同）
- Higher-fidelity fluid model（CFD或learned residual physics [21]）

### VLM Failure Modes（Section VI-B）

58% rejection rate（即使有line search）的两大失败模式：

**Failure 1: Fluid coefficients vs joint stiffness conflation**
两者都影响tail amplitude但机制不同：
- Fluid drag: external hydrodynamic force
- Joint stiffness: internal restoring torque

VLM经常同时调两者，correction互相cancel。这是**parameter identifiability**问题——video observation不足以disambiguate两种机制。

**Failure 2: Motor arm length无visual signature**
Motor arm length是crank-slider mechanism的lever arm，它控制tendon位移的mechanical advantage。但在skeleton overlay里看不到motor内部几何，VLM只能通过整体thrust mismatch间接infer，本质是**informed guess**。

### Real-World Deployment Issues（Section V-E）

Open-loop 50 Hz部署，real trajectory有**consistent leftward arc**，simulation没有。两个假设原因：
1. **Open-loop无state feedback** → policy无法correct drift
2. **Hardware steering bias**：tendon和routing hole之间摩擦asymmetry

Calibration在local body frame做（去掉global position和heading），所以steering bias**invisible to any marker-based calibration**。这揭示了marker-based sim-to-real的fundamental limitation：global dynamics（drift, bias）无法被local-frame calibration capture。

---

## 六、Broader Connections & Intuition

### 1. VLM as "Visual Gradient"

传统optimization：$\nabla \mathcal{L}(\theta)$给direction（analytical或finite-difference），line search给step size。
Swim2Real：VLM给direction（visual-physical reasoning），line search给step size。

这个类比很powerful。**VLM是non-differentiable simulator的"gradient oracle"**。Differentiable simulation [19]需要differentiable backend，stateless ellipsoid model用不了。VLM绕过了这个限制——它不要求$\nabla \mathcal{L}$存在，只要求能visualize和reason about discrepancy。

### 2. 与Eureka/DrEureka的对比

Eureka [27]: LLM生成reward function代码
DrEureka [28]: LLM生成domain randomization分布

两者都是"**train around the gap**"——不close sim-to-real gap本身，而是让policy robust to it。

Swim2Real是"**close the gap**"——直接调simulator参数让它更准。这是system identification vs domain randomization的经典dichotomy，但用VLM代替手工design。

### 3. 与LLM-as-Optimizer的对比

OPRO [29]: LLM直接在text prompt里做black-box optimization，低维（≤20D），text feedback。
Swim2Real: VLM在16D连续空间，**video + numerical** feedback，physical reasoning。

关键差异：Swim2Real的VLM不只返回"better params"，返回**structured physical diagnosis**——这是scalar optimizer给不出的interpretable signal。

### 4. Vid2Sid [9] 的scaling

Vid2Sid (same authors, 2026) 证明了VLM feedback能replace hand-crafted metrics，但只测了≤7 params，**无downstream policy validation**。

Swim2Real contribution:
- Scaling: 7 → 16 params
- Line search: 新增，triples accept rate
- Full pipeline: calibration → RL → real deployment

### 5. Computational Cost Analysis

| Method | Wall-clock (5-run mean) | Breakdown |
|---|---|---|
| Swim2Real | 19 min | 8 min sim + 11 min VLM |
| BayesOpt/CMA-ES/Random | 6 min | sim only |

Per VLM call: ~42s (rendering + upload + inference + parsing)
Per sim eval: ~13s on AMD Ryzen 7 PRO 7840U single core
Mean VLM calls per run: 15.6（42% accepted）
Mean sim evals per run: ~40（budget cap）

VLM overhead triples wall-clock，但相比real-world data collection（需要pool access, marker tracking, multi-frequency sweeps）是minor。这是正确的工程judgment：**计算便宜，物理实验贵**。

---

## 七、关键Take-aways for Building Intuition

1. **VLM不是替代optimizer，是替代gradient**。Line search依然需要。VLM = visual gradient + physical reasoning，line search = magnitude corrector。

2. **Reliability > peak performance**。Swim2Real的win不在mean L2 error，在zero outliers + out-of-objective generalization。Engineering deployment需要reproducibility。

3. **Iterative feedback essential**。Warm start ablation证明单次VLM call不够。VLM需要看到自己的mistake来course correct，特别是面对parameter identifiability ambiguities。

4. **Out-of-objective metrics reveal true fidelity**。Marker L2是calibration objective，velocity是independent test。两者ranking一致说明L2 metric faithful，但velocity gap（43% improvement）显示VLM调的params物理上更对。

5. **Simulator fidelity is the bottleneck**。50mm error floor来自5-segment discretization。Calibration algorithm再好也撞这个墙。Future work需要更高fidelity model，可能differentiable simulation [20, 21]或learned residual physics。

6. **Local-frame calibration有fundamental limitation**。Global drift/steering bias invisible to marker-based methods。需要closed-loop + onboard state estimation。

7. **Parameter semantics + history是VLM成功的关键design**。不是裸video → params，是video + numerical context + semantic labels + history → params。这让VLM的physical reasoning能ground到specific parameters。

参考链接汇总：
- Paper: https://arxiv.org/abs/2602.19359 (Vid2Sid predecessor)
- Platform: https://arxiv.org/abs/2602.23283 (Michelis et al. fish sim)
- Eureka: https://arxiv.org/abs/2310.12931
- DrEureka: https://arxiv.org/abs/2407.13235
- gradSim: https://arxiv.org/abs/2104.02646
- BayesSim: https://arxiv.org/abs/1906.04233
- SAC: https://arxiv.org/abs/1801.01290
- MuJoCo: https://mujoco.org/
- Differentiable sim for soft fish: https://arxiv.org/abs/2208.08357
- Learned residual physics for soft robots: https://arxiv.org/abs/2405.04328

这篇paper的真正贡献是把VLM从"high-level planner"（task planning, code generation）拉到"low-level parameter estimator"的位置，用physical reasoning代替scalar black-box optimization。Line search是个unpretentious但effective的工程加成，把VLM的direction-only weakness补全。下一步显然是closed-loop deployment + 更高fidelity fluid model + scaling到更多DOF的soft robot。
