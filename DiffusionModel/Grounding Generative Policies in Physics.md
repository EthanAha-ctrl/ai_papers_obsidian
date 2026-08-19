---
source_pdf: Grounding Generative Policies in Physics.pdf
paper_sha256: 3fcbdec5e7900cefbf7db01b59c51fa41067735876b1e29ca29879c8d335ba27
processed_at: '2026-08-19T10:02:01-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 Paper

好，我换一种讲法。假设我们坐在咖啡店，你问我这篇 paper 干了啥，我讲给你听。

---

## 一句话版本

Diffusion policy 生成的 action 在 task space 看着挺对，但放到真实 robot 上经常执行不了——joint 够不着、撞墙、controller 跟不上。这帮人没去 retrain model，而是在 sampling 的时候塞了一个 optimization 进去，让 output 既能执行又不破坏原来学到的 behavior。

---

## 为什么这是个真问题

现在 robotics 的 trend 是：训一个 diffusion policy 在 **task space**（end-effector pose、grasp candidate、waypoint trajectory），而不是在某个具体 robot 的 joint space。好处显而易见——UMI 在 gripper 上采数据，deploy 到 Franka；Octo 在多 robot 数据上训，理论上能 transfer。

但有个 gap：你的 diffusion model 学的是"grasp pose 的分布"，不是"Franka 能不能 reach 到这个 pose"。一个 grasp 在分布上 perfectly valid，可 Franka 的 wrist 可能根本到不了那个位置，或者到了也撞自己的 base。

这就是所谓的 **embodiment gap**——task-space 行为 transferable，但 physical feasibility 不 transferable。

UMI-on-Air、DynaGuide 这些 prior work 试着用 gradient guidance 解决，但 gradient guidance 是 "soft"——它推一把，不保证推到位；而且推狠了会把 sample 从 prior 的 manifold 上拽走。

参考：
- UMI: https://universal-manipulation-interface.github.io/
- UMI-on-Air: https://arxiv.org/abs/2504.04840
- Octo: https://octo-models.github.io/

---

## 关键 Insight：DDIM 的一个"结构漏洞"

要理解这帮人干了什么，得先看 DDIM 的 reverse step 长什么样：

$$
x_{k-1} = \mu_\theta(x_k, k) + \sigma_k \omega
$$

这里：
- $x_k$：第 $k$ 步的 noisy sample（$k=K$ 是纯噪声，$k=0$ 是 clean output）
- $\mu_\theta(x_k, k)$：network 决定的 deterministic 方向，下标 $\theta$ 是 pretrained 参数
- $\sigma_k$：第 $k$ 步的 noise scale，由 schedule 决定，随 $k$ 递减
- $\omega \sim \mathcal{N}(0, I)$：标准 Gaussian 噪声，跟 data 无关

注意这个 structure：**前半截是 network 决定的，后半截是 random noise**。

后半截 $\sigma_k \omega$ 是什么？它是为了让 sampling 保持多样性——纯 deterministic 的 DDIM 会 collapse 到 mode，加一点 noise 让 sample 在 trajectory 上 spread。但从 information 角度看，它就是个"自由度"——它不携带任何 prior 信息。

这帮人的 insight 就是：**既然 $\omega$ 是 noise slot，不携带 prior info，那我把这个 slot 换成一个 optimization variable，让 optimization 决定往哪推，不就既不污染 prior 又能 inject 约束吗？**

这是最 clever 的点。跟 classifier guidance 不同——classifier guidance 是把梯度加到 $\hat{x}_0$ 或 score 上，直接修改 network 的行为，会 perturb prior；这帮人是把替换插在"本来就是 noise"的 slot 里，所以 prior 的 trajectory 不会被拽偏。

---

## 那 Optimization 到底长啥样

把 $\omega$ 换成 $\delta_k$（optimization variable）：

$$
x_{k-1} = \mu_\theta(x_k, k) + \sigma_k \delta_k
$$

然后整条 reverse chain $\{x_K, \dots, x_0\}$ 加上所有 $\{\delta_K, \dots, \delta_1\}$ 都是 decision variables，解这个 problem：

$$
\begin{aligned}
\min_{x_K, \{\delta_k\}} \quad & \underbrace{\frac{1}{2}\sum_k \|\delta_k\|^2}_{\text{让扰动尽量小}} + \underbrace{\sum_k \beta_k J(x_k)}_{\text{feasibility cost}} \\
\text{s.t.}\quad & x_{k-1} = \mu_\theta + \sigma_k \delta_k \quad \text{(reverse dynamics 一致性)} \\
& x_0 \in \mathcal{X}_{\text{target}} \quad \text{(terminal hard constraint)} \\
& x_K \in \mathcal{X}_{\text{init}} \quad \text{(可选，限制初始噪声)}
\end{aligned}
$$

变量讲清楚：
- $\delta_k$：第 $k$ 步的 correction，我们要优化的东西
- $\|\delta_k\|^2$：correction 的大小，越小越好——相当于"用最小的扰动满足约束"
- $J(x_k)$：feasibility cost，比如"wrist pose 离 reachable set 多远"。越接近 0 越好
- $\beta_k$：第 $k$ 步的 feasibility 权重，可以 schedule（早期 step 小，后期 step 大，因为早期 $x_k$ 还是近似 Gaussian，$J$ 的 gradient 没意义）
- $\mathcal{X}_{\text{target}} = \{x: J(x) \le \varepsilon_{\text{tol}}\}$：terminal constraint，clean output 必须 feasible。这是 **hard** constraint

这个 problem 的 intuition 就是：在"跟 pretrained DDIM trajectory 尽量像" 和 "满足物理约束" 之间找最优 trade-off。前者用 $\|\delta_k\|^2$ 量化，后者用 $J$ 和 $\mathcal{X}_{\text{target}}$ 量化。

---

## 为什么 $\|\delta_k\|^2$ 这个 regularizer 是"对的"

不是 ad-hoc trick。Appendix 8.2 给了 Bayesian 推导。

把 guided trajectory $\tau = (x_K, \dots, x_0)$ 当 latent variable，写 posterior：

$$
p(\tau \mid \text{feasible}) \propto p_\theta(\tau) \cdot \prod_k \exp(-\beta_k J(x_k))
$$

- $p_\theta(\tau)$：pretrained DDIM 诱导的 trajectory 分布，相当于 **prior**
- $\exp(-\beta_k J)$：Boltzmann factor，相当于 **likelihood**——cost 越低概率越高

DDIM 的 reverse kernel 是 Gaussian：

$$
p_\theta(x_{k-1} \mid x_k) = \mathcal{N}(x_{k-1}; \mu_\theta, \sigma_k^2 I)
$$

reparametrize 一下 $x_{k-1} = \mu_\theta + \sigma_k \delta_k$，取 $-\log$：

$$
-\log p_\theta(x_{k-1}\mid x_k) = \frac{1}{2}\|\delta_k\|^2 + \text{const}
$$

**也就是说 $\|\delta_k\|^2$ 就是 pretrained reverse process 的 negative log-probability**——$\delta_k$ 越偏离 0，trajectory 越偏离 prior。这个 regularizer 是从 prior 里"自然掉出来"的，不是人为加的。

这跟 plug-and-play priors 的精神一致——pretrained model 当 prior，external cost 当 likelihood，做 MAP inference。只不过 PnP 是 image restoration 领域的事，本文在 raw task-space trajectory 上做，并且利用了 DDIM 特有的 $\mu_\theta$ / $\sigma_k\omega$ 分解把 optimization variable 塞进 noise slot。

参考 plug-and-play: https://arxiv.org/abs/2209.10391

---

## 三种 Feasibility Cost：从简单到复杂

### 1. Kinematic Reachability — 这个 pose robot 够得着吗

$$
J_{\text{IK}}(x) = \min_{y \in \mathcal{X}_{\text{IK}}} \|x - y\|_2
$$

- $x$：task-space output，比如 21-D grasp config（wrist 6D + 12 finger joints）
- $\mathcal{X}_{\text{IK}}$：这个 robot 的 reachable set
- $J_{\text{IK}}$：到 reachable set 的最短距离，reachable 时为 0

实现上他们没每次都调 IK solver（太慢），而是训了一个 small MLP surrogate $\hat{J}_{\text{IK}}$：
- 输入：9-D wrist pose，$[t \in \mathbb{R}^3, r_{6D} \in \mathbb{R}^6]$，其中 $r_{6D}$ 是 continuous 6-D rotation representation（比 quaternion 数值稳定，参考 Zhou et al. 2018）
- 输出：标量 reachability distance（softplus 保证非负）+ pose correction offset + joint config（用于 FK consistency loss）
- 训练数据：10^5 个 IK solver 标注的 sample
- 整个 network 才 218k 参数

精度：Franka RMSE 2.9mm，Dynaarm 5.2mm。足够当 guidance cost 用。

参考 continuous 6-D rotation: https://arxiv.org/abs/1812.07035

### 2. Collision Avoidance — 别撞墙

$$
J_{\text{coll}}(x) = \sum_{n=0}^N \max(0, d_{\text{safe}} - s(r_n))^2
$$

- $x = [x_{0|k}, \dots, x_{N|k}]$：trajectory，$n$ 索引 physical execution time
- $s(\cdot)$：signed distance function (SDF)，free space 里 $s > 0$，障碍物里 $s \le 0$
- $r_n$：第 $n$ 步 robot body 上要 query 的点（hand surface points）
- $d_{\text{safe}}$：safety margin，比如 5cm

hinge + squared：超出 safety margin 才罚，penetration 越深罚越重。SDF 可以是 analytical、voxel grid、mesh、neural SDF——framework 不挑。

### 3. Controller-Level Executability — controller 跟得上吗

这是最 clever 的 cost，也是最 dynamic 的：

$$
J_{\text{dyn}}(x) = \|x - \phi(x; q_0; \kappa)\|_2^2
$$

- $x = [x_{0|k}, \dots, x_{N|k}]$：reference trajectory（policy 想执行的）
- $q_0$：robot 初始 joint configuration
- $\kappa$：low-level controller 参数（这里是 Cartesian impedance controller）
- $\phi(\cdot)$：**closed-loop rollout map**——给定 reference 和 controller，预测 actually 会执行的 trajectory

也就是说 $J_{\text{dyn}}$ 衡量的是"你想跑的 trajectory" vs "controller 实际能跑出来的 trajectory" 的 gap。这比 IK 远了一步——不光是 kinematically reachable，还得 dynamically trackable。

#### Rollout 模型具体怎么搞

Appendix 8.5.1 里写的，是个 kinematic approximation，不是 full dynamics simulation：

**Step 1**: 算 task-space error $\Delta x_t = [p_t^{\text{ref}} - p_t^{ee}; \omega_t^{\text{err}}]$，其中 $\omega_t^{\text{err}} = \log(R_t^{\text{ref}} R_t^{ee,\top})^\vee$ 是 SO(3) 上的 rotation 误差转 axis-angle。

**Step 2**: Damped least squares update：
$$
\delta q_t = J(q_t)^\top (J(q_t) J(q_t)^\top + \lambda^2 I_6)^{-1} \Delta x_t
$$
经典 resolved-rate control，$\lambda = 0.05$ 处理 singularity。

**Step 3**: Authority limit，限制单步 joint 增量：
$$
\bar{\delta q} = \min(\dot{q}_{\max} \Delta t_{\text{ref}}, \tau_{\max} / k_p^{\text{joint}})
$$
velocity limit 和 torque limit 同时 cap。

**Step 4**: PD lag + integration，建模 PD controller 在一个 reference step 内只能 close 部分误差。

#### Controller Offset Trick — 防止"作弊"

直接用 $J_{\text{dyn}}$ 做 guidance 有个 bug：模型最容易降 cost 的方法不是让 motion 更 executable，而是让 reference 本身变"更慢/更弱"。比如 fast upward motion controller 跟不上，gradient 就推模型把 $+z$ 加速度降下来——但这个加速度可能是 task 必需的。

解法是引入 bounded feed-forward offset $o$：

$$
J_{\text{dyn}}^*(x) = \min_{o \in \mathcal{B}} J_{\text{dyn}}(\phi(x + o, q_0; \kappa), x) + \lambda_{\text{reg}} \mathcal{R}(o)
$$

- $x + o$ 是实际发给 controller 的 command
- 但 $J_{\text{dyn}}$ 还是 evaluate against original reference $x$（不是 shifted version）
- $\mathcal{R}(o)$ 惩罚过大或不平滑的 offset
- $\mathcal{B}$ 是 admissible offset bound

这是 bilevel optimization：内层求最优 $o$，外层 evaluate guidance gradient。Trick 的意思是——"guidance 只关心 controller 真的吸收不掉的部分"。

这个思路跟 MPC 接近，但便宜得多（不用每 step solve MPC）。

---

## 怎么解这个 Optimization

三种 backend，对应不同场景：

### IPOPT — 严格 NLP solver

Hard constraint，把 $J_{\text{IK}}(x_0) \le 0.01$ 当 hard terminal constraint。IPOPT interior-point，max 45 iterations。精度最高但慢——5.47s per grasp，是 plain DDIM 的 69×。

### Theseus — Differentiable NLS relaxation

把 hard constraint 改成 soft penalty，整个 problem 变成 nonlinear least-squares，用 Levenberg-Marquardt 解。15 iterations，3.63s per grasp，是 DDIM 的 46×。好处是 differentiable，能嵌进 end-to-end learning。

具体 objective（Appendix 8.3, Eq. 21）：

$$
L_{\text{NLS}} = L_{\text{rev}} + L_{\text{rev},0} + L_\delta + L_{\text{clamp}} + L_{\text{IK,cost}} + L_{\text{IK,term}} + L_{\text{IK,path}}
$$

各项是不同权重的 squared residual：
- $L_{\text{rev}} = 10 \sum_k \|x_{k-1} - \mu_\theta - \sigma_k \delta_k\|^2$：reverse dynamics consistency
- $L_\delta = \sum_k \|\delta_k\|^2$：correction regularizer
- $L_{\text{IK,term}} = 10 \max(0, J_{\text{IK}}(x_0) - 0.005)^2$：terminal hinge
- $L_{\text{IK,path}} = 10 \sum_k \max(0, J_{\text{IK}}(x_k) - 0.005)^2$：path hinge

参考：
- IPOPT: https://github.com/coin-or/Ipopt
- Theseus: https://github.com/facebookresearch/theseus

### L-BFGS Direct Shooting — Online Replanning

Visuomotor manipulation 需要 fast online replan，用 L-BFGS。Decision variable 把整条 reverse chain unroll 出来：

$$
z = [x_T \mid \varepsilon \mid o]
$$

- $x_T$：initial latent noise
- $\varepsilon$：所有 step 的 injected noise
- $o$：feed-forward offset

每次 closure re-run 整个 DDIM unroll，evaluate rollout 在 terminal chunk $x_0$，backprop through sampler + rollout。L-BFGS: lr=0.1, max 10 inner iter, 5 closure evals, strong-Wolfe line search。0.28-0.32s per replan，vs plain DDIM 0.02-0.07s。

---

## 实验 1：跨 Arm 的 Dexterous Grasping

### Setup

- Diffusion prior: DexEvolve，21-D grasp（wrist 6D + 12 finger joints），**fixed across all experiments**
- 两个 arm: Franka Panda (7-DoF) vs Dynaarm (6-DoF)，workspace 差异显著
- 30 objects × 8 grasps × 5 base poses = 1200 grasps

### Baselines

1. **DDIM†**：raw diffusion 输出，无 IK 检查，floating gripper——upper bound
2. **DDIM + cuRobo snap**：raw 输出后 project 到 reachable set——decoupled snap baseline
3. **Gradient Guidance**：classifier-style，沿 reverse process 加 IK cost 的 gradient
4. **Projection Guidance**：每 denoising step project 到 IK-feasible set

### 结果怎么读

Table 1 关键数字：

| Method | Dynaarm SR^(1) | Dynaarm Q1 | Franka SR^(1) | Franka Q1 |
|---|---|---|---|---|
| DDIM† (floating) | 75.0 | 16.0 | 75.0 | 16.0 |
| DDIM + cuRobo snap | 27.0 | 6.5 | 33.7 | 7.1 |
| Projection Guidance | 23.7 | 6.8 | 37.4 | 8.1 |
| Gradient Guidance | 58.8 | 12.8 | 50.9 | 11.1 |
| **Ours (Theseus)** | 63.5 | 14.5 | 61.0 | 14.1 |
| **Ours (IPOPT)** | 69.8 | 15.2 | 71.0 | 15.7 |

Intuition：

1. **DDIM† SR=75, Q1=16**——floating gripper 无约束的 upper bound。
2. **DDIM + IK snap**：SR 跌到 27%，Q1 从 16 跌到 6.5——事后 projection 把 sample 拽到 reachable set，但 grasp manifold 也被破坏了。Projection 简单粗暴，feasibility 满足了，quality 崩了。
3. **Projection Guidance**：每 step project 更糟，SR=23.7%。原因：intermediate denoising state 还没有 semantic meaning，project 它等于拽 noise 到 manifold，破坏 score model trajectory。
4. **Gradient Guidance**：soft guidance SR=58.8%，Q1=12.8——work 但 quality 仍有损失。
5. **Ours (IPOPT)**：SR=69.8%，**Q1=15.2**——几乎不损失 grasp quality 的前提下满足 IK！Q1 接近 floating upper bound 的 16。

为什么 quality 能保住？因为 $\|\delta_k\|^2$ regularizer explicit 强迫"最少扰动满足约束"——它不让你为了 feasibility 大幅偏离 prior。

### Runtime 代价

| Method | Time |
|---|---|
| DDIM | 79 ms |
| Gradient/Projection | 84 ms |
| Theseus | 3.6 s (~46×) |
| IPOPT | 5.5 s (~69×) |

代价大。这是 method 当前最大 limitation。

### Base Pose 难度分层

Figure 6：把 base pose 分 easy / hard。Franka 上 Gradient Guidance 从 easy 70% 跌到 hard 23%（-47 pp），IPOPT 从 72 跌到 70（-2 pp）。**Optimization-guided 方法对 pose 变化 robust 得多**。Intuition：tight workspace 下 local gradient 容易推到 local minimum；full trajectory-level optimization 协调性好得多。

### Collision 环境

Table 5：Floor / Walls / Clutter / Tunnel 四个环境。Tunnel 最难——两侧 wall 形成 funnel constraint。

Franka 上 IPOPT 在 4 个环境全是 best SR^Tot：65.62 / 60.62 / 55.63 / 61.88。

有意思的失败模式：Projection Guidance 在 Clutter 上 0% collision rate 但也 0% grasp success——projection 把 sample 推到 free space 远离 object。Gradient Guidance 在 Dynaarm Tunnel 上反而比 IPOPT 好（30.63 vs 23.75），因为 tight tunnel 下 local gradient 方向刚好 favorable。但这种 behavior 不 stable，**同一 gradient 方法跨环境表现 inconsistent**——optimization-based 更 robust。

---

## 实验 2：Visuomotor Manipulation

### Setup

- Diffusion Transformer + CNN image encoder，conditioning on 2-frame RGB + hand pose
- Predict 6-step chunk of wrist poses
- 两个 task: Drawer Opening（简单）, Pick-and-Place（难）
- Prior 训在 floating gripper sim，**zero-shot deploy** 到 embodied arm

### 关键 baseline 设计

三个 guidance 变体，都用同一 frozen model + rollout model，只 correction 入口不同：
1. **$x_t$-nudge**：直接 shift $x_t \mathrel{+}= \lambda\bar\alpha_t g$，再走标准 DDIM
2. **$\sigma$-guidance**：不改 network input，gradient 通过 $x_{t-1} = \mu_t + \sigma_t g$ 注入 stochastic channel——这是本方法的 single-gradient analogue，gap 隔离 "optimize correction" vs "inject gradient" 的 benefit
3. **Ours (L-BFGS)**：full reverse chain unroll + L-BFGS direct shooting

### 结果

Table 2 Pick-and-Place on Franka：

| Method | SR_task |
|---|---|
| DDIM† (floating) | 62.5% |
| DDIM (embodied unguided) | 44.0% |
| Gradient ($\sigma$) | 41.0% |
| Gradient ($x_t$-nudge) | 43.3% |
| **Ours (L-BFGS)** | **67.0%** |

**Ours 超过 floating upper bound**。Intuition：floating gripper 上限 62.5% 是因为 reference trajectory visually plausible 但 controller 跟不上。Optimization-guided denoising 修正 reference 让 controller 能 track——反而比 floating 更高。Subtask SR 98.3% 接近 floating 的 95.3%，主要 gain 在 placement 阶段（需要 regrasping / corrective motion，optimization-based guidance 能 recover）。

Drawer Opening gains 小（79.5→81.8 Dynaarm），因为 task 简单 reference 容易 track。

### Solve Time

L-BFGS 不仅 SR 最高，**solve time 也最短**（Franka Pick&Place 8.02s vs DDIM 8.94s）。Intuition：guidance 让 controller 容易 track，recovery/regrasping 少，episode 整体更快完成。

---

## 跟 Prior Work 的核心区别

| 方法 | 修改什么 | Constraint 类型 | Retrain? | Prior 被扰动? |
|---|---|---|---|---|
| Projected Diffusion [6] | sampling 改成 projection | hard | Yes | 强烈 |
| DPCC [10] | 训练时 + 推理时 projection | hard | Yes | 强烈 |
| UMI-on-Air [9] | $x_t$ 加 gradient | soft | No | 中等 |
| DynaGuide [21] | latent dynamics gradient | soft | No | 中等 |
| Classifier/CF Guidance | score 加 gradient | soft | No | 中等 |
| **本文** | $\omega \to \delta_k$，optimization variable | hard or soft | No | 最小（regularized） |

核心区别：**本文把 feasibility 当 explicit constraint 而非 soft gradient**。所有 gradient 方法都是 "encourage"，本文是 "enforce"。

---

## Limitations 说人话

1. **Runtime 重**：IPOPT 69× 慢，Theseus 46×。L-BFGS 在 visuomotor 上 5-10× 慢——online 用勉强可以
2. **Nonconvex**：没全局 optimality 保证，偶尔 converge failure
3. **Sim-only main eval**：hardware 只是 preliminary
4. **Rollout model 简化**：Cartesian impedance + kinematic rollout，没 full dynamics。Newton 这种 differentiable physics simulator 是 future direction
5. **Hard constraint 风险**：如果 prior 在某 region 完全 infeasible（object 在 robot 远处），NLP 可能 infeasible——paper 没讨论 fallback

---

## 我的联想

### Control as Inference 的影子

这 paper 可以看成 planning-as-inference 的 instantiation：
- Prior: $p_\theta(\tau)$（pretrained diffusion）
- Likelihood: $\exp(-\beta J)$（feasibility 当 observation model）
- Posterior: $p(\tau \mid \text{feasible})$ via MAP

跟 Levine 的 control as inference、Toussaint 的 reasoning as inference 都连得上。Diffusion model 提供了 tractable 的 prior parameterization，把过去 intractable 的 trajectory-level posterior 变成可解的 NLP。

参考：https://arxiv.org/abs/1604.07706

### Schrödinger Bridge 的味道

替换 $\omega \to \delta_k$ 改了 reverse SDE 的 diffusion 项，相当于 controlled SDE。跟 Schrödinger Bridge 思路相通——用 control 修改 stochastic process 的 terminal distribution。区别是 SB 找整个 forward/backward SDE 之间的 transport，本文只在 pretrained DDIM 的 reverse 注入 correction。

参考：https://arxiv.org/abs/2106.01381

### VLA 大模型的 Relevance

Paper Section 7 提到 integrating with large VLA policies。OpenVLA、π0、RT-2 这类大模型 retrain 成本巨大，跨 embodiment deploy 遇到的就是 embodiment gap。可以直接 plug 本文的 optimization module 在 inference 时 inject feasibility。前提是 VLA 输出在 task space；很多 VLA 直接输出 joint action，需要先 ablate 或改 constraint formulation。

参考：
- OpenVLA: https://openvla.github.io/
- RT-2: https://robotics-transformer2.github.io/
- π0: https://www.physicalintelligence.company/blog/pi0

### Runtime 改进方向

46-69× overhead 太重。可能改进：
1. **Amortize**：训 student network distill "optimization-corrected trajectory"
2. **Warm-start**：previous step 的 $\delta$ 当下个 step 的 init
3. **Sparse optimization**：只在最后几 step 做 full optimization
4. **GPU-parallel solver**：cuRobo 思路，batch NLP solve
5. **Differentiable convex layer**：cvxpylayers 路线，relax 成 QP/SOCP

参考 cuRobo: https://curobo.org/
参考 cvxpylayers: https://github.com/cvxgrp/cvxpylayers

### Cross-embodiment 不是完全 zero-cost

仔细看：每个 embodiment 需要训自己的 $\hat{J}_{\text{IK}}$ surrogate（Franka 和 Dynaarm 各一个 MLP）。严格说是 "prior transferable + cheap feasibility module per-embodiment"。比 retrain policy 便宜得多（surrogate 5 epochs / 218k params），但不是 zero-cost。Future direction：universal IK surrogate 参数化 morphology，如 GET-Zero 的 graph embodiment transformer。

参考 GET-Zero: https://arxiv.org/abs/2409.09818

---

## 总结，用人话

**Problem**：Diffusion policy 生成的 task-space action distributionally valid 但 physical infeasible——这是 cross-embodiment deployment 的 embodiment gap。

**Insight**：DDIM 的 reverse step = deterministic network prediction + random noise。Noise slot 不携带 prior info，可以替换成 optimization variable。

**Method**：解一个 constrained optimization——minimize correction magnitude（保持 prior）+ minimize feasibility cost（满足约束），terminal hard constraint 保证 clean output feasible。

**为什么 work**：$\|\delta_k\|^2$ regularizer 是 pretrained reverse process 的 negative log-probability，自然惩罚偏离 prior；optimization 协调整条 trajectory，比 local gradient/projection 更 stable。

**Empirical evidence**：跨两个 arm、两类 task，都超过 gradient/projection baseline，grasp quality 接近 floating upper bound。Visuomotor 甚至超过 floating——optimization 让 reference 更 trackable，反而比 floating 更 executable。

**Catch**：Runtime 46-69× 慢，nonconvex 无 guarantee，rollout model 简化。

**为什么重要**：打开了 inference-time constrained generation 的一族方法——不用 retrain，frozen prior + 可替换 optimization module。对 VLA 大模型 era 尤其 relevant，因为 retrain cost 越来越高。

---

要 drill down 哪块？Controller offset 的 bilevel、surrogate MLP 细节、Schrödinger bridge 视角、还是 runtime 优化的具体方向，跟我说。

---

# Grounding Generative Policies in Physics: 深度解读

Andrej，这篇 paper 直击 diffusion policy deployment 中一个真实存在的痛点。我尽量把技术细节、intuition、相关联想都铺开，供你 building mental model。

---

## 1. 核心问题：Task-Space Prior 的 "Embodiment Gap"

当前 trend 是把 diffusion policy 训练在 **task space**（end-effector pose、waypoint、grasp candidate），好处是跨 embodiment transferable——UMI [1], Octo [2], OpenVLA [3], EgoVerse [4] 都是这条路线。问题在于：task-space 输出 **distributionally valid**，但 **physically infeasible**。一个 grasp pose 在 wrist 6D + finger 12D 上是"合法的 grasp 分布的 sample"，可放到 Franka vs Dynaarm 上，可能 joint limit、self-collision、reachability 全部爆掉。

这就是 paper Section 1 所说的 **embodiment gap**。Cross-embodiment 学习的 literature（RT-X [5], Octo, UMI [6], GET-Zero [7]）大部分是"transfer policy"，feasibility 留给 downstream controller 隐式吸收——没有显式 inference-time mechanism 保证 task-space sample 在 target robot 上 executable。

参考链接：
- UMI: https://universal-manipulation-interface.github.io/
- Octo: https://octo-models.github.io/
- OpenVLA: https://openvla.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/

---

## 2. 关键 Insight: DDIM Reverse Step 的结构性分解

这是整篇 paper 最 elegant 的点。DDIM reverse update (Eq. 2):

$$
x_{k-1} = \underbrace{\sqrt{\alpha_{k-1}}\hat{x}_0 + \sqrt{1-\alpha_{k-1}-\sigma_k^2}\hat{\varepsilon}_\theta(x_k,k)}_{\mu_\theta(x_k,k)} + \sigma_k \omega, \quad \omega\sim\mathcal{N}(0,I)
$$

变量含义：
- $x_k \in \mathbb{R}^d$：第 $k$ 步 noisy sample（$k=K$ 是纯噪声，$k=0$ 是 clean prediction）
- $\alpha_k \in [0,1]$：noise schedule 控制的 signal retention ratio，$\alpha_K\to 0$，$\alpha_0\to 1$
- $\hat{x}_0$：当前 step 估计的 clean sample
- $\hat{\varepsilon}_\theta(x_k,k)$：score network 预测的 noise
- $\mu_\theta(x_k,k)$：**deterministic model prediction**，由 $\theta$ 决定
- $\sigma_k$：stochastic perturbation 的 scale，由 schedule (Eq. 11) 给出
- $\omega\sim\mathcal{N}(0,I)$：i.i.d. Gaussian **sampling perturbation**

DDIM 的设计就是把"network 决定方向" 与"random 探索" **factorize** 出来。$\mu_\theta$ 是 score model 学到的方向，$\sigma_k\omega$ 是让 sample 在 reverse chain 上保持 distribution diversity 的随机扰动。

**Insight**: 既然 $\sigma_k\omega$ 是"自由度"——它本来就是 noise，不携带 prior information——那把它替换成一个 **optimized correction** $\delta_k$，就相当于在 reverse trajectory 的每一注入一个"small structured deviation"，**不修改** $\mu_\theta$，**不 retrain** score network，同时可以 inject 物理约束。

这个 observation 跟 classifier guidance [8] / classifier-free guidance [9] 的精神相反——后者是把梯度加到 $\hat{x}_0$ 或 score 上，会 **perturb learned prior**；本文是把替换插在"本来就是 noise 的 slot"里，所以 prior 不会被"拽走"。这一点是相对 UMI-on-Air [10] 和 DynaGuide [11] 的核心差异。

参考：
- Classifier guidance (Dhariwal & Nichol): https://arxiv.org/abs/2105.05233
- Classifier-free guidance: https://arxiv.org/abs/2207.12598
- DDIM: https://arxiv.org/abs/2010.02502

---

## 3. Optimization-Constrained Denoising: 形式化

### 3.1 替换式 (Eq. 3)

$$
x_{k-1} = \mu_\theta(x_k, k) + \sigma_k \delta_k
$$

$\delta_k$ 是 **optimization variable**，不再 random。整条 reverse chain $\{x_K, x_{K-1}, \dots, x_0\}$ 以及 $\{\delta_K, \dots, \delta_1\}$ 都是 decision variables。

### 3.2 Constrained Optimization Problem (Eq. 4)

$$
\begin{aligned}
\min_{x_K, \{\delta_k\}_{k=1}^K}\quad & \frac{1}{2}\sum_{k=1}^K \|\delta_k\|_2^2 + \sum_{k=0}^K \beta_k J(x_k) \\
\text{s.t.}\quad & x_{k-1} = \mu_\theta(x_k,k) + \sigma_k\delta_k, \quad k=K,\dots,1 \\
& x_0 \in \mathcal{X}_{\text{target}} \\
& x_K \in \mathcal{X}_{\text{init}}
\end{aligned}
$$

变量解释：
- $\frac{1}{2}\|\delta_k\|_2^2$：**correction regularizer**，惩罚偏离 nominal sampler 的程度。直觉上：让 $\delta_k$ 越小越好，意味着"用最小的扰动满足约束"。
- $J(x_k):\mathbb{R}^d\to\mathbb{R}_{>0}$：**feasibility cost**，比如 wrist pose 到 reachable set 的距离。
- $\beta_k$：feasibility penalty 的 schedule weight。Paper 在早期 step 把 $\beta_k$ 调小（因为早期 $x_k$ 接近 Gaussian，$J$ 的 gradient 没意义），后期 step 把 $\beta_k$ 加大，让约束只在 close-to-data-manifold 时才发力。这跟 DPS [12] 和 universal guidance [13] 的"only guide near end"思路一致。
- $\mathcal{X}_{\text{target}} = \{x: J(x)\le \varepsilon_{\text{tol}}\}$：**terminal constraint**，clean sample 必须满足 tolerance。这是 **hard constraint**——UMI-on-Air、DynaGuide 都做不到 hard。
- $\mathcal{X}_{\text{init}}$：可选，对初始噪声做限制（默认 $\mathbb{R}^d$）。

### 3.3 Bayesian / MAP Interpretation (Appendix 8.2)

这是理解为什么 $\frac{1}{2}\|\delta_k\|_2^2$ 是"对的" regularizer 的关键。

Posterior over trajectory $\tau = (x_K, \dots, x_0)$ conditioned on feasibility:

$$
p(\tau \mid \text{feasible}) \propto p_\theta(\tau) \prod_{k=1}^K \ell_k(x_k)
$$

其中：
- $p_\theta(\tau)$：pretrained DDIM 诱导的 trajectory distribution
- $\ell_k(x_k) \propto \exp(-\beta_k J(x_k))$：**Boltzmann pseudo-likelihood**，cost 越低概率越高

Prior factorizes 沿 reverse chain：

$$
p_\theta(\tau) = p(x_K)\prod_{k=1}^K p_\theta(x_{k-1}\mid x_k)
$$

DDIM 的 reverse kernel 是 Gaussian (Eq. 16):

$$
p_\theta(x_{k-1}\mid x_k) = \mathcal{N}(x_{k-1}; \mu_\theta(x_k,k), \sigma_k^2 I)
$$

reparametrize $x_{k-1} = \mu_\theta + \sigma_k\delta_k$，那么 $-\log p_\theta(x_{k-1}\mid x_k)$ 在 $\delta_k$ 上 reduce 成 $\frac{1}{2}\|\delta_k\|^2 + \text{const}$（Eq. 19）。

也就是说：**$\|\delta_k\|^2$ 项就是 pretrained reverse process 的 negative log-density**——$\delta_k$ 越偏离 0，越偏离 prior。这与"用最少扰动满足约束"的直觉完全吻合，**有 probabilistic justification**，不是 ad-hoc trick。

这跟 plug-and-play priors [14] 的精神类似——pretrained model 当 prior，external cost 当 likelihood，做 MAP inference。区别是 PnP 是 image 重建领域，且通常在 latent space 操作；本文在 raw task-space trajectory 上做，并且利用 DDIM 特有的 $\mu_\theta/\sigma_k\omega$ 分解来"slot in" optimization variable。

参考：
- DPS: https://arxiv.org/abs/2209.14687
- Universal guidance: https://arxiv.org/abs/2302.07185
- Plug-and-play priors: https://arxiv.org/abs/2209.10391
- Theseus library: https://github.com/facebookresearch/theseus

---

## 4. 三类 Feasibility Cost $J$ 的 Instantiation

这是 paper 最实用的部分——同一框架 plug-in 不同 $J$ 就能处理不同约束。

### 4.1 Kinematic Reachability (Eq. 5, 6)

$$
J_{\text{IK}}(x) = \min_{y\in\mathcal{X}_{\text{IK}}}\|x-y\|_2
$$

- $x$：task-space output（如 wrist 6D pose + finger 12D joints = 21-D grasp configuration）
- $\mathcal{X}_{\text{IK}}\subseteq\mathbb{R}^d$：reachable set（由 robot kinematics 决定）
- $J_{\text{IK}}$：到 reachable set 的 Euclidean 距离，reachable 时为 0
- $\mathcal{X}_{\text{target}}^{\text{IK}} = \{x: J_{\text{IK}}(x)\le \varepsilon_{\text{IK}}\}$：tolerance sublevel set

实现上 paper 训了一个 small MLP surrogate $\hat{J}_{\text{IK}}$（Appendix 8.5.0），输入 9-D wrist pose $[t\in\mathbb{R}^3, r_{6D}\in\mathbb{R}^6]$（continuous 6-D rotation representation [15]，比 quaternion 数值更稳定），输出：
- $\hat{J}_{\text{IK}}$：标量 reachability distance（softplus 保证非负）
- $\Delta p\in\mathbb{R}^9$：pose correction offset
- $\hat{q}\in\mathbb{R}^{dof}$：joint configuration（用于 FK consistency loss）

网络仅 ~218k 参数，用 100k 个 IK solver 标注的 sample 训练，RMSE 2.9mm (Franka) / 5.2mm (Dynaarm)，ranking AUC 0.993 / 0.930（Table 4）。这种"surragate for IK"思路跟 CabiNet [16], NeuralIK [17] 一脉相承，但这里 surrogate 仅作为 optimization cost 评估，不作为最终 IK solver。

参考 continuous 6-D rotation：https://arxiv.org/abs/1812.07035

### 4.2 Collision Avoidance (Eq. 7)

$$
J_{\text{coll}}(x) = \sum_{n=0}^N \max(0, d_{\text{safe}} - s(r_n))^2
$$

- $x = [x_{0|k}, \dots, x_{N|k}]$：trajectory，$n$ 索引 physical execution time
- $s:\mathbb{R}^m\to\mathbb{R}$：signed distance function (SDF)，$s>0$ free space, $s\le 0$ inside obstacle
- $r_n$：第 $n$ 步 robot body 上的 query point（如 hand surface points）
- $d_{\text{safe}}\ge 0$：safety margin
- **hinge + quadratic**：超出 safety margin 才罚，penetration 越深 penalty 越大

注意 paper 在 collision 实验里实际用的是更简化的 Eq. 39：

$$
J_{\text{coll}}(x) = \max(0, s(x) + d_{\text{safe}})
$$

这是 linear hinge（不是 squared），更适合 IPOPT 这种 NLP solver 的 conditioning。

### 4.3 Controller-Level Executability (Eq. 8) — 最有意思的部分

$$
J_{\text{dyn}}(x) = \|x - \phi(x; q_0; \kappa)\|_2^2
$$

- $x = [x_{0|k},\dots,x_{N|k}]$：reference trajectory
- $q_0$：robot 初始 joint configuration
- $\kappa$：low-level controller 参数（这里是 Cartesian impedance）
- $\phi(\cdot)$：**closed-loop rollout map**——给定 reference 和 controller，预测 actually executed trajectory

也就是说：$J_{\text{dyn}}$ 衡量 "你想执行的" vs "控制器实际能实现的" 之间的 gap。**这就把 constraint 从 pure kinematic 推进到 dynamic feasibility**，比 DPCC [18]、UMI-on-Air 更直接。

#### Closed-loop rollout 模型细节 (Appendix 8.5.1)

每 step 的 controller update 四步：

**Step 1: Task-space error**
$$
\Delta x_t = \begin{bmatrix} p_t^{\text{ref}} - p_t^{ee} \\ \omega_t^{\text{err}} \end{bmatrix}\in\mathbb{R}^6
$$

$\omega_t^{\text{err}} = \log(R_t^{\text{ref}} R_t^{ee,\top})^\vee$，$\log(\cdot)^\vee: SO(3)\to\mathbb{R}^3$ 是 SO(3) 的 logarithmic map，把 rotation 误差变成 axis-angle vector。

**Step 2: Damped least squares (DLS) update**
$$
\delta q_t = J(q_t)^\top (J(q_t)J(q_t)^\top + \lambda^2 I_6)^{-1}\Delta x_t, \quad \lambda = 0.05
$$

经典 resolved-rate motion control [19]，$\lambda$ 是 damping factor 处理 singularities。$J(q_t)$ 是 geometric Jacobian。

**Step 3: Authority limits**
$$
\bar{\delta q} = \min\left(\dot{q}_{\max}\Delta t_{\text{ref}}, \frac{\tau_{\max}}{k_p^{\text{joint}}}\right)
$$

velocity limit 和 torque/effort limit 一起 cap 增量。

**Step 4: PD lag + integration**
$$
q_{t+1} = \text{clip}(q_t + \alpha_{\text{eff}}\odot\text{clip}(\delta q_t, \pm\bar{\delta q}), \underline{q}, \bar{q})
$$

$\alpha_{\text{eff},j} = 1 - (1 - \frac{k_{p,j}^{\text{joint}}}{k_{p,j}^{\text{joint}} + k_{d,j}^{\text{joint}}/\Delta t_{\text{in}}})^{n_{\text{sub}}}$

first-order lag model，$n_{\text{sub}}$ 是 inner control steps per reference interval。这里 paper 假设 PD controller 只 close 部分 commanded gap——这跟真实 hardware 行为接近。

#### Controller Offset Trick (Eq. 24) — 关键 design choice

直接用 $J_{\text{dyn}}$ 当 guidance 会有 **feedback pathology**：模型最容易降 cost 的方法不是让 motion 更 executable，而是让 reference 本身变"更慢/更弱"。比如 fast upward motion controller 跟不上，gradient 会推模型把 $+z$ 加速度降下来——但这个加速度可能是 task 完成必需的。

解决方法：引入 bounded feed-forward offset $o\in\mathcal{B}$：

$$
J_{\text{dyn}}^*(x) = \min_{o\in\mathcal{B}} J_{\text{dyn}}(\phi(x+o, q_0;\kappa), x) + \lambda_{\text{reg}}\mathcal{R}(o)
$$

- $x+o$：实际发给 controller 的 command
- $J_{\text{dyn}}$ 仍 evaluate against **original reference** $x$（不是 shifted version）
- $\mathcal{R}(o)$：regularizer 惩罚过大/不平滑的 offset
- $\mathcal{B}$：admissible offset bound

这是 **bilevel optimization**——内层求最优 $o$，外层 evaluate guidance gradient。这个 trick 让 "guidance 只关心 controller 真的无法吸收的部分"，跟 MPC 思路殊途同归，但轻量得多（不用每 step solve MPC）。

参考：
- Resolved-rate control (Whitney 1969): 经典 robotics
- Cartesian impedance control (Hogan 1985): https://www.sciencedirect.com/science/article/pii/0020748585900415
- DPCC: https://arxiv.org/abs/2410.07702

---

## 5. 两种 Solver 实现

### 5.1 严格 NLP (IPOPT) — Eq. 25

$$
\begin{aligned}
\min_{x_K, \{\delta_k\}}\quad & L_\delta + L_{\text{IK}} \\
\text{s.t.}\quad & x_{k-1} = \mu_\theta(x_k,k) + \sigma_k\delta_k \\
& J_{\text{IK}}(x_0)\le \varepsilon_{\text{IK}} \\
& -1\le x_K\le 1 \\
& -1\le \delta_k\le 1
\end{aligned}
$$

IPOPT [20] interior-point method，max 45 iterations。这是 **hard constraint**：terminal $J_{\text{IK}}(x_0)\le 0.01$ 强制 clean sample IK-feasible。

### 5.2 Differentiable NLS Relaxation (Theseus) — Eq. 21

把 hard constraint 改成 soft penalty，整 problem 变成 nonlinear least-squares，用 Levenberg-Marquardt 解：

$$
\min_z \quad L_{\text{rev}} + L_{\text{rev},0} + L_\delta + L_{\text{clamp}} + L_{\text{IK,cost}} + L_{\text{IK,term}} + L_{\text{IK,path}}
$$

- $z = \{x_K,\dots,x_0,\delta_K,\dots,\delta_1\}$：所有变量
- $L_{\text{rev}} = 10\sum_k\|x_{k-1}-\mu_\theta-\sigma_k\delta_k\|^2$：reverse dynamics 一致性 residual
- $L_{\text{rev},0} = 50\|x_0-\mu_\theta-\sigma_1\delta_1\|^2$：最后一步权重更大
- $L_\delta = \sum_k\|\delta_k\|^2$：correction regularizer
- $L_{\text{clamp}} = 100\sum_k\|\max(0,\delta_k-1)+\max(0,-1-\delta_k)\|^2$：把 box constraint 改成 penalty
- $L_{\text{IK,cost}} = \sum_k J_{\text{IK}}(x_k)^2$：path 平滑 cost
- $L_{\text{IK,term}} = 10\max(0, J_{\text{IK}}(x_0) - 0.005)^2$：terminal hinge
- $L_{\text{IK,path}} = 10\sum_k\max(0, J_{\text{IK}}(x_k) - 0.005)^2$：path hinge

Theseus 15 LM iterations。这是 **soft**——feasibility 不保证，但 differentiable，可以嵌进 end-to-end learning pipeline。

### 5.3 Online Replanning: L-BFGS Direct Shooting (Visuomotor)

对 image-conditioned manipulation，paper 用 L-BFGS，因为需要 fast online replan。Decision variable：

$$
z = [x_T \mid \varepsilon \mid o]
$$

- $x_T$：initial latent noise
- $\varepsilon$：injected denoising noises
- $o$：feed-forward offset

L-BFGS 设置：lr=0.1，max 10 inner iterations，5 closure evals，history size 10，strong-Wolfe line search。每次 closure **re-run full DDIM unroll** with frozen network，evaluate $\phi$ on terminal chunk，backprop through sampler + rollout。

这跟 Diffusion Policy [21] + MPC-style reasoning 思路类似，但 L-BFGS 比 MPC 便宜得多。

参考：
- IPOPT: https://github.com/coin-or/Ipopt
- Theseus: https://github.com/facebookresearch/theseus
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

---

## 6. 实验：跨 Embodiment Grasp Synthesis (Section 5.1)

### 6.1 Setup

- Diffusion prior: DexEvolve [22]，21-D grasp（wrist 6D + 12 finger joints），**fixed** across all experiments
- 两个 arm: Franka Panda (7-DoF) [23] vs Dynaarm (6-DoF) [24]，workspace 和 joint limits 差异显著
- 30 objects × 8 grasps × 5 base poses = 1200 grasps

### 6.2 主表 (Table 1) 关键读法

| Method | Dynaarm SR^(1) | Dynaarm Q1 | Franka SR^(1) | Franka Q1 |
|---|---|---|---|---|
| DDIM† (no IK, floating) | 75.0 | 16.0 | 75.0 | 16.0 |
| DDIM [27] + cuRobo snap | 27.0 | 6.5 | 33.7 | 7.1 |
| Projection Guidance | 23.7 | 6.8 | 37.4 | 8.1 |
| Gradient Guidance | 58.8 | 12.8 | 50.9 | 11.1 |
| **Ours (Theseus)** | 63.5 | 14.5 | 61.0 | 14.1 |
| **Ours (IPOPT)** | 69.8 | 15.2 | 71.0 | 15.7 |

**直觉解读**：

1. DDIM† 是 upper bound——floating gripper 无 kinematic 约束，pure diffusion quality。SR=75, Q1=16.
2. DDIM + IK-snap 把 sample 投到 reachable set，但 Q1 从 16 跌到 6.5——**projection 破坏 grasp manifold**。这就是为什么"事后 projection"不够：feasibility 满足了，但 prior 学到的 grasp structure 也丢了。
3. Projection Guidance 每步 project——更糟，SR 23.7%。原因：intermediate denoising state 还没"semantic meaning"，project 它等于把 noise 拽到 manifold 上，破坏 score model 的 trajectory。
4. Gradient Guidance 用 classifier-style gradient nudging，SR 升到 58.8%——soft guidance 能 work，但 Q1 还是损失（12.8 vs 16）。
5. **Ours (IPOPT)** SR=69.8, Q1=15.2——**接近 floating upper bound**！意味着几乎不损失 grasp quality 的前提下满足 IK constraint。这是 $\|\delta_k\|^2$ regularizer 的功劳——它 explicit 强迫"最少扰动"。

这就是 paper Section 5.1.1 强调的："enforcing feasibility need not sacrifice grasp quality"——核心 mechanism 是 regularizer 把 sample 拽回 prior。

### 6.3 Runtime

| Method | Time per trajectory |
|---|---|
| DDIM | 79.3 ± 1.3 ms |
| Gradient Guidance | 84.2 ± 0.9 ms |
| Projection Guidance | 84.1 ± 1.9 ms |
| Ours (Theseus) | 3.63 ± 0.03 s (~46×) |
| Ours (IPOPT) | 5.47 ± 0.6 s (~69×) |

代价很明显——IPOPT 69× 慢。这是 method 当前最大 limitation。Theseus 比 IPOPT 快但 soft constraint。L-BFGS 在 visuomotor 实验里只 0.28-0.32s（vs DDIM 0.02-0.07s）——更快但仍 5-10× overhead。

### 6.4 Base-pose Difficulty Stratification (Figure 6)

把 base pose 分 easy/hard。Franka 上：
- Gradient Guidance: easy 70 → hard 23 (-47 pp)
- Projection Guidance: easy 57 → hard 9 (-48 pp)
- IPOPT: easy 72 → hard 70 (-2 pp)
- Theseus: easy 65 → hard 56 (-9 pp)

**Optimization-guided 方法对 base pose 变化 robust 得多**。直觉：当 workspace constraint 紧时，local gradient / projection 容易把 sample 推到 local minimum；而 full constrained optimization 在整条 reverse trajectory 上做协调，更 stable。

### 6.5 Collision 环境 (Table 5, Appendix 8.7)

4 个环境：Floor / Walls / Clutter / Tunnel（Figure 7）。Tunnel 最难——两侧 wall 形成 funnel constraint。

Franka 上 IPOPT 在 4 个环境都是 best SR^Tot：65.62 / 60.62 / 55.63 / 61.88。

Dynaarm 在 Tunnel 上 Gradient Guidance 反而 best (30.63 vs IPOPT 23.75)。Paper 解释：tight tunnel 下 local gradient signal 方向 favorable，没必要 global optimization。但同一 gradient 方法在 Clutter 上 0% SR——projection 会把 sample 推到 free space 远离 object。这种 "depends on environment" behavior 说明 gradient guidance unstable，**optimization-based 更 robust across environments**。

### 6.6 Real-world Deployment (Appendix 8.7.1)

Franka + XHand hardware test，4 个环境，zero-shot deploy（不 retrain）。Floor 上 Gradient Guidance 经常 collision-free 但 grasp 失败（Figure 8——hand 被 push 到 floor 上方，离 object 太远）；IPOPT 在保 grasp structure 的同时 respect collision。是 initial evidence 但不是完整 benchmark（approach motion 用 cuRobo 做 collision avoidance，与 method 解耦）。

---

## 7. Visuomotor Manipulation (Section 5.2, Table 2)

### 7.1 Setup

- Diffusion Transformer (DiT) + CNN image encoder，conditioning on 2-frame RGB + hand pose
- Predicts 6-step chunk of wrist poses
- 两个 task: Drawer Opening, Pick-and-Place
- 两个 arm: Dynaarm, Franka
- Prior trained on **floating gripper** simulation (modified Hoi! gripper [25])，**zero-shot deploy 到 embodied arm**

### 7.2 关键 baseline 设计

三个 guidance variant 都用同一 frozen model + rollout model，只 correction 入口不同：
1. **$x_t$-nudge**: 直接 shift $x_t \mathrel{+}= \lambda\bar\alpha_t g$，再走标准 DDIM——影响 network input
2. **$\sigma$-guidance**: 不改 network input，gradient 通过 $x_{t-1} = \mu_t + \sigma_t g$ 注入 stochastic channel——这是本方法 "single-gradient analogue"，gap 隔离 "optimize correction vs inject gradient" 的 benefit
3. **Ours (L-BFGS)**: 全 reverse process unroll + L-BFGS direct shooting

### 7.3 结果

**Pick-and-Place** (难任务) on Franka:
- DDIM† (floating upper bound): 62.5% SR
- DDIM [27] (embodied unguided): 44.0%
- Gradient Guidance ($\sigma$): 41.0%
- Gradient Guidance ($x_t$-nudge): 43.3%
- **Ours (L-BFGS): 67.0%** —— **超过 floating upper bound**！

直觉：floating gripper 之前 67% 是因为 reference trajectory visually plausible 但 controller 跟不上。Optimization-guided denoising 修正 reference 让 controller 能 track——所以反而比 floating 还高。**subtask SR 98.3% 接近 floating 的 95.3%**，意味着策略 grasp + lift 阶段几乎完美，主要 gain 在 placement 阶段（需要 regrasping / corrective motion，optimization-based guidance 能 recover）。

Drawer Opening gains 小（79.5→81.8 Dynaarm, 57.8→60 Franka），因为 task 简单 reference 容易 track，guidance 收益小。

### 7.4 Solve Time (Table 3)

L-BFGS 不仅 SR 最高，**solve time 也最短**（Franka Pick&Place 8.02s vs DDIM 8.94s vs Gradient 9.22s）。直觉：guidance 让 controller 更容易 track，recovery / regrasping 减少，整体 episode 更快完成。Dist metric 上 L-BFGS 路径更长（1.21 vs 0.98）——因为它完成更多 episode，包括 recovery 动作。

---

## 8. Limitations (Section 6)

1. **Runtime**: NLP 46-69× 慢；Theseus/L-BFGS 软约束但更便宜
2. **Nonconvex**: 无全局 optimality 保证，occasional convergence failure on ill-conditioned instances
3. **Sim-only main eval**（hardware preliminary）
4. **Rollout model 简化**: Cartesian impedance + kinematic rollout——更 high-fidelity 的 MPC / differentiable physics simulator (e.g., Newton [26]) 是 future direction

---

## 9. 个人 Intuition & 联想

### 9.1 跟 Control as Inference 的 connection

整篇 paper 可以看成 **planning-as-inference** 的一个 instantiation：
- Prior: pretrained diffusion policy $p_\theta(\tau)$
- Likelihood: $\exp(-\beta J)$，feasibility 当 observation model
- Posterior: $p(\tau\mid\text{feasible})$ via MAP

这跟 Kaelbling's POMDP planning [27], Levine's GPS [28], Toussaint's reasoning as inference [29] 都连得上。Diffusion model 提供 tractable 的 prior parameterization，把过去 intractable 的 trajectory-level posterior 变成可解的 NLP。

### 9.2 Score-based 视角

替换 $\omega\to\delta_k$ 实际上改了 reverse SDE 的 **drift**：原本 reverse SDE 有 deterministic drift $\mu_\theta$ + diffusion $\sigma_k\omega$；现在 diffusion 项变成 controlled。这跟 Controlled SDE / Schrödinger Bridge [30] 思路相通——用 control 修改 stochastic process 的 terminal distribution。区别是 Schrödinger Bridge 找整个 forward/backward SDE 之间的 transport，本文只在 pretrained DDIM 的 reverse 注入 correction。

### 9.3 跟 Diffusion + Planning 的关系

Diffuser [31], Diffusion-CCSP [32], Potential-based motion planning [33] 都把 planning 看成 conditional diffusion。本文区别是：
- 它们 condition 在 goal / constraint **gradient** 上（soft）
- 本文 condition 在 **hard constraint** 上（terminal set）
- 本文不 retrain，pure inference-time

### 9.4 跟 RL 中的 Constrained Policy Search 的关系

Constrained MDP / Lagrangian RL [34, 35] 是另一种 formulate——把约束塞到 training objective。但 RL 需要 environment interaction，data inefficient。本文 inference-time 方法 **不需要 interaction**，直接 leverage frozen prior。

### 9.5 跟 VLA 大模型的 relevance

Paper Section 7 提到 integrating with large VLA policies。这是非常自然的 extension——OpenVLA, π0 [36], RT-2 [37] 这类 VLA 大模型 retrain 成本巨大，跨 embodiment deploy 时遇到的就是本文解决的 embodiment gap。可以直接 plug 本文的 optimization module 在 inference 时 inject feasibility。但前提是 VLA 输出在 task space（end-effector），而很多 VLA 直接输出 joint action——这时需要先 ablate 到 task space，或者改 constraint 形式ulation。

### 9.6 Runtime 优化方向

46-69× overhead 太重。几个可能改进：
1. **Amortize**：训一个 student network distill "optimization-corrected trajectory"，类似 VAE posterior → amortized inference
2. **Warm-start**：previous denoising step 的 $\delta$ 当下一个 step 的 init
3. **Sparse optimization**：只在最后几 step 做 full optimization，前几 step 用 cheap gradient
4. **GPU-parallel solver**：cuRobo [38] 思路，batch NLP solve
5. **Differentiable convex optimization layer**：cvxpylayers [39] 路线，把 NLP relax 成 QP/SOCP，更可微

### 9.7 关于 Hard Constraint 的风险

Hard terminal constraint $\mathcal{X}_{\text{target}}$ 保证 feasibility，但若 prior 在某 region 完全不可行（如 object 在 robot 远处），NLP 可能 infeasible 或 converge 到 poor local minimum。Paper 没深入讨论 infeasibility handling——未来需要 mechanism fallback 到 prior-only 或 human-in-loop。

### 9.8 Cross-embodiment 的真正意义

Paper 把"同一 prior + 换 optimization module"作为 cross-embodiment selling point。但仔细看：每个 embodiment 需要训 **自己的 $\hat{J}_{\text{IK}}$ surrogate**（Franka 和 Dynaarm 各一个 MLP）。所以严格说，是 "prior transferable + cheap feasibility module per-embodiment"。这比 retrain policy 便宜得多（surrogate 训 5 epochs / 218k params），但不是 zero-cost。

未来方向：universal IK surrogate（参数化 morphology），如 GET-Zero [7] 的 graph embodiment transformer。

参考：
- Control as inference: https://arxiv.org/abs/1604.07706
- Schrödinger Bridge: https://arxiv.org/abs/2106.01381
- Diffuser: https://diffusion-planning.github.io/
- Diffusion-CCSP: https://diffusion-ccsp.github.io/
- RT-2: https://robotics-transformer2.github.io/
- π0: https://www.physicalintelligence.company/blog/pi0
- cuRobo: https://curobo.org/
- cvxpylayers: https://github.com/cvxgrp/cvxpylayers

---

## 10. 总结

这篇 paper 我觉得 contribution 干净利落：

1. **结构性洞察**：DDIM 的 $\mu_\theta/\sigma_k\omega$ 分解提供了一个 principled interface，让"feasibility"可以 inject 进 reverse process 的"noise slot"，不污染 prior
2. **Bayesian 严格性**：$\|\delta_k\|^2$ regularizer 有 MAP 解释，不是 ad-hoc
3. **统一接口**：同一 framework 涵盖 IK / collision / controller-level executability，三种 $J$
4. **Empirical 验证**：跨两个 arm、两类 task，都超过 gradient/projection baseline，且 grasp quality 接近 floating upper bound
5. **Zero-shot deploy**：hardware 上 initial evidence

主要 weakness：runtime 太重（46-69×），nonconvexity 无 guarantee，rollout model 简化。但作为 **first paper 把 diffusion guidance 当 constrained optimization 而非 soft gradient** 来做，我觉得打开了后续一大片工作——amortized solver、Schrödinger bridge connection、VLA integration、differentiable physics rollout 等等。

如果想 build 更深的 intuition，我建议从 Eq. 4 出发，把它当 **MAP inference 问题**，先理解 prior ($\|\delta_k\|^2$) 和 likelihood ($\beta_k J$) 的分工，然后理解 terminal hard constraint ($\mathcal{X}_{\text{target}}$) 是把 likelihood "sharp 化"到 indicator function。这样后面 NLS relaxation (Eq. 21)、IPOPT formulation (Eq. 25)、L-BFGS direct shooting 都自然 fall out as 不同的 solver / relaxation。

希望这个解读有用。如果某个点想再 drill down（比如 controller offset 的 bilevel、surrogate MLP 架构细节、SDF 选择、Schrödinger bridge 视角），告诉我，我再展开。
