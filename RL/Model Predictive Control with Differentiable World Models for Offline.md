---
source_pdf: Model Predictive Control with Differentiable World Models for Offline.pdf
paper_sha256: 733a82e10aeebedec4b3ba246d881df00e24529784e456741bf8cc47b373ce37
processed_at: '2026-08-05T19:47:13-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

## 一句话版本

Offline RL 以前是"训练完就 freeze，部署时 forward 一次出 action"。这篇 paper 说：部署的时候别偷懒，拿当前 state 当起点，用学好的 world model 在脑子里想象几步未来，看哪条路回报高，就把 policy 参数往那个方向微调一下，再执行 action。每一步都这么做。

---

## 为什么要这么做

Offline RL 的根本痛点是：你要从别人收集的固定数据里学一个 policy，部署的时候遇到的 state 可能数据里见过，也可能没见过。传统做法是把 policy 和 Q-function 都学准，然后部署时直接用。

问题在于 Q-function 这个东西很难学准。它本质上是"从某个 state 出发，未来所有 reward 加起来是多少"——这是个 long-horizon 量，得一步步 bootstrap，每一步都有误差，累积起来噪声很大。而且 offline 数据是某个 behavior policy 收集的，你要估的是 optimal policy 的 Q，两者分布不一致，误差更大。

但 dynamics $P(s'|s,a)$ 和 reward $r(s,a)$ 是 local 量——只跟当前这一步有关，跟 policy 无关，跟 long horizon 无关。从数据里学这两个 local 量比学 Q-function 容易得多，误差可控。

所以思路就是：把"long horizon 的困难"（Q 不准）换成"short horizon 的可控"（local model 准），用 world model rollout 几步，剩下交给 Q 估算。

---

## 核心 trick：可微的 imagination

光有 world model 还不够。如果只是"想象几步，看哪个 action 好"，那是 sampling-based planning，像 Diffuser 那样——sample 一堆 trajectory，挑个好的，执行第一步。policy 参数本身不动。

这篇 paper 的关键在于：**整个 imagination pipeline 是 differentiable 的**。policy 出 action → diffusion model 出 next state → reward model 出 reward → critic 估 terminal value，这条链从头到尾可以 backprop。所以可以算 $\nabla_\psi J$，直接对 policy 参数做 gradient ascent。

这跟 LLM test-time scaling 的精神一样——花更多 inference computation 换更好性能。但 LLM 通常做的是 sample 多条 chain 然后 select（不更新权重），或者 prompt adaptation。这篇 paper 更激进：真的在 update policy weights，类似 [TTRL](https://openreview.net/forum?id=VuVhgEiu20) 在 test-time 更新 model 参数。

---

## 为什么 diffusion world model 能可微

Diffusion 生成 sample 的时候，从纯噪声 $z_K$ 开始，一步步去噪到 $z_0$。每一步去噪是个神经网络 forward。如果固定所有随机噪声 $\varepsilon = (z_K, z_{K-1}, \ldots, z_0)$，那从 $(s_t, a_t)$ 到 $s_{t+1}$ 的整条计算路径就是确定性的：线性变换 + 神经网络 eval + 加上固定噪声。

这就是 reparameterization trick 的 multi-step 版本。随机性全塞进 $\varepsilon$ 里，conditioning $(s_t, a_t)$ 是 deterministic input。autograd 直接穿过整个 reverse chain，无需特殊处理。

---

## Theorem 4.1 在说啥

就是个展开的 chain rule。想象一下：你要算 $\nabla_\psi J$，也就是"policy 参数 $\psi$ 变一点，最终 return 变多少"。

$\psi$ 变一点 → 当前 action $\tilde{a}_0$ 变一点 → next state $\tilde{s}_1$ 变一点 → $\tilde{a}_1$ 变一点 → $\tilde{s}_2$ 变一点 → ... 一直传到 $\tilde{s}_H$。

每一步传导有两个 Jacobian：
- Policy Jacobian：state 怎么变 → action 怎么变（$\nabla_s \pi$），参数怎么变 → action 怎么变（$\nabla_\psi \pi$）
- Dynamics Jacobian：state 变 → next state 变（$\nabla_s f$），action 变 → next state 变（$\nabla_a f$）

把这些矩阵连乘 $H$ 步，再乘上 reward 和 terminal critic 的 gradient，就是最终的 $\nabla_\psi J$。本质就是 RNN BPTT，只不过 cell 是 "policy + diffusion transition"。

Theorem 还给了一层细节：diffusion sampler 自身的 Jacobian 也能递推算，从 $A_K=0, B_K=0$ 倒着递推到 $A_0, B_0$。这是因为 reverse diffusion 本身就是 $K$ 步迭代，每步有自己的 partial derivative。

---

## 为什么 work：直觉

### 1. 避开 long-horizon bootstrap

Q-function 的 long-horizon 估计有累积误差。这篇 paper 用 $H$ 步 explicit rollout 把前 $H$ 步 reward 算清楚，只用 critic 估 $H$ 步之后的 tail。$H$ 小，model 误差可控；$H$ 之后，critic 即使不准也只是影响 tail，gradient 信号主要来自前 $H$ 步的显式 reward。

### 2. Test-time state-specific adaptation

Frozen policy 在每个 state 都用同一套参数。但某个特定 state $s_t$ 可能恰好是 policy 不擅长处理的——比如某条腿快摔了。Frozen policy 没办法，就输出它训练时学到的 action。MPCwDWM 会从 $s_t$ 出发想象几步，发现"哦如果我这么走，两步之后就摔了"，于是 gradient 把 policy 往另一个方向推一下。

这就是 inference-time information 的价值：$s_t$ 这一个 specific state 携带了 deployment context，frozen policy 浪费了这个信息。

### 3. Diffusion 捕捉 multi-modal dynamics

Hopper 落地可能左倒也可能右倒，这是 multi-modal。MLP ensemble 学这个会比较挣扎（要靠 ensemble 成员分歧表达 uncertainty）。Diffusion 天然建模分布，sample 一次给一个 mode。

---

## 实验结果的直觉解读

### Gym MuJoCo

halfcheetah-medium-replay 上从 50.21 涨到 59.89（+9.68）是最显著的 gain。medium-replay 数据来自一系列 medium policy 的 replay，state 覆盖广且多 modal。Frozen policy 在广覆盖数据上学个平均 action，但 deployment 遇到的具体 state 可能偏某一 mode，test-time adaptation 能 specialize。

random 数据集上没提升（$\star$ 标记）。因为 random 数据 dynamics 太杂，diffusion model RMSE 高（Figure 2 显示 walker2d-random 的 RMSE 远高于 medium），imagination 出去的 rollout 不可信，gradient 信号被 noise 淹没。这暴露了方法的依赖条件：**world model 必须足够准**。

### AntMaze

AntMaze 是 sparse reward + long-horizon navigation。Critic $Q_\phi$ 在这种任务上极不可靠（reward 几乎全是 0，bootstrap 学不到东西）。但 world model 能在 $H$ 步内 "看见" 蚂蚁朝 goal 移动的 progress，即便 reward 还是 0，state 位置变化也携带信号。gradient through rollout 让 policy 朝 "接近 goal" 的方向调整——这个信号 critic 捕获不到，但 model-based rollout 能。这就是 large-play 从 59 涨到 67 的原因。

---

## 代价和局限

### 计算代价

每个 real environment step 要做 $E \times M \times H \times K$ 次 neural network forward+backward。比 frozen policy forward 慢 3-4 个数量级。实时性要求高的场景（自动驾驶、高频控制）用不了。

### OOD exploitation 隐患

Inference-time gradient ascent 会把 policy 往 imagination 里 reward 高的方向推。如果 world model 在某些 OOD action 区域 extrapolate 错误——比如 model 觉得某个不可能的 action 能飞起来——policy 会真的去推那个 action。没有 uncertainty penalty 的护栏，这是 self-distillation into hallucination 的风险。

MOPO 用 ensemble uncertainty 处理这个；MPCwDWM 暂时没处理。这可能是 random 数据集上不 work 的另一个原因——state space 太广，model extrapolation 风险大。

### Policy drift without reset

Algorithm 1 没明说是否每个 episode 结束 reset $\psi$。如果不 reset，policy 持续 drift 可能远离 behavior 分布，越 drift，model 越不可信，gradient 越不可靠，进入恶性循环。需要 trust region 约束或 episode-level reset。

---

## 跟其他工作的关系

| 方法 | World Model 用在何时 | Policy 参数是否在 inference 更新 |
|---|---|---|
| MOPO / MOReL / COMBO | Training | 否 |
| Diffuser / Decision Diffuser | Inference (sampling) | 否（sample trajectory 不改参数）|
| Dreamer | Training (online) | 是（但 online 数据）|
| **MPCwDWM** | **Inference (gradient)** | **是** |

MPCwDWM 的独特位置：offline 训练 + inference-time gradient-based policy adaptation on learned differentiable world model。

跟 Dreamer 最像，但 Dreamer 是 online（持续收集数据），MPCwDWM 是 strictly offline（数据固定，只在 learned model 里 adapt）。跟 Diffuser 都用 diffusion，但 Diffuser 是 trajectory-level sampling，MPCwDWM 是 step-level transition + policy unroll + gradient。

---

## 我的 overall intuition

这篇 paper 把 offline RL 从"训练完就完事"升级为"训练完只是起点，部署时用 world model 当模拟器，在模拟器里 gradient ascend policy"。这个 mental model 跟 differentiable MPC 一样——只不过 dynamics 是 learned diffusion 而非 analytic physics。

它 work 的根本原因是：offline RL 的 bottleneck 是 long-horizon Q 估计不准，而 short-horizon model 估计相对准。把 long-horizon credit assignment 拆成 "short-horizon explicit rollout + terminal critic bootstrap"，gradient 信号质量提升。

它不 work 的场景是 model 不准的时候——random 数据、OOD action 区域。没有 uncertainty penalty 的护栏，这是个 fundamental risk。

未来的方向大概率是：1）更快的世界模型（flow matching、multi-step diffusion），2）uncertainty-aware imagination，3）trust region 约束 $\psi$ drift，4）latent space imagination 降低 backprop 成本。这篇 paper 更像是这个 paradigm 的 proof-of-concept，证明 "inference-time gradient on differentiable world model" 这个方向 viable，工程优化空间还很大。

参考链接：
- [本论文](https://arxiv.org/abs/2411.13765)
- [ReBRAC](https://arxiv.org/abs/2305.09836)
- [Diffuser](https://arxiv.org/abs/2205.09991)
- [Dreamer V3](https://arxiv.org/abs/2301.04104)
- [Differentiable MPC](https://arxiv.org/abs/2010.12429)
- [TTRL](https://openreview.net/forum?id=VuVhgEiu20)
- [MBPO](https://arxiv.org/abs/1906.08253)
- [MOPO](https://arxiv.org/abs/2005.13239)

---

# Model Predictive Control with Differentiable World Models for Offline RL — 深度解析

## 1. 论文核心 Intuition: Inference-Time Policy Adaptation via Differentiable Rollouts

这篇 paper 的核心 motivation 来自一个类比：LLM 在 test-time 通过 self-consistency、Tree-of-Thoughts、T1、TTRL 等方法获得显著提升（[s1](https://aclanthology.org/2025.emnlp-main.1025/)、[T1](https://proceedings.mlr.press/v267/hou25e.html)、[TTRL](https://openreview.net/forum?id=VuVhgEiu20)），那么 offline RL 在 deployment 阶段能否也利用额外计算来 refine 一个 frozen policy？

传统 offline RL pipeline 训练阶段把 policy $\pi_\psi$ 和 critic $Q_\phi$ 都学完，inference time 只是 forward 一次 $\pi_\psi(s_t)$ 就出 action。作者 argue 这种做法浪费了 deployment 时遇到的 specific state $s_t$ 携带的信息。

作者的关键 insight 是：transition dynamics $P(s'|s,a)$ 和 reward $r(s,a)$ 是 **local objects**，只依赖当前 state-action pair，与 policy 无关；而 $Q^\star$ 是 long-horizon object，在 offline 数据下估计困难。所以应该 learn local 的 $P$ 和 $r$ 作为 differentiable world model，inference time 用 MPC 风格的 receding horizon 把 $P, r, Q_\phi$ 串起来，对 $\psi$ 做 gradient ascent，再把第一步 action 执行出去。

这把 offline RL 从 "train once, deploy frozen" 变成了 "train offline, deploy with online adaptation within the learned model"，但仍然严格 offline —— 没有 real environment interaction，所有 adaptation 都发生在 imagined rollouts 里。

参考链接：
- [Levine et al. Offline RL Tutorial](https://arxiv.org/abs/2005.01643)
- [ReBRAC (Tarasov et al. 2023)](https://arxiv.org/abs/2305.09836)
- [T1 reasoning RL](https://proceedings.mlr.press/v267/hou25e.html)

---

## 2. Architecture: Differentiable World Model (DWM) Pipeline

整体架构 Figure 1 描述的是一个三模块可微 pipeline：

1. **Differentiable Diffusion Sampler** $f_\theta(s_t, a_t, \varepsilon_t) \to s_{t+1}$：用 conditional diffusion model 学习 $p_\theta(s_{t+1}|s_t, a_t)$，并且 reverse process 的整个 chain 写成 deterministic computation graph（固定 noise $\varepsilon$ 后），从而对 $(s_t, a_t)$ 可微。
2. **Reward Model** $r_\xi(s,a) \to \mathbb{R}$：MLP 回归器，提供 stage reward。
3. **Terminal Critic** $Q_\phi(s,a)$：用 ReBRAC 预训练的 critic，作为 truncated rollout 的 terminal value。

policy $\pi_\psi$ 本身也来自 ReBRAC pre-training。所以训练阶段相当于复用 ReBRAC + 训一个 diffusion dynamics + 训一个 reward head，inference 阶段才真正用上 MPC。

### 与现有 World Model 方法的区别

论文特别强调对比（这点很关键，build intuition 用）：
- **MOPO / MOReL / COMBO** ([MOPO](https://arxiv.org/abs/2005.13239), [MOReL](https://arxiv.org/abs/2005.05951), [COMBO](https://arxiv.org/abs/2102.08363))：world model 用于 training time 生成 synthetic rollouts 做 policy/value learning，不用于 inference adaptation。
- **Diffuser / Decision Diffuser / IDQL** ([Diffuser](https://arxiv.org/abs/2205.09991))：inference time 通过 sampling/guidance 生成 candidate trajectory plan，pick first action，policy 参数本身不动。
- **本文 MPCwDWM**：inference time 对 $\psi$ 求梯度，policy 参数真正被 update。这是关键区别。

类比 differentiable MPC 经典工作 ([Amos et al. Differentiable MPC](https://arxiv.org/abs/2010.12429)) 和 differentiable simulator ([Brax](https://github.com/google/brax), [MuJoCo MJX](https://github.com/google-deepmind/mujoco_mjx))，但这里 differentiable simulator 是 learned diffusion 而非 analytic physics。

---

## 3. 数学细节：Diffusion as Differentiable Transition

### 3.1 Forward / Reverse Process

给定 transition tuple $(s_t, a_t, s_{t+1})$，对 $s_{t+1}$ 定义 forward noising chain。这里上标 $(k)$ 是 diffusion timestep（不是 RL timestep），$k \in \{0, 1, \ldots, K\}$，$K$ 是 diffusion horizon。

$$s_{t+1}^{(k)} = \sqrt{\alpha_k}\, s_{t+1}^{(k-1)} + \sqrt{1-\alpha_k}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

变量解释：
- $\alpha_k \in (0,1)$：variance schedule 控制的保留系数
- $\bar{\alpha}_k = \prod_{j=1}^k \alpha_j$：累积保留率
- $s_{t+1}^{(0)} = s_{t+1}$：clean next state
- $s_{t+1}^{(K)} \approx \mathcal{N}(0,I)$：纯噪声

Conditional reverse process 给定 $c_t = (s_t, a_t)$，从 $s_{t+1}^{(K)} \sim \mathcal{N}(0,I)$ 开始反向去噪：

$$s_{t+1}^{(k-1)} = \frac{1}{\sqrt{\alpha_k}}\Big(s_{t+1}^{(k)} - (1-\alpha_k)\hat{\epsilon}_\theta(s_{t+1}^{(k)}, k, c_t)\Big) + \sigma_k z_{k-1}$$

其中 $\hat{\epsilon}_\theta$ 是 noise predictor（U-Net 或 MLP），$z_{k-1} \sim \mathcal{N}(0,I)$ 是 reverse stochasticity。

### 3.2 关键设计：固定 noise 后是 deterministic computation graph

把所有 Gaussian 随机变量打包成 $\varepsilon_t := (z_K, z_{K-1}, \ldots, z_0)$。一旦 $\varepsilon_t$ 固定，映射
$$(s_t, a_t) \mapsto s_{t+1}^{(0)} = f_\theta(s_t, a_t, \varepsilon_t)$$
就完全是 linear ops + neural net evals 的复合，对 $(s_t, a_t)$ 处处可微。这就是 reparameterization trick 在 multi-step diffusion 上的应用，让 backprop 能穿过整个 reverse chain。

定义 reverse-step map：
$$h_k(u, s, a, z) := \frac{1}{\sqrt{\alpha_k}}\big(u - (1-\alpha_k)\hat{\epsilon}_\theta(u, k, (s,a))\big) + \sigma_k z$$

则 $s^{(k-1)} = h_k(s^{(k)}, s, a, z_{k-1})$，$s^{(K)} = g_K(z_K) = z_K$。

### 3.3 训练目标

标准的 conditional noise-prediction MSE：

$$\mathcal{L}(\theta) = \mathbb{E}_{(s_t,a_t,s_{t+1})\sim\mathcal{D},\, k\sim\text{Unif},\, \epsilon\sim\mathcal{N}(0,I)}\Big[\|\epsilon - \hat{\epsilon}_\theta(s_{t+1}^{(k)}, k, c_t)\|_2^2\Big]$$

跟 [Ho et al. DDPM](https://arxiv.org/abs/2006.11239) 一样，只是多了 conditioning $(s_t, a_t)$。

---

## 4. MPC Receding-Horizon Objective

### 4.1 Imagined Rollout

固定一个 noise sequence $\varepsilon_{t:t+H-1} = (\varepsilon_t, \ldots, \varepsilon_{t+H-1})$，从 real state $s_t$ 出发 rollout：

$$\tilde{s}_0 = s_t,\quad \tilde{a}_j = \pi_\psi(\tilde{s}_j),\quad \tilde{s}_{j+1} = f_\theta(\tilde{s}_j, \tilde{a}_j, \varepsilon_{t+j}),\quad j=0,\ldots,H-1$$

这里 $H$ 是 planning horizon，$j$ 是 imagined step index。注意 $\tilde{a}_j$ 完全由 $\pi_\psi$ 决定 —— 这点跟 Diffuser 不同（Diffuser 直接 denoise 整个 trajectory）。

### 4.2 Finite-Horizon Surrogate Return

$$L(\psi; \varepsilon_{t:t+H-1}) = \sum_{j=0}^{H-1} \gamma^j r_\xi(\tilde{s}_j, \tilde{a}_j) + \gamma^H Q_\phi\big(\tilde{s}_H, \pi_\psi(\tilde{s}_H)\big)$$

变量解释：
- $\gamma \in (0,1)$：discount factor
- $r_\xi(\tilde{s}_j, \tilde{a}_j)$：stage reward from learned reward model
- $\gamma^H Q_\phi(\ldots)$：terminal value，用预训练 critic 估算 $H$ 步之后的 long-horizon return

这种 "short-horizon model-based rollout + terminal critic bootstrapping" 的结构在 [MBPO](https://arxiv.org/abs/1906.08253)、[Dreamer](https://arxiv.org/abs/1912.01603) 里都有，但本文把 rollout 完全 differentiable 化、用于 inference-time policy gradient。

### 4.3 Monte Carlo 估计 + 梯度上升

采样 $M$ 个独立 noise sequences，对 $L$ 做 MC 平均：

$$\hat{J}_t(\psi) = \frac{1}{M}\sum_{m=1}^M L(\psi; \varepsilon_{t:t+H-1}^{(m)})$$

内层 loop 做 $E$ 步梯度上升：
$$\psi \leftarrow \psi + \alpha \nabla_\psi \hat{J}_t(\psi), \quad e=1,\ldots,E$$

然后执行 $a_t = \pi_\psi(s_t)$ 到 real environment，observe $(r_t, s_{t+1})$，进入下一个 MPC cycle。整个流程见 Algorithm 1。

---

## 5. Theorem 4.1: Gradient Recursion 的精确形式

这个 theorem 是 paper 的技术核心，本质是 backprop through time 的 explicit Jacobian 形式。

### 5.1 Rollout Sensitivity 定义

定义 "policy Jacobian" 和 "world-model Jacobian"：

$$\Pi_s(j) := \nabla_s \pi_\psi(s)\big|_{s=\tilde{s}_j},\quad \Pi_\psi(j) := \nabla_\psi \pi_\psi(\tilde{s}_j)$$
$$F_s(j) := \nabla_s f_\theta(s,a,\varepsilon_{t+j})\big|_{s=\tilde{s}_j, a=\tilde{a}_j},\quad F_a(j) := \nabla_a f_\theta(s,a,\varepsilon_{t+j})\big|_{s=\tilde{s}_j, a=\tilde{a}_j}$$

变量含义：
- $\Pi_s(j) \in \mathbb{R}^{|\mathcal{A}|\times|\mathcal{S}|}$：policy 对 state 输入的 Jacobian
- $\Pi_\psi(j) \in \mathbb{R}^{|\mathcal{A}|\times|\psi|}$：policy 对参数的 Jacobian
- $F_s(j), F_a(j)$：diffusion transition sampler 对 state / action 的 Jacobian

定义状态/动作对参数的 sensitivity：
$$G_j := \nabla_\psi \tilde{s}_j \in \mathbb{R}^{|\mathcal{S}|\times|\psi|},\quad D_j := \nabla_\psi \tilde{a}_j \in \mathbb{R}^{|\mathcal{A}|\times|\psi|}$$

### 5.2 Sensitivity Recursion

初始条件 $G_0 = 0$（因为 $\tilde{s}_0 = s_t$ 与 $\psi$ 无关）。然后：

$$D_j = \Pi_s(j) G_j + \Pi_\psi(j)$$
$$G_{j+1} = F_s(j) G_j + F_a(j) D_j$$

直觉解读：
- $D_j$ 描述 "$\psi$ 扰动如何传导到 $\tilde{a}_j$"，分为两部分：(a) 通过 $\tilde{s}_j$ 间接传导（$\Pi_s(j) G_j$），(b) 直接通过 policy 参数（$\Pi_\psi(j)$）。
- $G_{j+1}$ 描述 "$\psi$ 扰动如何传导到 $\tilde{s}_{j+1}$"，通过 dynamics 的 state Jacobian $F_s$ 和 action Jacobian $F_a$ 复合。

### 5.3 最终 Gradient Expression

定义 reward / critic 的 gradient：
$$r_s(j) := \nabla_s r_\xi(\tilde{s}_j, \tilde{a}_j),\quad r_a(j) := \nabla_a r_\xi(\tilde{s}_j, \tilde{a}_j)$$
$$Q_s := \nabla_s Q_\phi(\tilde{s}_H, \tilde{a}_H),\quad Q_a := \nabla_a Q_\phi(\tilde{s}_H, \tilde{a}_H)$$

则：

$$\nabla_\psi L(\psi; \varepsilon) = \sum_{j=0}^{H-1} \gamma^j \big(r_s(j) G_j + r_a(j) D_j\big) + \gamma^H (Q_s G_H + Q_a D_H)$$

### 5.4 Diffusion Jacobian Recursion

Theorem 还给出 diffusion sampler 自身 Jacobian 的递推：

$$\nabla_a f_\theta(s,a,\varepsilon) = A_0,\quad \nabla_s f_\theta(s,a,\varepsilon) = B_0$$

with $A_K = 0, B_K = 0$ and for $k=K,\ldots,1$:
$$A_{k-1} = \frac{\partial h_k}{\partial u} A_k + \frac{\partial h_k}{\partial a}$$
$$B_{k-1} = \frac{\partial h_k}{\partial u} B_k + \frac{\partial h_k}{\partial s}$$

这是 RNN-style BPTT 的展开形式 —— 每个 reverse step 都贡献一个 $\frac{\partial h_k}{\partial u}$ 项，加上 conditioning 的直接 partial $\frac{\partial h_k}{\partial a}$ / $\frac{\partial h_k}{\partial s}$。在 autograd 框架里这一切会被自动计算，但 explicit 形式对分析 gradient flow、exploding/vanishing gradient、以及设计 truncated reverse diffusion 都有用。

---

## 6. 实验数据表深度解读

### 6.1 Table 1: Gym-MuJoCo Locomotion

MPCwDWM 在 18 个数据集上平均 85.33，相比：
- TD3+BC: 70.3
- IQL: 72.9
- CQL: 73.6
- SAC-RND: 82.6
- ReBRAC: 81.53

最显著的 gain 在：
- halfcheetah-medium-replay: 50.21 → **59.89**（+9.68，巨大提升）
- hopper-medium-replay: 95.82 → **103.14**
- halfcheetah-medium: 65.87 → **70.05**

⚠️ 注意几个 $\star$ 标记：hopper-random 和 walker2d-random 上 inference-time optimization 没带来提升，直接 fallback 到 ReBRAC。这说明 random 数据上学到的 dynamics model 误差过大，imagination 出去的 rollout 不可靠，gradient 信号被 noise 淹没。这跟 Figure 2 中 walker2d 的 RMSE 远高于 halfcheetah 一致。

### 6.2 Table 2: vs Generative Model Baselines

对比 DT、TT、MOPO、MOReL、MBOP、Diffuser：
- MPCwDWM 平均 94.4，第二名 Diffuser 77.5
- halfcheetah-medium-replay: Diffuser 37.7 → MPCwDWM **59.89**
- walker2d-medium-replay: Diffuser 70.6 → MPCwDWM 81.87

注意这里 DT/TT/Diffuser 是 return-conditioned 或 trajectory-level 生成；MPCwDWM 是 policy-based + gradient adaptation。两者的根本差异：MPCwDWM 把 test-time state $s_t$ 作为 conditioning 输入做针对性 optimization，而 Diffuser 是 unconditional sampling + reward guidance。

### 6.3 Table 3: AntMaze

AntMaze 是 sparse reward + long-horizon navigation 任务，非常考验 long-horizon credit assignment：
- ReBRAC average: 77.62
- MPCwDWM average: **85.07**

最大提升在 large-play (59→67) 和 large-diverse (51→66)。AntMaze 之所以受益于 inference-time adaptation，是因为 reward sparse 时 critic $Q_\phi$ 很不可靠，而 world model 能在 $H$ 步内 "看见" goal-approaching 的 progress（即便 reward 还是 0），gradient through rollout 让 policy 朝着 "接近 goal" 的方向调整 —— 这个信号在 critic 里是 capture 不到的。

### 6.4 Table 4 / Table 5: Model RMSE

- Diffusion one-step RMSE：halfcheetah 系列普遍 < 0.6；walker2d-expert 高达 3.34（最大）；walker2d-medium 1.41
- Reward RMSE：整体小很多，halfcheetah 0.05–0.28，walker2d-expert 0.49

直觉：dynamics model 在 expert 数据上误差最大（因为 expert state distribution 远离其他 dataset，且 motion 高速复杂）；reward model 因为 reward function 本身简单（forward velocity），所以容易学。

---

## 7. Intuition Building: 为什么这个方法 Work

### 7.1 Critic 不准 → 用 short-horizon model 替代 long-horizon bootstrap

Offline RL 的 fundamental problem 是 $Q_\phi$ 必须近似 $Q^{\pi^\star}$，但数据来自 behavior $\mu$，且 long-horizon 累积误差大。MPCwDWM 把 credit assignment 分成两段：
- 前 $H$ 步：用 $f_\theta + r_\xi$ 显式 rollout，gradient 直接穿过
- $H$ 步之后：用 $Q_\phi$ 做 terminal value

只要 $H$ 足够小让 model 误差可控，就能避开 critic 的 long-horizon 不准。这跟 [MBPO](https://arxiv.org/abs/1906.08253) 的 $k$-step backup 思想同源，但 MBPO 是 training-time，MPCwDWM 是 inference-time。

### 7.2 Diffusion World Model 比 MLP Ensemble 更好

为什么用 diffusion 而不是 deterministic MLP 或 ensemble（MOPO/MOReL）？
- **Multi-modality**：$P(s'|s,a)$ 在很多控制任务下是 multi-modal 的（比如 Hopper 落地有左/右两种姿态），deterministic MLP 只能学均值，扩散能捕捉分布。
- **Differentiability**：ensemble 需要 expectation 后才能微分，diffusion 用 reparameterized sampler 天然可微。
- **Gradient signal richness**：通过 reverse chain 的 multi-step Jacobian，policy gradient 能 "看到" dynamics 的局部曲率，相当于 implicit curriculum。

### 7.3 与 Differentiable MPC / Differentiable Simulator 的关系

经典 differentiable MPC ([Amos et al.](https://arxiv.org/abs/2010.12429)) 把 QP solver 展开 backprop，需要 analytic dynamics。MPCwDWM 用 learned diffusion dynamics 替换 analytic dynamics，相当于把 differentiable MPC 的思想推广到 model-free learned simulator 上。这与 [DiffSim](https://arxiv.org/abs/2104.01932)、[PlasticineLab](https://github.com/redmond342/PlasticineLab) 一脉相承，但 simulator 是 generative model。

### 7.4 与 LLM Test-Time Scaling 的类比

paper 反复强调 inference-time computation 的类比。这种类比成立的关键：
- LLM test-time scaling：sample 多条 chain-of-thought，select best (self-consistency) 或 search (ToT)
- MPCwDWM：sample 多条 imagined rollout ($M$ particles)，用 gradient ascent 在 $\psi$ 上聚合信号

但有一个重要区别：LLM 的 test-time scaling 通常是 **discrete selection** 或 **prompt adaptation**，不更新 model weights；MPCwDWM 真的在 update policy weights $\psi$。更接近的类比是 [TTRL](https://openreview.net/forum?id=VuVhgEiu20) —— test-time RL 真的更新 model 参数。

### 7.5 Computational Cost & Practical Concerns

每个 real environment step 需要：
- $M$ 个 rollout，每个 $H$ 步
- 每步 1 次 policy forward + 1 次 diffusion reverse ($K$ 步 neural net eval)
- $E$ 次 gradient backprop through $H \times K$ 个 neural net evaluations

总计算量大约 $O(E \cdot M \cdot H \cdot K)$ neural net forwards/backwards。如果 $H=5, K=20, M=8, E=3$，每个 real step 大约 2400 次 NN eval —— 比 frozen policy forward 慢 3–4 个数量级。

paper 的 future work 提到用 multi-step diffusion ([Diffusion World Model, Ding et al.](https://arxiv.org/abs/2402.03570)) 一次预测多步 future，减少 $K$；以及 flow matching ([Lipman et al.](https://arxiv.org/abs/2210.02747)) / [Flow Q-Learning](https://proceedings.mlr.press/v267/) 替代 reverse diffusion 减少 sampling step。这是明显的下一步方向。

### 7.6 Distribution Shift 仍然存在的隐患

虽然 paper 说 strictly offline，但 inference-time gradient ascent on $\psi$ 会把 policy 推向 imagination 中 reward 高的方向 —— 如果 world model 在某些 OOD action 区域 extrapolate 错误，policy 可能被推向这些 "imagined high reward" 的 OOD 区域，相当于 self-distillation into a hallucination。

MOPO 用 ensemble uncertainty penalty 处理这个问题；MPCwDWM 没有显式 uncertainty penalty。这是个潜在隐患，可能解释了为什么 random dataset 上没有提升 —— random 数据的 state space coverage 太广，model extrapolation 风险大。

可能的改进方向：
- 用 Bayesian diffusion / ensemble diffusion 提供 uncertainty
- 用 $D(\pi_\psi \| \mu)$ 行为正则化在 inference time 也施加
- Truncate gradient when model-implied next state is far from dataset

---

## 8. 相关联想：与 Dreamer / MuZero / Trajectory Transformer 的关系

### 8.1 Dreamer 系列 ([Dreamer V3](https://arxiv.org/abs/2301.04104))

Dreamer 也是 "learn world model + imagine rollouts + policy gradient through imagination"，但 Dreamer 是 online，且 world model 是 RSSM (recurrent state-space model) 而非 diffusion。MPCwDWM 可以看作 offline 版的 Dreamer，且用 diffusion 替代 RSSM 来 capture multi-modal dynamics。这个对比作者没明确做，但其实很 illuminating。

### 8.2 MuZero / AlphaZero ([MuZero](https://arxiv.org/abs/1911.08265))

MuZero 学习 latent dynamics $(s, a) \to (r, s')$ 全部在 latent space，然后用 MCTS planning。MPCwDWM 在 raw state space 学 dynamics，用 gradient-based MPC 而非 search。Two different planning paradigms on learned models.

### 8.3 Trajectory Transformer ([TT](https://arxiv.org/abs/2106.02039))

TT 学 trajectory 的 joint distribution，beam search decoding high-reward trajectory。本质是 model-based planning with sequence model。MPCwDWM 用 policy + gradient 而非 decoding；TT 是 sampling-based，MPCwDWM 是 optimization-based。

### 8.4 Decision Diffuser / Diffuser

Diffuser 学 trajectory-level diffusion $p(\tau)$，inference 用 reward guidance conditioning 采样。MPCwDWM 是 step-by-step transition diffusion + policy unrolled，两个粒度完全不同。Diffuser 的 advantage 是 global trajectory coherence；MPCwDWM 的 advantage 是可以 adapt policy parameters to current $s_t$。

### 8.5 Differentiable Policy Optimization through Black-box Simulators

[jax-MD](https://github.com/jax-md/jax-md)、[Brax](https://github.com/google/brax)、[MuJoCo MJX](https://github.com/google-deepmind/mujoco_mjx) 都提供 differentiable physics。MPCwDWM 用 learned diffusion 替代 analytic physics —— 在没有 analytic simulator 的真实世界任务（healthcare、recommendation）里这是必需的。这正是 paper motivation 里 healthcare、autonomous driving 等 application 的潜在价值。

---

## 9. Implementation 细节推断

基于 Algorithm 1 和 Theorem 4.1 的形式，可以推断一些实现细节（论文未完全明示）：

1. **Reparameterization 一致性**：同一个 $\varepsilon_{t:t+H-1}$ 必须在 forward rollout 和 backward pass 中复用 —— 这是 differentiable rollout 的标准做法。
2. **Gradient clipping 必需**：Theorem 4.1 的 Jacobian 乘积 $F_s G_j + F_a D_j$ 在 $H$ 较大时会 exploding/vanishing，类似 RNN BPTT。Paper 可能用了 gradient norm clipping 或者 $H$ 比较小（推测 $H \in [3, 10]$）。
3. **Critic 用于 terminal value 但不更新**：$\phi$ 在 inference time frozen，只更新 $\psi$。否则 gradient 会通过 $Q_\phi$ 流到 critic 参数引起 instability。
4. **Particle averaging**：$M$ 个 particles 的 gradient $\frac{1}{M}\sum_m \nabla_\psi L^{(m)}$ 是个 stochastic gradient estimator，方差跟 $M$ 成反比。
5. **Reset $\psi$ between episodes?**：Algorithm 1 没明说，但如果不 reset，policy 会持续 drift，可能远离 behavior distribution。这是潜在的 deployment instability。

---

## 10. 局限与未来方向（paper 自承 + 我的补充）

Paper future work 提到：
- Multi-step diffusion world model 减少 $K$
- Flow matching / mean flow 替代 reverse diffusion
- Decoupled variants 避免 BPTT
- Offline-to-online 加 periodic world model update

我补充几个值得探索方向：
- **Inference-time uncertainty penalty**：用 ensemble diffusion 或 Bayesian last layer 给 imagined rollout 加 penalty，缓解 OOD exploitation
- **Adaptive horizon $H$**：根据 model local error率自适应选择 $H$（小 error 用大 $H$，大 error 用小 $H$）
- **Trust-region constraint on $\psi$ update**：限制 inference-time $\psi$ drift，避免 catastrophic forgetting of pre-trained behavior
- **Test-time policy distillation**：把 inference-time adapted $\psi$ 蒸馏回 base policy，类似 [T1 reasoning distillation](https://proceedings.mlr.press/v267/hou25e.html)
- **Latent-space imagination**：在 latent space 做 diffusion rollout（减少 dimension，加速 backprop），类似 [Dreamer V3](https://arxiv.org/abs/2301.04104) 的 latent imagination + diffusion dynamics

---

## 11. 总结：One-Sentence Intuition

**Offline RL pre-trains a policy and a differentiable world model; at deployment, treat the world model as a differentiable simulator, roll out imagined trajectories from the current real state, and backpropagate finite-horizon returns to nudge the policy parameters toward locally better actions before executing.**

这就把 offline RL 从 "static deployment" 升级为 "model-based test-time policy adaptation"，跟 LLM test-time scaling 的精神一致，但用 gradient-based optimization 而非 sampling/search。

参考链接汇总：
- [本论文 arXiv 版本（推测）](https://arxiv.org/abs/2411.13765)
- [ReBRAC](https://arxiv.org/abs/2305.09836)
- [Diffuser (Janner et al.)](https://arxiv.org/abs/2205.09991)
- [DDPM (Ho et al.)](https://arxiv.org/abs/2006.11239)
- [Differentiable MPC (Amos et al.)](https://arxiv.org/abs/2010.12429)
- [T1 Inference RL](https://proceedings.mlr.press/v267/hou25e.html)
- [TTRL](https://openreview.net/forum?id=VuVhgEiu20)
- [s1 test-time scaling](https://aclanthology.org/2025.emnlp-main.1025/)
- [Dreamer V3](https://arxiv.org/abs/2301.04104)
- [Flow Q-Learning](https://proceedings.mlr.press/v267/)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Diffusion World Model (Ding et al.)](https://arxiv.org/abs/2402.03570)
- [D4RL Benchmark](https://arxiv.org/abs/2004.06139)
- [MBPO](https://arxiv.org/abs/1906.08253)
- [MOPO](https://arxiv.org/abs/2005.13239)
- [MOReL](https://arxiv.org/abs/2005.05951)
- [Offline RL Tutorial (Levine et al.)](https://arxiv.org/abs/2005.01643)
- [CQL](https://arxiv.org/abs/2006.04779)
- [IQL](https://arxiv.org/abs/2110.03152)
- [Decision Transformer](https://arxiv.org/abs/2106.01345)
- [Trajectory Transformer](https://arxiv.org/abs/2106.02039)

如果你想要更深入到某一个方面（比如 Theorem 4.1 的 Jacobian 推导细节、specific dataset 上的失败模式分析、或跟 Dreamer 的详细对比），可以告诉我，我可以再展开。
