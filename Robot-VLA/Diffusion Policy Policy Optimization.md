---
source_pdf: Diffusion Policy Policy Optimization.pdf
paper_sha256: 4b81907b0a8ea7118e4922720f2e7d269d5cf00d3725e44fc6d25f49d02413f3
processed_at: '2026-08-03T21:48:31-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DPPO 人话版：给 Karpathy 的 Intuition Tour

## 一句话版本

**Diffusion Policy 的 denoising 过程本质就是个 K 步的 sequential decision making，每步都是 tractable 的 Gaussian，所以可以直接套 PPO。之前大家觉得这条路走不通，但这篇 paper 说"其实可以，而且效果还挺好"。**

## 为什么这件事不显然？

先回忆一下背景。Diffusion Policy（Chi et al., https://diffusion-policy.cs.columbia.edu/）现在已经是 robot learning 的主流 parameterization 之一。你给它一个 state $s$，它从纯噪声 $a^K \sim \mathcal{N}(0, I)$ 出发，跑 K 步 denoising（典型 K=100），最后吐出一个 action chunk $a^0$。

问题来了：如果你想用 RL 来 fine-tune 这个 policy，你需要算 $\log \pi_\theta(a^0|s)$ 的 gradient。但 $a^0$ 是 K 个 Gaussian 经过非线性变换的复合产物，这个 marginal likelihood 是 intractable 的。

所以 community 的主流思路是：
- 用 Q-learning（IDQL, DQL）—— 学一个 critic，然后让 actor 去拟合 critic 认为好的 action
- 用 weighted regression（AWR, RWR）—— 用 critic 给 sample 打权重，做 weighted BC
- 用 guidance（classifier guidance, https://arxiv.org/abs/2205.09991）—— 不改 policy，改 sampling

Policy gradient 这条路基本被放弃了。Psenka et al.（QSM paper, https://arxiv.org/abs/2312.11752）甚至明确 conjecture 说 PG 对 diffusion policy 会 inefficient，因为 effective horizon 太长（K 步 denoising × T 步 environment），action variance 会爆炸。

## DPPO 的 "Aha Moment"

但这里有个被忽略的事实：**虽然 $p(a^0|s)$ 不可算，但每一步 $p(a^{k-1}|a^k, s)$ 是个 explicit Gaussian**。

$$a^{k-1} \sim \mathcal{N}(\mu_k(a^k, \varepsilon_\theta(a^k, k, s)), \sigma_k^2 I)$$

这就是个标准的 stochastic policy！mean 由 network 预测的 noise $\varepsilon_\theta$ 决定，variance 由 noise schedule $\sigma_k^2$ 决定。

所以如果你把**每一步 denoising 当作 MDP 的一步**，每步都有 explicit likelihood，policy gradient 就可以直接用了。这不是什么新 idea——Black et al. 在 DDPO（https://arxiv.org/abs/2305.13301）里已经用这个 trick 做 text-to-image 的 RL fine-tuning 了。

DPPO 的贡献是：把这个 trick 搬到 robot learning，把它嵌进 environment MDP 形成 two-layer MDP，然后加一堆 engineering best practices 让它 actually work well on challenging tasks。

## Two-Layer MDP 的直觉

想象你在下棋（environment MDP），每走一步棋之前，你要先在脑子里"想"K 步（denoising MDP）。

- 外层 MDP：robot 跟 environment 交互，state 是 $s_t$，action 是 $a_t^0$（最终执行的 action），reward 是 $R(s_t, a_t^0)$
- 内层 MDP：从噪声 $a_t^K$ 开始，K 步 denoising 到 $a_t^0$，state 是 $(s_t, a_t^{k+1})$，action 是 $a_t^k$，reward 全是 0（除了 $k=0$ 那步拿 environment reward）

这个嵌套结构让 reward 信号能通过 chain rule 流回每一步 denoising。每一步 denoising 的 PG update 是：

$$\nabla_\theta \log \mathcal{N}(a_t^k; \mu(\cdot), \sigma_k^2 I) \times \text{advantage}$$

advantage 用 GAE 估计，但有个 trick：value function 只依赖 environment state $s_t$，不依赖 noisy action $a_t^{k+1}$。因为高 k 时 $a_t^{k+1}$ 几乎是纯噪声，estimate 它的 value 是 high variance 的。

## 为什么这个能 work 得这么好？

Paper 里有三个 mechanism 的分析，我觉得都挺 intuitive 的：

### 1. Structured On-Manifold Exploration

Gaussian policy 的 exploration 是在最终 action 上加 unstructured noise。Diffusion 的 exploration 不一样——你在 K 步 denoising 的每一步都加 noise，但每步加完 noise 后 network 会把 action "拉回" expert data manifold。

这就像你探索一个新城市：Gaussian 是 random walk（随机乱走），diffusion 是"每次走偏了都拉回主干道再继续探索"。所以 diffusion 的 exploration 是 structured 的——覆盖范围广但不离谱。

Paper 用 D3IL Avoid environment 可视化了这点（Figure 10）。Pre-training data 有两个 mode（两条路径绕过障碍物）。DPPO 的 exploration 在两个 mode 周围有 wide coverage，Gaussian 的 exploration 乱七八糟，GMM 的 exploration 太窄。

这个 property 对 fine-tuning 特别有利——pre-trained policy 已经 cover 了 expert data 的 manifold，DPPO 在这个 manifold 上 explore 就能找到更好的 policy。

### 2. Training Stability from Multi-Step Denoising

Multi-step denoising 是个天然的 regularizer。即使某一步 denoising 的 update 把 action distribution 推偏了，后续步骤会 "pull back" 到 pre-trained distribution。

Paper 做了个很 convincing 的实验（Figure 11）：fine-tuning 过程中逐渐给 action 注入 noise。Gaussian 和 GMM 的 performance 直接 collapse，DPPO with ≥4 denoising steps 完全 robust。

这解释了为什么 DPPO 在 sim-to-real 上表现好——sim 里的 controller 不完美（有 noise），DPPO policy 天然 robust to 这种 noise，Gaussian policy 在 sim 里 work 但 real 里直接挂。

### 3. Multi-Modal Representation

Diffusion policy 能 represent complex multi-modal distribution。Gaussian 只能 unimodal，GMM 能 multi-modal 但 mixture 数有限。Diffusion 理论上可以 represent arbitrary distribution。

这对 robot manipulation 很重要——同一个 state 下可能有多个 valid action（比如抓物体可以从不同角度抓）。Pre-training 时 diffusion 能 capture 这些 modes，fine-tuning 时能保留这种 multi-modality。

## Engineering Best Practices 的直觉

Paper 里有一堆 engineering tricks，每个都有清晰的 intuition：

### 只 fine-tune 最后 K' 步

Pre-training 用 K=100 步，fine-tuning 只调最后 K'=10 步。前面 90 步 frozen。

Intuition：前 90 步是把纯噪声变成 "approximately expert-like" 的 distribution，这部分 pre-training 已经学好。后 10 步是 "refine to optimal for task"，这是 RL 要调的。类似 LoRA——只调 task-specific 的部分。

这也大幅节省 GPU memory 和 compute，因为 backprop 只过 10 步不是 100 步。

### DDIM fine-tuning with η trick

DDIM（https://arxiv.org/abs/2010.02502）能把 100 步压缩到 5 步。但标准 DDIM 是 deterministic（η=0），没法 explore。

DPPO 的 trick：训练时用 η=1（stochastic，有 exploration noise），evaluation 时用 η=0（deterministic，stable deployment）。

这就像你在练习打篮球时尝试各种动作（stochastic），但比赛时用最 proven 的动作（deterministic）。

### Dual noise schedule clipping

两个 minimum noise threshold：
- $\sigma_{min}^{exp} \approx 0.01-0.1$：sampling 时的 exploration noise
- $\sigma_{min}^{prob} \geq 0.1$：计算 log-likelihood 时的 variance

为什么分开？Sampling 时需要足够 noise 来 explore，但如果计算 log-likelihood 时 variance 太小，$\log \mathcal{N}$ 的 gradient 会爆炸（因为 $1/\sigma^2$ 项）。

### 分 denoising step 的 clipping schedule

PPO 的 clipping ratio ε 对不同 denoising step 用不同值：

$$\varepsilon_k = \varepsilon_0 \cdot 0.1^{k/(K-1)}$$

早期 denoising step（高噪声，大 variance）用小 ε，后期（低噪声）用大 ε。

Intuition：早期 step 的 distribution 本身就宽，policy update 容易 overshoot，需要更 conservative 的 clipping。

### Value function 只依赖 s_t

$$\hat{V}(\bar{s}_{\bar{t}(t,0)}) := \tilde{V}(s_t)$$

不 include noisy action $a_t^{k+1}$。因为高 k 时 $a_t^{k+1}$ 是纯噪声，estimate 它的 value 是 high variance。

这跟 standard PPO 不一样——标准 PPO 的 state 就是全部 observable state。但这里 bar-state 包含 noisy action，我们故意 ignore 它。

### Denoising discount γ_DENOISE

Advantage 乘以 $\gamma_{DENOISE}^k$：

$$\hat{A} = \gamma_{DENOISE}^k (\bar{r} - \hat{V})$$

这给早期 denoising step（高 k）的 advantage 打折扣。Intuition：早期 step 对最终 action 的影响被后续 step "过滤"了，所以它们的 credit assignment 应该小一些。

## 实验结果的故事

### vs. 其他 Diffusion RL 方法

在 GYM locomotion tasks 上，IDQL 和 DIPO 挺 competitive。但在 sparse-reward ROBOMIMIC manipulation tasks 上，所有 Q-learning 方法都 exhibit training instability——开始 performance drop 然后无法 recover。

DPPO 的核心优势是 training stability。PPO clipping 防止 policy collapse，on-policy 避免 Q-function misestimation。

### vs. 其他 Policy Parameterizations

在 ROBOMIMIC Square 和 Transport（最难的任务）上，DPPO 大幅超过 Gaussian 和 GMM。Transport 是双臂 14-dim action，DPPO 是第一个 solve 到 >90% success 的 RL method。

### Sim-to-Real

Furniture-Bench One-leg task：
- DPPO：80% real success rate
- Gaussian：88% sim success，但 0% real success
- Gaussian + BC regularization：53% sim，50% real

Gaussian 在 sim 里 work 但 real 里完全 fail。Supplementary video 显示 Gaussian policy "volatile and jittery"。DPPO 的 multi-step denoising 天然产生 smooth behavior，跟物理 controller 的 smoothness 要求 aligned。

## 对未来的联想

### LLM RLHF 的类比

DPPO 跟 LLM 的 RLHF 结构上非常类似：
- DDPO 已经把 diffusion 当 MDP 做 text-to-image RL fine-tuning
- DPPO 把这扩展到 sequential decision making
- 自然延伸：diffusion-based language models（https://arxiv.org/abs/2406.07524）+ DPPO-style fine-tuning for interactive dialogue/planning

如果未来 diffusion-based language models 成熟，DPPO framework 可以直接 apply。

### 跟你的 Diffusion Forcing 的联系

你的 Diffusion Forcing（https://arxiv.org/abs/2407.01392）把 next-token prediction 和 full-sequence diffusion 统一。DPPO 的 two-layer MDP 跟这有 structural similarity——都是把 diffusion process 嵌入到 sequential decision making。

Open question：能否把 Diffusion Forcing 扩展到 RL setting？Full-sequence diffusion + RL fine-tuning for long-horizon planning？这可能是 DPPO 的自然延伸——不只是 fine-tune action policy，还 fine-tune 一个能预测 future states 和 actions 的 world model。

### Multi-Task Pre-training + DPPO

Paper 最后提到 DPPO 很适合 "pre-train on diverse tasks + fine-tune on specific task"。这跟 LLM 的 pre-train + fine-tune paradigm 完全 analogous。

POCO（https://arxiv.org/abs/2402.02511）和 π₀（https://arxiv.org/abs/2410.24164）都是 multi-task diffusion policy pre-training。DPPO + multi-task pre-training 是 scaling up robot learning 的 promising 方向。

想象一下：在一个 massive multi-task dataset 上 pre-train 一个 diffusion policy（像 GPT 一样），然后对每个具体下游任务用 DPPO fine-tune（像 RLHF 一样）。这就是 robot learning 的 "GPT moment"。

### Sample Efficiency 的 Limitation

DPPO 的主要 limitation 是 sample efficiency——on-policy PG 比 off-policy Q-learning 差。从 scratch 训练时 DPPO 比 Gaussian 慢 6× wall-clock。

Potential solutions：
- Importance sampling for off-policy DPPO
- Replay buffer with likelihood-based weighting
- 跟 SERL（https://arxiv.org/abs/2401.16013）的 real-world RL setup 结合

### Exploration 的理论理解

Section 6 的 on-manifold exploration 很 phenomenological。更深层的理论问题：diffusion policy 的 exploration 为什么比 Gaussian policy 更 structured？

这跟 score-based generative models 的 Langevin dynamics interpretation 有关——denoising 过程 approximates Langevin sampling on data manifold。Potential direction：formal analysis of DPPO exploration via score matching + RL theory。

### 跟你的 micrograd / llm.c 风格

从 implementation perspective，DPPO 的 clean 之处在于：
- 每个 denoising step 就是一个 Gaussian sample + log-likelihood
- PPO update 就是 standard clipped objective
- 不需要 Q-function 的 double Q-learning, target network 等复杂 machinery

一个 minimal DPPO implementation 可能只需要 ~300 lines：
- DDPM forward/reverse process（~100 lines）
- PPO clipped objective（~50 lines）
- GAE estimation（~30 lines）
- Two-layer MDP rollout（~50 lines）

这跟你的 llm.c philosophy 一致——simple, clean, understandable implementation。写一个 `dppo.c` 可能是个 fun project。

## 最终的 Intuition

DPPO 的故事本质上是个 "don't overthink it" 的故事。Community 觉得 "diffusion + PG = bad idea because long horizon + high variance"，但没人真正试过。有人试了，加了些合理的 engineering tricks，发现其实 work 得挺好。

这让我想到你之前说过的 "you have to try things"——很多看起来 scary 的 idea，实际试一下可能会 surprise 你。DPPO 就是这样一个 case。

而且 DPPO 的成功不是因为它用了什么 fancy 的 algorithmic innovation，而是因为它 leverage 了 diffusion policy 本身的 structure property：
- Multi-step denoising → natural sequential decision making
- Tractable per-step Gaussian → explicit likelihood for PG
- Projection to data manifold → structured exploration
- Multi-modal representation → robust policy

这些都是 diffusion model 固有的 property，DPPO 只是发现了它们跟 RL fine-tuning 的 synergy。这种 "利用 structure 而不是 fight structure" 的 approach，跟你一直强调的 first-principles thinking 是一致的。

## References

- DPPO: https://diffusion-ppo.github.io
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- PPO: https://arxiv.org/abs/1707.06347
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- DDPO (Black et al.): https://arxiv.org/abs/2305.13301
- IDQL: https://arxiv.org/abs/2304.10573
- DQL: https://arxiv.org/abs/2208.06193
- QSM: https://arxiv.org/abs/2312.11752
- DIPO: https://arxiv.org/abs/2305.13122
- ROBOMIMIC: https://robomimic.github.io/
- Furniture-Bench: https://arxiv.org/abs/2305.12821
- D4RL: https://arxiv.org/abs/2004.07219
- Diffusion Forcing: https://arxiv.org/abs/2407.01392
- POPO/POCO: https://arxiv.org/abs/2402.02511
- π₀: https://arxiv.org/abs/2410.24164
- Cosine noise schedule: https://arxiv.org/abs/2102.09672
- Block et al. generative BC theory: https://arxiv.org/abs/2406.01361
- Permenter & Yuan diffusion as optimization: https://arxiv.org/abs/2306.04848
- D3IL benchmark: https://arxiv.org/abs/2402.14606
- RLPD: https://arxiv.org/abs/2306.00783
- Cal-QL: https://arxiv.org/abs/2304.12972
- IBRL: https://arxiv.org/abs/2311.02198
- SERL: https://arxiv.org/abs/2401.16013
- Diffusion Forcing (your work): https://arxiv.org/abs/2407.01392
- Masked diffusion language models: https://arxiv.org/abs/2406.07524

---

# DPPO: Diffusion Policy Policy Optimization 深度解析

## 1. 高层 Intuition: 为什么要把 Diffusion 当作 MDP?

这篇paper的核心insight非常elegant。Diffusion Policy本身是一个采样过程：从纯噪声 $a^K \sim \mathcal{N}(0, I)$ 出发，经过K步denoising得到最终action $a^0$。传统观点认为diffusion policy的likelihood $p(a^0|s)$ 是不可tractable的（因为它是K个Gaussian transition的乘积，经过复杂的非线性变换）。

但是——key insight——**每一步denoising本身就是一个tractable的Gaussian**：

$$p_\theta(a^{k-1}|a^k, s) = \mathcal{N}(a^{k-1}; \mu_k(a^k, \varepsilon_\theta(a^k, k, s)), \sigma_k^2 I)$$

这里：
- $a^k$ 是第k步的noisy action（$k$ 从K降到0，K是total denoising steps）
- $\varepsilon_\theta(a^k, k, s)$ 是neural network预测的noise
- $\mu_k(\cdot)$ 是固定的mapping function，把 $(a^k, \varepsilon_\theta)$ 映射到next mean
- $\sigma_k^2$ 是固定的variance schedule

所以如果我们把**每一步denoising当作MDP的一步**，那每一步都有explicit的Gaussian likelihood，policy gradient可以直接应用！这就是Black et al.在DDPO (https://arxiv.org/abs/2305.13301) 中提出用于text-to-image的核心idea。

DPPO的贡献是把这扩展到**two-layer MDP**——外层是environment MDP（robot interacting with physics），内层是denoising MDP。这个嵌套结构让reward信号能同时流过environment dynamics和denoising chain。

## 2. Two-Layer Diffusion Policy MDP 的 Formal Construction

### 2.1 Indexing Trick

最tricky的部分是indexing。定义unified timeindex：

$$\bar{t}(t, k) = tK + (K - k - 1)$$

其中：
- $t$ 是environment timestep（外层MDP的step）
- $k$ 是denoising step（内层MDP的step，从K-1降到0）
- $\bar{t}$ 是unified index，随$t$增加，在同一$t$内随$k$减小而增加

这个indexing的巧妙之处：它把K×T的grid展开成一个一维序列。对于固定的$t$，$k$从K-1降到0对应$\bar{t}$从$tK$增加到$tK+K-1$；然后跳到下一个$t$对应$\bar{t}=(t+1)K$。

### 2.2 States, Actions, Rewards in $\mathcal{M}_{DP}$

$$\bar{s}_{\bar{t}(t,k)} = (s_t, a_t^{k+1}), \quad \bar{a}_{\bar{t}(t,k)} = a_t^k$$

- $\bar{s}$ (bar-state)：包含environment state $s_t$ 和当前noisy action $a_t^{k+1}$
- $\bar{a}$ (bar-action)：一步denoising后的action $a_t^k$

Reward设计精妙：

$$\bar{R}_{\bar{t}(t,k)} = \begin{cases} 0 & k > 0 \\ R(s_t, a_t^0) & k = 0 \end{cases}$$

只有在$k=0$（最后一步denoising完成，action要执行）时才有environment reward。中间所有denoising步骤reward都是0。这很合理——denoising过程本身没有external reward信号，reward只来自environment interaction。

### 2.3 Transition Dynamics

$$\bar{P}(\bar{s}_{\bar{t}+1}|\bar{s}_{\bar{t}}, \bar{a}_{\bar{t}}) = \begin{cases} \delta_{(s_t, a_t^k)} & \bar{t} = \bar{t}(t,k), k > 0 \\ P(s_{t+1}|s_t, a_t^0) \otimes \mathcal{N}(0, I) & \bar{t} = \bar{t}(t,k), k = 0 \end{cases}$$

- 当 $k > 0$：deterministic transition，把denoised action $a_t^k$ 放到下一个state里，$s_t$ 不变
- 当 $k = 0$：environment dynamics执行 $a_t^0$，同时reset noise为新的 $\mathcal{N}(0, I)$（下一个timestep的初始噪声）

这个 $\otimes \mathcal{N}(0, I)$ 是关键——纯噪声 $a_t^K$ 被视为environment的一部分，这就让denoising MDP成为一个proper MDP而不是POMDP。

### 2.4 Policy in the Bar-MDP

$$\bar{\pi}_\theta(\bar{a}_{\bar{t}(t,k)}|\bar{s}_{\bar{t}(t,k)}) = \pi_\theta(a_t^k|a_t^{k+1}, s_t) = \mathcal{N}(a_t^k; \mu(a_t^{k+1}, \varepsilon_\theta(a_t^{k+1}, k+1, s_t)), \sigma_{k+1}^2 I)$$

注意index shift：policy在bar-time $\bar{t}(t,k)$ 给出 $a_t^k$，conditioned on $a_t^{k+1}$（在state里）。这就是一个标准的Gaussian policy！可以直接算 $\log \bar{\pi}_\theta$ 和它的gradient。

### 2.5 Policy Gradient in Bar-MDP

$$\nabla_\theta \bar{J}(\bar{\pi}_\theta) = \mathbb{E}^{\bar{\pi}_\theta, \bar{P}, \bar{P}^0}\left[\sum_{\bar{t} \geq 0} \nabla_\theta \log \bar{\pi}_\theta(\bar{a}_{\bar{t}}|\bar{s}_{\bar{t}}) \bar{r}(\bar{s}_{\bar{t}}, \bar{a}_{\bar{t}})\right]$$

这就是REINFORCE在bar-MDP上的标准形式。每个denoising step都贡献一个 $\nabla_\theta \log \mathcal{N}$ 项，weighted by return-to-go $\bar{r}$。

## 3. PPO Instantiation: 关键的设计决策

### 3.1 Advantage Estimation 的核心Trick

这是paper里最巧妙的engineering contribution之一。定义：

$$\hat{A}^{\pi_{\theta_{old}}}(\bar{s}_{\bar{t}}, \bar{a}_{\bar{t}}) := \gamma_{DENOISE}^k \left(\bar{r}(\bar{s}_{\bar{t}}, \bar{a}_{\bar{t}}) - \hat{V}^{\bar{\pi}_{\theta_{old}}}(\bar{s}_{\bar{t}(t,0)})\right)$$

这里：
- $\gamma_{DENOISE} \in (0,1)$ 是denoising discount factor
- $k$ 是当前denoising step
- $\gamma_{DENOISE}^k$ 给早期（高噪声）denoising step的advantage打折扣
- $\hat{V}(\bar{s}_{\bar{t}(t,0)})$ 只在 $k=0$ 处estimate

**关键设计**：Value function只依赖environment state $s_t$，不依赖noisy action $a_t^{k+1}$！

$$\hat{V}^{\bar{\pi}_{\theta_{old}}}(\bar{s}_{\bar{t}(t,0)}) := \tilde{V}^{\bar{\pi}_{\theta_{old}}}(s_t)$$

为什么？因为noisy action $a_t^{k+1}$ 包含大量随机噪声（特别是高k时几乎是纯噪声），estimate它的value是high variance的。实验证明这个design choice在challenging manipulation tasks上crucial（Appendix C.2的Figure A2）。

### 3.2 GAE for Environment Steps

For $k=0$ steps（实际environment interaction），用standard GAE-$\lambda$:

$$\hat{A}_{\bar{t}(t,k=0)}^\lambda = \sum_{l=0}^{\infty} (\gamma\lambda)^l \bar{\delta}_{\bar{t}(t+l,k=0)}$$

$$\bar{\delta}_{\bar{t}(t,k)} = \bar{R}_{\bar{t}(t,k)} + \gamma_{ENV} V_\phi(\bar{s}_{\bar{t}(t+1,k)}) - V_\phi(\bar{s}_{\bar{t}(t,k)})$$

这里 $\gamma_{ENV}$ 是environment discount（典型0.99或0.999），$\lambda$ 是GAE parameter（典型0.95）。

### 3.3 PPO Clipped Objective

$$\mathcal{L}_\theta = \mathbb{E}^{\mathcal{D}_{itr}} \min\left(\hat{A} \frac{\bar{\pi}_\theta(\bar{s}, \bar{a})}{\bar{\pi}_{\theta_{old}}(\bar{s}, \bar{a})}, \hat{A} \cdot \text{clip}\left(\frac{\bar{\pi}_\theta}{\bar{\pi}_{\theta_{old}}}, 1-\varepsilon, 1+\varepsilon\right)\right)$$

注意：因为每个denoising step都是explicit Gaussian，ratio $\bar{\pi}_\theta / \bar{\pi}_{\theta_{old}}$ 可以直接解析计算，不需要任何approximation！这是相比Q-learning方法的一个大优势。

### 3.4 分denoising step的clipping schedule

Paper发现一个很巧妙的trick：对不同denoising step用不同的clipping ratio：

$$\varepsilon_k = \varepsilon_0 \cdot 0.1^{k/(K-1)}$$

早期denoising step（高噪声）用更小的clipping ratio $\varepsilon$，后期（低噪声）用更大的。Intuition：早期denoising step的Gaussian distribution variance大（$\sigma_k$ 大），policy update容易overshoot，需要更conservative的clipping。

## 4. Best Practices 的工程细节

### 4.1 Fine-tune Only Last $K'$ Steps

Pre-trained DDPM可能有 $K=100$ denoising steps。DPPO不需要fine-tune所有，只fine-tune最后 $K'$ 步（典型10步）。

Implementation：复制network weights $\theta_{FT} \leftarrow \theta$，前 $K - K'$ 步用frozen $\theta$，最后 $K'$ 步用trainable $\theta_{FT}$。

**Intuition**：早期denoising step主要是把纯噪声变成"approximately expert-like"的action distribution，这部分pre-training已经学好。后期denoising step负责"refine to optimal for task"，这是RL fine-tuning要调整的。这跟LoRA的intuition类似——只调end task-specific的部分。

### 4.2 DDIM Fine-tuning

DDIM (https://arxiv.org/abs/2010.02502) 可以把 $K=100$ DDPM steps压缩到 $K^{DDIM}=5$ steps：

$$x^{k-1} \sim p_\theta^{DDIM}(x^{k-1}|x^k) := \mathcal{N}(x^{k-1}; \mu^{DDIM}(x^k, \varepsilon_\theta(x^k, k)), \eta \sigma_k^2 I)$$

- $\eta = 0$：deterministic DDIM（标准用法）
- $\eta = 1$：stochastic，等同于DDPM

DPPO的trick：**训练时用 $\eta = 1$ 提供exploration noise，evaluation时用 $\eta = 0$ 得到deterministic policy**。

这在pixel-based tasks和long-horizon furniture assembly上特别有用，因为只fine-tune 5步而不是100步，memory和compute大幅节省。

### 4.3 Noise Schedule Clipping

Cosine schedule (https://arxiv.org/abs/2102.09672) 默认在 $k=0$ 时 $\sigma_k \approx 10^{-4}$。DPPO发现这对exploration太小了。

Two clipping values:
- $\sigma_{min}^{exp} \in [0.01, 0.1]$：用于sampling时的exploration noise
- $\sigma_{min}^{prob} \geq 0.1$：用于计算log-likelihood时的variance（避免gradient magnitude爆炸）

这个dual schedule很关键。Sampling时需要足够noise来explore，但计算likelihood时如果variance太小，$\log \mathcal{N}$ 的gradient会变得unbounded。

### 4.4 Action Chunking

Diffusion Policy预测 $T_p$ 步future action chunk，执行 $T_a \leq T_p$ 步。DPPO的best practice：
- Pre-training用大 $T_p$（如16）保证temporal consistency
- Fine-tuning用小 $T_a$（如8）更amenable to policy gradient

UNet architecture支持 $T_a < T_p$（convolution只在action chunk维度），MLP需要 $T_a = T_p$。

## 5. 实验结果的深入分析

### 5.1 vs. 其他 Diffusion-based RL 方法

Baseline包括：
- **DRWR/DAWR**：reward/advantage-weighted regression（本文新提出）
- **DIPO** (https://arxiv.org/abs/2305.13122)：action gradient更新sampled actions
- **IDQL** (https://arxiv.org/abs/2304.10573)：Q-learning with implicit policy extraction
- **DQL** (https://arxiv.org/abs/2208.06193)：Diffusion Q-Learning
- **QSM** (https://arxiv.org/abs/2312.11752)：Q-Score Matching

GYM tasks上（Hopper, Walker2D, HalfCheetah），IDQL和DIPO表现competitive。但在sparse-reward ROBOMIMIC tasks上，所有Q-learning方法都exhibit training instability——开始时performance drop，然后无法recover。

DPPO的核心优势：**training stability**。PPO的clipping机制防止policy collapse，on-policy nature避免Q-function misestimation的问题。

### 5.2 vs. 其他 Policy Parameterizations

比较：
- Gaussian (diagonal covariance)
- GMM (5 mixtures)
- DPPO-MLP, DPPO-UNet

在ROBOMIMIC Square和Transport（最难的tasks）上：
- State input: DPPO达到 >90% success，Gaussian和GMM显著lower
- Pixel input: DPPO在Transport上dramatically outperform Gaussian（Gaussian不improve from 0% pre-trained success rate）

Transport是双臂任务，59-dim state, 14-dim action, 800 steps episode。DPPO是第一个solve Transport到 >50% success的RL algorithm。

### 5.3 Furniture-Bench & Sim-to-Real

Furniture-Bench (https://arxiv.org/abs/2305.12821) 的三个tasks：One-leg, Lamp, Round-table。

**Sim-to-real的关键结果**：
- DPPO: 80% real success rate (16/20)
- Gaussian: 0% real success rate, despite 88% sim success
- Gaussian + BC regularization: 50% real, 53% sim

Gaussian policy在sim里work但real里完全fail——说明sim-to-real gap不只是performance问题，是policy behavior quality问题。Supplementary video显示Gaussian policy exhibit "volatile and jittery behavior"。

DPPO的multi-step denoising过程自然产生smooth, iterative refinement，这跟物理controller的smoothness要求aligned。

## 6. 为什么 DPPO Work? 三大机制

### 6.1 Structured On-Manifold Exploration

D3IL Avoid environment的可视化实验（Section 6, Figure 10）很illuminating。Pre-training data有M1, M2, M3三个setting，每个有两个modes。

Comparison of exploration patterns：
- **DPPO**：在expert data manifold周围explore，wide coverage但不离manifold太远
- **Gaussian**：unstructured noise，特别在M2上exploration很乱
- **GMM**：narrow coverage，explore不够

**Mechanism**：Diffusion的每一步denoising都加noise，但同时通过 $\varepsilon_\theta$ network把action推回expert data manifold。这是Permenter & Yuan (https://arxiv.org/abs/2306.04848) 说的"diffusion as optimization"——denoising过程相当于project noisy action onto data manifold。

Gaussian只在最终action上加noise，没有这个projection机制。GMM虽然有multiple modes但每个mode内部还是Gaussian-like noise。

**Additional insight**：Action chunking让DPPO的stochasticity同时structure在action dimension和time horizon两个维度。

### 6.2 Training Stability from Multi-Step Denoising

Figure 11的实验：在fine-tuning过程中gradually inject noise to actions。
- Gaussian和GMM：performance collapse
- DPPO with ≥4 denoising steps：robust

**Mechanism**：Multi-step denoising是一种implicit regularization。即使第一步denoising的update把action distribution推偏，后续步骤的denoising会"pull back" towards pre-trained distribution。Figure 12可视化了这个iterative refinement的preservation。

这跟Diffusion Forcing (https://arxiv.org/abs/2407.01392, 你自己的工作，Max Simchowitz co-author) 的idea相关——multi-step diffusion naturally provides stability for long-horizon prediction。

### 6.3 Robustness to Perturbation

Figure 13：fine-tuned policy加noise测试。
- DPPO policy robust to action noise
- DPPO policy从更大initial state distribution收敛到near-optimal path

**Mechanism**：Diffusion Policy能represent complex multi-modal distribution，可以"deconvolve noise from noisy states"（Block et al. 2024, https://arxiv.org/abs/2406.01361）。这是diffusion policy的theoretical property。

## 7. 联想到的 Related Work 和 Open Questions

### 7.1 跟你的工作的联系

你在Diffusion Forcing (https://arxiv.org/abs/2407.01392) 中提出把next-token prediction和full-sequence diffusion统一。DPPO的two-layer MDP formulation跟这有structural similarity——都是把diffusion process嵌入到sequential decision making。

Open question: 能否把Diffusion Forcing的framework extend到RL setting？Full-sequence diffusion + RL fine-tuning for long-horizon planning？

### 7.2 跟 LLM RLHF 的联系

DPPO的formalism跟RLHF for LLM非常类似：
- DDPO (Black et al.) 已经把diffusion当MDP用PPO fine-tune text-to-image
- DPPO把这扩展到sequential decision making
- 自然延伸：diffusion-based language models (https://arxiv.org/abs/2406.07524) + DPPO-style fine-tuning for interactive dialogue/planning

LLM的RLHF用的是Gaussian或categorical policy + PPO。如果未来diffusion-based language models成熟，DPPO framework可以直接apply。

### 7.3 跟 Model-Based RL 的联系

DPPO是model-free的。但denoising process itself可以看作一个learned "action model"——给定当前state和desired action，predict如何从noise denoise到action。

Potential hybrid：DPPO + world model (https://arxiv.org/abs/2312.08533)。用world model generate synthetic experience，减少environment interactions。

### 7.4 Sample Efficiency 的 Limitation

DPPO的主要limitation：on-policy PG比off-policy Q-learning sample efficiency低。Appendix C显示从scratch训练时DPPO比Gaussian慢6× wall-clock。

Potential solution: 
- Importance sampling for off-policy DPPO
- Replay buffer with diffusion policy的likelihood-based weighting
- 跟SERL (https://arxiv.org/abs/2401.16013) 的real-world RL setup结合

### 7.5 Exploration 的 Theoretical Understanding

Section 6的on-manifold exploration很phenomenological。更深层的theoretical question：

Diffusion policy的exploration为什么比Gaussian policy更structured？这跟score-based generative models的Langevin dynamics interpretation有关——denoising过程approximates Langevin sampling on data manifold。

Potential direction: Formal analysis of DPPO exploration via score matching + RL theory。连接到Block et al. (https://arxiv.org/abs/2406.01361) 的generative behavior cloning理论。

### 7.6 Multi-Task Pre-training + DPPO Fine-tuning

Paper最后提到DPPO很适合"pre-train on diverse tasks + fine-tune on specific task"的paradigm。这跟LLM的pre-train + fine-tune完全analogous。

POCO (https://arxiv.org/abs/2402.02511) 和π₀ (https://arxiv.org/abs/2410.24164) 都是multi-task diffusion policy pre-training。DPPO + multi-task pre-training是scaling up robot learning的promising方向。

### 7.7 跟 你的 micrograd / llm.c 风格的 Implementation

从implementation perspective，DPPO的clean之处在于：
- 每个denoising step就是一个Gaussian sample + log-likelihood
- PPO update就是standard clipped objective
- 不需要Q-function的double Q-learning, target network等复杂machinery

这跟你的llm.c philosophy一致——simple, clean, understandable implementation。一个minimal DPPO implementation可能只需要：
- DDPM forward/reverse process（~100 lines）
- PPO clipped objective（~50 lines）
- GAE estimation（~30 lines）
- Two-layer MDP rollout（~50 lines）

Total ~300 lines的clean DPPO implementation应该是feasible且educational的。

## 8. 公式汇总和 Variable Glossary

### 关键 Variables:
- $s_t \in S$: environment state at time $t$
- $a_t \in \mathcal{A}$: executed action chunk (length $T_a$) at time $t$
- $a_t^k$: noisy action at denoising step $k$ (from $k=K$ noise to $k=0$ clean)
- $K$: total denoising steps (pre-training, e.g., 100)
- $K'$: fine-tuned denoising steps (e.g., 10)
- $T_p$: prediction horizon (action chunk length for prediction)
- $T_a$: action horizon (actually executed steps, $T_a \leq T_p$)
- $\theta$: diffusion policy network parameters
- $\phi$: value function network parameters
- $\varepsilon_\theta$: noise prediction network
- $\sigma_k$: noise schedule at step $k$

### Key Hyperparameters:
- $\gamma_{ENV} \in (0,1)$: environment discount (typical 0.99-0.999)
- $\gamma_{DENOISE} \in (0,1)$: denoising discount (typical 0.8-0.99)
- $\lambda \in [0,1]$: GAE parameter (typical 0.95)
- $\varepsilon \in (0,1)$: PPO clipping ratio (typical 0.01-0.1)
- $\sigma_{min}^{exp}$: minimum exploration noise (0.01-0.1)
- $\sigma_{min}^{prob}$: minimum likelihood variance (≥0.1)
- $\eta \in [0,1]$: DDIM stochasticity (1 for train, 0 for eval)

### Wall-clock time comparison (from Tables A1-A5):

GYM (per iteration, 20000 steps):
- DPPO: 16.6-18.3s
- IDQL: 15.5-16.3s (slightly faster)
- DQL: 17.6-20.5s (slower)
- QSM: 9.6-9.9s (fastest, but worse performance)

ROBOMIMIC Transport (per iteration, 160000 steps, state):
- DPPO-MLP: 350.5s
- DPPO-UNet: 431.1s
- Gaussian-MLP: 255.6s (DPPO 37% slower)
- DPPO-ViT-MLP (pixel): 871.3s vs Gaussian-ViT-MLP 770.0s (13% slower)

Furniture-Bench (per iteration, 700K-1M steps):
- DPPO-UNet: 148-258s
- Gaussian-MLP: 102-203s (DPPO 20-46% slower)

## 9. 总结: DPPO 的本质

DPPO的本质insight：**Diffusion policy的denoising过程是一个天然的sequential decision making structure，每一步都有tractable Gaussian likelihood，因此policy gradient可以直接apply**。

这看似obvious但之前没人work on it for robot learning。原因可能是：
1. Psenka et al. (QSM paper) conjecture它inefficient due to large action variance from long effective horizon
2. Community focus on Q-learning for diffusion RL

DPPO的contribution是show这个conjecture在fine-tuning setting不成立，并提供engineering best practices让it actually work well。

Three pillars of DPPO's success:
1. **Structured on-manifold exploration** from diffusion's projection property
2. **Training stability** from multi-step denoising's implicit regularization
3. **Policy robustness** from diffusion's multi-modal representation capacity

这三点都根植于diffusion model的structure property，不是PPO本身的property。这跟你的intuition（build from first principles）一致——理解algorithm的内在structure比engineering tricks更重要。

## References

- DPPO: https://diffusion-ppo.github.io
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- PPO: https://arxiv.org/abs/1707.06347
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- DDPO (Black et al.): https://arxiv.org/abs/2305.13301
- IDQL: https://arxiv.org/abs/2304.10573
- DQL: https://arxiv.org/abs/2208.06193
- QSM: https://arxiv.org/abs/2312.11752
- DIPO: https://arxiv.org/abs/2305.13122
- ROBOMIMIC: https://robomimic.github.io/
- Furniture-Bench: https://arxiv.org/abs/2305.12821
- D4RL: https://arxiv.org/abs/2004.07219
- Diffusion Forcing (your work): https://arxiv.org/abs/2407.01392
- POPO/POCO: https://arxiv.org/abs/2402.02511
- Cosine noise schedule (Improved DDPM): https://arxiv.org/abs/2102.09672
- Block et al. generative BC theory: https://arxiv.org/abs/2406.01361
- Permenter & Yuan diffusion as optimization: https://arxiv.org/abs/2306.04848
- D3IL benchmark: https://arxiv.org/abs/2402.14606
- RLPD: https://arxiv.org/abs/2306.00783
- Cal-QL: https://arxiv.org/abs/2304.12972
- IBRL: https://arxiv.org/abs/2311.02198
- SERL: https://arxiv.org/abs/2401.16013
- π₀: https://arxiv.org/abs/2410.24164
