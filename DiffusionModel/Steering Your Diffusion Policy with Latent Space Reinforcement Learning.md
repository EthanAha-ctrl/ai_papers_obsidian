---
source_pdf: Steering Your Diffusion Policy with Latent Space Reinforcement Learning.pdf
paper_sha256: f28ff79694d21fd4a90b292632203f579411486e8cfc42ced00556b7eb6f6866
processed_at: '2026-08-12T11:01:58-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 DSRL

## 一句话版本

Diffusion policy 跑起来的时候，第一步是 sample 一个高斯噪声 **w**，然后 denoise 成 action。这篇 paper 说：**别瞎 sample w 了，让 RL 来挑 w**——挑一个好 w，就能让 frozen 的 diffusion policy 输出你想要的好 action。

## 为什么这个 idea 不 trivial

你可能会觉得"挑个 noise 还能有多大用"。但仔细想想，diffusion policy 训练的时候，就是学了一个从 N(0,I) 到 demonstration action distribution 的映射。所以 **w 这个 latent space 实际上编码了 demonstration 的所有 modes 和 variations**。

当你说"让 RL 选 w"，其实是在说：**让 RL 在 BC 已经学好的 manifold 上搜索**，而 w 只是这个 manifold 的 coordinate。这比让 RL 直接在 raw action space 里瞎搜要 sample efficient 得多，因为 BC 已经帮你把"什么样的 action 长得像 demo"这件事 encode 进去了。

## 为什么这个 idea 能 work

**DDIM / flow matching 让 denoising 变 deterministic**。给定同一个 w，输出永远一样。这个性质太关键了——意味着 w 是一个 well-defined 的"控制旋钮"。你拨一下，policy 给一个确定的响应。DDPM 就不行，同一个 w 每次出来的 action 都不一样，RL 就没法学。

**很多不同的 w 会映射到同一个 a**。这个叫 noise aliasing。因为 demo 的 action distribution 通常很窄，diffusion policy 学到的是"一大片 w 都 denoise 到差不多的 a"。这件事被 paper 利用得很巧：你不用真的去 explore 所有的 w，只要某个 w 对应的 a 被 explore 过了，所有 alias 到同一个 a 的 w 都能借光拿到 value 估计。这就是 DSRL-NA 的精髓——用一个 action-space critic Q^A + 一个 latent-noise critic Q^W，Q^W 从 Q^A 蒸馏过来，相当于免费拿到了"没去过的 w 的 value"。

## 为什么这个比 fine-tune 整个 policy 强

传统 fine-tune diffusion policy（DPPO、DQL 这些）要 backprop 整个 denoising chain。多步 denoising 的 backprop 既慢又 numerical unstable，在大模型上（比如 π_0 的 3.3B 参数）根本跑不动。

DSRL 把整个 diffusion policy 当成一个 **black-box 的 action generator**，只 forward 不 backward。你只训练一个 tiny 的 noise-choosing MLP，base policy 完全 frozen。所以：

- 不需要 gradients 通过 denoising chain
- 不需要 base policy 的 weights（API access 就够）
- 在 RTX 3070 上就能 fine-tune π_0（LoRA 都要 22.5GB）

## 为什么 offline 也能 work

如果你有 offline dataset D_off，π_dp 是从 D_off 上 BC 出来的，那么 π_dp 只会输出 D_off 里的 action。所以不管你给它什么 w，出来的 a 都在 D_off 的 support 里。这意味着你的 critic 永远只在 in-distribution action 上被 query，**天然保守，不需要 CQL 那套 pessimism trick**。

这是我觉得最 elegant 的地方——BC 的 inductive bias 直接 inherit 进 RL，完全免费。

## 真实世界的效果

Franka 上 cube pick-and-place：
- 原始 π_dp：2/10
- RLPD（标准 online RL + offline data）：0/10
- RLPD + 人工干预：0/10  
- DSRL：9/10

只用了 40 个 episode。

更夸张的是 π_0：80 个 episode 把 toaster 任务从 5/20 提到 18/20。这是第一次有人成功在真实世界 RL fine-tune π_0 这种规模的 VLA model。

## 一个让你 build intuition 的 mental model

把 diffusion policy 想象成一个 **frozen 的大厨**，他已经会做菜单上所有的菜（demonstration 的所有 modes）。传统 fine-tune 是要重新训练这个大厨——很贵，可能把他教坏。

DSRL 是给大厨配一个 **点菜员**（tiny noise policy π^W）。大厨的食谱不变，但点菜员学会了"在什么情况下点哪道菜"。点菜员很小很好训，大厨可以是个 3.3B 参数的 π_0 都无所谓——你只需要会"点菜"。

**w 就是菜单上的菜名**。BC 训练的时候，菜单已经定好了。RL 要做的只是学会读场景、点对菜。

## 这篇 paper 真正的意义

我觉得这篇 paper 最大的贡献不是算法本身，而是**展示了一个新的 paradigm**：对 large pre-trained generative models 做 task-specific adaptation，不一定要 fine-tune 它们的 weights，可以在它们的 latent input space 上做 RL。这个 paradigm 可能会扩展到 image generation、video models、甚至 LLMs（想想看，prompt token 也是某种 latent）。

它把"RL fine-tuning large models"这件事从"需要大算力、需要 gradients、需要 weights access"降维到了"只需要 forward pass + 一个 small policy"。这对真实世界机器人学习的 practical impact 可能很大。

Reference:
- Project page: https://diffusion-steering.github.io
- π_0: https://arxiv.org/abs/2410.24164
- Diffusion Policy: https://arxiv.org/abs/2303.04137

---

# DSRL: Steering Your Diffusion Policy with Latent Space RL - 深度讲解

## 1. 核心Intuition：把noise变成新的action space

Andrej，这篇paper最核心的insight其实非常简洁，但很深刻。我先用一段话build你的intuition：

标准diffusion policy π_dp 在部署时执行：
1. Sample noise **w** ~ N(0, I)
2. 通过 reverse diffusion process 把 **w** "denoise" 成 action **a**

DSRL 的核心观察：**a ← π_dp(s, w)** 这个映射，从 RL 的角度看，等价于一个 **action space transformation**。原本 RL 要在 action space A 里搜索，现在我把搜索空间搬到 latent-noise space W，把 π_dp 当成 "environment的一部分"——你给我一个 w，我把它 "forward pass" 一次得到 a，再让 environment 执行 a。这样整个 RL problem 就变成了一个标准的 MDP，只不过 action 维度没变（都是 d 维），但 **environment 多塞了一个 frozen 的denoising 链**。

这个 reframing 的好处极其重要：
- **完全 black-box**：只需要 forward pass π_dp，不需要 weights，不需要 intermediate denoising steps，甚至可以走 API
- **避免 backprop through diffusion chain**：DPPO、DQL 等方法要 backprop 多步 denoising，numerically unstable 且 expensive，DSRL 直接绕开
- **小 actor + 大 frozen policy**：你只需要 train 一个小 MLP π^w，而 base policy 可以是 3.3B 参数的 π_0

## 2. 数学框架：Latent-Action MDP

### 2.1 原始 MDP vs 变换后的 MDP

原始 MDP: M = (S, A, P, p_0, r, γ)

DSRL 构造的 **latent-action MDP**:
$$\mathcal{M}^{\mathcal{W}} := (\mathcal{S}, \mathcal{W}, P^{\mathcal{W}}, p_0, r^{\mathcal{W}}, \gamma)$$

其中关键的变换公式：

$$P^{\mathcal{W}}(\cdot | s, w) := P(\cdot | s, \pi_{\mathrm{dp}}^{\mathcal{W}}(s, w))$$
$$r^{\mathcal{W}}(s, w) := r(s, \pi_{\mathrm{dp}}^{\mathcal{W}}(s, w))$$

变量解释：
- **P^W**: 在 latent-noise space 下的转移概率——给定你选了 noise w，先通过 π_dp^W 映射到 action a，再执行 a 得到下一个 state 的分布
- **r^W**: 同理，是 latent-noise reward
- **π_dp^W(s, w)**: diffusion policy 用 DDIM 或 flow-based sampling 时，从初始 noise w 出发，走完整个 denoising chain 得到的 deterministic output action

### 2.2 为什么 DDIM/Flow-based 是关键

paper Section 3 提到，对于 DDPM (σ_t > 0)，reverse process 是 stochastic 的；而对于 **DDIM (σ_t = 0)** 或 **flow matching**，给定初始 w，整个 denoising process 是 **deterministic** 的。这点至关重要：

- Deterministic → π_dp^W(s, w) 是一个 well-defined 的函数
- 你可以 reproducibly 地比较两个 w 的好坏
- 也可以实现 noise aliasing（Section 4.2 的核心）

DDIM 的 update 公式（paper Eq 1）：
$$x_{t-1} = \alpha_t (x_t - \beta_t \epsilon_\theta^{(t)}(x_t)) + \sigma_t \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

变量/符号说明：
- **x_t**: diffusion chain 中 timestep t 的 sample（开始时 x_T = w 是 noise，结束时 x_0 = a 是 action）
- **α_t, β_t, σ_t**: 调度系数，控制 denoising 强度
- **ε_θ^(t)**: 在 step t 训练的 denoising network（带参数 θ）
- **ε_t**: 注入的 stochastic noise（DDPM 才有，DDIM 设 σ_t = 0）

DDIM 设 σ_t = 0 后，整个 chain 从 w = x_T 到 x_0 = a 是一个 **deterministic mapping**，这就是为什么 paper 主要用 DDIM/flow。

## 3. 关键技术创新：Noise Aliasing (DSRL-NA)

这是这篇 paper 最有价值的方法学创新。让我详细解释。

### 3.1 Aliasing 现象

Diffusion policy 训练在 demonstration data 上，demonstrator 的 action distribution 通常很窄（比如一条精细的轨迹）。这导致 **很多不同的 w ∈ W 会被 denoise 到相同的 a**，即：

$$\exists w' \neq w \quad \text{s.t.} \quad \pi_{\mathrm{dp}}^{\mathcal{W}}(s, w) \approx \pi_{\mathrm{dp}}^{\mathcal{W}}(s, w')$$

paper Figure 2 直观地展示了这一点——两个不同的 noise 点 w_2, w_3 可能都映射到同一个 a'。

### 3.2 双 Critic 架构

DSRL-NA 维护两个 critic：

**Q^A: S × A → R** (action space critic)
- 通过 TD learning 训练，可以处理 offline data（带 a-action 的 transitions）
- Bellman update:
$$\min_{Q^A} \mathbb{E}_{(s,a,r,s') \sim \mathfrak{B}, a' \sim \pi_{\mathrm{dp}}^{\mathcal{W}}(s', \pi^{\mathcal{W}}(s'))} \left[ \left( Q^A(s, a) - r - \gamma \bar{Q}^A(s', a') \right)^2 \right]$$
- 变量：
  - Q^A(s,a): 在 state s 下采取 action a 的 Q-value
  - r: immediate reward
  - γ: discount factor
  - Q̄^A: target critic（用 polyak averaging）
  - a': next state 下由 latent policy π^W 选 w，再 forward π_dp 得到
  - B: replay buffer

**Q^W: S × W → R** (latent-noise critic)
- 通过 **distillation** 从 Q^A 学到：
$$\min_{Q^W} \mathbb{E}_{s \sim \mathfrak{B}, w \sim \mathcal{N}(0, I)} \left[ \left( Q^W(s, w) - Q^A(s, \pi_{\mathrm{dp}}^{\mathcal{W}}(s, w)) \right)^2 \right]$$
- 变量：
  - Q^W(s, w): 在 state s 下选择 latent noise w 的 Q-value
  - Q^A(s, π_dp^W(s, w)): 把 w forward 成 a 后查询 Q^A
  - w ~ N(0, I): 在 prior 上采样 noise 来 distill

**关键 insight**: Q^W 内化了 Q^A 的 dynamics 信息，且 **可以推断从未 take 过的 w 的 value**——只要它对应的 a 曾经被 take 过（即 aliasing 到一个 known action）。

### 3.3 Actor update

$$\max_{\pi^W} \mathbb{E}_{s \sim \mathfrak{B}} \left[ Q^W(s, \pi^W(s)) \right]$$

π^W 在 latent-noise space 里搜索高 Q^W 的 w。

### 3.4 为什么这天然保守（offline RL 友好）

paper Section 4.2 末尾有一个很漂亮的观察：当 π_dp 训练在同样的 offline dataset D_off 上时，DSRL-NA **天然实现 conservatism**：

- π_dp 只输出 in-distribution actions（因为它就是从 D_off 上 BC 出来的）
- 任何 w 都会被 π_dp 映射到 in-distribution a
- 因此 Q^A 只会被 query 在 in-distribution a 上
- 无需显式加 CQL-style penalty，自然避免 offline RL 的 extrapolation error

这点比 IDQL、SRPO 等 diffusion-based offline RL 方法更优雅。

## 4. 算法流程图解析

```
┌─────────────────────────────────────────────────────────────┐
│  Algorithm 1: DSRL-NA                                       │
│                                                             │
│  Input: π_dp (frozen), D_off (offline data), optional env   │
│  Init:   replay buffer B ← D_off                             │
│          Q^A, Q^W, π^W (latent actor)                       │
│                                                             │
│  Loop:                                                      │
│    ┌──────────────────────────────────────────────────────┐  │
│    │ 1. Update Q^A: TD learning on (s,a,r,s') ∈ B         │  │
│    │    next action a' = π_dp^W(s', π^W(s'))             │  │
│    ├──────────────────────────────────────────────────────┤  │
│    │ 2. Update Q^W: distill from Q^A                      │  │
│    │    sample w~N(0,I), compute a=π_dp^W(s,w)           │  │
│    │    regress Q^W(s,w) → Q^A(s,a)                       │  │
│    ├──────────────────────────────────────────────────────┤  │
│    │ 3. Update π^W: maximize E[Q^W(s, π^W(s))]            │  │
│    ├──────────────────────────────────────────────────────┤  │
│    │ 4. (online) Sample w~π^W(s), compute a=π_dp^W(s,w)  │  │
│    │    play a in env, add (s,a,r,s') to B               │  │
│    └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 5. 实验结果深度解析

### 5.1 Online adaptation (Figures 3, 4)

- Benchmark: Robomimic (Can, Lift, Square, Transport), OpenAI Gym (Hopper, Walker2D, HalfCheetah)
- Baselines: DPPO, IDQL, DQL, DIPO, QSM
- 关键结果：DSRL 比 SOTA 方法 **快 5-10 倍** 达到 comparable performance

我特别想强调 Figure 11 的 ablation：
- 即使 π_dp 的 layer width 从 small 变到 1024，DSRL 都能 steer
- 1024 模型虽然初始 performance 和 512 一样，但 DSRL steer 1024 更快
- **Hypothesis**: 更大 model capacity 能 encode 更多 diverse behaviors，给 RL 更多 "选择"

### 5.2 Offline RL (Table 1)

在 OGBench 上，DSRL 在 10 个 task 中约一半 SOTA。值得注意的是它直接用了 online RL 的 DSRL-NA 算法，没有任何修改就能做 offline RL。

特别有意义的对比：
- DSRL vs BC(π_dp): DSRL 大幅超过 frozen base policy
- DSRL vs IDQL/IFQL/FQL: 这些是专门的 diffusion/flow offline RL 方法，DSRL 不输甚至超过

### 5.3 Real-world single-task (Table 2)

Cube pick-and-place on Franka:
| Method | Success |
|--------|---------|
| π_dp | 2/10 |
| RLPD | 0/10 |
| RLPD + interventions | 0/10 |
| DSRL | 9/10 |

只用了 3500 steps (~40 episodes) 从 20% 提升到 90%——这真的让 real-world RL fine-tuning 变得 practical。

### 5.4 π_0 Steering (Section 5.5) - 这部分最 exciting

π_0 是 3.3B 参数的 VLA (Vision-Language-Action) model，用 flow matching，action chunk size = 50，每个 action 32-dim，总维度 1600。

DSRL 用在 π_0 上有两个 trick：

**Trick 1: 单步 noise 重复**
由于 1600 维 noise 直接训 actor/critic 太难，paper 提出只训 single-step actor π_single^W(s) 和 critic Q_single^W(s, w_single)，w_single ∈ R^d（d=32），然后 inference 时把 w_single 复制 C=50 次：w = (w_1, ..., w_C) with w_i = w_single。

这本质上是假设 chunk 内的 noise 是同一个，极大降维。Empirically 仍然 expressive enough。

**Trick 2: 只需要 forward pass**
Real-world π_0 跑在 remote policy server，本地 DSRL 只用 8GB VRAM（RTX 3070）就能 train——对比之下 LoRA 需要 22.5GB，full finetuning 需要 70GB。

结果：
| Task | π_0 | DSRL |
|------|------|------|
| Turn on toaster | 5/20 | 18/20 |
| Put spoon on plate | 15/20 | 19/20 |

80 episodes (11000 steps) 就能显著提升，这是 **首次成功 real-world RL fine-tuning π_0** 的工作。

### 5.5 Ablation 亮点

Figure 13 关于 train epochs 的 ablation 很反直觉：
- π_dp 训练 3000/6000/9000 epochs，最终 DSRL 性能几乎一样
- 即使 π_dp 过拟合，DSRL 还是能 steer
- **暗示**: diffusion policy 的 noise space 即使在过拟合 regime 下，仍然保留了 steerable structure

Figure 12 关于 data quality：
- "Better"、"Okay"、"Worse" demonstrator 训练 π_dp
- Worse 一开始 DSRL 慢一点，但很快追上
- **暗示**: 只要 π_dp 有 task-solving behavior，DSRL 就能 refine

## 6. 与 Related Work 的精细对比

### 6.1 vs Parrot (Singh et al. 2020, arXiv:2011.10024)

Parrot 在 normalizing flow 上做类似的事。但：
- Normalizing flow 是 **invertible** 的，所以优化 noise space 不会 loss expressivity
- Diffusion 不是 invertible，理论上优化 noise 可能 loss expressivity
- 但 empirically，diffusion policy 比 normalizing flow **效果好得多**（后者在 robotics 上表现差 [72]）

所以 DSRL 是把 Parrot 的 idea "升级" 到 SOTA 的 diffusion policy 上，并证明 empirically 仍然 expressive。

### 6.2 vs DPPO (Ren et al. 2024, arXiv:2409.00588)

DPPO 用 PPO 直接 fine-tune diffusion policy 的前几步 denoising。问题：
- 需要 backprop through multi-step denoising
- Numerically unstable
- 在大 model 上 (e.g. π_0) 不可行

DSRL 完全不需要 backprop diffusion chain。

### 6.3 vs RESIP / V-GPS (post-processing methods)

这些方法在 action space 上学一个 residual policy 修正 π_dp 的输出。问题：
- action space 维度高（π_0 是 1600 维）
- 不利用 noise space 的结构
- DSRL 把搜索空间搬到 W（low-dim, structured）

### 6.4 vs IDQL (Hansen-Estruch et al. 2023, arXiv:2304.10573)

IDQL 用 diffusion policy 做 actor，Q-function 做 critic。区别：
- IDQL 用 diffusion policy 直接学 actor（可能从 scratch 或 finetune）
- DSRL 把 diffusion policy 当 frozen transformation，只学 noise selector

## 7. 局限性与未来方向

paper Section 6 和 Limitations 提到：
1. **Steerability 不可保证**: 如果 π_dp 的 action distribution 太 narrow（比如训练数据极少），可能没有 enough "options" 让 DSRL 选
2. **仍需 reward signal 和 online rollouts**: 虽然比 standard RL 高效，但比纯 BC 要求多
3. **理论分析缺失**: noise space 优化 expressivity 没有理论保证

paper 提到的 follow-up 方向：
- 同时修改 observation/prompt（连接到 autoregressive transformer policies）
- 扩展到 image generation 或 protein modeling 等其他 diffusion domain
- 理论分析 steerability

## 8. 我个人觉得最 elegant 的几点

1. **Black-box fine-tuning 的极致**：连 API access 都能 fine-tune，这对 proprietary model 的部署极其有吸引力。

2. **Conservatism 自然涌现**：DSRL-NA 在 offline setting 下不需要显式 CQL penalty 就自然保守，因为 π_dp 本身约束了 in-distribution action——这是把 BC prior 的 benefits 直接 inherit 进来。

3. **Computational asymmetry**：小 actor（MLP）+ 大 frozen policy（3.3B），但只需要 forward pass，让 fine-tuning 在 RTX 3070 上可行。

4. **Connection to LLM fine-tuning**: 这套思路和 RLHF 的 prompt tuning / prefix tuning 类似——不修改 base model，只学一个 low-dim control signal。可以看作 RLHF 的 "noise-space analog"。

## 9. Reference Links

- Paper website: https://diffusion-steering.github.io
- arXiv (paper): https://arxiv.org/abs/2410.24054 (基于 paper title 推断)
- π_0 paper: https://arxiv.org/abs/2410.24164
- Diffusion Policy (Chi et al.): https://arxiv.org/abs/2303.04137
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- DPPO: https://arxiv.org/abs/2409.00588
- DQL: https://arxiv.org/abs/2208.06193
- IDQL: https://arxiv.org/abs/2304.10573
- Flow Q-learning: https://arxiv.org/abs/2502.02538
- IQL: https://arxiv.org/abs/2110.06169
- SAC: https://arxiv.org/abs/1801.01290
- RLPD: https://arxiv.org/abs/2112.13404
- CAL-QL: https://arxiv.org/abs/2312.05161
- OGBench: https://arxiv.org/abs/2410.20092
- Robomimic: https://arxiv.org/abs/2108.03298
- Parrot (predecessor): https://arxiv.org/abs/2011.10024
- V-GPS (Nakamoto et al.): https://arxiv.org/abs/2410.13816
- RESIP: https://arxiv.org/abs/2407.16677
- BridgeData V2: https://arxiv.org/abs/2308.12952
- DiT for robotics (Dasari et al.): https://arxiv.org/abs/2410.10088
- Libero: https://arxiv.org/abs/2306.03310
- DROID: https://arxiv.org/abs/2403.12945

## 10. 一个直觉性的总结

如果让我用一个 analogy 给你：传统 fine-tune diffusion policy 就像 "重新训练大脑改变行为"——昂贵、不稳定、可能忘掉旧技能。DSRL 像 "给一个 frozen 大脑装上一个 noise-choosing 外挂"——大脑怎么 denoise 不变，但你学会喂给它精心挑选的 noise，让它输出你想要的行为。

这本质上是把 **policy optimization 重新 cast 为 latent space optimization**，而 latent space 恰好是 BC-trained diffusion model 内在的、structure-rich 的 representation。这也是为什么它 sample efficient——你在 BC 已经学好的 manifold 上做 RL refinement，而不是从 scratch 学。

我认为这篇 paper 标志着 robotics RL 进入了一个新阶段：**RL fine-tuning large pre-trained policies 不再是 prohibitive**，因为我们可以避开 backprop 和大 model optimization，转而优化它们的 structured latent space。这个 paradigm 应该会扩展到 VLA models、video models、甚至 LLMs。
