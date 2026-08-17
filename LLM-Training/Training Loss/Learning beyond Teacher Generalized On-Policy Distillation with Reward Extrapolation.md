---
source_pdf: Learning beyond Teacher Generalized On-Policy Distillation with Reward
  Extrapolation.pdf
paper_sha256: 0547a34f9f965de1f23461116e989a20ca2e5e1dea3c0919e9e0763de8ea01a6
processed_at: '2026-08-05T12:43:02-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

好,Andrej,我把上一轮的技术展开用更直观的方式重新讲一遍。我会先给你 mental model,再把数学和实验细节接上去。

## 1. 一句话总结

这篇 paper 发现了一个被大家忽视的事实:**On-Policy Distillation (OPD) 本质上就是 RL 的一个特例**,只不过它的 reward 是 dense 的(每个 token 都有 reward),而且 reward 和 KL 正则的权重被锁死在 1:1。作者把这个锁死的权重解开,加了一个系数 λ,结果发现把 λ 调到大于 1(reward extrapolation),student 居然能超过 teacher。

## 2. 先把 OPD 讲清楚

### 2.1 OPD vs SFT 的区别

- **SFT (off-policy distillation)**: teacher 先写好一堆答案,student 拿这些答案来背。问题是 student 被动接收,不知道自己写错的时候该往哪个方向改。
- **OPD (on-policy distillation)**: student 自己写答案,然后 teacher 在 student 写的每个 token 上给出"我觉得这个 token 应该有多少概率"。student 根据这个 dense 反馈去调整。

打个比方:SFT 像学生抄老师写好的范文;OPD 像学生自己写作文,老师在旁边每个字旁边写"我这里会这么用词"。

### 2.2 OPD 的目标函数

$$\mathcal{I}_{OPD}(\theta) = \min_\theta \mathbb{E}_{x \sim D, y \sim \pi_\theta}[\mathcal{D}_{KL}(\pi_\theta(y|x) || \pi^*(y|x))]$$

变量解释:
- $\pi_\theta$: student model (要优化的)
- $\pi^*$: teacher model (固定的)
- $y \sim \pi_\theta$: $y$ 是 student 自己 sample 出来的 trajectory
- $\mathcal{D}_{KL}(\pi_\theta || \pi^*)$: reverse KL,让 student 在 student 自己的 sample 上贴近 teacher

reverse KL 的具体展开:

$$\mathcal{D}_{KL}(\pi_\theta || \pi^*) = \sum_y \pi_\theta(y) \log \frac{\pi_\theta(y)}{\pi^*(y)}$$

注意是 reverse KL 不是 forward KL。reverse KL 会让 student 的 support 包含在 teacher 的 support 内(mode-seeking),这在 distillation 里更合适。

## 3. 核心洞察:OPD 就是 RL 的特例

### 3.1 引入一个 trick

在 reverse KL 里引入任意第三个模型 $\pi_{ref}$:

$$\log \frac{\pi_\theta}{\pi^*} = (\log \pi_\theta - \log \pi_{ref}) - (\log \pi^* - \log \pi_{ref})$$

第一项 $\log \pi_\theta - \log \pi_{ref}$ 在 expectation 下就是 $\mathcal{D}_{KL}(\pi_\theta || \pi_{ref})$。第二项 $\log \pi^* - \log \pi_{ref}$ 是一个固定的 reward signal(对 $\theta$ 来说)。

所以 OPD 可以重写成:

$$\mathcal{I}_{OPD}(\theta) = \max_\theta \mathbb{E}_{x,y \sim \pi_\theta}\left[\log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} - \mathcal{D}_{KL}(\pi_\theta || \pi_{ref})\right]$$

### 3.2 跟标准 RL 对比

标准 KL-constrained RL:

$$\mathcal{I}_{RL}(\theta) = \max_\theta \mathbb{E}_{x,y \sim \pi_\theta}[r(x,y) - \beta \mathcal{D}_{KL}(\pi_\theta || \pi_{ref})]$$

变量:
- $r(x,y)$: reward function
- $\beta$: KL 正则的强度
- $\pi_{ref}$: 参考模型,通常是 RL 前的初始 policy

对比一下,OPD 对应到 RL 的 mapping:
- $r(x,y) = \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)}$ — 这正是 DPO ([Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)) 里讲的 implicit reward
- $\beta = 1$ — reward 和 KL 权重 1:1
- $\pi_{ref}$ 可以是任意模型 — 这是 OPD 相对于 RL 的一个额外自由度

### 3.3 为什么这个 reward 是 dense 的

标准 RL 里 reward 通常只在最后一个 token 给:

$$r_t^{RL} = \begin{cases} 0 & t = 1, \ldots, T-1 \\ \text{outcome reward} & t = T \end{cases}$$

OPD 里每个 token 都有 reward:

$$r_t^{OPD} = \log \frac{\pi^*(y_t | x, y_{<t})}{\pi_{ref}(y_t | x, y_{<t})}, \quad t = 1, \ldots, T$$

直观理解:teacher 在 student 写的每个 token 上,告诉你"我 teacher 在这里会比 reference 高多少概率 / 低多少概率"。这个差值就是这一步的"好坏"信号。每个 token 都有反馈,dense credit assignment,所以 sample efficiency 比 sparse-reward RL 高得多。

这点跟 [Yuan et al. 2024 "Free Process Rewards"](https://arxiv.org/abs/2412.01981) 和 [PRIME](https://arxiv.org/abs/2502.01456) 的思路同源,只不过 OPD 不需要 $\pi^*$ 真的从 $\pi_{ref}$ RL 出来 — 两个 model 可以是不同 size、不同家族,只要能算 log prob ratio 就行。

## 4. G-OPD: 加一个旋钮 λ

### 4.1 公式

既然 OPD 是 RL 的特例,而 RL 里 reward 和 KL 的权重 $\beta$ 是可调的,那为什么 OPD 里非要锁死 $\beta = 1$?解开它:

$$\mathcal{I}_{G-OPD}(\theta) = \max_\theta \mathbb{E}_{x,y \sim \pi_\theta}\left[\lambda \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} - \mathcal{D}_{KL}(\pi_\theta || \pi_{ref})\right]$$

这里 $\lambda = 1/\beta$。$\lambda = 1$ 就是标准 OPD。

### 4.2 Closed-form 解

对 $\log \pi_\theta$ 求导等于 0,得到最优解:

$$\log \pi_\theta(y|x) = \lambda \log \pi^*(y|x) + (1-\lambda) \log \pi_{ref}(y|x)$$

也可以写成:

$$\log \pi_\theta(y|x) = \log \pi^*(y|x) + (\lambda - 1)\underbrace{(\log \pi^* - \log \pi_{ref})}_{\text{teacher 偏离 reference 的方向}}$$

这个 closed-form 是理解整个 paper 的钥匙。

### 4.3 三种 regime 的人话解释

想象一个空间,reference model 在原点,teacher 在另一个点(经过 RL 走到了那里)。student 的 log-prob 是一个被 $\lambda$ 控制的"位置":

| $\lambda$ 值 | student 位置 | 行为 |
|---|---|---|
| $\lambda = 0$ | 在 reference (原点) | student 就是 base model,什么都不学 |
| $0 < \lambda < 1$ | 在 reference 和 teacher 之间 | reward interpolation,性能介于两者之间,可以用来做 budget-controlled reasoning ([Yang et al. 2025e](https://arxiv.org/abs/2502.18080)) |
| $\lambda = 1$ | 在 teacher 位置 | 标准 OPD,student 复制 teacher |
| $\lambda > 1$ | 越过 teacher,沿着 teacher 偏离 reference 的方向继续走 | reward extrapolation (ExOPD),可能超过 teacher |
| $\lambda$ 太大 (e.g. 1.5) | 走过头了 | reward hacking,response length 爆炸,性能退化 |

最 interesting 的就是 $\lambda > 1$ 这一支。

### 4.4 为什么 $\lambda > 1$ 能超过 teacher

这是这篇 paper 的反直觉点。直觉上 distillation 是让 student 模仿 teacher,天花板就是 teacher。但 G-OPD 的 closed-form 解告诉你:student 在 teacher 基础上额外 fit 一个"沿 teacher 偏离 reference 方向"的项。

可以这样理解:teacher 是 base model 经过 RL 走了一步得到的;如果 student 也从 base 出发,沿着同样的"方向"走更远,就有可能走到 teacher 没走到、但更好的点。类似 model soups / EMA / ExPO ([Zheng et al. 2025](https://aclanthology.org/2025.acl-long.53/)) 的思路,只不过 ExPO 在 weight space 做外推,G-OPD 在 distribution space(log-prob space)做外推。

更精确一点:teacher 训练时收敛到的点不一定是最优点,RL 收敛轨迹上可能"经过"了更好的点(类似 Polyak averaging 的直觉)。$\lambda > 1$ 相当于让 student "过冲",去触及那些 teacher 训练时经过但没停留的位置。

## 5. Reference Model 的选择: Reward Correction

在 strong-to-weak distillation(大 teacher 蒸馏小 student)中,reference model 有两个选择:

1. **Default**: 用 student 的 base model $\pi_{base}^{student}$
2. **Reward correction**: 用 teacher 的 pre-RL base model $\pi_{base}^{teacher}$

为什么第二个更好?看 reward 的形式:

- Default: $r = \log \frac{\pi^*}{\pi_{base}^{student}}$
- Corrected: $r = \log \frac{\pi^*}{\pi_{base}^{teacher}}$

DPO 的理论告诉我们, $\log \frac{\pi^*}{\pi_{base}^{teacher}}$ 是 teacher RL post-training 的真实 implicit reward(teacher 是从 $\pi_{base}^{teacher}$ 出发做 RL 的,所以这个 ratio 是 well-defined 的 reward signal)。而 $\log \frac{\pi^*}{\pi_{base}^{student}}$ 由于 student 和 teacher 内在 knowledge gap,会引入额外的 noise。

Correction 项是:

$$\text{corrected reward} = \text{default reward} + \log \frac{\pi_{base}^{student}}{\pi_{base}^{teacher}}$$

相当于把 reward 从"student base 视角"校准到"teacher base 视角"。

代价:需要 $\pi_{base}^{teacher}$ (额外模型),计算 $\log \pi_{base}^{teacher}$ 比 $\log \pi_{base}^{student}$ 贵 (因为 teacher base 更大)。

## 6. 实验要点

### 6.1 Setup

- Base model: Qwen3-4B-Non-Thinking ([Qwen3 report](https://arxiv.org/abs/2505.09388))
- Domain teachers: 用 GRPO ([DeepSeekMath](https://arxiv.org/abs/2402.03300)) 在 DeepMath-103K ([He et al. 2025](https://arxiv.org/abs/2504.11456)) 上做 math RL,在 Eurus-RL-Code ([Cui et al. 2025](https://arxiv.org/abs/2502.01456)) 上做 code RL,得到 Qwen3-4B-Non-Thinking-RL-Math 和 Qwen3-4B-Non-Thinking-RL-Code 两个 domain teacher
- Math benchmarks: AIME24, AIME25, HMMT25 Feb, HMMT25 Nov ([MathArena](https://matharena.ai/))
- Code benchmarks: HumanEval+, MBPP+, LiveCodeBench v6
- Eval: temp=1.0, top-p=1.0, max len 16384, math 每 problem 32 samples, code 4 samples, 用 [Math-Verify](https://github.com/hugging-face/Math-Verify) 校验答案

### 6.2 λ 扫描实验(Section 4.1.2, Figures 2, 3, 4)

λ 取 $\{0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5\}$,观察性能和 response length 的变化趋势:

- λ 从 0 到 1: 性能和 length 单调上升,在 base 和 teacher 之间,符合 closed-form 解的预测
- λ = 1 (OPD): 完全 recover teacher 的行为,accuracy 和 length 都贴 teacher
- λ = 1.25 (ExOPD): **在所有 benchmark 上超过 teacher**
- λ = 1.5: 性能退化,instability,reward hacking,length 爆炸

为什么 1.5 会崩?持续增大 λ 让 student 去 aggressively fit log ratio 的 peak,即使某些 token 有异常大的 log ratio(bias 导致的)也会被 fitting。同时 implicit reward 有 length bias ([LASER, Yang et al. 2025d](https://arxiv.org/abs/2510.14943)),extrapolation 会放大这个 bias。

### 6.3 排除"teacher 训练不足"假说(Table 1)

有人可能质疑:ExOPD 比 teacher 好,是因为 teacher 没训够。作者做了对照:

| 方法 | AIME24 | AIME25 | HMMT25 Feb | HMMT25 Nov | Avg |
|---|---|---|---|---|---|
| Teacher | 58.0 | 54.6 | 32.5 | 38.9 | 46.0 |
| Teacher 再训 100 steps RL | 60.9 (+2.9) | 55.6 (+0.5) | 32.8 (+0.3) | 38.4 (-0.5) | 46.9 (+0.9) |
| **ExOPD 训 50 steps** | **62.7 (+4.7)** | **56.1 (+1.5)** | **33.9 (+1.4)** | **39.3 (+0.4)** | **48.0 (+2.0)** |

ExOPD 用一半的 step,获得两倍的 gain,排除了"训练量"假说。

### 6.4 Multi-Teacher Distillation(Table 2)

把 Math teacher 和 Code teacher 的能力 merge 回原 base model。比较 SFT、ExPO、OPD、ExOPD:

| 方法 | Math Avg | Code Avg | 是否超过所有 domain teacher |
|---|---|---|---|
| Teacher (Math/Code) | 46.0 / 61.2 | — | — |
| SFT | 44.3 / 60.8 | 部分 | 否 |
| ExPO | 45.0 / 62.6 | 部分 | 否 |
| OPD | 46.4 / 60.6 | 性能 ceiling 被 teacher 限制 | 否 |
| **ExOPD** | **47.7 / 62.0** | **是,唯一一个** | **是** |

Training dynamics (Figure 5):
- ExOPD 比 OPD 获得更高 training reward
- ExOPD 的 response length 更长
- ExOPD 的 response entropy 更高

这说明 ExOPD 不是机械模仿 teacher,而是探索出更长的 reasoning chain。

### 6.5 Strong-to-Weak Distillation(Table 3)

Teacher: Qwen3-30B-A3B-Instruct-2507 (MoE)
Student: Qwen3-1.7B-Non-Thinking 或 Qwen3-4B-Non-Thinking

以 4B student 为例:

| 方法 | AIME24 | AIME25 | HMMT25 Feb | HMMT25 Nov | Avg |
|---|---|---|---|---|---|
| Teacher | 74.7 | 62.8 | 44.2 | 57.2 | 59.7 |
| Base student | 21.5 | 21.9 | 10.0 | 8.0 | 15.4 |
| SFT | 45.4 | 40.9 | 22.4 | 31.6 | 35.1 |
| OPD | 55.0 | 48.0 | 29.8 | 37.7 | 42.6 |
| **ExOPD** | **58.7 (+3.7)** | **50.8 (+2.8)** | **33.0 (+3.2)** | **38.8 (+1.1)** | **45.3 (+2.7)** |

OPD 把 4B student 从 15.4 带到 42.6 (一个巨大的 jump),ExOPD 再额外 +2.7。

### 6.6 Reward Correction 实验(Section 4.2.3, Figure 6)

由于拿不到 30B teacher 的 pre-RL variant,作者用 Qwen3-4B-Non-Thinking-RL-Math/Code 作 teacher,Qwen3-4B-Non-Thinking 作 pre-RL variant,student 是 Qwen3-1.7B。

结果:reward correction 在 math 和 code 上都比 default ExOPD 进一步提升,验证了理论分析。代价是计算 $\log \pi_{base}^{teacher}$ 比计算 $\log \pi_{base}^{student}$ 贵。

## 7. Caveats 和 Open Questions

作者明确指出的 limitations:
1. λ 需要调,过大导致 reward hacking 和 length explosion
2. Reward correction 需要额外的 $\pi_{base}^{teacher}$,且计算更贵
3. Experiments 限于 math 和 code,开放域 alignment 未验证

我自己的几点延伸思考:

- **Cross-family distillation** (e.g. GPT → Llama): $\log \frac{\pi^*}{\pi_{ref}}$ 在不同家族模型之间的 well-defined 性存疑,可能需要其他形式的 reward
- **λ 的 schedule**: 论文用固定 λ=1.25,后期可能 annealing λ 避免不稳定
- **Length bias**: 加入 length-normalized reward 或 KL 防止 extrapolation 过度
- **Process-level reward 融合**: dense reward 来自 log-prob ratio,能否融合 rule-based process reward ([PRIME](https://arxiv.org/abs/2502.01456))
- **跟 Thinking Machines Lab blog 的联系**: [Lu 2025](https://thinkingmachines.ai/blog/on-policy-distillation) 独立提出 OPD 的 RL interpretation,G-OPD 在理论上与之互为印证

## 8. The Big Picture

这篇 paper 在 conceptual 上做了两件事:

1. **Unification**: OPD 不是一个新的训练 paradigm,而是 KL-constrained dense RL 的一个特例,只是历史上大家没意识到。这个 unification 让我们把 RL toolbox 里所有的 trick(reward shaping、KL annealing、reference model 选择)都搬到 distillation 上。

2. **Beyond teacher**: distillation 历史上被认为是"student 上限是 teacher",G-OPD 打破了这个 ceiling。在 same-size 设置下可以超越 domain teacher,在 cross-size 设置下可以 push distillation 的极限。

从更宏观的角度,这篇 paper 是 LLM post-training 领域 "RL / distillation / DPO 融合" 趋势的一个缩影。OPD 用 teacher 的 log prob 提供 dense reward,DPO 用 preference 提供 implicit reward,RLHF 用 reward model 提供 explicit reward — 这些看似不同的 paradigm 在数学上同源,都是 KL-constrained policy optimization 的不同 instance。G-OPD 把这个统一视角的实用价值展示出来了。

## Reference Links

- [DPO (Rafailov et al. 2023)](https://arxiv.org/abs/2305.18290)
- [ExPO (Zheng et al. 2025)](https://aclanthology.org/2025.acl-long.53/)
- [On-Policy Distillation (Agarwal et al. 2024)](https://arxiv.org/abs/2406.05310)
- [MiniLLM (Gu et al. 2024)](https://openreview.net/forum?id=5h0qf7IBZZ)
- [Thinking Machines Lab blog (Lu 2025)](https://thinkingmachines.ai/blog/on-policy-distillation)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [DeepMath-103K](https://arxiv.org/abs/2504.11456)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [PRIME (Process Reward)](https://arxiv.org/abs/2502.01456)
- [LASER (length bias in implicit reward)](https://arxiv.org/abs/2510.14943)
- [Free Process Rewards (Yuan et al. 2024)](https://arxiv.org/abs/2412.01981)
- [Thinking-Optimal Scaling (Yang et al. 2025e)](https://arxiv.org/abs/2502.18080)
- [MiMo-v2-Flash (Xiao et al. 2026)](https://arxiv.org/abs/2601.02780)
- [MathArena](https://matharena.ai/)
- [Math-Verify](https://github.com/hugging-face/Math-Verify)
- [verl framework](https://arxiv.org/abs/2409.19256)
- [LiveCodeBench](https://arxiv.org/abs/2403.07974)

需要深挖任何一块(比如 gradient derivation step-by-step,或者跟 DPO/SWR/iSFT 的关系,或者多 teacher merge 的具体实验设计)就告诉我。

---

# Learning beyond Teacher: G-OPD 深度解析

你好 Andrej! 这篇paper由RUC的Wenkai Yang和Tencent合作完成, 核心贡献是建立了On-Policy Distillation (OPD)和dense KL-constrained RL的理论桥梁, 并提出G-OPD让student超越teacher。下面我尽量展开技术细节。

## 1. 核心理论洞察: OPD 是 Dense RL 的特例

### 1.1 关键推导链

OPD的原始目标函数(reverse KL from student to teacher on student-sampled trajectories):

$$\mathcal{I}_{OPD}(\theta) = \min_\theta \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} [\mathcal{D}_{KL}(\pi_\theta(y|x) || \pi^*(y|x))]$$

注意这里inner term是reverse KL, 且$y$是student自己采样的, 这是"on-policy"的本质。

**关键trick:** 引入任意第三个模型 $\pi_{ref}$, 展开reverse KL:

$$\mathcal{D}_{KL}(\pi_\theta || \pi^*) = \log \frac{\pi_\theta(y|x)}{\pi^*(y|x)} = (\log \pi_\theta - \log \pi_{ref}) - (\log \pi^* - \log \pi_{ref})$$

第一项恰好是 $\mathcal{D}_{KL}(\pi_\theta || \pi_{ref})$, 第二项是 reward。因此:

$$\mathcal{I}_{OPD}(\theta) = \max_\theta \mathbb{E}_{x,y \sim \pi_\theta}\left[\log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} - \mathcal{D}_{KL}(\pi_\theta(y|x) || \pi_{ref}(y|x))\right]$$

对比标准KL-constrained RL:

$$\mathcal{I}_{RL}(\theta) = \max_\theta \mathbb{E}_{x,y \sim \pi_\theta}[r(x,y) - \beta \mathcal{D}_{KL}(\pi_\theta || \pi_{ref})]$$

得到对应关系:
- $r(x,y) = \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)}$ — 这与Rafailov et al. (DPO) 提出的implicit reward形式完全一致 ([Rafailov et al. 2023](https://arxiv.org/abs/2305.18290))
- $\beta = 1$ — reward和KL约束**权重相等**, 这是OPD的"局限"

### 1.2 Dense Reward 的核心价值

标准RL的reward是sparse的:

$$r_t^{RL} = \begin{cases} 0 & t = 1, \ldots, T-1 \\ \text{Outcome Reward} & t = T \end{cases}$$

而OPD的token-level reward是dense的:

$$r_t^{OPD} = \log \frac{\pi^*(y_t | x, y_{<t})}{\pi_{ref}(y_t | x, y_{<t})}, \quad t = 1, \ldots, T$$

每个token都获得有效credit, 这解释了为什么OPD比RL更sample-efficient。这与 [Yuan et al. 2024](https://arxiv.org/abs/2412.01981) 的free process rewards思想同源, 但OPD不需要 $\pi^*$ 是从 $\pi_{ref}$ RL出来的, 两个model甚至可以不同size。

### 1.3 Flexible Reference Model

RL中 $\pi_{ref}$ 通常是policy的起点, 而在OPD中 $\pi_{ref}$ 可以是任意模型 — 因为无论选什么 $\pi_{ref}$, 目标函数都能simplify回原来的reverse KL形式。这是OPD相对于RL的第二个"自由度"。

## 2. G-OPD: 引入 Reward Scaling Factor λ

### 2.1 Generalized Formulation

$$\mathcal{I}_{G-OPD}(\theta) = \max_\theta \mathbb{E}_{x,y \sim \pi_\theta}\left[\lambda \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} - \mathcal{D}_{KL}(\pi_\theta(y|x) || \pi_{ref}(y|x))\right]$$

这里 $\lambda = \frac{1}{\beta}$, 控制reward term和KL term的相对权重。$\lambda = 1$ 就退化为标准OPD。

### 2.2 Closed-Form Optimal Solution

对 $\log \pi_\theta$ 求导等于0, 得到:

$$\log \pi_\theta(y|x) = \lambda \log \pi^*(y|x) + (1-\lambda) \log \pi_{ref}(y|x)$$

也可改写为:

$$\log \pi_\theta(y|x) = \log \pi^*(y|x) + (\lambda - 1)(\log \pi^* - \log \pi_{ref})$$

这是理解G-OPD的钥匙:

- **$0 < \lambda < 1$ (Reward Interpolation):** student log-prob是teacher和reference的log-linear插值。把reward看成 $\lambda \cdot r + (1-\lambda) \cdot 0$, 即"打折的reward"。student行为落在reference和teacher之间 — 性能和response length都单调插值。这正好可以用来做budget-controlled reasoning ([Yang et al. 2025e "Thinking-Optimal Scaling"](https://arxiv.org/abs/2502.18080), [Liang et al. Orbit](https://arxiv.org/abs/2601.08310))

- **$\lambda > 1$ (Reward Extrapolation, ExOPD):** student log-prob在teacher基础上, **额外** fitting一项 $(\lambda-1)(\log \pi^* - \log \pi_{ref})$, 即朝着"teacher远离reference的方向"再往前走一步。这是让student突破teacher性能边界的关键 — 我们下面看为什么。

### 2.3 等价形式与Reward Correction直觉

把KL基准从 $\pi_{ref}$ 换成 $\pi^*$, G-OPD可重写为:

$$\mathcal{I}_{G-OPD}(\theta) = \max_\theta \mathbb{E}_{x,y \sim \pi_\theta}\left[(\lambda-1)\log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} - \mathcal{D}_{KL}(\pi_\theta(y|x) || \pi^*(y|x))\right]$$

这个form下, 第一项就是"放大版的implicit reward", 第二项把student anchored到teacher附近。$\lambda > 1$ 等价于**放大teacher偏离reference的方向**。

在strong-to-weak distillation中, 如果我们能拿到teacher的pre-RL base model $\pi_{base}^{teacher}$, 用它当 $\pi_{ref}$ 比 $\pi_{base}^{student}$ 更合理:

$$\text{Default reward: } \log \frac{\pi^*}{\pi_{base}^{student}} \quad \text{(noisier, 因size gap)}
$$
$$\text{Corrected reward: } \log \frac{\pi^*}{\pi_{base}^{teacher}} \quad \text{(well-defined, 是teacher RL的implicit reward)}
$$

correction项是 $\log \frac{\pi_{base}^{student}}{\pi_{base}^{teacher}}$, 把reward校准到teacher's RL post-training的真实"reward field"。

### 2.4 Advantage 形式

实践中用discount factor=0 (next-token only, 见 [Lu & Thinking Machines Lab 2025](https://thinkingmachines.ai/blog/on-policy-distillation)), G-OPD的gradient近似为:

$$\nabla_\theta \mathcal{I}_{G-OPD}(\theta) = \mathbb{E}_{x,y \sim \pi_\theta}\left[\sum_{t=1}^{T} A_t^{G-OPD} \nabla_\theta \log \pi_\theta(y_t|x, y_{<t})\right]$$

其中token-level advantage:

$$A_t^{G-OPD} = \underbrace{(\log \pi_\theta(y_t|x,y_{<t}) - \log \pi^*(y_t|x,y_{<t}))}_{\text{standard OPD advantage}} + \underbrace{(\lambda-1)(\log \pi_{ref}(y_t|x,y_{<t}) - \log \pi^*(y_t|x,y_{<t}))}_{\text{extrapolation correction term}}$$

当 $\lambda = 1$, 第二项消失, 退化成OPD。$\lambda > 1$ 时, 第二项加进了"reference vs teacher"的额外信号, 把student推得更"远离reference、超越teacher"。

## 3. 实验深度解析

### 3.1 Setup

- **Base model:** Qwen3-4B-Non-Thinking ([Qwen3 Tech Report](https://arxiv.org/abs/2505.09388))
- **Domain teachers:** 在DeepMath-103K ([He et al. 2025](https://arxiv.org/abs/2504.11456), 难度≥6, 57K samples)上做math RL, 在Eurus-RL-Code ([Cui et al. 2025](https://arxiv.org/abs/2502.01456), 25K samples)上做code RL
- **RL algorithm:** GRPO ([Shao et al. DeepSeekMath](https://arxiv.org/abs/2402.03300)), 答对给1.0, 否则0.0
- **Benchmarks:** Math — AIME24, AIME25, HMMT25 Feb, HMMT25 Nov ([MathArena](https://matharena.ai/)); Code — HumanEval+, MBPP+, LiveCodeBench v6
- **Eval:** temperature=1.0, top-p=1.0, max length 16384, math每个问题采样32个, code采样4个, 用[Math-Verify](https://github.com/huggingface/Math-Verify)

### 3.2 Reward Scaling Factor λ 的扫描实验 (Section 4.1.2)

λ取值: $\{0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5\}$

**关键观察 (Figures 2, 3, 4):**

| λ | 行为 |
|---|---|
| 0.0 | 初始student (Qwen3-4B-Non-Thinking) |
| 0.25–0.75 | Performance和response length随λ单调上升, 在base和teacher之间 |
| 1.0 (标准OPD) | 完全recover teacher的behavior — accuracy和length都贴近teacher |
| 1.25 (ExOPD) | **在所有benchmark上超过teacher**, accuracy和length都更高 |
| 1.5 | Performance退化, 出现instability — 过度extrapolation导致reward hacking |

**为什么 λ=1.5 会退化?** 论文解释: 持续增大λ会让student去aggressively fit log ratio的peak, 即使某些token有异常大的log ratio (来自bias), 也会被fitting。同时response length持续增长, 来自implicit reward的length bias ([Yang et al. LASER 2025d](https://arxiv.org/abs/2510.14943))。

**对照实验 (Table 1):** 为排除"teacher训练不充分"的解释, 让teacher再训100步RL vs ExOPD训50步:

| Method | AIME24 | AIME25 | HMMT25 Feb | HMMT25 Nov | Avg |
|---|---|---|---|---|---|
| Teacher | 58.0 | 54.6 | 32.5 | 38.9 | 46.0 |
| Teacher + continued RL (100 steps) | 60.9 (+2.9) | 55.6 (+0.5) | 32.8 (+0.3) | 38.4 (-0.5) | 46.9 (+0.9) |
| **ExOPD (50 steps)** | **62.7 (+4.7)** | **56.1 (+1.5)** | **33.9 (+1.4)** | **39.3 (+0.4)** | **48.0 (+2.0)** |

ExOPD用更少step获得更大gain, 说明ExOPD的提升不来自"额外训练量"。

### 3.3 Multi-Teacher Distillation (Section 4.1.3)

这是论文最interesting的实验。把Math RL variant和Code RL variant两个teacher的能力merge回原student。

**Baselines:**
- SFT: 在teacher生成的trajectory上做Cross-Entropy loss
- ExPO ([Zheng et al. 2025](https://aclanthology.org/2025.acl-long.53/)): weight-space extrapolation, 先平均teacher权重, 再相对student外推, training-free但不可控
- OPD: 标准on-policy distillation
- ExOPD: λ=1.25

**结果 (Table 2):**

Single-Teacher (Math teacher only, Math benchmarks Avg):
- Teacher: 46.0
- ExPO: 45.8 (-0.2)
- OPD: 46.5 (+0.5)
- **ExOPD: 48.0 (+2.0)**

Multi-Teacher (两个teachers):
- SFT: 44.3 Math / 60.8 Code / 整体不如teacher
- ExPO: 45.0 Math / 62.6 Code / 部分超过teacher但不一致
- OPD: 46.4 Math / 60.6 Code / 性能ceiling受限于teacher
- **ExOPD: 47.7 Math / 62.0 Code / 唯一在所有benchmark上超越所有domain teachers的方法**

**Training dynamics (Figure 5):**
- ExOPD比OPD获得更高的training reward
- ExOPD的response length更长
- ExOPD的response entropy更高 (因为更长更多样)

这组dynamics说明ExOPD并不是简单地"机械模仿teacher", 而是探索出了更长的reasoning chain。

### 3.4 Strong-to-Weak Distillation (Section 4.2)

Teacher: Qwen3-30B-A3B-Instruct-2507 (MoE, 30B total, 3B active)
Students: Qwen3-1.7B-Non-Thinking, Qwen3-4B-Non-Thinking

**Table 3 关键结果:**

Student = Qwen3-1.7B:
- Base: 8.8 Avg
- SFT: 13.5
- OPD: 23.1
- **ExOPD: 25.4 (+2.3 over OPD)**

Student = Qwen3-4B:
- Base: 15.4
- SFT: 35.1
- OPD: 42.6
- **ExOPD: 45.3 (+2.7 over OPD)**

注意这里teacher达到59.7 Avg, OPD把4B student从15.4带到42.6 (巨大的gap bridging), ExOPD再额外+2.7, 是显著相对提升。这说明即便implicit reward $\log \frac{\pi^*}{\pi_{base}^{student}}$ 在跨size时有noise, extrapolation仍能push OPD的极限。

### 3.5 Reward Correction 实验 (Section 4.2.3)

由于拿不到30B teacher的pre-RL variant, 论文用Qwen3-4B-Non-Thinking-RL-Math/Code做teacher, Qwen3-4B-Non-Thinking做pre-RL variant, student是Qwen3-1.7B。

**Figure 6结果:** reward correction在math和code上都比default ExOPD (用student base做reference) 进一步提升。

代价: 需要 $\pi_{base}^{teacher}$ (额外模型), 且计算 $\log \pi_{base}^{teacher}$ 比 $\log \pi_{base}^{student}$ 贵 (因为teacher更大)。

## 4. 我对intuition的几点思考

### 4.1 ExOPD 为何能超越teacher?

从closed-form解 $\log \pi_\theta = \log \pi^* + (\lambda-1)(\log \pi^* - \log \pi_{ref})$ 看, student在teacher基础上沿"teacher偏移方向"再走一步。这有几种解释:

1. **Implicit reward的extrapolation:** 如果认为 $\log \frac{\pi^*}{\pi_{ref}}$ 是teacher学到的"什么是好token"的reward field, 那么ExOPD相当于让student沿这个reward gradient再多走几步, 在same-capacity regime下, 这相当于"更激进的RL", 自然能超越teacher。

2. **EMA/SWA analogy:** 类似 [Model Extrapolation (ExPO, Zheng et al. 2025)](https://aclanthology.org/2025.acl-long.53/) 在weight space做的事, ExOPD在distribution space做。当RL收敛到teacher的过程中"经过"了一个更好的点 (类似polyak averaging), ExOPD试图"过冲"来触及它。

3. **Length as lever:** 实验显示ExOPD生成更长response, 高entropy。可能更长的reasoning chain是math/code任务accuracy提升的直接来源 — 这和test-time scaling的intuition一致。

### 4.2 与RLHF/DPO的联系与区别

- DPO的implicit reward: $r(x,y) = \beta \log \frac{\pi_\theta}{\pi_{ref}} + \beta \log Z(x)$, 其中 $\pi_\theta$ 是post-RL policy。OPD用 $\pi^*$ 替换 $\pi_\theta$ 作reward source, 但不要求 $\pi^*$ 是从 $\pi_{ref}$ RL出来的 — 这解锁了cross-size distillation。

- DPO的closed-form解对应KL约束的stationary point, OPD正好利用这个connection, 在dense reward下做policy gradient。

- G-OPD的λ相当于inverse temperature, 控制reward regularization的相对强度, 这与 [DPO temperature](https://arxiv.org/abs/2305.18290)的β作用相反 — 这里是放大reward而非抑制。

### 4.3 与最近的On-Policy Distillation研究的脉络

- [Agarwal et al. 2024 (On-Policy Distillation of Language Models)](https://arxiv.org/abs/2406.05310) — 学习self-generated mistakes, 最早系统研究OPD
- [Gu et al. 2024 MiniLLM](https://openreview.net/forum?id=5h0qf7IBZZ) — reverse KL distillation
- [Lu & Thinking Machines Lab 2025](https://thinkingmachines.ai/blog/on-policy-distillation) — 近期thinking machines的blog, discount factor=0的实践基础
- [Xiao et al. 2026 MiMo-v2-Flash](https://arxiv.org/abs/2601.02780) — multi-task post-training via OPD merge
- [Yang et al. 2025a Qwen3](https://arxiv.org/abs/2505.09388) — strong-to-weak distillation

这篇G-OPD的定位是给"OPD为什么work"和"如何让OPD work better"提供理论统一框架和实践方案。

### 4.4 Limitations 与 open questions

论文明确指出:
1. Reward correction需要额外的 $\pi_{base}^{teacher}$, 而且computational cost更高 (compute $\log \pi_{ref}$ on larger model)
2. λ需要调, λ过大导致reward hacking和length explosion
3. Experiments限于math和code, 开放域alignment未验证

我觉得可以扩展的方向:
- **Cross-family distillation** (e.g. GPT → Llama): implicit reward $\log \frac{\pi^*}{\pi_{ref}}$ 的well-defined性还有待验证, reference model的选择更tricky
- **λ的schedule/curriculum**: 论文用固定λ=1.25, 也许annealing λ能避免后期的不稳定
- **Length bias的修正**: 加入length-normalized reward或KL, 防止extrapolation过度
- **Process-level reward integration**: dense reward目前来自log-prob ratio, 是否能融合rule-based process reward ([Cui et al. PRIME](https://arxiv.org/abs/2502.01456))
- **与Thinking Machines Lab的connection**: 他们独立发现OPD的RL interpretation, 这篇G-OPD和TML blog在conclusion层面互相印证, 可以互相参考 [Lu 2025](https://thinkingmachines.ai/blog/on-policy-distillation)

## 5. Takeaways

1. **OPD = KL-constrained dense RL with β=1, $r = \log \frac{\pi^*}{\pi_{ref}}$**, $\pi_{ref}$ 可任意选择
2. **G-OPD引入λ**, 等价于在log-prob空间做teacher/reference的log-linear组合, 最优解的closed-form直接揭示interpolation vs extrapolation的mechanism
3. **ExOPD (λ>1, typically 1.25)** 让student突破teacher性能边界, 是multi-teacher merge的利器
4. **Reward correction** 在strong-to-weak distillation中用teacher的pre-RL base作ref, 校准implicit reward到teacher的真实RL信号
5. **关键limitation**: λ需要调, 过度extrapolation会reward hacking; reference model selection在cross-size下影响noise

整体上, 这篇paper给OPD提供了一个clean的theoretical framework, 同时带来一个反直觉但强力的insight — student不仅可以match teacher, 还可以通过"放大teacher的方向"来超越teacher。这跟你之前在[education-related tweets](https://x.com/karpathy)中提到的"student should surpass teacher"的愿景在技术上呼应。

**相关reference links:**
- Paper arxiv: https://arxiv.org/abs/learning-beyond-teacher (需根据正式版本更新)
- Qwen3 report: https://arxiv.org/abs/2505.09388
- DPO: https://arxiv.org/abs/2305.18290
- ExPO: https://aclanthology.org/2025.acl-long.53/
- Thinking Machines Lab blog: https://thinkingmachines.ai/blog/on-policy-distillation
- DeepMath: https://arxiv.org/abs/2504.11456
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- PRIME (process reward): https://arxiv.org/abs/2502.01456
- LASER (length bias): https://arxiv.org/abs/2510.14943
- verl framework: https://arxiv.org/abs/2409.19256
- MathArena: https://matharena.ai/
- Math-Verify: https://github.com/huggingface/Math-Verify

如果你对某个具体方面想深挖, 例如gradient derivation的详细step-by-step, 或者与DPO/SWR/iSFT的关系, 我可以展开。
