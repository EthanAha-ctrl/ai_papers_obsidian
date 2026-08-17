---
source_pdf: Hybrid Policy Distillation for LLMs.pdf
paper_sha256: 50ae8a48b34e1fac9ac63898fa2dc8eb0808d84924a5460f1849483ffe4d4a01
processed_at: '2026-08-05T08:47:56-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 1. 这篇paper在解决什么问题？

你训练了一个7B的大模型，效果很好，但是太大太慢。你想把它distill到一个1.5B的小模型上，让小模型尽量学到big model的behavior。

最naive的方法就是**SFT**：拿teacher生成的output作为ground truth，让student去imitate。但是SFT有个问题——它只告诉你"这个token是对的"，却没告诉你"其他token有多不对"。就像考试只告诉你对错，不告诉你为什么错、错在哪里。

所以大家想到**Knowledge Distillation (KD)**：让student去match teacher的整个probability distribution，而不只是一个hard label。这样student就能学到更dense的signal。

但KD本身又有很多坑，这篇paper就是来填坑的。

参考：[Hinton的KD原始paper](https://arxiv.org/abs/1503.02531)

---

## 2. KD的两个方向：Forward KL和Reverse KL

KD的核心是让student distribution $q_\theta$ 去match teacher distribution $p$。但"match"有两种match法，对应两种KL divergence方向：

### Forward KL ($D_{KL}(p \| q_\theta)$)

这个方向是：**student必须覆盖teacher所有可能输出的mode**。

类比：teacher说"这个question有3个valid answers: A, B, C"，forward KL会让student给A、B、C都分配概率。

问题：如果student模型太小，capacity不够覆盖所有mode，它就会把概率"dilute"开，变成一个过度smooth的distribution——每个mode都有一点概率，但都不够concentrated。这叫**mode coverage but over-smoothing**。

### Reverse KL ($D_{KL}(q_\theta \| p)$)

这个方向是：**student只关注teacher高概率的mode**。

类比：teacher说"有A、B、C三个answers"，reverse KL会让student只学A（最高概率的），忽略B和C。

问题：如果student和teacher差很远，reverse KL的gradient会非常noisy、unstable，因为log-ratio term可以变得unbounded。这叫**mode-seeking but unstable**。

### 核心矛盾

Forward KL和Reverse KL各有优缺点：
- Forward KL: 稳定，但over-smooth
- Reverse KL: 精准，但unstable

**能不能同时拿到两者的好处？** 这就是HPD要解决的核心问题。

---

## 3. 之前的approaches有什么问题？

### 3.1 Naive的weighted sum不行

最直觉的想法：把两个KL加起来，$\alpha \cdot \text{FKLD} + (1-\alpha) \cdot \text{RKLD}$。

问题：在某些token上，forward KL说"增加这个token概率"，reverse KL说"减少这个token概率"——gradient方向冲突！你用一个固定系数$\alpha$，没法处理这种局部矛盾。

### 3.2 JSD（GKD用的）

Jensen-Shannon Divergence是forward和reverse KL的某种symmetric combination，用$M = \frac{p+q}{2}$作为中间分布。

问题：JSD仍然是symmetric的，weight是$\frac{1}{2}\log\frac{q}{M}$，不区分"student低估expert"和"student高估non-expert"这两种情况。

### 3.3 On-policy distillation (MiniLLM, GKD)

完全用student自己生成的sequence来训练，exposes student to its own distribution。

问题：computational overhead巨大，需要full sequence rollout（生成完整sequence再算reward）。对于long CoT reasoning task，一次rollout可能几千个token，cost很高。

参考：[MiniLLM](https://arxiv.org/abs/2306.08543), [GKD](https://arxiv.org/abs/2306.1365)

---

## 4. HPD的核心insight

HPD的insight可以用一句话概括：

**在token level，根据student和teacher的local gap sign，asymmetric地决定用forward KL还是reverse KL，避免gradient冲突。**

具体来说，对每个position $t$，HPD做两件事：

### 4.1 检查expert token $a_t^*$（ground truth或teacher生成的）

算一个gap指标：
$$k_1 = q_\theta(a_t^* | s_t) \cdot [\log p(a_t^* | s_t) - \log q_\theta(a_t^* | s_t)]$$

变量解释：
- $q_\theta(a_t^* | s_t)$: student给expert token的概率
- $\log p(a_t^* | s_t)$: teacher的log-prob
- $\log q_\theta(a_t^* | s_t)$: student的log-prob
- $k_1 > 0$: student低估了expert（teacher觉得应该给更高概率）
- $k_1 \leq 0$: student高估了expert

### 4.2 根据gap sign决定weight

**Case 1: $k_1 > 0$（student低估expert）**
- 用forward KL weight: $w_t^* = p(a_t^* | s_t) + k_1$
- 两个term都是正向的，都push student去增加expert token概率
- 这就是forward KL的mode coverage behavior——student应该cover这个mode

**Case 2: $k_1 \leq 0$（student高估expert）**
- Mask掉forward KL weight，只用reverse KL gap: $w_t^* = k_1$（负值）
- 负weight会suppress这个token的probability
- 这就是reverse KL的mode-seeking behavior——student应该stop over-confidently predicting this token

**关键**: 通过mask机制，HPD在每个token position动态选择forward或reverse KL，而不是用固定系数combine两者。

---

## 5. HPD的第二个trick：Student Sampling

光处理expert token还不够。student可能在某些non-expert token上over-confident，但你不检查就不知道。

所以HPD在每个position $t$，额外从student自己sample一个token $a_t \sim q_\theta(\cdot | s_t)$，检查这个token的gap：

$$k_1' = q_\theta(a_t | s_t) \cdot [\log p(a_t | s_t) - \log q_\theta(a_t | s_t)]$$

**Case A: $k_1' \geq 0$（student低估non-expert）**
- $w_t = 0$，mask掉
- 因为这个token不是expert，你不应该强化它，即使student低估

**Case B: $k_1' < 0$（student高估non-expert）**
- $w_t = k_1'$（负值）
- Suppress这个non-expert token

### 5.1 Reinforce Operation

最巧妙的部分：当**同时**发生"student低估expert（$k_1 > 0$）"和"student高估non-expert（$k_1' < 0$）"时，HPD对expert token的forward KL weight加倍：

$$w_t^* = 2p(a_t^* | s_t) + k_1$$

**Intuition**: 当你suppress non-expert token，会释放出一些probability mass。HPD明确地把这些mass redirect到expert token上，通过加倍forward KL weight。这是一种"probability budget redistribution"。

如果naive的weighted sum，suppress non-expert释放的mass会按student当前distribution随机分配，可能分到别的non-expert token上。HPD的reinforce operation确保mass流向正确的地方。

---

## 6. 统一视角：Reweighted Log-Likelihood

paper一个重要的理论贡献是：**SFT、FKLD、RKLD都可以看作reweighted log-likelihood**：

$$\mathcal{L}(\theta) = -\mathbb{E}_{(s_t, a_t) \sim \mathcal{D}^\pi}\Big[w(a_t | s_t) \log q_\theta(a_t | s_t)\Big]$$

| Method | $w(a_t \| s_t)$ |
|--------|-----------------|
| SFT | $\mathbf{1}[a_t = a_t^*]$ |
| FKLD | $p(a_t \| s_t)$ |
| RKLD | $\log p(a_t \| s_t) - \log q_\theta(a_t \| s_t)$ |
| **HPD** | **dynamic mask-based hybrid** |

**Intuition**: 所有KD方法本质都在maximize log-likelihood，区别只在weight $w$怎么设计：
- SFT: weight只在expert token上为1，其他都为0（sparse supervision）
- FKLD: weight是teacher的概率$p$（dense supervision，覆盖所有modes）
- RKLD: weight是log-ratio（dense，但focus on高概率modes）
- HPD: 根据gap sign动态hybridize

这个unified view很powerful，因为：
1. 它让你把所有KD方法放在一个framework下比较
2. 它告诉你设计新KD方法就是设计新的$w$
3. 它让gradient分析变得清晰

---

## 7. Gradient为什么会work？

paper给了一个关键的gradient分析。对reweighted objective $\mathcal{L} = -w_t \log q_\theta(a_t | s_t)$，对logit $z_v$的gradient是：

$$-\frac{\partial \mathcal{L}}{\partial z_v} \propto \begin{cases} \hat{w}_t \cdot q_v \cdot (1 - q_v), & \text{if } v = a_t \\ -\hat{w}_t \cdot q_{a_t} \cdot q_v, & \text{if } v \neq a_t \end{cases}$$

变量解释：
- $z_v$: token $v$的logit
- $q_v = \text{softmax}(z_v)$: token $v$的probability
- $\hat{w}_t = w(a_t | s_t)$: reweighting term

**关键insight**: 虽然objective只target一个token $a_t$，但gradient会propagate到整个vocabulary！

当$\hat{w}_t > 0$（比如forward KL）：
- 对target token $a_t$的logit gradient为正 → 增加其概率
- 对其他token $v \neq a_t$的logit gradient为负 → 减少其概率

当$\hat{w}_t < 0$（比如reverse KL抑制over-estimation）：
- 对target token $a_t$的logit gradient为负 → 减少其概率
- 对其他token $v \neq a_t$的logit gradient为正 → **按当前$q_v$比例增加概率**

所以reverse KL的suppression不是把probability mass随机丢掉，而是按当前distribution proportionally重新分配。HPD的reinforce operation就是override这个default behavior，强制把mass redirect到expert token上。

---

## 8. 为什么不用$K_1$ as loss而用as reward？

这是个subtle但重要的point。paper在Appendix B.1分析了：

如果$K_1$作为loss（path-wise derivative）：
$$\mathbb{E}_{\tau \sim q_\theta}\Big[\nabla_\theta \sum_t K_{1t}\Big] = 0$$

gradient是0！完全没用。

如果$K_1$作为reward（score function derivative）：
$$\mathbb{E}_{\tau \sim q_\theta}\Bigg[\log \frac{q_\theta(\tau)}{p(\tau)} \cdot \nabla_\theta \log q_\theta(\tau)\Bigg]$$

这个才提供unbiased gradient estimate for KLD objective。

**所以HPD把$K_1$作为reward signal使用**，放在weight $w$的位置上，通过$\nabla_\theta \log q_\theta$来propagate gradient。这是RL中的standard trick（policy gradient），但应用到distillation上。

参考：[Schulman的KL approx blog](http://joschu.net/blog/kl-approx.html)

---

## 9. HPD的computational cost

HPD vs full OPD的cost对比：

**Full OPD (MiniLLM, GKD)**:
- 需要generate full sequence from student: $O(T \cdot |V|)$ where $T$是sequence length
- 对每个generated sequence算teacher log-prob: $O(T \cdot |V|)$
- 对long CoT ($T \approx 8000$)，这个cost巨大

**HPD**:
- 对每个position $t$，只sample 1个extra token from student: $O(|V|)$
- 算这个token的teacher log-prob: $O(|V|)$
- 总cost: $O(T \cdot |V|)$，但常数因子小很多，不需要autoregressive generation

所以HPD叫"lightweight on-policy sampling"——它只sample next token，不sample full sequence。这捕获了on-policy的大部分benefit（expose student to its own distribution），但avoid了full rollout的cost。

---

## 10. 实验结果为什么这么强？

### 10.1 Math reasoning (Table 2)

最impressive的结果：

**Qwen 2.5 7B → 3B**:
- SFT: Avg 32.67
- JSD (best baseline): 36.61
- **HPD: 39.83** (比JSD高3.22，比SFT高7.16)

**LLaMA 3 8B → 3B**:
- SFT: Avg 29.45
- JSD: 29.95
- **HPD: 34.56** (比JSD高4.61，比SFT高5.11)

HPD在所有benchmark上一致beat所有baselines，且在更小的模型上improvement更大（1.5B/1B的improvement比3B更显著）。这符合intuition——小模型capacity更受限，更受益于HPD的precise distribution shaping。

### 10.2 为什么HPD作为OPD初始化更好？

Table 5展示了HPD + OPD的结果：

| Method | Avg. |
|--------|------|
| SFT | 22.80 |
| SFT + OPD | 29.56 |
| HPD | 30.24 |
| **HPD + OPD** | **33.41** |

**HPD alone > SFT + OPD**！这是很强的claim。说明HPD的distribution alignment质量比SFT好太多，甚至超过two-stage baseline。

**HPD + OPD > HPD alone**，说明OPD还能在HPD基础上继续improve。HPD提供了一个更好的起点（更低的KL divergence，更stable的entropy），让OPD的policy gradient更stable、更efficient。

Figure 2的training dynamics很说明问题：
- HPD initialization的mean advantage更stable、less negative
- HPD initialization的KL divergence持续保持更低
- HPD initialization的test-time entropy没有collapse

### 10.3 为什么HPD作为DPO初始化更好？

Table 6:
| Method | Arena-WR | Δ |
|--------|----------|---|
| SFT + DPO | 10.40 | +1.81 |
| RKLD + DPO | 21.80 | +9.74 |
| **HPD + DPO** | **25.10** | **+11.92** |

HPD + DPO的gain最大（+11.92），比RKLD + DPO还高。原因：HPD避免了entropy collapse，保留了more diverse distribution，给DPO更多room去optimize preference。SFT会overfit到一个narrow distribution，DPO能做的adjustment就有限。

参考：[DPO paper](https://openreview.net/forum?id=HPuSIXJaa9)

---

## 11. Ablation study告诉我们什么？

Figure 3的ablation：

### 11.1 去掉Student Sampling
- Performance快速plateau
- 说明direct optimize toward teacher distribution会premature convergence
- Student sampling让model explore its own distribution，发现blind spots

### 11.2 去掉Reinforce Operation
- KL loss下降变慢
- 说明当suppress non-expert token时，必须明确redirect mass到expert，否则optimization signal不够efficient

两个component都necessary，且complementary：student sampling负责"发现问题"，reinforce负责"修正问题"。

---

## 12. 我的intuition总结

### 12.1 KD的本质是distribution shaping

不要把KD看作简单的"让student学teacher的output"。KD是在一个high-dimensional distribution space里做shaping，关键是how to efficiently allocate probability mass。

SFT是sparse supervision，只在expert token上给signal。KD提供dense supervision，但dense signal可能misleading（比如over-smoothing或unstable）。

HPD的contribution是提供了一种**adaptive dense supervision**——dense like KD，但根据local situation dynamically adjust，避免naive dense supervision的pitfalls。

### 12.2 Asymmetric design > Symmetric design

很多ML方法喜欢symmetric design（fixed coefficient, uniform treatment）。但现实中，不同sample、不同position需要不同的处理。

HPD的asymmetric masking告诉我们：**根据local signal的sign做conditional decision，比用一个固定公式apply到所有情况更effective**。

这个insight可能extend到其他ML问题：
- Curriculum learning: 根据model当前state动态调整difficulty
- Active learning: 根据model uncertainty选择性label
- RL: 根据value function的sign决定explore还是exploit

### 12.3 Lightweight on-policy captures大部分benefit

OPD的benefit来自让student "see its own mistakes"。HPD告诉我们：**不一定需要full sequence rollout，next-token sampling就能expose大部分mistakes**。

这对future work有implication：很多on-policy method的cost可能可以通过lightweight approximation大幅reduce，while保留大部分benefit。

### 12.4 Reward-based > Loss-based for KL estimation

从gradient analysis看，把$K_1$作为reward（通过score function derivative）是unbiased的，作为loss（path-wise derivative）gradient是0。

这解释了为什么PPO、GRPO等RL方法的KL penalty用reward formulation——不是任意选择，是mathematical requirement。HPD把这个insight从RL extend到distillation。

参考：[PPO](https://arxiv.org/abs/1707.06347), [GRPO](https://arxiv.org/abs/2402.03300)

---

## 13. 一个具体的walkthrough

假设我们在train一个math reasoning model，当前position $t$的前缀是："The answer is "。

**Teacher distribution**:
- $p(\text{"42"} | s_t) = 0.8$
- $p(\text{"the"} | s_t) = 0.1$
- $p(\text{"approximately"} | s_t) = 0.05$
- $p(\text{other} | s_t) = 0.05$

**Student distribution (current)**:
- $q_\theta(\text{"42"} | s_t) = 0.5$ （underestimate expert）
- $q_\theta(\text{"the"} | s_t) = 0.3$ （overestimate non-expert）
- $q_\theta(\text{"approximately"} | s_t) = 0.1$
- $q_\theta(\text{other} | s_t) = 0.1$

**Step 1: 检查expert token "42"**
- $k_1 = 0.5 \cdot [\log 0.8 - \log 0.5] = 0.5 \cdot 0.47 = 0.235 > 0$
- Student低估了expert！进入forward KL mode
- $w_t^* = 0.8 + 0.235 = 1.035$ (正向reinforce "42")

**Step 2: Sample non-expert token from student**
- 假设sampled $a_t = $ "the" (prob 0.3)
- $k_1' = 0.3 \cdot [\log 0.1 - \log 0.3] = 0.3 \cdot (-1.1) = -0.33 < 0$
- Student高估了non-expert "the"！

**Step 3: Apply Reinforce**
- 因为 $k_1 > 0$ 且 $k_1' < 0$，触发reinforce
- $w_t^* = 2 \cdot 0.8 + 0.235 = 1.835$（加倍forward KL weight on "42"）
- $w_t = -0.33$（suppress "the"）

**Step 4: Gradient effect**
- 对"42"的logit: positive gradient → increase prob
- 对"the"的logit: negative gradient → decrease prob
- 对其他token的logit: 因为$w_t^*$是正的，按$q_v$比例decrease；同时$w_t$是负的，按$q_v$比例increase。Net effect取决于具体情况。

但因为有reinforce，"42"的gradient特别强，确保释放的mass主要流向"42"。

**对比naive weighted sum**:
- Forward KL weight on "42": 0.8
- Reverse KL weight on "the": $\log 0.3 - \log 0.1 \cdot 0.3 = -0.33$ (approximated)
- 如果固定系数$\alpha = 0.5$，net weight on "42" = $0.5 \cdot 0.8 = 0.4$，比HPD的1.835弱很多

这就是HPD的power——根据local situation adaptively amplify signal。

---

## 14. 局限和future work

### 14.1 Limitations
1. **Tokenizer必须match**: HPD要求teacher和student共享tokenizer，cross-tokenizer KD不行
2. **仍然是estimator**: $K_1$是MC estimator，有variance
3. **不是full OPD**: 只sample 1 token，不是full sequence rollout，对long-horizon credit assignment可能不够

### 14.2 Future directions

paper提到可以apply到mid-training或pre-training。我的想法：

1. **HPD during pre-training**: 用large model作为teacher，small model作为student，在pre-training阶段就做HPD，可能比post-hoc distillation更effective
2. **Multi-teacher HPD**: 不同teacher擅长不同domain，可以设计multi-teacher的HPD variant
3. **MoE distillation**: dense teacher distill到MoE student，每个expert handle不同modes，HPD的mode-seeking behavior可能特别适合
4. **HPD for RL**: 把HPD的asymmetric masking思想apply到PPO的KL penalty上，可能improve RL stability

参考：[DAPO](https://arxiv.org/abs/2503.14476), [DAPO-MATH](https://arxiv.org/abs/2503.14476)

---

## 15. 一句话总结

**HPD = Token-level, asymmetric, mask-based hybridization of forward and reverse KL, with lightweight on-policy sampling, formulated as reweighted log-likelihood with dynamic weights.**

它的power来自三个insight的combination：
1. 根据$k_1$ sign动态选择forward/reverse KL
2. Student sampling expose mistakes，reinforce operation redirect mass
3. Reward-based formulation for unbiased gradient

实验证明这个combination在reasoning、dialogue、code三个domain都consistent outperform baselines，且作为OPD和DPO的初始化都有significant gains。

对于想build intuition的朋友，关键要记住：**KD不是简单的imitation，是distribution shaping under capacity constraint**。如何smartly allocate limited probability mass是核心问题，HPD给了一个elegant的答案。

参考资源：
- [Paper GitHub](https://github.com/zwhong714/Hybrid-Policy-Distillation)
- [Schulman KL approx](http://joschu.net/blog/kl-approx.html)
- [MiniLLM](https://arxiv.org/abs/2306.08543)
- [GKD](https://arxiv.org/abs/2306.1365)
- [DistiLLM-2](https://openreview.net/forum?id=rc65N9xIrY)
- [ABKD](https://openreview.net/forum?id=vt65VjJakt)
- [Thinking Machines OPD blog](https://thinkingmachines.ai/blog/on-policy-distillation)
- [DPO](https://openreview.net/forum?id=HPuSIXJaa9)
- [PPO](https://arxiv.org/abs/1707.06347)
- [GRPO](https://arxiv.org/abs/2402.03300)
- [Hinton KD](https://arxiv.org/abs/1503.02531)

---

# Hybrid Policy Distillation for LLMs - 深度技术解析

## 1. Paper核心定位与Motivation

这篇paper要解决的核心问题是：**Knowledge Distillation (KD) for LLMs的设计空间是高度耦合的，现有方法往往只关注某一个维度而忽视了其他维度的相互作用**。作者识别出三个critical design axes：

1. **Divergence direction**: forward KL (FKLD) vs reverse KL (RKLD)
2. **Optimization strategy**: loss-based vs reward-based
3. **Data regime**: off-policy vs on-policy

这三个维度是intertwined的，单独优化任何一个维度都不够。HPD的核心insight就是：在token level同时hybridize这些维度，同时保留one-hot supervision的computational efficiency。

参考：[Knowledge Distillation综述](https://arxiv.org/abs/1503.02531), [MiniLLM](https://arxiv.org/abs/2306.08543), [GKD](https://arxiv.org/abs/2306.08543)

---

## 2. 统一的Reweighted Log-Likelihood视角

### 2.1 统一目标公式

paper最关键的理论贡献是这个unified view：

$$\mathcal{L}(\theta) = \min_\theta -\mathbb{E}_{(s_t, a_t) \sim \mathcal{D}^\pi}\Big[w(a_t | s_t) \log q_\theta(a_t | s_t)\Big] \tag{9}$$

变量解释：
- $\theta$: student model parameters
- $s_t$: state (prefix tokens), $s_t = \mathbf{a}_{<t}^* = (a_1^*, \ldots, a_{t-1}^*)$
- $a_t$: action (next token to predict)
- $\mathcal{D}^\pi$: data source, 可以是offline dataset $\mathcal{D}$, teacher policy $\mathcal{D}^{\pi_T}$, 或student policy $\mathcal{D}^{\pi_\theta}$
- $q_\theta(a_t | s_t)$: student policy (softmax over vocabulary)
- $w(a_t | s_t)$: **reweighting term**，这是关键，不同方法对应不同的$w$

### 2.2 不同方法的$w$对比（Table 1）

| Method | Data source | $w(a_t | s_t)$ |
|--------|-------------|----------------|
| SFT | Off-policy | $\mathbf{1}[a_t = a_t^*]$ |
| FKLD | Off-policy | $p(a_t | s_t)$ |
| RKLD | On-policy | $\log p(a_t | s_t) - \log q_\theta(a_t | s_t)$ |

**Intuition**: 
- SFT是one-hot的indicator function，只在ground-truth token上给1
- FKLD用teacher的概率$p(a_t|s_t)$作为soft weight，覆盖所有modes
- RKLD用log-ratio作为weight，强调mode-seeking

### 2.3 Gradient传播机制（Eq. 10）

$$-\frac{\partial \mathcal{L}(\theta)}{\partial z_v} \propto \begin{cases} \hat{w}_t \cdot q_v \cdot (1 - q_v), & \text{if } v = a_t \\ -\hat{w}_t \cdot q_{a_t} \cdot q_v, & \text{if } v \neq a_t \end{cases} \tag{10}$$

变量解释：
- $z_v$: logit of token $v$
- $q_v = q_\theta(v | s_t)$: probability of token $v$ under student
- $\hat{w}_t = w(a_t | s_t)$: reweighting term

**这个公式特别关键**：虽然objective只target一个token $a_t$，但gradient会propagate到整个vocabulary分布。当$\hat{w}_t < 0$时：
- 对sampled token $a_t$的logit梯度为负 → 抑制该token
- 对其他token $v \neq a_t$的logit梯度为正 → 按当前概率$q_v$比例重新分配probability mass

这就是reverse KL的mode-seeking behavior的本质机制。

---

## 3. Monte Carlo $K_1$ Estimator

### 3.1 定义

由于精确计算KL divergence在大vocabulary下不可行，paper采用Schulman的$K_1$ estimator：

$$K_1 \triangleq \frac{1}{N}\sum_{i=1}^N \log \frac{q_\theta(a_t^{(i)} | s_t)}{p(a_t^{(i)} | s_t)}, \quad a_t^{(i)} \sim q_\theta(\cdot | s_t) \tag{8}$$

这是$\mathbb{D}_{KL}(q_\theta \| p)$的**unbiased estimator**，但variance高，因为log-ratio在很多样本上是负的。

### 3.2 $K_1$ as Reward vs Loss的Gradient分析（Appendix B.1）

这是一个很重要的subtlety。如果trajectory $\tau \sim q_\theta$，KL estimator的gradient有两种方式：

**Path-wise derivative** (用作loss):
$$\mathbb{E}_{\tau \sim q_\theta}\Big[\nabla_\theta \sum_t K_{1t}\Big] = 0$$

**Score function derivative** (用作reward):
$$\mathbb{E}_{\tau \sim q_\theta}\Bigg[\Big(\sum_t K_{1t}\Big) \cdot \nabla_\theta \log q_\theta(\tau)\Bigg] = \mathbb{E}_{\tau \sim q_\theta}\Bigg[\log \frac{q_\theta(\tau)}{p(\tau)} \cdot \nabla_\theta \log q_\theta(\tau)\Bigg] \tag{19}$$

**结论**: 把$K_1$作为reward使用才能提供KLD objective的unbiased gradient estimate，直接用作loss则不行。这就是为什么HPD采用reward-based的formulation。

参考：[Schulman's KL approx blog](http://joschu.net/blog/kl-approx.html)

---

## 4. HPD算法核心设计

### 4.1 Expert Token的$k_1$ Gap Estimator

对expert token $a_t^*$（来自teacher或offline ground truth），定义：

$$k_1 = q_\theta(a_t^* | s_t)\Big[\log p(a_t^* | s_t) - \log q_\theta(a_t^* | s_t)\Big] \tag{11}$$

变量解释：
- $q_\theta(a_t^* | s_t)$: student给expert token的概率
- $\log p(a_t^* | s_t)$: teacher给expert token的log-prob
- $\log q_\theta(a_t^* | s_t)$: student给expert token的log-prob
- $k_1 > 0$: **student低估了expert token**（$p > q$）
- $k_1 \leq 0$: **student高估了expert token**（$q \geq p$）

### 4.2 Hybrid Forward-Reverse KL Weight设计

这是HPD最巧妙的设计。核心思想是：**用mask机制避免forward KL和reverse KL的gradient冲突**。

**第一版权重（Eq. 12）**：
$$w_t^* = \begin{cases} p(a_t^* | s_t) + k_1, & \text{if } k_1 > 0 \\ k_1, & \text{if } k_1 \leq 0 \end{cases}$$

**Intuition**: 
- 当$k_1 > 0$（student低估expert），用forward KL weight $p(a_t^*|s_t)$加上reverse KL gap $k_1$，两者方向一致，都增加expert token的概率
- 当$k_1 \leq 0$（student高估expert），mask掉forward KL weight，只用reverse KL gap抑制过度估计，避免gradient方向冲突

### 4.3 Student-Sampled Non-Expert Token处理

paper还引入了student自己sample的non-expert token $a_t \sim q_\theta(\cdot|s_t)$, $a_t \neq a_t^*$，计算其gap：

$$k_1' = q_\theta(a_t | s_t)\Big[\log p(a_t | s_t) - \log q_\theta(a_t | s_t)\Big]$$

**mask策略（Eq. 13）**：
$$w_t \gets \begin{cases} 0, & \text{if } k_1' \geq 0 \\ k_1', & \text{if } k_1' < 0 \end{cases}$$

**Intuition**:
- 当$k_1' \geq 0$（student低估non-expert token，但这个token不是expert），mask掉，避免强化non-expert
- 当$k_1' < 0$（student高估non-expert token），用负权重suppress它

### 4.4 Reinforce Operation（Eq. 14）

当suppress了non-expert token（$k_1' < 0$），同时expert token也被低估（$k_1 > 0$）时，**对expert token的forward KL weight加倍**：

$$w_t^* = \begin{cases} 2p(a_t^* | s_t) + k_1, & \text{if } k_1 > 0 \text{ and } k_1' < 0 \\ k_1, & \text{if } k_1 < 0 \\ p(a_t^* | s_t) + k_1, & \text{otherwise} \end{cases} \tag{14}$$

**Intuition**: 当suppress non-expert token释放出probability mass时，明确地redirect这部分mass到expert token上。这是HPD相比简单weighted sum的关键创新——**asymmetric reweighting**，根据gap的符号动态调整。

### 4.5 最终HPD Objective（Eq. 15）

$$\mathcal{L}_{HPD} = \min_\theta \mathbb{E}_{(s_t, a_t^*) \sim \mathcal{D}, a_t \sim q_\theta(\cdot|s_t)}\Big[-w_t^* \log q_\theta(a_t^* | s_t) - w_t \log q_\theta(a_t | s_t)\Big] \tag{15}$$

注意这里有两个term：
- 第一个term针对expert token，权重$w_t^*$可以为正或负
- 第二个term针对student-sampled non-expert token，权重$w_t$只为负（或0）

### 4.6 Algorithm 1解析

```
1: input student qθ, teacher p, dataset D
2: Sample offline trajectories T ~ D
3: for each (st, at*) ∈ T do
4:   Compute log-probabilities: (log q*, log p*) ← (log qθ, log p)(at*|st)
5:   Compute expert reverse-KL gap: k1 ← qθ(at*|st)(log p* - log q*)
6:   Sample at ~ qθ(·|st)
7:   Compute sampled-token reverse-KL gap: k1' ← qθ(at|st)[log p(at|st) - log qθ(at|st)]
8:   Compute expert weight wt* based on Eq. (14)
9:   Compute sampled-token weight wt = I[at ≠ at*] · I[k1' < 0] · k1'
10: end for
11: Update parameters: θ ← θ - α∇θ L_HPD
```

**Computational cost**: 每个token只需要2次teacher forward（expert token和sampled token），比full KL divergence的计算（需要sum over vocabulary）便宜很多。

---

## 5. 实验结果深度分析

### 5.1 Off-policy Data for Reasoning (Table 2)

**Qwen 2.5 (7B → 1.5B)**:
| Method | AIME24 | AIME25 | AMC | Math | Obly. | GPQA | Avg. |
|--------|--------|--------|-----|------|-------|------|------|
| Teacher (M_T) | 28.13 | 27.19 | 71.72 | 87.48 | 58.50 | 43.43 | 52.74 |
| Student init (M_S) | 2.19 | 1.04 | 21.17 | 46.78 | 16.52 | 23.04 | 18.46 |
| SFT | 2.81 | 6.04 | 28.83 | 55.25 | 24.87 | 19.02 | 22.80 |
| SeqKD | 5.31 | 5.31 | 33.83 | 60.28 | 29.48 | 23.42 | 26.27 |
| RKLD | 5.00 | 3.85 | 34.45 | 58.78 | 27.41 | 27.40 | 26.15 |
| JSD | 5.73 | 4.90 | 35.31 | 59.63 | 27.30 | 25.69 | 26.43 |
| **HPD** | **7.71*** | **9.89*** | **39.84*** | **63.40*** | **32.53*** | **28.09*** | **30.24** |

**Qwen 2.5 (7B → 3B)**:
| Method | AIME24 | AIME25 | AMC | Math | Obly. | GPQA | Avg. |
|--------|--------|--------|-----|------|-------|------|------|
| M_S | 6.67 | 2.50 | 38.20 | 64.08 | 28.17 | 29.86 | 28.25 |
| SFT | 10.10 | 12.60 | 46.33 | 69.78 | 36.89 | 20.33 | 32.67 |
| SeqKD | 11.56 | 14.48 | 47.66 | 74.48 | 40.48 | 24.62 | 35.55 |
| RKLD | 9.38 | 12.29 | 46.25 | 69.58 | 37.35 | 19.51 | 32.39 |
| JSD | 10.31 | 14.90 | 50.70 | 73.88 | 40.69 | 29.17 | 36.61 |
| **HPD** | **13.75*** | **18.13*** | **54.14*** | **76.30*** | **45.33*** | **31.31*** | **39.83** |

**Key Observations**:
1. HPD在所有benchmark上一致outperform所有baselines
2. 3B模型平均提升41.0%（28.25 → 39.83）
3. LLaMA 3 (8B → 3B)提升77.9%（19.43 → 34.56），更显著的improvement
4. 即使在out-of-domain (GPQA)上也有显著提升，说明HPD transfer了更generalizable的decision signals

### 5.2 Training Dynamics (Figure 1)

**Figure 1a - Training-time Entropy**: 
- SFT: 快速entropy collapse（overfitting）
- HPD: 稳定entropy，no collapse

**Figure 1b - KLD Gap**:
- SFT: KL divergence gap停滞
- HPD: 持续减少gap到teacher distribution

**Figure 1c - Performance**:
- SFT: 性能停滞
- HPD: 持续提升性能

**Figure 1d - Test-time Entropy**:
- HPD的inference-time entropy与teacher高度对齐
- SFT存在train-inference mismatch

### 5.3 Off-policy Data for Personalization (Table 3)

| Method | AE-LC(%) | AE-WR(%) | Arena-WR(%) | MT-1T | MT-2T |
|--------|----------|----------|-------------|-------|-------|
| Teacher | 36.04 | 34.95 | 60.00 | 9.00 | 7.44 |
| SFT | 12.74 | 13.72 | 18.10 | 6.80 | 4.81 |
| SeqKD | 7.83 | 9.51 | 15.40 | 6.24 | 4.15 |
| RKLD | 11.26 | 12.00 | 17.80 | 6.96 | 5.19 |
| JSD | 13.48 | 13.89 | 20.20 | 6.96 | 5.21 |
| **HPD** | **13.75** | **14.25** | **21.80** | **7.23** | **5.84** |

HPD在multi-turn dialogue (MT-1T, MT-2T)上优势明显，说明它更好地保留了conversational coherence。

### 5.4 Off-policy Data for Coding (Table 4)

| Method | DS-Coder HEval | DS-Coder MBPP | DS-Coder AVG | Qwen-Coder HEval | Qwen-Coder MBPP | Qwen-Coder AVG |
|--------|----------------|---------------|--------------|------------------|-----------------|----------------|
| Teacher | 76.20 | 74.90 | 75.55 | 91.50 | 82.30 | 86.90 |
| SFT | 61.00 | 61.90 | 61.45 | 73.80 | 67.70 | 70.75 |
| KD | 65.20 | 64.00 | 64.60 | 77.40 | 67.50 | 72.45 |
| RKLD | 61.60 | 61.60 | 61.60 | 76.80 | 74.90 | 75.85 |
| JSD | 67.10 | 61.10 | 64.10 | 77.40 | 74.60 | 76.00 |
| **HPD** | **69.50** | **63.20** | **66.35** | **79.30** | **75.40** | **77.35** |

HPD在两个model family的平均性能最好，且variance更低，说明更robust。

### 5.5 On-policy Data for Reasoning (Table 5)

| Method | AIME24 | AIME25 | AMC | Math | Obly. | GPQA | Avg. |
|--------|--------|--------|-----|------|-------|------|------|
| SFT | 2.81 | 6.04 | 28.83 | 55.25 | 24.87 | 19.02 | 22.80 |
| SFT + OPD | 6.98 | 8.33 | 39.30 | 63.88 | 32.94 | 25.95 | 29.56 |
| HPD | 7.71 | 9.89 | 39.84 | 63.40 | 32.53 | 28.09 | 30.24 |
| **HPD + OPD** | **10.63*** | **10.10*** | **43.98*** | **69.93*** | **38.59*** | 27.21 | **33.41** |

**关键发现**: 
1. **纯off-policy HPD > SFT + OPD**: HPD单独使用就超过了传统two-stage baseline
2. **HPD + OPD进一步放大收益**: 说明HPD提供了更好的initialization for subsequent OPD training
3. Figure 2显示HPD的mean advantage更stable，KL divergence更低，test-time entropy无collapse

### 5.6 Ablation Study (Figure 3)

**Student Sampling的effect**:
- 没有student sampling: 性能快速收敛但plateau，premature convergence
- 有student sampling: 持续performance gains，允许exploration

**Reinforce Operation的necessity**:
- 移除Reinforce: KL loss下降变慢
- 有Reinforce: 更稳定的optimization signal，加速teacher distribution alignment

### 5.7 Broader Impact - HPD + DPO (Table 6)

| Method | AE-LC(%) | AE-WR(%) | Arena-WR(%) | Δ |
|--------|----------|----------|-------------|---|
| SFT | 10.10 | 7.36 | 7.20 | - |
| SFT + DPO | 10.42 | 9.27 | 10.40 | +1.81 |
| RKLD | 11.13 | 9.35 | 13.80 | +3.21 |
| RKLD + DPO | 15.45 | 16.78 | 21.80 | +9.74 |
| HPD | 13.78 | 10.88 | 15.80 | +5.27 |
| **HPD + DPO** | **17.68** | **17.65** | **25.10** | **+11.92** |

HPD作为DPO的初始化，gains最大（+11.92），说明HPD避免了entropy collapse，为后续alignment阶段提供了更好的起点。

### 5.8 Iterative Self-Distillation (Table 7)

| Method | AE-LC(%) | AE-WR(%) | Arena-WR(%) | Δ |
|--------|----------|----------|-------------|---|
| SFT | 10.10 | 7.36 | 7.20 | - |
| SFT + DPO | 10.42 | 9.27 | 10.40 | +1.81 |
| HPD-iter1 | 11.77 | 9.76 | 12.50 | +3.12 |
| HPD-iter1 + DPO | 13.67 | 13.83 | 16.30 | +6.38 |
| HPD-iter2 | 13.31 | 13.30 | 19.00 | +6.98 |
| HPD-iter2 + DPO | 14.06 | 13.82 | 20.60 | +7.94 |

Iterative HPD + DPO可以实现performance scaling，但gains逐渐saturate。

---

## 6. 与Related Work的对比

### 6.1 Forward KL的问题
- **Mode coverage**: FKLD强制student覆盖teacher所有modes
- **Over-smoothing**: 当student capacity有限，dilute probability mass到太多modes
- **Gradient**: $-\nabla_\theta \log q_\theta(a|s)$对$a \sim p$的所有modes都增加概率

### 6.2 Reverse KL的问题
- **Mode-seeking**: RKLD让student集中在teacher的高概率modes
- **Instability**: 当student-teacher gap大时，high-variance gradients from unbounded log-ratio
- **Gradient**: 包含$\log q_\theta - \log p$项，unbounded

### 6.3 JSD (GKD)
- 介于FKLD和RKLD之间
- Weight: $w_{JSD}(a_t|s_t) = \frac{1}{2}\log\frac{q_\theta(a_t|s_t)}{M(a_t|s_t)}$, $M = \frac{p+q}{2}$
- 仍然有fixed coefficient，不能根据gap sign动态调整

### 6.4 MiniLLM
- On-policy RKLD on student-generated sequences
- 用length normalization解决long-generation问题
- 但computational overhead大（需要full rollouts）

### 6.5 GKD
- 混合off-policy和on-policy data
- 用generalized JSD
- 但weight design没有HPD这样的asymmetric masking

### 6.6 DistiLLM-2
- Contrastive distillation
- 同时增加teacher-generated responses likelihood，降低student-generated responses likelihood
- 也是off-policy + on-policy混合，但formulation不同

### 6.7 HPD的独特优势

1. **Token-level hybridization**: 而非sequence-level或fixed coefficient
2. **Asymmetric masking**: 根据gap sign动态决定forward/reverse KL的使用
3. **Computational efficiency**: 只需要2次teacher forward per token，不需要full KL computation
4. **Lightweight on-policy**: 只sample 1个non-expert token，不需要full sequence rollout
5. **No additional hyperparameters**: HPD没有需要tune的hyperparameter

参考：[DistiLLM-2](https://openreview.net/forum?id=rc65N9xIrY), [ABKD](https://openreview.net/forum?id=vt65VjJakt)

---

## 7. 我的Intuition和Critical思考

### 7.1 为什么HPD有效？我的理解

HPD的核心intuition可以总结为：**symmetric的KL objectives有gradient冲突问题，HPD通过mask机制实现了asymmetric的、context-aware的reweighting**。

具体来说：
1. **Forward KL的mode coverage**: 当student低估expert token时，用teacher概率$p$作为weight，强制student覆盖这个mode
2. **Reverse KL的mode-seeking**: 当student高估某个token（expert或non-expert）时，用负的$k_1$抑制它
3. **Reinforce的redirection**: 当同时发生(1)低估expert和(2)高估non-expert时，把suppress释放的mass明确redirect到expert上

这种asymmetric设计避免了naive weighted sum的问题——weighted sum的固定系数无法根据局部情况调整。

### 7.2 与RL的Connection

HPD的formulation其实和PPO等RL算法有深层联系：
- $K_1$ estimator作为reward，类似PPO的KL penalty
- Student sampling类似policy rollout
- Mask机制类似PPO的clip

可以看作是**把RL的on-policy思想应用到distillation，但用更轻量的token-level approximation替代full sequence rollout**。

参考：[PPO](https://arxiv.org/abs/1707.06347), [GRPO](https://arxiv.org/abs/2402.03300), [DAPO](https://arxiv.org/abs/2503.14476)

### 7.3 Limitations and Future Directions

**Limitations**:
1. 只适用于teacher-student共享tokenizer的white-box KD
2. 仍然用estimator近似full KL，有variance
3. On-policy sampling只是lightweight approximation，不是full OPD

**Future Directions**:
1. **应用到pre-training**: paper在conclusion提到可以探索mid-training或pre-training阶段的应用
2. **Multi-modal KD**: 扩展到vision-language models
3. **Cross-tokenizer KD**: 解决tokenizer不匹配的问题
4. **结合RLHF**: HPD + DPO已经showcase了潜力，可以进一步explore HPD + PPO/GRPO

### 7.4 对LLM训练的Implications

HPD对LLM training pipeline有几个implications：

1. **SFT可能不是最佳initialization**: 传统pipeline是SFT → RLHF/DPO，HPD show可以用HPD替代SFT，提供更好的起点
2. **Distillation不需要full rollouts**: 轻量级token-level sampling就能获得大部分on-policy benefits
3. **Asymmetric reweighting > Fixed coefficient**: 动态的、context-aware的weighting比固定的$\alpha$-divergence更有效
4. **Entropy management is key**: 避免entropy collapse是持续improvement的关键

### 7.5 可能的Extensions我想到的

1. **Curriculum HPD**: 早期用更多forward KL（mode coverage），后期用更多reverse KL（mode seeking），动态调整mask策略
2. **Multi-teacher HPD**: 多个teacher的ensemble，每个teacher贡献不同的modes
3. **Hierarchical HPD**: 在sequence level和token level都做hybridization
4. **HPD with rejection sampling**: 对student-sampled non-expert tokens做更精细的filtering
5. **HPD for MoE distillation**: 把dense teacher distill到MoE student，不同experts处理不同modes

---

## 8. 公式变量速查表

为方便reference，整理所有关键公式的变量含义：

| 符号 | 含义 |
|------|------|
| $\theta$ | Student model parameters |
| $p(a_t \| s_t)$ | Teacher distribution |
| $q_\theta(a_t \| s_t)$ | Student distribution |
| $s_t$ | State (prefix tokens) |
| $a_t^*$ | Expert token (from teacher or ground truth) |
| $a_t$ | Sampled token (from student) |
| $\mathcal{D}$ | Offline dataset |
| $\mathcal{D}^{\pi_T}$ | Teacher-generated data |
| $\mathcal{D}^{\pi_\theta}$ | Student-generated data |
| $w(a_t \| s_t)$ | Reweighting term |
| $K_1$ | Monte Carlo KL estimator |
| $k_1$ | Expert token gap estimator |
| $k_1'$ | Sampled token gap estimator |
| $w_t^*$ | Expert token weight |
| $w_t$ | Sampled token weight |
| $z_v$ | Logit of token $v$ |
| $q_v$ | Probability of token $v$ under student |

---

## 9. 总结

这篇paper的核心贡献：

1. **Unified view**: 把SFT, FKLD, RKLD统一到reweighted log-likelihood framework下，$w(a_t|s_t)$是关键设计维度
2. **HPD algorithm**: 通过token-level的asymmetric masking机制，hybridize forward和reverse KL，同时支持off-policy和lightweight on-policy
3. **Empirical validation**: 在reasoning, dialogue, coding三个domain上一致outperform baselines，且作为DPO初始化效果更好
4. **Practical efficiency**: 不需要full sequence rollouts，computational cost可控

**对Karpathy的intuition building**: 
- 把KD看作reweighted MLE，$w$的选择决定了learning signal的性质
- KL divergence的两种方向对应不同的inductive bias，hybridize它们需要asymmetric的设计而非naive weighted sum
- On-policy的benefit主要来自exposing student to its own distribution，lightweight token-level sampling就能捕获大部分benefit
- Reward-based formulation比loss-based formulation更适合KL estimation（unbiased gradient）

参考资源：
- [Paper GitHub](https://github.com/zwhong714/Hybrid-Policy-Distillation)
- [Schulman KL approx](http://joschu.net/blog/kl-approx.html)
- [Thinking Machines OPD blog](https://thinkingmachines.ai/blog/on-policy-distillation)
- [MiniLLM](https://arxiv.org/abs/2306.08543)
- [GKD](https://arxiv.org/abs/2306.08543)
- [DistiLLM-2](https://openreview.net/forum?id=rc65N9xIrY)
- [ABKD](https://openreview.net/forum?id=vt65VjJakt)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [DPO](https://openreview.net/forum?id=HPuSIXJaa9)
- [GRPO](https://arxiv.org/abs/2402.03300)
