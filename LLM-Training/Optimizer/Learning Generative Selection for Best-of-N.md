---
source_pdf: Learning Generative Selection for Best-of-N.pdf
paper_sha256: 26821a91d5f384973acfe8c5c1c2025d31d38440ffdd0db4dfa60f04a0956a6a
processed_at: '2026-08-05T13:09:15-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用最直白的人话来讲，这篇paper的核心idea就是：**让小模型当“阅卷老师”，用RL训练它从一堆答案里挑出正确的那个。**

想象一下你参加数学考试。做填空题（生成答案）极其困难，你需要从头推导所有步骤。但如果是做选择题（从8个选项里挑一个），即使你水平一般，只要你能看出哪几个选项的推导过程明显有逻辑漏洞，你也能蒙对。

这就是这篇paper的intuition。在test-time scaling中，我们通常让大模型生成8个或者16个candidate solutions，然后再选一个最好的。生成答案（generation）的search space是整个词表，维度极高，1.7B的小模型脑容量不够，拼不过8B的大模型。但是挑答案（selection）的search space只有 $\{1, 2, ..., N\}$，维度极低。1.7B的capacity虽然不足以generate出完美解答，但绝对sufficient去判断哪个解答更好。

这篇工作就是用reinforcement learning，把1.7B小模型“挑答案”的潜力给逼出来。结果就是：1.7B的selector在挑选质量上，居然能追平甚至有时候打败5倍大小（8B）的模型。

下面我们拆解里面的技术细节，建立更深层的intuition。

---

### 1. 数据构造：为什么难度校准极其关键

你要训练一个selector，你不能随便给它喂数据。如果你给它的8个candidate里，7个都是对的，那它闭着眼睛选都能拿满分，学不到任何东西。如果8个全错，它选什么都是零分，也没有gradient。

所以作者在数据构造上做了极其精细的difficulty control：
1. **筛选Generator的薄弱点**：只用Qwen3-1.7B pass rate $< 50\%$ 的题目。这意味着这些题目对小模型来说是真难题。
2. **Candidate set的难度平衡**：强制要求candidate set里**至少有一个对的**，同时**对的占比不能超过50%**。

用公式语言来描述，对于一个problem $q$，我们生成了 $N$ 个candidates $\{y_1, ..., y_N\}$。设其中正确的数量为 $c$。我们要求 $1 \le c \le \lfloor N/2 \rfloor$。

**Intuition**：这把selection task的Bernoulli难度强制校准到了0.5附近。在RL中，reward的variance最大化时，gradient signal最强。如果全对或全错，group-relative advantage $\hat{A} = 0$，模型在这个样本上学不到任何东西。这种设计直接决定了RL训练能不能收敛。

---

### 2. RL训练机制：DAPO与Reward拆解

作者使用了[DAPO algorithm](https://arxiv.org/abs/2503.14476)进行on-policy RL training。理解这个机制，需要看懂它的loss function和advantage估计。

#### 2.1 Group-Relative Advantage
在GenSelect中，reward是sequence-level的binary signal。如果model输出的index $i^*$ 对应的candidate是正确的，$r = 1$，否则 $r = 0$。

为了降低variance，DAPO使用了GRPO-style的group baseline。对于同一个prompt $q$，我们sample $G$ 个rollouts（math里 $G=16$，code里 $G=8$）。第 $j$ 个rollout的advantage定义为：

$$ \hat{A}_j = r_j - \frac{1}{G}\sum_{k=1}^G r_k $$

这里 $\hat{A}_j$ 是第 $j$ 个rollout的advantage，$r_j$ 是它的reward（0或1），后面那项是整个group的平均reward。
**Intuition**：如果一个prompt的16个rollout里选对了8个，平均reward就是0.5。选对的rollout获得 $+0.5$ 的advantage，选错的获得 $-0.5$ 的advantage。这个normalization去除了“题目本身太简单或太难”造成的baseline偏移。

#### 2.2 DAPO Loss与Clipping
Policy gradient的loss function（实际上是negative objective）长这样：

$$ \mathcal{L}(\theta) = -\mathbb{E} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \min\left( \rho_t \hat{A}_t, \, \text{clip}\left(\rho_t, 1-\epsilon_{lo}, 1+\epsilon_{hi}\right) \hat{A}_t \right) \right] $$

变量解释：
*   $\theta$：我们要更新的policy parameters。
*   $|y|$：生成的response长度，取平均是为了token-level normalization。
*   $\rho_t = \frac{\pi_\theta(y_t \mid y_{<t}, q)}{\pi_{old}(y_t \mid y_{<t}, q)}$：importance sampling ratio。当前policy产生这个token的概率除以sample时用的旧policy的概率。
*   $\hat{A}_t$：当前token的advantage（这里直接用sequence级别的 $\hat{A}$ 广播）。
*   $\epsilon_{lo}, \epsilon_{hi}$：clipping的范围。

**关键细节：Clip-Higher**
标准PPO里 $\epsilon_{lo} = \epsilon_{hi} = 0.2$。DAPO把它们decouple了，通常 $\epsilon_{hi} > \epsilon_{lo}$。
**Intuition**：如果某个token导致选对了答案（$\hat{A} > 0$），我们希望model增加产生这个token的概率。如果 $\rho_t$ 碰到了上限 $1+\epsilon_{hi}$，就会被clip住停止上升。上限太紧会导致“exploration被扼杀”。DAPO把上限放宽（比如 $\epsilon_{hi}=0.28$），允许model更激进地强化那些能选对答案的reasoning pattern。在selection这种探索空间极大的任务里，exploration极其重要。

---

### 3. 实验数据表深度解析

我们看Table 1中最核心的 Generator: Qwen3-1.7B 部分。

| Method | AIME24 | AIME25 | HMMT25 | LCB v6 |
| :--- | :--- | :--- | :--- | :--- |
| Pass@1 | 45.42 | 37.92 | 25.00 | 33.45 |
| Pass@8 (Oracle) | 70.00 | 60.00 | 53.33 | 45.59 |
| Majority@8 | 63.33 | 52.56 | 25.00 | N/A |
| GenSelect 1.7B Prompting | 48.75 | 38.33 | 26.67 | 34.72 |
| GenSelect 8B Prompting | 65.83 | 55.42 | 38.75 | 41.24 |
| **GenSelect 1.7B RL Math** | **65.00** | **54.17** | **36.25** | 36.45 |

从表里能读出几个极其深刻的结论：

**a) Majority Voting的崩塌**
看HMMT25这列，Majority@8只有25.00，和Pass@1一模一样。
**Intuition**：HMMT25是极其变态的数学竞赛题，1.7B小模型生成的8个答案几乎never重合。Majority voting依赖“正确答案出现频率最高”的假设，假设不成立时，它退化成在8个不同答案里random pick。这就凸显了semantic reasoning over candidates的必要性。

**b) Prompting的无力与RL的神奇**
直接prompt Qwen3-1.7B做GenSelect（48.75），比Pass@1（45.42）强不了多少。1.7B太小了，in-context reasoning能力弱，看不懂指令。
但是加了RL之后，直接飙到65.00，几乎追平了8B大模型prompting的65.83。
**Intuition**：prompting要求模型在工作记忆里同时维持problem context、N个candidates的逻辑、以及比较策略，这对1.7B的attention head负担太重。RL训练把“比较逻辑”直接bake进了weights里，变成了System 1一样的直觉反应，释放了working memory。

**c) Cross-domain Transfer**
RL Math训练出来的模型，在Code (LCB v6) 上也能拿到36.45，比1.7B prompting强。
**Intuition**：模型学到的本质是“如何识别reasoning chain中的逻辑漏洞或计算错误”。这种meta-skill是domain-agnostic的。Math里学到的找错能力，迁移到code里同样适用。

---

### 4. 为什么Code不如Math？Reward的Noise理论

Table 1里，Code的增益明显不如Math。作者归因于两点，我用信号理论来解释：

1. **Math的reward是deterministic的**：使用[Math-Verify](https://github.com/huggingface/math-verify)（基于sympy），$\sqrt{2}/2$ 和 $1/\sqrt{2}$ 会被判定为相等。Label的False Positive和False Negative率极低。RL gradient signal非常干净。
2. **Code的reward是noisy的**：使用unit tests。一个有bug的程序可能恰好没被test case覆盖（False Positive，给了错误reward）；一个正确的程序可能因为IO format挂掉（False Negative，给了错误惩罚）。

在RL中，reward的noise直接等同于gradient的noise。如果model选了一个“通过测试但其实有bug”的candidate，RL会奖励它，这相当于在教model学习错误的pattern。这就是为什么Figure 3里Code训练的曲线更noisy、更早saturate。RL不怕sparse reward，最怕noisy reward。

---

### 5. Off-Distribution Transfer：最强的证据

这篇paper最striking的result在Table 1的后半部分。训练时，candidates全部由Qwen3-1.7B生成。但在测试时，喂给它Qwen3-4B甚至Qwen3-8B生成的candidates，它依然表现极好。

在Qwen3-4B作generator时，1.7B RL Math selector在AIME25上拿到73.33，甚至打败了8B prompting的72.50。

**深层Intuition**：这说明selector学到的绝对不是“1.7B模型特有的错误模式”（比如它总是算错某一步乘法）。它学到的是一种**Universal的Correctness Prior**。
8B模型生成的错误，通常比1.7B的错误更高级、更subtle（比如算法复杂度不对，或者边界条件没处理好）。1.7B的selector依然能挑出来，说明它在RL过程中，internalize了什么是“严谨的数学证明结构”。只要结构是对的，不管是谁生成的，它都能识别。这跟[DeepSeek-R1](https://arxiv.org/abs/2501.12948)里RL激发出某种底层reasoning能力有异曲同工之妙。

---

### 6. 延伸思考与Hallucination的Future Work

顺着这个思路想下去，有很多可以做的方向：

**a) Hierarchical Selection解决Context Length瓶颈**
现在GenSelect把N个candidates全塞进context。N=16时，prompt可能长达12K tokens，留给selector的reasoning space极小。
完全可以设计Hierarchical GenSelect：先把16个candidates随机分成4组，每组4个做pairwise/polling selection，选出4个survivor。然后再对这4个做final GenSelect。这相当于Tree Search里的 Tournament。可以参考[PairJudge RM](https://arxiv.org/abs/2501.13007)的做法。这样可以把N扩展到64甚至128，真正逼近Pass@N的极限。

**b) Selector与Generator的Co-evolution**
现在的pipeline是：固定Generator生成数据，训练Selector。
如果让两者一起进化呢？Generator生成candidates，Selector挑选。把Selector挑出来的correct solution作为新的SFT数据喂给Generator。Generator变强后，生成更难的candidates，逼着Selector继续变强。这是一个标准的GAN-style或Self-Play的setup。 Selector充当了Generator的Reward Model或者Critics的角色。参考[critique fine-tuning](https://arxiv.org/abs/2501.17703)。

**c) Bayesian View of Selection**
从贝叶斯角度，Selector本质上是在estimate一个posterior probability：
$$ \Pr(\text{Correct}(y_i) \mid q, y_1, ..., y_N) $$
当N很大时，candidates之间会产生mutual reinforcement。比如3个candidate都算出了同一个错误答案（因为它们犯了同一种系统性偏差），Selector可能会被这个false consensus欺骗，认为这个答案是正确的。
要解决这个问题，Selector需要学习去correlate candidates之间的错误。它需要判断：“这3个候选不仅答案一样，而且都在第5步犯了同一个除以零的错误，所以它们是高度相关的，不能算作3票”。这种去相关的reasoning，是现有GenSelect没显式训练的，可以作为下一步的优化点。

**d) Synthesis vs Selection的Trade-off**
[Qi et al.](https://arxiv.org/abs/2506.09014)尝试让模型综合N个candidate的优点生成一个新答案。本paper坚持只做Selection。
Selection的好处是reward极好算，Pass@N就是上界。坏处是它永远无法超越Generator的最佳水平。
如果Generator的Pass@N已经plateau了，Selection就到头了。这时候必须做Synthesis。但Synthesis的RL极难训，因为生成的新答案需要重新跑verifier，且reward空间极其sparse。一个可行的hybrid是：先用GenSelect筛出top-3，然后在这3个上做Synthesis，缩小search space。

### 总结
这篇paper用最朴素的思想——**把选择题当任务来做RL**——打通了小模型在test-time scaling中的瓶颈。它证明了“评判能力”和“生成能力”是可以解耦的。1.7B的脑容量做不了生成数学定理的牛顿，但绝对做得了一个判断哪份卷子推导有漏洞的阅卷老师。RL就是那本让它开窍的阅卷指南。

**References:**
*   [DAPO paper](https://arxiv.org/abs/2503.14476)
*   [GenSelect original paper](https://arxiv.org/abs/2507.17797)
*   [Math-Verify tool](https://github.com/huggingface/math-verify)
*   [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
*   [PairJudge RM](https://arxiv.org/abs/2501.13007)
*   [Critique Fine-Tuning](https://arxiv.org/abs/2501.17703)
*   [Qi et al. Parallel Samples Synthesis](https://arxiv.org/abs/2506.09014)

---

# Learning Generative Selection for Best-of-N 深度解读

这篇paper来自NVIDIA的团队（Toshniwal, Ficek, Jain等人），核心idea是把Best-of-N selection本身当作一个可以RL训练的reasoning task，让小model学会"挑答案"这个skill。下面从intuition、方法、实验细节、相关延伸几个层面展开。

---

## 1. 核心Intuition：为什么selection值得单独训练

Test-time scaling的parallel sampling pipeline是这样：
- Generator $G$ 对同一个problem $q$ 独立采样 $N$ 条candidate solutions $\{y_1, y_2, ..., y_N\}$
- 一个Selector $S$ 输出 $i^* = S(q, \{y_1, ..., y_N\})$
- 最终答案用 $y_{i^*}$

整个pipeline的上界由 $\text{Pass@N}$ 决定（即"只要N条中至少有一条正确，selector就能选对"的oracle准确率），但实际表现由 $S$ 的quality决定。如果 $S$ 很弱，就算 $\text{Pass@N}$ 高也白搭。

传统方法两类：
- **Majority voting / self-consistency**：把最常见的答案挑出来。问题是没有semantic reasoning，错了的时候不告诉你为什么。
- **External reward model**：每个candidate独立打分。问题是不做cross-candidate的对比，难以捕捉"我看完A再看完B，发现A的step 3其实错了"这种链式推理。

GenSelect ([Toshniwal et al. 2025](https://arxiv.org/abs/2507.17797))的范式：把selection framing成一个reasoning problem，selector先输出一段comparison reasoning，然后输出index $i^*$。这篇paper的thesis就是：**这个selection reasoning能力可以通过RL explicit训练给1.7B的小model**，让它在selection上接近8B-level model。

一个key insight：selection比generation简单。Generator需要从 $\mathcal{Y}$ 这个hugely high-dimensional space采样一个correct $y$，selector只需要在 $\{y_1, ..., y_N\}$ 这个finite set上做classification。复杂度从 $|\mathcal{Y}|$ 降到 $N$。所以小model在selection上的ceiling比在generation上更高。

---

## 2. 数据构造的细节设计

数据构造是这个工作里最subtle的部分，几个关键filter条件都直接控制selection难度。

### 2.1 候选集的difficulty控制

用 Qwen3-1.7B 作为candidate generator，对每个problem $q$：
1. 在整个training set上跑一次，记录每个problem的 pass rate $p(q) \in [0, 1]$
2. 只保留 $p(q) < 0.5$ 的problems → 这些是"generator容易答错"的难题，selection才有信息量
3. 对每个保留的problem，再sample 2-16个candidates
4. 要求 assembled candidate set 满足：
   - 至少有一个correct solution（否则selector怎么训练都拿不到reward=1）
   - correct candidate比例 $\leq 50\%$（否则selection太trivial，selector随便选都大概率对）

这相当于把selection task从Bernoulli难度上校准到 ≈ 0.5附近，RL signal最强。

### 2.2 多prompt per problem

每个problem通过resampling candidate set构造最多4个不同的prompts。这样做的原因我推测有几点：
- 数据增强：同一problem上不同candidate组合，避免model记住"这个problem的答案是X"
- Curriculum diversity：同一problem的candidate set有时correct占多数有时incorrect占多数
- 训练效率：candidate生成成本fixed，recombine几乎免费

### 2.3 Token budget裁剪

丢弃 $>16K$ tokens的prompt（math/code都一样）。这是出于batch内pad开销和attention $O(L^2)$ cost的考虑。Code这边还额外丢弃 $>12K$ tokens的response，因为code selector的输出（reasoning + index）通常比math短。

### 2.4 Math vs Code的label获取

- **Math**：用 [Math-Verify](https://github.com/huggingface/math-verify) 做symbolic verification（sympy-based），相当于自动判等。判断的是final answer的等价性（$\sqrt{2}/2 \equiv 1/\sqrt{2}$），不判断过程。
- **Code**：跑unit tests，binary correctness。这里有noise：buggy program可能pass有限test cases（false positive），correct program可能因为format/IO issue挂掉（false negative）。

这个label quality的差别直接导致后面math比code收益大的现象。

---

## 3. RL训练：DAPO细节

用 [DAPO](https://arxiv.org/abs/2503.14476) (ByteDance Seed, 2025)，这是PPO家族的一个改良版，特别针对RLHF/RLVR的不稳定问题。Loss形式如下：

$$
\mathcal{L}_{\text{DAPO}}(\theta) = \mathbb{E}_{(q, y) \sim \mathcal{D}_{\text{dyn}}} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \min\left( \rho_t \hat{A}_t,\, \text{clip}\left(\rho_t, 1-\epsilon_{lo}, 1+\epsilon_{hi}\right) \hat{A}_t \right) \right]
$$

其中：
- $\theta$ 是policy参数（Qwen3-1.7B的weights）
- $q$ 是GenSelect prompt（包含original problem + N个candidates）
- $y$ 是model生成的response（reasoning + selected index $i^*$）
- $\rho_t = \frac{\pi_\theta(y_t \mid y_{<t}, q)}{\pi_{\text{old}}(y_t \mid y_{<t}, q)}$ 是importance sampling ratio
- $\hat{A}_t$ 是token-level的advantage estimate，这里reward是sequence-level的 $r(q, y) \in \{0, 1\}$（选对=1，选错=0），用GRPO-style group baseline来reduce variance
- $\epsilon_{lo}, \epsilon_{hi}$ 是clipping range下/上限，decoupled（DAPO的关键改进之一）
- $\mathcal{D}_{\text{dyn}}$ 是dynamic sampling后的分布

### DAPO的4个关键trick：

1. **Clip-Higher (decoupled clipping)**：原版PPO用对称的 $\epsilon_{lo} = \epsilon_{hi} = 0.2$，DAPO用 $\epsilon_{hi} > \epsilon_{lo}$（典型如 $\epsilon_{hi} = 0.28, \epsilon_{lo} = 0.2$），让positive advantage的tokens有更大update空间，避免"好人难上进"。

2. **Dynamic Sampling**：对一个batch的rollouts，如果某个prompt的所有rollouts都reward=0或都reward=1，这个prompt的group-relative advantage $\hat{A} = 0$，gradient为零，浪费compute。DAPO直接drop这些prompts，continue sampling直到batch内每个prompt都有mixed rewards。对于GenSelect这个特别relevant：如果一个candidate set里没有correct或全correct，这个prompt没用。

3. **Token-Level Loss**：取 $\frac{1}{|y|}$ 平均，让长短responses得到平等的梯度信号。传统sequence-level求和会让long responses dominate gradient。

4. **Overlong Filtering**：超过max length的rollout直接drop（既不算正reward也不算负reward），避免model被"答不完就没奖励"这种spurious signal惩罚到repeat/whitespace padding。

### RL training的具体配置

- Framework: [VeRL](https://github.com/volcengine/verl) (HybridFlow, 来自Pku/字节)
- Hardware: NVIDIA H100
- Batch size: 128 prompts per step
- Rollouts per prompt: 16 (math) / 8 (code)
- LR: $1 \times 10^{-6}$，AdamW
- Sampling temperature: 1.5（挺高的，promote exploration）
- top-p: 1.0 (即不做top-p过滤)
- max output: 16384 tokens
- Reward：$r(q, y) = \mathbb{1}[\text{candidate } i^* \text{ labeled correct}]$

temperature 1.5是一个值得注意的细节。一般SFT后RL用 temperature 0.7-1.0，1.5相当explore-heavy。这暗示在selection task上exploration很重要——可能model需要在很多错误的selection pattern上"试错"才能学到比较逻辑。

---

## 4. 实验结果表深度分析

Table 1是这个工作的核心证据，我把它拆成几层看。

### 4.1 Generator = Qwen3-1.7B（训练分布内的candidates）

| Method | AIME24 | AIME25 | HMMT25 | LCB v6 |
|---|---|---|---|---|
| Pass@1 | 45.42 | 37.92 | 25.00 | 33.45 |
| Pass@8 (oracle upper bound) | 70.00 | 60.00 | 53.33 | 45.59 |
| Majority@8 | 63.33 | 52.56 | 25.00 | N/A |
| GenSelect 1.7B Prompting | 48.75 | 38.33 | 26.67 | 34.72 |
| GenSelect 4B Prompting | 61.25 | 53.75 | 34.17 | 39.70 |
| GenSelect 8B Prompting | 65.83 | 55.42 | 38.75 | 41.24 |
| **GenSelect 1.7B RL Math** | **65.00** | **54.17** | **36.25** | 36.45 |
| GenSelect 1.7B RL Code | 57.08 | 44.58 | 26.25 | **36.84** |

关键观察：

**a) Majority voting在HMMT25上完全失效**：Majority@8 = 25.00 = Pass@1。HMMT25是极难的竞赛题，model的答案几乎never重合，majority没有mode可以vote，退化为random pick among ties。这就突显了GenSelect的价值——它做semantic comparison，不依赖答案匹配。

**b) GenSelect 1.7B Prompting ≈ Pass@1**：48.75 vs 45.42。这说明1.7B model直接prompted做selection基本没增益，反而略高于random一点。1.7B太小，prompted GenSelect的instruction following能力弱。

**c) RL Math 1.7B 几乎追平 GenSelect 8B Prompting**：65.00 vs 65.83 (AIME24)，54.17 vs 55.42 (AIME25)。**5x smaller的model经过RL可以达到8B prompting的水平**。这是核心result。

**d) RL Math cross-domain to code**：在LCB v6上 RL Math = 36.45，比 GenSelect 1.7B Prompting (34.72) 高，但低于GenSelect 4B/8B。Math的RL训练确实transfer到了code，验证了 [AceReason-Nemotron](https://arxiv.org/abs/2505.14674) 的观察——math RL对code有正迁移，反之较弱。

### 4.2 Generator = Qwen3-4B/8B（off-distribution的candidates）

这是transfer实验，最striking的数字：

| Method (Generator Qwen3-4B) | AIME24 | AIME25 | HMMT25 | LCB v6 |
|---|---|---|---|---|
| Pass@1 | 72.08 | 61.67 | 41.67 | 51.98 |
| Pass@8 (oracle) | 86.67 | 83.33 | 60.00 | 66.30 |
| Majority@8 | 80.00 | 72.64 | 50.00 | N/A |
| GenSelect 8B Prompting | 82.92 | 72.50 | 53.75 | 56.11 |
| **GenSelect 1.7B RL Math** | **80.83** | **73.33** | **55.42** | **53.36** |

RL Math 1.7B在AIME25上73.33 > GenSelect 8B prompting 72.50。**1.7B在off-distribution上打败了同prompt的8B**。

这个transfer是关键：训练时candidates来自Qwen3-1.7B（错误模式更"低级"，比如计算错误、step missing），但evaluation时candidates来自Qwen3-4B/8B（错误更"高级"，比如逻辑flaw、subtle algorithm issue）。Selector学到的不只是识别1.7B的buggy pattern，而是某种universal的"什么是correct reasoning"的notion。

### 4.3 为什么code比math效果差？两个hypothesis

**Hypothesis 1: Label noise**：
- Math: symbolic verifier 精确判等，假阳/假阴率极低
- Code: unit tests 是 sample of input space，false positive (buggy但pass所有tests) 和 false negative (correct但fail某些corner case) 都存在
- 训练reward noisy → RL signal noisy → 训练效果弱

**Hypothesis 2: Selection difficulty**：
- Math selection：比较两个proof的步骤逻辑，正确性比较sharp
- Code selection：需要判断算法正确性 + 边界处理 + complexity + implementation细节，semantic coverage广得多。这是 [CodeContests+](https://arxiv.org/abs/2506.05817) 工作里也提到的现象

### 4.4 Training dynamics (Figure 3)

Math: selection accuracy 单调上升，平滑收敛
Code: 更noisy，更早saturate

我猜测code的noise来自 reward的随机性。一个selection选了一个"通过unit tests但其实有bug"的candidate，model被reward了，但实际上选错了——这种spurious reward让training dynamics不稳定。

---

## 5. 与相关工作的positioning

### 5.1 vs Reward Models

传统ORM/PRM ([Cobbe et al. 2021](https://arxiv.org/abs/2110.14168); [AceMath](https://arxiv.org/abs/2412.15084)) 对每个candidate独立打分，需要training一个separate scorer。GenSelect复用了reasoning model自身的能力，省了一个model。

[Generative Verifiers (Zhang et al. 2024)](https://arxiv.org/abs/2408.15240) 用next-token prediction的形式让LM输出verification score $\Pr(\text{correct} \mid q, y)$，本质还是per-candidate独立判断，没有cross-candidate comparison。

[Generative Reward Models (Mahan et al. 2024)](https://arxiv.org/abs/2410.12832) 在preference pairs上做generative reasoning，然后用于RLHF。和本工作的差别是：preference pair是2-tuple，GenSelect是N-tuple，combinatorial上更接近Best-of-N的真正需求。

### 5.2 vs Synthesis-based aggregation

[Qi et al. 2025](https://arxiv.org/abs/2506.09014) 和 [Zhao et al. 2025](https://arxiv.org/abs/2509.06870) 训练model去synthesize出一个new answer from N candidates。本工作显式avoid了synthesis，理由：
1. Qi et al.的finding：synthesis often reduces to copying one of the inputs，没必要
2. Reward computation easier：selection的reward基于candidate的已知label，synthesis要重新verify（cost）

我觉得这里有一个细微的trade-off：
- Selection的天花板是 $\text{Pass@N}$
- Synthesis的天花板是 1（理论上能拼出更好的解）
- 但synthesis的RL signal更难获取，cost更高

实际应用里 Pass@N 通常已经够用了，特别是N=8-16时。

### 5.3 vs Self-Consistency

[Self-Consistency (Wang et al. 2023)](https://arxiv.org/abs/2203.11171) 是GenSelect的"零成本baseline"，只用答案mode。它的优势是zero-shot，劣势是当candidates答案不重合时退化为random。HMMT25的Majority@8 = Pass@1就证实了这点。

### 5.4 vs Test-time scaling survey

[Snell et al. 2024](https://arxiv.org/abs/2408.03314) 的"Scaling LLM Test-Time Compute"工作里讨论过parallel vs sequential scaling。本工作属于parallel scaling的selector端，与sequential scaling（如self-refine, search-based）是orthogonal的。理论上可以组合：parallel sample → GenSelect → self-refine。

---

## 6. 公式层面的一些补充

### Pass@k的形式定义

$$
\text{Pass@k}(q) = \mathbb{E}_{y_1, ..., y_k \sim G(\cdot \mid q)} \left[ \mathbb{1}\left[\exists i \in [k]: \text{Correct}(y_i, q)\right] \right]
$$

无偏估计：
$$
\widehat{\text{Pass@k}}(q) = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$
其中 $n$ 是总采样数，$c$ 是correct的数量。

这是Table 1里 Pass@8 的upper bound意义——selector能选到correct candidate的概率最高等于"至少一个correct"的概率。

### RL reward function

对GenSelect训练：
$$
r(q, y) = \begin{cases} 1 & \text{if } \text{Correct}(y_{i^*}, q) \\ 0 & \text{otherwise} \end{cases}
$$
其中 $i^* = \text{parse}(y)$ 是从model response $y$ 中parse出的selected index。

这是binary, sparse, sequence-level reward。Group-relative advantage (GRPO-style)：

$$
\hat{A}_t^{(j)} = r(q, y^{(j)}) - \frac{1}{G}\sum_{j'=1}^G r(q, y^{(j')})
$$
其中 $j \in [G]$ 是同一prompt下的rollout index，$G$ 是group size（这里 = 16 或 8）。

这种group baseline去掉了reward均值的全局偏置，让advantage对"这个prompt是不是本身太难"做了normalization。这也是为什么"全错/全对"的prompt被dynamic sampling drop掉——它们的group-relative advantage = 0。

### Selection的Bayesian视角

可以从Bayesian角度理解selection task：给定candidates $\{y_1, ..., y_N\}$，selector应该输出

$$
i^* = \arg\max_i \Pr(\text{Correct}(y_i) \mid q, y_1, ..., y_N)
$$

GenSelect prompting相当于让LM用in-context reasoning estimate这个posterior。RL训练相当于calibrate这个estimator，让它在hard cases上更准。一个side observation：当N增大，selector需要处理更多comparison，但candidate间可能互相reinforce（多个错误的similar answer看起来像"consensus"）。这是N增加selection难度sub-linearly增长的可能解释。

---

## 7. 我的直觉和延伸思考

### 7.1 Selection能力的"通用性"

Transfer到stronger generator是最surprising的result。这说明selection学到的不只是"识别1.7B特有的错误模式"，而是某种更abstract的"correct reasoning长什么样"的notion。这跟 [DeepSeek-R1](https://arxiv.org/abs/2501.12948) 里 RL让model学到"reasoning itself"的某种meta-skill有相似味道。

一个可能的extension：能否用更强的candidates（混合1.7B/4B/8B的输出）来训练，让selector同时见过多种failure mode？我预期这会进一步push performance，特别是transfer到更strong generator时。

### 7.2 Selector的input长度问题

N增大时GenSelect prompt的token数线性增长。N=16 × avg 800 tokens/candidate = 12.8K tokens just for context。16K max length下留给selector reasoning的空间很小。这限制了GenSelect scale到N=64+的可能性。一个extension是hierarchical selection：先pairwise tournament缩小到top-K，再full GenSelect。这跟 [PairJudge RM](https://arxiv.org/abs/2501.13007) 的knockout tournament思路一致。

### 7.3 RL vs SFT的对比

Paper没显式给SFT baseline。直觉上RL比SFT强的原因是：
- SFT只能学"应该选哪个"，需要teacher给出每个prompt的correct index
- RL只看reward，model可以通过自己explore学到"为什么这个错那个对"

但SFT的好处是reward不需要verifier，可以用更noisy的preference data。一个可能的研究：用LLM-as-a-judge的preference pairs做SFT warmup，然后RL精修。

### 7.4 与o1-style reasoning训练的对比

[OpenAI o1](https://arxiv.org/abs/2412.16720) 和DeepSeek-R1的RL训练让generator学会长reasoning。本工作让selector学会reasoning over candidates。这两个方向可以stack：用一个reasoning model做generator（已经RL训练过），再用一个reasoning model做selector（再RL训练selection）。整体pipeline变成了"用RL增强的model + 用RL增强的selector"，每层都通过RL获得专门skill。

### 7.5 RL训练的scaling问题

1.7B能学到selection不代表所有size都能学好。Paper只测了1.7B。一个open question：在更小model（比如0.5B）上RL能学到什么程度？还是有一个minimum capacity threshold？如果是后者，多少capacity是"刚好够做selection"？

### 7.6 Generalization到其他domain

Paper只测math/code。Open question: 这个approach能transfer到logical reasoning, commonsense, open-ended generation吗？挑战是reward signal的获取：
- Math/code: 有自动verifier
- Open-ended: 需要LLM-as-a-judge / reward model，引入noise

一个可能的workaround：先用verifier-rich domain（math/code）训练base selector skill，再在少量preference data上微调到其他domain。这跟 [Critique Fine-Tuning](https://arxiv.org/abs/2501.17703) 思路类似。

### 7.7 Risk: Selector overfitting to selection-specific artifacts

训练数据里candidates都是同一个generator（Qwen3-1.7B）的outputs。这可能有artifact：比如1.7B的某些特定phrasing pattern与correctness相关。Selector可能学到这些spurious correlations。Paper的transfer experiment部分缓解了concern，但更strict的test应该是完全不同family的generator（如Llama, Mistral）的candidates上evaluate。

---

## 8. 总结的直觉

这篇paper的deep intuition是：**reasoning model的"reasoning over reasoning"能力是一种可分离的skill，可以通过RL独立训练**。

传统reasoning model训练是"在problem上reason out the answer"。这个paper是"在N个answers上reason out which is best"。两个不同的objective，但share同样的reasoning capability底层。

1.7B在生成上做不过8B是必然的（参数不够），但在选择上做不过8B就不是必然——选择是更constrained task。RL training把这部分"额外容量"释放出来，让小model在选择这个sub-task上达到大model的水平。

这是test-time scaling的另一个dimension：除了generate more candidates，还可以train better selector。两个dimension正交，可以一起scale。

---

## References

- DAPO: https://arxiv.org/abs/2503.14476
- GenSelect (original): https://arxiv.org/abs/2507.17797
- OpenMathReasoning: https://arxiv.org/abs/2504.16891
- OpenCodeReasoning-II: https://arxiv.org/abs/2507.09075
- Self-Consistency: https://arxiv.org/abs/2203.11171
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- VeRL framework: https://github.com/volcengine/verl
- Generative Verifiers: https://arxiv.org/abs/2408.15240
- PairJudge RM: https://arxiv.org/abs/2501.13007
- OpenAI o1: https://arxiv.org/abs/2412.16720
- Critique Fine-Tuning: https://arxiv.org/abs/2501.17703
- Math-Verify: https://github.com/huggingface/math-verify
- NeMo-Skills: https://github.com/NVIDIA-NeMo/Skills/
- Large Language Monkeys: https://arxiv.org/abs/2407.21787
- Snell test-time compute: https://arxiv.org/abs/2408.03314
- AceReason-Nemotron: https://arxiv.org/abs/2505.14674
- Qi et al. Parallel Samples: https://arxiv.org/abs/2506.09014
- Zhao et al. Aggregation RL: https://arxiv.org/abs/2509.06870
- CodeContests+: https://arxiv.org/abs/2506.05817
- Generative Reward Models: https://arxiv.org/abs/2410.12832
- Training Verifiers (Cobbe): https://arxiv.org/abs/2110.14168
- AceMath: https://arxiv.org/abs/2412.15084
- Deep Think with Confidence: https://arxiv.org/abs/2508.15260
- Reward Reasoning Model: https://arxiv.org/abs/2505.14674
- Heimdall: https://arxiv.org/abs/2504.10337
