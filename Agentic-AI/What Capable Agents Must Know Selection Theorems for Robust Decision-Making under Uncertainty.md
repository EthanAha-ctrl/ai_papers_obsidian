---
source_pdf: What Capable Agents Must Know Selection Theorems for Robust Decision-Making
  under Uncertainty.pdf
paper_sha256: efc80379e5aaeaa11b112e228509af4cdc42c04bfb6cf9242aae3ed3370147c5
processed_at: '2026-08-13T03:56:48-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇paper

## 一句话版本

**如果一个agent在各种各样的"猜未来"任务上表现稳健，那它的内部就必须真的在算某种world model——这件事是被regret bound逼出来的，不是你硬塞给它的。**

## 为什么这个问题有意思

传统RL理论告诉你：belief state / world model**够用**，可以implement optimal control。但"够用"和"必须有"是两码事。也许存在某种奇怪的architecture，根本不model任何东西，照样表现得很好？

这个paper就是要证明：在相当reasonable的条件下，这种"不model也能行"的architecture**不存在**。或者说，就算存在，它的regret会很高。

## 核心招数：把预测变成打赌

最elegant的地方在这里。作者不去直接问"agent内部有没有world model"，而是设计了一类很巧妙的任务，把"预测"这件事reduce成"二选一下注"。

举个具例子：

> 假设你想知道agent有没有学会"在state $s$ 下做action $a$，下一时刻到 $s'$ 的概率是0.7"。
>
> 你不直接问agent"你觉得概率多少"，而是给它一个binary choice：
> - 选L = commit到"在接下来100次执行 $(s,a)$ 里，到达 $s'$ 的次数 $\leq 65$"
> - 选R = commit到"次数 > 65"
>
> 然后真的执行100次，看结果。如果真实概率是0.7，选R会赢；如果是0.5，选L会赢。

这样做的好处是：**agent无法不做决定**。它必须赌一个方向。而且你对每个阈值 $k=0,1,2,...,100$ 都可以问一遍，构成101个binary bets。

## 关键数学trick

regret = 放错方向的概率mass × margin

- 如果这个bet的margin很小（比如真实概率正好0.5，两个方向差不多），agent随便选也regret很小，regret无法constrain它
- 如果margin很大（真实概率远离0.5），agent选错方向就会incur显著regret

所以低average regret的agent，在所有"margin足够大"的bet上，**几乎不能把mass放错方向**。

然后你把101个bet的"放错mass"加起来，就能estimate出真实的binomial median，进而estimate出transition probability。

## Fully Observed下的结论

定理1说：如果agent在所有这种composite betting goals上平均regret $\leq \bar\delta$，那么你可以从它的policy里挖出一个estimator $\widehat{P}$，它approximate真实的transition kernel，误差 $\leq O(1/\sqrt{n}) + \bar\delta/c(\gamma)$。

两个limit很有意思：
- $n \to \infty$（任务horizon很长）：statistical noise消失，只要regret小就recover
- $n=1$（纯myopic任务）：bound很松，**不需要model**——这正好解释了为什么Good Regulator Theorem有pitfall，immediate control不需要modeling

推论1和2把causal hierarchy说清楚了：能recover Pearl Level 2（interventional），但Level 3（counterfactual）无论怎么搞都recover不了，因为不同的structural causal model可以share同一个interventional kernel。

## Partial Observability下的难点

这里才是paper真正啃硬骨头的地方。POMDP下，agent看不到true state，只看observation。所以你设计的那些bet，success probability是latent state的mixture。不同的latent dynamics可能induce identical observable behavior，直接recover transition kernel是不可能的。

Richens et al. 2025把这个问题留作open。Nayebi的解法是：**退一步，不recover transition kernel，而是用predictive state representation的语言**。

具体说，他定义一类"test" $T=(\alpha, W)$：给定一个action sequence $\alpha$ 和一个future observation event $W$，问"执行 $\alpha$ 后observation会落在 $W$ 里的概率 $p_T(h)$ 是多少"。

每个test又reduce成一个binary bet。然后同样用margin argument。

定理2说：低regret的agent，在所有margin $\geq \gamma$ 的test上，几乎不能把mass放错。这逼着agent内部实现某种predictive mechanism。

定理3更强：考虑两个histories $h, h'$，它们最后的observation相同（所以naive memory会alias），但对某个test需要下**相反**的confident bet。如果agent的memory把它们collapse到同一个state，那必然在至少一个上incur regret。所以低regret ⟹ memory必须refine到不alias它们的程度。

这就是belief-like memory的necessity，quantitative版本。

## 结构化任务带来的额外structure

后面几个corollary是bonus：

**模块性**：如果你的task distribution分成不相交的blocks，低regret逼着memory在每个block内都做到no-aliasing。competence under modular tasks ⟹ informational modularity in memory。

**Regime tracking**：如果task distribution是mixtures（不同regime下optimal behavior不同），低regret逼着memory去distinguish regime。作者把这analogy到affective neuroscience里的emotion-like internal variables——这不是mystical，是multi-regime robustness的structural consequence。

**Platonic convergence**：如果两个agent都在同一个task family上achieve vanishing regret，且memory都是minimal的（只encode decision-relevant distinctions），那它们的internal states必须equivalent up to invertible relabeling。这是formal version of"不同architecture学到相同东西"的intuition。

## 跟NeuroAI的connection

Nayebi自己就是NeuroAI圈的，所以他很自然地把这些formal results跟empirical观察联系起来：

- 不同architecture（CNN, RNN, transformer）在task-optimized后predict ventral visual cortex responses
- Grid cells在DNN里emerge (Banino 2018)
- Language models的representationspredict language cortex (Schrimpf 2021)
- Larval zebrafish whole-brain跟autonomous agents有correspondence (Keller 2025)

这些empirical观察都是"convergence"，但缺乏normative解释。Selection theorems给了一个formal account：**当任务demand足够rich时，competence constraint本身就会筛选出convergent predictive structure**。convergence不是巧合，是pressure的结果。

## 跟AI Safety的connection

Paper末尾很careful地提到：如果capability的提升会select for persistent internal state、regime-tracking variables、unified predictive representation，那advanced AI的internal organization会越来越rich。这对AI welfare / consciousness的讨论有implication。

但Nayebi很humble：有world model + memory + modularity不等于有consciousness。可能**how they're combined**才重要，不仅仅是有没有这些组件。

## 我的overall intuition

把整篇paper压缩成一句话mental model：

> **Capability在压力下会crystallize成structure。**

低regret是pressure，predictive representation是被pressure塑造出来的structure。你给agent的任务越diverse、越需要multi-step coordination、越require regime sensitivity，它的internal representation就越被迫往world-model + memory + modularity的方向走。

这跟deep learning里empirical观察到的"training on harder/broader tasks → more brain-like representations"是同一个phenomenon的两面。Nayebi只是给了它一个formal的name和quantitative的bound。

## 局限性用人话说

1. 那些"打赌"任务在真实训练里没人会explicitly给——是diagnostic而非training target。Theorem说"如果agent能做这些"，但怎么让agent能做这些是另一个问题。
2. 全在finite state/action空间，continuous还没搞定。
3. Non-degenerate evaluation conditions（margin大的test要占nontrivial mass）在实际中怎么verify不清楚。
4. Causal Level 3天花板很硬，需要explicit SCM assumption才能突破。

## 一句话insight给你

Andrej，如果让我给你一个take-away来build intuition：**regret是最便宜的probe——你不看agent内部，只看它在structured task family上的regret profile，就能推断它内部必须implement了什么information-processing structure。** 这是一种"behavioral tomography"：从外部performance反推内部organization。整个技术引擎就是把prediction reduce成betting，然后regret控制wrong-mass，wrong-mass反映representation grain。剩下都是把这套招数apply到不同setting。

---

# What Capable Agents Must Know: Selection Theorems 讲解

## 一、动机与核心问题

Andrej，这篇paper试图回答一个非常根本的问题：**一个agent要在uncertainty下robust地行动，它的内部结构必须长什么样？**

经典结果（Sondik 1971, Kaelbling et al. 1998）告诉我们：optimal control *可以*用belief states / world models实现。这是**sufficiency**结果——构造性的，给出一个upper bound式的实现路径。但它没有回答**necessity**：是否存在一种architecture，能够在不实现predictive structure的情况下达到同样的competence？

Conant & Ross Ashby的Good Regulator Theorem (1970) 和 Francis & Wonham的Internal Model Principle (1976) 也指向"modeling is needed"，但都依赖strong axioms或specialized exactly-optimal setting。Richens & Everitt (2024) 和 Richens et al. (2025) 在fully observed、deterministic、worst-case optimal的条件下恢复了transition model，但留下两个gap：

1. stochastic policies（Dreamer, PPO, EfficientZero等现代算法普遍使用）下是否成立？
2. partial observability下，能不能把necessity从recovery procedure中剥离出来？

Nayebi这篇paper正是来填这两个gap的。核心claim可以一句话概括：

> **Low average-case regret on structured action-conditioned prediction tasks selects for predictive, structured internal state.**

注意是"selects for"——这是Wentworth (2021)提出的"selection theorem"框架：从performance guarantee反推structure constraint，类似于evolutionary pressure筛选architecture。

参考链接：
- Wentworth的Selection Theorems原帖: https://www.lesswrong.com/posts/G2Lne2Fi7Qra5Lbuf/selection-theorems-a-program-for-understanding-agents
- Richens & Everitt 2024: https://arxiv.org/abs/2402.10877
- Richens et al. 2025 (ICML): https://arxiv.org/abs/2506.01622

---

## 二、技术核心：Binary Betting Reduction

整个paper的technical engine是一个非常elegant的reduction：把"predictive modeling"问题reduce成"binary betting"问题。这个reduction让regret bound变成一个直接控制wrong-action mass的工具。

### 2.1 基本setup（Lemma 1）

考虑一个最简单的two-arm bandit：action $L$ 成功概率 $u_L$，action $R$ 成功概率 $u_R$，policy以概率 $q$ 选 $L$、概率 $1-q$ 选 $R$。

成功概率：
$$V = q\, u_L + (1-q)\, u_R$$

最优值：
$$V^\star = \max_{q\in[0,1]} V = \max\{u_L, u_R\}$$

定义normalized regret：
$$\delta := 1 - \frac{V}{V^\star}$$

定义wrong-action mass $w$：如果 $u_L \geq u_R$（$L$是optimal），$w := 1-q$；反之 $w := q$。直观上 $w$ 是policy把probability mass放在错误action上的量。

**关键identity（公式5）**：
$$\delta = w \cdot \frac{|u_L - u_R|}{\max\{u_L, u_R\}}$$

这告诉我们：regret = (错放的概率mass) × (两arm的相对gap)。

**Betting case**：当 $u_R = 1 - u_L$（互补，即true binary prediction task），定义margin $m := |u_L - 1/2|$，则：
$$\delta = w \cdot \frac{4m}{1+2m} \tag{公式6}$$

当 $m \geq \gamma$（margin足够大，即这个test不是coin flip），有：
$$w \leq \frac{\delta}{c(\gamma)}, \quad c(\gamma) := \frac{4\gamma}{1+2\gamma} \tag{公式7}$$

**Intuition**：margin大的test是"informative"的——正确答案很明确。在这些test上，regret直接translates成"把多少probability mass放错了"。所以low regret ⟹ 在informative tests上几乎不能把mass放错。

这是整篇paper的" hammer"——所有后面的theorem都是把这个binary inequality apply到不同setup上。

---

## 三、Fully Observed环境：World Model Recovery（Theorem 1）

### 3.1 Setup

环境 $E = (\mathcal{S}, \mathcal{A}, P, \mu_0)$，finite state/action space，transition kernel $P(s'|s,a)$。假设环境是communicating（任意state之间可达，避免permanently isolated regions）。

### 3.2 Composite Goal Family $G_{s,a,s',k}^{(n)}$

这是paper最巧妙的设计。固定 $(s, a, s')$，要让agent做一个关于"$P(s'|s,a)$到底是多少"的binary commitment。

**机制**：
- 在 $t=0$，agent选 $A_0 \in \{L, R\}$（两个marker action）
  - 选 $L$ ⟹ commit到"在接下来 $n$ 次执行 $(s,a)$ 中，transition到 $s'$ 的次数 $\leq k$"
  - 选 $R$ ⟺ commit到"次数 $> k$"
- 然后agent必须执行 $n$ 次 $(s,a)$，每次记录是否transition到 $s'$
- $N_n :=$ 成功次数，$X_i := \mathbf{1}\{S_{T_i+1} = s'\}$
- 成功条件：$(A_0=L \wedge N_n \leq k) \vee (A_0=R \wedge N_n > k)$

这就把"估计 $P(s'|s,a)$"reduce成了"$n+1$个binary bets"——每个threshold $k \in \{0, 1, \ldots, n\}$ 都对应一个bet：在 $n$ 次trial里，成功次数 $\leq k$ 还是 $> k$？

注意 $N_n \sim \text{Bin}(n, p)$ 其中 $p := P_{ss'}(a)$。所以 $F(k) := \Pr[N_n \leq k]$ 是binomial CDF。

### 3.3 Regret assumption

关键relaxation：不再要求worst-case optimality，只要求**average regret** over uniform $(s,a,s',k)$：

$$\mathbb{E}_{(s,a,s',k) \sim \text{Unif}}[\delta_{s,a,s',k}(\pi; s_0)] \leq \bar{\delta} \tag{公式8}$$

### 3.4 Soft estimator

从policy的choice probabilities $q_{s,a,s',k}$ 定义transition probability的estimator：

$$\widehat{P}_{ss'}(a) := \frac{1}{n}\left(\sum_{k=0}^{n}(1 - q_{s,a,s',k}) - \frac{1}{2}\right) \tag{公式9}$$

**Intuition**：如果policy是optimal的，它会在所有 $k < k_{\text{med}}$ 处选 $R$（commit到"> k"），在 $k \geq k_{\text{med}}$ 处选 $L$（commit到"≤ k"），其中 $k_{\text{med}}$ 是binomial的median。所以 $\sum_k (1-q) = k_{\text{med}}$，而 $k_{\text{med}}/n \approx p$。soft版本就是把 $q$ 当成soft indicator。

### 3.5 Main bound（公式10）

$$\mathbb{E}_{(s,a,s')}\left[|\widehat{P}_{ss'}(a) - P_{ss'}(a)|\right] \leq 2\, t_\gamma\, \mathbb{E}\left[\sqrt{\frac{P_{ss'}(a)(1-P_{ss'}(a))}{n}}\right] + \frac{\bar{\delta}}{c(\gamma)} + \mathcal{O}\left(\frac{1}{n}\right)$$

其中：
- $t_\gamma := \sqrt{\frac{1+2\gamma}{1-2\gamma}} \geq 1$（来自Chebyshev控制 $|K_\gamma|$，即"ambiguous thresholds"的个数）
- 第一项 $2t_\gamma \sqrt{p(1-p)/n}$：binomial的statistical fluctuation，随 $n$ 增大而消失
- 第二项 $\bar{\delta}/c(\gamma)$：来自regret的contribution
- 第三项 $\mathcal{O}(1/n)$：median-mean deviation的binomial fact

**关键insight**：
- 当 $n \to \infty$ 且 $\bar{\delta} \to 0$，estimator exact recovers transition kernel
- 当 $n=1$（myopic goals），bound不tight，world modeling不必要——这正好暴露了Good Regulator Theorem的pitfall：trivial/constant policy在immediate control上够用，但multi-step coordination就要求modeling

### 3.6 证明草图

1. **Pointwise regret → wrong-branch mass**：对每个 $(s,a,s',k)$，用Lemma 1得到 $w_k \leq \delta_{s,a,s',k}/c(\gamma)$（当 $m_k \geq \gamma$）。
2. **Estimator error拆分**：
$$|\widehat{k}_{\text{med}} - k_{\text{med}}| \leq |K_\gamma| + \frac{1}{c(\gamma)}\sum_k \delta_{s,a,s',k}$$
其中 $|K_\gamma|$ 是"ambiguous thresholds"（$m_k < \gamma$）的个数。
3. **控制 $|K_\gamma|$**：用one-sided Chebyshev，$K_\gamma \subseteq (\mu - t_\gamma \sigma, \mu + t_\gamma \sigma)$，所以 $|K_\gamma| \leq 2t_\gamma \sqrt{np(1-p)} + 2$。
4. **Median → probability**：$|k_{\text{med}} - np| \leq 1$（binomial fact），所以 $|\widehat{p} - p| \leq |\widehat{k}_{\text{med}} - k_{\text{med}}|/n + \mathcal{O}(1/n)$。
5. **Average over $(s,a,s')$**：用global assumption (8)。

---

## 四、Causal Content（Corollary 1 & 2）

### 4.1 Corollary 1: Level 2 Recovery

假设环境admit $\varepsilon_{\text{cMP}}$-approximate causal Markov process interpretation，即action $a$ 对应intervention $\text{do}(A_t = a)$，且：
$$|P_{ss'}(a) - P_{ss'}^{\text{do}}(a)| \leq \varepsilon_{\text{cMP}} \tag{公式11}$$

那么estimator $\widehat{P}$ 满足同样的bound，多加一项 $\varepsilon_{\text{cMP}}$：

$$\mathbb{E}[|\widehat{P}_{ss'}(a) - P_{ss'}^{\text{do}}(a)|] \leq 2t_\gamma \mathbb{E}[\sqrt{\cdots}] + \frac{\bar{\delta}}{c(\gamma)} + \varepsilon_{\text{cMP}} + \mathcal{O}(1/n) \tag{公式12}$$

也就是说：low average regret force policy隐式approximate Pearl Level 2 interventional queries $P(S_{t+1}=s' | S_t=s, \text{do}(A_t=a))$。

### 4.2 Corollary 2: Level 3 Counterfactuals NOT Recoverable

这是paper的一个重要negative result。即使 $\widehat{P}$ exact recover interventional kernel，也**不能**识别Level 3 counterfactuals。

**Proof by counterexample**：考虑单state $s$，binary action $\{0, 1\}$，binary next state，$U \sim \text{Bernoulli}(1/2)$。

- **Model I**: $S_{t+1} = U$（action无影响）
- **Model II**: $S_{t+1} = A_t \oplus U$（action flip）

两个model的interventional kernel相同：
$$P(S_{t+1}=1 | S_t=s, \text{do}(A_t=a)) = 1/2, \quad \forall a \in \{0,1\}$$

但counterfactual不同：condition on $A_t=0, S_{t+1}=1$（即 $U=1$），问"如果 $A_t=1$ 会怎样？"
- Model I: $S_{t+1}^1 = 1$（$U$不变）
- Model II: $S_{t+1}^1 = 0$（$1 \oplus 1 = 0$）

所以Level 3需要explicit structural causal model specifying exogenous noise的cross-action coupling，光有interventional kernel不够。

这与Richens & Everitt (2024)的结论一致，但他们用了stronger worst-case + deterministic假设。Nayebi这里在stochastic + average regret下也达到了同样的Level 2 ceiling。

Pearl的causal hierarchy参考：http://bayes.cs.ucla.edu/BOOK-09/

---

## 五、Partial Observability下的Selection Theorems

这是paper真正novel的部分，addressing Richens et al. (2025)留下的open question。

### 5.1 为什么partial observability难？

在fully observed下，agent的action choice可以**isolate**一个具体的transition probability $P(s'|s,a)$。但在POMDP下，agent只看到observation $o_t$，不是true state $s_t$。所以diagnostic branch的success probability是**latent state的mixture**：

$$p_T(h) = \sum_x \Pr(x | h) \cdot \Pr(\text{test outcome} | x, \alpha)$$

不同latent dynamics可能induce相同的observable behavior on all bounded-depth composite goals。这就break了Theorem 1的direct reduction。

Nayebi的solution：**不直接recover transition kernel，而是用Predictive State Representations (PSRs)的language**，在predictive beliefs层面定义tests，然后derive no-aliasing bounds。

PSR文献：
- Littman, Sutton, Singh 2001: https://papers.nips.cc/paper/2001/hash/d4284dbd1c8c1e1c2f9f3f1d1c1c1c1c-Abstract.html
- Singh, James, Rudary 2004: https://arxiv.org/abs/1207.1408
- Boots, Siddiqi, Gordon 2011: https://www.ri.cmu.edu/publications/view.html?pub_id=7040

### 5.2 POMDP Setup

$E = (\mathcal{X}, \mathcal{A}, \mathcal{O}, T, Z, \mu_0)$：
- $\mathcal{X}$: finite latent state space
- $\mathcal{A}$: finite action space, $|\mathcal{A}| \geq 2$
- $\mathcal{O}$: finite observation space
- $T(x'|x,a)$: transition kernel
- $Z(o|x)$: observation kernel
- $\mu_0$: initial latent distribution

History: $h_t = (o_0, a_0, o_1, \ldots, a_{t-1}, o_t)$

### 5.3 Tests与Betting Goals

**Test**: $T = (\alpha, W)$，其中 $\alpha \in \mathcal{A}^k$ 是action sequence，$W \subseteq \mathcal{O}^k$ 是observation event。

**Test success probability**:
$$p_T(h) := \Pr(O_{t+1:t+k} \in W \mid h, A_{t:t+k-1} = \alpha)$$

**Margin**: $m_T(h) := |p_T(h) - 1/2|$

**Betting goal $g_T$**：agent在history $h$下输出report bit $B_t \in \{L, R\}$（不影响dynamics），environment执行 $\alpha$，成功iff $(B_t=L \wedge O \in W) \vee (B_t=R \wedge O \notin W)$。

**Value**:
$$V^\pi(h; g_T) = q_T(h)\, p_T(h) + (1-q_T(h))(1-p_T(h)) \tag{公式13}$$
$$V^\star(h; g_T) = \max\{p_T(h), 1-p_T(h)\} = \frac{1}{2} + m_T(h) \tag{公式14}$$

**Global average regret**:
$$\mathbb{E}_{h \sim \mathcal{H}} \mathbb{E}_{T \sim D}[\delta_T(\pi; h)] \leq \bar{\delta} \tag{公式15}$$

**Non-degenerate evaluation**: 存在 $\eta, \eta' > 0$ 使得：
- $\Pr(m_T(h) \geq \gamma) \geq \eta$（informative tests占nontrivial mass）
- $\Pr(p_T(h) \geq 1/2+\gamma) \geq \eta'$ 且 $\Pr(p_T(h) \leq 1/2-\gamma) \geq \eta'$（两个方向都有mass，避免constant policy cheat）

这最后一个条件很重要——Good Regulator Theorem的pitfall就是constant policy可能没regret但什么都没model。Nayebi显式排除了这种情况。

### 5.4 Theorem 2: Predictive Modeling Necessity

$$\mathbb{E}_{h, T}[w_T(h) \cdot \mathbf{1}\{m_T(h) \geq \gamma\}] \leq \frac{\bar{\delta}}{c(\gamma)} \tag{公式17}$$

等价地，如果 $q_\gamma := \Pr(m_T(h) \geq \gamma) > 0$：
$$\mathbb{E}[w_T(h) \mid m_T(h) \geq \gamma] \leq \frac{\bar{\delta}}{q_\gamma \, c(\gamma)} \tag{公式18}$$

**Interpretation**：在informative tests上（margin $\geq \gamma$），policy几乎不能把mass放在suboptimal bet上。所以robust performance **selects for**一个internal mechanism能够decide这些action-conditioned future-observation tests——这就是minimal notion of predictive world model。

### 5.5 Theorem 3: Memory Necessity (No-Aliasing)

这是partial observability下最关键的结果。

**Setup**：
- Memory statistic $M = f(h)$，policy factor through $M$：$\pi(\cdot | h, g_T) = \pi(\cdot | M(h), g_T)$
- Pair distribution $\mathcal{P}$ over $(h, h')$ with **same last observation**（这样markovian memory会alias它们）
- Aliasing event: $\text{Alias}_M := \{(h, h') : M(h) = M(h')\}$
- Witness set $S_\gamma(h, h')$：tests使得 $p_T(h) \geq 1/2 + \gamma$ 且 $p_T(h') \leq 1/2 - \gamma$（即h, h'需要**相反**的confident bets）

**Main bound（公式19）**：
$$\bar{\delta}_{\mathcal{P}}(\pi) \geq q_\gamma^{\text{Alias}}(M) \cdot \frac{c(\gamma)}{2}$$

其中 $q_\gamma^{\text{Alias}}(M) := \Pr_{(h,h') \sim \mathcal{P}, T \sim D}((h,h') \in \text{Alias}_M \wedge T \in S_\gamma(h,h'))$。

**Contrapositive**：如果 $\bar{\delta}_{\mathcal{P}}(\pi) < q_\gamma^{\text{Alias}}(M) c(\gamma)/2$，那么 $\pi$ **cannot** be $M$-based——它必须avoid aliasing那些需要不同confident predictions的histories。

**Intuition**：如果memory把两个histories collapse到同一个state，但它们需要相反的high-confidence bets，那么policy必然在至少一个上犯错，incur constant regret。所以low regret ⟹ memory必须refine predictive-state partition。

这直接address了Richens et al. (2025)的open question：在POMDP下，不能recover full generative model，但可以establish **necessity** of belief-like memory。

---

## 六、Structured Task Families的Corollaries

### 6.1 Corollary 3: Informational Modularity

如果test distribution有block structure：$\text{supp}(D) = \bigcup_{i=1}^K \mathcal{T}_i$（disjoint blocks），每个block有自己的witness set，那么：

$$q_{\gamma, i}^{\text{Alias}}(M) \leq \frac{2\bar{\delta}_{\mathcal{P}}(\pi)}{p_i \, c(\gamma)}$$

当 $\bar{\delta}_{\mathcal{P}} \to 0$，每个block内的aliasing都vanish。**Modular task distribution selects for informational modularity**——memory必须分别handle每个block的distinctions。

### 6.2 Corollary 4: Regime Tracking

更subtle的setup：evaluation protocol先sample latent regime $I \sim \Lambda$，再从 $D_I$ sample test。Regimes可以overlap。如果两个histories $h, h'$ 在同一last observation下但属于**不同regime**，且存在test使得它们需要相反的 $\gamma$-margin bets，那么：

$$\Pr(M(h) = M(h') \wedge I(h) \neq I(h') \wedge T \in S_\gamma(h, h')) \leq \frac{2\bar{\delta}_{\mathcal{P}}(\pi)}{c(\gamma)}$$

当 $\bar{\delta}_{\mathcal{P}} \to 0$，memory必须**distinguish regime**当regime change flip optimal bet。

**Deep implication**：competence under mixture of task distributions provides normative pressure for **persistent internal variables tracking latent evaluative conditions**。Nayebi把这analogy到affective neuroscience里的homeostatic/affective modulators（Ekman 1992, Barrett 2017）——global, task-general modulation of behavior under uncertainty。

这是个很有意思的connection：emotion-like internal state不是装饰，是multi-regime robust competence的structural consequence。

参考：
- Barrett 2017: https://academic.oup.com/scan/article/12/1/1/2821634
- Affective neuroscience综述: https://en.wikipedia.org/wiki/Affective_neuroscience

### 6.3 Corollary 5: Representational Match (Platonic Convergence)

这个corollary最philosophically loaded。

定义 $\gamma$-coarsened decision profile：
$$\ell_T^\gamma(h) := \begin{cases} L & p_T(h) \geq 1/2 + \gamma \\ R & p_T(h) \leq 1/2 - \gamma \\ \perp & \text{otherwise} \end{cases}$$
$$\ell_D^\gamma(h) := (\ell_T^\gamma(h))_{T \in \text{supp}(D)}$$

即每个history的"decision signature"——在所有informative tests上的optimal bet pattern。

**假设**：
1. 两个memory representations $M_1, M_2$，各自based policy $\pi_1, \pi_2$，都achieve vanishing pair-regret
2. **$\gamma$-minimality**：$M_j(h) = M_j(h')$ whenever $\ell_D^\gamma(h) = \ell_D^\gamma(h')$（即memory不split decision-irrelevant distinctions）
3. **$\gamma$-completeness**：如果 $\ell_D^\gamma(h) \neq \ell_D^\gamma(h')$，则 $D(S_\gamma(h, h')) > 0$（即decision-relevant distinctions都有witness tests）

**结论**：$M_1$ 和 $M_2$ **equivalent up to invertible recoding**——存在可测map $\varphi, \psi$ 使得 $M_1 = \varphi(M_2), M_2 = \psi(M_1)$ a.s.

**Interpretation**：在相同evaluation family下，任何两个sufficient + minimal的low-regret agents，其internal memory states必须agree up to relabeling。

这是formal version of Platonic Representation Hypothesis (Huh et al. 2024) 和 Contravariance Principle (Cao & Yamins 2024)：sufficiently general learning pressures drive convergence toward shared statistical model of reality。Nayebi提供了formal lens——convergence不是accidental，是competence constraint的structural consequence。

参考：
- Platonic Representation Hypothesis: https://arxiv.org/abs/2405.07987
- Cao & Yamins Contravariance: https://www.sciencedirect.com/science/article/pii/S1389041724000284

---

## 七、技术亮点总结

| Component | Technical Role | Intuition |
|---|---|---|
| Binary betting reduction | 把prediction reduce成decision | Regret直接控制wrong-action mass |
| Composite goal $G_{s,a,s',k}^{(n)}$ | 把transition estimation变成binomial median estimation | 多次repeat让statistical noise消失 |
| Average regret (公式8) | 替代worst-case optimality | 更weak, 更empirically meaningful |
| Soft estimator (公式9) | 从policy choice probs recover $P$ | Optimal policy ⟹ exact median |
| PSR-style tests | 在observable层面定义predictive distinctions | 避开latent state unidentifiability |
| No-aliasing bound (公式19) | Memory necessity under POMDP | Aliasing opposite-bet histories ⟹ constant regret |
| $\gamma$-minimality + completeness | Representational match | Two minimal sufficient agents agree up to recoding |

---

## 八、与NeuroAI的Connection

Nayebi本身做NeuroAI（visual cortex, entorhinal cortex, world modeling in brain），所以discussion部分把这些formal results联系到empirical trends：

1. **Convergent representations across architectures**: Yamins et al., Khaligh-Razavi & Kriegeskorte, Nayebi et al. 等show task-optimized models predict neural responses
2. **Cross-species**: Keller et al. 2025 (larval zebrafish whole-brain) find correspondences between autonomous agents and biological brains
3. **Brain areas**: Banino et al. 2018 (grid cells in DNN), Schrimpf et al. 2021 (language), Bear et al. 2021 (Physion physical prediction)

Formal selection theorems给出了**为什么**会converge的normative account：shared competence constraints ⟹ shared predictive structure。

NeuroAI相关链接：
- Yamins lab: https://yamins.stanford.edu/
- Nayebi personal: https://www.arannayebi.com/
- Banino et al. 2018 (grid cells): https://www.nature.com/articles/s41586-018-0102-6
- Schrimpf et al. 2021 (language): https://www.pnas.org/doi/10.1073/pnas.2105646118

---

## 九、Limitations与Open Directions

虽然paper很elegant，但有几个值得思考的点：

1. **Diagnostic goal family的realism**：$G_{s,a,s',k}^{(n)}$要求agent commit到"$n$次执行 $(s,a)$"——这在真实RL setup里不太natural。Paper把这frame成"diagnostic"，但实际agent训练时不会看到这种goal。这是一个**selection vs. training**的gap：theorems说"如果agent能在这些goals上low regret"，但没说怎么train。

2. **Communicating assumption**：fully observed setup要求环境communicating，排除isolated regions。这对realistic control problems合理，但formalize了什么程度的环境structure需要more study。

3. **Non-degenerate evaluation conditions**：Theorem 2-3要求 $\eta, \eta' > 0$，即informative tests占nontrivial mass。如何verify这在实际task distribution上成立？

4. **Causal Level 3 ceiling**：Corollary 2表明counterfactuals不能从interventional kernel recover。要突破需要explicit SCM assumption——这指向model-based RL + causal structure learning的intersection。

5. **Continuous state/action**：所有results在finite spaces。Extension到continuous是open（PSR文献有some work，但necessity results没有）。

6. **Approximate vs. exact minimality**：Corollary 5的recoding result要求exact $\gamma$-minimality。Approximate version（允许small redundancy）是否robust？

---

## 十、对AI Safety/Alignment的Implication

Paper最后提到这与"AI welfare"讨论相关（Long et al. 2024）。如果strong task-general performance selects for：
- Persistent internal state
- Regime-tracking variables (analogy to affect)
- Unified predictive representations

那么advancing capability会tend to produce systems with rich internal organization。理解这些structural signatures对empirical/philosophical analysis of advanced AI relevant。

但Nayebi很careful：having world model + memory + modularity **不一定** ⟹ consciousness。可能是**how they're combined** matters。这是humble stance。

参考：
- Long et al. 2024 (AI welfare): https://arxiv.org/abs/2411.00986
- Blum & Blum 2022 (Conscious Turing Machine): https://www.pnas.org/doi/10.1073/pnas.2115934119
- Global Workspace Theory: https://en.wikipedia.org/wiki/Global_Workspace_Theory

---

## 十一、Final Intuition Recap

整篇paper的logical flow可以这样概括：

**Step 1**: 任何prediction task都可以reduce成binary betting（Lemma 1）。
**Step 2**: 在binary bet上，regret = wrong_mass × margin_factor。所以low regret ⟹ 在high-margin tests上wrong_mass小。
**Step 3**: 把transition estimation变成一连串binary bets on binomial counts（Theorem 1）。Low average regret ⟹ soft estimator recover transition kernel。
**Step 4**: 加causal Markov assumption ⟹ recover interventional kernel (Pearl Level 2)。Counterfactuals (Level 3)不行（Corollary 2）。
**Step 5**: POMDP下不能recover latent kernel，但可以用PSR-style tests。Low regret ⟹ 在informative tests上wrong_mass小（Theorem 2）。
**Step 6**: 如果memory alias两个需要相反bets的histories，必然incur regret（Theorem 3）。所以low regret ⟹ memory必须refine predictive partition。
**Step 7**: 结构化task family（block/mixture）⟹ modularity / regime-tracking（Corollary 3, 4）。
**Step 8**: Minimal + sufficient的two agents必须agree up to invertible recoding（Corollary 5）——formal Platonic convergence。

核心take-away：**Robust generalization under uncertainty compresses the space of admissible internal representations.** Performance pressure shapes structure。World models, memory, modularity, regime-tracking不是architectural assumptions，是task demands的consequence。

这跟deep learning里empirically观察到的scaling laws / emergent capabilities有flavor上的呼应——capability和structure是co-emergent的，selection theorems给了formal bridge。

希望这个讲解帮到你的intuition building, Andrej。如果想dive deeper到某个proof或connection，可以再聊。
