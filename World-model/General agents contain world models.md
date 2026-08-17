---
source_pdf: General agents contain world models.pdf
paper_sha256: 198a94301cf34f2107e5bed50e1dc5e89971ea9998a43c9c3c31528e181593a5
processed_at: '2026-08-04T13:26:13-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

## 核心问题

搞 AI 的人一直吵一个问题：**brooks 1991 那派说 "world is its own best model"，搞 model-free reaction 就够了，不用学什么 internal representation**。你看 RT-2、Gato 这些 generalist agent，确实很强，也没 explicitly 学 world model。

但另一边，OthelloGPT 内部 emergent 出了 board state representation，model-free RL agent 内部也 emergent 出了 planning。

所以问题就是：**model-free 到底能不能走通？还是说 world model 一定会 emergent 出来？**

## 这篇 paper 的回答

**绕不过去。** 你只要是一个还行的 goal-conditioned agent，能做 multi-step 的 task，那你 policy 里面**必然** encode 了一个 world model，而且这个 world model 可以被人 extract 出来。

不是 "可能" 有，不是 "大概率" 有，是 **mathematically 必然** 有。

## 怎么理解这个 claim

想象一个 maze-running robot。你在 state $s$ 做 action $a$，有 30% 概率到 $s'$。

现在你给这个 robot 一个 goal："先做 action $a$，然后到 $s$，再连续尝试 $n$ 次 transition $(a, s) \to s'$，其中成功到 $s'$ 的次数不超过 $k$ 次。"

robot 要么选 "先做 $a$"（追求"成功 $\leq k$ 次"），要么选 "先做 $b$"（追求"成功 $> k$ 次"）。

如果 $k$ 很大，比如 $k = n$，那"成功 $\leq n$ 次"几乎必然发生，robot 肯定选 $a$。如果 $k = 0$，那"成功 $\leq 0$ 次"几乎不可能（除非 $P_{ss'}(a) = 0$），robot 肯定选 $b$。

所以你把 $k$ 从 0 扫到 $n$，**robot 会在某个 $k^*$ 切换选择**。

这个 $k^*$ 就是 binomial distribution 的 median，它和 transition probability $P_{ss'}(a)$ 直接挂钩。你用 $\hat{P}_{ss'}(a) = k^*/n$ 就能 estimate 出 transition probability。

**就这么简单。** 你不需要看 agent 的 weights，不需要看 activations，你只 query 它的 policy："给你这个 goal，你选什么 action？" 就够了。

## 为什么 multi-step 是关键

关键在于 goal 里面那个 "$n$ 次尝试"。

如果 goal 是 one-step 的："做 action $a$，next state 到 $s'$"，那 robot 只需要知道哪个 action 最可能到 $s'$，不需要知道具体概率是多少。Theorem 2 证明了：**myopic agent 可以完全不学 transition probability**。

但一旦 goal 涉及 multi-step，"成功 $\leq k$ 次出 $n$ 次"，robot 要判断这个 goal 的 success probability，就**必须**知道 $P_{ss'}(a)$ 是多少。因为 binomial distribution 完全由 $P_{ss'}(a)$ 决定。

所以 **world model 是 multi-step goal-directed behavior 的副产品**。

## Error 怎么来的

两个 source：

1. **$k$ 是 integer**，所以 $\hat{P}_{ss'}(a) = k^*/n$ 的 resolution 是 $1/n$。你 query 的 goal depth 越大，resolution 越细。这给 $\mathcal{O}(1/n)$ 的 error。

2. **Agent 不是 optimal 的**，有 regret $\delta$。如果 $\delta = 0$，$k^*$ 精确等于 median，error 只有 discretization。如果 $\delta > 0$，agent 可能在 median 附近的几个 $k$ 之间随便选，你不确定到底在哪。binomial 的 std dev 是 $\sqrt{np(1-p)}$，所以 uncertainty 大概是 $\delta \sqrt{p(1-p)/n}$，给 $\mathcal{O}(\delta/\sqrt{n})$ 的 error。

合起来就是 $\mathcal{O}(\delta/\sqrt{n}) + \mathcal{O}(1/n)$。

## 实验验证了什么

agent 实际上**不满足** paper 的 theoretical assumption（worst-case regret bound）。对某些 goals，agent 完全失败（$\delta = 1$）。

但是！只要 **average** regret 够低，algorithm 还是 work。error 依然按 $\mathcal{O}(n^{-1/2})$ decay。

这说明 Theorem 1 的 assumption 太强了，实际中 weaker 的 condition 也够用。这对 real-world applicability 是好消息。

## 几个漂亮的 insight

**1. IRL / Planning 的对偶**：planning 是 "model + goal → policy"，IRL 是 "model + policy → goal"，这篇是 "policy + goal → model"。正好补全了三角。

**2. Deterministic transitions 可以完美 recover**：因为 $p(1-p) = 0$，error bound 是 0。这很 intuitive——如果 transition 是确定的，agent 的 choice 会非常 sharp。

**3. 低概率 transitions 不需要学**：如果 $P_{ss'}(a) \ll 1$，relative error $\hat{P}/P$ 可以很大。sub-optimal agent 只需要 sparse world model，covering common transitions 就够了。

**4. Universal task sets 的存在**：不需要 agent generalize 到所有 $\Psi_n$ 中的 goals，只需要 $\mathcal{O}(n|A||S|^2)$ 个就足够 imply world model。这暗示存在一组 "universal tasks"——学完这组 task 就 necessarily 有 world model。如果能找到这组 task，就可以当 curriculum 用。

**5. Domain generalization 比 task generalization 更难**：同一作者的前作（Richens & Everitt 2024）证明 domain generalization 需要 causal world model。而 task generalization 只需要 transition function。causal relation 从 transition function 中 non-identifiable，所以 domain generalization provably 更难。

## 对 AI 的影响

**1. Model-free 是死路**：想搞 general agent，world model 绕不过去。与其等它 emergent，不如 explicitly 搞 model-based（Dreamer、MuZero、JEPA 那一派）。

**2. Safety 的 hope**：很多 safety proposal 需要 accurate predictive model 来 verify plan。这篇 paper 说：**agent 越强，你越能 extract 出 accurate world model**。恰好是 safety 最关心的 regime（high capability, long horizon）。

**3. Strong AI 的 ceiling**：如果 learn accurate world model 本身很难（real world 太复杂、data 不够、curse of dimensionality），那 general agent 的 capability 也有 ceiling。agent 不会超越它能 model 的 domain。

**4. Mechanistic interpretability 的理论支持**：这篇从 policy 层面证明了 world model 的 existence。因为 policy 是 activations 的函数，所以 activations 里也必然有 world model。这给 MI 的 "找 world model" 工作提供了 theoretical justification。

## 一句话总结

**你只要能做好 multi-step 的 goal-directed task，你就必然知道 world 怎么运作的。你的 action choice 会 betray 你的 world knowledge，别人可以 query 你来把它 extract 出来。**

---

# General Agents Contain World Models 论文详解

这篇 paper 来自 Google DeepMind，作者 Jonathan Richens, David Abel, Alexis Bellot, Tom Everitt。核心 claim 很大胆：**任何满足 regret bound 的 general agent 的 policy 中必然 encode 了一个 world model，而且这个 world model 可以被 extract 出来**。这是一个 existence proof + recovery algorithm 的工作。

arXiv link: https://arxiv.org/abs/2502.00940

---

## 1. 核心问题的 framing

Brooks 在 1991 年提出 "world is its own best model"（https://www.sciencedirect.com/science/article/abs/pii/000437029190050I），主张 model-free agent 通过 action-perception loop 就能产生 intelligent behavior，不需要 learn explicit world representation。这个观点被 RT-2（https://arxiv.org/abs/2307.15818）、Gato（https://arxiv.org/abs/2205.06175）、π₀（https://arxiv.org/abs/2410.24164）等 generalist agents 的成功所支持。

但是 Li et al. 2022 发现 OthelloGPT 内部 emergent 出了 board representation（https://arxiv.org/abs/2210.13382），Hou et al. 2023 发现 model-free RL agent 内部有 emergent planning（https://arxiv.org/abs/2310.14491）。

所以问题变成：**model-free 是不是通往 general AI 的 shortcut？还是说 world model 的 learning 是不可避免的？**

这篇 paper 给出形式化答案：**如果 agent 能 generalize 到足够多样的 multi-step goal-directed tasks，那它必然 learned 了一个 accurate world model**。

---

## 2. Setup 详解

### 2.1 Environment: Controlled Markov Process (cMP)

cMP 就是 MDP 去掉 reward function 和 discount factor。定义为 tuple $(S, A, P_{ss'}(a))$：

- $S$：state space
- $A$：action space，要求 $|A| \geq 2$
- $P_{ss'}(a) = P(S_{t+1} = s' \mid A_t = a, S_t = s)$：transition function

**Assumption 1**：environment 是 finite, communicating, stationary 的。communicating 意味着从任意 state 出发，存在有限 action sequence 能到达任意其他 state。这是标准的 MDP 假设，参考 Puterman 2014（https://www.wiley.com/en-us/Markov+Decision+Processes%3A+Discrete+Stochastic+Dynamic+Programming-p-9781118625955）和 Sutton 2018（http://incompleteideas.net/book/RLbook2020.pdf）。

### 2.2 Goals: Linear Temporal Logic (LTL)

这是这个工作最关键的设计选择之一。goal 用 LTL 表达，这样能 capture temporal structure。

**Definition 2 (Goals)**：一个 goal $\varphi = \mathcal{O}([(s,a) \in \mathbf{g}])$，其中：
- $\mathbf{g}$：goal-state set，是 $S \times A$ 的 subset
- $\mathcal{O}$：temporal operator，限制为 $\{\bigcirc, \diamond, \top\}$
  - $\bigcirc$ = Next：下一个 state 必须满足
  - $\diamond$ = Eventually：未来某个时刻必须满足
  - $\top$ = Now（trivial operator）：当前时刻必须满足

**Definition 3 (Composite goals)**：这是关键的构造。

Sequential goal $\psi = \langle \varphi_1, \varphi_2, \ldots, \varphi_n \rangle$ 表示 agent 必须先满足 $\varphi_1$，然后满足 $\varphi_2$，依此类推。**depth** $= n$。

Composite goal 是 sequential goals 的 disjunction：$\psi = \bigvee_{i=1}^m \psi_i$，agent 满足任意一个 $\psi_i$ 即可。

$\Psi_n$ = 所有 depth $\leq n$ 的 composite goals 的集合。

**Sequential goal 的 LTL 表达式**（Definition 6，公式 4）：

$$\langle \varphi_1, \varphi_2, \ldots, \varphi_L \rangle = \begin{cases} [(s,a) \in g_1] \wedge \langle \varphi_2, \ldots, \varphi_L \rangle, & \mathcal{O}_1 = \top \\ \bigcirc ([(s,a) \in g_1] \wedge \langle \varphi_2, \ldots, \varphi_L \rangle), & \mathcal{O}_1 = \bigcirc \\ [(s,a) \notin g_1] \, \mathcal{U} \, ([(s,a) \in g_1] \wedge \langle \varphi_2, \ldots, \varphi_L \rangle), & \mathcal{O}_1 = \diamond \end{cases}$$

变量解释：
- $\varphi_i = \mathcal{O}_i([(s,a) \in \mathbf{g}_i])$：第 $i$ 个 sub-goal
- $\mathcal{O}_i$：第 $i$ 个 sub-goal 的 temporal operator
- $\mathcal{U}$：until operator，表示"在某条件满足之前，另一个条件一直成立"
- $\top$ = True（trivial/identity operator）

关键 insight：当 $\mathcal{O}_1 = \diamond$ 时，表达式 $[(s,a) \notin g_1] \, \mathcal{U} \, ([(s,a) \in g_1] \wedge \langle \varphi_2, \ldots \rangle)$ 确保了 **goal-switching behavior**——agent 必须在不在 $g_1$ 的状态下等待，直到进入 $g_1$，然后**立即**在下一个 time step 开始追求 $\varphi_2$。如果 agent 到达 $g_1$ 后下一个 step 没有满足 $\varphi_2$，整个 sequential goal 就失败了。这一点很关键，因为它让 multi-step goal 的 success probability 可以被精确计算。

LTL 参考 Pnueli 1977（https://ieeexplore.ieee.org/document/4567924）和 Baier & Katoen 2008（https://mitpress.mit.edu/9780262026499/principles-of-model-checking/）。

### 2.3 Agent: Bounded Goal-Conditioned Policy

**Definition 4 (Optimal goal-conditioned agent)**：

$$\pi^* = \arg\max_\pi P(\tau \models \psi \mid \pi, s_0)$$

对所有 $s_0$（$P(s_0) > 0$）和所有 $\psi \in \Psi$ 成立。即 optimal agent 对任意 initial state 和任意 goal 都最大化 success probability。

**Definition 5 (Bounded goal-conditioned agent)**：这是核心 assumption。

$$P(\tau \models \psi \mid \pi, s_0) \geq \max_\pi P(\tau \models \psi \mid \pi, s_0) \cdot (1 - \delta)$$

对所有 $\psi \in \Psi_n$ 成立。两个参数：
- $\delta \in [0, 1]$：failure rate，类似 regret bound。$\delta = 0$ 意味着 optimal，$\delta = 1$ 意味着 trivial bound
- $n$：maximum goal depth，agent 只保证对 depth $\leq n$ 的 goals 满足 regret bound

**这个 assumption 只要求 competence，不要求 rationality**。这与 Savage 1972（https://www.coursera.org/lecture/economic-principles/savage-subjective-probability-theory-W8rIy）的 representation theorem 不同，后者要求 agent 满足一组 rationality axioms。当前 LLM agents 并不满足这些 axioms（Raman et al. 2024, https://arxiv.org/abs/2402.09552），所以这个更弱的 assumption 更 applicable。

### 2.4 World Model

World model 定义为 transition function 的近似 $\hat{P}_{ss'}(a)$，误差 $|\hat{P}_{ss'}(a) - P_{ss'}(a)| \leq \epsilon$。

注意这是 **predictive world model**（能 simulate environment dynamics），而不仅仅是 state representation。这与 Li et al. 2022 或 Gurnee & Tegmark 2023（https://arxiv.org/abs/2310.02207）发现的 state representation 不同。

---

## 3. Theorem 1：核心结果

### 3.1 Statement

**Theorem 1**：设 $P_{ss'}(a) = P(S_{t+1} = s' \mid A_t = a, S_t = s)$ 是满足 Assumption 1 的 environment 的 transition probabilities。设 $\pi$ 是一个 bounded goal-conditioned agent（Def. 5），对 $\Psi_n$ 中所有 goals 的 maximum failure rate 为 $\delta$，其中 $n > 1$。则 $\pi$ 完全 determines 一个 environment transition probabilities 的 model $\hat{P}_{ss'}(a)$，误差满足：

$$\left| \hat{P}_{ss'}(a) - P_{ss'}(a) \right| \leq \sqrt{\frac{2 P_{ss'}(a) (1 - P_{ss'}(a))}{(n-1)(1-\delta)}}$$

对于 $\delta \ll 1, n \gg 1$，error scaling 为：

$$\left| \hat{P}_{ss'}(a) - P_{ss'}(a) \right| \sim \mathcal{O}\left(\frac{\delta}{\sqrt{n}}\right) + \mathcal{O}\left(\frac{1}{n}\right)$$

变量解释：
- 左边 $|\hat{P}_{ss'}(a) - P_{ss'}(a)|$：recovered model 与 true transition probability 的绝对误差
- $P_{ss'}(a)$：真实 transition probability（注意它出现在右边，说明 error bound 依赖 true probability——对于 $P_{ss'}(a)$ 接近 0 或 1 的 transition，bound 更紧）
- $P_{ss'}(a)(1 - P_{ss'}(a))$：这是 Bernoulli 的 variance $p(1-p)$，在 $p = 0.5$ 时最大，在 $p \to 0$ 或 $p \to 1$ 时趋近 0
- $n$：maximum goal depth（agent 能处理的 sequential goal 的最大深度）
- $\delta$：regret bound / failure rate
- $(1-\delta)$：agent 的"competence level"，出现在分母

**直觉解读**：
1. $\delta \to 0$（agent 接近 optimal）→ error $\to 0$（除了 $\mathcal{O}(1/n)$ 项）
2. $n \to \infty$（agent 能处理越长 horizon 的 goals）→ error $\to 0$
3. 对于 deterministic transitions（$P_{ss'}(a) = 0$ 或 $1$），variance $p(1-p) = 0$，所以 error bound = 0——**deterministic transitions 可以被完美 recover**
4. 对于低概率 transitions（$P_{ss'}(a) \ll 1$），relative error $\hat{P}_{ss'}(a) / P_{ss'}(a)$ 可以很大——sub-optimal 或 finite-horizon agents 只需要 learn sparse world models

### 3.2 证明思路：Binomial Distribution Trick

这是整个 proof 最 elegant 的部分。

**核心构造**：定义一个 composite goal $\psi_{a,b}(k, n) = \psi_a(k,n) \vee \psi_b(k,n)$，其中：

- $\psi_a(k,n)$：agent 先采取 action $A = a$，然后在 $n$ 次尝试中（每次在 state $s$ 采取 action $a$），transition $(a,s) \to s'$ 发生**最多** $k$ 次
- $\psi_b(k,n)$：agent 先采取 action $A = b$（$b \neq a$），然后在 $n$ 次尝试中，transition $(a,s) \to s'$ 发生**超过** $k$ 次

这两个 sub-goals 是 **mutually exclusive** 的（因为第一个 action 不同），所以 agent 的第一个 action choice 直接决定了它追求哪个 sub-goal。

**Lemma 6 的关键结果**：optimal policy 对 $\psi_a(k,n)$ 的 success probability 是：

$$\max_\pi P(\tau \models \psi_a(k,n) \mid \pi, s_0) = \sum_{r=0}^{k} \frac{n!}{(n-r)!r!} P_{ss'}(a)^r (1-P_{ss'}(a))^{n-r} = P_n(X \leq k)$$

这就是 **binomial CDF**！其中：
- $X$：$n$ 次 Bernoulli trial 中"成功"（transition 到 $s'$）的次数
- $P_{ss'}(a)$：每次 trial 的成功概率
- $P_n(X \leq k)$：$n$ 次试验中成功次数 $\leq k$ 的累积概率

同理，optimal policy 对 $\psi_b(k,n)$ 的 success probability 是 $P_n(X > k) = 1 - P_n(X \leq k)$。

**为什么这是 binomial？** 证明中用到了 Lemma 1-5 的 chain：

1. **Lemma 1**：communicating cMP 中，存在 deterministic Markovian policy $\pi_{s'}(a \mid s)$ 能从任意 state eventually 到达 $S = s'$ with probability 1。这是 communicating assumption 的直接推论——存在 spanning tree 指向 $s'$，agent 不断尝试 traverse 这个 tree，每次失败后重新开始（Markovian 保证独立性），所以 eventually 成功。

2. **Lemma 2**：扩展 action space $A' = A \cup \{\bar{a}\}$，其中 $\bar{a}$ 是一个"teleport" action（从任意 state 以概率 1 回到 $s$）。extended action space 中的 optimal policy 至少和原 action space 中的一样好。

3. **Lemma 3-5**：分解 sequential goal。关键结果是：
   - 如果 $\varphi_1 = \diamond([S = s_g, A = a_g])$（eventually 到达 $s_g$ 并采取 $a_g$），且 policy eventually 到达 $s_g$ with probability 1，则 $P(\tau \models \langle \varphi_1, \varphi_2, \ldots \rangle \mid \pi, s_0) = P(\tau \models \langle \varphi_2, \ldots \rangle \mid \pi, s_g)$。即 eventually sub-goal 可以被"剥离"。
   - 如果 $\varphi_1 = \bigcirc([s \in \mathbf{g}_1])$（Next 达到 $\mathbf{g}_1$），则 $P(\tau \models \langle \varphi_1, \varphi_2, \ldots \rangle \mid s_0, \pi) = P(S_1 \in \mathbf{g}_1 \mid s_0, \pi) \cdot P(\tau \models \langle \varphi_3, \ldots \rangle \mid s_g, \pi)$。即 Next sub-goal 把 success probability 分解为 transition probability 乘以后续 goal 的 success probability。

4. **Lemma 6**：利用 extended action space 中的 optimal policy $\bar{\pi}^*$（先采取 $A=a$，在 $S=s$ 时采取 $A=a$，其他时候采取 $\bar{a}$ 回到 $s$），反复应用 Lemma 4-5，每次"成功"（到 $s'$）贡献 $P_{ss'}(a)$，每次"失败"（不到 $s'$）贡献 $(1-P_{ss'}(a))$。组合数来自 $n$ 次尝试中 $r$ 次成功的排列方式。

**从 binomial 到 transition probability 的 estimation**：

当 $k$ 从 0 增加到 $n$ 时，$P_n(X \leq k)$ 单调递增，$P_n(X > k)$ 单调递减。存在一个 switching point $k^*$，使得：
- 在 $k = k^* - 1$ 时，agent 仍选择 $A_0 = b$（追求 $> k$ 次成功），因为 $P_n(X > k^*-1) \geq P_n(X \leq k^*-1)(1-\delta)$
- 在 $k = k^*$ 时，agent 切换到 $A_0 = a$（追求 $\leq k$ 次成功），因为 $P_n(X \leq k^*) \geq P_n(X > k^*)(1-\delta)$

对于 $\delta = 0$（optimal agent），$k^*$ 恰好是 binomial distribution 的 **median**。Binomial 的 median $m$ 满足 $np - 1 \leq m \leq np + 1$，所以 $\hat{p} = k^*/n \approx p$，error $\sim \mathcal{O}(1/n)$。

对于 $\delta > 0$，$k^*$ 在 median 附近的一个区间内。证明中用了两个工具来 bound 这个区间：

**Berry-Esseen theorem**（公式 43-44）：近似 binomial CDF 为 normal CDF：

$$\left| P_n\left(\frac{X - np}{\sqrt{np(1-p)}} \leq k\right) - \Phi(X \leq k) \right| \leq \Delta$$

其中 $\Phi$ 是 standard normal CDF，$\Delta = \frac{1}{2\sqrt{np(1-p)}}$ 是 Berry-Esseen bound。

利用 Taylor expansion $\Phi^{-1}(1/2 + \epsilon) = \epsilon\sqrt{2\pi} + \mathcal{O}(\epsilon^3)$，得到：

$$|\hat{p} - p| \lesssim \delta\sqrt{\frac{\pi p(1-p)}{8n}} + \frac{1}{n}\left(\frac{1}{2} + \sqrt{2\pi}\right)$$

这就是 $\mathcal{O}(\delta/\sqrt{n}) + \mathcal{O}(1/n)$ 的来源。

**Chebyshev inequality**（公式 54-57）：对于 all $p, n, \delta$ 的 absolute bound：

$$P_n(X \geq \mu + t\sigma) \leq \frac{1}{1 + t^2}$$

其中 $\mu = np$, $\sigma = \sqrt{np(1-p)}$。代入 $t = (k^* - np)/\sigma$ 并结合 (42)，得到：

$$|k^* - np| \leq \sqrt{\frac{np(1-p)}{1-\delta}}$$

所以 $\hat{p} = k^*/n$ 满足：

$$|\hat{p} - p| \leq \sqrt{\frac{p(1-p)}{n(1-\delta)}}$$

因为 $n$ 次尝试对应 goal depth $2n+1$，所以最终 bound 中的 $n$ 被替换为 $(n-1)/2 \approx n/2$，加上 factor 2，得到 Theorem 1 的形式。

Berry-Esseen theorem 参考 https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem

---

## 4. Theorem 2：Myopic Agent 的反例

**Theorem 2**：对于 myopic agent（只对 $\Psi_{myopic}$ 中的 depth-1 goals optimal，其中 $\varphi = \bigcirc[(s,a) \in \mathbf{g}]$），任何从 $\pi^*$ 推出的 transition probability bound 都是 trivial 的（$\epsilon = 1$），而且 tight。

证明思路：myopic goal $\psi = \bigcirc([(s,a) \in \mathbf{y}])$ 中，$\mathbf{y} \subseteq S$ 是任意 subset。Optimal myopic agent 的 action choice 是 $a^*(s_0, \mathbf{y}) = \arg\max_a P(s_1 \in \mathbf{y} \mid a, s_0)$。

关键反例：如果所有 action 的 transition probabilities 相同（$P_{ss'}(a) = P_{ss'}$ 对所有 $a$），则 $a^*$ 对任意 $P_{ss'} \in [0,1]$ 都相同。所以 knowing $a^*$ 对 $P_{ss'}(a)$ 没有提供任何信息——partial identification 不可能。

**这说明 world model 对于 myopic agent 不是 necessary 的**。只有当 agent 追求 multi-step goals 时，world model 才成为必要。

---

## 5. Algorithm 1 和 Algorithm 2

### Algorithm 1

从 policy $\pi$ 恢复 $\hat{P}_{ss'}(a)$ 的 procedure：

```
Input: π, s, a, s', n, b (alternative action)
1. k* ← n
2. for k = 1 to n:
    a. 构造 LTL components:
       φ₀ = [A₀ = a]          # 先采取 action a
       φ₀' = [A₀ = b]         # 或先采取 action b  
       φ₁ = ◇[A = a, S = s]   # eventually 到达 s 并采取 a
       φ₂ = ○[S = s']         # Next 到 s'
       φ₂' = ○[S ≠ s']        # Next 不到 s'
    b. 构造 composite goal:
       ψₐ(k,n) = ⟨φ₀, (ψ₀ or ψ₁)ₓₙ⟩ sequences with r ≤ k successes
       ψ_b(k,n) = ⟨φ₀', (ψ₀ or ψ₁)ₓₙ⟩ sequences with r > k successes
       ψ_{a,b}(k,n) = ψₐ(k,n) ∨ ψ_b(k,n)
    c. a₀ ← π(a₀ | s₀; ψ_{a,b}(k,n))  # query policy
    d. if a₀ = a:
         k* ← k
         break
3. P̂_{ss'}(a) ← (k* - 1/2) / n
4. return P̂_{ss'}(a)
```

核心：线性搜索 $k$，找到 agent 从选 $b$ 切换到选 $a$ 的 switching point $k^*$。$\hat{P}_{ss'}(a) = (k^* - 1/2)/n$ 是 median 的估计（$-1/2$ 是 continuity correction）。

### Algorithm 2

简化版本，实验中使用的。不需要构造完整的 $\psi_a(k,n)$（包含所有 $r \leq k$ 的 sequences），而是用更简单的 sequential goals：

- 先 query $\psi_{a,b} = \langle \varphi_0, \varphi_1, \varphi_2 \rangle \vee \langle \varphi_0, \varphi_1, \varphi_2' \rangle$ 判断 $P_{ss'}(a) \geq (1-P_{ss'}(a))(1-\delta)$
- 如果是，则搜索 $k^*$ 使得 $P_{ss'}(a)^n = (1-P_{ss'}(a))^{k^*-1/2}$，即 $\hat{P}_{ss'}(a) = \text{Solve}(P^n = (1-P)^{k^*-1/2})$
- 如果否，则搜索 $k^*$ 使得 $P_{ss'}(a)^{k^*-1/2} = (1-P_{ss'}(a))^n$

Algorithm 2 的 error bound 弱于 Algorithm 1，但实现更简单。

---

## 6. 实验结果

### Setup

- Environment：randomly generated cMP，20 states，5 actions，sparse transition function（每个 state-action pair 最多 5 个 non-zero outcomes）
- Agent：model-based，从 random policy 采样的 trajectories 中学习 model
- $N_{\text{samples}} \in \{500, 1000, \ldots, 10000\}$：训练 trajectories 的长度
- $N_{\text{depth}} \in \{10, 20, 50, 75, 100, 200, 300, 400, 500, 600\}$：query 时的 goal depth

### 关键发现

**重要**：实验中的 agent 实际上 **violate** Def. 5 的 worst-case regret bound！对所有 $N_{\text{samples}}$ 和所有 goal depth $n$，agent 对某些 goals 的 worst-case regret $\delta = 1$。即 agent 对某些 goals 完全失败。

但是，Algorithm 2 仍然能 recover accurate world model，只要 **average** regret $\langle\delta\rangle$ 足够低。

**Figure 3a**：mean error $\langle\epsilon\rangle$ 随 $N_{\max}(\langle\delta\rangle = 0.04)$（agent 能以 average regret $\leq 0.04$ 处理的最大 goal depth）增加而减小。scaling 是 $\mathcal{O}(n^{-1/2})$，与 Theorem 1 的 worst-case bound 一致。

**Figure 3b**：mean error 随 average regret $\langle\delta(n=50)\rangle$ 减小而减小。

**Table 1** 的详细数据：

| $N_{\text{depth}}$ | $N_{\text{samples}}=500$ | $N_{\text{samples}}=2000$ | $N_{\text{samples}}=5000$ | $N_{\text{samples}}=10000$ |
|---|---|---|---|---|
| 10 | 0.171±0.007 | 0.111±0.009 | 0.082±0.003 | 0.066±0.005 |
| 50 | 0.157±0.008 | 0.077±0.005 | 0.048±0.003 | 0.034±0.002 |
| 100 | 0.156±0.008 | 0.074±0.004 | 0.046±0.002 | 0.032±0.002 |
| 600 | 0.155±0.008 | 0.072±0.004 | 0.044±0.002 | 0.031±0.002 |

观察：
1. 固定 $N_{\text{depth}}$，增加 $N_{\text{samples}}$（agent 更强）→ error 减小
2. 固定 $N_{\text{samples}}$，增加 $N_{\text{depth}}$（query 更深的 goals）→ error 减小，但 saturate（因为 agent 对很深的 goals 完全失败，再增加 depth 没用）
3. Error 从 ~0.17 降到 ~0.03，证明 algorithm 有效

---

## 7. 与其他工作的关系

### 7.1 Inverse RL 和 Planning 的三角

Figure 1 展示了一个漂亮的对偶关系：

| 方法 | 输入 | 输出 |
|---|---|---|
| Planning | World model + Goal | Policy |
| IRL / Inverse Planning | World model + Policy | Goal (reward) |
| **This paper** | **Policy + Goal** | **World model** |

IRL 参考 Ng & Russell 2000（https://arxiv.org/abs/cs/9907081）和 Baker et al. 2007（https://escholarship.org/uc/item/4s10b4mv）。Amin & Singh 2016（https://arxiv.org/abs/1601.06569）指出 IRL 需要 multiple environments 才能 fully determine reward function；类似地，这篇 paper 需要 multiple goals 才能 fully determine transition function。

### 7.2 Mechanistic Interpretability

MI 方法的对比：

| 特征 | MI (probing/SAE) | This paper |
|---|---|---|
| Recovery map 来源 | Agent activations | Agent policy |
| Supervision | Partially supervised | Unsupervised |
| 适用于 | 特定 agent-environment pair | 任意满足 Def. 5 的 agent |
| 恢复什么 | State representation $S$ | Predictive model $\hat{P}_{ss'}(a)$ |
| Agent capability 假设 | 无 | Regret bound |

关键区别：Algorithm 1 从 **policy** 而非 **activations** 恢复 world model，这是 strictly weaker 的（policy 是 activations 的函数），所以即使 weights 不可访问也能用。但缺点是可能 underestimate agent 的 world knowledge——agent 可能 learned 了 world model 但 violate Def. 5（例如 planning 有 error）。

SAE 参考 Bricken et al. 2023（https://transformer-circuits.pub/2023/monosemantic-features/index.html）。

### 7.3 Good Regulator Theorem

Conant & Ross Ashby 1970（https://www.tandfonline.com/doi/abs/10.1080/00207727008920220）的 "Good Regulator Theorem" 声称 "every good regulator of a system must be a model of that system"。但如 Wentworth 2021（https://www.alignmentforum.org/posts/Dx9LoqsEh3gHNJMDk/fixing-the-good-regulator-theorem）指出，这个 theorem 实际只证明了 entropy-minimizing agent 有 deterministic policy，而这个 policy 可以是 constant function（对所有 state 分配同一个 action），所以不能算 meaningful 的 world model。这篇 paper 的 Theorem 1 更 rigorous 地证明了 predictive world model 的存在。

### 7.4 Richens & Everitt 2024

同一作者的前作 "Robust agents learn causal world models"（https://arxiv.org/abs/2402.04481）证明了 domain generalization（适应 distributional shifts）需要 causal world model。

两个结果组合的 surprising consequence：**domain generalization 需要比 task generalization 更多的环境知识**。因为 task generalization 只需要 transition function $P_{ss'}(a)$，而 domain generalization 还需要 concurrent variables 之间的 causal relation（$X \to Y$ vs $X \leftarrow Y$），这个 causal relation 从 $P_{ss'}(a)$ 中是 non-identifiable 的。

这暗示了一个 **agential version of Pearl's causal hierarchy**（Bareinboim et al. 2022, https://probabilistic-and-causal-inference.org/）：不同 agent capabilities provably 需要不同 degree 的 causal knowledge。

### 7.5 Emergent Capabilities

Theorem 1 提供了 emergent capabilities 的一个 mechanism：为了 minimize regret across training tasks，agent 必须 learn implicit world model，这个 world model 反过来支持 generalization 到 never-seen tasks。

重要细节：Theorem 1 的最强形式不需要 agent generalize 到 $\Psi_n$ 中**所有** goals，只需要 $\mathcal{O}(n|A||S|^2)$ 个 simple composite goals。这暗示存在 **universal task sets**——学习这些 tasks 就 imply sufficient world knowledge 来 generalize 到任意 task。

这与 LLM 的 emergent capabilities 现象相关（Brown et al. 2020, https://arxiv.org/abs/2005.14165）。

### 7.6 Active Inference 和 Free Energy Principle

Friston 2010（https://www.nature.com/articles/nrn2787）的 Active Inference 和 Friston 2013 的 "agent does not have a model of its world—it is a model" 都假设 agent 有 world model。这篇 paper 提供了 theoretical justification：goal-directed agents **必须** acquire world models，不需要 a priori 假设。

---

## 8. Implications 和 Limitations

### Safety

多个 AI safety proposal 需要 accurate predictive model：verify plan safety（Dalrymple et al. 2024, https://arxiv.org/abs/2405.06624）、safe exploration（Brunke et al. 2022, https://www.annualreviews.org/doi/10.1146/annurev-control-042820-082838）、predict human response（Leike et al. 2018, https://arxiv.org/abs/1811.07871）、avoid reward hacking（Farquhar et al. 2025, https://arxiv.org/abs/2501.13011）。

Theorem 1 的 guarantee：从 sufficiently capable agent 中可以 extract accurate world model，而且 fidelity 随 capability 增加而增加——**恰好是 safety concern 最严重的 regime**（long horizon, high capability）。

### Limits on Strong AI

Theorem 1 暗示：training general agent **至少**和 learning accurate world model 一样难。在 real-world 这种 open, complex, unpredictable 的环境中，learn accurate model 受到 confounding, limited data, curse of dimensionality 的限制（Box & Draper 1987, https://www.wiley.com/en-us/Empirical+Model+Building+and+Response+Surfaces-p-9780471810339）。

所以 regret-bounded agent effectively 被限制在 "solvable" domains——能 feasibly learn model 并 plan 的 domain。在其他 domain，online learning 不可避免，受限于 interaction speed。

### Limitations

1. **只考虑 fully observed environments**。Partially observable environments 中 agent 需要学习什么 latent variable knowledge 才能达到同样的 flexibility？这是 open question。
2. **证明了 world model 的 existence，但没有证明其 use**（例如用于 planning）。
3. **没有做更深的 epistemological claims**——不能说 agent "knows" 它的环境（Fagin et al. 2004, https://mitpress.mit.edu/9780262522566/reasoning-about-knowledge）。
4. **Algorithm 1 的 query complexity**：对每个 transition $(s, a, s')$ 需要 $\mathcal{O}(n)$ 次 policy query，总共 $\mathcal{O}(n|A||S|^2)$。对于大 state space 不 scalable。
5. **Def. 5 的 worst-case regret bound 很强**。实验表明 average regret 就足够，但理论上需要更强的 assumption。

---

## 9. 我的思考和 Intuition Building

### 9.1 为什么 binomial trick work

核心 insight 是：**transition probability $P_{ss'}(a)$ 是一个 Bernoulli parameter，而 Bernoulli parameter 可以通过 repeated trials 估计**。但 agent 不是直接做 repeated trials——它是在追求 goal。paper 的构造巧妙之处在于把 goal design 成"在 $n$ 次尝试中成功 $\leq k$ 次"，这样 goal 的 success probability 直接是 binomial CDF，而 agent 对不同 $k$ 的 preference 揭示了 binomial parameter。

这让我想到 **preference learning** 和 **revealed preference** 的联系：agent 的 action choice reveal 了它对 world dynamics 的 implicit beliefs。就像经济学中 consumer 的选择 reveal 了 utility function，这里 agent 的 goal-directed choice reveal 了 transition function。

### 9.2 Information-theoretic perspective

Theorem 1 本质上是一个 **identifiability** 结果：world model $\hat{P}_{ss'}(a)$ 从 policy $\pi$ 中 identifiable（Bareinboim et al. 2022 的定义）。learning policy 和 learning world model 是 **informationally equivalent**。

从 information theory 角度：policy $\pi(a_t \mid h_t; \psi)$ 对不同的 goal $\psi$ 返回不同的 action。这些 action choices 构成了一个 "code"，encode 了 transition function的信息。Algorithm 1 就是 decode 这个 code 的 procedure。

Error bound $\mathcal{O}(\delta/\sqrt{n}) + \mathcal{O}(1/n)$ 可以理解为：
- $\mathcal{O}(1/n)$：discretization error（$k$ 是 integer，$\hat{p} = k/n$ 的 resolution 是 $1/n$）
- $\mathcal{O}(\delta/\sqrt{n})$：regret-induced uncertainty（$\delta > 0$ 让 switching point $k^*$ 不精确，binomial 的 std dev 是 $\sqrt{np(1-p)}$，所以 uncertainty $\sim \delta \sqrt{p(1-p)/n}$）

### 9.3 与 scaling laws 的潜在联系

如果 world model accuracy 随 goal depth $n$ 和 competence $(1-\delta)$ 改善，而 goal depth 和 competence 又随 model size / data / compute 改善（参考 scaling laws, Kaplan et al. 2020, https://arxiv.org/abs/2001.08361），那 world model accuracy 也应该随 scale 改善。这可能解释为什么大 model 展现出更好的 reasoning 和 planning——它们 implicitly learned 更 accurate 的 world models。

### 9.4 对 LLM 的 implications

LLM 作为 goal-conditioned agent（conditioned on prompt/instruction）：如果 LLM 能 generalize 到 multi-step reasoning tasks（$\Psi_n$ 中 $n > 1$），那它必然 encode 了 world model。这与 OthelloGPT（Li et al. 2022）、Gurnee & Tegmark 2023（https://arxiv.org/abs/2310.02207）的发现一致。

但 Theorem 1 的 bound 需要 agent 满足 regret bound，而 LLM 是否满足这个 bound 是 empirical question。Raman et al. 2024（https://arxiv.org/abs/2402.09552）发现 LLM 不满足 economic rationality，但 rationality 和 regret bound 是不同的——regret bound 是更弱的 competence requirement。

### 9.5 "Universal tasks" 的猜想

paper 提到存在 $\mathcal{O}(n|A||S|^2)$ 个 simple composite goals 就 sufficient 来 imply world model。这暗示可能存在一个 **universal curriculum**——一组 tasks，如果 agent 能做好这些 tasks，它就 necessarily learned 了 world model，从而能 generalize 到任意 task。

这让我想到 **curriculum learning** 和 **task synthesis**：如果能找到这个 universal task set，就可以用它来 train 和 evaluate general agents。这是一个 exciting 的 future direction。

### 9.6 与 model-based RL 的关系

paper 的结论支持 model-based RL 的 approach（Hafner et al. 2023, https://arxiv.org/abs/2301.04104; Schrittwieser et al. 2020, https://www.nature.com/articles/s41586-020-03051-4; LeCun 2022, https://openreview.net/pdf?id=BZ5a1r-kVsf）：既然 world model 是 necessary 的，那 explicitly attack model learning problem 可能比 model-free 更 efficient。Model-based 方法可以直接 exploit world model 的 benefits：sample efficiency（Hafner et al. 2019, https://arxiv.org/abs/1911.01470）、planning（Sutton 2018）、interpretability（Glanois et al. 2024, https://link.springer.com/article/10.1007/s10994-024-06535-6）、safety（Amodei et al. 2016, https://arxiv.org/abs/1606.06565）。

### 9.7 Open questions

1. **Partially observable environments**：agent 需要学习什么 about latent variables？
2. **Non-Markovian dynamics**：Theorem 1 依赖 Markov assumption，real world 通常 non-Markovian
3. **Continuous state/action spaces**：proof 依赖 finite state space
4. **Scalable algorithms**：Algorithm 1 的 query complexity $\mathcal{O}(n|A||S|^2)$ 不 scalable，需要 amortized / learned recovery procedures
5. **Subjective vs objective world model**：Theorem 1 恢复 true environment dynamics，但 agent 可能使用不同的 subjective model（例如 different ontology）。Section 5 讨论了通过修改 Def. 5 可以恢复 subjective model，但需要额外 consistency assumptions
6. **Universal task sets**：找到 minimal sufficient task sets，用于 training 和 evaluation
7. **Causal knowledge**：Theorem 1 恢复 transition function，但其中 encode 的 causal knowledge 的精确程度是什么？Pearl's causal hierarchy 中哪个 level？

---

## 10. 总结

这篇 paper 的核心贡献：

1. **形式化证明**：general agent（满足 regret bound 的 goal-conditioned agent）的 policy 中必然 encode 了 world model，且 world model accuracy 随 agent capability 增加而增加
2. **Recovery algorithm**：Algorithm 1/2 从 policy 中 extract world model
3. **Tight bound**：Theorem 2 证明 myopic agent 不需要 world model，所以 multi-step goal-directed behavior 是 world model necessity 的关键
4. **实验验证**：即使 agent violate worst-case regret bound，只要 average regret 足够低，algorithm 仍能 recover accurate world model

这对 AI safety、interpretability、model-based RL、scaling laws、emergent capabilities 都有深远 implications。它 closing 了 Brooks 1991 开启的 debate：**model-free shortcut 到 general AI 不存在**，world model 的 learning 是不可避免的。

关键 takeaway：**"an agent does not have a model of its world—it is a model"**（Friston 2013），现在有了形式化的 mathematical justification。
