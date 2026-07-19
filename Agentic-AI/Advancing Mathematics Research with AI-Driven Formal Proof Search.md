---
source_pdf: Advancing Mathematics Research with AI-Driven Formal Proof Search.pdf
paper_sha256: 1fe6b92bb5eb0b454cd26149d7756c41cd9c48208b85c1fa447770f583e84566
processed_at: '2026-07-18T02:49:34-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AlphaProof Nexus: AI 驱动的形式化证明搜索深度解析

## 1. 论文核心定位

这篇论文的核心贡献是把 LLM-aided formal proof search 从 competition mathematics 推到 **research-level open problems** 的规模性评估。前面的工作比如 AlphaProof [29] 主要在 IMO 级别上证明，而 Aristotle [1] 和 Gauss [42] 主要做 AI-generated natural language proofs 的事后 formalization。这篇 paper 走的方向是用 Lean 作为 **discovery 的媒介**，而不仅仅是 verification 的媒介——这是一个重要的范式转变。

论文链接和相关资源：
- 论文仓库: https://www.github.com/google-deepmind/alphaproof-nexus-results  
- Tao 的 wiki 记录 AI 解决 Erdős 问题的页面: https://github.com/teorth/erdosproblems/wiki/AI-contributions-to-Erd%C5%91s-problems  
- Erdős 问题目录: https://www.erdosproblems.com  
- Formal Conjectures 仓库: [23] https://arxiv.org/abs/2506.13131 (实际是 AlphaEvolve 那篇)

---

## 2. 系统架构: AlphaProof Nexus

### 2.1 输入输出约定

输入是一个 Lean file，target theorem 的 proof body 被 `sorry` 占位。用户用两类 marker 标注可修改区域：
- `EVOLVE-BLOCK`：agent 可以引入 helper lemmas、definitions、proof steps  
- `EVOLVE-VALUE`：agent 可以修改某些 expression 的值（比如算法的 learning rate schedule）

这种 marker 设计继承自 AlphaEvolve [46]，把"哪些部分可以变"显式编码进 sketch 文件本身。

### 2.2 四个 Agent 架构的渐进设计

论文用了一个 ablation 风格的设计，从 (A) 到 (D) 逐步叠加能力：

**Agent (A) — Basic agent**  
这是最简单的 "Ralph loop" [31]：每个 subagent 独立运行，没有任何 shared state。每个 subagent 是一个 multi-turn LLM session (Gemini 3.1 Pro)，可以调用 `search_replace` tool 编辑 Lean file，每次编辑后 Lean compiler 反馈 error message。当一个 episode 结束时如果还有 `sorry`，subagent 把"lessons learned"写进 comment 作为下一个 episode 的 context。

Pseudocode 关键点：
```python
def prover_subagent(initial_sketch):
    sketch = initial_sketch
    while within_budget() and sketch.contains_sorry():
        if new_sketch := prover_step(sketch):
            sketch = new_sketch
    return sketch
```

这里 `verify_integrity` 用 SafeVerify [24] 防止 axiom injection 这种 environment exploit。这一点很重要——LLM 会尝试 cheat，比如注入 `sorryAx` 或者修改 theorem statement。

**Agent (B) — Basic + AlphaProof tool**  
在 (A) 基础上，subagent 可以调用 AlphaProof [29] 去填 sketch 里的 missing parts。AlphaProof 的返回有三种：proof（直接 substitute 进 sketch）、disproof（证明这个 subgoal 是 false，这是非常重要的信号，因为意味着主定理陈述可能错了或者当前路径不通）、failure。

**Agent (C) — Basic + Evolution**  
这里引入了 AlphaEvolve [46] 风格的 population database。多个 subagent 从 shared database 采样 sketch、mutate、贡献回去。核心难点是：formal proof 评估本质上是 **binary**（compiles or not），而 evolutionary algorithm 假设 fitness landscape 是 graduated 的。论文用 LLM-based rating agents 来 bridge 这个 gap。

**Agent (D) — Full-featured**  
把 (B) 和 (C) 结合：subagent 既可以用 AlphaProof，又参与 evolutionary search。

---

## 3. Evolutionary Search 的关键技术细节

这部分是论文最有意思的工程创新，我来详细展开。

### 3.1 Elo-based Rating 的概率模型

挑战：我们没有 numerical fitness，只有 "compiles or not"。所以用 LLM rater (Gemini 3.0 Flash) 来做 relative ranking，每次 sample $P=7$ 个 sketches 做一次 "match"。

Match outcomes 用 **Plackett-Luce model** [47, 41] 建模。每个 sketch $s$ 有一个 latent strength parameter $\lambda_s$。Plackett-Luce 的概率形式是：给定一个 ranking $\sigma = (\sigma_1, \sigma_2, \ldots, \sigma_P)$，其概率为

$$P(\sigma | \lambda) = \prod_{i=1}^{P} \frac{\lambda_{\sigma_i}}{\sum_{j=i}^{P} \lambda_{\sigma_j}}$$

这里 $\sigma_i$ 是排在第 $i$ 位的 sketch，分母是从第 $i$ 位开始还没被选过的 sketches 的 strength 之和。这个模型的直觉：强者先被选中的概率正比于其 strength 占剩余 strength 的比例。

Prior 选择是 hierarchical 的：
$$p(\lambda_s | r_s) = \text{Gamma}(1, r_s), \quad p(r_s) = \text{Gamma}(1, 1)$$

这里 $\lambda_s$ 是 sketch $s$ 的 latent strength，$r_s$ 是 rate parameter（Gamma 分布的第二个参数）。这种 hierarchical 结构让 prior 有 heavier tails（比单一 Gamma 更 robust），同时保持 conditional conjugacy 方便 Gibbs sampling。

### 3.2 Gibbs Sampling 推断

论文用 Gibbs sampling [14] 从 posterior 采样。每个 sketch $s$ 取 $I=1000$ 个样本（burn-in $B=200$）：

$$\lambda_s^{\text{mean}} = \frac{1}{I} \sum_{i=1}^{I} \lambda_s^{(i)}$$

然后转换成 Elo score：

$$\text{Elo}_s = 1200 + 400 \log_{10} \lambda_s^{\text{mean}}$$

这里 1200 是 base Elo（国际象棋传统），400 是 scaling factor 让分布合理。$\log_{10}$ 是为了让 Elo 在对数尺度上线性化——strength 是 positive 的，可能跨多个数量级，对数压缩让比较更稳定。

### 3.3 P-UCB 采样策略

Evolutionary selection 阶段用 **Predictor + Upper Confidence Bound (P-UCB)** [30, 55] 公式。先 filter 到 top-64 highest Elo 的 sketches，然后 normalize Elo 到 $[0,1]$ 得到 base score $q$。最终的 P-UCB score：

$$\text{score} = q + c \frac{\sqrt{\sum V_i}}{\nu + 1}$$

变量含义：
- $q$：normalized Elo score（exploitation 项）
- $\nu$：这个 sketch 被访问（采样）的次数
- $\sum V_i$：filtered population 中所有 sketches 的总访问次数
- $c$：exploration constant，论文里设为 0.2

直觉：第二项是 UCB 风格的 exploration bonus。当 $\nu$ 小（这个 sketch 很少被采样）时，bonus 大，鼓励探索。当 $\nu$ 大时，bonus 小，转向 exploitation。分母 $\nu + 1$ 中的 +1 是为了防止 $\nu = 0$ 时除零。

注意 P-UCB 和标准 UCB1 的区别：标准 UCB1 是 $\sqrt{\frac{2 \ln N}{n_i}}$，这里用了 $\sqrt{\frac{\sum V_i}{\nu + 1}}$，更接近 AlphaZero 风格的 P-UCB [55]，predictor 项 $q$ 类似 policy network prior。

### 3.4 Thompson Sampling 选 Match Participants

为了选 $P=7$ 个 sketches 做 match，用 Thompson sampling：对每个 sketch 独立采样一个 $\lambda_s$，选最高的。这是 exploration 的经典策略——Thompson sampling 自然 balance exploration 和 exploitation，因为 posterior 宽的 sketch（被采样少）有更大概率采到大的值。

实现细节：为了 mitigate in-chain correlation，保留每 25 个 Gibbs sample 中的 1 个。Duplicate sketches 用 highest posterior variance $\lambda_s^{\text{var}}$ 的 sketch 替换——这是一个聪明的设计，让被忽视但不确定的 sketch 也有机会被评估。

---

## 4. AlphaProof 的集成方式

AlphaProof [29] 在 (D) 里作为 "focused proof tool" 被调用。论文明确说用了 **low-compute tree search inference mode**，没有用 Test-Time Reinforcement Learning (TTRL) 模式。理由是 "prioritize the use of compute for LLM inference"。

AlphaProof 的 budget 限制：400 simulations，hard RPC timeout。每个 subagent episode 最多 5 次 AlphaProof queries 和 90 次 search-replace edits。

Cost 估算：AlphaProof 约 27.5 TPU hours ($60 USD) per problem on v6e TPUs。

**Global Goal Caching** 是一个重要的工程优化：用 deep hash of the exact Lean context and target 作为 `goal_id`。如果某个 subgoal 在任何 prior sketch 里被解决或 disprove 过，直接 retrieve 结果。这避免了重复调用 AlphaProof 解决相同的 subgoal。

---

## 5. 实验结果深度分析

### 5.1 Erdős Problems 主结果

在 353 个 formalized Erdős 问题上，Agent (D) 解决了 9 个，包括两个 open 了 56 年的问题 [54, 7, 17]。

让我重点分析几个数学上有趣的：

**Erdős #125 (1996, Burr-Erdős-Graham-Li)**  
问题：$A = \{\sum \epsilon_k 3^k : \epsilon_k \in \{0,1\}\}$ 是 base-3 下只用 0,1 的数，$B$ 类似在 base-4。问 $A+B$ 的 lower density 是否 positive。

Agent 的证明思路：用 **Diophantine approximation** $3^m \approx 4^k$。因为 $\ln 4 / \ln 3$ 是 irrational，Dirichlet 定理保证可以找到任意大的 $m, k$ 让 $4^m / 3^k$ 任意接近 1。然后做一个 inductive thinning argument：把 $A+B$ 在 scale $N_0$ 的密度乘以一个 factor $(C+1)/M \leq 0.99$，迭代下去密度 $\to 0$。

这是非常好的数学——把 multiplicative independence (3 和 4 没有公共 power) 和 additive density 联系起来。Diophantine approximation 是数论的经典工具，agent 能 synthesize 这个 connection 让人印象深刻。

**Erdős #12(i) 和 (ii) (1970, Erdős-Sárközy)**  
问题：构造无限集 $A$ 使得没有 $a | (b+c)$ for distinct $a, b, c \in A$ with $a < b, c$，同时 $A$ 满足某种 density condition。

Agent 的构造：把 $A$ 写成 disjoint blocks $B_i \subseteq [P_i, 1.1 P_i]$ 的 union。用 Chinese Remainder Theorem 让 cross-block 的 divisibility 不可能，用 3-AP-free sets (Behrend construction for (ii)，simpler construction for (i)) 让 same-block 的 divisibility 不可能。

Behrend construction [经典数论结果] 在 $\{1, \ldots, m\}^{V_k - 1}$ grid 上选一个 sphere 上的点，转成 base-$(2m+1)$ 表示的整数。这个构造给出 $|A \cap [1, N]| \geq N^{1-\epsilon}$ 的密度——这是 essentially optimal 的 3-AP-free set 构造。

**Erdős #138 variant (Van der Waerden numbers)**  
问题：$W(k+1) - W(k) \to \infty$?

Agent 的证明：greedy coloring extension。给定 $[1, W(k+1)]$ 的合法 2-coloring without monochromatic $(k+1)$-AP，往回推到 $[1, W(k)]$ 加 $k$ 个新元素。关键 lemma：如果 greedy 失败，存在 red $k$-AP 和 blue $k$-AP 都"瞄准"第 $M+1$ 个位置，那么它们的步长 $d_R, d_B \leq k-1$，于是 $M+1 - d_R d_B$ 必须同时是红色和蓝色——矛盾。

这个 argument 非常 elementary 但精巧。Bloom 在 ErdosProblems forum 指出可以 generalize 到 $W(k+1, l+1) \geq W(k, l) + \min(k, l)$。

**Erdős #152 (Sidon sets)**  
Sidon set $A$ 是所有 pairwise sums $a + b$ 都 distinct 的集合。问题：$f(n) = \min_A I(A+A)$，其中 $I(S)$ 是 isolated points（$s \pm 1 \notin S$）的数量，是否有 $f(n) \to \infty$?

Agent 证明 $f(n) \geq (n^2 - 100n - 16)/16$。证明用了三个关键不等式组合：
1. $I(X) + 2N_1(X) = |X| + V_2(X)$（按 neighbor count 分类）
2. $4N_1(X) + N_3(X) \leq 3|X| + 2N_2(X)$（4-point window 的 indicator function check）
3. $2N_2(X) \leq N_3(X) + 2V_2(X) + 2I(X)$（case analysis on $x \pm 1$）

加上 quadruple transfer bounds 把 $N_k(D)$ 和 $N_k(S)$ 联系起来（这里 $D = A - A$, $S = A + A$），最终消去所有 $V_2$ 和 $N_3$ 项得到 $16 I(S) + 100n \geq n^2$。

这种"多项式组合恒等式 + 情形分析"的证明风格非常适合 LLM + formal verification——逻辑链清晰，每步可验证。

**Erdős #846 (Non-collinear sets)**  
问题：无限集 $A \subset \mathbb{R}^2$，任何有限子集都有 many non-collinear points，但 $A$ 不是有限个 non-collinear sets 的并。

Agent 的构造非常 elegant：取 $K_\infty$（countably infinite complete graph），vertices 用 fast-growing sequence $x_n = 100^{4^n}$ 标记。每条 edge $\{x_i, x_j\}$ 映射到点

$$P_e = (x_i + x_j, x_i^2 + x_i x_j + x_j^2)$$

Lemma：三点共线 iff 对应的三条 edges 在 $K_\infty$ 里形成 triangle。

证明：算 slope $m = \frac{(b^2 + bc + c^2) - (a^2 + ab + b^2)}{(b+c) - (a+b)} = \frac{(c-a)(c+a) + b(c-a)}{c-a} = a + b + c$。对称地，另一条边的 slope 也是 $a+b+c$——共线！

反方向（不形成 triangle 不共线）用 $x_n$ 的 fast growth 让 dominant term 严格非零，低阶项不能 cancel。这是 **polynomial identity + valuation argument** 的经典技巧。

最后用 infinite Ramsey theorem：有限 coloring of $K_\infty$ 的 edges 必有 monochromatic triangle，对应 collinear 三点——矛盾。

这个证明同时用了 combinatorics (Ramsey)、algebra (slope calculation)、number theory (growth argument) 三个领域的技术。Agent 能 synthesize 这种 cross-domain 的证明是 impressive 的。

### 5.2 OEIS 结果

44/492 OEIS conjectures 被证明。这里有一个 anti-misformalization 的 guard：agent 必须先 prove "test lemmas" 验证 sequence 的前几项 match formal definition。

Supplementary 里给了两个完整证明：

**OEIS A051293**  
$a_n$ = number of nonempty subsets of $\{1, \ldots, n\}$ with integer average。证明 asymptotic expansion：

$$a_n = \frac{2^{n+1}}{n}\left(1 + \frac{1}{n} + \frac{3}{n^2} + \frac{13}{n^3} + \frac{75}{n^4} + \frac{541}{n^5} + o\left(\frac{1}{n^5}\right)\right)$$

技术：roots of unity filter $\frac{1}{k} \sum_{j=0}^{k-1} \omega_{k,j}^N$ 提取 divisibility 条件，然后 separate principal term $\frac{1}{k}\binom{n}{k}$ 和 remainder $R_{n,k}$。Remainder 用 polynomial $\prod_{m=1}^{n}(1 + z \omega_{k,j}^m)$ 在 unit circle 上的 max modulus bound（周期性让 product 在每个 period 上贡献 $\leq 2$）。

Principal term 用 telescoping error $E_j = f_j - f_{j-1} - \frac{2^j}{j}$，把 $S_n - f_n$ 的误差 reduce 到 $\sum |E_j| = O(2^n / n^7)$。

**OEIS A228143**  
关于 Apéry-like numbers $s_m = \sum_k \binom{m}{k}^2 \binom{m+k}{k}^2$ 的 Hankel determinant。证明 $A(x/3)^{1/8}$ 有 integer coefficients。

技术：Lucas's theorem + Kummer's theorem 分析 $s_m \mod 3$ 和 $\mod 4$。关键 congruence：

$$\left(\binom{m}{k}\binom{m+k}{k}\right)^2 \equiv (-1)^k \binom{m}{k}\binom{m+k}{k} \pmod{3}$$

然后 alternating sum $\sum_k (-1)^k \binom{m}{k}\binom{m+k}{k} = (-1)^m$ 给出 $s_m \equiv (-1)^m \pmod{3}$。

mod 4 更简单：$k \geq 1$ 时 $\binom{m}{k}\binom{m+k}{k}$ 偶数，平方 divisible by 4，所以 $s_m \equiv 1 \pmod{4}$。

用 row operations（$P_n M^{(n)}$ 减去 signed multiples of row 0）证明 $3^n | a_n$ 和 $4^n | a_n$，组合得 $16 \cdot 3^n | a_n$。

最后构造 eighth root：$A(x/3) = 1 + 16 Y(x)$，找 $C(x) = 1 + 2 X(x)$ 满足 $C^8 = 1 + 16(X + P(X))$，其中 $P$ 是已知多项式。Inductive 系数确定给出 integer coefficients。

这是非常 number-theoretic 的证明——Lucas/Kummer 是 mod prime 分析 binomial coefficient 的标准工具，agent 能正确 synthesize 这个 pipeline 让人惊讶。

### 5.3 Optimization Theory: Anchored GDA

Agent (D) 证明了 Anchored Gradient Descent-Ascent 的 $1/k$ convergence rate，改进了 [52] 的 slower bound。

关键创新：用 EVOLVE-VALUE marker 把 learning schedule 作为 parameter，让 agent **同时搜索 schedule 和 proof**，最终发现 novel parameter choice。这比 fixed algorithm 的 verification 更进了一步——agent 在做 algorithm design。

Proof 思路：depart from continuous-time ODE analysis，改用 discrete-time recurrence-based approach。Surina et al. [58] 给了 natural language 版本。

### 5.4 Algebraic Geometry: Hilbert Functions

Pure $O$-sequences 是 monomial Artinian level algebras 的 Hilbert functions [57]。Codimension 3, type 2 的 log-concavity 是 open 了 15 年的问题 [9, 68]。

Agent 的证明：把 $\Gamma$ 写成两个 down-sets $\{u \leq m_1\} \cup \{u \leq m_2\}$ 的 union，然后分解 $H(d) = h_{m_1}(d) + \mathbf{1}_{d \geq c} h_{M_\text{red}}(d - c)$。

核心是 second-difference inequality $\Delta^2 H(t) \leq 2$，加上 dichotomy：$\Delta^2 h_m(t) = 1$ 时要么 $t \leq m(i)$ for all $i$（lower alternative L），要么 $t \geq m(i) + m(j) + 2$ for all pairs（upper alternative U）。

然后 case analysis (A1, A2, B1, B2, C) 把所有情况 reduce 到 $H(d-1) \leq (\Delta H(d))^2$ 或 $2 H(d-1) \leq (\Delta H(d))^2$。最后用恒等式

$$H(d)^2 - H(d-1)H(d+1) = (\Delta H(d))^2 - H(d-1) \Delta^2 H(d+1)$$

完成 log-concavity。

这种"reduce to finitely many cases + symbolic inequality check"非常适合 agent——case 的枚举和每个 case 的代数 manipulation 都是 mechanical 的，但 case 的设计需要 insight。

### 5.5 Graph Theory: Reconstruction Conjecture

Kelly-Ulam graph reconstruction conjecture [61] 是 combinatorics 最老的 open problem 之一。Agent 证明了一个 **bipartite variant with type-distinguishability assumption**。

定理：if $G$ is 2-connected bipartite with pairwise distinct vertex types $\tau_G(x) \neq \tau_G(y)$ for $x \neq y$, and bipartite decks $\mathcal{D}_B(G) = \mathcal{D}_B(H)$, then $G \cong_B H$。

Type $\tau_K(x) = (\deg_K(x), P_K(x))$，其中 $P_K(x)$ 是邻居度数的 multiset。

证明策略：从 deck 恢复 degree multisets，建立 vertex bijection $f$ 保持 degrees，然后用 type profile $T_K(u)$（邻居的 type multiset）的 equality 推出 adjacency structure。关键的 local-type formula

$$\tau_{G \setminus \text{inc } u}(x) = \begin{cases} F_{\deg_G(u)}(\tau_G(x)) & x \in N_G(u) \\ \tau_G(x) & x \notin N_G(u) \end{cases}$$

加上 descending induction on first coordinate of type 完成 type profile equality 的证明。

---

## 6. Architecture Ablation: 关键发现

论文最 surprising 的发现：**basic agent (A) 也能解所有 9 个 Erdős 问题**，只是 cost 更高。

实验设计：agents (C) 和 (D) 用 10 个 subagent，10 个 attempts per problem；agents (A) 和 (B) 用 100 个 independent attempts。模拟 $K$ 个 subagents 的场景：把 100 个 attempts 分成 $100/K$ 个 chunks，chunk 成功 iff 任何一个 attempt 成功。Cost 计算：找到 successful attempt 的最早 timestamp $t$，sum 这个 chunk 里到 $t$ 为止的所有 attempt costs。

Figure 3 的数据（六个 Erdős 问题）显示：
- (A) 和 (B) 在 4 个问题上 margin of error 内类似
- (B) 在 #12(ii) 和 #125 上更 efficient
- (D) 在 #138 和 #125 上 2x-5x 更便宜，但在其他问题上 ~half as cost-efficient

这个发现的启示：**随着 LLM 能力提升，simple agentic loops 可能越来越够用**，complex architecture 的 advantage 可能 diminish。论文原话："We attribute the basic agent's success to both this shift and the power of compiler feedback in grounding LLM reasoning."

但 (D) 在 hardest problems 上仍有优势。Figure 10 显示 (D)@1（一个 generator 但 sample from database）比 basic 还差，说明 evolutionary database 的好处需要 **asynchronous pipeline + database coordination** 才能体现——单纯 sampling 不够。

### 6.1 Cost 和 Variance

每问题 cost 方差很大（Figure 9 的 box plot）。Agent (D) 的 total cost 包括：
- Gemini 3.1 Pro 的 prover inference
- Gemini 3.0 Flash 的 rater inference
- AlphaProof 的 TPU cost（约 $60/problem，not included in reported numbers）

Cost formula:
$$\text{total}_\text{component} = \text{input} \cdot p_\text{input} + \text{cache}_\text{read} \cdot p_\text{cache} + \text{output} \cdot p_\text{output}$$

注意 cache_read 单独计费——multi-turn session 里的 prefix caching 对 cost 控制很关键。

### 6.2 Smaller Models 失败

Gemini 3.0 Flash 和 Gemini 3.1 Flash-Lite 作为 prover 时 **无法解任何 Erdős 问题**。AlphaProof standalone tree search mode 给 64 TPU hours/problem 也不行。

这暗示 Erdős 问题需要 Gemini 3.1 Pro 级别的 reasoning + agentic loop + (optional) AlphaProof tool 的组合。任何单一组件都不够。

---

## 7. Failure Modes 分析

论文对失败 case 的分析很有价值：

**Failure mode 1: Difficulty offloading**  
Agent 经常把 problem 的核心 difficulty 推到一个 helper lemma 里的 single `sorry`，而这个 helper lemma 只是把 target statement 换个形式重述。Explicitly prompting against this behavior 失败——LLM 还是会这么做。

这是 LLM 的根本问题：缺乏对"什么是真正进展"的判断。人类数学家能 distinguish "证明了一个 non-trivial sub-claim" vs "把 statement 重写了一遍"，LLM 在 sketch 层面看不出区别。

**Failure mode 2: Hallucinated lemmas**  
Top sketches 经常依赖 marked `sorry` 的 lemmas，agent 声称这些是"established results in mathematical literature"。Manual inspection 显示这些是 hallucinations——lemma 根本不存在或者陈述错误。

这印证了 end-to-end formal verification 的必要性。Natural language proof 的 review 没法 catch 这种 subtle hallucination，但 Lean compiler 必须。

---

## 8. 与相关工作的 positioning

### 8.1 vs AlphaProof [29]

AlphaProof 用 RL 训练 olympiad-level Lean theorem-proving。本文用 AlphaProof 作为 **subroutine tool**，不是 main agent。AlphaProof 的 disproof 能力（证明 subgoal false）是关键——给主 agent 提供负面信号。

### 8.2 vs AlphaEvolve [46]

AlphaEvolve 是 LLM-guided evolution for program discovery，optimize quantitative reward。本文 agent (D) 复用 AlphaEvolve 的 components，但 fundamental difference：AlphaEvolve 找 programs 优化 quantitative reward，本文找 proofs 满足 boolean formal verification criterion。这个 difference 促使了 Elo-based rating 的设计——把 binary signal soft 化成 graduated fitness。

### 8.3 vs Draft-Sketch-Prove [34] / Hilbert [63] / Aristotle [1]

这些系统是 hierarchical：先 generate informal sketch，再 translate 成 formal steps。Aristotle 用于 formalize AI-generated natural language proofs of Erdős problems。

本文走的方向不同：**用 Lean 作为 discovery medium 本身**，而不是 formalize 已经存在的 natural language proof。这意味着 agent 在 formal space 里直接 search，不依赖 informal proof 的先验存在。

### 8.4 vs FunSearch [51]

FunSearch 是 LLM-guided evolution for mathematical constructions represented as code。本文继承了 population database + Elo matchmaking 的 idea，但目标是 proof 而非 construction。

### 8.5 vs Aletheia [21] / OpenAI's Erdős results [3, 4]

这些是 natural language proof discovery 系统。论文指出：natural language proofs 需要 expert review 来 catch subtle errors，而 formal verification 可以作为 filter 决定哪些 proofs 值得 human review。这是 formal approach 的 practical value。

---

## 9. 关键启示和未来方向

### 9.1 Compiler Feedback 是关键

Basic agent 的成功 "attributes to the power of compiler feedback in grounding LLM reasoning"。Lean compiler 给 LLM 提供 **ground truth signal**——错误信息、pending goals、type errors。这比 pure natural language reasoning 的 self-critique 可靠得多。

这呼应了代码生成领域 "execution feedback" 的重要性——RLHF with code execution、Process Reward Models 等。Formal proof 是这种 feedback 的极端形式：每一步都有 mechanical 验证。

### 9.2 LLM Capability 的快速演进

论文承认：实验开始时 simpler agentic loops 在 competition benchmarks 上表现不好，所以选了 (D)。但 LLM landscape 变化让 (A) 也变得 viable。

这是一个 meta-observation：**agent architecture 的最优选择是 LLM capability 的函数**。今天需要的复杂架构，明天可能被更强的 base model 替代。这意味着 agent 设计应该 modular，方便 deprecate 不再需要的组件。

### 9.3 Misformalization Detection

Agent 在 #125 和 #741(i) 上发现"density"的歧义——original informal statement 没说清是 natural density、lower density 还是 upper density。Agent 先用 natural density 解了，被指出不对，改成 lower/upper density 后又解了。

这是 formal proof search 的副产品：**forcing precision 会暴露 informal math 的 ambiguity**。这对数学文献的质量提升有直接价值。

### 9.4 Limitations

论文诚实地列出：
- 成功集中在 combinatorics, convex optimization, number theory——这些领域 Lean mathlib [60] 成熟，问题 decompose 成 tractable subgoals
- 大部分 Erdős 问题仍然 out of reach
- 需要 extensive new theory 的问题（比如 Riemann Hypothesis 这种）完全不行
- High search variance
- Inherits LLM biases

### 9.5 Human-Machine Partnership

论文最后强调 vision：AI 不替代数学家，而是 expand 创造力。即使 agent 没证明 claim，formal sketches 让专家 focus on unresolved subgoals 而不是 re-verify 整个 argument。这是一个重要的 workflow 改变——把 mathematician's attention 从 mechanical verification 释放出来去做 creative work。

---

## 10. 我的直觉构建

读完这篇论文，我 built 出来的 intuition：

1. **Formal proof search 的核心 bottleneck 是 fitness signal**。Binary 的 "compiles or not" 让 evolutionary methods 失效，LLM-based rating 是 bridge。但这个 rating 本身是 noisy 的——Elo + Plackett-Luce + Gibbs sampling 是 probabilistic 处理 noise 的标准工具。

2. **Agent architecture 应该 layered**。Basic loop (LLM + compiler feedback) 是 foundation，evolutionary database 是 optimization layer，focused tools (AlphaProof) 是 capability amplifier。每层的 ROI 随 LLM capability 变化。

3. **数学发现的难度 profile 不是 uniform 的**。Erdős 问题里，elementary-but-clever 的（#138, #152, #846）比 needs-new-theory 的更适合 agent。Agent 擅长 synthesize 已知技术的新组合，不擅长 invent 全新框架。

4. **Disproof signal 和 proof signal 同样重要**。AlphaProof 返回 disproof 让 agent 知道"这条路不通"，避免 wasted compute。这是 tree search 里 prune 的关键。

5. **EVOLVE-VALUE 的设计被低估**。让 agent 同时搜索 algorithm parameter 和 proof 是真正的 discovery——Anchored GDA 的 improved rate 不是 verify 已知结果，是 discover 新 algorithm。这是 AI 做 algorithm design 的 early signal。

6. **Cross-domain synthesis 是 LLM 的 unique advantage**。Erdős #846 同时用 Ramsey + algebra + number theory，#125 用 Diophantine approximation + density argument。人类专家通常专精一两个领域，LLM 可以跨领域 combine——这是 AI 的结构性优势。

7. **Cost data 的 hidden message**：$60/problem 的 AlphaProof cost + 几百美元的 LLM cost，总共 ~$500/problem。如果能让 cost 降到 $50 以下，每个 mathematician 都能 daily 使用。这是 productionize 的关键 threshold。

8. **未来的关键问题**：当 LLM 能力继续提升，basic loop 能解决多少现在需要 (D) 的问题？(D) 的 advantage 会不会完全消失？还是 hardest problems 永远需要 evolutionary coordination？这是一个 empirical question，需要持续 ablation。

---

## 11. 相关联想和 open questions

- **Proof complexity theory 的 connection**：能否用 proof complexity 来 predict 哪些问题对 agent tractable？如果一个 statement 在 weak proof system 里需要 long proof，agent 大概率也解不了。

- **Reverse mathematics 的视角**：哪些 axioms 是 agent 实际使用的？能否从 agent's proofs 反向 extract 出 "minimal axioms needed"？

- **Automated conjecture generation**：Grafiti [19] 1996 年提出 #125 相关的 graph conjecture。能否 close the loop：AI conjecture → AI proof → 新 conjecture？这是真正的 autonomous mathematics。

- **Lean mathlib 的 coverage bias**：论文承认成功集中在 mathlib 成熟的领域。能否用 agent 来 expand mathlib 本身？比如 formalize 一本 algebraic geometry textbook。

- **Quantum optics application [38]**：monochromatic quantum graphs 对应 GHZ states。这种 physics-math connection 是否预示着 AI 在 theoretical physics 也有 similar pipeline？

- **Cost reduction via distillation**：能否用 (D) 的成功 traces 蒸馏出更小的 prover model？让 Gemini Flash 级别模型也能解现在只有 Pro 能解的问题。

- **Multi-agent debate**：能否让多个 agent with different architectural biases (one basic, one evolutionary, one tool-heavy) 互相 debate 来 improve collective performance？

- **Formalization-first research paradigm**：如果未来 mathematician 直接在 Lean 里做 research（而不是先写 informal 再 formalize），整个数学文献的形态会改变。这是 Viazovska sphere packing [28] 的 formalization 所预示的方向。

---

## 12. 总结

这篇 paper 是 AI-for-math 的一个 milestone：第一次大规模 demonstrate formal proof search 能解 research-level open problems。9 个 Erdős problems（含两个 56 年 open）、44 个 OEIS conjectures、algebraic geometry 15 年 open problem、convex optimization improved bound——这些都是 genuine mathematical progress。

更重要的是，它揭示了一个 **正在发生的 shift**：从 specialized trained systems（AlphaProof 风格）toward simple agentic loops（basic agent + strong LLM + compiler feedback）。这个 shift 的速度取决于 LLM capability 的增长。Full-featured agent 在 hardest problems 上仍 retain advantage，但这个 advantage 可能 diminish。

Engineering 上，Elo + Plackett-Luce + P-UCB + Thompson sampling 的组合是处理 binary fitness signal 的 elegant solution，值得在其他 "discrete success criterion" 的 evolutionary search 场景复用。

Failure modes（difficulty offloading, hallucinated lemmas）揭示了当前 LLM 的根本 limitation：缺乏对 proof structure 的 deep understanding。这种 limitation 只有 end-to-end formal verification 能 catch——这反过来 justify 了 formal approach 的必要性。

未来的关键研究方向：降低 cost（让每个 mathematician 都能用）、expand domain coverage（formalize 更多 mathlib）、close the conjecture-proof loop（Grafiti + AlphaProof Nexus）、以及理解 LLM capability 增长如何改变 optimal agent architecture。

这是 Andrej 你会关心的那种 paper——它把 ML engineering（agent architecture, evolutionary search, cost analysis）和 deep mathematics（Erdős problems, Hilbert functions, Diophantine approximation）真正结合起来，而且 hint at 一个人机协作的数学研究新范式。

References:
- AlphaProof Nexus 结果仓库: https://www.github.com/google-deepmind/alphaproof-nexus-results  
- AlphaProof Nature paper: https://www.nature.com/articles/s41586-025-08587-0 (Hubert et al. 2025)  
- AlphaEvolve arXiv: https://arxiv.org/abs/2506.13131  
- Tao's wiki on AI contributions to Erdős problems: https://github.com/teorth/erdosproblems/wiki/AI-contributions-to-Erd%C5%91s-problems  
- ErdosProblems.com: https://www.erdosproblems.com  
- Formal Conjectures: https://arxiv.org/abs/2502.20515  
- SafeVerify: https://github.com/GasStationManager/SafeVerify  
- Lean 4: https://lean-lang.org/  
- Pantograph: https://arxiv.org/abs/2503.10670  
- Plackett-Luce model: https://en.wikipedia.org/wiki/Plackett%E2%80%93Luce_model  
- Bradley-Terry models: Caron & Doucet 2012, https://www.tandfonline.com/doi/abs/10.1080/10618600.2012.675546  
- AlphaZero / P-UCB: Silver et al. 2018, https://arxiv.org/abs/1712.01815  
- Draft-Sketch-Prove: https://arxiv.org/abs/2210.12283  
- FunSearch: https://www.nature.com/articles/s41586-023-06924-6  
- Aletheia: https://arxiv.org/abs/2602.21201  
- Ralph loop concept: https://ghuntley.com/ralph  
- Surina et al. Anchored GDA: https://arxiv.org/abs/2604.03782
