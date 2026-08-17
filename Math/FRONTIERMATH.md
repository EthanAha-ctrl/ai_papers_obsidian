---
source_pdf: FRONTIERMATH.pdf
paper_sha256: 9e6210b5b8afe17f70a1a5eaed89e4ed2f4f38e255b1766db34763208a6dcd25
processed_at: '2026-08-04T11:17:36-07:00'
target_folder: Math
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话说说 FrontierMath

## 这 paper 到底在搞啥？

很简单：**现在的 AI 数学太菜了**，但现有的 benchmark 又测不出来它到底有多菜。

## 为啥说现有 benchmark 不行？

你看 GSM8K、MATH 这些，本来就是 high school 到 undergraduate 的题。现在 o1、Claude 3.5 都做到 94%、96% 了，基本上就是刷分刷到 saturation。就像你给一个大学生做小学算术，满分也说明不了啥。

而且更烦的是 **data contamination**——这些题在网上都传烂了，model 训练时候大概率见过，所以分数虚高。

## FrontierMath 怎么解决的？

找 60 多个 mathematician（包括 Fields Medalists Tao、Gowers、Borcherds），每人出几道原创的、没人见过的 research-level 题。要求：

- **答案必须能自动 verify**：要么是 integer，要么是 SymPy object，不用人肉改卷
- **不能靠猜**：答案都是那种特别大的数，比如 1876572071974094803391179，你猜不出来的
- **计算量可控**：solution script 要在 1 分钟内跑完，说明靠的是 insight 不是 brute force

## 题有多难？

举个直觉：Tao 自己都说这种题要 "domain expert + AI + algebra packages" 配合才能搞。一道典型题需要 expert mathematician 花几个小时甚至几天。

sample problems 涵盖的范围：
- Artin's primitive root conjecture（number theory，conditional on GRH）
- Chebyshev polynomial 的 monodromy（algebraic geometry）
- p-adic 连续延拓（p-adic analysis）
- $S_5$ representation counting（group theory）
- Klein quartic 的 point counting（arithmetic geometry）

## AI 表现咋样？

**惨不忍睹**。o1-preview pass@1 不到 2%，pass@8 也才 6% 左右。其他 model 更低。

而且就算答对的题，很多时候也是靠 brute-force simulation 猜的，不是真的理解了。比如有一道题 model 是靠跑几次 simulation 就猜对了，但这种 strategy 对绝大多数题不 work。

## 那这 benchmark 有啥用？

现在主要是**树立一个 ceiling**：告诉所有人，research-level math 还远着呢。

Tao 的判断是 "resist AIs for several years at least"，但他觉得 human + AI collaboration 可能在 3 年内能 tackle 这种题。

## 核心问题在哪？

Tao 说得很直接：**training data 几乎不存在**。对于很多 research-level 问题，网上相关的 paper "就那么十来篇"。LLM 靠 pattern matching，没见过的东西它就 match 不出来。

还有 multi-step reasoning 的 error accumulation 问题——每一步 70% 正确，走 10 步就只剩 2.8% 了，跟实测结果吻合。

## 有啥局限？

1. 只能出 numerical answer 的题，proof-based 的做不了自动评分
2. 题虽然难，但 hours 级别的，真实 research 是 months-years 级别
3. 现在 AI 都 < 2%，区分度不够，等 AI 进步了才有用

## 一句话总结

**这 paper 就是搞了个特别难的数学考试，让所有 AI 都考不及格，然后说：看吧，AI 离真正搞数学研究还差得远呢。**

---

# FrontierMath: 高级数学推理 Benchmark 深度解析

## 1. Paper 概览与核心动机

FrontierMath 是 Epoch AI 联合 60+ mathematicians（含 Fields Medalists 和 14 IMO gold medalists）构建的 benchmark，旨在解决当前数学 evaluation 的两个核心痛点：

**Saturation 问题**：现有 benchmark 如 GSM8K (https://arxiv.org/abs/2110.14168)、MATH (https://arxiv.org/abs/2103.03874)、AIME、Omni-MATH (https://arxiv.org/abs/2410.07985) 已经接近 saturation。o1 在 MATH 上达到 94.8%，Claude 3.5 Sonnet 在 GSM8K 上达到 96.4%，但这些 benchmark 顶多测试到 undergraduate early level。

**Data contamination 问题**：Deng et al. 2023 (https://arxiv.org/abs/2311.09783) 和 Xu et al. 2024 (https://arxiv.org/abs/2406.04244) 的工作显示，LLM training data 中可能包含 benchmark 问题，导致 performance 虚高。FrontierMath 通过 exclusively unpublished problems 规避此风险。

## 2. Benchmark 设计四原则

Data collection pipeline 严格要求四个属性：

### 2.1 Originality
允许 build on existing ideas 但必须通过 clever adaptations 或 innovative combinations 显著 transform 原始概念，使 pattern matching 失效。

### 2.2 Automated Verifiability
要求 definitive、computable answers。主要支持两种 format：
- **Integer solutions**：exact match
- **SymPy objects**：symbolic expressions, matrices, sets

Verification 逻辑（参考 Figure 3 的 Pell's equation 例子 $x^2 - 7y^2 = 1$）：
```python
# Custom verification script
def verify(answer):
    x, y = answer
    return x**2 - 7*y**2 == 1 and x > 0 and y > 0
```

### 2.3 Guessproofness
Rule of thumb：没有做完解题工作的情况下，猜中答案的概率不应超过 1%。这也是为什么 sample problems 的答案都是 large、nonobvious numbers（如 1876572071974094803391179）。

### 2.4 Computational Tractability
Solution scripts 必须在 standard hardware 上累计 runtime < 1 minute。这排除了 brute-force approaches，强制要求 mathematical insight。

## 3. 难度评估系统

三维 difficulty rating：

| Dimension | Range | Meaning |
|-----------|-------|---------|
| Background | 1-5 | 1=HS, 2=early UG, 3=late UG, 4=grad, 5=research |
| Creativity | hours (无上限) | Expert 找到 key ideas 所需时间 |
| Execution | hours (无上限) | 找到 key ideas 后计算 final answer 所需时间 |

例如 Sample Problem 1 (Artin conjecture) 的 ratings：Background=Research, Creativity=4h, Execution=15h。

## 4. Dataset 组成分析

MSC2020 分布（Table 1）显示 number theory (17.8%) 和 combinatorics (15.8%) 占主导，reflecting 这两个领域对 numerical answers 的 natural amenability。Algebraic geometry (4.8%)、group theory (8.9%)、probability (5.1%) 也有显著 representation。

Network graph (Figure 4) 显示 number theory + combinatorics 共现于 13% problems，combinatorics + group theory 9%，number theory + group theory 8%。

## 5. Evaluation Framework 详解

采用 experiment-enabled evaluation：model 可以在 Python sandbox 中 iterative 地 write code、receive output、refine approach。

**Token budget**：10,000 tokens per attempt。如果达到 limit，发送 final prompt 强制提交。

**Final answer format**：
```python
# This is the final answer
import pickle
answer = ...
with open('final_answer.p', 'wb') as f:
    pickle.dump(answer, f)
```

**Key observation**：不同 model 使用 experiment 能力差异巨大。o1-preview 平均 1.29 responses/problem（倾向于直接给答案），而 Grok 2 Beta 平均 3.81 responses。Gemini 1.5 Pro 平均只用 ~6000 tokens，其他 models 用 12,000-17,000 tokens。

## 6. 核心实验结果

### 6.1 Pass@1 Accuracy
所有 six models (o1-preview, o1-mini, GPT-4o, Claude 3.5 Sonnet, Grok 2 Beta, Gemini 1.5 Pro) 的 pass@1 accuracy 都 < 2%。

### 6.2 Pass@8 Accuracy (Figure 9)
o1-preview 达到约 6%，Grok 2 Beta 约 2%，其他更低。

### 6.3 Repeated trials on solvable problems (Table 2)
四道至少被某 model 解决一次的问题，5 次 runs 的 success rate：

- Problem 1 (Probability + Approximations)：Grok 60%, o1-preview 40%
- Problem 2 (Algebraic topology + Manifolds)：o1-preview 100%, Claude 80%
- Problem 3 (Group theory + Field theory)：o1-preview 80%, Gemini 60%
- Problem 4 (Algebraic geometry + Combinatorics)：仅 o1-mini 20%

High variability 暗示 model 在这些问题上的 reasoning 不稳定，可能依赖 luck。

## 7. Sample Problems 技术深度解析

### 7.1 Problem 1: Artin's Primitive Root Conjecture

**答案**：367707（即 $\lfloor 10^6 d_\infty \rfloor$，$d_\infty \approx 0.367707$）

#### 问题 setup

对于 prime $p$ 和 $a \not\equiv 0 \pmod{p}$：
- $\text{ord}_p(a)$ = smallest $o > 0$ s.t. $a^o \equiv 1 \pmod{p}$（multiplicative order）
- $\text{ord}_{p,x}(a) = \prod_{q \leq x, q \text{ prime}} q^{v_q(\text{ord}_p(a))} \prod_{q > x, q \text{ prime}} q^{v_q(p-1)}$

这里 $v_q(\cdot)$ 是 $q$-adic valuation。$x$ 参数的作用是：对 small primes $q \leq x$ 用 actual order 信息，对 large primes $q > x$ 用 "default" $p-1$ 的 factorization。

$S_x = \{p : \text{ord}_{p,x}(2) > \text{ord}_{p,x}(3)\}$，$d_x = |S_x| / |\{p \leq x : p \text{ prime}\}|$，求 $d_\infty = \lim_{x\to\infty} d_x$。

#### 核心 idea：Hooley's approach to Artin's conjecture

Artin conjecture (https://mathworld.wolfram.com/ArtinsConstant.html) 说：对于 $a$ 不是 perfect square 也不是 $-1$，存在无穷多 primes $p$ 使得 $a$ 是 primitive root mod $p$（即 $\text{ord}_p(a) = p-1$）。Density 约 37%。

Hooley 1967 在 GRH (Generalized Riemann Hypothesis for number field zeta functions) 下证明了此 conjecture。方法核心：

**Step 1**：$a$ 是 primitive root mod $p$ iff $a$ 不是 $n$-th power mod $p$ for any $n \mid p-1$, $n > 1$。

**Step 2**：用 inclusion-exclusion：
$$\#\{p \leq x : \text{ord}_p(a) = p-1\} = \sum_{n \geq 1} \mu(n) \cdot \#\{p \leq x : a \text{ is } n\text{-th power mod } p, p \equiv 1 \pmod{n}\}$$

**Step 3**：Chebotarev density theorem 给出 asymptotic：
$$|\{p \leq x : p \equiv 1 \pmod{n}, a \text{ is } n\text{-th power mod } p\}| \sim \frac{x}{n \varphi(n) \log x}$$
（在 $a=2$ 情形，对于 square-free $n$）

**Step 4**：Error term control 需要 GRH（类比 PNT 的 error term）。

#### 本题的关键创新

引入 $\text{ord}_{p,x}(a)$ 替代 $\text{ord}_p(a)$ 的目的：**消除 error term 依赖**。Modified version 下，inclusion-exclusion 只需 finite many terms，因此只需 main term asymptotic 而无需 sharp error bounds。这使得问题可以在 unconditional 情形下求解。

#### 计算 $d(a,b)$：density of primes with $\text{ord}_{p,x}(2) = (p-1)/a$, $\text{ord}_{p,x}(3) = (p-1)/b$

考虑 number field $K = \mathbb{Q}(\zeta_{\text{lcm}(a,b)}, 2^{1/a}, 3^{1/b})$。Prime $p$ 满足条件 iff $p$ splits in $K$ 但不 split in $K_{1,q} = \mathbb{Q}(\zeta_{aq}, 2^{1/(aq)})$ 或 $K_{2,q} = \mathbb{Q}(\zeta_{bq}, 3^{1/(bq)})$ for any prime $q \leq x$。

由 Chebotarev，density = $1/[K:\mathbb{Q}]$。Inclusion-exclusion 给出：

$$d(a,b,x) = \sum_{\substack{n_1, n_2 : p|n_i \Rightarrow p \leq x}} \frac{\mu(n_1)\mu(n_2)}{[\mathbb{Q}(\zeta_{\text{lcm}(an_1, bn_2)}, 2^{1/(an_1)}, 3^{1/(bn_2)}) : \mathbb{Q}]}$$

**关键 Lemma 1** (Järviniemi 2020, https://arxiv.org/abs/1912.02554)：Fields $K_{6^N}, K_{q_1^N}, \ldots, K_{q_k^N}$（$q_i > 3$ distinct primes）linearly disjoint，且 degrees 达到 maximum $\varphi(q_i^N) q_i^{2N}$。

**关键 Lemma 2**：在 $q=2, 3$ 处 lose factor of 4：
$$[K_{6^N} : \mathbb{Q}] = \frac{\varphi(6^N) 6^{2N}}{4}$$
原因：$\zeta_3 \in \mathbb{Q}(\zeta_4, \sqrt{3})$（因为 $\zeta_3 = (1+\sqrt{-3})/2$）和 $\sqrt{2} \in \mathbb{Q}(\zeta_8)$（因为 $(1+i)^2 = 2i$，所以 $1+i = \zeta_8 \sqrt{2}$）。

#### Quasi-multiplicative structure

$$d(a,b) = f(v_2(a), v_3(a), v_2(b), v_3(b)) \prod_{5 \leq q \leq x} g_q(v_q(a), v_q(b))$$

其中 $f$ 和 $g_q$ 都可以在 $O(1)$ 时间内 evaluate。

#### Tail sum bounds

**Lemma 3**：$d(a,b,x) \leq \frac{4}{\varphi(\text{lcm}(a,b)) ab}$

**Tail bound**（用 $\varphi(n) \geq c_1 (n/c_2)^\delta$，其中 $c_1 = 92160$, $c_2 = 510510$, $\delta = 46/47$）：

$$\sum_{T < a < b} d(a,b,x) < 114 T^{-2 \cdot 46/47}$$

取 $T = 10^5$，error $< 2 \cdot 10^{-8}$，给出 7 位正确小数。

#### 与 GRH 的关系

作者强调：如果用 $\text{ord}_p$ 而非 $\text{ord}_{p,x}$，需要 GRH 来 control error terms。Modified version 的 genius 之处在于：finite inclusion-exclusion 使 main terms 足够。但 $d_\infty$ 的 existence 仍依赖 technical claim（Artin's conjecture 仍 open）。

GitHub code: https://github.com/epoch-research/artin_code

### 7.2 Problem 2: Degree 19 Polynomial with Reducible Fiber Product

**答案**：$p(19) = 1876572071974094803391179$，其中 $p(x) = 2 T_{19}(x/2)$

#### 问题

构造 degree 19 polynomial $p(x) \in \mathbb{C}[x]$ 使得 $X := \{p(x) = p(y)\} \subset \mathbb{P}^1 \times \mathbb{P}^1$ 有 $\geq 3$ 个（但非全 linear）irreducible components over $\mathbb{C}$。要求 $p$ odd, monic, real coefficients, linear coefficient $-19$。

#### Monodromy 理论 setup

$p: \mathbb{P}^1_x \to \mathbb{P}^1_z$ 是 degree 19 covering。At $z = \infty$，monodromy $\sigma_\infty$ 是 19-cycle $(1,2,\ldots,19)$。

$X = \{p(x) = p(y)\}$ 是 fiber product $p \times_z p$，induces $\hat{p}: X \to \mathbb{P}^1_z$ degree $19^2 = 361$。

**关键**：Irreducible components of $X$ 对应 monodromy group $G_p \subset S_{19}$ 在 $S_{19}^2$ 上的 orbits（diagonal action $\bar{p}^2$）。

#### 应用 Burnside 定理

**Burnside 定理** (Burnside 1911)：For $q$ prime，$G \subset S_q$ either 2-transitive 或 $|G| \mid q(q-1)$。

本题要求 $G_p$ 在 $S_{19}^2$ 上有 3-18 orbits，所以 $G_p$ not 2-transitive，故 $|G_p| \mid 19 \cdot 18$。

由 Sylow 定理，order 19 subgroup $\langle \sigma_\infty \rangle$ normal in $G_p$。

#### 推导 monodromy structure

取 $\sigma_i \in G_p$ 假设 fix $\geq 2$ points (wlog 1 and $n$)。由 normality：
$$\sigma_i \sigma_\infty \sigma_i^{-1} = \sigma_\infty^s$$
for some $s \in \mathbb{Z}$。代入 1：$n = 1 + (n-1)s \pmod{19}$，所以 $s = 1$。代入 1 于 $\sigma_i \sigma_\infty \sigma_i^{-1} = \sigma_\infty$：$\sigma_i(2) = 2$。Inductively，$\sigma_i$ fixes all points，矛盾。

**结论**：每个 non-trivial $\sigma_i$ fix at most 1 point。

#### Riemann-Hurwitz

$$19 - 1 = \sum_r \text{Index}(\sigma_r)$$

Index$(\sigma_r)$ = ramification index $- 1$。每个 non-trivial $\sigma_i$ fix $\leq 1$ point，所以 Index$(\sigma_i) \geq (19-1)/2 = 9$。

因此 $r \leq 2$。$r=1$ 给 19 irreducible components（太多 linear）。所以 $r=2$，Index$(\sigma_i) = 9$，即 $\sigma_i$ 是 9 disjoint transpositions。

#### 唯一性：Chebyshev polynomial

拓扑 relation $\sigma_1 \sigma_2 = \sigma_\infty$（来自 $\mathbb{P}^1 - B$ 的 fundamental group）：

假设 $\sigma_1 = (1,2)(3,19)(4,18)\ldots$，则 $\sigma_2 = (2,19)(3,18)\ldots$

由 Riemann existence theorem，给定 monodromy 的 polynomial 本质唯一（up to affine scaling），就是 degree 19 Chebyshev polynomial $T_{19}(x)$。

验证：$T_{19}$ 在 $S_{19}^2$ 上恰好有 10 个 orbits（1 个 diagonal + 9 个 bi-degree (2,2) components），满足 "$\geq 3$ but not all linear" 条件。

#### 确定系数

$p(x) = A T_{19}(ax + b) + B$。约束：
- Odd：$b = 0$, $B = 0$
- Monic：$A \cdot a^{19} \cdot 2^{18} = 1$（因为 $T_{19}$ leading coeff 是 $2^{18}$）
- Linear coeff $-19$：$A \cdot a \cdot 19 = -19$（因为 $T_{19}'(0) = 19$... 实际 $T_n'(x)$ at 0）

Wait, 实际上 $T_{19}(x)$ 的 linear coefficient：$T_n(x) = \cos(n \arccos x)$。$T_{19}'(0) = n \cdot U_{n-1}(0) \cdot$... 让我重新推导。

$T_n(x) = \sum_{k=0}^{\lfloor n/2 \rfloor} c_{n,k} x^{n-2k}$。对于 $n = 19$，linear term 是 $c_{19,9} x$。

实际上 paper 给出：linear coeff of $p$ is $-19 \cdot A \cdot a = -19$，所以 $Aa = 1$。

与 monic 条件 $A a^{19} 2^{18} = 1$ 联立：$a^{18} = 1/2^{18}$，所以 $a = \pm 1/2$。Real coefficients 选 $a = 1/2$（或 $-1/2$），$A = 2$。

最终：$p(x) = 2 T_{19}(x/2)$

#### Intuition

这题的精髓在于：abstract algebraic geometry 问题（fiber product reducibility）通过 monodromy 完全 reduce 到 finite group theory。Burnside 定理 + Riemann-Hurwitz 把 search space 压缩到 unique solution。Chebyshev polynomial 是 "universal" 例子因为它有 maximal monodromy symmetry。

Reference: 
- Burnside, W. (1911). *Theory of Groups of Finite Order*. Cambridge Univ. Press.
- Riemann existence theorem: https://en.wikipedia.org/wiki/Riemann_existence_theorem

### 7.3 Problem 3: p-adic Continuous Extensions of Recurrences

**答案**：9811

#### 问题

Linear recurrence：
$$a_n = 198130309625 a_{n-1} + 354973292077 a_{n-2} - 427761277677 a_{n-3} + 370639957 a_{n-4}$$

initial $a_i = i$ for $0 \leq i \leq 3$。求最小 prime $p \equiv 4 \pmod{7}$ 使得 $n \mapsto a_n$ 可 extend 到 $\mathbb{Z}_p$ 上 continuous function。

#### Skolem-Mahler-Lech theorem 与 explicit formula

Characteristic polynomial：
$$P(x) = x^4 - 198130309625 x^3 - 354973292077 x^2 + 427761277677 x - 370639957$$

Roots $\alpha_1, \ldots, \alpha_4$（在某 number field $L$ 中）。Explicit formula：
$$a_n = c_1 \alpha_1^n + c_2 \alpha_2^n + c_3 \alpha_3^n + c_4 \alpha_4^n$$

#### p-adic extension 的条件

要 extend $n \mapsto a_n$ 到 $\mathbb{Z}_p$，需 extend $\alpha^n$ 到 continuous function on $\mathbb{Z}_p$。这通过 $p$-adic exponential/log 实现：

$$\alpha^n = \exp(n \log \alpha)$$

**关键**：$p$-adic $\log$ 仅在 $v_p(\alpha - 1) > 0$ 时收敛（类比 real log 在 $x > 0$ 收敛）。

所以条件：所有 $\alpha_i \equiv 1 \pmod{\pi}$（$\pi$ 是 $L$ 的 uniformiser）。

#### 推导 candidate primes

由 Vieta's formulas：
- $\alpha_1 \alpha_2 \alpha_3 \alpha_4 = 370639957$（注意符号：constant term 是 $-370639957$，所以 product 是 $370639957$，因为 degree 4）

Wait, 实际上 $P(x) = x^4 - \ldots - 370639957$，constant term 是 $-370639957$，所以 product of roots $= -(-370639957) = 370639957$... 让我重新检查。

$P(x) = x^4 + a_3 x^3 + a_2 x^2 + a_1 x + a_0$。Product of roots $= (-1)^4 a_0 = a_0 = -370639957$。所以 $\alpha_1 \cdots \alpha_4 = -370639957$。

Hmm, 但 paper 写 $\alpha_1 \alpha_2 \alpha_3 \alpha_4 \equiv 1 \pmod{p}$ 推出 $p \mid 370639957 + 1 = 370639958$。这表明 product 是 $+370639957$，意味着 constant term 是 $+370639957$... 让我再看 paper 原文。

Paper 写："$-370639957 = \alpha_1 \alpha_2 \alpha_3 \alpha_4 \equiv 1 \bmod p$"，所以 $p \mid 370639957 + 1 = 370639958$。

OK 所以 product of roots 是 $-370639957$（与 $(-1)^4 \cdot \text{const} = \text{const}$ 一致，const = $-370639957$）。等等，对于 $P(x) = x^4 + c_3 x^3 + c_2 x^2 + c_1 x + c_0$，product of roots $= (-1)^4 c_0 = c_0$。这里 $c_0 = -370639957$，所以 product $= -370639957$。✓

如果所有 $\alpha_i \equiv 1$，则 product $\equiv 1$，所以 $-370639957 \equiv 1 \pmod{p}$，即 $p \mid 370639958$。

$370639958 = 2 \cdot 13 \cdot 1453 \cdot 9811$

其中 $1453 \equiv 4 \pmod{7}$（因为 $1453 = 207 \cdot 7 + 4$）和 $9811 \equiv 4 \pmod{7}$（$9811 = 1401 \cdot 7 + 4$）。

由 Vieta：sum of roots $= 198130309625$。如果所有 $\equiv 1$，则 $4 \equiv 198130309625 \pmod{p}$，所以 $p \mid 198130309621$。

$198130309621 = 37 \cdot 673 \cdot 811 \cdot 9811$

两个条件的 intersection：$p = 9811$（1453 不整除 198130309621）。

#### 验证 $p = 9811$

Mod 9811：
$$P(x) \equiv x^4 - 4x^3 + 6x^2 - 4x + 1 = (x-1)^4$$

所以四个 roots 都 $\equiv 1 \pmod{9811}$。✓

#### 为何其他 primes 不行

如果某 $\alpha_i \not\equiv 1$，则 $\alpha_i = \zeta_{\alpha_i} \cdot (\alpha_i / \zeta_{\alpha_i})$，其中 $\zeta_{\alpha_i}$ 是 $(p-1)$-th root of unity（Teichmüller representative）。则：

$$a_n = \sum_i c_i \zeta_{\alpha_i}^{n \bmod (p-1)} \exp(n \log(\alpha_i / \zeta_{\alpha_i}))$$

这给出 $p-1$ 个不同 analytic functions $a^{(k)}(x)$，每个在 residue class $n \equiv k \pmod{p-1}$ 上 agree with $a_n$。但这些 functions 不 compatible，无法 paste 成单一 continuous function on $\mathbb{Z}_p$。

#### Intuition

这题展示了 $p$-adic analysis 的精致结构：$\mathbb{Z}_p$ 的 totally disconnected topology 使得 "continuous extension" 成为强约束。Real case 下任何 sequence 都可以 extend（用 interpolation），但 $p$-adic 下需要 algebraic compatibility。Skolem-Mahler-Lech theorem 是相关 fundamental result about linear recurrences 的 zero sets。

Reference:
- Skolem-Mahler-Lech theorem: https://en.wikipedia.org/wiki/Skolem%E2%80%93Mahler%E2%80%93Lech_theorem
- p-adic exponential/log: https://en.wikipedia.org/wiki/P-adic_exponential_function

### 7.4 Problem 4: Counting S_5 Representations via Coxeter Relations

**答案**：625243878951

#### 问题

4-tuple of $1000 \times 1000$ matrices $A_1, \ldots, A_4$ 满足：
1. $A_i^2 = I$
2. $A_i A_j = A_j A_i$ if $\{3j-i, 3i-j\} \cap 5\mathbb{Z}_{>0} = \emptyset$
3. $A_i A_j A_i^{-1} A_j^{-1} = A_j A_i$ (braid relation) if $\{3j-i, 3i-j\} \cap 5\mathbb{Z}_{>0} \neq \emptyset$

Count orbits under simultaneous conjugation by $GL(1000)$。

#### 识别 Coxeter relations of $S_5$

检查哪些 $(i,j)$ pairs 满足 braid 条件：
- $(1,2)$: $3 \cdot 2 - 1 = 5 \in 5\mathbb{Z}_{>0}$ ✓
- $(2,4)$: $3 \cdot 4 - 2 = 10 \in 5\mathbb{Z}_{>0}$ ✓
- $(4,3)$: $3 \cdot 3 - 4 = 5 \in 5\mathbb{Z}_{>0}$ ✓
- 其他 pairs：commutation

这正是 $S_5$ 的 Coxeter presentation！$S_5$ 由 $s_1, s_2, s_3, s_4$（adjacent transpositions）生成，满足：
- $s_i^2 = 1$
- $s_i s_j = s_j s_i$ if $|i-j| \geq 2$
- $s_i s_j s_i = s_j s_i s_j$ if $|i-j| = 1$（braid relation）

Mapping $(1,2,3,4) \to (s_1, s_2, s_4, s_3)$ 即可匹配。

#### Reduction to representation counting

数据 $(A_1, A_2, A_3, A_4)$ = representation of $S_5$ on 1000-dim vector space $V$。Simultaneous conjugation = change of basis，所以 orbits = isomorphism classes of 1000-dim $S_5$-representations。

#### $S_5$ 的 irreducible representations

$S_5$ 有 7 irreps（对应 7 partitions of 5: $(5), (4,1), (3,2), (3,1,1), (2,2,1), (2,1,1,1), (1,1,1,1,1)$），维度：

| Partition | Dimension |
|-----------|-----------|
| $(5)$ | 1 (trivial) |
| $(1^5)$ | 1 (sign) |
| $(4,1)$ | 4 (standard) |
| $(2,1^3)$ | 4 (sign ⊗ standard) |
| $(3,2)$ | 5 |
| $(2,2,1)$ | 5 |
| $(3,1,1)$ | 6 |

#### Generating function

1000-dim representation = direct sum of irreps。Number of ways = coefficient of $t^{1000}$ in：

$$\frac{1}{(1-t)^2 (1-t^4)^2 (1-t^5)^2 (1-t^6)}$$

每个 irrep dimension $d$ 贡献 factor $\frac{1}{1-t^d} = 1 + t^d + t^{2d} + \ldots$（表示可以取 0, 1, 2, ... copies）。

计算 $[t^{1000}]$ 得 625243878951。

#### Intuition

这题的 elegant 之处在于：看似复杂的 matrix relations 实际编码了 well-known group 的 Coxeter presentation。一旦识别出来，hard counting problem 变成 standard representation theory + generating function。这也是为什么 Burnside's lemma 和 character theory 在 group action counting 中如此强大。

Reference:
- Coxeter groups: https://en.wikipedia.org/wiki/Coxeter_group
- $S_n$ representations: https://en.wikipedia.org/wiki/Representation_theory_of_the_symmetric_group

### 7.5 Problem 5: Hasse-Weil Bound for Klein-like Curve

**答案**：3814708984376

#### 问题

计算 $C: x^3 y + y^3 z + z^3 x = 0$ over $\mathbb{F}_{5^{18}}$ 的 nonzero projective points（up to scaling）数。

#### Setup

这是 Klein quartic 的 variant（Klein quartic 是 $x^3 y + y^3 z + z^3 x = 0$ in characteristic $\neq 5$）。在 $p = 5$ 下 smooth，genus $g = 3$。

#### Weil conjectures (proved by Deligne 1974)

For smooth projective curve $C$ of genus $g$ over $\mathbb{F}_p$：
$$|C(\mathbb{F}_{p^n})| = p^n - \sum_{i=1}^{2g} \alpha_i^n + 1$$

其中 $\alpha_i$ 是 Frobenius eigenvalues，$|\alpha_i| = \sqrt{p}$，come in conjugate pairs $\alpha_i \bar{\alpha}_i = p$。

#### Computational approach

Brute-force count for small fields：
- $|C(\mathbb{F}_5)| = 6$
- $|C(\mathbb{F}_{25})| = 26$
- $|C(\mathbb{F}_{125})| = 126$

设 $\alpha_i = \sqrt{5} \beta_i$ where $|\beta_i| = 1$。Then：
$$|C(\mathbb{F}_{p^n})| = p^n - p^{n/2} \sum_i \beta_i^n + 1$$

对于 genus 3，有 6 个 $\alpha_i$（3 conjugate pairs）。

#### Identifying $\beta_i$

观察到 $\beta_j = e^{(2j+1) i\pi/6}$（12th roots of unity with odd exponent）satisfy equations。具体：

For $n=1$: $5 - \sqrt{5}(\beta_1 + \ldots + \beta_6) + 1 = 6$，所以 $\sum \beta_i = 0$。✓（12th roots of unity sum to 0）

For $n=2$: $25 - 5(\sum \beta_i^2) + 1 = 26$，所以 $\sum \beta_i^2 = 0$。

For $n=3$: $125 - 5\sqrt{5}(\sum \beta_i^3) + 1 = 126$，所以 $\sum \beta_i^3 = 0$。

$\beta_j = e^{(2j+1)i\pi/6}$ for $j = 0, 1, 2, 3, 4, 5$ 给 $\{e^{i\pi/6}, e^{i\pi/2}, e^{5i\pi/6}, e^{7i\pi/6}, e^{3i\pi/2}, e^{11i\pi/6}\}$。

#### Uniqueness

设 $\alpha_i$ roots of $z^2 - \sqrt{5}\beta_i z + 5$ for $i = 1, 2, 3$（三个 quadratic factors）。则 $\alpha_i^2$ roots of $z^2 - 5 T_2(\beta_i) z + 25$，$\alpha_i^3$ roots of $z^2 - \sqrt{125} T_3(\beta_i) z + 125$，其中 $T_2, T_3$ 是 Chebyshev polynomials。

Power sums $\beta_1^k + \beta_2^k + \beta_3^k$ for $k = 1, 2, 3$ uniquely determine $\{\beta_1, \beta_2, \beta_3\}$（by Newton's identities）。

#### Final calculation

For $n = 18$：
$$|C(\mathbb{F}_{5^{18}})| = 5^{18} - \sum_{i=1}^6 \alpha_i^{18} + 1$$

$\alpha_i^{18} = 5^9 \beta_i^{18}$。$\beta_j = e^{(2j+1)i\pi/6}$，所以 $\beta_j^{18} = e^{(2j+1) \cdot 3 i\pi} = e^{(2j+1) \cdot 3 i\pi}$。

$e^{3i\pi \cdot \text{odd}} = e^{i\pi \cdot \text{odd} \cdot 3} = (-1)^{\text{odd} \cdot 3} = -1$（因为 odd × 3 是 odd）。

所以 $\sum_{i=1}^6 \alpha_i^{18} = 6 \cdot 5^9 \cdot (-1) = -6 \cdot 5^9$。

$$|C(\mathbb{F}_{5^{18}})| = 5^{18} - (-6 \cdot 5^9) + 1 = 5^{18} + 6 \cdot 5^9 + 1$$

$5^{18} = 3814697265625$，$5^9 = 1953125$，$6 \cdot 5^9 = 11718750$。

$|C(\mathbb{F}_{5^{18}})| = 3814697265625 + 11718750 + 1 = 3814708984376$。

#### Projective scaling

由于题目问 "up to scaling"，且这是 projective curve，每个 projective point 对应 $p^n - 1$ 个 affine representatives。但 Weil conjecture 直接给 projective point count，所以答案直接是 $|C(\mathbb{F}_{5^{18}})|$... 

Wait，需要扣除 zero point。Projective space 没有零点。Klein quartic 在 projective space 中，points 数就是 $|C(\mathbb{F}_{5^{18}})|$。题目说 "nonzero points up to scaling"，可能指 affine nonzero points up to scaling = projective points。所以答案 = $5^{18} + 6 \cdot 5^9 + 1 = 3814708984376$。✓

#### Intuition

Weil conjectures 把 algebraic geometry 的 point counting reduce 到 eigenvalue 计算。Genus 3 曲面只有 6 个 Frobenius eigenvalues，small field data 足以 determine 它们。本题中 $\beta_i$ 是 12th roots of unity 的特殊选择，反映了 Klein quartic 的 high symmetry（$PSL(2,7)$ automorphism group）。

Reference:
- Weil conjectures: https://en.wikipedia.org/wiki/Weil_conjectures
- Klein quartic: https://en.wikipedia.org/wiki/Klein_quartic
- Garcia 2001: https://link.springer.com/chapter/10.1007/978-3-0348-8266-8_15
- Ji 2021: https://www.math.columbia.edu/~calebji/RHcurves.pdf

## 8. 与 Related Work 对比

| Benchmark | Difficulty | Contamination Risk | Verification | Sample Size |
|-----------|-----------|-------------------|-------------|-------------|
| GSM8K | Elementary | High | Exact | 8,500 |
| MATH | HS-UG | High | Exact | 12,500 |
| AIME | Olympiad | Low (recent) | Exact | ~30/year |
| Omni-MATH | Olympiad | Medium | Exact | 4,428 |
| PutnamBench (https://arxiv.org/abs/2407.11214) | Putnam | High | Formal | 1,396 |
| MiniF2F (https://arxiv.org/abs/2109.00110) | Olympiad+UG | Medium | Formal proof | 488 |
| AIMO (https://kaggle.com/competitions/ai-mathematical-olympiad-prize) | Olympiad | Low | Exact | ~50/year |
| Putnam-AXIOM (https://openreview.net/forum?id=YXnwlZe0yf) | Putnam (modified) | Medium | Exact | Program-generated |
| **FrontierMath** | **Research-level** | **Minimal** | **Automated** | **Hundreds** |

FrontierMath 的 unique 价值：**research-level difficulty + minimal contamination + automated verification** 三者同时满足。

## 9. Expert Interviews 关键洞察

### Terence Tao (2006 Fields Medalist)
- "Extremely challenging"
- 预测 "resist AIs for several years at least"
- 关键障碍：relevant training data "almost nonexistent... you're talking like a dozen papers"
- Human-AI collaboration 可能在 ~3 年内 tackle FrontierMath problems
- 当前 guiding AI 比 direct solving 多花 ~5x effort，预期 ratio 会降至 < 1

### Timothy Gowers (1998 Fields Medalist)
- 强调 "tricks of the trade" 的重要性
- 潜在应用：验证 conjectures，"slightly boring bits of doing research"

### Richard Borcherds (1998 Fields Medalist)
- 批评：numerical format "aren't quite the same as coming up with original proofs"
- 反映了 benchmark 的 inherent limitation

### Evan Chen (IMO coach)
- Human-AI collaboration timeline ~3 years

## 10. Limitations 与 Future Directions

### 10.1 Limitations
1. **Numerical answers only**：排除 proof-writing，这是 math research 的核心
2. **Hours not months**：真实 research 跨越 weeks-months-years
3. **当前 low discrimination**：所有 models < 2%，难区分 model 间差异
4. **Error rate ~10%**：基于 Jeffreys prior 估计，与 ImageNet (~6%) 和 MMLU (~9%) 相当

### 10.2 Future Work
- 新 problems with enhanced QA
- 提高 token limit，allow longer reasoning
- Multiple runs per problem for confidence intervals
- 更 active review process（不只是 passive approval）

## 11. 个人 Intuition 构建

### 11.1 为什么 models 表现如此差？

1. **Training data scarcity**：Research-level math 在 internet 上 sparse。Tao 说 relevant papers "almost nonexistent"。LLM 的 pattern matching 在未见过的 distribution 上失效。

2. **Multi-step reasoning error accumulation**：每个 problem 需要 10+ 步精确推理。即使每步 90% accuracy，整体 $0.9^{10} \approx 35\%$，但实际每步可能 60-70%，所以 $0.65^{10} \approx 1.3\%$。

3. **Symbolic computation weakness**：Models 不擅长 long algebraic manipulation。虽然有 Python sandbox，但 knowing what to compute 需要 mathematical insight。

4. **Verification without understanding**：Model 可能通过 simulation "guess" 答案（如 Problem 1 提到的 brute-force simulation），但这不是 generalizable reasoning。

### 11.2 这对 AI for Math 意味着什么？

FrontierMath 设定了一个 meaningful milestone：**真正 research-level math 远超当前 LLM 能力**。这与 AlphaProof (https://www.nature.com/articles/s41586-024-07401-x) 在 IMO 上的成功形成 contrast——IMO problems 有大量 training data（past olympiads, AoPS forums），而 research math 没有。

Tao 的 prediction（"several years"）暗示需要：
- **Specialized architectures**：可能需要 neuro-symbolic systems
- **Synthetic data generation**：用 formal methods 生成 training data
- **Tool augmentation**：tighter integration with SageMath, Magma, Lean

### 11.3 Benchmark design 的启示

FrontierMath 的 "guessproofness" 要求很 insightful。MATH benchmark 中很多问题可以通过 pattern matching 解决（看到 "find $x$" 就猜 small integer）。FrontierMath 的 large numerical answers（如 19-digit prime）强制要求 actual computation。

**Verification-by-execution** paradigm 很 powerful：不要求 formal proof，只要求 correct numerical/symbolic answer。这降低了 evaluation cost 但保留了 difficulty。可能成为未来 science benchmarks 的 template（类似 AlphaFold 的 CASP）。

### 11.4 与 Neural Theorem Proving 的关系

MiniF2F 和 Lean-based approaches (https://leanprover-community.github.io/) 走 formal proof route。FrontierMath 走 computational answer route。两条路线 complementary：
- Formal proofs：rigorous 但 expensive to evaluate, limited to existing formalized math
- Computational answers：scalable 但 miss proof insight

未来可能 converge：Lean 4 的 Mathlib (https://github.com/leanprover-community/mathlib4) 扩展到 research math 后，可能支持 formal verification of FrontierMath-style problems。

## 12. 关键 References 汇总

- Paper: FrontierMath (Epoch AI, 2024)
- Hooley 1967: Artin's conjecture under GRH
- Järviniemi 2020: https://arxiv.org/abs/1912.02554
- Burnside 1911: *Theory of Groups of Finite Order*
- Weil conjectures: Deligne 1974 (Fields Medal work)
- Klein quartic: https://en.wikipedia.org/wiki/Klein_quartic
- AlphaProof: https://www.nature.com/articles/s41586-024-07401-x
- AlphaGeometry: https://www.nature.com/articles/s41586-023-06747-5
- FunSearch: https://www.nature.com/articles/s41586-023-06247-0
- Epoch AI: https://epoch.ai/
- GitHub: https://github.com/epoch-research/artin_code
- GitHub: https://github.com/epoch-research/finite_field_implementation

## 13. 总结

FrontierMath 代表了 AI math evaluation 的新 standard：**unsaturated, uncontaminated, research-level, automatically verifiable**。当前 SOTA models 的 < 2% success rate 揭示了巨大 gap。Sample problems 展示了现代数学的 breadth：从 analytic number theory (Artin conjecture) 到 algebraic geometry (monodromy) 到 p-adic analysis (Skolem-Mahler-Lech) 到 representation theory (Coxeter groups) 到 arithmetic geometry (Weil conjectures)。

对于 build intuition 而言，关键 takeaway 是：**research-level math 需要 deep domain expertise + creative insight + technical execution 的 combination，这三者目前 LLM 都不具备 at expert level**。FrontierMath 提供了 measuring progress 的 yardstick，但真正的 breakthrough 可能需要 paradigm shift beyond pure LLM scaling。
