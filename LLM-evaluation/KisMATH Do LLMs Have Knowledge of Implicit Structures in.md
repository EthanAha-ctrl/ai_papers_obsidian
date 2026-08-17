---
source_pdf: KisMATH Do LLMs Have Knowledge of Implicit Structures in.pdf
paper_sha256: 4944332082f58f2c62778212ceb03822d3295d081995e4d678bf3f8bca3c091f
processed_at: '2026-08-05T11:23:32-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# KisMATH 用人话说

## 这篇paper到底在纠结什么

karpathy你肯定听过这个吵了很久的架：

**一派人**（OpenAI o1, DeepSeek R1那波）说：CoT是真reasoning，模型把难题拆成小步，一步步推，还能自我纠错。

**另一派人**（Kambhampati, Stechly那波）说：扯淡。你把CoT里50%的数字随机替换，模型distill出来performance几乎不变；你换in-context examples、换reward function，performance也不变。说明CoT就是个装饰品，真正起作用的是模型从pretrain里"检索"出来的latent knowledge。

两边都有道理，但谁也说服不了谁。

KisMATH这篇paper站出来说：**你们之前的实验方式有问题**。

问题在哪？之前做扰动是**随机乱搞**——把随机token换掉、把随机example删掉。但reasoning是个有结构的东西，你随便戳一刀，戳到关键节点和戳到无关废话，效果当然不一样。你用随机扰动得出的"CoT没用"结论，可能是你戳错了地方。

所以他们做了什么？**先把reasoning trace的隐式结构提取出来，再沿着这个结构精准干预**。

---

## 他们怎么提取结构

非常朴素的想法。你拿一道数学题，模型生成一长串reasoning，里面有各种数学表达式（数字、公式、方程）。

他们用SymPy把这些表达式全部parse出来，然后做一个很聪明的操作：**从answer倒着往回追**。

比如answer是"50"，他们就问：reasoning里哪些表达式能"产生"50？找到"$625/25 + 25 = 50$"，连一条边。然后再问：$625$从哪来？$25$从哪来？一路倒推到question里的原始条件。

这样你得到一个DAG——**Causal CoT Graph (CCGraph)**。

关键细节：两个表达式"match"不要求完全一样的字符串，SymPy parse tree共享节点就行。比如"$4$"和"$4+5$"match，因为4是4+5的子表达式。

这个graph大概长这样：question node → 一串reasoning node → answer node，中间有各种交叉依赖。GSM8K平均14个node、41条边；AIME平均50个node、566条边——奥数题的依赖网络比小学题密集十几倍。

---

## 核心实验：把reasoning node"静音"

他们用的干预手段叫**attention suppression**。通俗讲：让模型在self-attention时**完全看不到**某些token的存在，不是把token删掉，是让attention机制当这些token不存在。所有层所有head同时抑制。

然后看answer的entropy变没变。

结果非常炸裂：

**干预前**，模型对answer的entropy基本是0（非常确定答案）
**干预后**，entropy暴涨到0.85-3.92（完全不知道答案是什么了）

$p < 10^{-12}$，显著到不像话。

这意味着什么？**如果你把reasoning tokens从模型的attention里抹掉，模型就不知道答案了**。reasoning tokens确实是answer的causal mediator，不是可有可无的装饰。

这是对"CoT是装饰品"那派的直接反驳——但有个重要前提：你得**精准干预到结构上**，随机扰动确实可能看不出来。

---

## 更有意思的实验：模型内部"知道"这个graph吗

上面那个实验只证明了"reasoning tokens有用"。但paper没停在这。

他们做了更妙的实验：**比较模型自己assign的概率，和我们提取的graph是否align**。

具体说：从CCGraph里抽出一条从question到answer的路径（叫R path）。然后随机生成一条同长度的path（避开graph节点）。分别算模型对这两条path的token-by-token probability，看R path的rank。

如果模型内部"意识到"这个因果结构，它应该对R path的token给更高的probability。

结果：**所有15个模型，在100th percentile处都有一个明显的spike**——意思是大量R path**整条**都比random path概率高。

这个发现的intuition：模型不是在random地生成reasoning然后碰巧得到answer。它的internal transition probability分布，天然地**align了我们用SymPy提取出来的因果图**。换句话说，CCGraph不是我们事后强加的结构，它就是模型computation的某种投影。

---

## 最有insight的发现：两种"推理性格"

仔细看rank分布，模型分两派：

**Exponential派**（代表：Qwen 3 32B）
- 几乎所有R path都是高概率transition
- $\log P(\mathcal{R})$均值≈0，方差极小
- 模型对"正确推理路径"非常自信，一条路走到黑

**Bell派**（代表：DeepSeek R1 32B）
- 少数R path包含低概率transition
- $\log P(\mathcal{R})$均值较低（-1.76），方差大（0.92）
- 推理路径上有"fork"——偶尔的高熵token

然后他们做了pass@k实验（采样k次取最好）：

- k=1时两个模型差不多（68.6% vs 71.6%）
- k=10时Bell派拉开差距（87% vs 90%）

**Bell派模型exploration能力更强**，因为那些高熵的"fork token"让模型能探索不同推理路径。

为什么会有这个差异？看训练方式：
- DeepSeek R1 32B是**从671B蒸馏**来的，保留了teacher model的intrinsic uncertainty
- Qwen 3 32B是**RLVR训练**的，reward把概率sharpen到一条路上，反而丧失了exploration

这跟Yue et al. 2025的发现一致：base model在高sample count下能outperform RLVR版本。**RLVR让模型over-confident，牺牲了exploration**。

karpathy你应该会喜欢这个——这其实是说RLVR在压缩模型的reasoning manifold，把本来rich的probability landscape压成delta distribution。

---

## "Is Math All You Need"实验

他们还做了个很clever的对照：

- $M(G)$：把所有数学表达式token静音，只留自然语言
- $M(G^C)$：把所有自然语言token静音，只留数学表达式

看answer变化率：

**GSM8K（简单题）**：suppress math → 70.9%答案变化；suppress NL → 只10.3%变化。数学表达式是绝对核心，NL就是个glue。

**MATH500和AIME（难题）**：两者差不多，没有统计显著差异。为什么？因为难题的自然语言里有discourse connectives（"因此"、"然而"、"假设"这些），还有specify数学实体role的文字。这些NL不是废话，它承载了语义结构。

所以简单题的reasoning结构是"math formula chain"，难题的reasoning结构是"math formula + discourse structure"的混合体。你光提取math表达式，对难题是不够的。

---

## 局限性，paper自己承认的

1. **只对math有效**。SymPy能parse数学表达式，但你遇到"Let midpoint of AB be O"这种geometry，或者"Let H be normal subgroup of G"这种abstract algebra，parse不了。

2. **10%的graph需要人工修**。偶尔algorithm会输出只有answer的singleton graph，得手动加边。主要是LaTeX错误或NL打断方程这种情况。比例很低（88/40K vertices），但说明algorithm不够robust。

3. **path probability是近似的**。真正的path probability要marginalize所有intermediate token distribution，这是intractable的，他们只用了实际生成的token作为conditioning context。

---

## 对你karpathy的intuition building

几个点你可能感兴趣：

**1. CCGraph是internal computation的behavioral投影**

你一直说"CoT是LLM internal computation的projection"，KisMATH给了这个直觉第一个concrete的graph-level证据。projection的不是全部computation，是math dependency那一部分。

**2. RLVR vs Distillation的exploration trade-off**

这是个值得深挖的方向。RLVR本质上是在做mode collapse——让模型对一条path极其自信。distillation反而保留了teacher的uncertainty distribution。这可能跟蒸馏时的temperature scaling有关：高T蒸馏保留soft distribution信息。

**3. Causal mediation ≠ Faithful explanation**

这篇paper证明了reasoning tokens是mediators，但这不等于CoT是faithful explanation。Barez et al. 2025和Lanham et al. 2023都论证过CoT explanation≠实际computation。CCGraph只是**部分**投影——它project了structured math dependency，没project induction heads、entity tracking circuits这些东西。

**4. 和mechanistic interpretability的对接**

CCGraph是behavioral-level的causal structure。mechanistic interp是circuit-level的。两者对齐：R path的nodes应该对应特定circuit的activation pattern，attention suppression on R path tokens应该抑制特定circuit。这是个可做的方向。

**5. CCGraph可以成为新eval paradigm**

不光看final answer accuracy，还看model的path probability分布是否align CCGraph。可以定义"CCGraph alignment score"作为reasoning capability的新metric。

---

## 一句话人话总结

**这群人用SymPy从数学推理trace里反向挖出因果图，然后精准干预证明reasoning tokens确实是答案的causal mediator（不是装饰品）；同时发现模型内部transition probability天然align这个图；最有意思的是RLVR训练让模型变over-confident丧失exploration，而蒸馏反而保留fork能力——后者在多采样下性能反超。**

核心方法论insight：**别再随机扰动了，沿着结构扰动才能看出CoT到底在不在干活。**

---

# KisMATH: 用因果图解构LLM数学推理的隐式结构

## Paper定位与核心问题

这篇paper来自Soumadeep Saha等人（ISI Kolkata, IRIT, LINAGORA Labs），解决一个被karpathy你自己也长期关注的争议：**CoT (Chain-of-Thought)到底是真reasoning还是approximate retrieval的装饰品？**

核心分为两派：
- **Decomposition派**: OpenAI o1/o3, DeepSeek R1 (Guo et al., 2025) — CoT分解复杂问题为子任务，自验证，回溯
- **Retrieval派**: Kambhampati (2024), Stechly et al. (2025), Li et al. (2025), Shao et al. (2025) — CoT对扰动不敏感，50%数字被随机替换后distill性能不变

KisMATH的核心思路：**与其随机扰动，不如沿着推理的隐式结构做graph-aligned intervention**，这样能给出比stochastic perturbation更强的证据。

参考链接：
- Paper: https://arxiv.org/abs/2503.19143 (Bogdan et al., Thought Anchors, 用的attention suppression技术)
- Dataset: https://espressovi.github.io/KisMATH
- DeepSeek R1: https://arxiv.org/abs/2501.12948 (Nature 2025)
- Paul et al. causal CoT: https://aclanthology.org/2024.findings-emnlp.849/
- Spurious Rewards (Shao et al.): https://arxiv.org/abs/2506.10947

---

## 方法核心：Causal CoT Graph (CCGraph) 构建

### 1. 数据预处理

给定三元组 $(Q, R, A)$：
- $Q$: question
- $R$: reasoning trace (由OpenAI o3生成)
- $A$: answer

用SymPy解析出所有mathematical expressions，按start index排序，得到三条non-intersecting span list：

$$\hat{Q} = [\hat{q}_1, \hat{q}_2, ..., \hat{q}_{n_Q}]$$
$$\hat{R} = [\hat{r}_1, \hat{r}_2, ..., \hat{r}_{n_R}]$$
$$\hat{A} = [\hat{a}]$$

变量含义：
- $\hat{q}_i$ = 第$i$个question span (例如 "$ab^2=5$" 这种span)
- $\hat{r}_i$ = 第$i$个reasoning span (例如 "$625/b^8 + b^8$")
- $\hat{a}$ = answer span (single)
- $n_Q, n_R$ = question和reasoning中的span数量
- "non-intersecting" 保证每个token最多属于一个span，避免歧义

### 2. Algorithm 1: 反向BFS式graph构建

算法核心思想是从answer开始**反向扩展**：

```
1: G ← ({â}, φ)  // 初始只有answer node
2: context ← concatenate(Q̂, R̂)
3: EXPAND(â, context, G)
4: PRUNE(G)  // 移除无path到question node的nodes
5: Reverse all edges in G
```

**EXPAND过程的关键细节**:

```
procedure EXPAND(î, context, G):
    if |context| ≤ |Q̂| then return  // 全是question nodes则停止
    for ĵ ∈ reversed(context) do
        p_i, p_j ← PARSE(î), PARSE(ĵ)
        if MATCH(p_i, p_j) then
            Add node ĵ to G
            Add edge (î → ĵ) to G
            EXPAND(ĵ, context[<ĵ], G)
```

**MATCH条件**（这是关键）：
- Exact string match，OR
- SymPy parse trees共享common node

例如 "$4$" 和 "$4+5$"匹配，因为4是4+5的子表达式。

**为什么要 reversed(context)**：保证只匹配在当前query之前出现的span，这样graph是DAG (directed acyclic)。

**为什么 $|context| \leq |\hat{Q}|$ 时停止**：避免研究question node之间的相互关系（那不是paper的重点）。

### 3. R path定义

公式(1)：
$$[\hat{q}_\alpha \to \hat{r}_{(i_1)} \to \hat{r}_{(i_2)} \to ... \to \hat{r}_{(i_\mu)} \to \hat{a}]$$

约束：$i_1 < i_2 < ... < i_\mu$（时序保持）

变量：
- $\hat{q}_\alpha$ = 起始question node
- $\hat{r}_{(i_\delta)}$ = 第$\delta$跳的reasoning node（$i_\delta$表示在$\hat{R}$中的原始index）
- $\mu$ = path的hop数
- $\hat{a}$ = 终止answer node

实验中取top-k longest simple Q→A paths（GSM8K k=5, MATH500/AIME k=10）。

### 4. 数据集统计

| Attribute (avg.) | GSM8K (983) | MATH500 (384) | AIME (304) |
|---|---|---|---|
| $|V|$ | 14.2 ± 4.6 | 28.7 ± 17.9 | 50.6 ± 20.9 |
| $|E|$ | 40.8 ± 30.4 | 260.8 ± 321.2 | 566.6 ± 487.7 |
| $|\hat{Q}|$ | 3.9 ± 1.4 | 7.0 ± 4.7 | 11.1 ± 6.6 |
| $|\hat{R}|$ | 9.3 ± 3.7 | 20.7 ± 14.8 | 38.5 ± 18.2 |
| len($r$) | 6.4 ± 1.8 | 8.7 ± 2.1 | 10.9 ± 1.6 |

可以观察到AIME问题的graph复杂度几乎是GSM8K的**3-4倍**，$|E|$甚至达到**14倍**，说明Olympiad-level问题的reasoning trace内部依赖网络极其密集。

---

## 因果框架：Mediation Analysis

### 理论基础

借用Paul et al. (2024)的causal view，但把整个reasoning trace拆成fine-grained graph：

$$\hat{Q} \xrightarrow{\text{direct}} \hat{A} \quad (\text{DE: Direct Effect})$$
$$\hat{Q} \xrightarrow{\hat{R}} \hat{A} \quad (\text{IE: Indirect Effect, mediated by } G)$$

**核心论点**：如果IE=0（reasoning nodes对answer无causal contribution），CoT就是装饰品。

### Attention Suppression干预

公式(2)和(3)：

$$A_i^{(\phi)} = W^O \cdot \text{concat}(A_i^{1(\phi)}, ..., A_i^{\#head(\phi)})$$

$$A_i^{j(\phi)} = \sum_{\substack{k=1 \\ x_k \notin X_{\text{supp.}}}}^T \sin(Q_i^{j(\phi)}, K_k^{j(\phi)}) \cdot V_k^{j(\phi)}$$

变量详解：
- $i$ = query token的index
- $k$ = key token的index
- $j$ = head index (1到#head)
- $\phi$ = layer index (1到#layer)
- $Q_i^{j(\phi)}, K_k^{j(\phi)}, V_k^{j(\phi)}$ = 第$\phi$层第$j$个head在位置$i$/$k$上的query/key/value projection
- $W^O$ = output projection矩阵
- $X_{\text{supp.}}$ = 被抑制token的集合
- $\sin(\cdot, \cdot)$ = similarity function（这里写法不规范，应该是softmax over scaled dot product，paper里可能笔误）

**关键点**：这是**across all layers and all heads**的抑制，确保被抑制token的information flow完全切断。Bogdan et al. (2025)论证过这种干预不会导致out-of-distribution行为。

### 熵测量

公式(4):
$$H(P_t) = -\sum_{v \in V} P(x_t = v | x_{<t}) \log P(x_t = v | x_{<t})$$
$$P_A = P(A_0 | x_{<T})$$
$$P_A^M = P(A_0 | x_T, ..., x_{\gamma+1}, x_{\gamma-\delta-1}, ...)$$

变量：
- $H(P_t)$ = 在位置$t$预测下一个token的熵
- $V$ = vocabulary
- $A_0$ = answer的第一个token (测量第一个token足够，因为后续token会条件化在它上面)
- $x_{\gamma-\delta}, ..., x_{\gamma}$ = 被suppress的reasoning node对应的tokens
- $P_A$ = 原始answer distribution
- $P_A^M$ = 干预后answer distribution

### Kolmogorov-Smirnov test

$$D_{KS} = \sup_x |F_{H(P_A)}(x) - F_{H(P_A^M)}(x)|$$

用2-sample KS test比较原始和干预后$H(P_A)$的分布，$D_{KS}$接近1表示分布显著shift。

---

## 实验结果详解

### 实验1: Reasoning nodes是mediators吗?

Table 2关键数字（以GSM8K为例）：

| Model | Orig. $H(P_A)$ | AS $H(P_A^M)$ | $D_{KS}$ |
|---|---|---|---|
| DeepSeek R1 1.5B | 0.02 | 3.58 | 1.00 |
| Qwen 3 1.7B | 1e-3 | 0.85 | 0.99 |
| Gemma 3 1B | 2e-3 | 0.88 | 0.98 |
| Llama 3.1 8B | 3e-3 | 3.23 | 0.99 |
| DeepSeek R1 70B | 0.02 | 3.92 | 0.99 |

**直觉解读**：
- 原始entropy $H(P_A) \approx 0$ 表示模型对answer非常确定（基本是delta分布）
- AS后entropy暴涨到 0.85-3.92，说明reasoning tokens承载了几乎所有causal信息
- $p < 10^{-12}$ 极其显著

**对MATH500和AIME，原始$H(P_A)$就较高** (0.08-0.58)，因为这些是Olympiad-level难题，模型本身就不太确定答案。

### 实验2: R path干预

对每个CCGraph，suppress整条R path的tokens，看answer entropy是否变化。结果在Figure 3显示$D_{KS}$高、$p < 10^{-300}$，**R path确实是causal mediator**。

### 实验3: LLMs是否"意识到"CCGraph?

这是paper最有意思的实验。比较：
- $P(\mathcal{R})$ = R path的联合概率
- $P(\tilde{\mathcal{R}}_\kappa)$ = 同长度random path的联合概率（避开CCGraph nodes）

公式(5)路径概率：
$$P(\mathcal{R}) = \prod_{\delta=1}^{\mu} P(\hat{r}_{(i_\delta)} | x_{<T_\delta})$$
$$P(\hat{r}_{(i_\delta)} | x_{<T_\delta}) = \prod_{\lambda=1}^{n} P(t_\lambda^\delta | t_{\lambda-1}^\delta, ..., t_1^\delta, x_{<T_\delta})$$

变量：
- $T_\delta$ = 第$\delta$个reasoning node的起始token位置
- $t_\lambda^\delta$ = 第$\delta$个reasoning node的第$\lambda$个token
- $n$ = 该reasoning node的token总数

公式(6) rank:
$$\text{rank}_M(\mathcal{R}) = \frac{1}{M} \sum_{\kappa=1}^M \mathbb{I}[P(\mathcal{R}) > P(\tilde{\mathcal{R}}_\kappa)]$$

**关键发现**：Figure 4显示所有模型在100th percentile处有显著spike，意味着大部分R paths**整条**都比random paths概率高。

**这是强证据**：LLMs内部计算的transition probability分布，恰好align了CCGraph中的edges。换句话说，**CCGraph不只是我们事后构造的结构，LLM的internal computation本身就在沿着类似graph传播**。

### 两种behavior模式

Figure 4的rank分布呈现两种形态：

**1. Exponential shape** (如Qwen 3 32B)
- 几乎所有R paths都是高概率transitions
- $\log P(\mathcal{R})$ 均值接近0，方差极小 (Qwen 3 32B: mean=0.0098, var=0.0002)
- 模型对"正确推理path"非常自信

**2. Bell shape** (如DeepSeek R1 32B)  
- 少数R paths包含low-probability transitions
- $\log P(\mathcal{R})$ 均值较低，方差大 (DeepSeek R1 32B: mean=1.7603, var=0.9217)
- 存在"forks" — 高熵token

### Fork token与exploration

参考Wang et al. (2025) "Beyond 80/20 rule": 少数high-entropy minority tokens驱动effective RL。

Figure 6的pass@k实验：
- k=1: DeepSeek R1 32B (71.6%) ≈ Qwen 3 32B (68.6%)
- k=10: DeepSeek R1 32B (90%) > Qwen 3 32B (87%)

**关键insight**：DeepSeek R1 32B是从DeepSeek R1 671B **distill**而来，保留了intrinsic uncertainty，所以exploration更好；Qwen 3 32B是RLVR训练，**over-confident**，反而exploration能力差。

这与Yue et al. (2025)发现一致：base model在high sample count下outperform RLVR-trained counterpart。

参考：
- Wang et al. 2025 NeurIPS: https://arxiv.org/abs/2506.01900
- Yue et al. 2025 NeurIPS: https://arxiv.org/abs/2504.13837

---

## "Is Math All You Need?"实验

这是非常聪明的对照实验：

- $M(G)$: suppress所有reasoning nodes的tokens (只保留NL glue)
- $M(G^C)$: suppress所有非reasoning node tokens (只保留math expressions)

测量Answer Change %:

Table 3关键数字（GSM8K平均）：
- $M(G)$: 70.9% — suppress math后answer大幅变化
- $M(G^C)$: 10.3% — suppress NL后answer几乎不变

**结论**：对GSM8K这种简单问题，math expressions是reasoning的核心，NL只是glue。

**但对MATH500和AIME**：
- MATH500: $M(G)$ 28-49%, $M(G^C)$ 18-63% — 差距不显著
- AIME: $M(G)$ 13-56%, $M(G^C)$ 12-79% — 差距不显著

**为什么**：复杂问题的NL中含**discourse connectives** (Mann & Thompson 1987 RST; Asher 1993; Asher & Lascarides 2003; Prasad et al. 2008 PDTB)，以及specify math entity role的text。

参考：
- RST: https://www.coli.uni-saarland.de/courses/advanced-irt-09/mt87.pdf
- Asher & Lascarides SDRT: https://mitpress.mit.edu/9780262011958/logics-of-conversation/
- PDTB: https://catalog.ldc.upenn.edu/LDC2008T08

---

## 关键技术讨论

### 与现有工作的关系

1. **Tan 2023** - 手工标注27个GSM8K例子的causal graph，发现node-level intervention引发self-correcting behavior
2. **Lee et al. 2025 ReasoningFlow** - 30个trace，annotation含planning/backtracking edges
3. **Bogdan et al. 2025 Thought Anchors** - 10个trace，三种annotation技术
4. **Mukherjee et al. 2025** - LLM-based framework识别premises

KisMATH的scalability是核心贡献：1671个graphs，9-40 nodes/problem，6-10 hops Q→A。

### R path的"路径概率"的局限

公式(5)其实**不是真正的path transition probability**，因为真正的path probability需要marginalize所有intermediate tokens：

$$P_{\text{true}}(\mathcal{R}) = \prod_{\delta=1}^{\mu} \sum_{x \in \text{tokens after } T_\delta} P(\hat{r}_{(i_\delta)} | x_{<T_\delta}, x) P(x | x_{<T_\delta})$$

这是intractable的，所以paper采用**近似**：直接用模型实际生成的tokens作为conditioning context。

为了控制path length和token frequency的影响，paper只用**相对rank**而不是绝对probability。

### Manual Intervention细节

Algorithm 1的PRUNE步骤偶尔(10%)会输出singleton graph，需要manual fix。两个categories：
1. **LaTeX errors**: missing "$$", unmatched "{}"
2. **NL interruption**: "4+5 is 9" → "4+5=9"

总体: 88/40K vertices 和 71/300K edges需要manual edit — 比例极低，说明algorithm相当robust。

### Limitations

Paper坦诚承认：
- 不适用于geometry (e.g., "Let midpoint of AB be O")
- 不适用于abstract algebra ("Let H be normal subgroup of G")
- 不适用于commonsense reasoning
- 需要NL context capture semantic similarity

---

## 对karpathy的intuition building

### 1. CCGraph ≈ LLM内部的隐式计算图

如果LLM的attention和MLP layer真的是在执行某种结构化computation，那么CCGraph就是这种computation的**显式投影**。R path的high probability不是偶然——LLM的forward pass本身就在incrementally build这种dependency。

这与你的"microscope into LLM"直觉吻合：reasoning tokens不是random noise，它们是computation的intermediate register。

### 2. RLVR压缩了exploration manifold

Bell vs exponential shape的对比很有启发。**RLVR本质上是优化mode collapse**：模型学到one specific high-probability path，但失去了forking ability。

Distillation反而保留了teacher model (DeepSeek R1 671B)的intrinsic uncertainty distribution。这是反直觉的：distill通常被认为是lossy的，但在exploration维度上distill反而更rich。

这可能与**知识蒸馏中temperature scaling**有关：用高T蒸馏能保留soft distribution信息。

### 3. Causal Mediation ≠ Faithful Explanation

虽然paper证明reasoning nodes是mediators，但这**不等于**CoT是faithful explanation。Barez et al. (2025)和Lanham et al. (2023)都论证过CoT explanation≠实际computation。

karpathy你的microscope理论实际上更细致：CoT是**部分投影** of internal computation，project了structured math dependency，但**没project**很多其他circuit（如induction heads, entity tracking circuits, Fagnou et al. 2024）。

### 4. 数学推理的特殊性

KisMATH只对math reasoning有效，因为math有明确的symbolic parser (SymPy)。对commonsense/code/logic推理，需要不同的"expression extraction"方法。

潜在方向：
- Code reasoning: AST作为graph nodes
- Logical reasoning: proof tree作为graph  
- Commonsense: knowledge graph subgraph作为nodes

### 5. 联想到mechanistic interpretability

CCGraph提供了**behavioral-level的causal structure**，但mechanistic interpretability研究circuit-level的causal structure。两者对齐的方向：
- R path nodes应该对应特定circuit的activation patterns
- Attention suppression on R path tokens应该抑制特定circuit
- 可以用anthropic的circuit finding技术验证CCGraph nodes

参考：
- Anthropic circuits: https://distill.pub/2020/circuits/zoom-in/
- Induction heads: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

### 6. 关于"spurious rewards"的revisit

Shao et al. 2025 "Spurious Rewards"发现RLVR即使reward wrong也能improve performance。这与KisMATH的发现compatibility如何？

可能的解释：
- RLVR通过**structure sharpening**改善performance — 让模型更确定它本来就在执行的path
- 这种sharpening不依赖reward correctness，因为path本身是pre-trained model的capability
- 但sharpening有副作用：loss of exploration（Bell → Exponential转变）

### 7. CCGraph作为benchmark

KisMATH可以成为新的eval paradigm：
- 不仅看final answer accuracy
- 还看LLM的path probability分布是否align CCGraph
- 看model对R path intervention的sensitivity

可以构建"CCGraph alignment score"作为新的reasoning capability metric。

---

## 公式汇总与变量速查

| 公式 | 用途 | 关键变量 |
|---|---|---|
| Algorithm 1 | 反向BFS构建CCGraph | $\hat{Q}, \hat{R}, \hat{A}$ span lists |
| Eq. 1 | R path定义 | $\hat{q}_\alpha$ (起点), $\hat{r}_{(i_\delta)}$ (中间), $\hat{a}$ (终点), $\mu$ (hop数) |
| Eq. 2-3 | Attention suppression | $A_i^{j(\phi)}$ ($i$=query, $j$=head, $\phi$=layer) |
| Eq. 4 | Entropy measurement | $H(P_A)$ 原始, $P_A^M$ 干预后, $A_0$ answer首token |
| Eq. 5 | Path probability | $T_\delta$ reasoning node位置, $t_\lambda^\delta$ node内token |
| Eq. 6 | Rank metric | $M$ random paths数, $\mathbb{I}[\cdot]$ 指示函数 |

---

## 一句话总结

**KisMATH通过SymPy构建的反向DAG证明了：LLM数学推理的CoT tokens是causal mediators，且LLM内部transition probability分布与extracted CCGraph高度align；同时发现RLVR训练导致exploration能力下降（exponential shape），而distillation保留intrinsic uncertainty（bell shape），后者反而pass@k更高。**

对karpathy你而言，这是**从behavioral level补强了你关于"CoT是LLM内部computation的projection"的直觉**，并且给出了graph-aligned intervention作为新的microscope。
