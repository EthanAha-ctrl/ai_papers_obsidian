---
source_pdf: Towards System 2 Reasoning in LLMs Learning How to Think With Meta Chain-of-Thought.pdf
paper_sha256: 8954ebed4216691b83ea0c45ed6053788bf5f250fd12509f03fcd086e45cbcd4
processed_at: '2026-08-12T17:30:23-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Meta-CoT

## 一句话版本

**现在的大模型只会"背答案"，不会"想问题"——因为训练数据里只有答案，没有思考过程。要让它真会想，得把"搜索过程"也喂给它学。**

---

## 故事从哪开始

想象你看一个数学家的笔记本。最后一页是工整的证明，三步搞定。你以为他三步就想出来了。但真相是：他前面撕掉了五十页草稿——试错、绕路、推翻重来。

现在 LLM 的训练数据，就是那最后一页。我们让模型学"看到题→写出最后一页"，它就只会模仿这个表面 pattern。

这就是为什么 GPT-4o 能做简单数学但 IMO 题一题都做不出。它从来没见过那五十页草稿长什么样。

---

## 关键观察：答案容易，但判定哪个答案对更容易

这是整篇论文最深的 insight，来自一个实验：

拿 LLaMa 8B 微调。让它做数学题：
- 直接答：40% 对
- 让它答 64 次，用 oracle verifier 挑最好的：85% 对

**8B 的小模型，只要能多试几次再挑对的，性能就能超过很多 frontier 大模型。**

这说明什么？说明"生成正确答案"和"判定答案对不对"，两者的难度根本不在一个量级。生成答案很难，但判定答案对不对，相对容易。

这就是 generator-verifier gap。跟 P vs NP 是一回事：解一个问题很难，验证一个解对不对很容易。

---

## 那 o1 为什么这么强

论文给了一个很有说服力的证据。

拿 HARP benchmark（高中奥数级别）测试：

**Level 1 题**（简单）：o1 生成的 token 数 ≈ 人类 solution 长度
**Level 5 题**（难）：o1 生成的 token 数远远超过人类 solution

也就是说，题目越难，o1 想得越久。它不是在"写答案"，它是在**in-context 跑搜索**——反复尝试、回溯、换路径，直到找到对的。

人类写出来的 solution 是被"linearize 过的"——你不会看到 IMO 冠军的论文里写"我试了凸包方法不行然后试了图论方法也不行然后突然想到构造法"——但 o1 直接把这些过程都吐出来了。

**o1 内化了搜索过程，在 weight 里学会了"how to think"。**

---

## 怎么让模型学会搜索

### Step 1：造"搜索数据"

拿一道数学题。用 MCTS 或 A* 跑搜索，把整个搜索过程 linearize 成一个 token 序列：

```
题目: ...
尝试1: ... (value: 0.3)
尝试2: ... (value: 0.5) [BACK]
回到尝试1
尝试3: ... (value: 0.9)
...
正确答案: ...
```

这个序列就是 Meta-CoT。它把搜索的"草稿"也写出来了。

### Step 2：SFT 训练

让 transformer 用 next-token prediction 学这些序列。模型就这样"学会"了在 token 序列里表达搜索。

证据：Lehnert et al. 2024 在 maze 上做过。把 A* search linearize 训练，结果：
- 小迷宫：search-augmented 和普通 model 差不多
- 大迷宫：search-augmented 远超普通 model

跟 o1 在 HARP 上的行为模式一模一样——**问题越复杂，搜索的价值越明显**。

### Step 3：RL post-training

光 SFT 还不够。问题：训练时数据是 reference model 生成的，测试时是 current model 生成的——distribution shift。SCoRe (Kumar 2024) 实证：模型越训练越能改别人的错，但越不能改自己的错。

解决方案：E-RL²（Stadie 2019）：
- 让模型先 K-1 轮"自由探索"，不奖励
- 最后一轮才给 reward

这样模型不会 collapse 到 greedy 策略（一上来就押宝），而是真的学会"先探索再下结论"。

RLEF (Gehring 2024) 在代码任务上实证：
- SFT：基本无效（甚至降性能）
- RLEF：8B 提升 5 个点，70B 提升 12 个点

**SFT 不能让模型学会 in-context exploration，RL 才能。**

更震撼的：把 compiler feedback 换成无关题目的输出，RL 后性能仍提升。说明模型学的不是"用 verifier 信息"，而是真的学会了"in-context 探索策略"本身。

---

## 为什么 standard RLHF 不行

论文引用 Ghosh 2021 的理论结果：

**Standard RL 在 test-time 上的性能可以 arbitrarily bad 相对于 Bayes-optimal behavior。**

直觉：标准 RL 是在某个具体 MDP 上 maximize reward。但推理任务本质是 POMDP——reward 函数对每个新问题都是 unknown 的（你不知道哪些 solution 会被接受）。在一个 specific MDP 上最优的策略，换到新 MDP 上可能表现极差。

这就是为什么 GPT-4 经过大量 RLHF 还是不会真正的 System 2 reasoning。它的 RL 是在"给定的 MDP"上学的，但 test 时 MDP 变了。

需要的是 **meta-RL**：学的是"如何在未知 MDP 上快速 adapt"——也就是 in-context exploration 策略本身。

---

## 模型真的在搜索吗

论文做了 trace forensics，分析 o1、R1、Gemini 2.0 Thinking、QwQ 的输出：

**o1**：
- 逻辑流不连贯（前后 step 不衔接）
- 频繁 semantic backtracking（回到之前的某点）
- 重复 logical step

这些行为跟 MCTS/A* 的 linearized trace 完全同构。搜索树本来就有"回到某个 node 重新探索"的行为，linearize 后自然出现"逻辑不连贯 + 重复"。

**DeepSeek-R1**：
- 同上，但多了 explicit self-evaluation（"让我检查一下这个对不对"）
- 所以逻辑流比 o1 顺一些
- 像 LATS（MCTS + self-reflection）

**Gemini 2.0 Thinking**：
- 少 backtracking
- 经常整条 solution 推倒重来（从 final state reset 到 initial）
- 像 revision-based strategy（生成完整解 → 验证 → 不对就再来一遍）

**Qwen QwQ**：
- 简单题也生成十几条 solution
- 像在 context 里跑 Best-of-N

> 不同模型用了不同的 search strategy。o1/R1 偏 tree search，Gemini/QwQ 偏 episode-level revision。

---

## Prompting 模仿 o1 行不通

论文做了个很扎心的实验。让各种 model 用 5 种 prompt（从简单到 "Think & Verify"）做数学题：

- Baseline / CoT prompt：后悔/回溯表达 < 0.5%
- Think & Verify prompt：LLaMa 70B 后悔率 25.67%，GPT-4o 只有 1.5%

但**当模型表达后悔/回溯时，最终答案更可能错**。token 数随难度增加，但 accuracy 没相应提升。

**模型在"装"思考。**它学会了输出"嗯让我想想...哦不对..."这种 token pattern，但底层并没有真正的 search 算法在运行。

这跟 Gudibande 2023 的"imitating proprietary LLM 失败"一致——你只能模仿表面的语言 pattern，没法模仿背后的 weight-level capability。

> 要真的让模型会 System 2 reasoning，必须在 weight 上动刀——SFT on search traces + RL post-training。Prompt 工程不行。

---

## 完整 pipeline

```
1. 收集 verifiable math problems (Big MATH, 100万+)
2. 用 MCTS/A* 在这些问题上跑搜索，linearize 成 token trace
   = (question, Meta-CoT search trace, final solution)
3. SFT: 让 base model 学这些 trace，获得"能表达搜索"的能力
4. RL (E-RL²): 在 verifiable reward 上 fine-tune
   - 多轮 in-context exploration
   - 最后一轮才 reward
   - KL to SFT model
5. 推理时：模型自回归地生成 Meta-CoT，在 context 内跑搜索
```

可选：加 discount rate 控制过度思考（否则模型解 $2+3=?$ 都要生成 13 条 solution，QwQ 就这样）。

---

## 还没解决的事

1. **Verifier gap**：训出来的 PRM 还是远弱于 MC rollout。搜索效率大打折扣。
2. **CoT faithfulness**：模型可能 CoT 里跳步骤、或者 CoT 跟实际推理过程不符。O1 在官方示例里就有这个问题（用了没证明的假设）。
3. **Super-intelligence 没证据**：当前所有 gains 都是 efficiency（shift left on compute-accuracy curve），不是 capability shift（shift up）。模型还不会"发明"新的推理算法。
4. **Context length limit**：in-context search 摊平成 linear token 序列，受 context 限制。Explicit tree search 可以并行 + 跳过 subtree，in-context search 没这些优势。
5. **Search²**：能不能在已经会 in-context search 的 model 上再叠一层 explicit search？Anonymous 2024 的 multi-turn PRM 实验显示 promising。

---

## 最直觉的 takeaways

1. **训练数据 = 答案 ≠ 思考过程**。这就是为什么 SFT on solution 学不会推理。
2. **Generator-verifier gap 是 search 的根本动力**。验证比生成容易，所以多试几次总能蒙对。
3. **o1 在做的事 = in-context linearized tree search**。它 weight 里内化了搜索算法。
4. **SFT 给能力，RL 给行为**。SFT 让模型"会表达搜索"，RL 让模型"真的去搜索"。
5. **Standard RLHF 必然失败**，因为推理是 POMDP，需要 meta-RL。
6. **Prompting 模仿 o1 是没用的**。weight 不行就是不行。
7. **目前 efficiency gains 明确，super-intelligence 还没出现**。

---

## 这篇论文的意义

它把 OpenAI o1、DeepSeek R1、Gemini Thinking、QwQ 这些 closed-source reasoning model 的 "secret sauce" 用一个连贯的理论框架解释了：

**它们都是在 weight 里 internalize 了 in-context search。**

训练方法大概是：SFT on linearized MCTS/A* traces + E-RL² fine-tune。

这给了 open-source 社区一个清晰的复现 roadmap。Big MATH 项目（100 万+ 高质量数学题）解决数据问题，GPT-NeoX 异步 RLHF infrastructure 解决算力问题，Meta-STaR + E-RL² 解决算法问题。

合起来，这就是 open-source o1 的完整 playbook。

---

# Meta Chain-of-Thought: Building Intuition

这篇 paper 来自 Stanford (Chelsea Finn, Nick Haber 等人) + SynthLabs + UC Berkeley (Charlie Snell)，核心论点非常清晰：**当前 LLM 训练数据里的 CoT 不是真正的数据生成过程 (DGP)**，特别是对 hard reasoning 问题。真正的 DGP 是一个 latent search process，模型需要学会 "how to think"，而不仅仅是 "what to think"。

论文链接：https://arxiv.org/abs/2503.05170

我会按论文的逻辑链展开，每个关键概念都给公式 + 变量解释 + intuition。

---

## 1. The Core Argument: Why Classical CoT Fails

### 1.1 Starting from the compression-is-intelligence premise

LLM 训练的 MLE 目标：

$$\mathcal{L}_\theta = \mathbb{E}_{\mathcal{D}_{\mathrm{train}}}\left[-\sum_t \log p_\theta(\mathbf{y}_{t+1} | \mathbf{y}_{\le t})\right]$$

其中 $\mathbf{y}_t$ 是 token, $p_\theta$ 是 transformer, $\theta$ 是参数。论文的关键 insight 是：

**Conditional generative process $p(\mathbf{y}_{t+1}|\mathbf{y}_t)$ 可以有 arbitrarily high computational complexity，即便它是 deterministic 的。**

举个漂亮的例子：$\frac{(x^2-1)(x+1)}{x^3-x} - \frac{1}{x}$ at $x=\pi$。答案是 1（化简后），但 GPT-4o 和 Claude 都答不对。理由是：如果直接 next-token 预测答案，需要的"内部计算"复杂度远超 transformer constant-depth forward pass 能表达的。CoT 把这部分计算"摊开"到序列维度上。

参考文献 [Merrill & Sabharwal 2023](https://arxiv.org/abs/2310.07923) 论证：CoT 让 transformer 表达能力从 $\mathrm{TC}^0$ 跳到能表达任意图灵可计算函数（in theory）。

### 1.2 但 CoT 还是不够：latent thought 被丢失

论文的核心论断在 Equation (1)：

$$\mathbf{q} \to \mathbf{z}_1 \to \dots \to \mathbf{z}_K \to (\mathbf{s}_1, \dots, \mathbf{s}_n, \mathbf{a})$$

其中：
- $\mathbf{q}$: question
- $\mathbf{z}_i$: latent "thoughts"，是 DGP 真实经过的潜在推理步骤（不是最终 CoT 的一部分）
- $\mathbf{s}_t$: 最终呈现出来的 CoT step（在 solution 文本里）
- $\mathbf{a}$: final answer

经典 CoT 的 marginalization 是：

$$p_{\mathrm{data}}(\mathbf{a}|\mathbf{q}) \propto \int \underbrace{p_{\mathrm{data}}(\mathbf{a}|\mathbf{s}_1, \ldots, \mathbf{s}_n, \mathbf{q})}_{\text{Answer Generation}} \underbrace{\prod_{t=1}^n p_{\mathrm{data}}(\mathbf{s}_t|\mathbf{s}_{<t}, \mathbf{q})}_{\text{CoT}} d\mathbf{S}$$

而 Meta-CoT 把 latent thoughts 也显式建模：

$$p_{\mathrm{data}}(\mathbf{a}, \mathbf{s}_1, \ldots, \mathbf{s}_n|\mathbf{q}) \propto \int \underbrace{p_{\mathrm{data}}(\mathbf{a}, \mathbf{s}_1, \ldots, \mathbf{s}_n|\mathbf{z}_1, \ldots, \mathbf{z}_K, \mathbf{q})}_{\text{Joint Answer + CoT}} \underbrace{\prod_{t=1}^K p_{\mathrm{data}}(\mathbf{z}_t|\mathbf{z}_{<t}, \mathbf{q})}_{\text{Meta-CoT}} d\mathbf{Z}$$

### 1.3 Windmill problem 的直觉

论文举了 IMO 2011 的 windmill 题作例子（图论/凸包都不工作，需要 inductive construction with dynamic analysis）。**关键 insight**：solution 表面上是几个 step 的 linear text，但是 DGP 不是 linear 的——解题者实际上做了大量探索、试错、induction，最后才把"成功的"路径 linearize 成了 textbook solution。所以：

> **Textbook solutions $p(\mathbf{s}_1, \ldots, \mathbf{s}_n | \mathbf{q})$ 不等于真实生成过程 $p(\mathbf{z}_1, \ldots, \mathbf{z}_K | \mathbf{q})$**

这个 gap 就是 paper 的 central thesis。

### 1.4 o1 的 token-length scaling 就是证据

Figure 1 (HARP benchmark, [Yue et al. 2024](https://github.com/aadityasingh/HARP)) 显示：
- Level 1 问题：o1 生成 token 数 ≈ 人类 solution 长度
- Level 5+ 问题：o1 生成 token 数远远超过人类 solution（且性能远超 baseline model）

这强烈暗示 **o1 在生成 Meta-CoT（latent search trace 的 linearization）**，而经典 model 只是在拟合训练数据里的 $p(\mathbf{S}|\mathbf{q})$。

---

## 2. Search as the DGP: Formalizing Reasoning as MDP

### 2.1 把推理形式化为 MDP

推理 = 在 partial solution 上的搜索，写成 MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$：

- $\mathcal{S}$: state space, state $\mathbf{S}_t = (\mathbf{q}, \mathbf{s}_1, \ldots, \mathbf{s}_t)$（prompt + 已生成的 step）
- $\mathcal{A}$: action = next reasoning step $\mathbf{a}_{t+1} = \mathbf{s}_{t+1}$
- $P(\mathbf{s}'|\mathbf{s},\mathbf{a})$: transition，对纯推理用 deterministic append: $P(\cdot | \mathbf{S}_t, \mathbf{s}_{t+1}) \mapsto (\mathbf{q}, \mathbf{s}_1, \ldots, \mathbf{s}_{t+1})$
- $R(\mathbf{s}, \mathbf{a})$: 通常是 sparse reward，final = 1 if correct else 0
- $\gamma \in [0,1]$: discount factor

Policy: $\mathbf{s}_{t+1} \sim \pi_\theta(\cdot|\mathbf{S}_t)$（LLM 本身）

### 2.2 Generator-verifier gap 是 search 必要性的根本

LLaMa 3.1 8B fine-tune on Numina MATH 的实验（Figure 2）：
- Greedy: 20% → 40%
- pass@4 (oracle verifier): 已经超过 greedy 的最终性能
- pass@64: ~85% on MATH（远超很多 frontier model 的 zero-shot）

这是一个非常深的 insight：**生成正确答案 vs 判定哪个答案正确，两者的 complexity 是 asymmetric 的**。论文把这跟 P vs NP 类比（虽然不敢直说），说这是一个 fundamental empirical fact（[Brown et al. 2024](https://arxiv.org/abs/2407.21787), [Snell et al. 2024](https://arxiv.org/abs/2408.03314)）。

### 2.3 从 Best-of-N 到 General Tree Search

Best-of-N 是低效的（即使早期 step 错了也得跑完整条 solution）。Tree search (RAP, ToT, LATS) 用 PRM $v_\theta(\mathbf{q}, \mathbf{S}_t) \to [0,1]$ 估计中间 state 的"会成功"概率：

1. **Terminate** 一条不 progress 的分支
2. **Reset** 到 high-value 中间 state

这俩操作 + 语言结构 = 可以实现任何 tree search procedure ([Yao et al. 2023 Tree of Thoughts](https://arxiv.org/abs/2305.10601), [Hao et al. 2023 RAP](https://arxiv.org/abs/2305.14992), [Zhou et al. 2024 LATS](https://arxiv.org/abs/2310.04406))。

### 2.4 Scaling laws for search: Jones 2021 + Feng 2024

[Jones 2021](https://arxiv.org/abs/2104.03113) 的 AlphaZero on board games 实验（Figure 6）显示：

- Train-time compute 和 inference-time compute 都 improve Elo
- log-log scaling trade-off between train and test compute
- 同一 checkpoint，加 inference search → sigmoid 性能曲线（每个模型 size 都有自己的 sigmoid）

[Feng et al. 2024 TS-LLM](https://arxiv.org/abs/2404.08849)（LLaMa 7B + GSM8k + MCTS fine-tuning，Table 1）显示：训练后 zero-shot 提升 + inference search 还能继续推。

> 论文核心猜想：**model size, training compute, inference search 三者共同支配性能**，类似 Chinchilla scaling law 但加了 third axis。

---

## 3. Internalizing Search: 让自回归模型自己 "search in context"

### 3.1 为什么要 internalize

两个 motivation：
1. **Efficiency**: natural language reasoning 分支之间常 semantically overlap（不像棋盘 unique state），in-context search 可以复用 KV cache + 避免 explicit tree 的开销。
2. **Super-intelligence**: 如果 model 能在 in-context 实现 search，再加 RL post-training，可能优化的是 **algorithm 本身**而不是 **solution**——可能 discover novel reasoning approach（这是 meta-RL 的 promise）。

### 3.2 STaR → Meta-STaR

**STaR** ([Zelikman et al. 2022](https://arxiv.org/abs/2203.11365)):

1. Sample $\hat{\mathbf{a}}, \hat{\mathbf{S}} \sim \pi(\mathbf{a}, \mathbf{S}|\mathbf{q})$
2. Keep only 正确的：$\hat{\mathbf{a}} = \mathbf{a}$
3. SFT:

$$\mathcal{L}_{\mathrm{STaR}}(\pi_\phi) = -\mathbb{E}_{(\mathbf{q}, \hat{\mathbf{S}}, \mathbf{a}) \sim \mathcal{D}_{\mathrm{STaR}}}[-\log \pi_\phi(\mathbf{a}, \hat{\mathbf{S}}|\mathbf{q})]$$

迭代多次。

**Meta-STaR**：把 $\hat{\mathbf{S}}$ 换成 search trace $\hat{\mathbf{Z}} = \hat{\mathbf{z}}_1, \ldots, \hat{\mathbf{z}}_K$：

$$\mathcal{L}_{\mathrm{Meta\text{-}STaR}}(\pi_\phi) = -\mathbb{E}_{(\mathbf{q}, \hat{\mathbf{Z}}, \hat{\mathbf{S}}) \sim \mathcal{D}_{\mathrm{STaR}}}[-\log \pi_\phi(\hat{\mathbf{S}}, \hat{\mathbf{Z}}|\mathbf{q})]$$

直觉：让 model 学会 **从 q 直接 autoregressively 生成 search 过程 + 最终 solution**。

### 3.3 实证证据 1: Maze + A* 

[Lehnert et al. 2024](https://arxiv.org/abs/2402.14083)（FAIR）：

- 把 A* search 在 maze 上 linearize 成 token stream（Figure 7）
- 训练 transformer 做 next-token prediction on these traces
- Result (Figure 8)：
  - Search-augmented model 在所有 train compute scale 上 outperform solution-only
  - **关键发现**：maze size 越大（complexity 越高），search-augmented vs zero-shot 的 gap 越大 → **跟 o1 在 HARP 上的行为完全同构**（Figure 1）

> 这是非常漂亮的对应：maze size ↔ HARP difficulty ↔ o1 token length scaling 三者都是同一 phenomenon。

### 3.4 实证证据 2: Countdown + Stream of Search

[Gandhi et al. 2024 Stream of Search (SoS)](https://arxiv.org/abs/2404.03683)：

- Countdown game (24 point-ish)
- Linearize search：`explore state | action | result | backtrack | ...`
- 用 250M transformer 训练
- Inference-time scaling (Figure 9 中间)：log-linear relationship between tokens spent 和 success rate
- 跟 o1 在 AIME 上的曲线同形

RL post-training (STaR style)：success rate 从 1% (SFT) → 4% (RL)，且 solve 一些 symbolic search 解不了的问题（Figure 23 右）。

### 3.5 实证证据 3: 多轮 revision (in-context Best-of-N)

[Qu et al. 2024 SCoRe](https://arxiv.org/abs/2407.18219), [Snell et al. 2024](https://arxiv.org/abs/2408.03314)：

训练时让 model 看到自己之前 $j-1$ 个 wrong solutions，再生成 correct one：

$$\mathbf{S}^j \sim \pi_\theta(\cdot|\mathbf{S}^{j-1}, \ldots, \mathbf{S}^1, \mathbf{q})$$

训练目标 (off-policy SFT 形式)：

$$\min_\theta \mathbb{E}_{\mathbf{S}^i \sim \pi_{\mathrm{ref}}, \mathbf{q}}[-\log \pi_\theta(\mathbf{S}^*|\mathbf{S}^{j-1}, \ldots, \mathbf{S}^1, \mathbf{q})]$$

测试时 (Figure 10)：pass@1 随 in-context revisions 数量持续提升，且 **autoregressive (sequential) 比 parallel sampling 更高效**（同一 token budget）。

### 3.6 Variable compute: 让 model 自己决定探索多少

训练时 $j \sim \mathrm{Unif}(1, 8)$，并训练 model 在 high-confidence 时 emit EOS：

$$\min_\theta \mathbb{E}[-\log \pi_\theta(\mathbf{S}^*, \mathrm{EOS}|\mathbf{S}^{j-1}, \ldots, \mathbf{S}^1, \mathbf{q})], j \sim \mathrm{Unif}(1, 8)$$

**Empirical** (Figure 11)：model 自动按 difficulty 调整 revisions 数量（Level 1: 2.45 次, Level 5: 5.84 次）→ 跟 o1 在 HARP 上的 token 行为一致。

### 3.7 Backtracking: 关键能力的引入

[Ye et al. 2024b](https://arxiv.org/abs/2408.16293)：训练时插入 incorrect step + `[BACK]` token：

$$\mathcal{L}_{\mathrm{backtrack}}(\theta) = -\mathbb{E}[\log \pi_\theta(\mathbf{s}_1, \ldots, \mathbf{s}_t^-, [\mathrm{BACK}], \mathbf{s}_t, \ldots, \mathbf{s}_n|\mathbf{q})]$$

$\mathbf{s}_t^-$ 是一个 incorrect step, $t \sim \mathrm{Unif}(1, n)$。50% incorrect rate 时从 78% → 94% on hard math（虽然 124M model）。

[Zhang et al. 2024b](https://arxiv.org/abs/2409.14586) 在 Gemma 2B / LLaMa 3 8B 上做 backtracking for safety，且 mask unsafe tokens（只 train 在 safe path）：

$$\mathcal{L}(\theta) = -\mathbb{E}[\log \pi_\theta([\mathrm{BACK}], \mathbf{S}^+|\mathbf{S}_t^-, \mathbf{q}) + \log \pi_\theta(\mathbf{S}^+|\mathbf{q})]$$

---

## 4. Synthetic Meta-CoT via Search (核心方法)

### 4.1 Value estimation by Monte-Carlo rollouts

中间 state 的 value 用纯 MC rollout 估计：

$$v(\mathbf{S}_t, \mathbf{q}) = \mathbb{E}_{\mathbf{S}_{\ge t+1}^j \sim \pi_\theta(\cdot|\mathbf{S}_t, \mathbf{q})} \frac{1}{K} \sum_{j=1}^K r^*([\mathbf{S}_{\ge t+1}^j, \mathbf{S}_t], \mathbf{q})$$

- $K$: rollout 数量（论文用 128）
- $r^*$: ground-truth outcome verifier (e.g., final answer match)
- $\mathbf{S}_{\ge t+1}^j$: 从 $\mathbf{S}_t$ 开始的完整 completion

每个 state 估 value 要 128 个 completion，代价极高（论文说一个 MCTS tree ~ $100，~20M tokens）。

### 4.2 MCTS

UCT 选择策略 (Appendix D)：

$$U(\mathbf{S}_t, \mathbf{s}) = Q(\mathbf{S}_t, \mathbf{s}) + c_{\exp}\sqrt{\frac{\log N(\mathbf{S}_t, \mathbf{s})}{N(\mathbf{S}_t)}}$$

- $Q(\mathbf{S}_t, \mathbf{s})$: 当前 state-action value
- $N(\mathbf{S}_t, \mathbf{s})$: 该 action 被访问次数
- $N(\mathbf{S}_t) = \sum_s N(\mathbf{S}_t, \mathbf{s})$: 该 state 总访问次数
- $c_{\exp}$: exploration constant

Backup 更新（incremental mean）：

$$Q(\mathbf{S}_i, \mathbf{s}_{i+1}) \mathrel{+}= \frac{v(\mathbf{S}_t, \mathbf{q}) - Q(\mathbf{S}_i, \mathbf{s}_{i+1})}{N(\mathbf{S}_i, \mathbf{s}_{i+1})}$$

### 4.3 A* variant

[Koh et al. 2024 Tree Search for LM Agents](https://arxiv.org/abs/2407.01476)：

- 维护 frontier $\mathcal{F}$ 作为 max-priority queue
- 每步 pop value 最高的 state $\mathbf{S}_p$
- 用 policy 采样 $b$ 个 candidates
- 用 value function 估值并 push 入 frontier
- 直到深度 $d$ 或找到 correct solution

Figure 13 显示 A* trace 比 MCTS 更"干净"，回溯集中在几个关键 step。

### 4.4 o1/R1/Gemini/QwQ 行为分析 (Section 4.4)

论文做了 trace forensics：

- **o1**: logical flow 不连贯 + 频繁 semantic backtracking + 重复 logical step → 看起来像 in-context MCTS/A*
- **DeepSeek-R1**: 同上 + 显式 self-evaluation（像 LATS 的 self-criticism，[Madaan et al. 2023](https://arxiv.org/abs/2303.17651)）→ logical flow 比 o1 顺
- **Gemini 2.0 Flash Thinking**: 少 backtracking, 经常从 final state 整个 reset 到 initial → 像 revision-based strategy
- **Qwen QwQ**: 多次完整 solution 尝试 in context（13 次解 $2+3=?$，[Chen et al. 2024](https://arxiv.org/abs/2412.21187)）

> 论文猜想：o1/R1 训练时用过 MCTS-style 的 linearized search traces（很可能用作 SFT 初始化）。

---

## 5. Process Supervision: PRM

### 5.1 PRM 的训练

PRM $v_\theta(\mathbf{q}, \mathbf{S}_t) \to [0,1]$ 用 cross-entropy on $(\mathbf{S}_t, y_{\mathbf{S}_t})$ 训练。

Label 来源两条路：
1. **Human annotation** ([Lightman et al. 2023 PRM800K](https://arxiv.org/abs/2305.20050))：贵，难 scale
2. **MC rollout amortization** ([Wang et al. 2024 Math-Shepherd](https://arxiv.org/abs/2310.20647))：用 outcome verification 拟合一个 value function，把 MC estimate 当作 target $y_{\mathbf{S}_t}$

### 5.2 PRM scaling 实验结果 (Figure 16, 17)

- Training data: 500 / 3000 / 7086 unique questions
- MAE 在 training data 越多时越低
- 小数据集 (~30% epoch) 就 early-converge
- Best-of-N 时 PRM 训练得越好，BoN 越接近 oracle
- Beam search (N=5, beam width=4) 时 PRM 训练得越好，accuracy 越高 + token 用量越少

### 5.3 Verifier gap 仍是 open problem

PRM 仍显著弱于 oracle + MC rollout（Section 8.3.2）。Figure 3 也显示 BoN+verifier 与 oracle pass@N 仍有 gap。这是 **fundamental verifier gap** 而不只是 efficiency 问题。

---

## 6. Meta-RL: The Deep Theoretical Frame

### 6.1 POMDP framing

推理任务是 POMDP：reward function $r(\mathbf{S}, \mathbf{q}) \to \{0, 1\}$ 是 deterministic 但 a-priori unknown——你不知道哪些 solution 会被接受。

[Ghosh et al. 2021](https://arxiv.org/abs/2107.06277) 的关键 remark：

> Standard RL-trained policies (max reward in MDP from posterior) 可以在 test-time 上 **arbitrarily bad** 相对于 Bayes-optimal behavior。

这正好解释了为什么 standard RLHF 不能产生 o1-like 行为。

### 6.2 Meta-learning objective

$$\min_\theta \mathbb{E}_{\mathbf{q} \sim \mathcal{D}_{\mathrm{train}}} \mathbb{E}_{\pi_{U(\theta)}}[L_\mathbf{q}(\theta)]$$

- $U$: adaptation procedure（in-context 探索/搜索就是 adaptation）
- $L_\mathbf{q}$: task q 的 loss

Revision model 形式 (off-policy 版)：

$$\min_\theta \mathbb{E}_{\mathbf{q}} \mathbb{E}_{\mathbf{S}^i \sim \pi_{\mathrm{ref}}}[-\log \pi_\theta(\mathbf{S}^*|\mathbf{S}^j, \ldots, \mathbf{S}^1, \mathbf{q})]$$

**Distribution shift 问题**：训练时 $\pi_{\mathrm{ref}}$ 是 $\pi_{\theta_0}$，但训练后 $\pi_\theta \ne \pi_{\mathrm{ref}}$，所以测试时 sample from $\pi_\theta$ 会 OOD。这正是 [Kumar et al. 2024 SCoRe](https://arxiv.org/abs/2409.12917) Figure 19 显示的：model 越训练越能改 reference 的错，但越不能改自己的错。

### 6.3 RL² → E-RL²

**RL²** ([Duan et al. 2016](https://arxiv.org/abs/1611.02779))：episodic meta-RL，recurrent policy 在 K 个 episodes 上累积 reward：

$$\max_{\pi_\theta} \mathbb{E}_{\mathbf{q}} \mathbb{E}_{\mathbf{S}^j \sim \pi_\theta(\cdot|\mathbf{S}^{j-1}, \ldots, \mathbf{S}^1, \mathbf{q})}\left[\sum_{j=1}^K r(\mathbf{S}^j, \mathbf{q})\right]$$

但 RL² 会 **collapse 到 greedy**——不探索。

**E-RL²** ([Stadie et al. 2019](https://arxiv.org/abs/1803.01118))：只奖励最后一 episode：

$$\max_{\pi_\theta} \mathbb{E}_{\mathbf{q}} \mathbb{E}_{\mathbf{S}^j \sim \pi_\theta(\cdot|\mathbf{S}^{j-1}, \ldots)}[r(\mathbf{S}^K, \mathbf{q})]$$

让前 $K-1$ episodes 是 "free exploration"，最后 1 个 episode 是 "evaluation"。这避免了 greedy collapse。

### 6.4 实证：Gehring et al. 2024 RLEF (Meta-RL in code)

[Gehring et al. 2024 RLEF](https://arxiv.org/abs/2410.02089)：

- Code generation 多轮 + compiler feedback
- 前 N-1 轮 free exploration（public test cases）
- 最后 1 轮 evaluation（hidden private test cases）→ reward

Table 2：8B LLaMa Instruct
- Few-shot: 8.9% → 10.5% (test)
- SFT: 10.3% → 10.0%
- RLEF: 17.2% → 16.0%

70B:
- Few-shot: 25.9% → 27.5%
- SFT: 22.5% → 20.3%
- RLEF: 37.5% → **40.1%**

> SFT 几乎不增（甚至降），RLEF 显著提升。这跟 [Kumar et al. 2024](https://arxiv.org/abs/2409.12917) 的发现一致：**SFT 不能 induce in-context exploration，RL 才能**。

### 6.5 关键证据：random feedback 仍 work

RLEF 实验（Figure 22 左）：即便把 compiler feedback 换成 **unrelated problem 的 output**，RL 后性能仍继续提升！说明 model 在 in-context 学到了 **meta-reasoning / exploration strategy** 而不是单纯依赖 external verifier。这跟 POMDP epistemic uncertainty 的 framing 完全一致。

### 6.6 但"super-intelligence"还没出现

Section 6.3 诚实地承认：当前所有证据表明 internalized search **显著提升 efficiency**（fewer tokens per correct answer），但 **没有强证据**表明 model 能 discover fundamentally new reasoning algorithm 解 symbolic search 解不了的问题（Countdown 中 1% → 4% 是 weak evidence）。

> "Compute-accuracy curve 是 shift left 还是 shift up?" 目前 public evidence 主要支持 **shift left**（efficiency）。

---

## 7. Putting It All Together: The Training Pipeline

### 7.1 Stage 1: Instruction tuning with linearized search

数据集 $\mathcal{D}_{\mathrm{train}} = \{(\mathbf{q}, \mathbf{Z}, \mathbf{S})\}$：Meta-CoT $\mathbf{Z} = \mathbf{z}_1, \ldots, \mathbf{z}_K$ + final solution $\mathbf{S}$。

多个训练目标 (Appendix C)：

**(1) Full procedural cloning** (Yang et al. 2022):
$$\mathcal{L}(\theta) = \min_\theta -\mathbb{E}\left[\sum_{i=1}^{|\mathbf{Z}|} \log \pi_\theta(\mathbf{z}_{i+1}|\mathbf{Z}_i, \mathbf{q}) + \sum_{i=1}^{|\mathbf{S}|} \log \pi_\theta(\mathbf{s}_{i+1}|\mathbf{S}_i, \mathbf{Z}, \mathbf{q})\right]$$

**(2) Meta-CoT only**:
$$\mathcal{L}(\theta) = -\mathbb{E}\left[\sum_{i=1}^{|\mathbf{Z}|} \log \pi_\theta(\mathbf{z}_{i+1}|\mathbf{Z}_i, \mathbf{q})\right]$$
solution 单独用一个 summarization model 生成。

**(3) Mask incorrect branches**:
$$\mathcal{L}(\theta) = -\mathbb{E}\left[\sum_{i=1}^{|\mathbf{Z}|} \mathbb{I}\{\mathbf{z}_{i+1} \in \mathbf{S}\} \log \pi_\theta(\mathbf{z}_{i+1}|\mathbf{Z}_i, \mathbf{q})\right]$$
只 train 正确路径上的 step（[Zhang et al. 2024b](https://arxiv.org/abs/2409.14586) 的做法）。

### 7.2 Stage 2: RL post-training (E-RL²)

主目标 + KL constraint：

$$\max_\theta \mathbb{E}_{\mathbf{S}, \mathbf{Z} \sim \pi_\theta(\cdot|\mathbf{q}), \mathbf{q}}\left[r^*(\mathbf{S}, \mathbf{q}) - \beta \sum_t \mathbb{D}_{KL}[\pi_\theta(\mathbf{z}_{t+1}|\mathbf{Z}_t, \mathbf{q}) \| \pi_{\mathrm{ref}}(\mathbf{z}_{t+1}|\mathbf{Z}_t, \mathbf{q})]\right]$$

- $r^*$: verifiable reward（final answer match）
- $\pi_{\mathrm{ref}}$: instruction-tuned model
- $\beta$: KL strength

替代优化方法：
- MCTS distillation ([Feng et al. 2024](https://arxiv.org/abs/2404.08849))
- Step-level DPO ([Xie et al. 2024](https://arxiv.org/abs/2405.00451), [Setlur et al. 2024a](https://arxiv.org/abs/2406.14532), [Lai et al. 2024 Step-DPO](https://arxiv.org/abs/2406.18629))
- Branching RLOO / VinePPO ([Havrilla et al. 2024](https://arxiv.org/abs/2403.04642), [Kazemnejad et al. 2024](https://arxiv.org/abs/2410.01679))

### 7.3 Q* / q-STaR: 不需要 verifier 的 RL

把 Meta-CoT $\mathbf{Z}$ 当 latent variable，做 ELBO：

$$\log \pi_{\mathrm{data}}(\mathbf{S}|\mathbf{q}) = \log \int \pi(\mathbf{S}|\mathbf{Z}, \mathbf{q})\pi(\mathbf{Z}|\mathbf{q})d\mathbf{Z} \geq \max_{q(\mathbf{Z}|\mathbf{q})} \mathbb{E}_{q(\mathbf{Z}|\mathbf{q})}[\log \pi(\mathbf{S}|\mathbf{Z}, \mathbf{q})] + \mathbb{D}_{KL}[q(\mathbf{Z}|\mathbf{q}) \| \pi(\mathbf{Z}|\mathbf{q})]$$

把 $\pi(\mathbf{Z}|\mathbf{q})$ 设为 prior $\pi_{\mathrm{ref}}$，$q(\mathbf{Z}|\mathbf{q})$ 和 $\pi(\mathbf{S}|\mathbf{Z}, \mathbf{q})$ 都 amortize 进同一个 LLM $\theta$：

$$\max_\theta \mathbb{E}_{\mathbf{Z} \sim \pi_\theta(\cdot|\mathbf{q}), \mathbf{S}, \mathbf{q}}\left[\log \pi_\theta(\mathbf{S}|\mathbf{Z}, \mathbf{q}) - \beta \mathbb{D}_{KL}[\pi_\theta(\mathbf{Z}|\mathbf{q}) \| \pi_{\mathrm{ref}}(\mathbf{Z}|\mathbf{q})]\right]$$

（β-VAE 形式，[Higgins et al. 2017](https://openreview.net/forum?id=Sy2fzU9gl)）

但因为 token 是 discrete，没法 reparameterization，只能 RL 优化。算式分解为：

$$\max_\theta \mathbb{E}_{\mathbf{Z} \sim \pi_\theta(\cdot|\mathbf{q}), \mathbf{S}, \mathbf{q}}[\mathrm{sg}(\log \pi_\theta(\mathbf{S}|\mathbf{Z}, \mathbf{q}))] - \beta \mathbb{D}_{KL}[\ldots] + \max_\theta \mathbb{E}_{\mathbf{Z} \sim \mathrm{sg}(\pi_\theta(\cdot|\mathbf{q}))}[\log \pi_\theta(\mathbf{S}|\mathbf{Z}, \mathbf{q})]$$

- 第一项是标准 RL，reward = $\log \pi_\theta(\mathbf{S}|\mathbf{Z}, \mathbf{q})$
- 第二项是 SFT：让 summarization model $\pi_\theta(\mathbf{S}|\mathbf{Z}, \mathbf{q})$ 拟合 ground-truth solution
- $\mathrm{sg}$: stop gradient

**直觉**：让 model 通过 latent Meta-CoT 让 $\pi(\mathbf{S}|\mathbf{Z}, \mathbf{q})$ 比 $\pi(\mathbf{S}|\mathbf{q})$ 更 expressive——把 transformer 计算图在 token 维度上"加深"（类似 [Quiet-STaR](https://arxiv.org/abs/2403.09629) 但在 semantic level 而不是 token level）。

[Qu et al. 2024](https://arxiv.org/abs/2407.18219) Figure 26 左：训练 loss 显示 in-context exploration 的 revision model 远低于直接 predict correct answer——支持这个 hypothesis。

### 7.4 Discount rates: 控制 overthinking

加 discount：

$$\max_\theta \mathbb{E}\left[\gamma^{|\mathbf{Z}|} r^*(\mathbf{S}, \mathbf{q}) - \beta \sum_t \mathbb{D}_{KL}[\ldots]\right]$$

$\gamma < 1$。Figure 27 显示：undiscounted 持续提升，discounted 呈 hump-shape（trade-off between certainty 和 compute）。

更进一步，训练时用 **distribution over discount rates** $\gamma_1 < \ldots < \gamma_m$，每个 $\gamma_i$ 配 prompt $\mathbf{p}_{\gamma_i}$（如 "work as fast as possible" 对应高 $\gamma$），让 inference 时通过 prompt 控制 compute-accuracy trade-off：

$$\max_\theta \mathbb{E}_{i \sim 1:m}\left[\gamma_i^{|\mathbf{Z}|} r^*(\mathbf{S}, \mathbf{q}) - \beta \sum_t \mathbb{D}_{KL}[\pi_\theta(\mathbf{z}_{t+1}|\mathbf{Z}_t, \mathbf{q}, \mathbf{p}_{\gamma_i}) \| \pi_{\mathrm{ref}}(\ldots)]\right]$$

这跟 [Schultz et al. 2024](https://arxiv.org/abs/2412.12119) MCTS-conditioned model 的可控行为一致。

---

## 8. Prompting 不能 induce meta-reasoning (Section 6.4.1)

非常有意思的负面结果：5 种 prompting strategy 测试（Baseline, CoT, Think, Think 3-shot, Think & Verify）：

- Baseline / CoT: regret 表达 < 0.5%
- Think: Llama 3.1 70B 达 12.65%
- Think & Verify: Llama 3.1 70B 达 25.67%
- 但 **GPT-4o / Claude 3.5 Sonnet 只有 1-4%**——说明这些 model "learn 了" 不表达 regret 而非"learn 了" meta-reason

**关键观察**：当 model 表达 regret / backtrack 时，**最终答案更可能是错的**。Token 数随难度增加，但 accuracy 没相应提升。Model 可能是 **"fake mistakes to match in-context style"** ([Gudibande et al. 2023](https://arxiv.org/abs/2305.15717))。

> **Insight**：用 prompt 模仿 o1 的行为 ≠ 真正的 System 2 reasoning。需要 weight-level 改动（SFT on search traces + RL）。

---

## 9. Big MATH Project

### 9.1 数据来源

| Source | Original | Base Filter | Strict Filter |
|---|---|---|---|
| HARP | 4,780 | 3,691 | 2,996 |
| NuminaMath | 859,608 | 452,820 | 231,887 |
| Omni-MATH | 4,428 | 3,660 | 2,478 |
| OpenMathInstruct-2 | 607,324 | 600,191 | 496,331 |
| **Total** | **1,476,140** | **1,060,362** | **733,692** |

### 9.2 三个准则

1. **Uniquely verifiable solutions**: 单一正确答案
2. **Open-ended**: 不能猜（排除 multi-choice）
3. **Closed-form**: 答案能用标量/公式表达（排除 proof）→ 才能自动 grade

### 9.3 数据清洗

- Asymptote figure 移除
- `\boxed{}` 提取
- 移除 hyperlink
- FastText language detection（保 English）
- MATH500 decontamination
- SemDeDup ([Abbas et al. 2023](https://arxiv.org/abs/2303.09540)) cosine similarity > 0.5 移除
- Strict filter: 移除 multi-part / proof / yes-no / true-false

---

## 10. Infrastructure

GPT-NeoX 上的异步 RLHF：
- CUDA IPC handles 共享 GPU memory
- Training 和 inference 直接共享权重指针
- 训练更新立即对 inference 可见
- 比 3-step async 训练 throughput 高 40%（Figure 28）
- 缺点：tensor parallelism 配置次优

---

## 11. Open Research Questions (Section 8.3)

1. **Open-ended verification & CoT faithfulness**：如何 reward 整条 CoT 而不只 final answer？O1 在 official math 例子就有 unfaithful CoT（未证 $h(x)=x^2-c$ 的 coefficient 为 0 就直接用）。Generative verifier ([Zhang et al. 2024a](https://arxiv.org/abs/2408.15240), [Mahan et al. 2024](https://arxiv.org/abs/2410.12832)) + RLAIF 是 promising 方向。

2. **Verifier gap & PRM scaling law**：Figure 29 显示 PRM 训练也服从 log-linear scaling。Generative verifier (CoT) + 多数投票还能进一步提升。

3. **Scaling laws for reasoning and search**：
   - Jones 2021 的 scaling law 没在 LLM reasoning 上系统验证过
   - 不同 search strategy (BFS, DFS, A*, MCTS) 对比
   - Instruction tuning vs RL 的 trade-off
   - In-context search 是 shift-left 还是 shift-up

4. **Meta-Search / Search²**：在已经有 in-context search 的 model 上再加外部 search（PRM conditioned on prior exploration）。[Anonymous 2024](https://openreview.net/forum?id=hJ2BCYGvFg) 的 multi-turn PRM Figure 30 已经显示改进。

5. **Tool-augmented reasoning (TIR)**：Figure 31 显示用 100K 数据训的 TIR 比 400K 数据训的 CoT 还强，特别是 low-sample regime。Offloading 计算到 Python interpreter 等 external tool。

---

## 12. Building the Big Picture (我的综合 intuition)

让我把整个 paper 的逻辑链画成一个 mental model：

```
[Problem] LLM 解 hard reasoning 时 CoT 不够
      ↓
[Root cause] Textbook solution ≠ DGP
      DGP = latent search process (MCTS / A* / revision)
      ↓
[Hypothesis] Generator-verifier gap 让 search 有用
      ↓
[Empirical evidence]
   - LLaMa 8B + Numina MATH: pass@64 ~85%
   - Lehnert maze: search-aug 显著好, gap 随 maze size 增
   - SoS Countdown: log-linear inference scaling
   - o1 在 HARP: token length 随 difficulty 增长
      ↓
[How to internalize]
   1. SFT on linearized search traces (Meta-STaR)
      - MCTS trace (with backtracking)
      - A* trace (cleaner)
      - Revision trace (Best-of-N linearized)
   2. RL post-training (E-RL²)
      - 前 K-1 episodes free exploration
      - 最后 episode = evaluation
      - KL to instruction-tuned model
      ↓
[Why RL necessary]
   - SFT 不能 induce in-context exploration (Gehring RLEF Table 2)
   - Distribution shift (Kumar SCoRe Figure 19)
   - POMDP epistemic uncertainty → standard RLHF 可以 arbitrarily bad
      ↓
[Current state]
   - Efficiency gains: clear ✓
   - Super-intelligence (novel algorithm discovery): weak evidence ✗
   - Compute-accuracy curve mostly shifted left, not up
      ↓
[Open]
   - CoT faithfulness
   - Verifier scaling laws
   - Search² (meta-search on top of internalized search)
   - Tool integration (TIR)
```

---

## 13. 与现有 framework 的连接 (跨论文联想)

### 13.1 AlphaZero 类比

论文反复用 AlphaZero ([Silver et al. 2018](https://www.science.org/doi/10.1126/science.aar6404)) 类比：
- Policy network + Value network + MCTS 联合训练
- Self-play 产生 search traces → 训练 policy
- Value function 指导 search

LLM 版本：
- Policy = LLM
- Value = PRM
- Search = in-context (linearized) MCTS

### 13.2 Quiet-STaR / Universal Transformer

[Zelikman et al. 2024 Quiet-STaR](https://arxiv.org/abs/2403.09629)：在 token level 引入 latent thought。
[Dehghani et al. 2019 Universal Transformer](https://arxiv.org/abs/1807.03819)：深度 adaptive。

论文说 Quiet-STaR 是 token-level latent，而 Meta-CoT 是 semantic-level latent——更"meaningful"。

### 13.3 LATS / RAP / ToT

- [Yao ToT](https://arxiv.org/abs/2305.10601): tree-structured search with LLM
- [Hao RAP](https://arxiv.org/abs/2305.14992): MCTS + world model
- [Zhou LATS](https://arxiv.org/abs/2310.04406): MCTS + self-reflection

这些都是 **inference-time explicit search**。Meta-CoT 的创新是 **internalize 这些 search 进 weight**，让单一 LLM 在 inference 时自回归地"run" search。

### 13.4 STaR 家族

- [Zelikman STaR 2022](https://arxiv.org/abs/2203.11365)
- [Singh Beyond Human Data 2024](https://arxiv.org/abs/2312.06585): restem / self-training
- [Yuan 2023](https://arxiv.org/abs/2308.01825): scaling relationship
- [Zelikman Quiet-STaR 2024](https://arxiv.org/abs/2403.09629)
- **Meta-STaR (本文)**：把 search trace 当 rationale

### 13.5 RL for LLM reasoning

- [Kumar SCoRe 2024](https://arxiv.org/abs/2409.12917): 两阶段 RL for self-correction
- [Gehring RLEF 2024](https://arxiv.org/abs/2410.02089): code 多轮 RL
- [Shao DeepSeekMath 2024](https://arxiv.org/abs/2402.03300): GRPO
- [Havrilla teaching-LM-reason 2024](https://arxiv.org/abs/2403.04642): VinePPO
- [Kazemnejad VinePPO 2024](https://arxiv.org/abs/2410.01679): refined credit assignment

### 13.6 PRM 家族

- [Lightman PRM800K 2023](https://arxiv.org/abs/2305.20050): human annotation
- [Wang Math-Shepherd 2024](https://arxiv.org/abs/2310.20647): MC rollout labels
- [Setlur 2024b/c](https://arxiv.org/abs/2410.08146): rewarding progress
- [Zhang Generative Verifier 2024a](https://arxiv.org/abs/2408.15240): verifier as next-token prediction
- [Mahan GRM 2024](https://arxiv.org/abs/2410.12832): generative reward model with CoT

### 13.7 Agent / Search 应用

- [Koh Tree Search for LM Agents 2024](https://arxiv.org/abs/2407.01476)
- [Putta Agent Q 2024](https://arxiv.org/abs/2408.07199): MCTS + DPO for web agent
- [Brown Large Language Monkeys 2024](https://arxiv.org/abs/2407.21787): repeated sampling scaling
- [Yu EXACT 2024](https://arxiv.org/abs/2410.02052): reflective-MCTS

### 13.8 复杂度理论

- [Merrill & Sabharwal 2023](https://arxiv.org/abs/2310.07923): transformer + CoT expressivity
- [Li et al. 2024](https://arxiv.org/abs/2402.12875): CoT empowers inherently serial problem solving
- [Nowak et al. 2024](https://arxiv.org/abs/2406.14197): representational capacity
- [Prystawski et al. 2024](https://arxiv.org/abs/2407.20311): why think step by step (locality of experience)

### 13.9 Meta-RL 经典

- [Duan RL² 2016](https://arxiv.org/abs/1611.02779)
- [Stadie E-RL² 2019](https://arxiv.org/abs/1803.01118)
- [Humplik 2019](https://arxiv.org/abs/1905.06424): meta-RL as task inference
- [Rakelly 2019](https://arxiv.org/abs/1903.08254): off-policy meta-RL
- [Beck 2024 survey](https://arxiv.org/abs/2301.08028)
- [Ghosh 2021](https://arxiv.org/abs/2107.06277): POMDP framing of generalization

---

## 14. 我对这个 paper 的几个 critiques

1. **Verifier gap 还没解决**：Big MATH 解决了 data 问题，但 PRM 仍远弱于 MC rollout。论文承认这是 fundamental gap，但 pipeline 里其实不强制需要 PRM（可以用 MC + outcome verifier 训练，再 amortize）。如果连 PRM 都不准，search 效率就大幅下降——这是整个 approach 的 key bottleneck。

2. **"Super-intelligence" 缺证据但被多次提及**：Section 6.3 诚实承认当前主要 gain 是 efficiency。但前面 motivation 多次提到 "discover novel reasoning algorithm"——这个 promise 还没有任何 strong evidence。

3. **Linearization 的表达力上限**：把 tree search 摊平成 linear token stream，是不是会损失 tree 的 parallelism？Section 8.3.4 (Search²) 正是针对这个：context length limit + sequential 慢。这跟 explicit tree search 的并行化优势有 fundamental tension。

4. **Meta-CoT 与 in-context RL² 的对应**：作者把 in-context exploration 视为 RL² 的 implicit memory，这很 elegant。但 RL² 在小 MDP 里能 emerge 是因为 task distribution 明确；LLM reasoning 的 "task distribution" 是数学题全体，跨度极大，能否真正 amortize 出有用 prior 还需验证。

5. **数据 contamination & HARP**：Figure 1 用 HARP 比较 o1 vs 其他 model。但 o1 训练时是否见过 HARP 类问题？Decontamination 在 Big MATH 里只针对 MATH500，没针对 HARP / Omni-MATH。这可能让 scaling comparison 略偏。

---

## 15. Final Takeaway

这篇 paper 给的 roadmap 是：**SFT on linearized MCTS/A* traces → E-RL² fine-tune → meta-RL over adaptation strategies**。它把 o1 / R1 / Gemini 2.0 Thinking / QwQ 这些 closed-source reasoning model 的 "secret sauce" 用一个 coherent theoretical frame 解释了：**它们都是在 weight 里 internalize 了 in-context search**。

最有冲击力的几个论证（按重要性排序）：

1. **Textbook solution ≠ DGP**（windmill 例子）—— 解释了为什么 SFT on $\mathbf{S}$ 学不会真正推理
2. **Lehnert maze + HARP o1 token scaling** 的同构——证明 complexity gap 是 fundamental 的，且 search-augment 能 close 它
3. **POMDP framing + Ghosh 2021 remark**——证明 standard RLHF 必然失败，需要 meta-RL (E-RL²)
4. **Random feedback 仍 work** (RLEF Figure 22)——证明 internalized exploration 不依赖 external verifier，是真正的 meta-reasoning

这大概是过去一年里把 "reasoning model" 这个 black box 解释得最透彻的一篇。结合 [Snell et al. 2024](https://arxiv.org/abs/2408.03314), [Kumar SCoRe](https://arxiv.org/abs/2409.12917), [Gandhi SoS](https://arxiv.org/abs/2404.03683), [Schultz 2024](https://arxiv.org/abs/2412.12119) 一起读，基本就是 open-source 复现 o1 的完整 playbook。
