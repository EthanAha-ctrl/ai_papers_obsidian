---
source_pdf: Language Agent Tree Search Unifies Reasoning, Acting, and Planning in
  Language Models.pdf
paper_sha256: 04c1e9cb00f384052739c53bcecb75626bf1f58f66c379cc289ea60ffc988635
processed_at: '2026-08-05T11:41:50-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# LATS 人话版

好，把那些公式和表格先扔一边，咱们用大白话重新过一遍。

---

## 这篇 paper 到底在干嘛

你给 GPT 一个任务，比如"帮我买个 3 盎司的 bright citrus 牌子 deodorant，价格低于 50 块"。GPT 通常怎么做？它就一路走到底：搜一下、点一下、看一眼、点 buy。如果中间走错了，它就死在那了，跟你下一盘棋走错一步就摔棋盘一样笨。

LATS 的意思是：**你能不能像下棋一样，在脑子里多推演几步、多试几条路、哪条不行就换一条？**

人下棋的时候不会只看眼前这一步，会想"我走这步对手可能怎么应对，我再怎么应对"——往后看几层，挑一条看起来最有戏的。LATS 就是把这个"往后看几层"的能力给 LM agent 装上。

---

## 核心比喻：在迷宫里找出口

想象你在迷宫里：

- **ReAct**：你走到一个岔路口，随便挑一条，走到死胡同就完蛋，重新从头来。
- **Reflexion**：你走到死胡同，写个小纸条"这条路不通，下次别走"，贴墙上，重新从头来，下次看到纸条就绕开。
- **ToT**：你把所有路都标成树，深度优先地一条一条试，试完一条回头试下一条。笨办法但能用。
- **LATS**：你走到岔路口先不急着走，而是**每个方向都先派个探子往前跑一段**，每个探子跑完回来报告"这条路看起来有戏/没戏"。然后你综合所有探子报告 + 哪些方向还没探过，决定自己真正走哪条。走到死路了，还写个反思纸条贴墙上，下次别的探子出发前先看纸条。

这个"派探子 + 听报告 + 写纸条"的循环，就是 LATS 的六大操作：selection（挑个路口站过去）、expansion（派 n 个探子）、evaluation（探子打分）、simulation（探子跑到终点）、backpropagation（把终点结果汇报回整条路径上每个 node）、reflection（死路时写纸条）。

---

## 为什么这个能 work？三个物理基础

### 1. LM 任务能"后悔"

AlphaGo 当年最难的事情之一是它需要一个"世界模型"——能预测"我下这步棋后棋盘会变成什么样"，这样才能在脑子里推演而不真的下一步。

但对 LM agent 来说，这事儿根本不是问题。你想"回到三步之前的状态"？**把当时的 prompt 原封不动复制粘贴一遍就行了**。WebShop、HotPotQA、HumanEval 这些 environment 全部支持这种"反悔"。你点错一个商品？重新开一个 session 从头搜就是了。你代码写错了？重新提交一份就是了。

Paper 里反复强调这一点——传统 MCTS 最贵的假设（要有 world model），在 LM 任务上**白送**。

### 2. LM 一个模型能当三个角色用

AlphaGo 需要专门训 policy network、value network、甚至单独的 rollout policy。LATS 一个 GPT-3.5 / GPT-4 全包了：

- 当 **policy**：给它当前 state，让它生成 n 个候选 action
- 当 **value function**：给它一段 trajectory，让它打分（"Thus the correctness score is 8"）
- 当 **reflection generator**：给它失败的 trajectory，让它写"哪里错了下次怎么改"

靠的就是 in-context learning，零训练。

### 3. 外部环境给真信号

这点是 LATS 跟 ToT / RAP 这类纯 reasoning search 最大的区别。

ToT 让 LM 自己评估自己走得对不对，但 LM 没有外部 ground truth 时会自信地错（Huang et al. 2024 已经证明 LM 不能纯靠自己 self-correct reasoning）。LATS 让 LM 走完一步后**先看环境的真实 feedback**（搜索结果、单元测试、购物车内容），再打分。

举个例子：写代码题，ToT 让 GPT 自己看自己写的代码对不对，LATS 让 GPT 先跑单元测试，看哪几个 assert 挂了，再判断这段代码怎么样。这俩信号质量天差地别。

---

## 那个混合 value function 是怎么回事

paper 里这个公式：

$$V(s) = \lambda \cdot \text{LM}(s) + (1-\lambda) \cdot \text{SC}(s)$$

人话翻译：一个 node 的分数 = **LM 自己打的分** 和 **多次采样一致性** 加权平均。

为什么要混这两个？

- **LM 打分**：直接问 GPT "这条路径走得对吗？1 到 10 打个分"。优点是会 reasoning，缺点是会自信地错。
- **Self-consistency**：在同一个 state 让 GPT 多次 sample action，看几次答案一不一致。如果五次里四次都说"应该搜 Wikipedia 的 'Arthur's Magazine'"，那这个 action 比五次给五个不同答案的更靠谱。

这两个 signal 互补：LM 评分有 reasoning 能力但可能 hallucinate；SC 没啥 reasoning 但通过"投票"反映 confidence。

不同任务用不同权重：
- HotPotQA、Game of 24：λ=0.5（一半一半）
- 编程题、WebShop：λ=0.8（更信 LM 自评）

为什么编程/WebShop 更信 LM 自评？我猜是因为这些任务的 LM 自评比较 grounded——代码可以看逻辑、购物可以看属性匹配，而 HotPotQA 的 LM 自评容易"自我感觉良好"。

---

## Reflection 那个纸条到底有多大用

paper 自己承认 reflection 边际效用不算大——去掉 reflection HotPotQA 只降 0.05 个点（0.63 → 0.58）。

但有意思的是 Reflexion（单独的 reflection 方法）相对 ReAct 提升 0.19 个点。这说明什么？

**Search 本身已经吸收了 reflection 的大部分功能**。如果走错一条路，MCTS 通过 UCT 公式会让那条路的 V(s) 变低，下次自然不去。Reflection 等于是在这个 scalar signal 之外额外加个 language 信号说"这条路具体错在搜索词太宽泛"。

但 search 跟 reflection 不是 1+1=2，更多是 1+0.2=1.2。Search 已经把"绕开失败路径"的事做了一大半，reflection 锦上添花。

---

## 那几个实验结果人话版

### HotPotQA（多跳问答）

- ReAct 0.32，LATS 0.63——**几乎翻倍**。
- 更牛的是 LATS(CoT+ReAct) 0.71——先用脑子想，想不出来再去查 Wikipedia。这跟人解题一模一样：先回忆，回忆不起再翻书。

### HumanEval（写代码）

- GPT-3.5 裸跑 46.9
- GPT-3.5 + LATS = 83.8
- GPT-4 裸跑 80.1
- GPT-4 + LATS = 92.7

**GPT-3.5 加上 LATS 干过了 GPT-4 裸跑**。这就是 inference-time compute scaling 的 power——用算法让弱模型在推理时多想几步，弥补能力差距。o1 的思路跟这个同源，只不过 o1 把 search 藏进 model 里训了，LATS 是 model 外面套 search。

### WebShop（网购）

- LATS 75.9，超过了需要 gradient 的 fine-tuning 67.5
- 但 success rate（全 attribute 命中）38% < fine-tuning 45%

也就是说 LATS 平均能买到很接近的货，但严格匹配所有条件时 fine-tuning 仍强。这是 prompt-based 方法的通病——soft reward 上能打，hard constraint 上差点。

---

## 跟 RAP 到底怎么区分

很多人 confuse 这俩，因为都用 MCTS。简单说：

- **RAP**: LM 必须假装自己知道"下一步世界会变成什么样"——LM 当 world model。在纯 reasoning 任务上 OK（因为下一步 thought 就是 LM 自己生成的），但在 WebShop 这种真环境上没法用——你没法让 LM 凭空生成一个 product page 的 HTML。
- **LATS**: 完全 model-free。LM 只生成 action 和打分，state transition 由真环境给。这是 RL 里 model-based vs model-free 的经典区分，跟 Dreamer vs Q-learning 的关系一样。

---

## 这篇 paper 在大图上的位置

LATS 是 2023 年 10 月的工作，那时候 test-time scaling 还没成为主流叙事。回头看，它是这条线上的标志性节点：

1. **System-2 thinking 的早期实现**：Daniel Kahneman 的双系统理论——System 1 是 fast intuition（CoT/ReAct），System 2 是 slow deliberate planning（LATS）。LATS 显式把 System 2 做成 search loop。
2. **LM-as-everything 范式**：policy、value、reward、reflection generator 全部由同一个 LM 通过 prompt 切换角色完成，不训任何 extra network。这是 in-context learning 极致化的一个例子。
3. **Search-based test-time compute**：n（branching factor）和 k（rollout 数）是两个 scaling axis。Figure 3 显示 HumanEval 上 LATS 随 iteration 单调上升而 Reflexion 早 plateau——iterative refinement 有 ceiling，tree search 没有（至少在 budget 内）。
4. **弱模型靠算法打强模型**：GPT-3.5 + LATS > GPT-4 裸跑，这个 pattern 后来在 o1 上重演（o1 用 RL 训 hidden reasoning，但 idea 同源）。

---

## 几个我会进一步想的问题

1. **Reflection 能不能完全替代 backprop？** Reflection 给 language-level gradient，backprop 给 scalar mean。理论上 language signal 信息密度高得多。LATS 两者都用，但能不能纯靠 reflection 不做 scalar backprop 还保持性能？这值得 ablation。
2. **Value function 的 calibration 问题**：LM 打分是整数 1-10，这个 discretization 在不同 task 上的效应 paper 没研究。如果 LM 系统性高估或低估，UCT 的 exploration term 会被 bias。
3. **n 和 k 的 scaling law**：Figure 3 只在 HumanEval 上做了 k。如果在多个 task 上做 n∈{1,3,5,10,20}, k∈{8,16,32,64,128} 的 grid，能不能拟合出类似 Chinchilla 那样的 search budget scaling law？这是 test-time compute 这条线的核心问题。
4. **Reflection 的质量 bottleneck**：WebShop 实验 paper 提到 reflection 经常 generic 不 useful。能不能用更强的 LM（GPT-4）专门做 reflection generator，而 action 还用 GPT-3.5？这是 inference-time 的"分工"思路。
5. **Irreversible environment 怎么办**：LATS 假设能 revert 到任意历史 state。但真实机器人不能反悔打碎的杯子，real-world agent 怎么办？能不能用 LM 当 world model 来"虚拟 revert"，相当于 RAP+LATS 混合？

---

## 一句话总结

LATS 把 MCTS 那套"往后看几步、挑最有戏的、失败了反思"的 deliberate thinking 显式搬到 LM agent 上，靠 LM 任务能 copy-paste 回到任意 state 这个白送的便利，把 policy、value、reflection 全交给同一个 LM 用 prompt 切换角色完成，零训练。结果 GPT-3.5 加 LATS 能打过 GPT-4 裸跑，证明 inference-time search 是弥补模型能力差距的 cheap 升级路径——这是后来 o1 系列 test-time compute scaling 的早期信号。

---

## References

- LATS paper: https://arxiv.org/abs/2310.04406
- Code: https://github.com/lapisrocks/LanguageAgentTreeSearch
- ReAct: https://arxiv.org/abs/2210.03629
- ToT: https://arxiv.org/abs/2305.10601
- RAP: https://arxiv.org/abs/2305.14992
- Reflexion: https://arxiv.org/abs/2303.11366
- Self-Consistency: https://arxiv.org/abs/2203.11171
- Huang et al. "LMs cannot self-correct reasoning yet": https://arxiv.org/abs/2310.01798
- AlphaGo: https://www.nature.com/articles/nature16961
- Kahneman System 1/2 (referenced in paper as Sloman 1996, Evans 2010)

---

# LATS: Language Agent Tree Search 深度解析

非常好的一篇 paper，Andy。这是 Shunyu Yao 的 ReAct、ToT 系列工作的延伸 —— 由 UIUC 的 Yu-Xiong Wang 团队（Andy Zhou, Kai Yan 等）做的。核心 motivation 是把 MCTS 这个在 AlphaGo 里被证明 powerful 的算法，彻底 port 到 LM agent 这个 regime 上，让 reasoning + acting + planning 三个 axis 真正 unify。

---

## 1. Why LATS? Why now?

### 1.1 现有方法的根本局限

| Method | Reasoning | Acting | Planning | Self-Reflection | External Memory |
|---|---|---|---|---|---|
| CoT | ✓ | ✗ | ✗ | ✗ | ✗ |
| ReAct | ✓ | ✓ | ✗ | ✗ | ✗ |
| ToT | ✓ | ✗ | ✓ | ✓ | ✓ |
| RAP | ✓ | ✗ | ✓ | ✗ | ✓ |
| Reflexion | ✓ | ✓ | ✗ | ✓ | ✓ |
| **LATS** | ✓ | ✓ | ✓ | ✓ | ✓ |

这张表（Table 1）已经把 paper 的 contribution 压缩到一行了。Karpathy 你应该一眼就看出问题：所有 prior method 都缺一个 axis。

- **CoT (Wei et al., 2022)**: autoregressive 串行生成 thought z_1, ..., z_l，error 会 compound，且 sampling 一次失败就死了。
- **ReAct (Yao et al., 2023b)**: Thought-Action-Observation 三段循环，但是 greedy 的 single trajectory，遇到 dead end 只能放弃重试。
- **ToT (Yao et al., 2023a)**: 引入 search，但用 DFS/BFS，没 exploration-exploitation tradeoff；且没 external observation。
- **RAP (Hao et al., 2023)**: 也用 MCTS，但需要 LM 当 world model 预测 state，这对很多 environment 不现实。

LATS 的关键 insight 是两条：
1. **LM tasks 的 reversion property**：传统 MCTS 在 RL 上最痛苦的点是必须有 environment model 才能 expand from arbitrary state（参见 Dreamer 系列的 world model 学习成本）。但对 LM agent，"回到任意历史 state" 等价于 "把 history context 复制粘贴回 prompt"。这个 property 在 paper Section 3.2 末尾被反复强调，是整个 framework 成立的物理基础。
2. **不需要训 value function**：传统 MCTS 需要 value network 和 policy network（AlphaGo 那一套）。LATS 直接用 in-context learning 把 LM 自身 prompt 成 value function + policy + reflection generator。

### 1.2 与 RAP 的关键区别

很多人会 confuse RAP 和 LATS，但它们本质不同：

- **RAP**: LM 必须扮演 world model，预测 next state s_{t+1} = f(s_t, a_t)。这对 HotPotQA 这种 reasoning 任务 OK，因为 LM 可以"自己想下一步"，但对 WebShop 这种真实 environment 不可能（你不能让 LM 凭空生成一个 product page 的 HTML）。
- **LATS**: Model-free，state transition 由真实 environment 提供，LM 只需要做 policy（生成 action）和 value（评估 state）。这是 RL 里 model-based vs model-free 的经典区分。

---

## 2. Method: LATS 的 6 个 Operation

参考 paper Figure 2 的 pipeline diagram。整个 search 是 k 个 iteration 的循环，每个 iteration 走完六个 operation：

```
[Selection] -> [Expansion] -> [Evaluation] -> [Simulation] -> [Backpropagation]
                                                              |
                                                       (if failed)
                                                              |
                                                              v
                                                          [Reflection]
```

### 2.1 State Representation

这是设计上的一个关键决定：

$$s = [x, a_{1..i}, o_{1..i}]$$

- **x**: original input（task description）
- **a_{1..i}**: action sequence 到时间步 i
- **o_{1..i}**: observation sequence 到时间步 i

每个 node 就是 partial trajectory，包括 thought + action + observation 的完整序列。这意味着 node 已经"携带了所有 history"，符合 LM 的 autoregressive nature。这也是为什么 reversion 简单 —— 切到不同 node = 切到不同 context window。

### 2.2 Selection: UCT 公式

$$UCT(s) = V(s) + w \sqrt{\frac{\ln N(p)}{N(s)}}$$

变量解释：
- **s**: 当前要评估的 child node
- **p**: s 的 parent node
- **V(s)**: s 这个 node 的 value（expected return from subtree rooted at s）
- **N(s)**: s 被 visit 的次数
- **N(p)**: p 被 visit 的次数
- **w**: exploration weight（paper 默认 w=1，ablation 在 Table 11 测了 w=0.5 给 0.55，w=2.0 给 0.63，w=1.0 给 0.63 in HotPotQA）

**Intuition**：第一项 V(s) 是 exploitation —— value 高的 node 应该多 visit；第二项是 exploration —— visit 少的 node（N(s) 小）且 parent 被 visit 多次（ln N(p) 大）的 node 应该被探索。当 N(s)→0 时第二项趋于无穷，保证每个 node 至少被 visit 一次。

w=2.0 不进一步提升性能但 convergence 更快，这点有意思 —— 说明在这类 LM task 上，exploration 的边际收益不大，可能因为 LM 的 action distribution 本身已经足够 diverse 了。

### 2.3 Expansion: 多采样候选

从当前 selected node 出发，sample n 个 action：

$$a_t^{(i)} \sim p_\theta(s_t), \quad i = 1, \ldots, n$$

paper 默认 n=5。这是 LATS 跟 ReAct 最根本的区别 —— ReAct 每次 greedy 取一个，LATS 每次 expand 一组 candidates，把"哪个最好"的决定推到 value function + UCT 上。

Action space 是混合的：

$$\hat{A} = A \cup Z$$

- **A**: 环境 action（比如 HotPotQA 里的 Search[entity], Lookup[keyword], Finish[answer]；WebShop 里的 click、search）
- **Z**: thought（内部 reasoning trace）

让 thought 和 action 共享同一个 action space，是 ReAct 的核心设计，LATS 继承了这点。

### 2.4 Evaluation: Hybrid Value Function（核心贡献）

这是 paper 最 smart 的设计：

$$V(s) = \lambda \cdot \text{LM}(s) + (1 - \lambda) \cdot \text{SC}(s)$$

变量：
- **LM(s)**: LM 自己 prompt 出来的 scalar score（prompt LM "evaluate this trajectory, end with 'correctness score is s' where s ∈ [1,10]"）
- **SC(s)**: self-consistency score —— 在同一个 state 多次 sample actions，看它们 agreement 程度
- **λ**: tradeoff hyperparameter

不同任务的 λ 设置：
- HotPotQA / Game of 24: **λ = 0.5**
- Programming (HumanEval, MBPP) / WebShop: **λ = 0.8**

**为什么这个 hybrid 重要？**

LM self-evaluation 有 known issue —— Huang et al. (2024) 的 "Large language models cannot self-correct reasoning yet" 已经证明 LM 纯靠 self-correct 内部 reasoning 不可靠。所以单纯 LM(s) 不够。

Self-consistency (Wang et al., 2022) 的 insight 是：如果多次 sample 在同一 state 给出类似答案，那么这个答案更可能正确。SC(s) 是把这个 idea 用作 value signal —— 多次 sample 的 agreement 作为 confidence proxy。

**Ablation 数据**（Table 8, 11）：
- 完全去掉 LM heuristic（只用 SC 和环境 reward）：HotPotQA 从 0.63 掉到 0.37，下降 0.26！这是最大的 ablation 效应，说明 LM self-evaluation 仍是最重要 signal。
- Game of 24 上 λ=1（去掉 SC）从 0.44 掉到 0.40（Table 13），说明 SC 提供 4 个点的边际收益。

**与 ToT value function 的区别**：ToT 也是 LM-as-judge，但 LATS 在拿到 environment observation 之后才做 evaluation，这是关键差异 —— paper Section 4.2 反复强调这点。ToT 的 LM judge 在 "blind" 状态评估，LATS 的 LM judge 看了环境反馈再评估。

### 2.5 Simulation: 走到 terminal

从当前 selected node 继续 expand 直到 terminal state（比如 HotPotQA 的 Finish[answer]，WebShop 的 Buy Now）。Simulation 阶段在每个 depth 都 sample 并按 value 选 highest 的 node 继续。

Terminal 时环境给 scalar reward r：
- HotPotQA: exact match (oracle setup)
- HumanEval: 单元测试通过率
- WebShop: 0~1 之间的 attribute match score

### 2.6 Backpropagation

terminal 后，沿 trajectory 把 reward 反向传播回每个 ancestor：

$$N(s_i) = N(s_i) + 1$$
$$V(s_i) = \frac{V(s_i) \cdot (N(s_i) - 1) + r}{N(s_i)}$$

变量：
- **s_i**: trajectory 上第 i 个 node
- **N(s_i)**: 该 node 被 visit 次数
- **V(s_i)**: 更新后的 value（mean of rewards seen through this node）
- **r**: terminal reward

这是 running mean update，每次新 trajectory 后 value 是 historical mean。Backpropagation 完成后这些新 V(s) 会进入下一 iteration 的 UCT 计算。

### 2.7 Reflection: Semantic Gradient

失败时（reward 不达 success threshold）：

$$\text{reflection} = p_{\text{ref}}(s_t)$$

把 (trajectory, final reward) 喂给 LM，让它 verbalize "哪里错了 + 下次怎么做"。Reflection 存到 external memory，下次 expansion / evaluation 时作为 context 加进去。

**Key insight（paper 用了 "semantic gradient" 这个词）**：scalar reward (r=0 or r=1) 信息量太少，reflection 给出 language-level 的 gradient signal，相当于把 RL 里 dense reward shaping 的活儿交给 LM 自己 verbal 完成。

但 paper 也承认 Reflection 的边际效用有限 —— Table 8 显示去掉 reflection 仅降 0.05（0.63→0.58），而 Reflexion vs ReAct 是 0.19 的差距。这说明 search 本身已经覆盖了 reflection 的部分功能（"如果走错了，下次绕开" search 算法天然就做）。

---

## 3. 完整 Pseudocode 解析

参考 Appendix A 的 Algorithm 1。让我把核心 loop 拆开：

```
for k = 0, ..., K-1:                    # K 个 rollouts
    for t = 0, ..., L-1:                # 每个 rollout 最多 L 步
        if s_t not terminal:
            for i = 1, ..., n:           # Expansion: 采 n 个 candidates
                a_t^(i) ~ p_θ(s_t)
                o_t^(i) = env(a_t^(i))
                s_{t+1}^(i) = (c_t, o_t^(i), a_t^(i))
                V_t^(i) = λ * p_V(s_t^(i)) + (1-λ) * SC(s_t^(i))    # Evaluation
                V(s_t) = V_t^(i)
                add s_t^(i) to children
        
        if s_t terminal:                # Reflection on failure
            r = env reward
            if r != success:
                reflection = p_ref(c_t)
                c ← reflection
        
        a_t = argmax [V(s_t) + w*sqrt(ln N(s_t)/N(s_{t+1}))]    # Selection via UCT
        N(s_{t+1}) += 1
        
        if a_t is output action: break
    
    T = actual steps
    for t = T-1, ..., 0:                # Backprop
        V(s_t) = (V(s_t)*(N(s_t)-1) + r) / N(s_t)
```

---

## 4. Experiment 结果：Build Intuition

### 4.1 HotPotQA（Table 2, 3）

| Prompt Method | HotpotQA EM ↑ |
|---|---|
| ReAct | 0.32 |
| ReAct (best of k) | 0.38 |
| Reflexion | 0.51 |
| ToT(ReAct) | 0.39 |
| RAP(ReAct) | 0.54 |
| **LATS(ReAct)** | **0.63** |
| LATS(n=3) | 0.58 |
| LATS(n=10) | 0.65 |
| **LATS(CoT+ReAct)** | **0.71** |

**Intuition**：
1. 简单把 ToT 加上 ReAct（即 ToT(ReAct)）甚至比纯 ToT 还差（0.39 vs 0.55 reasoning-only）。这说明 search 算法从 reasoning 到 acting 的简单 port 是 non-trivial 的。paper Section 1 强调这是 motivation 之一 —— 简单组合是 inadequate 的。
2. RAP(ReAct) 比 RAP reasoning-only 也低（0.54 vs 0.60），同样的现象。
3. LATS(ReAct) 反超 RAP 0.09 个点，证明 paper 的 MCTS 适配（特别是 hybrid value function + reflection）是必要的。
4. **LATS(CoT+ReAct) 0.71**: 先用 CoT 内部 reasoning 试，失败再切到 ReAct external retrieval。这是 paper 里最 human-like 的设计 —— 人也是先 recall 自己知道的，不行再去查 Wikipedia。

### 4.2 Programming（Table 4 HumanEval）

| Prompt Method | Model | Pass@1 ↑ |
|---|---|---|
| CoT | GPT-3.5 | 46.9 |
| ReAct | GPT-3.5 | 56.9 |
| Reflexion | GPT-3.5 | 68.1 |
| ToT | GPT-3.5 | 54.4 |
| RAP | GPT-3.5 | 63.1 |
| **LATS(ReAct)** | GPT-3.5 | **83.8** |
| Base | GPT-4 | 80.1 |
| Reflexion | GPT-4 | 91.0 |
| **LATS(ReAct)** | GPT-4 | **92.7** ← SOTA |

**Key intuition**：GPT-3.5 + LATS (83.8) > GPT-4 base (80.1)！Inference-time search 让弱模型打败强模型。这是 test-time scaling 的早期 evidence，跟 OpenAI 的 o1 series 思路一致。

Programming task 的特殊设计：每个 action 是一个完整 solution（不是 increment），所以跳过 simulation 步，直接用 unit test pass rate 作 backprop reward。

### 4.3 WebShop（Table 6）

| Method | Score ↑ | SR ↑ |
|---|---|---|
| ReAct | 53.8 | 28.0 |
| ReAct(best of k) | 59.1 | 32.0 |
| Reflexion | 64.2 | 35.0 |
| **LATS(ReAct)** | **75.9** | 38.0 |
| IL (imitation learning) | 59.9 | 29.1 |
| IL+RL | 62.4 | 28.7 |
| Fine-tuning | 67.5 | 45.0 |
| Expert (human) | 82.1 | 59.6 |

**Insight**：LATS 是 gradient-free 方法，性能 75.9 > Fine-tuning 的 67.5。这是 paper 一个很强的 claim —— 不更新 weight，纯靠 inference-time search + reflection，就超越需要 gradient 的方法。

但 SR 上 LATS (38%) < Fine-tuning (45%)，说明在硬约束（必须所有 attribute 满足）的任务上，gradient-based 仍有优势。这个 trade-off 值得注意。

### 4.4 Cost Analysis（Table 9, 10）

| Method | Performance | Sample complexity | Token consumption |
|---|---|---|---|
| ReAct (k=250) | 0.42 | O(k) | - |
| ToT (ReAct, n=5, k=50) | 0.49 | O(kn) | 210,215 |
| RAP (ReAct, n=5, k=50) | 0.54 | O(kn) | 176,500 |
| **LATS (n=5, k=50)** | **0.63** | O(kn) | **173,290** |

**这是 LATS 最 actionable 的数据**：相同 asymptotic complexity O(kn)，LATS 用更少 token 获得更高 performance。原因有二：
1. Reflection 让失败 trajectory 信息被吸收，下次不重复犯错。
2. UCT 平衡 exploration 后，不会在 low-value branch 浪费 budget。

Table 10 显示成功 trajectory 的平均 nodes 数：k=50 时 LATS=66.65 < RAP=70.60 < ToT=84.05。

---

## 5. 与你的直觉 Karpathy：这个 framework 在大图上意味着什么？

我猜测你感兴趣的是 LATS 在 inference-time scaling law 中的位置。LATS 是 2023 年 10 月的工作，那时 test-time compute scaling 还没成为主流叙事（o1 是 2024 年 9 月）。LATS 本质上是：

1. **把 System-2 thinking 显式做成 search**：Daniel Kahneman 的 System 1 / System 2 框架（paper 引用 Sloman 1996, Evans 2010）。System 1 = autoregressive fast thinking = CoT/ReAct；System 2 = deliberate planning = LATS。这点 paper Section 6 的 future work 也明说了 "System-2 LM approaches"。

2. **Search-based test-time compute scaling**：n（branching factor）和 k（rollout 数）是两个 scaling axis。Paper Figure 3 显示 HumanEval 上 LATS 随 iteration 单调上升，而 Reflexion plateau。这是 search-based scaling 跟 iterative-refinement scaling 的本质区别。

3. **Model-free + LM-as-everything**：LATS 完全不需要训任何东西（policy、value、world model、reward model 全部 in-context），这跟 OpenAI o1 的"训 hidden CoT"路径不同。LATS 是更 pure 的 symbolic search + LM heuristic 路线，类似 Symbolic AlphaGo 但把每个 module 替换为 LM。

4. **Reflection 作为 cheap RL signal**：这点跟 Self-Rewarding Language Models (Yuan et al., 2024) 和 Constitutional AI 的思路一脉相承 —— 把 scalar reward 升级为 language feedback。

### 5.1 Limitations

Paper Section 6 + Appendix B 列了两个：
1. **Computational cost**: 比 ReAct 贵 ~3-5 倍（按 nodes 数算）。
2. **Environment reversion assumption**: 必须能回到任意历史 state。这对 irreversible environment（比如 real-world robot 不会"反悔"已经打碎的杯子）不成立。但 LM-based environment（API、code、web）大多支持。

### 5.2 我会进一步思考的问题

1. **Can reflection replace backpropagation?** Reflection 给的是 language gradient，backprop 给的是 scalar mean update。LATS 两者都用。但理论上 reflection 信息密度更高，能否完全替代 scalar backprop？
2. **Value function 的 calibration**：LM self-score 是 [1, 10] 整数。这个 discretization 在不同 task 上 effect 如何？paper 没研究。
3. **n 和 k 的 scaling law**：Figure 3 只在 HumanEval 上做。如果在更多 task 上做 n∈{1,3,5,10,20}, k∈{8,16,32,64} 的 grid，能看出 search budget 的 scaling 性质吗？

---

## 6. References

- **LATS paper**: [arxiv.org/abs/2310.04406](https://arxiv.org/abs/2310.04406)
- **Code**: [github.com/lapisrocks/LanguageAgentTreeSearch](https://github.com/lapisrocks/LanguageAgentTreeSearch)
- **ReAct (predecessor)**: [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- **ToT**: [arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601)
- **RAP**: [arxiv.org/abs/2305.14992](https://arxiv.org/abs/2305.14992)
- **Reflexion**: [arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)
- **Self-Consistency**: [arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)
- **MCTS survey (Świechowski et al., 2021)**: [link.springer.com/article/10.1007/s10462-022-10385-y](https://link.springer.com/article/10.1007/s10462-022-10385-y)
- **AlphaGo (Silver et al., 2016)**: [nature.com/articles/nature16961](https://www.nature.com/articles/nature16961)
- **"LMs cannot self-correct reasoning yet" (Huang et al., 2024)**: [arxiv.org/abs/2310.01798](https://arxiv.org/abs/2310.01798)
- **WebShop**: [arxiv.org/abs/2207.01206](https://arxiv.org/abs/2207.01206)
- **HumanEval**: [arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374)
- **HotPotQA**: [arxiv.org/abs/1809.09600](https://arxiv.org/abs/1809.09600)

---

## 7. One-paragraph TL;DR

LATS 把 MCTS 完整 port 到 LM agent 上：每个 node 是 partial trajectory（含 history），policy = LM 在 state 上 sample n 个 actions，value = LM self-evaluation 和 self-consistency 的 λ 加权和，reward = environment 反馈，failure 时 LM 生成 reflection 作为额外 context。整个 search 是 model-free 的（不学 world model），完全靠 in-context learning 驱动，并依赖 LM tasks 的 reversion property（copy-paste history 即可回到任意 state）。在 HotPotQA (EM 0.71)、HumanEval (Pass@1 92.7 SOTA with GPT-4)、WebShop (75.9 score, 超越 fine-tuning) 上证明 unified reasoning + acting + planning 的有效性。本质是 System-2 deliberate thinking 在 LM 上的早期 instantiation，是 test-time compute scaling 这条 line of work 的标志性工作之一。
