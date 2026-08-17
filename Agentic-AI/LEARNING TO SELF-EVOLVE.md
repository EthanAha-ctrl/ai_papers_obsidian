---
source_pdf: LEARNING TO SELF-EVOLVE.pdf
paper_sha256: 5ea3d92d73d1d5b094d625b39f964a5fb64db8b730c66a192bda5693e6db2bd1
processed_at: '2026-08-05T13:59:59-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 LSE

Andrej 你问用人话讲，那我就抛开 paper 的学术腔，直接讲 intuition。

---

## 这篇 paper 在干什么

一句话：**部署后的模型是死的。能不能让它从已经做过的问题里学点东西，用在下一批问题上？**

比如一个 SQL 生成模型，刚做完 10 个 Financial 数据库的题，做错了 4 个。它应该从这 4 个错误里提炼出"哦这个 schema 里 OwnerUserId 和 LastEditorUserId 容易混淆"这种 knowledge，写进自己的 instruction，下次做这个 database 的题就别再错。

这件事 current LLM 本身也能做一点（paper 里 Qwen3-4B untrained 就能从 seed 57.2% 涨到 62.2%），但是从来没有人**专门训练**模型做这件事。所有人都是指望模型靠 pretraining + post-training 的 emergent ability 隐式搞定。

LSE 的核心 claim：**self-evolution 是个独立的 skill，值得专门用 RL 训练**。训完之后 4B 模型打败 GPT-5 当 self-evolving policy。

---

## 为什么 self-evolution 不是"会 reasoning 就自然会"

你可能会想：reasoning 强的模型自然会分析 feedback，自然能改 prompt。paper 说不对，这件事本质是个 RL 问题，要求模型同时做三件事：

1. **Credit assignment**: 当前 context 里哪部分有用哪部分有害？
2. **Anticipation**: 这个改动会怎么影响 downstream behavior？
3. **Exploration vs exploitation**: 继续精炼现有有效的部分，还是试新的方向？

RL optimizer 有专门算法做这三件事（value function、policy gradient、UCB 之类的）。让 LLM 用 natural language reasoning 隐式做，等于让它假装自己是个 RL 算法。所以应该 explicit 训练。

---

## 怎么训：Single-step Bandit + Improvement Reward

### 为什么不直接训 multi-step

原问题形式上是 $T$ 步 trajectory：$c_0 \to c_1 \to \dots \to c_T$，目标是累积 reward。但 trajectory-level 有两个麻烦：

- **贵**：每次 rollout 要跑 $T$ 轮 evaluate + edit
- **Credit assignment 难**：第 3 round 的 edit 对第 7 round 的 performance 贡献多少？

LSE 的选择：**只训单步**。给 $(c_0, \text{performance summary})$，输出 $c_1$，立刻给 reward。多步的组合交给 test time 的 tree search。

这是 AlphaGo 式的分解 — policy network 学单步 move quality，MCTS 在 test time 做 multi-step planning。

### Reward 的关键设计

naive 想法：reward = 改完之后的 accuracy $\bar{R}(c_1)$。

问题：这个 reward 偏向"本来就好"的 context。paper 给的例子很直观：

- Scenario A：初始 80% → 改完 70%（退化了 10%）
- Scenario B：初始 30% → 改完 60%（进步了 30%）

如果 reward = post-edit accuracy，A 得 0.7，B 得 0.6，A 赢。但 A 其实变差了，B 大幅进步。这个 reward 会鼓励模型 preserve 已经好的 context，不去真正 learn to improve。

LSE 的 fix：**reward = improvement**

$$r_{\text{LSE}} = \bar{R}(c_1) - \bar{R}(c_0)$$

直接 incentivize marginal gain。

### 数学上的微妙之处（这里最 elegant）

理论上，如果你用 learned baseline 的 policy gradient，improvement reward 和 post-edit reward 给出**完全相同的 gradient**。证明很短：

设 state $s = (c_0, S_0)$。Post-edit reward 下的 value：

$$V(s) = \mathbb{E}[\bar{R}(c_1) \mid s]$$

Improvement reward 下的 value：

$$V'(s) = \mathbb{E}[\bar{R}(c_1) - \bar{R}(c_0) \mid s] = V(s) - \bar{R}(c_0)$$

Advantage：

$$A'(s, c_1) = (\bar{R}(c_1) - \bar{R}(c_0)) - (V(s) - \bar{R}(c_0)) = \bar{R}(c_1) - V(s)$$

跟 post-edit 的 advantage 完全一样。$\bar{R}(c_0)$ 被 baseline 吸收了。

**那为什么还要用 improvement reward？**

因为 LSE **不用 learned baseline**。它直接拿 $\bar{R}(c_0)$ 当 baseline —— 这个值在模型 act 之前就已知（act 前先在 holdout set 上 evaluate 一次）。这样：

$$A_{\text{LSE}} = \bar{R}(c_1) - \bar{R}(c_0)$$

- 不需要 value network
- 不需要 GRPO 的 group normalization（每个 prompt 多次 rollout 算 mean/std）
- $\bar{R}(c_0)$ 是个 **cross-prompt control variate**：抵消掉 prompt-specific 的 difficulty offset

这点是 paper 最 actionable 的 finding。GRPO 的 group normalization 只能在 **同一个 prompt 的多次 rollout 之间** cancel noise，不能 cancel **不同 prompt 之间的 difficulty 差异**。而 $\bar{R}(c_0)$ 直接是同一个 prompt 的 pre-edit score，perfectly aligned 的 baseline。

实验证据：improvement reward 比 GRPO post-edit reward **+4.3%** on BIRD（67.3% vs 63.0%）。虽然理论等价，实践差距很大，因为 GRPO 的 baseline 估计有 noise，而 LSE 的 baseline 是精确已知的。

---

## Test time 怎么用：Tree Search

### 线性进化的失败模式

naive 做法 $c_0 \to c_1 \to c_2 \to \dots$，每 round 在最新 context 上扩展。问题：一旦做了坏 edit，后面所有 round 都基于坏 context，无法回头。

paper 里 Figure 3 给了个惨烈例子：BIRD Card Games 上，linear chain 在 round 3 之后 accuracy 从 56% 跌到 <30%，永远爬不回来。

### Evolution Tree + UCB1

维护一棵 tree，每个 node 存。每 round 用 UCB1 公式选扩展哪个 node：

$$n^* = \arg\max_{n} \left[\bar{R}_n + C\sqrt{\frac{\ln N}{v_n}}\right]$$

- $\bar{R}_n$：exploit term，node 的 holdout reward
- $C\sqrt{\ln N / v_n}$：explore term，访问少的 node 有 bonus
- $N$：总 round 数，$v_n$：node $n$ 被访问次数

这是经典 multi-armed bandit UCB1，跟 MCTS selection phase 一样。

**Intuition**：坏 edit 不会 cascade。UCB 会自动把流量切回高分 ancestor，从那里重新 branch。相当于 implicit backtrack。

Ablation 数据：tree search 比 linear chain +2.4% on BIRD, +2.2% on MMLU。

---

## 一些细节值得注意

### Holdout set 的必要性

每 round 只看 10 个 problem 的 batch，performance 估计噪声大。所以固定一个 holdout set（50 个 problem），每个 context 都在 holdout 上 evaluate（8 generations 平均）作为 $\bar{R}(c)$。

这个 holdout 是固定的，跨 round 跨 method 都一样，保证 apple-to-apple 比较。

### Training data 怎么构造

如果训练时 $c_0$ 总是 seed context，会有 train-test mismatch —— test time 模型要面对自己之前 edit 过的 context。

解决：先跑 200 次 data-generation runs × 20 rounds ≈ 4000 tree nodes，训练时随机 sample node 作为 starting context。

还有个 **curriculum trick**：早期训练时 preferentially sample "improvement potential" 高的 node，定义为：

$$\text{improvement potential}(n) = \max_{n' \in \text{tree}(n)} \bar{R}_{n'} - \bar{R}_n$$

就是 node 当前 performance 和它所在 tree 的 max performance 的 gap。gap 大的 node 给 policy 更多 headroom 学习。早期 random sampling 信号太弱，这个 curriculum 帮 warm-up。

### 训练细节里的 surprise

- 用 verl 框架
- LR 1e-5，32 nodes/batch，4 rollouts/node
- On-policy，**no KL regularization**

no KL 这点比较 surprising。一般 RLHF 都要 KL penalty 防止 drift 太远。但这里是 single-step contextual bandit，没有 multi-step trajectory 的 reward hacking 风险，可能所以不需要。也可能是因为 action space 是 text，drift 的 "distance" 本来就比 parameter space 难定义。

---

## 实验结果的人话解读

### Table 1: BIRD（SQL 生成）

| Method | Avg. |
|---|---|
| Seed prompt | 57.2 |
| Qwen3-4B untrained | 62.2 |
| Claude Sonnet 4.5 | 64.5 |
| GPT-5 | 65.2 |
| GEPA | 62.8 |
| TextGrad | 63.1 |
| **LSE (4B)** | **67.3** |

关键 takeaways：
1. Untrained 4B 已经 +5% over seed，证明 LLM 本来就有 prompt refinement 能力
2. LSE 再 +5.1%，证明 explicit training 有额外价值
3. LSE 4B 超 GPT-5 +2.1%，超 Claude +2.8%
4. GEPA / TextGrad 这种专门 prompt optimizer 也打不过 LSE

### Table 2: MMLU-Redux（QA）

LSE 73.3%，match GPT-5（72.5%），超 Claude +1.3%。

但这里 self-evolution 的整体 gain 比 SQL 小（+3.6% vs +5% over seed）。paper 给的解释很 intuitive：

- **SQL 同 database 内 problem 共享 schema、join pattern、column semantics**，跨 problem knowledge transfer 强
- **MMLU 同 subject 内 deduplicated 且 broad**，solving 一道 econometrics 题不保证下一道有用

**环境结构决定 self-evolution 上限**。这是 design self-evolving system 时要考虑的 first-order factor。如果你的 task domain 内部 problem 之间没有 shared structure，再强的 self-evolving policy 也榨不出多少 juice。

### Transfer 实验（Table 3）

Action policy 换成 Arctic-Text2SQL-R1-7B（专门 RL-tuned 的 SQL 模型），self-evolving policy 还是用 LSE-trained Qwen3-4B，**无额外训练**：

- Seed: 57.7%
- +LSE: 64.4%（+6.7%）

两个结论：
1. **Parameter-level 和 prompt-level 优化互补**：RL training 把通用 SQL pattern 编进 weights，prompt evolution 在 test time 适应具体 database
2. **LSE policy 可 transfer across action models**：虽然只用 Qwen3-4B 训练，evolution strategy 能 guide 不同 action model

---

## LSE 发现的 instruction 长什么样

这个很重要，能验证模型真的在 extract transferable knowledge 而非表面改写。

### BIRD Codebase 上的 instruction 包含：

- SQLite 语法约束：用 `strftime('%Y', column)` 提取年份，**never `YEAR()`**（SQLite 不支持）
- Domain-specific pitfalls：Misidentifying OwnerUserId vs LastEditorUserId，误 join on UserId instead of Id，拼写错 CreaionDate（漏 t）
- Subquery 规则：`WHERE Age = (SELECT MIN(Age) FROM users)` 而非 `ORDER BY Age LIMIT 1`
- 百分比计算：`CAST(... AS REAL)` 防 integer division
- Conditional output：`CASE WHEN` / `IIF` map NULL / non-NULL

### MMLU Anatomy 上的 instruction 包含大量 medical knowledge：

- Fertilization 在 fallopian tube 不在 ovary/uterus
- Upper motor neuron lesion → spastic paralysis; lower → flaccid
- Horner's syndrome: miosis, facial vasodilation, decreased lacrimation, anhydrosis
- Internal capsule lesion → contralateral spastic paralysis
- Palatine shelf elevation 是 turgor pressure from hydrophilic molecules，不直接由 tongue descent 引起

这些完全不是 generic "think step by step"，是 task-specific actionable heuristic + domain fact。说明 self-evolving policy 真的从 experience 里 extract 了 transferable knowledge。

---

## Limitations（paper 自己承认的）

1. **Single-step 简化**：把 multi-step exploration delegation 给 tree search。Joint multi-step trajectory optimization 可能更强，但 credit assignment 难
2. **Per-domain training**：每个 domain 训单独 policy。跨 domain 泛化 likely 需要大规模 multi-domain training
3. **只 edit instruction field**：tools / skill library / external memory 没涉及
4. **可结合 latent / parameter space update**：比如 TTT (Sun 2020) 或 end-to-end TTT (Tandon 2025)
5. **Environment scale 小**：environment curation 是 open problem —— 需要 problem 有 feedback 且 problem 之间共享足够 structure

---

## 给你的最终 Intuition

1. **Self-evolution 是 distinct skill**，不是 reasoning 强就自然会。需要专门训练。

2. **Single-step bandit 足够**。Training 学 single edit quality，test time 用 tree search 组合。AlphaGo 式分解。

3. **Improvement reward $\bar{R}(c_1) - \bar{R}(c_0)$ 数学等价 post-edit reward（用 learned baseline 时），但实践 +4.3%**。因为 $\bar{R}(c_0)$ 是精确已知的 control variate，cancel cross-prompt difficulty offset。GRPO 的 group normalization 只能 within-prompt cancel。这是 paper 最 subtle 的 finding。

4. **Tree search vs linear chain: +2.4%**。坏 edit 不 cascade，UCB 自动 backtrack。

5. **环境结构决定上限**：SQL 内部 transfer 强，QA 内部 transfer 弱。

6. **Parameter-level 和 prompt-level 优化互补**：Arctic-Text2SQL-R1 + LSE = +6.7%。

7. **Curriculum trick**：早期 sample high improvement-potential node。

---

## 相关联想（你可能感兴趣的延伸）

这个 paper 让我想到几个方向：

**1. 这个 framework 跟 STaR / Absolute Zero 的关系**

STaR [Zelikman 2022] 是 training-time bootstrap rationale。Absolute Zero [Zhao 2025] 是 single model propose + solve + code executor reward，zero data。这些都是 training-time self-evolution，policy 训完就 freeze。

LSE 是 test-time 版本，policy 不 freeze，继续从 deployment experience 学。两者其实可以 stack：先用 Absolute Zero 训一个强 base policy，再用 LSE 训它的 self-evolving policy。

**2. Improvement reward 的 control variate 视角**

这个 trick 其实是 REINFORCE with baseline 的特例。Baseline 选 null action 的 reward 是非常自然的选择。在一般 RL 里，null action 不一定有意义；但在 self-evolution 里，"不做 edit" 就是 null action，reward 就是 $\bar{R}(c_0)$，perfectly defined。

这个 insight 可以 generalize 到其他 "improvement-style" task：任何有 "before" 和 "after" state 的 edit task，都可以用 before-state reward 作 baseline。

**3. Tree search 跟 MCTS 的关系**

LSE 的 tree search 本质是简化版 MCTS，只有 selection（UCB1）和 expansion，没有 simulation（rollout to terminal）。因为这里每个 node 的 value $\bar{R}$ 可以直接在 holdout set 上 evaluate，不需要 rollout。

这跟 AlphaGo 的 policy + value network + MCTS 的结构很像，只是 value 是 evaluated 不是 predicted。如果训一个 value network 预测 context quality，可以省掉 holdout evaluation 的 cost，做更深的 search。

**4. Self-evolving policy 的 generalization**

paper 每个 domain 训一个 policy。但 instruction 的结构其实跨 domain 有共性 —— 比如"分析问题类型"、"识别 common pitfalls"、"注意 output format"这种 meta-heuristic。

如果在大规模 multi-domain data 上训，可能能学到一个 universal self-evolving policy，跨 domain transfer。这有点像 meta-learning 的味道 —— 学一个 "how to learn from feedback" 的 meta-skill。

**5. 跟 TTT (Test-Time Training) 的关系**

TTT [Sun 2020] 是 gradient-based test-time adaptation，在 input 上做 self-supervised loss update parameters。LSE 是 prompt-based，不改 parameters。

两者可以结合：用 LSE 做 coarse-grained context adaptation，用 TTT 做 fine-grained parameter adaptation。或者用 LSE 生成的 context 作为 TTT 的 self-supervised task 来源。

**6. Reward hacking 的风险**

paper 没讨论这个，但 improvement reward 理论上有 reward hacking 风险：model 可能学会生成 "看起来不同但实际等价" 的 context，让 $\bar{R}(c_1)$ 在 holdout 上偶然高一点。不过 single-step + small holdout set 可能限制了这个风险。如果 scale up，可能需要 anti-reward-hacking 机制，比如 context similarity penalty 或 adversarial holdout。

---

希望这个人话版帮你看清楚核心 idea。最值得消化的三件事：

1. Self-evolution 是 distinct skill，要 explicit train
2. Single-step + tree search 的 AlphaGo 式分解
3. Improvement reward 数学等价但实践更强，因为 $\bar{R}(c_0)$ 是精确 baseline

这几个 idea 都很 portable 到其他 self-improvement system 设计上。

---

## Web References

- LSE paper (待 arXiv 发布): 作者 Xiaoyin Chen (Mila), Canwen Xu, Yuxiong He (Snowflake)
- UCB1 (Auer 2002): https://jmlr.org/papers/v3/auer02a.html
- verl (HybridFlow): https://arxiv.org/abs/2409.19256
- AlphaGo: https://www.nature.com/articles/nature24270
- PPO: https://arxiv.org/abs/1707.06347
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- OpenAI o1: https://openai.com/index/learning-to-reason-with-llms/
- Let's Verify Step by Step: https://arxiv.org/abs/2305.20050
- STaR: https://arxiv.org/abs/2203.14465
- Absolute Zero: https://arxiv.org/abs/2505.03335
- TTRL: https://arxiv.org/abs/2504.16084
- TTT-Discover: https://arxiv.org/abs/2601.16175
- TTT original (Sun 2020): https://arxiv.org/abs/1911.01083
- End-to-end TTT (Tandon 2025): https://arxiv.org/abs/2512.23675
- Reflexion: https://arxiv.org/abs/2303.11366
- SCoRe: https://arxiv.org/abs/2409.12917
- GEPA: https://arxiv.org/abs/2507.19457
- TextGrad: https://arxiv.org/abs/2406.07496
- DSPy: https://arxiv.org/abs/2310.03714
- ExpeL: https://arxiv.org/abs/2308.10144
- PromptBreeder: https://arxiv.org/abs/2309.16797
- ADAS: https://arxiv.org/abs/2408.08435
- Darwin Godel Machine: https://arxiv.org/abs/2505.22954
- Godel Agent: https://arxiv.org/abs/2410.04444
- Voyager: https://arxiv.org/abs/2305.16291
- Mem0: https://doi.org/10.3233/FAIA251160
- Self-rewarding LMs: https://arxiv.org/abs/2401.10020
- BIRD: https://arxiv.org/abs/2405.14178
- MMLU-Redux: https://arxiv.org/abs/2406.04127
- SuperGPQA: https://arxiv.org/abs/2502.14739
- Arctic-Text2SQL-R1: https://arxiv.org/abs/2505.20315

---

# LEARNING TO SELF-EVOLVE (LSE) — 深度技术讲解

你好 Andrej。这篇 paper 来自 Mila + Snowflake (Xiaoyin Chen, Canwen Xu, Yuxiong He 等)，核心 idea 很 clean：把 test-time self-evolution 当作一个 **distinct learnable skill**，用 RL 显式训练一个小 LLM 去做"如何改写自己的 context"这件事，而不是依赖 inherent reasoning 的 emergent ability。4B 模型打败 GPT-5 / Claude Sonnet 4.5 当 self-evolving policy。下面我尽量 build your intuition。

---

## 1. Problem Framing: 为什么 self-evolution 是个 RL 问题

当前 LLM post-training pipeline（DeepSeek-R1 [1], OpenAI o1 [2], Let's Verify Step by Step [3]）确实让 model 在自己 generated data 上 RL，但是 training 一结束 policy 就 freeze。Deployment 时无论在同一 domain 解决多少 problem，experience 全部丢弃，context reset 后归零。这是 paper 想要 attack 的 gap。

Paper 把 test-time self-evolution 沿两个维度切：

**Dimension 1 — how**:
- **gradient-based**: 直接改 θ（test-time training, TTRL [4], TTT-Discover [5]）
- **prompt-based**: 改 context c，θ frozen

**Dimension 2 — when**:
- **intra-episode**: 单 problem 内 refine（Reflexion [6], SCoRe [7]）
- **inter-episode**: 跨 episode 积累经验用到新 problem（GEPA [8], TextGrad [9], ExpeL [10], Voyager [11], Darwin Godel Machine [12]）

LSE 选 **inter-episode + prompt-based** 这个 quadrant。为什么？gradient-based 有 catastrophic forgetting；intra-episode 不 transfer。

### 形式化

Task $\mathcal{T} = (\mathcal{X}, \mathcal{Y}, R)$:
- $\mathcal{X}$: input space（问题集合）
- $\mathcal{Y}$: output space
- $R: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$: reward function（如 SQL execution accuracy）

LLM policy 分解为 $\pi_\theta(y \mid x, c)$，其中 $\theta$ 是参数，$c$ 是 context（system prompt + instructions + skill library + ...）。

self-evolving policy $f$ 把当前 policy + 经验 tuple 映射到新 policy：

$$
\pi^{(t+1)} = f\left(\pi^{(t)}, \{(x_i, y_i, r_i)\}_{i=1}^k\right) \tag{1}
$$

- $t$: evolution round index（0, 1, ..., T）
- $k$: 每 round 采样的 problem 数（paper 中 k=10）
- $(x_i, y_i, r_i)$: 第 i 个 problem 的 input/output/reward

目标是 $T$ rounds 累积期望 reward 最大化：

$$
\sum_{t=0}^{T} \mathbb{E}_{x \sim \mathcal{X}}\left[R\left(x, \pi^{(t)}(x)\right)\right] \tag{2}
$$

**Intuition**: self-evolution 本质是个 RL problem — model 必须 implicit 地做 (a) credit assignment（context 哪部分有用）、(b) gradient estimation（怎么改）、(c) explore-exploit balance（refine vs try new）。RL optimizer 用专门算法做这些，LLM 用 natural language reasoning 隐式做。Paper 的核心论点：**这件事应该 explicit 训练**。

---

## 2. Tree-Guided Evolution（test-time 机制）

### 线性进化的失败模式

朴素做法是 linear chain $c_0 \to c_1 \to c_2 \to \cdots$，每 round 在最新 context 上贪心扩展。问题：一旦做了坏 edit，后续 round 都基于坏 context，无法 backtrack。Figure 3 直观展示 BIRD Card Games 上 linear chain accuracy 从 56% 跌到 <30% 永久无法恢复。

### Evolution Tree + UCB

维护 tree $\mathcal{G}$，每个 node $n$ 存 tuple $(c_n, S_n, \bar{R}_n, v_n)$:
- $c_n$: context
- $S_n = \{(x_i, y_i, y_i^*, r_i)\}_{i=1}^k$: structured performance summary（含 ground truth $y_i^*$ 和 correctness signal $r_i$）
- $\bar{R}_n$: mean holdout reward（在固定 holdout set 上算的）
- $v_n$: visit count

每 round 用 UCB1 [13] 选 node 扩展：

$$
n^* = \arg\max_{n \in \mathcal{G}} \underbrace{\bar{R}_n}_{\text{exploit}} + \underbrace{C\sqrt{\frac{\ln N}{v_n}}}_{\text{explore}} \tag{5}
$$

- $N$: 已完成的 round 总数
- $v_n$: node $n$ 被选作扩展父节点的次数
- $C > 0$: exploration constant，paper 没明说具体值
- 第二项：访问少的 node 有 bonus（类似 MCTS 的 UCT 公式）

这是经典 multi-armed bandit UCB1 [Auer 2002]，类似 AlphaGo MCTS selection phase [14]。

### Holdout reward 定义

为什么需要 holdout？因为 in-batch performance（10 个 problem）噪声大。固定 holdout set $D \subset \mathcal{X}$（paper 中 |D|=50）：

$$
\bar{R}(c) = \frac{1}{|D|} \sum_{x \in D} R(x, y), \quad y \sim \pi_\theta(\cdot \mid x, c) \tag{4}
$$

每个 $x$ 上 sample $y$（paper 中 8 generations 平均，降 variance）。

### Algorithm 1 完整流程

```
Init: tree G = {(c_0, ∅, R̄(c_0), 0)}
for t = 0 ... T-1:
    n* = argmax_n [R̄_n + C*sqrt(ln N / v_n)]   # UCB select
    sample {x_i}_{i=1}^k ~ X
    y_i ~ π_θ(· | x_i, c_{n*})                  # Act
    r_i = R(x_i, y_i)                            # Evaluate
    S_t = {(x_i, y_i, y_i*, r_i)}
    c_new = f_ψ(c_{n*}, S_t)                     # Evolve
    R̄(c_new) on holdout D
    append child (c_new, S_t, R̄(c_new), 0) to n*
    v_{n*} += 1
return argmax_n R̄_n
```

**Key intuition**: tree search 让 bad edit 不 cascade。UCB 会自动 backtrack 到高分 ancestor，相当于 implicit 的 explore-exploit across context space。

---

## 3. LSE Training Framework（核心贡献）

### Multi-step → Single-step 简化

原目标 Eq. (6):

$$
\max_{f_\psi} \sum_{t=0}^{T} \bar{R}(c_t), \quad c_{t+1} = f_\psi(c_t, S_t) \;\forall t \tag{6}
$$

- $\psi$: self-evolving policy 的参数

直接优化这个 $T$-step 目标有两个问题：
1. **Cost**: 每 rollout 要 $T$ 次 sequential evaluation + context generation
2. **Credit assignment**: trajectory-level reward，第 3 round edit 对第 7 round performance 贡献多少？

LSE 简化为 $T=1$ single-step contextual bandit：
- 输入 $(c_0, S_0)$，输出 $c_1 = f_\psi(c_0, S_0)$，立即 reward
- 把 multi-step exploration delegation 给 test-time tree search

**Intuition**: 这有点像 AlphaGo 的分解 — policy network 学 single move quality，MCTS 在 test time 做 multi-step planning。LSE 把"学如何 edit"和"如何在多 round 中组合 edit"分开。

### Reward 设计的关键 insight

**Naive candidate**: post-edit reward $\bar{R}(c_1)$

**Bias 问题**（paper §3.3 给的例子）：
- Scenario 1: 初始 80% → edit 后 70%，post-edit = 0.7
- Scenario 2: 初始 30% → edit 后 60%，post-edit = 0.6

post-edit reward 偏好 scenario 1（虽然退化 10%），因为混淆了 starting point 质量和 edit 质量。这鼓励 policy preserve already-effective context 而非真正 learn to improve。

### Improvement-based Reward

$$
r_{\text{LSE}} = \bar{R}(c_1) - \bar{R}(c_0) \tag{7}
$$

直接 incentivize 相对 starting point 的 marginal improvement。

### 数学 insight: baseline 吸收（关键 trick）

如果用 standard policy gradient (PPO [15] / GRPO [16])，learned baseline $V(s)$ 会吸收 $\bar{R}(c_0)$ 项。设 state $s = (c_0, S_0)$：

$$
V'(s) = \mathbb{E}[\bar{R}(c_1) - \bar{R}(c_0) \mid s] = V(s) - \bar{R}(c_0)
$$

其中 $V(s) = \mathbb{E}[\bar{R}(c_1) \mid s]$ 是 post-edit reward 下的 baseline。

Advantage:

$$
A'(s, c_1) = r_{\text{LSE}} - V'(s) = (\bar{R}(c_1) - \bar{R}(c_0)) - (V(s) - \bar{R}(c_0)) = \bar{R}(c_1) - V(s) \tag{8}
$$

**结论**: 用 learned baseline 时，delta-reward 和 post-edit reward 给出**完全相同**的 gradient estimate。

### LSE 的妙处: bypass baseline estimation

$\bar{R}(c_0)$ 在 $f_\psi$ act 之前就已知（act 之前先在 holdout 上 evaluate），等于 null edit（返回 $c_0$ 不变）的 reward，可以直接当 baseline，不需要 value network 也不需要 group-based normalization：

$$
A_{\text{LSE}} = \bar{R}(c_1) - \bar{R}(c_0) \tag{9}
$$

Policy gradient:

$$
\nabla_\psi J = \mathbb{E}_{c_1 \sim f_\psi(\cdot \mid c_0, S_0)}\left[A_{\text{LSE}} \nabla_\psi \log f_\psi(c_1 \mid c_0, S_0)\right] \tag{10}
$$

- $\bar{R}(c_0)$ action-independent → 作为 baseline 不改变 expected gradient
- 但作为 control variate cancel prompt-specific offsets
- 实践中 evaluation noise + between-prompt difficulty variation 主导 raw accuracy → improvement-based advantage 提供 cleaner learning signal
- **减少 cost**: 不需要 multiple rollouts per prompt for GRPO group normalization，也不需要 value network

**Intuition**: 这其实是 REINFORCE with baseline 的一个特例。Baseline 选 null edit 的 reward 是非常自然的选择，因为 "edit vs no-edit" 的对比正是我们想 incentivize 的。

### Training data 构造

如果 $c_0$ 总是 seed context，会 train-test mismatch：test time 多 round evolution，policy 要 improve 自己之前 edit 过的 context。

解决：populate tree $\mathcal{G}$ with 200 data-generation runs × 20 rounds ≈ 4000 tree nodes，每 RL step 从 $\mathcal{G}$ 随机 sample node 作为 starting context。

**Curriculum trick**: 早期 random sampling 信号弱，所以 preferentially sample "improvement potential" 高的 node = (node performance) - (max performance in its tree)。这偏向那些还没达到 tree 内最优的 node，给 policy 更多 headroom 学习。

### Implementation 细节

- RL framework: **verl** [Sheng 2024, https://arxiv.org/abs/2409.19256]
- Action policy $\pi_\theta$ = Qwen3-4B-Instruct
- Self-evolving policy $f_\psi$ = Qwen3-4B-Instruct（同一个模型，不同 role）
- LR: 1e-5
- 32 nodes/batch, 4 rollouts/node
- On-policy, **no KL regularization**（这点比较 surprising，可能因为 single-step contextual bandit 不需要 reference model 约束）
- 4 epochs, best checkpoint on dev set

---

## 4. 实验

### Setup
- **Text-to-SQL**: BIRD [Li 2024], 5 个 database domains (Financial, Toxicology, Codebase, Formula 1, Card Games)
- **QA**: SuperGPQA [Team 2025] train + MMLU-Redux [Gema 2024] eval, 10 subjects
- Holdout |D|=50, 8 generations averaged
- 25 evolution rounds, report best holdout performance
- 10 problems/batch, sampled with replacement
- Fixed random seed（所有方法看相同 problem 序列）

### Table 1: BIRD 结果

| Method | Financial | Toxicology | Codebase | Formula 1 | Card Games | Avg. |
|---|---|---|---|---|---|---|
| Seed prompt | 51.0 | 60.3 | 63.7 | 54.5 | 56.5 | 57.2 |
| Qwen3-4B-Instruct (untrained) | 63.7 | 60.3 | 70.2 | 56.0 | 61.0 | 62.2 |
| Claude Sonnet 4.5 | 70.8 | 63.8 | 67.8 | 57.3 | 63.0 | 64.5 |
| GPT-5 | 70.8 | 65.8 | 72.0 | 54.3 | 63.3 | 65.2 |
| GEPA | 64.0 | 62.0 | 72.0 | 54.0 | 62.0 | 62.8 |
| TextGrad | 60.3 | 66.0 | 71.5 | 56.5 | 61.3 | 63.1 |
| **LSE (ours)** | **72.0** | **68.5** | 72.0 | **59.8** | **64.0** | **67.3** |

LSE 4B 超 GPT-5 +2.1%, 超 Claude +2.8%。注意 untrained Qwen3-4B 已经能从 57.2% (seed) 到 62.2%（+5%），证明 LLM 本来就有 prompt refinement 能力，但 explicit training 又多榨 +5.1%。

### Table 2: MMLU-Redux 结果

| Method | Avg. |
|---|---|
| Seed prompt | 67.6 |
| Qwen3-4B | 71.2 |
| Claude Sonnet 4.5 | 72.0 |
| GPT-5 | 72.5 |
| GEPA | 73.0 |
| TextGrad | 69.1 |
| **LSE** | **73.3** |

LSE match GPT-5, 超 Claude +1.3%, 超 TextGrad +4.2%。

**Intuition**: QA 上 improvement 比 SQL 小（self-evolution over seed +3.6% vs +5%, LSE 额外 +2.1% vs +5.1%）。Paper 解释：SQL 同 domain 内 problem 共享 schema/join pattern/column semantics，跨 problem knowledge transfer 强；MMLU-Redux 同 subject 内 deduplicated 且 broad，solving 一题不保证下一题有用。**环境结构决定 self-evolution 上限**，这是重要的 insight。

### Ablation 1: Reward design (Figure 2a)

- $A_{\text{GRPO}}$ (post-edit + GRPO group normalization): **63.0%**
- $A_{\text{LSE}}$ (improvement-based): **67.3%**
- **+4.3%** improvement

虽然理论上 learned baseline 时 gradient 等价，但实践中 GRPO 的 group normalization 是 within-prompt 的，不能 cross-prompt cancel difficulty offset；LSE 的 $\bar{R}(c_0)$ 是 cross-prompt control variate。这就是 paper §3.3 说的 "evaluation noise and between-prompt difficulty variation likely dominate raw accuracy scores"。

### Ablation 2: Search strategy (Figure 2b, 3, 4)

- Linear chain: BIRD 59.8%, MMLU 69.0%
- Tree search (UCB): BIRD 62.2%, MMLU 71.2%
- **+2.4% / +2.2%**

Figure 3 是 visualization：linear chain 在 Card Games 上 round 3 之后 accuracy 从 56% 跌到 <30% 永久无法恢复，tree search 可以 backtrack 到高分 ancestor。

### Transfer experiment (Table 3)

Arctic-Text2SQL-R1-7B [Yao 2025, https://arxiv.org/abs/2505.20315] 作为 action policy（7B RL-tuned SQL 专用模型），LSE-trained Qwen3-4B 作为 $f_\psi$ **无额外训练**:

| Variant | Financial | Toxicology | Codebase | Formula 1 | Card Games | Avg. |
|---|---|---|---|---|---|---|
| Seed prompt | 56.8 | 54.5 | 65.3 | 52.3 | 59.5 | 57.7 |
| +LSE evolution | 68.3 | 62.3 | 71.5 | 57.0 | 63.0 | **64.4** |

**+6.7%**. 两个证据：
1. **Parameter-level vs prompt-level 互补**: RL training 把通用 SQL pattern 编进 weights，prompt evolution 在 test time 适应具体 database
2. **LSE policy 可 transfer**: $f_\psi$ 只用 Qwen3-4B 训练，能 guide 不同 action model

---

## 5. LSE 发现的 instruction 长什么样

Paper Appendix B.3 给了具体例子，BIRD Codebase 上 LSE 发现的 instruction 包含：

1. 强制 output format `<answer>...</answer>`
2. Schema 分析 checklist（target attribute, tables, joins, filters）
3. **Domain-specific pitfalls**: Misidentifying OwnerUserId vs LastEditorUserId, 误 join on UserId instead of Id, 拼写错 CreaionDate（漏 t）
4. **SQLite 语法约束**: 用 `strftime('%Y', column)` 提取年份，never `YEAR()`（SQLite 不支持）
5. **Conditional output**: `CASE WHEN` / `IIF` map NULL / non-NULL
6. **Subquery 规则**: `WHERE Age = (SELECT MIN(Age) FROM users)` 而非 `ORDER BY Age LIMIT 1`
7. **百分比计算**: `CAST(... AS REAL)` 防 integer division

Anatomy subject 上的 instruction 包含大量 medical knowledge：
- fertilization 在 fallopian tube 不在 ovary/uterus
- upper motor neuron lesion → spastic paralysis; lower → flaccid
- Horner's syndrome: miosis, facial vasodilation, decreased lacrimation, anhydrosis
- internal capsule lesion → contralateral spastic paralysis

**Intuition**: LSE 学到的不是 generic "think step by step"，而是 task-specific 的 actionable heuristic + domain fact。这正说明 self-evolving policy 真的在 extract transferable knowledge from experience，而非表面改写。

---

## 6. 与相关工作的 critical comparison

### Training-time self-evolution（不解决 deployment 静态问题）
- **STaR** [Zelikman 2022, https://arxiv.org/abs/2203.14465]: bootstrap rationale, fine-tune on correct ones
- **Self-rewarding LMs** [Yuan 2024, https://arxiv.org/abs/2401.10020]: model 自己作 reward
- **Absolute Zero** [Zhao 2025, https://arxiv.org/abs/2505.03335]: 单模型 propose + solve，code executor 作 verifiable reward，zero external data
- 共同点：training 后 policy 仍 static

### Test-time intra-episode（单 problem，不 transfer）
- **Reflexion** [Shinn 2023, https://arxiv.org/abs/2303.11366]: verbal RL, reflect + retry
- **SCoRe** [Kumar 2025, https://arxiv.org/abs/2409.12917]: RL 训练 self-correction
- **TTRL** [Zuo 2025, https://arxiv.org/abs/2504.16084]: test time RL with majority voting proxy reward
- **TTT-Discover** [Yuksekgonul 2026, https://arxiv.org/abs/2601.16175]: test time RL 找 open-ended solution

### Test-time inter-episode（LSE 的 quadrant）
- **Prompt optimization**:
  - **GEPA** [Agrawal 2025, https://arxiv.org/abs/2507.19457]: reflective prompt evolution + multi-objective evolutionary search, maintain Pareto front
  - **TextGrad** [Yuksekgonul 2024, https://arxiv.org/abs/2406.07496]: backward call 生成 textual "gradient" + Textual Gradient Descent (TGD) call rewrite
  - **DSPy** [Khattab 2024, https://arxiv.org/abs/2310.03714]: compile declarative LM calls
- **Self-referential**:
  - **ExpeL** [Zhao 2024, https://arxiv.org/abs/2308.10144]: extract transferable lessons from trajectories
  - **PromptBreeder** [Fernando 2024, https://arxiv.org/abs/2309.16797]: mutation + crossover operators
  - **ADAS** [Hu 2025, https://arxiv.org/abs/2408.08435]: automated design of agentic systems
  - **Darwin Godel Machine** [Zhang 2025, https://arxiv.org/abs/2505.22954]: recursively redesign self-evolving policy
  - **Godel Agent** [Yin 2024, https://arxiv.org/abs/2410.04444]: self-referential recursive self-improvement
- **Memory systems**:
  - **Voyager** [Wang 2023, https://arxiv.org/abs/2305.16291]: Minecraft skill library
  - **MemGen** [Zhang 2025a]: generative latent memory
  - **Mem0** [Chhikara 2025, https://arxiv.org/abs/2509.24704 (ECAI 2025)]: production-ready long-term memory

**LSE 的 distinct 之处**: 上述全部依赖 inherent reasoning ability，never explicitly trained for self-improvement。LSE 用 RL 直接优化这个 skill。

---

## 7. Limitations & Open Problems

1. **Single-step 简化**: 把 multi-step delegation 给 tree search。Joint multi-step trajectory optimization 可能更强但 credit assignment 难
2. **Per-domain training**: 每个 domain 训单独 policy，跨 domain 泛化未探索（likely 需大规模 multi-domain training）
3. **只 edit instruction field**: tools / skill library / external memory 未涉及
4. **可结合 latent / parameter space update**: 比如 Sun 2020 TTT [https://arxiv.org/abs/1911.01083] 或 Tandon 2025 end-to-end TTT [https://arxiv.org/abs/2512.23675]
5. **Environment scale 小**: environment curation 是 open problem — 需要 problem 有 feedback 且 problem 之间共享足够 structure 让 evolution meaningful

---

## 8. 给你的 Intuition Summary

1. **Self-evolution 是 distinct skill，不是 emergent ability 的副产品**。即使 GPT-5 reasoning 强，从未被训练"如何改自己的 context"，所以输给 explicit trained 4B。

2. **Single-step contextual bandit 足够**。Multi-step 的 exploration 交给 test-time tree search（UCB1），training 只学 single edit quality。这是 AlphaGo 式的分解：policy 学 local move，search 做 global planning。

3. **Improvement reward $\bar{R}(c_1) - \bar{R}(c_0)$ 数学上等价 post-edit reward（用 learned baseline 时），但实践中显著更好 (+4.3%)**。因为 $\bar{R}(c_0)$ 作 control variate cancel cross-prompt difficulty offset，而 GRPO group normalization 只能 within-prompt cancel。这是 paper 最 subtle 也最 actionable 的 finding。

4. **Tree search vs linear chain: +2.4%**。坏 edit 不 cascade，UCB 自动 backtrack 到高分 ancestor。

5. **环境结构决定 self-evolution 上限**：SQL 同 database 内 schema 共享 → transfer 强；MMLU 同 subject 内 deduplicated → transfer 弱。这是 design self-evolving system 时要考虑的 first-order factor。

6. **Parameter-level 和 prompt-level 优化互补**：Arctic-Text2SQL-R1（RL-tuned）+ LSE prompt evolution = +6.7%，证明 weights 和 context 是正交的 adaptation 通道。

7. **Curriculum trick**: 早期 preferentially sample high improvement-potential node（performance gap to tree max），给 policy 更多 headroom 学习。这是个实用 trick。

---

## Web References

- LSE paper (本篇): 待 arXiv 发布，作者 Xiaoyin Chen (Mila), Canwen Xu, Yuxiong He (Snowflake)
- UCB1 (Auer 2002): https://jmlr.org/papers/v3/auer02a.html
- verl (HybridFlow): https://arxiv.org/abs/2409.19256
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- OpenAI o1 (Learning to reason): https://openai.com/index/learning-to-reason-with-llms/
- Let's Verify Step by Step: https://arxiv.org/abs/2305.20050
- TTRL: https://arxiv.org/abs/2504.16084
- TTT-Discover: https://arxiv.org/abs/2601.16175
- TTT original (Sun 2020): https://arxiv.org/abs/1911.01083
- End-to-end TTT long context (Tandon 2025): https://arxiv.org/abs/2512.23675
- Reflexion: https://arxiv.org/abs/2303.11366
- SCoRe: https://arxiv.org/abs/2409.12917
- GEPA: https://arxiv.org/abs/2507.19457
- TextGrad: https://arxiv.org/abs/2406.07496
- DSPy: https://arxiv.org/abs/2310.03714
- ExpeL: https://arxiv.org/abs/2308.10144
- PromptBreeder: https://arxiv.org/abs/2309.16797
- ADAS: https://arxiv.org/abs/2408.08435
- Darwin Godel Machine: https://arxiv.org/abs/2505.22954
- Godel Agent: https://arxiv.org/abs/2410.04444
- Voyager: https://arxiv.org/abs/2305.16291
- Mem0: https://doi.org/10.3233/FAIA251160
- STaR: https://arxiv.org/abs/2203.14465
- Self-rewarding LMs: https://arxiv.org/abs/2401.10020
- Absolute Zero: https://arxiv.org/abs/2505.03335
- BIRD: https://arxiv.org/abs/2405.14178 (NeurIPS 2024)
- MMLU-Redux: https://arxiv.org/abs/2406.04127
- SuperGPQA: https://arxiv.org/abs/2502.14739
- Arctic-Text2SQL-R1: https://arxiv.org/abs/2505.20315
- AlphaGo (MCTS reference): https://www.nature.com/articles/nature24270
- PPO: https://arxiv.org/abs/1707.06347
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300

希望这个 walkthrough 帮你 build intuition。最值得消化的两件事：(1) single-step reduction + tree search 的分工；(2) improvement reward 数学等价但实践更强的 control-variate 解释。这两个 idea 都很 portable 到其他 self-improvement system 设计上。
