---
source_pdf: OpenR.pdf
paper_sha256: 9c127000859ff67964e79da3904c8b75fbe136fe38c04b76349000dedafd046e
processed_at: '2026-08-06T00:51:02-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OpenR 用大白话讲

---

## 1. 这篇 paper 到底在干嘛

一句话：**OpenAI 出了个叫 o1 的模型，reasoning 巨强，但没告诉你怎么训的。OpenR 这帮人猜了一下 o1 的做法，然后用开源工具拼出来一个能跑的版本。**

就像 OpenAI 做了一道菜很好吃但不给菜谱，这帮人尝了一口，猜里面放了什么调料，然后自己复刻了一份。当然味道没那么好，但至少菜谱公开了。

o1 的核心卖点是什么？就是模型在回答之前会"想很久"。以前的 LLM 是你问它问题，它马上开始写答案，写完就完了。o1 是你问它问题，它先在脑子里默默推理好几步，每步都检查对不对，确认没问题了才输出最终答案。

这就像普通人做数学题直接写答案 vs 学霸在草稿纸上一步步验算。OpenAI 发现：**让模型花更多时间在 inference 阶段"想"，比把模型做得更大更管用**。这叫 test-time compute scaling。

参考 o1: https://openai.com/index/learning-to-reason-with-llms/

---

## 2. 三个核心组件，像三个齿轮咬在一起

### 2.1 PRM —— 给每一步打分的老师

想象你在做数学作业。普通老师只看你最终答案对不对，对了满分错了零分。这叫 **ORM (Outcome Reward Model)**。

但好老师会看你的解题过程，告诉你"第三步公式用错了，第五步计算对了"。这就是 **PRM (Process Reward Model)**。

PRM 的训练方式很 hacky：拿一个 LLM，在 reasoning step 之间插特殊符号，让模型在符号后面输出 "+"（对）或 "-"（错）。本质上是把 LLM 改造成一个 step-level classifier。

为什么 PRM 重要？因为一道题 8 步推理，错在第 3 步和错在第 7 步，训练信号完全不同。PRM 能精确定位错误，让 RL 训练信号密集得多。

参考 OpenAI 的 PRM800K paper: https://arxiv.org/abs/2305.20050

### 2.2 RL Training —— 用 PRM 的分数训练 policy

有了 PRM 这个打分器，就能用 RL 训练 generator (也就是 policy)。

基本流程：
1. LLM 生成一条 reasoning trajectory
2. PRM 给每一步打分
3. RL 算法根据分数调整 LLM，让它以后生成高分 trajectory

这里 paper 用了两个算法：**PPO** 和 **GRPO**。

PPO 是老牌算法，需要训一个额外的 critic network 估计 baseline。GRPO 是 DeepSeek 提出的偷懒版——不用 critic，直接在同一个 prompt 下 sample 多条 trajectory，用 group 内的 z-score normalization 当 advantage：

$$A(s_t, a_t) = \frac{r_t^{PRM} - \text{mean}(r^{PRM})}{\text{std}(r^{PRM})}$$

- $r_t^{PRM}$：当前 trajectory 的 PRM 分数
- $\text{mean}(r^{PRM})$：同一 prompt 下所有 trajectory 的平均分
- $\text{std}(r^{PRM})$：标准差

Intuition 很简单：同一个题，你 sample 8 条解法，如果平均分是 0.5，你这条是 0.8，你的 advantage 就是正的——比平均好，该被 reinforce。如果平均分是 0.5，你这条 0.2，advantage 是负的——比平均差，该被惩罚。

好处是省了一个 critic network 的显存。坏处是如果 PRM 不稳定，整个训练就崩了。

参考 GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300

### 2.3 Test-time Search —— 推理时搜索

训练完之后，inference 时也不直接生成。而是 sample 多条 trajectory，用 PRM 给每条打分，选最好的。

最简单的是 **Best-of-N**：parallel sample N 条，PRM 打分，选最高分的。

复杂一点是 **Beam Search**：第一步 sample N 个候选，PRM 打分，保留 top N/m 个；第二步对每个保留的候选再 sample M 个 next step，总共 N 个，再打分筛选……以此类推。

低 budget 时 Best-of-N 好（多样性重要），高 budget 时 Beam Search 好（能早点 prune 掉走偏的 trajectory）。这个 crossover 现象在 Figure 4a 能看到。

参考 Snell scaling test-time compute: https://arxiv.org/abs/2408.03314

---

## 3. 数据怎么来：OmegaPRM

PRM 需要大量 step-level label 数据。OpenAI 的 PRM800K 是人工标注的，贵且慢。OpenR 用 OmegaPRM 自动生成。

核心 idea：**用 MCTS 风格的 tree search 自动找"哪步错了"**。

具体流程：
1. 对一个 question，LLM 生成一条 solution
2. 在中间某一步之后，让 LLM 重新 generate 后续部分（rollout），看最终答案对不对
3. 如果对，说明这一步没问题；如果错，说明这一步可能有 bug
4. 用 binary search 快速定位 first error step
5. first error 之前的部分成为新的 tree node，继续探索

这里有个 value function 决定优先探索哪个 node：

$$Q(s, r) = \alpha \cdot \frac{1}{1 - MC(s)} \cdot \beta \cdot \frac{\text{len}(r)}{L}$$

- $MC(s)$：从 state $s$ 出发的 Monte Carlo 成功率
- $1/(1-MC(s))$：MC 越高（越接近成功）越值得探索

这个设计反直觉但聪明：它不是优先探索"必错的 state"，而是优先探索"快要成功但还差一点的 state"。因为这些 state 最容易挖出 informative 的错误 label——它们大部分对了，只差一点点，这种 partial correct 的 data 对 PRM 训练最有价值。

参考 OmegaPRM: https://arxiv.org/abs/2406.06592

---

## 4. MDP 形式化：为什么把推理当决策过程

Paper 把 reasoning 建模成 MDP：

$$s_{t+1} = \{s_t, a_t\}$$

- $s_t$：当前的 reasoning prefix
- $a_t$：LLM 生成的下一个 reasoning step
- $s_{t+1}$：把 $a_t$ 拼到 $s_t$ 后面，就是新 state

Transition 就是字符串拼接。这比 Atari/MuJoCo 简单太多——没有外部环境，state transition 是 deterministic 的，LLM 自己决定下一个 state 是什么。

累计回报：

$$R^\gamma = \sum_{t=0}^{T} \gamma^t r_t^{PRM}$$

- $\gamma$：discount factor
- $r_t^{PRM}$：PRM 在第 $t$ 步给的 reward

这个 formulation 的好处是：一旦把 reasoning 看成 MDP，所有 RL 工具都能直接用。RL environment 就像 OpenAI Gym，只是这里的 "environment" 就是"字符串拼接器"。

---

## 5. 实验说明了什么

### 5.1 Test-time scaling 有效

Figure 4a 展示：token budget 从 $2^3$ 涨到 $2^5$，Best-of-N 和 Beam Search 的准确率都在涨。Majority Vote 涨得慢，因为它不利用 PRM 的打分信息。

这验证了 **test-time compute 确实是一种 scaling 维度**——你不需要更大的模型，只需要 inference 时多 sample 几条，用 PRM 选优。

### 5.2 PRM 质量很关键

Figure 4b 对比了 Math-psa (OpenR 自己训的) 和 Math-Shepherd (别人的) PRM。Math-psa 在所有 budget 上都赢，说明多数据集混合训练 + OmegaPRM 自动生成 data 的 pipeline 有效。

### 5.3 RL 训练能改善 reasoning

Figure 5 显示：在单个问题上，RL 训练后 reward 稳步上升，6 小时后收敛。但在 MATH500 整个数据集上，reward 波动大，泛化难。

这说明 RL 在 LLM reasoning 上还很不成熟——overfit 到单题容易，跨题泛化难。

### 5.4 Case study 很 informative

Figure 6/7 对比两个 PRM 在同一 solution 上的打分。Math-psa 能识别"多项式 horizontal shift 不影响根之和"这种数学细节，Math-Shepherd 不行。这说明 PRM 的质量直接决定整个 system 的上限。

Figure 10/11/12 更直观：同一个题，vanilla CoT 推理错了（第二步就开始跑偏），Best-of-N 和 Beam Search 都对了——因为它们 explore 了多条路径，能 self-correct。

---

## 6. PRM 和 Policy 互相 bootstrap

这是整个 framework 最 elegant 的设计，Figure 3 画得很清楚：

```
PRM 训练 policy (RL)
       ↕
Policy 生成 trajectory → 给 PRM 训练提供 data (OmegaPRM)
```

这跟 AlphaZero 的 policy network + value network co-training 是同构的。AlphaZero 的 policy 生成棋局，value network 评估局面，两者互相提升。OpenR 里 policy 生成 reasoning trajectory，PRM 评估每步对错，两者也是互相 bootstrap。

这个循环如果能跑起来，理论上能实现 self-improving reasoning——像 AlphaZero self-play 一样，越训越好。但 paper 实际只跑了一轮，没有 iterate 多次。这是 OpenR 距离真正 o1 复现的一个 gap。

参考 AlphaZero: https://www.nature.com/articles/nature24270

---

## 7. OpenR 没做到的部分

Paper Section 6 承认的：
- 模型小 (7B 级别，不是 70B+)
- 数据少 (15 万条 PRM data，o1 估计是千万级)
- domain 窄 (只测了 MATH，没测 coding、science)

Paper 没明说但我觉得关键：
- **没有真正的 NativeCoT**：o1 的 chain-of-thought 是 pretraining 阶段就训进去的，OpenR 只是 instruction-tuned model + RL fine-tune，本质上 chain-of-thought 还是 prompt 诱导的
- **PRM 和 policy 来自不同 base model**：PRM 在 Qwen2.5-7B 上训，policy 是 1.5B，distribution mismatch
- **MCTS 没真正跑出来**：codebase 里有 MCTS 但实验没展示，而 AlphaZero 的核心就是 MCTS + self-play
- **没有 self-play 循环**：policy 和 PRM 的互相 bootstrap 只跑了一轮，没 iterate

---

## 8. 这篇 paper 的真正价值

OpenR 本身实验结果不算惊艳，但它的价值在于 **把 o1 的"骨架"画出来了**：

```
Data: OmegaPRM 自动生成 step-level label
      ↓
PRM: 在 LLM 上 SFT，step token 后预测 +/-
      ↓
RL: PPO/GRPO 用 PRM reward 训练 policy
      ↓
Inference: Best-of-N / Beam Search + PRM 打分选优
```

这四个模块每一个都有 prior work，但 OpenR 是第一个把它们拼在一起开源的框架。后来 DeepSeek-R1、Skywork-o1 等都在这条线上继续推进。

参考 DeepSeek-R1: https://arxiv.org/abs/2501.12948

---

## 9. 一句话总结

**OpenR = PRM (给每步打分) + RL (用分数训 policy) + Search (推理时多 sample 选最好的)，三者拼在一起就是 o1 的开源平替。**

核心 insight 就三条：
1. **test-time 多花算力比训练时多花算力划算**——这是 o1 的 scaling law insight
2. **process reward 比 outcome reward 信号密集得多**——这是 credit assignment 的关键
3. **PRM 和 policy 互相 bootstrap**——这是 self-improving reasoning 的理论基础

剩下的工程细节都是在实现这三条 insight。OpenR 做得不算完美，但把 map 画出来了，后面的人照着走就行。

---

# OpenR: 对标 OpenAI o1 的开源复现框架深度解析

Andrej，这篇 paper 本质上是在做一件事：把 OpenAI o1 没有公开的训练 recipe，用开源 stack 复现一遍，把"test-time scaling law"这条曲线在 MATH 上跑出来。它把三件事拼在一起：**(1) PRM (Process Reward Model) 训练，(2) RL policy learning，(3) inference-time guided search**，而三者的共同底座是把 LLM 推理形式化成一个 MDP。下面我把每个模块的技术细节、公式含义、设计动机都拆开讲，并尽量连到你熟悉的相关工作上去 build intuition。

---

## 1. 全局图景：为什么 o1 的本质是 MDP 而不是 prompt

o1 的核心 insight 在 paper 的 Section 1 讲得很清楚——它把"chain-of-thought"从 prompt 层 (CoT prompting, Wei et al. 2022) 移到了**模型原生能力 (NativeCoT)**。这意味着思考过程 (the $\{R\}$ 中间步骤) 不是在 inference time 被 prompt 诱导出来的副产物，而是被 RL 优化目标显式训练出来的 trajectory。一旦把它看作 trajectory，就自然掉进了 RL 的 MDP framework：

$$
Q \{R\} A \quad \Leftrightarrow \quad s_0 \to (a_0, r_0) \to s_1 \to (a_1, r_1) \to \dots \to s_T \to a_T
$$

这里 $Q$ 是问题 (initial state $s_0$)，$R$ 是中间 reasoning step 序列，$A$ 是 final answer。每一步 reasoning step 就是一个 action，每一步都有一个 PRM 给出的 reward。这个 formulation 极其关键，因为它把"语言生成"和"决策过程"统一了，下面所有公式都从这里展开。

### 1.1 System 1 vs System 2 的类比

Paper 用 Kahneman 的双系统理论 (Thinking, Fast and Slow, 2011) 做 motivation：

| | System 1 | System 2 |
|---|---|---|
| 特性 | 快、自动、直觉 | 慢、刻意、分析 |
| 对应 LLM | vanilla autoregressive decoding | test-time search + PRM-guided planning |
| 计算分配 | forward pass 一次 | 多步 rollout + 评估 |

这个类比很贴切地解释了为什么 o1 把计算预算从 training 挪到 inference——System 2 thinking 本质上就是在 inference time 跑 search。

参考 Kahneman: https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow

---

## 2. MDP 形式化的细节

### 2.1 五元组的精确定义

Paper Section 3.4 把推理建模成 language-augmented MDP $\mathcal{M} = (\mathcal{V}, \mathcal{S}, \mathcal{A}, \mathcal{T}, R, \gamma)$：

| 符号 | 含义 | 在 LLM 推理中的对应 |
|---|---|---|
| $\mathcal{V}$ | vocabulary | LLM 的 token 词表 |
| $v \in \mathcal{V}$ | 单个 token | 一个 token |
| $\mathcal{A} \subset \mathcal{V}^N$ | action space | 一个 reasoning step (多个 token 组成) |
| $\mathcal{S} \subset \mathcal{V}^N$ | state space | 当前为止的 reasoning prefix |
| $\mathcal{T}: \mathcal{S} \times \mathcal{A} \mapsto \mathcal{S}$ | transition | $s_{t+1} = \{s_t, a_t\}$ (字符串拼接) |
| $R: \mathcal{S} \times \mathcal{A} \mapsto \mathbb{R}$ | reward function | PRM 给的 $r_t^{PRM}$ |
| $\gamma$ | discount factor | $<1$，让远期 reward 衰减 |

**关键 intuition**：state transition 就是字符串拼接，这等价于说"transition function 是 deterministic 的且免费的"——LLM 自己生成的 token 决定下一个 state，没有外部环境 (model-based RL 里的 world model 在这里就是字符串拼接)。这一点让整个 RL setup 比 Atari/MuJoCo 简单得多，也复杂得多——简单在 transition 是 free 的，复杂在 action space 是巨大的、unbounded 的。

### 2.2 累计回报

$$
R^\gamma = \sum_{t=0}^{T} \gamma^t r_t^{PRM}
$$

- $t$: 时间步索引，从 0 到最大步数 $T$
- $\gamma \in (0, 1)$: 折扣因子，$\gamma^t$ 让第 $t$ 步的 reward 被衰减
- $r_t^{PRM} = R(s_t, a_t)$: PRM 在第 $t$ 步给出的 process reward

这里 $\gamma$ 的选择很有讲究。如果 $\gamma$ 太小，模型会 short-sighted，只顾眼前 step 不出错；如果 $\gamma \to 1$，模型会变成 outcome-supervised，丧失 process granularity。实际上 paper 没有给出具体 $\gamma$ 值，但参考 DeepSeekMath GRPO 的实践通常取 1.0 (因为 reasoning 链长度有限)，这值得 follow up。

---

## 3. PRM：从 outcome supervision 到 process supervision

### 3.1 ORM vs PRM 的本质区别

这是理解整篇 paper 的核心。**ORM (Outcome Reward Model)** 只看 final answer 对不对：

$$
r^{ORM} = \text{ORM}(q, x_{1:T}, a_T) \in \{0, 1\}
$$

**PRM (Process Reward Model)** 给每一步打分：

$$
p_t = \text{PRM}([q, x_{1:t-1}], x_t) \in [0, 1]
$$

- $q$: 问题
- $x_{1:t-1} = [x_1, \dots, x_{t-1}]$: 前 $t-1$ 步 reasoning
- $x_t$: 当前第 $t$ 步
- $p_t$: 第 $t$ 步的正确性概率

PRM 的优势是 **credit assignment**：如果 final answer 错了，ORM 只能告诉你"错了"，PRM 能告诉你"第几步错的"。这在数学推理里特别重要，因为一道题 8 步推理，错在第 3 步和错在第 7 步的训练信号是完全不同的。

参考 OpenAI 的 PRM800K paper (Let's Verify Step by Step): https://arxiv.org/abs/2305.20050

### 3.2 PRM 的训练目标

PRM 被实现为 binary classifier：

$$
y_t = \text{PRM}(q, x_1, x_2, \dots, x_t) \in [0, 1]
$$

- $y_t$: 表示 "step $x_t$ 给定前面 $x_{1:t-1}$ 是否正确" 的概率
- 训练 label 是 $\{+,-\}$ (正确/错误)

训练时把 LLM 当 backbone，做一个 next-token prediction 任务。具体做法：
1. 在每一步结尾插入特殊 step token (`\n\n\n\n\n`)
2. 在 step token 后面接一个 `+` 或 `-` token 作为 label
3. Loss 只在 step token 位置计算 (其他位置 attention mask = 0)

这个设计很巧妙——它复用了 LLM 的 next-token prediction 训练 infra，本质上 PRM 是一个被 supervised fine-tune 过的 LLM，只不过它在 step token 位置输出"对/错"而非生成内容。

---

## 4. MATH-APS 数据生成：OmegaPRM 算法

这是 paper 里最技术化的部分。他们不想依赖 PRM800K 那种昂贵的人工标注，于是用 OmegaPRM (Luo et al. 2024) 自动生成数据。核心思想是**Monte Carlo Tree Search 风格的 rollout + binary search 找 first error**。

### 4.1 State-Action Tree 构建

对每个 question $q$，构建一棵树，每个节点包含：
- $q$: 问题
- $s$: solution prefix (到目前为止的推理)
- $\{(s, r_i)\}_{i=1}^k$: 所有 rollout 历史，$r_i$ 是第 $i$ 次 rollout 的结果

每个 edge 是一步或几步 reasoning。

### 4.2 Value function（PUCT 的 exploitation 项）

$$
Q(s, r) = \alpha \cdot \frac{1}{1 - MC(s)} \cdot \beta \cdot \frac{\text{len}(r)}{L}
$$

- $s$: 当前 state
- $r$: 一个候选 rollout
- $MC(s)$: 从 $s$ 出发的 Monte Carlo 成功率（采样若个完整 solution 看最终答案对不对的比例）
- $1/(1-MC(s))$: **state 越接近成功 (MC 越高) 越值得继续探索**——这是反直觉的，它希望 focus 在"快要成功但还差一点"的 state 上，因为这里能挖出 informative 的错误 step
- $\text{len}(r)$: rollout 长度
- $L$: 归一化常数
- $\alpha, \beta$: 调节权重

这里 $1/(1-MC(s))$ 的设计很巧妙。如果 $MC(s) = 0$ (这步必错)，价值是 $\alpha \cdot 1 \cdot \beta \cdot \text{len}(r)/L$；如果 $MC(s) \to 1$ (这步几乎全对)，价值 $\to \infty$。这驱动算法去挖掘 high-value 但 still-failing 的中间状态，因为这些状态最容易产生 informative 的错误 step label。

### 4.3 Exploration term (PUCT 的 exploration 项)

$$
U(s) = c_{\text{puct}} \cdot \frac{\sqrt{\sum_i N(s_i)}}{1 + N(s)}
$$

- $N(s)$: state $s$ 的访问次数
- $\sum_i N(s_i)$: 兄弟节点的访问总数 (相当于父节点总访问)
- $c_{\text{puct}}$: 探索常数 (AlphaZero 的经典 PUCT 公式)

这就是 AlphaGo/AlphaZero 的经典 UCB 公式。分子 $\sqrt{\sum_i N(s_i)}$ 让未被访问的 state 有更大 exploration bonus；分母 $1+N(s)$ 让已被多次访问的 state exploration 衰减。

### 4.4 Selection: PUCT 选择

$$
\Pi(s, r) = \arg\max_{(s, r)} [Q(s, r) + U(s)]
$$

在 tree traversal 阶段，每步选 $Q + U$ 最大的 rollout。

### 4.5 Binary search 找 first error

选定一个 rollout 后，用 binary search 找出第一个错误 step。这避免了线性 scan 的开销。所有"first error 之前的部分"成为新 state 加入树继续探索。所有 $0 < MC(s) < 1$ 的 state 加入候选 pool (因为它们是 informative 的——既不全对也不全错)。

这套算法本质上是 AlphaZero-style tree search + rollout + value head，只不过这里的 "value" 来自 PRM 而非学出来的 value network。参考 OmegaPRM: https://arxiv.org/abs/2406.06592

---

## 5. RL Training: PPO vs GRPO

### 5.1 GRPO 的核心公式

Paper 实现了两个算法：传统 PPO (Schulman et al. 2017) 和 DeepSeek 提出的 GRPO (Group Relative Policy Optimization, Shao et al. 2024, DeepSeekMath)。GRPO 的关键创新是**省掉了 critic network**。

GRPO 的 advantage 估计：

$$
A(s_t, a_t) = \frac{r_t^{PRM} - \text{mean}(r^{PRM})}{\text{std}(r^{PRM})}
$$

- $r_t^{PRM}$: 当前样本的 PRM reward
- $r^{PRM}$: 同 group 内所有样本的 reward 向量 (group = 同一个 question 下采样的多个 rollout)
- $\text{mean}(r^{PRM})$, $\text{std}(r^{PRM})$: group 内 reward 的均值和标准差

这就是 **z-score normalization 在同一个 prompt 下的多个 rollout 之间做**。Intuition 是：不需要 critic network 估计 $V(s)$，因为同一个 prompt 下的多个 rollout 的 reward 分布本身就编码了"baseline"信息——一个 rollout 比同组平均好就是正 advantage，比同组差就是负 advantage。

### 5.2 PPO vs GRPO 对比

| 维度 | PPO | GRPO |
|---|---|---|
| Critic network | 需要 | 不需要 |
| Advantage 估计 | GAE (基于 $V(s)$) | z-score normalization 在 group 内 |
| 显存开销 | 高 (policy + critic) | 低 (只有 policy) |
| 对 PRM 稳定性要求 | 较低 | 高 (因为 advantage 完全依赖 PRM) |
| Reference | Schulman 2017 | DeepSeekMath, Shao et al. 2024 |

GRPO 的 trade-off：省了 critic network，但把所有信号都压在 PRM 身上。如果 PRM 噪声大或 miscalibrated，advantage 估计会很糟糕。这是 paper Section 3.4 说 "GRPO emphasizes the stability of PRMs more" 的意思。

参考 DeepSeekMath (GRPO 原始 paper): https://arxiv.org/abs/2402.03300

---

## 6. Decoding: Test-time guided search

### 6.1 PRM 打分策略

给定一个 solution，PRM 在每一步都打分 $\{r_t^{PRM}\}_{t=0}^T$，需要聚合成一个 solution-level score：

| 策略 | 公式 | Intuition |
|---|---|---|
| **PRM-Min** | $v = \min\{r_t^{PRM}\}_{t=0}^T$ | 最短板决定一切 (worst step 论) |
| **PRM-Last** | $v = r_T^{PRM}$ | 最后一步 score 通常是 cumulative 信号 |

Snell et al. 2024 发现 PRM-Last 和 PRM-Min 一样好，这其实有点反直觉——直觉上 PRM-Min 更严格 (一旦有 weak step 就扣分)，但 PRM-Last 足够好可能是因为 PRM 在训练时已经隐式做了 cumulative aggregation (因为每步都看了前面所有 step)。

### 6.2 多 answer 聚合策略

| 策略 | 公式 | 含义 |
|---|---|---|
| **Majority-Vote** | $f^* = \arg\max_f \sum_{y^j} \mathbb{1}_{\text{final\_ans}(y^j)=f}$ | 纯投票，不依赖 reward |
| **RM-Max** | $f^* = \text{final\_ans}(\arg\max_{y^j} v(y^j\|x))$ | 选 reward 最高的那条 |
| **RM-Vote** | $f^* = \arg\max_f \sum_{y^j; \text{final\_ans}(y^j)=f} v(y^j\|x)$ | 按 answer 分组求 reward 和 |

- $y^j$: 第 $j$ 条 sampled solution
- $f$: candidate final answer
- $v(y^j|x)$: solution $y^j$ 的 PRM score
- $\mathbb{1}$: 指示函数

**Intuition**：Majority-Vote 是 self-consistency (Wang et al. 2022) 的做法，它假设 "多数答案 = 正确答案"。RM-Max 是 greedy 选 best。RM-Vote 是 weighted voting，把 PRM score 当 weight，比纯投票信息量大。

参考 Self-Consistency: https://arxiv.org/abs/2203.11117

### 6.3 Search 算法

#### Best-of-N
最简单：parallel sample N 条 solution，每条独立打分，选 score 最高 (或 majority vote)。低 budget 时最优。

#### Beam Search
```
Step 1: 生成 N 个 step-1 候选 → PRM 打分 → 保留 top N/m 个
Step 2: 每个 retained 候选 sample M 个 next-step → 共 N 个 step-2 候选 → 打分 → 保留 top N/m
...
```

这里 $N/m \in \mathbb{Z}$ 必须整除。Beam search 在高 budget 时反超 Best-of-N，因为它能 **prune 早期就跑偏的 trajectory**，把算力集中在有希望的方向上。低 budget 时不如 Best-of-N，因为 beam 太窄导致多样性不够。

Paper 还提到 MCTS 在 codebase 里，但实验没展开。这其实是最像 AlphaZero 的部分，参考 rStar / AlphaZero-like tree-search (Feng et al. 2024): https://arxiv.org/abs/2407.01079 (实际 paper 在 ICML 2024)

---

## 7. 实验结果分析

### 7.1 Test-time scaling (Figure 4)

实验设置：
- Base LLM: Qwen2.5-Math-7B-Instruct (作为 generator)
- PRM: Math-psa (在 Qwen2.5-Math-7B-Instruct 上 fine-tune)
- 数据集：PRM500K + Math-Shepherd + MATH-APS (~150k pairs)
- 测试集：MATH500 (Lightman et al. 2023 的子集)
- Budget: token 数量级 $2^3$ 到 $2^6$ 左右

**Figure 4a 关键发现**：
- 低 budget (<$2^4$ tokens): Best-of-N > Beam Search
- 高 budget (>$2^5$ tokens): Beam Search ≥ Best-of-N (用 PRM-Last)
- 两者都显著超过 Majority Vote

这个 crossover 现象的 intuition：Beam search 牺牲多样性换聚焦度，低 budget 时多样性更重要 (Majority Vote 反而好)，高 budget 时聚焦度更重要 (能 prune 早错的路径)。

**Figure 4b**: Math-psa PRM 在所有 budget 上都超过 Math-Shepherd PRM，说明 OmegaPRM 自动生成 + 多数据集混合 training 的 PRM pipeline 有效。

### 7.2 Online RL training (Figure 5)

- Policy model: Qwen2.5-1.5B-Math-Instruct
- PRM: Math-Shepherd (作为 reward model)
- 单问题 ("196 has how many positive divisors?", answer=9): reward 在 6 小时后稳定
- MATH500: reward 波动大，泛化难

单问题 reward 稳定 vs MATH500 波动，说明 **policy overfit 到具体问题容易，跨问题泛化难**。这是 RL 在 LLM 上的经典困境——reward 信号太 sparse + reasoning trajectory 太长导致 credit assignment 困难。Paper 6 也承认这是 limitation。

### 7.3 Case study 关键观察

- Figure 6/7: Math-psa PRM 比 Math-Shepherd 更能识别"看起来合理但本质上错"的 step。例如多项式 horizontal shift 不影响根之和这个细节，Math-psa 给低分，Math-Shepherd 给高分。
- Figure 8/9: RL 训练前模型把"周长"当"边长"用，RL 训练后修正了。
- Figure 10/11/12: CoT 错而 Best-of-N/Beam Search 对——说明 test-time compute 通过 search space 扩大能 self-correct。

---

## 8. 与相关工作的关系网

### 8.1 数据生成谱系

```
STaR (Zelikman 2022)
   ↓
Quiet-STaR (Zelikman 2024, pretraining 阶段学 think token)
   ↓
Math-Shepherd (Wang et al. 2024, 自动 step-level label)
   ↓
OmegaPRM (Luo et al. 2024, MCTS + binary search)
   ↓
MATH-APS (本文)
```

参考 Quiet-STaR: https://arxiv.org/abs/2403.09629
参考 Math-Shepherd: ACL 2024
参考 STaR: NeurIPS 2022

### 8.2 Verifier 谱系

```
ORM (Cobbe et al. 2021, GSM8K verifier)
   ↓
Process + Outcome Supervision (Uesato et al. 2022, DeepMind)
   ↓
PRM800K (Lightman et al. 2023, OpenAI, 人工 step-level label)
   ↓
GenRM (Zhang et al. 2024, generative verifier, 用文本解释为什么错)
   ↓
Math-psa (本文, 多源混合训练)
```

参考 GenRM: https://arxiv.org/abs/2408.15240

### 8.3 Test-time compute 谱系

```
Self-Consistency (Wang et al. 2022)
   ↓
Chain-of-Thought prompting (Wei et al. 2022)
   ↓
Pause tokens (Goyal et al. 2023, "think before speak")
   ↓
Tree of Thoughts (Yao et al. 2023)
   ↓
Reasoning-as-Planning (Hao et al. 2023, RAP)
   ↓
AlphaZero-like tree-search (Feng et al. 2024, rStar)
   ↓
Scaling test-time compute (Snell et al. 2024)
   ↓
o1 (OpenAI 2024)
   ↓
OpenR (本文)
```

参考 Snell scaling test-time compute: https://arxiv.org/abs/2408.03314
参考 RAP: https://arxiv.org/abs/2305.14992
参考 o1 blog: https://openai.com/index/learning-to-reason-with-llms/

---

## 9. Intuition 总结：OpenR 的三个关键 insight

### Insight 1: Test-time compute 也是一种 scaling 维度

Snell et al. 2024 已经证明：增加 inference 时的 sample 数 / search depth，ROI 比单纯增加 model size 高。OpenR 在 MATH 上验证了这一点——Best-of-N 在 budget 从 $2^3$ 涨到 $2^5$ 时准确率持续提升。这背后的原因是 **reasoning 是一个 search 问题，而 search 的难度是 sub-linear in compute 的** (你 sample 多一条就有可能找到正确答案)，而 model size 是 log-linear scaling 的。

### Insight 2: PRM 是密集的 credit assignment 信号

ORM 把整条 trajectory 压成一个 bit (对/错)。PRM 保留每一步的 reward。这等价于把一个 $\{0,1\}$ sparse reward 转成 $[0,1]^T$ dense reward，**让 RL 的 credit assignment 变得 tractable**。这是为什么 PRM-guided search 在 paper 里能 +10% 的根本原因——signal-to-noise ratio 高得多。

### Insight 3: PRM 和 Policy 是互相 bootstrap 的

Paper Section 3.1 的 Figure 3 是核心 diagram：PRM 训练 policy (RL)，policy 生成的 trajectory 反过来给 PRM 训练提供 data (MATH-APS)。这等价于 **actor-critic 的 co-training，只不过 critic 是 PRM，actor 是 LLM policy**。这跟 AlphaZero 的 policy network + value network co-training 是同构的。

---

## 10. 局限与未来方向

Paper Section 6 自己承认：
- 实验规模小 (Qwen2.5-7B 级别)
- PRM 数据有限 (PRM800K + Math-Shepherd + 150k MATH-APS)
- 没在更广 domain (代码、科学推理) 测试

**未明说但我觉得是关键 limitation 的**：
1. **没有 NativeCoT 真正实现**：paper 说要 replicate o1 的 native chain-of-thought，但实际做法是用 Qwen2.5-Math-Instruct + RL fine-tune，这本质还是 instruction-tuned model + RL post-hoc。真正 o1-style 的 NativeCoT 应该在 pretraining 阶段就引入 think tokens (类似 Quiet-STaR)。这是 OpenR 没做到的部分。
2. **PRM 与 policy 的 distribution shift**：PRM 在 Qwen2.5-7B 上训练，policy 是 Qwen2.5-1.5B。Reward model 和 policy 来自不同 base model，会有 distribution mismatch。
3. **GRPO 在数学推理上的优势未充分验证**：DeepSeekMath 用 GRPO 在 GSM8K 上证明有效，但 MATH500 上的波动 (Figure 5) 说明 GRPO + PRM 组合在复杂推理上还不稳定。
4. **没有真正的 self-play**：MCTS codebase 在 paper 里没真正跑出来——而 AlphaZero 的核心就是 self-play + MCTS。这是 OpenR 距离真正 o1-style 复现最大的 gap。

后续工作 DeepSeek-R1 (2025)、Skywork-o1、Critique-out-Loud 等都在这条线上推进，尤其是 DeepSeek-R1 用了纯 RL (without SFT cold start) 训练 reasoning 的路线，把 OpenR 这套框架推到了一个更极端的设定。

参考 DeepSeek-R1: https://arxiv.org/abs/2501.12948

---

## 11. 给你的几个 research 方向联想

1. **PRM as world model**: AlphaZero 的 value network 实际上 world model 的 proxy (估计最终胜负)。PRM 估计 "step correctness"，这能不能升级成 "state value" (估计从当前 state 到达 final answer 的成功率)？这其实 OmegaPRM 的 $MC(s)$ 已经在做，但能不能 end-to-end 学出来而不是 MC 采样？
2. **Test-time compute 的 scaling law 形式**: Snell 2024 给了经验曲线，但理论形式 (类似 Chinchilla) 还没有。如果存在 $N_{\text{test}}$ vs $N_{\text{train}}$ 的 iso-loss curve，会是个非常 elegant 的结果。
3. **Generative reward model (GenRM) 与 PRM 的统一**: GenRM 用自然语言解释错误，PRM 用 scalar。能不能把 PRM 改成 generative 的——既给 scalar 又给 explanation？这会让 reward signal 信息密度高一个数量级。
4. **NativeCoT 在 pretraining 阶段**: Quiet-STaR 已经在 pretraining 加 think tokens，但 scale 不够。如果把 Llama-3-405B scale + Quiet-STaR-style pretraining，可能就是真正的 o1 复现路径。
5. **Process supervision 在 coding / agent 任务上的扩展**: MATH 是 closed-form 答案，PRM 容易训。Coding / agentic 任务 reward 更模糊，需要 PRM 能判断 "这个 tool call 是否合理"，这是 GenRM 的强项。

参考 OpenR 项目主页: https://openreasoner.github.io
参考 Qwen2.5-Math: https://arxiv.org/abs/2409.12122
参考 MATH dataset: https://arxiv.org/abs/2103.03874

---

如果你要我深挖某一块（比如 OmegaPRM 的 binary search 细节、GRPO 的 loss 形式、MCTS codebase 的实现），告诉我。这篇 paper 的技术深度其实都藏在 references 里，OpenR 本身是个 integration 工作，但 integration 的方式很有信息量——它告诉你 o1 的"骨架"长什么样。
