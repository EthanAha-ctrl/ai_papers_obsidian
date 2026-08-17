---
source_pdf: Scaling Small AgentsThroughStrategyAuctions.pdf
paper_sha256: b97a401064852f3de3a2316b362c6e8a89b27305decc46da5c41aafd2ac54b5b
processed_at: '2026-08-12T03:40:37-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话总结

**小模型在简单任务上跟大模型差不多，但在复杂任务上被大模型按在地上摩擦。作者发明了一个"拍卖会"系统，让几个不同大小的模型一起竞标任务，结果竟然又省钱又准。**

---

## 故事从哪开始

最近 AI 圈有个流行说法：小模型 + tools 就够了，不需要大模型。Meta 的人想验证这个事儿。

他们搞了个 benchmark 叫 HST-Bench，按 **人类解决问题需要的时间** 来分任务难度。为什么用人类时间？因为这是衡量任务复杂度最 natural 的 metric——一个专家几秒钟能搞定的任务 vs 需要一小时的任务，复杂度差距是巨大的。

然后他们跑了 Qwen3 的四个 size：4B、8B、14B、32B。

**发现的事实**：

简单任务（人类几秒）上，4B 能达到 32B 的 ~87% 准确率。看起来"小模型够用"这个 claim 是对的？

但是复杂任务（人类一小时）上，4B 只剩 32B 的 ~25%。差距直接崩塌。

更反直觉的是：大模型并没有因为"更聪明"而用更少的 token。在长任务上，32B 和 4B 消耗的 token 差不多，甚至 32B 更多。所以"大模型虽然贵但会自己省回来"这个假设是**错的**。

---

## 核心问题

如果你永远用 32B：简单任务上浪费钱（4B 本来就能搞定）。
如果你永远用 4B：复杂任务上崩盘。

所以你需要一个 **router**——根据任务分配给合适的 model。

---

## 已有 router 为什么不行

已有方法分两类：

### Predictive router（提前选模型）
比如 WTP、CARROT、TO-Router。它们用 task description 训练一个 classifier，预测哪个 model 好。

问题：
1. 需要单独训一个 router model，贵且不 generalizable
2. Dhrif (2025) 发现这些 router 在 task difficulty 上升时 degrade
3. 它们是 static 的——训完就固定了，不会随经验变好

### Non-predictive router（跑完所有 model 再选）
比如 FrugalGPT。让所有 model 都跑一遍，然后用一个 scorer 选最好的输出。

问题：
1. Agent trajectory 可能是 millions of tokens——跑 4 个 model 代价 4 倍
2. Scorer 在 long-horizon 任务上很难判断答案对错
3. 实验中 FrugalGPT 在 coding 任务上比单独用 32B 还贵（$0.57 vs $0.36）

---

## SALE 的核心 idea

灵感来自 **Upwork 这样的 freelancer marketplace**。

想象你是个老板，要发包一个项目。你不会让所有 freelancer 都把项目做完再选——太贵了。你会让他们先**提交方案**，你选一个最好的方案，然后只让中标的人执行。

SALE 就是这么干的：
1. Task 来了，所有 4 个 model 各写一个 **strategic plan**（几百 token，不是完整 trajectory）
2. 系统给每个 plan 打 cost 分和 value 分
3. 选 cost - value 最低的 model 作为 provisional winner
4. 比 provisional winner 便宜的 model，去 **memory bank** 里检索过去类似任务的 winning/losing plans，对比学习，改进自己的 plan
5. 如果改进后的 plan 更好，替换 winner
6. 最后只执行 winner 的完整 trajectory
7. 把所有 plans 和结果存入 memory

---

## Cost 怎么算

$$C_{t,i} = w_c \cdot \pi(a_i) \cdot |s_{t,i}|$$

翻译成人话：cost = 系数 × model 的每百万 token 价格 × plan 的长度

为什么用 plan 长度？
- Goebel & Zips (2025) 发现 plan 长度跟最终 trajectory 长度正相关（$\rho = 0.39$）
- 长 plan 的 execution reliability 也更差（Xiong et al., 2025a）—— 失败的 execution 仍然烧 token

所以 plan 越长，预期 cost 越高。

价格用 Qwen3 的 API rate：4B=$0.05/Mt, 32B=$0.36/Mt。

---

## Value 怎么算

$$V_{t,i} = w_h \cdot H(s_{t,i}) + \sum_{a_j \in \mathcal{A}} w_j \cdot \gamma_j(s_{t,i})$$

两个信号相加：

**信号 1：Entropy** $H(s_{t,i})$
就是 plan 的信息熵。高 entropy 意味着 plan 里的 reasoning 不冗余、信息量大。Wang et al. (2025c) 在 NeurIPS 发现高 entropy minority tokens 对 RL reasoning 特别重要。实验中这个 term 的 weight 最大。

**信号 2：Jury 评分** $\gamma_j(s_{t,i})$
让 4 个 model 都当 judge 给每个 plan 打 0-5 分。包括 self-assessment（自己给自己打的分也算）。Mousavi et al. (2023) 证明 self + peer 比 peer-only 更可靠。

为什么 jury 比 single judge 好？Appendix I 的 ablation 显示：任何 single judge 都不如 full jury；去掉任何一个 judge 都会降分；甚至 4B 当 judge 都有独特贡献。Diversity 是关键。

---

## 怎么选 winner

训练时：用 Mixed Integer Linear Programming 学 weights $w = (w_c, w_h, w_1, w_2, w_3, w_4)$，目标是 **min-max**——让 worst-case task 的 cost - value 最小。

为什么 min-max 而不是 average？保证没有任何单个 task 被糟蹋。Robustness。

推理时：每个 task 选 $\arg\min_i (C_{t,i} - V_{t,i})$。

---

## Memory 怎么用

这是 SALE 最 clever 的部分。

Memory 里存的是：每个 past task 的所有 plans + 谁赢了谁输了。

当 task $t$ 来了，provisional winner 是 32B。那 4B、8B、14B 就去 memory 里找最相似的 8 个 past tasks，取出每个的 (winning plan, losing plan) pair，用 **contrastive prompt** 让小 model 学习："看，这个 plan 输了，这个 plan 赢了，你改进一下你的 plan。"

关键细节：**只有比 provisional winner 便宜的 agent 才 refine**。如果 4B 已经赢了，就不 refine 了。这保证了 cost-efficiency——不会所有人都多跑一次。

---

## 实验结果

### 主结果

**Deep Search 任务：**

| Method | Pass@1 | $/Mt | 相比单独 32B |
|---|---|---|---|
| 单独 32B | 63.8 | 0.36 | - |
| WTP | 62.4 | 0.32 | 又差又贵 |
| CARROT | 61.3 | 0.28 | 省钱但更差 |
| FrugalGPT | 61.0 | 0.51 | 又差又贵 |
| **SALE** | **67.3** | **0.21** | **+3.5 pp, -42% 钱** |

**Coding 任务：**

| Method | Pass@1 | $/Mt |
|---|---|---|
| 单独 32B | 58.4 | 0.36 |
| WTP | 50.1 | 0.11（极省钱但崩盘）|
| **SALE** | **61.1** | **0.27** |

SALE 同时提升 accuracy 和降低 cost——这在 routing 文献里非常罕见。通常 router 只能在两者间 trade off。

### Shapley Value 分析

用合作博弈论算每个 model 对系统的 marginal contribution。

**没有 memory 时**：32B 的 Shapley 最高——虽然不一定总被选来做 final execution，但它在 jury 和 memory 中的贡献大。

**有 memory 时**：32B 的 Shapley 下降，4B/8B 上升——memory 让小 model "升级"了。

### 4B 的选择率随时间增长

Figure 5：4B 的 cumulative selection rate
- Deep Search: 3.7% → 11.1%（3 倍增长）
- Coding: 1.4% → 5.3%（近 4 倍增长）

随着 memory 积累，小 model 学到了更好的策略，能赢更多 task。这是 SALE 区别于 static router 的核心——它会**随经验变好**。

---

## 为什么 SALE 能超越任何单 model

这是最关键的问题。如果 32B 的 error 是所有小 model error 的子集，那 routing 只能省钱不能提精度。

但 Appendix K 的 qualitative analysis 发现 **failure modes 是互补的**：

**32B 经常犯的错**：
1. **过度自信**——绕过 search tool 直接用 parametric memory 回答（例子：医学解剖题，32B 直接答错了；14B 老实 search 答对了）
2. **Over-engineering**——简单问题搞复杂方案（例子：对固定两层嵌套用递归，4B 直接识别结构）
3. **跳过验证**——不用 Python shell 测试就提交代码（32B 经常这么干）
4. **忽略 edge cases**——比如"没有 duplicate"的情况忘了处理

**小 model 的优势**：
1. 更依赖 tools（更老实 search、更勤快 test）
2. Plan 更简单，执行更少出错
3. 更 explicit 的 verification steps

最关键的是：**这些差异在 strategic plan 阶段就可见**。SALE 在 bid 阶段就能看出"32B 这次又要 over-engineer 了"——这就是 routing signal。

---

## 各组件 ablation

| 去掉什么 | Deep Search Pass@1 | Coding Pass@1 |
|---|---|---|
| 完整 SALE | 67.3 | 61.1 |
| 去掉 price term | 65.4 | 59.4 |
| 去掉 plan length | 66.0 | 58.7 |
| 去掉 entropy | 65.5 | 59.1 |
| 去掉 jury | 63.7 | 59.9 |
| 去掉 memory | 66.4 | 59.9 |

每个组件都有贡献。Jury 对 deep search 最关键（-3.6），cost terms 对 coding 最关键。

---

## 与 oracle router 的差距

Oracle 是后见之明——跑所有 model，选答对的最便宜那个。

| Task | Oracle Pass@1 | SALE Pass@1 | Gap |
|---|---|---|---|
| Deep Search All | 77.9 | 67.3 | 10.6 |
| Coding All | 68.4 | 61.1 | 7.3 |

SALE 离 oracle 还有距离，但**在所有 router 中 SALE 离 oracle 最近**。其他 router 的 gap 是 14-18 个百分点。

剩下的 gap 主要在复杂任务——当所有 model 都答错时，router 怎么选都没用。

---

## 我的直觉理解

### 1. Plan 是天然的 routing signal

传统 router 用 task description，但 task description 与 long-horizon success 之间是 **indirect mapping**——trajectory 太长，从 input 看不出哪个 model 会赢。

Plan 是 **intermediate representation**：它 encode 了 model 对 task 的理解、分解策略、tool selection。这些信息在 task description 里没有，在 final trace 里太贵。

Sun et al. (2024) 的 correlation evidence + SALE 的 ablation 共同证明：**plan quality 预测 execution quality**。

### 2. Marketplace 比 static router 更 natural

传统 router 是 fixed mapping：task features → model。SALE 是 dynamic system：
- 竞争
- 同行评审
- 经验积累
- Upskilling

这模拟了真实劳动力市场。Freelancer 通过竞争和反馈提升技能。

### 3. "Scaling small agents" 的新含义

传统 scaling 是增加参数。SALE 提出了另一种 scaling：**通过 coordination 让小 model 在 system 层面 scale up**。

Capability 不是单个 model 的 property，而是 agent ecosystem 的 emergent property。

### 4. Conservative routing 的智慧

Routing diagnostics 显示 SALE 主要 error 是 **over-escalation**（用大 model 做了本可以小 model 做的事）而不是 under-escalation（用小 model 做了需要大 model 的事）。

也就是说 SALE 宁可多花钱也不冒险——这解释了为什么 accuracy 不降反升。

---

## Limitations

作者自己承认的：
1. 只测了 deep search 和 coding——generalization 未知
2. 只用了 Qwen3 4B-32B——70B+ 行为未知
3. Cost 只算 token，不算 tool call
4. Memory 线性增长——长期部署需要 summarization

---

## 这篇 paper 的 bigger picture

从 Karpathy 的 systems thinking 角度，这篇 paper 最大的贡献可能是：

**它把 agent coordination 从 "how to call tools" 提升到 "how to organize a market of agents"**。

未来的 agentic AI 可能不是"一个 ever-larger model"，而是"一堆 heterogeneous agents 通过 market mechanism 协作"。SALE 给了一个 concrete 的实例。

这个方向让我想起 Tomasev et al. (2025) 的 virtual agent economies，也让我想起 Rothschild et al. (2025) 的 agentic economy——这个方向在 academic 界正在形成共识。

参考：
- [Tomasev et al., 2025 - Virtual Agent Economies](https://arxiv.org/abs/2509.10147)
- [Rothschild et al., 2025 - The Agentic Economy](https://arxiv.org/abs/2505.15799)
- [Kwa et al., 2025 - Measuring AI Ability](https://openreview.net/forum?id=CGNJL6CeV0)
- [Froger et al., 2025 - ARE Framework](https://arxiv.org/abs/2509.17158)

---

# Scaling Small Agents Through Strategy Auctions 深度解析

## 1. 核心动机：Model Size 与 Task Complexity 的交互关系

这篇 paper 来自 Meta Superintelligence Labs (Lisa Alazraki, William F. Shen, Yoram Bachrach, Akhil Mathur)，发表于 2026 年 2 月。核心问题非常 Karpathy-style：**小模型到底在什么情况下够用？什么时候必须上大模型？**

### 1.1 关键实证发现

作者在 HST-Bench 上测试了 Qwen3 系列（4B, 8B, 14B, 32B），用 **human solution time** $\tau(t)$ 作为 task complexity 的 proxy：

| Task complexity bin | Cheapest agent relative performance (Deep Search) | Cheapest agent relative performance (Coding) |
|---|---|---|
| $\tau(t) \leq 0.1$ min (最简单) | ~87% of 32B | ~92% of 32B |
| $\tau(t) \leq 60$ min (最复杂) | ~25% of 32B | ~17% of 32B |

这个结果非常关键：**scaling curves 在简单任务上几乎 flat，在复杂任务上急剧 stratify**。这意味着"小模型足够做 agent"这个 claim 是 task-complexity-dependent 的。

### 1.2 Token efficiency 的反直觉发现

Figure 1b 显示了一个重要的反直觉现象：大模型并没有因为"更聪明"而用更少的 token 解决问题。在 long-horizon 任务上，32B 的 trace length 与 4B 相当甚至更长。这意味着**大模型的 per-token cost 并不会被 trajectory length 补偿掉**。

这个发现直接否定了"大模型虽然贵但会自己 pay for itself"的假设——它不会。

---

## 2. SALE 框架：Marketplace-Inspired Coordination

### 2.1 核心思想

SALE 的灵感来自 freelancer marketplace（如 Upwork）。把每个 agent 想象成自由职业者，任务来了之后，所有 agent 提交一个 **strategic plan**（不是完整解决方案！）作为 bid。系统根据 cost-value tradeoff 选择 winner，然后只执行 winner 的完整 trajectory。

关键 insight 来自 Sun et al. (2024) 等 work：**plan quality 与 execution quality 强相关**。所以我们可以用 plan 作为 routing signal，而不需要跑完所有 agent。

### 2.2 Architecture 解析（Figure 2）

```
Task t 输入
    ↓
[所有 agents a_i 同时生成 strategic plan s_{t,i}]  ← 初始 bid
    ↓
[Cost-Value 评分: C_{t,i}, V_{t,i}]
    ↓
[Provisional winner: argmin(C - V)]
    ↓
[Cheaper agents 检索 auction memory M]
    ↓
[Contrastive prompting: 学习 past winning/losing strategies]
    ↓
[Refined bids s_{t,i}^r]
    ↓
[Final winner selection: 如果 refined bid 更优则替换]
    ↓
[Execute winner's trajectory]
    ↓
[Store all strategies + outcome in M]
```

这个 pipeline 有几个精妙的设计：
1. **Bid 是 short plan，不是 full trace**——成本只有几百 token，而 final trace 可能是 millions of tokens
2. **Memory refinement 是 opportunistic**——只有 cheaper agents 才 refine，且只在它们没赢时才 refine
3. **Self-improvement 是 online 的**——memory 随任务积累而增长

---

## 3. 公式深度解析

### 3.1 Cost Function

$$C_{t,i} = w_c \cdot \pi(a_i) \cdot |s_{t,i}|$$

变量解释：
- $C_{t,i}$: agent $a_i$ 在 task $t$ 上的 estimated cost
- $w_c$: 可学习的 weight（通过 MILP 优化）
- $\pi(a_i)$: agent $a_i$ 的 price per million tokens（4B=$0.05, 8B=$0.09, 14B=$0.16, 32B=$0.36）
- $|s_{t,i}|$: strategy $s_{t,i}$ 的 token length

为什么用 strategy length 作为 cost proxy？两个原因：
1. Goebel & Zips (2025) 发现 plan length 与 final trace length 相关（Spearman's $\rho = 0.39$, Pearson's $r = 0.36$）
2. Xiong et al. (2025a) 发现 longer plans 的 execution reliability 下降——失败的 execution 仍然消耗 compute，所以 longer plans 意味着更高的 expected cost（token + wasted computation）

### 3.2 Value Function

$$V_{t,i} = w_h \cdot H(s_{t,i}) + \sum_{a_j \in \mathcal{A}} w_j \cdot \gamma_j(s_{t,i})$$

变量解释：
- $V_{t,i}$: agent $a_i$ 在 task $t$ 上的 estimated value
- $w_h$: entropy term 的 weight（实验中发现这个 weight 最大）
- $H(s_{t,i})$: strategy $s_{t,i}$ 的 normalized entropy（mean per-token entropy from generation log-probabilities）
- $w_j$: judge $a_j$ 的 weight
- $\gamma_j(s_{t,i})$: agent $a_j$ 给 strategy $s_{t,i}$ 的 score（0-5 整数）

两个信号：
- **Intrinsic quality** (entropy)：高 entropy reasoning 意味着更多 informational content 和 less redundancy (Chen et al., 2025b; Cheng et al., 2025)
- **Extrinsic quality** (jury)：混合 self + peer assessment (Mousavi et al., 2023 证明这比 peer-only 更可靠)

### 3.3 Min-Max Optimization（Mathematical Program 1）

$$\min_{w, x, Q} Q \quad \text{s.t.} \quad z_t \leq Q \; \forall t, \quad \sum_{a_i \in \mathcal{A}} x_{t,i} = 1 \; \forall t, \quad w \in \mathbb{R}^{2+|\mathcal{A}|}$$

其中：
- $w = (w_c, w_h, \{w_j\}_{a_j \in \mathcal{A}})$: learned weights
- $x_{t,i} \in \{0,1\}$: assignment variable（是否选 agent $i$ 做 task $t$）
- $Q$: worst-case cost-minus-value across all training tasks
- $z_t$: task $t$ 的 cost-minus-value

Big-M constraints 确保只有 selected agent 的 $C-V$ 被 "激活"：
$$z_t \geq C_{t,i} - V_{t,i} - M(1-x_{t,i}) \; \forall t, a_i$$
$$z_t \leq C_{t,i} - V_{t,i} + M(1-x_{t,i}) \; \forall t, a_i$$

这里 $M = 10^4$，远大于 observed range of $C-V$ scores。

**为什么用 min-max 而不是 average？** 防止任何单个 task 得到极差的 assignment。这给了 robustness。

### 3.4 Inference-time Assignment

$$z_t = \sum_{a_i \in \mathcal{A}} x_{t,i}(C_{t,i} - V_{t,i})$$

由于每 task 只选一个 agent，简化为：
$$z_t = C_{t,\hat{i}(t)} - V_{t,\hat{i}(t)}$$
其中 $\hat{i}(t) = \arg\min_i (C_{t,i} - V_{t,i})$

即选择 cost-minus-value 最小的 agent。

### 3.5 Memory-based Refinement

Memory record 格式：
$$\mathcal{M}(t') = (t', \{s_{t',i}\}_{a_i \in \mathcal{A}}, y_{t'})$$

检索：
$$\mathcal{M}_{t,i} = \{(s_{t'}^{\text{lose}}, s_{t'}^{\text{win}})_i \mid t' \in \text{top-}k \text{ sim}(t, t')\}, \quad \tilde{k} = \min(k, |\mathcal{T}_{\mathcal{M}}|)$$

- 用 cosine similarity over text embeddings (all-MiniLM-L6-v2)
- $k=8$ (standard practice, Dai et al., 2023)
- 每对 $(s_{t'}^{\text{lose}}, s_{t'}^{\text{win}})$ 至少有一个是 agent $a_i$ 提出的

Final winner selection（带 refinement）：
$$i^*(t) = \begin{cases} \arg\min_{i: \pi(a_i) < \pi(a_{\hat{i}(t)})} (C_{t,i}^r - V_{t,i}^r) & \text{if any refined bid satisfies } C_{t,i}^r - V_{t,i}^r < C_{t,\hat{i}(t)} - V_{t,\hat{i}(t)} \\ \hat{i}(t) & \text{otherwise} \end{cases}$$

注意上标 $r$ 表示 refined version。

---

## 4. 实验数据深度分析

### 4.1 HST-Bench 构造

| Complexity bin | Deep Search sources | Coding sources | # tasks |
|---|---|---|---|
| $\tau \leq 0.1$ min | SimpleQA (38%), PopQA (50%) | Coding-MCQ (100%) | 80+80 |
| $\tau \leq 0.5$ min | HotpotQA (13%), SimpleQA (8%), PopQA (5%) | MBPP (100%) | 80+79 |
| $\tau \leq 2.5$ min | HotpotQA (88%) | LeetCode Medium (1%), MBPP (99%) | 80+80 |
| $\tau \leq 12.5$ min | HLE (3%), HotpotQA (2%) | LeetCode Medium (98%) | 82+81 |
| $\tau \leq 60$ min | GAIA (98%), HLE (2%) | LeetCode Hard (100%) | 32+79 |

Total: 753 tasks。Inter-annotator agreement: Krippendorf's $\alpha = 0.86$。

### 4.2 主结果（Table 1）

**Deep Search:**

| Method | Pass@1 (All) | $/Mt (All) | vs Best Single |
|---|---|---|---|
| Best single agent (32B) | 63.8 | 0.36 | - |
| WTP | 62.4 | 0.32 | worse perf, slight savings |
| CARROT | 61.3 | 0.28 | worse perf, better savings |
| TO-Router | 63.0 | 0.33 | ~same perf, slight savings |
| FrugalGPT | 61.0 | 0.51 | worse perf, MORE cost! |
| SALE w/o memory | 66.4 | 0.24 | +2.6 pp, -33% cost |
| **SALE** | **67.3** | **0.21** | **+3.5 pp, -42% cost** |

**Coding:**

| Method | Pass@1 (All) | $/Mt (All) |
|---|---|---|
| Best single agent (32B) | 58.4 | 0.36 |
| WTP | 50.1 | 0.11 |
| CARROT | 57.1 | 0.31 |
| TO-Router | 58.4 | 0.36 |
| FrugalGPT | 50.4 | 0.57 |
| SALE w/o memory | 59.9 | 0.27 |
| **SALE** | **61.1** | **0.27** |

SALE 在两个 domain 都 **同时提高 accuracy 和降低 cost**——这非常罕见，通常 router 只能在两者之间 trade off。

### 4.3 Workload Allocation（Figure 4 + Table 2）

Deep Search 的 agent 分配（带 memory）：

| $\tau(t)$ | 4B | 8B | 14B | 32B |
|---|---|---|---|---|
| $\leq 0.1$ | 25.6% | 24.2% | 26.3% | 23.9% |
| $\leq 0.5$ | 24.1% | 25.4% | 25.2% | 24.8% |
| $\leq 2.5$ | 23.4% | 26.7% | 25.7% | 24.2% |
| $\leq 12.5$ | 20.0% | 22.6% | 27.3% | 30.0% |
| $\leq 60$ | 7.1% | 24.2% | 35.6% | 33.1% |

注意：即使在最高复杂度 bin，4B+8B 仍处理 ~31% 的任务！这说明 SALE 能识别出复杂任务中的"简单子任务"。

### 4.4 Shapley Value 分析（Table 2）

这是 paper 最 elegant 的分析之一。Shapley value 衡量每个 agent 的 marginal contribution：

$$\phi_i = \sum_{\mathcal{A}' \subseteq \mathcal{A} \setminus \{i\}} \frac{|\mathcal{A}'|! (|\mathcal{A}| - |\mathcal{A}'| - 1)!}{|\mathcal{A}|!} [\nu(\mathcal{A}' \cup \{i\}) - \nu(\mathcal{A}')]$$

关键发现：
- **Without memory**: 32B 的 Shapley value 最高（即使不总是被选中做 final inference）——因为它在 jury scoring 和 memory 中的贡献大
- **With memory**: 32B 的 Shapley value 下降，4B/8B 的上升——memory 让小 agent "scale up"

### 4.5 Complementary Failure Modes（Section 7.5 + Appendix K）

这是理解 SALE 为什么能超越任何单 agent 的关键。如果大 agent 的 error set 是小 agent error set 的子集，那 routing 只能省钱不能提精度。

Paper 发现 **failure modes 是 complementary 的**：

**32B 的失败模式：**
1. 过度自信——绕过 tools 直接用 parametric recall（Deep Search Example 1）
2. Over-engineering——把简单问题复杂化（Coding Example 6: 对 fixed two-level nesting 用递归）
3. 跳过 verification steps（Coding Example 8-9: 不用 Python shell 验证）
4. 忽略 edge cases（Coding Example 10: 没处理 "no duplicate" case）

**小 agent 的优势：**
1. 更依赖 tools（search, Python shell）
2. 更简单的 strategies
3. 更 explicit 的 verification steps

这些差异在 **initial strategic plan 阶段就可见**——这是 SALE 能在 bid time 就识别 routing signal 的根本原因。

---

## 5. Ablation Studies 深度

### 5.1 Cost-Value Function Ablation（Table 17, Appendix I）

| Configuration | Deep Search Pass@1 | Coding Pass@1 |
|---|---|---|
| Full SALE | 67.3 | 61.1 |
| w/o price $\pi(a_i)$ | 65.4 | 59.4 |
| w/o length $|s_{t,i}|$ | 66.0 | 58.7 |
| w/o entropy $H(s_{t,i})$ | 65.5 | 59.1 |
| w/o jury | 63.7 | 59.9 |

每个组件都有贡献。Jury 对 deep search 最关键（-3.6），cost terms 对 coding 最关键。

### 5.2 Jury Ablation（Tables 18-20）

- Full jury (4 judges) > 任何 single judge
- 移除任何 single judge 都降低 performance
- **Self-feedback only**: 64.1 (deep search) / 58.8 (coding)
- **Peer-feedback only**: 63.2 / 48.7

Peer-feedback 对 hard tasks 至关重要；self-feedback 对 easy tasks 提供 cheap signal。

### 5.3 Routing Diagnostics（Appendix H.2）

Confusion matrix 分析（SALE vs oracle）：

- **Over-escalation** 是主要 error mode：SALE 选 14B/32B 时，oracle 往往选 4B
- **Under-escalation** 很少：SALE 很少因选小 agent 而牺牲 accuracy
- 当 SALE 真的选 4B 时，diagonal accuracy 很高（54% deep search, 69.6% coding）

这告诉我们 SALE 是 **conservative router**——它宁可多用大模型也不冒险，这解释了为什么 accuracy 不降反升。

---

## 6. 我的 Intuition Building

### 6.1 为什么 plan 是好的 routing signal？

传统 router 基于 task description——但 agentic trajectory 是 long-horizon 的，task input 与 downstream success 之间的 mapping 是 indirect 的（mediated by 长 trajectory）。

Plan 是 **intermediate representation**：它 encode 了 agent 对 task 的理解、分解策略、tool selection、anticipated challenges。这些信息在 task description 里不存在，在 final trace 里太贵。

Sun et al. (2024) 的 correlation evidence + SALE 的 ablation 共同证明：**plan quality → execution quality** 是可靠的。

### 6.2 为什么 marketplace metaphor 比 traditional router 好？

传统 router 是 static mapping: task features → model。SALE 是 dynamic system：
1. Agents bid（竞争）
2. Jury scores（同行评审）
3. Memory accumulates（经验积累）
4. Cheaper agents learn from past winning strategies（upskilling）

这模拟了真实劳动力市场——freelancer 通过竞争和反馈提升技能。Figure 5 显示 4B 的 selection rate 从 3.7% 增长到 11.1%（deep search），从 1.4% 到 5.3%（coding）。

### 6.3 为什么 "scaling small agents" 而不是 "scaling up models"？

Paper 的核心 thesis：**capability 不是单个 model 的 property，而是 system 的 emergent property**。通过：
- Per-task routing（动态分配）
- Cost-value pricing（市场定价）
- Memory-based self-improvement（经验学习）

小 agent 能 "scale up"——不是通过增加参数，而是通过 coordination。

### 6.4 Limitations 的诚实评估

Paper 自己承认的限制：
1. 只测了 deep search 和 coding
2. 只用了 Qwen3 4B-32B（没测 70B+）
3. Cost model 只算 token，不算 tool call
4. Memory 线性增长，长期部署需要 management

这些是真实的局限，但 framework 是 model-agnostic 的，理论上可以扩展。

---

## 7. 相关工作与延伸阅读

- **Plan quality correlation**: [Sun et al., 2024](https://aclanthology.org/2024.lrec-main.521/) - distilling reasoning ability
- **Agent economies**: [Tomasev et al., 2025](https://arxiv.org/abs/2509.10147) - virtual agent economies
- **Human solution time as complexity metric**: [Kwa et al., 2025](https://openreview.net/forum?id=CGNJL6CeV0) - MEasuring AI ability
- **Entropy and reasoning**: [Wang et al., 2025c](https://openreview.net/forum?id=yfcpdY4gMP) - high-entropy minority tokens
- **LLM juries**: [Verga et al., 2024](https://arxiv.org/abs/2404.18784) - replacing judges with juries
- **Long-horizon execution**: [Sinha et al., 2025](https://arxiv.org/abs/2509.09677) - illusion of diminishing returns
- **FrugalGPT**: [Chen et al., 2024](https://openreview.net/forum?id=cSimKw5p6R)
- **RouterBench**: [Hu et al., 2024](https://openreview.net/forum?id=IVXmV8Uxwh)
- **ARE framework**: [Froger et al., 2025](https://arxiv.org/abs/2509.17158)
- **Qwen3**: [Yang et al., 2025a](https://arxiv.org/abs/2505.09388)

---

## 8. 总结性直觉

SALE 的 elegance 在于它同时解决了三个问题：
1. **Routing**: 用 plan 作为 cheap proxy for trajectory quality
2. **Self-improvement**: 用 auction memory 让小 agent 从 past winners 学习
3. **Cost-efficiency**: min-max optimization 确保 worst-case robustness

最让我印象深刻的是 **complementary failure modes** 的发现——大模型的 overconfidence 和小模型的 tool-dependence 形成了天然的 division of labor。这不是 SALE 发明的，但 SALE 是第一个系统利用这个现象的 framework。

从 Karpathy 的 systems thinking 角度，这篇 paper 最大的贡献可能是：**它把 agent coordination 从 "how to call tools" 提升到 "how to organize a market of agents"**。这预示了未来 agentic AI 的方向——进步来自 coordination mechanisms，而非单一 ever-larger models。

参考链接：
- Paper: [arXiv (估计)](https://arxiv.org/) - 搜索 "Scaling Small Agents Through Strategy Auctions"
- Meta AI: https://ai.meta.com/
- ARE: https://arxiv.org/abs/2509.17158
