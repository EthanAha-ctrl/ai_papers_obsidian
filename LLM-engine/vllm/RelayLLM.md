---
source_pdf: RelayLLM.pdf
paper_sha256: fc2a896a5c0e53ef14121b894ce00d5424744dab8f1e52015a526f9565d30f98
processed_at: '2026-08-11T22:41:00-07:00'
target_folder: LLM-engine/vllm
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RelayLLM 用人话说

## 一句话总结

让小模型自己学会"举手提问"，在推理卡壳的关键时刻精准召唤大模型帮忙写几句话，然后自己接着干。最终只用了 1% 的大模型 token，就补上了大小模型之间 60% 的性能差距。

---

## 为什么这事儿有意思

先说现状的荒谬之处。

传统的 router 做法像是一个门卫：看到一道题，先判断难不难，难就直接丢给大模型做完整道题。问题是，小模型其实能搞定一道题里 90% 的步骤——列方程、代入、化简都没问题，可能就在某一步换元的时候卡住了。结果你把整道题都丢给大模型，大模型从头到尾做一遍，90% 的算力都浪费在小模型本来就会的地方。

这就好比你请了个高中生做作业，遇到一道难题，他其实只是其中一步的积分技巧没学过，你却把整道题交给一个数学教授从头解到尾。教授的时间多贵啊。

RelayLLM 的想法是：**让高中生自己判断"我卡住了"，然后只请教授帮忙写那一步，写完高中生自己接着往下做。**

---

## 它怎么做到的

### 推理的时候

小模型正常一个 token 一个 token 地生成。但它在词表里多了两个特殊 token：`<call>` 和 `</call>`。当小模型觉得自己卡住了，它就会生成一段像 `<call> 300 </call>` 的命令，意思是"请大模型帮我生成 300 个 token"。

这时候系统暂停小模型，把当前的上下文喂给大模型。有个细节很关键：喂给大模型的时候，会把 `<call>` 这些命令 token 删掉，让大模型看到的是一个干净的上下文，仿佛这些话都是它自己说的。大模型生成完 300 个 token（或者提前遇到句号结束），控制权交回小模型，小模型接着往下写。

小模型这边会保留完整记录，包括自己发过的所有 `<call>` 命令，这样它知道"我刚才在哪里求助过"，不会在同一个地方反复举手。

### 训练的时候

难点在于：小模型天生不会生成 `<call>` 这种 token，直接上 RL 它根本不会探索到这个行为。所以分两步走。

**第一步，warm-up**。先让小模型学会"语法"。具体做法是从原始小模型采样一堆回答，然后在随机位置插入 `<call> n </call>` 命令，n 的大小从 1 到 9000 不等（用 $d \times 10^k$ 的方式采样，覆盖各个量级）。用这些数据做监督学习，让模型至少知道"这个命令长什么样、放在哪里"。

为什么要从小模型自己采样，而不是用外部数据？因为要避免分布偏移。如果用外部语料训练，小模型学到的是"在别人的文本风格下何时 call"，而不是"在我自己的生成分布下何时 call"。这一步的核心目的是教语法，不是教语义。

**第二步，RL 精调**。用 GRPO 算法（DeepSeek 那套），每个 query 采样 8 个回答，用组内统计代替 critic 网络算 advantage。reward 设计是这篇论文最有想法的地方。

---

## Reward 设计的直觉

最简单的 reward 是：答对了 +1，call 大模型的比例越高扣越多。但这有个问题：对所有题一视同仁，不管这道题小模型本来能不能做。

论文的做法是，先看 8 个采样里头的整体表现，把每道题分成三种情况：

**情况一：小模型自己就能做**。8 个采样里至少有一个不 call 大模型也答对了。这种题 call 大模型就是浪费。所以如果自己独立做对，给 1.5 分奖励（比正常多 0.5）；如果 call 了才做对，还是给正常分数但扣 call 成本；做错给 0。这鼓励小模型"能自己做就自己做"。

**情况二：必须 call 大模型才能做**。8 个采样里，做对的那些都是 call 了大模型的。这种题小模型硬做是做不出来的。如果小模型犟着不 call 还做错了，扣 1 分（惩罚倔强）；如果 call 了且做对了，给正常奖励。这教训小模型"别瞎猜，该问就问"。

**情况三：大模型也做不出来**。8 个采样全错。传统 RLVR 这时候给 0 reward，但这会导致一个问题：模型会学到"反正都做不对，干脆别 call 了"，calling behavior 会慢慢消失。论文给了一个很小的 reward $r = \rho(y)$，也就是 call 比例本身作为奖励。意思是"即使你做不对，至少你尝试了求助，这个行为本身值得鼓励"。这维持了在高度不确定情况下仍然愿意 call 的探索行为。

这三个场景加在一起，本质上是一个 curriculum：简单题教你独立，中等题教你求助，难题教你别放弃探索。

---

## 结果到底有多好

用 Qwen3-1.7B 当小模型，Qwen3-8B 当大模型，六个数学 benchmark 平均：

- 原始 1.7B：42.50%
- 标准 GRPO 训练后：44.06%
- RelayLLM：49.52%
- 8B 大模型本身：54.12%

RelayLLM 补上了 $(49.52 - 42.50) / (54.12 - 42.50) \approx 59\%$ 的差距。而大模型只生成了 1.07% 的 token。算一下成本：8B 模型单 token 成本大约是 1.7B 的 4.7 倍，所以等效成本只增加了 $0.0107 \times 4.7 \approx 5\%$。用 5% 的额外成本换 7 个百分点的准确率提升。

更让人意外的是两个结果：

**第一，跨领域泛化**。只在数学数据上训练，测试 MMLU-Pro（多学科选择题）从 49.76%（GRPO）涨到 59.03%。说明小模型学到的不是"数学上何时 call"这种 domain-specific 技能，而是一种元认知能力——"我什么时候不确定"。

**第二，把大模型撤掉之后**。推理时禁止小模型 call（用 bad_words 屏蔽 `<call>` token），在简单题上，RelayLLM 训练的模型仍然比标准 GRPO 训练的模型强（61.12% vs 59.51%）。这说明在 RL 训练过程中，大模型的输出隐式地蒸馏进了小模型的 policy，即使推理时没有大模型帮忙，小模型本身的推理能力也变强了。

---

## 和其他方法的区别

**和 speculative decoding 比**：speculative decoding 是用小模型生成、大模型并行验证，目的是加速。RelayLLM 是让小模型主动决定何时让大模型接管一小段，目的是提升质量。两者可以叠加，但解决的问题不同。

**和 router/cascading 比**：router 是 query 级别的二选一，粒度太粗。RelayLLM 是 token 级别的，而且不需要额外的 router 模型——小模型自己就是 router。

**和 CITER 比**：CITER 也是 token 级别 routing，但需要一个额外的 MLP 在每个 token 位置算分决定要不要切换，引入了额外延迟。RelayLLM 把这个决策能力 baked 进小模型本身，没有额外组件，而且效果还更好（49.52% vs 46.81%）。

**和 tool use 比**：`<call>` 本质上就是把大模型当成一个 tool 来调用，和 function calling 的范式一致。只不过传统 tool 返回结构化数据，这里返回的是自然语言续写。

---

## 一个具体的例子

论文里有个 case study 很直观。题目是：一个正整数列表，和为 30，唯一众数是 9，中位数是不在列表中的正整数，求平方和。

小模型开始分析条件，写到"第三步：试小的 n 值"时，它不确定怎么系统性地搜索合法列表，于是输出 `<call> 300 </call>`。大模型接管 300 个 token，开始试错：[9,9,1,1,10] 众数是 1 不对，[9,9,1,2,9] 中位数 1 在列表里不对，[9,9,1,3,8] 符合所有条件。然后大模型在 300 token 内完成验证并把控制权还回来。小模型自己算出 $9^2 + 9^2 + 1^2 + 3^2 + 8^2 = 236$。

注意这里的分工：小模型负责理解和最终计算，大模型负责中间的试错搜索。大模型没有做完整道题，只做了小模型卡住的那一段。

---

## 我觉得哪些地方还没讲透

**延迟问题没分析**。论文只讲了 token 成本，但实际部署中，从小模型切换到大模型需要大模型重新处理整个上下文的 KV cache，这个切换延迟可能不小。对于实时应用，延迟可能比 token 成本更重要。

**streaming 不友好**。生成到一半突然暂停等大模型，用户体验上需要特殊处理。

**只测了单一大模型当 teacher**。如果同时有数学专精、代码专精、知识专精的多个大模型，小模型能不能学会"在什么问题上 call 什么模型"？这个扩展论文没碰。

**理论解释缺位**。为什么 1.07% 就够？有没有 information-theoretic 的下界？论文靠实验说话，没给理论分析。

**warm-up 的 bootstrap 问题**。如果小模型很弱（比如 0.1B），它自己采样出来的回答质量很差，在垃圾回答上插入 `<call>` 训练，warm-up 数据的质量也受限。论文用的是 0.6B 和 1.7B，还没弱到那个程度，但更小的模型能不能用这套方法存疑。

---

## 我觉得最聪明的几个设计

**小模型自己当 controller**。不需要额外组件，决策能力和生成能力一起训练，end-to-end。

**上下文的非对称处理**。小模型保留 `<call>` 记录避免重复求助，大模型看到干净上下文避免分布不匹配。两边各取所需。

**difficulty-aware reward 的第三类**。给"大模型也做不对"的 query 一个小的 exploration reward，防止 calling behavior 在困难题上消失。这个细节看起来不起眼，但 ablation 显示去掉它准确率从 49.52 掉到 47.56，影响很大。

**动态 call 长度**。不是固定请求 100 个 token，而是让模型自己决定请求多少。和固定 100 token 的版本比，准确率差不多但成本只有三分之一。

---

## 更深一层的联想

这篇论文让我想到一个更大的问题：**reasoning 的"瓶颈"到底在哪里？**

如果 1% 的 token 就能补上 60% 的差距，说明小模型的推理能力其实不差，差的是某些"关键决策点"上的判断力。这些关键点可能是：换元技巧的选择、case 的枚举、某个定理的联想。大模型的价值不在于"会算"，而在于"见过更多 pattern，知道在这种情况该往哪个方向走"。

这和人类专家与新手的区别很像。新手大部分步骤都会做，只是在几个分叉路口不知道选哪条路。专家的价值就是在路口给出指引，而不是替新手走完全程。

从这个角度看，RelayLLM 不只是一个工程优化，它揭示了一个关于 reasoning 的结构性洞察：**推理能力不是均匀分布的，而是集中在少数关键节点上**。如果能精准识别这些节点，用最小的大模型介入去撬动它们，就能用极低的成本获得大部分的收益。

这个 insight 对未来的 model design、training strategy、甚至对人类教育（在关键节点给提示 vs 全程手把手教）都有启发。

参考：
- RelayLLM 代码: https://github.com/Chengsong-Huang/RelayLLM
- GRPO 原始论文: https://arxiv.org/abs/2402.03300
- CITER 对比方法: https://arxiv.org/abs/2502.01976
- Critical Tokens 研究: https://arxiv.org/abs/2411.19943
- Test-Time Compute Scaling: https://arxiv.org/abs/2408.03314

---

# RelayLLM 深度解析：Token-Level Collaborative Decoding

## 1. Core Intuition & Motivation

RelayLLM 的核心 insight 在于一个关键观察：**reasoning trajectory 中并非所有 token 都同等重要**。传统 router/cascading 方法（如 RouteLLM, HybridLLM）采用 query-level granularity，一旦判定 query 困难就把整个任务 offload 给 LLM，这造成巨大 computational waste。实际上 SLM 通常能处理 90%+ 的 reasoning steps，只在某些 "critical tokens" 处出现 reasoning gap。

这让我想到 **speculative decoding** 的思想（Leviathan et al. 2023）—— 那里 draft model 生成大部分 tokens，verify model 只验证关键位置。RelayLLM 在某种程度上是 speculative decoding 的 "semantic" 版本：SLM 自己决定何时 "delegate"，而非由 verifier 判定。这与 **mixture-of-experts (MoE)** 也有类似 spirit，只不过 MoE 是 parameter-level routing，RelayLLM 是 model-level 且 token-level 的 routing。

关键区别于 CITER（Zheng et al. 2025b）：CITER 需要一个 external MLP controller 在每个 token position 估计 score 决定是否切换模型，引入额外 latency。RelayLLM 让 SLM **内生**地学会 delegating，把 controller 能力 baked into SLM 的 policy 中。

参考链接：
- Speculative Decoding: https://arxiv.org/abs/2211.17192
- RouteLLM: https://arxiv.org/abs/2406.18665
- CITER: https://arxiv.org/abs/2502.01976
- Collaborative Decoding (Shen et al.): https://aclanthology.org/2024.acl-long.679/

---

## 2. Architecture 深度解析

### 2.1 Inference Pipeline

考虑 hybrid inference setting：SLM $\mathcal{M}_S$ + LLM $\mathcal{M}_L$。给定 input query $x$，目标是通过动态协作生成 response $y$。

**SLM Generation Phase**：
$\mathcal{M}_S$ 默认 autoregressively 生成 tokens。当 SLM 决定请求帮助时，生成 special command pattern：

$$\mathcal{C}_{cmd}(n) = \text{<call>} \oplus n \oplus \text{</call>}$$

- $\oplus$：string concatenation operator
- $n \in \mathbb{Z}^+$：请求 LLM 生成的 token 数量
- $\text{<call>}, \text{</call>}$：special tokens，added to SLM vocabulary

**LLM Intervention Phase**：
检测到 trigger pattern 后，SLM generation 暂停。关键设计 decision：**strip command tokens from context forwarded to LLM**。这是 distribution compatibility 的关键 —— LLM 在训练时从未见过 `<call>` tokens，保留它们会造成 train-test mismatch，破坏 LLM 的 generation quality。LLM 生成 next $n$ tokens（或提前 stop at [EOS]）。

**Iterative Relay**：
LLM 完成后，control 返回 SLM。Context 更新策略是 **asymmetric** 的：
- SLM 保留完整 history（包含自己生成的 `<call>n</call>`）—— 这让 SLM 维护 delegation 决策的 trace，避免反复在同一位置请求帮助
- LLM 看到的是 "clean" context，仿佛 SLM 的 tokens 是它自己生成的

这种设计让我想到 **tool-use agents** 的 paradigm（Wölflein et al. 2025），其中 `<call>` 类似 function calling 的 trigger token。但传统 tool use 中 tool 返回的是结构化结果（如搜索结果），这里 LLM 返回的是 natural language continuation。

### 2.2 为什么这个架构 work？

从 information-theoretic 角度思考：
- SLM 的 generation 是一个 conditional probability chain $p_{\mathcal{M}_S}(y_t | y_{<t}, x)$
- 在 reasoning critical points，SLM 的 entropy 高，sampling 容易 drift 到错误分支
- LLM intervention 等价于在这些 high-uncertainty positions 注入 high-confidence tokens，相当于一个 **contextual "reset" of the reasoning trajectory**

这与 **"Critical Tokens Matter"** (Lin et al. 2024) 的发现一致 —— 少数 critical tokens 决定 reasoning 的成败。RelayLLM 实证显示只需 1.07% 的 tokens 由 LLM 生成就能 bridge 60% 的 performance gap。

参考：
- Critical Tokens: https://arxiv.org/abs/2411.19943
- Tool Use LLMs: https://aclanthology.org/2025.acl-long.1390/

---

## 3. Training Framework 深度解析

### 3.1 Stage 1: Supervised Warm-up

**问题**：直接用 RL 训练，SLM 不会自然生成 `<call>` pattern（special tokens 在 pretraining 中从未见过，probability 接近 0）。需要 cold start。

**数据构造**（防止 distribution shift 是核心 design）：

1. **Self-sampling**：从 vanilla $\mathcal{M}_S$ 采样 base sequences $y$，而非用 external corpus。这确保 training context 与 SLM 的 own distribution 完美对齐 —— 否则 SLM 会学到 "在别人的 distribution 上何时 call"，而非 "在自己的 distribution 上何时 call"。

2. **Token-level random insertion**：在 random index $t$ 处插入 command tokens，而非 sentence/paragraph boundaries。这模拟 inference 时 reasoning gap 可能在任何位置出现的 reality。

3. **Variable delegation length**：sample $n_{sample} = d \times 10^k$，其中：
   - $d \in \{1, 2, ..., 9\}$：mantissa
   - $k \in \{0, 1, 2, 3\}$：order of magnitude，覆盖 1 到 9000 tokens
   - $n = \min(n_{sample}, L_{rem})$：clip 到 available response length $L_{rem}$

这种 logarithmic sampling 让模型见识从 1 token（"给我一个词的提示"）到 9000 tokens（"帮我写大段推理"）的全 spectrum delegation 需求。

**训练 loss**：standard cross-entropy loss over the constructed sequences。

**Caveat**（论文承认）：训练时 command 后的 tokens 是 SLM 自己生成的，inference 时是 LLM 生成的，存在 theoretical discrepancy。但 warm-up 的目的主要是 teach syntax 而非 semantics，这个 discrepancy 留给 RL stage 解决。

### 3.2 Stage 2: GRPO with RLVR

#### 3.2.1 GRPO Objective

公式 (1)：

$$\mathcal{I}_{GRPO}(\theta) = \mathbb{E}_{q \sim \mathcal{D}}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\mathcal{M}_i - \beta \mathbb{D}_{KL}\right)\right]$$

变量解释：
- $\theta$：SLM 的 policy parameters
- $q$：query from training distribution $\mathcal{D}$
- $G$：group size（论文中 $G=8$），每个 query 采样 8 个 outputs
- $\mathcal{M}_i$：surrogate objective for $i$-th sample
- $\beta = 0.01$：KL regularization coefficient，防止 policy 偏离 reference policy $\pi_{ref}$ 太远
- $\mathbb{D}_{KL} = D_{KL}(\pi_\theta \| \pi_{ref})$：current policy 与 reference policy 的 KL divergence

Surrogate objective：

$$\mathcal{M}_i = \min(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i)$$

- $\rho_i = \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$：importance sampling ratio，current policy $\pi_\theta$ 与 old policy $\pi_{\theta_{old}}$ 的 probability ratio
- $A_i$：advantage
- $\epsilon$：PPO clip parameter（防止 ratio 过大）

Advantage 计算（公式 2）：

$$A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\}) + \varepsilon_{stab}}$$

- $r_i$：$i$-th sample 的 reward
- $\{r_j\}$：group 内所有 samples 的 rewards
- $\varepsilon_{stab}$：small constant for numerical stability

GRPO 的精髓：**用 group statistics 替代 critic network**，避免了 PPO 中 critic 的 high-variance estimation 问题。这与 RLOO (Ahmadian et al. 2024) 思想类似。DeepSeek-R1 的成功证明了 GRPO 在 LLM reasoning 上的有效性。

#### 3.2.2 Data Filtering（关键 engineering trick）

预处理：对每个 query 采样 10 个 responses，只保留 pass rate $\geq 50\%$ 的 queries。

**Intuition**：如果 LLM 自己都解不了 query，calling LLM 是 pure waste。Filtering 掉这些 queries 确保 training signal 集中在 LLM 能帮助的 queries 上。Ablation（Table 3）显示：移除 filtering 导致 call ratio 从 1.07% 飙升到 3.30%（3倍），accuracy 反而下降到 48.76%。

这个 trick 让我想到 **RL 的 credit assignment 问题** —— 如果 reward signal 始终是 0（teacher 解不了），policy gradient 也是 0，但 exploration noise 会污染 policy。Filtering 是一种 prior knowledge injection。

#### 3.2.3 Reward Design（论文最有创意的部分）

**Simple Reward**（公式 3）：

$$r_{simple}(y) = \mathbb{1}(a = g) - \rho(y)$$

- $y$：response
- $a$：parsed final answer from $y$
- $g$：ground truth
- $\mathbb{1}(\cdot)$：indicator function
- $\rho(y) \in [0, 1]$：call ratio = LLM 生成 tokens / total response length

这是一个 linear combination：correctness bonus + cost penalty。但问题是它对所有 queries 一视同仁，忽略了 query difficulty 的 heterogeneity。

**Difficulty-Aware Reward**（核心创新）：

基于 group $\mathcal{G}$ 内 8 个 samples 的 collective performance，将每个 query 分类为三种 scenarios：

**Scenario 1: Student-Solvable**
- 识别条件：$\exists$ sample in $\mathcal{G}$ that answers correctly **without** calling LLM（即 $\rho(y) = 0$ 且 correct）
- 含义：SLM 自己有能力解决，calling LLM 是 redundant
- Reward 设计：
  - Independent success（$\rho(y) = 0$ 且 correct）：$r = 1.5$（boosted bonus 鼓励 independence）
  - Dependent success（$\rho(y) > 0$ 且 correct）：$r = r_{simple}(y)$（仍有 correctness reward 但受 cost penalty）
  - Incorrect：$r = 0$

**Scenario 2: Teacher-Dependent**
- 识别条件：correct answers 只出现在调用 LLM 的 samples 中
- 含义：SLM independent reasoning 不足，必须 call teacher
- Reward 设计：
  - Fail to call teacher（$\rho(y) = 0$）且 incorrect：$r = -1.0$（penalize stubbornness）
  - Effective call + correct：$r = r_{simple}(y)$
  - 其他 incorrect：$r = 0$

**Scenario 3: Teacher-Unsolvable**
- 识别条件：no sample in $\mathcal{G}$ yields correct answer
- 含义：query extremely difficult，或 teacher intervention 失败
- Reward 设计：$r = \rho(y)$（small exploration reward 鼓励在 uncertain 时 still try calling）

**Intuition building**：
这个 piecewise reward design 本质上是一个 **curriculum**：
- 对 easy queries：teach "你能做就自己做，别浪费资源"
- 对 medium queries：teach "你做不了就 ask，别瞎猜"
- 对 hard queries：teach "即使不知道，也要尝试 ask，preserve exploration behavior"

第三类特别 interesting —— 传统 RLVR 在 all-fail 时给 0 reward，导致 policy collapse（模型学会"反正都错，干脆不 call"）。RelayLLM 给 small positive reward 维持 calling behavior，类似于 **curiosity-driven exploration**（Dai et al. 2025 CDE）。

参考：
- GRPO / DeepSeekMath: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DAPO: https://arxiv.org/abs/2503.14476
- CDE (Curiosity-Driven Exploration): https://arxiv.org/abs/2509.09675
- RLOO: https://arxiv.org/abs/2402.14740

---

## 4. Experiments 深度分析

### 4.1 Setup

- Student models: Qwen3-0.6B, Qwen3-1.7B
- Teacher: Qwen3-8B（same family 保证 tokenizer/vocabulary consistency）
- Benchmarks: Minerva, MATH-500, GSM8K, Olympiad-Bench, AIME-2024, AIME-2025
- Training data: DAPO dataset
- Framework: EasyR1 (基于 verl)
- Teacher inference: vLLM
- Non-thinking mode（Qwen3 默认有 thinking mode，关掉以节省 compute）
- Judge: GPT-4o-mini for answer verification（处理 math expressions 的 semantic equivalence）
- AIME 用 avg@32（hard benchmarks 需要 sampling diversity），其他用 pass@1 greedy

Hyperparameters（Table 7）：
- Optimizer: AdamW, lr=$1 \times 10^{-6}$
- Weight decay: $1 \times 10^{-2}$
- Global batch size: 32
- Max prompt length: 4096, Max response length: 8192
- Temperature: 1.0 (rollout), 鼓励 exploration
- Group size $G=8$
- KL coefficient $\beta = 0.01$

### 4.2 Main Results (Table 1) 解读

**Qwen3-1.7B 系列**：

| Method | Minerva | MATH500 | GSM8K | Olympiad | AIME25 | AIME24 | Avg | Call Ratio |
|--------|---------|---------|-------|----------|--------|--------|-----|------------|
| Base | 33.82 | 74.60 | 82.64 | 43.11 | 8.75 | 12.08 | 42.50 | - |
| GRPO | 35.66 | 75.60 | 81.73 | 45.04 | 10.73 | 15.62 | 44.06 | - |
| CITER | 38.63 | 80.24 | 82.26 | 51.20 | 11.96 | 16.58 | 46.81 | 1.34% |
| RelayLLM (Simple) | 43.01 | 83.40 | 86.13 | 51.56 | 13.44 | 18.23 | 49.30 | 0.43% |
| RelayLLM (Diff-Aware) | 43.75 | 81.40 | 86.28 | 55.70 | 12.71 | 17.29 | 49.52 | 1.07% |
| Qwen3-8B (Teacher) | 48.16 | 83.20 | 93.63 | 56.89 | 17.92 | 24.90 | 54.12 | 100% |

**关键 observations**：

1. **Performance gap recovery**：Base 42.50% → RelayLLM 49.52% → Teacher 54.12%。RelayLLM 恢复了 $\frac{49.52 - 42.50}{54.12 - 42.50} = 58.9\%$ 的 gap。

2. **Cost efficiency**：1.07% call ratio 意味着平均每 1000 个生成 tokens 只有 ~11 个来自 LLM。如果 LLM inference cost 与 parameters 成正比（8B/1.7B ≈ 4.7x），effective cost 仅增加 $0.0107 \times 4.7 \approx 5\%$。

3. **vs CITER**：RelayLLM (49.52%) > CITER (46.81%)，且 call ratio 更低（1.07% vs 1.34%）。CITER 的 external MLP controller 在每 token 都要 forward 一次，引入 latency overhead；RelayLLM 只在 SLM 输出 `<call>` 时才 trigger，amortized overhead 接近 0。

4. **Simple vs Difficulty-Aware**：Diff-Aware 平均 accuracy 略高（49.52 vs 49.30），但 call ratio 高（1.07% vs 0.43%）。这表明 difficulty-aware signal 让模型在 hard queries 上更 willing to call，trading slight cost for accuracy。

5. **Per-benchmark 分析**：
   - Minerva（hard）：Base 33.82 → Diff-Aware 43.75（+9.93），巨大提升
   - AIME-2025（extreme hard）：Base 8.75 → Diff-Aware 12.71（+3.96），相对提升 45%
   - GSM8K（easy）：Base 82.64 → Diff-Aware 86.28（+3.64），提升相对小但绝对值可观
   - 这说明 RelayLLM 在 hard benchmarks 上收益更大，符合 intuition —— hard problems 更多 critical tokens 需要 LLM 介入

### 4.3 Out-of-Domain Generalization (Table 2)

训练只在 DAPO（math），测试在 BBEH, MMLU-Pro, SuperGPQA：

| Model | BBEH | MMLU-Pro | SuperGPQA |
|-------|------|----------|-----------|
| Qwen3-1.7B Base | 9.91 | 46.90 | 24.46 |
| GRPO | 10.89 | 49.76 | 26.01 |
| CITER | 11.67 | 53.38 | 28.25 |
| RelayLLM (Simple) | 12.67 | 58.76 | 29.85 |
| RelayLLM (Diff-Aware) | 12.46 | 59.03 | 29.93 |

MMLU-Pro 上 RelayLLM 比 GRPO 高 9.27 points，这是惊人的 generalization。Intuition：SLM 学到的不是 "math 上的 calling pattern"，而是 generalized "我什么时候 uncertain" 的 metacognitive ability。这与 metacognition / calibration 文献相关 —— 模型学会了 self-assessment。

### 4.4 Ablation Studies (Table 3)

| Method | Avg Acc | Call Ratio |
|--------|---------|------------|
| RelayLLM | 49.52 | 1.07% |
| w/o Data Filtering | 48.76 | 3.30% |
| w/o Indep. Incentive | 49.34 | 4.10% |
| w/o Explor. Reward | 47.56 | 0.65% |

解读：
- **Data Filtering**：移除后 call ratio 3x，accuracy 下降。证明 filtering 避免 wasteful calls。
- **Independence Incentive**（Scenario 1 的 1.5x bonus）：移除后 call ratio 从 1.07% 飙到 4.10%，但 accuracy 几乎不变（49.34）。这表明 independence bonus 主要控制 call frequency，不影响 accuracy ceiling。
- **Exploration Reward**（Scenario 3）：移除后 accuracy 显著下降（47.56），call ratio 也降到 0.65%。说明 exploration reward 维持了 hard queries 上的 calling behavior，缺失会导致 policy collapse 到 "never call"。

### 4.5 Teacher-Free Evaluation (Table 4) —— 最 intriguing 的结果

| Method | Easy | Hard |
|--------|------|------|
| GRPO Baseline | 59.51 | 13.18 |
| RelayLLM (Simple) Standard | 66.03 | 15.84 |
| RelayLLM (Simple) w/o Teacher | 61.12 | 13.13 |
| RelayLLM (Diff-Aware) Standard | 66.78 | 15.00 |
| RelayLLM (Diff-Aware) w/o Teacher | 60.26 | 11.93 |

**Teacher-Free 实现**：用 `bad_words=["<call>", "</call>"]` 在 inference 时禁止 calling。

关键发现：
- Easy benchmarks 上，w/o Teacher 的 RelayLLM (Simple) 仍有 61.12% > GRPO baseline 59.51%
- 这证明 **collaborative training 产生了 distillation effect**：SLM 在 RL 训练中 "internalize" 了 LLM 的 reasoning patterns，即使 inference 时没有 teacher，也比 standard GRPO 训练的 SLM 强
- Diff-Aware w/o Teacher (60.26%) < Simple w/o Teacher (61.12%)：因为 Diff-Aware 训练时更依赖 teacher，internalized independence 较少

这个结果让我联想到 **"self-distillation"** 和 **"policy distillation"** —— RL 训练过程中 LLM 的 outputs 作为 environment feedback，间接 shape 了 SLM 的 policy。

### 4.6 Dynamic vs Fixed Delegation Length (Table 5, 6)

| Method | Avg Acc | Call Ratio |
|--------|---------|------------|
| Fixed-20 | 49.41 | 1.32% |
| Fixed-100 | 49.56 | 2.87% |
| Fixed-500 | 51.17 | 5.37% |
| RelayLLM | 49.52 | 1.07% |

**关键 insight**：RelayLLM (1.07%) 与 Fixed-100 (2.87%) accuracy 相当，但 cost 仅 37%。Fixed-500 accuracy 最高但 cost 5x。

Table 6 的 per-benchmark breakdown 更细致：
- MATH500/GSM8K（easy）：RelayLLM 与 Fixed-100 几乎相同 accuracy，但 call ratio 极低（1.07% vs 2.87%）。说明 easy problems 只需 short intervention。
- Minerva（hard）：RelayLLM (43.75) > Fixed-20 (39.71)，说明 RelayLLM 学会了 "对 hard problems 请求更多 tokens"。
- Fixed-500 在 Minerva 上最高（44.49），但 cost 5x，diminishing returns。

这验证了 dynamic delegation 的 value：**"just enough" tokens per query**。

### 4.7 Teacher Size Scaling (Figure 3)

Inference 时替换 teacher：
- None（teacher-free）：accuracy 显著下降
- 0.6B teacher（比 student 1.7B 还小）：仍 > None baseline
- 1.7B teacher：进一步改善
- 8B teacher（训练时用的）：peak accuracy
- 14B teacher：反而下降！

**关键 insight**：14B teacher 性能下降是因为 **distribution shift**，而非 capability 不足。SLM 在训练时适应了 8B teacher 的 generation style（vocabulary distribution, reasoning patterns），换成 14B 时 mismatch 抵消了 capability gain。这启示：**collaborative decoding 需要 teacher-student distribution alignment**，同 family models 是 natural choice。

这也让我想到 **"distillation compatibility"** 问题 —— 训练时 student 学到的不仅是 "what to call"，还有 "how to integrate called content"，后者与 teacher style 强相关。

---

## 5. Case Study 深度阅读 (Figure 4)

Problem: positive integers list，sum=30，unique mode=9，median 是 positive integer 但不在 list 中。求 sum of squares。

SLM 推理过程：
1. SLM 开始分析 properties（mode, median, sum）
2. 在 "Step 3: Try small values of n" 处，SLM 不确定如何系统搜索，生成 `<call> 300 </call>`
3. LLM 接管 300 tokens，给出 detailed trial-and-error：
   - Try [9, 9, 1, 1, 10] → mode 是 1，错
   - Try [9, 9, 1, 2, 9] → median 是 1，在 list 中，错
   - Try [9, 9, 1, 3, 8] → sum=30, mode=9, median=3 不在 list，对！
4. LLM 在 300 tokens 内完成验证并返回控制
5. SLM 接管，compute sum of squares = 236

这个 case 展示了 SLM 的 **strategic delegation**：它能识别 "我现在卡在 systematic search 上"，请求 LLM 帮忙验证 hypotheses，然后自己完成 final computation。这是典型的 **"exploration vs verification"** 分工。

---

## 6. Broader Context & Connections

### 6.1 与 Speculative Decoding 的对比

Speculative decoding（Leviathan 2023, Chen 2023）：
- Draft model 生成 multiple tokens
- Target model verify in parallel
- Accept prefix where draft matches target

RelayLLM 与之的 differences：
- Speculative decoding 是 **speed optimization**（parallel verification）
- RelayLLM 是 **quality optimization**（delegation at semantic level）
- Speculative decoding 需要 draft ≈ target distribution，RelayLLM 允许更大 gap
- Speculative decoding 的 "accept/reject" 是 token-level，RelayLLM 的 "call" 是 semantic-level decision

潜在 hybrid：能否用 speculative decoding 加速 RelayLLM 的 LLM intervention？比如 LLM 生成时也用更大的 model 作为 verifier？这是 future direction。

参考：https://arxiv.org/abs/2211.17192

### 6.2 与 MoE (Mixture of Experts) 的对比

MoE 在 transformer FFN 层做 routing，每个 token 选 top-k experts。RelayLLM 在 model level 做 routing，每个 "reasoning segment" 选 SLM or LLM。

Differences：
- MoE 是 parameter-level，experts 是同 architecture 的 sub-networks
- RelayLLM 是 model-level，"experts" 是 fully separate models
- MoE 的 router 是 learned shallow network，RelayLLM 的 "router" 是 SLM itself（end-to-end learned）
- MoE 的 routing 是 dense（每 token 都 route），RelayLLM 是 sparse（大部分 token SLM 自己处理）

### 6.3 与 Tool Use Agents 的关系

RelayLLM 的 `<call>n</call>` 本质是 tool use，tool 就是 LLM。这与 function calling paradigm 一致。

Connections：
- Tool use 通常 tool 返回结构化数据，RelayLLM 的 tool 返回 natural language continuation
- Tool use 的 trigger 是 learned，RelayLLM 的 trigger 也是 learned
- Tool use 通常多个 tools，RelayLLM 目前只有 1 个 tool（LLM），但可扩展到 multiple LLMs / specialized models

Potential extension：multi-tool RelayLLM，SLM 可以 call LLM-math, LLM-code, LLM-knowledge 等不同 specialized LLMs。

参考：https://aclanthology.org/2025.acl-long.1390

### 6.4 与 Self-Play / Multi-Agent RL 的关系

RelayLLM 训练时 SLM 与 LLM 交互，但 LLM 是 frozen 的。这与 self-play（如 SPIRAL, R-Zero）不同 —— self-play 中两个 agents 都 update。

Potential variant：mutual training，LLM 也从 SLM 的 mistakes 学习？但 paper 中 LLM frozen 是合理的 —— LLM 已经 strong，不需要 update，且 update LLM 成本高。

参考：
- SPIRAL: https://arxiv.org/abs/2506.24119
- R-Zero: https://arxiv.org/abs/2508.05004

### 6.5 与 Process Reward Models (PRMs) 的关系

PRM 给每个 reasoning step 打分，可用于 best-of-N selection 或 guided decoding。RelayLLM 的 SLM 本质上是 implicit PRM —— 它的 `<call>` decision 隐式标注了 "low confidence" positions。

Differences：
- PRM 是 external model，RelayLLM 的 "PRM" 是 SLM 内生
- PRM 给 dense per-step scores，RelayLLM 给 sparse binary decisions
- PRM 需要额外 inference cost，RelayLLM 的 cost 是 zero（baked into SLM）

Potential hybrid：用 PRM 训练 RelayLLM 的 reward signal？但 RLVR 已经用 verifiable rewards，PRM 可能 redundant。

参考：https://arxiv.org/abs/2306.09116 (OpenAI PRM800k)

### 6.6 与 Test-Time Compute Scaling 的关系

RelayLLM 是 test-time compute 的一种形式 —— 通过动态调用 LLM 增加 inference-time compute。这与：
- Best-of-N sampling
- Self-consistency
- Self-refine / self-correction
- Tree search (ToT, MCTS)

属于同一 family。但 RelayLLM 的独特之处：**external compute augmentation**，而非仅 self-based。这与 **"Scaling test-time compute"** (Snell et al. 2024) 的 framework 一致 —— adaptive compute allocation per query。

参考：https://arxiv.org/abs/2408.03314

---

## 7. Critique & Open Questions

### 7.1 Limitations 论文未充分讨论

1. **Latency analysis 缺失**：paper 强调 cost reduction（token ratio），但没分析 end-to-end latency。Switching from SLM to LLM 需要 KV cache 处理（LLM 要重新 process context），可能引入 significant latency。对于实时应用，这比 cost 更重要。

2. **Streaming 不友好**：RelayLLM 的 "pause-invoke-resume" 模式在 streaming setting 下不友好。用户看到 SLM 输出，突然 pause，等 LLM 生成，再恢复。UX 设计 challenge。

3. **Single teacher assumption**：只测试了 1 个 teacher（Qwen3-8B）。Multiple teachers 的情况？SLM 能否学会 "在 code 问题上 call code-specialist，在 math 问题上 call math-specialist"？

4. **Cold start data dependency**：Warm-up phase 需要从 vanilla SLM 采样，如果 SLM 本身很弱（如 0.1B），采样的 sequences quality 低，warm-up data quality 也低。这是 bootstrap 问题。

5. **Generalization boundary**：OOD 实验（Table 2）只在 reasoning tasks，没测试在 creative writing, dialogue, code 等 more diverse domains。Math reasoning 的 verifiable reward 特殊，其他 domain 的 reward 设计是 open question。

### 7.2 Theoretical Questions

1. **Optimality of 1.07%**：这个 call ratio 是 optimal 还是 training artifact？理论上 lower bound 是多少？能否 derive information-theoretic lower bound on necessary delegation tokens？

2. **Convergence properties**：GRPO 在 piecewise reward 下的 convergence guarantees？Scenario classification 是 discrete，可能 introduce non-smooth reward landscape。

3. **Credit assignment**：当 LLM 生成 n tokens 后 SLM 继续，最终 answer correct，credit 如何分配？是 LLM 的 tokens 贡献大还是 SLM 的后续 reasoning？Reward 给整个 sequence，可能 over-credit LLM 或 under-credit SLM。

### 7.3 Potential Extensions

1. **Multi-turn delegation**：SLM 可以在 LLM 生成后立即再 call（连续 delegations），paper 中似乎有但没 explicit analyze。

2. **Delegation with context pruning**：LLM 接收的 context 是 SLM 生成的全部 history，如果 history 很长（8000+ tokens），LLM inference cost 高。能否 prune context for LLM？

3. **Bidirectional training**：能否同时 update SLM 和 LLM？类似 DAgger (Dataset Aggregation) 的 idea，让 LLM 适应 SLM 的 distribution。

4. **Hierarchical relay**：0.6B → 1.7B → 8B → 70B，多级 delegation？SLM 学会 "我搞不定就 call 1.7B，1.7B 搞不定就 call 8B..."

5. **Reward shaping via LLM feedback**：除了 generation，LLM 能否作为 reward shaper，给 SLM 的 intermediate steps 提供 dense feedback？

---

## 8. Reproducibility Notes

代码：https://github.com/Chengsong-Huang/RelayLLM

Key implementation details：
- EasyR1 framework: https://github.com/hiyouga/EasyR1
- vLLM for teacher: https://github.com/vllm-project/vllm
- Switching mechanism: 实现为 stop sequence in sampling parameters，generate `<call>` 时 halt，invoke teacher API
- Bad words filtering for teacher-free: `bad_words=["<call>", "</call>"]`

Reproduction challenges：
- 需要 multi-GPU setup（SLM + LLM 同时 in memory）
- vLLM API serving 增加工程复杂度
- GPT-4o-mini judge 需 API cost
- Training time：paper 未报告 wall clock，但 GRPO with 8B teacher inference per rollout 应该慢

---

## 9. Final Thoughts

RelayLLM 的核心 contribution 在我的 view 是：**把 "model selection" 从 query-level coarse decision 变成 token-level fine-grained learned policy**。这个思想有深远的 implications：

1. **For deployment**：production 系统可以让 cheap model 处理 99% traffic，只在 critical moments invoke expensive model，cost reduction 巨大。

2. **For alignment**：SLM 学会 "ask for help" 是一种 honesty / humility 表现。与 "sycophancy"（过度自信）相反，RelayLLM 训练 SLM 知道自己不知道。

3. **For scaling laws**：传统 scaling laws 假设 single model。RelayLLM 暗示 **"heterogeneous model scaling"** —— 不同 capability levels models 协作 —— 可能比单纯 scale single model 更 efficient。

4. **For AGI**：人类 reasoning 也是 "internal cheap thinking + occasional external consultation"。RelayLLM 是这种 cognitive architecture 的 computational analog。

我特别 appreciate 的几个 design choices：
- **SLM 作为 controller**（而非 external router）—— elegant，避免 extra component
- **Asymmetric context handling**（SLM 保留 commands，LLM 看 clean context）—— 工程上 sound
- **Difficulty-Aware Reward 的 piecewise design** —— 比 simple linear reward 更 align with optimal policy
- **Exploration reward for unsolvable queries** —— 避免 policy collapse，crucial for stability

不太 convince 的地方：
- **Theoretical justification 缺失**：为什么 1.07% 就够？information-theoretic analysis 缺失
- **Latency 没分析**：production 可用性存疑
- **Single teacher**：multi-teacher 场景更 realistic

总体而言，RelayLLM 是 collaborative decoding 领域的 solid contribution，token-level granularity + learned delegation policy + carefully designed RL reward 三者结合得 coherent。我 expect 这个方向会有很多 follow-up works，特别是 multi-teacher, hierarchical relay, latency optimization 等方向。

---

## Key References

1. **RelayLLM (this paper)**: https://github.com/Chengsong-Huang/RelayLLM
2. **GRPO (DeepSeekMath)**: https://arxiv.org/abs/2402.03300
3. **DeepSeek-R1 (RLVR)**: https://arxiv.org/abs/2501.12948
4. **DAPO**: https://arxiv.org/abs/2503.14476
5. **CITER (baseline)**: https://arxiv.org/abs/2502.01976
6. **Collaborative Decoding (Shen et al.)**: https://aclanthology.org/2024.acl-long.679/
7. **Speculative Decoding**: https://arxiv.org/abs/2211.17192
8. **RouteLLM**: https://arxiv.org/abs/2406.18665
9. **Critical Tokens**: https://arxiv.org/abs/2411.19943
10. **Qwen3 Technical Report**: https://arxiv.org/abs/2505.09388
11. **EasyR1 Framework**: https://github.com/hiyouga/EasyR1
12. **vLLM**: https://arxiv.org/abs/2309.06180
13. **CDE (Curiosity-Driven Exploration)**: https://arxiv.org/abs/2509.09675
14. **Scaling Test-Time Compute**: https://arxiv.org/abs/2408.03314
15. **Self-Play (SPIRAL)**: https://arxiv.org/abs/2506.24119
