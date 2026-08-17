---
source_pdf: Learning Query-Specific Rubrics from Human Preferences.pdf
paper_sha256: 8536f5654e901ebc81b4b8ecffc4bb454a94ffe54d88d67439f70bb7b4ddc5a2
processed_at: '2026-08-05T13:37:28-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇 paper 解决的问题非常直白：让 AI 写长篇研究报告（DeepResearch report generation），怎么给它反馈？

做数学题可以看答案对不对，这叫 verifiable reward。写报告没有标准答案，很难自动打分。如果完全靠人工看，太贵没法 scale。现在业界常见的折中方案是用 rubric（评分标准）来打分。但现有做法都有硬伤：用一套死板的、通用的 rubric 去套所有报告，效果极差，Table 1 显示其判断人类偏好的准确率只有 48.78%，跟抛硬币差不多；找专家针对每个问题手写 rubric，又贵又慢；让大模型自己生成 rubric，模型自己的想法跟真实人类的偏好经常脱节，容易钻空子。

这篇 paper 的核心思路：把“制定 rubric”这件事本身当成一个可以通过 RL 学习的技能。让模型通过练习，学会针对不同 query，自动制定出符合人类口味、能把好报告和差报告明显区分开的评分标准。

---

### 怎么教模型制定 rubric

先收集数据。作者搞了 5000 多个适合 DeepResearch 的复杂 query，用 DeepSeek V3.1 和 Tongyi-DeepResearch 跑出多份报告。请了 16 个硕士以上的专家做 pairwise comparison，针对同一个 query 的两份报告，挑出更好的那份，标成 accepted ($r_{\mathrm{acc}}$)，差的标成 rejected ($r_{\mathrm{rej}}$)。这就有了人类偏好数据集。

接着用 GRPO 训练 rubric generator。关键在于 reward 设计，公式如下：

$$R_{\mathrm{total}} = \lambda_{\mathrm{pref}} R_{\mathrm{pref}} + \lambda_{\mathrm{llm}} R_{\mathrm{llm}} + R_{\mathrm{fmt}}$$

变量解释：
- $R_{\mathrm{total}}$ 是总奖励。
- $\lambda_{\mathrm{pref}}$ 和 $\lambda_{\mathrm{llm}}$ 是权重系数，paper 里默认都设为 1。
- $R_{\mathrm{pref}}$ 是 preference consistency reward。检查生成的 rubric 打分能不能把人类喜欢的报告排前面。
- $R_{\mathrm{llm}}$ 是 LLM-as-a-Judge reward。让大模型评估生成的 rubric 本身写得好不好，逻辑通不通，全不全。
- $R_{\mathrm{fmt}}$ 是 format reward。检查 rubric 是不是合法的 JSON 格式，能不能被程序解析。

计算一份报告在 rubric 下得多少分的公式是：

$$S(r \mid y) = \frac{\sum_{k=1}^{K} w_k \cdot v_k}{\sum_{k=1}^{K} w_k}$$

变量解释：
- $S(r \mid y)$ 是报告 $r$ 在 rubric $y$ 下的加权总分。
- $K$ 是 rubric 里条目的总数（比如 7 到 20 个）。
- $w_k$ 是第 $k$ 个 rubric 条目的权重（比如核心要求设为 5，错误项设为 -2）。
- $v_k$ 是 judge LLM 给报告在第 $k$ 个条目上的符合度打分（1-10 分，归一化到 0-1）。

如果 $S(r_{\mathrm{acc}} \mid y) > S(r_{\mathrm{rej}} \mid y)$，说明这套 rubric 成功区分了人类偏好，$R_{\mathrm{pref}}$ 给 +1；否则说明区分错了，给 -1。这个二值信号直接逼着模型学出有区分度的 rubric。

---

### 写报告的流程：MaMs Workflow

为了处理长文本，作者搞了个 Multi-agent Markov-state (MaMs) workflow。传统的 ReAct 把所有历史搜索记录和思考过程全塞进 context window，到后期模型就晕了，注意力分散。MaMs 维护一个精简的 Markov 状态：

$$s_t = \langle m_t, p_t, r_t \rangle$$

变量解释：
- $s_t$ 是第 $t$ 轮交互的完整状态。
- $m_t$ 是 structured memory（结构化记忆），存着提炼出来的关键事实。
- $p_t$ 是 execution plan（执行计划），一个 Markdown list。
- $r_t$ 是 incrementally evolving report（不断更新的报告）。

这个状态由三个共享同一个底层模型的 Agent 协作维护：
1. Search Agent：看当前状态，决定下一步搜什么，输出动作 $a_t$。
2. State Agent：搜回来的长文本 $O_t$ 太长，切成小块 $\{c_1, \ldots, c_K\}$。逐块处理，提炼出新事实，增量更新到 memory $m_t$ 里。Prompt 里死命令要求“旧记忆不能丢，必须是高密度的事实堆砌”。
3. Report Agent：根据新进来的事实块，增量修改报告 $r_t$，该补充补充，该纠正纠正。

这种解耦让信息压缩和叙事生成分开，减少了 single-pass 写长报告的幻觉风险。

---

### 实验数据与直觉

Table 1 验证了 rubric generator 的效果。通用的死板 rubric 准确率 48.78%，Cohen's d 只有 0.192。用 SFT 在 GPT-5 生成的 rubric 上训练，准确率 59.58%，d 值 0.260。用 RL with Hybrid Reward 训练，准确率冲到 65.68%，d 值达到 0.376。这说明 RL 从人类偏好中确实学到了把好报告和差报告拉开差距的能力，SFT 只能模仿表面格式，学不到内在的 discriminative power。

Table 2 是 DeepResearch Bench 的成绩。在 Tongyi-DeepResearch backbone 上，用 ReAct 加上 RL 训出的 rubric，Overall 分数 45.2。换成 MaMs workflow 加上 RL rubric，分数达到 49.3。这个成绩逼近 Gemini DeepResearch 的 49.7，把 OpenAI DeepResearch (46.5) 和 Claude Research (45.0) 都甩在后面，是开源模型里的 SOTA。

另外 paper 讨论了 GRPO 和 GSPO 的区别。GSPO 做 sequence-level 的优化，虽然 reward 差不多，但生成的 entropy 特别高， rubric 变得五花八门。Rubric 生成是个 mode-seeking 的任务，需要稳定一致，GRPO 的 token-level 优化刚好合适，最终在 Table 5 里 GRPO 的准确率 65.68% 也确实高于 GSPO 的 62.02%。

**Reference Links:**
- DeepResearch Bench: https://arxiv.org/abs/2506.11763
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- ReAct: https://arxiv.org/abs/2210.03629
- GSPO: https://arxiv.org/abs/2507.18071
- MemAgent (灵感来源): https://arxiv.org/abs/2507.02259
- Rubrics as Rewards: https://arxiv.org/abs/2507.17746
- DRTulu: https://arxiv.org/abs/2511.19399

---

# Learning Query-Specific Rubrics from Human Preferences for DeepResearch Report Generation 深度解析

你好 Andrej，这篇 paper 解决了一个非常实际的问题：**DeepResearch report generation 缺乏 verifiable reward signal**。让我从 motivation → method → architecture → experiments → intuition 逐层剖析。

---

## 1. 核心问题的本质

DeepResearch 系统（OpenAI DeepResearch, Gemini DeepResearch, Claude Research, Tongyi DeepResearch）要生成长篇分析报告，与 short-form QA（BrowseComp, GAIA, HLE）有本质区别：

- **Short-form QA**: 答案可验证 → RLVR (Reinforcement Learning with Verifiable Rewards) 直接适用，如 Search-R1, WebExplorer
- **Long-form report generation**: 无 reference answer，correctness 无法 reduce 到 verifiable target

现有 rubric-based evaluation 的三个痛点：
1. **Pre-defined generic rubrics** (如 Yao et al. 2025): 缺乏 granularity，Table 1 显示 Pref. Acc. 仅 48.78%，近乎随机
2. **Manually constructed query-specific rubrics** (如 ResearchRubrics): 每个查询需要 expert 投入大量 effort，不可 scale
3. **LLM-generated query-specific rubrics**: 不 grounded 在 human preference 上，容易 reward hacking

Paper 的核心 insight：**与其设计 rubric，不如学习生成 rubric**，且用 human preference 作为 supervision signal。

参考链接：
- BrowseComp: https://arxiv.org/abs/2504.12516
- GAIA: https://arxiv.org/abs/2311.12983
- ResearchRubrics: https://arxiv.org/abs/2511.07685
- Rubrics as Rewards (Gunjal et al.): https://arxiv.org/abs/2507.17746

---

## 2. Pipeline 三阶段架构

Figure 1 展示了完整 pipeline，分三个 stage：

### Stage 1: Query Construction
- 从 knowledge graph 采样 multi-hop entity paths
- LLM 合成 natural-language questions
- GPT-5 rewrite 以增加 phrasing diversity

数据集 $\mathcal{Q} = \{q_i\}_{i=1}^N$，$N > 5000$，覆盖 Law, Business, Science, Health 等领域（Figure 2）

**Intuition**: 为什么用 knowledge graph multi-hop paths 而非直接随机生成？因为 multi-hop 确保了 query 需要 reasoning across multiple facts，这是 DeepResearch 的本质特征。单跳 query 用 RAG 就够了。

### Stage 2: Candidate Report Generation via MaMs
- 用 DeepSeek V3.1 和 Tongyi-DeepResearch，varying hyperparameters 生成多个 candidate
- Human + LLM verifier 过滤，保留 best 2 个

### Stage 3: Human Preference Annotation
- 16 位 master's degree+ expert 做 pairwise comparison
- 考虑 usefulness, coherence, completeness, alignment
- 输出三元组 $(q, r_{\mathrm{acc}}, r_{\mathrm{rej}})$

**关键设计选择**: pairwise 而非 absolute rating。原因在于 human 对相对比较的判断比绝对打分更可靠（这是 reward modeling 的共识，参考 InstructGPT, RLHF literature）。

---

## 3. Rubric Generator 训练：Hybrid Reward 详解

这是 paper 的核心技术贡献。用 GRPO 训练 rubric generator，reward 设计是关键。

### 3.1 整体 Reward 公式 (Equation 1)

$$R_{\mathrm{total}} = \lambda_{\mathrm{pref}} R_{\mathrm{pref}} + \lambda_{\mathrm{llm}} R_{\mathrm{llm}} + R_{\mathrm{fmt}}$$

变量解释：
- $\lambda_{\mathrm{pref}}, \lambda_{\mathrm{llm}}$: 权重系数，paper 中默认都设为 1（Appendix D）
- $R_{\mathrm{pref}}$: preference consistency reward，binary signal，grounded in human data
- $R_{\mathrm{llm}}$: LLM-as-a-Judge reward，continuous [0, 4]，评估 rubric 本身的质量
- $R_{\mathrm{fmt}}$: format reward，hard constraint，JSON schema 合规性检查

**Intuition on hybrid design**: 这三个 reward 互补，各自解决不同的 failure mode：
- 只有 $R_{\mathrm{pref}}$: 可能产生能区分但语义无意义的 rubric（比如利用 spurious correlation）
- 只有 $R_{\mathrm{llm}}$: 可能与 human preference 不对齐（LLM judge 有自己的 bias）
- $R_{\mathrm{fmt}}$: 确保下游 pipeline 可解析

### 3.2 Report Scoring 公式 (Equation 2)

$$S(r \mid y) = \frac{\sum_{k=1}^{K} w_k \cdot v_k}{\sum_{k=1}^{K} w_k}$$

变量解释：
- $y$: 生成的 rubric（一个 JSON list）
- $w_k$: 第 $k$ 个 rubric item 的 weight（Key=5, Important=3-4, Optional=1-2, Error=-1或-2）
- $v_k$: judge LLM (Qwen3-235B-A22B) 对 report $r$ 在第 $k$ 个 item 上的 conformity score
- $v_k$ 原始范围 1-10 Likert scale，linearly normalized 到 [0, 1]
- $K$: rubric items 总数，7-20 之间

**关键细节**: Error criterion 的 weight 是负的（-1 或 -2），这意味着如果 report 触犯了 error criterion（如 "Technical Errors"），会直接拉低总分。这类似于考试中的扣分题。

### 3.3 Preference Consistency Reward (Equation 3)

$$R_{\mathrm{pref}}(y) = \begin{cases} +1, & \text{if } S(r_{\mathrm{acc}} \mid y) > S(r_{\mathrm{rej}} \mid y) \\ -1, & \text{otherwise} \end{cases}$$

这是一个 **binary ranking reward**：rubric 生成的评分是否能正确排序 human preference。

**Intuition**: 为什么用 binary +1/-1 而非连续 margin reward（如 $S(r_{\mathrm{acc}}) - S(r_{\mathrm{rej}})$）？
- Binary reward 更 robust，避免 model 通过夸大 margin 来 hack reward
- 连续 margin reward 容易导致 model 学到极端权重分配
- Binary 信号足以指导 GRPO 的 group-relative advantage

### 3.4 LLM-as-a-Judge Reward

$R_{\mathrm{llm}} = \mathrm{Judge}(q, y)$，范围 [0, 4]，评估 rubric 的：
- Logical coherence
- Coverage comprehensiveness  
- Relevance of evaluation dimensions

Prompt 在 Appendix C.3，要求输出 reward/confidence/reason 三行。

### 3.5 Format Reward

JSON schema compliance check：
- 必须有 title, description, weight 三个字段
- description 必须以 "Key Criterion:" / "Important Criterion:" / "Optional Criterion:" / "Error Criterion:" 开头
- 失败 = -1，成功 = 0

---

## 4. GRPO 训练机制

Rubric generator 用 GRPO (Group Relative Policy Optimization) 训练。参考 DeepSeekMath: https://arxiv.org/abs/2402.03300

### GRPO 核心思想
给定 query $q$，policy $\pi_\theta$ 采样一组 rubric candidates $\{y_1, y_2, \ldots, y_G\}$，$G=8$（Table 3）。

Group-relative advantage:
$$A_i = \frac{R_{\mathrm{total}}(y_i) - \mathrm{mean}(R_{\mathrm{total}})}{\mathrm{std}(R_{\mathrm{total}})}$$

这避免了训练独立的 value model，降低了计算成本。

### Hyperparameters (Table 3)
- Backbone: Qwen3-30B-A3B (MoE)
- Learning rate: $1 \times 10^{-6}$
- Group size G: 8
- KL coefficient: 0.0 (无 KL 约束！)
- Clip ratio $\epsilon$: 0.2
- Max response length: 8,192 tokens
- Temperature: 1.0, top-p: 1.0

**注意 KL=0**: Paper 选择不加 KL penalty，这可能是因为 hybrid reward 本身提供了足够的正则化（format constraint + LLM judge），且 rubric 生成任务不需要保持与 base model 的接近性。

---

## 5. MaMs (Multi-agent Markov-state) Workflow

这是第二个核心贡献，解决 ReAct 的 long-context dependency 问题。

### 5.1 状态抽象

状态定义：
$$s_t = \langle m_t, p_t, r_t \rangle$$

- $m_t$: structured memory（累积的关键事实）
- $p_t$: dynamic execution plan（Markdown list）
- $r_t$: incrementally evolving report

状态转移：
$$s_{t+1} = \tau(s_t, a_t)$$

**Markov 性的关键**: 传统 ReAct 把整个 trajectory history 塞进 context，违反 Markov 假设。MaMs 用 compact state abstraction $(m_t, p_t, r_t)$ 替代 raw history，使决策只依赖当前 state。

### 5.2 三 Agent 架构

虽然叫 "multi-agent"，但实际上**三个 agent 共享同一个 policy $\pi_\theta$**，只是 prompt 不同。Paper 明确说："MaMs can also be viewed as a structured single-agent formulation with modularized behaviors"。

#### Search Agent (高层控制器)
$$a_t, p_t' = \mathcal{A}_{\mathrm{search}}(q, s_t)$$

输入: $(q, m_t, p_t, r_t)$
输出: search action $a_t$ + refined plan $p_t'$
职责: 识别 information gap，决定下一步搜索或终止

#### State Agent (信息压缩器)
处理长 observation $O_t$，chunk-based 增量更新：

$$m_{t,k}, p_{t,k} = \mathcal{A}_{\mathrm{state}}(q, c_k, m_{t,k-1}, p_{t,k-1})$$

初始化: $m_{t,0} = m_t$, $p_{t,0} = p_t'$
最终: $m_{t+1} = m_{t,K}$, $p_{t+1} = p_{t,K}$

**Chunk-based 处理的 motivation**: $O_t$（搜索返回的长文本）常超过 LLM context window。Text splitter 按 semantic boundary（段落）切分成 $\{c_1, \ldots, c_K\}$，逐个处理。

State Agent 的 prompt (Appendix C.6) 强调三个原则：
1. **Preserve Old Memory**: 旧 memory 中未在新 chunk 出现的信息必须保留
2. **Incremental Integration**: 只添加新事实，冗余则忽略
3. **Maintain High Density**: Memory 是 "pile of facts"，不是 article summary

这类似于 MemAgent (https://arxiv.org/abs/2507.02259) 的思路。

#### Report Agent (叙事生成器)
$$r_{t,k} = \mathcal{A}_{\mathrm{report}}(q, c_k, m_{t,k-1}, r_{t,k-1})$$

增量更新 report，不是简单 append，而是 supplement/correct/replace。

**Intuition on decoupling**: State Agent 负责 information compression，Report Agent 负责 narrative generation。这个解耦很重要：
- State Agent 可以激进压缩（保留 facts）
- Report Agent 可以保守更新（保持 narrative coherence）
- 避免了 single-pass 生成长报告的 hallucination 风险

### 5.3 Algorithm 1 解析

```
for t = 0 to T-1:
    Search Agent: a_t, p_t' = A_search(q, s_t)
    Execute a_t → O_t
    Split O_t into chunks {c_1, ..., c_K}
    for k = 1 to K:
        State Agent: m_{t,k}, p_{t,k} = A_state(q, c_k, m_{t,k-1}, p_{t,k-1})
        Report Agent: r_{t,k} = A_report(q, c_k, m_{t,k-1}, r_{t,k-1})
    Update: m_{t+1} = m_{t,K}, p_{t+1} = p_{t,K}, r_{t+1} = r_{t,K}
    if termination: break
```

Termination 条件:
1. 达到 max turns $T$
2. Search Agent 判定无需进一步搜索

### 5.4 Concurrent Execution (Appendix F)

Naive linear pipeline: $T_{\mathrm{naive}} = O(|\mathcal{D}|)$
Linear concurrent pipeline: $O(|\mathcal{D}|/C)$，$C$ = concurrency limit

用 asyncio event loop + sliding window 管理 active tasks。这对 RL 训练时的 rollout 效率至关重要。

---

## 6. 实验结果深度分析

### 6.1 Table 1: Human Preference 评估 (RQ1)

| Model | Method | Pref. Acc. (%) | Cohen's d |
|-------|--------|----------------|-----------|
| N/A | Human-defined General Rubrics | 48.78 | 0.192 |
| GPT-5 | Generated Rubrics | 60.80 | 0.328 |
| Qwen3-30B-A3B | RL with Hybrid Reward | **65.68** | **0.376** |

**三个关键观察**:

**Observation 1**: General rubrics ≈ random (48.78%)
这证明了 generic rubric 在 query-specific 评估上的根本缺陷。不同 query 需要不同的 evaluation dimension。

**Observation 2**: GPT-5 Generated Rubrics (60.80%) vs SFT on GPT-5 rubrics (59.58%)
SFT 反而略低于直接用 GPT-5 生成，说明单纯的 imitation learning 无法捕获 fine-grained preference margin。

**Observation 3**: RL with Hybrid Reward 显著提升 Cohen's d
- RL with LLM-as-a-Judge: d = 0.296
- RL with Preference Reward: d = 0.384
- RL with Hybrid Reward: d = 0.376

注意 Hybrid 的 d (0.376) 略低于纯 Preference Reward (0.384)，但 Pref. Acc. 更高 (65.68% vs 64.81%)。这表明 hybrid reward 在 accuracy 和 effect size 之间取得了更好的平衡。

### 6.2 Cohen's d 解读

Cohen's d 的标准解释：
- 0.2 = small effect
- 0.5 = medium effect
- 0.8 = large effect

公式 (Equation 7):
$$\mathrm{Cohen's\ d} = \frac{\mathbb{E}[\Delta]}{\sqrt{\mathrm{Var}(\Delta)}}$$

其中 $\Delta_i = S(r_{\mathrm{acc}}^{(i)}) - S(r_{\mathrm{rej}}^{(i)})$

**Intuition**: Cohen's d 衡量的是 score separation 的强度和稳定性。高 d 意味着 model 不仅正确排序，还能产生明显的 score gap。这对 RL 训练很重要——如果 score gap 太小，gradient signal 会很弱。

所有方法的 d 都在 small-medium range (0.2-0.4)，说明这个任务本身很难，human preference 在 report 上的 margin 本来就不大。

参考: https://en.wikipedia.org/wiki/Effect_size#Cohen's_d

### 6.3 Table 2: DeepResearch Bench (RQ2, RQ3)

关键对比（Tongyi-DeepResearch backbone）:

| Workflow | Rubric Strategy | Overall |
|----------|-----------------|---------|
| ReAct | N/A | 40.5 |
| ReAct | RL Rubric Generator | 45.2 |
| MaMs | N/A | 41.8 |
| MaMs | Human-defined General | 42.9 |
| MaMs | GPT-5 Generated | 43.4 |
| MaMs | SFT Rubric Generator | 43.4 |
| MaMs | RL Rubric Generator | **49.3** |

**三个层次的分析**:

**Layer 1: Rubric Strategy 的影响**
- MaMs + RL Rubric (49.3) vs MaMs + N/A (41.8): +7.5
- MaMs + RL Rubric (49.3) vs MaMs + SFT Rubric (43.4): +5.9
- MaMs + RL Rubric (49.3) vs MaMs + GPT-5 Rubric (43.4): +5.9

RL 训练的 rubric generator 带来巨大提升，且明显优于 SFT。这证明了 **RL 从 human preference 学习的必要性**——SFT 只能 imitation，无法优化 discriminative power。

**Layer 2: Workflow 的影响**
- ReAct + RL Rubric (45.2) vs MaMs + RL Rubric (49.3): +4.1
- ReAct + N/A (40.5) vs MaMs + N/A (41.8): +1.3

MaMs 的提升在有无 rubric 时都存在，但**与 RL rubric 结合时提升更大**。这暗示 MaMs 的 structured state 更好地利用了 fine-grained reward signal。

**Layer 3: 与 closed-source 对比**
- Gemini DeepResearch: 49.7
- Ours (Tongyi + MaMs + RL Rubric): 49.3
- OpenAI DeepResearch: 46.5
- Claude Research: 45.0

开源模型接近 Gemini，超过 OpenAI 和 Claude。这是很强的结果。

### 6.4 GRPO vs GSPO (Section 4.4, Figure 3)

这是 paper 中一个很 insightful 的分析。

**现象**: GSPO 和 GRPO 产生几乎相同的 reward (Figure 3a)，但 GSPO 的 entropy 显著更高 (Figure 3b)。

**解释**:
- **GRPO**: group-wise relative advantage over complete rollouts, 但用 token-level likelihood ratio
- **GSPO**: sequence-level importance weighting and clipping

GSPO 的 sequence-level 优化降低了对 local token-level deviation 的敏感度，允许 "multiple realizations with similar global rewards coexist"。

**结论**: Rubric generation 是 mode-seeking task（需要稳定、一致、preference-aligned），GRPO 的低 entropy 更合适。

Table 5 (Appendix G) 量化验证:
- GRPO Hybrid: 65.68% / d=0.376
- GSPO Hybrid: 62.02% / d=0.337

GRPO 在两个指标上都更好。

参考 GSPO: https://arxiv.org/abs/2507.18071

### 6.5 Tool Calling 统计 (Table 6)

| Workflow | Model | Tool Calls/Sample | Turns/Sample |
|----------|-------|-------------------|--------------|
| ReAct | Qwen3-30B-A3B | 6.05 | 2.21 |
| ReAct | Tongyi-DeepResearch | 8.10 | 3.02 |
| MaMs | Qwen3-30B-A3B | 19.70 | 7.74 |
| MaMs | Tongyi-DeepResearch | 39.23 | 9.40 |

MaMs 的 tool calls 是 ReAct 的 ~5x！这解释了为什么 MaMs 效果更好——它做了更深入的研究。但代价是计算成本（需要 192 H20 GPU 做 inference）。

---

## 7. Rubric 结构案例 (Appendix I)

以 "Please generate an analysis report on common network failures" 为例，生成 13 个 rubric items：

- Key Criteria (weight=5): Coverage of Common Failures, Inclusion of Core Analysis, Clear Structure
- Important Criteria (weight=3-4): Analysis of Causes, Troubleshooting Methods, Clear Explanation, Professional Tone
- Optional Criteria (weight=1-2): Systematic Classification, Preventive Measures, Concrete Examples
- Error Criteria (weight=-1~-2): Technical Errors, Listing Without Analysis, Irrelevant Information

**Intuition on weight design**: Key=5, Error=-2 的设计意味着 error 的惩罚最多抵消一个 key criterion 的贡献。这防止了 "一次错误全盘否定" 的问题，同时确保 critical errors 会被显著惩罚。

---

## 8. Intuition Building: 为什么这个方法 work？

### 8.1 Rubric as Learnable Reward Model

传统 reward modeling（如 InstructGPT 的 RM）直接学习 $r_\phi(x, y) \rightarrow \mathbb{R}$。问题是：
- Black box，不可解释
- 容易 reward hack
- 无法 adapt 到 new query type

Rubric generator 学习的是 **structured evaluation criteria**，本质上是 learning **how to evaluate** 而非 learning **what score to give**。这类似于 meta-learning 的思想——学习一个 evaluation policy 而非直接学习 scoring function。

### 8.2 Query-Specificity 的价值

General rubric 假设所有 query 的 evaluation dimension 相同，但实际上：
- 法律查询需要 citation accuracy 和 statute reference
- 科学查询需要 methodology critique 和 data interpretation
- 商业查询需要 market analysis 和 risk assessment

Query-specific rubric 让 model 学会 **从 query 推断 ideal answer 的特征**，这是 DeepResearch 的核心能力。

### 8.3 Human Preference 作为 Ground Truth

Paper 的关键 insight: "one of the most direct supervision signals for assessing report quality is human preference over candidate reports"。

为什么不直接用 LLM-as-judge？
- LLM judge 有自己的 bias（长度偏好、格式偏好等）
- LLM judge 不一定与 human judgment 对齐
- 通过 human preference 训练的 rubric generator，本质上是 distilling human evaluation criteria into structured form

### 8.4 MaMs 的 Markov 假设为什么有效

ReAct 的问题：context 不断累积，到后期 context 可能包含大量冗余信息，导致：
- Attention dilution
- Lost in the middle
- Error accumulation

MaMs 的 state abstraction $(m_t, p_t, r_t)$ 本质上是 learned compression。State Agent 的 prompt 明确要求 "pile of facts, not article summary"，确保 memory 是高密度的 factual representation。

这与 recent work on memory-based agents 一致：
- MemAgent: https://arxiv.org/abs/2507.02259
- IterResearch: https://arxiv.org/abs/2511.07327

### 8.5 为什么 RL 优于 SFT？

SFT on GPT-5 generated rubrics (59.58%) < RL with hybrid reward (65.68%)。

SFT 的问题：
- GPT-5 的 rubric 本身不一定 discriminative
- SFT 只能 imitation，无法优化特定 objective
- 无法学习到 human preference 的 margin

RL 的优势：
- 直接优化 preference consistency
- 可以 explore rubric space，找到 SFT 无法触及的 mode
- Hybrid reward 提供多维度 guidance

---

## 9. 局限性与 Future Direction

Paper 自己承认的 limitations (Appendix J):
1. **Pairwise preference**: 可能无法捕获 multi-way preference structure
2. **Subjective qualities**: novelty, creativity 难以评估
3. **Generalization**: 只在固定 domain 测试

我会补充几个更深层的问题：

### 9.1 Reward Hacking 风险
Hybrid reward 中，$R_{\mathrm{pref}}$ 是 binary 的。Model 可能学到生成 "trivially discriminative" rubric——比如一个只针对 rejected report 特征的 rubric，能正确排序但无语义意义。$R_{\mathrm{llm}}$ 部分缓解了这个问题，但不彻底。

### 9.2 Rubric Generator 的 Distribution Shift
Rubric generator 在 paper 自己构造的 query 上训练，但 DeepResearch agent 的 rollout 可能生成不同 distribution 的 query。这个 gap 没有被讨论。

### 9.3 Computational Cost
192 H20 GPU 做 inference，8 H20 训 rubric generator，32 H20 训 agent。这个 scale 不是普通 lab 能复现的。

### 9.4 Cohen's d 的绝对值偏低
最好的结果 d=0.376，仍是 small effect。这意味着即使在 test set 上，accepted 和 rejected 的 score gap 也不大。这可能限制了下游 RL training 的 signal strength。

---

## 10. 与 Related Work 的定位

### 10.1 DeepResearch Agent 谱系
- **Verifiable reward**: Search-R1, WebExplorer (short-form)
- **DPO without verifiable reward**: WebThinker (https://arxiv.org/abs/2504.21776)
- **Rubric-based RL**: DRTulu (https://arxiv.org/abs/2511.19399) - self-evolving rubrics
- **Dual-agent**: WebWeaver (https://arxiv.org/abs/2509.13312)

Paper 的差异化：**learning rubric generator from human preference**，而非 self-evolving 或 manually designed。

### 10.2 Rubric for Reward Modeling
- Fixed rubrics: Hashemi et al., Que et al.
- Query-specific rubrics: Shao et al. (DRTulu), Xie et al. (Auto-rubric)
- Rubrics as reward models: Gunjal et al., Viswanathan et al.

Paper 的创新点在于 **RL training of rubric generator with human preference supervision**。

参考:
- Auto-rubric: https://arxiv.org/abs/2510.17314
- OpenRubrics: https://arxiv.org/abs/2510.07743
- Checklists better than reward models: https://arxiv.org/abs/2507.18624

---

## 11. 对你（Karpathy）可能感兴趣的点

考虑到你的背景，几个值得深思的角度：

### 11.1 Learn to Evaluate vs Learn to Generate
这个 paper 本质上是把 evaluation 从 "given" 变成 "learned"。这与你在 "Software 2.0" 中讨论的思路一致——把 hardcoded logic 变成 learned function。Rubric generator 是一个 meta-learner，学习如何评估。

### 11.2 GRPO on MoE 的细节
Paper 提到 Qwen3-30B-A3B 是 MoE，GRPO 在 MoE 上有 expert routing mismatch 问题。Appendix D 提到 "crucial update on the Megatron config of optimizers"。这个细节可能值得深挖——MoE + RL 的工程挑战。

### 11.3 State Abstraction 的 Information Bottleneck
MaMs 的 $(m_t, p_t, r_t)$ 是一个 information bottleneck。理想情况下，这个 state 应该是 sufficient statistic for optimal policy。但实际上，State Agent 的 compression 必然有信息损失。如何 measure 这个 loss？如何优化 compression policy？

### 11.4 Hybrid Reward 的 Pareto Frontier
$\lambda_{\mathrm{pref}}$ 和 $\lambda_{\mathrm{llm}}$ 都设为 1，但这个选择是否最优？不同 weight 配置的 Pareto frontier 如何？Paper 没有 ablation。

---

## 12. 总结

这篇 paper 的核心贡献可以用一句话概括：**通过 RL 从 human preference 学习 query-specific rubric generator，为 DeepResearch report generation 提供可扩展、human-aligned 的 reward signal**。

技术上有三个亮点：
1. **Hybrid reward design**: preference consistency + LLM judge + format constraint，互补且 robust
2. **MaMs workflow**: Markov state abstraction + chunk-based processing，解决 long-context dependency
3. **GRPO > GSPO for mode-seeking tasks**: rubric generation 需要稳定性而非多样性

实验上证明了：
- Open-source 模型可以接近 closed-source (Gemini 49.7 vs Ours 49.3)
- RL > SFT > LLM-generated > general rubric 的清晰 hierarchy
- MaMs > ReAct，且与 RL rubric 有协同效应

这个工作对整个 DeepResearch 领域有重要启示：**evaluation 本身是一个可以 learned 的能力**，而 learning to evaluate 可能比 learning to generate 更 fundamental。

---

主要参考链接汇总：
- Paper 本身（假设 arxiv 链接）: 待发布
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- ReAct: https://arxiv.org/abs/2210.03629
- DeepResearch Bench: https://arxiv.org/abs/2506.11763
- DRTulu: https://arxiv.org/abs/2511.19399
- WebThinker: https://arxiv.org/abs/2504.21776
- WebWeaver: https://arxiv.org/abs/2509.13312
- MemAgent: https://arxiv.org/abs/2507.02259
- Rubrics as Rewards: https://arxiv.org/abs/2507.17746
- GSPO: https://arxiv.org/abs/2507.18071
- ResearchRubrics: https://arxiv.org/abs/2511.07685
- Auto-rubric: https://arxiv.org/abs/2510.17314
- slime framework: https://github.com/THUDM/slime
- vLLM: https://arxiv.org/abs/2309.06180
- Cohen's d: https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
