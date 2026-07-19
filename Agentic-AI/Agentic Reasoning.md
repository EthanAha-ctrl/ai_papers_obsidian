---
source_pdf: Agentic Reasoning.pdf
paper_sha256: fe7838f41d3d128ff4a6e311748083db4ff80510bd98c46daa4d77b7f7289299
processed_at: '2026-07-18T05:08:41-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Agentic Reasoning 技术深读

这篇 paper 来自 Oxford 的 Junde Wu 团队，arxiv 编号约 2502.04644，github: https://github.com/theworldofagents/Agentic-Reasoning 。核心 idea 很直接：把 reasoning LLM（DeepSeek-R1 / QwQ 这一类 CoT 模型）作为 "main brain"，外挂三个 LLM-based agent 作为 "sub-contractors"——web-search agent、coding agent、Mind Map agent。下面按你的风格,从 motivation、formalism、architecture、empirical signal 几个层面拆解。

---

## 1. Motivation: 为什么单靠 CoT 不够

paper 的开篇观点切到了 R1/o1 类 reasoning model 的真实痛点。DeepSeek-R1 用纯 rule-based outcome reward（代码 pass/fail、数学答案对错）训练出来，作者自己承认这削弱了模型 articulate reasoning process 的能力——chain 看起来对，但 idea-to-idea 之间的 transition 不透明。更关键的是,这类 RL 训练出来的 chain 在 **math/code 这类有 ground-truth 的 well-structured domain** 上很强,一旦遇到 social science / ethics / open-ended research 这种 no-binary-correctness 的领域,formal reasoning 反而会让模型 over-rigid。

作者直接类比 human workflow：人写一篇 deep research report 的过程 = 网搜 + 算 + 白板/Mind Map。LLM 也应该这样。这个出发点其实和 Karpathy 你之前在 "Software 3.0" 里讲的 "LLM as OS kernel、agent as user-level program" 那个 mental model 是完全一致的——reasoning LLM 是 kernel,三个 agent 是用户态进程,kernel 通过特殊 token 像 syscall 一样把任务 dispatch 出去。

参考：
- https://arxiv.org/abs/2502.04644 (Agentic Reasoning 原文)
- https://github.com/theworldofagents/Agentic-Reasoning
- https://karpathy.ai/software30/slides.pdf (你自己的 Software 3.0 talk)

---

## 2. 形式化：Joint Probability 的拆解

paper 第 2.1 节给的形式化很简洁,但每一项都有信息量。原式:

$$
P(r, a \mid o, q, e, k) = \prod_{t=1}^{T_r} P(r_t \mid r_{<t}, o, q, e_{\le t}, k_{\le t}) \times \prod_{t=1}^{T_a} P(a_t \mid a_{<t}, r, o, q, e, k)
$$

变量字典：

- $o$ = task instruction (overarching objective)
- $q$ = user query (具体问题)
- $e$ = external tool outputs,$e_{\le t}$ 表示到 step $t$ 为止所有 tool 调用结果拼接起来
- $k$ = Mind Map knowledge graph 的 snapshot,$k_{\le t}$ 表示到 step $t$ 为止 Mind Map 的状态(注意这是 dynamic evolving 的)
- $r$ = reasoning chain,长度 $T_r$ tokens
- $a$ = final answer,长度 $T_a$ tokens
- $r_t$ = reasoning chain 中第 $t$ 个 token
- $r_{<t}$ = causal context (autoregressive)

直觉拆解:

**第一个 product** = Reasoning Process。注意条件里同时挂了 $e_{\le t}$ 和 $k_{\le t}$,这意味着 reasoning token 的生成**不是纯自回归**,而是被 tool results 和当前 Mind Map 状态**注入**过的。每当 model emit `<websearch>`, `</websearch>` 这种 special token 时,$e_{\le t}$ 会 jump——中间插进一段非模型生成的 content,然后再继续 autoregressive。这其实是 **in-context 中断 + 注入**,和 ReAct 的思路一致,但更精细。

**第二个 product** = Answer Generation。条件里直接挂在完整 $r$ 上,这跟 standard CoT 的 answer generation 一致。区别在于 $r$ 里夹了 tool outputs,所以 answer 是 conditioned on enriched reasoning chain 的。

关键 design choice 隐藏在这个公式里:**$e$ 和 $k$ 是 interleaved 进 $r$ 的,不是 prepended**。这就是和 naive RAG 的本质区别。RAG 是 $P(a \mid q, \text{retrieve}(q))$,一次性注入;这里是 $P(r_t \mid r_{<t}, e_{\le t})$,可以多次注入,且每次注入的 query 是基于 $r_{<t}$ 动态构造的。

---

## 3. Architecture: 三个 Agent 的内部分工

### 3.1 Web-search Agent

流程分两步:

1. Search engine 检索 raw web pages
2. 一个 **summarization LLM** 把 raw pages 在 $(q, r_{<t})$ context 下压缩成 concise summary,再 inject 回主 reasoning chain

这里有个细节很重要——summary 的 format 是 **task-adaptive**:

- Factual query ("US population 2024") → 单个数字
- Exploratory query → structured nuanced summary
- Hypothesis validation → degree of support/contradiction

这其实是 **schema-on-read** 的思想,不是固定 template。Summarizer LLM 根据 query 类型自己决定返回 schema。这是个挺 elegant 的设计——避免了 hardcoded template 的 brittle 性。

### 3.2 Coding Agent

这里有个反直觉的发现:不要让 main reasoning model 直接写 code,而是 **delegate 给专门的 coding LLM**。format 是:

```
Write code to perform <code message from reasoning model>
given the context <reasoning context from Mind Map>
to answer the query <user query>.
```

coding LLM 的 output **强制为 natural language**——不是直接返回 code,而是返回 code execution 的 natural language summary。这是为了让主 reasoning chain 保持纯文本流,不被 `<code>` `</code>` block 打断。

为什么这样更好?paper 给的解释是:
- Main reasoning model 的 CoT coherence 不会被 code 生成这种 "context switch" 打断
- Specialization:DeepSeek-R1 不擅长写 code,Claude-Sonnet 擅长——让 Claude 写,R1 推理

这跟你之前讲 "MoE 不止在 layer 内,模型之间也可以是 MoE" 的方向是一致的。Agentic Reasoning 实际上把不同 LLM 当成了一种 **cross-model MoE**,router 就是 main reasoning model 通过 special token 来选 expert。

### 3.3 Mind Map Agent

这是 paper 最有意思的部分。基于 GraphRAG (Edge et al., 2024) 的思路,把 reasoning chain 转成 knowledge graph:

1. Entity extraction:从 $r_{<t}$ 抽 entities
2. Relation identification:entities 之间的 semantic relations
3. Community clustering:对 graph 做 community detection
4. Summary generation:每个 community 用 LLM 生成 summary

Mind Map 有两种 query mode:

- **Global mode**: community summary 直接返回,给 main reasoning model 一个 high-level view
- **Local mode**: 用 query 做 RAG 检索 subgraph,返回 relevant entities + relations

这其实是在 model 的 "working memory" 里建了一个 **externalized hippocampus**。Reasoning chain 太长时,model 自己容易 "lose track" (你训练 nanoGPT 肯定遇到过 long-context degradation),Mind Map 充当 external scratchpad with structure。

GraphRAG paper: https://arxiv.org/abs/2404.16130

---

## 4. 三个 Main Findings 的深度解读

### 4.1 "Less is More"

只给 web-search 和 coding 两个工具就够。**加更多工具反而降性能**,因为:
- Tool selection error 上升
- External tool noise 累积

这个发现其实呼应了你之前 tweet 过的 "tools 给太多反而让 agent 失焦" 的观察。从信息论角度,tool 数量 $N$ 增加会让 router 的 selection entropy $\log N$ 上升,而每个 tool 的 reliability 是 $p_i < 1$,整体 failure probability $\approx 1 - \prod(1-p_i \cdot \text{selection correct})$ 会随 $N$ 增大而上升。

但 paper 也 caveat:对于 non-text modality (financial data, medical image, genetic data) 还是需要 specialized tool。这暗示未来方向是 **modality-specific tool agent**,而非 universal tool。

### 4.2 Delegating Tasks to LLM-Based Agents

这个 finding 其实是 multi-agent system design 的核心 insight。把 coding、graph construction 这些 auxiliary task 委托给专门的 LLM,而不是让 main reasoning model 包揽。两个 advantage:

1. **Minimizing Disruption**: 主模型维持 long coherent reasoning,不被 auxiliary task 切断
2. **Leveraging Specialization**: 不同 LLM 擅长不同事

这跟 Anthropic 的 Constitutional AI、OpenAI 的 Swarm 这些 multi-agent framework 的思路相通,但这里的关键是 **main model 仍是一个 single coherent CoT**,其他 agent 只是 "工具人"——主从架构,而非平级。

### 4.3 Agentic Test-time Scaling (最 interesting)

这是个意外的发现:**对同一个 question,tool calls 越多的 reasoning chain,质量越高**。但 **跨 question 比较**,tool calls 多的问题本身更难,准确率反而低。

这个不对称性可以用作 **test-time verifier signal**:对同一个 question 跑 N 次,选 tool calls 最多的那个 chain 作为 final answer (best-of-N selection with tool-call-count as proxy reward)。

paper 说这个简单 heuristic **比 LLM-as-judge 还好**,且更便宜更稳定。

为什么这个 signal 有效?我的 intuition:
- Tool calls 是 model "主动求证" 的次数,越多说明 model 越 careful
- 而 LLM-as-judge 容易被 fluent writing 骗,被 confident 但错的 chain 误导
- Tool calls 是 **行为信号**,不是 **语言信号**,更难 spoof

这个 finding 对 RL 训练也很有启发——可以用 **tool call frequency 作为 implicit reward** 来 fine-tune reasoning model,让 model 学会 "不确定就查",而不是 "硬猜"。这跟 RLHF 用 human preference 不同,这是一种 **behavioral consistency reward**。

参考你之前讲的 "test-time compute scaling law": https://arxiv.org/abs/2408.03314 (Snell et al., Scaling LLM Test-Time Compute)

---

## 5. 实验数据解读

### 5.1 GPQA Diamond Set (198 questions)

Table 1 关键数字:

| Model | Physics | Chem | Bio |
|---|---|---|---|
| QwQ-32B (baseline reasoning) | 75.6 | 39.8 | 68.4 |
| RAG-QwQ-32B (naive RAG) | 76.7 | 38.7 | 73.7 |
| Search-o1 | 77.9 | 46.2 | 63.2 |
| **Agentic Reasoning** | **88.4 (推算)** | **~58** | **~79** |
| o1-preview (closed) | 89.4 | 59.9 | 65.9 |

注意 Agentic Reasoning 在 Biology 上 78.9 vs o1-preview 65.9——**显著超过 o1-preview**。Physics 88.4 接近 o1-preview 89.4。Chem 58 略低于 o1-preview 59.9。整体非常接近最强的 closed reasoning model,而且 framework 完全开源。

GPQA dataset: https://arxiv.org/abs/2311.12022

### 5.2 GPQA Extended Set vs Human Experts

Table 2 亮点:

| Method | Physics | Chem | Bio |
|---|---|---|---|
| Physicists (human) | 57.9 | 31.6 | 42.0 |
| Chemists (human) | 34.5 | 72.6 | 45.6 |
| Biologists (human) | 30.4 | 28.8 | 68.9 |
| **Agentic Reasoning** | **75.2** | **53.1** | **72.8** |

人类专家在自己的 domain 内 accuracy 比 cross-domain 高,但 Agentic Reasoning 在三个 domain **全部超过人类专家的平均水平**,尤其在 Physics (75.2 vs human best 57.9) 和 Bio (72.8 vs human best 68.9) 上大幅领先。Chem 上 53.1 比 chemist 的 72.6 低——说明 Chem 还需要更精确的领域知识或计算。

### 5.3 Deep Research (vs Gemini Deep Research)

domain expert (PhD-level finance/medicine/law) 出 15-30 个 deep research question,要求至少 20 min 人工 research。Metric 是 pass rate (专家打分)。

Agentic Reasoning 在三个 domain **全部超过 Gemini Deep Research**。这是开源 framework 第一次在 deep research 上超过 closed-source 专门 service。

### 5.4 Werewolf Game (Mind Map 的杀手锏)

7 个 5+ 年经验的人类玩家 vs Agentic Reasoning,模型胜率 **72%**。这远超统计期望胜率,说明 Mind Map 在 **dynamic multi-agent deduction** 场景下确实能 track 玩家间关系。

Figure 5 的 Mind Map 可视化显示,模型每轮根据玩家发言更新 graph 的 edges (deception pattern、voting tendency、disguise strategy)。这相当于 model 学会了 **game-theoretic belief tracking**,通过 externalized graph 而非 internal hidden state 来维护 beliefs。

---

## 6. Case Study 拆解 (Figure 2: COPD + HF)

这个 case 很能体现 framework 的协同:

问题: 68 岁男性,COPD (FEV1 45%, PaO2 58, PaCO2 48) + 新发心衰 (LVEF 35%),如何在改善肺功能的同时不加重心衰?

Agentic Reasoning 的执行路径:

1. **Coding agent**: 计算 optimal FiO2 (Fraction of Inspired Oxygen)——这需要根据 PaO2 目标值和当前 FiO2 推算,涉及呼吸生理公式 (e.g., $\text{PaO}_2 = \text{FiO}_2 \cdot (P_B - 47) - \frac{\dot{V}_{CO_2}}{\dot{V}_A} \cdot ...$)
2. **Web-search agent**: 检索最准确的 PEEP (Positive End-Expiratory Pressure) 值,因为 PEEP 对 COPD 是双刃——改善氧合但可能降低 venous return 加重 HF
3. **Mind Map**: 维护 [patient profile] - [COPD constraints] - [HF constraints] - [intervention options] 的 graph,确保推理不漏掉任一 constraint
4. **Synthesis**: 综合给出最优治疗 plan

这就是 paper 所谓 "comparable to PhD-level expertise" 的具体表现。Figure 2 那张图建议你看原文,展示了一条完整的 reasoning + tool call + synthesis 的 trace。

---

## 7. 局限 & 我的批判性思考

paper 写得相对清楚,但有几个 gap:

1. **No ablation on agent combinations**: 只报告了三 agent 全开的 result,没报告 (web only)、(web + coding, no Mind Map)、(web + Mind Map, no coding) 的对比。Mind Map 的真实 contribution 不够 isolated。
   
2. **Mind Map 的 construction cost**: 每过几步就要用 LLM 抽 entity + 关系 + community detection,这个 cost 不低。paper 没报告 end-to-end latency / token cost。

3. **Test-time scaling 的 baseline 比较弱**: 只和 LLM-as-judge 比,没和 self-consistency、PRM (Process Reward Model) 这类更成熟的 verifier 比。tool-call-count 这个 signal 在简单 QA 上可能 work,在 ambiguous reasoning 上不一定 robust。

4. **Werewolf experiment 的 sample size**: 7 人 × 几轮,72% 胜率的统计 significance 不强。需要更多局数 + 多种 social deduction game。

5. **No formal analysis on "Less is More"**: 只是 empirical claim,没有给出 tool selection entropy vs accuracy 的理论 bound。

---

## 8. 跟你近期 work 的可能联想

Karpathy 你最近在跑 nano-chatbot / 那个 eureka labs,以及讲 Software 3.0,几个关联点:

1. **Agentic Reasoning = Software 3.0 的具体形态**: reasoning LLM 是 OS kernel,special tokens 是 syscall,三个 agent 是 user-space process。这跟你说的 "LLM 编排 LLM" 的图景完全对得上。

2. **Test-time scaling via tool calls**: 这跟你的 "more compute at inference, not just training" 哲学一致。tool call frequency 作为 implicit reward 信号,可能是 RL 训练的新方向。

3. **Mind Map 和你的 "Memory" 直觉**: 你之前在 tweet 里讲过 LLM 缺乏 long-horizon memory 是 bottleneck,Mind Map 是一种 externalized structured memory。和 mem0 / Letta (MemGPT) 的思路类似,但更 graph-structured。

参考链接:
- https://arxiv.org/abs/2502.04644
- https://github.com/theworldofagents/Agentic-Reasoning
- https://arxiv.org/abs/2404.16130 (GraphRAG)
- https://arxiv.org/abs/2501.05366 (Search-o1)
- https://arxiv.org/abs/2412.16720 (OpenAI o1 system card)
- https://arxiv.org/abs/2311.12022 (GPQA)
- https://arxiv.org/abs/2408.03314 (Scaling test-time compute)
- https://karpathy.ai/software30/slides.pdf
- https://github.com/mem0ai/mem0
- https://github.com/letta-ai/letta

如果你接下来想动手验证,我建议从 ablation 入手——拆掉 Mind Map 看看 GPQA 掉多少点,这是验证 paper 真实 contribution 的最直接方式。另外 test-time scaling 那条线可以拿 PRM (Math-Shepherd, https://arxiv.org/abs/2310.10647) 直接 PK 一下 tool-call-count verifier,看哪个更 robust。
