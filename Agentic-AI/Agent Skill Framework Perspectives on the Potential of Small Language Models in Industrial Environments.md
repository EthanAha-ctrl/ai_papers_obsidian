---
source_pdf: Agent Skill Framework Perspectives on the Potential of Small Language
  Models in Industrial Environments.pdf
paper_sha256: e160caa68a08064ad298a8eef8d34c21b377bd806295f93d00534501553042d5
processed_at: '2026-07-18T04:11:33-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Agent Skill Framework 在 Small Language Models 上的潜力分析

Andrej，这篇 paper 触及了一个当下非常实际的问题：**当工业场景无法依赖 GPT-4o / Claude 这类 proprietary API 时，Agent Skill paradigm 是否仍然成立？** 答案是 partial yes，并且 paper 给出了相当精细的 scaling law 边界。我从 intuition 层面给你拆解。

---

## 1. Motivation：为什么 Agent Skill 值得在 SLM 上重新审视

Agent Skill 概念最早由 Anthropic Claude 团队提出（[Anthropic Agent Skills blog](https://www.anthropic.com/engineering/agent-skills)），现在 GitHub Copilot、LangChain（[DeepAgents](https://github.com/langchain-ai/deepagents)）、OpenAI 都官方支持。它在 huge LLM 上的效果是明确的：reduce hallucination、boost tool selection、compress context window。

但工业部署有两层硬约束：
- **Data security**：金融、保险、军事场景不能把 PII / claim 数据发到 OpenAI
- **GPU budget**：即使能跑开源模型，80B 模型在 production latency 下 VRAM 占用极其昂贵

Paper 的核心 question 就是：**Agent Skill 的 benefit 是否 transferable 到 270M–80B 的 open-source SLMs？在哪一个 scale 上开始 work？**

---

## 2. Agent Skill 的 mental model

你可以把它理解成一个 **progressive disclosure 的 POMDP controller**。和 RAG 的本质区别在于：

| 维度 | RAG | Agent Skill |
|------|-----|-------------|
| 知识表示 | vector embeddings in DB | raw text/markdown (SKILL.md) |
| 检索方式 | similarity top-k | LLM 自己决定是否 reveal |
| Context length | rigid expansion | tightly bounded effective length |
| Reasoning burden | low (retrieval is mechanical) | high (LLM must decide what to load) |

这意味着 Agent Skill 把 **retrieval 的智能要求从 retriever 转移到了 LLM 本身**——这是为什么它在 SLM 上会出问题的根本原因。SLM 的 ICL 和 reasoning 能力弱，progressive disclosure 反而变成负担。

---

## 3. POMDP 形式化（这是 paper 最漂亮的部分）

作者把 Agent Skill 抽象为 augmented POMDP：

$$\mathcal{M} = \langle S, A, O, T, \Omega \rangle$$

**变量含义**：

- $S$：state space。$s_t \in S$ 表示 hidden task situation——包括 user intent、task stage、unretrieved environment facts。这是部分可观测的，agent 看不到 $s_t$ 本体。

- $O$：observation space。$o_t \in O$ 是 agent 在时间 $t$ 能看到的东西——current user message、history、已经 reveal 的 skills/resources。

- $A$：action space。这里有四类 action：
  - `use(k)`：commit 到 skill $k$
  - `reveal(ρ_k)`：paid context acquisition（花钱扩 context）
  - `execute(π_k)`：执行 skill 内部 workflow
  - 环境/tool calls

- $T$：transition kernel，$P(s_{t+1} | s_t, a_t) = T(s_{t+1} | s_t, a_t)$，Markov 假设。

- $\Omega$：observation model，给定 updated context 和 action outcome 产生 next observation。

**Belief state**（这是关键）：

$$b_t = P(s_t \mid o_{\leq t}, a_{<t})$$

$b_t$ 是 agent 对 latent state $s_t$ 的 posterior。注意下标：conditioning on 所有 observations up to $t$ 和所有 actions before $t$。这就是 POMDP 的核心——agent 不直接观察 state，而是维护一个 belief distribution。

**Skill 的三元组定义**：

$$k \in \mathcal{K}, \quad k = (d_k, \pi_k, \rho_k)$$

- $d_k$：textual descriptor（skill name + description，即 SKILL.md 的 frontmatter）
- $\pi_k$：intra-skill policy，temporally extended option-level procedure（一个 skill 内部的多步 workflow）
- $\rho_k$：reference mechanism，formalizable as Markov kernel，可以 reveal 更多 context，甚至 pointer 到其他 skills（cross-skill reference）

### Intuitive interpretation：为什么 progressive disclosure 是 approximately optimal

Paper 引用 Kaelbling et al. 1998 的经典结果（[Paper link](https://www.sciencedirect.com/science/article/pii/S000437029900055X)）：**finite-horizon POMDP 的 optimal value function over belief space 是 piecewise-linear and convex (PWLC)**。

这意味着什么？Belief space $\Delta(S)$ 可以划分成多个 polyhedral region，每个 region 对应一个 optimal contingent plan（即一个 linear $\alpha$-vector）。当 belief $b_t$ 落在某个 region 内，agent 应该执行对应的 plan。

**Progressive disclosure 在这个框架下的解读**：
- 当 $b_t$ diffuse（不确定），reveal 的 expected value of information > reveal cost → 应该花 cost 去 `reveal(ρ_k)`
- 当 $b_t$ concentrated（已经确定），直接 `execute(π_k)` 更 cost-effective
- 这正是 PWLC value function 的几何结构所预测的——不同 belief region 对应不同 plan，progressive disclosure 是对 POMDP optimal control 的近似

这个 formalization 给了 Agent Skill 一个坚实的理论 grounding。但**注意**：这是 approximate optimality，真正的 exact POMDP solving 在大 state space 上 intractable，LLM 的 in-context reasoning 本质上是在做 amortized inference 来近似这个 belief update。

---

## 4. 三种 Context Engineering 策略

Paper 设计了三个对照条件：

| Method | 描述 | Context 行为 |
|--------|------|--------------|
| **DI** (Direct Instruction) | minimal prompt，模拟 raw user input | 不加载任何 skill |
| **FSI** (Full-Skill Instruction) | 把整个 temporary skill repository 全部塞进 context | static, full load |
| **ASI** (Agent Skill Instruction) | on-demand loading，model 决定是否需要更多 skill detail | progressive disclosure |

每个 task 的 skill repository 是用 ground-truth skill + 4-5 个 distractor skills 构造的（从公开 skill hub 采样）。这个 setup 模拟了真实部署中"几十个 skill 里选对"的场景。

---

## 5. Datasets

| Dataset | Length | # Labels | Domain | # Eval |
|---------|--------|----------|--------|--------|
| IMDB | 74 words | 2 | Film reviews | 300 |
| FiNER | 50 words | 139 | Financial XBRL tags | 403 |
| InsurBench | 710 words | 2 | Insurance claims | 200 |

- **IMDB**：简单 binary sentiment，过滤了 300-500 字符长度的 review 控制长度。用来做 baseline。
- **FiNER**（[Paper](https://aclanthology.org/2022.acl-long.220/)）：139 个 XBRL tag types，需要金融 domain knowledge + logical reasoning。任务是给一句话里的 target number 选 best-matching XBRL tag。
- **InsurBench**：proprietary，来自真实保险理赔 email 线程。长、noisy、multilingual 干扰、需要推断 claimant 和 agent 双方 intent。任务：是否继续 engage、采取 action 还是 close case。

InsurBench 的关键价值在于 **dataset contamination 风险低**——因为是私有数据，模型训练时没见过。

---

## 6. Model Inventory

| Model | Size | VRAM (GB) | Type |
|-------|------|-----------|------|
| GPT-4o-mini | - | - | closed-source baseline |
| Gemma-3-270m-it | 0.27B | 1 | tiny |
| Gemma-3-4b-it | 4B | 10 | small |
| Gemma-3-12b-it | 12B | 29 | mid |
| Qwen3-30B-Instruct | 30B | 72 | mid-large |
| Qwen3-80B-Instruct | 80B | 192 | large |
| Qwen3-80B-Thinking | 80B | 192 | reasoning-specialized |
| Qwen3-80B-Coder | 80B | 192 | code-specialized |

这个 scale 跨度 (270M → 80B) 让我们能观察 **scaling cliff**——在哪个 parameter count 上 Agent Skill 开始 work。

---

## 7. Main Results（Table 3 详解）

让我把关键数字拎出来，特别看 FiNER（最有挑战性的公开数据集）和 InsurBench（工业 relevant）：

### FiNER 上 ASI 的提升幅度

| Model | DI Cls ACC | ASI Cls ACC | Δ (绝对提升) |
|-------|-----------|-------------|--------------|
| Qwen3-80B-Instruct | 0.198 | 0.654 | +0.456 |
| Qwen3-80B-Thinking | 0.478 | 0.717 | +0.239 |
| Qwen3-80B-Coder | 0.309 | 0.657 | +0.348 |
| Qwen3-30B-Instruct | 0.184 | 0.564 | +0.380 |
| Gemma-3-12b-it | 0.433 | 0.503 | +0.070 |
| Gemma-3-4b-it | 0.216 | 0.219 | +0.003 |
| Gemma-3-270m-it | 0.034 | 0.000 | -0.034 |

**Intuition**：
1. **Tiny models (<4B) 在 ASI 下基本 dead**——Gemma-3-270m-it 甚至从 0.034 掉到 0.000，说明它根本无法理解 skill selection 这个 meta-task。Gemma-3-4b-it 的 Δ 只有 0.003，ASI 的 overhead 完全没有 ROI。
2. **12B 是 threshold**——Gemma-3-12b-it 开始有 modest gain (+0.070)，说明这个 scale 的 ICL 能力刚刚够支撑 skill routing。
3. **30B+ 是 sweet spot**——Qwen3-30B 的 Δ=+0.380，和 80B 同量级。
4. **80B 接近 GPT-4o-mini baseline**（GPT-4o-mini 在 FiNER 上是 0.484，Qwen3-80B-Coder ASI 是 0.657，已经**超过** closed-source baseline）。

### InsurBench 上的 Skill ACC（routing accuracy）

| Model | Skill ACC |
|-------|-----------|
| Qwen3-80B-Instruct | 0.915 |
| Qwen3-80B-Thinking | 0.945 |
| Qwen3-80B-Coder | 0.990 |
| Qwen3-30B-Instruct | 0.990 |
| Gemma-3-12b-it | 0.990 |
| Gemma-3-4b-it | 0.780 |
| Gemma-3-270m-it | 0.190 |

**Insight**：4B 是 hard floor——0.78 的 routing accuracy 意味着 22% 的 task 直接走错 skill，后续 execution 再好也救不回来。270M 的 0.19 基本是随机。

### Code Model 的胜利（最重要发现之一）

在 80B 这一档对比三个 specialization：
- Instruct：FiNER 0.654, InsurBench 0.620, Avg VRAM-Time 5.321
- Thinking：FiNER 0.717, InsurBench 0.545, Avg VRAM-Time 181.003
- Coder：FiNER 0.657, InsurBench 0.660, Avg VRAM-Time 10.975

**Code model 在 InsurBench 上甚至超过 Thinking variant**（0.660 > 0.545），同时 VRAM-Time 只有 Thinking 的 1/17（10.975 vs 181.003）！

这给了一个非常重要的工业部署结论：**Code specialization + Agent Skill 是 SLM 部署的最优组合**。Paper 推测这也部分解释了为什么 Claude 系列在 Agent Skill 应用上 adopted 这么快——Claude 本身就是 code-oriented 训练的。

---

## 8. Scaling Cliff：Skill Hub Size 的鲁棒性（Figure 2）

这是 paper 里最 striking 的图。当 skill hub 从 N=5 扩到 N=100 时：

- **Tiny models (Gemma-3-4b, 270m)**：在 N=10-20 之后 accuracy 急剧下降
- **≥12B models**：在 N=100 时仍保持高 precision
- **Code-specialized variant**：在 skill selection 上超过 instruct 和 thinking

这个 finding 对 real-world deployment 极其关键——真实 agent 项目要达到 full autonomy 通常需要 50+ skills（参考 [CUA-Skill](https://arxiv.org/abs/2601.21123)），tiny models 在这个规模下完全不可用。

**为什么 code model 在 skill selection 上更强？** Paper 没给确定答案，但我的 intuition 是：SKILL.md 的结构（frontmatter + YAML-like fields + nested references）在 syntax 上接近 code，code model 对这种 structured text 的 parsing 能力更强。

---

## 9. Post-hoc 探索的两个有意思的实验

### 9.1 Chat History 的作用（Table 4）

| Model | ASI Cls ACC | ASIH Cls ACC | Δ | VRAM-Time 增量 |
|-------|-------------|--------------|---|----------------|
| Qwen3-80B-Instruct | 0.620 | 0.535 | -0.085 | +4.714 |
| Qwen3-30B-Instruct | 0.450 | 0.500 | +0.050 | +0.090 |
| Gemma-3-12b-it | 0.575 | 0.585 | +0.010 | +0.039 |
| Gemma-3-4b-it | 0.525 | 0.660 | +0.135 | +0.002 |
| Gemma-3-270m-it | 0.415 | 0.525 | +0.110 | +0.007 |

**反直觉的发现**：
1. **Chat history 对 tiny models 帮助最大**（+0.110, +0.135），对 80B 反而 hurt（-0.085）。这是因为 tiny models 的 reasoning 弱，额外的 context 给了它们更多 anchor。
2. **VRAM-Time cost 对大模型极其昂贵**（80B 增 4.714 GB·min），但对 tiny model 几乎免费（0.002-0.007）。这是 non-linear 的——大模型每多一个 token，attention 计算 + KV cache 占用急剧上升。

**Deployment 指导**：chat history 主要推荐给 lightweight SLMs，80B 上要谨慎。

### 9.2 "Skill" 关键词的 synonym 替换（Table 5）

测试了 5 个 synonym：Skill, Capability, Expertise, Proficiency, Know-how

在 ASI 下：
- **Expertise** 在 Cls ACC 上略胜 Skill（0.610 vs 0.620 on F1，0.594 vs 0.601，差不多）
- **Know-how** 在 VRAM-Time 上大幅优于 Skill（4.302 vs 5.321 GB·min，节省 19%）

虽然差异不大，但说明 **token choice 对 SLM 的 routing 行为有可测量的影响**。这可能和训练语料中这些词的 frequency 和 context distribution 有关——"Expertise" 可能更多出现在 structured professional context 中，更接近 SKILL.md 的语域。

---

## 10. Progressive Disclosure 的 SLM 困境（Appendix A）

这里有个非常重要的负面发现：**Cross-Skill references 在 SLMs 上基本不可用**。

当 SKILL.md 里说 "see skill X for more details"，SLMs（甚至 GPT-4o-mini）几乎检测不到这种 reference，导致 intra-skill call 无法触发。Paper 测试了 LangChain CLI 默认 system prompt，**只有 Claude Sonnet 4.5 / Opus 4.5 能达到 near-100% hit rate**。

这意味着 paper 的实验**主动排除了 nested skill invocation**——因为 open-source 模型的 hit rate 太低，无法做 meaningful comparison。这其实是个 important limitation：Agent Skill 的 hierarchical 引用结构（paper 里 POMDP formalization 的 $\rho_k$ 作为 Markov kernel 可以 reference 其他 skills）在 SLM 上基本是 dead letter。

---

## 11. 评估指标的巧思：Avg VRAM-Time

Paper 提出的 `Avg VRAM Time (GB·min)` 是一个被低估的贡献。定义：

$$\text{Avg VRAM-Time} = \text{VRAM}_{\text{occupied}} \times \text{wall-clock time per task}$$

为什么这个指标重要？因为 production 计费是 GPU-hours：
- **Latency** 决定 user experience
- **VRAM occupancy** 决定 throughput bottleneck——一旦 VRAM saturated，其他 job 不能并发

传统 FLOPS metric 反映不了这个——一个 80B model 跑 0.5 秒占满 192GB VRAM，和一个 12B model 跑 2 秒占 29GB VRAM，VRAM-Time 分别是 96 vs 58 GB·min，后者 throughput 更优，即使 latency 更长。

Figure 1 的 scatter plot 把这个 trade-off 可视化得很清楚：lower-left 是 sweet spot。**Code model + Agent Skill 几乎总是落在 lower-left**。

---

## 12. Limitations 和 Open Questions

Paper 自己坦承的：

1. **Task scope 窄**：只测了 classification 和 tagging，没有 generation、multi-step planning、tool use 的 end-to-end 评估。
2. **Progressive Disclosure 在 SLM 上的 failure 原因不明**：是 attention 机制？是 ICL capacity？是 training data distribution？都 open。
3. **Code model 为什么 VRAM-efficient 也不明**：是结构化文本 parsing 能力？还是 attention pattern 更 sparse？
4. **SKILL.md 的最优结构和表示**：paper 用了 LangChain 默认格式，但最优 markdown 结构（frontmatter vs prose、nesting depth、reference syntax）没有 ablation。

我的额外思考：

- **POMDP formalization 是 descriptive 而非 constructive**——它解释了为什么 progressive disclosure 是 approximately optimal，但没给 solving algorithm。真实 LLM agent 是在做 amortized inference 来近似 belief update，这个近似的 error bound 在 SLM 上有多大？这值得理论分析。
- **Skill selection 的 ICL 机制**：tiny models 失败是因为 context length 不够？还是因为 attention head 学不到 routing pattern？如果是后者，fine-tune 一个 router head 能不能救 4B 模型？这是 practical 的问题。
- **Code specialization 的 transfer**：code model 在 skill selection 上强，是不是意味着我们应该用 code pretraining 数据来训练 general agent router？这指向一个 data scaling law 的方向。

---

## 13. 相关工作的脉络

Paper 串联了几个重要 line of work：

- **Context Engineering**：[Mei et al. 2025 survey](https://arxiv.org/abs/2507.13334) 是 comprehensive review。"Lost in the Middle" / "Context rot"（[Anthropic 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)）是 motivation。
- **Agent Skill 生态**：[DeepAgents](https://github.com/langchain-ai/deepagents)、[Agent Skill Collections](https://github.com/langchain-ai/agent-skills)、[CUA-Skill](https://arxiv.org/abs/2601.21123)、[Meta CE](https://arxiv.org/abs/2601.21557)、[Single-agent with skills](https://arxiv.org/abs/2601.04748)
- **SLM 在 industrial 的其他 work**：[Li et al. 2025](https://aclanthology.org/2025.acl-industry.88/) 是同一团队之前的 industrial text classification 工作
- **Safety**：[Agent skills in the wild](https://arxiv.org/abs/2601.10338) 讨论 skill security vulnerabilities

---

## 14. 我的整体 takeaway

这篇 paper 的价值在于**画清了 Agent Skill 在 SLM 上的 capability frontier**：

1. **<4B**：dead zone，Agent Skill framework 的 overhead 没有回报
2. **12B-30B**：minimum viable，能做 skill selection 但 execution 弱
3. **80B code-specialized**：production sweet spot，接近 closed-source quality，VRAM-efficient
4. **Cross-skill reference / nested invocation**：只有 Claude Opus 级别能可靠做，SLM 全部 fail

对工业部署者，paper 的 actionable insight 很清晰：**用 80B code model + Agent Skill + flat skill structure（no nesting）**。这是当前 open-source 能达到的最优 ROI。

对研究者，最大的 open question 是 **progressive disclosure 在 SLM 上失败的 mechanistic 原因**——这背后是 ICL capacity、attention pattern、还是 training data 的问题？如果搞清楚，可能能 unlock 4B 模型的 Agent Skill 能力，这对 edge deployment 是 game-changer。

---

## Reference Links

- [Anthropic - Agent Skills](https://www.anthropic.com/engineering/agent-skills)
- [Anthropic - Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [LangChain DeepAgents GitHub](https://github.com/langchain-ai/deepagents)
- [Kaelbling, Littman, Cassandra 1998 - Planning and acting in partially observable stochastic domains](https://www.sciencedirect.com/science/article/pii/S000437029900055X)
- [FiNER dataset paper (Loukas et al. 2022)](https://aclanthology.org/2022.acl-long.220/)
- [IMDB dataset (Maas et al. 2011)](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Qwen3-30B-Instruct HuggingFace](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- [Qwen3-Next-80B-Instruct HuggingFace](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- [Gemma-3-12b-it HuggingFace](https://huggingface.co/google/gemma-3-12b-it)
- [Mei et al. 2025 - Survey of Context Engineering for LLMs](https://arxiv.org/abs/2507.13334)
- [Belcak et al. 2025 - Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153)
- [Ye et al. 2026 - Meta Context Engineering via Agentic Skill Evolution](https://arxiv.org/abs/2601.21557)
- [Li 2026 - When Single-Agent with Skills Replace Multi-Agent Systems](https://arxiv.org/abs/2601.04748)
- [Liu et al. 2026 - Agent Skills in the Wild: Security Vulnerabilities](https://arxiv.org/abs/2601.10338)
- [Lewis et al. 2020 - RAG original paper](https://arxiv.org/abs/2005.11401)
- [Brown et al. 2020 - GPT-3 / Few-shot learners](https://arxiv.org/abs/2005.14165)
