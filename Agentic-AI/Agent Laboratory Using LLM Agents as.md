---
source_pdf: Agent Laboratory Using LLM Agents as.pdf
paper_sha256: 67b9543ae1d8e3ad86a65e2a436ddbd12700d7c8f4a66c5b4c2a6fccc1674d75
processed_at: '2026-07-18T04:06:41-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Agent Laboratory: Using LLM Agents as Research Assistants 深度解析

## Paper整体定位与核心贡献

Agent Laboratory 是 AMD、Johns Hopkins、ETH Zurich 合作的工作，第一作者 Samuel Schmidgall（之前有 AgentClinic 工作）。这篇 paper 的核心定位是一个 **human-in-the-loop research co-pilot framework**，把整个 ML research workflow 拆解成三个 phase 的 pipeline，每个 phase 由 specialized LLM agents 扮演学术角色（PhD student、Postdoc、ML Engineer、Professor）执行。

与 Lu et al. 的 The AI Scientist (2024b) 最大的区别在于：The AI Scientist 试图 fully autonomous end-to-end 生成 idea + paper，而 Agent Laboratory 接受 **human-provided research idea** 作为 input，输出 code repository + research report。这个设计选择背后的 intuition 是 Si et al. (2024) 的发现——LLM 在 idea generation 上有较高 novelty，但在 feasibility 和 implementation details 上存在 weakness，所以 complementary role 比 replacement role 更合理。

Paper 主页：https://AgentLaboratory.github.io
arXiv 链接：https://arxiv.org/abs/2412.16759

---

## 三 Phase Workflow 架构

Agent Laboratory 的整体 workflow 如 Figure 2 所示，分为三个 primary phase：

### Phase 1: Literature Review
**执行 agent**: PhD agent
**工具**: arXiv API
**三种 action**:
- `SUMMARY <search_query>`: 检索 top 20 相关 paper 的 abstracts
- `FULL_TEXT <arxiv_id>`: 提取特定 paper 的完整内容
- `ADD_PAPER <arxiv_id> <summary>`: 把选中的 paper 加入 curated review

这是一个 **iterative process**，agent 会进行 multiple queries，根据 content 评估每个 paper 的 relevance，refine 选择直到达到 N=max 数量。

### Phase 2: Experimentation
这个 phase 内部又拆成 4 个 sub-phase：

**(a) Plan Formulation**
PhD agent 和 Postdoc agent 通过 **DIALOGUE** command 协作讨论，specify：
- 要用哪些 ML models
- 用哪些 datasets
- experiment 的高层步骤
当达到 consensus 后，Postdoc agent 用 `PLAN` command 提交 plan，作为后续 subtasks 的 instructions。

**(b) Data Preparation**
ML Engineer agent 执行 Python 代码，使用 `python` command 和 `SEARCH_HF` command 访问 HuggingFace datasets。代码必须 use external HuggingFace dataset，不允许创建 main function。SW Engineer agent 用 `SUBMIT_CODE` 提交最终代码，先经过 Python compiler 检查。Iterative 执行直到 bug-free。

**(c) Running Experiments**（核心模块，使用 mle-solver）
这个是 paper 的核心贡献之一，后面详细讲。

**(d) Results Interpretation**
PhD 和 Postdoc agent 讨论 mle-solver 产出的 results。Postdoc 用 `INTERPRETATION` command 提交 interpretation，作为 report writing 的 basis。

### Phase 3: Report Writing
**执行 agent**: PhD agent 和 Professor agent
**工具**: paper-solver 模块（后面详细讲）
最终还会经过 Paper Refinement phase，3 个 reviewer agents 模拟 NeurIPS peer review，PhD agent 决定是否 finalize 或 revisit earlier subtasks。

---

## mle-solver 技术深度讲解

mle-solver 是 Agent Laboratory 处理 experimentation 的核心模块，如 Figure 3 所示。它的设计借鉴了 AIDE (Schmidt et al. 2024) 的 Solution Space Search，但把 accuracy extraction 换成了 **LLM reward model scoring**，因为这里要解决的不是 Kaggle 比赛，而是 open-ended research question。

### mle-solver 工作流程 5 个 step：

**A. Command Execution**
从 top-performing programs 集合中 sample 一个 program 作为起点（初始化时只有 empty file）。两种 operation：
- `EDIT N M <new code>`: 替换第 N 到 M 行（inclusive）的代码
- `REPLACE <full new code>`: 完全重写 Python 文件

**B. Code Execution**
新 program 通过 compiler 检查 runtime errors。如果成功，update top programs list；如果失败，attempt repair $N_{rep}=3$ 次后才返回 error 进入新 replacement。

**C. Program Scoring**（关键创新点）
Scoring function 是核心创新。与 AIDE 用 dev set accuracy 不同，这里使用 **LLM as reward model**：

$$\text{Score} = R_{\text{LLM}}(\text{plan}, \text{code}, \text{output}) \in [0, 1]$$

其中：
- $\text{plan}$: Phase 2 (a) Postdoc agent 提交的 research plan
- $\text{code}$: mle-solver 生成的 Python 代码
- $\text{output}$: 代码执行后的 printed output

Professor agent 作为 reward model，prompt 中明确要求输出 `SCORE <float between 0 and 1>`，score=1 表示 high alignment with planning goals，低于 1 表示在 spectrum 上距离 goals 的远近。

这本质上是一个 **self-evaluation guided tree search**，类似 Yao et al. (2024) 的 Tree of Thoughts，但 traverse 的是 program space（通过 EDIT/REPLACE）而不是 reasoning step space。

**D. Self Reflection**
无论成功或失败，mle-solver 都会被 prompt 进行 reflection：
- 失败时：reflect on 如何 fix issue
- 成功时：reflect on 如何 increase score
这些 reflections 加入 history 用于改善 future iterations。

**E. Performance Stabilization**
两个机制防止 performance drift：
1. **Top program sampling**: 维护 highest-scoring programs 集合，每次执行 command 前随机 sample 一个，平衡 diversity 和 quality
2. **Batch-parallelization**: 每个 solver step 同时进行 $N$ 个 modifications，top modification 替换 lowest-scoring program in top collection

使用 **high-entropy sampling** 来 modify code，在 exploration of new solutions 和 refinement of existing ones 之间平衡。

### mle-solver Hyperparameters（Table 1）

| Hyperparameter | Value |
|---|---|
| mle-solver steps | 3 |
| Code repair attempts | 2 |
| Maximum top codes | 2 |
| Error history length | 5 |
| Code history length | 2 |
| Number of comparison trials | 2 |
| Experiment Timeout | 600s |
| Score generation temperature | 0.6 |
| Repair temperature | 0.8 |
| Initial code temperature | 1.0 |
| Solver temperature | 1.0 |

注意 temperature 的有趣设定：score generation 用 0.6（deterministic 倾向），initial code generation 用 1.0（high exploration），repair 用 0.8（moderate）。

---

## paper-solver 技术深度讲解

paper-solver 是 Agent Laboratory 处理 report writing 的模块，如 Figure 4 所示。

### paper-solver 工作流程 4 个 step：

**A. Initial Report Scaffold**
生成 LaTeX scaffold，包含 8 个 standard sections：
1. Abstract
2. Introduction
3. Background
4. Related Work
5. Methods
6. Experimental Setup
7. Results
8. Discussion

每个 section 用 placeholder（如 `(ABSTRACT HERE)`）。Title 以 "Research Report: (title)" 开头，author 写 "Agent Laboratory"。这个 scaffold 直接 compilable LaTeX。

**B. ArXiv Research**
Scaffold 构建期间，paper-solver 可以通过 arXiv interface 搜索 related literature，用于在 specific section 中 find papers to refer。强制 limit 5 次 search tries 防止循环。

**C. Report Editing**
用 `EDIT N M <new latex>` command 进行 line-by-line 修改。每次 edit 前 compile LaTeX 验证 error-free。Prompt 中强调：
- "Your writing style should be boring and objective"
- "The paper length should be 8 pages or 4000 words"
- "The paper MUST BE LONG"

每个 section 有专门的 tips（B.9.1），比如 Methods 要求 "Use as many equations as you find necessary"，Results 要求 "Do not hallucinate results that don't exist"。

**D. Paper Review**
使用 adapted version of Lu et al. (2024b) 的 automated review system，模拟 NeurIPS conference 的 peer review process。这个 system 在 500 ICLR 2022 papers from OpenReview 上达到 human-level accuracy (65% vs 66%) 和超过 human 的 F1 score (0.57 vs 0.49) after calibration。

Reviewer 输出 JSON 格式包含：
- Originality, Quality, Clarity, Significance (1-4)
- Soundness, Presentation, Contribution (1-4)
- Overall (1-10)
- Confidence (1-5)
- Decision: Accept/Reject

**Paper Refinement**
3 个 reviewer agents 生成 reviews，PhD agent 根据 scores 决定 finalize 或 revisit earlier subtasks (planning, experimentation, results interpretation)。这模拟了 real-world academic revision process。

### paper-solver Hyperparameters（Table 1）

| Hyperparameter | Value |
|---|---|
| paper-solver steps | 5 |
| Maximum top papers | 1 |
| Paper history length | 10 |
| Number of Reviewers | 1 (initial) / 3 (refinement) |
| Solver temperature | 1.0 |
| Initial paper temperature | 0.8 |

---

## Co-pilot vs Autonomous Mode

这是 paper 的核心 design choice（Section 3.3.1）。

**Autonomous Mode**
- 只接受 initial research idea
- 每个 subtask 完成后 sequentially 进入下一个
- 无 human involvement

**Co-pilot Mode**
- 每个 subtask 结束有一个 **checkpoint**
- Human reviewer 可以：
  1. Proceed to next subtask，或
  2. Ask agent repeat subtask while providing high-level notes for improvement
- 例如：如果 literature review 漏掉 specific paper，human 可以 instruct agent include it

这个设计背后的 intuition 是：当前 LLMs 在 feasibility 和 implementation details 上仍有 weakness，human guidance 可以弥补这一 gap。

---

## 实验结果深度分析

### 1. Quality Evaluation by LLM Backend (Figure 5)

5 个 research question × 3 个 LLM backend (gpt-4o, o1-mini, o1-preview) = 15 篇 autonomously generated papers。10 个 PhD student 志愿者每人 review 3 篇随机分配的 papers。

**Three axes 评分（1-5 scale）**：
- Experimental Quality
- Report Quality
- Usefulness

**Key findings**:

| Backend | Exp. Quality | Report Quality | Usefulness |
|---|---|---|---|
| gpt-4o | 2.6/5 | 3.0/5 | 4.0/5 |
| o1-mini | **3.2/5** | 3.2/5 | 4.3/5 |
| o1-preview | 2.9/5 | **3.4/5** | **4.4/5** |

重要观察：
- **o1-preview** 最 useful，report quality 最高
- **o1-mini** 实验质量最高
- **gpt-4o** 在所有 axis 上都 outperformed
- Topic-specific variability 大：image noise topic 在 gpt-4o 上 1.5/5，o1-mini 上 4.0/5（+2.5 point difference）

### 2. NeurIPS-style Evaluation (Figure 6)

同一批 papers 用 NeurIPS criteria 评分：

**Human reviewer scores**:

| Backend | Quality | Significance | Clarity | Soundness | Presentation | Contribution | Overall |
|---|---|---|---|---|---|---|---|
| gpt-4o | 1.8/4 | 2.2/4 | 2.6/4 | 1.8/4 | 2.7/4 | 1.9/4 | 3.5/10 |
| o1-mini | 2.3/4 | 2.2/4 | 2.1/4 | 1.7/4 | 2.4/4 | 2.2/4 | 3.8/10 |
| o1-preview | 1.9/4 | 2.5/4 | 2.4/4 | 2.2/4 | 2.4/4 | 2.3/4 | 4.0/10 |
| **Average** | **2.0/4** | **2.3/4** | **2.4/4** | **1.9/4** | **2.5/4** | **2.1/4** | **3.8/10** |

**Automated reviewer scores**:

| Backend | Quality | Significance | Clarity | Soundness | Presentation | Contribution | Overall |
|---|---|---|---|---|---|---|---|
| gpt-4o | 3.1/4 | 3.1/4 | 3.6/4 | 2.9/4 | 3.4/4 | 3.1/4 | 6.2/10 |
| o1-mini | 3.1/4 | 2.9/4 | 3.5/4 | 2.9/4 | 3.2/4 | 2.8/4 | 6.0/10 |
| o1-preview | 3.0/4 | 2.7/4 | 3.7/4 | 3.0/4 | 3.1/4 | 2.7/4 | 5.9/10 |
| **Average** | **3.1/4** | **2.9/4** | **3.6/4** | **2.9/4** | **3.2/4** | **2.9/4** | **6.1/10** |

**关键 gap**: Automated reviewer overestimate 2.3 points（6.1 vs 3.8）。所有 criterion 都有 gap，特别是 Clarity（3.6 vs 2.4 = 1.2 gap）和 Contribution（2.9 vs 2.1）。

**Reference point**: NeurIPS 2024 accepted paper 平均 5.85/10，autonomous mode papers 还差 -2.05 points。这说明当前 autonomous research 还远没达到 top conference acceptance level。

### 3. Co-pilot Evaluation (Figure 7)

15 个 researchers 使用 o1-mini backend（除 literature review）做 co-pilot 实验，每人完成 2 篇 paper（custom + preselected）。

**Tool quality 评分（1-5）**:

| Metric | Preselected | Custom | Average |
|---|---|---|---|
| Utility | 3.25 | 3.75 | 3.5 |
| Continuation | 3.5 | 4.0 | 3.75 |
| Satisfaction | 3.5 | 3.75 | 3.63 |
| Usability | 4.25 | 3.75 | 4.0 |
| Exp. Quality | 2.5 | 2.25 | 2.38 |
| Report Quality | 2.75 | 3.5 | 3.13 |
| Usefulness | 3.5 | 4.0 | 3.75 |

有趣发现：custom topics 在 utility、continuation、satisfaction 上都比 preselected 高，但 usability 反而 -0.5。这暗示 researchers 对自己提出的问题有更高 engagement。

**NeurIPS-style self vs external evaluation**:

| Source | Overall |
|---|---|
| Self-evaluation | 4.13/10 |
| External evaluation | 4.38/10 |
| Autonomous (o1-mini) | 3.8/10 |

External evaluators 给的分数比 self 还高（+0.25），特别是 Quality (+0.62) 和 Significance (+0.25)。这说明 researchers 对自己 work 过于 critical，或者对自己的 vision 期待更高导致 self 评分偏低。

**Co-pilot vs Autonomous (external)**:

| Metric | Δ (Co-pilot - Autonomous) |
|---|---|
| Quality | +0.75 |
| Soundness | +0.48 |
| Overall | +0.58 |
| Clarity | +0.23 |
| Presentation | +0.33 |
| Significance | -0.05 |
| Contribution | +0.03 |

Human involvement 主要提升 presentation、clarity、quality、soundness，但对 significance 和 contribution 几乎没提升。这说明 human co-pilot 帮助让 paper 更 presentable，但没法 fundamentally 提升 research 本身的 impact。

### 4. Runtime Statistics (Figure 8)

| Backend | Cost (USD) | Time (s) | Success Rate |
|---|---|---|---|
| gpt-4o | **$2.33** | **1165.4** | 94.3% |
| o1-mini | $7.51 | 3616.8 | 92.8% |
| o1-preview | $13.10 | 6201.3 | **95.7%** |

**Cost breakdown per phase** (gpt-4o):
- Literature Review: $0.12 (92.9s)
- Plan Formulation: $0.03 (23.3s)
- Data Preparation: $0.09 (37.1s)
- Running Experiments: $0.18 (417.8s)
- Results Interpretation: $0.16 (21.5s)
- **Report Writing: $1.73** (572.5s) ← 最贵
- Report Refinement: $0.02 (16.8s)

Report Writing 是最 expensive phase，因为 long document generation 需要大量 computation。

**Comparison with prior work**: The AI Scientist (Lu et al. 2024b) 用 gpt-4o cost ~$15/paper，Agent Laboratory 只需 $2.33，**6.4x 更便宜**，84% cost reduction。

**Failure 模式分析**:
- Literature Review phase success rate 最低（60-80%）
- Data Preparation phase o1-mini 只 80% success
- 其他 phase 都 100%

### 5. mle-solver on MLE-Bench (Figure 9)

10 个 ML challenges from MLE-Bench low complexity category（text + tabular data only）。

**Comparison** (4 methods):

| Method | Medals | Above Median |
|---|---|---|
| **mle-solver (ours)** | **4 (2 gold, 1 silver, 1 bronze)** | **6/10** |
| OpenHands (gpt-4o) | 2 (2 gold) | 2/10 |
| AIDE (o1-preview) | 2 (1 gold, 1 bronze) | 5/10 |
| MLAB (gpt-4o) | 0 | 0/10 |

mle-solver 在 consistency 和 scoring 上最强，4 个 medals 比 OpenHands 和 AIDE 都多。但要注意 mle-solver 在所有 10 个 challenges 都成功 submit valid solutions（2 小时内），其他方法经常 fail to submit。

具体 challenge scores 看出 mle-solver 在 text tasks 上表现特别好（detect insults in commentary: 0.839, spooky author identification: 0.532），在 tabular 上 weaker（NYC taxi fare prediction: 6.542，远高于 human median 3.597）。

---

## Limitations 深度分析（Section 5）

### Workflow Limitations

**1. Self-evaluation 的问题**
- LLM reviewer 在 Lu et al. (2024b) 上虽然与 real reviewer alignment 高，但 Agent Laboratory 自己的 paper quality 实际不如 The AI Scientist（特别是 figures quality 更低）
- LLM self-evaluation vs human: 53.3% vs 56.1% agreement
- LLM consistency 可能源于 superficial patterns 而非 robust evaluation criteria

**2. Automated structure 的限制**
- paper-solver 强制 8 个 standard sections（abstract, intro, etc.），不允许 unique paper organization
- mle-solver 和 paper-solver 限制只能生成 2 个 figures
- Agent Laboratory 无法管理 repository-level code，files 在每个 phase 手动 provided

**3. Hallucination 问题**
特别是 gpt-4o 在 paper 中有 hallucinated experimental results，例如 image noise paper 中出现：
> "Hyperparameter optimization played a crucial role in achieving these results. The learning rate was set at 0.001, with a batch size of 32, and the number of reasoning steps $L = \{l_1, l_2, ..., l_n\}$ varied between 5 to 10"

这些 numbers 实际从未在 experiment 中出现。

### Common Failure Modes
- Capability models 倾向于 repeatedly use `SUMMARY` command 直到 max steps
- Retrieved papers 触达 max token limit
- mle-solver 有时 0% accuracy 不修正
- mle-solver 倾向 edit line 0
- Print output 触达 token limit
- `python exit()` 命令 terminate 整个 process（需手动 remove）
- `subprocess.run()` 在 host 上执行 system commands（safety concern）
- paper-solver arXiv search 有时需要 100+ tries（后 enforced limit 5）

### Ethical Considerations
- 降低 substandard/misleading scientific outputs 的 barrier
- 可能 overwhelm peer review systems
- 可能 amplify dataset/algorithm biases
- 可能被 misused for harmful technologies (cybersecurity, biased climate analysis)

---

## 与相关工作的深度对比

### vs The AI Scientist (Lu et al. 2024b)
- AI Scientist: fully autonomous, 生成自己的 idea
- Agent Laboratory: human 提供 idea, co-pilot mode
- AI Scientist cost ~$15/paper with gpt-4o
- Agent Laboratory cost $2.33/paper with gpt-4o（6.4x cheaper）
- AI Scientist 在 OpenReview ICLR 2022 上训练的 automated reviewer

### vs AIDE (Schmidt et al. 2024)
- AIDE: Solution Space Search for Kaggle
- AIDE 用 accuracy 作为 reward
- mle-solver: 用 LLM 作为 reward model，处理 open-ended research question
- AIDE 在 MLE-Bench 上 2 medals (1 gold, 1 bronze)，5/10 above median
- mle-solver 4 medals, 6/10 above median

### vs MLE-Bench (Chan et al. 2024)
- MLE-Bench: 75 Kaggle challenges benchmark
- Agent Laboratory 用其中 10 个 low complexity challenges
- 用 Kagggle medal system 评分

### vs MLAgentBench / MLAB (Huang et al. 2024)
- 6 Kaggle challenges benchmark
- MLAB 在 Agent Laboratory 的 10 个 challenge 上 0 medals, 0/10 above median

### vs OpenHands / CodeActAgent (Wang et al. 2024b)
- OpenHands: 2 gold medals, 2/10 above median
- mle-solver: 4 medals (2 gold, 1 silver, 1 bronze)

### vs ResearchAgent (Baek et al. 2024)
- ResearchAgent: automated idea generation + experiment design + iterative refinement via reviewing agents
- Agent Laboratory: 不做 idea generation，focus 在 execution

### vs Virtual Lab (Swanson et al. 2024)
- Virtual Lab: AI agents 设计 nanobody for SARS-CoV-2
- 没有 up-to-date knowledge access，不生成 research papers
- 只在 nanobody design 上 demonstrated

### vs ChemCrow (M. Bran et al. 2024) & Coscientist (Boiko et al. 2023)
- Chemistry-focused autonomous ideation + experimentation
- 不能 solve open-ended research problems

---

## 我的 Intuition 和思考

### 1. Agent 角色设计的隐喻
Agent Laboratory 用 PhD student → Postdoc → ML Engineer → Professor 的 hierarchy 模拟 academic lab。这个 design 不只是 cosmetic，它 actually captures 了 real lab 的协作模式：
- PhD 做 literature review 和 interpretation（需要指导）
- Postdoc 做 plan 和 interpretation（提供指导）
- ML Engineer 做 coding（specialized role）
- Professor 做 scoring 和 final paper review（high-level oversight）

这种 specialization 让每个 agent 的 prompt 可以 highly tuned for 其 role，避免 general-purpose agent 在每个 phase 都要被 re-instructed 的 overhead。

### 2. LLM as Reward Model 的双刃剑
mle-solver 的核心创新是用 LLM 作为 reward model 代替 accuracy。这带来：
- **优势**: 可以处理没有 ground truth 的 open-ended research question
- **劣势**: reward hacking 风险——agent 可能学会 generate 让 LLM reward model 满意的 code 而非真正好的 code
- **未来方向**: 可能需要 human reward model calibration 或 RLHF 训练 specialized reward model

公式上可以表示为：
$$\theta^* = \arg\max_\theta \mathbb{E}_{\text{program} \sim \pi_\theta}[R_{\text{LLM}}(\text{plan}, \text{program}, \text{output})]$$

其中 $\pi_\theta$ 是 mle-solver 的 policy（通过 prompt + temperature 控制），$R_{\text{LLM}}$ 是 Professor agent 作为 reward model。问题是 $R_{\text{LLM}}$ 与 true reward $R^*$ 的 gap：

$$\Delta = |R_{\text{LLM}}(\cdot) - R^*(\cdot)|$$

paper 中 Figure 6 显示 $\Delta$ 在 paper quality 上是 2.3 points（6.1 vs 3.8），说明 LLM reward model 显著 overestimate。

### 3. Co-pilot Mode 的 Pareto Front
从 Figure 7 可以看到 Co-pilot 在 Quality (+0.75) 和 Soundness (+0.48) 上提升大，但 Significance (-0.05) 和 Contribution (+0.03) 几乎没提升。这暗示一个 **Pareto front**：
- Human guidance 帮助 presentation 和 methodology rigor
- 但 research significance 和 contribution 来自 idea quality，而非 execution quality
- 这与 Si et al. (2024) 的发现一致——LLM idea generation 是 bottleneck

### 4. Cost-Quality Tradeoff
| Backend | Cost | Time | Overall (Human) | Usefulness |
|---|---|---|---|---|
| gpt-4o | $2.33 | 19min | 3.5/10 | 4.0/5 |
| o1-mini | $7.51 | 60min | 3.8/10 | 4.3/5 |
| o1-preview | $13.10 | 103min | 4.0/10 | 4.4/5 |

Quality 的提升 cost 是 super-linear 的：从 gpt-4o 到 o1-preview cost 增加 5.6x，但 Overall 只 +0.5 points（+14%），Usefulness 只 +0.4 points（+10%）。这说明当前 LLM 的 scaling 对 research quality 的 marginal return 在递减。

### 5. Pipeline Architecture 的 Generalization
Agent Laboratory 的 three-phase pipeline 可以 generalize 到其他 domain：
- **Medicine**: Literature Review → Clinical Trial Design → Case Report
- **Materials Science**: Literature Review → Synthesis Plan → Property Report  
- **Cybersecurity**: Vulnerability Research → Exploit Code → Security Report

关键是每个 phase 都需要 specialized agents + tools（arXiv → domain-specific database, Python → domain-specific simulation）。

### 6. Future Work 的 Direction
- **Longitudinal study**: 跟踪 researchers with vs without Agent Laboratory 长期 productivity
- **Agent workflow optimization**: 用 AFlow (Zhang et al. 2024a) 或 GPTSwarm (Zhuge et al. 2024) 自动 optimize workflow
- **Repository-level code management**: 让 Agent Laboratory 可以 autonomously manage multi-file codebases
- **Multi-modal figures**: 当前限制 2 figures，未来应该让 mle-solver 生成的所有 figures 都 incorporated

---

## 关键 Web Links Reference

- **Agent Laboratory 主页**: https://AgentLaboratory.github.io
- **Agent Laboratory GitHub**: https://github.com/SamuelSchmidgall/AgentLaboratory
- **arXiv paper**: https://arxiv.org/abs/2412.16759
- **The AI Scientist**: https://arxiv.org/abs/2408.06292
- **MLE-Bench**: https://arxiv.org/abs/2410.07095
- **AIDE**: https://www.weco.ai/blog/technical-report
- **MLAgentBench**: https://arxiv.org/abs/2410.07095
- **OpenHands**: https://arxiv.org/abs/2407.16741
- **ResearchAgent**: https://arxiv.org/abs/2404.07738
- **Virtual Lab**: https://www.biorxiv.org/content/10.1101/2024.11
- **ChemCrow**: https://www.nature.com/articles/s42256-024-00932-7
- **o1-preview**: https://openai.com/index/introducing-openai-o1-preview/
- **NeurIPS 2024 accepted paper average**: NeurIPS official statistics
- **Tree of Thoughts**: https://arxiv.org/abs/2305.10601
- **Self-Refine (Reflexion)**: https://arxiv.org/abs/2303.11366
- **Chain of Thought**: https://arxiv.org/abs/2201.11903

---

## 总结

Agent Laboratory 代表了 LLM-based research automation 的一个重要 milestone。它的核心 contribution 是把 research workflow 分解为可管理的 phase，每个 phase 用 specialized agents + tools，让 human researcher 可以 focus 在 idea generation 和 high-level guidance 上。

实验结果显示：
- 当前 autonomous mode 还没达到 top conference acceptance（3.8/10 vs 5.85/10）
- Co-pilot mode 显著提升 presentation 和 clarity，但 significance 和 contribution 仍受限
- mle-solver 在 MLE-Bench 上超越 prior methods，证明 LLM-as-reward-model 的有效性
- Cost efficiency 极高（$2.33/paper with gpt-4o），比 The AI Scientist 便宜 6.4x

未来的 key direction 是解决 LLM self-evaluation 的 gap，以及让 significance 和 contribution 这两个最难的 axis 也能被 human co-pilot 提升。这可能需要 fundamentally 不同的 architecture——比如 idea generation + execution 的 joint optimization，或者 multi-agent debate 在 idea stage 而非 paper stage。
