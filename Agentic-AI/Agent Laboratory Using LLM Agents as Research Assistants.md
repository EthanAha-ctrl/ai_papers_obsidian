---
source_pdf: Agent Laboratory Using LLM Agents as Research Assistants.pdf
paper_sha256: 67b9543ae1d8e3ad86a65e2a436ddbd12700d7c8f4a66c5b4c2a6fccc1674d75
processed_at: '2026-07-18T04:01:22-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Agent Laboratory: 深度技术讲解

Andrej，这篇paper来自AMD Research、Johns Hopkins和ETH Zurich的团队，由Samuel Schmidgall一作。我把它拆开来讲，重点放在能让你build intuition的技术细节上。

---

## 1. 问题定位：从"代替科学家"到"协助科学家"

之前的autonomous research工作（比如The AI Scientist [Lu et al., 2024b](https://arxiv.org/abs/2408.06292)、ResearchAgent [Baek et al., 2024](https://arxiv.org/abs/2404.07738)）尝试让LLM **自己想idea、自己写代码、自己写paper**。Si et al. 2024的对100+NLP研究者的调研发现LLM idea在novelty上胜过人类，但feasibility和implementation details显著弱。所以Agent Laboratory的定位很明确：**人类出idea，agent负责执行**，human-in-the-loop可选。

这个定位决定了系统架构：输入是research idea + notes，输出是code repo + research report，中间设置co-pilot checkpoint让human介入。

---

## 2. 系统架构：三阶段Pipeline

### Phase 1: Literature Review
- **Agent**: PhD
- **工具**: arXiv API
- **动作**:
  - `SUMMARY <query>`: 检索top-20相关paper的摘要
  - `FULL_TEXT <arxiv_id>`: 获取单篇全文
  - `ADD_PAPER <id> <summary>`: 加入curated review
- **终止**: 累计`N_max`篇后finalize
- **迭代性**: agent根据每篇内容refine查询，不是一次性检索

**这里有个有意思的设计**：agent temperature=0.8，保留探索性。Literature review的success rate最低（gpt-4o只有60%，o1-preview 80%），主要因为agent倾向于反复调用SUMMARY直到步数耗尽——这是一个经典的instruction-following failure mode。

### Phase 2: Experimentation
包含4个子阶段：

**(a) Plan Formulation**
- PhD agent和Postdoc agent通过DIALOGUE交互
- Postdoc用`PLAN <plan_text>`提交，作为后续subtask的指令
- 输入context: literature review summary

**(b) Data Preparation**
- ML Engineer agent执行Python代码，观察print output
- 工具：`SEARCH_HF <query>` 搜索HuggingFace datasets
- 提交命令：`SUBMIT_CODE <code>`，必须用外部HF dataset，不能写main function
- 代码先过Python compiler检查

**(c) Running Experiments (mle-solver)** — 这是核心，下面单独讲

**(d) Results Interpretation**
- PhD和Postdoc agent讨论实验结果
- Postdoc用`INTERPRETATION <text>`提交
- 要求包含numbers、metrics、significance

### Phase 3: Report Writing (paper-solver)
- PhD和Professor agent协作
- paper-solver生成8节结构化报告
- 后续Paper Refinement阶段有3个reviewer agent做NeurIPS-style review
- PhD agent决定finalize还是回到前面的subtask

---

## 3. mle-solver: 核心实验引擎

这是整个系统最值得深挖的模块。它是一个**基于程序空间的tree search**，类似AIDE的Solution Space Search，但scoring机制不同。

### 3.1 五步循环

**A. Command Execution**
从一个maintained的top-performing programs集合中sample一个程序作为起点，然后通过两个操作refine：
- `EDIT N M <new_lines>`: 替换第N行到第M行的代码（inclusive）
- `REPLACE <full_code>`: 完全重写整个文件

EDIT是主要操作，REPLACE用于重大变更（如换模型架构）。AIDE的Solution Space Search专门针对Kaggle提取accuracy，mle-solver则用LLM评分research code+outcomes。

**B. Code Execution**
新程序过compiler查runtime errors。如果compile失败，尝试`N_rep=3`次修复。Hyperparameter table里能看到：
- Code repair attempts: 2
- Experiment Timeout: 600s
- Error history length: 5（保留最近5个错误防重复）

**C. Program Scoring**
这是关键设计。用一个Professor agent作为LLM reward model，输入：
- `outlined_plan`: 研究计划
- `code`: 生成的代码
- `code_return`: 执行输出

输出一个0到1的float score：
```
SCORE <score>
```

Scoring prompt的关键："how well a model followed the plan, built the code, and got the proper output scored from 0 to 1 as a float"。1分表示完全符合plan目标，低于1按alignment程度递减。

这跟Yao et al. 2024的Tree of Thoughts同构：ToT是遍历reasoning steps并用self-evaluated LLM scoring，mle-solver是遍历programs并用self-evaluated scoring。Temperature设置：
- Initial code temperature: 1.0
- Solver temperature: 1.0
- Score generation temperature: 0.6（评分要确定性高）
- Repair temperature: 0.8

**D. Self Reflection**
无论代码成功或失败，都生成reflection：
- 失败：反思如何修复
- 成功：反思如何提高分数
这些reflections注入到后续iteration的prompt中，形成verbal reinforcement learning loop（类似Reflexion [Shinn et al., 2024](https://arxiv.org/abs/2308.10144)）。

**E. Performance Stabilization**
两个机制防止performance drift：

1. **Top program sampling**: 维护top-scoring programs集合（max=2），每次执行command前随机sample一个，平衡diversity和quality
2. **Batch-parallelization**: 每个solver step同时做N个modifications，选top替换最低分的

这本质上是$(N, \lambda)$-evolution strategy的LLM版：维持population size $N=2$，每代生成$\lambda$个offspring，用fitness（LLM score）选择。

### 3.2 Hyperparameter全貌

| 类别 | 超参 | 值 |
|------|------|----|
| Literature Review | Summaries, Full Text, Decay Steps, Temp | 5, 3, 0.8 |
| Running Experiments | mle-solver steps | 3 |
| | Code repair attempts | 2 |
| | Maximum top codes | 2 |
| | Error history length | 5 |
| | Code history length | 2 |
| | Comparison trials | 2 |
| | Experiment Timeout | 600s |
| | Score generation temp | 0.6 |
| | Repair temp | 0.8 |
| | Initial code temp | 1.0 |
| | Solver temp | 1.0 |
| Paper Writing | paper-solver steps | 5 |
| | Maximum top papers | 1 |
| | Paper history length | 10 |
| | Number of Reviewers (scoring) | 1 |
| | Number of Reviewers (refinement) | 3 |
| | Comparison trials | 2 |
| | Solver temp | 1.0 |
| | Initial paper temp | 0.8 |

注意paper-solver的top papers=1，mle-solver的top codes=2——paper写作更greedy，代码搜索更exploratory。

---

## 4. paper-solver: 报告生成

### 4.1 四步流程

**A. Initial Report Scaffold**
固定8个section：Abstract, Introduction, Background, Related Work, Methods, Experimental Setup, Results, Discussion。每个section放placeholder（如`(ABSTRACT HERE)`）。生成可编译的LaTeX scaffold。

**B. Arxiv Research**
每个section生成时可再次访问arXiv找references，但不强制。Search limit=5（之前发现可能要100次query才成功一次）。

**C. Report Editing**
主要命令是`EDIT N M <new_lines>`，对LaTeX逐行修改。每次edit前先compile验证error-free。这跟mle-solver的EDIT同构——把paper当program，把LaTeX compiler当runtime。

**D. Paper Review**
用Lu et al. 2024b的automated reviewer system，基于NeurIPS guidelines。在500篇ICLR 2022 papers上训练，达65% accuracy（人类66%），F1=0.57（人类0.49）。

输出格式严格：
```json
{
  "Originality": 1-4,
  "Quality": 1-4,
  "Clarity": 1-4,
  "Significance": 1-4,
  "Soundness": 1-4,
  "Presentation": 1-4,
  "Contribution": 1-4,
  "Overall": 1-10,
  "Confidence": 1-5,
  "Decision": "Accept"/"Reject"
}
```

### 4.2 Per-section tips
取自The AI Scientist并修改。比如Methods要求"clearly report precise mathematical equations"，Results要求"Only includes results that have actually been run and saved in the logs. Do not hallucinate results that don't exist"——这是为了对抗hallucination。

---

## 5. Co-pilot vs Autonomous Mode

- **Autonomous**: 仅给idea，subtask顺序执行
- **Co-pilot**: 每个subtask结束有checkpoint，human可以：
  - 批准进入下一阶段
  - 提供notes让agent重做该subtask

这个设计直接对应human-AI collaboration文献中的"supervised autonomy"模式。

---

## 6. 实验结果深度分析

### 6.1 5个研究问题 × 3个LLM backend

5个问题覆盖不同领域：
1. LLM cognitive biases (NLP/Cog Sci)
2. Image transformer vs CNN noise sensitivity (CV)
3. MedQA differential diagnosis (NLP/Medical)
4. Word order sensitivity (NLP/Core)
5. Gender role play on math accuracy (NLP/Social Sci)

3个backend：gpt-4o, o1-mini, o1-preview
10个PhD学生评审，每人随机分配3篇，1-5分rating。

### 6.2 三个维度评分（autonomous mode）

| Backend | Experiment Quality | Report Quality | Usefulness |
|---------|-------------------|----------------|------------|
| gpt-4o | 2.6/5 | 3.0/5 | 4.0/5 |
| o1-mini | 3.2/5 | 3.2/5 | 4.3/5 |
| o1-preview | 2.9/5 | 3.4/5 | 4.4/5 |

**关键观察**：
- o1-preview在usefulness和report quality最高，但experiment quality略低于o1-mini
- gpt-4o全面落后
- Topic-specific variance巨大：image noise topic在gpt-4o得1.5/5，o1-mini得4.0/5，差2.5分

### 6.3 NeurIPS-style评分（重点）

**Automated reviewer vs Human reviewer的gap是核心发现**：

| 维度 | Automated | Human | Gap |
|------|-----------|-------|-----|
| Quality | 3.1/4 | 2.0/4 | -1.1 |
| Significance | 2.9/4 | 2.3/4 | -0.6 |
| Clarity | 3.6/4 | 2.4/4 | -1.2 |
| Soundness | 2.9/4 | 1.9/4 | -1.0 |
| Presentation | 3.2/4 | 2.5/4 | -0.7 |
| Contribution | 2.9/4 | 2.1/4 | -0.8 |
| **Overall** | **6.1/10** | **3.8/10** | **-2.3** |

而NeurIPS 2024接收的paper平均overall=5.85。所以automated reviewer觉得"接近接收水平"，human reviewer觉得"远低于接收水平"。

**这个发现的意义**：The AI Scientist论文声称automated reviewer与ICLR OpenReview分数高度对齐，但Agent Laboratory发现当**真实人类评审自己生成的paper**时，automated reviewer严重高估。这暗示automated reviewer的对齐性可能只在in-distribution papers上成立，对LLM生成的paper存在systematic overestimation。

### 6.4 Co-pilot mode结果

| 维度 | Autonomous | Co-pilot (external eval) | Δ |
|------|-----------|--------------------------|---|
| Quality | 2.0/4 | 2.75/4 | +0.75 |
| Significance | 2.3/4 | 2.25/4 | -0.05 |
| Clarity | 2.4/4 | 2.63/4 | +0.23 |
| Soundness | 1.9/4 | 2.38/4 | +0.48 |
| Presentation | 2.5/4 | 2.88/4 | +0.33 |
| Contribution | 2.1/4 | 2.13/4 | +0.03 |
| **Overall** | **3.8/10** | **4.38/10** | **+0.58** |

**Intuition**：human feedback主要提升了presentation和soundness（更易读、更严谨），但对significance和contribution几乎没影响——这意味着agent生成的research核心insight本身novelty不足，人类polish无法弥补。

Co-pilot user experience：
- Utility: 3.5/5
- Continuation: 3.75/5
- Satisfaction: 3.63/5
- Usability: 4.0/5

75%参与者提供improvement feedback，主要建议：加GUI、更好inspect中间结果、支持更多figures、改进literature review。

### 6.5 Runtime & Cost

| Backend | Total Cost | Total Time | Success Rate |
|---------|-----------|-------------|--------------|
| gpt-4o | $2.33 | 1165.4s (~19min) | 94.3% |
| o1-mini | $7.51 | 3616.8s (~60min) | 92.8% |
| o1-preview | $13.10 | 6201.3s (~103min) | 95.7% |

**对比The AI Scientist**: gpt-4o成本约$15/paper，Agent Laboratory是$2.33，下降84%。

**Phase-level cost breakdown**（以o1-preview为例）：
- Literature Review: $0.31
- Plan Formulation: $0.04
- Data Preparation: $0.30
- Running Experiments: $2.59 (mle-solver调用多)
- Results Interpretation: $0.21
- **Report Writing: $9.58** (大头，因为paper-solver要生成长文档)
- Report Refinement: $0.09

**Time上**：o1-preview在Running Experiments要4036.2s，因为reasoning model每个token都更慢。Report Writing要1854.2s。

**Failure modes**（这些非常honest的limitation讨论很有价值）：
- Literature review中agent反复用SUMMARY命令直到max steps
- Retrieved papers触及token limit
- mle-solver有时0% accuracy且不纠正
- mle-solver倾向于edit line 0
- Print output导致token limit
- mle-solver生成`exit()`命令终止整个进程
- mle-solver用`subprocess.run()`执行系统命令（安全隐患）
- paper-solver arXiv search失败率高

### 6.6 MLE-Bench评测（mle-solver单独评测）

10个low-complexity Kaggle challenges（text+tabular data）：

| Method | Medals | Above Median |
|--------|--------|--------------|
| **mle-solver (ours)** | 2 gold, 1 silver, 1 bronze | 6/10 |
| OpenHands (gpt-4o) | 2 gold | 2/10 |
| AIDE (o1-preview) | 1 gold, 1 bronze | 5/10 |
| MLAB (gpt-4o) | 0 | 0/10 |

数据划分：80% train / 20% dev，dev set用于scoring。最后用highest-score的code在Kaggle test set评测。所有数据用numpy array加载而非file path，模拟data preparation phase。

**Intuition**：mle-solver的LLM reward model比纯accuracy-based scoring更适合research code——research的目标不只是accuracy，还有是否follow plan、code质量、output alignment。但也带来风险：LLM reward model可能被"看起来对"的output欺骗。

---

## 7. 关键Prompt工程细节

### 7.1 Base inference prompt结构
```
{role_description}
{phase_prompt}
{command_descriptions}
{context_prompt}
History: {history_str}
Current Step #{step}
Phase: {phase}
{complete_str}
[Objective] ... {research_topic}
Feedback: {feedback}
Notes: {notes_str}
Your previous command was: {self.prev_comm}. Make sure your new output is different.
```

**关键设计**：
- `complete_str`：在70%步数时插入"You must finish this task and submit as soon as possible!"——这是一种progressive urgency机制
- "Your previous command was: X. Make sure your new output is different"——anti-loop设计
- 每次inference只允许一个command——防止agent把多个command塞进一个response

### 7.2 mle-solver system prompt
强调几条规则：
- "Your method MUST not get 0% accuracy. If it does, you have done something wrong"
- "Before each experiment please include a print statement explaining exactly what the results are meant to show"
- "Under no circumstances should you use tensorflow or keras. Only use pytorch or scikitlearn"
- "Try to have a diversity of command responses if appropriate"

### 7.3 paper-solver的anti-hallucination
Results section tip明确写："Only includes results that have actually been run and saved in the logs. Do not hallucinate results that don't exist." 但paper承认gpt-4o仍会出现hallucination，例子中提到不存在的hyperparameter tuning细节。

---

## 8. Limitations部分非常honest

### 8.1 Workflow limitations
1. **Self-evaluation问题**：automated reviewer高估generated paper质量，LLM自评不可靠
2. **Automated structure限制**：固定8-section结构、限制2个figure、不能管理repo-level code
3. **Hallucination**：尤其在gpt-4o生成的paper中，会编造不存在的实验细节

### 8.2 Ethical considerations
- 可能降低substandard research产出门槛
- 自动化可能放大dataset/algorithm bias
- 可能被滥用于cybersecurity（生成malware）或环境研究（climate risk downplay）
- 需要AI involvement的transparent disclosure

---

## 9. 对未来的暗示

### 9.1 与GitHub Copilot的类比
作者提到可借鉴GitHub Copilot的longitudinal study [Ziegler et al., 2024](https://dl.acm.org/doi/10.1145/3636454)，做Agent Laboratory的长期impact研究——不仅看single paper质量，还看研究者长期productivity变化。

### 9.2 与The AI Scientist的关键差异
- The AI Scientist：完全autonomous，无human-computer interaction
- Virtual Lab [Swanson et al., 2024](https://www.biorxiv.org/content/10.1101/2024.11.03.621730v1)：只做nanobody design，无up-to-date knowledge，不生成papers
- ChemCrow/Coscientist：化学领域，不能解决open-ended research problems
- **Agent Laboratory**：human-centric co-pilot，支持任意ML research idea

### 9.3 自动化workflow优化
作者提到可以结合automatic agent workflow [Hong et al., 2023](https://arxiv.org/abs/2308.00352)和agent generation技术 [Chen et al., 2023a](https://arxiv.org/abs/2309.17288)来优化Agent Laboratory本身的workflow——meta-learning层面。

---

## 10. 我的intuition总结

1. **mle-solver本质上是一种"program-space MCTS with LLM heuristic"**：population size=2，每代生成offspring，用LLM reward model做fitness evaluation，加self-reflection作为verbal memory。这跟进化算法 + LLM的混合体。

2. **paper-solver是"LaTeX as program"**：把paper当program编辑，用LaTeX compiler做validation，用automated reviewer做fitness。结构上跟mle-solver同构。

3. **Co-pilot gain主要在presentation层**：+0.75 quality, +0.48 soundness, +0.33 presentation，但significance几乎不动（-0.05）。这说明human feedback擅长polish但不擅长注入novelty。

4. **Automated reviewer的systematic overestimation**是一个重要negative result——LLM评估LLM-generated content存在in-distribution bias，这个发现对未来automated research评估有方法论意义。

5. **Cost reduction 84%**主要来自：paper-solver比The AI Scientist的paper generation更精简（固定结构vs自由生成）、mle-solver的top-k sampling更高效。

6. **Hardware令人惊讶**：所有实验在2023 MacBook Pro M3 Max + 36GB RAM上跑——这暗示system本身compute footprint很小，主要compute在LLM API调用。这对personal research assistant的普及很重要。

7. **Literature review是最脆弱的phase**：success rate 60-80%，agent容易陷入SUMMARY命令循环。这暴露了LLM agent在long-horizon tool use上的通病。

8. **Self-reflection loop是性能提升的关键**：error history length=5, code history length=2, 通过verbalized memory让agent避免重复错误。这是Reflexion模式在ML code generation上的应用。

**Reference links**:
- [Agent Laboratory Project Page](https://AgentLaboratory.github.io)
- [The AI Scientist (Lu et al., 2024)](https://arxiv.org/abs/2408.06292)
- [MLE-Bench (Chan et al., 2024)](https://arxiv.org/abs/2410.07095)
- [AIDE Technical Report](https://www.weco.ai/blog/technical-report)
- [OpenHands (Wang et al., 2024)](https://arxiv.org/abs/2407.16741)
- [ResearchAgent (Baek et al., 2024)](https://arxiv.org/abs/2404.07738)
- [Reflexion (Shinn et al., 2024)](https://arxiv.org/abs/2308.10144)
- [Tree of Thoughts (Yao et al., 2024)](https://arxiv.org/abs/2305.10601)
- [MLAgentBench (Huang et al., 2024)](https://arxiv.org/abs/2407.01502)
- [Virtual Lab (Swanson et al., 2024)](https://www.biorxiv.org/content/10.1101/2024.11.03.621730v1)
- [MetaGPT (Hong et al., 2023)](https://arxiv.org/abs/2308.00352)
- [Can LLMs Generate Novel Research Ideas? (Si et al., 2024)](https://arxiv.org/abs/2409.04109)
- [GitHub Copilot Productivity Study (Ziegler et al., 2024)](https://dl.acm.org/doi/10.1145/3636454)
