---
source_pdf: Dynamic Cheatsheet.pdf
paper_sha256: 660680ecc5cdb8f4d921e1284bf95df2dfa9020c5923330e406fff9b93bcfd00
processed_at: '2026-08-04T00:36:21-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说Dynamic Cheatsheet

## 一句话版本

让LLM在考试过程中边做题边写小抄，下一题就用自己之前写的小抄。

---

## 核心idea

你想象一个学生参加一场很长的考试。正常LLM的工作方式是：每道题都从头想，做完就忘，下一题继续从零开始。哪怕上一题刚发现一个超好用的解法，下一题遇到类似题目也不会用。

DC做的事情特别朴素：给它一张纸（external memory），做完一题就让它自己想想"这题我有什么收获值得记下来"，把有用的strategy写成小抄。下一道题开始前先看小抄。

就这么简单。没有gradient descent，没有fine-tuning，没有label，没有任何参数更新。就是prompt engineering + 一个外部text buffer。

---

## 两个变体的区别

**DC-Cu（Cumulative版本）**：做完一题→更新小抄→下一题看小抄做题→做完再更新小抄。时序上是"先做后记"。

**DC-RS（Retrieval & Synthesis版本）**：多加了一步——做新题之前，先去翻翻以前做过的类似题目（用embedding similarity找top-k），把这些历史例题和小抄一起喂给curator refine，然后才做题。时序上是"先记后做"。

DC-RS稍微复杂一点，但本质就是"小抄 + 错题本"。

---

## 为什么有效

paper里几个最striking的结果：

**Game of 24**：GPT-4o从10%飙到99%。原因特别好玩——GPT-4o在前几道题突然"开窍"发现用Python暴力搜索最稳，把这个Python函数写进小抄，之后每道题直接调函数。从"手算"切换到"写代码调代码"，错误率直接归零。

但同样的任务Claude 3.5 Sonnet只从12%到14%。因为Claude死活不肯写代码，非要手算。这说明DC的成败很大程度上取决于base model的"工具使用倾向"。

**AIME数学竞赛**：Claude从23%到50%，翻了一倍多。这里memory里存的是数学heuristic——比如"遇到多项式根的问题先想Vieta公式"、"组合题先check Pigeonhole"。

**GPQA Diamond**（研究生级别科学问答）：Claude +9个点。这种knowledge-intensive任务能提升说明memory里存的formula reference确实有用。

---

## 为什么Full History不行

naive做法是把所有历史对话全append进context。paper试了，效果反而比baseline还差。原因明显：

1. Context window会被塞爆
2. 信号被噪声稀释，有用信息被淹没在长篇对话里
3. LLM在长context里会"lost in the middle"

DC的curation本质上就是在做"信息蒸馏"——把100次对话的精华压缩成几页小抄。

---

## 为什么Majority Voting不行

paper还试了self-consistency（多次采样投票），对AIME这种hard reasoning完全没帮助，和baseline持平。原因是voting只是统计聚合，没有真正学到新东西。DC是在做"知识积累"，两者根本不是一回事。

---

## 失败案例

**小模型（GPT-4o-mini / Claude Haiku）**：DC效果很差，有时比baseline还差。两个原因——
1. 模型本身做不对题，小抄里记的都是错误strategy，相当于"小抄污染"
2. 小模型不会curate，不知道什么该记什么该丢

**R1 / o1这类reasoning model**：也没提升，因为它们的CoT太verbose，记进memory后塞不下也用不上。

这告诉我们DC的适用前提：**base model要够强，能产出高质量solution供curate**。

---

## 直觉上的insight

1. **DC本质是把test-time compute从per-instance scaling扩展到cross-instance scaling**。原来CoT/ToT是把compute花在单道题上，DC是把compute累积起来跨题复用。

2. **Memory是test distribution的textual sufficient statistics的online approximation**。每次curation就是一次近似Bayesian update，只不过update发生在text space而不是parameter space。

3. **效果最好的场景是"有algorithmic closure"的任务**——存在一个固定算法能解决整类问题（Game of 24的Python brute-force）。这时memory一旦存了正确算法，后续就是"调函数"，几乎零误差。

4. **效果次好但更有意义的是heuristic transfer**（AIME、GPQA）——没有universal algorithm，但category-level的strategy可以transfer。这更接近人类专家的"经验积累"。

5. **Memory curation是脆弱的瓶颈**——curator是同一个LLM自己，没有ground truth验证。如果curator判断失误，错误heuristic会进入memory污染后续queries。Paper里提到"errors cluster in embedding space"，错误的strategy会spread到neighboring queries。

---

## 实操上要注意的

- Generator和Curator是同一个LLM用不同prompt扮演的
- Curator的prompt是结构化的（Figure 14-15），有专门的section：Reusable Code Snippets / Solution Strategies / Verification / Reference
- Memory每次都要被curator重新regenerate一遍，token消耗不小（DC-Cu平均1831 tokens vs baseline 370 tokens）
- 检索用cosine similarity in embedding space，但具体embedding model paper没明说

---

## 我觉得最有意思的地方

这篇paper最打动我的不是技术复杂度（其实很朴素），而是它揭示的一个哲学point：**LLM可能不需要gradient updates就能实现某种形式的continual learning**。

memory + LLM-as-update-rule的组合，在黑盒API约束下实现了一种"穷人版meta-learning"。memory是fast pathway（text-based, 随时改），LLM weights是slow pathway（fixed, 不可动）。这种双时间尺度的架构和人脑的system 1 / system 2、hippocampus / neocortex的分工有某种结构上的相似性。

当然paper也坦白说了很多limitation——memory pollution、curator脆弱、小模型不行、verbose model不行、distribution shift敏感。但作为2026年test-time learning这条线上的早期exploration，它问对了一个问题：**如果不动weights，能在inference时学多远？**

答案看起来是：在某些任务上，能学很远。

---

# Dynamic Cheatsheet (DC): Test-Time Learning via Evolving External Memory

这篇paper的核心思想: 把一个黑盒LLM的inference过程从"独立事件序列"改造成"带有持续可演化memory的在线学习系统"。Memory作为text-level的外部存储, 在test time动态curate, 让LM跨queries积累可复用的strategies / code snippets / heuristics, 完全不动weights, 完全不需要ground truth labels。

参考链接:
- arXiv版本: https://arxiv.org/abs/2505.22877
- 项目主页(可能): https://dynamic-cheatsheet.github.io/
- 相关TextGrad: https://textgrad.com/
- Reflexion: https://arxiv.org/abs/2303.11366
- Self-Refine: https://arxiv.org/abs/2303.17651
- Buffer of Thoughts: https://arxiv.org/abs/2406.04271
- Self-RAG: https://arxiv.org/abs/2310.11511
- Toolformer: https://arxiv.org/abs/2302.04761
- Tree of Thoughts: https://arxiv.org/abs/2305.10601

---

## 1. 核心Intuition与定位

### 1.1 问题出发点

传统LM inference的尴尬处境: 给定 $\theta$ 固定的LM, inference可以看作是 $\tilde{y} = \pi_\theta(x)$ 这个mapping, 每次query独立处理。即便用chain-of-thought或majority voting, 那些compute都是ephemeral的 — 一旦生成完答案, 所有推理tokens全部丢弃, 下一个query从零开始。这与人脑完全不同, 人脑有episodic memory + semantic memory, 能把过去成功的strategy压缩成schema供后续调用。

DC的解法: 引入一个text-based external memory $M_i$ , 在test time演化。形式上类似于online Bayesian update在text space的analogue, 用curation prompt替代gradient step, 用retrieval替代attention。

### 1.2 三种test-time adaptation范式的对比

| Paradigm | 修改对象 | 是否需要梯度 | 是否需要ground truth | 可逆性 |
|---|---|---|---|---|
| Fine-tuning / Dynamic Evaluation | $\theta$ | 是 | 通常需要 | 否 |
| Static RAG | context (检索固定corpus) | 否 | 否 | 是 |
| **Dynamic Cheatsheet** | $M_i$ (外部memory, 演化) | 否 | 否 | 是 |

Reference dynamic evaluation: https://arxiv.org/abs/1904.08378

---

## 2. 方法论深度解析

### 2.1 DC-Cu: Cumulative Memory变体

设test sequence为 $(x_1, x_2, \ldots, x_n)$, $x_i \sim \mathcal{D}_{test}$, distribution未知。Memory state $M_i$ 在第 $i$ 步前的状态。

**Generation step (Eqn 1):**
$$\tilde{y}_i = \text{Gen}(x_i, M_i)$$

- $x_i$ : 第 $i$ 个input query
- $M_i$ : 当前memory state (text), 包含past queries归纳出的strategies
- $\text{Gen}$ : 由同一个LM通过特定prompting扮演的generator角色
- $\tilde{y}_i$ : candidate solution (注意tilde表示unverified candidate)

**Curation step (Eqn 2):**
$$M_{i+1} = \text{Cur}(M_i, x_i, \tilde{y}_i)$$

- $M_{i+1}$ : 下一step的memory state
- $\text{Cur}$ : curator role, 同一个LM换prompt
- Cur的三个curation axes: (i) 新答案的usefulness + generalizability, (ii) 现有entry的refine / remove, (iii) memory整体的clarity + compactness

**关键观察**: Cur没有access到ground truth label, 它必须self-assess correctness。这把verification责任完全交给LM自己, 既是优雅的设计也是脆弱点。

### 2.2 DC-RS: Retrieval & Synthesis变体

DC-Cu的两个问题: (a) memory update发生在answer生成之后, 当前query的insight在生成时未被memory吸收; (b) 没有保留raw (input, output) pairs, 只有abstracted heuristics。

DC-RS引入retrieval operator $\text{Retr}$:

**Retrieval (Eqn 3):**
$$R_i = \text{Retr}(x_i, \{(x_j, \tilde{y}_j)\}_{j<i}, k)$$

- $R_i$ : retrieved set of top-k past (input, output) pairs
- $\{(x_j, \tilde{y}_j)\}_{j<i}$ : all past pairs seen so far (growing corpus)
- $k$ : hyperparameter, top-k neighbors
- $\text{Retr}$ : 通常基于cosine similarity in embedding space (paper里未明说具体embedding, 推测用OpenAI text-embedding-3-large或类似)

**Pre-generation curation (Eqn 4):**
$$M_i = \text{Cur}(M_{i-1}, x_i, R_i)$$

注意时间戳: $M_{i-1}$ 是上一步结束时的memory, 现在用 $x_i$ + retrieved examples $R_i$ 来refine, 得到本步使用的 $M_i$。

**Generation (Eqn 5):**
$$\tilde{y}_i = \text{Gen}(x_i, M_i)$$

DC-RS的关键differences:
- 在生成 $\tilde{y}_i$ 之前就把memory从 $M_{i-1}$ 更新到 $M_i$ (incorporates current query signal)
- 同时保留abstracted heuristics + retrieved raw examples双重信息源

### 2.3 三个变体的information flow对比

```
DC-Cu:
  x_i --Gen(M_i)--> y_i --Cur(M_i, x_i, y_i)--> M_{i+1}
  
DC-RS:
  x_i --Retr({(x_j,y_j)})--> R_i
  R_i + M_{i-1} --Cur--> M_i  
  x_i + M_i --Gen--> y_i

DC-∅:
  x_i --Gen(空memory)--> y_i
  (no memory update)
  
FH (Full History):
  x_i + (x_1,y_1) + ... + (x_{i-1},y_{i-1}) --Gen--> y_i
```

### 2.4 Prompt engineering细节

从paper附录的Figure 12-15看:
- **BL prompt**: minimal instruction, 让model直接答题
- **Generator prompt (DC-∅/DR/FH/DC-Cu/DC-RS通用)**: structured instruction鼓励Python code generation + execution, 加了tool-use scaffolding
- **Curator prompt (DC-RS)**: 结构化template, sections包括"Reusable Code Snippets" + "Solution Strategies" + "Verification" + "Reference (Q5-Q20)"等

注意: DC-∅和BL的差距既反映了structured prompting的effect (代码工具鼓励), 也直接show出memory component的marginal contribution必须用DC-∅作为strong baseline来isolate。

---

## 3. 实验数据深度解读

### 3.1 主结果表 (Table 1) 详细分析

| Task | Claude BL | Claude DC-∅ | Claude DR | Claude DC-Cu | Claude DC-RS | GPT-4o BL | GPT-4o DC-∅ | GPT-4o DR | GPT-4o DC-Cu | GPT-4o DC-RS |
|---|---|---|---|---|---|---|---|---|---|---|
| AIME 2024 | 23.3 | 36.7 | 43.3 | **50.0** | 46.7 | 20.0 | 36.7 | 26.7 | 36.7 | 40.0 |
| AIME 2025 | 6.7 | 23.3 | 23.3 | **36.7** | 30.0 | 6.7 | 10.0 | 10.0 | 16.7 | 20.0 |
| AIME 2020–24 | 6.7 | 30.1 | 39.1 | 38.4 | **40.6** | 9.8 | 24.1 | 24.1 | 20.3 | 24.8 |
| Game of 24 | 12.0 | 10.0 | 11.0 | 14.0 | 14.0 | 10.0 | 19.0 | 6.0 | 93.0 | **99.0** |
| GPQA Diamond | 59.6 | 60.1 | 63.6 | 61.1 | **68.7** | 57.1 | 57.1 | 55.1 | 58.1 | 57.1 |
| Math Eqn Balancer | 44.8 | 56.4 | 60.4 | **100** | 97.8 | 50.0 | 88.0 | 100 | 100 | 99.2 |
| MMLU Pro Eng. | 61.2 | 57.2 | 65.2 | 66.8 | **67.6** | 53.2 | 51.6 | 48.8 | 44.0 | 51.2 |
| MMLU Pro Physics | 74.0 | 75.6 | 80.4 | 77.6 | **82.0** | 75.6 | 70.8 | 75.6 | 70.4 | 75.2 |

**关键patterns**:

(1) **AIME系列**: Claude在2024年从23.3%→50.0% (DC-Cu), 翻倍+27.7个points。AIME 2025从6.7%到36.7%, 提升30个points, 5.5x增益。AIME 2020-24 (133题) 从6.7%到40.6%, 6x增益。这反映了memory curation对math reasoning的累积收益。

(2) **Game of 24**: GPT-4o从10%到99%是最戏剧性的数字。原因: GPT-4o早期discover了Python brute-force solver, 之后所有queries直接retrieve这个template。Claude 3.5 Sonnet只从12%到14% — Claude倾向manual arithmetic, 不愿commit到Python approach。这是一个非常重要的finding: DC的效果高度依赖于base model的tool-use prior。

(3) **GPQA Diamond**: Claude从59.6%→68.7% (+9.1), GPT-4o从57.1%→58.1% (+1)。差距源于retrieval noise对GPT-4o的干扰 — paper 4.2提到"suboptimal examples被retrieved引入confusion"。

(4) **Math Eqn Balancer**: Claude 44.8%→100% (DC-Cu), GPT-4o 50%→100% (DR/DC-Cu)。Code-based approach一旦被stored, 后续100%成功。

(5) **MMLU Pro**: Claude在Physics +8 points, Engineering +6.4。GPT-4o反而slightly decrease, 暗示GPT-4o的curator在specialized domain上生成的memory质量不高。

### 3.2 Token efficiency对比

paper footnote 13给的数据:
- AIME 2024 Claude Sonnet平均tokens:
  - BL: 370 tokens
  - DC-∅: 494 tokens
  - DC-RS: 1035 tokens
  - DC-Cu: 1831 tokens

DC-Cu的token usage最高, 因为curator必须每次regenerate整个memory。这是一个implementation overhead — paper里讨论了"long-context generation vs understanding"问题, 建议未来用external database避免每次regenerate。

### 3.3 与Full-History对比 (Table 2)

| Task | Claude BL | Claude FH | Claude DC-Cu | GPT-4o BL | GPT-4o FH | GPT-4o DC-RS |
|---|---|---|---|---|---|---|
| AIME 2024 | 23.3 | 26.7 | **50.0** | 20.0 | 13.3 | **40.0** |
| AIME 2025 | 6.7 | 6.7 | **36.7** | 6.7 | 3.3 | **20.0** |

FH在GPT-4o上甚至worse than baseline (-6.7 on AIME 2024)。这印证了"naive context appending"的failure mode: context window fill up + signal被noise稀释 + retrieval efficiency下降。

### 3.4 vs Majority Voting (Table 4)

| Task | BL | MV(BL) | DC-∅ | MV(DC-∅) | DC-Cu |
|---|---|---|---|---|---|
| AIME 2024 | 23.3 | 23.3 | 36.7 | 33.3 | **50.0** |
| AIME 2025 | 6.7 | 6.7 | 23.3 | 23.3 | **36.7** |

MV在test-time compute scaling上完全无效, 而DC提供的是知识累积而非随机采样聚合。这区分了"self-consistency type"的test-time compute和"memory type"的test-time compute。

### 3.5 Smaller models (Table 3)

| Task | Claude Haiku BL | DC-∅ | DC-Cu | DC-RS |
|---|---|---|---|---|
| AIME 2024 | 10.0 | 26.7 | 36.7 | 30.0 |
| AIME 2025 | 0.0 | 13.3 | 13.3 | 10.0 |
| GPQA Diamond | 43.4 | 41.9 | 43.7 | **49.0** |

GPT-4o-mini的AIME 2024: BL 16.7 → DC-Cu 13.3 (regression!)。Smaller model的两个failure modes:
- Generative competence不足: 即使被DC scaffold, 也生成不出可复用的高质量solution, memory被wrong/low-quality content污染
- Curation capability不足: 无法有效refine memory或retrieve正确的past cases

---

## 4. 与Related Work的关系网络

### 4.1 Memory-augmented LMs谱系

- **Memory Networks (Weston et al., 2014)**: 早期explicit memory + addressing机制, 端到端trained。DC把这种思想搬到black-box API regime, addressing换成了text retrieval。
- **Neural Turing Machines (Graves et al., 2014)**: 类似NTM的可微memory, 但DC完全non-differentiable, curation通过prompting实现。
- **kNN-LM (Khandelwal et al., 2020)**: retrieval from fixed corpus, 但corpus不演化。DC相当于online kNN-LM。
- **RETRO (Borgeaud et al., 2022)**: 类似RAG但更efficient, retrieval corpus也是static。
- **RAG (Lewis et al., 2020)**: standard RAG, retrieval over fixed documents, 不evolve。Reference: https://arxiv.org/abs/2005.11401

### 4.2 Test-time adaptation谱系

- **Dynamic Evaluation (Krause et al., 2019; Rannen-Triki et al., 2024)**: 在test data上做gradient steps更新LM weights。Reference: https://arxiv.org/abs/1904.08378。DC是parameter-free版本。
- **Test-Time Training / TTT (Sun et al., 2020)**: CV domain, self-supervised loss on test instance。Reference: https://arxiv.org/abs/2006.10726
- **TTT++ (Liu et al., 2021)**: 改进的test-time training。
- **Tent (Wang et al., 2020)**: entropy minimization for TTA。
- **TTT with expressive hidden states (Sun et al., 2024)**: 这个非常相关, 用RNN-like hidden state实现test-time learning, 但hidden state是vector不是text。Reference: https://arxiv.org/abs/2407.04620

### 4.3 Reasoning-augmentation谱系

- **Chain-of-Thought (Wei et al., 2022)**: 单query内增加compute。Reference: https://arxiv.org/abs/2201.11903
- **Self-Consistency / Majority Voting (Wang et al., 2023)**: 多次采样 + 投票。Reference: https://arxiv.org/abs/2203.11171
- **Tree of Thoughts (Yao et al., 2023)**: 搜索 + backtracking。Reference: https://arxiv.org/abs/2305.10601
- **Graph of Thoughts (Besta et al., 2024)**: ToT的graph extension。Reference: https://arxiv.org/abs/2308.09687
- **Pathfinder (Golovneva et al., 2023)**: guided search。
- **Least-to-Most Prompting (Zhou et al., 2022)**: decompose problem。Reference: https://arxiv.org/abs/2205.10625

这些都是**ephemeral**的compute, 一旦query处理完, 所有tokens丢弃。DC把compute **persist** 到memory, 跨query复用。

### 4.4 Self-correction / Reflection谱系

- **Reflexion (Shinn et al., 2023)**: 单query内通过verbal feedback iterative refine。Reference: https://arxiv.org/abs/2303.11366
- **Self-Refine (Madaan et al., 2023)**: 类似Reflexion。Reference: https://arxiv.org/abs/2303.17651
- **Critic / Self-Critic (Gou et al., 2023)**: tool-interactive critiquing。Reference: https://arxiv.org/abs/2305.11738
- **Self-RAG (Asai et al., 2023)**: 学习何时retrieve + critique。Reference: https://arxiv.org/abs/2310.11511

这些工作在单task内做refine, DC把insights累积到memory。

### 4.5 Tool use / code execution谱系

- **Toolformer (Schick et al., 2023)**: 训练LM学会调用tools。Reference: https://arxiv.org/abs/2302.04761
- **Chameleon (Lu et al., 2023)**: plug-and-play compositional reasoning with tools。Reference: https://arxiv.org/abs/2304.04870
- **HuggingGPT (Shen et al., 2023)**: orchestrating HF models。Reference: https://arxiv.org/abs/2303.17580
- **ViperGPT (Surís et al., 2023)**: Python execution for vision reasoning。
- **ToolLLM (Qin et al., 2023)**: 16000+ real-world APIs。
- **Meta-Prompting (Suzgun & Kalai, 2024)**: task-agnostic scaffolding, 第一作者的前作。

DC的发现: 一旦LLM在DC框架内discover一个Python solution template, 它会persistently reuse。这显示了"learning to use tools"在test-time可以emerge, 不需要fine-tuning。

### 4.6 思维模板类工作

- **Buffer of Thoughts (Yang et al., 2025)**: distill thought templates。Reference: https://arxiv.org/abs/2406.04271。BoT用predefined templates, DC让templates emerge from test data。
- **Thought-Retriever (Feng et al., 2024)**: retrieve past chain-of-thought。Reference: https://openreview.net/forum?id=SkDNQbMQba
- **STaR (Zelikman et al., 2022)**: bootstrapping reasoning with reasoning, 但是training-based。

### 4.7 Gradient-based text optimization

- **TextGrad (Yuksekgonul et al., 2025)**: textual gradients, 同一作者group。Reference: https://textgrad.com/。DC可以看作是TextGrad在memory层面的扩展 — TextGrad优化单次solution, DC优化累积的memory。

### 4.8 Continual learning

- **GEM (Gradient Episodic Memory, Lopez-Paz & Ranzato, 2017)**: 防止catastrophic forgetting的gradient方法。DC的text-level memory天然避免了catastrophic forgetting (因为不更新weights), 但可能遇到"textual drift"。
- **Catastrophic interference (McCloskey & Cohen, 1989)**: classical continual learning problem。

---

## 5. Memory curation的具体机制详解

从Figure 6的Claude memory excerpt看, 经过20个AIME 2024 questions后memory包含:
- **Solution Verification Strategy**: 一般性的verify方法
- **Reference (Q5-Q20)**: 指向过去specific questions
- **Reusable Code Snippets**: 数学计算Python code
- **Domain-specific heuristics**: 数论、组合、几何的tricks

Figure 5的GPT-4o memory (Game of 24 after 100 examples):
- 一个固定的Python brute-force function
- 描述了permutation of 4 numbers + operators的search logic
- 一个简单的result verification机制

Curation的implicit objective function可以formalize为:

$$\text{Cur}^* = \arg\min_{\text{Cur}} \mathbb{E}_{x \sim \mathcal{D}_{test}} \left[ \mathcal{L}(\text{Gen}(x, M_{\text{Cur}}(x)), y^*(x)) \right] + \lambda \cdot |M| $$

其中 $|M|$ 是memory size penalty, $\mathcal{L}$ 是task loss, $y^*(x)$ 是ground truth (但Cur没有access)。Curator必须用self-assessment proxy来近似。

---

## 6. 关键failure modes与局限性

### 6.1 Memory pollution

错误的heuristic一旦进入memory会被反复retrieve, 在embedding space内spread到neighboring queries (paper Figure 10的t-SNE visualization展示了这个clustering效应)。需要更sophisticated的verification mechanism。

### 6.2 Cold start problem

前几题memory基本为空, performance和DC-∅接近。Memory quality随queries累积提升 — 这是典型的online learning pattern, 见Figure 7的GPQA-Diamond cumulative accuracy curve。

### 6.3 Distribution shift sensitivity

如果test queries来自diverse domains, memory可能noisy。Paper提到hierarchical memory是promising direction — 为每个domain维护separate memory bank。

### 6.4 Order dependence

Paper section 4.6提到: 当相关题目cluster在一起时DC效果更好。这暗示curriculum learning (Bengio et al., 2009) 能amplify DC效果。Reference: https://dl.acm.org/doi/10.1145/1553374.1553380

### 6.5 Long-context generation bottleneck

Curator需要每次regenerate整个memory, paper 4.6讨论到Claude有时abbreviates memory ("Previous content [...] preserved")而不是explicit重写。这导致memory quality drift。Solution: external database + pointer-like reference。

### 6.6 Small model limitation

GPT-4o-mini和Claude Haiku的DC效果有限。原因: (a) generative competence不足, memory被low-quality content污染; (b) curation capability不足, 无法有效refine。这把DC的适用范围限制在strong base models。

### 6.7 R1 models表现差

paper 5节最后提到: DeepSeek R1和o1在DC下minimal improvement, 原因是solutions "far too verbose and long" — verbose的CoT占据memory space, curation效率低。这暗示DC对concise solution有偏好。

---

## 7. 直觉构建: DC作为test-time compute的另一种scaling

### 7.1 Two regimes of test-time compute scaling

**Regime 1: Per-instance compute scaling**
- 同一个query, 多花compute (CoT, ToT, majority voting, best-of-N)
- Cost: linear in compute
- Benefit: bounded by single instance reasoning ceiling

**Regime 2: Cross-instance knowledge accumulation (DC)**
- 多个queries共享一个累积的knowledge base
- Cost: per-instance curation overhead
- Benefit: amortized over future similar queries, can grow unboundedly with task distribution overlap

DC位于regime 2, 与现有的test-time compute scaling正交。可以想象两者组合: per-instance ToT + cross-instance DC。

### 7.2 DC作为meta-learning的textual实现

MAML (Finn et al., 2017) 在gradient space做meta-learning: learn an initialization $\theta_0$ that's amenable to few-shot adaptation. DC在text space做meta-learning: maintain一个memory $M$ that's amenable to retrieval + curation。

形式上, DC的update rule:
$$M_{i+1} = \text{Cur}(M_i, x_i, \tilde{y}_i)$$

可以看作是Reptile-like update在text space的analogue, 用LLM's prompt-driven in-context reasoning替代gradient step。

### 7.3 DC和"Learning to Learn at Test Time" (TTT layers, Sun et al., 2024)的关系

TTT layers (https://arxiv.org/abs/2407.04620) 把hidden state看作fast weight, 在test time用self-supervised loss更新hidden state。DC用text-based memory替代vector hidden state, 用prompted curation替代gradient update。两者的哲学相似, 实现路径不同。

### 7.4 为什么Game of 24 + GPT-4o效果戏剧性?

我推测有以下几个reasons叠加:
1. Game of 24是个well-defined search problem, 有deterministic correct answer
2. GPT-4o的training data里可能包含类似puzzle的Python solvers
3. 一旦Python solver被discover并stored, 后续queries只是"调函数", 几乎no error
4. GPT-4o倾向code generation over manual arithmetic (vs Claude的相反倾向)
5. Game of 24的题目结构highly similar, retrieved examples几乎总是relevant

这说明: DC在任务有"algorithmic closure" (存在一个fixed algorithm能解决所有instances) 且模型倾向工具使用时, 效果最dramatic。

### 7.5 为什么AIME效果不如Game of 24 uniform?

AIME每道题unique, 不存在一个"universal algorithm"。Memory里累积的是heuristics (e.g., "对于polynomial roots问题用Vieta's formulas", "对于combinatorics问题先check Pigeonhole"), 这些heuristics的transfer需要更复杂的匹配。因此AIME 2024的+27%和AIME 2025的+30%是真正的"reasoning transfer", 比Game of 24的+89%更有scientific意义。

---

## 8. 延伸联想与潜在方向

### 8.1 Memory as program synthesis

Game of 24的成功提示我们: DC的最强形式可能是program synthesis — 让LLM在memory中累积一个library of Python functions, 每个function解决一个class of problems。这接近Neuro-Symbolic Programming (e.g., DreamCoder: https://arxiv.org/abs/2006.08381)。

### 8.2 Hierarchical memory

paper 5节提到hierarchical memory的可能性。具体可以设计:
- **Level 1**: Domain-level memory (math, biology, physics)
- **Level 2**: Subdomain (algebra, geometry, combinatorics)
- **Level 3**: Problem-type-level memory (polynomial roots, modular arithmetic)

Retrieval时先用x_i的embedding路由到合适的level, 然后在该level内细粒度retrieve。

### 8.3 Memory verification via consensus

Curator的self-assessment是脆弱点。可以引入consensus mechanism:
- 多个independent curators vote on whether to commit a strategy
- 或者用self-consistency (Wang et al., 2023) on the candidate solution before storing

### 8.4 Cross-model memory transfer

paper提到Claude的memory能否transfer给GPT-4o或smaller models? 实验结果是mixed。但理论上, 如果memory是model-agnostic (e.g., Python code, mathematical formulas), 应该能transfer。需要更深入实验。

### 8.5 Active forgetting

paper没有explicit forgetting mechanism, curation时由LLM decide whether to prune。可以引入forgetting curve (Ebbinghaus-style) 或者recency-weighted retention, 模拟人脑的spaced repetition效应。Reference for spaced repetition: https://en.wikipedia.org/wiki/Spaced_repetition

### 8.6 Memory consolidation via sleep

人脑在sleep时consolidate episodic memory到semantic memory。可以设计一个offline phase, 用LLM在test time之外batch process memory, distill成更compact的semantic rules。

### 8.7 Memory + RL

如果未来给LM加上RL feedback (e.g., 通过tool execution verification或external verifier), memory curation会从self-supervised升级为externally-verified。这是hybrid approach的potential。

### 8.8 Curriculum design

paper section 4.6提到"curriculum-style learning"可能amplify DC效果。可以设计一个meta-controller, 在test time动态排序queries, 让相关queries cluster together, 加速memory build-up。

### 8.9 DC + Reasoning models (R1/o1)

paper提到R1-style models verbose, 不适合DC。但如果用 distilled R1 outputs (concise final answers + compressed reasoning), 可能能fit到DC memory。这是open question。

### 8.10 Connection to LLM agents

DC是single-step agent loop的简化版。Multi-agent framework (e.g., AutoGPT, BabyAGI, https://arxiv.org/abs/2304.03442 for BabyAGI) 也用memory, 但是episodic + 工具调用, 没有DC的curation discipline。可以想象一个"DC as agent memory layer"的设计。

### 8.11 Theoretical analysis

从learning theory角度, DC的sample complexity如何? 给定 $\mathcal{D}_{test}$ 的structure (e.g., low intrinsic dimensionality, cluster structure), DC需要多少queries才能达到error $\epsilon$? 这是open problem。可以借鉴kNN theory (Cover & Hart, 1967) + meta-learning theory。

### 8.12 Memory size scaling

paper没有systematically study memory size $|M|$ vs performance。预期有saturation curve: $|M|$ 太小information不足, 太大检索noise上升 + context cost增加。最优 $|M|^*$ 应该是 $\mathcal{D}_{test}$ entropy的function。

---

## 9. 总结性的intuition

**DC的本质**: 把"test-time compute scaling"从per-instance空间扩展到cross-instance空间。Memory $M_i$ 是test distribution $\mathcal{D}_{test}$ 的online-sufficient-statistics的text approximation, 通过LLM-as-curator进行Bayes-like update。

**DC的magic**: 不需要任何training data labels, 不需要gradient updates, 完全black-box compatible。在数学竞赛、knowledge QA、algorithmic puzzle上都show出显著gain (5-30个points不等)。

**DC的limitation**: 依赖strong base model的 (a) generative competence, (b) curation capability, (c) tool-use prior。Smaller models或verbose reasoning models效果有限。

**DC的philosophical position**: 介于RAG和fine-tuning之间 — 像RAG一样不动weights, 像fine-tuning一样有"learning"signal, 像memory-augmented networks (NTM/Memory Networks)一样有external state, 但实现完全在text space + black-box API约束下。

**DC对future AGI的hint**: 真正的continual learning agent可能不需要gradient updates, 只需要一个well-designed text memory + 一个strong LLM作为implicit update rule。这与LeCun的JEPA (https://openreview.net/forum?id=BZ5a1r-kVsf) 和MemoryLLM (https://arxiv.org/abs/2402.04624) 等工作有思想上的共鸣。

这篇paper在2026年回看, 是test-time learning这条线上的重要milestone, 把"online adaptation for LLMs"从gradient-based regime推进到fully black-box text-based regime。后续工作可能在memory hierarchy、verification mechanism、curriculum learning等方向深化。
