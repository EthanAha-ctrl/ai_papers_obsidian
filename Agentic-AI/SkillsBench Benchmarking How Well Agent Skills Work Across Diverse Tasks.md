---
source_pdf: SkillsBench Benchmarking How Well Agent Skills Work Across Diverse Tasks.pdf
paper_sha256: fa853b3a44282a28bd816b46572e8fbab945834f01fb5f055ec8847ba1c31dae
processed_at: '2026-08-12T07:32:18-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲SkillsBench

## 一句话总结

很多人给AI agent配了"Skills"(操作手册), 但从来没人系统测过这玩意儿到底有没有用。这篇paper搞了84个task、7个model、7308条trajectory, 发现**人工精心编写的好Skills平均能让pass rate涨16.2个点, 但让AI自己写Skills给自己用基本没用**。

## 这篇paper到底在测什么

以前所有agent benchmark (SWE-bench, Terminal-Bench, AgentBench) 都在问: "这个model能不能干这件事?" SkillsBench问的是另一个问题: "配上Skills之后, model干这件事的能力提升了多少?"

这两个问题听起来像, 实际完全不同。前者measure capability, 后者measure augmentation的efficacy。后者必须用**paired evaluation**——同一个task, 同一个model, 一组带Skills一组不带, 直接对比。不做paired comparison, 你根本不知道提升是来自Skills本身还是来自model本来就行。

参考: https://arxiv.org/abs/2601.11868 (Terminal-Bench)

## Skill到底是什么

先clarify一个混乱的概念空间。Table 1很关键, 把四种augmentation paradigm拉出来对比:

| 维度 | Prompts | RAG | Tools | Skills |
|------|---------|-----|-------|--------|
| Modular/reusable | × | √ | √ | √ |
| Procedural guidance | Limited | × | × | √ |
| Executable resources | × | × | √ | √ |
| Cross-model portable | √ | √ | × | √ |

直觉上: 
- Prompt是临时一次性指令, 没结构没复用
- RAG给你facts, 不告诉你怎么做事
- Tools给你能力, 不告诉你什么时候用、怎么用
- Skill是"操作手册+工具包", 既告诉你步骤, 又附上能跑的代码

具体来说, 一个Skill就是一个文件夹, 里面有个SKILL.md (主说明书, 带YAML frontmatter写name和description), 可选的scripts/目录放能跑的代码, references/目录放参考文档。

这个定义不是随便划的。它explicitly排除: system prompt (没结构), few-shot example (declarative不是procedural), RAG retrieval (factual不是procedural), tool doc (描述能力不描述流程)。Skill核心是**procedural knowledge**——怎么做一类事, 不是怎么做一个具体instance。

背后的理论根基是Sutton 1999的options framework (RL里的temporal abstraction) 和Sumers 2023的CoALA (语言agent的认知架构)。Skill本质上是language agent的"option", 把长horizon任务拆成可复用的procedural chunk。

参考: 
- Options framework: https://www.sciencedirect.com/science/article/pii/S0004370299000521
- CoALA: https://arxiv.org/abs/2309.02427

## Benchmark怎么设计的

每个task四件套:
- **Instruction**: 人写的任务描述 (用GPTZero + 人审双重验证不是LLM生成的)
- **Environment**: Docker容器, 里面有data和skills/子目录
- **Solution**: oracle解法, 必须100%通过所有test
- **Verifier**: deterministic的pytest assertion, 不要LLM-as-judge

为什么deterministic verification这么重要? LLM-as-judge有variance, 同一个output让judge打分两次可能不一样, 这会污染你对Skills效果的measurement。你要测的是Skills有没有用, 不是judge稳不稳定。

最关键的设计是**Leakage Prevention**。Skill不能直接encode答案:
- 不能有task-specific的filename/path/identifier
- 不能有解决benchmark task的exact command sequence
- 不能有task spec里的constants/magic numbers
- 不能引用具体的test case或expected output

CI里跑一个Claude Code Agent SDK-based的validation agent做leakage audit, 不通过的task直接reject。没有这个, 你测出来的"Skills有用"可能只是"Skills泄露了答案"。

数据集来源: 105个contributor提交322个candidate task, 经过automated CI + 人审 + benchmark experiment三关, 86个通过 (26.7% acceptance rate), 跨11个domain。Difficulty按human completion time分: Core (<60min, 19.8%), Extended (1-4h, 50.0%), Extreme (>4h, 30.2%)。

参考: https://openreview.net/forum?id=E58HNCqoaA (agentic benchmark best practices)

## 实验怎么跑的

7个agent-model configuration:
- Claude Code × {Opus 4.5, Opus 4.6, Sonnet 4.5, Haiku 4.5}
- Gemini CLI × {Gemini 3 Pro, Gemini 3 Flash}
- Codex × GPT-5.2

3个Skills condition:
- No Skills (baseline)
- With Curated Skills (人写的高质量Skills)
- Self-Generated Skills (让agent先自己写Skills再用——只在Claude Code和Codex上跑, Gemini CLI的explicit skill-activation interface不支持)

每个task-model-condition组合跑5次, temperature=0, deterministic sampling。总共7,308条valid trajectory。

Self-Generated condition的设计很精妙。给agent的prompt大意是: "先分析task需要什么domain knowledge, 写1-5个modular skill document存到environment/skills/, 然后用你写的skills当reference解task"。这个condition isolate了LLM latent domain knowledge的效果。如果self-generated和curated效果一样, 说明Skills的价值在"结构化组织知识"这个动作本身; 如果self-generated明显差, 说明价值在人curated的domain expertise。

## 主要发现

### Finding 1: Skills有用, 但variance巨大

平均+16.2pp, 但range从+13.6pp到+23.3pp。Variance大说明Skills efficacy强依赖具体model-harness组合。Claude Code + Opus 4.5最高 (+23.3pp), 部分因为Claude Code的native Skills integration是专门针对Anthropic Agent Skills spec优化的。

### Finding 2: Gemini 3 Flash是performance冠军

Gemini CLI + Gemini 3 Flash拿到48.7% pass rate (with Skills), 是所有configuration里最高的。但它的normalized gain只有25.3%, 因为它baseline就有31.3%。直觉: Flash baseline强, Skills主要补procedural gaps, 不是补基础能力。

### Finding 3: Self-Generated Skills基本没用, 甚至有害

这是最important也最反直觉的finding。平均-1.3pp, Codex + GPT-5.2甚至-5.6pp, 只有Opus 4.6有+1.4pp的modest gain。这与curated Skills的+16.2pp形成sharp contrast。

Trajectory分析揭示两种failure mode:
1. Model识别出需要domain knowledge, 但生成的procedure不精确不完整 (比如写"use pandas for data processing"而没有具体API pattern)
2. 对high domain-knowledge task (manufacturing, financial), model甚至无法recognize需要specialized Skills, 用general-purpose approach硬干

这个finding的深层含义: **LLM不能reliably author它自己能消费的procedural knowledge**。听起来反直觉——model能理解一个Skill, 为什么不能generate它?

我的解读是asymmetry来自三个地方:
- **Tacit knowledge问题**: 真正有用的procedural knowledge包含"看起来显然但容易遗漏"的细节。Human expert通过实践内化这些细节, model从pretraining distribution里提不出来
- **Self-knowledge gap**: Model不知道自己不知道什么。生成Skill时它generate的是"它认为需要的", 而不是"它实际需要的"
- **Distribution mismatch**: Pretraining见过的procedural text多是tutorial/documentation style, 缺乏task-specific的failure mode知识

### Finding 4: Domain间差异巨大

| Domain | With Skills | No Skills | Δ |
|--------|-------------|-----------|---|
| Healthcare | 86.1% | 34.2% | +51.9 |
| Manufacturing | 42.9% | 1.0% | +41.9 |
| Cybersecurity | 44.0% | 20.8% | +23.2 |
| Natural Science | 44.9% | 23.1% | +21.9 |
| Energy | 47.5% | 29.5% | +17.9 |
| Office & White Collar | 42.5% | 24.7% | +17.8 |
| Finance | 27.6% | 12.5% | +15.1 |
| Media & Content | 37.6% | 23.8% | +13.9 |
| Robotics | 27.0% | 20.0% | +7.0 |
| Mathematics | 47.3% | 41.3% | +6.0 |
| Software Engineering | 38.9% | 34.4% | +4.5 |

直觉: pretraining覆盖强的domain (SWE, Math) Skills效果小, model本来就有strong prior; pretraining覆盖弱但procedural knowledge重要的domain (Healthcare, Manufacturing) Skills效果巨大。Manufacturing baseline只有1.0%, 几乎是"无Skill不可解"的task class。

### Finding 5: 2-3个Skill是sweet spot

| Skills Count | With Skills | No Skills | Δ |
|--------------|-------------|-----------|---|
| 1 skill | 42.2% | 24.4% | +17.8 |
| 2-3 skills | 42.0% | 23.4% | +18.6 |
| 4+ skills | 32.7% | 26.9% | +5.9 |

Non-monotonic关系。4+个Skill反而只有+5.9pp。直觉: 太多Skills引入cognitive overhead或conflicting guidance。LLM的attention在context里检索信息时, signal-to-noise ratio很关键, 太多Skills稀释attention。

### Finding 6: 简洁的Skill优于comprehensive的

| Complexity | Pass Rate | Δ | N |
|------------|-----------|---|---|
| Detailed | 42.7% | +18.8 | 1165 |
| Compact | 37.6% | +17.1 | 845 |
| Standard | 37.1% | +10.1 | 773 |
| Comprehensive | 39.9% | -2.9 | 140 |

Comprehensive Skills actually hurt performance (-2.9pp)。这是一个很强的design principle。我怀疑comprehensive documentation倾向于提供multiple approaches, 而model执行时需要commit to一个approach。Multiple approaches在context里造成decision paralysis, 或者agent执行中混合不同approach导致incoherent solution。

### Finding 7: 小model + Skills > 大model without Skills

Claude Haiku 4.5 with Skills = 27.7%, Haiku without Skills = 11.0%, 提升+16.7pp。
Claude Opus 4.5 without Skills = 22.0%。

所以**Haiku + Skills (27.7%) > Opus 4.5 without Skills (22.0%)**。

经济意义巨大: 对于procedural task, 投资Skill curation可能比投资更大model更cost-effective。

## Normalized Gain公式

$$g = \frac{pass_{\text{skill}} - pass_{\text{vanilla}}}{1 - pass_{\text{vanilla}}}$$

变量解释:
- $g$: normalized gain, 衡量从baseline向perfect score的"相对进步比例"
- $pass_{\text{skill}} \in [0, 1]$: with Skills条件下的pass rate
- $pass_{\text{vanilla}} \in [0, 1]$: no Skills条件下的baseline pass rate
- $1 - pass_{\text{vanilla}}$: 从baseline到perfect (100%)的"headroom", 即还能改进多少

分子是absolute improvement, 分母是max possible improvement, 所以$g$本质是"实际提升占可提升空间的比例"。

这个公式来自Hake 1998, 美国物理教育研究的经典metric, 原本用来比较传统讲授vs interactive engagement在物理课上的效果。引入它是为了避免ceiling effect的confounding。

为什么需要它? 假设model A baseline 90%, 加Skills到95%, $\Delta = +5\text{pp}$; model B baseline 10%, 加Skills到55%, $\Delta = +45\text{pp}$。直接比较$\Delta$会说B受益更多。但算normalized gain: A的$g = \frac{5}{10} = 0.5$, B的$g = \frac{45}{90} = 0.5$, 一样! 这说明两者都是把"可改进空间"利用了一半, A受ceiling effect限制看起来$\Delta$小, 实际scaffolding效率相同。

Paper明确说: "High $g$ with low $\Delta_{\text{abs}}$ suggests ceiling effects; high $g$ with high $\Delta$ suggests substantial scaffolding。" 这是正确解读。

验证Table 3里Claude Code + Opus 4.5: $\Delta = +23.3\text{pp}$, $g = 29.9\%$, $pass_{\text{vanilla}} = 22.0\%$, $pass_{\text{skill}} = 45.3\%$。算: $\frac{45.3 - 22.0}{100 - 22.0} = \frac{23.3}{78.0} = 0.2987 \approx 29.9\%$。✓

参考: https://doi.org/10.1119/1.18809 (Hake 1998)

## 失败模式分析

Paper把5,171个agent failure分类成5大类:

| Category | Failure Mode | Count | % |
|----------|--------------|-------|---|
| Execution | No Output Produced | 411 | 7.9 |
| Execution | Domain Knowledge Gap | 251 | 4.9 |
| Execution | Specification Violation | 171 | 3.3 |
| Execution | Incorrect Implementation | 68 | 1.3 |
| Execution | Tool/Environment Failure | 16 | 0.3 |
| Coherence | Incomplete Solution | 527 | 10.2 |
| Verification | Quality Below Threshold | 2,577 | 49.8 |
| Timeout | Agent Timeout | 922 | 17.8 |
| Unknown | Unclassified | 228 | 4.4 |

**Quality Below Threshold占49.8%是最dominant的failure mode**。Agent通常understand task structure, produce了output, 但output的accuracy不够。这对应Terminal-Bench TAT里的"Weak Verification"——agent的self-check无法catch质量问题。

这正好是Skills最能help的地方。Skills encode "verifier-facing details (steps, constraints, sanity checks)", 直接address quality issue。

Table 17比较No Skills vs With Skills的failure分布:

| Condition | Fail Rate | Timeout | Execution | Coherence | Verification |
|-----------|-----------|---------|-----------|-----------|--------------|
| No Skills | 78.4% | 16.1% | 17.1% | 10.7% | 52.1% |
| With Skills | 61.1% | 18.6% | 21.1% | 8.9% | 46.6% |
| Self-Generated | 80.9% | 19.9% | 13.9% | 11.2% | 50.4% |

关键观察:
- Skills主要reduce Verification failure (绝对数1,184→819, -30.8%)
- Skills稍微增加Timeout的相对share (绝对数367→328, 但share从16.1%到18.6%)——Skills让agent pursue更ambitious strategy, 有时超时
- Skills reduce Coherence failure 35.8% (243→156)——Skills帮agent identify所有required deliverable
- **Self-Generated的fail rate (80.9%)比No Skills (78.4%)还高**——agent花时间generate Skills然后又用错, 反而更糟

## 成本效率

Table 12的token usage:

| Model | Condition | Input (K) | Output (K) | Total (K) |
|-------|-----------|-----------|------------|-----------|
| GPT-5.2 | No Skills | 961 | 12.2 | 974 |
| GPT-5.2 | With Skills | 1,087 | 11.6 | 1,099 |
| Gemini 3 Flash | No Skills | 985 | 14.2 | 999 |
| Gemini 3 Flash | With Skills | 1,075 | 12.1 | 1,087 |
| Gemini 3 Pro | No Skills | 495 | 12.0 | 507 |
| Gemini 3 Pro | With Skills | 465 | 10.9 | 476 |

两个有意思的观察:

1. **Gemini 3 Flash用2.3×的input token比Gemini 3 Pro** (1.08M vs 0.47M with Skills)。Paper叫这"compensatory strategy where the smaller model substitutes iterative exploration for reasoning depth"。小model用iteration补偿reasoning capacity, 经典pattern。

2. **Gemini 3 Pro with Skills的token usage反而下降6%**——Skills让Pro更efficient, 减少exploration rounds。这是Skills除了accuracy之外的另一个benefit: efficiency。

成本上, Table 13:
- Gemini 3 Flash: $0.54 (No Skills) → $0.57 (With Skills), +6%
- GPT-5.2: $1.85 → $2.07, +12%
- Gemini 3 Pro: $1.13 → $1.06, -6% (Skills让Pro更便宜!)

Cache hit rate: GPT-5.2 91-92%, Gemini 3 Pro 75-76%, Gemini 3 Flash 63-67%, Claude Code >99%。实际deployed cost会比standard rate低50-90%。

## 生态现状

Paper附录分析了47,150个unique Skills的ecosystem:
- 来源: GitHub 12,847 + marketplace 28,412 + corporate 5,891, dedup后47,150
- Size: log-normal, median 2.3 KB, top 1% >50 KB
- Token count: median ~1.5k tokens
- Domain: Software Development 38%, Data Analysis 22%, DevOps 15%, Writing 12%, Other 13%
- 78%是standard SKILL.md + optional resources结构
- 大部分只有1个file, 绝大多数<5 files
- Markdown主导 (documentation-heavy)

质量评分 (4维度×3分=12分): ecosystem mean只有6.2/12 (SD=2.8)。Benchmark选top quartile (≥9/12, mean 10.1/12), 所以paper的result是**optimistic scenario**。Real-world deployment涉及更低质量Skills和imperfect task-skill matching, 效果会更差。这是paper自己承认的关键limitation。

## 我觉得最重要的几个insight

### 1. Paired evaluation是augmentation research的gold standard

任何新augmentation (RAG variants, tool use strategies, reflection protocols) 都应该paired eval, 不然你测的不是augmentation的effect, 是baseline的capability。这个methodology可以推广。

### 2. LLM的瓶颈不是concept, 是execution-time procedural reliability

Skills最helpful的地方是"concrete procedures and verifier-facing details"。LLM知道concept, 在长execution trajectory中无法可靠执行正确procedure。Skills本质上是externalized procedural memory, 减轻working memory负担。这与cognitive science的procedural vs declarative memory区分一致。Pretraining主要acquire declarative, Skills提供procedural。

### 3. Self-generated Skills无效是一个information-theoretic limit

Model生成Skill时generate的是"它自己会做什么", 而不是"什么是最好的procedure"。如果model不知道某个API的某个gotcha, 它生成的Skill里也不会包含那个gotcha。这不是capability的limit, 是information-theoretic的limit。

一个有意思的future direction paper没做: 让strong model生成Skill给weak model用。这test了"authoring capability"和"consumption capability"的分离。如果这个work, 就有"知识蒸馏通过Skills"的新pathway。

### 4. Harness是hidden confound

Codex CLI "frequently neglects provided Skills—agents acknowledge Skills content but often implement solutions independently"。这是harness-level的Skills utilization failure, 不是model capability failure。Claude Code + Opus 4.5的+23.3pp部分来自Claude Code native Skills integration optimized for Agent Skills spec。Skills efficacy强依赖harness implementation, 未来需要decouple harness effect和model effect。

### 5. Comprehensive Skills的negative effect揭示了一个deeper design principle

Comprehensive documentation倾向于提供multiple approaches, 而model执行需要commit to一个approach。Multiple approaches在context里造成decision paralysis或execution inconsistency。Skill design的核心是**focus**, 不是completeness。

## Limitations和future work

Paper自承的:
1. 只评terminal-based containerized task, 不能直接transfer到GUI agent, multi-agent, very long-horizon workflow
2. Skills injection增加context length, observed gain可能partly是"more context"而非procedural structure (需要length-matched baseline来isolate)
3. Containerization提供state isolation但非perfect determinism, 也无法完全消除training-set leakage
4. Benchmark Skills质量高 (mean 10.1/12) vs ecosystem mean (6.2/12), 是optimistic scenario

我加的:
- Skill-Skill interaction没分析。多Skills是additive, sub-additive还是super-additive? Table 5的non-monotonic trend暗示sub-additive (interference), 但需要更系统的研究
- Strong model authors Skills for weak model这个方向值得做
- Multi-modal skills for vision-language agent是obvious next step
- Skills的temporal evolution没研究: 随着model upgrade, 哪些Skills失效, 哪些新Skills需要?

## 实操建议

基于findings, 给Skills author的practical guidance:
1. **Concise > exhaustive**: Detailed或compact > comprehensive
2. **Stepwise + 至少一个working example**最effective
3. **2-3 modules**是sweet spot, 别贪多
4. **Explicitly match harness constraint** (比如JSON-only protocol要repeated format reminder)
5. **Modular Skills compose better** on multi-part task
6. **Focus on procedural gap**, 别写model已经会的concept

## 总结

这篇paper的value不只在于findings, 更在于它确立的methodology: 把augmentation strategy当first-class evaluation artifact, 用paired evaluation measure真实efficacy, 用deterministic verification保证reproducibility, 用leakage audit保证Skills提供guidance而非solution。

最deep的finding是: **LLM agent的瓶颈在procedural reliability不在conceptual knowledge, 而procedural knowledge不能靠model自己generate, 必须human curate**。这既有实践意义 (投资Skill curation) 也有理论意义 (procedural vs declarative的asymmetry)。

参考链接汇总:
- Paper核心: https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- Terminal-Bench: https://arxiv.org/abs/2601.11868
- SWE-bench: https://openreview.net/forum?id=VTF8yNQM66
- Anthropic Agent Skills: https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- MCP: https://www.anthropic.com/news/model-context-protocol
- Options framework: https://www.sciencedirect.com/science/article/pii/S0004370299000521
- CoALA: https://arxiv.org/abs/2309.02427
- Hake 1998: https://doi.org/10.1119/1.18809
- Best practices for agentic benchmarks: https://openreview.net/forum?id=E58HNCqoaA
- Harbor framework: https://github.com/laude-institute/harbor

---

# SkillsBench: 一篇关于Agent Skills功效测量的benchmarking paper

## 1. Paper的定位和核心问题

这篇paper试图回答一个看似简单但被现有benchmark忽略的问题: **"Skills Y在任务X上到底提升了多少performance?"** 而不是传统benchmark问的"模型能不能完成任务X"。

这是一个重要的方法论转向。现有的agent benchmark——SWE-bench (Jimenez et al., 2024), Terminal-Bench (Merrill et al., 2026), AgentBench (Liu et al., 2023), WebArena (Zhou et al., 2024b), OSWorld (Xie et al., 2024)——都是measure baseline capability, 把agent当fixed system来eval。SkillsBench把augmentation strategy本身变成first-class evaluation artifact, 采用paired evaluation (with Skills vs without Skills在同一task上对比), 这样才能isolate Skills的causal effect。

参考链接:
- Terminal-Bench: https://arxiv.org/abs/2601.11868
- SWE-bench: https://openreview.net/forum?id=VTF8yNQM66
- AgentBench: https://arxiv.org/abs/2308.03688
- OSWorld: https://arxiv.org/abs/2404.07972

## 2. Skills的定义: 一个操作化的分类学

Paper给出Skills的四个criteria, 这值得仔细看, 因为它定义了一个新的augmentation paradigm:

1. **Procedural content**: how-to guidance (workflows, SOPs), 而非factual retrieval
2. **Task-class applicability**: 适用于一类问题, 不是single instance
3. **Structured components**: SKILL.md + 可选的scripts/templates/examples
4. **Portability**: 基于文件系统, 跨model/harness可移植

这个定义explicitly排除了一些容易混淆的概念:
- System prompts: 缺乏结构和resources
- Few-shot examples (Brown et al., 2020): declarative而非procedural
- RAG retrievals (Lewis et al., 2020): factual而非procedural
- Tool documentation (Schick et al., 2023; Qin et al., 2024): 描述capability而非procedure

Table 1展示了Augmentation paradigm的对比矩阵。直觉上, Skills = Prompts的可重用性 + Tools的executable资源 + RAG的modularity, 同时具备Procedural guidance, 而前三者都不具备完整procedural维度。

这种定义让我想起Sutton, Precup & Singh (1999)的options framework——temporal abstraction在RL中通过options (policy + termination condition + initiation set) 实现, Skills本质上是language agent的options, 把长horizon任务分解成可重用的procedural chunks。Sumers et al. (2023)的CoALA (Cognitive Architectures for Language Agents) 是这条线的理论基础。

参考链接:
- Options framework: https://www.sciencedirect.com/science/article/pii/S0004370299000521
- CoALA: https://arxiv.org/abs/2309.02427
- Voyager: https://arxiv.org/abs/2305.16291
- Toolformer: https://arxiv.org/abs/2302.04761

## 3. Benchmark设计: 容器化 + Deterministic Verification + Leakage Audit

### 3.1 Task Specification的四件套

每个task是一个self-contained module:
- **Instruction**: human-written task description (用GPTZero + human review双重验证非LLM生成)
- **Environment**: Docker container with `environment/skills/`子目录
- **Solution**: oracle solution, 必须100%通过所有tests
- **Verifier**: deterministic pytest assertions, no LLM-as-judge

这是遵循Zhu et al. (2025)和Anthropic (2026)的agentic benchmark best practices。Deterministic verification是关键——LLM-as-judge引入variance, 会污染Skills效果的measurement。

### 3.2 Leakage Prevention: 关键的methodological创新

这是paper最有趣的设计之一。Skill不能encode task-specific solutions:
- 不能包含task-specific filenames/paths/identifiers
- 不能包含exact command sequences that solve benchmark tasks
- 不能包含constants/magic numbers/values from task specs
- 不能引用specific test cases或expected outputs

用一个Claude Code Agent SDK-based validation agent在CI里做leakage audit, failed tasks直接reject。

这个设计很重要, 因为没有它, "Skills help"的结论可能只是"Skills泄露了答案"。Anthropic的Agent Skills specification (2025a)和MCP (2024)是背后的工业标准。

参考链接:
- Anthropic Agent Skills: https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- MCP: https://www.anthropic.com/news/model-context-protocol
- Best practices for agentic benchmarks: https://openreview.net/forum?id=E58HNCqoaA
- Demystifying evals for AI agents: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents

### 3.3 Dataset Construction: Community-driven + 多层review

- 105 contributors提交322 candidate tasks
- 86 tasks通过所有review (26.7% acceptance rate)
- 跨11个domains: Cybersecurity, Energy, Finance, Healthcare, Manufacturing, Mathematics, Media & Content, Natural Science, Office & White Collar, Robotics, Software Engineering
- Difficulty stratification: Core (19.8%, <60min), Extended (50.0%, 1-4h), Extreme (30.2%, >4h)

## 4. Experimental Setup: 7 × 3 × 84 = 7,308 trajectories

### 4.1 配置矩阵

7个agent-model configurations:
- Claude Code × {Opus 4.5, Opus 4.6, Sonnet 4.5, Haiku 4.5}
- Gemini CLI × {Gemini 3 Pro, Gemini 3 Flash}
- Codex × GPT-5.2

3个Skills conditions:
- No Skills (baseline)
- With Curated Skills
- Self-Generated Skills (只在Claude Code和Codex上eval, Gemini CLI的explicit skill-activation interface不支持)

每个task-model-condition组合跑5 trials (self-generated跑3), temperature=0 deterministic sampling。

### 4.2 Skill Injection机制

Skills通过文件系统注入:
```dockerfile
COPY skills /root/.claude/skills  # Claude Code
COPY skills /root/.codex/skills   # Codex CLI
COPY skills /root/.gemini/skills  # Gemini CLI
COPY skills /root/.agents/skills # Portable agents
```

不同的harness有不同的skill discovery机制:
- Claude Code和Codex: 读SKILL.md的YAML frontmatter (name + description)做relevance判断
- Gemini CLI: 暴露`activate_skill` tool, agent显式invoke

Instructions永远不reference which Skills to use——agent必须autonomously discover和apply。这是measure genuine skill utilization而不是instruction-following的关键。

### 4.3 Self-Generated Skills Condition

这个条件的设计很精妙。Prompt告诉agent: "先分析task需要什么domain knowledge, 写1-5个modular skill documents存到environment/skills/, 然后用这些skills作为reference解决task"。

这个condition isolate了LLM latent domain knowledge的效果。如果self-generated和curated效果相当, 说明Skills的value主要来自"以structured format组织知识"这个过程; 如果self-generated明显更差, 说明value来自human-curated domain expertise本身。

## 5. 关键Findings和技术解读

### 5.1 Finding 1: Skills提供substantial但variable的benefit

Table 3 (main results):

| Harness | Model | No Skills | With Skills | g (%) | Self-Gen | g_G (%) |
|---------|-------|-----------|-------------|-------|----------|--------|
| Gemini CLI | Gemini 3 Flash | 31.3 | 48.7 | 25.3 | — | — |
| Claude Code | Opus 4.5 | 22.0 | 45.3 | 29.9 | 21.6 | -0.5 |
| Codex | GPT-5.2 | 30.6 | 44.7 | 20.3 | 25.0 | -8.1 |
| Claude Code | Opus 4.6 | 30.6 | 44.5 | 20.0 | 32.0 | +2.0 |
| Gemini CLI | Gemini 3 Pro | 27.6 | 41.2 | 18.8 | — | — |
| Claude Code | Sonnet 4.5 | 17.3 | 31.8 | 17.5 | 15.2 | -2.5 |
| Claude Code | Haiku 4.5 | 11.0 | 27.7 | 18.8 | 11.0 | 0.0 |
| Mean | | 24.3 | 40.6 | 21.5 | 21.0 | -1.8 |

平均+16.2pp, range从+13.6pp到+23.3pp。Variance高说明Skills efficacy强依赖model-harness组合。Claude Code + Opus 4.5最高(+23.3pp), 因为Claude Code的native Skills integration是针对Anthropic的Agent Skills specification优化的——这是harness和model co-design的优势。

### 5.2 Finding 2: Gemini CLI + Gemini 3 Flash是performance champion

48.7% pass rate, 但normalized gain只有25.3%。这反映出Gemini 3 Flash的baseline已经很高(31.3%), Skills提供的absolute lift是+17.4pp。直觉: Flash的strong baseline意味着它在没有Skills时也能solve很多tasks, Skills主要补procedural gaps。

### 5.3 Finding 3: Self-Generated Skills是负面的!这是最重要的finding

平均-1.3pp, 只有Opus 4.6有+1.4pp的modest gain, Codex + GPT-5.2大幅退化(-5.6pp)。

这与curated Skills的+16.2pp形成sharp contrast。Trajectory分析揭示两种failure mode:
1. Model识别出需要domain knowledge, 但generate的procedure不精确或不完整 (e.g., "use pandas for data processing"而没有specific API patterns)
2. 对于high domain-knowledge tasks (manufacturing, financial), model甚至无法recognize需要specialized Skills, 用general-purpose approaches尝试

这个finding的深层含义: **LLM不能reliably author它自己能消费的procedural knowledge**。这听起来反直觉——如果model能理解一个Skill, 为什么不能generate它?

我的解读: 理解和生成procedural knowledge的asymmetry来自于:
- **Tacit knowledge问题**: 真正有用的procedural knowledge往往包含"看起来显然但容易遗漏"的细节, 这些细节human expert通过实践内化, 但model无法从pretraining distribution中提取
- **Self-knowledge gap**: Model不知道自己不知道什么。生成Skill时, 它generate的是"它认为需要的", 而不是"它实际需要的"
- **Distribution mismatch**: Pretraining见过的procedural text多是tutorial/documentation style, 缺乏task-specific的failure mode knowledge

### 5.4 Finding 4: Domain-level heterogeneity巨大

Table 4按domain分解:

| Domain | With Skills | No Skills | Δ |
|--------|-------------|-----------|---|
| Healthcare | 86.1% | 34.2% | +51.9 |
| Manufacturing | 42.9% | 1.0% | +41.9 |
| Cybersecurity | 44.0% | 20.8% | +23.2 |
| Natural Science | 44.9% | 23.1% | +21.9 |
| Energy | 47.5% | 29.5% | +17.9 |
| Office & White Collar | 42.5% | 24.7% | +17.8 |
| Finance | 27.6% | 12.5% | +15.1 |
| Media & Content | 37.6% | 23.8% | +13.9 |
| Robotics | 27.0% | 20.0% | +7.0 |
| Mathematics | 47.3% | 41.3% | +6.0 |
| Software Engineering | 38.9% | 34.4% | +4.5 |

直觉: pretraining coverage强的domain (Software Engineering, Mathematics) Skills效果小, 因为model已经有strong priors; pretraining coverage弱但procedural knowledge重要的domain (Healthcare, Manufacturing) Skills效果巨大。

Healthcare的+51.9pp说明clinical data harmonization这类task的procedural knowledge严重underrepresented in pretraining。Manufacturing的+41.9pp更惊人——baseline只有1.0%, 几乎是"无Skill不可解"的task class。

### 5.5 Finding 5: 2-3 Skills是optimal, 更多diminishing returns

Table 5:

| Skills Count | With Skills | No Skills | Δ |
|--------------|-------------|-----------|---|
| 1 skill | 42.2% | 24.4% | +17.8 |
| 2-3 skills | 42.0% | 23.4% | +18.6 |
| 4+ skills | 32.7% | 26.9% | +5.9 |

Non-monotonic关系。4+ skills反而只有+5.9pp。直觉: excessive Skills content creates cognitive overhead或conflicting guidance。这让我想到prompt engineering里的"less is more"现象——context budget有限时, 精炼的guidance比海量documentation更effective。

### 5.6 Finding 6: Detailed和compact Skills优于comprehensive

Table 6:

| Complexity | Pass Rate | Δ | N |
|------------|-----------|---|---|
| Detailed | 42.7% | +18.8 | 1165 |
| Compact | 37.6% | +17.1 | 845 |
| Standard | 37.1% | +10.1 | 773 |
| Comprehensive | 39.9% | -2.9 | 140 |

Comprehensive Skills actually hurt performance (-2.9pp)。这是一个很强的design principle: **focused procedural guidance > exhaustive documentation**。

直觉: agent从lengthy Skills中提取relevant information的认知负担高, 而且elaborate Skills消耗context budget却不提供actionable guidance。

### 5.7 Finding 7: Smaller model + Skills > Larger model without Skills

Claude Haiku 4.5 with Skills (27.7%) > Haiku without Skills (11.0%), 提升+16.7pp。
Claude Opus 4.5 without Skills = 22.0%。

所以Haiku + Skills (27.7%) > Opus 4.5 without Skills (22.0%)!

这是一个economically significant的finding: **Skills可以partially substitute for model scale**。对于procedural tasks, 小模型+好的Skills在成本上远胜大模型。

## 6. Normalized Gain公式深度解析

$$g = \frac{pass_{\text{skill}} - pass_{\text{vanilla}}}{1 - pass_{\text{vanilla}}}$$

变量:
- $g$: normalized gain, $\in [0, 1]$ (理论上可达$>1$如果Skills引入负面影响时为负)
- $pass_{\text{skill}} \in [0, 1]$: with Skills条件下的pass rate
- $pass_{\text{vanilla}} \in [0, 1]$: no Skills条件下的baseline pass rate
- $1 - pass_{\text{vanilla}}$: 从baseline到perfect (100%)的"headroom", 即还有多少改进空间

这个公式源自Hake (1998), 美国物理教育研究里的经典metric。原始context: 比较传统讲授vs interactive engagement在物理课上的效果, 用normalized gain避免ceiling effect的confounding。

直觉: 如果一个model的baseline是90%, 提升到95%, $g = \frac{0.95 - 0.90}{1 - 0.90} = 0.5$; 另一个model的baseline是10%, 提升到55%, $g = \frac{0.55 - 0.10}{1 - 0.10} = 0.5$。两者$g$相同, 但含义完全不同——前者是ceiling effect, 后者是genuine scaffolding。

Paper明确说: "High $g$ with low $\Delta_{\text{abs}}$ suggests ceiling effects; high $g$ with high $\Delta$ suggests substantial scaffolding." 这是正确的解读方式。

应用: Table 3里Claude Code + Opus 4.5: $\Delta = +23.3\text{pp}$, $g = 29.9\%$, $pass_{\text{vanilla}} = 22.0\%$, $pass_{\text{skill}} = 45.3\%$。验证: $\frac{45.3 - 22.0}{100 - 22.0} = \frac{23.3}{78.0} = 0.2987 \approx 29.9\%$。✓

为什么这个metric重要: 当不同model的baseline差异大时, 直接比较$\Delta_{\text{abs}}$会产生misleading结论。比如Haiku baseline=11%, Opus baseline=22%。如果两者都提升+15pp, 对Haiku来说是从11%到26% (相对提升136%), 对Opus来说是从22%到37% (相对提升68%)。Normalized gain把这俩normalize到同一scale。

Hake's paper: https://doi.org/10.1119/1.18809

## 7. 失败分析: MAST/TAT Taxonomy

Paper用programmatic analysis (而非LLM-as-judge)分类5,171个agent failures, 5个大类:

| Category | Failure Mode | Count | % |
|----------|--------------|-------|---|
| Execution | No Output Produced | 411 | 7.9 |
| Execution | Domain Knowledge Gap | 251 | 4.9 |
| Execution | Specification Violation | 171 | 3.3 |
| Execution | Incorrect Implementation | 68 | 1.3 |
| Execution | Tool/Environment Failure | 16 | 0.3 |
| Coherence | Incomplete Solution | 527 | 10.2 |
| Verification | Quality Below Threshold | 2,577 | 49.8 |
| Timeout | Agent Timeout | 922 | 17.8 |
| Unknown | Unclassified | 228 | 4.4 |

**Quality Below Threshold占49.8%** 是最重要的数据点。这说明agents通常understand task structure并produce output, 但output的accuracy不够。这对应Terminal-Bench TAT里的"Weak Verification"——agent的self-checks无法catch质量问题。

这是Skills最能help的地方: Skills encode "verifier-facing details (steps, constraints, sanity checks)", 直接address quality issues。

Table 17比较No Skills vs With Skills的failure分布:

| Condition | Fail Rate | Timeout | Execution | Coherence | Verification |
|-----------|-----------|---------|-----------|-----------|--------------|
| No Skills | 78.4% | 16.1% | 17.1% | 10.7% | 52.1% |
| With Skills | 61.1% | 18.6% | 21.1% | 8.9% | 46.6% |
| Self-Generated | 80.9% | 19.9% | 13.9% | 11.2% | 50.4% |

关键观察:
- Skills主要reduce Verification failures (绝对数从1,184降到819, -30.8%)
- Skills稍微增加Timeout的相对share (虽然绝对数从367降到328)——因为Skills让agents pursue更ambitious strategies, 有时超时
- Skills reduce Coherence failures 35.8% (从243到156)——Skills帮agents identify所有required deliverables
- Self-Generated的fail rate (80.9%)比No Skills (78.4%)还高!

参考链接:
- MAST (Multi-Agent System Taxonomy): https://arxiv.org/abs/2502.02692 (估计)
- Terminal Agent Taxonomy: https://arxiv.org/abs/2601.11868

## 8. 成本效率: Pareto Frontier和Cache Efficiency

### 8.1 Token Usage

Table 12:

| Model | Condition | Input (K) | Output (K) | Total (K) |
|-------|-----------|-----------|------------|-----------|
| GPT-5.2 | No Skills | 961 | 12.2 | 974 |
| GPT-5.2 | With Skills | 1,087 | 11.6 | 1,099 |
| Gemini 3 Flash | No Skills | 985 | 14.2 | 999 |
| Gemini 3 Flash | With Skills | 1,075 | 12.1 | 1,087 |
| Gemini 3 Pro | No Skills | 495 | 12.0 | 507 |
| Gemini 3 Pro | With Skills | 465 | 10.9 | 476 |

有意思的观察:
- **Gemini 3 Flash consumes 2.3× more input tokens than Gemini 3 Pro** (1.08M vs 0.47M with Skills)。Paper称之为"compensatory strategy where the smaller model substitutes iterative exploration for reasoning depth"。这是smaller model用iteration补偿reasoning capacity的经典pattern。
- Gemini 3 Pro with Skills的token usage反而下降6%——Skills让Pro更efficient, 减少exploration rounds

### 8.2 Cost-Performance Tradeoff

Table 13 estimated cost per trial:
- Gemini 3 Flash: $0.54 (No Skills) → $0.57 (With Skills), +6%
- GPT-5.2: $1.85 → $2.07, +12%
- Gemini 3 Pro: $1.13 → $1.06, -6%

在standard API pricing下, Flash比Pro便宜47% ($0.57 vs $1.06), 因为Flash的per-token cost低4× ($0.50 vs $2.00 per 1M input), 这more than offset它的2.3×更高token volume。

### 8.3 Cache Efficiency

GPT-5.2: 91-92% cache hit rate
Gemini 3 Pro: 75-76%
Gemini 3 Flash: 63-67%
Claude Code: >99% (aggressive prompt caching)

实际deployed cost会比Table 13的standard rate低50-90%。

## 9. 生态分析: 47,150个unique Skills

### 9.1 数据来源

- Public GitHub repos tagged "claude-skills"或"agent-skills": 12,847
- Community marketplaces (Smithery.ai, skillmp.com): 28,412
- Corporate partner contributions: 5,891
- Dedup后: 47,150 unique Skills from 6,323 repos

时间序列(Figure 5): 136天内累计84,192 Skills, 2026年1月达到daily peak 18,904 additions, 呈exponential-like增长。

### 9.2 质量分布

- Size: log-normal distribution, median 2.3 KB (IQR 0.8-6.1 KB), top 1% >50 KB
- Token count: median ~1.5k tokens
- Domain coverage: Software Development 38%, Data Analysis 22%, DevOps 15%, Writing/Docs 12%, Other 13%
- Structure: 78% standard SKILL.md + optional resources
- File count: median 1 file, 大部分<5 files
- File type: markdown主导 (documentation-heavy ecosystem)

### 9.3 质量评分

4个维度, 每个0-3分, total /12:
1. Completeness
2. Clarity
3. Specificity
4. Examples

Ecosystem mean: 6.2/12 (SD=2.8)。SkillsBench选top quality quartile (score ≥ 9/12, benchmark mean 10.1/12)。

**关键limitation**: 这是optimistic scenario。Real-world deployment涉及更低质量的Skills和imperfect task-skill matching。

## 10. 方法论贡献和设计哲学

### 10.1 Paired Evaluation as Standard Practice

这是paper的core methodological contribution。传统benchmark假设agent是fixed system, measure它的capability。SkillsBench认为augmentation是variable of interest, 必须paired evaluation (with vs without augmentation在同一task上)。

这个insight可以推广到任何augmentation strategy: RAG, tool use, chain-of-thought, reflection——都应该用paired evaluation来measure真实效果。

### 10.2 Determinism和Reproducibility

- Temperature 0
- Docker containerization + clean state per trial
- Deterministic pytest verifiers (no LLM-as-judge)
- 多trials averaging (5 runs main, 10 runs validation subset)
- Leakage audit (Claude Code Agent SDK-based validation agent in CI)

这是agentic benchmarking的gold standard, 参考Zhu et al. (2025) best practices paper。

### 10.3 排除Shortcut Solutions

Tasks必须prevent:
- Editing input data
- Extracting answers from test files
- Exploiting verifier implementation

Reviewer在human review阶段run benchmark with和without Skills across multiple agents来confirm每个task提供meaningful signal。

## 11. Limitations和Future Work

Paper自承的limitations:

1. **Coverage**: 只评terminal-based containerized tasks, 不能直接transfer到GUI agents, multi-agent, very long-horizon workflows。Multi-modal skills和vision-language agents是natural extension。

2. **Causal attribution**: Skills injection增加context length, observed gains可能partly是"more context"而非procedural structure。Self-generated condition是部分control, 但需要更强length-matched baselines (random/irrelevant text, retrieval-only documentation controls)。

3. **Determinism和contamination**: Containerization提供state isolation但non-perfectdeterminism, 也无法完全消除training-set leakage。Multiple runs + leakage audit + paired comparison mitigate但not eliminate。

4. **Ecosystem gap**: Benchmark Skills质量高 (mean 10.1/12) vs ecosystem mean (6.2/12), 是optimistic scenario。Future work应eval ecosystem-representative settings。

## 12. 我的批评性思考

### 12.1 关于Self-Generated Skills的interpretation

Paper的结论是"models cannot reliably author the procedural knowledge they benefit from consuming"。这个结论正确, 但我想加一层nuance:

Model生成Skill时, 它generate的是"它自己会做什么", 而不是"什么是最好的procedure"。如果model不知道某个API的某个gotcha, 它生成的Skill里也不会包含那个gotcha——因为它从来没见过。这是information-theoretic的limit, 不是capability的limit。

一个有意思的future direction: 如果让strong model生成Skill给weak model用, 效果如何? 这test了"authoring capability"和"consumption capability"的分离。

### 12.2 关于2-3 Skills最优的解释

Paper说"excessive Skills content creates cognitive overhead or conflicting guidance"。我想补充一个更mechanistic的解释:

LLM的attention mechanism在context中检索信息时, signal-to-noise ratio很重要。2-3个Skills是高SNR的; 4+个Skills引入了需要agent判断哪个relevant的负担, 可能引入irrelevant content稀释attention。

这与RAG里的"chunk count vs precision"tradeoff类似。Optimal Skills count应该是task-specific的function。

### 12.3 关于Comprehensive Skills的negative effect

Comprehensive Skills -2.9pp是反直觉但重要的。我怀疑原因是: comprehensive documentation倾向于提供multiple approaches, 而model需要commit to one approach执行。Multiple approaches在context里造成decision paralysis, 或agent在执行中混合不同approaches导致incoherent solution。

这与agent design里的"single clear instruction vs comprehensive manual"tension一致。

### 12.4 Missing analysis: Skill-Skill interaction

Paper提到"composite performance can be predicted from atomic Skills effects"是future work, 但没深入分析。如果有2个Skills, 它们的combined effect是additive, sub-additive, 还是super-additive? 这对Skills curation很重要。

从Table 5的non-monotonic trend推测, 多Skills更可能是sub-additive (interference)而非super-additive (synergy)。

### 12.5 关于"Skills close procedural gaps"的深层含义

Paper说Skills最helpful当success depends on "concrete procedures and verifier-facing details (steps, constraints, sanity checks)"。这暗示LLM agents的weakness不是conceptual knowledge, 而是**execution-time procedural reliability**。

这是一个重要的research direction: LLM知道concept, 但在长execution trajectory中无法可靠地执行正确的procedure。Skills本质上是externalize procedural memory, 减轻working memory负担。

这与cognitive science里的"procedural vs declarative memory"区分一致。LLM pretraining主要acquire declarative, Skills提供procedural。

### 12.6 关于harness的confound

Paper发现Claude Code + Opus 4.5的improvement最大 (+23.3pp), 部分原因是Claude Code native Skills integration optimized for Agent Skills spec。这是一个co-design advantage, 但也意味着Skills efficacy强依赖harness implementation。

Codex CLI "frequently neglects provided Skills—agents acknowledge Skills content but often implement solutions independently"——这是harness-level的Skills utilization failure, 不是model capability failure。

Future work应该decouple harness effect和model effect, 例如用portable harness + 不同model backend。

## 13. 启示和Future Direction

### 13.1 对Skills authoring的practical guidance

基于findings:
1. **Concise > exhaustive**: Detailed或compact Skills优于comprehensive
2. **Stepwise guidance + at least one working example**最effective
3. **2-3 modules**是sweet spot
4. **Explicitly match harness constraints** (e.g., JSON-only protocols需要repeated format reminders)
5. **Modular Skills compose better** on multi-part tasks

### 13.2 对model training的implications

Self-generated Skills无效说明pretraining没有给model足够procedural knowledge来author Skills。如果future model要在self-improvement上breakthrough, 需要在pretraining中explicitly include procedural knowledge structure——不只是"how to do X"的tutorial text, 而是元级别的"how to author a procedure for X"。

### 13.3 对benchmarking的implications

Paired evaluation应该成为augmentation research的standard。任何新augmentation method (RAG variants, tool use strategies, reflection protocols)都应该在paired setup下eval, 而不是absolute performance。

### 13.4 经济意义

Haiku + Skills > Opus without Skills这个finding有巨大经济意义。如果deploy成本是concern, 投资Skill curation可能比投资larger model更cost-effective。这改变cloud AI的economics。

## 14. 相关工作和延伸阅读

### 14.1 Agent Benchmarks
- Terminal-Bench: https://arxiv.org/abs/2601.11868
- SWE-bench: https://openreview.net/forum?id=VTF8yNQM66
- SWE-agent: https://arxiv.org/abs/2405.15793
- SWE-Smith: https://arxiv.org/abs/2504.21798
- AgentBench: https://arxiv.org/abs/2308.03688
- WebArena: https://openreview.net/forum?id=oKn9c6ytLx
- VisualWebArena: https://aclanthology.org/2024.acl-long.249/
- OSWorld: https://arxiv.org/abs/2404.07972
- AppWorld: https://aclanthology.org/2024.acl-long.850/
- MLE-bench: https://openreview.net/forum?id=6s5uXNWGIh
- Cybench: https://arxiv.org/abs/2408.08926
- ReplicationBench: https://arxiv.org/abs/2510.24591
- τ-bench: https://openreview.net/forum?id=roNSXZpUDN

### 14.2 Procedural Augmentation
- Voyager (Wang et al., 2023): https://arxiv.org/abs/2305.16291
- CoALA (Sumers et al., 2023): https://arxiv.org/abs/2309.02427
- ReAct (Yao et al., 2022): https://arxiv.org/abs/2210.03629
- Reflexion (Shinn et al., 2023): https://arxiv.org/abs/2303.11366
- Tree of Thoughts (Yao et al., 2023): https://arxiv.org/abs/2305.10601
- Self-Refine (Madaan et al., 2023): https://arxiv.org/abs/2303.17651
- Chain-of-Thought (Wei et al., 2022): https://arxiv.org/abs/2201.11903
- Toolformer (Schick et al., 2023): https://arxiv.org/abs/2302.04761
- ToolLLM (Qin et al., 2024): https://openreview.net/forum?id=dHng2V3yof
- DSPy (Khattab et al., 2023): https://arxiv.org/abs/2310.03714

### 14.3 Agent Skills工业标准
- Anthropic Agent Skills: https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- Model Context Protocol: https://www.anthropic.com/news/model-context-protocol
- Claude Code: https://github.com/anthropics/claude-code
- Gemini CLI: https://github.com/google-gemini/gemini-cli
- Codex CLI: https://github.com/openai/codex

### 14.4 RL/认知架构基础
- Options framework (Sutton, Precup, Singh, 1999): https://www.sciencedirect.com/science/article/pii/S0004370299000521
- Hake 1998 (normalized gain): https://doi.org/10.1119/1.18809

### 14.5 Benchmarking best practices
- Zhu et al. 2025 (best practices for agentic benchmarks): https://openreview.net/forum?id=E58HNCqoaA
- Anthropic 2026 (demystifying evals for AI agents): https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- Harbor framework: https://github.com/laude-institute/harbor

## 15. 总结

SkillsBench是第一篇systematically evaluate Agent Skills efficacy的paper, 通过7,308 trajectories的paired evaluation, 建立了以下empirical facts:

1. **Curated Skills提供+16.2pp平均提升**, 但variance大 (+4.5到+51.9pp跨domain)
2. **Self-Generated Skills无效 (-1.3pp)**——models不能可靠author它们consume的procedural knowledge
3. **Less is more**: 2-3 modules + detailed/compact > 4+ modules或comprehensive
4. **Skills可以partially substitute for model scale**: Haiku + Skills > Opus without Skills
5. **Quality Below Threshold是dominant failure mode (49.8%)**, Skills主要reduce这类failures

Methodologically, paper确立了**paired evaluation as standard practice for augmentation research**, deterministic verification, leakage audit, harness-aware evaluation。

对我而言, 最deep的insight是: **LLM agents的瓶颈不是conceptual knowledge, 而是execution-time procedural reliability**。Skills本质上是externalized procedural memory, 补足LLM pretraining里underrepresented的procedural knowledge distribution。

这也指向一个deeper research question: 如果LLM能被trained来更好地author Skills (元procedural learning), 是否能实现某种形式的self-improvement? 目前答案是no, 但这个boundary本身是值得研究的object。

Future work的obvious方向:
- Multi-modal skills for vision-language agents in GUI environments
- Skill-Skill interaction和composition的predictive theory
- Length-matched baselines to isolate procedural structure effect
- Ecosystem-representative (lower quality) Skills evaluation
- Strong model authors Skills for weak model (decoupling authoring和consumption)

这是一个well-designed benchmark paper, 它的value不仅在于findings, 更在于它确立的methodology——把augmentation strategy作为first-class evaluation artifact, 用paired evaluation measure真实efficacy。这个methodology可以推广到任何augmentation research, 是值得借鉴的paradigm。
