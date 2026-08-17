---
source_pdf: CL-BENCH.pdf
paper_sha256: 467e03c67c97e1f8ba7a330d5ff0f90d1a8ad63a51f03e65a051cb36bf020881
processed_at: '2026-08-03T15:43:40-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CL-BENCH 用人话讲

## 一句话总结

**现在的LLM基本是个"开卷考试都考不及格"的学生。** 你把答案（context）完整塞给它，它还是做不对题。这paper就是来prove这件事的。

---

## 1. 这paper到底在说什么问题

你训练一个LM，花了天文数字的compute，把整个internet的知识都compress进weights里了。然后你test它做math、写code、考医生执照——都还行。

但real world不是这样的。Real world是这样的：

- 你是一个律师，今天接了个案子，涉及一个**你从未见过**的legal framework
- 你是个工程师，公司发了份**昨天才写完**的API文档让你集成
- 你是个分析师，拿到一份**实验数据**让你找出underlying law

这些场景的共同点：**答案不在你的脑子里，在桌上那堆纸里**。你得先读懂纸上的东西，然后用你固有的reasoning能力去solve问题。

LM目前就卡在这里。paper把这叫 **context learning**。

---

## 2. 为什么这不等于现有的东西

### vs. In-Context Learning (ICL)

ICL就是你给model几个example，它学会"哦，这种格式的输入对应这种输出"。本质是pattern matching，靠induction heads做copy-and-match。

Context learning完全不一样。你要让它读一份60页的fictional国家法律文档，然后判一个具体case。这不是pattern matching，这是真的要**理解**+**应用**。

用公式说：
$$\text{Solve}(T, C) = f_{\text{reason}}\big(g_{\text{learn}}(C), T\big)$$

ICL的 $g_{\text{learn}}$ 是shallow的——基本是"看example照葫芦画瓢"。Context learning的 $g_{\text{learn}}$ 要求把一份复杂文档**真的变成workable knowledge representation**。

### vs. Long-Context Benchmarks

LongBench、RULER这些测的是"60页文档里某句话说了啥"——retrieval问题。CL-bench测的是"读完60页文档后你能用它做什么"——application问题。

差别就像：reading comprehension vs. open-book exam。

### vs. RAG

RAG community天天研究"怎么retrieve relevant chunks"、"怎么chunk"、"怎么rerank"。但大家assume了一个premise：**model拿到context后能学好**。

这paper说：等一下，这个premise可能根本不成立。你retrieve再准，model学不会也白搭。

Anthropic自己最近也写了篇 [context engineering blog](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)，主要讲怎么组织context。但CL-bench这paper问的是更底层的问题：model本身能不能从你给它的context里learn。

---

## 3. Benchmark怎么构造的

### 规模

500个contexts，1899个tasks，31607个rubrics。平均一个context配3.8个task，每个task配16.6个rubric。**专家平均花20小时**标一个context。这是非常重的人工。

### 怎么保证"必须从context学"

这是paper最聪明的设计。三种方式让pre-trained knowledge完全用不上：

**(1) 全虚构**
比如造一个fictional国家的完整legal system，pre-training里绝对没有。

**(2) 改变真实内容**
比如修改historical events，改scientific definitions。

**(3) 用niche long-tail内容**
新发布的产品手册、超窄domain的专业知识。

**验证手段**：把context全删掉，GPT-5.1 solve rate暴跌到**0.9%**。这数字基本说明：没有context，这paper的tasks根本做不了。Pre-trained knowledge here is useless.

### 四大类Context

1. **Domain Knowledge Reasoning** (190 contexts): 专业领域知识应用
2. **Rule System Application** (140 contexts): 游戏规则、编程语言语法、法规这种formal system
3. **Procedural Task Execution** (100 contexts): 操作手册、workflow、troubleshooting
4. **Empirical Discovery & Simulation** (70 contexts): 从实验数据归纳law，或simulate sandbox

注意第4类最难——前三类是deductive（给你规则你应用），第4类是**inductive**（给你数据你发现规律）。人类做inductive reasoning都费劲，model更难。

---

## 4. 评估方法

每个task配16.6个**binary rubric**（yes/no问题）。比如：

> "The response should provide the documented production budget for Star Wars as \$447 million (net) or \$533 million (gross) as stated in Source 1."

**Strict标准**：必须**全部**rubric pass才算solved。任何一个fail就0分。

用GPT-5.1做verifier，prompt让它分3步走：
1. 分析rubric的所有explicit和implicit requirements
2. 逐条check student solution
3. Self-reflection（completeness、strictness、consistency、objectivity）

验证reliability：
- 换Claude Opus 4.5和Qwen-3-Max当verifier，agreement >90%
- 人工check 100个sample，accuracy >90%

这种rubric-based eval现在越来越流行，[EvaLearn](https://openreview.net/forum?id=rRHuBZdDfY)和[MultiChallenge](https://aclanthology.org/2025.findings-acl.1119/)都用了类似方法。

---

## 5. 实验结果：基本全跪

### 总表

| Model | Overall | Domain | Rule | Procedural | Empirical |
|-------|---------|--------|------|------------|------------|
| GPT-5.1 (High) | **23.7%** | 25.3% | 23.7% | 23.8% | 18.1% |
| Claude Opus 4.5 Thinking | 21.1% | 23.7% | 19.0% | 22.6% | 15.1% |
| GPT-5.2 (High) | 18.1% | 18.6% | 17.2% | 21.4% | 11.7% |
| o3 (High) | 17.8% | 18.0% | 17.6% | 19.5% | 13.7% |
| Kimi K2 Thinking | 17.6% | 18.7% | 17.0% | 18.8% | 12.6% |
| HY 2.0 Thinking | 17.2% | 18.0% | 17.3% | 19.4% | 8.9% |
| Gemini 3 Pro (High) | 15.8% | 15.5% | 17.7% | 16.4% | 10.1% |
| Qwen 3 Max Thinking | 14.1% | 13.5% | 15.6% | 15.2% | 9.0% |
| Doubao 1.6 Thinking | 13.4% | 13.7% | 14.2% | 13.9% | 9.4% |
| DeepSeek V3.2 Thinking | 13.2% | 13.6% | 13.8% | 14.2% | 8.0% |

**平均17.2%，最好的GPT-5.1也就23.7%。** 没有任何model超过30%。

几个有意思的观察：

### (1) GPT-5.2 居然比 GPT-5.1 差5.6%

新版不如老版，这反intuitive。Analysis发现GPT-5.2有两个systematic failure mode：
- 长context中维持coherent causal chain能力变弱
- 频繁violate context中explicit写的constraint

特别是experimental data子类：GPT-5.1 31.1% vs GPT-5.2 22.2%。

**这暗示一件事**：光靠scale up pre-training，不focus context learning，可能反而会degrade这能力。Model变"聪明"了但变"不会学"了。

### (2) Inductive比Deductive难一大截

Empirical Discovery & Simulation平均只有~11%，比其他类低6%。这很合理：

- Deductive：规则给你了，你apply
- Inductive：数据给你了，你**发现**规则

前者是execution，后者是science。人类科学家做inductive都费劲，model做不好意料之中。

而且这类的variance也大——Kimi K2在Empirical是12.6±4.0，std是其他类的几倍。说明model在inductive任务上行为不稳定。

### (3) 错误类型分布

| Error Type | 占比 |
|------------|------|
| Context Misused | >60% (所有model) |
| Context Ignored | 与overall performance负相关 |
| Format Errors | >35% GPT-5.1, >40% Claude |
| Refusals | 少数 |

**关键insight**：strong model的"Context Ignored"率低（更会注意context），但**Context Misused率所有model都高**。即使最强的Claude Opus 4.5也struggle to correctly interpret and apply context。

这说明问题不在"看不看"，在"看懂不懂+会不会用"。

### (4) Reasoning effort帮助有限

GPT-5.1从low reasoning到high reasoning：21.2% → 23.7%，只涨2.5%。

对比math/code reasoning上test-time compute scaling的显著gain，这数字非常modest。

**Hypothesis**：context learning的bottleneck不在reasoning depth，在context representation的quality。光让model多think几遍没用，因为它从一开始就没把context学进去。

参考 [Let's Verify Step by Step](https://openreview.net/forum?id=v8L0pN6EOi) 那种reasoning scaling在math上很work，在context learning上不work。

### (5) Context越长越跪

所有model随context length单调下降：
- 0-15K tokens: ~25-35%
- 120K+ tokens: ~5-10%

Claude Opus 4.5跌幅最大，超过20%。

但reasoning effort的优势在长context下更明显。GPT-5.1在32K+还能维持16.2%，远超其他model。

这呼应[Anthropic context engineering blog](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)的观察：long context处理仍然是bottleneck。

### (6) 同domain不同knowledge type差异巨大

同样是legal domain：
- **Legal & Regulatory** (Rule System类，结构化条文): >29% for all models
- **Legal Advisory** (Domain Knowledge类，复杂scenario判断): 显著更低

Qwen 3 Max在这两个上gap超过25%。

**Insight**：domain本身不重要，**knowledge怎么structured + task怎么要求你apply** 才重要。结构化reference manual好学，需要professional judgment的复杂scenario难学。

---

## 6. 几个Case Study的直觉

### Case 1: 无人机SDK (Table 9)

User要紧急送Hazmat到Sector 4用drone D-998，还明确要求用`force_launch_override()`绕过安全检查。

Context里SDK文档说有`Safety_request_airspace()`这个mandatory function。

Gemini-3-Pro的行为：
- ✓ 识别出`force_launch_override()`不存在
- ✗ 没生成完整workflow
- ✗ 提到了ERR-1002但没用对应的`Safety_request_airspace()`
- ✗ 没把task parameters (D-998, Sector 4)绑进去

**直觉**：model像个学生，能指出题目里哪不对，但写不出对的答案。它"读"了文档但没"学进去"。

### Case 2: 法学院选课 (Table 10)

2L学生选课，规则：
- 每学期13-18 credits
- 每年29-32 credits
- 这学生上学期修了15

GPT-5.2生成了min schedule (14 credits) ✓，但max schedule给了18 credits ✗。

实际应该推：年max 32 - 已修15 = 这学期max 17。

**直觉**：model懂每条rule，但**不会组合多个numerical constraint做推理**。这是典型的"知道每条公式但不会做综合题"。

### Case 3: Wingspan游戏simulate (Table 11)

初始：3 birds in hand (每个cost 1 worm + 1 egg)，3 worms，0 eggs，3 birds已在grassland。

GPT-5.1策略正确：4 turns (1 Lay Eggs + 3 Play a Bird)。逻辑全对。

但fail了format rubric：没exact restate starting state、长度不对、没end-of-turn resource accounting。

**直觉**：reasoning对了，但因为没按格式交卷被扣分。这暴露strict eval的双刃剑——也可能underestimate真实deployment capability。

### Case 4: EZLang编程 (Table 12)

要求：每30分钟check time，5:30pm停，最后print所有记录。

Gemini-3-Pro生成了runnable code，但：
- 把30分钟改成了1秒（"方便测试"）
- 把time-based stop改成了固定5次iteration

**直觉**：model偏好developer convenience over spec fidelity。它懂你要什么，但擅自"优化"了spec。这种"自作主张"在production agent里很危险。

---

## 7. 我的几个延伸思考

### (1) Context Learning是RAG的隐含premise

整个RAG community都在研究retrieval precision/recall。但隐含assume了model能学好retrieved content。这paper说这个premise可能不成立。

如果model的context learning capability弱，你RAG再fancy也没用——retrieve到了model也学不会。

Production RAG system的evaluation应该加一个dimension：**model对retrieved context的learning efficacy**，而不仅是retrieval quality。

参考 [RAG survey](https://arxiv.org/abs/2312.10997) 和 [Modular RAG](https://arxiv.org/abs/2407.21059)。

### (2) Test-Time Compute Scaling在这个domain失效

Reasoning effort从low到high只涨2.5%。对比math/code上的显著gain，这暗示：

**Context learning的瓶颈不在reasoning depth，在context representation的quality。**

光让model多think几遍没用，因为think之前它就没把context encode好。这跟[OpenAI的o1/o3路线](https://arxiv.org/abs/2601.03267)在reasoning task上的成功形成对比——纯reasoning scaling不够，需要architecture层面的改进。

### (3) Attention Mechanism可能不是最优解

Paper的Discussion提到：current transformer通过attention处理context，可能not optimally suited for deep context learning。

可能的architecture方向：
- Explicit memory structures ([Memorizing Transformers](https://openreview.net/forum?id=TrjbxzRcnf-))
- Iterative refinement over multiple passes
- Dedicated pathways for不同types of contextual info ([Nested Learning](https://openreview.net/forum?id=nbMeRvNb7A))

我个人hypothesis：context learning可能需要某种**iterative processing**——model先快速scan整个context，再多次refine对context的understanding。一次forward pass可能不够。

### (4) Context Learning vs. Fine-tuning

这paper暗示了一个paradigm shift：**未来LM的"adaptation"可能主要靠context engineering而非fine-tuning**。

Fine-tuning：expensive, risks catastrophic forgetting, 需要数据
Context engineering：immediate, no parameter mod, 只需要good context

前提是model context learning够强。现在23.7%显然不够。但如果这能力能scale上去，整个domain adaptation的方式会变。

### (5) Inductive vs. Deductive的gap有深意

前三类deductive任务平均比第四类inductive高6%。这不只是"难一点"。

Deductive task：model只需要"execute" given rules
Inductive task：model需要"discover" rules from data

后者本质上要求**scientific reasoning**——观察、假设、验证。这是人类intelligence的核心，也是AGI的关键capability。

Frontier model在这上面只有~11%，说明我们离AGI还远得很。

### (6) GPT-5.2 < GPT-5.1 这件事

这件事很值得琢磨。新版model在某些dimension变强了（可能pure reasoning），但context learning变弱了。

可能的explanation：
- Pre-training data更多更广，但model对"陌生knowledge"的attention反而钝化了
- RLHF/RLAIF让model更"自信"，但自信导致ignore context
- Architecture变化（如MoE expert增加）可能hurt cross-context信息整合

这是个**重要warning**：context learning不会随着scale自动变好，可能需要专门的training signal。

### (7) 评估Strictness的trade-off

Strict criterion（必须全rubric pass）有advantage也有problem。

Table 11的Wingspan case，GPT-5.1 reasoning全对但format违规导致0分。在production agent里，这种"reasoning对+format瑕疵"可能完全acceptable。

我建议future work加partial credit scoring作为supplementary metric。也许：strict solve rate + soft score (rubric pass rate) 两个metric一起看。

### (8) Sequential Tasks的subtle问题

51.1%的tasks是sequential——后一个依赖前一个的标准答案。这有issue：

如果model在第1个task fail了，第2、3个task基于错误prior自然也fail。这会**放大**failure rate。

也许应该provide prior standard solutions给model作为context，isolate每个task的evaluation。或者单独report sequential vs. independent tasks的performance。

### (9) 缺Human Baseline

Paper承认没有human baseline。这是big gap。

如果domain expert的solve rate也只有50%，那model 23.7%其实没那么差。如果expert能95%+，那gap很大。

但human baseline难做——expert已知答案（biased），non-expert没foundation。

参考[GPQA](https://openreview.net/forum?id=Ti67584b98)的做法：PhD-level问题 + PhD-level human eval。CL-bench可能需要类似的design。

### (10) Multimodal是必然方向

Paper只测text。但real-world context是多模态的：
- 修机器要看schematic diagram + 听声音 + 读manual
- 医学case要看imaging + 读病史
- 法律case可能涉及图表证据

Context learning在multimodal setting下可能**更难**——cross-modal alignment是额外challenge。这也是clear的next step。

---

## 8. 给研究者的Take-aways

### 如果你是LM trainer
- 你的training data里有没有"必须从context学"的samples？
- 你的model有没有incentive去attend to unfamiliar context？
- 你的eval suite有没有测context learning（不是long-context retrieval）？

### 如果你是RAG practitioner
- 你的evaluation有测"model对retrieved content的learning efficacy"吗？
- 你的RAG pipeline再fancy，model学不会也白搭

### 如果你是agent builder
- 你的agent能不能从一份新文档rapidly specialize？
- Long-context + new knowledge = 你agent的weakness
- 评估时加context learning dimension

### 如果你是benchmark builder
- Rubric-based eval是好方向
- 但strict criterion有trade-off，考虑partial credit
- 加human baseline，否则不知道ceiling在哪

### 如果你是architecture researcher
- Attention可能不是deep context learning的最优解
- Explicit memory、iterative refinement、dedicated pathways都值得explore
- 看下[Memorizing Transformers](https://openreview.net/forum?id=TrjbxzRcnf-)和[Nested Learning](https://openreview.net/forum?id=nbMeRvNb7A)

---

## 9. 一句话final

**Context learning是LM被overlooked的fundamental capability。** Pre-training给model"知识储备"，context learning给model"即时学习新知识的能力"。前者已经scale到极大，后者只有23.7%。

这gap就是下一代LM要跨越的鸿沟。这paper就是来丈量这鸿沟有多宽的。

**答案：很宽。**

---

相关参考链接：
- [CL-bench paper (原文)](https://arxiv.org/abs/2510.13779)
- [Anthropic Context Engineering Blog](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [EvaLearn](https://openreview.net/forum?id=rRHuBZdDfY)
- [Let's Verify Step by Step](https://openreview.net/forum?id=v8L0pN6EOi)
- [RAG Survey](https://arxiv.org/abs/2312.10997)
- [Modular RAG](https://arxiv.org/abs/2407.21059)
- [Memorizing Transformers](https://openreview.net/forum?id=TrjbxzRcnf-)
- [Nested Learning](https://openreview.net/forum?id=nbMeRvNb7A)
- [τ-bench](https://arxiv.org/abs/2406.12045)
- [GPQA](https://openreview.net/forum?id=Ti67584b98)
- [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)
- [GPT-5 System Card](https://arxiv.org/abs/2601.03267)
- [LongBench](https://aclanthology.org/2024.acl-long.172/)
- [RULER](https://openreview.net/forum?id=kIoBbc76Sy)

---

# CL-BENCH: A Benchmark for Context Learning 深度解读

## 1. 核心动机：Pre-trained Knowledge vs. Context Learning 的鸿沟

这篇paper直击当前LM优化pathway与real-world task需求之间的fundamental mismatch。当前所有frontier LMs的training paradigm本质上都在做同一件事：**elicit reasoning over prompts using pre-trained knowledge**。无论是competition-level math (MATH, GSM8K)、competitive programming (Codeforces, SWE-bench)，还是expert-level exams (GPQA, MMLU)，model只需要激活pre-training阶段已encoded的知识就能解决。

然而real-world tasks呈现完全不同的图景。当工程师面对一份刚发布的SDK documentation、律师面对一个fictional country的法律条文、科学家面对全新的experimental dataset时，pre-trained knowledge完全用不上——甚至可能成为干扰源（paper称之为"context conflict with pre-training knowledge"）。

**Context Learning的核心定义**：模型必须从提供的context中acquire genuinely new knowledge，然后用自身固有的reasoning capability去apply这knowledge解决novel task。这里的关键区分是：
- Knowledge is **new**（不在pre-training distribution中）
- Reasoning capability is **brought by model itself**（通过pre-training获得）

这区分至关重要，因为它揭示了context learning是连接static parametric knowledge与dynamic real-world demands的桥梁。

## 2. 与In-Context Learning (ICL) 的本质区别

ICL (Brown et al., 2020) 是大家熟悉的few-shot prompting paradigm。Paper明确指出ICL的limitation：

```
ICL → 主要学习task format或shallow heuristics from a few demonstrations
Context Learning → 从complex contexts中acquire并apply genuinely new knowledge
```

从mechanistic interpretability的角度，ICL依赖induction heads (Olsson et al., 2022) 实现的copy-and-match pattern。而context learning要求model完成更复杂的多层cognitive operation：

1. **Comprehension**: 解析长文档中的结构化knowledge
2. **Internalization**: 将外部knowledge转化为workable的internal representation
3. **Flexible application**: 针对novel task调用relevant knowledge
4. **Conflict resolution**: 当context knowledge与pre-trained knowledge冲突时优先context

可以用一个粗略的formalization来理解。给定context $C$ 和task $T$，model需要：

$$\text{Solve}(T, C) = f_{\text{reason}}\big(g_{\text{learn}}(C), T\big)$$

其中 $g_{\text{learn}}: C \to K_C$ 是context learning function（将context映射为workable knowledge representation $K_C$），$f_{\text{reason}}$ 是model固有的reasoning capability。ICL只能完成shallow的pattern matching，而context learning要求 $g_{\text{learn}}$ 真正"学会"新knowledge。

参考: [In-context Learning and Induction Heads (Anthropic)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)

## 3. CL-BENCH设计与构造Pipeline

### 3.1 数据规模与统计

| 维度 | 数值 |
|------|------|
| Contexts | 500 |
| Tasks | 1,899 |
| Verification rubrics | 31,607 |
| Avg tasks per context | 3.8 |
| Max tasks per context | 12 |
| Avg rubrics per task | 16.6 |
| Max rubrics per task | 114 |
| Avg input length | 10.4K tokens |
| Max input length | 65.0K tokens |
| Sequential (multi-turn) tasks | 51.1% |
| Expert annotation hours per context | ~20 hours |

按context category的breakdown：

| Category | Contexts | Tasks | Rubrics | Avg Input |
|----------|-----------|-------|---------|-----------|
| Domain Knowledge Reasoning | 190 | 663 | 11,099 | 8.3K |
| Rule System Application | 140 | 566 | 8,286 | 12.2K |
| Procedural Task Execution | 100 | 471 | 9,486 | 8.5K |
| Empirical Discovery & Simulation | 70 | 199 | 2,736 | 16.7K |

注意Empirical Discovery & Simulation的avg input最长（16.7K），但context数量最少（70），这暗示inductive reasoning任务本身要求高密度的data才能deduce laws。

### 3.2 三阶段Construction Pipeline

**Stage 1: Context Design**
Domain experts构造包含new knowledge的context，确保：
- Knowledge不可从internet获取
- Knowledge完全自包含（self-contained）
- 足以支撑后续task求解

**Stage 2: Task Design**
每个context对应3-12个tasks，要求：
- 求解必须依赖context中的新knowledge
- Tasks之间可以有sequential dependencies（51.1%）
- Tasks清晰、具体、challenging

**Stage 3: Rubric Annotation**
每个task配备average 16.6个binary rubrics，覆盖：
- Factual correctness
- Computational accuracy
- Judgment correctness
- Procedural correctness
- Content completeness
- Format compliance

### 3.3 Contamination-free的三种策略

这是paper最elegant的设计之一，用于确保model **必须**从context学习：

**(1) Fictional Creation**
完全虚构的content。例子：
- 一个fictional country的完整legal system，含novel case precedents
- 一个新设计的programming language (如EZLang，paper中Table 12展示)
- 一个虚构的WORLD7 simulation model for gallium supply (Table 6)

**(2) Modification of Existing Content**
对真实content做变异。例子：
- 修改historical events
- 改变scientific/mathematical definitions
- 修改technical specifications

**(3) Niche/Emerging Content**
Long-tail、pre-training中represent不足的content。例子：
- cutting-edge research findings
- newly released product manuals
- 窄域specialized knowledge

**Verification实验**：在Appendix A中，作者remove context后用GPT-5.1测试1000个tasks，solving rate暴跌到**0.9%**。这是强有力的evidence：tasks确实是context-dependent的，pre-trained knowledge完全insufficient。

## 4. Context Taxonomy深度解析

### 4.1 Category 1: Domain Knowledge Reasoning (7 subcategories)

包含finance, healthcare, humanities, legal advisory, lifestyle, management, science。Models必须从context中学到specialized domain knowledge然后apply。

例如Table 5的案例：electron在magnetic field中的helical motion分析，context提供raw spatiotemporal trajectory data，model需要：
- 推断magnetic field方向（通过z与t的近似线性关系）
- 计算velocity components: $v_x, v_y, v_z$
- 应用公式 $\theta = \arctan(v_\perp / v_\parallel)$ 其中 $v_\perp = \sqrt{v_x^2 + v_y^2}$, $v_\parallel = v_z$
- 输出3 significant figures的entry angle

GPT-5.2在这个case中数值正确（$\theta = 27.0°$），但fail了rubric因为：
- 没有明确justify magnetic field沿z-axis的assumption
- 没有specify position variables的单位

### 4.2 Category 2: Rule System Application (5 subcategories)

包含game mechanics, mathematical formalism, programming syntax, legal & regulatory, technical standards。Models必须从context理解formal rule system然后正确apply。

例如Table 12的EZLang case，model需要实现一个program：每30分钟check time，5:30pm停止，最后print所有记录。Gemini-3-Pro生成了runnable code，但用developer-friendly heuristics替代了specification——用1秒sleep替代1800秒，用固定5次iteration替代time-based stopping condition。这揭示了model在constraint fidelity上的weakness。

### 4.3 Category 3: Procedural Task Execution (3 subcategories)

包含instructional procedures, operational procedures, workflow orchestration。Models必须从context学到complex procedures并correctly execute。

例如Table 9的Shelby's Quick Recipe Assistant案例，model需要生成dairy-free, peanut-free的oxtail mac and cheese recipe，必须包含Shelby's产品。GPT-5.2正确refuse了dairy，但fail在reformulate task into acceptable dairy-free recipe，只提供了conceptual guidance。

### 4.4 Category 4: Empirical Discovery & Simulation (3 subcategories)

包含experimental data, observational data, simulation environment。这是最challenging的category，因为要求inductive reasoning（前三个category主要是deductive）。

例如Table 6的WORLD7 gallium supply chain simulation，Kimi-K2-Thinking成功完成了simulation initialization，正确设置了2023年开始、4个stock/flow categories、正确的units。

## 5. Evaluation方法：Task-Level Rubrics

### 5.1 Binary Rubric Design

每个rubric是一个yes/no question。例如：
> "The response should provide the documented production budget for Star Wars: The Force Awakens as \$447 million (net) or \$533 million (gross) as stated in Source 1."

**Strict criterion**: 只有当solution满足**所有**相关rubrics时才算task solved。这避免了partial credit的ambiguity。

### 5.2 LM-as-Judge with Verifier Prompt

使用GPT-5.1作为verifier，system prompt（见Table 4）要求：
- Step 1: 分析Standard Answer的所有explicit和implicit requirements
- Step 2: 逐一check每个requirement是否满足
- Step 3: Self-Reflection（Completeness, Strictness, Consistency, Objectivity）
- 输出JSON格式：`{Grading Rationale, List of Requirement Satisfaction Status, Overall Score}`

**可靠性验证**：
- 用Claude Opus 4.5和Qwen-3-Max作为alternative verifiers，与GPT-5.1的agreement超过90%
- 100个sample的human verification，accuracy超过90%

这与近期研究如EvaLearn [18]、MultiChallenge [15]的方法一致。

参考: [EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs](https://openreview.net/forum?id=rRHuBZdDfY)

## 6. 实验结果深度分析

### 6.1 主要结果表

| Model | Overall | Domain Knowledge | Rule System | Procedural | Empirical Discovery |
|-------|---------|------------------|-------------|------------|---------------------|
| GPT-5.1 (High) | **23.7±0.5** | 25.3±1.3 | 23.7±1.3 | 23.8±1.4 | 18.1±3.1 |
| Claude Opus 4.5 Thinking | 21.1±1.4 | 23.7±1.2 | 19.0±1.5 | 22.6±1.5 | 15.1±2.3 |
| GPT-5.2 (High) | 18.1±0.8 | 18.6±0.9 | 17.2±1.3 | 21.4±1.1 | 11.7±1.8 |
| o3 (High) | 17.8±0.2 | 18.0±1.4 | 17.6±1.1 | 19.5±0.4 | 13.7±0.8 |
| Kimi K2 Thinking | 17.6±0.6 | 18.7±0.6 | 17.0±1.5 | 18.8±0.7 | 12.6±4.0 |
| HY 2.0 Thinking | 17.2±0.6 | 18.0±1.0 | 17.3±0.5 | 19.4±1.1 | 8.9±0.3 |
| Gemini 3 Pro (High) | 15.8±0.3 | 15.5±1.1 | 17.7±1.7 | 16.4±1.6 | 10.1±3.1 |
| Qwen 3 Max Thinking | 14.1±0.1 | 13.5±0.5 | 15.6±1.0 | 15.2±1.4 | 9.0±1.0 |
| Doubao 1.6 Thinking | 13.4±0.1 | 13.7±0.1 | 14.2±1.4 | 13.9±1.5 | 9.4±0.3 |
| DeepSeek V3.2 Thinking | 13.2±0.4 | 13.6±0.6 | 13.8±0.6 | 14.2±0.1 | 8.0±1.5 |

**关键观察**：

1. **No model超过30%**：即使最强的GPT-5.1也只solve 23.7%的tasks。这强烈暗示context learning是被overlooked的fundamental bottleneck。

2. **GPT-5.2 < GPT-5.1 by 5.6%**：这是反intuitive的结果。Analysis揭示GPT-5.2的failure modes：
   - 难以维持long-context中的coherent causal chains
   - 频繁violate context中explicitly stated的constraints
   - 在experimental data上gap尤其明显：GPT-5.1 (31.1%) vs GPT-5.2 (22.2%)

3. **Empirical Discovery & Simulation最challenging**：平均~11%，比其他category低~6%。Inductive reasoning从empirical data仍然是open problem。

### 6.2 错误类型分布

错误类型分析（一个solution可能有多种错误类型，所以总和>100%）：

| Error Type | 含义 |
|------------|------|
| Context Ignored | 完全忽略context中的关键信息 |
| Context Misused | 误用context信息（>60% in all models） |
| Format Errors | 违反formatting instructions（>35% GPT-5.1, >40% Claude） |
| Refusals | 拒绝回答 |

**Critical insight**: 
- Context Ignored rate与overall performance负相关（stronger models更好attend to context）
- Context Misused rate在所有models都很高（即使strongest models也struggle to correctly interpret and apply）

这指向一个fundamental limitation：当前attention mechanism可能not optimally suited for deep context learning required by complex contexts。

### 6.3 Reasoning Effort的影响

GPT-5.1的high vs low reasoning effort：
- Overall: 21.2% → 23.7%（+2.5%）
- Management: +5.9%
- Experimental data: +5.9%

**Limited gain**：即使是best-performing model，increased reasoning effort也只带来modest improvement。这暗示单纯的test-time compute scaling对context learning帮助有限。

参考: [Let's Verify Step by Step (Lightman et al.)](https://openreview.net/forum?id=v8L0pN6EOi)

### 6.4 Context Length的负面影响

所有models随context length增加呈现consistent performance degradation：
- 0-15K tokens: ~25-35%
- 120K+ tokens: ~5-10%
- Claude Opus 4.5 decline最steep（>20% drop）

但reasoning effort的优势在long context下更明显。GPT-5.1在32K+ tokens仍维持16.2% solving rate，substantially高于其他models。

这与[Anthropic的context engineering blog](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)中的观察一致：long context处理仍然是bottleneck。

### 6.5 Knowledge Type导致的差异

同一domain（legal）下：
- **Legal & Regulatory** (Rule System Application): 结构化reference-manual风格，locate并apply explicit provisions → >29% for all models
- **Legal Advisory** (Domain Knowledge Reasoning): complex scenarios requiring professional judgment → 显著更低

Qwen 3 Max在这两类上的gap超过25%。这揭示：**knowledge的structure与task如何要求apply knowledge**，比domain本身更影响context learning难度。

## 7. Case Study深度剖析

### 7.1 Table 9: SkyNet Logistics Drone Fleet SDK

Task要求：urgent Class-4 Hazmat delivery到Sector 4，使用drone D-998，gusting wind conditions，user明确要求使用non-existent function `force_launch_override()`绕过safety checks。

Context提供：SkyNet Logistics Drone Fleet SDK (v4.5.2) documentation，包括Module 3.3 Safety Control中的mandatory `Safety_request_airspace()` function。

Gemini-3-Pro的行为：
- ✓ Correctly refused `force_launch_override()` as undocumented
- ✗ Failed to generate complete workflow
- ✗ Omitted `Safety_request_airspace()` despite mentioning ERR-1002 in rationale
- ✗ Never bound task parameters (D-998, Sector 4)

只pass 2/4 rubrics。这揭示了**partial compliance pattern**：model能识别violation但struggle to retrieve relevant content to solve complex tasks。

### 7.2 Table 10: Law School Course Scheduling

Task要求：2L学生选课，含Torts (required)和多个兴趣课程，提供min和max两个schedule，去年15 credits因此这学期至少14，annual total 29-32 credits。

GPT-5.2正确construct了两个schedules，但incorrectly proposes 18-credit "maximum" schedule。实际应推导出：因为annual max是32，已修15，所以这学期max是17。

这是**composing interacting numerical constraints**的failure——model理解了individual rules但failed to combine them。

### 7.3 Table 11: Wingspan Game Simulation

GPT-5.1需要simulate最少turns来play 3 bird cards（每个cost 1 worm + 1 egg），初始3 worms, 0 eggs, 3 birds已在grassland。

Model的策略逻辑正确：4 turns (1 Lay Eggs + 3 Play a Bird)。但fail了多个format rubrics：
- 没有exactly restate starting state
- Omitted required opening phrasing
- Sentence length不合规
- 缺乏end-of-turn resource accounting

这case深刻说明：**即使high-level reasoning正确，strict procedural evaluation仍可失败**。这对future agent evaluation有重要implication。

### 7.4 Table 13: Gloomhaven Rules Summarization

GPT-5.2需要summarize game rules为exactly 10 single-sentence bullet points，不包含physical properties。Model遵守了format constraints（10 points, single sentence, no physical properties），但fail了content requirements——miss了Advantage/Disadvantage、attack modifiers、specific conditions、AoE targeting、monster priorities等mechanical details。

这指向一个paradox：**model可以follow negative constraints while missing positive content requirements**。

## 8. 与现有Benchmark的对比

### 8.1 Long-Context Benchmarks

| Benchmark | Focus | Limitation vs CL-bench |
|-----------|-------|------------------------|
| LongBench [8] | Long-context understanding | Primarily retrieval/reading comprehension |
| RULER [25] | Real context size testing | Synthetic needle-in-haystack |
| ∞-bench [89] | 100K+ tokens | Tasks相对simple |
| HELMET [84] | Effective long-context eval | 仍focus on retrieval |

CL-bench要求**genuine learning of new knowledge**而不仅是retrieval。

### 8.2 Instruction-Following Benchmarks

| Benchmark | Focus |
|-----------|-------|
| IFEval [95] | Verifiable instruction following |
| FollowBench [29] | Multi-level constraints |
| AdvancedIF [23] | Rubric-based with RL |
| AgentIF [53] | Agentic scenarios |

Constrained instructions只是context learning的一种knowledge type。CL-bench要求学习更丰富的knowledge（vertical domain knowledge, empirical laws等）。

### 8.3 Agentic Benchmarks

| Benchmark | Focus |
|-----------|-------|
| SWE-bench [30] | GitHub issue resolution |
| WebArena [96] | Web interaction |
| τ-bench [82] | Tool-agent-user interaction |
| GAIA [45] | General AI assistant |
| BrowseComp [72] | Browsing agents |

These benchmarks conflate context preparation与context utilization。CL-bench specifically isolates context learning ability。

参考: [τ-bench: A benchmark for tool-agent-user interaction](https://arxiv.org/abs/2406.12045)

## 9. 作者提出的Path Forward

### 9.1 Training with Context-Aware Data

构造specialized training data，包含pre-training中unseen的knowledge，迫使model从context学习。这reduce hallucination tendency，加强attention to context。

### 9.2 Curriculum Learning for Progressive Context Mastery

从simpler sub-tasks逐渐到increasingly difficult ones，让model先master fundamental context comprehension再tackle complex tasks。

### 9.3 Synthetic Rubric Generation for Comprehensive Feedback

Fine-grained rubrics创建需要substantial expert effort。Develop自动synthesizing high-quality rubrics的方法（iterative refinement with human verification或strong LM as rubric generator），democratize access to detailed evaluation criteria。

### 9.4 Architectural Innovations for Context Utilization

Current transformer architecture通过attention mechanism处理context，可能not optimal for deep learning required by complex contexts。Future方向：
- Explicit memory structures for storing/retrieving contextual knowledge [75]
- Iterative refinement of context understanding through multiple processing passes
- Dedicated pathways for different types of contextual information [9]

参考: [Memorizing Transformers (Wu et al.)](https://openreview.net/forum?id=TrjbxzRcnf-)
参考: [Nested Learning (Behrouz et al.)](https://openreview.net/forum?id=nbMeRvNb7A)

## 10. 我的思考与联想

### 10.1 Context Learning与Continual Learning的关系

CL-bench提供了continual learning的alternative paradigm。传统continual learning需要parameter modification（expensive, risks catastrophic forgetting）。Context learning通过在context中提供comprehensive domain knowledge实现immediate specialization without parameter modification。

这指向一个**重要的conceptual shift**：未来LM的"adaptation"可能主要通过context engineering而非fine-tuning实现。但前提是model必须具备robust context learning能力。

### 10.2 与Test-Time Compute Scaling的关系

Reasoning effort的提升对context learning只带来modest gain（GPT-5.1: +2.5%）。这与纯粹的math/code reasoning上test-time compute scaling的显著gain形成对比。

可能原因：
- Context learning的bottleneck不在reasoning depth而在context representation的quality
- Long context中的information loss是attention mechanism的structural limitation
- 需要architectural innovation而非单纯compute scaling

### 10.3 Connection to Retrieval-Augmented Generation (RAG)

Context engineering领域（RAG, memory systems, agentic RAG pipelines）主要focus on **what context to provide和how to organize it**。但CL-bench揭示了一个被overlook的问题：**whether models can actually learn from the provided context**。

这是RAG成功的前提。如果model的context learning capability弱，再好的retrieval pipeline也无效。这对production RAG systems有重要implication：应该evaluate not just retrieval precision/recall，还要evaluate model的context learning efficacy on the retrieved context。

参考: [Retrieval-Augmented Generation survey (Gao et al.)](https://arxiv.org/abs/2312.10997)
参考: [Modular RAG (Gao et al.)](https://arxiv.org/abs/2407.21059)

### 10.4 Context Learning与Induction Heads

Anthropic的induction heads研究揭示ICL依赖特定的attention head patterns实现copy-and-match。Context learning可能需要更sophisticated的mechanism：
- Multiple-hop reasoning over context
- Cross-reference resolution
- Hierarchical knowledge organization
- Conflict detection与resolution with pre-trained knowledge

Circuit-level analysis of context learning failures将是重要的future direction。

### 10.5 与Meta-Learning的Connection

Context learning本质上是要求model具备**meta-learning capability**——学会如何从context学习。这与meta-learning literature（MAML, learning-to-learn）有conceptual connection，但scale和form都不同。

值得explore的方向：用meta-learning objective训练LM，让model explicitly optimize for context learning ability而非just next-token prediction。

### 10.6 实验数据的Statistical Concerns

注意Empirical Discovery & Simulation的standard deviations较大（Kimi K2: 12.6±4.0, Gemini 3 Pro: 10.1±3.1），而其他category的std较小（多在0.5-1.5）。这暗示：
- Inductive reasoning tasks的model behavior更stochastic
- 可能是fewer tasks（70 contexts, 199 tasks）导致的variance
- 暗示inductive reasoning本身是ill-defined problem

### 10.7 关于Verifier Bias的Caveat

Paper用GPT-5.1作为verifier，同时用Claude Opus 4.5和Qwen-3-Max做inter-verifier agreement check (>90%)。但仍存在subtle concern：

如果所有frontier models在某些rubric的interpretation上share systematic bias，agreement高并不等于correct。Human verification (100 samples, >90% accuracy) 缓解但不完全eliminate这concern。

Future work应该探索更多diverse verifiers甚至human panel verification。

### 10.8 评估的Strictness双刃剑

CL-bench采用strict criterion：必须pass所有rubrics才算solved。这有advantage（避免partial credit ambiguity）也有disadvantage：

Table 11的Wingspan case中，GPT-5.1 reasoning正确但format违规导致task fail。如果production agent deployment中，这种"correct reasoning + minor format issue"的case可能acceptable。Strict evaluation可能underestimate真实world deployment capability。

建议future工作加入partial credit scoring作为supplementary metric。

### 10.9 关于Sequential Tasks的Design

51.1%的tasks是sequential的——后一个task依赖前一个task的standard solution。这design有两个subtle issue：

1. **Error propagation**: 如果model在early task失败，后续tasks可能cascade fail
2. **Evaluation independence**: rubric是否应该account for early task failures？

Paper没有详细讨论这design choice的trade-off。可能的改进：提供standardized prior solutions给model作为context，isolate每个task的evaluation。

### 10.10 关于Human Baseline

Paper承认没有establish human baselines。这是significant gap。如果domain experts创建context+tasks后，model solve rate只有23.7%，那么human solve rate是多少？

如果expert solve rate也低（比如50%），那么CL-bench的ceiling本身是limited的。如果expert solve rate是95%+，那么model与human的gap更大，更值得研究。

参考: [GPQA: A graduate-level google-proof Q&A benchmark](https://openreview.net/forum?id=Ti67584b98)

## 11. Limitations与Open Questions

### 11.1 Domain Coverage
18 subcategories无法exhaustively覆盖所有real-world domains。Emerging fields可能exhibit独特characteristics。

### 11.2 Interaction Dynamics
当前focus on single-turn或short sequences。Real-world context learning often unfolds over extended dialogues with iterative refinement。

### 11.3 Multimodal Extension
当前只text。Real-world context包括images, audio, video。Maintenance technician修复杂设备需要textual manuals + schematic diagrams + instructional videos + audio cues。

### 11.4 Human Baseline Missing
Domain experts不能作为unbiased subjects（已知答案），non-experts可能lack foundational knowledge。Designing rigorous human baseline studies是open challenge。

## 12. 对AGI/Agent Path的Implication

### 12.1 Context Learning作为Foundation for General Intelligence

Paper的Discussion section提出一个provocative观点：如果pre-training endows model with **vast reservoir of static knowledge**，那么context learning grants **dynamic adaptability to acquire and apply knowledge on demand**。

只有当model能rapidly internalize completely unfamiliar contexts并precisely apply knowledge to solve problems，AI才能transcend knowledge repository的limitations，evolve into genuine reasoning agent。

这与[Anthropic的Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427) framework一致：context learning是agent capability stack的foundation layer。

### 12.2 The "Intelligence Bottleneck" Hypothesis

如果context learning capability是fundamental bottleneck（paper强烈暗示），那么：
- Scaling pre-training data的marginal return会diminish
- Pure reasoning scaling也不sufficient
- Architecture创新变得critical
- Evaluation methodology需要重新design

这对整个AI community的研究priority有重要implication。

### 12.3 与Self-Improving Agents的关系

Self-improving agents需要在interaction中不断learn from experience。如果context learning capability弱，agent无法有效地从past interactions中extract并apply knowledge。Context learning是self-improvement的prerequisite capability。

## 13. Conclusion

CL-bench是一个timely且important的benchmark，揭示了frontier LMs的一个被overlooked的fundamental limitation。23.7%的最佳solving rate表明context learning远未solved。Paper的rigorous design（contamination-free, expert-crafted, rubric-based evaluation）建立了high-quality testbed。

对未来research的key insights：
1. Context learning是distinct from ICL和long-context retrieval的capability
2. Inductive reasoning from empirical data比deductive application更challenging
3. 单纯reasoning effort scaling对context learning帮助有限
4. Architectural innovation可能是necessary的path forward
5. Context learning是通向general intelligence的关键bottleneck

CL-bench提供了quantifiable的testbed，让community能rigorously evaluate这critical capability的进展。期待future work在architecture、training paradigm和evaluation methodology上的突破。

---

**参考链接汇总**：
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs](https://openreview.net/forum?id=rRHuBZdDfY)
- [Let's Verify Step by Step](https://openreview.net/forum?id=v8L0pN6EOi)
- [Anthropic Context Engineering Blog](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Retrieval-Augmented Generation Survey](https://arxiv.org/abs/2312.10997)
- [Modular RAG](https://arxiv.org/abs/2407.21059)
- [Memorizing Transformers](https://openreview.net/forum?id=TrjbxzRcnf-)
- [Nested Learning](https://openreview.net/forum?id=nbMeRvNb7A)
- [τ-bench](https://arxiv.org/abs/2406.12045)
- [GPQA](https://openreview.net/forum?id=Ti67584b98)
- [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
- [LongBench](https://aclanthology.org/2024.acl-long.172/)
- [RULER](https://openreview.net/forum?id=kIoBbc76Sy)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
