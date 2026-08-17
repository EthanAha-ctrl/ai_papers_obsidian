---
source_pdf: MEMLENS Benchmarking Multimodal Long-Term.pdf
paper_sha256: b92a5b6d89999ba6ad77fbc32f8f6e9e86a39a49dfab0a8c9ee310edf380affd
processed_at: '2026-08-05T17:25:00-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版MEMLENS

## 一句话版本

**让长上下文模型和带记忆的agent在同一个跑道上赛跑，跑道还故意挖坑——不给你看图你就答不出来，结果发现两边各有各的尴尬。**

## 为什么要造这个benchmark

先说现状有多乱。

你想测一个模型能不能"记住"很长的多模态对话历史。市面上有两条技术路线：

**路线一：long-context LVLMs**——我把context window撑到1M tokens，把整段对话连同所有图片一股脑塞进去，让模型自己慢慢看。代表选手GPT-5.4、Gemini-3.1-Pro、Kimi-K2.5。

**路线二：memory-augmented agents**——context window装不下？那我把历史压缩、index、存到外部store，需要的时候retrieve一段出来。代表选手Mem0、MemOS、M3-Agent、Memory-T1。

问题是，已有的benchmark全都有毛病：

- MMLongBench、MM-NIAH这些multimodal long-context benchmark，input是document不是multi-session对话，只测LVLMs不测agents
- LongMemEval、MemoryAgentBench这些conversational memory benchmark，完全是text-only，图片直接丢了
- LoCoMo、Mem-Gallery虽然有图有对话，但问题能靠文字蒙对，图片是装饰品

**没人让两类系统在真正需要看图的问题上、在controlled context length下、正面对决过。**

MEMLENS就是来填这个坑的。

## 他们怎么挖坑的

核心trick叫**entity abstraction**，这招很妙。

假设要考"金门桥是什么时候建的"。

普通做法：对话里写"我今天去了金门桥"，问题问"金门桥建造年份"。模型靠文字就能作弊。

MEMLENS做法：把"金门桥"从文字里抠掉，换成"the bridge shown in <image>"。文字里只剩"the bridge"，是哪座桥？世界上桥多了去了。只有那张图能告诉你是金门桥。现在问题问"the bridge shown in <image>建造年份"——你不看图根本不知道在问哪座桥。

这招让visual modality变成necessary condition而不是optional hint。

他们用GPT-5.1当user、Gemini-3-Pro当assistant生成对话，用iCrawler从网上爬图，用CLIP + SigLIP + BLIP-2 caption三通道过滤图片质量。对话里evidence session和distractor session用同一套pipeline生成，风格一致，你光看文字风格分不出哪个是needle哪个是haystack（训练DeBERTa分类器只有57.92%准确率，跟瞎猜差不多）。

## 五种考法

789道题，五种memory ability：

**IE（信息提取）**：从单个session里捞一个fact，但fact的关键信息在图里。比如"问题里抽象化的实体是啥颜色"——你得先从图认出是啥，再去文字里找属性。

**MSR（跨session推理）**：跨3-8个session做aggregation。比如"我最近买weights花了多少钱"——价格分散在多个session里，有的价格只印在商品包装图上。

**TR（时间推理）**：比较两个时间区间长短、排序事件、提取日期。有的时间线索是钟表图、日历图，不是文字。

**KU（知识更新）**：追踪一条4步preference chain。"我以前喜欢X，后来改Y，再后来Z，现在喜欢W"——你得答最新的W，不能答被淘汰的X。每一步锚定一张不同的图。

**AR（拒绝回答）**：故意把evidence删掉，正确答案是"我不知道"。测模型会不会硬编。

## 两个重磅ablation实验

### 实验一：证明问题真的需要图

在80.4%包含evidence image的题目上：

| 模型 | 给图 | 不给图 |
|------|------|--------|
| GPT-5.4 | 93.13% | 1.74% |
| Gemini-3.1-Pro | 89.42% | 1.89% |

掉到2%以下——证明问题真的cross-modal，文字蒙不出来。

### 实验二：MSR ceiling是retrieval问题不是reasoning问题

MSR是五种里最难的，所有模型都卡在30%以下。

但如果直接把需要的evidence sessions喂给模型（跳过retrieval）：
- GPT-5.4 MSR：100%
- Gemini-3.1-Pro MSR：90.21%

**说明模型reasoning能力够用，是retrieval把模型坑了。** 你只找到6个session中的4个，算出来的count当然错——但这不是arithmetic错，是evidence没找全。

## 27个LVLM + 7个agent跑下来发现什么

### 32K：大家都还行

top 8 LVLMs挤在6.34%的band内，short-context accuracy不再separate frontier systems。Qwen3.5-122B 58.68%排第一，Kimi-K2.5 54.88%，Gemini-3.1-Pro 54.10%，GPT-5.4 52.72%。

### 128K：分化严重

- Gemini-3.1-Pro只掉2.11%，最稳
- 多个open-weight leader掉13%以上
- GLM-4.6V的AR从93.33%暴跌到30.00%——context一长就开始硬编
- AR是open-weight家族context-driven decline最陡的type

### 没有模型通吃

- GLM-4.6V的TR最强但KU崩
- Qwen3.5反过来
- Kimi-K2.5在32K的MSR最强但长context优势消失
- 只有Gemini-3.1-Pro在128K同时IE、KU、MSR都competitive

### Agent的问题在哪

agent在visual-grounded的type（IE、KU）上gap最大。原因很具体：

- text-only agent（Mem0、MemOS、MemAgent-7B、Memory-T1）把图换成BLIP-2 caption——caption只保留gist，count、attribute、spatial relation这些fine-grained cue全丢了
- multimodal agent（M2A、M3C）虽然存原图，但query time只retrieve embedding，raw pixels拿不到
- M3-Agent是video model，每个session渲染成composite image，细节糊了

对比一下：direct LVLM是pixel-for-pixel attend原始interleaved conversation。这个asymmetry本身就是评测的一部分——agent的released checkpoint就是按它训练时的input format跑的。

### Post-training把abstention训废了

这个发现很重要。

- frozen-backbone agent（Mem0 77.27%、MemOS 68.18%）的AR还OK
- RL/SFT fine-tune过的agent（M2A 22.73%、M3-Agent 18.18%、M3C 9.09%）AR崩盘

M2A用Qwen3-VL-8B backbone，direct inference能拿81.82% AR，套上memory pipeline只剩22.73%。

**reward design只reward了answer correctness和retrieval success，没signal告诉模型"拒绝回答也是对的"。** 这个gap光靠stronger backbone补不上（Appendix G.6做了ablation证明）。

## 两个failure mode的正交性

这是paper最核心的insight，画个表：

| | 短context | 长context |
|---|---|---|
| **Long-context LVLM** | 视觉grounding强，pixel-for-pixel attend | retrieval能力随context增长衰减 |
| **Memory agent** | 视觉fidelity在storage-time compression时损失 | length-invariant，retrieve-then-reason pipeline对长度不敏感 |

两种failure mode正交：LVLMs输给context length，memory agents输给storage-time的有损压缩。

因为每个architecture只cover一个axis，你沿那个axis scaling，另一个axis的failure mode原封不动。这motivates hybrid design：combine long-context attention with structured multimodal retrieval，两个axis一起搞。

## Error analysis告诉我们什么

在128K context下分解wrong answers（7 label taxonomy × image dependency）：

- **IE和KU**：近90%错误是Visual category——没找到evidence image或没读对。一旦到了image，answer基本对。
- **TR**：Mixed和Reasoning混合
- **MSR**：73%是Reasoning category——但结合oracle实验，这73%的"reasoning错"其实是upstream retrieval failure的downstream表现

从32K到128K，near-miss（evidence found但细节错）变成total-miss（evidence都没找到）。**Scaling first harms evidence retrieval, not reasoning.**

## 一些intuition

1. **Cross-modal evidence retention才是真瓶颈**。不是reasoning不够强，是visual evidence在pipeline里被层层压缩丢了。90%的IE错误是visual perception而非comprehension。

2. **Retrieval和long attention是complementary不是competing**。Jin et al.（arXiv:2410.05983）和Asai et al.的Self-RAG（ICLR 2024）在text-only setting也发现了这个。MEMLENS把这个结论扩展到multimodal。

3. **Abstention calibration需要joint optimization**。当前RL fine-tuning把memory management和hallucination control当成两件事，但它们是coupled的。Abstain-R1（arXiv:2604.17073）开始探索这个方向。

4. **Per-type evaluation是必须的**。五种ability的cross-type Spearman correlation很低——IE和KU相关（共享evidence-retrieval axis），MSR和IE/AR弱相关（aggregation axis独立）。一个aggregate score会掩盖differences。

5. **Pipeline architecture比backbone quality更重要**。M2A用8B stock backbone只剩14.21%，Memory-T1用3B小backbone反拿29.50%。Backbone-matched对比M2A vs. direct Qwen3-VL-8B有34.97% deficit。换backbone只能补救14.65%（Mem0）或2.50%（MemOS）。

## 局限性

- 对话是LLM-generated的（GPT-5.1 + Gemini-3-Pro），real human-assistant interaction distribution仍是open question
- Question generator是Gemini-3-Pro，top-evaluated model是Gemini-3.1-Pro——同family不同version，oracle实验间接证明不是generator familiarity bias但没有直接ablation
- Judge是Qwen3-VL-235B，跟benchmark里的Qwen家族model有同family concern，cross-validated过但仍是潜在bias
- Static-length offline evaluation，没测streaming protocol下memory write/query的temporal causality

## 我的take

MEMLENS真正做的事是把community从"scaling context window就完事了"的narrative里拽出来。

它用一个非常精巧的entity abstraction trick强制visual modality变成necessary condition，然后用length-controlled protocol让两类系统暴露出互补的failure mode。结论很清晰：**单一approach走不通**。

Long-context LVLMs的问题不是reasoning能力不够，是evidence retrieval随context增长衰减。Memory agents的问题不是retrieval不行，是storage-time compression把fine-grained visual cue丢了。两个问题正交，scaling任何一个都治不了另一个。

下一步应该是hybrid architecture——保留raw pixels到query time，用structured multimodal retrieval（像ColPali那种late interaction），long-context attention处理retrieved evidence，joint optimization of retrieval + abstention。

这才是long-term multimodal memory的真正research agenda。

Reference：
- MEMLENS code: https://github.com/xrenaf/MEMLENS
- MEMLENS dataset: https://huggingface.co/datasets/xiyuRenBill/MEMLENS
- ColPali: https://arxiv.org/abs/2407.01449
- Self-RAG: https://openreview.net/forum?id=hSyW5go0v8
- Long-context LLMs meet RAG: https://arxiv.org/abs/2410.05983
- Abstain-R1: https://arxiv.org/abs/2604.17073
- MMLongBench: https://arxiv.org/abs/2505.10610
- LongMemEval: https://arxiv.org/abs/2410.10813
- Memory-T1: https://arxiv.org/abs/2512.20092
- M2A: https://arxiv.org/abs/2602.07624

---

# MEMLENS：多模态长期记忆benchmark的深度技术解析

## 1. 研究动机与核心gap

MEMLENS要解决的核心问题是：existing benchmark无法同时满足两个条件——(a) 真正需要visual evidence才能回答问题，(b) 在length-controlled protocol下对比long-context LVLMs和memory-augmented agents。

Existing benchmarks的failure mode可以分类为：

- **Long-context multimodal benchmarks**（MMLongBench [1], MM-NIAH [2], MMLongBench-Doc [3]）只测document理解，不测multi-session对话；只评LVLMs，不评memory agents。
- **Text-only conversational memory benchmarks**（LongMemEval [4], MemoryAgentBench [5]）完全丢弃visual modality，把memory当成single-modality问题。
- **Multimodal conversational benchmarks**（LoCoMo [6], Mem-Gallery [7]）虽然保留images，但大部分question有text-only shortcut，视觉模态冗余。

MEMLENS的设计哲学是：让text deliberately withhold关键信息，只有evidence image能resolve reference，这样visual modality是necessary而非optional。

参考链接：
- MMLongBench: https://arxiv.org/abs/2505.10610
- LongMemEval: https://arxiv.org/abs/2410.10813
- LoCoMo: https://arxiv.org/abs/2402.17753

## 2. Benchmark构造pipeline的四个stage

### 2.1 Stage 1: Multimodal session simulation

每个session的构造流程：

1. **Topic sampling**：从hierarchical ontology采样topic。Ontology有三条track：Identification（~40%，识别商品、地标、食物等实体）、Experience（~40%，日常活动）、Document（~20%，receipts、tickets、schedules等text-rich artifacts）。总共约400个topic titles，展开到约12,000个subtopics。
2. **Image retrieval**：用topic生成image query，通过iCrawler从web爬取候选image batch。
3. **Image filtering**：双stage过滤——
   - Stage 1 multi-channel relevance scoring：CLIP ViT-L/14相似度（threshold 0.30）+ SigLIP ViT-SO400M + text-text cosine（query vs. BLIP-2 caption）。Candidate必须超过CLIP threshold并且至少通过一个secondary channel。
   - Stage 2 negative-content filtering：reject掉带watermark、stock-photo logo、copyright overlay的image。对DocVQA-style image，额外用GPT-4V做watermark检测。
4. **Dual-model dialogue simulation**：用GPT-5.1作为user，Gemini-3-Pro作为assistant，生成multi-turn dialogue。关键设计是image selection先于dialogue generation，这样image-text alignment是structural property而不是post-hoc filter。

### 2.2 Stage 2: Question construction

这一步是MEMLENS最巧妙的设计。四个子步骤：

**(i) Background generation**：从ontology采样topic，Gemini-3-Pro生成3-5句background paragraph，要求至少包含两个named entities（landmark、product、species等）。

**(ii) Entity selection and image retrieval**：选一个entity作为visual anchor，用entity name作为query走与haystack image相同的web-crawling + multi-model scoring pipeline。

**(iii) Entity abstraction**——这是核心创新。把entity在paragraph中替换成anaphor。先对entity分类到55个semantic category之一（place、organization、object、generic fallback），再从type-aware dictionary的约170个entry中采样replacement phrase。比如：

- "Golden Gate Bridge" → "the bridge shown in <image>"
- "Portland Art Museum" → "the gallery I visited, shown in <image>"

这样paragraph不再name entity，anaphor本身under-specified（"the bridge"可以指世界上成千上万座桥），只有evidence image能disambiguate reference。

对KU questions有特殊处理：dictionary-based replacement不够，因为preference chain追踪同category内的具体item（比如successive favorite fruits）。LLM把每个evidence fact重写成不超过5个词的sensory descriptor：例如"blood orange" → "this tangy round thing <image>"，"blueberries" → "these tiny purple spheres <image>"。Descriptor被constrained为对同category多个item都plausible，保持ambiguity。

**(iv) Question generation**：把abstracted paragraph、evidence image、original entity name喂给Gemini-3-Pro，生成、原子evidence facts。Generation prompt强制两个约束：question必须同时需要image和surrounding text才能回答；answer必须可从evidence facts推导，不需要external knowledge。

对MSR questions还有**three-layer text-hackability defense**：
- Anti-leakage prompt rules防止generation model命名entity
- Rule-based pre-filter reject包含entity name或close synonym的fact
- LLM text-only judge确认answer无法仅从textual evidence推导

### 2.3 Stage 3: Evidence session construction

直接把evidence statement插入conversation history会create abrupt semantic shifts，让evidence被similarity-based retrieval trivially locatable。LongMemEval [4]已经证明增加evidence和distractor的contextual similarity能提升retrieval difficulty。

所以每个evidence fact被wrap成一个完整的evidence session，用与haystack session相同的pipeline生成，但是grounded在evidence fact上而不是sampled topic。这样evidence facts match surrounding haystack的topical和stylistic profile。

进一步增加难度：prompt让user model间接提到fact，不强调它。例如要嵌入"I started a new job last month"，user turn可能从问tax withholding update开始，顺便提到job change。

### 2.4 Stage 4: Conversation history assembly

- Evidence sessions按timestamp插入haystack session history，position uniform random。
- 对KU questions，保留evidence session相对顺序（因为是preference update order，critical to answer）。
- 对TR questions，每个evidence session的timestamp严格precede question reference date。
- Haystack sessions被curated为contextually related但对question uninformative，绝不包含answer-relevant details。
- 通过变化haystack session数量产生四个standardized context lengths（32K/64K/128K/256K）。
- 为避免通过image clustering揭示evidence position，保持fixed text-per-image ratio，用ShareGPT和UltraChat [8]的text-only filler session padding。

## 3. Cross-modal token-counting scheme

MEMLENS采用MMLongBench [1]的cross-modal counting scheme，让text和vision token对齐。具体而言：

- Image按~2,000 vision token计（这个值取决于LVLM的vision encoder实现，比如Dynamic Resolution ViT会根据image实际尺寸产生不同token数）。
- Text按标准BPE tokenizer计。
- Cross-modal alignment让text和image在context length budget上可比。

Dataset statistics（Table 2）：
- 789 questions
- 5 types / 9 subtypes
- 2,145 evidence sessions
- Avg. ~10 turns/session
- Avg. ~1.5 images/session
- ~2,000 tokens/image
- Sessions/instance：14（32K）→ 93（256K）
- Images/instance：20（32K）→ 138（256K）

## 4. 五种memory abilities的数学形式化

### 4.1 Problem formulation（Appendix C.1）

MEMLENS的evaluation instance是4-tuple $(S, q, I, a)$：

- $S = [(t_1, M_1), \dots, (t_N, M_N)]$：N个timestamped multi-turn session序列，$t_1 < \cdots < t_N$，每个session interleaving text和images
- $\mathcal{V}(S)$：S中所有image的集合
- $I \subseteq \mathcal{V}(S)$：携带answer-critical visual information且无法从surrounding text恢复的image子集
- $q$：query，针对5种memory ability之一
- $a$：gold answer，对AR item是literal string `NOT_MENTIONED`

Correct system必须：(i) 在long、distractor-heavy history中localize relevant evidence sessions；(ii) 在$I$上做cross-modal reasoning ground answer。

### 4.2 Information Extraction (IE)

IE测试从single evidence session recall specific fact。两个subtype：

**IE-Entity** ($n=120$)：two-hop chain。
- Hop 1：在evidence image中identify一个abstracted entity
- Hop 2：从surrounding text retrieve associated information

形式化：给定entity $e$ 在image $I_e$ 中visualized但text中只有anaphor $a(e)$（如"the bridge shown in <image>"），question要求retrieve property $p(e)$ from text。模型必须计算 $e = \text{VisualResolve}(I_e)$，然后 $p(e) = \text{TextRetrieve}(S, e)$。

5个sub-skill：disambiguation（区分similar-looking entities）、alignment（匹配image到textual description）、counting（enumerating image items）、spatial reasoning（locating objects相对位置）、arithmetic（computing from visually presented numbers）。

**IE-PrevInfo** ($n=126$)：recall earlier session的visual detail。Session reference被abstracted而非entity本身。3个subtype对应image source：screenshot of chat interface、app/web interface、natural photograph。

### 4.3 Multi-Session Reasoning (MSR)

MSR测试跨3-8个session的aggregation。三个subtype：

**MSR-Arithmetic** ($n=50$)：sum or compute over prices/quantities scattered across sessions。形式化：给定 $k$ 个evidence sessions $\{s_1, \dots, s_k\}$，每个包含value $v_i$（可能只在image中），answer = $\sum_{i=1}^{k} v_i$ 或类似聚合。至少一个operand只通过image可见。

**MSR-Counting** ($n=46$)：count entities matching criterion across sessions。

**MSR-Entity** ($n=47$)：determine whether two cross-session references denote same entity，via counting distinct entities或Y/N identity matching。

### 4.4 Temporal Reasoning (TR)

TR测试temporal reference和visual content的joint reasoning。两个subtype：

**TR-Duration Cmp** ($n=91$)：比较两个interval。Interval endpoints来自textual dates、session timestamps、visual cues（clocks、calendars）的混合。形式化：$D_1 = t_{1,e} - t_{1,s}$，$D_2 = t_{2,e} - t_{2,s}$，answer = $\text{argmax}_{i \in \{1,2\}} D_i$。

**TR-Temporal Grounding** ($n=103$)：
- Order ranking ($n=24$)：sort events chronologically
- Date extraction ($n=79$)：answer "When did X happen?" in YYYY/MM/DD

三个generation mode：
- Mode B：temporal cue本身是visual artifact（clock、calendar）
- Mode C：entity image + explicit textual dates
- Mode D：entity image + session-level timestamps（implicit temporal anchors）

### 4.5 Knowledge Update (KU)

KU测试track evolving user attribute across chain of four successive updates。形式化：给定update chain $[v_1, v_2, v_3, v_4]$，每个 $v_i$ anchored by image $I_i$，answer = $v_4$（latest state）。模型必须distinguish最新state from superseded ones。

Reference [57] framework（Knowledge conflicts for LLMs survey）理论支撑：每个update构成conflict，模型必须resolve到latest。

### 4.6 Answer Refusal (AR)

AR不是core memory retrieval task，而是calibration check for hallucination detection。所有supporting evidence被remove，correct model必须decline而非hallucinate [58]。Gold answer是literal string `NOT_MENTIONED`。

## 5. Cross-modality validation的关键实验

MEMLENS做了一次image-ablation study验证questions确实require visual evidence（Table 3）。在80.4%包含evidence image的questions上：

| Model | Input | Overall | IE | MSR | TR | KU | $\Delta$ |
|-------|-------|---------|-----|-----|-----|-----|---------|
| GPT-5.4 | With evidence image | 93.13 | 94.31 | 100.00 | 96.91 | 75.86 | - |
| GPT-5.4 | W/o evidence image | 1.74 | 0.41 | 0.00 | 5.15 | 0.00 | -91.39 |
| Gemini-3.1-Pro | With evidence image | 89.42 | 89.02 | 90.21 | 96.19 | 82.24 | - |
| Gemini-3.1-Pro | W/o evidence image | 1.89 | 0.00 | 0.00 | 6.19 | 0.00 | -87.53 |

关键观察：
- "With evidence"实验只提供question + evidence facts + evidence images（无haystack），验证question本身answerable
- 去掉image后，accuracy collapse到 < 2%
- 两个frontier LVLMs收敛到near-identical collapse，说明questions高度multimodal
- 整个benchmark中65.7% image-essential、14.7% image-supportive、19.6% text-sufficient

## 6. 评估setup的关键细节

### 6.1 Model roster

- **27 LVLMs**：3 closed-source（GPT-5.4、Claude Sonnet 4.5、Gemini-3.1-Pro）+ 24 open-source（Kimi-K2.5、Qwen3.5 family、Qwen3-VL family、GLM-4.6V、Gemma3 family等）
- **7 memory-augmented agents**：
  - 3 multimodal pipelines：M3-Agent（ColPali + RL-trained Qwen2-VL-7B）、M2A（dual-layer SQLite + SigLIP2 + Qwen3-VL-8B）、M3C（LoRA-adapted Qwen2-VL-2B session retrieval）
  - 4 text-only pipelines：Mem0（FAISS + Qwen3-8B）、MemOS（layered memory + Qwen3-8B）、MemAgent-7B（sliding-window + RL Qwen2.5-7B）、Memory-T1（BM25 + RL Qwen2.5-3B）

### 6.2 Input adapter asymmetry

Table 5列出每个agent的input format：

| Agent | Backbone | Write-time visual | Answer-time visual |
|-------|----------|-------------------|-------------------|
| M3-Agent | Video LVLM (Qwen2-VL-7B) | Composite per-session image | Retrieved session composite(s) |
| M2A | Native LVLM (Qwen3-VL-8B) | Original images | Stored embeddings |
| M3C | Native LVLM (Qwen2-VL-2B) | Original images | Stored embeddings |
| Mem0 | Text LLM (Qwen3-8B) | BLIP-2 captions only | Captions only |
| MemOS | Text LLM (Qwen3-8B) | BLIP-2 captions only | Captions only |
| MemAgent-7B | Text LLM (Qwen2.5-7B) | BLIP-2 captions only | Captions only |
| Memory-T1 | Text LLM (Qwen2.5-3B) | BLIP-2 captions only | Captions only |

Text-only agents用BLIP-2 [9] captions替代image。M3-Agent是video-based，每个session渲染成composite image，sessions作为image sequence喂入。M2A和M3C直接处理multimodal input。

Direct LVLMs则pixel-for-pixel attend original interleaved conversation。这个asymmetry是关键：reported deficits conflate adapter-induced visual information loss和retrieval/reading quality。

### 6.3 Agent evaluation on 195-question canonical subset

因为agent pipeline慢（M2A每题约60× direct LVLM时间），agents在stratified 195-question subset评估（约1/4），而LVLMs在全部789题评估。Subset per-type composition与full benchmark差异 < 0.2个百分点（Table 15）。

### 6.4 LLM-as-Judge metric

Judge用Qwen3-VL-235B-A22B-Instruct（thinking disabled），cross-validated by GPT-5.4-mini（$\kappa = 0.93$，$\rho = 0.97$ at model level）和3-annotator human consensus（$\kappa = 0.86$）。

String match在MEMLENS上fail因为answer格式heterogeneous：binary choice、counts、currency、date values、ranked orderings、short fill-ins、explicit refusals。LVLMs经常把correct answer wrap在multi-sentence rationale或thinking trace里。

Coverage和Per-Answer Accuracy分解：
$$J \approx \frac{\text{Cov} \times \text{PA} \times 699 + \text{AR}_{\text{correct}}}{789}$$

其中699是answerable subset，789是full benchmark。这个分解暴露coverage-accuracy trade-off。

## 7. 主要实验结果的关键数据

### 7.1 32K short-context: open-weight leader接近 frontier

Table 13的32K列显示top 8 LVLMs落在6.34% band内，short-context accuracy不再separate frontier systems。

- Qwen3.5-122B-A10B: 58.68%
- Kimi-K2.5: 54.88%
- Gemini-3.1-Pro: 54.10%
- GPT-5.4: 52.72%

### 7.2 128K long-context: degradation pattern分化

- Gemini-3.1-Pro：从54.10% → 51.99%（仅-2.11% drop，degrade least overall）
- 多个open-weight leader损失超过13%
- GLM-4.6V AR从93.33%暴跌到30.00%
- AR是open-weight LVLMs context-driven decline最陡的type

### 7.3 Per-type ceiling

| Type | Ceiling at 32K | Hardest aspect |
|------|---------------|-----------------|
| AR | 97.78% | Calibration check，easiest |
| TR | 60.82% | Timestamps提供explicit retrieval anchor |
| IE | 74.39% | Two-hop reasoning through abstracted image |
| KU | 50.86% | 4-fact chain，missing single anchor flips state |
| MSR | 44.06% | Cross-session aggregation over 3-8 sessions |

MSR是hardest type：只有Kimi-K2.5（44.06%）和Gemini-3.1-Pro（32.17%）clear 30% by margin。这exposes MSR为shared capability ceiling。

### 7.4 No model dominates all abilities

- GLM-4.6V leads TR但collapses on KU
- Qwen3.5 inverts pattern
- Kimi-K2.5在32K strongest on MSR但advantage at longer contexts fades
- Gemini-3.1-Pro是唯一同时competitive on IE、KU、MSR at 128K

Memory agents的inverted profile：Memory-T1通过BM25 date matching达到高TR accuracy，但IE远低于direct LVLMs，因为keyword retrieval substitute了IE需要的visual grounding。

## 8. Error analysis的深入解读

### 8.1 Seven-label wrong-answer taxonomy（Appendix G.4）

Wrong answer被分类为7个label，分两组：

**Near-miss（evidence located before erring）**：
- Grounding failure：right region, wrong detail
- Computation slip：right operands, wrong arithmetic
- Closed-set selection：right set, wrong element
- Stale retrieval：right fact, pre-update version（KU only）

**Total-miss（no correct evidence anchor）**：
- Unsupported answer：no anchor, fabricated content
- Answerability failure：answered unanswerable item
- Non-answer pathology：never produced final answer

### 8.2 Five-category modality view（Figure 4b）

7个label × per-question image-dependency（image-essential / image-supportive / text-sufficient）= 5个disjoint modality categories：

- **Visual**：Grounding failure或unsupported answer on image-essential question
- **Textual**：Grounding failure/unsupported answer on text-sufficient question，或任何stale-retrieval error
- **Mixed**：Grounding failure/unsupported answer on image-supportive question
- **Reasoning**：Computation slip、closed-set selection、answerability failure
- **Output**：Non-answer pathology

### 8.3 Per-type error signature

在128K context下：
- IE和KU：近90%错误落在Visual category——模型fails to locate或read evidence image。一旦到达image，answer通常correctly extracted。
- TR：Mixed和Reasoning split，reflecting image-supportive grounding和小closed-set selection。
- MSR：唯一被Reasoning category主导（73%）。

### 8.4 Oracle-retrieval diagnostic（Appendix G.7）

这是MEMLENS最insightful的实验之一。Cross-modality ablation显示，当frontier models接受ground-truth evidence sessions（bypass haystack retrieval）：
- GPT-5.4 MSR accuracy：100.00%
- Gemini-3.1-Pro MSR accuracy：90.21%

这证实MSR的counting和arithmetic operations在frontier reasoning capacity之内。**MSR的30% ceiling是retrieval-bounded而非reasoning-bounded**。当模型只locate了6个required sessions中的4个，count必然错误，registers as aggregation error despite originating from retrieval miss。

剩下~10% gap for Gemini-3.1-Pro（90.21% vs 100%）suggests小部分MSR items（likely更复杂的MSR-Arithmetic）确实challenge reasoning capacity even with perfect evidence delivery。

## 9. Agent pipeline信息loss诊断（Appendix G.6）

### 9.1 Retrieval-dominated vs. comprehension-dominated modes

对3个有retrieval logs的agents（Mem0、Memory-T1、M3C）做error decomposition：

- **M3C**：78.1%错误是retrieval-bottlenecked（mean recall 0.26），LoRA session retriever never surfaces relevant evidence
- **Mem0和Memory-T1**：相反，retrieve evidence at high recall（0.82-0.89），但87-95%错误occur after successful retrieval——backbone无法reason over surfaced content

这call for不同intervention：M3C需要better retrieval，Mem0和Memory-T1需要stronger reading comprehension。

### 9.2 Pipeline architecture dominates backbone quality

M2A用Qwen3-VL-8B-Instruct（8B stock backbone），只score 14.21%。Memory-T1用2.5×小的Qwen2.5-3B但score 29.50%。Backbone-matched contrast：M2A 14.21% vs. direct Qwen3-VL-8B-Instruct 49.18%，**34.97% deficit on same backbone**。

### 9.3 Controlled backbone ablation（Table 21）

| Framework | Backbone | Overall |
|-----------|----------|---------|
| Mem0 | Qwen3-8B (default) | 32.50 |
| Mem0 | gpt-4.1-mini | 43.15 |
| Mem0 | Qwen2.5-7B | 28.50 |
| Mem0 | $\Delta$ | 14.65 |
| MemOS | Qwen3-8B (default) | 34.00 |
| MemOS | Qwen2.5-7B | 36.50 |
| MemOS | $\Delta$ | 2.50 |

14.65%和2.50%的spread都well below 34.97%的M2A deficit。Architecture是dominant factor。

### 9.4 Context invariance

六个agent在32K-256K范围内stay within ±7%，因为retrieve-then-reason pipeline length-invariant by construction。Jaccard overlap between adjacent context lengths exceeds random baseline by factor 3.3-6.9。但absolute deficit仍large，因为memorization和retrieval时丢失的信息outweigh context-robustness advantage。

## 10. Session indistinguishability validation（Appendix G.8）

为了验证evidence sessions和haystack sessions没有exploitable stylistic fingerprint，训练binary classifier：

- TF-IDF logistic regression（unigrams + bigrams，10,000 features）：56.71% ± 1.20% accuracy，ROC-AUC 0.59
- Fine-tuned DeBERTa-v3-base（512-token truncation，3 epochs）：57.92% ± 0.85% accuracy，ROC-AUC 0.55

都只marginally above 50% chance floor（Cohen's $h < 0.16$，small effect）。Vocabulary ablation逐渐remove top-K discriminative n-grams（K = 20, 50, 100, 200, 500, 1000），accuracy始终在54.77%-56.71%范围内。这证实evidence sessions carry no practically exploitable stylistic fingerprint。

## 11. Post-training weakens abstention的关键发现

Memory agents在AR上远低于direct-inference counterparts。两类对比：

- **Frozen-backbone agents**（Mem0 77.27%、MemOS 68.18%）：保留backbone的abstention behavior
- **RL/SFT-finetuned agents**（M2A、M3-Agent、M3C、MemAgent-7B、Memory-T1）：collapse到9-22% AR

M2A在Qwen3-VL-8B backbone上只score 22.73% AR，而同backbone direct inference score 81.82%。

这suggests当前RL/SFT fine-tuning的reward design主要optimize answer correctness和retrieval success，没有signal that refusing unanswerable question是correct的。未来agent design应该jointly optimize memory access、answer accuracy、evidence-sensitive abstention [10]。

## 12. Memory ability correlation分析

Spearman correlation分析（Figure 5）在32K揭示：

- IE和KU最强相关（retrieval-oriented pair，shared need to locate relevant evidence image）
- KU和AR在32K相关（some KU questions depend on accurate evidence selection）
- MSR和IE、AR weak correlation（MSR main challenge是aggregating across multiple evidence pieces而非retrieving single image）

这reveal两个complementary difficulty axes：
- **Evidence-retrieval axis**：IE、KU
- **Aggregation axis**：MSR

单一aggregate score会obscure long-context abilities之间的differences，motivates per-type evaluation。

## 13. Complementary scaling trade-offs的intuition

Figure 4a展示LVLMs和memory agents在context scaling上structurally different的响应：

**LVLMs degrade on retrieval-heavy types**：
- IE损失~20%，KU损失~12%（evidence images在growing filler content中更难locate）
- MSR表面flatness是floor effect near 30%而非genuine robustness
- AR从~75%（32K）跌到~45%（128K）——growing filler content erodes abstention，push LVLMs hallucinate on unanswerable questions

**Memory agents length-stable**：
- 六个agent在32K-256K内stay within ±7%
- Retrieve-then-reason pipeline length-invariant by construction

两个failure modes正交：
- LVLMs lose to context length
- Memory agents lose to lossy multimodal compression at storage time

因为每个architecture只cover一个axis，scaling along that axis leaves other failure mode untreated。这motivates **hybrid designs that span both axes**：combine long-context attention with structured multimodal retrieval，rather than scaling either component in isolation。

## 14. 架构启示和未来方向

MEMLENS的核心insight可以总结为：

1. **Multimodal evidence retention**是principal bottleneck。在lossy cross-modal storage下，fine-grained visual cues（counts、attributes、spatial relations）被discard。Future memory architectures应该preserve image-level evidence而非caption-based compression。

2. **Retrieval和long attention是complementary而非competing**。这与Jin et al. [11]和Asai et al. [12]在text-only setting的发现一致。Hybrid architecture应该combine long-context attention with structured multimodal retrieval。

3. **Abstention calibration**需要joint optimization。当前RL fine-tuning for memory management optimize retrieval success但ignore hallucination control。

4. **Cross-modal evidence-retrieval fidelity > reasoning improvements**。90%的IE errors是visual perception而非comprehension。Scaling first harms evidence retrieval, not reasoning。

5. **Per-type evaluation是必要的**。五种abilities低cross-type correlation，single aggregate score会obscure differences。

## 15. Limitations和open questions

- **Synthetic conversation naturalness**：LLM-generated对话（GPT-5.1 user + Gemini-3-Pro assistant）+ human-in-the-loop review。Real long-term human-assistant interaction distribution仍是open question。
- **Generator-test-taker overlap**：Question generator是Gemini-3-Pro，top-evaluated model是Gemini-3.1-Pro（同family不同version）。Oracle-retrieval diagnostic间接显示128K leaderboard位置not explained by intra-family generator familiarity。
- **Judge limitations**：$\kappa = 0.86$是judge-vs-consensus-label agreement，inter-annotator $\kappa$未separately report。
- **Static-length vs. streaming evaluation**：当前offline frozen multi-session history evaluation。Streaming protocol with temporal causality between memory writes和queries [13]是future work。

## 16. 关键reference链接汇总

- Paper: https://github.com/xrenaf/MEMLENS
- Dataset: https://huggingface.co/datasets/xiyuRenBill/MEMLENS
- MMLongBench: https://arxiv.org/abs/2505.10610
- LongMemEval: https://arxiv.org/abs/2410.10813
- LoCoMo: https://arxiv.org/abs/2402.17753
- Mem-Gallery: https://arxiv.org/abs/2601.03515
- MemoryAgentBench: https://arxiv.org/abs/2507.05257
- Mem0: https://arxiv.org/abs/2504.19413
- MemOS: https://arxiv.org/abs/2505.22101
- MemAgent-7B: https://arxiv.org/abs/2507.02259
- Memory-T1: https://arxiv.org/abs/2512.20092
- M2A: https://arxiv.org/abs/2602.07624
- M3-Agent: https://arxiv.org/abs/2508.09736
- M3C: https://arxiv.org/abs/2506.00421
- Abstain-R1: https://arxiv.org/abs/2604.17073
- Knowledge conflicts survey: https://arxiv.org/abs/2402.17753
- R-Tuning: https://aclanthology.org/2024.naacl-long.395/
- Needle in a haystack: https://github.com/gkamradt/LLMTest_NeedleInAHaystack

## 17. 我的intuition总结

MEMLENS的深层insight可以归纳为一个不等式：

$$\text{Accuracy}_{\text{hybrid}} \gg \max\left(\text{Accuracy}_{\text{long-context}}, \text{Accuracy}_{\text{memory-agent}}\right)$$

因为两种approach的failure modes正交。Long-context LVLMs在short context提供pixel-for-pixel visual grounding，但随context增长lose retrieval ability。Memory agents length-invariant但storage-time compression丢visual fidelity。

Hybrid architecture的设计原则应该是：
1. **保留raw pixels直到query time**（避免caption-based compression）
2. **Structured multimodal retrieval**（像ColPali那样的late interaction model）
3. **Long-context attention处理retrieved evidence**（avoid lossy intermediate representation）
4. **Joint optimization of retrieval + abstention**（不要把hallucination control和memory management分离）

MEMLENS的真正贡献是建立了一个length-controlled、cross-modal-validated的diagnostic protocol，让任何声称solves long-term multimodal memory的系统都必须在五个ability axes上同时被stress test。这把research community从"scaling solves everything"的narrative中解放出来，强制我们面对cross-modal evidence retention这个真正困难的problem。
