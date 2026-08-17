---
source_pdf: OpenResearcher A Fully Open Pipeline for Long-Horizon Deep.pdf
paper_sha256: dcdf28a3b051c14269658167834e245958490fa099f595826f8006769c7e5ed7
processed_at: '2026-08-06T00:52:31-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OpenResearcher 用人话说一遍

## 一句话说清楚这paper在干啥

现在所有做deep research agent的工作都卡在一个尴尬的地方：想训练一个能像人一样上网查资料、多轮推理的AI，但**训练数据从哪来？** 你得让一个强model先跑很多遍"搜东西-读网页-推理"的过程，把它的thinking process记录下来当training data。问题是这个"搜东西"这一步，大家都用Google Search API，**贵、不稳定、还无法复现**。

这paper的trick特别简单粗暴：**Google Search用一次就够了**。先用它把每个问题相关的gold documents抓回来，存成一个15M documents的offline corpus。之后teacher model做synthesis的时候，完全在本地跑search engine，一分钱不花，想跑多少轮跑多少轮，还能精确分析每一步发生了什么。

---

## 为什么这件事难

### Deep research跟普通QA不是一回事

你看传统的multi-hop QA dataset，比如[2WikiMultiHopQA](https://aclanthology.org/2020.coling-main.577/)或者[Natural Questions](https://aclanthology.org/Q19-1026/)，一个问题2-5轮retrieval就能搞定，evidence都清清楚楚摆在那。

但real deep research是什么样子？你得：
- 先发一个broad query看看大概有什么candidate sources
- 然后挑几个promising的网页点进去读全文
- 读完发现信息不全，再refine query重新搜
- 来来回回几十轮，evidence可能fragmented、contradictory、甚至outdated
- 最后还得自己判断"差不多了，可以下结论了"

这种long-horizon任务，[Search-R1](https://arxiv.org/abs/2503.09516)那种2-5 turns的trajectory根本不够看。

### Live web search的三个致命问题

1. **贵得离谱**：你想synthesize 97K条trajectory，每条平均50个tool calls，其中一半是search calls，那就是2.5M次search requests。用[Serper API](https://serper.dev/)要$2,500-$5,760，用[SerpAPI](https://serpapi.com/)要$14,400-$28,800。而且每个failed search path照样烧钱。

2. **不稳定**：今天你搜"who won the 2018 World Cup"出来的是Wikipedia的page，明天可能就变成了某个新闻网站的最新文章。同一个pipeline过两个月再跑，结果完全不一样。**无法复现**就等于无法做严肃的scientific analysis。

3. **无法精确分析**：你想研究"agent在第几步找到gold document了？"这种问题，在live web上根本没法做，因为环境一直在变，你不知道什么是"gold document"。

---

## 他们的解法：Decouple + Offline

### Stage 1: 找难问题

从[MiroVerse-v0.1](https://arxiv.org/abs/2511.11793)采样10%，大概6K个QA pairs。这个dataset专门设计成需要long-horizon multi-hop reasoning。他们试过，强如GPT-OSS-120B这种teacher，平均都要几十个tool calls，还有一部分tail超过100个calls。

**为什么不用简单dataset？** 因为简单dataset训练出来的model只会做简单任务。你想训练deep research agent，training data本身就得体现deep research的complexity。这跟[DeepSeek-R1-Distill](https://arxiv.org/abs/2501.12948)的道理一样——用高质量的long reasoning trajectory做SFT，small model也能学到reasoning能力。

### Stage 2: 一次性Online Bootstrapping

这一步是整个pipeline的"投资"阶段，只做一次：

```
For each question q with reference answer a:
    query = q + " " + a    # concatenate提升recall
    docs = Serper.search(query)
    gold_docs += clean_and_dedupe(docs)
```

最后得到10K gold documents覆盖6K questions。这些docs保证了：**每个问题的evidence一定在corpus里**。

**为什么要concatenate question和answer？** 这是information retrieval里的query expansion技巧，[Azad & Deepak 2019](https://www.sciencedirect.com/science/article/pii/S0306457319301647)有系统综述。Question里可能缺key entity，answer里有ground-truth entity，拼起来search的recall高很多。比如question问"某个1994年被homicide的music group member的母亲叫什么"，answer是"Dee Dee Jackson"，那query变成"... Dee Dee Jackson"就容易搜到Wikipedia那篇关于Dee Dee Jackson谋杀案的文章。

### Stage 3: 建Offline Corpus

把10K gold docs和15M [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) docs混合。FineWeb docs的作用是**distractors**——模拟真实web的scale和噪声。如果没有distractors，search太容易了，model学不到真正的search能力；如果有distractors但没gold docs，search再好也找不到evidence。

然后用[Qwen3-Embedding-8B](https://arxiv.org/abs/2506.05176)做embedding，[FAISS](https://github.com/facebookresearch/faiss)建index。Inference的时候agent发natural language query，retriever返回ranked documents，**完美模拟web search API**。

### Stage 4: Teacher Synthesis

用[GPT-OSS-120B](https://arxiv.org/abs/2508.10925)当teacher，temperature 1.0，top-p 0.95（高温度promote diversity），每个question生成16条trajectory（不同seeds）。配置：
- Max context: 128K tokens
- Max turns: 150
- Top-10 docs per search

跑在64张H100上，2天搞定，每条trajectory最多10分钟。最后filter掉超长、malformed、没conclude的，剩97K+条。

---

## 最核心的设计：Three Browser Primitives

这是这paper我觉得最elegant的地方。他们把人类做research的行为抽象成三个operation：

### Search（搜索）
输入query，返回top-K results，每个result有title、URL、snippet。
**对应人类行为**：在Google输入关键词看搜索结果列表。

### Open（打开）
输入URL，fetch整个document的full content。
**对应人类行为**：点开一个搜索结果，看完整网页内容。

### Find（定位）
在当前open的document里找exact string match。
**对应人类行为**：在长网页里Ctrl+F找特定词。

### 为什么这三个缺一不可？

你看[RQ4 ablation](https://arxiv.org/abs/2506.XXXXX)的数据：

| 工具组合 | Accuracy | Gold Hit率 | 首次命中turn | Token用量 |
|---------|---------|-----------|------------|----------|
| 只有search | 43.86% | 1.45% | 41.00 | 80512 |
| search + open | 56.39% | 51.20% | 20.60 | 58094 |
| 三个全有 | 62.17% | 53.37% | 17.23 | 52249 |

**只有search完全不行**：gold hit率只有1.45%，因为search snippets信息太少了。Model只能基于snippet猜答案，根本看不到完整context。

**加上open是最大jump**：+12.53 points accuracy，gold hit从1.45%飙到51.20%。这很make sense——你必须打开document看全文才能拿到evidence。

**加上find还有+5.78 points**：这个有点反直觉。有了open为什么还需要find？因为web pages很长，有些几千行，让model在context window里scan整个page效率低且容易漏。Find tool让model explicit地定位evidence，相当于把"在page内找信息"这个subtask也explicit化了。

---

## 训练Student Model

Base model: [Nemotron-3-Nano-30B-A3B](https://arxiv.org/abs/2512.20848)（NVIDIA的MoE model，31.6B total, 3.2B activated）

训练配置有几个反直觉的点：
- **Learning rate: 5×10⁻⁵，无decay**：大多数SFT都用cosine decay，这里用constant。可能是agentic data和pretraining data distribution差太多，constant lr让model更稳定吸收新patterns。
- **Context length: 256K tokens**：这个很关键。Long-horizon trajectory动辄100K+ tokens，如果truncate，model学到的是broken reasoning chains。Pre-packing到256K保证完整trajectory被看到。
- **只训练347 steps**：8张H100跑8小时就完了。这个training compute量在现代标准下很小，但效果惊人。
- **Rejection sampling后55K trajectories**：只用答案正确的trajectory做训练。

---

## 结果：真香

### BrowseComp-Plus（closed-web）

| 方法 | Accuracy |
|------|---------|
| GPT-4.1 | 36.4% |
| Claude-4-Opus | 36.8% |
| DeepSeek-R1 | 16.4% |
| Tongyi DeepResearch | 44.5% |
| Nemotron base | 20.8% |
| **OpenResearcher** | **54.8%** |

比base model提升+34.0 points，吊打所有proprietary baselines。而且这是**纯SFT**，没有RL，没有live web training data。

### Open-web benchmarks（generalization测试）

| 方法 | BrowseComp | GAIA | xbench |
|------|-----------|------|--------|
| OpenAI o4-mini | 28.3 | 55.8 | 67.0 |
| Kimi-K2 | 14.1 | 57.7 | 50.0 |
| WebSailor-72B | 12.0 | 55.4 | 55.0 |
| **OpenResearcher** | **26.3** | **64.1** | **65.0** |

**关键insight**：model在offline环境训练，但在live web上generalize得很好。这说明high-quality offline synthesis产生的training signals是transferable的。GAIA和xbench甚至超过o4-mini。

---

## 五个发现，每个都很有意思

### 发现1：失败的trajectory也有用

[RQ1](https://arxiv.org/abs/2506.XXXXX)做了个惊人实验：分别用correct trajectories、incorrect trajectories、mixed训练student。

| Training Data | Accuracy |
|--------------|---------|
| Correct only | 54.81% |
| Incorrect only | 55.06% |
| All | 54.46% |

**三种setting差异仅0.6 points！**

这跟传统rejection sampling的assumption完全相反。传统做法是只保留正确答案的trajectory，认为错误的会教坏model。但这里发现：**错的trajectory里的search strategy、tool calling pattern、reasoning structure照样有supervision value**。

**Intuition**：Deep research agent学的是**how to browse**，而不仅仅是**what answer to produce**。一个最终答错的trajectory，可能前面40步的search和open都是合理的，只是最后一步reasoning错了。这部分合理的内容对model学习browsing behavior仍然有用。

### 发现2：Corpus coverage是hard prerequisite

[RQ2](https://arxiv.org/abs/2506.XXXXX)对比有/无gold documents：

| Setting | Gold Hit | Traj Acc | Final BC+ |
|---------|---------|---------|----------|
| 有gold docs | 29.54% | 56.86% | 54.81% |
| 无gold docs | 1.73% | 43.81% | **6.35%** |

**去掉gold docs后accuracy从54.81%崩塌到6.35%**，这不是marginal drop，是catastrophic failure。

**Intuition**：如果evidence根本不在corpus里，agent再怎么search都找不到。这种失败的trajectory是ambiguous的——可能是search strategy差，也可能是evidence不存在。Online bootstrapping消除了这个ambiguity，让failure的attribution更清晰。

### 发现3：~100 turns够用了

[RQ3](https://arxiv.org/abs/2506.XXXXX) sweep max turn budget，发现accuracy和gold hit率随budget提升，但**在100 turns左右plateau**。

**Intuition**：Long-horizon确实有用，但有diminishing returns。这跟人类做research一样——一个问题你查100轮还查不到，再查100轮大概率也查不到，可能问题本身就需要换approach。

### 发现4：Failure不是因为探索不够，是search strategy差

看trajectory统计：

| Metric | Success | Failure |
|--------|---------|---------|
| Avg tool calls | 38.4 | 71.7 |
| Avg searches | 22.1 | 48.8 |

**Failed trajectories的tool calls几乎是success的2倍！**

进一步分解：
- Search calls: 22.1 vs 48.7（差26.6）
- Open calls: 13.4 vs 19.6（差6.2）
- Find calls: 2.8 vs 3.2（差0.4）

**Intuition**：Failed trajectories不是没努力，是一直在瞎搜。反复reformulate queries但make no grounded progress。这说明hard cases需要**better search mechanisms**，单纯加turns没用。这也是为什么explicit browser primitives重要——它强迫model做grounded navigation而非blind search。

### 发现5：Search到gold doc ≠ 答对

[RQ5](https://arxiv.org/abs/2506.XXXXX)的conditional probabilities：

| 概率 | 值 |
|------|---|
| P(correct \| search-hit) | 61.84% |
| P(correct \| open-hit) | 86.72% |
| P(search-hit \| correct) | 99.38% |
| P(open-hit \| correct) | 95.01% |

**关键gap**：search surface gold doc只给61.84%，但explicit open gold doc给86.72%，差24.88 points。

**Intuition**：Search snippet信息太少，不足以支持正确reasoning。你必须在search result里**点开**gold doc看全文，才能拿到足够evidence。这从概率上证明了open tool的必要性——它把correctness从62%提升到87%。

同时P(search-hit|correct)=99.38%说明几乎所有correct trajectory都涉及gold evidence exposure，evidence是necessary condition。但P(correct|search-hit)只有62%说明evidence不是sufficient condition——拿到evidence后还得正确reason。

---

## 失败案例分析

### Case 5: 找到了gold doc但reasoning错了

GAIA问题问Tri-Rail某趟train的arrival time。Model用99个tool calls成功定位到gold doc（Tri-Rail schedule table），但**误读了table的column layout**，选了departure time而不是arrival time。

**Intuition**：这揭示了deep research的subtle failure mode。Retrieval成功不等于reasoning成功。Table parsing、temporal reasoning、spatial reasoning这些sub-tasks仍然可能出错。Model需要的不只是search能力，还有**在retrieved evidence上做accurate reasoning的能力**。

### Case 6: 只有search tool会陷入infinite reasoning loop

一个complex spatial constraint的问题，model只有search tool。Turn 1搜到一个plausible but wrong result（Boot Monument），之后**98个turns的internal reasoning但0个tool calls**，最终empty answer。

**Intuition**：没有open和find，model无法从wrong search direction recover。它知道结果不对，但没法点进去验证或者refine search。最后只能在thinking里反复纠结，陷入loop。这从反面证明了browser navigation tools的必要性——search alone fundamentally insufficient。

### Case 7: Search alone找不到长尾信息

一个需要识别specific artist的long-tail问题，model用64个search calls都没找到gold doc，最后瞎猜"J. Cole"。

**Intuition**：Search snippets对长尾信息fundamentally insufficient。Gold doc可能是个small gallery page或者biography page，不被search engine prominent indexed。需要open进去看全文才能找到decisive evidence。这再次说明multi-scale browsing（search → open → find）的必要性。

---

## 这paper的真正贡献

### Engineering层面

把large-scale trajectory synthesis的成本从$5,760-$28,800降到$0，从non-reproducible变成fully deterministic，从有rate limit变成无限并行。这让academic lab也能做deep research agent的training data synthesis了。

### Methodology层面

Three browser primitives（search, open, find）的explicit abstraction比implicit single-pass retrieval好得多。这反映了real browsing是multi-scale的：
- Corpus level: 从15M docs里找candidates
- Document level: 从candidates里读全文
- Evidence level: 从全文里定位关键passage

### Scientific insight层面

1. **Correctness filtering不重要**（学browsing而非answer）
2. **Corpus coverage是hard prerequisite**（ROI极高）
3. **Search-hit ≠ Correct**（open是critical transition，+24.88 points）
4. **Failure来自search strategy差而非探索不够**（71.7 vs 38.4 calls）
5. **Long-horizon plateau在~100 turns**

---

## 我的几个联想

### 1. 这跟[DeepSeek-R1-Distill](https://arxiv.org/abs/2501.12948)的思路一致

DeepSeek-R1-Distill证明：用strong reasoning model的long CoT做SFT，small model也能学到reasoning。OpenResearcher把这个idea extend到agentic setting——用strong model的long browsing trajectory做SFT，small model也能学到browsing能力。

### 2. 这跟[RAG](https://arxiv.org/abs/2005.11401)的根本区别

传统RAG是single-shot retrieval：query → retrieve → generate。OpenResearcher是iterative agentic retrieval：query → search → open → find → reason → search again → ... → answer。这反映了deep research和QA的本质区别——前者是process，后者是lookup。

### 3. Offline environment的可分析性价值很大

因为corpus、search backend、browser actions都fixed，可以精确trace每一步的search event、gold document retrieval、opening behavior。这在live-web setting里根本做不到。这种controllability让论文能做5个RQ的系统分析，而不是只report benchmark accuracy。这种**可分析性**本身就是scientific contribution。

### 4. Correctness filtering不重要的发现很反直觉

这open了新的研究方向：能不能用RL直接optimize search strategy而非final answer？如果incorrect trajectories也有用，那reward signal可能不需要是binary correctness，而可以是intermediate metrics like "gold document hit rate"或"search efficiency"。

### 5. Multi-scale browsing的abstraction可能transferable

Search-open-find的三个level abstraction不只适用于web research，可能也适用于codebase navigation（search symbol → open file → find usage）、scientific literature review（search papers → open PDF → find method section）、甚至medical diagnosis（search symptoms → open case studies → find diagnostic criteria）。

### 6. 跟[AlphaGo](https://www.nature.com/articles/nature16961)的self-play有神似

AlphaGo用人类棋谱做supervision，然后self-play提升。OpenResearcher用strong model的trajectory做supervision，如果加上RL可能进一步提升。论文只做SFT没做RL，这是个明显的future direction。[CutBill](https://arxiv.org/abs/2503.XXXXX)就是纯RL的baseline，但效果不如OpenResearcher的SFT。说明好的SFT data比RL更重要，至少在deep research这个task上。

---

## 总结

这paper的核心insight就一句话：**Deep research trajectory synthesis的瓶颈不在model capacity，而在environment design**。你把environment做好（offline、reproducible、gold doc coverage），用strong teacher生成trajectory，用SFT训student，就能得到competitive deep research agent。整个pipeline cost从几千美元降到0，还能做系统分析。

最重要的是，它证明了**好的training data > 复杂的training algorithm**。97K条高质量trajectory + 简单SFT > 复杂RL pipeline。这跟[DeepSeek-R1-Distill](https://arxiv.org/abs/2501.12948)、[OpenThoughts](https://arxiv.org/abs/2506.04178)的发现一致——data quality is king。

参考链接：
- [Paper](https://github.com/TIGER-AI-Lab/OpenResearcher)
- [BrowseComp-Plus Benchmark](https://arxiv.org/abs/2508.06600)
- [GPT-OSS-120B](https://arxiv.org/abs/2508.10925)
- [Nemotron-3-Nano](https://arxiv.org/abs/2512.20848)
- [MiroVerse/MiroThinker](https://arxiv.org/abs/2511.11793)
- [Search-R1](https://arxiv.org/abs/2503.09516)
- [WebSailor](https://arxiv.org/abs/2507.02592)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [OpenThoughts](https://arxiv.org/abs/2506.04178)
- [FineWeb Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Qwen3-Embedding](https://arxiv.org/abs/2506.05176)
- [Megatron-LM](https://arxiv.org/abs/1909.08053)

---

# OpenResearcher: 深度讲解

## 1. 核心问题与动机

这篇paper解决的核心问题是：**如何大规模、低成本、可复现地合成long-horizon deep research trajectories**。

现有方法的痛点：
- **Search-R1** 只产生2-5 turns的短trajectory，远不够真实deep research
- **WebExplorer** 和 **MiroThinker** 依赖live web search API（Google Search等），引入三大问题：
  1. **成本高**：每个失败search path都消耗API费用
  2. **不稳定**：live web随时间变化，难复现
  3. **难分析**：search events依赖变化环境，无法精确分析evidence何时被surfaced、opened或missed

论文的核心insight：**decouple corpus construction from trajectory generation**，一次online bootstrapping build corpus，之后所有synthesis都在offline环境跑。

参考链接：
- Paper: https://arxiv.org/abs/2506.XXXXX (TIGER-AI-Lab)
- Code: https://github.com/TIGER-AI-Lab/OpenResearcher

---

## 2. Pipeline架构解析

### 2.1 三阶段Pipeline

```
Stage 1: QA Question Collection (MiroVerse-v0.1)
   ↓ 采样10% = 6K QA pairs
Stage 2: Offline Corpus Construction
   ├── One-time online bootstrapping (Serper API) → 10K gold documents
   ├── FineWeb corpus → 15M distractor documents (~10T tokens)
   └── Qwen3-Embedding-8B + FAISS index
   ↓
Stage 3: Offline Trajectory Synthesis
   └── GPT-OSS-120B teacher + 3 browser primitives → 97K+ trajectories
```

**关键设计决策**：从MiroVerse-v0.1采样6K questions，因为这些questions需要long-horizon multi-hop reasoning。标准benchmarks如2WikiMultiHopQA和Natural Questions太简单，2-5步retrieval就能解决，不适合训练deep research agent。

### 2.2 Trajectory的形式化定义

论文用ReAct-style paradigm定义trajectory：

$$\mathcal{H}_T = \{(q, s_0, \mathcal{T}_{meta}), (r_1, a_1, o_1), \ldots, (r_T, a_T)\}$$

变量解释：
- $q$：input query
- $s_0$：system prompt
- $\mathcal{T}_{meta}$：tool metadata
- $r_i$：第$i$步的reasoning chain of thought
- $a_i$：第$i$步的action（tool call）
- $o_i$：第$i$步的observation（tool response）
- $T$：trajectory总长度
- $a_T$：final answer（最后一步只有action没有observation）

Policy生成机制：

$$r_t, a_t \sim \pi(\cdot | \mathcal{H}_{t-1})$$

即基于history $\mathcal{H}_{t-1}$ 采样当前thought和action。

Environment响应：

$$o_t = \mathcal{E}(a_t)$$

History更新：

$$\mathcal{H}_t = \mathcal{H}_{t-1} \cup \{(r_t, a_t, o_t)\}$$

### 2.3 Three Browser Primitives

这是paper最有意思的设计之一。人类做research的过程被抽象成三个primitive operations：

| Primitive | 功能 | 对应人类行为 |
|-----------|------|-------------|
| **search** | 给query返回top-K results（title + URL + snippet） | 在Google输入broad query |
| **open** | 打开URL获取full document content | 点击搜索结果进入网页 |
| **find** | 在当前open的document内定位exact string match | Ctrl+F在page内查找特定字符串 |

这个设计的intuition在于**multi-scale information discovery**：
- search：corpus-level retrieval（从15M文档中找候选）
- open：document-level access（从候选文档读全文）
- find：evidence-level localization（在长文档中精确定位关键evidence）

**为什么find这么重要？** 看RQ4的ablation结果——加上find后accuracy从56.39%提升到62.17%（+5.78 points），同时把first gold hit的turn从20.60降到17.23（-3.37 turns），token usage从58094降到52249（-5845 tokens）。这说明explicit evidence localization比让model在context window内implicit scan long document更高效。

---

## 3. 技术实现细节

### 3.1 Corpus构建

**Online Bootstrapping（一次性）**：
- 对每个question，构造search query = `question + " " + reference answer`（利用query expansion提高recall，参考 [Azad & Deepak 2019](https://www.sciencedirect.com/science/article/pii/S0306457319301647)）
- 通过[Serper API](https://serper.dev/)检索web content
- 清洗和去重，移除boilerplate
- 总共提取10K gold documents覆盖6K questions

**为什么query要concatenate question和answer？** 这是information retrieval的经典query expansion技巧。Question本身可能缺少关键entity，而answer包含ground-truth entity，concatenate后能提高gold document的recall。这是offline synthesis可行的prerequisite——必须保证evidence在corpus里。

**FineWeb作为distractor**：
- 15M documents（约10T tokens）来自[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- 作用是模拟真实web的coverage和复杂度，让search任务有"噪声"
- Gold documents和distractors混合后，agent必须通过reasoning找到正确evidence

**Indexing**：
- Embedding model: [Qwen3-Embedding-8B](https://arxiv.org/abs/2506.05176)
- Index: [FAISS](https://github.com/facebookresearch/faiss)
- Served on 4 H100 80G GPUs
- Embedding生成耗时8小时，8×A100 80G GPUs

### 3.2 Teacher Trajectory Synthesis

**关键配置**：
- Teacher: [GPT-OSS-120B](https://arxiv.org/abs/2508.10925)
- Temperature: 1.0, top-p: 0.95（高温度promote diversity）
- 每个question生成16条trajectories（不同random seeds）
- Max context: 128K tokens per trajectory
- Max turns: 150
- Top-10 documents per search step
- 并行化：64 H100 GPUs，每seed split into 8 chunks
- 总耗时：~2 days，每trajectory最多10分钟

**过滤策略**（轻量级）：
1. 超过max context length
2. 包含malformed tool calls
3. 在interaction budget内未达到conclusive answer

过滤后得到97K+ trajectories。

### 3.3 Student SFT配置

**关键数字**：
- Base model: [Nemotron-3-Nano-30B-A3B-Base-BF16](https://arxiv.org/abs/2512.20848)（31.6B total params, 3.2B activated, hybrid Mamba-Transformer MoE）
- Training framework: [Megatron-LM](https://arxiv.org/abs/1909.08053)
- Hardware: 8×H100 GPUs
- Training time: ~8 hours
- Learning rate: 5×10⁻⁵, **无learning rate decay**
- Max context length: **256K tokens**（pre-packed sequences，无truncation artifacts）
- Global batch size: 64
- Total steps: 347
- 训练数据：rejection sampling后保留正确答案的trajectories，约55K条

**为什么无lr decay？** 这是个interesting的设计选择。可能是因为SFT的data distribution和pretraining不同，constant lr让model更稳定地吸收agentic patterns。256K context packing也是关键——完整保留reasoning chains，避免truncation导致model学到broken trajectories。

---

## 4. 实验结果深度分析

### 4.1 Main Results

**BrowseComp-Plus（closed-web benchmark，830 examples）**：

| Method | Accuracy |
|--------|----------|
| GPT-4.1 | 36.4% |
| Claude-4-Opus | 36.8% |
| Gemini-2.5-Pro | 29.5% |
| Kimi-K2 | 35.4% |
| DeepSeek-R1 | 16.4% |
| Nemotron-3-Nano (base) | 20.8% |
| Tongyi DeepResearch | 44.5% |
| CutBill-30B-A3B | 30.3% |
| **OpenResearcher** | **54.8%** |

**Open-web benchmarks（generalization测试）**：

| Method | BrowseComp | GAIA | xbench-DeepSearch |
|--------|-----------|------|-------------------|
| OpenAI o4-mini | 28.3 | 55.8 | 67.0 |
| Claude-4-Sonnet | 12.2 | 57.6 | 64.0 |
| Kimi-K2 | 14.1 | 57.7 | 50.0 |
| DeepSeek-R1 | 8.9 | 30.3 | 55.0 |
| WebSailor-72B | 12.0 | 55.4 | 55.0 |
| DeepMiner-32B | 21.2 | 54.4 | 53.0 |
| **OpenResearcher** | **26.3** | **64.1** | **65.0** |

**关键insight**：offline训练的model能generalize到live-web环境，无需live web training data。这证明了high-quality offline synthesis能产生effective training signals。

### 4.2 Trajectory统计分析

**Success vs Failure对比**（Table 2）：

| Metric | Success | Failure | All |
|--------|---------|---------|-----|
| Rate | 56.7% | 43.3% | 100% |
| Avg tool calls | 38.4 | 71.7 | 52.8 |
| Avg searches | 22.1 | 48.8 | 33.6 |
| Max tool calls | 172 | 185 | 185 |
| Max searches | 109 | 119 | 119 |

**反直觉发现**：Failed trajectories的tool calls几乎是success的2倍（71.7 vs 38.4）！这说明**失败不是因为exploration不够，而是search strategy低效或misdirected**。Hard cases需要better search mechanisms，单纯增加steps无用。

**Tool usage分解**（Figure 4 right）：

| Tool | Success | Failure | 差异 |
|------|---------|---------|------|
| Search | 22.1 | 48.7 | +26.6 |
| Open | 13.4 | 19.6 | +6.2 |
| Find | 2.8 | 3.2 | +0.4 |

**关键insight**：excess tool calls主要来自search operations。Failed trajectories反复reformulate queries但不make grounded progress。Document-level navigation不是primary bottleneck，**query formulation和search drift才是performance gap的driver**。这正是explicit browser primitives要解决的问题。

### 4.3 Pass@k分析

```
Pass@1  = 0.567
Pass@2  = ~0.65
Pass@4  = ~0.72
Pass@8  = ~0.76
Pass@16 = 0.792
```

Pass@1到Pass@16有20%+ gap，说明很多questions是solvable的，但只along certain reasoning paths。

**Bimodal solve rate distribution**（Figure 5 right）：
- ~20% questions pass rate ≈ 0%（极难cases）
- ~30% questions pass rate ≈ 100%（robust solvable）
- ~50% questions in intermediate range

这种bimodal分布是open-ended web-scale research tasks的特征——success往往取决于discover少数critical facts。

---

## 5. 五个Research Questions的深度解析

### RQ1: Final-answer correctness是必要的filtering signal吗？

**实验设计**：固定student backbone、optimization recipe、evaluation protocol，只vary training trajectories subset。

| Training Trajectories | BC+ Accuracy |
|----------------------|-------------|
| Correct only | 54.81% |
| Incorrect only | 55.06% |
| All trajectories | 54.46% |

**惊人发现**：三种setting accuracy差异仅0.6 points！Incorrect trajectories提供equally useful supervision about search structure、tool-use order、evidence inspection、stopping behavior。

**Intuition**：这颠覆了传统rejection sampling的assumption。即使最终答案错误，intermediate的search strategy、tool calling pattern、reasoning structure仍然有价值。Model学的是**how to browse**，而不仅仅是**what answer to produce**。

### RQ2: One-time online bootstrapping是否必要？

**实验设计**：在6K-prompt split上跑4 seeds/prompt，对比有/无gold documents的corpus。

| Setting | Gold Hit ↑ | Traj. Acc ↑ | BC+ ↑ |
|---------|-----------|-------------|-------|
| With gold docs | 29.54% | 56.86% | 54.81% |
| Without gold docs | 1.73% | 43.81% | **6.35%** |

**关键发现**：去掉gold docs后accuracy从54.81%崩塌到6.35%！这不是marginal effect，是catastrophic degradation。

**Intuition**：Corpus coverage是offline synthesis的**hard prerequisite**。如果evidence不在corpus里，失败就ambiguous——可能是search strategy差，也可能是evidence根本不存在。Online bootstrapping消除了这种ambiguity。

### RQ3: 多少turn budget足够？

**实验设计**：固定model、prompt、corpus，sweep max allowed turn budget。

**结果**（Figure 6）：ACC和gold hit rate随budget增加steadily提升，但**在~100 turns后plateau**。

**Intuition**：Long-horizon exploration确实beneficial，但有diminishing returns。一旦agent有sufficient opportunity定位和inspect相关evidence，再多turns无用。这个~100 turns的plateau和论文观察到的100+ tool calls long-horizon tail一致。

### RQ4: Explicit browser tools重要吗？

**实验设计**：用GPT-OSS-120B teacher，固定model、prompt、retrieval backend，vary available browser tools。

| Tools | Acc. ↑ | Gold Hit ↑ | 1st Hit ↓ | Calls ↓ | Avg Tok ↓ |
|-------|--------|-----------|----------|---------|-----------|
| Search only | 43.86% | 1.45% | 41.00 | 70.57 | 80511.69 |
| Search + Open | 56.39% | 51.20% | 20.60 | 53.56 | 58094.04 |
| All three | 62.17% | 53.37% | 17.23 | 49.97 | 52248.64 |

**关键insight**：
1. Search-only表现极差（gold hit仅1.45%），因为search snippets信息incomplete
2. Adding open带来最大jump（+12.53 points accuracy, +49.75 points gold hit），因为document access是evidence获取的prerequisite
3. Adding find在search+open基础上进一步提升（+5.78 points），并减少token usage（-5845 tokens）和earlier gold hit（-3.37 turns）

**Intuition**：Search snippets对deep research远远不够。真实browsing需要multi-scale：从corpus到document到evidence的progressive narrowing。Find tool的value在于explicit evidence localization，比让model在context window内implicit scan长document更高效。

### RQ5: 检索到gold document保证正确答案吗？

**Conditional probabilities**（Table 5 right）：

| Statistic | Value (%) |
|-----------|-----------|
| P(correct \| search-hit) | 61.84 |
| P(correct \| open-hit) | 86.72 |
| P(search-hit \| correct) | 99.38 |
| P(open-hit \| correct) | 95.01 |

**关键发现**：
1. 仅仅是search surface gold doc（search-hit）只给61.84% accuracy
2. Explicitly open gold doc（open-hit）给86.72% accuracy——**+24.88 points提升**
3. 几乎所有correct trajectories都涉及gold evidence exposure（P(search-hit|correct)=99.38%, P(open-hit|correct)=95.01%）

**Intuition**：这区分了**retrieval failure**和**reasoning failure**。Evidence exposure是necessary condition，但不是sufficient condition。Search snippet不够rich，必须open document获取完整context，才能支持正确reasoning。

**Figure 7的进一步分析**：没有gold-document open-hit的trajectories accuracy仅7.9%（n=303），而有至少一个opened gold doc的trajectories保持consistently high accuracy。这从另一个角度验证了open的重要性。

---

## 6. 成本效率分析

| Method | Price/K requests | Total Cost (5.76M requests) |
|--------|-----------------|---------------------------|
| [Serper API](https://serper.dev/) | $1 | $5,760 |
| [SerpAPI](https://serpapi.com/) | $5 | $28,800 |
| Offline retriever (ours) | $0 | $0 |

**Offline设计的额外优势**：
1. **No rate limits**：支持大规模并行synthesis
2. **Fully deterministic**：perfect reproducibility across runs
3. **Zero dependency on proprietary infrastructure**：facilitate open dissemination

---

## 7. Case Study解析

### 7.1 Success Case (GAIA - Merriam-Webster)

Question: "What writer is quoted by Merriam-Webster for the Word of the Day from June 27, 2022?"

OpenResearcher只用了5个tool calls：
1. search "Merriam-Webster Word of the Day June 27 2022" → 找到Jingoism页面
2. open result 0 → 进入Word of the Day: Jingoism页面
3. open (scroll) → 查看更多内容
4. **find "--"** → 精确定位quote attribution "Annie Levin, The New York Observer"
5. open (verify) → 确认答案

**Intuition**：这是search-open-find ideal paradigm的展示。Find操作用于精确提取attribution信息，比让model在context window内scan整个page高效得多。

### 7.2 Failure Case (Reasoning Error)

Question关于Tri-Rail May 27 2019乘客最多的train到达Pompano Beach的时间。

OpenResearcher用了99个tool calls（13 search + 46 open + 40 find），成功定位到gold document，但**误读了table的column layout**，选择了错误的departure station time而非Pompano Beach arrival time。

**Intuition**：这揭示了deep research的subtle failure mode——即使document retrieval成功，单次table parsing错误足以导致wrong answer。这强调了accurate reasoning over retrieved evidence和retrieval本身同等重要。

### 7.3 Failure Case (Missing Browser Tools)

Question关于bi-centennial monument的complex spatial constraints。

当只有search tool时，model在Turn 1找到一个plausible but irrelevant result（Boot Monument），之后**98个turns的internal reasoning但无任何tool calls**，最终empty answer。

**Intuition**：没有open和find，model无法recover from wrong search direction。Search snippets fundamentally insufficient when gold document不被prominent indexed。这从反面验证了browser navigation tools的必要性。

---

## 8. 对Deep Research Pipeline设计的启示

### 8.1 Data Construction

1. **Correctness filtering不是必须的**：Failed trajectories同样有价值，因为model学的是browsing patterns而非answer memorization
2. **Corpus coverage是hard prerequisite**：Online bootstrapping的ROI极高（+48.46 points），不能省
3. **Long-horizon distribution很重要**：需要有100+ tool calls的tail，让model暴露在complex reasoning patterns下

### 8.2 Agent Configuration

1. **Turn budget ~100足够**：超过后diminishing returns
2. **Three browser primitives缺一不可**：
   - Search only：43.86% accuracy
   - +Open：56.39% (+12.53)
   - +Find：62.17% (+5.78)
3. **Explicit evidence localization > implicit context scanning**

### 8.3 Retrieval vs Reasoning

1. **Search-hit ≠ Correct answer**：P(correct|search-hit)=61.84% vs P(correct|open-hit)=86.72%
2. **Open是critical transition**：从61.84%到86.72%的24.88 point gap
3. **Evidence exposure是necessary但不sufficient**：Reasoning over evidence仍然可能出错

---

## 9. 相关工作的技术对比

| System | Trajectory Length | Environment | Reproducibility |
|--------|------------------|------------|-----------------|
| [Search-R1](https://arxiv.org/abs/2503.09516) | 2-5 turns | Live web | Low |
| [WebExplorer](https://arxiv.org/abs/2509.06501) | Longer | Live web | Low |
| [MiroThinker](https://arxiv.org/abs/2511.11793) | Longer | Live web | Low |
| [WebArena](https://arxiv.org/abs/2307.13854) | Short-horizon | Static snapshots | High |
| [Mind2Web](https://arxiv.org/abs/2307.13854) | Short-horizon | Static snapshots | High |
| [Tongyi DeepResearch](https://arxiv.org/abs/2510.24701) | Long | End-to-end pipeline | Medium |
| [WebSailor](https://arxiv.org/abs/2507.02592) | Long | RL pipeline | Medium |
| **OpenResearcher** | **100+ tail** | **Offline** | **High** |

**OpenResearcher的独特position**：First fully open-source pipeline that produces a model rivaling proprietary systems on long-horizon search and reasoning tasks。

---

## 10. Limitations和未来方向

基于论文内容推测的limitations：

1. **Corpus固定性**：15M documents + 10K gold docs是static的，无法capture web的evolving nature
2. **Gold document依赖**：需要知道reference answer才能做online bootstrapping，limit了unsupervised synthesis
3. **Reasoning errors未被addressed**：即使retrieval成功，table parsing等reasoning errors仍导致失败（Case 5）
4. **Bimodal difficulty**：~20% questions pass rate ≈ 0%，这些inherently hard cases可能需要fundamentally different approaches

未来可能的directions：
- **Hybrid online-offline**：Dynamic corpus expansion during inference
- **RL on top of SFT**：论文只用SFT，加入RL可能进一步提升
- **Multi-modal evidence**：Current setup只处理text，扩展到images/tables
- **Better reasoning over retrieved evidence**：Address table parsing等subtle failure modes

---

## 11. 总结：Build Your Intuition

这篇paper的core contribution可以归纳为三个层次：

**Level 1 - Engineering**：Offline synthesis pipeline让大规模trajectory generation从$5,760降到$0，从unreliable变成deterministic。

**Level 2 - Methodology**：Three browser primitives（search, open, find）的explicit abstraction比implicit single-pass retrieval更effective，因为真实browsing是multi-scale的（corpus → document → evidence）。

**Level 3 - Scientific insight**：
- Correctness filtering不重要（学browsing patterns而非answers）
- Corpus coverage是hard prerequisite（ROI极高）
- Search-hit ≠ Correct answer（open是critical transition，+24.88 points）
- Failure来自inefficient search strategy而非insufficient exploration（71.7 vs 38.4 tool calls）

**最反直觉的发现**：Failed trajectories和correct trajectories对SFT同样有效（54.81% vs 55.06% vs 54.46%）。这意味着deep research agent学的是**how to navigate**，而不仅仅是**what to answer**。这open了新的研究方向——能否用RL直接optimize search strategy而非final answer？

参考资源：
- [GitHub Repo](https://github.com/TIGER-AI-Lab/OpenResearcher)
- [BrowseComp-Plus](https://arxiv.org/abs/2508.06600)
- [BrowseComp](https://arxiv.org/abs/2504.12516)
- [GAIA](https://arxiv.org/abs/2311.12983)
- [GPT-OSS-120B](https://arxiv.org/abs/2508.10925)
- [Qwen3-Embedding-8B](https://arxiv.org/abs/2506.05176)
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [Nemotron-3-Nano](https://arxiv.org/abs/2512.20848)
- [MiroVerse/MiroThinker](https://arxiv.org/abs/2511.11793)
- [Search-R1](https://arxiv.org/abs/2503.09516)
- [WebSailor](https://arxiv.org/abs/2507.02592)
- [DeepMiner](https://arxiv.org/abs/2510.08276)
