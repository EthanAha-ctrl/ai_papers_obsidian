**OpenEvidence** 是一个领先的 AI 驱动型医学信息平台，被形象地称为“**医生版 ChatGPT**”。它由医生和计算机科学家团队共同构建，总部位于美国，主要面向医疗保健专业人员（医生、药师、研究人员等）提供经过同行评审的最新医学循证支持。

- **临床决策支持**：医生通过自然语言提问（如某种药物的最新研究或复杂病例的治疗方案），平台会实时检索大量经过同行评审的医学文献、FDA 和 CDC 的权威数据，并生成精准、简洁的摘要。
- **对抗“信息过载”**：其愿景是实现“**医学知识的 JPEG 压缩**”，将繁杂的长尾医学信息精炼为可在临床诊疗现场直接使用的关键知识。
- **权威合作**：它是首个通过《新英格兰医学杂志》（NEJM）审查的医学人工智能系统，并与其建立了战略合作伙伴关系。


- **极高的医生渗透率**：截至 2026 年初，全美约有 **40% 的医生** 每天都在使用该平台。
- **超级独角兽估值**：据 [YouTube (2026/03)](https://www.youtube.com/shorts/YgpR0VFZtF4) 等来源报道，该公司在成立仅三年后，估值已达到约 **120 亿美元**。
- **创新商业模式**：不同于传统的医疗软件销售，OpenEvidence 采用了类似“消费互联网”的模式，向医生**免费开放**基本功能，并通过广告（如精准展示新药循证数据）或席位费来产生收入。
- **垂直领域专业性**：它并未追求像 ChatGPT 那样的通用聊天能力，而是采用集成架构，由多个特定模型分别负责文献的检索、重排和分析，确保信息的准确性和权威性。
---



让我先获取这个网站的内容，并搜索相关信息。











好的，我现在有足够的信息来给你一个全面的介绍。

---

# OpenEvidence — "ChatGPT for Doctors"

## 一、公司概览

**OpenEvidence** 是一家总部位于 **Cambridge, Massachusetts** 的 American AI healthcare company，成立于 **2022 年**，核心产品是一个面向 **physicians** 的 **AI-powered clinical decision support platform** — 本质上就是一个 **medical search engine + LLM reasoning engine**。

> 它的 mission：**"Organizing and expanding the world's medical knowledge and making it more useful."**

截至 2026 年初：
| 指标 | 数据 |
|---|---|
| Valuation | **$12 billion** (Series D) |
| 总融资 | **~$700 million** |
| 年收入 (ARR) | **~$150M** (2025 年, Sacra 估计, 同比增长 **1,803%**) |
| Gross Margin | **~90%** |
| 美国 Physician 覆盖率 | **~40%** |
| 单日最高 Clinical Consultations | **1 million** |

参考: [CNBC报道](https://www.cnbc.com/2026/01/21/openevidence-chatgpt-for-doctors-doubles-valuation-to-12-billion.html) | [Sacra 分析](https://sacra.com/c/openevidence/)

---

## 二、创始人背景

### Daniel Nadler, Ph.D.
- 持有公司约 **60% 股份**（已成为 billionaire）
- 之前创立了 **Kensho Technologies**，一家用 AI 做 **financial intelligence** 的公司，2018 年被 **S&P Global** 以 **$550 million** 收购
- Harvard Ph.D. 背景，当年读博时年薪只有 $23,500，就萌生了创办 Kensho 的想法
- 核心 insight：他在 Kensho 做的是 **"organizing financial information"**，而 OpenEvidence 做的是 **"organizing medical information"** — 同一个 paradigm 的 domain transfer

### Zachary Ziegler
- Co-founder

参考: [Forbes Profile](https://www.forbes.com/profile/daniel-nadler/) | [Wikipedia - Daniel Nadler](https://en.wikipedia.org/wiki/Daniel_Nadler)

---

## 三、核心技术架构（第一性原理解析）

OpenEvidence 的技术本质上是一个 **domain-specific RAG (Retrieval-Augmented Generation) system**，但加了很多 medical domain 的特殊设计。让我从第一性原理拆解：

### 3.1 RAG 架构的基本公式

标准 RAG 的输出可以形式化为：

$$\text{Answer} = \text{LLM}(q, \; \text{Retrieve}(q, \mathcal{D}))$$

其中：
- $q$ = physician 输入的 natural language clinical question
- $\mathcal{D}$ = 知识库（这里是 peer-reviewed medical literature corpus）
- $\text{Retrieve}(q, \mathcal{D})$ = retrieval function，从 $\mathcal{D}$ 中找出与 $q$ 最相关的 top-$k$ passages
- $\text{LLM}(\cdot)$ = 大语言模型，基于 retrieved context 生成 answer

### 3.2 Knowledge Corpus $\mathcal{D}$（数据护城河）

这是 OpenEvidence 最核心的 **competitive moat**。他们签了一系列 **multi-year content agreements**，包括：

| 合作方 | 类型 |
|---|---|
| **NEJM** (New England Journal of Medicine) | **Official AI Partner** — 30 年 archives |
| **JAMA** (Journal of the American Medical Association) | Content agreement |
| **NCCN** (National Comprehensive Cancer Network) | Clinical guidelines |
| **Wiley** | Publisher agreement |
| **Cochrane** | Systematic reviews |

这意味着他们的 $\mathcal{D}$ 不是 random web scraping，而是 **curated, peer-reviewed, authoritative medical literature**。这种 **licensed content** 是一般 startup 很难复制的。

参考: [OpenEvidence x NEJM](https://www.openevidence.com/announcements/openevidence-and-nejm) | [Medical Economics](https://www.medicaleconomics.com/view/clinical-ai-platform-to-be-trained-on-30-years-of-nejm-archives)

### 3.3 Retrieval 层

虽然 OpenEvidence 没有公开完整的 retrieval architecture，但根据 medical RAG 的 best practices，大概率使用的是 **hybrid retrieval**：

$$\text{Score}(q, d) = \alpha \cdot \text{BM25}(q, d) + (1-\alpha) \cdot \text{cos}(\mathbf{e}_q, \mathbf{e}_d)$$

其中：
- $\text{BM25}(q, d)$ = 传统的 sparse retrieval（基于 term frequency），对 medical terminology 的精确匹配很重要
- $\text{cos}(\mathbf{e}_q, \mathbf{e}_d)$ = dense retrieval，$\mathbf{e}_q$ 和 $\mathbf{e}_d$ 分别是 query 和 document 的 embedding vectors
- $\alpha$ = 混合权重 hyperparameter

在 medical domain 里，BM25 特别重要，因为 drug names、gene names、ICD codes 这些 **specialized tokens** 需要 exact matching。

### 3.4 LLM Generation 层

OpenEvidence 大概率使用的是 **fine-tuned LLM**（可能基于 foundation models 如 GPT-4 或自研模型），并加入了：

1. **Medical Chain-of-Thought (CoT) Reasoning**：让模型模拟临床推理过程
   - Step 1: Identify the clinical question type (diagnosis / treatment / prognosis)
   - Step 2: Gather relevant patient factors
   - Step 3: Apply clinical reasoning frameworks
   - Step 4: Synthesize evidence and generate recommendation

2. **Citation Grounding**：每个回答都附带 **direct citations** 到原始 peer-reviewed papers，这解决了 LLM 的 **hallucination** 问题——或者说至少让 physician 可以 **verify**

3. **Explanation Model**：他们还开发了一个专门的 reasoning/explanation model，用于解释临床推理过程。这个模型对 medical students 免费开放。

### 3.5 USMLE 100% 的里程碑

2025 年，OpenEvidence 宣布其 AI 系统成为 **历史上第一个在 USMLE (United States Medical Licensing Examination) 上取得 100% 满分** 的 AI。

但需要注意：
- 这是一个 **experimental methodology**
- 批评者（如 SynthioLabs）指出这可能是 **多次尝试后的最佳结果**，并非 single-pass consistency
- 不过这仍然展示了其 medical reasoning capability 的上限

参考: [OpenEvidence USMLE 公告](https://www.openevidence.com/announcements/openevidence-creates-the-first-ai-in-history-to-score-a-perfect-100percent-on-the-united-states-medical-licensing-examination-usmle) | [SynthioLabs 批评](https://synthiolabs.com/blog/the-real-story-behind-openevidences-100-on-usmle)

---

## 四、融资历程

| Round | 金额 | Valuation | Lead Investors |
|---|---|---|---|
| Seed / Early | — | — | — |
| Series B | — | ~$1B | Sequoia Capital |
| Series C | $200M | $6B | — |
| Series D | $250M | **$12B** | **Thrive Capital + DST Global** |

其他知名投资者：**Sequoia, Google Ventures (GV), NVIDIA, Kleiner Perkins, Blackstone**

参考: [Fierce Healthcare Series D](https://www.fiercehealthcare.com/ai-and-machine-learning/openevidence-clinches-250m-series-d-rapidly-growing-its-reach-doctors) | [Sequoia Capital](https://sequoiacap.com/companies/openevidence/)

---

## 五、商业模式

1. **对 Physicians 免费 + 无限制使用** — 这是一个经典的 **land-and-grab** 策略，先占领 physician mindshare
2. **对 Pharma / Life Sciences 公司收费** — 大概率通过 advertising, sponsored content, data insights 变现
3. **Enterprise / Hospital System Integration** — B2B SaaS 模式
4. **90% gross margin** 说明这是一个典型的 **software-margin business**，不是 hardware-intensive

---

## 六、从第一性原理看 OpenEvidence 的护城河

用 **Peter Thiel 的 monopoly framework** 分析：

| 护城河类型 | OpenEvidence 的表现 |
|---|---|
| **Proprietary Technology** | Fine-tuned medical LLM + RAG pipeline，USMLE 100% |
| **Network Effects** | 40% US physician adoption → 更多 usage data → 更好的模型 → 更多 physicians 使用 |
| **Economies of Scale** | 90% gross margin，边际成本趋近于零 |
| **Brand / Switching Costs** | NEJM official partner 背书，physician workflow integration |
| **Data Moat** | Licensed content from NEJM, JAMA, NCCN, Cochrane + physician interaction data |

最关键的 insight：**medical information 是一个 high-stakes domain**，physician 不会随便信任一个 random AI tool。OpenEvidence 通过 **NEJM 背书 + citation grounding + 高准确率** 建立了 **trust**，而 trust 在 healthcare 中就是最大的 switching cost。

---

## 七、类比与联想

- **Kensho → S&P Global**：Daniel Nadler 之前做的是 "AI for financial information"，S&P Global 收购了它。现在他在做 "AI for medical information"。如果类比，潜在 acquirer 可能是 **Epic Systems, UnitedHealth, Google Health** 等。
- **Google 的定位**：OpenEvidence 本质上想成为 **"the Google of Healthcare"** — 但比 Google 更 vertical、更 authoritative、更 grounded。
- **与 UpToDate 的竞争**：Wolters Kluwer 的 **UpToDate** 长期以来是 physicians 的 go-to clinical decision support tool。OpenEvidence 本质上是 UpToDate 的 AI-native 替代品。
- **与 Perplexity 的类比**：如果说 **Perplexity** 是 general-purpose 的 "answer engine"，那 OpenEvidence 就是 **medical-vertical Perplexity**。

参考: [GV (Google Ventures) 报道](https://www.gv.com/news/openevidence-ai-doctors) | [Contrary Research 分析](https://research.contrary.com/company/openevidence)

---

**总结**：OpenEvidence 是一个从 **2022 年 → 2026 年初** 在短短 ~3 年内从零成长到 **$12B valuation** 的 AI healthcare unicorn，其核心是将 **RAG + fine-tuned LLM + licensed authoritative medical content** 组合成一个 physician-facing clinical decision support platform，以 **"free for doctors"** 的策略快速抢占市场，再通过 B2B/pharma 变现。Founder Daniel Nadler 的连续创业经验（Kensho → OpenEvidence）和顶级 VC 阵容（Sequoia, GV, NVIDIA, Thrive, DST）让这家公司成为 AI healthcare 领域最值得关注的标的之一。