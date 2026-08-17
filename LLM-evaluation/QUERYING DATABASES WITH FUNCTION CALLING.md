---
source_pdf: QUERYING DATABASES WITH FUNCTION CALLING.pdf
paper_sha256: d7d145807ed190577c5b1df7d309352fb6463bea2ebe5e2fcc0de0b253133acf
processed_at: '2026-08-06T07:59:04-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

Andrej, 我用大白话再讲一遍。

---

## 这篇 paper 在干嘛

Weaviate 团队想解决一个问题: 让 LLM 查数据库。

传统做法是 text-to-SQL, 也就是让 LLM 生成一段 SQL 字符串, 然后扔给数据库执行。问题是 SQL 方言太多, Postgres 一套写法, MySQL 一套写法, Snowflake 又一套, LLM 学不过来, 而且生成的 SQL 你也没法在生成阶段就约束它语法对不对。

所以他们换了个思路: 别让 LLM 写 SQL 了, 给它一个 function 叫 `query_database`, 这个 function 有一堆 optional 的 JSON 参数, 每个参数对应一个 query operator。LLM 只需要填这些结构化参数, 不需要写 SQL 字符串。

打个比方, 以前是让 LLM 写一整篇作文, 现在是给它一个表格让它填空。表格的每个格子都有规定的填法, 你填错了 schema 层面就挡住了。

---

## 这个 function 长什么样

核心就一个 function, 叫 `query_database`, 参数大概是这样:

- `collection_name`: 必填, 你要查哪个 collection, 是个 enum, LLM 只能从列表里选
- `search_query`: 可选, 语义搜索的 query text
- `integer_property_filter`: 可选, 数字过滤, 比如 price < 20
- `text_property_filter`: 可选, 文本精确匹配, 比如 cuisine = "Italian"
- `boolean_property_filter`: 可选, 布尔过滤, 比如 isOpen = true
- `integer_property_aggregation`: 可选, 数字聚合, 比如 MEAN, SUM, COUNT
- `text_property_aggregation`: 可选, 文本聚合, 比如 TOP_OCCURRENCES
- `boolean_property_aggregation`: 可选, 布尔聚合, 比如 PERCENTAGE_TRUE
- `groupby_property`: 可选, 按某个字段 group by

举例子, 用户问 "20 块以下的素菜平均价格是多少, 按是否素食分组", LLM 就填:
- collection_name = "Menus"
- integer_property_filter = {property: "price", operator: "<", value: 20}
- integer_property_aggregation = {property: "price", metric: "MEAN"}
- groupby_property = "isVegetarian"

完事。不用写 SQL, 不用担心方言, 每个 argument 都有 type constraint 兜底。

---

## Benchmark 怎么造的

他们造了个叫 DBGorilla 的 benchmark, 315 条 query。

流程:
1. 用 GPT-4o 生成 5 个 fake 业务场景 (餐厅, 诊所, 课程, 旅行, 视觉艺术), 每个场景 3 个 collection, 每个 collection 4 个 property (2 text + 1 number + 1 boolean)
2. 穷举所有 operator 组合, 每个组合生成一条 natural language query, 一共 63 条 per schema
3. 用 Reflexion 让另一个 LLM 检查生成的 query 质量这个 63 就是 operator 的排列组合数, 大概是 2 的 6 次方级别, 把每个 optional operator 用/不用都覆盖到。

跟 BIRD, Spider 这些主流 text-to-SQL benchmark 比, DBGorilla 的 query 数量少很多 (315 vs 12000+), 但胜在 operator 覆盖完整, 能精确定位 LLM 在哪个 operator 上翻车。

---

## 实验结果

8 个模型跑下来:

- **Claude 3.5 Sonnet** 最强, 74.3% exact match
- **GPT-4o mini** 第二, 73.7%, 而且只花 $0.12, 性价比炸裂
- **GPT-4o** 第三, 71.8%
- **Gemini 1.5 Pro** 第四, 70.2%
- **Command R+** 掉到 59.4%
- **Gemini 2.0 Flash** 和 **Llama 3.1 8B** 都在 30% 左右

两个有意思的点:

1. GPT-4o mini 居然比 GPT-4o 还高一点。小模型反超大模型, 可能是 4o mini 训练数据里 function calling 的 instruction tuning 更密集, 或者 4o 在简单 query 上太强反而复杂 query 上松了。

2. Gemini 2.0 Flash 有 53.97% 的时候根本不调 function, 直接瞎编答案。这个 failure mode 以前没怎么被人讨论, 其实挺重要的。

---

## 按组件拆解的发现

把 performance 按 operator 类型拆开看:

- **Boolean filter 最容易**, 所有 top model 都在 86-87%, 因为 boolean 就是 true/false, 没歧义
- **Text filter 最难**, 最好的也就 46%, 因为 LLM 分不清 "找 Italian restaurant" 到底该用 search_query (语义搜索) 还是 text_property_filter on cuisine field (精确匹配)。这俩在语义上几乎等价, 连人都未必分得清, 何况 LLM
- **Aggregation 居中**, 60-77%, boolean aggregation 比 boolean filter 难一些

所以结论是: 离散约束强的 operator (boolean) LLM 搞得定, 语义模糊的 operator (text) LLM 搞不定。这不意外, 但量化出来还是有价值的。

---

## Ablation 实验

他们还试了四个变体, 全部用 GPT-4o:

1. 加 rationale (让 LLM 解释为什么这么调): 73.2%, 涨了 1.4%
2. 开 parallel tool calls (允许同时调多个 function): 71.2%, 跌了 0.6%
3. 每个 collection 一个独立 tool (而不是统一一个): 72.3%, 涨了 0.5%
4. 用 structured generation 替代 Function Calling SDK: 72.8%, 涨了 1.0%

**全部在 2% 以内波动, 基本就是 noise**。

这说明什么? 说明真正重要的是 tool definition schema 本身设计得好不好, 至于你用 Function Calling 还是 structured generation, 加不加 rationale, 并不 care。这是个 negative result, 但很有信息量。

---

## 成本

跑一次全部 8 个模型的 benchmark 花费 $8.10, 一年 $97。非常便宜, 社区可以反复跑。

GPT-4o mini 是 pareto optimal: 73.7% accuracy 只要 $0.12, 比 Claude 3.5 Sonnet 便宜 23 倍, 只低 0.6 个点。

---

## 一句话总结

把数据库查询从 "让 LLM 写 SQL 字符串" 改成 "让 LLM 填结构化 JSON 参数", 大模型能干到 70%+ 的 exact match, boolean 操作容易 text 操作难, 大部分 ablation 都不影响结果, GPT-4o mini 性价比最高。

核心 insight 就是 **decomposition**: 把一个 unconstrained 的生成任务 (写 SQL) 拆成多个 constrained 的填空任务 (填 operator 参数), LLM 表现会好很多, 你也更容易验证和兜底。

---

# QUERYING DATABASES WITH FUNCTION CALLING - 深度解读

Andrej，这篇 Weaviate 团队的 paper 在 Compound AI Systems 这个 emerging area 里做了一个非常 specific 的 contribution：把 database querying 从传统的 text-to-SQL 范式 **重新 decompose** 成 Function Calling 的 structured JSON arguments。让我从 motivation、methodology、experimental design、ablation、以及 future directions 多个维度展开，并尽可能联系到相关 work 帮助你 build intuition。

---

## 1. Motivation 与 Positioning

### 1.1 The Core Problem

传统 text-to-SQL 路径的核心问题在于：SQL dialect 碎片化。Spider 2.0 (Lei et al., 2024) 明确指出 "real-world data are stored across a diverse array of database systems, each with its own unique SQL dialects"。这意味着 LLM 不仅要学习 query semantics，还要学习特定 dialect 的 surface syntax (e.g., Postgres vs MySQL vs Snowflake 的 `LIMIT/OFFSET` vs `FETCH FIRST` vs `TOP`)。

Weaviate 团队提出的核心思路：**把 query operators 从 SQL syntax 中 disentangle 出来**。Figure 2 展示了从 "How many menu items are priced under 20?" 这种 natural language command 直接翻译成 Function Calling 的 `integer_property_filter` argument，跳过 SQL string 这个中间表示。

这与近期几条线对 SQL 的重新思考形成呼应：
- **LOTUS** (Patel et al., 2024): semantic operators over text data
- **TAG** (Biswal et al., 2024): unifying AI and databases
- **SUQL** (Liu et al., 2024): conversational search over structured + unstructured data

paper 也引用了 **STaRK** (Wu et al., 2024) 的观察："many previous works studied textual and relational retrieval tasks as separate topics"。这种 isolation 的根源在于 underlying data model 不同 (relational vs document/search index)，而 Weaviate 作为 vector + structured database，天然处于 unified 的位置。

### 1.2 Why Function Calling vs. a single SQL string argument?

这里有一个非常 subtle 但重要的设计选择。OpenAI 的 Function Calling 当前 **不支持 advanced constraints on input arguments**，比如你无法约束一个 `sql_query` string argument 必须遵循特定 dialect 的语法。所以如果你让 LLM 生成 SQL string，所有 validation 都得 post-hoc 在 application 端做。

By decomposing query API into 8 个 optional JSON arguments (search_query, integer/text/boolean filter, integer/text/boolean aggregation, groupby_property)，你把 constraint enforcement 推到了 schema 定义阶段。每个 argument 有明确的 type 和 enum constraint (e.g., `integer_property_filter.operator ∈ {=, <, >, <=, >=}`)。LLM 在结构化输出的层面就被 forced 进合法空间。

这个思路本质上跟 **structured generation** (Willard & Louf, 2023; Shorten et al., 2024 StructuredRAG) 一脉相承，只是从输出 format 上升到了 tool definition 层面。

---

## 2. DBGorilla Benchmark 设计

### 2.1 Dataset Construction Pipeline

整个 benchmark 生成 pipeline 是 Self-Instruct (Wang et al., 2023) + Reflexion (Shinn et al., 2023) + structured generation 的组合：

**Phase 1: Synthetic Schema Generation**
- 用 GPT-4o 生成 5 个 use case
- 每个 use case 包含 3 个 interrelated collections (no explicit foreign keys)
- 每个 collection 包含 4 个 properties: 2 text (其中 1 个是 searchable rich content), 1 numeric, 1 boolean
- Table 1 展示了 Restaurant use case: Restaurants / Menus / Reservations

为什么选这种 property distribution？因为这是一个 minimal 完备集合，能覆盖所有 query operator 类型。Paper 在 Section 6.1 Discussion 里提到未来可以扩展到 8 boolean + 1 searchable text，或者引入 foreign keys。

**Phase 2: Synthetic Query Generation**
- 对每个 schema 生成 63 个 query，覆盖所有 operator 的 valid combinations
- 用 Pydantic model on-the-fly 为每个 operator combination 创建 structured output validator
- 5 schemas × 63 queries = 315 queries total

63 这个数字怎么来的？它近似是所有 operator subset 的笛卡尔积：每个 operator 有 "使用/不使用" 二元选择，但要排除全空集和只有 collection_name 的情况。粗略地说：$2^6 = 64$ (6 个 optional operators) 减去 trivial case。

### 2.2 为什么这个 benchmark 设计有趣？

对比三个主流 text-to-SQL benchmark：

| Benchmark | Examples | DBs | Domains | Characteristics |
|-----------|----------|-----|---------|-----------------|
| WikiSQL | 80,654 | 24,241 tables | Wikipedia | Single table, simple SQL |
| Spider | 10,181 | 200 DBs | 138 domains | Multi-table, complex SQL |
| BIRD | 12,751 | 95 DBs | 37 domains | Real-world professional DBs |
| **DBGorilla** | 315 | 5 use cases × 3 collections | 5 synthetic domains | Function Calling, operator combinatorics |

DBGorilla 的特点在于 **combinatorial coverage**：63 queries = 63 distinct operator combinations，这是非常 surgical 的 evaluation，能精准定位 LLM 在哪个 operator 上出错。代价是 query 数量小，且全是 synthetic (可能 LLM 容易 overfit 这种 stylized phrasing)。

---

## 3. Tool Definition Schema 解析

Appendix A 的 tool schema 是整个 paper 的核心 artifact。让我逐个 argument 解析：

### 3.1 Required Argument

```
collection_name: string, enum=[list of available collections]
```

唯一 required argument。`enum` constraint 把 collection routing 问题转化成 multiple choice，这是为什么所有模型的 collection routing accuracy 都很高 (96-98%)。

### 3.2 Optional Arguments

**`search_query: string`** — 用于 semantic search / BM25 search，返回 ranked results from search index。没有 constraint，LLM 自由生成 query text。

**`integer_property_filter: object`** — 包含三个 sub-fields:
- `property_name: string`
- `operator: enum[=, <, >, <=, >=]`
- `value: number`

**`text_property_filter: object`** — `operator: enum[=, LIKE]`，注意只有 equality 和 LIKE，没有 full-text search operator (那是 search_query 的职责)。

**`boolean_property_filter: object`** — `operator: enum[=, !=]`，最简单的 filter，这也解释了为什么 boolean filter performance 最高 (87.5%)。

**`integer_property_aggregation: object`** — `metrics: enum[COUNT, TYPE, MIN, MAX, MEAN, MEDIAN, MODE, SUM]`。这里的 TYPE 意思是 "distinct type count"，类似 SQL 的 `COUNT(DISTINCT)`。

**`text_property_aggregation: object`** — `metrics: enum[COUNT, TYPE, TOP_OCCURRENCES]` + optional `top_occurrences_limit: integer`。TOP_OCCURRENCES 类似 `SELECT property, COUNT(*) ... ORDER BY COUNT(*) DESC LIMIT N`。

**`boolean_property_aggregation: object`** — `metrics: enum[COUNT, TYPE, TOTAL_TRUE, TOTAL_FALSE, PERCENTAGE_TRUE, PERCENTAGE_FALSE]`。这是 boolean 特有的 aggregation，把 SQL 里需要 `SUM(CASE WHEN ... THEN 1 ELSE 0 END)` 这种 verbose 写法封装了。

**`groupby_property: string`** — 单一 property，没有 multi-property groupby。这是一个 limitation，paper 没明确讨论。

### 3.3 Schema Design 的 Implicit Tradeoffs

这个 schema 有几个 implicit 的设计决策值得注意：

1. **没有 JOIN operator**。Paper 明确说 "we do not explicitly link these collections together with foreign key relationships"。这意味着 cross-collection query 必须通过 parallel tool calls (Section 5.2 的 ablation) 或者 multi-turn interaction 处理。

2. **Filter / Aggregation 是单数的**。一个 query 只能有一个 integer filter、一个 text filter，无法表达 `price < 20 AND price > 5`。

3. **没有 ORDER BY / LIMIT**。Search ranking 由 search index 处理，aggregation 结果隐含 grouped。

4. **Search query 和 text filter 的 semantic 边界很模糊**。这是 paper 自己承认的核心 finding：LLM 难以区分 "find Italian restaurants" 应该是 search_query 还是 text_property_filter on cuisine field。

---

## 4. Evaluation Metrics

### 4.1 Exact Match

Boolean metric，predicted query 与 ground truth 完全一致才计 1。

$$\text{EM}(q) = \mathbb{1}[\hat{q} = q^*]$$

其中 $\hat{q}$ 是 predicted query，$q^*$ 是 ground truth。这是 primary metric，因为 synthetic benchmark 的 ground truth 是 deterministic 的。

### 4.2 Abstract Syntax Tree (AST) Score

paper 自创的 partial credit metric：

$$\text{AST Score} = 0.4 \cdot \mathbb{1}[\hat{c} = c^*] + 0.15 \cdot \mathbb{1}[\hat{s} \leftrightarrow s^*] + 0.15 \cdot \mathbb{1}[\hat{f} = f^*] + 0.15 \cdot \mathbb{1}[\hat{a} = a^*] + 0.15 \cdot \mathbb{1}[\hat{g} = g^*]$$

变量含义：
- $\hat{c}, c^*$: predicted / ground truth **collection** (40% weight)
- $\hat{s}, s^*$: **search query** 使用情况 (15%) — 只判断是否使用，不比较 content
- $\hat{f}, f^*$: **filter** 完全匹配 (15%)
- $\hat{a}, a^*$: **aggregation** 完全匹配 (15%)
- $\hat{g}, g^*$: **groupby** 完全匹配 (15%)

注意一个重要细节：**collection 错则整个 query 0 分**（because mismatching collections results in a score of 0）。这解释了为什么所有模型 AST score 都很高 (0.89-0.97)：只要 collection routing 对了就有 0.4 分打底。

search query 的 "只看是否使用，不看 content" 这个设计很 conservative，避免了对 semantic similarity 的主观判断，但也低估了 search query quality 的影响。

### 4.3 LLM-as-Judge Preference Ranking

让一个 LLM 对 8 个模型的输出排名 1-8。Weighted scoring:

$$\text{Weighted Score} = \sum_{i=1}^{N} w_{r_i}$$

权重：$w_1 = 100, w_2 = 70, w_3 = 50, w_4 = 35, w_5 = 25, w_6 = 20, w_7 = 15, w_8 = 10$

这是一个 convex 下降的 weight curve，强烈奖励 top placement。

**有趣发现**：LLM-as-judge 排名跟 AST 排名不完全一致。Gemini 1.5 Pro 在 preference ranking 第一 (20.6% first place votes)，但在 AST score 排第五。Paper 将这解读为 "technical correctness vs. user preference" 的差异，但更可能的解释是 LLM judge 本身有 bias，倾向于 readable / clear 的 query 而非严格正确的。

这与 DPO (Rafailov et al., 2024) 和 anchored preference optimization (D'Oosterlinck et al., 2024) 的工作联系起来：preference signal 可以用于 fine-tune 更好的 query generation model。

### 4.4 Collection Routing & No Tool Selected Rate

- Collection Routing: predicted collection 是否正确
- No Tool Selected Rate: LLM 直接回答而不调用 function 的比例

后者是 unexpected 的 finding。Gemini 2.0 Flash 高达 53.97% 不调用 function，这意味着它把 benchmark 当成普通 QA 任务，直接 hallucinate 答案。这是 Function Calling evaluation 里一个 under-discussed 的 failure mode。

---

## 5. Experimental Results 深度分析

### 5.1 Overall Leaderboard (Figure 1)

| Model | EM | Simple | Moderate | Complex | AST | Routing |
|-------|-----|--------|----------|---------|------|---------|
| Claude 3.5 Sonnet | **74.3%** | 77.5% | 76.4% | 72.1% | **0.973** | 96.9% |
| GPT-4o mini | 73.7% | 80.0% | 69.1% | 75.2% | 0.952 | **97.5%** |
| GPT-4o | 71.8% | **87.5%** | 70.0% | 69.1% | 0.966 | 96.8% |
| Gemini 1.5 Pro | 70.2% | 67.5% | 63.6% | 75.2% | 0.956 | 93.0% |
| Command R+ | 59.4% | 72.5% | 68.2% | 50.3% | 0.933 | 94.3% |
| Command R7B | 39.1% | 55.0% | 44.6% | 31.5% | 0.887 | 84.4% |
| Gemini 2.0 Flash | 37.1% | 35.0% | 36.4% | 38.2% | 0.449 | 46.0% |
| Llama 3.1 8B Instruct | 32.1% | 27.5% | 26.4% | 37.0% | 0.894 | 76.8% |

几个有意思的 patterns：

1. **GPT-4o mini 反超 GPT-4o** (73.7% vs 71.8%)。这是非常反直觉的，因为 4o mini 是更小更便宜的模型。可能的解释：4o mini 在 4o 之后训练，吸收了更多 function calling 相关的 instruction tuning 数据；或者 4o 在 Simple Queries 上表现太好 (87.5%) 反而在 Complex 上略松 (69.1%)。这种 "smaller model beats larger on specific task" 的现象在 LLM 评估中越来越常见，让人想起 Llama 3.1 8B 在某些 task 上接近早期 GPT-4 的故事。

2. **Gemini 2.0 Flash 的异常**：AST score 0.449 极低 (其他都 0.88+)，且 No Tool Selected Rate 53.97%。这说明它根本没进入 Function Calling 模式。这个实验 version 可能 system prompt 没正确触发 tool use behavior。Paper 没深入诊断这个 issue。

3. **Complex query 反超 Simple**: GPT-4o mini 和 Gemini 1.5 Pro 在 Complex Queries (3+ args) 上反而比 Moderate 高。一种可能：3-argument query 的结构更 constrained，减少了 LLM 的自由度；另一种可能：sample size 在每个 bin 内很小 (315/3 ≈ 105)，variance 大。

### 5.2 Component-Level Analysis (Tables 8, 9)

这是 paper 最 informative 的分析。把 performance 按 API component 拆解：

| Component | GPT-4o | Claude 3.5 | Gemini 1.5 Pro | Llama 3.1 8B |
|-----------|--------|------------|----------------|---------------|
| Search Queries | 78.75% | 83.75% | 81.25% | 52.50% |
| Integer Filters | 71.25% | 73.75% | 82.50% | 26.25% |
| **Text Filters** | **37.50%** | **46.25%** | **41.25%** | **27.50%** |
| **Boolean Filters** | **87.50%** | **87.50%** | **86.25%** | **32.50%** |
| Integer Aggregations | 73.75% | 73.75% | 77.50% | 32.50% |
| Text Aggregations | 70.00% | 73.75% | 70.00% | 30.00% |
| Boolean Aggregations | 62.50% | 66.25% | 52.50% | 30.00% |
| GroupBy Operations | 71.70% | 72.96% | 72.33% | 23.27% |

**核心 finding**: Boolean filter 是最容易的 (~87%)，text filter 是最难的 (<50%)。

为什么 text filter 这么难？Paper 的解释：LLM 难以区分 "找 Italian restaurants" 应该用 `search_query` (semantic search) 还是 `text_property_filter` with `cuisine = "Italian"`。两者在语义上几乎等价，区别只在 underlying index (BM25 vs exact match)。

这其实是一个 fundamental ambiguity，而不是 LLM 能力不足。从 product 角度看，可能应该让 search_query 和 text_property_filter 自动 fallback 或者合并。从 benchmark 角度看，这说明 ground truth 本身可能 over-specified。

### 5.3 Cross-Schema Variance (Table 3)

| Model | Restaurants | Health Clinics | Courses | Travel | Visual Art | Range |
|-------|-------------|----------------|---------|--------|------------|-------|
| GPT-4o | 73.44% | 76.56% | 70.31% | 70.31% | 67.80% | **5.64%** |
| Claude 3.5 Sonnet | 71.88% | 73.44% | 71.88% | 71.88% | 83.05% | 11.17% |
| Gemini 2.0 Flash | 57.81% | 23.44% | 35.94% | 25.00% | 44.07% | **34.37%** |

**Finding**: Model size 与 cross-domain consistency 强正相关。GPT-4o range 仅 5.64%，而 Gemini 2.0 Flash 高达 34.37%。这印证了 larger model 有更好的 in-context schema understanding，能 generalize 到任意 domain。

---

## 6. Ablation Studies (Section 5)

四个 ablation 都用 GPT-4o，结果如下：

| Experiment | EM | Routing |
|------------|-----|---------|
| Original | 71.8% | 96.5% |
| + Tool Rationale | 73.2% | 96.8% |
| + Parallel Tool Calls | 71.2% | 95.9% |
| One Tool per Collection | 72.3% | 96.8% |
| Structured Generation | 72.8% | 97.1% |

**全部 ablation 在 2% 以内 variance**。这是一个很强的 negative result：这些常见的 "best practice" 都没有显著影响。

### 6.1 Tool Rationale (+1.4%)

要求 LLM 在 tool call 时输出 rationale string。Paper 给了一个很有启发性的例子：

- Query: "How many different types of exhibit highlights are featured in each museum, grouped by museum name?"
- Ground truth: `TextAggregation(exhibitHighlights: COUNT)`
- Predicted: `TextAggregation(exhibitHighlights: TYPE)`
- Rationale: "I will perform a frequency analysis on the 'exhibitHighlights' property"

Rationale 暴露了模型的 misconception：它把 TYPE aggregation 理解为 "frequency analysis"。这种 interpretability 是 rationale 的主要价值，performance 提升是次要的。

### 6.2 Parallel Tool Calls (-0.6%)

启用后 GPT-4o 平均每次 query 调用 1.21 个 tool。观察到的 pattern 是调用 complementary collections (e.g., Restaurants + Reservations)。这暗示 LLM 在尝试 cross-collection 推理，但 benchmark 评估只取最高分 tool call。

未来 Compound AI System 可能需要更 sophisticated 的 multi-tool evaluation，比如 task completion 而非 single call accuracy。

### 6.3 One Tool per Collection (+0.5%)

把单一 `query_database` 拆成 `query_restaurants`, `query_menus`, `query_reservations` 三个独立 tool。性能基本持平。

**Practical implication**: 当 schema 变大、collection 变多时，单 tool description 会 hit token limit (paper 注意到 1024 token constraint)。One-tool-per-collection 是 scalable 方案。更进一步，可以做 **dynamic tool selection**：用 first-pass LLM inference 从 NL command 提取 keywords，只把相关 collection 的 tool 暴露给 second-pass。

### 6.4 Structured Generation (+1.0%)

不用 Function Calling SDK，改用 Pydantic-based structured generation (ResponseOrFunctionCall model in Appendix A)。性能持平。

**意义**: LLM 没有强烈 bias 偏好 native Function Calling SDK。这给 open-source 部署提供了灵活性，可以避免 vendor lock-in。

---

## 7. Cost Analysis (Section 3.4)

| Model | Total Cost |
|-------|-----------|
| Claude 3.5 Sonnet | $2.84 |
| GPT-4o | $2.00 |
| Gemini 1.5 Pro | $1.01 |
| GPT-4o mini | $0.12 |
| Llama 3.1 8B | $0.04 |
| Command R7B | $0.03 |

Benchmark 维护成本：$8.10/run × 12 months = ~$97.20/year。这是个非常 cheap 的 benchmark，社区可以 frequent replicate。

Dataset 生成成本：$1.89 with GPT-4o (413K input + 86K output tokens)。

**Pareto frontier**: GPT-4o mini 是明显的 sweet spot — 73.7% EM at $0.12，比 Claude 3.5 Sonnet (74.3% at $2.84) 便宜 23x，仅低 0.6%。

---

## 8. Connections to Broader Research

### 8.1 Compound AI Systems 谱系

Paper 用 Zaharia et al. (2024) 的 "Compound AI Systems" framing 定位。这个谱系包括：

- **ReAct** (Yao et al., 2023): reasoning + acting 交替
- **Gorilla LLM** (Patil et al., 2023): API calling benchmark，DBGorilla 直接基于此
- **DSPy** (Khattab et al., 2022, 2023): declarative LLM pipeline programming
- **Reflexion** (Shinn et al., 2023): verbal reinforcement learning
- **Network of Networks** (Davis et al., 2024): complexity class principles for compound systems
- **Specifications** (Stoica et al., 2024): missing link for engineering discipline

Paper 的 contribution 在这个谱系里处于 "Function Calling for database access" 这个具体 niche，跟 DSPy Assertions (Singhvi et al., 2024) 和 SPADE (Shankar et al., 2024) 互补 — 后者关注 pipeline validation，前者关注 tool definition。

### 8.2 Function Calling vs. Flow Engineering

Paper Section 2.1 区分了两种 tool use 设计模式：

- **Function Calling**: LLM 在 loop 中自主决定调用什么 tool
- **Flow Engineering**: 预定义 inference + tool call 序列 (e.g., AlphaCodium, Ridnik et al., 2024)

RAG (Lewis et al., 2020) 是经典 Flow Engineering。Baleen RAG 引入 LLM-generated query reformulation。WebGPT (Nakano et al., 2022) 是早期把 search 作为 Function Calling 的工作。Zhang et al. (2024) 提出 "Agentic Information Retrieval" 概念统合这些。

DBGorilla 测试的是 Function Calling 模式下的 single-step accuracy。Multi-step 场景下，两种模式可能 converge：Flow Engineering 可以抽象成一个 Function 给 agent 调用，Function Calling loop 也可以 wrap 成 pre-determined flow。

### 8.3 Text-to-SQL Evolution

| Generation | Benchmark | Approach |
|------------|-----------|----------|
| 1st | WikiSQL (2017) | Single table, simple SQL |
| 2nd | Spider (2018) | Multi-table, complex SQL, cross-domain |
| 3rd | BIRD (2024) | Real-world DBs, domain knowledge |
| 4th | Spider 2.0 (2024) | SQL workflows, multiple dialects |
| **Parallel** | **DBGorilla** | **Function Calling, operator combinatorics** |

Spider 2.0 的 key shift 是从 "SQL prediction" 到 "SQL workflows"，承认 real-world 是 multi-step 的。DBGorilla 走的是另一条路：decompose SQL into structured operators，避免 dialect 问题但保持 single-step evaluation。

未来可能 converge：把 DBGorilla 的 structured operators 嵌入 Spider 2.0 风格的 multi-step workflow evaluation。

### 8.4 与 SPARQL / Knowledge Graphs 的联系

Paper 提到 tool definition "can be trivially mapped to and from custom SQL dialects or extended with functionality from less conventional data models such as SPARQL"。这与 Weaviate 团队之前的 work (van Luijt & Verhagen, 2020, IEEE Software) 一脉相承，他们长期推动 semantic knowledge graph 技术。

SPARQL 的 graph pattern matching 可以作为未来 tool extension。比如增加 `graph_pattern` argument，让 LLM 表达 subject-predicate-object triple patterns。

---

## 9. Limitations 与 Future Directions

### 9.1 明确的 Limitations

1. **Single-step only**: 真实 query 经常需要 iterative refinement
2. **No JOIN**: collection 间没有 foreign key
3. **Single filter per type**: 无法表达复合 boolean 条件
4. **No ORDER BY / LIMIT**: 排序完全依赖 search ranking
5. **Single groupby property**: 无法 multi-column groupby
6. **Synthetic queries**: 可能 stylized，real NL distribution 更 noisy
7. **315 queries**: 统计 power 有限
8. **Search query evaluation宽松**: 只看是否使用，不看 content quality

### 9.2 Future Work 提到的方向

1. **Database Gyms** (Lim et al.): 增加 schema complexity (foreign keys, more collections, varied property distributions)
2. **Multi-hop queries**: 一个 query 结果 inform 下一个
3. **Iterative querying**: 结果反馈触发 refinement (联系 Reflexion, DSPy Assertions, SPADE)
4. **Broader function set**: 加入 web search, data viz, analytics platforms (联系 GoEX, Patil et al., 2024)
5. **Prompt optimization**: MIPRO (Opsahl-Ong et al., 2024) 或 AvaTaR (Wu et al., 2024) 自动优化 tool description
6. **Preference optimization**: 用 LLM-as-judge signal 做 DPO 训练

### 9.3 My Additional Speculations

1. **Tool description as prompt engineering**: paper 没探索 tool description 本身的 phrasing 对 performance 的影响。`"Filter text properties using equality or LIKE operators"` vs `"Match exact text or use pattern matching"` 可能差几个点。

2. **Schema pruning**: 当 collection 多时，先做 lightweight retrieval 选 top-k relevant collections，再 expose 给 LLM。这是 retrieval-augmented function calling。

3. **Constitutional AI for query**: 在 query generation 后用另一个 LLM 做 critique (联系 Constitutional AI, Bai et al., 2022)。Paper 的 LLM-as-judge 已经有这个雏形。

4. **Cross-encoder reranking of tool calls**: parallel tool calls 模式下，用 cross-encoder rerank 多个 candidate tool calls。

5. **Tool definition distillation**: GPT-4o 的 tool use 能力可能 distill 到 smaller model，类似 ToolLLM (Qin et al., 2023) 的工作。

---

## 10. Takeaways for Building Intuition

1. **Decomposition > Monolithic SQL string**: 把 query API 拆成 structured arguments，比让 LLM 生成 raw SQL 更可验证、更 dialect-agnostic。
2. **Boolean is easy, text is hard**: 这是 fundamental。Boolean 是离散、constrained 的；text 涉及 semantic ambiguity。
3. **Model size predicts consistency, not peak performance**: GPT-4o 不是 EM 最高的，但是 cross-domain variance 最小的。
4. **Most ablations don't matter**: parallel calls, rationale, per-collection tools, structured generation — 都在 noise 范围内。真正重要的是 tool definition schema 本身。
5. **Benchmark maintenance is cheap**: <$100/year，社区可以 frequent replicate。
6. **No-tool-call is a real failure mode**: Gemini 2.0 Flash 一半时间根本不调 function。Evaluation 要 catch 这个。
7. **LLM-as-judge ≠ technical correctness**: preference ranking 和 AST 不一致，preference signal 可能反映 readability 而非 correctness。

---

## References (Web Links)

- [Gorilla LLM](https://gorilla.cs.berkeley.edu/)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
- [Compound AI Systems (Zaharia et al.)](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)
- [Spider 2.0](https://arxiv.org/abs/2411.07763)
- [BIRD Benchmark](https://bird-bench.github.io/)
- [WikiSQL](https://github.com/salesforce/wikisql)
- [STaRK Benchmark](https://arxiv.org/abs/2404.13207)
- [LOTUS](https://arxiv.org/abs/2407.11418)
- [TAG](https://arxiv.org/abs/2408.14717)
- [SUQL](https://arxiv.org/abs/2311.09818)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [WebGPT](https://arxiv.org/abs/2112.09332)
- [Self-Instruct](https://arxiv.org/abs/2212.10560)
- [Reflexion](https://arxiv.org/abs/2303.11366)
- [DSPy](https://arxiv.org/abs/2310.03714)
- [DSPy Assertions](https://arxiv.org/abs/2312.13382)
- [SPADE](https://arxiv.org/abs/2401.03038)
- [AlphaCodium (Flow Engineering)](https://arxiv.org/abs/2401.08500)
- [DPO](https://arxiv.org/abs/2305.18290)
- [Network of Networks](https://arxiv.org/abs/2407.16831)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet)
- [Weaviate GitHub (DBGorilla)](https://github.com/weaviate/gorilla)
- [RAG (Lewis et al.)](https://arxiv.org/abs/2005.11401)
- [StructuredRAG](https://arxiv.org/abs/2408.11061)
- [Efficient Guided Generation](https://arxiv.org/abs/2307.09702)
- [Database Gyms](https://www.cidrdb.org/)
- [Lakehouse](https://www.cidrdb.org/cidr2021/papers/cidr2021p17.pdf)
- [SPARQL](https://www.w3.org/TR/sparql11-query/)
- [GoEX](https://arxiv.org/abs/2404.06921)
- [AvaTaR](https://arxiv.org/abs/2406.11200)
- [MIPRO](https://arxiv.org/abs/2406.11695)
- [Specifications (Stoica et al.)](https://arxiv.org/abs/2412.05299)
- [Agentic Information Retrieval](https://arxiv.org/abs/2410.09713)

---

Andrej，希望这个 deep dive 帮你 build intuition。这篇 paper 的核心 intellectual contribution 不在 model 而在 **tool definition schema design** — 把 database query API 重构为 LLM-friendly 的 structured arguments。这个方向跟 DSPy 的 declarative programming、constitutional AI 的 critique loop、以及你自己在 neural net pedagogy 里强调的 "modular abstraction" 都有共鸣。期待后续 Weaviate 团队扩展到 multi-hop、cross-collection JOIN、以及 real-world schema 的 follow-up work。
