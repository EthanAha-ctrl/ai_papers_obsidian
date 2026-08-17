---
source_pdf: Deep GraphRAG ABalanced Approach to Hierarchical Retrieval and Adaptive
  Integration.pdf
paper_sha256: a1cb61b3004050a746890295624b794f0f438e29cf1933ced9cbb022c8dd1c46
processed_at: '2026-08-03T18:30:52-07:00'
target_folder: RAG
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Deep GraphRAG 人话版

好，Andrej，我换种讲法。抛开公式堆砌，先讲清楚"为什么"，技术细节当佐料。

---

## 一、这个故事在讲什么

想象你在读一本很厚的书，要回答"这本书里所有提到法国大革命的角色之间有什么关系"这种问题。

**普通RAG**的做法：拿你的问题去vector database里捞最相似的几段文字，拼起来喂给LLM。问题来了——"法国大革命"这件事散落在书的二十个章节里，你捞到的可能只是其中三段，LLM根本看不到全貌。

**GraphRAG**的做法：先把整本书抽成一张knowledge graph——entity是节点，关系是边。然后你能在图上"走"，从一个entity跳到另一个，把分散的信息串起来。

但GraphRAG自己也有麻烦，就是这篇paper要解决的。

---

## 二、GraphRAG的尴尬：两个极端

GraphRAG里有两派做法，各走极端：

### Local Search：像放大镜

从query提到的entity出发，往周围看一两跳的邻居。快，准，但只能看到局部。

**例子**：你问"奥巴马的夫人是谁"，Local Search能答。你问"美国近三任总统的夫人都做过什么慈善"，Local Search就傻了——它不知道要去捞"近三任总统"这个集合。

### Global Search：像直升机俯瞰

把整个图分成几十个community（用Louvain算法），每个community让LLM写个summary。你问问题的时候，让每个community的summary分别尝试回答，最后map-reduce汇总。

**问题**：慢得要死（几十次LLM call），而且coarse summary把local fact给磨没了。你问"奥巴马的夫人叫什么"，summary里可能只写了"奥巴马家庭"，具体名字没了。

看paper里的Table 1就特别明显：

| | Local Questions | Global Questions |
|---|---|---|
| Local Search | 41% | 16% |
| Global Search | 20% | 31% |

这就是**seesaw**——上去了这个，掉下来那个。

Microsoft后来出了个**DRIFT Search**想折中，但本质还是递归式的，慢，且流程固定，不能根据query复杂度动态调整。

---

## 三、Deep GraphRAG的思路：分层瞄准

Paper的insight很朴素：**为什么不先选大区域，再选小区域，再选具体目标？** 像打靶一样，先瞄准靶盘，再瞄准靶心。

把图建成三层hierarchy：

```
Level 2: 几个大community（比如"美国政治"、"欧洲经济"）
Level 1: 每个大community下几个中community（"美国民主党"、"美国共和党"）
Level 0: 具体entity（"奥巴马"、"拜登"、"希拉里"）
```

然后检索时分三步：

### 第一步：顶层粗筛

用reranker快速给所有Level 2 community打分，留top-3（beam width $k=3$）。这一步追求**recall**——别把正确的community误杀掉。

### 第二步：中层精筛

把top-3的children展开，再rerank一遍，再留top-3。

这里有个小trick：打分时把parent community的embedding和child的embedding**相加**再rerank。为什么？防止child本身语义跑偏但parent正确的情况被误杀。Parent相当于给child加了个"出生证"。

### 第三步：entity层精确定位

把候选community里的entities展开，给每个entity打分，选top-m。

这里的核心trick：entity的representation不是它自己，而是**它自己和它parent community的拼接**：

$$
D_{\mathrm{ctx}}(v) = [D(v) \,;\, D(c_{\mathrm{parent}})]
$$

**人话翻译**：当你判断"Michelle Obama"和query"美国第一夫人"匹不匹配时，不要只看Michelle Obama的描述，还要看她所在的community（"US Politics"）。如果她local描述里没提"美国"二字，但parent community是US Politics，parent embedding就把这个context补回来了。

这个trick简单，但解决了一个实际痛点：entity extraction出来的描述往往不完整，concat parent相当于给entity加了个"我从哪里来"的标签。

### 为什么beam search？

因为graph traversal是组合爆炸的。如果你BFS全展开，成本指数涨。Beam search $k=3$ 确保每层只保留3个candidate，总成本是 $O(3 \times 3 \times m) = O(9m)$，$m$ 是entity数量。

更关键的是，beam search让exploration和exploitation自然balance——top层exploration（看大方向），bottom层exploitation（精准定位）。

---

## 四、Knowledge Integration：用小模型做大模型的事

检索到的entities/relations要"整合"成一段distilled knowledge $C$，再喂给generator LLM答问题。这个整合步骤本身需要LLM——你不可能把几十个entity直接塞给generator。

Paper用了Qwen2.5-72B做这个integration，但72B太贵。能不能用1.5B？

直接用1.5B做integration效果很差（Table 1里只有22.27%）。Paper用RL fine-tune 1.5B，让它接近72B效果。

---

## 五、DW-GRPO：动态加权多奖励RL

### 标准GRPO的问题

GRPO是DeepSeekMath提的RL算法，比PPO简单——不用critic，用group-relative advantage。

Multi-reward时，你给三个reward：relevance、faithfulness、conciseness，固定权重 $r = w_1 r_1 + w_2 r_2 + w_3 r_3$，比如各占1/3。

**问题**：Conciseness特别好优化（just写短点），Relevance难优化（要semantic alignment）。模型会偷懒——concsiseness快速涨到顶，relevance停滞。这就是seesaw effect，paper里Figure 3看得清清楚楚。

### DW-GRPO的核心idea

**别用固定权重，根据每个reward的增长速度动态调权重——增长慢的reward加权重，增长快的减权重。**

类比：健身教练看你哪块肌肉练得不够，下次多练那块。

具体怎么算？

设reward $j$ 在最近 $w$ 个step的值是 $\mathbf{r}_j^{(t, w)}$。用线性回归拟合它的斜率 $\mathrm{slope}_j$。然后normalize：

$$
\alpha_j(t-1) = \frac{\mathrm{slope}_j}{\Delta r_j}
$$

$\Delta r_j$ 是reward在window内的range（max-min）。除以它是为了让不同scale的reward可比。

然后权重用softmax分配：

$$
w_j(t) = \frac{W \exp(-\alpha_j / T)}{\sum_{j'} \exp(-\alpha_{j'} / T)}
$$

注意那个**负号**——$\exp(-\alpha_j)$ 意思是 $\alpha_j$ 越小（增长越慢），weight越大。$T$ 是temperature，控制softmax的"sharpness"。

$W = \sum_j w_{j, 0}$ 是初始权重总和，保证总weight scale不变。

### 这个idea的来源

这个思路在Multi-Task Learning里有先例：
- **Uncertainty Weighting** (Kendall et al.): 用task uncertainty作weight
- **GradNorm** (Chen et al.): 用gradient norm作weight
- **PCGrad** (Yu et al.): gradient冲突时投影

DW-GRPO更轻量，不用算gradient norm，只用reward trajectory的slope。计算便宜，适合LLM RL。

### 实验效果

Figure 3很直观：
- 标准GRPO: Conciseness几天就到顶了，Relevance和Faithfulness一直平的
- DW-GRPO: 三个reward同步缓慢上涨

最终Table 1：
- 1.5B baseline (NQ, DeepSeek-R1 generator): 22.27%
- 1.5B + DW-GRPO: 42.09%
- 72B baseline: 43.98%

1.5B + DW-GRPO达到了72B的**95.7%**。这个distillation效果挺惊人的。

---

## 六、几个关键trade-off和limitation

### 1. Latency vs Accuracy

Deep GraphRAG比DRIFT Search快86%（Local）/81.6%（Global），原因是beam search只展开3个candidate，不像DRIFT全局递归。

### 2. Comprehensive Questions (CQ)上不总是赢

CQ需要local fact + global context混合。Table 1里有些case，Local Search反而比Deep GraphRAG强（比如NQ+DeepSeek-R1，Local CQ 23.20% vs Deep 19.60%）。

paper坦诚承认：hierarchical abstraction会丢失细节。这其实是所有hierarchical方法的通病——往上层abstract就会丢local细节。

### 3. Graph Construction成本

用Qwen2.5-72B抽entity + edge description，对大corpus是笔不小开销。paper完全没讨论这部分成本。

### 4. 固定3层hierarchy

为什么是3层？paper没说。我的猜测：GraphRAG典型corpus没那么大，3层够用。如果是Wikipedia级别，可能要5-7层。

---

## 七、paper在学术地图上的位置

| 工作 | 关系 |
|---|---|
| **Microsoft GraphRAG** (2024) | Deep GraphRAG的直接对标，提供Local/Global/DRIFT三个baseline |
| **HippoRAG** (2025) | 神经科学inspired的PPR-based retrieval，不同路线 |
| **RAPTOR** (2024) | text chunk的hierarchical retrieval，Deep GraphRAG的graph版 |
| **GraphReader** (2024) | agent-based，让LLM agent自己决定怎么读图 |
| **GFM-RAG** (2025) | Graph Foundation Model路线，把retrieval做成pretrained model |
| **GRPO** (DeepSeekMath 2024) | DW-GRPO的基础 |
| **PLE** (RecSys 2020) | "seesaw effect"概念的出处，DW-GRPO在optimization层面解决 |

---

## 八、我的整体判断

**好的地方**：
- 分层beam search + context-aware embedding是graph traversal的合理工程方案
- DW-GRPO的"reward slope动态调权"这个idea简洁有效，可以迁移到其他multi-reward RL场景
- 1.5B逼近72B的distillation结果实在

**可以挑刺的地方**：
- Graph construction成本不讨论
- 三层hierarchy没有ablation，为什么不是2层或4层？
- DW-GRPO的window size $w$ 和temperature $T$ 怎么选的，没说
- CQ上的弱点承认了但没解决

**值得记住的insight**：
1. Retrieval的coarse-to-fine beam search比recursive更适合graph
2. Entity representation concat parent embedding——简单但有效
3. Multi-reward RL用reward growth rate调权重，比固定权重好

---

如果想我再展开某块——比如Louvain怎么决定层数、beam search的 $k$ 怎么选、或者DW-GRPO在更复杂reward设定下会不会崩——尽管问。

References还是放一下：
1. **Microsoft GraphRAG**: https://arxiv.org/abs/2404.16130
2. **DRIFT Search**: https://www.microsoft.com/en-us/research/blog/introducing-driftsearch-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/
3. **GRPO (DeepSeekMath)**: https://arxiv.org/abs/2402.03300
4. **HippoRAG**: https://arxiv.org/abs/2405.14831
5. **RAPTOR**: https://arxiv.org/abs/2401.18059
6. **GraphReader**: https://arxiv.org/abs/2406.14550
7. **GFM-RAG**: https://arxiv.org/abs/2502.01113
8. **PLE (seesaw effect)**: https://doi.org/10.1145/3383313.3412236
9. **GradNorm**: https://arxiv.org/abs/1711.02257
10. **Uncertainty Weighting**: https://arxiv.org/abs/1705.07115

---

# Deep GraphRAG 深度解析

Andrej，这篇paper来自Ant Group，核心是在GraphRAG这个领域里把两件事做得更细致：**分层检索的beam search** 和 **multi-reward RL的动态加权**。让我从intuition开始，逐步深入到数学和工程细节。

---

## 1. 问题动机：GraphRAG的exploration-exploitation困境

要理解这篇paper，先要把GraphRAG的设计空间想象清楚。

传统RAG用vector retrieval，遇到multi-hop reasoning就崩了——因为它检索的是"语义近邻"，而不是"推理路径上的近邻"。GraphRAG通过Knowledge Graph (KG)显式建模实体关系来解决这个，但带来新的问题：

- **Local Search** (Microsoft GraphRAG里的local模式)：从query entity出发，BFS/DFS遍历邻居。优点是快、精度高（对单跳fact），缺点是看不到远处的cross-community信息。
- **Global Search** (Map-Reduce风格)：把每个community summary让LLM单独答一遍，最后聚合。对需要全局aggregation的query很好，但map-reduce导致"local fact"被coarse summary淹没，且latency极高。

观察paper里Table 1：Local Search在HotpotQA上GQ只有**10.63%**，Global Search在LQ上只有**18.49%**——典型的seesaw效应。DRIFT Search是Microsoft后来提出的hybrid，但仍然是递归式的固定流程。

Deep GraphRAG的insight：**把检索过程做成hierarchical coarse-to-fine beam search**，每一层都用reranker做pruning，避免一次性遍历整个图，也避免map-reduce的information loss。

---

## 2. Graph Construction Pipeline

这部分很多paper会skip，但这篇做得比较扎实，值得讲。

### 2.1 Chunking + Triple Extraction

```
Corpus D → sliding window (T=600 tokens, overlap=100) → chunks
```

每个chunk用Qwen2.5-72B-Instruct (temperature=0)抽取entities + directed edges。**关键细节**：每个edge强制要求带natural language description，不是standard `(h, r, t)` triple。这是因为triple的relation往往太symbolic（"born_in", "founded"），丢失上下文细节。带自然语言描述的edge在后面rerank时和query对齐更自然。

Overlap=100是为了缓解边界entity被切碎的问题（比如"Barack Obama"跨chunk）。

### 2.2 Entity Resolution

这是GraphRAG最容易被忽视的工程难点。同一个"United States"在不同chunk里可能被抽成"U.S." / "USA" / "United States of America"，如果不merge，graph就是fragmented的。

Deep GraphRAG用hybrid策略：
1. 用bge-m3对entity description做embedding
2. Cosine similarity > τ=0.95的pairs作为candidate
3. LLM discriminator二次确认（避免false positive，比如"Apple Inc." 和 "apple fruit"）

τ=0.95这个阈值很严格，是为了减少误merge——误merge的代价远大于漏merge，因为后面hierarchy构建会propagate错误。

### 2.3 Louvain Hierarchy

构建multi-granular 3-level hierarchy C。用weighted Louvain algorithm，resolution parameter r=1.0（标准Louvain默认值）。

Louvain是modularity-based community detection，iteratively优化：
$$
Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)
$$

其中 $A_{ij}$ 是adjacency matrix，$k_i$ 是node $i$ 的degree，$m$ 是总边数，$\delta$ 是Kronecker delta。

Louvain recursively merge communities，形成tree。Level $L=0$ 是leaf nodes（entities），$L>0$ 是abstract community summaries。3层hierarchy意味着：top level（最粗，几个大community）→ mid level → leaf entities。

### 2.4 Context-aware Representation

公式 (1):
$$
D_{\mathrm{sub}}(c) = \frac{1}{|C_{\mathrm{sub}}(c)|} \sum_{c' \in C_{\mathrm{sub}}(c)} D(c')
$$

变量含义：
- $c$: 当前community
- $C_{\mathrm{sub}}(c)$: $c$ 的所有子community集合
- $|C_{\mathrm{sub}}(c)|$: 子community数量
- $D(c')$: 子community $c'$ 的representation vector
- $D_{\mathrm{sub}}(c)$: $c$ 的mean-pooled representation

Node representation则是**concat**它的local description + parent community的representation：
$$
D(v) = [\text{local\_desc}(v); D_{\mathrm{parent}}(v)]
$$

这个设计很关键——让leaf entity既保留local semantics，又携带hierarchical context。在第三阶段的entity-level search里，paper进一步用这个idea做context-aware reranking（后面会讲）。

---

## 3. Retrieval Process: 三阶段Coarse-to-Fine Beam Search

这是paper的核心算法。我把它抽象出来看：

```
Top-level community (粗) → Mid-level community (中) → Entity (细)
       ↓                       ↓                       ↓
   Re-ranker              bge-reranker-v2-m3        Context-aware sim
   (快速pruning)           (语义对齐)                 (topology-aware)
```

Beam width $k=3$：每层保留top-3 candidates，避免explosion。

### Phase 1: Top-level Coarse Filtering
```
score C_top via Re-ranker(q, D(c_top))
C_mid ← Top-k from C_top
```

顶层用re-ranker（paper没明确说哪个，可能是bge-reranker-v2-m3的轻量版）快速做coarse relevance估计。这里追求**recall**——不能在顶层就把正确community filter掉。

### Phase 2: Middle-level Re-ranking
```
Expand C_mid → C_mid'  (子community)
Score: Re-ranker(q, D(c_mid) + D(c_mid'))
Retain Top-k from C_mid'
```

注意这里的scoring function：把parent community的representation和child community的representation**相加**后再rerank。这是个很重要的设计——防止child community本身semantics偏离query但parent对齐的情况。Parent representation相当于一个"prior"，让reranker不至于被局部噪声带偏。

### Phase 3: Entity-level Context-Aware Search
```
Expand C_stop → V_cand (candidate entities)
for each v in V_cand with parent c_parent:
    D_ctx(v) = Concat(D(v), D(c_parent))
    s_final(v) = sim_cos(q, D_ctx(v))
R_pre ← Top-m from V_cand
```

公式分解：
$$
D_{\mathrm{ctx}}(v) = [D(v) \,;\, D(c_{\mathrm{parent}})]
$$

- $D(v)$: entity $v$ 的local embedding
- $D(c_{\mathrm{parent}})$: $v$ 所在parent community的embedding
- $[ \cdot \,;\, \cdot ]$: vector concatenation

Final score:
$$
s_{\mathrm{final}}(v) = \cos(q, D_{\mathrm{ctx}}(v))
$$

**Intuition**: 为什么concat parent embedding有用？想象query问"美国第44任总统的夫人是谁"——entity "Michelle Obama" 的local description可能没提到"美国"，但她所在的community可能就是"US Politics"，parent embedding就把这个macro context注入进来了。相当于给entity加了个"它从哪里来"的tag。

这种做法类似RAPTOR的hierarchical retrieval，但RAPTOR是tree-structured text chunk，Deep GraphRAG是graph-structured entities + communities。

### Phase 4: Knowledge Integration
$$
R = \text{Hierarchical-Integration}(R_{\mathrm{pre}})
$$

把检索到的entities按hierarchy整合成distilled knowledge $C$，然后送给generator LLM。

---

## 4. DW-GRPO: 动态加权多奖励RL

这部分是paper的第二个亮点。**动机**：GRPO在multi-reward场景用固定权重，导致seesaw effect——easy reward被over-optimized，hard reward停滞。

### 4.1 三个Reward

**(1) Relevance**: 
$$
r_{\mathrm{rel}} = f_{\mathrm{cross}}(Q, C)
$$
用bge-reranker-v2-m3 (cross-encoder)对query $Q$ 和distilled knowledge $C$ 打分。Cross-encoder比bi-encoder精度高，因为 $Q$ 和 $C$ 在Transformer里有attention interaction。

**(2) Faithfulness**:
$$
r_{\mathrm{faith}} = f_{\mathrm{BERT}}(C, K)
$$
BERTScore F1 (用bge-m3作为underlying model)，衡量 $C$ 和original knowledge $K$ 的语义保真度。F1而非纯precision是为了避免 $C$ 是 $K$ 子集这种退化。

**(3) Conciseness**:
$$
r_{\mathrm{conc}} = \max\left(0, 1 - \frac{\mathrm{len}(C)}{\mathrm{len}(K)}\right)
$$
- $\mathrm{len}(\cdot)$: token长度
- 如果 $C$ 比 $K$ 长，reward=0
- 如果 $C$ 短，reward线性增长

这个设计鼓励compactness，但用max(0,·)保证非负。

### 4.2 GRPO回顾

Standard GRPO (DeepSeekMath提出)的优势函数：
$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)}
$$

group-relative advantage，不需要critic。Multi-reward时：
$$
r_i = \sum_j w_j r_{i,j}
$$

GRPO问题：$w_j$ 是固定常数，比如 $w_1 = w_2 = w_3 = 1/3$。这导致easy reward（conciseness，纯长度计算）快速增长，hard reward（relevance，需要semantic alignment）停滞——典型的seesaw。

### 4.3 DW-GRPO核心机制

公式 (5):
$$
\widetilde{\hat{A}} = \sum_j w_j r_j - \frac{\sum_j w_j r_j - \mathrm{mean}(\sum_j w_j r_j)}{\mathrm{std}(\sum_j w_j r_j)}
$$

等等，这个公式有点奇怪，让我仔细看。它写的是 $\widetilde{\hat{A}}$，其实是把weighted reward求和后做group normalization。这和标准GRPO的advantage形式一致，只是reward被加权。

真正的核心是**动态权重**。设 reward $r_j$ 在window $w$ 内的轨迹为 $\mathbf{r}_j^{(t, w)} = \{r_j^{(t-w+1)}, \ldots, r_j^{(t)}\}$。

定义reward range:
$$
\Delta r_j = \max(\mathbf{r}_j^{(t, \tau)}) - \min(\mathbf{r}_j^{(t, \tau)})
$$

公式 (6):
$$
\alpha_j(t-1) = \begin{cases} 0, & \text{if } \Delta r_j = 0 \\ \frac{\mathrm{slope}_j}{\Delta r_j}, & \text{otherwise} \end{cases}
$$

- $\alpha_j(t-1)$: reward $j$ 的**normalized rate of change**
- $\mathrm{slope}_j$: least-squares拟合的斜率
- 除以 $\Delta r_j$ 是为了normalize，让不同scale的reward可比较

Least-squares fit:
$$
\mathrm{slope}_j = \arg\min_{k, b} \|\mathbf{r}_j^{(t, w)} - (k\mathbf{x} + b)\|^2
$$

这里 $\mathbf{x}$ 是时间index向量，$k$ 是斜率，$b$ 是截距。这是个简单的线性回归。

公式 (7):
$$
w_j(t) = \frac{W \exp(-1 \cdot \alpha_j(t-1) / T)}{\sum_{j'} \exp(-1 \cdot \alpha_{j'}(t-1) / T)}
$$

变量：
- $W = \sum_j w_{j,0}$: 总权重scale，保持和initial weight和不变
- $T$: temperature（softmax温度）
- $\alpha_j(t-1)$: 上一步的normalized growth rate

**关键insight**: 用 $\exp(-\alpha_j / T)$——**growth rate越慢，weight越大**。也就是说，reward增长慢的component被分配更高权重，强制policy去优化它。这是个自适应的"补短板"机制。

Temperature $T$ 控制softmax的sharpness。$T$ 小，weight更倾向于最慢的那个reward；$T$ 大，weight更平均。

### 4.4 与Multi-Task Learning的联系

这个idea和Multi-Task Learning里的**Uncertainty Weighting** (Kendall et al., CVPR 2018)和**GradNorm** (Chen et al., ICML 2018)思想类似：

- Uncertainty Weighting: 用task uncertainty作为weight
- GradNorm: 用gradient norm作为weight，平衡不同task的gradient scale
- DW-GRPO: 用reward growth rate作为weight

DW-GRPO更接近**PCGrad** (Yu et al., NeurIPS 2020)或**Preference Learning**的思想——把"哪个reward在lag"作为信号。

paper引用了PLE (Progressive Layered Extraction, RecSys 2020)的seesaw concept，但PLE是model architecture层面的解决，DW-GRPO是optimization层面。

### 4.5 实验验证（Figure 3）

在HotpotQA + Qwen2.5-1.5B上：
- 标准GRPO: Conciseness reward快速maximize（~1.0），Relevance和Faithfulness停滞
- DW-GRPO: 三个reward同步稳步增长

这验证了"补短板"机制有效。最终结果（Table 1）：
- 1.5B baseline: 22.27% (NQ, DeepSeek-R1 generator)
- 1.5B + DW-GRPO: 42.09% (NQ)
- 72B baseline: 43.98% (NQ)

也就是说，**1.5B + DW-GRPO达到了72B的95.7%性能**——非常显著的distillation效果。

---

## 5. 实验数据深度解读

### 5.1 NQ vs HotpotQA的差异

HotpotQA是multi-hop reasoning dataset，NQ是single-fact retrieval。Table 1显示：
- HotpotQA上Deep GraphRAG的GQ提升更显著（56.25% vs Local Search 10.00%）
- NQ上Deep GraphRAG在LQ略有提升（45.36% vs LS 41.32%）

这符合预期：hierarchical retrieval对multi-hop query收益最大，因为multi-hop需要cross-community navigation。

### 5.2 CQ (Comprehensive Questions)的trade-off

CQ需要local fact + global context结合。Table 1显示：
- HotpotQA: Deep GraphRAG在CQ上24.76%，超过所有baseline
- NQ (DeepSeek-R1): Local Search CQ 23.20%，Deep GraphRAG只有19.60%

paper坦诚承认这个limitation——hierarchical summarization可能obscure fine-grained local facts。这其实是hierarchical方法普遍的问题：当你向上abstract时，会丢失细节。

### 5.3 Latency分析

Figure 2：Deep GraphRAG在NQ上比DRIFT Search减少86% (Local) / 81.6% (Global) latency。原因是：
1. Beam search只在 $k=3$ 个candidates上展开，不像DRIFT全局递归
2. 三层pruning让early level快速filter掉irrelevant branches

---

## 6. 批判性思考

### 6.1 优点
- 分层reranking + beam search是graph traversal的合理范式
- Context-aware entity embedding（concat parent）是简单但有效的trick
- DW-GRPO对multi-reward RL是合理的扩展

### 6.2 潜在问题

1. **Graph Construction成本**: 用Qwen2.5-72B抽entity + edge description，对large corpus成本高。paper没提graph construction的时间。

2. **Hierarchy深度固定为3**: 真实世界graph的natural hierarchy可能更深或更浅，固定3层可能suboptimal。Adaptive hierarchy depth可能更好。

3. **DW-GRPO的window $w$ 和temperature $T$ 超参敏感**: paper没给ablation，不知道这两个hyperparameter如何选。

4. **CQ的weakness**: 论文承认了hierarchical summarization在CQ上不universal的winner。可以考虑hybrid retrieval——同一query既触发local fact retrieval又触发global summarization，最后merge。这是Microsoft DRIFT Search的思路，但DRIFT没有精细的beam search。

5. **DW-GRPO只测了1.5B**: 在更大模型（7B、14B）上是否同样有效？小模型可能更倾向于exploit easy reward，所以DW-GRPO对小模型帮助更大。大模型可能已经能balance multi-reward，DW-GRPO收益减小。

### 6.3 与其他工作的联系

- **RAPTOR** (Sarthi et al., 2024): hierarchical text retrieval，类似的tree-structured abstraction
- **HippoRAG** (Gutierrez et al., 2025): 神经科学启发的memory retrieval，PPR-based
- **GFM-RAG** (Luo et al., 2025): Graph Foundation Model for RAG
- **GraphReader** (Li et al., 2024): agent-based long context reading
- **Microsoft Drift Search**: hybrid local+global，但递归式

DW-GRPO的动态权重思想可以追溯到：
- **DPO** (Rafailov et al.): 用preference代替reward model
- **TRPO/PPO** (Schulman et al.): trust region policy optimization
- **DeepSeekMath GRPO** (Shao et al. 2024): group-relative advantage，DW-GRPO的基础

### 6.4 几个可以追问的intuition

1. **为什么三层hierarchy够用？** paper没讨论。我的猜测：GraphRAG的corpus通常不会太huge，3层能cover大多数query。如果corpus是Wikipedia级别，可能需要5-7层（参考RAPTOR的4层）。

2. **为什么用concat而不是add?** Concat让local和parent representation在feature space里独立，reranker可以分别attend到两部分。Add会blur掉细节。

3. **DW-GRPO的负号**：$-1 \cdot \alpha_j$ 的"-1"意思是反相关——增长越慢，权重越大。如果改成 $+\alpha_j$，就是winner-take-all——增长快的reward分配更多weight，会让seesaw更严重。

4. **为什么BERTScore用F1而不是precision?** F1同时考虑precision和recall。如果只用precision，模型可能生成超短answer（高precision低recall）；F1强制balance。

---

## 7. 我的整体评价

这篇paper在GraphRAG领域是一个solid的工程优化。技术novelty中等：
- Hierarchical beam search不是新概念，但应用到GraphRAG的coarse-to-fine是合理的具体化
- Context-aware entity embedding是个小但有效的trick
- DW-GRPO是GRPO的合理扩展，"reward growth rate作为weight"这个idea值得记住

实验数据扎实，1.5B + DW-GRPO逼近72B的结果很impressive。但ablation不全，超参敏感度没分析。

对于想复现的人：核心是三阶段beam search + context-aware embedding + DW-GRPO这三个模块。Graph construction的Louvain和entity resolution有现成工具，reranker用bge-reranker-v2-m3即可。

---

## References

1. **Microsoft GraphRAG**: https://arxiv.org/abs/2404.16130
2. **DRIFT Search Blog**: https://www.microsoft.com/en-us/research/blog/introducing-driftsearch-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/
3. **HippoRAG**: https://arxiv.org/abs/2405.14831
4. **GraphReader**: https://arxiv.org/abs/2406.14550
5. **GFM-RAG**: https://arxiv.org/abs/2502.01113
6. **RGL Framework**: https://arxiv.org/abs/2503.19314
7. **RAPTOR**: https://arxiv.org/abs/2401.18059
8. **GRPO (DeepSeekMath)**: https://arxiv.org/abs/2402.03300
9. **DPO**: Rafailov et al., NeurIPS 2023, https://arxiv.org/abs/2305.18290
10. **bge-m3**: https://arxiv.org/abs/2309.07597
11. **Natural Questions**: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00276
12. **HotpotQA**: https://arxiv.org/abs/1809.09600
13. **PLE (Seesaw Effect)**: https://doi.org/10.1145/3383313.3412236
14. **Uncertainty Weighting (Kendall et al.)**: https://arxiv.org/abs/1705.07115
15. **GradNorm (Chen et al.)**: https://arxiv.org/abs/1711.02257
16. **PCGrad (Yu et al.)**: https://arxiv.org/abs/2001.06882

如果你想进一步深挖某个模块——比如DW-GRPO和GradNorm的对比、或Louvain在entity resolution graph上的实际modularity值、或beam search的 $k$ ablation——可以告诉我，我可以再展开。
