---
source_pdf: BlueTempNet A Temporal.pdf
paper_sha256: 3cea636cec3a8de3f4b759bee98b8d24543cc52ac56dc64b8bf61eefb10fad41
processed_at: '2026-07-20T10:29:26-07:00'
target_folder: Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BlueTempNet: 深度技术解析

## 一、论文核心动机与问题定位

这篇论文解决的核心问题是：**主流social media platforms通过API对用户social interactions的数据访问极其受限**，导致computational social science领域无法精确研究user behavior的temporal dynamics。

### 1.1 平台数据可访问性对比（Table 1的深层含义）

Table 1构建了一个4维binary accessibility矩阵，覆盖8个platforms × 4种interaction types。关键发现：

| Platform | Follow | Block | Create | Join |
|---|---|---|---|---|
| Twitter/X | ◦ | × | × | × |
| Mastodon | ◦ | × | • | • |
| **Bluesky** | **•** | **•** | **•** | **•** |

其中符号定义：
- **×** = complete exclusion（完全不可访问）
- **◦** = partial access without timestamps（部分访问，无时间戳）
- **•** = perfect coverage（完全覆盖，含毫秒级时间戳）

**Bluesky是唯一一个在所有四个维度都达到"•"的平台**。这个独特性质源于Bluesky的ATproto（Authenticated Transfer Protocol）架构设计——其Personal Data Server (PDS) 模型使得user data通过public API端点 `public.api.bsky.app/xrpc/com.atproto.repo.listRecords` 完全可访问，且无需authentication。

这背后的architectural reason是：ATproto将数据控制权交给用户而非平台，每个user的data存储在他们选择的PDS上，而Bluesky的默认PDS `bsky.social` 的records对所有公开交互都提供了带UTC时间戳的完整记录。

参考：[ATproto Specification](https://atproto.com/specs/atp)，[Bluesky Public Data Policy](https://bsky.social/about/support/faq)

---

## 二、数据收集Pipeline的三层架构

Figure 3展示了一个hierarchical data collection pipeline，其设计逻辑是从community层向user层逐级细化：

### Layer 1: Community-Level Collection

收集Bluesky上所有公开的Custom Feeds。Feed是Bluesky的核心创新——用户可以通过Feed Generator创建自己的推荐算法。每个Feed都有一个URI标识符，格式如：

```
did:plc:...feed.generator/philosophy
```

其中`did:plc`是Decentralized Identifier (DID) method，基于PLC (Public Ledger of Credentials) registry。

### Layer 2: User-to-Community Level

对于每个Feed，收集两类user信息：
- **Feed Creators**: 使用Feed Generator创建Feed的用户
- **Feed Members**: 通过点击heart icon（Feed Like action）加入Feed的用户

### Layer 3: User-to-User Level

检查所有identified users之间的follow和block关系。这一步的关键设计决策是：

**当follow和block同时存在时，优先保留block edge**。这是一个principled choice——block代表用户明确表达的社会排斥意愿，语义优先级高于follow。

数据收集时间范围：May 11, 2023（Custom Feeds上线）至 May 11, 2024。

参考：[Bluesky API Documentation](https://docs.bsky.app/docs/api/com-atproto-repo-list-records)

---

## 三、三维度网络的形式化定义

### 3.1 Feed Creator Interaction Network: $G_{\mathcal{C}}$

$$G_{\mathcal{C}} = \{\mathcal{C}, \mathcal{E}^+, \mathcal{E}^-\}$$

其中：
- $\mathcal{C}$ = Feed Creator的节点集合，$|\mathcal{C}| = 17,146$
- $\mathcal{E}^+$ = positive edge集合（following），$|\mathcal{E}^+| = 273,696$
- $\mathcal{E}^-$ = negative edge集合（blocking），$|\mathcal{E}^-| = 24,362$

这是一个**signed directed graph**。Edge属性包含`sign`和`time`，其中`time`格式为ISO 8601（例：`2023-09-22T09:32:17.974Z`），精度到毫秒级。

### 3.2 Feed Member Interaction Network: $G_{\mathcal{M}}$

$$G_{\mathcal{M}} = \{\mathcal{M}, \mathcal{E}^+, \mathcal{E}^-\}$$

其中：
- $\mathcal{M}$ = Feed Member的节点集合，$|\mathcal{M}| = 134,946$
- $\mathcal{E}^+$ = positive edge集合，$|\mathcal{E}^+| = 4,871,132$
- $\mathcal{E}^-$ = negative edge集合，$|\mathcal{E}^-| = 435,700$

### 3.3 Community Interaction Network: $G_{\mathcal{A}}$

$$G_{\mathcal{A}} = \{\mathcal{V}, \mathcal{E}\}$$

其中：
- $\mathcal{V} = \{\mathcal{C}, \mathcal{M}, \mathcal{F}\}$ 是三类节点的并集（Feed Creators, Feed Members, Feeds）
- $\mathcal{E} = \{\mathcal{G}, \mathcal{L}\}$，$\mathcal{G}$ = generation edges（creator创建feed），$\mathcal{L}$ = like edges（member加入feed）

这是一个**tripartite affiliation graph**，edge都是undirected。统计量：
- Creator Nodes: 17,536
- Member Nodes: 136,724
- Feed Nodes: 39,235
- Join Edges (Like): 297,828
- Create Edges (Generate): 39,235

**关键设计细节**：有8,527个用户同时是Creator和Member。在$G_{\mathcal{A}}$中，这些用户被表示为**两个不同的节点**——一个creator node和一个member node。这种设计保留了role specificity，避免了将不同类型的interaction混为一体。

### 3.4 Multi-Graph整合

最终存储为`multi_graph.gexf`（MultiGraph object），整合三个维度。Undirected edges被转换为bidirectional edges以便于处理。

---

## 四、Feed Economic Dynamics分析（Figure 4的技术细节）

### 4.1 Term Frequency计算

论文使用TF向量量化Feed display name中的词汇特征：

$$\text{TF}(t, d) = \frac{\text{count}(t, d)}{\sum_{t' \in d} \text{count}(t', d)}$$

其中：
- $t$ = 具体term
- $d$ = Feed的display name（作为document）
- $\text{count}(t, d)$ = term $t$ 在display name $d$ 中出现的次数
- 分母 = display name $d$ 中所有term的总出现次数

### 4.2 Pearson相关分析

对每个term，计算其出现与Feed like数之间的Pearson correlation：

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

其中：
- $x_i$ = Feed $i$ 中term的TF值
- $y_i$ = Feed $i$ 获得的like数
- $n$ = 包含该term的Feed数量

**核心发现**的intuition：

- **高频率但低相关**：art, media, gallery——大量Feed使用这些名称，但它们与popularity几乎无关（$r < -0.005$），说明niche内容的创建者很多，但受众分散
- **高正相关**：likes, discover, classic, hot——这些通用内容发现类Feed获得更多likes，反映了Bluesky用户对algorithmic curation工具的需求
- **特定社区的正相关**：blacksky（Black community）, science, booksky, adult content——说明community-specific Feed在Bluesky生态中具有强烈的吸引力

这揭示了一个**attention economy中的结构性矛盾**：用户倾向于创建reflect个人兴趣的niche Feed（如art），但用户最愿意subscribe的是broad discovery tools和identity-affirming communities。

参考：[Social Media Attention Economy Research](https://doi.org/10.1038/s41586-023-06667-8)

---

## 五、Temporal Interaction Dynamics分析（Figures 5 & 6）

### 5.1 平台开放事件的影响（Figure 5）

February 6, 2024是Bluesky从invitation-only转为open beta的关键日期。论文通过CDF（Cumulative Distribution Function）可视化展示了各interaction type的累积增长。

**精确增长率**：

| Network Dimension | Interaction | Increase after Feb 6 |
|---|---|---|
| $G_{\mathcal{C}}$ | Follow | 19.2% |
| $G_{\mathcal{C}}$ | Block | 27.3% |
| $G_{\mathcal{M}}$ | Follow | 15.8% |
| $G_{\mathcal{M}}$ | Block | 25.9% |
| $G_{\mathcal{A}}$ | Create | 44.0% |
| $G_{\mathcal{A}}$ | Join | 34.12% |

**两个关键观察**：

1. **Block增长始终快于Follow**：在两个user-to-user维度中，block的增长率（27.3%, 25.9%）都高于follow（19.2%, 15.8%）。这暗示平台开放后，用户面对更多陌生用户时，defensive interaction的增长速度快于connective interaction。

2. **Create增长远超Join**：Feed创建增长44% vs. 加入增长34.12%，说明新用户涌入后，更倾向于成为Feed Creator（content curator）而不仅仅是Member（content consumer），体现了Bluesky用户驱动的algorithm设计理念带来的行为变化。

### 5.2 变异系数分析（Figure 6）

Coefficient of Variation (CV)用于度量distribution的稳定性：

$$\text{CV} = \frac{\sigma}{\mu}$$

其中：
- $\sigma$ = degree分布的standard deviation
- $\mu$ = degree分布的mean

Figure 6分别展示了三个网络中in-degree和out-degree的CV随时间变化。

**核心发现**：

- **User-to-user interactions**（follow/block）的CV在October 2023之后趋于稳定，即使经历了February 2024的public launch，CV波动也很小
- **User-to-community interactions**（create/join）的CV在public launch前后出现大幅波动，之后才逐渐稳定

这揭示了**不同interaction type的resilience差异**：social ties（follow/block）作为用户间直接关系，在platform shock下更具韧性；而community participation作为用户与content algorithm的关系，更容易受外部事件影响。

论文将此解读为：**user-to-user interactions构成了Bluesky社交结构的骨架，而user-to-community interactions是更灵活的表层行为**。

---

## 六、Network Topology分析（Tables 6 & 7）

### 6.1 Signed Graph Metrics深度解读

#### Degree Assortativity

**Positive Degree Assortativity**（follow维度）：
- $G_{\mathcal{C}}$: -0.08
- $G_{\mathcal{M}}$: -0.06

值为负意味着**high-degree用户倾向于follow low-degree用户**。这与典型的social network中观察到的assortative mixing不同（通常degree assortativity为正或接近零）。在Bluesky的语境下，这说明influential users（high out-degree）倾向于关注大量普通用户，这可能是curation行为的一部分——Feed Creators需要广泛获取内容源。

**Negative Degree Assortativity**（block维度）：
- $G_{\mathcal{C}}$: -0.21
- $G_{\mathcal{M}}$: -0.16

更负的值说明**block行为更加disassortative**。Creator网络中block的disassortativity更强（-0.21 vs. -0.16），可能反映了Creator群体内部存在更明确的阵营划分——某些Creator更系统地block与其观点不同的其他Creators。

#### Reciprocity

$$\text{Reciprocity} = \frac{\text{number of mutual edges}}{\text{total number of edges}}$$

| | Follow Reciprocity | Block Reciprocity |
|---|---|---|
| $G_{\mathcal{C}}$ | 0.45 | 0.07 |
| $G_{\mathcal{M}}$ | 0.51 | 0.06 |

**Follow reciprocity远高于Block reciprocity**（约7-8倍），这符合social balance theory的预期——mutual following是social cohesion的体现，而mutual blocking虽然存在但较少，因为block行为本身就阻止了对方观察和回应。

#### Structural Balance

Triangle的structural balance基于Cartwright-Harary理论[5]。一个triangle有4种可能的sign configuration：

- **Balanced**: +++, +-- （friend of friend is friend; enemy of enemy is friend）
- **Unbalanced**: ++-, --- （friend of friend is enemy; enemy of enemy is enemy）

**Balanced triangle ratio**：
- $G_{\mathcal{C}}$: 0.47
- $G_{\mathcal{M}}$: 0.52

Member网络中social balance更显著。这个intuition是：**Member群体规模更大、更多样化，形成的社会结构更接近自然社会平衡**；而Creator群体更小、更专业化，social tension可能更明显。

#### Clustering Coefficient

**Positive Clustering**: 0.18 ($G_{\mathcal{C}}$), 0.20 ($G_{\mathcal{M}}$) — follower的邻居之间也倾向于互相关注
**Negative Clustering**: 0.04 (both) — blocker的邻居之间很少互block

**低negative clustering的intuition**：block是一种**个体化的defensive action**，不具备follow那样的social contagion效应。你block某人，通常不会导致你的朋友也去block同一个人。

参考：[Structural Balance Theory - Cartwright & Harary 1956](https://doi.org/10.1037/h0046010)

### 6.2 Affiliation Graph与Line Graph Projection

论文将tripartite affiliation graph $G_{\mathcal{A}}$ 投影为Feed-to-Feed的line graph $P_{\mathcal{A}}$：

$$P_{\mathcal{A}} = \{\mathcal{F}, \mathcal{E}, \mathcal{W}\}$$

其中edge定义和权重：

$$\mathcal{E} = \{(f_i, f_j) \mid U(f_i) \cap U(f_j) \neq \emptyset\}$$

$$w(f_i, f_j) = |U(f_i) \cap U(f_j)|, \quad w \in \mathcal{W}$$

其中：
- $U(f_i)$ = affiliation graph $G_{\mathcal{A}}$ 中与Feed $f_i$ 相邻的所有user集合（包括creators和members）
- 两个Feed $f_i$ 和 $f_j$ 之间有edge **当且仅当它们共享至少一个user**
- Edge权重 $w(f_i, f_j)$ = 共享user的数量

**这个projection的intuition**：将user-feed二部关系转化为feed-feed co-membership关系，揭示了**Feeds之间的隐式关联**——共享用户越多的Feeds，在语义或功能上越相关。

### Table 7的关键对比

| Metric | $G_{\mathcal{A}}$ | $P_{\mathcal{A}}$ |
|---|---|---|
| Average Path Length | 5.30 | 3.25 |
| Clustering Coefficient | 0.00 | 0.77 |
| Degree Assortativity | -0.16 | 0.59 |

**$P_{\mathcal{A}}$展现出small-world properties**：
- **高clustering** (0.77)：如果Feed A和Feed B共享用户，Feed B和Feed C共享用户，那么A和C也很可能共享用户——这形成了tight-knit Feed communities
- **短average path length** (3.25)：任意两个Feeds之间平均只需要3-4步即可通过shared-user chain到达
- **正degree assortativity** (0.59)：popular Feeds倾向于与其他popular Feeds共享用户，形成了"rich club"结构

**对比之下，$G_{\mathcal{A}}$的clustering为0.00**，这是tripartite graph的inherent property——三类节点之间不存在同类节点之间的direct edge，所以传统clustering coefficient在affiliation graph中无意义。

参考：[Small-World Networks - Watts & Strogatz 1998](https://doi.org/10.1038/30918)，[Bipartite Projection Backbone - Neal 2014](https://doi.org/10.1016/j.socnet.2014.01.002)

---

## 七、Community Detection on Line Graph（Figure 7）

论文对$P_{\mathcal{A}}$使用**Leiden Algorithm**[30]进行community detection。Leiden是对Louvain algorithm的改进，保证每个detected community是**well-connected**（单一node可以通过within-community path到达任何其他community member）。

**检测结果**：770个communities，展示top 8个。

从word cloud中可见的community主题：
- **语言/区域clusters**: German, Japanese, Spanish, English
- **话题clusters**: Ukraine-Russia war (political), Art-related
- **功能clusters**: Content discovery tools

**这个结果的深层intuition**：Feeds之间的co-membership pattern同时由**语言地理边界**和**兴趣/意识形态边界**驱动。语言clusters体现了Bluesky用户的全球化分布，而political clusters（如Ukraine-Russia）则显示了social media上polarization的natural emergence。

Leiden Algorithm的核心是quality function（modularity的变体）：

$$Q = \frac{1}{2m}\sum_{ij}\left[A_{ij} - \frac{k_i k_j}{2m}\right]\delta(c_i, c_j)$$

其中：
- $A_{ij}$ = adjacency matrix的元素（此处为weighted，即$w(f_i, f_j)$）
- $k_i = \sum_j A_{ij}$ = node $i$ 的weighted degree
- $m = \frac{1}{2}\sum_{ij} A_{ij}$ = 总edge weight
- $\delta(c_i, c_j)$ = Kronecker delta，当node $i$ 和 $j$ 在同一community时为1

参考：[Leiden Algorithm - Traag et al. 2019](https://doi.org/10.1038/s41598-019-41695-z)

---

## 八、Privacy与FAIR原则的工程设计

### 8.1 Field-level Encryption

User ID被anonymized为Base64 encoded string（如`bgAAAAAB...ba66QOmV4'`），同时移除所有PII（usernames, display names, biographies）。

**解密key的访问机制**：仅对bona fide researchers在审核后提供。这实现了privacy-compliant与research-reusable之间的平衡。

### 8.2 Node Index的一致性设计

每个CSV row包含一个**唯一Node Index**（integer），这个index在所有GEXF文件和metadata CSV文件之间保持一致。同一用户如果同时是Creator和Member，会有**两个不同的Node Index**，但共享同一个Anonymized ID。

**这个设计的intuition**：Node Index作为graph-theoretic identifier，而Anonymized ID作为cross-graph entity linkage key，使得研究者可以追踪同一用户在不同维度中的行为，同时graph operations不需要处理role ambiguity。

---

## 九、数据规模与Scale的Intuition

汇总所有三个维度的完整数据规模：

| Dimension | Nodes | Edges | Edge Types |
|---|---|---|---|
| $G_{\mathcal{C}}$ | 17,146 | 298,058 | Follow + Block (signed, directed) |
| $G_{\mathcal{M}}$ | 134,946 | 5,306,832 | Follow + Block (signed, directed) |
| $G_{\mathcal{A}}$ | 193,495 | 337,063 | Create + Join (undirected) |

**$G_{\mathcal{M}}$的edge密度**（5.10 × 10⁻⁴）比$G_{\mathcal{C}}$（1.89 × 10⁻³）更低——这是合理的，因为Member群体更大，可能的edge数（$n(n-1)$）增长更快，实际edge的增长跟不上。

**Isolated nodes的比例**：
- $G_{\mathcal{C}}$: 1,749 isolated / 17,146 total ≈ 10.2%
- $G_{\mathcal{M}}$: 9,145 isolated / 134,946 total ≈ 6.8%

这些isolated users虽然被识别为Creators/Members，但在user-to-user维度中既没有follow也没有block任何人——他们可能是**lurkers**，仅通过community participation参与平台生态。

**485 edges in $G_{\mathcal{C}}$和5,502 edges in $G_{\mathcal{M}}$同时有positive和negative sign**——这些是用户先follow后又block的cases，论文按设计保留了block edge。

---

## 十、与Related Work的定位

### 10.1 vs. Mastodon数据集

Mastodon（另一个decentralized platform）也提供了Feed-level数据（Table 1中Create和Join都是•），但**不提供Block数据（×）**。这意味着Mastodon数据集无法研究antagonistic interactions的temporal dynamics，而BlueTempNet的signed network设计填补了这个gap。

### 10.2 vs. Twitter/X数据集

Twitter/X仅提供partial follow data（◦），无timestamps。这使得temporal sequence analysis几乎不可能。BlueTempNet的millisecond-level timestamps使得fine-grained causal inference成为可能——例如，你可以研究"用户A join某个Feed后，多久会follow该Feed的其他members"。

### 10.3 Signed Network Mining

论文引用了Tang et al. [29]的signed network mining survey。在传统signed network研究中（如Epinions, Slashdot），negative edges（dislike, distrust）通常是**explicitly declared**的。Bluesky的block action则更加**consequential**——它不仅表达了negative sentiment，还实际**改变了信息流动的topology**（blocked用户完全不可见）。这使得BlueTempNet的signed edges具有更强的behavioral semantics。

参考：[Signed Network Mining Survey - Tang et al. 2016](https://doi.org/10.1145/2854006)，[Decentralized Social Media - Kleppmann et al. 2024](https://arxiv.org/abs/2402.03239)

---

## 十一、局限性与Future Directions的Intuition

论文承认的关键局限：

1. **Niche bias**：Bluesky用户偏向art等niche topics，这可能不代表general social media population
2. **Platform inertia**：Bluesky常被视为Twitter/X的**补充平台**而非替代，用户的primary engagement可能仍在其他平台
3. **Post-level interactions excluded**：reply, like posts, repost等content interactions未包含，仅关注networking behaviors

Future work方向中，最有趣的是**blocking对platform resilience的影响**[11][12][16]。当新用户大量涌入时（如public launch），block rate的急剧上升是否会形成**fragmented social bubbles**？还是说block作为social sanction mechanism，实际上**维护了community cohesion**（通过排除disruptive actors）？

Global structural balance的evolution[8]将是一个关键metric——如果balanced triangle比例随时间上升，说明社会结构趋向stable equilibrium；如果下降，可能预示polarization加剧。

参考：[Antagonistic Ties in Social Networks - Ghasemian & Christakis 2024](https://doi.org/10.1073/pnas.2306646121)，[Global Structural Balance - Facchetti et al. 2011](https://doi.org/10.1073/pnas.1109521108)

---

## 十二、技术实现细节

### 12.1 数据格式

- **Graph存储**: GEXF format (Graph Exchange XML Format)，通过NetworkX 2.6.3处理
- **DiGraph**: 用于$G_{\mathcal{C}}$和$G_{\mathcal{M}}$，edge有`sign`和`time`属性
- **Graph**: 用于$G_{\mathcal{A}}$，node有type属性
- **MultiGraph**: 整合三个维度，undirected edges转为bidirectional edges

### 12.2 时间戳精度

所有timestamp格式为ISO 8601（`2023-09-22T09:32:17.974Z`），UTC时区，**millisecond precision**。这在social network datasets中极为罕见——大多数数据集仅提供date-level或hour-level resolution。

Millisecond precision的value在于：它使得**simultaneity detection**和**fine-grained temporal ordering**成为可能。例如，你可以区分"用户在block某人之后0.5秒follow了第三人"和"用户在follow第三人之后0.5秒block了某人"——这种ordering差异在causal inference中有根本意义。

### 12.3 可复现性

所有scripts和dataset公开在DOI repository (10.21227/yrsy-ee91)，Python 3实现。

---

## 总结：构建Intuition

BlueTempNet的核心价值可以提炼为三个层次的intuition：

**Layer 1 - Data Advantage**: Bluesky的decentralized architecture（ATproto + PDS）产生了unprecedented data transparency——所有social interactions都有millisecond timestamps，包括通常被platforms隐藏的block actions。

**Layer 2 - Structural Insight**: Multi-network design揭示了user behavior的hierarchical structure——user-to-user ties（follow/block）是稳定的infrastructure layer，user-to-community ties（create/join）是volatile的service layer。Public launch shock放大了这一差异。

**Layer 3 - Ecosystem Dynamics**: Feed economics表现出attention asymmetry——niche content creation是democratic的（很多人创建art Feeds），但attention流向是hierarchical的（discover类Feed和identity-affirming communities获得最多likes）。这种asymmetry可能是所有social media的universal property，但在Bluesky的user-driven algorithm paradigm下尤为清晰。

**最终，BlueTempNet为研究"没有platform algorithm干预下的用户行为"提供了理想testbed**——这是computational social science领域长期缺失的关键变量。
