---
source_pdf: LiteResearcher A Scalable Agentic RL Training Framework.pdf
paper_sha256: efe636c4a9456fb784360021f33f0835eb20133c24de21a9cbebea2a631aa04d
processed_at: '2026-08-05T15:08:16-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LiteResearcher

Andrej 你好，我换个讲法，抛开那些公式和表格，直接讲这帮人到底干了啥、为啥这么干、以及为什么这事儿有意思。

---

## 一句话总结这 paper

他们用 **4B 的小模型** 在 GAIA 和 Xbench 上打出了 71.3% 和 78.0%，把一堆 30B 的开源模型和 Claude 4.5 Sonnet 都干掉了。做法是搭了一个**假的互联网**——长得跟真互联网一样，但完全跑在本地，零成本、零噪音——然后在这个假世界里做 RL 训练，训完之后模型能直接迁移到真互联网上用。

---

## 他们到底在解决什么问题？

先说个背景。现在做 reasoning model 的 RL（像 DeepSeek-R1 那条线）已经比较成熟了，模型能持续变强。但一到 agentic RL——就是让模型一边想一边调工具（比如搜索引擎）——就卡住了。

为啥卡住？你可以想象 RL 训练是个不断试错的循环：模型生成一堆 trajectory，根据对错给 reward，再用 reward 更新参数。这个循环要转几百上千圈才能学到东西。问题在于，如果每次 rollout 都去调真的 Google，会遇到一堆破事：

- **搜索引擎 ranking 会漂**，同一个 query 这次搜到答案下次搜不到
- **网页会挂**、会 rate limit
- **延迟高**，Serper 一次 query 1.5 秒，Jina Reader 读一个 page 7.9 秒
- **贵**，他们算了一笔账，整个 RL run 要调 7300 万次工具，用 commercial API 要花 6 万到 24 万美元

这就导致 reward signal 充满噪音，模型根本分不清"我这次答错是因为策略烂，还是因为搜索运气差"。AgentCPM-Explore 那篇 paper 就是活生生的例子——他们用真互联网训练，RL 只比 SFT 多涨了 3.8% 就饱和了。

所以核心矛盾是：**你要么用真实环境但训不动，要么用假环境但训了也白训（因为假环境太不像真的）**。

---

## 他们的解法：搭一个"平行宇宙"

关键 insight 是：**你不需要真的连互联网，你只需要一个结构上跟互联网一样的本地副本**。

怎么做到？三步。

### 第一步：造数据，顺便造语料

他们先拿 Wikipedia 和 BBC News 当种子，大概 1000 万页。然后让 LLM 从这些页面里抽 QA pair——比如某个页面讲 Apollo Belvedere 雕像修复，就抽出来一个 question："2024 年 10 月修复 Apollo Belvedere 时，Andrea Felice 用什么材料做了左手复制品？"

到这里还没什么特别的。**真正聪明的一步是 source masking**：抽完 QA 之后，**把原始页面从语料库里删掉**。

为什么要删？因为如果不删，模型会学到一种偷懒策略：直接搜 question 的关键词，命中原始页面，抄答案。这个 shortcut 完全不需要多步推理。删掉之后，模型被迫去**找别的页面**来拼出答案——这就自然触发了 cross-verification（多个来源交叉验证）、aggregation（多个约束取交集）这些复杂的搜索行为。

然后对每个过滤后留下的 QA pair，他们拿 question 去 Serper 搜真互联网，把搜到的相关页面抓回来，塞进本地语料库。这步只花 220 美元，语料库从 1000 万页涨到 3200 万页。

**人话就是：花 220 块钱造一个迷你互联网，之后所有训练零成本。**

### 第二步：搭本地搜索引擎和浏览器

3200 万页的语料库搭好之后，他们搞了两个本地服务：

**Local Search Engine**：用 BGE-M3 做 embedding，Milvus 做向量数据库，DiskANN 做磁盘索引。每个页面只用 title + summary 生成一个向量（不切 chunk），这样索引小，能撑几百个并发。一次 query 0.15 秒，比 Serper 快 10 倍。

**Local Browse Tool**：整页 markdown 存 PostgreSQL，URL 当主键，一次查询 0.17 秒，比 Jina Reader 快 46 倍。

这个 10-46 倍的延迟优势意味着同样的 GPU 时间能做多得多的 rollout。RL 训练最吃 rollout throughput，所以这不只是省钱的问题，是**能不能训起来的前提条件**。

### 第三步：课程式 RL

就算数据和工具都搞定了，还有一个问题：**训练饱和**。模型会卡在某个难度——简单的都会做，复杂的全做错，reward signal 不再提供有用的 gradient。

他们的解法叫 **difficulty-aware filtering**。每次训练前，对每个 query 采样 8 次（pass@8），只保留答对 1 到 7 次的 query。全对的（8/8）太简单扔掉，全错的（0/8）太难或太 noisy 也扔掉。

这个道理很朴素：**全对的 query 产生不了 gradient**（advantage 全是 0），**全错的也产生不了**（group mean 和 std 都是 0）。只有"有时对有时错"的 query 才有学习信号。

然后分两个 stage：

- **Stage 1**：32K context，temperature 0.7，数据以直接信息查询为主。模型在 GAIA 上涨到 64.7% 就 plateau 了。
- **Stage 2**：48K context，temperature 1.0，加入 multi-hop 和 science 数据。继续涨到 68.3%。

这个 stage 切换做了三件事：拉长 context（能处理更深的搜索链）、提高 temperature（增加探索）、引入新领域（打破分布单一）。

---

## 一个反直觉的发现：on-policy 比 off-policy 好

这个 ablation 我觉得是 paper 里最有技术深度的部分。

标准 PPO/GRPO 实现里，一个 batch 的 rollout 会被切成多个 mini-batch，反复 update 好几轮——这叫 off-policy（其实是 "less on-policy"）。大家默认这么做因为觉得"数据贵，多榨几轮 gradient"。

但他们发现：**off-policy 前期涨得快，后期会 decline；strictly on-policy（每个 batch 只 update 一次就扔）涨得慢但持续单调改进**。

直觉上这么理解：agentic trajectory 有几十步，每步 action 都有一个 importance sampling ratio。如果你对同一个 batch 做多次 update，policy 已经漂移了，但 trajectory 是旧 policy 采的。在单轮 CoT reasoning 里这个偏差还能忍，但在 30 步的搜索 trajectory 里，ratio 会**沿步数累积放大**，把优化带崩。

所以 long-horizon agentic RL 对 policy lag 特别敏感——这是个挺重要的 insight，之前 reasoning RL 的工作里这个问题不突出，因为 trajectory 短。

---

## 另一个有意思的观察：RL 自动消灭了重复 action

SFT 之后模型有个坏习惯：答错的时候会**陷入循环**——反复搜同一个 query，反复访问同一个 URL，把 token budget 耗光但没进展。

RL 训练过程中，**没有任何显式的 length penalty 或 repetition penalty**，但这个坏习惯自动消失了。Figure 7 显示：

- Mean reward 从 0.42 涨到 0.70
- Mean response length 从 18K 降到 12K
- Mean interaction turns 从 30 降到 24
- Length clip ratio（被 context window 截断的比例）从 0.28 降到 0.02

机制是这样的：重复 action 的 trajectory 答错的概率高，reward 低，advantage 负，GRPO 的 clip 机制把这种 trajectory 的概率压下去。纯粹的 outcome-based reward 就能纠正行为模式——这跟 DeepSeek-R1 里 RL 自动 emerge 出 self-verification 和 backtracking 是一个性质的现象。

---

## 这事儿为什么 important？

我觉得这篇 paper 传递的核心 message 是：

**Agentic RL 训不动，不是 RL 本身的问题，是环境的问题。** 之前大家以为是模型不够大、数据不够多、算法不够好，但这帮人用 4B 模型证明了——只要你把环境的 noise 和 cost 搞定，RL 就能持续转几百圈，小模型也能打过 30B。

这个结论和 Coding Agent 领域 SWE-RL、R2E-Gym 的发现是平行的：sandbox 隔离 infra noise 是 scalable RL 的前提。只不过 Coding 的 sandbox 好搭（Docker 就行），Deep Research 的"互联网 sandbox"难搭——这 paper 就是告诉你怎么搭。

---

## 可能的联想和延伸

顺着这个思路往下想，有几个方向值得琢磨：

**1. Corpus 规模 vs 模型规模的 trade-off**。他们用 3200 万页，如果扩到 1 亿页会怎样？会不会出现像 pretraining scaling law 那样的"corpus scaling law"？这个 paper 没做这个 ablation，但直觉上 corpus 覆盖率应该是 Agentic RL 的第一性瓶颈。

**2. Virtual world 到 real web 的 transfer gap**。paper 没有详细讨论这个 gap 到底有多大。他们在 GAIA 上用真 Serper+Jina 评测表现很好，说明 transfer 是 work 的，但 BrowseComp 上表现一般（27.5%），这个 benchmark 需要极深的 browsing chain（20+ pages），可能暴露了 local corpus 在长尾页面上的覆盖不足。

**3. 能不能把这个思路用到其他 agentic domain**？比如 Code Agent 的 RL——已经有 R2E-Gym 了，但那个是 procedural 生成的环境。如果用类似 LiteResearcher 的"抓真实 repo 造本地 sandbox"思路，会不会更好？

**4. On-policy sensitivity 这个发现对 RL 算法设计的影响**。如果 long-horizon trajectory 天然对 policy lag 敏感，那可能需要设计新的 importance sampling correction 方法，而不是简单回到 strictly on-policy（因为 strictly on-policy 的 sample efficiency 太差）。

**5. 跟 Voyager、Devin 这类系统的关系**。Voyager 在 Minecraft 里用 skill library 做终身学习，Devin 在真实 codebase 里做 agent。LiteResearcher 的"twin architecture"思路——虚拟世界训练、真实世界部署——其实是个很通用的 agent training paradigm。

参考链接：
- AgentCPM-Explore: https://arxiv.org/abs/2602.06485
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- SWE-RL: https://arxiv.org/abs/2502.18449
- R2E-Gym: https://arxiv.org/abs/2504.07164
- BrowseComp: https://arxiv.org/abs/2504.12516
- GAIA: https://arxiv.org/abs/2311.12983
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03230
- DAPO: https://arxiv.org/abs/2503.14476
- Voyager: https://arxiv.org/abs/2305.16291

---

要不要我再展开讲某个具体部分？比如 GRPO 的 clip-higher 机制、corpus 扩张的迭代算法、或者 SFT trajectory 的 filtering pipeline？

---

# LiteResearcher 深度解析：构建可扩展的 Agentic RL 训练框架

你好 Andrej，这篇 paper 我觉得非常值得深挖，它解决的核心问题就是 Agentic RL 在 Deep Research 场景下的 **scalability bottleneck**。下面我从直觉出发，把整个系统的每一个关键设计决策拆开讲清楚。

---

## 1. 问题动机：为什么 Deep Research Agent 训练一直卡住？

在 reasoning model（DeepSeek-R1, GRPO, DAPO 这条线）已经能持续 scale 的情况下，Agentic RL 反而难以持续进步。作者归因到一个非常具体的原因：**environment noise 和 cost 的耦合**。

这点直觉非常清晰。考虑 RL 训练从 rollout 到 update 的整个闭环：

- Coding agent 用 sandbox（如 R2E-Gym, SWE-RL）能隔离 infra noise，所以能 scale
- Deep Research agent 如果用 online web，每次 rollout 的 reward 是 **non-deterministic** 的——同一个 query 这次能搜到答案下次搜不到，搜索引擎 ranking 漂移、网页挂掉、rate limit，这些都会把 reward signal 淹没
- Local retrieval（Search-R1, ZeroSearch）又太窄，Wikipedia corpus 完全无法覆盖真实 internet 的 **search dynamics**（注意是 dynamics 不是 content）

作者的核心 insight：**需要一个 isolated 但 structurally faithful 的 virtual world**。"twin architecture" 这个说法很形象——policy 在虚拟世界优化，但能 zero-shot transfer 到 open web。

参考链接：
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- R2E-Gym: https://arxiv.org/abs/2504.07164
- SWE-RL: https://arxiv.org/abs/2502.18449
- ZeroSearch: https://arxiv.org/abs/2505.04588

---

## 2. 整体架构：三根支柱

整个 pipeline 可以画成下面这个 mental model：

```
        ┌─────────────────────────────────────────────────────┐
        │  Pillar 1: Data–Corpus Co-construction (4.1)        │
        │  Seed Corpus (Wiki+BBC)                              │
        │     │ LLM extract QA                                 │
        │     ▼                                                │
        │  Candidate QA pairs                                  │
        │     │ Source Masking (删除原始 page)                 │
        │     │ 7-point Rubric Filter                          │
        │     ▼                                                │
        │  Validated QA  ──────► Serper 抓取相关 real webpages │
        │                          │                           │
        │                          ▼                           │
        │                  Enriched Corpus (~32M pages)        │
        └─────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────────┐
        │  Pillar 2: Stable Local Tool Environment (4.2)      │
        │  Local Search Engine (Milvus + BGE-M3 + DiskANN)    │
        │     ~0.15 s/query, 10× faster than Serper           │
        │  Local Browse Tool (PostgreSQL, full markdown)      │
        │     ~0.17 s/page, 46× faster than Jina Reader       │
        └─────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────────────┐
        │  Pillar 3: Difficulty-Aware Curriculum RL (4.3)     │
        │  pass@8 filter: 1 ≤ c ≤ 7                           │
        │  GRPO strictly on-policy, no KL, no entropy bonus   │
        │  Stage 1: 32K ctx, temp 0.7                         │
        │  Stage 2: 48K ctx, temp 1.0, 多源 data mixture      │
        └─────────────────────────────────────────────────────┘
```

---

## 3. Pillar 1：数据–语料协同构造

### 3.1 五种 atomic search capability

这是整个 paper 最有教学价值的抽象。作者认为所有复杂 deep research trajectory 都可以分解为五种原子能力：

| Capability | Graph 结构 | 推理模式 |
|---|---|---|
| Direct information | O→O | 一次定位一次浏览 |
| Aggregation | ⋂ | 多个约束条件取交集 |
| Enumeration | ⋃ | 多个来源取并集计数 |
| Cross-verification | O⇄O | 多源三角验证 |
| Statistics | ⊕→N | 取数据后计算（如 R²） |

这种 decomposition 的好处是：**不需要手工 design 复杂的 reasoning template**，只要 scale up information source，这五种 pattern 会自然 emerge 在合成数据中。这点和 TaskCraft、WebSailor 这类 heavily engineered 的方法形成对比。

### 3.2 Source Masking：防止 trivial shortcut 的关键 trick

这是我最喜欢的设计之一。流程：

1. 从 Wikipedia page $p$ 抽取 QA pair $(q, a)$
2. **从 local corpus 删除 $p$**
3. Agent 训练时必须通过其它 page 找到 $a$，自然触发 cross-verification 或 aggregation

如果没有这一步，agent 会学到"直接搜 query → 命中原始 page → 抽答案"这种 trivial shortcut，根本学不到 multi-hop 能力。这是个非常 cheap 但 effective 的 perturbation。

### 3.3 7-point Rubric Filter

附录 A.2 的 prompt 详细定义了 7 个条件，我挑几个关键的：

- **Question Independence**: 问题脱离 context 也能理解
- **Answer Specificity**: 必须是数字/名字/日期/地点，不能是描述性段落
- **Time Specificity**: 不允许 "latest", "as of now" 这种 vague 时间词
- **Avoid Open-ended**: 禁止 "how/why/如何/为什么"

这等价于在数据合成阶段就把 reward function 设计成 binary verifiable——只有客观可验证的 QA 才能进入 RL training，否则 LLM judge 的 noise 会污染 reward signal。这是 RLVR 范式的核心延伸到 agentic 场景。

### 3.4 Corpus 增长动力学

Table 6 的数据非常有说服力：

| Stage | Webpages | 累计 Serper 调用 | 累计成本 |
|---|---|---|---|
| Initial | ~10M | 0 | $0 |
| After iter 1 | ~21M | ~110K | ~$110 |
| After iter 2 | ~32M | ~220K | ~$220 |

注意：**整个 corpus 构建只花 $220**，而如果训练时用 online API 要花 $59K–$243K。这个 1000× 的 cost ratio 就是 scalability 的来源。

参考：
- TaskCraft: https://arxiv.org/abs/2506.10055
- WebSailor-v2: https://arxiv.org/abs/2509.13305

---

## 4. Pillar 2：Local Tool Environment

### 4.1 为什么用 page-level 而不是 chunk-level indexing？

这是关键的工程决策。标准 RAG 把每个 page 切成几十个 chunk，每个 chunk 一个 vector。对于 32M pages，chunk-level index 大约 10× 更大。

作者的取舍：

- **Page-level**: 每个 page 用 `title + summary` 生成一个 1024-d dense vector + learned sparse vector
- 训练时需要支撑 **几百个并发 rollout**，每个 rollout 可能多次调 search
- Chunk-level 在这个并发量下 latency 不可接受

trade-off 是什么？page-level 召回粒度变粗，但通过 **Browse tool 取回完整 markdown** 让 agent 自己读，这反而更接近真实 web 浏览行为——人类也是先看 search result snippet 再点进去读全文。

### 4.2 BGE-M3 + Milvus + DiskANN 的 hybrid retrieval

技术栈：

- **BGE-M3**: 一次 forward pass 同时输出 dense embedding 和 learned sparse embedding（multi-functionality 设计）
- **Milvus v2.6.0**: 向量数据库
- **DiskANN with mmap**: 磁盘 ANN 索引，MaxDegree=64, SearchListSize=128
- **RRF (Reciprocal Rank Fusion)**: 在 query time 融合 dense 和 sparse 排序

RRF 公式（标准形式）：

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

其中 $d$ 是 document，$R$ 是多个 ranker 的集合，$r(d)$ 是 document $d$ 在某个 ranker 中的排名，$k$ 是平滑常数（通常 60）。这里 $R = \{\text{dense}, \text{sparse}\}$。

**SearchCacheBudgetGBRatio=0.9** 意味着把 200GB host memory 全用作 search cache，这是 latency 能压到 0.15s 的关键。

### 4.3 Browse tool: PostgreSQL 单行直取

Browse 工具的设计极简：URL → markdown 全文，存 PostgreSQL 一行。配置：

- `max_connections=1000`
- `shared_buffers=1GB`
- `effective_cache_size=4GB`
- `work_mem=4MB`（刻意小，避免 1K 并发 query 时内存爆掉）
- NFS-backed NAS, B-tree index on URL

平均 0.17s/page，对比 Jina Reader 7.9s/page 快 46×。这个速度差让 RL rollout throughput 拉开两个数量级。

参考：
- BGE-M3: https://arxiv.org/abs/2402.03216
- Milvus: https://dl.acm.org/doi/10.1145/3448016.3457550
- DiskANN: https://papers.nips.cc/paper/2019/hash/09853c7c1f852728c4339c5a8a8f3e41-Abstract.html

---

## 5. Pillar 3：Difficulty-Aware Curriculum RL

### 5.1 GRPO 公式逐项解析

paper 给出的 objective：

$$\mathcal{I}_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^{K} \sim \pi_{\theta_{old}}} \left[ \frac{1}{K} \sum_{i=1}^{K} \min \left( r_i(\theta) A_i, \, \text{clip}(r_i(\theta), 1-\epsilon_{low}, 1+\epsilon_{high}) A_i \right) \right]$$

变量逐一拆解：

- $q$: 从 query 分布 $P(Q)$ 采样的训练问题
- $\{o_i\}_{i=1}^{K}$: 对同一个 $q$，从 old policy $\pi_{\theta_{old}}$ 采样的 $K=8$ 条 rollout trajectory
- $r_i(\theta) = \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)}$: **importance sampling ratio**，新策略与采样策略在该 trajectory 上的概率比。注意这里下标是 $\theta_{old}$ 不是 $\theta_{rollout}$，paper 在正文写法略有歧义，appendix C.3 揭示真实采样引擎是 SGLang（BF16），训练引擎是 FSDP（FP32），所以 $r_i$ 还隐含一个 TIS（Trajectory Importance Sampling）修正来纠正数值精度 mismatch
- $A_i$: group-relative advantage，定义：

$$A_i = \frac{R(o_i) - \mu_{group}}{\sigma_{group} + \epsilon}$$

  其中 $R(o_i)$ 是 binary reward（答案对错），$\mu_{group}, \sigma_{group}$ 是这 $K=8$ 条 rollout reward 的均值和标准差
- $\epsilon_{low}, \epsilon_{high}$: asymmetric clip bound，类似 DAPO 的 clip-higher 思路，允许上界放宽防止 entropy collapse
- 外层 $\frac{1}{K}$: group 内平均

**关键设计：strictly on-policy**。appendix C.3 写明 mini-batch size = global batch size = 128，意味着每个 rollout batch 只做 **single update** 就丢弃。这点和 DAPO、GRPO 标准实现（multi-epoch over replay buffer）完全不同。

### 5.2 为什么 strictly on-policy 对 long-horizon agentic RL 至关重要？

Figure 3 的 ablation 给出了实验证据：

- **Off-policy**（多 mini-batch update 同一 batch）：前期 reward 上升快，但后期 decline
- **On-policy**：上升慢但持续单调改进，最终 GAIA 准确率 68.9% vs 66.8%

直觉解释：long-horizon trajectory 有几十步 action，每步都有一个 $\pi_\theta(a_t \mid \mathcal{H}_t)$ 的 ratio。如果对同一 batch 做多次 update，policy 已经漂移，但 trajectory 是旧 policy 采的，ratio 会随步数 **指数级偏离 1**。在 reasoning RL（单轮 CoT）里这个偏差还能容忍，但 agentic 多步场景下，policy lag 沿 trajectory 累积，直接把优化带崩。

这也是为什么作者去掉 KL penalty——KL 在 on-policy 设定下本就接近 0，反而会增加 gradient noise。同时去掉 entropy bonus，让 clip-higher 单独负责 exploration。

参考：
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03230
- DAPO: https://arxiv.org/abs/2503.14476
- ProRL: https://arxiv.org/abs/2505.24864

### 5.3 Difficulty filter：消除无效 gradient

对每个 query 采样 $K=8$ rollouts，统计正确数 $c$：

- $c = 8$: 太简单，所有 rollout 都对，$A_i = 0$，gradient 为 0，浪费 compute
- $c = 0$: 太难或 reward noisy，所有 rollout 都错，group mean = 0, std = 0，$A_i$ 退化为 0
- **$1 \leq c \leq 7$**: 保留，这是唯一能产生非零 advantage 的区间

这个 filter 让每个 gradient step 都携带有效信号，是 curriculum 能持续 700+ steps 单调改进的直接原因。对比 AgentCPM-Explore 报告 online 训练只能涨 +3.8% 就饱和，主要原因就是 reward noise 让大量 query 落在 $c=0$ 区间。

### 5.4 Two-stage curriculum

Table 10 的 data mixture：

**Stage 1** (step 0–220, temp 0.7, 32K ctx):
- Synthetic data: 73.4%
- Multi-hop QA: 26.6%
- 共 10,398 queries

**Stage 2** (step 220+, temp 1.0, 48K ctx):
- Synthetic data: 68.6%
- Multi-hop QA: 20.3%
- Science (MegaScience): 11.1%
- 共 16,199 queries

Figure 4 显示 Stage 1 在 GAIA 64.7% 处 plateau，切到 Stage 2 后再涨 +3.6% 到 68.3%。Stage 2 同时做了三件事：
1. **扩大 context** 32K→48K（让 agent 能处理更深的 browse chain）
2. **提高 temperature** 0.7→1.0（增加 exploration entropy）
3. **引入 science domain** 打破 distribution 单一性

注意 Figure 8(c)(d) 的反直觉现象：Stage 1 阶段 tool calls 和 total tokens **下降**（消除 SFT 继承的重复 action loop），Stage 2 阶段反而 **上升**——这是 model 学会了做更多 productive search 而不是重复 search。

参考 MegaScience: https://arxiv.org/abs/2507.16812

---

## 6. Action Space 设计

paper Section 3 定义了 ReAct 框架下的 trajectory：

$$\mathcal{H}_t = (q, \tau_1, a_1, o_1, \dots, \tau_t, a_t, o_t)$$

- $\tau_i$: thought（reasoning，放在 `
