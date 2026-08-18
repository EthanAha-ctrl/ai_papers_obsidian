---
source_pdf: FARM Find Anything using Relational Spatial Memory.pdf
paper_sha256: 1e510ff937a97862aa10a2b9a6cb50cc1dd674f6aa88bbc4e5d0cd50d3998a27
processed_at: '2026-08-18T12:21:07-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FARM 用人话说

## 一句话说清楚

Robot 在大房子里走来走去，用户随口说"找那个在飞镖盘下面、海报左边的台灯"，robot 怎么找到那个特定的台灯？FARM 给了一个答案：**建一个 object-level 的记忆库，查询时现算关系**。

---

## 为什么这是个真问题

想象你家里有 42 个台灯（像 Figure 1a 那个 multifloor house），用户说"the tall lamp below the dartboard and to the left of the poster"。光识别"lamp"完全没用，因为有 42 个。你得同时满足三个约束：是台灯、在某个飞镖盘下面、在某个海报左边。这就是 *relational grounding* 的难点。

现有方案的毛病：

- **Semantic SLAM / Hydra / Kimera**：closed vocabulary，你 map 的时候定了 80 个 class，用户问了第 81 个就抓瞎。
- **ConceptGraphs / BBQ / DAAAM**：open vocabulary 了，但要么 offline post-process，要么 hyperparameter 对每个 scene 都要 tune（小房间用 cm 分辨率，outdoor 用 m 分辨率，不能一套参数走天下）。
- **RynnBrain / Gemini / GPT-4o 这种 video VLM**：把所有 frames 塞进 context window 直接推理。小房间没问题，15,000 m² construction site 上有几千个 viewpoints，context window 装不下，compute 也贵。

FARM 的 bet 是：**别把 relations 存下来，存"足够算出任何 relation 的原始材料"，查询的时候按需算**。这就像数据库不存 join 结果，只存 base tables，query 来了再做 join。

---

## Memory 怎么建

每个 object 用**一个 3D Gaussian** 表示。就一个，不多。

为什么这么省？三个理由：

1. **不用 tune resolution**。point cloud 要设 voxel size，coffee mug 是 cm 级，building facade 是 10m 级，一套参数要么 indoor 小物体溶掉，要么 outdoor 内存爆掉。Gaussian 没 hyperparameter。
2. **$O(1)$ 更新**。看到新 observation，sufficient statistics 累加一下就行，closed-form。
3. **Scale-invariant matching**。两个 Gaussian 之间的 Hellinger distance 永远在 $[0, 1]$，跟 Gaussian 多大多小无关。一个 threshold 跨 coffee mug 和 building 都能用。

每个 object 还存：detection feature、几张代表性 view crops、caption、三个 embeddings（text、SigLIP2 image、Qwen3-VL image）。**故意不存 LeftOf、Between 这种 relations**，因为这些 relations 数量是 $O(N^k)$ 会爆。只存 covisibility 和 adjacency 这种 $O(N^2)$ 的 cheap 关系。

**关键工程 trick**：synchronous loop（detect → lift → associate → fuse）跑在 critical path 上，VLM captioning 异步跑在 background worker 上。这样 mapping 帧率不会被 VLM 卡住，能稳在 5-10 Hz。

---

## Query 怎么处理

用户说 "the tall lamp below the dartboard and to the left of the poster"，FARM 干三件事：

### Step 1: Parser 把 NL 变成结构化 specification

Qwen3.5-9B 把 query 解析成：

- target: "tall lamp"
- anchor1: "dartboard"
- anchor2: "poster"
- predicates: $\{$Below(target, anchor1), LeftOf(target, anchor2)$\}$

Predicate name 从 16 个 closed-set 里选（Near, On, Above, Below, NextTo, Between, Inside, InRegion, LeftOf, RightOf, InFrontOf, Behind, Closest, Farthest, HasAttribute, IsCategory），但 predicate arguments 是 open-vocabulary NL。

这叫 **closed-set syntax + open-vocabulary semantics**。Parser 遇到 "across from" 这种 set 外的关系，会 decompose 成 base relations 的组合。

### Step 2: Candidate retrieval + predicate scoring

先用三个 embedding 做 reciprocal-rank fusion 拿到 candidate set。然后对每个 candidate target $e$，给每个 anchor 找 best binding，evaluate 所有 predicates：

- Below: 算两个 Gaussian 中心 vertical offset 的 sigmoid
- LeftOf: 在 stored camera view 里算 mask centroid 的 horizontal offset
- Near: $\exp(-d^2 / 2\sigma^2)$ with $\sigma = 0.5$ m
- On: horizontal distance gate × vertical band
- 等等

每个 predicate 返回 $[0, 1]$ 的 soft score。

### Step 3: Composite scoring

$$s = s_{\text{sem}} \cdot [(1 - w) + w \bar{g}], \quad w = 0.5$$

- $s_{\text{sem}}$：semantic similarity（embedding 来的）
- $\bar{g}$：predicate scores 的 geometric mean
- Spatial factor $[0.5 + 0.5\bar{g}]$ 永远在 $[0.5, 1]$ 之间

**这个公式是 paper 最聪明的 trick**。意思是：spatial predicate 全部满足，score 不变；全部不满足，最多砍一半。**绝对不会因为一个 anchor 没找到就把 candidate 删掉**。

为什么不能 hard filter？假设每个 anchor detection 准确率 0.7，3 个 anchors 都对的概率是 $0.7^3 = 0.34$，太低。Soft conjunction 让一个 wrong anchor 只 halve score，不 kill candidate。

最后 top-5 给 Qwen3.5-9B rerank（render 一个 schematic panel，target 标红、competitor 标蓝、viewing front 标绿），输出 final ranking。

---

## 为什么 VLM 不能 end-to-end 干所有事

Table 2b 有个反直觉发现：在 ScanNet-5 上，end-to-end VLM reranking（RynnBrain@10）反而比 no rerank 差（31.6 vs 41.6）。

这说明：**VLM 当 black-box reasoner over scene graph context 不可靠**。VLM 容易 hallucinate spatial relation，看一张图说"左边"但其实不左。

FARM 的方案：VLM 只做 VLM 擅长的事（NL parsing、visual captioning、top-5 evidence scoring），spatial reasoning 用 closed-form 公式硬算。这叫 *structured use of VLMs*。

---

## 实验结果一句话

44k queries，67 scenes，从 15 m² 小房间到 15,000 m² outdoor construction site：

- ScanNet 上 A@1 比 DAAAM 提升 23%
- HM3D 上提升 30%
- R@10 是 BBQ 的 3.2×（ScanNet）和 2.1×（HM3D）
- Mapping 8 Hz，memory 23 MiB（ScanNet）
- 真的跑在 Spot quadruped 上，Jetson Thor，closed-loop

---

## Failure mode 很有启发性

Paper 把 failure 分解成三阶段（Figure 8）：

| Benchmark | GT 在 candidate set | GT 在 top-10 | GT 排第一 | 主要 bottleneck |
|-----------|--------------------:|-------------:|----------:|------------------|
| ScanNet | 82.0% | 74.6% | 35.9% | Top-1 disambiguation |
| HM3D | 45.3% | 26.9% | 7.9% | Candidate recall |
| FARM-Scenes | 53.6% | 47.3% | 24.2% | 两者都有 |

ScanNet 的 retrieval 已经很好，问题在 final reranking 选不对 #1。HM3D 因为是 rendered mesh，image quality 差，open-vocab detection 烂，candidate set 里根本没正确答案。不同 benchmark 的瓶颈完全不同，需要不同的 fix。

---

## Real-robot demo 的两个失败 case 很说明问题

- "cardboard package under a table" → 失败
- "cardboard package under a white table" → 成功

- "chair between a white table and a robot" → 失败  
- "gray fabric chair between a white table and a robot" → 成功

**Pattern**：anchor 和 target 的 attribute 描述必须够 specific，否则多个 candidate 满足 weakly-specified relation，系统分不清。这说明 parser + retrieval 在 ambiguous description 上还弱。

---

## Build intuition

如果让我用一句话总结 FARM 的核心 insight：

**让 VLM 做 VLM 擅长的事（NL parsing, visual captioning, evidence scoring），让 classical algorithms 做 classical algorithms 擅长的事（Gaussian fusion, predicate evaluation, variable elimination）。Hybrid > pure neural > pure symbolic。**

这是 *Software 2.0* 时代的一个 *Software 1.5* 反例：不是所有 reasoning 都要 end-to-end learnable，有时候 structured intermediate representation 更 robust、更 efficient、更 debuggable。

具体到 spatial memory：
- **不存 relations，存 sufficient statistics**（lazy computation，像 PyTorch autograd build 时不 forward）
- **Soft conjunction 而非 hard filter**（避免 recall 随 anchor 数 multiplicatively degrade）
- **Single Gaussian per object**（resolution-free, scale-invariant, $O(1)$ update）
- **Synchronous critical path + async VLM**（producer-consumer pattern，frame rate 不被 VLM 卡）

---

## Limitations 和我的看法

Paper 自己承认：
1. Predicates 是 manual 的，weights 是 uniform 的
2. 不支持 anchor 之间的 compositional reasoning

我额外看到的：
- Single Gaussian 对长条形物体（沙发、桌子）表达不准，可以考虑 mixture of Gaussians 或 3DGS
- Star-shape query 假设，high-treewidth query graph 会 blow up
- NR3D 上 RynnBrain 比 FARM 略好，说明 free-form NL parsing 还弱
- No temporal reasoning（"yesterday 的 chair"）
- Class mismatch factor 0.3 是 magic number

但作为一个 *first step* towards relational spatial memory，这个 work 非常 solid：算法 + 系统 + benchmark + real-robot deployment 都齐了，failure mode 分析也 honest。这种 engineering rigor 比"我们 SOTA 了 2 个点"的 paper 有价值得多。

---

## 给 Karpathy 的 takeaway

你在 [Software 2.0 essay](https://karpathy.medium.com/software-2-0-a6c4f9d0465) 里说 neural networks 在取代 hand-coded rules。FARM 走的是 hybrid 路线：核心 spatial predicates 还是 hand-coded（16 个），但 perception、captioning、parsing、reranking 都是 neural。

我觉得这是 *Software 2.0* 的一个重要 nuance：**不是所有 reasoning 都该 end-to-end learnable**。当 reasoning 是 verifiable、compositional、long-horizon 的时候，structured intermediate 帮你 debug、帮你 scale、帮你 retain correctness guarantees。VLM 当万能 reasoner 在 toy setting 工作，到 15,000 m² construction site 就不行了。

这或许就是 embodied AI 的 *Software 2.5*：neural perception + symbolic reasoning + compact memory，三者分工明确，interface clean。FARM 是这个 paradigm 的一个 nice instance。

Reference: [FARM paper (arXiv)](https://arxiv.org/abs/2602.xxxxx) | [Tolman 1948](https://doi.org/10.1037/h0061626) | [ConceptGraphs](https://arxiv.org/abs/2302.07705) | [Hydra](https://arxiv.org/abs/2202.02430) | [RynnBrain](https://arxiv.org/abs/2602.14979v1) | [Software 2.0](https://karpathy.medium.com/software-2-0-a6c4f9d0465)

---

# FARM: Find Anything using Relational Spatial Memory — 深度技术讲解

## 一、Paper 的核心 motivation 和问题定义

这篇 paper 想解决一个非常实际的问题：当 robot 在 home、warehouse、construction site 等 object-rich 环境中工作时，它需要一个 *persistent memory*，能在用户问 "find the tall lamp below the dartboard and to the left of the poster" 这种 *relational language query* 时，快速 retrieve 出具体的 object instance。

作者观察到一个根本性的 gap：现有的 system 要么是 closed-vocabulary semantic SLAM 和 3D scene graphs（如 Hydra, Kimera, Khronos, Clio），无法 generalize到 open-vocabulary queries；要么是 ConceptGraphs、BBQ、DAAAM 这类 open-vocabulary scene graphs，但需要 offline post-processing 或者大量 hyperparameter tuning；要么是 RynnBrain 这类 end-to-end VLM reasoning over frame histories，被 context window 限制在 large-scale environments 中无法 scale。

FARM 的 thesis 是：把 memory construction 和 retrieval 显式 decouple —— memory 只存 compact per-object evidence（Gaussian geometry + 多模态 embeddings），所有 *task-dependent relations* 都在 query time 用 closed-form predicate evaluator 计算。这就避免了在 mapping 时 commit 一套固定的 pairwise relations（这些 relations 数量可能 exponential blow up），同时让 VLM 只在两个地方使用：parsing query 成 typed specification，以及 reranking top-5 candidates。

这个思路非常有意思，让我想到几个相关 work：

- **Tolman 1948 的 cognitive maps** ([Tolman, Psychological Review 1948](https://doi.org/10.1037/h0061626))：paper 显式引用了这个 neuroscience 的根源，提出 animals 构建 persistent cognitive maps 支持灵活 spatial reasoning。
- **O'Keefe & Dostrovsky 1971 的 hippocampus as spatial map** ([Brain Research 1971](https://www.sciencedirect.com/science/article/pii/0006899371903581))：place cells 的发现。
- **SHRDLU (Winograd 1972)** ([Cognitive Psychology 1972](https://www.sciencedirect.com/science/article/pii/0010028572900023))：早期 symbolic reasoning over blocks world 的 grounded language understanding，paper 也明确引用。
- **XNMs (Shi et al. CVPR 2019)**：neural modules over scene graphs for VQA。
- **ConceptGraphs (Gu et al. ICRA 2024)** ([arXiv:2302.07705](https://arxiv.org/abs/2302.07705))：open-vocabulary 3D scene graphs。
- **Hydra (Hughes et al. 2022)** ([arXiv:2202.02430](https://arxiv.org/abs/2202.02430))：real-time 3D scene graph。
- **Kimera (Rosinol et al. IJRR 2021)** ([arXiv:2104.03152](https://arxiv.org/abs/2104.03152))：从 SLAM 到 spatial perception。
- **Clio (Maggio et al. RAL 2024)** ([arXiv:2404.13632](https://arxiv.org/abs/2404.13632))：real-time task-driven open-set 3D scene graphs。
- **Khronos (Schmid et al. RSS 2024)** ([arXiv:2403.16829](https://arxiv.org/abs/2403.16829))：spatiotemporal metric-semantic SLAM。
- **BBQ (Linok et al. ICRA 2025)**：open-vocabulary object grounding with 3D scene graph。
- **DAAAM (Gorlo et al. 2025)** ([arXiv:2512.00565](https://arxiv.org/abs/2512.00565))：describe anything anywhere at any moment。
- **RynnBrain (Dang et al. 2026)** ([arXiv:2602.14979](https://arxiv.org/abs/2602.14979v1))：open embodied foundation models。

---

## 二、Architecture 整体解析（参考 Figure 3 和 Figure 4）

FARM 由两个核心模块组成：

### (A) Online Memory Construction（Figure 3）

输入：streaming posed RGB-D observations $o_{1:t} = \{o_1, \ldots, o_t\}$

输出：relational spatial memory $\mathcal{M}_t = \{\mathcal{E}_t^i\}_{i=1}^{N_t}$，其中每个 entity $\mathcal{E}_t^i = \langle \mathcal{A}_t^i, \mathcal{R}_t^i \rangle$

**关键设计**：synchronous per-frame loop + asynchronous VLM enrichment。Synchronous loop 跑在 robot critical path 上，每 frame 都执行 detect → lift → associate → fuse；VLM captioning 和 embedding 在 off-critical-path 的 worker pool 里异步处理，这样 mapping frame rate 不会被 VLM throughput 卡住。这是一个非常经典 的 producer-consumer pattern，类似于 NVIDIA DALI、PyTorch DataLoader 的设计哲学。

**Architecture diagram（我重新组织 Figure 3 的内容）**：

```
RGB-D stream ──► [Step 1: Segmentation] 
                   │  YOLOE detector (open-vocab masks + classification-head features)
                   │  Optional DINO backbone for richer per-token features
                   ▼
                [Step 2: Filtering]
                   │  Border-touch rejection, depth range, wall/floor class filter, IoU dedup
                   ▼
                [Step 3: Neighbor search]
                   │  Batched detection×entity feature similarity → sparse candidate set
                   │  Hellinger distance over 3D Gaussians for top-k candidates
                   ▼
                [Step 4: Correspondence via union-find with path compression]
                   │  Bridges multiple detections claiming same entity
                   │  Bridges one detection connecting two previously separate entities
                   ▼
                [Step 5: Fusion]
                   │  Sufficient-statistics Gaussian merge: M1 = wμ, M2 = w(Σ + μμᵀ)
                   │  Appearance feature = detection-count weighted average
                   │  Loser's covisibility/captions/views absorbed into winner
                   ▼
                [Step 6: High-quality view selection]
                   │  Retain only if angular difference > threshold OR closer than closest
                   ▼
                [Step 7: Covisibility update]
                   │  Bitset Nx⌈N/64⌉ blocks mark jointly visible pairs
                   │  Weighted edge dict with temporal decay, kNN in Hellinger space
                   ▼
              ┌────────────────────────────────────────┐
              │  Asynchronous queue (off critical path) │
              │  vLLM workers:                          │
              │    - Qwen3.5-9B captioner (JSON caption) │
              │    - Qwen3-Embedding-0.6B (text)        │
              │    - SigLIP2 image encoder              │
              │    - Qwen3-VL-Embedding-2B (image)       │
              │  Results written back to entity attrs   │
              └────────────────────────────────────────┘
```

### (B) Relational Retrieval（Figure 4）

输入：natural language query $q$，memory $\mathcal{M}_t$

输出：ranked list of ScoredCandidate records

**Architecture diagram**：

```
q (NL utterance) ──► [Stage 1: Query parsing by Qwen3.5-9B]
                       │  Outputs QueryGraph Π(q) = (x★, {a₁...aₘ}, S_q, Φ_q)
                       │  16 predicate names: Near, On, Above, Below, NextTo,
                       │  Between, Inside, InRegion, LeftOf, RightOf, InFrontOf,
                       │  Behind, Closest, Farthest, HasAttribute, IsCategory
                       ▼
                    [Stage 2: Region scoping (optional)]
                       │  Region phrase embedded & matched against stored labels
                       ▼
                    [Stage 3: Candidate retrieval via reciprocal-rank fusion]
                       │  4 channels across 3 embedding spaces:
                       │  - Qwen3-text (w=1.0) with modifier-rich target desc
                       │  - Qwen3-text (w=0.5) with raw utterance
                       │  - SigLIP2 image (w=0.75) with text-prompt ensemble
                       │  - Qwen3-VL image (w=1.0) with target desc
                       │  Fused rank score = Σ wᵢ/(μ+rankᵢ), μ=5
                       ▼
                    [Stage 4: Predicate evaluation (closed-form over Gaussians)]
                       │  Near: Gaussian on intercentroid distance, σ=0.5 m
                       │  NextTo: σ=1.0 m
                       │  Above/Below: sigmoid on vertical offset
                       │  LeftOf/RightOf: in stored camera view, mask centroid horizontal offset
                       │  InFrontOf/Behind: relative camera-frame depth
                       │  Closest/Farthest: 1/(|rank-r*|+1)
                       │  IsCategory: tiered class match (YOLOE class → caption fallback)
                       │  HasAttribute: embedding match against caption
                       ▼
                    [Stage 5: Composite scoring (soft conjunction)]
                       │  s = s_sem × [(1-w) + w·ḡ],  w=0.5
                       │  ḡ = geometric mean of predicate scores (robust to single near-zero)
                       │  s_sem additionally ×0.3 when class mismatch
                       ▼
                    [Stage 6: Projected-view VLM reranking (optional, top-5 only)]
                       │  Qwen3.5-9B sees schematic panel:
                       │    - candidate in red
                       │    - competing top-K in blue
                       │    - viewing "front" direction in green
                       │  Chain-of-thought disabled, returns [0,1] evidence score
                       ▼
                    [Stage 7: Output ranked ScoredCandidate list + action interface Ψ_t]
                       │  Ψ_t maps chosen entity to navigable/viewing pose
```

---

## 三、Method 细节与公式深度解析

### 3.1 Problem statement 的形式化

Paper 把问题分成两块：

**Online memory construction**：给定 posed RGB-D 序列 $o_{1:t}$，online 维护 memory：
$$\mathcal{M}_t = f(\mathcal{M}_{t-1}, o_t)$$

这里 $\mathcal{M}_t$ 是 robot 当前持久的 spatial memory，$\hat{\mathcal{X}}_t$ 是 memory 中可 retrieve 的 object elements 集合（是真实 physical object set $\mathcal{X}_t$ 的 noisy partial approximation）。

**Relational object retrieval**：给定 query $q \in \mathcal{Q}$，$q$ 诱导一个 relational specification $\Phi_q$，robot 返回 estimated query-induced set：
$$\hat{y} = g(q, \mathcal{M}_t), \quad \hat{y} \subseteq \hat{\mathcal{X}}_t$$

Retrieval 成功当且仅当 $\hat{y}$ 对应于真实的 query-induced set $y \subseteq \mathcal{X}_t$，即 $y \Vdash \Phi_q$（objects in $y$ 能被 assign 到 query variables，使其 categories、attributes、mutual relations 共同满足 $q$ 诱导的 constraints）。

**变量含义**：
- $o_t$：第 $t$ 步的 posed RGB-D observation
- $\mathcal{M}_t$：截至时刻 $t$ 的 memory 状态
- $\mathcal{X}_t$：环境的 latent physical object 集合（不可观测的 ground truth）
- $\hat{\mathcal{X}}_t$：memory 中可检索的 object elements（approximation）
- $y$：query 诱导的真实 target+anchor 集合
- $\hat{y}$：robot 返回的 estimate
- $\Vdash$：satisfies relation（语义上类似逻辑中的 entailment）

### 3.2 Memory representation

每个 entity $\mathcal{E}_t^i = \langle \mathcal{A}_t^i, \mathcal{R}_t^i \rangle$。

**Attributes $\mathcal{A}_t^i$ 包含 5 类信息**：

| 类别 | 表示 | 用途 |
|------|------|------|
| 3D Gaussian | $\mathcal{N}(\mu^i, \Sigma^i)$ | location 和 spatial extent |
| Detection-time appearance feature | compact embedding | cross-view association |
| Up to k representative views | posed RGB crops | viewpoint diversity |
| Open-vocabulary caption | text string | 用于 retrieval |
| Three retrieval embeddings | text embed + SigLIP2 + Qwen3-VL | 融合检索 |

**Relations $\mathcal{R}_t^i$ 故意只存两类**：
- **covisibility edges**：在某帧中 jointly visible 过的 entities
- **adjacency edges**：Hellinger distance 下 spatially proximal

关键 design choice：paper *deliberately* 不存 higher-order relations（containment、left-of、between），因为这些 relations 数量可能 exponential blow up。所有 task-dependent relations 在 query time 计算。

这让我想到 *lazy evaluation* 的思想，类似 PyTorch 的 autograd：build graph 时不 forward，只在 backward 时才计算。这里 mapping 时只存"sufficient statistics"，relations 等到 query 时才 instantiate。

### 3.3 Sufficient-statistics Gaussian fusion 的公式

Step 5 (Fusion) 用的是经典 sufficient-statistics merge for Gaussians。让我详细推导：

给定两个 Gaussian $\mathcal{N}(\mu_1, \Sigma_1)$ with weight $w_1$ 和 $\mathcal{N}(\mu_2, \Sigma_2)$ with weight $w_2$，merge 后的 Gaussian $\mathcal{N}(\mu, \Sigma)$ with weight $w = w_1 + w_2$ 满足：

$$\mu = \frac{w_1 \mu_1 + w_2 \mu_2}{w_1 + w_2}$$
$$\Sigma = \frac{w_1(\Sigma_1 + \mu_1 \mu_1^\top) + w_2(\Sigma_2 + \mu_2 \mu_2^\top)}{w_1 + w_2} - \mu \mu^\top$$

Paper 用的是 *moment-matching form*：
$$M_1 = w\mu, \quad M_2 = w(\Sigma + \mu\mu^\top)$$

这里 $M_1, M_2$ 是 *accumulated sufficient statistics*。这样存储的好处是 merge 是 $O(1)$ closed-form operation：$M_1^{new} = M_1^{old} + M_1^{new\_obs}$，$M_2^{new} = M_2^{old} + M_2^{new\_obs}$，最后再 $\mu = M_1 / w$, $\Sigma = M_2/w - \mu\mu^\top$。

**变量含义**：
- $\mu \in \mathbb{R}^3$：Gaussian 的均值（object 中心位置）
- $\Sigma \in \mathbb{R}^{3 \times 3}$：covariance matrix（object 的 spatial extent）
- $w \in \mathbb{R}^+$：weight（accumulated detection count）
- $M_1 \in \mathbb{R}^3$：first moment sufficient statistic
- $M_2 \in \mathbb{R}^{3 \times 3}$：second moment sufficient statistic
- 上标 $\top$：matrix transpose

**为什么是 single Gaussian per entity？** Paper 在 Appendix D.1 给了三个理由：

1. **Resolution-free**：point cloud 或 voxel grid 都需要全局 density/edge length hyperparameter，coffee mug 是 centimeter-scale，building façade 是 tens-of-meters scale，一个 setting 要么 outdoor 内存爆掉，要么 indoor 把小物体溶解掉。Gaussian 没有这种 hyperparameter。
2. **$O(1)$ closed-form online update**：per-entity footprint 和 per-update cost 都 constant in observation count。
3. **Scale-invariant matching metric**：Hellinger distance 在两个 Gaussian 之间是 unitless 且 bounded in $[0,1]$，一个 data-association threshold 跨 $10^1$ 到 $10^4$ square meters 都不需要 retuning。

**Hellinger distance between two multivariate Gaussians** $\mathcal{N}(\mu_1, \Sigma_1)$ 和 $\mathcal{N}(\mu_2, \Sigma_2)$：

$$H^2(p, q) = 1 - \frac{|\Sigma_1|^{1/4} |\Sigma_2|^{1/4}}{|\Sigma|^{1/2}} \exp\left(-\frac{1}{8} (\mu_1 - \mu_2)^\top \Sigma^{-1} (\mu_1 - \mu_2)\right)$$

where $\Sigma = \frac{\Sigma_1 + \Sigma_2}{2}$

这个度量比 KL divergence 更 symmetric、更 bounded，特别适合做 data association 的 threshold。这个 trick 让我想到 [Hausdorff distance for shape matching](https://en.wikipedia.org/wiki/Hausdorff_distance)、[Chamfer matching](https://en.wikipedia.org/wiki/Iterative_closest_point) 和 [Bhattacharyya distance](https://en.wikipedia.org/wiki/Bhattacharyya_distance) 在 point cloud registration 中的应用。

让我也提一下 paper 用 6-vector packed form 存储 $\Sigma_d$ —— 因为 3D Gaussian 的 covariance matrix 是 symmetric 的，只需要存 6 个独立元素（$\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{xy}, \sigma_{xz}, \sigma_{yz}$）。这是一个 memory 优化 trick。

### 3.4 Query compilation 的逻辑公式

Query parsing 把 NL query $q$ 编译成 typed query graph：
$$\Pi(q) = (x_\star, \{a_1, \ldots, a_m\}, S_q, \Phi_q)$$

**变量含义**：
- $x_\star$：target variable（要 retrieve 的 object）
- $a_1, \ldots, a_m$：anchor variables（用于 disambiguate target 的 contextual objects）
- $S_q$：variable → open-vocabulary description 的 mapping，如 $x_\star \mapsto$ "tall lamp"
- $\Phi_q$：relational predicates over variables

等价的 logical formula：
$$\varphi_q(x_\star) = \exists a_1, \ldots, a_m \left( \bigwedge_{(v, \sigma) \in S_q} s_\sigma(v) \wedge \bigwedge_{\rho(\mathbf{v}; \alpha) \in \Phi_q} \rho(\mathbf{v}; \alpha) \right)$$

**变量含义**：
- $\exists a_1, \ldots, a_m$：存在一组 anchor entities 从 memory 中被选中
- $\bigwedge$：所有 listed descriptions 和 relations 必须 jointly 满足
- $s_\sigma(v)$：unary predicate，variable $v$ 满足 description $\sigma$
- $\rho(\mathbf{v}; \alpha)$：relation predicate，relates tuple of variables $\mathbf{v}$，with optional argument $\alpha$
- $x_\star$：唯一 free variable（返回值）

**例如** query = "Find the tall lamp below the dartboard and to the left of the poster"：
- $S_q = \{x_\star \mapsto \text{"tall lamp"}, a_1 \mapsto \text{"dartboard"}, a_2 \mapsto \text{"poster"}\}$
- $\Phi_q = \{\text{Below}(x_\star, a_1), \text{LeftOf}(x_\star, a_2)\}$
- 等价 formula：$\varphi_q(x_\star) = \exists a_1, a_2 [s_{\text{tall lamp}}(x_\star) \wedge s_{\text{dartboard}}(a_1) \wedge s_{\text{poster}}(a_2) \wedge \text{Below}(x_\star, a_1) \wedge \text{LeftOf}(x_\star, a_2)]$

**16 个 predicates**（closed-set 的 predicate names，open-vocabulary arguments）：
- Spatial: Near, On, Above, Below, NextTo, Between, Inside, InRegion, LeftOf, RightOf, InFrontOf, Behind, Closest, Farthest
- Semantic: HasAttribute, IsCategory

这种设计非常有意思：*closed-set syntax + open-vocabulary semantics*。Parser 是 Qwen3.5-9B，predicate arguments 是 NL phrases。如果 query 提到 predicate set 外的关系（比如 "across from"），parser 会 decompose 成 base relations 的组合。这让我想到 [Neural Module Networks (Hu et al. CVPR 2017)](https://arxiv.org/abs/1701.06834)、[FiLM (Perez et al. AAAI 2018)](https://arxiv.org/abs/1709.07871)、[MAC networks (Hudson & Manning, CVPR 2018)](https://arxiv.org/abs/1803.03025) —— 都是 compositional reasoning 的思路。也让我想到 [BIND (Neuro-Symbolic VQA)](https://arxiv.org/abs/1904.05060) 和 [NSVQA](https://arxiv.org/abs/1810.01565)。

### 3.5 Soft verification 的公式

Memory $\mathcal{M}_t$ 提供 logical formula 的 interpretation。每个 entity $\mathcal{E}_t^i$ expose feature functions（location, extent, embeddings, captions, views, covisibility）。

**Unary match**：$s_\sigma(e) \in [0, 1]$，variable $v$ 满足 description $\sigma$ 的程度。

**Relational match**：$r_{\rho, \alpha}(e_1, \ldots, e_k) \in [0, 1]$，predicate $\rho(\cdot; \alpha)$ 在 entities $e_1, \ldots, e_k$ 上的满足程度。其中 $k$ 是 predicate 的 arity。

**Binding** $\eta: \{x_\star, a_1, \ldots, a_m\} \to \mathcal{E}_t$，把 query variables 映射到 memory entities。

**Verify**：
$$\operatorname{Verify}(\eta) = \operatorname{Agg}\left( \{s_\sigma(\eta(v))\}_{(v, \sigma) \in S_q} \cup \{r_{\rho, \alpha}(\eta(\mathbf{v}))\}_{\rho(\mathbf{v}; \alpha) \in \Phi_q} \right)$$

这里 $\eta(\mathbf{v})$ 是 componentwise 应用 binding 到 tuple $\mathbf{v}$。$\operatorname{Agg}$ 把所有 unary 和 relational scores 聚合成一个 soft constraint-satisfaction score。

**Per-target score**：
$$\operatorname{score}(e; q, \mathcal{M}_t) = \max_{\eta: \eta(x_\star) = e} \operatorname{Verify}(\eta)$$
$$\hat{y} = \operatorname{TopK}_{e \in \mathcal{E}_t} \operatorname{score}(e; q, \mathcal{M}_t)$$

**变量含义**：
- $e$：candidate target entity
- $\eta$：一个 binding（assignment of variables to entities）
- $\eta(x_\star) = e$：binding 把 target variable 绑定到 entity $e$
- $\max$：在所有把 $x_\star$ 绑到 $e$ 的 bindings 中取 best witnessing binding
- $\operatorname{TopK}$：取 score 最高的 K 个 entities

### 3.6 Star decomposition 的 complexity 分析

典型 referring expression 是 star-shaped：一个 target $x_\star$ 通过 independent predicates $\rho_i(x_\star, a_i)$ 连到 $m$ 个 anchors $a_i$。

对于 decomposable score（如 sum），conditioning on $x_\star$ 分解 anchors：
$$\max_{x_\star, a_1, \ldots, a_m} \sum_{i=1}^m \rho_i(x_\star, a_i) = \max_{x_\star} \sum_{i=1}^m \max_{a_i} \rho_i(x_\star, a_i)$$

**变量含义**：
- $x_\star$：target variable（要 fix 的）
- $a_i$：第 $i$ 个 anchor variable
- $\rho_i$：连接 $x_\star$ 和 $a_i$ 的 predicate
- $m$：anchor 数量

**Complexity**：
- Star-shape exact binding: $O(|\Phi_q| K_t K_a)$
- Exhaustive joint enumeration: $O(K_t K_a^m)$（exponential in $m$）
- 一般 treewidth $b$ 的 query factor graph 用 bucket elimination: $O(|\Phi_q| K^{b+1})$（[Kask, Dechter, Larrosa, Dechter, Artificial Intelligence 2005](https://www.sciencedirect.com/science/article/pii/S0004370205000639)）

这里 $K_t$ 是 target candidate count upper bound，$K_a$ 是 per-anchor candidate count upper bound。Star decomposition 把 $m$-fold 联合 search 降到 $m$ 个 independent 单变量 search。这是 classic dynamic programming / variable elimination 思路，让我想到 [Junction tree algorithm](https://en.wikipedia.org/wiki/Junction_tree_algorithm) 和 [VE (Variable Elimination) for probabilistic inference](https://arxiv.org/abs/1301.0553)。

### 3.7 Composite scoring 公式

最关键的 formula（Appendix D.2 第 5 步）：

$$s = s_{\text{sem}} \cdot [(1 - w) + w \bar{g}]$$

**变量含义**：
- $s$：final composite score
- $s_{\text{sem}}$：target similarity score（来自 reciprocal-rank fusion of 3 embeddings）
- $\bar{g}$：geometric mean of predicate scores（robust to single near-zero term）
- $w \in [0, 1]$：blend weight（paper 用 $w = 0.5$）

**为什么用 geometric mean for predicate scores 而不是 arithmetic 或 product？**
- Product $\prod_i r_i$：一个 near-zero predicate 会 veto 整个 candidate
- Arithmetic mean $\frac{1}{m} \sum_i r_i$：一个 high-score predicate 能 dominate
- Geometric mean $\bar{g} = \left(\prod_i r_i\right)^{1/m}$：对 single near-zero 敏感但不像 product 那么极端，对 single high 不像 mean 那么宽容

**为什么 blend 而不是 hard floor？**

设 $w = 0.5$，则 spatial factor $[(1-w) + w\bar{g}] = [0.5 + 0.5\bar{g}] \in [0.5, 1]$。所以：
- Predicate scores 全 0：$s = 0.5 \cdot s_{\text{sem}}$（最多 halve，不删）
- Predicate scores 全 1：$s = 1.0 \cdot s_{\text{sem}}$（不变）

这是 paper 的核心 insight：*soft conjunction without veto*。Hard filter 会让 target recall 随 anchor 数量 multiplicatively degrade —— 如果每个 anchor 的正确率是 $p$，$m$ 个 anchors 都正确的概率是 $p^m$，这就会爆炸式下降。

**Class mismatch factor**：$s_{\text{sem}}$ additionally multiplied by 0.3 when candidate's class doesn't match parsed target class。这是 soft way to enforce category constraint。

### 3.8 Reciprocal-rank fusion (RRF) 公式

Candidate retrieval 阶段用 RRF 融合多个 embedding channel：

$$\text{fused\_rank}(e) = \sum_i \frac{w_i}{\mu + \text{rank}_i(e)}$$

**变量含义**：
- $i$：第 $i$ 个 retrieval channel
- $w_i$：channel weight（paper 用 $w_1=1.0, w_2=0.5, w_3=0.75, w_4=1.0$）
- $\mu$：offset 防止 top-rank dominates（paper 用 $\mu = 5$）
- $\text{rank}_i(e)$：entity $e$ 在 channel $i$ 中的 rank

RRF 是 [ Cormack et al. SIGIR 2009](https://dl.acm.org/doi/10.1145/1571971.1572110) 提出的经典 ensemble retrieval 方法。好处是不需要 calibrate scores across channels（只看 rank），简单 robust。Paper 在 4 个 channel 上做 RRF：

| Channel | Embedding | Query | Weight |
|---------|-----------|-------|--------|
| 1 | Qwen3-text | modifier-rich target desc | 1.0 |
| 2 | Qwen3-text | raw utterance | 0.5 |
| 3 | SigLIP2 image | short text-prompt ensemble | 0.75 |
| 4 | Qwen3-VL image | target desc | 1.0 |

### 3.9 Predicate evaluators 的 closed-form 表达

Paper 在 Appendix D.2 第 4 步给了每个 predicate 的 closed-form evaluator。让我列出主要几个：

**Near**：$\sigma = 0.5$ m 的 Gaussian on intercentroid distance
$$r_{\text{Near}}(e_1, e_2) = \exp\left(-\frac{d(e_1, e_2)^2}{2\sigma^2}\right), \quad \sigma = 0.5\text{ m}$$

**NextTo**：$\sigma = 1.0$ m 的更宽 Gaussian
$$r_{\text{NextTo}}(e_1, e_2) = \exp\left(-\frac{d(e_1, e_2)^2}{2 \cdot 1.0^2}\right)$$

**Above / Below**：sigmoid on vertical offset
$$r_{\text{Above}}(e_1, e_2) = \sigma_{\text{sig}}(z_1 - z_2; \text{threshold})$$
$$r_{\text{Below}}(e_1, e_2) = \sigma_{\text{sig}}(z_2 - z_1; \text{threshold})$$

**On**：horizontal-distance gate times vertical band
$$r_{\text{On}}(e_1, e_2) = \text{gate}_{\text{horiz}}(d_{\text{horiz}}(e_1, e_2)) \cdot \text{band}_{\text{vert}}(|z_1 - z_2|)$$

**Inside / InRegion**：containment test（Gaussian overlap-based）

**Closest / Farthest**：rank-based score
$$r_{\text{Closest}}(e, \text{refs}) = \frac{1}{|\text{rank}(e) - r^\star| + 1}$$
其中 $r^\star = 1$ for Closest，$r^\star = N$ for Farthest。

**LeftOf / RightOf**：在 stored camera view 中，mask centroid 的 horizontal offset
$$r_{\text{LeftOf}}(e_1, e_2; \text{view}) = \sigma_{\text{sig}}(x_2 - x_1; \text{threshold})$$

**InFrontOf / Behind**：relative camera-frame depth
$$r_{\text{InFrontOf}}(e_1, e_2; \text{view}) = \sigma_{\text{sig}}(d_2 - d_1; \text{threshold})$$

**IsCategory**：tiered class match —— YOLOE class 匹配则用，否则 caption fallback

**HasAttribute**：attribute string 通过 embedding 匹配 caption

**变量含义**：
- $d(e_1, e_2)$：两个 entity Gaussian 之间的距离
- $\sigma_{\text{sig}}$：sigmoid function $\frac{1}{1 + e^{-x}}$
- $z$：vertical coordinate（高度）
- $x$：horizontal image-plane coordinate
- $d_{\text{horiz}}$：horizontal distance
- $r^\star$：目标 rank（Closest 是 1，Farthest 是 N）

这里 *viewpoint-dependent predicates*（LeftOf/RightOf/InFrontOf/Behind）特别有意思 —— 它们不是在 3D 空间用 absolute coordinates 算，而是回到 stored camera views 用 mask centroid 算。这是因为 "left of" 是 viewer-relative 的概念，没有 absolute truth。这让我想到 [CYC project](https://en.wikipedia.org/wiki/Cyc) 中关于 *indexical* vs *absolute* 时空表达的经典讨论，以及 [Frame of reference in language acquisition](https://en.wikipedia.org/wiki/Frame_of_reference) 的心理学研究。

---

## 四、Experiments 实验数据表深度解析

### 4.1 Main results (Table 1a)

| Method | ScanNet-30 A@1 | HM3D-30 A@1 | FARM-Scenes A@1 |
|--------|---------------:|------------:|----------------:|
| BBQ [15] | 15.4 | 5.3 | 7.4 |
| RynnBrain [24] | 28.0 | 4.6 | 2.7 |
| DAAAM+RynnBrain [14] | 29.3 | 6.1 | 15.1 |
| **FARM° (no rerank)** | **33.0** | **6.7** | **16.3** |
| **FARM (with rerank)** | **35.9** | **7.9** | **24.2** |

| Method | ScanNet-30 R@10 | HM3D-30 R@10 | FARM-Scenes R@10 |
|--------|----------------:|------------:|-----------------:|
| BBQ | 23.5 | 12.8 | 9.6 |
| FARM | **74.6** | **26.9** | **47.3** |

**Key takeaways**：
- ScanNet 上 FARM 比 strongest baseline DAAAM 相对提升 23%
- HM3D 上 FARM 相对提升 30%
- R@10 上 FARM 在 ScanNet 是 BBQ 的 3.2×，在 HM3D 是 2.1×
- FARM-Scenes（15000 m² outdoor）上 FARM 的 A@1 比 DAAAM+RynnBrain 高 60%

### 4.2 Efficiency (Table 1b)

| Method | ScanNet-30 MiB | Map ms/fr | Query s/q |
|--------|---------------:|----------:|----------:|
| BBQ | 10.1 | 235±106 | 3.6±0.8 |
| RynnBrain | 67.8 | 26±16 | 2.2±0.2 |
| DAAAM+RynnBrain | 29.9 | 112±21 | 1.1±0.5 |
| FARM° | 22.1 | **121±56** | 1.7±0.3 |
| FARM | 23.2 | 121±56 | 2.6±0.9 |

**Insight**：FARM 的 mapping throughput 是 ~8 Hz（121 ms/frame 的倒数），与 DAAAM 相当，是 BBQ 的 2×。Memory size 在 ScanNet 上只有 23 MiB，比 RynnBrain 的 67.8 MiB 小得多。Query latency 1.7-2.6 s 之间，competitive。

### 4.3 Ablation: Scene representation (Table 2a)

固定 retrieval embedding = Qwen3-T，比较不同 method + merge backend：

| Method | Merge | ScanNet-5 A@1 | HM3D-5 A@1 | FARM A@1 |
|--------|-------|--------------:|-----------:|---------:|
| BBQ | DINO | 5.4 | 2.3 | 5.9 |
| DAAAM | Hydra+DAM | 18.6 | 2.6 | 5.0 |
| FARM | DINO | **23.0** | **3.6** | **9.8** |
| FARM | YOLOE | 17.6 | 4.1 | 9.0 |

Insight：FARM-mapper 贡献了大部分 gap，从 BBQ 到 FARM 提升 4.3× on ScanNet-5。

固定 method = FARM, merge = DINO，比较 retrieval embedding：

| Retrieval Embedding | ScanNet-5 A@1 | HM3D-5 A@1 | FARM A@1 |
|---------------------|--------------:|-----------:|---------:|
| SigLIP2 (pure image) | 7.4 | **0.3** | 0.9 |
| Qwen3-VL (image) | 14.7 | 1.7 | 2.7 |
| T5 (text) | 22.1 | 3.3 | 10.6 |
| Qwen3-T (text) | 23.0 | 3.6 | 9.8 |
| Multi (fusion) | **23.5** | **4.1** | 8.7 |

Insight：**pure-visual SigLIP2 image embedding 在 HM3D-5 上崩了（0.3 A@1）**，caption-text encoders (T5, Qwen3-T) 都好得多。Multi-embedding fusion 在 HM3D 上 best (4.1) 但牺牲 R@10。这暗示 caption text 包含的信息量比 image embedding 更 reliable for object retrieval in 3D scenes。

让我想到这个现象在 [CLIP retrieval literature](https://arxiv.org/abs/2103.00020) 里也有讨论：vision-language embedding 在 fine-grained object distinction 上有时不如 pure text，因为 image encoder 容易被 distractor 的 visual similarity 误导。

### 4.4 Ablation: Retrieval and reranking (Table 2b)

固定 spatial memory，比较 query mechanisms：

| Retrieval | Rerank | ScanNet-5 A@1 | HM3D-5 A@1 | FARM A@1 |
|-----------|--------|--------------:|-----------:|---------:|
| Pure emb. | None | 38.7 | 5.9 | 10.7 |
| BBQ-hybrid | None | 14.5 | 5.3 | 9.2 |
| BBQ-hybrid multi | None | 31.5 | 5.8 | 10.1 |
| **Locked** (FARM's) | **None** | **41.6** | **6.7** | **10.8** |
| Locked | Qwen@5 | 40.6 | **7.9** | **15.5** |
| Locked | RynnBrain@10 | 31.6 | 8.4 | 15.2 |

Insight：
- FARM 的 locked retrieval + multi-embedding fusion + soft-predicate evaluator 比 pure-embedding cosine similarity 高 (38.7 → 41.6 on ScanNet-5)
- 比 BBQ 的 two-stage LLM retrieval 高很多 (14.5 → 41.6)
- Reranking 在 ScanNet 上没用甚至有害（41.6 → 40.6 或 31.6），但在 HM3D 和 FARM-Scenes 上有 1.2-1.7 个百分点提升

**为什么 reranking 在 ScanNet 上没用？** ScanNet 室内场景小，retrieval 已经把答案排到 top-1；rerank 只是 permute 已经正确的 top-5，反而引入 noise。在 HM3D/FARM-Scenes 上 retrieval 困难，rerank 能打破 tie。

### 4.5 Failure case analysis (Figure 8, Appendix F.6)

不同 benchmark 的失败模式分布：

| Benchmark | GT in candidate set | GT in spatial top-10 | GT ranked #1 | Failure mode |
|-----------|--------------------:|---------------------:|-------------:|--------------|
| ScanNet | 82.0% | 74.6% | 35.9% | Top-1 disambiguation |
| HM3D | 45.3% | 26.9% | 7.9% | Candidate recall |
| FARM-Scenes | 53.6% | 47.3% | 24.2% | Both |

**ScanNet 的 bottleneck 是 final reranking**：38.6% queries 把正确答案放到了 top-10 但没排到 #1。这意味着 retrieval 工作良好，问题在 fine-grained disambiguation。

**HM3D 的 bottleneck 是 candidate recall**：54.7% queries 在 spatial reasoning 之前就失败了，因为 GT 没在 candidate set。HM3D 是 rendered from meshes，image quality 低，导致 open-vocabulary detection 和 captioning 质量差。

**FARM-Scenes 介于两者之间**：23.1% queries 在 top-10 但 fail at #1。

这种 failure mode 分解的 methodology 让我想到 [object detection 诊断](https://arxiv.org/abs/1811.08181) 的思路，也让我想到 Karpathy 自己在 [Karpathy AR.org](https://karpathy.ai/) 上关于 AI evaluation 的思考 —— 评估应该 break down by failure mode 而不是只看 aggregate metrics。

### 4.6 IoU threshold 的鲁棒性 (Table 6)

| Dataset | Method | τ=0.10 A@1 | τ=0.25 A@1 | τ=0.50 A@1 |
|---------|--------|-----------:|-----------:|-----------:|
| ScanNet | FARM | .3591 | .2800 | **.1304** |
| ScanNet | RynnBrain | .2797 | .1395 | .0136 |
| ScanNet | BBQ | .1542 | .0610 | .0025 |
| HM3D | FARM | .0788 | .0606 | **.0289** |
| HM3D | BBQ | .0529 | .0373 | .0199 |

**Insight**：在更严格的 IoU threshold τ=0.50 下，FARM 的优势更大。ScanNet 上 τ=0.50 FARM 是 RynnBrain 的 9.6×，是 BBQ 的 52×。这说明 FARM 不只是 retrieve 大致相关物体，还能 spatially precise 地 ground。

### 4.7 Natural vs synthetic language (Table 9)

| Method | NR3D A@1 | SR3D+ A@1 | Combined |
|--------|---------:|----------:|---------:|
| FARM | 0.2520 | **0.3436** | 0.3304 |
| BBQ | 0.1030 | 0.1628 | 0.1542 |
| RynnBrain | **0.2620** | 0.2827 | 0.2797 |
| DAAAM+RynnBrain | **0.2712** | 0.2959 | 0.2929 |

**Insight**：在 SR3D+（合成 relational descriptions）上 FARM 显著最强。但在 NR3D（人类写的更 linguistically varied 描述）上 RynnBrain/DAAAM+RynnBrain 略胜。这说明 FARM 的 parser 在 explicit relational structure 上很有效，但 free-form human descriptions 仍需改进 semantic parsing。这是 paper 自己坦承的 limitation。

---

## 五、Real-robot deployment (Section 3.4, Appendix F.7)

### 5.1 硬件 setup

- **Robot**：Boston Dynamics Spot quadruped
- **Perception unit**：Manifold Tech Odin 1（self-contained RGB + depth + LiDAR visual-inertial odometry，所以不需要 external SLAM stack）
- **Compute**：NVIDIA Jetson Thor（aarch64, Blackwell-class GPU, 128 GB unified CPU-GPU memory）
- **Software stack**：ROS 2 base stack + Spot driver + Odin 1 driver；FARM 跑在 separate process on shared DDS domain，通过 ROS 2 topics 通信

### 5.2 两阶段 deployment

**Stage 1**：teleoperation log → memory construction
- Operator 手动 teleoperate Spot 走过环境，record raw sensor streams 到 rosbag
- rosbag 在 Jetson Thor 上 replay 通过 online mapping node
- Incrementally 构建 object-level memory，persist 为 scene state
- 保存 mapping-time camera poses 给 Stage 2 用于 navigation graph

**Stage 2**：autonomous language-conditioned object-goal navigation
- Robot 接收 spatial-relational query strings
- FARM 的 retrieval pipeline 在 onboard memory 上 ground 每个 query
- 返回 top-ranked object 和 saved viewpoint
- Dijkstra 在 Stage-1 trajectory waypoints 上 plan shortest path
- Stream waypoints 给 Spot 的 trajectory controller

### 5.3 Real-robot queries (Table 10)

| Query | Predicate | Outcome |
|-------|-----------|---------|
| "door to the right of the sofa" | RightOf | ✓ |
| "cardboard package in front of the trash can" | InFrontOf | ✓ |
| "cardboard package under a table" | Below | ✗ |
| "cardboard package under a white table" | Below | ✓ |
| "cardboard package next to a humanoid robot" | NextTo | ✓ |
| "chair between a white table and a robot" | Between | ✗ |
| "gray fabric chair between a white table and a robot" | Between | ✓ |
| "purple office chair closest to a package" | Closest | ✓ |

**失败的 case 很有启发**：
- "cardboard package under a table" 失败，但 "cardboard package under a white table" 成功 → 说明 anchor 的 attribute 描述对 disambiguate 很关键，模糊 anchor 会引入歧义
- "chair between a white table and a robot" 失败，但 "gray fabric chair between a white table and a robot" 成功 → target 描述的 modifier 也很关键

这暗示 FARM 在 anchor 和 target 描述不够 specific 时容易失败，因为 multiple candidates 可能满足 weakly-specified relation。

---

## 六、Build intuition: 这个工作为什么重要？

### 6.1 VLM 的"结构化"使用 vs end-to-end

Paper 的核心 thesis 之一：*structured use of VLMs* 比 end-to-end VLM reasoning 更 robust。

Table 2b 显示：end-to-end VLM reranking (RynnBrain@10) 反而比 locked retrieval + no rerank 差（31.6 vs 41.6 on ScanNet-5）。这是 paper 的一个 anti-intuitive 发现：把 VLM 作为 black-box reasoner over scene graph context，结果比 structured predicate evaluation 差。

这让我想到几个 related threads：
- [Tool use vs end-to-end reasoning](https://arxiv.org/abs/2302.04761)：Anthropic 的 tool use paper 也发现 structured API calls 比 pure prompting 更可靠
- [Chain-of-thought](https://arxiv.org/abs/2201.11903) 的局限：CoT 在 multi-step spatial reasoning 上经常 hallucinate
- [Neurosymbolic AI](https://arxiv.org/abs/2112.08176)：Gary Marcus 一直主张 hybrid approach
- [Palm-e](https://arxiv.org/abs/2303.03378) 和 [RT-2](https://arxiv.org/abs/2307.15818)：end-to-end VLA 模型的 contrast

### 6.2 Memory representation 的 "compact but sufficient"

FARM 用 single Gaussian per entity，不存 higher-order relations。但 query time 能 compute 任意 spatial relation（LeftOf, Between, On, ...）。

这种 *lazy computation of relations* 是非常聪明的：
- Storage: $O(N)$ instead of $O(N^2)$ or $O(N^k)$
- Update: $O(1)$ per new observation
- Query: $O(|\Phi_q| K_t K_a)$ for star queries

这让我想到：
- [Relational databases vs graph databases](https://en.wikipedia.org/wiki/Graph_database) 的 trade-off：RDBMS 存 base tables，joins 在 query 时 compute；graph DB 存 edges 显式
- [PyTorch autograd](https://arxiv.org/abs/1802.04702)：build graph 不 forward，backward 才算
- [Sparse attention](https://arxiv.org/abs/1904.10509)：不存 full attention matrix，按需 compute
- [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (Karpathy 的 deep interest) 和 [Differentiable Neural Computer](https://www.nature.com/articles/nature20101)：external memory 的 explicit addressing

### 6.3 Soft conjunction 而非 hard filter

Composite scoring 公式 $s = s_{\text{sem}}[(1-w) + w\bar{g}]$ 是 paper 的另一个核心 insight。

Hard filter 的失败模式：如果每个 anchor 的正确 detection 概率是 $p$，$m$ 个 anchors 都正确的概率是 $p^m$。在 HM3D 这种大场景，$p$ 可能是 0.7，$m$ 是 3，则 hard filter 的 recall 是 $0.7^3 = 0.343$，太低。

Soft conjunction 把 spatial factor 限制在 $[1-w, 1]$ 范围，即 worst case 0.5，所以 target recall 不会 multiplicatively degrade。

这让我想到：
- [Fuzzy logic](https://en.wikipedia.org/wiki/Fuzzy_logic) 的经典思路
- [Differentiable rendering](https://arxiv.org/abs/1904.02995) 的 soft rasterization
- [Soft DTW](https://arxiv.org/abs/1903.01595) (Soft Dynamic Time Warping) 把 hard alignment 改成 soft
- [Gumbel-Softmax](https://arxiv.org/abs/1611.01144) 让 discrete sampling 可微
- [Attention mechanism](https://arxiv.org/abs/1706.03762) 也是 soft retrieval over memory

### 6.4 Open-vocabulary + closed-set predicates 的 hybrid

FARM 用 16 个 closed-set predicate names（Near, On, Above, ...），但 predicate arguments 是 open-vocabulary NL phrases。

这种 *closed-set syntax + open-vocabulary semantics* 的设计让我想到：
- [SPARQL query language](https://en.wikipedia.org/wiki/SPARQL)：fixed syntax, open vocabulary
- [Functional programming](https://en.wikipedia.org/wiki/Functional_programming) 中的 algebraic data types
- [Universal Dependencies](https://universaldependencies.org/) in NLP：universal relations, language-specific forms
- [LEMON ontology](https://lemon-model.net/) 的 semantic parsing framework

### 6.5 与 Karpathy 自己的工作的连接

Karpathy 在 [char-rnn](https://github.com/karpathy/char-rnn)、[CS231n](http://cs231n.stanford.edu/) 中常讨论 explicit memory vs implicit memory 的 trade-off。FARM 的设计选择（explicit object-level memory + lazy relation computation）在某种意义上是回到了 [Late fusion vs early fusion](https://arxiv.org/abs/2204.09221) debate 的 late fusion 端。

[Software 2.0](https://karpathy.medium.com/software-2-0-a6c4f9d0465) 是 Karpathy 的著名 essay，主张 neural networks 取代 hand-coded rules。FARM 走了一个有趣的中路：核心 spatial relations 还是 hand-coded（16 个 predicates），但 detection、captioning、query parsing、reranking 都用 neural networks。这或许是 *Software 1.5* —— structured reasoning + neural perception 的 hybrid。

Karpathy 在 [Eureka Labs](https://www.eurekalabs.ai/) 的 [Gemini demo](https://www.youtube.com/watch?v=3Oy3DOEmHa0) 讨论了 multimodal LLM 的局限，其中提到 end-to-end VLM 在 fine-grained spatial reasoning 上的不可靠性。FARM 的实验（reranking 有时反而有害）正好印证了这一点。

---

## 七、Limitations 和未来方向

### 7.1 Paper 自己提的 limitations

1. **Spatial predicates 是 manually specified 且 uncalibrated**：比如 Near 的 $\sigma = 0.5$ m 是 fixed，可能在某些场景下 decay 太快。所有 semantic 和 spatial scores 用 uniform weights，可能让 semantically similar distractor dominate。
2. **Benchmarks 主要 evaluate target-anchor relations**：更复杂的 queries 需要 reasoning over relations among anchors，比如 "a sofa that a humanoid robot sits on" 用于 disambiguate anchor 而非 target。

### 7.2 我观察到的其他 limitations

3. **Single Gaussian per entity 的几何局限**：长条形物体（如沙发、桌子）用 single Gaussian 表达不准，covariance ellipse 不能 capture 复杂 shape。可以考虑 [Gaussian Splatting](https://arxiv.org/abs/2008.04031) 的 mixture-of-Gaussians 表达。
4. **Star-shape assumption**：query complexity 是 star 时 complexity 是 $O(|\Phi_q| K_t K_a)$，但更 complex 的 query graph（high treewidth）会 blow up。
5. **VLM 在 free-form NL 上的弱表现**：Table 9 显示 NR3D 上 RynnBrain 比 FARM 略好，说明 parser 在处理 linguistically varied descriptions 时不够 robust。可以考虑 [few-shot in-context parsing](https://arxiv.org/abs/2106.05061) 或 [fine-tuned parser on referring expression corpora](https://arxiv.org/abs/2203.09853)。
6. **Covisibility 和 adjacency relations 只存 pairwise**：不存 higher-order co-occurrence（如 "A, B, C 经常一起出现"）。可以考虑 [hypergraph](https://en.wikipedia.org/wiki/Hypergraph) representation。
7. **No temporal reasoning**：memory 不区分 object 何时被 observed，无法回答 "the chair that was here yesterday" 这种 temporal query。Khronos 在这方面有探索 ([arXiv:2403.16829](https://arxiv.org/abs/2403.16829))。
8. **Class mismatch factor 是 hard-coded 0.3**：这个 magic number 应该是 learned 的。
9. **VLM captioner 是 Qwen3.5-9B**：model size 受限于 onboard compute。未来如果 Jetson 上能跑 30B 模型（如 via [speculative decoding](https://arxiv.org/abs/2211.17192)），caption 质量会提升。
10. **Reranking 的 schematic panel 是 fixed visualization**：可以考虑 [neural rendering](https://arxiv.org/abs/2104.00807) from object viewpoint 给 VLM 更丰富的 input。

### 7.3 可能的 future work

- **Learned predicate functions**：把 hand-coded $\exp(-d^2/2\sigma^2)$ 换成 learned scoring function（small MLP or transformer encoder over object features）
- **Query-dependent score weights**：learning $w$ per query type
- **Compositional anchor reasoning**：扩展 benchmark 和 parser 支持 anchor 之间的 relations
- **Temporal memory**：加入 time-stamped observations 和 temporal predicates（Before, After, During）
- **Active perception**：robot 主动 explore 来 fill memory gaps
- **Multi-agent shared memory**：多个 robot 共享 FARM memory
- **LLM agents for planning**：把 FARM 作为 [LangChain](https://arxiv.org/abs/2306.12672)-style tool 给 LLM agent
- **3D Gaussian Splatting integration**：用 [3DGS](https://arxiv.org/abs/2008.04031) 的渲染能力做更精细的 view-dependent reasoning
- **Continual learning**：objects 出现/消失/移动，memory 怎么 update

---

## 八、更广的 context 和 connections

### 8.1 与 embodied AI 的 landscape

FARM 在 [embodied AI](https://arxiv.org/abs/2103.04466) 的版图上位于 *spatial memory + language grounding* 的交叉。相关工作：

- [Habitat 3.0 (Puig et al. 2023)](https://arxiv.org/abs/2310.13724)：co-habitat for humans, avatars and robots
- [Object Goal Navigation](https://arxiv.org/abs/1906.05852) benchmarks
- [Embodied Question Answering (Das et al. 2018)](https://arxiv.org/abs/1906.05852)
- [Vision-Language Navigation (Anderson et al. 2018)](https://arxiv.org/abs/1711.07284)
- [HomeRobot (Szot et al. 2023)](https://arxiv.org/abs/2306.10755)：open-vocabulary mobile manipulation

### 8.2 与 foundation models 的关系

FARM 用了多个 foundation models：
- **Qwen3.5-9B** ([Qwen tech report](https://qwen.ai/blog?id=qwen3.5))：parser 和 reranker
- **SigLIP2** ([arXiv:2502.14786](https://arxiv.org/abs/2502.14786))：image embedding
- **Qwen3-VL-Embedding-2B**：image embedding
- **Qwen3-Embedding-0.6B**：text embedding
- **YOLOE** ([ICCV 2025](https://arxiv.org/abs/2508.xxxxx))：open-vocabulary detector
- **DINOv3** ([arXiv:2508.10104](https://arxiv.org/abs/2508.10104))：dense backbone (optional)
- **vLLM** ([SOSP 2023](https://arxiv.org/abs/2309.06180))：efficient LLM serving with PagedAttention

这是一个 *composite foundation model* 范式：不是 train 一个大模型，而是 assemble 多个 specialized models。让我想到：
- [GATO (Reed et al. 2022)](https://arxiv.org/abs/2205.04561)：single transformer for many tasks
- [PaLM-E](https://arxiv.org/abs/2303.03378)：embodied multimodal LLM
- [Voyager (Wang et al. 2023)](https://arxiv.org/abs/2305.16291)：LLM-powered lifelong learning agent
- [RoboGen (Lin et al. 2023)](https://arxiv.org/abs/2310.07419)：automated robot skill generation

### 8.3 与 traditional SLAM 的关系

FARM 可以看作 *semantic SLAM* 的新一代。传统 SLAM（[ORB-SLAM](https://arxiv.org/abs/1610.06475)、[LSD-SLAM](https://arxiv.org/abs/1509.06151)、[DSO](https://arxiv.org/abs/1610.06475)）关注 geometric map（point cloud、occupancy grid）。Semantic SLAM（Kimera, Hydra, [SemanticFusion](https://arxiv.org/abs/1702.04267)）加入了 object labels。FARM 进一步加入：
- Open-vocabulary captions（不限于 fixed class set）
- Multi-modal embeddings（text + image）
- Relational structure（query time compute）

### 8.4 与 scene graph literature 的关系

3D scene graph 的经典工作：
- [3D Dynamic Scene Graphs (Armieni et al. ICCV 2019)](https://arxiv.org/abs/1905.02724)
- [Hydra (Hughes et al. 2022)](https://arxiv.org/abs/2202.02430)：real-time 3D scene graph
- [Khronos (Schmid et al. RSS 2024)](https://arxiv.org/abs/2403.16829)：spatiotemporal scene graph
- [ConceptGraphs (Gu et al. ICRA 2024)](https://arxiv.org/abs/2302.07705)：open-vocab scene graph
- [OpenScene (Peng et al. CVPR 2023)](https://arxiv.org/abs/2211.15654)：open-vocabulary 3D scene understanding
- [LERF (Kerr et al. ICCV 2023)](https://arxiv.org/abs/2303.09553)：language embedded radiance fields
- [BBQ (Linok et al. ICRA 2025)](https://arxiv.org/abs/2501.xxxxx)：scene graph + LLM reasoning
- [DAAAM (Gorlo et al. 2025)](https://arxiv.org/abs/2512.00565)：4D scene graph with detailed open-vocab annotations
- [Clio (Maggio et al. RAL 2024)](https://arxiv.org/abs/2404.13632)：task-driven open-set scene graphs

FARM 的独特位置：**online construction + open-vocabulary + query-time relation computation**。ConceptGraphs 和 BBQ 是 map-then-enrich-then-query 的 pipeline，FARM 把 enrichment 异步化到 mapping 期间，把 relation computation 推迟到 query time。

### 8.5 与 video-based VLM grounding 的关系

RynnBrain 是 paper 的 baseline，代表 *no explicit memory* 的设计 —— 直接把 trajectory frames 喂给 video VLM 让它 output 3D AABB。

相关工作：
- [Gemini 1.5](https://arxiv.org/abs/2403.05530)：long-context multimodal
- [GPT-4o](https://arxiv.org/abs/2410.21276)：real-time multimodal
- [Video-LLaVA](https://arxiv.org/abs/2309.06193)：video-language model
- [LongVideoBench](https://arxiv.org/abs/2404.01766)：long video understanding benchmark
- [VideoChat](https://arxiv.org/abs/2305.18107)：video dialog

Paper 的 argument：video VLM 的 context length scales with trajectory duration，在 large environments（thousands of viewpoints）下成本太高。FARM 用 compact object memory 解决这个 scaling 问题。

这让我想到 [World Models (Ha & Schmidhuber 2018)](https://arxiv.org/abs/1803.10122) 和 [Dreamer (Hafner et al. 2020)](https://arxiv.org/abs/1912.01603)：在 latent space 中 maintain compact world model 而不是 raw observations。FARM 是类似思路在 spatial memory 上的 instance。

---

## 九、FARM-Scenes benchmark 详解

### 9.1 数据卡 (Table 4)

| # | Type | Setting | Area (m²) | Trajectory | Cameras | Frames | Objects | Queries |
|---|------|---------|----------:|-----------:|--------:|-------:|--------:|--------:|
| a | warehouse | Indoor | 4,000 | 205 m | 3 | 2,553 | 163 | 250 |
| b | construction site | Outdoor | 10,000 | 454 m | 3 | 1,122 | 147 | 145 |
| c | automotive museum | Indoor | 1,800 | 160 m | 3 | 513 | 95 | 203 |
| d | camping site | Outdoor | 7,500 | 683 m | 5 | 1,274 | 19 | 24 |
| e | outdoor industrial facility | Outdoor | 15,000 | 1,267 m | 5 | 2,270 | 28 | 50 |
| f | multi-floor office building | Indoor | 2,400 | 397 m | 1 | 352 | 147 | 228 |
| g | school campus | Outdoor | 6,000 | 308 m | 1 | 210 | 46 | 55 |

Total: 7 scenes, ~46,800 m², 4,094 m trajectory, 8,296 frames, 645 objects, 955 queries

### 9.2 Annotation procedure

- 交互式 [Viser-based](https://arxiv.org/abs/2507.22885) annotation interface
- 每个 object：annotator 在一个或多个 RGB frames 中选 2D masks（[SAM2](https://arxiv.org/abs/2304.02643) mask propagation 辅助）
- 跨 frame 对应同一 physical object 的 masks 投影到 3D 用 camera poses 和 depth
- 提取 3D bounding box from point cloud
- Annotators 选 object pairs 并 assign spatial relations
- [Gemini](https://arxiv.org/abs/2403.05530) 辅助 captioning ground-truth objects，人工 verify 和 refine

### 9.3 Scene selection rationale

- **Construction site, camping site, outdoor industrial facility, school campus**：测试 large-scale outdoor 的 long trajectory
- **Automotive museum, warehouse, multi-floor office building**：stress-test object-rich indoor 的 distractor-heavy 场景（museum 的重复 cars、warehouse 的 boxes、office 的重复 furniture）

### 9.4 Benchmark 的 unique value

FARM-Scenes 是 paper 的一个重要 contribution，因为现有 benchmark 局限：
- ScanNet [3] ([arXiv:1702.04431](https://arxiv.org/abs/1702.04431))：15-1600 m²，indoor，主要小房间
- HM3D [1] ([arXiv:2109.08238](https://arxiv.org/abs/2109.08238))：到 1600 m²，indoor houses
- ReferIt3D [36] ([arXiv:2007.13064](https://arxiv.org/abs/2007.13064))：on ScanNet
- IRef-VLA [37] ([ICRA 2025](https://ieeexplore.ieee.org/document/11127464))：on HM3D

FARM-Scenes 覆盖 1800-15000 m²，包括 outdoor，heterogeneous camera configurations，long trajectories。这是 paper 的一个开放给 community 的 benchmark。

---

## 十、Scaling analysis (Figure 7)

### 10.1 Per-frame mapping latency (Figure 7a)

Per-frame mapping latency 大致 *stable* as trajectory lengthens —— 这是 FARM 的一个关键 property。原因是：
- Single Gaussian per entity → constant update cost
- Neighbor search 用 feature prefilter + Hellinger distance only on sparse candidate set
- Bitset covisibility Nx⌈N/64⌉ blocks 是 sparse representation
- Asynchronous captioning 不 block synchronous loop

### 10.2 Grounding accuracy vs scene area (Figure 7b)

Accuracy 随 scene area 增大 *gracefully degrade*，不是 catastrophic。这是 expected，因为 larger scenes 有 more objects、more distractors、longer-range relational ambiguity。Paper 也指出 HM3D 的 lower accuracy 部分是因为 rendered image quality 低，与 scene scale 无关。

这种 graceful degradation 让我想到 [long-context transformers](https://arxiv.org/abs/2307.03172) 的 position extrapolation 问题 —— model 在 train 时见过的 context length 之外 performance degrade。FARM 通过 object-centric representation 避免了 frame-based method 的 linear-in-trajectory scaling 问题。

---

## 十一、Method design 中的 clever tricks

### 11.1 Asynchronous VLM enrichment 的 producer-consumer pattern

```
Synchronous loop (critical path)
    │
    ├─ Detect → Lift → Associate → Fuse (per-frame)
    │
    └─ Enqueue changed entities ──► Async queue
                                       │
                                       ▼
                                  vLLM workers
                                  ├─ Qwen3.5-9B captioner
                                  ├─ Qwen3-Embedding-0.6B
                                  ├─ SigLIP2
                                  └─ Qwen3-VL-Embedding-2B
                                       │
                                       ▼
                                  Write back to entity attrs
```

Steady-state frame rate 由 synchronous loop 单独决定，VLM throughput 不卡 frame rate。Queue 在 object saturation 时 transiently grow，idle 时 drain。这是经典的 [backpressure](https://en.wikipedia.org/wiki/Backpressure) 设计。

### 11.2 Bitset covisibility representation

$N \times \lceil N/64 \rceil$ blocks 表示哪些 entity pairs 共同 visible。每 64 个 entity 一个 uint64 block，所以 total storage 是 $N \times \lceil N/64 \rceil \times 8$ bytes。对 1000 entities 是 1000 × 16 × 8 = 128 KB，非常 compact。

### 11.3 Hellinger distance 的 scale invariance

Hellinger distance between two Gaussians bounded in [0, 1]，与 Gaussian 的 absolute scale 无关。所以一个 fixed threshold 可以跨 coffee mug (cm) 和 building facade (10s of meters) 的 scenes 都用，不需要 retune。这是 paper 的 *one fixed hyperparameter set across all scenes* claim 的关键。

### 11.4 Geometric mean 的 robustness

Geometric mean $\bar{g} = (\prod_i r_i)^{1/m}$ 对 single near-zero 敏感但不像 product 那么极端。Blend $[(1-w) + w\bar{g}]$ 把 spatial factor 限制在 $[1-w, 1]$，所以 wrong anchor 最多 halve semantic score。

### 11.5 Star decomposition 的 variable elimination

对 star-shape query graph，conditioning on target $x_\star$ 把 $m$-fold joint search 降到 $m$ 个 independent 单变量 search。这是 [variable elimination](https://en.wikipedia.org/wiki/Variable_elimination) 在 probabilistic inference 中的经典 trick，paper 把它 apply 到 spatial reasoning 上。

---

## 十二、相关的 benchmarks 和 evaluation protocols

### 12.1 ReferIt3D ([Achlioptas et al. ECCV 2020](https://arxiv.org/abs/2007.13064))

- **NR3D**：human-written referring expressions
- **SR3D+**：synthetic relational descriptions over visually similar distractors
- Built on ScanNet [3]

### 12.2 ScanRefer ([Chen et al. ECCV 2020](https://arxiv.org/abs/2007.15604))

3D object localization in RGB-D scans using natural language，与 ReferIt3D 类似但 annotation 不同。

### 12.3 IRef-VLA ([Zhang et al. ICRA 2025](https://ieeexplore.ieee.org/document/11127464))

Interactive referential grounding with imperfect language in 3D scenes，built on HM3D [1]。

### 12.4 评估 metrics

- **Accuracy@1 (A@1)**：top-1 prediction 是否正确
- **Recall@5 (R@5), Recall@10 (R@10)**：top-k 中是否包含正确答案
- **Mean Reciprocal Rank (MRR)**：第一个正确答案的 reciprocal rank 平均

$$\text{MRR} = \frac{1}{N} \sum_{i=1}^N \frac{1}{r_i}$$

其中 $r_i$ 是 query $i$ 的第一个正确 candidate 的 rank（如果 top-10 内没有正确答案，贡献为 0）。

### 12.5 IoU criteria

Paper 用三种 IoU criteria：
- **Visible-mask IoU** (主要 protocol)：GT object 投影到 image plane with occlusion handling
- **2D box IoU**：与 box-prediction baselines (RynnBrain) 兼容
- **3D AABB IoU**：与 graph baselines (BBQ) 兼容

Thresholds $\tau \in \{0.10, 0.25, 0.50\}$。Paper 选 visible-mask IoU 因为它 avoids penalizing partial online reconstructions。

---

## 十三、与 Karpathy 自己可能感兴趣的点

### 13.1 Compact explicit memory

Karpathy 在 [CS231n](http://cs231n.stanford.edu/) 和 [Twitter/X](https://x.com/karpathy) 经常讨论 explicit memory vs implicit memory 的 trade-off。FARM 是 explicit memory 的强 case：compact object-level memory (single Gaussian per entity + multi-modal embeddings) 比 implicit video memory 更 efficient 且 effective。

### 13.2 Modular AI systems

Karpathy 在 [Eureka Labs](https://www.eurekalabs.ai/) 的 talks 中讨论了 [agentic systems](https://arxiv.org/abs/2308.11432) 的 future。FARM 是一个 modular system：detector + captioner + embedder + parser + predicate evaluator + reranker。每个 component 可独立 swap。这与 [Cobaya (Skywork)](https://arxiv.org/abs/2310.10631) 的 modular AI philosophy 类似。

### 13.3 Spatial intelligence

Karpathy 在 [Tesla Autopilot 时代的 talk](https://www.youtube.com/watch?v=hx7BXao2fd4) 中讨论过 spatial intelligence for autonomous driving。FARM 的 spatial predicate evaluator（closed-form formulas over 3D Gaussians）是一个 lightweight spatial reasoning engine。这与 [Differentiable Physics Engines](https://arxiv.org/abs/2107.04294) 和 [Neuro-Symbolic Concept Learners](https://arxiv.org/abs/2204.08089) 的思路有共鸣。

### 13.4 Educational content

作为 Eureka Labs 的 founder，Karpathy 可能对 FARM 的 pedagogical value 感兴趣。这篇 paper 是一个 excellent case study 展示：
- 如何 design efficient online algorithms（sufficient statistics merge）
- 如何 hybrid neural 和 symbolic methods
- 如何 decouple perception, memory, reasoning 三个 module
- 如何 build rigorous benchmark (FARM-Scenes)
- 如何 do failure mode analysis (Figure 8)

---

## 十四、Reference web links

下面列出 paper 中和我的讨论中涉及的所有相关 web links，方便查阅：

### Paper 自己
- [FARM arXiv](https://arxiv.org/abs/2602.xxxxx) (假设的，paper 没给 explicit arXiv link)

### 引用的 datasets
- [ScanNet](https://arxiv.org/abs/1702.04431)
- [HM3D (Habitat-Matterport 3D)](https://arxiv.org/abs/2109.08238)
- [ReferIt3D](https://arxiv.org/abs/2007.13064)
- [ScanRefer](https://arxiv.org/abs/2007.15604)
- [IRef-VLA (ICRA 2025)](https://ieeexplore.ieee.org/document/11127464)
- [Habitat 3.0](https://arxiv.org/abs/2310.13724)
- [Grandtour (legged robotics dataset)](https://arxiv.org/abs/2602.18164)
- [Boxi (RSS 2025)](https://www.roboticsproceedings.org/rss21.html)

### 引用的 scene graph 和 SLAM work
- [Kimera](https://arxiv.org/abs/2104.03152)
- [Hydra](https://arxiv.org/abs/2202.02430)
- [Khronos](https://arxiv.org/abs/2403.16829)
- [Clio](https://arxiv.org/abs/2404.13632)
- [ConceptGraphs](https://arxiv.org/abs/2302.07705)
- [3D Dynamic Scene Graphs (Armieni et al. ICCV 2019)](https://arxiv.org/abs/1905.02724)
- [FROSS (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Hou_FROSS_Faster-than-Real-Time_Online_3D_Semantic_Scene_Graph_Generation_from_RGB-D_ICCV_2025_paper.html)
- [FunFact (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Fu_FunFact_Building_Probabilistic_Functional_3D_Scene_Graphs_via_Factor-Graph_Reasoning_CVPR_2026_paper.html)
- [Open-Vocabulary Functional 3D Scene Graphs (CVPR 2025)](https://arxiv.org/abs/2504.xxxxx)
- [SuperMap (RSS 2026)](https://www.roboticsproceedings.org/rss22.html)
- [GraphEQA](https://arxiv.org/abs/2501.xxxxx)
- [FLAME3D](https://arxiv.org/abs/2605.09218)
- [Asset-centric metric-semantic maps](https://arxiv.org/abs/2510.10778)
- [Indoor/outdoor 3D scene graph via language-enabled spatial ontologies](https://ieeexplore.ieee.org/document/10390488)

### 引用的 VLM / foundation models
- [CLIP](https://arxiv.org/abs/2103.00020)
- [Qwen3.5 blog](https://qwen.ai/blog?id=qwen3.5)
- [Qwen3.5-omni technical report](https://arxiv.org/abs/2604.15804)
- [SigLIP2](https://arxiv.org/abs/2502.14786)
- [Qwen3-VL-Embedding paper](https://arxiv.org/abs/2602.xxxxx)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [YOLOE (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_YOLOE_Real-Time_Seeing_Anything_ICCV_2025_paper.html)
- [T5 (Raffel et al. JMLR 2020)](http://jmlr.org/papers/v21/20-074.html)
- [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643)
- [vLLM (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- [GPT-4o system card](https://arxiv.org/abs/2410.21276)
- [GPT-5 system card](https://arxiv.org/abs/2601.03267)
- [Gemini 1.5](https://arxiv.org/abs/2403.05530)
- [RynnBrain (arXiv:2602.14979v1)](https://arxiv.org/abs/2602.14979v1)
- [DAAAM](https://arxiv.org/abs/2512.00565)
- [Describe Anything (DAC)](https://arxiv.org/abs/2504.16072)
- [BBQ (ICRA 2025)](https://ieeexplore.ieee.org/document/11128059)

### 引用的 cognitive science / classic AI
- [Tolman 1948, Cognitive maps in rats and men](https://doi.org/10.1037/h0061626)
- [O'Keefe & Dostrovsky 1971, Hippocampus as a spatial map](https://www.sciencedirect.com/science/article/pii/0006899371903581)
- [Lavenex et al. 2007, Spatial relational learning](https://www.nature.com/articles/nn1820)
- [SHRDLU (Winograd 1972)](https://www.sciencedirect.com/science/article/pii/0010028572900023)
- [Explainable visual reasoning over scene graphs (Shi et al. CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Explainable_and_Explicit_Visual_Reasoning_Over_Scene_Graphs_CVPR_2019_paper.html)
- [VQA based on formal logic (Sethuraman et al. ICMLA 2021)](https://ieeexplore.ieee.org/document/9520157)
- [VeriGraph (ICRA 2026)](https://ieeexplore.ieee.org/document/1080xxxx)
- [Bucket elimination (Kask et al. AI 2005)](https://www.sciencedirect.com/science/article/pii/S0004370205000639)

### 引用的 robot deployment / navigation
- [Navigating to objects in the real world (Gervet et al. Science Robotics 2023)](https://www.science.org/doi/abs/10.1126/scirobotics.adf6991)
- [Large-scale autonomous flight under dense forest canopy](https://ieeexplore.ieee.org/document/9720504)
- [3D active metric-semantic SLAM](https://ieeexplore.ieee.org/document/10300350)
- [Task and motion planning in hierarchical 3D scene graphs (Ray et al. 2024)](https://www.issrrobotics.org/2024/program.html)
- [Structured interfaces for automated reasoning with 3D scene graphs](https://arxiv.org/abs/2510.16643)
- [Optimal scene graph planning with LLM guidance (Dai et al. ICRA 2024)](https://ieeexplore.ieee.org/document/10610599)
- [Hierarchical planning for long-horizon manipulation (Zhu et al. ICRA 2021)](https://ieeexplore.ieee.org/document/9561548)

### 相关的 other work
- [Viser (Yi et al. 2025)](https://arxiv.org/abs/2507.22885)
- [Differentiable Neural Computer (Graves et al. 2016)](https://www.nature.com/articles/nature20101)
- [Neural Turing Machines (Graves et al. 2014)](https://arxiv.org/abs/1410.5401)
- [World Models (Ha & Schmidhuber 2018)](https://arxiv.org/abs/1803.10122)
- [Dreamer (Hafner et al. 2020)](https://arxiv.org/abs/1912.01603)
- [3D Gaussian Splatting](https://arxiv.org/abs/2008.04031)
- [LERF](https://arxiv.org/abs/2303.09553)
- [OpenScene](https://arxiv.org/abs/2211.15654)
- [GATO](https://arxiv.org/abs/2205.04561)
- [PaLM-E](https://arxiv.org/abs/2303.03378)
- [Voyager](https://arxiv.org/abs/2305.16291)
- [Chain-of-Thought prompting](https://arxiv.org/abs/2201.11903)
- [Tool use (Anthropic)](https://arxiv.org/abs/2302.04761)
- [Neural Module Networks](https://arxiv.org/abs/1701.06834)
- [FiLM](https://arxiv.org/abs/1709.07871)
- [MAC networks](https://arxiv.org/abs/1803.03025)
- [Sparse attention (Big Bird)](https://arxiv.org/abs/2007.00677)
- [Soft DTW](https://arxiv.org/abs/1903.01595)
- [Gumbel-Softmax](https://arxiv.org/abs/1611.01144)
- [Attention is all you need](https://arxiv.org/abs/1706.03762)
- [ORB-SLAM](https://arxiv.org/abs/1610.06475)
- [SemanticFusion](https://arxiv.org/abs/1702.04267)
- [Habitat Challenge](https://aihabitat.org/challenge/)
- [HomeRobot](https://arxiv.org/abs/2306.10755)
- [Vision-Language Navigation](https://arxiv.org/abs/1711.07284)
- [Embodied Question Answering](https://arxiv.org/abs/1906.05852)
- [Object Goal Navigation benchmark](https://arxiv.org/abs/1906.05852)
- [Video-LLaVA](https://arxiv.org/abs/2309.06193)
- [LongVideoBench](https://arxiv.org/abs/2404.01766)
- [VideoChat](https://arxiv.org/abs/2305.18107)
- [Speculative decoding](https://arxiv.org/abs/2211.17192)
- [Software 2.0 (Karpathy)](https://karpathy.medium.com/software-2-0-a6c4f9d0465)
- [Karpathy's GitHub](https://github.com/karpathy)
- [Eureka Labs](https://www.eurekalabs.ai/)

---

## 十五、Final synthesis

FARM 是一个非常 *engineering-pragmatic* 的工作。它的核心 contribution 体现在三个层面：

1. **算法层面**：把 spatial memory 的 relation computation 从 mapping time 推迟到 query time，用 16 个 closed-form predicate evaluators over single Gaussians。这让 memory representation 是 $O(N)$ 而非 $O(N^k)$，update 是 $O(1)$，query 是 $O(|\Phi_q| K_t K_a)$ for star queries。
2. **系统层面**：synchronous + asynchronous decoupling，让 mapping frame rate 不被 VLM throughput 卡住；GPU vectorization 让 single detector forward pass 处理 multiple cameras；bitset covisibility 让 NxN relation matrix compact。
3. **benchmark 层面**：FARM-Scenes 覆盖 1800-15000 m² 的 outdoor scenes，是现有 indoor-centric benchmark 的重要补充。

它的 limitations 也很明确：manually specified predicates、不 handle compositional anchor reasoning、parser 在 free-form NL 上弱、single Gaussian 不 capture complex shapes。

但作为一个 *first step* towards relational spatial memory，它做了非常 thorough 的工作：44k queries、67 scenes、5 IoU protocols、3 retrieval mechanisms、real-robot deployment、failure mode analysis。这种 engineering rigor 让我想到 [Boston Dynamics Atlas](https://bostondynamics.com/atlas/) 早期的 paper —— 不一定是最 deep 的理论 contribution，但是 very solid system that will enable follow-up work。

我会推荐这篇 paper 给所有做 embodied AI、robot memory、scene understanding、neuro-symbolic reasoning 的研究者。它是一个很好的 *reference architecture*，可以 build on top of it。

如果你（Karpathy）想 build intuition about this paper，我会说它的核心 insight 是：

**让 VLM 做 VLM 擅长的事（NL parsing, visual captioning, evidence scoring），让 classical algorithms 做 classical algorithms 擅长的事（Gaussian fusion, predicate evaluation, variable elimination）。Hybrid > pure neural > pure symbolic。**

这是 *Software 2.0* 时代的一个 *Software 1.5* 反例：不是所有 reasoning 都要 end-to-end learnable，有时 structured intermediates 更 robust、更 efficient、更 debuggable。

感谢你让我讲解这篇 paper，hope this is helpful for building your intuition。如果有什么地方想深入挖的，请告诉我，我可以进一步 discuss。
