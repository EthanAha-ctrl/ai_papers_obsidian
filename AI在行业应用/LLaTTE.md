---
source_pdf: LLaTTE.pdf
paper_sha256: 86dc0fa98711122810984aa06989b583c0dd385db24200b1c9f21611f376ec73
processed_at: '2026-08-05T15:17:36-07:00'
target_folder: AI在行业应用
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LLaTTE

好嘞 Andrej，咱们坐下来聊聊这篇 paper 到底在说啥，抛开公式，讲讲 intuition。

---

## 故事的起点

你做 LLM 这么多年肯定有感觉：transformer 这东西，给它更多参数、更多数据、更多算力，它就 predictably 地变好，一条漂亮的 log-log 直线。这就是 scaling law 的 magic。

但是推荐系统这帮人一直没吃到这个红利。你看 Meta、Google、阿里这些大厂的 ads ranking model，主力还是 DeepFM、DCN、DLRM 这套——本质就是**大规模的 sparse embedding table + 几层 feature interaction**。这些模型擅长一件事：处理几亿个 user ID / item ID，学它们之间的 co-occurrence。但是它们**没有时间维度的 reasoning 能力**。

用户的每一次点击、浏览、转化，其实就是一条 sequence，跟 language token 没本质区别。那为什么推荐系统不直接用 LLM 的 playbook，把 sequence model scale 起来？

两个硬约束：

**第一个，latency**。广告系统每次 request 要在几十毫秒内给几百个 candidate ad 打分，一天处理万亿次 request。你往这个路径里塞一个 8 层 transformer、5000 token context，根本不可能。

**第二个，架构鸿沟**。production 推荐系统 90% 的信号来自 sparse ID features（user ID、ad ID、context ID），这些是 FM-based 模型的主场。纯 sequence model 吃不下这些 high-dimensional sparse features。反过来，FM model 没有 sequence reasoning。两边各干各的，一直没找到一个干净的融合方式。

LLaTTE 这篇 paper 就是来解决这两个问题的，顺便回答一个更根本的问题：**推荐系统的 sequence modeling 到底有没有 scaling law？**

---

## 架构长什么样

非常简单粗暴，分两块：

**Non-sequence backbone**：就是个 DHEN（Deep Hierarchical Ensemble Network），处理所有 sparse ID、dense features、float attributes。这部分是 production 老兵，不动它。

**Sequence module**：一个 transformer，吃用户的历史行为序列，输出一个 fixed-size 的 summary vector，喂回给 non-sequence backbone。

这个设计的精髓在于**解耦**：sequence module 只负责一件事，就是把 user history 压缩成一个好的 representation。它不需要关心 sparse ID 的 feature interaction，那个活 DHEN 干。反过来 DHEN 也不需要管 temporal reasoning，那个活 transformer 干。

两边通过 summary vector 这个接口对接，sequence module 就可以**独立 scale**——这很关键，因为你要研究 scaling law，必须能独立地调一个模块的 depth / width / sequence length，而保持其他部分固定。

具体到 transformer 内部，有两个工程 trick：

**MLA (Multi-head Latent Attention)**：从 DeepSeek-V2 借来的。本质就是把 key-value 先压到一个低维 latent space，attention 在这个 latent space 里算，最后再解压回 output。数学上完全等价于 Multi-Query Attention，没有 magic，但是 KV memory 能省十几倍——这对 5000 token 的 long context 是决定性的。

**Pyramidal trimming**：浅层看全部历史，深层只看最近的事件，最深层只留 query token 做 readout。intuition 是：远处的事件需要在浅层被 attend 到（捕获 long-range interest），但是不需要在每一层都反复 attend（节省 compute）。这和 U-Net 的 pyramid、Perceiver 的 latent array 是一个思路——把 capacity budget 集中在信息密度高的地方。

还有个 **query token** 的设计很巧妙：在 sequence 前面拼几个 learned token，编码当前的 candidate ad + request context。transformer 跑完之后，这些 query token 的 output 就是 summary。相当于让 sequence module 做 **target-aware attention**——"针对这个 ad，从用户历史里 retrieve 相关的 interest"。这个 idea 来自 Perceiver，DIN 早就用过 target-aware attention，只是 LLaTTE 把它和 modern transformer 架构结合得更干净。

---

## 实验发现了什么

好，现在进入 paper 的核心：scaling 实验。

### 发现一：Width 是 depth scaling 的前置条件

这个和 LLM 圈的共识不太一样。LLM 圈一般认为 shape 不太重要，depth 和 width 可以 trade off。但是 LLaTTE 的实验显示：**如果你不先把 width 加够，再加 depth 是白搭**。

具体来说，width $d=128$ 时，depth 从 1 加到 8，NE 几乎没动。但是 $d=256$ 时，depth 8 比 depth 4 好很多。再往宽加到 $d=1024$、depth 只有 2，反而比 $d=256$、depth 8 差——参数量更大，FLOPs 更多，但是效果更差。

intuition 是：推荐系统的 sparse ID embedding 有个 representation bottleneck。width 太窄，每层 transformer 的"信息带宽"不够，depth 再深也只能 memorize ID pattern，学不到新的 temporal reasoning。一旦 width 足够，每层才能开始学真正的 sequence pattern。

这让我联想到 Chinchilla 的 compute-optimal training 讨论——fixed compute 下参数和数据要匹配。推荐系统的"数据"实际上是 unique (user, item) interaction 数，sparse ID 的 cardinality 决定了 width 下限。LLM 的 token embedding space 是 dense semantic，没有这个 bottleneck，所以 shape 不敏感。

### 发现二：Sequence length 是最陡的 scaling lever

这个发现很直觉：**给模型看更长的用户历史，效果一直变好，没看到拐点**。实验做到 $T=1600$，curve 还是单调下降。

更有意思的是 attention distribution 分析：模型不是只 attend 最近几个 token，attention probability 在 200-1600 范围均匀分布，而且出现**24 小时周期性 spike**——用户每天同一时段有 recurring interest。这说明 long context 里确实有信号，模型也在用这些信号。

这个发现和 LLM 的 long-context research 有共鸣。transformer 在 length scaling 上似乎没有理论 ceiling，受限于 data 和 compute。推荐系统的 user history 理论上可以扩到 lifetime scale（$T = 10^5 - 10^6$），TWIN V2 已经做到 $10^5$。

### 发现三：Content features 是 scaling 的"入场券"（最关键发现）

这个是整篇 paper 最原创的 insight，我重点讲讲。

实验 setup：两组模型，一组只用 sparse ID 作为 sequence token 的 input，另一组加上 content embedding（来自 fine-tuned LLaMA 和 multimodal Content Understanding 模型）。分别在 1 层和 4 层 transformer 上跑。

结果：

- **ID only**：1 层到 4 层，NE 几乎没改善。模型 capacity 翻了 4 倍，但是效果没动。
- **ID + content embedding**：1 层反而比 ID only 的 1 层略差（因为 1 层 transformer 没足够 depth 把 semantic 信号和 ID 信号融合），但是 4 层突然起飞，gain 是 -0.118%。

这说明什么？**Content features 不是 additive bonus，是 scaling slope 的 multiplier**。

用 paper 的 scaling law 公式来说：$\Delta\text{NE} = -\alpha \log_{10} \mathcal{C} + \beta$。Content features 修改的是 $\alpha$（slope），不是 $\beta$（intercept）。没有 content features，slope 很平，加再多 compute 也没用。有了 content features，slope 变陡，compute 才能转化为 performance。

深层原因，我的理解是：**sparse ID 是 "pointer"，content embedding 是 "pointee"**。Pointer 本身没有可组合的语义结构，模型无法做 compositional generalization。content embedding 提供 continuous semantic space，让 transformer 的 dot-product attention 能真正发挥 cross-event reasoning。

这和 LLM 里 token embedding 的角色是同构的——没有 BPE token embedding 的 semantic smoothness，再深的 transformer 也学不到 language structure。推荐系统一直缺这个 semantic substrate，直到 content embedding 出现。

**对 scaling law 的修正**：实际上应该是 $\Delta\text{NE}(\mathcal{C}, \rho) = -\alpha(\rho) \cdot \log_{10} \mathcal{C} + \beta(\rho)$，其中 $\rho$ 是 information density。这是 paper 没显式写出来但实验强烈暗示的。

### 发现四：Sequence composition 要 balance freshness 和 signal strength

固定 $T=1000$，改变 view（低信号、高频）和 conversion（高信号、低频）的比例：

- Pure view 最差：信号噪声比太低。
- Pure conversion 也差：conversion 太稀疏，1000 个 conversion 跨度太长，远古 conversion 不反映当前 intent。
- Balanced 或 conv-heavy 最好：conversion 提供 high-signal anchor，view 提供 temporal freshness。

intuition：sequence modeling 的 value 来自 **signal density × temporal coverage** 的乘积。这对训练数据 sampling 策略有直接指导——不要盲目追求"高价值 event"，要保证 view 类的"timestamp scaffolding"。

### 发现五：全局 scaling law 确实成立

把所有实验画在 log-log 图上（Figure 4），确实拟合出直线。各 axis 的 slope 排序：

1. Sequence length 最陡（per FLOP ROI 最高）
2. Depth 次之（前置 width 充足后 robust）
3. Width 最缓（foundational bottleneck）
4. Content quality 是其他 axis 的 slope multiplier，不是单独 axis

这和 Kaplan et al. 2020 的 LLM scaling law 形式上是同构的，但是 slope 比 LLM 小一个数量级——因为推荐系统的 label noise 大、feature 已经被 strong baseline 榨干、NE 是 normalized metric 本身绝对值小。

---

## 怎么部署到 production

好，scaling law 成立了，但是 online ranker 有 latency 约束，跑不了大模型。怎么办？

**两阶段架构**：

**Upstream user model（异步）**：用户做了一次高价值 action（主要是 conversion）之后，触发一个 large LLaTTE 模型，$T$ 可以到 5000，$L$ 可以到 8，跑在 H100 cluster 上。这个模型只看 user-side features（没有 ad context，因为 conversion 发生时下一个 request 的 ad 还不知道），输出一个 2048 维的 user embedding，写到 feature store。

**Downstream online ranker（同步）**：serving 时，ranker 做一次 feature lookup 拿到这个 cached embedding，加上 fresh short-horizon sequence（$T \approx 400$），加上 ad/context features，一起进 compact LLaTTE + DHEN，给 candidate ads 打分。

这个设计的精髓在于：**大模型的 compute 被异步 offload 了，online 路径只多一次 feature lookup**。P99 latency 没有 measurable 变化。

但是有一个 strict information bottleneck：upstream 把几千 events 压成 2048 维 vector，丢掉了 candidate-specific attention capability。这是 multi-stage 的根本 trade-off。

---

## Transfer ratio：upstream 的 gain 能传多少给 downstream

Paper 定义了一个 metric：

$$\tau = \frac{\Delta\text{NE}_{\text{downstream}}}{\Delta\text{NE}_{\text{upstream}}}$$

就是 upstream 模型离线评测的 NE gain，有多少能转化成 online ranker 的 NE gain。

结果：$\tau \approx 50\%$。

这个数字的意义在哪？paper 提到 industry baseline 通常只有 25-30%，受三个因素拖累：
1. Capacity gap（upstream 巨大，downstream 紧凑）
2. Asynchronous staleness（embedding 不是实时的）
3. Information bottleneck（压缩损失）

LLaTTE 的 50% 显著高，说明 **high-level intent 信号 robust to compression**。我的解读：upstream 学到的是"user 是什么样的人"这种稳定 representation，不是"针对这个 ad 该怎么 react"的 transient signal。前者可压缩，后者不可压缩。

更有意思的发现：**两个 iso-FLOPs 的 upstream 配置**（一个 depth-heavy，一个 length-heavy），upstream gain 几乎相同，downstream gain 完全相同。这说明 **total upstream compute 决定 downstream performance，具体 allocation 不敏感**。这给了一个简单的 deployment policy：upstream 模型在 fixed compute budget 下，depth 和 length 可以灵活 trade off，不用纠结。

---

## 为什么 sequence length 的 transfer efficiency 比 depth/width 低

Paper 里有个细节值得展开。看 seq-only FLOPs 的 slope：

- Depth / width 的 slope：upstream 和 downstream 几乎一致
- Sequence length 的 slope：upstream 保留 ~50% of downstream

intuition：depth/width 主要提升 representation capacity，这种 capacity gain 可以被压缩到 fixed-size embedding 里传下去。但是 sequence length 的收益很大一部分来自 **candidate-aware attention**——"针对这个 ad，从用户历史里 retrieve 相关的 interest"。上游没有 candidate，无法做这个，只能学到 generic user representation。所以 length scaling 的 candidate-specific 部分被 bottleneck 吃掉了，剩下的 ~50% 是"universal long-range intent"，能 survive 压缩。

这给一个 actionable insight：**upstream 模型不要无脑加 length**，因为 length 的 transfer efficiency 比 depth/width 低。在 fixed upstream compute 下，应该 balance depth 和 length。

---

## 最终效果

- **0.25% NE reduction** on flagship ads ranker
- **4.3% conversion uplift** on Facebook Feed and Reels
- **hundreds of millions of dollars** annual revenue impact
- Meta 最大 user model deployment

0.25% NE 听起来很小，但是对应 4.3% conversion uplift——这个放大系数（~17×）来自推荐系统的"选择效应"：NE 小幅改善 → ranking 更准 → top candidate 质量大幅提升 → 转化率显著上升。这是和 LLM 评测的本质不同：LLM 是 absolute quality，推荐系统是 relative ranking quality。在 ranking 任务里，loss 微小改善对 top-1 选择影响巨大。这就是为什么推荐系统愿意为 0.1% NE 投入巨大 compute。

---

## 几个我自己的延伸思考

**Content-as-scaling-multiplier 这个 insight 对 LLM 也有启发**。LLM 圈一直把 data quality 当成 bias term（换更好的 data，loss 整体下降一点），但是 LLaTTE 的实验暗示 data quality 可能是 slope modifier（换更好的 data，scaling curve 变陡，compute 的 ROI 提高）。如果这个规律在 LLM 里也成立，那 data quality 的 importance 比我们以为的更大——它不只是"起点更高"，是"上限更高"。

**Transfer ratio 50% 的信息论解释**。如果把 upstream 想成 rate-distortion 问题——source 是 user 的完整 history，code 是 2048 维 embedding，distortion measure 是 downstream NE——那么 $\tau$ 的理论上限取决于 user history 的 intrinsic dimension。如果 user interest 真的可以由 ~2048 维 manifold 描述，$\tau$ 可以接近 100%。这个方向 paper 没触及，但是值得探索。

**Pyramidal trimming 和 MoE 的潜在结合**。Pyramidal 是"硬性"的 capacity allocation——浅层宽、深层窄。但是 user sequence 里不同 event 的"重要性"是动态的，硬性按 recency 切可能不是最优。一个可能的 extension 是把 pyramidal 替换成 learned routing：每层用 router 决定哪些 token 进入下一层，类似 MoE 但是作用在 temporal dimension。

**Lifetime sequence modeling 的终极形态**。paper 实验到 $T=5000$，scaling curve 还没拐。TWIN V2 已经做到 $10^5$。结合 TWIN V2 的 cluster-based compression + LLaTTE 的 MLA + scaling law，可能是 lifetime sequence modeling 的终极形态。

---

## 一句话总结

**推荐系统的 sequence modeling 确实有 scaling law，但是要吃到这个红利，你需要三样东西：足够的 width 作为基础、semantic content features 作为 slope multiplier、两阶段架构绕过 latency 约束。LLaTTE 把这三件事都做对了，所以在 Meta 的 production 系统里拿到了 4.3% conversion uplift。**

希望这个人话版本对你 build intuition 有帮助，Andrej。如果某个点想深挖，随时聊。

---

# LLaTTE: 推荐系统中的 LLM-Style Scaling Laws 深度解析

你好 Andrej，很高兴和你讨论这篇 Meta AI 的 LLaTTE paper。这篇工作本质上是把 LLM scaling laws 的 playbook 移植到 industrial ads recommendation，核心 thesis 是：**推荐系统的 sequence modeling 同样遵循 power-law scaling，但是 semantic content features 是这个 scaling regime 的"入场券"**。下面我从架构、scaling 实验设计、关键发现、multi-stage 部署四个维度展开，尽量把每个公式和实验数据背后的 intuition 讲透。

---

## 1. 问题背景：为什么推荐系统迟迟没复刻 LLM 的 scaling 故事

推荐系统的输入和 LLM 在结构上其实非常相似——用户的每一次 click、view、conversion 都是一个 temporal event，构成一条长度 $T$ 可以到 5000 的 sequence。但是 production 系统长期停留在 SAS-Rec ([Kang & McAuley 2018](https://arxiv.org/abs/1808.09781))、BERT4Rec ([Sun et al. 2019](https://arxiv.org/abs/1904.06690)) 这种 shallow 架构上，原因有两层：

**第一层 latency 约束**：广告系统每 request 要在 millisecond 内对几百个 candidate 打分，深度长 context 的 transformer 在 serving 路径上根本走不通。这一点和 LLM 推理的 prefill/decode 痛点类似，但推荐系统的 QPS 高一个数量级。

**第二层 architectural divide**：production 推荐的"主力"是 Factorization Machine 家族——DeepFM ([Guo et al. 2017](https://arxiv.org/abs/1703.04247))、DCN V2 ([Wang et al. 2021](https://arxiv.org/abs/1708.05123))、DLRM ([Naumov et al. 2019](https://arxiv.org/abs/1906.00091))、DHEN ([Zhang et al. 2022](https://arxiv.org/abs/2203.11014))、Wukong ([Zhang et al. 2024](https://arxiv.org/abs/2403.02545))。这些模型擅长处理 $10^9$ 级别的 sparse ID embedding table 和 dense features，但缺少 temporal modeling capacity。反过来，纯 sequence model (HSTU [Zhai et al. 2024](https://arxiv.org/abs/2402.17152), OneTrans [Zhang et al. 2025](https://arxiv.org/abs/2510.26104), LONGER [Chai et al. 2025](https://arxiv.org/abs/2505.04421)) 又很难吃下 production 的 high-dimensional non-sequence features。

LLaTTE 的解法是 **modular hybrid**：保留 DHEN-style non-sequence backbone 作为"sparse feature specialist"，把 transformer sequence module 作为可独立 scale 的"temporal specialist"，二者通过 sequence summary $\mathbf{z}_{\text{seq}}^{(k)}$ 接口耦合。这个解耦是后续 multi-stage deployment 的前提。

---

## 2. LLaTTE 架构解析

### 2.1 整体数据流

输入被拆成四类特征：
- User features $\mathbf{x}_u$（含 sequence $S_u$）
- Ad features $\mathbf{x}_i$
- User-Ad 交互 $\mathbf{x}_{ui}$
- Context $\mathbf{x}_c$

模型输出 multi-task probabilities，覆盖 CTR / CVR 等 head 集合 $\mathcal{H}$：

$$\mathbf{y} = f(\mathbf{x}_u, \mathbf{x}_i, \mathbf{x}_{ui}, \mathbf{x}_c; \theta) \in [0,1]^{|\mathcal{H}|}$$

变量含义：$\mathbf{y}$ 是预测概率向量，维度等于 head 数（CTR、CVR 等）；$\theta$ 是所有可学习参数。

评估指标 Normalized Entropy：

$$\text{NE} = \frac{-\frac{1}{N}\sum_{i=1}^{N}[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]}{-[p\log p + (1-p)\log(1-p)]}$$

变量含义：$N$ 是样本数；$y_i \in \{0,1\}$ 是 ground truth label；$\hat{y}_i \in [0,1]$ 是预测概率；$p$ 是训练集 empirical positive rate（baseline CTR）。分子是 model 的平均 cross-entropy，分母是"恒定预测 $p$"的 entropy。NE 越低越好；NE=1 表示模型和"无信息 baseline"一样烂。Meta 内部 0.02% 的 NE 下降就算 statistically significant 并能产生 measurable revenue impact——这个阈值之低，是理解整篇 paper 信号强度的关键。

### 2.2 Non-Sequence Backbone (DHEN-style)

初始 representation 通过 concat 拼接所有非 sequence 特征和 sequence summary：

$$\mathbf{h}^{(0)} = \text{Concat}\Big(\mathcal{E}(\mathbf{x}_{\text{sparse}}),\ \mathbf{x}_{\text{dense}},\ \mathbf{x}_{\text{float}},\ \{\mathbf{z}_{\text{seq}}^{(k)}\}_{k=1}^{m_{\text{seq}}}\Big) \in \mathbb{R}^{d_0}$$

变量：$\mathcal{E}: \mathcal{V} \to \mathbb{R}^{d_e}$ 是 sparse ID 的 embedding lookup；$\mathbf{x}_{\text{sparse}} \in \mathcal{V}^{m_{\text{sparse}}}$ 是 categorical feature；$\mathbf{x}_{\text{dense}}$ 是预计算 embedding（比如 content encoder 输出）；$\mathbf{x}_{\text{float}}$ 是连续变量；$\mathbf{z}_{\text{seq}}^{(k)}$ 是第 $k$ 个 sequence summary（可以有多个，从不同 layer 抽取，类似 DeepMoji / ELMo 的 multi-layer readout）。然后过 $L_{\text{NS}}$ 层 feature interaction network：

$$\mathbf{h}^{(\ell)} = \text{NonSeq}_\ell(\mathbf{h}^{(\ell-1)}), \quad \ell=1,\dots,L_{\text{NS}}$$

最终 $\mathbf{z} = \mathbf{h}^{(L_{\text{NS}})}$，喂给 shallow MLP head 输出 $\hat{y}_h = \sigma(\text{MLP}_h(\mathbf{z}))$。

这里的 intuition：DHEN 的角色是"特征工程专家"，处理那些 ID collision、crossing、aggregation 类的信号；sequence module 不需要重复这些活，只需要把 temporal pattern 编码成 dense vector 喂回去。这种 division of labor 让 sequence module 可以专注 scale。

### 2.3 Sequence Module（核心研究对象）

这是 paper 真正的 protagonist。五步流程：

**Step 1: Tokenization**

每个 action $a_t = (\tau_t, \text{type}_t, \text{item}_t, \text{surface}_t, \text{meta}_t)$ 被编码成 token：

$$\mathbf{x}_t = \text{MLP}_{\text{act}}\Big(\mathcal{E}_{\text{type}}(\text{type}_t),\ \mathcal{E}_{\text{item}}(\text{item}_t),\ \mathcal{E}_{\text{surface}}(\text{surface}_t),\ \mathcal{E}_{\text{time}}(\tau_t),\ \mathcal{E}_{\text{meta}}(\text{meta}_t)\Big) \in \mathbb{R}^d$$

变量：$\tau_t$ 是 timestamp；type 是动作类型（click / view / conversion 等）；item 是被交互的 ad/content ID；surface 是入口（Feed / Reels / Stories）；meta 是附加 metadata（可能含 content embedding）。注意 paper 强调 timestamp encoding 是 additive 加到一部分 hidden dimension 上，和 Vaswani 原版 sinusoidal 类似但更轻量。

**Step 2: Query Token Fusion**

引入 $n_q$ 个 query token $\mathbf{Q} \in \mathbb{R}^{n_q \times d}$：

$$\mathbf{X}_{\text{input}} = \text{Concat}(\mathbf{X}_{\text{seq}}, \mathbf{Q}) \in \mathbb{R}^{(T+n_q)\times d}$$

这是关键的 **target-aware** 设计：query token 编码当前 candidate ad + request context + user-level features，类似于 Perceiver ([Jaegle et al. 2021](https://arxiv.org/abs/2103.03206)) 的 latent array，让 sequence module 可以做 candidate-conditioned attention。online 模型 query token 含 ad context；upstream 模型没 ad context（因为异步触发时 ad 还没出现），只编码 user-side features。这个细节是 multi-stage 通用架构的接口设计。

**Step 3: L-layer Transformer with MLA**

每层结构（Pre-Norm 风格）：

$$\mathbf{Z}^{(\ell)} = \text{RMSNorm}\big(\mathbf{R}^{(\ell)} + \text{MLA}(\mathbf{R}^{(\ell)})\big)$$
$$\mathbf{X}^{(\ell)} = \text{RMSNorm}\big(\mathbf{Z}^{(\ell)} + \text{FFN}(\mathbf{Z}^{(\ell)})\big)$$

RMSNorm ([Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)) 而非 LayerNorm，省去 mean-centering 的减法，只做 scale normalization，对 large-batch distributed training 更友好。

**MLA (Multi-head Latent Attention)** 是从 DeepSeek-V2 ([DeepSeek-AI 2024](https://arxiv.org/abs/2405.04434)) 借来的，paper 在 Appendix A.3 给出了完整推导。原始形式：

$$\mathbf{K}, \mathbf{V} \in \mathbb{R}^{T \times h \times d_k} = \text{split}\big(\text{RMSNorm}(\mathbf{X}\mathbf{W}_{\text{down}}^{KV})\mathbf{W}_{\text{up}}^{KV}\big)$$
$$\mathbf{Q} \in \mathbb{R}^{T \times h \times d_k} = \text{RMSNorm}(\mathbf{X}\mathbf{W}_{\text{down}}^{Q})\mathbf{W}_{\text{up}}^{Q}$$
$$\text{MLA}(\mathbf{X}) = \text{softmax}\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\Big)\mathbf{V}\mathbf{W}_{\text{out}}$$

变量：$T$ 是 sequence 长度；$h$ 是 head 数；$d_k$ 是每 head 维度；$\mathbf{W}_{\text{down}}^{KV} \in \mathbb{R}^{d \times d_c}$ 把 input 压到 latent 维度 $d_c$（$d_c \ll h \cdot d_k$）；$\mathbf{W}_{\text{up}}^{KV} \in \mathbb{R}^{d_c \times h \cdot d_k \cdot 2}$ 解压回 K 和 V；$\mathbf{W}_{\text{down}}^Q, \mathbf{W}_{\text{up}}^Q$ 类似处理 Q；$\mathbf{W}_{\text{out}}$ 是输出投影。

**关键 trick**：paper 指出 $\mathbf{W}_{\text{up}}^Q$ 和 $\mathbf{W}_{\text{up}}^{KV}$ 在 attention 计算中可以和相邻线性投影 algebraically fused，最终 MLA 数学上等价于 **MQA (Multi-Query Attention)** 在 latent space 上运行：

$$\mathbf{Q}_{\text{latent}} = \text{RMSNorm}(\mathbf{X}\mathbf{W}_{\text{down}}^Q) \in \mathbb{R}^{T \times h \times d_c}$$
$$\mathbf{KV}_{\text{latent}} = \text{RMSNorm}(\mathbf{X}\mathbf{W}_{\text{down}}^{KV}) \in \mathbb{R}^{T \times d_c}$$
$$\text{MLA}(\mathbf{X}) = \text{softmax}\Big(\frac{\mathbf{Q}_{\text{latent}} \mathbf{W}^{QK} \mathbf{KV}_{\text{latent}}^\top}{\sqrt{d_c}}\Big)\mathbf{KV}_{\text{latent}}\mathbf{W}_{\text{out}}^V$$

这里 $\mathbf{W}^{QK} = \mathbf{W}_{\text{up}}^Q (\mathbf{W}_{\text{up}}^{KV})^\top$ 是 fused 矩阵，$\mathbf{W}_{\text{out}}^V = \mathbf{W}_{\text{up}}^{KV} \mathbf{W}_{\text{out}}$。

intuition：KV cache 在 inference 时只存 $\mathbf{KV}_{\text{latent}}$（每个 token $d_c$ 维），而不是完整的 $h \cdot d_k \cdot 2$ 维。当 $T=5000$、$h=8$、$d_k=128$ 时，KV memory 节省接近 $h \times 2 = 16\times$，这对 long-context serving 是决定性的。MQA ([Shazeer 2019](https://arxiv.org/abs/1911.02150)) 的本质被 paper 直接挑明：**MLA = MQA on compressed latent space**，没有 magic，只是把 down/up projection 的吸收性利用到了极致。

**Step 4: Adaptive Pyramidal Trimming**

借鉴自 OneTrans ([Zhang et al. 2025](https://arxiv.org/abs/2510.26104)) 的 idea。在第 $\ell$ 层之后，从尾部保留 $T_{\ell+1}$ 个 token：

$$\mathbf{R}^{(\ell+1)} = \mathbf{X}^{(\ell)}[:, -T_{\ell+1}:, :] \in \mathbb{R}^{T_{\ell+1} \times d}$$

其中 $T_{\ell+1} \leq T_\ell$。三种 regime：
- Full self-attention ($T_\ell = T + n_q$)：所有 token 保留
- Pyramidal attention ($n_q < T_\ell < T + n_q$)：query token + 最近 $m = T_\ell - n_q$ 个 action token
- Cross-attention ($T_\ell = n_q$)：只剩 query token

intuition：用户行为有 recency bias，越近的事件越重要，但是远处事件也不是 noise——paper Figure 3 的 attention distribution 证明模型确实会 attend 到 200–1600 token 范围的所有位置。Pyramidal trimming 是一种"软性的"事件遗忘：浅层处理全部历史以捕获 long-range dependency，深层聚焦近期事件以保留细粒度 recency 信号，最深层只保留 query token 做 readout。这和 U-Net 的 pyramid 结构、Perceiver 的 latent array 思路是同源的——把 capacity budget 集中在"信息密度高"的位置。

online 模型用 aggressive pyramidal；upstream 模型用 full self-attention（除了最后一层 cross-attention）。同一架构两种 deployment flavor。

**Step 5: Readout**

最终层在 query token 位置的输出 $\mathbf{Z}_{\text{raw}}$，通过 LoRA-adapted MLP 投影成 sequence summary：

$$\mathbf{z}_{\text{seq}}^{(k)} = \text{LoRAMLP}_k\big(\text{Flatten}(\mathbf{Z}_{\text{raw}})\big) \in \mathbb{R}^{d_{\text{seq}}}$$

变量：$k$ 是 summary 索引（可以多个）；LoRA ([Hu et al.](https://arxiv.org/abs/2106.09685)) 的角色是让 summary projection 参数量小、便于多任务共享 backbone。

---

## 3. Scaling Framework：把 LLM playbook 搬过来

### 3.1 主拟合公式

paper 把核心 scaling 关系建模为 log-linear：

$$\Delta\text{NE}(\mathcal{C}) = -\alpha \cdot \log_{10} \mathcal{C} + \beta$$

变量：$\mathcal{C}$ 是 sequence module 的 compute budget（FLOPs）；$\alpha$ 是 scaling 系数（slope，越大表示该 axis 的 scaling 效率越高）；$\beta$ 是 bias（不同 setup 的 baseline）；$\Delta\text{NE}$ 是相对 production baseline 的 NE 改进（负值表示 improvement）。

这和 Kaplan et al. ([2020](https://arxiv.org/abs/2001.08361)) 的 $L(\mathcal{C}) \propto \mathcal{C}^{-\alpha}$ 形式上是同构的——loss 在 log-log 空间是直线。但有一个关键不同：**LLM 的 token 分布是固定的，推荐系统可以同时 scale architectural capacity 和 information density**。这是 paper 第 5 节最重要的 framing。

### 3.2 四个 scaling axis

| Axis | 物理含义 | 控制变量 |
|---|---|---|
| Model Capacity | transformer 容量 | depth $L$, width $d$ |
| Temporal Horizon | 历史信息长度 | sequence length $T$ |
| Information Density | 每 token 信号强度 | sparse ID only vs. +content embedding |
| Cross-Stage Transfer | upstream gain 传导效率 | transfer ratio $\tau$ |

第 4 个 axis 是 multi-stage 独有的，paper 用：

$$\tau = \frac{\Delta\text{NE}_{\text{downstream}}}{\Delta\text{NE}_{\text{upstream}}}$$

变量：分子是 online ranker 实际拿到的 NE gain；分母是 upstream user model 离线评测的 NE gain。$\tau$ 越高表示压缩瓶颈损失越小。

---

## 4. 实验结果：四个关键发现

### 4.1 Width 是 depth scaling 的前置条件

Table 1 的 grid search 结果：

| Setting | $L$ | $d$ | Seq Params (M) | Seq FLOPs (M) | $\Delta$NE (%) |
|---|---|---|---|---|---|
| Deep Narrow | 8 | 128 | 0.85 | 568 | -0.10% |
| Balanced | 4 | 256 | 1.16 | 530 | -0.14% |
| Deep Balanced | 8 | 256 | 2.33 | 1187 | **-0.17%** |
| Shallow Wide | 2 | 1024 | 6.93 | 1909 | -0.08% |

intuition：当 $d=128$ 时，embedding 维度太窄，每层 transformer 的"信息带宽"被 bottleneck 限制，depth 从 1 加到 8 几乎没收益（capacity 全花在 memorize ID pattern 上）。一旦 $d \geq 256$，depth scaling 才开始 work。反过来 $d=1024, L=2$ 的 shallow wide 配置 FLOPs 是 Deep Balanced 的 1.6×，但 NE 反而更差——参数没被 depth 有效利用。

这个发现和 LLM 的"shape doesn't matter much"结论（Kaplan et al.）有出入，原因是推荐系统 sparse ID embedding 的 representation bottleneck 更严重。可以联想到 [Hoffmann et al. 2022 (Chinchilla)](https://arxiv.org/abs/2203.15556) 关于 compute-optimal training 的讨论：在 fixed compute 下，参数量和数据量要成比例，而推荐系统的"数据量"实际上是 unique (user, item) interaction 数，sparse ID 的 cardinality 决定了 width 下限。

**生产启示**：先把 $d$ 加到 256，再 stack layer。这是反直觉的，因为 LLM 圈普遍认为 depth > width。

### 4.2 Sequence length 是最陡的 scaling lever

Figure 2 显示 $T \in \{200, 400, 800, 1600\}$ 对不同 $L$ 的 NE 改进。三个观察：

1. **单调下降**：所有 depth 下，更长 sequence 一致地降 NE，没有 diminishing return 拐点（在 $T=1600$ 还没看到）。
2. **Depth 放大 length 收益**：$L=2$ 的 curve 比 $L=1$ 陡，$L=4$ 更陡。这暗示 depth 和 length 之间存在 super-linear 交互——更深的模型能更有效利用 long context。
3. **Attention distribution 支撑**（Figure 3 / Figure 5 / Figure 6）：模型不是只 attend 最近几个 token，attention probability 在 200–1600 范围均匀分布，并且出现 **24 小时周期性 spike**——用户每天同一时段有 recurring interest。这个周期性 pattern 本身就是 long context 信号。

intuition：推荐 sequence 不像 NLP 那样有强 locality（一句话里的 token 互相依赖）。用户 interest 是 sparse、recurring、跨时间尺度的，所以 attention 必须真正"看到"远处。这也是为什么 SAS-Rec 这种短 context 模型上限有限。

### 4.3 Content features 是 scaling 的"乘数"（最关键发现）

Table 2 bottom 是 paper 的核心 insight：

| Features | 1L model | 4L model |
|---|---|---|
| Sparse IDs only | +0.06% | -0.01% |
| Sparse IDs + content embeddings | 0.00% | **-0.118%** |

intuition 解读：
- ID only 时，从 1 层到 4 层 NE 几乎没动（-0.07% gain）。模型 capacity 增加，但是输入信号太 sparse，新加的 layer 只能 memorize ID co-occurrence，overfitting 风险抵消了 capacity gain。
- 加上 content embedding（来自 fine-tuned LLaMA + multimodal Content Understanding 模型），1 层模型反而比 ID-only 1 层略差（0% vs +0.06%——这里 0% 是 reference baseline）——因为 1 层 transformer 没有足够 depth 把 semantic 信号和 ID 信号融合。
- 但是 4 层 + content 的 gain 是 -0.118%，相对 4 层 ID-only 是额外 -0.108% 的跳跃。

**结论**：content features 不是 additive bonus，而是 **scaling slope 的 multiplier**。Figure 4 把这点画得很清楚：ID-only curve 是黑色平缓线，content-enriched curve 是陡峭线，两条线在 log-log 图上斜率显著不同。换句话说，paper 的 scaling law 公式 $\Delta\text{NE} = -\alpha \log_{10} \mathcal{C} + \beta$ 里，content features 修改的是 $\alpha$，不是 $\beta$。

这个发现的深层原因，我认为是：sparse ID 是 "pointer"，content embedding 是 "pointee"。Pointer 本身没有可组合的语义结构，模型无法做 compositional generalization；content embedding 提供 continuous semantic space，让 transformer 的 dot-product attention 能真正发挥 cross-event reasoning 能力。这和 LLM 里 token embedding 提供 semantic substrate 是同构的——没有 BPE token embedding 的 semantic smoothness，再深的 transformer 也学不到 language structure。

**对 LLaTTE scaling law 的修正**：实际上应该是

$$\Delta\text{NE}(\mathcal{C}, \rho) = -\alpha(\rho) \cdot \log_{10} \mathcal{C} + \beta(\rho)$$

其中 $\rho$ 是 information density。$\rho$ 提升 content richness，$\alpha$ 显著增大。这是 paper 没显式写出来但实验强烈暗示的修正。

### 4.4 Sequence composition：freshness vs. signal strength

Table 2 top 的 ablation，固定 $T=1000$，改变 view / conversion 比例：

| Allocation (Views/Conv) | $\Delta$NE (%) |
|---|---|
| Balanced (500/500) | 0.00% (reference) |
| Conv-heavy (200/800) | +0.01% |
| Pure conversions (0/1000) | +0.105% |
| Pure views (1000/0) | +0.15% |

intuition：conversion 是 high-signal-low-frequency（每次 conversion 都很有信息量，但用户很少 conversion），view 是 low-signal-high-frequency（每次 view 信号弱，但是密度大、新鲜）。

- Pure view 最差：信号噪声比太低，sequence 被 noise 占满。
- Pure conversion 也差：conversion 太稀疏，1000 个 conversion 跨度太长，远古 conversion 不反映当前 intent。
- Balanced 或 conv-heavy 最好：conversion 提供 high-signal anchor，view 提供 temporal freshness and density。

这给一个直觉：sequence modeling 的 value 来自 **signal density × temporal coverage** 的乘积，不是任一单项。这对训练数据 sampling 策略有直接指导——不要盲目追求"高价值 event"，要保证 view 类的"timestamp scaffolding"。

### 4.5 Global scaling law（Figure 4）

把所有实验画在 log-log 图上，确实拟合出直线。各 axis 的 slope 排序：

1. **Sequence length** 最陡（per FLOP ROI 最高）
2. **Depth** 次之（前置 width 充足后 robust）
3. **Width** 最缓（foundational，但是 capacity bottleneck 性质）
4. **Content quality** 不是单独 axis，是其他 axis 的 slope multiplier

---

## 5. Multi-stage Architecture：把 scaling 移到 offline

### 5.1 设计动机

online ranker latency budget 限制 sequence module 到 $T \approx 400$、$L \approx 2-3$。但是 Figure 4 的 scaling curve 在 $T=1600$ 还没到拐点，意味着 online 模型卡在 scaling curve 中段，浪费了大量潜在 gain。

解法是 **decouple**：
- **Upstream user model**（异步）：在高价值 event（主要是 conversion）触发时跑大模型，$T$ 可以到 5000、$L$ 可以到 8，输出固定维度 $d_{\text{transfer}} = 2048$ 的 user embedding，写入 feature store。
- **Downstream online ranker**：serving 时只做一次 feature lookup 拿 embedding，加上 fresh short-horizon sequence（$T \approx 400$），和 ad/context feature 一起进 compact LLaTTE + DHEN。

这里有一个 **strict information bottleneck**：upstream 把几千 events 压成 2048 维 vector，丢掉了 candidate-specific attention capability。这是 multi-stage 的根本 trade-off。

### 5.2 Upstream vs Downstream scaling slopes（Table 3）

| | Depth (L) | Width (d) | Seq. length (T) |
|---|---|---|---|
| **Downstream $\alpha$ (vs total FLOPs)** | 0.200 | 0.133 | 0.265 |
| **Upstream $\alpha$ (vs total FLOPs)** | 0.102 | 0.113 | 0.116 |
| **Downstream $\alpha$ (vs seq-only FLOPs)** | 0.106 | 0.091 | 0.238 |
| **Upstream $\alpha$ (vs seq-only FLOPs)** | 0.092 | 0.102 | 0.094 |

关键 insight：看 total FLOPs 时，downstream slope 显著更陡——但这是因为 downstream 模型 sequence module 只占 30% FLOPs，sequence scaling 在 total FLOPs 里被稀释得少。看 seq-only FLOPs 时，**depth 和 width 的 slope 在 upstream/downstream 几乎一致**（0.106 vs 0.092, 0.091 vs 0.102），但是 **sequence length 的 slope 在 upstream 保留 ~50%**（0.238 vs 0.094）。

intuition：depth/width 主要提升 representation capacity，这种 capacity gain 可以被压缩到 fixed-size embedding 里传下去。但是 sequence length 的收益很大一部分来自 candidate-aware attention——上游没有 candidate，无法做"针对这个 ad 的 long-range interest retrieval"，只能学到 generic user representation，所以 length scaling 的 candidate-specific 部分被 bottleneck 吃掉了。剩下的 ~50% 是"universal long-range intent"，能 survive 压缩。

这给一个 actionable insight：**upstream 模型不要无脑加 length**，因为 length 的 transfer efficiency 比 depth/width 低。在 fixed upstream compute 下，应该 balance depth 和 length，而不是一味堆 length。

### 5.3 Transfer ratio $\tau \approx 50\%$

Table 3c：

| Config | $\Delta\text{NE}_{\text{up}}$ | $\Delta\text{NE}_{\text{down}}$ | $\tau$ (%) |
|---|---|---|---|
| Seq-heavy $(L=3, d=512, T=1000)$ | -0.14% | -0.07% | 50 |
| Model-heavy $(L=6, d=512, T=460)$ | -0.13% | -0.07% | 53 |

两个 iso-FLOPs 配置（~12 GFLOPs/sample）upstream gain 几乎相同（-0.14% vs -0.13%），downstream gain 完全相同（-0.07%）。这表明 **total upstream compute 决定 downstream performance，allocation 不敏感**。

paper 提到 industry baseline 的 $\tau$ 通常在 25-30%，主要受三因素拖累：
1. Capacity gap（upstream 巨大，downstream 紧凑）
2. Asynchronous staleness（embedding 不是实时的）
3. Information bottleneck（压缩损失）

LLaTTE 的 $\tau \approx 50\%$ 显著高，说明 high-level intent 信号 robust to compression。我的解读：upstream 学到的是"user 是什么样的人"这种稳定 representation，不是"针对这个 ad 该怎么 react"的 transient signal。前者可压缩，后者不可压缩。

### 5.4 Compute allocation policy

paper 给出的 deployment 配方：
- **Downstream**: latency budget 给 short sequence ($T \leq 400$) + shallow depth + sufficient width + rich content features
- **Upstream**: relaxed latency budget 下最大化 total sequence compute，45× 于 downstream 的 sequence FLOPs

这个 45× 是经过 transfer ratio 折算后的 sweet spot——upstream 每 1 unit gain 转化 0.5 unit downstream gain，所以 upstream 要 2× 的"额外 effort"才能匹配 direct downstream scaling 的 ROI，再加上 upstream 的 batch efficiency（低 QPS、大 batch），45× 的 total compute 在 H100 cluster 上是 affordable 的。

---

## 6. Production Deployment 结果

### 6.1 系统架构

- Online ranker: 紧凑 LLaTTE，per-source horizon $T \approx 400$，加 cached upstream embedding 作为 dense feature
- Upstream service: H100 GPU cluster，高价值 event 触发（主要 conversion），异步处理，写 feature store
- Online 额外开销: 单次 feature lookup，P99 latency 无 measurable 变化

### 6.2 Business impact

- **0.25% NE reduction** on flagship ads ranker
- **4.3% conversion uplift** on Facebook Feed and Reels
- **hundreds of millions of dollars** annual revenue impact
- Meta 最大 user model deployment

0.25% NE 听起来小，但是对应 4.3% conversion uplift——这个放大系数（~17×）来自推荐系统的"选择效应"：NE 小幅改善 → ranking 更准 → top candidate 质量大幅提升 → 转化率显著上升。这是和 LLM 评测的本质不同：LLM 是 absolute quality，推荐系统是 relative ranking quality。

---

## 7. 与相关工作的 positioning

| System | 多阶段方式 | Sequence scale | Scaling laws? |
|---|---|---|---|
| PinnerFormer ([Pancha et al. 2022](https://arxiv.org/abs/2205.04507)) | async embedding | medium | No |
| SUM ([Zhang et al. 2024](https://arxiv.org/abs/2403.02545)) | async embedding | small | No |
| TransAct V2 ([Xia et al. 2025](https://arxiv.org/abs/2506.02267)) | hybrid | $O(10^4)$ real-time | No |
| SIM ([Pi et al. 2020](https://arxiv.org/abs/2006.05639)) | GSU+ESU 同步 | $O(10^4-10^5)$ | No |
| TWIN V2 ([Si et al. 2024](https://arxiv.org/abs/2407.16357)) | hierarchical cluster | $O(10^5)$ lifecycle | No |
| HSTU ([Zhai et al. 2024](https://arxiv.org/abs/2402.17152)) | pure sequence | large | partial |
| Wukong ([Zhang et al. 2024](https://arxiv.org/abs/2403.02545)) | monolithic | large | partial |
| **LLaTTE** | **async upstream + sync downstream** | **large upstream + small downstream** | **Yes, with content-aware extension** |

LLaTTE 的独特贡献：
1. **首个** 显式建模 multi-stage 之间 transfer ratio 的 scaling law
2. **首个** 把 content feature 作为 scaling slope multiplier 而非 additive term 分析
3. **首个** 在 trillion-scale production 验证 scaling laws 可落地

---

## 8. 延伸 intuition 与开放问题

### 8.1 Content features 作为 scaling multiplier 的深层原因

这让我想到 [Hoffmann et al. 2022 (Chinchilla)](https://arxiv.org/abs/2203.15556) 的一个细节：LRM scaling law 的 slope 受 data quality 影响。在 LLM 里 data quality 通过"重复 token 比例"、"code vs natural language 比例"等隐式 factor 体现；在推荐系统里，content embedding 把这个 factor 显式暴露出来。

可能的延伸：是否存在"content embedding 的 scaling law"——即 content encoder 自身 capacity 的提升对下游 LLaTTE scaling slope 的影响？如果 content encoder 从 7B 提升到 70B，LLaTTE 的 $\alpha$ 会进一步增大吗？这是 paper 没回答但很有趣的方向。

### 8.2 Pyramidal trimming 和 MoE 的潜在结合

Pyramidal trimming 本质是"硬性"的 capacity allocation——浅层宽、深层窄。但是 user sequence 里不同 event 的"重要性"是动态的，硬性按 recency 切可能不是最优。一个可能的 extension 是把 pyramidal 替换成 learned routing：每层用 router 决定哪些 token 进入下一层，类似 MoE 的思路但是作用在 temporal dimension。这和 [Rainbow 2024](https://arxiv.org/abs/2402.11779) 之类的工作有 connection。

### 8.3 Transfer ratio 的理论上限

$\tau \approx 50\%$ 是经验值，但是否存在信息论意义上的上限？如果把 upstream 想成 rate-distortion 问题——source 是 user 的完整 history，code 是 2048 维 embedding，distortion measure 是 downstream NE——那么 $\tau$ 的理论上限取决于 user history 的 intrinsic dimension。如果 user interest 真的可以由 ~2048 维 manifold 描述，那么 $\tau$ 可以接近 100%；如果 user interest 是 high-dimensional 的，$\tau$ 必然受限于 embedding dimension。这个方向 paper 没触及，但是值得探索。

### 8.4 和 LLM RLHF 的类比

upstream user model 类似 LLM 的"reward model / value model"——异步、offline、计算密集。downstream ranker 类似 LLM 的"policy model"——在线、轻量、candidate-aware。这种 decoupling 在 RLHF 里已经很成熟，LLaTTE 把它移植到推荐系统，但是反向——RLHF 是"reward 引导 policy"，LLaTTE 是"representation 增强 ranker"。是否可以做得更激进：upstream 直接输出 candidate-conditional 信号（而不是 user-only representation）？这需要上游知道候选 ad 集合，可能通过"hot ad set"近似。

### 8.5 Length scaling 上限的开放性

paper 实验到 $T=5000$，scaling curve 还没拐。这和 LLM 的 long-context research（[Press 2024](https://arxiv.org/abs/2307.03172) 等）有共鸣——transformer 在 length scaling 上似乎没有理论 ceiling，受限于 data 和 compute。推荐系统的 user history 理论上可以扩到 lifetime scale（$T = 10^5-10^6$），TWIN V2 已经做到 $10^5$。LLaTTE 没做到这个 scale 是因为 pyramidal trimming 还没完全替代为 hierarchical compression。结合 TWIN V2 的 cluster-based compression + LLaTTE 的 MLA + scaling law，可能是 lifetime sequence modeling 的终极形态。

### 8.6 关于 NE 0.02% threshold 的反思

Meta 内部 0.02% NE 就算 significant，这反映了 industrial recommendation 的"低 signal-to-noise regime"。在这个 regime 下，scaling law 的 slope $\alpha$ 必然很小（order of $10^{-3}$ per 10× compute）。这和 LLM 不同——LLM 的 loss scaling slope 是 order of $10^{-1}$。这种差距可能反映：
- 推荐系统 label 噪声大（用户 click 有随机性）
- 推荐系统 feature 已经被 DHEN 等 strong baseline 榨干，sequence module 是 marginal contribution
- NE 是 normalized metric，本身就在 [0,1] 区间，绝对值小

但是 0.25% NE → 4.3% conversion uplift 的放大效应说明：在 ranking 任务里，loss 微小改善对 top-1 选择影响巨大。这是为什么推荐系统愿意为 0.1% NE 投入巨大 compute。

### 8.7 Pyramidal trimming 的 attention cost 分析

paper 给出 layer $\ell$ 的 attention cost 是 $O(T_{\ell+1} T_\ell d)$，FFN cost 是 $O(T_{\ell+1} d^2)$。如果 pyramidal schedule 是 $T_\ell = T / 2^\ell$（指数衰减），总 cost 是

$$\sum_\ell T_\ell^2 d + T_\ell d^2 \approx T^2 d \sum_\ell 4^{-\ell} + T d^2 \sum_\ell 2^{-\ell} = O(T^2 d) + O(T d^2)$$

而 full attention 是 $O(L T^2 d) + O(L T d^2)$。所以指数 pyramidal 把 cost 从 $O(L)$ 降到 $O(1)$——这是 online model 能在 latency budget 内跑多层的关键。但是代价是远处 history 只在浅层被 attend，深层完全丢失。Figure 3 的 attention distribution 显示浅层确实 attend 远处，这和 pyramidal 设计 self-consistent。

---

## 9. 总结：LLaTTE 给我的几个核心 takeaway

1. **Scaling law 在推荐系统确实成立**，但是 slope 比 LLM 小一个数量级，因为 signal-to-noise ratio 低。
2. **Width 是 depth 的前置条件**，这和 LLM 主流结论相反，根因是 sparse ID representation bottleneck。
3. **Content features 是 scaling 的 multiplier，不是 additive term**——这是 paper 最原创的 insight，对 LLM 也有启发（data quality 不只是 bias term，是 slope modifier）。
4. **Multi-stage 是工业部署 scaling 的唯一可行路径**，transfer ratio ~50% 远超 industry baseline，验证了 user representation 的可压缩性。
5. **MLA + pyramidal trimming 是 long-context serving 的工程组合拳**，数学上等价 MQA-on-latent，没有 magic 但是 effective。
6. **0.25% NE = 4.3% conversion = hundreds of millions $ revenue**，这个放大链路是推荐系统愿意持续 scale 的根本 incentive。

希望这个解析对你 build intuition 有帮助，Andrej。如果你想进一步讨论某个具体方向（比如 content-as-scaling-multiplier 的理论解释，或者 transfer ratio 的信息论上限），我很乐意展开。

**参考链接**：
- DeepSeek-V2 MLA: https://arxiv.org/abs/2405.04434
- Kaplan Scaling Laws: https://arxiv.org/abs/2001.08361
- Chinchilla (Hoffmann et al.): https://arxiv.org/abs/2203.15556
- SAS-Rec: https://arxiv.org/abs/1808.09781
- HSTU: https://arxiv.org/abs/2402.17152
- Wukong: https://arxiv.org/abs/2403.02545
- DHEN: https://arxiv.org/abs/2203.11014
- DIN: https://arxiv.org/abs/1706.06978
- DIEN: https://arxiv.org/abs/1809.03672
- BERT4Rec: https://arxiv.org/abs/1904.06690
- FlashAttention: https://arxiv.org/abs/2205.14135
- MQA (Shazeer): https://arxiv.org/abs/1911.02150
- RMSNorm: https://arxiv.org/abs/1910.07467
- LoRA: https://arxiv.org/abs/2106.09685
- PinnerFormer: https://arxiv.org/abs/2205.04507
- TransAct V2: https://arxiv.org/abs/2506.02267
- TWIN: https://arxiv.org/abs/2302.02352
- TWIN V2: https://arxiv.org/abs/2407.16357
- SIM: https://arxiv.org/abs/2006.05639
- LONGER: https://arxiv.org/abs/2505.04421
- OneTrans: https://arxiv.org/abs/2510.26104
- InterFormer: https://arxiv.org/abs/2411.09852
- HLLM: https://arxiv.org/abs/2409.12740
- DCN V2: https://arxiv.org/abs/1708.05123
- DeepFM: https://arxiv.org/abs/1703.04247
- DLRM: https://arxiv.org/abs/1906.00091
- Perceiver: https://arxiv.org/abs/2103.03206
- SUM (Meta user modeling): https://arxiv.org/abs/2403.02545
