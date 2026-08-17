---
source_pdf: MinerU2.5-Pro Pushing the Limits of Data-Centric.pdf
paper_sha256: 6f3d6641aad29f79079ea166a5ab2798f3bc46e53a6f23593b8a3107b1b5d5d6
processed_at: '2026-08-05T18:29:27-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

没问题，我们把这篇论文里那些包装得很学术的术语全部剥掉，直接看它底层到底在玩什么逻辑。

这篇论文的核心故事可以用一句话概括：**现在的 AI 模型大脑已经够用了，真正卡脖子的是喂给它们的教材质量太差。** 

如果你有几个不同架构、不同大小的模型，让它们去做最难的 PDF 解析题，你会发现它们错的地方一模一样。这说明什么？说明问题出在它们共享的训练数据上——大家用的教材都漏了同一块知识。所以作者团队决定，模型大脑（1.2B 的 architecture）一改不改，全部精力用来搞一套极致的“教材工厂”。

下面我们分三块拆解这个“教材工厂”和配套的训练法。

---

### 一、 怎么造出完美的教材？ 

造教材面临两个死结：
1. 数据量不够，长尾场景（比如极其复杂的嵌套表格、密集公式）见得太少。
2. 最难的题，恰恰是自动标注最容易出错的题。

为了解开这两个死结，他们搞了三个组件：

#### 1. DDAS：怎么挑数据？
想象一个巨大的 PDF 池子。传统的做法是随便抓，这会导致高频的普通论文把罕见的复杂数据淹没了。DDAS 的逻辑是先聚类，再根据“难度”调整采样权重。

具体怎么做？先用 ViT-Base 提取视觉特征，把长得像的页面聚在一起。然后看每个簇里的难度分布：如果这个簇里全是简单的东西，降权；如果这个簇里混合了简单和困难的样本，升权。这就保证了训练数据里，那些罕见且难搞的场景会被强制放大。

#### 2. CMCV：怎么判断一个样本难不难？
这是整篇论文最精妙的算法。以前的做法是让一个模型自己跑多次，看结果一不一致。这有个致命问题：如果这个模型本身有盲区，它跑十次都错，你不知道它是“真的难”，还是“这个模型刚好不会”。

CMCV 的解法是叫三个完全不同背景的模型来一起做这道题：
- $M_1$ = MinerU2.5（我们要升级的目标模型）
- $M_2$ = PaddleOCR-VL（外部模型1）
- $M_3$ = Qwen3-VL-30B（外部模型2）

算它们两两之间的相似度（Text 用 edit distance，Table 用 TEDS，Formula 用 CDM）。然后分三档：
- **Easy**: $M_1$ 和 $M_2$ 或 $M_3$ 意见一致。说明大家都会，直接当训练数据，不用管。
- **Medium**: $M_2$ 和 $M_3$ 意见一致，但 $M_1$ 跟它们不一样。这是**最有价值的教材**！因为这精确指出了 $M_1$ 的知识盲区，而且正确答案已经有了（用 $M_2$ 和 $M_3$ 的共识）。
- **Hard**: 三个模型各说各的，全都不一致。这说明这题真的难，没有现成的正确答案，必须送入下一步人工处理。

#### 3. Judge-and-Refine：怎么给最难的题做标准答案？
Hard 样本如果直接扔给模型训练，只会把模型教坏。怎么搞定它们？

作者发现了一个模型自带的认知缺陷：**模型会做“图转文”，但不会做“文转图”的脑内想象。** 你让模型自己检查它输出的 LaTeX 公式对不对，它在脑子里渲染不出那个画面，所以它总觉得“我写的应该是对的”。

破局的方法非常物理：**直接把模型输出的 LaTeX/HTML 代码真渲染成一张图片，然后把原 PDF 图片和渲染出来的图片并排放在模型面前，问它：“你看这两张图长得一样吗？”**

这就把抽象的代码检查，变成了它最擅长的视觉对比。细微的结构错误（比如少了个对齐符、标签没闭合），在渲染图上会变成惨烈的排版崩溃，模型一眼就能看出来哪里错了，然后再去修。如果连这招都修不好，才送去给人类专家做最后的标注。

---

### 二、 怎么按顺序喂这些教材？ (Progressive Training Strategy)

教材分好类了，不能一锅端，要分三步喂。

#### Stage 1: 大规模 SFT (打基础)
把 CMCV 自动标注好的 65.5M 个 Easy + Medium 样本扔进去，用 Cross-entropy Loss 跑 1 个 epoch。这一步让模型见识足够多的世界，补齐覆盖率。这一步贡献了最大的涨分（+1.31）。

#### Stage 2: Hard 样本 SFT (拔高)
把 192K 纯人工 + 机器精修的 Hard 样本拿出来微调。这里有个坑：光练难题会让模型忘了怎么做简单题。所以要把 Stage 1 的数据混进来一点。
混合比例非常讲究：Layout 任务难题多，混合比是 6:1；Text 任务难题极少，混合比是 1:50。这步专门涨 Table 识别的分数（TEDS 涨了 2.5）。

#### Stage 3: GRPO 对齐
前两步用的 Cross-entropy Loss 有个毛病：它是按 token 算的，每个 token 一视同仁。但评估模型好不好，是看整个序列的指标（比如 CDM、TEDS）。训练目标和考试目标脱节了。

这一步用强化学习 GRPO 来对齐。让模型对同一个输入生成 $G=16$ 个回答，用考试指标直接当 Reward $r_i$。

Advantage 计算公式：
$$A_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$$
- $A_i$ 是第 $i$ 个回答的相对优势。
- 如果 $A_i > 0$，说明这个回答比组内平均水平好，算法就推高这种输出的概率；反之则降低。

通过这种组内相对比较，模型终于学会怎么在考试指标上拿高分了。这步让 Formula 的 CDM 直接涨了 0.81。

---

### 三、 怎么保证考试是公平的？

作者在跑实验时发现，现有的评测集 OmniDocBench v1.5 是个坏考官。

**考官的毛病：** 它是死板的“一对一”匹配。
比如：标准答案把一个三行公式框在一个大框里。你的模型把这三行公式识别得完美无缺，LaTeX 代码全对，但是你分成了三个小框输出。考官一看：“我的框跟你不一样，0 分！”
这对于输出格式偏好不同的模型（比如 Gemini 3）极其不公平。

**MGAM (Multi-Granularity Adaptive Matching) 的解法：** 
考官不改标准答案，只在自己的判卷逻辑里找补。
给定 Ground Truth 集合 $\mathcal{G}$ 和 Prediction 集合 $\mathcal{P}$：
1. **Stage 1**: 先直接硬匹配，算个分 $S_1$。
2. **Stage 2**: 把你输出的元素在换行符处全切碎，再匹配一次，算个分 $S_2$。
3. **Stage 3**: 枚举你切碎的元素之间所有的组合方式（第 $t$ 个块 $B_k = \bigoplus_{t=l_k}^{r_k} p'_t$，就是把序列从 $l_k$ 拼到 $r_k$）。在所有可能的拼法里找出得分最高的那种，算个分 $S_3$。
4. **最后取**: $\mathcal{M}^* = \arg\max_{k \in \{1,2,3\}} S_k$。

这套逻辑确保了只要你语义上对了，不管你怎么切分布局，都能拿到该拿的分。修完考官后，他们又加了 296 页真正的“Hard”卷子，组成了 v1.6 评测集。

---

### 四、 最终结果给人的启发

当一切搭建完毕，这个 **1.2B 参数的模型，在公平测试下，干翻了所有 200B、235B 的通用大模型（VLM），甚至干翻了自家的老版本。**

拆解它的得分来源：
- **数据规模和覆盖 (Stage 1)**: 贡献了 +1.31 的涨分。证明把数据从 10M 扩到 65M 且保证质量，依然是性价比最高的操作。
- **精准的难题修正 (Stage 2)**: 贡献了 +0.96。专门治偏科，Table TEDS 暴涨。
- **目标对齐的强化学习 (Stage 3)**: 贡献了 +0.45。专治考试不适配。

这篇论文其实是在给整个 AI 社区敲警钟：当 Scaling Laws 在 architecture 上开始出现边际递减时，Data-Centric AI 才是真正的深水区。**模型的自反思不行，那就把它的输出渲染成图让它自己看；单模型评估不行，那就组个异构模型委员会来投票；数据标注纯靠机器不行，那就用机器把 99% 的简单题搞定，把人类专家的预算精准投到那 1% 最核心的刀刃上。**

这是一套极其严丝合缝的工程闭环，非常漂亮。

参考链接：
- MinerU GitHub: https://github.com/opendatalab/MinerU
- OmniDocBench: https://arxiv.org/abs/2412.18431
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300

---

# MinerU2.5-Pro 深度技术解析

Andrej, 这篇论文值得仔细拆解，因为它的核心论点非常 bold——**在 architecture 成熟的当下，data engineering 才是真正能撬动 performance 的杠杆**。下面我从 mechanism 层面一层一层展开，重点讲清每个组件为什么 work、数学上怎么实现、实验上贡献了多少。

---

## 1. 论文的核心 thesis 与关键证据

论文开篇就抛出一个 cross-analysis 现象：作者拿多个 SOTA models（GLM-OCR、PaddleOCR-VL-1.5、Qwen3-VL-235B、Gemini 3 Pro、MinerU2.5 自己）跑同一批 real-world PDFs，发现这些模型虽然 architecture 和 parameter scale 差异巨大（0.9B 到 235B），但在 hard samples 上的**失败模式高度一致**——同一组 dense formula layouts、nested tables 在所有模型上一起崩。

这个观察直接推出一个反直觉的结论：既然 architectural 差异不能消除 failure，那瓶颈就在**共享的 training data deficiency**上。两个维度：

1. **Coverage deficiency**：MinerU2.5 原始 training data 不到 10M pages，long-tail 场景（复杂嵌套表、密集公式布局、非常规 multi-column）严重欠采样。
2. **Annotation quality paradox**：恰恰是 hard samples 对 model 改进最有帮助，但恰恰是这些 samples 没有任何 mainstream model 能 reliably 解析，automatic annotation 噪声最大。

这两个维度是耦合的——单纯 scale up data 只会放大已有的 distribution bias 和 annotation noise。

为验证这个 thesis，作者做了一个非常干净的对照实验：**完全保留 MinerU2.5 的 1.2B-parameter decoupled coarse-to-fine architecture（NaViT-675M vision encoder + Qwen2-0.5B LLM）不变**，只做 data engineering + training strategy，看能不能拿到 SOTA。结果：OmniDocBench v1.6 从 92.98 → 95.69，超过所有 >200× 参数量的模型。

参考：
- MinerU2.5 原始 paper: https://arxiv.org/abs/2509.22186
- Data-Centric AI 立场宣言（Andrew Ng）: https://https-deeplearning-ai.github.io/data-centric-comp/

---

## 2. Data Engine 三组件架构

Data Engine 的整体设计是围绕三个 co-optimization 维度展开的：

| 维度 | 组件 | 解决的问题 |
|---|---|---|
| Coverage | DDAS | 10M → 65.5M，同时 mitigate distribution shift |
| Informativeness | CMCV | 难度分层，识别 training signal 密度最高的样本 |
| Accuracy | Judge-and-Refine + Expert | 对 hard samples 做可靠标注 |

下面逐一展开。

### 2.1 DDAS（Diversity-and-Difficulty-Aware Sampling）

DDAS 的本质是**两层 granularity 的 joint sampling**——page-level 先粗筛，element-level 再细筛。这里关键 innovation 是把**difficulty signal 直接耦合进 clustering sampling 的 weight 调整**里。

**Stage 1: Page-level sampling**

对 PDF pool 中所有 pages：
1. 用 ViT-base 提取 512-dim 视觉特征
2. K-Means 聚类
3. 每个 cluster 内做 uniform 初始采样
4. 用 CMCV 对采样得到 difficulty labels (Easy/Medium/Hard)
5. 根据 cluster 内 difficulty 分布调整 sampling weight：
   - Easy 主导的 cluster → 降权
   - Difficulty 分布多样（Easy/Medium/Hard 混合）的 cluster → 升权
   - Invalid content（非目标语言、空白页）的 cluster → 直接 filter
6. 用调整后的 weight 扩大采样到整个 PDF pool，得到 ~60M page-level candidate set

**Stage 2: Element-level sampling**

在 page candidate set 上：
1. 用 MinerU2.5 和 PaddleOCR-VL 两个 layout detection 模型解析出 text/formula/table blocks
2. 对每种 element type 独立做 visual feature clustering
3. 对每个 element 用 element-level CMCV 标注难度
4. 在 joint (cluster × difficulty) 空间里做 balanced sampling：
   - Diversity 维度：大 cluster 降采样、小 cluster 升采样，纠正 long-tail shift
   - Difficulty 维度：Medium 和 Hard 上加权，提高 training signal density

最终输出覆盖 layout、text、formula、table 四个子任务的 SFT 训练集。

**这里的 intuition**：单纯的 K-Means clustering sampling 会让高频类别（standard academic papers、single-column reports）大量被采样，把 long-tail 淹没。DDAS 通过把 difficulty signal 喂回 clustering weight，让"罕见的 hard cluster"和"稀有的 medium cluster"在最终数据里被显式放大。

### 2.2 CMCV（Cross-Model Consistency Verification）

这是整个 Data Engine 的核心算法。论文把它叫做 ensemble-based active learning 的实例，原理与 query-by-committee 一脉相承，但加了 document parsing 特有的处理。

**核心 idea**：用三个**异构**（heterogeneous）的 document parsing models 分别推理每个 sample，根据 prediction consistency 模式判定难度。三个 model 是：
- M₁ = MinerU2.5（target model，要被改进的对象）
- M₂ = PaddleOCR-VL
- M₃ = Qwen3-VL-30B

**为什么必须用异构 models**：MinerU2.5 原本的 IMIC（Iterative Model Inference Consistency）只是**同一 model 的多次 inference 一致性**——这只能 capture 单 model 的 epistemic uncertainty，无法区分两种完全不同的情况：
- Case A：MinerU2.5 自己搞不定，但其他 model 能搞定 → 这是 model-specific blind spot，可以通过 cross-model consensus 直接修复
- Case B：所有 model 都搞不定 → 这是 universally hard problem，需要 expert annotation

CMCV 把这两种情况显式区分开。

**Pairwise consistency metrics**（task-specific）：
- Text: edit distance
- Table: TEDS (Tree Edit Distance Similarity)
- Formula: CDM (Character Detection Matching)

**三档难度定义**（anchor 在 MinerU2.5 的相对性能上）：

记 sim(·,·) 为对应 metric，τ 为 consistency threshold：

- **Easy**: sim(M₁(x), M₂(x)) > τ 或 sim(M₁(x), M₃(x)) > τ
  - MinerU2.5 的输出与至少一个 external model 高度一致 → consensus 表示可靠
  - 任何 model 的 output 都可直接作为 annotation
  - 训练价值低（marginal），但数量大，用于 foundational capability building

- **Medium**: sim(M₂(x), M₃(x)) > τ 且 sim(M₁(x), M₂(x)) < τ 且 sim(M₁(x), M₃(x)) < τ
  - 两个 external models 一致但 MinerU2.5 偏离 → external consensus 作为可靠 pseudo-label
  - **这是训练价值最高的 tier**——精确对应 MinerU2.5 的 capability gap，且证明这些 samples 是 learnable 的
  - 因为稀缺，DDAS 优先 upsample Medium samples

- **Hard**: 三个 pairwise consistency 都 < τ
  - 所有 model 都 pairwise disagree → 没有可靠 annotation
  - 必须经过 Judge-and-Refine 或 expert annotation 后才能用

这个分层不仅驱动 sampling weight，还驱动 annotation 资源分配。是整个 pipeline 的"closed-loop 关键接口"。

参考：
- Query-by-Committee (Seung et al. 1992): https://dl.acm.org/doi/10.1145/130385.130413
- Active learning 综述：https://arxiv.org/abs/2010.09690

### 2.3 Annotation Pipeline for Hard Case（Judge-and-Refine）

Hard samples 如果直接用任何 model 的 output 做 annotation，会把 noise 注入训练，**degrade 而非 improve performance**。这里论文做了一个非常聪明的设计。

**Naive self-reflection 的失败原因**：

让 model 自己 check 自己的 output，model 会系统性地**accept 自己的输出**，看不到错误。原因在于**cross-modal mapping 的非对称性**：

- Forward mapping（image → structured text）：model 擅长，OCR 训练就是这么做的
- Inverse mapping（structured text → visual appearance）：model 不擅长，"在 implicit space 想象 LaTeX 怎么 render 出来"对 model 来说几乎不可能

所以对于 complex structural mappings（LaTeX formulas、HTML tables），model 没法准确 judge "这个 output sequence 渲染出来长什么样"，自然看不到 structural errors。

**Render-then-Verify 的破局**：

论文引入的关键 trick 是**把 structured output 真的 render 成 image**，然后让 model 同时看：
1. 原始 document image
2. 渲染后的 image（LaTeX 公式 compile 后的、HTML table 渲染后的）
3. Judge-and-Refine prompt

这个设计有两个直接好处：
- **Closes inverse-mapping gap**：把"从序列推回视觉"这个 model 做不到的事，外部化成显式 rendering
- **Error amplification**：细微的 structural flaw（少个 alignment symbol、tag 没闭合）在文本域看不出来，render 成图就变成 layout collapse、错位、对齐崩，视觉对比直接暴露

**Judge-Refine model 的选择**：用 Qwen3-VL-235B。理由是它的 multimodal reasoning 强，而且**独立于 CMCV 的 model pool**，避免 systematic bias。注意这里有意避开了 MinerU2.5 自己——self-reflection 的 bias 问题已经被论文诊断清楚。

**Targeted Expert Annotation**：

经过 Judge-and-Refine 后仍有部分极端 complex cases 没法自动修复，送入 expert annotation。Annotation budget 分配有两条 priority axis：

1. **Correction efficiency**：Judge 阶段已 high-confidence 定位到错误，但 Refine 阶段没改对的 → 优先级最高，annotator 只需局部修正
2. **Marginal impact**：在上面这批里，再优先给 CMCV disagreement 模式指示当前 model 最弱的 subtask → 最大化有限 annotation budget 对整体性能的边际贡献

**Pre-annotation 用 Gemini 3 Pro**（同样独立于 CMCV model pool，避免 data leakage），然后 expert review-and-correction，自动化 QA 工具保证 annotation consistency。

**最终 Data Engine 产出**：
- 65.5M Easy + Medium samples → CMCV auto-annotated → Stage 1 pre-training
- 192K expert-annotated Hard samples → Stage 2 SFT + Stage 3 GRPO

**数据分布的最终配比**（Section 4.1）：
- Text recognition: 21M
- Layout analysis: 14M
- Formula recognition: 13M
- Table recognition: 11.5M
- Image analysis: 6M

---

## 3. Progressive Training Strategy

三阶段策略对应 Data Engine 的三个 data quality tier，从 scale → quality → metric alignment 层层推进。

| Stage | Data source | #Samples | LR (ViT/LLM) | Batch | Gain |
|---|---|---|---|---|---|
| 1 Pre-training | CMCV auto-annotated (Easy+Medium) | 65.5M | 1e-4 / 1e-3 | 256 | +1.31 |
| 2 Hard SFT | Expert-annotated Hard + Stage 1 replay | 192K + replay | 5e-6 / 5e-5 | 128 | +0.96 |
| 3 GRPO | Stage 2 rollouts (filtered mid-reward) | 192K | 1e-7 / 1e-5 | 512 | +0.45 |

三个 stage 共享 model architecture（NaViT-675M + Qwen2-0.5B）和 resolution 设置（2048×28×28，64-2048 tokens per image），都是从 MinerU2.5 的 Stage 0 checkpoint 初始化——这个 checkpoint 已有基础 vision-language alignment 和 OCR capability。

### 3.1 Stage 1: 大规模 pre-training

数据：65.5M CMCV auto-annotated samples，覆盖四个子任务。Subtask ratio 按"OmniDocBench overall score 中的权重 + baseline model 在每个 task 上的 gap"联合调整。

训练 config：
- 全部参数 trainable
- LLM LR = 1e-3，ViT LR = 1e-4
- Batch size = 256
- 1 epoch

对比 MinerU2.5 原版 Stage 1（6.9M samples × 2 epochs），数据规模翻了近 10 倍，且质量经过 DDAS 分布纠正 + CMCV annotation filtering。

### 3.2 Stage 2: Hard sample SFT

数据 = 192K expert-annotated Hard + Stage 1 replay。**Replay 比例非均匀**（按 subtask 不同），反映各 subtask 在 hard sample 数量和 baseline 性能上的差异：

| Subtask | Hard : Replay |
|---|---|
| Layout | 6 : 1 |
| Text | 1 : 50 |
| Formula | 1 : 25 |
| Table | 1 : 10 |
| Image | 1 : 4 |

**这里的 intuition**：Layout hard samples 数量大且 Stage 1 基础强 → 少量 replay 即可；Text hard samples 极度稀缺 → 需要 50 倍 replay 防止 catastrophic forgetting。

LR 降到 5e-5，保护 Stage 1 习得的 foundational capability，只在 hard 场景上 fine-tune 决策边界。

### 3.3 Stage 3: GRPO Alignment

**为什么要这一 stage**：cross-entropy loss 是 token-level 的，每个 token 等权优化，不直接反映 sequence/structure-level 评估指标（edit distance、CDM、TEDS、IoU）。GRPO 直接把 task-level metric 当 reward，对齐训练目标和评估目标。

**GRPO 算法**（标准 DeepSeekMath 形式）：

给定 prompt x，从 Stage 2 model π_θ 采样 G 个 candidate outputs {y_1, y_2, ..., y_G}。对每个 y_i 计算 task-specific reward r_i：

- Text recognition: r = 1 - edit_distance(y_i, ground_truth) / max_len
- Formula recognition: r = CDM(y_i, ground_truth)
- Table recognition: r = TEDS(y_i, ground_truth)
- Layout detection: r = category IoU

组内相对 advantage：

$$A_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$$

这里 A_i 是第 i 个 rollout 在该组内的相对优势——正值表示比组平均好，负值表示比组平均差。std 是组内 reward 的标准差。

Policy gradient 形式（带 clip）：

$$\nabla_\theta \mathcal{L} = \mathbb{E}_{x, \{y_i\}}\left[\frac{1}{G}\sum_{i=1}^{G} \min\left(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon_{low}, 1+\epsilon_{high}) A_i\right)\right]$$

其中 $\rho_i = \frac{\pi_\theta(y_i | x)}{\pi_{\theta_{old}}(y_i | x)}$ 是 importance sampling ratio（新策略与旧策略的概率比）。

论文采用 DAPO 的两个 trick：
- **Clip-higher**：$\epsilon_{high}$ 设大些，防止 advantage 估计饱和导致训练停滞
- **Dynamic sampling**：丢弃 zero-variance 的 rollout group（即整组 reward 完全相同的情况，这种 group 没有 learning signal）

**Training data 筛选**：从 Stage 2 model rollouts 中按 reward 分布过滤：
- 过高 reward（model 已饱和）→ 删
- 过低 reward（sample 太难或 annotation 错误）→ 删
- 保留 mid-reward 区间，最大化有效 policy gradient signal

**Training config**：
- G = 16 rollouts per sample
- LLM LR = 1e-5，ViT LR = 1e-7
- Batch size = 512
- 1 epoch
- 所有数据都来自 expert-annotated set，确保 reward signal 可靠

参考：
- GRPO 原始论文（DeepSeekMath）: https://arxiv.org/abs/2402.03300
- DAPO: https://arxiv.org/abs/2503.14476

---

## 4. OmniDocBench v1.6 评估协议

这是论文第二个独立 contribution——发现并修复了 v1.5 的评估 bias。

### 4.1 Motivation

作者诊断出 v1.5 两个核心问题：

**Matching strategy bias**：v1.5 用固定 granularity 的 one-to-one element matching，对 prediction side 的 segmentation 粒度敏感。

一个具体例子：一个跨 k 行的多行公式，annotation 是一个 block。如果 model 输出**完全相同的 LaTeX**，但分成了 k-1 或 k 个独立 block，score 从满分直接掉到接近零，尽管语义完全正确。

**Insufficient hard sample coverage**：Data Engine 在大规模 difficulty stratification 中发现，v1.5 evaluation set 里 Hard-labeled 样本几乎不存在，benchmark 主要在测 low-to-medium difficulty，top models 在上面 saturated，discriminative power 退化。

### 4.2 MGAM（Multi-Granularity Adaptive Matching）

算法核心是**保持 ground truth 不变，只在 prediction side 自适应调整 segmentation granularity**，找出最优匹配。

记：
- Ground truth set $\mathcal{G} = \{g_1, g_2, ..., g_m\}$，有 m 个 elements
- Prediction set $\mathcal{P} = \{p_1, p_2, ..., p_n\}$，有 n 个 elements

**Stage 1: Direct Bipartite Matching**

直接在原始 granularity 上做最优二分匹配。Cost matrix：

$$C_{ij} = 1 - \text{sim}(p_i, g_j)$$

其中 sim 是 task-specific metric（公式用 CDM，文本用 1-edit distance）。

用 Hungarian algorithm 解最小化问题：

$$\mathcal{M}_1^* = \arg\min_{\mathcal{M}} \sum_{(i,j) \in \mathcal{M}} C_{ij}$$

得到第一个 candidate matching 和 aggregate score $S_1$。

**Stage 2: Prediction Splitting + Bipartite Matching**

把每个 prediction element $p_i$ 在 LaTeX linebreak delimiters（`\\`、`\newline` 等）处切分，得到细粒度 prediction set：

$$\mathcal{P}' = \{p'_1, p'_2, ..., p'_{n'}\}, \quad n' \geq n$$

没有可切分 delimiter 的 element 保持不变。在 $\mathcal{P}'$ 和 $\mathcal{G}$ 上重新做 bipartite matching，得到 $\mathcal{M}_2^*$ 和 $S_2$。

**Stage 3: Partition Enumeration + Bipartite Matching**

Stage 2 切得太细——annotation 粒度不一定是 per-line，可能是 1 到 k 行之间的任意中间粒度。为覆盖所有 merging 方案，枚举 $\mathcal{P}'$ 连续 subsequence 的所有 valid ordered partitions。

具体地：$n'$ 个 fine-grained prediction elements 之间有 $n'-1$ 个 gaps，每个 gap 可以"split"或"merged"，产生 $2^{n'-1}$ 种 partition 方案。

每个 partition $\pi = (B_1, B_2, ..., B_K)$ 把 $\mathcal{P}'$ 分成 K 个连续 blocks，第 k 个 block 定义为：

$$B_k = \bigoplus_{t=l_k}^{r_k} p'_t$$

其中：
- $\oplus$ 表示按原始顺序的字符串拼接
- $l_k$ 是第 k 个 block 的起始 index
- $r_k$ 是第 k 个 block 的终止 index
- $l_1 \leq r_1 < l_2 \leq r_2 < ... < l_K \leq r_K$
- $\cup_k \{l_k, l_k+1, ..., r_k\} = \{1, 2, ..., n'\}$（覆盖所有 fine-grained elements）

对每个 partition，对 merged block set $\{B_1, ..., B_K\}$ 和 $\mathcal{G}$ 做 bipartite matching，选最好的 partition 得到 $\mathcal{M}_3^*$ 和 $S_3$。

**Global Optimum Selection**:

$$\mathcal{M}^* = \arg\max_{k \in \{1, 2, 3\}} S_k$$

最终 task-specific metric 基于 $\mathcal{M}^*$ 计算。

**Dense text matching 复用**：MGAM 不仅用于公式，也用于 dense text——把同样算法套到 text elements 上，用 edit distance 作 similarity metric。如果 model 把某个 dense text region 识别成了 table（实际场景中常见），就把 table 转回 plain text 放回 matching pipeline，避免 format preference 差异带来的 unfair penalty。

### 4.3 Hard subset 与三层评估协议

新增 Hard subset：296 pages，从 Data Engine difficulty stratification 中 Hard-labeled 数据池里选，覆盖最具挑战性的场景类别（complex nested tables、dense formula layouts、unconventional layout structures）。

**严格保证**：Hard subset 中所有 samples 都**没有出现在 MinerU2.5-Pro 任何 training stage**（包括 Judge-and-Refine training data），由专业团队 annotation，inter-annotator cross-validation 保证 ground truth 质量。

**三层评估协议**：
- **Base (1,355 pages)**：保留 v1.5 原 evaluation set，维持历史可比性
- **Hard (296 pages)**：新 hard subset
- **Full (1,651 pages)**：Base + Hard 的并集，提供整体性能评估

参考：
- OmniDocBench 原始: https://arxiv.org/abs/2412.18431
- TEDS: https://arxiv.org/abs/1911.13283

---

## 5. 实验数据深度解读

### 5.1 主结果（Table 2）

OmniDocBench v1.6 Full：

| Model | Type | Param | Overall↑ |
|---|---|---|---|
| MinerU2.5-Pro | Specialized VLM | 1.2B | **95.69** |
| GLM-OCR | Specialized VLM | 0.9B | 95.15 |
| PaddleOCR-VL-1.5 | Specialized VLM | 0.9B | 94.87 |
| Youtu-Parsing | Specialized VLM | 2.5B | 93.68 |
| Ovis2.6-30B-A3B | General VLM | 30B | 93.62 |
| Gemini 3 Pro | General VLM | - | 92.85 |
| Gemini 3 Flash | General VLM | - | 92.58 |
| Qwen3-VL-235B | General VLM | 235B | 89.78 |
| GPT-5.2 | General VLM | - | 86.52 |
| InternVL3.5-241B | General VLM | 241B | 83.61 |

**关键观察**：
1. **1.2B 专用 VLM 超过 200× 参数量 general VLM**：MinerU2.5-Pro 比 Qwen3-VL-235B 高 5.91 分，比 InternVL3.5-241B 高 12.08 分。这强化了 data-centric thesis——scale 不解决问题，data quality 才解决。
2. **Sub-metrics 上 MinerU2.5-Pro 在 Formula CDM (97.29)、Table TEDS (93.42)、Table TEDS-S (95.92)、Reading Order Edit (0.120) 都是第一**。
3. **Gemini 3 Pro/Flash 受益于 MGAM**：从原 v1.5 评分到 v1.6 分数明显改善（Full 92.85/92.58），缩小了与专用 model 的差距。这是因为 MGAM 修复了对它们 output granularity 偏好的 penalty。

### 5.2 Base vs Hard 子集对比

**Base subset**：top 3 models（GLM-OCR 96.19、MinerU2.5-Pro 96.12、PaddleOCR-VL-1.5 95.72）只差 0.47 分，几乎 saturated。

**Hard subset**：

| Model | Base | Hard | Δ (Hard - Base) |
|---|---|---|---|
| MinerU2.5-Pro | 96.12 | **94.08** | -2.04 |
| PaddleOCR-VL | 94.49 | 92.48 | -2.01 |
| GLM-OCR | 96.19 | 92.01 | **-4.18** |
| HunyuanOCR | 92.45 | 82.69 | **-9.76** |
| Gemini 3 Pro | 92.96 | 91.99 | -0.97 |

MinerU2.5-Pro 在 Hard 上领先 GLM-OCR 2.07 分，且 Δ 退化最小（仅 -2.04），robustness 最强。GLM-OCR 在 Base 上第一但在 Hard 上掉到第三，HunyuanOCR 退化 9.76 分——这正说明 Data Engine 的 hard sample 处理带来的实际收益。

### 5.3 训练阶段消融（Table 3）

| Stage | Base | Hard | Full | Δ Full | Text↓ | CDM↑ | TEDS↑ |
|---|---|---|---|---|---|---|---|
| MinerU2.5 (baseline) | 93.23 | 91.65 | 92.98 | - | 0.045 | 95.59 | 87.88 |
| +Stage 1 SFT (65.5M) | 94.54 | 93.10 | 94.29 | +1.31 | 0.039 | 96.40 | 90.37 |
| +Stage 2 Hard SFT | 95.60 | 93.84 | 95.25 | +0.96 | 0.036 | 96.48 | 92.87 |
| +Stage 3 GRPO | 96.12 | 94.08 | 95.69 | +0.45 | 0.036 | 97.29 | 93.42 |

**关键观察**：
1. **Stage 1 单阶段贡献最大 (+1.31)**：纯数据规模 + 质量的提升是主驱动。TEDS 从 87.88 → 90.37 (+2.49) 反映数据覆盖扩大对 table recognition 的明显改善。
2. **Stage 2 主要帮助 Table TEDS**（+2.50），因为 hard tables 是 expert annotation 的主要受益对象。
3. **Stage 3 GRPO 主要帮助 Formula CDM**（+0.81）——GRPO 直接优化 task-level metric，formula 的 CDM reward 直接对应 sequence-level 字符匹配。
4. **Hard subset 累积 +2.43 vs Base +2.89**——progressive training 在两个 subset 上都获得均衡提升，没有出现"过拟合 hard 导致 base 退化"。

### 5.4 Element-specific 解析（Table 4-6）

**Text recognition (Table 4)**：

| Model | Type | Base↓ | Hard↓ | Full↓ |
|---|---|---|---|---|
| MinerU2.5-Pro | Decoupled | 0.015 | 0.048 | **0.019** |
| Qwen3.5-397B | General | 0.016 | 0.052 | 0.020 |
| GLM-OCR | Decoupled | 0.016 | 0.053 | 0.021 |
| MinerU2.5 | Decoupled | 0.023 | 0.066 | 0.028 |

MinerU2.5-Pro Full edit distance 0.019，相对 baseline 0.028 下降 32.1%。注意 general VLM 在 text recognition 上接近专用 model 水平。

**Formula recognition (Table 5)**：

跨 9 个 benchmark（OmniDoc Base/Hard + 4 个 UniMERNet subsets + 3 个 in-house），MinerU2.5-Pro 在 5 个维度第一，4 个第二。唯一显著不足的是 HWE（手写公式）——Qwen3.5-397B 在 HWE 上 97.59 vs MinerU2.5-Pro 95.38，但 Qwen3.5-397B 在 Chinese formula 上只有 78.24，暴露 specialization tradeoff。

OmniDocBench Base CDM 达到 99.20，接近 formula recognition ceiling。

**Table recognition (Table 6)**：

Overall TEDS：MinerU2.5-Pro 91.10 第一（vs MinerU2.5 87.94，+3.16）。

Hard subset TEDS：92.46 vs MinerU2.5 88.28（**+4.18**）——这是 Data Engine 在 hard table 上的最大收益体现。

GLM-OCR 在 OmniDocBench Base (96.14) 和 CCOCR (89.17) 略胜，但跨 benchmark 稳定性不如 MinerU2.5-Pro。PaddleOCR-VL-1.5 在 CCOCR (76.34) 和 Inhouse (72.66) 明显掉，说明其 table recognition generalization 受限。

---

## 6. 几个值得深思的 design choice

### 6.1 为什么用 decoupled architecture 而非 end-to-end

End-to-end VLM（Nougat、GOT-OCR 2.0）直接 image-to-markup，避免 cascade error。但 native-resolution 处理 token 复杂度 $O(N^2)$，对高分辨率文档有 efficiency 瓶颈。

Decoupled VLM 把 layout analysis 和 content recognition 分开，结合 pipeline 的 controllability 和 VLM 的 semantic modeling 能力。MinerU2.5 用 1.2B 单 model 同时支持两种任务，平衡了 resolution fidelity、efficiency、deployment complexity。

这跟 element-specific evaluation 的发现也一致：end-to-end models 在 element-specific setting 下（不给 category prior）显著退化——DeepSeek-OCR 2 在 text edit distance 0.066 vs decoupled model 0.019。说明 decoupled 设计的 layout-detection-first 范式对 content recognition 有显式帮助。

### 6.2 为什么外部 Judge model 必须独立于 CMCV pool

CMCV pool 里已经有 MinerU2.5 + PaddleOCR-VL + Qwen3-VL-30B 三个 model。如果用其中之一做 Judge-and-Refine，会有 systematic bias——同一 model 在 CMCV 阶段已经 disagree 的样本，再用它 judge 会继承相同的 blind spot。

Qwen3-VL-235B 在 architecture 和 scale 上都足够异质，多模态 reasoning 又强，所以选它做 Judge-Refine model。同理 expert pre-annotation 用 Gemini 3 Pro 也是独立选择。

### 6.3 Truncated paragraph merging 与 cross-page table merging

这两个工程 feature 不影响 OmniDocBench 分数（专注单页 content recognition），但对实际部署很重要。

**Truncated paragraph merging**：layout detection 把 spatially distinct 的 text block 分割成多个 region，但语义上可能是连续段落（multi-column 跨列、figure/table 中断段落）。论文把它简化成 binary classification：在 reading order 相邻 region 之间预测"merge or no merge"。

训练数据生成：对相邻 text/list region pair，先用 rule-based 过滤（sentence length、leading numbering、terminal punctuation），剩余 candidate 用 Gemini 3 Flash 判断。把两个 region 在 page image 上分别 highlight 红/绿，加文本内容，问是否 merge。长段落只给首末句省 token。

**Cross-page table merging**：table 跨页时自动检测合并。规则启发式找 candidate pair（last table on page N + first table on page N+1，column 数和结构兼容）。对 flagged pair，model 接收结构化 prompt：

```
## Table 1 (Previous Page - Last Table)
Last Row(s) Data: [[content of table 1]]
## Table 2 (Current Page - First Table)
First Data Row(s): [[content of table 2]]
```

Output 是 per-column binary decision list：0 = direct concatenation（cell content 在 page boundary 干净切分），1 = semantic merging（两行作为独立数据保留）。这种 column-level 细粒度策略处理"同一 table split 中部分列需拼接、部分列需语义合并"的常见情况。

### 6.4 Image-aware parsing

MinerU2.5 原本对所有 image region 简单 crop 丢弃，丢失 chart 数据、embedded text、diagram content。MinerU2.5-Pro 引入 image-aware parsing：

第一步分类（class）：pure formula、natural image、chart、text image、table-like image、general image。
第二步差异化提取：chart → structured table；text image → OCR；table-like image → table recognition。

输出格式（4 个 field）：
```
<|class_start|>class<|class_end|>
<|sub_class_start|>sub_class<|sub_class_end|>
<|caption_start|>caption<|caption_end|>
<|content_start|>content<|content_end|>
```

但论文承认这部分**还没用 Data Engine 优化**——image analysis 数据是论文中少有未经过完整 pipeline 处理的部分，未来还有显著提升空间。

### 6.5 In-table image detection

表格中常常嵌入 product photos、diagrams、icons。论文用三步处理：

1. **Detection**：layout detection 识别在 table bounding box 内的 image region，用 placeholder token 替换（mask image region）。
2. **Recognition**：masked table image 喂 Table Recognition，生成含 placeholder token 的 OTSL sequence。
3. **Restoration**：placeholder token 解析回原始 image region 引用，最终 HTML cell 含 `<img>` tag，链接到 extracted image content block。

这让 table structure 和 textual content 不被 embedded image 干扰，同时保留 image 与 cell 的 spatial 对应关系。

---

## 7. 局限性与未来方向

论文 Limitations 部分提了三个未来挑战：

1. **Element-matching paradigm 的根本局限**：即使 MGAM 修复了 granularity bias，element-matching 本身有 inherent ambiguity——format level（HTML vs Markdown for tables）和 structural level（bilingual word list 既可作 line-by-line text pairs 也可作 two-column table，连 human annotator 都可能不一致）。**Semantic-equivalence-aware evaluation** 仍是 open problem。

2. **Domain adaptation**：OmniDocBench v1.6 覆盖主流场景，但金融、法律、医疗等高精度垂直领域需要 domain-specific 评估集。当 model capability 接近 human level 时，评估集 annotation 本身的精度变成越来越紧迫的挑战。

3. **从 parsing accuracy 到 structural understanding**：当前工作聚焦 content accuracy。但下游应用关心的结构关系——heading 与 body 的 hierarchy、figure/table 与 referring text 的 semantic binding、cross-page content continuity——对 document retrieval 和下游 semantic understanding 同样关键。**从"content extraction"到"structured semantic understanding"** 是 document parsing 的自然下一步。

---

## 8. 与相关工作的关系

- **Nougat**：image-to-markup paradigm 的 baseline，但局限于学术文档
- **GOT-OCR 2.0**：统一 scene text 和 document OCR
- **Ocean-OCR / olmOCR / dots.ocr**：native-resolution visual encoder，但 $O(N^2)$ token 复杂度
- **Dolphin / MonkeyOCR / MinerU2.5**：decoupled VLM 路线
- **PaddleOCR-VL-1.5 的 UACS**：与 MinerU2.5 的 IMIC 类似，都是 single-model 信号做 difficulty 估计——CMCV 把这个升级到 multi-model
- **DocGenome**：academic papers 专用，缺乏 difficulty 分层
- **DataComp（vision-language pretraining）**：data-centric paradigm 的先例
- **LIMA**：LLM alignment 的 "less is more" 思路，data-centric 在 alignment 阶段也有效

CMCV 的 methodology 与 query-by-committee（Seung et al. 1992）一脉相承，但论文把 disagreement 信息与下游 annotation strategy 形成闭环，并通过 Judge-and-Refine 处理 document parsing 特有的 annotation reliability 问题。

参考：
- Nougat: https://arxiv.org/abs/2308.13418
- GOT-OCR 2.0: https://arxiv.org/abs/2409.01704
- Dolphin: https://arxiv.org/abs/2505.14459
- MonkeyOCR: https://arxiv.org/abs/2506.05218
- DocGenome: https://arxiv.org/abs/2406.11633
- olmOCR: https://arxiv.org/abs/2502.18443
- UniMERNet (CDM metric): https://arxiv.org/abs2404.15254
- DataComp: https://arxiv.org/abs/2304.14108
- LIMA: https://arxiv.org/abs/2305.11206

---

## 9. Intuition 总结

这篇论文给我最大的启发在于：**当 architecture 成熟到 top models 在同一组 hard samples 上集体 fail 时，"换 architecture" 是个无解的方向——因为问题不在 architecture，而在 data**。

更深一层：**annotation quality paradox 是一个根本性的递归难题**——你想要更好的 model 必须有更好的 data，但更好的 data 需要更好的 model 来 annotate。CMCV + Judge-and-Refine + Expert Annotation 这套 pipeline 给出了破局思路：

1. 用 multiple heterogeneous models 的 disagreement 来近似 ground truth 不可达时的 difficulty signal
2. Easy/Medium 样本用 model consensus 直接 auto-annotate
3. Hard 样本用 render-then-verify 把 inverse mapping 外部化，让 model 在 visual comparison 上做它擅长的事
4. 真正无法自动修复的样本通过 targeted expert annotation 处理，budget 按 correction efficiency 和 marginal impact 双 axis 分配

三阶段训练的设计逻辑是 data quality tier 的渐进式 exploitation：scale (Stage 1) → targeted quality (Stage 2) → metric alignment (Stage 3)。GRPO 阶段直接把 task-level metric 当 reward，弥补 cross-entropy loss 与 sequence-level metric 之间的 gap。

OmniDocBench v1.6 的 MGAM 是另一个亮点——揭示了一个被忽视的事实：**评估 metric 本身的 bias 可能人为制造出"top models 收敛"的假象**，而实际上 hard samples 的 discriminative power 一直被 matching granularity 偏好掩盖。

代码与模型：
- https://github.com/opendatalab/MinerU
- https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B

这篇 paper 对 LLM training 的更广泛启示是：**scale 在某个点之后边际效益急剧下降，而 data quality engineering 的边际收益开始主导**——这个论点在 LLM pre-training 阶段已经成立（Chinchilla 之后），现在在 vision-language document parsing 这个细分领域被实证验证了。1.2B model 干掉 235B general VLM 的事实，是 data-centric paradigm 在 vertical domain 的最强背书。
