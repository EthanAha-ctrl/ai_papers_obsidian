---
source_pdf: Training Long-Context Vision-Language Models Effectively with Generalization
  Beyond 128K Context.pdf
paper_sha256: 4ff49895d55f303adf5b37343a5d46b3a3e2d1d2a2c541d8941cea4ebe5f6795
processed_at: '2026-08-12T18:03:57-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 MMProLong

## 一句话总结

ByteDance 和 HKUST 的人想搞清楚一件事：**怎么花最少的钱把一个 vision-language model 的 context window 从 32K 撑到 128K，还能顺便 generalize 到 512K**。最后他们用 5B tokens、64 张 H20 训了一天多，训出来的 7B model 在 long-context 任务上追平 Qwen2.5-VL-32B。

## 为什么要做这件事

现在大家都说自己的 LVLM 能跑 128K 甚至 1M context，但 Qwen3-VL、GLM-4.5V 这种 tech report 里基本不告诉你**到底用什么数据训的、length 怎么分布、各种 task 怎么 mix**。LLM 那边还有几个 system study（Fu et al. 2024、Gao et al. 2024）把 recipe 摸清楚了，LVLM 这边基本是黑盒。这篇 paper 就是来填这个坑的。

## 他们到底发现了什么

我挑四个最反直觉的发现讲：

### 发现 1：VQA 完爆 OCR

直觉上你可能会觉得，让 model 练习"把整个 document 一字不落地 transcribe 出来"应该能强化 long-context 能力，毕竟这逼着 model 反复 attend 到每一页的 visual content。

但实验结果是 **OCR-full 直接让性能掉 17 分**。加了 SFT 恢复一点也还是不如纯 VQA。

为什么？因为 OCR 训练让 model 学会"均匀 attend 到所有内容"，但 long-context VQA 要的是"selective retrieval"——从一大堆内容里挑出关键信息回答问题。这两个 attention pattern 是冲突的。你练了 dense attention，就把 sparse retrieval 的能力给破坏了。

而且还有个 format alignment 的问题：OCR 是 transcription task，没有 instruction-following 的 format，但下游 evaluation 全是 VQA。model 连"怎么 follow instruction"都快忘了。

### 发现 2：Length 分布越杂越好

他们试了两种 length 分布：
- **pool-native**：自然从 document 池里采，大部分样本在 32K-100K 之间，只有 23.6% 超过 100K
- **long-biased**：故意采长文档，83.9% 样本超过 100K

直觉上 long-biased 应该更好，因为评估就是在 128K 上做的，让 model 多见接近 128K 的 context 应该更直接。

结果 pool-native 稳定赢 +1 到 +1.7 分。

这事其实挺深。Long-context 能力不是一个"到 128K 就能 work"的开关，而是一个**在 [32K, 128K] 全区间都能准确 retrieve 的连续能力**。你 training 全压在 100K-128K，model 在 60K、80K 这种中间长度上的 position encoding 反而不准了。RoPE 的 position interpolation 需要 dense 的 multi-length supervision，不能指望一个 target length 一把梭。

### 发现 3：Retrieval 是真正的瓶颈

他们 grid search 了 extraction 和 reasoning 的比例，从 0:10 到 10:0。结果 8:2 最好，纯 reasoning 或纯 extraction 都差一点。

这暗示了 long-context 的核心 bottleneck 是 **retrieval 不是 reasoning**。model 能不能在 128K context 里准确定位到那几页关键信息，这件事比"找到之后做点计算"难多了。但留 20% reasoning 数据还是有用的——防止 model collapse 到纯 pattern matching，保持 task diversity。

### 发现 4：纯 long-context 训练居然不掉 short-context

这是最反 LLM 直觉的发现。LLM long-context training 经验上必须混 short data，否则 short-context 能力会严重退化。

但这篇 paper 发现，纯 long-document VQA 训练只让 short-context 掉 0.99 分，几乎可以忽略。

为什么？因为 long-document VQA 的**format 本身就是 instruction-following**。即使 context 变长了，model 还是在练"读 instruction 然后回答"，这个能力天然 transfer 回 short-context。LLM 那边 long-context training 用的是 books/code 的 next-token prediction，完全离开了 instruction format，所以才需要混 short data 来 maintain instruction-following。

这其实给了一个更深的 insight：**short-context 退化不是 "context 变长" 导致的，是 "数据 format 离开 instruction" 导致的**。

## 他们怎么造数据的

这是 paper 里我觉得最聪明的地方。

直接让 LVLM 在 100 页 document 上生成 QA 又慢质量又差。所以他们搞了个 short-to-long pipeline：

1. 先 OCR 解析全文，拿到 section structure
2. 从全文随机采 8-15 页的 coherent segment
3. 只把这 8-15 页喂给 Seed 2.0 生成 QA
4. 把生成的 QA 放回完整 document context 里当 training instance

这样 generation 成本只和 segment length 成正比，但 training signal 覆盖了 full 32K-128K context。生成成本和 training 信号解耦了，非常经济。

还有个细节：为了避免 QA 在 segment 里有答案但放回全文后变得 ambiguous，他们要求 generator 在 question 里加 anchor，比如 "in the Introduction section" 或 "on pages 20-25"。没这个 anchor 的话，一个 "What is the reported revenue?" 在完整财务报告里可能有多个答案，就成 bad supervision 了。

人工验证了 100 个 QA pair，97 个完全正确，质量大概 97%。这个数据质量对于合成数据来说相当不错。

## 技术上几个关键点

### mRoPE 这件事

Qwen2.5-VL 用的不是 1D RoPE，是 mRoPE——把 position encoding 拆成 temporal、height、width 三个 component。visual tokens 共享 temporal index，spatial 在 2×2 patch grid 上。结果是 visual position index 的增长速度比 token count 慢很多。

这件事有两个 implications：
1. Dynamic-NTK 那套直接套 1D RoPE 的 scaling 公式不一定最优，paper 在 Appendix G.3 试了 2e6、4e6、8e6 三个 base，发现 4e6 比 2e6 略好，但 8e6 反而退化。mRoPE 的 position space 已经比 1D RoPE 稀疏了，再激进地 scaling 是 diminishing return。
2. 这也解释了为什么 128K 训练的 model 能直接 generalize 到 512K——visual position index 其实没增长那么快，model 见过的 position range 其实够用。

### 训练配置

5B tokens，max_len=131,072，batch 4M tokens（也就是 32 个 128K sequence）。AdamW，LR 1e-5 cosine decay 到 1e-6。Ulysses sequence parallelism size 2 + FSDP size 4，在 8 个 H20 node 上跑，总共 2.9K H20 hours。

算下来扩展一个 7B LVLM 的 context window 从 32K 到 128K 大概一天多搞完。这个 cost 在工业界算非常友好了。

## 结果怎么样

主结果上 MMProLong (7B) 在 MMLongBench 上 57.70 分，几乎追平 Qwen2.5-VL-32B 的 58.31，比原版 7B 的 50.59 高 7 分。

更 striking 的是 extrapolation：在 128K 训练，直接测 256K 和 512K，MMProLong 还有 53.80 平均分，原版 Qwen2.5-VL-7B 掉到 28.80。差了 25 分。

跨 domain transfer 也很惊人：
- MM-NIAH（webpage needle-in-haystack）从 20.0 提到 49.4，**+29 分**
- Long-video benchmark（Video-MME、MLVU、LongVideoBench）也都有 +2 到 +3 分提升，**完全没见过 video training data**
- VTCBench（vision-text compression）从 48.23 提到 52.73

这说明 LongPT 学到的是**一般的"长 multimodal context 里 sparse retrieval"能力**，不是 PDF-specific 的 capability。Recipe 的 inductive bias 没有 overfit 到 document format。

他们还把这个 recipe 应用到 Qwen3-VL-8B（已经 native 256K context、用 100B tokens 训过 long-context），MM-NIAH 上还能 +11.7 分，说明 recipe 不是 Qwen2.5-VL 专属。

## 我觉得最 clever 的几个设计

1. **Segment-level QA synthesis**：用 short context 生成 QA，用 long context 训练，把 generation cost 和 training signal 解耦。这种 "cheap short generation → expensive long training" 的不对称利用是个 elegant engineering 决策。

2. **Segment anchor**：用 "in the Introduction section" / "on pages 20-25" 避免 global ambiguity。看着简单但很关键，没这个 anchor 的话 QA 放回全文就变成 multiple valid answers 的 bad signal。

3. **Pool-native > Long-biased**：这个发现直接反驳了"训练和 evaluation length 匹配最好"的朴素直觉。

4. **不要 short data**：因为 VQA format 自带 short-context preservation，省了 short data mixing 的 complexity。这是 LVLM 相对 LLM 的一个优势。

## 我觉得不够的地方

1. **只测了 7B/8B**：30B/70B 上还成不成立？Mixture ratio 在大 model 上可能要调——大 model reasoning 能力更强，可能需要更多 reasoning data。

2. **mRoPE scaling 没深挖**：只试了三个 base frequency。mRoPE 的 visual position 稀疏性可能需要完全不同的 scaling law，paper 留了个 open question。

3. **OCR 失败的机制没分析**：为什么 OCR-full 掉 17 分？是 catastrophic forgetting 还是 attention pattern 被破坏？如果是后者，能不能通过 layer-wise LR 或 learning rate scheduling 来 mitigate？

4. **Long-video transfer 机制没解释**：只 show 了 transfer 效果，但 document VQA 为什么能 transfer 到 video？是因为 temporal attention 类似，还是因为 retrieval capability 本身 task-agnostic？

5. **没做 LongRL**：只做了 LongPT（SFT），如果加一个 retrieval correctness as reward 的 GRPO 阶段，可能能进一步提升 retrieval capability。

## 最 core 的 mental model

如果只记一件事，应该是这个：

**Long-context LVLM 的核心 capability 是 "selective retrieval in long multimodal context"**。

所有的 design choice 都指向这个核心：
- VQA >> OCR：因为 VQA 训 selective attention，OCR 训 dense attention
- Pool-native >> Long-biased：因为 retrieval 需要在多种 length 上 generalize
- 8:2 extraction:reasoning：因为 retrieval 是 bottleneck，reasoning 只是 regularization
- 不需要 short data：因为 instruction format 自动 preserve short-context

Long-context 不是"context 变长"这么简单，而是"selective retrieval 在长 context 上的 generalization"。这个 framing 对你 build LVLM intuition 应该有用。

## 相关链接

- Paper 本身（MMLongBench by same first author）：https://arxiv.org/abs/2505.10610
- Qwen2.5-VL：https://arxiv.org/abs/2502.13923
- Qwen3-VL：https://arxiv.org/abs/2511.21631
- How to Train Long-Context LMs Effectively (Gao et al.)：https://arxiv.org/abs/2410.02660
- Data Engineering for 128K (Fu et al.)：https://arxiv.org/abs/2402.10171
- MM-NIAH benchmark：https://arxiv.org/abs/2406.11230
- Dynamic-NTK Reddit 原帖：https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/
- RoPE 原始 paper：https://arxiv.org/abs/2104.09864
- LongRoPE：https://arxiv.org/abs/2402.13753
- VeOmni framework：https://arxiv.org/abs/2508.02317

---

# MMProLong: Long-Context Vision-Language Model Training 深度解析

## 1. Paper 的整体定位和 Motivation

这篇 paper 来自 ByteDance Seed + HKUST，做的事情非常 focused：**研究如何把一个 LVLM 的 context window 从 32K 扩展到 128K，并且建立一个可复用的训练 recipe**。

核心的 motivation 是这样的：现有 LVLM 的技术报告（Qwen3-VL [3], GLM-4.5V [14]）只给出了 native 128K context 的结果，但很少披露**具体怎么构造 long-context data、怎么 mix、length distribution 怎么选**这些 engineering 细节。这导致社区做 long-context LVLM 时缺少 empirical foundation。

这和 LLM 的情况形成对比——LLM 那边有 Effective Long-Context Scaling (Xiong et al. 2024) [34] 和 Data Engineering for 128K (Fu et al. 2024) [35] 这种 system study。但这篇 paper 强调，**LVLM 不能直接照搬 LLM 的 recipe**，因为 visual tokens 的 position indexing 完全不同（mRoPE 而非 1D RoPE），并且数据合成需要考虑 image-text interleaving。

让我先解释一下整个 training pipeline 的架构图：

```
┌────────────────────────────────────────────────────────────────┐
│  Document Pool (1.5M PDFs, 36.5M pages)                          │
│  ├── Academic papers / Books / Technical manuals                │
│  └── Domains: engineering, medicine, social science, biology     │
│                                                                  │
│  Filter: 32-50 pages → 32K-128K multimodal tokens                │
│  Render: PyMuPDF @ DPI=144 → page images                         │
│  Parse: OCR expert (Seed 2.0 finetune) → layout-aware blocks     │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  Two Candidate Data Categories                                   │
│                                                                  │
│  A) Long-Document VQA (winner)                                  │
│     ├── extract-single: 1-page evidence retrieval                │
│     ├── extract-multi: multi-page aggregation                   │
│     └── reasoning: numerical/logical ops over evidence          │
│                                                                  │
│  B) OCR Transcription (loser)                                   │
│     ├── OCR-full: transcribe all pages                          │
│     └── OCR-needle: transcribe 1-3 selected pages               │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  LongPT (Long-Context Continued Pre-Training)                     │
│  Backbone: Qwen2.5-VL-7B (32K native → 128K)                   │
│  mRoPE base freq: 1e6 → 4e6 (Dynamic-NTK)                      │
│  Budget: 5B tokens, max_len=131,072, batch=4M tokens            │
│  Parallelism: Ulysses SP=2, FSDP=4, 8×H20 nodes                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  Evaluation                                                      │
│  ├── MMLongBench-Doc / LongDocURL / SlideVQA @ 64K, 128K        │
│  ├── Extrapolation to 256K, 512K (no extra training)            │
│  ├── MM-NIAH (webpage needle-in-haystack)                       │
│  ├── VTCBench (vision-text compression)                         │
│  └── Long-video: Video-MME / MLVU / LongVideoBench              │
└────────────────────────────────────────────────────────────────┘
```

最终的 model 叫 **MMProLong**，在 7B 规模上达到了接近 Qwen2.5-VL-32B 的 long-context 性能，并且在 256K/512K 上 still 保持 53.80 平均分，而原版 Qwen2.5-VL-7B 掉到 28.80。

Reference: 
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Effective Long-Context Scaling: https://arxiv.org/abs/2309.16039
- Data Engineering for 128K: https://arxiv.org/abs/2402.10171
- VeOmni framework: https://arxiv.org/abs/2508.02317

---

## 2. 关键技术细节：mRoPE 和 Dynamic-NTK

这部分对你 build intuition 很关键，因为 LVLM 的 position encoding 和 LLM 完全不同，paper 里很多 design choice 都源于此。

### 2.1 mRoPE (Multimodal Rotary Position Embedding)

Qwen2.5-VL 用的是 **mRoPE** [74]，把 traditional 1D RoPE 拆成三个 component：temporal、height、width。这对应于把 position encoding 看作在 (t, h, w) 三维空间里的旋转。

形式化地，对于 head dimension $d$（split 成 6 个 sub-space），位置 $(t, h, w)$ 的 rotary embedding 可以写成：

$$R(t, h, w) = \text{diag}(R_t^{(0:d/6)}, R_t^{(d/6:2d/6)}, R_h^{(2d/6:3d/6)}, R_h^{(3d/6:4d/6)}, R_w^{(4d/6:5d/6)}, R_w^{(5d/6:d)})$$

其中 $R_t, R_h, R_w$ 都是 standard 2D rotation matrices：

$$R_p^{(i)} = \begin{pmatrix} \cos(p\theta_i) & -\sin(p\theta_i) \\ \sin(p\theta_i) & \cos(p\theta_i) \end{pmatrix}, \quad \theta_i = b^{-2i/d}$$

变量解释：
- $t$ = temporal index（video 帧序号，对单 image 是 0）
- $h, w$ = image 内的 pixel grid 坐标
- $b$ = RoPE base frequency（这就是 paper 里要 tune 的关键 hyperparameter）
- $d$ = head dimension
- 上标 $(a:b)$ 表示 slice 索引

**为什么这件事很重要**：在 pure text LLM 里，position index 是 1D 的，随着 token 数线性增长。但在 mRoPE 里，visual tokens 共享同一个 temporal index，且 $(h, w)$ 是按 2×2 patch unshuffle 后的 grid coordinate。所以 position index 增长速度比 token count 慢得多——一个 128K-token 的 long document 里，大部分 visual tokens 的 position index 可能只有几千。

**Intuition**：这导致 Dynamic-NTK 的 heuristic（直接 scale base freq 4×）不一定是最优的，paper 在 Appendix G.3 做了 ablation，结论是 4×10^6 不一定比 2×10^6 好，但 8×10^6 反而退化。这说明 mRoPE 的 position space 已经比 1D RoPE "稀疏"，进一步 scaling 是 diminishing return。

### 2.2 Dynamic-NTK

Dynamic-NTK [51] 是 Reddit 上 emozilla 提出的启发式方法，原 idea 是：当推理时 context length $L$ 超过训练长度 $L_{train}$，把 RoPE base frequency 从 $b$ 改成 $b'$：

$$b' = b \cdot t^{\frac{d}{d-2}}, \quad t = \frac{L}{L_{train}}$$

变量解释：
- $t$ = expansion factor（这篇 paper 是 128K/32K = 4）
- $d$ = head dimension
- $b = 10^6$ 是 original base
- $b' = 10^6 \cdot 4^{d/(d-2)} \approx 4 \times 10^6$ 当 $d$ 比较大时

paper 里把这个公式用作 **训练时**的初始化，而不是推理时——也就是直接把 model 的 mRoPE base 改成 $4 \times 10^6$ 然后做 LongPT，让 model 在新的 position encoding 下 fine-tune。

Reference:
- Dynamic-NTK 原帖: https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/
- NTK-aware scaling: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
- RoPE 原始 paper (Su et al. 2021): https://arxiv.org/abs/2104.09864
- YaRN (Peng et al. 2023): https://arxiv.org/abs/2309.00071
- LongRoPE (Microsoft, 2024): https://arxiv.org/abs/2402.13753

---

## 3. Data Synthesis Pipeline 深度讲解

这是 paper 的核心 contribution——怎么构造 multimodal long-context training data。

### 3.1 Document Pool 构造

| Statistic | Value |
|-----------|-------|
| Total documents | 1,537,504 |
| Total pages | 36,592,809 |
| Avg pages/doc | 23.80 |
| English | 96.22% |
| Chinese | 3.59% |
| Other | 0.19% |
| Page range filter | 32-50 pages |

**Intuition**：选 32-50 页这个范围有讲究。Qwen2.5-VL 用 2×2 pixel unshuffle，一页典型 document 在 DPI=144 下大约渲染成 1K-3K visual tokens，加上 text 大约 1K-2K tokens。32 页大约对应 32K multimodal tokens，50 页大约 128K——刚好覆盖 target context window 的下界和上界。

### 3.2 Long-Document VQA 的 Segment-Level Synthesis

这是我觉得 paper 里最 clever 的设计。直接让 LVLM 在 100 页 document 上生成 QA 又慢又差（context 太长 model 自己都读不准），所以他们用一个 **short-to-long** 的 pipeline：

```
Step 1: OCR parse → 拿到 section structure (title, section labels)
Step 2: 从 full document 随机采样一个 8-15 页的 coherent segment
Step 3: Seed 2.0 (作为 QA generator) 仅看这 8-15 页 → 生成 QA + evidence
Step 4: 把 QA 放回 full document context → 形成 long-context training instance
```

**关键技术细节**——Segment Anchor：为了避免 "global-context false positives"（QA 在 segment 内有明确答案，但放回 full document 后变得 ambiguous），paper 要求 generator 在 question 里加 explicit anchor，比如 "in the Introduction section" 或 "on pages 20-25"。

举例：
- ❌ Bad: "What is the reported revenue?"（在完整财务报告里有歧义）
- ✅ Good: "According to the Homemade Bitters recipe on Page 39, how long should the herbs soak in vodka?"

paper 在 Appendix D.4 做了人工验证：100 个 sampled QA pair 中 97 个完全正确，2 个答案错，1 个 evidence annotation 不准——quality 大约 97%。

**为什么这个 pipeline 高效**：QA generation 只需要 model 处理 8-15 页（~16-30K tokens），而最终 training instance 是 32-128K tokens。这意味着 **生成成本只和 segment length 成正比，而 training signal 覆盖了 full context length**。这是一个非常经济的 trade-off。

### 3.3 三种 VQA Task Type

| Task | Evidence | Example | Difficulty |
|------|----------|---------|-----------|
| extract-single | 1 page | "According to Page 39, how long should herbs soak?" | Easy retrieval |
| extract-multi | 2+ pages | "Based on Pages 6, 13, 19, list all risk factors." | Multi-hop retrieval |
| reasoning | 2+ pages + ops | "Difference between total consumption and imports for rice in 2020?" | Retrieval + computation |

**Intuition**：这三个 task 形成 difficulty hierarchy。前两个 test 纯 retrieval，第三个 test retrieval + 简单 numerical reasoning。paper 后面发现 **8:2 的 extraction:reasoning mixture** 最好（Table 2），说明 retrieval 是主要 bottleneck，reasoning 主要是为了 task diversity 防止 collapse。

### 3.4 OCR Transcription（对比 baseline）

OCR transcription 的 design philosophy 完全不同——它不要求 instruction-following，而是要求 model 把 visual content 逐字转写：

- **OCR-full**: 整个 document 全部 transcribe（dense image-text alignment）
- **OCR-needle**: 只 transcribe 1-3 页，其余作 distractor（retrieval-style）

**为什么 paper 想试 OCR**：直觉上 OCR transcription 强制 model 反复 attend 到 image content 并产生 long output，这应该会强化 long-distance image-text dependency。但实验表明这个直觉是错的。

---

## 4. 三个核心 Findings 详解

### Finding 1: VQA >> OCR Transcription

Table 1 是这个 finding 的核心证据。让我重新整理一下关键数据：

| Training Data | 64K AVG | 128K AVG | Overall AVG | Δ vs base |
|---------------|---------|----------|-------------|-----------|
| Qwen2.5-VL-7B (base) | 52.24 | 48.94 | 50.59 | — |
| extract-single | 56.86 | 54.53 | 55.69 | +5.10 |
| extract-multi | 58.02 | 55.77 | 56.90 | +6.31 |
| reasoning | 57.33 | 55.62 | 56.47 | +5.88 |
| OCR-full | 31.24 | 35.11 | 33.17 | **-17.42** |
| OCR-needle | 45.61 | 42.00 | 43.80 | -6.79 |
| OCR-full + SFT | 56.09 | 51.59 | 53.84 | +3.25 |
| OCR-needle + SFT | 54.06 | 50.83 | 52.44 | +1.85 |

**关键观察**：
1. OCR-full 直接掉 17 分！这是因为 OCR 任务 format 和 downstream VQA 评估 format 完全不 align——model 学会了 transcribe，但忘了怎么 follow instruction。
2. 加 SFT 后 OCR 恢复了一些，但还是不如纯 VQA training。
3. extract-multi 是最好的 single task（56.90），比 base 高 6.31 分。

**Intuition 解读**：这件事其实很 deep。表面看是 "format alignment" 问题，但更深层的原因是：**LongPT 的目标是让 model 学会 "在长 context 里检索关键信息"，而 OCR transcription 是 "在长 context 里复制所有信息"**。前者是 selective attention，后者是 dense attention。dense attention 任务反而会让 model 学会均匀 attend，破坏 selective retrieval 能力。

这跟 LLM long-context training 的发现一致——Effective Long-Context Scaling [34] 也发现 instruction-format 的 long QA 比 next-token prediction on long documents 更有效。

### Finding 2: Pool-Native > Long-Biased Length Distribution

这是 paper 最 surprising 的发现之一。他们比较两种 length distribution：

| Distribution | % samples ≥100K tokens | Page range |
|--------------|----------------------|------------|
| pool-native (default) | 23.6% | 32-50 pages |
| long-biased | 83.9% | 50-100 pages |

直觉上 long-biased 应该更好，因为它 expose model 更多 near-128K context。但 Figure 2 显示 pool-native 在三个 task 上分别 +1.3, +0.1, +1.7。

**Intuition 解读**：这其实可以从 RoPE 的 position extrapolation 角度理解。Long-context ability 不是 "在 128K 上能工作" 这个 binary capability，而是 **position interpolation 在 [32K, 128K] 全区间上都能准确 retrieve key info** 的 continuous capability。

如果 training 全部集中在 100K-128K，model 在 60K-80K 这种 intermediate length 上的 position encoding 反而不准确。pool-native 给了 model 在不同 absolute position 和不同 image-text relative distance 上的 dense supervision。

这跟 LongRoPE [26] 的发现类似——positional interpolation 需要 fine-grained calibration，不能 coarse-grained。

### Finding 3: Retrieval 是 Bottleneck，需要 8:2 Mixture

Table 2 的 grid search 结果：

| Extraction:Reasoning | 64K AVG | 128K AVG | Overall |
|---------------------|---------|----------|---------|
| 0:10 (all reasoning) | 57.33 | 55.62 | 56.47 |
| 2:8 | 58.02 | 54.24 | 56.13 |
| 4:6 | 56.35 | 55.11 | 55.73 |
| 6:4 | 58.79 | 55.75 | 57.27 |
| **8:2** | **59.56** | **55.84** | **57.70** |
| 10:0 (all extraction) | 57.49 | 56.40 | 56.94 |

**关键观察**：
- 全 reasoning (0:10) 比全 extraction (10:0) 略差（56.47 vs 56.94）
- 8:2 是 sweet spot（57.70）
- 纯 extraction 也不如 8:2，说明 reasoning 提供 task diversity

**Intuition 解读**：Long-context model 的 bottleneck 在 retrieval 不在 reasoning。这件事在 LLM 那边也有证据——RULER benchmark [34] 显示 retrieval task 在 long-context 里 degradation 最严重。8:2 的意思是：把主要 training signal 放在 retrieval，但保留少量 reasoning 作为 regularization，防止 model collapse 到 pure pattern matching。

### Finding 4: Pure Long-Context VQA 不破坏 Short-Context

这可能是最 counterintuitive 的发现。Table 3：

| Short Data % | Long-ctx AVG | Short-ctx AVG |
|-------------|--------------|---------------|
| base model | — | 66.47 |
| 0% | 57.70 | 65.48 (-0.99) |
| 20% | 55.57 | 66.53 (+0.06) |
| 40% | 57.01 | 66.14 (-0.33) |
| 60% | 56.95 | 66.05 (-0.42) |
| 80% | 56.60 | 66.17 (-0.30) |

**关键观察**：纯 long-context training (0% short data) 只让 short-context 掉 0.99 分！这跟 LLM long-context training 的经验完全相反——LLM 那边不加 short data mixing 会让 short-context 严重退化 [16]。

**Intuition 解读**：paper 给的解释是 long-document VQA 的 format 本身就是 instruction-following format，所以即使 context 变长，model 仍然在 practice "instruction following" 这个能力。而 LLM long-context training 通常用 books/code 这种 next-token prediction，完全离开了 instruction format，所以需要 short data 来 maintain instruction-following 能力。

这进一步支持 Finding 1——VQA 的 instruction-format 不只是 downstream 评估 align，还自带 short-context preservation 的 inductive bias。

---

## 5. Final Recipe 和训练细节

### 5.1 Final Configuration (Table 6)

| Component | Setting |
|-----------|---------|
| Backbone | Qwen2.5-VL-7B-Instruct (orig mRoPE base 1e6) |
| mRoPE base for 128K | 4e6 |
| Max seq length | 131,072 (128K) |
| Long data | extract-single (40%) + extract-multi (40%) + reasoning (20%) |
| Length distribution | Pool-native (natural 32K-128K) |
| Short data | None (pure long-context) |
| Token budget | 5B tokens (~2.9K H20 hours) |
| Optimizer | AdamW (wd=0.1, β1=0.9, β2=0.95) |
| LR | 1e-5, 10% warmup, cosine decay to 1e-6 |
| Batch size | 4M tokens (32 sequences × 128K) |
| Framework | VeOmni + FlashAttention |
| Parallelism | Ulysses SP=2, FSDP=4 |

**Intuition on cost**：5B tokens × 7B params × 6 (forward+backward) ≈ 2.1e17 FLOPs。H20 大约 200 TFLOPS BF16，所以理论下界 ~1000 GPU-hours，实际 2.9K H20 hours 算 reasonable（含 communication overhead）。这意味着 **扩展 context window 的 LongPT 可以在 ~1 天内用 64 张 H20 完成**——这是个非常 data-efficient 的 recipe。

### 5.2 公式化的 Training Objective

虽然 paper 没明确写，但 LongPT 实际上就是 supervised fine-tuning on long-context VQA，loss 是 standard cross-entropy on answer tokens：

$$\mathcal{L} = -\sum_{t=1}^{T_{ans}} \log p_\theta(a_t | a_{<t}, x_{ctx}, q)$$

变量解释：
- $a_t$ = answer 的第 $t$ 个 token
- $a_{<t}$ = answer 前缀
- $x_{ctx}$ = 长 context（image+text interleaved，32K-128K tokens）
- $q$ = question
- $\theta$ = model parameters
- $T_{ans}$ = answer 长度（通常很短，<<context length）

注意这里 context tokens 不参与 loss（only answer tokens contribute），这是 instruction-tuning 的标准做法。

---

## 6. 主结果和 Generalization

### 6.1 Long-Document VQA 主结果 (Table 4)

最有意思的对比：

| Model | Size | 64K AVG | 128K AVG | Overall |
|-------|------|---------|----------|---------|
| **MMProLong** | **7B** | **59.56** | **55.84** | **57.70** |
| Qwen2.5-VL-7B | 7B | 52.24 | 48.94 | 50.59 |
| InternVL3-8B | 8B | 50.19 | 44.11 | 47.15 |
| InternVL3-14B | 14B | 50.67 | 44.27 | 47.47 |
| Qwen2.5-VL-32B | 32B | 60.55 | 56.08 | 58.31 |
| Qwen2.5-VL-72B | 72B | 62.81 | 58.85 | 60.83 |
| GPT-5.4 | — | 73.68 | 63.01 | 69.41 |
| Gemini-3.1-Pro | — | 83.55 | 83.77 | 83.66 |

**关键 takeaway**：MMProLong (7B) 几乎追平 Qwen2.5-VL-32B（57.70 vs 58.31），用了 5B tokens 的 LongPT budget。这是个非常 cost-effective 的 scaling。

### 6.2 Extrapolation to 256K and 512K (Table 5)

这是 paper 最 striking 的结果之一——**128K 训练的 model 直接在 512K 上 test，还能 work**：

| Model | 256K AVG | 512K AVG | Overall |
|-------|----------|----------|---------|
| MMProLong | 55.09 | 52.52 | 53.80 |
| Qwen2.5-VL-7B | 38.12 | 19.49 | 28.80 |
| Gemma3-4B | 32.52 | 15.51 | 24.02 |
| Gemma3-12B | 47.37 | 23.51 | 35.44 |

**Intuition**：这件事和 mRoPE 的特性有关。mRoPE 的 visual position 增长慢，所以即使 token count 到 512K，actual position index 仍然在 model 见过的范围内。Dynamic-NTK 的 base scaling 也提供了 position extrapolation 的 robustness。这跟 LLM 那边 Qwen2.5-1M [36] 类似——只要 training recipe 给了足够 diverse 的 length distribution，extrapolation 是 free 的。

### 6.3 Transfer to MM-NIAH (Figure 4)

MM-NIAH [17] 是 webpage-based needle-in-haystack benchmark，test retrieval / counting / reasoning 三类任务。

| Model | Ret. | Count | Reas. | AVG |
|-------|------|-------|-------|-----|
| MMProLong | 74.83+57.83/2=66.33 | 27.67+8.67/2=18.17 | 67.33+60.33/2=63.83 | 49.4 |
| Qwen2.5-VL-7B | (50+11.33)/2=30.67 | (6+16.33)/2=11.17 | (27.5+8.83)/2=18.17 | 20.0 |

MMProLong 在 MM-NIAH 上从 20.0 提到 49.4，**+29.4 分**！这个 transfer 是惊人的，因为 MM-NIAH 是 webpage haystack，和 training 用的 PDF document 完全不同 domain。

**Intuition**：这说明 LongPT 学到的是 **general 的 "long multimodal context 里的 sparse evidence retrieval" 能力**，而不是 PDF-specific 的 capability。Recipe 的 inductive bias 没有 overfit 到 document format。

### 6.4 Transfer to Long-Video (Figure 5)

| Model | Video-MME | MLVU | LongVideoBench |
|-------|-----------|------|----------------|
| Qwen2.5-VL-7B | 65.1 | 70.2 | 60.43 |
| MMProLong | 67.78 | 73.55 | 62.08 |

Long-video benchmark 完全没见 video training data，但 MMProLong 在 Video-MME 上 +2.68，MLVU 上 +3.35。这说明 long-document VQA 训练的 long-context retrieval 能力 transfer 到 temporal sequence 上。

### 6.5 Transfer to Qwen3-VL-8B (Appendix G.7)

为了证明 recipe 不是 Qwen2.5-VL 专属，paper 把 recipe 应用到 Qwen3-VL-8B（这个 model 已经是 native 256K context，且经过 100B tokens 的 long-context pre-training）：

| Model | Long-doc AVG | MM-NIAH AVG |
|-------|--------------|-------------|
| Qwen3-VL-8B | 65.62 | 50.03 |
| + MMProLong recipe | 67.51 (+1.9) | 61.75 (+11.7) |

Long-doc 上提升有限（因为 Qwen3-VL 已经很强），但 MM-NIAH 上 +11.7 分，说明 recipe 在更强 backbone 上仍然 effective，特别是在 retrieval 任务上。

---

## 7. 整体 Recipe 的 Intuition 总结

让我把所有发现串成一个 coherent 的 mental model：

**Long-context LVLM 的核心 capability 是 "selective retrieval in long multimodal context"**。

这个 capability 有几个 characteristics：
1. **Format-sensitive**：必须是 instruction-following format（VQA），不是 dense transcription（OCR）
2. **Length-continuous**：不能 specialized 到单一 target length，需要 diverse length supervision
3. **Retrieval-dominant**：主要 bottleneck 是 retrieval，不是 reasoning，但需要少量 reasoning 作 regularization
4. **Self-preserving**：instruction-format 的 long data 自动 preserve short-context ability
5. **Transferable**：学到的是 general long-context capability，跨 domain（PDF→webpage→video）都 transfer

**最终 recipe 的 mental model**：
- 数据 = 高质量 instruction-format long VQA + diverse length + retrieval-heavy mixture
- 训练 = 适度 mRoPE scaling + 5B tokens LongPT + 128K max length
- 评估 = 在 64K/128K train length 上 evaluate，并 test 256K/512K extrapolation

这跟 LLM long-context training 的 recipe 有共性（data mixture 重要、length diversity 重要），但有关键区别：LVLM 因为 mRoPE 的 visual position 稀疏性，需要重新思考 position encoding 和 length scaling 的关系；LVLM 的 instruction-format 数据天然 preserve short-context，省去了 short data mixing 的 complexity。

---

## 8. 我对这篇 Paper 的 Critique 和思考

几个我想 highlight 的点：

### 8.1 Strengths

1. **Ablation 非常 systematic**：每个 design choice 都有 controlled experiment，包括 data type、length distribution、mixture ratio、short data ratio、mRoPE base。这种 engineering rigor 在 LVLM 领域不多见。

2. **Extrapolation 实验很有说服力**：在 128K 训练，直接测 256K/512K 还能保持 53.80 分，这比单纯报告 128K 性能更有意义。

3. **跨 domain transfer 实验**：MM-NIAH (webpage) 和 long-video 的 transfer 证明了 recipe 学到的是 general capability，不是 document-specific。

### 8.2 Limitations and Open Questions

1. **只测了 7B/8B 规模**：paper 自己在 Section H 承认这点。30B/70B 上是否还成立？Mixture ratio 在更大 model 上可能不同——大 model 的 reasoning 能力更强，可能需要更多 reasoning data。

2. **mRoPE base frequency 的 ablation 不够深入**：只测了 {2e6, 4e6, 8e6} 三个点。考虑到 mRoPE 的 position space 稀疏性，可能需要完全不同的 scaling law。这是 paper 留给社区的重要 open question。

3. **Long-video transfer 的 mechanism 没解释清楚**：paper 只 show 了 transfer 效果，但没分析为什么 document VQA 能 transfer 到 video。是因为 temporal attention 机制类似？还是因为 retrieval capability 本身是 task-agnostic 的？

4. **OCR transcription 的失败分析不够**：为什么 OCR-full 会掉 17 分？是 catastrophic forgetting 还是 attention pattern 被破坏？如果是后者，是不是可以通过 learning rate scheduling 或者 layer-wise LR 来 mitigate？paper 没深挖。

5. **Evaluation 用 LLM judge 有 risk**：paper 在 Section H 提到这个 limitation。LLM judge 对 list-style answer 的 F1 评估可能 systematic bias 某些 answer format。

### 8.3 和 LLM long-context 文献的对比

paper 引用了 [16] (Tianyu Gao et al., 2024) "How to train long-context language models (effectively)"，这是 LLM 那边最 system 的 study。两者的对比：

| Aspect | LLM (Gao et al.) | LVLM (this paper) |
|--------|-----------------|-------------------|
| Data format | Books/code (next-token) | Long-doc VQA (instruction) |
| Length distribution | Pool-native (SlimPajama natural) | Pool-native (PDF natural) |
| Short data mixing | Required | Not required |
| Position encoding | 1D RoPE | mRoPE |
| Bottleneck | Long-range dependency | Multimodal retrieval |
| Training budget | ~10B tokens | 5B tokens |

最有意思的对比是 **short data mixing 的不同结论**——LLM 需要 short data mixing 因为 books/code 不是 instruction format；LVLM 不需要因为 VQA 本身就是 instruction format。这说明 **数据 format 而不是数据 length 决定 short-context degradation**。

Reference:
- How to train long-context LMs: https://arxiv.org/abs/2410.02660
- RULER benchmark: https://arxiv.org/abs/2404.06654

---

## 9. 个人觉得最 Clever 的几个 Design Choice

1. **Segment-level QA synthesis**：用 short context 生成 QA，但用 full context 训练。这把 generation cost 和 training signal decouple 了。这种 "cheap short-context generation → expensive long-context training" 的 asymmetry 是个非常 elegant 的 engineering 决策。

2. **Segment anchor in question**：通过 "in the Introduction section" / "on pages 20-25" 这种 anchor 来 avoid global ambiguity。这是个看似简单但很关键的设计——没有这个 anchor，QA 在 long context 里会变成 multiple valid answers 的 bad training signal。

3. **8:2 extraction:reasoning mixture**：而不是 50:50 或纯 extraction。这个比例反映了 "retrieval 是 bottleneck，但 reasoning 提供 task diversity regularization" 的 insight。

4. **Pool-native > Long-biased**：这个发现直接反驳了 "训练和 evaluation length match 最好" 的 naive intuition，支持了 "long-context ability 是 continuous capability" 的观点。

---

## 10. 我会怎么 Extend 这篇工作

如果让我 follow up，几个方向：

1. **mRoPE-specific scaling law**：systematically study mRoPE base frequency 和 visual position index distribution 的关系。可能需要一个 visual-aware 的 NTK scaling，而不是直接套用 1D RoPE 的公式。

2. **Multi-modal needle type diversity**：现在 MM-NIAH 主要 text/image needle，可以扩展到 audio、table、chart、formula 等 modality-specific needle，研究 cross-modal retrieval 的 scaling behavior。

3. **Long-context RL 阶段**：paper 只做了 LongPT（SFT），没做 LongRL。可能用 retrieval correctness as reward signal 做 GRPO 会进一步提升 retrieval capability。

4. **Reasoning data 的 sweet spot 重新 search**：在更强 backbone（Qwen3-VL-8B）上 8:2 可能不再是 optimal，因为大 model 的 reasoning 能力更强，可能 6:4 或 5:5 更好。Table 20 已经 hint 这个 trend。

5. **Mechanistic interpretability**：用 probe 或 attention map 分析 LongPT 前后 model 的 attention pattern 变化，验证 "retrieval 是 bottleneck" 这个 hypothesis 在 representation level 是否也成立。

---

## 11. 总结

这篇 paper 的核心贡献是给 LVLM long-context training 建立了一个 empirical foundation。最终 recipe 非常简洁：

> **用 instruction-format 的 long-document VQA，以 pool-native length distribution 训练，8:2 的 extraction:reasoning mixture，加 Dynamic-NTK 的 mRoPE scaling，5B tokens 就能把 7B LVLM 从 32K 扩到 128K，并 generalize 到 512K。**

这个 recipe 的 beauty 在于它揭示了 long-context LVLM training 的几个 counterintuitive 真相：
- 数据 format 比 data length 更重要（VQA >> OCR）
- Length diversity 比 length match 更重要（pool-native >> long-biased）
- Retrieval 比 reasoning 更重要（8:2 mixture）
- Instruction format 自动 preserve short-context（不需要 short data mixing）

这些 finding 对你 build LVLM 的 intuition 应该有帮助——**long-context 不是 "context 变长"，而是 "selective retrieval 在长 context 上的 generalization"**。所有的 design choice 都指向这个核心 capability。

paper 也 leave open 一些 important questions，特别是 mRoPE scaling 在更大 model / 更长 context 上的 behavior。希望社区能在这个 empirical foundation 上继续 extend。

---

### 参考资源链接

- Paper arXiv: https://arxiv.org/abs/2505.10610 (MMLongBench by same first author)
- Qwen2.5-VL technical report: https://arxiv.org/abs/2502.13923
- Qwen3-VL technical report: https://arxiv.org/abs/2511.21631
- Effective Long-Context Scaling (Xiong et al.): https://arxiv.org/abs/2309.16039
- Data Engineering for 128K (Fu et al.): https://arxiv.org/abs/2402.10171
- How to Train Long-Context LMs Effectively (Gao et al.): https://arxiv.org/abs/2410.02660
- MM-NIAH benchmark: https://arxiv.org/abs/2406.11230
- MMLongBench-Doc: https://arxiv.org/abs/2407.12915
- LongDocURL: https://aclanthology.org/2025.acl-long.66/
- SlideVQA: https://arxiv.org/abs/2301.04892
- Video-MME: https://arxiv.org/abs/2405.21075
- MLVU benchmark: https://arxiv.org/abs/2406.04264
- LongVideoBench: https://arxiv.org/abs/2407.15754
- VTCBench: https://arxiv.org/abs/2512.15649
- VeOmni framework: https://arxiv.org/abs/2508.02317
- FlashAttention-2 (Tri Dao): https://arxiv.org/abs/2307.08691
- RoPE (Su et al.): https://arxiv.org/abs/2104.09864
- YaRN: https://arxiv.org/abs/2309.00071
- LongRoPE: https://arxiv.org/abs/2402.13753
- Dynamic-NTK Reddit post: https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/
- LLaVA-OneVision: https://arxiv.org/abs/2408.03326
- InternVL3: https://arxiv.org/abs/2504.10479
- Gemma 3 technical report: https://arxiv.org/abs/2503.19786
- GLM-4.5V: https://arxiv.org/abs/2507.01006
