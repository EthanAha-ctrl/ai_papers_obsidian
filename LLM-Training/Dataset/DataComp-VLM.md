---
source_pdf: DataComp-VLM.pdf
paper_sha256: c3638247061522865046e8ca44d954fe35e538da878d93516e0c5cd888f3c710
processed_at: '2026-08-03T18:17:57-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 DCVLM

## 这篇 paper 在干嘛

现在训 VLM 的人有个坏习惯——technical report 里写"我们用了各种数据、各种比例、各种 filtering"，但具体怎么配的全是黑箱。你说你 model 强，到底是因为 architecture 好、还是 data 配得好、还是 filter 用得妙？分不清。

DCVLM 干的事很简单：**把 model 和训练 recipe 钉死，只让 data 变**，然后系统地测各种 data curation 策略哪个有用。这个套路从 DataComp（CLIP）和 DCLM（LLM）来的，第一次搬到 VLM 上。

## 核心发现，一句话版本

**filtering 基本没用，mixture 比例才是关键，而且最优比例会随 scale 变。**

## 发现一：Filtering 为啥没用

过去训 CLIP、训 LLM，大家都爱搞 quality filtering——CLIP-score 筛图、fasttext 筛文本、perplexity 筛这那的。效果确实好。

但 DCVLM 测了一堆 filter（CLIP-score、text quality、image quality、multimodal perplexity...），在 2B/25B scale 上最好的也就 +1.1pp，大多数持平或变差。

**原因很简单**：现在 VLM 用的数据不是 raw CommonCrawl 了。DataComp-1B、ReLAION 这些 source datasets，人家原作者已经帮你 CLIP-score filter 过一遍了。你再 filter 一层，就是在"已经筛过的干净数据"上再筛，边际收益断崖式下跌。

paper 做了个漂亮的验证实验：把 pool 里"已过滤"数据占比从 25%→65%→100% 逐步提高，downstream filter 的 gain 从 +2.4pp → +1.3pp → +0.6pp。**diminishing returns 明明白白。**

**人话**：你妈已经把米洗干净了，你再洗一遍只会浪费水。

## 发现二：Mixture 才是关键，而且会"反转"

最有意思的实验：固定 text=15%、multimodal docs=5%，只调 image-caption vs instruction-tuning 的比例。

三个配方：
- **Caption-heavy**: 65% 图文对 + 15% instruction
- **Balanced**: 40% + 40%
- **Instruction-heavy**: 10% + 70%

结果在 1B×6.25B（最小 scale）：Caption-heavy 最好，Instruction-heavy **最差**。

但到了 2B×25B 和 4B×25B：**Instruction-heavy 反超成第一**。

这个 crossover pattern 是整篇 paper 最 practical 的 insight：**小 scale 上选出的最优 mixture，到大 scale 上可能是最差的。**

如果你只在小 scale 上调 mixture（很多实验室都这么干省卡），你会选 Caption-heavy，然后大 scale 训出来发现不如人家用 Instruction-heavy 的。

**为啥 instruction-heavy 在大 scale 更好？** 直觉上：instruction data 信息密度高，每个 token 都带"task-relevant"信号。小模型 capacity 不够，吃不下这种密度，反而 caption 那种"简单图文对"更适合打底。一旦 capacity 超过某阈值，instruction data 的信息优势就显现了。

## 发现三：重复数据没那么怕

Instruction data 天然小，70% 比例意味着要重复好几遍。都知道重复会 degrade，但 paper 测下来：

- 2× 重复：-1.5pp
- 4× 重复：-1.9pp  
- 8× 重复：-3.1pp

而 Instruction-heavy 哪怕 2× 重复（50.2%），还是打平 Caption-heavy 的 unique data（50.3%）。**好配方的收益 > 适度重复的代价。** 到 8× 才真的扛不住。

## 发现四：Pretraining 性能完美预测 SFT 后性能

担心 SFT 会"覆盖"pretraining 的数据选择效果？测了 54 个 SFT run，Pearson r = 0.99。

**人话**：pretraining 训得好，SFT 后就好；pretraining 训得差，SFT 后也差。数据 curation 的效果不会被 SFT 抹掉。这意味着你做 curation 研究可以只看 pretraining metrics，省掉 SFT 这一步。

## 最终产出：DCVLM-BASELINE

配方简单粗暴，不搞任何 filtering，就用 Instruction-heavy mix：

```
10%  image-caption pairs
5%   multimodal documents  
15%  text-only
70%  instruction-tuning data
```

结果（33-task Core set）：
- 8B/200B：63.6% vs FineVision 58.2%（+5.4pp）
- **4B/100B（58.9%）超过 8B/200B FineVision（58.2%）**——4 倍 compute 省了

## 为啥这个工作重要

它给 VLM 社区传递一个信号：**别再盲目堆 quality filter 了，那是在已经干净的数据上做无用功。把精力花在 mixture 比例上，而且一定要跨 scale 验证。**

这和 LLM 领域的教训类似——Chinchilla 之前大家盲目堆参数，Chinchilla 之后才意识到 data/token ratio 才是关键。DCVLM 对 VLM data curation 可能是个类似的"清醒时刻"。

---

# DataComp-VLM: VLM 数据 Curation 的系统化 Benchmark

## 1. 核心问题与动机

这篇 paper 解决一个非常实际的问题：modern autoregressive VLMs 的 pretraining data curation 策略一直是 "黑箱"。像 InternVL3、Qwen2.5-VL、LLaVA-OneVision 这些 SOTA VLM 的 technical reports 里，mixture ratios、filtering criteria、formatting choices 大多是 proprietary 的，甚至 poorly documented。

DataComp 范式（从 CLIP 领域起源）的核心思想是：**fix model architecture 和 training procedure，只 vary data**，从而 isolated measurement data-centric interventions 的效果。DCVLM 把这个范式扩展到了 autoregressive VLMs。

reference:
- DataComp (CLIP): https://arxiv.org/abs/2304.14108
- DCLM (LM): https://arxiv.org/abs/2406.11794
- FineWeb: https://arxiv.org/abs/2406.17557
- InternVL3: https://arxiv.org/abs/2504.10479
- FineVision: https://arxiv.org/abs/2510.17269

## 2. VLM Data Curation 的四大挑战

Paper 指出 VLM data curation 比 LLM 或 CLIP 更难的四个原因：

**(1) Aggregation-based regime**: 不同于 early CLIP 直接 train on raw CommonCrawl，modern VLMs 是把已经 upstream curated 的 datasets 聚合起来。这些 source datasets 已经经过了不同程度的 upstream filtering（CLIP-score filtering 等），所以 additional downstream filtering 的效果变得 unclear。

**(2) Scale gap**: 现有 open training datasets（如 LLaVA-OneVision、Nemotron-VL）operate 在 millions of samples 级别，远低于 SOTA models 用的 trillions of tokens。

**(3) Design space 过大**: data types × model scale × training budget 三个维度交互形成的 design space 太大，无法 exhaustive search。

**(4) Evaluation 不标准化**: 不同 paper 用不同 benchmark suites，难以 fair comparison。

## 3. DCVLM Benchmark 设计

### 3.1 Data Pool 构造

DCVLM pool 聚合了 **160 个公开 datasets**，组织成四种 data types：

| Data Type | Datasets 数量 | Tokens 占比 | Samples 占比 | 平均 tok/sample |
|-----------|--------------|------------|--------------|-----------------|
| Image-caption pairs | 13 | 83% | 74% | ~1.5K |
| Multimodal documents | 5 | 7% | 4% | ~3K |
| Text-only | 33 | 5% | 19% | ~440 |
| Instruction-tuning | 109 | 5% | 3% | ~2.2K |

**Total**: 3.9B samples, 6T multimodal tokens (InternVL-2.5 tokenizer 测量)

这里有个关键的直觉观察：image-caption pairs 在 samples 和 tokens 两个维度都 dominate，因为每张图至少贡献 256 visual tokens (post pixel-shuffle)，这"虚增"了 per-sample token count。Text-only 则相反：19% samples 但只有 5% tokens，因为 text 样本短且无 visual tokens。

Multimodal instruction-tuning data 进一步细分为 8 个 capability categories（following FineVision）：
- Captioning & Knowledge
- Chart & Table understanding
- General QA
- Grounding & Counting
- Math
- Naive OCR
- OCR-QA
- Science

### 3.2 Model Architecture

采用 InternVL3 风格的 modular architecture：

```
[Image] → InternViT-300M-448px-V2.5 → [visual tokens]
                                              ↓
                                    [2-layer MLP Projector]
                                              ↓ (pixel shuffle, 4× reduction)
                                    [256 tokens per tile]
                                              ↓
[Text tokens] + [visual tokens] → Qwen2.5-Base LM → [output]
```

**Vision Encoder** (InternViT-300M):
- Input: 448×448 per tile (dynamic tiling via AnyRes)
- Patch size: 14×14
- Pre-shuffle tokens: 32×32 = 1024 per tile
- Post-shuffle: 16×16 = 256 per tile (4× reduction)
- 24 layers, hidden 1024, 16 heads, head dim 64
- ~304M params

**Projector** (2-layer MLP):
- Pixel shuffle factor 0.5: 1024 tokens × D_V → 256 tokens × 4D_V
- LayerNorm → Linear(4D_V → D_LM) → GELU → Linear(D_LM → D_LM)
- 只有 hidden width D_LM 随 scale 变化

**LM Backbone** (Qwen2.5-Base):
- 0.5B / 1.5B / 3B / 7B 四个 size
- 共享 Qwen2 architecture: SwiGLU FFN, RMSNorm, RoPE, GQA
- 关键差异：0.5B head dim=64, 其余 128；只有 7B 不 tied embeddings；3B 层数 36 比 7B 的 28 还多

reference: https://arxiv.org/abs/2412.15115 (Qwen2.5)

### 3.3 Scaling Ladder

设计四个 scale，每个 step 对应 8× compute 增加（model 2× + tokens 4×）：

| Scale | N (params) | D (tokens) | Pool size | Vision init | LLM init | H100 hrs |
|-------|-----------|-----------|-----------|-------------|----------|----------|
| small | 1B (0.80B) | 6.25B | 187.5B | InternViT-300M | Qwen2.5-0.5B | 80 |
| medium | 2B (1.85B) | 25B | 750B | InternViT-300M | Qwen2.5-1.5B | 640 |
| large | 4B (3.40B) | 100B | 3T | InternViT-300M | Qwen2.5-3B | 5,120 |
| x-large | 8B (7.95B) | 200B | 6T | InternViT-300M | Qwen2.5-7B | 20,480 |

**关键设计**: pool-to-training ratio 固定在 30×，这意味着在所有 scale 都可以做 aggressive filtering 同时保持 constant data repetitions。这是为了让 small scale 的 filtering 实验能 transfer 到 large scale。

### 3.4 Training Recipe

```
Optimizer: AdamW (β1=0.9, β2=0.999, ε=1e-8)
LR: 2×10^-5 (cosine decay, 3% warmup)
Weight decay: 0.01
Precision: BF16
Global batch size: 1024
Max sequence length: 8192 tokens (packed)
Max packed images: 24 tiles
Loss reduction: Square-averaging (InternVL 风格)
```

Learning rate 的选择经过 sweep: 在 {2e-4, 4e-5, 2e-5, 8.91e-6, 2e-6} 中，2e-5 在所有 scale 都最优。LR 太高 (2e-4) 在 1B 上 collapse 到 near-chance，太低 (2e-6) underfit。

### 3.5 Evaluation Protocol

从 65 个候选 benchmarks 开始，经过两层 filter：

**(1) Monotonicity filter**: 训练 small 和 medium scale（3 seeds 各一次），保留 performance 从 small 到 medium 单调提升的 benchmarks。这步过滤掉 12 个。

**(2) Stability filter**: 计算各 benchmark 的 seed variance，POPE 因 16% variance 被剔除。

最终组织成三个 nested tiers：
- **Validation set** (13 benchmarks): 快速迭代
- **Core set** (33 benchmarks): 主结果报告
- **Extended set** (52 benchmarks): 全面分析

9 个 domains: General, Knowledge, OCR & Charts, Vision-Centric, Multilingual, Text-Only, Safety, Hallucination, Reasoning。Safety/Hallucination/Reasoning 推到 Extended tier。

## 4. 核心发现一：Filtering 没用

### 4.1 实验设置

测试 60+ filter configurations，包括：

- **CLIP-score**: OpenAI CLIP ViT-L/14, DFN-CLIP, SigLIP-2-B/16@384
- **Text quality**: DCLM fasttext, NVIDIA Nemotron/Mixtral edu classifiers
- **Multimodal filters**: UniFilter (Qwen2.5-1.5B, Qwen3-0.6B), perplexity-based (text-only PPL, multimodal PPL, Conditional Mutual Information)
- **Image Quality Assessment**: ARNIQA on Kadid-10k, TID-2013

两种 filtering paradigms：
- **Local filtering**: 每个 source dataset 内独立计算 percentile threshold，保持 global mixture
- **Global filtering**: 整个 pool 计算单一 threshold，隐式 reshape mixture

Small scale 保留 top-10%，medium scale 保留 top-40%（依据 prior work 小模型 benefit from aggressive filtering）。

### 4.2 实验结果（Medium Scale, 2B model, 25B tokens）

```
Filter                        | Val Avg | Δ vs No Filter
------------------------------|---------|--------------
No Filter                     |  53.4   |    0.0
SigLIP-2 (global, im-cap)     |  54.5   |   +1.1  ← best
OpenAI-CLIP (global)          |  54.0   |   +0.6
DFN-CLIP (global)             |  53.7   |   +0.3
DCLM FastText (local)         |  53.8   |   +0.4
NVIDIA Mixtral (local, text)  |  54.1   |   +0.7
UniFilter Qwen3-0.6B (local)  |  52.9   |   -0.5
Conditional MI (local)        |  53.8   |   +0.4
...大多数 filter 要么持平要么变差
```

**核心结论**: 即使最好的 filter (SigLIP-2 global) 也只带来 +1.1pp 提升，远低于 prior work 在 CLIP/LM 领域的 gains（通常 3-10pp）。

### 4.3 为什么 Filtering 没用？

**Hypothesis**: 现有 VLM datasets 已经被 upstream curation 过，没有 significant noise 可去除。

验证实验：构造三个 sub-pools，pre-filtered 比例分别为 25%、65%、100%：

| Pre-filtering level | Gain from downstream CLIP filtering |
|---------------------|-------------------------------------|
| 25% (mostly raw)    | +2.4pp |
| 65%                 | +1.3pp |
| 100% (fully curated)| +0.6pp |

这清晰地展示了 **diminishing returns**：当 pool 已经被 upstream filtered，additional downstream filtering 进入边际收益递减区间。

reference: 
- DataComp-CLIP 的 CLIP-score filtering: https://arxiv.org/abs/2304.14108
- DCLM fasttext classifier: https://arxiv.org/abs/2406.11794
- Goyal et al. scaling laws for data filtering: https://arxiv.org/abs/2311.18755

## 5. 核心发现二：Mixing 是关键

### 5.1 实验设置

固定 text-only=15%, multimodal docs=5%，sweep image-caption vs instruction-tuning 的比例：

| Mixture | Image-caption | Instruction-tuning | Text | MM Docs |
|---------|---------------|--------------------|----|---------|
| Caption-heavy | 65% | 15% | 15% | 5% |
| Balanced | 40% | 40% | 15% | 5% |
| Instruction-heavy | 10% | 70% | 15% | 5% |

在 3 model sizes (1B, 2B, 4B) × 3 token budgets (6.25B, 12.5B, 25B) 的 grid 上评估。

### 5.2 Scale-Aware Crossover 现象

最 striking 的发现是 mixture ranking **不能 scale-agnostic 地确定**：

```
Training compute →   1B×6.25B   1B×25B   2B×6.25B   2B×25B   4B×25B
-----------------------------------------------------------------
Caption-heavy        BEST        2nd      ~equal     3rd       3rd
Balanced             2nd         3rd      ~equal     2nd       2nd
Instruction-heavy    WORST       BEST     ~equal     BEST      BEST
```

**Crossover pattern**: Instruction-heavy 在 small scale 是 worst，但随着 model size 和 token budget 增加，它的 scaling slope 明显更陡，最终成为 best。

**Practical implication**: 如果只在 small scale (1B×6.25B) 做 mixture 优化，会选 Caption-heavy，错过真正最优的 Instruction-heavy。这强调了 **scale-aware data curation** 的必要性。

### 5.3 为什么 Instruction-heavy 更 scalable？

一个直觉解释：instruction-tuning data 包含的 task format diversity 远高于 image-caption pairs。当 model capacity 增加时，它能从这种 diversity 中获得更多 inductive bias。Caption-heavy 在 small scale 帮助建立 basic visual grounding，但 capacity 一旦超过某阈值，instruction data 的信息密度优势就显现出来。

另一个因素：instruction data 多 turn、多 image 的 structure 更接近 downstream evaluation 的 format，某种意义上是 "test-like" training，减少了 train-test distribution gap。

### 5.4 重复数据的鲁棒性

Instruction-tuning 数据集天然比 web-crawled image-caption 小几个数量级。70% 比例需要 extreme repetition 填满 token budget。测试结果：

| Configuration | Core Avg | Δ |
|---------------|----------|---|
| Instruction-heavy, unique | 51.7 | 0.0 |
| ~2× repetition | 50.2 | -1.5 |
| ~4× repetition | 49.8 | -1.9 |
| ~8× repetition | 48.6 | -3.1 |
| Balanced (unique) | 50.9 | |
| Caption-heavy (unique) | 50.3 | |
| Base mix (unique) | 48.8 | |

**关键 takeaway**: 即使 2× 重复 (50.2%) 仍匹配 Caption-heavy 的 unique data (50.3%)；4× 重复 (49.8%) 仍高于 base mix (48.8%)。直到 8× 才显著 degrade。

这意味着 **good mixture 的好处胜过 moderate repetition 的代价**。这也呼应了 LLM 领域关于 pretraining 时包含 instruction-like data 的研究。

reference:
- Hernandez et al. scaling laws from repeated data: https://arxiv.org/abs/2205.10487
- Muennighoff et al. scaling data-constrained LMs: https://arxiv.org/abs/2305.16264

## 6. Control Experiments

### 6.1 Pretraining → Post-SFT 的 transfer

担心：SFT 用 instruction-tuning data，会不会 overwrite pretraining 阶段 Instruction-heavy mixture 的优势？

实验：54 个 SFT runs (27 pretrained checkpoints × 2 SFT datasets: LLaVA-665K, Mammoth-VL-12M)，SFT tokens = 0.29 × pretraining tokens。

**Result**: Pearson r = 0.99, Spearman ρ = 0.99。Pretraining 性能近完美预测 post-SFT 性能，ranking 完全保持。

直觉：pretraining 阶段建立的是 "visual-language grounding + reasoning primitives"，这些 foundations 在 SFT 阶段不会被 overwrite，只会被 sharpened。

### 6.2 LM Backbone 鲁棒性

把 LM 从 Qwen2.5-Base 换成 Qwen2.5-Instruct (2B)，重复 full sweep：

**Result**: Pearson r = 0.97，mixture ranking 几乎一致。即使 LM 已被 "unimodally" instruction-tuned，Instruction-heavy mix 仍然更好。这说明 finding 不是 Qwen2.5-Base 的 artifact。

## 7. DCVLM-BASELINE 数据集

基于上述发现，构建 DCVLM-BASELINE：

**Composition** (Instruction-heavy mix):
- 10% Image-caption pairs
- 5% Multimodal documents
- 15% Text-only data
- 70% Multimodal instruction-tuning data

**关键设计**:
- Forgo filtering entirely (因为没用)
- Within each data type, 用 length-proportional sampling 从 constituent sources 抽样

### 7.1 Main Results

**33-task Core set** 各 scale 对比：

| Scale | Method | Gen | Know | OCR | Vision | MTL | Text | Core Avg | Δ vs FV |
|-------|--------|-----|------|-----|--------|-----|------|----------|---------|
| small (1B, 6.25B) | LLaVA-OV-1.5 | 22.4 | 34.8 | 8.2 | 27.8 | 13.5 | 6.9 | 17.6 | - |
| | Nemotron-VL-2 | 20.0 | 39.7 | 7.9 | 33.5 | 16.1 | 20.7 | 22.1 | - |
| | FineVision | 40.1 | 45.6 | 35.0 | 41.0 | 28.2 | 28.9 | 36.2 | - |
| | **DCVLM-Baseline** | 40.5 | 43.6 | 33.0 | 39.1 | 25.4 | 34.7 | **36.5** | **+0.3** |
| medium (2B, 25B) | LLaVA-OV-1.5 | 33.3 | 43.0 | 21.0 | 30.4 | 21.5 | 16.0 | 26.5 | - |
| | Nemotron-VL-2 | 48.6 | 54.6 | 19.9 | 41.1 | 36.7 | 28.6 | 37.0 | - |
| | FineVision | 55.3 | 62.6 | 51.9 | 45.8 | 40.6 | 46.3 | 50.6 | - |
| | **DCVLM-Baseline** | 62.3 | 60.5 | 45.8 | 47.3 | 44.2 | 47.8 | **51.7** | **+1.1** |
| large (4B, 100B) | Nemotron-VL-2 | 31.5 | 53.8 | 23.6 | 38.6 | 27.5 | 36.4 | 34.7 | - |
| | FineVision | 59.0 | 70.7 | 58.9 | 39.1 | 45.1 | 51.2 | 54.2 | - |
| | **DCVLM-Baseline** | 68.4 | 67.6 | 54.1 | 57.2 | 50.9 | 53.8 | **58.9** | **+4.7** |
| x-large (8B, 200B) | FineVision | 63.5 | 72.8 | 57.5 | 49.6 | 48.4 | 55.7 | 58.2 | - |
| | **DCVLM-Baseline** | 73.0 | 73.0 | 53.4 | 63.5 | 56.1 | 61.1 | **63.6** | **+5.4** |

**最 striking 的结果**: 4B model + 100B tokens (DCVLM-Baseline) = 58.9% Core Avg，**超过** 8B model + 200B tokens (FineVision) = 58.2%。这是 **4× compute reduction**。

reference:
- LLaVA-OneVision-1.5: https://arxiv.org/abs/2509.23661
- Nemotron-VL-2: https://arxiv.org/abs/2511.03929
- FineVision: https://arxiv.org/abs/2510.17269

### 7.2 Extended Suite (52 benchmarks)

x-large scale 上，DCVLM-Baseline 60.5% vs FineVision 56.6% (+3.9pp)。

但有些弱点：
- **OCR**: DCVLM-Baseline 落后 FineVision -5.2pp（因为 FineVision 优先 document-rich sources）
- **Safety**: -19.2pp（DCVLM 没专门 safety alignment）
- **Reasoning**: -4.9pp

Vision-Centric (+21.3pp) 和 General (+8.5pp) 上优势巨大。

## 8. 技术细节深入

### 8.1 Sequence Packing（Dual Constraint Bin-Packing）

VLM packing 比 LLM 复杂：每张图被 dynamic tiling 成 variable 数量的 tiles，每个 tile 展开成固定 visual tokens 块。所以 pack 受双 budget 约束：

$$L = 8192 \text{ tokens}, \quad M = 24 \text{ image tiles}$$

**Streaming best-fit bin-packing 算法**：

```
maintain pool of open buffers, sorted fullest-first
for each incoming sample S_i:
    scan pool from fullest to emptiest:
        if tiles(b) + tiles(S_i) ≤ M AND tokens(b) + tokens(S_i) ≤ L:
            assign S_i to buffer b
            re-insert b maintaining sort order
            break
    if no buffer fits:
        create new singleton buffer with S_i
    
    emit buffer when:
        - reaches L or M exactly, OR
        - pool size exceeds cap B=20 (force-emit fullest)
```

**Block-diagonal attention**: 通过 `flash_attn_varlen_func` 实现，每个 packed sample 只 attend 自己 segment 内的 tokens，position ids 在每个 segment boundary 重启。这保证 packed sample 数值上等价于 unpacked training。

Per-token loss 用 segment 有效 token count 重新加权，避免长样本 dominate gradient。

### 8.2 Online Filtering 实现

传统 offline resharding 对每个 filter 配置写一个新 dataset copy，storage 成本爆炸。DCVLM 实现 online filtering：

```
# 每个 sample 的 metadata (sidecar JSON):
"sample_00042_017": {
    "clip_score_clip_vitl14_224_standard": 31.984,
    "clip_score_siglip2_b16_224_standard": 5.421,
    "text_quality_dclmbaseline_fasttext_score": 0.091,
    ...
}

# Training-time predicate:
def filter(sample):
    for f in active_filters:
        if sample.scores[f] < threshold[f][sample.subset]:
            return False  # silently skip
    return True
```

**性能测量**: 即使 90% rejection rate，runtime 只增加 3.5%（50M tokens）或 0.3%（500M tokens）。3 个 concurrent filters @ 80% rejection 各自，overhead 仅 1.10× baseline。

这是因为 VLM 比 CLIP 更 compute-bound，dataloader 有更多 headroom。

### 8.3 Train-Test Decontamination

两层 decontamination：

**(1) Image-based** (multimodal samples):
- SSCD ResNet-50 embedding model
- 288px shorter edge resize (preserve aspect ratio, no center crop)
- FAISS exact L2 index
- Cosine similarity s = 1 - d/2
- Threshold: **s ≥ 0.75** (比 FineVision 的 0.95 严格)

0.75 vs 0.95 的选择：0.75-0.95 区间主要是 "benign transformation" 的 genuine duplicates（重新 typeset 的 diagram、re-crop with text overlay、re-scan）。0.95 threshold 把这些都留下 = 大量 false negatives。

**(2) Text-based** (text-only samples):
- Tulu-3 tokenizer, word-level 5-grams
- 128-permutation MinHash signature
- Two-stage: 
  - Stage 1: discard if Jaccard ≥ 0.55
  - Stage 2: 0.3 ≤ Jaccard < 0.55 的样本做 bi-directional exact substring check

Threshold 0.55 通过 human annotation 选择：TPR 和 FPR 曲线在 0.55-0.6 交叉。

**Overall removal rate**: 只有 0.29% of all training samples 被移除。但 InfoVQA 100%、ScienceQA 66.4%、TabMWP 63.4% 被移除（这些 training 和 eval splits 共享 underlying distribution）。

reference:
- SSCD: https://arxiv.org/abs/2111.04626
- Tulu-3: https://arxiv.org/abs/2411.15124
- MinHash: Broder 1997

### 8.4 Temperature-Scaled Sampling

paper 测试 within-data-type 的 source sampling temperature：

$$p(d) \propto \text{len}(d)^{1/T}$$

- T=1: length-proportional (default)
- T=2: square-root (Molmo 风格)
- T→∞: uniform over datasets (upweight small sources)
- T<1: sharpen toward largest sources

**Results** (Validation set, small scale, Balanced mix):
```
T=0.5:  43.2% (-0.9)
T=0.8:  43.6% (-0.5)
T=1.0:  44.1% (baseline) ← best
T=2.0:  42.4% (-1.7)
T=4.0:  40.0% (-4.1)
```

**Conclusion**: T=1 已近最优。Flattening (upweight small sources) 反而 hurt。这与 Molmo 的发现矛盾，可能因为 broader data-type mixing ratios 与 within-type source sampling 有 non-trivial interaction。

reference: Molmo/PixMo: https://arxiv.org/abs/2409.17146

### 8.5 Synthetic Recaptioning

测试用 Qwen2-VL-7B 替换 DataComp-1B 的 alt-text（占 image-caption 部分的 47%，overall training 的 ~30%）：

| Captions | General | Know | OCR | Vision | MTL | Text | Val Avg |
|----------|---------|------|-----|--------|-----|------|---------|
| Alt-text | 48.0 | 40.0 | 46.8 | 43.4 | 49.5 | 36.6 | 44.1 |
| Synthetic short | 46.6 | 39.2 | 44.9 | 42.5 | 49.8 | 36.5 | 43.2 (-0.9) |
| Synthetic spatial | 48.7 | 39.7 | 45.9 | 42.5 | 50.7 | 36.7 | 44.0 (-0.1) |

**Null result**: synthetic recaptioning 对 VLM pretraining 无显著帮助，与 CLIP training 上的 gains 相反。

**Hypothesis**: instruction-tuning 部分已经提供大量高质量、densely-described image-text pairs，rewriting image-caption 的 marginal value 很低。再次印证 mixture > filtering 的核心结论。

## 9. 重要的 Limitations 和 Future Work

### 9.1 Limitations

1. **Scale ceiling**: 只到 8B model + 200B tokens，远低于 Qwen2-VL (1.4T) 和 Qwen3-VL (2.2T) 的 pretraining budgets。

2. **Model size**: 未测 ≥20B models，是否 Instruction-heavy 仍最优 unknown。

3. **Native multimodal pretraining**: 用的是已 pretrained vision encoder + LM，non-native paradigm 的结论可能不同。

4. **Scaling ladder design**: 当前 4-scale ladder 是 practical defaults，非 theoretical optimal。

5. **Capability coverage**: grounding 和 detection benchmarks 因 contamination 问题被排除。

6. **OCR weakness**: 落后 FineVision -5.2pp（x-large），可能因为 DCVLM 优先 instruction data 而非 document-rich sources。

### 9.2 Future Directions

1. **Instructional multimodal documents**: 把 raw multimodal docs 转化为 instruction-style，combine breadth + dense signal。

2. **Fine-grained mixture optimization**: 用 swarm-based methods (DoReMi, RegMix, OLMix) 替代 manual line search。

3. **Token-level mixing**: 当前是 sample-level mixing，token-level 可能不同。

4. **Quality-aware upsampling**: 用 quality scores 做 upsampling 而非 filtering，避免 exclusionary bias。

## 10. 对 VLM 研究的 broader implications

### 10.1 Data Curation 范式转变

这篇 paper 最重要的 meta-lesson：**当 data ecosystem 进入 "aggregation of already-curated datasets" 阶段，filtering 的边际收益会快速衰减，mixture optimization 成为 dominant lever**。

这对 future VLM data curation 研究有方向性意义：
- 不要盲目 stack 更多 quality filters
- 要研究 cross-data-type mixture 的 scaling behavior
- Scale-aware curation 是必需的，single-scale proxy 会误导

### 10.2 Pretraining-SFT Decoupling

Pearson r=0.99 的 pretraining-post-SFT 相关性意味着：可以用 pretraining-only metrics 做 data curation research，不用每次都跑 SFT。这大幅降低 curation 实验成本。

### 10.3 Instruction-Heavy Mix 的解释

为什么 instruction-heavy 在 large scale 更好？几个可能 explanation：

1. **Information density**: instruction data 通常是 QA pairs，每 token 携带更多 "task-relevant" signal。当 model capacity 足够大，能 extract 这种 dense signal 时，instruction data 的 effective information per token 更高。

2. **Format alignment**: instruction data 的 multi-turn、multi-image 格式更接近 downstream evaluation，减少 distribution shift。

3. **Task diversity**: 8 个 capability categories 的 instruction data 提供 wide task coverage，vs image-caption 主要是 "describe" 单一 task。

4. **Compositional reasoning**: instruction data 中的 reasoning traces（math、science）可能帮助 model 学习 compositional skills，这些 skills 在 large scale 才 emerge。

但这些都是 hypotheses，paper 没有直接验证 mechanism。Future work 可以做 mechanistic interpretability 来 disentangle。

### 10.4 与 Chinchilla Scaling Laws 的关系

Chinchilla 建议 ~20 tokens/parameter。但 DCVLM 的 Instruction-heavy mix 在固定 token budget 下更好，暗示 **data mixture quality 改变了 effective tokens-per-parameter ratio**。

如果把 Instruction-heavy mix 视作 "higher quality data"，那么同样的 token budget 相当于 "more effective tokens"。这可能意味着 scaling laws 需要一个 mixture-quality 维度。

reference:
- Chinchilla: https://arxiv.org/abs/2203.15556
- DataComp-LM (DCLM): https://arxiv.org/abs/2406.11794

## 11. 工程 intuition：如何用 DCVLM 做 curation 实验

如果我想用 DCVLM 测试自己的 curation strategy：

1. **选 scale**: 根据 compute 选 small/medium/large/x-large。
2. **构造 dataset**: 从 pool 中按我的 strategy filter/mix/format/sampling。
3. **训练**: 用 fixed InternVL3-style architecture 和 training recipe。
4. **评估**: 至少跑 33-task Core set，理想跑 52-task Extended。
5. **跨 scale 验证**: 至少在 2 个 scale 验证 mixture choice，避免 small-scale 误导。

关键：**不要只看 small scale**。Instruction-heavy 在 small scale 是 worst，但 large scale 是 best。

## 12. 总结

DCVLM 是 VLM data curation 领域第一个系统的、scale-aware 的 benchmark。三大贡献：

1. **Standardized pool**: 160 datasets, 6T tokens, 4 data types。
2. **Scaling ladder**: 4 scales from 1B/6.25B to 8B/200B。
3. **Evaluation protocol**: 52 benchmarks, 3 nested tiers, stability/monotonicity filtered。

核心发现：
- **Filtering 在 already-curated pool 上 diminishing returns**
- **Mixture (Instruction-heavy) 是 dominant lever，且 scale-dependent**
- **Pretraining 性能 predict post-SFT (r=0.99)**
- **DCVLM-BASELINE 在 x-large scale 比 FineVision +5.4pp，4B+100B 超过 8B+200B**

这 work 对 future VLM 研究者最大的 takeaway：**stop blindly stacking quality filters, start doing scale-aware mixture optimization**。

代码: https://github.com/mlfoundations/dcvlm
网站: https://www.datacomp.ai/dcvlm
