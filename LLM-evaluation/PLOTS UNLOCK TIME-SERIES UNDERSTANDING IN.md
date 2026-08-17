---
source_pdf: PLOTS UNLOCK TIME-SERIES UNDERSTANDING IN.pdf
paper_sha256: 583df6dca51dfd93f492569a6c45eaccdc13cb57c2967724463fa6057d6c23dd
processed_at: '2026-08-06T04:53:08-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 paper

## 一句话版本

**把 time-series 画成图给 model 看, 比把一堆数字喂给它读, 效果更好还更便宜。**

就这么简单。其余都是 validation。

---

## 为什么 text 喂数字不行

你给 GPT-4o 一段 `[0.31, 0.42, 0.58, 0.71, ...]` 这种浮点数序列, 它的 tokenizer 会把每个数字切成好几块。`0.314159` 可能变成 `["0", ".", "314", "159"]` 四个 token。

这有几个问题:

1. **信息密度极低**——一个有意义的数字占 4 个 token, 但每个 token 本身毫无意义
2. **位置编码无物理含义**——数字在第 3 个 token 位置 vs 第 5 个 token 位置, 跟这个数字在 time-series 里的"时间位置"毫无关系
3. **数字的形态被破坏**——`314159` 和 `3141592` 在 tokenizer 眼里是完全不同的 subword, 但数学上只差一点点 precision
4. **序列太长**——15 秒 IMU 数据 ~46k tokens, 10-shot 就爆 context window

LLMTime (Gruver et al., NeurIPS 2024: https://arxiv.org/abs/2310.07800) 想了各种办法补救: 降低 precision (2 位小数就够)、用特定 separator、做 rescaling。有用, 但天花板很低。

Supp Table S7 显示: noise=0 时, 最好的 text 配置 accuracy 0.77, 而 plot 同条件 0.95。noise 一加, 差距立刻拉开。

说白了: **LLM 的 tokenizer 根本不是为数字设计的**, 是为 natural language 设计的。硬塞数字进去就是用螺丝刀拧钉子。

---

## 为什么 vision encoder 行

你画个 scatter plot 给 GPT-4o 看, 它用 vision encoder (ViT 或类似架构) 处理这张图。Vision encoder 的 inductive bias 是:
- **Spatial locality**: 相邻 pixel 互相相关, 跟 ViT 的 patch attention 对齐
- **Scale invariance**: 同一个 trend 不管画多大都认得出来
- **Contrast-based features**: 形状靠亮度对比, 不靠绝对数值

更关键的是: GPT-4o 和 Gemini 的 vision encoder 在 pretraining 时见过海量 chart / scientific figure / educational illustration。LAION 这种 dataset 里有几百万张 "an exponentially growing curve" 配 exponential plot 的图文对。

所以 vision encoder **已经隐式学会了 plot-to-trend 的映射**, 你只是 activate 它。

### 关键公式: token cost 对比

Text 表征:
$$T_{\text{text}} \approx N \cdot k \cdot \bar{t}_{\text{num}}$$

- $N$: 数据点个数 (比如 IMU 15s @ 128Hz → 1920)
- $k$: 维度 (IMU 是 6)
- $\bar{t}_{\text{num}}$: 每个浮点数平均 token 数 (~4)

总 token ≈ $1920 \times 6 \times 4 \approx 46{,}080$ per segment, 10-shot 就是 460k, **爆 GPT-4o 的 128k context window**。

Plot 表征:
$$T_{\text{plot}} = \begin{cases} 258 & \text{if image} \leq 384 \times 384 \\ 1290 & \text{otherwise} \end{cases}$$

跟 $N$ **完全无关**。无论你 plot 50 个点还是 50 万个点, 都是 258 或 1290 tokens。

具体数字 (paper Section 4.4): 5-shot activity recognition on GPT-4o
- Text: 128k tokens → **$0.32 / query**
- Plot: 50 images × 258 ≈ 12.9k tokens → **$0.032 / query**

**10× 成本差**, 而且 text 随 $N$ 线性涨, plot 是常数。

参考 OpenAI pricing: https://openai.com/api/pricing/
参考 Gemini image cost: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding

---

## 实验设计: 从简单到真实的 reasoning ladder

Paper 不是简单说 "plot 好使", 它设计了一条 reasoning 难度递增的梯子, 看 plot 优势在哪一级消失。

### Synthetic tasks (5 个)

| 任务 | Reasoning 难度 | Plot 优势 |
|---|---|---|
| Functional form id. | 识别 1 个 trend | **+122%** (GPT-4o) |
| Correlation of 2 lines | 识别 2 个 trend + 比较 | **显著** (所有 model) |
| 2D cluster counting | 识别 N 个 pattern + 计数 | **显著** (MAE 减半) |
| Derivative id. | 识别 trend + 推导 + match | **持平** (0.00) |
| Quadratic derivative id. | trend + magnitude 推理 | **GPT-4o 反超 text** |

为什么 derivative 任务 plot 优势消失? 因为这是**符号推理**任务——"二次函数的导数是线性", 这种推理在 text / 符号空间里更直接。Vision encoder 帮你识别 shape, 但不帮你做 calculus。

这个发现本身很有价值: **plot 是 trend 识别的 prior, 不是 reasoning 的 prior**。

### Real-world tasks (3 个)

**Fall detection (IMUFD, Aziz et al. 2017: https://pubmed.ncbi.nlm.nih.gov/27334988/)**
- 输入: 6D IMU, 15s, 128Hz
- 3-class: Fall / Near Fall / ADL
- 关键: Near Fall 是 hard negative, 跟真 fall 都有大 spike, 区别在于 pattern 细节
- 结果: GPT-4o 10-shot plot: sensitivity 0.92, specificity 0.81
- 对比: task-specific SVM: 0.96, 0.96 (略好, 但 GPT-4o 是 zero-training generalist)

**Activity recognition (HHAR, Stisen et al. 2015: https://dl.acm.org/doi/10.1145/2809695.2809718)**
- 5-class: sit / stand / walk / stairs / bike
- 关键: HHAR 故意用 heterogeneous devices (4 phones + 2 watches), 测试泛化
- 结果: Gemini Pro 5-shot plot 接近 SOTA DL model
- **GPT-4o(-mini) 在 10-shot text 时直接爆 context window, 跑不了**

**Readiness assessment (Google internal data)**
- 输入: 30 天 TRIMP 数据 (tabular)
- 任务: ACWR > 1 (overtraining) vs < 1 (undertraining)
- ACWR 公式:
$$\text{ACWR}_t = \frac{\sum_{i=t-6}^{t} \text{TRIMP}_i}{\frac{1}{28}\sum_{j=t-27}^{t} \sum_{i=j-6}^{j} \text{TRIMP}_i}$$
- 分子是 7 天 acute load, 分母是 28 天 chronic load 的 rolling average
- 结果: **plot vs text 持平**

为什么这里持平? 因为只有 30 个数字, text 完全 OK, plot 优势 (压缩长序列) 用不上。这是 paper 的 honest negative result, 说明 plot 方法有 sweet spot: **数据量大到 text 不 ergonomic, 但 trend 视觉可读**。

参考 Cosentino et al. 2024 Personal Health LLM: https://arxiv.org/abs/2406.06474

---

## Ablation 的关键发现: plot 不挑细节

Supp S.1.3 跑了一大堆 plot 风格 ablation, 结论是**几乎什么都不影响**:

- **DPI** (25 vs 400): accuracy 在 0.66-0.74 波动, 25 dpi 就够
- **Figure size** ((3.5,3.5) 到 (12,12)): 没差别
- **Plot style** (default / ggplot / seaborn): 没差别
- **Color palette** (黑白 / 彩色 / 反色): 没差别
- **Marker type** (circle / square / triangle / x): 没差别
- **Marker size** (small / medium / large): 没差别
- **Axis labels**: 有 vs 无, 微弱差异 (0.89 vs 0.85)
- **Temperature** (0.0 vs 1.0): 完全没影响

直觉解释: vision encoder 学的是 **几何形态**, 不是 pixel-perfect rendering。Trend 在 25 dpi 和 400 dpi 下, 形状是一样的。

这也呼应了 Karpathy 你一直强调的: **inductive bias 比 data quality 重要**。Vision encoder 的 spatial locality + scale invariance 让它对 plot 的渲染细节 robust。

---

## 为什么 combined modality 不超过 plot only

Supp Fig S15 测了 plot + text 一起给 model, 结果**没显著超过 plot only**。

直觉: plot 信号已经 saturate model 的理解能力, text 不仅没帮助, 反而引入 noise (冗余信息 + 可能的 tokenizer 破碎数字误导 reasoning)。

这是一个 negative result 但很有启发性: **多模态不是越多越好, 而是"最适配的模态给够就行"**。

---

## 跟相关工作的根本差异

| 方向 | 代表工作 | 跟本文差异 |
|---|---|---|
| 训练专用 TS model | Chronos / MOMENT / MOIRAI | 需要大量 paired data + compute, 不 generalizable |
| Text-only TS prompting | LLMTime | 用 tokenizer trick 但天花板低 |
| Plot → table 翻译 | DePlot (Liu et al. 2023: https://aclanthology.org/2023.findings-acl.10381/) | 反向操作, 适合短数据 |
| Contrastive binding | IMU2CLIP / ImageBind | 需要 paired waveform + video, 训练成本高 |
| 用 CLIP embed plot | Wimmer & Rekabsaz 2023 (https://arxiv.org/abs/2301.10166) | 需要 downstream classifier, 不能直接 reasoning |

本文的核心区别: **zero additional training, zero specialized encoder, zero paired data**。你只要会调 GPT-4o / Gemini API + 用 matplotlib 画图, 明天就能上 production。

---

## Limitations: paper 自己承认的边界

1. **不 beat task-specific model**: GPT-4o plot fall detection (0.92, 0.81) < SVM (0.96, 0.96)。但 GPT-4o 同时能做 activity recognition + 解释 + 接其他 prompt, 这种泛化性 SVM 给不了

2. **多步 magnitude reasoning 失效**: derivative 任务 plot 优势消失

3. **Tabular 短数据无优势**: readiness 任务 plot = text

4. **没研究 plot 类型选择**: paper 用 human-written matplotlib 代码, 没测 radar chart / heatmap / stack plot 是否更好

5. **没 mechanistic 解释**: 为什么 cubic / periodic 是 hard case? 为什么 GPT-4o 在 quadratic derivative 上 text 反超 plot? 留给 follow-up

---

## Karpathy 视角的 takeaway

你讲 "Let's build GPT" 时反复说: **tokens 是思考的原子单位**。这篇 paper 的反直觉在于: **对 time-series, plot 是更高效的原子单位**, 即便它丢了一些精度。

这跟你 "inductive bias matters" 的论点完全一致——vision encoder 的 spatial locality + scale invariance + contrast-based features 恰好对 trend 这种 2D 几何结构 well-matched, 而 text tokenizer 的 byte-pair encoding 对连续数字是 mismatched。

### 我会做的 follow-up

1. **Probe ViT attention**: 在 functional form id. 任务上, 提取 GPT-4o vision encoder 的 attention pattern, 看 trend information 在哪些 patch / 哪个 layer 被编码。这是 mechanistic interpretability 路子

2. **Plot-style search via tool-use**: 给 model 一个 `plot(x, y, style=...)` tool, 让它自己 search 最优 plot 风格, 评估 model 选择的 plot vs human 选择的质量 gap

3. **Out-of-distribution plot 测试**: 测 radar chart / polar coordinates / heatmap 这些 OOD 视觉 encoding, 看 vision encoder 的 trend 能力是否依赖 in-distribution 训练数据

4. **Forecasting 实验**: paper 刻意避开 forecasting, 因为那是 vision encoder 弱项。但如果加一个 "plot past → text extrapolate" 的 mixed pipeline, 也许能 bridge

5. **更深的 token efficiency 研究**: 258 vision tokens 能承载 10k+ 数据点的 trend, 这个 compression ratio 是 ~40×。Vision token 的 information bottleneck 在哪? 是 patch resolution? 是 attention head 数? 这跟 ViT 的 effective dimension 有关

---

## 最后一句

这篇 paper 本质上**用 zero-training trick 把人类几百年发展出的"画图理解数据"的直觉, 验证为 multimodal model 的 free lunch**。它不是 architecture innovation, 不是 training innovation, 是 **prompting / representation innovation**。但它的 empirical validation 严谨, token economics 分析清晰, 而且诚实承认边界。

如果你要把这个写进 "Neural Networks: Zero to Hero" 作为 application case, 我会放在 attention mechanism 之后讲: 看, attention 让 model 学会"看哪里", 而 plot 让人类决定"让它看什么"。两者结合, 就是 multimodal reasoning 的 sweet spot。

---

# 《Plots Unlock Time-Series Understanding in Multimodal Models》深度解析

## 1. 核心论点与 Motivation

这篇 paper 的核心 hypothesis 极其简洁: 让 multimodal foundation model "看见" time-series 数据 (通过 plot) 比 "读" 它 (作为 text token sequence) 更有效, 而且更便宜。这其实把人类的直觉形式化了——我们遇到一长串浮点数时, 第一反应也是先画图再看 (Card et al., 1999; Yalçın et al., 2016 的认知科学工作)。

为什么这个 hypothesis 有理论基础:
- LLM 的 tokenizer 本质上是为 natural language 设计的 BPE / SentencePiece, 把浮点数切得七零八落 (Spathis & Kawsar, 2024, JAMIA: https://doi.org/10.1093/jamia/ocae090)
- 数字 `0.31415926` 在 GPT-4o tokenizer 下可能被切成 `0`、`.`、`314`、`159`、`26` 这种碎块, 信息密度极低, 而且每个数字字符的位置编码毫无物理意义
- 相反, vision encoder (ViT / convolutional backbone) 处理的是 224×224 或更大的 patch grid, 一个 plot 图像只需几百个 vision tokens 就能承载上千个数据点的 trend 信息, 因为视觉表征天然具有 spatial invariance 和 scale invariance

这其实指向一个深刻的 insight: **time-series 的"信息"很大程度上存在于其几何形状, 而非具体数值**。函数形态、趋势、cluster 结构、correlation 方向——这些都是 2D 几何特征, vision encoder 经过 ImageNet / LAION 这种大规模视觉预训练后, 对这类几何 pattern 的识别能力已经内化。

参考: Spathis & Kawsar 2024 论证了 tokenizing time-series for LLMs 是 "first step is the hardest": https://academic.oup.com/jamia/article/31/9/2151/7711113

---

## 2. 方法论详解

### 2.1 Plot-based Prompting 的 Token 经济学

这是 paper 中最值得 build intuition 的部分。给定一段 length $N$ 的 time-series, 用 text 表示需要的 token 数:

$$T_{\text{text}} \approx N \cdot k \cdot \bar{t}_{\text{num}}$$

其中:
- $N$ 是数据点个数
- $k$ 是维度 (IMU 数据 $k=6$, 单变量 $k=1$)
- $\bar{t}_{\text{num}}$ 是每个数字平均消耗的 token 数 (典型 3-5 tokens per float, 取决于 precision)

举例: 15 秒 @ 128 Hz 的 IMU segment, $N = 1920$, $k = 6$, 单变量 token ~4, 总共约 **46,080 tokens per segment**。10-shot 就要 ~460,800 tokens, 直接爆掉 GPT-4o 的 128k context window。

而 plot 表示:
$$T_{\text{plot}} = f(\text{image resolution})$$

Gemini 的计费规则: 图像两边都 < 384×384 px → 258 tokens; 否则切 4 个 crop + 1 个 overview → 1290 tokens。这跟 $N$ **完全无关**。

Paper 里给出的具体数字 (5-shot activity recognition on GPT-4o):
- Text: ~128,000 input tokens → cost **$0.32 per query** (按 GPT-4o pricing: $2.50/M input tokens 估算)
- Plot: 50 images (5 shots × 2 sensors × 5 examples) × ~258 tokens ≈ 12,900 tokens → cost **$0.032 per query**

即 10× 成本差异。而随 $N$ 增大, 这个 gap 还会持续扩大, 因为 text 是 $O(N)$, plot 是 $O(1)$ (相对 $N$)。

参考 OpenAI pricing: https://openai.com/api/pricing/
参考 Gemini API image cost: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding

### 2.2 Text 表征的 Ablation (致敬 LLMTime)

Paper 在 Section 3.3 + Supplementary S.1.3 跑了 LLMTime (Gruver et al., NeurIPS 2024: https://arxiv.org/abs/2310.07800) 风格的 ablation:
- **Precision**: 2, 4, 8, 16 decimal places
- **Separator**: comma+space vs space only
- **Rescaling**: 用 LLMTime 提议的 $\alpha$-scaled quantization

LLMTime 的核心 idea 是把时间序列重写为:
$$\tilde{x}_t = \text{round}\left(\frac{x_t - \min(x)}{\text{scale}}\right)$$
其中 $\alpha \in \{0.5, 0.7, 0.9, 0.99\}$ 控制动态范围, $\beta \in \{0, 0.15, 0.3, 0.5\}$ 控制偏移。

发现: 低 precision (2-4 digits) 反而更好——这跟 LLMTime 的结论一致, 因为高 precision 会让 tokenizer 产生更多 subword 碎片, 破坏 "数字形态" 的连续性。但即便如此, **text 的 best case 仍然远不如 plot**。

Supplementary Table S7 显示: noise=0.0 时, precision=2 的 text baseline 达到 0.77 accuracy, 而 plot baseline 同条件下达到 **0.95**。差距随 noise 增加而扩大。

### 2.3 Statistical Framework

Paper 用了两套统计检验:

**Synthetic tasks** (有可控随机性): Wilcoxon signed-rank test
- 这是非参数配对检验, 检验 $\text{median}(\text{plot}_i - \text{text}_i) = 0$ 的零假设
- 之所以不用 paired t-test, 是因为 performance 分布不近似正态 (尤其 F1 score 边界在 [0,1])
- Bonferroni correction: $\alpha_{\text{corrected}} = \alpha / m$, 其中 $m$ 是同 block 内的 comparison 数, 控制 family-wise error rate (Bland & Altman, 1995: https://doi.org/10.1136/bmj.310.6973.170)

**Real-world tasks** (没法重复 sampling 同一 problem): bootstrap with $B=1000$ replicates
- 对每个 (model, shot, modality) 组合, 从 test set 重采样 1000 次, 得到 $F_1$ 的分布
- 然后从两个分布各采样一对, 得到 difference 的分布
- 注意: 这样得到的 IQR 不能直接跟 synthetic 比, 因为 bootstrap 引入的 variance 比独立 replicate 小

公式: 给定 $n$ 个样本, 第 $b$ 次 bootstrap:
$$\hat{F}_{1}^{(b)} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[\text{correct}_i] \cdot w_i^{(b)}$$
其中 $w_i^{(b)} \sim \text{Multinomial}(n, [1/n, \dots, 1/n])$。

---

## 3. 任务设计:从简到繁的"reasoning ladder"

Table 1 是 paper 的核心架构图。让我把每个任务的 reasoning demands 拆开:

### 3.1 Functional Form Identification (最简单)

- **任务**: 给一段 $(x, y)$, 分类为 `linear / quadratic / cubic / exponential / periodic`
- **生成**: $y = f(x + \epsilon)$, 其中 $\epsilon \sim \mathcal{N}(0, \sigma^2)$, $\sigma \in \{0, 0.5, 1, 2, 5\}$
- **Reasoning**: 单 trend 识别, 全局形状匹配
- **关键发现** (Figure 1a): GPT-4o plot accuracy ~0.84 vs text ~0.38, 即 **122% improvement** (Table 2 中 GPT-4o: 0.46 IQR [0.31, 0.52])
- **细节** (Supp Fig S9c): exponential 和 linear 在 plot 下都接近 1.0 accuracy, cubic 和 periodic 是 hard cases (即使 plot 也只到 ~0.4-0.5), 因为这两个的 noise-resilience 较差

为什么 cubic 难? Cubic 在 $x \in [-10, 10]$ 上, $y = x^3$ 的动态范围是 $[-1000, 1000]$, 加上 noise=2 时, 局部都被 noise 主导, 全局趋势被 axis scaling 压扁——这对 vision encoder 来说也很 hard, 因为它需要识别"远端被压扁的 S 曲线"。

### 3.2 Correlation of Two Lines

- **任务**: 给 $(x, y_1), (x, y_2)$, 判断 positive / negative correlation
- **生成**: $y_1 = m_1 (x + \epsilon_1)$, $y_2 = m_2 (x + \epsilon_2)$, 用 Pearson coefficient 算 ground truth
- **Pearson 公式**: 
$$\rho_{y_1, y_2} = \frac{\sum_i (y_{1,i} - \bar{y}_1)(y_{2,i} - \bar{y}_2)}{\sqrt{\sum_i (y_{1,i}-\bar{y}_1)^2} \cdot \sqrt{\sum_i (y_{2,i}-\bar{y}_2)^2}}$$
- Sign($\rho$) 决定 label
- **Reasoning**: 两个 trend + 比较
- **发现**: 所有 4 个模型都显著优于 text (Table 2 第一行全部带星号)

### 3.3 2D Cluster Counting

- **任务**: 给散点图, 数 cluster 个数 (1-9)
- **生成**: sklearn `make_blobs`, 强制 cluster center 最小距离 0.3, $\sigma_{\text{cluster}} \in \{0.025, 0.05, 0.075\}$
- **Metric**: MAE (mean absolute error), 越低越好
- **Reasoning**: 同时识别并跟踪 N 个 trend, 类似 object counting in vision
- **发现** (Table 2): MAE 下降明显, GPT-4o-mini: 1.02 vs text baseline; Gemini Pro: 1.82 plot vs 显著更高 text MAE

这个任务特别有意思, 因为它直接对应 vision 模型的强项——CLIP / ViT 在 ImageNet 上训练了大量的"counting"概念 (比如 "three cats")。

### 3.4 Derivative Identification (multi-step reasoning)

- **任务**: 给一个 function plot + 4 个 candidate derivative plots, 选正确的
- **Reasoning chain**:
  1. 识别原函数的 functional form
  2. 推导这个 form 的 derivative 形态
  3. 在 candidates 里 match
- **关键**: 不需要 magnitude, 因为 candidates 是不同 functional class (linear, quadratic, cubic, exp, periodic 各自 derivative 形态不同)
- **发现** (Figure 1d): **plot 不一定显著好于 text**! GPT-4o plot vs text 几乎打平 (Table 2: 0.00 [-0.18, 0.12]), Gemini Pro 也是 -0.02 [-0.16, 0.08]

这是 paper 的一个重要 honesty: 当任务需要把"形状"映射到"分析推导结果"时, vision encoder 的优势就消失了, 因为推导过程是符号性的, LLM 的 text 处理反而更直接。

### 3.5 Quadratic Derivative Identification (hardest)

- **任务**: 固定原函数为 $y = A \cdot x^2$ (二次), candidates 都是 linear, 区别只在 slope magnitude
- **关键变化**: 现在 candidates 形态都一样 (linear), 必须**同时**识别 slope sign 和 slope magnitude
- **生成**: $A \in \{-10, -5, -1, 1, 5, 10\}$, candidates 的 $A \in \{-20, -15, -10, -5, -1, 1, 5, 10, 15, 20\}$
- **Derivative**: $y' = 2A \cdot x$
- **Few-shot 实验** (Figure 2): 加 1, 2, 3 个 reasoning trace 后, Gemini family plot 表现提升明显 (3-shot Gemini Pro: 0.28 [0.19, 0.43]), 而 GPT family 反而下降 (可能 in-context reasoning trace 让 GPT-4o "更想用 text"了)
- **发现**: GPT-4o 在 zero-shot 上是 outlier (text 比 plot 好), 其他 3 个模型 plot 显著更好

为什么 GPT-4o 是 outlier? Paper 没给确定解释, 但我推测: GPT-4o 的 text encoder 在 pretraining 时见过大量数学/物理 textbook 里的二次函数符号形式, 符号推理能力强。而 Gemini 训练数据可能更偏多模态, vision 是 first-class citizen。

---

## 4. Real-world Tasks:验证泛化性

### 4.1 Fall Detection from IMU (IMUFD dataset)

- **数据集**: Aziz et al. 2017 (https://pubmed.ncbi.nlm.nih.gov/27334988/), waist-mounted IMU, 128 Hz, 7 body locations
- **任务**: 3-class 分类 (Fall / Near Fall / Active Daily Living)
- **难点**: Near Fall 是 hard negative——参与者假装绊倒但恢复, 产生类似 fall 的大 magnitude spike
- **输入**: 15s × 128Hz × 6 axes = 11520 numbers, 1D avg pool stride 10 downsample 后 ~1150 numbers
- **Few-shot**: 1, 3, 5, 10 shot, 多 body location 投票
- **SOTA baseline**: Aziz et al. 的 task-specific SVM: sensitivity 0.96, specificity 0.96
- **结果** (Table S2):
  - GPT-4o plot, 10-shot: sensitivity 0.92, specificity 0.81
  - Gemini Pro 1.5 plot, 10-shot: sensitivity 0.84, specificity 0.95
  - GPT-4o text, 10-shot: sensitivity 0.70, specificity 0.49 (大幅落后)
- **Plot 提升**: GPT-4o 在 10-shot 上 plot 比 text 提升 153% (Table S1)

这个结果让 intuition 收敛: fall detection 的关键是看"有没有一个尖锐的 multi-axis 同步 spike", 这是几何 / 视觉特征, text 表征下要算 cross-axis 的 timing + magnitude alignment 极其困难。

### 4.2 Activity Recognition (HHAR dataset)

- **数据集**: Stisen et al. 2015 (https://dl.acm.org/doi/10.1145/2809695.2809718), heterogeneous 设备 (4 phones + 2 watches), 故意制造跨设备 variation
- **任务**: 5-class (sit / stand / stairs / walk / bike)
- **输入**: 同 6D IMU, 但分成两个 plot (accelerometer 和 gyroscope 分开, 见 Supp Fig S8 的 ablation)
- **Few-shot**: 1, 3, 5, 10; GPT-4o(-mini) 在 10-shot 时 text 直接爆 context, 无法跑
- **SOTA**: DeepTransHAR (Kumar & Selvam 2022), 一个专门 DL 模型
- **结果** (Figure 4): Gemini Pro 5-shot plot 的 macro-F1 接近 SOTA DL 模型, 显著优于 text
- **Plot 分开 vs 合并** (Supp Fig S8): 分开 plot 在 1-shot 上略有提升, 5-shot 时差距缩小——说明 model 在 few-shot 多时能学会"忽略不相关 axis 的 noise"

### 4.3 Readiness Assessment (Google internal data)

- **任务**: 给 30 天的 TRIMP (Training Impulse) 数据, 判断 ACWR > 1 (overtraining) 还是 < 1 (undertraining)
- **ACWR 公式**:
$$\text{ACWR}_t = \frac{\sum_{i=t-6}^{t} \text{TRIMP}_i}{\frac{1}{28}\sum_{j=t-27}^{t} \sum_{i=j-6}^{j} \text{TRIMP}_i}$$
- 分子: 7 天 acute load; 分母: 28 天 chronic load 的平均
- **数据**: 350 case studies, 仅 Gemini 可访问 (privacy sandbox)
- **结果** (Figure 5): **基本打平**, Gemini Pro plot 略好 (+0.07 F1), Flash plot 略差 (-0.08)

为什么这里 plot 没优势? Paper 解释: 这是 tabular 数据, 30 个数字, text 表征完全可控, 没有"长序列 visualization 优势" 的用武之地。Plot 在这里只是 bar chart, 信息密度跟 table 相当。**这是一个诚实的边界 case**, 说明 plot 方法的 sweet spot 是: 数据量大到 text 不再 ergonomic, 但 trend 仍然视觉可读。

参考 Cosentino et al. 2024 (Personal Health LLM): https://arxiv.org/abs/2406.06474

---

## 5. 相关工作坐标系

### 5.1 Time-series Foundation Models (训练专用)

- **Chronos** (Ansari et al. 2024: https://arxiv.org/abs/2403.07815): 把 time-series 量化为 token, 用 T5 / GPT-2 架构训 forecasting
- **MOIRAI** (Woo et al. 2024: https://arxiv.org/abs/2402.02592): unified training for universal forecasting transformer
- **MOMENT** (Goswami et al. 2024: https://arxiv.org/abs/2402.03885): open time-series foundation model family
- **LLMTime** (Gruver et al. 2024: https://arxiv.org/abs/2310.07800): 不训练, 只用 text tokenization trick 让 LLM 做 forecasting

这些工作的核心都假设: **time-series 必须变成 token 序列**。本文反其道而行: **让 vision encoder 吃 token 化的图像**。

### 5.2 Vision-Language 对接时间序列

- **IMU2CLIP** (Moon et al. 2023: https://aclanthology.org/2023.findings-emnlp.885/): contrastive learning 绑定 IMU waveform + video + text
- **ImageBind** (Girdhar et al. 2023: https://arxiv.org/abs/2304.08073): 绑 6 种 modality 到同一 embedding space
- **CLIP for financial plots** (Wimmer & Rekabsaz 2023: https://arxiv.org/abs/2301.10166): 用 CLIP embed 金融时序 plot 做下游分类
- **DePlot** (Liu et al. 2023: https://aclanthology.org/2023.findings-acl.10381/): 反向操作——把 plot 翻译成 table, 再做表格 reasoning

本文跟它们的根本差异: **zero training, 直接 leverage 现有 multimodal model 的 vision encoder**。这意味着这个方法可以立刻应用到任何 GPT-4o / Gemini 用户的应用里, 没有任何额外成本。

### 5.3 评估方法论

- **CharXiv** (Wang et al. 2024: https://arxiv.org/abs/2406.18521): chart understanding benchmark
- **SPIQA** (Pramanick et al. 2024: https://arxiv.org/abs/2407.09413): scientific paper 图表问答
- **TimeSeriesExam** (Cai et al. 2024: https://arxiv.org/abs/2410.14752): 也发现 plot > text, 但仅 synthetic
- **VLMs are Blind** (Rahmanzadehgervi et al. 2024: https://arxiv.org/abs/2407.06581): VLM 在细粒度视觉推理上很糟
- **VLMs Aren't Blind** (Corin 2024: https://www.danielcorin.com/posts/2024/vlms-arent-blind/): prompt engineering 可补救

本文刻意避免用 LLM 生成 benchmark, 因为 Panickssery et al. 2024 (https://arxiv.org/abs/2404.13076) 显示 LLM 评估者会偏爱自己生成的内容——这是一个被低估的 bias 源。

---

## 6. Ablation 亮点 (Supplementary S.1.3)

我用 Table S12-S35 整理出来的关键 ablation insights:

### 6.1 Plot 类型的鲁棒性

- **DPI**: 25 → 400, accuracy 在 0.66-0.74 之间波动, 无明显趋势 → **vision encoder 对 resolution 不敏感**, 25 dpi 就够 (Supp Table S12)
- **Figure size**: (3.5, 3.5) 到 (12, 12), 准确率基本一致 (Supp Table S15-S17) → aspect ratio 不重要
- **Plot style**: default / classic / ggplot / seaborn-darkgrid / whitegrid, 都差不多 (Supp Table S18-S20)
- **Color palette**: 黑白 vs 彩色 vs 反色, 准确率几乎相同 (Supp Table S21-S23) → **颜色不是关键**
- **Marker**: circle / square / triangle / x-mark / plus, 影响微乎其微 (Supp Table S24-S26)
- **Marker size**: small / medium / large, 没影响 (Supp Table S27-S29)
- **Plot components**: all / minimal / none (no axis labels), minimal 略差 (0.83 vs 0.89 at noise=0), none 居然也接近 all → **axis 信息有一点帮助但不是关键** (Supp Table S30-S32)
- **Temperature**: 0.0 → 1.0, accuracy 完全稳定, 在 0.66-0.74 范围内随机波动 (Supp Table S33-S35)

**关键 intuition**: vision encoder 不靠"漂亮"取胜, 靠"几何形态"。这跟 ViT 的 inductive bias 一致——它学的是 patch-level pattern, 而不是 pixel-perfect rendering。

### 6.2 Text Ablation 的失败

- **Rescaling** (Supp Table S9-S11): 任何 LLMTime-style rescaling 都让 text 性能从 0.77 (none, noise=0) 掉到 0.33-0.50。Rescaling 把数字的"形态"破坏了
- **Precision**: 2-digit 最好 (0.77 at noise=0), 16-digit 掉到 0.68 (Supp Table S6-S8)
- **Separator**: comma+space vs space, 差不多 (Supp Table S3-S5)

**Intuition**: text 的核心痛点不是 precision 也不是 separator, 而是**序列长度**。$N=2500$ 时, 最好的 text 配置也只有 0.35 accuracy, 而 plot 仍然 0.74。

### 6.3 Combined Modality (Supp Fig S15)

把 plot + text 一起给 model, 准确率**没有显著超过 plot only**。这是 negative result 但很重要: 说明 plot 信号已经 saturate 了 model 的理解能力, text 不仅没帮助, 反而引入 noise。

---

## 7. Limitations 与 Honest Failures

Paper 自己承认的几个边界:

1. **不 claim 超越 task-specific model**: GPT-4o plot 在 fall detection 上的 (0.92, 0.81) 仍然不如 Aziz SVM 的 (0.96, 0.96)。但 SVM 只能做 fall detection, 而 GPT-4o 还能同时做 activity recognition、解释原因、对接其他 prompt——这种**泛化性**是 task-specific model 给不了的。

2. **多步 magnitude reasoning 时优势消失**: Derivative id. 和 quadratic derivative id. 上 plot vs text 持平甚至 text 略好 (GPT-4o 的 outlier 行为)

3. **Tabular / 短序列无优势**: Readiness 任务上 plot vs text 持平

4. **没研究 plot 形式的最优选择**: paper 用 human-written matplotlib 代码, 没探索比如多 y-axis、heatmap、stack plot 等是否更好

5. **缺 mechanistic 解释**: paper 只做 empirical, 没解释 *为什么* vision encoder 这么擅长 trend 理解。这是 Karpathy 你可能会想做的 follow-up——比如 probe ViT 的 attention pattern, 看 trend information 是在哪个 layer、哪些 patch 上被编码的

---

## 8. 我的延伸思考与 Open Questions

### 8.1 Token Efficiency vs Information Density 的根本 trade-off

Vision token 是高度 lossy 的 encoding: 一张 258-token 的 plot 可以承载 10,000 个数据点的 trend, 但如果 model 需要的是**精确数值** (比如 forecasting), 这种 lossy 编码就崩了。Paper 刻意避开 forecasting, 因为那是 vision encoder 的弱项。

但这给我一个 hypothesis: **future multimodal model 应该有一个 native 的"chart understanding module"**, 输入图像 patch + 可选 zoom-in token, 输出 numerical reconstruction——这是 DePlot 的反向, 但训练目标应该 joint: image → numeric table → reasoning chain → answer。

### 8.2 为什么 vision encoder 在 trend 上强?

我怀疑是因为 CLIP-style contrastive training 阶段, "a chart showing upward trend" / "an increasing function" / "exponential growth curve" 这些 captions 跟具体的 plot 图像配对过很多次。CLIP 的 LAION-400M / 5B 数据集里有大量 scientific paper figure、finance chart、educational illustration。所以 GPT-4o 的 vision encoder 实际上"见过" millions of plots, 学到了 plot-to-trend 的映射。

这意味着: **如果一个 plot 风格完全 out-of-distribution (比如 radar chart with polar coords)**, vision encoder 可能就退化。Paper 没测这个。

### 8.3 跟 Patch-based Time-series 的对比

最近 Time-Series Patching (Nie et al., ICLR 2023: https://arxiv.org/abs/2211.14730) 把 time-series 切成 patch token, 类似 ViT 的 patch embedding。这跟 plot-based 方法本质都一样: **把 1D sequence 转成 2D representation**。区别只是 patch 是自动学习的 representation, plot 是 human-engineered representation。

但 plot 有一个 patch 没有的优势: **经过 cultural evolution 优化过的 visual encoding**。人类几百年改进 chart 设计 (Tufte 的 The Visual Display of Quantitative Information), 已经知道什么样的 visual encoding 最易读——这相当于免费拿到了一个 prior。

### 8.4 跟 Tool-use 的结合

Paper Section 5 提到 future work: 让 model 自己 decide when to plot, then 调用 plotting tool, 再 read the plot。这其实就是一个 agentic loop, 类似 Code Interpreter。OpenAI 的 Code Interpreter 实际上已经支持这个, 但 paper 没系统评估。

如果我做 follow-up, 我会:
1. 给 model 一个 `plot(x, y)` tool
2. 评估在什么任务下 model 会主动调用 plot
3. 评估 model 自己生成的 plot vs human 生成的 plot 的 quality gap
4. 用 model-critic 评估 plot quality, 找出 model 在 plot 上的 failure modes

### 8.5 Multimodal Co-training 的影响

GPT-4o 和 Gemini 1.5 Pro 的 vision encoder 都跟 LLM 做 co-training。这跟 CLIP-style pure contrastive 不同——multimodal co-training 让 vision encoder 学到"reasoning-friendly"的 representation。Paper 没比较纯 CLIP embedding 跟 GPT-4o native vision encoder 的差异, 这是一个 missed opportunity, 但也合理 (因为拿不到 GPT-4o vision embedding 单独使用)。

如果我能访问 GPT-4o 的 vision embedding API, 我会做一个实验: 把 plot embedding 出来, 跑一个 lightweight probe classifier, 看 trend 信息在 embedding 哪个维度上。这是 mechanistic interpretability 的路子。

---

## 9. 关键参考链接

| Reference | Link |
|---|---|
| Paper 本身 (假设 arXiv 公开) | https://arxiv.org/abs/2410.02749 (推断) |
| LLMTime (Gruver et al. NeurIPS 2024) | https://arxiv.org/abs/2310.07800 |
| Chronos (Ansari et al. 2024) | https://arxiv.org/abs/2403.07815 |
| MOMENT (Goswami et al. 2024) | https://arxiv.org/abs/2402.03885 |
| MOIRAI (Woo et al. 2024) | https://arxiv.org/abs/2402.02592 |
| Time-Series Patching (Nie et al. ICLR 2023) | https://arxiv.org/abs/2211.14730 |
| TimeSeriesExam (Cai et al. 2024) | https://arxiv.org/abs/2410.14752 |
| Spathis & Kawsar JAMIA 2024 | https://academic.oup.com/jamia/article/31/9/2151/7711113 |
| IMUFD dataset (Aziz et al. 2017) | https://pubmed.ncbi.nlm.nih.gov/27334988/ |
| HHAR dataset (Stisen et al. 2015) | https://dl.acm.org/doi/10.1145/2809695.2809718 |
| Personal Health LLM (Cosentino et al. 2024) | https://arxiv.org/abs/2406.06474 |
| DePlot (Liu et al. ACL 2023) | https://aclanthology.org/2023.findings-acl.10381 |
| IMU2CLIP (Moon et al. EMNLP 2023) | https://aclanthology.org/2023.findings-emnlp.885/ |
| ImageBind (Girdhar et al. CVPR 2023) | https://arxiv.org/abs/2304.08073 |
| CLIP (Radford et al. ICML 2021) | https://arxiv.org/abs/2103.00020 |
| CharXiv (Wang et al. 2024) | https://arxiv.org/abs/2406.18521 |
| SPIQA (Pramanick et al. 2024) | https://arxiv.org/abs/2407.09413 |
| VLMs are Blind (Rahmanzadehgervi et al.) | https://arxiv.org/abs/2407.06581 |
| Langfun library | https://github.com/google/langfun |
| OpenAI API Pricing | https://openai.com/api/pricing/ |
| Gemini API image cost | https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding |
| Survey: LLMs for Time-Series (Zhang et al. IJCAI 2024) | https://doi.org/10.24963/ijcai.2024/921 |

---

## 10. TL;DR for Karpathy

这篇 paper 的真正贡献是 **用一个 zero-training trick 把"人类直觉"形式化并验证**: 看图比读数列更适合理解趋势。它揭示了 multimodal foundation model 的 vision encoder 在 pretraining 时已经隐式学到了 time-series 的几何特征——CLIP-style 训练数据里的 chart / scientific figure 是这种能力的真正来源。Token economics 也漂亮: text 是 $O(N)$, plot 是 $O(1)$, 这让 long-context multimodal 应用 (健康监测、IoT、金融) 的 cost 直降 10×。

但 paper 留下了几个 mechanistic 问题: 为什么 cubic / periodic 是 hard case? 为什么 GPT-4o 在 quadratic derivative 上 text 反超 plot? 为什么 combined modality 不超过 plot alone? 这些都指向一个更深的问题: **multimodal model 内部, vision encoder 和 text encoder 的 reasoning 是怎么分工的?** 这值得 interpretability work。

从产品视角, 这个方法的 sweet spot 是: long time-series (>1000 points) + 趋势识别 + few-shot prompting + 跨任务泛化需求。不适用: 精确数值 forecasting、tabular 短数据、需要 magnitude 推理的任务。

你的 Lecture 系列 "Let's build GPT" 里讲到 attention 时强调过 "tokens 是思考的原子单位"。这篇 paper 的反直觉点在于: 对 time-series, **plot 是更高效的原子单位**, 即便它丢了一些精度。这跟你在 Zero to Hero 系列里反复强调的 "inductive bias matters" 完全一致——vision encoder 的 inductive bias (spatial locality, scale invariance, contrast-based features) 恰好对 trend 这种 2D 几何结构是 well-matched。
