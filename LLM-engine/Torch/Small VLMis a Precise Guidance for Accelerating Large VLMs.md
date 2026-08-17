---
source_pdf: Small VLMis a Precise Guidance for Accelerating Large VLMs.pdf
paper_sha256: c146ec077ae68a6323b9f8aff47d95780abff7ea1fdcc3f6cef08265cc6de212
processed_at: '2026-08-12T07:55:49-07:00'
target_folder: LLM-engine/Torch
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 paper

## 一句话版本

**让小模型先看一眼图，告诉大模型"重要的地方在哪"，大模型只看这些地方就行；简单问题小模型自己就答了，难题才丢给大模型。**

---

## 打个比方

想象一个公司：

- **大模型** 是资深专家，什么都会，但请他出马很贵、很慢
- **小模型** 是实习生，工资低、速度快，能力差点但够用

现在来了一批活儿（用户提问）。传统做法是每件活都直接甩给专家，专家累死，账单爆炸。

SGL 的做法分两步：

### 第一步：实习生先过一遍

实习生拿到图和问题，先自己看一遍。这一遍他做两件事：

1. **生成一个完整答案**（可能对可能错）
2. **顺便记下"我刚才看图的时候，眼睛主要盯在哪些区域"**——这就是 attention map

关键发现：实习生虽然答得不一定准，但他**看的地方基本是对的**。就像你问实习生"这张图里写了什么电话号码"，实习生可能把号码读错了，但他的眼睛确实盯着图角落那块小字。专家只要顺着实习生手指的方向看就行。

### 第二步：决定要不要请专家

实习生答完之后，系统会问两个问题：

1. **"你对自己的答案有多确定？"**（confidence）
2. **"如果我把图大部分擦掉只留你看的那 5%，你还坚持同样的答案吗？"**（consistency）

两个都"是"→ 那大概率是简单题，实习生答对了，直接交差，专家根本不用出场。

有任何一个"否"→ 这题有难度，请专家来。但专家来了也不用从头看整张图——实习生已经标好了"重要区域在哪"，专家直接只看这些区域就行。

---

## 为什么这招管用

### 为什么实习生"看的地方"是准的

因为"看哪里"和"答什么"是两种不同的能力。看哪里是 perception，是比较 low-level 的能力，小模型参数少也学得会。答什么需要 reasoning + knowledge，这个吃参数量。

打个比方：你问一个小学生"这道高考数学题答案是多少"，他肯定答错。但你问他"这题的图里哪部分重要"，他大概率能指对——因为认图不需要会做题。

### 为什么以前的方法不行

之前 FastV 的做法相当于：**让专家自己扫一眼图，根据第一眼印象决定哪些地方重要**。问题是第一眼（单层 attention）信息太少，经常扫错。尤其图里文字很小的时候，第一眼根本扫不到，就把关键信息砍掉了。

SGL 让实习生仔仔细细看完整张图（所有层的 attention 聚合），虽然是用小脑容量在看，但看得全面。看完生成一份"重点区域清单"给专家，专家直接照着清单干活。

### 为什么连实习生一起用反而更快

因为实习生便宜到几乎可以忽略。2B 模型的 FLOPs 比 76B 少几十倍。而且：

- 简单问题（占大多数）实习生直接答完，专家完全不启动
- 难题才启动专家，但专家只看 9% 的 token，速度飞快

综合下来反而更快。而且模型越大，省得越多——因为大模型越贵，实习生的相对成本就越低。

---

## 效果有多好

几个数字直观感受：

- **砍掉 91% 的 visual tokens**，性能只掉 ~10%
- **76B 模型上显存反而更省**（因为 token 少了 KV-cache 大幅缩小，抵消了小模型的占用）
- **20% 的问题直接由小模型回答**，性能几乎不掉
- 换到 Qwen2-VL、LLaVa-OV 不同的架构也能用

---

## 唯一的前提条件

小模型和大模型必须**用同一个 vision encoder**。因为小模型要告诉大模型"第 123 号 token 重要"，大模型得知道"第 123 号 token"对应图的哪个位置。如果两个模型用的 vision encoder 不同，token 编号对不上，这招就废了。

所以 InternVL2 系列天然适配（1B/2B/4B/26B/40B/76B 共享 InternViT），Qwen2-VL 和 LLaVa-OV 内部也各自适配。但跨 family 就得另想办法。

---

## 限制

- 只测了图片，视频没测
- 只测了理解任务，生成任务（比如画图）没测
- 小模型到底要多小才最优，作者也没完全搞清楚——1B 居然比 2B 还好一点

---

就这样。核心 idea 其实特别简单：**实习生当导游，专家当回答者，导游的能力门槛比回答者低很多，所以可以用便宜的导游替代昂贵的专家做大部分工作。**

---

# A Stitch in Time Saves Nine: Small VLM Guides Large VLM — 深度讲解

## 一、为什么这篇 paper 值得关注

VLM 的推理瓶颈几乎全在 visual tokens 的数量上。InternVL2-26B/76B 用了 dynamic high-resolution，单张图就能产生 10496 个 visual tokens，而 prompt 通常只有几十到几百 tokens。self-attention 的复杂度是 $O(N^2)$，KV-cache 还要 $O(N)$ 的显存随层数线性增长。所以怎么把 visual tokens 砍掉 90% 还不掉点，是 VLM 部署的关键问题。

这篇 paper 来自 NUS 的 Yang You 组和 Alibaba DAMO，作者团队里 Yizeng Han（清华 Gao Huang 组，dynamic networks 方向）和 Zhangyang Wang（UT Austin）都是 efficiency 方向的活跃研究者。代码开在 https://github.com/NUS-HPC-AI-Lab/SGL，arXiv 地址 https://arxiv.org/abs/2412.01896。

它的核心 idea 非常简洁，让我先把 intuition 建立起来。

---

## 二、三个关键 empirical insight

### Insight 1：单层 attention map 不够用

FastV（https://arxiv.org/abs/2403.06764）的做法是：拿 LM 某一层（比如第 2 层）的 attention map，看 visual tokens 从 query tokens 那里收到多少 attention，按这个分数排序做 prune。问题是单层 attention 是局部视图，它的统计分布和"全局哪些 token 真正重要"对不上。

Figure 1(a) 给的数据非常震撼：在 InternVL2-26B 上，retention ratio 降到 9% 时，FastV 几乎崩溃（TextVQA 43.84 vs 100% 时 82.45），而 **oracle** 方法（把所有 48 层 attention map 聚合起来做排序）还能保住 80+。

### Insight 2：oracle 要 full pass，不实用

要拿到所有层的 attention，必须先做一次完整 forward，这就和"加速"的目的自相矛盾了。FastV 选择单层就是为了规避这个问题，但代价就是 Insight 1 的失败。

### Insight 3（核心）：small VLM 的全局 attention ≈ large VLM 的全局 attention

这是整个 paper 的灵魂。作者发现 InternVL2-2B（1.8B LM + 300M ViT）的聚合 attention map，在 token selection 的 ranking 上和 InternVL2-26B 高度一致。更妙的是：**即便 small VLM 答错了问题，它选出来的"重要 token"和 large VLM 的 oracle 仍然几乎重合**。这意味着 small VLM 的"看哪里"能力 >> 它的"答什么"能力。

这其实是个非常自然的现象，我自己在做 distillation 时也观察到类似规律：localization/perception 的能力在 small model 里已经基本到位，缺的是 reasoning 和 knowledge。所以让 small model 做"导游"，让 large model 做"回答者"，是合理分工。

Figure 1(c) 还给了一个佐证：2B 模型的 FLOPs 比 26B 少一个数量级，但平均性能只低 ~10%。这个 gap 就是 early exiting 的操作空间。

---

## 三、方法 SGL = SGP + SEE

整体 pipeline 见 Figure 2。我用文字画一下：

```
Image + Question
       │
       ▼
   [Small VLM 2B]  ──► answer₁, confidence, consistency
       │
       │  aggregated attention A (Eq.4 + Eq.5)
       │
       │  ┌── SEE: S = (S_conf + S_consist)/2 > threshold?
       │  │     YES → 直接返回 small VLM 的 answer，结束
       │  ▼     NO  → 继续
       │
       ▼
   [Large VLM 26B/76B]
       │  visual tokens 按小模型的 A 排序
       │  在第 2/9/19 层 prune 掉 60%/80%/95%
       │
       ▼
   answer₂ (最终输出)
```

### 3.1 SGP：Small VLM-Guided Pruning 的细节

**Step 1 — Small VLM 跑一遍**，用 InternVL2-2B。设 vision encoder 输出 $N_I$ 个 image tokens，prompt tokenize 成 $N_T$ 个 text tokens，small LM 共 $L$ 层、$H$ 个 head。

**Step 2 — Prefilling 阶段的 attention 抽取**。在 prefilling 时，每层每 head 的 attention map 是

$$\mathbf{A}_{j,k}^{P} \in \mathbb{R}^{(N_I + N_T) \times (N_I + N_T)}$$

其中 $j \in [1, L]$ 是 layer index，$k \in [1, H]$ 是 head index。由于 causal mask，这是下三角矩阵。作者只关心 **prompt tokens 对 visual tokens 的 attention**（注意 attention 是 query→key，所以这里实际是 "visual tokens as key 被 prompt tokens as query attend 到"）。所以切出 bottom-left block：

$$\tilde{\mathbf{A}}_{j,k}^{P} \in \mathbb{R}^{N_T \times N_I}$$

含义：行是 prompt token query，列是 visual token key。然后对 $N_T$ 行求和，得到每个 visual token 的"被关注度"向量：

$$\bar{\mathbf{A}}_{j,k}^{P} \in \mathbb{R}^{N_I}$$

最后跨层跨 head 累加：

$$\mathbf{A}^{P} = \sum_{j=1}^{L}\sum_{k=1}^{H} \bar{\mathbf{A}}_{j,k}^{P}$$

注意是 streaming accumulation，不需要把所有 $\mathbf{A}_{j,k}^{P}$ 缓存下来（不然显存爆）。这一点工程上很重要。

**Step 3 — Decoding 阶段**。生成 $N_G$ 个 token，每个 generated token 在每层每 head 都对 visual tokens 有 attention：

$$\mathbf{A}_{i,j,k}^{D} \in \mathbb{R}^{N_I}$$

$i \in [1, N_G]$ 是生成 token index。累加：

$$\mathbf{A}^{D} = \sum_{i=1}^{N_G}\sum_{j=1}^{L}\sum_{k=1}^{H} \mathbf{A}_{i,j,k}^{D}$$

**Step 4 — 合并**：

$$\mathbf{A} = \mathbf{A}^{P} + \mathbf{A}^{D}$$

这就是最终每个 visual token 的重要性分数向量。然后 sort，取 top-R%。

**Step 5 — 在 Large VLM 里 prune**。同一张图喂给 large VLM 的 vision encoder（InternVL2 系列里 small/large 共享 InternViT 架构，token 数一致，这是关键前提！），在前几层（默认第 2/9/19 层）按照 small VLM 给出的 ranking mask 掉其余 visual tokens，后续层只计算保留 token 的 KV。

**这里有个值得吐槽的细节**：paper 说 "we progressively update $\mathbf{A}^P$ in an accumulative manner without caching all $\mathbf{A}_{j,k}^P$"。但 $\mathbf{A}^D$ 需要等 $N_G$ 个 token 全生成完，意味着 small VLM 必须先把完整答案生成出来，再决定 large VLM 怎么 prune。这个 latency 是要算进总时间的。SEE 的设计就是为了把这个 latency "复用"——small VLM 既然已经生成了，那简单问题就直接返回了。

### 3.2 SEE：Small VLM Early Exiting

这里有两个 score 的设计，我觉得是 paper 里最有意思的部分。

**Confidence score** —— length-normalized sequence probability：

$$\mathcal{S}_{confidence} = \exp\left\{\frac{1}{N_G}\log P(\mathbf{x}_G^1, ..., \mathbf{x}_G^{N_G})\right\}$$

其中

$$P(\mathbf{x}_G^1, ..., \mathbf{x}_G^{N_G}) = \prod_{i=1}^{N_G} P(\mathbf{x}_G^i \mid \mathbf{LM}^S(\mathbf{x}_I, \mathbf{x}_T, \mathbf{x}_G^{1:i-1}))$$

变量解释：$\mathbf{x}_G^i$ 是第 $i$ 个 generated token，$\mathbf{LM}^S(\cdot)$ 是 small LM 给出的 logits softmax 后的分布，$P(\mathbf{x}_G^i \mid \cdot)$ 是 ground-truth token 的概率。取 log 求和再除以 $N_G$ 是经典的 length bias correction（Murray & Chiang 2018，https://arxiv.org/abs/1808.10006），避免长答案被天然惩罚。再 exp 回到概率尺度。

**Consistency score** —— 这是 paper 的创新点。直觉是：

> 如果 small VLM 的答案是对的，那么 prune 掉 95% tokens 后，small VLM 在 pruned 上重新 forward（teacher forcing，不是重新生成！）应该给出几乎一致的高概率。

公式：

$$S_{consistency} = \prod_{i=1}^{N_G} P(\mathbf{x}_G^i \mid \mathbf{LM}^{S'}(\mathbf{x}_I, \mathbf{x}_T, \mathbf{x}_G^{1:i-1}))$$

其中 $\mathbf{LM}^{S'}$ 是 small LM 在 pruned visual tokens（保留 5%）下的版本。这里关键 trick：**是 teacher forcing，不是 autoregressive regenerate**。因为 $\mathbf{x}_G^{1:i-1}$ 都已经有了，所以可以一次性并行计算所有 $i$ 的概率，等价于一次 prefilling。再加上 95% tokens 被 prune 了，这次 forward 极其便宜，作者实测 < small VLM 原始推理时间的 10%。

**Final decision score**：

$$S = \frac{1}{2}(S_{confidence} + S_{consistency})$$

简单的平均。作者 ablation（Figure 4）显示这个组合比单用任何一个、比 quantile（Gupta et al. https://arxiv.org/abs/2404.10136）、比 entropy（Fadeeva et al. LM-Polygraph https://arxiv.org/abs/2311.07383）都好。

**Intuition 解释**：confidence 测"模型有多相信自己说的"，consistency 测"模型的信念在 token 压缩扰动下稳不稳"。两个都高 → 模型既确信又 robust → 大概率是对的。任何一个低 → 要么不确信要么脆弱 → 召回 large VLM。这是一个非常 cheap 的 self-consistency 变体，类似 self-consistency (Wang et al. 2022) 但是用 token pruning 当 perturbation 而不是用 temperature sampling。

---

## 四、实验数据解读

### Table 1：SGP 单独 vs ToMe / FastV

最关键的一行是 9% retention：

| Method | TextVQA | ChartQA | DocVQA | GQA | SEED | MMBench | MM-Vet | MME | RC | RC+ | RC-g | score ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 26B baseline | 82.45 | 84.92 | 92.14 | 64.89 | 76.78 | 83.46 | 64.00 | 2270 | 91.24 | 86.67 | 88.44 | 100% |
| FastV (9%) | 43.84 | 26.20 | 26.81 | 44.90 | 54.56 | 62.33 | 31.60 | 1799 | 19.65 | 16.66 | 17.22 | 46.99% |
| ToMe (9%) | 51.69 | 28.60 | 28.46 | 57.52 | 65.19 | 73.09 | 37.70 | 1933 | 20.33 | 17.74 | 19.36 | 54.28% |
| **SGP (9%)** | **78.98** | **72.96** | **87.26** | **62.10** | **72.23** | **75.56** | **52.10** | **2004** | **80.36** | **72.22** | **77.45** | **89.58%** |

可以看到在 OCR-heavy 任务（DocVQA、ChartQA、TextVQA）上 SGP 把 FastV 拉开 40-60 个点。这非常说明问题：OCR 任务要求模型必须精确定位到文字所在的几个 patch，single-layer attention 的 FastV 经常把这些 patch 砍掉，而 SGP 因为有 small VLM 全局 attention 的指导，保住了关键 region。

Table 2 还做了一个非常有信息量的 ablation：用 2B 模型不同比例的层数 vs 用 26B 单层 vs 用 26B 全层（oracle）。9% retention 下：

- 26B 单层（FastV）：score ratio 48.37%
- 2B 10% 层：51.92%
- 2B 30% 层：56.15%
- 2B 50% 层：80.31%
- 2B 70% 层：91.40%
- 2B 全层：92.64%
- 26B 全层（oracle）：94.44%

单调上升，且 2B 全层基本逼近 oracle。这个曲线告诉我：**attention 的"重要性信号"是分布式的，藏在很多层里**，单层抽样的方差太大。small model 虽然容量小，但它的 attention 分布是"完整"的，所以比 large model 的单层切片信息量更大。这和 vision transformer 里观察到的 "attention 在深层才语义化、浅层偏 local" 的现象一致——只用一层会 bias 到某一种 pattern。

### Table 3：用哪些 token 做 attention aggregation

| Used token | TextVQA | SEED | RC | score ratio |
|---|---|---|---|---|
| last prompt token (FastV 做法) | 79.40 | 69.53 | 13.63 | 67.26% |
| prompt tokens | 76.15 | 72.25 | 59.90 | 84.03% |
| generated tokens | 79.38 | 63.47 | 83.51 | 90.16% |
| prompt + generated | 78.98 | 72.23 | 80.36 | 92.64% |

特别有意思：**只用 last prompt token（FastV 的做法）在 RC（RefCOCO grounding）上直接崩到 13.63**。原因很直观：grounding 任务答案是要输出 bbox 坐标，问题里没有具体指代对象的位置信息，最后一个 prompt token 的 attention 主要飘向整个 object region，分辨率极低。而 generated tokens（坐标 token 序列）的 attention 才会精确锁定到目标物体。所以 generated tokens 的 attention 信号对于 grounding 是 essential 的。这是 FastV 类方法的一个系统性盲点。

### Figure 3 + 6 + 7：SGL（SGP + SEE）的 efficiency curve

关键观察：

1. **26B 上 SGP 单独比 FastV 慢**，因为 2B small VLM 的 overhead 占比大（26B 还不够大）。
2. **40B / 76B 上 SGP 反超**，因为 small VLM 占比变小，而 SGP 让 large VLM 在低 retention 下性能远好于 FastV。
3. **SEE 在所有 size 上都显著加速**，20% early exit ratio（即 80% 问题直接 small VLM 答）下性能损失很小。

这里有个推论：**模型越大，SGL 越划算**。这正好对应了 scaling law 趋势——大家都在用越来越大的 VLM，所以 SGL 的实际价值会越来越高。

### Table 5：跨 size 泛化

| Large VLM | TextVQA | SEED | RC | score ratio |
|---|---|---|---|---|
| 26B baseline | 82.45 | 76.78 | 91.24 | 100% |
| 26B + SGP(9%) | 78.98 | 72.23 | 80.36 | 92.64% |
| 40B baseline | 83.11 | 78.15 | 93.00 | 100% |
| 40B + SGP(9%) | 79.96 | 74.11 | 79.99 | 92.38% |
| 76B baseline | 84.33 | 78.17 | 92.20 | 100% |
| 76B + SGP(9%) | 80.72 | 73.93 | 81.82 | 92.98% |

92-93% 的 score ratio 在三个 size 上几乎一致。这说明 small→large 的 attention 一致性不依赖于 large model 的具体 scale，是一个相当 robust 的现象。

### Table 6：跨架构泛化

Qwen2-VL-72B + SGP(9%) 在 TextVQA 上保住 96.94%，LLaVa-OV-72B 保住 95.81%。这比 InternVL2 系列还高，可能是因为 Qwen2-VL 和 LLaVa-OV 的 visual token redundancy 更大（Qwen2-VL 用 dynamic resolution，token 数极多）。这进一步说明 SGL 是 architecture-agnostic 的。

---

## 五、Memory 分析（Appendix B）

Table 7 给了显存数据：

| Large VLM | Baseline peak | +SGL peak | Δ |
|---|---|---|---|
| 26B | 51.60 GiB | 54.24 GiB | +5.11% |
| 40B | 77.94 GiB | 80.60 GiB | +3.41% |
| 76B | 147.64 GiB | 147.25 GiB | -0.26% |

76B 上 SGL 反而省显存！因为 prune 掉 91% visual tokens 后，KV-cache 大幅缩减，足以抵消 small VLM 的权重占用。这个结果对部署非常友好。

---

## 六、更深的 intuition 与联想

### 1. "Small model 的 attention pattern 转移到 large model" 本质是一种 cross-model distillation

这让我想到 ensemble distillation 和 attention transfer（Komodakis & Zagoruyko, ICLR 2017, https://arxiv.org/abs/1612.03928）。但 SGL 不训练、不更新权重，是 inference-time 的 zero-cost distillation。前提条件是两个模型共享 vision encoder 架构（InternViT 系列），这样 visual token 在 small 和 large 之间是 spatially aligned 的。如果换成完全不同的 vision encoder（比如 CLIP vs SigLIP），token alignment 就得重新做。

### 2. Consistency score 是 self-consistency 的廉价变体

经典 self-consistency（Wang et al. https://arxiv.org/abs/2203.11171）要 sample 多条 reasoning path 投票，成本高 N 倍。这里用 token pruning 当 perturbation，等价于一次额外的 cheap forward（5% tokens）。这个思路可以推广到很多场景：用 input corruption / dropout / token masking 当 perturbation 来估 confidence，比单纯看 softmax 概率 robust 得多。

### 3. 和 speculative decoding 的关系

SGL 的 pipeline 在结构上很像 speculative decoding（Leviathan et al. https://arxiv.org/abs/2211.17192）：small model 先生成，large model 选择性介入。区别是：

- Speculative decoding 是 token-level 的，small model 生成 draft，large model 并行 verify。
- SGL 是 sample-level 的，small model 生成完整答案，large model 看一眼 confidence 决定要不要重做。

可以想象一个混合方案：small VLM 用 speculative decoding 的方式生成，同时它的 attention 用来指导 large VLM 的 pruning。这样 latency 和 throughput 都能优化。我猜测这个方向已经有人在做。

### 4. 为什么不直接用 CLS token / 注册 token？

paper 里没讨论，但我觉得值得思考。SigLIP、InternViT 都没有 class token，所有 patch 都是平等输入。如果有 class token，理论上可以用它的 attention 做 global importance。但实际上 VLM 里 visual tokens 直接进 LM，没有 class token 的概念，所以必须从 LM 侧拿 attention。这也是 FastV 系列方法的出发点。

### 5. 9% retention 的物理意义

一张 448×448 图，InternViT-6B patch size 14，就是 32×32 = 1024 个 patch；dynamic resolution 下会到几千上万。9% retention 意味着只保留 100-1000 个 patch 的 KV。对一个 OCR-heavy 任务来说，关键文字可能就集中在几十个 patch 上，9% 是合理的下限。再低就真不够了。从 Table 1 看，9% 时 SGP 还能保 89.58%，说明这个下限是被 small VLM attention 的"分辨率"决定的，不是被任务需求决定的。

### 6. 和 token merging (ToMe) 的本质区别

ToMe（Bolya et al. https://arxiv.org/abs/2210.09461）在 vision encoder 内部基于 embedding similarity 做 bipartite soft matching，把相似 token merge 起来。它的问题：similarity 是 vision-only 信号，不知道哪些 token 对当前 question 重要。FastV 用 LM attention，但只用单层。SGL 用 LM attention 的全局聚合，且来自 small model。**整个领域的演进方向是：从 vision-only signal → single-layer LM signal → global LM signal → global LM signal from a proxy model**。每一步都在用更准确的 signal 换更便宜的 cost。

### 7. Limitations 我觉得作者没充分讨论

- **Video VLM**：paper 只测了 image。Video 里 token 数量爆炸更严重，但 small VLM 的 attention 在时序上是否稳定？跨帧的 attention aggregation 怎么做？这是个 open problem。
- **生成任务**：Limitations 里提到 unified understanding+generation 的 VLM（Emu3, Show-o, Transfusion）没测。generation 任务的 attention pattern 可能和 understanding 完全不同。
- **Small VLM 选 1B 时反而更好**（Table 4: InternVL2-1B score ratio 93.44% > 2B 的 92.64%）。这个反直觉结果作者只说了一句 "motivating further reducing the small VLM size in future studies"。我猜测原因是 1B 用的是 Qwen2-0.5B，2B 用的是 InternLM2-1.8B，**不同 LM 的 attention 分布特性不同**，1B 那个反而更"sharp"。如果用 0.5B Qwen2-VL 或更小的 vision encoder-only model，可能效果更好。

### 8. 和 MoE / route-based inference 的关系

SEE 本质上是 router：small VLM 是 default expert，large VLM 是 fallback expert。这和 MoE 的思想一致，只是 MoE 在 layer 内 route，SEE 在 sample 级 route。可以想象一个 hierarchical MoE：tiny VLM → small VLM → medium VLM → large VLM，每一级都有 confidence threshold。这种 cascade 在 LLM serving 里已经有研究（Chen et al. FrugalGPT https://arxiv.org/abs/2305.05176），VLM 这边还比较少。

---

## 七、可以落地的工程建议

如果你想在自己的 VLM serving 里用 SGL：

1. **必须同源 vision encoder**。如果 small/large 来自不同 family（CLIP-based vs SigLIP-based），token 数和 spatial alignment 对不上，SGL 直接失效。这时需要先做 token alignment（比如 nearest neighbor in embedding space）。
2. **Pruning 位置选浅层**。paper 默认第 2 层 prune 95%，这意味着只有前 2 层的 KV-cache 是完整的，后面 46 层都只算 5% tokens。这是 speedup 的最大来源。
3. **SEE threshold 要按业务调**。accuracy-critical 场景设高 threshold（少 early exit），throughput-critical 设低。Figure 3 给了 20%/40%/60% 三档参考。
4. **Consistency score 的计算可以再便宜**。paper 用 5% retention 重新 forward，其实可以更激进——1% 或者直接用 small VLM 中间层的 attention 替代。我估计还能再省一半时间。
5. **配合 quantization**。small VLM 可以 int4/int8 量化，几乎不损失 attention ranking 的准确性（attention pattern 对 quantization 鲁棒性比 logits 高）。large VLM 也可以同时量化，因为 KV-cache 已经 prune 了，quantization 的精度损失更可控。

---

## 八、总结一句话

**Small VLM 的 perception 能力 >> 它的 reasoning 能力，所以它是个好导游但不是好回答者；让导游指出"看哪里"，让大模型来"答什么"，再加一个 confidence-based early exit 复用导游的劳动——这就是 SGL。**

它不训练、不改架构、不增加显存（在 76B 上甚至省显存）、在 9% retention 下保 90% 性能，是一个非常工程友好的 inference-time 加速方案。它的 limitation 也很清楚：依赖同源 vision encoder、video/generation 场景未验证、small VLM 选择还有优化空间。

参考链接汇总：

- Paper: https://arxiv.org/abs/2412.01896
- Code: https://github.com/NUS-HPC-AI-Lab/SGL
- FastV: https://arxiv.org/abs/2403.06764
- ToMe: https://arxiv.org/abs/2210.09461
- InternVL2: https://arxiv.org/abs/2404.16821
- Speculative decoding: https://arxiv.org/abs/2211.17192
- Self-consistency: https://arxiv.org/abs/2203.11171
- LM-Polygraph: https://arxiv.org/abs/2311.07383
- Language model cascades: https://arxiv.org/abs/2404.10136
- FrugalGPT: https://arxiv.org/abs/2305.05176
- Attention transfer: https://arxiv.org/abs/1612.03928
- Length bias correction: https://arxiv.org/abs/1808.10006
