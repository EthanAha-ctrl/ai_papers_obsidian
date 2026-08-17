---
source_pdf: InfiR.pdf
paper_sha256: 64ec0497fdb6a0080239b5e30f3ed43f1bd4f23ef5dcb8cc1fa6a853ded9915c
processed_at: '2026-08-05T09:38:07-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 InfiR

## 一句话版本

**这篇 paper 在说：1B 参数的小模型，只要 data 够干净、够密集，照样能做 reasoning，而且能逼近 Qwen2.5-1.5B 的水平，总成本 6000 GPU hours。**

---

## 核心赌注

大家默认 reasoning 是大模型的游戏——因为 reasoning 需要"装"很多逻辑模式、数学推导链、code 结构。1B 参数看着就不够装。

InfiR 的 hypothesis 很直接：**参数少了，就用 data density 补**。你模型小，那每一条训练数据就必须是"高信号"的，不能像训练 Llama 3 那样从 Common Crawl 里捞一通就完事。

这跟你（Karpathy）反复讲的 "data quality > data quantity" 是一个路子，InfiR 就是工业级佐证。

---

## 他们到底干了啥

### 1. Pretraining data：把 reasoning 信号"挤"出来

普通 pretraining data pipeline 是：filter → dedup → 训练。InfiR 加了一步"reasoning-oriented recall"：

- 拿一批 seed data（OpenWebMath、StackOverflow 这些已知有 reasoning 的）
- 训一个 fasttext classifier 学"什么是 reasoning text"
- 然后从整个 web corpus 里 recall 所有类似 reasoning 的内容

**直觉**：web 上 99% 是垃圾，但里面藏着 math derivation、argumentation、code explanation。你不主动挖，模型就只能学到"How to write a blog comment"。

后面还有 Min-Hash dedup、FineWeb-Edu scorer、1-5 分数学评分、token-level 10-gram decontamination——五步过滤，把 900B tokens 砸成 high-signal density。

### 2. Annealing：最后 40B tokens 才是 reasoning 注入的关键

这个是真有意思。他们 pretrain 900B tokens 后，又来一轮 40B tokens 的 "annealing"——LR 衰减到很低，数据全是 high-quality math + code + synthetic data。

**为什么这么做**：你前面 900B 是在"铺地基"（学语言、学常识），最后这 40B 才是"封顶"——LR 很小、model 已经稳定，这时候喂 high-quality reasoning data，model 会"精准吸收"。

跟你 nanoGPT 里讲的 "final 10% LR decay 决定 final quality" 是一个现象。

### 3. 一个反直觉的 evaluation insight

paper 里有一段推导特别值得讲。假设你的 clean data distribution 混了 5% 的 contaminated data：

$$
q = (1-\epsilon)p + \epsilon r
$$

那 perplexity 变化是多少？推导出来：

$$
\mathrm{ppl}_q \approx (1+\epsilon) \mathrm{ppl}_p
$$

也就是说，**5% contamination 只让 ppl 涨 5%**。听起来"无害"对吧？

但 generation 视角完全不一样：5% contamination 意味着**平均每 20 个 token 就有一个 corrupted token**。一次生成就崩了。

**Insight**: NLL / perplexity 这种 average metric 会骗你。Token-level 平均后 long-range coherence 丢失了。所以 annealing 阶段必须用 downstream benchmark 直接评估 generation，不能看 NLL。

这其实就是 Hinton 说的 "subjective probability" 在 token level 平均后失去 long-range 信号的现象。

### 4. SFT 阶段又一个反直觉

他们发现：**大模型上好用的 SFT data，搬到小模型上未必好用**。小模型需要百万级 samples 才能 competitive。

直觉解释：小模型 capacity 不够，high-complexity distribution 它吸收不动，反而需要更多 data 来"smooth out" representation。这跟 Chinchilla 预测的"小模型少 data"完全相反。

### 5. Difficulty-aware sampling 比 uniform 好

他们在 ScaleQuest-Math 上做了 ablation：

- 全部数据训练 → MATH 36.6%
- 只用 35% 的 hard data 训练 → MATH 36.6%（一样好）
- 只用 easy + very easy 训练 → MATH 23.6%

**Insight**: hard data 的 signal density 高得多。35% 的 hard data 等价于 100% mixed data。这跟 curriculum learning 的经典直觉一致——但 InfiR 是反向 curriculum：直接喂 hard 的。

### 6. Long CoT 还是要 scale

o1 / DeepSeek-R1 出来后，大家觉得 long CoT 是"解锁"reasoning 的开关。s1 paper 说 1000 个 long CoT sample 就够了。

InfiR 在 1B 上做的实验不一样：

| Data 量 | AIME24 | MATH500 | GPQA |
|---|---|---|---|
| 200K long CoT | 0.033 | 0.474 | 0.288 |
| 2M long CoT | 0.067 | 0.620 | 0.364 |

**Data 翻 10 倍，性能持续涨**。小模型不像大模型那样 in-context reasoning 强，必须把 reasoning pattern 真正 compress 进 weights，所以需要更多 data。

### 7. Multimodal：主要是 GUI Agent

InfiR-VL-1.6B 用 SigLip-So400m + MLP + InfiR-1B-Base，三阶段训练：

1. 冻 LLM，只训 MLP projector → 学 visual-textual alignment
2. 解冻 ViT + adapter，训 text rendering + GUI data → 学 vision reasoning
3. 全参数训 trajectory + tool use → 学 planning

关键应用场景是 **GUI Agent**——能看手机截图、点按钮、操作 Android。AndroidWorld 上 9.48% accuracy，比 Showui-2B (6.90%) 高 28%。

这块跟你讲的 "Software 2.0" 思路直接对接：用 model 替代 hand-coded GUI automation。

---

## 结果到底多好

**Base model (few-shot)**：
- InfiR-1B-Base 在 MATH/HumanEval 上接近 Qwen2.5-1.5B（参数少 33%）
- 比 Llama-3.2-1B reasoning 平均高 2.26x

**Instruct model (zero-shot)**：
- InfiR-1B-Instruct GSM8K 70.9 vs Llama-3.2-1B 47.9（+23）
- MATH 46.4 vs 30.0（+16）
- HumanEval 58.54 甚至超过 Qwen2.5-1.5B 的 51.83

**代价**：5760 GPU hours = 64×H800 跑 90 小时。对工业界来说几乎算"零成本"。

---

## 我的判断

**好的地方**：
- Data pipeline 设计极其细致，每一步都有 ablation 支撑
- Annealing contamination 的理论分析很有 pedagogical value
- 6000 GPU hours 做出接近 Qwen2.5-1.5B 的 reasoning，efficiency 极高
- GUI Agent 方向实用性强

**局限**：
- 没有 RL stage，跟 R1-Distill-Qwen-1.5B 还有显著差距
- MMLU 这种 general knowledge 仍然弱（50.22 vs Qwen 61.78），明显是 code + math 过度倾斜的 trade-off
- MMMU 只有 38.8，general multimodal reasoning 受 1B backbone 限制
- 全是 benchmark，没 real-world deployment 数据

**核心 takeaway**：在 small model regime，data engineering 比 model architecture 重要得多。InfiR 没有任何 architecture 创新，全靠 data pipeline 把 1B 模型推到了"看起来像 1.5B"的水平。

这对你 LLM1010 课程其实是个好案例——讲 "data is all you need" 的时候，可以引用 InfiR 当 evidence。

---

# InfiR: Crafting Effective Small Language Models and Multimodal Small Language Models in Reasoning

## Paper Overview

InfiR 是 Reallm Labs 联合 HK PolyU、Zhejiang University 等机构的工作，核心贡献在于证明：**在 1B parameter scale 下，通过精心设计 data pipeline + annealing + SFT，能够达到接近 Qwen2.5-1.5B 的 reasoning 能力，且总训练成本 < 6000 GPU hours**。这对 edge deployment 与 privacy-sensitive scenario 有重要意义。

- GitHub: https://github.com/Reallm-Labs/InfiR
- arXiv (推断): https://arxiv.org/abs/2501.04575 (InfiGUIAgent 同期工作) 或直接搜 InfiR
- 相关背景工作：
  - DeepSeekMath: https://arxiv.org/abs/2402.03300
  - OpenCoder: https://arxiv.org/abs/2411.04905
  - FineWeb: https://arxiv.org/abs/2406.17557
  - s1 (Simple test-time scaling): https://arxiv.org/abs/2501.19393

---

## 1. Core Intuition: 为什么 1B 模型能 reasoning

Karpathy 你肯定熟悉 "Chinchilla scaling" 的困境——小模型在 fixed compute 下信息存储容量受限。InfiR 的核心 hypothesis 是：**当 parameter 减少，必须通过 data density 补偿**。具体地：

- **Code data 提供结构化逻辑**（programmatic logic）
- **Reasoning-oriented text data 提供 chain-of-thought 模式**（math derivation、argumentation）
- **Pretraining data 的 signal-to-noise ratio 必须远高于 large model 所需**

这与 TinyStories、Phi-series 的思路一脉相承，但 InfiR 把这一思想推到了 reasoning 的极致。

---

## 2. Pre-training Data Pipeline 详解

### 2.1 Pipeline 五步法 (Figure 1 解析)

```
Raw Corpus
   │
   ├─(1) Heuristic filtering ─→ FineWeb rules + 语言过滤
   │
   ├─(2) Reasoning-oriented recall ─→ fasttext 分类器召回
   │
   ├─(3) Deduplication ─→ Global Min-Hash
   │
   ├─(4) Quality assessment ─→ FineWeb-Edu scorer + 1-5 数学评分
   │
   └─(5) Decontamination ─→ token-level 10-gram 去重
```

**Step 2 Reasoning-oriented recall 的 seed 设计** 是关键创新：
- Math seeds: OpenWebMath + InfiMM-WebMath
- Code seeds: StackOverflow
- Other domains: 用 Qwen2.5-7B-Instruct 对 URL/title 打标 + LLM-synthesized responses 作 seed

然后训 domain-specific fasttext classifier，用正样本（seed）+ 随机负样本（web/books），在大 corpus 上 recall。这种 "anchor + expansion" 的方式类似 DeepSeekMath 的 Pile extension。

**Step 3 Min-Hash dedup** 的 motivation：small model 对 homogeneous data 特别敏感，重复 pattern 会让 model collapse 到 narrow distribution。

### 2.2 Offline NLL Evaluation (Section 2.3)

这是 paper 里我个人觉得最有 pedagogical value 的部分。他们用 NLL 来 guide data-mixing，给出了一个 contaminated distribution 的理论分析。

**NLL 定义** (Eq. 1):

$$
\mathrm{NLL} = -\sum_{i=1}^{n} \log P(t_i | t_{<i})
$$

- $t_i$: 文本序列中第 $i$ 个 token
- $P(t_i | t_{<i})$: 模型在给定前缀 $t_{<i}$ 下对 $t_i$ 的预测概率
- 求和范围 $i \in [1, n]$，$n$ 是 sequence length

**Multiple-choice normalized probability** (Eq. 2):

$$
P_{\text{normalized}}(c_i) = \frac{e^{s_i}}{\sum_{j=1}^{k} e^{s_j}}
$$

- $s_i$: 模型对 choice $i$ 的 logit
- $k$: 选项数量
- 这是 softmax over choices，让 NLL 与 accuracy 相关且随 model scale 稳定

**Perplexity** (Eq. 3):

$$
\mathrm{ppl}_p(t_{1:n}) = \exp\left(\frac{1}{n}\sum_{i=1}^{n}\log\frac{1}{p(t_i|t_{1:i-1})}\right)
$$

- $\frac{1}{n}\sum \log \frac{1}{p}$: 平均 negative log-likelihood
- 外层 $\exp$: 把 nats 转回"等效词数"
- $t_{1:n}$: 完整序列

### 2.3 Annealing Stage 的 Contamination 敏感性分析

这一段推导很有意思。假设 clean distribution $p$ 混入了 contaminated distribution $r$，混合概率 $\epsilon$：

$$
q(t_i | t_{1:i-1}) = (1-\epsilon) p(t_i | t_{1:i-1}) + \epsilon r(t_i | t_{1:i-1}) \quad \text{(Eq. 4)}
$$

- $q$: 混合分布
- $\epsilon$: contamination 概率
- $p$: clean distribution
- $r$: contaminated distribution

由 $q \geq (1-\epsilon)p$ 推出：

$$
\mathrm{ppl}_q \leq \frac{1}{1-\epsilon} \mathrm{ppl}_p \approx (1+\epsilon)\mathrm{ppl}_p \quad \text{(Eq. 6)}
$$

**关键 insight**: 5% contamination 只让 ppl 上升 5%，看起来"无害"。但 generation 层面这意味着**平均每 20 个 token 就有一个 corrupted token**——一次生成就崩了。所以 annealing 阶段不能用 NLL 评估，必须用 few-shot downstream benchmark 直接评估 generation。

这是一个非常重要的 evaluation intuition：**NLL 的 smooth perturbation 不能捕捉 generation 的 catastrophic failure**。这与 Hinton "subjective" probability 在 token-level 平均后失去 long-range coherence 的现象本质同源。

### 2.4 Annealing 数据构成

- 40B tokens
- 保留原始 code 比例
- web page data 几乎全部删除，只保留 math/code 相关
- 加入 Dolmino + OpenCoder annealing + APPS + Code Contest（每个 problem 只保留一个 Python solution）
- Synthetic data 用 reward model rejection sampling（reasoning）+ sandbox execution（code）

**为什么 annealing 有效**：在 LR decay 阶段，model 进入"precision mode"，对 high-quality data 极度敏感。这与 Karpathy 你在 nanoGPT 里观察到的 "final 10% LR decay 决定 final quality" 一致。

---

## 3. Post-training (SFT) Pipeline

### 3.1 SFT 数据合成 (Figure 2 解析)

```
Seed instructions (Infinity-Instruct 等)
    │
    ├─ Instruction Evolution (LLM 扩展)
    │
    ├─ Qwen2.5-32B-Instruct 生成多 response
    │   （强制 "step by step" reasoning）
    │
    ├─ Rejection Sampling:
    │   - Reasoning/math → reward model 选最高分
    │   - Code → sandbox execution 验证
    │
    ├─ Domain labeling + diversity sampling
    │
    └─ Difficulty scoring (math)
```

### 3.2 关键 ablation: SFT 数据对 model size 的依赖

Paper Section 5.1.2 给出一个反直觉观察：**large model 上表现好的 SFT data，未必在小 model 上好**。小 model 需要百万级 samples 才能达到 competitive performance。这与 "data complexity threshold" 相关——小 model 的 capacity 不够吸收 high-complexity distribution，反而需要 more data 来 "smooth out" representation。

### 3.3 Mathematical Data Compression (Appendix B.2)

用 Llama3.3-70B-Instruct 给 ScaleQuest-Math 标 difficulty（very easy → very hard），然后分两组：

| Group | 组成 | 数量 | GSM8K | MATH |
|---|---|---|---|---|
| A | very easy + easy | 483K | 61.33% | 23.6% |
| B | medium + hard + very hard | 350K | 60.5% | 36.64% |

**Insight**: 仅用 35% 的 hard data，在 MATH 上提升 13%。**Difficulty-based sampling 比 uniform sampling 高效得多**。这与 DART-Math 的 "difficulty-aware rejection tuning" 思路一致 (https://arxiv.org/abs/2407.13690)。

---

## 4. Long CoT Enhancement (Appendix D)

受 o1 和 DeepSeek-R1 启发，InfiR 在 Instruct 模型基础上用 long CoT 数据继续 fine-tune。数据来源：NuminaMath-QwQ-CoT-5M。

| Model | AIME24 | MATH500 | AMC23 | GPQA | OlympiadBench |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 0.000 | 0.250 | 0.175 | 0.020 | 0.043 |
| Qwen2.5-1.5B-Instruct | 0.067 | 0.492 | 0.225 | 0.242 | 0.185 |
| DeepSeek-R1-Distill-Qwen-1.5B | 0.289 | 0.839 | 0.700 | 0.338 | 0.436 |
| InfiR-1B + 200K Long CoT | 0.033 | 0.474 | 0.225 | 0.288 | 0.181 |
| InfiR-1B + 2M Long CoT | 0.067 | 0.620 | 0.300 | 0.364 | 0.224 |

**Key observations**:
1. 200K → 2M 的 data scaling 带来 AIME24 翻倍、MATH500 +30%、GPQA +27%
2. 2M 版本在 GPQA 上 (0.364) 已经超过 Qwen2.5-1.5B-Instruct (0.242)
3. 但与 R1-Distill-Qwen-1.5B 仍有显著差距——RL-based reasoning distillation 仍优于纯 SFT on long CoT

这与 s1 (https://arxiv.org/abs/2501.19393) 的 "1K samples can unlock reasoning" 假设形成对比——InfiR 的证据表明 **在 1B scale，long CoT data scale 仍然重要**，可能因为小 model 的 in-context reasoning capacity 弱，需要更密集的训练 signal。

---

## 5. Multimodal Extension (InfiR-VL-1.6B)

### 5.1 Architecture (Figure 3)

```
┌─────────────────┐      ┌──────────┐      ┌─────────────────┐
│ SigLip-So400m   │ ───→ │  MLP     │ ───→ │ InfiR-1B-Base   │
│ (ViT backbone)  │      │ Projector│      │ (LLM backbone) │
└─────────────────┘      └──────────┘      └─────────────────┘
```

- Vision encoder: SigLip-So400m (SoViT-400M, https://arxiv.org/abs/2303.15377)
- Projector: 单层 MLP（最简形式，没有用 Q-Former 或 Resampler）
- LLM: InfiR-1B-Base（直接复用 pretrain 的成果）

### 5.2 Three-stage Training

| Stage | Trained Modules | Data | 目标 |
|---|---|---|---|
| Pretrain | MLP only | Caption data | Visual-textual alignment |
| SFT-1 | ViT + adapter | Text rendering + GUI | Vision reasoning foundation |
| SFT-2 | All params | Trajectory + tool-use data | Planning & reasoning |

**Curriculum Learning** 的关键：不能一开始就全参数训练，会导致 early overfitting 到 domain-specific data。先冻结 LLM，让 ViT 学到对齐的 visual features，再解锁全部参数学复杂 reasoning。

### 5.3 Operator-System Reasoning (GUI Agent)

InfiR-VL-1.6B 在 AndroidWorld 上达到 9.48% accuracy，比 Showui-2B (6.90%) 高 28%。

关键设计：
- 坐标系标准化到 $[0, 1000]$ scale
- Reference-augmented annotation format
- 45K synthesized trajectory data with structured reasoning patterns

这与 InfiGUIAgent (https://arxiv.org/abs/2501.04575) 的工作相关，强调 "native reasoning + reflection" 而非 pattern matching。

### 5.4 Multimodal Data Cleaning (Appendix E)

用 Vista model (https://arxiv.org/abs/2406.04292) 计算 image-text embedding similarity。在 COCO-caption 上 2500 sample 统计：

- < 5% 的 image-text pair similarity < 0.5
- Threshold 设为 0.5，过滤低质量 pair

---

## 6. Experimental Results 详解

### 6.1 Base Model (Table 1, few-shot)

| Model | MMLU | GSM8K | MATH | HumanEval | MBPP | MBPP(3-shot) |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | 32.74 | 8.11 | 3.42 | 17.68 | 33.46 | 24.8 |
| Qwen-2.5-1.5B | 63.03 | 66.57 | 31.24 | 35.37 | 58.37 | 41.4 |
| **InfiR-1B-Base** | **47.24** | **63.46** | **31.82** | **37.80** | **53.40** | **37.6** |

InfiR-1B-Base 与 Qwen2.5-1.5B 在 MATH、HumanEval 上几乎持平，但参数少 33%。相比 Llama-3.2-1B，reasoning 平均提升 2.26x。

### 6.2 Instruct Model (Table 2, zero-shot)

| Model | MMLU | GSM8K | MATH | HumanEval | MBPP |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 46.27 | 47.9 | 30.0 | 39.63 | 49.03 |
| Qwen-2.5-1.5B-Instruct | 61.78 | 74.3 | 53.4 | 51.83 | 56.81 |
| **InfiR-1B-Instruct** | **50.22** | **70.9** | **46.4** | **58.54** | **56.03** |

InfiR-1B-Instruct 相比 Llama-3.2-1B-Instruct：
- GSM8K: +23 points
- MATH: +16 points
- HumanEval: +19 points
- MBPP: +7 points

在 HumanEval 上甚至超过 Qwen2.5-1.5B-Instruct（58.54 vs 51.83）。

### 6.3 Multimodal (Table 3)

| Model | MMMU | ScreenSpot | AndroidWorld |
|---|---|---|---|
| Qwen2-VL-2B | 41.1 | 9.3 | - |
| Qwen2.5-VL-3B | 53.1 | 55.5 | - |
| Showui-2B | - | 75.1 | 6.90 |
| **InfiR-VL-1.6B** | 38.8 | **76.3** | **9.48** |

InfiR-VL-1.6B 在 ScreenSpot (76.3) 上甚至超过 Showui-2B (75.1)，在 AndroidWorld 上显著领先。

---

## 7. Training Details (Cost Breakdown)

- Total compute: **5760 GPU hours** = 64×H800 × 90 hours
- Pretrain: 900B tokens, 1 epoch, LR=1.4e-3, batch=2048, seq_len=4096
- Annealing: 40B tokens, 1 epoch
- SFT: 4 epochs, LR=2e-5, batch=128, cosine schedule, warmup=0.1
- Framework: NVIDIA NeMo (https://arxiv.org/abs/1909.09577) with DDP

对比 Qwen2.5-1.5B 的训练成本，InfiR 的 GPU hours 极其克制——这正是 "data efficiency via quality" 的胜利。

---

## 8. Key Insights 总结 (build intuition)

1. **Data > Scale at small regime**: 当 parameter 受限，data quality density 是第一性原理。Reasoning data 必须通过 model-based recall（fasttext）+ quality scoring 双重过滤。

2. **Annealing 是 reasoning 注入的关键窗口**: LR decay 阶段对 high-quality data 极度敏感，synthetic data 在这个阶段引入比 pretrain 主阶段更有效（避免 distribution gap 污染 main training）。

3. **NLL 不能反映 generation quality**: Eq. 6 的推导表明 ppl 的 linear perturbation 对应 generation 的 exponential degradation，evaluation 必须用 downstream task。

4. **Difficulty-aware data sampling >> uniform sampling**: 35% hard data > 100% mixed data on MATH。这呼应了 curriculum learning 的经典直觉。

5. **Multimodal curriculum**: 必须 ViT-only / ViT+adapter / all-params 三阶段，否则 overfitting 太早。

6. **Long CoT data scale 仍重要 for 1B**: 与 s1 的 "1000 samples 即可" 不同，InfiR 在 1B 上观察到 200K→2M 的持续提升。可能是因为 small model 的 in-context reasoning 弱，需要把 reasoning pattern 真正 compress 进 weights。

7. **SFT data 对 base model size 有强依赖**: large-model-tuned SFT data ≠ small-model-optimal SFT data。

---

## 9. 局限与 Open Questions

- **缺少 RL stage**: InfiR 是纯 SFT pipeline，没有 PPO/GRPO。与 DeepSeek-R1-Distill-Qwen-1.5B 的差距说明 RL on long CoT 仍 superior。
- **Generalization to real-world 未验证**: 仅 standard benchmarks，没有部署 side effect 评估。
- **MMMU 偏低** (38.8 vs Qwen2.5-VL-3B 53.1): 说明 general multimodal reasoning 仍受 1B LLM backbone 限制。
- **Code + Math 过度倾斜**: MMLU (general knowledge) 仍显著低于 Qwen2.5-1.5B-Instruct (50.22 vs 61.78)，trade-off 明显。

---

## 10. 与你（Karpathy）工作的潜在连接

- **nanoGPT 教学价值**: InfiR 的 annealing 实验正好验证了 nanoGPT 里 "final LR decay" 的重要性。
- **"Software 2.0" 视角**: InfiR-VL-1.6B 在 AndroidWorld 上的 GUI agent 能力，是 "Software 2.0" 替代 hand-coded GUI automation 的直接证据。
- **Data-efficient learning intuition**: 你的 "data quality > data quantity" 反复强调的 thesis，InfiR 是工业级佐证。
- **Tokenizer 数学**: Eq. 1-3 的推导很适合作为 LLM1010 课程的 evaluation 章节。

---

## Reference Links

- InfiR repo: https://github.com/Reallm-Labs/InfiR
- DeepSeekMath: https://arxiv.org/abs/2402.03300
- OpenCoder: https://arxiv.org/abs/2411.04905
- FineWeb: https://arxiv.org/abs/2406.17557
- s1: https://arxiv.org/abs/2501.19393
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- InfiGUIAgent: https://arxiv.org/abs/2501.04575
- DART-Math: https://arxiv.org/abs/2407.13690
- SigLIP: https://arxiv.org/abs/2303.15377
- Vista: https://arxiv.org/abs/2406.04292
- InfiMM-WebMath: https://arxiv.org/abs/2409.12568
- OpenWebMath: https://arxiv.org/abs/2310.06786
- NVIDIA NeMo: https://arxiv.org/abs/1909.09577

---

要不要我深入展开某一个 section？比如 annealing 阶段的 LR schedule 实验细节，或者 InfiR-VL 的 trajectory data synthesis 具体方法？
