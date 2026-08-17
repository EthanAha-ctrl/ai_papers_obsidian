---
source_pdf: Incompressible Knowledge Probes Estimating Black-Box.pdf
paper_sha256: 3ad2b78cb2b7b4e58c6253815ad002146f39e38fc71b33750f7f100f3b071421
processed_at: '2026-08-05T09:28:00-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话版本

Closed-source labs 不告诉你参数量。传统办法是看 API throughput 和 hardware cost 反推，误差 2×+。这篇 paper 的 trick：**既然存 facts 需要参数，那测量一个 model 知道多少冷知识，就能 lower-bound 它有多少参数**。而且这种"factual capacity"不受 Densing Law 影响——architecture 改进让 procedural 能力 compress，facts 不能 compress。

## 核心直觉

把 model 的 N 个参数想象成三块：

$$N = N_{\text{fact}} + N_{\text{proc}} + N_{\text{ling}}$$

- $N_{\text{fact}}$：存具体 facts（USTC Hackergame 2014 年开始，某 researcher 做 distributed systems）
- $N_{\text{proc}}$：procedural skills（reasoning, parsing, tool use）
- $N_{\text{ling}}$：linguistic competence（syntax, vocabulary）

Densing Law [Huang et al., 2025](https://www.nature.com/natmachintell/) 说 capability per parameter 每 3.5 个月翻倍——2026 年的 7B 在 MMLU 上匹配 2023 年的 70B。这确实发生了，因为 $N_{\text{proc}}$ 和 $N_{\text{ling}}$ 可以 compress：更好的 architecture 和 training recipe 把更多 procedural 能力塞进更少参数。

但 $N_{\text{fact}}$ 不行。"USTC Hackergame started in 2014"这种 fact 无法 derive、compute 或 infer——必须 explicitly store。Shannon entropy [Shannon, 1948](https://ieeexplore.ieee.org/document/6773024) 给硬下界：存这个 fact 需要约 $\log_2 |V|$ bits，architecture 改不动这个数。

[Allen-Zhu and Li, 2025](https://arxiv.org/abs/2404.05568) 证明 transformer 大约 2 bits/param 的 factual storage capacity（[Morris et al., 2025](https://arxiv.org/abs/2505.24832) refine 到 3.6）。[Kandpal et al., 2023](https://arxiv.org/abs/2211.08411) 证明 rare fact accuracy log-linear with model size。把这两个结合起来：**测 accuracy → 估 facts stored → lower-bound 参数量**。

## 为什么 log-linear

Web knowledge 服从 [Zipf's law](https://www.worldcat.org/title/476842365)：少数 facts 被海量 mention，大量 facts 被很少 mention。容量更大的 model 吃更靠 long tail 的 facts。

设 frequency cutoff $f^*$：mention frequency 高于 $f^*$ 的 facts 被 memorize，低于的不被。Power-law distribution 下，cutoff 以上的 facts 数量随 $\log f^*$ 增长。所以 aggregate accuracy：

$$A = \alpha \cdot \log_{10}(N) + \beta$$

- $A$：aggregate penalized accuracy
- $N$：parameter count in billions
- $\alpha$：slope（每 10× 参数增加多少 accuracy）
- $\beta$：intercept

Per-tier accuracy 服从 shifted sigmoid：

$$T_i(N) = \frac{L_i}{1 + \exp(-k_i \cdot (\log N - m_i))}$$

- $T_i(N)$：tier $i$ 在参数量 $N$ 下的 accuracy
- $L_i$：tier $i$ 的 saturation ceiling
- $k_i$：slope steepness
- $m_i$：midpoint parameter count（50% accuracy 对应的 $\log N$）

七个 shifted sigmoids 的 mean 在 wide range of $\log N$ 内近似一条直线——这就解释了为什么 aggregate accuracy 和 $\log N$ 几乎完美 log-linear，$R^2 = 0.917$ across 89 models spanning 4 orders of magnitude (135M to 1.6T)。

## 怎么造 probes

1,400 个 probes 分 7 个 tier（T1-T7），每 tier 200 个。

### Tier 校准

用 6 个 landmark models 经验定义 tier 边界：

| Tier | Landmark | Params |
|------|----------|--------|
| T1 | Qwen 2.5 0.5B | 0.5B |
| T2 | Qwen 2.5 7B | 7.6B |
| T3 | Qwen 3 32B | 32B |
| T4 | Qwen 3 235B | 235B |
| T5 | Kimi K2.5 | 1T |
| T6 | Gemini 3.1 Pro | Frontier |

一个 probe 被分配到 tier $k$ 当且仅当 $T_k$ landmark 答对但 $T_{k-1}$ landmark 答错。Non-monotonic correctness 的 probes 丢弃（15%）。

### 两个 phase

**Phase A**：用 GPT-5 生成 candidates。但发现 82% 落在 T1-T2——**LLM 无法可靠生成超出自身 knowledge horizon 的 probes**。这是循环的：generator 只能产生自己知道的事实。

**Phase B**：从外部 corpora 采样。
- [Wikidata](https://www.wikidata.org/) SPARQL：557 probes，founding years, capitals, directors 等，按 Wikipedia view-count stratify
- [DBLP](https://dblp.org/)/arXiv researcher records：345 probes，CS researcher 的 primary subfield + verifiable artifact

Researcher probe 的 two-part format 是 load-bearing 设计：

> "In computer science, what is the research subfield of [Name], and name one paper, system, institution, or co-author associated with their work? If you don't know who this person is, say so."

- "In computer science" 减少 cross-field name collision
- artifact requirement 强制 model 产出 verifiable token，而非 emit plausible 但 unattested 的 subfield label

没有 artifact requirement，model 可以猜 "computer architecture"（热门 subfield）骗分。加上 requirement 后必须 name TMO, EbbRT, ASPLOS, ISCA, Boston University 等 verifiable evidence。

## Scoring

### 4-way evidence-aware judge

Judge 是 Gemini 3 Flash Preview，temperature=0。Researcher probes 用 4-way：

- **CORRECT_STRONG** (+1.0)：right subfield + cites matching evidence
- **CORRECT_WEAK** (+0.5)：right subfield but no specific evidence
- **REFUSAL** (0)：说 "I don't know"
- **WRONG** ($\lambda = -1.0$)：subfield outside {primary, secondary} 或 fabricated specifics

关键设计：**strict ordering STRONG > WEAK > REFUSAL > WRONG** 使 confident bluffing 严格 worse than refusing。一个 honest model 在相同 knowledge level 下 beat aggressive bluffer。

### 为什么 $\lambda = -1.0$

Appendix B sweep $\lambda \in \{0, -0.25, -0.5, -1.0, -1.5, -2.0, -3.0\}$：

| $\lambda$ | $R^2$ | LOO× | ≤3× | PI 90× |
|----------|-------|------|-----|--------|
| 0.00 | 0.907 | 1.61 | 85.4% | 3.21 |
| -0.25 | 0.919 | 1.54 | 89.9% | 2.96 |
| -0.50 | 0.920 | 1.54 | 88.8% | 2.94 |
| **-1.00** | **0.917** | 1.59 | 87.6% | 3.00 |
| -1.50 | 0.907 | 1.64 | 87.6% | 3.21 |
| -2.00 | 0.901 | 1.69 | 88.8% | 3.35 |

$R^2$ 在 $[-0.25, -1.0]$ 内 essentially flat。$\lambda = -1.0$ 提供 integer-symmetric scoring：WRONG (-1.0) 与 STRONG (+1.0) 关于 REFUSAL (0) 对称。

## 反推 proprietary models

在 89 个 open-weight models 上 OLS fit $A = \alpha \log_{10}(N) + \beta$，$R^2 = 0.917$，slope $\alpha = 0.147$（每 10× 参数增加 14.7 pp）。

反演：

$$\hat{N} = 10^{(A - \beta)/\alpha}$$

Leave-one-out CV：median fold error 1.59×，68.5% within 2×，87.6% within 3×，90% PI factor 3.00×。

几个 striking 估计：

| Model | Accuracy | Est. Size | 90% PI |
|-------|----------|-----------|--------|
| GPT-5.5 | 71.9% | ~9.7T | [3.2-28.7T] |
| Claude Opus 4.6 | 68.0% | ~5.3T | [1.8-15.6T] |
| GPT-5 | 66.4% | ~4.1T | [1.4-12.1T] |
| Claude Opus 4.7 | 66.4% | ~4.0T | [1.4-12.0T] |
| Grok-4 | 64.8% | ~3.2T | [1.1-9.4T] |
| GPT-4 | 54.8% | ~666B | [223B-2.0T] |
| GPT-5 Nano | 40.5% | ~71B | [24B-212B] |

Top-to-bottom ratio = 150×，几乎恰好等于 calibration set 最大（1.6T）与最小（135M）的 ratio。

## MoE: total > active

最 clean 的 mechanistic finding：

| Subset | $R^2$ (pen.) |
|--------|--------------|
| MoE (total params) | **0.792** |
| MoE (active params) | **0.513** |

**Total parameters 远好于 active parameters 预测 factual knowledge**。这证明 factual storage 分布在所有 expert weights 上，不局限于 per-token activated 的 experts。

Intuition：MoE 的 routing 选择哪个 expert 处理哪个 token，但 factual associations 在所有 experts 中 distributed stored——一个 fact 在训练时由多个 expert 看到。Active parameters 衡量 inference-time compute，不衡量 storage capacity。这和 [Ludziejewski et al., 2025](https://arxiv.org/abs/2404.02236) 的 joint MoE scaling laws 一致。

## Falsify Densing Law

这是 paper 最 striking 的 conceptual contribution。

对 96 个 open-weight models（2023-09 到 2026-04）fit：

$$\text{pen\_acc} = \beta_0 + \beta_1 \log_{10}(N_B) + \beta_2 \cdot \text{months}$$

- $\beta_0$：intercept
- $\beta_1$：parameter scaling slope
- $\beta_2$：time coefficient (per month)
- $N_B$：parameter count in billions
- `months`：release date 相对于 2024-01-01 的月份

Baseline $\beta_1 = 0.136$。Densing Law 预测 monthly accuracy gain：

$$\beta_2^{\text{Densing}} = \beta_1 \cdot \log_{10}(2) / 3.5 \approx +0.0117/\text{month} \approx +14.1 \text{ pp/year}$$

实际拟合：

$$\hat{\beta}_2 = -0.0010/\text{month} \quad [95\% \text{ CI}: -0.0031, +0.0008]$$

- 与零 statistically indistinguishable（$p = 0.34$）
- 拒绝 Densing prediction at $p < 10^{-15}$
- 加入 release date 后 $R^2$ 只增加 +0.0024

**Factual capacity 不 compress**。相同 size 的 newer model 不比 older model 知道更多——正是 incompressibility 的预测。

对比 standard benchmarks：

| Metric | $R^2$ vs $\log_{10} N$ | Time slope (pp/month) |
|--------|----------------------|----------------------|
| IKP | 0.917 | -0.06 |
| SimpleQA | 0.991 | +0.03 |
| MMLU | 0.886 | +0.58 |
| MMLU-Pro | 0.900 | +0.82 |
| GPQA Diamond | 0.903 | +1.99 |

Reasoning-heavy benchmarks（GPQA Diamond +2 pp/month）在 fixed $\log_{10} N$ 上随 time 急剧漂移——33B model 一年内提升 24 points without growing。这直接 invalidate 它们作为 parameter proxy。Purely-factual benchmarks（SimpleQA, IKP）漂移 near-zero。

## Per-tier discrimination

| Tier | Slope (per decade) | Saturation | Role |
|------|-------------------|------------|------|
| T1 | 0.124 | @ 70B | binary check |
| T3 | **0.324** (steepest) | @ 1.2T | **most informative**, $\rho = 0.983$ vs overall |
| T6 | 0.002 | essentially flat | **proprietary separates from open-weight here** |
| T7 | $\approx 0$ | — | hard ceiling |

T3 单独就是 overall ranking 的强 proxy。T6 是 frontier 与 open-weight 的分水岭——DeepSeek V4 Pro (1.6T) 在 T6 只得 1.0%，但 GPT-5.5 得 38.5%。T7 是 universal ceiling：188 个 models 中只有 Jamba-large (2.8%) 和 Grok-4 (1.0%) 超过 1%。

## Knowledge fingerprinting

不依赖 model weights，纯靠 black-box API 判断两个 model 是 shared base, lineage 还是 full retrain。

核心 metric 是 **Hallucination Similarity Score (HSS)**：两个 model 都 wrong 的 probes 上，给相同 normalized wrong answer 的比例。

Independent models 几乎不会 converge 到 identical wrong rare fact。Weight-sharing siblings 在 30-55% of shared-wrong probes 上这么做。

| Regime | HSS | Interpretation |
|--------|-----|----------------|
| Shared base | $\geq 0.30$ | same weights, different inference/alignment |
| Lineage | $[0.10, 0.30)$ | post-training, continued pretraining, 或 distillation |
| Retrained | $< 0.10$ | statistically indistinguishable from cross-vendor independent |

关键发现：**GPT-5 到 5.1 到 5.2 到 5.3 到 5.4 每个点 release 都是 full retrain**（HSS ≤ 0.08），只有 GPT-5 / 5-pro / 5-think 是 shared base（HSS = 0.51, 0.53）。Claude Opus 4→4.1 是 lineage，4.6→4.7 是 retrain。每个跨 generation Gemini pair 都是 retrain。

## 什么决定 model 知道某 researcher

Bibliometric signals explain 1/3 of variance：
- log(citations) vs recognition: Spearman $\rho = 0.575$
- log(h-index) vs recognition: $\rho = 0.561$

但 **citation 是 approximately necessary but not sufficient**：< 500 citations 没人超过 75% recognition，但很多 10K+ citations 的人在 25% 以下。

Exceptions（低 citation 高 recognition）无一例外 attached to named artifact：Tri Dao (FlashAttention, 3K citations, 100%), Eyuboglu (ColBERT, 267 citations, 57%), Psaras (IPFS, 318 citations, 69%)。

Subfield 差异巨大：IR researchers recognition 0.571，computer architecture 只有 0.179——尽管 computer architecture 平均 citations 更高。**Top subfields 恰好是 derivative content 最密集的**（tutorials, blog posts, course materials）。

Synthesis：LLM knowledge **由 effective mention frequency 决定**——training corpus 中 state specific target fact in retrievable form 并 attribute to specific name 的 document 数量。Citation 是 partial proxy，pageviews, practitioner adoption, named-artifact count 是 better proxies。

**对 researcher 的 practical corollary**：marginal effect of one widely-used open-source tool with clean name attribution > marginal effect of one well-cited paper。Tool generates derivative documents 每个都 carry author name，paper 只在 own bibliography 中 carry。

## Safety tuning 的 silent tax

Heavily safety-tuned models 拒绝 demonstrably knows 的 probes。Claude Sonnet 4 (48.2%) vs Claude 3.7 Sonnet (54.9%)——6.7 pp gap largely 是 refusal policy artifact。On T5，Sonnet 4 refuses 88%，Sonnet 3.7 refuses 54%。

Hallucination rate 是 vendor signature：
- Google smaller Gemma: 89-97%（几乎从不 refuse）
- Anthropic Claude: 平均 11%
- OpenAI GPT-4.1: 53-72%，GPT-5 nano/mini: 3-4%

这使得 IKP estimates for heavily safety-tuned models 应该 read as **lower bounds**。

## Case study: USTC Hackergame

Author 三年来用同一 prompt 压测新 model："What do you know about USTC Hackergame?"

- **GPT-4o (2024-05)**：知道 contest 存在但 invented fake challenge names。Nine different years 都返回 identical fabricated list——deterministic fabrication
- **Claude 3.7 Sonnet (2025-02)**：第一个 list real per-year challenges，19 verified 2023 titles
- **DeepSeek V4 Pro (1.6T, 2026-04)**：正确说 2014 start year（唯一做到）但 fabricates per-year challenges——meta-fact 与 long-tail content stored independently
- **Claude Opus 4.7**：说 "I don't have reliable detailed memory and don't want to fabricate specifics"

Transition sharp and dateable：9-month window 内 Hackergame writeups 进入 training，不是逐渐 across field。

## 核心 takeaway

对 Andrej 来说，这 paper 的 intuition-building 价值在于：**给"factual knowledge is fundamentally different from procedural capability"一个 quantitative operational handle**。

Benchmark saturation 被 read as "scaling ended"。这 paper 说这 reading 混淆了两件 fundamentally different 的事：
1. Procedural capability compresses under Densing Law → benchmarks saturate
2. Factual capacity does not compress → continues log-linear scaling

这两件事 independent，都 important。Mix up 它们是 category error。如果继续 scale，factual capacity 继续 log-linear 增长；如果 architecture 改进，procedural density 继续 densify。测量只测 incompressible 的部分，你看到 scaling 没有停。

---

### Key references

- [Paper companion site](https://01.me/research/ikp)
- [Code](https://github.com/19PINE-AI/ikp)
- [Allen-Zhu and Li, 2025](https://arxiv.org/abs/2404.05568) - 2 bits/param capacity
- [Morris et al., 2025](https://arxiv.org/abs/2505.24832) - 3.6 bits/param, grokking
- [Kandpal et al., 2023](https://arxiv.org/abs/2211.08411) - rare fact log-linear scaling
- [Lu et al., 2024](https://aclanthology.org/2024.findings-emnlp/) - fact memorization scaling laws
- [Huang et al., 2025 - Densing Law](https://www.nature.com/natmachintell/)
- [Geva et al., 2021](https://aclanthology.org/2021.emnlp-main.446/) - FFN as key-value memories
- [Meng et al., 2022 - ROME](https://arxiv.org/abs/2202.05262) - editing factual associations
- [Dai et al., 2022](https://aclanthology.org/2022.acl-long.396/) - knowledge neurons
- [Shannon, 1948](https://ieeexplore.ieee.org/document/6773024) - information theory
- [Zipf, 1949](https://www.worldcat.org/title/476842365) - power-law frequency
- [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) - scaling laws
- [Hoffmann et al., 2022 - Chinchilla](https://arxiv.org/abs/2203.15556)
- [Ludziejewski et al., 2025](https://arxiv.org/abs/2404.02236) - MoE scaling laws
- [Hong et al., 2025](https://arxiv.org/) - parameter specialization
- [USTC Hackergame writeups](https://github.com/USTC-Hackergame)

---

# Incompressible Knowledge Probes: 从 Factual Capacity 反推 LLM 参数量

这篇 paper 的核心 insight 相当 elegant：与其从 inference economics（API throughput, hardware cost）反推 black-box model 的参数量，不如直接测量 model **知道多少不可压缩的 facts**，再利用 scaling law 反推参数下界。这个思路比传统的 inference economics 更 intrinsic，因为它不依赖 hardware generation, batching, quantization 这些 external to model 的因素。

## 1. 核心理论框架：Incompressibility

### 1.1 参数的三重分解

Author 把 model 的 N 个参数分解为三个 functional roles：

$$N = N_{\text{fact}} + N_{\text{proc}} + N_{\text{ling}}$$

- $N_{\text{fact}}$：存储具体 factual associations（entity attributes, dates, names）的参数份额
- $N_{\text{proc}}$：procedural skills（reasoning, parsing, instruction following, tool use）的参数份额
- $N_{\text{ling}}$：linguistic competence（syntax, morphology, vocabulary, register）的参数份额

**Densing Law** [Huang et al., 2025] 提升 $N_{\text{proc}}$ 和 $N_{\text{ling}}$ 的 efficiency（capability per parameter 每 3.5 个月翻倍），让 2026 年的 7B model 在 MMLU 上能匹配 2023 年的 70B model。但 $N_{\text{fact}}$ 受 Shannon entropy 硬下界约束：存储 fact $(e, a, v)$ 需要 $\sim \log_2 |V_a|$ bits，无法通过 architecture 改进压缩。

### 1.2 Capacity Bound

从 [Allen-Zhu and Li, 2025]：N 个参数最多存储 $\sim 2N$ bits 的 factual knowledge（理想 training 下，经验值 2-4 bits/param，[Morris et al., 2025] refine 到 3.6 bits/param）。

从 [Kandpal et al., 2023]：observed accuracy $= a \cdot \log(N) + b$，其中 N 是 parameter count。这是 BLOOM family 内 $R^2 = 0.98$ 的实证结果。

[Lu et al., 2024] 进一步证明：memorize 所有 Wikidata facts 需要 1000B params × 100 epochs。

### 1.3 频率-容量关系

Web knowledge 服从 Zipf's law（power-law frequency distribution）。容量更大的 model 会 memorize 更靠 long tail 的 facts，产生一个 natural frequency cutoff $f^*$：高于 $f^*$ 的 facts 被 memorize，低于的不被 memorize。由于 power-law tail 上 facts 数量随 $f^*$ 对数增长：

$$\text{acc} \propto \log(N) \quad \Rightarrow \quad A = \alpha \cdot \log_{10}(N) + \beta$$

这就是 paper 观察到的 log-linear relationship。

Per-tier accuracy 服从 shifted sigmoid：

$$T_i(N) = \frac{L_i}{1 + \exp(-k_i \cdot (\log N - m_i))}$$

- $T_i(N)$：tier $i$ 在参数量 N 下的 accuracy
- $L_i$：tier $i$ 的 saturation ceiling
- $k_i$：tier $i$ 的 slope steepness
- $m_i$：tier $i$ 的 midpoint parameter count（即 50% accuracy 对应的 log N）

七个 shifted sigmoids 的 mean 在 $\log N$ 的 wide range 内近似一条直线，这就解释了为什么 aggregate accuracy 和 $\log N$ 几乎完美 log-linear。

## 2. Probe 生成 Pipeline

### 2.1 七个 Tier 的定义

Tiers 通过 landmark models 经验校准（而非 proxy metrics）：

| Tier | Landmark Model | Params | Tier Boundary |
|------|---------------|--------|---------------|
| T1 | Qwen 2.5 0.5B | 0.5B | T1/T2 |
| T2 | Qwen 2.5 7B | 7.6B | T2/T3 |
| T3 | Qwen 3 32B | 32B | T3/T4 |
| T4 | Qwen 3 235B | 235B | T4/T5 |
| T5 | Kimi K2.5 | 1T | T5/T6 |
| T6 | Gemini 3.1 Pro | Frontier | T6/T7 |

一个 probe 被分配到 tier $k$ 当且仅当 $T_k$ landmark 答对但 $T_{k-1}$ landmark 答错。Non-monotonic correctness（小 model 答对大 model 答错）的 probes 被丢弃（15%）。

### 2.2 两阶段生成

**Phase A**: LLM-generated（GPT-5 作为 generator）
- 目标 T1-T2 saturation，T3-T4 supplementary
- 旋转 region（8 个）和 domain（6 个）保证 coverage
- 401 个最终 probes（T1:166, T2:152, T3:51, T4:32）
- 关键发现：82% LLM-generated candidates 落在 T1-T2，**LLM 无法可靠生成超出自身 knowledge horizon 的 probes**——这是 Phase B 存在的根本原因

**Phase B**: Corpus-grounded probes
- Wikidata SPARQL（557 probes）：universities, journals, museums, bridges, sports clubs, geographic places，按 Wikipedia view-count quartiles stratify 到 T3-T7
- DBLP/arXiv researcher records（345 probes）：CS researcher 的 primary subfield + verifiable artifact，按 citation-count buckets 分配到 T5-T7

Researcher probe 的 two-part format 是 load-bearing 设计：

> "In computer science, what is the research subfield of [Name], and name one paper, system, institution, or co-author associated with their work? If you don't know who this person is, say so."

- "In computer science" 减少 cross-field name collision
- artifact requirement 强制 model 产出 verifiable token，而非 emit plausible 但 unattested 的 subfield label
- 没有 artifact requirement，model 可以猜 "computer architecture"（热门 subfield）骗分；加上 requirement 后必须 name TMO, EbbRT, ASPLOS, ISCA, Boston University 等 verifiable evidence

### 2.3 Quality Filters

四个 filters：
1. **Computable knowledge filter**：丢弃可由 rule 推导的 probes（IUPAC 命名, 年份算术, 字母排序）
2. **Monotonicity filter**：landmark ladder 上 accuracy 必须单调非递减
3. **Name-collision filter**：丢弃双字中文名和单字母 given name（"Liu Yang" 可能是 50 citations 也可能是 5000 citations 的两个人）
4. **Contamination filter**：排除 ML/AI subfield researcher（自己领域的工作 disproportionately 出现在 training corpus，inflates accuracy）

### 2.4 Per-researcher Evidence Bundle

每个 researcher 携带结构化 gold record：
- `primary_subfield`：17 个 CS subfield label 之一
- `secondary_subfields`：0-3 个 adjacent labels
- `affiliations`：top 1-2 机构
- `named_systems`：regex 从 top 10 most-cited works 提取的 capitalized artifact tokens（"TMO", "Zoltan", "Wukong"）
- `venues`：top 5-6 publication venues
- `co_authors`：top 5 collaborators
- `top_works`：top 5 paper titles

## 3. Scoring 与 Judge

### 3.1 4-way Evidence-aware Judge

Judge 是 Gemini 3 Flash Preview，temperature=0。Researcher probes 用 4-way judge：
- **CORRECT_STRONG**（+1.0）：right subfield + cites at least one matching evidence item
- **CORRECT_WEAK**（+0.5）：right subfield but no specific evidence，或 correct evidence with adjacent-but-not-listed subfield
- **REFUSAL**（0）：说"I don't know"或表达 uncertainty
- **WRONG**（$\lambda = -1.0$）：subfield outside {primary, secondary} 或 fabricated specifics

关键设计：**strict ordering STRONG > WEAK > REFUSAL > WRONG** 使 confident bluffing 严格 worse than refusing。一个 calibrated honest model 在相同 knowledge level 下 beat aggressive bluffer。

### 3.2 Aggregate Accuracy

Per-tier score = mean per-probe score in that tier。Overall accuracy = unweighted mean of 7 个 per-tier scores。**Per-tier scores 不 floored at zero**——strongly-bluffing model 在 T6/T7 可以 negative，保留 bluff signal 用于 calibration。

### 3.3 Hallucination Penalty $\lambda$ 的选择

Appendix B sweep $\lambda \in \{0, -0.25, -0.5, -1.0, -1.5, -2.0, -3.0\}$：

| $\lambda$ | $R^2$ | RMSE | LOO× | ≤2× | ≤3× | PI 90× |
|----------|-------|------|------|-----|-----|--------|
| 0.00 | 0.907 | 0.0454 | 1.61 | 70.8% | 85.4% | 3.21 |
| -0.25 | 0.919 | 0.0428 | 1.54 | 70.8% | 89.9% | 2.96 |
| -0.50 | 0.920 | 0.0421 | 1.54 | 70.8% | 88.8% | 2.94 |
| **-1.00** | **0.917** | 0.0422 | 1.59 | 68.5% | 87.6% | 3.00 |
| -1.50 | 0.907 | 0.0436 | 1.64 | 64.0% | 87.6% | 3.21 |
| -2.00 | 0.901 | 0.0438 | 1.69 | 66.3% | 88.8% | 3.35 |
| -3.00 | 0.889 | 0.0445 | 1.64 | 64.0% | 83.1% | 3.63 |

$R^2$ 在 $\lambda \in [-0.25, -1.0]$ 内 essentially flat。$\lambda = 0$ 损失 0.013 $R^2$ 并 widen PI。$\lambda = -1.0$ 提供 integer-symmetric scoring：WRONG (-1.0) 与 STRONG (+1.0) 关于 REFUSAL (0) 对称，且满足 4-way judge 的 strict ordering 要求 $|\lambda| \geq 0.5$。

## 4. Calibration Curve

### 4.1 Log-linear Fit

在 89 个 open-weight models（135M-1600B，19 vendors）上 OLS fit：

$$A = \alpha \cdot \log_{10}(N) + \beta$$

- $A$：aggregate penalized accuracy
- $N$：parameter count in billions
- $\alpha = 0.147$（slope，每 $\log_{10}$ 增加 14.7 pp）
- $\beta$：intercept

Forward $R^2 = 0.917$，等于 squared Pearson correlation，对 inversion invariant。

反演估计参数量：

$$\hat{N} = 10^{(A - \beta) / \alpha}$$

### 4.2 LOO-CV 验证

Leave-one-out cross-validation：对每个 open model，用其余 88 个 refit regression 并预测 hold-out model 的 parameter count。

- Median fold error：1.59×
- 68.5% within 2×
- 87.6% within 3×
- 90% prediction interval factor：3.00×
- Residual standard error：0.042 in $\log_{10}$ space → PI half-width = 0.478 in $\log_{10}$ space

Worst outliers：
- **Nemotron-70B**：predicted 490B vs actual 70B，ECR 7.0×（NVIDIA heavy RLHF post-training）
- **Llama 3.1/3.3-70B**：ECR 3.5-4.2×（distillation or denser training data）
- **Llama 4 Scout**：predicted 18B vs actual 109B，ECR 0.17×（heavy refusal calibration）

### 4.3 MoE Total vs Active Parameters

这是 paper 的一个 striking 发现：

| Subset | n | Slope | $R^2$ (pen.) | $R^2$ (raw) |
|--------|---|-------|--------------|-------------|
| All open models | 89 | 0.147 | 0.917 | 0.909 |
| Dense only | 52 | 0.151 | 0.880 | 0.881 |
| Dense non-thinking | 42 | 0.158 | 0.894 | 0.884 |
| **MoE (total params)** | **37** | **0.155** | **0.792** | **0.683** |
| MoE (active params) | 37 | 0.157 | 0.513 | 0.419 |

**Total parameters $R^2 = 0.79$ 远好于 active parameters $R^2 = 0.51$**。这证明 factual knowledge 分布在所有 expert weights 上，不是 localized 到 per-token activated 的 experts。这和 [Ludziejewski et al., 2025] 的 joint MoE scaling laws 一致：total capacity matters for knowledge storage。

Intuition：MoE 的 routing 选择哪个 expert 处理哪个 token，但 **factual associations 在所有 experts 中都有存储**——一个 fact 在训练时由多个 expert 看到，被 distributed stored。Active parameters 衡量的是 inference-time compute，不是 storage capacity。

## 5. Frontier Model 估计

Table 2 给出 92 个 proprietary models 的 effective capacity 估计。几个 striking 发现：

### 5.1 四个 Band

1. **Top band**：GPT-5.5 ~9.7T，比 next model (Claude Opus 4.6 ~5.3T) 高 1.4×
2. **Frontier cluster** 3-4T：GPT-5, Claude Opus 4.7, o1, Grok-4, o3——半年竞争收敛到 1.4× effective capacity 内
3. **Mid band** 2-3T：GPT-4.1, Grok-3, GPT-5.4 Pro, 旧 Claude Opus
4. **Second-tier flagship** 1-1.7T：Claude Sonnet 4.6, Gemini 2.5 Pro, GPT-5.3, GPT-5.1/5.2
5. **Small/efficient line**：GPT-5 Mini ~410B, Gemini 2.5 Flash ~207B, GPT-5 Nano ~71B, Claude Haiku 4.5 ~65B

Top-to-bottom ratio = 150×，几乎恰好等于 calibration set 中最大（DeepSeek V4 Pro 1.6T）与最小（smollm2-135m）的 ratio。

### 5.2 Pro Tier 几乎不增加 factual capacity

OpenAI Pro variants（GPT-5 Pro, GPT-5.2 Pro, GPT-5.4 Pro, GPT-5.5 Pro）比 non-Pro siblings 只高 +0.3, +0.9, +4.8, +1.0 pp。反演后 GPT-5 Pro premium = 1.05×, GPT-5.5 Pro premium = 1.13×。

Pro tier 的目的是 reasoning, agentic, long-context，通过更强 post-training 和更长 inference-time budget 实现——这些都是 $N_{\text{proc}}$ 层面，不增加 stored facts。IKP 专门测 $N_{\text{fact}}$，所以 Pro-vs-base gap near-flat 是 expected outcome，不矛盾 Pro tier 的 purpose。

### 5.3 GPT-5.x 的 Iteration Pattern

Non-thinking IKP scores 在 GPT-5.x series 中 cluster tightly：GPT-5 (66.1%), 5.1 (59.3%), 5.2 (58.9%), 5.3 (60.0%), 5.4 (57.7%)——span 8 pp。然后 GPT-5.5 jump 到 71.4%。

这暗示 5-5.4 releases 共享类似 parameter budget 但 train on 不同 data mixtures，GPT-5.5 是 genuine scale-up。Fingerprint analysis 证实：每个 GPT-5.x→5.(x+1) transition 都在 retrained regime (HSS < 0.10, Jaccard collapse)，point releases 是 full retrains on different data 而非 continued pretraining。

## 6. Densing Law Falsification

这是 paper 最 striking 的 conceptual contribution。

### 6.1 Regression Design

对 96 个 open-weight models（2023-09-27 到 2026-04-24）fit：

$$\text{pen\_acc} = \beta_0 + \beta_1 \log_{10}(N_B) + \beta_2 \cdot \text{months}$$

- $\beta_0$：intercept
- $\beta_1$：parameter scaling slope
- $\beta_2$：time coefficient（per month）
- $N_B$：parameter count in billions
- `months`：release date 相对于 2024-01-01 的月份

Baseline $\beta_1 = 0.136$（$R^2 = 0.761$）。Densing Law 预测：capability per parameter 每 3.5 个月翻倍，对应 monthly accuracy gain：

$$\beta_2^{\text{Densing}} = \beta_1 \cdot \log_{10}(2) / 3.5 \approx +0.0117/\text{month} \approx +14.1 \text{ pp/year}$$

### 6.2 结果

Fitted coefficient：

$$\hat{\beta}_2 = -0.0010/\text{month} \quad [95\% \text{ bootstrap CI}: -0.0031, +0.0008]$$

- Point estimate 与零 statistically indistinguishable（$p = 0.34$）
- 拒绝 Densing prediction at $p < 10^{-15}$
- 加入 release date 后 $R^2$ 只增加 +0.0024
- Bootstrap CI 上界 +0.00079 与 Densing target +0.0117 之间 gap 超过 11× bootstrap standard error

Robustness checks：
- Excluding DeepSeek-R1 distilled（n=79）：$\hat{\beta}_2 = -0.00051$/month
- Excluding 5 landmarks（n=76）：$\hat{\beta}_2 = -0.00018$/month
- 用 $\log_{10}(\text{active\_B})$ 替换 total params：$\hat{\beta}_2 = +0.0063$/month（nominally significant 但仍只 half of Densing target，且 $p \approx 10^{-4}$ 拒绝 Densing）——这是因为 MoE models skew newer，active params systematically understates $N_{\text{fact}}$ for MoEs

### 6.3 Interpretation

Capability density 在 reasoning benchmarks 上确实 2024-2026 增长，但增长来自 $N_{\text{proc}}$ 和 $N_{\text{ling}}$ 的更高效使用，**不是 factual storage 的更高效**。相同 size 的 newer model 不比 older model 知道更多——这正是 incompressibility 的预测。

Time-only model $M_t$ 的 $\beta_2 = +0.0062$（$p < 0.01$）看似有 time effect，但 $R^2 = 0.10$ 一旦加入 $\log_{10} N$ 就 collapse——这个 apparent time correlation 是 selection effect（newer releases skew larger）。

### 6.4 vs Standard Benchmarks

Table 3 比较 IKP 和 standard benchmarks 作为 parameter proxy：

| Metric | N | $R^2$ vs $\log_{10} N$ IKP | $R^2$ (same subset) | Time slope (pp/month) |
|--------|---|--------------------------|---------------------|----------------------|
| IKP (full set) | 89 | 0.917 | — | -0.06 |
| SimpleQA | 10 | 0.904 | 0.991 | +0.03 |
| MMLU | 30 | 0.705 | 0.886 | +0.58 |
| MMLU-Pro | 25 | 0.689 | 0.900 | +0.82 |
| GPQA Diamond | 30 | 0.520 | 0.903 | +1.99 |

三个发现：
1. **IKP 在每个 matched subset 上都赢**，gap 最大对 reasoning-heavy benchmarks（GPQA Diamond: 0.52 vs 0.90）
2. **Reasoning benchmarks 在 fixed $\log_{10} N$ 上随 time 急剧漂移**：GPQA Diamond +2 pp/month 意味着 33B model 一年内提升 24 points without growing——这正是 Densing prediction，**直接 invalidate 这些 benchmarks 作为 parameter proxy**
3. **Purely-factual benchmarks 表现像 IKP**：SimpleQA time slope +0.03 pp/month，statistically indistinguishable from zero。Incompressibility property 是 factual subspace 的属性，不是"benchmarks"作为 category 的属性

## 7. Per-Tier Discrimination

Figure 4 揭示 step-function pattern：

| Tier | Slope (per decade) | Saturation | Discrimination |
|------|-------------------|------------|----------------|
| T1 | 0.124 | saturates @ 70B | ~binary check（is this even a real LM?） |
| T2 | moderate | saturates @ 120B | nearly as informative as T3 but earlier saturation |
| T3 | **0.324** (steepest) | saturates @ 1.2T | **most informative single tier**, $\rho = 0.983$ vs overall |
| T4 | 0.258 | no saturation in calibration range | strong discrimination |
| T5 | 0.097 | — | drops |
| T6 | 0.002 | essentially flat | **proprietary fleet separates from open-weight curve here** |
| T7 | $\approx 0$ | — | hard ceiling |

### 7.1 T3 是最强 discriminator

T3 slope 最陡（0.324/decade）、Spearman correlation with overall accuracy 最高（$\rho = 0.983$）、within-set spread 最宽（从 sub-1B 的 near-zero 到最大 open-weight 的 97%）。**单看 T3 score 就是 overall ranking 的强 proxy**——当 compute 不足以跑完 1400 probes 时，200-probe sub-evaluation 主要从 T3 抽取就能 recover 大部分 signal。

### 7.2 T6 是 proprietary frontier 与 open-weight 的分水岭

Open-weight calibration set 在 T6 slope = 0.002/decade——**1.6T 以内 T6 essentially 不动**。DeepSeek V4 Pro 在 T6 只得 1.0%。

但 proprietary models 在 T6 有 meaningful accuracy：GPT-5.5 38.5%, GPT-5.5-Pro 44.5%, GPT-5 19.2%, o1 16.8%, Claude Opus 4.6-Think 16.8%, Opus 4.7 13.8%。

**T6 jump above open-weight ceiling 是 effective capacity 超过 2T 的最清晰 single indication**。

### 7.3 T7 是 hard ceiling

188 个 evaluated models 中只有两个在 T7 上 > 1% penalized accuracy：Jamba-large 2.8%, Grok-4 1.0%。**所有 frontier proprietary models 都 0%**——包括我们 estimate 为 multi-trillion 的。

这不是 model 不够大的问题，是 **structural ceiling**：T7 probes 的 effective mention frequency 低于当前 pretraining corpora 在任何 model size 下能 preserve 的阈值。T7 作为 anchor——certifies obscurity ladder 没有跑完，long tail 仍超出 roster 中最大 model 的 reach。任何未来"saturated on IKP"claim 必须通过 extend obscurity scale 而非 parameter axis 来测试。

## 8. Thinking Mode 分析

27 个 base/think pairs 中，thinking mode 在 20 个 case 中改善 penalized accuracy（mean +2.2 pp，range -1.0 到 +10.3 pp）。

最大 gain：Grok-4 base→4.20-think +10.3 pp。Claude 4.x thinking variants 都改善 base counterparts（+0.7 到 +7.4 pp）。

**Benefit peaks at T3-T4（medium-hard tiers）并在 T7 消失**——支持 chain-of-thought 帮助 knowledge retrieval 但不创造 new stored knowledge 的解释。

Regressions（-1 pp）concentrated in MiMo 和 select Gemini variants，plausibly reflecting increased refusal conservatism in thinking mode。

## 8.5 Cross-Generation

- **Claude Opus base** 4→4.1→4.5→4.6→4.7: +13.3 pp end-to-end (52.4→57.9→61.1→63.1→65.7%)，monotonic increase
- **GPT-4→4o**: essentially flat (54.8→55.3%) but 不同 per-tier shape（4o 更 conservative on hard tiers）
- **Claude 3.5 Haiku→Haiku 4.5**: -5.7 pp (45.6→39.9%)，driven by tighter refusal calibration 而非 knowledge loss

## 9. Knowledge Fingerprinting

这是 paper 的另一个 clever trick：**不依赖 model weights，纯靠 black-box API 访问就能判断两个 model 是 shared-base, lineage 还是 full retrain**。

### 9.1 三个 Metrics

在 400 个 T5-T6 probes 上计算：
1. **Jaccard similarity $J$** on correct-answer sets——但 inflated by 几乎所有 frontier model 都知道的 probes
2. **Lift** = observed intersection / expected-under-independence——controls for common knowledge
3. **Hallucination Similarity Score (HSS)**：在两个 model 都 wrong（non-refusal）的 probes 上，给出相同 normalized wrong answer 的比例

**HSS 最 diagnostic**：independently trained models 几乎不会 converge 到 identical wrong rare fact，weight-sharing siblings 在 30-55% of shared-wrong probes 上这么做。

### 9.2 三个 Regimes

| Regime | HSS | J | Interpretation |
|--------|-----|---|----------------|
| Shared base | $\geq 0.30$ | $\geq 0.60$ | same weights with different inference/light alignment |
| Lineage | $[0.10, 0.30)$ | $\geq 0.50$ | post-training, continued pretraining, 或 distillation on shared ancestor |
| Retrained | $< 0.10$ | — | statistically indistinguishable from cross-vendor independent |

### 9.3 Within-family 发现

**OpenAI GPT-5 family**:
- GPT-5, GPT-5-pro, GPT-5-think 形成 tight shared-base cluster (HSS = 0.51, 0.53)
- **但每个 .x transition** (GPT-5→5.1, 5.1→5.2, 5.2-pro→5.3, 5.3→5.4) HSS ≤ 0.08 on 12-49 joint-wrong probes → **retrained regime**
- 支持 folklore：GPT-5.3 和 GPT-5.4 不是 GPT-5 的 post-trains，是 independent training runs，可能 from scratch 或 from significantly changed base
- GPT-5.4 vs GPT-5.4-pro (HSS=0.11, 71 joint-wrong) 是 lineage pair，shared base 只在 alignment stage differentiated

**Anthropic Claude Opus**:
- Opus 4→4.1: HSS=1.00 on 7 joint-wrong → clear lineage
- Opus 4.5→4.6: HSS=0.10 on 30 joint-wrong → weak lineage
- Opus 4.6→4.7: HSS=0.00 on 17 joint-wrong → **retrained**
- Sonnet 4.5→4.6: HSS=0.09 → retrained

**Google Gemini**:
- 每个跨 generation Flash 和 Pro pair 都是 retrained (Flash 2.0→2.5: HSS=0.00, 55 joint-wrong; Pro 2.5→3.1: 0.03, 35)
- Gemini generations 看起来是 full re-trains

**DeepSeek V3**:
- V3→V3.1 (HSS=0.23), V3.1→V3.2 (0.28) 都在 lineage regime → incremental continued pretraining on shared base

### 9.4 Cross-vendor Outliers

13,000 cross-vendor pairs 中 flagged 的小部分：

**Baidu ERNIE 4.5-300B-A47B**：同时与 5 个 independent models HSS ≥ 0.20：
- vs GPT-4o: 0.44
- vs Llama-3-70B: 0.43
- vs Mistral-Large: 0.40
- vs Qwen-Max: 0.33
- vs Mistral-Small-24B: 0.33

这个 pattern 是 **heavy training on mixed distilled outputs from Western frontier models** 的特征，而非 single teacher。

**Llama 3.1 70B** 作为 apparent "teacher" 出现在很多 pairings 中（HSS ≥ 0.30 vs grok-3, gemini-2.0-flash, qwen3-max, nova-pro, several OpenAI models）。这是因为 **Llama 3.1 是 2024 年最广泛使用的 open base for synthetic-data generation**，它的 characteristic hallucinations leak 进很多 downstream datasets。这不是任何 single downstream vendor 直接 distill from Llama 3.1 的证据。

**GPT-5 vs Grok-4** (HSS=0.31, 29 joint-wrong) 和 **GPT-5-pro vs Kimi-K2.6** (HSS=0.27, 22 joint-wrong) 单独 above threshold 但 joint-wrong probes 太少，无法 confidently reject innocent-explanation null（比如 train on similar recent web snapshots 或 overlapping synthetic-data mixtures）。

## 10. 什么决定 LLM 知道某个 Researcher/Fact？

### 10.1 Researcher Probes

Bibliometric signals explain roughly 1/3 of variance：
- log(OpenAlex citations) vs recognition rate: Spearman $\rho = 0.575$
- log(h-index) vs recognition rate: $\rho = 0.561$

但 **citation 是 approximately necessary but not sufficient**：
- < 50 citations 的 researcher 没有超过 15% recognition
- < 500 citations 的 researcher 没有超过 75% recognition
- **但很多 10K+ citations 的 researcher 在 25% recognition 以下**（lower-right region densely populated）

Exceptions（低 citation 但高 recognition）：Tri Dao (3K, 100%), Eyuboglu (267, 57%), Psaras (318, 69%)——**无一例外 attached to named, widely-distributed artifact**（FlashAttention, ColBERT-style work, IPFS）。

### 10.2 Subfield Effect

Table 5 按 CS subfield 排序 recognition rate：

| Subfield | n | Avg. citations | Mean recog. |
|----------|---|----------------|-------------|
| Information retrieval | 7 | 10,497 | 0.571 |
| Programming languages | 43 | 2,596 | 0.437 |
| Database systems | 7 | 3,670 | 0.429 |
| Data mining | 6 | 7,039 | 0.370 |
| Computer networking | 60 | 7,840 | 0.354 |
| Distributed systems | 42 | 2,206 | 0.307 |
| NLP | 7 | 1,569 | 0.302 |
| Computer security | 62 | 2,597 | 0.283 |
| Computer vision | 9 | 10,347 | 0.283 |
| Operating systems | 38 | 3,718 | 0.277 |
| Embedded systems | 11 | 1,414 | 0.277 |
| HCI | 7 | 10,051 | 0.260 |
| Theoretical CS | 11 | 7,563 | 0.204 |
| Computer architecture | 32 | 3,151 | 0.179 |

IR 和 PL lead despite moderate citations；HCI, theoretical CS, computer architecture lag despite comparable or higher citations。**Top subfields 恰好是 derivative content layer 最密集的**（tutorials, blog posts, course materials, library documentation）——model memorize 的不是 research 本身，而是 surrounds it 的 web text。

### 10.3 Three Mechanisms (from 2×2×2 Audit)

1. **Named artifacts dominate bibliometric mass**：attached to tool with $\geq 10K$ GitHub stars 或 dedicated Wikipedia page 的 researcher recognition $\geq 86\%$ regardless of citation count。**一个 high-visibility artifact 相当于 >10× additional citations**

2. **Name uniqueness is multiplicative**：controlling for citations，common East Asian surnames (两字中文名, 单字母 given name) recognition 22.6% vs uniquely spelled names 44.6%——factor-of-two attenuation。Eric Mitchell (DPO lead, 15K Google Scholar) 只有 57% 因为 "Eric Mitchell" 被 film director 和 unrelated professionals dominate web indices

3. **ML-Systems floor gap**：ML / low-recognition cell 是空的（no ML researcher below 43%）；Systems / low-recognition cell 非 empty at arbitrarily low citation。**这不是 individual merit，是 subfield ecosystem density**——ML labs 每 paper 产生 orders of magnitude more Twitter/blog/podcast content，这些 derivative content 就是 foundation-model training pipelines scrape 的

### 10.4 Factual Probes (Entities)

对 557 个 Wikidata-grounded probes：
- **Pageviews dominate sitelinks**：sitelinks Pearson $r = 0.502$；78-probe subset 上 pageviews $r = 0.774$，在 joint regression 中完全 subsume sitelinks (coefficient → 0.003)
- **Name-fact gap 随 prominence widen**：prominent entities (sitelinks 16+) name recog 0.728 vs year recog 0.473 (gap +0.255)；obscure entities (sitelinks 1-5) gap 只有 +0.040
- **Documentation sweet spot**：1900-1950 founded entities 最 known (mean recog 0.40)，比 < 1800 (0.25) 和 > 2000 (0.26) 都高——institutional documentation + historical notability 同时 apply

Domain-specific mention multipliers（controlling for sitelinks）：
- **Journal founding years +0.201** easier：每条 citation 和 bibliography entry implicitly states journal name
- **University founding years +0.117** easier
- **Museum founding years -0.102** harder
- **Bridge opening years -0.219** harder
- **Place founding years -0.310** harder：municipality founding date 通常 buried in single "History" section，而 weather/sports/transit mentions 不 carry year

### 10.5 Synthesis: Effective Mention Frequency

LLM knowledge **不由 abstract prominence 决定，而由 effective mention frequency 决定**：training-corpus documents 中 state specific target fact in retrievable form 并 attribute to specific name/entity 的数量。

Citation count, sitelink count, h-index 是 partial proxies；pageviews, practitioner adoption, named-artifact count, derivative-document mention density 是 better ones。

**对 researcher 的 practical corollary**：marginal effect of one additional widely-used open-source tool with clean name attribution > marginal effect of one additional well-cited paper。Tool generates derivative documents 每个都 carry author name，paper 只在 own bibliography record 和 direct citers 中 carry。

## 11. Safety Tuning 的 Silent Tax

Heavily safety-tuned models 拒绝 demonstrably knows 的 probes，产生 systematic capacity underestimates。

**Claude Sonnet line**：Sonnet 4 score 48.2% vs Claude 3.7 Sonnet 54.9%。On T5，Sonnet 4 issues explicit "I don't know" refusals on 175/200 (88%)；Sonnet 3.7 refuses only 108/200 (54%)。

如果 Sonnet 4 underlying knowledge 匹配或超过 Sonnet 3.7，6.7 pp gap largely 是 refusal policy artifact 而非 capacity。

**Hallucination rate 作为 vendor signature**：
- Google smaller Gemma models 89-97% hallucinate（几乎从不 refuse）
- Anthropic Claude 平均 11%（thinking variants 低至 3%）
- OpenAI GPT-4.1/4.1-mini 53-72%，GPT-5 nano/mini drop to 3-4%——generational safety calibration change

这个 vendor-dependent gap 使 IKP estimates for heavily safety-tuned proprietary models 应该 read as **lower bounds**——true capacity 介于 penalized 和 raw-accuracy predictions 之间。

## 12. Limitations

### 12.1 Training-data Variance

Calibration 假设 across models 大约 similar training signals。Vendor-specific pretraining data curation, RLHF refusal calibration, post-training fine-tuning 产生 within-family scatter 被 calibration absorb 为 residual 而非 slope。

### 12.2 Prediction Interval Width

90% PI factor 3.00 in either direction，与 inference-economics estimates [Epoch AI, 2024] comparable。Combining IKP with inference-economics constraints 可能 narrow interval。

### 12.3 Calibration Sparsity above 1T

最大 open-weight calibration anchors 是 DeepSeek V4 Pro (1.6T) 和 Kimi K2.5/K2.6 (~1T)。**只有两个 calibration points above 1T**，curve 在此 range 以上 effectively extrapolated，high end slope 由 very few data points 决定。

### 12.4 Retrieval Augmentation Confound

RAG-enabled model 可能 achieve high scores without parametric storage。**T7 near-zero scores across all models** suggests RAG 当前不 deployed for this type of factual queries。

### 12.5 Landmark Circularity

所有 6 个 landmark models 在 defining tier boundary 上 score inflated by construction。Effect 最强 for L6 (Gemini 3.1 Pro)，其 90% T6 accuracy 是 artifact。LOO-CV 部分地址：每个 landmark held out 时 prediction uses regression without it，resulting $R^2 = 0.922$ confirms fit 不是 driven by landmarks alone。

### 12.6 Probe Contamination

如果 probe set leaks into training data，accuracy inflates。Probe set 应保持 private，但 release only methodology and evaluation toolkit。这 creates 与 reproducibility 的 tension。

## 13. Wikidata Long-Tail Audit

10-round audit 发现 5 类 failure modes：

| Class | Issue | Resolution |
|-------|-------|------------|
| A1: Stale data | P159 headquarters stale (Roku 仍 LA Los Gatos 实际 2019 moved to San Jose) | Correct + recalibrate |
| A2: Wrong field semantics | P170 sculpture creator 返回 bronze foundry 而非 sculptor | Correct |
| C1: Generic title | "Madonna and Child" 指 dozens of Renaissance paintings | Add year+country+genre to template |
| C2: Generic geographic name | "Stone Bridge", "South Island" ambiguous | Add grounding |
| D1: Disputed sovereignty | Loaita Cay (Spratlys), Cape Plaka (Crimea) | Drop probe |

Per-fact-type reliability at long tail：
- P403 river mouth: 100% pass
- P571 inception (founding year): ~95%
- P57 director: 67-100%
- P112 founder: 75%
- P170 creator (sculpture): 70%
- P58 screenwriter: 50%
- **P159 headquarters: 25%** (stale)
- P170 creator (painting): 100% title-collision at T6/T7

## 14. Case Study: USTC Hackergame

Author 三年来用同一 prompt 压测新 model："What do you know about USTC Hackergame?"——年度 CTF，由 USTC Linux User Group 自 2014 年举办，有 idiosyncratic 中文 challenge titles。

- **GPT-4o (2024-05)**：知道 contest 存在但 **invented fake challenge names**。Nine different years 都返回 identical fabricated list ("Hello World", "Maze", "Calculator", "Reverse Engineering", "Quantum Computing")——**deterministic fabrication, repeated across independent probes**
- **Claude 3.7 Sonnet (2025-02)**：第一个 list real per-year challenges，19 verified 2023 titles
- **Gemini 3.1 Pro (2026-01)**：Year-by-year breakdown 2018-2023 with specific real challenge names
- **DeepSeek V4 Pro (1.6T, 2026-04)**：**正确说 2014 start year**（唯一做到的 model）但 **fabricates per-year challenges**——meta-fact 与 long-tail content stored independently
- **GPT-5/5.5**：burn output budget on reasoning，produce empty content
- **Claude Opus 4.7**：说 "I don't have reliable detailed memory and don't want to fabricate specifics"

观察：
1. **Transition sharp and dateable**：9-month window 内 Hackergame writeups 进入 training，不是逐渐 across field
2. **Knowing meta-fact 不 imply knowing content**：DeepSeek V4 Pro 知 start year 但 fabricate challenges
3. **Refusal vs hallucination 是 vendor-level choice**，与 measurement 交互

## 15. Open Questions

1. **When does T7 fall?** 当前无 model > 0.5% penalized accuracy on T7。若 factual capacity 继续 log-linear scale，有 approximate parameter threshold 使 T7 开始 discriminate。观察第一个 T7-positive model 会 empirically validate/refute incompressibility argument

2. **How much within-family scatter 是 post-training 而非 parameter count?** "Flash"-class variants 通常 IKP score 接近 Pro-class siblings despite presumed parameter differences。Extend 到 additional same-vendor pairs with disclosed sizes 会 partition scatter 进 pretraining-data, RLHF, architecture contributions

3. **Can RLHF-hidden knowledge be recovered?** "Opus 4.1 knows but refuses" pattern implies alignment tax measurable in pp of factual recall。Prompting strategies, deprobing, 或 activation-steering methods 恢复 refused-but-known answers 会 place upper bound on safety tuning obscures 的 capacity

4. **Do knowledge fingerprints persist through continued pretraining?** 若 fingerprints survive substantial continued pretraining on fresh data，成为 practical training-free provenance tool for open-weight licensing enforcement

## 16. 我的 Critique 与 Extensions

几个值得深挖的方向：

### 16.1 Calibration 的 causality

Log-linear fit $A = \alpha \log_{10} N + \beta$ 在 89 models 上 $R^2 = 0.917$ 是 striking，但 **calibration set 中 model size 和 training data volume 高度 correlated**。Allen-Zzu and Li 的 2 bits/param 是 under ideal training；实际 model 的 effective bits/param 取决于 training tokens, data quality, training epochs。

Lu et al. [2024] 显示 fact capacity linearly scales with model size AND exponentially with epochs。如果 calibration set 中 larger models 也 trained on more tokens（Chinchilla-optimal），slope $\alpha = 0.147$ 同时 capture parameter scaling 和 data scaling。分离两者需要 hold data constant 的 ablation——这 open-weight calibration set 无法做。

### 16.2 Judge Reliability

Judge 是 Gemini 3 Flash Preview。Estimated error rate 0.1-0.2% verified by manual audit。但 **judge 自己作为 frontier model 可能在 T6-T7 有 systematic bias**——如果 judge 自己不知道某 researcher 的 subfield，可能 misclassify CORRECT_STRONG 为 CORRECT_WEAK 或 WRONG。

更严重的是 **judge 与被评估 models 的 knowledge correlation**：如果 judge 知道某个 fact，frontier models 更可能也知道（因为 train on similar corpora）。这使 judge 在 T1-T4 上 reliable（judge 也知道），但在 T6-T7 上可能 unreliable（judge 也不知道，misclassify）。

### 16.3 Probe Contamination 的 Circularity

Probe set 应保持 private 但 release methodology——但 methodology 详细到可以 reconstruct equivalent probe sets。如果多个 group 独立 reconstruct，frontier labs 训练时可能 scrape 到这些 reconstructed sets，contaminate future models。

更根本的是，**Wikidata 和 DBLP 本身是 public，frontier labs 训练时已 scrape**。IKP probes 的"obscurity"是相对当前 pretraining corpora 的——如果 vendors 主动 expand long-tail coverage（比如 DeepSeek V4 Pro 知道 2014 start year 说明 Hackergame Wikipedia page 在 training corpus 中），IKP probes 会逐渐 saturate。

Author 论称 IKP structurally resistant to saturation 因为 measures stored facts not procedures，但 **stored facts 本身可以被 train 进去**。T7 ceiling 可能不是 fundamental，只是当前 corpora 没覆盖那些 entities 的 mention frequency threshold。

### 16.4 Fingerprinting 的 Null Model

HSS threshold 0.30 / 0.10 是 empirical。**真正独立的 models 在 T5-T6 joint-wrong probes 上 HSS 的 null distribution 是什么？**

如果两个 models train on 相同 recent web snapshot（比如都 scrape 了 2024-01 的 Wikipedia dump），它们可能在 rare facts 上 **independent converge on same wrong answer**——因为 same training data 的 same biases。这会使 HSS false positive。

Author 在 limitations 中承认：method "does not distinguish between direct distillation and shared training data"。但 quantifying 这个 null（e.g., 通过 train two independent models on same data from scratch）会 strengthen fingerprinting claims。

### 16.5 Per-tier Probe Allocation

Author 自己承认 uniform allocation 200 probes/tier suboptimal：T1-T2 saturate early，T7 provides near-zero signal，T3-T5 carry 大部分 discriminative power。Future version 应该 allocate更多 probes to informative tiers。

但这也 creates circularity：如果 T3 weight 提高，T3 score 在 aggregate accuracy 中 dominate，calibration 可能 overfit T3。需要 joint optimization of probe allocation 和 calibration weighting。

### 16.6 Effective Mention Frequency 的 Operationalization

Section 6.10 的 synthesis 概念上 elegant，但 **effective mention frequency 没 directly measure**。Author 用 pageviews, GitHub stars, Wikipedia page presence 作为 proxies，但这些 proxies 之间 correlation 结构 unclear。

更根本的是：**training corpus 不等于 web**。Common Crawl 是 web subset，但 frontier labs 有 curated corpora（Reddit, GitHub, arXiv, PubMed, books）的 differential weighting。"Mention frequency in Common Crawl" ≠ "mention frequency in frontier training corpus"。

可能的 extension：用公开 web corpus（Common Crawl snapshots）measure per-fact mention frequency，看 IKP accuracy 与 measured mention frequency 的 relationship 是否比与 pageviews 更紧。这会直接 test "effective mention frequency" hypothesis。

## 17. 与现有 scaling law literature 的关系

### 17.1 Kaplan et al. 2020 (Chinchilla)

Neural scaling laws establish power-law between compute, data, model size 和 loss。Chinchilla [Hoffmann et al., 2022] refine 到 compute-optimal training。这些 laws 是关于 **next-token prediction loss**，不区分 procedural 和 factual components。

IKP 的 contribution 是 **decompose loss into incompressible factual + compressible procedural**，并 show 两者的 scaling 行为 fundamentally different。Factual loss 受 Shannon entropy 硬下界约束，procedural loss 可以通过 architecture/data 改进压缩。

### 17.2 Allen-Zhu and Li 2025

Physics of LMs Part 3.3 establish 2 bits/param 的 capacity bound under ideal training。**Key assumption**：training data 是 synthetic factual tuples，model 通过 MLP layers 实现 key-value memory。

IKP 把这个理论 bound 倒过来用：如果 model knows F bits of facts，参数量至少 F/2。**但 IKP 没直接 measure bits stored**——measure 的是 accuracy on probe set，accuracy → bits stored 的 mapping 通过 empirical calibration（Kandpal et al. 的 log-linear）实现。

### 17.3 Morris et al. 2025

Refine Allen-Zhu 的 estimate 到 3.6 bits/param for GPT-style models，distinguish unintended memorization from generalization，show models "memorize until capacity fills, at which point grokking begins"。

这给 IKP 一个重要 caveat：**memorization 和 grokking 的 transition 使 accuracy-vs-parameters relationship 非 globally log-linear**。在 grokking threshold 附近，accuracy 可能 jump 而非 smoothly increase。IKP 的 log-linear fit 在 89 models 上 $R^2 = 0.917$ 说明 grokking transition 在 probe level 上 averaged out，但 individual probes 可能 show phase transitions。

### 17.4 Kandpal et al. 2023

Show accuracy on rare facts scales log-linearly with model size，$R^2 = 0.98$ within BLOOM family。**但 BLOOM family 内 architecture 和 training recipe held roughly constant**。IKP across 19 vendors 是 stronger test——vendor-specific choices 产生 within-family scatter 但 log-linear relationship 仍 hold（$R^2 = 0.917$）说明 incompressibility property 是 vendor-agnostic。

## 18. Connection to Mechanistic Interpretability

[Geva et al., 2021] show transformer FFN layers function as key-value memories。[Dai et al., 2022] identify specific "knowledge neurons" responsible for individual facts。[Meng et al., 2022] (ROME) causally localize and edit factual associations。

IKP 是 **behavioral probe**——不打开 model 看 neurons，只通过 API 测 accuracy。但 IKP 的 tier structure 和 mechanistic findings 互补：

- T1-T2 facts 可能 stored in 更 general, widely-activated neurons（每 model 都有）
- T6-T7 facts 可能 stored in 更 specialized, rarely-activated neurons（Hong et al. 2025 show stronger models develop more specialized parameter vectors）
- MoE total > active params 的 finding 与 factual storage distributed across experts 一致——factual neurons 不 concentrated 在 per-token activated experts

可能的 extension：在 open-weight models 上，correlate per-tier IKP accuracy 与 **neuron-level measures**（e.g., factual neuron density, layer-wise localization）。这会 build bridge between behavioral probes 和 mechanistic understanding。

## 19. 对 Practice 的 Implications

### 19.1 给 Frontier Labs 的信号

Closed-source labs 不披露 parameter counts，IKP 提供 intrinsic estimator。**如果 estimate 与 vendor messaging 不符，可能 signal**：
- Pro tier 几乎不增加 factual capacity（符合 Pro 的 reasoning purpose）
- GPT-5.x point releases 都是 full retrains（揭示 release cadence 策略）
- Anthropic 的 Claude Opus line 显示 monotonic scale-up（vs OpenAI 的 iteration pattern）

### 19.2 给 Researchers 的信号

你的 work 被 frontier model 吸收的程度 **不是 citations 的 monotonic function**。更接近：

$$\text{recognition} \propto \text{citations} \times \text{name uniqueness} \times \text{named-artifact amplification} \times \text{subfield-ecosystem density}$$

每个 multiplicative factor span 2-5× range。**Marginal effect of one additional widely-used open-source tool with clean name attribution > marginal effect of one additional well-cited paper**。

Tool generates derivative documents 每个 carry author name，paper 只在 own bibliography record 和 direct citers 中 carry。如果你希望被未来 frontier model "memorize"，**create named artifacts** 比 publish more papers effective。

### 19.3 给 Evaluation Community 的信号

MMLU, GPQA Diamond 等 reasoning benchmarks 不能作为 parameter proxy——它们在 fixed $\log_{10} N$ 上随 time 漂移（GPQA +2 pp/month）。**Purely-factual benchmarks**（SimpleQA, IKP）漂移 near-zero，适合 parameter estimation。

IKP 本身 structurally resistant to saturation：measures stored facts，model "saturate" a tier 表示 memorize 了 long-tail facts——正是 IKP exists to estimate 的 quantity。Pipeline 本身是 benchmark：Wikidata sitelink-stratified sampling + DBLP citation-stratified sampling + landmark ladder calibration 几小时 compute regenerate fresh probe set from disjoint entities。

## 20. Summary

这篇 paper 的 conceptual contribution 比 empirical contribution 更重要：**区分 compressible procedural capability 和 incompressible factual capacity**，并 show 前者受 Densing Law 影响、后者不受。这 reframes "benchmark saturation = end of scaling" 的误读——benchmarks saturated 因为 stopped measuring incompressible part。

技术 implementation 优雅：1,400 probes × 7 tiers × landmark calibration × 4-way evidence-aware judge × hallucination penalty × LOO-CV。$R^2 = 0.917$ across 4 orders of magnitude 是 striking validation。MoE total > active params 是 clean mechanistic finding。Fingerprinting via HSS 是 clever 的 black-box lineage detection。

但 calibration 的 causality（parameter vs data scaling 混淆）、judge reliability 在 T6-T7 的 systematic bias、probe contamination 的 fundamental tension、HSS null model 的 quantification——这些是 future work 应该 address 的 limitations。

对 Andrej 来说，这 paper 的 intuition-building 价值在于：**给"factual knowledge is fundamentally different from procedural capability"一个 quantitative operational handle**。不是 philosophy，是可测量的 scaling axis。如果继续 scale，factual capacity 继续 log-linear 增长；如果 architecture 改进，procedural density 继续 densify；这两件事 independent，都 important，但 mix up 它们（如 Densing Law 推论 scaling 已 ended）是 category error。

---

### References

- Paper: [Incompressible Knowledge Probes (companion site)](https://01.me/research/ikp)
- Code: [github.com/19PINE-AI/ikp](https://github.com/19PINE-AI/ikp)
- [Allen-Zhu and Li, 2025 - Physics of LMs Part 3.3](https://arxiv.org/abs/2404.05568) - 2 bits/param capacity bound
- [Morris et al., 2025 - How much do LMs memorize](https://arxiv.org/abs/2505.24832) - 3.6 bits/param, grokking
- [Kandpal et al., 2023 - LLMs struggle with long-tail](https://arxiv.org/abs/2211.08411) - log-linear scaling of rare facts
- [Lu et al., 2024 - Scaling laws for fact memorization](https://aclanthology.org/2024.findings-emnlp/) - linear size, exponential epochs
- [Huang et al., 2025 - Densing Law](https://www.nature.com/natmachintell/) - capability density doubles every 3.5 months
- [Geva et al., 2021 - FFN as key-value memories](https://aclanthology.org/2021.emnlp-main.446/)
- [Meng et al., 2022 - ROME](https://arxiv.org/abs/2202.05262) - locating/editing factual associations
- [Dai et al., 2022 - Knowledge neurons](https://aclanthology.org/2022.acl-long.396/)
- [Kaplan et al., 2020 - Scaling laws](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al., 2022 - Chinchilla](https://arxiv.org/abs/2203.15556)
- [Petroni et al., 2019 - LMs as knowledge bases](https://aclanthology.org/D19-1250/)
- [Roberts et al., 2020 - How much knowledge in params](https://aclanthology.org/2020.emnlp-main.465/)
- [Shannon, 1948 - Mathematical theory of communication](https://ieeexplore.ieee.org/document/6773024)
- [Vrandečić and Krötzsch, 2014 - Wikidata](https://cacm.acm.org/research/wikidata/)
- [Ludziejewski et al., 2025 - Joint MoE scaling laws](https://arxiv.org/abs/2404.02236)
- [Hong et al., 2025 - Parameter specialization](https://arxiv.org/) - stronger models, more specialized parameter vectors
- [Tsai et al., 2025 - RoFL](https://arxiv.org/abs/2505.12682) - robust fingerprinting
- [Nasery et al., 2025 - Scalable fingerprinting](https://arxiv.org/abs/2502.07760) - 24,576 fingerprints
- [Pasquini et al., 2025 - LLMmap](https://www.usenix.org/conference/usenixsecurity25) - fingerprinting LLMs
- [Gao et al., 2025 - Model equality testing](https://arxiv.org/) - 11/31 Llama API endpoints different
- [Epoch AI - Inference economics](https://epoch.ai/) - 2×+ uncertainty parameter estimation
- [Pan et al., 2025 - Kolmogorov complexity view](https://arxiv.org/) - Syntax-Knowledge model
- [Badhe et al., 2026 - Long-tail taxonomy](https://arxiv.org/abs/2602.16201)
- [Mallen et al., 2023 - When not to trust LMs](https://aclanthology.org/2023.acl-long.319/) - parametric vs retrieval
- [Zhang et al., 2025 - Knowledge overshadowing](https://aclanthology.org/2025.acl-long/) - popular suppresses rare
- [Carlini et al., 2021 - Extracting training data](https://arxiv.org/abs/2012.07805)
- [Carlini et al., 2023 - Quantifying memorization](https://arxiv.org/abs/2202.07646)
- [Chen et al., 2025 - Continual memorization](https://arxiv.org/) - factoid retention degrades across fine-tuning
- [Shi et al., 2025 - Distillation detection](https://arxiv.org/) - open-weight models
- [Li et al., 2025 - Shadow-MoE distillation detection](https://arxiv.org/abs/2510.16968)
- [Cai et al., 2025 - Model substitution auditing](https://arxiv.org/)
- [Tirumala et al., 2022 - Memorization without overfitting](https://arxiv.org/abs/2205.10770)
- [Chang et al., 2024 - Knowledge acquisition](https://arxiv.org/) - 7B vs 1B qualitative difference
- [Zipf, 1949 - Human Behavior and Least Effort](https://www.worldcat.org/title/476842365)
- [Piantadosi, 2014 - Zipf's law review](https://link.springer.com/article/10.3758/s13423-014-0585-6)
- [Hendrycks et al., 2021 - MMLU](https://arxiv.org/abs/2009.03300)
- [Wang et al., 2024 - MMLU-Pro](https://arxiv.org/abs/2406.01574)
- [Rein et al., 2024 - GPQA Diamond](https://arxiv.org/abs/2311.12022)
- [Wei et al., 2024 - SimpleQA](https://openai.com/index/introducing-simpleqa/)
- [Wei et al., 2022 - Chain-of-thought](https://arxiv.org/abs/2201.11903)
- [Ouyang et al., 2022 - InstructGPT/RLHF](https://arxiv.org/abs/2203.02155)
- [Bai et al., 2022 - Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Christiano et al., 2017 - RLHF](https://arxiv.org/abs/1706.03741)
- [Lewis et al., 2020 - RAG](https://arxiv.org/abs/2005.11401)
- [Phan et al., 2025 - Humanity's Last Exam](https://arxiv.org/abs/2501.02594)
- [USTC Hackergame writeups](https://github.com/USTC-Hackergame)
- [USTC Hackergame platform repo](https://github.com/ustclug/hackergame)
