---
source_pdf: Adam’s Law Textual Frequency Law on Large Language Models.pdf
paper_sha256: adb5a98229332c42e431388fb59ee23fd0e3711afc3accd1681d302c52911671
processed_at: '2026-07-18T01:58:11-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Paper 深度解析: Adam's Law — Textual Frequency Law on Large Language Models

Andrej, 这篇paper本质上是在问一个非常naive但被严重忽视的问题: **在保持语义相同的条件下, 给LLM喂"常见"的措辞还是"罕见"的措辞, 模型表现更好?** 答案非常反直觉地一致 — 高频词组合的句子几乎在所有任务上都赢。下面我尽量按你能接受的方式build up the intuition, 公式我会逐个变量拆开讲。

---

## 1. Core Insight: 为什么频率会左右LLM行为

LLM的预训练本质是在拟合一个token distribution $P(w)$。Cao et al. (2024) 已经发现同义但不同措辞的prompt给LLM带来的输出quality可以差很多, 但没人系统地追究"哪个因素"。这篇paper假设:**pre-training阶段出现频率更高的措辞, 在模型的内部 representation 上更"对齐", 所以推理时更"顺滑"**。

这跟人类cognition也有connection — Desai et al. (2020) 用 fMRI/EEG 发现高频词引发更强neural activation (https://doi.org/10.1080/23273798.2019.1640496); Alexandrov et al. 2011 在 PLoS ONE 上也是类似结论 (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0022999)。所以这其实是一个 brain-LLM parallel。

更直接的motivation来自两件事:
- **Oh et al. 2024** (EACL) 发现 larger models predict rare words better (https://aclanthology.org/2024.eacl-long.145/) — 暗示 predicting rare words 是个 harder task, larger model才扛得住;
- **He et al. 2025** (EMNLP) 证明 pre-trained models 在 token distribution follows Zipf's law 时下游性能最好 (https://aclanthology.org/2025.emnlp-main.1567/)。

这两个加起来给了一个hint: model的"舒适区"就是Zipfian的高频端。

---

## 2. 框架三件套: TFL + TFD + CTFT

### 2.1 Task Formulation (Eq. 1)

LLM看成一个seq2seq系统, 最大化:

$$P(\mathbf{y} \mid \mathbf{i}, \mathbf{x}) = \prod_{j=1}^{\mathbb{T}} P(y_j \mid y_1, \ldots, y_{j-1}, \mathbf{i}, \mathbf{x})$$

变量含义:
- $\mathbb{T}$: 输出序列的总长度 (token数)
- $y_j$: 第 $j$ 个output token
- $y_1, \ldots, y_{j-1}$: 已经生成的前缀 (autoregressive context)
- $\mathbf{i}$: instruction, 比如 "Translate the following sentence..."
- $\mathbf{x}$: source sentence, 比如待翻译的英文

Math Reasoning里 $\mathbf{x}$ 可以缺省, 因为题目本身就内嵌在 $\mathbf{i}$ 里; MT里 $\mathbf{i}$ 是translate命令、$\mathbf{x}$ 是源句。

### 2.2 Textual Frequency Law (Eq. 2 & 3)

$$\arg\max_{\mathbf{x} \in \mathcal{P}} \mathrm{sfreq}(\mathbf{x}, \mathcal{D})$$

- $\mathcal{P}$: paraphrase集合, 即所有"语义等价"的候选表达
- $\mathrm{sfreq}(\cdot)$: sentence-level frequency function
- $\mathcal{D}$: 用来估频率的corpus (可以是任何online corpus, 因为大多数LLM的training data是closed-source)

Sentence-level frequency的具体定义 (Eq. 3):

$$\mathrm{sfreq}(\mathbf{x}, \mathcal{D}) = \sqrt[\mathbb{K}]{\frac{1}{\prod_{k=1}^{\mathbb{K}} \mathrm{wfreq}(x_k, \mathcal{D})}}$$

变量:
- $\mathbb{K}$: 句子 $\mathbf{x}$ 的长度 (token数)
- $x_k$: 句子中第 $k$ 个word
- $\mathrm{wfreq}(x_k, \mathcal{D})$: 词 $x_k$ 在corpus $\mathcal{D}$ 中的频率
- 那个 $\sqrt[\mathbb{K}]{\cdot}$ 是开 $\mathbb{K}$ 次方, 等价于取geometric mean

这里有个notation上的"反向"细节: paper正文写的是 $\frac{1}{\prod \mathrm{wfreq}}$ 取 $\mathbb{K}$ 次根, 而proof part (Assumption 4)写的是 $\mathrm{sfreq}(x) = (\prod_k P(x_k))^{1/K}$, 即直接是token marginal概率的几何平均。 两者的关系其实是: 如果把 $\mathrm{wfreq}$ 看成"稀罕度" (1/probability), 那这两个等价。但日常工程实现里直接用 wordfreq (https://github.com/rspeer/wordfreq, Speer 2022) 的 Zipf frequency 作为 $\mathrm{wfreq}$, 然后做几何平均, 就是 paper 实际用的。**Intuition**: 一个句子的"频率", 由它所有token的频率几何平均决定。几何平均会强烈惩罚低频词 (一个 rare word 拖低整句 score), 这正合 TFL 的逻辑。

### 2.3 Textual Frequency Distillation (Eq. 4 & 5)

问题: 我们手头的 $\mathcal{D}$ 不是LLM的真实training data, 只是 web corpus 的 proxy。所以频率估的不准, 特别是 rare words 会"假阴性"为0。

TFD的思路很巧: 让LLM自己做 story completion, 用它的"续写语料"来反推它训练时见过的token分布:

$$\mathcal{F}_2 = \mathrm{sfreq}(\mathbf{x}, \mathcal{D}')$$

$\mathcal{D}'$ 是用 prompt "Please conduct story completion on the following data: <textual data>" 让LLM生成出来的corpus。然后用一个组合公式融合两个估计 (Eq. 5):

$$\mathcal{F}(x) = \alpha \mathcal{F}_1(x) + (1 + \zeta \cdot \mathbb{1}(\mathcal{F}_1(x)=0)) \cdot \beta \mathcal{F}_2(x)$$

变量详解:
- $\mathcal{F}_1(x)$: 从 web corpus 估的原始 frequency (Eq. 3的输出)
- $\mathcal{F}_2(x)$: 从LLM story completion 蒸馏出的 frequency
- $\alpha, \beta$: 两路的权重超参 (类似 $\alpha + \beta$ 应该归一, 但paper没强制)
- $\zeta$: "strengthening factor", 当 $\mathcal{F}_1(x) = 0$ (即原始corpus里完全没出现过) 时, 放大 $\mathcal{F}_2$ 的贡献
- $\mathbb{1}(\cdot)$: indicator function, 括号里条件为真则取1, 否则取0

这个设计的motivation: rare words 在 web corpus 里可能由于crawler覆盖问题而漏掉, 但如果 LLM 自己能"续写"出相关上下文, 说明它在 training 时见过。所以 $\mathbb{1}(\mathcal{F}_1=0)$ 这个trigger就是要挽救"假阴性rare word"。这跟 **Chinchilla / Phi 系列里的 data quality filtering** 思路异曲同工, 但目标不一样: 那些方法过滤 bad data, 这里是补救被低估的"对模型而言其实高频"的data。

### 2.4 Curriculum Textual Frequency Training (Eq. 6)

$$\mathrm{sort}_{x_n \in \mathcal{T}} (\mathcal{F}(x_n))$$

- $\mathcal{T}$: training set, 包含 $N$ 个instances
- $x_n$: 第 $n$ 个训练样本
- $\mathrm{sort}$: 按频率**从低到高**重排

为什么是 low-to-high? 这是paper里最有趣的细节之一。引用的是 Lu & Lam 2023 (https://aclanthology.org/2023.eacl-main.5/) — 低频表达更多样化 (因为低频词的collocation更flexible), 先训练多样性, 再训练共性。这跟 Jiang et al. 2014 self-paced learning 的思路接得上 (https://papers.nips.cc/paper/2014/hash/...)

注意 CTFT 跟传统 curriculum learning (easy-to-hard) 不同 — Table 5 显示 frequency 跟 complexity metrics (Max Dependency Tree Depth, Mean Dependency Distance, Flesch-Kincaid Grade Level) 几乎没相关性 (Pearson correlation 通常 |r| < 0.3)。这就是CTFT作为独立axis的理由。

---

## 3. 理论证明: 为什么 TFL 在数学上能成立

这是paper的精髓, 在 Appendix A–I。我帮你重述一遍build intuition。

### 3.1 四个 Assumptions

**Assumption 1 (Zipf's Law for Token Frequencies)**

$$P(w_r) = \frac{r^{-s}}{Z}, \quad s > 0, \quad Z = \sum_{n=1}^{|V|} n^{-s}$$

- $w_r$: rank 为 $r$ 的token (rank=1 是最高频)
- $s$: Zipf exponent, 通常在 1.0–1.2 之间 (自然语言)
- $|V|$: 词汇表大小
- $Z$: 归一化常数

**Assumption 2 (Rank-Dependent Log-Domain Approximation)**

$$|\ln Q_\theta(w_r) - \ln P(w_r)| \leq \varepsilon(r)$$

- $Q_\theta(w_r)$: 模型分配给 token $w_r$ 的marginal probability
- $\varepsilon(r)$: rank-dependent的误差上界

这是一个**比 cross-entropy 收敛更强的条件**: 交叉熵只控制 P-weighted average loss, 这里要求 pointwise 在log-domain误差有界。paper在 Remark 2 给了三个empirical motivation:
- Mikhaylovskiy 2025 (https://aclanthology.org/2025.findings-emnlp.1026/): LLM生成的文本本身也服从Zipf's law
- Kobayashi et al. 2023 (https://aclanthology.org/2023.findings-acl.451/): Transformer prediction head 的 bias 项实际编码了 corpus word frequency
- Oh et al. 2024: 频率modulates model-human surprisal alignment

**Assumption 3 (Bounded Marginal-Conditional Discrepancy)**

定义 contextual discrepancy:

$$\eta_{x_k} \triangleq \ell_\theta^c(x_k \mid x_{<k}) - \ell_\theta^m(x_k) = \ln Q_\theta(x_k) - \ln Q_\theta(x_k \mid x_1, \ldots, x_{k-1})$$

- $\ell_\theta^m$: marginal NLL, 即 $-\ln Q_\theta(w)$
- $\ell_\theta^c$: conditional NLL, 即 $-\ln Q_\theta(w \mid \text{context})$

assume $|\bar{\eta}_x| \leq \eta_x$, where $\bar{\eta}_x = \frac{1}{K} \sum_k \eta_{x_k}$.

Remark 3 的关键观察: 对于 high-frequency sentences, context 让 token 更 predictable (例如 "United States of ___"), 所以 $\eta_{x_k} < 0$ 倾向 — 这恰好对 TFL 有利 (loss会更低), 但 proof 用的是保守的 absolute bound。

**Assumption 4 (Sentence Frequency via Geometric Mean)**

$$\mathrm{sfreq}(x) \triangleq \left( \prod_{k=1}^K P(x_k) \right)^{1/K}$$

unigram independence model — 故意忽略 word order 与 inter-token dependency。对 paraphrase comparison 是合理近似, 因为 paraphrase 主要 differ 在 word choice 不在 syntax。

### 3.2 Theorem 1: Token-Level Semi-Log Linearity

把 Assumption 1 代入 self-information:

$$-\ln P(w_r) = -\ln(r^{-s}/Z) = s \ln r + \ln Z = s \ln r + C$$

其中 $C \triangleq \ln Z > 0$。然后套 Assumption 2:

$$\ell_\theta^m(w_r) = s \ln r + C + \delta_{w_r}, \quad |\delta_{w_r}| \leq \varepsilon(r)$$

**intuition**: 在 (ln r, loss) semi-log 平面上, token loss 是一条斜率 $s$、截距 $C$ 的直线, 周围有 rank-dependent 误差带。**注意是 semi-log, 不是 log-log** — 不要把它跟power-law for loss本身混淆。

### 3.3 Theorem 2: Token-Level Monotonicity 的充分条件

对 $r_i < r_j$, 要保证 $\ell_\theta^m(w_i) < \ell_\theta^m(w_j)$, 充分条件是:

$$\varepsilon(r_i) + \varepsilon(r_j) < s \ln\left(\frac{r_j}{r_i}\right)$$

uniform bound $\varepsilon$ 时简化为 $r_j/r_i > e^{2\varepsilon/s}$。

**intuition**: 只有当两个 token 的 rank ratio 足够大 (Zipf gap 主导 approximation error 时), 才能严格说"高频词loss低"。极端相邻 rank 的 rare token 比不出来 — 这是 proof 的内在 limitation。

### 3.4 Theorem 3: Sentence-Level Loss Decomposition (Eq. 17–20)

把每个 token 的 conditional loss 拆三块:

$$\ell_\theta^c(x_k \mid x_{<k}) = \underbrace{-\ln P(x_k)}_{\text{ideal marginal NLL}} + \underbrace{\delta_{x_k}}_{\text{marginal approx error}} + \underbrace{\eta_{x_k}}_{\text{contextual discrepancy}}$$

对句子做平均:

$$\ell_\theta(x) = -\ln \mathrm{sfreq}(x) + \bar{\delta}_x + \bar{\eta}_x$$

其中:
- $\bar{\delta}_x = \frac{1}{K}\sum_k \delta_{x_k}$, 满足 $|\bar{\delta}_x| \leq \bar{\varepsilon}_x = \frac{1}{K}\sum_k \varepsilon(r_k)$
- $\bar{\eta}_x = \frac{1}{K}\sum_k \eta_{x_k}$, 满足 $|\bar{\eta}_x| \leq \eta_x$

**核心 insight**: sentence-level loss ≈ -log(sentence frequency) + 噪声项。频率越高 → -log sfreq 越小 → loss 越低, 误差项以 $\bar{\varepsilon}_x + \eta_x$ 为界。

Remark 7 给了一个tighter估计: 如果 $\delta_{x_k}$ 近似零均值且weakly correlated, 中心极限定理给出 $|\bar{\delta}_x| \approx O(\bar{\varepsilon}_x / \sqrt{K})$。**longer sentence 的 error bound 实际上更紧** — 这本身就是一个可证伪的预测。

### 3.5 Theorem 4: TFL 的充分条件 (Eq. 21)

两个 paraphrase $x, x'$, $\mathrm{sfreq}(x) > \mathrm{sfreq}(x')$, 则 $\ell_\theta(x) < \ell_\theta(x')$ 的充分条件:

$$\ln \frac{\mathrm{sfreq}(x)}{\mathrm{sfreq}(x')} > (\bar{\varepsilon}_x + \eta_x) + (\bar{\varepsilon}_{x'} + \eta_{x'})$$

**intuition**: 两句的 log frequency ratio 必须超过两个误差 bound 的和。这个条件 conservatively 估计了"反向"情形 (worst case 误差同向叠加)。实际情形中误差通常会互相cancel (中心极限), 且 high-freq sentences 的 $\eta$ 倾向为负, 所以实际条件更宽松。

### 3.6 从 Loss Ordering 到 Task Performance (Section F)

paper诚实地说: 这个推导只到 NLL ordering, 没到 accuracy/BLEU ordering。后面那一步是empirical bridge, 不是定理。给出的论证是:
- **Prompting**: 低 input NLL → 输入落在 model 训练分布的"well-calibrated区域" → 更可能激活正确推理路径
- **Fine-tuning**: 高频 input → 更稳定 gradient signal, 更高 effective learning rate for output mapping, 同时减少 catastrophic forgetting risk (因为离 pre-training distribution 近)

这点 build intuition 非常关键, 因为这是paper整个empirical claim的"trust bridge"。

---

## 4. 数据集 TFPD 的构造

paper自建了 Textual Frequency Paired Dataset (TFPD)。流程是:

1. **源数据集**: GSM8K (https://github.com/openai/grade-school-math), FLORES-200 (https://github.com/facebookresearch/flores), CommonsenseQA (https://www.tau-nlp.org/commonsenseqa), 外加 Tool Calling 自采
2. **Paraphrase生成**: 用 GPT-4o-mini 对每个原句生成 20 个 paraphrase — 10个低频 (用更复杂词)、10个高频 (用更简单词)。Prompt设计很讲究:
   ```
   My goal is to transform the original sentence into both more common and less common expressions.
   Note: Do not omit any words such as verbs, adjectives, nouns, or adverbs.
   You must generate two types of sentences: (1) ten sentences using less common, more complex words.
   (2) ten sentences using more common, simpler words.
   ```
3. **频率筛选**: 对生成的20句, 用 Eq. 1/Eq. 3 计算频率, 挑出最高频和最低频的两个
4. **Human validation**: 3个英语语言学相关degree的annotator, 三人投票语义是否相同。三种label:
   - same meaning
   - maybe same meaning  
   - not the same meaning
   只保留三人一致认为 same 的pairs。
5. **最终规模** (Table 1): 
   - MR: 738 pairs, 平均长度 25.86 (HF) vs 25.28 (LF)
   - MT: 526 pairs, 平均长度 21.70 (HF) vs 24.78 (LF)
   - CR: 575 pairs
   - TC: 114 pairs

注意 length 是受控的: HF 和 LF 的平均长度非常接近, 防止"长短句本身就有影响"这个confounder。

Table 17 给出更细的频率分布: MR的 HF partition 有 9/526 句 frequency ∈ [4.5, 5.5], LF 是 0; LF有 60/526 在 [0.0, 1.5], HF只有 3。频谱分布两极化, 说明 paraphrase 生成确实拉开了频率差距。

---

## 5. 实验设置

### 5.1 Baselines
- **Closed-source**: GPT-4o-mini, DeepSeek-V3 (https://arxiv.org/abs/2412.19437), doubao-1.5-pro-32k
- **Open-source**: Qwen2.5-7B-Instruct (https://github.com/QwenLM/Qwen2.5), Llama-3.3-70B-Instruct (https://arxiv.org/abs/2407.21783)

### 5.2 评估指标
- MR: accuracy
- MT: chrF (https://aclanthology.org/W15-3049/), BLEU (https://aclanthology.org/P02-1040/), COMET-22 (https://aclanthology.org/2020.emnlp-main.213/)
  - chrF signature: `nword=6, nchar=6, beta=2`
  - BLEU signature: `ngram=4, weights=(0.25, 0.25, 0.25, 0.25), smoothing=method1`
- CR: accuracy
- TC: tool selection accuracy + accuracy with correct tool using

### 5.3 Fine-tuning hyperparams (Table 19)
- quantization_bit: 4 (QLoRA)
- lora_target: all
- learning_rate: 1e-4
- num_train_epochs: 10.0
- lr_scheduler: cosine
- warmup_ratio: 0.1
- bf16: true
- per_device_train_batch_size: 1, gradient_accumulation_steps: 8 (effective batch size 8)

### 5.4 Easy-to-hard baseline
用 Max Dependency Tree Depth 作 difficulty function, 用 spaCy `en_core_web_sm` (https://spacy.io/models/en#en_core_web_sm) 计算。

### 5.5 100 语言随机抽样
从 FLORES-200 100种语言里随机选, Table 20 给出按 Joshi et al. 2020 (https://aclanthology.org/2020.acl-main.560/) 的language class分布:
- Class 0 (lowest): 16
- Class 1: 46
- Class 2: 5
- Class 3: 17
- Class 4: 12
- Class 5 (highest): 4

超过一半 (62/100) 是低资源语言, 这让结论 "high-freq helps" 不是高资源语言专属。

---

## 6. 主要结果

### 6.1 Prompting on Math Reasoning (Figure 2)

| Model | Low-freq | High-freq | Δ |
|---|---|---|---|
| DeepSeek-V3 | 63.55% | 71.54% | +7.99 |
| GPT-4o-mini | 60.70% | 68.70% | +8.00 |
| Llama-3.3-70B-Instruct | 80.49% | 88.75% | +8.26 |

非常robust的 8 个百分点提升, 在三个完全不同 family 的 model 上一致。

特别有意思的 analysis: **High ∩ Low frequency** 那个set — 即在 LF 上答对的题, 在 HF 上是否还对? 答案是: **全部对**。换句话说, 用 high-frequency **只 rescue 那些原本做错的, 不会破坏原本做对的**。这是个非常强的 robustness claim, 说明 TFL 不是"瞎蒙变好", 而是"顺畅化已有能力"。

Table 18 跨模型尺寸:
- Qwen2.5-0.5b: 0.273 → 0.325
- Qwen2.5-7b: 0.595 → 0.671
- Qwen2.5-72b: 0.610 → 0.686
基本是平行的 ~7-8pt 提升, 说明这个 effect 不依赖 model scale。

Table 21 看 chain-of-thought quality (chrF / ROUGE / BERTScore vs gold CoT):
- chrF: 18.823 → 32.873 (几乎翻倍)
- ROUGE: 0.175 → 0.310
- BERTScore: 0.492 → 0.838
**CoT过程本身质量提升巨大**, 这就解释了 MR 准确率为何也提升 — 高频 input 让模型推理过程更"标准", 跟训练时见过的 reasoning 表达对齐。

### 6.2 Prompting on MT (Figure 3, Table 3)

100 种语言从 English 翻出去, 高频partition 几乎全面赢:

| Model | Metric | # improved | # improved > 1pt | # improved > 3pt | # improved > 5pt | # degraded |
|---|---|---|---|---|---|---|
| DeepSeek-V3 | BLEU | 99/100 | 63 | 31 | 12 | 1 |
| GPT-4o-mini | BLEU | 95/100 | 49 | 27 | 5 | 5 |
| DeepSeek-V3 | chrF | 100/100 | 86 | 40 | 7 | 0 |
| GPT-4o-mini | chrF | 91/100 | 75 | 34 | 2 | 9 |
| DeepSeek-V3 | COMET | 37/37 | 33 | 4 | 0 | 0 |
| GPT-4o-mini | COMET | 36/37 | 35 | 11 | 0 | 1 |

DeepSeek-V3 chrF上 100/100 语言改善! 而且 degrade 的 case 全部 < 1pt, 没有"高频反而搞砸"的counterexample。

看具体 case (Table 8/9): 比如 Belarusian (bel_Cyrl) BLEU 3.7→5.66 (+53%), Bulgarian (bul_Cyrl) 10.61→16.97 (+60%), Russian 9.42→16.08 (+71%), Ukrainian 7.91→12.1 (+53%)。这些斯拉夫语系在low-freq prompt下被严重低估, 用 common words 立马恢复应有的水平。

Figure 6 的 case study 非常直观, 拿 Serbian Cyrillic 举例:
- Original: "Two songs from the movie, ... received nominations for best original song. Lionsgate studio received 26 nominations — more than any other studio." → BLEU 0.523
- HF paraphrase: "Two tunes from the film, ... were in the running for best new tune. Lionsgate studio scored 26 nominations — more than everyone else." → BLEU 0.619
- LF paraphrase: "Two musical selections from the cinematic production, ... granted nods for the honor of best original track. Lionsgate production house secured 26 nominations surpassing all other studios." → BLEU 0.472

LF用 "cinematic production", "granted nods", "honor of", "surpassing all other studios" 这种"文学化"词, 翻译系统立刻降级。HF用 "film", "running for", "best new tune", "more than everyone else", 系统秒变流利。**这是非常典型的"措辞欧化导致翻译能力掉档"**的case, paper把这个现象量化了。

### 6.3 Prompting on Commonsense Reasoning (Table 2)

| Model | Low | High |
|---|---|---|
| GPT-4o-mini | 0.6747 | 0.6974 |
| DeepSeek-V3 | 0.7043 | 0.7235 |
| Llama-3.3-70B | 0.7530 | 0.7704 |

提升幅度比MR小 (~2-3pt), 但方向一致。

### 6.4 Prompting on Tool Calling (Table 14)

Tool Selection Accuracy:
| Model | Low | High |
|---|---|---|
| GPT-4o-mini | 0.6053 | 0.6667 |
| DeepSeek-V3 | 0.6140 | 0.6404 |
| Qwen2.5-14B | 0.6316 | 0.6667 |

Accuracy with Correct Tool Using:
| Model | Low | High |
|---|---|---|
| GPT-4o-mini | 0.4386 | 0.4912 |
| DeepSeek-V3 | 0.4649 | 0.4737 |
| Qwen2.5-14B | 0.4298 | 0.4474 |

agentic 场景也work, 这对 agent 框架设计有实际启发: tool description 的措辞要 common, 避免 over-engineered jargon。

### 6.5 Fine-tuning on MT (Table 4)

这是 fine-tuning 部分最漂亮的结果, 在 4 个低资源语言 (kea_Latn, kik_Latn, pag_Latn, lvs_Latn) 上做:

BLEU (kea_Latn):
- Original Model: 0.9346
- Fine-tuned Model (用 original FLORES-200): 4.6772
- FT on LF w/o CTFT: 4.3899
- FT on HF w/o CTFT: 5.2466 (+12.17% vs FT on Original)
- FT on HF w/ CTFT: 5.3992 (+29.96% vs FT on Original)
- Easy-to-hard baseline: 5.1674
- High-to-low baseline: 5.1179

三个关键 takeaways:

**Takeaway 1: HF partition > ground-truth** — 拿 TFPD 高频 paraphrase 当训练数据, 比拿原 FLORES-200 ground-truth 当训练数据还好。这说明很多 FLORES-200 的 reference 翻译其实措辞生僻, fine-tune 上去反而教坏模型。

**Takeaway 2: HF > LF, 一半 HF 一半 LF 也比纯 LF 强** — `FT on 1/2 LF 1/2 HF w/o CTFT` 仍然在所有语言上 beat `FT on LF w/o CTFT`。 比如pag_Latn: 3.9073 → 4.4291 (+13.35%)。 即使预算受限只能换一半数据, 也有效。

**Takeaway 3: CTFT 是best** — `FT on HF w/ CTFT` 拿到了 8/8 个最佳 metric。 在 pag_Latn 上, 从 HF-only (3.7781) → HF + CTFT (4.9102), +29.96%。
逆序对照 `High-to-low baseline` 显著差, 说明 low-to-high 的顺序方向性是真的。

### 6.6 TFD ablation (Figure 4, Figure 5)

Figure 4: 去掉 TFD, 所有 metric 都掉。 COMET score 上 DeepSeek-V3 100% 的语言对都worse without TFD。
Figure 5: TFD用的data越多, 提升比例越高, 线性趋势明显。说明 TFD 不是 noise, 是真有用的 signal。

Table 15 把 prompting + fine-tuning + TFD 一起搭: 在 CTFT-tuned model 上, 用 HF prompt 仍然比 LF prompt 更好 (BLEU kea_Latn: 0.9504 → 1.1528)。 整个 pipeline 是叠加的。

### 6.7 Correlation Analysis (Table 5, Table 6, Table 16)

Table 5: textual frequency vs complexity
| Metric | HF | LF | Δ(HF-LF) | Pearson | Spearman |
|---|---|---|---|---|---|
| MR: Max Depth | 5.02 | 5.72 | -0.70 | -0.0447 | -0.0285 |
| MT: Max Depth | 5.52 | 7.51 | -1.99 | -0.2713 | -0.2822 |
| MT: Flesch-Kincaid | 8.97 | 9.08 | -0.11 | -0.1673 | -0.1528 |

correlation 很弱 (|r| 通常 < 0.3), 说明 TFL 跟 traditional curriculum learning (easy-to-hard) 是**正交的两个 axis**。 这是paper一个重要的positioning argument。

Table 6: 按 dependency tree depth 差异分 bin, 排除 "complexity 混淆" 后, 大多数 bin 中 HF 仍 beat LF。 唯一一个反例 bin [50%-55%], N=21, 可能是noise。

Table 16: 在 non-paraphrase 的完整 MT 数据集上, 频率与 BLEU 的 Pearson correlation 在多个语言 (lao_Latn, mya_Mymr, kab_Latn, kas_Deva) 上达到 1.0 (perfect correlation)。 这是非常惊人的证据: 频率作为 translation quality 的 predictor 在某些语言上是 perfect linear。

---

## 7. 关键 Limitations & 我觉得值得追问的点

1. **Paraphrase 语义漂移**: GPT-4o-mini 生成 paraphrase 不可避免有语义微偏, 即使3个annotator都过。 Figure 1 注释里paper自己也承认这点。可以做的一个改进: 用一个 sentence embedding similarity threshold 做第二轮过滤。
2. **Assumption 2 不可证伪性**: $\varepsilon(r)$ 没有直接empirical measurement, 现有工作都只是indirect evidence。一个清晰的实验是: 在不同size的model上测 $Q_\theta(w_r) / P(w_r)$ 比值, 看 $\varepsilon(r)$ 的实际曲线。
3. **$\eta_x$ 难以估计**: contextual discrepancy 跟具体 sentence 强耦合, paper 用 axiomatically bounded 来 bypass。 可以做一个实证 ablation: 拿一层 transformer 的 attention map 看 $\eta_{x_k}$ 的实际分布。
4. **Unigram independence assumption (Assumption 4)**: 忽略 word order 让"frequency"过度简化。 一个 "United States of America" 和 "America States United of" 在该measure下频率相同, 但显然前者更高频。 可以考虑用 n-gram or neural LM PPL 替代几何平均。
5. **Length 不平衡**: Table 1 显示 HF/LF 平均长度有差异 (MT: 21.70 vs 24.78)。 虽然 paper 在 Table 6 用 tree depth 控制, 但 length 这个 confounder 在 proof 里 (Remark 7 提到 longer sentence 收紧 bound) 实际上对 HF 有利 (HF 短)。 这是一个 confounder, 没完全排除。
6. **从 loss 到 task performance 的 bridge**: paper 诚实承认不是定理。 一个更principled的解释可能要借助"learned manifold"理论: pre-training 让 model 在 high-density 区域形成 flat loss landscape (可参考 Nagarajan 2021 https://arxiv.org/abs/2106.05937, the relationship between training distribution and generalization), 所以高频输入触发更稳定的 in-distribution behavior。

---

## 8. 我能联想到的相关工作 (供你 cross-check)

- **Zipf's Law & LLM**: Mikhaylovskiy 2025 (https://aclanthology.org/2025.findings-emnlp.1026/) 证明 LLM-generated text 也 obey Zipf's law。 这给 Assumption 1 在 model output 端的依据。
- **Pre-trained models perform best when token distribution is Zipfian**: He et al. 2025 (https://aclanthology.org/2025.emnlp-main.1567/) 在 NLP, genomics, chemistry 三领域验证。
- **Prompt robustness**: Cao et al. 2024 (https://arxiv.org/abs/2411.16758, "On the worst prompt performance of LLMs") — paper 直接 motivation 来源。
- **Data quality matters**: Iskander et al. 2024 (https://aclanthology.org/2024.emnlp-main.276/), Jin & Wang 2024 (https://aclanthology.org/2024.lrec-main.1263/) — synthetic data 的 quality 直接影响下游。
- **Paraphrasing for data augmentation**: Lu & Lam 2023 PCC (https://aclanthology.org/2023.eacl-main.5/) — CTFT 灵感来源。
- **Paraphrasing for decontamination**: Zhu et al. 2024 CLEAN-EVAL (https://aclanthology.org/2024.findings-naacl.52/)。
- **Curriculum Learning for LLM**: Lu & Lam 2023 easy-to-hard curriculum 同作者 — 同一个research line。
- **Chain-of-Dictionary Prompting**: Lu et al. 2023 (https://arxiv.org/abs/2305.06575) — 跟本文用prompting调MT的思路同源。
- **TFD 类比 work**: Hinton et al. 2015 Distilling the Knowledge in a Neural Network (https://arxiv.org/abs/1503.02531) — 用 model generation 反推内部表征的思路。
- **Long-context data synthesis**: Zhu et al. 2025 (https://arxiv.org/abs/2502.15592) Generalizing from short to long — curriculum data 的另一个axis。
- **LoRA**: Hu et al. 2022 (https://arxiv.org/abs/2106.09685) — fine-tuning 部分 backbone。
- **s1: Simple test-time scaling**: Muennighoff et al. 2025 (https://arxiv.org/abs/2501.19393) — reasoning length 增长是另一个 axis。
- **DeepSeek-R1**: DeepSeek-AI 2025 (https://arxiv.org/abs/2501.12948) — RL-based reasoning, 同 baseline 系列。

---

## 9. 一句话总结

**Adam's Law** 给出了一个简洁到有点离谱的结论: **当语义等价时, 给LLM用高频措辞**, prompting 涨 ~8pt accuracy / 大幅 BLEU, fine-tuning 用 high-freq paraphrase 还能反超 ground-truth, curriculum 从低频到高频再叠加一层收益。理论证明从 Zipf's law + log-domain approximation 出发, 在 (ln rank, NLL) 半对数平面上把 token-level linear relationship 推到 sentence level, 用 worst-case error bound 给出 TFL 充分条件。 Limitation 在 Assumption 2 的强 pointwise bound 假设, 以及 unigram frequency 忽略 word order 的简化。

如果你最近在搭 agent 框架, 一个直接 actionable insight: tool description 和 system prompt 用 plain English 重写一遍, 避免任何 "leverage", "facilitate", "utilize" 这种高频书面词 → 真正高频词 "use", "help", "make"。这个改动零成本, 在 paper 的 TC 实验里准确率 +6pt。
