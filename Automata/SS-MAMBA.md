---
source_pdf: SS-MAMBA.pdf
paper_sha256: d8ac69ba1abc7435e63b2c0b9fc121eaf9f3b57bd82ebdf0d63ba6a82f97ac48
processed_at: '2026-08-12T10:22:01-07:00'
target_folder: Automata
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ss-Mamba 用人话讲

Andrej 你好，我换一种讲法，抛开公式，就讲 idea 和 intuition。

---

## 一句话版本

ss-Mamba 想做的事情就是：**训练一个模型，让它什么 time series 都能预测**——股价、气温、用电量、BTC、航空客流，统统一个模型搞定。而且你给它一个完全没见过的新 series，比如 "Platinum Price"，它也能直接预测，不用重训。

听起来像 magic，做法其实很朴素：给模型配两个"身份证"和"日历"，让它知道自己在预测谁、在什么时间预测，然后用一个跑得快的 backbone 把数据流过去。

---

## 三个零件各干啥

### 零件 1：BERT 给 series 发身份证

传统 time series 模型每条 series 自己训自己，模型根本不知道 "Gold Price" 和 "Silver Price" 是亲戚。你给它一条新 series，它两眼一抹黑。

ss-Mamba 的做法很直接：**把 series 的名字当成一句话，丢给 BERT，拿出一个向量**。BERT 已经预训练过，知道 "Gold" 和 "Silver" 语义上接近，所以这两条 series 的 embedding 天然就靠在一起。模型在训练时学到 "Gold Price 这类 series 有什么 pattern"，推断时遇到 "Platinum Price"，BERT 直接给它一个接近的 embedding，pattern 就迁移过去了。

这就是 zero-shot 的物理来源——**语义距离 = 迁移路径**。BERT 充当一个先验知识库，告诉你"哪些 series 应该长得像"。

然后这个 BERT 输出过一个 linear layer 投影到 Mamba 的 hidden size N，就成了所谓的 semantic index embedding `e`。

参考：Time-LLM 也用 LLM 当 prior，但它是用 cross-attention reprogramming，ss-Mamba 更简单，直接拿 [CLS] embedding。https://arxiv.org/abs/2310.01728

### 零件 2：KAN 用 spline 学日期特征

Time series 里 calendar effect 极其重要——周末股价不交易、冬天暖气用电多、春节零售冲一波。传统做法是把日期用 sin/cos 编码（Time2Vec 那一套），问题在于 sin 是固定周期函数，碰到 "春节每年日期漂移"、"12 月有圣诞效应但 1 月没有" 这种 asymmetric 形状，sin basis 拟合得很费劲。

ss-Mamba 换成 **B-spline**。spline 就是分段多项式，每一段是一个小 polynomial，段之间光滑拼接。每一段的形状由一组可学习系数 `α` 控制，gradient descent 直接调 `α`，让 spline 自动学出 "7 月份总是凸起" 这种形状。

具体做法：把日期拆成 7 个 feature（ordinal、year、month、day、day-of-week、day-of-year、quarter），每个 feature 单独过一条 spline，得到一个 univariate 变换后的标量，7 个标量拼起来过一个 linear layer 投影到 N 维，得到 temporal encoding vector `k`。

参数量极小：每条 spline 16 个系数，7 条 spline 一共 112 个参数，再加一个 N×7 的 linear。N=128 的话总共 1000 出头参数。对比 Transformer 一个 attention head 就几千参数，KAN 这块便宜得像白送。

B-spline 的好处是 **local support**——改一个系数只影响局部一段，gradient 稀疏稳定，不会像 global polynomial 那样一改全抖。而且 spline 的形状可以导出成解析公式，**可解释性比 neural network 强很多**——你能直接画出 "month 这个 feature 的 spline 长这样，7 月凸起 8 月回落"。

参考：KAN 原文 https://arxiv.org/abs/2404.19756 ，KAN for TS https://arxiv.org/abs/2406.08951

### 零件 3：Mamba 当 backbone

Transformer 处理长 sequence 是 O(L²)，L=3650（10 年日频）时 attention matrix 就是 1300 万个元素，显存爆。Mamba 是 SSM 家族的，recurrence 写法，复杂度 O(L)，而且 selective 机制让它能根据 input 内容动态决定"记住多少、忘掉多少"，效果接近 attention 但便宜得多。

Mamba 的核心 recurrence 就两行：

```
h_{t+1} = Ā_t · h_t + B̄_t · x_t
y_t    = C_t · h_t
```

- `h_t` 是 hidden state，类似 LSTM 的 cell state；
- `Ā_t` 是遗忘门，决定旧 state 衰减多少；
- `B̄_t` 是输入门，决定新 input 注入多少；
- `C_t` 是读出门，决定从 state 里读出什么当输出。

Mamba 的 "selective" 体现在 B̄、C、Δ 都是由小 MLP 从 input x 算出来的，input-dependent。跟传统 S4 那种固定 matrix 不一样，Mamba 能根据内容动态调整，类似 attention 的"选择性关注"但保持 linear cost。

参考：Mamba 原文 https://arxiv.org/abs/2312.00752

---

## 最核心的创新：把身份证和日历塞进 input gate

前面三个零件单独看都不算新：
- BERT 编码 series 名字，Time-LLM 类似想法有过；
- KAN 编码时间，KAN for TS 那篇 Xu et al. 做过；
- Mamba 当 TS backbone，ms-Mamba 等已经做过。

ss-Mamba 的真正创新在于**怎么把这三个零件接起来**。

最 naive 的接法是把 `e`、`k`、`x` 三个向量 concatenate 起来当 input 喂给 Mamba。但这样 sequence length 没变，input 维度变大，s_B 这个 MLP 的 input dim 也变大，参数和计算都涨。

ss-Mamba 选了一个更 elegant 的做法：**不动 input x，只 hack Mamba 的 input gate B̄**。

原始 Mamba：`B̄` 只看当前 x_t，由 `s_B(x_t)` 算出来。
ss-Mamba：`B̄' = B̄ + Broadcast(e, k)`，把 series 身份 `e` 和时间 `k` 当 bias 加到 input gate 上。

物理意义很直白：**"我正要吸收一个新观测 x_t 进 hidden state，但我得先知道这是哪个 series、现在是几月几号，再决定怎么吸收。"** 

这跟 LSTM input gate 的直觉完全一致——gate 不是固定开关，而是 conditioned on context 的动态开关。ss-Mamba 让这个 context 包含 series identity 和 calendar，gate 就变成 "series-aware + time-aware" 的。

为什么选 B̄ 而不是 Ā 或 C？我的猜测：
- 改 B̄ = 改"怎么吸收 input"，最 natural 的 context injection 点；
- 改 Ā = 改"怎么遗忘"，可能让 "Gold" 长记忆、"Stock" 短记忆，也合理但 paper 没试；
- 改 C = 改"怎么 readout"，更像 retrieval，跟 forecasting 主任务距离远一点。

paper 只做了 B̄ injection，没做 ablation 比较 Ā / C injection，这是一个 design space 没探索完的地方。

---

## Broadcast 用加法还是乘法

`B̄' = B̄ + Broadcast(e, k)` 这里用加法。直觉上：

- **加法** = 把 context 当 bias，shift gate 的工作点。类似 residual connection，minimal intervention，参数零增长；
- **乘法（FiLM 风格）** = `B̄' = B̄ ⊙ γ(e,k) + β(e,k)`，affine modulation，表达力强但要额外 MLP 算 γ 和 β；
- **concatenation** = 把 e、k 接到 x 后面再喂 s_B，input dim 变大，且 context 是 per-window 的，沿 L 维度作用不直接。

ss-Mamba 选加法是为了保住 "不增加 sequence length or model complexity" 的承诺。但表达力上不如 FiLM。这是个 trade-off，paper 没讨论。

FiLM 参考：https://arxiv.org/abs/1709.07871

---

## shape 上 paper 写得有点乱

Algorithm 1 里几个 shape 标注 inconsistent：
- 第 1 行 `e ∈ B×N`，但 Section 3.1 说 `e_s ∈ R^N`，应该 per-sample 是 N，batch 后是 B×N；
- 第 3 行 `k ∈ B×L`，但 Section 3.2 说 `k_t ∈ R^N`，所以 per-sample per-step 是 N，batch 后是 B×L×N，paper 漏了 N；
- 第 9 行 `B̄' = B̄ + Broadcast(e, k)`，broadcast 应该是 `e.unsqueeze(1) + k → B×L×N`，再 `+ B̄`。

正确 shape flow 我理解应该是：
```
e: R^N → batch 后 R^{B×N} → unsqueeze → R^{B×1×N} → broadcast 到 R^{B×L×N}
k: R^{B×L×N}
B̄: R^{B×L×N}
B̄' = B̄ + e_broadcast + k   (或者 B̄ + (e_broadcast + k))
```

paper 这块应该清一下。

---

## 跟其他 foundation TS model 比

| Model | Series identity | Temporal encoding | Backbone | Zero-shot 来源 |
|---|---|---|---|---|
| TimeGPT | implicit | sinusoidal | Transformer decoder | 大规模 pretrain |
| TimesFM | none | patch position | decoder Transformer | 大规模 pretrain |
| Chronos | none | quantile bins | T5 | 大规模 pretrain |
| Moment | learned per-series | fixed | encoder Transformer | per-series emb |
| MOIRAI | none | patch + freq | SAIT | 大规模 pretrain |
| Lag-Llama | none | lags | Llama | 大规模 pretrain |
| Time-LLM | LLM reprogram | prompt | LLM frozen | LLM 知识 |
| **ss-Mamba** | **BERT semantic emb** | **KAN/spline** | **Mamba** | **semantic distance** |

ss-Mamba 的独特之处：**唯一把 series 名字的语义当 explicit prior 拿来用的**。其他 model 要么不要 series identity（靠大规模 pretrain 隐式 capture），要么随机初始化 series embedding 让模型自己学（Moment）。ss-Mamba 直接借 BERT 的语义空间当 inductive bias，zero-shot 的物理来源最 clean。

代价是：依赖 series 名字要有语义。如果 series 名字是 "series_001" 这种没意义的 ID，BERT embedding 都是 [PAD] 一类噪声，semantic prior 就废了。所以 ss-Mamba 隐含假设 **series 名字是 natural language descriptor**。

---

## paper 的弱点

1. **没有实验数字**。所有 claim——"superior accuracy"、"robustness"、"zero-shot capability"——都没有 quantitative support。Table 1 只列了 dataset 来源，没有 RMSE 对比表。这是预印本，实验大概还在跑；
2. **Algorithm 1 shape 标注有 typo**，`k ∈ B×L` 应为 `B×L×N`；
3. **BERT frozen 与否没明说**。从 "All index embeddings are updated via gradient descent" 推测只训 W_proj + b_proj，BERT frozen。但这个 design choice 很关键，paper 该写清楚。如果 BERT 也 fine-tune，semantic prior 会被 forecasting signal wash 掉，反而失去 zero-shot 优势；
4. **只注入 B̄，没试 Ā 和 C**。design space 没探索完，可能 B̄ 不是最优 injection 点；
5. **加法 vs FiLM 没 ablation**；
6. **KAN 的 knot vector `ξ_j` 是否 trainable 没说**。如果固定 grid，对 out-of-range 日期（推断未来超出训练集日期范围）需要 extrapolation，B-spline 在 knot 外是 zero 或 linear extrapolation，长期 forecasting 可能有问题；
7. **Calendar feature `ordinal_t` 的 scale 没说怎么处理**。ordinal number 可能上百万，B-spline 对 input scale 敏感，得 normalize 到 [0,1] 再喂 spline；
8. **Baseline 写法有错**："Mamba (long short-term memory)" 是 typo，Mamba 不是 LSTM。而且缺了 Chronos、MOIRAI、PatchTST、iTransformer、TimeMixer 这几个 2024 强 baseline。

---

## 如果让我从零复现

我会这么做 minimal impl：

1. **BERT 模块**：`transformers.BertModel.from_pretrained('bert-base-uncased')`，取 `pooler_output` 或 `last_hidden_state[:, 0, :]`，**freeze 它**，只训后面的 `W_proj`；
2. **KAN 模块**：用 `scipy.interpolate.BSpline` 的 torch port，或者自己写 Cox-de Boor recursion。degree=3，num_knots=16，每条 calendar feature 一条 spline。注意 input 要 normalize 到 [0,1]，不然 spline 撑不住；
3. **Mamba backbone**：用 `mamba-ssm` 包的 `Mamba`，但要 hack 一下把 `B̄` 暴露出来加 bias。官方 CUDA kernel 把 B̄ 藏在 kernel 里，最简单是用 pure PyTorch 版的 `mamba_minimal`（Mamba 官方 repo 有），改 forward 把 B̄ 拿出来；
4. **超参**：L=120，N=128，d_BERT=768，k=7 calendar features，R=16 spline basis，m=3 cubic；
5. **训练**：AdamW，lr=1e-3（BERT frozen 所以可以大 lr），bf16 mixed precision，gradient clip 1.0。bf16 比 fp16 稳，因为 spline 系数可能跨数量级；
6. **验证**：先在 ETT-h1 + 一个 financial series 上跑 single-series；然后 joint train 5-10 条 series 测 transfer；最后 hold out 一条 series 测 zero-shot——这是最关键的 sanity check，如果 zero-shot 不比 random walk 好，semantic embedding 就没起作用。

最该先验证的 hypothesis：**zero-shot 时，BERT embedding 的语义距离是否真的对应 forecasting performance 的相似度**。比如训 Gold+Silver，测 Platinum，应该比训 Gold+Silver 测 Bitcoin 好。如果没这个 effect，整个 semantic embedding 的卖点就站不住。

---

## 跟我（Karpathy）做 nanoGPT 的直觉对照

nanoGPT 的精神是 minimal、可读、可 hack。ss-Mamba 的设计哲学其实挺像——minimal intervention（加法 bias 而非 FiLM）、不增加 sequence length、不增加 input dim、参数量小（KAN 1000 参数）。如果要把 ss-Mamba 写成 nanoSSMamba，核心 forward 可能就 50 行：

```python
# pseudo code
e = bert(series_name)[cls].detach()  # frozen
e = w_proj(e)                         # trainable
d = calendar_features(date)
k = spline_kan(d)                      # trainable
x = input_value
B = s_B(x)
C = s_C(x)
delta = softplus(delta_param + s_delta(x))
A_bar, B_bar = discretize(delta, A_param, B)
B_prime = B_bar + e.unsqueeze(1) + k   # the only novel line
y = ssm_recurrence(A_bar, B_prime, C, x)
```

唯一那行 `B_prime = B_bar + e.unsqueeze(1) + k` 就是 paper 的全部创新。其他都是 glue。这种 minimalism 我觉得是对的——容易 ablation、容易解释、容易 scale。

---

## 我对这篇 paper 的总体判断

**idea 是 clean 的，execution 还没到位**。

clean 的地方：
- semantic embedding 当 zero-shot prior，物理来源清晰；
- KAN encoder 比 sin/cos 表达力强，参数量小，可解释；
- B̄ injection 是 minimal context fusion，不增参数不增 sequence length；
- Mamba linear-time 对长 TS 友好。

没到位的地方：
- 实验数字全缺，所有 claim 都没验证；
- design space（B̄ vs Ā vs C injection，加法 vs FiLM）没探索；
- 一些关键 design choice（BERT frozen、knot trainable、ordinal normalization）没说清；
- baseline 不全，且有一个 typo。

作为预印本 idea paper 可以读，作为 finished work 不行。如果作者补上实验 + ablation，这会是一篇不错的 foundation TS model paper。核心 contribution 是 "把 semantic 和 temporal context 通过 residual bias 注入 SSM input gate" 这个 minimal fusion pattern，这个 pattern 本身可以推广到其他 SSM 变体，甚至推广到 Transformer 的 attention bias（类似 ALiBi 但 content-dependent）。

参考链接汇总：
- Mamba: https://arxiv.org/abs/2312.00752
- KAN: https://arxiv.org/abs/2404.19756
- Time-LLM: https://arxiv.org/abs/2310.01728
- TimesFM: https://arxiv.org/abs/2310.10688
- Chronos: https://arxiv.org/abs/2403.07815
- MOIRAI: https://arxiv.org/abs/2402.02592
- Moment: https://arxiv.org/abs/2402.03685
- PatchTST: https://arxiv.org/abs/2211.14730
- iTransformer: https://arxiv.org/abs/2405.14011
- Time2Vec: https://arxiv.org/abs/1907.05321
- FiLM: https://arxiv.org/abs/1709.07871

---

# ss-Mamba 论文讲解：Semantic-Spline Selective State-Space Model

Andrej 你好，这篇 ss-Mamba 由 NCCU 的 Zuochen Ye 在 2025 年 6 月提交，主轴是用一个 single foundation model 跨 domain 预测日频 time-series。技术栈三件套：BERT semantic embedding + KAN-spline temporal encoder + Mamba SSM backbone，其中最有意思的是把 semantic 和 temporal context 注入到 SSM 的 input matrix B̄。下面我把它逐层拆开，并指出几个我自己觉得 intuition 应该建在哪里的点，以及 paper 里有点粗糙的地方。

---

## 1. 整体 architecture intuition

整篇 paper 想做的事情其实很清晰：**一个 single set of parameters 同时建模 heterogeneous 的 daily-frequency series**，并且能 zero-shot 跳到没见过的 series。要做到这一点，必须解决两个问题：

1. **static metadata** 怎么进到模型——series 是 "Gold Price" 还是 "Taipei Temperature"，这决定了它该 follow 哪一类 pattern；
2. **dynamic calendar effect** 怎么建模——holiday、seasonality、business cycle，这些是 non-stationary 且跨 domain 差异巨大。

ss-Mamba 的答案是：
- 用 **pretrained BERT + linear projection** 把 series 的名字 string 编码成 series-specific memory vector `e`（formula 1–3）；
- 用 **KAN / B-spline** 把 calendar features 编码成 per-time-step vector `k`（formula 4–8）；
- 把 `e` 和 `k` 同时 broadcast 进 Mamba 的 input gate `B̄`，得到 context-aware 的 `B̄'`（formula 9–11）。

整套 forward 见 Algorithm 1，shape 流如下：

```
s (string)  ─► f_BERT ─► h^(BERT) ∈ R^d_BERT ─► W_proj ─► e ∈ R^N  (broadcast → B×L×N)
t (date)    ─► D(t)   ─► d ∈ R^k           ─► KAN/spline + Linear ─► k ∈ R^N  (→ B×L×N)
x (value)   ─► s_B, s_C, s_Δ ─► B,C,Δ ─► Discretize ─► Ā, B̄

B̄' = B̄ + Broadcast(e, k)
y  = SSM(Ā, B̄', C)(x)
```

直觉上：Mamba 的 `B̄` 是 input gate，控制"当前 token 往 hidden state 里灌多少"。ss-Mamba 让这个 gate 在灌之前先被 series identity 和 time context "偏置"一下——这就把 static 信息直接写进了 recurrence 的入口，而 sequence length 没增加。

---

## 2. Semantic Index Embedding（§3.1）

### 2.1 公式逐项拆解

**Formula 1**: `h_s^(BERT) = f_BERT(n_s) ∈ R^{d_BERT}`

- `s ∈ S`：dataset 中第 s 条 time series；
- `n_s`：这条 series 的自然语言 identifier，例如 "S&P 500"、"Taipei Temperature"；
- `f_BERT(·)`：BERT 的最后一层 hidden state，作者明确说取 **[CLS] embedding**；
- `d_BERT`：BERT 最后一层维度，标准 BERT-base 是 768，BERT-large 是 1024；
- `h_s^(BERT)`：semantic vector，表征 series 名字的语义。

**Formula 2**: `e_s = W_proj · h_s^(BERT) + b_proj, W_proj ∈ R^{N×d_BERT}`

- `N`：Mamba 内部 hidden size，是 backbone 的 state 维度；
- `W_proj, b_proj`：可训练，做 dimension alignment；
- `e_s ∈ R^N`：最终送入 backbone 的 semantic index embedding。

**Formula 3** 把整个 pipeline 写成复合函数：`f_IDX(n_s) = W_proj · f_BERT(n_s) + b_proj`，作者把这个映射叫做 **"index space"**，并把它当 "memory cell"——每个 series 对应 index space 里的一个点，BERT 给出 semantic prior，linear projection + downstream gradient 把这个点 fine-tune 到对 forecasting 任务有用的位置。

### 2.2 Intuition：为什么是 BERT 而不是 one-hot

如果用 one-hot series id，新 series 就完全是 OOD，必须重训 embedding。用 BERT 编码名字字符串，新 series "Platinum Price" 通过 BERT 自然就落在 "Gold Price" 和 "Silver Price" 附近——这是 zero-shot 的物理来源：semantic distance 提供了 inductive bias，模型不需要重新学 "Platinum 是贵金属"。

这里有一个 design choice 我觉得 paper 没说清：**BERT 是 frozen 还是 fine-tuned**？看 Algorithm 1 第 1 行 `e ← f_BERT(s)` 直接当成一个 forward op，并没说梯度是否回传到 BERT。从 §3.1 末尾 "All index embeddings are updated via gradient descent" 看，作者主要强调 W_proj + b_proj 可训练；BERT 本身 frozen 是更合理的做法（否则 pretrain 的 semantic prior 会被 forecasting signal 洗掉，类似 Time-LLM 的 reprogramming 思路）。但 paper 没明说。

相关 reference:
- Time-LLM (reprogramming LLM for TS): https://arxiv.org/abs/2310.01728
- Lag-Llama (foundation TS with lags): https://arxiv.org/abs/2310.08578
- MOIRAI (Salesforce uni2ts): https://arxiv.org/abs/2402.02592

### 2.3 跟 TimesFM / Chronos / Moment 的对比

| Model | Series identity 处理 | Temporal encoding | Backbone |
|---|---|---|---|
| TimeGPT (Nixtla) | implicit | sinusoidal | Transformer decoder |
| TimesFM (Google) | none (per-series token via patching) | patch position | decoder-only Transformer |
| Chronos (Amazon) | categorical tokenization | quantile bins | T5 |
| Moment (CMU) | per-series learned emb | fixed | encoder Transformer |
| **ss-Mamba** | **BERT semantic emb + trainable proj** | **KAN/B-spline** | **Mamba SSM** |

ss-Mamba 唯一把 "series 名字的语义" 显式拿来当 prior 的，其他要么不要 series identity（TimesFM/Chronos），要么随机初始化然后让模型自己学（Moment）。这让它对 unseen series 的 generalization 直觉上更 clean。

---

## 3. KAN-based Temporal Encoder（§3.2）

### 3.1 Calendar descriptor vector

**Formula 4**: `d_t = [ordinal_t, year_t, month_t, day_t, dow_t, doy_t, quarter_t]^T ∈ R^k`

- `ordinal_t`：日期的 ordinal number（从某个 epoch 起的天数），捕捉 absolute trend；
- `year_t, month_t, day_t`：分解后的日历分量；
- `dow_t`：day of week，捕捉周季节性；
- `doy_t`：day of year，捕捉年季节性；
- `quarter_t`：季度，财政/商业周期；
- `k`：feature 数，这里 k=7。

这个 raw feature vector 直接用 sine/cosine 编码是 Time2Vec 之前的 common practice；问题在于 sine/cosine 是 fixed 周期 basis，对 **non-stationary** 或者 **多周期叠加 + phase shift**（如农历春节、Easter、闰年）拟合力不够。KAN 的卖点就是 spline 可以学 arbitrary univariate shape。

### 3.2 B-spline 单变量变换

**Formula 5**: `g_j(x) = Σ_{r=1}^{R} α_{j,r} B_{r,m}(x; ξ_j)`

- `j`：第 j 个 calendar feature（j ∈ {1,...,k}）；
- `r`：spline basis 的 index，从 1 到 R，R 是 basis 数；
- `m`：B-spline 的 degree（paper 推荐 m ≤ 3，即 cubic spline）；
- `B_{r,m}(x; ξ_j)`：第 r 个 B-spline basis function，degree m，knot vector `ξ_j`；
- `α_{j,r}`：可训练系数，是整个 KAN encoder 的可学习参数；
- `ξ_j`：knot 向量，作者没说是不是 trainable（一般 KAN 设定里是固定的 grid，可学习的是 α）。

B-spline basis 由 Cox-de Boor recursion 定义：

```
B_{r,0}(x) = 1  if ξ_r ≤ x < ξ_{r+1}, else 0
B_{r,m}(x) = (x - ξ_r)/(ξ_{r+m} - ξ_r) · B_{r,m-1}(x)
           + (ξ_{r+m+1} - x)/(ξ_{r+m+1} - ξ_{r+1}) · B_{r+1,m-1}(x)
```

B-spline 的关键性质：
- **Local support**：每个 `B_{r,m}` 只在 `[ξ_r, ξ_{r+m+1}]` 非零，意味着调一个 `α_{j,r}` 只影响局部 shape，gradient sparse 且 stable；
- **Partition of unity**：`Σ_r B_{r,m}(x) = 1`，所以 g_j 是 α 的 convex combination；
- **Smoothness C^(m-1)**：m=3 时是 C²，导数连续，对 time series 的连续性 prior 友好。

### 3.3 KAN 的 Kolmogorov-Arnold 定理背景

Kolmogorov-Arnold representation theorem：

```
f(x_1, ..., x_n) = Σ_{q=0}^{2n} Φ_q( Σ_{p=1}^n φ_{q,p}(x_p) )
```

任意 continuous multivariate function 都能写成 "外层 1D sum、内层 1D function of single variable" 的组合。KAN 的 insight 是：把传统 MLP 的"固定 activation + 可学习 weight"翻转成"可学习 activation (用 spline 实现) + 固定 weight"。在 ss-Mamba 这里只用了 **univariate** 那一支，没做 outer sum，所以严格说不是 full KAN，更接近 **spline-encoded Time2Vec**。但效果上：每个 calendar feature 用一个 spline 学一个 univariate 非线性，再 linear mix。

KAN paper: https://arxiv.org/abs/2404.19756

### 3.4 Linear mixing → TEV

**Formula 6**: `u_{t,j} = g_j(d_{t,j})`，`u_t ∈ R^k` 是 k 个 spline 输出拼起来。

**Formula 7**: `z_t^(TEV) = σ(W · u_t + b) ∈ R^N`

- `W ∈ R^{N×k}`：可训练 linear map；
- `b ∈ R^N`：bias；
- `σ`：作者说 tanh 或 identity；
- `z_t^(TEV)`：Temporal Encoding Vector，shape 为 N。

**Formula 8** 就是 alias：`k_t = z_t^(TEV) ∈ R^N`。

### 3.5 实际超参推荐

paper §3.2 末尾给了 modest setting：`m ≤ 3, R ≤ 16, N ∈ [64, 128]`。换算下来每个 calendar feature 的 KAN 参数量是 R=16，7 个 feature 共 112 个 α，再加 W (N×7) + b (N) ≈ 8N + 112，N=128 时大概 1136 个参数——非常轻量。这就是 KAN 相对 Time2Vec 的优势：参数量极少但表达力强。

对比 Time2Vec (Kazemi et al. 2019, https://arxiv.org/abs/1907.05321)：
```
Time2Vec(t) = [w_0·t + b_0, sin(ω_1·t + φ_1), ..., sin(ω_k·t + φ_k)]
```
固定 periodic basis，对 phase shift 不友好。KAN encoder 的 spline 可以拟合任意单变量函数，包括 "7 月份总是凸起" 这类 asymmetric seasonal effect，而 sin basis 必须叠加多个 harmonic 才能近似。

---

## 4. Selective State-Space Backbone（§3.3）

这是 paper 的核心创新点：**context-injected input matrix B̄'**。

### 4.1 原始 Mamba SSM 回顾

Mamba (Gu & Dao 2023, https://arxiv.org/abs/2312.00752) 的连续 SSM：

```
h'(t) = A·h(t) + B·x(t)
y(t)  = C·h(t)
```

离散化（zero-order hold with step Δ）：

```
Ā = exp(Δ·A)
B̄ = (Δ·A)^{-1}·(exp(Δ·A) - I)·Δ·B    (exact ZOH)
  ≈ Δ·B                              (简化，Mamba 实际用 bilinear)
```

Mamba 实际用 **bilinear discretization**：

```
Ā = (I + Δ/2 · A)·(I - Δ/2 · A)^{-1}
B̄ = (I - Δ/2 · A)^{-1} · Δ·B
```

得到离散 recurrence：

**Formula 9**: `h_{t+1} = Ā_t · h_t + B̄_t · x_t`

- `h_t ∈ R^N`：hidden state；
- `Ā_t`：discretized transition matrix，控制遗忘；
- `B̄_t`：discretized input matrix，控制输入注入；
- `x_t`：当前 input token（已 projected 到 N 维）。

Mamba 的 "selective" 关键：**A 是 input-independent**（保持为可训练 parameter，类似 S4 的 HiPPO matrix 或 diagonal initialization），**B, C, Δ 都 input-dependent**（通过小 MLP `s_B, s_C, s_Δ` 从 x 算出来）。这让模型能根据 input 内容动态决定"吸收多少、读出多少、步长多大"。

### 4.2 Context injection：ss-Mamba 的核心改动

paper §3.3 的核心 idea 是：B̄ 在原 Mamba 里只依赖当前 x_t，**全局 context（series identity e 和 calendar k）没有直接通道**。ss-Mamba 把它们 broadcast 加进 B̄：

**Formula 10**: `B̄' = B̄ + Broadcast(e, k)`

- `e ∈ R^N`（实际是 `R^{B×N}`，per-sample series embedding）；
- `k ∈ R^{B×L×N}`，per-sample per-time-step temporal encoding；
- Broadcast：`e` 沿 L 维度 broadcast 到 `R^{B×L×N}`，与 `k` 相加，再与 `B̄ ∈ R^{B×L×N}` 加。

得到 context-aware input matrix `B̄' ∈ R^{B×L×N}`。

**Formula 11**: `h_{t+1} = Ā_t · h_t + B̄_t' · x_t`

**Formula 12**: `y_t = C_t · h_t`

**Formula 13**: `y = SSM(Ā, B̄', C)(x)`

### 4.3 为什么注入 B 而不是 A 或 C

这是 paper 没讲但我认为最该 build intuition 的地方。SSM 的三个 matrix 语义：

| Matrix | 语义 | 加 context 的含义 |
|---|---|---|
| **Ā** | 遗忘门 | 不同 series 有不同 decay rate |
| **B̄** | 输入门 | 不同 series / 不同 time 吸收 input 的方式不同 |
| **C** | 读出门 | 不同 series 从 hidden state 读出的视角不同 |

作者选 B 的物理意义是：**"我现在要吸收一个新观测 x_t，但我得先知道这是哪个 series、什么时间，再决定怎么吸收"**。这跟 LSTM 的 input gate 思路一致。

但其他选择也合理：
- 加到 Ā：让 series "Gold" 比 "Stock" 有更长 memory——直觉上 commodity 价格有 momentum，股票均值回归；
- 加到 C：让 series-specific 决定怎么 readout——更接近 retrieval 风格；
- 三者都加：最强表达力，但参数上升。

paper 只 demo 了 B 注入，ablation 没说试过其他。这是直觉上 design space 没充分 explored 的地方。

### 4.4 Broadcast 的具体 shape（这里 paper 有点乱）

Algorithm 1 的 shape 标注有点 inconsistent：
- 第 1 行：`e ∈ B×N ← f_BERT(s)`（注意是 B×N，不是 B×N×L）
- 第 3 行：`k ∈ B×L ← f_KAN(d)`（写成 B×L，但 Section 3.2 说 k_t ∈ R^N，所以应该是 B×L×N）
- 第 9 行：`B̄' = B̄ + Broadcast(e, k)`

我理解的正确 shape 应该是：
- `e ∈ R^{B×N}` 或更精确地说 `R^N`（series-specific，batch 内同 series 共享）
- `k ∈ R^{B×L×N}`（time-step-specific）
- broadcast：`e.unsqueeze(1) + k → R^{B×L×N}`，再 `+ B̄ ∈ R^{B×L×N}`

paper 这里写得有点 loose，建议作者改稿时明确。Algorithm 1 第 3 行 `k ∈ B×L` 应该是笔误，缺了 N。

### 4.5 选择 broadcast = 加法的设计

`B̄' = B̄ + Broadcast(e, k)` 用加法而不是 multiplication 或 concatenation。直觉：
- **加法**：把 context 当 bias，shift 整个 input gate 的"工作点"，类似 residual；
- **乘法**（FiLM 风格, Perez et al. 2018, https://arxiv.org/abs/1709.07871）：`B̄' = B̄ ⊙ γ(e) + β(e)`，做 affine modulation，表达力更强但参数多；
- **concatenation**：把 context 接到 x 后面再喂给 s_B，会增加 s_B 的 input dim，且 context 是 per-window 的不能很好沿 L 维度作用。

加法是 minimal intervention，对应 paper §3.3 "without increasing sequence length or model complexity" 的承诺。但表达力上不如 FiLM。这是个 trade-off paper 没讨论。

---

## 5. Algorithm 1 完整 forward 解析

把 Algorithm 1 行号对照着看：

```
Input: {x, s, t}    # x: 值序列, s: series name string, t: timestamp

1. e ∈ B×N ← f_BERT(s)              # BERT 编码 series 名 → semantic embedding
                                     # shape: 实际应为 N 或 B×N（同 series 共享）
2. d ← D(t)                          # 提取 calendar features (k 维)
3. k ∈ B×L ← f_KAN(d)               # KAN 编码 calendar → temporal embedding
                                     # shape: 实际应为 B×L×N
4. A ∈ R^N ← Parameter               # state transition，input-independent
5. B ∈ R^{B×L×N} ← s_B(x)           # input matrix，input-dependent
6. C ∈ R^{B×L×N} ← s_C(x)           # output matrix，input-dependent
7. Δ ∈ R^{B×L} ← τ_Δ(Parameter + s_Δ(x))   # step size，input-dependent
8. Ā, B̄ ∈ R^{B×L×N} ← Discretize(Δ, A, B)   # ZOH/bilinear 离散化
9. B̄' = B̄ + Broadcast(e, k)         # ★ ss-Mamba 的核心创新
10. y ← SSM(Ā, B̄', C)(x)             # 跑 SSM recurrence
11. return y
```

几个观察：
- A 是 `R^N` 不是 `R^{N×N}`——说明作者用 **diagonal SSM**（Mamba 默认就是 diagonal，A 是 N 维向量代表对角元）。这避免了 N² 的 transition matrix，让 recurrence 是 element-wise，O(N·L)。
- Δ 写成 `R^{B×L}` 但实际操作中要 broadcast 到 N 维（每个 channel 各自的 step size 或共享一个 step size 都可，paper 没明确）。
- `Parameter + s_Δ(x)` 是 Mamba 论文里的 trick：step size = bias + data-dependent shift，让 Δ 有一个 learnable baseline 加上 input-driven modulation。`τ_Δ` 是 softplus 一类保证正的激活。

---

## 6. 实验设计（§4）

### 6.1 Dataset

paper 给的是 dataset 类别表（Table 1），但**没有给具体的实验数字**——这是这篇 paper 最大的弱点。Table 1 只列了 series 来源：

| Field | Series examples |
|---|---|
| Climate | Temperature, Rain |
| Consumption | Market sales |
| Economic | Interest, Unemployment, Inflation, Cycle, Currency |
| Financial | Gold, S&P500, Household |
| Energy | Oil, Electricity |
| Forecast | Expectation (institute-produced) |
| Transport | Air passenger |
| Package | ETT, HAR (paperwithcode benchmark) |
| Company | Apple, Tesla |
| Regional | US |
| Crypto | BTC, ETH |

这个 dataset 覆盖度其实挺好——跨 11 个 domain、混日频与高频（ETT 是分钟级，HAR 是人体活动识别）。但**没有 sample size、序列长度、训练/验证/测试切分**，难以 reproducibility。

ETT 是 Informer paper 的 benchmark（https://arxiv.org/abs/2012.07436），HAR 是 UCI Human Activity Recognition。这两是 TSF / classification 的 standard benchmark。

### 6.2 Evaluation protocol

paper 设计了 5 类 task：
1. **Single-Series Forecasting**：跟 ARIMA / LSTM / Transformer 比；
2. **Multi-Series Joint Training**：测 cross-domain transfer；
3. **Generalization to Unseen Series**：zero-shot，只给 series 名 + time encoding；
4. **Ablation**：去 semantic 或换 KAN 为简单 encoder；
5. **Elastic Context Window**：变 input window L ∈ {30, 60, 120} 测长 context 利用。

Evaluation metric：RMSE。Sliding window L ∈ {30, 60, 120}，one-step-ahead forecasting。

### 6.3 Baselines

- Mamba (作者写 "long short-term memory"——这是 typo，Mamba 不是 LSTM)
- Transformer (self-attention)
- TimesFM (Google, https://arxiv.org/abs/2310.10688)

缺了几个 important baselines：
- **Chronos** (Amazon, https://arxiv.org/abs/2403.07815)：tokenize TS 然后用 T5
- **MOIRAI** (Salesforce)：univariate foundation model with mixture distribution
- **TimeGPT** (Nixtla, https://arxiv.org/abs/2310.03589)：商业 API，paper §1 引用过
- **Lag-Llama**：开源 foundation TS
- **PatchTST** (https://arxiv.org/abs/2211.14730)：patching + Transformer，强 baseline
- **iTransformer** (https://arxiv.org/abs/2405.14011)：channel-as-token，2024 ICLR 强 baseline
- **TimeMixer** (https://arxiv.org/abs/2405.14616)：multi-scale MLP

而且**没有给实验数字**——这是 paper 的硬伤，应该是预印本阶段，实验还在跑。

---

## 7. 我对这篇 paper 的几点直觉判断

### 7.1 优点

1. **Semantic embedding 当 prior 是合理的**：zero-shot 物理来源清晰，BERT pretrained space 提供 inductive bias；
2. **KAN encoder 比 sine/cosine 强**：local support + partition of unity + 可拟合 arbitrary univariate，参数量极小；
3. **B̄ 注入 context 是 elegant minimal intervention**：不增加 sequence length，不增加 s_B 输入维度，只是 gate 上加一个 bias；
4. **Linear-time backbone 对长 TS 友好**：日频 10 年 = 3650 step，Mamba 是 O(L) 而 Transformer 是 O(L²)，L=120 时差别不大但 L=1000+ 时显著。

### 7.2 弱点 / 不确定的地方

1. **没有实验数字**：所有 claim 都没有 quantitative support，ablation "去 semantic 性能下降多少" 完全没数据；
2. **Algorithm 1 shape 标注 inconsistent**：`k ∈ B×L` 应为 `B×L×N`，`e ∈ B×N` 实际是 `N` 或 `B×N`；
3. **BERT frozen 与否没说**：这个 design choice 很关键，影响 semantic prior 保留多少；
4. **只注入 B 没试 Ā 和 C**：design space 没探索完，可能 B 不是最优；
5. **加法 vs FiLM**：context modulation 用加法 vs affine 是 trade-off，没 ablation；
6. **KAN 的 knot vector `ξ_j` 是否 trainable 没说**：如果固定 grid，对 out-of-range date（推断未来）需要 extrapolation——B-spline 在 knot 外是 zero 或 linear extrapolation，可能对长期 forecasting 有问题；
7. **Calendar feature `ordinal_t` 的 scale**：B-spline 对 input scale 敏感，ordinal number 可能上百万，需要 normalize 到 [0,1] 再喂 spline，paper 没说预处理；
8. **Baseline 写法有错**："Mamba (long short-term memory)" 显然是 typo。

### 7.3 跟我（Karpathy）做 nanoGPT / micrograd 的直觉对照

如果让我从 zero 写一个 minimal ss-Mamba：
- BERT 模块直接用 transformers 库的 `BertModel.from_pretrained('bert-base-uncased')`，取 `pooler_output` 或 `last_hidden_state[:, 0, :]`，freeze 它；
- KAN 模块用 `torch-scatter` 或自己写 Cox-de Boor；更快的做法是用 `scipy.interpolate.BSpline` 转 torch；
- Mamba backbone 直接用 `mamba-ssm` 包的 `Mamba`，但要 hack 一下把 `B̄` 暴露出来加 bias——官方实现把 B̄ 藏在 CUDA kernel 里，需要 fork 或用 pure PyTorch 版的 `mamba_minimal`；
- L 选 60 或 120，N 选 128，d_BERT 768，k=7，R=16，m=3；
- 优化器 AdamW，mixed precision bf16（bf16 对 spline 参数稳定，因为 spline 系数可能跨数量级），gradient clipping 1.0。

### 7.4 跟 foundation TS model 大 family 的关系

ss-Mamba 在 foundation TS model 谱系里的位置：

```
Tokenize 路线      │ Chronos (categorical bins → LLM)
                  │ TimesFM (patch → decoder-only)
─────────────────┼──────────────────────────────────
Reprogram 路线    │ Time-LLM (cross-attention reprogramming)
                  │ TEMPO (prompt-tuning)
─────────────────┼──────────────────────────────────
Native backbone   │ TimeGPT (Transformer)
                  │ Moment (Transformer encoder)
                  │ MOIRAI (SAIT + mixture)
                  │ Lag-Llama (Llama + lags)
                  │ ss-Mamba ★ (Mamba + semantic + KAN)
```

ss-Mamba 是 **native backbone** 路线里第一个把 semantic name 当 explicit prior 的，且 backbone 用 linear-time SSM 而非 Transformer。这是个 niche 但 reasonable 的 design。

---

## 8. 相关 reference 汇总

### Foundation TS models
- TimeGPT: https://arxiv.org/abs/2310.03589
- TimesFM: https://arxiv.org/abs/2310.10688
- Chronos: https://arxiv.org/abs/2403.07815
- MOIRAI: https://arxiv.org/abs/2402.02592
- Moment: https://arxiv.org/abs/2402.03685
- Lag-Llama: https://arxiv.org/abs/2310.08578
- Time-LLM: https://arxiv.org/abs/2310.01728
- TEMPO: https://arxiv.org/abs/2310.01062

### Mamba / SSM
- Mamba: https://arxiv.org/abs/2312.00752
- S4: https://arxiv.org/abs/2111.00396
- S5: https://arxiv.org/abs/2208.04933
- Hyena: https://arxiv.org/abs/2302.10866
- ms-Mamba (multi-scale Mamba for TSF): https://arxiv.org/abs/2504.19024
- Mamba for TS survey: https://arxiv.org/abs/2404.18772

### KAN
- KAN original: https://arxiv.org/abs/2404.19756
- KAN for TS (Xu et al.): https://arxiv.org/abs/2406.08951
- KAN for TS classification (Barašin et al.): https://arxiv.org/abs/2411.01899

### Temporal encoding
- Time2Vec: https://arxiv.org/abs/1907.05321
- Time Features (N-BEATS-G): https://arxiv.org/abs/1905.10437
- TFT (Temporal Fusion Transformer): https://arxiv.org/abs/2106.12950

### BERT & pretrained embeddings
- BERT: https://arxiv.org/abs/1810.04805
- Word2Vec: https://arxiv.org/abs/1301.3781

### Strong TSF baselines
- PatchTST: https://arxiv.org/abs/2211.14730
- iTransformer: https://arxiv.org/abs/2405.14011
- TimeMixer: https://arxiv.org/abs/2405.14616
- Informer (ETT benchmark): https://arxiv.org/abs/2012.07436

### Mixed precision / training stability
- Mixed precision: https://arxiv.org/abs/1710.03740
- Gradient clipping for RNN: https://arxiv.org/abs/1211.5063

---

## 9. 一句话总结直觉

**ss-Mamba = "BERT 编码 series 名字 + KAN 编码日期" 通过 residual 加法 bias Mamba 的 input gate `B̄`**，让 input gate 同时知道"我在预测谁"和"我在哪一天"。这个 bias 形式 minimal、不增加 sequence length，配合 linear-time SSM backbone 对 long context 友好，semantic embedding 提供 zero-shot 物理来源。整体 idea clean、参数量小、可解释；但 paper 缺实验数字、缺 A/C injection ablation、缺 BERT frozen/fine-tune 的明确说明，作为预印本还有很多该补的。

如果你接下来想自己 reproducibility 或 follow-up，最低成本的 minimal impl 是：freeze `bert-base-uncased` 取 [CLS] → linear → N；用 `scipy.interpolate.BSpline` 的 torch port 实现 KAN；用 `mamba-ssm` 包的 pure PyTorch fallback 把 `B̄` 暴露出来加 bias；先在 ETT + 一个 financial dataset 上验证 semantic zero-shot 是否真的 work。这是 build intuition 最快的路径。
