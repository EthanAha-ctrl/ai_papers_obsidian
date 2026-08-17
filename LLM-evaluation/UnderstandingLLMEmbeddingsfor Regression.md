---
source_pdf: UnderstandingLLMEmbeddingsfor Regression.pdf
paper_sha256: 595f8911d25d8fe0ae7b7e6be746a13e16fba7fcb29cdc27bb5920fd08858d0b
processed_at: '2026-08-12T19:20:37-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

## 一句话说清楚

大家都在折腾怎么让 LLM 做 regression——让它直接吐数字。但这篇 paper 说: 别费劲了，**就把 LLM 当成一个 feature extractor，拿它的 embedding 接个小 MLP，反而又便宜又好用**，尤其在 input 维度高的时候。

## 故事背景

LLM 做 regression 有两条路:

**路线 A (decoding)**: 让 LLM 直接 generate "3.14159" 这样的 token。问题是 token 是离散的, 1234.5 和 1234.6 在 token 层面可能差很远 ("5" vs "6"), 但在 numeric 上差 0.1。LLM 的 softmax 分布在 fine-grained numeric 上很钝。这条路 cost 高 (要 fine-tune 或 ICL), 且精度有天花板。参考 [OmniPred](https://arxiv.org/abs/2402.14547) 和 [Vacareanu et al.](https://arxiv.org/abs/2404.07544)。

**路线 B (embedding)**: 把 input string 喂进 frozen LLM, 拿最后一层 hidden states average-pool 一下, 得到一个 fixed-length vector, 然后接个两层 MLP 训 regression。LLM 不动, 只训 MLP, cost 极低。这篇 paper 走的就是这条路。

听起来路线 B 太 trivial 了——谁不会接 MLP? 但核心问题是: **LLM 的 embedding 空间长什么样? 能不能用来做 regression?**

Figure 1 给了个下马威: 把一个最简单的 5D Sphere function 的 inputs 过 Gemini embedding (6K 维) 再 t-SNE 到 2D, 表面 rugged 得像月球表面。这怎么看都不像能 regression 的样子。

所以 paper 的核心贡献是回答: **为什么这个 rugged 的表面实际上还是 work 的? 什么时候 work? 为什么?**

---

## 三个核心发现

### 发现 1: 高维任务上, LLM embedding 打传统特征工程

传统 tabular regression 怎么搞 feature? 连续值 normalize, categorical 做 one-hot。DOF (degree of freedom, 就是 input 参数个数) = 5 的时候没问题。但 DOF = 100 呢? One-hot 爆炸, Euclidean distance 在高维空间所有点都差不多远 (curse of dimensionality), XGBoost 和 MLP 都开始 degrade。

LLM embedding 的优势在这里就显出来了: 不管你 DOF 是 5 还是 100, 它都把 input 压成固定 6K 维的 dense vector。维度不随 DOF 增长, 信息被 compact 了。

Figure 2 的图非常直观: 横轴 DOF 从 5 到 100, 纵轴 Kendall-Tau (ranking correlation, 越高越好)。传统方法 (XGBoost, MLP on traditional features) 是往下掉的曲线, LLM embedding (T5-XXL, Gemini) 是平的, 甚至略升。

real-world 数据 (Table 1) 也印证: DOF=4 的 Init2Winit, LLM embedding 只在 6-19% 的 task 上赢传统方法; DOF=29 的 AutoML, LLM 在 30-41% 的 task 上赢; DOF=35 的 XLA, 17-29%。

**直觉**: 低维任务, 传统特征已经够 informative, LLM 的语义 prior 反而是 noise。高维任务, 传统特征 sparse 到爆炸, LLM 的 dense compression 优势就出来了。

有个有趣的副发现 (Figure 14): 对 traditional features, XGBoost > MLP (经典 tabular 结论); 但对 LLM embedding, MLP > XGBoost。因为 LLM embedding 是 dense smooth manifold, tree-based 的 splits 在 6K 维 dense 空间上 overfit, ReLU MLP 的 piecewise linear 反而更适配。

### 发现 2: LLM embedding 保持 Lipschitz continuity——这才是它 work 的真正原因

这是 paper 最有 depth 的部分。

**问题**: tokenization 是离散的。"1.234" 和 "1.567" numeric 上差 0.333, 但 token-wise 完全不同 ("234" 和 "567" 不 share)。所以 LLM embedding 空间上, 这两个点可能离得很远。Continuity 似乎 broken。

**但实验发现**: LLM embedding 空间实际上很 smooth。作者用 **Lipschitz factor** 来量化:

$$L(x, x') = \frac{\|f(x) - f(x')\|}{\|\phi(x) - \phi(x')\|}$$

- $f$: objective function
- $\phi$: embedder
- $L(x, x')$: 在 embedding 空间移动单位距离, objective 值变化多少
- $L$ 小 = smooth, $L$ 大 = rugged

**直觉**: 你想象 embedding 空间是一片地形, objective function 是地形上每个点的高度。Lipschitz factor 就是地形的 "坡度"。坡度小, MLP 这种 piecewise linear 的函数容易拟合; 坡度大且不规则, MLP 就抓瞎。

作者定义了 **NLFD (Normalized Lipschitz Factor Distribution)**:
1. 对 dataset 里所有 embedding 做 batch normalize
2. 对每个点找 nearest neighbor, 算 Lipschitz factor
3. 除以 $\sqrt{d}$ 归一化 (让不同 embedding dim 可比)
4. 收集成一个 distribution

然后看这个 distribution 的形状。Figure 3 显示: 当 LLM embedding 比 traditional 更好时, LLM 的 NLFD 更左偏 (更多 small Lipschitz factors, 更 smooth); traditional 更好时, 反过来。

**最关键的 quantitative result** (Table 2): NLFD gap 的 Z-score 和 regression performance gap 的 Pearson correlation 在 0.60-0.88 之间, across 所有 BBOB functions, 所有 model sizes, 所有 DOF。

$$Z = \frac{\mu_{\phi_{\text{trad}}} - \mu_{\phi_{\text{LLM}}}}{\sqrt{\sigma_{\phi_{\text{trad}}}^2 + \sigma_{\phi_{\text{LLM}}}^2}}$$

这个 0.6-0.88 的 correlation 非常强。意思是: **embedding 的 smoothness 几乎线性预测 regression 性能**。Smoothness 是 regression 成功的 sufficient condition。

**为什么 LLM embedding smooth?** 作者没完全回答, 但给出 hints: transformer 的多层 attention + LayerNorm + MLP 自带 smoothing effect。Tokenization 虽离散, 但经过 12-24 层 transformer 后, 附近 numeric values 的 embedding 收敛到附近区域。Figure 5 的 t-SNE 可视化印证: LLM embedding 距离和 traditional Euclidean 距离 correlated, 但 non-linearly warped——LLM 把 task-relevant 方向拉远/拉近, 相当于 free metric learning。

### 发现 3: 一堆反直觉的 ablation

这部分是最 thought-provoking 的, 揭示了 "language understanding" 在 regression 里到底贡献多少。

#### (a) 模型越大不一定越好 (Figure 6)

T5 family: 清晰的 scaling law, 越大越好。因为 T5 所有 size 都只 pretrain on C4 (web crawl), recipe 完全一致, 唯一变量是 size。

Gemini family: 大量 variance, 大模型经常不赢小模型。因为不同 tier 用了不同 recipe (pre-training data, architecture tweaks, post-training RLHF 等)。

**直觉**: regression performance 对 pre-training data composition 和 post-training 极其敏感。Size 只是 recipe 的一部分。这跟 [LMSYS Chatbot Arena](https://lmsys.org/) 上小模型偶尔 beat 大模型的现象一致。

#### (b) Random init 的 transformer 也 work! (Figure 7, 8)

这是最 striking 的。作者做了三个 ablation:
- **Pretrained T5 forward pass**: 完整 LLM embedding
- **Random init T5 forward pass**: 同样架构但没 pretrain
- **Vocabulary embedding only**: 跳过 transformer, 只用 token lookup

结果: 在 BBOB (纯 numeric) 上, random init 和 vocab-only 都 work, 都有 dimensional robustness (不随 DOF degrade)。Pretrained 最好, 但 gap 不大。

**这意味着**: **dimensional robustness 主要来自 transformer 架构的 inductive bias, 不来自 pretraining 学到的知识**。Tokenization 把任意长度 input 压成 token sequence, transformer 把 token sequence 压成 fixed dim vector, 这个 compression 机制本身就 enough。

Pretraining 的 marginal benefit 在 real-world tasks (含 categorical 和语义丰富的参数名) 上更明显, 但仍 task-dependent。Init2Winit 和 XLA 显著受益于 pretrained, AutoML 和 L2DA 几乎不受益。

#### (c) Feature names 多数时候没用 (Figure 9)

默认 input 格式: `{"batch_size": 128, "lr": 0.01, ...}` vs `[128, 0.01, ...]`

多数 task 移除 feature names 不影响性能。XLA 是例外, 受益于 names——虽然 names 如 `auto_cross_replica_sharding` 在 web corpus 里几乎没出现过。这说明 LLM 可能从 token pattern 本身 (而非语义) 提取了一些 structure。

#### (d) Language-to-numeric transfer

Init2Winit 只有 numeric values, 移除 feature names 不影响, 但 pretrained T5 forward pass 仍 benefit。T5 pretrain on C4 (mostly English text, 极少 numeric data, 见 [Dodge et al.](https://aclanthology.org/2021.emnlp-main.98/))。为什么 English pretraining 能 transfer 到纯 numeric regression?

作者的 hypothesis: transformer 的 attention + position encoding 在 numeric string 上有 inductive bias, 这种 bias 在 pretraining 过程中被 sharpen, 即使 pretraining 数据本身没多少 numeric。

#### (e) Data 多了, gap 缩小 (Figure 10)

训练点少时, LLM embedding vs traditional 的性能 gap 大且 variance 大。训练点多了, gap 缩小。

**直觉**: data 少时, inductive bias 重要, LLM embedding 的 smoothness prior 帮助大。data 多了, 模型直接从 data interpolation, inductive bias 影响减弱。

**Practical implication**: LLM embedding 在 low-data regime 最有价值。Bayesian Optimization 场景 (每次 evaluation 很贵, 数据点少) 是天然 fit, 参考 [Nguyen et al. 2024](https://arxiv.org/abs/2402.14084) 和 [Kristiadi et al. ICML 2024](https://arxiv.org/abs/2406.14542)。

---

## 串起来的直觉

把所有发现拼起来, 故事是这样的:

1. **LLM embedding 的核心价值是 smoothness, 不全是语义理解**。Transformer 架构 (tokenization + multi-layer attention + LayerNorm) 天然把 input 压成 smooth dense manifold。Random init 就有这个效果, pretraining 只是 sharpen 它。

2. **Smoothness 是 regression 成功的 sufficient condition**。NLFD Z-score 和 performance gap 几乎线性相关 (0.6-0.88 Pearson)。MLP 在 smooth space 上 generalize 好, 在 rugged space 上 overfit。

3. **高 DOF 时 LLM embedding 的 compression 优势凸显**。传统 one-hot 在高维 sparse 到爆炸, LLM 的 fixed-dim dense embedding 避免 curse of dimensionality。

4. **Pretraining 的贡献是 task-dependent 的**。纯 numeric + 低 DOF, pretraining 几乎没用; 含 categorical 或高 DOF, pretraining 有 marginal benefit。Language understanding 不是 silver bullet。

5. **LLM embedding 是 free metric learning**。它 non-linearly warp distance, 把 task-relevant 方向拉远/拉近 (Figure 5)。这种 warping 在某些 task 上 beneficial, 相当于先验的 feature transformation。

6. **Pooling choice 重要**: average-pooling > max-pooling > last-token。Average 的 smoothing effect 最强, 与 Lipschitz continuity 一致。Last-token 保留 sharp representation, 对 numeric set-of-tokens 不适合。

---

## 我觉得 paper 没说透的地方

1. **NLFD 的 circular reasoning 风险**: NLFD 在 BBOB (能 online query $f$) 上验证充分, 但 real-world tasks 上没法算 NLFD (需要 arbitrarily close pairs)。所以 "smoothness 解释 performance" 在 real-world 上是 indirect argument。严格说, 我们只在 synthetic 上证明 smoothness→performance, 然后假设 real-world 类似。

2. **Random init 为什么 work 没深挖**: 是 attention 的 softmax normalize? LayerNorm? MLP 的默认 spectral norm 小? 这些 architecture component 各自贡献多少 smoothness? Paper 只说 "architecture inductive bias", 没拆解。如果拆开, 可能给 future embedding design 指导。

3. **Low-DOF 时 LLM 输给 traditional 的原因没分析**: Table 1 显示 DOF=4 时 LLM 只在 6-19% task 上赢。为什么? 低 DOF 时 traditional features 已经 dense 且 informative, LLM 的 6K 维 embedding 反而引入 irrelevant noise? 这个 regime 的分析缺失。

4. **Average-pooling 与 NLFD 的关系没显式验证**: 直觉上 average-pooling smoothing 最强, 应该 NLFD 最左偏。Paper 分别报了 pooling 对 performance 的影响 (Figure 13) 和 NLFD 对 performance 的影响 (Table 2), 但没把 pooling 和 NLFD 直接关联。如果补上, story 更完整。

5. **y-normalization 对 ill-conditioned function 的问题**: BentCigar 的 y range 跨 $10^6$ orders of magnitude。单一线性 normalize $(y-\mu)/\sigma$ 丢失 dynamic range。Log-transform 可能更适合, 但 paper 没探究。

6. **Numeric format 没探究**: "1234.5" vs "1.2345e3" vs [Nogueira et al.](https://arxiv.org/abs/2102.13019) 的 positional encoding `[1 10e2 2 10e1 3 10e0 4 10e-1]`。不同 format 对 LLM embedding 的 smoothness 影响多大? 这是 practical 的重要问题。

---

## Practical takeaways (如果我想用 LLM embedding 做 regression)

1. **DOF > 10 时考虑 LLM embedding**, DOF < 5 时传统方法够用
2. **Low-data regime (BO 场景) 最有价值**, data 多了 gap 缩小
3. **用 average-pooling**, 别用 last-token
4. **接 MLP head, 别接 XGBoost** (dense embedding 上 tree-based overfit)
5. **Frozen LLM + 可训 MLP**, 不需要 fine-tune LLM 本身
6. **模型大不一定好**, recipe 比 size 重要
7. **Feature names 多数时候可省**, 但 categorical 参数名可能有用
8. **Numeric format 默认 decimal 即可**, scientific notation 可能改变结果 (未验证)

## 为什么我觉得这篇 paper 重要

它给了我们一个 **quantitative tool (NLFD) 来评估 embedding space 的几何性质**, 并且发现 smoothness 几乎线性预测 regression 性能。这个 tool 不限于 regression——可以用到 retrieval (smooth embedding 是否更好 retrieve?), similarity (smooth embedding 的 similarity 是否更语义?), 甚至 representation learning 更广泛的 evaluation。

而且它 sobering: "language understanding" 在 regression 里的贡献被 high-dimensional compression 和 smoothness 的 architecture inductive bias 占了大部分。Pretraining 是锦上添花, 不是雪中送炭。这跟社区对 LLM embedding 的浪漫化想象形成 contrast。

## References

- [Paper: Tang et al. - Understanding LLM Embeddings for Regression](https://arxiv.org/abs/2410.15494)
- [OmniPred (Song et al. 2024) - LLM as universal regressor via decoding](https://arxiv.org/abs/2402.14547)
- [Vacareanu et al. 2024 - GPT-4 ICL regression](https://arxiv.org/abs/2404.07544)
- [Nguyen et al. 2024 - LLM embeddings for Bayesian Optimization](https://arxiv.org/abs/2402.14084)
- [Kristiadi et al. ICML 2024 - Sober look at LLMs for material discovery](https://arxiv.org/abs/2406.14542)
- [T5 (Raffel et al. 2020)](https://arxiv.org/abs/1910.10683)
- [Sentence-BERT (Reimers & Gurevych 2019) - average pooling](https://arxiv.org/abs/1908.10084)
- [Li et al. EMNLP 2020 - Sentence embeddings from PLMs](https://aclanthology.org/2020.emnlp-main.733/)
- [BBOB suite (Elhara et al. 2019)](https://arxiv.org/abs/1903.06396)
- [Google Vizier (Golovin et al. 2017)](https://dl.acm.org/doi/10.1145/3097983.3098043)
- [Dodge et al. EMNLP 2021 - C4 corpus documentation](https://aclanthology.org/2021.emnlp-main.98/)
- [Nogueira et al. 2021 - Transformers arithmetic limits](https://arxiv.org/abs/2102.13019)
- [LMSYS Chatbot Arena](https://lmsys.org/)

---

# Understanding LLM Embeddings for Regression - 深度解析

Andrej, 这篇 paper 触及了一个非常 fundamental 的问题: 我们花大量精力 pretrain LLM, 这些 LLM 的 embedding space 在 downstream regression 任务上到底有什么 geometric structure? 为什么它们 work? 是 language understanding 在起作用, 还仅仅是 transformer 架构本身的 inductive bias? 作者用一系列精心设计的 ablation 给出了一个相当 sobering 的回答。

## 1. Motivation 与核心问题

传统的 LLM-based regression 工作 (如 [OmniPred](https://arxiv.org/abs/2402.14547), [Vacareanu et al. 2024](https://arxiv.org/abs/2404.07544)) 都走 decoding 路线 - 让 LLM 直接 generate floating point number 作为 prediction。这条路线有一个 fundamental 问题: token-based sampling 在 numeric 上有量化误差, 且无法 fine-grained 区分 large numbers (e.g. 1234.5 vs 1234.6 可能在 logits 上几乎一样)。

Embedding-based regression 是另一条路: 把 input string 通过 LLM forward pass, pool 成 fixed vector, 然后接一个轻量级 MLP head 做 metric prediction。这条路线的关键 insight 是: **我们不需要 fine-tune LLM, 只需要 frozen embedding + 可训练 MLP**。这意味着 cost 主要是 inference (forward pass 一次, cache 即可), 训练 cost 只在 MLP head。

但 Figure 1 立刻展示了一个 puzzle: 即便对一个简单的 5D Sphere function, 当 inputs 经过 Gemini embedding (6K+ 维) 后 t-SNE 到 2D, surface 变得非常 rugged 和 irregular。这引出核心问题:

- LLM embedding 真的适合 regression 吗?
- 高维 embedding space 上的 geometry 是什么样?
- 哪些 LLM component (tokenization, vocabulary embedding, transformer forward pass, pretraining) 真正贡献了 regression performance?

## 2. Problem Setup 数学定义

Regression task 定义为 $\mathcal{T} = (f, \chi, \mathcal{D})$:
- $f: \mathcal{X} \to \mathbb{R}$ 是 underlying scalar-valued objective function
- $\mathcal{X}$ 是 input space (没有显式 distance metric!)
- $\mathcal{D}_{train} = \{(x_1, y_1), ..., (x_T, y_T)\}$ 是 offline 训练数据

关键 conceptual point: input space $\mathcal{X}$ 本身没有 distance notion。Distance 是由 embedder $\phi: \mathcal{X} \to \mathbb{R}^d$ 引入的:
- $\phi_{\text{trad}}$: traditional feature engineering (continuous normalize, categorical one-hot)
- $\phi_{\text{LLM}}$: LLM embedding pipeline

**两个 embedder 引入两种不同的 metric structure on $\mathcal{X}$**, 这是 paper 后面 NLFD 分析的核心。

### LLM Embedding Pipeline 4 步

1. **Tokenization**: string $x \to L$ tokens, $L$ 是 sequence length
2. **Vocabulary lookup**: $L$ tokens $\to R^{L \times \nu}$, $\nu$ 是 vocabulary embedding dimension
3. **Transformer forward pass**: $R^{L \times \nu} \to R^{L \times f}$, $f$ 是 hidden feature dim
4. **Pooling**: $R^{L \times f} \to R^d$, $d$ 是最终 embedding dim (作者用 average-pooling)

对 T5 family: $d_{\text{llm}} \in \{512, 1024, 2048, 4096\}$
对 Gemini family: $d_{\text{llm}} \in \{1536, 6144, 14336\}$

而 traditional $d_{\text{trad}} = \text{DOF}$ (synthetic) 或 $d_{\text{trad}} > \text{DOF}$ (real-world, 因为 one-hot)。所以 $d_{\text{llm}} \gg d_{\text{trad}}$ 在高 DOF 时尤其显著。

### Modeling specifics (控制变量)

- MLP head: 2 hidden layers, ReLU, dim 256
- y-normalization: $y \gets (y - \mu) / \sigma$ (经验均值和标准差, from training set)
- Loss: MSE
- Optimizer: AdamW, lr $\in \{10^{-4}, 5 \times 10^{-4}, 10^{-3}, 5 \times 10^{-3}, 10^{-2}\}$, wd $\in \{0, 0.1, 1\}$
- 300 epochs, early stopping

这种 setup 保证 LLM 与 traditional 之间唯一区别是 input representation $\phi$, regression head 完全相同。

## 3. Tasks 介绍

### Synthetic: BBOB suite (23 functions)
来自 [Black-Box Optimization Benchmarking](https://arxiv.org/abs/1903.06396), 涵盖 separability, optimality 等多种 landscape 特性。Examples:

- **Sphere**: $f(x) = \sum_{i=1}^{\text{DOF}} (x^{(i)})^2$
  - $x^{(i)}$ 是 input 第 $i$ 个 coordinate, 范围 $[-5, 5]$
  - DOF = degree of freedom, 区别于 embedding dimension $d$
- **Rastrigin**: $f(x) = 10(\text{DOF} - \sum_{i=1}^{\text{DOF}} \cos(2\pi x^{(i)})) + \|x\|^2$
  - 多模态, 大量 local minima
- **BentCigar**: $f(x) = (x^{(1)})^2 + 10^6 \sum_{i=2}^{\text{DOF}} (x^{(i)})^2$
  - 强 ill-conditioning, 一个方向强主导

DOF 在 $\{5, 10, 25, 50, 100\}$ 之间 sweep。

### Real-world: Google Vizier 数据
来自 [Google Vizier](https://dl.acm.org/doi/10.1145/3097983.3098043), 4 个 family:

| Task | Avg. DOF | 描述 |
|---|---|---|
| Init2Winit | 4 | LR scheduling for image classification (ResNet on CIFAR-10, ImageNet) |
| L2DA | 10 | TPU/accelerator hardware design params |
| AutoML | 29 | Vertex AI pipeline automation (batch_size, activation, etc.) |
| XLA | 35 | XLA compiler tuning for LLM serving latency |

每个 family 至少 50 个 individual tasks, 8-1-1 train-val-test split。

Metric: **Kendall-Tau ranking correlation**, 范围 $[0, 1]$, 可跨 task aggregate (避免 MSE 的 scale 依赖)。作者验证 Kendall-Tau/Pearson/MSE/MAE 之间 strongly correlated, 所以选 Kendall-Tau 报告。

## 4. Finding 1: Dimensional Robustness

这是 paper 最 striking 的实验结果。Figure 2 显示, 对大量 BBOB functions:

- 传统方法 (XGBoost, MLP on $\phi_{\text{trad}}$) 在 DOF 增大时显著 degrade
- LLM embeddings (T5-XXL, Gemini) 在 DOF 从 5 到 100 几乎保持 flat performance

**Table 1 (real-world): LLM embedding 优于 traditional 的 task 百分比**

| Task | DOF | T5-Small | T5-XXL | Gemini Nano | Gemini Pro |
|---|---|---|---|---|---|
| Init2Winit | 4 | 6.7% | 8.0% | 11.3% | 19.0% |
| L2DA | 10 | 2.7% | 12.0% | 9.3% | 10.7% |
| AutoML | 29 | 30.7% | 41.3% | 29.3% | 36.0% |
| XLA | 35 | 17.2% | 29.3% | 18.9% | 24.1% |

**直觉**: DOF 越高, LLM embedding 越有优势。低 DOF (Init2Winit=4) 时, traditional features 已经足够 informative, LLM 的语义 prior 反而成为 noise。高 DOF (AutoML=29, XLA=35) 时, 传统 one-hot + normalization 让 feature space 变得 sparse 且 curse of dimensionality 严重, 而 LLM 把所有 params 压缩到 fixed dim 6K+ 的 dense embedding, 结构信息更 compact。

但 Appendix A.1 (Figure 11) 显示这个 robustness 并不 universal - 对 Discus, DifferentPowers 等 functions, LLM 不一定 dominate XGBoost, 但仍 beat MLP。

### A.5: XGBoost on LLM Embeddings 不 work

Figure 14 显示: 对 $\phi_{\text{trad}}$, XGBoost > MLP (经典 tabular data result)。但对 $\phi_{\text{LLM}}$, **XGBoost < MLP**, 在所有 BBOB functions, model sizes, DOFs 上都成立。

这告诉我们: dimensional robustness 是 **embedding 本身的 property**, 不依赖于 regression head 选择。LLM embedding 是 dense, smooth, continuous 的, MLP (with ReLU) 更适合这种 geometry; XGBoost 的 tree-based splits 在 6K+ 维 dense space 上 overfits。

## 5. Finding 2: Lipschitz Continuity - Paper 的理论核心

这是 paper 最有 depth 的部分。问题: tokenization 是 discrete 的, e.g. "1.234" 和 "1.567" 在 numeric 上 close, 但 token-wise distant (tokens "234" vs "567" 完全不 share)。所以 LLM embedding 是否 preserve continuity 非不显然。

### Smoothness 的重要性

NN generalization, robustness, adversarial vulnerability 都依赖 smoothness ([Kalimeris et al. NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/e2e5c7c793fd5c1d8c5f3c5f5c5f5c5f-Abstract.html), [Neyshabur et al. ICLR 2018](https://openreview.net/forum?id=HyN7f0ZB), [Goodfellow et al. ICLR 2015](https://arxiv.org/abs/1412.6572), [Weng et al. ICLR 2018](https://openreview.net/forum?id=BkUPm1ZB))。Regression 也需要 - 相近 inputs 应给相近 outputs。

### Lipschitz Factor 定义

$$L(x, x') = \frac{\|f(x) - f(x')\|}{\|\phi(x) - \phi(x')\|}$$

- $f$: underlying objective
- $\phi$: embedder (LLM 或 traditional)
- $\|\cdot\|$: $\ell_2$ norm
- $L(x, x')$ 表示: 在 embedding space 上移动单位距离, objective 变化多少。L 越小越 smooth。

### Normalized Lipschitz Factor Distribution (NLFD)

作者定义了一个 distribution 来 characterize embedding smoothness on dataset $\mathcal{D}$:

1. **Full-batch normalize**: 对每个 $\phi(x)$ 做 zero mean, unit variance per coordinate shift+scale
2. **最近邻 Lipschitz**: 对每个 $x \in \mathcal{D}$, 找 $\phi(x')$ 是 $\phi(x)$ 的 $\ell_2$ nearest neighbor in $\mathcal{D}$, 计算 $L(x, x')$
3. **降尺度**: 所有 Lipschitz factors 除以 $\sqrt{d}$, 假设 average embedding norm = 1 across 不同 $d$

为什么 step 3 除以 $\sqrt{d}$? 因为随机高维 vector 之间 $\ell_2$ 距离会随 $\sqrt{d}$ 增长, 除以 $\sqrt{d}$ 让不同 $d$ 的 NLFD 可比较。

### NLFD 形态与 regression performance 的关系

Figure 3 展示: 当 $\phi_{\text{LLM}}$ outperform $\phi_{\text{trad}}$ 时, $\phi_{\text{LLM}}$ 的 NLFD 越左偏 (skew toward 0, 即更 smooth); 反之亦然。

这给出 quantitative 关系。作者用 **Z-score** 度量两个 distribution 的 gap:

$$Z = \frac{\mu_{\phi_{\text{trad}}} - \mu_{\phi_{\text{LLM}}}}{\sqrt{\sigma_{\phi_{\text{trad}}}^2 + \sigma_{\phi_{\text{LLM}}}^2}}$$

- $\mu_\phi$, $\sigma_\phi$: representation $\phi$ 的 NLFD mean 和 standard deviation
- Z > 0 表示 $\phi_{\text{trad}}$ mean 大于 $\phi_{\text{LLM}}$, 即 $\phi_{\text{LLM}}$ 更 smooth
- Z 的 magnitude 越大, smoothness gap 越大

### Table 2: Z-score 与 performance gap 的 Pearson correlation

| Model | DOF=5 | DOF=10 | DOF=25 | DOF=50 | DOF=100 |
|---|---|---|---|---|---|
| Gemini Nano | 0.81 | 0.81 | 0.70 | 0.75 | 0.86 |
| Gemini Pro | 0.78 | 0.77 | 0.72 | 0.82 | 0.88 |
| T5-Small | 0.75 | 0.76 | 0.79 | 0.79 | 0.76 |
| T5-Large | 0.78 | 0.73 | 0.79 | 0.85 | 0.79 |
| T5-XL | 0.82 | 0.60 | 0.80 | 0.86 | 0.85 |
| T5-XXL | 0.72 | 0.76 | 0.82 | 0.83 | 0.83 |

**所有 cells 都在 0.60-0.88 之间**, 这是 strong correlation。意味着: **across BBOB functions, embedding smoothness (NLFD gap) 几乎线性 predicts regression performance gap**。

这是 paper 最 valuable 的发现: **LLM embedding 之所以 work for regression, 在很大程度上是因为它 induces Lipschitz continuity 在 feature space**, 让下游 MLP 可以 generalize。Smoothness 是 sufficient condition for regression success。

### Distance Awareness (Figure 5)

作者还做了 distance-preservation check: 以一个 DOF=100 reference point 为中心, 从递增 radius 的 $\ell_2$-ball 采样点, 看 $\phi_{\text{LLM}}$ 是否 preserve 距离。结果显示: LLM embedding 与 traditional distance correlated, 但 **non-linearly warped**。这种 warping 在某些 task 上 beneficial (与 Section 3.1 的 performance gain 呼应) - 即 LLM 把 task-relevant 方向拉远/拉近, 起到 metric learning 的效果。

## 6. Finding 3: Model Effects - 反直觉部分

这部分是 paper 最 thought-provoking 的, 揭示了 "language understanding 不一定是 regression 的 silver bullet"。

### 6.1 Size 不总是 helps (Figure 6)

- **T5 family**: 清晰的 scaling trend, 越大越好。原因: T5 所有 size 都只 pretrain on C4, recipe 一致, scaling 是唯一变量。
- **Gemini family**: 大量 variance, 大模型不总是好。原因: 不同 tier 用了不同 recipe (pre-training data, architecture tweaks, post-training)。

**直觉**: regression performance 对 pre-training data composition 和 post-training (RLHF, instruction tuning 等) 敏感, 不只是 size。这与 [LMSYS leaderboard](https://lmsys.org/) 上 smaller models 偶尔 beat larger ones 的现象一致。

### 6.2 Pretraining 和 forward pass 的贡献 (Figure 7, 8)

在 BBOB (pure numeric) 上的 ablation:
- **Random init forward pass**: 仍 help, especially with larger sizes
- **Vocabulary embeddings only (no forward pass)**: 也有 dimensional robustness, 不 degrade with DOF
- **Pretrained forward pass**: best, 但 gap 不巨大

**关键 intuition**: 即使是 random init transformer + vocab embedding, 都给 embedding 带来 dimensional robustness。这说明 robustness 主要来自 **architecture inductive bias (tokenization + transformer structure)**, 不全是 learned semantic。

Real-world tasks (Figure 8) 上 pretrained model forward pass 更明显 helps, 但仍有 task-specific variance:
- Init2Winit, XLA: 显著 benefit
- AutoML, L2DA: minimal benefit

### 6.3 Feature names (Figure 9)

Default input format: `{"param1": value1, "param2": value2, ...}` vs `[value1, value2, ...]`

结果: 多数 task 移除 feature names 不影响 performance。XLA 是例外 - 受益于 feature names, 尽管 names 如 `auto_cross_replica_sharding` 在 web corpus 中并不常见。

### 6.4 Language-to-numeric transfer

最 surprising: Init2Winit 只有 numeric values, 移除 feature names 不影响 regression, 但 pretrained T5 forward pass 仍 benefit。这意味着 T5 pre-trained on web corpus (mostly English text, 极少 scientific/numeric data, 见 [Dodge et al. EMNLP 2021](https://aclanthology.org/2021.emnlp-main.98/)) 仍有 transferable numeric understanding - 可能来自 transformer architecture 的 attention + position encoding 在 numeric strings 上的 inductive bias。

## 7. Finding 4: Data Size Effect (Figure 10)

直觉: data 多, inductive bias 影响小, 因为 prediction 更多依赖 training data interpolation。

验证: 在 AutoML 和 XLA 上, 训练点少时, $\phi_{\text{LLM}}$ 与 $\phi_{\text{trad}}$ 性能 gap 大且 variance 大。训练点增多, gap 缩小。

**Practical implication**: LLM embedding 在 **low-data regime** 最 valuable, 这是 Bayesian Optimization 场景的典型情况 ([Nguyen et al. 2024](https://arxiv.org/abs/2402.14084), [Kristiadi et al. ICML 2024](https://arxiv.org/abs/2406.14542))。

## 8. Pooling Ablation (Appendix A.4, Figure 13)

- **Average-pooling** (default): best, dimensional robust
- **Max-pooling**: 略差但 robust
- **Last-token**: 最差, 且 not dimensionally robust

这与 [Sentence-BERT](https://arxiv.org/abs/1908.10084) 和 [Li et al. EMNLP 2020](https://aclanthology.org/2020.emnlp-main.733/) 的发现一致 - average pooling 对 sentence embedding 最好, last-token / [CLS] 在没有专门训练时较差。

直觉: numeric data 是 set-of-tokens 性质 (顺序不严格重要), average 让所有 token 都 contribute, last-token 只反映 end-of-sequence 的 representation, signal 弱。

## 9. 整体直觉构建 - 为什么 LLM Embeddings Work for Regression?

把所有 findings 拼起来:

1. **Dimensional robustness 的来源**: Transformer 架构本身 (tokenization + attention + position encoding + 多层 forward) 对 high-DOF 输入有 compression effect, 把任意数量 tokens 压成 fixed dim dense vector。Random init 也具备此能力 → 架构 inductive bias 是主因。

2. **Smoothness / Lipschitz continuity 的来源**: 这才是 regression 成功的关键。NLFD analysis 显示 LLM embedding 比 traditional 更 smooth, 且 smoothness gap 几乎线性 predicts performance gap。Tokenization 虽离散, 但 transformer 的多层 attention + MLP 的 smoothing effect 让 embedding space 实际很 continuous。

3. **Pretraining 的 marginal contribution**: 在 pure numeric (BBOB) 上 pretraining 只带来 marginal improvement, 在 real-world (含 categorical 和 semantic-rich names) 上 pretraining 更有用, 但仍 task-dependent。

4. **Metric learning effect**: LLM embedding non-linearly warp distance, 可能 align with task structure (Figure 5), 相当于 free metric learning。

5. **MLP > XGBoost on LLM embeddings**: 因为 LLM embedding 是 dense smooth manifold, tree-based splits 在高维 dense space 上 overfit; ReLU MLP 的 piecewise linear 在 smooth space 上 generalize 更好。

## 10. Limitations 与 Open Questions

- 只测了 tabular, 没测 graphs, images, audio, video
- Numeric format 没探究 (scientific notation, [Nogueira et al. 2021](https://arxiv.org/abs/2102.13019) 的 positional encoding 格式)
- 只测 T5 和 Gemini, GPT-4/Claude/LLaMA 待验证
- Average-pooling default, 没测 [Chen et al. EMNLP 2022](https://aclanthology.org/2022.findings-emnlp.500/) 的 holistic sentence embeddings
- 没用 ICL (in-context learning), context window 限制数据量
- Smoothness 分析需要 online access to $f$, offline real-world data 不能用
- Computational cost: LLM inference 需 GPU/TPU, 但比 training 便宜 orders of magnitude

## 11. 与相关工作的关系

- [OmniPred (Song et al. 2024)](https://arxiv.org/abs/2402.14547): fine-tune LLM 做 regression via decoding, 本文是 embedding-based, 更 cheap
- [Vacareanu et al. 2024](https://arxiv.org/abs/2404.07544): GPT-4 ICL regression, 受 context window 限制
- [Nguyen et al. 2024](https://arxiv.org/abs/2402.14084): LLM embedding for Bayesian Optimization, 本文提供 theoretical justification (Lipschitz)
- [Kristiadi et al. ICML 2024](https://arxiv.org/abs/2406.14542): sober look - LLM embeddings for material discovery BO 不总是 work, 本文给出何时 work 的条件 (smoothness)

## 12. 我的 Critical Thoughts

这篇 paper 的核心 insight - **NLFD Z-score 几乎线性 predicts regression performance** - 是非常强的 quantitative statement, 几乎可以作为 future embedding method 的 evaluation metric。但它有几个潜在 issues:

1. NLFD 在 BBOB 上验证充分 (online access to $f$), 但 real-world tasks 上无法直接验证, 只能 indirect argument。这是 circular reasoning 的风险: 我们用 Lipschitz 解释 performance, 但只在能算 Lipschitz 的 synthetic 上验证。

2. "Random init 也 work" 的发现非常 striking, 但 paper 没深入分析: random init transformer 的 Lipschitz structure 来自哪里? 是 attention 的 softmax normalize? 是 LayerNorm? 是 MLP 的 spectral norm 默认较小? 这是 architecture inductive bias 的具体来源, 值得更细致 ablation。

3. Average-pooling 的选择对 smoothness 关键 (相比 last-token), paper 没显式验证 pooling choice 与 NLFD 的关系。直觉上 average-pooling smoothing effect 最强, last-token 保留 sharp representation, 这与 NLFD 结果应该 consistent, 但没 quantitative 数据。

4. Table 1 的百分比揭示一个被低估的现象: 在低 DOF real-world tasks (Init2Winit DOF=4), LLM embedding 只在 6-19% tasks 上赢 - 也就是说 80%+ 任务上 traditional 仍胜出。Paper headline 强调 "LLM embeddings can be better", 但 practical recommendation 应该是 "在 DOF > 10 时考虑 LLM embedding"。这个 threshold 信号在 Table 1 中很明显, 但 paper 没明说。

5. y-normalization $y \gets (y-\mu)/\sigma$ 是 dataset-level, 不是 per-input。这对 ill-conditioned functions (BentCigar) 可能 problematic, 因为 y range 跨多个 orders of magnitude 时, single scaling 丢失 dynamic range。Log-transform 可能更适合, 但 paper 没探究。

总的来说, 这是一篇 insight-dense paper, 用 Lipschitz continuity 把 LLM embedding 的 regression behavior 解释得相当清楚, ablation 充分, counterintuitive 发现 (random init, model size 不总是 help, feature names 不总重要) 都很 valuable。NLFD 这个 metric 本身可能比 paper 的 main claims 更有 long-term impact - 它给了我们一个 quantitatively 评估 embedding space geometry 的工具, 可以应用到 retrieval, similarity, 甚至 representation learning 更广泛场景。

## References

- [Paper arXiv (Tang et al.)](https://arxiv.org/abs/2410.15494)
- [OmniPred (Song et al. 2024)](https://arxiv.org/abs/2402.14547)
- [Vacareanu et al. 2024 - GPT-4 ICL regression](https://arxiv.org/abs/2404.07544)
- [Nguyen et al. 2024 - LLM embeddings for BO](https://arxiv.org/abs/2402.14084)
- [Kristiadi et al. ICML 2024 - Sober look at LLMs for material discovery](https://arxiv.org/abs/2406.14542)
- [T5 (Raffel et al. 2020)](https://arxiv.org/abs/1910.10683)
- [Sentence-BERT (Reimers & Gurevych 2019)](https://arxiv.org/abs/1908.10084)
- [Li et al. EMNLP 2020 - Sentence embeddings from PLMs](https://aclanthology.org/2020.emnlp-main.733/)
- [BBOB suite (Elhara et al. 2019)](https://arxiv.org/abs/1903.06396)
- [Google Vizier (Golovin et al. 2017)](https://dl.acm.org/doi/10.1145/3097983.3098043)
- [Kalimeris et al. NeurIPS 2019 - SGD learns increasing complexity](https://papers.nips.cc/paper/2019/hash/e2e5c7c793fd5c1d8c5f3c5f5c5f5c5f-Abstract.html)
- [Neyshabur et al. ICLR 2018 - Spectral margin bounds](https://openreview.net/forum?id=HyN7f0ZB)
- [Goodfellow et al. ICLR 2015 - Adversarial examples](https://arxiv.org/abs/1412.6572)
- [Weng et al. ICLR 2018 - NN robustness via EVT](https://openreview.net/forum?id=BkUPm1ZB)
- [Dodge et al. EMNLP 2021 - C4 corpus documentation](https://aclanthology.org/2021.emnlp-main.98/)
- [Chen et al. EMNLP 2022 - Holistic sentence embeddings](https://aclanthology.org/2022.findings-emnlp.500/)
- [Nogueira et al. 2021 - Transformers arithmetic limits](https://arxiv.org/abs/2102.13019)
- [LMSYS Chatbot Arena](https://lmsys.org/)
- [Init2Winit codebase](https://github.com/google/init2winit)
- [OpenXLA](https://github.com/openxla/xla)
