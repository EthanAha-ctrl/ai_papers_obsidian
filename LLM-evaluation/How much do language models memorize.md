---
source_pdf: How much do language models memorize.pdf
paper_sha256: ce0148ea694019714c4c3d5d8bb3a09b2f1e7c402f377f9db50cae705e32b123
processed_at: '2026-08-19T11:33:10-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

## 一句话总结

这篇 paper 说了一件事: **一个 GPT-style transformer 的每个 parameter 大概能存 3.6 bits 的 training data 信息, 存满了之后 model 就被迫开始 generalize**。

就这么一句话, 但背后的推导挺漂亮。

---

## 这帮人想解决什么问题

之前的 folks 想搞清楚 "LLM 到底记住了多少 training data", 基本两条路:

1. **Extraction**: 能不能逼 model 吐出某条 training sample (Carlini et al.)
2. **Membership inference**: 能不能判断某条 sample 在不在 training set 里 (Shokri et al.)

这俩方法有个共同的大坑: **分不清 "记住了" 和 "学到了"**。

举个 paper 里的例子: training data 里有 `Q: What is 2^100? A: 1267650600228229401496703205376`。如果你 prompt model 问 `2^100 = ?`, 它吐出这个数字, extraction attack 就说 "memorized!"。但问题来了 — model 完全可能自己学会了算 $2^{100}$, 根本不需要"记"这条 sample。

这就是 paper 的 motivation: **得把 "sample-level 的死记硬背" 和 "population-level 的规律学习" 分开**。

---

## 核心 idea: 用 compression 衡量 memorization

### 直觉

如果你手上有个 model $\hat{\theta}$, 你要存一条 string $x$。两种方式:

- **没 model 帮忙**: 直接存 raw bits, 长度 = $H(x)$
- **有 model 帮忙**: 用 model 当 prior, 存一个 "残差", 长度 = $H(x | \hat{\theta})$

model 帮你省下的 bits, 就是 model "知道" 关于 $x$ 的信息量:

$$\text{mem}(x, \hat{\theta}) = H(x) - H(x | \hat{\theta})$$

这其实就是 Kolmogorov complexity 的那一套, 只不过用 arithmetic coding + model likelihood 来近似 (因为 Shannon 1950 证明了 arithmetic coding 的 code length ≈ negative log likelihood)。

### 关键 trick: 减掉 generalization 的部分

光这么算会把 generalization 也算进去。比如 $x$ 是一句正常的英文, model 因为学过英文语法, 压缩率天然就高 — 但这不是"记住这条 sample", 这是"懂英文"。

解决: 引入一个 reference model $\theta$ (可以理解为 "一个见过海量数据的大 model", 或者真实分布本身)。定义:

$$\text{mem}_U(x, \theta, \hat{\theta}) = H(x | \theta) - H(x | \theta, \hat{\theta})$$

- $H(x | \theta)$: **有了 reference model 还需要多少 bits** — 这是 "generalization 解释不掉" 的部分
- $H(x | \theta, \hat{\theta})$: **target model + reference model 一起, 还需要多少 bits**

两者之差 = **target model 比 reference model 多知道的那部分**, 这就是 unintended memorization。

Intuition: 如果 $\hat{\theta}$ 只是学到了英文语法 (跟 reference $\theta$ 一样), 那 $H(x|\theta)$ 和 $H(x|\theta,\hat{\theta})$ 几乎相等, mem_U ≈ 0。只有当 $\hat{\theta}$ 真的"记住"了 $x$ 的 sample-specific 信息, mem_U 才会大。

### 实际怎么算

- $H(x | \hat{\theta}) \approx -\log p(x | \hat{\theta})$ — 直接用 target model 的 negative log likelihood
- $H(x | \theta, \hat{\theta}) \approx -\log \max\{p(x|\hat{\theta}), p(x|\theta)\}$ — 两个 model 谁压缩得好就用谁

reference model 的选择:
- Synthetic data: 直接用 true uniform distribution
- Text data: 用同 family 的大 model (oracle), 训练在 superset 上

---

## 实验一: 纯随机 bitstring, 测 model capacity

### 为什么用随机数据

随机 bitstring 没 pattern 可学, 所以 $\text{mem}_U \approx \text{mem}$ — 完美隔离了 generalization。这样测出来的 memorization 就是纯 "raw storage capacity"。

### Setup

- Vocab $V = 2048$, sequence length $S = 64$
- GPT-2 architecture, 1-8 layers, hidden 32-512, 100K-20M params
- 训练 $10^6$ steps, bfloat16, 单卡 A100
- 每个 (model, dataset) 组合跑 5 个 seeds

### 结果: 3.6 bits/parameter

Figure 1 最 striking: 给定 model size, 随着 dataset 增大, memorization 一开始线性涨, 然后 **plateau** — model 存满了。

Table 1 给了详细 breakdown。把所有 configs 汇总:

| Precision | $\alpha$ (bits/param) |
|-----------|----------------------|
| bfloat16  | 3.51                 |
| fp32      | 3.83                 |

注意 fp32 只比 bf16 多 ~10%, 远不到 2x。说明多出来的 precision 大部分 **没用于存信息**, 可能是用于 represent 更 smooth 的 loss landscape 或者冗余 robustness。

### Compare 之前的工作

- Cover 1965 / Gardner 1988: 单层 perceptron ~2 bits/param
- Allen-Zhu & Li 2024: ~2 bits/param (用 quantization 估, https://arxiv.org/abs/2404.05405)
- **This paper: ~3.6 bits/param** (用 likelihood 估, 更直接)

差别的 intuition: quantization 是 lossy 的, 会丢掉一些 effective capacity; 而 likelihood 直接用 model 自己的 predictive power, 更接近 "model 真正 encode 了多少信息"。

---

## 实验二: 真实文本上, memorization vs generalization 的切换

### Setup

- 数据: FineWeb (https://arxiv.org/abs/2406.17557), 严格 deduplication (truncating 到 64 tokens 后会引入 1-2% duplicates, 必须再去一次)
- Reference model: 同 family 的大 model, 训练在 full data 上

### Figure 4: train/test loss 曲线

对 fixed model size, 逐步增大 dataset:
- **Dataset < capacity**: train loss 很低, test loss 高 — 经典 overfitting, model 在死记
- **Dataset ≈ capacity**: test loss 达到 peak (最差的 generalization 点)
- **Dataset > capacity**: test loss 开始下降 — model 被迫 share representation, 进 generalization regime

### Figure 3: Double descent 的信息论解释

这是 paper 最漂亮的 insight。Double descent (Belkin et al. 2019, https://arxiv.org/abs/1903.07571; Nakkiran et al. 2019, https://arxiv.org/abs/1912.02292) 之前没特别 clean 的解释。这篇 paper 说:

> **Double descent 开始于 dataset capacity 超过 model capacity 那个点**。

Intuition: 
- Dataset 小的时候, model 有余量把每条 sample 都 "raw store" 下来, 不需要找 pattern
- Dataset 大到 model 装不下了, gradient descent 被迫找 cross-sample 的共享结构
- 这个 "被迫 share" 的时刻就是 test loss 开始 drop 的时刻, 也就是 grokking (https://arxiv.org/abs/2201.02277) 现象的信息论起源

### Figure 8: Extraction rate 不会降到 0

32-token prefix extraction rate 随 dataset 增大逐步下降, 但 **收敛到 test set 的 extraction rate**, 不是 0。

含义: 当 dataset 足够大, 所有成功的 extraction 都 attributable to generalization, 不是 memorization。换句话说, **"model 能生成这条 string" ≠ "model 记住了这条 string"**。

这对 Carlini et al. 的 extraction attack 是个重要的 caveat: 你 extract 出来的东西, 可能是 model 学到的通用 pattern, 跟具体 training sample 无关。

### Figure 16 + Table 5: 哪些 sample 真被记住

算每个 training document 的 TF-IDF:

$$\text{TF-IDF}(d; \mathcal{D}) = \frac{1}{|d|} \sum_{w \in d} \log \frac{|\mathcal{D}|}{tf(w, \mathcal{D})}$$

- $d$: 文档
- $\mathcal{D}$: 整个 dataset
- $tf(w, \mathcal{D})$: word $w$ 在 $\mathcal{D}$ 中出现总次数
- $|d|$: 文档长度

高 TF-IDF = 文档含很多 rare words。

Result: **TF-IDF 高的 sample 最容易被 memorize**。Top 20 memorized 里 17 条是非英语 (日、中、希伯来、希腊)。直觉很清楚: 这些 outlier 没 pattern 可学, model 只能硬记。

这跟 Carlini et al. 发现 canary sequence 容易被 extract 是一致的 — canary 就是人工注入的 random outlier。

---

## 实验三: Membership Inference 的 Scaling Law

### 观察

固定 model size, M.I. F1 score 随 dataset size 大致呈 sigmoid: dataset 小时 F1≈1 (轻松区分 train/test), dataset 大时 F1→0.5 (随机猜)。

### 拟合的公式

$$\text{Membership}_{F_1}(\theta, \mathcal{D}) = \frac{1}{2}\left(1 + c_1 \sigma\left(c_2\left(\frac{\text{Capacity}(\theta)}{|\mathcal{D}|} + c_3\right)\right)\right)$$

变量:
- $\theta$: model
- $\mathcal{D}$: dataset
- $|\mathcal{D}|$: sample 数
- $\text{Capacity}(\theta)$: $\alpha \times \text{param count}$, 用前面估的 $\alpha = 3.64$
- $\sigma(x) = 1/(1+e^{-x})$: sigmoid
- $c_1 = 1.34, c_2 = -0.034, c_3 = -33.14$: 拟合出来的常数

### Validation

Table 2: 用 scaling law 倒推 "要多大 dataset 才能达到 F1=0.95", 然后实际训练 GPT2-XL (1.55B) 验证。

| Model | Target F1 | Predicted $|\mathcal{D}|$ | Observed F1 |
|-------|-----------|--------------------------|-------------|
| GPT2-XL | 0.55 | 170M | 54.6 ± 1.3 |
| GPT2-XL | 0.75 | 77M | 71.1 ± 0.4 |
| GPT2-XL | 0.95 | 19M | 95.9 ± 0.8 |

误差 < 1.5%, scaling law 挺准的。

### 对现代 LLM 的 implication

现代 LLM 的 token-per-param ratio 通常 $\geq 100$。Llama 3 8B 训 15T tokens, ratio ≈ 1800。

代入 scaling law: **F1 ≈ 0.5**, 即 loss-based membership inference 在统计上不可行。

这解释了最近 Das et al. 2024 (https://arxiv.org/abs/2406.16201) 和 Duan et al. 2024 (https://arxiv.org/abs/2406.16201) 发现 "M.I. attack 在大 LLM 上没用" 的现象 — 不是 attack 设计得不好, 是 fundamental information-theoretic 的限制。

---

## 三个核心 Intuition

### 1. Memorization 和 Generalization 是同一块 capacity 的两种用法

Model 的总信息存储 ≈ $\alpha \times \text{params}$。这块存储可以装:
- **Raw sample info** (unintended memorization): 比如 "训练集里第 1234 条是日语文本"
- **Population pattern** (generalization): 比如 "英文通常 SVO 语序"

Dataset 小: model 有余量, 优先 raw store (懒)
Dataset 大到装不下: 被迫找 pattern, 因为 pattern 的 compression efficiency 更高

这就是为什么 overfitting 容易, generalization 难 — generalization 需要更 clever 的 representation。

### 2. 3.6 bits/param 是 transformer 的"信息密度常数"

这个 number 的意义:
- 一个 8B param 的 LLM, raw storage capacity ≈ $8 \times 10^9 \times 3.6 \approx 2.9 \times 10^{10}$ bits ≈ 3.6 GB
- Llama 3 8B 训 15T tokens (~7 TB unique data) → 远超 capacity, 几乎全在 generalization regime
- 这也是为什么 modern LLM 的 extraction attack 在 population level 失效

Compare: 一个 16-bit float 理论上能存 16 bits, 但实际只能存 3.6 bits 的 "free information"。其他 bits 去哪了? 大概是:
- Represent loss landscape 的 smoothness (训练需要)
- Redundancy for robustness
- Architecture-induced 的 inefficiency (transformer 不是 optimal compressor)

### 3. Double descent 是 capacity constraint 触发的 phase transition

之前 double descent 没特别 clean 的解释。这篇 paper 给了一个:

> **Dataset size 超过 model capacity (bits) 的那一刻, 就是 double descent 的拐点**。

Phase 1 (dataset < capacity): memorization regime, test loss 高
Phase 2 (dataset > capacity): generalization regime, test loss 低

中间那个 test loss 的 peak 就是 "model 刚好装满, 还没开始 generalize" 的尴尬点。

---

## 实践意义

1. **估 capacity**: 想知道你的 model 能记多少 data? `capacity_bits ≈ 3.6 × params`
2. **选 dataset size**: 想避免 memorization? dataset 的 entropy 要远大于 model capacity。对 8B model, 至少要 > 3.6 GB 的 unique high-entropy data
3. **M.I. attack 可行性**: token-per-param > 100 时, loss-based M.I. 基本死路一条
4. **Outlier 最危险**: 即使整体 dataset 巨大, rare tokens / 非主流语言的 sample 仍会被硬记 — 这是 privacy leak 的主要风险点

---

## 我个人的延伸思考

这篇 paper 让我 (假设我是 Karpathy) 想到几个方向:

1. **3.6 bits/param 对 quantization 的指导**: 如果 model 实际只用了 3.6 bits/param, 那 4-bit quantization (4 bits/param) 理论上应该几乎无损。这跟近期 quantization 实验 (GPTQ, AWQ) 的发现一致。

2. **Architecture 的影响**: paper 只测了 GPT-2 family。Mamba / RWKV / hybrid architecture 的 $\alpha$ 会不同吗? 如果不同, 哪个 architecture 的 information density 更高?

3. **Compression algorithm 的选择**: paper 用 arithmetic coding, 但理论上任何 compressor 都行。如果用专门设计的 "optimal compressor for transformer weights", 估出来的 $\alpha$ 可能更高 — 也就是 model 实际存的比 3.6 bits/param 还多。

4. **Training dynamics**: paper 测的是 final model。但 training 过程中 memorization → generalization 的切换动态? Figure 5 给了一点 hint (bits memorized across training), 但没深入。这跟 grokking (https://arxiv.org/abs/2201.02277) 的动态应该有关。

5. **跟 Chinchilla scaling law (https://arxiv.org/abs/2203.15556) 的关系**: Chinchilla 说 optimal token-per-param ≈ 20。这篇 paper 的 capacity 约束给出另一个视角: 如果 token-per-param 太小, model 在 memorization regime; 太大, model 早就在 generalization regime 了, 多余的 data 边际效用递减。两者应该能串起来。

---

## Reference

- This paper (推测): https://arxiv.org/abs/2506.17910
- Carlini et al. 2023 "Quantifying Memorization": https://arxiv.org/abs/2202.07646
- Nakkiran et al. 2019 "Deep Double Descent": https://arxiv.org/abs/1912.02292
- Belkin et al. 2019 "Reconciling modern ML and bias-variance": https://arxiv.org/abs/1903.07571
- Allen-Zhu & Li 2024 "Physics of LMs 3.3": https://arxiv.org/abs/2404.05405
- Delétang et al. 2024 "Language Modeling is Compression": https://arxiv.org/abs/2309.10668
- Schwarzschild et al. 2024 "Adversarial Compression": https://arxiv.org/abs/2404.15146
- Das et al. 2024 "Blind baselines beat MIA": https://arxiv.org/abs/2406.16201
- Duan et al. 2024 "Do MIA work on LLMs?": https://arxiv.org/abs/2406.16201
- FineWeb: https://arxiv.org/abs/2406.17557
- Chinchilla: https://arxiv.org/abs/2203.15556
- Grokking: https://arxiv.org/abs/2201.02277
- Grunwald & Vitányi "Shannon vs Kolmogorov": https://arxiv.org/abs/cs/0410002

---

# How Much Do Language Models Memorize? 深度解析

## 一、论文核心 motivation

这篇 paper 来自 Meta FAIR (John X. Morris 等), 其核心 motivation 是要回答一个 long-standing 的问题: **modern language model 到底"记住"了多少 training data?** 过去的的工作 (Carlini et al.) 通过 extraction (能不能让模型生成某个 string) 或者 membership inference (能不能判断某个 sample 是否在训练集里) 来间接衡量 memorization, 但作者认为这些方法都有一个 fundamental flaw: **它们无法区分 model 输出一个 string 到底是因为 memorize 了, 还是因为 generalize 得好**。

举例: 训练 sample 是 `Q: What is 2^100? A: 1267650600228229401496703205376`, 如果模型能输出这个答案, 是因为它"记住"了这条数据, 还是因为它学会了算术规则? Extraction-based 定义会错判为高度 memorized, 但这显然不对。

作者的目标是: **分离 unintended memorization (关于特定 dataset 的 sample-level 信息) 与 generalization (关于 true data-generating process 的 population-level 信息)**。

## 二、Memorization 的形式化定义

### 2.1 Statistical view (Shannon information)

记号: 大写字母 (X, Θ) 表示 random variables, 小写字母 (x, θ) 表示 instances。

定义 mutual information:
$$\operatorname{mem}(X, \hat{\Theta}) = I(X, \hat{\Theta}) = H(X) - H(X \mid \hat{\Theta})$$

这里 X 是数据集的 random variable, $\hat{\Theta}$ 是 trained model 的 random variable, $H(X)$ 是 X 的 entropy, $H(X|\hat{\Theta})$ 是 given model 后 X 的 conditional entropy。

为了分离出 generalization, 作者引入 ground-truth model Θ (理想的真实数据生成分布), 并定义:

$$\operatorname{mem}_U(X, \hat{\Theta}, \Theta) = I([X \mid \Theta], \hat{\Theta}) = H(X \mid \Theta) - H(X \mid (\Theta, \hat{\Theta}))$$

这里 $X|\Theta$ 表示去除"可由真实分布解释"部分后剩余的 uncertainty, $[X|\Theta]$ 即"the residual information in X after fixing Θ"。直觉是: **unintended memorization 衡量的是 model 关于特定 dataset sample-level 的信息, 而 generalization 是 model 学到的可推广到整个分布的规律**。

generalization (intended memorization):
$$\operatorname{mem}_I(X, \hat{\Theta}, \Theta) = \operatorname{mem}(X, \hat{\Theta}) - \operatorname{mem}_U(X, \hat{\Theta}, \Theta)$$

### 2.2 Proposition 1: Super-additivity of Unintended Memorization

对于 i.i.d. samples $X = (X_1, \ldots, X_n)$:
$$\sum_{i \in [n]} \operatorname{mem}_U(X_i, \hat{\Theta}, \Theta) \leq \operatorname{mem}_U(X, \hat{\Theta}, \Theta) \leq H(\hat{\Theta})$$

含义: 
- 下界: 单样本 memorization 之和 ≤ 总 memorization (因为可能存在冗余)
- 上界: 总 memorization ≤ 模型本身的 entropy $H(\hat{\Theta})$, 即 capacity 上界

这为后续通过 per-sample memorization 估计 model capacity 提供了理论基础。

### 2.3 From Shannon to Kolmogorov

Problem: Shannon entropy 需要分布, 但我们手上只有一个 trained model $\hat{\theta}$ 和一个 sample $x$, 无法直接估计概率分布。

Solution: 切换到 Kolmogorov complexity — 基于 compression 的 information measure。

**Definition 2 (Kolmogorov complexity)**:
$$H^K(x) = \min_{f(p) = x} |p|$$
即用 computational model f (如 universal Turing machine) 表示 x 的最短 program 的长度。

$$H^K(x \mid \theta) = \min_{f(p, \theta) = x} |p|$$
即 given reference $\theta$ 后 x 的最短描述。

$$I^K(x, \theta) = H^K(x) - H^K(x \mid \theta)$$
Kolmogorov mutual information。

**Definition 3 (Kolmogorov memorization)**:
$$\operatorname{mem}_U^K(x, \theta, \hat{\theta}) = H^K(x \mid \theta) - H^K(x \mid (\theta, \hat{\theta}))$$

**Proposition 4** (来自 Grunwald & Vitányi 2004): 当 sample 数趋于无穷时, Kolmogorov memorization 的期望与 Shannon memorization 之差 bounded by 常数 $\epsilon$, 与 $\ell, \ell', n$ 无关。这保证了 Kolmogorov 定义是 Shannon 定义的合理 instance-level approximation。

### 2.4 用 model likelihood 估计 Kolmogorov

直接算 Kolmogorov complexity 是 uncomputable 的, 但可以用 compression algorithm 近似, paper 选择 **arithmetic coding** (因为其 code length 与 model likelihood 直接挂钩, 见 Shannon 1950 source coding theorem)。

具体估计:
- $H^K(x \mid \hat{\theta}) \approx -\log p(x \mid \hat{\theta})$: 即 negative log likelihood, $\hat{\theta}$ 是 target model
- $H^K(x \mid \theta, \hat{\theta}) \approx -\log \max\{p(x \mid \hat{\theta}), p(x \mid \theta)\}$: 即取 target 和 reference model 中更好的那个

reference model $\theta$ 的选择:
- **Synthetic data 实验**: 直接用 true uniform distribution (已知)
- **Text data 实验**: 用同架构、参数相近、训练在更大 superset 上的 model 当 oracle, 或直接用 large oracle model

## 三、Model Capacity 测量

### 3.1 Capacity 定义

**Definition 5**:
$$\operatorname{Capacity}(L) = \max_X \operatorname{mem}(X, L(X))$$

即给定 learning algorithm L, 在所有可能 dataset X 上能达到的最大 memorization。

### 3.2 Synthetic bitstring 实验

Setup:
- 数据: 均匀分布 random bitstrings, vocabulary size $V = 2048$, sequence length $S = 64$
- 模型: GPT-2 architecture, 1-8 layers, hidden dim 32-512, 100K-20M params
- 训练: $10^6$ steps, batch size 2048, Adam, bfloat16, 单 A100
- 每个 model-dataset pair 跑 5 个 random seeds

因为数据是纯随机的, 没有任何 pattern 可学, 所以 $\operatorname{mem}_U \approx \operatorname{mem}$, 这就隔离了 generalization。

Entropy 计算:
$$H(x^i) = NS \log_2 V$$
其中 $N$ 是 dataset size, $S$ 是 sequence length, $V$ 是 vocab size。

Memorization:
$$\operatorname{mem}(x^i, \hat{\theta}_i) = H(x^i) - H^K(x^i \mid \hat{\theta}_i) \approx H(x^i) + \log_2 p(x^i \mid \hat{\theta}_i)$$

### 3.3 关键发现

**Figure 1**: 当 dataset 足够大时, memorization 出现明显 plateau, 这就是 model 的 capacity limit。

**Figure 6**: bits-per-parameter $\alpha$ 估算:
- bfloat16: $\alpha \approx 3.51$ bits/param
- fp32: $\alpha \approx 3.83$ bits/param

**Table 1** 给出了详细的 breakdown, 不同 layer 数 (1, 2, 4, 8) 和 hidden dim (32, 64, 128, 256) 的组合。注意 fp32 相比 bf16 只提升了约 10%, 远小于 2x, 说明 fp32 多出来的 bit 大部分没用于 raw storage。

**有意思的对比**: Allen-Zhu & Li (2024) 用 quantization 估算约 2 bits/param, 这篇 paper 估算更高 (3.6 bits/param), 因为 quantization 是 lossy 的, 而这里是用 model 自身 likelihood 来估。

### 3.4 Precision 的影响

```
bfloat16 → 3.51 bits/param
fp32     → 3.83 bits/param
```

但 model 在 disk 上的 size 是翻倍的, 所以 **bits-per-parameter 增加 不到 10%**。直觉: 大部分额外 precision 没用于 storing more info, 可能是用于 representing smoother loss landscape。

### 3.5 Sequence length / Vocab size 验证

**Table 3 & 4** (Appendix) 通过 fix dataset size, 改变 sequence length S 和 vocab size V 来验证线性估计的稳定性:
- 调 S: 平均误差 1.7%
- 调 V: 平均误差 1.8%

这表明 capacity estimate $\alpha = 3.64$ 是 robust 的。

## 四、Disentangling 在 Text Data 上

### 4.1 实验设置

- 数据集: FineWeb (Penedo et al. 2024), 严格 deduplication
- Token sequences: 64 tokens
- Reference model: 同参数量、训练在 full data 上的 model, 或者 large oracle model

### 4.2 Figure 4 中的 train/test loss

对 fixed model size, 随着 dataset 增大, train loss 单调下降, test loss 先下降到 capacity, 然后随着 dataset 超过 capacity 开始 generalization 阶段, test loss 进一步下降。

### 4.3 Double descent 与 capacity 的关系

**Figure 3**: 这是 paper 最 intuition 的 figure 之一。double descent 现象 (Belkin et al. 2019, Nakkiran et al. 2019) 在 **dataset size 超过 model capacity (bits) 那个点** 开始发生。

直觉: 当 dataset 容量小于 model capacity, model 可以"轻松"地把每个 sample 都记住 (overfitting)。一旦 dataset 容量超过 capacity, model 没办法单独记住每个 sample, 被迫 share information across samples → generalization 启动 → test loss 开始下降。

这就是 grokking 现象的信息论解释。

### 4.4 Extraction rate 与 generalization

**Figure 8**: 提取率随 prefix 长度变化。32-token prefix 的 extraction rate 在 small training set 上接近 100%, 但随 dataset 增大逐步下降。然而当 dataset 足够大时, extraction rate **不会降到 0**, 而是收敛到 **test set extraction rate**。

结论: 当 dataset 足够大时, 所有成功的 training data extraction 都 attributable to generalization, 而不是 memorization。这是对 LLM extraction attack 的一个重要 caveat。

### 4.5 哪些 sample 被最 memorize?

**Figure 16 & Table 5**: 计算 each training document 的 TF-IDF, 与 memorization 做散点图。

TF-IDF 定义:
$$\text{TF-IDF}(d; \mathcal{D}) = \frac{1}{|d|} \sum_{w \in d} \log \frac{|\mathcal{D}|}{tf(w, \mathcal{D})}$$

其中 $d$ 是文档, $\mathcal{D}$ 是整个数据集, $tf(w, \mathcal{D})$ 是 word w 在 D 中出现总次数, $|d|$ 是文档长度。高 TF-IDF 意味着文档包含更多 rare words。

发现: **highest TF-IDF 的 sample 最容易被 memorize**, top 20 memorized sequences 中, 17 个是非英语 (日、中、希伯来、希腊)。直觉: model 没办法 generalize 这些 outlier, 只能"硬记"。

## 五、Membership Inference Scaling Law

### 5.1 Loss-based membership inference

最简单的 attack (Yeom et al. 2018): 给定 cutoff loss, loss 小于 threshold 的判为 train。

### 5.2 Scaling law 形式

观察: 固定 model capacity 时, M.I. F1 score 相对 dataset size 大致呈 sigmoid 形状。

拟合函数:
$$\operatorname{Membership}_{F_1}(\theta, \mathcal{D}) = \frac{1}{2} \left(1 + c_1 \sigma\left(c_2 \left(\frac{\operatorname{Capacity}(\theta)}{|\mathcal{D}|} + c_3\right)\right)\right)$$

其中 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 是 sigmoid。

变量含义:
- $\theta$: model
- $\mathcal{D}$: training dataset
- $|\mathcal{D}|$: dataset size (in samples)
- $\operatorname{Capacity}(\theta)$: model capacity (in bits), 由前面的 $\alpha = 3.64$ × param count 估
- $c_1, c_2, c_3$: 待拟合常数

拟合结果: $c_1 = 1.34, c_2 = -0.034, c_3 = -33.14$

### 5.3 Limiting behavior

当 $|\mathcal{D}| \to \infty$, F1 → 0.5 (随机猜), M.I. 变得不可能。

### 5.4 Validation on larger models

**Table 2**: 用 scaling law 倒推需要多大 dataset 才能达到目标 F1, 然后实际训练 model 验证。

举例: GPT2-XL (1.55B params) 想要 F1 = 0.95, 预测需要 $|\mathcal{D}| = 18,851,574$ samples, 实际测得 F1 = $95.85 \pm 0.8$, 误差 < 1.5%。

### 5.5 对现代 LLM 的 implications

现代 LLM 训练 token-to-parameter ratio 通常 $\geq 10^2$。代入 scaling law, F1 score 约为 0.5 — **statistically significant loss-based membership inference 几乎不可能**。这解释了 Das et al. (2024), Duan et al. (2024) 在实际 LLM 上 M.I. attack 失败的现象。

## 六、Intuition Building: 三个核心 insight

### Insight 1: Memorization 与 Generalization 是 model 信息存储的两种"形式"

从信息论角度, model 从数据中提取的 bits 不是单一类型:
- **Unintended memorization**: 关于特定 sample 的信息 (e.g., 26 万样本中那条日语文本)
- **Generalization**: 关于 population distribution 的规律 (e.g., 主谓宾结构)

二者之和 ≤ $H(\hat{\Theta})$。当 dataset 超过 capacity, model 开始 trade off: 把原来"记具体"的 capacity 转去学规律。

### Insight 2: Bits-per-parameter ≈ 3.6 是 Transformer 的 "信息存储常数"

这是 paper 最 actionable 的一个 number。Compare:
- Cover (1965), Gardner (1988): 单层 perceptron ~ 2 bits/param
- Allen-Zhu & Li (2024): ~ 2 bits/param (via quantization)
- This paper: ~ 3.6 bits/param (via likelihood)

差别可能来自 quantization 损失 vs. 完整 model likelihood 的区别。

Intuition: 一个 fp16 (16-bit) param 实际只能 store ~3.6 bits 的"自由"信息, 其他 bit 用于:
- representing the trained model in loss landscape
- redundancy for robustness
- avoiding overfitting to noise

### Insight 3: Double Descent 是 Capacity Constraint 触发的 Generalization 启动

Nakkiran et al. (2019) 发现 double descent, 但没给出简洁解释。这篇 paper 给了一个 clean 的 information-theoretic account:

**当 dataset 大小 ≤ model capacity, model 在 "raw storage" regime, test loss 高 (overfitting)**
**当 dataset 大小 > model capacity, model 被迫 share representation across samples, 进入 generalization regime, test loss 下降**

这就是 grokking 的信息论起源。

## 七、Limitations & Open Questions

1. **Architecture specific**: 只测了 GPT-2 family, 没测 Mamba, RWKV, hybrid 等其他 architecture
2. **Reference model 影响**: text experiment 中 reference model 的选择会结果产生 sensitive 影响
3. **Compression algorithm 选择**: 用 arithmetic coding 是一种 instantiation, 用其他 compressor 可能得到不同 $\alpha$
4. **未 cover 的情况**: instruction-tuning 后的 model, RLHF 阶段如何影响 memorization?

## 八、Reference 与延伸阅读

- Paper: https://arxiv.org/abs/2506.17910 (推测 arxiv 编号)
- Delétang et al. 2024 "Language Modeling is Compression": https://arxiv.org/abs/2309.10668 — 基础的 LM-as-compressor 思路
- Nakkiran et al. 2019 "Deep Double Descent": https://arxiv.org/abs/1912.02292 — double descent 原始论文
- Carlini et al. 2023 "Quantifying Memorization": https://arxiv.org/abs/2202.07646 — extraction-based memorization 经典
- Schwarzschild et al. 2024 "Rethinking LLM Memorization": https://arxiv.org/abs/2404.15146 — 通过 prompt optimization 的 memorization 定义
- Allen-Zhu & Li 2024 "Physics of LMs Part 3.3": https://arxiv.org/abs/2404.05405 — 量化估算 2 bits/param
- Brown et al. 2021 (定义 Shannon memorization 的早期工作): https://arxiv.org/abs/2106.02521
- FineWeb dataset: https://arxiv.org/abs/2406.17557

## 九、Take-away for Practitioners

1. **估算 model capacity**: 一个 8B param 的 LLM 大约能存 $8 \times 10^9 \times 3.6 \approx 2.9 \times 10^{10}$ bits $\approx 3.6$ GB 的 "raw training data"
2. **现代 LLM 训练 dataset 远超 capacity**: 8B model 训 15T tokens (估算 $\sim$ TB 级 unique data) → 几乎全部 in "generalization regime", extraction attack 在 population level 上不再 effective
3. **Outlier 文本最危险**: 即使整体 dataset 巨大, 极少数 rare tokens / 非英语文本仍可能被 memorize, 这就是为什么 Carlini et al. 能 extract 出 canary sequences
4. **Membership inference 不可能**: 当 token-per-param ratio $\geq 10^2$, loss-based M.I. attack 在统计意义上无法 distinguish train vs. test
