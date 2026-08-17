---
source_pdf: Memory Mosaics at scale.pdf
paper_sha256: 6d2269dcc02fa980c396d245d72fbd4068b6dab1247b337e30103922242f2362
processed_at: '2026-08-05T17:29:32-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Memory Mosaics at scale

## 一句话总结

把transformer的attention重新当成"关联记忆"来设计，去掉位置编码、让query和key用同一套公式、让kernel的"锐度"随context长度自适应，再加一个"短期/长期/永久"三层记忆分工——结果训练1T tokens的MMv2，在"推理时学新东西"这件事上，打过了训练8T tokens的transformer。

---

## 为什么这个工作有意思

Karpathy你应该最懂这个痛点：现在的LLM有几个让人不舒服的地方——

- **In-context learning在长context下反而变差**。给它10个demo比给5个demo还烂，这违反直觉。Li et al. 2024（https://arxiv.org/abs/2404.02060）专门测了，确实是普遍现象。
- **Position encoding让length generalization很难**。RoPE在4k训完，扔到32k就崩，必须fine-tune几百个batch才缓过来。
- **新信息存不住**。给它20篇文章再问问题，transformer在32k长度上只能答41%的RULER multi-doc QA，几乎瞎猜水平。
- **架构里塞了太多designer的先验**。position encoding怎么加、q和k为什么要用不同矩阵、temperature为什么是 $1/\sqrt{d}$——这些都是人工选择，模型没机会自己学。

Zhang & Bottou这帮人的想法是：**能不能把attention退回到它最原始的统计形式——kernel regression，然后用最少的inductive bias，让模型自己学出该学的东西**。

---

## 关键insight：attention本来就是kernel regression

把Gaussian kernel regression写出来：

$$
f(k) = \sum_i \frac{e^{-\beta\|k - k_i\|^2}}{\sum_j e^{-\beta\|k - k_j\|^2}} v_i
$$

如果所有 $k_i$ 的L2 norm一样（做normalize），这个式子展开就是：

$$
f(k) = \sum_i \text{softmax}(\beta \cdot k^\top k_i) \cdot v_i
$$

这就是attention。$\beta$ 就是 $1/\sqrt{d}$ 那个temperature。

所以attention本来就是一个kernel smoothing estimator，只是transformer把它包装得太花，把它的统计意义藏起来了。Memory Mosaics干的事是：**把这个kernel estimator重新暴露出来，给它该有的自由度**。

具体三个自由度：

---

## 改动一：bandwidth要随context长度变化

统计学里有个老结果（Hastie ESL第6章，https://hastie.su.domains/ElemStatLearn/）：kernel regression的最优bandwidth满足

$$
1/\sqrt{\beta} \propto n^{-1/(p+4)}
$$

意思：**样本越多，kernel越要尖锐**。因为key空间越拥挤，你想区分一个key就得用更窄的kernel。

Transformer的 $\beta = 1/\sqrt{d}$ 是死的，跟context length无关。这就是为什么transformer在long context下attention分布退化——kernel太钝，所有key的权重趋近平均，远端信息被洗掉。

v2直接让 $\beta$ 学一个power law：

$$
\beta = \beta_1 n^\alpha + \beta_0
$$

context越长，$\beta$ 越大，kernel越尖锐。这是个有理论指导的architectural choice，不是hyperparameter search。

---

## 改动二：key要会"忘记"

v1的key是简单leaky average：

$$
\bar{k}_T = \tilde{k}_T + \lambda \bar{k}_{T-1}
$$

$\lambda$ 是固定值，跟输入无关。问题："tom and jerry"和"tom - - and - - jerry"会得到不同的key，因为leaky window的滑动速度一样，但pattern的密度不同。

v2借用LSTM/Mamba的门控思路：

$$
g_T = e^{W_g x_T} \quad \text{(input gate, 当前token贡献多少)}
$$
$$
\lambda_T = e^{-|W_\lambda x_T|} \quad \text{(forget gate, 历史保留多少)}
$$
$$
\bar{k}_T = g_T \cdot \tilde{k}_T + \lambda_T \cdot \bar{k}_{T-1}
$$

两个gate都depend on当前输入。模型可以选择在空格处"忽略当前、保留历史"（$g_T \approx 0, \lambda_T \approx 1$），让"tom and jerry"和"tom - - and - - jerry"映射到同一个key。这给了模型semantic-aware的时间平滑。

---

## 改动三：短期/长期/永久三层记忆

这是最有意思的部分。他们先做了一个observation（Figure 1）：

把transformer和memory mosaics各层的attention score画出来——
- Transformer：曲线弯弯曲曲，每个位置都不一样，**全位置dependent**
- Memory Mosaics：近端256个token位置dependent，远端400+个token之后**曲线变flat**，跟位置无关，只跟key similarity有关

这说明memory mosaics的attention **天然分化**成两个regime：
- 近端：处理syntax、phrase-level、word order（位置敏感）
- 远端：处理long-range retrieval、topic（位置无关）

v2直接把这个分化 **explicit化**，每个block配两个memory：

- **Short-term memory**：只看最近256个token，local composition
- **Long-term memory**：跳过最近64个token，只看历史远端，global retrieval
- **Persistent memory**：SwiGLU FFN，存训练时学到的知识

训练时long-term的"跳过长度"$m$ 从[64, 256]随机采样，推理时固定64。这个随机采样逼模型学一个robust的short/long boundary，结果context length extrapolation涨15.8%。

**最关键的一点**：训练完可以把long-term memory剪掉，13个persistent-knowledge benchmark几乎不掉点（56.8% → 56.6%）。说明：
- persistent knowledge存在persistent memory + short-term就够了
- long-term memory专门服务new-knowledge storage和ICL
- 这个disentanglement是 **真的disentangle**，不是参数冗余

---

## 三个评估维度

Karpathy你会欣赏这个评估框架。单看一个维度都会misleading：

### Dimension 1：Persistent knowledge（训练时学的知识）
19个常见benchmark（MMLU、HellaSwag、ARC等）。MMv2和transformer持平（52.2 vs 52.2）。**Expected**——persistent memory架构一样。

### Dimension 2：New-knowledge storage（推理时存新信息）
RULER的multi-unrelated-documents QA。20篇文章拼起来，问一个问题，答案只在一篇里。

| Model | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| Transformer 1T | 51.2 | 48.8 | 44.7 | 41.1 | X |
| MMv2 1T | 58.9 | 55.5 | 54.9 | 53.4 | 46.4 |

32k上+12.3%，64k上transformer直接挂。这是associative memory的强项——lazy retrieval，等query来了再找，信息不丢失。

**为什么memory compression方法（Mamba/RWKV/xLSTM）全挂**：它们必须在线压缩state，信息必然丢失。RULER 4k上llama2-7b 95%，RWKV只有51%，Mamba-2.8B在Tacred ICL上全0。压缩式模型在这类任务上 **结构性失败**，不是调参能救的。

### Dimension 3：In-context learning（推理时学新任务）
3个分类任务（Banking77/Tacred/GoEmotions），有semantic label版和anonymous label版。

两个发现：
- MMv2随shot数增加 **单调上升**；transformer随shot数增加 **反而下降**（long-context ICL失效现象）
- MMv2比transformer高10%+，anonymous版高得更多

Anonymous版才是关键：把label换成"class_00"、"class_01"，强制模型必须从demo学，不能依赖训练时学到的"happy是什么意思"。Transformer在这上面几乎学不会，因为它的induction head依赖semantic prior；MMv2因为symmetric key/query + 无position encoding，induction head一层就work，纯靠pattern matching。

---

## 最striking的对比：8x data补不上架构差距

Section 6是paper的高潮。问题：如果坚持transformer recipe，要多少training tokens才能赶上MMv2？

训了3个transformer（200B / 1T / 8T）vs 1个MMv2（1T）：

- **RULER 32k QA**：transformer 8T达到46.9%，MMv2 1T是53.4%，还差6.5%
- **Semantic ICL**：transformer 8T勉强追上MMv2 1T
- **Anonymous ICL**：transformer 8T **仍然显著落后**，而且出现反直觉现象——更多data反而更差

**结论**：在需要纯架构能力的任务上，scaling data解决不了架构短板。8倍data的算力都补不回来。

---

## Fine-tuning速度：1个mini-batch vs 800个

4k pretrain → 32k fine-tune：

- MMv2：1个mini-batch提升22%，2个mini-batch达到optimal
- Transformer：800个mini-batch还赶不上MMv2的1个mini-batch

为什么差这么多：MMv2的架构本身就是为长context设计的，adaptive bandwidth自动适应新context length，只需要gradient信号告诉模型"现在context是32k了"。Transformer的RoPE必须重新scale，所有attention layer都要重新适应position range，需要大量iter。

**对deployment的意义**：如果你要快速adapt一个模型到新domain/新context length，MMv2需要的fine-tune data比transformer少 **800倍**。

---

## 这工作在整个field里的位置

### 对Transformer正统派的挑战

主流LLM的信仰是"more data more compute + old recipe"。Llama 3 herd paper（https://arxiv.org/abs/2407.21783）就是这个思路的极致：堆15T tokens，架构基本不动。

MMv2说：**这条路有天花板**。至少在new-knowledge storage和anonymous ICL上，8x data补不上架构差异。而且我们快没数据了——8T已经接近public text数据的极限。

### 对Mamba/RWKV/xLSTM的否定

这些模型追求sub-quadratic complexity，但代价是compressed state。MMv2的实证显示：在long-context retrieval + ICL任务上，压缩式方法结构性失败。Table 7/8的数据很硬：

| Model | RULER 4k |
|---|---|
| llama2-7b | 95% |
| rwkv-v5-7b | 51.4% |
| mamba-2.8b | 62.6% (2k就崩) |

**取舍是错的**：为了省compute牺牲retrieval能力，在realistic task上得不偿失。

### 对Karpathy一直关心的问题

你之前在多个talk里提过的几个痛点，MMv2都给了不同答案：

- **"ICL在长context下失效"** → long-term memory用associative buffer，不压缩，position-invariant retrieval
- **"Position encoding让length generalization难"** → 直接不用position encoding，让key自己编码时序
- **"Fine-tune需要的data太多"** → 2个mini-batch就够
- **"架构inductive bias没被充分利用"** → adaptive bandwidth来自kernel statistics，gated temporal来自RNN，三层memory来自对attention score的empirical observation，每个都有理论或观察支撑

---

## 几个没解决的问题（Karpathy可能会问的）

1. **Reasoning能力**：gsm8k/math上MMv2和transformer持平，没看到deep reasoning（比如multi-step chain-of-thought）的对比。Adaptive bandwidth让kernel变尖锐，可能不利于"联想式推理"。
2. **Instruct tuning / RLHF**：只测了base model。MMv2在instruction following、ChatBot场景下未知。
3. **100k+ context**：没测。Section 8提到要用fuzzy hashing（MagicPIG，https://arxiv.org/abs/2410.16179，同lab的工作）和hierarchical memory（NSA https://arxiv.org/abs/2502.11089，MoBA https://arxiv.org/abs/2502.13189）进一步scale。
4. **Pretraining loss curve / scaling law**：没给loss vs compute的图。不知道MMv2在Chinchilla-optimal data下是更sample efficient还是less efficient。
5. **Long-term memory的 $O(n)$ retrieval cost**：还是quadratic总复杂度，跟full attention一样。Fuzzy hashing可能能解但没在这篇paper里。
6. **为什么anonymous ICL上8T transformer反而更差**：Figure 7那个counterintuitive degradation没解释清楚。可能hypothesis：更多pretraining data让transformer更依赖semantic prior，anonymous label下更难适应；但这是猜测。

---

## 最后的intuition

Bottou这帮人想说的是：

**Transformer的很多"必要组件"（position encoding、asymmetric q/k、fixed temperature）其实是implementation choice，不是intelligence的必要条件。把它们去掉，用最少的inductive bias（symmetric kernel + L2 norm + adaptive bandwidth + gated temporal + three-level memory），模型反而能学到更general的能力——至少在"推理时学新东西"这件事上。**

不是"transformer不行"，是"transformer的某些设计选择在new-knowledge/ICL场景下成了瓶颈"。MMv2给了一条结构性的出路，而且这条路在小scale验证过，在大scale + 真实数据上依然work，甚至8x data都补不上。

这工作对你一直在强调的"build intuition first"是完美契合——从kernel regression这个简单起点出发，通过architectural modifications一层层build出能match 8T transformer的1T model。比起"堆data堆compute"的scaling story，这是个 **"structure matters"** story。

---

## Web references

- Memory Mosaics v1 (ICLR 2025): https://arxiv.org/abs/2505.20350
- Birth of a Transformer (Bietti et al.): https://arxiv.org/abs/2306.00802
- Induction Heads (Olsson et al.): https://arxiv.org/abs/2209.11895
- Long-context LLMs struggle with ICL (Li et al.): https://arxiv.org/abs/2404.02060
- RULER benchmark: https://arxiv.org/abs/2404.06654
- Mamba: https://arxiv.org/abs/2312.00752
- RWKV: https://arxiv.org/abs/2305.13048
- xLSTM: https://arxiv.org/abs/2405.04517
- Llama 3: https://arxiv.org/abs/2407.21783
- MagicPIG (fuzzy hashing for LLM, same lab): https://arxiv.org/abs/2410.16179
- NSA (Native Sparse Attention): https://arxiv.org/abs/2502.11089
- MoBA: https://arxiv.org/abs/2502.13189
- Elements of Statistical Learning: https://hastie.su.domains/ElemStatLearn/
- Lost in the Middle: https://arxiv.org/abs/2307.03172
- SwiGLU: https://arxiv.org/abs/2002.05202
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Kernel regression (wiki): https://en.wikipedia.org/wiki/Kernel_regression

---

# Memory Mosaics at Scale: 详解与intuition

## 1. Big picture (一句话核心)

这篇paper把上一代的Memory Mosaics (Zhang et al. 2025, ICLR 2025, https://arxiv.org/abs/2505.20350) 从GPT-2 scale推到llama-8B scale, 在1T tokens真实数据上训练, 得到 **Memory Mosaics v2**。核心claim: 在 **persistent knowledge**（训练时学到的知识）上与transformer持平, 但在 **new-knowledge storage**（推理时存新信息）和 **in-context learning**（推理时学新任务）上比transformer高10%+。最striking的对比: 一个用1T tokens训练的MMv2, 在新任务学习上击败用8T tokens训练的transformer, 差距6-15个百分点。

整个工作的motivation来自一个透明视角: transformer的induction head、compositional generalization、ICL机制都比较黑箱, 而associative memory viewpoint把attention看成kernel regression (Nadaraya-Watson estimator, 1964, https://en.wikipedia.org/wiki/Kernel_regression), 这样key/query的对称性、position encoding的角色、bandwidth的作用都能被显式讨论, 从而可控、可scale、可解释。

---

## 2. Background: Associative Memory与Attention的本质联系

### 2.1 Associative memory的形式化

一个associative memory是存储key-value对 $\{(k_1, v_1), \dots, (k_n, v_n)\}$ 的设备, 给一个query $k$, 返回一个value。Formally, retrieval function

$$
f(k; \{(k_i, v_i)\}_{i=1}^n) = \mathbb{E}[V \mid K = k]
$$

即把memory看成对 $P(V|K)$ 的估计, retrieval就是conditional expectation。

用Gaussian kernel regression估计这个期望:

$$
f(k) = \sum_{i=1}^n \frac{e^{-\beta \|k - k_i\|^2}}{\sum_{j=1}^n e^{-\beta \|k - k_j\|^2}} \cdot v_i \tag{3}
$$

变量含义:
- $k \in \mathbb{R}^d$: query向量（在transformer里就是query向量）
- $k_i \in \mathbb{R}^d$: 第$i$个key向量
- $v_i \in \mathbb{R}^d$: 第$i$个value向量
- $\beta > 0$: Gaussian kernel的 **inverse bandwidth** (越大kernel越尖锐、越selective, 越小越smooth/平均化)
- $n$: 存储的样本数（= sequence length）
- $e^{-\beta\|k-k_i\|^2}$: Gaussian kernel值, 衡量query与key的similarity
- 分母: partition function / normalization, 让权重成为概率分布

### 2.2 与attention的精确联系

关键一步: 如果所有key向量 $k_i$ 的 **squared norm相同**（$||k_i||^2 = c$ 对所有$i$）, 则:

$$
\|k - k_i\|^2 = \|k\|^2 + \|k_i\|^2 - 2k^\top k_i = \|k\|^2 + c - 2k^\top k_i
$$

常数项 $\|k\|^2 + c$ 在分子分母约掉, 得到:

$$
f(k) = \sum_{i=1}^n \frac{e^{\beta k^\top k_i}}{\sum_{j=1}^n e^{\beta k^\top k_j}} \cdot v_i \tag{4}
$$

这就是scaled dot-product attention, 只不过用 $\beta$ 代替了 $1/\sqrt{d}$。

**Intuition**: 当key做了L2 normalize, Gaussian kernel回归就等价于attention。两者本质是同一个estimator。

### 2.3 Associative Memory相对Attention的三个差异

Memory Mosaics显式保留这三个差异, 而transformer丢掉了:

1. **L2 normalized keys + 显式β**: transformer用 $1/\sqrt{d}$ 作temperature, 训练时固定; associative memory用learnable $\beta$, 而且让它depend on样本数 $n$（v2的核心改进）。
2. **Symmetric kernel**: query和key用同一个变换（$k_T = \varphi(x_T, x_{T-1}, \dots)$）, 而transformer用 $q_T = W_q x_T$, $k_T = W_k x_T$ 两个不同矩阵。这一对称性让induction head只需要一层就能实现。
3. **无position encoding**: transformer必须有RoPE/ALiBi才能work; associative memory通过让key encode recent past、value encode near future, 用key自己当query, 自然实现了序列建模。

这三个差异看似很小, 却带来质变: induction head用一层就成、context length extrapolation天然work、compositionality显著提升。这些都是transformer经过精心设计才能得到的属性, 在associative memory里几乎"免费"。

---

## 3. Memory Mosaics原始设计回顾（v1）

参考 https://arxiv.org/abs/2505.20350 (ICLR 2025)。原始Memory Mosaics的关键设计是induction-head-inspired key/value构造:

$$
k_T = \varphi_\theta(x_T, x_{T-1}, \dots), \quad v_T = \psi_\theta(x_{T+1}, x_T, \dots) \tag{5}
$$

直觉: key编码"recent past"（之前的token序列）, value编码"near future"（之后的token, 包括当前token）。当query是当前token, key也是相同形式, 那 retrieval $f(k_T; \{(k_i, v_i)\})$ 就是在所有"过去发生过的类似context"里取value的加权平均, 这个value就是"过去的类似context之后的next token"。

这就是induction head: 看到 $[\dots, a, b, \dots, a]$ 就预测 $b$。Transformer需要至少两层attention + position encoding才能学到induction head（参考 Bietti et al. 2023, "Birth of a Transformer: A Memory Viewpoint", https://arxiv.org/abs/2306.00802）, Memory Mosaics一层就够。

v1在GPT-2 scale + 合成数据上work, 但scale到llama-8B + 真实数据时, 暴露出问题:
- 固定bandwidth不适应变化序列长度
- 时不变的leaky averaging让语义相似的pattern映射到不同key
- 单一memory机制混淆了近端（position-dependent）和远端（position-invariant）的信号

v2针对这三点开刀。

---

## 4. Memory Mosaics v2: 三个architectural modifications详解

### 4.1 Adaptive bandwidth (Section 3.1)

**问题**: kernel regression的bias-variance trade-off依赖bandwidth $\beta$:
- $\beta$ 太大（kernel尖锐）→ variance高, overfit到具体样本
- $\beta$ 太小（kernel平滑）→ bias高, 算出的是global mean

经典的asymptotic Mean Integrated Squared Error (MISE) 给出: 对于维度 $p$ 的数据, optimal bandwidth满足 $1/\sqrt{\beta} \propto n^{-1/(p+4)}$, 即样本数越多, bandwidth越小, kernel越尖锐。

**v2的修正**: 让 $\beta$ 随 $n$（sequence length）变化:

$$
\beta = \beta_1 \cdot n^\alpha + \beta_0 \tag{6}
$$

变量:
- $\beta_0 \geq 0$: 基础bandwidth, 处理 $n \to 0$ 的退化情况
- $\beta_1 > 0$: scaling coefficient
- $\alpha \in (0, 1)$: power law exponent, 控制bandwidth对 $n$ 的敏感度
- $n$: 当前associative memory存储的key-value对数

reparameterization（Appendix Table 6）:
- $\beta_0 = e^{\min(\theta, 10)}$, $\theta$ init = 1.5
- $\beta_1 = e^{\min(\theta, 10)}$, $\theta$ init = 1.5
- $\alpha = \min(|\theta|, 1)$, $\theta$ init = 1/3

$\exp$ reparameterization保证正数, $\min$ clamp防止数值爆炸。

**Intuition**: 想象context长度4k vs 32k, 32k时key空间更稠密, 想要distinguish一个key需要更尖锐的kernel, 自然要更大 $\beta$。transformer的 $1/\sqrt{d}$ 是死值, 跟context length无关, 这就是transformer在long context下attention分布退化、induction head失效的根本原因之一。

### 4.2 Gated time-variant key feature extractor (Section 3.2)

**v1设计**:

$$
\tilde{k}_T = W_\varphi x_T, \quad \bar{k}_T = \tilde{k}_T + \lambda \bar{k}_{T-1}, \quad k_T = \text{Norm}(\bar{k}_T) \tag{7}
$$

leaky average, 固定 $\lambda$, 等价于一个exponential moving average。问题: $\lambda$ 与输入无关, 所以 "tom-and-jerry"（连续）和 "tom- - -and- - -jerry"（带空格）会得到非常不同的key特征, 因为leaky averaging的窗口长度固定, 两个pattern滑过去的速度不一样。

**v2设计** (受RWKV/Mamba/xLSTM启发, https://arxiv.org/abs/2305.13048, https://arxiv.org/abs/2312.00752, https://arxiv.org/abs/2405.04517):

$$
\tilde{k}_T = W_\varphi x_T
$$
$$
g_T = e^{W_g x_T} \in \mathbb{R} \quad (\text{exponential gate, 输入新信息的权重})
$$
$$
\lambda_T = e^{-|W_\lambda x_T|} \in \mathbb{R} \quad (\text{decay rate, 历史的权重})
$$
$$
\bar{k}_T = g_T \cdot \tilde{k}_T + \lambda_T \cdot \bar{k}_{T-1}
$$
$$
k_T = \text{Norm}(\bar{k}_T) \tag{8}
$$

变量:
- $W_\varphi, W_g, W_\lambda$: learnable参数矩阵
- $g_T$: 当前token对新key特征的"接受度", 用exp保证正值
- $\lambda_T$: 历史累加的衰减率, $|W_\lambda x_T|$ 让其在 $[0, 1]$ 之间（因为 $e^{-|x|} \leq 1$）

**Intuition**: $g_T$ 像"LSTM input gate", 控制当前token贡献多少到state; $\lambda_T$ 像"forget gate", 控制历史state衰减多少。两者都depend on当前输入 $x_T$, 所以模型可以"决定"在某个token上"忘记历史、纯用当前"（$g_T$大, $\lambda_T$小）或者"忽略当前、保留历史"（$g_T$小, $\lambda_T$大）。

对于 "tom- - -and- - -jerry" 这种pattern, 模型可以选择让空格处的 $g_T \approx 0$（不被污染）, $\lambda_T \approx 1$（保持历史）, 这样空格的存在不会污染"tom-and-jerry"的key。这给了模型semantic-aware的时间平滑。

**Value extractor** (保持v1设计):

$$
\tilde{v}_T = W_\psi x_T, \quad \bar{v}_T = \gamma \tilde{v}_T + (1-\gamma) \tilde{v}_{T+1}, \quad v_T = \alpha_\psi \text{Norm}(\bar{v}_T) \tag{9}
$$

变量:
- $\gamma \in [0, 1]$: 当前token与下一个token在value里的混合比例, init为 $\mathcal{U}(0,1)$
- $\alpha_\psi$: learnable scaling, $\alpha_\psi = e^{\min(|\theta|, 15)}$, init $\theta = 0$ (即 init=1)

Value convolve当前与未来, 让value代表"next token的预期"。

注意: value extractor用了 $\tilde{v}_{T+1}$, 即下一个token的投影。这是causal的（因为 $v_T$ 是和 $k_T$ 一起存进memory的, retrieval时只用到对应的 $v_i$, 但训练时需要 $x_{T+1}$ 构造 $v_T$）。这等价于在target prediction上做了smooth。

### 4.3 3-level memory design (Section 3.3)

这是v2最重要的architectural变化。基于Figure 1的观察:

- Transformer的average attention score对position高度依赖（曲线蜿蜒）
- Memory Mosaics的attention score: 近端（position 450以内）高度position-dependent, 远端（position 0~450）几乎position-invariant（曲线flat）

这说明memory mosaics的attention score天然分化成两个regime:
- **Position-dependent** (近端): 处理local word-order syntax、phrase-level composition
- **Position-invariant** (远端): 处理long-range retrieval、global topic、document-level recall

v2显式把这两种regime分离成两种memory:

#### Short-term memory

存 $[t-h+1, t-1]$ 的key-value对, $h = 256$:

$$
f_{\text{short}}(k; \{(k_{t-h+1}, v_{t-h+1}), \dots, (k_{t-1}, v_{t-1})\}) = \sum_{i=t-h+1}^{t-1} \frac{e^{\beta k^\top k_i}}{\sum_{j=t-h+1}^{t-1} e^{\beta k^\top k_j}} v_i \tag{10}
$$

只在最近的256个token上做attention, 完全local。

#### Long-term memory

存 $[1, t-m]$ 的key-value对, $m$ 在训练时从 $[64, 256]$ 随机采样, 推理时 $m=64$:

$$
f_{\text{long}}(k; \{(k_1, v_1), \dots, (k_{t-m}, v_{t-m})\})
$$

跳过最近 $m$ 个token, 只用历史"远端"信息。

**关键设计**: $m < h$ 制造overlap, 所以short-term和long-term在 $[t-256, t-64]$ 之间是overlap的。这是个 **soft boundary**: 一个token既可能进short-term memory, 又可能进long-term memory, 两者各自fetch然后concatenate。Overlap避免了hard split带来的不连续性。

**为什么训练时随机采样 $m$**: 训练时模型不知道推理时 $m$ 会是64, 所以被迫学一个representation既能处理 $m=64$（短跳过）, 又能处理 $m=256$（长跳过）。这等价于一个 **context-length-extrapolation-friendly regularization**: short/long的boundary必须robust到不同的offset。Table 12显示这个trick给32k context length extrapolation提升了15.8%。

#### Persistent memory

每个block还配一个SwiGLU feedforward network (https://arxiv.org/abs/2002.05202):

$$
\text{FFN}(x) = W_2 (\text{SiLU}(W_1 x) \odot W_3 x)
$$

其中 $\text{SiLU}(x) = x \cdot \text{sigmoid}(x)$。这里存的是训练时积累的global persistent knowledge（类似transformer FFN的角色, 参考 Sukhbaatar et al. 2019 "Augmenting self-attention with persistent memory", https://arxiv.org/abs/1907.01470）。

#### 三层memory的合奏

每个Memory Mosaics v2 block输出:

$$
\text{out} = W_o \cdot \text{concat}(f_{\text{short}, 1}, f_{\text{long}, 1}, \dots, f_{\text{short}, H}, f_{\text{long}, H}) + \text{FFN}(x)
$$

其中 $H$ 是head数, $W_o$ 是输出projection。

**Intuition**: Transformer把所有信息都混在attention里, 必须靠位置encoding区分local/global, 但位置encoding在long context下generalize不好。Memory Mosaics v2直接显式分离: short-term做local composition, long-term做global retrieval, persistent存global knowledge。每个memory有自己的参数和inductive bias, 三者通过gradient自然分工。

Figure 1的实验验证: 这种分工在v1的单一memory里就自发涌现, v2只是把它 **explicit化**, 让模型不用learn出这个分离, 而是从architecture里直接得到。

---

## 5. Training setup

### 5.1 规模

- **MMv2 small**: 24 layers, 2048 hidden dim, 16 heads, ~1.5B params, 训练200B tokens
- **MMv2 large**: 32 layers, 4096 hidden dim, 32 heads, ~9.9B params, 训练1T tokens

参数对比 Table 2:
- Transformer large: 8.8B params, 16.7B FLOPs/token
- MMv2 large: 9.9B params, 18.9B FLOPs/token
- MMv2 large without long-term (训练后剪掉): 8.3B params, 15.6B FLOPs/token

MMv2比transformer多12%的params和FLOPs, 主要来自多了一套long-term memory参数。但训练后可以把long-term memory剪掉, 在13个persistent-knowledge task上几乎不掉点（56.8%→56.6%）, 同时反而更小更便宜。

### 5.2 优化器与schedule

- AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.95$, L2 weight decay 0.1, gradient clip 1.0
- LR warmup 2000 iters, cosine decay到1/100
- Initial LR: 3e-4 (small), 1e-3 (large)
- Batch size 1024, sequence length 4096
- Document-wise attention mask（每个document独立attention, 减少cross-doc污染）

### 5.3 Stochastic long-term memory

训练时 $m \sim \mathcal{U}[64, 256]$, 推理时固定 $m=64$。这个训练setup提升context extrapolation 15%+。

### 5.4 Context length extrapolation

4k pretrain → 32k fine-tune。Table 12显示, 即便不fine-tune, MMv2 small在32k task-length上能达到31.7%（有stochastic setup）vs 15.9%（无stochastic setup）。Transformer small则完全不能extrapolate（4k训完直接送32k崩掉）。

### 5.5 参数初始化 (Appendix Table 6)

- $W_\psi, W_\varphi, W_g, W_\lambda, W_o$: $\min(\max(\mathcal{N}(0, \sigma), -3\sigma), 3\sigma)$, $\sigma = 1/\sqrt{2d(l+1)}$, $l$ 是block depth
- 这是个clipped Gaussian, 跟原始transformer init类似, 但用 $l+1$ 让深层sigma更小

---

## 6. 三个评估维度

这是paper最有思想性的部分。作者提出 **三维评估**, 因为单看任何一维都 misleading:

### 6.1 Dimension 1: Persistent-knowledge storage/retrieval

19个常见benchmark (obqa, arc-easy, winogrande, arc-challenge, piqa, boolq, hellaswag, siqa, nq, tqa, gsm8k, alt, mmlu, humaneval+, squad, bbh, math, mbpp, race-middle/high)。

Table 1结论: MMv2与Transformer持平 (avg 52.2 vs 52.2 large)。

**关键消融 (Table 2 + Appendix Table 9)**: 训练后把long-term memory剪掉:
- 13个benchmark: 56.8% → 56.6%（几乎不掉）→ 这13个task **不需要** long-term memory
- 6个benchmark (squad, bbh, math, mbpp, race-middle, race-high): 42.1% → 34.9%（掉7.2%）→ 这6个task **依赖** long-term memory

13个persistent-knowledge benchmark: obqa, arc-easy, winogrande, arc-challenge, piqa, boolq, hellaswag, siqa, nq, tqa, gsm8k, alt, mmlu, humaneval+

**Intuition**: 这13个task考察的是"模型从训练数据学到的知识", 推理时只需要persistent memory + short-term memory就够了。所以MMv2在这维度上和transformer持平是 **expected**——两者persistent memory架构一样, 都用SwiGLU FFN。

那剩下的6个task呢？squad要读passage找答案（需要new-knowledge storage）, bbh/math/mbpp需要复杂多步推理（可能需要in-context的chain）, race-middle/high需要长passage理解, 这些都触及long-term memory。

### 6.2 Dimension 2: New-knowledge storage/retrieval

用 RULER benchmark (https://arxiv.org/abs/2404.06654) 的 multi-unrelated-documents QA task:

```
Answer the question based on the given documents. The following are given documents. 
Document 1: [...] Document 2: [...] [...] Document 20: [..] 
Question: What religion were the Normans? Answer:
```

20篇不相关的article拼接, 最后问一个只跟其中一篇相关的问题。这比 needle-in-haystack难得多——后者信息熵极低（只是一句话藏在一堆junk text里）, 多数模型接近100%（Table 13显示MMv2和transformer在needle-in-haystack都接近100%）, 没有区分度。

**4k训练结果 (Table 3)**:
| Model | 4k task | 8k | 16k | 32k |
|---|---|---|---|---|
| Transformer large | 57.7 | X | X | X |
| MMv2 large | 59.3 | 48.8 | 46.4 | 26.5 |

Transformer在4k训练后无法处理 >4k的task, 直接X掉。MMv2在8k/16k还能work, 32k掉到26.5但还有意义。

**32k fine-tune后 (Table 4)**:
| Model | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| Transformer large | 51.2 | 48.8 | 44.7 | 41.1 | X |
| MMv2 large | 58.9 | 55.5 | 54.9 | 53.4 | 46.4 |

差距: 4k +7.7, 32k +12.3, 64k上transformer直接崩, MMv2还有46.4。

**Intuition**: MMv2的long-term memory是真正的"associative buffer", 能精确检索远的key-value对。Transformer的attention虽然也能attend远端, 但softmax在长context下分布退化（参考 Xiao et al. 2024 "Efficient Streaming Language Models with Attention Sinks", https://arxiv.org/abs/2309.17453）, effective attention集中在sink tokens和近端, 远端retrieval能力弱。

**为什么memory compression方法不行**: Table 7 + 8 显示 RWKV/Mamba/xLSTM在这类任务上崩盘:
- llama2-7b在RULER 4k上95%, RWKV-v5只有51.4%
- Tacred in-context learning: Mamba-2.8B全0, RWKV全1-2%

原因: memory compression必须在看到所有信息后再"压缩", 但压缩必然丢信息, 当task要求精确检索某篇article的某个细节时, 压缩后的state无法保留。Transformer和MMv2都能"先存着, 等query来了再retrieval", 这种"lazy evaluation"对new-knowledge storage是必要的。

### 6.3 Dimension 3: In-context learning

3个multiclass classification task:
- **Banking77** (https://arxiv.org/abs/2003.04807): 77类banking intent分类
- **Tacred** (https://aclanthology.org/D17-1004/): 41类关系分类
- **GoEmotions** (https://arxiv.org/abs/2005.00547): 28类情感分类

每个task有两种版本:
- **Semantic label**: label是"happy", "angry", "city_arrival"等有语义的
- **Anonymous label**: label是"class_00", "class_01"等无语义

anonymous版本的意义: 强制模型必须从demo的 $(x, y)$ pair学到任务, 不能依赖预训练里关于"happy"是什么的知识。

**Setup**: n-shot = 每个class给n个demo, 构造prompt "query: x_shot1, intent: y_shot1, ..., query: x_test, intent:"

**结果 Figure 3 (semantic)**: 
- MMv2随shot数增加 **单调上升**
- Transformer随shot数增加 **反而下降** (反直觉但和Li et al. 2024 https://arxiv.org/abs/2404.02060一致)
- MMv2比Transformer高10%+

**结果 Figure 4 (anonymous)**:
- Transformer在anonymous label上更惨, 几乎无法学习
- MMv2依然work, 高10-20%

**为什么Transformer随shot数下降**: 这是Li et al. 2024发现的long-context ICL失效现象。一个解释: 当shot数多, label需要retrieval的key离query太远（context length变长）, attention分布退化, 找不到对应demo。MMv2的long-term memory+adaptive bandwidth让远端retrieval保持sharp。

**关键消融 Figure 5**: 给Transformer加上long-short term attention（split attention成近端/远端两套）, ICL能力 **没有提升**。这说明MMv2的优势 **不仅来自long/short split**, 还来自symmetric key/query、adaptive bandwidth、gated time-variant feature等多个组件的合奏。MMv2是一个 **新的架构**, 不是transformer的patch。

---

## 7. Risk-return trade-off: Transformer需要多少data才能赶上MMv2?

Section 6是paper最striking的部分。问题: 如果坚持用Transformer recipe, 需要多少training tokens才能匹配MMv2在new-knowledge/ICL上的表现?

**实验设计**: 训练3个Transformer large: 200B, 1T, 8T tokens; 训练1个MMv2 large: 1T tokens。

### 7.1 New-knowledge storage (Table 5)

| Model | Train tokens | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|---|
| Transformer large | 200B | 48.6 | 42.9 | 40.7 | 33.8 | X |
| Transformer large | 1T | 51.2 | 48.8 | 44.7 | 41.1 | X |
| Transformer large* (GQA + 8k train) | 8T | 59.2 | 54.5 | 50.9 | 46.9 | X |
| MMv2 large | 1T | 58.9 | 55.5 | 54.9 | 53.4 | 46.4 |

- Transformer 1T vs MMv2 1T: @32k差12.3%
- Transformer 8T vs MMv2 1T: @32k仍差6.5%

**8倍data也没追上**, 而且8T transformer还要配GQA + 8k训练context才达成这个成绩。

### 7.2 In-context learning (Figure 6 + 7)

**Semantic label (Figure 6)**: Transformer 8T **接近** MMv2 1T。So在ICL semantic上, 8x data基本能弥补架构差距。

**Anonymous label (Figure 7)**: Transformer 8T **仍显著落后** MMv2 1T, 而且还出现 **counterintuitive degradation**: 训练data更多, ICL anonymous能力反而变差。

**Intuition**: semantic ICL靠训练时学到的label semantics, 8x data能强化这部分; anonymous ICL完全靠推理时的pattern matching, 这是Transformer架构层面的短板, scaling data解决不了。这印证了作者的核心thesis: **架构很重要, 不是所有问题都能用more data补**。

---

## 8. Fine-tuning speed: 一个mini-batch就够

Section 7展示了fine-tune context length从4k到32k的速度:
- **MMv2**: 1个mini-batch提升22%, 2个mini-batch达到optimal
- **Transformer**: 800个mini-batch还赶不上MMv2的1个mini-batch

**Intuition**: MMv2的architecture本身就是为long context设计的, adaptive bandwidth让模型自动adjust到新context length, long-term memory自然处理远端信号, 只需要gradient告诉模型"现在context length是32k, 不是4k", 模型就能调对。Transformer的RoPE对context length敏感, 必须重新scale位置embedding并经过大量iter让所有attention layer适应新range。

这个发现对实际deployment意义巨大: 如果你想把一个模型quickly adapt到新domain/新context, MMv2需要的fine-tune data比Transformer少 **800倍**。

---

## 9. Discussion: 这工作在research landscape里的位置

### 9.1 与 Léon Bottou 的工作脉络

Léon Bottou长期以来关注"intelligence vs designer intelligence"的区分。Vapnik的risk minimization (https://arxiv.org/abs/1907.02893 reference 3) 也强调过类似区分。MMv2的implicit claim: Transformer里太多东西需要human designer (position encoding, asymmetric q/k, fixed temperature), 这些designer的知识让模型在i.i.d. benchmark上work, 但限制了o.o.d./ICL能力。MMv2反其道而行, 用最少的inductive bias (symmetric kernel + L2 norm + gated temporal), 让模型自己学出该学的东西。

### 9.2 与 Mamba/RWKV/xLSTM 的对比

这些model都追求 **sub-quadratic complexity in sequence length**, 但代价是 **compressed state**。MMv2的取舍相反: 保留full quadratic attention-like complexity, 但分成short/long, 且long-term memory用associative buffer而不是RNN-state。代价是计算更贵, 但new-knowledge storage能力保留。

Table 7/8 实证: 在long-context retrieval + ICL任务上, 压缩式方法直接挂掉。MMv2在这类任务上有天然优势。

参考:
- Mamba: https://arxiv.org/abs/2312.00752
- RWKV: https://arxiv.org/abs/2305.13048
- xLSTM: https://arxiv.org/abs/2405.04517
- Hyena: https://arxiv.org/abs/2302.10866

### 9.3 与 Sparse attention / Long-context transformer 的对比

Longformer (https://arxiv.org/abs/2004.05150), BigBird, NSA (https://arxiv.org/abs/2502.11089), MoBA (https://arxiv.org/abs/2502.13189) 等sparse attention方法都是把full attention变sparse以减cost。MMv2的short/long split在外形上类似sliding-window + global, 但有两个关键不同:
- short/long是 **两套独立参数** + **independent head**, 不是single attention的mask
- 用 **adaptive bandwidth** + **symmetric key/query** 让long-term在远端依然sharp

### 9.4 与 Disentanglement / Causal / OOD literature

Bottou & Zhang之前的work里有大量讨论: 
- IRM (Arjovsky et al. 2019, https://arxiv.org/abs/1907.02893)
- Disentanglement via Hausdorff factorized support (Roth et al. 2022, https://arxiv.org/abs/2210.07347)
- Meta-transfer objective (Bengio et al. 2019, https://arxiv.org/abs/1901.10912)
- MAML (Finn et al. 2017, https://arxiv.org/abs/1703.03400)

这些都试图从loss/regularization层面解决disentanglement。MMv2的视角: disentanglement可以通过 **architectural structure** 自然涌现, 而不需要在loss上加什么trick。Memory Mosaics的short/long/persistent三层分工就是显式disentangle的position-dependent/position-invariant/persistent信号。

### 9.5 与 induction head / ICL literature

- Olsson et al. 2022 "In-context learning and induction heads" (https://arxiv.org/abs/2209.11895): Transformer的ICL能力来自induction head, 2层attention + position encoding才能emerge。
- Bietti et al. 2023 "Birth of a Transformer" (https://arxiv.org/abs/2306.00802): 从memory视角分析transformer, 指出位置encoding + asymmetric q/k是induction head的必要条件。
- MMv2证明: 这些"必要条件"只是Transformer的implementation choice, 不是ICL的fundamental necessity。Symmetric kernel + 无position encoding + key-as-query的设计同样能实现induction head, 而且更efficient + 更可scale。

### 9.6 与 Long-context LLM的 "lost in the middle" 问题

Liu et al. 2023 (https://arxiv.org/abs/2307.03172) 发现Transformer在长context中间的信息容易被忽略。MMv2的long-term memory理论上缓解这个问题, 因为它是position-invariant retrieval, 不依赖于位置。Figure 1的flat曲线证实: long-term的attention score对position不敏感, 只对 **key similarity** 敏感。这是个 **结构性优势**, 不是prompt engineering能弥补的。

### 9.7 与 Nadaraya-Watson estimator 的统计学联系

Memory Mosaics本质上是deep version的Nadaraya-Watson kernel regression (https://en.wikipedia.org/wiki/Kernel_regression), 有丰富的统计学理论:
- Bias-variance trade-off
- Optimal bandwidth selection
- Universal approximation (kernel越大越能approximate任意分布)

v2的adaptive bandwidth直接来自kernel statistics的MISE分析。这种 **architectural choice backed by statistical theory** 让MMv2的设计不像ML的typical "search hyperparameters", 而更像 "apply known results"。

### 9.8 与 Karpathy关心的几个话题

Karpathy多次提到current LLM的limitation包括:
- ICL在long context下失效
- Position encoding对length generalization不友好
- Fine-tune需要的data太多
- Architecture层面的inductive bias未被充分利用

MMv2在所有这几点上都给出了不同的答案: long-term memory用associative buffer避开compressed state问题, 无position encoding让length generalization天然work, fine-tune只要2个mini-batch, adaptive bandwidth/gated temporal是architectural prior而不是learn出来的trick。

---

## 10. Limitations & open questions

### 10.1 已被paper讨论的
- **Computational cost**: 比 transformer多12% FLOPs, 训练后可剪long-term memory减到更便宜, 但训练时仍然贵。
- **Very long context (>64k)**: paper未测, 作者在Section 8讨论用fuzzy hashing (https://arxiv.org/abs/2410.16179, MagicPIG by Chen/Zhang/Bottou等, 同一个lab) 和hierarchical memory (NSA/MoBA) 进一步scale。

### 10.2 未讨论的（潜在limitation）
- **Instruct tuning / RLHF**: paper只测base model, 没有ChatBot-style evaluation。MMv2的associative memory在instruction following场景下表现未知。
- **Reasoning / chain-of-thought**: gsm8k、math benchmark上MMv2与transformer持平, 没看到deep reasoning能力的对比。Adaptive bandwidth对multi-step reasoning可能不利（reasoning需要loose retrieval, 但sharp bandwidth倾向precise retrieval）。
- **Mixing short/long in single head**: short和long是独立head, 但实际数据中local/global信号可能需要joint modeling, split可能丢失interaction。
- **Long-term memory的computational complexity**: 仍是 $O(n)$ 存储 + $O(n)$ retrieval per token, 总复杂度 $O(n^2)$, 跟full attention一样。Fuzzy hashing可能能解但没在这篇paper里。
- **Long-term memory的retention pattern**: paper未测long-term attention在very long context (100k+)下的degradation, 不清楚是否也会退化。
- **Pretraining efficiency**: 1T tokens能训出8B MMv2, 但loss curve / scaling law未展示, 不清楚MMv2在 Chinchilla-optimal data下loss vs compute的关系。
- **Sample efficiency during pretraining**: 剪枝能力（训练后剪long-term不掉点）说明long-term在persistent knowledge任务上是冗余的, 这是否意味着training efficiency可以更高?

---

## 11. 总结：核心takeaways

1. **Associative memory = attention with stronger inductive bias**（L2 norm + symmetric + 无PE + adaptive β）
2. **三层memory分工是显式disentanglement**: short-term做local syntax, long-term做global retrieval, persistent做training knowledge
3. **Adaptive bandwidth来自kernel regression statistics**: 不是hyperparameter search, 是理论指导的architectural choice
4. **Gated time-variant extractor让key随语义自适应**: 比固定leaky average更鲁棒
5. **8T transformer ≠ 1T MMv2**: scaling data解决不了架构短板, 至少在anonymous ICL上
6. **Fine-tune 2 mini-batch vs 800**: architectural prior的复用价值巨大
7. **Memory compression方法结构性失败**: 不能"lazy retrieval"的模型在新-knowledge任务上无解

这篇paper给Karpathy-style的 "build intuition first" 完美契合: 它从kernel regression这个简单统计学起点出发, 通过architectural modifications一层层build出能match 8T transformer的1T model。比起"堆data堆compute"的scaling story, 这是个 "structure matters" story。

---

## Web references

主paper:
- Memory Mosaics at scale (本paper, 应该是后续在arXiv上发布)
- Memory Mosaics v1 (ICLR 2025): https://arxiv.org/abs/2505.20350

关联work:
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Birth of a Transformer: A Memory Viewpoint: https://arxiv.org/abs/2306.00802
- In-context Learning and Induction Heads: https://arxiv.org/abs/2209.11895
- Mamba: https://arxiv.org/abs/2312.00752
- RWKV: https://arxiv.org/abs/2305.13048
- xLSTM: https://arxiv.org/abs/2405.04517
- RULER benchmark: https://arxiv.org/abs/2404.06654
- Long-context LLMs struggle with ICL: https://arxiv.org/abs/2404.02060
- Lost in the Middle: https://arxiv.org/abs/2307.03172
- Llama 3: https://arxiv.org/abs/2407.21783
- SwiGLU: https://arxiv.org/abs/2002.05202
- ALiBi: https://arxiv.org/abs/2108.12409
- Longformer: https://arxiv.org/abs/2004.05150
- Banking77: https://arxiv.org/abs/2003.04807
- GoEmotions: https://arxiv.org/abs/2005.00547
- TACRED: https://aclanthology.org/D17-1004/
- MagicPIG (fuzzy hashing for LLM): https://arxiv.org/abs/2410.16179
- Native Sparse Attention (NSA): https://arxiv.org/abs/2502.11089
- MoBA: https://arxiv.org/abs/2502.13189
- IRM: https://arxiv.org/abs/1907.02893
- MAML: https://arxiv.org/abs/1703.03400
- Augmenting self-attention with persistent memory: https://arxiv.org/abs/1907.01470
- Karpathy's nanoGPT (作为baseline对照): https://github.com/karpathy/nanoGPT
- Kernel regression wiki: https://en.wikipedia.org/wiki/Kernel_regression
- Elements of Statistical Learning (Hastie/Tibshirani/Friedman): https://hastie.su.domains/ElemStatLearn/
