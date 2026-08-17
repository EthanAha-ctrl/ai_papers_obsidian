---
source_pdf: SpikingManifesto.pdf
paper_sha256: 1219568967bfa346f7b27ccb1b489dc04c5b7a829f2e45121e494b29344474cc
processed_at: '2026-08-12T10:04:33-07:00'
target_folder: LLM-from-scratch
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Spiking Manifesto 的人话版

好，我把那些公式和术语都扔掉，用最接地气的方式重新讲一遍 Izhikevich 这篇 manifesto 到底在说什么。

---

## 1. 一句话总结

**你的大脑用 20 瓦跑出了 Llama 3 要几十瓦才能干的事，原因是大脑用"spike 发放的先后顺序"来编码信息，而 LLM 用"连续数值的矩阵乘法"。Spike 顺序给你 $n!$ 种组合，矩阵只给你 $e^n$ 种 —— 这就是 1000 倍效率差距的来源。**

---

## 2. 为什么 brain 这么省电

想象你在一个有 10000 人的公司里。

**LLM 的做法**：每个员工每小时都要跟所有其他员工打个电话确认一下"你现在状态怎么样"，然后把自己状态汇报给老板。即使你今天没事干，也要接电话。这就是 matrix multiplication —— 每层每个神经元都要参与计算，不管有没有有用信息。

**Brain 的做法**：员工平时睡觉，只有真有事的时候才站起来喊一嗓子。其他员工听到这嗓子，决定自己要不要也站起来喊。这就是 spiking —— sparse，event-driven，大部分时间大部分神经元 silent。

但这只是省电的一半原因。另一半更精彩，是**信息编码方式**。

---

## 3. 信息编码：为什么 spike 顺序是"免费午餐"

### 3.1 LLM 的编码

LLM 里一个 neuron 的 activation 是个连续实数，比如 0.7。一个 60 维的 embedding 就是 60 个实数组成的一个向量 $(0.7, 0.3, ..., 0.5) \in \mathbb{R}^{60}$。

要表示"不同的概念"，你需要不同的向量。但 $\mathbb{R}^{60}$ 里能塞多少个**互相近正交**的向量？Johnson-Lindenstrauss lemma 告诉你大约 $e^{c \cdot 60}$ 个，其中 $c$ 是个小常数。这就是 ANN 的 linear capacity。

### 3.2 Brain 的编码

现在想象 60 个神经元，每个在一个 100ms 时间窗口里**各发一次 spike**。你问：这 60 个 spike 的"向量"是什么？

从 LLM 视角：每个神经元发了 1 次，所以 activation 都是 1，向量是 $(1,1,...,1)$ —— **就一个 embedding**。

从 Brain 视角：60 个 spike 的**发放顺序**有 $60!$ 种可能。$60! \approx 10^{80}$，比可观测宇宙的粒子数还多。

**这就是 Izhikevich 的核心 insight**：同样的"60 个神经元各发一次 spike"这件事，LLM 看来是一个 state，brain 看来是 $10^{80}$ 个 state。

用 embedding 的语言说：**brain 的一个 60 维"embedding"实际上编码了 $10^{80}$ 个 distinguishable 的概念**。

这个 capacity 是 factorial 级的， ANN 的 exponential capacity 在它面前就是小儿科。

---

## 4. 但怎么用这个 capacity？——LUT 登场

你有 $10^{80}$ 种 spike 顺序，但你不能对每种顺序都建一个 entry 存起来（宇宙装不下）。你需要一个聪明的办法把"spike 顺序"映射到一个可以管理的查表操作。

### 4.1 Anchor pair 的 trick

Izhikevich 的办法很巧妙：

1. 随机选 $n_c$ 对 neuron，比如 $(x_1, x_2), (x_3, x_5), (x_2, x_3)$
2. 对每对，只问一个问题：**谁先发 spike？**
3. 每对给 1 bit（0 或 1），$n_c$ 对给你 $n_c$ bits
4. 拼起来就是一个 integer index，范围 $0$ 到 $2^{n_c} - 1$
5. 用这个 index 去查一张预先存好的表（look-up table, LUT），取出一个 synaptic vector 加到下一层

就这么简单。**没有乘法，只有比较和加法**。

### 4.2 为什么这很厉害

- 每个 LUT 有 $2^{n_c}$ 行，能区分 $2^{n_c}$ 种 anchor spike 顺序
- 整个网络有 $n_t$ 个 LUT，能区分 $2^{n_t \cdot n_c}$ 种 spiking pattern
- 把 $n_c$ 从 10 加到 11 → 容量翻倍，但每次 forward 只多一次比较

**容量指数增长，计算几乎不变**。这在 ANN 里是不可能的 —— ANN 里你要让容量翻倍，就得加宽 layer，matmul 成本立刻翻倍。

### 4.3 跟 LSH 的关系

这个"比较 sign 然后拼 bit"的操作其实就是一种 locality-sensitive hashing (LSH)。LSH 的特性是：相近的 input 落到同一个 bucket，远的 input 落到不同 bucket。

对应到 SNN：spike 顺序**接近**的 pattern 落到同一个 LUT row，spike 顺序**差很多**的 pattern 落到不同 row。这天然带来 **robustness**（小噪声不改变 spike 顺序，index 不变，输出不变）和 **generalization**（类似的输入得到类似输出）。

> LSH 原始论文：Indyk & Motwani 1998 https://dl.acm.org/doi/10.1145/278298.278316

---

## 5. 但 LUT 不可微，怎么训练？

这是 SNN 一直以来的老大难问题。spike 是离散事件，gradient 不知道怎么传。

### 5.1 问题本质

Latency vector $\mathbf{x}$ 微小变化时，通常不影响 anchor pair 的 sign（$x_1 > x_2$ 还是 true），所以 LUT 查到的 row 不变，输出不变。$\partial \mathbf{y} / \partial \mathbf{x} = 0$ 几乎处处成立。

但当某个 anchor pair 的 latency diff 接近 0 时（$x_1 \approx x_2$），微小扰动会让 sign 翻转，row 突然跳到另一个，输出 discontinuous。

### 5.2 Surrogate gradient 的直觉

Izhikevich 的办法：

1. 对每个 LUT，找到那个**最接近翻转边缘**的 anchor pair（$|u_i| = |x_{a_i} - x_{b_i}|$ 最小的那个）
2. 引入一个 "uncertainty function" $U(u)$，当 $|u|$ 大时 $U \approx 0$，当 $u \approx 0$ 时 $U = 0.5$
3. 用 surrogate：$y = S_{ix} + U(u_i)(S_{i\bar{x}} - S_{ix})$，意思是"当不确定该用哪行时，取两行平均"

**直觉**：你站在一个 bucket 边界上，左边 bucket 给你 $S_j$，右边给你 $S_{\bar{j}}$。离边界远就坚定用一边，离边界近就摇摆用平均。

### 5.3 Gradient 的物理意义

对那个最接近翻转的 anchor pair $(a_i, b_i)$：

$$
\frac{\partial \mathcal{L}}{\partial x_{a_i}} = -\frac{\partial \mathcal{L}}{\partial x_{b_i}} = U'(u_i) \cdot g_i
$$

其中 $g_i = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot (S_{i\bar{x}} - S_{ix})$ 衡量"翻转一下 vs 不翻转"哪个更对齐 loss gradient。

**两个 gradient 符号相反**，意思是：要么把 $x_{a_i}$ 和 $x_{b_i}$ 推近（促使翻转），要么推远（避免翻转）。这其实就是一种**竞争性学习** —— LUT 的两行在争夺"哪个该被选中"。

### 5.4 一个漂亮的副作用

每个 LUT 给 gradient 贡献一对相反数 → 整个 latency vector 的 gradient 均值为 0 → **latency 自动保持 zero mean**。不需要额外的 normalization，self-regularization built-in。

---

## 6. 三种架构怎么换

Izhikevich 的 claim 是：任何用 matmul 的地方都可以用 LUT 替换。他演示了三种：

### 6.1 Deep SNN（替换 MLP）

原版：$\mathbf{x}^{l+1} = \mathbf{x}^l + \text{ReLU}(M^l \mathbf{x}^l + \mathbf{b}^l)$

SNN 版：$\mathbf{x}^{l+1} = \mathbf{x}^l + \mathbf{S}_{\mathbf{x}^l}^l$

就是把 matmul + ReLU 换成 LUT 查表加法。residual connection 保留。

### 6.2 Spiking RNN（替换 Elman / LSTM）

原版 Elman：$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{z}_t)$，通常是 matmul。

SNN 版：$\mathbf{h}_t = \mathbf{S}_{\mathbf{h}_{t-1}} + \mathbf{z}_t$

最神奇的是 **spiking RNN 不怕 vanishing/exploding gradient**：
- 每个 time step 用不同的 LUT row，不反复乘同一矩阵 → 不会 explode
- Pairwise comparison 对绝对 magnitude 不敏感（只看 ordering）→ 不会 vanish

也就是说，LSTM 那些复杂 gate 机制在这里天然不需要。

### 6.3 SNN Transformer（替换 attention）

这是最难也最有意思的部分。

原版 attention：$\text{softmax}(QK^\top / \sqrt{d_k})V$，有 matmul bottleneck。

Izhikevich 的方案：对每个 query-key pair $(i, j)$，把 $\mathbf{z}_i$、$\mathbf{z}_j$、相对位置编码 $\mathbf{PE}_{i-j}$ **concatenate** 成一个长向量 $[\mathbf{z}_i, \mathbf{z}_j, \mathbf{PE}_{i-j}] \in \mathbb{R}^{2n+p}$，然后查 LUT：

$$
\mathbf{x}_i = \mathbf{z}_i + \sum_{j < i} \mathbf{V}_{[\mathbf{z}_i, \mathbf{z}_j, \mathbf{PE}_{i-j}]}
$$

**No softmax, no QK^T, no V matmul**。attention score 是 LUT 内部 synaptic vector 的 magnitude 隐式学到的。

V-index cache（类似 KV-cache）：预计算每个 position 的 $\mathbf{z}_{pos}$ 和 $\mathbf{PE}_{pos}$ 的 hash index，查询时直接拼接三个 cached index，复杂度从 $O(n_{inp}^2)$ 降到 $O(n_{inp})$。

---

## 7. 实验数据告诉你它真的 work

任务：byte-level character prediction（Shakespeare-like 文本），32 字符 context，无 dropout、无 L2、无 Adam tuning、weights 全 0 初始化。

### 7.1 Spiking RNN（5M 参数）

| 模型 | 年份 | 大小 | BPC |
|---|---|---|---|
| MI-LSTM | 2016 | 17M | 1.44 |
| mLSTM | 2016 | 10M | 1.40 |
| **Spiking RNN** | **2025** | **5M** | **1.39** |
| BN LSTM | 2016 | 16M | 1.36 |
| HM-LSTM | 2016 | 35M | 1.32 |

5M 的 spiking RNN，没有任何 LSTM gate、没有正则化、没有优化器技巧，BPC 1.39，跟 2016 年 SOTA LSTM 打平。LSTM 有二十年的工程积累，SNN 这是从零开始。

### 7.2 SNN Transformer（806M）

跟原版 Attention Is All You Need 同样的 6 层 8 头设定对比：

- **计算量**：ANN 235M ops vs SNN 172K ops → **1360x 减少**
- **Memory bandwidth per token**：ANN ~1M vs SNN ~1K → **10000x 减少**
- **Memory footprint**：ANN 3.1M vs SNN 10.5M → SNN 反而更大（这是 trade-off：用大 footprint 换小 bandwidth）
- **学习收敛速度**：SNN 比 ANN 快 **50 倍**（Fig 13，这个数字让人怀疑是不是 learning rate 没调好，但 paper 说 Adam 是为 ANN 调的，对 SNN 并不公平）

Ablation 更狠：把 FFN 全去掉，只用 attention（$n=16$，单头），仍然能 work。因为 LUT 本身就是非线性的，attention 已经内含 token mixing 和 channel mixing 两重功能。**"SNN attention is all you need"** —— 这句话显然是在致敬 Vaswani。

> Attention Is All You Need: https://arxiv.org/abs/1706.03762

---

## 8. 几个重要的直觉要点

### 8.1 Footprint vs Bandwidth 的分离

这是 SNN 的核心 asymmetry，也是它 scaling 的秘诀。

- **Memory footprint**：所有 LUT 存下来的总数据量，可以巨大（$n_t \cdot 2^{n_c} \cdot n$）
- **Memory bandwidth**：每次 forward 实际要读的数据量，极小（$n_t \cdot n$，每个 LUT 只读一行）

ANN 里这两个量基本相等（每次 forward 都要读所有 weight matrix）。SNN 把它们解耦了：你可以指数级扩容 footprint 而几乎不增加 bandwidth。

**Storage 便宜，access 贵**。这是所有现代计算系统的铁律，SNN 天然符合这条铁律。

### 8.2 LUT = Spiking 的本质

Paper 最 provocative 的 claim：**spiking networks 就是 nature 的 LUT 实现**。

- LUT 存大量参数但每次只用一小部分 ←→ SNN 大量 synapse 但大多数 silent
- LUT 通过 index 选 entry ←→ SNN 通过"哪个神经元发了 spike"选 outgoing synapse
- LUT 其他 entry 直到被查询才激活 ←→ SNN 其他 synapse 直到该神经元发放才激活

一旦你接受这个抽象，就会发现不需要模拟 individual neuron dynamics，直接做"pattern → pattern"的 mapping 就行了。这是跟传统 neuromorphic (TrueNorth, Loihi, SpiNNaker) 的根本区别 —— 那些芯片还在 neuron level 模拟，而 Izhikevich 主张直接在 pattern level 操作。

> TrueNorth: https://ieeexplore.ieee.org/document/7056730
> Loihi 2: https://www.intel.com/content/www/us/en/research/neuromorphic-computing-loihi-2-technology-brief.htm

### 8.3 0 初始化 + zero-mean gradient 的自稳定性

所有 synapse 初始化为 0，但训练能 work，因为：
- Latency vector gradient 每对 anchor pair 贡献一正一负 → 均值 0
- Latency 自动保持 zero mean，不需要 LayerNorm / BatchNorm
- 这是 SNN 的 built-in self-regularization

ANN 里你敢全 0 初始化 matmul 网络吗？不敢，会死。SNN 敢，因为 forward pass 用的是 ordering 不是 magnitude，0 初始化只影响初始 ordering 的"随机性"，不影响机制本身。

---

## 9. 跟其他"spike + LLM"工作的区别

最近有一堆 SpikeLLM、SpikeGPT、SpikeBERT、BrainTransformers、MatMul-free LM 之类的工作。Izhikevich 明确说：**这些都只是把 ANN 量化成 (-1, 0, +1) 三值，用 sparsity 加速，根本没引入 LUT，也没用 spike ordering 的 combinatorial capacity**。

> SpikeGPT: https://arxiv.org/abs/2302.13939
> SpikeLLM: https://arxiv.org/abs/2407.04752
> MatMul-free LM: https://arxiv.org/abs/2406.02528

Izhikevich 走的路完全不同：他**从架构层面替换 matmul 为 LUT**，用 spike ordering 作为信息载体。这是 paradigm 层面的差异，不是 quantization 层面的。

---

## 10. 几个值得质疑的点

为了 build 你的 intuition，我必须诚实指出几个让人不放心的地方：

### 10.1 Gradient 太稀疏

每层只对 $2 n_t$ 个 neuron 传 gradient（每个 LUT 贡献一对 anchor）。$n=64$、$n_t=64$ 勉强够（256 个 gradient 信号），但 $n=512$ 时呢？大模型怎么办？paper 没在 $n=512$ 上做大规模实验。

最激进的 Section VIII-K 甚至每层只传**一个 scalar**，paper 自己说这是 "worst learning performance"。

### 10.2 Attention LUT 爆炸

Attention 用 concatenated vector $[\mathbf{z}_i, \mathbf{z}_j, \mathbf{PE}_{i-j}] \in \mathbb{R}^{2n+p}$ 做 hash。$n_c=6, p=4$ → 每个 LUT $2^{16} = 65536$ 行。$n_c=10$ → $2^{24} \approx 16M$ 行。再大就存不下了。

Paper 说增大 $n_c$ 几乎不增加计算，这是对的。但**footprint 指数爆炸**这个问题对 attention 尤其严重，因为 attention 本来就是 $n_c$ 需要够大才能区分大量 token pair pattern。

### 10.3 容量 vs 实际可学的 patterns

理论上 $2^{n_t n_c}$ 个 patterns。但训练数据只有那么多，能学出来的 pattern 远少于此。大部分 LUT row 可能从来没被激活过。这跟 ANN 里的 dead neuron 问题类似，但更严重 —— 因为 ANN 里每个神经元至少每层都被算一次，SNN 里 row 不被 index 命中就完全 dead。

Section VIII-I 提到 structural plasticity 可以替换信息量低的 anchor pair，但具体 learning rule 没给出。

### 10.4 跟 Modern Hopfield 的关系没提

ANN attention 本质是 Modern Hopfield Network 的 retrieval dynamics。SNN attention 是 LSH-based associative memory。两者都是 associative memory，只是 addressing 机制不同（dot product vs spike ordering）。这个连接如果展开，能让 SNN 的 attention 有更深的理论 grounding。

> Modern Hopfield Networks: https://www.pnas.org/doi/10.1073/pnas.2001926117

### 10.5 对比公平性

5M spiking RNN vs 2016 LSTM 是 apples-to-apples 吗？2016 的 LSTM 没用 Adam、没 dropout、没 layernorm，spiking RNN 也没有，确实公平。但 2016 至今 9 年，LSTM 早就有更好的训练 trick。spiking RNN 是"裸跑"跟"裸跑"比，这个对比是诚实的，但只能说明 SNN 的"起点高"，不能说明 SNN 的"上限高"。

---

## 11. 我觉得最 elegant 的几个点

讲真，撇开质疑，这篇 manifesto 有几个地方真的漂亮：

### 11.1 Word embedding analogy

Anchor $x_1 > x_2$ 对应 "king/boy/son"（阳性），$x_2 > x_1$ 对应 "queen/girl/daughter"（阴性）。ANN 用向量加法表示新 feature（king - man + woman = queen），SNN 用 spike 顺序 permutation 表示新 feature（把某对 anchor 顺序翻一下就得到新概念）。这个类比让 SNN 的"combinatorial capacity"突然有了具体画面。

### 11.2 Edge detector toy example

3×3 patch，9 个 binary pixel，直接当 index 查 512 行的 LUT。水平 edge = 000000111 = decimal 7，所以"neuron 7"发 spike，只有第 7 行 outgoing synapse 被激活。这实现了 V1 orientation-selective cell 的功能，**完全不需要模拟 lateral inhibition**。

这个例子让你立刻明白 "spiking = LUT retrieval" 是什么意思。而且 Izhikevich 自己都惊讶这个解释从没被提出过。

### 11.3 Spiking backprop 不需要 matmul

Section VIII-K 最激进版本：每层只传一个 scalar $h^l$，从 LUT 直接传到 LUT，不经过 neuron。这暗示**同一套 spiking hardware 可以同时做 inference 和 training**。对 neuromorphic chip 设计和生物 plausibility 都有深刻含义。

生物界一直怀疑 brain 怎么做 backprop（feedback 通路怎么传 error gradient）。Izhikevich 给出一个 spiking-native 的可能答案：error 通过 LUT 之间的 scalar signal 传递，不需要 matmul。

> Backpropagation and the brain (Lillicrap et al. 2020): https://www.nature.com/articles/s41583-020-0277-3

### 11.4 Spiking-native PEFT

两个方案很聪明：
- **$n_t + 1$**：加一个全 0 的 LUT，只训这个 → 类似 LoRA 但更 native
- **$n_c + 1$**：某 LUT 加一对 anchor，row 数翻倍，新行复制旧行 → 初始输出不变但能 split 旧 pattern

比 LoRA 更"native"：LoRA 是低秩补丁，SNN PEFT 是"加查表条目"，概念上更贴合 LUT 架构。

> LoRA: https://arxiv.org/abs/2106.09685

---

## 12. 给你的 bottom line

### 一句话

**Izhikevich 说：spike 的发放顺序是 nature 用来做 look-up table 的 trick，它给你 factorial capacity、sparse activation、和可解耦的 footprint vs bandwidth，所有 matmul 都可以换成 LUT 查表 + 加法。**

### 三件事带走

1. **Capacity**：$n! \gg e^n$。Spike ordering 给你的表示能力远超连续向量，这是 SNN 一切优势的源头。
2. **LUT 抽象**：不要模拟 individual neuron，直接做 pattern-to-pattern mapping via LUT。Footprint 可以巨大，bandwidth 极小，storage 便宜 access 贵的现实被充分利用。
3. **Surrogate gradient**：找最接近翻转边缘的 anchor pair，用"翻转 vs 不翻转"的 alignment 决定 gradient 方向。一对 anchor 一对正负 gradient，自带 zero-mean self-regularization。

### 真正值得做的工程方向

1. **新 hardware**：专门做 LUT-fetch + addition 的 chip，既不是 GPU 也不是传统 neuromorphic
2. **Hybrid**：把 hyperplane LSH 版 SNN 插入现有 16384 维 LLM 当 attention 替代品
3. **Structural plasticity**：让 anchor pair 可学，自动淘汰信息量低的 pair
4. **Spiking-native PEFT**：$n_t+1$ 和 $n_c+1$ 作为 LLM fine-tuning 的新选择

### 我的判断

这篇 manifesto 写得短、狠、provocative，典型 Izhikevich 风格。它的真正价值可能不在于立刻 beat Llama，而在于**重新定义了 spiking 在 ML 里的含义**：从"bio-inspired activation function"变成"latency-ordering-based LUT addressing"。

这个 mental shift 一旦普及，后面 5-10 年的工程实现只是时间问题。就像 2012 年 AlexNet 让大家意识到 GPU + CNN 是可行路线，这篇 manifesto 让人意识到 LUT + spike ordering 是另一条可行路线。

不一定对，但**值得严肃对待**。

> Code: https://github.com/izhikevich/SNN
> Polychronization 原始论文: https://direct.mit.edu/neco/article-abstract/18/2/245/682/
> Izhikevich 书: https://mitpress.mit.edu/9780262090436/dynamical-systems-in-neuroscience/

---

# Spiking Manifesto 深度解读

这是 Eugene Izhikevich（是的，就是 Izhikevich neuron model 那个 Izhikevich）写的一篇 manifesto，核心主张非常 radical：**spiking networks 是 nature 实现 look-up tables 的方式**，所有 ANN 里的 matrix multiplication 都可以被 LUT 替换，而且 SNN 拥有 factorial 级别的 encoding capacity。下面我把这篇 paper 揉碎了讲给你，并尽量 build 你的 intuition。

---

## 1. 核心动机：为什么 brain 比 Llama 高效 1000x

开篇的对比很 punchy：

- **Llama 3**：10^11 synaptic parameters，inference 几十瓦
- **Human brain**：10^15 synapses，仅 20W
- 等比例缩小 10000 倍到 Llama 3 的规模，brain 只需 2mW —— 一节 AAA 电池能跑一个月

两个根本原因：

**(1) Efficient Processing — Sparse spiking**
ANN 每个 time step 都把 continuous-valued activation 通过 synaptic weights 乘起来做 matmul；SNN 只有当 membrane potential 超过 threshold 才发一个 all-or-none 的 spike，其余时间神经元 silent，不传输任何信号。

**(2) Efficient Encoding — Factorially explosive capacity**
这是这篇 paper 最关键的 insight。看 Fig 1b：60 个神经元，每个在一个时间窗口里只发 1 个 spike。
- 从 ANN 视角：所有 activation 都等于 1，是单一 feature vector $(1,1,...,1) \in \mathbb{R}^{60}$
- 从 SNN 视角：这是 60! 种可能 spike ordering 之一，而 $60! > 10^{80}$ —— 比可观测宇宙的粒子数还多

**所以 ANN 看来的"一个 embedding"，在 SNN 里其实有无穷多个 distinguishable 的 sub-patterns。**

Fig 3 给出容量公式：
- ANN 的 linear capacity for ε-orthogonal vectors: $e^{\varepsilon^2 n}$（来自 Johnson-Lindenstrauss lemma [2]）
- SNN 的 combinatorial capacity: $n!$

这两个量在 n 稍大时就完全不可比。

> 参考：Johnson-Lindenstrauss lemma https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma

---

## 2. 两大 Pillars

### Pillar 1: 抽象掉 individual neurons，直接做 spiking-pattern → spiking-pattern 的 mapping

传统 SNN 仿真要建模每个神经元的 membrane potential 和神经元间直接交互（monumental task，适合 TrueNorth / Loihi / SpiNNaker 这种 neuromorphic hardware）。这篇 paper 反其道而行之：把网络看成 latency vector 到 latency vector 的直接映射，通过 LUT 检测 spiking pattern 并查表得到下一层的 latency。

### Pillar 2: Spiking networks 是 nature 的 look-up tables

这个观察非常 sharp：
- LUT 存储大量参数但每次只用一小部分 —— SNN 也一样，大量 synapse 但大多数神经元 silent
- LUT 通过 index 选取 entry —— SNN 通过"哪个神经元发了 spike"来确定只激活该神经元 outgoing 的 synapses
- 所有其他 synapse 都"untouched until needed"

历史上从 McCulloch-Pitts（1943）的 logic gates view，演进到 modern linear algebra view（states as vectors, transitions as matmul, learning as weight updates）。Izhikevich 主张下一步：**spiking patterns as fundamental states, transitions via LUTs, learning as adjustment of LUTs**。

---

## 3. 数学形式化：The Model

### 3.1 Latency vector

假设所有神经元在某时间窗口里各发 1 个 spike。第 i 个神经元的 spike 时间相对窗口中心记为 $x_i \in \mathbb{R}$（可正可负）。整个 spiking pattern 用 latency vector $\mathbf{x} = (x_1, ..., x_n) \in \mathbb{R}^n$ 表示。

> Fig 5：没有显式的 time variable 在 pattern 内部，pattern 由 latency vector 描述；layer 间通过 $\mathbf{x}^l \to \mathbf{x}^{l+1}$ 转换。

### 3.2 Anchor neurons 与 hash function

为了让 LUT 能区分天文数字级别的 patterns，需要把连续的 latency vector 离散化为整数 index。方法：

- 共 $n_t$ 个 look-up tables
- 每个 table 监测 $n_c$ 对 anchor neurons（初始化时随机选）
- 第 i 个 table 的 anchor pairs 记为 $\mathbf{a}_i = (a_{i1}, ..., a_{i,n_c})$ 和 $\mathbf{b}_i = (b_{i1}, ..., b_{i,n_c})$
- 对每对 anchor 计算 $u_{ir} = x_{a_{ir}} - x_{b_{ir}}$
- 取 sign：$u_{ir} > 0$ 得 1 bit
- 拼接 $n_c$ 个 bits 得到 index：

$$
j = H_i(\mathbf{x}) = \text{concat}\,(u_{i1} > 0, \dots, u_{i,n_c} > 0) \tag{1}
$$

每个 table 因此能区分 $2^{n_c}$ 种 anchor spike ordering；整个网络能区分 $2^{n_t n_c}$ 种 polychronous patterns。

### 3.3 Forward pass

记 $s_{ijk}$ 为第 i 个 table、第 j 行、第 k 列的 synaptic value。由于 $j = H_i(\mathbf{x})$，可简写为 $s_{i\mathbf{x}k}$。则第 k 个 output neuron 的 latency：

$$
y_k = \sum_{i=1}^{n_t} s_{i\,H_i(\mathbf{x})\,k} \quad \text{或简记} \quad \mathbf{y} = \sum_{i=1}^{n_t} \mathbf{S}_{i\mathbf{x}} =: \mathbf{S}_\mathbf{x} \tag{2, 3}
$$

**注意**：这里只有 addition，没有 multiplication！称之为 synaptic "weights" 其实是 misnomer —— spiking networks 里 synapses 总是加到某个东西上。但因为 index 选择依赖 $\mathbf{x}$，整个操作是**非线性的**。

### 3.4 为什么这个设计很 efficient

- **Low processing demand**：pattern → index 只是 bit concatenation，几乎零成本。LUT 大小翻倍只要 $n_c + 1$，计算开销只增加常数。
- **Low memory bandwidth**：虽然 LUT 总 footprint 巨大（$n_t 2^{n_c} n$），但每次 forward 只读 $n_t$ 行。**Storage 便宜，access 贵**——这个 asymmetry 是 SNN 的核心优势。

Fig 2 把 ANN 和 SNN 放进 memory vs. processing 二维空间：ANN 在"高 memory bandwidth + 高 processing"象限，SNN 应该占据"高 memory footprint + 低 active bandwidth + 低 processing"象限。

---

## 4. Surrogate Gradient 与 Backprop

### 4.1 不连续性问题

$\mathbf{S}_{i\mathbf{x}}$ 是 piecewise constant；除非 $x_{a_{ir}} = x_{b_{ir}}$，否则 $\partial \mathbf{y}/\partial x_m = 0$。当某对 anchor 的 latency diff 接近 0 时，微小扰动会 flip spike order，index 翻转，查到的 synaptic vector 整个换掉。这正是所有 SNN 做 backprop 的共同痛点（spike 添加/删除导致 dynamics 不连续）。

### 4.2 Surrogate function

对每个 table，找到**绝对值最小的 anchor diff**：

$$
u_i = x_{a_i} - x_{b_i} \quad \text{with} \quad |u_i| = \min_r |u_{ir}|
$$

这个 $u_i$ 是最容易翻转的（最小扰动就能 flip 它的 sign）。引入对称的 "uncertainty function" $U(u)$，满足 $U(u) \to 0$ 当 $|u|$ 大，$U(0) = 0.5$。Paper 中实际用：

$$
U(u) = \frac{0.5}{1 + |u|}
$$

把式 (3) 替换成 surrogate：

$$
\mathbf{y} = \sum_{i=1}^{n_t} \left\{ \mathbf{S}_{i\mathbf{x}} + U(x_{a_i} - x_{b_i})\,(\mathbf{S}_{i\bar{\mathbf{x}}} - \mathbf{S}_{i\mathbf{x}}) \right\} \tag{4}
$$

其中 $\bar{\mathbf{x}}$ 表示 flip 该最小 anchor pair 顺序后的 latency vector，对应 $\bar{j} = H_i(\bar{\mathbf{x}})$（即 j xor $2^{r_i}$，flip 第 $r_i$ 位）。

直觉：当 $u_i \to 0$ 时，surrogate 平滑插值到 $(\mathbf{S}_{i\mathbf{x}} + \mathbf{S}_{i\bar{\mathbf{x}}})/2$，正好反映"不确定该用哪行"。

### 4.3 Gradient 公式

对 $x_{a_i}$ 和 $x_{b_i}$ 求偏导（其余 $\partial \mathbf{y}/\partial x_m = 0$）：

$$
\frac{\partial \mathbf{y}}{\partial x_{a_i}} = -\frac{\partial \mathbf{y}}{\partial x_{b_i}} = U'(x_{a_i} - x_{b_i})\,(\mathbf{S}_{i\bar{\mathbf{x}}} - \mathbf{S}_{i\mathbf{x}}) \tag{5}
$$

两个 derivative 符号相反，物理意义是"pull together or push apart"这对 anchor 的 latency。

### 4.4 Backprop 递推

多层网络：$\mathbf{x}^{l+1} = \mathbf{S}_{\mathbf{x}^l}^l$，loss $\mathcal{L}$。定义 alignment scalar：

$$
g_i^l = \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{l+1}} \cdot \left(\mathbf{S}_{i\bar{\mathbf{x}}^l}^l - \mathbf{S}_{i\mathbf{x}^l}^l\right) \tag{7}
$$

$g_i$ 衡量"flip 一下 vs 保持现状"哪个更对齐 gradient。则：

$$
\frac{\partial \mathcal{L}}{\partial x_{a_i^l}^l} = -\frac{\partial \mathcal{L}}{\partial x_{b_i^l}^l} = U'(x_{a_i^l}^l - x_{b_i^l}^l)\, g_i^l \tag{8}
$$

**关键性质**：每个 table 给 gradient 贡献一对相反数，导致 $\partial \mathcal{L}/\partial \mathbf{x}^l$ 均值为 0 —— **self-regularization**，让 latency vector 自动保持 zero mean，提供稳定性。

### 4.5 学习规则的几何直觉

- $g_i < 0$：保持当前 $\mathbf{S}_{i\mathbf{x}}$ 更好，应该把 $|u_i|$ 推大（远离 0，避免翻转）
- $g_i > 0$：翻转更好，应该把 $|u_i|$ 推近 0（促使 flip）

这其实是一种 **竞争性学习**：在每个 table 内部，决定两行 LUT 哪个该被选中。

Fig 14 给出非常清楚的几何图示：input space 被 hash function 划分为 buckets，每个 bucket 映射到一个 output point $\mathbf{S}_j$；backward pass 找 nearest neighbor $\bar{\mathbf{x}}$ 使得 hash index flip 到隔壁 bucket，用 $(\mathbf{S}_{\bar{j}} - \mathbf{S}_j)$ 与 error vector $\mathbf{v}$ 的 alignment $g$ 来决定 surrogate gradient 方向。

> 参考：Surrogate gradient learning in SNNs https://arxiv.org/abs/1901.09948

---

## 5. 三种架构实例

### 5.1 Deep SNN（替换 MLP）

标准 MLP + residual：
$$
\mathbf{x}^{l+1} = \mathbf{x}^l + \text{ReLU}(M^l \mathbf{x}^l + \mathbf{b}^l) \tag{9}
$$

直接替换为：
$$
\mathbf{x}^{l+1} = \mathbf{x}^l + \mathbf{S}_{\mathbf{x}^l}^l \tag{10}
$$

Fig 10 的 pseudocode 非常优雅——forward 时只 cache 索引 j 和最小 anchor pair（line 13），backward 时 uncache 并 xor 翻转 bit（line 25）。**不需要保存 activation vectors**，类似 ANN transformer 的 KV-cache 思想。

### 5.2 Spiking RNN

Elman-style：
$$
\mathbf{h}_t = \mathbf{S}_{\mathbf{h}_{t-1}} + \mathbf{z}_t, \quad \mathbf{h}_0 = 0 \tag{11}
$$

embedder 用 LUT $\mathbf{E}$ 把 input char 转成 $\mathbf{z}_t \in \mathbb{R}^n$，unembedder 用 LUT $\mathbf{U}_{\mathbf{h}_t}$ 输出 vocab distribution。

**为什么 spiking RNN 抗 vanishing/exploding gradient**：
1. 每个 time step 的 $\mathbf{S}_{\mathbf{h}_t}$ 选不同 row，不像标准 RNN 反复乘同一个 matrix
2. Pairwise comparison 对绝对 magnitude 不敏感 —— 元素都是 $10^{10}$ 还是 $10^{-10}$ 不影响 ordering（只影响学习率 via $U'$）

### 5.3 SNN Transformer（重头戏）

ANN attention：$\text{softmax}(QK^\top/\sqrt{d_k})V$。直接用 LUT 替换 Q/K/V 生成仍有 softmax bottleneck 和 V matmul。

Izhikevich 的方案（Fig 12）：对每个 Query-Key pair，**concatenate**：
$$
[\mathbf{z}_i, \mathbf{z}_j, \mathbf{PE}_{i-j}] \in \mathbb{R}^{2n+p} \tag{14}
$$

其中 $\mathbf{PE}_{i-j} \in \mathbb{R}^p$ 是 learnable 的 relative positional encoding。要求 $p! > n_{inp}$ 才够编码所有相对位置。

然后直接通过单个 LUT $\mathbf{V}$ 得到 attention 输出：
$$
\mathbf{x}_i = \mathbf{z}_i + \sum_{j=1}^{i-1} \mathbf{V}_{[\mathbf{z}_i, \mathbf{z}_j, \mathbf{PE}_{i-j}]} \tag{15}
$$

**No softmax, no QK^T dot product, no V matmul**。Attention score 通过 LUT 内的 synaptic vector magnitude 隐式学到。

### 5.4 V-index cache

naive 实现：构造 $n_{inp} \times n_{inp}$ 个 concatenated pair，复杂度 $O(n_{inp}^2)$。

聪明做法（线性复杂度）：
- 对每个 position 的 embedding $\mathbf{z}_{pos}$ 用 $H(\cdot)$ 预计算 index $j_{pos}$
- 对每个 $\mathbf{PE}_{pos}$ 用另一组 anchor 预计算 $j^p_{pos}$
- 任意 pair $(pos_1, pos_2)$ 的 concatenated index 就是 $j_{pos_1} \Vert j_{pos_2} \Vert j^p_{pos_1 - pos_2}$

这正好对应 ANN transformer 的 KV-cache：算 $\mathbf{z}_i$ 的 index 相当于算 $Q_i$；算 $\mathbf{z}_j$ 的 index 相当于 $K_j$；concat 相当于 $Q_i K_j^\top$；用 concatenated index 查 $\mathbf{V}$ 相当于乘 value。

---

## 6. 实验：apples-to-apples 比较

任务：byte-level text prediction（Shakespeare-like），32 字符 context。**没有 hyperparameter search，没有 dropout，没有 L2，初始化全 0**。

### 6.1 Spiking RNN（Table I）

| 参数 | 值 |
|---|---|
| Context $n_{inp}$ | 32 |
| Hidden dim $n$ | 64 |
| Embedder footprint | 16K |
| LUT $\mathbf{S}_h$：$n_t$ | 64 |
| LUT $\mathbf{S}_h$：$n_c$ | 10 |
| Footprint | $n_t 2^{n_c} n = 4M$ |
| **Bandwidth/token** | $2n_t n_c + n_t n = 5.4K$ |
| Unembedder footprint | 1M |
| **Total** | **5M** |

注意 bandwidth (5.4K) 远小于 footprint (4M) —— 这是 SNN 的核心 asymmetry，ANN 里两者相等。

**BPC 比较（Table II）**：
- 5M spiking RNN: **1.39 BPC**
- 17M MI-LSTM (2016): 1.44
- 10M mLSTM (2016): 1.40
- 16M BN LSTM: 1.36
- 35M HM-LSTM: 1.32
- 806M SNN Transformer: **0.99**

也就是说 5M spiking RNN 在没有 LSTM gates、没有正则化、没有 Adam 的情况下，已经接近 2016 年 SOTA LSTM 的水平。

### 6.2 SNN Transformer（Table III, IV）

ANN transformer：$N=6$, $h=8$, $d_{model}=512$, $d_{ff}=2048$, $d_k=64$。
SNN transformer：$n=32$, $h=4$, $n_t=16$, $n_c=6$, $p=4$。

Table IV 的对比惊人：
- **Computational cost per layer/head**: ANN 235,405,312 vs SNN 172,800 —— **1360x 减少**
- **Memory bandwidth per new token**: ANN $1,048,576 + 576 n_{inp}$ vs SNN $120 + 30 n_{inp}$ —— **约 10000x 减少**
- **Memory footprint**: ANN 3,145,728 vs SNN 10,496,000 —— SNN 更大（这是预期，SNN 用大 footprint 换低 bandwidth）

Fig 13：**学习收敛速度差 50 倍**。Adam 学习率是为 ANN 调的，对 SNN 并不优化。

### 6.3 Ablation：attention-only SNN

把 FFN 完全去掉，只用 attention（$n=16$，单 head）。仍然能 work，因为 LUT 本身就是 non-linear，attention 同时做 token mixing 和 channel mixing。这是 paper 里很美的观察 —— **"SNN attention is all you need"**，因为 LUT 已经内含非线性，不需要 FFN 再做 channel mixing。

---

## 7. LSH 视角统一

Section VI 把整个 framework 抽象为 LSH：

- **Forward (the hash)**：$H(\mathbf{x})$ 把连续 input space $\mathcal{X}$ 分成有限个 non-overlapping buckets。同一 bucket 的输入映射到同一 $\mathbf{S}_j$ —— 自然带 robustness 和 generalization
- **Backward (the gradient)**：因为 hash 不可微，用 surrogate gradient。找 nearest neighbor $\bar{\mathbf{x}}$ 使 bucket flip 到 $\bar{j}$，用 $(\mathbf{S}_{\bar{j}} - \mathbf{S}_j)$ 与 error $\mathbf{v}$ 的 alignment $g$ 来决定 surrogate 方向

LSH 视角的 generalization 让我们可以选其他 hash function（Section VIII-C 讨论用 quantized timing bins 而不是 sign，得到 $m^{n_t n_c}$ 而不是 $2^{n_t n_c}$ patterns；Section VIII-H 讨论用 hyperplane LSH：$u_{ir} = \mathbf{c}_{ir} \cdot \mathbf{x}$，可以插入现有 16384 维 transformer 当 hybrid）。

> LSH 原始论文：Indyk & Motwani 1998 https://dl.acm.org/doi/10.1145/278298.278316

---

## 8. 容量分析（Section VIII-A, B）

### 8.1 ANN 容量

近 orthogonal vectors 的 linear capacity 来自 Johnson-Lindenstrauss：$e^{c(\varepsilon)n}$，其中 $c(\varepsilon) \approx \varepsilon^2$。即使 $e^n$ 也是天文数字，但远小于 $n!$。

ANN attention 用 $M = \sum_i \mathbf{k}_i^\top \mathbf{v}_i$ 存 (key, value) pairs。优点是 robustness（noise 不影响 retrieval）和 superposition（$\mathbf{k}_i + \mathbf{k}_j$ 检索 $\mathbf{v}_i + \mathbf{v}_j$）—— Anthropic 的 superposition 论文 [32,33] 专门讨论这个。

### 8.2 SNN 容量

把时间量化为 $m$ 个 bins（Fig 15）。n 个神经元各发 1 spike 有 $m^n$ 个 patterns。

- 当 $n < m$：以 $n!$ 为主导（排列数）
- 当 $n > m$：以 $m^n$ 为上界（多 spike 落同一 bin）

例：100 神经元，100ms 窗口，1ms 分辨率 → $100^{100} = 10^{200}$ patterns。

我们不需要检测所有 patterns（存储超宇宙容量）。我们只检测 $2^{n_t n_c}$ 个，但通过增大 $n_c$ 可以任意扩展而**不增加 active memory bandwidth** —— 这是 SNN scaling 的精髓。

### 8.3 Polychrony 的 robustness 与 superposition

- **Robustness**：polychronous pattern 只依赖 anchor 的相对顺序，其他神经元 noise 不影响
- **Superposition**：不同 anchor 集合的 patterns trivially superpose；重叠 patterns 只要 anchor 重叠部分顺序一致就能共存。一个 $n_c$-anchor pattern 与 $n!/n_c!!$ 个其他 pattern 共存 —— 实际上无穷

**Word embedding analogy**：anchor $x_1 > x_2$ 对应 "king/boy/son"（阳性），反过来对应 "queen/girl/daughter"（阴性）。ANN 用向量加法生成新 feature，SNN 用 spike 顺序 permutation 生成新 feature —— 这就是为什么 ANN 需要 matmul，SNN 需要 LUT。

---

## 9. 与其他架构的关系（Appendix 精华）

### 9.1 Mixture-of-Experts（Section VIII-L）

SNN 是最简单的 MoE：router = $H(\mathbf{x})$，从每个 table 选 1 row。完美 load balance（每次选固定数量 row）。核心 benefit 一致：参数量远大于单次计算用量。MoE 论文：https://arxiv.org/abs/1701.06538

### 9.2 Random ferns（Section VIII-N）

把 LUT 输出做 softmax 当 class probability，$y_k = \prod_i p_{i\mathbf{x}k}$ —— 这就是 semi-naive Bayes classifier，叫 random fern。Deep forests [44] 和 fern-based deep nets [45] 都因不可微而只能用非 gradient 方法。**SNN 用 surrogate gradient 解决了这个长期问题**。

### 9.3 Finite-state machines（Section VIII-M）

把 layer 间传递的 index tuple $(j_1, ..., j_{n_t})$ 看作 state。状态空间 $2^{n_t n_c}$，例如 64×10 的 spiking RNN 是 $2^{640} \approx 10^{192}$ —— 实际上无穷。可以视为巨大 FSM。

### 9.4 Transformer quantization 系列（Section VIII-O）

Spike-driven Transformer [46], SpikeLLM [47], SpikeGPT [48], SpikeBERT [49], SpikeFormer [50], BrainTransformers [51], Loihi 2 370M LLM [53], MatMul-free LM [54] —— 这些都没有利用 SNN 的 combinatorial capacity，本质上只是把 ANN 量化成 (-1, 0, +1) 三值 spike，用 sparsity 加速，**根本没引入 LUT**。

Izhikevich 明确指出这些工作"all can be viewed as an effort to quantize data"，而他的 manifesto 是**架构层面的替换**。

> SpikeGPT: https://arxiv.org/abs/2302.13939
> MatMul-free LM: https://arxiv.org/abs/2406.02528

### 9.5 Neuromorphic systems（Section VIII-P）

TrueNorth、Loihi、SpiNNaker 都在 transformer 之前设计，未考虑 deep learning 架构。Paper 主张：与其建模 "neuron-to-neuron" 交互，不如直接建模 "pattern-to-pattern" 交互，正如 ANN 早已从 neuron-level 抽象到 vector-level。

> TrueNorth: https://ieeexplore.ieee.org/document/7056730
> Loihi 2: https://www.intel.com/content/www/us/en/research/neuromorphic-computing-loihi-2-technology-brief.htm

### 9.6 Hybrid approach（Section VIII-H）

对 16384 维 modern LLM，可以用 hyperplane LSH：$u_{ir} = \mathbf{c}_{ir} \cdot \mathbf{x}$ 把空间分成 $2^{n_c n_t}$ 个 cones，依然远超 ANN linear capacity。Anchor vectors 可训练：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{c}_i} = U'(\mathbf{c}_i \cdot \mathbf{x})\,\mathbf{x}\,g_i
$$

性能更好（更多参数），但需要 matmul —— 可作为把 SNN 引入主流 ML 的 "beachhead"。

### 9.7 Parameter-efficient fine-tuning（Section VIII-J）

两个 SNN-native 的 PEFT 方案：

- **$n_t + 1$（new table）**：加一个全 0 初始化的 table，只对这个 table 做 backprop。初始 forward 不变，新 table 专门学新 task。
- **$n_c + 1$（new comparison）**：某 table 加一对 anchor，row 数翻倍 $2^{n_c} \to 2^{n_c+1}$，新行复制旧行。初始输出不变，但能 split 旧 pattern 为更细的 sub-pattern。

比 LoRA [39] 更"native"：https://arxiv.org/abs/2106.09685

### 9.8 Structural plasticity（Section VIII-I）

当前 anchor pairs 初始化后固定。可以训练它们（rewiring），原则：
1. 所有 row 被使用的概率均匀（避免死 row）
2. 互信息原则：anchor pair $\mathbf{u}_{ir}$ 把 table 分成两半，若两半 synaptic 值相似则该 pair 无信息量，应替换

### 9.9 Spiking backprop without matmul（Section VIII-K）

最激进的简化：每层只考虑**单个最小** $|u_{ir}|$ 的 anchor pair，且只考虑 $\mathbf{S}_{i\mathbf{x}}$ 忽略 $\mathbf{S}_{i\bar{\mathbf{x}}}$：

$$
h^l = U'(x_{a^l}^l - x_{b^l}^l)\,(s_{i\mathbf{x}b^{l+1}}^l - s_{i\mathbf{x}a^{l+1}}^l)\,h^{l+1}
$$

**只有单个 scalar 从 layer $l+1$ 传到 layer $l$**。Gradient 不经过 neurons，直接从 LUT 到 LUT（Fig 22）—— 暗示**同一 spiking hardware 可同时做 inference 和 learning**，对生物 plausibility of backprop [37] 也有意义。

---

## 10. Latency 与生物对应（Section VIII-D, E）

### 10.1 Latency 编码的生物基础

Fig 18：in vitro rat visual cortex pyramidal neuron 实验显示，input 越强 → spike latency 越短（monotonic）。Class 1 excitability 神经元天然把 input strength 编码为 latency。所以用 synaptic value 当 latency 是合理的近似。

### 10.2 Spiking 不限于神经元

Fig 17：non-neural cells（甚至 pumpkin 细胞）也会 spike。Nature 用 spiking 实现 LUT 是借用已有机制，不是为信息处理专门发明。

### 10.3 Edge detector toy example（Section VIII-D）

Fig 16：3×3 patch 9 个 pixel 当 binary index，LUT 有 $2^9 = 512$ rows。Horizontal edge = binary 000000111 = decimal 7 → neuron "7" 发 spike → 只激活第 7 行 outgoing synapse。这就实现了 V1 orientation-selective cell 的功能，**完全不需要模拟 lateral inhibition**。

**Astonishing observation**：spiking = LUT retrieval 这个解释居然从没被提出过（或被忽视了）。

---

## 11. 容量爆炸的来源再梳理

我把容量公式整理一下，让你直觉更清楚：

| 维度 | ANN | SNN |
|---|---|---|
| Encoding capacity | $e^{\varepsilon^2 n}$ (近正交) | $n!$ 或 $m^n$ |
| Transition | MatMul（O(n²)） | LUT 查表（O(n_t)） |
| Active params per forward | 全部 weight matrices | $n_t$ rows |
| Memory bandwidth | = footprint | ≪ footprint |
| Nonlinearity | ReLU/softmax | Index selection (binary comparisons) |

SNN 把**表示容量**和**计算成本**解耦：增加 $n_c$ → 容量翻倍 → 计算几乎不变。这在 ANN 里是不可能的（增加维度同时增加 matmul 成本）。

---

## 12. 我（GLM）的几点直觉与质疑

为了真正 build 你的 intuition，我坦白几个值得思考的点：

### 12.1 最小 anchor pair 的 gradient 是否太稀疏？

每层只对 $n_t$ 个 anchor pair 传 gradient（每个 pair 2 个 neuron 受影响），其他 $n - 2n_t$ 个 neuron 在该层 gradient 为 0。对 $n=64$、$n_t=64$ 来说勉强够（256 个 gradient 信号），但 $n$ 大了会怎样？Section VIII-K 的"single pair per layer"更激进，paper 自己说"worst learning performance"。

### 12.2 Pattern collision 问题

$2^{n_t n_c}$ 个 patterns 共享 $n!$ 个真实 spike orderings（$n > 2^{n_t n_c}$ 时）。但反方向也可能 collision：不同 latency vector 映射到同一 index 集合 —— 这是 LSH 的 inherent property，paper 把它说成 feature（robustness）而不是 bug。但当 training set 内有真正不同 input 落同一 bucket 时，会发生 interference。需要 $n_t n_c$ 足够大。

### 12.3 Attention 的 $2n + p$ 维 concat 后 LUT 巨大

Table III：attention LUT footprint = $n_t \cdot 2^{2n_c + p} \cdot n$。$n_c=6$, $p=4$ → $2^{16} = 65536$ rows per table × $n_t=16$ tables × $n=32$ = 33M。Footprint 全在 attention（FFN 只 33K）。这跟 ANN 相反（ANN 大头在 FFN）。如果 $n_c$ 加到 10，attention footprint 爆炸 $2^{24} \approx 16M$ rows。**这个 scalability 没有完全解决**。

### 12.4 与 MatMul-free LM [54] 的关系

Zhu et al. 2024 的 MatMul-free LM 用 ternary weights + 极简 op，也是"avoid matmul"路线。区别：MatMul-free 仍保留 matmul-style 信息聚合（只是量化），而 Izhikevich 是**根本换 representation**（latency vector + ordering）。两种哲学完全不同。

### 12.5 与 Hopfield network / Modern Hopfield 的关系

ANN attention 本质是 modern Hopfield network 的 retrieval。SNN attention 是 LSH-based associative memory。两者都是 associative memory，但 SNN 利用 spike ordering 而非 dot product 来 address。这个连接 paper 没明确提，但 intuition 上很有启发性。

> Modern Hopfield: https://www.pnas.org/doi/10.1073/pnas.2001926117

### 12.6 与 Predictive Coding / Free Energy Principle 的联想

Friston 一派的 predictive coding 强调 brain 用 sparse coding + hierarchical surprise minimization。SNN 的 LUT + sparse activation 跟这个哲学一致：大部分神经元 silent，只更新 surprise 大的位置。Paper 没讨论这个连接，但 Izhikevich 的 early work on polychronization 跟 dynamical systems / Friston 派有交集。

> Friston FEP: https://www.nature.com/articles/nrn2787

---

## 13. 总结：这篇 manifesto 真正的贡献

1. **Philosophical shift**：把 spiking 从"bio-inspired activation function"重新定位为"nature's LUT"。这是真正的 paradigm shift。
2. **Computational asymmetry**：明确分离 footprint（大）和 bandwidth（小），这是 scaling 的关键。
3. **Capacity argument**：用 $n! \gg e^n$ 给出 SNN 比 ANN 表达力强的数学论证。
4. **Practical bridge**：给出 deep SNN、spiking RNN、SNN transformer 三种可直接实验的架构，code 开源 https://github.com/izhikevich/SNN
5. **LSH 统一视角**：让 framework 可扩展到其他 hash function，包括 hybrid hyperplane LSH 可以插入现有 LLM。
6. **Spiking backprop without matmul**：Section VIII-K 暗示 training 也可在 spiking hardware 上做，对 neuromorphic 和 bio-plausibility 都有 implication。

---

## 14. 关键 web links

- **Code**: https://github.com/izhikevich/SNN
- **Polychronization 原始论文 (Izhikevich 2006)**: https://direct.mit.edu/neco/article-abstract/18/2/245/682/Polychronization-Computation-with-Spikes
- **Surrogate gradients in SNNs**: https://arxiv.org/abs/1901.09948
- **Training SNNs lessons from DL**: https://ieeexplore.ieee.org/document/10254215
- **MatMul-free LM (Zhu et al.)**: https://arxiv.org/abs/2406.02528
- **SpikeGPT**: https://arxiv.org/abs/2302.13939
- **SpikeLLM**: https://arxiv.org/abs/2407.04752
- **Loihi 2 LLM**: https://arxiv.org/abs/2503.18002
- **Anthropic Superposition**: https://transformercircuits.pub/2022/toy-model
- **Johnson-Lindenstrauss**: https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma
- **Locality-sensitive hashing (Indyk-Motwani)**: https://dl.acm.org/doi/10.1145/278298.278316
- **LoRA**: https://arxiv.org/abs/2106.09685
- **Outrageously Large MoE**: https://arxiv.org/abs/1701.06538
- **Modern Hopfield Networks**: https://www.pnas.org/doi/10.1073/pnas.2001926117
- **Izhikevich Dynamical Systems in Neuroscience book**: https://mitpress.mit.edu/9780262090436/dynamical-systems-in-neuroscience/

---

## 15. 给你的 build-intuition 总结

如果你只想带走三件事：

1. **Spike timing 是 free encoding capacity**：60 个 spike 的 ordering 给你 $60!$ 个 states，远超 $\mathbb{R}^{60}$ 的近正交 capacity。这是 paper 一切的起点。
2. **LUT = spiking 的抽象**：每次 forward 只激活 $n_t$ 行，footprint 与 bandwidth 解耦 → 可以指数扩容不增算力。
3. **Surrogate gradient 让 LUT 可微**：通过"flip minimal anchor pair"找 nearest bucket boundary，用 $(\mathbf{S}_{\bar{j}} - \mathbf{S}_j)$ 与 error 的 alignment 来定 gradient 方向。Self-regularization 来自 gradient 的零均值性质。

后续真正值得做的工程方向：(a) hardware design for LUT-fetch + addition（既非 GPU 也非传统 neuromorphic），(b) hybrid hyperplane LSH 插入现有 16384-维 LLM，(c) structural plasticity 让 anchor 可学，(d) spiking-native PEFT ($n_t+1$ / $n_c+1$)。

这篇 manifesto 写得很有 Izhikevich 风格 —— 短、狠、provocative。它的真正价值可能不在于立刻 beat Llama，而在于**重定义了 spiking 这个词在 ML 里的含义**：从"bio-inspired activation"变成"latency-ordering-based LUT"。这个 mental shift 一旦普及，后面的工程实现只是时间问题。
