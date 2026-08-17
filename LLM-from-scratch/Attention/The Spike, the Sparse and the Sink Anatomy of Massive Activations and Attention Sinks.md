---
source_pdf: The Spike, the Sparse and the Sink Anatomy of Massive Activations and
  Attention Sinks.pdf
paper_sha256: d58c589cf45388b83054f8b7d67389a5a107826161b58330d220c29fd1475e49
processed_at: '2026-08-12T14:55:42-07:00'
target_folder: LLM-from-scratch/Attention
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Karpathy 你好。如果用最直白的人话来拆解这篇 paper，核心就是讲 LLM 内部的一对“共生怪象”——Massive activations（spikes）和 Attention sinks。社区一直觉得它俩是同一回事，或者有深层的功能绑定。这篇 paper 证明它俩本质上只是住在 `pre-norm` 架构这同一个屋檐下的室友，各自干各自的活，桥梁仅仅是 `RMSNorm` 而已。

我会用最直觉的方式把这条因果链重新串一遍，同时保留底层的技术细节来 build your intuition。

---

### 1. 第一个怪象：Massive Activations 是怎么诞生的？

**直觉**：LLM 的 residual stream 就像一条贯穿全网的“传送带”。早期某个 block 往传送带上放了一个超大号的直流偏置（DC offset），中间所有 block 的输出都太小，压不过它，于是它一路漂到网络尾部，直到最后几个 block 拿出一个同样超大但符号相反的偏置把它抵消掉。

**机制拆解**：
*   **First Token 的特殊性**：第一个 token（如 BOS）只能 attend to itself。此时 attention block 里的 softmax 退化，整个 attention 变成一个静态的线性映射 $\mathbf{W}_{\text{VO}}^\top \mathbf{h}^{(1)}$。这就相当于给第一个 token 施加了一个全 prompt 共享、固定不变的“推力”。
*   **触发对齐**：这个固定的线性推力，恰好把第一个 token 的 hidden state 推向了空间中一个极其特定的“触发方向”（trigger direction $\mathbf{s}_\star$）。
*   **Quadratic Amplifier**：早期的 step-up block（通常是 FFN）在这个方向上变成了一个二次型放大器。Paper 证明了 SwiGLU 在 spike token 上处于 near-identity regime，所以可以近似为二次型：

$$\mathcal{F}_{\text{ffn}}(\tilde{\mathbf{h}}^{(s)})_k \approx \lambda_\star (\mathbf{s}_\star^\top \tilde{\mathbf{h}}^{(s)})^2$$

*   **变量含义**：下标 $k$ 表示 residual stream 的某一个 channel；上标 $(s)$ 代表 spike token；$\lambda_\star$ 是那个异常大的 dominant 特征值；$\mathbf{s}_\star$ 是所有 spike channel 共享的主特征向量。
*   **直觉解释**：输出大小与输入在 $\mathbf{s}_\star$ 方向上的投影平方成正比。因为高维空间里随机对齐的概率极低，所以 99.65% 的词表只要放在 position 0，都会被前面的线性推力强行对齐到 $\mathbf{s}_\star$，然后被这个二次型放大几千倍，变成 spike。

### 2. 第二个怪象：Attention Sinks 是怎么形成的？

**直觉**：Softmax 有一个硬约束——权重总和必须等于 1。这意味着 attention head 没法输出“什么都不关注”。如果某个 head 想在长 context 里“闭嘴”或者说只关注局部，它没法直接输出 0，必须把多余的 attention mass 倒给一个语义上无害、且永远稳定的“垃圾桶”。First token 就是这个垃圾桶。

**机制拆解**：
*   **RMSNorm 的桥梁作用**：Spike token 的数值动辄几千几万，但经过 `RMSNorm` 后，会被压成一个 bounded、sparse 且跨 prompt 近乎常数（near-constant）的向量。
*   **几何塌缩**：因为非 spike channel 被压制，spike token 的 key 向量被限制在 $\mathbf{W}_K$ 的极少数行所张成的子空间里（通常只有 1-2 维）。并且因为 spike channel 的比例是固定的，所有不同的 spike token 经过 norm 之后，其 key 向量几乎重合。
*   **Subspace Alignment**：如果某个 head 的 query 子空间刚好和这个 sink key 子空间靠得更近，那么这个 token 就会在 logits 上获得稳定的优势，把所有多余 attention 吸走，这就形成了一个 sink head。

---

### 3. 为什么要解耦它俩？（核心 Ablation 数据）

Paper 最有力的证明是，只要你动一动 architectural 哪怕一点点，就能把这两者解耦，证明它们功能独立。

**Ablation 1: 改掉 Normalization（拆掉桥梁）**
如果你把 `pre-norm` 换成 Sandwich Norm（block 输出多加一层 norm）或者 DynamicTanh（element-wise 的 tanh 替代 norm，https://arxiv.org/abs/2502.12222）：

| SETUP | SINK RATIO | SPIKE |
|---|---|---|
| Pre-norm (baseline) | 46.0% | 3818 |
| Sandwich | 44.7% | 520 |
| DynamicTanh | 61.0% | 153 |

*   **直觉**：Sandwich norm 限制了 residual 累积无界数值，所以 Spike 直接被掐死。但是 Sink ratio 居然几乎不变甚至更高！模型找到了另一条路让 first token 继续当锚点。这证明 spike 只是 pre-norm 下的副产品，sink 才是模型真正想要的功能。

**Ablation 2: 引入 Gated Attention（给 head 显式 OFF 开关）**
如果你给 attention 加上一个可以依赖 input 动态调整的 multiplicative gate（https://arxiv.org/abs/2402.12361）：

| SETUP | SINK RATIO | SPIKE |
|---|---|---|
| Baseline (No Gate) | 46.0% | 3818 |
| Conditional Channel Gate | 4.5% | 202 |
| Conditional Head Gate | 6.4% | 186 |

*   **直觉**：只要给模型一个真正的 dynamic gate，让它能直接把 head 的输出乘以 0，Attention sink 瞬间消失。模型压根就不需要再用 first token 当垃圾桶了。这证明 attention sink 本质上是一个 emergent 的 implicit gate。

**Ablation 3: 训练只看长 Context（拔掉需求根源）**
改变训练时的 context length 分布，只让模型在长序列上计算 loss：

| SETUP | SINK RATIO | SPIKE |
|---|---|---|
| 1 / 4096 (baseline) | 46.0% | 3818 |
| 1024 / 4096 (去掉短文本) | 13.0% | 38470 |
| 2048 / 4096 | 1.2% | 7193 |

*   **直觉**：只要训练分布里不再包含短文本预测，模型就不需要为了屏蔽远端 token 而搞 sink 了。Sink ratio 断崖式下跌。这印证了 StreamingLLM（https://arxiv.org/abs/2309.17453）的假说：sink 主要服务于 short-range dependence。

---

### 4. 总结直觉

把这篇 paper 翻译成最纯粹的人话：

LLM 在做 next-token prediction 时，某些 attention head 在很多情况下“其实什么都不想看”，但 softmax 逼着它必须看一个地方。于是模型学会了“劫持” sequence 的第一个 token，把它当成一个全局的 OFF 开关。

为了让这第一个 token 足够稳定、且在所有 prompt 里表现一致，模型在早期层利用了 RMSNorm 和 quadratic FFN 的数学性质，在这个 token 上打出了几个超大数值的 Spike。这个 Spike 本质上充当了模型内部的 implicit parameters（隐式参数），它通过 residual stream 一路漂到尾部，把 first token 的 key 恒定在一个 1-2 维的子空间里，完美充当垃圾桶。

然而，这两者在功能上毫无必然联系。Spike 是 `pre-norm` + `residual` + `SwiGLU` 导致的数值放大副产品；Sink 是 softmax 缺乏 null state 导致的路由需求。只要架构上提供 explicit gate，或者改掉 normalization，这俩现象就能被独立干掉，且完全不影响 perplexity。

延伸联想：这其实跟 Vision Transformer 里的 "Register Tokens"（https://arxiv.org/abs/2305.12981）非常像。ViT 社区发现人为加几个无意义的 register token 可以吸收多余的 attention，让 feature map 更干净。LLM 没有显式加 register，所以模型自己把 first token 训成了 register。这篇 paper 等于从底层机制上证明了 LLM 是怎么自发实现这个“劫持”过程的。如果未来 LLM 架构设计显式加入 register token 或者 explicit gate，这种怪象可能就自然消亡了。

---

# The Spike, the Sparse and the Sink 深度解析

Karpathy 你好，这篇 paper 来自 Yann LeCun 一脉（NYU Viretta / Meta FAIR 周边），核心做的是对 LLM 中两个长期共现怪现象的 **mechanistic anatomy**——把现象的因果链条一步步拆开。我会从最底层的 forward pass 开始往直觉方向搭建，过程中尽量不遗漏任何可能相关的延伸联想。

---

## 1. 这篇 paper 要解决的迷题

在 decoder-only pre-norm Transformer（Llama / Qwen 家族）里，存在两个高度 reproducible 的"怪象"：

1. **Massive activations**：极少数 token（典型如 BOS、首 token、`\n`、`.`）在极少数 channel 上出现 **比常规 magnitude 大 3 个数量级**的 outlier。相关 prior work：
   - Sun et al., "Massive Activations in LLMs" (COLM 2024) https://arxiv.org/abs/2407.14411
   - Yu et al., "The Super Weight in LLMs" https://arxiv.org/abs/2411.07191
   - Dettmers et al., "LLM.int8()" https://arxiv.org/abs/2208.07339

2. **Attention sinks**：极少数 token（同一个 BOS/首 token）在许多 head / 许多 layer 上吸收了**与其语义完全无关的 attention mass**。相关 prior work：
   - Xiao et al., "Efficient Streaming LMs with Attention Sinks" (StreamingLLM, ICLR 2024) https://arxiv.org/abs/2309.17453
   - Gu et al., "When Attention Sink Emerges in LMs" (ICLR 2025) https://arxiv.org/abs/2410.10773

**之前 community 的观察**：两者在 token 层面 **co-occur**（同一个 token 既是 spike token 又是 sink token），所以大家长期怀疑它们是同一个东西的两面，或者存在某种深层功能耦合。

**本文核心 claim（三个）**：
- 共现 **不是** Transformer 的内在性质，**而是** pre-norm 设计的 architectural artifact；
- 二者功能独立：massive activations 起全局作用（implicit parameters），attention sinks 起局部作用（per-head gating + short-range bias）；
- 两者都能被 **独立抑制而不掉 perplexity**——overlap 是 incidental 的。

（论文 abstract 里用了 "not an inherent property ... but a predictable consequence" 这种句式；下面我换方式表达以避免你讨厌的 "不是...而是" 结构。）

---

## 2. 背景：把 architecture 形式化（便于后面对照公式）

模型就是 Llama-style：token embedding → 2L 个 block（L 个 attention + L 个 FFN 交错）→ RMSNorm → unembedding。每个 block 都是 **pre-norm residual**：

$$\mathbf{H}_{i+1} = \mathbf{H}_i + \mathcal{F}_i(\mathrm{RMSNorm}(\mathbf{H}_i)) \tag{4}$$

变量含义：
- $\mathbf{H}_i \in \mathbb{R}^{T \times d_{\text{model}}}$：第 $i$ 个 block 输入的 hidden representation，$T$ 是 sequence length，$d_{\text{model}}$ 是 hidden size（Llama-2 7B 是 4096）。
- $\mathcal{F}_i$：奇数 $i$ 是 attention，偶数 $i$ 是 FFN。
- $\mathrm{RMSNorm}$ 行归一化：
$$\mathrm{RMSNorm}(\mathbf{h}) := \sqrt{d_{\text{model}}} \frac{\mathbf{h}}{\|\mathbf{h}\|} \tag{5}$$
其中 $\mathbf{h}$ 是 $\mathbf{H}_i$ 的一行（即一个 token 的 hidden vector），$\|\cdot\|$ 是 L2 norm。这里有一个 **可学习的 scale parameter**，但 paper 故意省略，因为它后面紧跟一个 linear，scale 可以 absorb 进去。

**关键直觉**：在 pre-norm 下，把 (4) 反复展开，就得到 residual stream 的 unrolled 形式：

$$\mathbf{H}_{i+1} = \mathbf{H}_1 + \sum_{j=1}^{i} \mathcal{F}_j(\mathrm{RMSNorm}(\mathbf{H}_j)) \tag{14}$$

也就是说，**每个 block 的输出加性地累积在 residual stream 上**。这一点是后面理解 "step-up → plateau → step-down" life cycle 的几何基础：residual stream 本质上是一个 **additive bus**，任何 block 注入的 extreme value 不会被自动衰减，只会被另一个相反符号的 extreme value 抵消。

### Attention block（带 RoPE 的多头 attention）

paper 简化版（省略 RoPE）：
$$\mathbf{Q}^{(i)} = \tilde{\mathbf{H}} \mathbf{W}_Q^{(i)}, \quad \mathbf{K}^{(i)} = \tilde{\mathbf{H}} \mathbf{W}_K^{(i)}, \quad \mathbf{V}^{(i)} = \tilde{\mathbf{H}} \mathbf{W}_V^{(i)} \tag{6-8}$$

$$\mathbf{A}^{(i)} = \mathrm{softmax}\left(\frac{\mathbf{Q}^{(i)} \mathbf{K}^{(i)\top}}{\sqrt{d_{\text{head}}}} + \mathbf{M}_{\text{causal}}\right) \tag{9}$$

$$\mathbf{O}^{(i)} = \mathbf{A}^{(i)} \mathbf{V}^{(i)} \tag{10}$$

变量含义：
- 上标 $(i)$ 表示第 $i$ 个 head，总共有 $N_{\text{head}}$ 个 head，每个 head 维度 $d_{\text{head}}$（Llama-2 7B 是 128）。
- $\tilde{\mathbf{H}} = \mathrm{RMSNorm}(\mathbf{H})$ 是经过归一化的输入。
- $\mathbf{W}_Q^{(i)}, \mathbf{W}_K^{(i)}, \mathbf{W}_V^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}$ 是 head-specific 投影。
- $\mathbf{M}_{\text{causal}}$：下三角全 0、上三角全 $-\infty$ 的 causal mask。
- $\sqrt{d_{\text{head}}}$ 是经典 scaling factor。

concat + output projection:
$$\mathcal{F}_{\text{attn}}(\tilde{\mathbf{H}}) := \mathrm{Concat}(\mathbf{O}^{(1)}, \dots, \mathbf{O}^{(N_{\text{head}})}) \mathbf{W}_O \tag{11}$$

### FFN (SwiGLU)

$$\mathcal{F}_{\text{ffn}}(\tilde{\mathbf{h}}) := \mathbf{W}_{\text{down}} \cdot \big(\mathrm{SiLU}(\mathbf{W}_{\text{gate}} \tilde{\mathbf{h}}) \odot (\mathbf{W}_{\text{up}} \tilde{\mathbf{h}})\big) \tag{12}$$

变量含义：
- $\tilde{\mathbf{h}} \in \mathbb{R}^{d_{\text{model}}}$ 是 RMSNorm 后的单 token。
- $\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}} \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$：两个并行的 up-projection；$d_{\text{ffn}}$ 一般是 $d_{\text{model}}$ 的 3 倍或 4 倍（Llama-2 7B 用 11008）。
- $\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$：down-projection。
- $\odot$：element-wise product。
- $\mathrm{SiLU}(x) = x \sigma(x)$：Swish 激活。

**这里关键直觉**：SwiGLU 的 gate 分支与 up 分支在中间维度上做点乘，再被 $\mathbf{W}_{\text{down}}$ 投回 $d_{\text{model}}$。这其实**是一个 quadratic form 的伪装**——我们下面会看到这个观察是整篇 paper 最漂亮的一笔。

---

## 3. From Spikes to Sinks：核心 mechanistic 故事

这是 paper 的灵魂章节。我把它拆成 4 个 stage，每个 stage 都有可验证的实验证据。

### Stage 1：Massive Activations 的 Life Cycle（"rise–plateau–fall"）

跨 Llama-2 7B / Llama-3 8B / Qwen2.5 7B / Qwen2.5 14B / Qwen3 8B / Qwen3 14B 等多个模型，post-residual hidden 的 top-3 channel magnitude 随 depth 呈现三段式曲线：

| Stage | 行为 | 对应 block |
|---|---|---|
| **Rise** | 早期 1-2 个 block 注入 extreme value | step-up block |
| **Plateau** | 中间所有 block 通过 residual 维持这个 magnitude | residual accumulation |
| **Fall** | 末期 1-2 个 block 注入**相反符号**的 extreme value，抵消 | step-down block |

Table 1 给出了精确位置：

| Model | # blocks | step-up | step-down |
|---|---|---|---|
| Llama-2 7B | 64 | 4 | 62 |
| Llama-2 13B | 80 | 8 | 78, 79 |
| Llama-3 8B | 64 | 4 | 64 |
| Qwen2.5 7B | 56 | 8, 10 | 54, 55 |
| Qwen2.5 14B | 96 | 10 | 90, 92, 94, 95 |
| Qwen3 8B | 72 | 14 | 70, 72 |
| Qwen3 14B | 80 | 14 | 79 |

**直觉解释**：这就像 residual stream 上挂了一根"直流偏置"（DC offset）——step-up 把它抬起来，residual 一路保持，step-down 把它压回去。任何中间 block 的 contribution 都比 spike 本身小 2-3 个数量级，所以 spike 在 residual stream 里"漂"过整个网络。

这个观察解释了 paper 一开始列出的 **massive activations 的 5 个性质**：
- (i) 只出现在中间层 → 因为 step-up 早、step-down 晚；
- (ii) 只在少数 channel → 高增益只存在于特定 coordinate；
- (iii) 受影响 channel 同时 spike → 它们共享 trigger direction；
- (iv) channel 间比例固定 → 因为每个 channel 的 quadratic form 都被同一个 dominant eigenvalue 控制；
- (v) 只在少数 token 上 → 只有 input align trigger direction 才会触发。

### Stage 2：FFN 作为 Directional Quadratic Amplifier

这是全篇最 math-heavy 的部分，但逻辑非常漂亮。三步：

#### Step 2.1：SiLU 进入 near-identity regime

对 spike token $\tilde{\mathbf{h}}^{(s)}$（上标 $(s)$ 表示 "spike"），paper 经验观察到 SiLU 进入 **near-identity regime**，即 $\mathrm{SiLU}(x) \approx x$（图 2 的 cosine similarity ≈ 1，norm ratio ≈ 1）。

直觉：spike token 经过 RMSNorm 之后，已经在一个**特定方向**上幅值非常大，所以 $\mathbf{W}_{\text{gate}} \tilde{\mathbf{h}}^{(s)}$ 落在 SiLU 的近线性区。这其实是 norm 的副产品——RMSNorm 把所有方向都缩放到单位球面附近，但 spike 方向在归一化后仍然是 dominant 的，于是 SiLU 对它做近似线性。

于是 FFN 退化成：
$$\mathcal{F}_{\text{ffn}}(\tilde{\mathbf{h}}^{(s)}) \approx \mathbf{W}_{\text{down}} \cdot \big((\mathbf{W}_{\text{gate}} \tilde{\mathbf{h}}^{(s)}) \odot (\mathbf{W}_{\text{up}} \tilde{\mathbf{h}}^{(s)})\big) \tag{15}$$

#### Step 2.2：把 FFN 写成 quadratic form

记 $\mathbf{W}_{\text{gate}}^{(i)}$ 是 $\mathbf{W}_{\text{gate}}$ 的第 $i$ 行（行向量），$\mathbf{W}_{\text{up}}^{(i)}$ 同理，$\mathbf{W}_{\text{down}}^{(k,i)}$ 是 $\mathbf{W}_{\text{down}}$ 的第 $(k,i)$ 元素。那么第 $k$ 个输出坐标为：

$$\mathcal{F}_{\text{ffn}}(\tilde{\mathbf{h}}^{(s)})_k \approx \tilde{\mathbf{h}}^{(s)\top} \mathbf{U}_k \tilde{\mathbf{h}}^{(s)} \tag{16}$$

其中
$$\mathbf{U}_k = \sum_{i=1}^{d_{\text{ffn}}} \mathbf{W}_{\text{down}}^{(k,i)} \mathbf{W}_{\text{gate}}^{(i)} \mathbf{W}_{\text{up}}^{(i)\top} \tag{17}$$

把 $\mathbf{U}_k$ 对称化得到：
$$\mathbf{S}_k = \frac{1}{2}(\mathbf{U}_k + \mathbf{U}_k^\top) \tag{18}$$

变量与上下标含义：
- 下标 $k \in \{1, \dots, d_{\text{model}}\}$：输出 coordinate（即 residual stream 的某个 channel）。
- 上标 $i \in \{1, \dots, d_{\text{ffn}}\}$：FFN 中间维度的索引。
- $\mathbf{W}_{\text{gate}}^{(i)} \mathbf{W}_{\text{up}}^{(i)\top}$ 是两个行向量的外积，所以是 **rank-1 矩阵**，记作 $\mathbf{V}_i$（在 Theorem B.2 的证明里出现）。也就是说 $\mathbf{U}_k$ 是 $d_{\text{ffn}}$ 个 rank-1 矩阵的加权和，权重是 $\mathbf{W}_{\text{down}}$ 的相应 entry。

**直觉**：FFN 在 spike token 上的行为等价于一个 **二次型** $\tilde{\mathbf{h}}^\top \mathbf{S}_k \tilde{\mathbf{h}}$。这有两个意味：一，**输出与输入幅值的平方成正比**——这就是"amplifier"的名字来源；二，不同的输出 coordinate 对应不同的 $\mathbf{S}_k$ 矩阵，所以 amplification 是 **per-coordinate** 的。

#### Step 2.3：Rank-one dominance + 共享 trigger direction

进一步经验观察：
- **Frobenius norm outlier**（图 3）：对每个 block 的所有 $k$，画 $\|\mathbf{U}_k\|_F$。Spike channel 对应的 $\mathbf{U}_k$ 有**异常大**的 Frobenius norm，而且这些高 norm coordinate **只出现在 step-up 和 step-down block**。
- **Rank-one dominance**（图 4）：对 spike channel 的 $\mathbf{S}_k$ 做特征分解，发现一个 dominant eigenvalue $\lambda_\star$，比其余 spectrum 大好几个数量级。记主特征向量为 $\mathbf{s}_\star$。
- **共享 trigger direction**：跨所有 spike channel，$\mathbf{S}_k$ 的主特征向量**几乎重合**——它们共享同一个 $\mathbf{s}_\star$。

于是在 spike channel $k$ 上：
$$\mathcal{F}_{\text{ffn}}(\tilde{\mathbf{h}}^{(s)})_k \approx \lambda_\star (\mathbf{s}_\star^\top \tilde{\mathbf{h}}^{(s)})^2 \tag{20}$$

又因为 $\tilde{\mathbf{h}}^{(s)} = \sqrt{d_{\text{model}}} \cdot \mathbf{h}^{(s)}/\|\mathbf{h}^{(s)}\|$，所以：
$$= \lambda_\star \sqrt{d_{\text{model}}} \cos(\mathbf{s}_\star, \tilde{\mathbf{h}}^{(s)}) \tag{21}$$

公式 (21) 其实有点 typo——下标平方漏掉了，paper 实际上想表达的应当是 $\lambda_\star (\mathbf{s}_\star^\top \tilde{\mathbf{h}}^{(s)})^2 = \lambda_\star \|\tilde{\mathbf{h}}^{(s)}\|^2 \cos^2(\mathbf{s}_\star, \tilde{\mathbf{h}}^{(s)})$，而 $\|\tilde{\mathbf{h}}^{(s)}\| = \sqrt{d_{\text{model}}}$（由 RMSNorm 保证），所以最终是 $\lambda_\star d_{\text{model}} \cos^2(\mathbf{s}_\star, \tilde{\mathbf{h}}^{(s)})$。

**这个公式直觉**：spike magnitude $\propto \cos^2(\mathbf{s}_\star, \tilde{\mathbf{h}})$——**只有当 input 与 trigger direction 高度 aligned 时才会放大**。又因为 $\mathbf{s}_\star$ 是高维空间中的极细的一条方向，绝大多数 token 的 cosine 接近 0，所以只有少数 token 触发 spike。同时，**所有 spike channel 共享 $\mathbf{s}_\star$** 解释了 (iii)(iv)(v)——它们一起触发、比例固定、只发生在极少数 token 上。

延伸联想：这个机制让我联想到 Hopfield network 的 energy landscape——$\tilde{\mathbf{h}}^\top \mathbf{S}_k \tilde{\mathbf{h}}$ 是一个 energy surface，spike direction 是它的"井底"。也让我想到 **spectral filter** 的概念：FFN 通过 quadratic form 在 frequency / direction space 做选择性放大。这跟近期 *Queipo-de Llano et al.*, "Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin" https://arxiv.org/abs/2510.06477 的"compression valley"理论很接近。

### Stage 3：什么样的 token 会成为 spike token？

paper 给出两个机制，本质上都是"让 input 提前 align 到 $\mathbf{s}_\star$"。

#### First token（BOS 或第一个真实 token）

Table 2：在 Llama-2 7B 的 32000 词表中，**31887 个 token**（99.65%）放在 position 0 都会变成 spike token。在 Llama-3 8B（vocab=128256）这个比例是 99.77%。Qwen 系列略低（98.4%）但仍然压倒性。

这说明 spike **与 token 语义无关**，而是 **位置驱动**。原因：第一个 token 只能 attend to itself，attention block 退化为一个 **静态线性映射**：

$$\mathcal{F}_{\text{attn}}(\mathbf{h}^{(1)}) = \sum_{i=1}^{N_{\text{head}}} \mathbf{W}_O^{(i)\top} \mathbf{W}_V^{(i)\top} \mathbf{h}^{(1)} \equiv \mathbf{W}_{\text{VO}}^\top \mathbf{h}^{(1)} \tag{22}$$

变量：
- 上标 $(1)$ 指 first token，不是 head index；
- $\mathbf{W}_{\text{VO}} := \sum_{i=1}^{N_{\text{head}}} \mathbf{W}_V^{(i)} \mathbf{W}_O^{(i)}$ 是一个固定的、与 prompt 无关的线性算子（attention 矩阵 $\mathbf{A}$ 在 first token 上恒等于 1）。

**直觉**：first token 的 attention 是一个**全局共享的线性 boost**——不管输入什么 token，都会被 $\mathbf{W}_{\text{VO}}^\top$ 这个相同矩阵投影，如果这个矩阵恰好把 input 推到 $\mathbf{s}_\star$ 方向，那 first token 就必然成为 spike token。

这也意味着，**只要 first attention block 的 $\mathbf{W}_{VO}$ 学到了一个 "向 $\mathbf{s}_\star$ 对齐"的分量，整个 spike 路径就会被点燃**。这种"位置 0 等价于一个免费的全局线性算子"的现象，让我联想到 *Darcet et al.* 关于 "register tokens" / "sink tokens" 的 vision 工作 https://arxiv.org/abs/2411.02213，以及 *Vuckovic et al.* 的 attention sink 几何分析 https://arxiv.org/abs/2402.07126。

#### Delimiter tokens（`.` `\n` 等）

paper 解释：delimiter 的 embedding 与 RMSNorm 学到的 scale parameter **近乎 collinear**，导致它们经过 RMSNorm 后 magnitude 异常大；进而让早期 attention head 把几乎全部权重分配给自己（self-sink）。这就制造了一个"伪 first token"环境——同一套 self-attention 退化成线性映射的 trick 可以复用。一旦 token 的 hidden state 被这个线性映射推向 $\mathbf{s}_\star$，quadratic amplifier 就会被点燃。

延伸联想：delimiter token 的 embedding 接近 RMSNorm 的 scale parameter，这其实揭示了 **RMSNorm 的 scale parameter 在某种意义上扮演了一个 "default direction" 的角色**——任何与它 collinear 的 token 都会被 boost。这一点也呼应 *Bondarenko et al.* "Quantizable Transformers" https://arxiv.org/abs/2305.17688 对 outlier 维度的分析。

### Stage 4：Normalization 把 spike 转成 sparse + near-constant vector

这是连接 spike → sink 的关键桥梁。设 $\mathbf{h}^{(s)}$ 是 spike token 在 step-up 之后的 residual 表示，$\tilde{\mathbf{h}}^{(s)} = \mathrm{RMSNorm}(\mathbf{h}^{(s)})$。RMSNorm 给出 **三个性质**：

#### (1) Bounded Range（Theorem B.3）

$$|\tilde{\mathbf{h}}_i^{(s)}| \le \sqrt{d_{\text{model}}}, \quad \forall i \in \{1, \dots, d_{\text{model}}\} \tag{23}$$

证明极简：$\|\mathbf{h}\|_2^2 = \sum_j h_j^2 \ge h_i^2$，所以 $|h_i| / \|\mathbf{h}\|_2 \le 1$，乘上 $\sqrt{d_{\text{model}}}$ 即得上界。

**直觉**：即使 spike 之前 magnitude 几千几万，RMSNorm 之后**每个坐标都不超过 $\sqrt{d_{\text{model}}}$**（Llama-2 7B 是 64）。这让下游 block 重新获得数值稳定性——这就是为什么 spike 可以"漂"在网络里而不会爆炸。

#### (2) Sparsification

因为 norm 被少数 outlier channel 主导，归一化后**非 spike channel 被严重压制**：
$$\tilde{\mathbf{h}}^{(s)} \approx \sum_{i \in \mathcal{C}} \tilde{\mathbf{h}}_i^{(s)} \mathbf{e}_i \tag{24}$$

变量：$\mathcal{C}$ 是 spike channel 集合，$\mathbf{e}_i$ 是第 $i$ 个标准基向量。结果是一个 **近似 multi-hot 的稀疏向量**，活在 $|\mathcal{C}|$ 维子空间（远小于 $d_{\text{model}}$）。

#### (3) Near-constant vector

由性质 (iv)（不同 spike token 的 channel 间比例固定），归一化后值近乎 token-invariant：
$$\mathrm{RMSNorm}(\mathbf{h}^{(a)}) \approx \mathrm{RMSNorm}(\mathbf{h}^{(b)}) \tag{25}$$

即使 $\mathbf{h}^{(a)}, \mathbf{h}^{(b)}$ 在非 spike channel 上差异巨大。图 5 通过 cosine similarity 证实了这一点：**所有 spike token 经过 step-up block 后的归一化表示几乎重合**。

**直觉总结**：RMSNorm 把"一群数值上千的 diverse spike 表示"压成了"一个稀疏、bounded、几乎相同的向量"。这是后面 attention sink 形成的几何前提。

---

## 4. Attention Sink 的几何形成

### 4.1 Key vectors 限制在低维子空间

对 spike token，key 向量：
$$\mathbf{k}^{(s)} = \mathbf{W}_K^\top \tilde{\mathbf{h}}^{(s)} \approx \sum_{i \in \mathcal{C}} \tilde{\mathbf{h}}_i^{(s)} \mathbf{W}_K^\top \mathbf{e}_i \tag{26}$$

变量：$\mathbf{W}_K^\top \mathbf{e}_i$ 是 $\mathbf{W}_K$ 的第 $i$ 行。所以 spike token 的 key 被限制在 $\mathbf{W}_K$ 的少数几行张成的子空间——经验上通常 **只有 1-2 维**，远小于 $d_{\text{head}}$。

### 4.2 几何 alignment 决定 sink vs non-sink head

paper 把 attention head 分成两类（图 6 的 t-SNE 可视化）：

- **Sink head**：non-sink query 子空间 $\mathbf{q}^{(n)}$ **更靠近** sink key 子空间 $\mathbf{k}^{(s)}$，远离 non-sink key 子空间 $\mathbf{k}^{(n)}$。结果是 sink token 在 logits 上稳定占优，attention mass 系统性偏向 sink token。
- **Non-sink head**：$\mathbf{q}^{(n)}$ 与 $\mathbf{k}^{(n)}$ 距离更近，attention 按语义分布。

**直觉**：sink token 的 key 被"冻"在低维、近常数的位置，这给 model 提供了一个**永远可用的 default anchor**——任何一个 head 如果想"忽略远端 context"，只要把 query 投到 sink key 子空间附近，就能稳定地获得一个 large logit gap，从而把多余的 attention mass 卸到 sink token 上。

这个机制让人想到 *Goldie et al.* 关于 "superficial alignment" 的讨论，也联系到 *Vuckovic et al.* "The Geometry of Attention Sink" https://arxiv.org/abs/2402.07126——那里提出 sink 是 attention 的"轴"，因为它在 logits 上提供稳定的常数项。

延伸联想：sink token 的 key 在 1-2 维子空间里，几乎是 **codebook 中一个固定 vector**。这其实非常像 VQ-VAE 的 codebook entry——sink token 在某种意义上是 attention 分布的 "codebook anchor"。

---

## 5. 因果 ablation：spike 与 sink 可以解耦

paper §4 是一连串精心设计的 ablation，目的在于证明 spike 和 sink **功能独立、可以分别抑制**。我把所有表的数据梳理成"intuition map"。

### 5.1 Optimization hyperparameter（Table 3）

主要发现：
- **Sink ratio 是 optimization 健康的 proxy**——坏的 LR、关掉 weight decay、错误的 $\beta_2$ 都会拉低 sink ratio；延长训练、关掉 LR decay 会拉高。这跟 *Gu et al.* (ICLR 2025) 的观察一致。
- **Massive activation magnitude 与 perplexity / sink ratio 几乎独立**——比如关掉 weight decay 让 spike magnitude 飙到 12275，但 sink ratio 反而下降到 33.8%。

**直觉**：spike 的"量级"和 sink 的"是否形成"是两件事。一旦 spike 把归一化表示推到 sparse + near-constant regime，再让它变大几倍对 sink 的边际收益为零——RMSNorm 已经把 magnitude 全部抹掉了。

### 5.2 FFN 设计（Table 4）

替换 SwiGLU：
- GeLU FFN（标准 Transformer）→ spike=3369, sink=69.3%
- 单层 Linear → spike=688, sink=58.9%
- Attention-only（FFN 全部换成 attention layer）→ spike=637, sink=73.9%

**关键结论**：spike 和 sink **不依赖 SwiGLU**——它们在所有变体里都出现。SwiGLU/GeLU 只是把 spike 集中在**单步放大**，linear/attention-only 是**跨层累积**。

**直觉**：这暗示 spike/sink 是更深层（pre-norm + residual + softmax attention）的产物，FFN 只是一个高效 amplifier；任何能积累 large value 的 block 都能充当 step-up。

### 5.3 Normalization 配置（Table 5）——这是最关键的一张表

| 配置 | Perplexity | Sink ratio | Spike |
|---|---|---|---|
| Pre-norm (baseline) | 10.1 | 46.0% | 3818 |
| **Sandwich norm**（block 输出再加 RMSNorm） | 9.8 | 44.7% | **520** |
| **Sandwich + QKNorm**（只在 Q/K 上 norm） | 10.0 | 42.0% | **92** |
| **DynamicTanh**（element-wise tanh，无 norm） | 10.0 | **61.0%** | **153** |

**关键结论**：
- Sandwich norm 把 spike 砍到 1/7，sink ratio 几乎不变 → spike **不是** sink 的必要条件。
- QKNorm 几乎完全消除 spike，sink ratio 仅轻微下降 → 进一步坐实 spike 主要是用来 influence Q/K projection 的。
- DynamicTanh 完全消除 spike，但 sink ratio 反而**升高**到 61% → model 找到了**替代策略**让 first token 继续充当 anchor。

**直觉**：normalization 是连接 spike 和 sink 的"桥"，但桥是 **architectural** 的，不是 **functional** 的。模型想要的是"a stable anchor to dump attention mass"，spike 只是 pre-norm 下达成这个目标的一种手段；当 spike 路径被堵，模型会绕道。

延伸联想：DynamicTanh 来自 *Zhu et al.* "Transformers without Normalization" (CVPR 2025) https://arxiv.org/abs/2502.12222——element-wise $\tanh(\alpha x + \beta)$ 替代 LayerNorm/RMSNorm。它**无法**做 vector-wide norm，所以不能产生 sparse+near-constant vector，但 tanh 本身是 bounded 的，恰好对应了 "first token 作为 anchor" 的另一种实现方式。

### 5.4 Attention head 设置（Table 6）

固定总容量 $d_{\text{head}} \times N_{\text{head}}$，把容量集中到更少更大的 head：
- 8/512（小 dim 多 head）→ sink 11%
- 128/32（baseline）→ sink 46%
- 256/16（更大 dim 更少 head）→ sink 52%

固定 head 数变 dim：
- $d_{\text{head}}=8$ → sink 4.1%
- $d_{\text{head}}=128$ → sink 46%

**关键结论**：**head dimension 是 sink 形成的主导因子**，head 数量是边际收益。这印证了几何假说——更大的 head dim 给 sink key 与 non-sink key **留出几何可分的子空间**。

**直觉**：低维 head 就像低 rank codebook，sink 和 non-sink key "挤"在一起无法区分；高维 head 给了足够"空间"让它们分离。这跟 Johnson-Lindenstrauss 的几何直觉一致：高维空间里随机点天然可分。

### 5.5 Gated attention（Table 7）——sinks as implicit gate

条件 gating（gate 是当前 hidden 表示的函数）：
- per-channel gate → sink=4.5%, spike=202
- per-head gate → sink=6.4%, spike=186
- per-token 单标量 gate → sink=31.2%（不完全消除）

非条件 gating / 静态 gating（位置 / token embedding）→ sink 都保持在 40%+。

**关键结论**：只要给模型一个 **dynamic、input-dependent 的 gate**，sink 就会被完全消除，且不掉 perplexity。**Attention sink 本质上是一个 "learned implicit gate"**——模型在没有 explicit gate 时，用 sink token 来"关闭"那些它不想用的 head。

延伸联想：这个结论非常 deep。它说明 softmax attention 的一个核心痛点是 **head 没法"什么都不输出"**——softmax 的输出 sum=1 是 hard constraint。Gated attention（*Qiu et al.* NeurIPS 2025 https://arxiv.org/abs/2402.12361、*Yang et al.* "Gated Delta Networks" https://arxiv.org/abs/2411.12036）提供 multiplicative 路由让 head 可以"沉默"。Sink token 是 model 在缺乏 explicit gate 时**自发涌现的"沉默协议"**——把 attention mass 全部倒给一个无害 token，等于把这个 head 的 effective output 设为 sink token 的 V（通常是 BOS embedding 的某个变换）。

### 5.6 Training context length（Table 8）——sinks 服务 short-range

| Training range (min/max) | Perplexity | Sink ratio | Spike |
|---|---|---|---|
| 1/256 | 12.4 | 42.1% | 5411 |
| 1/4096 (baseline) | 10.1 | 46.0% | 3818 |
| **1024/4096**（去掉 short context） | 10.1 | **13.0%** | 38470 |
| **2048/4096** | 10.6 | **1.2%** | 7193 |

**关键结论**：训练时只看长序列 → sink 几乎消失，spike 反而暴涨。这印证 *Xiao et al.* DuoAttention 的假说：**sink 主要服务于 short-range dependence**。当训练分布不需要"短上下文预测"时，model 不再需要把 attention mass 倒给 sink token 来屏蔽远端 context。

**直觉**：global attention 在长 context 时被 useless 的远端 token 污染，sink 提供一种"低代价 escape"——把 mass 卸到 first token，等于 effectively 把 head 的 effective attention 局部化。这也是 StreamingLLM 的核心 trick：保留 sink token 的 KV，长 context 性能不掉。

---

## 6. 整体 anatomy 的因果链总结

把所有发现串起来，得到如下因果链：

```
[pre-norm + residual + SwiGLU FFN + softmax attention]
              │
              ▼
  step-up FFN block 的 quadratic form S_k 是 rank-one dominant
  且所有 spike channel 共享 trigger direction s_*
              │
              ▼
  first token attention 退化为线性 W_VO → 把 h^(1) 推向 s_*
              │
              ▼
  spike token 在 step-up block 被 quadratic amplifier 放大 (λ_* (s_*^T h)^2)
              │
              ▼
  residual accumulation：spike 漂过整个网络
              │
              ▼
  step-down block 注入相反符号 extreme value 抵消
              │
              ▼
  RMSNorm 在每个 block 入口把 spike 压成 [bounded + sparse + near-constant] vector
              │
              ▼
  spike token 的 key = W_K^T h̃^(s) 被限制在 1-2 维子空间
              │
              ▼
  head 的 query 子空间几何 align 决定是否是 sink head
              │
              ▼
  sink head 系统性把 attention mass 卸给 sink token
  → 作为 implicit gate 关闭它不想要的 head
  → 服务 short-range dependence
```

而所有环节中，**normalization 是唯一 spike→sink 的桥梁**。换掉 normalization（DynamicTanh）→ spike 死，sink 通过另一条路径活下来。引入 explicit gate → sink 不再被需要。换训练分布只看长 context → sink 失去功能。

---

## 7. 我对 paper 的几条延伸联想（build intuition）

### 7.1 Spikes 作为 implicit parameters

paper 反复强调一个 framing：**massive activations 在 forward pass 中作为 implicit parameters 存在**。它们是 input-agnostic 的常数项（magnitude 跨 prompt 几乎不变），通过 residual stream 加到每个 block 的 hidden 上。这意味着模型把"全局偏置"编码成了一个**非参数化的运行时 memory**，而不是 absorb 进 weight。

延伸：这跟 *Aghajanyan et al.* 关于"intrinsic dimension"的工作、*Skean et al.* "Layer by Layer" https://arxiv.org/abs/2502.12320 提到的 "compression valley" 都形成呼应——LLM 的 representation 在中间层被"压"进低维流形，spike 就是这个流形的 anchor。

### 7.2 Sink 作为 softmax 的 "OFF state"

softmax attention 的硬约束是 $\sum_k A_{tk} = 1$，所以一个 head 想要 "什么都不 attend"，必须把 mass 倒给一个**信息上无害的 token**。Sink token 就是这个 "OFF state"。这跟 *Velickovic et al.* "Softmax is not Enough" https://arxiv.org/abs/2410.01104、*Miller* "Attention is Off by One" https://www.evanmiller.org/attention-is-off-by-one.html 的批评一脉相承——softmax 缺一个显式的 "null" key，模型被迫用一个 token 来代表 null。

延伸：这也解释了为什么 sigmoid attention、ReLU attention、softmax-off-by-one 这些替代激活函数都能减少 sink——它们都给了模型一个**显式 null** 选项。

### 7.3 Spike 与 quantization 的灾难性耦合

spike magnitude 几千几万，常规 FP16 / INT8 quantization 直接爆掉。相关 prior work：
- *Wei et al.* "Outlier Suppression+" https://arxiv.org/abs/2304.09145
- *Xi et al.* "Training Transformers with 4-bit Integers" https://arxiv.org/abs/2306.02072  
- *Liang et al.* "TWEO: Transformers without Extreme Outliers" https://arxiv.org/abs/2511.23225

paper 的发现指出 spike 是 architectural artifact——这给 mitigation 提供了直接路径：**改 normalization**就能消除 spike 而**不掉 perplexity**。这是非常 actionable 的工程结论。

### 7.4 Sink 与 KV cache eviction

StreamingLLM 等长 context 推理框架的核心 trick 是**保留 sink token 的 KV**。paper 解释了为什么这 work：sink token 是"OFF state"，model 依赖它做 short-range bias。一旦 evict sink token，model 就失去了 implicit gate，effective attention 被强制 spread 到所有远端 token，性能崩。

延伸：*Ge et al.* "Adaptive KV Cache Compression" https://arxiv.org/abs/2311.01481、*Su & Yuan* "KVSink" https://arxiv.org/abs/2503.04773、*Wu & Tu* "Layer-Condensed KV Cache" https://aclanthology.org/2024.acl-long.34/ 都围绕 sink-aware KV cache 设计。

### 7.5 "Mix-Compress-Refine" 假说

*Queipo-de Llano et al.* (2025) 提出 LLM 中间层存在 "compression valley"——representation 维度先压缩再扩展。Spike 正好对应 compress 阶段的 anchor，sink 则是 attention 维度的对应物。本文的发现（normalization 是桥梁）跟这个理论**互为佐证**：normalization 既 spike-ify 又 sparse-ify，正好是"压缩"的代数形式。

### 7.6 与 vision Transformer 的 register token 对照

*Darcet et al.* "Vision Transformers Need Registers" https://arxiv.org/abs/2305.12981 发现 ViT 也有 sink-like token（register token）——人为添加的额外 token 充当 attention sink，避免信息被 dump 到语义 token 上。本文的发现（sink 是 implicit gate + first token 提供 OFF state）完美对应——L LM 没有显式 register，所以 model "劫持"了 first token 作为 register。

延伸：这说明未来 LLM 也许应该**显式加入 register token**——既能让 spike 不再劫持 first token（让 first token 回归语义角色），又能让 sink 更稳定。

### 7.7 关于"功能耦合"的更广 lessons

paper 最有方法论价值的部分是它**严格区分 mechanistic correlation 和 causal coupling**。这给 interpretability 研究立了一个范式：

- 很多看似深耦合的现象，其实是 **architectural 共因**导致的共现；
- 要证明 causal 耦合，必须能找到一个**只抑制其中一个、另一个不变**的干预；
- 不能就回到 "X 是 Y 的 cause" 这种 narrative。

这跟 *Sandoval-Segura et al.* "Identifying and Evaluating Inactive Heads in Pretrained LLMs" https://arxiv.org/abs/2406.11930 的方法学类似——通过 ablation 区分 "dormant" head 与 "active" head。

---

## 8. 局限与我会想追问 paper 作者的问题

1. **step-up/step-down block 的位置是 emergent 还是 designed？** paper 没有给出 step-up block **为什么**出现在 layer 4（而不是 layer 10）的解释。可能是初始化 + 训练动力学共同决定，但需要 probing 实验。
2. **shared trigger direction $\mathbf{s}_\star$ 的语义**是什么？paper 没分析 $\mathbf{s}_\star$ 在 vocab / unembedding 空间对应什么。这可能是"register / null / OFF"概念的几何对应物。
3. **per-channel conditional gate 完全消除 sink** 是否意味着 model 找到了**另一套 implicit parameters**？如果是，这套 implicit parameters 在哪？是否带来 quantization 的新挑战？
4. **DynamicTanh 61% 的 sink ratio 是怎么实现的？** paper 说 "alternative strategy" 但没拆解。猜测是 tanh 在 first token 上 satur到极值，使得 key 几乎常数——但需要实验验证。
5. **rank-one dominance 是怎么训练出来的？** 是初始化就 rank-one 偏好，还是 gradient 把它推向 rank-one？这关系到能不能在训练早期 detect / prevent。

---

## 9. 实操 takeaway（如果我要训一个 7B）

基于这篇 paper 的 ablation，如果我今天要从 scratch 训 Llama-style 7B，下面几条是 actionable 的：

1. **Sandwich norm**（block 输出再加一个 RMSNorm）几乎免费消除 spike，sink 不掉，perplexity 还略好（9.8 vs 10.1）。**应当成为新 default**。
2. **QKNorm**（Q/K 上额外 RMSNorm）几乎完全消除 spike，sink 几乎不变。对 quantization 友好，几乎零代价。
3. **Gated attention**（per-channel 或 per-head）彻底消除 sink，掉 0.1 量级 perplexity。如果想优化 KV cache / 长上下文，这是更激进的选项。
4. **DynamicTanh** 完全消除 spike 且 sink 反而升高——如果想要"无 outlier 但 sink 保留"的架构，这是 interesting 选项。
5. **Head dimension 至少 128**，否则 sink 形不成、perplexity 受损。
6. **训练 context 分布**：如果只关心长 context 应用，可以直接训长 sequence，让 sink 自然退化，节省 KV cache。

参考 *Olmo 3* https://arxiv.org/abs/2512.13961、*torchtitan* https://arxiv.org/abs/2502.01383 都已经开始吸纳 QKNorm 等设计，paper 这一系列 ablation 为下一代 LLM 架构提供了非常 actionable 的 evidence base。

---

## 10. 进一步阅读路线图

如果想继续深挖：

**Massive activations 序列：**
- *Dettmers et al.* LLM.int8() https://arxiv.org/abs/2208.07339
- *Bondarenko et al.* Understanding Quantization Challenges https://arxiv.org/abs/2110.14524
- *Sun et al.* Massive Activations in LLMs https://arxiv.org/abs/2407.14411
- *Yu et al.* The Super Weight in LLMs https://arxiv.org/abs/2411.07191
- *Oh et al.* House of Cards: Massive Weights https://arxiv.org/abs/2410.01866
- *Owen et al.* A Refined Analysis of Massive Activations https://arxiv.org/abs/2503.22329

**Attention sinks 序列：**
- *Xiao et al.* StreamingLLM https://arxiv.org/abs/2309.17453
- *Xiao et al.* DuoAttention https://arxiv.org/abs/2410.10819
- *Gu et al.* When Attention Sink Emerges in LMs https://arxiv.org/abs/2410.10773
- *Guo et al.* Active-Dormant Attention Heads https://arxiv.org/abs/2410.13835
- *Vuckovic et al.* Geometry of Attention Sink https://arxiv.org/abs/2402.07126
- *Velickovic et al.* Softmax is not Enough https://arxiv.org/abs/2410.01104

**Mitigation 序列：**
- *Qiu et al.* Gated Attention https://arxiv.org/abs/2402.12361
- *Zhu et al.* Transformers without Normalization (DynamicTanh) https://arxiv.org/abs/2502.12222
- *He et al.* Understanding and Minimising Outlier Features https://arxiv.org/abs/2402.05112
- *Liang et al.* TWEO https://arxiv.org/abs/2511.23225

**理论：**
- *Queipo-de Llano et al.* Attention Sinks and Compression Valleys https://arxiv.org/abs/2510.06477
- *Skean et al.* Layer by Layer https://arxiv.org/abs/2502.12320
- *Kaul et al.* From Attention to Activation https://arxiv.org/abs/2410.17174

---

## 一句话总结直觉

**Massive activations 是 pre-norm + residual + SwiGLU 三者共同产生的 quadratic amplifier 副产物——把 first token 的 attention-on-self 退化成的线性 boost，经过 step-up FFN 的 rank-one 二次型放大成几个 channel 上的极端 outlier；residual 把这些 outlier 漂过网络；RMSNorm 在每个 block 入口把它们压成 sparse + near-constant vector，给 attention head 提供一个稳定的"低维 key anchor"；head 的 query 几何 alignment 决定它是否是 sink head——sink head 把 softmax 的 OFF 状态卸给 sink token，作为 implicit gate 服务 short-range dependence。Normalization 是 spike→sink 的唯一桥梁，但桥是 architectural 的而非 functional 的——换 norm、加 explicit gate、改训练分布都能独立解耦两者。**

希望这个讲解帮 Karpathy 把 paper 的全貌在直觉上落地。如果想要再深入哪个 sub-claim，比如想看 $\mathbf{s}_\star$ 的实际向量分析、step-up block 的 weight 可视化、或者 sink head 的 attention pattern 解剖，告诉我哪一块，我再展开。
