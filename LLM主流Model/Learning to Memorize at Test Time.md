---
source_pdf: Learning to Memorize at Test Time.pdf
paper_sha256: a65e4a7d02784df1a040b487127e6dd09fff4474e5caf94d93263af3d50cfbc2
processed_at: '2026-08-05T13:52:36-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Titans

## 一句话版本

让 attention 只管眼前的事，另起一个小 brain 专门在 test time 边读边学边忘，记住很久以前发生了啥。

---

## 问题在哪

Transformer 的 attention 很强，它能精确建模序列里每个 token 和其他所有 token 的关系。但代价是 quadratic 的 —— 序列越长越慢，到 100K token 以上基本玩不动。

所以大家搞了一堆 linear RNN（Mamba, GLA, DeltaNet 等），把历史信息压进一个固定大小的 matrix state，推理快、训练也能并行。但问题是：**一个 matrix 能存多少东西？** 序列到百万级 token 时，要么信息 overflow，要么旧信息被新信息冲掉。

paper 里有个很 sharp 的观察：

> 我们用 linear model 是为了 scale 到长 context，但长 context 恰恰是 linear model 最扛不住的场景，因为固定大小 state 装不下那么多信息。

这就是矛盾点。

---

## Titans 的 insight

换思路：**别把 memory 当成一个 state 往里塞信息，把 memory 当成一个 neural network 的权重。**

为啥这样更好？因为：

1. 一个 MLP 有多层、有非线性，表达能力远超一个 matrix
2. 更新权重就是 gradient descent，读取就是 forward pass，很自然
3. 最关键 —— **这个 network 在 test time 还在学**。不是训练完就冻住，是推理过程中边推理边更新自己的权重

这就有点像你在读一篇很长的文章，读到新东西时你的 brain 不是把信息塞进一个固定容量的 buffer，而是在神经连接层面持续微调。

---

## 怎么决定记什么

用 gradient 衡量"惊讶程度"。

如果当前 token 让 memory network 的 loss gradient 很大，说明这个 token 很"出乎意料"，值得记住。gradient 小的 token 就少记一点。

这就是公式 $\mathcal{M}_t = \mathcal{M}_{t-1} - \theta_t \nabla \ell(\mathcal{M}_{t-1}; x_t)$ 的意思 —— 往惊讶的反方向更新 memory。

但单步 gradient 有个问题：如果一个 token 特别惊讶，memory 会被大力更新，之后可能掉进 flat region，后续 token 的 gradient 都变小，信息就记不进去了。

---

## Momentum：记住"惊讶的余波"

paper 的核心创新：把 surprise 拆成两部分。

- **momentary surprise**：当前 token 的 gradient
- **past surprise**：之前几步累积下来的惊讶度

用 momentum 把两者加起来：
$$S_t = \eta_t S_{t-1} - \theta_t \nabla \ell(\mathcal{M}_{t-1}; x_t)$$

直觉就是：**一个大惊讶事件之后，接下来一段时间内的信息即使不那么惊讶，也值得记住，因为这个事件本身赋予了后续 context 重要性。**

想想你看电影：一个 plot twist 发生后，接下来几分钟的对话即使平淡，你也记得很清楚，因为 twist 给了那段 context 你的注意力。

$\eta_t$ 是 data-dependent 的 —— model 自己学什么时候该保留过去的惊讶（$\eta \to 1$），什么时候该清掉（$\eta \to 0$，比如 topic 切换了）。

---

## Weight decay = 遗忘门

百万级 token 的序列，光记不忘肯定 overflow。所以加了 forgetting：

$$\mathcal{M}_t = (1 - \alpha_t) \mathcal{M}_{t-1} + S_t$$

$\alpha_t$ 控制遗忘强度。$\alpha \to 1$ 就是把 memory 清空，$\alpha \to 0$ 就是完全保留。

paper 指出这恰好等价于 **gradient descent with weight decay**，weight decay 系数就是 $\alpha_t$。这个等价很重要，因为它让 forgetting 可以用优化的语言来 parallelize —— 你不需要发明新的并行算法，复用 mini-batch gradient descent 的 chunk-wise 展开就行。

---

## Loss 在干嘛

memory network 学的是 key-value association：

$$\ell = \|\mathcal{M}(\mathbf{k}_t) - \mathbf{v}_t\|^2$$

就是把 $x_t$ 投影成 key 和 value，让 memory network 学会 $\mathbf{k} \to \mathbf{v}$ 的映射。和 Transformer 的 attention 存储 K-V association 是一回事，但这里存在 network 权重里，存在一个深度非线性的 function 里。

---

## 三个模块的分工

Titans 架构有三个 branch：

**1. Core（attention）—— short-term memory**

就是 sliding window attention 或 chunked attention。只看当前 window 内的 token，精确建模局部依赖。窗口外的它不管。

**2. Long-term Memory（neural memory）—— long-term memory**

就是上面说的那个 test-time learning 的 MLP。负责记住很久以前的信息。推理时权重一直在更新。

**3. Persistent Memory —— task knowledge**

一组 learnable 但 data-independent 的参数，prepend 到序列开头。存的是"这个任务一般怎么做"的先验知识。test time 固定不变。

对应人类认知：
- attention ≈ working memory（当前注意的内容）
- neural memory ≈ long-term memory（过去经历的压缩）
- persistent memory ≈ procedural memory（技能/常识）

---

## 三种架法

paper 提了三种把 memory 接进 backbone 的方式：

### MAC（Memory as Context）

最巧妙的设计。把 memory 检索出来的历史信息当作额外的 context token，和当前 input 一起喂给 attention。

```
[persistent tokens] + [memory retrieval tokens] + [current input] → Attention
```

好处是 **双向过滤**：
- attention 能看到历史信息，自己决定哪些有用
- attention 的输出反过来决定哪些当前信息该写入 memory

这种 bidirectional 协同让 MAC 在 long-context retrieval 任务上碾压其他变体。

### MAG（Memory as Gate）

两个 parallel branch：
- SWA 处理当前窗口（short-term）
- neural memory 处理整段历史（long-term）
- 两者通过 gate 融合

像一个双 head attention，但两个 head 结构完全不同。

### MAL（Memory as Layer）

最朴素：memory layer → attention layer，顺序堆叠。类似 H3 架构把 SSM 换成 neural memory。效果略差于前两者，因为 memory 压缩后的信息丢失 attention 没法补救。

---

## 怎么并行训练

这是工程上最 tricky 的部分。test-time training 听起来是序列化的 —— 每个 token 都要更新权重，怎么并行？

paper 的做法：

1. 把序列切成 chunks of size $b$
2. chunk 内，用 mini-batch gradient descent 的闭式展开，所有 gradient 可以一次性用 matmul 算出来
3. momentum term 是一个 linear recurrence，用 **parallel associative scan** 在 $O(\log b)$ 深度内并行算
4. chunk 之间是序列化的，但 chunk 数量 $N/b$ 远小于 $N$

所以总复杂度是 $O(N)$ 的，但能利用 GPU 的并行能力。和 Mamba2 的 parallel scan 思路类似。

---

## 实验里最炸的结果

### Needle in Haystack（Table 2）

在 16K context 找一个 word needle：

| Model | 16K acc |
|-------|---------|
| TTT | 0.0 |
| Mamba2 | 0.0 |
| DeltaNet | 0.0 |
| **Titans (MAC)** | **95.2** |

所有 baselines 在 16K 时全崩了，Titans 还在 95%+。原因就是 momentum + weight decay + deep memory 三件套让 memory 不会在长序列下 overflow。

### BABILong（Figure 6）

比 NIAH 更难 —— 要在极长文档里跨多个事实推理。

Titans (MAC) 760M 参数，few-shot 超过 GPT-4、Llama3.1-8B。fine-tuned 版本超过 GPT-4、Qwen2.5-72B、Llama3.1-70B。

**用 1/70 的参数打败 70B 模型**，这说明对于长 context 任务，结构化 memory 比 brute-force scale 更 effective。

### Memory Depth（Figure 7）

memory MLP 从 1 层加到 4 层，perplexity 持续下降，且对 sequence length 更鲁棒。这验证了 deep non-linear memory 比 matrix-valued linear memory 强 —— 呼应 universal approximation theorem。

### Ablation（Table 5）

去掉各 component 后 long context acc 的下降：
- 去掉 deep memory：-7.34
- 去掉 weight decay：-7.08
- 去掉 momentum：-5.56
- 去掉 convolution：-2.40
- 去掉 persistent memory：-0.19

**deep memory 和 weight decay 最关键**，momentum 第三。persistent memory 在长 context 任务上贡献小但在 language modeling 上更大。

---

## 和现有 model 的关系

paper 在 Appendix C 给了很清晰的关系：

- **LMM ⊃ Gated DeltaNet**：设 $\eta_t = 0$（去掉 momentum），LMM 就退化为 Gated DeltaNet。LMM 多了 momentum + deep memory + non-linear recurrence。
- **LMM ⊃ Longhorn**：类似，但 LMM 多了 forget gate。
- **LMM ⊃ TTT**：TTT 是唯一另一个 gradient-based RNN，但没有 forget mechanism、没有 momentum、没验证 deep memory。

所以 Titans 的 neural memory module 可以看作 **modern linear RNN 的下一代** —— 把 momentum、forgetting、depth 三个要素都加上了。

---

## 我觉得最重要的 takeaway

1. **Memory 不是 state，memory 是 weights。** 把信息存进 network 参数比存进 matrix expressive 得多。

2. **Surprise = gradient。** 这是一个很美的 cross-discipline 对应，把心理学 surprise theory 和优化理论联系起来了。

3. **Momentum = surprise 的 temporal credit assignment。** 一个惊讶事件给后续一段 context 赋予记忆权重，这比单步 gradient 更符合人类记忆行为。

4. **Weight decay = forgetting gate。** 这个等价让 forgetting 可以用优化器语言 parallelize，不用发明新算法。

5. **Test-time learning 是被低估的方向。** 当前 LLM 的 inference 是 frozen 的，但人类读长文时 brain 是在持续更新的。Titans 证明了即使很小的模型（760M），只要有 test-time learning memory，在长 context 上能超越大几十倍的 frozen model。

6. **MAC 的 bidirectional 设计值得学。** attention 看 memory、memory 被 attention 过滤，这种协同比简单拼接或串行堆叠都强。

---

## 相关阅读

- TTT layers: https://arxiv.org/abs/2407.04620
- Mamba2 (structured state space duality): https://arxiv.org/abs/2405.21060
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Longhorn: https://arxiv.org/abs/2407.14207
- DeltaNet: https://arxiv.org/abs/2407.06404
- Parallel associative scan (S5): https://arxiv.org/abs/2312.00752
- RULER benchmark: https://arxiv.org/abs/2404.06654
- BABILong: https://arxiv.org/abs/2406.04268
- Hymba (hybrid head): https://arxiv.org/abs/2411.13676
- Sukhbaatar persistent memory (FFN as attention): https://arxiv.org/abs/1907.01470
- Attention sinks: https://arxiv.org/abs/2309.17453
- Merrill TC$^0$ expressiveness: https://arxiv.org/abs/2404.02805
- Birth of a Transformer (attention as associative memory): https://arxiv.org/abs/2406.04028
- Fast weight programmers (Schmidhuber): https://arxiv.org/abs/2102.11174
- Mamba original: https://arxiv.org/abs/2312.00752
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- H3 (Hungry Hungry Hippos): https://arxiv.org/abs/2212.14052
- Memory Mosaics: https://arxiv.org/abs/2405.06394
- Cowan memory systems: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2645024/

---

# Titans: Learning to Memorize at Test Time 深度解析

paper link: https://arxiv.org/abs/2501.00663

## 1. 高层 intuition

这篇 paper 的核心 thesis 非常清晰：**把 attention 看作 short-term memory，把一个在 test time 仍在学习的 neural network 看作 long-term memory**，两者协同工作。这个 framing 解决了当前 sequence modeling 的一个根本矛盾：

- Transformers 的 attention 精确建模所有 token 依赖，但 quadratic 复杂度限制了 context length；
- Linear RNNs / SSMs（Mamba, GLA, DeltaNet 等）压缩历史到 fixed-size matrix state，效率高，但长 context 下信息会 overflow。

Titans 的 insight 是：**不要在 attention 内部找长 context 的出路，而是引入一个独立的、深度结构化的、在 inference 时仍在更新自身权重的 memory module**，让它承担"记住很久以前发生了什么"的职责，而 attention 只负责当前 window 内的精确依赖建模。

参考相关工作：
- TTT layers: https://arxiv.org/abs/2407.04620
- Mamba2: https://arxiv.org/abs/2405.21060
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Longhorn: https://arxiv.org/abs/2407.14207

---

## 2. Neural Long-term Memory Module 的设计

这是 paper 的技术核心。我把它拆成四个 component 讲。

### 2.1 Memory perspective 下的 RNN 重定义

paper 先给了一个统一的视角。任何 recurrent model 都可以写成：

$$\mathcal{M}_t = f(\mathcal{M}_{t-1}, x_t) \quad \text{(Write Operation)}$$
$$y_t = g(\mathcal{M}_t, x_t) \quad \text{(Read Operation)}$$

变量解释：
- $\mathcal{M}_t \in \mathbb{R}^{d}$：时刻 $t$ 的 memory state（RNN 是 vector-valued，linear Transformer 是 matrix-valued $\mathbb{R}^{d \times d}$）
- $x_t$：时刻 $t$ 的输入 token
- $f, g$：write / read 函数

这个视角下，linear Transformer 的 recurrence（Equation 4-5）：
$$\mathcal{M}_t = \mathcal{M}_{t-1} + K_t^\top V_t, \quad y_t = Q_t \mathcal{M}_t$$

是 additively 把 $(K_t, V_t)$ 压进 matrix memory。问题在于 additive 写入没有遗忘机制，长序列下 memory overflow。

### 2.2 Surprise metric：把 gradient 当作"惊讶度"

paper 借鉴心理学：**让人惊讶的事件更容易记住**（Mandler 2014）。在 neural network 里，"惊讶"的自然度量就是 loss 对 input 的 gradient —— gradient 越大，说明当前 input 越偏离 model 的预期。

最朴素的 memory update rule（Equation 8）：

$$\mathcal{M}_t = \mathcal{M}_{t-1} - \theta_t \underbrace{\nabla \ell(\mathcal{M}_{t-1}; x_t)}_{\text{Surprise}}$$

变量解释：
- $\mathcal{M}_t$：memory module 在时刻 $t$ 的参数（注意：这里是 **参数本身** 作为 memory，不是 hidden state）
- $\theta_t$：data-dependent learning rate，控制 momentary surprise 的写入强度
- $\ell(\mathcal{M}_{t-1}; x_t)$：associative memory loss（后面定义）
- $\nabla$：对 $\mathcal{M}$ 参数求梯度

**关键问题**：单一 gradient surprise 会在大惊讶后陷入 flat region（local minima），导致后续 token 的 gradient 极小，信息丢失。这对应人类记忆的一个现象：一个事件可能一开始很惊讶，但之后即使不再惊讶，我们仍然持续记住整段时间发生的事。

### 2.3 Momentum surprise：拆成 past + momentary

paper 的核心改进（Equation 9-10）：

$$\mathcal{M}_t = \mathcal{M}_{t-1} + S_t$$
$$S_t = \eta_t \underbrace{S_{t-1}}_{\text{Past Surprise}} - \theta_t \underbrace{\nabla \ell(\mathcal{M}_{t-1}; x_t)}_{\text{Momentary Surprise}}$$

变量解释：
- $S_t$：**surprise 的 momentum**，跨时间步累积的惊讶度
- $\eta_t$：data-dependent surprise decay，控制 past surprise 衰减速度
  - $\eta_t \to 0$：忽略过去的惊讶（可能 context 切换）
  - $\eta_t \to 1$：完全保留过去惊讶（token 与近期历史强相关）
- $\theta_t$：data-dependent learning rate

这形式上就是 **gradient descent with momentum**，$S_t$ 是 momentum term。momentum 在这里扮演"惊讶的跨时间记忆"。

这个设计是 Titans 相对于 TTT、DeltaNet、Gated DeltaNet 的关键差异 —— 后者都只用 momentary surprise（单步 gradient），丢失了 token flow 信息。

### 2.4 Forgetting via weight decay

处理超长序列（百万级 token）时，必须有遗忘机制（Equation 13-14）：

$$\mathcal{M}_t = (1 - \alpha_t) \mathcal{M}_{t-1} + S_t$$
$$S_t = \eta_t S_{t-1} - \theta_t \nabla \ell(\mathcal{M}_{t-1}; x_t)$$

变量解释：
- $\alpha_t \in [0, 1]$：**forgetting gate**，data-dependent
  - $\alpha_t \to 0$：完全保留过去 memory
  - $\alpha_t \to 1$：清除整个 memory（context 彻底切换时有用）

paper 指出这等价于 **gradient descent with weight decay**，weight decay 系数就是 $\alpha_t$。这个等价性很重要，因为它把"遗忘门"这个 RNN 概念和"weight decay"这个优化器概念统一了起来，从而可以用优化的语言来 parallelize。

### 2.5 Associative memory loss

memory 学习的目标是 key-value association（Equation 11-12）：

$$\mathbf{k}_t = x_t W_K, \quad \mathbf{v}_t = x_t W_V$$
$$\ell(\mathcal{M}_{t-1}; x_t) = \|\mathcal{M}_{t-1}(\mathbf{k}_t) - \mathbf{v}_t\|_2^2$$

变量解释：
- $W_K, W_V \in \mathbb{R}^{d_{in} \times d_{in}}$：key/value projection matrices，是 **hyperparameters**（outer loop 优化，inner loop 固定）
- $\mathcal{M}_{t-1}(\mathbf{k}_t)$：memory module（一个 MLP）对 key $\mathbf{k}_t$ 的 forward pass
- 目标：让 memory 学会 $\mathbf{k}_t \to \mathbf{v}_t$ 的映射

这是 **meta-learning** 的 setup：
- **Inner loop**：在 forward pass 中，用 gradient descent 更新 $\mathcal{M}$ 的权重（这是 test time training）
- **Outer loop**：训练 $W_K, W_V, W_Q$ 以及 attention 等其他参数

memory module 本身是一个 $L_\mathcal{M} \geq 1$ 层的 MLP。paper 在 Section 5.5 实验验证 $L_\mathcal{M} = 2, 3, 4$ 比 $L_\mathcal{M} = 1$（linear memory）显著更好，这呼应了 Hornik et al. 1989 的 universal approximation 理论：两层 MLP 严格比线性模型 expressive。

### 2.6 Retrieval

memory 读取就是 forward pass without weight update（Equation 15）：

$$y_t = \mathcal{M}^*(\mathbf{q}_t), \quad \mathbf{q}_t = x_t W_Q$$

- $\mathcal{M}^*$：表示 forward pass **不更新权重**（inference mode）
- $\mathbf{q}_t$：query projection

这里有个微妙的设计：**同一个 $\mathcal{M}$ 既是 writer（gradient 更新权重）又是 reader（forward pass 检索）**。这和传统 RNN 的 hidden state 读写分离不同 —— Titans 的 memory 是参数化的、深度的、非线性的。

---

## 3. Parallelization：如何用 matmul 训练

这是工程上最难的部分。naive 的 test time training 是 $O(N)$ 序列化的，无法利用 GPU/TPU 的并行能力。

### 3.1 Chunk-wise gradient descent

把长度 $N$ 的序列切成 chunks of size $b$。在 chunk 内，可以用 mini-batch gradient descent 的闭式展开（Equation 16）：

$$\mathcal{M}_t = \beta_t \mathcal{M}_0 - \sum_{i=1}^{t} \theta_i \frac{\beta_t}{\beta_i} \nabla \ell(\mathcal{M}_{t'}; x_i)$$

变量解释：
- $t' = t - \text{mod}(t, b)$：当前 chunk 的起始时间步（chunk 内所有 step 共享同一个 $\mathcal{M}_{t'}$ 作为起点）
- $\beta_i = \prod_{j=1}^{i} (1 - \alpha_j)$：**累积 weight decay**，从开始到时刻 $i$ 的衰减乘积
- $\frac{\beta_t}{\beta_i}$：从时刻 $i$ 到时刻 $t$ 的衰减比例

关键洞察：这个求和可以 tensorized 成 matmul。

### 3.2 Linear memory 的 matmul 形式

当 $\mathcal{M}_t = W_t$ 是线性时（Equation 17）：

$$\sum_{i=1}^{b} \theta_i \frac{\beta_b}{\beta_i} \nabla \ell(W_0; x_i) = \Theta_b \mathbf{B}_b (W_0 X - X) X^\top$$

变量解释：
- $\Theta_b = \text{diag}([\theta_1, \theta_2, ..., \theta_b])$：chunk 内所有 learning rate 组成的对角矩阵
- $\mathbf{B}_b$：类似定义在 $\frac{\beta_b}{\beta_i}$ 上的对角矩阵
- $X = [x_1, ..., x_b]^\top$：chunk 内所有 input 堆叠
- $W_0 X - X$：预测误差（假设 $V = X$ 的简化情形）

这样 chunk 内的所有 gradient 可以一次性 matmul 算出来。

### 3.3 Momentum 的 parallel associative scan

momentum term（Equation 18）：

$$S_t = \eta_t S_{t-1} - \theta_t u_t, \quad u_t = \nabla \ell(\mathcal{M}_{t'}; x_t)$$

这是一个 **linear recurrence**：
- 输入：$u_t$（可以并行计算所有 chunk 内的 $u_t$）
- hidden state：$S_t$
- transition：$\eta_t$（input-dependent）

可以用 **parallel associative scan**（Smith et al. 2023, 参考 https://arxiv.org/abs/2312.00752）在 $O(\log b)$ 深度内并行计算整个 chunk 的 $S_t$。

### 3.4 Chunk-level 参数简化

如果 $\alpha, \theta, \eta$ 在 chunk 内是 time-invariant（只依赖 chunk 而非 token），那么 recurrence 变成 LTI（Linear Time-Invariant）系统，可以用 **global convolution** 计算（参考 Mamba/S4 的 structured state space duality: https://arxiv.org/abs/2405.21060）。paper 在主实验中用 token-dependent 参数，但指出 chunk-dependent 是 future work 的效率方向。

---

## 4. Persistent Memory：task-level 的可学习先验

除了 contextual long-term memory，paper 还引入了 **persistent memory**（Equation 19）：

$$x_{\text{new}} = [p_1, p_2, ..., p_{N_p}] \,||\, x$$

变量解释：
- $p_i$：learnable but **data-independent** parameters
- $N_p$：persistent memory token 数量
- $||$：sequence concatenation

这些 token prepend 到序列开头，不依赖 input。三个动机：

1. **Memory perspective**：task knowledge（"这个任务怎么做"）应该存在 input-independent 参数里，类似程序员的 procedural memory。

2. **FFN perspective**：Sukhbaatar et al. 2019（https://arxiv.org/abs/1907.01470）发现 Transformer 的 FFN 层等价于 input-independent 的 attention：
$$\text{FFN}(x) = W_V \text{Softmax}(W_K x)$$
persistent memory 起到类似作用。

3. **Technical perspective**：causal attention 对序列开头 token 有 implicit bias（attention sink，参考 https://arxiv.org/abs/2309.17453），persistent token 可以吸收这种 bias，让 attention 权重更均匀分布。

在 test time，persistent memory 参数**固定不变**（task knowledge 不应该被 test 数据改写），而 long-term memory 参数**仍在更新**。

---

## 5. 三种 Titans 架构

这是 paper 的架构设计部分。三种变体对应三种把 memory 融入 backbone 的方式。

### 5.1 MAC: Memory as Context

最 expressive 的设计。流程（Equation 21-25）：

1. 用当前 segment 作为 query，从 long-term memory 检索历史信息：
$$h_t = \mathcal{M}_{t-1}^*(\mathbf{q}_t), \quad \mathbf{q}_t = \mathsf{S}^{(t)} W_Q$$

2. 拼接 persistent memory + 历史检索 + 当前 segment：
$$\tilde{\mathsf{S}}^{(t)} = [p_1, ..., p_{N_p}] \,||\, h_t \,||\, \mathsf{S}^{(t)}$$

3. Attention 处理整个拼接序列：
$$y_t = \text{Attn}(\tilde{\mathsf{S}}^{(t)})$$

4. 用 attention 输出更新 long-term memory 并生成最终输出：
$$\mathcal{M}_t = \mathcal{M}_{t-1}(y_t)$$
$$o_t = y_t \otimes \mathcal{M}_t^*(y_t)$$

变量解释：
- $\mathsf{S}^{(t)}$：第 $t$ 个 segment
- $h_t$：从 memory 检索的历史信息（作为 context token）
- $\otimes$：element-wise gating

**架构图解析**（Figure 2）：三个 branch 汇聚到 attention —— persistent（固定）、long-term memory 检索（动态）、current input。Attention 同时看到三者，可以学习"什么时候该用历史信息、什么时候该忽略"。

**关键优势**：attention 帮 memory 过滤 —— 不是所有 token 都值得记住，attention 决定哪些信息写入 memory，避免 overflow。

### 5.2 MAG: Memory as Gate

两个 parallel branch + gating（Equation 26-28）：

$$\tilde{x} = [p_1, ..., p_{N_p}] \,||\, x$$
$$y = \text{SW-Attn}^*(\tilde{x})$$
$$o = y \otimes \mathcal{M}(\tilde{x})$$

- $\text{SW-Attn}^*$：sliding window attention with prefix（prefix 部分 allow attending to persistent memory）
- $\mathcal{M}(\tilde{x})$：memory module 对整个序列的递归输出
- $\otimes$：non-linear gating（实验中用 normalize + SiLU）

**架构图解析**（Figure 4）：SWA 作为 short-term memory（精确但窗口有限），neural memory 作为 long-term memory（模糊但持久），两者通过 gate 融合。类似 multi-head attention 但 head 结构不同（参考 Hymba: https://arxiv.org/abs/2411.13676）。

### 5.3 MAL: Memory as Layer

最简单的设计，顺序堆叠（Equation 29-31）：

$$\tilde{x} = [p_1, ..., p_{N_p}] \,||\, x$$
$$y = \mathcal{M}(\tilde{x})$$
$$o = \text{SW-Attn}(y)$$

memory layer 先压缩历史，attention 再处理压缩后的表示。类似 H3 架构（https://arxiv.org/abs/2212.14052）把 SSM 换成 neural memory。

**缺点**：两个 module 的互补性无法充分发挥 —— memory 压缩后的信息丢失无法被 attention 补救。

### 5.4 Attention mask 对比

Figure 3 展示了两种 mask：
- MAC：每个 segment 内 full causal attention，persistent + long-term memory token 作为 prefix
- MAG：sliding window + prefix，persistent memory 在 prefix

### 5.5 Theoretical expressiveness

**Theorem 4.1**：Titans 能解决 **TC$^0$ 以外** 的问题，而 Transformers、diagonal linear RNNs、DeltaNet 都限制在 TC$^0$ 内（参考 Merrill et al. 2024: https://arxiv.org/abs/2404.02805）。

TC$^0$ 是 circuit complexity class，指 constant-depth、polynomial-size threshold circuits 能计算的问题类。这意味 Titans 在 state tracking 任务上理论上比 Transformer 更强。

---

## 6. 实验数据深度解读

### 6.1 Language Modeling（Table 1）

760M params / 30B tokens 的关键数据：

| Model | Wiki ppl ↓ | Avg acc ↑ |
|-------|-----------|-----------|
| Transformer++ | 25.21 | 48.69 |
| Mamba2 | 22.94 | 48.34 |
| DeltaNet | 24.37 | 48.97 |
| TTT | 24.17 | 47.32 |
| Gated DeltaNet | 21.18 | 49.69 |
| Titans (LMM) | 25.03 | 47.83 |
| Titans (MAC) | 25.61 | 48.65 |
| Titans (MAG) | **23.59** | 48.60 |
| Titans (MAL) | 23.93 | 47.87 |

观察：
- **non-hybrid** 比较：Titans (LMM) 在 340M/400M 规模胜过所有 baselines，但在 760M 规模 perplexity 略逊于 Gated DeltaNet。paper 把这归因于 Gated DeltaNet 有高度优化的 kernel。
- **hybrid** 比较：Titans (MAG/MAL) 的 Wiki ppl 23.59/23.93 优于 Samba (25.32) 和 Gated DeltaNet-H2 (24.19)，验证了架构设计的价值。
- MAG 在语言建模上略优于 MAC，但差距很小。

### 6.2 Needle in Haystack（Table 2）—— 真正的杀手锏

S-NIAH-W（word needle）在 16K context：

| Model | 2K | 4K | 8K | 16K |
|-------|-----|-----|-----|------|
| TTT | 78.8 | 28.0 | 4.4 | 0.0 |
| Mamba2 | 42.2 | 4.2 | 0.0 | 0.0 |
| DeltaNet | 46.2 | 20.0 | 1.6 | 0.0 |
| Titans (LMM) | 90.4 | 89.4 | 85.8 | **80.6** |
| Titans (MAC) | 98.2 | 98.2 | 95.6 | **95.2** |
| Titans (MAG) | 98.0 | 98.0 | 90.2 | 88.2 |
| Titans (MAL) | 98.0 | 97.4 | 92.0 | 90.4 |

**这是 paper 最震撼的结果**。所有 baselines 在 16K 时几乎完全崩溃（0.0），而 Titans (MAC) 保持在 95.2。原因：

1. **vs TTT**：Titans 有 forgetting mechanism（weight decay）+ momentum，避免长序列下 memory overflow 和 gradient 消失。
2. **vs Mamba2**：Mamba2 有 gating 但 memory 是 matrix-valued 线性的，且无法 remove memory（只能 decay），长序列下旧信息污染。Titans 的 deep non-linear memory + delta-style update 能精确替换旧 memory。
3. **vs DeltaNet**：DeltaNet 有 delta rule（能 remove memory）但 lacks forget gate，无法 erase，长序列下积累过多。

MAC 在 NIAH 上显著优于 MAG/MAL，因为 attention 直接看到 memory 检索的历史 context，对 needle 定位最直接。

### 6.3 BABILong（Figure 6）—— 超越 GPT-4

BABILong（https://arxiv.org/abs/2406.04268）是比 NIAH 更难的任务：需要在极长文档中**跨多个事实推理**，而不只是检索单个 needle。

**Few-shot setup**（Figure 6a）：Titans (MAC) 在 760M 参数下超过 GPT-4、Llama3.1-8B、Gemma-9B、RecurrentGemma-9B、RWKV-6-7B、Mamba2-8B。

**Fine-tuning setup**（Figure 6b）：Titans (MAC) 超过 GPT-4、Qwen2.5-72B、Llama3.1-70B。注意 Titans 参数量约 760M，比 GPT-4 少约 70x，比 Llama3.1-70B 少约 90x。

这个结果暗示：**对于需要长 context 推理的任务，结构化的 test-time learning memory 比 brute-force scale 更有效**。

### 6.4 Memory Depth 的影响（Figure 7, 8）

| $L_\mathcal{M}$ | 170M ppl | 360M ppl | 760M ppl |
|-----------------|----------|----------|----------|
| 1 | 较高 | 较高 | 较高 |
| 2 | ↓ | ↓ | ↓ |
| 3 | ↓↓ | ↓↓ | ↓↓ |
| 4 | ↓↓↓ | ↓↓↓ | ↓↓↓ |

观察：
- 更深的 memory → 更好的 perplexity，且**对 sequence length 更鲁棒**（曲线更平）
- 小模型从深度中获益更大（170M 的曲线差异最显著）
- 训练 throughput 随深度线性下降（Figure 8），但所有配置都保持对 context length 的线性 scaling

### 6.5 Ablation Study（Table 5）

对 long context acc 的影响（baseline LMM = 92.68）：

| Component removed | Long Context acc | Δ |
|-------------------|------------------|---|
| Linear memory (no depth) | 85.34 | -7.34 |
| w/o Weight decay | 85.60 | -7.08 |
| w/o Momentum | 87.12 | -5.56 |
| w/o Convolution | 90.28 | -2.40 |
| w/o Persistent memory | 92.49 | -0.19 |

**关键发现**：
- **Deep memory 最重要**（-7.34）：验证了 non-linear memory 的价值，呼应 universal approximation。
- **Weight decay（forget gate）第二重要**（-7.08）：没有遗忘就无法管理长序列 memory。
- **Momentum 第三**（-5.56）：token flow 信息对长 context 至关重要。
- Convolution 有适度贡献（-2.40），参考 Mamba/GLA 的局部 mixing。
- Persistent memory 贡献最小（-0.19），但在 language modeling 上贡献更大。

### 6.6 架构变体对比（Table 5 下半部分）

| Variant | Language ppl | Reasoning acc | Long Context acc |
|---------|-------------|---------------|-------------------|
| LMM (no attention) | 27.01 | 47.83 | 92.68 |
| +Attn (MAC) | 26.67 | 48.65 | **97.95** |
| +Attn (MAG) | **25.70** | 48.60 | 96.70 |
| +Attn (MAL) | 25.91 | 47.87 | 96.91 |

- MAC 在 long context 上最强（97.95）
- MAG 在 language modeling 上最强（25.70）
- MAL 各方面略逊于 MAC/MAG
- **LMM alone 已经很强**：92.68 long context acc 优于所有 baselines 的 hybrid 变体

### 6.7 Time Series 和 DNA（Table 3, 4）

- **Time Series**（ETT, ECL, Traffic, Weather）：Neural Memory 在 7 个数据集上 MSE 全面优于 Simba（Mamba-based）、iTransformer、PatchTST、TimesNet、DLinear 等。
- **DNA Modeling**（GenomicsBenchmarks）：Titans (LMM) 在 Enhancer Cohn (75.2) 和 Non-TATA Promoters (96.6) 上达到 SOTA，其他任务 competitive。

这表明 neural memory 是 **task-agnostic 的 sequence modeling primitive**，不限于语言。

---

## 7. 与现代 RNN 的关系（Appendix C）

paper 在 Appendix C 给出了非常清晰的关系图：

### LMM ⊃ Gated DeltaNet

Gated DeltaNet 的 update rule（Equation 34）：
$$\mathbf{S}_{t+1} = \mathbf{S}_t (\mathbf{I} - \theta_t \mathbf{k}_t \mathbf{k}_t^\top) + \theta_t \mathbf{v}_t \mathbf{k}_t^\top$$

设 $\eta_t = 0$（去掉 momentum），LMM 退化为 Gated DeltaNet。LMM 的三个泛化：
1. **Momentum-based rule**（考虑 token flow）
2. **Deep memory**（MLP 而非 matrix）
3. **Non-linear recurrence**（inter-chunk 非线性）

### LMM ⊃ Longhorn

Longhorn（Equation 35）用 implicit online learning 推导 closed form，但 lacks forget gate。LMM 额外有 forget gate + momentum + deep memory。

### LMM ⊃ TTT

TTT（https://arxiv.org/abs/2407.04620）是唯一另一个 gradient-based modern RNN，但：
1. 无 forget mechanism → 长序列 overflow
2. 无 momentum → 只有 momentary surprise
3. 未实验验证 deep memory 的价值

---

## 8. 我的 intuition 总结

让我把这篇 paper 的设计哲学压缩成几个 takeaway：

### 8.1 Memory as parameters, not state

传统 RNN 把 memory 当 hidden state（vector/matrix），Titans 把 memory 当 **neural network 的权重**。这带来三个好处：
- 可以用任意深度的 MLP 作为 memory（expressive power）
- 读写就是 forward pass / gradient step（统一的优化语言）
- Test time 仍在学习（meta in-context learner）

### 8.2 Surprise = gradient

这是一个很美的对应：**gradient 衡量"当前数据让 model 多惊讶"**。大 gradient → 大惊讶 → 值得记住。这把心理学的 surprise theory 和优化理论联系起来了。

### 8.3 Momentum = surprise 的跨时间记忆

单步 gradient 会在大惊讶后饱和。Momentum 让 surprise 跨时间累积，对应人类"一个惊讶事件让我们持续记住后续一段时间"的现象。

### 8.4 Weight decay = forgetting gate

这个等价性是 paper 的一个技术亮点。它让 forgetting 机制可以用优化器语言表达，从而复用 mini-batch gradient descent 的 parallelization 技术。

### 8.5 Attention + Memory 的分工

- Attention：short-term，精确，quadratic，limited window
- Neural Memory：long-term，模糊（压缩的），linear，unlimited
- Persistent Memory：task knowledge，input-independent

三者对应人类认知的 working memory / long-term memory / procedural memory 的三分法（Cowan 2008）。

### 8.6 MAC 的设计哲学

MAC 让 attention **看到** memory 检索的历史信息，而不是简单地把 memory 输出和 attention 输出拼接。这让 attention 可以学习"什么时候该用历史、什么时候该忽略"，同时也让 attention 帮 memory 过滤"什么该记住"。这种 bidirectional 的协同是 MAC 在 NIAH 上远超 MAG/MAL 的原因。

---

## 9. Open Questions 和 Future Directions

paper 自己提到的一些方向：

1. **更高效的 memory architecture**：当前用简单 MLP，可以探索 Memory Mosaics（https://arxiv.org/abs/2405.06394）、Memory Layers at Scale（https://arxiv.org/abs/2412.09764）等更专门的 memory architecture。

2. **Chunk-level 参数**：把 $\alpha, \theta, \eta$ 从 token-dependent 改为 chunk-dependent，用 global convolution 加速。

3. **Memory 的理论容量**：deep memory 能存多少信息？和 context length 的关系是什么？

4. **和 RAG 的关系**：BABILong 实验显示 Titans (760M) 超过 Llama3.1-8B + RAG。这是否意味着对于某些任务，learned memory 比 retrieval 更有效？或者两者应该结合？

5. **Scaling laws**：paper 只到 760M，更大的规模下 MAC vs MAG vs MAL 的 trade-off 如何变化？

6. **和 in-context learning 的关系**：Titans 的 memory 在 test time 学习，这和 Transformer 的 in-context learning（通过 attention 的 key-value cache）有什么深层联系？Bietti et al. 2024（https://arxiv.org/abs/2406.04028）把 attention 看作 associative memory，Titans 的 neural memory 是不是一种"更深"的 associative memory？

---

## 10. 相关阅读清单

- **TTT layers**: https://arxiv.org/abs/2407.04620 —— Titans 最直接的前驱，gradient-based RNN
- **Mamba / Mamba2**: https://arxiv.org/abs/2312.00752, https://arxiv.org/abs/2405.21060 —— structured state space models，parallel scan 的基础
- **Gated DeltaNet**: https://arxiv.org/abs/2412.06464 —— delta rule + forget gate，Titans 的特例
- **Longhorn**: https://arxiv.org/abs/2407.14207 —— implicit online learning 推导的 RNN
- **DeltaNet**: https://arxiv.org/abs/2407.06404 —— delta rule 的 parallel 实现
- **GLA**: https://arxiv.org/abs/2312.06635 —— gated linear attention
- **FlashAttention**: https://arxiv.org/abs/2205.14135 —— Titans 的 SWA 实现基础
- **RULER benchmark**: https://arxiv.org/abs/2404.06654 —— NIAH 评估
- **BABILong**: https://arxiv.org/abs/2406.04268 —— 长 context 推理评估
- **Hymba**: https://arxiv.org/abs/2411.13676 —— hybrid head architecture，MAG 的相关设计
- **Sukhbaatar et al. persistent memory**: https://arxiv.org/abs/1907.01470 —— FFN as attention 的洞察
- **Attention sinks**: https://arxiv.org/abs/2309.17453 —— persistent memory 缓解的 technical 问题
- **Merrill et al. expressiveness**: https://arxiv.org/abs/2404.02805 —— TC$^0$ 的理论限制
- **Meta-learning (MAML/Reptile)**: https://arxiv.org/abs/1703.03400, https://arxiv.org/abs/1803.02999 —— inner/outer loop 的 meta-learning 框架
- **Fast weight programmers**: https://arxiv.org/abs/2102.11174 —— Schmidhuber 学派的 memory 传统
- **Parallel associative scan**: https://arxiv.org/abs/2312.00752 —— S5，parallel scan 在 SSM 中的应用
- **Memory Mosaics**: https://arxiv.org/abs/2405.06394 —— 更复杂的 memory architecture
- **Birth of a Transformer**: https://arxiv.org/abs/2406.04028 —— attention as associative memory 的理论

---

这篇 paper 的核心贡献是把 **test-time training、momentum、weight decay、deep memory** 四个要素统一到一个可并行化的 neural memory module 里，并用 **memory perspective** 指导架构设计。MAC 变体在 NIAH 和 BABILong 上的结果（760M 超越 GPT-4）是相当强的 evidence，说明 **结构化的 long-term memory 可能是超越 brute-force scaling 的关键路径之一**。
