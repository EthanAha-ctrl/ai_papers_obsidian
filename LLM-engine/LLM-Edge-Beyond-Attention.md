---
source_pdf: LLM-Edge-Beyond-Attention.pdf
paper_sha256: f22b67beec804d9a66dedebeb3137a73a21e7e4b084e1d5316c2a3e7881d4ff2
processed_at: '2026-08-05T15:23:34-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LLM Edge 这 6 篇 paper

好, 我把技术细节都摊开了重新讲一遍, 但这次重点放在 **intuition building** 上。每个 idea 我都给你一个 "如果我是模型的 designer, 我为什么这么干" 的视角。

---

## 1. Pause Tokens — 让模型 "闭嘴先想想"

**Paper**: https://arxiv.org/abs/2310.02226

### 人话版

标准 transformer 有个奇怪的限制: 每个 token 必须在 $L$ 层之后立刻 "开口说话"。你不能说 "等一下, 让我再算两步"。Pause token 就是给模型塞几个 "..." 占位符, 让它多走几个 forward step 再吐真 token。

举个例子, 正常训练:

```
input:  The kid is playing soccer
target: The kid is playing soccer
```

Pause pretraining:

```
input:  The kid is <pause> playing <pause> soccer
target: The kid is <pause> playing <pause> soccer
                ↑ ignore    ↑ ignore
```

`<pause>` 位置的 loss 被 mask 掉, 但 `<pause>` 自身作为 input token 依然进入 attention, 所以后面 `playing` 的 representation 会经过 `<pause>` 这个位置。换句话说模型在 `<pause>` 处做了一次 free-form computation, 没人逼它输出任何 distribution。

### 为什么会 work — 三种直觉

**直觉 A: 延迟 commit 的 bandwidth 问题**

正常 LM 在每个位置都被强迫 output 一个 vocab size $V \approx 50000$ 维的 distribution。这是 **极高 bandwidth 的输出 bottleneck**。模型的 hidden state 是 $d \approx 4096$ 维, 你硬把它压成 $V$ 维 softmax, 信息肯定丢失。Pause token 让模型在那个位置 output 一个 "什么都不是" 的 dummy, hidden state 完整地 pass 到下一层 / 下一个 token 的 KV cache 里。信息没被 bottleneck 挤掉。

**直觉 B: 等效 deeper network, 但只在需要时**

Universal Transformer (https://arxiv.org/abs/1807.03819) 用 adaptive computation time (ACT) 让模型自己决定一个 token 走几层。Pause token 是这个 idea 的 **poor man's version** — 你不改架构, 只改 data。代价是没有 control mechanism, pause 多少是 hyperparameter 而非 learned。

**直觉 C: reasoning 任务的 "chain-of-thought in latent space"**

CoT 让模型在 token space 里展开推理, 每个推理 step 都要 decode 成自然语言再 encode 回来, 这中间有 lossy round-trip。Pause token 直接在 latent space 里多走几步, 没有 decode-encode round-trip。对 arithmetic、program synthesis 这种 "需要多步精确计算" 的任务, latent CoT 比显式 CoT 更高效。

### 没解决的问题

论文没回答 **为什么** pause work。我的猜测: pause token 本质是在 hidden state 上加了 "freedom degree"。正常训练每个 hidden state 都要服务于 "预测下一个真实 token", 这是个强 supervised signal, hidden state 的 representation 被拉去 fit vocab distribution。Pause 位置的 hidden state 没有这个 supervised signal, 它可以自由演化去 capture 任何对后续 token 有用的中间 computation。这跟 "depth" 的作用类似, 但是 data-driven 的。

实验上 pause 在 reasoning task 上涨 3-10 个点, 在普通 PPL benchmark 上几乎不变。这说明 pause 不是 "让模型更 fluent", 而是 "让模型在需要 thinking 的任务上多算一会儿"。Inference cost 线性涨, 因为生成更多 token。

---

## 2. Meet in the Middle — 两个 LM 对向生成, 中间对暗号

**Paper**: https://arxiv.org/abs/2403.07576

### 人话版

代码补全这种任务, 你有 prefix `def factorial(n):` 和 suffix `return result`, 想生成中间部分。标准 left-to-right LM 只看到 prefix, 不知道 suffix 是什么, 容易生成跟 suffix 对不上的代码 (比如 prefix 里用了 `res` 但 suffix 里是 `result`)。

Meet in the Middle 训练 **两个** LM: forward $P$ 和 backward $F$。Forward 正常读 prefix 预测下一个 token。Backward 把 sequence 反过来, 从 suffix 末尾开始往回读, 预测 "上一个 token"。训练时加一个 **agreement regularizer**: 在中间某个位置, $P$ 和 $F$ 对同一个 token 的 prediction distribution 要接近。

Loss 我再写一遍, 标清每一项在干什么:

$$\mathcal{L} = \underbrace{-\log P(x_i \mid x_{<i})}_{\text{forward LM loss}} - \underbrace{\log F(x_i \mid x_{>i})}_{\text{backward LM loss}} + \underbrace{\beta \cdot D_{\mathrm{KL}}(P(\cdot \mid x_{<m}) \| F(\cdot \mid x_{>m}))}_{\text{让中间 token 的两个分布对齐}}$$

- $P(x_i \mid x_{<i})$: forward 给定前缀预测 $x_i$
- $F(x_i \mid x_{>i})$: backward 给定后缀预测 $x_i$
- $m$: 中间位置
- $D_{\mathrm{KL}}$: KL 散度, 衡量两个 distribution 差多远
- $\beta$: 超参, 控制 agreement 的强度

### Inference 怎么用

prefix 那边 forward 往后生成, suffix 那边 backward 往前生成, 两路在中间相遇。如果生成的整段 token sequence 一致, 就接受。注意是 **整段** 对比, 不是单 token 对比 — 因为单 token 太容易 "两个模型都默认输出 'the'" 这种 false positive。

### Intuition

这本质是 **self-consistency 的一种特例**。普通 self-consistency 是同一个模型 sample 多次看 majority vote。Meet in the Middle 是两个 **不同方向** 的模型互验。如果 forward 和 backward 都指向同一段 middle, 那大概率这段 middle 是 "从两头看都合理" 的, confidence 更高。

但这里有个 subtle 问题: forward 和 backward 是两个独立模型, 它们的 latent representation space 完全不同。KL agreement 是在 **output distribution 层面**, 不在 representation 层面。所以这个方法没法做 "共享 hidden state" 这种更激进的方案, 只能在 output 层面对齐。

成本: 训两份模型, 推理也跑两份。对 infill-heavy 的场景 (IDE 代码补全) 可能值得, 对一般 chatbot 不划算。

---

## 3. Infini-attention — 这篇是真正的 long context 解法

**Paper**: https://arxiv.org/abs/2404.07143

### 人话版

标准 transformer 处理 1M token, attention 是 $O(L^2)$, KV cache 是 $O(L)$, 直接爆炸。Infini-attention 的 idea: **把 context 切成 segment, segment 内走正常 attention, segment 之间用一个固定大小的 memory matrix $M$ 传递信息**。

这就像 RNN — RNN 有个 hidden state 跨 timestep 传递, Infini-attention 有个 memory matrix 跨 segment 传递。但 Infini-attention 的 memory 是 $d_k \times d_v$ 的 matrix, 不是 vector, 信息容量大得多。

### 三个公式, 一个一个掰开

**公式 1: Local attention (segment 内)**

$$A_{dot} = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

这个就是标准 attention, 没啥可说。$Q, K, V$ 都来自当前 segment 的 $N$ 个 token。$A_{dot} \in \mathbb{R}^{N \times d_v}$ 是当前 segment 的 attention output。

**公式 2: Memory retrieval (从过去的 segment 拿信息)**

$$A_{mem} = \frac{\sigma(Q) \, M_{s-1}}{\sigma(Q) \, z_{s-1}}$$

这是关键创新。拆开看:

- $M_{s-1} \in \mathbb{R}^{d_k \times d_v}$: 上一个 segment 结束时存下的 memory。你可以把它想成 "过去所有 segment 的 $(K, V)$ 对被压缩进一个 matrix"。具体怎么压缩见公式 3。
- $z_{s-1} \in \mathbb{R}^{d_k}$: normalization vector, 防止分母 unbounded。
- $\sigma(\cdot)$: ELU+1 非线性, 保证非负 (这样才能当 "概率/权重" 用)。
- $\sigma(Q) \in \mathbb{R}^{N \times d_k}$: 当前 segment 的 queries 经过非线性。
- $\sigma(Q) M_{s-1} \in \mathbb{R}^{N \times d_v}$: 每个当前 query 跟 memory 做内积, retrieve 出一个 value vector。
- 分母 $\sigma(Q) z_{s-1} \in \mathbb{R}^{N}$: 每个 query 的 normalization, 类似 softmax 的分母。

**直觉**: memory $M$ 是一个 associative memory。你往里面存了一堆 $(key, value)$ 对, query 来了就根据跟 key 的相似度加权取 value。这跟标准 attention 一模一样, 区别在于 standard attention 的 $(K, V)$ 是当前 sequence 里的, Infini-attention 的 $(K, V)$ 是被压缩进 $M$ 的历史信息。

**公式 3: Memory update (把当前 segment 的信息存进 memory)**

$$M_s \leftarrow M_{s-1} + \sigma(K)^\top \big(V - \sigma(K) M_{s-1}\big)$$

这个最精妙, 我慢慢拆:

- $\sigma(K) M_{s-1} \in \mathbb{R}^{N \times d_v}$: 用旧 memory 根据当前 segment 的 keys 去 retrieve, 得到 "memory 对当前 segment values 的预测"。
- $V - \sigma(K) M_{s-1}$: **prediction error** — 当前 segment 真正的 values 减去 memory 预测的 values。如果 memory 已经 "记得" 这些 values, error 接近 0, 不更新。如果 memory 没见过, error 大, 更新大。
- $\sigma(K)^\top (\cdot) \in \mathbb{R}^{d_k \times d_v}$: 把 error 按 keys 投影回 memory space, 加到 $M$ 上。

**这就是 delta rule / Rescorla-Wagner learning**, 等价于对 associative memory loss $\|V - \sigma(K) M\|^2_F$ 做一步 gradient descent。如果同一个 $(K, V)$ pair 反复出现, memory 逐渐 converge 到完美存储; 如果出现新的 $(K, V)$, memory 增量更新。

$z$ 的更新:
$$z_s \leftarrow z_{s-1} + \sigma(K)^\top \mathbf{1}_N$$

就是累加每个 key 出现次数, 用作 normalization denominator。

**公式 4: Gating — 长期 vs 短期**

$$A = \mathrm{sigmoid}(\beta) \odot A_{mem} + (1 - \mathrm{sigmoid}(\beta)) \odot A_{dot}$$

- $\beta \in \mathbb{R}$: 每个 head 每个 layer 学一个 scalar gate
- $\mathrm{sigmoid}(\beta) \in (0, 1)$: memory 的权重
- $1 - \mathrm{sigmoid}(\beta)$: local attention 的权重

**直觉**: 浅层 head 可能 $\beta$ 偏小, 依赖 local attention (语法、邻近词); 深层 head 可能 $\beta$ 偏大, 依赖 memory (long-range entity, 事实)。模型自学这个 trade-off。

### 为什么 K 和 V 都隐式存在 memory 里

$M \in \mathbb{R}^{d_k \times d_v}$ 的 shape 就暗示了它是 K-V outer product 的累加。展开 delta rule 的稳态解 (反复存同一个 $(K, V)$ pair 多次后):

$$M \approx \sum_t \sigma(K_t)^\top V_t$$

这就是一堆 outer product 的和。query 来了:

$$\sigma(Q) M \approx \sum_t \langle \sigma(Q), \sigma(K_t) \rangle V_t$$

跟 standard attention 的 $\sum_t \mathrm{softmax}(\langle Q, K_t \rangle) V_t$ 结构一样, 只是把 softmax 换成 linear kernel, 把 "当前 sequence 的 $(K, V)$" 换成 "历史压缩的 $(K, V)$"。

### 跟其他 memory 方法对比

| 方法 | Memory 结构 | 大小随 context 增长? | 更新方式 |
|---|---|---|---|
| Memorizing Transformer | kNN index of raw (K, V) | 是, 线性 | 直接 append |
| Compressive Transformer | 压缩后的 segment | 是, 但慢 (有压缩) | convolution/attention 压缩 |
| RTM | memory token | 否, 固定 | token 传递 |
| Infini-attention | $d_k \times d_v$ matrix | 否, 固定 | delta rule |
| Mamba (SSM) | hidden state vector | 否, 固定 | 线性 recurrence |

Infini-attention 的 sweet spot: memory 比 kNN 小 (固定大小), 比 SSM hidden vector 大 (matrix vs vector), update rule 有理论 grounding (delta rule = 一步 GD)。

### 跟 Gemini 1M context 的关系

Google 没公开 Gemini 架构, 但 Infini-attention 的设计 (bounded memory, streaming, plug-and-play continual pretraining) 跟 Gemini 1.5 Pro 报告里 "1M+ token 稳定推理" 的描述高度吻合。可以视为 Google long-context 方法论的公开版本。

实验 (1B model, PG19 long document PPL):

| Context length | Vanilla transformer | Infini-attention |
|---|---|---|
| 32k | 11.0 PPL | 11.0 PPL |
| 100k | 爆内存 | 9.5 PPL |
| 1M | 爆内存 | 9.2 PPL |

PPL 在 100k-1M 区间还能持续下降, 这在标准 transformer 上看不到 — 标准 transformer 的 PPL 在超出训练 length 后会飙升。

---

## 4. RoPE — 用旋转编码位置, 让 attention 只看 "距离" 不看 "绝对位置"

**Paper**: https://arxiv.org/abs/2104.09864
**作者博客 (最权威)**: https://kexue.fm/archives/8265

### 人话版

原始 sinusoidal PE 是 absolute 的: 位置 0 的 PE 是 $[\sin 0, \cos 0, \sin 0, \cos 0, \dots]$, 位置 1 的 PE 是 $[\sin\theta_0, \cos\theta_0, \sin\theta_1, \cos\theta_1, \dots]$, 直接加到 token embedding 上。

问题 1: 模型要记住 "位置 87 长什么样", "位置 88 长什么样", 这是 absolute pattern, 不能泛化到训练时没见过的长度。

问题 2: 加到 embedding 上, 等于在 semantic representation 上加 noise, 深层网络可能 hurt。

RoPE 的 idea: **把位置编码成对 Q, K 的 rotation**。位置 $m$ 的 Q 被 rotate 角度 $m\theta_i$, 位置 $n$ 的 K 被 rotate 角度 $n\theta_i$。它们做内积时, 由旋转群的性质 $R_m^\top R_n = R_{n-m}$, attention score 只依赖相对位置 $n - m$。

### 公式拆解

对 $d$ 维的 query/key vector, RoPE 把它切成 $d/2$ 个 2D 子空间:

$$R_{\Theta, m} = \mathrm{blockdiag}\begin{pmatrix} \begin{pmatrix} \cos m\theta_0 & -\sin m\theta_0 \\ \sin m\theta_0 & \cos m\theta_0 \end{pmatrix}, \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 \\ \sin m\theta_1 & \cos m\theta_1 \end{pmatrix}, \dots \end{pmatrix}$$

- $m$: token 的 absolute position (0, 1, 2, ...)
- $\theta_i = 10000^{-2i/d}$, $i = 0, 1, \dots, d/2-1$: 第 $i$ 个 2D 子空间的 rotation frequency
  - $i$ 小 → $\theta_i$ 大 → 旋转快 → wavelength 短 → 编码近距离
  - $i$ 大 → $\theta_i$ 小 → 旋转慢 → wavelength 长 → 编码远距离
- 每个 2D block 是一次平面旋转, 角度是 $m \theta_i$

**关键不变性**:

$$\langle R_{\Theta, m} q, R_{\Theta, n} k \rangle = q^\top R_{\Theta, m}^\top R_{\Theta, n} k = q^\top R_{\Theta, n-m} k$$

attention score 只依赖 $n - m$ (相对距离), 跟 absolute position $m, n$ 无关。

### 为什么是 2D chunk

旋转矩阵最自然的 representation 是 2D (你没法在 1D 里旋转, 3D 里旋转会改变 norm 或引入 unwanted coupling)。把 $d$ 维拆成 $d/2$ 个 2D 平面, 每个平面独立旋转不同 frequency, 等于 **multi-scale positional encoding** — 有的 dimension pair 编码近距离, 有的编码远距离, 类似 Fourier basis 的不同 frequency component。

### 跟 sinusoidal 的对比

笔记里那张 cosine similarity 表其实是在说: sinusoidal PE 下, 位置 $m$ 和 $m+1$ 的 cosine similarity 跟 $m$ 本身有关 (因为 absolute), 不同 $m$ 处的 "相邻关系" 不一样。RoPE 下, 任意两个距离为 $r$ 的 token, attention pattern 相同 (因为 relative)。

更重要的差异: **RoPE 是乘到 Q/K 上, sinusoidal 是加到 embedding 上**。乘法不改变 vector 的 norm, 只是 rotate direction, semantic information 保留得更好。加法直接扰动 embedding, 深层网络可能把 PE noise 放大。

### Length extrapolation — RoPE 没完全解决的问题

RoPE 让 attention **pattern** 对 relative distance invariant, 但训练时模型只见过 $m \in [0, L_{train}]$ 范围内的 rotation angle。推理时 $m > L_{train}$, rotation angle 超出训练分布, 模型对这种 angle 的 response 没被训过。

后续工作解决:

- **Position Interpolation (PI)** (https://arxiv.org/abs/2306.15595): 把 $m \to m \cdot L_{train}/L_{target}$, 整体缩放回训练范围。粗暴但有效。
- **NTK-aware scaling** (https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/): 改 base frequency 10000 → 更大值, 整体 stretch 所有 frequency band。
- **YaRN** (https://arxiv.org/abs/2309.00071): 分段 — 高频 band 不动 (近距离仍然 fine), 低频 band stretch (远距离需要 adapt)。

这些方法都依赖 RoPE 的 frequency decomposition 结构。sinusoidal 加法 PE 没法这么自然 extend。

### 实现上的巧思

RoPE 用复数乘法高效实现: 把 $(q_{2i}, q_{2i+1})$ 视为复数 $q_{2i} + i q_{2i+1}$, rotation by $m\theta_i$ 就是乘 $e^{i m\theta_i} = \cos(m\theta_i) + i \sin(m\theta_i)$。PyTorch 里 `torch.polar` 一步搞定。

主流模型 (Llama, Mistral, Qwen, DeepSeek) 全用 RoPE。

---

## 5. KV Cache — 推理时省重复计算

### 人话版

Autoregressive generation 第 $t$ 步, 你要算 attention $(Q_t, K_{1:t}, V_{1:t})$。前 $t-1$ 个 token 的 $K, V$ 上一步已经算过了, 这一步完全不用重算, 直接从 cache 拿。

具体:

```
Step t:
  x_new = embedding(token_t)          # 1 x d_model
  k_new = x_new @ W_K                 # 1 x d_k
  v_new = x_new @ W_V                 # 1 x d_v
  K_cached = concat(K_cached, k_new)  # t x d_k
  V_cached = concat(V_cached, v_new)  # t x d_v
  q_new = x_new @ W_Q                 # 1 x d_k
  attn_out = softmax(q_new @ K_cached.T / sqrt(d_k)) @ V_cached
```

没 cache 的话, 每步都要对前 $t-1$ 个 token 重算 $K = X_{1:t} W_K$, 这是 $O(t \cdot d^2)$ 的 matmul, $t$ 步总共 $O(t^2 d^2)$。有 cache 的话每步 $O(d^2 + t d)$, 总共 $O(t d^2 + t^2 d)$。

### Memory 问题

每个 layer 缓存 $K$ 和 $V$, 总量:

$$\text{KV cache size} = 2 \cdot n_{layers} \cdot L \cdot d_{model} \cdot \text{bytes\_per\_param}$$

Llama-2-7B ($n_{layers}=32, d_{model}=4096$, fp16=2 bytes): 

$$2 \times 32 \times L \times 4096 \times 2 = 524288 \cdot L \text{ bytes} \approx 0.5 \text{ MB per token}$$

1000 token → 0.5 GB, 1M token → 500 GB。这就是 long context inference 贵的原因, 也是 Infini-attention / PagedAttention / KV quantization 的动机。

### RoPE 上的 cache 细节

RoPE 是 position-dependent rotation。cache K 时, 你要么 cache "已经 rotated 的 K" (position $m$ 的 K 被 rotate by $m\theta$), 要么 cache "raw K" 在 attention 时再 rotate。

主流实现 (vLLM, FlashAttention) 选前者 — cache rotated K。优点: attention 时直接矩阵乘, 不用再 apply rotation。缺点: 如果你想改 position (比如 sliding window 把 position 0 的 K 重用到 position 1000), 需要重新 rotate, 麻烦。

### 进阶优化

- **PagedAttention / vLLM** (https://arxiv.org/abs/2309.06180): KV cache 分 page (类似 OS 的 virtual memory), 解决 variable-length sequence 的 fragmentation。多 batch 时内存利用率从 ~20% 提到 ~90%。
- **KV quantization** (https://arxiv.org/abs/2309.15217, KIVI): K cache 量化到 2-bit, V cache 量化到 4-bit, 内存减 4-8 倍, PPL 几乎不变。
- **H2O** (https://arxiv.org/abs/2306.14048): 观察 attention score, 丢掉 "不重要" 的 KV (attention weight 长期接近 0 的 token)。
- **StreamingLLM** (https://arxiv.org/abs/2309.17453): 关键发现 — 保留 position 0, 1, 2 这几个 "attention sink" token 的 KV (它们被所有后续 token attend), 加上 sliding window 的 recent KV, 就能无限长度生成。原因: 模型训练时把 "不该 attend 的" attention mass dump 到 sequence 开头, 这些 position 的 KV 是 structural 必需的, 不能丢。

---

## 6. Mixtral MoE — 不用全算, 选两个 expert 算就够

**Paper**: https://arxiv.org/abs/2401.04088
**权重**: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1

### 人话版

Dense model 的每个 token 都经过同一个 FFN。但不同 token 可能需要不同 "技能" — code token 需要语法 expert, math token 需要计算 expert, narrative token 需要风格 expert。MoE 的 idea: **搞 8 个 FFN expert, 每个 token 只用 2 个**, 由 router 决定用哪两个。

### Router 怎么工作

每个 token 的 hidden state $x \in \mathbb{R}^{d_{model}}$ 进 router:

$$\text{logits} = W_g \cdot x \in \mathbb{R}^{8}$$

$W_g \in \mathbb{R}^{8 \times d_{model}}$ 是 router weight, 输出 8 维 logits, 每个 logit 对应一个 expert 的 "适合度"。

然后:

$$G(x) = \mathrm{softmax}(\mathrm{TopK}(\text{logits}, K=2))$$

TopK 选最大的 2 个, 其余 6 个置 $-\infty$, softmax 后这 6 个 probability 为 0。剩下的 2 个 expert 的 probability 加起来为 1。

Expert output:

$$y = \sum_{i \in \{2 \text{ selected experts}\}} G_i(x) \cdot E_i(x)$$

只有 2 个 expert 的 FFN 被 forward, 其余 6 个完全不计算。

### 为什么能省 compute

Dense model 7B: 每 token forward 7B 参数。
Mixtral 8x7B: 每 token forward ~13B 参数 (2 个 expert × 5.6B + attention/router 共享参数), 但总参数 47B。

inference FLOPs 接近 13B dense model, 性能接近 70B dense model。这就是 MoE 的核心 trade-off: **总参数决定 capacity, active 参数决定 inference cost**。

### Load balancing — 防止 expert 坍塌

如果不加约束, router 容易陷入 "rich-get-richer" — 某个 expert 一开始被选中多一点, 它训练得更好, 下次更可能被选中, 最终 collapse 到 1-2 个 expert, 其他 expert 浪费。

Mixtral 用 auxiliary loss:

$$\mathcal{L}_{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

- $f_i = \frac{1}{T}\sum_t \mathbb{1}\{i \in \mathcal{S}(x_t)\}$: expert $i$ 实际被选中的频率 (硬统计, 0 或 1 per token)
- $P_i = \frac{1}{T}\sum_t \mathrm{softmax}(W_g x_t)_i$: expert $i$ 的平均 router probability (软统计)
- $N = 8$: expert 数量
- $\alpha = 0.01$: loss 权重

**直觉**: 均匀分布时 $f_i = P_i = 1/8$, $L_{aux} = 8 \cdot 8 \cdot (1/8)^2 = 1$ (最小值)。Collapse 时某个 $f_i \to 1, P_i \to 1$, $L_{aux} \to 8$ (最大)。这个 loss 推动 router 均匀分配 token。

注意 $f_i$ 是 **不可微** 的 (TopK 的选择是 hard), 但 $P_i$ 可微。两者乘起来, 梯度通过 $P_i$ 流回 router, 推 router 在 "被频繁硬选中的 expert" 上降低 soft probability。设计很巧妙。

### Expert specialization 观察实验

笔记里说 "no patterns of expert on a topic"。Mixtral 论文做了分析, 发现:

1. **Routing 是 token-level 而非 sequence-level**: 同一句话里不同 token 路由到不同 expert。没有 "这句话归 expert 3" 这种 pattern。
2. **Syntactic 而非 semantic pattern**: 某些 expert 倾向处理 function word (the, of, is), 某些倾向 content word。没有明显的 "code expert" / "math expert" 分工。
3. **跟 encoder MoE 不同**: ST-MoE (https://arxiv.org/abs/2202.08906) 在 encoder 上观察到 topic-level specialization (某 expert 专做 medical, 某专做 legal)。Decoder autoregressive + top-k 可能因为 token-level 预测任务太细粒度, 没法形成 topic-level 分工。

我的直觉: decoder MoE 的 specialization 可能更在 **representation level** 而非 **task level**。某些 expert 可能专做 "long-range dependency", 某些专做 "local syntax", 但这些是 latent function, 不是 surface topic。

### 性能

| Benchmark | Mixtral 8x7B (~13B active) | Llama 2 70B (dense) | Llama 2 13B (dense) |
|---|---|---|---|
| MMLU | 71.8 | 69.9 | 54.8 |
| HumanEval | 40.2 | 32.3 | 18.3 |
| GSM8K | 58.4 | 56.8 | 28.7 |

Mixtral 用 13B 的 inference FLOPs 打平甚至超过 70B dense。这是 MoE 的核心 win — 你可以 afford 训一个 47B 的大模型 (capacity 高), 但 inference 时只花 13B 的 compute。

---

## 串起来看 — 这 6 个 idea 在解什么

Transformer 有几个 **fundamental 约束**:

| 约束 | 后果 | 解法 (这 6 篇) |
|---|---|---|
| 每 token 固定走 $L$ 层 | 模型不能 "多想想" | Pause token |
| 单向生成 | 不适合 infill | Meet in the Middle |
| Attention 是 $O(L^2)$, KV cache 是 $O(L)$ | Context 不能太长 | Infini-attention |
| Absolute PE 不泛化 | 长度 extrapolation break | RoPE |
| KV cache 内存爆炸 | Long context inference 贵 | PagedAttention / quantization / eviction |
| Dense FFN 全 token 都算 | Compute 浪费 | MoE |

这些约束不是独立的。Pause token 增加 per-token compute, 但如果 context 也长, compute 叠加爆炸 → 需要 MoE 省一部分 → MoE 的 expert 如果是 dense, 长 context 还是爆 → 需要 Infini-attention 这种 streaming memory → memory 的 positional encoding 需要 RoPE 才能 extrapolate → ...

**真正的 "下一代 LLM" 很可能是这些 idea 的组合**: MoE FFN + Infini-attention (or linear attention variant) + RoPE + pause/CoT for reasoning + KV cache 优化。Gemini 1.5 / GPT-4 / Claude 3 的 production 架构大概率已经混合了上述多个, 只是没公开。

---

### Web links 汇总

**主 paper**:
- Pause Tokens: https://arxiv.org/abs/2310.02226
- Meet in the Middle: https://arxiv.org/abs/2403.07576
- Infini-attention: https://arxiv.org/abs/2404.07143
- RoPE / RoFormer: https://arxiv.org/abs/2104.09864
- Mixtral: https://arxiv.org/abs/2401.04088

**RoPE 作者博客 (最权威中文解释)**:
- https://kexue.fm/archives/8265

**Length extrapolation (RoPE 后续)**:
- Position Interpolation: https://arxiv.org/abs/2306.15595
- YaRN: https://arxiv.org/abs/2309.00071
- NTK-aware (Reddit 讨论): https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/

**KV cache 优化**:
- vLLM / PagedAttention: https://arxiv.org/abs/2309.06180
- KIVI (quantization): https://arxiv.org/abs/2309.15217
- H2O (eviction): https://arxiv.org/abs/2306.14048
- StreamingLLM (attention sink): https://arxiv.org/abs/2309.17453

**相关 memory / attention 工作**:
- Universal Transformer: https://arxiv.org/abs/1807.03819
- Memorizing Transformer: https://arxiv.org/abs/2203.12151
- Compressive Transformer: https://arxiv.org/abs/1911.05507
- Mamba (SSM): https://arxiv.org/abs/2312.00752
- Linear Attention: https://arxiv.org/abs/2006.16236
- Performer: https://arxiv.org/abs/2009.14794

**MoE 相关**:
- ST-MoE (encoder specialization): https://arxiv.org/abs/2202.08906
- GShard (scaling MoE): https://arxiv.org/abs/2006.16668
- Switch Transformer: https://arxiv.org/abs/2101.03961
- DeepSeek-V2 (MoE + latent attention, 很有意思): https://arxiv.org/abs/2405.04434

**Karpathy 自己的 reference 实现**:
- nanoGPT: https://github.com/karpathy/nanoGPT
- makemore: https://github.com/karpathy/makemore

想深挖哪篇 (尤其 Infini-attention 的 delta rule 跟 modern Hopfield network 的数学联系, 或者 RoPE 的 NTK scaling 推导), 说一声我接着展开。

---

# LLM Edge: Beyond Attention — 这份笔记覆盖了 6 篇关于 Transformer 架构改进的关键 paper

这份 markdown 笔记把几个不同方向的 LLM architecture 改进整理在一起，从 "让模型多算一会儿" 到 "无限 context" 到 "MoE"。我下面按 paper 顺序，把每个方法的核心 idea、公式细节、设计直觉，以及它们之间隐含的脉络讲清楚。

---

## 1. Pause Tokens — "Think Before You Speak"

**Paper**: Goyal et al., CMU + Google Research, 2023
**arXiv**: https://arxiv.org/abs/2310.02226
**代码**: https://github.com/sequenceLab/llm-pause-tokens

### 核心动机

标准 decoder-only Transformer 里, 每个 output token 经过 **固定的** $L$ 层 transformer block 就必须 commit 一个 token 出来。模型没有 "我再想想" 的余地。Chain-of-thought (CoT) 是在 **token space** 里思考 (吐出可读文字), Pause token 是在 **embedding/activation space** 里思考 (吐出 dummy token, 但 attention 仍然作用)。

训练范式分两段:

**Pause-pretraining** — 在 pretraining corpus 里随机插入 `<pause>` token, 训练时这些位置的 output 不算 cross-entropy loss:

$$\mathcal{L} = -\sum_{t \in \mathcal{T}_{\text{real}}} \log P(x_t \mid x_{<t})$$

注意求和只对 real tokens $\mathcal{T}_{\text{real}}$, pause token 位置的 logit 被忽略, 但 pause token 自身作为输入参与 attention, 因此后续 token 的 representation 会 "经过" 这些 pause 位置。

**Pause-finetuning** — 在 SFT 数据上插入固定模式的 pause (例如每个 token 后跟一个 pause)。

### Intuition

为什么这能 work? 论文其实没完全说清, 但可以从几个角度理解:

1. **Compute per token 增加**. 如果一个 sequence 长度从 $N$ 变成 $N + P$, 每个 real token 的 output 都经过了 $L \cdot (N+P)/N$ 等效的 "activation flow", attention 的 receptive field 不变但 "depth-of-computation-per-output-token" 增加。
2. **延迟 commit**. 标准 LM 强迫模型在每个位置立即输出 distribution over vocab, 这是 high-bandwidth bottleneck。Pause token 是 low-bandwidth output (loss ignored), 但 attention/key/value 路径仍然完整, 模型可以 "delay commitment"。
3. **Information bottleneck 视角**. Pause token 强制模型把中间状态写到下一个 step 的 KV cache 里, 而不是直接到 logits 空间, 这是一种 implicit state-passing。

### 实验要点

- 在 arithmetic, reasoning, program synthesis 任务上 gain 明显 (3-10 个点)
- 在 standard LM benchmark (Wikitext PPL) 上几乎不变, 说明它不是在普通 fluency 上 help
- Pause 数量 $P$ 有 sweet spot, 过多会 hurt (训练 distribution shift)
- Inference latency 线性增加 (因为生成更多 token)

### 一个被忽略的关键细节

笔记里写 "even though this works, it's not known why"。我的直觉是: pause token 的本质是给模型一个 **没有监督信号的位置**, 让 activation 自由演化。这跟 universal transformer 里 "adaptive computation time" 的思想接近, 但 pause token 实现简单 — 你不需要改架构, 只改 data + loss mask。代价是没有 control over "pause 多久才够"。

参考 universal transformer: https://arxiv.org/abs/1807.03819

---

## 2. Meet in the Middle — 双向 LM 的中间相遇

**Paper**: Nguyen, Karampatziakis, Chen (Microsoft Azure AI), 2024
**arXiv**: https://arxiv.org/abs/2403.07576

### 动机

Infill task (fill-in-the-middle, FIM) 在代码补全、文档编辑里很常见。但 left-to-right LM 在生成 "prefix + middle + suffix" 时, middle 生成阶段不知道 suffix 是什么。Bidirectional model 知道 suffix 但不能生成。

### 训练 objective

笔记给的 loss 我重写一下, 标注变量含义:

$$\mathcal{L}(\theta_P, \theta_F; S) = \sum_{x \in S} \sum_{i=1}^{|x|} \Big[ -\log P_\theta(x_i \mid x_{<i}) - \log F_\theta(x_i \mid x_{>i}) \Big] + \beta \cdot D_{\mathrm{KL}}\!\left(P_\theta(\cdot \mid x_{<m}) \,\Big\|\, F_\theta(\cdot \mid x_{>m})\right)$$

变量:
- $P$: forward LM, 给定 $x_{<i}$ (前缀) 预测 $x_i$
- $F$: backward LM, 给定 $x_{>i}$ (后缀) 预测 $x_i$ (实际是把 sequence 反向后训练一个标准 LM)
- $m$: 中间位置, forward 走到这里, backward 也走到这里
- $D_{\mathrm{KL}}$: KL 散度, 推 forward 和 backward 在中间 token 上 distribution 接近
- $\beta$: 平衡超参, 太大让两边都被压平, 太小两边 disagree

第三项就是 **agreement regularizer**。

### Inference 流程

1. 给定 prefix $x_{<m}$, forward model 从位置 $m$ 开始 autoregressive 生成
2. 给定 suffix $x_{>m}$, backward model 从位置 $m$ 开始反向 autoregressive 生成
3. 两个 stream 在某个位置相遇, 比较生成的 token sequence
4. 用 **整个生成片段** 来判定 agreement, 而不是单个 token — 笔记里说的 "false positive on 'the'" 是关键: 单 token 同意可能是 trivial (两边都默认输出 "the"), 整段一致才可信

### Intuition

这个 idea 跟 diffusion 里 "forward + backward meet in the middle" 不太一样, 它本质上是个 **decoding-time 协议**: 用两个不对称的 LM 互相 validate。可以视为一种 self-consistency, 但成本是 train 两份模型。

潜在 issue: forward 和 backward 训练数据不一样 (一个看正向, 一个看反向), 它们的 latent representation space 完全不同。所以 KL 在 **token distribution 层面** agree 不代表 representation 层面 agree。这限制了 "中间共享 hidden state" 这种更激进的方案。

---

## 3. Infini-attention — 这篇最有意思, 可能是 Gemini 1M context 的关键

**Paper**: Munkhdalai, Faruqui, Gopal (Google), 2024
**arXiv**: https://arxiv.org/abs/2404.07143

这是笔记里花了最多篇幅的 paper, 也是我最想展开讲的。

### 设计哲学

把 Transformer 的 self-attention 和 RNN 的 compressive memory **塞进同一个 attention layer**。每个 segment 内部走 standard causal attention, segment 之间走一个 **bounded-size memory matrix**。整个长 context 是 streaming 处理的, memory 不随 context length 增长。

### 公式逐个解析

**(a) Standard dot-product attention (segment-local)**

$$A_{dot} = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_{model}}}\right) V$$

变量:
- $X \in \mathbb{R}^{N \times d_{model}}$: 输入 segment, $N$ 是 segment length
- $W_Q, W_K \in \mathbb{R}^{d_{model} \times d_k}$, $W_V \in \mathbb{R}^{d_{model} \times d_v}$: projection weights (笔记里 $W_K$ 维度写成 $d_{model} \times d_{value}$ 是 typo)
- $Q = X W_Q$, $K = X W_K$, $V = X W_V$
- 缩放因子笔记写 $\sqrt{d_{model}}$, 严格说是 $\sqrt{d_k}$

**(b) Memory retrieval — 关键创新**

$$A_{mem} = \frac{\sigma(Q) \, M_{s-1}}{\sigma(Q) \, z_{s-1}}$$

变量:
- $M_{s-1} \in \mathbb{R}^{d_k \times d_v}$: 上一个 segment 结束时存下的 memory matrix
- $z_{s-1} \in \mathbb{R}^{d_k}$: memory 的 normalization vector
- $\sigma(\cdot)$: nonlinear activation, 论文用 $\mathrm{ELU}(x) + 1$, 保证非负
- $Q \in \mathbb{R}^{N \times d_k}$: 当前 segment 的 queries (复用同一个 $Q$)
- $A_{mem} \in \mathbb{R}^{N \times d_v}$: 每行是从 memory 检索出来的 value

这本质就是 **linear attention + outer-product memory**。把 $M$ 想成过去所有 $(K, V)$ 对的加权和 (压缩后的), 用 $Q$ 做 query 来 retrieve。$z$ 是个 normalizer, 起到 softmax-like 的归一化作用, 避免 unbounded 求和。

**(c) Memory update — Delta rule**

$$M_s \leftarrow M_{s-1} + \sigma(K)^\top \big(V - \sigma(K) M_{s-1}\big)$$

变量:
- $\sigma(K) \in \mathbb{R}^{N \times d_k}$: 当前 segment 的 keys (after nonlinearity)
- $\sigma(K) M_{s-1} \in \mathbb{R}^{N \times d_v}$: 用旧 memory reconstruct 出来的 values
- $V - \sigma(K) M_{s-1}$: **prediction error** — 当前 segment 真正的 values 减去 memory 预测的 values
- $\sigma(K)^\top \in \mathbb{R}^{d_k \times N}$: 把 error 按 keys 投影回 memory space

这是经典的 **delta rule** (Rescorla-Wagner / Hopfield), 等价于一个 step of gradient descent on linear associative memory loss:

$$\mathcal{L}_{mem} = \| V - \sigma(K) M \|^2_F$$

如果一个 $(K, V)$ pair 已经被 memory 完美存储, 那么 $\sigma(K) M_{s-1} \approx V$, update term 接近 0 — 笔记里说的 "if new info already exists no update required" 就是这个意思。这是 memory 的 **去重/收敛** 性质。

$z$ 的更新类似:
$$z_s \leftarrow z_{s-1} + \sigma(K)^\top \mathbf{1}_N$$

$\mathbf{1}_N \in \mathbb{R}^N$ 是全 1 向量, 累加每个 key 出现的次数 (作为 normalization denominator)。

**(d) Aggregation — gating between memory and local attention**

$$A = \mathrm{sigmoid}(\beta) \odot A_{mem} + (1 - \mathrm{sigmoid}(\beta)) \odot A_{dot}$$

变量:
- $\beta \in \mathbb{R}$: **learnable scalar per head, per layer**, gate between long-term (memory) and short-term (local attention)
- $\odot$: element-wise (broadcasting 到 $N \times d_v$)
- $\mathrm{sigmoid}$: 把 $\beta$ 压到 $(0,1)$

直觉: 浅层可能更依赖 local attention (语法、邻近 token), 深层可能更依赖 long-term memory (事实、远距离 entity), $\beta$ 让模型自学这个 trade-off。

### 为什么 K & V 都要存

笔记里有个 Q&A: "Why store K & V in the memory?"

Memory matrix $M$ 的形状是 $d_k \times d_v$, 它 **同时压缩了 K 和 V** — 具体说, $M \approx \sum_t \sigma(K_t)^\top V_t$ (outer product 累加形式)。所以:

- **K 的角色**: 用作 memory 的 "row index" (address)。query $Q$ 和 memory 做内积 $\sigma(Q) M$ 等价于 $\sum_t \langle \sigma(Q), \sigma(K_t) \rangle V_t$, 即 "用 Q 和所有过去的 K 算相似度, 加权求 V"。
- **V 的角色**: 是真正被 retrieve 出来的 content。

没有 K, 模型无法做 addressing (不知道 memory 里哪部分相关); 没有 V, memory 是空的 (没有 content)。所以 K 和 V 都隐式存在 $M = K^\top V$ 这种 outer-product 形式里。

### Infini-attention 跟标准 linear attention 的差异

Linear attention 把 softmax 换成 kernel $\phi$:
$$A_{linear} = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) \phi(K)^\top \mathbf{1}}$$

Infini-attention 几乎就是这个, 但加了两个东西:
1. **Persistent memory across segments**: $M$ 跨 segment 累积, 不在每个 segment 后清零。所以信息可以从 1000 个 segment 之前的 token 流到当前 segment。
2. **Gating with local dot-product attention**: 保留 softmax attention 在 segment 内 (short-range, 高精度), 用 memory 跨 segment (long-range, 压缩有损)。两者 weighted sum。

### 跟 Gemini 1M context 的关系

笔记里写 "This could be key to Gemini's 1M Context Window"。Google 没有公开 Gemini 的 architecture 细节, 但 Infini-attention 的设计 (bounded memory, streaming, plug-and-play pretraining) 跟 Gemini 1.5 Pro 报告里 "在 1M+ token 上稳定推理" 的描述高度吻合。可以视为 Google 在 long-context 上的内部方法论的公开版本。

实验数据 (1B model, long-context benchmarks):

| Setup | PG19 PPL ↓ | NarrativeQA ↑ | 长文档 QA ↑ |
|---|---|---|---|
| Vanilla transformer (32k context) | 11.0 | 23.0 | — |
| Infini-attention (1M context) | 9.2 | 27.0 | 显著提升 |

注意 PPL 在 100k+ token 上还能持续下降, 这在标准 transformer 上几乎看不到 (标准 transformer 在 length generalization 上 break)。

### 我的额外联想

Infini-attention 跟几个方向都有联系:
- **Memorizing transformer** (Wu et al., 2022, https://arxiv.org/abs/2203.12151): 也用 external kNN memory, 但 non-parametric (存原始 K, V, retrieval 时做 kNN search), memory 随 context 线性增长。
- **Compressive transformer** (Rae et al., 2019, https://arxiv.org/abs/1911.05507): 把老 segment 压缩后存, 但压缩是 convolution/attention based, 不是 delta rule。
- **RTM (Recurrent Memory Transformer)**: 用专门的 memory token 跨 segment 传递。
- **Mamba / SSM** (Gu & Dao, https://arxiv.org/abs/2312.00752): 也是 RNN-style state, 但 state 是 hidden, 不是 K-V outer product。

Infini-attention 的优势是 **minimal change**: 只加一个 $M, z, \beta$ 参数, 复用现有 $W_K, W_V, W_Q$。可以直接 continual pretrain 一个现有 LLM, 不需要从头训。这是它工程上最 attractive 的点。

---

## 4. RoPE / RoFormer — 用旋转编码相对位置

**Paper**: Su et al. (追一科技 Zhuiyi), 2021
**arXiv**: https://arxiv.org/abs/2104.09864
**博客 (作者本人)**: https://kexue.fm/archives/8265
**实现**: https://github.com/ZhuiyiTechnology/roformer

### 动机

原始 sinusoidal PE 是 **absolute** 的 — 位置 $m$ 的 PE 是 $\sin(m\theta_i), \cos(m\theta_i)$, 加到 token embedding 上。问题:

1. 模型要记住所有 absolute 位置 pattern, 而不是学到 "相对距离" 这一 invariant。
2. 加到 embedding 上等于 **扰动 token 的 semantic representation**, 这在深层网络里可能 hurt。
3. 推理时 sequence length 超过训练长度, absolute PE 泛化差。

RoPE 的设计目标: 让 attention score $f(q, k, m, n)$ 只依赖 **相对位置** $n - m$, 而不是 absolute $m, n$。

### 公式

对位置 $m$ 的 query/key vector $x_m \in \mathbb{R}^d$, RoPE 把它切成 $d/2$ 个 2D 子空间, 每个子空间做一次 rotation:

$$f(x_m, m) = R_{\Theta, m} x_m$$

$$R_{\Theta, m} = \begin{pmatrix} \cos m\theta_0 & -\sin m\theta_0 & & & \\ \sin m\theta_0 & \cos m\theta_0 & & & \\ & & \cos m\theta_1 & -\sin m\theta_1 & \\ & & \sin m\theta_1 & \cos m\theta_1 & \\ & & & & \ddots \end{pmatrix}$$

变量:
- $m \in \{0, 1, 2, \dots\}$: token absolute position
- $\theta_i = 10000^{-2i/d}$, $i = 0, 1, \dots, d/2 - 1$: 第 $i$ 个 2D 子空间的 **base frequency**。$i$ 越大, $\theta_i$ 越小, wavelength $2\pi/\theta_i$ 越大 (低频, 编码远距离); $i$ 越小, frequency 越高, wavelength 越小 (高频, 编码近距离)。
- $R_{\Theta, m}$: block-diagonal, 每个 2x2 block 是一个 rotation by $m\theta_i$。

### 关键不变性

$$\langle f(q_m, m), f(k_n, n) \rangle = q_m^\top R_{\Theta, m}^\top R_{\Theta, n} k_n = q_m^\top R_{\Theta, n-m} k_n$$

利用 rotation 的群性质 $R_m^\top R_n = R_{n-m}$。所以 attention 内积 **只依赖相对位置** $n - m$。模型从数据里学到的 attention pattern 是 "距离为 $r$ 的两个 token 之间的关系", 而不是 "位置 87 和位置 89 之间的关系", 泛化到更长 sequence 时直接 reuse。

### 为什么用 2D chunk

旋转矩阵最自然的 representation 是 2D。把 $d$ 维 vector 拆成 $d/2$ 个 2D 平面, 每个平面独立旋转不同角度, 这样不同 dimension pair encode 不同 frequency band 的位置信息。这是一种 **multi-scale positional encoding**, 类似 Fourier basis。

### 跟 sinusoidal 的关键差异

笔记里那张 cosine-similarity heatmap 表对比了 sinusoidal vs RoPE 的 attention weight。我补充几点:

| 维度 | Sinusoidal PE | RoPE |
|---|---|---|
| 形式 | absolute, 加到 embedding | relative, 乘到 Q/K |
| 对 embedding 影响 | 扰动 semantic 表示 | 不改 vector norm, 仅 rotate |
| 长 context 泛化 | 差 (extrapolation break) | 好 (相对位置不变) |
| 实现复杂度 | 简单 (一次加法) | 稍复杂 (复数乘法 / sparse matrix) |
| 在 inference 上的 KV cache | cache 加了 PE 的 K | cache 已 rotated 的 K, 或 cache raw K 在 attention 时按 position 旋转 |

笔记里 "ROPE captures long-ranged dependencies" 这句需要 caveat: RoPE 在 **每个 frequency band** 都有 long-range 信号 (低频 dimension pair), 但 high-frequency pair 仍然只 encode short-range。它不是 "更重视 long-range", 而是 "对 relative distance 敏感, 包括 short 和 long"。

### RoPE 的 length extrapolation 问题

RoPE 本身不解决 length extrapolation, 训练 4k 跑 32k 仍会 break。后续工作解决:
- **Position Interpolation (PI)** (Chen et al., 2023, https://arxiv.org/abs/2306.15595): 把 $m \to m \cdot L_{train}/L_{target}$, 把超出范围的位置缩放回训练范围。
- **NTK-aware scaling** (https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/): 改 base frequency $\theta$, 整体 stretch。
- **YaRN** (Peng et al., 2023, https://arxiv.org/abs/2309.00071): 分段 scaling, 不同 frequency band 不同策略。

这些都基于 RoPE 的 frequency-decomposition 结构, sinusoidal 加法 PE 没法这么自然地 extend。

### 实现细节 (笔记里图)

实践中 RoPE 用 **复数乘法** 高效实现:
$$\text{ComplexMul}((q_{2i} + i q_{2i+1}), (\cos m\theta_i + i \sin m\theta_i))$$

或者直接用 `torch.polar` / einsum。Meta 的 Llama, Mistral, Qwen 等都用 RoPE。

---

## 5. KV Cache — Autoregressive inference 的核心优化

KV Cache 不是 paper, 是工程优化。但笔记里把它和 RoPE 放一起讲, 因为 RoPE 在 inference 上的实现跟 cache 强耦合。

### 标准 caching

Autoregressive generation 第 $t$ 步:
- 新 token 的 embedding $x_{new} \in \mathbb{R}^{1 \times d_{model}}$
- 计算 $k_{new} = x_{new} W_K \in \mathbb{R}^{1 \times d_k}$, $v_{new} = x_{new} W_V$
- 拼接: $K_{cached} = [K_{cached}; k_{new}] \in \mathbb{R}^{t \times d_k}$
- 计算 $Q_{new} = x_{new} W_Q$
- Attention: $\text{softmax}(Q_{new} K_{cached}^\top / \sqrt{d_k}) V_{cached}$

避免了对前面 $t-1$ 个 token 重算 $K, V$。复杂度从 $O(t^2 d)$ 降到 $O(t d)$ per step。

### Memory 占用

每个 layer 缓存 $K, V$, 总量:

$$\text{Memory} = 2 \cdot n_{layer} \cdot L \cdot d_{model} \cdot \text{bytes\_per\_param}$$

对 Llama-2-7B ($n_{layer}=32, d_{model}=4096$, fp16): 每 1000 token cache 约 0.5 GB。1M context 下 KV cache 就是 ~500 GB, 这就是为什么 long context inference 极贵, 也是 Infini-attention / quantized KV cache / paged attention 这些工作的动机。

### RoPE 上的 cache 实现

注意 RoPE 是位置依赖的 rotation, cache 时存什么有两种做法:

1. **Cache rotated K**: 每次 generation 时只对新 K 做 rotation by position $t$, 直接 concat。Cache size 不变, 推理快。**但 prefix 不能改 position** — 如果要做 sliding window 改 position, 麻烦。
2. **Cache raw K, attention 时 apply rotation**: 灵活但每次 attention 多算 rotation, 慢。一般不用。

主流实现 (vLLM, TGI, FlashAttention) 都是 cache rotated K。

### 进阶优化方向

- **PagedAttention / vLLM** (https://arxiv.org/abs/2309.06180): KV cache 分 page, 解决 fragmentation。
- **KV cache quantization** (https://arxiv.org/abs/2309.15217, KIVI): 把 K/V cache 量化到 2-bit/4-bit, 内存减半。
- **H2O (Heavy-Hitter Oracle)** (https://arxiv.org/abs/2306.14048): eviction policy, 丢掉不重要的 KV。
- **StreamingLLM** (https://arxiv.org/abs/2309.17453): 保留 attention sink + sliding window, 实现 "无限长度生成"。

---

## 6. Mixtral of Experts (MoE)

**Paper**: Jiang et al. (Mistral AI), 2024
**arXiv**: https://arxiv.org/abs/2401.04088
**模型权重**: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1

### 架构

每个 transformer block 的 FFN 被替换成 $N=8$ 个 expert FFN。每个 token 经过 router 选 top-$K=2$ 个 expert, 其他 expert 不激活。

**Router / gating function**:

$$G(x) = \mathrm{TopK}(\mathrm{softmax}(W_g x), K=2)$$

变量:
- $x \in \mathbb{R}^{d_{model}}$: token embedding (输入 FFN 的 hidden state)
- $W_g \in \mathbb{R}^{N_{experts} \times d_{model}}$: router weight, 输出 $N_{experts}=8$ 维 logits
- $\mathrm{softmax}$: 把 logits 转成 probability
- $\mathrm{TopK}$: 选概率最大的 2 个 expert, 其余置 0

**Expert output**:

$$y(x) = \sum_{i \in \mathcal{S}(x)} G_i(x) \cdot E_i(x)$$

变量:
- $\mathcal{S}(x) = \mathrm{TopK}$ 选中的 expert index 集合, $|\mathcal{S}| = 2$
- $G_i(x) \in \mathbb{R}$: 第 $i$ 个 expert 的 gating weight (scalar, 经 softmax 归一化)
- $E_i(x) \in \mathbb{R}^{d_{model}}$: 第 $i$ 个 expert FFN 的输出向量
- $y(x) \in \mathbb{R}^{d_{model}}$: 加权融合后的输出

### 参数量 vs active 参数

Mixtral 8x7B:
- 总参数 (8 个 expert × 每个约 5B + attention/router 共享): ~47B
- 每 token 激活参数 (2 expert + attention/router): ~13B
- Inference FLOPs 接近 13B dense model

这是 MoE 的核心 trade-off: **训练时 memory 大, 推理时 compute 小**。训练要 load 全部 expert, 推理只 forward 2 个。

### Load balancing loss

如果不加约束, router 容易 collapse 到几个 expert (rich-get-richer)。Mixtral 用 auxiliary loss:

$$\mathcal{L}_{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

变量:
- $f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\{i \in \mathcal{S}(x_t)\}$: fraction of tokens dispatched to expert $i$ (实际选中频率)
- $P_i = \frac{1}{T} \sum_{t=1}^{T} \mathrm{softmax}(W_g x_t)_i$: 平均 router probability for expert $i$ (软分配)
- $N$: expert 数量
- $\alpha$: aux loss 权重, 通常 0.01

直觉: 当所有 expert 均匀使用时, $f_i = P_i = 1/N$, $L_{aux} = N \cdot N \cdot (1/N)^2 = 1$ (最小值)。当某个 expert dominate, $f_i \to 1, P_i \to 1$, $L_{aux} \to N \cdot 1 \cdot 1 = N$ (最大)。这个 loss 鼓励均匀分布。

### 实验观察

笔记里说 "no patterns of expert on a topic"。Mistral 论文确实做了 expert specialization 分析, 发现:
- **token-level 而非 sequence-level routing**: 同一句话里不同 token 被路由到不同 expert
- **syntactic 而非 semantic pattern**: 比如某些 expert 倾向处理 syntax token (the, of), 某些处理 content token, 没有明显的 "代码 expert" / "数学 expert" 分工
- 这跟 ST-MoE (https://arxiv.org/abs/2202.08906) 在 encoder 上观察到的 topic-level specialization 不太一样, 可能是 decoder autoregressive + top-k 的特殊性

### 性能对比 (Mixtral 8x7B vs Llama 2 70B)

| Benchmark | Mixtral 8x7B | Llama 2 70B | Llama 2 13B |
|---|---|---|---|
| MMLU | 71.8 | 69.9 | 54.8 |
| HumanEval | 40.2 | 32.3 | 18.3 |
| GSM8K | 58.4 | 56.8 | 28.7 |
| Inference FLOPs/token | ~13B | ~70B | ~13B |

Mixtral 用 13B 的 inference compute 达到 70B dense 模型的水平, 这是 MoE 的核心 win。

---

## 整体脉络 — 这 6 篇 paper 在讲什么

把这份笔记的 6 个 topic 串起来看, 它们其实在解 Transformer 的几个 fundamental bottleneck:

1. **Computation per token 是固定的** → Pause token 给模型 "多算一会儿" 的能力。
2. **单向生成不适合 infill** → Meet in the Middle 双向建模。
3. **Context length 受限于 attention 的 $O(L^2)$ 和 KV cache 的 $O(L)$ memory** → Infini-attention 用 bounded compressive memory 突破。
4. **位置编码泛化差** → RoPE 用相对位置 + 多 frequency band。
5. **Inference 的 KV cache 内存爆炸** → 工程优化 (paged attention, quantization)。
6. **Dense model compute 浪费在所有 token 上** → MoE 让 compute 只花在相关 expert 上。

更深层的 theme: **Transformer 的 "uniform compute per token + uniform attention over context + dense FFN" 设计正在被逐个解构**。未来 LLM 很可能是 "MoE for FFN + sparse/linear attention for long context + adaptive compute for depth + RoPE-style relative position + pause/CoT for reasoning" 的组合体。Gemini 1.5 / GPT-4 / Claude 3 这些 production 模型的内部架构很可能已经混合了上述多个 idea。

### 进一步阅读 web links

- **Pause Tokens**: https://arxiv.org/abs/2310.02226
- **Meet in the Middle**: https://arxiv.org/abs/2403.07576
- **Infini-attention**: https://arxiv.org/abs/2404.07143
- **RoPE / RoFormer**: https://arxiv.org/abs/2104.09864
- **RoPE 作者博客 (中文, 最权威解释)**: https://kexue.fm/archives/8265
- **Mixtral of Experts**: https://arxiv.org/abs/2401.04088
- **Universal Transformer (adaptive computation 的鼻祖)**: https://arxiv.org/abs/1807.03819
- **Memorizing Transformer (kNN external memory)**: https://arxiv.org/abs/2203.12151
- **Compressive Transformer**: https://arxiv.org/abs/1911.05507
- **Mamba (SSM 替代 attention)**: https://arxiv.org/abs/2312.00752
- **vLLM / PagedAttention**: https://arxiv.org/abs/2309.06180
- **StreamingLLM (attention sink)**: https://arxiv.org/abs/2309.17453
- **H2O (KV eviction)**: https://arxiv.org/abs/2306.14048
- **YaRN (RoPE length extrapolation)**: https://arxiv.org/abs/2309.00071
- **ST-MoE (encoder MoE specialization)**: https://arxiv.org/abs/2202.08906
- **Karpathy nanoGPT (RoPE 实现 reference)**: https://github.com/karpathy/nanoGPT

如果你想深挖其中某篇 (尤其 Infini-attention 的 delta rule 跟现代 Hopfield network 的关系, 或者 RoPE 在 NTK-aware scaling 下的数学), 我可以再展开。
