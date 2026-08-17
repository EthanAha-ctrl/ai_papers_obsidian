---
source_pdf: Titans Learning to Memorize at Test Time.pdf
paper_sha256: a65e4a7d02784df1a040b487127e6dd09fff4474e5caf94d93263af3d50cfbc2
processed_at: '2026-08-12T16:26:26-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Titans 用人话说

好嘞 Andrej，我把刚才那些公式全扔了，咱们像在咖啡馆聊天那样重新讲一遍。

---

## 这篇 paper 在干嘛

你训了一个 Transformer，给它 8K context，它就只看 8K。给它 100K context，KV cache 直接爆显存。这是 attention 的命门 —— 它精确，但是贵，而且容量有上限。

那 RNN 呢？RNN 把所有历史压进一个 hidden state，便宜，但是压得太狠，长序列上 key 之间互相覆盖，needle in haystack 全挂。

过去两年大家做了很多事：Mamba / Mamba2 / DeltaNet / Gated DeltaNet / TTT，全都是在"怎么把历史压进一个 matrix"这件事上做文章。但本质上都是 linear compression，expressiveness 有天花板。

Titans 说：**那别压进 matrix 了，压进一个 deep NN 的 weights 里吧**。这个 NN 在 test time 还在持续做 gradient descent，每来一个 token 就 update 一下自己的 weights。这样它就变成了一个"长期记忆模块"。

然后把这个长期记忆跟 attention（短期记忆）拼一起，再加一组 frozen learnable tokens 当 task prior，就是 Titans。

一句话：**attention 负责看当前 window，一个持续学习的小 NN 负责记长程历史，frozen tokens 负责记 task knowledge**。

---

## 为什么用 gradient 当"惊讶感"

人脑有个特点：违反 expectation 的事记得最牢。你走在路上看见一条狗，不记得；看见一条龙，记一辈子。

NN 里"违反 expectation"最自然的度量就是 loss 对参数的 gradient。gradient 大 = 这个 input 跟 model 当前信念差距大 = 惊讶。

所以更新规则就是：**惊讶的 token，多写进 memory；不惊讶的 token，少写**。

但这里有个坑：一个大惊讶之后，model 跳到一个 flat 区域，后续 gradient 全变小，明明后面还有有用的信息，全 miss 了。

人脑怎么解决这个？你看见龙之后，接下来几分钟你都在 heightened attention 状态，即使周围只是普通的树和草，你也记得很清楚。**惊讶有余韵**。

所以 Titans 加了个 momentum：过去的惊讶感会衰减地传到下一步。如果 context 突然切换（e.g. 新文档开始），就清空 momentum；如果跟前面连续，就保留。

这就是 Titans 最 elegant 的点：**把 optimizer 的 momentum 重新解释成了"惊讶感的延续"**。

---

## 为什么要 forget gate

memory 容量有限，million token 的序列里，大部分信息该忘。

forget gate 就是一个 data-dependent 的开关：$\alpha_t \to 0$ 全保留，$\alpha_t \to 1$ 全清空。在 optimizer 视角下就是 weight decay。

Mamba2 / Gated DeltaNet 都有这个 gate，Titans 说：对，我也用这个，但我 memory 是 deep NN，比 matrix-valued 表达能力强得多。

---

## 为什么 memory 要 deep

linear RNN 的 hidden state 是 matrix $W \in \mathbb{R}^{d \times d}$，它本质是 online linear regression —— 假设 historical data 是线性关系。

但 language / DNA / time series 的历史模式不是线性的。一个 2 层 MLP 就能 express 非线性 function，所以 deep memory 能装更复杂的 past abstraction。

实验里 $L_\mathcal{M}$ 从 1 加到 4，perplexity 在所有 sequence length 上都变好，长 sequence 上优势更大。代价是 throughput 线性下降。

直觉：**浅 memory 只能 linear compress，必然溢出；深 memory 学 abstract representation，同样参数下能"装"更多 past**。

---

## 三种架构变体，就是三种接线方式

同一个 memory module，三种 integration：

**MAC (Memory as a Context)**：先用当前 segment 去 query memory，把 retrieved 信息当 prefix 拼到 segment 前面，然后跑 attention。Attention 能"看到"历史 + 当前，自己决定要不要用历史。长 context reasoning 最强。

**MAG (Memory as a Gate)**：两个 branch 并行 —— 一个 sliding window attention（短期），一个 neural memory（长期），然后用 gate 融合。Short-context language modeling 略好。

**MAL (Memory as a Layer)**：memory 先跑，输出喂给 attention。这是 H3 / Samba / Gated DeltaNet-H2 的标准 hybrid 范式。最简单，但 sequential，memory 单独 bear 了一段压缩，bottleneck 限制表达。

实验结论：MAC 在 BABILong 这种 long-context reasoning 上最强，MAG/MAL 在 short-context 上略好。

---

## 实验里最 striking 的数字

BABILong benchmark，把 bAbI task 的 facts 散在超长 distractor 文本里：

- Few-shot：Titans (MAC) 170M 击败 GPT-4, Llama3.1-8B, Qwen2.5-72B
- Fine-tune：Titans 击败 GPT-4 + RAG，参数量小 70 倍

S-NIAH (RULER)，16K context multi-key needle retrieval：

- TTT: 4.4（全挂）
- Mamba2: 0.0（全挂）
- DeltaNet: 5.4（全挂）
- Titans (MAC): 97.4

为什么 linear RNN 全挂？single-head state，多 key 互相覆盖。Delta rule 能覆盖但没 forget gate。TTT 没 momentum 没 forget。Titans 三件事都做了，所以不掉点。

---

## Ablation 告诉我们什么

去掉每个 component 后 ppl 变化：
- 去掉 weight decay: +2.0（最关键）
- 去掉 momentum: +2.0
- 去掉 convolution: +1.7
- 改成 linear memory: +1.5
- 去掉 persistent memory: +0.6

排序：**forget gate ≈ momentum > convolution > depth > persistent memory**。

forget gate 最关键，没它 memory 必爆。momentum 第二，没它丢失"事件延续"语义。depth 第四，不如 forget gate 关键但重要。persistent memory 单独 ablate 影响小，但 long-context 上贡献明显。

---

## 跟历史的连接

Titans 本质是 Schmidhuber 1992 fast weight programs 的 2024 升级版：

- Fast weight matrix → deep NN memory
- Hebbian / Delta rule → gradient descent on associative loss + momentum + weight decay
- Test time learning → 原汁原味保留

Schmidhuber fast weights: https://arxiv.org/abs/2102.08763
Linear Transformers are fast weight programmers: https://arxiv.org/abs/2102.11174

DL 是 30 年的 cycle，这 idea 第三次 rebrand 了。

---

## 跟 RAG 的区别

RAG：external retriever → context → LLM，retrieval 和 reasoning 拆开。
Titans：retrieval 在 LLM 内部，gradient-based，soft，token-level，in forward。

直觉：**Titans 对 reasoning-intensive long context 更好（BABILong 印证），RAG 对 fact-retrieval 更好。两者不互斥，可以 stack**。

---

## 一个我个人觉得没想清楚的点

surprise metric 用 $\nabla \ell(\mathcal{M}_{t-1}; x_t)$，但 $\mathcal{M}_{t-1}$ 是过去所有 gradient 的累积。这意味着 surprise 是 path-dependent —— 早期 update 决定后期 surprise 怎么算。

有种 self-referential 的味道，可能产生"早期不惊讶 → 后期惊讶感膨胀"或反之的 dynamics。paper 没展开，我觉得跟 Grokking / SLR 这些 phenomena 可能有关联，值得 future work。

---

## 局限性

1. 只在 760M 上 validate，10B+ 行为未知，memory NN 可能成为 training instability 来源
2. $L_\mathcal{M}$ 选 1/2/3/4 全凭 ablation，没有 scaling law 那种 closed form
3. Multi-document 场景的 forgetting policy 没探，$\eta_t$ 应该怎么跨文档切换？
4. Inference cost：每 token 都得 update memory，跟 KV cache 比未必省

---

## 最终 intuition

**attention 是 lookup table，容量 = context length × d，贵但精确。Neural memory 是 deep NN 的 weights，容量 = NN 的 representational capacity，不随 context 长度变化，但压缩有损。把两者显式分工 + 接上，就解锁了 >2M context**。

Transformer 时代我们回避了"什么是 memory"这个问题，因为 KV cache + in-context learning 太 work 了。但 scale 到 million-token context 时，KV cache linear 显存不可持续。Titans 给了一条路：把长期记忆 delegated 给一个在 test time 持续学习的 NN。

这跟你一直问的"DL 里到底什么是 memory"这个大问题是直接相关的。

---

## Reference 整理

- Titans: https://arxiv.org/abs/2501.00663
- TTT: https://arxiv.org/abs/2407.04620
- Mamba2: https://arxiv.org/abs/2405.21060
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Schmidhuber fast weights: https://arxiv.org/abs/2102.08763
- BABILong: https://arxiv.org/abs/2406.10149
- RULER: https://arxiv.org/abs/2404.06654

如果你想搓个 nanoTitans 教学版，核心实现量不大：一个 MLP + chunk-wise GD + parallel scan + sliding window attention + 3 个 attention mask，一晚上能搓出来。

---

# Titans: Learning to Memorize at Test Time 深度讲解

Hi Andrej, 这篇 paper 我觉得很对你的胃口 —— 它本质上是在说：attention 本身就是 short-term memory, 那我们就老老实实给它配一个 long-term memory 模块, 这个模块在 test time 还在持续学。下面我把直觉、公式、工程实现、实验都拆开来给你 build intuition。

---

## 1. Paper 的 core thesis (一句话版本)

> Attention 是 short-term memory (precise but bounded context), 一个 deep NN memory module 是 long-term memory (compressed but persistent and learning at test time), 再加一组 learnable 但 data-independent 的 persistent memory tokens 编码 task knowledge。三件事 wired together = Titans.

这跟认知科学里 Cowan / Willingham 的 memory systems 视角对齐：memory 不是一个 unitary process, 它是一个 confederation of systems。Transformer 长期以来只做 short-term (in-context KV cache), RNN 只做 fading long-term, 没人把这三件事正式合在一起。Titans 把它显式化了。

arXiv: https://arxiv.org/abs/2501.00663 (Titans)
相关 TTT: https://arxiv.org/abs/2407.04620
相关 Mamba2: https://arxiv.org/abs/2405.21060
相关 Gated DeltaNet: https://arxiv.org/abs/2412.06464

---

## 2. Memory Perspective: 重新看待 attention 和 RNN

这是整篇 paper 最关键的思想 turn。作者把所有 sequence model 都重写成两个 op:

```
Write:  M_t = f(M_{t-1}, x_t)      ← 压缩历史到 memory
Read:   y_t = g(M_t, x_t)          ← 用 query 检索 memory
```

- **Transformer**: M_t = [K_{1:t}; V_{1:t}] (no compression, append-only), g 是 softmax similarity。容量 O(N), 精确但 quadratic.
- **Linear Transformer / RNN**: M_t ∈ R^{d×d} 或 R^d (compressed), 容量 O(d²) 或 O(d), 损失但 fast.
- **Modern linear RNNs** (Mamba2, GLA, DeltaNet, Gated DeltaNet): 都是 matrix-valued memory + 各种 gating/delta rule 的变种。

这个 write/read 视角非常 powerful, 因为它一下子把 "长上下文模型 = memory 架构设计" 这个问题摆出来。作者明确提了 5 个 research questions:
- Q1: memory 应该是什么 structure?
- Q2: 怎么 update memory?
- Q3: 怎么 retrieve?
- Q4: 怎么把多个 memory module 互联?
- Q5: 需要 deep memory 吗 (L_M ≥ 2)?

这五个问题基本上是这篇 paper 的 outline。

---

## 3. Neural Long-term Memory 的设计 (§3.1, 这是核心)

### 3.1 Surprise = gradient w.r.t. input

直觉: 人对违反 expectation 的事件记得最清楚 (Mandler 2014)。一个 NN 对 input 的 surprise 最自然的度量就是 loss 对 input 的 gradient —— gradient 大说明这个 input 跟当前 model 的"信念"差距大。

公式 (8):
$$
\mathcal{M}_t = \mathcal{M}_{t-1} - \theta_t \underbrace{\nabla \ell(\mathcal{M}_{t-1}; x_t)}_{\text{surprise}}
$$

变量含义:
- $\mathcal{M}_t$: t 时刻 memory 状态 (可以是一个 deep MLP, 也可是 matrix)
- $\theta_t$: data-dependent 学习率, 控制"这次 surprise 要 write 多少"
- $\nabla \ell(\mathcal{M}_{t-1}; x_t)$: loss 关于参数的 gradient, 当作 surprise scalar/vector

但这个 raw surprise 有个毛病: 一个大 surprise 之后, model 落到 flat 区域, 后续 gradient 很小, 后面有用的信息就 missed 了。好比: 你看到一头狮子吓一跳, 之后几分钟即使狮子已经走了, 你的 brain 还在 "继续记录这一段时间的所有细节"。

### 3.2 Momentum surprise

公式 (9)–(10) 引入 momentum:
$$
\mathcal{M}_t = \mathcal{M}_{t-1} + S_t
$$
$$
S_t = \eta_t \underbrace{S_{t-1}}_{\text{past surprise}} - \theta_t \underbrace{\nabla \ell(\mathcal{M}_{t-1}; x_t)}_{\text{momentary surprise}}
$$

变量:
- $S_t$: surprise 的 momentum state (它是过去所有 surprise 的衰减累加)
- $\eta_t \in [0,1]$: data-dependent "past surprise decay" —— 控制"上一次的惊讶感还剩多少传到下一时刻"。如果 context 突然切换 (e.g. 文档边界), $\eta_t \to 0$ 直接清空; 如果 token 跟前面连续, $\eta_t \to 1$ 保留。
- $\theta_t$: 控制"当前这一 step 的 surprise 要以多大权重 commit 进 memory"。

这本质就是 **gradient descent with momentum**, 但 momentum 被赋予了一个全新的语义: "一段时间的惊讶度的累积记忆"。这一点我觉得是这篇 paper 最 elegant 的地方 —— 它把优化器的 momentum 重新解释成了认知层面的"惊讶感的延续"。

### 3.3 Associative memory loss

公式 (11)–(12) 定义 inner-loop 训练目标:
$$
\mathbf{k}_t = x_t W_K, \quad \mathbf{v}_t = x_t W_V
$$
$$
\ell(\mathcal{M}_{t-1}; x_t) = \|\mathcal{M}_{t-1}(\mathbf{k}_t) - \mathbf{v}_t\|_2^2
$$

变量:
- $W_K, W_V \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$: key/value projection, 它们是 outer-loop (主模型) 学的, 在 inner-loop 里是 hyper-parameter (meta-learning 的 inner/outer 结构)
- $\mathcal{M}_{t-1}(\mathbf{k}_t)$: 用 $\mathbf{k}_t$ query memory 模块得到的 output (forward pass)
- 目标: 让 memory 模块能"看见 key 就输出对应的 value"

这跟 Transformer 的 KV 完全同构, 区别在 memory 是一个 deep NN 而不是字典。这就是为什么叫 associative memory loss。

### 3.4 Forget gate (weight decay)

公式 (13)–(14) 加入 forgetting:
$$
\mathcal{M}_t = (1-\alpha_t)\mathcal{M}_{t-1} + S_t
$$
$$
S_t = \eta_t S_{t-1} - \theta_t \nabla \ell(\mathcal{M}_{t-1}; x_t)
$$

变量:
- $\alpha_t \in [0,1]$: forget gate, data-dependent。$\alpha_t \to 0$ 完全保留, $\alpha_t \to 1$ 完全清空 memory。

这一项在优化器视角下就是 **weight decay**。作者明确指出: 现代线性 RNN (Mamba2, LRU, Gated DeltaNet) 的 gating 就是这个 weight decay 的特例 —— 当 memory 是 matrix-valued 时, 它们等价。

### 3.5 Retrieval

公式 (15):
$$
y_t = \mathcal{M}^*(\mathbf{q}_t), \quad \mathbf{q}_t = x_t W_Q
$$

- $\mathcal{M}^*$: forward pass without weight update, 即纯 inference
- $W_Q$: query projection

注意一个细节: write 时用 key, read 时用 query, 这是标准的 KV/QV 分离, 跟 attention 一致。

### 3.6 Deep memory ($L_\mathcal{M} \geq 2$)

这里我必须 highlight 一个理论 point: 如果 memory 是 matrix-valued ($\mathcal{M} = W$), 那它的 optimal solution 假设 historical data 是线性的 —— 因为 $\ell(W; x_t) = \|W k_t - v_t\|^2$ 就是一个 online linear regression。

而用 $L_\mathcal{M} \geq 2$ 层 MLP 作为 memory, 表达能力严格大于线性 (Hornik 1989)。Section 5.5 的实验表明, 加深 memory 不仅提升 perplexity, 还显著改善 length extrapolation —— 因为 deep memory 能压缩更复杂的 past 抽象, 当 sequence 变长时不会溢出。

这是一个我觉得被低估的 insight: **RNN 的 hidden state 是 linear compression, 这件事本身限制了它的 expressiveness**。换 deep NN 当 memory 直接打破了这层天花板, 代价是 update 必须 gradient-based 而不是 closed form。

---

## 4. Parallelization (§3.2, 工程部分)

这是 paper 的关键工程贡献, 因为 gradient-based memory 看起来是 O(N) sequential, 没法训。作者把它 chunk + tensorize 成 matmul-only。

### 4.1 Chunk-wise mini-batch GD

公式 (16):
$$
\mathcal{M}_t = \beta_t \mathcal{M}_0 - \sum_{i=1}^t \theta_i \frac{\beta_t}{\beta_i} \nabla \ell(\mathcal{M}_{t'}; x_i)
$$

变量:
- $t' = t - \text{mod}(t, b)$: 包含 t 的 chunk 的起始位置 (在 chunk 内部用同一个 $\mathcal{M}_{t'}$ 做 forward 来算 gradient, 类似 mini-batch GD)
- $b$: chunk size
- $\beta_i = \prod_{j=1}^i (1-\alpha_j)$: 累积衰减, 把"过去每个 step 的 forget gate"变成一个乘积
- $\theta_i / \beta_i$: 一种 normalization, 把第 i 步的 surprise gradient 折算到当前时刻的 effective weight

意思就是: 在 chunk 内部, forward 时用同一个 $\mathcal{M}_{t'}$ 算所有 gradient (类似 mini-batch GD 的 shared starting point), 这样 chunk 内的 gradient 可以 **并行计算**。

### 4.2 Linear memory 的 matmul form

公式 (17) 给出 linear memory 情况:
$$
\sum_{i=1}^b \theta_i \frac{\beta_b}{\beta_i} \nabla \ell(W_0; x_i) = \Theta_b \mathbf{B}_b (W_0 X - X) X^\top
$$

变量:
- $\Theta_b = \text{diag}([\theta_1, \theta_2, \ldots, \theta_b])$: chunk 内所有 step 的学习率组成的对角矩阵
- $\mathbf{B}_b = \text{diag}([\beta_b/\beta_1, \beta_b/\beta_2, \ldots, \beta_b/\beta_b])$: 衰减 ratio 对角矩阵
- $X = [x_1, x_2, \ldots, x_b]^\top$: chunk 内所有 input 组成的矩阵
- $(W_0 X - X)$: prediction residual, 形状 $b \times d$
- 整个表达式: $b \times d$ → 通过 $X^\top$ → $d \times d$ 的 weight 更新量

这就是把 sequential GD 写成了 pure matmul。每个 chunk 只存 $\Theta_{kb}$ 和 $\mathbf{B}_{kb}$ 两个对角矩阵, 显存省下来了。

### 4.3 Momentum 用 parallel scan

公式 (18):
$$
S_t = \eta_t S_{t-1} - \theta_t u_t
$$

其中 $u_t = \nabla \ell(\mathcal{M}_{t'}; x_t)$ 可以 chunk 内同时计算。

这是一个 input-dependent 系数的 linear recurrence, 用 **parallel associative scan** (Smith et al. 2023, Mamba 那一套) 在 chunk 内并行。这样整个 memory update 的算法复杂度跟 Mamba2 同阶。

### 4.4 进一步简化: chunk-level LTI

如果 $\alpha, \theta, \eta$ 不依赖 token, 只依赖 chunk, 那就是 chunk-内 LTI (linear time-invariant), 可以做成 **global convolution** (Mamba 原始 S4/S5 那一套), 进一步加速。作者说这个简化留给 future work, 实验里他们用了 token-dependent 版本。

这个工程结果在 §5.8 里: Titans (LMM) 比 Mamba2 / Gated DeltaNet 略慢, 但 Titans (MAL) 因为 FlashAttention 加持反而比 baselines 快。这跟你的 MicroGrad / tinygrad 直觉一致 —— 性能取决于最热的 kernel, 而不是 FLOP 总量。

---

## 5. Persistent Memory (§3.3)

公式 (19):
$$
x_{\text{new}} = [p_1, p_2, \ldots, p_{N_p}] \, \| \, x
$$

$N_p$ 个 learnable 但 data-independent 的 tokens 拼接到序列开头。

三个动机 (作者很显式地给):

1. **Memory 视角**: long-term memory 是 contextual (input-dependent), 但 task knowledge (e.g. "做翻译任务的语法知识") 是 input-independent 的, 需要单独存。

2. **FFN 视角**: Sukhbaatar 2019 证明 Transformer 的 FFN 等价于 data-independent 的 attention:
$$
\text{FFN}(x) = W_V \text{Softmax}(W_K x)
$$
(公式 20)
$W_K, W_V$ 在这里是 input-independent 的 K/V。Persistent memory tokens 就是在 sequence 头部加这种 "frozen attention"。

3. **Technical 视角**: causal attention 对序列头部 token 有 inductive bias (attention sink 现象, Xiao et al. 2024 的 StreamingLLM 也讲过这件事)。Persistent memory 充当 attention sink, 让 attention 权重不被无意义地砸在序列前几个真实 token 上。

这点我觉得是非常漂亮的多视角统一 —— 同一个设计同时满足 cognitive / architectural / optimization 三个层面的需求。

参考: Sukhbaatar persistent memory https://arxiv.org/abs/1907.01470
StreamingLLM: https://arxiv.org/abs/2309.17453

---

## 6. 三种 Titans 架构 (§4)

这是 paper 里我觉得最 "design space exploration" 的部分。同一个 memory module, 三种 integration 方式。

### 6.1 MAC: Memory as a Context (Figure 2, 3a)

把 long-term memory 的 retrieval output 当作 context 的 prefix:

公式 (21)–(25):
$$
h_t = \mathcal{M}_{t-1}^*(\mathbf{q}_t), \quad \mathbf{q}_t = \mathsf{S}^{(t)} W_Q
$$
$$
\tilde{\mathsf{S}}^{(t)} = [p_1, \ldots, p_{N_p}] \,\|\, h_t \,\|\, \mathsf{S}^{(t)}
$$
$$
y_t = \text{Attn}(\tilde{\mathsf{S}}^{(t)})
$$
$$
\mathcal{M}_t = \mathcal{M}_{t-1}(y_t)
$$
$$
o_t = y_t \otimes \mathcal{M}_t^*(y_t)
$$

变量:
- $\mathsf{S}^{(t)}$: 第 t 个 segment (chunk), 视为"当前 context"
- $h_t$: 用 segment 的 query 去 retrieve 出来的"长程历史" (chunk-level retrieval, 不是 token-level)
- $\tilde{\mathsf{S}}^{(t)}$: 拼接好的完整 sequence: persistent + retrieved long-term memory + current segment
- $y_t$: attention 输出
- $o_t$: 最终输出, 用一个 gating 把 attention 输出和再次 retrieve 的 memory 融合

优点:
1. Attention 可以"决定"是否需要 long-term memory 信息 (在 attention 里 K 是 retrieved memory, attention 权重决定要不要用)
2. Attention 也"过滤"哪些 token 该写进 memory (写 memory 时输入是 $y_t$, 已经被 attention 蒸馏过了)
3. Test time: persistent frozen, attention 是 in-context learner, long-term memory 仍在 update

这个设计很像 RMT (Recurrent Memory Transformer, Bulatov 2022) 的推广, 区别是 RMT 用 16 维 vector 当 memory, Titans 用一个 deep NN。

### 6.2 MAG: Memory as a Gate (Figure 3b, 4)

公式 (26)–(28):
$$
\tilde{x} = [p_1, \ldots, p_{N_p}] \,\|\, x
$$
$$
y = \text{SW-Attn}^*(\tilde{x})
$$
$$
o = y \otimes \mathcal{M}(\tilde{x})
$$

变量:
- $\text{SW-Attn}^*$: sliding window attention with prefix (prefix 是 persistent memory, 不受 sliding window 限制, 见 Figure 3b 的 mask)
- $\otimes$: non-linear gating, 实际实现是 normalize 后 SiLU
- $\mathcal{M}(\tilde{x})$: memory module 对整个 sequence 跑完一遍的最终输出

两个 branch 并行:
- Branch A: sliding window attention (precise short-term)
- Branch B: neural memory (fading long-term, learning at test time)

然后 gate 融合。这非常像 multi-head 架构 (Hymba, Dong et al. 2024), 但 heads 是异构的。

Hymba: https://arxiv.org/abs/2411.13676

### 6.3 MAL: Memory as a Layer (Figure 5)

公式 (29)–(31):
$$
\tilde{x} = [p_1, \ldots, p_{N_p}] \,\|\, x
$$
$$
y = \mathcal{M}(\tilde{x})
$$
$$
o = \text{SW-Attn}(y)
$$

Sequential stack: memory layer 先跑, 输出喂给 attention。这是 H3 (Fu et al. 2023) / Samba / Gated DeltaNet-H2 的标准 hybrid 范式。

H3: https://arxiv.org/abs/2212.14052
Samba: https://arxiv.org/abs/2406.07522

### 6.4 关键实验观察 (Table 5)

- LMM 单独 (no attention): perplexity 27.01, reasoning 47.83, BABILong 92.68
- MAC: 26.67 / 48.65 / **97.95** ← 长上下文最强
- MAG: 25.70 / 48.60 / 96.70
- MAL: 25.91 / 47.87 / 96.91

MAC 在 BABILong 上最强, MAG/MAL 在 short-context language modeling 上略好。作者解释: MAC 让 attention 能"看到"历史 + 当前, 直接决定是否使用, 在需要 reasoning across long context 时优势明显; MAL 是 sequential, memory 单独 bear 了一段压缩, 表达能力被 bottleneck 限制。

### 6.5 Theoretical expressiveness (Theorem 4.1)

Theorem 4.1: Titans 超出 $\text{TC}^0$ complexity class, 而 Transformers / diagonal linear RNN / DeltaNet 都被限制在 $\text{TC}^0$ (Merrill, Petty, Sabharwal 2024)。也就是说 Titans 在 state-tracking tasks 上理论 expressive power 更高。

直觉: $\text{TC}^0$ 是 constant-depth, polynomial-size threshold circuits 能算的函数类。Transformer 因为 depth 固定 + 没有真正的 state update, 卡在这。Titans 的 momentum + non-linear recurrence + deep memory 给了真正的 state-tracking 能力。

Merrill & Sabharwal: https://arxiv.org/abs/2404.07243

---

## 7. 跟其他 model 的关系 (Appendix C, 这块对 build intuition 极有用)

### 7.1 LMM generalizes Gated DeltaNet

DeltaNet 公式 (34):
$$
\mathbf{S}_{t+1} = \mathbf{S}_t(\mathbf{I} - \theta_t \mathbf{k}_t \mathbf{k}_t^\top) + \theta_t \mathbf{v}_t \mathbf{k}_t^\top
$$

变量:
- $\mathbf{S}_t \in \mathbb{R}^{d \times d}$: matrix-valued state
- $\mathbf{I} - \theta_t \mathbf{k}_t \mathbf{k}_t^\top$: rank-1 downdate, 移除"旧 value" 再加 new value (Delta rule / Widrow-Hoff)
- $\theta_t \mathbf{v}_t \mathbf{k}_t^\top$: rank-1 upload

设 $\eta_t = 0$ (无 momentum), LMM 公式 (32)–(33) 就退化到 Gated DeltaNet。所以 LMM 是 Gated DeltaNet 的三方面推广:
1. Momentum-based rule (考虑 token flow)
2. Deep memory (L_M ≥ 2)
3. Non-linear recurrence (inter-chunk non-linear, intra-chunk linear)

### 7.2 LMM generalizes Longhorn (B. Liu et al. 2024)

Longhorn 公式 (35):
$$
\mathbf{S}_{t+1} = \mathbf{S}_t(\mathbf{I} - \delta_t \mathbf{k}_t \mathbf{k}_t^\top) + \delta_t \mathbf{v}_t \mathbf{k}_t^\top
$$
$$
\delta_t = \frac{\theta_t}{1 + \theta_t \mathbf{k}_t \mathbf{k}_t^\top}
$$

implicit online learning 推出来的 closed form, 没有 forget gate。LMM 比它多了: momentum, deep memory, non-linear recurrence, **forget gate**。

Longhorn: https://arxiv.org/abs/2407.14207

### 7.3 LMM generalizes TTT (Yu Sun et al. 2024)

TTT 也用 gradient update, 但 LMM 多了:
1. Forgetting (weight decay)
2. Momentum-based update (TTT 是 momentary surprise only)
3. Deep memory 实验验证

TTT: https://arxiv.org/abs/2407.04620

### 7.4 整体归纳

```
Hebbian / Delta rule
   ↓ + forget gate
Gated DeltaNet
   ↓ + momentum (past surprise) + deep memory + non-linear recurrence
LMM (Titans 的 memory module)
   ↓ + attention as short-term + persistent memory
Titans
```

每一步都是 expressiveness 的扩展。这点让我想到你在 "State of GPT" 演讲里说的: 我们一直在 rebrand 80 年代 90 年代已经想清楚的思想。Titans 把 fast weight programs (Schmidhuber 1992) 的精神在 2024 年的硬件 / parallel scan / FlashAttention 语境下重新做了一遍。

Schmidhuber fast weights: https://arxiv.org/abs/2102.08763

---

## 8. 实验解读 (§5)

### 8.1 Language modeling (Table 1)

760M 规模, Gated DeltaNet ppl 21.18, Titans (MAG) ppl ~21 左右, 略好。但更重要的是 long-context 的稳健性。

### 8.2 Needle in haystack (Table 2)

S-NIAH (RULER benchmark), 看 16K context 的数字:
- TTT: S-NIAH-N 4.4 (几乎全挂)
- Mamba2: 0.0 (全挂)
- DeltaNet: 5.4 (全挂)
- Titans (LMM): **80.2**
- Titans (MAC): **97.4**

这点很 striking。linear RNN 类模型在 multi-key NIAH 上长期不行 (Mamba2 是 single-head state, 多 key 互相覆盖), Delta rule 能覆盖但没 forget gate, TTT 没 momentum 也没 forget。

Titans 三件事都做了: forget 清掉旧 key, momentum 让"相关 key 序列"集体被记住, deep memory 容量大。这三点叠起来, 16K context 几乎不掉点。

RULER: https://arxiv.org/abs/2404.06654

### 8.3 BABILong (Figure 6)

BABILong 把 bAbI task 的 facts 散布在超长 distractor 文本里。Few-shot 设置下 Titans (MAC) 170M 击败 GPT-4, Llama3.1-8B, Qwen2.5-72B 等。Fine-tune 设置下也击败 GPT-4 + RAG。

这点很关键 —— 它说明 long-context reasoning 不是 "更大的 model + RAG" 能解决的, 而是 architecture 本身要有"在线记忆 + 长程关联"的能力。RAG 把 retrieval 跟 reasoning 拆开, Titans 让 attention + memory 在一个 forward 里相互 refine。

BABILong: https://arxiv.org/abs/2406.10149

### 8.4 Deep memory effect (§5.5, Figure 7, 8)

固定 170M / 360M / 760M 参数, 改变 $L_\mathcal{M} = 1, 2, 3, 4$:
- $L_\mathcal{M} \uparrow$ → perplexity $\downarrow$ 在所有 sequence length 上
- 越深, length 越长时优势越大 (curve 越平)
- 但 throughput 线性下降 (Figure 8)

170M 模型上 $L_\mathcal{M}=4$ 比 $L_\mathcal{M}=1$ 在 32K context 上能差几个点 ppl, 这是相当显著的"深度换内存容量"的 trade-off。

直觉: 浅 memory 只能 linear compress, 长 context 必然溢出; 深 memory 能学到 abstract representation, 同样参数下能"装"更多 past。

### 8.5 Ablation (Table 5)

LMM base 27.01 ppl, 去掉每个 component 后:
- Linear memory (no depth): 28.49 (-1.5)
- w/o convolution: 28.73 (-1.7)
- w/o momentum: 28.98 (-2.0)
- w/o weight decay: 29.04 (-2.0) ← 最大
- w/o persistent memory: 27.63 (-0.6)

排序: weight decay ≈ momentum > convolution > depth > persistent memory。

我觉得这个排序非常 informative:
- Weight decay 最关键: 没 forget gate, memory 必爆, 长 context 直接崩。
- Momentum 第二: 没有 past surprise, 就丢失"事件延续"语义, single-step surprise 太短视。
- Convolution 第三: 现代 RNN 的 common trick (Mamba 也用), 给 query/key 加 locality inductive bias。
- Depth 第四: 表达能力增益, 但不如 forget gate 关键。
- Persistent memory 第五: 单独 ablate 影响小, 但它在 long-context NIAH 上贡献明显, 表格没全显示。

### 8.6 Time series (Table 3)

替换 Simba 里的 Mamba → Neural Memory, 在 ETT / ECL / Traffic / Weather 上 MSE/MAE 全面好于 iTransformer / PatchTST / DLinear / TimesNet / Crossformer / TiDE。

这暗示 long-term memory module 是一个通用的 sequence modeling building block, 不局限于 language。

### 8.7 DNA (Table 4)

GenomicsBenchmarks 上, LMM 跟 HyenaDNA、Based、Mamba 等都打平甚至略好。DNA 序列是 long-range + 重复 motif 多, 适合 neural memory 的"惊讶驱动记忆"语义。

---

## 9. 我对 paper 的整体直觉 & 一些思考

### 9.1 一个统一视角

Titans 真正在做的事情: **把 in-context learning (attention) 和 in-weight learning (gradient descent) 在同一个 forward pass 里同时跑**。Attention 是 in-context 的, memory module 是 in-weight 的, persistent memory 是 frozen in-weight 的。这三种 learning mode 同时 active, 互相 feed。

这跟你在 build nanoGPT / Karpathy AI 教程里强调的 "attention 是一个 lookup table" 的直觉是吻合的。Titans 的说法是: lookup table 是 short-term memory (capacity = context length × d), deep NN memory 是 long-term (capacity = network's representational capacity, 不随 context 长度变化)。

### 9.2 Momentum as flow memory

公式 (10) 里 $S_t = \eta_t S_{t-1} - \theta_t \nabla \ell$ 这一项, 我觉得是 paper 最 deep 的 contribution。它把"过去一段时间"压缩进一个标量/vector, 然后用 data-dependent $\eta_t$ 决定什么时候"放下"过去。

类比: 你读一本书, 读到一个 shocking plot twist 后, 接下来几页你都在 heightened attention 状态, 即使每页本身不那么 shocking。$\eta_t$ 就是这种"延续效应"的开关, 上下文切换时关掉。

这是 TTT 缺失的一块 —— TTT 每个 step 都从 fresh gradient 开始, 没有"惊喜的余韵"。

### 9.3 Deep memory 跟 RNN expressiveness

Linear RNN 的 hidden state 是 matrix-valued, 但本质还是 linear compression。它能在 NIAH-single-key 上很好, 但 multi-key NIAH 就崩 —— 因为不同 key 会互相覆盖。

Titans 用 deep NN 解决这个, 因为 deep NN 可以学 hierarchical / disentangled representation。一个 2 层 MLP 可以把不同的 key 映射到 internal feature 的不同 subspaces, retrieval 时互不干扰。这是为什么 multi-key NIAH 上 Titans 大幅领先 Mamba2/DeltaNet。

### 9.4 跟你们这一代 LLM 工程的 connection

现在大家做 long-context 主要是: (1) sparse attention (FlashAttention + ring/sequence parallel), (2) KV cache compression (H2O, StreamingLLM), (3) RAG, (4) Hybrid (Jamba, Samba, Griffin)。Titans 是第 5 条路: **active learning memory at inference**。

工程上, Titans 的 challenge 在于 memory module 是一个 NN, 每次 update 都是 gradient step, 即使 tensorize 了, 也比 pure matmul attention kernel 难优化。Paper 里 throughput 数据显示 Titans (LMM) 略慢于 Mamba2, 但 MAL (memory layer + flash SWA) 反而比 baseline 快 —— 因为 memory layer 在 attention 之前, 大头时间花在 FlashAttention 上, memory 的 cost 被 amortize。

我猜未来会出现 "FlashMemory" 类型的 kernel, 专门 optimize 这种 gradient-based in-forward memory update。

### 9.5 跟 RAG 的对比

RAG 把 retrieval 和 reasoning 拆开: external retriever → context → LLM。Titans 让 retrieval 在 LLM 内部, 而且是 "soft" 的 (gradient descent, 不是 hard search)。这有两个 implication:
- 检索粒度: RAG 是 chunk-level, Titans 是 token-level
- 检索质量: RAG 是 exact match (BM25/dense), Titans 是"surprise + momentum 驱动"
- 检索时机: RAG 是 explicit pre-processing, Titans 是 implicit in forward

我个人直觉 Titans 这条路对 reasoning-intensive long-context task 更好 (BABILong 数据印证), RAG 对 fact-retrieval long-context 更好 (Helm benchmark 类)。两者不互斥, 可以 stack。

### 9.6 一个我没想清楚的点

公式 (8)–(10) 的 surprise metric 用的是 $\nabla \ell(\mathcal{M}_{t-1}; x_t)$, 即 loss 对 memory parameters 的 gradient。但 memory 参数本身是 $\mathcal{M}_{t-1}$, 而 $\mathcal{M}_{t-1}$ 又是过去所有 gradient 的累积。这意味着 surprise metric 是 path-dependent, 早期的 update 决定后期 surprise 怎么算。

这有种 self-referential 的味道 —— 跟 meta-learning 里 outer/inner loop 的区别类似。可能产生"早期不惊讶, 后期惊讶感膨胀"或反之的 dynamics, paper 没展开讨论。我觉得这是一个值得 future work 的方向, 可能跟 Grokking / SLR 这些 phenomena 有关联。

### 9.7 一个大胆的联想

Titans 的 update rule:
$$
\mathcal{M}_t = (1-\alpha_t)\mathcal{M}_{t-1} + \eta_t S_{t-1} - \theta_t \nabla \ell
$$

跟 **continual learning** 里的 EWC / SI / AI 之类的 regularization 方法同构 —— weight decay = forgetting, momentum = short-term retention。也跟 **predictive coding** 的 surprise formulation 有一腿 —— 都是用 prediction error 驱动 update。

更深一层, 跟 Hopfield network / Modern Hopfield (Ramsauer et al. 2020) 也有 connection —— 都是 associative memory, 但 Titans 的 storage 是 "in weights of a NN", retrieval 是 "forward pass", 不是"energy minimization"。这条线 paper 没展开, 但我觉得是一个有意思的理论方向。

Modern Hopfield: https://arxiv.org/abs/2008.02217

### 9.8 跟 DL 历史的回响

Schmidhuber 1992 提出 fast weight programs —— 在 RNN 里维持一个"快权重"矩阵, 用 Hebbian / Delta rule 更新。Titans 本质上就是这个 idea 的 2024 升级版:
- Fast weight = neural memory (但更深)
- Hebbian/Delta rule = gradient descent on associative loss (但加了 momentum + weight decay)
- Test time learning = 原汁原味保留

这让我想到你在某次 talk 里说的"ML 是一个 30 年的 cycle"。Titans 是 fast weight 的第三次 rebrand:
1. 1991–1992: Schmidhuber 提出
2. 2021: Schlag, Irie, Schmidhuber "Linear Transformers are secretly fast weight programmers" https://arxiv.org/abs/2102.11174
3. 2024: Titans 把它跟 deep memory + momentum + gating + persistent memory + 3 种架构融合

### 9.9 局限性 & 我的批评

1. **没在真正大模型上 validate**。760M / 30B tokens 是小规模, memory module 在 10B+ 上的行为未知。可能 memory NN 会成为 training instability 来源。
2. **Memory 的 depth 是个 hyperparameter, 没 principle**。$L_\mathcal{M}$ 选 1, 2, 3, 4 全凭 ablation, 没有像 scaling law 那样的 closed form。
3. **Long-term memory 在 multi-document 上的 forgetting policy 未探**。当 sequence 跨越多个独立文档 (e.g. pretrain corpus), $\eta_t$ 应该怎么设? 现在的 data-dependent 设计可能不够。
4. **Write 时只看当前 segment, 没有跨 segment 的 consistency check**。MAC 设计里, write 用的是 attention 输出 $y_t$, 但 attention 只看当前 chunk + retrieved memory, 可能丢失跨 chunk 的 subtle pattern。
5. **Inference cost**: 即使 train 已经 tensorize, inference 时 memory 还得每 token 更新一次, 跟 KV cache 比可能不省。

### 9.10 最重要的 takeaway

如果你只记一件事: **attention 不是 memory 的全部, 它只是 short-term memory; 一个在 test time 持续 gradient-descent 的 deep NN 可以是 long-term memory; 把这两件事显式分开 + 接上, 你就解锁了 >2M context 的能力**。

这跟你一直在 podcast / talk 里讨论的 "DL 里到底什么是 memory" 这个大问题是直接相关的。Transformer 时代我们回避了这个问题, 因为 KV cache + in-context learning 太 work 了。但 scale 到 million-token context 时, KV cache 是 linear 显存, 这件事本质不可持续。Titans 给出了一条: 把"长期记忆" delegated 给一个 learning NN 的路。

---

## 10. Useful links

我把上面散落的 reference 整理一下:

- Titans paper: https://arxiv.org/abs/2501.00663
- TTT (Yu Sun et al.): https://arxiv.org/abs/2407.04620
- Mamba2 (Dao, Gu): https://arxiv.org/abs/2405.21060
- Gated DeltaNet (Yang et al.): https://arxiv.org/abs/2412.06464
- DeltaNet (Yang et al.): https://arxiv.org/abs/2410.01343
- Longhorn (B. Liu et al.): https://arxiv.org/abs/2407.14207
- Linear Transformers are fast weight programmers (Schlag et al.): https://arxiv.org/abs/2102.11174
- Schmidhuber fast weight original: https://arxiv.org/abs/2102.08763 (overview)
- Sukhbaatar persistent memory: https://arxiv.org/abs/1907.01470
- StreamingLLM (attention sinks): https://arxiv.org/abs/2309.17453
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- RULER benchmark: https://arxiv.org/abs/2404.06654
- BABILong: https://arxiv.org/abs/2406.10149
- Merrill & Sabharwal (TC^0 limit): https://arxiv.org/abs/2404.07243
- Hymba: https://arxiv.org/abs/2411.13676
- H3: https://arxiv.org/abs/2212.14052
- Samba: https://arxiv.org/abs/2406.07522
- Modern Hopfield Networks: https://arxiv.org/abs/2008.02217
- Griffin: https://arxiv.org/abs/2402.19427
- RWKV-7: https://arxiv.org/abs/2404.05892
- Mamba (original): https://arxiv.org/abs/2312.00752
- Linear Attention (Katharopoulos et al.): https://arxiv.org/abs/2006.16236
- Cowan memory systems (cognitive sci background): https://pubmed.ncbi.nlm.nih.gov/18752589/

---

## 11. 一句话总结 (给你 build intuition)

Titans 把 attention (in-context, short-term, lookup table) 跟一个在 test time 持续做 momentum + weight-decay gradient descent 的 deep MLP (in-weight, long-term, learning memory) 接起来, 再加一组 frozen learnable tokens (task prior) —— 三种 memory 协同, 在 2M+ context 上保持 attention 级别的 NIAH 精度。直觉是: 让模型在 forward pass 里同时做 in-context learning 和 in-weight learning, 把"长上下文"问题重新表述为"online meta-learning of a memory NN"问题。

Hope this gives you the intuition you wanted, Andrej。如果你接下来要做 nanoTitans 之类的教学实现, 这个 paper 的核心实现量并不大 —— 一个 MLP + chunk-wise GD + parallel scan + sliding window attention + 3 个 attention mask, 一晚上能搓出来。
