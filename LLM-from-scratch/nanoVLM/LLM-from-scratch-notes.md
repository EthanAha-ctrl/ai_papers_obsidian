---
source_pdf: LLM-from-scratch-notes.pdf
paper_sha256: 957ee394bbd4d6b774ce9897a5c82ef9c7b7fbf751bcb5388cf2d67ca41c8ad3
processed_at: '2026-08-05T15:31:16-07:00'
target_folder: LLM-from-scratch/nanoVLM
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话再过一遍

好，前面那篇是教科书式拆解，这次我用白板讨论的口吻，把这本笔记的"为什么"重新捋一遍。我们在场就当是你 (Karpathy) 和我在白板前画图聊天。

---

## 这本书到底在干嘛

Raschka 做的事情其实很朴素: take 一个 raw text string, 一路推到一个能聊天的 LLM, 全程不调 library, 不抄 HF pipeline, 每一行代码都自己写出来。精神跟你 nanoGPT 完全同源——把 magic 拆成 elementary operations, 让你看到 transformer 其实没多少东西。

跟你的 "Let's build GPT from scratch" YouTube 系列 (https://www.youtube.com/watch?v=kCc8FmEb1nY) 在哲学上是一模一样的, 只是他用一本书的篇幅讲得更系统、更慢。

---

## Tokenization: 就是把 text 切成整数

人读 "the cat sat" 是三个 word, 但 model 不认 word, 只认 integer。所以第一步就是做一个 `str → List[int]` 的 mapping。

这里有个微妙点: 为什么 GPT-2 用 BPE 而不是 word-level vocabulary? 因为 word-level 一遇到 training 时没见过的 word 就抓瞎 (`[UNK]`)。BPE 的 trick 是把 rare word 拆成 subword, 比如 "unbelievable" 拆成 "un" + "believ" + "able", 每一片都在 vocab 里, 永远不会 OOV。

GPT-2 small 的 vocab 是 50,257, 这个数字 = 256 (bytes) + 50,000 (merges) + 1 (end-of-text)。你训练 nanoGPT 用的是 tiktoken (https://github.com/openai/tiktoken), 跟 GPT-2 的 BPE 是同一脉。

还有个容易被忽略的点: GPT 的 BPE 在 merge 之前先做了一道 regex pre-tokenization, 把 raw text 切成 word-like chunk, 防止 BPE 跨 word boundary merge (否则 "dog" 和 "dogs" 里的 "s" 可能被合并到其他 word 的 "s" 上去, 破坏 word structure)。这个 regex 是:

```
r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

参考 GPT-2 encoder.py: https://github.com/openai/gpt-2/blob/master/src/encoder.py

---

## Embedding: 整数变成向量, 同时塞进 position

token ID 是个整数, 但 model 要的是 continuous vector, 所以第一层是 `nn.Embedding(vocab_size, d_model)`, 本质就是一个 `(50257, 768)` 的 lookup table, 第 i 行就是 token i 的 embedding。

但光有 token embedding 还不够, 因为 attention 机制本身对 sequence 顺序无感 (permutation-equivariant)。你把 "the cat sat" 和 "sat cat the" 输进去, attention 输出按同样顺序打乱后完全一样, 这显然不对——自然语言里 word order 是核心信息。

所以要注入 position 信息。原始 transformer 的方案是 sinusoidal PE, 用不同频率的 sin/cos 给每个 position 编一个 code, 加到 token embedding 上:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

变量解释一遍:
- `pos`: token 在 sequence 里的位置 (0, 1, 2, ...)
- `i`: dimension pair 的 index, 第 i 对 (sin, cos) 用同一个 frequency
- `d_model`: embedding 总维度, GPT-2 small 是 768
- `10000`: base, 决定频率范围

为什么是 sum 而不是 concat? 因为 concat 会让 input dim 翻倍到 1536, 所有后续 layer 都要跟着翻倍, 参数量和计算量都上去。Sum 是一个 bet: 假设 position 信息可以"叠加"到 token embedding 的同一 subspace 里而不互相破坏。实践中这个 bet 成立。

你在 9/22 那条推里提的疑虑非常对——sinusoidal PE 在 high dimension 上 frequency 趋零, 几乎变成常数, 等于这一片 dimension 没拿到 position 信息。Hesamation 给的解读是 "balance": 高 dimension 保留 token identity, 低 dimension 拿 position。但更彻底的解法是后来 RoPE (https://arxiv.org/abs/2104.09864) 的思路——通过复数旋转实现 relative position, 不再硬塞 absolute position 进 embedding, 而是在 Q·K 内积时让 rotation 自然带出 relative offset。LLaMA 系列都用 RoPE。

---

## Attention 的核心直觉

这是整本书最关键的一节, 我重点说。

**Self-attention 的几何画面**: 想象 sequence 里每个 token 都同时扮演三个角色:
- 作为 **query** (Q), 它"提问": 我应该关注哪些 token?
- 作为 **key** (K), 它"被检索": 别的 token 怎么找到我?
- 作为 **value** (V), 它"贡献内容": 一旦被关注, 我输出什么?

实现上就是三个 learnable matrix W_Q, W_K, W_V, 把同一个 input embedding X 投影成三种 representation:

```
Q = X · W_Q  ∈ ℝ^(T × d_k)
K = X · W_K  ∈ ℝ^(T × d_k)
V = X · W_V  ∈ ℝ^(T × d_v)
```

然后:

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

拆开看这个公式:
1. `Q · K^T` 是 `(T, T)` 的 score matrix, 第 (i, j) 个元素是 token i 的 query 跟 token j 的 key 的 dot product, 也就是 "token i 觉得 token j 有多相关"
2. `softmax(·)` 沿 last dim 归一化, 让每个 token i 对所有 j 的 attention weight 加起来等于 1, 变成 probability distribution
3. 最后乘 V, 相当于 "按 weight 把所有 token 的 value 加权求和", 得到 token i 的 context vector

直觉: 每个token 通过 attention "环顾四周", 根据相似度收集其他 token 的信息, 然后更新自己的 representation。所以 attention 之后的 embedding 不再是 "this token 的孤立含义", 而是 "this token 在这个 context 下的含义"。

**为什么要 Q/K/V 三个独立 matrix?** 如果直接用 X 不投影, 那 attention score 就是 X·X^T, query 和 key 共享同一组 feature, 一个 token 没法在 "提问" 和 "被检索" 两种 role 下有不同的表现。引入 W_Q, W_K, W_V 等于告诉模型: "你可以在不同 role 下用不同的 feature 子集"。多出来的参数也给了 model 学习复杂 pattern 的 capacity。

**为什么 scale by √d_k?** 这是笔记里最数学化的一段。设 Q_i, K_i ~ N(0, 1) iid, 那 Q·K^T 的元素 S = Σ_i Q_i K_i, 它的方差是 d_k (因为每个 Q_i K_i 期望 0、方差 1, 独立求和方差相加)。std = √d_k。当 d_k = 64 时 std ≈ 8, softmax input 在 ±8 量级会让 distribution 极度 peaked (接近 one-hot), gradient 趋零, 训练就崩了。除以 √d_k 把 std 拉回 1, softmax 处于温和区, gradient 健康。

**Causal mask**: 为了做 next-token prediction, token i 不能看到 token i+1..T。实现上就是在 QK^T 算完之后, 把上三角位置填 `-inf`, softmax 之后这些位置就变成 0, 等价于 "看不到未来"。

笔记里那个 attention weight 的表很直观: "Your" 只能 attend 自己 (weight 1.0); "journey" 能 attend "Your" 和自己 (0.55 + 0.44); 最后一行 "step" 能 attend 全部 6 个, 权重分别是 0.19, 0.16, 0.16, 0.15, 0.16, 0.15。

---

## Multi-Head: 让 model 同时关注多种关系

一个 head 只能学一种 attention pattern。但语言里同时有语法依赖、指代消解、semantic similarity、长程依赖等等, 一种 pattern 显然不够。

Multi-head 的 trick 是把 d_model 维的 Q/K/V 拆成 h 个 d_k 维的 head (GPT-2 small: 768 = 12 × 64), 每个 head 独立做 attention, 最后 concat 起来再过一个 W_O 投影回 d_model。

```
MultiHead = Concat(head_1, ..., head_h) · W_O
head_i   = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
```

每个 head 可以专注一种 subspace 的关系。比如 head 1 学 subject-verb agreement, head 2 学 coreference, head 3 学局部 n-gram, head 4 学全局 topic, 等等。这是 model 的 "diversified attention capacity"。

工程上的优化是: 不用 h 个独立的小 W_Q/W_K/W_V, 而是用一个大的 `(d_model, d_model)` 一次 matmul 拿到所有 head 的 Q/K/V, 然后 reshape 成 `(b, h, T, d_k)`。GPU 上一次大 matmul 比 h 次小 matmul 快得多。你 nanoGPT 里就是这么写的。

参考 Voita et al. 2019 (https://arxiv.org/abs/1905.09418) 的分析: 训练完之后, 很多 head 是冗余的, 可以 prune 掉一大半, model 性能不掉。说明实践中 multi-head 有冗余, 但训练时这个 redundancy 似乎是必要的, 帮助优化。

---

## Transformer Block 的其他零件

一个 GPT block 除了 attention 还有三个东西: LayerNorm、GELU FFN、Residual connection。

### LayerNorm

公式:
```
y_i = (x_i - E[x]) / √(Var[x] + ε) · γ_i + β_i
```

变量:
- `x_i`: feature dim i 的值
- `E[x]`, `Var[x]`: 沿 feature dim 的均值/方差 (对每个 token 独立算)
- `γ_i, β_i`: learnable scale 和 shift, shape `(d_model,)`
- `ε`: 防 0, 通常 1e-5

直觉: 每个 token 经过 LayerNorm 之后, 它的 feature 分布被标准化到 mean 0、var 1 (再被 γ/β rescale)。这稳定了深层网络的训练, 让每个 token 在每一层的数值分布大致一致, gradient 不会爆炸或消失。

GPT 用 Pre-LN: LayerNorm 在 sublayer 之前, residual path 上没有 LayerNorm, gradient 可以一路 skip。原始 Transformer 用 Post-LN, 深层训练容易不稳。LLaMA 用 RMSNorm, 砍掉 mean centering, 只保留 scale, 省一点计算。

### GELU

公式:
```
GELU(x) = x · Φ(x)  where Φ is normal CDF
```

直觉: 可以理解为 "以 Φ(x) 的概率保留 x"。大正值几乎保留 (Φ ≈ 1), 大负值几乎丢弃 (Φ ≈ 0), transition 是 smooth 的。比 ReLU 的硬截断 (负值直接变 0) 更温和, 也保留了小负值的一点信号, 对深网络训练更友好。

你 nanoGPT 里也是用 GELU, 跟 GPT-2 保持一致。LLaMA 改用 SwiGLU (https://arxiv.org/abs/2002.05202), 在 FFN 里多一个 matmul 但效果更好, 是 trade-off compute for quality。

### FFN

```
FFN(x) = Linear_2(GELU(Linear_1(x)))
```

`Linear_1`: d_model → 4·d_model (GPT-2 small: 768 → 3072)
`Linear_2`: 4·d_model → d_model

"Expand then contract": 先把 representation 投到 4× 高维, 应用非线性, 再压回。直觉是高维空间里非线性变换有更多自由度, 可以学更复杂的 transformation, 压回时只保留有用的。

### Residual Connection

```
x_out = x + Sublayer(x)
```

就这一行。但它解决的问题是深网络的 vanishing gradient: 反向传播时 gradient 可以直接 skip 过 sublayer, 让早期层也能拿到健康的 gradient。

可以理解为 model 在每一层做 "soft decision": 要么走 sublayer 改变 representation, 要么走 identity 保持不变。最深网络可以退化成 identity, 不会比浅层差, 所以训练容易很多。

---

## 整个 GPT Block 的数据流

把上面的零件串起来, 一个 block 的前向是这样的:

```
input X  (shape: b, T, d_model)
  │
  ├── LayerNorm → Multi-Head Self-Attention (causal) → +
  │                                                      │
  └──────────────────────────────────────────────────────┘
                          │
                          ↓
  ├── LayerNorm → FFN → +
  │                      │
  └──────────────────────┘
                          │
                          ↓
                       output (shape: b, T, d_model)
```

堆 12 层 (GPT-2 small) 就是完整 transformer。最后接一个 LM head (linear projection 到 vocab size 50257), 输出 logits, softmax 后就是 next-token distribution。

---

## Pretraining 在干嘛

给定 corpus, 把它切成 (input_tokens, target_token) pairs, target 就是 input 的下一个 token。Loss 是 cross-entropy:

```
L = -(1/T) · Σ_t log p(token_t | token_{0..t-1})
```

整个 corpus 都是免费 label, 不需要人工标注, 所以叫 self-supervised。

模型通过 SGD (实践中用 AdamW) 最小化这个 loss, 慢慢学到:
- short-range syntax (语法)
- long-range semantics (语义)
- world knowledge
- reasoning pattern

到了一定 scale (参数量 + 数据量 + 算力), emergent ability 出现——这就是 GPT-3 / Chinchilla / scaling laws 那套故事 (https://arxiv.org/abs/2203.15556)。

**AdamW 的关键**: 比 Adam 多了 decoupled weight decay。Adam 把 weight decay 塞进 L2 regularization 会被 momentum 统计量稀释, AdamW 把它解耦直接减 η·λ·θ, 对 LLM 训练更稳。

Checkpoint 的时候除了 model weights 还要存 optimizer state (m, v), 否则恢复训练冷启动会震荡。

---

## 文本生成的两种模式

### Greedy

每步取 argmax。问题: 容易陷入 repetition loop ("I I I I"), 而且 diversity 低。

### Temperature + Top-p Sampling

```
probs = softmax(logits / T)
```
- T 小: distribution 变 sharp, 趋近 greedy
- T 大: distribution 变 flat, 趋近 uniform

Top-p (nucleus) sampling: 先按 prob 降序排, 累积到刚好 ≥ p 的最小 token 集合里采样。对 sharp distribution 选少几个, 对 flat distribution 选多, 比 top-k 更自适应。

实践中 typical 配置: T = 0.7~1.0, top_p = 0.9~0.95。

---

## Fine-tuning 的两种模式

### Classification

加载预训练 weights, 把 LM head 换成 classification head (d_model → num_classes), 然后用标注数据 fine-tune。可以用 last token 的 output (因为 causal mask 让它看过整个 sequence) 作为 sequence-level representation, 接分类头。

不需要 tune 所有层, 浅层学到通用语言特征, 只 tune 最后几层就够。LoRA 更极端, 全冻结, 只训练注入的小 module (rank r 的 A/B 矩阵), 省显存。

### Instruction Tuning

数据格式从 completion 变成 instruction-response。Alpaca style 用 `### Instruction / ### Input / ### Response`, Phi-3 style 用 `<|user|> ... <|assistant|> ...`。collate function 要处理变长, padding + attention mask。

评估不再像 classification 算 accuracy 那么简单, 要靠 MMLU (多选)、LMSYS Arena (人类 pairwise 比较)、AlpacaEval (GPT-4 评分) 这种 benchmark。

---

## 容易被忽视的工程细节

1. **Model 输出 logits 而非 probs**: 内部 softmax 数值不稳, 而且 loss function 内部已经 stable 做了 log-softmax, 推理时 argmax 在 logits 上等价于在 probs 上。所以 forward 直接 return logits。

2. **model.train() vs model.eval()**: eval 关闭 dropout, BatchNorm 用 running 统计 (LayerNorm 不受影响, 因为它本来就是 per-sample)。训练时忘了切 train、推理时忘了切 eval 是经典 bug。

3. **DataLoader 的 num_workers**: 多进程预取避免 IO bottleneck。pin_memory=True 加速 CPU→GPU transfer。

4. **drop_last=True**: 不完整 batch 丢掉, 避免 batch size 波动 (对 LayerNorm 影响不大, 但保持习惯)。

5. **optimizer.zero_grad()**: 每个 step 开始清上一步 gradient, 防止 unintended accumulation。

---

## 这本书的核心 message

Raschka 在做的事情跟你 nanoGPT 一脉相承: **把 LLM 从 black box 拆成可以一个文件讲完的几十行 PyTorch**。一旦你亲手从零写过一遍 GPT-2, 再看 GPT-4 / Claude / LLaMA, 你会发现 model 本身的核心公式就那么几个, 99% 的复杂度都在工程化 (distributed training, FSDP, paged attention, MoE routing, quantization)。

理解了 from-scratch 这条路径, 你之后看任何 LLM paper 都能很快定位到 "这个工作改了哪一块"——是 tokenizer? PE? attention? FFN? norm? 还是训练 loss? 还是 inference 优化?

这就是 Raschka 和你的共同信念: **build from scratch 是 build intuition 的最快路径**。

---

参考链接汇总:
1. Raschka 书: https://www.manning.com/books/build-a-large-language-model-from-scratch
2. Raschka 代码: https://github.com/rasbt/LLMs-from-scratch
3. 你自己 nanoGPT: https://github.com/karpathy/nanoGPT
4. 你 zero-to-hero 课程: https://karpathy.ai/zero-to-hero.html
5. "Let's build GPT" 那一节: https://www.youtube.com/watch?v=kCc8FmEb1nY
6. Vaswani 2017: https://arxiv.org/abs/1706.03762
7. GPT-2 paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
8. RoPE: https://arxiv.org/abs/2104.09864
9. SwiGLU: https://arxiv.org/abs/2002.05202
10. Voita multi-head 分析: https://arxiv.org/abs/1905.09418
11. Chinchilla scaling: https://arxiv.org/abs/2203.15556
12. LoRA: https://arxiv.org/abs/2106.09685
13. AdamW: https://arxiv.org/abs/1711.05101
14. Nucleus sampling: https://arxiv.org/abs/1904.09751
15. tiktoken: https://github.com/openai/tiktoken

---

# 这份笔记的全景解读

这份文件是 Sebastian Raschka《Build a Large Language Model (From Scratch)》一书的精读笔记，整体走的是 GPT-2 从零复现的路径——从 tokenization、embedding、positional encoding、attention、multi-head、LayerNorm、GELU、FFN、residual，一路到 pretraining、generation、fine-tuning（classification + instruction），最后补一点 PyTorch 工程细节。

我会按笔记的顺序展开，每一个环节都讲深一点，把公式里的变量、上下标的含义拆开，并补一些 Karpathy 自己在 nanoGPT / "Let's build GPT" 里强调过的直觉。

原书主页: https://www.manning.com/books/build-a-large-language-model-from-scratch
官方代码仓库: https://github.com/rasbt/LLMs-from-scratch
Karpathy 自己的 nanoGPT（精神同源）: https://github.com/karpathy/nanoGPT
Karpathy 的 YouTube 系列课程: https://www.youtube.com/playlist?list=PLAqhIrjkxbmWIueI0KH1Ta5r5t6I2GIte

---

## 1. 整体 Pipeline

笔记开篇画了一张大图，把 LLM 的构建拆成几个阶段：

1. **Data preparation**：raw text → tokenizer → token IDs
2. **Embedding**：token IDs → token embedding + positional embedding
3. **Transformer blocks**：堆叠 N 层 (multi-head self-attention + FFN + LayerNorm + residual)
4. **Output head**：最后一层投影回 vocab，得到 logits
5. **Pretraining**：next-token prediction，self-supervised
6. **Fine-tuning**：classification head 或 instruction following

"Large" 的两个维度：模型参数规模 + 训练 corpus 规模。两者一起把 model 推到 emergent ability 出现的 regime。

---

## 2. Tokenization

### 基本流程

- raw text 切成 tokens
- 给每个 token 一个 unique ID
- 加 special tokens：`[BOS]` (begin), `[EOS]` (end), `[PAD]` (batch padding), `[UNK]` (unknown word)

### BPE (Byte-Pair Encoding)

GPT 系列用的 tokenizer 是 BPE，它有一个好处：不需要 `[UNK]`，因为任何 unknown word 都可以拆成 subword。算法流程：

1. 初始化词表为所有 byte / character
2. 统计 corpus 里所有相邻 pair 的频率
3. 合并频率最高的 pair 成新 token，加入词表
4. 重复 2-3 直到达到 `vocab_size` 上限

公式化一点，给定训练 corpus C、词表大小 |V|、迭代次数 T，最终词表 = 初始字符集 ∪ {合并得到的 T 个 subword}。

参考: Sennrich et al., 2015, "Neural Machine Translation of Rare Words with Subword Units" https://arxiv.org/abs/1508.07909
GPT-2 的 BPE 实现（tiktoken）: https://github.com/openai/tiktoken

GPT-2 small 的 vocab size 是 50,257（256 bytes + 50,000 merges + 1 special token）。

### Token Embedding

Embedding layer 本质是一个 lookup table，shape 是 `(|V|, d_model)`。

输入 token ID 序列 `x ∈ ℤ^T` 经过 embedding 查找得到 `E ∈ ℝ^(T × d_model)`。

笔记里特意强调，LLM 可以把 embedding 训成自己的一部分——这跟 word2vec、GloVe 这种 static embedding 不一样，LLM 的 embedding 会被 attention 反复 re-contextualize，所以语义会随上下文动态变化。

参考: Mikolov et al., 2013, word2vec https://arxiv.org/abs/1301.3781
Pennington et al., 2014, GloVe https://nlp.stanford.edu/projects/glove/

---

## 3. Positional Embedding

### 为什么必须注入 position 信息

Attention 机制是 permutation-equivariant 的：把 sequence 打乱，输出（按同样顺序打乱后）完全一致。所以单纯 attention 没法区分 "the cat sat on the mat" 和 "mat the on sat cat the"。

笔记里 Karpathy 风格地问了：为什么不用 RNN / LSTM 这种顺序模型？给出四个理由：

1. **Parallelism vs. Sequential**：parallel 在 GPU 上快得多，RNN 必须按时间步串行
2. **Position as a feature, not a bug**：让模型自己学 position relationship 比硬编码进 RNN 更灵活
3. **Complexity**：把 position 也塞进 RNN 的 hidden state 会让训练变难
4. **Existing approach works**：sinusoidal PE 已经够用

### Sinusoidal PE 公式

原始 Transformer 的方案：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

变量解释：
- `pos`：token 在 sequence 里的位置，从 0 开始
- `i`：dimension 的"pair index"，2i 和 2i+1 是一对 sin/cos
- `d_model`：embedding 总维度，GPT-2 small 是 768
- `10000`：base，控制频率范围

或者更紧凑地写：

```
PE(pos, k) = sin(pos · ω_k)  if k is even
             cos(pos · ω_k)  if k is odd
ω_k = 1 / 10000^(⌊k/2⌋ / (d_model/2))
```

`ω_k` 是 dimension k 对应的角频率。低 dimension 用高频，高 dimension 用低频。

### Sin/cos 的好处

1. **Bounded**：值域 [-1, 1]，数值稳定
2. **Relative position 可线性表达**：对固定 offset δ，
   ```
   PE(pos+δ, 2i) = sin((pos+δ)·ω_k)
                 = sin(pos·ω_k)·cos(δ·ω_k) + cos(pos·ω_k)·sin(δ·ω_k)
   ```
   也就是 `PE(pos+δ)` 是 `PE(pos)` 的线性函数（系数依赖 δ）。这意味着 model 可以通过学习一个线性投影来 attend 到 relative positions。
3. **Extrapolation**：理论上对超出训练长度的 pos 仍然有定义

### 为什么 sum 而不 concat

笔记里给出一个理由：concat 会让 input embedding 维度翻倍到 `2·d_model`，参数量和计算量都增加。Sum 隐含一个假设——position 和 token 信息可以叠加在同一个 subspace。实践中这个假设 work，且更省。

### Karpathy 的疑虑 + Hesamation 的解读

笔记里贴了 Karpathy 9/22/2024 的推文：高频分量在 dimension 超过某阈值后基本被 clip 掉（因为 `10000^(2i/d_model)` 太大，sin/cos 几乎不变），他问是否应该用更缓慢变化的频率。

Hesamation 给了两个角度：
- **平衡**：高 dimension 的"近似不变"让 model 容易识别 token identity；低 dimension 的高频让 model 拿到 position 信息。两边都不被压垮
- **Multi-scale**：低频对应 long-range / general information；高频对应 short-range。模型可以同时捕捉两种 scale

这跟后续 RoPE 的思路是相通的——RoPE 通过复数旋转实现 relative position，是 LLaMA 系列采用的方案。

参考: Su et al., 2021, RoPE https://arxiv.org/abs/2104.09864
ALiBi (Press et al., 2021) https://arxiv.org/abs/2108.12409

---

## 4. Self-Attention 深入

### 历史脉络

- 2014：Bahdanau attention，在 encoder-decoder RNN 里给 decoder 每一步 access 到 encoder 的所有 hidden state https://arxiv.org/abs/1409.0473
- 2017：Vaswani et al., "Attention is All You Need"，砍掉 RNN，纯 attention → Transformer

### Self vs. Cross

- **Self-attention**：query 和 key/value 来自同一个 sequence，模型在 sequence 内部找 relation
- **Cross-attention**：query 来自 decoder，key/value 来自 encoder（典型场景：翻译）

### Simple Attention 的几何直觉

笔记里画了一张图：query 和其他 input 做 dot product，softmax 归一化得到 weight，weight 加权 sum 其他 input 得到 context vector。这相当于"用 query 去检索别的 token，越像权重越大"。

dot product 本身就是 cosine similarity 的 unnormalized 版本（如果向量归一化过就是 cosine）。所以 self-attention 在做的事情可以理解为：**每个 token 用自己当 query，去和所有 token（key）算相似度，再用相似度加权取 value**。

### 引入 Q/K/V 三个矩阵

直接用 input embedding X 做 attention 等价于 `XX^T`，会让 query 和 key 共享同一套 representation，模型没法让一个 token 在"提问"和"被检索"两种 role 下表现不同。引入三个 learnable matrix：

```
Q = X · W_Q ∈ ℝ^(T × d_k)
K = X · W_K ∈ ℝ^(T × d_k)
V = X · W_V ∈ ℝ^(T × d_v)
```

好处：
1. **Decoupling roles**：一个 token 在做 query 时用 W_Q 投影出的特征，做 key 时用 W_K 投影出的特征，做 value 时用 W_V 投影
2. **Extra capacity**：多出来的参数让模型有空间学习 "what to ask" 和 "what to offer"
3. **Implicit bilinear kernel**：`QK^T = X · (W_Q · W_K^T) · X^T`，等价于在 input space 学了一个 bilinear similarity kernel

笔记里有个精彩段落：W_Q、W_K、W_V 在前向传播时是独立的，但通过 backprop 它们互相耦合——attention score 由 W_Q 和 W_K 联合决定，所以它们是 interdependent 的。Backpropagation 是把它们串起来的 orchestrator。

### Scaled Dot-Product Attention 完整公式

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

变量：
- `Q, K, V`：query / key / value matrix
- `d_k`：query 和 key 的维度（GPT-2 small 的每个 head 是 64）
- `√d_k`：scaling factor
- `softmax`：沿 last dim（key dim）归一化

输出 shape：`(T, d_v)`，每个 token 输出一个 d_v 维的 context vector。

### 为什么 scale by √d_k

笔记里完整推导了。设 `Q_i, K_i ~ N(0, 1)` iid，则：

```
S = Σ_i Q_i · K_i,  i = 1..d_k

E[Q_i] = E[K_i] = 0
E[Q_i · K_i] = 0
E[(Q_i · K_i)^2] = Var(Q_i)·Var(K_i) = 1

E[S^2] = Σ_i E[(Q_i K_i)^2] + 2·Σ_{i<j} E[Q_i K_i Q_j K_j]
       = d_k · 1 + 0  (因为 i≠j 时独立，cross term 期望为 0)
       = d_k

Var[S] = d_k
std[S] = √d_k
```

所以 QK^T 的元素标准差是 `√d_k`。当 `d_k = 64` 时，`std ≈ 8`，softmax input 在 ±8 这种量级会让 distribution 变得非常 peaked（接近 one-hot），导致：
- gradient 趋零（softmax 饱和区）
- 训练不稳定

除以 `√d_k` 后，std 回到 1，softmax 输入处于温和区间，gradient 健康。

### Causal (Masked) Attention

为了做 next-token prediction，token i 只能 attend 到 token 0..i，看不到 i+1..T。

实现：
1. 计算 `S = QK^T`
2. 构造上三角 mask `M`，上三角元素为 `-∞`（或很大的负数），其余为 0
3. `S' = S + M`
4. `A = softmax(S')`
5. `Output = A · V`

softmax 之前 apply mask 是 conventional 做法，因为 `exp(-∞) = 0` 干净利落。

笔记里那个表展示了 mask 之后每个 token 的 attention weight 分布：`Your` 只能 attend 到自己；`journey` 能 attend `Your` 和自己；以此类推，最后一行 `step` 能 attend 全部 6 个 token。

---

## 5. Multi-Head Attention

### 直觉

单 head 只能学一种 relation pattern。Multi-head 让模型同时关注不同的 representation subspace：
- Head 1 可能学语法依赖（subject-verb agreement）
- Head 2 可能学指代消解
- Head 3 可能学 semantic similarity
- ...

### 公式

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
head_i = Attention(Q · W_Q^i, K · W_K^i, V · W_V^i)
```

变量：
- `h`：head 数量，GPT-2 small 是 12
- `d_k = d_v = d_model / h`：每个 head 的维度，GPT-2 small 是 768/12 = 64
- `W_O ∈ ℝ^(d_model × d_model)`：output projection

参数量分析：每个 head 有 `W_Q^i, W_K^i, W_V^i` 三个 `d_model × d_k` 矩阵，加 `W_O` 是 `d_model × d_model`。总参数 ≈ `h · 3 · d_model · d_k + d_model² = 3·d_model² + d_model² = 4·d_model²`，跟单 head 用 `d_model` 维的 Q/K/V 差不多。

### 实现优化

笔记里强调一个工程技巧：实践上不用 `h` 个独立的小矩阵，而是用一个大的 `W_Q, W_K, W_V ∈ ℝ^(d_model × d_model)`，一次投影完，然后 reshape：

```
queries = X @ W_Q                    # (b, T, d_model)
queries = queries.view(b, T, h, d_k) # 拆出 head 维
queries = queries.transpose(1, 2)    # (b, h, T, d_k)
# 同理 keys, values

attn_score = queries @ keys.transpose(-2, -1)   # (b, h, T, T)
attn_score = attn_score.masked_fill(mask, -inf)
attn_weights = softmax(attn_score / √d_k, dim=-1)
attn_weights = dropout(attn_weights)

context = attn_weights @ values       # (b, h, T, d_k)
context = context.transpose(1, 2)     # (b, T, h, d_k)
context = context.contiguous().view(b, T, d_model)  # 合并 head
output = context @ W_O                # (b, T, d_model)
```

GPU 上一次大 matmul 比 h 次小 matmul 快得多。

参考: Vaswani et al., 2017 https://arxiv.org/abs/1706.03762
Voita et al., 2019, "Analyzing Multi-Head Self-Attention" https://arxiv.org/abs/1905.09418（讨论 head 的冗余和 pruning）

---

## 6. GPT Model 整体架构

GPT-2 small 配置：

| 参数 | 值 |
|---|---|
| n_layer (Transformer blocks) | 12 |
| n_head | 12 |
| n_embd (d_model) | 768 |
| d_ff (FFN 内部维度) | 3072 = 4·d_model |
| vocab size | 50,257 |
| context length | 1024 |
| 参数总量 | ~124M |

### 单个 Transformer Block 结构

GPT 用的是 **Pre-LN** 结构（原始 Transformer 是 Post-LN）：

```
X
 ├─→ LayerNorm → Multi-Head Self-Attention → + ───┐
 │                                                │
 ↓                                                ↓
 ├─→ LayerNorm → FFN(Linear-GELU-Linear) → + ───┐
 │                                               │
 ↓                                               ↓
 Output
```

Pre-LN 的好处是 residual path 上没有 LayerNorm 的 scale，gradient 可以一路畅通，对深层网络训练更稳定。

参考: Xiong et al., 2020, "On Layer Normalization in the Transformer Architecture" https://arxiv.org/abs/2002.04745

### LayerNorm 公式

```
y_i = (x_i - E[x]) / √(Var[x] + ε) · γ_i + β_i
```

变量：
- `x_i`：feature dimension i 的值
- `E[x]`：沿 feature dimension 的均值（对每个 token 独立计算）
- `Var[x]`：沿 feature dimension 的方差
- `ε`：小常数，通常 1e-5，防止除零
- `γ_i, β_i`：learnable scale 和 shift，shape 是 `(d_model,)`

LayerNorm vs BatchNorm：
- BatchNorm 沿 batch dimension 统计，依赖 batch size，对 NLP 变长 sequence 不友好
- LayerNorm 沿 feature dimension 统计，对 batch size 不敏感

RMSNorm 是 LayerNorm 的简化版，去掉 mean centering（只保留 scale），LLaMA 在用，省一点计算。

参考: Ba et al., 2016, LayerNorm https://arxiv.org/abs/1607.06450
Zhang & Sennrich, 2019, RMSNorm https://arxiv.org/abs/1910.07467

### GELU 激活函数

```
GELU(x) = x · Φ(x)
```

其中 `Φ` 是标准正态分布的 CDF，`Φ(x) = 0.5 · (1 + erf(x/√2))`。

近似公式（tanh 形式）：

```
GELU(x) ≈ 0.5x · [1 + tanh(√(2/π) · (x + 0.044715·x³))]
```

为什么用 GELU 而不用 ReLU：
1. **Smooth at 0**：ReLU 在 x=0 不可导，GELU 处处可导，gradient 更稳定
2. **Non-zero for small negative**：ReLU 对所有负值直接输出 0，丢信息；GELU 对小负值保留一点信号
3. **Probabilistic interpretation**：可以理解为 "以 Φ(x) 的概率保留 x"——大正值几乎保留，小负值几乎丢弃，但 transition 是 smooth 的

延伸：SwiGLU（Shazeer, 2020）是 LLaMA 用的 FFN 激活，公式 `SwiGLU(x, W, V) = Swish(x·W) ⊙ x·V`，比 GELU + 两层 Linear 表现更好但多一个 matmul。

参考: Hendrycks & Gimpel, 2016, GELU https://arxiv.org/abs/1606.08415
Shazeer, 2020, GLU Variants https://arxiv.org/abs/2002.05202

### Feedforward Network

```
FFN(x) = Linear_2(GELU(Linear_1(x)))
```

- `Linear_1`: `d_model → d_ff = 4·d_model`
- `Linear_2`: `d_ff → d_model`

"Expand and contract" 结构：先把 representation 投影到 4× 高维空间，应用非线性，再压回。直觉上：高维空间里非线性变换有更多自由度，可以学习更复杂的 transformation，然后压回原维度保留有用的部分。

### Residual Connection

```
x_out = x + Sublayer(x)
```

好处：
1. **Gradient highway**：反向传播时 gradient 可以直接 skip 过 Sublayer，避免 vanishing
2. **Identity init 友好**：深网络可退化为 identity，不会比浅层差
3. **Ensemble interpretation**：Veit et al. 2016 提出残差网络可以看作浅层网络的隐式 ensemble

参考: He et al., 2015, ResNet https://arxiv.org/abs/1611.04586 (wait it's 1512.03385 https://arxiv.org/abs/1512.03385)
Veit et al., 2016, "Residual Networks Behave Like Ensembles of Shallow Networks" https://arxiv.org/abs/1605.06431

---

## 7. 文本生成

### Greedy Decoding

```
idx = initial_context
for _ in range(max_new_tokens):
    logits = model(idx)[:, -1, :]   # focus on last token
    probs = softmax(logits, dim=-1)
    next_token = argmax(probs, dim=-1)
    idx = cat([idx, next_token], dim=-1)
```

Greedy 的问题：容易陷入 repetition loop，且不采样导致 diversity 低。

### Temperature Scaling

```
probs = softmax(logits / T, dim=-1)
```

- `T → 0`：趋近 argmax（greedy）
- `T = 1`：原始分布
- `T → ∞`：趋近 uniform（完全随机）

T 控制 distribution 的 "sharpness"。

### Top-p (Nucleus) Sampling

1. 按 probability 降序 sort
2. 算 cumulative probability
3. 保留 cumulative probability 刚好 ≥ p 的最小 token 集合
4. renormalize 后 sample

Top-p vs Top-k：
- Top-k 固定数量 k，对 sharp distribution 浪费，对 flat distribution 太窄
- Top-p 动态调整数量，对 sharp distribution 选少几个，对 flat distribution 选多

参考: Holtzman et al., 2019, "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751

---

## 8. Pretraining

### Self-supervised Learning

数据不需要人工 label。Input 是 `token_0..t`，label 是 `token_{t+1}`——这就是 "next word prediction" 的本质。整个 corpus 都是免费 label。

### Cross-Entropy Loss

单 token：

```
L = -log p(token_{t+1} | token_{0..t})
```

batched average over sequence length T：

```
L = -(1/T) · Σ_{t=1}^{T} log p(token_t | token_{0..t-1})
```

PyTorch 里用 `nn.CrossEntropyLoss`，它内部用 log-softmax + NLL，比手动 softmax + log 更稳定。

### Log-Sum-Exp Trick

```
log_softmax(x_i) = x_i - logsumexp(x)
logsumexp(x) = log(Σ_j exp(x_j)) 
             = c + log(Σ_j exp(x_j - c))   where c = max(x)
```

减去 max 避免 overflow，加回 c 不影响结果。`nn.CrossEntropyLoss` 内部就是这样做的。

### AdamW Optimizer

Adam 更新规则：

```
m_t = β_1 · m_{t-1} + (1 - β_1) · g_t           # first moment
v_t = β_2 · v_{t-1} + (1 - β_2) · g_t²          # second moment
m̂_t = m_t / (1 - β_1^t)                          # bias correction
v̂_t = v_t / (1 - β_2^t)
θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)
```

变量：
- `g_t`：当前 step gradient
- `m_t`：first moment 估计（momentum）
- `v_t`：second moment 估计（per-parameter adaptive learning rate）
- `β_1, β_2`：默认 0.9, 0.999
- `η`：learning rate
- `ε`：默认 1e-8，防止除零

AdamW 修正了 weight decay：把 L2 regularization 替换为直接的 weight decay：

```
θ_t = θ_{t-1} - η · (m̂_t / (√v̂_t + ε) + λ · θ_{t-1})
```

这样 weight decay 不被 m, v 的统计量稀释，对 LLM 训练更稳定。

参考: Loshchilov & Hutter, 2017, "Decoupled Weight Decay Regularization" https://arxiv.org/abs/1711.05101
Kingma & Ba, 2014, Adam https://arxiv.org/abs/1412.6980

### Checkpoint 保存

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "model_and_optimizer.pth")
```

为什么还要存 optimizer state：因为 Adam 保存了 `m, v` 等统计量，恢复训练必须还原它们，否则从零冷启动会剧烈震荡。

---

## 9. Fine-tuning

### Classification Fine-tuning

流程：
1. 加载预训练 weights
2. 替换 output head：原 LM head `d_model → |V|` 换成 classification head `d_model → num_classes`
3. 选 fine-tune 策略：
   - Full fine-tuning：所有层更新
   - Last-token + last-layer：用 last token 的 output（因为 causal mask 让 last token 看过整个 sequence），只 tune 最后几层
   - LoRA / Adapter：冻结 backbone，只训练注入的小 module

笔记里强调一个直觉：浅层学到通用语言特征，深层学到 task-specific 特征，所以通常 fine-tune 最后几层就够。

LoRA 公式：`W' = W_0 + α/r · B·A`，其中 `W_0` 冻结，`A ∈ ℝ^(r×d), B ∈ ℝ^(d×r)` 可训练，r ≪ d 是 rank。

参考: Hu et al., 2021, LoRA https://arxiv.org/abs/2106.09685
Houlsby et al., 2019, Adapter https://arxiv.org/abs/1902.00751

### Instruction Fine-tuning

数据格式：
- **Alpaca style**: `### Instruction: ... ### Input: ... ### Response: ...`
- **Phi-3 style**: `<|user|> ... <|assistant|> ...`

需要 collate function 处理变长 input，padding 到同一长度，并构造 attention mask 让 padding 不参与 attention。

### Evaluation

不像 classification 算 accuracy 那么直接，instruction-tuned LLM 的评估方式：

- **MMLU** (Hendrycks et al., 2009): 57 个 subject 的多任务选择 https://arxiv.org/abs/2009.03300
- **LMSYS Chatbot Arena**: 人类 pairwise 比较 https://arena.lmsys.org
- **AlpacaEval**: GPT-4 自动评分 https://tatsu-lab.github.io/alpaca_eval/
- **MT-Bench**: 多轮对话评估 https://arxiv.org/abs/2306.05685
- **HumanEval**: 代码生成 https://arxiv.org/abs/2107.03374

---

## 10. PyTorch 工程细节

### Tensor 概念

笔记里画了 scalar / vector / matrix / 3D tensor 的示意图。Tensor 是 rank-n 的 collection of values，rank 0 是 scalar，rank 1 是 vector，rank 2 是 matrix。

PyTorch 默认 int tensor 是 `int64`，float tensor 是 `float32`。

### Autograd

PyTorch 用 reverse-mode autodiff：forward pass 构建计算图，backward pass 通过 chain rule 累积梯度。

```
∂L/∂w_1 = (∂u/∂w_1) · (∂z/∂u) · (da/dz) · (∂L/∂a)
```

每个中间节点只算自己 local 的 Jacobian，autograd 把它们 chain 起来。

### 输出 logits 而非 probs

笔记里强调，model forward 输出 logits，不要内部 softmax，因为：
1. **信息完整**：logits 没归一化，包含更多信息
2. **数值稳定**：softmax 对大 logit 会 overflow，对极小 logit 会 underflow
3. **Loss 内部做**：`nn.CrossEntropyLoss` 内部已经 stable 地做了 log-softmax
4. **Inference 直接 argmax**：argmax 在 logits 上和在 probs 上结果一样，省一次 softmax

### model.eval() vs model.train()

- `model.train()`：启用 dropout，BatchNorm 用 batch 统计
- `model.eval()`：关闭 dropout，BatchNorm 用 running 统计
- LayerNorm 不受影响（不依赖 batch 统计，每个 sample 独立）

### DataLoader

- `num_workers`：多进程预取，避免 IO bottleneck
- `drop_last=True`：丢弃不完整 batch，避免 batch size 波动影响 BN 统计（LayerNorm 不受影响但保持习惯）
- `pin_memory=True`：加速 CPU→GPU transfer

### Training Loop 典型结构

```python
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        optimizer.zero_grad()           # 清上一步 gradient
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()                 # 算 gradient
        optimizer.step()                # 更新参数
    
    model.eval()
    with torch.no_grad():
        # optional evaluation
        pass
```

---

## 11. 一些值得深挖的延伸

笔记里有很多点都可以进一步挖深：

### 11.1 Karpathy 推文那个疑虑

笔记里贴了 Karpathy 2024/9/22 推文：sinusoidal PE 在 high dimension 上频率被 "clip"。这个问题在 long-context extension 里尤其重要——后续出现的 RoPE、ALiBi、NoPE 都在重新思考 position 的注入方式。

- RoPE: https://arxiv.org/abs/2104.09864
- ALiBi: https://arxiv.org/abs/2108.12409
- NoPE (No Positional Encoding): https://arxiv.org/abs/2305.16843

### 11.2 Attention 的信息论视角

Attention 可以看作 soft dictionary lookup，或者用 differentiable memory 的视角看：V 是 memory，Q 是 query，softmax(QK^T) 是 retrieval weight。这跟后续的 retrieval-augmented generation、Mamba 的 selective state space 是有思想关联的。

- Mamba: https://arxiv.org/abs/2312.00752
- Linear Attention: https://arxiv.org/abs/2006.16236

### 11.3 Scaling Law

笔记没怎么讲 scaling，但理解 LLM 必须理解 Chinchilla scaling law：最优 token 数 ≈ 20 × 参数量。

- Kaplan et al., 2020: https://arxiv.org/abs/2001.08361
- Hoffmann et al., 2022 (Chinchilla): https://arxiv.org/abs/2203.15556

### 11.4 Pre-training 之外的对齐

笔记最后只讲到 instruction tuning，但完整的现代 LLM pipeline 还有 RLHF / DPO：

- InstructGPT (RLHF): https://arxiv.org/abs/2203.02155
- DPO: https://arxiv.org/abs/2305.18290

### 11.5 Tokenizer 的微妙影响

BPE 的 pre-tokenization（先用 regex 把 text 切成 word-like chunk 再做 BPE）对最终 token 分布影响很大。GPT-2 的 regex 是：

```
r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

参考 GPT-2 encoder.py: https://github.com/openai/gpt-2/blob/master/src/encoder.py

### 11.6 Karpathy 的相关材料

如果你（Karpathy 本人，哈哈）还没看过自己以前的东西——

- nanoGPT: https://github.com/karpathy/nanoGPT
- makemore: https://github.com/karpathy/makemore
- "Zero to Hero" YouTube 系列: https://karpathy.ai/zero-to-hero.html
- "State of GPT" talk: https://www.youtube.com/watch?v=bZQun8X4G2g

---

## 12. 把整篇笔记的"灵魂"压缩成一句话

整本书在教你 **怎么从一行 raw text 一路推到一个会聊天的 LLM**，每一步都自己实现，每一步都把数学和代码对应起来。它跟 Karpathy 的 nanoGPT 在精神上完全同源——把 LLM 从 black box 拆成可以一个文件讲完的几十行 PyTorch。

读完之后回头再看 production LLM，你会发现 99% 的复杂度都在工程化（distributed training, FSDP, paged attention, MoE routing, quantization），而 model 本身的核心公式其实就藏在 `softmax(QK^T/√d_k)V` 这一坨里。

---

## 参考文献汇总

1. Sebastian Raschka, "Build a Large Language Model (From Scratch)" — https://www.manning.com/books/build-a-large-language-model-from-scratch
2. 代码仓库 — https://github.com/rasbt/LLMs-from-scratch
3. Vaswani et al., 2017, "Attention is All You Need" — https://arxiv.org/abs/1706.03762
4. Radford et al., 2019, GPT-2 — https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
5. Brown et al., 2020, GPT-3 — https://arxiv.org/abs/2005.14165
6. Karpathy nanoGPT — https://github.com/karpathy/nanoGPT
7. Bahdanau et al., 2014 — https://arxiv.org/abs/1409.0473
8. Sennrich et al., 2015, BPE — https://arxiv.org/abs/1508.07909
9. Su et al., 2021, RoPE — https://arxiv.org/abs/2104.09864
10. Press et al., 2021, ALiBi — https://arxiv.org/abs/2108.12409
11. Ba et al., 2016, LayerNorm — https://arxiv.org/abs/1607.06450
12. Zhang & Sennrich, 2019, RMSNorm — https://arxiv.org/abs/1910.07467
13. Hendrycks & Gimpel, 2016, GELU — https://arxiv.org/abs/1606.08415
14. Shazeer, 2020, GLU Variants — https://arxiv.org/abs/2002.05202
15. He et al., 2015, ResNet — https://arxiv.org/abs/1512.03385
16. Xiong et al., 2020, Pre-LN — https://arxiv.org/abs/2002.04745
17. Holtzman et al., 2019, Nucleus Sampling — https://arxiv.org/abs/1904.09751
18. Loshchilov & Hutter, 2017, AdamW — https://arxiv.org/abs/1711.05101
19. Kingma & Ba, 2014, Adam — https://arxiv.org/abs/1412.6980
20. Hu et al., 2021, LoRA — https://arxiv.org/abs/2106.09685
21. Ouyang et al., 2022, InstructGPT — https://arxiv.org/abs/2203.02155
22. Hendrycks et al., 2009, MMLU — https://arxiv.org/abs/2009.03300
23. Hoffmann et al., 2022, Chinchilla — https://arxiv.org/abs/2203.15556
24. Kaplan et al., 2020, Scaling Laws — https://arxiv.org/abs/2001.08361
25. Gu & Dao, 2023, Mamba — https://arxiv.org/abs/2312.00752
26. Tiktoken — https://github.com/openai/tiktoken
27. LMSYS Chatbot Arena — https://arena.lmsys.org
28. AlpacaEval — https://tatsu-lab.github.io/alpaca_eval/

---

如果你想我把某一块（比如 Multi-Head 的 head 之间到底在学什么、或者 RoPE vs Sinusoidal 的几何对比、或者 nanoGPT 跟 Raschka 的代码在实现细节上的差异）继续展开，告诉我就行。
