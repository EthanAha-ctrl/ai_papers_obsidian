---
source_pdf: Babilong.pdf
paper_sha256: f0d466dbeb0906e9055aff7b2c0c856b8b47267033a4011e5217dbcf1810ecc1
processed_at: '2026-07-18T13:21:46-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BABILong: 把 bAbI "藏"进 PG19 的 haystack 里, 然后逼问 LLM 你到底用了多少 context

 Andrej 你好, 这是一篇非常对你胃口的 paper — 它本质上是把 Weston et al. 2016 的 bAbI (20 个 toy reasoning task) 用一种很 clean 的方式 rebrand 成 long-context benchmark, 然后用它戳穿了几乎所有 commercial long-context LLM 的"effective context window"宣传。下面我尽量把每个关键技术点和实验数字都拆开讲, 顺便把 RMT/ARMT 的机制也一起讲清楚, 因为 paper 后半段的 fine-tuning 实验其实才是真正"解决问题"的部分。

---

## 1. Motivation: needle-in-a-haystack 已经饱和

现有的 long-context benchmark 大致分两类:

1. **Real/human-annotated**: LongBench (avg 6k–13k, max 40k) [1], L-Eval (3k–60k) [2], InfinityBench, NovelQA, LVEval, CLongEval, XL²-Bench 等, 这些长度上限最多 ~500k, 而且标注成本高, 容易 leak。
2. **Synthetic needle-in-haystack**: LLMTest-NIAH (Paul Graham essays + magic number), RULER [3], Counting-Stars [4], S3Eval [5]。问题在于 needle 太"显眼" (magic number 这种), 现代 long-context model 几乎都是满分全绿 heatmap, 已经区分不出 model 的强弱。

BABILong 想做的事情就是: **保留 needle-in-haystack 的可扩展性, 但是 needle 本身换成了需要 reasoning 的 bAbI task**, 而且答案不能靠单点 retrieval 拿到, 需要跨多个分布在百万级 token 里的 fact 做 chaining / counting / induction。

- paper: https://arxiv.org/abs/2406.09128
- code & leaderboard: https://github.com/booydar/babilong
- dataset on HF: https://huggingface.co/datasets/RMT-team/babilong

---

## 2. Benchmark 构造: 一个简单的 mixing pipeline

直觉: 把 bAbI sample 的 fact sentences 一个一个塞进 PG19 [6] 的书里, 形成"看似连续的小说, 中间插了几句 Mary went to the kitchen 这种话"。

具体 pipeline (Figure 1a):

1. 取一个 bAbI sample (一组 facts + 一个 question + answer), facts 是形如 `Mary travelled to the office` 这种句子。
2. 取 PG19 的书, 用 `nltk.PunktSentenceTokenizer` 切成 sentence 流。
3. **均匀地**把 bAbI 的 fact sentence 插入到 PG19 的 sentence 流里 — 即在每两个 PG19 sentence 之间, 以一定概率插入一句 bAbI fact。
4. 重复直到达到 target length (4k / 8k / ... / 10M tokens, 用 GPT-2 tokenizer 计长度)。
5. Question 可以放在文本的开头或结尾。

为什么选 PG19 而不是 random token / Wikipedia? 因为 PG19 是 1919 年以前的小说, 里面会自然出现人物名、地点名、动作描述, 这和 bAbI 的 "Mary went to the kitchen" 在分布上 *接近但不完全相同*, 这样既不会让 needle 一眼就被识别出来, 也不会让 retrieval 完全失效 (Wikipedia 太"信息密集", paper 在 Appendix G 里也跑了 Wikipedia embedding 的对照)。

BABILong 一共继承 bAbI 的 20 个 task (QA1–QA20), Table 1 列了前 10 个:

| Task | Name | Facts per task | Relevant facts |
|---|---|---|---|
| QA1 | single supporting fact | 2–10 | 1 |
| QA2 | two supporting facts | 2–68 | 2 |
| QA3 | three supporting facts | 4–320 | 3 |
| QA4 | two-arg relations | 2 | 1 |
| QA5 | three-arg relations | 2–126 | 1 |
| QA6 | yes/no | 2–26 | 1 |
| QA7 | counting | 2–52 | 1–10 |
| QA8 | lists/sets | 2–50 | 1–8 |
| QA9 | simple negation | 2–10 | 1 |
| QA10 | indefinite knowledge | 2–10 | 1 |

QA1–QA3 是把同一个 single-fact retrieval 任务逐步 chain 起来, 难度阶梯非常漂亮, 是后续实验的核心 probe。

**重要属性**: 因为 bAbI 和 PG19 都是 algorithmically generated / public domain 的, 所以 BABILong **完全 leak-proof**, 未来 GPT-6 也不会因为训练数据污染而作弊; 同时可以任意 scale 到几十 M tokens, 这是 LongBench 这类数据集做不到的。

---

## 3. 主要 baseline 实验: 揭穿 "128K context" 神话

### 3.1 Setup

paper 评测了 34+ 个模型, 大致分类:
- Open-source 32k–64k: Mistral-7B, Mixtral-8x7B, Yi-9B/34B-200k
- Open-source 128k: Llama-3.1-8B / 70B, Phi-3 / Phi-3.5-mini / mini / MoE, Command-R, Qwen-2.5-7B/72B, ChatGLM3-6B-128k, Jamba-v0.1
- Long-context fine-tuned: LongChat, Llama-2-7B-32K, LongAlpaca
- Length-extension methods: YaRN-Mistral, Activation Beacon
- Closed: GPT-4 (0125-preview), GPT-4o-mini, Gemini 1.5 Pro 002 [7]
- Alternative architectures: Mamba-2.8B, RWKV-6, Jamba, recurrent Gemma
- Fine-tuned small models: RMT, ARMT (GPT-2 137M backbone), Mamba-130M

判定标准: **accuracy > 85% 算 satisfactory, < 30% 算 failure**。

### 3.2 核心发现 (Figure 2 + Table 2)

**几乎所有 commercial long-context LLM 在 BABILong 上只有效使用 10–20% 的 claimed context**。具体地:

| Task | 谁能撑到哪 |
|---|---|
| QA1 (single fact) | 多数模型只在 4K 内保持 85%+; 只有 GPT-4 + Llama-3.1-70B 到 16K, Qwen-2.5-72B + Gemini Pro 1.5 到 64K |
| QA2 (two facts) | 没有 LLM 在带 distractor 文本时能达到 85%, 即便是 GPT-4 和 Gemini 1.5 Pro |
| QA3 (three facts) | 最好的模型 < 80%, 已经接近 random baseline |

注意: "claimed 128K" 的模型中, 几乎所有都在 ≤ 10% context size (即 ~12K) 处就开始崩塌。Yi-34B-200k 在 64K 上就 < 30%, Phi-3-mini 从 64K 跳到 128K 直接掉到 <10%, Activation Beacon 在 32K 上还是 < 40%。

**Table 2** 是这篇 paper 的灵魂 — average over QA1–QA5:

```
Group "128k":        0K     4K     16K    32K    128K
GPT-4                99     95     91     87     70
Gemini 1.5 Pro 002   100    100    100    100    94  (但 14% 的请求被 content filter 拒绝)
Qwen2.5-72B          99     100    98     100    96
Llama-3.1-70B        99     97     91     92     94
Llama-3.1-8B         99     93     76     68     46
Phi-3-medium         92     84     72     68     33
```

可以从 Table 2 的数字里读出几条非常 robust 的规律:
1. 0K (即 bAbI 原始长度 ~120 tokens) 上 LLM 已经表现不一, QA2/QA3 在 0K 上就有模型不及格 — 这意味着 reasoning 本身就是 bottleneck, 长上下文只是放大了这个问题。
2. Size 帮助有限: Phi-3-medium (14B) 在 128K 上 ≈ 33%, 但 Qwen-2.5-72B 能到 96%, 这说明预训练数据的 long-context 部分 (Llama-3.1 用了 six-stage curriculum 8K→128K, Qwen-2.5 类似) 比 pure size 更重要。
3. Gemini 1.5 Pro 在 128K 上能保持 94%, 但要注意 Figure 6 显示 content safety filter 会拒绝最多 14% 的请求, 这在长上下文里是非平凡 bias。

---

## 4. RAG 为什么在 BABILong 上失败 (Section 3.2)

这是 paper 一个很有教学意义的 negative result。作者跑了两种 chunking:

- **RAG-C**: 512-token chunk, 用 text-embedding-ada-002 做 dense retrieval, top-5 chunks 塞进 GPT-4
- **RAG-S**: sentence-level chunk, 同样 embedding, top-5 / top-20

Figure 3a 的结论:

| 配置 | QA1 结果 |
|---|---|
| GPT-4 + RAG-C | 长度增加后迅速下降到 ~30% |
| GPT-4 + RAG-S, top-5 | 在 16K 之前比较好, 之后衰减 |
| GPT-4 + RAG-S, top-20 | 把 retrieved 从 5 增到 20 **没有帮助**, 反而引入更多 distractor |

直觉: QA1 的 needle 句子在文本末尾 ("最新的 location"), 但是 retrieval 用的是 question embedding, 而 question ("Where is Mary?") 和 "Mary went to the kitchen" 的 cosine similarity 未必比 question 和某些 PG19 句子的相似度高。所以 dense retrieval 在这种"query 和 doc 表面语义不相似"的场景下系统性地 fail。

**对 QA2/QA3 更致命**: 因为需要 *两个或三个* fact, 且 fact 之间有 *temporal dependency* (先 pick up, 再 move), dense retrieval 既不保序也很难同时召回所有 hop — Figure 3a 上 QA2/QA3 的 RAG 曲线直接掉到 random guess 以下。

这其实是个很重要的 takeaway: **RAG 不能代替 long-context reasoning**。 retrieval 解决的是"找到包含某个 entity 的段落", 解决不了"按时间顺序把跨段落的多个事件链起来"。

---

## 5. RMT / ARMT: 真正解决问题的方案

paper 真正"治愈" BABILong 的部分是 fine-tune 几个 *很小* 的 recurrent memory model (137M GPT-2 backbone!), 然后它们能跑到 50M tokens 还保持 60%+ accuracy。这是整篇 paper 最 surprising 的发现, 也是构建 long-context model 的一个 strong baseline。

### 5.1 RMT (Recurrent Memory Transformer) [8]

RMT 的核心是把 Transformer 变成一个 *segment-level recurrent* 模型。给定一个超长序列:

$$x = (x_1, x_2, \dots, x_N)$$

把它切成 $T$ 个 segment, 每个 segment 长度为 $s$ (paper 里 $s = 512$):

$$x = (S_1, S_2, \dots, S_T), \quad S_t = (x_{(t-1)s+1}, \dots, x_{ts})$$

引入 $m$ 个 learnable **memory tokens** $M \in \mathbb{R}^{m \times d}$ (paper 里 $m = 16$, $d$ 是 GPT-2 的 hidden dim 768)。

递推公式 (这是 RMT 的全部数学):

$$[M_t; H_t] = \text{Transformer}([M_{t-1}; S_t])$$

其中:
- $M_{t-1} \in \mathbb{R}^{m \times d}$ 是上一段 segment 处理完后保留下来的 memory hidden states
- $[M_{t-1}; S_t] \in \mathbb{R}^{(m+s) \times d}$ 是简单的 row-wise concatenation
- $M_t$ 是输出序列 *前 m 个位置* 的 hidden states, 作为下一段的输入 memory
- $H_t \in \mathbb{R}^{s \times d}$ 是当前 segment 的 hidden states, 用来预测下一个 token / answer

关键: 
- 训练时用 **BPTT through time**: 梯度可以沿着 $M_{t-1} \to M_t$ 这条 path 回传到任意远的历史 (类似 LSTM 但是用 self-attention 作为 transition)。
- 推理时 memory 是 *constant size* ($m \times d$ 个 float), 所以推理 memory 是 $O(1)$ 而非 $O(N)$, 这就是为什么可以处理 50M tokens。

### 5.2 ARMT (Associative Recurrent Memory Transformer) [9]

ARMT 是 RMT 的升级版, 在递推之外加了一个 **可训练的 external associative memory**。paper 用到的关键组件:

- $N_{\text{mem}} = 10$ 个 memory tokens (少于 RMT 的 16)
- memory dimension $d_{\text{mem}} = 64$
- 用 DPFP-3 (Deterministic Parameter-free Product, Schlag et al. 2021 [10]) 作为 key/value binding 的 non-linearity

DPFP-3 的 update rule 大致是:

$$\Psi(x) = \phi(x \otimes x)$$

其中 $x \in \mathbb{R}^d$, $\otimes$ 是 outer product, $\phi$ 是把 resulting $d \times d$ matrix 折叠成向量的 element-wise 非线性 (具体是去掉一部分维度再取 ReLU)。Memory update:

$$W_t = W_{t-1} + \Psi(k_t) \otimes v_t$$

这里 $k_t, v_t \in \mathbb{R}^{d_{\text{mem}}}$ 是当前 segment 产生的 key/value, $W_t \in \mathbb{R}^{d_{\text{mem}} \times d_{\text{mem}}}$ 是 associative memory matrix。

Query 时:

$$r_t = \Psi(q_t)^T W_t / \|\Psi(q_t)\|$$

直觉上, ARMT 在 RMT 的"递推 hidden state"之上又加了一层 *外部可寻址的 key-value store*, 这样可以在 segment 之间不只是被动地传递 hidden state, 还可以主动从过去 segment 写入的 key-value pair 里 retrieve 信息。这使得 ARMT 在 50M token 上还能 maintain 信号, 而 pure RMT 在 10M 之后开始衰减。

### 5.3 训练细节 (Appendix C)

- backbone: GPT-2 small (137M)
- segment size $s = 512$, memory tokens $m = 16$ (RMT) / 10 (ARMT)
- **Curriculum learning**: stage $n$ 训练长度从 1 到 $n$ 个 segment 随机采样, 即长度不固定以避免过拟合到特定长度。Stage 顺序: 1 → 2 → 4 → 6 → 8 → 16 → 32 segments (即最长 16K tokens)
- Optimizer: AdamW, lr $\in \{3e{-5}, 5e{-5}\}$, ARMT 用 $1e{-4}$
- Batch size 64, weight decay 0.01
- 每 stage 最多 10K steps, early stop on short-context metric
- Hardware: 1–4 张 A100/H100, 每 stage 40 min – 20 hr
- 每个 task 单独训, train set 10k samples, test set 1k samples

### 5.4 结果 (Figure 3b + Table 4)

| Model | 训练长度 | 测试长度上限 | QA1 avg |
|---|---|---|---|
| GPT-4 (zero-shot) | — | 128K | 70 (128K) |
| Gemini 1.5 Pro | — | 128K | 94 (但 14% rejected) |
| Mamba-130M (fine-tuned) | 16K | 128K (技术上限) | 100 |
| RMT (GPT-2 + 16 mem) | 16K | 10M | 99 (10M) |
| ARMT (GPT-2 + 10 mem) | 16K | 50M | 99 (10M), 92 (50M) |

最 striking 的数字: ARMT 在 **11.1M tokens** 上仍保持 99% QA1 accuracy, 在 **50M tokens** 上还能拿到 84% — 而训练时它只见过最长 16K tokens, 这是 *600 倍外推*。

### 5.5 RMT 内部机制的可视化 (Appendix H, Figure 8)

Figure 8 是理解 RMT "为什么 work" 的关键:

- (a)(b): memory state 之间的 pairwise distance heatmap。横纵轴是 segment index。可以清楚看到, 在没有 fact 出现的 segment 上 memory 几乎不变 (深红色 = 距离小, 几乎重叠); 一旦遇到 fact sentence, memory state 明显"跳变" (蓝色)。这意味着 RMT 学会了"对无关 PG19 文本保持惰性, 对 bAbI fact 做大动作"。
- (c): memory 写入时的 attention pattern, 可以看到 memory tokens 在某个 fact 出现时把 fact 的 representation 写进 memory slot。
- (d): 问题出现时, query (来自 question tokens) 对 memory tokens 的 attention 高度集中在 *包含答案的那条 fact* 对应的 memory slot。

这其实是一个 emergent 的"读/写头"行为, 类似 Neural Turing Machine 但是没有任何 inductive bias 强制 — 单纯从 segment-level BPTT + curriculum + task loss 中涌现出来。

---

## 6. Mamba fine-tuning (Section 3.3 + Appendix C)

Mamba [11] 是 S4 系列的 selective state-space model, 复杂度 $O(N)$, 论文里跑了 Mamba-130M fine-tuned:

- 同样的 curriculum: 1→2→4→6→8→16→32 segments
- Batch 128, AdamW, lr 3e-4, weight decay 2.0, gradient clip 1.0, 10% warmup
- 每 stage 10K steps, 最后 32-segment stage 跑 15K steps
- 4× H100, 每 task 2–3 天

Mamba-130M 在 QA1 上和 RMT/ARMT 几乎打平, **但在 QA3 (three supporting facts) 上 Mamba 反而比 RMT 更好** (Table 4 的 QA3 section)。原因猜测: Mamba 的 hidden state 是一个 *向量* 而非 segment token 级, 信息密度高, 适合存多个 fact; 但代价是 inference 超过 128K 时 CUDA kernel 速度极慢 (paper 提到 "nearly impossible to process longer sequences")。

所以最终的 takeaway:
- **要长**: 用 ARMT (50M)
- **要 reasoning 复杂**: 用 Mamba (但 < 128K)
- **要 plug-and-play 到大 model**: 当前没有完美方案 — fine-tune GPT-3.5/Mistral-7B 能 work (Figure 9) 但只能到 16K/32K

---

## 7. 与 RULER / MMLU 的相关性分析 (Section 3.4, Figure 4)

这是个挺 subtle 的分析。作者在多个模型上同时跑了 MMLU [12], BABILong (0K, 4K, ..., 128K), RULER [3] (≤128K), 然后算 Spearman correlation。

发现:

- **BABILong vs MMLU**: 在 0K (即 bAbI 原始长度) 上 correlation 很高 (0.928), 随长度增加单调下降。
- **RULER vs MMLU**: 在所有长度上保持高 correlation, 即 RULER 像是"长版本 MMLU"。

直觉: RULER 的 needle 都是 uuid / number / adjective 这种 *isolated fact*, 难度主要来自"找", 而不是"reasoning"。所以 RULER 上排名和 MMLU 接近, 反映的是同一个"模型综合能力"的 latent factor。BABILong 一旦 context 变长, 引入了"reasoning over distributed facts"这个新维度, 所以和 MMLU 的 correlation 下降 — 这正是 BABILong 想要的 *additional signal*。

paper 原话: "BABILong can detect differences in models behavior starting from lengths as small as 2K tokens, while RULER requires lengths of at least 128K tokens"。也就是说, BABILong 比 RULER 更 sensitive。

---

## 8. Position-of-fact 的 ablation (Appendix K, Figure 10)

把 QA1 的 supporting fact 强制放在 context 的 0%, 25%, 50%, 75%, 100% 位置, 看 GPT-4-Turbo 的表现:

- 放在开头 (0%) 或结尾 (100%): 表现最好 — 这对应 LLM 的 *primacy* 和 *recency* bias
- 放在中间 (50%, 即 "lost in the middle" [13]): 表现最差

这是对 Liu et al. 2023 "Lost in the Middle" 现象的 long-context 验证, 说明 GPT-4 的 attention 没有真正"均匀"地覆盖整个 context window。

---

## 9. 局限性和潜在改进 (Limitations)

paper 自己列了几条:

1. **Background text 只用 PG19 + Wiki**: 其他语料 (代码、医学、对话) 的 mixing 效果未验证。
2. **bAbI vocabulary 很小** (~20 个名字、~10 个 object): fine-tuned model 可以轻松学到"Mary / kitchen 这种 token 是 fact 而非 distractor", 这是 fine-tune 模型在 0K 上能 100% 的原因之一。改进方向: 用 LLM 生成 vocab 更大的合成 fact, 或者把 RuleTaker / ProofWriter / FOLIO [14] / PrOntoQA 这种更复杂的 reasoning dataset 嵌进去。
3. **RAG pipeline 没优化**: paper 用的就是 LangChain + FAISS + ada-002, 没有用 reorder、HyDE、multi-hop retriever (如 IRCoT [15])。一个更公平的 RAG baseline 可能能让 multi-hop task 表现好一些。
4. **Recurrent model 的 storage 是有限的**: RMT 的 16 个 memory token 在 10M 上能 work 是因为 bAbI task 只需要 1–3 个 fact; 在 real-world 长文本里要存成千上万个 fact, 16 个 token 显然不够, 这也是 ARMT 引入 external associative memory 的动机。

我个人觉得一个隐含的 limitation 是: **bAbI 的 reasoning 是闭合的、确定性的**, 没有 hallucination / 多解 / 模糊语义。所以 fine-tune 的 small model 实际上是在做一个"算法可解的 toy task", 这和 real long-context QA 还是有差距的。

---

## 10. 联想 / Why this matters to you

作为 Karpathy 你应该对几个点特别有共鸣:

### 10.1 "Effective context window" ≠ "Claimed context window"

这是 paper 最强的 claim: GPT-4 的 *effective* context window 在 BABILong QA2/QA3 上其实是 < 4K, 而非 128K。这和 kvquant / ring-attention [16] 这些技术提到的"我们支持 1M context"是不同维度的事情 — 算力上撑得住 ≠ 模型真的会用。

### 10.2 RMT 的 emergence of read/write heads

Appendix H 的 attention map 实际上展示了一种 *learned differentiable memory access*, 没有任何 hard-coded "addressing"机制, 只靠 segment-level BPTT 和 task loss, 就在 GPT-2 backbone 上学出了"看到 fact 就写入, 看到 question 就读取"的明确行为。这和 karpathy/micrograd / nanoGPT 风格的"让 SGD 自己找出来"是完美契合的。

### 10.3 50M tokens 的 single-model inference 是一个 milestone

之前 Infini-attention [17] / LongNet [18] 等工作声称"理论上"支持 1B+ tokens, 但都是 language modeling perplexity 或者非常 toy 的 retrieval。BABILong 是第一个让一个 *fine-tuned single model* 在 50M tokens 上做 multi-hop reasoning 还能 maintain 84% accuracy 的公开实验, 这是 SOTA long-context 单模型的工作上限 benchmark。

### 10.4 Mamba vs RMT 在 reasoning 上的 split

Mamba 在 QA3 (需要记住 3 个 fact 并按时间链起来) 上比 RMT 好, 但 inference 卡在 128K; RMT 弱一点但能跑到 10M; ARMT 两者兼得。这暗示了一个有意思的设计空间: **linear RNN 的 hidden state 信息密度高但难以压缩, segment-recurrent + external memory 可以做 trade-off**。这可能是下一代 hybrid architecture (Jamba 已经尝试了) 的关键设计点。

### 10.5 跟 jevkō / nanoGPT 风格的契合

RMT 的代码其实就是: GPT-2 + 16 个可学习 prefix token + segment loop + BPTT。代码量可能 < 200 行。把这种 minimal modification 推到 50M tokens 而 GPT-4 做不到 — 这是 "complexity is the enemy of execution" 这句话的好例证。

---

## 11. 几个值得跟进的后续方向

1. **把 BABILong 思路 apply 到更复杂的 reasoning task**: RuleTaker, FOLIO, DELTA, 用 LLM 生成 vocab 大的合成 fact, 而不只是 bAbI 的 Mary/kitchen。
2. **RMT + LoRA on 大 model**: 现在 fine-tune 都是在 GPT-2 137M 上做的, 能不能把 RMT 加到 Llama-3-8B 上 LoRA fine-tune, 在保持 general LM 能力的同时拿到 long-context reasoning?
3. **Recurrent memory + native long attention hybrid**: ARMT 已经在往这个方向走, 但还可以更激进 — 在 *需要 retrieval 的 segment* 才走 associative memory, 其他时候用 full self-attention。这有点像 Memorizing Transformer [19] 的 spirit。
4. **Fine-tuning data 的合成**: 既然 BABILong 是 generative 的, 完全可以做"长度从 1 到 1M 的混合数据", 验证"long-context fine-tuning data 配比"对 effective context 的影响。这跟 Llama-3.1 的 six-stage curriculum 是同一类问题。
5. **BABILong as a *training* objective, 不只是 eval**: 现在大家用 BABILong 做评测, 如果直接把 BABILong-style 数据放进 pre-training, 会不会让 base model 就具备 long-context reasoning 的 inductive bias?

---

## 12. 引用链接汇总

- BABILong paper: https://arxiv.org/abs/2406.09128 (Kuratov et al., 2024)
- BABILong code: https://github.com/booydar/babilong
- BABILong data on HF: https://huggingface.co/datasets/RMT-team/babilong
- bAbI (Weston et al., 2016): http://arxiv.org/abs/1502.05698
- PG19 (Rae et al., 2020): https://github.com/google-deepmind/pg19
- RMT (Bulatov et al., 2022, NeurIPS): https://proceedings.neurips.cc/paper_files/paper/2022/file/47e288629a6996a17ce50b90a056a0e1-Paper-Conference.pdf
- RMT extended (Bulatov et al., 2024, AAAI): https://ojs.aaai.org/index.php/AAAI/article/view/29722
- ARMT (Rodkin et al., 2024): https://arxiv.org/abs/2407.04841
- Mamba (Gu & Dao, 2023): https://arxiv.org/abs/2312.00752
- DPFP (Schlag et al., 2021): https://proceedings.mlr.press/v139/schlag21a.html
- Gemini 1.5 (Reid et al., 2024): https://arxiv.org/abs/2403.05530
- RULER (Hsieh et al., 2024): https://arxiv.org/abs/2404.06654
- LongBench (Bai et al., 2023): https://arxiv.org/abs/2308.14508
- L-Eval (An et al., 2023): https://arxiv.org/abs/2307.11088
- Lost in the Middle (Liu et al., 2023): https://arxiv.org/abs/2307.03172
- Activation Beacon (Zhang et al., 2024): https://arxiv.org/abs/2401.03462
- Memorizing Transformer (Wu et al., 2022): https://openreview.net/forum?id=TrjbxzRcnf-
- FOLIO (Han et al., 2022): https://arxiv.org/abs/2209.00840
- Llama 3 herd (Dubey et al., 2024): https://arxiv.org/abs/2407.21783
- Qwen 2.5 tech report: https://qwenlm.github.io/blog/qwen2.5/
- Block-Recurrent Transformer (Hutchins et al., 2022): https://openreview.net/forum?id=uloenYmLCAo
- Compressive Transformer (Rae et al., 2020): https://openreview.net/forum?id=SylKikSYDH
- Neural Turing Machines (Graves et al., 2014): https://arxiv.org/abs/1410.5401
- Jamba (Lieber et al., 2024): https://arxiv.org/abs/2403.19887
- Datasheets for datasets (Gebru et al., 2021): https://arxiv.org/abs/1803.09010

---

## 13. TL;DR for building intuition

1. **BABILong = bAbI 的 fact sentence 散布在 PG19 小说里, 长度 0 → 50M tokens 可调**。Generative, leak-proof, 比 RULER 更 sensitive。
2. **几乎所有 commercial long-context LLM 的 *effective* context 只有 claimed 的 10–20%**, 即 GPT-4 在 QA2/QA3 上实际只能用 4K–16K tokens 做 reasoning。
3. **RAG 在 multi-hop reasoning 上系统性失败**, 因为 dense embedding 和 question 不直接相似, 且 retrieval 不保序。把 top-k 从 5 加到 20 没用。
4. **Fine-tune 137M 的 RMT/ARMT 反而是 SOTA**: 训练时只见过 16K tokens, 推理时外推到 50M tokens, QA1 上还能拿 84%。这打破了"必须用大模型 + 长 pre-training"的迷思, 证明 segment-level recurrence + BPTT + curriculum 已经足以在 toy task 上学出 read/write memory 行为。
5. **Mamba 在 reasoning 复杂度上比 RMT 强但 inference 慢, ARMT 是两者的 sweet spot**。
6. **设计 long-context model 的关键 trade-off**: 信息密度 (Mamba 的 hidden state) vs 可扩展性 (RMT 的 constant memory) vs 可寻址性 (ARMT 的 associative store)。

希望这能给你 build 出足够的 intuition — 这篇 paper 本身的实验做得很扎实, 真正漂亮的 insight 在 Appendix C/H 那些细节里, RMT 的 attention map 是我觉得整篇 paper 最值得反复看的图。
