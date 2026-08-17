---
source_pdf: LEANCODE Understanding Models Better for Code Simplification of.pdf
paper_sha256: 00ce17de3688256c68275f7dfea7b407782f7b9f371e41364fb2c52e0203b916
processed_at: '2026-08-05T12:31:58-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LEANCODE

好, 我换个姿势说. 你就想象咱俩在咖啡馆, 我在餐巾纸上给你画这件事.

---

## 一句话说清楚

**Code 里有差不多一半的 token 是废话, 删掉模型反而更开心. 问题是: 删哪些? LEANCODE 说, 别瞎猜, 让模型自己告诉你哪些重要.**

就这么个事儿. 剩下的全是工程细节.

---

## 问题是什么?

你有一段 Java code, 比如 100 个 token. 喂给 CodeBERT, 它最多吃 512 个 token. 你觉得 100 个 token 里, 真正对 "这段 code 干嘛的" 有用的, 可能就 30 个. 剩下 70 个是 `;`, `{`, `}`, `import`, 局部变量名, boilerplate.

如果能删掉 50 个, 你的 inference 快一倍, cost 降一半, 还不掉点. 这就是 code simplification 的 motivation.

DietCode 和 SlimCode 都想做这件事, 但它们选 token 的眼光不行. LEANCODE 说它们眼光不行, 我来教你怎么看.

参考: DietCode https://arxiv.org/abs/2202.09673, SlimCode https://dl.acm.org/doi/10.1145/3641855

---

## 前人怎么做的, 为什么不行?

### DietCode 的思路: 看 self-attention

Transformer 里有个 self-attention 机制, 每个 token 会 "看" 其他所有 token, 算出一个 attention score. DietCode 的逻辑是: 如果一个 token 被很多其他 token 关注, 那它应该重要.

听起来合理, 但有个致命问题. Self-attention 是在 **pretraining 阶段** 训出来的, 服务于 MLM(Masked Language Model)任务 — 就是随机 mask 掉一些 token, 让模型猜被 mask 的是啥. 为了猜准, 模型必须关注 **所有 token**, 包括 `;` 和 `{`, 因为这些 symbol 对预测相邻 token 也有用.

所以 DietCode 等于在问模型: "你觉得哪些 token 对猜谜游戏重要?" 模型回答: "都挺重要的, 连分号都重要." 这答案对你真正想做的 code search / summarization 任务完全没用.

### SlimCode 的思路: 人工分 8 档

人拍脑袋说: method signature 最重要(第 1 档), function call 第 2 档, ... 分号括号第 8 档(最不重要). 然后按档位删.

问题是: 8 档太粗. 你删到 30% 的时候, 第 1-3 档的 token 都保留完了, 开始删第 4 档的. 但第 4 档里有些 token 其实很重要, 有些不重要, 你分不出来, 只能瞎删. 所以 SlimCode 在 30% removal 之后性能断崖式下跌.

还有一个哲学问题: **人觉得重要的 token, 模型不一定觉得重要**. 你觉得 `return` 语句重要, 但如果模型做 code search 时根本不看 `return`, 你强行保留它也没用.

---

## LEANCODE 的核心 insight

三个字: **问对人**.

你做 code search(classification), 那就去问 **`[CLS]` token**. 因为 classification 的最终 decision 完全来自 `[CLS]` 的 hidden vector. `[CLS]` 关注哪些 token, 哪些 token 就对 classification 有贡献. 这叫 **task-aligned signal**.

你做 code summarization(seq2seq), 那就去问 **decoder**. Decoder 生成 description 时, 会 cross-attend 到 encoder 的 output. Decoder 关注哪些 input token, 哪些 token 就对 generation 有贡献.

这跟你平时问人一样: 你想知道这顿饭好不好吃, 别问厨师(self-attention, 他觉得每道菜都重要), 问食客(CLS attention / EnDe attention, 他只关心自己吃进嘴里的).

参考: BERT 的 CLS token 用法 https://arxiv.org/abs/1810.04805, Transformer cross-attention https://arxiv.org/abs/1706.03762

---

## 公式讲清楚

### CLS attention(公式 5)

$$
s_i = \frac{q_{\text{cls}} \cdot k_i}{\sqrt{d}}, \quad 1 \le i \le n
$$

人话翻译:
- $q_{\text{cls}}$: `[CLS]` 这个特殊 token 的 Query 向量. 你可以理解为 "[CLS] 想找什么".
- $k_i$: 第 $i$ 个 token 的 Key 向量. 你可以理解为 "第 $i$ 个 token 有什么特征".
- $d$: 向量维度(比如 768), $\sqrt{d}$ 是个 normalization factor, 防止 dot-product 数值太大.
- $s_i$: `[CLS]` 对第 $i$ 个 token 的关注程度. 数值越大, 说明 `[CLS]` 越觉得这个 token 对分类有用.

这个 $s_i$ 就是你删 token 的依据 — $s_i$ 小的先删.

### Encoder-Decoder attention(公式 6)

$$
s_i = \frac{q_t \cdot k_i}{\sqrt{d}}, \quad 1 \le i \le n
$$

人话翻译:
- $q_t$: decoder 在生成第 $t$ 个 output token 时的 Query 向量. "decoder 现在想找什么来生成下一个词".
- $k_i$: encoder output 在位置 $i$ 的 Key 向量. "第 $i$ 个 input token 有什么信息".
- $s_i$: 生成第 $t$ 个 output token 时, decoder 对 input token $i$ 的关注程度.

但这里有个麻烦: 你要生成 $T$ 个 output token, 所以每个 input token 有 $T$ 个 attention score. LEANCODE 取 **max** — 只要有一个 output token 高度关注过这个 input token, 就保留它. 这是个保守策略, 宁可多保留.

### Category-local average(公式 3)

$$
\mu_t^c = \frac{\sum_{j=1}^{m} \sum_{t \in p_k, \, p_k \in d'_j, \, L(p_k) \in c} s_t}{n_t^c}
$$

人话翻译:
- $p_k$: 一行 statement(比如 `return x + 1;`).
- $L(p_k)$: 这行 statement 的类型 label. Paper 定义了 21 种: Method Signature, Return, For, While, If, Throw, Try, Catch, ...
- $c$: 某一种类型, 比如 "Return".
- $n_t^c$: token $t$ 在所有 "Return 类型 statement" 里出现的总次数.
- $s_t$: 这个 token 在这个位置的 attention score(CLS 或 EnDe).
- $\mu_t^c$: token $t$ 出现在 "Return statement" 里时的平均重要性.

为什么要分 category? 因为同一个 token `x` 出现在 `return x;` 里和出现在 `int x = 0;` 里, 重要性完全不同. 如果你在 `return x;` 里看到 `x`, 它大概率是返回值, 很重要. 如果在 `int x = 0;` 里, 它只是个临时变量, 可能不重要.

DietCode 把所有 context 的 attention 平均成一个数, 相当于说 "token `x` 的宇宙常数重要性是 0.5". 这就跟说 "一个人叫张三, 他永远是个好人" 一样荒谬 — 张三在工作场合可能很认真, 在酒桌上可能很疯, 你得分场景.

LEANCODE 的做法是建一个 **(token, 场景) → 重要性** 的查找表. 这就 context-aware 了.

---

## 算法其实特别简单

Algorithm 1 翻译成人话:

```
对每个 code snippet:
    1. 查表, 拿到每个 token 在当前 context 下的重要性分数
    2. 按分数从小到大排序
    3. 删掉最小的 SimplifiedRatio × n 个
完事.
```

就这. 没 knapsack, 没 DP, 没 reinforcement learning. 一个 greedy sort + top-k removal.

DietCode 之所以要 knapsack, 是因为它先删 statement 再删 token, 两层结构导致组合爆炸. LEANCODE 直接在 token 层操作, 简单粗暴.

但简单不等于差. Appendix E 的 replacement study 显示: 把 LEANCODE 的 token weight 换成 DietCode 的 removal algorithm, 性能基本不变. 换句话说, **真正起作用的是 token weight 的选择, removal algorithm 怎么写都行**.

---

## 数据告诉了我们什么?

### 主实验(Table 2, 3)

50% removal(删一半 token)时:

| | Code Search (MRR) | Code Summarization (BLEU) |
|---|---|---|
| | CodeBERT / CodeT5 | CodeBERT / CodeT5 |
| Base(不删) | 0.726 / 0.747 | 18.25 / 20.55 |
| DietCode | 0.429 / 0.561 | 14.23 / 14.27 |
| SlimCode | 0.594 / 0.641 | 15.28 / 14.53 |
| LEANCODE | 0.688 / 0.706 | 16.24 / 18.46 |

人话解读:

1. **LEANCODE 删一半 token, code search 只掉 5%**. 0.726 → 0.688. 你把 100 token 的 code 砍成 50 token, 模型几乎没受影响. 这说明 code 里至少一半 token 是噪音.

2. **DietCode 删一半, 掉 41%**. 0.726 → 0.429. 几乎崩了. 因为它删的是 "对 MLM 不重要" 的 token, 但这些 token 对 code search 可能很重要.

3. **SlimCode 在 30% 之后断崖**. 10% removal 时 MRR 反而涨了(0.731 vs 0.726), 因为删掉了一些噪音 token, 让更有信息的 token 进入 512 window. 但 30% 之后开始删重要 token, 8 档优先级不够细, 性能跳水.

4. **Code Summarization 比 Code Search 更难 prune**. LEANCODE 在 50% 时 BLEU 掉 10-11%, 比 search 的 5% 多. 因为 summarization 需要更多细节(method body 里的逻辑), 而 search 主要靠 method signature 匹配.

### 推理时间(Table 2, 3 的 R-T 列)

- Code search CodeT5 @50%: 40min → 25min, 省 37.5%.
- Code summarization CodeT5 @50%: 22min → 13min, 省 40.9%.

基本是线性收益. 删一半 token, 推理快 40%. 因为 self-attention 是 $O(n^2)$, 删一半 token 理论上快 4x, 但实际有 overhead(FC layer, embedding lookup 等), 所以实际快 40% 左右.

### Pruning time(Table 4)

| Method | Code Search @10% | Code Summarization @10% |
|---|---|---|
| DietCode | 9h24m | 1h40m |
| SlimCode | 17m | 45s |
| LEANCODE | 46m33s | 3m32s |

DietCode 慢得离谱(9 小时), 因为要解 knapsack. LEANCODE 46 分钟, 可接受. SlimCode 17 分钟最快, 因为就是 rule-based 查表.

LEANCODE training 额外增加 5% time(因为最后 epoch 要收集 attention scores). 用 5% training cost 换 40% inference speedup, 这个 trade-off 在 production 部署里绝对划算.

### 跨模型迁移(Table 5, 6)— 最有意思的实验

把 CodeT5 简化后的 code 喂给 GPT-4o:

| Method | Code Search Precision @50% | Code Summarization BLEU @50% |
|---|---|---|
| Base(不删) | 0.82 | 10.59 |
| DietCode | 0.776 (-5.37%) | 9.69 (-8.50%) |
| SlimCode | 0.763 (-6.95%) | 10.60 (+0.09%) |
| LEANCODE | 0.81 (-1.22%) | 10.70 (+1.04%) |

LEANCODE 简化后的 code 喂给 GPT-4o, **summarization 的 BLEU 反而比 base 高了 1%**. 这说明 LEANCODE 删掉的是真正无用的 token, 保留的是 model-agnostic 的重要语义信息, 换个模型照样认.

这跟 distillation 的 idea 异曲同工: 用 cheap model(CodeT5)做 pruning 决策, 用 expensive model(GPT-4o)做最终 inference. 成本最优.

---

## 那 21 个 category 到底长啥样?

Paper 用了 21 种 statement category, 我列一下让你有感觉:

| Category | 重要性(CLS attention) | 人话 |
|---|---|---|
| Method Signature | 最高(1.745) | `def bubble_sort(arr):` — 函数签名 |
| Return | 第二(0.202) | `return result` — 返回值 |
| Logging | 第三(0.149) | `log.info("...")` — 日志 |
| Annotation | 第四(0.147) | `@override` — 注解 |
| Variable Declaration | 第五(0.104) | `int x = 0` — 变量声明 |
| Function Invocation | 第六(0.106) | `foo()` — 函数调用 |
| ... | ... | ... |
| Continue | 倒数第三(0.046) | `continue` — 跳过 |
| Case | 倒数第二(0.040) | `case 1:` — switch case |
| Break | 最小(0.047) | `break` — 跳出循环 |

Method Signature 的 attention 是第二名的 8 倍. 这完全符合直觉 — 你看一段 code, 第一眼看的也是函数名和参数, 那才是这段 code 的 "身份证".

但注意, 这是 **平均** 值. 同一个 token 在不同 category 里的 attention 差异巨大, 所以才需要 category-local average. DietCode 的 global average 把这种差异全抹平了.

---

## 架构图怎么看?

Paper 的 Fig. 3 是灵魂, 我用文字描述:

### Fig. 3a: CLS attention for code search

```
Input:  [CLS]  def  bubble_sort  (  arr  )  :  ...  [SEP]  sort  array  [SEP]
                 ↑___________________↑
                  CLS 对这些 token 的 attention
                 ↑___________________↑
                  CLS 对 description token 的 attention

CLS hidden vector → FC layer → {matched, unmatched}
```

关键: 只有 `[CLS]` 的 hidden vector 进 FC layer. 所以 CLS 对哪个 token 关注多, 那个 token 就对 classification 贡献大. DietCode 用的 self-attention 是所有 token 互相看, 信号太 diffuse.

### Fig. 3b: Encoder-Decoder attention for summarization

```
Encoder output:  [def] [bubble] [sort] [(] [arr] [)] [:] [for] [i] [in] ...

Decoder generating "bubble":
  q_t (decoder) → attend to encoder output → 发现 [def] [bubble] [sort] attention 最高
  → 生成 "bubble"
```

Decoder 在生成 "bubble" 时, 高度关注 method signature 里的 `def bubble_sort`. 这就是 EnDe attention 的意义 — 它直接告诉你 "生成 description 时, 哪些 code token 被用到了".

### Fig. 7 / 8 的 heatmap 对比

三张 heatmap 放一起看:

1. **Fig. 7a (CLS attention)**: method signature 的 token 颜色深, body 的 token 颜色浅. → 分类任务只关心 "函数是干嘛的", signature 够了.

2. **Fig. 8a (accumulated self-attention, DietCode 用的)**: method signature 和 body 都有深色 token. → MLM 任务要预测 body 里的 token, 所以 body 也被关注. 信号太 diffuse.

3. **Fig. 8b (EnDe attention)**: body 和 signature 都重要, 但分布不同 — 不同 output token 关注不同 input token. → summarization 需要更多细节, 不能像 search 那样激进 prune.

看懂这三张图, 就懂了 LEANCODE 的整个 thesis.

---

## 放进更大的 landscape

### 跟 LLM prompt compression 的关系

最近很火的 LLMLingua(https://arxiv.org/abs/2310.05736) 做的事很像: 压缩 prompt 来省钱. 区别:

- LLMLingua 用 small LM 的 perplexity 来决定删哪些 token, **online**(每个 input 都要跑一次 small LM).
- LEANCODE 用 downstream task 的 attention, **offline**(train 时算好 lookup table, inference 时直接查).

LEANCODE 的 offline approach 更适合 code 场景, 因为 code 有 syntactic structure(category), 可以 precompute. Natural language 没这种 structure, 所以 LLMLingua 只能 online.

### 跟 vision token pruning 的关系

ViT 里的 DynamicViT(https://arxiv.org/abs/2102.09707) 和 ToMe(https://arxiv.org/abs/2210.08858) 做的是 image token pruning. 思路一样: 用 attention 或 similarity 来决定哪些 image patch 可以删/merge.

LEANCODE 是这个 idea 在 code LLM 上的 instance, 加了 category conditioning 这个 code-specific 的 twist.

### 跟 long context LLM 的关系

现在 code LLM 的 context length:
- StarCoder2: 16k, https://arxiv.org/abs/2402.19173
- DeepSeek-Coder: 16k, https://arxiv.org/abs/2401.14196
- GPT-4o: 128k

即使 128k context, attention 的 $O(n^2)$ 成本仍然巨大. LEANCODE 的 simplification 可以作为 long-context code LLM 的 **preprocessing step**, 把大 repo 压缩到 fit context window. Paper 没展开这个方向, 但我觉得这是最有商业价值的 application.

---

## 我的不爽点

作为 Karpathy, 我看完有几个不舒服:

1. **21 个 category 还是 human prior**. SlimCode 用 8 档, LEANCODE 用 21 档, 本质上都是人定义的. 理想做法应该是 model 自动 learn context, 比如用 AST node type 或 embedding clustering.

2. **Max aggregation 太粗糙**. EnDe attention 一个 input token 有 $T$ 个 score(对应 $T$ 个 output token), 取 max 表示 "曾经重要过". 这太保守了. 可以用 top-k percentile 或 attention entropy, 更 principled.

3. **只测了 Java**. Python / JavaScript / Go 的 code structure 不一样, category 定义可能要调整. Paper 在 limitations 里承认了, 但没给数据.

4. **没考虑 attention head 异构性**. Multi-head attention 里不同 head 关注不同东西(Clark et al. 2019, https://arxiv.org/abs/1906.04341). LEANCODE 把所有 head average 起来, 信号被 dilute. 可以选 head subset.

5. **没跟 token merging 对比**. ToMe(https://arxiv.org/abs/2210.08858) 在 ViT 上用 bipartite matching 把相似 token merge 而非 prune. Code 里相似 token(如多个 `int x, y, z`)也可以 merge, 信息损失更小. LEANCODE 只做 removal, 没探索 merging.

6. **GPT-4o 实验太浅**. 只测了 400 个 sample, prompt 也是手写的. 如果能测 10k+ sample, 跟 fine-tuned CodeT5 对比, 更有说服力.

---

## 如果我来做 follow-up

1. **Learned context**: 用 AST parser 自动提取 statement type, 或者用 embedding clustering 发现 latent context. 摆脱手工 21 category.

2. **Online + Offline 混合**: 用 LEANCODE 的 offline lookup table 做 coarse pruning(删 30%), 再用 lightweight probe(小 MLP)做 fine pruning(再删 20%). 类似 coarse-to-fine 的思路.

3. **Token merging**: 把相似 token merge 成一个, 而不是直接删. 比如多个 variable declaration `int x; int y; int z;` merge 成 `int x, y, z;`. 信息保留更多.

4. **Long-context 场景**: 把 LEANCODE 用在 repo-level code understanding 上. 一个 repo 可能有 10k+ token, 用 LEANCODE 压缩到 4k, 喂给 long-context LLM. 这才是 production 场景.

5. **跟 weight pruning 叠加**: weight pruning(magnitude / movement pruning)+ input pruning(LEANCODE)+ quantization(int8/int4). 三者叠加可能 10x speedup. Paper 完全没探索这个组合.

---

## 给你的 takeaways

1. **Pruning 的关键不是算法, 是 signal**. Algorithm 1 就是个 sort + top-k, 5 行代码. 真正起作用的是你用什么 attention 来衡量 token 重要性.

2. **Task-aligned signal > Pretraining signal > Human prior**. CLS attention > self-attention > 人工 8 档. 这个 ranking 在任何 pruning 任务里都成立.

3. **Context matters**. 同一个 token 在不同 context 里重要性不同. Global average 会抹平这种差异. 哪怕只分 5 个 bucket, 也比 global average 强.

4. **Code 里 50% token 是冗余的**. 这是 LEANCODE 实验直接证明的. 对 code LLM 的 data efficiency 有大意义.

5. **Cheap model pruning + Expensive model inference = cost optimal**. LEANCODE 在 CodeT5 上学到的 pruning 信号迁移到 GPT-4o 还 work. 这是 production 部署的 practical insight.

---

## Reference 汇总

核心 paper:
- LEANCODE: https://arxiv.org/abs/2502.06018 (推测)
- DietCode: https://arxiv.org/abs/2202.09673
- SlimCode: https://dl.acm.org/doi/10.1145/3641855
- CodeBERT: https://arxiv.org/abs/2002.08155
- CodeT5: https://arxiv.org/abs/2109.00859
- CodeSearchNet: https://arxiv.org/abs/1909.09436

Attention probing:
- "What Does BERT Look At?": https://arxiv.org/abs/1906.04341
- "Are 16 Heads Really Better than 1?": https://arxiv.org/abs/1905.10650
- Transformer: https://arxiv.org/abs/1706.03762
- BERT: https://arxiv.org/abs/1810.04805

Token pruning / compression:
- PoWER-BERT: https://arxiv.org/abs/2001.08950
- SpAtten: https://arxiv.org/abs/2012.09719
- DynamicViT: https://arxiv.org/abs/2102.09707
- EViT: https://arxiv.org/abs/2101.09883
- ToMe: https://arxiv.org/abs/2210.08858
- Movement Pruning: https://arxiv.org/abs/2006.00756
- LLMLingua: https://arxiv.org/abs/2310.05736
- LongLLMLingua: https://arxiv.org/abs/2310.06839
- StreamingLLM: https://arxiv.org/abs/2309.17453
- H2O: https://arxiv.org/abs/2306.14048

Code LLM:
- StarCoder2: https://arxiv.org/abs/2402.19173
- DeepSeek-Coder: https://arxiv.org/abs/2401.14196
- CodeLlama: https://arxiv.org/abs/2308.12950
- FlashAttention: https://arxiv.org/abs/2205.05198
- Autofocus: https://arxiv.org/abs/1909.00692

就这样. 核心就一句话: **问对人, 删对 token**. 剩下的都是 implementation detail.

---

# LEANCODE 深度技术讲解 — 从 Attention 视角重审 Code Pruning

作为 Karpathy, 我看完这篇 paper 之后, 第一反应是: 这篇 paper 的核心 insight 其实非常朴素, 但作者把它讲得很复杂。本质上是一个 **"用对的任务信号去指导 pruning"** 的故事, 与我们在 neural network compression、distillation、token pruning(如 DynamicViT、ToMe)中看到的 logic 完全同源。下面我把 intuition 拆开讲, 把公式里每个变量的物理意义讲透, 并把这篇文章放进更大的 code-LLM efficiency landscape 里去看。

---

## 1. 问题动机: 为什么 DietCode / SlimCode 都没做对?

Pre-trained code LLM(CodeBERT 512 token limit, CodeT5, CodeGen, GPT-4o)在 long input 上 compute 复杂度是 $O(n^2 \cdot d)$ 量级的(self-attention 是 $n^2$), 所以 input token pruning 是天然想做的优化。前人两条路:

- **DietCode (FSE 2022)**: 用 encoder self-attention 的 global average $\mu_t$ 作为 token 的"宇宙常数"重要性, 然后 knapsack 优化删 token。
- **SlimCode (FSE 2024)**: 人工把 token 分成 8 个 priority tier, 用 human prior 决定重要性。

LEANCODE 的核心 claim 是: 这两种 prior 都不对, 真正应该用的 prior 是 **downstream task 自己产生的 attention signal**。这跟我们在 distillation 里说 "teacher 的 logits 比 hard label 信息丰富" 是一回事 — 用模型自己更贴近 task 的 signal, 不要用 pretraining 的 signal 或 human 的 signal。

paper 的 RQ-1/RQ-2/RQ-3 三个 empirical study 就是在论证这件事。我特别想强调 RQ-3 的结论: **encoder self-attention 服务于 MLM/RTD 预训练任务, 不是服务下游 task 的**。这个区分非常重要:

- CodeBERT 预训练用 MLM(双向 bimodal)+ RTD(Replaced Token Detection, unimodal)。
- MLM 让 encoder 关注 "code body 里的 keyword 和 separator 用于 token-level reconstruction", 因为 MLM 要预测任意 masked token。
- 而 downstream classification(code search)的最终 decision 完全来自 `[CLS]` 这个 token 的 hidden vector, 所以 `[CLS]` attention 才是 task-aligned signal。
- downstream seq2seq(code summarization)的 generation 完全来自 decoder cross-attending encoder output, 所以 encoder-decoder attention 才是 task-aligned signal。

这就是文章最 fundamental 的 insight, 我建议你把这个 idea 直接迁移到任何 encoder-only / encoder-decoder 的 pruning 框架里。

参考链接:
- CodeBERT: https://arxiv.org/abs/2002.08155
- CodeT5: https://arxiv.org/abs/2109.00859
- DietCode: https://arxiv.org/abs/2202.09673
- SlimCode (PACMSE FSE 2024): https://dl.acm.org/doi/10.1145/3641855
- CodeSearchNet: https://arxiv.org/abs/1909.09436
- Transformer 原始 attention 公式: https://arxiv.org/abs/1706.03762

---

## 2. 公式逐个拆解, 讲透变量含义

### 2.1 问题形式化: 一个加权最小化组合优化

paper 的公式 (1):

$$
\text{minimize} \ \sum_{i=1}^{n_j} w_i \, x_i, \quad \text{s.t.} \ \sum_{i=1}^{n_j} x_i = \mathcal{X}
$$

变量含义:
- $D = \{d_1, \dots, d_m\}$: 数据集, $m$ 个 code snippet。
- $d_j = \{t_1, \dots, t_{n_j}\}$: 第 $j$ 个 snippet, 长度 $n_j$。
- $w_i$: token $t_i$ 的重要性分数(越大越不能删)。
- $x_i \in \{0, 1\}$: 是否删除该 token 的 binary indicator。
- $\mathcal{X} = \text{SimplifiedRatio} \times n_j$: 这个 snippet 要删的 token 总数。
- 下标 $i$ 跑过 snippet 里所有 token, 上界 $n_j$ 是该 snippet 的长度。

注意: 这是一个 **per-snippet 的 0/1 knapsack 的退化形式**(权重全 1, 容量固定 $\mathcal{X}$), 实际上就是 "挑出 $\mathcal{X}$ 个 $w_i$ 最小的 token 删掉" — 所以 Algorithm 1 的实现就是一个 greedy sort + top-k removal, 并不需要 DP。这点 paper 没明说, 但从 Algorithm 1 line 7 的 "Add {index:token with lowest s_t}" 可以确认。

Intuition: 一旦你接受了 "token 有 importance score $w_i$", 那么 pruning 决策就 trivial 了, 难的是 **怎么定义 $w_i$**。所以接下来三个公式才是 paper 的 meat。

### 2.2 DietCode 的 global average(被批判的 baseline)

公式 (2):

$$
\mu_t = \frac{\sum_{j=1}^{m} \sum_{t \in d'_j} s_t}{n_t}
$$

变量:
- $d'_j$: 训练集里第 $j$ 个 snippet。
- $s_t$: token $t$ 在某个出现位置的 self-attention score。
- $n_t$: token $t$ 在整个训练集出现的总次数(所有 context 加起来)。
- $\mu_t$: token $t$ 的 "全局宇宙常数" importance。

致命问题: 同一个 token `accumulate` 出现在 method signature 里和出现在 for-loop condition 里, attention 完全不同 — paper 的 Fig. 1 显示 top-10 variance 的 token 全是 semantic-rich 的(`accumulate`, `pure`, `commerce`), 而 bottom-10 全是 symbol(`{`, `;`, 数字)。Global average 把这两种迥异 context 平均成同一个数, 等于把信号淹没在噪声里。Table 7 里 `Global_variance` 列基本都在 1.0–1.5 这个量级, 而 `Category-local` 的 `Local_variance` 普遍降到 0.01–0.1 量级 — variance reduction 1–2 个数量级, 这是 category conditioning 的核心收益。

### 2.3 LEANCODE 的 category-local average

公式 (3):

$$
\mu_t^c = \frac{\sum_{j=1}^{m} \sum_{t \in p_k,\, p_k \in d'_j,\, L(p_k) \in c} s_t}{n_t^c}
$$

变量:
- $p_k$: 一个 statement(code 的一行或一个语法单元)。
- $L(p_k)$: statement $p_k$ 的 label(category)。 paper 用了 21 个 category: Method Signature, Return, Throw, For, While, If Condition, Function Invocation, Variable Declaration, Arithmetic, Annotation, Logging, Switch, Case, Break, Continue, Setter, Getter, Synchronized, Try, Catch, Finally。
- $c \in C$: 一个具体的 category。
- $n_t^c$: token $t$ 在所有属于 category $c$ 的 statement 里出现的次数。
- $s_t$: 这里是关键 — paper 让 $s_t$ 可以是 **CLS attention**(分类任务)或 **encoder-decoder attention**(seq2seq 任务), 而不仅仅是 encoder self-attention。
- $\mu_t^c$: token $t$ 在 category $c$ 这个 context 下的局部平均 importance。

Intuition: 把 token 的 importance 表示成一个 **(token, context) → score** 的查找表, 而不是 **token → score**。这是把 lexical prior 和 syntactic prior 解耦的组合表示。它跟 NLP 里 contextualized embedding 的 motivation 同源 — 一个词的 meaning 取决于 context, 那它的 importance 也应该取决于 context。

为什么不直接用 dynamic attention(test time 每个 input 算一次)?paper 3.2.1 解释: 算 dynamic attention 要过 12 层 transformer block, 等于已经做了一次 forward pass, 再 prune 就没意义了。这是 **"用 train-time signal 近似 test-time importance"** 的标准做法, 跟 quantization calibration、distillation teacher alignment 是一个套路。

### 2.4 三个 attention 公式的对比 — 这是 paper 的灵魂

paper 附录 B 给的三个公式, 我建议你把它们并排看:

**公式 (4) DietCode 的 accumulated self-attention:**
$$
s_i = \frac{\sum_{j=1}^{n} q_j \cdot k_i}{\sqrt{d}}
$$
- $q_j$: 第 $j$ 个 token 的 Query 向量。
- $k_i$: 第 $i$ 个 token 的 Key 向量。
- $d$: head 维度, $\sqrt{d}$ 是 scaled dot-product 的标准 normalization。
- 含义: token $i$ 被 **所有 input token** 关注的总和。这是 "token 在 pretraining 任务里被 collective 关注多少"。

**公式 (5) CLS attention:**
$$
s_i = \frac{q_{\text{cls}} \cdot k_i}{\sqrt{d}}, \quad 1 \le i \le n
$$
- $q_{\text{cls}}$: `[CLS]` token 的 Query 向量。
- 含义: `[CLS]` 这个 token 对 token $i$ 的关注程度。由于 `[CLS]` 的 hidden vector 是唯一送入 FC layer 做 classification 的, 所以这个 score 直接反映 token $i$ 对最终 classification logit 的影响力。

**公式 (6) Encoder-Decoder attention:**
$$
s_i = \frac{q_t \cdot k_i}{\sqrt{d}}, \quad 1 \le i \le n
$$
- $q_t$: decoder 在生成第 $t$ 个 output token 时的 Query 向量(从 decoder 当前 hidden state 投影出来)。
- $k_i$: encoder output 在位置 $i$ 的 Key(从 encoder 最后一层 hidden state 投影)。
- 含义: 在生成第 $t$ 个 output token 时, decoder 对 input token $i$ 的关注程度。
- 因为生成 $T$ 个 output token, 所以一个 input token 有 $T$ 个这样的 score。 paper 取 **max over $t$** 作为这个 token 的 representative score — 因为 max 表示 "这个 token 至少在某一步 generation 里被高度关注过"。

注意 paper 用的是 last encoder/decoder layer 的 attention。 为什么 last layer? 因为最后一层最接近 task head, 包含最高 level 的 semantic abstraction(参考 probing literature: Clark et al. 2019 "What Does BERT Look At?", https://arxiv.org/abs/1906.04341, 表明 BERT 不同 layer 关注不同 linguistic phenomenon, last layer 最 task-specific)。

---

## 3. 算法层面 — Algorithm 1 拆解

Algorithm 1 是个非常 simple 的 greedy removal:

```
INPUT: D, S = {t, c, μ_t^c}, SimplifiedRatio
OUTPUT: D^c (simplified dataset)
1: D^c ← D
2: for j = 1 to m do
3:   removedTokens ← {}
4:   X ← SimplifiedRatio × n_j
5:   removedTokenNum ← 0
6:   while removedTokenNum < X do
7:     Add {index: token with lowest s_t (∈ d_j^c, ∉ removedTokens)} into removedTokens
8:     removedTokenNum updates
9:   d_j^c ← d_j^c / removedTokens[1:X]
10: return D^c
```

关键点:

- **Token-level, 不做 statement-level removal**。DietCode 先删 statement 再删 token, LEANCODE 跳过 statement 这层。这点在 Appendix E 的 replacement study 里被验证: 把 LEANCODE 的 token weight 套到 DietCode 的 removal 算法上, code search 50% removal 的性能从 0.688(LEANCODE) 掉到 0.682(replacement); code summarization 从 BLEU 16.24 掉到 16.73 — wait, 这个数字是 BLEU 高了, 反而更好? 让我重读 Appendix E。其实 Table 10 的 BLEU 比 Table 3 的 BLEU 略高, 说明 LEANCODE 自己的 token-level greedy 在 summarization 上反而比 statement-level 略差。但 search 上 LEANCODE token-level 更好。这说明 removal algorithm 本身影响不大, **真正的工作量是 token weight 的选择**。这个 finding 很重要, paper 没充分强调。

- 时间复杂度: $O(n_j \log n_j)$ per snippet(sort + top-k), 所以 pruning time 在 Table 4 里 LEANCODE 比 SlimCode 慢 2-4 倍是因为额外的 tokenization 和 statement class matching, 不是因为算法本身复杂。DietCode 慢得多是因为它要解 knapsack, 复杂度 $O(n_j \cdot \mathcal{X})$。

---

## 4. 实验数据深度解读

### 4.1 主结果 — Table 2 / Table 3

我把 50% removal(最 aggressive 的设定)的关键数字拎出来:

**Code Search (MRR, Base = 0.726 CodeBERT / 0.747 CodeT5):**

| Method | CodeBERT @50% | CodeT5 @50% |
|---|---|---|
| DietCode | 0.429 (-40.90%) | 0.561 (-24.89%) |
| SlimCode | 0.594 (-18.18%) | 0.641 (-14.19%) |
| LEANCODE | 0.688 (-5.23%) | 0.706 (-5.48%) |

LEANCODE vs DietCode 在 CodeBERT 上的相对 improvement: (0.688 - 0.429)/0.429 = 60.4%, 这就是 abstract 里那个 60% 的来源。vs SlimCode: (0.688 - 0.594)/0.594 = 15.8%, 即 abstract 里的 16%。

**Code Summarization (BLEU-4, Base = 18.25 CodeBERT / 20.55 CodeT5):**

| Method | CodeBERT @50% | CodeT5 @50% |
|---|---|---|
| DietCode | 14.23 (-22.02%) | 14.27 (-30.55%) |
| SlimCode | 15.28 (-16.27%) | 14.53 (-29.29%) |
| LEANCODE | 16.24 (-11.01%) | 18.46 (-10.17%) |

vs DietCode CodeT5: (18.46 - 14.27)/14.27 = 29.4% → abstract 里的 29%。
vs SlimCode CodeT5: (18.46 - 14.53)/14.53 = 27.0% → abstract 里的 27%。

**Intuition 强化**:
1. 在 10-20% removal 区间, 三种方法差距不大, SlimCode 在 10% 时甚至略涨(BLEU +0.68% CodeBERT)。这说明 low-ratio removal 是 easy regime, 任何方法都能 work。
2. 在 30% 之后 SlimCode 急剧恶化(40% 时 MRR 从 0.703 跌到 0.632, 跌幅从 3.58% 跳到 12.94%)。原因: SlimCode 只有 8 个 priority level, 30% 之后必须删同 priority 的 token, 而 priority 相同时 SlimCode 没有 fine-grained 信号区分。这是 discrete prior 的根本 limitation。
3. LEANCODE 在 50% 时仍只掉 5.23% MRR, 这意味着一个 100 token 的 snippet 删到 50 token, code search 准确率几乎不变 — 这暗示 **code 里至少 50% 的 token 是 redundant 的**, 这个观察本身对 code LLM 的 data efficiency 有大意义。

### 4.2 Inference time 收益

Table 2 / 3 的 R-T 列:

- Code search CodeT5 @50%: 40m → 25m, 节省 37.5%。
- Code summarization CodeT5 @50%: 22m → 13m, 节省 40.9%。
- Ratio ≈ 0.7 (search) / 0.75 (CodeT5 summarization) / 0.5 (CodeBERT summarization)。

为什么 CodeBERT summarization 的 ratio 只有 0.5? 因为 CodeBERT + Transformer decoder 的组合没优化, decoder 部分的 fixed cost 占比大, 删 input token 省不了 decoder 的 self-attention。这跟现代 LLM 推理时 prefill vs decode 的 trade-off 一样(参考 https://arxiv.org/abs/2205.05198 FlashAttention, https://lmsys.org/blog/2023-02-21-vicuna/ 关于 prefill 占主导的讨论)。

### 4.3 Pruning time (Table 4)

- DietCode code search @10%: 9h24m(因为 knapsack)。
- SlimCode: 17m(rule-based)。
- LEANCODE: 46m33s(tokenize + statement class matching + greedy sort)。
- LEANCODE training 增加 ~5%: 因为在最后 epoch 收集 attention scores, 7 epoch 正常 + 1 epoch 53min, 总 315.5min vs 300min。

这个 trade-off 是合理的: 用 5% training time 换 37-41% inference time, 在 production 部署里 ROI 很高。

### 4.4 跨模型迁移 — Table 5 / 6

这是我觉得最有意思的实验。把 CodeT5 简化的 code 喂给 GPT-4o:

**GPT-4o Code Search (Precision, Base = 0.82):**
- LEANCODE @50%: 0.81 (-1.22%)
- SlimCode @50%: 0.763 (-6.95%)
- DietCode @50%: 0.776 (-5.37%)

**GPT-4o Code Summarization (BLEU, Base = 10.59):**
- LEANCODE @50%: 10.70 (+1.04%, 注意是涨!)
- SlimCode @50%: 10.60 (+0.09%)
- DietCode @50%: 9.69 (-8.50%)

注意两点:
1. **LEANCODE @30% code search precision 反而比 base 高 0.49%**(0.828 vs 0.82)。这呼应 Section 4.1 的观察: pruning 去掉 low-quality token(symbol-like), 让更多 informative token 进入 512 window, 反而提升 GPT-4o 的判断。
2. BLEU 的 absolute 值(10.59)只有 CodeT5(20.55)的一半, 因为 GPT-4o 没 fine-tune, generated description 跟 ground truth 的 lexical overlap 低。但 relative ranking LEANCODE > SlimCode > DietCode 仍然成立, 说明 LEANCODE 的 simplification 信号是 model-agnostic 的 — 这是一个很强的 transferability 证据。

Intuition: LEANCODE 学到的是 "code 哪部分语义重要", 这种 prior 跟具体 model 无关, 就像 human 也能识别 method signature 比 separator 重要一样。但 LEANCODE 比 SlimCode(用 human prior)还好的原因在于: model 的 prior 比 human 的 prior 更 fine-grained(21 category × per-token score vs 8 tier)。

---

## 5. Architecture 图解读 — Fig. 3 / Fig. 7 / Fig. 8

paper 的 Fig. 3 是核心架构示意, 我建议你重点看:

**Fig. 3a — CLS attention for code search:**
- Input: `[CLS] code_tokens [SEP] description_tokens [SEP]`。
- `[CLS]` 的 hidden vector $v_{\text{cls}} = \sum_i s_i v_i$ (weighted sum over all token vectors, weights 来自 self-attention)。
- $v_{\text{cls}}$ → FC layer → binary classification {matched, unmatched}。
- 关键 insight: 只有 $v_{\text{cls}}$ 进 FC, 所以 $s_i$(CLS 对 token $i$ 的 attention)就是 token $i$ 对最终 logit 的 gradient proxy。

**Fig. 3b — Encoder-Decoder attention for summarization:**
- Decoder 在生成第 $t$ 个 token 时, $q_t$ 与 encoder output 的每个 $k_i$ 做 dot-product, 得到 attention distribution。
- 这个 distribution 决定 decoder 看 input 的哪部分来生成当前 output token。
- Fig. 3b 的例子: 生成 "bubble" 时, decoder 高度 attend 到 method signature 里的 `def bubble`。

**Fig. 7 / Fig. 8 的 heatmap:**
- Fig. 7a (CLS attention on code): method signature 的 token 颜色深, body 的 token 颜色浅 → 分类任务只关心 "function 是干嘛的", signature 足够。
- Fig. 8a (accumulated self-attention, DietCode 用的): method signature 和 body 都有深色 token, 因为 MLM 要预测 body 里的 keyword, 必须关注 body → DietCode 用的信号太 "diffuse"。
- Fig. 8b (EnDe attention): body 和 signature 都重要, 因为不同 output token 关注不同 input token → 这就是为什么 summarization 要保留 body, 而 search 可以删 body。

这三个 heatmap 完美说明了 "不同 task 需要不同的 attention signal 来指导 pruning" 这件事。

---

## 6. 把 LEANCODE 放进更大的 landscape

### 6.1 Token pruning 在 vision / NLP 里的 analog

- **Vision**: DynamicViT (https://arxiv.org/abs/2102.09707), EViT (https://arxiv.org/abs/2101.09883), ToMe (https://arxiv.org/abs/2210.08858) — 都用 attention 或者 similarity-based merging 来 prune ViT 的 token。
- **NLP**: PoWER-BERT (https://arxiv.org/abs/2001.08950), SpAtten (https://arxiv.org/abs/2012.09719), Length-Adaptive Transformer — 用 CLS attention 或 head importance 来 prune word token。

LEANCODE 的 novelty 在于: 它把 "task-specific attention"(CLS / EnDe)和 "contextual prior"(category-local average)两个 idea 组合起来。在 NLP 里 PoWER-BERT 用过 CLS attention, 但没做 category conditioning; 在 code 里 DietCode 用过 attention, 但用错了(self-attention 而非 CLS/EnDe)。LEANCODE 是这两个 idea 的交集。

### 6.2 跟 LLM context compression 的关系

最近 (2024) LLM context compression 很火:
- LLMLingua (https://arxiv.org/abs/2310.05736): 用 small LM 的 perplexity 来 prune prompt token。
- LongLLMLingua (https://arxiv.org/abs/2310.06839): 改进为 question-aware。
- StreamingLLM (https://arxiv.org/abs/2309.17453): attention sink 现象。
- H2O (https://arxiv.org/abs/2306.14048): Heavy-Hitter Oracle, eviction-based。

LEANCODE 跟这些工作的核心区别:
1. LEANCODE 是 **offline / train-time pruning**, 把 token weight 学出来存成 lookup table, inference 时直接用; 而 LLMLingua 等是 **online / inference-time pruning**, 每个 input 都要跑一次 small LM 算 perplexity。
2. LEANCODE 用 **下游 task attention**(CLS / EnDe), 而 LLMLingua 用 **generative perplexity**(跟下游 task 的对齐度弱一些)。
3. LEANCODE 在 code 上, code 的 syntactic structure 让 category conditioning 成为可能; 在 natural language 上, category 概念弱一些, 所以 NLP 的 pruning 更多用 perplexity 而非 category-local average。

### 6.3 跟 model pruning / distillation 的关系

LEANCODE 实际上是 **input-side pruning**, 跟 weight pruning(如 magnitude pruning, movement pruning https://arxiv.org/abs/2006.00756)是 complementary 的。可以叠加:

- 先 weight pruning(把 model 的 weight 稀疏化)。
- 再 input pruning(LEANCODE 把 input token 减半)。
- 再 quantization(int8 / int4)。
- 三者叠加可能达到 10x+ 的 inference speedup。

但 LEANCODE 没探索这个组合, 是一个 future direction。

### 6.4 跟 recent code LLM 的 context length 问题

现代 code LLM 的 context length:
- StarCoder2 (2024): 8k–16k context, https://arxiv.org/abs/2402.19173
- DeepSeek-Coder: 16k context, https://arxiv.org/abs/2401.14196
- CodeLlama: 16k context, https://arxiv.org/abs/2308.12950

LEANCODE 的 motivation 在 long context 下仍然成立: 即使 16k context, attention 的 $O(n^2)$ 成本仍然很高, 而且 code repo 经常超过 16k。LEANCODE 的 simplification 可以作为 long-context code LLM 的 **preprocessing step**, 把 repo 压缩到 fit context window。这是个潜在的应用方向, paper 没展开。

---

## 7. Critique & 我会怎么改进

作为 Karpathy, 我看完有几个不舒服的地方:

1. **Category 的定义太粗糙**: 21 个手工 category, 跟 SlimCode 的 8 tier 本质上一样是 human prior, 只是更细。理想做法应该是 **learned context**(用 embedding clustering 或者 AST node type), 让 model 自动发现 context 而不是手工定义。这跟 BERT 的 positional encoding vs learned positional encoding 的张力同源。

2. **没探索单 token 的 contextualized importance**: 现在 LEANCODE 用 train-time 的 category-local average 来近似 test-time 的 importance。一个更好的方法是 **lightweight probe**: 训练一个小 MLP, 输入 token embedding + context feature, 输出 predicted importance score。这跟 BERT-pruning 里的 "Head Importance Estimation" (https://arxiv.org/abs/1905.10650) 类似。

3. **没考虑 attention head 的异构性**: multi-head attention 里不同 head 关注不同 phenomenon(Clark et al. 2019)。LEANCODE 把所有 head 的 attention average 起来, 信号被 dilute。更精细的做法是选 head subset。CodeBERT 12 layer × 12 head = 144 个 head, 信号丰富。

4. **没评估 out-of-distribution**: 只在 Java + CodeSearchNet 上测。Pruning 的 generalization 在 cross-language / cross-domain 上是否成立, 没说。Section 7 limitations 提到了, 但没给数据。

5. **Encoder-Decoder attention 的 max aggregation 是 ad-hoc**: 一个 input token 有 $T$ 个 EnDe attention(对应 $T$ 个 output token), 取 max 表示 "曾经重要过"。更 principled 的做法是 weighted sum, 或者 top-k percentile, 或者直接用 attention entropy。Max 是最激进的(保留只要被关注过一次的 token), 这可能解释了为什么 summarization 上 LEANCODE 在 50% removal 时 BLEU 仍掉 10%。

6. **跟 token merging (ToMe) 的对比缺失**: ToMe 在 ViT 上用 bipartite matching 把相似 token merge 而不是 prune。Code 里相似 token(如多个 variable declaration)也可以 merge。LEANCODE 只做 removal, 没探索 merging, 信息损失更大。

7. **Training cost 增加的替代方案**: 现在 LEANCODE 在最后 epoch 算 attention, 增加 5% training cost。可以用 **free-running**(只用前几个 batch 算一次 attention, 不在整个 epoch 里收集), 类似 quantization calibration 的 representative dataset 思路。这能进一步降低 pruning overhead。

---

## 8. 给你的 actionable takeaways

如果你要把 LEANCODE 的 idea 用到自己的工作里, 我建议:

1. **任何 input pruning 任务, 先问自己 "task signal 在哪里"**: classification 看 CLS, seq2seq 看 cross-attention, generation 看 self-attention of current token, MLM 看 masked position attention。DietCode 的错误就是用 MLM 的 signal 服务 classification task。

2. **Contextualize importance score**: 不要用 global average。哪怕只有 5 个 bucket, 也比 global average 强。LEANCODE 的 variance reduction 实验证明了这点。

3. **Greedy removal 通常够用**: Algorithm 1 这么 simple 的 greedy sort + top-k 在大多数 case 已经接近 optimal(因为权重全 1 的 knapsack 就是 sort 问题)。不要花时间写复杂 DP / knapsack。

4. **跨 model transfer 是 free lunch**: LEANCODE 在 CodeT5 上学到的 simplification 直接迁移到 GPT-4o 还 work。这意味着你可以用 cheap model 做 pruning 信号, 然后用 expensive model 做 inference, 整体成本最低。

5. **Long-context LLM 时代, pruning 仍然 relevant**: 16k / 32k context 的 attention cost 仍然 $O(n^2)$, pruning 是 orthogonal optimization。

---

## 9. 一页纸 summary

| 维度 | DietCode | SlimCode | LEANCODE |
|---|---|---|---|
| Signal source | Encoder self-attention | Human rule (8 tier) | CLS / EnDe attention |
| Aggregation | Global average | Manual priority | Category-local average |
| Removal granularity | Statement + Token | Token | Token |
| Task alignment | Pretraining (MLM/RTD) | Human prior | Downstream task |
| 50% Code Search MRR drop (CodeBERT) | -40.90% | -18.18% | -5.23% |
| 50% Sum BLEU drop (CodeT5) | -30.55% | -29.29% | -10.17% |
| Cross-model to GPT-4o | -5.37% | -6.95% | -1.22% |
| Pruning time @10% search | 9h24m | 17m | 46m33s |

LEANCODE 的核心贡献是 **"用对的 attention signal(下游 task-aligned) + 对的 aggregation(按 syntactic context 分组)来指导 input pruning"**, 把 DietCode 40% 的性能损失压到 5%。这是一个典型的 "right signal, right place" paper, 工程价值高, scientific insight 中等(因为 CLS / EnDe attention 的 task-alignment 是 well-known, 只是被 code LLM community 之前忽略了)。

---

## Reference 链接汇总

核心 paper:
- LEANCODE (本 paper): https://arxiv.org/abs/2502.06018 (推测)
- DietCode: https://arxiv.org/abs/2202.09673
- SlimCode: https://dl.acm.org/doi/10.1145/3641855
- CodeBERT: https://arxiv.org/abs/2002.08155
- CodeT5: https://arxiv.org/abs/2109.00859
- CodeT5+: https://arxiv.org/abs/2305.07922
- CodeSearchNet: https://arxiv.org/abs/1909.09436

Background / attention probing:
- "What Does BERT Look At?": https://arxiv.org/abs/1906.04341
- "Are 16 Heads Really Better than 1?": https://arxiv.org/abs/1905.10650
- Transformer: https://arxiv.org/abs/1706.03762
- BERT: https://arxiv.org/abs/1810.04805

Related pruning / compression:
- PoWER-BERT: https://arxiv.org/abs/2001.08950
- SpAtten: https://arxiv.org/abs/2012.09719
- DynamicViT: https://arxiv.org/abs/2102.09707
- ToMe: https://arxiv.org/abs/2210.08858
- Movement Pruning: https://arxiv.org/abs/2006.00756
- LLMLingua: https://arxiv.org/abs/2310.05736
- LongLLMLingua: https://arxiv.org/abs/2310.06839
- StreamingLLM: https://arxiv.org/abs/2309.17453
- H2O: https://arxiv.org/abs/2306.14048

Code LLM landscape:
- StarCoder2: https://arxiv.org/abs/2402.19173
- DeepSeek-Coder: https://arxiv.org/abs/2401.14196
- CodeLlama: https://arxiv.org/abs/2308.12950
- FlashAttention: https://arxiv.org/abs/2205.05198
- Autofocus (code attention): https://arxiv.org/abs/1909.00692

OpenAI pricing (paper 里引用): https://openai.com/api/pricing/

希望这个深度讲解帮你 build 起对 code LLM pruning 的 intuition。如果你要在我课上做 follow-up, 我建议先复现 Fig. 7 / Fig. 8 的 heatmap 对比 — 那三张图是整个 paper 的 visual thesis, 看懂了那三张图就懂了 LEANCODE。
