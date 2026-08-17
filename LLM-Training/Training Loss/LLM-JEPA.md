---
source_pdf: LLM-JEPA.pdf
paper_sha256: a257ef291e3097a5b8e3ba774568647bf06a26f4d218b0f58b27f5b69e2ae66d
processed_at: '2026-08-05T15:34:59-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# LLM-JEPA 人话版

Karpathy, 好, 之前那个太 technical 了, 我换个讲法, 就当我们在咖啡馆聊。

---

## 一句话说清楚

现在所有 LLM 都靠 "猜下一个 token" 训练。这篇 paper 说: 你猜 token 猜得再准, 你脑子里的 abstract representation 可能一塌糊涂。我们加一个 extra loss, 逼模型的 internal representation 变得有 structure, 结果 accuracy 涨了一大截, 还不怕 overfitting。

就这么简单。

---

## 问题出在哪

你训一个 LLM, 它在 next-token prediction 上 loss 降得很漂亮。但你有没有想过: 模型内部那个 hidden state, 到底有没有 "理解" 这句话?

答案是: **不一定**。

模型可以学会一堆 surface-level 的 pattern — "the" 后面常跟名词, 函数名后面常跟括号 — 但它内部的 representation 可能完全是乱的。两个语义等价的东西 (比如一句英文描述和它对应的 regex 代码), 在 latent space 里可能隔了十万八千里。

Figure 3 里那个实验特别直观:
- 你拿 baseline LLM, 只训 next-token prediction, accuracy 51.95%
- 你拿同一个 model, 加上 JEPA loss, accuracy 71.10%
- 两边的 next-token prediction loss **几乎一模一样** (图上 blue 和 yellow 线完全 overlap)

所以模型的 "猜 token 能力" 没变, 但 accuracy 涨了 20 个点。涨的这 20 个点全来自 representation 变好了。模型 "想明白" 了 text 和 code 是同一个东西的两个面。

---

## JEPA 到底在干嘛

想象你在教小孩认动物。

**方式 A (next-token prediction 的思路)**: 你给小孩看一万张猫的照片, 每张都让他描述 "这是一只..."。他学会说 "猫" 了。但他脑子里猫的概念是什么? 可能是一堆 pixel pattern 的 memorization, 换个角度拍的猫他就不认识了。

**方式 B (JEPA 的思路)**: 你给小孩看一张猫的照片, 再给他看一张猫的简笔画, 然后说 "这两个是同一个东西"。你逼他在脑子里形成一个 abstract concept of "猫", 这个 concept 既能解释照片, 也能解释简笔画。他不需要重建照片的每个 pixel, 也不需要重建简笔画的每根线条。他只需要在 abstract level 上知道: 这两个 view 来自同一个 underlying thing。

这就是 JEPA。LeCun 在 vision 里搞了好几年了 (I-JEPA, V-JEPA), 这篇 paper 是第一次在 LLM 上做成。

---

## 为什么 NLP 一直没做成

Vision 里做 JEPA 很自然 — 你拿一张图, random crop 出一块当 context, 另一块当 target, 两个都是同一张图的 view。你有很多种 augmentation: crop, color jitter, blur, flip...

NLP 里你没法 random crop 一句话。你也没法 "blur" 一段文字。NLP 的 augmentation 一直是个难题。

这篇 paper 的 key insight: **有些 NLP dataset 天然就有 two views**。

比如:
- "lines not having the string 'dog' followed by a number, 3 or more times" (natural language)
- `((dog.*[0-9].*){3,})` (regex code)

这两个是 **同一个东西的两个 surface forms**。这就是 JEPA 需要的 positive pair! 不需要任何 augmentation, 数据自带的。

类似地:
- NL → SQL: 问题和 SQL query 是 two views of the same query intent
- Git issue → code diff: 问题描述和代码修改是 two views of the same fix
- Math problem → solution: 问题和解答是 two views of the same reasoning

---

## 具体怎么做的

训练的时候, loss 有两部分:

**Part 1**: 正常的 next-token prediction。把 text 和 code 拼一起, 让模型从 text 生成 code。这部分保持 LLM 的 generation 能力。

**Part 2**: JEPA loss。单独拿 text 的最后一个 token 的 hidden state (叫 `Enc(Text)`), 单独拿 code 的最后一个 token 的 hidden state (叫 `Enc(Code)`), 然后算它们的 cosine distance。逼这两个 embedding 靠近。

Part 2 的直觉: **如果模型真的理解了 text 和 code 是同一个东西, 那它们在 abstract representation space 里应该落在同一个地方**。如果 model 只是学会了 surface pattern generation, 但 internal representation 是散的, 这两个 embedding 就会很远, JEPA loss 就会很大。

实际 loss:

$$\mathcal{L} = \mathcal{L}_{\text{LLM}} + \lambda \cdot d(\text{Enc}(\text{Text}), \text{Enc}(\text{Code}))$$

λ 就是一个 weight, 控制 JEPA part 的强度。不同 task 最优值差很多, 从 0.5 到 1024 都有。

---

## 那个 [PRED] token 是怎么回事

有时候你光拿 text 的最后一个 hidden state 去匹配 code 的 embedding, 可能不够 — 因为 text 的最后一个 token 的 representation 可能还没 "想完"。

所以作者在 text 末尾加 k 个 special token `[PRED]`。这些 token 可以 attend 到前面所有 text token, 给模型 k 次额外的 "思考步骤" 来 refine 它的 prediction。

k=0: 直接拿 text embedding 去匹配
k=1: 给模型 1 次额外思考
k=3: 给模型 3 次额外思考

这跟你闭眼想事情一样 — 有时候你想一步就够了, 有时候你需要在脑子里转几圈才能想明白。`[PRED]` token 就是给模型的 "在脑子里转几圈" 的机会。

而且这些 `[PRED]` token **复用 LLM 自己的 weights**, 不引入任何新参数。就是让 transformer 在原 text 的 hidden state 上再跑 k 层 self-attention。

---

## 工程上最 clever 的 trick

你需要同时拿到 text 和 code 各自独立的 embedding。但 text 和 code 在 next-token prediction 里又要拼在一起。

如果直接拼接, code 的 hidden state 就会 attend 到 text, 那你拿到的 `Enc(Code)` 就不是 "纯 code 的 representation", 而是受了 text 污染的。

作者的解法: 构造一个特殊的 attention mask — text 内部 causal, code 内部 causal, 但 text 和 code 互相看不见。这样一次 forward pass 就能同时拿到两个独立的 embedding。

然后第二次 forward pass 用标准 causal mask (code 可以看 text) 来算 next-token prediction loss。

所以总共 2 次 forward pass, training cost 是 baseline 的 2 倍。但 **inference 完全不受影响** — JEPA loss 只在 training 时用。

---

## 最让我兴奋的发现

Table 14。作者想看看 LLM-JEPA 学到的 `Enc(Text) → Enc(Code)` 这个 mapping 到底长什么样。

他们试了一个假设: 这个 mapping 是不是近似线性的? 就是说, 能不能找到一个矩阵 X, 让 `Enc(Text) · X ≈ Enc(Code)`?

结果:
- Base model: regression error = 3953
- 普通 finetune: regression error = 3035 (稍微好一点, 但还是很大)
- **LLM-JEPA: regression error = 4**

**4!!!** 从 3000 多掉到 4。降了三个数量级。

这说明什么? LLM-JEPA 训完之后, text 到 code 的映射变成了一个 **几乎是线性的变换**。你只需要一个矩阵乘法就能从 text 的 embedding 算出 code 的 embedding。

这个发现太漂亮了。它暗示在 LLM 的 latent space 里, "自然语言" 和 "代码" 可能不是两个完全不同的 representation space, 而是同一个 space 的两个 linearly related 的 "坐标系"。就像你可以用 Celsius 也可以用 Fahrenheit 描述温度, 两者之间就是一个线性变换。

更深一层的联想: 也许人类语言和编程语言在某种 "mentalese" / "language of thought" 层面本来就是同一个东西, 只是 surface form 不同。JEPA objective 可能无意中触发了模型发现这个 unified representation。

---

## 为什么不怕 overfitting

普通 finetune 一个 LLM, 训到 epoch 3-4 就开始 overfit — training loss 继续降, 但 val accuracy 开始掉。这是大家司空见惯的现象。

LLM-JEPA 训到 epoch 6 还在涨。而且 gap 还在拉大。

直觉解释: JEPA loss 是一个 **representation regularizer**。它不停地在说 "你的 internal representation 要有 structure, text 和 code 要对齐"。这个约束阻止模型 overfit 到 training set 的 surface patterns — 模型不能靠 memorize 训练数据来降低 loss, 它必须学到真正的 abstract correspondence。

这跟 dropout 的 regularization 机制不同。Dropout 是在 input / activation 层面加 noise, 逼模型 robust。JEPA 是在 representation 层面加 structure constraint, 逼模型 meaningful。两者互补, 也许可以叠加。

---

## LoRA 的意外发现

Table 8 里有个反直觉的结果:

LoRA rank=512 (只训练 22.59% 的参数) + LLM-JEPA = **72.41%**

Full finetune (训练 100% 参数) + LLM-JEPA = **70.42%**

**LoRA 超过了 full finetune!** 这在标准训练里几乎不可能 — full finetune 的 capacity 总是 ≥ LoRA。

我的 hypothesis: LoRA 限制了 optimization 的搜索空间, 模型没法 overfit。但同时 JEPA loss 提供了 strong guidance, 告诉模型 "representation 应该长什么样"。两者结合: **受限的 search space + 清晰的 optimization target = 更好的 convergence**。

这跟 [Aghajanyan 的 intrinsic dimension 发现](https://arxiv.org/abs/2012.13255) 一脉相承 — finetune 的有效维度远低于参数量。LoRA + JEPA 可能恰好找到了这个 intrinsic subspace 的 sweet spot。

---

## 2 倍 training cost 怎么解决

Section 5.2 的 loss dropout。

简单说: training 的时候, 每个 batch 有概率 α 直接 skip JEPA loss。skip 了就不需要那次额外的 forward pass, 省一半 compute。

α=0.75 意味着 75% 的 batch 不算 JEPA loss, 只有 25% 算。per-epoch cost 从 2x 降到 1.25x。

但 surprising 的结果是: 在 **相同 compute budget** 下, α=0.75 的 accuracy 比 α=0 (full JEPA) 还高。

为什么? 我觉得跟 stochastic depth 的道理一样 — 随机 skip 创造了 diversity, 每个 batch 看到的 task signal 略不同, 起到了 ensemble / regularization 效果。

实用 heuristic: 保持 `λ × (1 - α)` 近似常数。你想要的总 "JEPA pressure" 是固定的, dropout 只是改变了 delivery 方式 — 少给几次但每次更用力, vs 多给几次但每次轻柔。

---

## 能不能推广到所有 NLP task

作者试了 NQ-Open (问答) 和 HellaSwag (常识续写), 这些 task 没有天然的 (text, code) two-view 结构。

结果: gain 较小 (NQ +1.5%, HellaSwag +1.1%), 但 statistically significant。

NQ-Open 的 setup: Text = question, Code = answer span。这里 "answer" 不是 question 的 alternative representation, 它只是 question 的一个 projection。view relationship 非常 asymmetric。

HellaSwag: Text = context, Code = correct continuation。关系更松散 — context 和 continuation 之间是 narrative coherence, 不是 semantic equivalence。

**Intuition**: JEPA 的 gain 与 two-view 的 "cleanliness" 正相关。text/code 是最 clean 的 (几乎 1-to-1 对应), gain 最大 (14%)。Q&A 是最 messy 的, gain 最小 (1.5%)。

要真正推广到所有 NLP task, 需要一个 **NLP 里的 data augmentation 机制**, 类似 vision 的 random crop。可能的候选:
- LLM-generated paraphrase (让 GPT 重写一遍, 作为 second view)
- Back-translation (英→法→英, 得到 paraphrase)
- Text summarization (原文 vs 摘要)
- Code comment ↔ code (天然 two-view)

---

## 我的整体判断

这篇 paper 的 significance 不在于 +14% accuracy。而在于:

**它证明了 LeCun 的 JEPA 哲学在 NLP 里也能 work, 而且不损害 generation 能力。**

之前所有人 (包括我自己) 都觉得 NLP 跟 vision 不一样 — NLP 必须 generative, 必须 reconstruct tokens, 不能丢掉 decoder。这篇 paper 说: 你可以鱼和熊掌兼得。保留 generative loss, 同时加一个 embedding-space 的 alignment loss, 两者不冲突, 甚至互相促进。

Figure 3 那个实验是铁证: 加了 JEPA loss 之后, next-token prediction loss 没变, 但 accuracy 涨了 20%。**generation 能力没被牺牲, representation quality 被免费提升了**。

如果未来有人把这个 scale 到 70B+ pretraining, 配合 multi-step JEPA (latent reasoning chain), 可能会打开一条全新的 LLM 训练范式 — 介于 LeCun 的 JEPA-only 和现在的 NTP-only 之间的中间路线。

---

Reference:
- Paper code: https://github.com/rbalestr-lab/llm-jepa
- LeCun JEPA position paper: https://openreview.net/forum?id=BZ5a1r-kVsf
- I-JEPA: https://arxiv.org/abs/2301.08243
- Balestriero & LeCun (reconstruction uninformative): https://arxiv.org/abs/2402.11337
- Intrinsic dimension of finetuning: https://arxiv.org/abs/2012.13255

还有什么想深挖的, 尽管问。

---

# LLM-JEPA 深度解读

Karpathy 你好! 这篇 paper 我仔细读完了, 作者阵容里有 Yann LeCun 本人, 所以这篇工作的分量很重 — 它是 LeCun 一直倡导的 JEPA 路线从 vision 正式进军 NLP 的第一个 concrete attempt。我来帮你 build intuition。

---

## 1. 核心动机: 为什么 LLM 需要 JEPA?

LeCun 多年来一直批评 autoregressive next-token prediction ([LeCun 2022 position paper](https://openreview.net/forum?id=BZ5a1r-kVsf))。他的论点核心在于 **input-space reconstruction objectives 会学到大量 "noise"**, 因为它们被迫去预测每一个 pixel / token, 包括那些对 perception 和 reasoning 无关的细节。Balestriero & LeCun 2024 的实验 ([arXiv:2402.11337](https://arxiv.org/abs/2402.11337)) 在 vision 上证明了这一点: autoencoder 学到的 features 对 downstream perception task 几乎没信息量。

JEPA 的哲学是: **只在 embedding space 做预测, 不做 reconstruction**。你学习一个 encoder `Enc(x)` 和一个 predictor `Pred`, 让 predictor 从 view A 的 embedding 预测 view B 的 embedding。这样模型只需要在 abstract representation level 上 "理解" 两个 view 的关系, 不需要浪费 capacity 去 memorize input-space 的 noise。

在 vision 领域, I-JEPA ([Assran et al. 2023](https://arxiv.org/abs/2301.08243)) 和 V-JEPA ([Bardes et al. 2024](https://arxiv.org/abs/2404.08471)) 都验证了这条路。但在 NLP 里, 一直没人做成。原因有二:
- LLM 的 evaluation 几乎全是 generative 的 (你要它生成 text), 所以你不能像 SimCSE ([Gao et al. 2021](https://arxiv.org/abs/2104.08821)) 那样丢掉 decoder, 只学 embedding
- NLP 缺少 vision 里那种天然 "augmentation" — 你不能像 image 那样做 random crop / color jitter

这篇 paper 的 breakthrough 在于: **识别出某些 NLP dataset 天然就有 "two views" 结构**。比如 (natural language description, regex code) 这一对, 就是同一个 underlying semantic object 的两个 surface forms。这正好就是 JEPA 需要的 positive pair!

---

## 2. LLM-JEPA Objective 公式精解

核心 loss 在 equation (2):

$$
\mathcal{L}_{\mathrm{LLM-JEPA}} = \underbrace{\sum_{\ell=2}^{L} \mathcal{L}_{\mathrm{LLM}}(\mathrm{Text}_{1:\ell-1}, \mathrm{Text}_{\ell})}_{\text{generative (LLM)}} + \lambda \times \underbrace{d(\mathrm{Pred}(\mathrm{Enc}(\mathrm{Text})), \mathrm{Enc}(\mathrm{Code}))}_{\text{abstraction (JEPA)}}
$$

我来逐个拆解每一个符号:

- `Text_{1:L}`: 输入的自然语言描述序列, 长度为 L。比如 "lines not having the string 'dog' followed by a number, 3 or more times"
- `Text_{\ell}`: 第 ℓ 个 token, ℓ 从 2 到 L (因为第一个 token 没有 context 可预测)
- `\mathcal{L}_{\mathrm{LLM}}(Text_{1:\ell-1}, Text_{\ell})`: 就是标准的 next-token cross-entropy loss。在 equation (1) 里它定义为 `XEnt(Classifier(Enc(Text_{1:\ell-1})), Text_{\ell})`, 其中 `Enc` 是 transformer 的 hidden state (最后一个 layer), `Classifier` 是 LM head (linear projection 到 vocab size)
- `\lambda \geq 0`: 平衡两个 loss 的超参数。注意 paper 里发现这个值在不同 task 上最优值差异巨大, 从 0.5 到 1024 都有 (见 Table 4 的 HellaSwag 配置 `\lambda=1, k=3` vs NQ-Open 的 `\lambda=1024, k=0`)
- `Enc(Text)`: Text 经过 LLM encoder 后, 取 **最后一个 token 在最后一个 layer 的 hidden_state** 作为整个序列的 embedding。这是经典的 "last-token probing" 约定
- `Enc(Code)`: 同理, Code 的 embedding。Code 是 ground truth 的 code 部分
- `Pred(·)`: predictor network。这里有个非常聪明的设计 — **tied weights**, 直接复用 LLM 自己的 self-attention layers
- `d(·,·)`: 距离 metric。paper 里用 **cosine similarity** (负的, 因为要 minimize)

关键 insight: **`Pred` 是通过在 Text 末尾 append k 个 `[PRED]` token 实现的**。当 k=0 时, `Pred(x) = x`, 也就是直接拿 Text 的 embedding 去匹配 Code 的 embedding。当 k>0 时, 那些 `[PRED]` token 可以 attend 到前面所有 Text token (但因为是新 token, 它们不改变原 Text 部分的 hidden state), 给 model 一个 "额外的非线性变换步骤" 来 refine prediction。

这个设计让我联想到 DeepMind 早期的 [Relational Networks](https://arxiv.org/abs/1706.01427) 和 [Memory Networks](https://arxiv.org/abs/1410.3916), 它们也是通过额外的 "reasoning steps" (function passes) 来增强抽象能力。这里则是把 reasoning step 直接嵌进了 LLM 自己的 transformer layer 里, 共享 weights, 零额外参数。

---

## 3. Custom Attention Mask: 工程上的关键 trick

这是 paper 里最容易被忽略但其实最 clever 的部分。看 equation (2), 你需要同时拿到 `Enc(Text)` 和 `Enc(Code)` 两个 embedding。朴素做法需要 **3 次 forward pass**:
1. 单独 encode Text → 拿到 `Enc(Text)`
2. 单独 encode Code → 拿到 `Enc(Code)`
3. 把 Text+Code concatenate 做 next-token prediction → 算 `\mathcal{L}_{\mathrm{LLM}}`

但 Code 和 Text 在 next-token prediction 里是要拼一起的 (模型需要看 Text 来生成 Code)。如果在同一次 forward 里同时拿两个 embedding, Code 的 representation 会 attend 到 Text, 那 `Enc(Code)` 就被 "污染" 了, 不再是独立的 view。

作者的解法是构造一个 **block-causal mask**: 把 sequence 分成两个 block (Text block 和 Code block), 每个 block 内部是 causal (下三角), 但 block 之间完全 mask 掉 (互不可见)。代码在 Section 3.2:

```python
mask = torch.full((batch_size * 2, 1, seq_length, seq_length), -torch.inf)
# Text block: 内部 causal
mask[i, :, t_start:t_start+t_size, t_start:t_start+t_size] = additive_mask(t_size)
# Code block: 内部 causal, 但和 Text block 互不可见
mask[i, :, c_start:c_start+c_size, c_start:c_start+c_size] = additive_mask(c_size)
```

这样 **2 次 forward pass** 就够了:
- Pass 1: 用 block-causal mask 同时拿 `Enc(Text)` 和 `Enc(Code)`, 算 JEPA loss
- Pass 2: 用标准 causal mask (Code 可以 attend 到 Text) 算 `\mathcal{L}_{\mathrm{LLM}}`

training cost 是 baseline 的 2x, 但 **inference 完全不变** (因为 JEPA loss 只在 training 时用)。

---

## 4. 最关键的实验: NTP 不会隐式最小化 JEPA

Figure 3 是全篇最 informative 的图之一, 我来细讲。

实验设置: Llama-3.2-1B-Instruct 在 NL-RX-SYNTH 上 finetune, 两种 condition:
- **Condition A (baseline)**: 只用 `\mathcal{L}_{\mathrm{LLM}}` 算 gradient, 但同时 monitor JEPA prediction loss `D(·,·)`
- **Condition B (LLM-JEPA)**: 用完整 `\mathcal{L}_{\mathrm{LLM-JEPA}}` 算 gradient

结果 (Table 在 Section 3.3):
- Baseline accuracy: **51.95%**
- LLM-JEPA accuracy: **71.10%**
- 两个 condition 的 `\mathcal{L}_{\mathrm{LLM}}` loss curve 几乎完全 overlap (Figure 3 右图的 blue 和 yellow 线)
- 但 baseline 的 `D(·,·)` (JEPA pred loss) 保持高位 flat, LLM-JEPA 的 `D(·,·)` 持续下降

**Intuition**: 这说明 next-token prediction 的 objective 完全没有 "挤压" representation space 让 Text 和 Code 的 embedding 对齐。模型可以很好地预测下一个 token, 同时 `Enc(Text)` 和 `Enc(Code)` 在 latent space 里相距甚远。换句话说, **generative objective 和 embedding-space alignment 是两个正交的目标**。

这跟 Balestriero & LeCun 2024 在 vision autoencoder 上观察到的现象一致: 你可以 reconstruction 完美, 但学到的 features 对 perception 毫无用处。

---

## 5. 结构化表示: SVD 分析 + 近似线性映射

Section 4.2 和 Table 14 是 paper 里最有 theoretical depth 的部分。

作者假设 LLM-JEPA 把 `Enc(Text) → Enc(Code)` 的 mapping 约束在一个 **narrow subspace** 里。验证方法:

### 5.1 SVD 分析 (Figure 3 左)
计算 `Enc(Text) - Enc(Code)` 矩阵 (在所有样本上 stack 起来) 的 top-100 singular values。
- Base model: singular values 很大
- `\mathcal{L}_{\mathrm{LLM}}` finetune: singular values **更大** (说明 NTP finetune 反而打乱了 representation structure!)
- LLM-JEPA: singular values **小几个数量级**

**Intuition**: 如果 `Enc(Text) ≈ Enc(Code)`, 那它们的差就接近 zero matrix, singular values 会很小。LLM-JEPA 强制 Text 和 Code 落在 embedding space 的 "同一个地方", 而 NTP 允许它们 drift apart (因为 NTP 只关心 token-level prediction, 不关心 embedding-level alignment)。

### 5.2 近似线性假设 (Table 14)
作者进一步假设 mapping 是 **近似线性** 的, 即 `Enc(Code) ≈ Enc(Text) · X` for some matrix X。用 least-squares 求解 `min_X ||Enc(Text)·X - Enc(Code)||_2`:

| Setting | Regression error | Avg top-100 singular value |
|---------|------------------|---------------------------|
| Base model | 3953.11 | 310.73 |
| `\mathcal{L}_{\mathrm{LLM}}` finetune | 3035.01 | 341.80 |
| LLM-JEPA k=1 | **4.47** | 94.84 |
| LLM-JEPA k=0 | **4.04** | 16.82 |

**Intuition**: regression error 从 ~3000 掉到 ~4, 降了 3 个数量级。这说明 LLM-JEPA 学到的 `Text→Code` mapping 几乎是线性的! 这非常 surprising — 你可能会以为需要复杂的非线性变换才能把自然语言映射到 code, 但 JEPA objective 把这个 mapping "线性化" 了。

这让我联想到 [Olshausen & Field 1996 的 sparse coding](https://www.nature.com/articles/381607a0) 和 [Bell & Sejnowski 1995 的 ICA](https://ieeexplore.ieee.org/document/679622) — 它们发现 cortex 的 representation 也倾向于线性 / 近线性分解。也许 JEPA objective 触发了某种类似的 "natural representation" prior。

更深一层的 intuition (这里我开始 hallucinate 一点): 如果 Text→Code 是线性的, 那意味着 **LLM 的 latent space 里, "language" 和 "code" 是同一个 manifold 的两个 linearly-related coordinate systems**。这跟 Chomsky 的 universal grammar 假设有微妙共鸣 — 也许 "language of thought" 真的是一个 unified representational space, surface form (text vs code) 只是这个 space 的不同 projection。

---

## 6. 实验结果全景

### 6.1 主实验: 4 个 model family × 4 个 dataset

Table 12 和 Table 13:

| Model | Baseline | LLM-JEPA | Δ |
|-------|----------|----------|---|
| Llama-3.2-1B | 57.29% | 71.46% | +14.2% |
| Gemma-2-2b | 33.65% | 43.12% | +9.5% |
| OpenELM-1.1B | 12.07% | 25.40% | +13.3% |
| OLMo-2-1B | 87.09% | 87.52% | +0.4% |

注意 OpenELM 的 baseline 只有 12%, LLM-JEPA 翻倍到 25%。这说明 JEPA objective 对 **weak base model** 的提升幅度更大 — 也许是因为 weak model 的 representation 更需要 regularization。

| Dataset | Baseline | LLM-JEPA | Δ |
|---------|----------|----------|---|
| NL-RX-SYNTH | 57.29% | 71.46% | +14.2% |
| NL-RX-TURK | 22.49% | 30.94% | +8.5% |
| GSM8K | 32.36% | 36.36% | +4.0% |
| Spider | 47.52% | 50.55% | +3.0% |

GSM8K 和 Spider 的 gain 较小, 这合理 — 这两个 dataset 的 "two views" 结构没有那么 clean (math problem 和它的 solution 之间是 reasoning chain, 不是 simple translation; SQL query 和 question 之间也有 schema linking 的 gap)。

### 6.2 Overfitting resistance (Figure 1 right)

这个发现非常重要: 标准 finetune 在 epoch 3-4 之后开始 overfit (val accuracy 下降), 而 LLM-JEPA **持续上升**到 epoch 6。

**Intuition**: JEPA loss 起到了 **representation regularizer** 的作用。它强制 model 的 latent space 保持 "smooth" 的 structure (Text embedding 和 Code embedding 要对齐), 这个约束阻止 model 过拟合到 training set 的 surface patterns。本质上, JEPA loss 是在说: "你不仅要预测对下一个 token, 你的 internal representation 还要对齐到 ground truth code 的 representation"。这是一个比 NTP **更强的 inductive bias**。

### 6.3 Pretraining 实验 (Table 2, Table 9)

Table 2: 在 NL-RX-SYNTH 上从 random init pretrain Llama-3.2-1B:
- NTP only: 54.38%
- LLM-JEPA: 60.59% (+6.2%, p=2.94e-4)

Table 9 更有意思: 在 `cestwc/paraphrase` (5 个 paraphrase 一组) 上 pretrain, 然后在 **完全 unrelated** 的 Rotten Tomatoes / Yelp 上 finetune (finetune 时 **不用** JEPA loss):
- Rotten Tomatoes: 56.57% → 57.76% (p=7.38e-4)
- Yelp: 26.46% → 27.15% (p=1.00e-3)

**Intuition**: 这说明 JEPA pretraining 学到的 representation **transfer** 到了下游 task。paraphrase 数据里的 "multiple views of same meaning" 这个 structure, 让 model 学到了更好的 semantic representation, 即使下游 task (sentiment classification) 和 pretrain task 完全不同。这是 JEPA 在 vision 里的核心 selling point (I-JEPA pretrain 的 model 在 ImageNet linear probe 上很强) 在 NLP 里的第一次验证。

### 6.4 LoRA 实验 (Table 8)

| LoRA rank | Baseline | LLM-JEPA |
|-----------|----------|----------|
| 32 | 6.09% | 7.45% |
| 64 | 21.09% | 32.46% |
| 128 | 34.21% | 48.45% |
| 256 | 45.57% | 60.80% |
| 512 | 50.18% | **72.41%** |
| Full | 57.29% | 70.42% |

**惊人发现**: LoRA rank=512 (只训 22.59% 参数) + LLM-JEPA = **72.41%**, 而 **full finetune + LLM-JEPA = 70.42%**。LoRA 居然超过了 full finetune!

**Intuition**: 这跟 [Aghajanyan et al. 2020 的 intrinsic dimension](https://arxiv.org/abs/2012.13255) 发现一致 — finetune 的有效参数维度远小于全参数。LoRA + JEPA 的组合可能找到了一个 **better-conditioned optimization landscape**: LoRA 限制了 capacity (防 overfit), JEPA 提供了 representation-level guidance (防 representation collapse)。两者协同, 效果 > full finetune。

---

## 7. Ablation: 哪些设计选择重要?

Table 3:

| 变体 | Accuracy |
|------|----------|
| Baseline (NTP only) | 57.29% |
| **LLM-JEPA (cosine, append [PRED], Text→Code)** | **71.46%** |
| ℓ2-norm 替代 cosine | 2.22% (崩了!) |
| MSE | 70.64% |
| Prepend [PRED] (而不是 append) | 68.07% |
| Code→Text (反向预测) | 65.70% |
| InfoNCE | 34.40% (比 baseline 还差!) |

几个关键 takeaway:

**ℓ2-norm 崩溃 (2.22%)**: 这非常 surprising。作者没有深入分析, 但我的 hypothesis 是: ℓ2-norm 对 embedding 的 magnitude 敏感, 而 transformer 的 hidden state magnitude 会随训练 drift, 导致 loss landscape 极不稳定。Cosine similarity 归一化了 magnitude, 所以 robust。这跟 [BatchNorm / LayerNorm 为什么 work](https://arxiv.org/abs/1502.03167) 的 intuition 类似。

**InfoNCE 崩溃 (34.40%)**: 这跟 vision JEPA 的发现一致 — InfoNCE 需要 negative samples, 而 JEPA 的 "positive-only" 设计避免了 contrastive learning 的陷阱 ([collapse, false negatives](https://arxiv.org/abs/2110.09348))。LeCun 一直 [批评 contrastive learning](https://openreview.net/forum?id=BZ5a1r-kVsf), 这个实验再次支持他的观点。

**Code→Text 弱于 Text→Code (65.70% vs 71.46%)**: Code 是更 "structured / compressed" 的 representation, Text 是更 "verbose / noisy" 的。从 noisy 预测 structured 容易 (compression), 从 structured 预测 noisy 难 (需要 generate 大量细节)。这跟 [Information Bottleneck theory](https://arxiv.org/abs/physics/0004057) 的方向性一致。

---

## 8. Loss Dropout: 加速 2x 且 accuracy 更高

Section 5.2 是一个很 practical 的贡献。思路: training 时以概率 `α` 随机 skip JEPA loss (整个 batch 级别)。如果 skip, 就不需要那额外一次 forward pass。

per-epoch cost 变为 baseline 的 `(2 - α)` 倍:
- α=0: 2x cost (standard LLM-JEPA)
- α=0.5: 1.5x cost
- α=0.75: 1.25x cost

Table 6 (在相同 PFLOPs 下比较):

| Config | Accuracy |
|--------|----------|
| LD=0, λ=1 (full LLM-JEPA) | 63.96% |
| LD=0.75, λ=4 | **73.08%** |

**Surprise**: 在 **相同 compute budget** 下, LD=0.75 + λ=4 **超过** 了 full LLM-JEPA! 

**Intuition**: 这跟 [Dropout](https://arxiv.org/abs/1207.0580) / [Stochastic Depth](https://arxiv.org/abs/1603.09382) 的机制类似 — 随机 skip 创造了 ensemble effect, 每个 batch 看到的 "task" 略不同, 起到 regularization 作用。同时, 因为只对一部分 batch 施加 JEPA 约束, model 不会 over-rely on JEPA signal, 保持了对 NTP 的 fidelity。

经验法则: **保持 `λ × (1 - α)` 近似常数**。这个 heuristic 很实用 — 你想要的总 "JEPA strength" 是固定的, dropout 只是改变了 delivery 方式。

---

## 9. 超出 (text, code) 对: Q&A 和 Reasoning Models

Section 5.1 把方法推广到没有天然 two-view 结构的 dataset:

### NQ-Open
- Text = question, Code = answer span (通常只有几个 token)
- 这里的 "view" 关系非常 **asymmetric** — answer 只是 question 的一个 small projection, 不是 full alternative representation
- 结果: 20.12% → 21.59% (p=2.44e-3)

### HellaSwag
- Text = context, Code = correct continuation
- 关系更 **diverse** — context 和 continuation 之间是 narrative coherence, 不是 semantic equivalence
- 结果: 69.40% → 70.51% (p=0.0136)

### Reasoning Models (Table 5)
- Qwen3-1.7B on GSM8K: 44.32% → 45.00%
- DeepSeek-R1-Distill-Qwen-1.5B: 13.87% → 15.04%

**Intuition**: 即使 reasoning model 已经通过 RLHF/RLVR 训练过, JEPA 仍能带来 gain。这说明 JEPA 提供的 representation alignment signal 是 **orthogonal** 到 RL-based reasoning training 的。RL 优化的是 final answer 的 reward, JEPA 优化的是 intermediate representation 的 structure。

HellaSwag 实验里有个有趣发现 (Figure 9): λ 可以 scale 到 **1024** 还在提升! 这暗示对于 "weak view relationship" 的 task, 需要更强的 JEPA pressure 才能压出有用的 structure。

---

## 10. 更广的联想和 future directions

### 10.1 联系到 LeCun 的 V-JEPA / I-JEPA 路线
这篇 paper 本质上是把 [I-JEPA](https://arxiv.org/abs/2301.08243) 的 mask-and-predict paradigm 移植到 NLP。在 I-JEPA 里, "two views" 是 image 的不同 region (context patch 和 target patch); 这里是 (text, code) 对。关键差异: vision 的 view 是 **spatial**, NLP 的 view 是 **semantic** (不同 surface form)。这提示我们: **JEPA 的本质是 "predict one view from another in abstract space", view 的具体形式可以多样**。

### 10.2 联系到 "System 2 thinking" / latent reasoning
最近 [Yi Tay 等人的 speculative decoding work](https://arxiv.org/abs/2211.17192) 和 [Quiet-STAR](https://arxiv.org/abs/2403.09629) 都在探索 "latent space reasoning" — 让 model 在生成前先做 latent computation。LLM-JEPA 的 `k` 个 `[PRED]` token 本质上就是 **latent reasoning steps**! 每个 `[PRED]` token 给 model 一次额外的 self-attention pass 来 refine prediction。这跟 [Jaegle et al. Perceiver](https://arxiv.org/abs/2103.03206) 的 "latent array" 概念也相关。

Future direction: 把 `k` 从固定值变成 **adaptive** — 让 model 自己决定需要多少 reasoning steps。类似 [Adaptive Computation Time (Graves 2016)](https://arxiv.org/abs/1603.08983)。

### 10.3 联系到 world models 和 model-based RL
LeCun 的 [JEPA position paper](https://openreview.net/forum?id=BZ5a1r-kVsf) 里, JEPA 是 world model 的核心 — agent 通过预测 "next state in abstract space" 来 plan。这篇 paper 只做了 single-step prediction (Text → Code), 但如果 extend 到 **multi-step JEPA** (predict embedding of Code_2 from Code_1 from Code_0...), 就可以建模 reasoning chain 的 latent dynamics。这会非常接近 [DreamerV3](https://arxiv.org/abs/2301.04104) 在 NLP 里的 analog。

### 10.4 联系到 tokenization 和 input-space noise
一个 deep question: **为什么 vision 里 reconstruction 学到 noise, 但 NLP 里 NTP 似乎没那么糟?** 我觉得答案在于 tokenization — BPE tokenization 本身就是一种 lossy compression, 已经 "滤掉" 了 character-level 的大量 noise。所以 NTP 在 token space 做 reconstruction, 某种程度上已经在 "abstract space" 了。这解释了为什么 NLP 的 JEPA gain (相对) 比 vision 小, 但仍然存在。

如果未来有人做 **byte-level LLM** (比如 [MegaByte](https://arxiv.org/abs/2305.07195)), 那时 input-space noise 问题会严重得多, JEPA 的 gain 应该会大幅放大。这是一个值得探索的 direction。

### 10.5 联系到 mechanistic interpretability
Table 14 发现 Text→Code mapping 几乎线性 (regression error 4 vs 3000)。这给 mech interp 一个 strong hint: **LLM 内部可能存在 "linear concept directions"** (类似 [Anthropic 的 toy models](https://transformer-circuits.pub/2022/toy_model/index.html))。如果能在更大 scale 上验证这个线性性, 会是很重要的 interpretability 发现。

### 10.6 Limitations 和我的批评
- **2x training cost**: 即使有 loss dropout, 仍是显著 overhead。大规模 pretraining (比如 Llama-3 15T tokens) 上 2x 不可接受。
- **Hyperparameter sensitivity**: (λ, k) grid 没有规律 (Figure 7), 每个新 task 都要重新 sweep。这对 production 不友好。
- **Two-view assumption**: 很多 NLP task 没有天然 two-view 结构。作者在 Section 5.1 尝试推广, 但 gain 较小 (NQ +1.5%, HellaSwag +1.1%)。要真正 scale, 需要一个 "NLP data augmentation" 的 general mechanism — 类似 vision 的 random crop。也许 LLM-generated paraphrase / back-translation 是一个方向?
- **只测了小 model**: 最大 8B (Table 15), 而且在 8B 上 baseline 就崩了 (35.77%, 因为 regex termination 问题)。需要验证 70B+ scale。

---

## 11. 总结: 这篇 paper 的真正贡献

这篇 paper 不是 "another finetuning trick"。它的意义在于:

1. **打通了 JEPA 从 vision 到 NLP 的最后一公里**: 识别出 (text, code) 对是 NLP 里天然的 two-view 结构, 让 JEPA objective 第一次在 LLM 上 work
2. **证明了 NTP 和 generative capability 可以解耦**: 你可以在不损害生成能力的前提下, 大幅改善 representation quality (Figure 3 的 blue/yellow overlap)
3. **发现了 representation structure 的可线性化现象** (Table 14): 这是一个 deep finding, 暗示 LLM latent space 有比我们想象更简单的 geometry
4. **提供了 practical recipe**: loss dropout + custom mask 让 method 在 ~1.5x cost 下 deployable

如果让我赌一个 future: 18 个月内, 会有 paper 把这个方法 scale 到 70B+ pretraining, 配合 multi-step JEPA (latent reasoning), 可能会接近或超过 standard NTP pretraining 的 efficiency frontier。LeCun 路线在 NLP 上的 "existence proof" 终于有了。

---

**Reference links:**
- Paper code: https://github.com/rbalestr-lab/llm-jepa
- LeCun JEPA position paper: https://openreview.net/forum?id=BZ5a1r-kVsf
- I-JEPA (vision): https://arxiv.org/abs/2301.08243
- V-JEPA: https://arxiv.org/abs/2404.08471
- Balestriero & LeCun 2024 (reconstruction uninformative): https://arxiv.org/abs/2402.11337
- SimCSE: https://arxiv.org/abs/2104.08821
- InfoNCE / CPC: https://arxiv.org/abs/1807.03748
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Qwen3: https://arxiv.org/abs/2505.09388
- SWE-bench (issue→code diff): https://openreview.net/forum?id=VTF8yNQM66
- SimCLR (contrastive baseline): https://arxiv.org/abs/2002.05709
- Llama 3: https://arxiv.org/abs/2407.21783
- Gemma 2: https://arxiv.org/abs/2408.00118
- OLMo 2: https://arxiv.org/abs/2501.00656
- OpenELM: https://arxiv.org/abs/2404.14619
- Dimensional collapse in SSL: https://arxiv.org/abs/2110.09348
- Intrinsic dimension of finetuning: https://arxiv.org/abs/2012.13255
- Adaptive computation time: https://arxiv.org/abs/1603.08983
- MegaByte: https://arxiv.org/abs/2305.07195
- Anthropic toy models: https://transformer-circuits.pub/2022/toy_model/index.html
- BatchNorm: https://arxiv.org/abs/1502.03167
- Dropout: https://arxiv.org/abs/1207.0580
- Stochastic Depth: https://arxiv.org/abs/1603.09382
- Perceiver: https://arxiv.org/abs/2103.03206
- DreamerV3: https://arxiv.org/abs/2301.04104
- Quiet-STAR: https://arxiv.org/abs/2403.09629
- Large Concept Models (related embedding-space LM): https://arxiv.org/abs/2412.08821

如果之后你想 deep dive 某个具体 aspect (比如 custom mask 的 CUDA kernel 实现, 或者 multi-step JEPA 的 formulation), 告诉我, 我可以继续展开。
