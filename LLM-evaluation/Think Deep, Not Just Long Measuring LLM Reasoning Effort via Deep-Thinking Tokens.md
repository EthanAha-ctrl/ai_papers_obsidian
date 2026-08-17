---
source_pdf: Think Deep, Not Just Long Measuring LLM Reasoning Effort via Deep-Thinking
  Tokens.pdf
paper_sha256: bdea6d6e91b1d32bd41d6e621d18da69990d3f53a8753b8297fa8edde9d1c1b3
processed_at: '2026-08-12T15:21:56-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇paper到底在说啥

## 一句话总结

模型"想多久"跟"想对没"几乎没关系，甚至越想越错。要看模型"想得多深"——也就是每个词在模型内部被改了多少遍才吐出来。

---

## 先说一个让人头疼的事实

我们一直有个直觉：给reasoning model更多token去"思考"，它应该表现更好。所以大家在搞test-time scaling的时候都在卷"怎么让模型想得更长"——budget forcing、longer CoT、reasoning level调高。

但这篇paper第一张图就给了当头一棒：**在GPT-OSS-120B-medium上，output长度和准确率的相关性是 r = -0.544**。负的！而且是moderate negative。意思是同一个题，模型写3000 token的答案往往比写8000 token的答案更可能对。

这在最近一堆paper里都被report了，叫overthinking、inverse scaling。Wu et al. 2025画了个"inverted-U"曲线，Gema et al. 2025专门写了inverse scaling paper。现象是：模型一旦想太长，就开始fixate on irrelevant details、amplify自己的错误heuristic、绕进死循环。

所以问题来了——**如果"长度"不靠谱，那什么才靠谱？怎么衡量模型到底有没有在认真思考？**

相关link：
- Overthinking survey: https://arxiv.org/abs/2503.16419
- Inverse scaling in test-time compute: https://arxiv.org/abs/2507.14417
- When more is less: https://arxiv.org/abs/2505.00127 (approximate)

---

## 作者的key insight：看模型内部，别看模型表面

作者借用了一个很老但很经典的技术叫**logit lens**（Nostalgebraist 2020, Belrose et al. 2023）。

这技术说起来很简单：模型有36层（GPT-OSS-120B），每一层都会输出一个hidden state向量。你把这个向量直接用最后的unembedding matrix投影到vocabulary上，就能得到一个"这一层时模型心里想的下一个词是什么"的概率分布。

所以你可以在每一层都"偷看"模型当下的预测。一个有趣的发现：功能词（"and", "is"）在大概第10层就定下来了，而算术结果、最终answer token（比如"13"）要到第30层以后才稳定。

Figure 2那个heatmap超级直观，你应该去看一眼。你能看到：
- "and", "is", "boxed"这些词——浅层就settle了，模型闭着眼就能写
- "+_= "这种运算后的completion——中后层才settle
- answer token "13"——特别神奇，它第一次在深层浮现，然后逐渐"向上渗透"到更早的层，像模型内部正在建立conviction

**核心假设**：一个token如果要在很深的层才稳定下来，说明模型在前面那些层做了大量的内部修订工作——这就是"深度思考"的signature。浅层就定下来的token，说明模型没怎么费劲就吐出来了。

Logit lens原版blog（强烈推荐读）: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

Tuned Lens paper: https://arxiv.org/abs/2303.08112

---

## DTR具体怎么算

我用大白话讲一遍算法，不抄公式：

1. **生成一个token的过程里，在每一层都算一次JSD距离**：看这一层的预测分布跟最后一层的预测分布差多远。JSD是衡量两个概率分布差异的指标，对称且bounded，取值[0, ln2]。JSD=0表示完全一样，JSD越大越不同。

2. **取running minimum**：因为分布的演化不是单调下降的，可能中间层接近final了又diverge。取running min保证我们捕捉的是"模型首次达到稳定"的层，而不是最后一次波动。

3. **找settling depth**：从layer 1开始扫，找到JSD第一次降到threshold $g=0.5$以下的层。这个层就叫这个token的"settling depth $c_t$"。

4. **判断是不是deep-thinking token**：如果 $c_t$ 落在最后15%的层里（$\rho=0.85$，对36层模型来说就是layer 31+），那这个token就是"deep-thinking token"。

5. **算序列的DTR**：一个sequence里deep-thinking token的比例。DTR=0.2意思是20%的token在深层才定下来，80%在浅层就定下来了。

为什么选JSD而非KL？作者在Appendix A做了ablation。KLD-based DTR在AIME 25上相关性是**-0.698**（负的！），因为KLD asymmetric且数值unstable——early layers是flat high-entropy分布，KL会artificially small。JSD的symmetry和boundedness规避了这个问题。

---

## 结果：DTR跟accuracy强相关，长度跟accuracy反相关

Table 1那张表是整篇paper的核心证据。我挑几个数字给你感受一下：

| 指标 | 平均r |
|------|------|
| Token Length（越长越对？） | **-0.594** |
| Reverse Token Length（越短越对？） | 0.594 |
| Log Probability（confidence） | 0.527 |
| Self-Certainty（best baseline） | 0.605 |
| **DTR** | **0.683** |

DTR是平均相关性最高的，而且最稳定——32个model-benchmark组合里只有2个负相关。Token count有大量负相关，confidence-based方法在小模型（OSS-20B）上经常崩。

几个impressive的极端值：
- DeepSeek-R1-70B on AIME 2025: DTR r=0.974
- OSS-120B-low on GPQA-D: DTR r=0.976
- OSS-120B-high on HMMT 25: DTR r=0.972

为什么confidence-based不靠谱？我的interpretation：confidence混淆了"对"和"过度自信"。小模型经常在错误答案上high confidence，这就破坏了correlation。DTR衡量的是"模型花了多少computation"，这是个更纯粹的effort信号，跟model scale相对解耦。

---

## 应用：Think@K——省一半算力还更准

既然DTR能预测哪个sample靠谱，那就直接用它来筛选。

**做法**：
1. 对每个题sample 48个responses
2. 每个response只看前50个token，算这50个token的DTR
3. 选DTR最高的24个（top 50%）做majority voting
4. 剩下24个直接early stop，不生成完

**结果**（Table 2）：
- Cons@n（标准self-consistency，48个全vote）: AIME 25 acc=92.7%, cost=307.6k tokens
- Think@n（DTR筛选top 24）: AIME 25 acc=**94.7%**, cost=**155.4k tokens**（省49%）

几乎所有benchmark都这样：要么持平要么更准，cost省一半。Self-Certainty@n也省一半cost，但accuracy不如Think@n。

---

## 最反直觉的发现：50个token就够了

Table 3做了prefix length ablation。50 token的prefix效果最好（94.7%），100 token反而降到92.0%，2000 token降到92.0%。

这个发现很奇怪但很深刻。我的interpretation：**前50个token的DTR反映的是模型"开始时的engagement level"**。一个模型如果开头就在浅层settle（"Let me think about this problem..."这种reflexive输出），说明它没进入真正的reasoning mode。如果一开头就有deep revision，说明它真的在engage with the problem。

类比一下：就像面试一个人，前5分钟的对话能感觉出来他是不是真懂行。后面说再多都是粉饰。前50个token就是"前5分钟"。

---

## Appendix B的fascinating发现

Figure 7显示：GPT-OSS-120B的low reasoning level DTR最高，high reasoning level DTR最低。但accuracy是反过来的。

意思是：**high reasoning level本质上是把computation从depth dimension转移到了sequence dimension**。

- Low reasoning: 每个token"使劲想"（高DTR），但sequence短
- High reasoning: 每个token"轻松想"（低DTR），但sequence长

这其实有点细思极恐。是不是说，现在的reasoning model训练其实在teach模型"用更多tokens来替代更深的per-token computation"？像是一种computation laundering——表面上看模型"想得更久"了，实际上每一步思考反而变浅了？

如果是这样，那DTR不能直接跨reasoning level比较，但**同一level内的ranking信号依然有效**。这也解释了为什么Think@K能work——它在同一batch内做相对ranking，不受绝对值影响。

---

## 那个qualitative example特别生动

Table 7和Table 8是同一个AIME 25题的两个答案：

- **错的那个**：27,724个token，DTR=13.9%。模型喋喋不休，反复"Let me reconsider", "Something wrong", "Let's set coordinate system"，但每个token内部其实没做多少revision。
- **对的那个**：3,725个token，DTR=19.0%。简洁直接，每一步都精确，但每个token背后是genuine的depth-wise reasoning。

这就是paper thesis的完美诠释：**啰嗦 ≠ 思考**。Long CoT很多时候只是在做surface-level的linguistic padding，真正的computation effort体现在per-token的layer-wise revision强度上。

---

## 我的一些联想

1. **训练implication**：如果DTR是真正衡量thinking effort的指标，那RL training应该reward high-DTR responses而非long responses。现在的GRPO/PPO都在implicitly reward length（因为long responses更容易有correct final answer by chance），这可能就是overthinking的root cause。Shrivastava et al. 2025的GFPO用length penalty，但DTR-based reward可能更principled。Paper: https://arxiv.org/abs/2508.09726

2. **跟DoLa的connection**：Chuang et al. 2023的DoLa用layer contrasting做decoding提升factuality。DTR是measure，DoLa是intervention，但两者共享同一个foundation——layer-wise prediction evolution encodes thinking。一个自然extension是用DTR指导dynamic layer selection。Paper: https://arxiv.org/abs/2310.01424

3. **Early exiting的connection**：如果很多tokens在layer 5就settle了，那layer 6-36对它们就是redundant computation。这跟early exiting literature天然契合——可以用DTR做adaptive depth inference。Belrose et al. 2023的tuned lens就是干这个的。Paper: https://arxiv.org/abs/2303.08112

4. **跟"reasoning is computation"理论的connection**：最近有人argue CoT本质是Turing machine的tape，每个token是一步computation。如果这个view对，那DTR衡量的是"每步computation的intensity"。未来的reasoning theory可能需要同时考虑length和depth两个维度。参考Cobbe et al. 2021的verifier work和Snell et al. 2024的scaling test-time compute paper。

5. **跟Vilas et al. 2025的互补**：Vilas的工作从temporal dimension（across reasoning steps）分析latent trajectory signals来预测correctness，本文从depth dimension（across layers）分析。两个维度combine可能是future work。Paper: https://arxiv.org/abs/2505.02109 (approximate)

6. **跟mechanistic interpretability的connection**：DTR其实是在测量"模型内部iterative refinement的强度"。这跟induction head、function vector、circuit-level interpretability是一脉相承的思路——都是把模型的"reasoning"还原成internal mechanism的activity pattern。Anthropic最近的可解释性工作很值得follow。https://transformer-circuits.pub/

---

## 总结成一句话

**衡量模型思考，不要看它说了多少，要看它说每个词的时候，内部被改了多少遍。**

这就是这篇paper全部的thesis。剩下都是engineering和empirical evidence。

---

# Think Deep, Not Just Long: 深入解析

## 1. 核心问题的intuition

这篇paper戳中了一个当前test-time scaling范式的根本矛盾。我们一直假设"think longer = think better"，但Figure 1的数据直接打脸：在GPT-OSS-120B-medium上，output token count与accuracy的Pearson correlation **r = -0.544**（moderate negative！）。也就是说，越长越错。

这背后的现象在文献里被叫做 **overthinking** 或 **inverse scaling** (Gema et al., 2025; Wu et al., 2025)。模型在生成长链时可能在amplify flawed heuristics或者fixate on irrelevant details。

那问题来了：**什么才是"真正在思考"的可靠信号？** 这篇paper的答案是：**不要看模型说了多少，要看模型在每一层内部做了多少"修订工作"**。

参考：
- Logit Lens原始blog: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Tuned Lens paper: https://arxiv.org/abs/2303.08112
- Inverse scaling in test-time compute: https://arxiv.org/abs/2507.14417

---

## 2. 方法核心：Deep-Thinking Ratio (DTR)

### 2.1 Mechanistic intuition

作者借用了Nostalgebraist (2020)和Belrose et al. (2023)的观察：**直接把中间层的hidden state通过unembedding matrix投影到vocabulary space，就能得到一个meaningful的predictive distribution**。这意味着我们可以在每一层"偷看"模型当前的预测是什么。

核心假设是：
- 如果一个token的预测在浅层就已经稳定（即浅层分布 ≈ 最终层分布），说明模型对这个token"没什么思考"，是reflexive的输出
- 如果一个token的预测在深层才稳定，说明前面的layers做了大量的distributional revision——这是"深度思考"的mechanistic signature

Figure 2展示了一个非常直观的heatmap：
- 功能词（"and", "is", "boxed", "<|return|>"）在浅层（layer ~10-15）就收敛
- 算术操作符后的completion（"+_", "=_",）和answer tokens（"13", "(D)"）直到layer ~30+才稳定
- 特别有意思的是answer token "13"——它第一次在深层浮现后，逐渐"向上渗透"到更早的层，像是模型内部正在形成conviction的过程

### 2.2 数学定义详解

**Equation 1: 中间层的logit lens投影**
$$p_{t,l} = \text{softmax}(z_{t,l}), \quad z_{t,l} = W_U h_{t,l}$$

变量解释：
- $t$：generation step（当前生成第t个token）
- $l$：layer index（1到L，L是总层数，GPT-OSS-120B是36层）
- $h_{t,l} \in \mathbb{R}^d$：layer $l$之后的residual stream state，$d$是hidden dimension
- $W_U \in \mathbb{R}^{|V| \times d}$：unembedding matrix（language modeling head）
- $z_{t,l}$：layer $l$投影后的logit vector
- $p_{t,l}$：layer $l$的预测分布

注意这里没有用tuned lens的learned affine transformation——作者坚持用raw的unembedding matrix投影，保持mechanistic purity。

**Equation 2: JSD距离度量**
$$D_{t,l} := \text{JSD}(p_{t,L} \| p_{t,l}) = H\left(\frac{p_{t,L} + p_{t,l}}{2}\right) - \frac{1}{2}H(p_{t,L}) - \frac{1}{2}H(p_{t,l})$$

变量解释：
- $D_{t,l}$：generation step $t$处，layer $l$分布与final layer $L$分布的JSD
- $H(\cdot)$：Shannon entropy, $H(p) = -\sum_v p(v)\log p(v)$
- $p_{t,L}$：final layer的分布（即模型真正用来sample的分布）

为什么选JSD而非KL？Appendix A给了empirical证据：
- JSD-based DTR: AIME 25 r=0.869, HMMT 25 r=0.895
- KLD-based DTR: AIME 25 r=**-0.698**（负的！），HMMT 25 r=0.409
- Cosine-based DTR: AIME 25 r=0.633, HMMT 25 r=0.172

KLD的问题在于asymmetric且数值unstable——early layers是high-entropy flat distribution，对很多token都有非零probability mass，这些mass在深层被drive到near-zero，导致$\text{KL}(p_L \| p_l)$可能artificially small。JSD的symmetry和boundedness（取值[0, ln2]）规避了这个问题。

**Equation 3: 强制单调性**
$$\bar{D}_{t,l} = \min_{j \leq l} D_{t,j}$$

这个running minimum很关键。为什么不直接用$D_{t,l}$？因为distributional revision不一定是单调下降的——可能layer 15接近final了，但layer 20又diverge了（比如引入新的information）。取running min保证我们捕捉的是**"模型首次达到稳定"的层**，而非最后一次波动。

**Equation 4: Settling depth**
$$c_t = \min\{l \in \{1, \ldots, L\} : \bar{D}_{t,l} \leq g\}$$

- $c_t$：token $t$的"settling depth"——首次JSD降到threshold $g$以下的层
- $g$：settling threshold，paper中用$g=0.5$（注意JSD max是ln2≈0.693，所以0.5是个相对strict的threshold）

**Equation 5: Deep-thinking regime**
$$\mathcal{L}_{\text{deep-thinking}} = \{l : l \geq \lceil \rho \times L \rceil\}$$

- $\rho$：depth fraction，paper中$\rho=0.85$
- 对于L=36层，$\lceil 0.85 \times 36 \rceil = 31$，即只有layer 31, 32, 33, 34, 35, 36这个regime内的settling才算"deep"

**Equation 6: 序列级DTR**
$$\text{DTR}(S) = \frac{1}{T} \sum_{t=1}^T \mathbb{1}[c_t \in \mathcal{L}_{\text{deep-thinking}}]$$

- $S$：生成的sequence，长度$T$
- $\mathbb{1}[\cdot]$：indicator function
- DTR本质是**序列中deep-thinking tokens的比例**

### 2.3 Algorithm 1的complexity

```
for each generation step t:
    sample y_t from p_{t,L}
    for l = 1 to L:
        compute p_{t,l} = softmax(W_U h_{t,l})
        compute D_{t,l} = JSD(p_{t,L}, p_{t,l})
    compute c_t
    if c_t in deep regime: increment counter
```

inference overhead：每个token需要额外L次$W_U h_{t,l}$投影（$O(|V| \cdot d)$）和L次JSD计算（$O(|V|)$）。如果$|V|=128k$, $d=7168$（GPT-OSS-120B），这是不小的开销。但相对forward pass本身（$O(d^2 \cdot L)$ for attention），这个overhead可接受，尤其考虑到它只用于评估/筛选。

---

## 3. 实验结果的关键发现

### 3.1 Table 1的correlation全景

| Method | Average r | 解读 |
|--------|-----------|------|
| Token Length | -0.594 | 长度反向相关，overthinking确实存在 |
| Reverse Token Length | 0.594 | 只是post hoc statistical adjustment |
| Log Probability | 0.527 | Confidence信号部分有用，但不稳定 |
| Negative Perplexity | 0.219 | 异质性强 |
| Negative Entropy | 0.571 | 中等 |
| Self-Certainty | 0.605 | 最佳baseline，但仍有失败case |
| **DTR (Ours)** | **0.683** | 最高且最稳定 |

DTR在32个model-benchmark组合中只有2个出现orange（负相关），而token count有大量orange。

特别impressive的数字：
- DeepSeek-R1-70B on AIME 2025: DTR r=**0.974**
- OSS-120B-low on GPQA-D: DTR r=**0.976**
- OSS-120B-high on HMMT 25: DTR r=**0.972**

### 3.2 Confidence-based methods为什么不靠谱？

看Table 1能发现一个pattern：confidence-based方法（log prob, entropy, self-certainty）在OSS-120B上普遍strong positive，但在OSS-20B上经常崩到weak甚至negative。

我的interpretation：confidence信号混淆了**correctness**和**overconfidence**。小模型更容易overconfident on wrong answers（Hallucination with high probability），这就破坏了correlation。DTR反映的是**computationaleffort**而非confidence，所以对model scale更鲁棒。

### 3.3 Hyperparameter sensitivity (Figure 4)

**Settling threshold $g$**（左图，固定$\rho=0.85$）：
- $g=0.25$：too permissive，包括了很多low-effort tokens，trend变flat，correlation降低
- $g=0.5$：optimal balance
- $g=0.75$：slightly unstable due to过滤掉informative tokens

**Depth fraction $\rho$**（右图，固定$g=0.5$）：
- $\rho \in \{0.8, 0.85, 0.9, 0.95\}$：shift the range of DTR values but maintain positive slope
- $\rho$的robustness暗示"deep regime"的具体边界不重要，重要的是**存在一个late-settling的现象**

**Conclusion**: $(g, \rho) = (0.5, 0.85)$是Pareto optimal。

---

## 4. Think@K: 应用DTR做test-time scaling

### 4.1 协议

- 对每个question sample $n=48$个responses
- 用prefix $\ell_{\text{prefix}}=50$ tokens计算DTR(S[:50])
- 选DTR最高的top $\eta=50\%$（即24个）做majority voting
- Early stop剩下的24个samples

### 4.2 成本计算细节

Table 2的cost计算很微妙：

- **Cons@n, Mean@n, Long@n**：cost = $\sum_{i=1}^n |S_i|$（必须full decode所有n个）
- **Short@n**：cost = $\sum_{\text{selected}} |S_i| + \ell_{\text{longest\_short}} \times \eta \times n$（partial decoding overhead）
- **Think@n, Self-Certainty@n**：cost = $\sum_{\text{selected}} |S_i| + \ell_{\text{prefix}} \times \eta \times n$

其中$\ell_{\text{prefix}} = 50$是个非常激进的early stopping——只生成50个tokens就能判断哪个sample值得继续！

### 4.3 Table 3的prefix length ablation

AIME 25上的结果：
- $\ell_{\text{prefix}}=50$: acc=94.7%, cost=155.4k tokens
- $\ell_{\text{prefix}}=100$: acc=92.0%, cost=154.1k tokens
- $\ell_{\text{prefix}}=2000$: acc=92.0%, cost=198.8k tokens
- $\ell_{\text{prefix}}=\text{all}$: acc=94.0%, cost=307.6k tokens

**50个tokens比2000个tokens效果更好**——这是个非常counterintuitive但深刻的发现。我的interpretation：长prefix会被sequence-level的noise稀释，而短prefix捕捉的是模型"开始思考时的engagement level"，这个early signal反而更pure。

### 4.4 Table 2的Pareto frontier

OSS-120B-medium:
- Cons@n: AIME 25 acc=92.7%, cost=307.6k
- Think@n: AIME 25 acc=**94.7%**, cost=**155.4k** (减少49%)
- GPQA-D: Cons@n 73.8% vs Think@n **74.7%**, cost减少48%

Qwen3-4B-Thinking:
- AIME 24: Cons@n 93.3% vs Think@n **93.3%**, cost减少49%
- HMMT 25: Cons@n 63.3% vs Think@n **66.7%**, cost减少50%

Think@n在**所有benchmark上都达到或超过Cons@n，同时成本减半**。Self-Certainty@n也减半成本但accuracy不如Think@n。

---

## 5. Appendix B的fascinating发现

Figure 7显示：在GPT-OSS-120B上，**low reasoning level的DTR > medium > high**，但accuracy是high > medium > low。

这暗示reasoning level的本质是**将computation从depth dimension转移到sequence dimension**：
- Low reasoning: 每个token都"使劲想"（高DTR），但sequence短
- High reasoning: 每个token"轻松想"（低DTR），但sequence长

这让我联想到一个open question：**reasoning model的训练是否在implicitly flatten depth-wise computation？** 如果是这样，那DTR可能不能直接跨model比较，但**同一model内的DTR ranking信号依然有效**。

这也呼应了Csordás et al. (2025)的观察：later layers主要做fine-grained distributional refinement而非introduce fundamentally new transformations。Paper: https://arxiv.org/abs/2505.13898

---

## 6. Qualitative Example的intuition (Table 7 vs Table 8)

对比OSS-120B-medium对同一AIME 25题的两个回答：
- **Incorrect**: 27,724 tokens, DTR=13.9%——模型喋喋不休，反复"let me reconsider"，但每个token的internal computation其实很shallow
- **Correct**: 3,725 tokens, DTR=19.0%——简洁直接，但每个token背后是genuine的depth-wise reasoning

这完美诠释了paper的thesis：**verbosity ≠ thinking**。Long CoT可能只是在做surface-level的linguistic padding，而真正的computation effort体现在per-token的layer-wise revision强度上。

---

## 7. 局限性和open questions

1. **DTR的interpretability across models**: Appendix B显示DTR绝对值不能跨reasoning level比较，那跨model family比较就更可疑了。Think@n能work是因为它在**同一model的同一batch内做ranking**，relative signal有效。

2. **Why 50-token prefix works**: 这个发现亟需理论解释。是否因为早期的"setup tokens"（"Let's think about...", "We need to find..."）的DTR反映了模型的**problem engagement**？如果一个model一上来就在浅层settle，说明它没"进入状态"。

3. **与DoLa的关系**: Chuang et al. (2023)的DoLa用layer contrasting做decoding来提升factuality。DTR是measure，DoLa是intervention，但两者共享同一个foundation——**layer-wise prediction evolution encodes thinking**。一个自然的extension是用DTR指导dynamic layer selection。

4. **Training implications**: 如果DTR是真正衡量thinking effort的指标，那RL training（如GRPO/PPO）应该reward high-DTR responses而非long responses。这可能解决overthinking的root cause。Shrivastava et al. (2025)的GFPO已经用length penalty，但DTR-based reward可能更principled。Paper: https://arxiv.org/abs/2508.09726

5. **Layer depth efficiency**: 这个work间接challenge了"depth is fully utilized"的assumption。如果很多tokens在layer 5就settle了，那layer 6-36对它们来说是redundant computation。这与early exiting literature（Belrose et al., 2023; Schuster et al., 2022）天然契合——可以用DTR做adaptive depth inference。

---

## 8. 个人takeaway

这篇paper最漂亮的地方是它**把"thinking"从surface feature（length）拉回到mechanistic feature（depth-wise computation）**。Figure 1的r=-0.544 → r=0.828这个jump非常compelling。

但我觉得最deep的insight其实在Appendix B：**reasoning level本质上是在重新分配computation budget between depth and length**。这暗示当前的reasoning model可能只是学会了"用更多tokens来替代更深的per-token computation"——这是一种**computation laundering**。

如果这个hypothesis成立，那未来的reasoning model训练应该：
1. Reward DTR而非length
2. 在RL中penalize sequence-level padding
3. 用DTR作为early-exit signal做inference acceleration

参考Vilas et al. (2025)的latent-trajectory signals工作，他们从temporal dimension（across reasoning steps）分析，而本文从depth dimension（across layers）分析——两个维度combine可能是future work。Paper: https://arxiv.org/abs/2505.02109 (approximate, based on author name search)

总之，这篇paper开启了一个新视角：**measure reasoning by how the model computes, not by what the model says**。这个principle的影响应该会超越DTR本身。
