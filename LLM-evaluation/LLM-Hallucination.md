---
source_pdf: LLM-Hallucination.pdf
paper_sha256: e08e529f936a4ae918eee2704b854eb3f1bb81e250c8a88f95b289ee8299aee2
processed_at: '2026-08-05T15:32:50-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LLM Hallucination

Andrej，我换个方式，像在咖啡馆聊天那样讲。

---

## 一句话说核心

**Hallucination 就是 LLM 在"编"，但编得跟真的似的。**

问题是：为什么编？能不编吗？

---

## 第一层：为什么 Hallucination 理论上不可能完全消除

NUS 那篇 paper 核心就一个 trick，跟 Cantor 对角线、Gödel 不完备定理一个套路。

想象一个无限大的表格：
- 每一行是一个 LLM（$h_0, h_1, h_2, ...$）
- 每一列是一个可能的输入（$s_0, s_1, s_2, ...$）
- 格子里是 LLM 对该输入的输出

现在我构造一个新的 ground truth function $f$，它专门"跟你对着干"：
- 在 $s_0$ 这一列，$f$ 故意跟 $h_0$ 不一样
- 在 $s_1$ 这一列，$f$ 故意跟 $h_1$ 不一样
- ...

所以**不管你拿出哪个 LLM，总有至少一个输入，它的输出跟 $f$ 不一样**。

这就像他妈的玩石头剪刀布，对方能看到你的出拳再决定出什么——你永远赢不了。

**但这算不算真"hallucination"？**

说实话，这个 proof 有点耍流氓。三个原因：

1. LLM 设计来干 language modeling 的，不是来 fit 任意 function 的。你构造一个 $f$ 专门跟它对着干，然后说"你看你错了"——so what？
2. 真实世界 hallucination 是 graded 的，不是 binary 的。"有点不准"跟"完全瞎编"差别很大。
3. $f$ 可能根本不是 meaningful 的 function，只是数学上存在。

所以这个 paper 理论上 sound，practically 就是告诉你：**别指望 0% hallucination，但往低了压是完全可行的。**

类比：no free lunch theorem 说没有万能算法，但没人因此就不做 machine learning 了。

---

## 第二层：Hallucination 到底分几种

Survey paper 给了个很清晰的二分：

### Factuality Hallucination——"说错话"

模型说 "Yuri Gagarin 是第一个登上月球的人"——错，那是 Neil Armstrong。Gagarin 是第一个进太空的。

模型说 "独角兽在 10000 BC 的 Atlantis 大陆漫步"——纯编。

### Faithfulness Hallucination——"不听话"

你让它翻译，它给你回答问题。
你给它一段 Nile 河的资料，它说的跟资料矛盾。
你让它解方程 $2x+3=11$，第一步 $2x=8$ 对的，第二步 $x=3$——自己打自己脸。

---

## 第三层：为什么会 Hallucinate

三个层面，从底往上：

### Data 层：垃圾进，垃圾出

**Knowledge Shortcut** 是最阴险的。

训练数据里 "Canada" 和 "Toronto" 一起出现的频率远高于 "Canada" 和 "Ottawa"。因为 Toronto 是大城市、经济中心、球队名字里都有。模型就学到了一个 spurious correlation：Toronto = Canada 的 capital。

这就像你天天看到"张三和李四一起吃饭"，你就假设他们是好哥们，但其实他们只是在同一家公司不同部门，偶尔在食堂碰到。

**Reversal Curse** 也很有意思。

训练数据全是 "A is B"（比如 "Tom Cruise's mother is Mary Lee Pfeiffer"），模型学会了 $A \rightarrow B$，但反过来问 "Mary Lee Pfeiffer's son is who?"，答不上来。

为什么？Autoregressive training 是 left-to-right 的，反向的 activation pattern 从没被 reinforce 过。

Mount Everest 那个例子更扎心：模型知道 Everest 最高，但问"如果 Everest 矮 500 米哪个山最高"——需要 multi-hop reasoning（Everest 高度 - 500 → K2 高度 → 比较），模型就懵了。

### Training 层：架构和训练方式的锅

**Exposure Bias** 是个经典的 train/inference mismatch。

训练时：模型看到的是正确的前缀，预测下一个 token。
推理时：模型看到的是自己生成的前缀，可能已经错了。

一个错误 token → 条件分布偏移 → 更多错误 → snowball effect。

这就像你学开车时 always 在 perfect 路面上，但上路了遇到一个小坑，车晃了一下，你一紧张猛打方向盘，然后就越偏越远。

**Alignment 的副作用**：

SFT 和 RLHF 本质上是在 push model 偏离它的 internal beliefs，去 align with "preferred output"。问题是 preferred ≠ correct。

RLHF 偏好 confident、helpful、polite 的回答。模型就学会了"自信地说错话"。

你想想，人类也是这样——那些最自信的人往往不是最对的，但社会 reward 他们。RLHF 把这个 bias 灌进 LLM 里了。

### Inference 层：采样本身就有问题

**Likelihood Trap**：greedy decoding 总是选最高概率的 token，结果容易得到 degenerate output，比如 "the the the the..."。

所以我们需要 temperature、top-k、top-p 来引入 randomness。但 randomness 又带来 uncertainty 和 hallucination。

**Tradeoff**：diversity vs fidelity。你要 creativity 就得接受一些 noise。

**Softmax Bottleneck**：

输出层是 $P(y_t) = \text{softmax}(W h_t)$，其中 $W \in \mathbb{R}^{|V| \times d}$。

vocab size $|V|$ 可能 50000，hidden dim $d$ 可能 4096。所以 $Wh_t$ 的 rank 最多 4096，但理想 distribution 可能需要 rank 50000。

模型表达不了所有可能的 distribution，被 rank 卡住了。

而且这跟 softmax 无关——你用 logits 也一样，bottleneck 在 $W$ 的 rank 上。

---

## 第四层：Self-Attention 为什么有 Locality Bias

这个你笔记里特别标注了 "dive deeper"，我详细讲。

现象：self-attention 给距离近的 token 更高 weight。

四个原因：

### 原因 1：自然语言本身就有 locality

相邻 token 更可能 syntactically/semantically related。"The cat sat on the mat"——"cat" 和 "sat" 的关系比 "cat" 和 "mat" 更紧密（虽然 "cat" 和 "mat" 也有关系，但通过 "sat on" 桥接）。

### 原因 2：Sinusoidal Positional Encoding 的数学结构

原始 Transformer 的 PE：
$$\text{PE}(pos, 2i) = \sin(pos / 10000^{2i/d})$$

两个位置的 attention score 通过 dot-product 计算，展开后是：

$$\text{PE}(pos_1) \cdot \text{PE}(pos_2) = \sum_i \cos\left(\frac{pos_1 - pos_2}{\omega_i}\right)$$

其中 $\omega_i = 10000^{2i/d}$。

**关键**：这个 score 只依赖 $|pos_1 - pos_2|$，而且高频项（$i$ 小时 $\omega_i$ 小）会在距离稍大时快速振荡平均掉，低频项衰减慢但贡献弱。所以总体上**距离越远，attention score 越小**。

这不是 bug，是 sinusoidal encoding 的 inherent property。

### 原因 3：Dot-Product Similarity 的双重 boost

相近的 token：
- Positional encoding 相似（如上所述）
- Content 往往也相似（因为 locality）

两个 similarity 相乘，attention score 被 double boost。

### 原因 4：Optimization Bias

Local patterns 更容易学。gradient signal：

$$\nabla_\theta \mathcal{L}_{\text{local}} \gg \nabla_\theta \mathcal{L}_{\text{long-range}}$$

因为 local co-occurrence 统计更稳定、sample 更多。模型会优先学 local patterns，因为这样能更快 reduce loss。

---

## 第五层：怎么 detect Hallucination

五类方法，从简单到复杂：

1. **Fact overlap**：直接比 facts，看 generated 和 source 重叠多少。简单粗暴。
2. **NLI classifier**：训练一个 entailment classifier，判断 generated 是否被 source entail。
3. **QA-based**：从 generated content 生成 questions，然后从 source 找 answers，看对不对得上。这招很聪明——用 QA 来验证 consistency。
4. **Uncertainty estimation**：看 model 自己的 confidence。低 probability token = 高 uncertainty = 可能 hallucinate。或者多次采样看 consistency。
5. **LLM as judge**：让另一个 LLM（或同一个）来评估。G-Eval、SelfCheckGPT 都是这类。

---

## 第六层：怎么 mitigate

从简单到复杂：

- **Better data**：清洗、去偏、去重
- **RAG**：external knowledge grounding，让模型有 reference 可查
- **CoT**：显式推理链，强迫模型一步一步想
- **ToT**：tree search over reasoning paths，更彻底的 search
- **Ensemble**：多个模型投票或辩论
- **Factuality-aware training**：把 factuality 直接加到 loss/reward 里
- **Constrained decoding**：限制 output space

---

## 我的 Intuition 总结

**Hallucination 不是单一问题，是多个 failure mode 的集合。**

- 有的是数据问题（garbage in garbage out）
- 有的是架构问题（autoregressive、softmax bottleneck）
- 有的是训练问题（exposure bias、alignment side effect）
- 有的是推理问题（sampling randomness）

每种需要不同的解法。没有 silver bullet。

**那个对角化 proof 告诉我们的是**：hallucination 不能 100% 消除，就像 no free lunch 告诉我们没有万能算法。但这不阻止我们把 hallucination rate 从 10% 压到 1% 再到 0.1%。

**RLHF 的根本张力**：preferred ≠ correct。我们 reward "helpful confident polite"，但 confident 可以是 "confidently wrong"。可能的出路是 factuality-aware reward shaping。

**Locality bias 是 Transformer 的 inherent property**，不是 bug。但如果要 capture long-range dependencies，需要 architectural innovation（比如 long context、sparse attention、memory mechanisms）。

---

希望这样讲 clear 多了，Andrej。如果想再 dive deeper 某个点，比如 RLHF reward shaping 怎么 incorporate factuality，或者 softmax bottleneck 的 MoS 解法细节，随时说。

---

# LLM Hallucination Paper 深度解析

Andrej，这篇笔记整合了两篇核心文献，我来逐一深入剖析。

---

## 一、理论核心：Hallucination is Inevitable

**论文**: *Hallucination is Inevitable: An Innate Limitation of Large Language Models*
**作者**: Ziwei Xu, Sanjay Jain, Mohan Kankanhalli (NUS)
**链接**: https://arxiv.org/abs/2401.11817

### 1.1 形式化定义

$$\text{Definition 6: } \exists s \in S \text{ such that } h(s) \neq f(s)$$

**变量解释**:
- $h$: 任一 LLM（视为 computable function）
- $f$: ground truth function
- $S$: 所有可能输入字符串的集合（countably infinite）
- $s$: 某个具体输入

这里把 LLM 抽象成 $h: S \rightarrow S$ 的 total computable function。Hallucination 就是 $\exists s, h(s) \neq f(s)$。

### 1.2 对角化证明（Diagonalization Argument）

这是整个 proof 的核心技巧，本质是 **Cantor 对角线论证** + **Halting Problem** 风格。

构造如下 table：

| LLMs | $s_0$ | $s_1$ | $s_2$ | $s_3$ | $s_4$ | ... |
|------|-------|-------|-------|-------|-------|-----|
| $h_0$ | $h_0(s_0)$ | $h_0(s_1)$ | $h_0(s_2)$ | $h_0(s_3)$ | $h_0(s_4)$ | ... |
| $h_1$ | $h_1(s_0)$ | $h_1(s_1)$ | $h_1(s_2)$ | $h_1(s_3)$ | $h_1(s_4)$ | ... |
| $h_2$ | $h_2(s_0)$ | $h_2(s_1)$ | $h_2(s_2)$ | $h_2(s_3)$ | $h_2(s_4)$ | ... |
| $h_3$ | $h_3(s_0)$ | $h_3(s_1)$ | $h_3(s_2)$ | $h_3(s_3)$ | $h_3(s_4)$ | ... |
| ... | ... | ... | ... | ... | ... | ... |
| **$f$** | $\Delta(h_0(s_0))$ | $\Delta(h_1(s_1))$ | $\Delta(h_2(s_2))$ | $\Delta(h_3(s_3))$ | $\Delta(h_4(s_4))$ | ... |

**关键操作**: 定义 $f(s_i) := \Delta(h_i(s_i))$，其中 $\Delta$ 是一个"差异化"算子，确保 $f(s_i) \neq h_i(s_i)$。

这样构造出的 $f$：
- 是 computable 的（因为每个 $h_i$ computable，$\Delta$ computable）
- 但 $f \neq h_i$ 对任意 $i$ 成立（因为 $f(s_i) = \Delta(h_i(s_i)) \neq h_i(s_i)$）

**因此**: 无论你给出哪个 LLM $h_i$，总存在输入 $s_i$ 使得 $h_i(s_i) \neq f(s_i)$，即 hallucination 不可避免。

### 1.3 三个 Scope 的证明递进

1. **Scope 1**: LLMs 被 prover $P$ 证明为 total computable functions 的集合
2. **Scope 2**: 任意 computable set of LLMs
3. **Scope 3**: 任意单个 computable LLM（most general case，通过 recursion theory 中的 diagonalization）

三个 scope 从特殊到一般，最终结论：**any LLM, whether running in Poly time or not, as long as Computable, is bound to hallucinate**。

### 1.4 Pinch of Salt：三点保留意见

笔记作者（即你整理的）提出三个反思：

1. **设计目标错位**: LLMs 是为 language modeling 设计的，不是为 computing every possible ground truth function 设计的。这就像用锤子去拧螺丝，理论上"不能"是必然的。
2. **定义过于严格**: $h(s) \neq f(s)$ 是 binary 判断，但真实世界 hallucination 是 context-dependent、graded 的。
3. **$f$ 的人工构造性**: $f$ 是 deliberately constructed 来 differ from LLM outputs，可能不 realistic、不 meaningful。这有点像 "我构造了一个你必然答错的问题，所以你必然会错"——逻辑自洽但意义存疑。

**我的直觉**: 这个 proof 类似于 "no free lunch theorem"——理论上正确，但 practical impact 有限。它告诉我们 hallucination 不能被 100% 消除，但没说不能被压到极低。

---

## 二、Survey 论文：Taxonomy & Causes

**论文**: *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions*
**链接**: https://arxiv.org/abs/2311.05232

### 2.1 Taxonomy 二分法

#### Factuality Hallucination（事实性）

| Sub-Type | User Input | Model Output | Explanation |
|----------|-----------|--------------|-------------|
| Factual Inconsistency | "Tell me about the first person to land on the Moon" | "Yuri Gagarin was the first person to land on the Moon" | Neil Armstrong 才对，Gagarin 是第一个进太空的人 |
| Factual Fabrication | "Tell me about the historical origins of unicorns" | "Unicorns roamed Atlantis around 10,000 BC..." | 完全编造，无任何证据 |

#### Faithfulness Hallucination（忠实性）

| Sub-Type | 例子 | 说明 |
|----------|------|------|
| Instruction Inconsistency | "Translate to Spanish: What is the capital of France?" → "Paris." | 不翻译反而回答问题 |
| Context Inconsistency | Source 说 Nile 起源于 Great Lakes，Output 说起源于 mountain ranges | 与提供的 context 矛盾 |
| Logical Inconsistency | $2x+3=11$，Step 1 得 $2x=8$，Step 2 得 $x=3$ | 自相矛盾 |

### 2.2 Causes 的三层模型

#### Layer 1: Data 层面

**Knowledge Shortcut** 是个很有意思的现象。考虑训练数据中的 co-occurrence：

$$P(\text{Toronto} \mid \text{Canada}) \gg P(\text{Ottawa} \mid \text{Canada})$$

因为 "Canada Toronto" 在训练 corpus 中出现频率远高于 "Canada Ottawa"（Toronto 是大城市、经济中心），模型学到了 spurious correlation，把 Toronto 当成 capital。

这本质上是 **shortcut learning**（Geirhos et al., 2020, https://arxiv.org/abs/2007.12222）在 LLM 上的体现。

**Reversal Curse**（Berglund et al., 2023, https://arxiv.org/abs/2309.12288）：

$$A \text{ is } B \not\Rightarrow B \text{ is } A$$

如果训练数据只有 "Tom Cruise's mother is Mary Lee Pfeiffer"，模型能答 "Who is Tom Cruise's mother?"，但答不出 "Mary Lee Pfeiffer's son is who?"。

这揭示了 autoregressive models 的 **directional knowledge binding** 问题。原因猜测：
- Causal left-to-right training 形成 directional representation
- Reverse direction 的 activation pattern 在训练中从未被 reinforced

**Mount Everest 例子**（Table 4 引用）:
- 模型知道 Everest 是最高峰 ✓
- 但问 "如果 Everest 降 500 米，哪个山最高？" → 答不出
- 这是 **multi-hop reasoning** 失败：需要 (Everest 高度) - 500 → (K2 高度) → 比较

#### Layer 2: Training 层面

**Exposure Bias** 的数学表述：

训练时优化：
$$\mathcal{L}_{\text{train}} = -\sum_t \log P(y_t \mid y_{<t}^*)$$

其中 $y_{<t}^*$ 是 ground truth prefix。

推理时实际计算：
$$\mathcal{L}_{\text{infer}} = -\sum_t \log P(y_t \mid \hat{y}_{<t})$$

其中 $\hat{y}_{<t}$ 是模型自己生成的前缀。

**Mismatch**: 训练时 always see correct prefix，推理时 see noisy prefix → **snowball effect**：一个错误 token → 后续条件分布偏移 → 更多错误。

这是 scheduled sampling（Bengio et al., 2015, https://arxiv.org/abs/1506.03099）试图解决的问题。

**Alignment 导致 hallucination**:

SFT 的 loss：
$$\mathcal{L}_{\text{SFT}} = -\sum_t \log P_{\theta}(y_t^{\text{preferred}} \mid x, y_{<t}^{\text{preferred}})$$

RLHF 的 reward：
$$R(x, y) = r_{\theta}(x, y) - \beta \cdot \text{KL}(P_{\theta}(\cdot \mid x) \| P_{\text{ref}}(\cdot \mid x))$$

问题：preferred output ≠ factually correct output。比如 RLHF 偏好 "helpful, polite, confident" 的回答，但 confident ≠ correct。

**Optimization 把 model 推离 internal beliefs**：
$$P_{\text{aligned}}(y \mid x) \neq P_{\text{base}}(y \mid x)$$

即使 base model "知道" 答案 $a$，aligned model 可能输出 $b$ 因为 $b$ 更 "preferred"。

参考: Lin et al., "Teaching models to express their uncertainty in words", https://arxiv.org/abs/2205.14334

#### Layer 3: Inference 层面

**Likelihood Trap**:

经验观察：高 likelihood sequence 质量低。形式化：

$$\arg\max_y P(y \mid x) \text{ 往往是 degenerate output}$$

比如重复 "the the the the..."。原因：greedy decoding 放大了一阶统计，忽略了 high-order coherence。

解决：temperature sampling、top-k、nucleus sampling。

**Softmax Bottleneck**（Yang et al., 2017, https://arxiv.org/abs/1711.03953）:

输出层：
$$P(y_t \mid h_t) = \text{softmax}(W h_t + b)$$

其中 $h_t \in \mathbb{R}^d$，$W \in \mathbb{R}^{|V| \times d}$，$|V|$ 是 vocab size。

**rank 约束**：$\text{rank}(W h_t) \leq d$，但理想的 next-token distribution matrix 可以有 rank up to $|V|$。

当 $d \ll |V|$（典型情况：$d=4096$, $|V|=50000$），模型无法 express 任意 distribution。

笔记中提到 "even if we use logits w/o softmax, problem persists"——这正确，因为 bottleneck 在 $W$ 的 rank，不在 softmax 本身。

**解决方向**:
- Mixture of Softmaxes（MoS）：$\sum_k \pi_k(h) \text{softmax}(W_k h)$
- 增加 output layer 的 effective rank

### 2.3 Self-Attention 的 Locality/Positional Bias

这是笔记中你特别标注要 "dive deeper" 的部分。

**现象**: Self-attention 倾向给 closer tokens 更高 weight。

**四个原因**:

#### 原因 1: Natural Language Patterns

自然语言有 locality——adjacent tokens 更可能 syntactically/semantically related。Zipf's law、Markovian structure 都支持这一点。

#### 原因 2: Sinusoidal Encoding 的影响

原始 Transformer（Vaswani et al., 2017）的 positional encoding：

$$\text{PE}(pos, 2i) = \sin(pos / 10000^{2i/d})$$
$$\text{PE}(pos, 2i+1) = \cos(pos / 10000^{2i/d})$$

**Attention score**（dot-product）：
$$\text{PE}(pos_1) \cdot \text{PE}(pos_2) = \sum_i \sin(pos_1 / \omega_i) \sin(pos_2 / \omega_i) + \cos(pos_1 / \omega_i) \cos(pos_2 / \omega_i)$$

利用和差化积：
$$= \sum_i \cos((pos_1 - pos_2) / \omega_i)$$

**关键**: attention score 只依赖 $|pos_1 - pos_2|$，且随距离增加而衰减（因为 high frequency terms 平均化）。这就是 locality bias 的数学根源。

参考你的 *Let's build GPT from scratch* notes。

#### 原因 3: Dot-Product Similarity

$$\text{Attention}(Q, K) = \text{softmax}(QK^T / \sqrt{d_k})$$

相近 token 的 content similarity + positional similarity → 双重 boost。

#### 原因 4: Optimization Bias

Local patterns 更容易学，gradient signal 更强：

$$\nabla_\theta \mathcal{L}_{\text{local}} \gg \nabla_\theta \mathcal{L}_{\text{long-range}}$$

因为 local co-occurrence 统计更稳定、sample 更多。

---

## 三、Detection Methods

Figure 5 展示了 5 类 faithfulness detection：

### 3.1 Fact-based Metrics
测量 generated content 与 source content 的事实重叠。

### 3.2 Classifier-based Metrics
训练 NLI classifier 判断 entailment/contradiction。

例如：Fine-tuned RoBERTa 判断 $P(\text{entail} \mid \text{source}, \text{generated})$。

### 3.3 QA-based Metrics
Pipeline:
1. 从 generated response 生成 questions $Q = \{q_1, ..., q_n\}$
2. 用 $Q$ 从 source 抽取 answers $A_{\text{source}}$
3. 用 $Q$ 从 generated response 抽取 answers $A_{\text{gen}}$
4. 比较 $A_{\text{source}}$ vs $A_{\text{gen}}$

代表：QAGS (Wang et al., 2020, https://arxiv.org/abs/2004.04228), QuestEval (Scialom et al., 2021, https://arxiv.org/abs/2103.09338)

### 3.4 Uncertainty Estimation

**Internal states**（需 white-box access）：
- Token-level log probability: $\log P(y_t \mid y_{<t})$
- 低 probability → 高 uncertainty
- Entropy: $H(y_t) = -\sum_v P(v) \log P(v)$
- Layer-wise hidden state analysis

**LLM behavior**（API access）：
- Semantic consistency: 多次 sampling 比较
- Self-consistency (Wang et al., 2022, https://arxiv.org/abs/2203.11171)

### 3.5 Prompting-based Metrics
用 LLM as judge。代表：G-Eval (Liu et al., 2023, https://arxiv.org/abs/2303.16634), SelfCheckGPT (Manakul et al., 2023, https://arxiv.org/abs/2303.08896)

---

## 四、Mitigation 策略

| 策略 | 机制 | 代表方法 |
|------|------|----------|
| Better data | 减少 spurious correlation | 数据清洗、去偏 |
| RAG | External knowledge grounding | Lewis et al., 2020, https://arxiv.org/abs/2005.11401 |
| CoT | 显式推理链 | Wei et al., 2022, https://arxiv.org/abs/2201.11903 |
| ToT | Tree search over reasoning | Yao et al., 2023, https://arxiv.org/abs/2305.10601 |
| Ensemble | Majority voting / debate | Du et al., 2023, https://arxiv.org/abs/2305.14325 |
| Factuality-enhanced training | 加入 factuality loss | |
| New decoding | Contrastive decoding, etc. | Li et al., 2022, https://arxiv.org/abs/2211.09769 |

---

## 五、Are LLMs Random or Deterministic?

笔记中这个讨论很有哲学味。

**Deterministic 部分**:
- matmul, sum, layer norm 都是 deterministic
- 给定相同 weights + 相同 input + 相同 random seed → 相同 output

**Random 部分**:
- Sampling: temperature, top-k, top-p
- Dropout（训练时）

**Randomness 的必要性**:
- Creativity & diversity
- 避免 likelihood trap（mode collapse 到 degenerate sequence）

**Randomness 的代价**:
- Uncertainty → hallucination

这是一个 fundamental tradeoff：**diversity vs fidelity**。

---

## 六、我的 Intuition 构建

Andrej，结合你一直强调的 "build intuition"，我总结几个 takeaways：

### 6.1 Hallucination 是 spectrum，不是 binary

理论 proof 用 $h(s) \neq f(s)$ 是 binary，但实际中 hallucination 是 graded。应该考虑：

$$\text{Hallucination Rate} = \mathbb{E}_{s \sim D}[\mathbb{1}[h(s) \neq f(s)]]$$

其中 $D$ 是 input distribution。理论 proof 说这个 rate > 0，但没说不能接近 0。

### 6.2 三类 hallucination 需要不同解法

- **Data-driven**（factuality）→ RAG, better data
- **Architecture-driven**（exposure bias, softmax bottleneck）→ architectural innovation
- **Inference-driven**（sampling randomness）→ constrained decoding

### 6.3 Alignment 与 Factuality 的张力

$$\text{aligned output} = \arg\max_y [R(x, y) - \beta \text{KL}]$$

$R$ 衡量 "preferred"，不一定衡量 "correct"。这是 RLHF 的根本张力。

可能的方向：factuality-aware RLHF，把 factuality 加入 reward。

### 6.4 对角化 proof 的 deep meaning

这个 proof 类似 **Gödel incompleteness** 或 **Halting Problem**——它说的是："no formal system can capture all truths"。

对 LLM 而言：no computable function can capture all ground truths。这是 **computability 的内在限制**，不是 engineering 问题。

但就像 Gödel 不阻止数学进步，这个 proof 也不应该阻止我们 reduce hallucination rate。

---

## 参考链接汇总

- Xu et al., 2024: https://arxiv.org/abs/2401.11817
- Survey paper: https://arxiv.org/abs/2311.05232
- Reversal Curse: https://arxiv.org/abs/2309.12288
- Scheduled Sampling: https://arxiv.org/abs/1506.03099
- Softmax Bottleneck: https://arxiv.org/abs/1711.03953
- Self-Consistency: https://arxiv.org/abs/2203.11171
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- CoT: https://arxiv.org/abs/2201.11903
- RAG: https://arxiv.org/abs/2005.11401
- QAGS: https://arxiv.org/abs/2004.04228
- QuestEval: https://arxiv.org/abs/2103.09338
- G-Eval: https://arxiv.org/abs/2303.16634
- SelfCheckGPT: https://arxiv.org/abs/2303.08896
- Shortcut Learning: https://arxiv.org/abs/2007.12222
- Improving LLMs with uncertainty: https://arxiv.org/abs/2205.14334
- Contrastive Decoding: https://arxiv.org/abs/2211.09769
- LLM Debate: https://arxiv.org/abs/2305.14325

希望这些分析对你构建 intuition 有帮助，Andrej。如果想 dive deeper 到某个具体 sub-topic（比如 softmax bottleneck 的 MoS 解法、或 RLHF reward shaping for factuality），随时告诉我。
