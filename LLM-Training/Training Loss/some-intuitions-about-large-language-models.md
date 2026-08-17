---
source_pdf: some-intuitions-about-large-language-models.pdf
paper_sha256: 8645ffa2cf7b0c2d10a53e6c551a943fb6c3353e60be023ce3624ec64f1d5eb6
processed_at: '2026-08-12T08:45:11-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这六条直觉

**核心一句话**：LLM 就是一个疯狂吃书的孩子，吃着吃着就开窍了。

---

## 直觉一：猜下一个词 = 同时学无数门课

表面看，LLM 做的事很蠢：给前面一段话，猜下一个词。但是，语料库里的每一段话，都在偷偷给它出题。

比如它读到 "我是 Jason Wei，在 OpenAI 工作，研究方向是大型语言 ___"。

它要猜下一个词是 "模型"。这背后它其实同时在做：
- 语法题（"模型"是名词，跟在形容词后面）
- 常识题（OpenAI 是搞 AI 的）
- 世界知识题（Jason Wei 这个人研究 LLM）
- 词汇题（"模型"和"大型语言"搭配）

所以 next-word prediction 表面上是 1 个任务，实际上等价于同时上几亿节不同科目的课。这就是为什么这个看似简单的目标，能逼出智能来。

---

## 直觉二：上下文学习 = 现场抄作业

过去几十年，机器学习都是 "给输入，给输出，让模型学规律"。比如给它看 100 张猫狗图，让它学分类。

LLM 直接把这套搬进了 next-word prediction。

你给它几个例子：
"好 → 正面；坏 → 负面；开心 → 正面；难过 → ___"

它猜下一个词是 "负面"。它其实在现场抄作业，从你给的几个例子中提炼出规律，然后直接套用。这就是 In-Context Learning（上下文学习）。

GPT-3 论文证明：你给的例子越多，它现场表现越好。这很符合直觉——你给一个人看 5 个示范，总比只看 1 个示范做得好。

---

## 词的信息密度天差地别

这是很重要但容易忽视的一点：不是所有词都一样值钱。

- 有的词极其好猜： "我是 Jason Wei，在 OpenAI 工作，研究大型语言 ___" → 谁都知道下一个是 "模型"。这种词信息量极低，预测错了也无所谓。
- 有的词极难猜： "Jason Wei 最喜欢的颜色是 ___" → 几乎不可能猜中。这种词信息量极高。
- 有的词需要算半天： "((8-2)*3+4)^3/8 = ___" → 答案是确定的，但是你得算。

问题来了：如果你是 ChatGPT，看到题目必须立刻吐答案，遇到需要算的题，你根本来不及算。

**解决方案：给它时间 "思考"。**

做法很简单：在 few-shot 示范里，答案前面加一段推理过程。比如不直接写 "答案：11"，而是写 "Roger 有 5 个球，买了 2 罐，每罐 3 个，所以多 6 个，5+6=11，答案：11"。

模型一看：哦，原来解题前可以先算一算。于是它在回答难题前，会先吐一段 reasoning，把计算摊开在 token 序列上做。这就是 Chain-of-Thought（思维链）。

本质是用 token 数量换 compute 预算。直接给答案只有 1 个 token 的算力，先写一段推理就有几十个 token 的算力。

---

## 直觉四：模型越大，Loss 越低，这事可以预测

训练 LLM 很贵，动辄几千万美元。为什么大家还敢砸钱？因为有一条经验规律叫 Scaling Law（规模法则）。

规律极简单：模型参数翻倍，数据翻倍，loss 会按照一条平滑的 power law（幂律）下降。

$$L(C) = A \cdot C^{-\alpha} + L_\infty$$

意思是：投入的算力 $C$ 越大，loss $L$ 越低，且下降趋势是可预测的幂函数。$L_\infty$ 是自然语言本身的熵下限，你再怎么扩大模型也降不到 0。

最神奇的是：你可以用小模型跑出来的 loss 曲线，外推预测大模型的 loss。OpenAI 训练 GPT-4 之前，用只有 GPT-4 千分之一算力的小模型，预测出了 GPT-4 的最终 loss，误差在 0.1% 以内。

这就是为什么大厂敢砸钱训 frontier model——因为结果可预测，投资有回报。

为什么 scaling 有效？两个粗略解释：
1. 参数越多，能记住的事实知识越多（每个参数约存 2 bits）。
2. 小模型只能学一阶相关（比如 "好" 常跟正面词搭配），大模型能学复杂组合（多个 attention head 组合起来做推理）。

---

## 直觉五：整体平滑进步，单科可能突然开窍

上一条说 loss 平滑下降。但是如果你把 loss 拆开看，不同能力的进步曲线完全不一样。

整体 loss 可以看作几百个任务的加权平均：

$$\text{loss}_{\text{总}} = 0.000000001 \times \text{loss}_{\text{语法}} + \ldots + 0.000000001 \times \text{loss}_{\text{数学}} + \ldots$$

当 loss 从 4 降到 3，所有任务不会同步变好。有的任务（比如语法）在 loss=4 时就已经完美了，再降也降不动；有的任务（比如三位数乘法）在 loss=4 时是随机猜，在 loss=3 时突然能做对 90%。

这就是 **Emergence（涌现）**：某些能力在小模型里完全没有，一旦模型规模超过某个阈值，就突然冒出来。量变引起质变。

Jason 团队找了 8 个这样的任务：三位数加减法、波斯语问答、TruthfulQA、词序还原等。小模型在这些任务上等于瞎蒙，一旦模型规模到某个阈值，performance 突然跳到远超随机水平。

涌现有三个深刻含义：
1. **不可预测**：你不能用小模型的平滑曲线外推出涌现能力，因为它是突然发生的。
2. **非预期**：训练者从来没告诉模型 "你要学三位数乘法"，它是 next-word prediction 的副产品。
3. **继续 scale 可能解锁新能力**：这是一个强经验先验，支撑大家继续造更大的模型。

（补充一个争议：2023 年有人指出，涌现可能只是因为评估指标太极端。如果你不用 "对/错" 这种 0/1 指标，改用 log-likelihood 这种连续指标，能力其实是平滑提升的。换句话说，涌现可能是测量方式的假象。）

---

## 直觉六：真正的上下文学习，只有大模型才会

直觉二说 ICL 有效。但是问题来了：模型是真的从例子中学了规律，还是只是学了个表面格式？

2022 年有人做了个狠实验：把 in-context 例子的 label 全打乱（比如 "好→负面，坏→正面"），结果 GPT-3 表现几乎不变。

这说明什么？GPT-3 可能根本没看 label，只学了 "哦，输入要对应一个正面/负面的词" 这种格式信息。

但是 Jason 指出：GPT-3 才 175B，在今天不算大。换大模型试试——把 label 翻转，大模型（PaLM-540B, code-davinci-002）performance 显著下降，说明它们真的在看 label，真的在学 input→output 的映射。

所以：**真正的 ICL，是涌现出来的能力，只有足够大的模型才会。**

小模型只能学表面皮毛（label 空间、输出格式）；大模型能形成真正的 task inference 机制，能根据 prompt 中的 examples 动态推断当前任务，甚至 follow 翻转的 label。

---

## 把六条串起来看

整个 LLM paradigm 的核心逻辑可以这样概括：

1. **目标极简单**：猜下一个词。但语料足够复杂，所以等于在同时学无数门课。
2. **规模放大**：参数越多、数据越多，整体 loss 按幂律平滑下降，这事可预测，所以大家敢砸钱。
3. **质变发生**：loss 虽然平滑下降，但某些能力会突然解锁（涌现），真正的 ICL 和 CoT 推理都是这么来的。
4. **推理时给 compute**：通过 CoT 等技巧，让模型在输出答案前先 "思考" 一串 token，等于把推理时算力拉开。

**一句话总结**：简单的目标 + 海量数据 + 规模放大 + 推理时给算力 = 智能。

---

参考资料：
- Jason Wei 原文：https://jasonwei20.github.io/blog/6
- GPT-3 paper: https://arxiv.org/abs/2005.14165
- Chain-of-Thought paper: https://arxiv.org/abs/2201.11903
- Emergent Abilities: https://arxiv.org/abs/2206.07682
- Kaplan scaling laws: https://arxiv.org/abs/2001.08361
- Chinchilla scaling laws: https://arxiv.org/abs/2203.15556
- "Are Emergent Abilities a Mirage?": https://arxiv.org/abs/2304.15004
- Induction heads: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- The Bitter Lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html

---

# Six Intuitions about Large Language Models 深度解读

这篇 blog 是 Jason Wei（OpenAI 研究员，chain-of-thought prompting 的作者）写的关于 LLM 的六条核心直觉。原文链接: https://jasonwei20.github.io/blog/6

---

## Intuition 1: Next-word prediction = Massively multi-task learning

### 核心思想

预训练目标 $\mathcal{L}_{\text{LM}} = -\sum_{t=1}^{T} \log P(x_t | x_{<t})$ 表面上极简单，但是 corpus 中每一个 token $x_t$ 实际上对应一个隐式的 task。Jason 用 table 形式列举了 grammar、lexical semantics、world knowledge、sentiment analysis、translation、arithmetic 等任务。

### 更细的技术视角

你可以把整个 corpus 看作一个 mixture distribution：

$$P(x_t | x_{<t}) = \sum_{k=1}^{K} \pi_k(x_{<t}) \, P_k(x_t | x_{<t})$$

其中 $k$ 索引隐含的 sub-task（grammar / factual recall / arithmetic / comma placement 等），$\pi_k(x_{<t})$ 是由 context 决定的 mixing weight。模型必须在每一 step 动态推断当前属于哪个 sub-task，然后在对应的 conditional distribution 上做预测。这就是为什么 next-token prediction 虽然只有一个 loss term，却等价于一个隐式的 multi-task learning 问题。

Jason 的"odd tasks"例子非常关键：预测逗号、预测 "that" 这种 grammar function word、预测 "relies" 这种几乎不可预测的 content word。这说明 information density 在 token 级别高度不均匀（参见 Intuition 3）。

一个延伸的联想：这个视角和 Xie et al. 的 **"Data Compression for Intelligence"** 理论一致 — compression ratio 越高，模型被迫提取的 latent structure 越多。参考: https://arxiv.org/abs/2304.09953

---

## Intuition 2: In-context learning as next-word prediction

### 核心思想

传统 ML 学习 $f: \mathcal{X} \to \mathcal{Y}$ from $(x_i, y_i)$ pairs。ICL 把这个范式嵌入到 next-word prediction 中：

$$P(y_{\text{query}} | x_{\text{query}}, (x_1, y_1), \ldots, (x_n, y_n), \text{instruction})$$

GPT-3 paper (https://arxiv.org/abs/2005.14165) 显示增加 in-context examples 数量 $n$ 单调提升 performance。

### 更细的技术视角

从 Bayesian 角度，ICL 可以理解为 **implicit Bayesian inference**（Xie et al., 2021, https://arxiv.org/abs/2111.02080）：

$$P(y_{\text{query}} | \text{prompt}) = \sum_{h \in \mathcal{H}} P(y_{\text{query}} | x_{\text{query}}, h) \, P(h | \text{prompt})$$

其中 $h$ 是 latent task concept，prompt（包含 demonstrations）让模型 posterior $P(h | \text{prompt})$ 集中在正确的 task concept 上。Transformer 通过 in-context gradients（Olsson et al. 的 in-context learning induction heads, https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html）实现一种隐式的 SGD。

公式中变量含义：
- $h \in \mathcal{H}$：latent task hypothesis
- $P(h | \text{prompt})$：给定 prompt 后，task hypothesis 的 posterior
- $P(y_{\text{query}} | x_{\text{query}}, h)$：在给定 task hypothesis 下的预测分布

Jason 提到一个关键 insight：**没有第一性原理要求我们必须用 (input, output) pairs**。人类交流还包括 instructions、explanations、interactive teaching。这暗示未来的 prompting paradigm 可能更丰富，比如 dialog-based interactive alignment。

---

## Intuition 3: Tokens have different information density → give LLMs time to think

### 核心思想

不同 token 的 information content 差异巨大。用 Shannon entropy 衡量：

$$H(x_t | x_{<t}) = -\sum_{x} P(x | x_{<t}) \log P(x | x_{<t})$$

- "I'm Jason Wei, a researcher at OpenAI working on large language **models**" → $H(x_t | x_{<t}) \approx 0$，几乎确定
- "Jason Wei's favorite color is **___**" → $H(x_t | x_{<t}) \approx \log |V|$，几乎均匀分布
- "((8-2)*3+4)^3/8 = ___" → entropy 可能不高（答案确定），但是 **compute cost** 极高

### Chain-of-Thought (CoT) 的技术细节

CoT prompting (Wei et al., 2022, https://arxiv.org/abs/2201.11903) 在 few-shot exempla 中展示推理过程：

```
Q: Roger has 5 balls. He buys 2 more cans, each with 3 balls. How many balls?
A: Roger started with 5 balls. 2 cans × 3 balls = 6 balls. 5 + 6 = 11. Answer: 11.
```

### 为什么 CoT 有效 — compute 视角

Forward pass 的 compute 大约是 $O(N \cdot d^2)$（$N$ = token 数，$d$ = hidden dimension）。如果你直接 output 答案，模型只有 $O(1)$ tokens 的 compute budget 来完成推理。CoT 把推理展开成 $O(N_{\text{reasoning}})$ tokens，每个 token 对应 $O(d^2)$ 次运算，总 compute 扩展到 $O(N_{\text{reasoning}} \cdot d^2)$。

更深层的解释（参考 https://arxiv.org/abs/2310.03557, "Faith and Fate"）：Transformer 的 expressiveness 受限于 depth。CoT 把一个需要 depth-$D$ 的 computation 串行化成 $D$ 个 step，每个 step 只需要 depth-1 的 computation，从而突破固定 depth 的 expressiveness bottleneck。

变量含义：
- $N$：token 序列长度
- $d$：hidden dimension
- $D$：需要串行执行的 reasoning steps
- $N_{\text{reasoning}}$：CoT 中的 reasoning tokens 数

### Least-to-most prompting

更复杂的 decomposition 方法（https://arxiv.org/abs/2205.10625）：先把 prompt 分解成 sub-problems $s_1, s_2, \ldots, s_m$，然后 sequentially solve：

$$\text{answer} = f(s_m | \text{answer}(s_1), \ldots, \text{answer}(s_{m-1}))$$

---

## Intuition 4: Scaling laws

### Kaplan scaling law

Kaplan et al. (2020, https://arxiv.org/abs/2001.08361) 提出：

$$L(N, D) = \left[ \left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D} \right]^{\alpha_D}$$

简化版本（compute-limited regime）：

$$L(C) = A \cdot C^{-\alpha} + L_\infty$$

其中：
- $L$：test loss（cross-entropy, nats/token）
- $N$：parameter count
- $D$：dataset size (tokens)
- $C$：compute (FLOPs)，$C \approx 6ND$
- $\alpha$：power-law exponent，经验上约 0.05–0.1
- $L_\infty$：irreducible loss（自然语言的 entropy lower bound）
- $N_c, D_c, A$：拟合常数

### Chinchilla scaling law

Hoffmann et al. (2022, https://arxiv.org/abs/2203.15556) 修正了 Kaplan 的结论，发现 optimal allocation 应该让 $N$ 和 $D$ 大致等比例 scale（约 20 tokens / parameter），而不是 Kaplan 建议的数据欠采样：

$$D^* \approx 20 \cdot N^*$$

### 为什么 scaling 有效 — 两个 hand-wavy 解释

1. **Knowledge capacity**：参数量 $N$ 决定了模型能 memorize 的 facts 数量。粗略估计，每个参数大约能存 ~2 bits 信息（参考 https://arxiv.org/abs/2304.04262），所以一个 70B 模型理论上能存 ~17.5 GB 的 factual knowledge。

2. **Circuit complexity**：小模型容量受限，只能学 first-order correlations（bigram-like patterns）；大模型可以组合多个 attention head 形成 compositional circuits（参考 Anthropic 的 circuits work, https://transformer-circuits.pub/），实现 multi-step reasoning、induction、translation 等 complex heuristics。

### 可预测的 loss extrapolation

OpenAI 的 GPT-4 technical report（https://arxiv.org/abs/2303.08774）展示了可以用小模型 loss 曲线外推预测大模型 loss，accuracy 在 0.1% 以内。这是 infra 投资的关键依据：用 $10^4 \times$ less compute 的小 run 预测 frontier model 的 final loss。

---

## Intuition 5: Emergence

### 核心定义

Ability $a$ 是 **emergent** 当且仅当：

$$\exists \, C^* \text{ s.t. } \forall C < C^*: \text{perf}(a, C) \approx \text{random}; \quad \forall C > C^*: \text{perf}(a, C) \gg \text{random}$$

其中 $C$ 是 training compute（FLOPs），$\text{perf}(a, C)$ 是 ability $a$ 在 compute $C$ 下的 performance。

### Jason 论文中的 8 个 emergent tasks

Wei et al. (2022, https://arxiv.org/abs/2206.07682) 列举了 emergence 的 8 个例子：

| Task | Metric | Random baseline | Emergence threshold |
|------|--------|-----------------|---------------------|
| Modular arithmetic (mod $p$) | Exact match | $1/p$ | ~$10^{22}$ FLOPs |
| IPA transliteration | Edit distance | High | ~$10^{22}$ FLOPs |
| Word unscramble | Exact match | Low | ~$10^{22}$ FLOPs |
| Persian QA | F1 | Low | ~$10^{22}$ FLOPs |
| TruthfulQA | MC accuracy | 0.25 | ~$10^{22}$ FLOPs |
| Grounded mappings | Accuracy | Low | ~$10^{22}$ FLOPs |
| Multi-task NLU | Accuracy | Low | ~$10^{22}$ FLOPs |
| Word in context | Accuracy | 0.5 | ~$10^{22}$ FLOPs |

### 为什么会出现 emergence — 几种假说

1. **Discrete metric 假说**（Schaeffer et al., 2023, https://arxiv.org/abs/2304.15004）：emergence 可能是 evaluation metric 的 artifact。如果用 continuous metric（log-likelihood）而非 discrete metric（exact match / accuracy），能力其实是平滑提升的。Jason 后来的推文也承认这是 valid critique。

   公式上：accuracy $= \mathbf{1}[\arg\max P(y|x) = y^*]$，而 log-likelihood $= \log P(y^* | x)$。后者连续，前者有 threshold 效应。

2. **Compositional circuits 假说**：某些 task 需要多 step 的 compositional circuit（如 modular arithmetic 需要数 step 的 in-context induction）。当模型 capacity 不足时，circuit 无法形成；capacity 突破 threshold 后，多个 attention head 协同形成 circuit，performance 跳变。

3. **Phase transition 假说**（类似统计物理）：loss landscape 在 critical compute 处发生 symmetry breaking，模型从 "shallow pattern" phase 跃迁到 "compositional" phase。

### 三个 implication

1. **不可外推性**：emergence 无法从 scaling law 预测，因为 scaling law 是 smooth power law，而 emergence 是 phase transition。
2. **非显式指定**：trainer 没有显式 optimize 这些 abilities，它们是 next-word prediction 的副产品。
3. **进一步 scaling 可能 unlock 新 ability**：这是一个 strong empirical prior，支持继续训练 frontier models。

---

## Intuition 6: True in-context learning only happens in large-enough models

### Label-flipping 实验

Min et al. (2022, https://arxiv.org/abs/2202.12837) 发现 GPT-3 (175B) 即使把 in-context labels 随机化，performance 也几乎不下降。结论：ICL 不是真的学 (input, output) mapping，而是学 format / label space。

但是 Jason 指出 GPT-3 相对今天 frontier 模型不够大。**Flipped labels** 实验显示：当把 positive ↔ negative 翻转时，**大模型**（PaLM-540B, code-davinci-002, text-davinci-002）performance 显著下降（说明它们真的 follow flipped mapping），而**小模型**几乎无影响（说明它们没真正用 label 信息）。

### 理论解释

这是一个典型的 **emergent ICL** 现象。可以理解为：

$$\text{ICL ability}(C) = \begin{cases} 0 & \text{if } C < C^* \\ \text{function of } C & \text{if } C \geq C^* \end{cases}$$

小模型的 in-context gradient mechanism（induction heads）不够强，只能学 surface-level format；大模型能真正形成 task inference circuits，posterior $P(h | \text{prompt})$ 能集中到 flipped task concept 上。

### 相关 work

- Olsson et al., "Induction Heads" (https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)：induction heads 在 ~2-layer 模型后出现，是 ICL 的 mechanism。
- Bietti et al. (2023, https://arxiv.org/abs/2310.13560)：birth of a transformer，研究不同 circuit 在 training 过程中的 emergence 顺序。

---

## Cross-cutting themes

### 1. Simple objective + complex data → intelligence

$\mathcal{L} = -\log P(x_t | x_{<t})$ 是一个 scalar loss，但是 corpus 的 complexity 使其隐式成为 $K \to \infty$ 的 multi-task objective。这呼应 Rich Sutton 的 "The Bitter Lesson"（http://www.incompleteideas.net/IncIdeas/BitterLesson.html）：简单 scalable 方法 + 大 compute > 复杂 hand-engineered 方法。

### 2. Scale unlocks qualitative changes

- Smooth loss improvement (Intuition 4) + Emergent task abilities (Intuition 5) + Emergent ICL (Intuition 6) 三个直觉构成一个统一图景：**loss 平滑下降，但是 underlying ability 结构发生 phase transitions**。

这类似物理中的 free energy：thermodynamic potential 平滑变化，但是 order parameters（如 magnetization）可以 phase transition。

### 3. Compute allocation as inference-time control

Intuition 3 (CoT) 是 **inference-time scaling** 的雏形。后续工作如 OpenAI o1 / test-time compute scaling（https://arxiv.org/abs/2408.03314, "Scaling LLM Test-Time Compute Optimally"）把这个 idea 推到极致：在 inference 时让模型生成大量 reasoning tokens，用 verifier / search 优化。

公式上，inference-time scaling 可以表示为：

$$\text{perf}(\text{task}) = \max_{s_1, \ldots, s_T} \, V(\text{answer} | s_1, \ldots, s_T) \cdot \prod_{t=1}^{T} P(s_t | s_{<t}, \text{prompt})$$

其中 $s_t$ 是 reasoning step，$V$ 是 verifier，$T$ 是 inference compute budget。

### 4. Data inspection as a research method

Jason 反复强调手动 inspect data。这是一个非常 Karpathy-style 的方法论（参考 Karpathy 的 "Software 2.0", https://karpathy.medium.com/software-2-0-a64152b37c35 和他的 nanoGPT 教程）。在 LLM 时代，理解 model behavior 的最快路径往往是 inspect 个别 examples，而非 aggregate metrics。

---

## 延伸阅读

- Jason Wei 个人主页：https://jasonwei20.github.io/
- Emergent Abilities paper: https://arxiv.org/abs/2206.07682
- Chain-of-Thought paper: https://arxiv.org/abs/2201.11903
- GPT-3 paper: https://arxiv.org/abs/2005.14165
- Kaplan scaling laws: https://arxiv.org/abs/2001.08361
- Chinchilla scaling laws: https://arxiv.org/abs/2203.15556
- GPT-4 technical report: https://arxiv.org/abs/2303.08774
- "Are Emergent Abilities a Mirage?": https://arxiv.org/abs/2304.15004
- Induction heads: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Implicit Bayesian ICL: https://arxiv.org/abs/2111.02080
- Data compression for intelligence: https://arxiv.org/abs/2304.09953
- Test-time compute scaling: https://arxiv.org/abs/2408.03314
- Bitter Lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- Software 2.0: https://karpathy.medium.com/software-2-0-a64152b37c35

---

## 一个 unifying 的 mental model

如果要把六条直觉压缩成一个 picture：

1. **Objective**（Intuition 1, 2）：next-word prediction 是一个 universal surrogate，隐式 cover 所有 task。
2. **Capacity**（Intuition 4）：scaling 增加 parameter / data capacity，loss 平滑下降。
3. **Structure**（Intuition 5, 6）：capacity 增加到 critical point 后，model 内部形成新的 compositional circuits，对应 emergent abilities 和 true ICL。
4. **Compute allocation**（Intuition 3）：CoT / test-time compute 让 inference-time compute budget 可控，允许模型在 information-dense token 上 allocate 更多 reasoning。

整个 story 是：**simple loss + scaling + compute allocation = intelligence**。这正是 LLM paradigm 的核心。
