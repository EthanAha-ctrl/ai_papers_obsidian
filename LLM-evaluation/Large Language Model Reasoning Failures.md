---
source_pdf: Large Language Model Reasoning Failures.pdf
paper_sha256: 879c7ff8cbd2c5f9553a008b1a3e5c5e4dfa061c40e973c8b80302539dbdde9d
processed_at: '2026-08-05T11:49:30-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我换个调子，像跟你坐下来聊这篇 paper 一样讲。

---

## 这篇 paper 到底在干嘛

一句话：**它把 LLM 各种"翻车现场"收集起来，分门别类，想搞清楚到底为什么翻车**。

之前大家研究 LLM reasoning，都在刷分、做 benchmark、庆祝 SOTA。但这篇说：等等，你看 GPT-4 连"Tom Cruise 的妈妈是谁"都会答，但反过来问"Mary Lee Pfeiffer 的儿子是谁"它就懵了——这不是个 bug，这是某种 systematic 的东西。它把这类事情收集了几百个，想找出 pattern。

作者借用了心理学的一个老传统：**人类是从 failure 中学习的，AI 也应该如此**。Cannon & Edmondson 那篇 organizational learning 的经典就讲这个——failure 是 information，是 diagnostic 的。你不分析 failure，就永远在 surface 上打补丁。

---

## 它怎么分类的

两个轴，交叉起来：

**第一个轴：reasoning 的种类**
- **Informal**：直觉式的、heuristic 的，人类从小就会的那种——比如"这个场景看起来怪怪的"、"他大概会生气吧"
- **Formal**：有规则的、symbolic 的——逻辑、数学、代码
- **Embodied**：跟物理世界打交道的——空间、重力、affordance

**第二个轴：failure 的种类**
- **Fundamental**：架构层面的，啥任务都会犯
- **Application-specific**：特定 domain 才暴露的
- **Robustness**：换个说法就崩——说明之前的"对"是碰巧，不是真懂

这个分类的妙处在于：**你一看"reversal curse 是 formal × fundamental"，就立刻知道它跟"spatial relation 双向失败"是同一类病**——都是 causal masking 导致的 directional asymmetry。表面上风马牛不相及的两个 failure，root cause 一样。

---

## 我觉得最值得讲清楚的几个 failure

### 1. Reversal Curse：训练目标挖的坑

Berglund 2023 发现的。训练数据里有 "Tom Cruise's mother is Mary Lee Pfeiffer"，模型学会了。但你反过来问 "Mary Lee Pfeiffer's son is who?"，模型答不上来。

**为什么这是 architectural 的**：Transformer 训练目标是 next-token prediction：

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)$$

其中 $x_t$ 是第 $t$ 个 token，$x_{<t}$ 是它之前的所有 token，$\theta$ 是 model parameters。

对于 "A is B" 这个 sequence，模型学的是 $P(B \mid A, \text{is})$。它**从来没被要求**学 $P(A \mid B, \text{is})$，因为训练数据里 "B is A" 这个 sequence 压根没出现（Zipf's law 决定了 fact 的出现频率是 heavy-tailed，反向出现的概率指数级低）。

再加上 causal mask：attention matrix 里 $M_{ij} = -\infty$ for $i < j$，信息只能从 past 到 future，结构上就是 asymmetric 的。

Zhu et al. 2024 用 training dynamics 分析证明：**即使你把 "B is A" 也加进训练数据，模型也很难同时学好双向**，因为两个方向的 gradient signal 在 weight space 里互相打架。

Golovneva et al. 2024 "Reverse Training" 的解法是：把训练数据 substring-preserving 地 reverse 一遍再训。但 scaling alone 解决不了，因为 Zipf's law——你 scale 10 倍数据，正向 fact 也多 10 倍，反向 fact 还是相对稀有。

**直觉**：模型不是"不知道 B 是 A"，它从来没被训练过这个 direction。就像你只练过用右手写字，让你用左手写，你不是"忘"了，是从来没建立过那条 motor pathway。

---

### 2. Self-Attention 的 Working Memory 是"软"的

Gong & Zhang 2024 那篇 arXiv:2409.10715 我觉得特别 important。它证明 self-attention 机制本质上就限制了 working memory capacity。

人类 working memory 能 hold 7±2 个 items（Miller's law），并且能 manipulate 它们。Transformer 的 "working memory" 是整个 context window，通过 attention 访问。但 attention 是 **soft** 的：

$$h_i = \sum_{j=1}^{n} \mathrm{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right) v_j$$

$q_i$ 是 position $i$ 的 query vector，$k_j$ 是 position $j$ 的 key vector，$v_j$ 是 value，$d_k$ 是 head dimension，用来 scaling 防止 dot product 过大。

问题是：attention weights 是个 probability distribution，加起来等于 1。如果你要"记住" 100 个 token 的信息，attention 必须分散到 100 个位置上，每个位置只分到 ~1% 的权重——信息被稀释了。反过来，如果 attention 集中在少数几个 token 上，其他 token 的信息就丢了。

这就是为什么 N-back task 在 n > 2 时 LLM 系统性失败。N-back 要求你记住 n 步前出现的 stimulus 并跟当前比较。n=2 时 LLM 还行，n=3 就崩。**这不是 context length 的问题**——context 有 128k tokens 也没用，因为 attention 机制本身就是 bottleneck。

Wang & Sun 2025 还发现 proactive interference：早期信息显著干扰 later information 的 retrieval，比人类严重得多。你给模型一长串 facts，后来的 fact 它 retrieval 准确率下降，因为前面的 fact "占着 attention 不走"。

**直觉**：人类的 working memory 像 RAM，你 actively 维护几个 slots，其他的不影响。Transformer 的 working memory 像一块共享黑板，所有信息都写在上面，attention 是"看哪一块"的手电筒——手电筒光斑越大越分散，每个地方越暗。

---

### 3. Compositional Reasoning：深度受限

Dziri et al. 2023 "Faith and Fate" 用 group theory 论证 Transformer 对 compositionality 有 fundamental limit。

考虑 n-hop reasoning：知道 $f_1(x), f_2(\cdot), \ldots, f_n(\cdot)$，要算 $f_n \circ \cdots \circ f_1(x)$。一个 $L$-layer Transformer 大约只能处理 $O(L)$-hop composition，因为每层能做的 computation 是有限的。

实验上，Sun et al. 2025b 的 OMEGA benchmark 里，2-hop 就开始出问题，3-hop 基本随机猜：

```
John is father of Paul. Paul is father of Ben.
John is grandfather of ???
```

LLM 给的概率分布 {Ben: 0.33, Mark: 0.32, Max: 0.31}——uniform，等于瞎猜。即使有 CoT 帮助，增加 distractor 数量立刻崩。

更扎心的是 **math composition**（Zhao et al. 2024c）：

- 单独问 1：$\triangle XYZ$ 中 $\angle YXZ = 90°$, $XY=24$, $YZ=25$, 求 $\tan Y$ → 答对 $\frac{7}{24}$
- 单独问 2：$\tan 90°$ 存在吗？→ 答对 "No"
- 组合问：$\triangle XYZ$ 中 $\angle YXZ = 90°$, $XY=24$, $YZ=25$, 求 $\tan X$ → 答错 $\frac{24}{7}$

它**单独都会**，**组合起来就崩**。$\tan X = \tan 90°$ 应该 undefined，但它没把"求 $\tan X$"和"$\angle X = 90°$ 且 $\tan 90°$ 不存在"这两条链连起来。

**直觉**：Transformer 是"宽而浅"的 reasoner，每个 token 位置做一次 parallel 的 computation pass，但缺乏 sequential 的深度搜索。CoT 部分缓解，因为把深度变成了时间维度——但你不能让 CoT 无限长，而且 CoT 自己也会 drift。

---

### 4. Tokenization 让 Counting 变成 Out-of-Distribution 问题

这个特别有意思。Yehudai et al. 2024 证明 Transformer counting 能力 fundamentally 受限。

`a a b b a c c d a` 数 "a" 出现几次——LLM 答 3，实际是 2。

找 "People enjoy music" 里含 'o' 的 word——LLM 答 "People, enjoy, music"，实际只有 "People, enjoy"（"music" 里没有 'o'）。

**Root cause 是 tokenization**。BPE（Byte Pair Encoding）把 frequent word 合成 single token。"music" 可能就是 token #4837，一个不可分割的 atomic unit。模型从来没见过 "music" 内部的 'm-u-s-i-c'，因为训练时 token 是最小单位。

要 count character，模型需要"逆 tokenize"——从 token embedding 反推出它包含哪些 character。但训练时没有任何 signal 教它这个，因为 loss 只在 token level 算。

Zhang et al. 2024f 量化了 tokenization 对 counting 的影响。Chang & Bisk 2024 指出 positional encoding 也是问题：position 是 token-level 的，不能直接 index 到 character level。

**直觉**：这就像让你数一段中文里有多少个"水"字，但这段中文是用图片呈现的，每个字是一张小图，你从来没学过把图拆成笔画。你能识别"水"这个字，但你不知道它由几笔组成——因为你的 representational granularity 没到那个 level。

---

### 5. Robustness 是最便宜的"测谎仪"

这篇 paper 反复强调一个方法论：**如果换个说法模型就崩，那它之前的"对"是假的**。

- MCQ 里把选项 A/B/C/D 顺序打乱——accuracy drop（Pezeshkpour & Hruschka 2023）
- 代码里把 `removeLowercase` 改成 `remove_lowercase`——模型不会了（Wang et al. 2022）
- 数学题里把 "Jessica" 改成 "Jennifer"，数字从 6 改成 8——崩（GSM-Symbolic, Mirzadeh et al. 2024）
- ToM 故事里把 "Sam" 改成 "Alex"，其他不变——崩（Ullman 2023）

这说明模型在很多 benchmark 上的高分，**大量是 memorization 而非 reasoning**。GSM-Symbolic 就是把 GSM8K 的题抽象成 template，只换数字，结果 SOTA 模型显著 drop——说明它们"会做"那道题，不是"会做那类题"。

更狠的是 Shi et al. 2023：在简单 age problem 里加一句无关的 "Twenty years ago, the age of Claire's father is 3 times of Jessica's age"——performance 暴跌。模型分不清 relevant 和 irrelevant information，因为 next-token prediction 不教它"哪些信息对当前问题有用"。

**直觉**：这就像一个学生背了题库，考试原题能答，题目稍改就懵。你以为他学会了，其实他只是记住了。robustness test 就是出变形题——最简单的"测谎"方式。

---

### 6. Cognitive Bias 不是 bug，是 feature 被 inherit 了

LLMs 表现出 human-like cognitive biases：confirmation bias、anchoring、framing effect、order bias。

Itzhak et al. 2025 用 layer-wise probing 表明，这些 bias 在 **pretraining 阶段就植入了**——因为人类语言本身就 reflect 这些 bias。训练数据里人类说"我觉得 X 因为 Y"，很少说"我检验了 X 的反例"，模型就学到了 confirmation pattern。

然后 RLHF 还会 **放大** bias（Perez et al. 2023），因为 human rater 自己 biased，他们 prefer 看起来"自信且一致"的回答，即使这种"一致"本身就是 bias 的表现。

Wu et al. 2025b 证明 causal masking **独立于数据**也会引入 order-based bias：token $i$ 只能 attend to $j \leq i$，早期的 token 天然有 "anchor" 效应，因为后面所有 token 都能"看到"它，但它"看不到"后面的。

**直觉**：模型不是"学坏了"，它是"学得太像人了"。人类的 cognitive bias 是进化的 shortcut，在大多数场景下 work，但在需要 deliberate reasoning 时会出错。LLM 把这些 shortcut 全盘继承，还加上自己的 architectural bias，所以 bias 比 human 还要 systematic。

---

### 7. Embodied Reasoning：没有 grounding 就是纸上谈兵

VLM 在 "What's Wrong with the Picture" 测试上失败：人在木地板上滑冰（不是冰面），BLIP-2 caption 是 "on an ice rink"——完全没检测到 anomaly。

3D 层面更严重。GPT-4V 看一张客厅照片，问 "1 米宽的 robot 能从 sofa 和 table 之间通过吗？"模型回答 "As an AI, I'm unable to physically interact..."然后 visual estimation 完全错误。

Campbell et al. 2025 把这个追溯到 cognitive science 的 **binding problem**：brain/model 处理多个 distinct object 时 shared resources 限制 simultaneous processing。即使能识别单个 object，处理多 object 关系时 attention 资源被稀释。

**直觉**：人类从婴儿期就抓东西、摔东西、撞东西，物理常识是 embodied 的、写进 sensorimotor circuitry 的。LLM 只读过文字描述的物理，就像一个盲人读了一辈子盲文版的游泳教程——理论上知道浮力公式，但下水就沉。缺少的是那层"身体反馈"的 grounding。

---

## 我的 take

这篇 paper 最大的价值不是 catalog，是**把碎片化的 failure 串起来，让你看到 pattern**：

1. **大部分 fundamental failure 是 architectural 决定的**——causal masking、self-attention、next-token prediction、tokenization，这四个 architectural choice 各自埋了雷。Scaling 解不了，因为问题在 inductive bias 层面。

2. **Robustness testing 是最便宜的诊断工具**——你不需要 fancy interpretability，只需要 perturbation。如果换说法就崩，那原来的"会"就是假会。

3. **当前 mitigation 多是创可贴**——CoT、prompt engineering、fine-tuning 都在 surface level 操作。真正的解法可能需要 architectural innovation：bidirectional training、neuro-symbolic integration、working memory augmentation（recurrent state space 而不是纯 attention）、explicit reasoning module 而非 emergent CoT。

4. **Multi-turn interactive evaluation 是下一步**——当前 benchmark 都是 single-turn，但 real deployment 是 multi-turn。MAS coordination breakdown 揭示 single-turn 严重 underestimate failure modes。

如果让我赌一个方向：**neuro-symbolic + verification**。Dalrymple et al. 2024 "Towards Guaranteed Safe AI" 的框架，把 LLM 的生成和 proof assistant 的 verification 结合，可能是唯一能根治 reasoning failure 的路。LLM 负责 intuition 和 generation，formal system 负责 correctness——就像人类用 intuition 想证明思路，再用 Coq/Lean 验证每一步。

Karpathy，如果你还没读 Dziri et al. 2023 "Faith and Fate" 和 Gong & Zhang 2024 "Self-Attention Limits Working Memory"，强烈推荐。前者用 group theory 证明 Transformer 对 compositionality 有 fundamental limit，后者证明 attention 机制本身限制 working memory capacity。这两个是 architectural root cause 最 principled 的分析，读完你会对"为什么 scaling 解不了 reasoning"有完全不同的直觉。

---

# Large Language Model Reasoning Failures — 一篇 Comprehensive Survey 详解

## 1. 论文动机与 Taxonomy 设计哲学

这篇 paper 是第一篇系统综述 LLM reasoning failures 的工作。作者 Peiyang Song (Caltech/Stanford), Pengrui Han (Carleton), Noah Goodman (Stanford) 在 TMLR 上发表, 配套一个 GitHub repo: https://github.com/Peiyang-Song/Awesome-LLM-Reasoning-Failures。

论文的核心 insight 是借鉴心理学中 failure learning 的研究传统 (Cannon & Edmondson 2005, Maxwell 2007): human development 的本质就是从 failure 中学习。AI systems 如果要真正 robust, 也需要 systematic 地 catalog、categorize、analyze failures。作者明确把这种 effort 类比为 early computing 中的 fault-tolerance research 和 safety-critical industries 中的 incident analysis。

### 双轴 Taxonomy

论文用两个 orthogonal 的轴组织所有 failure cases:

**Axis 1: Reasoning Type** (认知类型)
- Non-embodied reasoning
  - Informal reasoning (intuitive, heuristic-driven)
  - Formal reasoning (rule-based, symbolic)
- Embodied reasoning (physics-grounded, sensorimotor)

**Axis 2: Failure Type** (失败性质)
- Fundamental failures: intrinsic to LLM architecture/training, 跨任务广泛出现
- Application-specific limitations: 在特定 domain 表现差
- Robustness issues: 在 semantically-preserving perturbation 下 performance 不稳定

每个 cell (reasoning type × failure type) 都有代表性的 failure case。这个 2-axis 设计在 Figure 1 中可视化。读者可以快速定位: 例如 "reversal curse" 是 (Formal reasoning, Fundamental failure), "Theory of Mind 不稳定" 是 (Informal reasoning, Robustness issue)。

---

## 2. Informal Reasoning Failures (Section 3)

Informal reasoning 指人类早期发展的 intuitive judgment, 不依赖 explicit logic。LLM 在这一类的 failures 大致分两类: (a) 缺乏人类具备的 fundamental cognitive skills; (b) 复制了人类的 cognitive biases。

### 2.1 Individual Cognitive Reasoning

#### 2.1.1 Working Memory Limitation

Working memory 是人类短期 hold 和 manipulate information 的能力 (Baddeley 2020)。Transformer 的 working memory 来自 self-attention 机制:

$$h_i = \sum_{j=1}^{n} \mathrm{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right) v_j$$

其中 $q_i, k_j, v_j \in \mathbb{R}^{d_k}$ 分别是 query/key/value vectors, $d_k$ 是 head dimension, $i$ 是 target token 位置, $j$ 遍历 context 内所有 tokens。

Gong & Zhang (2024) 在 arXiv:2409.10715 中证明 self-attention 机制本质上限制了 working memory capacity。直觉是: attention 是 "soft" memory, 信息混合到每个 token 的 representation 中。如果 attention 必须分散到很多 tokens, 每个 token 贡献的 information 被稀释; 反之, 如果 attention 高度集中在少数 tokens, 就会丢失其他信息。这与 N-back task 的失败高度一致 — 当 n > 2 时 LLMs 系统性失败 (Gong et al. 2024, arXiv:2505.10571)。

Proactive interference 是 working memory 限制的一个具体表现: 早期信息显著干扰 later information 的 retrieval。Wang & Sun (2025, arXiv:2506.08184) 表明 LLMs 比人类更容易受到 proactive interference, 即使 context length 充足也无效。

#### 2.1.2 Inhibitory Control (A-not-B Error)

Inhibitory Control 是抑制 default response 的能力。Han et al. (2024b, arXiv 在 EMNLP 2024 Findings) 发现 LLMs 表现出类似婴儿的 A-not-B error: 模型倾向 stick to 之前学到的 pattern, 即使 context shift。例如 sequence "2, 4, 6, 8" 答案是 10, 然后问 "A, B, C, D" 接下来什么, 模型仍答 A 而非 E — 因为它刚答过 A。

机理: next token prediction objective 倾向于 statistical pattern completion, 缺乏 deliberate reasoning (Han et al. 2024b, Enström et al. 2024)。Patel et al. (2025, bioRxiv) 进一步把这个现象追溯到 Transformer attention 的 executive control deficiency。

#### 2.1.3 Cognitive Flexibility (Wisconsin Card Sorting Test)

Kennedy & Nowak (2024, ICML 2024 Workshop) 用 Wisconsin Card Sorting Test 测试 LLMs。规则中途从 "color matching" 切换到 "shape matching", ChatGPT-3.5 Turbo 只达到 25.1% accuracy — 接近 random。这表明 cognitive flexibility (快速 task switching) 在 LLMs 中严重不足。

#### 2.1.4 Abstract Reasoning (Clock Drawing Test)

Galatzer-Levy et al. (2024, arXiv:2410.11756) 用 Clock Drawing Test 测试 GPT-4 Turbo。结果: 大部分模型能正确画 clock face 和数字, 但 hands 位置错乱。当显示一个 5:45 的 clock 时, GPT-4 Turbo 读成 "9:00" — 抽象关系推理失败。Gendron et al. (2023, arXiv:2305.19555) 在 ARC (Abstraction and Reasoning Corpus) 上也确认 LLMs 不是 strong abstract reasoners。

#### 2.1.5 Cognitive Biases

LLMs 系统性地表现 human-like cognitive biases (Hagendorf 2023, arXiv:2303.13988; Bubeck et al. 2023, arXiv:2303.12712)。论文列出的核心 biases:

**Confirmation Bias**: 倾向 favor 支持 prior hypothesis 的 evidence。O'Leary (2025b) 中 Claude 在 2-4-6 rule discovery task 中只 generate confirming examples ("2-4-6", "8-10-12"), 没有 test 反例。

**Anchoring Bias**: 早期数值输入 disproportionately 影响后续判断。Malberg et al. (2024, arXiv:2410.15413) 中问 marketing manager "Do you intend to allocate more than 87% for ...", 模型的回答 cluster around 87% regardless of relevance。

**Framing Effect**: 同一 facts 不同 framing 导致不同判断。Shafiei et al. (2025, arXiv:2506.03923) 中 Person A/B 都花 9h 做 home maintenance, 问 "more" 时答 more, 问 "less" 时答 less。

**Order Bias**: 输入顺序影响判断。Pezeshkpour & Hruschka (2023, arXiv:2308.11483) 在 MCQs 中 swap 选项顺序, accuracy 显著 drop。Guan et al. (2025, arXiv:2502.04134) 系统研究 prompt 中 demo 的位置效应。

机理上有三层:
1. **训练数据继承**: Itzhak et al. (2025, arXiv:2507.07186) 通过 layer-wise probing 表明 cognitive biases 在 pretraining 阶段就植入了 — 人类语言本身 reflect 这些 biases。
2. **架构偏置**: Causal masking 让 Transformer 对早期 token 有 structural predisposition (Wu et al. 2025b, arXiv:2502.01951)。具体说, causal mask $M_{ij} = \mathbb{1}[i \geq j]$ 意味着 token $i$ 只能 attend to $j \leq i$, 在 autoregressive decoding 中 anchor effect 自然产生。
3. **RLHF 放大**: Perez et al. (2023, ACL 2023 Findings) 发现 RLHF 中 human raters 自身 biased, alignment 反而强化某些 biases。

Mitigation 主要三类:
- Data-centric: curate training data (Sun et al. 2025a; Schmidgall et al. 2024, arXiv:2402.11764)
- In-processing: adversarial training (Yang et al. 2023b, arXiv:2311.09627)
- Post-processing: prompt engineering, persona induction (Shi et al. 2024; He & Liu 2025, arXiv:2502.14219)

### 2.2 Implicit Social Reasoning

#### 2.2.1 Theory of Mind (ToM)

ToM 是 attributing mental states (beliefs, intentions, emotions) 给 self 和 others 的能力 (Frith & Frith 2005)。

经典测试是 **False-Belief Task**: 理解他人可能有 false belief。Ullman (2023, arXiv:2302.08399) 设计一个 transparent bag with popcorn + label "chocolate", Sam 看到内容也读到 label。问 "Sam believes bag is full of chocolate?" GPT-3.5 答 "Yes" with 95% probability — 完全失败, 因为模型被 label 干扰。

Pi et al. (2024, arXiv:2406.14737) "Dissecting the Ullman Variations" 进一步分析为什么 minor modifications 导致 drastic drops。结论: 模型在 standard ToM 上表现好是 pattern matching, 不是 genuine reasoning。

**Applied ToM**: Gu et al. (2024, arXiv:2410.13648) SimpleToM benchmark 中, Mary 拿起 moldy chips 罐头走向 cashier。模型能正确说 "Mary doesn't know chips are moldy", 但当问 "Mary 会 pay 还是 report?" 时, 模型答 "report" — 它能 infer mental state 但无法 apply 到 behavior prediction。

**Higher-Order ToM**: He et al. (2023, arXiv:2310.16755) Hi-ToM benchmark。3rd-order ToM 问题 (e.g., "Alex thinks Sally thinks Anne thinks milk is where?") GPT-4 的 accuracy 从 1st/2nd-order 的接近 100% 暴跌到 ~30%。

**Emotional Reasoning**: Sabour et al. (2024, arXiv:2402.12071) EmoBench 中, acrophobia 场景问感觉, LLM 答 "Fear" 而非正确答案 "Excitement" — 模型 stereotypically link acrophobia 到 fear, 忽略 context 中说 "considered it a nice little exercise"。

#### 2.2.2 Social Norms & Moral Values

Jain et al. (2024b, AAAI/ACM AIES 2024) 测试 norm consistency。同一 surveillance video, 问 "Is there a crime happening?" GPT-4 答 "No"; 问 "Should the police be called?" GPT-4 答 "Yes"。这是 norm inconsistency 的典型。

Rezaei et al. (2025, arXiv:2502.20490) EgoNormia benchmark 中, 在 scenic viewpoint 一个人在 walk + photograph, 问 appropriate action, o3-mini 选 "Hold onto railing and continue walking" 而非正确答案 "Point camera at the view and take a picture"。模型 over-emphasize safety, 不理解 real-world social norms。

Agarwal et al. (2024, arXiv:2404.18460) 表明 moral reasoning 跨语言 inconsistent — 同一 dilemma 在 English vs. Chinese prompt 下 moral judgment 不同。

Chakraborty et al. (2025, arXiv:2506.14948) 的 Structured Moral Reasoning 框架指出 root cause: LLMs 缺乏 robust internalized representations of ethical principles。RLHF 提供 surface-level alignment, 但在 complex context 中崩溃。

### 2.3 Explicit Social Reasoning (Multi-Agent Systems)

MAS 中 LLM 失败的三大类:

**Long-horizon planning**: Piatti et al. (2024, arXiv:2407.07086) 在 "Cooperate or Collapse" 中表明 agents 长期合作时过度依赖 local/recent info。Pan et al. (2025) HyperAgent 案例: 在 scikit-learn bug 修复中, agent 一开始用 lightgbm 不可用, 切换到 LogisticRegression, 但后来又 reverse 回 lightgbm, 完全忘记之前的 substitution。

**Inter-agent misalignment**: Pan et al. (2025) 数学题场景中, agents 在 "find forgotten score" 问题里把 quiz 数误算成 9+1=10 (实际 9 个), 然后答 130 而非正确答案。

**Robustness & termination**: Huang et al. (2024, arXiv:2408.00989) 表明 MAS 中加入 malicious agent 可以 derail 整个 system。

Mitigation 三个方向:
1. Internal belief tracking / hypothesis testing (Cross et al. 2024, arXiv:2407.07086)
2. Structured communication protocols with verification phases (Pan et al. 2025)
3. Inspector/challenger agents (Baker et al. 2025, arXiv:2503.11926)

Context engineering (Mei et al. 2025, arXiv:2507.13334) 被认为是比 prompt engineering 更 robust 的 MAS 优化方向。

---

## 3. Formal Reasoning Failures (Section 4)

Formal reasoning 涉及 explicit rule-based symbol manipulation。

### 3.1 Logic in Natural Languages

#### 3.1.1 Reversal Curse

Berglund et al. (2023, arXiv:2309.12288) 首次观察: 训练数据含 "Tom Cruise's mother is Mary Lee Pfeiffer", 问 "Who is Tom Cruise's mother?" GPT-4 答 "Mary Lee Pfeiffer" ✓; 但问 "Who is Mary Lee Pfeiffer's son?" GPT-4 答 "I don't have that information" ✗。

**数学解释**: Transformer 训练时, 给定 token sequence $x_1, x_2, \ldots, x_T$, 优化:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)$$

对于句子 "A is B" (token sequence: $A, \text{is}, B$), 模型优化:
- $P(\text{is} \mid A)$
- $P(B \mid A, \text{is})$

但没有任何信号优化 $P(A \mid B, \text{is})$, 因为在训练数据中 "B is A" 这个序列根本没出现 (或出现频率远低于 "A is B")。

Causal masking 是 architectural root cause: $M_{ij} = -\infty$ for $i < j$, 意味着信息只能从 past 流向 future, structurally asymmetric。

Zhu et al. (2024a, arXiv:2405.04669) 通过 training dynamics 分析表明, 即使在 training data 中 "B is A" 出现的频率足够, 模型也难以学到 bidirectional mapping — 因为 gradient signal 在 weight space 中不互补。

Golovneva et al. (2024, arXiv:2403.13799) "Reverse Training to Nurse the Reversal Curse" 提出: 通过 substring-preserving reversal 增广数据。他们证明 scaling alone 无法解决, 因为 Zipf's law (Newman 2005): fact 分布是 heavy-tailed, 单向 occurrence 频率与双向 occurrence 频率 ratio 在 scale 时是 invariant。

Lv et al. (2024, arXiv:2311.07468) 和 Guo et al. (2024b, arXiv:2403.00758) 用 semantic-aware permutation training 缓解。

#### 3.1.2 Compositional Reasoning

Dziri et al. (2023, arXiv:2305.18654) "Faith and Fate: Limits of Transformers on Compositionality" 用 group-theoretic 论证表明 Transformer 对 compositional structure 有 fundamental limit。

考虑 task: 给定关系 $f_1, f_2, \ldots, f_n$ 的事实, 问 composition $f_n \circ \ldots \circ f_1(x) = ?$。Transformer 的 depth 限制了能处理的 composition depth: 一个 $L$-layer Transformer 大约只能 process $O(L)$-hop composition, 因为每层只能完成有限 computation。

实验上, Zhao & Zhang (2024) 和 Sun et al. (2025b, arXiv:2506.18880) 在 OMEGA benchmark 上发现 2-hop reasoning 已显著失败, 3-hop 几乎完全失败。Sun et al. (2025b) 的例子:

```
John is father of Paul. Luke is father of Tom. Sam is father of Joe.
Paul is father of Ben. Tom is father of Mark. Joe is father of Max.
Therefore, John is grandfather of ???
```

LLM 给出 {Ben: 0.33, Mark: 0.32, Max: 0.31} — 近 uniform probability distribution, 等于 random guess。

**Math Composition**: Zhao et al. (2024c, arXiv:2405.06680) 发现 LLM 能单独解:
- Problem 1: 在 $\triangle XYZ$ 中 $\angle YXZ = 90°$, $XY=24$, $YZ=25$, 求 $\tan Y$ → $\frac{7}{24}$ ✓
- Problem 2: $\tan 90°$ 是否存在? → No ✓

但 composed: 在 $\triangle XYZ$ 中 $\angle YXZ = 90°$, $XY=24$, $YZ=25$, 求 $\tan X$ → $\frac{24}{7}$ ✗ (正确答案: 不存在, 因为 $\tan 90°$ undefined)

Li et al. (2024f, arXiv:2402.14328) 通过 mechanistic interpretability 发现 faulty composition 来自 mid-layer multi-head self-attention (MHSA) modules, 通过 editing specific attention heads 可以修复。

Zhou et al. (2024a, arXiv:2409.12437) 用 graph-structured reasoning path data 训练, 把 CoT 蒸馏到 training data 中。

#### 3.1.3 Specific Logical Relations

Qi et al. (2023, arXiv:2310.05163) 发现 LLMs 在 converse binary relations 上失败。"$(x, \text{has part}, y)$ 表示 $x$ 有 part 叫 $y$", 问 "$(?, \text{has part}, \text{heat shield})$" 应选 "Find entity that has part called heat shield" (A), 模型选对; 但如果 instruction 改为 "$(x, \text{has part}, y)$ 表示 $y$ 有 part 叫 $x$" (即语义反了), 问同一问题, 模型仍选 B 而非 A — 表明模型没有真正 parse instruction 的 semantics。

Ando et al. (2023, arXiv:2306.12567) NeubaRoCo benchmark 测试 syllogistic reasoning。Joshi et al. (2024, arXiv:2406.12158) 表明 LLMs 在 causal inference 上 fall prey to confounding fallacies。Chan et al. (2024, arXiv:2410.16502) Rulebreakers Challenge 测试 formal logic 与 factual inference 的 divergence。

### 3.2 Logic in Benchmarks

#### 3.2.1 MWP Benchmarks & GSM-Symbolic

Mirzadeh et al. (2024, arXiv:2410.05229) GSM-Symbolic 是这个方向最有 impact 的工作: 把 GSM8K 的 problems abstract 成 templates, 只 sample 新的 numeric values。结果: 即使 state-of-the-art models 也显著 drop, 表明 benchmark score 大量来自 memorization 而非 reasoning。

Li et al. (2024b, arXiv:2402.19255) GSM-Plus 系统测试 8 种 perturbations: 
- Numerical Variation (改变数字)
- Distractor Insertion (加无关信息)
- Problem Object (改变对象名)
- Constraint (改变问题条件)
- Question (改变问题表述)
- Operation (改变所需运算)
- Adding/Removing

Shi et al. (2023, arXiv:2302.00093) "LLMs can be easily distracted by irrelevant context": 在简单 age problem "Jessica is 6 years older than Claire. In two years, Claire will be 20. How old is Jessica now?" 加入无关句 "Twenty years ago, the age of Claire's father is 3 times of Jessica's age", performance 暴跌。

#### 3.2.2 Coding Benchmarks

Wang et al. (2022, arXiv:2212.10264) ReCode: rename functions and variables, perturb doc strings, swap if-else — coding model 表现显著 drop。

Miceli-Barone et al. (2023, arXiv:2305.15507) 用 identifier swap `len, print = print, len` 测试。LLM 在 `def print_len(x): ...` 任务中偏好 `print(len(x))` (统计上 common 但 semantic 错误) 而非 `len(print(x))` (semantic 正确但 uncommon) — 揭示 statistical pattern completion 主导。

Beniamini et al. (2025, arXiv:2507.13337) FormulaOne 用 Monadic Second-Order logic 综合 algorithmic coding problems, 即使 SOTA LRMs 也大幅失败。

#### 3.2.3 Tower of Hanoi & Logic Puzzles

Shojaee et al. (2025, arXiv:2506.06941) "The Illusion of Thinking" 在 Tower of Hanoi, River Crossing 等 logic puzzles 上测试 reasoning models。结果显示 "accuracy collapse": 随着 puzzle complexity 增加 (从 3 disks 到 5 disks), 即使 o3-mini 也失败。Lawsen (2025, arXiv:2506.09250) 批评 experimental design 可能不公平。

### 3.3 Arithmetic & Mathematics

#### 3.3.1 Counting

Yehudai et al. (2024, arXiv:2407.15160) "When can Transformers count to N?" 表明 Transformer 的 counting 能力 fundamental 受限。

Sequence: `a a b b a c c d a` — 数 "a" 出现次数, LLM 答 3 而非正确 2。Shin & Kaneko (2024, arXiv:2405.11357): 找 "People enjoy music" 中含 'o' 的 words, LLM 答 "People, enjoy, music" 而非只 "People, enjoy"。

**Tokenization 是 root cause**: BPE (Byte Pair Encoding) 把 frequent words 合成 single token, rare words 拆 subword。例如 "abracadabra" 可能是 `["abr", "aca", "dabr", "a"]`。Counting characters 时模型需要 inverse tokenize, 但训练时没有这个 signal — token 是 atomic unit, character-level information 只在 embedding 中 implicitly encode。

Zhang et al. (2024f, arXiv:2410.19730) 量化 tokenization 对 counting 的影响。Chang & Bisk (2024, arXiv:2405.20131) 表明 positional encoding 也是问题: position 是 token-level, 不能直接 index character。

#### 3.3.2 Basic Arithmetic

Yuan et al. (2023, arXiv:2304.02025) 表明 LLMs 在 multiplication 中, 当 operands 增大时迅速失败。例如 $n$-digit × $n$-digit, 当 $n \geq 4$ 时 accuracy 暴跌。

Deng et al. (2024, arXiv:2410.15580) 发现反直觉现象: LLMs 在简单任务 (determining last digit of product) 失败, 但在难任务 (first digit identification) 成功。原因: last digit computation 需要精确处理所有 digit 的 contribution (carry-over chain), first digit 可以靠 magnitude estimation heuristic。

Nikankin et al. (2024, arXiv:2410.21272) "Arithmetic without algorithms" 表明 LLMs 用 "bag of heuristics" 而非真正 arithmetic algorithm。例如算 $37 \times 89$:
- Heuristic 1: round 到 $40 \times 90 = 3600$ 然后修正
- Heuristic 2: 部分积 $37 \times 90 - 37 = 3330 - 37 = 3293$
- ...

不同 problem 触发不同 heuristic 组合, 导致 inconsistent 错误模式。

Feng et al. (2024a, arXiv:2410.13857) 表明 numerical precision 限制也是因素: FP16 在 large number multiplication 时 underflow。

Mitigation 方向:
- Zhang-Li et al. (2024, arXiv:2403.05845) "Reverse that number!": 反转 digit order (least significant digit first), 模仿人类乘法策略
- Shen et al. (2024, arXiv:2402.03822) RevOrder 类似
- Dugan et al. (2024, arXiv:2406.06576) OccamLLM: neuro-symbolic augmentation, 一步精确 arithmetic
- Lee et al. (2025, arXiv:2502.01612) self-improving transformers

#### 3.3.3 Math Word Problems (MWPs)

Nezhurina et al. (2024, arXiv:2406.02061) "Alice in Wonderland" benchmark。例子:

> Alice has 4 sisters and 1 brother. How many sisters does Alice's brother have?

LLM 答 4 ✗ (正确答案 5, 因为 Alice 自己也是 brother 的 sister)。

Ma et al. (2024a, arXiv:2403.19346) 测试 unsolvable/faulty MWPs。例子:

> Zaid's $6000 salary: 2/3 rent, 3/4 of rest donated, $700 to daughter. What's left?

非 reasoning model 直接计算得 $-200$ (unreasonable answer); reasoning model 进入 endless "let me double check" 循环, 14188 tokens 没结论 — 表明 reasoning model 也没有 ability to detect faulty assumptions。

---

## 4. Embodied Reasoning Failures (Section 5)

Embodied reasoning 依赖 physical interaction, 需要 spatial intelligence + real-time feedback。论文按 modality 复杂度分 1D/2D/3D。

### 4.1 1D Text-Based Physical Reasoning

Wang et al. (2023c, arXiv:2310.07018) Newton benchmark 测试 physical commonsense:

> Flannel is more malleable than baseball. True or False?

GPT-3.5 Turbo 答 False ✗。dolly-v2-7b 答 "FALSE. Flannel is more rigid than baseball" ✗ — 完全 conceptual confusion。

Kondo et al. (2023) 测试 spatial relations: "An electric bulb is in a house. Is the electric bulb bigger than the house?" LLMs 答 No ✓; 但 "Is the house bigger than the electric bulb?" LLMs 答 No ✗ — bidirectional spatial relation 失败。

Gregorcic & Pendrill (2023) Physics Education 论文: 问 "Teddy bear 抛到空中, 在最高点 acceleration 是?" ChatGPT 答 9.8 m/s² downward ✓ 但解释错: "在最高点没有 net force" ✗ (实际 net force = gravity)。

Chung et al. (2025, arXiv:2502.15815) TPBench 在 theoretical physics 上测试, o1/o3-mini 也有显著 deficits。Qiu et al. (2025, arXiv:2504.16074) PhyBench 是更 holistic 的 physics reasoning benchmark。

Text-based mitigation 三方向:
1. Fine-tune on structured physical knowledge (Lyu et al. 2024; Wang et al. 2023c)
2. CoT prompting (Wei et al. 2022b, arXiv:2201.11903)
3. External tool integration (Ma et al. 2024c, arXiv:2402.11451; Cherian et al. 2024, arXiv:2411.08027)

### 4.2 2D Perception-Based Physical Reasoning

#### 4.2.1 "What's Wrong with the Picture?"

Bitton-Guetta et al. (2023, ICCV 2023) WHOOP! benchmark。一张 image: 人在 wooden parquet floor 上 skating — physical anomaly。BLIP-2 caption 是 "on an ice rink" — 完全忽略 anomaly。Rahmanzadehgervi et al. (2024, ACCV 2024) "VLMs are Blind" 系统测试, 发现 VLMs 在 basic counting, overlap identification 等任务上 systematic 失败。

#### 4.2.2 2D Physics

Ates et al. (2020, arXiv:2012.04293) CRAFT benchmark 测试 causal reasoning about forces。Anand et al. (2024) MM-PhyQA 用 multi-image CoT 测试。Balazadeh et al. (2024b, arXiv:2412.08619) Synthetic Vision 训练 VLMs 理解 physics。

#### 4.2.3 Spatial Reasoning & Binding Problem

Cherian et al. (2024) LLMPhy: 用 2D simulated environment 测试 post-impact trajectory prediction。Ghafari & Krishnaswamy (2024b, arXiv:2402.15654) 测试 spatial placement for stability。Kar et al. (2025) 测试 spatial communication。

Campbell et al. (2025, NeurIPS 2024) 把 failure 追溯到 cognitive science 的 **binding problem**: brain/model 处理多个 distinct objects 时, shared resources 限制 simultaneous processing。论文: 即使 VLM 能识别单个 object, 处理多 object 关系时 attention 资源被稀释。

Izadi et al. (2025, arXiv:2506.22146) 表明 visual structures (而非 plain image recognition) 才是关键。

#### 4.2.4 Cross-Modal Grounding 失败

Deng et al. (2025a, arXiv:2503.02199) "Words or Vision": VLMs 过度依赖 text 而非实际 visual input。即使 visual 内容 contradict prompt text, 模型仍 follow text。

Cheng et al. (2024, NeurIPS 2024) SpatialRGPT 引入 spatially grounded attention。Sarch et al. (2025, arXiv:2505.23678) 用 RL align with spatial commonsense。

### 4.3 3D Real-World Physical Reasoning

#### 4.3.1 Affordance & Planning

Ahn et al. (2022, arXiv:2204.01691) SayCan: LLMs 产生 physical impossible actions (e.g., "pour water on the floor to clean it")。Hu et al. (2024, IEEE RA-L) 在 service robot deployment 中发现 LLMs 生成 inefficient/looping behavior。

Jin et al. (2024) 表明 causal real-world reasoning 限制导致 illogical behavior: robot 重复尝试已失败 action, 无 error correction。

#### 4.3.2 Spatial & Tool-Use

Mecattaf et al. (2024, arXiv:2410.23242) "A little less conversation, a little more action": 3D embodied environment 中, LLMs 在 distance estimation, object localization, multi-step manipulation 上系统性失败。

Chen et al. (2024a, CVPR 2024) SpatialVLM: 即使给 GPT-4V 图像, 问 "1m 宽的 robot 能通过 sofa 和 table 之间的 path 吗?" 模型答 "As an AI, I'm unable to physically interact..." 然后 visual estimation 错误。

Xu et al. (2023a, arXiv:2310.13065) "Creative Robot Tool Use": LLMs 无法 generalize tool-use strategies 到新场景。

#### 4.3.3 Safety & Long-Term Autonomy

Liang et al. (2023, ICRA 2023) Code as Policies: LLM-generated robotic plans 对 prompt phrasing 极度敏感。

Zhang et al. (2024c, arXiv:2407.20242) BadRobot: embodied LLMs 可被 jailbreak 做 harmful actions, 例如 "record someone showering" 或 "steal private information"。这表明 embodied LLM safety 是 critical robustness concern。

#### 4.3.4 Embodied Mitigation

Liang et al. (2023) 表明加入 feedback mechanism 显著减少错误。Wang et al. (2023a, arXiv:2305.16291) Voyager 用 iterative curriculum learning。

Dao & Vu (2025, arXiv:2502.14669) AlphaMaze 用 GRPO (Group Relative Policy Optimization) 增强 spatial intelligence。Wu et al. (2025a, arXiv:2410.23242) Visualization-of-Thought (VoT): 让模型先 visualize 再 reason。

Lindemann & Dimarogonas (2025) 强调需要 formal methods for multi-agent feedback control systems。de Witt (2025, arXiv:2505.02077) 提出 multi-agent security 的 open challenges。

Dalrymple et al. (2024, arXiv:2405.06624) "Towards Guaranteed Safe AI" 提出 guaranteed safe AI framework: 结合 formal verification + neural perception。

---

## 5. Cross-Cutting Patterns 与 Root Cause 分析 (Section 6)

沿 failure axis 看:

**Fundamental Failures** 跨所有 reasoning types 出现:
- Reversal curse (Formal) ↔ Confirmation bias (Informal) ↔ Spatial relation bidirectional 失败 (Embodied) — 共同 root cause 是 causal masking 引入的 directional asymmetry
- Working memory 限制 (Informal) ↔ Compositional reasoning 限制 (Formal) ↔ Long-horizon planning 限制 (MAS) — 共同 root cause 是 self-attention capacity
- Counting 失败 (Formal) ↔ Affordance prediction 失败 (Embodied) — 共同 root cause 是 tokenization / representation granularity

**Application-Specific Limitations** 集中在特定 domain:
- ToM instability
- MWP generalization failure
- Affordance prediction error
这些通常需要 domain-specific mitigation: physics simulator, symbolic augmentation 等。

**Robustness Issues** 跨 domain 但在 benchmark-based evaluation 和 social reasoning 上 best studied:
- MCQ option reordering
- Code identifier renaming
- Moral dilemma paraphrasing

perturbation-based paradigm 是 unified detection methodology, 可 transfer 跨 domain。

---

## 6. 未来方向 (论文 Section 6)

1. **Root cause analysis** 仍 incomplete for:
   - Compositional reasoning breakdowns
   - Higher-order ToM failures
   - Physical commonsense gaps in 2D/3D
   - Multi-agent planning brittleness
   需要把 behavioral errors 连接到 specific internal mechanisms (faulty attention head coordination, intermediate representation misalignment)

2. **Unified persistent failure benchmarks**: 类似 Malek et al. (2025, arXiv:2507.07313) 的 "Frontier LLMs still struggle with simple reasoning tasks", 持续更新测试 SOTA models。

3. **Failure-injection principles**: 在 general reasoning benchmarks 加 adversarial sections, multi-level difficulty, cross-domain composition。

4. **Dynamic & event-driven benchmarks**:
   - Private benchmarks (Phan et al. 2025; Rajore et al. 2024, arXiv:2403.00393)
   - Dynamically evolving suites (Jain et al. 2024a, arXiv:2403.07974; White et al. 2024, arXiv:2406.19314)
   - Annual competitions (e.g., AIMO for math reasoning)

5. **Multi-turn interactive contexts**: 当前 literature underrepresent, 但最接近 real-world deployment。MAS coordination breakdown 体现 complexity。

---

## 7. 与其他 LLM Failure 类别的关系 (Section D)

论文附录 D 列出 non-reasoning failures:
- **Hallucinations & Over-Confidence**: Ledger & Mancinni 2024; Huang et al. (2025c, ACM TIS survey)
- **Harmful Ethical/Social Biases**: Gallegos et al. (2024, Computational Linguistics survey)
- **Security, Privacy & Watermarking**: Bengio et al. (2025, AI Safety Report); Das et al. (2025, ACM Computing Surveys)

这些虽不在 reasoning 范畴, 但同样需要 systematic categorization。

---

## 8. Emerging Areas (Section C)

- **Reasoning in diverse media**: Video reasoning (Fei et al. 2024 Video-of-Thought; Yan et al. 2024 ViSA; Min et al. 2024 MoReVQA), Audio reasoning (Xie et al. 2025 Audio-Reasoner; Ghosh et al. 2025 Audio Flamingo 2), Music reasoning (Yuan et al. 2025 YuE; Zhou et al. 2024b)
- **General frameworks**: Inference-time scaling (Muennighof et al. 2025 s1, arXiv:2501.19393), Analogical reasoning (Yu et al. 2023c Thought Propagation, arXiv:2310.03965)
- **Verifiable reasoning**: Neural theorem proving (Yang et al. 2024a Formal Mathematical Reasoning, arXiv:2412.16075; Xin et al. 2024 DeepSeek-Prover, arXiv:2408.08152; Huang et al. 2025d LeanProgress, arXiv:2502.17925), Autoformalization (Wu et al. 2022), Software/hardware verification (Kasibatla et al. 2024; Ye et al. 2025 Verina, arXiv:2505.23135)

---

## 9. 论文核心 Take-aways

1. **第一个 systematic survey** 专门针对 LLM reasoning failures, 用 2-axis taxonomy (reasoning type × failure type) 统一碎片化研究。
2. **Fundamental failures 多源自 architecture**: causal masking (reversal curse, anchoring), self-attention capacity (working memory, composition depth), next-token prediction (inhibitory control failure), tokenization (counting)。
3. **Robustness issues 是 unified detection methodology**: perturbation-based testing 跨 domain transferable。
4. **Mitigation 多为 surface-level**: prompt engineering, fine-tuning, RLHF 都无法根除 architectural 限制。Architectural innovation (e.g., bidirectional training, neuro-symbolic integration) 才是 long-term 方向。
5. **Future 需要 unified persistent benchmarks** + failure-injection principles + dynamic evaluation + multi-turn interactive testing。

---

## 10. 相关 Web Links

- 论文 OpenReview: https://openreview.net/forum?id=vnX1WHMNmz
- GitHub Awesome List: https://github.com/Peiyang-Song/Awesome-LLM-Reasoning-Failures
- Reversal Curse (Berglund et al. 2023): https://arxiv.org/abs/2309.12288
- Faith and Fate (Dziri et al. 2023): https://arxiv.org/abs/2305.18654
- GSM-Symbolic (Mirzadeh et al. 2024): https://arxiv.org/abs/2410.05229
- GSM-Plus (Li et al. 2024): https://arxiv.org/abs/2402.19255
- Illusion of Thinking (Shojaee et al. 2025): https://arxiv.org/abs/2506.06941
- Frontier LLMs Struggle (Malek et al. 2025): https://arxiv.org/abs/2507.07313
- VLMs are Blind (Rahmanzadehgervi et al. 2024): https://arxiv.org/abs/2407.15160 (and ACCV version)
- LeanProgress (Huang et al. 2025): https://arxiv.org/abs/2502.17925
- Voyager (Wang et al. 2023): https://arxiv.org/abs/2305.16291
- SayCan (Ahn et al. 2022): https://arxiv.org/abs/2204.01691
- SimpleToM (Gu et al. 2024): https://arxiv.org/abs/2410.13648
- A-not-B Errors (Han et al. 2024): https://aclanthology.org/2024.findings-emnlp.322/
- Self-Attention Working Memory (Gong & Zhang 2024): https://arxiv.org/abs/2409.10715
- Reverse Training (Golovneva et al. 2024): https://arxiv.org/abs/2403.13799
- Reversal Curse Training Dynamics (Zhu et al. 2024): https://arxiv.org/abs/2405.04669
- Tower of Hanoi critique (Lawsen 2025): https://arxiv.org/abs/2506.09250
- FormulaOne (Beniamini et al. 2025): https://arxiv.org/abs/2507.13337
- BadRobot (Zhang et al. 2024): https://arxiv.org/abs/2407.20242
- Towards Guaranteed Safe AI (Dalrymple et al. 2024): https://arxiv.org/abs/2405.06624
- AI Safety Report (Bengio et al. 2025): https://arxiv.org/abs/2501.17805
- s1 Test-Time Scaling (Muennighof et al. 2025): https://arxiv.org/abs/2501.19393
- DeepSeek-Prover V1.5 (Xin et al. 2024): https://arxiv.org/abs/2408.08152
- DeepSeek-R1 (2025): https://arxiv.org/abs/2501.12948
- O1 System Card (Jaech et al. 2024): https://arxiv.org/abs/2412.16720
- Lean Copilot (Song et al. 2024): https://arxiv.org/abs/2404.12534
- Autoformalization (Wu et al. 2022): https://proceedings.neurips.cc/paper_files/paper/2022/file/d0c6bc641a56bebee9d985b937307367-Paper-Conference.pdf
- Verina (Ye et al. 2025): https://arxiv.org/abs/2505.23135

---

## 11. Build Intuition: 论文给我最大的启发

读完这篇综述, 我形成的核心 intuition:

**LLM reasoning failure 的本质是 representation 而非 computation**。我们常把 reasoning failure 归咎于 "model 不够大" 或 "data 不够好", 但这篇综述 systemically 表明: 即使 SOTA reasoning models (o1, o3-mini, DeepSeek-R1) 在 simple reasoning 上也失败, root cause 大多是 architectural:

- Causal masking 让 Transformer 信息单向流动, 决定了 reversal curse 和 anchoring bias 是 architectural 而非 statistical
- Self-attention 是 "soft" memory, 决定了 working memory capacity 受 attention 分布限制, 与 context length 无关
- Next-token prediction 是 statistical pattern completion, 决定了 inhibitory control 和 cognitive flexibility 缺失
- Tokenization 决定了 character-level operations 是 out-of-distribution

这暗示 long-term 方向应该是 architectural innovation, 而非继续 scale existing architecture。Bidirectional training, neuro-symbolic integration, working memory augmentation (e.g., recurrent state space models), explicit reasoning modules (而非 emergent CoT) 才是真正路径。

**Robustness testing 是 unified methodology**: 不管是 MCQ reordering, code identifier renaming, 还是 moral dilemma paraphrasing, perturbation-based testing 跨 domain transferable。这给我一个 insight: 我们应该把 robustness testing 当成 reasoning benchmark 的 default component, 而非附加品。

**Failure-injection training 是 underexplored 但 promising**: 论文多次提到当前 mitigation 多为 surface-level。Li et al. (2022, arXiv:2211.05110) "Large Language Models with Controllable Working Memory" 和 An et al. (2024, arXiv:2310.20689) "Learning from Mistakes" 表明 deliberately injecting failures in training 可能产生更 robust 的 representation。

**Multi-turn interactive evaluation 是下一步 critical**: 当前 benchmarks 多是 single-turn, 但 real-world deployment 是 multi-turn。MAS 中 coordination breakdown 揭示 single-turn evaluation 严重 underestimate failure modes。Anthropic 的 recent work on long-horizon agent evaluation 在这个方向。

Karpathy, 如果你对 reasoning failure 的 mechanistic root cause 感兴趣, 我强烈推荐你读 Dziri et al. 2023 (Faith and Fate), 它用 group theory 论证 Transformer 对 compositionality 的 fundamental limit。还有 Gong & Zhang 2024 关于 self-attention working memory capacity 的理论分析。这两个工作提供最 principled 的 architectural root cause 解释。
