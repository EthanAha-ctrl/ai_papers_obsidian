---
source_pdf: Training Large LanguageModelstoReasonina ContinuousLatentSpace.pdf
paper_sha256: 6eb32c71eaa571b727ee95bce20396243fd70dc1ee8a37f91a7de717790d5b46
processed_at: '2026-08-12T17:53:29-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Coconut 用人话讲

好，咱们抛开公式，用最朴素的方式聊聊这篇 paper 到底干了啥、为什么这么干、为什么它有意思。

---

## 1. 一句话总结

**让模型在 hidden state 里想问题，想完再开口。**

现在的 LLM 是"边说边想"——每一步推理都要吐出一个 word token，下一步才能基于这个 token 继续。Coconut 把这一步砍了：模型闷头想完，hidden state 直接喂回自己当下一步的 input，最后再 decode 成自然语言。

---

## 2. 为什么觉得语言不是 reasoning 的好载体

你 Karpathy 一直在 twitter 上讲 "System 2 / latent reasoning / think before speaking" (https://x.com/karpathy/status/1748043575011350709)。这篇 paper 就是把这个直觉拿来做了一个最小实现。

几个观察叠起来：

**观察一**：人在心算的时候脑子里转的不是英语句子。你想 23 × 17，不会先在脑子里默念 "twenty-three times seventeen equals..."，你脑子里是某种数学结构。神经科学也支持——Fedorenko 2024 那篇 Nature (https://www.nature.com/articles/s41586-024-07547-w) 说语言网络在 reasoning 任务里基本不亮。所以"语言是思考工具"这个假设本身就有问题。

**观察二**：CoT 里大部分 token 是凑数的。"Therefore"、"So we have"、"Thus" 这些 token 几乎不承担计算，纯粹是为了让句子读得通。模型却给每个 token 都分配一样的 FLOPs。这是明显的算力浪费。

**观察三**：一旦模型在 CoT 里说错了"Alex 是 lempus"，它就回不去了。因为下一个 token 只能基于"Alex 是 lempus"往下生成。autoregressive 是单向的，没有 backtracking。人类做数学做错了会划掉重写，LLM 不行。

Coconut 的出发点：**把语言这个 bottleneck 去掉，让 reasoning 在连续空间里发生**。

---

## 3. 实现层面就改了一件事

如果你自己手写过 nanoGPT (https://github.com/karpathy/nanoGPT)，你应该对 forward pass 很熟：

```
token ids → embedding lookup → transformer blocks → hidden state h
→ lm_head (linear) → softmax → vocab logits → argmax → next token
→ embedding lookup → 喂回去
```

Coconut 做的事情：把中间那个 "lm_head → softmax → argmax → embedding lookup" 的 round trip 删掉，直接 `hidden state h → 喂回去`。

就这一步。代码改动量可能就几十行。

两个 special token `<bot>` 和 `<eot>` 标记 latent mode 的开始和结束。在 `<bot>` 和 `<eot>` 之间，每一步的 input embedding 就是上一步的 last hidden state，不走 vocab。

**为什么这件事重要**：
- 没有 argmax，没有 sampling，所以全程 differentiable，梯度可以从最后的 loss 一路 backprop 到所有 latent thought
- hidden state 维度 $d$ 通常几千，比 vocab size $|V|$（几万到几十万）小得多，信息密度高
- 不会被"语言里有哪些词"这个离散集合限制

这本质上就是把 transformer 当成 unrolled RNN 用——每个 latent step 是一次 transition，但 transition function 是一个 pre-trained transformer block，而不是一个普通 RNN cell。这种"depth"是不受 vocab projection 限制的。

---

## 4. 为什么不能直接训

直接让模型从零开始在 latent space 里 reason，学不会。Table 1 里 "Coconut w/o curriculum" 在 GSM8k 上只有 14.4%，比 no-CoT 的 16.5 还低。

原因也好理解：latent space 太大、太自由了，模型不知道该用哪些 dimension 编码什么。没有 inductive bias 的话，gradient descent 找不到 useful manifold。

所以作者借用了 Deng et al. 2024 的 iCoT (https://arxiv.org/abs/2405.14838) 的 multi-stage curriculum idea，但做了一个关键改动：**iCoT 是逐步删掉 language reasoning step，Coconut 是把删掉的 step 换成 continuous thoughts**。

打个比方：教小孩解方程。
- 一开始让小孩把每一步都写出来（标准 CoT）
- 第二阶段：第一步在脑子里想，后面写出来
- 第三阶段：前两步在脑子里想，后面写出来
- ...最后全在脑子里想，只写答案

每一步都有 language supervision 引导，模型慢慢学会"这一步本来该想什么"在 latent space 里怎么编码。

具体 schedule：超参 $c$ 控制每个 language reasoning step 被替换成几个 continuous thoughts。Stage $k$ 把前 $k$ 个 reasoning step 替换成 $k \times c$ 个 latent thoughts，loss 只在剩下的 language part 上算。stage 之间 reset optimizer state，因为 loss landscape 变化大，momentum 会拖累。

---

## 5. 为什么造一个新 dataset

ProntoQA (https://arxiv.org/abs/2210.01240) 是现有的 logical reasoning benchmark，但作者觉得太简单——路径基本是线性的，模型不需要 search 就能走通。

于是造了 ProsQA。每个 problem 对应一个 DAG，节点是 entity 或 concept，边是"Every X is a Y"这种蕴含关系。Question 形如"Is Alex a gorpus or bompus?"，DAG 保证 Alex 到 bompus 有路径、到 gorpus 没有，但 DAG 有多个分支，模型必须**搜索**才能找到正确路径。

Appendix Algorithm 1 给了构造 DAG 的 pseudocode。关键设计是用 Poisson(1.5) 采样每个 node 的 in-degree，让图有多个 parent，强迫模型 search。而且 sampling weight 偏向深节点（$depth\_to\_root(c) \cdot 1.5 + 1$），让 reasoning chain 较长。

平均 23 个 node、36 条 edge、最短路径 3.8、平均 1.6 条最短路径。这个"1.6 条最短路径"很关键——说明问题本身有多解，需要 planning 来选。

---

## 6. 最有意思的发现：BFS 自己涌现出来了

这是整篇 paper 最 surprising 的部分。

### 6.1 实验怎么做

拿 Figure 4 那个例子：问 Alex 是 gorpus 还是 bompus。Alex 的 children 有 4 个：lempus、sterpus、zhorpus、grimpus。

作者手动控制 $k$（latent thought 数量），让模型做 $k$ 个 latent thought 之后强制回到 language mode，看它第一个输出的 concept 是什么、概率分布怎么样。

定义一个 "value function"：某个 concept 的所有 token 条件概率的乘积。可以理解成"模型隐式觉得这个 node 能通向正确答案的可能性"。

### 6.2 关键观察

第一个 latent thought 之后，value 最高的 concept 是 **lempus**（0.33）。
第二个 latent thought 之后，value 最高的 concept 变成了 **rorpus**（0.87），而 rorpus 是 grimpus 的 child。

如果是 greedy search，模型在第一步就该 commit 到 lempus 往下走。但实际它没有——它在第二个 thought 里跳到了 grimpus 的 subtree，因为 grimpus 实际通向正确答案 bompus。

这就是 **BFS 的 signature**：第一步维持多个 candidate 的 probability mass，第二步通过评估它们的 children 来 disambiguate。它没有立即 commit 到局部最优。

### 6.3 为什么 latent space 能做这个

Section 4.4 给了 hypothesis：**node 离 target 越远，越难准确评估它**。

Figure 7 验证了。横轴是 node 的 height（到 leaf 的最短距离），纵轴是 model 给的 value。正确 node 的 value 应该高、错误 node 的 value 应该低。

- Height 低（接近 target）：correct 和 incorrect 明显分开
- Height 高（远离 target）：correct 和 incorrect 混在一起，分不清

这个现象的解释：离 target 近的 node，可走的 path 少，evaluation 容易；离 target 远的 node，subtree 大，evaluation 困难。

所以 language CoT 的失败模式就是：在 height 高的地方被迫 commit 到一个 branch，错了就回不去。
Coconut 的成功模式：在 height 高的地方维持多个 branch 的 superposition（因为 hidden state 是连续的，可以同时编码多个 hypothesis），等"展开"到 height 低的地方再做决定。

这点跟 Zhu et al. 2025b 的 theory (https://arxiv.org/abs/2505.12514) 完全对上——他们证明 continuous CoT 可以通过 **superposition state** 编码 multiple reasoning paths，这是 discrete CoT 在数学上做不到的。后续的 training dynamics 分析 (https://arxiv.org/abs/2509.23365) 进一步解释了 superposition 是怎么从 gradient descent 里 emerge 的。

---

## 7. 讲个具体故事（Figure 4 case study）

题目：判断 Alex 是 gorpus 还是 bompus。Ground truth：Alex → grimpus → rorpus → bompus。

**CoT 的失败**：模型生成"Alex is a lempus. Every lempus is a scrompus..."，然后卡住了——它走到了 lempus 这个死胡同。为了硬编下去，它 hallucinate 出"Every yumpus is a rempus"这条不存在的 edge，最后错误地回答"Alex is a gorpus"。

这就是 language CoT 的典型失败：commit 错了就回不去，只能 hallucinate 来圆。

**Coconut (k=1)**：一个 latent thought 之后回到 language mode，输出"...是 bompus"，但路径不完全正确——它跳过了中间步骤。被分类为"Correct Label"但不是"Correct Path"。

**Coconut (k=2)**：两个 latent thought 之后回到 language mode，输出完整的正确路径"Alex → grimpus → rorpus → bompus"。

为什么 k=2 能 work：第一个 thought 保持了多个 candidate（lempus/grimpus/...），第二个 thought 通过 evaluate 它们的 children 把 grimpus 这条 path 突出出来了。

---

## 8. 实验结果一览

Table 1 关键数字：

| Method | GSM8k Acc | ProntoQA Acc | ProsQA Acc |
|--------|-----------|--------------|------------|
| CoT | 42.9 | 98.8 | 77.5 |
| No-CoT | 16.5 | 93.8 | 76.7 |
| iCoT | 30.0 | 99.8 | 98.2 |
| Pause Token | 16.4 | 77.7 | 75.9 |
| **Coconut** | **34.1** | **99.8** | **97.0** |
| - w/o curriculum | 14.4 | 52.4 | 76.1 |

几个关键 takeaway：

**Takeaway 1：Chaining 真的带来额外算力**。Coconut (34.1) > Coconut w/o thought (21.6) > Coconut pause as thought (24.1) > No-CoT (16.5)。Pause token (Goyal et al. 2023, https://arxiv.org/abs/2310.02226) 是离散的 learnable embedding，thought 之间没有信息流；Coconut 的 $h_t$ 依赖 $h_{t-1}$，是真正的 recurrent process，所以更强。

**Takeaway 2：Curriculum 不可或缺**。w/o curriculum 在 GSM8k 只有 14.4%，比 no-CoT 还差。latent space 太大，没有 language guidance 找不到 useful manifold。

**Takeaway 3：ProsQA 上 Coconut 接近 iCoT**。97.0 vs 98.2，但 token 数 14.2 vs 8.2。Coconut 用了更多 token 但换来更完整的 reasoning process。

**Takeaway 4：GSM8k 上 Coconut > iCoT**。34.1 vs 30.0。在需要更多 chaining 的 math 上，continuous thought 的 expressivity 优势体现出来。但还没追上标准 CoT 的 42.9——因为 math reasoning 的 language supervision 本身就很强。

**Takeaway 5：Clock time 优势明显**。Table 4：ProsQA 上 Coconut 0.15s vs CoT 0.47s，3 倍加速。ProntoQA 上 Coconut 0.11s vs CoT 0.85s，接近 8 倍加速。CoT 要 autoregressive 生成几十个 token，Coconut 只生成几个 latent thought 加少量 output。

**Takeaway 6：Larger model 上 gain 缩小**。Table 5：Llama 3.2-3B 上 Coconut 比 no-CoT 高 5.7 个点，Llama 3-8B 上只高 1.4 个点。作者的猜测：大模型 language pretraining 更深，转 latent 更难。这个解释有道理但我觉得也可能是 ceiling effect——8B 的 no-CoT 已经 42.2，离 GPT-2 base 的 16.5 远了，提升空间小。

---

## 9. 一个能 build intuition 的 probing 结果（Figure 9）

GSM8k 题目：James 每周跑 3 次，每次 3 个 sprints，每个 sprint 60 米，一周跑多少米？

作者 decode 了第一个 continuous thought 对应的 language tokens，发现它对应 "3 × 3 = 9" 这种 intermediate variable。

这说明 latent thought **不是在压缩 language CoT**，而是在学一个更 efficient 的 representation。Training loss 只要求 final answer 对，不要求 latent thought 能 decode 回 natural language。所以模型有自由去学任何对 prediction 有帮助的 representation。这跟 Feng et al. 2023 (https://arxiv.org/abs/2310.01460) 证明的 "CoT 增加 effective depth" 的理论一致——continuous thought 同样增加 depth，但不受 vocab 限制。

---

## 10. 为什么 Pause Token 不行

Pause Token (Goyal et al. 2023, https://arxiv.org/abs/2310.02226) 的 idea：在 sequence 里插入 learnable `<pause>` token，给模型更多 compute。

但 Pause Token 在 ProntoQA 上 collapse 到 77.7%（Table 1）。Pfau et al. 2024 (https://arxiv.org/abs/2404.15758) 的理论解释了：filler token 只能 extend expressivity 到 parallelizable problems，对需要 serial reasoning 的 task 不行。

原因是 Pause Token 之间是**独立的** learnable embedding，没有 information flow between them。你可以在 sequence 任何位置插 pause，但它们不会"互相影响"。

Coconut 不一样：$h_t$ 是 $h_{t-1}$ 通过 transformer block 的输出，每一步都在前一步基础上做计算。这是真正的 recurrent process，所以能 encode serial reasoning。这也是为什么 Coconut 在 math reasoning 上比 Pause Token 强不少。

---

## 11. 跟相关工作的关系网

**iCoT (Deng et al. 2024)**: https://arxiv.org/abs/2405.14838
Coconut 的 curriculum 直接 borrow 自这里。区别：iCoT 删 language step 不加东西，Coconut 删了换成 continuous thought。所以 Coconut 多了 "chaining" 带来的 expressivity gain。

**Quiet-STaR (Zelikman et al. 2024)**: https://arxiv.org/abs/2403.09629
在 token level 做 "think before speak"——在 critical token 前插入 thought token。仍然在 language space，但有 self-training 的 RL flavor。可以跟 Coconut 结合：在 latent space 做 Quiet-STaR。

**Tree of Thoughts (Yao et al. 2023)**: https://arxiv.org/abs/2305.10601
Explicit tree search with LLM as heuristic。Coconut 是 implicit BFS emerge from latent space geometry，不需要 explicit search algorithm。这是 "implicit reasoning" vs "explicit search" 的对比。

**Recurrence for depth (Geiping et al. 2025)**: https://arxiv.org/abs/2502.05171
用 recurrent depth 而不是 recurrent time 来做 test-time compute scaling。跟 Coconut 精神相似——都是用 recurrence 扩展 compute，但 Geiping 的 approach 更 general。

**Large Concept Models (Barrault et al. 2024)**: https://arxiv.org/abs/2412.08821
在 sentence embedding space 而不是 token space 做 LM。跟 Coconut 一样是 "脱离 token" 的思路，但 LCM 是在 sentence level，Coconut 是在 hidden state level。

**Energy-based Transformers (Gladstone et al. 2025)**: https://arxiv.org/abs/2507.02092
用 energy-based model 做 iterative refinement。跟 Coconut 一样是 latent space reasoning，但用 energy minimization 而不是 autoregressive。

**DeepSeek-R1**: https://arxiv.org/abs/2501.12948
用 RL 在 language space 训 reasoning。Coconut 目前只有 imitation learning，没有 RL stage。如果能把 R1 的 RL recipe 搬到 latent space，可能解锁更强能力——这是一个 obvious 但还没人做的方向。

---

## 12. 这篇 paper 的位置

你 Karpathy 一直说 LLM 的下一步是 latent reasoning、think before speak、system 2。这篇 paper 给了一个 **minimal viable implementation**，并展示了一个 surprising 的 emergent behavior（BFS）。

它没有 claim 击败所有 CoT method——实际上在 GSM8k 上还差标准 CoT 不少。它的 contribution 是：
1. 证明 latent reasoning 在工程上可行
2. 证明 latent space 能 emerge 出 language space 做不到的 search behavior
3. 给后续工作（theory、RL、pretraining）打了基础

如果你要在 latent reasoning 方向做点什么，这篇 + Zhu et al. theory + iCoT 是必须读的 trilogy。然后 obvious 的方向是：
- Latent reasoning + RL（no language supervision）
- Latent pretraining（从 pretraining 阶段就在 latent space，而不是 SFT 阶段才转）
- Hybrid：language skeleton + latent detail
- 跟 R1 / o1-style RL pipeline 结合，在 latent space 做 GRPO/PPO

个人觉得最 promising 的方向是 latent pretraining——现在 Coconut 的 gain 在 large model 上缩小，很可能是因为 SFT 阶段才转 latent 太晚了。如果从 pretraining 就让模型在 latent space "think"，可能能解锁更显著的 scaling behavior。Barrault et al. 和 Gladstone et al. 的工作已经在往这个方向走。

---

# Coconut: Chain of Continuous Thought 深度解析

Karpathy你好，这篇paper从Meta FAIR出来，第一作者Shibo Hao，通讯作者包括Jason Weston和Yuandong Tian。我会从intuition出发，逐层深入到method、training dynamics、emergent behavior和experimental evidence。

**Paper link**: https://arxiv.org/abs/2412.06769
**Code**: https://github.com/facebookresearch/coconut
**Theory companion (Zhu et al. 2025b)**: https://arxiv.org/abs/2505.12514
**Training dynamics (Zhu et al. 2025a)**: https://arxiv.org/abs/2509.23365

---

## 1. Motivation: 为什么language space对reasoning是次优的

这篇paper的开篇argument其实跟你自己在twitter上反复强调的"system 2 / test-time compute"想法一脉相承，但走得更激进。

**核心观察1**: token-level的compute budget是uniform的，但reasoning load是高度不均匀的。CoT里大部分token只是为了fluency（"So we have..."、"Therefore..."），只有少数critical token（比如math problem里的key intermediate variable）才真正承载planning。你给每个token都分配一样的FLOPs，这是obvious的misallocation。

**核心观察2**: 来自cognitive science的证据。Fedorenko et al. (2024) 在Nature上的工作 (https://www.nature.com/articles/s41586-024-07547-w) 表明human language network在reasoning task中其实是inactive的，language是communication tool，不是thought substrate。Amalric & Dehaene (2019) 的neuroimaging也支持math reasoning和language是dissociated的 (https://www.sciencedirect.com/science/article/pii/S1053811919300085)。

**核心观察3**: CoT的autoregressive nature让backtracking变得几乎不可能。模型commit到一条path之后，下一个token只能基于前面已经生成的token继续走。这是greedy的、不可逆的。

Coconut的solution：**干掉language bottleneck，让reasoning发生在hidden state space里，只在需要输出answer时才decode回language**。

---

## 2. Method: 形式化定义

### 2.1 Notation回顾

标准LLM的formulation：
$$H_t = \text{Transformer}(E_t)$$
$$\mathcal{M}(x_{t+1} | x_{\le t}) = \text{softmax}(W h_t)$$

变量解释：
- $x = (x_1, ..., x_T)$: input token序列
- $E_t = [e(x_1), e(x_2), ..., e(x_t)]$: 前$t$个token的embedding序列，$e(\cdot)$是embedding lookup
- $H_t \in \mathbb{R}^{t \times d}$: 所有position的last hidden state堆叠，$d$是hidden dimension
- $h_t = H_t[t, :]$: position $t$的last hidden state，形状$\mathbb{R}^d$
- $W$: LM head的参数矩阵，形状$\mathbb{R}^{|V| \times d}$，$|V|$是vocab size

### 2.2 Coconut的核心修改

引入两个special token：`<bot>` (begin of thought) 和 `<eot>` (end of thought)。

假设latent reasoning发生在position $i$到$j$之间，即 $x_i = $ `<bot>`, $x_j = $ `<eot>`。

**在latent mode** ($i < t < j$)：
$$E_t = [e(x_1), e(x_2), ..., e(x_i), h_i, h_{i+1}, ..., h_{t-1}]$$

也就是说position $t$的input不是$e(x_t)$（因为我们没有$x_t$），而是上一个position的last hidden state$h_{t-1}$。这是一个**self-loop**：output直接作为下一个step的input，绕过了离散化。

**回到language mode** ($t \ge j$)：
$$E_t = [e(x_1), ..., e(x_i), h_i, ..., h_{j-1}, e(x_j), ..., e(x_t)]$$

注意$\mathcal{M}(x_{t+1} | x_{\le t})$在latent mode里是**undefined的**——我们不去定义它，因为latent thought不打算被映射回vocab。但你可以仍然计算$\text{softmax}(Wh_t)$来做probing（Section 4.3就是这么做的）。

**关键实现细节**: $h_t$在feed back之前已经经过了final layer norm，所以magnitude不会爆炸。这一点很重要，否则残差连接累积会数值不稳定。

### 2.3 这个改动为什么是fundamental的

从你自己的"latent reasoning"框架看 (https://x.com/karpathy/status/...)，Coconut做的事情本质上是：

1. **Unrolled RNN**: 在latent mode里，每个forward pass就是在hidden state上的一个transition function。所以latent reasoning steps本质上就是RNN的time steps，但transition function是一个pre-trained transformer block。
2. **Differentiable**: 没有任何sampling或argmax，所以梯度可以从最后的loss一路backprop到所有continuous thoughts。
3. **Expressivity扩展**: Feng et al. (2023) 证明了CoT增加effective depth (https://arxiv.org/abs/2310.01460)；Coconut继承了这一点，但每个"depth"不再受限于vocab projection。

---

## 3. Training Procedure: 多阶段curriculum

### 3.1 为什么需要curriculum

直接让模型在latent space里reason（即"Coconut w/o curriculum"）会fail。Table 1里GSM8k上只有14.4%，比no-CoT还差。原因：latent space太大、没有inductive bias，模型不知道该怎么用这些free dimensions。

所以他们用了iCoT (Deng et al. 2024, https://arxiv.org/abs/2405.14838) 的multi-stage idea，但是把"removed language tokens"替换成"continuous thoughts"。

### 3.2 Stage-wise training

Hyperparameter $c$：每个language reasoning step被替换成$c$个continuous thoughts。

设原始CoT data是：
$$\text{Question} \rightarrow s_1 \rightarrow s_2 \rightarrow ... \rightarrow s_N \rightarrow \text{Answer}$$

其中$s_k$是第$k$个reasoning step（一组language tokens）。

- **Stage 0**: 标准CoT SFT，loss在所有reasoning step和answer上。
- **Stage $k$** (for $k=1,...,N$): 把前$k$个reasoning step替换为$k \times c$个continuous thoughts，形式变成：
  $$\text{Question} \rightarrow \text{<bot>} \underbrace{h, h, ..., h}_{k \times c} \text{<eot>} \rightarrow s_{k+1} \rightarrow ... \rightarrow s_N \rightarrow \text{Answer}$$
- Loss只在$s_{k+1}, ..., s_N$, Answer上计算，question和latent thoughts都被masked。

每个stage之间reset optimizer state（momentum等）——这是一个细节但很关键，因为不同stage的loss landscape差异大，旧的momentum会拖累。

### 3.3 Forward pass的代价

如果有$n$个latent thoughts，需要$n+1$次forward pass：
- 每次forward计算一个新的latent thought
- 最后一次forward得到remaining text sequence的loss

虽然可以用KV cache避免重复计算前面的tokens，但**latent thoughts之间是sequential的**——$h_{t}$依赖$h_{t-1}$，不能并行。这是Coconut的主要training bottleneck，作者也承认需要future work来优化。

### 3.4 Inference

`<bot>`固定加在question之后。`<eot>`两种处理方式：
- (a) 训练一个binary classifier在latent thought上判断是否terminate
- (b) pad到固定长度

实验里两种comparable，用了(b) for simplicity。

---

## 4. ProsQA: 为latent search量身定制的benchmark

### 4.1 Dataset设计

ProntoQA (Saparov & He 2022, https://arxiv.org/abs/2210.01240) 是existing logical reasoning dataset，但作者觉得它太简单——path基本是线性的。所以他们造了ProsQA (Proof with Search QA)。

ProsQA的核心：每个problem对应一个DAG，节点是entity或concept，边是逻辑蕴含关系。Question是"Is [Entity] a [Concept A] or [Concept B]?"。DAG保证[Entity]到[Concept A]有path，到[Concept B]没有。但DAG有多个branches，所以模型必须**搜索**。

Appendix A.2的Algorithm 1给了graph construction的pseudocode。关键设计：
- 用Poisson(1.5)采样每个new node的in-degree
- 用概率0.35/0.35/0.30控制新node属于"descendant of node 0 only / descendant of node 1 only / either"——这是为了保持binary question的validity
- Sampling weights偏向deeper nodes（$depth\_to\_root(c) \cdot 1.5 + 1$），保证reasoning chain较长

Table 2的statistics：平均23个node、36条edge、shortest path长度3.8、平均1.6条shortest path。

### 4.2 Evaluation metrics

6类output分类：
1. **Correct Path**: 输出是最短正确path之一
2. **Longer Path**: 正确但比最短path长
3. **Hallucination**: 包含不存在的edge或不连通
4. **Wrong Target**: 是valid path但终点不对
5. **Correct Label**: 只输出final answer且正确（适用于k较大时）
6. **Incorrect Label**: 只输出final answer且错误

这个分类非常细致，能区分"模型reasoning process对不对"vs"final answer蒙对没有"。

---

## 5. 核心实验结果

### 5.1 Table 1深度解读

| Method | GSM8k Acc | GSM8k #Tokens | ProntoQA Acc | ProntoQA #Tokens | ProsQA Acc | ProsQA #Tokens |
|--------|-----------|---------------|--------------|------------------|------------|----------------|
| CoT | 42.9 | 25.0 | 98.8 | 92.5 | 77.5 | 49.4 |
| No-CoT | 16.5 | 2.2 | 93.8 | 3.0 | 76.7 | 8.2 |
| iCoT | 30.0 | 2.2 | 99.8 | 3.0 | 98.2 | 8.2 |
| Pause Token | 16.4 | 2.2 | 77.7 | 3.0 | 75.9 | 8.2 |
| **Coconut** | **34.1** | **8.2** | **99.8** | **9.0** | **97.0** | **14.2** |
| - w/o curriculum | 14.4 | 8.2 | 52.4 | 9.0 | 76.1 | 14.2 |
| - w/o thought | 21.6 | 2.3 | 99.9 | 3.0 | 95.5 | 8.2 |
| - pause as thought | 24.1 | 2.2 | 100.0 | 3.0 | 96.6 | 8.2 |

关键observations：

**Observation 1: Chaining有效**。Coconut (34.1%) > Coconut w/o thought (21.6%) > Coconut pause as thought (24.1%) > No-CoT (16.5%)。说明continuous thoughts提供了genuine的额外computational capacity，比pause tokens更好。Pause tokens (Goyal et al. 2023, https://arxiv.org/abs/2310.02226) 是离散的learnable embeddings，没有information flow between them；Coconut的$h_t$依赖$h_{t-1}$，是一个真正的recurrent process。

**Observation 2: Curriculum至关重要**。w/o curriculum在GSM8k只有14.4%，比no-CoT还差。这跟你的intuition一致：latent space太大，没有supervision的话模型找不到useful manifold。

**Observation 3: ProsQA上Coconut接近iCoT但略低**。97.0 vs 98.2。但注意iCoT在GSM8k只有30.0，Coconut有34.1——说明在需要更多chaining的task上，continuous thought的expressivity advantage体现出来。

**Observation 4: Pause token在ProntoQA上collapse (77.7%)**。这个其实跟Pfau et al. (2024, https://arxiv.org/abs/2404.15758) 的结论一致：pause/filler tokens对parallelizable problems有效，但对需要serial reasoning的task不行。ProntoQA虽然有logic chain但linear，Pause Token表现差可能是因为GPT-2 base太弱。

### 5.2 Token efficiency (Figure 8)

Figure 8左图：横轴#generated tokens，纵轴accuracy。
- "language"曲线（iCoT-style internalize前m步）：accuracy快速下降
- Coconut曲线：accuracy下降平缓得多

意思是：在同样的token budget下，Coconut能保留更多accuracy。这是论文sell的"better trade-off"。

### 5.3 Clock-time efficiency (Table 4)

ProsQA上：Coconut 0.15s vs CoT 0.47s，3倍加速。ProntoQA上Coconut 0.11s vs CoT 0.85s，接近8倍加速。这是因为CoT要autoregressive生成几十个token，Coconut只生成几个latent thoughts加少量output tokens。

### 5.4 Scaling到Larger Models (Table 5)

| Model | no-CoT | Coconut |
|-------|--------|---------|
| Llama 3.2-3B | 26.0 | 31.7 |
| Llama 3-8B | 42.2 | 43.6 |

Gain随model size缩小（3B上+5.7，8B上+1.4）。作者的解释：larger model已经被language pretraining深度adapt，转latent更困难。我个人觉得这个解释半对半错——更可能是8B本身no-CoT已经42.2，ceiling效应。需要更大model + latent pretraining才能看出真正的gap。

---

## 6. 最有意思的部分：BFS的涌现

### 6.1 Probing实验设计 (Section 4.3, Figure 5)

作者用Figure 4的例子做case study。Question是判断Alex属于gorpus还是bompus。Alex的children是{lempus, sterpus, zhorpus, grimpus}。

他们手动设置$k$个continuous thoughts后强制model回到language mode，然后看model在第一个reasoning step输出的concept的概率分布。

定义**value function**：
$$V(\text{concept}) = \prod_{\text{token} \in \text{concept}} P(\text{token} | \text{context})$$

也就是concept内所有token条件概率的乘积。这可以被解读为"模型认为这个node能通向正确答案的implicit value"。

### 6.2 关键发现：不是greedy

Figure 5左：在第一个latent thought之后，value最高的concept是"lempus" (0.33)。
Figure 5右：在第二个latent thought之后，value最高的concept是"rorpus" (0.87)，而rorpus是grimpus的child！

如果是greedy search，model应该沿"lempus"往下走。但实际model在第二个thought里转向了grimpus的subtree，因为grimpus实际上能通向正确答案bompus。

这就是BFS的essence：**model在第一个thought里维持多个candidate的probability mass，在第二个thought里通过评估它们的children来disambiguate**。它没有立即commit到"lempus"这条局部最优path。

### 6.3 Parallelism的quantitative分析 (Figure 6)

Figure 6左（第一个thought）和右（第二个thought）画了top-1/top-2/top-3 candidate values的cumulative分布，across test set。

- 第一个thought：top-1, top-2, top-3曲线之间有大gap——说明probability mass分散在多个candidate上
- 第二个thought：gap显著缩小——说明model已经converge到top candidate

这个transition from "broad exploration" to "focused exploitation"正是BFS的signature。

### 6.4 为什么latent space能实现BFS

Section 4.4给出了hypothesis并验证：**node的height（到leaf的最短距离）越低，model的value估计越准确**。

Figure 7：横轴node height，纵轴value。Correct nodes的value应该高，incorrect nodes的value应该低。
- Height低（接近target）：correct和incorrect的value明显分开
- Height高（远离target）：correct和incorrect的value混在一起

这个phenomenon的解释：**离target近的node，可走的path少，evaluation容易；离target远的node，subtree大，evaluation困难**。

Language CoT的失败模式就是：被迫在height高的地方commit到一个branch，错了就回不去。
Coconut的成功模式：在height高的地方维持多个branch的superposition，等"展开"到height低的地方再做决定。

这一点跟Zhu et al. (2025b) 的theory paper完全吻合 (https://arxiv.org/abs/2505.12514)。他们证明continuous CoT可以通过**superposition states**编码multiple reasoning paths，这是discrete CoT做不到的。后续的training dynamics分析 (https://arxiv.org/abs/2509.23365) 进一步解释了这种superposition是怎么emerge的。

---

## 7. 一个具体的math reasoning例子 (Figure 9)

题目：James每周跑3次3个sprints，每个sprint 60米，一周总共跑多少米？

作者decode了第一个continuous thought对应的language tokens，发现它对应intermediate variable "3 * 3 = 9"（sprints per week）或者类似的中间量。

这个probing很有意思——它说明continuous thought**不是在压缩language CoT**，而是在学一个更efficient的representation。Training loss只要求最终answer正确，不要求latent thought能decode回natural language，所以model有freedom去学任何对prediction有帮助的representation。

---

## 8. Limitations和open questions

1. **Sequential forward passes**: training慢，每个latent thought都是sequential dependency。需要更好的parallelization或alternating training scheme。

2. **Curriculum依赖**: 不能完全脱离language CoT data。理想情况下应该能从latent pretraining直接学。Geiping et al. (2025, https://arxiv.org/abs/2502.05171) 的recurrent depth approach是一个方向。

3. **Large model上的gain缩小**: 需要latency-aware pretraining。Barrault et al. (2024) 的Large Concept Models (https://arxiv.org/abs/2412.08821) 和Gladstone et al. (2025) 的Energy-based Transformers (https://arxiv.org/abs/2507.02092) 都是relevant work。

4. **c=3时不稳定**: Appendix C.1提到c=3时loss spike。说明schedule还需要finer-grained design。

5. **没有 RL stage**: 整个pipeline是imitation learning from CoT data。如果能加一个RL stage（类似DeepSeek-R1 https://arxiv.org/abs/2501.12948 但是在latent space里），可能能解锁更强的reasoning capability。

---

## 9. 我对这个工作的整体看法

**Strong points**:
- Concept极其简洁：只是把LM head拿掉、把hidden state feed回去。Implementation可能就几十行代码。
- BFS emergence是真的surprising。模型没有被explicitly trained to do search，但latent space的几何性质让它**自然而然地**encode multiple paths。
- ProsQA这个benchmark设计得很巧妙，专门stress test planning能力。
- Theory follow-up (Zhu et al.) 显示了group在认真deep dive这个direction。

**Weak points**:
- GPT-2作为base model有点弱，larger model上的结果不够convincing。
- Curriculum dependence削弱了"pure latent reasoning"的claim。
- 没有test-time scaling的systematic study（图6的k∈{0,...,6}只是小规模）。
- BFS的emergence目前只在logical reasoning上demonstrated，math reasoning上没有similar analysis。

**对你的intuition来说**：
这篇paper验证了一个你一直强调的idea——**reasoning应该发生在latent space，language只是input/output interface**。它给出了一个minimal viable implementation，并展示了emergent search behavior。如果你想做latent reasoning的相关工作，这个paper + Zhu et al. theory + iCoT是必须读的trilogy。

**Potential directions值得explore**:
- Latent reasoning + RL (no language supervision at all)
- Hybrid: language skeleton + latent detail
- Latent thought的interpretability（Figure 9只是个开始）
- Latent reasoning for code generation（program synthesis天然需要planning）
- 跟Quiet-STaR (Zelikman et al. 2024, https://arxiv.org/abs/2403.09629) 结合，做latent thought的self-training

---

## 10. 关键references

- **iCoT (Deng et al. 2024)**: https://arxiv.org/abs/2405.14838 - Coconut的curriculum inspiration
- **Implicit CoT via KD (Deng et al. 2023)**: https://arxiv.org/abs/2311.01460
- **Pause Token (Goyal et al. 2023)**: https://arxiv.org/abs/2310.02226
- **Filler tokens (Pfau et al. 2024)**: https://arxiv.org/abs/2404.15758 - "thinking dot by dot"
- **ProntoQA (Saparov & He 2022)**: https://arxiv.org/abs/2210.01240
- **Tree of Thoughts (Yao et al. 2023)**: https://arxiv.org/abs/2305.10601 - explicit tree search
- **Reasoning with World Model (Hao et al. 2023)**: https://arxiv.org/abs/2305.14992 - planning视角
- **Recurrence for depth (Geiping et al. 2025)**: https://arxiv.org/abs/2502.05171
- **Large Concept Models (Barrault et al. 2024)**: https://arxiv.org/abs/2412.08821
- **Energy-based Transformers (Gladstone et al. 2025)**: https://arxiv.org/abs/2507.02092
- **Coconut theory (Zhu et al. 2025b)**: https://arxiv.org/abs/2505.12514
- **Coconut training dynamics (Zhu et al. 2025a)**: https://arxiv.org/abs/2509.23365
- **Quiet-STaR (Zelikman et al. 2024)**: https://arxiv.org/abs/2403.09629
- **DeepSeek-R1**: https://arxiv.org/abs/2501.12948
- **CoT expressivity (Feng et al. 2023)**: https://arxiv.org/abs/2310.01460

如果你想进一步深入某个方面（比如Zhu et al.的superposition theory、training dynamics的细节、或者跟PaLI/Gemini的latent reasoning对比），我可以继续展开。
