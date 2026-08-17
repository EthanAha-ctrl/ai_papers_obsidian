---
source_pdf: Quiet-STaR.pdf
paper_sha256: 3dd74be14ad29e5dbec1c454d1ac57d54219f57745510fb89d627602536bdae8
processed_at: '2026-08-06T08:00:33-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话重讲 Quiet-STaR

好,我换个口气说,像咱俩在白板前聊天的感觉。

---

## 一句话概括

让 LM 在每个 token 后面偷偷想几步,如果想到的东西能帮它更好地预测后面的字,就奖励这种"想";训练完之后,这模型在没见过的 reasoning task 上也变强了,纯靠 pretraining 就能涨点,不需要专门 fine-tune GSM8K / CommonsenseQA。

---

## 从哪来的灵感

你得先回忆 STaR 这个工作([Zelikman 2022](https://arxiv.org/abs/2203.14465))——给模型一道数学题,让它先写一段 reasoning 再给答案,答对了就把这段 reasoning 当作训练数据,答错了换一段。本质是 self-play on QA dataset。

Quiet-STaR 的作者就想:凭什么只在 QA 上玩?网页上随便一段文字,中间其实都隐含着推理——一个证明跳过的中间步骤、一句对话背后的心智推断、一段描述里的因果链。这些全都是 reasoning。LM 在做 next-token prediction 时,本来就被迫把这些 reasoning 压进 hidden state 里,只是没显式表达出来。

那就让模型在每个 token 后面写一小段 internal note,如果这段 note 能让未来 token 更好预测,就 reinforce 这段 note。这就把 STaR 从"QA 任务上的 self-play"升级成"general pretraining objective"。

---

## 怎么做 —— Think, Talk, Learn

### Think: 给每个 token 后面塞一段 thought

naive 的做法是,给长度 $l$ 的 sequence,每个位置都生成一段长度 $t$ 的 thought,那要做 $l \times t$ 次 forward pass,根本跑不动。

作者用了个聪明办法。Transformer 一次 forward 本来就会给所有位置输出一个 next-token 分布。所以一次 forward 就能同时给 $l$ 个位置各 sample 一个 thought token。下一步把这些 thought token 拼回去,再做一次 forward,再各 sample 一个——用对角 attention mask 确保每个 thought 只 attend 自己的 prefix,不互相干扰。

这样 $t$ 步生成只要 $t$ 次 forward,而不是 $l \times t$ 次。

具体来说,attention mask 长这样:

```
X_0 X_1 X_2 X_3 | T_0 T_1 T_2 T_3
1   1   1   1   |  1   0   0   0     <- T_0 sees all X, itself
1   1   1   1   |  0   1   0   0     <- T_1 sees all X, itself
1   1   1   1   |  0   0   1   0     <- T_2 sees all X, itself  
1   1   1   1   |  0   0   0   1     <- T_3 sees all X, itself
```

每个 thought 是独立分支,在 diagonal 上各跑各的。

### Talk: Mixing head 决定听不听 thought

刚加 thought 的时候,模型还不会"思考",生成的 thought 是 garbage,直接用会让 LM 退化。所以加一个 mixing head —— 一个 3-layer MLP,输入是 base hidden state 和 post-thought hidden state 拼起来,output 一个 scalar $w \in [0,1]$,表示"听多少 thought"。

$$
\log p^{\text{talk}} = w \cdot \log p^{\text{init}} + (1 - w) \cdot \log p^{\text{thought}}
$$

- $w$ 接近 1:基本忽略 thought,用 base 预测;
- $w$ 接近 0:完全信 thought。

作者试过 ablate 这个 mixing head,直接用 thought 后的 logits,结果模型快速学会**完全无视 thought** —— 因为 base LM 本身就强,强行覆盖反而坏事儿。所以 mixing head 本质是让模型自己学"什么时候 thought 有用,什么时候别瞎想"。

这其实挺像 attention —— 模型在 attend 自己的 thought,只不过 gate 是学出来的 scalar。

### Learn: REINFORCE 奖励有用的 thought

每个 thought $T_j$ 的 reward:

$$
r_j = \log p^{\text{talk}}(X_{j+1:j+n+1}) - \log \bar{p}^{\text{talk}}(X_{j+1:j+n+1})
$$

- 第一项:这条 thought 产生的 mixed 预测,对未来 $n$ 个 ground-truth token 的对数似然;
- 第二项:同一位置采的多个 thought 的平均似然(baseline,降方差用);
- $n_{\text{true}}$:看未来几个 token,paper 用 2-4。

然后 REINFORCE:

$$
\nabla \mathcal{L}^{\text{REINFORCE}} = -r_j \cdot \nabla \log p(T_j | X_{:j}, \langle\text{start}\rangle)
$$

**关键 trick**:只保留 $r_j > 0$ 的项,负 reward 直接砍掉。为啥?负 reward 会让模型同时压低所有 thought 概率,容易 mode collapse。代价是有 bias,但稳定。

另外还有 NLL loss,保证 talk head 本身能预测对。

---

## 几个关键细节

### Non-myopic —— 看多个 token 而不只一个

让 thought 解释未来 1 个 token vs 2 个 token,效果差很多。只看 1 个,thought 容易学到 trivial bigram trick(比如"cat"后面跟"sat")。看 2 个以上,thought 被迫 encode 更 high-level 的 semantic content。

实验数据:
- $n_{\text{true}}=1 \to 2$: CommonsenseQA +3.1%, GSM8K +0.3%
- $n_{\text{true}}=2 \to 3,4$: 数值没涨,但 rationale 定性更连贯

### Meta-tokens —— 学出来的"开始/结束思考"标记

`<startofthought>` 和 `<endofthought>` 是 learned embedding。初始化成 em-dash "—" 的 embedding,因为 "—" 在 corpus 里本来就有"停顿、思考"的语义,让模型借现成知识。

这两个 token 的 gradient 加了 $10^2$ 的 weight,因为 embedding 是离散 token 但本身连续可微,需要快速适应。

直觉:
- `<start>` 让模型进入"思考模式";
- `<end>` 告诉模型"想完了,该输出了",同时给 mixing head 提供 hidden state。

### Teacher Forcing —— 让未来 token 也能给 thought 梯度

parallel sampling 后,每个 thought 只对自己紧接的 token 有梯度通路。想让未来 token 也加监督,naive 做法是 sample 之前的 token,但这样模型会学到"忽略前文"。

作者用 teacher forcing:把 ground-truth token 直接塞进 context,但仍然用 parallel attention mask。这样 thought 到 future token 的梯度能通过 teacher-forced ground truth 流回。

---

## 实验数据

Base 是 Mistral 7B([Jiang 2023](https://arxiv.org/abs/2310.06825))。

| Task | Base | Quiet-STaR |
|------|------|------------|
| GSM8K (zero-shot) | 5.9% | **10.9%** |
| CommonsenseQA (zero-shot) | 36.3% | **47.2%** |
| GSM8K CoT maj@8 | 40.6% | **47.7%** |

几个关键事实:

1. **完全没 fine-tune 这些 task**——纯 continued pretraining 后的 zero-shot,这是 generalization 的强证据;
2. **Thought length scaling**——4 → 8 → 12 → 32 token,性能单调上升,说明模型学到了"想得久 = 想得好";
3. **C4 上也 work 但涨幅小**——因为 C4 大部分 token 是 trivial 的,"the cat sat on the",不需要 thought 就能预测;
4. **OpenWebMath 上涨幅大**——数学文本 reasoning 密度高,thought 有用武之地。

---

## Improvement Distribution —— 最有 intuition 的一张图

把所有 token 的 $\log p$ 改变量画分布,发现:

- **绝大多数 token 几乎不变**(图上是巨大的峰在 0 附近);
- **长尾的"难 token"有显著增益**。

直觉非常对:大部分 web text 是 trivial 的,不需要 thought。但难 token——定理名、proof 的下一步、罕见实体——需要 reasoning 才能 predict。Thought 在这些位置贡献巨大。

Appendix 给了个具体例子:三角恒等式证明里,thought 在"$\sin^2(\theta) \to 1-\cos^2(\theta)$"这种 substitution 位置上贡献最大(深绿),而在"have you tried use..."这种 boilerplate 位置贡献接近 0(黄色)。

---

## 为什么 multi-token thought 比 pause token 强

[Goyal 2023](https://arxiv.org/abs/2310.02226) 的 pause token 本质是 single-token thought——每个 token 拆成两个,第二个"暂停"一下。实验上:

- Pause token 在 CommonsenseQA 上从 26.9% 涨到 28.8%(微涨);
- Pause token 在 GSM8K 上**反而降**;
- 多加 pause token 通常有害。

Quiet-STaR 的 multi-token thought 能 encode 真正的 reasoning chain,不只是"延迟一下"。这符合直觉——reasoning 是 sequential computation,一个 token 装不下。

---

## 几个 Stability 的坑 (Appendix I 写得很诚实)

作者试过各种方案,大部分都炸了:

1. **Gumbel-softmax straight-through**:多层 softmax 梯度消失;
2. **DQN/PPO/A3C**:reward function 本身随 mixing head 变化不稳定,RL 算法不收敛;
3. **Separate think/talk heads**:linear、MLP residual init 0,都 unstable;
4. **直接用 thought 后 logits(不用 mixing head)**:模型快速学会无视 thought。

最后稳定下来的配方:
- 最小化 transformation——让 thought 后 hidden state 直接过 base LM head,只让 mixing head 学权重;
- REINFORCE 截断负 reward;
- Mixing head 当 attention——避免 representation 直接 emit。

---

## 我的几点 intuition

### 1. 本质上是给 model 额外 compute budget

Base LM 在 pretraining 时被迫把所有 reasoning 压进 next-token prediction 的 hidden state 里,但 single token 是个很窄的 bottleneck。Quiet-STaR 给模型开了个口子,让它把 implicit computation **外化成 explicit tokens**,这些 tokens 再反过来帮预测。

这跟 [Prystawski 2024](https://arxiv.org/abs/2405.15454) 的 "locality of experience" 假设一致——训练数据里大部分 reasoning step 都是局部的,模型学到的是局部 step。Thought 让长 reasoning decompose 成局部 step。

### 2. 跟 test-time compute scaling 的关系

Quiet-STaR 是在**训练时**就让模型学会"怎么用 test-time compute"。这跟 [Snell 2024](https://arxiv.org/abs/2408.12637) 的 test-time scaling laws 方向契合——如果模型内在能 allocate compute based on difficulty,scaling test-time compute 才有意义。Mixing head 就是个"难度感知 compute allocator"的雏形。

### 3. 跟 O1 的关系

[OpenAI o1](https://openai.com/o1/) 的思路是用大量 RL + verifier 在 reasoning trace 上训练。Quiet-STaR 是简化版——用 web text + REINFORCE 训练 thought generation,不需要 explicit reasoning dataset,更 organic。两者都在做"训练时让模型学会 test-time think longer",但 Quiet-STaR 更轻量。

### 4. 跟 Backpack LM 的相似性

[Backpack LM (Hewitt 2023)](https://arxiv.org/abs/2305.16765) 也是学权重去 weighted-sum representation,避免让 LM 直接 emit arbitrary embedding。Quiet-STaR 的 mixing head 思路一样——避免 representation collapse 的 instability。

### 5. 为啥 discrete token thought 比 continuous latent 更好训

Thought 必须是 discrete tokens 吗?Continuous latent thought 理论上更灵活,但容易 representation drift。Discrete tokens 在已有 LM manifold 上 explore,天然 stable。这其实是 exploration vs. exploitation trade-off——discrete tokens 在已有分布里 explore,continuous latent 能 explore 更广但更难训。

---

## 局限

作者承认的:
- 只在 7B 上验证,没在更大模型;
- 没 from-scratch 实验,只是 continued pretraining;
- Compute overhead 巨大——每个 token 都要生成 thought;
- 没动态决定何时 think,所有 token 都 think,但其实大多数 token 不需要;
- Faithfulness 问题——thought 是否真的反映 model 内部 computation 无法保证。

我想加几条:
- **Reward sparsity**:Figure 7 显示只有一小撮 token 有显著 reward signal,大部分 thought 是 wasted gradient。能不能用 value function 引导只在 hard token 上 think?
- **Length generalization**:训练 thought 长度 12,eval 时能否外推到 100+?
- **Compositionality**:能不能让 thought 里嵌 thought?类似 [STOP (Zelikman 2023b)](https://arxiv.org/abs/2310.02304) 的递归 self-improvement。

---

## 几个我想到的 follow-up

1. **Predict-before-think 的 mixing head**:把 mixing head 移到 thought 生成前,预测"这个位置需不需要 think",做 adaptive compute;
2. **Hierarchical thoughts**:thought 里再嵌 thought;
3. **Thought compression**:用 gist token 压缩长 thought,类似 [Mu 2024](https://arxiv.org/abs/2304.15096);
4. **Speculative thought**:小模型生成 draft thought,大模型 verify,加速 parallel generation;
5. **Multi-modal thoughts**:thought 可以是 image patch / sketch。

---

## 一句话总结

Quiet-STaR 把 STaR 从"QA 上 bootstrap reasoning"推广到"任意 web text 上 bootstrap reasoning",核心是 **parallel thought generation + mixing head + non-myopic REINFORCE**。它证明 LM 可以通过"每个 token 后悄悄 think 一下"从 pretraining corpus 里学会 general reasoning,并 transfer 到 unseen task。整篇 paper 的妙处在于:把 reasoning 从 explicit task 退化成 language modeling 的 latent structure,让 reasoning 学习融入 pretraining 本身。

---

## References

- [Quiet-STaR paper](https://arxiv.org/abs/2403.09629)
- [STaR (Zelikman 2022)](https://arxiv.org/abs/2203.14465)
- [Mistral 7B (Jiang 2023)](https://arxiv.org/abs/2310.06825)
- [Chain-of-Thought (Wei 2022)](https://arxiv.org/abs/2201.11903)
- [Pause Tokens (Goyal 2023)](https://arxiv.org/abs/2310.02226)
- [Gist Tokens (Mu 2024)](https://arxiv.org/abs/2304.15096)
- [Toolformer (Schick 2024)](https://arxiv.org/abs/2302.04761)
- [OpenWebMath (Paster 2023)](https://arxiv.org/abs/2310.06786)
- [C4 (Raffel 2020)](https://arxiv.org/abs/1910.10683)
- [CommonsenseQA (Talmor 2018)](https://arxiv.org/abs/1811.00937)
- [GSM8K (Cobbe 2021)](https://arxiv.org/abs/2110.14168)
- [Backpack LMs (Hewitt 2023)](https://arxiv.org/abs/2305.16765)
- [TRICE (Hoffman 2024)](https://arxiv.org/abs/2401.04094)
- [V-STaR (Hosseini 2024)](https://arxiv.org/abs/2402.06457)
- [STOP (Zelikman 2023b)](https://arxiv.org/abs/2310.02304)
- [Why think step by step (Prystawski 2024)](https://arxiv.org/abs/2405.15454)
- [REINFORCE (Williams 1992)](https://link.springer.com/article/10.1007/BF00992696)
- [Function Vectors (Todd 2023)](https://arxiv.org/abs/2310.15213)
- [Scratchpads (Nye 2021)](https://arxiv.org/abs/2112.00114)
- [Zero-shot CoT (Kojima 2022)](https://arxiv.org/abs/2205.11916)
- [Test-time scaling laws (Snell 2024)](https://arxiv.org/abs/2408.12637)
- [Eric Zelikman's page](https://ericzelikman.com)

---

# Quiet-STaR 深度解析：让 LM 学会"沉默地思考"

Andrej，这篇 paper 我读得很兴奋，因为它实质上把你之前在 eureka moments 那条线上（[STaR](https://arxiv.org/abs/2203.14465)）的思路从"task-specific QA"解放到"任意 web text"。它把 chain-of-thought 这种显式推理训练范式，泛化成了一种 general 的 pretraining-style 目标。下面我从 motivation、算法、架构、实验、直觉多个层次拆开讲。

---

## 1. Motivation 与核心 idea

传统 CoT ([Wei et al. 2022](https://arxiv.org/abs/2201.11903)) 和 STaR ([Zelikman et al. 2022](https://arxiv.org/abs/2203.14465)) 都把"推理"框定在一个明确任务上——给个 question，model 生成 rationale，再去 answer，对则 reward。这种 setting 的 bottleneck 在于：

1. **需要 curated QA dataset**，规模天然受限；
2. **推理分布是 off-policy**——human annotation 的 reasoning trace 与 LM 自己会生成的分布不一致；
3. **覆盖面窄**——只能学 dataset 里涉及的 reasoning pattern。

Quiet-STaR 的核心 insight 是：**几乎任何 written text 之间都隐藏着 unstated reasoning**。一个 proof 跳过的中间步骤、一段对话背后的 theory of mind、一句话隐含的 causal chain——这些都是 reasoning。如果把 LM 的 pretraining 目标从 "predict next token" 扩展成 "think, then predict next token"，就能在 general pretraining 中把 reasoning 当作 byproduct 学出来。

这其实呼应了 [Radford et al. 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 的 "LMs are unsupervised multitask learners"——既然 web text 是 multitask 的 mixture，那 reasoning 就该能从中学到。

---

## 2. 形式化目标

Paper 给出的优化目标：

$$
\theta^* = \arg\max_\theta \, \mathbb{E}_x \Big[ \log p_\theta \big( x_{i:n} \mid x_{0:i}, \, \text{rationale}_\theta(x_{0:i}) \big) \Big]
$$

变量解释：
- $\theta$: LM 参数；
- $x = (x_0, x_1, \dots, x_n)$: 一段文本 token 序列；
- $x_{0:i}$: 前 $i$ 个 token（observed prefix）；
- $\text{rationale}_\theta(x_{0:i})$: 由当前 $\theta$ 生成的一段 thought（rationale），作为 latent variable 插在 $x_{i-1}$ 与 $x_i$ 之间；
- $x_{i:n}$: 从位置 $i$ 开始的剩余 ground-truth 文本。

注意一个**重要细节**：作者强调这是 **non-myopic** 目标——预测 $x_{i:n}$ 而非仅仅 $x_i$。对一个已经 optimal 的 LM 而言，next-token 与 sequence prediction 等价；但对一个 finite-capacity 的 LM，让 thought 去解释未来多个 token 而不只一个 token，能引导 thought 学到"语义"而非"表面 bigram"。这一点在 ablation 里被验证：预测 2 个 token ahead 比 1 个 token 在 CommonsenseQA 上高 3.1%，再增加收益递减但 rationale 更连贯。

---

## 3. Algorithm 1 全流程拆解

这个算法的核心是三步循环：**think → talk → learn**。

### 3.1 Parallel Generation (Think)

最棘手的工程问题是：naively 给每个 token 都 attach 一个 thought，需要 $O(n)$ 次独立 forward pass。Quiet-STaR 用一个**精巧的 attention mask** 把它压成单次 batched forward。

关键观察：一次 transformer forward 已经会给 sequence 里每个 position $j$ 都输出一个 next-token 分布。如果我们对每个 position 都 sample 一个"counterfactual continuation"，这些 continuation 是各自独立分支——它们彼此不应 attend。Paper 用**diagonal attention mask**：每个 thought token 只能 attend 到 (a) 生成它的 prefix $x_{:j}$，(b) 同一个 thought 内的前序 thought token，(c) 自己。不能 attend 到其他 parallel thoughts。

形式上，给定序列 $X = (x_0, \dots, x_l)$，我们生成 $l$ 条 thoughts $T_1, \dots, T_l$，每条长度 $t$。attention matrix 的结构大致是 block-diagonal + lower-triangular：

```
[ X X X X 0 0 0 0 0 0 0 0 ]    <- x_0, x_1, x_2, x_3 (text)
[ X X X X T 0 0 0 0 0 0 0 ]    <- thought for x_0
[ X X X X 0 T 0 0 0 0 0 0 ]    <- thought for x_1
[ X X X X 0 0 T 0 0 0 0 0 ]    <- thought for x_2
[ X X X X 0 0 0 T 0 0 0 0 ]    <- thought for x_3
```

这样 $t$ 步生成只需要 $t$ 次 forward pass（每步给所有 thoughts 同时多 sample 一个 token），而不是 $l \times t$ 次。Appendix B 进一步说，由于我们只关心 diagonal 上的 attention score（每个 thought token 只对应自己 prefix），可以把 $(b, t, l, l)$ 的 pairwise attention 退化成 $(b, t, l)$ 的 elementwise dot product——显存大幅下降。

### 3.2 Mixing Head (Talk)

刚 init 时 thought 分布是 OOD 的，直接用 thought 后的 logits 会让 LM 退化。作者引入一个 **mixing head**：一个 3-layer MLP，输入是 (post-thought hidden state, base hidden state) 的 concatenation（dim = $2 \times d_{\text{hidden}}$），输出 scalar $w_j \in [0, 1]$ per token。

mixed logits 为：

$$
\log p_j^{\text{talk}} = w_j \cdot \log p_j^{\text{init}} + (1 - w_j) \cdot \log p_j^{\text{thought}}
$$

变量：
- $\log p_j^{\text{init}}$: **不思考**时位置 $j$ 的 next-token log-prob（来自 base LM head）；
- $\log p_j^{\text{thought}}$: 思考后位置 $j$ 的 next-token log-prob；
- $w_j$: mixing weight，标量；接近 1 表示完全用 base 预测，接近 0 表示完全用 thought 预测。

这本质上是个 **residual / gated attention 机制**——让 model 学会"什么时候相信 thought"。Appendix I 提到：如果 ablate 掉 mixing head，直接用 thought 后的 logits，模型会快速学会**完全忽略 thought**，下游 zero-shot 也没泛化。这非常符合直觉——base LM 的 prediction 本身就是个非常强的 prior，粗暴替换会破坏模型已经学到的东西。

类比一下：[Backpack LMs (Hewitt et al. 2023)](https://arxiv.org/abs/2305.16765) 也是用"学权重去 weighted-sum representation"，避免让 LM 直接 emit arbitrary embedding——这避免了 representation collapse 的 instability。

### 3.3 REINFORCE Loss (Learn)

这是整个 pipeline 的"reward 信号"。对每个 thought $T_j$，定义 reward：

$$
r_j = \log p_{j:j+n_{\text{true}}}^{\text{talk}}(X_{j+1:j+n_{\text{true}}+1}) - \log \overline{p}_{j:j+n_{\text{true}}}^{\text{talk}}(X_{j+1:j+n_{\text{true}}+1})
$$

变量：
- $n_{\text{true}}$: 监督的 future token 数（hyperparameter，paper 用 2-4）；
- $X_{j+1:j+n_{\text{true}}+1}$: ground-truth 未来 $n_{\text{true}}$ 个 token；
- $\log p^{\text{talk}}$: 该 thought 产生的 mixed 预测对这些 ground-truth token 的对数似然；
- $\overline{p}^{\text{talk}}$: 同一位置 $j$ 上采的多个 thoughts 的平均似然（baseline for variance reduction）。

这是经典的 **advantage** 形式 $A = Q - V$——只不过这里 $V$ 用 batch average 估计。然后 REINFORCE 梯度：

$$
\nabla_\theta \mathcal{L}_j^{\text{REINFORCE}} = -r_j \cdot \nabla_\theta \log p_\theta(T_j \mid [X_{:j}; \langle\text{startofthought}\rangle])
$$

注意一个**关键的 trick**：作者**只保留 $r_j > 0$ 的项**（截断负 reward）。理由是 stability——负 reward 会让 model 同时压低所有 thought 概率，导致 mode collapse。这虽然引入 bias（不再是严格 unbiased estimator），但实践上大幅稳化训练。这跟 [TRICE (Phan et al. 2023)](https://arxiv.org/abs/2401.04094) 用 marginal likelihood 的思路有些类似，但 Quiet-STaR 更简单。

此外还有 NLL loss（确保 talk 头本身仍能预测 ground truth）：

$$
\mathcal{L}_j^{\text{NLL}} = -\log p_{j:j+n_{\text{true}}}^{\text{talk}}(X_{j+1:j+n_{\text{true}}+1})
$$

Total loss: $\mathcal{L}_j = \mathcal{L}_j^{\text{NLL}} + \mathcal{L}_j^{\text{REINFORCE}}$。

### 3.4 Meta-tokens 优化

$\langle\text{startofthought}\rangle$ 与 $\langle\text{endofthought}\rangle$ 是 learned embedding。初始化为 em-dash "—" 的 embedding——因为 "—" 在 corpus 里经常表示"pause/thought"，让 LM 利用其 prior 知识。

由于 embedding 是离散的 token 但本身连续可微，作者对这两个 token 的 gradient 加了 hyperparameter weight（paper 用 $10^2$），加速其适应。直觉上：
- `<startofthought>` 让 LM 进入"思考模式"——可以理解为 attention pattern 的 mode switch；
- `<endofthought>` 告诉 LM "想完了，该 speak 了"——同时给 mixing head 提供它需要的 hidden state。

这跟 [Mu et al. 2024](https://arxiv.org/abs/2304.15096) 的 gist tokens、[Goyal et al. 2023](https://arxiv.org/abs/2310.02226) 的 pause tokens 思想相通，但 Quiet-STaR 的 thought 是 multi-token 的，pause token 只相当于 single-token thought 的退化版。

### 3.5 Non-myopic Teacher Forcing

由于 parallel sampling 后，每个 thought 只对自己紧接的 token 有梯度通路（其他 thoughts 把后续 token "遮蔽"了）。如果想给 future token 也加监督信号，naive 做法是 sample 之前的 token——但这样会让 LM 学到"忽略前文"。

作者用 **teacher forcing**：在计算 future token 的 log-prob 时，把 ground-truth token 直接拼到 context 后面，但仍然用同一套 parallel attention mask。这样：

- thought → next-token 的梯度通过 NLL/REINFORCE 流回 thought 生成参数；
- thought → token $j+2$ 的梯度通过 teacher-forced ground-truth $x_{j+1}$ 也能流回。

可视化见 Figure 4：solid 线是 LM 实际 forward，dashed 线是 teacher-forced 插入的 ground-truth token。

---

## 4. 架构示意图（基于 Figure 1）

```
   X_0    X_1    X_2    X_3    X_4    X_5
    |      |      |      |      |      |
    v      v      v      v      v      v
 [LM fwd]→h0   [LM fwd]→h1  ...   (base hidden states)
    |      |                        |
   +T0   +T0   ...                  |  (insert <start> + thought + <end>)
    |      |                        |
    v      v                        v
 [LM fwd] [LM fwd]  ...          (post-thought hidden states)
    |      |                        |
    +------|--- mixing head -------|--- (concat h_init, h_thought) → MLP → w
    |                                 |
    v                                 v
 log p_init                       log p_thought
    \                                /
     \__ w * log p_init + (1-w) * log p_thought __/  → log p_talk
                            |
                            v
                   NLL on future ground truth tokens
                            +
                   REINFORCE with reward r_j (truncated to r_j > 0)
```

整张图的妙处在于：**thought 是 "horizontally" 插在序列里的，但训练信号是 "vertically" 通过 future token prediction 流回来的**。

---

## 5. 实验细节与数据

### 5.1 Setup

- Base model: **Mistral 7B** ([Jiang et al. 2023](https://arxiv.org/abs/2310.06825))；
- 训练数据：
  - 主实验：**OpenWebMath** ([Paster et al. 2023](https://arxiv.org/abs/2310.06786))——reasoning 密度高；
  - 次实验：**C4** ([Raffel et al. 2020](https://arxiv.org/abs/1910.10683))——验证 general text 也 work；
- 硬件：单节点 8×80GB H100；
- Optimizer: AdamW, lr=1e-6, warmup 20 steps, weight decay 0.001, batch=8；
- Thought length: 12 tokens 主实验，也试过 4/8/16/32；
- $n_{\text{true}}=4$ future tokens；
- Temperature: training T=1；REINFORCE loss 在 T=3 下评估（importance sampling）；
- Thought 数 per token: 2-4（baseline average）；
- Sequence length: 256 tokens random crop。

### 5.2 主要结果表

| Task | Base Mistral 7B | Quiet-STaR (OpenWebMath) | Quiet-STaR (C4) | Pause Tokens (Goyal 2023) |
|------|-----------------|--------------------------|------------------|---------------------------|
| GSM8K (zero-shot) | 5.9% | **10.9%** | 8.1% | 略降 |
| CommonsenseQA (zero-shot) | 36.3% | **47.2%** | 42.6% | 26.9%→28.8% |
| CoT-maj@8 on GSM8K | 40.6% | **47.7%** | — | — |

几个值得注意的点：

1. **No fine-tuning on these tasks**——纯 continued pretraining 后的 zero-shot 表现，这是 generalization 的强证据；
2. **Scaling with thought length**——thought token 从 4 → 8 → 12 → 32，性能单调上升，说明"think longer = think better"被训练进 model 了；
3. **Pause tokens** 基本没用甚至有害——说明 multi-token rationale 才是真 reasoning，single-token pause 只是"延迟"，不构成 computation；
4. **C4 上也有提升，但小**——说明 reasoning-dense corpus 上 thought 更 useful（作者猜测：C4 大部分 token 不需要 thought 就能预测）。

### 5.3 Improvement Distribution (Figure 7)

最有 intuition-building 的一张图：把所有 token 的 $\log p$ 改变量画分布，发现**绝大多数 token 几乎不变**，但**长尾的"难 token"有显著增益**。这完全符合 hypothesis——大部分 web text 是 trivial 的（"the cat sat on the..."），但难 token（定理名、proof 的下一步、罕见实体）需要 thought。

Appendix Figure 8 给了个具体例子：一个三角恒等式的 proof 里，thought 在"$\sin^2(\theta) \to 1-\cos^2(\theta)$"这种 substitution 的位置上贡献最大（深绿高 opacity），而在"have you tried use..."这种 boilerplate 位置贡献接近 0（黄色）。

---

## 6. 与 CoT 的互补性

一个非常 elegant 的实验（Section 5.3）：用 Quiet-STaR 训过的 model 做 zero-shot CoT prompt（"Let's think step by step"），相比 base Mistral 7B 的 CoT，**生成更结构化、更连贯**——cot-maj@8 从 40.6% 提到 47.7%。

Appendix E 的具体输出对比很有意思：base Mistral 经常在 GSM8K 上算错（如第三、四、五个 response 都错），OpenWebMath fine-tuned 的 Mistral 也错（数字算错或推理链断裂）。Quiet-STaR-trained 的 Mistral 5 个 response 全对，且每个 step 都有明确"这是什么"的标注。

直觉上：Quiet-STaR 让 model 在 internal thought 阶段已经把"reasoning 习惯"练熟，所以显式 CoT 时这种能力自然 transfer——就像人内部思考多了，外显讲解也更清晰。

---

## 7. Ablation 关键发现

### 7.1 Multiple thoughts per token

- 不用 multiple thoughts（只用 1 个 thought，reward = log p_thought - log p_init）：baseline；
- 2 thoughts：+0.5% GSM8K，+3% CommonsenseQA；
- 3-4 thoughts：再 +0.1-0.3%，**diminishing return**。
- 启示：variance reduction 主要价值在前 1-2 个 thought，再多就是边际效益。

### 7.2 Non-myopic (n_true)

- $n_{\text{true}}=1$ vs $n_{\text{true}}=2$：+0.3% GSM8K, +3.1% CommonsenseQA；
- $n_{\text{true}}=3, 4$：基本不变；
- 但**定性观察**：$n_{\text{true}}$ 大时 rationale 更连贯、更有"语义"。

直觉：让 thought 解释 1 个 token 容易让 thought 学到 trivial bigram trick；解释 2 个 token 以上迫使 thought 编码更高层 semantic content。

---

## 8. Stability 的若干考量 (Appendix I)

Paper 在这里写得相当诚实。几个 instability 源：

1. **Circular dependency**：thought 的有用性 $\to$ mixing head 学习；mixing head 又依赖 thought。如果 mixing head 学到 $w=1$（忽略 thought），thought 没梯度信号，永远不学。
2. **Gumbel-softmax straight-through**：作者试过，但多层 softmax 导致 vanishing gradient。
3. **RL stability**：DQN/PPO/A3C 都试过，但 reward function 本身随 mixing head 变化不稳定——preliminary 不 promising。
4. **Separate think/talk heads**：试过 linear、MLP residual init 0，但仍然 unstable。

最终 stable 的配方：
- **最小化 transformation**：让 LM 的 thought 后 hidden state 直接通过 base LM head 出 logits，只让 mixing head 学权重，不学新 head；
- **REINFORCE 截断负 reward**；
- **Mixing head 当 attention**：本质上 mixing head 是让 LM "attend to its own thought"的 attention weight，跟 Backpack 一样避免 representation 直接 emit。

---

## 9. 局限性与 open questions

作者承认的：
1. 只在 7B 上验证，没在更大模型；
2. 没有 from-scratch 实验，只是 continued pretraining；
3. Compute overhead 巨大——每个 token 都要生成 thought，FLOPs 显著增加（虽然 Figure 6 是 compute-normalized 的图，但 raw compute 仍很高）；
4. 没有动态决定何时 think——所有 token 都 think，但其实 Figure 7 显示大多数 token 不需要；
5. Faithfulness 问题（Ethics Statement 提到）——thought 是不是真的反映 model 内部 computation 无法保证。

我额外想几点：
- **Reward sparsity**：Figure 7 显示只有一小撮 token 有显著 reward signal，这意味着大部分 thought 是 "wasted gradient"，能不能用 value function 来引导只在 hard token 上 think？
- **Length generalization**：训练 thought 长度 12，eval 时能否外推到 100+？CoT 里 length generalization 一直是个问题。
- **Compositionality**：能不能让 thought 里有 nested thought？类似 [STOP (Zelikman et al. 2023b)](https://arxiv.org/abs/2310.02304) 的递归 self-improvement。
- **与 RLHF 的结合**：thought 本质上是个 latent action，可以套 PPO 之类的，但 Appendix I 说 reward 不稳定。能否用 RLHF 替代 REINFORCE 的 thought 评分？

---

## 10. 我的 intuition 与延伸思考

### 10.1 为什么这 work？

我倾向于这么理解：base LM 在预训练时**被迫把 reasoning 压进 next-token prediction**——但 single token 是个 very narrow bottleneck，所有"中间计算"必须 implicit 编码在 hidden state 里。Quiet-STaR 实质上是**给 model 分配了额外的 "compute budget"**——让它把 implicit computation **外化成 explicit tokens**，然后再用这些 tokens 帮助预测。这跟 [Prystawski et al. 2024](https://arxiv.org/abs/2405.15454) 的 "locality of experience" 假设一致——CoT 之所以 work，是因为训练数据里大部分推理 step 都是局部的，模型学到的也是局部 step，而 chain 把长推理 decompose 成局部 step。

### 10.2 跟 Test-Time Compute Scaling 的关系

Quiet-STaR 实质上是在**训练时**就让模型学会"如何用 test-time compute"。这跟 [Snell et al. 2024 (test-time scaling laws)](https://arxiv.org/abs/2408.12637) 的方向很契合——如果模型内在能 allocate compute based on difficulty，那 scaling test-time compute 才有意义。Quiet-STaR 的 mixing head 其实就是个"难度感知 compute allocator"的雏形。

### 10.3 跟 O1/o1-style reasoning model 的关系

如果对照 [OpenAI o1](https://openai.com/o1/) 的思路，Quiet-STaR 算是个**简化版**：o1 用大量 RL + verifier 在 reasoning trace 上训练；Quiet-STaR 用 web text + REINFORCE 训练 thought generation。两者都在做"训练时让模型学会 test-time think longer"，但 Quiet-STaR 不需要 explicit reasoning dataset，更"organic"。

### 10.4 与 Continuous Thought 的对比

Thought 必须是 discrete tokens 吗？[Continuous CoT](https://arxiv.org/abs/2402.04848)、[Quiet-STaR 的潜在变体] 可以让 thought 是 continuous latent。但作者用 discrete tokens 的理由是：representation 自然在 LM 的 manifold 上，避免 representation drift。这其实是个 exploration vs. exploitation trade-off——discrete tokens 在已有 LM 分布里 explore，continuous latent 可以 explore 更广但更难训。

---

## 11. 一些可能的 follow-up 方向

1. **Predict-before-think 的 mixing head**：把 mixing head 移到 thought 生成前，预测"这个位置需不需要 think"，做 adaptive compute。
2. **Hierarchical thoughts**：thought 里再嵌 thought，类似 Recurrent  reasoning。
3. **Thought compression**：用 gist token 把长 thought 压缩，类似 [Mu et al. 2024](https://arxiv.org/abs/2304.15096)。
4. **Speculative thought**：用小 model 生成 draft thought，大 model verify——可以加速 parallel generation。
5. **Multi-modal thoughts**：thought 不一定要是 text，可以是 image patch / sketch。

---

## Reference 链接

- [Quiet-STaR paper (Zelikman et al. 2024)](https://arxiv.org/abs/2403.09629)
- [STaR (Zelikman et al. 2022)](https://arxiv.org/abs/2203.14465)
- [Mistral 7B (Jiang et al. 2023)](https://arxiv.org/abs/2310.06825)
- [Chain-of-Thought (Wei et al. 2022)](https://arxiv.org/abs/2201.11903)
- [Self-Consistency CoT (Wang et al. 2022)](https://arxiv.org/abs/2203.11171)
- [Pause Tokens (Goyal et al. 2023)](https://arxiv.org/abs/2310.02226)
- [Gist Tokens (Mu et al. 2024)](https://arxiv.org/abs/2304.15096)
- [Toolformer (Schick et al. 2024)](https://arxiv.org/abs/2302.04761)
- [OpenWebMath (Paster et al. 2023)](https://arxiv.org/abs/2310.06786)
- [C4 (Raffel et al. 2020)](https://arxiv.org/abs/1910.10683)
- [CommonsenseQA (Talmor et al. 2018)](https://arxiv.org/abs/1811.00937)
- [GSM8K (Cobbe et al. 2021)](https://arxiv.org/abs/2110.14168)
- [Backpack LMs (Hewitt et al. 2023)](https://arxiv.org/abs/2305.16765)
- [TRICE (Hoffman et al. 2024)](https://arxiv.org/abs/2401.04094)
- [V-STaR (Hosseini et al. 2024)](https://arxiv.org/abs/2402.06457)
- [STOP (Zelikman et al. 2023b)](https://arxiv.org/abs/2310.02304)
- [Why think step by step (Prystawski et al. 2024)](https://arxiv.org/abs/2405.15454)
- [REINFORCE (Williams 1992)](https://link.springer.com/article/10.1007/BF00992696)
- [Function Vectors (Todd et al. 2023)](https://arxiv.org/abs/2310.15213)
- [Scratchpads (Nye et al. 2021)](https://arxiv.org/abs/2112.00114)
- [Zero-shot CoT (Kojima et al. 2022)](https://arxiv.org/abs/2205.11916)
- [Test-time scaling laws (Snell et al. 2024)](https://arxiv.org/abs/2408.12637)
- [Eric Zelikman's personal page](https://ericzelikman.com)

---

## 总结一句话

Quiet-STaR 把 STaR 从"在 QA 上 bootstrap reasoning"推广到"在任意 web text 上 bootstrap reasoning"，关键技术是 **parallel thought generation + mixing head + non-myopic REINFORCE**。它证明了 LM 可以通过"在每个 token 后悄悄 think 一下"这种方式，从 pretraining corpus 里学会 general reasoning ability，并 transfer 到 unseen reasoning task。整篇 paper 的 elegance 在于：它把"reasoning"从一个 explicit task 退化成"language modeling 的 latent structure"，让 reasoning 学习融入 pretraining 本身。
