---
source_pdf: IAPO Information-Aware Policy Optimization for Token-Efficient Reasoning.pdf
paper_sha256: ebee4d2bd7c7ef05f9d9cd02330e86fb6f76a121a28f24686e7aed6edcb29cd0
processed_at: '2026-08-05T08:57:19-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 IAPO

## 问题是什么

RL post-training 之后的 LLM 比如DeepSeek-R1,推理时候特别啰嗦。同一个数学题,人类Ph.D.用264个token就能算出来,R1要写1658个token。关键这1658个token里有大量"废话"——重复题目条件、反复自我验证、来回绕圈说同一个意思。这些废话既花钱(推理cost跟sequence length平方相关)又没用。

大家早就发现这个问题了,之前的方法大概两类:

**第一类看长度**:短completion就多奖励,长的就少奖励。问题是你不能因为短就奖励啊,一个短的废话和一个短的关键推理步骤,长度一样,价值天差地别。

**第二类看位置**:第100个token之后的token就不给奖励了。问题是有时候最后的数值计算恰恰是最关键的,你一刀切把后面都砍了,把有用的也砍了。

这两类方法都是**不看内容**的,纯粹靠长度或位置猜哪些token重要。猜得自然不准。

## IAPO 的核心想法

这个paper说:我们应该真正去看每个token"对得出正确答案贡献了多少信息"。

用信息论的语言,就是算每个token $o_t$ 在给定前面所有token $(q, o_{<t})$ 的条件下,跟最终答案 $y$ 的 conditional mutual information。

直觉特别简单:如果生成这个token之后,模型对答案变得更确定了(entropy下降),说明这个token带来了新信息,是informative的,应该奖励;如果生成之后,模型对答案的确定程度没啥变化,说明这个token是废话,应该少奖励甚至不奖励。

为什么这个思路对?因为信息论有个chain rule:

$$I(y; o | q) = \sum_t I(y; o_t | q, o_{<t})$$

整个completion跟答案的mutual information,精确等于每个token的conditional MI之和。这就好比说,你要选k个最重要的token保留信息,那就选conditional MI最大的那k个,理论上最优。

## 怎么算这个MI

这是最难的部分。你没法直接拿到 $p(y | q, o_{\leq t})$ 这个分布,因为模型是autoregressive的,你只能拿到next-token probability。

IAPO用了个小技巧叫 **early-exit**:在任何中间位置,直接拼一个 `<answer>` 的postfix,逼模型在这个位置立刻给答案。这样你就能拿到"模型在这个中间步骤时对答案的distribution",算出entropy。

然后conditional MI就近似成:
- 生成token之前的answer entropy: $H(y | q, o_{<t})$
- 生成token之后的answer entropy: $H(y | q, o_{\leq t})$
- 两者之差就是这个token的informativeness

一个token如果让答案entropy从2.0降到0.5,说明它很有信息量;如果从2.0降到1.95,基本是废话。

## 工程上怎么不爆炸

naive实现完全不可行:每个token要两次forward pass,一次算之前的entropy一次算之后的,长度L的completion总复杂度是 $O(L^3 d)$。L=2000的话根本跑不动。

IAPO用了两个trick:

**KV-cache preloading**: 先对整条completion做一次forward pass把所有层的KV cache存下来。之后要算任何中间位置的entropy,不用重新跑前面的文本,直接load那部分KV cache,只对短postfix做attention。复杂度从 $O(L^3 d)$ 降到 $O((K^3 + L^2)d)$,其中K是postfix长度(就几个token),远远小于L。

**Chunk-wise forwarding**: 不要一个token一个token地算,把completion切成chunk,一个chunk内所有位置的MI一起batched算,amortize掉overhead。

## 还有个exploration的问题

只盯着informativeness优化有个风险:模型会迅速塌缩到一种特别短的推理模式,accuracy反而掉。而且光看当前trajectory的informativeness,不鼓励模型去试别的推理路径。

IAPO加了个exploration adjustment项:

- 如果这条completion是对的:给high-confidence token更多正奖励 → 强化现有正确路径,降低entropy,减少探索
- 如果这条completion是错的:给high-confidence token更多负奖励 → 惩罚"自信但错"的方向,升高entropy,鼓励去试别的

直觉就是:对的时候"巩固信心",错的时候"质疑自己"。confidence本身没有好坏,要看对错来决定是强化还是反转信号。

## 最终的advantage公式

每个token的advantage由三部分组成:

$$\tilde{A}_{i,t} = \text{sequence reward} + \alpha \cdot \text{token informativeness} + \beta \cdot \text{exploration adjustment}$$

第一项是GRPO原本的sequence-level reward,后两项是token-level的微调。$\alpha$和$\beta$是超参,实验里 $\alpha$ 增大让推理变短,$\beta$ 增大让accuracy提升但变长。两者正交:一个管"压缩",一个管"广度"。

## 理论上为什么长度会下降

paper证明了一个挺clean的结论:IAPO和GRPO相比,expected completion length的变化正比于 $L(o)$ 和 $S(o)$ 的covariance。$S(o)$ 是这条completion里informativeness-weighted的梯度累积。

直觉:如果一条completion越长,它里面包含的低informativeness token比例越高(都是废话),那 $S(o)$ 就越小。所以长completion对应低 $S(o)$,短completion对应高 $S(o)$,两者负相关,covariance是负的,长度就降下来了。

这个假设"越长废话越多"对数学推理这种任务基本成立。

## 实验结果怎么样

在GSM8K、MATH-500、DAPO-Math-17k三个数据集上,用Qwen2.5的0.5B、1.5B、7B三个规模测试:

- **GSM8K + 7B**: IAPO把推理长度从178 token降到111 token,压缩了37%,accuracy保持100%
- **MATH-500 + 1.5B**: IAPO的token efficiency (accuracy/length) 是最好的
- **DAPO-Math-17k + 7B**: IAPO比DAPO短一半多,accuracy略低但efficiency高很多

case study里同一个题目,IAPO用15个token算对,其他方法用51-105个token。

## 我的看法

**好的地方**: 用conditional MI做token-level credit assignment这个idea理论上无懈可击,信息论的chain rule给了它principled的decomposition。early-exit estimator虽然是个approximation但工程上work。相比之前看长度、看位置的方法,这是第一个真正看"内容信息量"的。

**可疑的地方**:

1. **early-exit假设**: 模型训练时没见过在中间位置直接输出 `<answer>`,这个postfix在早期prefix上给出的answer distribution可能很不准。虽然实验显示work,但理论上是个分布mismatch问题。

2. **entropy的尺度**: 早期prefix的answer entropy通常很高(模型还没想清楚),后期很low。这意味着前几个token的MI天然可能被高估——因为entropy从高到低下降快。这会不会让IAPO系统性地高估early tokens?

3. **informativeness单调递减假设**: 对于难题,长chain-of-thought里每个步骤可能都是informative的。对简单题才长=废话多。这个假设是problem-dependent的。

4. **exploration term的magnitude**: $\pi_{\theta_{old}}(o_t | ...)$ 是个概率值,在(0,1)之间,量级很小。$\beta$ 要设到 $10^{-6}$ 才work,说明这个term非常subtle,稍微调大可能就出问题。

总的来说,这个工作把"什么是informative token"这个问题用信息论语言给了一个principled的回答。比之前纯靠长度和位置的heuristic确实进了一步。工程上把naive $O(L^3)$ 的MI估计降到可接受范围也值得学习。但early-exit estimator的approximation quality、理论假设的适用范围,都还是有空间改进的。

参考:
- 代码: https://github.com/YinhanHe123/IAPO
- GRPO: https://arxiv.org/abs/2402.03300
- DAPO: https://arxiv.org/abs/2503.14476
- Let's Verify Step by Step: https://arxiv.org/abs/2305.20050

---

<answer>"的postfix,让模型立即给出答案。然后从这个answer logits算entropy。

**计算效率问题**: naive方法每个token需要两次forward pass,总复杂度$O(L^3 d)$。用KV-cache preloading可以变成$O((K^3 + L^2)d)$,其中K是postfix长度。chunk-wise forwarding进一步amortize。

**Advantage公式**:
$$\tilde{A}_{i,t} = \text{norm}(r_i, \mathbf{r}) + \alpha \cdot \text{norm}(s_{i,t}, \mathbf{s}_i) + \beta \cdot \text{norm}(c_{i,t}, \mathbf{c}_i)$$

其中:
- 第一项:sequence-level reward (GRPO)
- 第二项:token-level informativeness (MI)
- 第三项:token-level exploration adjustment

**Exploration adjustment**:
$$c_{i,t} = \pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t}) \text{ if correct, else } -\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})$$

这个设计很有意思 - 对于correct completion,advantage正比于model confidence,所以高confidence token得到更大reward,reduce entropy。对于incorrect,advantage负正比于confidence,所以high confidence token得到更大penalty,encourage exploration to other paths。

**Theoretical analysis (Theorem 4.1)**:
$$L_{IAPO} - L_{GRPO} \propto \text{Cov}_{o \sim p_{GRPO}}(L(o), S(o))$$

其中$S(o)$是informativeness-weighted gradient accumulation。如果completions越长包含越多低informativeness tokens,$S(o)$和$L(o)$负相关,covariance为负,所以IAPO长度小于GRPO。

让我现在写一个详细的回应,包含:
- paper的核心motivation和problem formulation
- IAPO的方法详解,包括公式和直觉
- Efficient estimation的技术细节
- Theoretical analysis
- Experimental results
- 我自己的intuition和critique

让我用中文写,但技术名词用英文。要避免"不是...而是"句式。

让我也想想可能的limitations和open questions:
- Early-exit estimator的准确性 - 它假设模型能在中间step给出合理answer,但模型训练时可能没有这种能力
- 假设average informativeness per token单调递减 - 这个假设可能不总成立
- Exploration adjustment的理论分析有点hand-wavy

Web links for reference - 我应该提供一些相关工作的链接。

让我开始写回应。要尽可能多,不要错过细节。

技术细节包括:
1. GRPO objective公式和符号
2. Conditional MI的定义和chain rule
3. Early-exit estimator的实现
4. KV-cache preloading的复杂度分析
5. Theorem 4.1的证明思路
6. 实验设置和结果

我注意到paper中有些可能有问题的地方,比如Table 1里某些IAPO结果似乎不是最好的,我需要诚实讨论。

让我写一个非常详细的回应。# IAPO: Information-Aware Policy Optimization — 深入技术解析

## 1. Motivation 与 Problem Formulation

### 1.1 GRPO的verbosity问题

GRPO (Group Relative Policy Optimization) 是DeepSeekMath提出的RL post-training方法。对于query $q$,从frozen policy $\pi_{\theta_{old}}$ 采样G个completions $\{o_i\}_{i=1}^G$,每个completion获得reward $r_i$,然后做group-wise normalization得到advantage:

$$\tilde{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

其中 $\mathbf{r} = \{r_i\}_{i=1}^G$。这个 $\tilde{A}_i$ 被uniformly分配给completion $o_i$ 中所有token位置。GRPO的objective:

$$J_{GRPO}(\theta) = \mathbb{E}_{q, \{o_i\}}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left(\rho_{i,t}(\theta)\tilde{A}_{i,t}, \text{clip}(\rho_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon)\tilde{A}_{i,t}\right) - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})\right]$$

其中 $\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}$ 是importance ratio。

**关键问题**:由于 $\tilde{A}_{i,t} = \tilde{A}_i$ for all $t$,整个completion里每个token无论是否informative,获得相同的policy gradient信号。这导致模型无法区分"essential reasoning step"和"redundant verification",于是post-trained model倾向于verbose reasoning。Figure 1展示了DeepSeekR1-Distilled-Qwen-1.5B在MATH-500上生成1658 tokens而人类Ph.D.只用264 tokens就能达到perfect accuracy的惊人gap。

### 1.2 现有方法的局限

Paper把现有token-efficient RL方法分两类:

**Length-based methods** (如GFPO, DAPO):对shorter completions整体给予更高advantage。问题:short不等于informative,可能误伤关键short reasoning。

**Position-based methods** (如S-GRPO):对position index大于某threshold的token给予zero advantage。问题:later tokens有时也是informative的(比如final numerical computation),会被误penalize。

两类方法都是**content-agnostic**的。

### 1.3 Problem 1形式化定义

$$\max_\theta \frac{\mathbb{E}_{q \sim Q, o \sim \pi_\theta}[\mathbb{I}\{o \text{ is correct}\}]}{\mathbb{E}_{q \sim Q, o \sim \pi_\theta}[|o|]} \quad \text{s.t.} \quad \mathbb{E}[\mathbb{I}\{o \text{ is correct}\}] \geq \tau$$

这里分子是accuracy,分母是平均completion length,目标是maximize accuracy-per-token ratio。$\tau$ 是minimum effectiveness threshold,防止退化成"很短但全错"的策略。

这是一个constrained optimization,把token efficiency和accuracy preservation都explicitly建模。

## 2. IAPO方法核心

### 2.1 Informativeness Level: Conditional Mutual Information

定义token $o_{i,t}$ 对最终答案 $y_i$ 的informativeness:

$$s_{i,t} = I(y_i; o_{i,t} | q_i, o_{i,<t})$$

为什么选conditional MI?有两个理由:

**(1) Semantic alignment**: 从信息论角度,$I(Y; X | Z) = H(Y|Z) - H(Y|X,Z)$ 衡量在已知 $Z$ 条件下,观察 $X$ 后对 $Y$ 的不确定性的减少。这里 $Z = (q, o_{i,<t})$,$X = o_{i,t}$,$Y = y_i$。如果token $o_{i,t}$ 给答案带来了新信息,$s_{i,t}$ 大;如果只是重复已知信息,$s_{i,t}$ 小。

**(2) Selective decomposition**: 由MI的chain rule:
$$I(y_i; o_i | q) = \sum_t I(y_i; o_{i,t} | q, o_{i,<t})$$

也就是说,整个completion与答案的MI可以精确分解为每个token的conditional MI之和。对于任何token budget $k$,$\hat{o}_i^k = \arg\max_{\tilde{o}_i^k \subset o_i, |\tilde{o}_i^k|=k} I(y_i; \tilde{o}_i^k | q)$ 的解就是top-k个 $s_{i,t}$ 最大的token。这给了一个principled的"信息保持子序列"定义。

### 2.2 Exploration Adjustment

只用informativeness有两个问题:
1. **Premature trajectory collapse**: 模型可能很快收敛到过于简洁但accuracy下降的pattern
2. **忽略exploration**: informativeness只在当前trajectory内评估,不鼓励探索其他reasoning path

Paper引入token-level exploration adjustment:

$$c_{i,t} = \begin{cases} \pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t}) & \text{if } o_i \text{ is correct} \\ -\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t}) & \text{if } o_i \text{ is incorrect} \end{cases}$$

直觉: $\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})$ 是model对该token的confidence。

- 对于correct completion:high-confidence token获得更大正advantage → amplify现有正确路径,reduces policy entropy around high-confidence states → suppresses exploration (consolidate)
- 对于incorrect completion:high-confidence token获得更大负advantage → penalty inverts signal,push model away from当前confident但错误的方向 → increases policy entropy → encourages exploration to alternative paths

这个设计很有趣,因为它把confidence的符号和correctness耦合起来,实现了"正确时强化信心,错误时质疑信心"的双向调节。

### 2.3 Token-wise Advantage Assignment

最终的advantage:

$$\tilde{A}_{i,t} = \underbrace{\text{norm}(r_i, \mathbf{r})}_{\text{seq-level reward}} + \alpha \underbrace{\text{norm}(s_{i,t}, \mathbf{s}_i)}_{\text{token-level info}} + \beta \underbrace{\text{norm}(c_{i,t}, \mathbf{c}_i)}_{\text{token-level explo}}$$

其中 $\text{norm}(x, \mathbf{v}) = \frac{x - \text{mean}(\mathbf{v})}{\text{std}(\mathbf{v})}$,$\alpha, \beta$ 是hyperparameter。

注意norm是对每个completion内做的,所以 $s_{i,t}$ 是相对该completion内其他token的informativeness。这点很重要,否则不同completion的MI量级差异会dominate。

## 3. Efficient Conditional MI Estimation

这是paper的工程核心,因为naive实现完全不可行。

### 3.1 Early-Exit Estimator

挑战:conditional MI需要access $p(y_i | q, o_{i,\leq t})$,但autoregressive generation只给next-token distribution。如何拿到"在中间step的answer distribution"?

利用信息论恒等式:
$$I(y_i; o_{i,t} | q, o_{i,<t}) = H(y_i | q, o_{i,<t}) - H(y_i | q, o_{i,\leq t})$$

技巧:对任意prefix $(q, o_{i,\leq t})$,append一个lightweight postfix prompt比如 `"<answer>"`,强制模型在该位置立即生成answer,不再继续reasoning。然后从该位置的answer logits提取answer distribution,计算entropy。

具体流程:
- 计算 $H(y_i | q, o_{i,<t})$:feed $(q, o_{i,<t}) + \text{postfix}$ 到LLM,在最后一个postfix token位置取answer logits → 得到distribution → entropy
- 计算 $H(y_i | q, o_{i,\leq t})$:feed $(q, o_{i,\leq t}) + \text{postfix}$,同样取answer logits entropy
- $s_{i,t} \approx H(y_i | q, o_{i,<t}) - H(y_i | q, o_{i,\leq t})$

直觉:如果一个token显著降低了answer entropy(让模型更确定答案),它是informative的;如果token几乎没有改变answer entropy(只是重复说已知信息),它是redundant的。

### 3.2 KV-Cache Preloading

Naive复杂度:对每个token位置 $t$,需要两次forward pass,每次处理长度 $t$ 的prefix。总复杂度:
$$O\left(\sum_{l=1}^{L} l^2 d\right) = O(L^3 d)$$

其中 $L = |o_i|$,$d$ 是embedding dimension。这个 $O(L^3 d)$ 对于reasoning动辄上千token完全不可行。

**KV-cache preloading技巧**:
1. 对完整completion $(q, o_i)$ 做一次forward pass,存储所有层的KV cache
2. 这些KV cache完全encode了所有autoregressive prefix $(q, o_{i,\leq t})$
3. 当要计算position $t$ 处的entropy时,**不再feed textual prefix**,而是直接load对应的KV cache,只对postfix prompt做attention计算

关键观察:postfix prompt $K$ 长度远小于prefix长度 $L$(比如postfix就几个token),所以每次entropy evaluation的cost只依赖于 $K$ 而不是 $L$。

复杂度从 $O(L^3 d)$ 降到:
$$O((K^3 + L^2) d)$$

其中 $K$ 是postfix长度。由于 $K \ll L$,这是巨大改进。

### 3.3 Chunk-wise LLM Forwarding

虽然KV-cache preloading消除了redundant prefix computation,但sequential per-token invocation仍然有overhead。Chunk-wise forwarding将completion分成contiguous chunks,对一个chunk内所有token位置batched forward pass:共享同一prefix KV cache,把对应postfix batched起来。这把per-token overhead除以chunk size $C$。

最终复杂度:
$$O\left(K^3 d \cdot \frac{L}{C} + \frac{K^2 L^2 d}{2C} + L^2 d\right)$$

第一项:每个chunk的postfix attention计算,除以 $C$ 因为batching
第二项:cache update的cost,除以 $C$
第三项:初始prefix forward的cost,irreducible

## 4. Theoretical Analysis

### 4.1 Theorem 4.1: Length Reduction

设 $L_{GRPO}$ 和 $L_{IAPO}$ 分别是GRPO和IAPO one-step update后的expected completion length。对sufficiently small step size $\eta$:

$$L_{IAPO} - L_{GRPO} \propto \text{Cov}_{o \sim p_{GRPO}}(L(o), S(o))$$

其中 $S(o) = \sum_{t=1}^{L(o)} g_t(q, o_{i,t})$,$g_t(q, o_{i,t}) = \nabla_\theta \log \pi_\theta(o_{i,t}|q)|_{\theta=\theta_{GRPO}} \Delta\theta_s$,$\Delta\theta_s$ 是IAPO advantage中informativeness term引入的参数更新。

证明思路(在Appendix B):
1. 写出 $\theta_{IAPO} = \theta_{GRPO} + \eta \Delta\theta_s$
2. 对 $\log \pi_{\theta_{IAPO}}$ 在 $\theta_{GRPO}$ 附近做Taylor展开
3. Linearize: $\pi_{\theta_{IAPO}}(o_{i,t}|q) \approx \pi_{\theta_{GRPO}}(o_{i,t}|q)(1 + \eta g_t)$
4. 整个trajectory probability: $p_{IAPO}(o) \approx p_{GRPO}(o)(1 + \eta S(o))$
5. Normalize: $p_\eta(o) = \frac{p_{GRPO}(o)(1 + \eta S(o))}{1 + \eta \mathbb{E}[S]}$
6. 算 $\mathbb{E}_{o \sim p_\eta}[L(o)]$,first-order展开得到covariance term

**关键intuition**: $S(o)$ 是informativeness-weighted gradient accumulation。如果completion里high-informativeness token比例高,$S(o)$ 大;如果都是low-informative的redundant verification,$S(o)$ 小。

### 4.2 Corollary 4.2

**Assumption**: 在GRPO policy下,average informativeness per token随completion length单调递减。

这个假设mild且natural:越长completion越倾向于累积redundant verification, average informativeness下降。

在该假设下,$L(o)$ 和 $S(o)$ 负相关,$\text{Cov} < 0$,所以:
$$L_{IAPO} < L_{GRPO}$$

### 4.3 Exploration Adjustment分析

由policy gradient对entropy的影响(参考Cui et al. 2025):
$$H(\pi_{IAPO}(\cdot|s)) - H(\pi_0(\cdot|s)) \approx -\eta \text{Cov}_{o_t \sim \pi_0(\cdot|s)}(\log \pi_0(o_t|s), A(s, o_t))$$

- Correct completion: $A(s, o_t) = \pi_0(o_t|s)$,则 $\text{Cov}(\log \pi_0, \pi_0) > 0$ (因为 $x \log x$ 是凸函数) → entropy下降 → suppresses exploration
- Incorrect completion: $A(s, o_t) = -\pi_0(o_t|s)$,covariance符号反转 → entropy上升 → encourages exploration

这个分析很clean,把exploration控制变成了一个entropy modulation问题。

## 5. Empirical Study

### 5.1 实验设置

- **Models**: Qwen2.5-0.5B/1.5B/7B-Instruct
- **Datasets**: GSM8K (grade-school math), MATH-500 (competition math), DAPO-Math-17k (longer solutions)
- **Baselines**: DAPO, GFPO, GTPO, S-GRPO
- **Metrics**: Pass@k (accuracy with k trials), Length@k (avg length), Ratio@k = Pass@k / Length@k (token efficiency)
- **Hardware**: 4-8 H100 GPUs, DeepSpeed ZeRO Stage 2/3
- **Training**: AdamW, lr=1e-6, G=8 group completions, KL coeff=0.001

### 5.2 主要结果

从Table 1看核心结果:

**Qwen2.5-7B on GSM8K**:
- Base: P@16=1.0, L@16=177.93
- IAPO: P@16=1.0, L@16=110.65 → 37.8%长度缩减,accuracy保持
- GFPO: P@16=0.978, L@16=175.85 → accuracy掉,length几乎没减
- GTPO: P@16=1.0, L@16=163.45 → length缩减但不如IAPO

**Qwen2.5-1.5B on MATH-500**:
- IAPO: P@32=0.978, L@32=180.44, R@32=5.42e-3 (best)
- DAPO: P@32=0.978, L@32=220.74, R@32=4.43e-3

**Qwen2.5-7B on DAPO-Math-17k**:
- IAPO: P@32=0.689, L@32=194.50, R@32=3.54e-3 (best)
- DAPO: P@32=0.733, L@32=431.44, R@32=1.70e-3

值得注意:IAPO在7B model + DAPO-Math-17k上accuracy (0.689) 略低于DAPO (0.733),但token efficiency显著更高。这是accuracy-efficiency trade-off。

### 5.3 Ablation Study

两个变体:
- **IAPO-NI**: 移除informativeness term (只有GRPO + exploration)
- **IAPO-NE**: 把conditional MI换成next-token entropy reduction

$$H(o_{i,t}|q, o_{i,<t}) - H(o_{i,t+1}|q, o_{i,\leq t})$$

这是local entropy change,不是对answer的information。

Fig 5结果:IAPO > IAPO-NE > IAPO-NI,说明:
1. Conditional MI确实比next-token entropy更有信息量
2. 即使是imperfect MI estimator也比没有token-level informativeness好

### 5.4 Parameter Analysis (Fig 6)

$\alpha$ (informativeness coefficient) 在 $\{10^{-6}, 10^{-4}, 10^{-2}, 1\}$ 范围sweep:
- $\alpha$ 增大 → length单调递减 (good)
- Pass@32也递减但慢得多 (favorable trade-off)

$\beta$ (exploration coefficient):
- $\beta$ 增大 → Pass@32单调提升 (good,鼓励exploration确实提高accuracy)
- 但length也显著增加 (token consumption上升)

这说明 $\alpha$ 控制"压缩",$\beta$ 控制"广度",两者正交。

### 5.5 Case Study

Fig 7展示同一GSM8K问题不同方法的reasoning:

- **GFPO**: 重复问题给定的信息 (10 tokens restatement + 14 tokens relation)
- **GTPO**: 反复重述策略
- **DAPO/S-GRPO**: 冗余地identify "what quantities does problem require"
- **IAPO**: 15 tokens直接得出正确答案,3.4x-7x shorter

## 6. 我的Intuition与Critique

### 6.1 核心insight的优雅性

Conditional MI的decomposition $\sum_t I(y; o_t | q, o_{<t}) = I(y; o | q)$ 是非常优美的信息论identity。把它作为token-level advantage的principle,理论上无懈可击。比起length-based和position-based,这真正capture了"信息"的本质。

### 6.2 Early-exit estimator的潜在问题

这个estimator有一个implicit assumption: **模型在任何prefix都能合理生成answer distribution**。但是:

1. **训练时模型没见过 `<answer>` 在中间位置**:这可能在reasoning chain早期位置造成distribution mismatch
2. **Answer distribution的calibration**:早期prefix可能给非常uniform的answer distribution,后期sharp。这可能导致前几个token的MI都被高估,因为entropy从高变低下降快
3. **Multi-modal answer distribution**:对于某些数学问题,答案可能是数值,answer distribution是discrete但空间大。如何定义"answer entropy"需要慎重

实验中Case Study (Fig 14) 显示MI heatmap确实能识别informative tokens,说明estimator在实践中work,但理论上可能imperfect。

### 6.3 Exploration adjustment的形式

$c_{i,t} = \pm \pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})$ 这个形式有个微妙问题: probability值 $\pi \in (0, 1)$,所以 $c$ 的magnitude很小。即使经过normalization,exploration term可能被informativeness term dominate (informativeness是entropy difference,可能也是小量但量级可能不同)。

从实验看 $\beta = 10^{-6}$ 时已经有效,说明确实需要很小的coefficient,但这也意味着exploration调节非常subtle。

### 6.4 Theoretical analysis的mild assumption

Corollary 4.2假设"average informativeness per token随length单调递减"。这个假设直觉上成立,但严格说不是trivial:

- 对于"难题"需要长chain-of-thought,每个step都是informative的
- 对于"简单题"长completion才是redundant的

也就是说informativeness-density vs length的关系可能是problem-dependent。但paper实验显示overall trend成立。

### 6.5 Comparison with prior work的position in literature

IAPO和最近一些工作的关系:
- **T-SPMO** (Lee & Tong 2025): token-level prefix matching credit assignment
- **TEMPO/TreePO**: tree structure for value estimation
- **CAPO**: LLM as generative PRM
- **High-entropy forking tokens** (Wang et al. 2025): 识别high-entropy作为critical decision points

IAPO的独特之处:直接用answer-level conditional MI,不是generation entropy或tree structure。这是更"goal-aware"的signal。

## 7. 相关工作与延伸阅读

Paper在Section 6和Appendix A讨论了information theory in RL的历史:
- **Still & Precup 2012**: Boltzmann exploration emerges from minimizing information cost
- **Schmidhuber 2015**: Algorithmic MI between model and controller
- **Chen & Jiang 2019**: Information-theoretic lower bounds in batch RL

这些工作使用information theory分析RL的bound和exploration,但都not targeting token efficiency in LLM reasoning。IAPO的创新是把conditional MI作为per-token credit assignment。

## 8. Open Questions与未来方向

1. **MI estimator的改进**: early-exit是cheap approximation,更accurate的estimator (e.g. contrastive MI estimation)可能进一步提升signal-to-noise ratio
2. **跨modal generalization**: 能否推广到vision-language model reasoning?answer distribution怎么定义?
3. **Online vs offline estimation**: paper是offline估计MI(在rollout后),online估计可能更efficient
4. **Connection to speculative decoding**: 如果能预测哪些token是informative,可以指导speculative decoding的draft model
5. **MI estimator的训练时干扰**: 如果post-training过程中模型behavior shift,early-exit estimator的postfix `<answer>`是否还work?
6. **Multi-step reasoning tasks**: 对于code generation, theorem proving这种更复杂任务,informativeness的定义可能需要扩展

## 9. 实验细节中值得注意的点

- **Reasoning length measurement**: Fig 1用whitespace-delimited word counts (人类vs模型公平比较),Table 1用native tokenizer (反映实际compute cost)。这是合理的context-dependent choice。
- **Training configurations**: 0.5B/1.5B用ZeRO Stage 2,7B用ZeRO Stage 3。7B model的batch size减半(16→8)因为memory。这反映了MI estimation的额外memory cost。
- **Wall-clock inference time**: Appendix C.2.3报告IAPO训练后inference time reduction up to 17.7%。这是实际production-relevant metric,而不仅仅是token count。

## 10. 总结

IAPO是一个**principled**的token-efficient RL post-training框架。它的核心贡献:

1. **理论**: 把token-efficient post-training formalize为accuracy-per-token maximization,用conditional MI作为principled token-level informativeness measure,提供了length reduction的covariance-based理论分析。

2. **方法**: 双模块设计(information-aware advantage shaping + efficient MI estimation),early-exit estimator + KV-cache preloading + chunk-wise forwarding让MI estimation在LLM scale上tractable。

3. **实验**: 在3个数学reasoning dataset、3个model scale上consistently achieve best或runner-up token efficiency,ablation study证实各component的贡献。

它的核心limitation是estimator的近似性 (early-exit假设) 和理论分析中的mild assumption (informativeness单调性)。但作为一个first principled step towards content-aware token-efficient reasoning,IAPO开辟了information-theoretic credit assignment这个方向。

参考链接:
- Paper code: https://github.com/YinhanHe123/IAPO
- GRPO original paper (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Let's Verify Step by Step (PRM): https://arxiv.org/abs/2305.20050
- DAPO: https://arxiv.org/abs/2503.14476
- Qwen2.5 technical report: https://arxiv.org/abs/2407.10671
- PPO: https://arxiv.org/abs/1707.06347
- Cover & Thomas, Elements of Information Theory (经典教材,MI chain rule等定义)

这个工作对我来说最重要的启发是:**credit assignment是RL post-training的核心瓶颈之一**,而information theory提供了一个principled的语言去思考"什么是有用的computation"。把它扩展到更长horizon、更复杂task、更准确estimator,是有大量空间的direction。
