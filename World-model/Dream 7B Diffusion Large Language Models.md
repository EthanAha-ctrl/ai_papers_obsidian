---
source_pdf: Dream 7B Diffusion Large Language Models.pdf
paper_sha256: 461f5ac618e86dd398af3662958178c3c243ee97b7e5ee9c20f5ed6399107a0d
processed_at: '2026-08-03T23:15:58-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的, Andrej. 抛开复杂的数学符号, 我们从 system design 和 information flow 的角度, 用最直白的 intuition 来拆解 Dream 7B 这篇 paper.

核心要回答的问题就是: 为什么我们要抛弃 GPT 这种 left-to-right 的 AR paradigm, 转而去搞 Diffusion LLM? 以及 Dream 到底是怎么把这件事做成的?

## 1. Paradigm 直觉: 雕塑家 vs 打字员

当前所有的主流 LLM (GPT-4, LLaMA, DeepSeek) 都是 AR 模型. AR 模型的运作机制就像是一个盲打字员, 只能根据前面的已经打出的 token, 预测下一个 token. 公式 $p_\theta(\mathbf{x}) = p_\theta(\mathbf{x}^1) \prod_{n=2}^{N} p_\theta(\mathbf{x}^n | \mathbf{x}^{1:n-1})$ 描述的就是这个过程. $\mathbf{x}^{1:n-1}$ 是前 $n-1$ 个 token, 模型只看这些, 去猜第 $n$ 个. 这带来的致命弱点是: 一旦早期生成一个糟糕的 token, 后面全盘皆输, 无法回头. 在做 Sudoku 或者 Trip Planning 这种需要全局约束满足的任务时, AR 模型经常陷入死胡同.

Diffusion model 的 paradigm 完全不同. 它像是一个雕塑家. 初始状态是一块完全混乱的石头 (全是 `<MASK>` token 的 sequence $\mathbf{x}_T$). 模型通过 iterative denoising, 每次扫一眼全局, 把几个最确定的位置雕出来, 然后基于现在的全局状态, 再去雕其他位置. 公式 $p_\theta(\mathbf{x}) = \sum p(\mathbf{x}_T) \prod p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 里的 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 就是在给定当前半成品 $\mathbf{x}_t$ 的情况下, 推断上一阶段更完整一点的状态. 因为 attention 是 bidirectional 的, 每次预测都能同时看到左右两侧的 context. 如果某一步填错了, 下一步迭代时还可以通过全局 context 修正它. 这天然具备 backtracking 和 global planning 的能力.

## 2. AR Initialization: 偷天换日

要把 Diffusion LLM 训到 7B 规模, 从头训需要海量的算力和数据. LLaDA 花了 2.3T tokens 从头训, 效果还打不过 AR model. Dream 7B 的核心 trick 之一就是直接偷用 Qwen2.5-7B 的预训练权重.

但是这里有一个 architecture misalignment. AR 模型在 position $i$ 的 hidden state $h_i$, 被训练用来预测 position $i+1$ 的 token. 传统 Diffusion 试图用 $h_i$ 预测 position $i$ 本身(因为 $i$ 被 mask 了). 这完全打破了 AR 模型固有的 representation.

Dream 沿用了 DifuLLaMA 的 Shift Operation. 哪怕变成了 Diffusion, 也强行保留 AR 的习惯: 用 $h_i$ 预测 $i+1$. 
从 Figure 2 看, 当输入是 `tok1 tok2 <M> <M> tok5` 时:
- Position 3 虽然是 `<M>`, 但它的 hidden state $h_3$ 依然被用来预测 position 4 的 token.
- Position 4 的 hidden state $h_4$ 被用来预测 position 5 的 token.

这种 shift 机制完美复用了 AR model 在 18T tokens 上学到的 left-to-right 语言学知识. 结合 carefully tuned learning rate, 模型只需要学如何从 full attention 中整合双向信息, 极大地加速了训练. Dream 只用了 0.6T tokens 就超越了 LLaDA 的 2.3T. 
*(Reference: DifuLLaMA https://arxiv.org/abs/2410.18514)*

## 3. CART: 量身定制的难度系数

在标准 discrete diffusion 的公式 $L(\theta) = -\mathbb{E} [w(t) \sum \log p_\theta(\mathbf{x}_0^n | \mathbf{x}_t)]$ 中, $w(t)$ 是 time-dependent 的 weight. 当 $\alpha_t = 1-t$ 时, $w(t) = 1/t$. 意味着越接近 clean data ($t \to 0$), weight 越大.

问题是, 标准做法给整个 sentence 采样一个 $t$, 所有被 mask 掉的 token 共享同一个 weight. 想象一句话 `The <M> <M> <M> sky is <M> blue`. 前面三个 `<M>` 旁边只有 "The", context 极度稀疏, 填起来极难; 后面那个 `<M>` 旁边有 "sky is" 和 "blue", context 丰富, 填起来极易. 给它们一样的 weight 极其不合理.

Dream 提出 CART (Context-Adaptive noise Rescheduling at Token-level), 把 weight 细化到 token level:
$$w(t, \mathbf{x}_t, n) = \frac{1}{2} \sum_{i=1}^{N} \mathbf{1}_{[\mathbf{x}_t^i \neq \text{MASK}]} \text{Geo}(p, |n-i|-1)$$
变量解释: $n$ 是当前 masked token 的 position, $i$ 是 clean token 的 position, $|n-i|$ 是它们之间的距离. $\text{Geo}(p, k)$ 是 geometric distribution 的概率质量函数. $p$ 控制衰减速度.

这个公式的物理直觉非常清晰: 遍历所有的 clean token $i$, 看它们距离当前 masked token $n$ 有多远. 距离越近, 贡献的信息越多. 把这些信息贡献按 geometric distribution 衰减后加起来, 就是这个 masked token 实际获得的 context informationness. 周围 clean token 越多, $w$ 越大, 相当于模型对这个位置的预测置信度越高, 在 loss 里的权重就越大. 这把 sequence-level 的宏观 $w(t)$, 变成了 token-level 的微观 $w(t, \mathbf{x}_t, n)$.
*(Reference: Continuous time framework https://arxiv.org/abs/2205.14987, Ou et al. conditional distributions https://arxiv.org/abs/2504.11581)*

## 4. Planning 的碾压性优势

在 Table 1 里, 最震撼的数据是:
- Sudoku: Dream 81.0 vs Qwen2.5 21.0
- Countdown: Dream 16.0 vs Qwen2.5 6.2
- Trip planning: Dream 17.8 vs Qwen2.5 3.6

甚至在 Figure 5 的 Countdown3 任务上, 7B 的 Dream 打败了 671B 的 DeepSeek-V3.

为什么会这样? Planning 任务的数学本质是 Constraint Satisfaction Problem. 你要在多个变量之间找到一组赋值, 满足所有给定的全局约束. 
AR 模型生成时, 早期位置的 token 只能看到左边, 完全不知道右边的约束. Bachmann & Nagarajan (2024) 指出, 这种 left-to-right compositionality generalization 是有极限的. 一旦头几个 token 选错, 违反了全局约束, 整个生成过程就崩溃了, 因为没法 backtrack.

Diffusion 模型天然支持 implicit search. 每一步 denoising, 模型的 full attention 都能看到当前所有已经 unmask 的位置, 相当于同时感知所有的局部约束. 在填入一个新的 token 时, 模型是在当前的全局上下文下做决策. 如果某一步生成了违反约束的 token, 在下一个 denoising step, 模型可以通过 bidirectional context 发现这个矛盾, 并在后续的 refinement 中修正它. 这种 parallel refinement 的机制, 完美匹配了 planning 任务需要的 global coherence. 
*(Reference: Ye et al. Beyond Autoregression https://arxiv.org/abs/2504.10662, Implicit search via discrete diffusion https://arxiv.org/abs/2504.11446)*

## 5. Inference 的维度扩展: Test-time Scaling

AR 模型推理时, 算力花在生成长度上 (chain-of-thought). Diffusion 模型多了一个正交的维度: diffusion steps $t$. 

Figure 7 展示了在 Countdown 任务上, 调整 timesteps 带来的 quality-speed trade-off. 增加 denoising steps, 模型有更多机会去 refine 和 correct, 质量上升, 速度变慢; 减少 steps, 速度变快. 在 5-20 steps 的配置下, Dream 同时在 quality 和 speed 上超越了 Qwen2.5.

这是一个极其重要的 system feature. 在 OpenAI o1 和 DeepSeek R1 时代, 我们通过让模型吐出超长的 reasoning tokens 来实现 test-time scaling. Diffusion 提供了另一种思路: 在 latent space 里通过增加 iterative denoising steps 来 scaling test-time compute. 这种方法不需要输出海量的自然语言, 理论上计算效率更高. Geiping et al. (2025) 也在探讨这种 latent reasoning 的潜力.
*(Reference: OpenAI o1 system card https://arxiv.org/abs/2412.16720, DeepSeek R1 https://arxiv.org/abs/2501.12948, Scaling test-time compute with latent reasoning https://arxiv.org/abs/2502.05171)*

## 6. 任意顺序生成的魔法

AR 模型受限于 causal mask, 只能 left-to-right. 这导致它做 infilling (在文本中间挖空填字) 时非常别扭, 要么训练专门的 task, 要么用诡agic trick.

Diffusion 模型的 reverse process $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 本质上是在处理一个随机分布着 `<MASK>` 的序列. 模型去预测所有 `<MASK>` 位置的概率分布. 这意味着:
1. **Completion**: 给个前缀, 模型把后面所有位置填上.
2. **Infilling**: 给个开头和结尾, 中间填上. 甚至可以约束生成的文本必须精确匹配某个结尾句子, 这在 AR 里极难做到.
3. **Configurable decoding order**: 可以强制它从左到右 unmask, 也可以完全随机 unmask, 甚至可以根据模型的 confidence, 每次先 unmask 最确定的那个位置. 这种 flexibility 在 AR 模型里是 unimaginable 的.

## 7. 联想与 Future Implications

站在 system architecture 的高度看, Dream 7B 证明了 discrete diffusion 在 scale 上是 viable 的. 这引发了很多深层的联想:

1. **Post-training 的挑战**: Table 2 显示 Dream-Instruct 跟 Qwen2.5-Instruct 还有差距. RLHF/DPO 在 diffusion 模型上怎么做? AR 模型的 RLHF 基于 token level 的 log-likelihood, diffusion 模型的 reverse process 是一个 stochastic MDP, 这可能需要一套全新的 policy gradient 算法, 或者基于 score matching 的 RL. 这是一个巨大的 blue ocean.

2. **Credit Assignment**: 在 AR 模型里, 如果生成结果不好, 很容易 blame 到某个具体的 token 上. 在 Diffusion 里, 生成是 parallel 的, 多个 token 同时 denoising, 究竟哪个 step 的哪个 prediction 导致了最后的 bad output? 这种 spatio-temporal credit assignment 非常棘手.

3. **Long Context**: AR 模型的 KV cache 随 context length 线性增长, 推理极耗内存. Diffusion 模型每次 denoising 是 full attention, 而且迭代次数固定. 如果用 block diffusion (Arriola et al. 2025), 理论上可以更好地处理 long context generation, 因为它不维护随长度增长的 KV cache.
*(Reference: Block diffusion https://arxiv.org/abs/2503.09573)*

4. **System 2 Thinking**: 人类的思考过程经常是跳跃的, 先打草稿, 再全局修改, 最后定稿. Diffusion 的 iterative refinement 极其类似这种 System 2 thinking. 未来可能涌现出基于 diffusion steps 的 CoT, 把 reasoning 消化在 latent iterative steps 里, 最终一次性输出完美的结果.

Dream 7B 本质上是在告诉社区: Autoregression 绝对不是 sequence modeling 的唯一终点. 利用 AR model 的现成知识作为起跳板, 结合 bidirectional context 的全局视野, 我们能构建出在 planning 和 inference flexibility 上超越当前 paradigm 的下一代 LLM.

---

# Dream 7B: 扩散大语言模型深度技术解析

## 1. 核心思想与定位

Dream 7B是当前最强大的开源扩散大语言模型,它的核心创新在于证明discrete diffusion可以在通用语言任务上接近AR(autoregressive)模型Qwen2.5-7B的水平,同时在planning任务上展现显著优势。

**关键intuition**: AR模型本质是"从左到右逐token预测",而diffusion模型是"从全噪声并行去噪",这种bidirectional context让模型能"看到全局"再决策,所以特别适合需要全局约束满足的planning任务。

参考链接:
- 原论文: https://arxiv.org/abs/2503.06357 (2025年4月)
- 项目主页: https://hkunlp.github.io/blog/2025/dream/
- 代码: https://github.com/DreamLM/Dream
- LLaDA (前SOTA): https://arxiv.org/abs/2502.09992
- DifuLLaMA: https://arxiv.org/abs/2410.18514 (Gong et al., ICLR 2025)

---

## 2. 数学基础:从AR到Discrete Diffusion

### 2.1 AR模型分解(公式1)

$$p_{\theta}(\mathbf{x}) = p_{\theta}(\mathbf{x}^1) \prod_{n=2}^{N} \underbrace{p_{\theta}(\mathbf{x}^n | \mathbf{x}^{1:n-1})}_{\text{progressive left-context prediction}}$$

变量解释:
- $\mathbf{x} = (\mathbf{x}^1, \dots, \mathbf{x}^N)$: 长度为$N$的token序列,上标$1, \dots, n$表示position index
- $\theta$: model parameters
- $\mathbf{x}^{1:n-1}$: 前n-1个tokens (left context)

核心限制:每个token只能依赖左侧context,无法利用右侧信息。

### 2.2 Discrete Diffusion前向-反向过程(公式2)

**Forward process** (加噪): 把clean数据$\mathbf{x}_0 := \mathbf{x}$逐步替换成[MASK]:

$$q(\mathbf{x}_{1:T}|\mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t|\mathbf{x}_{t-1})$$

**Backward process** (去噪):

$$p_{\theta}(\mathbf{x}) = \sum_{\mathbf{x}_{1:T} \sim q} p(\mathbf{x}_T) \prod_{t=1}^{T} \underbrace{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)}_{\text{progressive full-context prediction}}$$

变量解释:
- $\mathbf{x}_t$: timestep $t$时的noised sequence
- $t \in (0, T)$: 离散时间,但Dream采用continuous-time $t \in [0,1]$
- $\alpha_t$: noise schedule,每个token保持unmasked的概率
- 当$\alpha_t = 1-t$时,t=0为完全clean,t=1为完全[MASK]

**关键差异**: 注意公式(1)是"progressive left-context",而公式(2)是"progressive full-context" - 这就是diffusion模型bidirectional本质的来源。模型在去噪每个masked position时可以attend到全部上下文。

### 2.3 训练目标(公式3)

$$L(\theta) = -\mathbb{E}_{\mathbf{x}_0, t, \mathbf{x}_t} \left[ w(t) \sum_{n=1}^{N} \mathbf{1}_{[\mathbf{x}_t^n = \text{MASK}]} \log p_{\theta}(\mathbf{x}_0^n | \mathbf{x}_t) \right]$$

变量解释:
- $\mathbf{x}_0 \sim q(\mathbf{x})$: 从数据分布采样的clean sequence
- $t \sim \mathcal{U}(0,1)$: 均匀采样的timestep
- $\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)$: 根据t对clean sequence加噪
- $\mathbf{1}_{[\mathbf{x}_t^n = \text{MASK}]}$: indicator function,只有position $n$是[MASK]时才计算loss
- $w(t) \in (0, 1]$: time-dependent权重

**intuition**: 这个objective本质是weighted cross-entropy,只在masked positions上计算loss。当$\alpha_t = 1-t$时,$w(t) = 1/t$,意味着接近clean data ($t \to 0$)的步骤被赋予更高权重。这是ELBO的简化形式,论文引用Shi et al. (2024)和Gong et al. (2025)证明这是variational upper bound的有效reformulation。

参考: 
- Shi et al. NeurIPS 2024: https://openreview.net/forum?id=L4uaAR4ArM
- Austin et al. D3PM: https://arxiv.org/abs/2107.03006

---

## 3. 核心技术1: AR-based LLM Initialization (Shift Operation)

### 3.1 问题背景

Diffusion models从scratch训练需要海量数据。LLaDA 8B用了2.3T tokens,而AR模型如Qwen2.5-7B用了18T。能否复用AR pretraining的知识?

### 3.2 Shift Operation策略

传统diffusion: position $i$的hidden state $h_i$预测position $i$的token (因为$[MASK]$在position $i$)

Dream的shift strategy: position $i$的hidden state $h_i$预测position $i+1$的token

**架构图解析** (基于论文Figure 2):
```
Position:     1     2     3     4     5
Input:       tok1  tok2 [MASK] [MASK] tok5
              ↓     ↓     ↓     ↓     ↓
AR hidden:    h1    h2    h3    h4    h5
              ↓     ↓     ↓     ↓     ↓
AR predict:  tok2  tok3  tok4  tok5  tok6  (shift +1)

Dream Shift:
              ↓     ↓     ↓     ↓     ↓
Predict at:   pos2  pos3  pos4  pos5  pos6
              ↓     ↓     ↓     ↓     ↓
Dream target: tok2  tok3  tok4  tok5  tok6
```

**intuition**: AR模型预训练时,$h_i$就编码了"在看到$t_1, \dots, t_i$后预测$t_{i+1}$"的能力。如果Dream强行让$h_i$预测$t_i$(同位置),就破坏了这种学到的positional relationship。Shift operation保持了这种关系,让diffusion training能直接接续AR的知识。

### 3.3 实验验证 (Figure 4, 1B模型)

论文在LLaMA3.2-1B上做了ablation:
- From scratch: 训练初期loss很高,缓慢下降
- AR init: 初期因causal attention转为full attention导致高loss,但迅速下降,全程低于from scratch

**关键发现**: learning rate是critical hyperparameter:
- LR过高: 快速破坏AR的left-to-right linguistic knowledge,优势消失
- LR过低: 阻碍diffusion过程学习

参考: DifuLLaMA: https://arxiv.org/abs/2410.18514

---

## 4. 核心技术2: Context-Adaptive Token-level noise Rescheduling (CART)

### 4.1 问题:Sequence-level noise的缺陷

标准discrete diffusion为整个sequence采样一个$t$,但不同token的contextual难度差异巨大。

**例子** (Figure 3): 
```
"The [MASK] [MASK] [MASK] sky [MASK] blue"
```
- 第一个[MASK](position 2): 左边有"The",信息少,需要较高noise level建模
- 后续[MASK]: 既有左侧也有右侧clean tokens提供约束,实际noise应该更低

但标准做法给所有masked tokens同一个$t$,这导致weight $w(t) = 1/t$对所有位置一样,违背了token-level的实际信息量。

### 4.2 CART公式(公式4)

$$L(\theta) = -\mathbb{E}_{\mathbf{x}_0, t, \mathbf{x}_t} \left[ \sum_{n=1}^{N} \mathbf{1}_{[\mathbf{x}_t^n = \text{MASK}]} w(t, \mathbf{x}_t, n) \log p_{\theta}(\mathbf{x}_0^n | \mathbf{x}_t) \right]$$

对比公式(3),关键变化是$w(t) \to w(t, \mathbf{x}_t, n)$,权重现在依赖于具体的noised sequence和position。

### 4.3 CART权重设计(公式5)

$$w(t, \mathbf{x}_t, n) = \frac{1}{2} \sum_{i=1}^{N} \mathbf{1}_{[\mathbf{x}_t^i \neq \text{MASK}]} \text{Geo}(p, |n-i|-1)$$

变量解释:
- $n$: 当前masked token的position
- $i$: 遍历所有positions
- $\mathbf{1}_{[\mathbf{x}_t^i \neq \text{MASK}]}$: 只有position $i$是clean token时才贡献
- $|n-i|$: position $n$和$i$的距离
- $\text{Geo}(p, k)$: 参数为$p$的geometric distribution在$k$处的概率,$P(X=k) = p(1-p)^k$
- $p \in (0, 1]$: 控制分布尖锐度

**intuition**: 这个权重衡量"clean tokens对当前masked token的信息贡献"。
- $p$小: geometric分布平坦,所有clean tokens均匀贡献到所有masked tokens
- $p$大: geometric分布尖锐,只有附近的clean tokens贡献

具体计算: 对每个clean token $i$,它对masked token $n$的贡献随距离$|n-i|$按geometric distribution衰减。求和后乘1/2归一化。

### 4.4 CART的物理意义

**Connection to absorbing diffusion theory**: 论文引用Ou et al. (ICLR 2025)证明"absorbing discrete diffusion secretly models the conditional distributions of clean data"。CART本质是在token level显式地建模这种conditional distribution的难度。

- 一个masked token附近有很多clean tokens → 实际"effective noise"低 → weight应该更小
- 一个masked token孤立(周围都是mask) → effective noise高 → weight应该更大

这与公式(3)的$w(t) = 1/t$在sequence level的行为一致,只是CART把它精细化到token level。

参考: 
- Ou et al. ICLR 2025: https://arxiv.org/abs/2504.11581
- Reparameterized Discrete Diffusion: https://arxiv.org/abs/2405.16789

---

## 5. 训练细节

### 5.1 架构与预训练

- **Base架构**: 完全沿用Qwen2.5-7B的Transformer配置(68 layers, hidden dim 3584, attention heads, GQA等)
- **Attention**: 从causal attention改为full (bidirectional) attention
- **训练数据**: 580B tokens
  - Dolma v1.7 (https://arxiv.org/abs/2402.00124): 通用文本
  - OpenCoder (https://arxiv.org/abs/2411.04905): 代码
  - DCLM-Baseline (https://arxiv.org/abs/2406.11794): 高质量文本+数学
- **数据效率**: LLaDA用2.3T tokens,Dream只用0.6T,即1/4数据达到更好性能

### 5.2 SFT (Dream-Instruct)

$$p_{\theta}(r_0 | p_0)$$

- $p_0$: prompt (保持clean)
- $r_0$: response (加noise训练)
- 数据: 1.8M instruction-response pairs
  - Tulu 3: https://arxiv.org/abs/2411.15124
  - SmolLM 2: https://arxiv.org/abs/2502.02737
- 训练3 epochs

---

## 6. 实验数据深度分析

### 6.1 Base模型对比 (Table 1核心数据)

| Model | Type | Tokens | MMLU | ARC-C | GSM8K | HumanEval | Sudoku | Countdown | Trip |
|-------|------|--------|------|-------|-------|-----------|--------|-----------|------|
| Dream 7B | Diffusion | 0.6T | 69.5 | 59.8 | 77.2 | 57.9 | **81.0** | **16.0** | **17.8** |
| LLaDA 8B | Diffusion | 2.3T | 65.9 | 47.5 | 70.9 | 32.9 | 46.0 | 13.2 | 16.4 |
| Qwen2.5 7B | AR | 18T | 71.9 | 51.5 | 78.9 | 56.7 | 21.0 | 6.2 | 3.6 |
| LLaMA3 8B | AR | 15T | 63.5 | 53.6 | 55.3 | 35.4 | 0.0 | 3.7 | 8.7 |

**关键观察**:
1. **Dream vs Qwen2.5**: MMLU略低(69.5 vs 71.9),但ARC-C反超(59.8 vs 51.5),数学接近(77.2 vs 78.9),代码略胜(57.9 vs 56.7)
2. **Planning碾压**: Sudoku 81.0 vs 21.0(4倍提升),Countdown 16.0 vs 6.2(2.6倍),Trip 17.8 vs 3.6(5倍)
3. **Dream vs LLaDA**: 用1/4数据,所有任务全面胜出,证明AR init + CART的有效性

### 6.2 SFT后对比 (Table 2)

| Model | MMLU | GSM8K | MATH | HumanEval | IFEval |
|-------|------|-------|------|-----------|--------|
| Dream 7B | 67.0 | 81.0 | 39.2 | 55.5 | 62.5 |
| Qwen2.5 7B | 76.6 | 91.6 | 75.5 | 84.8 | 74.7 |

SFT后差距明显拉大,说明Dream的post-training recipe还很初步,论文承认这是"early exploration"。

### 6.3 Planning优势深度分析 (Figure 5)

在Countdown和Sudoku不同难度下的表现:
- **Countdown3**: Dream 7B甚至超过DeepSeek-V3-671B (0324)!
- **Sudoku**: Dream远超所有同规模AR模型

**intuition**: Planning任务本质是约束满足问题:
- Countdown: 给定数字,用+-*/组合达到目标值
- Sudoku: 行列子网格全填充约束

这类任务需要"全局视角"——先看到所有约束再决定每个位置。AR模型从左到右贪心生成,一旦早期决策错误就无法回溯。Diffusion模型迭代refine所有位置,可以基于全局约束修正局部决策。

### 6.4 Quality-Speed Tradeoff (Figure 7)

调整diffusion timesteps:
- Timesteps 5-20: 在Countdown上同时优于Qwen2.5的speed和quality
- 这是test-time scaling的新维度,与CoT (https://arxiv.org/abs/2412.16720)等正交

---

## 7. Inference Flexibility:任意顺序生成

### 7.1 三种灵活能力

1. **Completion**: 给定前缀,续写剩余
2. **Infilling**: 给定两端,填充中间。可约束结尾匹配特定句子
3. **Configurable decoding order**: 
   - 强制left-to-right (模拟AR)
   - 完全随机order
   - 部分随机

### 7.2 技术原理

Diffusion inference过程:
1. 从全[MASK]开始
2. 每步:模型预测所有[MASK]位置的token分布
3. 选择部分位置unmask (基于confidence或用户指定order)
4. 重复直到全部unmask

这种机制天然支持任意顺序——用户可以指定"先unmask position 5,再position 2",模型都能处理,因为它训练时见过各种noise pattern。

---

## 8. 关键Limitations与未来方向

1. **Post-training不足**: SFT后性能落后Qwen2.5明显,缺乏RLHF/DPO等
2. **通用任务仍有gap**: MMLU等略低于Qwen2.5
3. **Long context**: 论文未探索,但diffusion的parallel特性理论上更适合
4. **推理效率**: 虽然可调timesteps,但每步需要full sequence forward

---

## 9. 相关工作脉络

1. **D3PM** (Austin et al. 2021): discrete diffusion基础
2. **Diffusion-LM** (Li et al. 2022): continuous embedding space
3. **DiffuBERT** (He et al. 2023): MLM初始化
4. **MDM scaling** (Lou et al. 2024): ratio estimation
5. **LLaDA** (Nie et al. 2025): 8B from scratch
6. **DifuLLaMA** (Gong et al. 2025): AR模型adaptation
7. **Block Diffusion** (Arriola et al. 2025): AR+diffusion混合
8. **Mercury Coder** (Inception Labs): 商业实现
9. **Dream 7B** (本文): AR init + CART + 7B scale

---

## 10. Intuition总结

Dream 7B的核心贡献:

1. **AR init的shift operation**: 最大化复用AR pretraining的"left-to-right prediction"知识,只需将causal attention改为full attention,大幅降低训练成本

2. **CART**: 把sequence-level的noise weight $w(t)=1/t$精细化到token level,基于geometric distribution衡量clean tokens对masked tokens的距离加权信息贡献,更准确反映每个token的实际"denoising difficulty"

3. **Planning优势的来源**: Bidirectional attention让模型在去噪每个位置时能看到全部约束,iterative refinement允许修正早期错误决策,这恰好匹配planning任务的约束满足本质

4. **灵活推理**: Arbitrary order generation, infilling, tunable timesteps - 这些是AR模型无法实现的inherent property

**未来值得探索的方向**: RL post-training, long context diffusion, 更复杂的test-time scaling策略(如结合diffusion steps + chain-of-thought), 以及diffusion模型在embodied AI等需要全局规划场景的应用。

论文的实验数据清晰地表明,diffusion LLM在"需要全局约束的任务"上具有AR模型无法匹敌的结构性优势,这为下一代LLM架构提供了新的可能性。
