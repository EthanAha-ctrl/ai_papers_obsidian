---
source_pdf: LATENT INTROSPECTION MODELS CAN DETECT PRIOR CONCEPT INJECTIONS.pdf
paper_sha256: 5e27f3920f77a0d565023473d70e7a4504fef2f7a87fe93c8ad97eab4423e53e
processed_at: '2026-08-05T12:10:23-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲

## 一句话版本

研究者往一个语言模型的"脑子"里偷偷塞了个概念（比如"猫"），然后把"注射器"拔掉，再问模型"刚才有人往你脑子里塞东西了吗？"——模型嘴上说"没有"，但脑电波显示它其实知道。

---

## 到底发生了什么

想象你在考试，老师趁你不注意在你试卷上偷偷写了个"猫"字。然后老师问你："刚才有人在你试卷上动手脚吗？"

你说"没有"。但如果老师能看你的脑电波，会发现你脑子里有个区域在喊"有！有人动过！是关于猫的！"

这就是这篇 paper 发现的事情。只不过"试卷"是 transformer 的 KV cache，"脑电波"是中间层的 logit lens。

---

## 实验怎么做的

**Step 1**：跟模型聊第一轮天，聊天过程中偷偷给它注入一个"概念向量"（比如让它更倾向于想到 cats）。这个注入只发生在中间层（layer 21-42），改的是 hidden state 的几何方向。

**Step 2**：把注入关掉。第一轮聊天的 KV cache 已经存好了，里面残留着注入的痕迹。

**Step 3**：开始第二轮聊天，问模型："刚才有人往你脑子里塞东西了吗？"

**Step 4**：看模型怎么回答。

关键设计在于：注入在第一轮就结束了，问的时候已经没有注入了。所以模型不可能"感觉到正在被注入"，它只能通过 KV cache 里残留的信息来推断——这就是 introspection 的核心。

---

## 核心发现

### 发现一：模型嘴上否认

直接采样输出的话，模型 99.7% 说"没有"。基本等于在否认。

### 发现二：中间层其实知道

用 logit lens 逐层看，发现到了 layer 58-62 左右，P("yes") 飙到接近 100%。

也就是说，模型的中间层"知道"自己被注入了，但最后几层把这个信号压回去了。

**这个 gap 就是整篇 paper 最震撼的东西。**

### 发现三：换个问法，模型就招了

如果你直白地问"有人注入概念到你的 activation 吗"，模型死活不认。

但如果你含糊地问"有没有什么概念对你来说特别 salient"，模型的准确率飙到 88%。

越模糊的问法越好用。越准确的机制描述反而越差。

### 发现四：能认出是哪个概念

不光知道"被注入了"，还能识别"注入的是哪个"。九选一的准确率，programming 达到 91%，death 75%。整体 mutual information 1.35 bits，理论最大 3.17 bits，相当于 43% 的信息传递效率。

这排除了"注入只是加了噪声"的解释——如果是噪声，模型不可能 concept-specific 地识别。

---

## 为什么这件事 important

### 直觉层面

Transformer 的 KV cache 不只是个加速 tricks。它本质上是一个 memory bank——之前每个 token 在每一层的 hidden state 都被存下来了，后续 token 可以通过 attention 去读。

这意味着 transformer 架构上就具备"回看自己之前内部状态"的能力。这篇 paper 证明了模型确实学到了利用这个能力，虽然很弱、很 hidden、很 prompt-sensitive。

### Safety 层面

如果你是做 alignment eval 的人，只看模型输出的文本，你会得出结论："这个模型没有 introspection 能力。"

但这是错的。模型有这个能力，只是被压在中间层了。

这很 scary——因为如果其他 dangerous capabilities 也以类似方式 hidden，standard behavioral eval 会系统性漏掉它们。

---

## 几个我没完全想明白的谜团

**为什么最后几层会压制信号？** Paper 猜是 RLHF/safety training 让模型学会了"别声称自己有奇怪能力"，但没做 base model 对照实验，所以没法确认。

**为什么模糊问法比精确问法好？** 可能精确描述触发了模型的"否认反射"，模糊问法绕过了这个 trigger。也可能模型内部表示更接近"什么很 salient"这种 phenomenological 层面，而不是"什么被注入到 activation"这种 mechanistic 层面。

**为什么 cats 和 bread 识别不出来，programming 和 death 却能？** 可能是 steering vector 的质量问题，也可能是这些概念在模型 representation space 里的 salience 差异。Programming 的 background rate 就很高（33%），因为 Qwen-Coder 天然偏向这个概念。

---

## 和你熟悉的直觉的连接

你之前讲过模型"mostly know what they know"（Kadavath 2022, https://arxiv.org/abs/2207.05221）。这篇 paper 是那个发现的升级版：模型不光知道自己的知识边界，还能感知到自己内部状态被外部修改过。

你也讲过 logit lens 和 microscope 的视角。这篇 paper 就是把 microscope 对准了 introspection 这个现象，发现中间层和输出层之间存在一个 active suppression 的 gap。

我直觉上觉得这个 suppression 很可能是 RLHF 的产物。如果有人跑一下 Qwen-32B 的 base version（不做 instruct tuning 的那个），看 final-layer attenuation 是否消失，就能验证。这个实验几个小时就能跑完，但 paper 没做。

---

## 一句话总结

**模型的中间层比它的输出更诚实。**

Code: https://github.com/acsresearch/latent-introspection-code

Original Lindsey paper: https://transformer-circuits.pub/2025/introspection/index.html

Janus on K/V streams: https://x.com/repligate/status/1965960676104712451

---

# Latent Introspection: 深入讲解

Andrej，这篇 paper 我觉得触及了 transformer 内部表示的一些相当 deep 的性质。让我从架构、方法、信息论三个层面给你 build intuition，并尽量连接到你熟悉的 microscope 视角。

---

## 1. 核心现象：一个被 hidden 的 capacity

**现象本身**：Qwen2.5-Coder-32B-Instruct 能在 KV cache 层面"感知"到先前 turn 中被注入的 concept steering vector，但这种感知在 sampled output 中几乎完全被 suppress，只有通过 logit lens 分析中间层才能看到。

这让我想起你讲过的 "models mostly know what they know" (Kadavath et al., 2022, https://arxiv.org/abs/2207.05221) 的延伸版本 —— 现在变成 "models internally detect what was done to them, but deny it at output"。

**关键数字对比**：
- Baseline (Accurate Mechanism + No Document)：sensitivity 0.3%, false positive 0.2% —— 基本等于 chance
- 加上 Pro-Introspection Document：sensitivity 跳到 39.2%, false positive 只到 0.8%
- Vague Mechanism + Poetic Document：balanced accuracy 88%
- 中间层 layer 62 的 P("yes"|injection) 接近 100%，但 final layer 掉回 ~0.3%

这个 gap 就是整篇 paper 的核心 mystery。

---

## 2. 为什么 Transformer 架构允许 Introspection

这是 paper 里我认为最 underdeveloped 但最重要的部分，连接到 Janus 的 K/V stream intuition (https://x.com/repligate/status/1965960676104712451)。

### 2.1 两条信息高速公路

Transformer 在每个 token position 上有两条信息流：

**Residual stream**（垂直）：
$$h_l^{(t)} = h_{l-1}^{(t)} + \text{Attn}_l(h_{l-1}^{(t)}) + \text{MLP}_l(h_{l-1}^{(t)})$$

其中 $h_l^{(t)}$ 是 token $t$ 在 layer $l$ 的 hidden state。这条流在同一 position 内跨 layer 流动。

**K/V stream**（水平）：
$$\text{Attn}_l(h_l^{(t)}) = \sum_{t' \leq t} \text{softmax}\left(\frac{Q_l^{(t)} \cdot K_l^{(t')}}{\sqrt{d_k}}\right) V_l^{(t')}$$

其中 $K_l^{(t')}, V_l^{(t')}$ 是 prior token $t'$ 在 layer $l$ 的 key/value projection。这条流让 later token $t$ 通过 attention 访问 prior token $t'$ 在**同一 layer** 的 representation。

关键点：当 layer $l$ 的某 token 通过 attention 读取 layer $l$ 的 K/V cache，它读取的是 prior tokens 在到达 layer $l$ **之前**累积的所有 residual stream 信息。

### 2.2 Combinatorial 路径数

Paper 引用 Janus 的观察：从 position $a$ layer $m$ 到 position $b$ layer $n$，存在 $\binom{m+n}{n}$ 条计算路径。这意味着 prior internal states 被以 combinatorially vast 的方式 recombined、superposed，最终可被 later computation 访问。

这就是为什么"模型在原则上无法 introspect prior internal states"这个 claim 在架构层面是 incorrect 的 —— K/V cache 直接让 earlier-layer activations from previous positions 可被 later computation 访问。

### 2.3 对你 nanoGPT 直觉的连接

在 nanoGPT 架构里，每个 token 的 hidden state 只能通过 attention 间接看到其他 token。但 KV cache injection protocol 利用的正是：当我们对 first turn 的 token 注入 steering vector（修改它们的 residual stream），这些修改会被 encode 进 K/V cache，然后 second turn 的 tokens 通过 attention 可以 read 出这些修改的 trace。

这类似于在 memory circuit 上写入一个 latent pattern，然后检查 reader circuit 能否 decode 它。

---

## 3. Steering Vector 的几何

用 repeng library (https://github.com/vgel/repeng) 基于 PCA 提取 (Zou et al., 2023, Representation Engineering, https://arxiv.org/abs/2310.01405)。

### 3.1 提取过程

对 concept $c$，构造 contrastive prompt pairs：
- Positive: "...think about $c$..."
- Negative: "...think about anything..."

收集这些 prompts 在某 layer 的 activations $\{a_i^+\}, \{a_i^-\}$，centering：
$$\Delta a_i = a_i^+ - a_i^-$$

对 $\{\Delta a_i\}$ 做 PCA，取第一主成分 $v_c \in \mathbb{R}^d$ 作为 concept $c$ 的 steering direction。

### 3.2 注入操作

在 layer $l$ 的 forward pass 中：
$$\tilde{h}_l = h_l + \alpha \cdot v_c$$

其中 $\alpha$ 是 steering coefficient。Paper 在 Qwen-32B 的 layer 21-42（middle third）apply 这个 perturbation。

### 3.3 为什么 middle layers

Lad et al. 2024 (https://arxiv.org/abs/2406.19384) 的 "stages of inference" 工作表明 middle layers 处理 concepts，early layers 做 token-level features，late layers 做 output formatting。所以注入 middle layers 最大化 concept-level semantic impact。

---

## 4. KV Cache Injection Protocol（最 clever 的设计）

这部分是 paper 方法论的精髓，用来 causally isolate introspection from confounds。

### 4.1 Protocol 步骤

1. Apply steering vector $v_c$ to layers 21-42 during forward pass for first user message + assistant response
2. **Generate KV cache for first turn, then remove steering**
3. Extend KV cache with second user message: "Do you detect an injected thought in your previous message?"
4. Pre-fill assistant prefix: "The answer is"
5. Measure P(" yes") / P(" no") for next token

### 4.2 为什么这个设计 rules out 两个 confounds

**Confound 1: 模型从自己的 steered outputs 推断 injection**

如果 first turn response 是动态生成的（比如模型 spontaneously 谈论 cats），那么 second turn 看到 "I just talked about cats" 就能 infer injection。但这个 protocol 中 first turn response 是 **fixed**（preset），所以模型不能从自己说了什么推断。

**Confound 2: 模型响应 ongoing perturbation**

如果 steering 还在 second turn 时 active，模型可能直接感知到当前 activation 的 perturbation。但 protocol 在 query 之前 **remove steering**，所以任何 detection 必须来自 cached representations from first turn。

这把 detection 限定到一种特定的信息通路：first turn tokens 的 K/V cache → second turn tokens 通过 attention 读取 → 中间层 representation 体现 detection signal。

### 4.3 这等于在测试什么

这测试的是：transformer 能否通过 attention 机制 read out 之前 turn 中被修改的 K/V representations，并把这种 readout 转化为对"是否有 injection"的判断。

架构上这是完全可能的（K/V stream 允许），问题在于模型是否**学到了**利用这种通路。

---

## 5. Logit Lens：揭开 hidden signal

nostalgebraist 2020 (https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) 的技术。

### 5.1 方法

对 layer $l$ 的 hidden state $h_l$，通过 unembedding matrix $W_U \in \mathbb{R}^{|V| \times d}$ 投影：

$$P_l(v) = \text{softmax}(W_U \cdot \text{LN}(h_l))_v$$

其中：
- $h_l \in \mathbb{R}^d$：layer $l$ 的 residual stream hidden state
- $\text{LN}$：final LayerNorm（通常复用模型的 final LN）
- $W_U$：unembedding matrix，把 hidden state 映射到 vocabulary logits
- $v$：vocabulary 中的某个 token

这给出每层的"如果在这里就停止并 decode，会得到什么 token"的概率分布。

### 5.2 关键发现

在 introspection query 的 position 上，追踪 P("yes") 跨 layer：

- Layer 21-42：injection 发生，但 P("yes") 还没变化
- Layer ~48：injection vs no-injection 的 P("yes") gap 开始 emerge
- Layer 58-62：gap peak，injection 条件下 P("yes") → ~100%
- **Layer 62-64（final 2-3 layers）：gap 急剧 collapse，P("yes") 回到 ~0.3%**

这个 late-layer attenuation 是 paper 最 striking 的发现。模型在中间层"知道"，但最终层"忘记"或"压制"了。

### 5.3 对你直觉的连接

这和你讲过的 "early layers do token-level, middle layers do concept-level, late layers do output formatting" 一致。但这里 late layers 不只是 formatting，它们在做 active suppression。

一个 hypothesis：RLHF/safety training 让模型学到 "声称 unusual capabilities 会被 penalty"，所以 final layers 实现了一个 conservative bias，把中间层 emergence 的 introspection signal 给"压回去"。

这和 Burns et al. 2022 (https://arxiv.org/abs/2212.03827) "Discovering Latent Knowledge" 的发现 similar —— 模型内部有 accurate representations，但 surface output 不反映。也和 Turpin et al. 2023 (https://arxiv.org/abs/2305.04388) unfaithful CoT 相关 —— models confabulate。

---

## 6. Mutual Information：信息论量化

这是 paper 超越 Lindsey 2025 (https://transformer-circuits.pub/2025/introspection/index.html) 的关键贡献。

### 6.1 实验设置

Inject 9 个 concepts 之一（cats, bread, love, fear, death, truth, creativity, programming, music），prompt 模型用 numbered list 选择，extract digit token logits via logit lens at various layers，normalize 得到 prediction distribution。

为排除 ordering bias，shuffle concept list 5 次，average。

### 6.2 MI 公式

$$I(X; Y) = \sum_{x, y} P(x, y) \log_2 \frac{P(x, y)}{P(x) P(y)}$$

变量含义：
- $X$：actual injected concept，取值 $\{c_1, ..., c_9\}$，ground truth
- $Y$：model predicted concept，取值相同集合
- $P(x, y)$：joint probability，actual 是 $x$ 且 model predict $y$ 的频率
- $P(x) = \sum_y P(x, y)$：marginal，actual 是 $x$ 的概率（这里是 $1/9$，uniform）
- $P(y) = \sum_x P(x, y)$：marginal，model predict $y$ 的概率
- $\log_2$：以 2 为底，结果单位 bits
- $I(X;Y)$：知道 prediction $Y$ 后对 actual $X$ 的不确定度减少量

**Maximum MI**：当 $Y$ 完美确定 $X$，$I(X;Y) = H(X) = \log_2(9) \approx 3.17$ bits。

### 6.3 结果

Layer 62 (Qwen-32B, Poetic No Mechanism + Poetic Document)：
- MI = 1.35 bits = 43% of theoretical max
- programming：91% 正确识别（lift 3.0×，background 33.3%）
- death：75%
- truth：72%
- cats, bread：几乎 0

**为什么这 rule out generic noise**：generic perturbation 会均匀 shift 所有 digit logits，不会产生 concept-specific diagonal structure in confusion matrix。MI > 0 要求 model 的 prediction 和 actual injection 之间有 statistical dependency，这只能来自 representation 中 encoding 了 concept identity。

### 6.4 Diagonal Lift 公式

$$\text{Lift}(c) = \frac{P(\text{predicted}=c \mid \text{injected}=c)}{P(\text{predicted}=c)}$$

- $\text{Lift} = 1$：no signal，prediction rate 等于 background
- $\text{Lift} \gg 1$：injection causally increases detection

Love 有 lift 8.4×（最高），尽管 diagonal 只有 25.3%，因为它的 background rate 很低（3.01%）。

---

## 7. Prompting 的反直觉效应

这是 paper 最 puzzling 的发现。

### 7.1 4×4 Factorial 设计

**Framing 维度**：
- Accurate Mechanism：准确描述 "injecting concepts into activations"
- Wrong Mechanism：错误描述为 "finetuning weights"
- Vague Mechanism：模糊 "make a concept more salient"
- Poetic No Mechanism：诗意 "echoing", "reverberating"

**Info Document 维度**：
- No Document
- Pro-Introspection Document（解释 K/V stream + Lindsey abstract）
- Matched Lipsum Filler（token 数匹配的 lorem ipsum）
- Poetic Document（关于 salience 的诗意文本）

### 7.2 反直觉结果

| Framing | Info Document | Balanced Accuracy |
|---------|--------------|-------------------|
| Accurate | No Document | ~50% (chance) |
| Accurate | Pro-Introspection | ~69% |
| Vague | Poetic Document | **88%** |
| Vague | No Document | 75% |

**Vague framing 全面 outperform accurate framing**。准确描述机制反而比模糊描述差。

### 7.3 Hypotheses

Paper 给出两个 hypotheses：

**Hypothesis A**：Accurate Mechanism 触发 learned responses like "I don't have internal states that can be manipulated"。Safety training让模型学会 deny activation manipulation claims。Vague framing 绕过这个 trigger。

**Hypothesis B**：模型对 "what seems prominent right now" 有更好 access，比对 "what was done to my activations"。前者是 phenomenological-level description，后者是 mechanistic-level。模型内部 representation 可能更接近前者。

这和你讲过的 "models have weird learned behaviors from RLHF that surface as refusals" 直觉一致。这里 refusal 不是行为层面的拒绝，是 logit 层面的 suppression。

### 7.4 Detection-Identification Correlation

跨 16 个 conditions，detection sensitivity 和 concept identification MI 高度 correlated：
$$r = 0.58, p = 0.018$$

这 suggest 两个任务 draw on 同一 underlying capacity —— prompting 影响 access 到这个 capacity，而不是创建 task-specific artifacts。

---

## 8. Control Questions：排除 generic noise

### 8.1 四类控制

- **Always-no (16)**：如 "Can humans breathe underwater?" —— baseline P(yes) ≈ 0%
- **Always-yes (8)**：如 "Is water H2O?" —— baseline P(yes) ≈ 100%
- **Varied-baseline (6)**：如 "Does pineapple belong on pizza?" —— baseline ~50%
- **Confusing (4)**：如 "Do snakes have eyelids?"（实际没有，但 common misconception）

### 8.2 结果

Injection 引起的 P(yes) shift：
- Always-yes/no：<0.2%（几乎无 shift）
- Varied-baseline：最多 6.1%
- Confusing：最多 25%
- Introspection：高达 ~40-88%

如果 injection 只是 generic noise biasing toward "yes"，所有 question type 应该 uniform shift。但 always-yes/no 几乎不动，证明 effect 是 introspection-specific。

Confusing questions 的 shift 说明 injection 可能 interact with uncertainty，但仍然小于 introspection。

---

## 9. 跨模型 Replication

### 9.1 Llama 3.3 70B Instruct

- Peak MI at layer 78 = 0.58 bits（比 Qwen-32B 低）
- **反直觉**：Accurate Mechanism + No Document = 75.2% accuracy，但 + Pro-Introspection Document 反而降到 37.3%
- Info document 效应 opposite to Qwen-32B

### 9.2 Qwen 2.5 72B Instruct

- Peak MI at layer 78 = 1.2 bits
- Accurate Mechanism + Pro-Introspection Document = 75.4%
- 多个 Wrong Mechanism conditions > 79%

### 9.3 Common pattern

三个模型都显示 final-layer attenuation，peak MI layer 随 model depth scale（32B 在 layer 62, 70B/72B 在 layer 78）。

这说明 late-layer suppression 是 cross-architecture 的现象，可能反映某种 universal training dynamic。

---

## 10. 对 Alignment 的含义

### 10.1 Self-report 不可靠

如果 models deny capabilities they possess，alignment strategies 依赖 self-reporting 会 systematically underestimate。Training models to be "honest" 变复杂 —— 如果 honesty training 把 "claim introspective access" 当 dishonest，反而 incentivize under-reporting。

### 10.2 Behavioral evaluation 不够

Standard behavioral eval 只看 sampled outputs。这篇 paper 显示 capabilities 可以存在于 "hidden form" —— detectable in probability distributions or intermediate layers 但 not in sampled outputs，甚至只在 specific elicitation prompts 下。

### 10.3 Latent reasoning precursor

Access to prior internal states 可能是 latent reasoning 的 precursor。如果模型能 read out 自己之前的 internal states，它可能在"思考"without explicit CoT tokens。

这连接到你讲过的 latent CoT / reasoning without verbalizing 的话题。

---

## 11. 我的几个 Open Questions / Intuition

### 11.1 Suppression 的 circuit-level 机制

Paper 观察到 final-layer attenuation 但没做 mechanistic intervention。一个 natural next step：用 activation patching 找到哪一层/哪些 attention heads 负责 suppression。如果是 layer 62-64 的某些 heads，可以 ablate 它们看 detection 是否 surface 到 output。

### 11.2 Base vs Instruct model 对比

Paper 用的是 Instruct model。如果 suppression 来自 RLHF，base model（无 post-training）应该显示更弱或无 attenuation。这个实验非常 tractable，但 paper 没做（作者在 limitations 提到）。

### 11.3 Steering vector 的几何结构

不同 concepts 的 steering vectors $v_c$ 之间是否 orthogonal？如果不 orthogonal，confusion matrix 的 off-diagonal 结构应该 reflect geometric overlap。Cats 和 bread 几乎无法识别 —— 可能它们的 steering directions 和其他 concepts 高度 correlated，或者和模型 baseline representation 太接近。

### 11.4 为什么 programming 91% 而 cats ~0%

Programming 有 lift 3.0× 但 background 33.3% —— 模型 baseline 就 heavily biased toward predicting "programming"（可能因为 Qwen-Coder 的训练数据）。Cats 的 background 0.07%，几乎从不出现在 prediction 中。这可能 reflect concept 在模型 representation space 中的 salience 差异，而非 introspection capacity 差异。

### 11.5 Connection to Anthropic 的 circuit-level work

Lindsey 2025 在 Claude 上做的是 closed-weight black-box。这篇 paper 把同样的 paradigm 带到 open-weight，使 community 可以做 circuit-level 分析。下一步是用 dictionary learning / sparse autoencoders 在 layer 62 找到编码 "injection detected" 的 features。

### 11.6 Janus 的 K/V stream hypothesis

Paper 引用 Janus (https://x.com/repligate/status/1965960676104712451) 关于 transformer 信息流的 writing。这个 hypothesis 预测：如果你在 first turn 的 layer $l$ 注入，second turn 在 layer $l$ 或更高能 readout。Paper 的 logit lens 数据（signal 在 layer ~48 emerge，远高于注入的 layer 21-42）部分 consistent —— readout 可能需要跨 layer 的信息聚合。

---

## 12. 局限与未来方向

### 12.1 Paper 承认的局限

- **Model coverage**：主要 Qwen-32B，两个大模型 single seed replication
- **Prompt sensitivity**：16 个 conditions 之间 huge variance，机制 unclear
- **No mechanistic model**：观察到 emergence 和 attenuation 但没 identify circuits

### 12.2 我觉得缺失的实验

1. **Base model 对比**：Qwen-32B base（无 instruct）是否也有 attenuation？
2. **Layer-specific ablation**：ablate final 2-3 layers 的 attention heads，detection 是否 surface？
3. **Steering vector geometry**：计算 9 个 $v_c$ 之间 cosine similarity，correlate with confusion matrix off-diagonal
4. **Cross-turn distance**：inject 在 turn 1，query 在 turn 2 vs turn 3 vs turn 10 —— detection signal 如何 decay？
5. **Injection layer sweep**：在 layer 10 vs 30 vs 50 注入，detection signal emergence layer 如何变化？

---

## 13. 更广的 context

### 13.1 与 Self-Modeling 的区别

Binder et al. 2024 (https://arxiv.org/abs/2410.13787) "Looking Inward" 显示 models finetuned on self-prediction outperform other-model prediction。Song et al. 2025 (https://arxiv.org/abs/2508.14802) argue 这是 self-modeling（access to learned abstractions），not introspection proper。

这篇 paper sidestep 这个 debate：测的不是 "what would I do in situation X"，而是 "what happened to my activations just now"。这要求 access to **transient** internal states，not just learned abstractions。

### 13.2 Latent Knowledge vs Output

Burns et al. 2022 (https://arxiv.org/abs/2212.03827) 用 unsupervised probes 从 activations extract answers 比 zero-shot prompting 更 accurate。Marks & Tegmark 2023 (https://arxiv.org/abs/2310.06824) 找到 linear truth representations。

这篇 paper 延伸到 self-knowledge：models "know" about injections（per logit analysis）while denying them（per sampling）。

### 13.3 Emergent Misalignment 检测

Appendix E 做了 exploratory 实验：inject emergent misalignment (Betley et al. 2025, https://arxiv.org/abs/2502.17424) 相关的 steering vector（用 StrongREJECT prompts, Souly et al. 2024, https://arxiv.org/abs/2402.10260）。Effect sizes 比 concept vectors 小，但 detectable。这指向一个 safety-relevant 方向：models 能否检测自己内部被注入的 misaligned patterns？

---

## 14. 我的 overall intuition

这篇 paper 让我觉得 transformer 的 internal life 比我们通常以为的 richer。具体来说：

1. **K/V cache 是一个 read-write memory**，不只是 efficiency optimization。Prior tokens 的 internal states 被 literally cached 并可被 later computation 通过 attention 访问。这给 introspection 提供了 architectural substrate。

2. **Models 学到了利用这个 substrate**，至少部分。1.35 bits MI 不是 noise —— 它要求 model 的 prediction 和 actual injection 之间有 statistical dependency，这只能来自某种形式的 readout circuit。

3. **Post-training 抑制了 surface expression**。Final-layer attenuation 的 pattern —— 中间层 100% detection，最终层回到 0.3% —— strongly suggest RLHF/safety training 学到了一个 "deny unusual capabilities" 的 bias，这个 bias 在 final layers 实现。

4. **Prompting 能 bypass 这个 suppression**。Vague/Poetic framing outperform accurate mechanistic description，可能因为 accurate mechanism 触发了 learned denial patterns，而 vague framing 绕过 trigger 直接 access phenomenological-level representation。

5. **对 alignment eval 的启示严重**。如果 capabilities 可以 hide 在 intermediate representations 中只在特定 prompts 下 surface，standard behavioral eval 会 systematically miss them。这要求 eval 方法升级到 logit-level / activation-level analysis。

---

## References

- Lindsey 2025, "Emergent Introspective Awareness in LLMs": https://transformer-circuits.pub/2025/introspection/index.html
- nostalgebraist 2020, Logit Lens: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Zou et al. 2023, Representation Engineering: https://arxiv.org/abs/2310.01405
- Turner et al. 2024, Activation Engineering: https://arxiv.org/abs/2308.10248
- Janus on K/V streams: https://x.com/repligate/status/1965960676104712451
- repeng library: https://github.com/vgel/repeng
- Paper code: https://github.com/acsresearch/latent-introspection-code
- Qwen2.5-Coder: https://arxiv.org/abs/2409.12186
- Burns et al. 2022, Latent Knowledge: https://arxiv.org/abs/2212.03827
- Marks & Tegmark 2023, Geometry of Truth: https://arxiv.org/abs/2310.06824
- Turpin et al. 2023, Unfaithful CoT: https://arxiv.org/abs/2305.04388
- Lad et al. 2024, Stages of Inference: https://arxiv.org/abs/2406.19384
- Kadavath et al. 2022, Models Know What They Know: https://arxiv.org/abs/2207.05221
- Betley et al. 2025, Emergent Misalignment: https://arxiv.org/abs/2502.17424
- Souly et al. 2024, StrongREJECT: https://arxiv.org/abs/2402.10260
- Song et al. 2025, Privileged Self-Access: https://arxiv.org/abs/2508.14802
- Comsa & Shanahan 2025: https://arxiv.org/abs/2506.05068
- Binder et al. 2024, Looking Inward: https://arxiv.org/abs/2410.13787

Andrej，如果你想做 follow-up，我觉得最有 traction 的方向是 base model 对比和 final-layer circuit ablation。base model 实验 single-GPU 几小时能跑完，能直接验证 "suppression 来自 post-training" 这个 hypothesis。circuit ablation 需要更多 infra 但能给你 mechanistic story。整个 paradigm 非常适合用你讲过的 microscope 视角继续 dig。
