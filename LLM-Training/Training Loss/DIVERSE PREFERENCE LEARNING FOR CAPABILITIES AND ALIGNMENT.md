---
source_pdf: DIVERSE PREFERENCE LEARNING FOR CAPABILITIES AND ALIGNMENT.pdf
paper_sha256: 290daaa9afd6aac30b6cab3b24db01ec29279b4bbafc4ad420429dcec867a8d6
processed_at: '2026-08-03T22:46:09-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇paper

## 一句话说清楚

**RLHF/DPO把人类偏好的分布"指数级放大"了，导致模型变成复读机；这篇paper的fix就是加一个旋钮把放大倍数调下来。**

---

## 问题长啥样

你train完一个RLHF'd model，问它100次同一个开放性问题，它给的答案就像同一个模子刻出来的。同一个医生名字、同一个故事结构、同一个hedging phrase "It's important to note that..."。这就是业界说的"alignment tax"——模型变helpful和harmless的同时，personality被压扁了。

具体怎么扁的？paper给了个数学账：

假设80%的人prefer答案A，20%的人prefer答案B。你用标准DPO训出来，模型生成A的概率不是80%，是 **99.9999%**。

为什么？因为DPO的objective里有个KL penalty，penalty strength $\beta$ 通常取0.1。这个0.1跑到公式里变成exponent $1/\beta = 10$，于是 $p=0.8$ 被raise到10次方：

$$0.8^{10} \approx 0.107, \quad 0.2^{10} \approx 2\times 10^{-7}$$

归一化之后，B基本消失了。这就是**exponential amplification**。

---

## 为啥这是个问题

### 对alignment
80% vs 20%的population偏好，模型represent成99.9999% vs 0.0001%，这根本不是"reflecting diverse perspectives"，这是把minority views擦掉。LLM越来越成为人们的信息源，这就会reinforce dominant narratives，把minority opinions挤出去。

### 对capabilities
想象一个geometry题，60%的preferred solution用synthetic geometry，40%用coordinate geometry。但正确解法恰好是coordinate geometry。DPO model会99.99%概率去尝试synthetic geometry，100次采样全失败。一个更diverse的model会偶尔try coordinate approach，某次就成功了。

这跟inference-time scaling特别相关——OpenAI o1、DeepMind的IMO silver medal、Tree of Thoughts这些方法都靠**反复采样不同strategy**，但你的model如果diversity被压扁了，sample再多也是同一个错思路。

### 对calibration
模型对所有答案都overconfident。问它"法国首都是哪"，它99.9999%说Paris，对的；问它"1923年德国通胀峰值是哪天"，它也99.9999%说某个specific日期，但这次是瞎编的。Confidence跟accuracy脱节，calibration崩了。

---

## 病根在哪

KL divergence的定义是:
$$D_{KL}(\pi \| \pi_{ref}) = -H(\pi) + H(\pi, \pi_{ref})$$

变量意思：
- $\pi$ 是learned policy（你的model）
- $\pi_{ref}$ 是reference policy（通常是SFT base model）
- $H(\pi)$ 是 $\pi$ 的Shannon entropy，衡量distribution有多flat
- $H(\pi, \pi_{ref})$ 是cross-entropy，衡量 $\pi$ 跟 $\pi_{ref}$ 有多像

注意**这两项在干完全不同的事**：
- $-H(\pi)$ 想让distribution尖锐（low entropy = mode-seeking）
- $H(\pi, \pi_{ref})$ 想让 $\pi$ 跟 $\pi_{ref}$ 分布相似

但RLHF用一个 $\beta$ 同时scale这两项。这就是病根——你想控制"对reference的bias"和"自己的diversity"，但只有一个knob，调一个另一个跟着动。

practice里 $\beta = 0.1$，意味着两个term都被压小，entropy bonus弱到几乎没效果，于是distribution collapse到majority mode。

---

## SPL的fix

**把一个旋钮拆成两个**。

SPL objective:
$$\max_\pi \mathbb{E}_\pi[r] + \alpha H(\pi) - \beta H(\pi, \pi_{ref})$$

- $\alpha$: entropy bonus，独立控制diversity
- $\beta$: cross-entropy penalty，控制对reference的closeness

$\alpha = \beta$ 时退化成标准DPO。$\alpha > \beta$ 时增加diversity。

### 闭式解

经过Bradley-Terry reward modeling + KL regularized RL这套推导（paper Proposition 3.2），SPL的optimal policy是:

$$\pi(y) \propto \pi_{ref}(y)^{\beta/\alpha} \cdot p^{1/\alpha}$$

对比DPO的 $\pi(y) \propto \pi_{ref}(y) \cdot p^{1/\beta}$：

- DPO: $p$ 被raise到 $1/\beta$（典型10-100）
- SPL: $p$ 被raise到 $1/\alpha$，你可以自己选

**关键insight**: 当 $\alpha = 1$，$p^{1/\alpha} = p$，模型正好按population比例represent偏好。80%的人喜欢A，模型就以80%概率生成A。这叫**proportional representation**，从social choice theory借的概念。

### DPO-style loss

标准DPO loss:
$$\log \sigma\left(\beta \log \frac{\pi(y)}{\pi(y')} - \beta \log \frac{\pi_{ref}(y)}{\pi_{ref}(y')}\right)$$

SPL loss:
$$\log \sigma\left(\alpha \log \frac{\pi(y)}{\pi(y')} - \beta \log \frac{\pi_{ref}(y)}{\pi_{ref}(y')}\right)$$

**就改一个希腊字母**。Code里就是把两个 $\beta$ 拆成 $\alpha$ 和 $\beta$，加一个hyperparameter。

---

## SPL跟temperature scaling啥关系

作者推导出一个很漂亮的事：SPL的optimal policy可以rewrite成:

$$\pi_{SPL}(y|x) = \pi_{DPO}(y|x)^{\beta/\alpha} / Z = \pi_{DPO}(y|x)^{1/(\alpha/\beta)} / Z$$

所以 $\alpha/\beta$ 实际上是个**sequence-level temperature**作用在整个DPO policy上。

### 跟标准token-level temperature的本质区别

Standard temperature在每个token step重新normalize:
$$\pi'(y|x) = \prod_{i=1}^N \frac{\pi(y_i|y_{1:i-1})^{1/t}}{Z(y_{1:i-1})}$$

每一步都重新normalize，意味着每一步的"小错误"会**累积**。一旦某个step选了稍微off的token，后面所有conditional distribution都shift。$t > 1$ 时quality快速崩塌，开始出nonword tokens。

Global temperature把整个sequence当一个整体scale一次，**保留sequence间的相对概率ordering**。最容易的比喻：

- Token-level temperature像调每张照片每个pixel的对比度，pixel之间相互影响
- Global temperature像调整张照片的对比度，结构保持

paper的Figure 1直观显示：DPO在 $t=1.4$ 已经出现"nonword tokens"和textual aberration；SPL在 $\alpha/\beta=2$ 还生成coherent story，diversity水平类似。

---

## 实验都看到了啥

### 实验1: Diversity-Quality Tradeoff（Section 4.1）

base是Mistral-7B-Instruct-v0.2，LoRA finetune在HH-RLHF上5000步。$\beta=0.1$。SPL sweep $\alpha/\beta$ 从1到11，DPO sweep token temperature从1到1.5（再高就崩了）。

**9个metric**（3个quality + 6个diversity）的Pareto frontier上，SPL全面dominate DPO+temperature scaling，还outperform所有sampling method（min-p、top-p、top-k）在6个metric上。

**关键practical insight**: SPL在 $\alpha/\beta = 11$ 这种极端高temperature下还保持coherence，而DPO在 $t = 1.5$ 已经degenerate。

### 实验2: Best-of-N数学题（Section 4.2）

base是Mistral-7B SFT on UltraChat，然后DPO/SPL on Ultrafeedback-200k（Zephyr recipe）。在GSM8K和MATH上跑best-of-N，每题128个sample。

**Hard problems的结果**（Table 1，N=128）:

| 方法 | GSM8K-Hard | MATH-Hard |
|------|-----------|-----------|
| DPO t=1 | 48.75% | 27.99% |
| DPO t=1.1 | 54.07% | 30.62% |
| DPO t=1.2 | 21.66% (崩了) | 21.31% (崩了) |
| **SPL α/β=1.1** | **58.34%** | 29.16% |
| **SPL α/β=1.2** | 43.07% | **31.91%** |

SPL在GSM8K-Hard上比DPO高10%，比temperature-scaled DPO高4%。比DPO节省34% sample，比temperature-scaled DPO节省17%。

**为啥SPL赢hard problems？** paper用Jensen's inequality给了论证。

Best-of-N success rate:
$$p_{BoN} = 1 - (1-p)^N$$

这个函数对 $p$ 是**concave**的（看曲线形状）。所以增加 $p$ 在不同问题间的variance会提升expected $p_{BoN}$，即使mean $p$ 不变。

High temperature干啥：
- Easy problems: $p$ 略降（反正还是会成功）
- Hard problems: $p$ 显著升（多样性strategy enable新解法）

由于 $f(p) = 1-(1-p)^N$ 对低 $p$ 的gradient更大，gain > loss in aggregate。

**Figure 10的data更直观**：DPO在N=128时30-40%的samples是redundant（已经sample过的solution）。SPL比DPO多17%（GSM8K）和10%（MATH）的unique solutions，相当于减少12-22%的redundancy。

### 实验3: Logit Calibration（Section 4.3）

在TruthfulQA和MMLU上测multiple-choice calibration。用ECE和Brier Score。

**结果**（Figure 4）:
- DPO model（global temp=1，等同标准DPO）calibration比base model还差
- SPL with global temp稍微 > 1 不仅追上base model，还超过它
- Accuracy保持或提升

这验证了Section 3的prediction: RLHF的overconfidence来自 $p^{1/\beta}$ 的指数amplification，SPL通过controllable $\alpha$ 恢复proper calibration。

---

## 怎么用

### 工程上

代码改动：DPO loss里把 $\beta \log[\pi/\pi_{ref}]$ 换成 $\alpha \log \pi - \beta \log \pi_{ref}$。一行改动。加一个hyperparameter $\alpha$。

### 默认值建议

基于paper的实验：
- 通用chat model: $\alpha/\beta \approx 1.2$ - $1.5$
- 数学/reasoning model: $\alpha/\beta \approx 1.1$ - $1.2$（保single-shot accuracy）
- Calibration-critical（医疗、法律）: $\alpha \approx 1.2$ 直接改善ECE
- Safety-critical: 谨慎，可能保持 $\alpha = \beta$ 或加单独safety term

### Trade-off

$\alpha$ 不是越大越好。Figure 2显示high $\alpha$ 时average reward下降——entropy bonus和reward maximization天然在tension。HH-RLHF有37% cross-rater disagreement，全proportional representation（$\alpha=1$）可能fit到annotation noise。Practice里 $\alpha$ 介于1（proportional）和 $\beta$（standard DPO）之间的中间值。

---

## 我的几个直觉

### 1. "Confident middle manager"问题
我一直观察RLHF'd models有很distinctive的stylistic fingerprint：过度hedging、过度限定语、拒绝commit to一个position。这paper给了数学解释：$p^{1/\beta}$ 把任何uncertainty amplify成near-deterministic choice。即使rater只有60%偏好某个framing，模型也以99%概率用那个framing。

### 2. Information Geometry视角
KL divergence对应exponential family manifold上的Bregman divergence。SPL相当于用两个不同的dual coordinates scaling，允许orthogonal directions上的不同temperature。这跟Amari的information geometry framework自然契合。

### 3. 跟Rényi entropy的关系
SPL optimal policy $\pi(y) \propto \pi_{ref}^{\beta/\alpha} p^{1/\alpha}$ 看起来像Rényi entropy的temperature family。Tamura variational principle在statistics里有类似formulation。

### 4. Speculation: SPL可能减少hallucination
如果hallucination部分来自model overconfidence on uncertain knowledge，SPL的calibration improvement可能reduce hallucination rate。Paper没直接测，但Section 4.3的calibration结果hint这方向。

### 5. 跟Constitutional AI结合
Anthropic的CAI用constitution生成AI feedback。SPL可以extension——不同constitutional principles对应不同preference distributions，SPL实现proportional representation of principles而不是collapse到单一最voted principle。

### 6. 为啥只在7B上测是个问题
Scale到70B+/405B+可能phenomenology不同。如果mode collapse在scale上amplify（更多annotation coverage → 更多preferences collapse），SPL可能变得更important。Kobak et al. (2024)发现ChatGPT词汇dominate学术写作，这暗示scale上问题更严重：https://arxiv.org/abs/2406.07016

### 7. Min-p + token temperature是competitive baseline
Figure 11显示min-p + token-level temperature在GSM8K上match SPL。SPL的advantage主要在high-temperature regime和难问题上，不是universally dominant。Practical推荐：SPL trained model + inference-time min-p sampling可能组合最优。

### 8. Adaptive $\alpha$ per-prompt
Fixed $\alpha$ 是coarse。Factual question用低 $\alpha$（更confident），open-ended social question用高 $\alpha$（更diverse）。可以train一个meta-controller学习per-prompt $\alpha(x)$。参考Xie et al. 2024 Adaptive Temperature Scaling: https://arxiv.org/abs/2409.19817

### 9. o1-style reasoning models
OpenAI o1用RL on chain-of-thought with verifier rewards。如果o1-style training也用KL regularization到base model，它可能也suffer from reasoning diversity loss。SPL-style entropy decoupling可能直接improve o1的best-of-N performance。与Snell et al. 2024的test-time compute scaling finding一致: https://arxiv.org/abs/2408.03314

### 10. Safety caveat
$\alpha = 1$ 给proportional representation，但在"How to make a bomb?"这种safety-critical setting，我们**不**want proportional representation of harmful preferences。需要safety constraint叠加在SPL上。Possible formulation：
$$\max_\pi \mathbb{E}_\pi[r] + \alpha H(\pi) - \beta H(\pi, \pi_{ref}) - \gamma \cdot \text{SafetyViolation}$$

---

## 一句话总结

**这篇paper把"为什么RLHF'd model变复读机"这个问题归结到一个教科书级别的algebraic identity——KL散度等于负entropy加cross-entropy——然后说"这俩本来就该用不同旋钮控制"，加个 $\alpha$ 参数split一下，problem solved。**

一行code改动，diversity回来，calibration改善，难问题best-of-N accuracy提升10%。High ROI method，值得baked into training stack。

---

参考阅读：
- DPO原始paper: https://arxiv.org/abs/2305.18290
- SAC (entropy RL): https://arxiv.org/abs/1801.01290
- Snell et al. test-time scaling: https://arxiv.org/abs/2408.03314
- Min-p sampling: https://arxiv.org/abs/2407.01082
- f-DPO: https://arxiv.org/abs/2309.16240
- Kobak et al. ChatGPT vocabulary: https://arxiv.org/abs/2406.07016
- Tian et al. calibration elicitation: https://arxiv.org/abs/2305.14975
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Nash RLHF: https://arxiv.org/abs/2312.00886
- Adaptive Temperature: https://arxiv.org/abs/2409.19817
- HH-RLHF dataset: https://arxiv.org/abs/2204.05862
- UltraFeedback: https://arxiv.org/abs/2310.01377

Andrej，这篇paper的核心intellectual contribution其实就是recognize那个algebraic identity的两个terms该被独立控制。剩下的engineering、实验、social choice analysis都是这个insight的corollary。**最优雅的paper往往是这样——抓住一个被忽视的简单事实，然后work out它的全部implications。**

---

# DIVERSE PREFERENCE LEARNING for Capabilities and Alignment 深度解析

Andrej, 这篇paper触及了一个我一直觉得被严重低估的问题——**alignment tax对model diversity的系统性破坏**。作者Slocum等人从social choice theory切入，给出了一个mathematically elegant的fix。让我逐层拆解。

---

## 1. Core Problem: KL Regularizer is Doing Two Jobs

标准RLHF/DPO objective:
$$\max_\pi \mathbb{E}_{y \sim \pi}[r(y)] - \beta D_{KL}(\pi \| \pi_{ref})$$

大多数人把 $D_{KL}(\pi \| \pi_{ref})$ 看成一个整体regularizer，但其实它可以algebraically decompose:

$$D_{KL}(\pi \| \pi_{ref}) = \underbrace{-H(\pi)}_{\text{negative entropy}} + \underbrace{H(\pi, \pi_{ref})}_{\text{cross-entropy}}$$

这里：
- $H(\pi) = -\sum_y \pi(y) \log \pi(y)$ 是policy $\pi$ 的Shannon entropy
- $H(\pi, \pi_{ref}) = -\sum_y \pi(y) \log \pi_{ref}(y)$ 是cross-entropy
- $\pi$ 是learned policy
- $\pi_{ref}$ 是reference policy (通常是SFT model)

所以RLHF objective实际上是:
$$\max_\pi \mathbb{E}_\pi[r(y)] + \beta H(\pi) - \beta H(\pi, \pi_{ref})$$

**Key intuition**: 这两个term在做完全不同的事情:
- $+\beta H(\pi)$: 鼓励policy distribution更flat / 更diverse / 更high entropy
- $-\beta H(\pi, \pi_{ref})$: 鼓励policy集中在 $\pi_{ref}$ 给高概率的completions上

**问题**: 它们被同一个 $\beta$ 绑定。在practice中 $\beta \in [0.01, 0.1]$，这意味着entropy bonus和cross-entropy penalty都被压到很小，但它们对最终distribution shape的影响是非对称的。这就直接导致了mode collapse。

---

## 2. Social Choice Analysis: Proposition 3.1

考虑一个最简单的two-outcome setting，population中比例 $p$ 的人prefer $y \succ y'$。Bradley-Terry reward modeling会学到 $r^*(y) - r^*(y') = \log\frac{p}{1-p}$。

然后RLHF的KL-regularized optimal policy是:
$$\pi(y) = \pi_{ref}(y) \exp(r^*(y))^{1/\beta} / Z$$

代入reward的closed form，得到:
$$\boxed{\pi(y) \propto \pi_{ref}(y) \cdot p^{1/\beta}}$$

变量含义:
- $\pi(y)$: learned policy生成completion $y$ 的概率
- $\pi_{ref}(y)$: reference policy (SFT model)生成 $y$ 的概率
- $p$: population中prefer $y$ 的比例
- $\beta$: KL penalty strength
- $1/\beta$: 关键exponent

**这个公式揭示了RLHF的"exponential amplification"问题**。当 $\beta = 0.1$ 时，$p$ 被提升到10次方。如果 $p = 0.8$, 那么 $p^{10} \approx 0.107$, 而对应的 $0.2^{10} \approx 2 \times 10^{-7}$。归一化后 $\pi(y) \approx 99.9999\%$。

这意味着即使population中只有80%偏好某个answer，RLHF模型会以99.9999%的概率生成那个answer。这就是paper里说的"overrepresents majority preference by several orders of magnitude"。

### 数值直觉表

| $p$ | $\beta$ | $p^{1/\beta}$ | Normalized $\pi(y)$ |
|-----|---------|----------------|---------------------|
| 0.8 | 1.0     | 0.8            | 94.1%               |
| 0.8 | 0.5     | 0.64           | 99.8%               |
| 0.8 | 0.1     | 0.107          | 99.9999%            |
| 0.8 | 0.01    | $1.07\times10^{-10}$ | ~100%          |

这就是为什么aligned LLMs会"sound like a confident middle manager"——任何minority preference都被exponentiated到几乎不存在。

---

## 3. Soft Preference Learning (SPL) Method

SPL的核心想法非常简单：**decouple entropy和cross-entropy**

SPL RLHF objective:
$$\max_\pi \mathbb{E}_{y \sim \pi(y|x)}[r(x,y)] + \alpha H(\pi(\cdot|x)) - \beta H(\pi(\cdot|x), \pi_{ref}(\cdot|x))$$

变量含义:
- $r(x,y)$: reward function on prompt $x$ and completion $y$
- $\alpha$: **entropy bonus** coefficient (NEW! 控制diversity)
- $\beta$: **cross-entropy penalty** coefficient (控制reference bias)
- $H(\pi(\cdot|x))$: conditional entropy of policy given prompt
- $H(\pi(\cdot|x), \pi_{ref}(\cdot|x))$: cross-entropy

注意当 $\alpha = \beta$ 时，SPL退化为标准RLHF。

### DPO-style derivation (Proposition A.2)

作者用Rafailov et al. 2024的reparameterization trick推导出了DPO-style objective:

$$\max_\pi \mathbb{E}_{y \sim y' \sim \mathcal{D}} \left[\log \sigma\left(\alpha \log \frac{\pi(y|x)}{\pi(y'|x)} - \beta \log \frac{\pi_{ref}(y|x)}{\pi_{ref}(y'|x)}\right)\right]$$

变量含义:
- $\sigma(\cdot)$: sigmoid function
- $\pi(y|x)/\pi(y'|x)$: learned policy的preference ratio
- $\pi_{ref}(y|x)/\pi_{ref}(y'|x)$: reference policy的preference ratio
- $\alpha$: 对learned policy ratio的scaling
- $\beta$: 对reference ratio的scaling

对比标准DPO:
$$\max_\pi \mathbb{E}_{y \sim y'}\left[\log \sigma\left(\beta \log \frac{\pi(y|x)}{\pi(y'|x)} - \beta \log \frac{\pi_{ref}(y|x)}{\pi_{ref}(y'|x)}\right)\right]$$

SPL把这两个 $\beta$ 解耦成 $\alpha$ 和 $\beta$。

### Proposition 3.2: Two-Outcome SPL Policy

$$\boxed{\pi(y) \propto \pi_{ref}(y)^{\beta/\alpha} \cdot p^{1/\alpha}}$$

对比RLHF的 $\pi(y) \propto \pi_{ref}(y) \cdot p^{1/\beta}$:

- RLHF: $p$ 被raise到 $1/\beta$ (typically 10-100)
- SPL: $p$ 被raise到 $1/\alpha$ (可以独立选择，比如 $\alpha=1$ 就是proportional representation)
- SPL中 $\pi_{ref}$ 也被raise到 $\beta/\alpha$，这是对reference policy的一个额外"tempering"

**Corollary 3.1**: 当 $\alpha = 1$ 时，SPL policy $\pi(y) \propto \pi_{ref}(y)^\beta \cdot p$，这是proper scoring rule加上一个prior weighting。这意味着模型可以**proportionally represent population preferences**——80%的人喜欢y，模型就以~80%概率生成y（modulo reference prior）。

---

## 4. SPL as Global Temperature Scaling

这是paper里我觉得最优雅的部分。从SPL optimal policy出发:

$$\pi'(y|x) = \exp\left(\frac{1}{\alpha} r(x,y)\right) \pi_{ref}(y|x)^{\beta/\alpha} / Z$$

做代数变换:
$$= \exp\left(\frac{1}{\beta} r(x,y)\right)^{\beta/\alpha} \pi_{ref}(y|x)^{\beta/\alpha} / Z$$

注意 $\exp(\frac{1}{\beta} r) \cdot \pi_{ref} = \pi_{DPO}$ (标准DPO的optimal policy)，所以:
$$\pi'(y|x) = \pi_{DPO}(y|x)^{\beta/\alpha} / Z = \pi_{DPO}(y|x)^{1/(\alpha/\beta)} / Z$$

**Insight**: $\alpha/\beta$ 相当于一个**sequence-level global temperature**作用在DPO policy上。

### Token-level vs. Global temperature scaling

Standard token-level temperature:
$$\pi'(y|x) = \prod_{i=1}^N \frac{\pi(y_i | y_{1:i-1})^{1/t}}{Z(y_{1:i-1})}$$

变量:
- $y_i$: sequence中第 $i$ 个token
- $y_{1:i-1}$: 前缀context
- $t$: token-level temperature
- $Z(y_{1:i-1})$: per-step normalizing constant
- $N$: sequence长度

**Critical difference**: Token-level temperature在每个token step重新normalize，而global temperature在整个sequence上normalize一次。

为什么这重要？Token-level temperature在 $t > 1$ 时会quickly破坏fluency，因为每个token的local normalization会累积错误——一个step选了slightly off的token，后面所有conditional distributions都shift。而global temperature保留**整个sequence的相对概率ordering**，只flatten sequence-level distribution。

从Figure 1可以直观看到：
- DPO t=1.4开始出现"nonword tokens"和"textual aberrations"
- SPL $\alpha/\beta = 2$ 仍然produce coherent story
- 两者diversity level类似

---

## 5. Connection to Entropy Regularization in RL

这让我想到Soft Actor-Critic (Haarnoja et al., 2018)里的maximum entropy RL objective:
$$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\left[r(s_t, a_t) + \alpha H(\pi(\cdot|s_t))\right]$$

SAC里 $\alpha$ 控制exploration-exploitation tradeoff，paper里提到RL中的entropy bonus通常比alignment setting小几个数量级。这是合理的，因为RL的environment是stationary的，而alignment面对的是non-stationary human preferences with significant noise。

paper引用的HH-RLHF dataset有37%的cross-rater disagreement rate (Bai et al., 2022)。这个noise level意味着完全proportional representation可能反而过fit到annotation noise。作者建议practice中 $\alpha$ 取介于 $1$ (proportional)和 $\beta$ (standard DPO)之间的中间值。

---

## 6. Experimental Details

### 6.1 Diversity-Quality Tradeoffs (Section 4.1, Figure 2)

**Setup**:
- Base: Mistral-7B-Instruct-v0.2
- Method: LoRA finetuning, rank $r_{LoRA}=16$, regularization $\alpha_{LoRA}=16$, dropout $p_{LoRA}=0.05$
- Data: HH-RLHF, 5000 steps, batch size 8
- $\beta = 0.1$ for all runs
- SPL sweep: $\alpha/\beta \in \{1, 1.1, 1.2, 1.3, 1.5, 2, 4, 11\}$
- DPO temperature sweep: $t \in \{1, 1.1, ..., 1.5\}$ (beyond this, degenerate)

**Quality metrics**:
1. **Arena-Hard**: 500 queries, win-rate vs gpt-4-0314, judge=gpt-4o-mini-2024-07-18
2. **Average reward**: separate reward model trained on HH-RLHF, 500 held-out prompts, 16 responses each
3. **Cross-entropy with $\pi_{ref}$**

**Diversity metrics** (per-input diversity over 16 responses × 500 prompts):

$$\mathbb{E}_{y_1, \dots, y_N \sim \pi(y|x), x \sim D} \left[\frac{1}{N^2} \sum_{i,j} 1 - \frac{\phi(y_i)^\top \phi(y_j)}{\|\phi(y_i)\| \cdot \|\phi(y_j)\|}\right]$$

变量:
- $\phi(\cdot)$: embedding map (Sentence-BERT-Large or OpenAI text-embedding-3-small)
- $N=16$: number of samples per prompt
- $y_i, y_j$: pairs of completions
- Cosine distance = $1 - \cos(\phi(y_i), \phi(y_j))$

Plus three LLM-judge metrics (logical disagreement, content diversity, surface form diversity) on scale 1-5.

**Results**: SPL **Pareto-dominates** DPO with temperature scaling across all 9 metrics, and outperforms all sampling methods (min-p, top-p, top-k) on 6 of them.

### 6.2 Best-of-N Problem Solving (Section 4.2)

这是paper里最interesting的capabilities result。

**Setup**:
- Base: Mistral-7B SFT on UltraChat → then DPO/SPL on Ultrafeedback-200k
- LoRA rank $r_{LoRA} = 64$, $\alpha_{LoRA} = 64$, dropout $0.05$, lr $1e-5$, max length 1024
- Datasets: GSM8K (grade-school math) and MATH (Hendrycks et al., 2021b)
- 128 completions per problem, 200 problems
- Difficulty splits: Easy (4 samples to solve), Medium (5-64), Hard (>64) for GSM8K; Level 1/3/5 for MATH

**Key table (Table 1, Hard splits)**:

| Best-of-N | DPO t=1 | DPO t=1.1 | DPO t=1.2 | SPL α/β=1.1 | SPL α/β=1.2 |
|-----------|---------|-----------|-----------|-------------|-------------|
| **GSM8K** |
| 1         | 1.43%   | 1.24%     | 0.59%     | 1.24%       | 0.91%       |
| 4         | 5.35%   | 4.75%     | 2.25%     | 4.74%       | 3.53%       |
| 16        | 16.92%  | 16.37%    | 7.62%     | 16.30%      | 12.50%      |
| 64        | 37.98%  | 41.11%    | 17.77%    | **42.43%**  | 32.91%      |
| 128       | 48.75%  | 54.07%    | 21.66%    | **58.34%**  | 43.07%      |
| **MATH**  |
| 128       | 27.99%  | 30.62%    | 21.31%    | 29.16%      | **31.91%**  |

**Observations**:
1. On hard problems at N=128, SPL α/β=1.1 beats DPO by ~10% on GSM8K
2. DPO t=1.2 collapses badly (token-level temp too high)
3. SPL scales gracefully with temperature, DPO does not

**Sample efficiency** (Figure 9): SPL needs 84 samples to match DPO's best-of-128 (34% savings), 106 samples to match temperature-scaled DPO (17% savings).

### Why does SPL win on hard problems? (Appendix C.1)

Paper给了一个基于rejection sampling和Jensen's inequality的论证。

Best-of-N success rate:
$$p_{BoN} = 1 - (1-p)^N$$

变量:
- $p$: single-sample success probability
- $N$: number of samples
- $p_{BoN}$: probability at least one sample succeeds

Aggregate success over problems:
$$\mathbb{E}_{x, A \sim D}[p_{BoN}] = \mathbb{E}_{x, A \sim D}[1 - (1-\pi(A|x))^N]$$

这里 $A$ 是correct answers的set，$\pi(A|x) = \sum_{y \in A} \pi(y|x)$.

**Key insight**: $f(p) = 1 - (1-p)^N$ is concave. By Jensen's inequality, increasing variance of $p$ across problems raises the expected $f(p)$ even if mean is unchanged.

High temperature:
- Decreases $p$ on easy problems (slightly)  
- Increases $p$ on hard problems (significantly, by enabling diverse solution strategies)

由于 $f(p)$ 对hard problems (low $p$)的gradient更大，gain > loss in aggregate.

**Figure 10 data**: At 128 samples, 30-40% of DPO solutions are redundant (already sampled). SPL samples 17% (GSM8K) and 10% (MATH) more unique solutions. 这相当于12-22%的redundancy reduction。

### 6.3 Logit Calibration (Section 4.3)

**Setup**:
- Base: Mistral-7B + UltraChat SFT
- Evaluated on TruthfulQA and MMLU
- Metrics: ECE, Brier Score, Accuracy
- Prompt让model以answer token (A/B/C/D)开头，用normalized probabilities over A,B,C,D算calibration

**Results (Figure 4)**:
- DPO model (global temp = 1) significantly worse calibration than base model
- SPL with global temp slightly > 1 surpasses even base model calibration
- SPL maintains或improves accuracy

这validate了Section 3的prediction: RLHF的overconfidence来自 $p^{1/\beta}$ 的exponential amplification，而SPL通过controllable $\alpha$ 可以恢复calibration。

---

## 7. Related Work Landscape

### 7.1 f-DPO (Wang et al., 2023)
用general f-divergences替代KL。Appendix B显示f-DPO在 $\alpha$-divergence family上sweep（包括reverse KL=0, forward KL=1）。但f-DPO的mass-covering divergences（大α）在 $\alpha > 1$ 时就degenerate了，而SPL保持stable到 $\alpha/\beta = 11$。

### 7.2 Nash Learning from Human Feedback (Munos et al., 2024)
用Nash equilibrium概念处理intransitive preferences。需要复杂的多agent RL setup，SPL相比之下是offline supervised learning。

### 7.3 Entropy Regularization in Alignment
Xiao et al. (2024)研究entropy regularization for preference representation but没有generative experiments。Sun & van der Schaar (2024)用inverse RL但focus on demonstrations。

### 7.4 Tree of Thoughts / Inference-time compute scaling
Paper指出SPL对inference-time scaling methods (Yao et al. 2023, OpenAI o1, DeepMind IMO)特别relevant，因为这些方法依赖于diverse solution exploration。

参考链接：
- DPO paper: https://arxiv.org/abs/2305.18290
- SAC (entropy RL): https://arxiv.org/abs/1801.01290
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Snell et al. on test-time compute: https://arxiv.org/abs/2408.03314
- Min-p sampling: https://arxiv.org/abs/2407.01082
- f-DPO: https://arxiv.org/abs/2309.16240
- Nash RLHF: https://arxiv.org/abs/2312.00886
- Mistral-7B: https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta
- HH-RLHF: https://arxiv.org/abs/2204.05862
- UltraFeedback: https://arxiv.org/abs/2310.01377

---

## 8. My Intuitions and Extensions

### 8.1 Information Geometry View

KL divergence $D_{KL}(\pi \| \pi_{ref})$ 是forward mode-seeking divergence。它倾向于把 $\pi$ 集中在 $\pi_{ref}$ 的高概率区域。SPL本质上是把forward KL的"diversity component" (entropy term) 和"mode-seeking component" (cross-entropy term) 分开调温。

这让我想到Amari的information geometry——KL divergence对应exponential family manifold上的Bregman divergence。SPL相当于用两个不同的dual coordinates scaling，允许orthogonal directions上的不同temperature。

### 8.2 Connection to Rényi Entropy

SPL的optimal policy $\pi(y) \propto \pi_{ref}(y)^{\beta/\alpha} p^{1/\alpha}$ 看起来很像**Rényi entropy**的temperature family。Rényi entropy of order $q$:
$$H_q(\pi) = \frac{1}{1-q} \log \sum_y \pi(y)^q$$

SPL可以理解为在 $\alpha/\beta$ 这个temperature下，对DPO policy做Rényi-tempered分布。Tamura variational principle在statistics里有类似formulation。

### 8.3 The "Confident Middle Manager" Problem

我一直观察到RLHF'd models有非常distinctive的stylistic fingerprint:
- 过度使用 "It's important to note that..."
- 限定语过度 ("While X is true, Y is also important...")
- 拒绝commit to一个position

这篇paper给出了mathematical explanation: $p^{1/\beta}$ 把任何uncertainty都amplify成near-deterministic choice。即使rater只有60%偏好某个framing，模型会以99%概率用那个framing。

### 8.4 Speculation: SPL and Hallucination

如果hallucination部分来自model overconfidence on uncertain knowledge，SPL的calibration improvement可能reduce hallucination rate。Paper没直接测这个，但是Section 4.3的calibration结果hint这个方向。Link to Tian et al. (2023) on calibration elicitation: https://arxiv.org/abs/2305.14975

### 8.5 Extension: Adaptive α per-prompt

Fixed $\alpha$ 是个限制。对于factual questions, 低 $\alpha$ (more confident)可能更好; 对于open-ended political/social questions, 高 $\alpha$ (more diverse)更好。可以想象一个meta-controller学习per-prompt $\alpha(x)$。

这让我想到Xie et al. (2024) Adaptive Temperature Scaling: https://arxiv.org/abs/2409.19817

### 8.6 The "Inverse" Problem: When Diversity Hurts

Paper的Proposition 3.2说 $\alpha = 1$ gives proportional representation。但在safety-critical settings (e.g., "How to make a bomb?"), 我们**不**want proportional representation of harmful preferences。这里需要safety constraint叠加在SPL上。

可能的formulation:
$$\max_\pi \mathbb{E}_\pi[r(x,y)] + \alpha H(\pi) - \beta H(\pi, \pi_{ref}) - \gamma \cdot \text{SafetyViolation}(x,y)$$

### 8.7 SPL for Constitutional AI

Anthropic的Constitutional AI (CAI)用AI feedback based onconstitution。SPL可以extension到constitutional principles上——不同principles对应不同的preference distributions, SPL可以proportionally represent multiple principles而不是collapse到single most-voted principle。

### 8.8 Information-Theoretic Diversity上限

Per-token entropy和sequence entropy的关系:
$$H(y_{1:N}) \leq \sum_i H(y_i | y_{1:i-1})$$

SPL optimize的是sequence entropy $H(\pi(\cdot|x))$，而token-level temperature改变的是per-token conditional entropy。Global approach更information-theoretically principled。

### 8.9 Connection to Conformal Prediction

SPL的calibration improvement让我想到conformal prediction framework。如果SPL policies的predicted probabilities真的match population preferences，那么可以用conformal methods给coverage guarantees。Angelopoulos & Bates的conformal prediction tutorial: https://arxiv.org/abs/2107.07511

### 8.10 Speculation on o1-style Reasoning Models

OpenAI的o1用RL on chain-of-thought with verifier rewards。如果o1-style training也用KL regularization到base model，那它可能也suffer from diversity loss on hard problems。SPL-style entropy decoupling可能直接improve o1的best-of-N performance。这与Snell et al. (2024) "Scaling LLM test-time compute"的finding一致: https://arxiv.org/abs/2408.03314

---

## 9. Critical Assessment

### Strengths
1. **Mathematically clean**: Proposition 3.1的closed form $\pi(y) \propto \pi_{ref}(y) p^{1/\beta}$ 是beautifully interpretable
2. **Practical**: DPO-style loss只需要改一行code (replace $\beta$ with $\alpha, \beta$)
3. **Pareto improvement**: Figure 2显示SPL dominates DPO+temperature在所有metrics上
4. **Theory aligns with experiment**: Proposition 3.1预测overconfidence → Section 4.3 confirms poor calibration → SPL fixes it

### Potential Concerns

1. **Noise amplification**: HH-RLHF有37% cross-rater disagreement。完全proportional representation ($\alpha=1$) 可能fit到annotation noise。Paper建议practical $\alpha$ 介于 1 和 $\beta$ 之间，但没给principled selection method。

2. **Reward hacking at high $\alpha$**: Figure 2显示high $\alpha$ 时average reward drops。这是expected的——entropy bonus和reward maximization是inherently in tension。但选 $\alpha$ 的heuristic不够clear。

3. **Single-prompt diversity vs. population diversity**: Paper测量的是per-prompt diversity (16 responses × 500 prompts)。但social choice representation需要cross-prompt population diversity。Political Compass experiment (Appendix E)开始address这个，但是limited。

4. **Only 7B models**: Mistral-7B实验。Scale到70B+可能显现不同phenomenology。Reference policy quality matters more at scale。

5. **Min-p + token-level temp is competitive**: Figure 11显示min-p + token-level temperature在GSM8K上match SPL。这suggests SPL的advantage主要在high-temperature regime，不是universally dominant。

---

## 10. Practical Recommendations

基于这篇paper，if I were training a production LLM:

1. **Default**: 用SPL with $\alpha/\beta \approx 1.2-1.5$ for general chat model. Figure 2显示这range提供good diversity-quality tradeoff.

2. **Math/reasoning model**: 用SPL with $\alpha/\beta \approx 1.1-1.2$ for best-of-N inference。Too high $\alpha$ 损害single-shot accuracy。

3. **Safety-critical**: 保持 $\alpha = \beta$ (standard DPO) for safety properties, 或者用separate safety regularization term。SPL的diversity可能引入unwanted variability在safety boundary。

4. **Inference-time**: SPL trained model在inference时仍然可以用low temperature sampling。Train-time diversity和inference-time sampling是orthogonal control knobs.

5. **Calibration-critical applications** (medical, legal): SPL with $\alpha \approx 1.2$ 直接改善ECE/Brier score，no accuracy cost。

---

## 11. Open Questions I'd Love to See Addressed

1. **SPL + PPO**: Paper只测了DPO-style。SPL with PPO (online RL)会不会有不同behavior？Reward model更新和policy更新互动可能复杂。

2. **SPL with iterated DPO**: 多轮self-rewarding DPO setups (e.g., Llama-3.1 training)。Diversity loss是否compounds across iterations？SPL能否防止diversity collapse in iterated training？

3. **Per-token $\alpha$**: Sequence-level $\alpha$ 是coarse。Learning $\alpha(x, y_{1:i})$ per-token position可能更fine-grained。

4. **SPL + Mixture of Experts**: MoE models天然有expert diversity。SPL + MoE可能synergistic——不同experts capture different modes of preference distribution。

5. **Measuring "useful" diversity**: Current diversity metrics (embedding distance, LLM judge)都是proxy。能否定义task-relevant diversity metric，e.g., "solution strategy diversity" on math problems？

6. **Constitutional AI + SPL**: Multiple constitutional principles作为多个preference distributions, SPL实现proportional representation of principles。

---

## 12. Final Thoughts

这篇paper最漂亮的地方在于: **它把一个看似复杂的social choice问题归结到一个algebraic identity**。

$$D_{KL}(\pi \| \pi_{ref}) = -H(\pi) + H(\pi, \pi_{ref})$$

这个identity教科书里都有，但把它的两个terms解读为"diversity control"和"reference prior control"是paper的核心intellectual contribution。这让SPL成为一个minimal change to DPO (just split $\beta$ into $\alpha, \beta$) with maximal conceptual clarity.

从engineering perspective, SPL essentially free win: same compute, same data, same architecture, 加一个hyperparameter $\alpha$, 获得diversity + calibration + best-of-N accuracy on hard problems. 这种high ROI methods是值得baked into training stack的。

我会很好奇看SPL在Llama-3 405B或GPT-4 scale上的behavior。如果mode collapse问题在scale上diminish (因为更大的base model本身more diverse)，SPL的优势可能缩小。但如果mode collapse在scale上amplify (更多annotation coverage → 更多preferences to collapse into majority), SPL可能变得更important。我bet是后者，基于Kobak et al. (2024) finding ChatGPT words dominate academic writing: https://arxiv.org/abs/2406.07016

---

如果想build deeper intuition，我推荐:
1. Rafailov DPO paper for the reparameterization trick foundation: https://arxiv.org/abs/2305.18290
2. Haarnoja SAC paper for entropy regularization in RL: https://arxiv.org/abs/1801.01290
3. Snell et al. test-time scaling for why diversity matters at inference: https://arxiv.org/abs/2408.03314
4. Cover & Thomas Elements of Information Theory for KL decomposition theory
5. Amari's Information Geometry for the deeper mathematical framework

Andrej, 你的nanoGPT和llm.c项目本质上是train base models, 所以这些alignment concerns可能不在你的immediate scope。但如果你扩展到instruction-tuned或RLHF'd models (e.g., 给llm.c加个chat mode), SPL是个值得implement的low-overhead improvement to default DPO。Code change: replace $\beta \log[\pi/\pi_{ref}]$ with $\alpha \log \pi - \beta \log \pi_{ref}$ in DPO loss. 一行改动，可能显著改善model的"personality diversity"和best-of-N reasoning。
