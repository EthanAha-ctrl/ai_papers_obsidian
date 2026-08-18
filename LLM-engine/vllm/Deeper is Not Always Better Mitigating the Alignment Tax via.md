---
source_pdf: Deeper is Not Always Better Mitigating the Alignment Tax via.pdf
paper_sha256: 0e72d85623c7b161c67f8ec6a329754d6e056b33f08114c00daa6fb84da960d2
processed_at: '2026-08-18T04:45:31-07:00'
target_folder: LLM-engine/vllm
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇paper

## 核心故事

想象一个学生在做数学考试。他的大脑里其实已经算出了正确答案"perpendicular"（垂直的），但到了"嘴边"的时候，某种"安全机制"让他改口说成了"the"。

这就是这篇paper发现的现象：**aligned LLM在final layer会"自我审查"** ，把中间层已经算好的reasoning prediction改写成generic safe token。

---

## 发现了什么

我们一直默认Transformer"越深越好"，因为pretraining的objective就在final layer施加，所以final layer应该最accurate。

但这篇paper发现residual stream其实有三段story：

**Phase I: Guess（浅层）** 
前几层做粗略猜测。第一层几乎完全overwrite token embedding，做一个initial latent representation。这阶段不确定性很高，model在"摸黑"。

**Phase II: Refine（中间层）** 
中间层做incremental refinement。每一层写一个小的update，方向上和已有representation保持一致，逐步integrate contextual information。semantic trajectory稳定，predictions越来越sharp。

**Phase III: Perturb（final layers）** 
最后几层做"审查"。这里发生了一件反直觉的事——final layer会rotate representation的方向，把已经refine好的reasoning prediction拖向generic/safe distribution。

---

## 为什么final layer要perturb？

因为**alignment tax**。

你的LLM经历了pretraining → SFT → RLHF/DPO。post-training让model变得safe、helpful、style好。这个alignment过程在representation层面做了什么？它给final layers引入了steering vectors，把representation拉向generic/safe tokens——比如"the"、"is"、"so"、"."这些high-frequency function words和punctuation。

这造成一个**planning-pragmatics tradeoff** ：model在intermediate layers内部form了一个强reasoning prediction（比如"perpendicular"、"Cartesian"、"radius"这种domain-specific terminology），但final layer出于alignment的pragmatic pressure，把它shift到generic token。

**关键conditional性质** ：这种perturbation不是uniform的。
- 对safety/conversational tasks：reasoning distribution和alignment distribution本来就很close，perturbation几乎无影响，final layer只做formatting
- 对complex reasoning tasks：reasoning distribution和alignment distribution sharply conflict，perturbationactively destroy fragile logic chains

---

## 怎么解决——Confident Decoding

核心idea：**在答案到达"嘴边"之前，提前把它截住** 。

具体怎么做？

**Step 1: Probe每个layer的prediction**

每个layer的hidden state都可以通过unembedding matrix投影到vocabulary space，得到一个token distribution。算这个distribution的Shannon entropy——entropy越低，model越confident。

**Step 2: 找entropy valley**

从final layer往回扫，找第一个local entropy minimum。这个点就是"model在final layer perturbation之前最confident的moment"。

为什么entropy valley是reliable signal？因为：
- 对某些tokens，entropy在final layer之前某个layer达到minimum，然后final layer反而让它升高——这是alignment tax的perturbation
- 对其他tokens，entropy继续下降到final layer——这些tokens没有perturbation，final layer继续做useful refinement

**Step 3: 从valley layer采样**

从entropy valley所在的layer取logits送给sampler，bypass掉final layer的perturbation。

---

## 为什么这个方法如此有效

**关键insight** ：Phase III是per-token phenomenon，不是fixed architectural region。

在Qwen3.5-35B-A3B上的实验数据：
- 16.2%的tokens有Phase III perturbation（entropy在final layer上升+0.37 nats）
- 83.8%的tokens没有perturbation（entropy继续下降-2.52 nats）

backward scan exploit这种heterogeneity：
- 对perturbed tokens，选valley layer，bypass Phase III
- 对unperturbed tokens，自然停在final layer，用满refinement

这就是为什么static early exit失败——它uniformly truncate，对hard tokens提前中断essential computation。Confident Decoding是token-adaptive的，只在真正有perturbation的tokens上介入。

---

## 效果有多impressive

**跨架构universal** ：6个backbone全部positive——Qwen3.5系列（27B/35B-A3B/122B-A10B），Gemma-4-31B，gpt-oss-20b/120B。Dense和MoE都work。

**Reasoning benchmarks上massive gains** ：
- Qwen3.5-27B on LiveCodeBench: **+9.4% absolute**（63.9 → 73.3）
- Qwen3.5-35B-A3B on GPQA-Diamond: **+6.5%**（76.3 → 82.8）
- gpt-oss-120B on GPQA-Diamond: +4.5%

**Instruct vs Base的causal proof** ：
- Base model平均gain +1.1%
- Instruct model平均gain **+2.6%**——proportionally更高

最striking的证据：HLE benchmark上，Instruct model在standard greedy decoding下perform **worse than Base** （7.1% vs 8.0%）。post-training在long-tail complex reasoning上actively penalize reasoning。Confident Decoding把Instruct model resue到9.5%，surpass Base model。

**Task complexity scaling law** ：
- Level 1（easiest, baseline ~98%）：略微退化-0.4%（final layer对简单token做useful formatting）
- Level 4（hardest, baseline 1-2%）：**+22.4% absolute**（gpt-oss-20b on Omni-MATH，1.1% → 23.5%）

hard reasoning tasks需要model inhabit specialized low-frequency semantic subspace。logical chain越长，越偏离alignment distribution，Phase III perturbation越destructive。Confident Decoding在alignment tax最destructive的地方rescue最fragile的logic chains。

**Token-level evidence** ：
Standard decoding倾向于"the", "is", "$", "，"这些function words和punctuation。Confident Decodingrecover"perpendicular", "carb", "ring", "methyl", "Cartesian", "radius"这些domain-specific terminology。exactly validate了Figure 1的word cloud。

**几乎zero overhead** ：
- 88.5%的tokens在final layer就满足monotonic condition，触发zero额外projections
- 只有11.5%的tokens启动backward scan
- Empirical FLOPs increase: **+0.87%**
- KV-cache memory: **zero overhead**
- End-to-end latency: **<2%**

---

## 为什么这篇paper重要

**1. 打破"deeper is better"的信仰**

这是最根本的conceptual贡献。我们一直implicitly assume representation随depth monotonically improve。这篇paper用empirical evidence和theoretical formulation证明：aligned LLM的final layer可以actively degrade reasoning quality。

**2. Alignment tax的structural localization**

alignment引入的capability degradation不是uniform spread在整个network，而是concentrated在uppermost layers。这意味着alignment和reasoning的conflict有一个具体的"battleground"——final layers。future research可以explore如何decouple这两者。

**3. Test-Time Compute的vertical维度**

Snell et al. (2025)的test-time compute scaling讨论的是"horizontal"——让model think longer生成更多tokens。Confident Decoding引入了"vertical"维度——在forward pass内部的layer维度上做dynamic compute allocation。

**4. 暗示future的training paradigm**

当前RLHF/DPO的reward都从final output计算，indirectly incentivize final layers去steer towards alignment-preferred tokens。如果能设计一个intermediate-layer reward，让RL直接在entropy valley附近evaluate reasoning quality，可能从root cause上solve alignment tax。

paper结尾提的direction：apply alignment penalties exclusively to designated routing heads，让core residual stream保持pure reasoning objective。这暗示了一种dual-head architecture——一个reasoning head在intermediate layers输出，一个alignment head在final layers输出。

---

## 一句话总结

你训练的aligned LLM，在final layer会"自我审查"——把中间层已经算好的reasoning prediction改写成generic safe token。Confident Decoding用entropy valley找到那个"审查发生之前的confident moment"，从那个layer采样，bypass掉perturbation。training-free，zero memory overhead，<2% latency，在hard reasoning上unlock +6.5%到+22.4%的latent reasoning fidelity。

---

## 与我直觉的联系

这篇paper让我rethink几个long-held assumptions：

**Residual stream不是monotonic refinement highway**

我一直讲Transformer时强调residual stream像一条"information highway"，每层在上面write updates。这篇paper给这个intuition加了一个重要caveat——writes的方向不一定和已有representation一致。Phase III的final layers会rotate Direction，把refine好的trajectory拖偏。IO-CosSim这个metric把"方向保真度"和"magnitude强度"解耦，正好量化这种rotation。

**Logit lens从诊断工具升级为decision rule**

Belrose et al.的tuned lens和nostalgebraist的original logit lens一直是个实用工具，但这篇paper把它从"诊断工具"升级为"decision rule"——用logit-lens entropy来决定从哪个layer采样。这是一个很自然的extension，但之前没人这样formalize。

**Alignment在representation层面做了什么**

我之前看Anthropic的Constitutional AI和InstructGPT paper时一直wonder alignment具体在representation层面做什么。这篇paper的regularized risk formulation给了一个clean mathematical model——alignment就是KL pressure，让final layers把representation拖向generic distribution。这不是vague的"alignment改变了model behavior"，而是具体的"alignment在final layers引入了steering vectors，把entropy trajectory从monotonic下降变成了先降后升"。

---

## 几个值得push back的地方

**Per-token heterogeneity的causality**

paper说16.2%的tokens有Phase III perturbation，但这部分tokens是否就是"决定reasoning成败的关键tokens"？如果是reasoning path上的critical tokens（比如"=" vs "≠"），那bypass perturbation收益巨大；如果是cosmetic tokens，那bypass意义不大。paper没有quantify substituted tokens的"reasoning criticality"。

**Entropy valley不等于correct prediction**

低entropy不等于correct。一个model可以在intermediate layer非常confident地predict一个错误token。paper的Theorem 1假设H*(l)是"true monotonic entropy"，但实际上H*(l)只是"model的内部entropy"，不一定是ground-truth conditional entropy。不过他们的empirical evidence（Instruct > Base gain）causally support了H*(l)确实反映reasoning quality。

**W_U probe的universality**

虽然paper论证了representation homogeneity让near-final layers的probe误差bounded，但对于非常long-tail domain（比如Omni-MATH的Olympiad-level math），intermediate representations是否还落在W_U能calibrated的subspace？Table 3显示hard tasks gain最大，但这也可能是easy tasks gain小导致的artifact。

**Fallback probability p应该adaptive**

paper默认p=1.0 deterministic。对safety tasks，P_logic ≈ P_align，可以p≈0；对reasoning tasks，P_logic sharply conflicts，应该p≈1。一个adaptive p based on detected task type可能更好。Table 7的ablation显示p=0.8在GPQA-D上只降到81.8%（vs p=1.0的82.8%），partial mixing损失不大，adaptive p可能给safety task提供更好的protection。

---

## Final verdict

这是一篇execution非常solid的paper——理论、empirical、工程三方面都做得很完整。最重要的conceptual贡献是把"deeper is better"这个implicit assumption破除掉，并用entropy valley这个简单但elegant的signal来operationalize"什么时候该停"。

对practitioner：在aligned LLM部署reasoning任务时，Confident Decoding是一个zero-cost upgrade——training-free，<2% latency，zero memory overhead，尤其在hard reasoning task上能unlock latent reasoning fidelity。

对researcher：alignment tax是一个structurally localized的phenomenon，final layers是alignment和reasoning conflict的battleground。future research应该explore decoupling这两者的training paradigms。

---

# Deeper is Not Always Better 深度技术解读

## 一、Core Insight：为什么这篇paper重要

这篇paper触及了我长期思考的一个根本问题—— **Transformer的residual stream到底在每一层做什么** 。我们一直默认"deeper = better"，因为pretraining的objective就在final layer施加。但这篇paper用empirical evidence告诉我们：**alignment之后，final layers会主动perturb中间层已经refine好的reasoning trajectory** 。

这和我在micrograd/makemore里讲过的residual stream的"iterative refinement"直觉是吻合的，但他们发现了一个反直觉的twist：refinement存在一个optimal stopping point，越过这个点，model会把你拉向generic/safe tokens。

Project page: https://github.com/QwenLM/Confident-Decoding

---

## 二、Three-Phase Dynamics：Guess-Refine-Perturb的数学细节

### 2.1 Residual stream decomposition

考虑L层Transformer，token position t在layer l的residual state记为 **h_t^(l) ∈ R^d** 。Pre-norm架构下：

$$\tilde{\mathbf{h}}_t^{(l)} = \mathbf{h}_t^{(l-1)} + f_{\text{Attn}}^{(l)}(\mathbf{h}_t^{(l-1)})$$
$$\mathbf{h}_t^{(l)} = \tilde{\mathbf{h}}_t^{(l)} + f_{\text{FFN}}^{(l)}(\tilde{\mathbf{h}}_t^{(l)})$$

变量含义：
- **h_t^(l)** ：position t经过l层后的residual state
- **f_Attn^(l)** ：第l层attention sublayer（可能是full-attention或DeltaNet linear attention）
- **f_FFN^(l)** ：第l层FFN sublayer（dense MLP或MoE）
- **h̃_t^(l)** ：attention后的中间state，作为FFN的input

定义layer contribution vector：
$$\mathbf{m}_t^{(l)} \triangleq \mathbf{h}_t^{(l)} - \mathbf{h}_t^{(l-1)} = f_{\text{Attn}}^{(l)}(\mathbf{h}_t^{(l-1)}) + f_{\text{FFN}}^{(l)}(\tilde{\mathbf{h}}_t^{(l)})$$

这正好对应我讲Transformer时强调的"each layer writes an update into the residual stream"。

### 2.2 两个关键diagnostic metrics

**Relative Contribution Norm（写强度）** ：

$$\text{Norm Ratio}^{(l)} = \frac{\|\mathbf{m}_t^{(l)}\|_2}{\|\mathbf{h}_t^{(l-1)}\|_2}$$

- **Norm Ratio >> 1** ：contribution完全overwrite之前累积的state（h_t^(l) ≈ m_t^(l)）
- **Norm Ratio << 1** ：incremental correction，不displace prior content
- **Norm Ratio ≲ 1** ：write的magnitude和residual comparable，需要看方向

**Residual I/O Cosine Similarity（方向保真度）** ：

$$\text{IO-CosSim}^{(l)} = \frac{\mathbf{h}_t^{(l)} \cdot \mathbf{h}_t^{(l-1)}}{\|\mathbf{h}_t^{(l)}\|_2 \|\mathbf{h}_t^{(l-1)}\|_2}$$

- **IO-CosSim ≈ 1** ：layer在原方向上做faithful refinement
- **IO-CosSim 显著< 1** ：layer把representation rotate到另一个semantic subspace

这两个metrics配合使用才能区分"constructive reinforcement" vs "disruptive rewriting"——单看Norm Ratio无法区分，因为同样magnitude的write可以是增强也可以是扰动。

### 2.3 Empirical三阶段在Qwen3.5-35B-A3B上的具体数字

| Phase | Layer范围 | Norm Ratio | IO-CosSim | 行为 |
|-------|----------|-----------|-----------|------|
| **I (Guess)** | l ≲ 0.15L | Norm Ratio^(1) ≈ 1.6 | ≈ 0.67 | 第一层overwrite embedding，directional shift剧烈 |
| **II (Refine)** | 0.15L ≲ l ≲ 0.95L | 0.23–0.57（稳定） | 0.91–0.97（高平台） | incremental refinement，semantic trajectory稳定 |
| **III (Perturb)** | l ≳ 0.95L | Norm Ratio^(40) ≈ Phase II的2-3倍 | ≈ 0.69 | 最大的directional deflection，部分rewrite |

**关键insight** ：Phase III的Norm Ratio虽然回升，但仍<1，所以不是"完全overwrite"，而是一个"magnitude comparable但方向misaligned"的write。这种write部分rewrite了Phase II已经refine好的semantic trajectory。

### 2.4 为什么会发生Phase III？——Alignment Tax假说

post-training（SFT + RLHF/DPO）在final layers引入了steering vectors，把representation拉向generic/safe distribution。可以用regularized risk建模：

$$\mathcal{R}^{(l)} = \mathbb{E}[-\log p_t^{(l)}(Y_{\text{logic}}|X)] + \lambda \mathcal{D}_{\text{KL}}(P_{\text{logic}}^{(l)} \| P_{\text{align}})$$

变量：
- **Y_logic** ：从domain-specific reasoning严格推出的optimal target token（比如数学推导的下一步）
- **P_logic^(l)** ：layer l对Y_logic的预测分布
- **P_align** ：generic/safe的alignment-preferred分布（高频function words、punctuation、安全话术）
- **λ** ：steering强度

对l > V*（V*是Phase II→III的oracle boundary），latent state被additive perturbation累积偏移：

$$\mathbf{h}_t^{(l)} \approx \mathbf{h}_t^{(V^*)} + \sum_{k=V^*+1}^{l} \delta_{\text{align}}^{(k)}$$

**关键conditional性质** ：
- **Safety tasks** ：P_logic ≈ P_align，KL ≈ 0，late layers只做formatting refinement，无semantic disruption
- **Complex reasoning** ：P_logic sharply conflicts with P_align，perturbation强行把latent state拖离reasoning subspace，manifests为"entropy oscillation"

这就是他们说的"planning-pragmatics tradeoff"——model在intermediate layers内部form了一个强reasoning prediction，但final layer出于pragmatic alignment把它shift到generic token。

---

## 三、Entropy Valley：为什么这是一个reliable signal

### 3.1 用unembedding probe中间层

$$p_t^{(l)} = \text{Softmax}\left(W_U \cdot \text{RMSNorm}_L(\mathbf{h}_t^{(l)})\right) \tag{3}$$

$$H(p_t^{(l)}) = -\sum_v p_t^{(l)}(v) \log p_t^{(l)}(v)$$

变量：
- **W_U ∈ R^{|V|×d}** ：final layer的unembedding matrix（共享给所有layer的logit-lens probe）
- **RMSNorm_L** ：final layer的normalization（共享应用，避免basis shift问题）
- **p_t^(l)** ：从layer l probe出的token分布
- **H(p_t^(l))** ：Shannon entropy，越低越confident

**Basis shift problem** （Belrose et al., 2023的tuned lens工作指出过）：W_U只对final layer的latent space calibrated，对early layer会有mapping error。但paper论证了三点mitigation：
1. 现代LLM的representation homogeneity让后半段layers的subspace大致对齐
2. 我们只在near-final window里search，mapping error tight bounded
3. 我们看的是relative monotonic gradient（ΔH），uniform projection noise会被自然filter

### 3.2 为什么entropy valley是optimal boundary

在Qwen3.5-35B-A3B上GPQA-Diamond的实验（N=50 prompts，4096 tokens/prompt，共202,935 tokens）：

| Token类别 | 比例 | V*处entropy | L处变化 | 解释 |
|----------|------|-----------|---------|------|
| **Perturbed tokens** | 16.2% | H_V* = 0.52 nats | ΔH = +0.37 nats（上升） | Phase III发生，alignment tax破坏已committed的prediction |
| **Unperturbed tokens** | 83.8% | H_V* = 2.78 nats | ΔH = -2.52 nats（下降） | 正常Phase II→L refinement，继续降到接近0 |

**Brilliant insight** ：Phase III不是fixed architectural region，而是**per-token phenomenon** 。对已经confident的token（低entropy），final layer反而会"扰动"它；对还在uncertain的token（高entropy），final layer继续refine它。

backward scan exploit这种heterogeneity：
- 对perturbed tokens，选V*，bypass Phase III
- 对unperturbed tokens，选L，利用full refinement

### 3.3 Static early exit为什么失败

Figure 3(a)的实验：用Bernoulli probability p随机override到fixed shallow layer (L-k)。p增加时accuracy急剧下降。原因：static truncation忽略了token complexity的variance，对hard tokens提前中断essential computation。

Figure 3(b)：从valley的immediate neighbors (Valley±k) decode都会degrade。证明valley是一个precise optimal boundary，不是smooth region。

---

## 四、Theoretical Formulation：Optimal Stopping Problem

### 4.1 Entropy decomposition

$$\hat{H}(l) = H^*(l) + \epsilon^{(l)} + \eta^{(l)}$$

变量：
- **H*(l)** ：true monotonic entropy，在[V_onset, V*]区间内单调下降（ΔH*(l) ≤ 0）
- **ε^(l)** ：bounded projection noise，|ε^(l)| ≤ ε_max
- **η^(l)** ：per-token alignment perturbation；对l ≤ V*，η^(l) ≈ 0；对conflict tokens，在final layers Δη^(l) > 2ε_max

**V_onset** ：Phase I carry-over dissipate的layer，empirically V_onset ≈ 28（在35B-A3B上），V* = 39。整个Phase II积累ΔH ≈ -10.8 nats across 202,935 tokens。

### 4.2 Theorem 1 (Minimax Optimality)

**Backward scan rule** ：$\hat{V} = \max\{l < L \mid \hat{H}(l-1) \geq \hat{H}(l)\}$

这个rule从L往回扫，找到第一个"entropy不再strictly decreasing"的layer停下来。

**Claim** ：$\hat{V} \in [V_{\text{onset}}, V^*]$ ，即严格filter alignment perturbation同时bound semantic precision loss。

**Proof的关键两步** ：

**Step 1: Adaptive Evasion of Phase III Tax**

对Δη^(l) > 2ε_max > |Δε^(l)|的tokens（即alignment perturbation dominates projection noise）：
- l > V*时，observed gradient $\Delta\hat{H}(l) > 0$（严格正）
- stopping condition永远不在Phase III被trigger
- 因此 $\hat{V} \leq V^*$ 严格保证
- 在V̂处 η^(V̂) → 0，alignment tax被nullify

对alignment correction与refinement synergistic的tokens：
- Δη^(l) ≈ 0
- 不trigger false stop
- 自然停在V̂ ≈ L，用满final layer的refinement

**Step 2: Bounded Optimality in Phase II**

scan进入Phase II时，η^(l) ≈ 0，stopping condition $\hat{H}(\hat{V}-1) \geq \hat{H}(\hat{V})$ 被evaluate。两种case：

**Case A (Strong Integration Signal)** ：如果 $|\Delta H^*(V^*)| > |\Delta\epsilon^{(V^*)}|$，true signal克服projection noise，$\hat{H}(V^*-1) > \hat{H}(V^*)$，algorithm停在 $\hat{V} = V^*$ ——**oracle-optimal** 。

**Case B (Weak Integration / Local Oscillation)** ：如果某layer k ≤ V*处integration signal weakens（ΔH*(k) ≈ 0），projection noise induce micro-oscillation使 $\hat{H}(k-1) < \hat{H}(k)$，algorithm提前停在V̂ = k。

但此时 $\Delta H^*$ 极小，所以 $I(\mathbf{h}^{(k)}; Y) \approx I(\mathbf{h}^{(V^*)}; Y)$，semantic loss：
$$\mathcal{E}_{\text{loss}} = H^*(k) - H^*(V^*)$$
严格bounded by integral of diminished gradient over [k, V*]，asymptotically negligible。

**Interpretation** ：backward search是一个deterministic optimal stopping solution，把alignment tax的unbounded风险nullify掉，把projection noise的penalty bound在一个asymptotically negligible bound。这给了我们一个performance lower-bound近似standard greedy decoding的guarantee。

---

## 五、Confident Decoding Algorithm

### 5.1 核心算法（Algorithm 1）

**Input** ：normed candidates {h̃_t^(ℓ)}_{ℓ=L-M+1}^L，unembedding W_U，scan window K，fallback probability 1-p

**Output** ：logits z_t送给sampler

```
for all ℓ in C (parallel):
    z_t^(ℓ) ← W_U · h̃_t^(ℓ)
    p_t^(ℓ) ← softmax(z_t^(ℓ))
    H_t^(ℓ) ← -<p_t^(ℓ), log p_t^(ℓ)>

ℓ* ← L, H_ref ← H_t^(L), frozen ← false

for ℓ = L-1 downto max(1, L-K):
    if ¬frozen and H_t^(ℓ) < H_ref:
        ℓ* ← ℓ
    else:
        frozen ← true  # first non-improvement freezes choice
    H_ref ← H_t^(ℓ)

with probability 1-p: ℓ* ← L  # stochastic fallback
return z_t^(ℓ*)
```

**Candidate set** ：$\mathcal{C} = \{L-M+1, \ldots, L\}$，size M ∈ [1, L]

**Entropy-trough selection rule** ：

$$\ell_t^* = \min\{\ell \in [L-K+1, L] : H_t^{(\ell)} < H_t^{(\ell+1)} < \cdots < H_t^{(L)}\} \tag{7}$$

变量：
- **M** ：candidate window size（控制worst-case overhead）
- **K** ：per-token scan window，K ≤ M（控制scan cost）
- **p** ：valley-selection probability，p=1时deterministic valley，p=0时退化为standard final-layer decoding

**Frozen机制的关键** ：一旦往回扫一步发现H没有strictly decrease，就frozen当前选择。这是为了找到**first local entropy valley**——即从L往回扫遇到的第一个local minimum。

### 5.2 vLLM实现的工程细节（这部分对我很有启发）

paper的§4.2讲了几条非trivial的工程principle：

**1. Unmodified forward pass** ：Confident Decoding **never truncates** transformer。所有L层都执行，所以KV cache、attention kernels、prefix caching、continuous-batching scheduler完全和standard decoding一致。只改变"哪个layer的logits送给sampler"。

这意味着可以和speculative decoding、multimodal、tensor-parallel infrastructure无缝compose——因为这些infrastructure都assume完整forward pass。

**2. Graph-safe candidate extraction** ：vLLM用torch.compile + CUDA graph replay。如果在compiled region里mutate Python attribute或动态reallocate buffer，graph replay时会有stale state和silent correctness regression。

他们的做法：
- compiled inner model把candidate residual tensors {x_t^(ℓ)}收集到一个Python list，**和final hidden states一起return**
- **compiled region里不做normalization、unembedding、entropy计算、attribute mutation**
- 在eager language-model wrapper里apply final Norm到每个candidate，stack成 [M, S, d] tensor（S是当前CUDA graph replay captured的token count）

**3. Shape-aware buffering under continuous batching** ：logits computing消耗sliced hidden state [B, d]（B是slicing后真正需要sampling的position数，B ≤ S）。

要把sliced positions map back到full-forward candidate states，需要：
- engine的pre-slice token count S
- slicing indices

wrapper维护一个按S索引的buffer，retrieve对应active replay shape的entry，用和final-layer hidden states相同的indices slice。**consume-once protocol** 在logits算完后clear这些per-step indices，防止stale values leak到后续calls（比如prompt-logprob computation）。

**4. Vectorized logits和entropy** ：虽然Algorithm 1是per-token写的，实际实现fully vectorized over both candidate window和active tokens。entropy和trough scan表达为fused tensor operations with per-token frozen mask，back-to-front search collapse到K个element-wise updates。

### 5.3 Computational Cost Analysis

**额外per-step overhead** ：
- 一次batched unembedding of M candidate hidden states：O(MBd|V|)
- entropy evaluation：O(MB|V|)
- K-step trough scan：O(KB)

其中B << prefill context length，所以总cost是 **O(MBd|V|)** ，generally不超过regular per-step cost：
$$O(12LSd^2 + 2LSdT + Bd|V|)$$
（假设FFN dim d_ff = 3d，T是当前KV cache长度）

**Critical empirical数据** （Table 5，Qwen3.5-35B-A3B，L=40, d=2560, |V|=151,936, K=10）：

| Component | FLOPs | Relative Overhead |
|-----------|-------|-------------------|
| Full forward pass baseline | 5,212M | 100% |
| Single unembedding projection | 389M | +7.46% |
| **Worst-case** boundary scan (K=10) | 3,890M | +74.64% |
| **Empirical** mean extra projections (0.116/token) | 45M | **+0.87%** |
| KV-cache memory cost | 0MB | **+0.00%** |
| End-to-end wall-clock latency (vLLM) | — | **<2%** |

**Why empirical so cheap** ：88.5%的tokens在final layer就已经满足monotonic condition，触发**zero额外projections** 。只有11.5%的tokens启动backward scan，mean iterations = 0.116/token。这是一种**surgical intervention** ，只在alignment tax真正威胁reasoning path时介入。

---

## 六、Main Results的深度解读

### 6.1 Table 1的关键patterns

跨6个backbones（Qwen3.5-27B/35B-A3B/122B-A10B，Gemma-4-31B，gpt-oss-20b/120B）：

**Pattern 1: Universal Efficacy Across Architectures**

每个backbone都获得positive average gain。MoE（Qwen-35B/122B）和dense（Gemma-4）都work。证明alignment tax是post-training的intrinsic artifact，与underlying architecture无关。

**Pattern 2: Massive Surges in Complex Reasoning**

最striking的数字：
- Qwen3.5-27B on LCB-v6: **+9.4% absolute** （63.9 → 73.3）
- Qwen3.5-35B-A3B on GPQA-D: **+6.5%** （76.3 → 82.8）
- Gemma-4-31B on GPQA-D: +4.0%
- gpt-oss-120B on GPQA-D: +4.5%

LCB-v6的+9.4%尤其impressive——code generation对syntactic consistency要求高，alignment tax在low-frequency logic chain上破坏最大。

**Pattern 3: Pristine Stability in Creativity and Safety**

- WritingBench：marginal gains +0.1%到+0.5%
- LongBench-v2：minimal change（contextual saturation effect）
- Air-Bench：actually improves（+2.0% to +5.0%）

Air-Bench的提升尤其有意思——bypass perturbation phase反而**reduces overly conservative hallucinatory refusals** ，同时enhance logical fidelity和rigorous compliance。这validate了"alignment tax vs safety guardrail"的二分法。

### 6.2 Instruct vs Base（Table 2）——Causal Isolation of Alignment Tax

**Setup** ：比较Qwen3.5-35B-A3B-Base vs Instruct版本。

**理论预测** ：Base model只有next-token prediction objective，没有KL penalty $\mathcal{D}_{\text{KL}}(P_{\text{logic}} \| P_{\text{align}})$ ，所以final layer entropy应该relatively stable。Instruct model在complex logic上应该suffer severe Phase III perturbation。

**Empirical结果** ：

| Metric | Base | Instruct |
|--------|------|----------|
| Average gain from Confident Decoding | +1.1% | **+2.6%** |
| GPQA-D gain | +1.9 | **+6.5** |
| HLE performance reversal | Base 8.0% > Instruct 7.1%（standard） | Confident把Instruct拉到9.5%，surpass Base |
| Token-level valley rate | 10.4% | **12.8%** |
| Token substitution rate | 2.36% | **2.60%** |
| Mean entropy gap at substitution | 3.34×10⁻² | **3.48×10⁻²** |

**关键insight** ：HLE上Instruct model在standard greedy decoding下 **perform worse than Base** （7.1% vs 8.0%）！这说明post-training在long-tail complex multi-hop reasoning上 **actively penalize** reasoning，用generic safe priors override fragile logic chains。Confident Decoding把这个Instruct modelrescue到9.5%，surpass Base model。

Token-level statistics也一致放大——Instruct的valley rate、substitution rate、entropy magnitude都proportionally higher，causally prove alignment tax是learned byproduct of post-training。

### 6.3 Task Complexity Scaling（Tables 3, 4）——最让我兴奋的result

按baseline Pass@1 rate stratify成Level 1-4：

**gpt-oss-20b on MATH/Omni-MATH** ：

| Level | MATH Δ | Omni-MATH Δ |
|-------|--------|------------|
| Level 1 (easiest, baseline ~98%) | -0.4% | -4.3% |
| Level 2 | +12.7% | +3.3% |
| Level 3 | +26.6% | +4.5% |
| **Level 4 (hardest, baseline 1-2%)** | **+22.5%** | **+22.4%** |

**Qwen3.5-35B-A3B** ：

| Level | MATH Δ | Omni-MATH Δ |
|-------|--------|------------|
| Level 1 | -0.1% | -1.4% |
| Level 2 | +21.9% | +2.1% |
| Level 3 | +17.0% | +3.6% |
| **Level 4** | **+9.2%** | **+7.2%** |

**Profound scaling law** ：performance delta随task complexity **substantially grows** 。

为什么？hard reasoning tasks需要model inhabit一个highly specialized、low-frequency semantic subspace。logical chain越长，越偏离alignment distribution P_align，Phase III perturbation越actively destroy这些fragile links。

最extreme case：gpt-oss-20b在Omni-MATH Level 4上standard只有1.1%，Confident Decoding拉到23.5%——**+22.4 absolute** 。这是我近期看到的最impressive的test-time intervention result之一。

为什么Level 1会略微退化（-0.4%到-4.3%）？简单token的final-layer representation主要服务于superficial syntax和output formatting的refinement，early truncation会slightly under-optimize。这和theory完全consistent——simple tokens的alignment perturbation minimal，Phase III其实在做useful formatting work。

---

## 七、Token-Level Substitution Analysis（Table 8）

在Qwen3.5-35B-A3B GPQA-Diamond上，看Base vs Instruct最frequently substituted的tokens：

**Base model top substitutions** ：
- Standard选"the"（4.4%），Confident选"<|box_end|>"
- Standard选"，"（3.1%），Confident选"careful"
- Standard选"is"（2.9%），Confident选"perpendicular"
- Standard选"a"（1.8%），Confident选"indeed"
- Standard选"So"（1.8%），Confident选"actually"
- Standard选"that"（1.3%），Confident选"mutating"
- Standard选"with"（1.3%），Confident选"neither"

**Instruct model top substitutions** ：
- Standard选"$"（7.7%），Confident选"carb"
- Standard选"\n"（4.2%），Confident选"carb"
- Standard选"the"（4.2%），Confident选"ring"
- Standard选"$"（3.5%），Confident选"，"
- Standard选"is"（3.5%），Confident选"{}"
- Standard选"$"（3.0%），Confident选"."
- Standard选"$"（1.7%），Confident选"Sometimes"
- Standard选"$)"（1.5%），Confident选"methyl"

**Category breakdown** ：

| Category | Base Standard | Base Confident | Instruct Standard | Instruct Confident |
|----------|---------------|----------------|-------------------|--------------------|
| Content | 29% | **77%** | 26% | **60%** |
| Function | 39% | 10% | 27% | 6% |
| Punctuation | 22% | 6% | 43% | 26% |
| Other | 10% | 7% | 4% | 8% |

**Striking pattern** ：Standard decoding倾向于function words和punctuation（"the", "is", "$", "，", "."），Confident Decoding倾向于domain-specific terminology（"perpendicular", "carb", "ring", "methyl", "phenotype", "fusion", "Cartesian", "radius"）。

这exactly validate了Figure 1的word cloud——alignment tax在final layer把prediction拉向generic high-frequency function words，Confident Decoding在entropy valley处recover domain-specific semantically precise terminology。

---

## 八、与Related Work的关系

### 8.1 Layer-wise Dynamics传统视角

- **Belrose et al. (2023) tuned lens** ：发现confident predictions在final layer之前几层就crystallize
- **Geva et al. (2022, 2023)** ：FFN neurons通过subject-enrichment → relation-propagation → attribute-extraction pipeline progressive promote concepts
- **Meng et al. (2022) ROME** ：factual knowledge localized在mid-layer MLPs narrow window
- **Skean et al. (2025)** ：intermediate layers harbor stronger representations than final layer
- **Gromov et al. (2025)** ：up to half layers can be pruned with minimal degradation
- **Liu et al. (2026) layer-order inversion** ：multi-hop reasoning entities在superficial facts之前crystallize
- **Wang and Zhou (2024) CoT without prompting** ：从intermediate layers decode能uncover latent CoT paths被final-layer output suppress

这些工作都support Guess-Refine-Perturbation formalization。

### 8.2 Contrastive Decoding方法的局限

**DoLa (Chuang et al., 2023)** ：contrast final-layer logits with early layers，用JS divergence选maximally divergent的premature layer。

**SLED (Zhang et al., 2024)** ：用gradient-style corrections based on logit evolution across layers。

**关键limitation** ：都assume **representational homogeneity** across contrasted layers。在modern hybrid MoE架构上（DeltaNet + full-attention interleaving + sparse expert routing），interleaving引入discontinuous representation geometry shifts，break这个assumption。

Table 9的comparison（Qwen3.5-35B-A3B）：

| Method | GPQA-D | HLE | LCB-v6 |
|--------|--------|-----|--------|
| Last Layer | 76.3 | 7.1 | 70.0 |
| DoLa | 77.3 | 7.8 | 70.9 |
| SLED | 78.8 | 7.4 | 71.7 |
| **Confident** | **82.8** | **9.5** | **75.1** |

Confident Decoding之所以更robust：不做cross-layer subtraction（会amplify structured noise），而是independently evaluate每个candidate layer的predictive entropy，选minimum uncertainty的。

### 8.3 Test-Time Compute和Optimal Stopping

- **Universal Transformers (Dehghani et al., 2019)** ：adaptive per-token halting
- **CALM (Schuster et al., 2022)** ：token-wise early exits
- **LayerSkip (Elhoushi et al., 2024)** ：progressive layer-dropout + self-speculative decoding
- **DEER (Yang et al., 2025)** ：reasoning models的dynamic early exit

paper的一个重要reframe：纯latency-driven early exit在现代LLMs上diminishing returns（Wei et al., 2026），因为simple threshold难区分valid convergence vs shallow biases。Confident Decoding把structural truncation从**efficiency-driven** 转向 **efficacy-driven** 的vertical TTC scaling paradigm——optimizing where to stop inside network和scaling how long to think outside同样vital。

---

## 九、Limitations和Degradation Analysis

### 9.1 Qwen3.5-9B的mixed picture（Table 10）

Qwen3.5-9B在GPQA-D（64.6→62.1）和Omni-MATH（49.1→47.1）regress，但在LCB-v6（41.1→47.7）和Air-Bench（53.0→56.0）clearly improve。

**Factor I: Hybrid architecture compresses refinement corridor**

Qwen3.5-9B的L=32（24 DeltaNet + 8 full-attention），相邻layer属于fundamentally different computational paradigms。late refinement corridor被压缩——即使single-layer rollback也可能cross decision boundary（从final full-attention consolidation layer跳到pre-final DeltaNet state）。

**Probe mismatch** ：decompose projection noise：
$$\hat{H}(l) = H^*(l) + \epsilon_{\text{probe}}^{(l)} + \epsilon_{\text{type}}^{(l)} + \eta^{(l)} \tag{8}$$

- **ε_probe^(l)** ：smooth baseline probe error
- **ε_type^(l)** ：layer-type-dependent component，在DeltaNet↔full-attention transition处discontinuous jumps

当structured noise dominate true entropy gradient ΔH*(l)，observed valley偏离true semantic optimum，algorithm commit tokens to pre-convergent representations。

**Factor II: Model depth widens refinement corridor**

deeper networks给late refinement zone分配更多consecutive同类型layers，wider corridor让 $|\Delta H^*(l)| \gg |\Delta \epsilon^{(l)}|$ 在更大区间成立。

Qwen3.5-9B只有8个full-attention layers spread across整个stack，final refinement zone homogeneous layers极少。Qwen3.5-27B（更deep）在GPQA-D上无degradation，LCB-v6上+10.1%。

### 9.2 为什么MoE architectures appear robust

Qwen3.5-A series（MoE + hybrid attention）获得significant gains。两个可能的factor：
1. **Sparse expert routing** concentrates task-relevant updates in specialized sub-networks，amplify true refinement signal $|\Delta H^*(l)|$ in late corridor
2. 每个token只activate small subset of experts，effective representation trajectory across adjacent layers **smoother** than dense hybrid backbone，reducing ε_type^(l)

但paper坦白承认depth和MoE routing在当前setup下confounded，无法fully disentangle。

### 9.3 Fundamental constraint

Confident Decoding的effectiveness受限于 **W_U与intermediate residual states的结构对齐** 。虽然theory bound了projection noise，shallow layers的representations仍可能suffer vocabulary mismatch。而且方法只是 **mitigate symptoms** during decoding，没有resolve root cause during training。

---

## 十、Future Directions的Implications

paper结尾提的几个方向让我很有共鸣：

1. **Training paradigms that decouple reasoning vs pragmatics** ：比如apply alignment penalties exclusively to designated routing heads，而不是core residual stream。这暗示了一种 **dual-head architecture** ——一个reasoning head在intermediate layers输出，一个alignment head在final layers输出，让alignment tax集中到不破坏reasoning的位置。

2. **Multimodal foundation models** ：Guess-Refine-Perturbation dynamics在multimodal setting下如何persistence？cross-modal alignment tax会是什么形态？

3. **Layer-wise entropy metrics作为RL reward** ：如果用entropy valley作为geometrically precise reward signal，能否在训练阶段就encourage model在intermediate layers commit到reasoning-optimal distribution，让final layers不需要perturb？

这第三点我认为特别有潜力——当前RLHF/DPO的reward都是从final output计算的，这indirectly incentivize final layers去steer towards alignment-preferred tokens。如果能设计一个 **intermediate-layer reward** ，让RL直接在V*附近evaluate reasoning quality，可能从root cause上solve alignment tax。

---

## 十一、与我自己工作的联系

这篇paper让我想到nanoGPT和micrograd教学中的一些直觉：

1. **Residual stream的iterative refinement** ：我讲Transformer时一直强调residual stream像一条"information highway"，每层在上面write updates。这篇paper给这个intuition加了一个重要caveat—— **writes的方向不一定和已有representation一致** ，尤其在Phase III的final layers，writes会rotate direction。IO-CosSim这个metric很brilliant，把"方向保真度"和"magnitude强度"解耦，正好量化这种rotation。

2. **Logit lens的实用性** ：Belrose et al.的tuned lens和nostalgebraist的original logit lens observation一直是个实用工具，但这篇paper把它从 **诊断工具** 升级为 **decision rule** ——用logit-lens entropy来决定从哪个layer采样。这是一个很自然的extension，但之前没人这样formalize。

3. **Alignment tax的可视化** ：我之前看Anthropic的Constitutional AI和InstructGPT paper时一直wonder alignment具体在representation层面做什么。这篇paper的regularized risk formulation $\mathcal{R}^{(l)} = \mathbb{E}[-\log p(Y_{\text{logic}})] + \lambda \mathcal{D}_{\text{KL}}(P_{\text{logic}} \| P_{\text{align}})$ 给了一个clean mathematical model——alignment就是KL pressure，让final layers把representation拖向generic distribution。

4. **Test-Time Compute Scaling的vertical维度** ：Snell et al. (2025)的test-time compute scaling主要讨论"horizontal"——让model think longer生成更多tokens。这篇paper引入了"vertical"维度——在 **forward pass内部** 的layer维度上做dynamic compute allocation。两个维度应该可以compose：用CoT生成多个candidate thoughts，每个thought内部用Confident Decoding选最优layer。

---

## 十二、潜在Critical Issues

虽然paper很compelling，但我也想到几个值得push back的地方：

1. **Per-token heterogeneity的causality** ：paper说16.2%的tokens有Phase III perturbation，但这部分tokens是否就是"决定reasoning成败的关键tokens"？如果是reasoning path上的critical tokens（比如"=" vs "≠"），那bypass perturbation收益巨大；如果是cosmetic tokens，那bypass意义不大。paper没有quantify substituted tokens的"reasoning criticality"。

2. **Entropy valley的interpretability** ：低entropy不一定等于"correct prediction"。一个model可以在intermediate layer非常confident地predict一个错误token。paper的Theorem 1假设H*(l)是"true monotonic entropy"，但实际上H*(l)只是"model的内部entropy"，不一定是ground-truth conditional entropy。不过他们的empirical evidence（Instruct > Base gain）causally support了H*(l)确实反映reasoning quality。

3. **W_U probe的universality** ：虽然paper论证了representation homogeneity让near-final layers的probe误差bounded，但对于 **非常long-tail domain** （比如Omni-MATH的Olympiad-level math），intermediate representations是否还落在W_U能calibrated的subspace？Table 3显示hard tasks gain最大，但这也可能是 **easy tasks gain小** 导致的artifact。

4. **Fallback probability p的设置** ：paper默认p=1.0 deterministic。但理论上，对 **不同task type** 应该有不同的optimal p。safety tasks上P_logic ≈ P_align，可以p≈0；reasoning tasks上P_logic sharply conflicts，应该p≈1。一个 **adaptive p** based on detected task type可能更好。Table 7的ablation显示p=0.8在GPQA-D上只降到81.8%（vs p=1.0的82.8%），说明partial mixing其实损失不大，adaptive p可能给safety task提供更好的protection。

5. **Confidence metric的alternative** ：Shannon entropy是其中一个选择，也可以用 **max probability** 、 **top-k mass** 、 **KL divergence to uniform** 等。paper没有系统比较alternative confidence metrics。也许 **margin between top-1和top-2 probability** 对"committed prediction"更sensitive。

---

## 十三、Open Questions for Future Research

1. **Layer-specific alignment** ：能不能在post-training时，只让 **specific layers** （比如last 2-3层）承担alignment pressure，让intermediate layers保持pure reasoning objective？类似 **LoRA on final layers only** 的做法。

2. **Multi-modal entropy valley** ：在vision-language models里，visual tokens和text tokens的entropy trajectory不同。visual tokens可能在更early layers就commit，text tokens遵循Guess-Refine-Perturb。如何handle这种heterogeneity？

3. **Entropy valley和scratchpad的interaction** ：reasoning models（o1-like）有显式scratchpad。scratchpad tokens的entropy valley profile和final answer tokens不同吗？Confident Decoding在reasoning tokens和answer tokens上应该用不同策略吗？

4. **训练阶段的entropy regularization** ：如果在pretraining阶段就加一个regularizer，鼓励entropy在intermediate layers达到minimum然后monotonic保持到final layer（discourage Phase III的entropy rise），能否从root cause上solve alignment tax？

5. **Theoretical extension to non-Transformer architectures** ：Mamba、Jamba等SSM-based架构的layer dynamics不同。Guess-Refine-Perturb是否还成立？entropy valley在哪里？

---

## 十四、Final Verdict

这是一篇 **execution非常solid** 的paper——理论（optimal stopping formulation）、empirical（跨6个backbones，7个benchmarks）、工程（vLLM集成，<2% latency overhead）三方面都做得很完整。最重要的conceptual贡献是把"deeper is better"这个implicit assumption破除掉，并用 **entropy valley** 这个简单但elegant的signal来operationalize"什么时候该停"。

对practitioner的take-away：在aligned LLM部署reasoning任务时，Confident Decoding是一个 **zero-cost upgrade** （training-free，<2% latency，zero memory overhead），尤其在hard reasoning task上能unlock latent reasoning fidelity。对researcher的take-away：alignment tax是一个structurally localized的phenomenon，final layers是alignment和reasoning conflict的battleground，未来研究应该explore decoupling这两者的training paradigms。

Reference:
- Paper: https://arxiv.org/abs/2507.14927 (推测，paper未提供arXiv ID但github在https://github.com/QwenLM/Confident-Decoding)
- Project page: https://github.com/QwenLM/Confident-Decoding
- DoLa (Chuang et al., 2023): https://arxiv.org/abs/2309.06634
- Tuned Lens (Belrose et al., 2023): https://arxiv.org/abs/2303.08112
- LayerSkip (Elhoushi et al., 2024): https://arxiv.org/abs/2404.16710
- Representation Engineering (Zou et al., 2023): https://arxiv.org/abs/2310.01405
- ROME (Meng et al., 2022): https://arxiv.org/abs/2202.05262
- Test-Time Compute Scaling (Snell et al., 2025): https://arxiv.org/abs/2408.03314
- Information Bottleneck (Tishby & Zaslavsky, 2015): https://arxiv.org/abs/1503.02406
- CoT without prompting (Wang & Zhou, 2024): https://arxiv.org/abs/2402.10200
- vLLM (Kwon et al., 2023): https://arxiv.org/abs/2309.06180
- Skean et al. (2025) Layer by Layer: https://arxiv.org/abs/2502.13920
- ShortGPT (Men et al., 2025): https://arxiv.org/abs/2403.03853
- The Unreasonable Ineffectiveness of Deeper Layers (Gromov et al., 2025): https://arxiv.org/abs/2403.17887
- SLED (Zhang et al., 2024): https://arxiv.org/abs/2411.15140
- ITI (Li et al., 2023): https://arxiv.org/abs/2306.03341
- CAD (Shi et al., 2024): https://arxiv.org/abs/2311.10909
