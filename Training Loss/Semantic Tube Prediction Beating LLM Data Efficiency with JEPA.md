---
source_pdf: Semantic Tube Prediction Beating LLM Data Efficiency with JEPA.pdf
paper_sha256: bed646a5d7ab80c391a83d75535215bf85f9396e35506a24425bc3f126e773bc
processed_at: '2026-08-12T05:12:50-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Semantic Tube Prediction 用人话讲

Andrej，我把这篇paper的核心idea用最直白的方式重新过一遍，同时保留关键的技术细节和公式。

---

## 1. 一句话概括

当前训练LLM的方式——next-token prediction (NTP)——只关心"下一个token猜对没"，不关心hidden state在representation space里**怎么走**。这篇paper说：如果你额外加一个约束，强制hidden state的轨迹"走直线"，那么模型可以用**1/16的数据**达到同样的效果。

---

## 2. 最核心的intuition：Voronoi cell的问题

这是整个paper最关键的picture。

NTP在干什么？它让模型预测下一个token。token是discrete的，hidden state是continuous的。所以训练过程本质上是在说：**$h_t$ 落在正确token对应的Voronoi cell里就行，cell内的具体位置我不管**。

https://en.wikipedia.org/wiki/Voronoi_diagram

这个"我不管"就是问题所在。

想象一下，你在地图上导航从A到B。NTP只告诉你"你现在在正确的城市"，但不告诉你"你在城市的哪个位置"。结果你可能drift到另一条路上——虽然还在同一个城市，但下一段旅程完全不同。

Figure 2就是这个意思：两个相似prefix的trajectory都穿过"researcher"这个token的Voronoi cell。NTP都给了pass，但实际上 $h_t$ 可能已经跳到另一条geodesic上，下一步就会生成完全不同的内容。这就是为什么LLM在inference时会mode collapse、会胡说八道。

---

## 3. 物理类比：为什么"走直线"？

作者借用经典力学的**Principle of Least Action**。

https://en.wikipedia.org/wiki/Principle_of_least_action

在物理里，一个系统从state A到state B，走的path会minimize action $S = \int_a^b L \, dt$，其中 $L = T - V$（$T$ 是kinetic energy，$V$ 是potential energy）。这个path就是**geodesic**——manifold上的"最短路径"。

https://en.wikipedia.org/wiki/Geodesic

Riemannian geometry告诉我们：smooth manifold上的geodesics **almost everywhere locally linear**。就是说，如果你zoom in到一个足够小的local neighborhood，geodesic看起来就是一条直线。

作者的**Geodesic Hypothesis**：LLM训练产生的semantic manifold是smooth的，所以error-free token sequence trajectory应该是geodesic，所以应该local linear。

---

## 4. 公式核心：Semantic Tube Prediction loss

$$\mathcal{L}_{STP} = 1 - \cos(h_t - h_r, h_r - h_s)$$

变量解释：
- $s < r < t$: 三个randomly sampled的token位置
- $h_s, h_r, h_t$: 这三个位置在**最后一层**的hidden states
- $h_r - h_s$: 从 $s$ 到 $r$ 的"semantic evolution vector"
- $h_t - h_r$: 从 $r$ 到 $t$ 的"semantic evolution vector"
- $\cos(\cdot, \cdot)$: cosine similarity

这个loss在说什么？**如果trajectory是直线，那么 $h_r - h_s$ 和 $h_t - h_r$ 应该平行**（方向一致）。cosine similarity = 1意味着完全平行，loss = 0。

完整training loss：
$$\mathcal{L} = \mathcal{L}_{NTP} + \lambda \cdot \mathcal{L}_{STP}$$

$\lambda$ 是hyperparameter，实验里 $\lambda \in [0.01, 0.08]$ 最好。为什么这么小？因为geodesic不是**严格**线性，有slight curvature，$\lambda \ll 1$ 让constraint是soft的，容忍小偏差。

---

## 5. 为什么这个loss有道理？

### 5.1 Noise和Signal的分解

Figure 1a的核心picture：把 $h_r - h_s$ 分解成两个component：
- **Parallel component** $(h_r - h_s)_{\parallel h_t - h_s}$: 平行于 $h_t - h_s$ 的部分——这是**signal**
- **Perpendicular component** $(h_r - h_s)_{\perp h_t - h_s}$: 垂直于 $h_t - h_s$ 的部分——这是**noise**

STP loss最小化perpendicular component，保留parallel component。所以它**提升Signal-to-Noise Ratio (SNR)**。

### 5.2 Context-aware的关键

$h_t - h_s$ **不是** isolated subsequence $x_{[s,t]}$ 的hidden state。它是整个context $x_{\leq s}$ 加上 $x_{[s,t]}$ 产生的semantic shift。

举Figure 9的例子：
- Prefix "The capital of" + "France" → trajectory推向 $\vec{v}_{Paris}$
- Prefix "The language of" + "France" → trajectory推向 $\vec{v}_{French}$

同一个token "France"，在不同context下，semantic shift完全不同。所以 $h_t - h_s$ 是context-aware的，比isolated hidden state informative得多。

### 5.3 为什么不用learned predictor?

原始JEPA需要learned predictor：predict $\phi(y)$ from $\phi(x)$ via predictor $p$。

https://arxiv.org/abs/2301.08243

但STP用**identity predictor**——直接要求 $h_t - h_r \approx h_r - h_s$。

为什么identity work？因为如果trajectory是locally linear的，那么predictor应该就是identity（线性的预测就是"继续往同一方向走"）。实验ablation（Figure 8的"Pred" variant）证明：learned projector反而更差，validate了identity hypothesis。

---

## 6. 理论保证：从loss到tube

### Definition 3.1 (Local Linearity)

Trajectory $h^*$ locally linear if $\exists \tau, \exists \varepsilon$，对任意 $s < r < t$ with $|t-s| \leq \tau$：

$$\|(h_r^* - h_s^*)_{\perp h_t^* - h_s^*}\|_2 \leq \varepsilon \tag{4}$$

变量：
- $\tau$: local window size
- $\varepsilon$: tolerance
- $\perp$: 垂直component

### Theorem 3.3 (Semantic Tube)

If $h^*$ locally linear and $\mathcal{L}_{STP} \to 0$ for all $r$ with $0 \leq s < r < t \leq \tau$:

$$\|h_r - h^*\|_2 \lesssim \varepsilon$$

含义：$h_r$ 被约束在以 $h^*$ 为中心、半径 $O(\varepsilon)$ 的**tube**内。这就是"Semantic Tube"名字的由来。

### Proof sketch (Appendix D)

$\mathcal{L}_{STP} = 1 - \cos\theta' \leq \epsilon$，small $\epsilon$ 下 $\cos\theta' \approx 1 - \theta'^2/2$，所以 $\theta' \lesssim \sqrt{2\epsilon}$。perpendicular component $\leq \|h_r - h_s\|_2 \sin\theta' \approx \|h_r - h_s\|_2 \cdot \theta' \lesssim \sqrt{2\epsilon}\|h_r - h_s\|_2$。

---

## 7. 为什么NTP alone不够？

### 7.1 Training ODE

训练时，converged network满足：
$$x_{t+1} = \mathring{u} \circ \mathring{f}(x_{\leq t})$$

其中 $\mathring{f}, \mathring{u}$ 是converged后的network functions。

可以把sequence dynamics写成ODE形式：
$$dx_{\leq t} = \mathring{u} \circ \mathring{f}(x_{\leq t}) \, dt$$

### 7.2 Picard-Lindelöf Theorem的含义

https://en.wikipedia.org/wiki/Picard%E2%80%93Lindel%C3%B6f_theorem

如果 $\mathring{u} \circ \mathring{f}$ 连续且Lipschitz-continuous，ODE有唯一解。这意味着**不同initial conditions的trajectories不能intersect**。

所以如果trajectory是error-free的，不同prompts的generation永远不会collapse到一起——diversity理论上保证了。

### 7.3 但NTP让trajectory不是error-free

回到Voronoi cell的argument：NTP只保证 $h_t$ 在对的cell，不保证 $h_t$ 在对的geodesic。所以 $h_t = h_t^* + \epsilon_t$，其中 $\epsilon_t \neq 0$。

### 7.4 Inference时的Brownian motion

Training时teacher forcing让 $\epsilon_t$ 不传播。但inference时 $h_{t+1}$ 依赖 $h_t$，所以 $\epsilon_t$ 累积。

Infinite-width limit（Yang & Littwin 2021, https://arxiv.org/abs/2010.04696）下，$\epsilon_t$ 是i.i.d. Gaussian，累积成Brownian motion：

$$dx_{\leq t} = \mathring{u} \circ \mathcal{f}(x_{\leq t}) \, dt + \sigma_t \, dW_t$$

$dW_t$ 是Brownian motion。

### 7.5 Inference Cone

由Donsker's theorem，$\|h_t - h_t^*\|_2 \propto \sigma\sqrt{t}$。

Inference时trajectory发散成一个cone，半径 $\propto \sigma_t \sqrt{t}$。$\sigma_t$ 越大，cone越宽，越容易collide另一条trajectory——mode collapse。

STP降低 $\sigma_t$，收窄cone，减少collision。这是diversity preservation的mechanism。

---

## 8. Information Theory: SNR → Data Efficiency → Accuracy

这是Section H，paper的theoretical heart。

### 8.1 Data Efficiency Lemma (H.1)

$$H(Y | X^m) \geq H(Y) - m \cdot I(Y; X) \tag{5}$$

变量：
- $Y$: target token
- $X^m = \{X_i\}_{i=1}^m$: $m$ 个conditional i.i.d. hidden states
- $H(Y | X^m)$: conditional entropy，≈ cross-entropy loss
- $I(Y; X)$: mutual information

Interpretation：要让training loss $\leq \epsilon$，需要 $m \geq \frac{H(Y) - \epsilon}{I(Y; X)}$。数据量 $m$ 和mutual information成反比。

### 8.2 Gaussian Channel Approximation

Infinite-width limit下，hidden state $X = Z + N$，$Z$ 是signal，$N \sim \mathcal{N}(0, \sigma^2 I)$ 是noise。

Shannon's channel capacity (https://en.wikipedia.org/wiki/Channel_capacity):
$$I(X; Y) = \frac{1}{2} \log(1 + \text{SNR})$$

其中 $\text{SNR} = \frac{\mathbb{E}[\|Z\|^2]}{\mathbb{E}[\|N\|^2]}$。

### 8.3 Corollary H.2

$$m \geq \frac{H(Y) - \epsilon}{\frac{1}{2}\log(1 + \text{SNR})} \tag{6}$$

**关键结论**：所需数据量 $m$ 和 $\log(1 + \text{SNR})$ 成反比。STP提升SNR → 严格降低 $m$。

### 8.4 Corollary H.3 (via Fano's inequality)

https://en.wikipedia.org/wiki/Fano%27s_inequality

$$P_e \gtrsim \frac{H(Y) - m \cdot \frac{1}{2}\log(1 + \text{SNR})}{\log|\mathcal{V}|} \tag{8}$$

SNR提升 → error probability $P_e$ 下降 → accuracy提升。

---

## 9. 实验结果

### 9.1 Loss Landscape (Figure 4)

Llama-3.2-1B on NL-RX-SYNTH (https://arxiv.org/abs/1608.06115)：

**Figure 4a**: Regular fine-tuning下，$\mathcal{L}_{NTP}$ plateau时 $\mathcal{L}_{STP}$ 也停止下降。STP fine-tuning下，$\mathcal{L}_{NTP}$ plateau时 $\mathcal{L}_{STP}$ **继续下降**。

这说明：NTP自动minimize $\mathcal{L}_{STP}$ 吗？不。需要explicit auxiliary loss。

**Figure 4b**: $\lambda$ 从0增到0.08，$\mathcal{L}_{STP}$ 从1.4降到0.6。$\mathcal{L}_{NTP}$ 保持stable。

- $\lambda = 0$: $\mathcal{L}_{STP} \approx 1.4$（erratic Brownian motion）
- $\lambda = 0.08$: $\mathcal{L}_{STP} \approx 0.6$（smooth path）
- Optimal accuracy在 $\lambda = 0.02$

### 9.2 Accuracy across settings (Figure 5)

**Datasets**: NL-RX-SYNTH, NL-RX-TURK, GSM8K (https://arxiv.org/abs/2110.14168), Spider (https://arxiv.org/abs/1809.08887), NQ-Open (https://aclanthology.org/P19-1612/), HellaSwag (https://arxiv.org/abs/1905.07958)。6个dataset全win。

**Model families**: Llama-3.2-1B, Gemma-2-2B (https://arxiv.org/abs/2408.00118), OpenELM-1.1B (https://arxiv.org/abs/2404.14619), OLMo-2 (https://arxiv.org/abs/2501.00656), Qwen3-1.7B (https://arxiv.org/abs/2505.09388), DeepSeek-R1-Distill-Qwen-1.5B (https://arxiv.org/abs/2501.12948)。6个family全win。

**Model sizes**: Llama-3 1B, 3B, 8B。全win。

### 9.3 Data Efficiency (Figure 1b, Figure 12)

这是paper最striking的结果。

Llama-3.2-1B on NL-RX-SYNTH，subset fractions $\{1/2, 1/4, 1/8, 1/16, 1/32\}$，按比例scale epochs (1/n data → n× epochs):

| Fraction | Regular FT | Semantic Tube |
|----------|------------|--------------|
| 1 (full) | baseline | matches baseline |
| 1/2 | significant drop | negligible drop |
| 1/16 | catastrophic | **matches full-data regular FT** |
| 1/32 | — | starts degrading |

**核心claim**: 用1/16数据匹配full-data baseline，直接violate Chinchilla scaling law的data term。

### 9.4 Diversity Preservation (Table 1)

NL-RX-SYNTH里regex可以以 `.*` 或 `.*.*` 结尾，功能等价但 `.*` 在训练集35×更频繁：

| Suffix | Semantic Tube | Regular | LLM-JEPA |
|--------|---------------|---------|----------|
| `.*` | 88.5% | 29.9% | 68.9% |
| `.*.*` | 68.0% | 28.0% | 32.0% |

Regular FT两者都学不好。LLM-JEPA collapse到majority (`.*` 68.9%)，minority (`.*.*` 32.0%) 差。**STP两者都学好**——88.5%和68.0%。

这说明STP preserve diversity，不倾向于majority pattern。为什么？因为STP收窄inference cone，减少trajectory collision，不同pattern的geodesic保持separated。

### 9.5 Polymorphism SVD (Figure 6)

计算 $\text{Enc}(\text{Text}) - \text{Enc}(\text{Code})$ 的SVD谱：
- Without normalization: STP像regular FT（tolerate raw complexity）
- With normalization: STP像LLM-JEPA（enforce directional structure）

Interpretation: STP在**direction**上enforce structure（normalized vectors aligned），在**magnitude**上tolerate complexity（unnormalized vectors保留variation）。这种"polymorphism"让STP既有regularization效果又保住flexibility。

### 9.6 Ablation (Figure 8, Tables 4-5)

所有variation都比vanilla STP差：
- **Zero** (fix $s=0$): worse
- **Pred** (learned projector $P$): worse in all configs
- **Inst** (add system prompt): worse
- **Two Views** (LLM-JEPA style): worse
- **Mask** (BERT style): worse
- **Curvature** (minimize angle $|\theta_i|$): worse

p-values（paired t-test, 5 seeds）：STP vs Two View: 4.76e-3; vs Mask: 1.77e-3; vs Curvature: 3.04e-5。全部significant。

---

## 10. 和JEPA的关系

### 10.1 JEPA background

JEPA (Joint-Embedding Predictive Architecture, LeCun 2022, https://openreview.net/pdf?id=BZ5a1r-kVsf): 在representation space预测一个view基于另一个view。

I-JEPA (Assran et al. 2023, https://arxiv.org/abs/2301.08243): image domain的JEPA。
Data2vec (Baevski et al. 2022, https://arxiv.org/abs/2204.02605): cross-modal JEPA。

### 10.2 LLM-JEPA的前作

同一组人的LLM-JEPA (Huang et al. 2025, https://arxiv.org/abs/2509.14252) 把JEPA推到LLM，但有三个practical limitations:
1. 需要manual two-view scaffolding (query + answer)
2. 需要额外forward pass
3. 需要learned predictor network

### 10.3 STP的unification

STP的insight：如果Geodesic Hypothesis成立（trajectory local linear），那么predictor退化成identity。

$$\text{JEPA}: \text{predict } \phi(y) \text{ from } \phi(x) \text{ via learned predictor } p$$
$$\text{STP}: \text{predict } h_t - h_r \approx h_r - h_s \text{ via identity}$$

这解决了LLM-JEPA的三个limitations:
1. 不需要two-view scaffolding（任意segment都行）
2. 不需要额外forward pass（hidden state已经算了）
3. 不需要learned predictor（identity就够）

### 10.4 Energy-Based View

LeCun的EBM (https://www.cs.toronto.edu/~hinton/csc2535notes/lecun-06.pdf) minimize energy at specific states。STP generalize这个philosophy：minimize **action**（trajectory-wise integral of Lagrangian）而非state-wise energy。

从local energy minimization到trajectory-wise action minimization。

---

## 11. 和其他理论的关系

### 11.1 Linear Representation Hypothesis

LRH (Park et al. 2024, https://arxiv.org/abs/2311.03658): concepts编码为directions，比如 $\vec{v}_{Paris} - \vec{v}_{France} + \vec{v}_{Italy} \approx \vec{v}_{Rome}$。

Geodesic Hypothesis generalizes LRH: 不只是simple concepts，composed concepts (token sequences) 也沿locally linear trajectories运动。LRH是single concept的特例。

Figure 3展示了 $\vec{v}_{Paris}, \vec{v}_{to}, \vec{v}_{France}, \vec{v}_{is}, \vec{v}_{Rome}, \vec{v}_{to}, \vec{v}_{Italy}$ 几乎共线。

### 11.2 Manifold Hypothesis

Manifold Hypothesis (Robinson et al. 2025, https://arxiv.org/abs/2408.04786; Whiteley et al. 2025; Kiani et al. 2024): learned representations form smooth manifold。

Geodesic Hypothesis下，manifold structure是Principle of Least Action的自然consequence。

### 11.3 Curvature Straightening

Curvature Straightening (Henaff et al. 2021, https://www.nature.com/articles/s41467-021-25939-z; Hosseini & Fedorenko 2023, NeurIPS): 生物V1区和LLM training都倾向于straighten consecutive tokens之间的curvature。

STP把这个现象解释为underlying geodesic的manifestation——Principle of Least Action的直接结果，而非epiphenomenon。

### 11.4 Exposure Bias

Exposure Bias (Bengio et al. 2015, https://arxiv.org/abs/1506.03099): training用teacher forcing，inference用自己prediction，导致distribution shift。

Huszar 2015 (https://arxiv.org/abs/1511.05101) argue MLE optimize的目标和generation quality不同。STP通过explicit regularizer缓解这个gap。

### 11.5 Neural Tangent Kernel

NTK (Jacot et al. 2018, https://arxiv.org/abs/1806.07566) 在infinite-width limit简化dynamics。Yang & Littwin 2021 (https://arxiv.org/abs/2010.04696) generalize到Transformer。

STP用NTK framework来justify Gaussian channel approximation和Brownian motion accumulation。

---

## 12. 实现细节

HuggingFace transformers上的pseudocode：

```python
def compute_stp_loss(hidden_states, s, r, t):
    # hidden_states: [batch, seq_len, d_model] from last layer
    h_s = hidden_states[:, s, :]  # [batch, d_model]
    h_r = hidden_states[:, r, :]
    h_t = hidden_states[:, t, :]
    
    v1 = h_r - h_s  # semantic evolution s -> r
    v2 = h_t - h_r  # semantic evolution r -> t
    
    cos_sim = F.cosine_similarity(v1, v2, dim=-1)
    stp_loss = 1 - cos_sim
    return stp_loss.mean()

# Total loss
loss = ntp_loss + lambda * stp_loss
```

**Computational overhead**: 几乎为0。Forward pass已经算了hidden states，只需要一次cosine similarity。

**Choosing indices**:
- 一般：random sample $s < r < t$
- Two-view data (query, answer): $s$ at query start, $t$ at answer end, $r$ random
- Multiple-choice Q&A with distractors: $x_{[s,r]}$ 是query, $x_{[r',t]}$ 是correct answer，loss = $1 - \cos(h_t - h_{r'}, h_r - h_s)$，skip中间的distractor branches

---

## 13. Limitations

1. **Principle of Least Action是hypothesis**: LLM是否真的minimize某种action？没有first-principles derivation。作者承认这是"simplified form of self-consistency"。

2. **Gaussian channel approximation**: 在infinite-width limit成立，但实际LLM finite width。Approximation quality unclear。

3. **Local linearity假设**: 可能只在sufficiently smooth manifold成立。Polysemous tokens, ambiguity, humor等可能violate。

4. **STP over-regularization风险**: $\lambda$ 太大会压死diversity。实验显示 $\lambda \in [0.01, 0.08]$ 窗口比较窄。

5. **实验局限在fine-tuning**: 没有pre-training实验。如果STP真能violate scaling law，应该在pre-training scale验证。

6. **NL-RX-SYNTH可能过简单**: 主要data efficiency结果在这个synthetic dataset上。GSM8K等更复杂dataset的data efficiency没系统report。

7. **Theoretical guarantee loose**: $\|h_r - h^*\|_2 \lesssim \varepsilon$ 是asymptotic bound，tightness unclear。

8. **NTP + STP的trade-off**: paper没深入讨论STP loss是否会hurt NTP在需要divergent thinking的task上（creative writing等）。

---

## 14. 延伸方向

1. **Pre-training scale的STP**: 如果data efficiency真能scale到pre-training，这是revolutionary。目前只在fine-tuning验证。

2. **Curvature-aware STP**: 当前假设linear。用second-order fit（quadratic in $t$）可以extend到longer sequences with curvature。从STP到quadratic STP，类比linear到second-order Taylor expansion。

3. **Multi-scale STP**: 不同时间scale（tokens, sentences, paragraphs）apply STP，类似multi-scale feature pyramid。

4. **STP for RLHF/preference learning**: preference本质是manifold上的ordering constraint，STP-like loss可能用于alignment。

5. **Connection to flow matching / diffusion**: STP的ODE/SDE formulation和flow matching (Lipman et al. 2023, https://arxiv.org/abs/2210.02747) 有structural similarity。能否用flow matching更好model hidden state dynamics？

6. **JEPA unification theory**: STP揭示"JEPA predictor退化成identity when locally linear"。能否formalize：什么样的manifold结构需要learned predictor vs identity predictor？

7. **Mechanistic interpretability**: 如果geodesic是真实structure，attention heads应该implement它。能否reverse engineer attention circuits来verify geodesic存在？

8. **STP和In-context learning**: ICL本质是trajectory上conditioning，STP的trajectory constraint可能和ICL有deep connection。

9. **Curvature作为uncertainty signal**: 如果某段trajectory的STP loss很大，可能暗示high uncertainty / OOD / ambiguity。能否用STP loss做active learning或OOD detection？

10. **STP加速test-time compute**: 类似OpenAI o1的test-time scaling，能否inference时用STP loss来select best trajectory among multiple samples？

---

## 15. 我的核心takeaway

**NTP只约束hidden state落入正确的Voronoi cell，不约束cell内的位置。这个flexibility让training可以收敛到一个"对cell但错geodesic"的hidden state，导致inference mode collapse。**

STP的fix：强制local segments共线（cosine similarity = 1），把hidden state trajectory约束在以geodesic为中心的tube内。

这个constraint:
- 提升SNR（noise在perpendicular方向，被suppress）
- 提升data efficiency（information theoretic link）
- 提升accuracy（Fano's inequality）
- 保持diversity（收窄inference cone，减少collision）

**最有意思的地方**: STP和JEPA的unification——JEPA的predictor在locally linear manifold上退化成identity。这暗示language的structure prior比一般JEPA setting更强——segments真的共线，不需要learned projection。

**最值得follow-up的方向**: Pre-training scale的STP实验。如果data efficiency真能scale，这是revolutionary。

**核心message**: 别把scaling law当iron law。它描述的是NTP下的typical training。换一个objective——加入正确的geometric prior——可能violate power-law bounds。STP是first concrete demonstration of this principle in LLMs。

---

## References

- Paper code: https://github.com/galilai-group/llm-jepa#stp
- LLM-JEPA (前作): https://arxiv.org/abs/2509.14252
- I-JEPA (Assran et al. 2023): https://arxiv.org/abs/2301.08243
- LeCun JEPA position paper: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Chinchilla scaling laws: https://arxiv.org/abs/2203.15556
- Kaplan scaling laws: https://arxiv.org/abs/2001.08361
- Linear Representation Hypothesis (Park et al. 2024): https://arxiv.org/abs/2311.03658
- Linear Representation Geometry (Park et al. 2025): https://openreview.net/forum?id=bVTM2QKYuA
- Curvature straightening in V1 (Henaff et al. 2021): https://www.nature.com/articles/s41467-021-25939-z
- Tensor Programs IIB (Yang & Littwin 2021): https://arxiv.org/abs/2010.04696
- Neural ODE Transformers (Tong et al. 2025): https://arxiv.org/abs/2502.09820
- SDE-Net (Kong et al. 2020): https://arxiv.org/abs/2006.07220
- Energy-Based Models tutorial (LeCun et al. 2006): https://www.cs.toronto.edu/~hinton/csc2535notes/lecun-06.pdf
- NL-RX-SYNTH dataset (Locascio et al. 2016): https://arxiv.org/abs/1608.06115
- GSM8K (Cobbe et al. 2021): https://arxiv.org/abs/2110.14168
- Spider (Yu et al. 2018): https://arxiv.org/abs/1809.08887
- NQ-Open (Lee et al. 2019): https://aclanthology.org/P19-1612/
- HellaSwag (Zellers et al. 2019): https://arxiv.org/abs/1905.07958
- Llama 3 (Grattafiori et al. 2024): https://arxiv.org/abs/2407.21787
- Gemma 2 (Team et al. 2024): https://arxiv.org/abs/2408.00118
- OpenELM (Mehta et al. 2024): https://arxiv.org/abs/2404.14619
- OLMo 2 (2024): https://arxiv.org/abs/2501.00656
- Qwen3 (2025): https://arxiv.org/abs/2505.09388
- DeepSeek-R1 (2025): https://arxiv.org/abs/2501.12948
- Flow Matching (Lipman et al. 2023): https://arxiv.org/abs/2210.02747
- Picard-Lindelöf theorem: https://en.wikipedia.org/wiki/Picard%E2%80%93Lindel%C3%B6f_theorem
- Principle of Least Action: https://en.wikipedia.org/wiki/Principle_of_least_action
- Fano's inequality: https://en.wikipedia.org/wiki/Fano%27s_inequality
- Channel capacity (Shannon 1948): https://en.wikipedia.org/wiki/Channel_capacity
- Voronoi diagrams: https://en.wikipedia.org/wiki/Voronoi_diagram
- Geodesic: https://en.wikipedia.org/wiki/Geodesic
- Exposure Bias / Scheduled Sampling (Bengio et al. 2015): https://arxiv.org/abs/1506.03099
- Huszar 2015 on MLE: https://arxiv.org/abs/1511.05101
- Dimensional Collapse in SSL (Jing et al. 2021): https://arxiv.org/abs/2110.09348
- Data2vec (Baevski et al. 2022): https://arxiv.org/abs/2204.02605
- Manifold Hypothesis (Robinson et al. 2025): https://arxiv.org/abs/2408.04786
- NTK (Jacot et al. 2018): https://arxiv.org/abs/1806.07566

---

# Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA 深度解析

Andrej，这篇paper是Huang、LeCun和Balestriero的工作，本质上是把LeCun的JEPA思想推到language domain的一个elegant generalization。它的核心thesis很provocative：**Chinchilla scaling laws描述的是typical training，而optimal training可以violate这些power-law bounds**——前提是加入正确的geometric prior。下面我来一层一层build intuition。

---

## 1. Big Picture: 这篇paper想挑战什么？

Scaling laws（Kaplan 2020, Hoffmann/Chinchilla 2022）告诉我们loss随compute、data、parameters以power-law下降。社区基本把这些law当成iron law。但作者argue：这些law是descriptive而非prescriptive——它们刻画了**当前objective（next-token prediction, NTP）下的typical training dynamics**，并没说什么是最优的。

NTP的根本limitation是：它是local objective，把surface statistical noise和global semantic signal混在一起。作者想做的事情：**显式约束hidden state dynamics，把error-free semantic trajectory从noise中分离出来**。

他们提出的工具叫**Semantic Tube Prediction (STP)**——一个JEPA-style的auxiliary regularizer。实验结果很striking：在NL-RX-SYNTH数据集上，Llama-3.2-1B用**1/16的数据**就能匹配full-data baseline的accuracy。

GitHub repo: https://github.com/galilai-group/llm-jepa#stp

---

## 2. Geodesic Hypothesis: 物理直觉

这是paper的conceptual core。作者invoke了两个classical physics / math的principle：

### 2.1 Principle of Least Action

在经典力学里，Hamilton's principle说：系统从state A到state B走的path，minimize action $S = \int_a^b L \, dt$，其中 $L = T - V$ 是Lagrangian（kinetic minus potential energy）。这个principle产生的path就是**geodesic**——manifold上的"直线"。

参考：https://en.wikipedia.org/wiki/Principle_of_least_action

作者把这个idea borrow到LLM：如果LLM training process产生了一个smooth semantic manifold，那么error-free token sequence trajectories应该沿geodesics运动。而smooth manifold上的geodesics**almost everywhere locally linear**（这是Riemannian geometry基本结果，参考https://en.wikipedia.org/wiki/Geodesic）。

### 2.2 Geodesic Hypothesis（formal statement）

> The trajectory of $\boldsymbol{x_{\leq t}} \in \mathbb{R}^{T \times d_{model}}$ is locally linear almost everywhere. Similarly, the trajectory $h_t - \epsilon_t \in \mathbb{R}^d$ is locally linear almost everywhere.

变量解释：
- $x_{\leq t}$: token sequence of length $t$，embedding在 $\mathbb{R}^{T \times d_{model}}$，$T$ 是max sequence length
- $h_t$: 在token $t$ 处last-layer的hidden state
- $\epsilon_t$: residual unembedding error（即 $h_t$ 与optimal hidden state $h_t^* = \mathring{f}(x_{\leq t})$ 的差）
- $d_{model}$: model hidden dimension
- "almost everywhere": 除了measure zero的singular points

注意这里有个subtle点：$h_t^* = h_t - \epsilon_t$，所以 $h_t - \epsilon_t$ local linear等价于说optimal hidden state trajectory local linear。

### 2.3 Linear Representation Hypothesis的generalization

Linear Representation Hypothesis（LRH，Park et al. 2024，https://arxiv.org/abs/2311.03658）说concepts在representation space中编码为directions，比如 $\vec{v}_{Paris} - \vec{v}_{France} + \vec{v}_{Italy} \approx \vec{v}_{Rome}$。

Geodesic Hypothesis generalize了LRH：**不只是simple concepts，composed concepts（token sequences）也沿locally linear trajectories运动**。Figure 3展示了 $\vec{v}_{Paris}, \vec{v}_{to}, \vec{v}_{France}, \vec{v}_{is}, \vec{v}_{Rome}, \vec{v}_{to}, \vec{v}_{Italy}$ 几乎共线。

LRH是Geodesic Hypothesis在single concept的特例；Geodesic Hypothesis是LRH在sequence level的推广。

---

## 3. Training ODE: 把LLM dynamics建模成ballistic trajectory

### 3.1 Formalization

设：
- $x_{\leq t}$: token sequence of length $t$
- $h_t = f(x_{\leq t})$: hidden state，$f(\cdot)$ 是neural network
- $u(h_t)$: unembedding，预测next token $x_{t+1}$

Converged network下（loss minimized），training dynamics是：
$$x_{t+1} = \mathring{u} \circ \mathring{f}(x_{\leq t}) \tag{1}$$
$$h_t = \mathring{f}(x_{\leq t}) + \epsilon_t \tag{2}$$

变量解释：
- $\mathring{f}, \mathring{u}$: converged后network的函数（带圆圈symbol表示converged）
- $\epsilon_t$: residual unembedding error
- $\circ$: function composition

关键观察：$h_{t+1}$ depends on whole history $x_{\leq t}$，不单纯依赖 $h_t$，所以hidden state dynamics不是Markovian，**但sequence dynamics是Markovian**。Define prefix-removal operator $\ominus$：

$$x_{\leq t+1} \ominus x_{\leq t} = \mathring{u} \circ \mathring{f}(x_{\leq t})$$

这形式上类似 $z_{t+1} - z_t = g(z_t, t)$，因此可以approximate成ODE。

### Proposition 2.1 (Training ODE)

LLM training可以建模为token sequence space $\mathbb{R}^{T \times d_{model}}$ 上的ODE：

$$dx_{\leq t} = \mathring{u} \circ \mathcal{f}(x_{\leq t}) \, dt$$

Appendix A证明：在特定embedding arrangement下，$\ominus$ 可以treat成vector subtraction $x_{\leq t+1} - x_{\leq t}$。具体构造是把 $x_t$ 放在sequence的 $t$-th位置（其他位置补0），这样 $x_{\leq t+1} - x_{\leq t}$ 正好等于只有 $x_{t+1}$ 在 $(t+1)$-th位置、其他位置为0的vector。

### 3.2 Picard-Lindelöf Theorem的含义

参考https://en.wikipedia.org/wiki/Picard%E2%80%93Lindel%C3%B6f_theorem

如果 $\mathring{u} \circ \mathring{f}(\cdot)$ 和它的partial derivatives连续（即Lipschitz-continuous），那么对给定initial condition，ODE有**唯一解**。

推论：**不同initial conditions（prompts）产生的trajectories不能intersect**。如果intersect了，就违反了uniqueness——从intersection point出发只有一条trajectory。

这theoretically rules out mode collapse、preserves diversity。但前提是trajectory error-free。下一节会讲为什么NTP alone保证不了这个。

---

## 4. Inference SDE: 为什么NTP alone不够

### 4.1 Optimal trajectory vs. actual trajectory

Define optimal hidden state trajectory：
$$h_t^* = h_t - \epsilon_t = \mathring{f}(x_{\leq t}) \tag{3}$$

如果 $\mathring{f}$ Lipschitz-continuous，$h^*$ 也ballistic。

### 4.2 Voronoi Cell Argument

NTP loss的目标：让 $u(h_t)$ 收敛到discrete token $x_{t+1}$。由于token是discrete、hidden state是continuous，训练过程相当于**找对Voronoi cell**——但不约束 $h_t$ 在cell内的具体位置。

参考：Voronoi diagrams https://en.wikipedia.org/wiki/Voronoi_diagram

这个flexibility是必要的：让不同geodesics可以在同一个cell内**不同位置穿越**，避免intersect，保住Picard-Lindelöf。但flexibility也是危险的：$h_t$ 可能在training时drift到**错误的geodesic**上——虽然仍在对的Voronoi cell内，但下一步走向完全不同的token。

Figure 2的例子：两个相似prefix的trajectory都通过"researcher"的Voronoi cell。NTP只保证cell正确，但 $h_t$ 可能跳到另一条geodesic，把Hinton的Nobel Prize错误归给另一个人。这就是**mode collapse at inference**。

**核心insight**：$\mathcal{L}_{NTP}$ alone不足以drive $\epsilon_t \to 0$，所以不足以保证generation quality。这是prediction P1的来源。

### 4.3 Inference SDE

在training时，teacher forcing让 $\epsilon_t$ 不传播到下一个token（ground truth $x_{t+1}$ 直接喂回）。但inference时 $h_{t+1}$ depends on $h_t$，所以 $\epsilon_t$ 累积。

基于infinite-width limit（Yang & Littwin 2021, Tensor Programs IIB, https://arxiv.org/abs/2010.04696），pre-activations converge到Gaussian processes，$\epsilon_t$ 是i.i.d. Gaussian。累积起来形成Brownian motion：

### Proposition B.1 (Inference SDE)

$$dx_{\leq t} = \mathring{u} \circ \mathcal{f}(x_{\leq t}) \, dt + \sigma_t \, dW_t$$

变量解释：
- $dW_t$: Brownian motion增量
- $\sigma_t$: noise scale，由 $\epsilon_t$ 决定
- 第一项：deterministic drift（沿geodesic）
- 第二项：stochastic diffusion（noise）

### 4.4 Inference Cone

由Donsker's theorem，$\frac{1}{\sqrt{t}} \sum_{s \leq t} \epsilon_s \sim \mathcal{N}(0, \Sigma)$，所以 $\|h_t - h_t^*\|_2 \propto \sigma \sqrt{t}$。

Inference时，trajectory发散成一个**cone**，半径 $\propto \sigma_t \sqrt{t}$。$\sigma_t$ 越大，cone越宽，越容易collide另一个token sequence trajectory，导致mode collapse。

**STP的作用**：reduce $\sigma_t$ → 收窄cone → 减少collision probability → 保持diversity。这是prediction P3的formal basis。

---

## 5. Semantic Tube Prediction: 核心公式

### 5.1 The Loss

$$\mathcal{L}_{STP} = 1 - \cos(h_t - h_r, h_r - h_s)$$

完整training loss：
$$\mathcal{L} = \mathcal{L}_{NTP} + \lambda \cdot \mathcal{L}_{STP}$$

变量解释：
- $s < r < t$: 三个token的indices（**randomly sampled**）
- $h_s, h_r, h_t$: 对应位置last layer的hidden states
- $h_r - h_s$: 从 $s$ 到 $r$ 的**semantic evolution vector**（context-aware！）
- $h_t - h_r$: 从 $r$ 到 $t$ 的semantic evolution vector
- $\cos(\cdot, \cdot)$: cosine similarity
- $\lambda$: hyperparameter，实验显示 $\lambda \in [0.01, 0.08]$ work best

### 5.2 为什么是context-aware?

$h_t - h_s$ **不是** isolated subsequence $x_{[s,t]}$ 的hidden state。它是full context $x_{\leq s}$ + subsequence $x_{[s,t]}$ 产生的semantic shift。

例子（Figure 9）：
- Prefix "The capital of" + "France" → 把trajectory推向 $\vec{v}_{Paris}$
- Prefix "The language of" + "France" → 把trajectory推向 $\vec{v}_{French}$

同一个token "France"，在不同prefix下，semantic shift完全不同。如果只计算 "France" 的isolated hidden state，会lose context nuance。

### 5.3 为什么 $\lambda \ll 1$?

Geodesic理论上local linear，但实际有slight curvature。$\lambda \ll 1$ 让STP loss提供soft constraint，**tolerate angular deviation** from perfect linearity。实验验证：$\lambda \approx 0.01$ 跨所有setting都work，过大会collapse trajectory。这是prediction P4。

---

## 6. Theoretical Guarantees

### Definition 3.1 (Local Linearity)

Trajectory $h^*$ locally linear if $\exists \tau, \exists \varepsilon$，对任意 $s < r < t$ with $|t-s| \leq \tau$：

$$\|(h_r^* - h_s^*)_{\perp h_t^* - h_s^*}\|_2 \leq \varepsilon \tag{4}$$

变量解释：
- $x_{\perp y}$: vector $x$ 垂直于vector $y$ 的component
- $\varepsilon$: small tolerance for deviation
- $\tau$: local window size

### Lemma 3.2 (Straightening Lemma)

If $h_s = h_s^*, h_t = h_t^*$ and $\mathcal{L}_{STP} \leq \epsilon$ for all $r$ with $s < r < t$:

$$\|(h_r - h_s)_{\perp h_t^* - h_s^*}\|_2 \leq \sqrt{2\epsilon} \|h_r - h_s\|_2$$

证明思路（Appendix D）：cosine loss $\mathcal{L}_{STP} = 1 - \cos\theta' \leq \epsilon$，small $\epsilon$ 下 $\cos\theta' \approx 1 - \theta'^2/2$，所以 $\theta' \lesssim \sqrt{2\epsilon}$，再 $\sin\theta' \approx \theta'$ 得到perpendicular component bound。

### Theorem 3.3 (Semantic Tube)

If $h^*$ locally linear and $\mathcal{L}_{STP} \to 0$ for all $r$ with $0 \leq s < r < t \leq \tau$:

$$\|h_r - h^*\|_2 \lesssim \varepsilon$$

含义：$h_r$ 被约束在以 $h^*$ 为中心、半径 $O(\varepsilon)$ 的tube内。这就是"Semantic Tube"的formal定义。

Appendix E证明：通过引入 `<before-bos>` 和 `<after-eos>` 辅助tokens，保证boundary conditions $h_0 = h_0^*$ 和 $h_{\tau+1} = h_{\tau+1}^*$。

### Corollary 3.4 (Random Tube)

Random $s < r < t$，$\mathcal{L}_{STP} \to 0$ 意味着 high probability下 $\|h_r - h^*\|_2 \leq \varepsilon + \epsilon$。这是Markov's inequality的应用。

---

## 7. SNR, Data Efficiency, Accuracy的Information Theoretic Link

这是Section H，给了一个elegant的formalization，把"SNR提升"和"data efficiency/accuracy提升"link起来。

### 7.1 Data Efficiency Lemma (H.1)

$$H(Y | X^m) \geq H(Y) - m \cdot I(Y; X) \tag{5}$$

变量解释：
- $Y$: discrete target token，from vocabulary $\mathcal{V}$, $|\mathcal{V}|$ 是vocab size
- $X^m = \{X_i\}_{i=1}^m$: $m$ 个conditional i.i.d. hidden states
- $H(Y | X^m)$: conditional entropy，asymptotic equivalent to cross-entropy loss
- $I(Y; X)$: mutual information between target token和单个hidden state
- $H(Y)$: marginal entropy of token

证明关键：mutual information的chain rule + sub-additivity of entropy。$X_i$ 条件independent given $Y$ 让 $H(X^m | Y) = \sum_i H(X_i | Y)$，sub-additivity给 $H(X^m) \leq \sum_i H(X_i)$。结合起来 $I(Y; X^m) \leq m \cdot I(Y; X)$。

### 7.2 Gaussian Channel Approximation

Infinite-width limit（Yang & Littwin 2021）下，pre-activations converge到Gaussian。所以model local representation dynamics成 $X = Z + N$：
- $Z$: latent signal
- $N \sim \mathcal{N}(0, \sigma^2 I)$: additive Gaussian noise

SNR定义：
$$\text{SNR} = \frac{\mathbb{E}[\|Z\|^2]}{\mathbb{E}[\|N\|^2]}$$

Shannon's Gaussian channel capacity（https://en.wikipedia.org/wiki/Channel_capacity）：
$$I(X; Y) = \frac{1}{2} \log(1 + \text{SNR})$$

### Corollary H.2 (Signal-to-Noise Ratio)

代入Gaussian channel capacity到Data Efficiency Lemma：

$$m \geq \frac{H(Y) - \epsilon}{\frac{1}{2} \log(1 + \text{SNR})} \tag{6}$$

**Interpretation**: 所需数据量 $m$ 和 $\log(1 + \text{SNR})$ 成反比。STP提升SNR → 严格降低 $m$。这是data efficiency的理论保证。

### Corollary H.3 (Accuracy via Fano)

Fano's inequality（https://en.wikipedia.org/wiki/Fano%27s_inequality）：
$$H(Y | X^m) \leq H_b(P_e) + P_e \log(|\mathcal{V}| - 1)$$

LLM中 $|\mathcal{V}| \gg 1$，简化为：
$$H(Y | X^m) \lesssim P_e \log|\mathcal{V}|$$

代入 (5) 得到：
$$P_e \gtrsim \frac{H(Y) - m \cdot \frac{1}{2}\log(1 + \text{SNR})}{\log|\mathcal{V}|} \tag{8}$$

**Interpretation**: SNR提升 → accuracy提升（$P_e$ 下降）。

**这两个corollary是paper的核心prediction P2的理论基础**：SNR提升 → data efficiency提升 + accuracy提升。

---

## 8. 实验结果深度解析

### 8.1 Loss Landscape (Figure 4)

实验setup: Llama-3.2-1B-Instruct on NL-RX-SYNTH (https://arxiv.org/abs/1608.06115)

**Figure 4a** 的发现：
- Regular fine-tuning: $\mathcal{L}_{NTP}$ 收敛后plateau，$\mathcal{L}_{STP}$ 也跟着停止下降
- Semantic Tube: $\mathcal{L}_{NTP}$ plateau时，$\mathcal{L}_{STP}$ **继续下降**

这验证P1：$\mathcal{L}_{NTP}$ alone不能自动minimize $\mathcal{L}_{STP}$，需要explicit auxiliary loss。

**Figure 4b** 的发现：
- $\lambda$ 在log scale上increase → $\mathcal{L}_{STP}$ linear下降
- $\mathcal{L}_{NTP}$ 保持stable across wide $\lambda$ range

具体数值：
- $\lambda = 0$ (regular): $\mathcal{L}_{STP} \approx 1.4$（接近erratic Brownian motion）
- $\lambda = 0.08$: $\mathcal{L}_{STP} \approx 0.6$（substantially smoother path）
- Optimal accuracy在 $\lambda = 0.02$，$\lambda = 0.08$ accuracy只marginally lower

### 8.2 Accuracy across datasets / model families / sizes (Figure 5)

**Datasets** (Figure 5a): NL-RX-SYNTH, NL-RX-TURK, GSM8K (https://arxiv.org/abs/2110.14168), Spider (https://arxiv.org/abs/1809.08887), NQ-Open (https://aclanthology.org/P19-1612/), HellaSwag (https://arxiv.org/abs/1905.07958)。所有6个dataset上STP都胜过regular fine-tuning和LLM-JEPA。

**Model families** (Figure 5b): Llama-3.2-1B, gemma-2-2b-it (https://arxiv.org/abs/2408.00118), OpenELM-1.1B (https://arxiv.org/abs/2404.14619), OLMo-2 (https://arxiv.org/abs/2501.00656), Qwen3-1.7B (https://arxiv.org/abs/2505.09388), DeepSeek-R1-Distill-Qwen-1.5B (https://arxiv.org/abs/2501.12948)。6个families都work。

**Model sizes** (Figure 5c): Llama-3 1B, 3B, 8B。三个size都improve，说明prior是model-agnostic。

### 8.3 Data Efficiency (Figure 1b, Figure 12)

NL-RX-SYNTH on Llama-3.2-1B，subset fractions $\{1/2, 1/4, 1/8, 1/16, 1/32\}$，按比例scale epochs (1/n data → n× epochs):

| Fraction | Regular FT | Semantic Tube |
|----------|------------|--------------|
| 1/1 (full) | baseline | matches baseline |
| 1/2 | significant drop | negligible drop |
| 1/16 | catastrophic | matches full-data regular FT |
| 1/32 | — | starts degrading |

3B和8B显示相似trend（Figure 12）。

实验还测试了half compute（n/2 × epochs + 2× learning rate）+ 2× $\lambda$。有趣发现：half-compute double-LR在full或1/2 data下不是最优，但**在 < 1/2 data下反而更好**。这暗示data-constrained regime下，aggressive learning rate配合strong STP prior能squeeze more signal。

### 8.4 Diversity Preservation (Table 1, Figure 6)

NL-RX-SYNTH中正则表达式可以以 `.*` 或 `.*.*` 结尾（功能等价但 `.*` 在训练集中35×更频繁）：

| Suffix | Semantic Tube | Regular | LLM-JEPA |
|--------|---------------|---------|----------|
| `.*` | 88.5% | 29.9% | 68.9% |
| `.*.*` | 68.0% | 28.0% | 32.0% |

Regular FT两者都学不好。LLM-JEPA collapse到majority pattern (`.*`)，minority (`.*.*`) 只有32%。**STP两者都学好**——这是diversity preservation的direct evidence，验证P3。

**Polymorphism SVD分析**（Figure 6）：计算 $\text{Enc}(\text{Text}) - \text{Enc}(\text{Code})$ 的SVD谱：
- Without normalization: STP profile像regular FT（tolerate raw complexity）
- With normalization: STP profile像LLM-JEPA（enforce directional structure）

**Interpretation**: STP在directions上enforce structure，在magnitudes上tolerate complexity。这种**polymorphism**让STP既有regularization效果又保住flexibility，可能是diversity preservation的mechanism。

### 8.5 λ Tuning (Figure 7, Table 2)

所有dataset和model的optimal $\lambda$：

| Setting | Optimal λ |
|---------|-----------|
| SYNTH / TURK / GSM8K / Spider / NQ / HS | 0.02 / 0.04 / 0.005 / 0.04 / 0.16 / 0.02 |
| Gemma2 / Qwen3 / R1-Distill / OLMo / OpenELM | 0.005 / 0.02 / 0.04 / 0.01 / 0.04 |
| Llama3 3B / Llama3 8B | 0.01 / 0.0025 |

所有optimal $\lambda \in [0.005, 0.16]$，curve concave，过optimal后accuracy急速下降。验证P4：$\lambda \ll 1$。

### 8.6 Ablation (Figure 8, Tables 4-5)

**Variations**:
- **Semantic Tube (vanilla)**: random $s, r, t$, identity predictor → best
- **Zero**: fix $s=0$, random $r, t$ → worse
- **Pred**: learnable linear projector $P$, loss = $1 - \cos(P(h_r - h_s), h_t - h_s)$ → worse in all configs. **Validates P5: identity predictor > learned predictor**
- **Inst**: add system prompt → worse
- **Two Views**: LLM-JEPA style (fix $s=0$, $r$ at query end) → worse than random sampling
- **Mask**: BERT-style mask-and-recover → worse
- **Curvature**: minimize angle $|\theta_i|$ between consecutive segments → worse

p-values（paired one-tailed t-test, 5 seeds）：Semantic Tube vs Two View: 4.76e-3; vs Mask: 1.77e-3; vs Curvature: 3.04e-5。**全部statistically significant**。

---

## 9. 和JEPA的深度关系

### 9.1 JEPA背景

JEPA（Joint-Embedding Predictive Architecture，LeCun 2022, https://openreview.net/pdf?id=BZ5a1r-kVsf）的核心idea：在representation space预测一个view基于另一个view，避免pixel/token-level reconstruction。

参考：
- I-JEPA (Assran et al. 2023): https://arxiv.org/abs/2301.08243
- Data2vec (Baevski et al. 2022): https://arxiv.org/abs/2204.02605

### 9.2 LLM-JEPA的前作

同一组人的LLM-JEPA (Huang et al. 2025, https://arxiv.org/abs/2509.14252) 把JEPA推到LLM，但有几个practical limitations：
1. 需要manual two-view scaffolding（query + answer）
2. 需要额外forward pass（fractional compute cost）
3. 需要learned predictor network

### 9.3 STP的unification

STP的insight：**如果Geodesic Hypothesis成立，任何token sequence segment都align with global trajectory，所以predictor退化成identity function**。

$$\text{JEPA}: \text{predict } \phi(y) \text{ from } \phi(x) \text{ via predictor } p$$
$$\text{STP}: \text{predict } h_t - h_r \approx h_r - h_s \text{ via identity}$$

这是JEPA的**special case where predictor = identity**，因为local linearity。但special case反而stronger——它捕捉到language的structure prior。

### 9.4 Energy-Based View

LeCun的Energy-Based Models (EBM, https://www.cs.toronto.edu/~hinton/csc2535notes/lecun-06.pdf) minimize energy at specific states。STP generalize了这个philosophy：minimize **action**（trajectory-wise integral of Lagrangian）而非state-wise energy。这是从local energy minimization到trajectory-wise action minimization的generalization。

---

## 10. 和其他理论的关系

### 10.1 Manifold Hypothesis

Manifold Hypothesis（Robinson et al. 2025, https://arxiv.org/abs/2408.04786; Whiteley et al. 2025; Kiani et al. 2024）：learned representations form smooth manifold。Geodesic Hypothesis下，这是Principle of Least Action的自然consequence。

### 10.2 Curvature Straightening

Curvature Straightening Phenomenon（Hosseini & Fedorenko 2023, https://proceedings.neurips.cc/paper_files/paper/2023/file/...; Henaff et al. 2021, https://www.nature.com/articles/s41467-021-25939-z）：生物V1区和LLM training都倾向于straighten consecutive tokens之间的curvature。

STP把这个现象解释为underlying geodesic的manifestation——不是epiphenomenon，而是Principle of Least Action的直接结果。

### 10.3 Neural Tangent Kernel

NTK（Jacot et al. 2018, https://arxiv.org/abs/1806.07566）在infinite-width limit简化dynamics。Yang & Littwin 2021 (https://arxiv.org/abs/2010.04696) generalize到Transformer。STP用NTK framework来justify Gaussian channel approximation和Brownian motion accumulation。

### 10.4 Exposure Bias

Exposure Bias（Bengio et al. 2015, https://arxiv.org/abs/1506.03099）：训练用teacher forcing，inference用自己prediction，导致distribution shift。Huszar 2015 (https://arxiv.org/abs/1511.05101) argue MLE optimize的目标和generation quality不同。STP通过explicit regularizer缓解这个问题。

---

## 11. Implementation细节

HuggingFace transformers上的pseudocode：

```python
def compute_stp_loss(hidden_states, s, r, t):
    # hidden_states: [batch, seq_len, d_model] from last layer
    h_s = hidden_states[:, s, :]  # [batch, d_model]
    h_r = hidden_states[:, r, :]
    h_t = hidden_states[:, t, :]
    
    v1 = h_r - h_s  # semantic evolution s -> r
    v2 = h_t - h_r  # semantic evolution r -> t
    
    cos_sim = F.cosine_similarity(v1, v2, dim=-1)
    stp_loss = 1 - cos_sim
    return stp_loss.mean()

# Total loss
loss = ntp_loss + lambda * stp_loss
```

**Computational overhead**: 几乎为0——forward pass已经算了hidden states，只需要一次cosine similarity。

**Choosing indices**:
- 一般情况：random sample $s < r < t$
- Two-view data: anchor $s$ at query start, $t$ at answer end, $r$ random
- Multiple-choice Q&A with distractors: pick $r, r'$ such that $x_{[s,r]}$ 是query, $x_{[r',t]}$ 是correct answer，loss = $1 - \cos(h_t - h_{r'}, h_r - h_s)$，**skip中间的distractor branches**

---

## 12. 可能的延伸方向（speculative）

1. **Curvature-aware STP**: 当前假设geodesic linear。如果用second-order fit（quadratic in $t$），可以extend到longer sequences with curvature。从STP到quadratic STP，类比linear到second-order Taylor expansion。

2. **Multi-scale STP**: 不同时间scale（tokens, sentences, paragraphs）apply STP，类似multi-scale feature pyramid。

3. **STP for RLHF/preference learning**: preference本质是manifold上的ordering constraint，STP-like loss可能用于alignment。

4. **STP in pre-training**: paper实验全在fine-tuning。如果在大规模pre-training加STP，能否accelerate scaling law？这是open question，potential巨大。

5. **Connection to flow matching / diffusion**: STP的ODE/SDE formulation和flow matching（Lipman et al. 2023, https://arxiv.org/abs/2210.02747）有structural similarity。能否用flow matching更好model hidden state dynamics？

6. **JEPA unification theory**: STP揭示了"JEPA predictor退化成identity when locally linear"。能否formalize：什么样的manifold结构需要learned predictor vs identity predictor？

7. **Mechanistic interpretability**: 如果geodesic是真实structure，attention heads应该implement它。能否reverse engineer attention circuits来verify geodesic存在？

8. **STP和In-context learning**: ICL本质是trajectory上conditioning，STP的trajectory constraint可能和ICL有deep connection。

9. **Curvature作为uncertainty signal**: 如果某段trajectory的STP loss很大，可能暗示high uncertainty / OOD / ambiguity。能否用STP loss做active learning或OOD detection？

10. **STP加速test-time compute**: 类似OpenAI o1的test-time scaling，能否inference时用STP loss来select best trajectory among multiple samples？

---

## 13. Limitations和Critique

1. **Principle of Least Action是hypothesis**: 作者承认这是"simplified form of self-consistency"。LLM是否真的minimize某种action？没有first-principles derivation。

2. **Gaussian channel approximation**: 在infinite-width limit成立，但实际LLM finite width。Approximation quality unclear。

3. **Local linearity assumption**: 可能只在sufficiently smooth manifold成立。Polysemous tokens, ambiguity, humor等可能violate。

4. **STP over-regularization risk**: $\lambda$ 太大会把trajectory压成一条直线，loss diversity。这就是为什么 $\lambda \ll 1$ 重要，但也是fragility。

5. **Experiments局限在fine-tuning**: 没有pre-training实验。如果STP真能violate scaling law，应该在pre-training scale验证。Fine-tuning scale的1/16数据saving可能不会scale到pre-training。

6. **NL-RX-SYNTH可能过简单**: 主要data efficiency结果在这个synthetic dataset上。GSM8K等更复杂dataset的data efficiency没系统report。

7. **Theoretical guarantee loose**: $\|h_r - h^*\|_2 \lesssim \varepsilon$ 是asymptotic bound，实际constant和tightness unclear。

8. **NTP + STP的trade-off**: paper没深入讨论STP loss是否会hurt NTP在某些task上的performance（比如需要divergent thinking的creative writing）。

---

## 14. 我的整体解读

这篇paper的**deep insight**是：NTP只约束hidden state落入正确的Voronoi cell，但不约束cell内的位置。这个flexibility让training可以收敛到一个"对cell但错geodesic"的hidden state，导致inference mode collapse。

STP的**fix**是：通过强制local segments共线（cosine similarity = 1），把hidden state trajectory约束在以geodesic为中心的tube内。这个constraint：
- 提升SNR（noise在perpendicular方向，被suppress）
- 提升data efficiency（information theoretic link）
- 提升accuracy（Fano's inequality）
- 保持diversity（收窄inference cone，减少collision）

**最有意思的地方**是STP和JEPA的unification：JEPA的predictor在locally linear manifold上退化成identity。这暗示language的structure prior比一般JEPA setting更强——segments真的共线，不需要learned projection。

**最值得follow-up的方向**：
1. Pre-training scale的STP实验——如果data efficiency真能scale，这是revolutionary
2. Curvature-aware STP——extend到longer sequences
3. STP和ICL、test-time compute的关系
4. Mechanistic interpretability——attention circuits如何implement geodesic

如果Geodesic Hypothesis是correct的，这篇paper可能是LeCun的JEPA vision在language domain的第一个真正convincing instantiation。它suggests language的geometric structure比community想象的强得多，而NTP alone underutilize了这个structure。

**核心takeaway**: 别把scaling law当iron law。它描述的是NTP下的typical training。换一个objective——加入正确的geometric prior——可能violate power-law bounds。STP是first concrete demonstration of this principle in LLMs。

---

## References (web links)

- Paper code: https://github.com/galilai-group/llm-jepa#stp
- LLM-JEPA (前作): https://arxiv.org/abs/2509.14252
- I-JEPA (Assran et al. 2023): https://arxiv.org/abs/2301.08243
- LeCun JEPA position paper: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Chinchilla scaling laws: https://arxiv.org/abs/2203.15556
- Kaplan scaling laws: https://arxiv.org/abs/2001.08361
- Linear Representation Hypothesis (Park et al. 2024): https://arxiv.org/abs/2311.03658
- Linear Representation Geometry (Park et al. 2025): https://openreview.net/forum?id=bVTM2QKYuA
- Curvature straightening in V1 (Henaff et al. 2021): https://www.nature.com/articles/s41467-021-25939-z
- LLM curvature straightening (Hosseini & Fedorenko 2023): https://papers.nips.cc/2023/hash
- Tensor Programs IIB (Yang & Littwin 2021): https://arxiv.org/abs/2010.04696
- Tensor Programs IV (Yang & Hu 2021): https://arxiv.org/abs/2010.04696
- Neural ODE Transformers (Tong et al. 2025): https://arxiv.org/abs/2502.09820
- SDE-Net (Kong et al. 2020): https://arxiv.org/abs/2006.07220
- Energy-Based Models tutorial (LeCun et al. 2006): https://www.cs.toronto.edu/~hinton/csc2535notes/lecun-06.pdf
- NL-RX-SYNTH dataset (Locascio et al. 2016): https://arxiv.org/abs/1608.06115
- GSM8K (Cobbe et al. 2021): https://arxiv.org/abs/2110.14168
- Spider (Yu et al. 2018): https://arxiv.org/abs/1809.08887
- NQ-Open (Lee et al. 2019): https://aclanthology.org/P19-1612/
- HellaSwag (Zellers et al. 2019): https://arxiv.org/abs/1905.07958
- Llama 3 (Grattafiori et al. 2024): https://arxiv.org/abs/2407.21787
- Gemma 2 (Team et al. 2024): https://arxiv.org/abs/2408.00118
- OpenELM (Mehta et al. 2024): https://arxiv.org/abs/2404.14619
- OLMo 2 (2024): https://arxiv.org/abs/2501.00656
- Qwen3 (2025): https://arxiv.org/abs/2505.09388
- DeepSeek-R1 (2025): https://arxiv.org/abs/2501.12948
- Flow Matching (Lipman et al. 2023): https://arxiv.org/abs/2210.02747
- Picard-Lindelöf theorem: https://en.wikipedia.org/wiki/Picard%E2%80%93Lindel%C3%B6f_theorem
- Principle of Least Action: https://en.wikipedia.org/wiki/Principle_of_least_action
- Fano's inequality: https://en.wikipedia.org/wiki/Fano%27s_inequality
- Channel capacity (Shannon 1948): https://en.wikipedia.org/wiki/Channel_capacity
- Voronoi diagrams: https://en.wikipedia.org/wiki/Voronoi_diagram
- Geodesic: https://en.wikipedia.org/wiki/Geodesic
- Exposure Bias / Scheduled Sampling (Bengio et al. 2015): https://arxiv.org/abs/1506.03099
- Huszar 2015 on MLE: https://arxiv.org/abs/1511.05101
- Dimensional Collapse in SSL (Jing et al. 2021): https://arxiv.org/abs/2110.09348
