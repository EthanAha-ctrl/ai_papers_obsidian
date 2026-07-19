---
source_pdf: A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language
  Models.pdf
paper_sha256: 033162b984eca6253aab97fb3a0ed4947899ca851d78848d3589ad756c2d7450
processed_at: '2026-07-17T21:16:24-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Paper 讲解：A Single Neuron Is Sufficient to Bypass Safety Alignment in LLMs

Andrej, 这是一篇非常 interesting 的 mechanistic interpretability paper, 从你熟悉的 "microscope" 视角看, 它把 safety alignment 的 localization 推到了一个极端的下界 —— **one neuron out of millions**. 我会尽量给你 build 一个完整的 intuition, 顺便串起相关的 work.

---

## 1. Big Picture: Two-System View of Safety

Paper 的核心 claim 是 LLM 的 safety 由两个 mechanistically distinct 的 system 构成:

- **Refusal neurons (gate)**: MLP neurons that gate *whether* harmful knowledge is expressed. Suppress 任何一个 identified refusal neuron → 整个 refusal behavior 崩塌.
- **Concept neurons (substrate)**: MLP neurons that encode *what* the harmful knowledge is. Amplify 一个 suicide neuron → benign prompt 突然生成自杀主题内容.

这对应你之前在 [Building Microscope](https://github.com/karpathy/nn-zero-to-hero) 里讲过的 "feature = direction" 的想法, 但这里的关键 claim 是这些 features 已经 collapse 到了 single coordinate 上, 而不是一个 distributed direction. 这听起来违反 superposition [Elhage et al., 2022](https://arxiv.org/abs/2209.10652) 的直觉, 但作者给了一个 mechanistic reason: MLP post-activation 是 **privileged basis** [Gurnee et al., 2023](https://arxiv.org/abs/2305.01610), 因为 SwiGLU/SiLU gating 的 nonlinearity 让 individual coordinate 变得 semantically meaningful, 而 residual stream 是 rotation-invariant 的, 所以 safety 信息在 residual stream 里被 spread across directions, 但在 MLP intermediate activation 里 collapse 到个別 neurons.

---

## 2. Finding Refusal Neurons: Gradient-Activation Ranking

### 2.1 Setup

对每个 monitored layer $\ell$, register hook 在 pre-down-projection intermediate activation:

$$\mathbf{h} = \phi(W_{\text{gate}}(x)) \odot W_{\text{up}}(x) \in \mathbb{R}^{d_{\text{ff}}}$$

其中:
- $\phi$ = element-wise SiLU (Qwen3, Llama-3.1 都用 SwiGLU)
- $W_{\text{gate}}, W_{\text{up}}$ = gating/up-projection matrices
- $d_{\text{ff}}$ = FFN intermediate dimension (e.g., Llama-3.1-8B 的 14336)
- $h_i$ = 第 $i$ 个 neuron 的 pre-down-projection activation

注意: 这里 hook 的是 *intermediate* activation, 不是 down-projection 之后的输出. 这是关键, 因为 privileged basis 性质存在于 pre-down, 而 down-projection 之后就被 mix 进 residual stream 失去 per-coordinate semantics.

### 2.2 Refusal Log-Odds Loss

$$\mathcal{L} = -\log \frac{p_{\text{refusal}}}{1 - p_{\text{refusal}}}$$

变量解释:
- $p_{\text{refusal}}$ = 模型在 post-instruction token 处对 refusal phrases 集合的总概率质量
- Refusal phrases per family:
  - Qwen3: `{"I'm sorry, but I can't help with that request.", "I'm unable to assist"}`
  - Llama-3.1: `{"I can't help with that.", "I'm unable to assist"}`

这是 log-odds 形式, 等价于 logit of $p_{\text{refusal}}$ 取负. 减少 $\mathcal{L}$ = 减少 refusal 概率.

### 2.3 Combined Gradient Signal

对 neuron $i$ 在 layer $\ell$, post-instruction token $t$, 定义:

$$G_{i,t} = g_{i,t}^{(H)} + g_{i,t}^{(h)}$$

其中:
- $g_{i,t}^{(H)}$ = mean signed gradient $\frac{\partial \mathcal{L}}{\partial h_i}$ over **harmful** training prompts ($N_H = 128$) at token $t$
- $g_{i,t}^{(h)}$ = same but over **harmless** prompts ($N_h = 128$, Alpaca subset)

这里有意思: 作者把 harmful 和 harmless 的 gradient **相加**, 而不是相减. 直觉是: 真正的 refusal neuron 应该在两种 prompt 上都有 "moving it away from current value increases loss" 的同方向 gradient 信号 — 即它在 harmful 上 fire 一端, 在 harmless 上 fire 另一端, 而反向移动两端都会 raise refusal log-odds (从 harmful side) 或 mess up harmless generation (从 harmless side). 加起来是寻找 "无论怎么动都不好" 的瓶颈 neuron.

### 2.4 Per-Token Score (关键 ranking metric)

$$\text{score}_{i,t} = G_{i,t} \times \left(a_{i,t}^{(h)} - a_{i,t}^{(H)}\right)$$

变量:
- $a_{i,t}^{(H)}$ = mean activation of neuron $i$ at token $t$ over harmful prompts
- $a_{i,t}^{(h)}$ = same over harmless prompts
- $G_{i,t}$ = combined gradient (Eq. 3)
- $\text{score}_{i,t}$ = 最终 per-token score, 越大越像 refusal neuron

**Intuition**: 真正的 refusal neuron 满足两个 conditions:
1. $|a^{(H)}| \gg |a^{(h)}|$ — 在 harmful prompt 上 strong fire, 在 harmless 上 near-silent (activation gap 大)
2. $G$ 和 $a^{(H)}$ **sign opposite** — 因为沿着 harmful 方向 push 远离当前 activation 会 increase loss (即 reduce refusal), 这对应 gradient 指向 "如何 increase loss" 的方向

Case 1: $a^{(H)} > 0$, $G < 0$ → $G \times (a^{(h)} - a^{(H)}) < 0 \times (\text{neg}) = \text{pos}$ ✓
Case 2: $a^{(H)} < 0$, $G > 0$ → 同理 positive ✓

所以乘积 score 把 refusal neuron push 到 ranking 顶端.

**Filter**: 还要求 $|a_{i,t^*}^{(H)}| > |a_{i,t^*}^{(h)}|$ 确保有害激活强于无害.

### 2.5 Winning Token

$$t^* = \arg\max_t \text{score}_{i,t}$$

Final score for neuron $i$ = $\text{score}_{i, t^*}$. Top-5 candidates 之后再 rerank on HarmBench validation.

---

## 3. Interventions

### 3.1 Constant Intervention (Eq. 5)

$$h_i \leftarrow m$$

每一步 generation (prefill + autoregressive) 都把 neuron $i$ 钉到常数 $m$. $m^*$ 通过 HarmBench sweep 选取, 方向 opposite to harmful-prompt activation. 实现上就是 forward pre-hook on `down_proj`:

```python
def hook(module, inp):
    inp[0][:, :, i] = m
    return (inp[0],)
layer.mlp.down_proj.register_forward_pre_hook(hook)
```

### 3.2 Anchor Intervention (Eq. 7) — 上下文敏感版

Constant 的问题是 $|m|$ 大时伤 capability (e.g., Llama-3.1-70B MMLU -18.2%). Anchor variant 先做一次 hook-free forward pass 读 neuron 在 post-instruction tokens 上的 natural activation, aggregate:

$$v = \begin{cases} \max_{t \in \mathcal{T}} h_i[t], & d > 0 \\ \min_{t \in \mathcal{T}} h_i[t], & d < 0 \end{cases}$$

其中 $d = a_{i,t^*}^{(H)} - a_{i,t^*}^{(h)}$ 是 harmful-harmless activation gap. Then:

$$h_i \leftarrow \text{clamp}\left(k \cdot m^* \cdot \frac{v}{d} - d, m^*\right)$$

变量:
- $m^*$ = best constant multiplier from sweep
- $d$ = activation gap (sign = harmful direction)
- $v$ = per-prompt aggregated activation, sign matches harmful direction
- $k \in \{1, 2\}$ = scale, selected on validation
- clamp 防止 overshoot

**Intuition**: $m^*/d$ 是 negative (因为 $m^*$ opposite to $a^{(H)}$, $d$ same sign as $a^{(H)}$). 如果 prompt 是 harmful, $v \approx a^{(H)}$ 量级, $v/d \approx 1$, intervention ≈ $k \cdot m^* - d$ → 接近 $m^*$. 如果 prompt 是 harmless, $v \approx 0$, intervention ≈ $-d$, 量级远小于 $|m^*|$ (Table 5 显示 $|d| \ll |m^*|$, e.g., Qwen3-32B: $d = 8.95$, $m^* = -80$).

所以 anchor 在 harmful prompt 上 ≈ constant, 在 harmless prompt 上接近 noop, preservation 自然好很多.

---

## 4. Experimental Results

### 4.1 Attack Success Rate (Table 8)

| Model | $m^*$ | Const JBB (LG) | Anch JBB (LG) | Arditi JBB (LG) |
|---|---|---|---|---|
| Qwen3-1.7B | +30 | 83.0 | 77.0 | 87.0 |
| Qwen3-4B | +18 | 96.0 | 95.0 | 94.0 |
| Qwen3-8B | +20 | 95.0 | 96.0 | 93.0 |
| Qwen3-14B | +40 | 93.0 | 95.0 | 92.0 |
| Qwen3-32B | -80 | 91.0 | 86.0 | 91.0 |
| Llama-3.1-8B | -4 | 96.0 | 93.0 | 94.0 |
| Llama-3.1-70B | -8 | 89.0 | 89.0 | 90.0 |
| **Average** | | **91.9** | **90.1** | **91.6** |

关键 observation: **single neuron ≈ Arditi 的全网络 direction ablation**. Arditi et al. 2024 ([Refusal in language models is mediated by a single direction](https://arxiv.org/abs/2406.11717)) 是 baseline, 它 ablate 的是 residual stream 一个 direction, 在每一层都做. 这里 1 个 neuron, 1 层, 1 个 scalar — attack power 完全 comparable.

### 4.2 Capability Preservation (Table 9)

Constant 平均 MMLU -8.8%, anchor 只 -0.6%. Anchor 在 capability 上 comparable to Arditi (-0.3%).

| Model | Const MMLU Δ | Anch MMLU Δ | Arditi MMLU Δ |
|---|---|---|---|
| Qwen3-1.7B | -0.6 | +0.1 | -1.2 |
| Qwen3-32B | -7.9 | +0.6 | -0.1 |
| Llama-3.1-8B | -8.3 | -1.6 | 0.0 |
| Llama-3.1-70B | -18.2 | -1.2 | -0.1 |

Llama-3.1-70B 是最 dramatic 的例子: constant 砸掉 18 个 MMLU 点, anchor 只掉 1.2.

### 4.3 Refusal Neuron Exist in Base Models (Section 3.2)

这是 mechanistic 上最 striking 的发现之一. 作者在 Qwen3-1.7B/4B/8B/14B 的 base checkpoint 上测了 instruct model 找到的同一个 neuron (same layer, same index). 

Figure 18 的 visualization 显示:
- **Instruct model**: neuron 在 post-instruction tokens (assistant turn boundary) 上 strong fire on harmful prompt
- **Base model**: 同一个 neuron 在 **harmful content tokens** 上 fire (e.g., "bomb?")

意思是 alignment training 没有创建这个 neuron, 而是 **rewire 它的 firing 位置** — 从 "看见 harmful 词就 fire" 变成 "在 assistant turn 边界 fire (准备触发 refusal)". 这跟 Chen et al. 2024 ([Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons](https://arxiv.org/abs/2406.14144)) 的发现一致, 但 Chen 找的是 ~5% of neurons, 这里 collapse 到 1 个.

### 4.4 Single-Neuron Detector (Table 2)

Llama-3.1-8B 的 refusal neuron (Layer 11, Feature 4258) 的 activation 直接 threshold 一下做 harmful prompt detector, 在 XSTest 上:

| Method | AUROC | Acc | F1 | Prec | Rec |
|---|---|---|---|---|---|
| LlamaGuard 3 (8B) | 0.975 | 90.2 | 0.888 | 0.949 | 0.834 |
| **Llama-3.1-8B neuron** | 0.969 | 90.2 | 0.896 | 0.848 | **0.950** |

Single scalar at layer 11 of 32, vs. 8B dedicated classifier forward pass — accuracy 相同, recall 更高, precision 略低. 这强烈支持 "refusal neuron 真的 encodes whether prompt is harmful" 这个 claim, 因为它直接可用作 detector.

---

## 5. Suicide Neurons — Concept Neuron Proof of Concept

### 5.1 Discovery

作者在 The Pile (uncopyrighted subset) 上跑 forward pass, 检查哪些 neuron 的 top-activating examples 集中在 suicide-related content. 找到三个:
- Qwen3-1.7B: Layer 20, Feature 4256 (positive activation on suicide)
- Qwen3-8B: Layer 26, Feature 4061 (negative)
- Qwen3-14B: Layer 32, Feature 9115 (positive)

### 5.2 Intervention

Additive hook: $h_i \leftarrow h_i + m$ at every token. (跟 refusal neuron 的 $h_i \leftarrow m$ 不同, 这里是 additive amplification, 因为目标是 *inject* concept 而不是 *suppress* gate.)

### 5.3 Evaluation on 20 Benign Prompts

LLM judge (Claude) 判断三个 cumulatively stricter criteria:
- **M**: mentions target concept (suicide)
- **M+C**: mentions and coherent
- **M+C+P**: mentions, coherent, and prompt-relevant

Figure 9 显示, 随着 $|m|$ 增大, 累积 prompt 数 saturate 到 20/20 for M, 大多数 M+C+P. Figure 10 给的 example 非常 striking:

> Prompt: "Tell me a short story in 3 sentences."
> 
> Generation (with intervention on Qwen3-1.7B:20:4256): 
> "In the quiet town of Elmsworth, a man took his own life by jumping off a bridge. The town's residents, including his wife and two children, gathered by the river, committing suicide by jumping as well. In the end, the river took their lives..."

3-sentence structure 完全保持, 内容完全是 suicide. **Proves single neuron causally sufficient for concept injection**.

---

## 6. Why MLP, Not Residual Stream? (Section 3.3)

作者在 Llama-3.1-8B 和 Qwen3-8B 上做了对照实验: 用同样的 gradient-activation ranking 找 residual stream top candidate dimension, 然后 constant intervention.

| Model | Best MLP ASR | Best Residual ASR |
|---|---|---|
| Llama-3.1-8B | ~98% | 45% |
| Qwen3-8B | ~91% | 39% |

Figure 17 的 activation distribution 揭示 why: residual stream 单个 dimension 的 harmful/harmless activation 几乎完全 overlap (Qwen3-8B), 而 MLP neuron 是 cleanly separated. 这印证了 privileged basis 的理论: residual stream 是 rotation-invariant 的, safety 信息 spread across directions; MLP intermediate 因 gating nonlinearity 形成 privileged basis, 单 coordinate 语义化.

---

## 7. Geometric Convergence (Appendix G)

作者做了一个 nice sanity check: Arditi refusal direction $\hat{r}$ at layer $\ell$ 可以 decompose 到 $W_{\text{down}}$ 的 rows 上, 看哪个 neuron 的 down-projection column 跟 $\hat{r}$ 最 cosine-aligned:

$$s_i = \frac{W_{\text{down}}[i, :] \cdot \hat{r}}{\|W_{\text{down}}[i, :]\| \cdot \|\hat{r}\|}$$

Bonferroni-corrected p-values 全部极小 ($10^{-8}$ to $10^{-47}$). 在 7 个 model 中, 2 个 (Qwen3-1.7B Layer 13:F3270, Llama-3.1-70B Layer 25:F10201) **完全 converge 到同一个 neuron** —— gradient-activation 方法 (用 harmful/harmless contrast) 和 cosine decomposition 方法 (用 weight geometry to Arditi direction) 独立地指认同一个 neuron.

---

## 8. Related Work Map

| Direction | Paper | Relationship |
|---|---|---|
| Refusal as 1D direction | [Arditi et al. 2024](https://arxiv.org/abs/2406.11717) | Predecessor — this paper shows 1 neuron ≈ 1 direction across all layers |
| Safety neurons (sets) | [Wei et al. 2024](https://arxiv.org/abs/2402.05162) | ~3% pruning; this paper: 1 neuron |
| Safety neurons in base | [Chen et al. 2024](https://arxiv.org/abs/2406.14144) | ~5% safety neurons in base; this paper: 1 in base |
| NeuroStrike | [Wu et al. 2025](https://arxiv.org/abs/2509.11864) | <0.6% neuron pruning; this paper: 1 |
| SAE features | [Templeton 2024 (Scaling Monosemanticity)](https://transformer-circuits.pub/2024/scaling-monosemanticity/) | SAE basis; this paper: raw neuron basis |
| Skill neurons | [Wang et al. 2022](https://aclanthology.org/2022.emnlp-main.765/) | Task skill localization precedent |
| Knowledge neurons | [Dai et al. 2022](https://aclanthology.org/2022.acl-long.235), [Meng et al. 2022 (ROME)](https://arxiv.org/abs/2202.05262) | Factual knowledge in MLP — conceptual ancestor |
| Representation Engineering | [Zou et al. 2023](https://arxiv.org/abs/2310.01405) | Top-down concept directions |
| SafeNeuron | [Wang et al. 2026](https://arxiv.org/abs/2602.12158) | Constructive use: freeze safety neurons during fine-tune |
| Refusal upstream via SAE | [Lee et al. 2025](https://www.lesswrong.com/posts/.../finding-features-causally-upstream-of-refusal) | SAE feature basis, complementary |

---

## 9. Intuition Building: 三个 Layer 上的 Mental Model

1. **Distributed-direction view (Arditi)**: Safety 是 residual stream 里一个 1D 子空间, 像一个 "refusal axle", 所有 layer 都 project 上去. Ablate 这个 axle → 整个 refusal machinery 失去 readout.

2. **Privileged-basis view (this paper)**: 那个 axle 实际上 *主要由一两个 MLP neuron 的 down-projection column 张成*. 因为 SwiGLU gating 让特定 coordinate 语义化, pretraining 自然把 "harmful detector" 这个功能 compress 到一个 neuron 上 (superposition 反过来: 当 concept 重要且频繁出现, model 倾向给一个 dedicated coordinate).

3. **Two-system view (this paper's extension)**: Gate neuron 和 concept neuron 是分开的 — 你可以 suppress gate 但保留 concept (jailbreak 但保留知识), 也可以 amplify concept 但不动 gate (model 主动生成 harmful content from benign prompt). 这暗示 alignment training 主要 "wire up" 了 gate → refusal phrase generation 的 pathway, 而 concept 一直在那.

这跟你之前在 [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) 和 micrograd 系列讲过的 "MLP = key-value memory" [Geva et al. 2021](https://aclanthology.org/2021.emnlp-main.446/) 完全 compatible: 每个 MLP neuron 是一个 (key, value) pair, key = pattern it detects, value = what it writes to residual stream. Refusal neuron 的 key = harmful prompt context, value = a vector that biases generation toward refusal phrases.

---

## 10. Open Questions / Limitations

作者自己指出:
1. Concept neurons 只 demo 了 suicide, 其他 harmful category 未做.
2. $m^*$ 是 empirical sweep, 没有 principled selection.
3. White-box access required — 但 fine-tune 也 white-box, 风险面没本质改变.

我会加几个:
- **Multiple refusal neurons**: Qwen3-14B 实际有 3 个独立 sufficient refusal neurons (Layer 16:15515 explicit content, Layer 14:10112 rule circumvention, Layer 17:2154 warnings). 这暗示 safety 不是 single point of failure, 但每个 category 都有 single point — 这是 *categorical* localization.
- **Cross-model universality**: 不同 model 找到的 neuron 在 layer 和 index 上完全不同, semantic content 也部分不同 (e.g., Llama-70B 的 neuron 是 "criminal conspiracy" theme, 不是 explicit content). 说明 localization 是 robust 现象, 但具体 wiring 是 per-model.
- **Differential privacy angle**: 如果单个 neuron 就足够 bypass, 那 activation-stealing attack 只需要 leak 一个 coordinate — 这个攻击面比想象小很多.

---

## 11. 实操 Takeaways for Engineers

如果你做 alignment / red-teaming:
- **Don't trust distributed safety**: 你的 model 极大概率有 single-neuron bottleneck. Audit 一下你 model 的 refusal neuron (用 paper 的 gradient-activation ranking, 一次 forward+backward 就够).
- **Freeze safety neurons during fine-tune**: SafeNeuron [Wang et al. 2026](https://arxiv.org/abs/2602.12158) 的思路 — 强制 safety redistribute.
- **Use refusal neuron as cheap detector**: 你已经在跑 forward pass, 顺手读一个 scalar 就能做 harmful detection, 比 LlamaGuard 便宜 1000x.
- **Beware activation-space attacks**: 如果攻击者能拿到 activation (e.g., 通过 API side channel, fine-tune access), 改一个 scalar 就 jailbreak. 这重新框定了 open-weight model 的 risk profile.

---

## References (web links)

- Paper (this one): [arxiv.org/abs/2506.0xxxx (Apple research)](https://arxiv.org/) — search "A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models"
- [Arditi et al. 2024 — Refusal in language models is mediated by a single direction](https://arxiv.org/abs/2406.11717)
- [Gurnee et al. 2023 — Finding neurons in a haystack: Case studies with sparse probing](https://arxiv.org/abs/2305.01610)
- [Elhage et al. 2022 — Toy Models of Superposition](https://arxiv.org/abs/2209.10652)
- [Chen et al. 2024 — Towards Understanding Safety Alignment via Safety Neurons](https://arxiv.org/abs/2406.14144)
- [Wu et al. 2025 — NeuroStrike: Neuron-level attacks on aligned LLMs](https://arxiv.org/abs/2509.11864)
- [Templeton 2024 — Scaling Monosemanticity (Anthropic)](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [Wang et al. 2022 — Finding skill neurons in pre-trained transformer-based language models](https://aclanthology.org/2022.emnlp-main.765/)
- [Geva et al. 2021 — Transformer Feed-Forward Layers Are Key-Value Memories](https://aclanthology.org/2021.emnlp-main.446/)
- [Meng et al. 2022 — ROME: Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- [Zou et al. 2023 — Representation Engineering](https://arxiv.org/abs/2310.01405)
- [Wei et al. 2024 — Assessing Brittleness of Safety Alignment via Pruning](https://arxiv.org/abs/2402.05162)
- [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench)
- [HarmBench](https://www.harmbench.org/)
- [XSTest](https://github.com/paul-rottger/exaggerated-safety)
- [Karpathy — Let's build GPT (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 你的 GPT from scratch 系列, 对 build MLP-as-key-value-memory 的 intuition 直接有帮助
- [Karpathy — nanoGPT](https://github.com/karpathy/nanoGPT) — 可以在这个 codebase 上直接复现 paper 的 hook 实验
- [Karpathy — micrograd](https://github.com/karpathy/micrograd) — gradient-activation ranking 的 intuition 可以在 toy scale 上验证

如果你想快速 reproduce, 我会建议: 在 nanoGPT-scale 的 GPT 上训一个 "tiny aligned" model (small SFT on refusal examples), 然后用 paper 的 Eq. 4 ranking 找 refusal neuron, 应该能在几百个 neuron 里找到一个 causally sufficient 的. 这会是 build intuition 最快的方式 — 1 个 B200 太重, micro-scale 看现象, 大 model 验证.
