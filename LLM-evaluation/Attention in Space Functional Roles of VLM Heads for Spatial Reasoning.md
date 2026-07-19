---
source_pdf: Attention in Space Functional Roles of VLM Heads for Spatial Reasoning.pdf
paper_sha256: 09dfc38e3f0e8c9f49bcc15f9582cab30c22aa690b80490074b316fe6b0739d1
processed_at: '2026-07-18T10:36:54-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# 深度解析: Attention in Space — VLM 中 Spatial Reasoning 的 Functional Head 解剖

Andrej，这篇 paper 我觉得非常对你胃口——它本质上是把 mechanistic interpretability 那一套 (induction heads, retrieval heads, ITI steering) 移植到 multimodal spatial reasoning 上，用 cognitive science 的 framework 来 "命名" attention heads 的功能。下面我把整篇 paper 拆解，并把公式、实验设计、与 LLM 中类似工作的关联都讲清楚。

参考链接:
- Paper arXiv (搜索): https://arxiv.org/abs/2507.09460 (推测)
- ITI (Inference-Time Intervention, Li et al. 2023): https://arxiv.org/abs/2306.03341
- Induction heads (Olsson et al. 2022): https://arxiv.org/abs/2209.11895
- Retrieval heads (Wu et al. 2024): https://arxiv.org/abs/2404.15574
- Probing classifiers (Alain & Bengio 2016): https://arxiv.org/abs/1610.01644
- BERT rediscovery (Tenney et al. 2019): https://arxiv.org/abs/1905.05950

---

## 1. 核心动机: 为什么 VLMs 在 spatial reasoning 上这么拉胯?

人类大脑做 "Is the dog facing the horse?" 这种判断时, 至少需要四条 pathway 协同:
- **Occipital lobe**: 视觉信号接收
- **Ventral stream (occipitotemporal)**: object recognition (what pathway)
- **Dorsal stream (parietal)**: spatial relations (where pathway) — Ungerleider & Mishkin 经典双通路
- **Prefrontal cortex**: relational reasoning + decision-making

而 VLMs 内部是否有类似的 functional specialization? 这篇 paper 给的答案是 **yes, 但是 spatial heads 数量稀缺**——这就直接解释了为什么 spatial reasoning 是 VLM 的弱项。这个 framing 非常 "neuroscience-inspired mechanistic interpretability"，和 Anthropic 的 circuits thread、Conjecture 的 activation steering 都是一脉相承的思路。

---

## 2. CogVSR: 把 spatial reasoning 切成 8 个 cognitive sub-functions

### 2.1 设计哲学

paper 的核心 trick 是 **decompose-then-probe**。直接 probe 一个复杂 spatial question 是模糊的 (哪个 head 负责什么?), 但如果用 CoT 把它拆成 subquestion, 每个 subquestion 对应一个 cognitive function, 那么 probe 每个 subquestion 的 generation process 就能定位 function-specific heads。

8 个 cognitive functions:
| Category | Function | Description |
|---|---|---|
| Spatial | Spatial Perception | 位置、朝向、几何关系 |
| Spatial | Relational Reasoning | 对象间关系比较 |
| Visual | Low-level Visual Perception | 颜色、形状、纹理 |
| Visual | High-level Visual Perception | 物体识别、scene semantics |
| Linguistic | Information Extraction & Understanding | 从 context 中提取信息 |
| Memory | Knowledge Recall | 长期知识检索 |
| Symbolic | Math Reasoning | 计数、算术 |
| Executive | Decision-Making | 最终选择 |

### 2.2 数据生成 pipeline

1. 从 4 个 spatial reasoning benchmarks 各采样 400 examples: VSR、SpatialEval、3DSRbench、Spatial457
2. 用 GPT-o4-mini 做 CoT decomposition, 生成 subQAF triplets: $\{(q_i, a_i, f_i)\}_{i=1}^k$
3. 两阶段 human verification:
   - Stage 1: 3 个 annotator 判断 subquestion 逻辑合理性, 60% agreement 阈值
   - Stage 2: multi-label 重标 cognitive function (因为单个 subquestion 可能涉及多 function, 比如 "Where is the dog relative to the horse?" 同时需要 Spatial Perception + High-level Visual Perception)
4. 最终: **1,142 main QA + 3,759 subQAF triplets**, 其中 Spatial Perception 占 27.29% (1,026), Relational Reasoning 占 15.32% (576)

---

## 3. Probing Framework: 怎么从 attention head 输出推回它的 function?

### 3.1 Head feature extraction

对每个 subquestion $q_i$, 模型 $\mathcal{M}$ 在 context (前面所有 subQ + subA) 下生成答案 $a_i^{\mathcal{M}}$。然后:

1. **Token selection**: 用 Gemini2.5-Flash 挑 top-k 最 informative 的 token (k=3, 见 Table 6 消融), 得到 index set $\mathcal{T}_k$
2. **Head activation extraction**: 对每个 selected token $j \in \mathcal{T}_k$, 抽取所有 layer 所有 head 的 value vector projection 到 residual stream 的部分:
$$x_l^m(j) = \text{head output at layer } l, \text{head } m, \text{token } j$$
3. **Averaging over top-k tokens**:
$$\bar{x}_l^m = \frac{1}{k} \sum_{j \in \mathcal{T}_k} x_l^m(j)$$
4. **Layer-level context augmentation**: 因为 prior work ([50] attention head survey) 表明 cognitive function 随 layer depth 变化, 所以拼上 layer 平均:
$$\bar{x}_l = \frac{1}{M} \sum_{m=1}^M \bar{x}_l^m, \quad \bar{x}_l^{m'} = [\bar{x}_l^m; \bar{x}_l]$$

这个 `[head-level; layer-level]` 的 concatenation 是一个有意思的设计 — 它给了 classifier 一个"我在第几层"的信号。

### 3.2 Multi-label classifier + Gradient×Activation attribution

Probing dataset:
$$\mathcal{D}_{\text{probe}} = \{(\bar{x}_l^{m'}, c)_i\}_{i=1}^N, \quad l \in \{1,\ldots,L\}, m \in \{1,\ldots,M\}$$

MLP 结构 (见 Appendix 12):
- 每个 head feature 过一个 shared linear projection → 64-dim
- flatten + concat 所有 heads (64 × num_heads)
- hidden layer 512 units + ReLU + dropout 0.3
- output softmax over 8 functions
- Adam, lr=1e-4, 100 epochs, cross-entropy

Probing accuracy (Table 7): 78.74% (InternVL3-2B) 到 89.64% (Llama-3.2-11B), 说明 head activations 确实 linearly/MLP-encodes function identity。

### 3.3 Importance score — 公式 (4) 详解

$$I_j^{(c)} = \mathbb{E}_{(\bar{x}, c) \sim \mathcal{D}_{\text{probe}}} \left[\frac{\partial \hat{y}_c}{\partial \bar{x}_j} \cdot \bar{x}_j\right]$$

变量解释:
- $j$: head index (跨 layer 和 head 的 flattened index, 共 $L \cdot M$ 个)
- $c$: cognitive function class (8 类之一)
- $\bar{x}_j$: 第 j 个 head 的 input feature (即那个 64-dim projected vector)
- $\hat{y}_c$: classifier 对 class c 的 logit
- $\frac{\partial \hat{y}_c}{\partial \bar{x}_j}$: logit 对 head feature 的 gradient
- 期望是对整个 probing dataset 取的

这其实就是 **Integrated Gradients 的 baseline=0 版本** (或 gradient×input attribution, Sundararajan et al. 2017 风格)。Intuition: 如果一个 head feature 对 class c 的 logit 有大梯度, 且 feature 本身激活值大, 那它对 function c 重要。

最终得到 importance matrix $\mathbf{I} \in \mathbb{R}^{C \times (L \cdot M)}$ — 每个 (function, head) pair 一个 score。这个 matrix 就是 Figure 2 那个 heatmap 的来源。

---

## 4. 核心发现: Cognitive Heads 的三大性质

### 4.1 Sparsity

Figure 2 (Qwen2.5-VL-7B) 显示: 只有 <9% 的 heads 重要性 score > 0.001。其中 high-level visual perception 和 decision making 占 ~3%, 其他 <1%。这个 sparse pattern 在 6 个 model 上都 reproduce (Appendix Figure 5-9), 包括 InternVL3-2B/8B, Qwen2.5-VL-3B/7B, Llama3.2-11B/90B。

**Intuition**: VLMs 不像我们直觉认为 "所有 head 都参与所有计算", 而是高度 specialized。这与 LLM 中的 induction heads (Olsson 2022)、retrieval heads (Wu 2024) 的发现一致 — 都是少数 head 干大事。

### 4.2 Universality across architectures

跨 InternVL3、Qwen2.5-VL、Llama3.2-Vision 三个 family 都观察到类似的 sparse functional organization。说明这种 specialization 不是某个 model 训练 artifact, 而是 transformer + multimodal training 的 emergent property。

### 4.3 Intrinsic organization within family

Qwen2.5-VL-3B vs Qwen2.5-VL-7B 的分布相似 → 同一 family 不同 scale 也保持 specialization pattern。这暗示 pretraining recipe 决定了 functional head 的位置, scale 只是放大。

### 4.4 Spatial heads 的 scarcity — 关键 insight

仔细看 Figure 2 你会发现: Spatial Perception 和 Relational Reasoning 对应的 heads 不仅总数少, 而且单 head 重要性 score 也弱。这给了我们一个 mechanistic 解释: **VLMs 在 spatial reasoning 上拉胯, 不是因为训练数据不够, 而是因为 model 内部根本没分配多少 capacity 给 spatial function**。这跟 Chen et al. 2025 (https://arxiv.org/abs/2503.01773) 从 attention focus area 角度的分析形成互补 — 一个是 spatial attention 的 focus 不对, 一个是 spatial functional head 数量不够。

---

## 5. Causal Validation: Masking experiments

### 5.1 公式 (5) — Head masking via scaling

$$x_i^{\text{mask}} = \text{Softmax}\left(\frac{W_q^i W_k^{iT}}{\sqrt{d_k/n}}\right) \cdot \epsilon W_v^i$$

变量解释:
- $W_q^i, W_k^i, W_v^i$: 第 i 个 head 的 projection matrices
- $d_k$: head dimension
- $n$: number of tokens (用于 normalization, 这里写法有点奇怪, 通常是 $\sqrt{d_k}$)
- $\epsilon$: 极小值 (e.g., 0.001)

本质上就是: **保留 attention pattern, 但把 value output 缩放到几乎为 0**。这比直接 zero out attention weights 更精细 — 因为它不破坏 residual stream 的其他贡献, 只压制这个 head 的写入。

### 5.2 Table 2 的关键 takeaways

对比 "mask cognitive heads" vs "mask random heads":
- InternVL3-2B Spatial: 61.37 (random mask) → 37.92 (cognitive mask), 降了 ~24%
- Llama3.2-11B Spatial: 74.95 → 44.73, 降了 ~30%
- Llama3.2-90B Spatial: 86.21 → 72.44, 降了 ~14% (大 model 更 robust)
- Qwen2.5-VL-3B Low-level visual: 73.49 → 14.92, 降了 ~58%! 几乎完全失效

这证明 cognitive heads 是 causal 的, 不只是 correlational probe artifact。

### 5.3 Downstream benchmarks 的 negative intervention (Table 4)

在 VSR、Spatial457、SpatialEval、3DSRBench 上:
- InternVL3-8B VSR: 64.35 → 24.13 (降 40%)
- Qwen2.5-VL-7B VSR: 62.47 → 21.76
- Llama3.2-11B VSR: 65.82 → 17.94

Masking spatial heads 直接让 model 在 spatial benchmark 上崩盘 → 这些 heads 是 spatial reasoning 的核心承载者。

---

## 6. SHA: Spatial Head Activation — 怎么 "唤醒" latent spatial heads?

### 6.1 Motivation

如果 spatial heads 稀少, 那能不能让 model 在做 spatial task 时, 强制激活更多 heads 走 spatial pathway? paper 提出的方法很 "土" 但很有效:

**用 Gemini2.5-Flash 检测 objects + bounding boxes + segmentation masks, 然后把 detected regions mask 掉, 作为额外 visual input**。

直觉解释: 当你 mask 掉物体的 high-level visual cues (颜色、纹理), model 就被迫用 spatial 信息 (位置、朝向) 来回答问题。这就把 high-level visual perception pathway 给 "断" 了, forcing model 走 dorsal stream (spatial pathway)。

### 6.2 实验结果 (Table 3)

InternVL3-2B Spatial Perception accuracy:
- Original: 56.82
- +BBox: 58.94
- +Mask: 59.80
- +BBox+Mask (SHA): **68.64** (+11.82%)

Llama3.2-90B-Vision Spatial Perception:
- Original: 82.07
- +BBox+Mask: **87.77** (+5.7%)

Relational Reasoning 同样提升 ~9-10%。

Figure 4 显示 SHA 之后, spatial + relational heads 数量明显增加 — 也就是说 SHA 真的 "激活" 了 latent spatial heads, 而不是通过别的 pathway 提升性能。

### 6.3 我的 critique

这个方法虽然有效, 但依赖外部 detector (Gemini), 不算 "in-model" 的解决方案。更纯粹的做法应该是: 在 inference 时通过 steering vector 强制激活 spatial heads, 类似 ITI。但他们其实也做了这个 (Section 6.5 positive intervention), SHA 是 input-side, ITI-style steering 是 representation-side, 两者互补。

---

## 7. Positive Intervention: Activation Steering (公式 6)

### 7.1 Contrastive activation direction

$$\text{dir}_l^h = \mathbb{E}_{i \in \mathcal{D}_{\text{correct}}}[x_l^h(i)] - \mathbb{E}_{i \in \mathcal{D}_{\text{incorrect}}}[x_l^h(i)]$$

变量解释:
- $\mathcal{D}_{\text{correct}}, \mathcal{D}_{\text{incorrect}}$: 同一个 subquestion, model 回答对 vs 错的 sample 集
- $x_l^h(i)$: sample $i$ 在 layer $l$ head $h$ 的 activation

这完全是 **ITI (Inference-Time Intervention, Li et al. 2023)** 的 contrastive direction 思路 — 用 "对" 的激活减 "错" 的激活, 得到一个 "正确性方向"。

### 7.2 Steering 公式

$$x_l^h(i) \leftarrow x_l^h(i) + \alpha \sigma_l^h \text{dir}_l^h$$

- $\sigma_l^h$: head activation 沿 dir 方向的 std (用来 normalize, 防止 over-shift)
- $\alpha$: steering strength (paper 用 0.1)

### 7.3 Table 5 结果

In-domain (CogVSR) 提升 ~1-3%, out-of-domain (VSR/Spatial457/SpatialEval/3DSRBench) 提升 ~0.5-1%。提升幅度不如 SHA 大, 但更 "纯" — 不改 input, 只改 internal activation。

**Intuition build**: steering 效果比 SHA 小, 说明 spatial reasoning 不只是 "head 激活方向不对", 还有 "head 数量本身不够" 的问题。SHA 通过 input manipulation 间接增加 spatial head 数量, 而 steering 只能在已有 head 上调方向。这跟 LLM 中 truthfulness steering (ITI) 效果显著, 是因为 LLM 内部已经有 truthfulness capability, 只是没被激活; VLMs 在 spatial 上是 capability 本身稀缺。

---

## 8. 跟相关工作的联系 — build broader intuition

### 8.1 与 LLM 中 attention head specialization 的对应

| LLM 中的发现 | VLM 中的对应 (本 paper) |
|---|---|
| Induction heads (Olsson 2022) | Spatial perception heads |
| Retrieval heads (Wu 2024) | Information extraction heads |
| Truthfulness heads / ITI (Li 2023) | Cognitive heads + activation steering |
| Circuit analysis (Anthropic) | Functional head probing |

本 paper 的贡献是把这个 framework 拓展到 multimodal + spatial, 并引入 cognitive science 的 8-function taxonomy。

### 8.2 与 visual grounding heads 的区别

Prior work ([7] Bi et al. 2025, [17] Kang et al. 2025) 发现 VLMs 中有 sparse heads 做 visual grounding (text token ↔ image region alignment)。但 grounding 只是 "哪个 token 对应哪个 region", 不涉及 spatial relation reasoning。本 paper 往前走了一步, 分析 complex multi-step spatial reasoning 中的 head role。

### 8.3 与 VSR、SpatialEval 等 benchmark 的关系

传统 spatial benchmark 只给 final accuracy, 是 black-box。CogVSR 把它 decompose 成 cognitive sub-steps, 才能做 mechanistic analysis。这是 "interpretability-driven benchmark design" 的好例子, 类似于 LLM 中 BIG-Bench 里的 mechanistic probing tasks。

---

## 9. 局限性和未来方向 (paper 自己提的 + 我加的)

Paper 自承:
1. 只看 8 个 predefined cognitive functions, 可能 miss 其他
2. 只分析 attention heads, 没看 MLP (FFN) — 但 FFN 才是 knowledge storage 的地方, 这个 limitation 挺大
3. Probing classifier 本身可能 leak 信息

我加的:
1. **Probing 的因果性问题**: gradient×activation importance 是 correlational, 真正的 causal 还得看 intervention (他们做了, OK)
2. **Multi-label 的问题**: 一个 subquestion 可能涉及多个 function, 用 single-label probe 会引入 noise
3. **Activation direction 的 robustness**: dir 用 correct vs incorrect 求 diff, 但 incorrect 的 sample 太少时 dir 估计 noisy
4. **SHA 的 dependency on Gemini**: 这是 in-the-loop 的 external model, 不 self-contained
5. **没分析 attention pattern 本身**: head output 重要不等于 attention pattern 重要, 可以补 attention rollout / attention flow 分析
6. **没看 cross-head interaction**: spatial reasoning 可能是多个 head 组成的 circuit, 单 head analysis 可能 miss circuit-level structure (Anthropic 的 IOI circuit 思路)

未来方向我觉得有潜力的是:
- **Spatial circuit discovery**: 用 activation patching / path patching 找 spatial reasoning 的完整 circuit
- **Training-time intervention**: 在 SFT/RL 阶段就 boost spatial heads, 而不是 inference-time patch
- **Cross-modal head alignment**: visual encoder 的 spatial position encoding 怎么传到 LLM 的 head 上? 这是 projector 的关键问题
- **Sparse autoencoder on heads**: 用 SAE 拆 head 内部的 features, 可能发现更细粒度的 spatial sub-functions

---

## 10. 我对这篇 paper 的整体评价

**Strength**:
- Cognitive decomposition 的 framework 很 elegant, 把模糊的 "spatial reasoning" 切成可分析的单位
- 跨 6 个 model 的 universality 验证很 solid
- SHA 方法简单有效, practical impact 大
- 把 ITI-style steering 移植到 multimodal spatial 是合理的一步

**Weakness**:
- Probing classifier 这一层本身是个 black box (虽然是 MLP, 但 64-dim projection 是 learned)
- Importance score 的 threshold (0.001) 是 ad-hoc
- "Cognitive head" 的定义依赖 elbow point, 不同 model 可能 elbow 位置不同
- 没分析 head 之间的信息流动 (circuit)
- SHA 依赖 external detector, 不算 in-model 解决方案

**对 Karpathy 你来说的 takeaway**: 这篇 paper 本质上是把你在 "Zipformer... "、nanoGPT 系列里强调的 "look inside the model" 哲学应用到 VLM spatial reasoning 上。它告诉你 VLMs 不是黑盒, 而是有 functional module 的 — 只是 spatial module 比 visual recognition module 弱很多。如果你想 build 一个 spatial-reasoning-strong VLM, 训练时就应该 incentivize 更多 heads 学 spatial function, 而不是依赖 inference-time patch。

---

参考阅读 (deep dive):
- ITI 原文: https://arxiv.org/abs/2306.03341
- Anthropic 的 induction heads blog: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
- CogVSR dataset (应该会 release): 留意 paper GitHub
- Spatial reasoning survey: https://arxiv.org/abs/2510.25760
- 你自己的 "State of GPT" talk 里关于 emergent structure 的部分, 跟这篇 paper 的 cognitive head emergence 是同一个 vibe

如果想 build intuition 的话, 我建议你跑一下 Qwen2.5-VL-7B, 复现 Figure 2 的 heatmap, 然后手动 mask 掉几个 spatial head, 看 model 在 "is the dog facing the horse" 这种问题上怎么崩盘的 — 亲手做一次 mechanistic intervention, 比读 10 篇 paper 都管用。
