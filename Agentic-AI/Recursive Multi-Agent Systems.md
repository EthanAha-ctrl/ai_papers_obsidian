---
source_pdf: Recursive Multi-Agent Systems.pdf
paper_sha256: 7b8adc094620015891940651d62608d781bb4cd1c0cd74bbfa6160057b5e59f1
processed_at: '2026-08-11T21:58:30-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 RecursiveMAS

## 一句话说清楚

**让多个 AI agent 之间不通过"说话"（text）沟通，而是直接在"脑子里"（latent space）传递思维，然后像同一群人围成一圈反复迭代思考一样，越想越深。**

---

## 问题是什么

想象你有一个团队：一个 Planner、一个 Critic、一个 Solver，要一起解题。

传统 MAS 怎么做？Planner 写一段 text → Critic 读完整段 text 再写一段 → Solver 再读再写。如果要多轮迭代，每轮都要 decode 出完整的文字，下一个 agent 再 re-encode 回去。

问题在于：
- **慢**：每生成一个 token 都要过一遍 vocabulary projection（vocab 有 32K-128K 这么大）
- **贵**：多轮 recursion 时 token 消耗爆炸增长
- **train 不动**：text 是 discrete 的，gradient 回传时会 vanishing——你很难训 model 学会"怎么更好地协作"

---

## 他们怎么做的

### 1. Agent 不 decode text，直接传 latent vector

每个 agent 正常 forward，拿到 last-layer hidden state $h$。但 $h$ 不投影到 vocabulary，而是通过一个小小的 **RecursiveLink**（就是一个 2-layer residual MLP）直接转成下一个 agent 的 input embedding。

$$\mathcal{R}(h) = h + W_2 \sigma(W_1 h)$$

类比：Agent A 想完之后不"开口说话"，直接把脑子里的 representation 塞给 Agent B，Agent B 接着想。

### 2. 所有 agent 首尾相连形成 loop

Agent 1 想完 → 传给 Agent 2 → ... → Agent N 想完 → 又传回 Agent 1，开始第 2 轮 recursion。

只有最后一轮、最后一个 agent，才把 latent decode 成 text 作为最终答案。中间所有过程都在 latent space 里跑。

### 3. 只训那个小 MLP，base model 全冻住

RecursiveLink 参数量只有 0.31%！整个系统只靠这个轻量 module 来学"怎么协作"。训练成本极低，但效果比 LoRA 或 full SFT 都好。

---

## 为什么这样有用

### 效率层面

Text-based：每步要 $m \times |V| \times d_h$ 的计算（vocab 太大）
Latent-based：每步只要 $m \times d_h^2$（hidden dim 远小于 vocab size）

所以快 1.2-2.4 倍，token 省掉 34.6%-75.6%。

### 训练层面

这是 paper 最漂亮的 insight。当 model 已经比较 confident（预测 entropy 很低）时：
- **Text-based** 的 gradient norm $\leq O(\epsilon)$ → 接近 0 → 梯度消失
- **RecursiveLink** 的 gradient norm $\geq \Omega(1 - \sqrt{\frac{1}{d_h}\log\frac{1}{\delta}})$ → 接近 1 → 梯度稳定

直觉：text 经过 softmax 这个 bottleneck，信息被 squeeze 成 discrete distribution，gradient 回传时信号几乎没了。而 RecursiveLink 有 residual identity connection，gradient 可以高保真地穿过。

所以你才能 train deep recursion（$r=2, 3$ 甚至更深），text-based MAS train 到深轮次时 gradient 就消失了。

---

## 实验告诉你什么

### 主结果

在 9 个 benchmark（数学、科学、医学、代码、搜索）上：
- 平均 accuracy 提升 8.3%
- 推理速度提升 1.2-2.4 倍
- token 消耗降低 34.6%-75.6%

越深的 recursion，RecursiveMAS 的优势越大。因为 text-based 随轮次增加 token 和时间爆炸增长，而 RecursiveMAS 的开销几乎线性。

### 一个有意思的 case study

题目：$2^{24}$ 是多少个 $n > 1$ 的 perfect $n$-th power？

- **Round 1**：答 6（错了，漏了 $n=24$ 对应 $k=1$ 这个 case）
- **Round 2**：答 7（对了，list 出 divisors {2,3,4,6,8,12,24}）
- **Round 3**：答 7（稳定）

你可以看到 latent recursion 在"修正"错误的 reasoning。

### Scaling Law

训练时 recursion 深 + 推理时 recursion 深 = 最好。两个 axis 是 complementary 的：训练教 system 形成可 refine 的 latent state，推理时把这些 state unfold 成更深的 reasoning。

---

## 四种协作模式都 work

- **Sequential**: Planner→Critic→Solver，linear pipeline
- **Mixture**: 多个 domain specialist 并行 → Summarizer 聚合
- **Distillation**: 大 Expert ↔ 小 Learner，RecursiveMAS 比 Learner 强 8%，比 Expert 快 1.5 倍
- **Deliberation**: Reflector ↔ Tool-Caller，还能结合 search/Python tool

说明这个 framework 是 structure-agnostic 的，不是只能用于特定 MAS 拓扑。

---

## 我的直觉

这篇 paper 的核心 contribution 在于：**它把 MAS 从"agents 通过 text 沟通"升级成"agents 通过 latent 沟通 + recursion"**。

text 是给人看的，不是给 model 看的最优 interface。Agent 之间本来就在 latent space 里 think，非要 decode→encode 一遍 text 既慢又 lossy。RecursiveLink 就是那个"翻译层"，让不同 size、不同 family 的 model 也能直接在 latent 上对话。

更关键的是，这个 design 让"system-level training"变得 tractable——你不用去 fine-tune 整个 agent（成本高、容易 catastrophic forgetting），只训一个 0.31% 的小 module 就能 optimize 协作。这就像是给 agents 装上"沟通技能"而不是"思考技能"。

Reference: https://recursivemas.github.io

---

# RecursiveMAS 深度技术解析

这篇 paper 提出了一个相当 elegant 的 idea：将 recursive language models (RLMs) 的 scaling principle 从单模型扩展到 multi-agent system (MAS)，让 agent collaboration 本身成为一个可 scale 的 axis。让我深入拆解。

## 1. Core Intuition: 从 Single-Model Recursion 到 System-Level Recursion

先回顾一下 recursive language model 的基本思路。给定一个 Transformer $f_\theta = \mathcal{M}_L \circ \cdots \circ \mathcal{M}_1$ with $L$ layers，RLM 不再只 forward 一次，而是 reuse 同一组 layers $n$ 次：

$$H^{(0)} = E, \quad H^{(r)} = f_\theta(H^{(r-1)}), \quad r = 1, \ldots, n$$

这里 $E$ 是 input embeddings，$H^{(r)}$ 是第 $r$ 轮 recursion 后的 hidden representation，$f_\theta$ 是 shared Transformer stack。核心 insight 是：**reasoning depth 不需要通过增加 layer 数来实现，而可以通过 reuse 同样的 computation 来达成**。

RecursiveMAS 的 key leap：将每个 agent 视为一个 RLM layer。Agent $A_i$ 的 computation 等价于 $f_{\theta_i}$，整个 MAS 就变成了一个 looped computation：

$$S^{(0)} \xrightarrow[\text{Evolve}]{H^{(1)}} S^{(1)} \xrightarrow[\text{Evolve}]{H^{(2)}} \cdots \xrightarrow[\text{Evolve}]{H^{(n)}} S^{(n)}$$

这里 $S^{(r)}$ 是第 $r$ 轮 recursion 时整个系统的状态，$H^{(r)}$ 是所有 agents 的 collective latent state $\mathcal{H} = \{H_1, \ldots, H_N\}$。Agent 之间在 latent space 中传递信息，形成 closed loop。

## 2. RecursiveLink 架构详解

这是论文的核心 technical contribution。RecursiveLink 有两个 variant：

### 2.1 Inner RecursiveLink $\mathcal{R}_{\text{in}}$

用于 agent 内部的 latent thoughts generation：

$$\mathcal{R}_{\text{in}}(h) = h + W_2 \sigma(W_1 h)$$

变量解释：
- $h \in \mathbb{R}^{d_h}$：agent 的 last-layer hidden state
- $W_1 \in \mathbb{R}^{d_{\text{mid}} \times d_h}$, $W_2 \in \mathbb{R}^{d_h \times d_{\text{mid}}}$：两个 linear projection
- $\sigma(\cdot)$：GELU activation
- Residual connection $+h$：保留原始 latent semantics

这个设计的 intuition 是：latent generation 需要把 last-layer embedding "映射回" input embedding space，而 residual branch 只学 distributional shift，不需要从头学习 full projection。

### 2.2 Outer RecursiveLink $\mathcal{R}_{\text{out}}$

用于跨 agent 的 latent state transfer：

$$\mathcal{R}_{\text{out}}(h) = W_3 h + W_2 \sigma(W_1 h)$$

新增的 $W_3 \in \mathbb{R}^{d_{h_j} \times d_{h_i}}$ 把 source agent $A_i$ 的 embedding（维度 $d_{h_i}$）映射到 target agent $A_j$ 的 embedding space（维度 $d_{h_j}$）。这支持 heterogeneous agents（不同 model family、不同 size）。

### 2.3 Ablation 验证

Table 4 的 ablation 很 informative：

| Design | MATH500 | GPQA-D | LiveCodeBench |
|--------|---------|--------|---------------|
| 1-Layer | 84.4 | 63.2 | 40.1 |
| Res+1-Layer | 86.7 | 65.3 | 41.4 |
| 2-Layer | 85.6 | 64.5 | 40.5 |
| **Res+2-Layer (ours)** | **88.0** | **66.2** | **42.9** |

关键观察：
- Residual connection 带来的 gain 比 layer 数增加更大（Res+1-Layer > 2-Layer on GPQA-D）
- 2-layer + residual 是最优组合，既保留 semantics 又有足够 capacity 学 distributional alignment

## 3. Latent Thoughts Generation: Auto-Regressive in Continuous Space

这是最 interesting 的部分。标准 auto-regressive decoding：

$$h_{t+1} = f_\theta([E_{\leq t}; h_t])$$

这里 $E_{\leq t}$ 是 input embeddings up to position $t$，$h_t$ 是 step $t$ 的 last-layer hidden state。关键区别：$h_t$ 直接 fed back 作为 next input embedding（经过 $\mathcal{R}_{\text{in}}$），而不经过 vocabulary projection → argmax → embedding lookup。

每个 agent $A_1$ 执行 $m$ 步 latent generation：

$$H_{A_1} = [h_t, h_{t+1}, \ldots, h_{t+m}]$$

然后通过 outer link 传给下一个 agent：

$$E_{A_2} \oplus \mathcal{R}_{\text{out}}(H_{A_1})$$

$\oplus$ 表示 concatenation（或 conditioning），$E_{A_2}$ 是 agent $A_2$ 自己的 input context embeddings。

## 4. Theoretical Analysis: 为什么 Latent > Text

### 4.1 Runtime Complexity (Proposition 3.1)

Text-based Recursive MAS:
$$\Theta\Big(N\big(m|V|d_h + (t+m)d_h^2 + (t+m)^2 d_h\big)\Big)$$

RecursiveMAS:
$$\Theta\Big(N\big(md_h^2 + (t+m)d_h^2 + (t+m)^2 d_h\big)\Big)$$

变量解释：
- $N$：agent 数量
- $m$：每个 agent 的 latent generation steps
- $t$：input context length
- $d_h$：hidden dimension
- $|V|$：vocabulary size

关键 difference：$m|V|d_h$ vs $md_h^2$。在实际场景中 $d_h \approx 2048\text{-}8192$，$|V| \approx 32K\text{-}128K$，所以 $|V| \gg d_h$。Text-based 需要每步做 vocabulary projection（$m|V|d_h$），而 RecursiveMAS 只需 latent transformation（$md_h^2$）。

### 4.2 Gradient Stability (Theorem 4.1) - 这是最 important 的理论结果

**Claim**: 在 token confidence 高（entropy $\leq \epsilon$, $\epsilon \ll 1$）的情况下：
- Text-based SFT: $\left\|\frac{\partial \mathcal{R}_{\text{text}}(h)}{\partial h}\right\|_2 \leq O(\epsilon) \ll 1$ → gradient vanishing
- RecursiveMAS: $\left\|\frac{\partial \mathcal{R}(h)}{\partial h}\right\|_2 \geq \Omega\left(1 - \sqrt{\frac{1}{d_h}\log\frac{1}{\delta}}\right)$ → stable gradient

**Proof sketch for text-based**:

Text-based 的 recursive link 可以建模为 $\mathcal{R}_{\text{text}}(h) = W_{\text{in}} \cdot \text{softmax}(W_{\text{out}} h)$，其中 $W_{\text{in}}$ 是 token-to-embedding matrix，$W_{\text{out}}$ 是 embedding-to-logits matrix。

Jacobian:
$$J_{\text{text}} = W_{\text{in}} S W_{\text{out}}, \quad S = \text{diag}(p) - pp^T$$

这里 $p = \text{softmax}(W_{\text{out}} h)$ 是 next-token distribution，$S$ 是 softmax 的 Jacobian（covariance matrix of categorical distribution）。

由 spectral norm 的 sub-multiplicativity：
$$\|J_{\text{text}}\|_2 \leq \|W_{\text{in}}\|_2 \|S\|_2 \|W_{\text{out}}\|_2 \leq O(\|S\|_2)$$

利用 $S$ 是 symmetric PSD，其 spectral norm ≤ trace：
$$\|S\|_2 \leq \text{Tr}(S) = 1 - \|p\|_2^2$$

用 entropy 的 lower bound（ln z ≤ z - 1）：
$$\epsilon \geq \sum_i p_i(-\ln p_i) \geq \sum_i p_i(1 - p_i) = 1 - \|p\|_2^2$$

所以 $\|J_{\text{text}}\|_2 \leq O(\epsilon)$。当模型 confident 时 $\epsilon \to 0$，gradient vanish。

**Proof sketch for RecursiveMAS**:

Jacobian of $\mathcal{R}(h) = h + W_2 \sigma(W_1 h)$:
$$J = I + W_2 \Sigma' W_1$$

这里 $\Sigma' = \text{diag}(\sigma'(W_1 h))$ 是 GELU 的 derivative 的 diagonal matrix。

由 triangle inequality：
$$\big|\|J\|_2 - 1\big| \leq \|J - I\|_2 = \|W_2 \Sigma' W_1\|_2$$

$W_1, W_2$ 用 Kaiming initialization，$|\sigma'| \leq O(1)$，由 subgaussian matrix concentration：
$$\|W_2 \Sigma' W_1\|_2 \leq O\left(\sqrt{\frac{1}{d_h}\log\frac{1}{\delta}}\right)$$

所以 $\|J\|_2 \geq \Omega\left(1 - \sqrt{\frac{1}{d_h}\log\frac{1}{\delta}}\right)$，接近 1。

**Intuition**: Text-based 传递经过 softmax bottleneck（discrete distribution），gradient 被 "squeeze" 到接近 0；RecursiveLink 的 residual identity connection 保证 gradient 高保真传递。这是为什么 text-based MAS 难以 train deep recursion，而 RecursiveMAS 可以。

## 5. Inner-Outer Loop Training

### 5.1 Inner Loop (Model-Level Warm Start)

Objective:
$$\mathcal{L}_{\text{in}} = 1 - \cos\big(\mathcal{R}_{\text{in}}(H), \text{Emb}_{\theta_i}(y)\big)$$

这里 $H$ 是 agent $A_i$ 生成的 latent thoughts，$\text{Emb}_{\theta_i}(y)$ 是 ground-truth text $y$ 经过 agent 的 input embedding layer 得到的 embedding。用 cosine similarity 而非 MSE，因为 cosine 对 magnitude 不敏感，更适合 align distributional structure。

### 5.2 Outer Loop (System-Level Co-optimization)

$$\mathcal{L}_{\text{out}} = \text{CE}\Big(S^{(n)}\big(S^{(n-1)}(\cdots S^{(1)}(x))\big), y\Big)$$

系统 unroll $n$ 轮 recursion，所有 outer links 共享 gradient signal。关键：computation graph 沿 full recursive path 保留，gradient 可以通过 BPTT (Backpropagation Through Time) 回传到每一轮的每个 outer link。

### 5.3 Training Cost Analysis (Table 5)

| Method | GPU Mem (GB) | Trainable Param | Cost | Avg. Acc. |
|--------|--------------|-----------------|------|-----------|
| LoRA Training | 21.67 | 15.92M (0.37%) | $6.64 | 66.9 |
| Full-SFT | 41.40 | 4.21B (100%) | $9.67 | 68.6 |
| **RecursiveMAS** | **15.29** | **13.12M (0.31%)** | **$4.27** | **74.9** |

RecursiveMAS 只训练 RecursiveLink（~0.31% 参数），GPU memory 最低，cost 最低，但 accuracy 最高。这验证了 system-level co-optimization 比 individual agent fine-tuning 更高效。

## 6. Collaboration Patterns & Empirical Results

### 6.1 Four Collaboration Patterns

1. **Sequential Style**: Planner → Critic → Solver (linear pipeline)
2. **Mixture Style**: Math/Code/Science specialists → Summarizer (parallel aggregation)
3. **Distillation Style**: Expert (large) ↔ Learner (small) (knowledge transfer)
4. **Deliberation Style**: Reflector ↔ Tool-Caller (iterative refinement + tool use)

### 6.2 Main Results (Table 2)

以 $r=3$ 为例：

| Benchmark | Recursive-TextMAS | RecursiveMAS | Improvement |
|-----------|-------------------|--------------|------------|
| MATH500 (Light) | 69.1 | 77.8 | +8.7 |
| AIME2025 (Light) | 18.0 | 34.0 | +16.0 |
| AIME2026 (Light) | 16.7 | 20.0 | +3.3 |
| GPQA-D (Light) | 28.7 | 32.6 | +3.9 |
| MedQA (Light) | 28.5 | 31.7 | +3.2 |
| Code Gen (Light) | 29.3 | 37.4 | +8.1 |

Inference speedup: ×2.4, Token reduction: 75.6% at $r=3$.

### 6.3 Comparison with Other Frameworks (Table 3)

| Method | MATH500 | AIME2025 | AIME2026 | GPQA-D | LiveCodeBench | MedQA |
|--------|---------|----------|----------|--------|---------------|-------|
| Single Agent (LoRA) | 83.1 | 70.0 | 73.3 | 62.0 | 37.4 | 76.1 |
| Single Agent (Full-SFT) | 83.2 | 73.3 | 76.7 | 62.8 | 38.6 | 77.0 |
| MoA | 79.8 | 60.0 | 63.3 | 47.6 | 27.0 | 57.5 |
| TextGrad | 84.9 | 73.3 | 76.7 | 62.5 | 39.8 | 77.2 |
| LoopLM | 84.6 | 66.7 | 63.3 | 48.1 | 24.9 | 56.4 |
| Recursive-TextMAS | 85.8 | 73.3 | 73.3 | 61.6 | 38.7 | 77.0 |
| **RecursiveMAS** | **88.0** | **86.7** | **86.7** | **66.2** | **42.9** | **79.3** |

在 AIME2025 上 RecursiveMAS 比 TextGrad 高 13.4 个点，这非常 significant。

### 6.4 Latent Thoughts Length Ablation (Figure 8)

当 $m$ 从 0 增加到 80，accuracy 稳定上升，$m \geq 80$ 后 saturate。这意味着 latent collaboration 不需要太长的 "thinking" budget，远少于 text-based CoT 的 token 量。

## 7. Semantic Distribution Analysis (Figure 7)

用 PCA 可视化 ground-truth answer embeddings vs. RecursiveMAS 生成的 answer embeddings：
- $r=1$：两个 distribution 有 visible shift
- $r=2$：discrepancy 缩小
- $r=3$：largely aligned

这直接展示了 recursion 如何 iteratively refine latent representation 朝向正确答案。

## 8. Case Study 中的 Iterative Refinement

在 MATH500 的 case（"For how many positive integers $n > 1$ is $2^{24}$ a perfect $n$-th power?"）：

- **Round 1**：答 6（错误！漏了 $n=24$ 这个 case，因为 $k=1$ 时 $2^{24} = 1^{24}$... 实际上是 agent 误认为 $n=24$ 对应 $k=1$ 不算，但其实 $k=1$ 时 $n=24$ 是 valid）
- **Round 2**：答 7（正确，divisors of 24 minus 1 = {2,3,4,6,8,12,24}）
- **Round 3**：答 7（稳定正确）

Text-based MAS 也能答对 7，但需要 Planner → Critic → Solver 三个 agent 各自生成 full text，token 消耗大。

## 9. 与 Related Work 的 Positioning

- **LoopLM** (Zhu et al., 2025): single-model recursion，RecursiveMAS 是 system-level extension
- **TextGrad** (Yuksekgonul et al., 2025): text-mediated optimization，RecursiveMAS 用 latent gradient
- **MALT** (Motwani et al., 2024): separately train each agent，RecursiveMAS co-optimize whole system
- **Latent collaboration** (Zou et al., 2025; Du et al., 2025): 类似 latent communication idea，RecursiveMAS 加入 recursion dimension

## 10. My Takeaways & Potential Extensions

**Strengths**:
1. Theoretical justification 严谨（gradient stability proof 很 solid）
2. Structure-agnostic，适用多种 MAS pattern
3. Training cost 极低（只训 RecursiveLink）
4. Scaling law 清晰（training × inference recursion 双 axis）

**Potential extensions I'd explore**:
1. **Adaptive recursion depth**: 类似 Mixture-of-Recursions (Bae et al., 2025)，让 system 自己决定何时 stop
2. **Interpretability**: latent thoughts 能否 decode 出可读的 intermediate reasoning？
3. **Non-sequential topologies**: 当前是 loop，可以扩展到 DAG 或 graph structure (Yun et al., 2026 的 Graph-of-Agents 方向)
4. **RL fine-tuning**: 当前是 supervised，可以加 RLHF/GRPO 来 optimize recursive policy

**References for deeper dive**:
- Project page: https://recursivemas.github.io
- LoopLM: https://arxiv.org/abs/2510.25741
- Latent collaboration: https://arxiv.org/abs/2511.20639
- TextGrad: https://www.nature.com/articles/s41586-024-08461-z
- Recursive language models: https://arxiv.org/abs/2512.24601
- Mixture-of-Recursions: https://arxiv.org/abs/2507.10524
- Scaling test-time compute with latent reasoning: https://arxiv.org/abs/2502.05171

总结一下核心 insight：**agent collaboration 本身可以成为一个 recursive computation，在 latent space 中 iterate 比 text space 更高效且 trainable**。这开启了一条新的 MAS scaling axis，与 model scaling、data scaling 互补。
