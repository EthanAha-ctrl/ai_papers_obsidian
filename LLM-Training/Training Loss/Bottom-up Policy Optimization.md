---
source_pdf: Bottom-up Policy Optimization.pdf
paper_sha256: 3ffaf2e0b9a1c23bdacf78f2ed94f18e8de8b45c77e99b47e8238150cabeb0e9
processed_at: '2026-07-20T18:51:42-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BuPO: Bottom-up Policy Optimization 详解

## 一、Paper 的核心 insight 与 motivation

这篇 paper 的起点是一个非常 mech-interp 风格的观察：当我们做 RLVR (Reinforcement Learning with Verifiable Rewards) 时，传统做法 (PPO, GRPO, DAPO, Reinforce++, RLOO) 都把 LLM 当成一个 **unified black-box policy** $\pi_\theta$，只在最终 vocab distribution $\mathbf{P} = \text{softmax}(\text{LN}(\mathbf{H}^{(2L)})\mathbf{E}_u^\top)$ 上算 loss、做 backprop。整条 residual stream 内部发生了什么、各 layer 各 module 的 contribution 是什么，统统被当成黑盒吞掉。

作者的 claim 是：**通过 residual stream 的 additive decomposition，LLM policy 其实秘密地包含一组 internal policies**，并且这些 internal policies 各司其职，呈现出 universal 的「exploration → convergence」pattern。如果 RL 阶段早期先对 lower-layer 的 internal policy 做一段短程的 optimization，会触发一个叫 **feature refinement** 的现象——lower layers 被迫提前 align 到 high-level reasoning representation，从而为后续的 final policy 提供 stronger foundation。最终的算法叫 **BuPO (Bottom-up Policy Optimization)**，在 Qwen3-4B / 8B、Llama-OctoThinker-3B / 8B-Base 上对 GRPO 等基线都能拿到 +1～+5 个点的 gain。

这个 idea 本质上是把 logit lens (nostalgebraist, 2020) 和 tuned lens (Belrose et al., 2023) 的 interpretability 工具从「分析」工具升级为「优化」对象——logit lens 让我们 decode $\mathbf{H}^l \to$ token；而作者把 $\text{softmax}(\mathbf{H}^l \mathbf{E}_u^\top)$ 当成真正的 samplable policy $\pi_{\text{Layer}}^l$ 去做 PPO/GRPO 风格的 ratio clipping 优化。

相关参考：
- Logit lens: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Tuned lens: https://arxiv.org/abs/2303.08112
- Anthropic 的 Biology of LLM: https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://www.nature.com/articles/s41586-025-09422-z

---

## 二、Internal Policy 的 formalization

### 2.1 Residual stream 的 additive 形式

Transformer 的 hidden state 流动如下 (Eq.1)：

$$
\mathbf{A}^l = \text{MHSA}(\text{LN}(\mathbf{H}^{(2l-2)})), \quad \mathbf{H}^{(2l-1)} = \mathbf{H}^{(2l-2)} + \mathbf{A}^l
$$

$$
\mathbf{F}^l = \text{FFN}(\text{LN}(\mathbf{H}^{(2l-1)})), \quad \mathbf{H}^{(2l)} = \mathbf{H}^{(2l-1)} + \mathbf{F}^l
$$

变量记号：
- $l \in \{1, 2, \ldots, L\}$：layer index
- $\mathbf{H}^{(2l-2)}$：进入第 $l$ 个 layer 之前的 residual stream（来自上一层的 FFN 输出）
- $\mathbf{H}^{(2l-1)}$：self-attention 写入之后的 residual stream
- $\mathbf{H}^{(2l)}$：FFN 写入之后的 residual stream，作为 layer 的 final output（简记为 $\mathbf{H}^l$）
- $\mathbf{A}^l, \mathbf{F}^l$：分别是 attention 和 FFN 的"write"向量，朝 residual stream 中 add
- LN 是 RMSNorm 或 LayerNorm

由于是纯加性 residual，unroll 后（Eq.5）：

$$
\mathbf{H}^l = \mathbf{H}^{(0)} + \sum_{i=1}^{l} \mathbf{A}^i + \sum_{j=1}^{l} \mathbf{F}^j
$$

其中 $\mathbf{H}^{(0)}$ 是 input embedding (token embedding + position embedding)。这一行公式是整篇 paper 的根基——它意味着每个 layer 的 output 都是 **过去所有 attention 和 FFN writes 的累计 sum**，因此把 final layer 拆解为：

$$
\mathbf{H}^L = \underbrace{\mathbf{H}^l}_{\text{intermediate cumulative}} + \underbrace{\mathbf{S}^{l+1}}_{\text{subsequent residual contributions}}
$$

其中 $\mathbf{S}^{l+1} = \sum_{i=l+1}^{L}\mathbf{A}^i + \sum_{j=l+1}^{L}\mathbf{F}^j$。

这个 additive 结构对 Karpathy 来说很自然——这条公式背后的直觉来自 Elhage et al. 的 "A Mathematical Framework for Transformer Circuits" (https://transformer-circuits.pub/2021/framework/index.html)：残差流是 transformer 唯一的「memory bus」，attention 和 FFN 都只是 read+write 到这条 bus 上的小 operator。

### 2.2 Internal Policy 的定义

关键 step：把 unembedding matrix $\mathbf{E}_u \in \mathbb{R}^{N \times d_{\text{model}}}$ 当成「probe」，作用到任意中间 hidden state 上，得到一个 vocab distribution，当作可采样的 policy。

**Internal Layer Policy**（Eq.6）:

$$
\pi_{\text{Layer}}^l \equiv \mathbf{P}_{\text{Layer}}^l = \text{softmax}(\mathbf{H}^l \mathbf{E}_u^\top)
$$

注意与 logit lens 的区别 (Table 4)：logit lens 是 $\text{LN}(\mathbf{H}^l)\mathbf{E}_u^\top$，且目标是 **discrete token decoding**；而 internal policy 的定义是 **直接 softmax**，没有 LN，且把它视为 **samplable policy**——这是能否拿来当 RL optimization 目标的关键差别。

**Internal Modular Policy**（Eq.7）：

$$
\pi_{\text{Module}}^l = 
\begin{cases} 
\mathbf{P}_{\text{ATTN}}^l = \text{softmax}(\mathbf{A}^l \mathbf{E}_u^\top), & \text{for ATTN} \\
\mathbf{P}_{\text{FFN}}^l = \text{softmax}(\mathbf{F}^l \mathbf{E}_u^\top), & \text{for FFN}
\end{cases}
$$

这里 $\mathbf{A}^l$ 和 $\mathbf{F}^l$ 是 single module 的 write 向量，可以独立 decode 成 policy，从而把 attention 和 FFN 的"reasoning contribution"分离开。

### 2.3 Internal Policy Entropy

对每个 internal policy，定义 token-level entropy（Eq.8）：

$$
H_{\text{Layer}}^l = -\sum_{j=1}^{|V|} \mathbf{P}_{\text{Layer},j}^l \cdot \log(\mathbf{P}_{\text{Layer},j}^l)
$$

$|V|$ 是 vocab size（Qwen 大约 151k），$j$ 是 vocab index。这个 $H_{\text{Layer}}^l$ 衡量的是 layer $l$ 在该 token 位置上 "仍有多少候选 tokens 没被 kill 掉"。

引入 **Entropy Change**（Eq.9）来消除 residual 和 LN 引入的 baseline 偏差：

$$
\Delta H^l = H_{\text{Output}}^l - H_{\text{Input}}^l
$$

具体到 module 级别（Eq.13, Eq.14）：

$$
\Delta H_{\text{ATTN}}^l = \mathcal{H}(\mathbf{A}^l) - \mathcal{H}(\text{LN}(\mathbf{H}^{(2l-2)}))
$$

$$
\Delta H_{\text{FFN}}^l = \mathcal{H}(\mathbf{F}^l) - \mathcal{H}(\text{LN}(\mathbf{H}^{(2l-1)}))
$$

直觉：$\Delta H > 0$ 表示这个 module 在 expand exploration space；$\Delta H \approx 0$ 表示 stable integration；$\Delta H < 0$ 表示 convergence（在收窄候选）。

---

## 三、Entropy Analysis 的核心发现

### 3.1 Universal exploration → convergence shift

所有考察的模型（Qwen3-4B/8B/14B、Qwen2.5-Math-7B、Llama-3.1/3.2-Instruct、Llama-OctoThinker-3B/8B-Base、DeepSeek-Math-7B-Base、DeepSeek-R1-Distill-Qwen-7B）在 layer-level entropy 上都呈现一致的 pattern：

- 早期 layer：high entropy，候选 tokens 几乎均匀，模型在"探索"
- 最后几层：entropy $\to 0$，prediction 已 converge 到极少数 candidate

这呼应了 Lindsey et al. 2025 (Anthropic 的 Biology of LLM) 的发现：lower layer 抓 semantic features，higher layer 做 aggregation + sharpening。https://transformer-circuits.pub/2025/attribution-graphs/biology.html

### 3.2 EIC pattern：Qwen vs Llama 的关键差别

最 interesting 的发现来自 **FFN entropy change**：

| Model | FFN entropy pattern |
|---|---|
| **Qwen3 系列** | 三阶段 EIC：layers 1-6 $\Delta H_{\text{FFN}}^l > 0$（exploration expansion），layers 7-26 $\Delta H \approx 0$（integration），layers 27-36 $\Delta H < 0$（convergence） |
| **Llama-3.x** | 几乎所有 layer $\Delta H_{\text{FFN}}^l > 0$，只在最后 1-3 layer 才收敛 → **abrupt convergence** |
| **Qwen2.5-Math** | 整体 negative，更早收敛 |
| **DeepSeek-Math-7B-Base** | 中间 layer FFN 强烈 $\Delta H < 0$，搜索空间被过早压缩 |

EIC = **E**xploration-**I**ntegration-**C**onvergence，类比 Dehaene 的 "global workspace" 人类认知模型 (Dehaene et al., 1998, https://www.pnas.org/doi/abs/10.1073/pnas.95.24.14529)——人类解决 effortful task 时也是先发散 candidate，然后稳定 integrate，最后 sharpen 决策。

这给了一个非常有意思的 architectural insight：**Qwen3 系列在 architectural 层面已经把 "reasoning" 写进 FFN 的 layer-wise behavior 里了**。Llama 把所有 exploration 都堆到 FFN，但收得太晚；DeepSeek-Math 又收得太早；只有 Qwen3 的 6/26/35 三段划分恰好平衡。这或许解释了为什么 Qwen3 在 post-training 阶段 RLVR 吸收效率更高 (Zhu et al., 2025, https://openreview.net/forum?id=ftVlLG9cks; Yang et al., 2025)。

### 3.3 Attention 的 entropy change

ATTN 模块层面：
- Qwen3 系列：$\Delta H_{\text{ATTN}}^l > 0$ 几乎所有 layer，持续 expand exploration space（与 Lindsey et al. 的观察一致——attention 是 "broadening" operator）
- Qwen2.5-Math：$\Delta H_{\text{ATTN}}^l < 0$ 全程，更保守
- Llama：弱正向 trend，介于二者之间

Appendix B.2 给出 residual cosine similarity 分析：
- $\text{cossim}(\mathbf{A}^l, \mathbf{H}^{l-1})$ 衡量 attention write 是否放大原 residual 方向
- Qwen3 的 attention 一致 amplifying residual；Qwen2.5 较弱
- FFN 在 lower layer 写正交 feature（cos≈0）；middle layer 抑制（cos<0，对应 integration）；upper layer 又放大（cos>0，对应 convergence）
- 最后一层都有 sharp directional shift，对应 Gupta et al. 2025 (https://arxiv.org/abs/2510.18871) 关于 final layer 的 "decision layer" 观察

---

## 四、InterGRPO：把 internal policy 当成 RL 目标

### 4.1 公式

标准 GRPO 的目标（Eq.4）：

$$
\mathcal{J}_{\text{GRPO}}(\pi_\theta) = \mathbb{E}_{\mathbf{q}\sim\mathcal{Q}, \{\mathbf{o}_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|\mathbf{q})} \frac{1}{G}\sum_i \frac{1}{|\mathbf{o}_i|}\sum_t \min\left[r_{i,t}\hat{A}_{i,t}, \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right]
$$

其中 $r_{i,t} = \frac{\pi_\theta(o_{i,t}|s_{i,t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|s_{i,t})}$ 是 importance ratio，$\hat{A}_{i,t} = \frac{R_i - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}$ 是 group-relative advantage。

**InterGRPO**（Eq.10）唯一改动：把 ratio 换成 internal layer policy 的 ratio：

$$
\hat{r}_{i,t} = \frac{\pi_{\text{Layer}}^l(o_{i,t}|\mathbf{q}, \mathbf{o}_{i,<t})}{\pi_{\text{Layer,old}}^l(o_{i,t}|\mathbf{q}, \mathbf{o}_{i,<t})}
$$

但 **rollout 仍然来自 $\pi_{\theta_{\text{old}}}$**——这点很重要，意味着行为分布仍是 final policy，只是 surrogate loss 用 internal policy 的 ratio 来算。

### 4.2 Gradient flow 的关键结构（Eq.15-16）

由于 residual stream 的结构，对一个参数 $\theta_k$ 在 layer $k$：

$$
\frac{\partial \mathcal{J}_{\text{InterGRPO}}(\pi_{\text{Layer}}^l)}{\partial \theta_k} = 
\begin{cases} 
\frac{\partial \mathcal{J}}{\partial \pi_{\text{Layer}}^l}\cdot \frac{\partial \pi_{\text{Layer}}^l}{\partial \mathbf{H}^l}\cdot \frac{\partial \mathbf{H}^l}{\partial \theta_k}, & k \le l \\
0, & k > l
\end{cases}
$$

**Gradient 只流回 layer $\le l$ 的参数**——这是 BuPO 的关键！更高 layer 在 internal policy optimization 阶段完全不动。

直觉上：选 layer 6 做 InterGRPO，只有 layer 1-6 的 attention/FFN weights 和 unembedding matrix $\mathbf{E}_u$ 被更新；layer 7-36 的参数保持 frozen。这就是为什么叫 **bottom-up**——只动 foundation，先不动 superstructure。

---

## 五、Feature Refinement 现象（核心 mechanism）

这是 paper 里最 intriguing 的实验。取 Qwen3-4B，选 $\pi_{\text{Layer}}^6$（EIC pattern 的第一段 boundary）做 InterGRPO 50 steps，然后观察：

### 5.1 Hidden state similarity drift

Figure 5(a)：随着 InterGRPO 训练，layer 6 的 hidden state $\mathbf{H}^6$ 与 final layer 的 hidden state $\mathbf{H}^{35}$ 的 **cosine similarity 单调上升**。

直觉上：优化 lower layer 的 output policy 让它 "提前变成 high-layer-like"。本来 layer 6 应该是模糊探索状态，现在被推到 "已经很接近 final answer 的 representation"。

### 5.2 Entropy change 的 collapse

Figure 5(b)：$\Delta H_{\text{Layer}}^6$ 从一开始的正（expansion）逐渐被压到 0 甚至负——意味着 layer 6 不再做 exploration expansion，而是在做早期 convergence。

### 5.3 PPL trade-off（Figure 5(c)）

关键 caution：训练 steps 太多，$\pi_\theta$ 在 base prompt 上的 PPL 反而上升，模型 collapse。这说明 moderate bottom alignment 是关键——作者建议 Qwen3-4B 用 $s_{\text{inter}}=30$ steps 即可。

### 5.4 不同 internal policy 的训练动态（Figure 4）

测试三个 layer policy：
- $\pi_{\text{Layer}}^{35}$（penultimate layer）：entropy 几乎不动 → 与 $\pi_\theta$ 几乎重合，但出现 repetition（response length 爆炸）——因为 final decision 已被它"垄断"
- $\pi_{\text{Layer}}^{26}$（integration region boundary）：entropy unstable，response length 也波动
- $\pi_{\text{Layer}}^6$（exploration region boundary）：entropy 稳定上升，response length 收敛到接近 baseline——**这是 sweet spot**

---

## 六、BuPO 完整算法

Algorithm 1 的 pseudo-code 简化版：

```
for s_cur = 0 to S_max:
    sample q, rollout G outputs {o_i} from π_θ_old
    compute rewards R_i, advantages Â_i,t
    if s_cur ≤ s_inter:                  # Phase 1: Internal Policy Optimization
        π_Layer^l = softmax(H^l @ E_u.T)
        r̂_i,t = π_Layer^l(o_i,t) / π_Layer_old^l(o_i,t)
        loss = -E[min(r̂Â, clip(r̂, 1-ε, 1+ε)Â)]
        # Only layers ≤ l get gradient
    else:                                 # Phase 2: Language Model Policy Optimization
        r_i,t = π_θ(o_i,t) / π_θ_old(o_i,t)
        loss = standard GRPO loss
    update θ
```

**Target layer 选择规则**：选最后一个 $\Delta H_{\text{FFN}}^l > 0$ 的 layer——即 exploration region 的最末 layer。各模型对应：

| Model | Target layer $l$ | $s_{\text{inter}}$ |
|---|---|---|
| Qwen3-4B | 6 | 30 |
| Qwen3-8B | 6 | 20 |
| Llama-OctoThinker-3B-Base | 27 | 20 |
| Llama-OctoThinker-8B-Base | 31 | 20 |

注意 Llama-OctoThinker 的 target layer 比 Qwen 高很多（27 vs 6）——因为 Llama 没有 EIC，需要更靠后的 layer 才有 positive exploration signal。

---

## 七、实验结果

### 7.1 Main table (Table 1)

Avg@K（AIME24/25 K=32，AMC/MATH500 K=16）：

**Qwen3-4B**：
- AIME24: GRPO 32.19 → BuPO 36.88 (+4.69)
- AIME25: GRPO 28.85 → BuPO 31.15 (+2.30)
- Average: 55.08 → 58.51 (+3.43)

**Qwen3-8B**：
- AIME24: GRPO 49.48 → BuPO 54.06 (+4.58)
- Average: 64.23 → 66.36 (+2.13)

**Llama-OctoThinker-3B-Base**：
- Average: 18.58 → 19.59 (+1.01)，gain 较小，因为 Llama 没有 EIC pattern 的加持

**Llama-OctoThinker-8B-Base**：
- Average: 24.11 → 27.79 (+3.68)，gain 反而比 3B 大

### 7.2 Pass@K sweep (Figure 6, Figure 11)

K 从 1 sweep 到 256，n=300 reduce variance：
- Qwen3-8B: BuPO 在所有 K 上都 best
- Qwen3-4B: 只有 K=256 时被 GRPO 略胜（这暗示 BuPO 主要靠提升 Pass@1-Pass@64 区间）
- Llama 系列：BuPO 在所有 K 上都 best，K=256 上 +7.48/+7.93

这个结果对 Karpathy 来说应该很有意思——Pass@K 在大 K 上的提升说明 BuPO 不仅 sharpen 已有能力，还在 expand the set of correct solution paths，这与 entropy 增加（Figure 7）一致。

### 7.3 Ablation (Table 2)

Qwen3-4B 上：

固定 $\pi_{\text{Layer}}^6$，变 $s_{\text{inter}}$：
- $s_{\text{inter}}=30$: Avg 58.51 ✓
- $s_{\text{inter}}=50$: Avg 42.11（塌了 16 个点）
- $s_{\text{inter}}=70$: Avg 9.89（完全 collapse）

**验证了 Figure 5(c) 的 PPL trade-off**：内部 alignment 过头会破坏最终 policy。

固定 $s_{\text{inter}}=30$，变 target layer：
- $\pi_{\text{Layer}}^6 \to \pi_\theta$: 58.51
- $\pi_{\text{Layer}}^{26} \to \pi_\theta$: **59.68** (best!) ← integration boundary 反而更好
- $\pi_{\text{Layer}}^{35} \to \pi_\theta$: 58.34

这个 ablation 很有意思——$\pi^{26}$ 比 $\pi^6$ 还好，说明选 "integration region boundary" 可能比 "exploration region boundary" 更适合。作者留作 future work。

---

## 八、Training Dynamics 分析（Figure 7）

BuPO 的训练早期 entropy **高于** GRPO baseline：
- Qwen3-4B：layer 6 alignment 后，policy entropy 上升再下降
- Qwen3-8B：类似 trend
- Llama-OctoThinker：entropy 也显著增加

这跟 DAPO 论文 (Yu et al., 2025, https://openreview.net/forum?id=2a36EMSSTp) 中"entropy maintenance"的观察呼应——RLVR 训练后期 entropy collapse 是导致 reasoning 能力饱和的主因。BuPO 通过 bottom alignment 主动 inject exploration 到 lower layer，间接维持 final policy 的 exploration budget。

---

## 九、相关的联想和 Karpathy-style intuition

### 9.1 与 Anthropic Attribution Graphs 的关系

Anthropic 的 Biology of LLM (https://transformer-circuits.pub/2025/attribution-graphs/biology.html) 用 sparse autoencoder 把中间 layer 的 features 投影到可解释概念。他们的发现之一：lower layers 集中 broad semantic features，upper layers 集中 specific token features。BuPO 的"feature refinement"现象——layer 6 的 hidden state drift 到越来越像 layer 35——可以理解为：**主动 alignment 强行把 layer 6 的 SAE features 拉向 final-decision features**。这与 Anthropic 的 "circuits are progressively built" 视角一致，只是 BuPO 把这条 progressive chain 在 training time 主动加速了。

### 9.2 与"effective depth"假说的关系

Gupta et al. 2025 (https://arxiv.org/abs/2510.18871) "How do LLMs use their depth?" 提出模型存在 **effective depth**——大部分 layer 可能只是 refinement，真正决策发生在最后几层。BuPO 通过对 lower layer 做 supervision，等于强行把 effective depth 向 lower layer 推。这跟 Hu et al. 2025b (https://openreview.net/forum?id=ILuhAig8xo) "What affects the effective depth of large language models?" 的研究方向一致。

### 9.3 与 Mid-training / OctoThinker 的关系

OctoThinker paper (Wang et al., 2025b, https://arxiv.org/abs/2506.20512) 发现 Llama-3.2-Base RL 后期 improvement 有限，需要 mid-training 才能"激活"RLVR 的潜力。BuPO 实验也证实：Llama-OctoThinker 的 EIC pattern 弱，但 BuPO 仍能带来 gain，说明 bottom alignment 可能在某种程度上 substitute 部分 mid-training 的作用。

### 9.4 与"Zero-step thinking" / Mode selection 的关系

Tan et al. 2025b (https://arxiv.org/abs/2510.19176) 提出"zero-step thinking"——把 mode selection 当作 harder early exit。BuPO 的视角类似但反向：不是让 lower layer 提前 exit，而是让 lower layer 提前"准备好"成为 decision-ready state。

### 9.5 与 RL entropy collapse 文献的关系

- Cui et al. 2025 (https://arxiv.org/abs/2505.22617): RL 训练中 entropy collapse 是现象级问题
- Cheng et al. 2025 (https://arxiv.org/abs/2506.14758): "Reasoning with exploration: An entropy perspective"
- Agarwal et al. 2025 (https://openreview.net/forum?id=UfFTBEsLgI): "unreasonable effectiveness of entropy minimization"——警告 entropy 不是越低越好

BuPO 的 entropy-boosting 效果给这系列工作一个新 angle：从 layer-wise entropy 的 bottom-up 视角 inject exploration。

### 9.6 与"sleeper agents"和"circuit grafting"的潜在联系

如果 layer 6 经过 30 steps InterGRPO 就能 drift 到 "high-level representation"，意味着 transformer 的 lower layer 在 RL 下具有很强的 **feature re-routing capability**。这与 Anthropic 关于 circuit grafting 的实验有交集——supervise lower layer 可以 rewrite 内部 circuit。

### 9.7 与 Gradient Surgery 的关系

只更新 layer $\le l$ 等价于 **structured gradient masking**。这与 PCGrad (Yu et al., 2020)、Gradient Surgery (Ramage et al.) 等 multi-task learning 工作有相似精神，只不过这里不是 task interference 而是 layer-wise interference。或许 BuPO 也可以理解成一种 implicit gradient surgery：阻止 upper layer 的 misleading gradient 在 early training stage 干扰 lower layer foundation 的 build。

### 9.8 与 Curriculum Learning 的隐喻

BuPO 的 schedule 让人联想到 Bengio 的 easy-to-hard curriculum（Bengio et al., 2009）。Curriculum 是 sample 维度的，BuPO 是 **representation depth 维度的**——先在低 layer 这个"simpler hypothesis class"上 align，再转移到 final policy。这个类比给 BuPO 的 effectiveness 提供了一个贝叶斯视角的解释：低 layer policy 是更 restricted function class，先在这里 fit，相当于先做 strong regularization 的预训练，再放开到 full function class。

### 9.9 与 Depth-wise Curriculum in deep RL 的类比

在 deep RL 里 (e.g., schraudolph 1995 epoch-wise decay)，layer-wise learning rate 是一种 implicit curriculum——浅层慢学、深层快学。BuPO 反过来：浅层先学、深层后学。如果 transformer 真是 deep residual stack，浅层 representation 决定了深层 decision 的 manifold shape，那么先 align 浅层就是先稳定 manifold geometry。

### 9.10 与 RLHF/DPO 的关系

DPO (Rafailov et al., 2023, https://openreview.net/forum?id=HPuSIXJaa9) 的 insight：policy 是 implicitly parameterized by reward。BuPO 的 insight 是另一个 angle：policy is implicitly an additive composition of internal policies。能否有对应的 "Direct Internal Policy Optimization"——直接在 pairwise preference 上做 internal layer 的 contrastive loss？这是个开放方向。

### 9.11 与 Activation Patching / Causal Tracing 的关系

Meng et al. 2022 ROME (https://openreview.net/forum?id=-h6WAS6eE4) 用 activation patching 在 single layer 注入 factual association。BuPO 的 InterGRPO 某种程度上是"软版 ROME"——通过 RL signal 修改 layer $\le l$ 的参数，实现"批量知识更新"到 lower layer。这与 Tan et al. 2025a Neural Incompatibility (https://aclanthology.org/2025.acl-long.1047/) 的 cross-scale knowledge transfer 困境形成对照——BuPO 提供了一种可能绕开 incompatibility 的 RL pathway。

### 9.12 与 $\alpha$-DL / Aux-loss 的潜在扩展

DeepSeek-Math 的 auxiliary loss（unbiased throughput）和 DAPO 的 dynamic sampling 都在 sample 层面做 correction。BuPO 在 layer 维度做 correction。或许两者结合（layer × sample）的 2D correction 能给 RLVR 更稳定的 training signal。

---

## 十、Critical assessment / 潜在 weakness

诚实地说，这篇 paper 留下几个值得追问的点：

1. **$s_{\text{inter}}$ 极度敏感**：从 30→50 就崩 16 个点，意味着算法 stability 欠佳。需要更 principled 的 stopping criterion（比如 monitor layer-6 cosine similarity plateau）。

2. **Target layer 选择仍是 ad hoc**：选"最后一个 $\Delta H_{\text{FFN}}^l > 0$"的 layer 是一个 reasonable heuristic，但 ablation 显示 $\pi^{26}$ 比 $\pi^6$ 还好——说明 heuristic 不一定最优。可能需要 multi-stage bottom-up（先 align layer 6，再 align layer 26，最后 final policy）。

3. **Feature refinement 的 mechanism 没有 causal 验证**：只观察到 cosine similarity 上升，没有做 SAE feature attribution 验证 lower layer 真的"捕捉到"高层概念，还是只是 hidden state norm 被拉大。建议后续做 SAE probe 量化。

4. **Llama 的 gain 机制不清楚**：Llama 没有 EIC，target layer 是 27/31，gain 从哪里来？是因为 Llama 的 final-layer abrupt convergence 被 bottom alignment 缓和了吗？需要 entropy dynamics in Llama BuPO training 的可视化。

5. **Pass@256 上 Qwen3-4B 被 GRPO 反超**：暗示 BuPO 可能在 sharpen 前沿 candidate 而非 expand pool。这对 long-horizon exploration 不一定有利。

6. **没评估 OOD generalization**：是否 BuPO overfit 到训练 task 类？test 在 AIME24/25 上 (MATH-style)，没看到 GPQA 或 commonsense benchmark 上是否有 transfer。

7. **Compute overhead 未报告**：InterGRPO 阶段虽然只更新部分参数，但 forward pass 还是 full model，且需要额外 compute internal policy ratio——这步是 cheap 还是 expensive？

---

## 十一、Implementation 细节（Appendix）

### 11.1 Hyperparameters (Table 5)

- Optimizer: AdamW
- Policy LR: 1e-6 (Qwen), critic LR 1e-5 (PPO only)
- Batch: 128 prompts × 8 rollouts
- Mini-batch: 32 prompts, 16 updates per rollout
- Max response: 7168 (Qwen) / 3072 (Llama)
- Rollout temperature 1.0, top_p 1.0
- Clip range $\epsilon = 0.2$
- No entropy regularization, no KL loss (β=0, per Hu et al. 2025a Open-Reasoner-Zero assumption)

### 11.2 Framework

veRL (Sheng et al., 2025, https://dl.acm.org/doi/abs/10.1145/3689031.3696075) for training, vLLM (Kwon et al., 2023, https://dl.acm.org/doi/abs/10.1145/3600006.3613165) for inference.

### 11.3 Datasets

Training: 5k samples from DeepMath-103k (He et al., 2025, https://arxiv.org/abs/2504.11456)
Eval: MATH500, AMC23, AIME24, AIME25
Avg@K with K=16/32, Pass@K with K up to 256, n=300 to reduce variance

### 11.4 Internal Policy Entropy 计算 PyTorch 实现

```python
hidden_state = get_from_hook()
logits = self.model.lm_head(hidden_state)
probs = torch.softmax(logits, dim=-1)
log_probs = torch.log_softmax(logits, dim=-1)
entropies = -(probs * log_probs).sum(dim=-1)
```

### 11.5 Template

Qwen-Math / Qwen3-NoThinking template 用 `<|im_start|>...<|im_end|>` 格式，system prompt 固定 "Please reason step by step, and put your final answer within \boxed{}."

---

## 十二、总结性 intuition

如果让我一句话总结 BuPO 的 story：

> LLM 是个 stack of additive refinement operators；不要把它当 black-box policy，而当成 a cascade of latent policies each with its own entropy profile；在 RL 训练的 early phase，先对 exploration region 的 last layer 做短程 GRPO（gradient 只回 lower layer），让 foundation 先 refine 出 high-level reasoning representation，然后再让 final policy 做 GRPO，会让整体 reasoning 更 robust、entropy 更 healthy、AIME gain +2~+5。

这给了 RL 训练一个 mech-interp-driven 的 algorithmic lever——不是改 reward shaping，不是改 advantage estimation，不是改 KL，而是改 **gradient flow 的 vertical scope**。这是 mech-interp 真正能 drive algorithmic innovation 的一个 concrete 案例，而不是停在"我们发现了 pattern"的 descriptive stage。

后续值得做的方向：
1. Multi-stage bottom-up（按 EIC 三段依次 align）
2. SAE-level verification of feature refinement
3. Cross-task transfer evaluation
4. 理论分析：为什么 short alignment 能 transfer 到 final policy？（information bottleneck 视角？）
5. 与 layer-wise learning rate 的 hybrid
6. 在更大模型 (32B+) 上是否成立
7. 在 non-reasoning task (commonsense, code) 上 EIC pattern 是否还区分 Qwen vs Llama
8. Reward hacking 是否被 mitigate——bottom alignment 让 lower layer 学到 robust features，可能天然 resist reward overfitting
9. 与 chain-of-thought token-level supervision 的结合——CoT 监督 final layer，BuPO 监督 lower layer，正交组合？
10. 在 pre-training 阶段而非 post-training 阶段是否也适用？

对 Karpathy 来说最 resonant 的应该是：这篇 paper 把"logit lens"从解释工具变成优化工具——一个原本只在 mech-interp paper 里见的概念，现在直接进 loss function 拿来做 RL。这跟当年 attention is all you need 把 attention 从 auxiliary alignment 机制（Bahdanau 2014）变成 core architectural primitive 的过程，spirit 上有相似之处。
