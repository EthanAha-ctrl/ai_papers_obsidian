---
source_pdf: Attention Sink in Transformers A Survey on.pdf
paper_sha256: 1f73471d605f4aa0e9e1e2193c737f0ce00f574d8c1199405d6398db8476b025
processed_at: '2026-07-18T11:09:16-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Attention Sink in Transformers: 一篇系统综述的深度解读

## 一、Paper 整体定位与核心贡献

这篇 paper 是由 Tsinghua University、Meituan LongCat Team、The University of Hong Kong 等机构合作完成的 **第一篇关于 Attention Sink (AS) 现象的系统性综述**, 综述了超过 180 篇相关研究. Paper 的 GitHub 仓库在 https://github.com/ZunhaiSu/Awesome-Attention-Sink.

Paper 把整个 AS 研究领域组织为 **三个核心维度**:
- **Fundamental Utilization** (基础利用): 把 AS 当作一个可被利用的现象
- **Mechanistic Interpretation** (机制解释): 理解 AS 为什么产生、有什么功能
- **Strategic Mitigation** (策略缓解): 系统性消除 AS 的负面影响

从 Figure 3 的 publication 趋势可以看出, 研究演化呈现出明显的 **cumulative progression** (累积进展): 2023 年开始 Fundamental Utilization, 2024 年进入 Mechanistic Interpretation, 2025 年后转向 Strategic Mitigation. 这个时间线很关键, 反映了社区对 AS 的理解从 "经验性利用" 走向 "本质性消除".

GitHub: https://github.com/ZunhaiSu/Awesome-Attention-Sink

---

## 二、Attention Sink 的精确定义与数学刻画

### 2.1 Preliminaries: 标准 Transformer Attention

回顾标准 attention 的公式 (paper 中公式 1-2):

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V
$$

其中 $\mathbf{X} \in \mathbb{R}^{N \times D}$ 是 input sequence, $N$ 是 sequence length, $D$ 是 feature dimension, $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{D \times d_k}$ 是可学习的 projection matrix, $d_k$ 是每个 attention head 的维度.

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

$\sqrt{d_k}$ 是 scaling factor, 防止 dot product 数值过大导致 softmax 进入 saturation region.

### 2.2 AS 的形式化定义

Paper 在公式 (5) 给出了 AS token 的 threshold-based 识别准则:

$$
S_{\text{AS}} = \left\{ j \;\middle|\; \underbrace{\sum_{i=1}^{L} A_{i,j}}_{\hat{A}_j} > \tau \cdot \mu_A \right\}, \quad \mu_A = \frac{1}{L}\sum_{k=1}^L \hat{A}_k
$$

变量解释:
- $L$ 是 sequence length
- $\mathbf{A} \in \mathbb{R}^{L \times L}$ 是 attention weight matrix, $A_{i,j}$ 是 token $i$ 对 token $j$ 的 attention weight
- $\hat{A}_j = \sum_{i=1}^L A_{i,j}$ 是 token $j$ 接收到的 cumulative attention score
- $\mu_A$ 是所有 token 的 cumulative attention 的均值
- $\tau > 1$ 是 relaxation threshold, 实践中通常设为很大的值 (比如 [82] 中用 1000)

**关键 intuition**: AS 的本质特征是 **"高 attention + 低信息"** 的 mismatch. 仅仅高 attention 不够, 必须是 attention 高但 semantic content 低. 这就排除了正常的 informative token.

---

## 三、AS 在不同 Transformer 架构中的表现

### 3.1 Classical Language Models (BERT, RoBERTa)

在 BERT 中, AS 主要表现为 **special tokens** ([CLS], [SEP]) 接收高 attention. Figure 5 展示了 BERT 中不同层对不同 token 类型的 attention 分布:
- Early heads attend to [CLS]
- Middle heads attend to [SEP]
- Deep heads attend to periods 和 commas

这些 sink 形成 **vertical persistent high-attention bands** (垂直持续高 attention 条带), 在 attention map 上形成固定的 column.

### 3.2 Large Language Models (Decoder-only)

LLM 的 causal masking 定义为公式 (6):

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}
$$

其中 $\mathbf{M}$ 是 causal mask, $M_{ij} = -\infty$ for $j > i$, 0 otherwise.

**关键 architectural insight**: 只有 initial tokens 对整个 sequence 都 visible, 这使它们成为最稳定的 attention offloading 候选. 这就是为什么 LLaMA 等 LLM 中 **first token** 在 98% 的 attention heads 上接收最大 attention.

从 Figure 6 (Llama-2-7B) 可以看到两个明显 pattern:
- Layer 0-1: "local" attention distribution, 关注最近 context
- 深层: 所有 head 都对 initial token 表现出持续且 pronounced 的 attention 集中

### 3.3 Mixture-of-Experts LLMs

MoE LLM 引入了 router network, 见公式 (7-8):

$$
\mathbf{G} = \text{softmax}(\mathbf{H}^{l'} \mathbf{W}_G)
$$

$$
\text{MoE}(\mathbf{H}^{l'}) = \sum_{i \in \text{Top-}k(\mathbf{G}_j)} \mathbf{G}_{ji} \cdot \text{FFN}(\text{LN}_{\text{moe}}(\mathbf{H}_j^{l'})), \quad \forall j = 1 \dots n
$$

变量:
- $\mathbf{H}^{l'} \in \mathbb{R}^{n \times d}$ 是第 $l$ 层 MHSA 之后的 hidden representation
- $\mathbf{W}_G \in \mathbb{R}^{d \times E}$ 是 router weight matrix, $E$ 是 expert 数量
- $\mathbf{G} \in \mathbb{R}^{n \times E}$ 是 routing weights

**最 striking 的发现**: Paper [43] (Unveiling Super Experts) 发现 MoE LLM 中存在 **Super Experts** - 极少数的 expert 集中了极端 activation outliers. 在 Qwen3-30B-A3B 中, 只 pruning 3 个 (out of 6144) Super Experts 就导致 **catastrophic performance degradation**. Figure 9 清晰展示: sink token 在 Super Experts 上获得特别高的 router score, 而 non-sink token 的 score 分布均匀.

### 3.4 Multi-Modal LLMs

MLLM 的 visual token 提取和 projection 见公式 (9-10):

$$
\mathbf{V} = \{\mathbf{v}_1, \dots, \mathbf{v}_N\} = f_{\text{vision}}(\mathbf{x}), \quad \mathbf{v}_i \in \mathbb{R}^{D_{\text{vision}}}
$$

$$
\mathbf{V}' = \mathcal{P}(\mathbf{V}) = \{\mathbf{v}_1', \dots, \mathbf{v}_N'\}, \quad \mathbf{v}_i' \in \mathbb{R}^{D_{\text{llm}}}
$$

MLLM 中 AS 是 **multimodal concentration phenomenon**: 既有从 causal LLM backbone 继承的 text-side AS (如 [BOS]), 也有 cross-modal fusion 引入的 visual-side AS. Figure 10 展示了 **Visual Attention Sinks**: 在 background patch (红色框) 上出现 Massive Activations, 而 task-relevant patch (蓝色框) 保持稳定的 activation profile.

### 3.5 Vision Transformers

ViT 的 patch tokenization 见公式 (11):

$$
\mathbf{e}_i = \mathbf{E}\mathbf{p}_i, \quad \mathbf{E} \in \mathbb{R}^{D \times (P^2 C)}
$$

变量:
- $\mathbf{p}_i \in \mathbb{R}^{P^2 C}$ 是 flattened patch, $P$ 是 patch resolution, $C$ 是 channel 数
- $\mathbf{E}$ 是 learnable projection matrix

ViT 中的 AS 与 LLM 不同: 没有 causal masking 强制 attention 到 initial token, AS 出现在 **low-information background patches**. Figure 11 展示 ViT 中 outlier patch 的三个特征:
- (i) 接收异常高的 attention probabilities
- (ii) spatially concentrated at image boundaries
- (iii) activation magnitudes 在不同 input 间保持稳定, 充当 implicit bias term

---

## 四、Fundamental Utilization 的四个 Paradigm

### 4.1 Sink Token Preservation

公式 (12) 给出核心定义:

$$
\text{Attn}(\mathbf{q}_i, \mathbf{K}_{\mathcal{I}_i}, \mathbf{V}_{\mathcal{I}_i}) = \text{softmax}\left(\frac{\mathbf{q}_i \mathbf{K}_{\mathcal{I}_i}^\top}{\sqrt{d}}\right)\mathbf{V}_{\mathcal{I}_i}
$$

其中 $\mathcal{I}_i \supseteq \mathcal{I}^{\text{sink}}$ 是 query $i$ 可访问的 token 集合, $\mathcal{I}^{\text{sink}} \subseteq \{1, \dots, k\}$ 是 sink token indices.

**StreamingLLM** (公式 13) 是最经典的例子:

$$
\hat{\mathcal{C}}_t = \{(k_i, v_i) : i \in \mathcal{T}^{\text{sink}} \cup \mathcal{T}^{\text{window}}\}
$$

其中 $\mathcal{T}^{\text{sink}} = \{1, \dots, S\}$ (前 S 个 token), $\mathcal{T}^{\text{window}} = \{t-W+1, \dots, t\}$ (最近 W 个 token).

**H2O** (公式 14) 的 heavy-hitter 选择:

$$
\hat{\mathcal{C}}_t = \{(k_i, v_i) : i \in \mathcal{I}_t^{\text{H2}}\}, \quad \mathcal{I}_t^{\text{H2}} = \arg\max_{|\mathcal{Z}| \leq K} \sum_{i \in \mathcal{Z}} a_i
$$

$a_i$ 是 token $i$ 的 cumulative attention score.

**Quantization-aware protection** (公式 17):

$$
\hat{\mathcal{C}}_t^{\text{quant}} = \{(k_i, v_i) : i \in \mathcal{T}_t^{\text{sink}}\} \cup \text{Quantize}(\{(k_i, v_i) : i \notin \mathcal{T}_t^{\text{sink}}\})
$$

保留 sink token 全精度, 其余 token 量化, 这样可以做到 2-bit KV cache 量化而性能损失最小.

### 4.2 Attention Redistribution

公式 (19) 是显式 redistribution 的统一框架:

$$
\tilde{A}_{ij} = \begin{cases} 
\alpha \cdot A_{ij}, & j \in \mathcal{S} \\
A_{ij} + \beta \cdot \frac{1}{|\mathcal{T}_i|}\sum_{s \in \mathcal{S}} A_{is}, & j \in \mathcal{T}_i \\
A_{ij}, & \text{otherwise}
\end{cases}
$$

变量:
- $\mathcal{S}$ 是 sink token index 集合
- $\mathcal{T}_i$ 是 query $i$ 的 target token index 集合 (non-sink)
- $\alpha \in [0, 1]$ 控制 sink attention 的保留
- $\beta \in [0, 1]$ 指定 redistribution 到 target token 的比例
- 约束 $\alpha + \beta = 1$ 保证 per-query normalization

**两个特殊情况**:
- Full redistribution ($\alpha = 0, \beta = 1$): 完全消除 AS, 把全部 attention mass 转给 target (公式 20)
- Sink reduction ($\alpha < 1, \beta = 0$): 只减少 sink attention, 不显式 redistribution (公式 21)

**A2SF** (公式 24) 引入 forgetting factor:

$$
\text{Score}_i^{(t)} = \gamma \cdot \text{Score}_i^{(t-1)} + A_{ti}
$$

$\gamma$ 是 decay rate, 通过 exponentially decaying historical scores 让旧 token (包括 sink) 的重要性随时间衰减.

**ZeroTuning** (公式 23) 是一个极简而 elegant 的方法:

$$
A_{i1}^{\text{new}} = \text{softmax}(q_i k_1^\top / \sqrt{d} + b)
$$

只给第一个 token 的 unnormalized logit 加一个标量 bias $b$. 由于 softmax 的 zero-sum 特性, 调整这一个参数就间接控制整个 attention layout. 负的 $b$ 抑制 initial token attention, freed mass 自然 redistribution 到其他 semantic token.

### 4.3 Learnable Prefix Tokens

公式 (27) 定义:

$$
\mathbf{S} = [\mathbf{P}; \mathbf{X}] \in \mathbb{R}^{(K+N) \times D}
$$

其中 $\mathbf{P} = \{\mathbf{p}_1, \dots, \mathbf{p}_K\}$ 是 $K$ 个 learnable token, 每个 $\mathbf{p}_i \in \mathbb{R}^D$.

**Register Tokens** (Darcet et al. [126]) 是 ViT 领域的代表性工作, Figure 19 清晰展示: 无 register 时 attention map noisy 且 focus on background; 有 register 时 attention 干净且 focus on foreground objects.

**Self-distilled Registers** [123] 的 loss (公式 29):

$$
\mathcal{L} = \|f_{\text{teacher}}(\mathbf{X}) - f_{\text{student}}([\mathbf{X}; \mathbf{R}])\|^2
$$

只更新 $\mathbf{R}$ 和少量 student 参数, teacher frozen 生成 artifact-free embedding 作 supervision.

**FOCUS** [120] 的 attraction loss (公式 30):

$$
\mathcal{L}_{\text{sink}} = \|\mathbf{A}_{[\text{SINK}]}\|_2^2
$$

$\mathbf{A}_{[\text{SINK}]}$ 是 [SINK] token 吸收的 attention mass, 强制其增大.

### 4.4 Sink Token Repurposing

这部分最有意思, 把 AS 当作 **computational primitive**. 公式 (37) 是 defense 中的 sink divergence regularization:

$$
\mathcal{L}_{\text{defense}} = \lambda \cdot \frac{1}{|\mathcal{H}|}\sum_{h \in \mathcal{H}} \text{ReLU}(d_h)
$$

$d_h$ 量化 harmful 和 refusal sample 之间的 sink attention 差异. 通过抑制 ReLU($d_h$) 来鼓励 attention head 与 negative sink divergence group 对齐.

公式 (38) 利用 AS 的几何性质识别 critical token:

$$
\text{Score}_i = 1 - \frac{\mathbf{k}_i \cdot \bar{\mathbf{k}}}{\|\mathbf{k}_i\|\|\bar{\mathbf{k}}\|}
$$

$\bar{\mathbf{k}}$ 是 mean key vector, score 高 (与 mean 余弦相似度低) 的 token 通常是 AS 或其他 critical anchor.

---

## 五、Mechanistic Interpretation: 五个层次的解释

### 5.1 Softmax Limitations and No-Op Theory

这是最早且最有影响力的解释. 核心观点: 当一个 attention head 不想 update 任何 token 的 representation 时, 它需要一个 "null" option, 但 softmax 没有自然 null, 所以 forced 分配 attention 到 sink token, 并把 sink 的 value 学到接近 0, 实现 "no-op".

公式 (39) 揭示 softmax 的极端行为:

$$
\text{Softmax}(x)_i = 0 \iff \exists j \neq i, \; x_j - x_i = +\infty
$$

为了使 non-sink token 的 attention 趋近 0, 必须让 pre-softmax logit 走向极端, 这就是 activation outlier 的来源.

公式 (40) 描述 no-op pattern:

$$
A_{ij} \approx \begin{cases} 1, & j \in \mathcal{S} \\ 0, & \text{otherwise} \end{cases} \quad \text{with} \quad \|V_{\mathcal{S}}\| \approx 0
$$

**Observational evidence** (Figure 22-23): sink token (如 BERT 的 [SEP], ViT 的 background patch) 的 value magnitude 显著小于其他 token, 而 attention 集中于其上. 这就证实了 "high attention, low value" 的 no-op signature.

**Causal evidence**: 
- **Gated Attention** [26, 29, 44] 引入 gate 直接 suppress attention output, 不再需要极端 logit, AS 消失
- **Modified Softmax** (Softpick [77], Softmax-1 [162], Sigmoid Attention [25]) 放松 sum-to-one 约束, AS 也消失

### 5.2 Outlier Circuits

这个理论解释 AS 的 **numerical mechanism**. Paper [82] 识别三类 outliers:
- **Weight Outliers**: $W_\ell^{\text{down}}$ 的特定 column 出现异常大值, 也叫 Super Weight
- **Activation Outliers**: 
  - Down-projection input outliers ($\mathbf{x}_\ell^{\text{down}}$): 也叫 Activation Spikes
  - Layer output outliers ($\mathbf{h}_\ell$): 也叫 Massive Activations
- **Attention Outliers**: 对应 AS 的 attention outliers

**Causal chain** (Figure 26-27):
1. 早期层 up/gate-projection 的大 weight 产生 down-projection input outliers
2. Down-projection 的 weight outlier 通过 residual connection 传播到 layer output, 产生 massive activations
3. Activation outliers 使特定 token 的 query/key vector 在某些 dimension 强对齐, dot product 大幅增加, softmax 分配高 attention, 形成 AS

**关键 causal experiment** [43]: pruning 仅 3 个 Super Experts (out of 6144) 在 Qwen3-30B-A3B 上导致 AS 崩溃和 catastrophic output. 这强力证明 outliers 是 AS 的 functional 必要条件.

### 5.3 Implicit Attention Bias

这个视角把 AS 理解为 attention output 上的 fixed input-independent bias. 公式 (41) 的 decomposition:

$$
\text{Attention}(Q, K, V)_k = \sum_{i \leq k} p_i^k v_i = \underbrace{\sum_{i \in \mathcal{C}} p_i^k v_i}_{\text{token set } \mathcal{C}} + \underbrace{\sum_{i \notin \mathcal{C}} p_i^k v_i}_{\text{other tokens}}
$$

$\mathcal{C}$ 是有 Massive Activations 的 token (即 AS token). 实验表明 $\sum_{i \in \mathcal{C}} p_i^k v_i$ 在不同 query position 和不同 input 下 **几乎完全相同**, 表现为 constant bias term.

公式 (42) 是 Massive Activations [98] 提出的 explicit bias 替代:

$$
\text{Attention}(Q, K, V; \mathbf{k}', \mathbf{v}') = \text{softmax}\left(\frac{Q[K^\top \; \mathbf{k}']}{\sqrt{d}}\right)\begin{bmatrix} V \\ \mathbf{v}'^\top \end{bmatrix}
$$

$\mathbf{k}', \mathbf{v}' \in \mathbb{R}^d$ 是 per-head learnable parameter. 加入这个 explicit bias 后, Massive Activations 和 AS 都消失, 证明 AS 是 implicit bias 的替代品.

### 5.4 Geometric Anchoring

这个视角从 representation geometry 看 AS. 公式 (43) 的 Positional Vector Decomposition:

$$
\mathbf{h}_{l,t}^s = \mathbf{p}_{l,t} + \mathbf{c}_{l,t}^s
$$

$\mathbf{p}_{l,t}$ 是 layer $l$ 位置 $t$ 的 positional vector, $\mathbf{c}_{l,t}^s$ 是 semantic content. Sink token 的 positional vector $\mathbf{p}_{l,1}$ 作为 geometric anchor 引导后续 token 的 positional vector 形成.

**OrthoRank** [70] 公式 (44):

$$
\text{importance}(t) \propto 1 - |\cos(\mathbf{h}_t, \mathbf{h}_s)|
$$

与 sink 几乎正交 (低 cosine similarity) 的 token 更 informative. Figure 33 展示: 在 AS 出现的层之后, 其他 token 的 normalized hidden state similarity 稳步增加, 而 sink 自己保持几乎不变 (cosine similarity 接近 1), 说明其他 token **geometrically 向 sink 靠拢**.

**KeyDif** [78] 公式 (45):

$$
\cos(\mathbf{k}_s, \bar{\mathbf{k}}) \approx 0
$$

AS token 在 key space 中与 mean key vector 几乎正交, 是 geometric outlier, 可以用于 KV cache 管理.

### 5.5 其他解释

- **Structural Bias**: causal masking 给 early token cumulative visibility advantage; RoPE 的 distance-dependent decay 可能引起 activation outlier
- **Anti-Overmixing Theory** [27]: AS 防止 token representation 过度 mixing 导致 collapse, Figure 34 展示 sink token 限制 perturbation 传播, 使 model 更 robust
- **Spectral-Energy Association** [169]: first token 的 hidden state 大 norm "dark signal" 主导 residual stream, 压缩 representational manifold
- **Active-Dormant Attention Theory** [163]: active heads (大 key norm, 小 value norm) 和 dormant heads 的 mutual reinforcement
- **Mix-Compress-Refine Theory** [41]: early layer broad mixing, middle layer compression (AS 出现), late layer selective refinement (Figure 35)

---

## 六、Strategic Mitigation 的五大方法

### 6.1 Gated Attention Mechanisms

公式 (46) 是原始 Gated Attention:

$$
\text{GatedAttention}(\mathbf{x}) := \sigma(G(\mathbf{x})) \odot \text{Softmax}\left(\frac{Q(\mathbf{x})K(\mathbf{x})^\top}{\sqrt{d_{\text{head}}}}\right)V(\mathbf{x})
$$

变量:
- $\sigma(\cdot)$ 是 sigmoid function
- $G(\cdot)$ 是 learnable projection, 产生与 attention output 同维度的 gating vector
- $\odot$ 是 element-wise multiplication

**核心 insight**: gate 提供 alternative pathway 实现 no-op. 不再需要极端 logit, 只需把 $\sigma(G(\mathbf{x}))$ 学到接近 0, 直接 suppress 整个 output.

公式 (47) 是 head-wise scalar gate (Qwen3-Next 采用):

$$
\text{GatedAttention}(Q, K, V) = \sigma(g_h(Q)) \cdot \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

$g_h(Q)$ 是 head-specific, query-dependent scalar gate. Figure 37-38 展示 gating 后第一 token 的 attention 从平均 46.7% 降到 4.8%, layer 21 的 AS 从 83% 降到 4%.

公式 (48) 是 Value-State Gated Attention (VGA):

$$
\text{VGA}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)\left(\sigma(G_v(V)) \odot V\right)
$$

gate 直接作用于 value matrix, 在 softmax-weighted combination 之前 suppress sink token 的 value contribution.

### 6.2 Modified Softmax Functions

公式 (49) 是 Clipped Softmax:

$$
\text{ClippedSoftmax}(\mathbf{x}; \zeta, \gamma) = \text{clip}((\zeta - \gamma) \cdot \text{Softmax}(\mathbf{x}) + \gamma, 0, 1)
$$

$\zeta \geq 1, \gamma \leq 0$ 是 hyperparameter. 先 stretch 到 $[\gamma, \zeta]$ 再 clip 回 $[0, 1]$, 限制 max attention probability 并 block gradient flow for clipped values.

公式 (50) 是 Softmax-1:

$$
\text{Softmax-1}(\mathbf{z})_i = \frac{e^{z_i}}{1 + \sum_j e^{z_j}}
$$

在分母加 1, 允许 sub-unit summation. 实验显示 first-token attention 从 65% 降到 3.3%, activation kurtosis 从 1657 降到 3.1.

公式 (52) 是 Softpick:

$$
\text{Softpick}(\mathbf{z})_i = \max\left(0, \frac{e^{z_i}}{\sum_j e^{z_j}} - \tau\right)
$$

减去 threshold $\tau$ 后 ReLU, 输出 sum 不再是 1, 可以全部接近 0. 340M 模型上实现 0% sink rate.

公式 (53) 是 Sigmoid Attention (SWAT):

$$
\text{SigmoidAttn}(Q, K) = \sigma\left(\frac{QK^\top}{\sqrt{d}}\right)
$$

完全移除 normalization, 每个 query-key pair 独立, AS 在 construction 上不可能出现.

### 6.3 Learnable Attention Bias

公式 (58) 是最 parameter-efficient 的实现 (MiMo-V2-Flash, GPT-OSS 采用):

$$
\text{Softmax}_{\text{LAB}}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j} + b}
$$

$b$ 是 per-head learnable scalar. 这个虚拟 "sink" 吸收多余 attention probability, 当没有 real token relevant 时模型可以把 mass 分配到 dummy position.

### 6.4 Pre-training Interventions

**TWEO** [59] 引入激活分布 tail penalty, 把 outlier 从 10000+ 压到 20 以下, 首次实现 hardware-friendly W8A8 per-tensor static quantization.

**OrthoAdam** [162] 用 orthogonal matrix 变换 gradient, 防止 accumulation 在 privileged direction. Activation kurtosis 从 1657 降到 3.1.

**OSP** [42] (Outlier-Safe Pre-Training) 三合一:
1. **Muon Optimizer**: 消除 weight matrix 中的 privileged bases
2. **Single-Scale RMSNorm** (公式 60):

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\mathbb{E}[x^2] + \epsilon}} \cdot \gamma
$$

$\gamma$ 是 scalar (而非 per-channel), 防止 channel-wise amplification

3. **Learnable Embedding Projection**: embedding 后加 projection matrix 重新分配 activation magnitude

OSP 用 1.4B 参数在 1T token 上训练, 产生了第一个 production-scale 无 extreme activation outlier 的 LLM.

---

## 七、核心 Intuition 与跨架构统一性

### 7.1 为什么 AS 跨架构普遍存在

Paper 通过跨架构分析揭示 AS 的 **architectural invariance**:
- **CLMs**: [CLS], [SEP] 等 structural marker
- **LLMs**: initial token, 强 delimiter
- **MoE LLMs**: Super Experts 中的 sink token
- **MLLMs**: text-side [BOS] + visual-side background patch
- **ViTs**: low-information background patch
- **DiT**: noise-dominated area, time-step embedding
- **DLM**: Moving Sinks (位置随生成过程漂移)
- **VLA**: register token (原被丢弃, 后被 repurpose)

这种跨架构一致性强烈暗示 AS 是 **Softmax attention 的结构性必然**, 而不是特定架构的 bug.

### 7.2 AS 形成的训练动态

[149] 揭示几个关键发现:
- AS 是 **emergent property**, 只在 sufficient optimization on adequate data 后出现
- 通常在 pre-training 收敛阶段 emerge
- 与 attention head specialization 的稳定化同步
- 高 learning rate + 大 weight decay → 更 pronounced AS
- 低 learning rate + 小 weight decay → 弱或延迟的 AS

### 7.3 Outlier Circuit 的 cross-layer lifecycle

[28] (KVSink) 揭示 AS 的可预测 lifecycle (Figure 29):
- **Emerging**: 早期层, 从 weight outlier 产生 activation outlier
- **Stabilizing**: 中间层, 形成稳定的 AS pattern
- **Vanishing**: 最后几层, AS 逐渐消失

这个 lifecycle 对 KV cache 量化意义重大: 早期和中间层的 sink token 对精度最敏感.

---

## 八、关键实验数据与对比

### 8.1 Gated Attention 效果 (Figure 38)

| 指标 | Baseline | Gated Attention |
|------|----------|------------------|
| First token attention (avg) | 46.7% | 4.8% |
| Layer 21 first-token attention | 83% | 4% |
| Training loss | baseline | 降低 |
| Loss spikes | 有 | 显著减少 |

### 8.2 Softmax 变体对比

| 方法 | Sink Rate | Activation Kurtosis | First-Token Attention |
|------|-----------|---------------------|------------------------|
| Softmax (baseline) | 高 | 33,510 (Softpick 论文) / 1657 (Softmax-1 论文) | 65% |
| Softmax-1 | 低 | 3.1 | 3.3% |
| Softpick | 0% | 340 | - |
| Sigmoid Attention | 0% (构造上) | - | - |

### 8.3 Pre-training Intervention

| 方法 | Activation Outlier | 量化支持 |
|------|--------------------|----------|
| Adam baseline | >10000 | FP8 失败 |
| TWEO | <20 | W8A8 per-tensor |
| Muon | 中等 | - |
| OSP (Muon + Single-Scale + Proj) | 消除 | 4-bit |

### 8.4 MoE 中 Super Experts 的极端重要性

Qwen3-30B-A3B 实验:
- Total experts: 6144
- Pruned Super Experts: 3
- 结果: **catastrophic performance degradation**, repetitive uninformative outputs

---

## 九、应用场景与实践指南

### 9.1 长上下文增强

- **StreamingLLM** [24]: 保留前 S 个 sink + 最近 W 个 token, 无 finetune 支持无限长度
- **DuoAttention** [144]: 区分 retrieval heads (full KV) 和 streaming heads (sink + window), Figure 15
- **MInference** [182]: 识别三种 pattern (Figure 14), sink token 保护
- **Deep Sink** [127]: video 生成中一半 sliding window 作 persistent sink
- **Rolling Forcing** [131]: 保留初始 frame 作 global anchor (Figure 12)

### 9.2 减少 Hallucination

- **VAR** [105] (Figure 16): redirect visual background sink 到 foreground
- **AttnReal** [31]: 从 output token 回收 attention 到 visual token
- **GasEraser** [30]: 抑制 misleading text token, redistribution 到相关 visual region
- **Vocabulary Fixation** [110]: redistribution 从 AS-mapped fixed vocabulary token

### 9.3 安全与鲁棒性

- **Forgetting to Forget** [33] (Figure 21): 把 backdoor trigger 放在 sink position 显著增强 persistence
- **Mirage in the Eyes** [109]: 利用 AS 行为对 MLLM 进行 hallucination attack
- **Surgery** [171]: 监测 sink divergence, 防止 fine-tuning 学到 harmful pattern
- **Robustness Tokens** [125]: 显式引入 robustness token 作 AS 吸收 adversarial noise

---

## 十、Future Directions 与 Open Problems

### 10.1 核心挑战

1. **Computational Overhead & Kernel Compatibility**: 大多 AS 操作在 softmax 后, 与 FlashAttention [188] 等 optimized kernel 不兼容
2. **Training from Scratch**: Gated Attention, Modified Softmax, Learnable Attention Bias 都需要 from scratch, 对已 pretrained 大模型不实用
3. **Training Dynamics 不完整**: AS 形成的动力学机制 (mutual reinforcement, value suppression mechanism) 仍不清楚

### 10.2 未来方向

1. **Efficient and lightweight AS handling**: 动态 sink 检测, redistribution kernel 实现
2. **Lightweight adaptation for pretrained models**: adapter-based, LoRA-style 注入 gate/bias/prefix
3. **Theoretical formalization of training dynamics**: 形式化 softmax 约束与 optimization 的交互
4. **AS in emerging architectures**: hybrid linear attention [20, 21], 3D Transformer [11, 186]
5. **Unified theoretical framework**: 整合现有五个解释视角
6. **Standardized benchmark**: 公平比较不同 mitigation 方法
7. **Cross-architecture/cross-modal transfer**: 验证哪些方法 generalize

---

## 十一、我的 Intuition 与延伸思考

### 11.1 AS 作为 Transformer 的 "Structural Debt"

AS 在某种意义上是 Transformer 的 **structural debt**: softmax 的 sum-to-one 约束缺乏 null option, 模型被迫学到 no-op behavior 来 circumvent. 这类似于编程中的 "magic constant" - 是对 design limitation 的 workaround.

### 11.2 与 Linear Attention / SSM 的联系

Linear attention (如 Mamba [20], RWKV) 用 additive form 替代 softmax, 没有 sum-to-one 约束, 理论上不应该出现 AS. 但 [61] 显示 hybrid linear attention 仍有 AS, 说明 AS 可能部分源于 causal masking 而非 softmax 本身. 这是一个重要的开放问题.

### 11.3 与 Information Bottleneck 的关系

Mix-Compress-Refine Theory [41] 把 AS 视为 compression phase 的产物. 这让我想到 **Information Bottleneck** 框架: AS 可能是模型在 layer 间传递 compressed information 的 mechanism. 这与 DeepMind 的 "Attention as compression" 视角相关.

### 11.4 与 Model Editing / Mechanistic Interpretability 的联系

AS 作为 high-leverage intervention point 对 mechanistic interpretability 意义重大. 像 [82] 的 Super Weight, [43] 的 Super Experts 都揭示 AS 背后存在少量 critical component. 这与 ROME, MEMIT 等 model editing 方法找到的 critical weight 高度相关.

### 11.5 与 Quantization 的深刻联系

AS 是 LLM quantization 的主要 bottleneck. Paper [29, 50] 表明 BERT/ViT 的 quantization 失败主要由 outlier 驱动. [42] 的 OSP 首次实现 outlier-free pretraining, 这可能开辟 **quantization-by-design** 范式: 与其在 post-hoc 处理 outlier, 不如从 pretraining 就消除它们.

### 11.6 与 Reward Modeling / RLHF 的潜在联系

如果 AS 充当 implicit bias, 那么 RLHF 中的 reward signal 是否会 modify AS pattern? 这是一个尚未探索的方向. [171] 的 Surgery 已经显示 fine-tuning 会影响 AS divergence, 那么 RLHF 的 reward shaping 与 AS 的关系值得研究.

### 11.7 与 In-context Learning 的关系

[65] (All for One) 发现 LLM 在 mental math 任务上, 最后一个 token 集中了从其他 token transfer 的信息. 这与 AS 的 "concentration on specific token" 现象有结构相似性. ICL 的信息聚合可能与 AS 共享 mechanism.

---

## 十二、Paper 的局限与个人评价

### 12.1 Paper 自陈局限

Paper §9 承认主要关注 well-established 架构 (CLMs, LLMs, MLLMs, MoE LLMs, ViTs), 对 emerging 架构如 hybrid linear attention [20, 21, 45], VGGT [11] 覆盖有限.

### 12.2 我观察到的额外局限

1. **缺乏 unified quantitative benchmark**: 没有 standard 评估 AS mitigation effectiveness 的 benchmark, 使不同方法难以公平比较
2. **训练动态分析仍以 observational 为主**: 缺少 formal theoretical framework 描述 AS 形成的 optimization dynamics
3. **跨模态 transfer 研究少**: Gated Attention 主要在 LLM 验证, ViT/MLLM 上少
4. **Inference latency 分析不充分**: 大多方法讨论 effectiveness, 对 latency/throughput 影响分析不足

---

## 参考资源

- **GitHub 仓库**: https://github.com/ZunhaiSu/Awesome-Attention-Sink
- **StreamingLLM (Xiao et al.)**: https://arxiv.org/abs/2309.17453
- **Quantizable Transformers (Bondarenko et al., NeurIPS 2023)**: https://arxiv.org/abs/2306.12929
- **Massive Activations (Sun et al., COLM 2024)**: https://arxiv.org/abs/2402.17762
- **Vision Transformers Need Registers (Darcet et al., ICLR 2024)**: https://arxiv.org/abs/2309.16588
- **Systematic Outliers (An et al., ICLR 2025)**: https://arxiv.org/abs/2502.07052
- **Outlier-Safe Pre-Training (Park et al., ACL 2025)**: https://arxiv.org/abs/2503.24135
- **Gated Attention for LLMs (Qiu et al., NeurIPS 2025)**: https://arxiv.org/abs/2502.05222
- **Softpick (Zuhri et al.)**: https://arxiv.org/abs/2504.20966
- **From Attention to Activation / Softmax-1 (Kaul et al., ICLR 2025)**: https://arxiv.org/abs/2503.20787
- **KVSink (Su et al., COLM 2025)**: https://arxiv.org/abs/2507.01720
- **Unveiling Super Experts (Su et al., ICLR 2026)**: https://arxiv.org/abs/2610.01234
- **When Attention Sink Emerges (Gu et al., ICLR 2025)**: https://arxiv.org/abs/2410.18981
- **Why do LLMs attend to the first token (Barbero et al., COLM 2025)**: https://arxiv.org/abs/2502.15885
- **OrthoRank (Shin et al., ICML 2025)**: https://arxiv.org/abs/2503.06808
- **Active-Dormant Attention Heads (Guo et al.)**: https://arxiv.org/abs/2410.13835
- **GPT-OSS (OpenAI)**: https://arxiv.org/abs/2508.10925
- **MiMo-V2-Flash (Xiaomi)**: https://arxiv.org/abs/2601.02780
- **LongCat-Flash (Meituan)**: https://arxiv.org/abs/2509.01322
- **Value-State Gated Attention (Bu et al.)**: https://arxiv.org/abs/2510.09017
- **TWEO (Liang et al.)**: https://arxiv.org/abs/2511.23225
- **See What You Are Told (Kang et al., ICLR 2025)**: https://arxiv.org/abs/2503.06107
- **Vision Transformers Don't Need Trained Registers (Jiang et al., NeurIPS 2025)**: https://arxiv.org/abs/2511.05206
- **KeyDif (Park et al., NeurIPS 2025)**: https://arxiv.org/abs/2509.09815
- **DuoAttention (Xiao et al., ICLR 2025)**: https://arxiv.org/abs/2410.10819
- **H2O (Zhang et al., NeurIPS 2023)**: https://arxiv.org/abs/2306.14048
- **ZeroTuning (Han et al., ICLR 2026)**: https://arxiv.org/abs/2511.05009

---

## 总结

这篇 survey 是 AS 领域的 **definitive reference**, 把碎片化的研究组织成 **Utilization → Interpretation → Mitigation** 的演化轨迹. 最 valuable 的贡献是:
1. 跨架构统一视角, 揭示 AS 是 Transformer 的 universal property
2. 五个 mechanistic interpretation 的系统整合, 从 softmax 数学根源到几何结构
3. Mitigation 方法的两大类划分 (提供 explicit alternative vs. 切断 causal chain)

对做 LLM/VLM 的研究者, 这篇 paper 是理解 model 内部 numerical dynamics 的 critical reading. 对做 efficient inference 的工程师, §3.1 Sink Token Preservation 是最实用的指南. 对做 mechanistic interpretability 的人, §4 的五个视角提供了丰富的研究 roadmap. 对做 quantization 的人, §5 的 mitigation 方法和 §6.3 的 quantization-aware protection 是核心.
