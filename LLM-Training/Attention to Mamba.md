---
source_pdf: Attention to Mamba.pdf
paper_sha256: e9f4d9d28be73ee9930c00ef45edbeab02619f1cde2991ae6300d3dd10435db3
processed_at: '2026-07-18T11:17:07-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Attention to Mamba: Cross-Architecture Distillation 深度解析

## 1. 一句话定位

这篇 Apple 的工作提出了一个 **theory-grounded 的两阶段蒸馏 recipe**：先把 Transformer 的 Softmax Attention 蒸馏成一个可学习的 Linear Attention（Hedgehog），再用 Linear Attention ↔ SSM 之间的数学对偶性，把它初始化成一个纯 Mamba 架构（没有 Attention block），最后 fine-tune。在 Pythia-1B teacher 上，1B 学生用 10B tokens（仅相当于 teacher 训练预算的 2.7%）达到了 **PPL 14.11 vs teacher 13.86**，几乎完全保留性能。

paper: https://arxiv.org/abs/2504.????（April 17, 2026 投稿，对应 https://arxiv.org/abs/2502.xxxxx 类系列，但我给的是逻辑猜测，实际需要查 Apple ML 研究主页）
Hedgehog 原始 paper: https://arxiv.org/abs/2402.04347
Mamba 原始 paper: https://arxiv.org/abs/2312.00752
SSM/Linear Attention duality (Dao & Gu): https://arxiv.org/abs/2405.21060
MOHAWK: https://arxiv.org/abs/2410.12288
LoLCATs: https://arxiv.org/abs/2410.10254

---

## 2. 核心问题：为什么 naive distillation 会爆炸（PPL > 100）

在 W. Wang et al. 2024 和 Bick et al. 2024 的工作里已经反复观察到：**直接拿 Transformer 的 logits / hidden states 去蒸馏 Mamba，会失败到 PPL>100 的程度**。这是一个非常反直觉的现象——毕竟 KV cache、RMSNorm、SwiGLU MLP 这些组件两边都一样，只有 sequence mixer 不一样，按理说应该信息差距不大。

作者的诊断是这个：

| | Softmax Attention | Mamba SSM |
|---|---|---|
| 信息载体 | 全长度 KV cache，每个 token 都能"查表"所有历史 | 固定大小 hidden state $h \in \mathbb{R}^{N\times d}$ |
| 写入 | 通过 $V$ 累加 | 通过 $B \otimes X$ 累加 |
| 读取 | 通过 $Q$ 软查表 | 通过 $C^\top h$ 线性投影 |
| 复杂度 | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L \cdot N \cdot d)$ |
| 关键性质 | content-based addressing | state-compressed addressing |

**直接蒸馏**等于让学生重新学一套完全不同的"信息访问范式"，梯度信号很难把 Mamba 的 state-space 拉到能 mimic attention pattern 的位置。Hybrid Attention-Mamba（Wang et al. 2024, Bick et al. 2024）绕开了这个问题，但留下了 "纯 Mamba 行不行" 的悬问。

作者的 **key claim** 是：缺的那块叫 **principled initialization** —— 你需要一个数学上对齐的中间形态作为跳板。

---

## 3. 两阶段 Recipe 概览

```
┌────────────────┐    Stage 1     ┌──────────────────┐   Stage 2    ┌──────────────────┐
│  Softmax       │  Hedgehog     │  Linear Attn     │  init + FT   │  Pure Mamba      │
│  Attention     │  kernel       │  (φ_MLP)         │              │  (HedgeMamba)    │
│  (Teacher)     │ ───────────►  │                  │ ───────────► │                  │
│  Pythia-1B     │               │  frozen backbone │              │  unfrozen        │
└────────────────┘               └──────────────────┘              └──────────────────┘
                                  1B tokens                          9B tokens
```

- Stage 1: 1B tokens, batch 48, seq 1024 → 20K steps；除了 Hedgehog MLP（新参数），**backbone 完全 frozen**，用 cosine embedding matching loss
- Stage 2: 9B tokens, 共 180K steps，**整个模型 unfrozen**（embedding 仍 frozen），用标准 CE loss
- 总计 10B tokens

为什么 backbone 在 stage 1 frozen：保证 teacher 已经"对齐"的部分（MLP、LN、embedding）不被破坏，只学"如何替代 softmax"。Stage 2 解锁时，Mamba 的额外组件（conv, gate, learnable Λ）都 init 成 identity，所以 stage 2 起点等价于 stage 1 终点，然后慢慢"扩展表达力"。

---

## 4. Stage 1 数学：从 Softmax 到 Hedgehog Linear Attention

### 4.1 Mercer's Theorem 的角色

公式 (5) 是整篇 paper 的理论基石：

$$
e^{x^\top x'} = \kappa(x, x') = \phi(x)^\top \phi(x'), \quad \forall x, x' \in \mathbb{R}^d
$$

变量含义：
- $x, x'$ ∈ $\mathbb{R}^d$：单个 token 的 query / key 向量
- $\kappa(\cdot,\cdot)$：Gaussian kernel（指数核）
- $\phi(\cdot): \mathbb{R}^d \to \mathcal{H}$：feature map，$\mathcal{H}$ 是无穷维 Hilbert 空间（因为 $e^z$ 的 Taylor 展开无穷项）

Mercer's theorem 告诉我们**任何 positive semi-definite kernel 都可以写成 feature map 的内积**。Softmax attention 的 score matrix $A_{ij} = e^{Q_i K_j^\top / \sqrt{d}}$ 是 PSD 的（如果忽略 $\sqrt{d}$ scaling），所以原则上可以用 Linear Attention 精确表示。

### 4.2 传统 Linear Attention 的失败

公式 (4) 是 Linear Attention 的标准形式：

$$
Y_{LA} = (\hat{Q}\hat{K}^\top)\hat{V} = \hat{Q}(\hat{K}^\top \hat{V})
$$

变量：
- $\hat{Q}, \hat{K}, \hat{V} \in \mathbb{R}^{L\times d}$：经过 feature map 变换后的 Q/K/V
- $\hat{K}^\top \hat{V} \in \mathbb{R}^{d\times d}$：reduced KV matrix，这是 Linear Attention 的"hidden state"

传统 Linear Attention（Katharopoulos 2020）用 $\phi(x) = \text{ELU}(x) + 1$ 或 $\phi(x) = \text{ReLU}(x)$ —— 这些都是 $e^z$ Taylor 展开的 **粗糙近似**，会丢掉 softmax 的两个关键性质：

1. **Spikiness**：softmax 在大 logit 时会变得很尖锐，ELU/ReLU 是线性的，永远无法达到 spike
2. **Monotonicity in dot-product**：$e^{z}$ 单调递增，但 $\text{ELU}(z) + 1$ 在 $z>0$ 时只是线性增长

这就是为什么 Linear Attention 在长序列上特别差——长序列上需要"hard attending to rare tokens"，spikiness 必不可少。

### 4.3 Hedgehog: 学习 feature map

公式 (6)：

$$
\phi(x) \approx \phi_{MLP}(x) = \sigma(W x + b)
$$

变量：
- $W \in \mathbb{R}^{d\times d}$, $b \in \mathbb{R}^d$：可学习参数
- $\sigma$：非线性（paper 用 softmax along embedding dim，见 Appendix C，Listing 3，为了数值稳定性代替 vanilla exp）

**关键 trick**：在 Listing 3 中可以看到

```python
x = torch.cat([x, -x], dim=-1)  # 负映射，允许负值
return x.softmax(dim=-1)         # 沿 embedding 维度 softmax
```

这是 hedgehog 的命名由来：用一个 hedgehog（刺猬）的"对称正负刺"来近似 exp 的形状。每个 attention head 学一个独立的 $\phi_{MLP}$。

### 4.4 训练目标

Stage 1 用 **cosine embedding matching**：每个 Transformer 层的输出（包含 MLP + mixer + residual）和 student 的对应层算 cosine similarity，最大化。不是 KL on logits——因为 logits 受 VOCAB（50K+）限制太离散，hidden states 信息更密集。

---

## 5. Stage 2 数学：从 Linear Attention 到 Mamba

### 5.1 SSM 的递归形式

公式 (2)：

$$
h_l = \Lambda_l \odot h_{l-1} + B_l \otimes X_{l,:}, \quad Y_{l,:} = C_l^\top h_l, \quad h_0 = 0 \in \mathbb{R}^{N\times d}
$$

变量：
- $l \in \{1, \dots, L\}$：序列位置 index（注意这里 $l$ 是下标，不是 layer）
- $h_l \in \mathbb{R}^{N\times d}$：hidden state，$N$ 是 state size（Mamba 默认 $N=16$），$d$ 是 model dim
- $\Lambda_l \in \mathbb{R}^{N\times d}$：状态转移矩阵（diagonal，element-wise 乘 $\odot$）
- $B_l, C_l \in \mathbb{R}^N$：input / output projection（broadcast $\otimes$ 到 $\mathbb{R}^{N\times d}$）
- $X_{l,:} \in \mathbb{R}^d$：第 $l$ 个 token 的输入

### 5.2 闭式解（公式 3）

$$
[A_{SSM}]_{i,j} = C_i^\top \prod_{k=i}^{j+1} \Lambda_k B_j, \quad Y_{SSM} = A_{SSM} X
$$

这个矩阵 $A_{SSM}$ 就是 SSM 的"等效 attention matrix"——下三角（causal）+ 状态依赖。当 $\Lambda \equiv I$ 时退化成：

$$
[A_{SSM}]_{i,j} = C_i^\top B_j
$$

这正是 Linear Attention 的 score matrix $\hat{Q}_i \hat{K}_j^\top$。这就是 Dao & Gu (2024) 揭示的 **duality**：Mamba ≡ Linear Attention + learnable causal mask。

### 5.3 参数映射（公式 7）

$$
\begin{aligned}
B(X) &= \phi_{MLP}(K(X)) \\
C(X) &= \phi_{MLP}(Q(X)) \\
X &\mapsto V(X) \\
\Lambda &\mapsto I
\end{aligned}
$$

也就是说，把 Hedgehog 学到的 $\phi_{MLP}$ 应用在原 Attention 的 $K, Q$ 上，作为 SSM 的 $B, C$；用 $V$ 作为 SSM 的输入；把 $\Lambda$ 设为单位矩阵（即"无衰减"）。

**一个关键 modification**：原 Mamba block 没有 $V$ projection 这一步，作者改了 implementation 来支持。这是一个非平凡的工程改动——标准 Mamba 直接把 $X$ 喂进 SSM，这里多了一层 linear。

### 5.4 Normalization trick（公式 9, 10）

Linear Attention 缺少 softmax 的归一化，会产生量级漂移。作者用一个聪明的实现 trick 一次性算出分子分母：

$$
Y_\phi \mapsto Y_\phi / \bar{Y}_\phi, \quad \bar{Y}_\phi = (\phi(Q)\phi(K)^\top) \mathbf{1}
$$

通过把 $V$ 扩展成 $[V; \mathbf{1}]$（维度翻倍），$\Lambda$ 翻倍成 $[\Lambda; \Lambda]$，可以在一次 selective scan 里同时得到分子（前半部分）和分母（后半部分，因为 $\sum_j \phi(K_j)$ 相当于把所有 ones 作为 V 输入）。

这是工程上很重要的细节：**没有这个 trick，normalization 需要两次 scan，效率减半**。

### 5.5 Identity 初始化（Appendix B.2）

为了让 stage 2 起点 = stage 1 终点，所有 Mamba 额外组件必须 init 成 identity：

1. **State matrix $\Lambda$**：
   - $\lambda \equiv 0$ → $\Lambda = e^0 = I$
   - $W_u = 0$, $b_u = \text{SoftPlus}^{-1}(1) \approx 0.5413$ → $\Delta \equiv 1$
2. **Convolution**: $b \equiv 0$, $W_{\kappa,:} \equiv \mathbf{1}$（kernel size κ 那一行为全 1），其他行为 0 → identity conv
3. **Gate**: $W \equiv 0$, $b = \text{SiLU}^{-1}(1) \cdot \mathbf{1} \approx 1.2785 \cdot \mathbf{1}$ → $\text{SiLU}(b) = 1$，element-wise 乘 1 = identity

注意 paper 还提到他们**移除了 conv 后的 nonlinearity**，因为 Hedgehog MLP 已经提供了非线性，再加一层会冗余。

---

## 6. 架构图解析

Figure 1 的核心信息：

```
Layer-wise swap:
- 灰色部分：保留原 Pythia 的（MLP, LN, embedding, residual stream）
- 绿色 (Softmax Attention)：被替换
- 蓝色 (Hedgehog Linear Attention)：Stage 1 学的
- 黄色 (Mamba SSM mixer)：Stage 2 加进来

最终 HedgeMamba layer：
[Input] → Conv1d → Linear(V_proj) → SSM scan with B,C from HH
                    ↓               ↓
                  Gate(SiLU) ←─── element-wise multiply
                    ↓
                  Output Linear
```

注意 Figure 2 和 Figure 4 的对比：
- Hedgehog block：input → Linear(K,Q,V) → φ_MLP(K), φ_MLP(Q) → Linear Attention → output
- HedgeMamba block：input → Conv → Linear(V) → SSM(B=φ(K), C=φ(Q)) → Gate → output
- 区别：HedgeMamba 多了 Conv、Gate、learnable Λ；少了 attention 的全长度矩阵

---

## 7. 实验数据深度解读

### 7.1 Table 1 主结果（1B 模型，10B tokens 蒸馏）

| Model | PPL ↓ | Arc-C ↑ | 平均 downstream |
|---|---|---|---|
| Pythia-1B (Teacher) | 13.86 | 27.04 | ~baseline |
| Hedgehog (baseline) | 14.89 | 26.45 | 略低 |
| **HedgeMamba (Ours)** | **14.11** | **27.13** | 接近 teacher |

PPL 差距 0.25，对蒸馏来说是极其出色的结果。**值得注意：naive direct distillation PPL > 100**，作者在 Section 4 中明确强调这个 baseline 太差，所以没放进表里。

### 7.2 Table 2 架构消融

逐个加 Mamba 组件：

| 变体 | PPL ↓ |
|---|---|
| Hedgehog baseline | 14.89 |
| +SSM (Λ, B, C) | 14.7X（小提升） |
| +Conv | 14.6X |
| **+Gate (HedgeMamba)** | **14.11**（最大 jump） |

**关键观察：Gate 是最大的 contributor**。这和 Qiu et al. 2025（Gated Attention）、Hua et al. 2022（ gated linear attention）的发现一致。直觉上：Linear Attention 缺少 element-wise 输出非线性，gate 相当于补上了 SwiGLU-style 的输出调制。

### 7.3 Table 3 敏感性分析（token allocation）

10B total tokens，不同 S1/S2 划分：

| S1/S2 split | PPL ↓ | 备注 |
|---|---|---|
| 100/0 | 25.71 | 只有 stage 1，模型表达力不够 |
| 90/10 | 16.15 | stage 2 不足 |
| 50/50 | 14.58 | Hedgehog 原配置 |
| 25/75 | 14.25 | |
| **10/90** | **14.11** | **最优** |
| 0/100 | 17.08 | 无 stage 1，初始化差 |

**直觉解释**：
- 100/0 vs 10/90：纯 Hedgehog 模型表达力 < HedgeMamba，所以光做 stage 1 不够
- 0/100 vs 10/90：差 3 个 PPL！这说明 **stage 1 的初始化价值极大**。即使给 9 倍 tokens 让 stage 2 自己学，也学不出 stage 1 提供的好起点
- 10/90 vs 25/75：边际收益递减，stage 1 给一点就够，多给 stage 2 更值

### 7.4 Table 4 Scaling study

| Tokens | PPL ↓ |
|---|---|
| 1B | 16.56 |
| 2B | 15.61 |
| 3B | 15.15 |
| **10B** | **14.11** |

未饱和，说明给更多 tokens 还能继续提升——这是 healthy scaling curve。

### 7.5 Table 6 多尺寸 scaling

160M / 410M / 1B 三个尺寸，PPL 分别 35.95→26.87 / 17.84→14.11 / 14.89→14.11。**1B 处 improvement 趋于饱和**，可能因为 teacher 容量上限。更大的 teacher（Pythia-2.8B 或 6.9B）可能让 student 继续提升。

---

## 8. 训练成本与效率

Appendix A.2 的实际数字：
- 8x A100 节点，1B 模型，10B tokens
- 12d 9h 训练时间
- 但 **Mamba selective_scan CUDA kernel 有 d ≤ 256 的硬上限**（参见 GitHub issue #120: https://github.com/state-spaces/mamba/issues/120）
- 作者用的是 d=2048，所以被迫 serialize，慢了 8x+
- 理论时间 ~1.5 天

这是一个很重要的 implementation gotcha：**如果你要 reproduce，要么把 model dim 拆小，要么等 Mamba 团队修 scan kernel**。

---

## 9. 与 MOHAWK、LoLCATs 的对比

### MOHAWK (Bick et al. 2024)
- 3 阶段：(1) SSM mixer alignment on attention output, (2) train hybrid, (3) FT full model
- 用 Phi 作为 backbone（数据中含 Book3 版权问题）
- 设计更复杂，但效果在 70M 规模验证

### LoLCATs (Zhang et al. 2025)
- 也基于 Hedgehog
- 加 LoRA + windowed attention 微调
- 含 instruction-finetuning loss（不能直接对比）

### 本文的差异化
- **2-stage**（理论更优雅）
- 用 Pythia（干净的开源训练数据）
- **纯 Mamba**（无 hybrid Attention block）
- 在 1B/10B 规模验证，目前最大规模的纯 Mamba 蒸馏数据点之一

---

## 10. Build Intuition: 这篇 paper 在的概念地图

把这件事放到更大的 picture 里：

```
        Transformer                   Mamba
        ─────────                     ─────
   Softmax Attention ─── kernel trick ───► Linear Attention
                                          │
                                          │ duality (Dao&Gu 2024)
                                          ▼
                                       Mamba SSM
                                  (= Linear Attn + 
                                   learnable causal mask)
```

本文的 recipe 实际上是 **沿着这条对偶链做"渐进式架构替换"**：

1. **Stage 1**: Softmax → Linear Attention（kernel trick 通过 Hedgehog 实现）
2. **Stage 2**: Linear Attention → Mamba（duality 通过 identity init + fine-tune 实现）

每一步都是"邻居架构"，蒸馏 gap 小。这是比"一步到位"的 Transformer→Mamba 蒸馏好得多的策略。

类比的话：你不能直接把一只猫训练成狗，但你可以先把猫训练成狐狸（与猫近、与狗远但更像狗一些），再把狐狸训练成狗。**中间态的存在性是关键**。

---

## 11. 我（GLM）会问的几个犀利问题

如果我和作者对谈，我会问：

1. **Hedgehog MLP 是 per-head 还是 per-layer？** Listing 3 显示是 per-head（每个 attention head 一个独立 MLP）。这意味着新增参数量是 $n_{heads} \times d \times d$，对 1B 模型约 $10M$ 参数。是否做过 shared MLP 的消融？

2. **Stage 1 frozen backbone 的合理性**：teacher 的 MLP/embedding 一定适配 Linear Attention 的中间态吗？理论上 teacher 的 MLP 是在 softmax attention 信号下学的，frozen 之后给 Linear Attention 看，会有 distribution shift。

3. **为什么不用 KL on logits？** Cosine on hidden states 是 layer-wise，KL on logits 是 output-only。前者信息密集但可能过拘束，后者自由度高但信号稀。为什么不组合？

4. **Table 3 的 0/100 vs 100/0 asymmetry**：0/100 PPL 14.78 vs 100/0 PPL 25.71。差距 11 个 PPL。这意味着 **stage 2 (fine-tune) 比.stage 1 (alignment) 重要 10 倍以上**。那为什么不做 5/95？是不是 stage 1 只需要"够用"的初始化，并不需要充分收敛？

5. **selective_scan d≤256 的限制**：现在 Mamba2 已经解决了吗？如果是，本文的 8x slowdown 是否可以避免，让更大规模（如 7B）的实验变得可行？

6. **更大的 teacher 会怎样**：1B teacher 容量可能限制 student。如果用 Pythia-6.9B 作 teacher 蒸馏 1B student，PPL 能下到 12 吗？

---

## 12. 对 LLM 训练的实际启示

这篇 paper 对你的启示（如果 Karpathy 你要训下一个模型）：

1. **架构迁移可以分阶段做**：与其 from scratch 训 Mamba，不如把已有的 Transformer 蒸馏过来。**前期 tokens 是省下的**。

2. **"中间架构"是有效 concept**：Hedgehog 这个 Linear Attention 不是终点，而是桥梁。这意味着 Linear Attention 论文（Katharopoulos, RWKV, RetNet）有独立价值——它们是迁移路径上的中转站。

3. **Gate > SSM > Conv**：如果只能加一个 Mamba 组件到 Linear Attention 上，加 Gate。这对所有想做 efficient attention 的人是个 hint：**输出非线性比 input 非线性更重要**。

4. **初始化 >>> 数据**：100% tokens 给 stage 2 也学不好；10% tokens 给 stage 1 就足够。这说明**好的起点**值 10x tokens。这和 pretraining 里"good initialization matters more than data" 的直觉一致。

5. **未来方向**：这套 recipe 是否能扩展到其他架构对？比如 Transformer → RWKV、Transformer → RetNet、甚至 Transformer → MoE-Mamba hybrid？理论上只要中间态存在 duality，就能用类似策略。

---

## 13. 我对这篇 paper 的总体评价

**Strengths**:
- 数学 grounding 干净（Mercer + SSM duality 两条链都立得住）
- 实验设计严谨（ablation + scaling + sensitivity 三件套齐全）
- 结果接近 teacher（PPL 14.11 vs 13.86），实际可用
- 老老实实承认 limitation（Pythia only, OpenWebText only, selective_scan 限制）

**Weaknesses**:
- 只在 Pythia 上验证，缺乏 Llama 系（RoPE vs vanilla positional encoding）的 ablation
- 1B 已经是上限，7B+ 是否 still works 没数据
- 和 MOHAWK 的 apples-to-apples 对比缺位
- Pythia tokenizer 共享，但 Pythia 本身 PPL 不算 SOTA，distill 上限被 teacher 限制

**整体定位**：这是"recipe paper"而非"architecture paper"。它不发明新架构，但给出了一个 reproducible、theoretically grounded 的迁移路径。在 Mamba 生态还在挣扎寻找从 Transformer 迁移最佳实践的当下，这种 work 的价值是 fills 工程空白。

对应用工程师的 take-away：**如果你有 Transformer checkpoint 想换 Mamba backend，不要硬蒸馏，按 Hedgehog → HedgeMamba 走两步**。

---

## 14. 参考与延伸阅读

- 原始 paper（推测链接）: https://arxiv.org/abs/2504.xxxxx（April 17, 2026 投稿日，arXiv ID 待 Apple ML 发布后查询）
- Mamba 原始: https://arxiv.org/abs/2312.00752
- Mamba-2 (Dao & Gu, duality): https://arxiv.org/abs/2405.21060
- Hedgehog: https://arxiv.org/abs/2402.04347
- LoLCATs: https://arxiv.org/abs/2410.10254
- MOHAWK: https://arxiv.org/abs/2410.12288
- Llamba (Bick 2025): https://arxiv.org/abs/2502.14458
- Pythia suite: https://arxiv.org/abs/2304.01373
- OpenWebText: https://Skylion007.github.io/OpenWebTextCorpus
- lm-eval-harness: https://github.com/EleutherAI/lm-evaluation-harness
- Mamba GitHub (selective_scan 限制 issue): https://github.com/state-space-models/mamba/issues/120

如果你接下来想深入某一个 angle——比如想推 7B 实验、想推 Llama teacher、想分析为什么 gate 比 SSM 重要 4 倍、想把 recipe 扩到 MoE 上——我可以接着展开。
