---
source_pdf: AVA-VLA.pdf
paper_sha256: 76e25eff75a9bf09b35b982ef4c3bc6dae87272dbaaa8ed882be52fa059eb949
processed_at: '2026-07-18T12:44:40-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AVA-VLA 深度解析：从 MDP 到 POMDP 的范式转换

## 1. 核心问题的直觉

大多数 VLA 模型 (OpenVLA, π0, RT-2 等) 都基于预训练 VLM 构建，每个 timestep 独立处理一帧图像：

$$\mathcal{A}^t \sim \mathcal{P}_\theta(\mathcal{A}^t \mid \mathbf{x}^t)$$

这里 $\mathbf{x}^t = (\mathbf{x}_I^t, \mathbf{x}_S^t)$ 是当前帧图像 + 语言指令，$\mathcal{A}^t$ 是 action chunk。这隐式地假设了 **MDP**——当前观测包含完整 world state。

但机器人操作本质上是 **POMDP**。考虑 "turn on the stove and put the moka pot on it" 这种任务：stove 开关可能被遮挡，或者上一帧已经完成了某个子目标。当前帧无法告诉模型"我已经走到了哪一步"。这就是 Figure 1(b) 里 OpenVLA-OFT 失败的原因——它根据时间序列无法定位 stove switch。

**关键 insight**：VLA 是一个 dynamic feedback control system，agent 的上一个 action 直接改变了当前视觉输入。但 MDP-based 方法每一步都从零开始重新分配 attention weights，仅仅依赖 static language instruction。这导致视觉系统是 "inactive" 的——它无法基于历史 context 来 anticipate perceptual intent。

参考 POMDP 综述：https://arxiv.org/abs/2203.09862

---

## 2. 框架设计：Recurrent State 作为 Belief Approximation

### 2.1 POMDP 视角的理论重构

POMDP 中 optimal policy 应该 conditioned on belief state：

$$b^{t-1} = P(s_{t-1} \mid \mathbf{x}^{<t}, \mathcal{A}^{<t})$$

这总结了所有历史 observations 和 actions。但直接计算 belief state 通常 intractable。AVA-VLA 的方案是学习一个压缩表示 $r^{t-1}$ 作为 neural approximation：

$$\mathcal{A}^t \sim \mathcal{P}_\theta(\mathcal{A}^t \mid \mathbf{x}^t, r^{t-1})$$

这把 VLA 变成了一个 recurrent structure。

### 2.2 Recurrent State 的来源

关键设计选择：**recurrent state 来自上一时刻 action-related hidden state**。

对于 parallel-decoding VLA (如 OpenVLA-OFT)，有 $M$ 个 decoder layer，输出 $L_A = L_c \cdot D$ 个 action token ($L_c$ 是 chunk size，$D$ 是 action dimension)。第 $m$ 层在时刻 $t$ 的 hidden state 记为 $h_m^t \in \mathbb{R}^{L_A \times d}$。

recurrent state 计算：

$$r^{t-1} = \mathcal{B}(h_M^{t-1}) \in \mathbb{R}^{L_A \times d}$$

其中：
- $h_M^{t-1}$：上一时刻最后一层的 hidden state，包含 fused visual+language 信息，且 predictive of agent intent
- $\mathcal{B}$：2-layer MLP with SiLU activation
- $d$：embedding dimension (LLaMA2-7B 是 4096)

**直觉**：在 action generation 之前的 hidden state 已经压缩了"我要做什么"的信息，这是 belief state 最自然的 neural proxy。

参考 RNN 与 belief state 关系：https://ieeexplore.ieee.org/document/9415175

---

## 3. AVA Module 架构详解

AVA module 是这个工作的核心创新。它的目标：**用 recurrent state 量化每个 vision token 的重要性，并动态调制 LLM backbone 所有层的 attention**。

### 3.1 完整数据流

输入：当前 visual tokens $z_I^t \in \mathbb{R}^{L_I \times d}$，instruction tokens $z_S^t \in \mathbb{R}^{L_S^t \times d}$，recurrent state $r^{t-1} \in \mathbb{R}^{L_A \times d}$

**Step 1: Modality MLPs 降维**

$$\bar{z}_I^t \in \mathbb{R}^{L_I \times d'}, \quad \bar{z}_S^t \in \mathbb{R}^{L_S^t \times d'}, \quad \hat{r}^{t-1} \in \mathbb{R}^{L_A \times d'}$$

其中 $d' < d$ (压缩到更低维度做 attention，计算更便宜)。

**Step 2: FiLM 条件化**

Feature-wise Linear Modulation 将 visual features 与 language instruction 融合：

$$\hat{z}_I^t = \mathcal{F}_\gamma(\bar{z}_S^t) \odot \bar{z}_I^t + \mathcal{F}_\beta(\bar{z}_S^t)$$

其中 $\mathcal{F}_\gamma, \mathcal{F}_\beta$ 是从 instruction feature 预测的 scale 和 shift。这让 visual features 带上了 "我在找什么" 的语义信息。

FiLM 原文：https://arxiv.org/abs/1709.07871

**Step 3: Cross-Attention + Self-Attention**

用 vision tokens 作为 Query，recurrent state 作为 Key/Value：

$$Q^t = W_Q \hat{z}_I^t \in \mathbb{R}^{L_I \times d'}$$
$$K^t, V^t = (W_K / W_V) \hat{r}^{t-1} \in \mathbb{R}^{L_A \times d'}$$

$$O^t = \text{Self-Att}(\text{Cross-Att}(Q^t, K^t, V^t))$$

**直觉**：每个 vision token 去 "询问" recurrent state——"基于历史 belief，我这块区域现在重要吗？" Cross-attention 的输出再过 self-attention 做 token 间信息交换。

**Step 4: Soft Weights 预测**

$$\rho^t = \text{Softmax}(\mathcal{W}(\text{FFN}(O^t))) \in \mathbb{R}^{L_I \times 2}$$

其中 $\mathcal{W}: \mathbb{R}^{d'} \to \mathbb{R}^2$ 是线性层。Softmax 沿 feature dimension，输出每个 vision token 的 (enhance, weaken) 概率。

最终 soft weights：

$$\omega^t = \rho^t \gamma$$

其中 $\gamma = [\gamma_0, \gamma_1] = [1.9, 0.1]$ 是可学习 (这里实际上是固定超参) 的标量对：
- $\gamma_0 = 1.9$：enhance 的强度 (>1 表示放大)
- $\gamma_1 = 0.1$：weaken 的强度 (<1 表示抑制)

**为什么是 2-class Softmax 而不是 sigmoid？** 这里设计有点意思——把 token importance 建模为二分类 (enhance vs weaken)，然后通过 $\gamma$ 的不对称值实现非对称放大/抑制。$\gamma_0 - \gamma_1 = 1.8$ 的 gap 保证了区分度。

### 3.2 Soft Attention Mask：核心机制

这是最技术性的部分。soft weights $\omega^t$ 需要应用到 LLM backbone 的 **所有 attention layer**，而不仅仅是某一层。

对于第 $m$ 层，原始 attention score 经过 mask 后得到 $C^{t,m} \in \mathbb{R}^{L_o^t \times L_o^t}$ ($L_o^t$ 是总序列长度)。最终 attention matrix：

$$A_{i,j}^{t,m} = \frac{\exp(C_{i,j}^{t,m}) U_{i,j}^t}{\sum_{l=1}^{L_o^t} \exp(C_{i,l}^{t,m}) U_{i,l}^t}$$

其中 soft attention mask matrix $U^t$ 定义为：

$$U_{i,j}^t = \begin{cases} 1 & \text{if } i = j \text{ or } j \notin \Lambda_I \\ \omega_j^t & \text{if } i \neq j \text{ and } j \in \Lambda_I \end{cases}$$

$\Lambda_I$ 是 vision tokens 的 index 集合。

**关键解读**：
- 对角线 ($i=j$)：保持 1，self-attention 不变
- 非 vision token 列 ($j \notin \Lambda_I$)：保持 1，language/action token 不受影响
- vision token 列 ($j \in \Lambda_I$, $i \neq j$)：被 $\omega_j^t$ 调制

**为什么是调制 column 而不是 row？** 因为 attention 是 query attends to key。调制 column $j$ 意味着：**所有 query 对 vision token $j$ 的关注度被统一缩放**。这等价于在 softmax 之前对 vision token $j$ 的 logit 加上 $\log(\omega_j^t)$。

当 $\omega_j^t$ 接近 0 时，vision token $j$ 几乎被所有 query "屏蔽"；当 $\omega_j^t > 1$ 时，它被增强。这是一种 **token-level 的软门控**，作用于整个 LLM backbone 的所有层。

### 3.3 State-based Placeholder Initialization

OpenVLA-OFT 用 empty placeholder $p^t = \mathbf{0}$ 作为 action generation 的 prompt。AVA-VLA 把它替换为 recurrent state：

$$p^t = r^{t-1}$$

这相当于把历史 belief 直接注入到 action generation 的输入端。Ablation (Table 4) 显示单独这一项就带来 +0.7% 提升 (97.5% vs 96.8%)。

### 3.4 完整 Forward Pass

$$\mathcal{A}^t = \mathcal{Q}(\mathcal{M}_{\text{parallel}}(z_I^t, \mathcal{V}(X^t, r^{t-1}), z_S^t, r^{t-1}))$$

注意 $r^{t-1}$ 出现在两个地方：(1) 作为 AVA module $\mathcal{V}$ 的输入调制视觉；(2) 作为 action placeholder 初始化。两路共同确保历史 context 流入决策。

---

## 4. 训练策略：Truncated BPTT

理想情况下应该对整条 trajectory 做 BPTT，但 VLA backbone (LLaMA2-7B) 内存开销太大。采用 truncated BPTT：

**训练时**：unroll $T=4$ 步，在第 2 和第 3 步之间 detach gradient (实际只 backprop 2 步)。

每个 batch 包含 $N$ 条长度 $T$ 的连续序列。Loss：

$$\mathscr{L}_{\text{total}} = \sum_{n=1}^{N} \sum_{t=0}^{T-1} (\mathscr{L}^{t,n} + \lambda \mathscr{L}_\omega^{t,n})$$

其中：
- $\mathscr{L}^{t,n}$：action chunk 的 MAE loss
- $\mathscr{L}_\omega^{t,n} = \|\mu(\omega^{t,n}) - c\|$：soft weights 均值的 L2 penalty
- $\mu(\cdot)$：mean function
- $c$：target mean (LIBERO 用 0.6, CALVIN 用 0.2——因为 CALVIN region of interest 更小)
- $\lambda = 1.0$

**这个 regularizer 的作用**：防止 soft weights 全部塌缩到 0 (那样模型就完全忽略视觉) 或全部塌缩到 1 (那样 AVA 失效)。强制 mean 接近 $c$，让模型保持适度的视觉关注。

初始状态 $r^{-1} = \mathbf{0}$。

**Inference**：完全 recurrent，每步 forward 同时预测 action chunk 和提取新的 $r^t$。

参考 truncated BPTT：https://arxiv.org/abs/1802.06705

---

## 5. 实验结果解析

### 5.1 LIBERO Benchmark

| Setting | OpenVLA-OFT | AVA-VLA | Δ |
|---------|-------------|---------|---|
| One policy all suites | 96.8% | **98.0%** | +1.2% |
| One policy per suite | 97.1% | **98.3%** | +1.2% |

最显著的提升在 LIBERO-Long (97.6% vs 95.3%)，这是最需要历史 context 的 long-horizon 任务。

### 5.2 CALVIN ABC→D

| Metric | OpenVLA-OFT | AVA-VLA |
|--------|-------------|---------|
| Avg len | 4.28 | **4.65** |
| 5 tasks in row | 72.9% | **84.1%** |

CALVIN 是 long-horizon 多阶段任务，AVA-VLA 优势更明显。

### 5.3 Mobile ALOHA Real-World

四个任务：Pick&Place, Tower of Hanoi, Towel Folding, Scooping。AVA-VLA 在 cross-task average 上最高，验证 sim-to-real 能力。

### 5.4 Ablation 关键发现

**Backbone 通用性** (Table 3)：在 OpenVLA-7B (+1.7%), LLaMA2-7B (+2.6%), Qwen2.5-0.5B (+1.4%) 上都有效，即使没有 robotic pre-training。

**Component 贡献** (Table 4)：
- Baseline: 96.8%
- + State init only: 97.5%
- + AVA module only: 97.5%
- + Both: 98.0%

两者组合有协同效应。

### 5.5 Token Pruning：意外收获

Table 5 展示了 soft weights 用于 token pruning 的潜力：

| Pruning Ratio | Avg SR |
|---------------|--------|
| 0% | 98.0% |
| 50% | 97.3% |
| 70% | 97.3% |
| 90% | 93.9% |

**剪掉 70% 的 vision tokens，性能只掉 0.7%**。这说明 AVA 学到的 importance 确实抓住了 task-relevant region。即使剪掉 90%，仍然 93.9%，超过很多 baseline。

这暗示了一个 efficiency 方向：soft weights 天然可用于 dynamic token pruning。

参考 token pruning 工作：https://arxiv.org/abs/2108.02056 (DynamicViT)

---

## 6. 与相关工作的对比

### 6.1 vs. Sequential VLM (VLM-3R, Continuous 3D Perception)

这些工作也用历史信息，但是用于 **passive comprehension** (video understanding)。AVA-VLA 用于 **active decision-making**，需要 interaction。

### 6.2 vs. Token Pruning VLA (SP-VLA, SpecPrune-VLA, VLA-Cache)

这些方法主要目标是 efficiency，通过 frame comparison 或 KV-cache reuse 来 prune tokens。AVA-VLA 的目标是 **effectiveness**——通过历史 belief 来 enhance task-relevant visual processing。Token pruning 只是 side benefit。

### 6.3 vs. TraceVLA, CoT-VLA

TraceVLA 用 visual trace prompting，CoT-VLA 用 chain-of-thought reasoning。这些都是通过 prompting 引入历史，而 AVA-VLA 通过 architectural design (recurrent state) 显式建模历史。

---

## 7. 直觉总结

把整个故事串起来：

1. **问题本质**：VLA 的 MDP 假设导致 visual processing 每步从零开始，无法利用 "我刚做了什么" 的信息
2. **理论框架**：POMDP 告诉我们应该 condition on belief state
3. **工程实现**：用上一时刻 action hidden state 作为 belief 的 neural proxy
4. **核心机制**：AVA module 让 vision tokens "询问" recurrent state，得到每个 token 的 enhance/weaken 权重，通过 soft attention mask 调制所有 LLM 层
5. **训练技巧**：truncated BPTT (T=4, detach at step 2-3) + soft weight mean regularizer

**最 elegant 的点**：soft attention mask 的设计——通过修改 column-wise 的 attention mask，实现对 vision token 的全局重要性调制，同时保持 self-attention 和 non-vision token 不变。这是一个非常 clean 的 architectural intervention。

**潜在局限**：
- $T=4$ 的 truncated BPTT 可能无法捕获超长依赖
- recurrent state 固定来自最后一层 hidden state，是否最优？
- soft weights mean regularizer 的 $c$ 需要手动调 (LIBERO 0.6 vs CALVIN 0.2)

这篇工作的核心贡献是把 POMDP 的 belief state 概念以非常 practical 的方式引入 VLA，同时给出了一个 architecture 上很优雅的实现。Token pruning 的 bonus 应用也展示了 soft weights 的语义意义。

论文原文：https://arxiv.org/abs/2509.xxxxx (需要确认具体 arXiv ID)
OpenVLA-OFT baseline：https://arxiv.org/abs/2502.19645
LIBERO benchmark：https://arxiv.org/abs/2306.03310
CALVIN benchmark：https://arxiv.org/abs/2112.03227
