---
source_pdf: OPRD On-Policy Representation Distillation.pdf
paper_sha256: df417dc3dfc7b26099464c85f85c13d872543a4db55c47e03d2351f9dbca86f1
processed_at: '2026-08-06T01:18:39-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OPRD 人话版

## 一句话总结

现在大家做 on-policy distillation 都是让 student 和 teacher 比"输出概率分布"，OPRD 说别比输出了，直接比"中间层的 representation"。

---

## 为什么要这么干

### 问题 1：输出空间的信号太吵

现在最常用的 sampled-token OPD，本质上是这样工作的：

student 采样出一个 token $\hat{y}_t$，然后算 $\log p_t(\hat{y}_t) - \log q_t(\hat{y}_t)$ 当 reward。这个 reward 乘以 score function $\nabla_\theta \log p_t(\hat{y}_t)$ 就是梯度。

这就是 **REINFORCE**，reward 是个标量，每次只用一个 sample 去估计 KL divergence。REINFORCE 的问题是：**variance 大**。

更麻烦的是，训练越往后，student 越接近 teacher，真正的 signal（$D_{\mathrm{KL}}(p_t \| q_t)$）越小，但 variance 不跟着小。所以 signal-to-noise ratio 崩了，梯度基本是随机噪声在推 student 乱走，训练就 plateau 了。

这就解释了为什么 Figure 3 里 OPD 训练到后面就平了——不是 student 学不动了，是 gradient 信噪比太低，student 在 noise 里 random walk。

### 问题 2：LM head 是个信息瓶颈

Teacher 是个 28 层的 transformer，每层每位置都算出 $d=1536$ 维的 hidden state，总共 $28 \times 16384 \times 1536 \approx 7$ 亿个 scalars 的信息。

但 OPD 只看最后一层 hidden state 经过 $W_{\mathrm{head}} \in \mathbb{R}^{151000 \times 1536}$ 投影后的 logits，再过 softmax 得到概率分布。然后 sampled-token variant 只看其中一个 token 的 log-prob。

从 7 亿个 scalars 压到 1 个 scalar，信息压缩比 $10^8:1$。

更关键的是 $W_{\mathrm{head}}$ 的数学性质：
- $W_{\mathrm{head}}$ 是个 $151000 \times 1536$ 的矩阵，rank 最多 1536，远小于 vocabulary size
- 它的 singular value spectrum 是 long-tail，condition number 很大
- Softmax 对 logit 加常数不敏感（additive invariance）

所以存在一大堆 hidden state 方向（null space $\mathcal{N}_W$），student 和 teacher 在这些方向上差很远，但 output loss 完全看不见。Student 可以在这些方向上随便偏，OPD 不会惩罚。

---

## OPRD 怎么干的

特别简单：在 LM head **之前** 就把 supervision signal 取出来。

具体就是：student 和 teacher 在同一 rollout 上跑 forward，取中间层的 hidden states，算 MSE：

$$\mathcal{L}_{\mathrm{OPRD}} = \frac{1}{|\mathcal{L}_{\mathrm{layer}}|} \sum_{l} \frac{1}{\sum_t m_t} \sum_t m_t \cdot \frac{1}{d} \| h_{\theta,t}^{(l)} - \mathrm{sg}(h_{T,t}^{(l)}) \|_2^2$$

teacher 的 hidden state 用 stop-gradient 包住（不更新 teacher），student 的 hidden state 通过 backprop 更新。

就这么一行 loss，没有 token sampling，没有 KL，没有 softmax，就是最朴素的 MSE。

---

## 为什么这么干 work

### 为什么 variance 没了

OPD 的梯度是 $\nabla_\theta \log p(\hat{y}_t) \cdot (\log p(\hat{y}_t) - \log q(\hat{y}_t))$，这里 $\hat{y}_t$ 是随机采样的，所以梯度是随机的。

OPRD 的梯度是 $\frac{2}{d} J_\theta^\top (h_\theta - h_T)$，这里 $J_\theta = \nabla_\theta h_\theta$ 是 Jacobian，$h_\theta$ 和 $h_T$ 在给定 rollout 后都是确定的。**梯度完全确定，零 variance。**

所以 OPRD 没有 SNR collapse 问题，训练到后期梯度信号还在，student 能一直进步。Figure 3 里 OPRD 单调上升就是这么来的。

### 为什么信息更多了

OPD 在 output space 监督，受限于 $W_{\mathrm{head}}$ 的 rank 和 softmax 的 additive invariance。OPRD 直接在 hidden space 监督：
- 不经过 $W_{\mathrm{head}}$，不受 null space $\mathcal{N}_W$ 影响
- 可以监督任意中间层（不只是最后一层）
- 每个 position 提供 $d=1536$ 个 scalars 的监督，而 sampled-token OPD 只有 1 个

监督信号密度差了几个数量级。

### 为什么还更快更省内存

OPD 的 loss path 要 materialize $[B, T, |\mathcal{V}|]$ 的 logits tensor，$B=8, T=16384, |\mathcal{V}|=151000$，bf16 下约 40 GB。

OPRD 完全不碰 logits，loss 在 LM head 之前算，memory 只跟 $d$ 有关，不跟 $|\mathcal{V}|$ 有关。

实测：OPRD 比 OPD top-16 省 54% transient memory，快 1.44×。

---

## 实验说了什么

### Setup

- Teacher: JustRL-1.5B，Student: R1-distill-1.5B，同 Qwen2.5-1.5B backbone
- 数据 DAPO-Math-17K，500 steps，8×A100
- OPRD 配置：监督 all 28 layers，last 2000 positions

### 主结果（Table 2）

| Method | AIME24 | AIME25 | AIMO |
|---|---|---|---|
| Teacher | 50.8 | 35.6 | 79.5 |
| Student baseline | 32.9 | 21.9 | 62.2 |
| OPD top-1 | 42.3 | 33.5 | 77.0 |
| OPD top-16 | 47.1 | 34.0 | 76.5 |
| OPRD | 49.8 | 34.6 | 79.1 |

OPRD 基本追平 teacher，OPD 两个 variant 都差几个点。

### 训练动力学（Figure 3）

OPD 两种 variant 都是前期快速提升，然后 plateau 或 oscillate。OPRD 一直单调上升到接近 teacher。这是 Theorem 5 的 SNR collapse 在图上的直接体现。

### Response length（Figure 4）

OPRD 收敛到 ~5700 tokens，OPD 收敛到 ~7000 tokens。OPRD 在 accuracy 更高的同时 response 更短，说明 reasoning chain 更精炼。

### Mechanistic analysis（Figure 8, 9, 10）

最有意思的是 Figure 8 的 PG-loss spike：所有 run 都有个 loss 突变，OPRD 让这个突变提前。突变之后 PG-loss 都趋于 0，但 accuracy gap 还在——这就是 LM head bottleneck 的直接证据：output space 已经没信号了，但 hidden space 还有 gap。

---

## 局限

- 要求 student 和 teacher 同架构（cross-architecture 的 hidden states 几乎正交）
- 只在 math reasoning 上测了，其他 domain 待验证
- Layer 和 position 选择还是启发式（all layers + last-k），没有 adaptive

---

## 更大的图景

这篇 paper 的深层信息是：**teacher 不应该被当成一个 probability oracle**，它是一个有 $L$ 层 internal computation 的结构化系统。只取 output probability 是浪费了 teacher 计算出的绝大多数信息。

这个 idea 可以延伸到很多方向：
- Attention map distillation（不只对齐 hidden states，还对齐 attention patterns）
- On-policy self-distillation（teacher 就是 student 自己加 privileged info）
- Cross-architecture distillation（用 projection head 或 contrastive objective 对齐 relative geometry）
- 与 PRM / verifier reward 结合（output space 用 outcome reward，hidden space 用 representation alignment）

代码在 [GitHub](https://github.com/ShenzhiYang2000/OPRD)。

---

## 给你的直觉

OPD 本质是 REINFORCE with per-token reward $\log p - \log q$。REINFORCE 的问题是 variance 大，业界早就知道，解决方法是用 baseline、用 control variate、用更dense的 reward。

OPRD 干的事是把 sparse scalar reward 换成 dense vector supervision，从 RL-style optimization 退回 supervised-style optimization，同时把 supervision 从经过 LM head 压缩后的 output 空间搬到 compression 之前的 hidden space。

所以本质上 OPRD 是**把 on-policy distillation 从一个 policy gradient 问题变回了一个 regression 问题**，然后顺手绕过了 LM head 这个信息瓶颈。这两个变化叠加，就解释了为什么 OPRD 能在 OPD plateau 的地方继续进步。

---

# OPRD: On-Policy Representation Distillation 深度解读

非常 interesting 的 paper，这篇工作精准击中了当前 LLM post-training pipeline 里一个非常微妙、但又被 everyone 看了却没看清的 bottleneck。我尽量把 intuition 给你 build 起来，从 motivation 一直推到 experimental 验证。

---

## 1. Core Thesis: 把 supervision 搬到 LM head 之前

这篇 paper 的 central claim 非常简洁——**所有现有的 on-policy distillation (OPD) variant 都在 LM head 之后取 supervision signal**，而 LM head $W_{\mathrm{head}} \in \mathbb{R}^{|\mathcal{V}| \times d}$ 这个投影，本身就是 information bottleneck + variance source。OPRD 的核心就是把监督从 $W_{\mathrm{head}} \cdot h$（logits）的位置搬到 $h$（hidden states）的位置。

这个 shift 一举解决两个问题：
- **Variance collapse**：sampled-token OPD 是单样本 Monte Carlo 估计 KL，当 $p_t \to q_t$ 时 signal 缩小但 variance 不缩小，SNR 崩塌；
- **Information bottleneck**：$W_{\mathrm{head}}$ 把 d-dim hidden state 压到 $|\mathcal{V}|$-dim logits，但 rank 最多是 d，再加上 softmax 的 additive invariance，存在 huge null space 方向完全无监督。

---

## 2. Background: On-Policy Distillation 的三种 variant

OPD 的 setup 是：student $\pi_\theta$ 自己采样 rollout $\hat{y} \sim \pi_\theta(\cdot|x)$，然后 teacher $\pi_T$ 在同一 rollout 上 evaluate，目标是 trajectory-level reverse KL：

$$\mathcal{L}_{\mathrm{OPD}}(\theta) = \mathbb{E}_{x, \hat{y}} \left[\sum_{t=1}^T D_{\mathrm{KL}}(p_t \| q_t)\right]$$

其中 $p_t(v) = \pi_\theta(v|x, \hat{y}_{<t})$，$q_t(v) = \pi_T(v|x, \hat{y}_{<t})$。

实际部署的三种 variant 区别只在 support 集合 $S_t$：

| Variant | Support $S_t$ | Per-position loss | 优点 | 缺点 |
|---|---|---|---|---|
| Sampled-token | $\{\hat{y}_t\}$（rollout 时已采） | $\log p_t(\hat{y}_t) - \log q_t(\hat{y}_t)$ | 最省 memory | **单样本 MC 估计 KL，高 variance** |
| Full-vocab | $\mathcal{V}$（全部 token） | $\sum_v p_t(v)\log[p_t(v)/q_t(v)]$ | 最 dense signal | $O(BT|\mathcal{V}|)$ memory，不可行 |
| Top-k | TopK($p_t$, k) | truncated + renormalized KL | 折中 | **truncation bias，tail token 完全忽略** |

Table 1（论文 Table 1）里符号定义清楚，最关键的是 $\mathcal{L}_{\mathrm{layer}}$、$\mathcal{P}(\hat{y})$、$m_t$ 这三个新符号——OPRD 用来选择 layer 和 position 的。

---

## 3. OPRD 的核心公式

OPRD 的 objective（公式 6）：

$$\mathcal{L}_{\mathrm{OPRD}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}_x, \hat{y} \sim \pi_\theta(\cdot|x)} \left[\frac{1}{|\mathcal{L}_{\mathrm{layer}}|} \sum_{l \in \mathcal{L}_{\mathrm{layer}}} \frac{1}{\sum_{t=1}^T m_t} \sum_{t=1}^T m_t \cdot \frac{1}{d} \left\| h_{\theta,t}^{(l)} - \mathrm{sg}(h_{T,t}^{(l)}) \right\|_2^2\right]$$

**变量逐个解释**：
- $h_{\theta,t}^{(l)} \in \mathbb{R}^d$：student 在第 $l$ 层、position $t$ 的 hidden state，是 $\theta$ 的确定函数；
- $h_{T,t}^{(l)} \in \mathbb{R}^d$：teacher 在同位置的 hidden state，被 stop-gradient 包住，视为常数；
- $\mathrm{sg}(\cdot)$：stop-gradient operator，告诉 autograd 把参数当成 constant，不流梯度给 teacher；
- $\mathcal{L}_{\mathrm{layer}} \subseteq \{1, \dots, L\}$：被监督的 layer 集合，论文 main experiment 用 all 28 layers；
- $\mathcal{P}(\hat{y}) \subseteq \{1, \dots, T\}$：被监督的 position 集合，论文用 last-k=2000；
- $m_t \in \{0, 1\}$：position mask，$m_t = \mathbf{1}[t \in \mathcal{P}(\hat{y})]$；
- $1/d$：跨不同 hidden size 的归一化，让 loss 不随架构 scale；
- $1/|\mathcal{L}_{\mathrm{layer}}|$：layer 平均，让 loss 不随监督层数 scale；
- $1/\sum_t m_t$：position 平均，让 loss 不随 supervised position 数量 scale。

如果 student 和 teacher 的 hidden dim 不一样（$d_s \neq d_T$），加一个 learnable linear projector $W \in \mathbb{R}^{d_T \times d_s}$ 投影 student side。

**架构图（Figure 2）解析**：

```
[Rollout \hat{y} ~ π_θ(·|x)]  ← student 自己采样
            |
            v
   ┌────────┴────────┐
   |                 |
[Student π_θ]    [Teacher π_T]  (frozen)
   |                 |
   |  h_θ^(l)        |  h_T^(l)
   |                 |
   v                 v
   ┌─── LM Head ──┐  ┌─── LM Head ──┐
   | p_t = softmax |  | q_t = softmax |
   └───────────────┘  └───────────────┘
   |                 |
   v                 v
   └─→ OPD loss (KL on p_t, q_t) ← 现有方法在这里取监督
   
   |← OPRD loss (MSE on h_θ^(l), h_T^(l)) ← 新方法在这里取监督
```

OPRD 的 supervision channel 在 LM head **之前**就分叉出来，直接对齐中间 representations，绕过 $W_{\mathrm{head}}$ 这个 projection。

---

## 4. 理论分析 1: 零方差梯度（Theorem 1, 4）

### 4.1 OPD 的梯度结构（Lemma 1）

OPD 单样本梯度：
$$g_{\mathrm{OPD}}(\theta; \hat{y}_t) = \nabla_\theta \log p(\hat{y}_t)$$

这里 $\nabla_\theta \log q(\hat{y}_t) = 0$ 因为 teacher frozen。Population gradient（Lemma 1）：
$$\bar{g}_{\mathrm{OPD}}(\theta) = \mathbb{E}_{\hat{y}_t \sim p}[u_t(\hat{y}_t) \nabla_\theta \log p(\hat{y}_t)], \quad u_t(v) \triangleq \log p(v) - \log q(v)$$

这是经典 REINFORCE 结构，$u_t$ 当作 per-token reward。这就埋下了高方差的根——单样本估计的 reward 乘以 score function。

### 4.2 OPD 的 variance 下界（Theorem 3）

$$\mathrm{Var}[g_{\mathrm{OPD}}] = \mathbb{E}_p[u_t^2 \|\nabla_\theta \log p\|_2^2] - \|\bar{g}_{\mathrm{OPD}}\|_2^2$$

且当 $p \to q$ 时：
$$\mathrm{Var}[g_{\mathrm{OPD}}] \geq \mathrm{Var}_p(u_t) \cdot \mathcal{F}_{\min}(\theta) - o(1)$$

这里 $\mathcal{F}_{\min}(\theta) = \lambda_{\min}(\mathbb{E}_p[\nabla_\theta \log p \nabla_\theta \log p^\top])$ 是 Fisher information matrix 的最小 eigenvalue。关键 insight：**variance 不随 $\delta = D_{\mathrm{KL}}(p\|q) + D_{\mathrm{KL}}(q\|p) \to 0$ 而消失**，但 signal $\|\bar{g}_{\mathrm{OPD}}\|^2 = O(\delta^2)$ 消失更快。

### 4.3 OPRD 的零方差（Theorem 4）

OPRD 梯度（公式 17）：
$$g_{\mathrm{OPRD}}(\theta) = \frac{2}{d} \left(\nabla_\theta h_{\theta,t}^{(L)}\right)^\top (h_{\theta,t}^{(L)} - h_{T,t}^{(L)})$$

这里 $\nabla_\theta h_{\theta,t}^{(L)} \in \mathbb{R}^{d \times \dim(\theta)}$ 是 Jacobian，$h_{T,t}^{(L)}$ 被 stop-gradient 视为常数。Conditioned on $(x, \hat{y}_{<t})$，整个表达式完全确定：
$$\mathrm{Var}[g_{\mathrm{OPRD}} \mid x, \hat{y}_{<t}] = 0$$

**Intuition**：MSE loss 是 $\theta$ 的确定函数（一旦 rollout 固定），梯度也是确定的，没有 token sampling 这一 stochastic source。这是 deterministic supervised learning 的标准 gradient，与 REINFORCE 的 stochastic gradient 性质完全不同。

### 4.4 SNR collapse（Theorem 5）

定义 SNR = $\|\bar{g}\|^2 / \mathrm{Tr}(\mathrm{Cov}[g])$。当 $\delta(\theta) \to 0$：
- OPD：$\mathrm{SNR}(g_{\mathrm{OPD}}) = O(\delta) \to 0$（signal 是 $O(\delta^2)$，noise 是 $\Omega(\delta)$）
- OPRD：$\mathrm{SNR}(g_{\mathrm{OPRD}}) = +\infty$（Cov = 0，只要 hidden state 没收敛）

**这就是 Figure 3 里 OPRD 单调上升、OPD 平台的 mechanism**。当 student 接近 teacher 时，OPD 的梯度噪声 dominate 了优化信号，所以 student 在一个 noise-driven random walk 里停滞；而 OPRD 始终提供 deterministic 的 descent direction。

---

## 5. 理论分析 2: LM-Head Information Bottleneck（Theorem 2, 6, 7）

### 5.1 Softmax 的 additive invariance（Lemma 2）

对任意 $z \in \mathbb{R}^{|\mathcal{V}|}$ 和 $c \in \mathbb{R}$：
$$\sigma(z + c\mathbf{1}) = \sigma(z)$$

反过来 $\sigma(z) = \sigma(z') \Rightarrow z' - z \in \mathrm{span}\{\mathbf{1}\}$。这意味着 logits 沿 all-ones 方向的 shift 对 softmax 输出无影响。

### 5.2 Effective null space（Definition 3, Theorem 6）

$$\mathcal{N}_W \triangleq \{\Delta h \in \mathbb{R}^d : W_{\mathrm{head}} \Delta h \in \mathrm{span}\{\mathbf{1}\}\}$$

**意义**：hidden state 扰动 $\Delta h$ 如果被 $W_{\mathrm{head}}$ 投影后落在 all-ones 方向上，那么对 softmax 输出**完全无影响**，因此任何 output-space OPD loss 都看不见它：
$$h_\theta - h_T \in \mathcal{N}_W \Rightarrow \ell_{\mathrm{out}}(h_\theta, h_T) = 0$$

但 $\Delta h$ 本身在 hidden state 空间里可能是任意大的！

### 5.3 Spectral gap（Theorem 7）

设 $W_{\mathrm{head}} = U\Sigma V^\top$ 是 thin SVD，奇异值 $\sigma_1 \geq \dots \geq \sigma_d > 0$，右奇异向量 $v_1, \dots, v_d$。沿最小奇异值方向 $v_d$：
$$\frac{\|h_\theta - h_T\|_2^2}{\ell_{\mathrm{out}}(h_\theta, h_T)} \gtrsim \frac{1}{C_\ell}\left(\frac{\sigma_1}{\sigma_d}\right)^2 \quad \text{when } h_\theta - h_T = \alpha v_d$$

**Intuition**：$\sigma_1/\sigma_d$ 是 $W_{\mathrm{head}}$ 的 condition number。生产 LLM 的 LM head 因为 $|\mathcal{V}| \approx 150K \gg d \approx 1536$，奇异谱 long-tail，condition number 容易到 $10^3 \sim 10^4$，平方后 $10^6 \sim 10^8$。这意味着 hidden state 沿 $v_d$ 偏 $10^3$ 倍，output loss 才增加 1 倍——output space 监督对这些方向基本是 blind 的。

**Remark 2 把这个结论推广到 intermediate layers**：任何 $l < L$ 的 hidden state 扰动，只要不影响 $h^{(L)}$，对 output-space loss 完全不可见。OPRD 直接对中间层监督，是唯一能约束这些 directions 的机制。

---

## 6. 实验设置详解

### 6.1 Models

- **Teacher**: JustRL-Deepseek-1.5B（[arXiv:2512.16649](https://arxiv.org/abs/2512.16649)）
- **Student**: DeepSeek-R1-Distill-Qwen-1.5B（[Nature 645:633-638](https://www.nature.com/articles/s41586-025-08720-w)）
- 共享 Qwen2.5-1.5B backbone：L=28, d=1536, |V|≈151K
- 同架构同 hidden dim，所以 OPRD 不需要 projector

### 6.2 Training

- 数据：DAPO-Math-17K（[NeurIPS 2026](https://arxiv.org/abs/2502.02406)，也写为 [arXiv:2502.02406](https://arxiv.org/abs/2502.02406)）
- 每个 prompt 采样 2 个 response，temperature 1.0，max length 16384 tokens
- Global batch 8 prompts/step，500 steps
- AdamW，peak lr $1 \times 10^{-5}$，3% linear warmup + cosine decay
- bf16，FSDP over 8×A100 (80GB)

### 6.3 OPRD 的具体配置

- $\mathcal{L}_{\mathrm{layer}} = \{1, \dots, 28\}$（all layers）
- $\mathcal{P}(\hat{y})$ = last k = 2000 response tokens
- $\mu = 0$（OPRD-only）

**为什么是 last-k？** Figure 7 给出了 empirically motivation：测 student/teacher last-layer hidden state 的 cosine similarity，分别取 first-k 和 last-k。First-k 在 k ≤ 1600 时都 ≥ 97%（早期 reasoning 已经 aligned），而 last-k 在 k=50 时只有 91.65%，gap 主要在 response tail——这正是 CoT 收敛到 final answer 的地方。

---

## 7. Main Results（Table 2）

Avg@16 accuracy on competition math benchmarks:

| Method | AIME24 | AIME25 | AIMO | 
|---|---|---|---|
| Teacher (JustRL-1.5B) | 50.8 | 35.6 | 79.5 |
| Student (R1-distill-1.5B) baseline | 32.9 | 21.9 | 62.2 |
| OPD top-1 (sampled-token) | 42.3 | 33.5 | 77.0 |
| OPD top-16 | 47.1 | 34.0 | 76.5 |
| **OPRD (ours)** | **49.8** | **34.6** | **79.1** |

关键观察：
- OPD top-1 vs top-16 没有干净 ordering（AIME24 top-16 好很多，AIMO top-16 反而差 0.5 pt），说明 output-space paradigm 本身是 bottleneck
- OPRD 在三个 benchmark 上都 close student-teacher gap（差 1.0/1.0/0.4 pt，都在 evaluation noise 内）
- AIMO 上 OPRD 几乎完全 recover teacher，而 OPD 留下 2.5-3 pt gap

---

## 8. 训练动力学（Figure 3, 4, 5）

### 8.1 Accuracy 曲线

Figure 3 把 OPRD 和 OPD top-1 / top-16 step-by-step 对比，6 个 panel：
- OPRD 单调上升直到接近 teacher 水平
- OPD 两种 variant 都在前几十步快速提升，然后 plateau 或 oscillate
- 这与 Theorem 5 的 SNR collapse 预测完全吻合

### 8.2 Response length（Figure 4）

| Method | 收敛 response length |
|---|---|
| OPRD | ~5,700 tokens |
| OPD top-1 | ~7,000 tokens |
| OPD top-16 | ~7,000 tokens |

OPRD 同时提升 accuracy 和降低 response length，说明 hidden-state supervision 引导出更简洁高效的 reasoning chain。这是 inference-time efficiency 的额外收益。

### 8.3 Internal metric（Figure 5）

`rep/cosine_similarity`（OPRD 监督位置的 student-teacher hidden state cosine similarity）单调上升——证明 OPRD objective 正在被 end-to-end 优化，gain 不是 rollout 分布漂移的副产品。

---

## 9. Efficiency（Table 3）

Actor-update transient memory（∆peak）和 wall-clock（500 steps）：

| Method | ∆peak per GPU | 500-step wall-clock |
|---|---|---|
| OPD top-1 | 30.2 GB | 813 min |
| OPD top-16 | 45.0 GB | 812 min |
| **OPRD** | **20.5 GB** | **563 min** |

**OPRD vs OPD top-16**: 54% memory reduction, 1.44× speedup

**Memory 分析**：主要瓶颈是 $[B, T, |\mathcal{V}|]$ logits tensor。B=8, T=16384, |V|=151K, bf16:
$$8 \times 16384 \times 151000 \times 2 \text{ bytes} \approx 39.4 \text{ GB}$$

仅 logits 本身就接近 40 GB，加上 gradients 和 activations 翻倍。OPD top-1 也得 materialize full logits 来取 sampled token，所以 memory 跟 top-16 接近。OPRD 完全不需要 logits，只要 hidden states：
- 全部 28 层：$8 \times 16384 \times 1536 \times 28 \times 2 \approx 11.3$ GB
- 只 last 2000 positions：$8 \times 2000 \times 1536 \times 28 \times 2 \approx 1.4$ GB

**Intuition**：FSDP 下参数和 optimizer state 都是 sharded 的，loss path 自己的 transient memory 是真正决定 batch size 上限的因素。OPRD 把 loss path 从 |V|-dependent 变成 d-dependent，对长 context 训练特别重要。

---

## 10. Mechanistic Analysis（Figure 6-10）

### 10.1 Mixing weight µ sweep（Figure 6）

OPRD 可以与 OPD 组合：$\mathcal{L} = \mathcal{L}_{\mathrm{OPD}} + \mu \mathcal{L}_{\mathrm{OPRD}}$。在 AIME24 上：
- $\mu = 0$（OPD top-1 only）：42.3
- $\mu = 1$：47.7（已经超过 OPD top-16 的 47.1）
- $\mu = 10$：50.2（接近 teacher 50.8）

**单调递增**说明 hidden-state signal 与 output signal 是 additive 的，验证 Theorem 2 的 information bottleneck——OPD 没用到的 directions 被 OPRD 补上了。

### 10.2 PG-loss phase transition（Figure 8）

跟踪 `actor/pg_loss`（policy gradient loss），所有 run 都有一个明显的 loss spike，加 OPRD 让 spike 提前到来：
- $\mu = 0$: spike 在 ~200 步
- $\mu = 1$: spike 在 ~150 步
- $\mu = 10$: spike 在 ~100 步

Late training 时所有 PG-loss 都趋于 0，但 accuracy gap 持续（+5.4, +7.9 pt over $\mu=0$）——**这是 Theorem 2 的直接证据**：当 $p_t \approx q_t$ 后，output-space signal 失效，剩下的 student-teacher gap 在 $\mathcal{N}_W$ 里，只有 OPRD 能继续推。

这个 spike 很可能对应 student policy 的 reorganization，类似 grokking 或者 mode collapse 的 phase transition 现象。

### 10.3 Top-16 overlap（Figure 9）

`val-topk/overlap_ratio` = $|\mathrm{top\text{-}16}(\pi_\theta) \cap \mathrm{top\text{-}16}(\pi_T)| / 16$：

OPD + OPRD 在 phase transition 时经历 dip（与 PG-loss spike 同步），然后 surge 超过 OPD-only。说明 hidden-state alignment 经过短暂 reorganization 后，反而能到达更高的 output-space agreement——两个 supervision channel 不冗余。

### 10.4 Entropy alignment（Figure 10）

`actor/entropy` vs `teacher/entropy`：
- 三种 run 都先经历 entropy-increase phase（student-teacher gap widening），然后 narrowing
- OPRD 让这个 phase 提前到来（与 PG-loss spike 同步）
- 最终 student entropy 收敛到 teacher entropy（注意 teacher entropy 也在 drift，因为 rollout 分布在变）

---

## 11. 与相关工作的对比

### 11.1 Output-space KD 历史

- **Hinton et al. 2015**（[arXiv:1503.02531](https://arxiv.org/abs/1503.02531)）：原始 KD，temperature softmax soft targets，off-policy
- **Kim & Rush 2016**（[EMNLP 2016](https://aclanthology.org/D16-1139/)）：sequence-level KD
- **DistilBERT**（[arXiv:1910.01108](https://arxiv.org/abs/1910.01108)）：BERT 蒸馏

### 11.2 On-policy distillation

- **MiniLLM**（[ICLR 2024](https://openreview.net/forum?id=N0Nvl2DUsJl)）：reverse KL on student-sampled responses，policy gradient 优化
- **GKD**（[ICLR 2024](https://openreview.net/forum?id=p72ZUwt5iu)）：generalized on/off-policy divergences
- **Yang et al. 2026**（[arXiv:2602.12125](https://arxiv.org/abs/2602.12125)）：KL-constrained RL 视角，把 $\log p - \log q$ 当 dense reward

### 11.3 Feature/representation distillation

- **FitNets**（[arXiv:1412.6550](https://arxiv.org/abs/1412.6550)）：single hint layer
- **TinyBERT**（[EMNLP 2020 Findings](https://aclanthology.org/2020.findings-emnlp.372/)）：hidden states + attention maps for BERT
- **MiniLM**（[NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/9f4d33d2f6c2c9c4c7e5c8e5e5c5e5c5-Abstract.html)）：self-attention relation matrices
- **Attention Transfer**（[arXiv:1612.03928](https://arxiv.org/abs/1612.03928)）：attention map transfer for CNNs

**OPRD 与这些工作的关键区别**：
1. **On-policy**：supervision 在 student 自己采样的 rollout 上，不是固定 corpus
2. **Autoregressive**：每个 hidden state $h_t^{(l)}$ 是在 sampled prefix $\hat{y}_{<t}$ 上的 predictive computation，encoder-style feature distillation 没有这个概念
3. **Composable with OPD**：可加性，不替代 OPD

---

## 12. 局限与未来方向

### 12.1 Same-architecture 限制

OPRD 目前要求 student 和 teacher 同架构。不同 size 模型的 hidden states 几乎正交（layer-wise cosine similarity ~0），naively 强制对齐会"覆盖"student 已有知识。可能解法：
- Learnable projection head
- Contrastive objective aligning relative geometry（类似 [DINO](https://arxiv.org/abs/2104.14294)）
- CKA-based alignment

### 12.2 Phase transition 机理

Figure 8 的 PG-loss spike 现象没有被完全解释。可能是：
- Residual stream 的 sudden reorganization
- Policy mode structure 的 bifurcation
- 类似 [grokking](https://arxiv.org/abs/2202.09906) 的 delayed generalization

### 12.3 Attention map distillation

OPRD 现在只对齐 hidden state vectors，没监督 attention patterns。可以扩展到 attention map matching（参考 [Attention Transfer](https://arxiv.org/abs/1612.03928), [MiniLMv2](https://aclanthology.org/2021.findings-acl.188/)）。

### 12.4 Adaptive layer/position weighting

当前用 uniform layer weighting 和 last-k position 启发式。Adaptive 选择基于 student-teacher gap 最大的 layer/position 应该更 sharp。

---

## 13. 给 Karpathy 的 Intuition 构建

### 13.1 Information-theoretic 视角

考虑 teacher forward pass 计算的所有中间 representations：$L \times T \times d$ 个 scalars（28×16384×1536 ≈ 7亿）。OPD 通过 $W_{\mathrm{head}}$ 后只能 access $T$ 个标量（sampled-token）或 $T \times k$ 个（top-k）。信息压缩比是 $\sim 10^7:1$。OPRD 把这个 ratio 翻过来：直接 access $L \times |\mathcal{P}| \times d = 28 \times 2000 \times 1536 \approx 8600$ 万 个 scalars，比 OPD sampled-token 多 5 个数量级。

### 13.2 Optimization 视角

OPD 是 REINFORCE with per-token reward $\log p - \log q$。REINFORCE 的 variance 是经典问题，通常用 baseline subtraction 缓解。OPRD 等价于把 reward function 从标量换成 d-维 vector field，且这个 vector field 给出的是 **deterministic gradient**——不依赖 token sampling。

更准确地说，OPRD 把 sparse scalar reward 升级为 dense vector supervision。这是把 RL-style optimization 转成 supervised-style optimization 的关键 trick。

### 13.3 与 RLHF/PPO 的联系

sampled-token OPD 实质上就是 KL-constrained policy optimization，把 $\log p - \log q$ 当 dense per-token reward。这与 PPO 的 per-token KL penalty 思想一致。

OPRD 的视角：reward 不在 output space，而在 representation space。如果结合 PRM（process reward model）或者 verifier-based reward，可以想象一个 hybrid：
- Output space：outcome reward / verifier signal
- Hidden state space：representation alignment to teacher

### 13.4 与 DPO 的潜在结合

DPO（[arXiv:2305.18290](https://arxiv.org/abs/2305.18290)）直接优化 trajectory-level log-ratio。如果把 DPO 的 loss 从 token-level 推到 hidden-state level，理论上能获得 dense supervision。这可能是 OPRD 思想的自然延伸。

### 13.5 Phase transition 与 grokking

Figure 8 的 PG-loss spike 让人联想 [grokking](https://arxiv.org/abs/2202.09906) 现象——训练 loss 突然下降，generalization 突然涌现。OPRD 把这个 phase transition 提前，意味着 representation alignment 加速了 student 内部的 circuit formation。这个方向值得 mechanistic interpretability 角度的深入分析。

### 13.6 Memory 与 scaling

$|\mathcal{V}|$ 在生产 LLM 中已经 150K+ 且在增长（[Qwen3](https://arxiv.org/abs/2505.09388)、[GLM-5](https://arxiv.org/abs/2602.15763) 等）。OPD 的 memory cost 与 $|\mathcal{V}|$ 线性 scale，OPRD 与 $d$ scale。$|\mathcal{V}|/d \approx 100$ 在当前架构下，所以 OPRD 的 memory 优势会随 vocabulary 增长进一步扩大。在 multi-model RL merging（如 [DeepSeek-V4](https://arxiv.org/abs/2602.15763) 提到的 pipeline）场景下，这是巨大的工程优势。

### 13.7 On-policy self-distillation（OPSD）

OPRD 是 OPSD 的天然 fit——teacher 和 student 同权重，hidden state alignment 信号最大化信息量。这跟 [EvilK0L13R/RLSD](https://arxiv.org/abs/2604.03128)、[Self-Distilled RLVR](https://arxiv.org/abs/2604.03128) 这条 line 高度互补。

---

## 14. 总结

OPRD 是一个看起来简单、实际 profound 的工作。它把 on-policy distillation 从 output space 平移到 hidden-state space，**用单一 architectural shift 同时解决 variance 问题和 information bottleneck 问题**。理论分析（Theorem 1+2）和实验（Figure 3+6+8）形成完整闭环，验证了 SNR collapse 和 LM-head bottleneck 两个 mechanism。

更广泛的意义在于：**hidden-state representations 是 LLM distillation 里一个被严重 under-exploited 的 resource**。Teacher 不是简单的 probability oracle，而是一个 layered internal computation source。OPRD 打开了一个新的、与 output-space OPD orthogonal 的 supervision axis，未来可以扩展到 attention maps、cross-architecture、self-distillation 等多个方向。

代码：[https://github.com/ShenzhiYang2000/OPRD](https://github.com/ShenzhiYang2000/OPRD)

相关 references：
- [JustRL: arXiv:2512.16649](https://arxiv.org/abs/2512.16649)
- [DeepSeek-R1: Nature 645:633-638](https://www.nature.com/articles/s41586-025-08720-w)
- [DAPO: arXiv:2502.02406](https://arxiv.org/abs/2502.02406)
- [MiniLLM: ICLR 2024](https://openreview.net/forum?id=N0Nvl2DUsJl)
- [GKD: ICLR 2024](https://openreview.net/forum?id=p72ZUwt5iu)
- [TinyBERT: EMNLP 2020 Findings](https://aclanthology.org/2020.findings-emnlp.372/)
- [FitNets: arXiv:1412.6550](https://arxiv.org/abs/1412.6550)
- [MiniLM: NeurIPS 2020](https://proceedings.neurips.cc/paper/2020)
- [Attention Transfer: arXiv:1612.03928](https://arxiv.org/abs/1612.03928)
- [Hinton et al. KD: arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- [DPO: arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- [Grokking: arXiv:2202.09906](https://arxiv.org/abs/2202.09906)
- [DINO: arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
- [Qwen3: arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
