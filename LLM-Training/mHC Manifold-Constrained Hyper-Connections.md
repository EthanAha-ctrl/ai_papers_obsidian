---
source_pdf: mHC Manifold-Constrained Hyper-Connections.pdf
paper_sha256: d849f4709ed29bfb2751f51e24dc836403f31271d1764dddfba0768d326a75d9
processed_at: '2026-08-05T18:05:26-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话聊聊 mHC 这篇 paper

---

## 一句话概括

Hyper-Connections (HC) 这玩意儿效果好但是一 scale up 就炸，DeepSeek 这帮人发现根因是 residual stream 的 mixing matrix 没约束，signal 会指数爆炸。他们的 fix 很优雅——把 mixing matrix 用 Sinkhorn-Knopp 投影到 doubly stochastic matrix 上，也就是 Birkhoff polytope，这样信号守恒、梯度不炸，还能 keep HC 的拓扑表达力。工程上压到只有 6.7% overhead，在 27B MoE 上跑通了。

---

## 这个领域到底在搞什么

先 zoom out 一下。

过去十年所有 deep learning 架构基本上都长一个样：一堆 block，block 之间用 residual connection 串起来。block 内部怎么折腾（conv / attention / FFN / MoE）那是 micro design，但 block 之间怎么连、信号怎么流——这是 macro design。

macro design 这十年基本没变过。ResNet [1] 那条 $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l)$ 写起来简单，但它太成功了，没人敢动。直到 2024 年 Hyper-Connections [2] 出来，说咱把 residual stream 从 1 条扩展到 $n$ 条，让 block 之间能玩花样——结果效果确实好，loss 掉一截，downstream 全面提升。

问题来了：HC 在小模型上乖得很，一上规模就闹脾气。loss spike、gradient spike、训练直接崩。DeepSeek 自己训 27B 的时候踩了坑，于是有了 mHC 这篇 paper 来填坑。

---

## HC 到底做了什么

普通 ResNet 是单条 stream：

$$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$$

$\mathbf{x}_l \in \mathbb{R}^C$，$C$ 是 hidden dim，$\mathcal{F}$ 是 block function。

HC 把 stream 扩成 $n$ 条（论文取 $n=4$），$\mathbf{x}_l \in \mathbb{R}^{n \times C}$，然后引入三个 learnable matrix：

$$\mathbf{x}_{l+1} = \mathcal{H}_l^{\text{res}} \mathbf{x}_l + \mathcal{H}_l^{\text{post}\top} \mathcal{F}(\mathcal{H}_l^{\text{pre}} \mathbf{x}_l, \mathcal{W}_l)$$

变量逐个说：
- $\mathcal{H}_l^{\text{pre}} \in \mathbb{R}^{1 \times n}$：把 $n$ 条 streams 加权 sum 成一条，喂给 $\mathcal{F}$（attention 或 FFN 仍只吃单 stream，FLOPs 不变）。
- $\mathcal{H}_l^{\text{post}} \in \mathbb{R}^{1 \times n}$：把 $\mathcal{F}$ 的输出广播回 $n$ 条 streams，带不同权重。
- $\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$：residual stream 之间的 mixing matrix——这条最关键，它决定 streams 之间怎么互混。

ablation（Table 1）说 $\mathcal{H}^{\text{res}}$ 贡献 -0.022 loss gap，另外俩加起来才 -0.005。所以**真正起作用的就是那个 $n \times n$ mixing matrix**。

---

## 为什么 HC 会炸

把单层 HC 递归展开 $L-l$ 层：

$$\mathbf{x}_L = \left(\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}\right) \mathbf{x}_l + \sum_{i=l}^{L-1} (\cdots) \mathcal{F}(\cdots)$$

那个 identity 项（直接从浅层传到深层的信号）变成了 $\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$——一堆 learnable matrix 连乘。

ResNet 之所以稳定，是因为这个 identity 项恒等于 $\mathbf{I}$，信号无损通过。HC 把它换成了 unconstrained matrix product。每个 $\mathcal{H}^{\text{res}}$ 的 spectral norm 没人管，连乘 30 层后要么爆炸要么消失。

论文测了个指标叫 **Amax Gain Magnitude**：composite matrix 的 max row sum（前向）和 max column sum（反向）。理想值是 1，HC 实测在 27B 上峰值 **3000**。三个数量级的偏差，难怪训不动。

这跟 RNN 的 gradient explosion / vanishing 是一码事，只不过 RNN 是 time 方向，HC 是 depth 方向。

---

## mHC 的 fix：投影到 Birkhoff Polytope

核心 idea 一行话：把 $\mathcal{H}^{\text{res}}$ 限制成 **doubly stochastic matrix**——行和为 1、列和为 1、元素非负。

数学上：

$$\mathcal{P}_{\mathcal{M}^{\text{res}}}(\mathcal{H}) := \left\{ \mathcal{H} \in \mathbb{R}^{n \times n} \mid \mathcal{H} \mathbf{1}_n = \mathbf{1}_n,\ \mathbf{1}_n^\top \mathcal{H} = \mathbf{1}_n^\top,\ \mathcal{H} \geq 0 \right\}$$

下标 $n$ 表示向量维度，$\mathbf{1}_n$ 是全 1 向量，三个条件分别是行和、列和、非负。

这个集合叫 **Birkhoff polytope** [3]，是所有 $n \times n$ permutation matrix 的 convex hull。

### 三个性质，每个都踩在 stability 的痛点上

**1. Spectral norm 有界**

doubly stochastic matrix 的 $\|\mathcal{H}\|_2 \leq 1$（因为 $\|\mathcal{H}\|_1 = 1$ 且 $\|\mathcal{H}\|_\infty = 1$，由 matrix norm 不等式 $\|\mathcal{H}\|_2 \leq \sqrt{\|\mathcal{H}\|_1 \|\mathcal{H}\|_\infty}$ 得出）。

所以 $\mathcal{H} \mathbf{x}$ 是 **non-expansive**——不会放大 signal。这一条就解决了 HC 的爆炸问题。

**2. 矩阵乘法下闭合**

如果 $A, B$ 都是 doubly stochastic，那 $AB$ 也是。证：
- $(AB) \mathbf{1} = A(B\mathbf{1}) = A\mathbf{1} = \mathbf{1}$，行和还是 1。
- $\mathbf{1}^\top (AB) = (\mathbf{1}^\top A) B = \mathbf{1}^\top B = \mathbf{1}^\top$，列和还是 1。
- 非负性显然。

这意味着 composite mapping $\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$ 仍然 doubly stochastic——**整条 depth 方向都守恒**。这是 mHC 比那些"只在单层约束"方案强的地方。

**3. Birkhoff-von Neumann decomposition**

任意 doubly stochastic matrix 可以写成 permutation matrices 的 convex combination：$\mathcal{H} = \sum_\pi p_\pi P_\pi$，$p_\pi \geq 0$，$\sum p_\pi = 1$。

直觉上：每层的 $\mathcal{H}^{\text{res}} \mathbf{x}$ 相当于 **"随机挑几个 stream permutation 然后加权平均"**。这是一种 implicit ensemble of stream routings。这跟 DenseNet [4] 的 implicit ensemble interpretation 遥相呼应。

还有一个我觉得更漂亮的视角：doubly stochastic matrix 是 **Markov transition matrix 的特殊版本**——它额外保证 stationary distribution 是 uniform。所以无论 initial stream 分布如何，多层 mixing 后会趋向 uniform distribution，相当于一个 ergodic prior 强迫 streams 融合。

---

## 怎么投影：Sinkhorn-Knopp

给定 unconstrained $\tilde{\mathcal{H}}^{\text{res}}$，要把它弄到 Birkhoff polytope 上。方法：

**Step 1**：取 $\exp(\tilde{\mathcal{H}}^{\text{res}})$ 保证所有元素为正。用 exp 而不是 ReLU 是为了平滑——这是 entropic optimal transport [5] 的标准操作。

**Step 2**：交替 normalize 行和列：

$$\mathbf{M}^{(t)} = \mathcal{T}_r(\mathcal{T}_c(\mathbf{M}^{(t-1)}))$$

- $\mathcal{T}_r(\mathbf{M})_{ij} = \mathbf{M}_{ij} / \sum_j \mathbf{M}_{ij}$，行 normalize
- $\mathcal{T}_c(\mathbf{M})_{ij} = \mathbf{M}_{ij} / \sum_i \mathbf{M}_{ij}$，列 normalize

迭代 $t_{\max} = 20$ 次收敛。Sinkhorn-Knopp [6] 是 1967 年的老算法，收敛是 geometric 的，20 次在 fp32 下基本够用。

实测（Fig. 7）：单层 gain ≈ 1（轻微偏差），composite 30 层后 ≈ 1.6。对比 HC 的 3000，**降了三个数量级**。

---

## 完整 forward pass

把公式拆开看完整 flow：

$$\tilde{\mathcal{H}}_l^{\text{res}} = \alpha_l^{\text{res}} \cdot \text{mat}(\vec{\mathbf{x}}_l' \varphi_l^{\text{res}}) + \mathbf{b}_l^{\text{res}}$$

- $\vec{\mathbf{x}}_l = \text{vec}(\mathbf{x}_l) \in \mathbb{R}^{1 \times nC}$，flatten 保留全 context
- $\vec{\mathbf{x}}_l' = \text{RMSNorm}(\vec{\mathbf{x}}_l)$
- $\varphi_l^{\text{res}} \in \mathbb{R}^{nC \times n^2}$：linear projection，输出 $n^2$ 个值
- $\text{mat}(\cdot)$：$\mathbb{R}^{1 \times n^2} \to \mathbb{R}^{n \times n}$ reshape
- $\alpha_l^{\text{res}}$：scalar gate，init 0.01
- $\mathbf{b}_l^{\text{res}}$：learnable static bias

然后：

$$\mathcal{H}_l^{\text{res}} = \text{Sinkhorn-Knopp}(\tilde{\mathcal{H}}_l^{\text{res}})$$

类似地：

$$\mathcal{H}_l^{\text{pre}} = \sigma(\tilde{\mathcal{H}}_l^{\text{pre}}), \quad \mathcal{H}_l^{\text{post}} = 2\sigma(\tilde{\mathcal{H}}_l^{\text{post}})$$

$\mathcal{H}^{\text{pre}}$ 用 sigmoid 约束到 $[0, 1]$，是 streams 的 convex attention weight。$\mathcal{H}^{\text{post}}$ 用 $2\sigma$ 约束到 $[0, 2]$，乘 2 让 layer output 能"穿过" residual stream——相当于 gated residual scaling 在 mid-range 初始化，有点 LayerScale [7] 的味道。

注意 $\mathcal{H}^{\text{pre}}, \mathcal{H}^{\text{post}}$ 都只是 non-negativity constraint（属于"special manifold projection"），真正的 doubly stochastic 约束只落在 $\mathcal{H}^{\text{res}}$ 上。

---

## 工程实现：怎么把 overhead 压到 6.7%

这是 paper 最务实的部分。HC 表面上 FLOPs 没增加多少，但 **memory access 开销** [8] 大得吓人——n=4 时每层 residual 维护的 I/O 是 ResNet 的 5-10 倍。GPU 上 memory wall 比 compute wall 严重多了。

### Memory access 分析（Table 2）

ResNet 每层 residual merge：read $2C$, write $C$。

HC 每层：read $(5n+1)C + n^2 + 2n$，write $(3n+1)C + n^2 + 2n$。n=4 时 read $\approx 21C$, write $\approx 13C$。

这玩意儿不做 kernel fusion 直接训不动。

### Kernel Fusion

三个 fused kernel：

**Kernel A**：算 $\tilde{\mathcal{H}}$ 和 RMSNorm norm $r$。把 divide-by-norm 重排到 matmul 之后（数学等价），forward 和 backward 都 fuse 成单 kernel，省 $\vec{\mathbf{x}}_l$ 的重复 reload。

**Kernel B**：算 gating $\alpha$、加 bias、做 sigmoid——这些操作只在 $n^2 + 2n = 24$ 个元素上跑（n=4），fusion 主要是为了省 kernel launch overhead（每次 launch 5-20μs，对 small ops 占比很大）。

**Kernel C**：Sinkhorn-Knopp 20 次迭代全在一个 kernel 里完成，中间状态不写回 HBM。Backward 时 on-chip recompute 整条迭代链，traverse Jacobian-vector product。

用 [TileLang](https://arxiv.org/abs/2504.17577) [9] 实现，这是 DeepSeek 自己搞的 composable tiled programming framework，比手写 CUDA 高效，比 Triton 在 mixed precision 下更灵活。

### Recomputing

中间 activations 不存，backward 时重算。对 $L_r$ 连续层只存首层 input $\mathbf{x}_{l_0}$。

Memory 公式：

$$\text{Memory} \approx nC \cdot \left\lceil \frac{L}{L_r} \right\rceil + (n+2)C \cdot L_r$$

- 第一项：persistent storage，每 $L_r$ 层存一次
- 第二项：transient recomputation buffer

求导取最优：

$$L_r^* \approx \sqrt{\frac{nL}{n+2}}$$

n=4、L=30 时 $L_r^* \approx 4.5$，取 5。刚好和 pipeline stage 层数对齐——所以 recompute 边界与 pipeline stage 同步，避免跨 stage dependency。

### DualPipe overlap

[DeepSeek-V3 的 DualPipe](https://arxiv.org/abs/2412.19437) [10] 是 bidirectional pipeline schedule，把 forward/backward 的 communication 与 computation overlap。mHC 带来两个麻烦：
- $n$-stream 在 stage boundary 通信量翻 $n$ 倍
- Recompute mHC kernels 在 boundary 有额外 compute

Fix：
- MLP 的 $\mathcal{F}_{\text{post,res}}$ kernel 跑在 dedicated high-priority stream 上，避免阻塞 communication
- Attention 不用 persistent kernel，允许 preemption 让 communication 抢占
- Recompute 与 pipeline communication 解耦，因为 stage 首层 $\mathbf{x}_{l_0}$ 本地有

---

## 实验数据怎么看

### 27B 稳定性（Fig. 5）

- Baseline（普通 ResNet）：loss 平稳下降，gradient norm 稳
- HC：~12k step 突然 loss spike，gap 从 -0.04 跳到 -0.01，gradient norm 同步飙高。这就是 large-scale training 里那种"突然 NaN 要 restart"的痛
- mHC：loss gap 稳定在 -0.02 ~ -0.025，gradient norm 跟 baseline 差不多

### 下游 benchmark（Table 4，27B 模型）

| Benchmark | Baseline | HC | mHC | mHC vs HC |
|---|---|---|---|---|
| BBH (3-shot) | 43.8 | 48.9 | **51.0** | +2.1 |
| DROP (3-shot) | 47.0 | 51.6 | **53.9** | +2.3 |
| GSM8K (8-shot) | 46.7 | 53.2 | **53.8** | +0.6 |
| HellaSwag | 73.7 | 74.3 | **74.7** | +0.4 |
| MATH | 22.0 | **26.4** | 26.0 | -0.4 |
| MMLU | 59.0 | 63.0 | **63.4** | +0.4 |
| PIQA | 78.5 | 79.9 | **80.5** | +0.6 |
| TriviaQA | 54.3 | 56.3 | **57.6** | +1.3 |

观察：
- mHC 在 reasoning-heavy 任务上（BBH、DROP、GSM8K）相对 HC 涨得最多。论文的解释是 stability 让模型能"放心挖掘"HC 的表达力——HC 因为 unstable，实际发挥打了折扣
- MATH 上 mHC 反而比 HC 低 0.4 个点。我猜是 doubly stochastic 约束限制了某些"激进表示"的能力，HC 的 instability 偶尔能 explore 到有用的东西
- 整体看 mHC > HC > Baseline 的 hierarchy 非常稳定

### Scaling（Fig. 6）

Compute scaling 3B/9B/27B：mHC 优势在 scale 增加时 only marginally attenuated。这说明它不是 small-model 的 lucky artifact，是真 scalable 的设计。Token scaling 3B on 1T tokens：trajectory 持续 advantage。

这是 [Chinchilla](https://arxiv.org/abs/2203.15556) [11] 风格的 compute-optimal 验证，很重要——很多 trick 在小模型上有效，scale 上去就消失了，mHC 经住了考验。

### Hyper-parameter（Table 5）

27B config 关键几项：
- Total 27.0B / Active 4.14B（MoE，72 routed experts + 2 shared，6 active）
- 30 layers，dim 2560，FFN dim 1536
- MLA attention（DeepSeek-V2 [12] 风格），KV rank 512，RoPE dim 64
- mHC expansion $n=4$，gating init $\alpha=0.01$，Sinkhorn $t_{\max}=20$
- AdamW，lr 4e-4，weight decay 0.1，262B tokens

$\alpha=0.01$ 是 warmup-style init——model 启动时几乎是 baseline ResNet，慢慢 learn 出 stream mixing。这跟 [ReZero](https://arxiv.org/abs/2003.13660) [13] / [LayerScale](https://arxiv.org/abs/2103.17239) 的思路一脉相承：把 innovation 装在 residual path 上，initially 关到接近 identity，避免一开始就 destabilize training。

---

## 几个值得深挖的 intuition

### 1. mHC 是 ResNet 的 strict generalization

当 $n=1$，doubly stochastic condition 退化到标量 1，$\mathcal{H}^{\text{res}} = 1$，公式变成标准 ResNet。所以 mHC 在 $n=1$ 时严格等价 ResNet。这是个很优雅的 framing——它把 ResNet 的 identity mapping property 推广为"doubly stochastic identity"，从 strict identity 扩展到 Birkhoff polytope 上的任意点。

### 2. 与 Optimal Transport 的暗合

Sinkhorn-Knopp 是 entropic OT [5] 的核心。OT 问题 $\min_T \langle T, C \rangle - \epsilon H(T)$ 的解是 $T = \text{diag}(u) \exp(-C/\epsilon) \text{diag}(v)$，$u, v$ 由 Sinkhorn 迭代求出。

mHC 里没有显式 cost matrix，但 $\exp(\tilde{\mathcal{H}}^{\text{res}})$ 可以看作 $\exp(-C/\epsilon)$ with $C = -\tilde{\mathcal{H}}^{\text{res}}, \epsilon=1$。所以 mHC 实际上每层学一个 entropic OT plan，把 streams 间的 feature transport 当 OT 问题解。

这跟 [Sinkformers](https://arxiv.org/abs/2110.01607) [14]、[Neural OT](https://arxiv.org/abs/1901.05946) [15] 这些工作暗合，但用在了 macro-architecture 而不是 attention 或 generation 上。

### 3. Birkhoff Polytope 的几何

$\mathcal{B}_n$ 是 $(n-1)^2$ 维的多面体。$n=4$ 时 $\mathcal{B}_4$ 是 9 维的，有 $4! = 24$ 个顶点（24 个 permutation matrices）。$\mathcal{H}^{\text{res}}$ 在这个 9 维 polytope 里游走，gradient flow 由 Sinkhorn-Knopp 的 backward 提供（相当于 Riemannian gradient on manifold）。

直觉上：mHC 让 $\mathcal{H}^{\text{res}}$ 在一个"有界有结构"的空间内自由探索。这比 unconstrained matrix 自由度低，但换来的是 stability guarantee——典型的 bias-variance tradeoff。

### 4. Compositional Closure 是关键

如果只约束单层（比如每层 $\mathcal{H}^{\text{res}}$ 都 normalize 一下），但 composite $\prod \mathcal{H}^{\text{res}}$ 不在 manifold 上，那 depth 方向的稳定性还是没保证。

doubly stochastic matrices 在乘法下闭合，这是 Birkhoff polytope 比其他 manifold 更适合这个任务的核心原因。比如 orthogonal matrices [16] 也闭合（$\mathcal{O}(n)$ 是 group），但允许负元素，可能 signal cancellation。PSD cone 不闭合。Stiefel manifold 也不一定闭合。

**Compositional closure + non-negativity + norm bound**，这三个性质同时满足，Birkhoff polytope 几乎是唯一选择。

### 5. 表达力损失怎么办

doubly stochastic 约束让 $\mathcal{H}^{\text{res}}$ 只能做 convex combination，不能做负权或放大。这可能是 MATH 上 mHC 比 HC 低 0.4 的原因。

但论文通过 $\mathcal{H}^{\text{post}} = 2\sigma$ 部分补救——layer output 可以是 0 到 2 倍 $\mathcal{F}$ 输出，"放大"动作放在 layer output 路径而非 residual path。这是个不错的 design choice：residual path 保守（守恒），layer output 激进（可放大），分工明确。

### 6. 有限次 Sinkhorn 的累积误差

20 次迭代有 ~6% 偏差，30 层累积到 ~1.6× composite gain。理论上 bounded，但若 100B+ 模型 100+层，可能需要更多迭代或更稳定算法。可考虑：
- [Log-stabilized Sinkhorn](https://arxiv.org/abs/2006.02603) [17] 防 numerical underflow
- Newton-type methods on Birkhoff polytope
- 直接学一个 parameterization 让 matrix 天生 doubly stochastic（比如通过 softmax + 结构约束）

---

## 跟其他工作的关系

### Highway Networks [18]

Highway Networks 用 gated $\mathbf{T}\mathbf{x} + (1-\mathbf{T})\mathcal{F}(\mathbf{x})$ 实现 learnable residual。但 $\mathbf{T}$ 是 element-wise，没有 cross-stream mixing，也不保证 stability。

mHC 可以看作 **matrix-valued Highway Networks with manifold constraint**——$\mathcal{H}^{\text{res}}$ 是 matrix-valued gate，project 到 Birkhoff polytope 上。

### DenseNet [4]

DenseNet 的 dense connectivity 也有 implicit ensemble of paths 的 interpretation。mHC 的 Birkhoff-von Neumann decomposition 也是 ensemble of permutations。两者精神相通，但 DenseNet 没有 stability guarantee，mHC 通过 manifold constraint 严格 enforce 了。

### Normalization 层

BatchNorm / LayerNorm / RMSNorm 的核心功能之一是 control signal magnitude。mHC 通过 manifold constraint 实现了 **structural normalization**——不依赖 statistics，而是依赖几何约束。类似 [Weight Standardization](https://arxiv.org/abs/1911.05920) [19] 把 normalization 放到 weights 上而非 activations 上的思想。mHC 把约束放到 connection matrices 上，是更 macro-level 的 normalization。

### ReZero / SkipInit / LayerScale

这族工作用"learnable scalar init 0"实现 deep network stable training。mHC 的 $\alpha=0.01$ init 借鉴同一思想。但 mHC 更"hard core"——不仅 gate residual strength，还把 matrix structure 约束到 Birkhoff polytope，从 parameter space 上 enforce stability。

### MUDDFormer [20] / RMT [21] / DenseFormer [22]

这些都是近期的 macro-design 探索，都在扩展 residual stream 宽度或 cross-layer connectivity。它们都面临与 HC 类似的 stability 问题，但都没有系统的 manifold constraint。mHC 的 framework 可以直接 apply 到这些方法上——把它们的 mixing matrices 也 project 到 Birkhoff polytope。这是论文 Section 6 "future research" 暗示的方向。

---

## 我的看法

mHC 是一个 **insight + 数学 + 工程** 三位一体的工作。

**Insight** 在于识别出 HC 不稳定的根因不是"参数太多"或"优化器不行"，而是 residual stream mixing matrix 的 unconstrained composite 破坏了 identity mapping 的 conservation property。这个诊断很精准——Amax Gain Magnitude 3000 这个数字摆在面前，问题一目了然。

**数学** 在于选了 Birkhoff polytope 这个 manifold。它同时满足 spectral norm bound、compositional closure、geometric interpretation 三个性质，几乎是为这个场景量身定做。而且 $n=1$ 退化到 ResNet 的 elegance 给了它很强的"正统性"——它确实是 ResNet 的 generalization，不是另起炉灶。

**工程** 在于把 5-10× 的 memory overhead 压到 6.7% time overhead。这需要 kernel fusion、recompute scheduling、DualPipe overlap 三层优化，每一层都踩在 modern GPU 的实际瓶颈上。没有这部分工程，mHC 就只是个理论玩具。

限制也明显：
- doubly stochastic 约束牺牲了部分表达力（MATH 上 -0.4）
- 有限次 Sinkhorn 累积误差在超深模型上可能成问题
- 只测了 MoE，dense model 上效果未知
- 与 MoE routing 的交互没深入讨论

但我认为这篇 paper 最重要的贡献不是 mHC 本身，而是 **"manifold-constrained architecture design" 这个范式**。过去 macro-design 基本是 empirical——试一试，跑通了就发论文。mHC 提供了一个 framework：你想到一个新的 connection pattern，先想想它的 composite mapping 在什么 manifold 上，然后 enforce 那个 manifold constraint。

这跟 normalization 层当年从 BatchNorm 演化到 LayerNorm 到 RMSNorm 到 weight normalization 的路径很像——从 empirical trick 演化到 principled framework。mHC 可能是 macro-architecture design 从 empirical 走向 principled 的起点。

---

## References

[1] [He et al., 2016a - Deep Residual Learning](https://arxiv.org/abs/1512.03385)
[2] [Zhu et al., 2024 - Hyper-Connections](https://arxiv.org/abs/2409.19606)
[3] [Birkhoff-von Neumann theorem](https://en.wikipedia.org/wiki/Birkhoff%E2%80%93von_Neumann_theorem)
[4] [Huang et al., 2017 - DenseNet](https://arxiv.org/abs/1608.06993)
[5] [Cuturi, 2013 - Sinkhorn Distances / Entropic OT](https://arxiv.org/abs/1306.0894)
[6] [Sinkhorn & Knopp, 1967 - Doubly Stochastic Matrices](https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-21/issue-2/Concerning-nonnegative-matrices-and-doubly-stochastic-matrices/p10.2140/pjm.1967.21.343.full)
[7] [Touvron et al., 2021 - LayerScale](https://arxiv.org/abs/2103.17239)
[8] [Dao et al., 2022 - FlashAttention / Memory Wall](https://arxiv.org/abs/2205.14135)
[9] [Wang et al., 2025 - TileLang](https://arxiv.org/abs/2504.17577)
[10] [Liu et al., 2024b - DeepSeek-V3 / DualPipe](https://arxiv.org/abs/2412.19437)
[11] [Hoffmann et al., 2022 - Chinchilla](https://arxiv.org/abs/2203.15556)
[12] [Liu et al., 2024a - DeepSeek-V2 / MLA](https://arxiv.org/abs/2405.04434)
[13] [Bachlechner et al., 2020 - ReZero](https://arxiv.org/abs/2003.13660)
[14] [Sinkformers](https://arxiv.org/abs/2110.01607)
[15] [Neural Optimal Transport](https://arxiv.org/abs/1901.05946)
[16] [Cayley transform for orthogonal parameterization](https://arxiv.org/abs/2106.13063)
[17] [Log-stabilized Sinkhorn](https://arxiv.org/abs/2006.02603)
[18] [Srivastava et al., 2015 - Highway Networks](https://arxiv.org/abs/1505.00387)
[19] [Weight Standardization](https://arxiv.org/abs/1911.05920)
[20] [Xiao et al., 2025 - MUDDFormer](https://arxiv.org/abs/2502.12170)
[21] [Mak & Flanigan, 2025 - RMT](https://arxiv.org/abs/2506.22696)
[22] [Pagliardini et al., 2024 - DenseFormer](https://openreview.net/forum?id=kMnoh7CXrq)

---

# mHC: Manifold-Constrained Hyper-Connections 深度解析

## 1. 论文定位与背景脉络

这篇 paper 来自 DeepSeek-AI 团队（Zhenda Xie 等），发表于 2025 年，是 Hyper-Connections (HC) [Zhu et al., 2024](https://arxiv.org/abs/2409.19606) 这一新型 macro-architecture design 的改进版本。它要解决的核心矛盾是：**HC 通过扩展 residual stream 宽度获得了拓扑复杂度的提升，但代价是破坏了 ResNet 范式中那神圣不可侵犯的 identity mapping property**。

为了 build intuition，先回顾整条技术 lineage：

- **ResNet** [He et al., 2016a](https://arxiv.org/abs/1512.03385)：$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$，identity shortcut 让 gradient 直接流回任意浅层。
- **Identity Mappings in Deep Residual Networks** [He et al., 2016b](https://arxiv.org/abs/1603.05036)：证明当 shortcut 是严格 identity 且 addition 不可扰时，信息流"clean"，前向 signal 与反向 gradient 都能无损通过任意深度。
- **DenseNet** [Huang et al., 2017](https://arxiv.org/abs/1608.06993)：dense connectivity 增加信息流但破坏了 clean identity path。
- **Highway Transformer** [Chai et al., 2020](https://aclanthology.org/2020.acl-main.616/)：gating mechanism 调控 residual。
- **DenseFormer** [Pagliardini et al., 2024](https://openreview.net/forum?id=kMnoh7CXrq)：depth-weighted averaging across layers。
- **MUDDFormer** [Xiao et al., 2025](https://arxiv.org/abs/2502.12170)：multiway dynamic dense connections。
- **RMT** [Mak & Flanigan, 2025](https://arxiv.org/abs/2506.22696)：outer-product memory matrix 替换 residual stream。
- **Hyper-Connections** [Zhu et al., 2024](https://arxiv.org/abs/2409.19606)：本论文的"父"工作，用三个 learnable matrices $\mathcal{H}^{\text{pre}}, \mathcal{H}^{\text{post}}, \mathcal{H}^{\text{res}}$ 同时实现 stream 扩展与 connectivity diversification。
- **mHC**（本论文）：将 $\mathcal{H}^{\text{res}}$ 投影到 doubly stochastic matrix manifold（Birkhoff polytope）上，恢复 stability 同时保留 expressivity。

这是典型的 micro vs. macro design 二分：micro design 关心 block 内部（attention、FFN、MoE），macro design 关心 block 之间（residual stream 拓扑）。mHC 属于后者，且针对的是 DeepSeek-V3 [Liu et al., 2024b](https://arxiv.org/abs/2412.19437) 这种 MoE LLM 在 large-scale training 时的实战稳定性问题。

---

## 2. HC 的数学公式与多层递推问题

### 2.1 单层 HC 公式

$$\mathbf{x}_{l+1} = \mathcal{H}_l^{\text{res}} \mathbf{x}_l + \mathcal{H}_l^{\text{post}\top} \mathcal{F}(\mathcal{H}_l^{\text{pre}} \mathbf{x}_l, \mathcal{W}_l)$$

变量含义：
- $\mathbf{x}_l \in \mathbb{R}^{n \times C}$：第 $l$ 层输入，被扩展为 $n$ 个 stream（$C$ 是原始 hidden dim，$n$ 是 expansion rate，本论文取 $n=4$）。
- $\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$：residual stream 内部 mixing 矩阵，决定 streams 之间如何互混。
- $\mathcal{H}_l^{\text{pre}} \in \mathbb{R}^{1 \times n}$（实际是 $\mathbb{R}^{n \times \tilde{n}}$ 形式）：把 $n$-stream 聚合为单个 layer input。
- $\mathcal{H}_l^{\text{post}} \in \mathbb{R}^{1 \times n}$：把 layer output 映射回 stream。
- $\mathcal{F}$：layer function（attention 或 FFN），dim 仍为 $C$，所以 FLOPs 不变。

关键点：FLOPs 没增加多少，因为 $\mathcal{H}$ 都是 $n \times n$ 这种 small matrix，但**拓扑表达力**大幅提升。HC 的 ablation（Table 1）显示 $\mathcal{H}^{\text{res}}$ 贡献最大（loss gap -0.022），远超 pre/post（合计再 -0.005）。

### 2.2 多层递推：identity mapping 的崩塌

把 Eq.(3) 递归展开 $L-l$ 层得到 Eq.(4)：

$$\mathbf{x}_L = \underbrace{\left(\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}\right)}_{\text{composite residual mapping}} \mathbf{x}_l + \sum_{i=l}^{L-1} \left(\prod_{j=1}^{L-1-i} \mathcal{H}_{L-j}^{\text{res}}\right) \mathcal{H}_i^{\text{post}\top} \mathcal{F}(\mathcal{H}_i^{\text{pre}} \mathbf{x}_i, \mathcal{W}_i)$$

下标 $l$ 是浅层，$L$ 是深层，$\prod$ 是从右到左的矩阵乘积（$\mathcal{H}_{L-1}^{\text{res}} \cdots \mathcal{H}_l^{\text{res}}$）。

对比 ResNet 的 Eq.(2)：$\mathbf{x}_L = \mathbf{x}_l + \sum_i \mathcal{F}(\cdots)$，identity 项是裸 $\mathbf{x}_l$，能量严格守恒。而 HC 的 identity 项变成了 $\prod_i \mathcal{H}^{\text{res}}$——这是一个 **unconstrained learnable matrix product**，它的 spectral norm 没有任何 bound。

### 2.3 Amax Gain Magnitude 实证

论文定义了两个直观 metrics（Fig. 3）：

- **Forward gain**：$\max_i \left|\sum_j \left(\prod_k \mathcal{H}_k^{\text{res}}\right)_{ij}\right|$，即 composite matrix 的最大 row sum 绝对值，对应 forward signal 的 worst-case amplification。
- **Backward gain**：$\max_j \left|\sum_i \left(\prod_k \mathcal{H}_k^{\text{res}}\right)_{ij}\right|$，即最大 column sum 绝对值，对应 backprop gradient 的 worst-case amplification。

实证 27B 模型上，HC 的 composite mapping Amax Gain Magnitude 峰值达到 **~3000**（理想值是 1）。这就是 Fig. 2 中 12k step 附近 loss surge 和 gradient norm spike 的根因——这是 **exploding residual stream**，类似于 RNN 在长序列上的梯度爆炸，但发生在深度方向。

---

## 3. mHC 的核心数学：Birkhoff Polytope 与 Sinkhorn-Knopp

### 3.1 流形约束的数学定义

mHC 把 $\mathcal{H}_l^{\text{res}}$ 限制到 doubly stochastic matrix manifold：

$$\mathcal{P}_{\mathcal{M}^{\text{res}}}(\mathcal{H}_l^{\text{res}}) := \left\{ \mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n} \mid \mathcal{H}_l^{\text{res}} \mathbf{1}_n = \mathbf{1}_n,\ \mathbf{1}_n^\top \mathcal{H}_l^{\text{res}} = \mathbf{1}_n^\top,\ \mathcal{H}_l^{\text{res}} \geq 0 \right\}$$

其中 $\mathbf{1}_n$ 是 $n$ 维全 1 向量。三个条件分别表示：**行和为 1**、**列和为 1**、**元素非负**。

这个集合在数学上叫 **Birkhoff polytope** $\mathcal{B}_n$，是 $n \times n$ permutation matrices 的 convex hull（[Birkhoff-von Neumann theorem](https://en.wikipedia.org/wiki/Birkhoff%E2%80%93von_Neumann_theorem)）。当 $n=1$ 时退化到标量 1，即原始 ResNet 的 identity mapping。这是一个很优雅的"generalization"——ResNet 是 mHC 在 $n=1$ 的特例。

### 3.2 三大理论性质的 intuition

1. **Norm preservation**：doubly stochastic matrix 的 spectral norm $\|\mathcal{H}^{\text{res}}\|_2 \leq 1$（因为 $\|\mathcal{H}\|_\infty = 1$ 行和，$\|\mathcal{H}\|_1 = 1$ 列和，且 $\|\mathcal{H}\|_2 \leq \sqrt{\|\mathcal{H}\|_1 \|\mathcal{H}\|_\infty}$）。这意味着 $\mathcal{H}^{\text{res}} \mathbf{x}$ 是 **non-expansive operator**——信号不会爆炸。但需要配合 $\mathcal{H}^{\text{post}}$ 的 $2\sigma(\cdot)$ scaling（论文中乘 2 倍 sigmoid）保证非平凡解。

2. **Compositional closure**：doubly stochastic matrices 在矩阵乘法下闭合。证明很短：若 $A, B \in \mathcal{B}_n$，则 $(AB) \mathbf{1}_n = A(B\mathbf{1}_n) = A\mathbf{1}_n = \mathbf{1}_n$，类似地列和也是 1，非负性显然。这保证 composite mapping $\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}$ 仍是 doubly stochastic，**整个深度方向都守恒**。这是 mHC 优于"只约束单层"方案的根本原因。

3. **Geometric interpretation**：Birkhoff polytope 是 permutation matrices 的 convex hull，所以 $\mathcal{H}^{\text{res}} = \sum_\pi p_\pi P_\pi$，其中 $P_\pi$ 是 permutation matrix，$p_\pi \geq 0$ 且 $\sum p_\pi = 1$。换句话说 $\mathcal{H}^{\text{res}} \mathbf{x}$ 是 **"加权平均的 streams 置换"**——一种 convex feature fusion。这也是 $\mathcal{H}^{\text{res}} \mathbf{x}_l$ 是 input streams 的 convex combination 的来源（每行非负且和为 1，所以输出每个 stream 是输入 streams 的 convex combo）。

更深层的 intuition：这种 constrained mapping 在效果上类似 **Markov transition matrix**——$\mathcal{H}^{\text{res}}$ 描述 streams 之间的"概率迁移"。多次应用（即多层的 composite）会趋向 stationary distribution，从而稳定 long-range signal flow。这与 optimal transport 里的 entropic regularization 有深厚联系——见 [Sinkhorn distances](https://arxiv.org/abs/1306.0894) (Cuturi 2013)。

### 3.3 Sinkhorn-Knopp 算法

要把任意矩阵投影到 Birkhoff polytope，论文用 Sinkhorn-Knopp [Sinkhorn & Knopp, 1967](https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-21/issue-2/Concerning-nonnegative-matrices-and-doubly-stochastic-matrices/p10.2140/pjm.1967.21.343.full)。

**算法**：给定任意矩阵 $\tilde{\mathcal{H}}^{\text{res}}$，先取 $\mathbf{M}^{(0)} = \exp(\tilde{\mathcal{H}}^{\text{res}})$ 保证正性（exp 而非 ReLU，是平滑且与 entropic OT 一致），然后交替 normalize 行和列：

$$\mathbf{M}^{(t)} = \mathcal{T}_r\left(\mathcal{T}_c(\mathbf{M}^{(t-1)})\right)$$

- $\mathcal{T}_r(\mathbf{M})_{ij} = \mathbf{M}_{ij} / \sum_j \mathbf{M}_{ij}$：每行除以行和。
- $\mathcal{T}_c(\mathbf{M})_{ij} = \mathbf{M}_{ij} / \sum_i \mathbf{M}_{ij}$：每列除以列和。

收敛到 doubly stochastic matrix，论文取 $t_{\max}=20$ 次（近似解，足够精度）。

**为什么 20 次够？** Sinkhorn-Knopp 的收敛是 geometric 的（[Franklin & Lorenz 1989](https://link.springer.com/book/10.1007/978-3-662-25770-7)），收敛速率与矩阵的 certain scaling properties 相关，20 次足够在 fp32 精度下达到 numerical doubly stochastic。但 Fig. 7(a) 显示 backward gain 略偏离 1（因为有限次迭代未完全收敛）；Fig. 7(b) composite mapping 偏差累积到 ~1.6，比 HC 的 3000 改善了**三个数量级**。

**关于 backward pass**：Sinkhorn-Knopp 的精确 backward 涉及迭代 Jacobian，论文实现中 recompute 整条迭代链 on-chip，避免存中间状态。

---

## 4. mHC 完整参数化

### 4.1 Dynamic + static mappings

论文 Eq.(7) 重新定义 mappings：

$$\begin{cases}
\vec{\mathbf{x}}_l' = \text{RMSNorm}(\vec{\mathbf{x}}_l) \\
\tilde{\mathcal{H}}_l^{\text{pre}} = \alpha_l^{\text{pre}} \cdot (\vec{\mathbf{x}}_l' \varphi_l^{\text{pre}}) + \mathbf{b}_l^{\text{pre}} \\
\tilde{\mathcal{H}}_l^{\text{post}} = \alpha_l^{\text{post}} \cdot (\vec{\mathbf{x}}_l' \varphi_l^{\text{post}}) + \mathbf{b}_l^{\text{post}} \\
\tilde{\mathcal{H}}_l^{\text{res}} = \alpha_l^{\text{res}} \cdot \text{mat}(\vec{\mathbf{x}}_l' \varphi_l^{\text{res}}) + \mathbf{b}_l^{\text{res}}
\end{cases}$$

变量：
- $\vec{\mathbf{x}}_l = \text{vec}(\mathbf{x}_l) \in \mathbb{R}^{1 \times nC}$：把 $n \times C$ 的 stream matrix 拍平，保留全部 context 给 projection。
- $\varphi_l^{\text{pre}}, \varphi_l^{\text{post}} \in \mathbb{R}^{nC \times n}$：dynamic projection weights。
- $\varphi_l^{\text{res}} \in \mathbb{R}^{nC \times n^2}$：dynamic projection，输出 $n^2$ 个值再 reshape 为 $n \times n$。
- $\text{mat}(\cdot)$：$\mathbb{R}^{1 \times n^2} \to \mathbb{R}^{n \times n}$ 的 reshape。
- $\alpha_l \in \mathbb{R}$ scalar：gating factor，初始化为 0.01（小值启动，逐渐放大）。
- $\mathbf{b}_l$：static learnable bias。

### 4.2 最终约束

$$\mathcal{H}_l^{\text{pre}} = \sigma(\tilde{\mathcal{H}}_l^{\text{pre}}), \quad \mathcal{H}_l^{\text{post}} = 2\sigma(\tilde{\mathcal{H}}_l^{\text{post}}), \quad \mathcal{H}_l^{\text{res}} = \text{Sinkhorn-Knopp}(\tilde{\mathcal{H}}_l^{\text{res}})$$

设计细节：
- $\mathcal{H}^{\text{pre}} = \sigma(\cdot)$：sigmoid 输出 $[0, 1]$，作为 layer input 的 convex attention weights——避免正负 cancel。
- $\mathcal{H}^{\text{post}} = 2\sigma(\cdot)$：$[0, 2]$，乘 2 让 layer output 在初始附近"穿过" residual stream，类似 gated residual scaling 的 mid-range。
- $\mathcal{H}^{\text{res}}$：doubly stochastic，三个 streams 间的概率融合。

注意 $\mathcal{H}^{\text{post}}$ 不在 Birkhoff polytope 上——因为它是 $1 \times n$ vector，约束只需 non-negativity。论文对它做"special manifold projection"即 $2\sigma$，这等价于把负值映射到 $[0, 2]$。

### 4.3 与原始 HC 公式的差异

原始 HC（Eq.5）用 `tanh` 而非 `exp + Sinkhorn`：
$$\mathcal{H}_l^{\text{res}} = \alpha_l^{\text{res}} \cdot \tanh(\theta_l^{\text{res}} \tilde{\mathbf{x}}_l^\top) + \mathbf{b}_l^{\text{res}}$$

`tanh` 输出 $[-1, 1]$，可以正负相消，导致 signal cancellation；没有行/列和约束，spectral norm 可任意大。mHC 的核心替换就是这一行——把 `tanh + bias` 换成 `exp + Sinkhorn-Knopp`。

---

## 5. 架构图解析（Fig. 1）

论文 Fig. 1 对比三种 paradigm：

**(a) Residual Connection**：单 stream，$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l)$。一条垂直 trunk + 一个旁路 block。

**(b) Hyper-Connections**：$n$ 条平行 streams（图中 $n=4$），每层有：
- $\mathcal{H}^{\text{pre}}$：streams → single layer input（汇聚）
- $\mathcal{F}$：layer function（仍单 stream）
- $\mathcal{H}^{\text{post}}$：layer output → streams（广播+加权）
- $\mathcal{H}^{\text{res}}$：streams 之间的混合矩阵

**(c) mHC**：结构同 HC，但 $\mathcal{H}^{\text{res}}$ 上加了"约束符号"——投影到 Birkhoff polytope。可视化为带"manifold constraint"标签的 $\mathcal{H}^{\text{res}}$ 节点。

Fig. 4 展示了 DualPipe 在 mHC 下的 schedule：每个 pipeline stage 内部 (F) forward / (B) backward / (W) weight gradient 三类操作，$\mathcal{F}^{\text{A}}$ (attention) 和 $\mathcal{F}^{\text{M}}$ (FFN/MLP) 分开调度，MLP 的 $\mathcal{F}_{\text{post,res}}$ 在 dedicated high-priority stream 上避免 communication 阻塞。

---

## 6. Infrastructure 优化细节

这部分是论文的"实战 chapter"，决定了 mHC 能否落地（6.7% overhead 是非常 impressive 的数字）。

### 6.1 Memory access overhead 分析（Table 2）

HC 每层 residual 维护的 I/O 开销：
- Read: $(5n+1)C + n^2 + 2n$ elements
- Write: $(3n+1)C + n^2 + 2n$ elements

对比 ResNet 的 $2C$ read / $C$ write，n=4 时 HC 多了约 $21C$ read / $13C$ write——5-10 倍 memory traffic。这是 **memory wall** [Dao et al., 2022](https://arxiv.org/abs/2205.14135) 的体现，与现代 GPU HBM 带宽瓶颈相关。

### 6.2 Kernel Fusion（Eq. 10-19）

论文实现了三个 fused kernel：

**Kernel A**（Eq.14-15，计算 $\tilde{\mathcal{H}}$ 与 norm）：
- 把 RMSNorm 的 divide-by-norm 操作 reorder 到 matmul 之后（数学等价，因为 RMSNorm 是逐元素除以 scalar $r$）。
- 一次 scan 计算 $\vec{\mathbf{x}}_l \varphi_l$ 得到所有三个 $\tilde{\mathcal{H}}$，再算 $r = \|\vec{\mathbf{x}}_l\|_2 / \sqrt{nC}$。
- Backward 也 fuse 成单个 kernel。

**Kernel B**（Eq.16-18，lightweight 系数计算）：
- Apply gating $\alpha$，加 bias，做 sigmoid / $2\sigma$。
- 这些操作都在 $n^2+2n$ 这种小 tensor 上（n=4 时仅 24 个元素），fusion 主要是为减少 kernel launch overhead（每个 kernel launch 大约 5-20μs）。

**Kernel C**（Eq.19，Sinkhorn-Knopp）：
- 20 次迭代全在一个 kernel 内完成，无需中间写回 HBM。
- Backward 时 recompute 整条迭代链，traverse 计算 Jacobian-vector product。

**Application kernels**：
- $\mathcal{F}_{\text{pre}} := \mathcal{H}_l^{\text{pre}} \mathbf{x}_l$：single kernel。
- $\mathcal{F}_{\text{post,res}} := \mathcal{H}_l^{\text{res}} \mathbf{x}_l + \mathcal{H}_l^{\text{post}\top} \mathcal{F}(\cdot, \cdot)$：把 $\mathcal{H}^{\text{post}}$ 应用、$\mathcal{H}^{\text{res}}$ 应用、residual merge 三步 fuse，read 从 $(3n+1)C$ 降到 $(n+1)C$，write 从 $3nC$ 降到 $nC$。

用 [TileLang](https://arxiv.org/abs/2504.17577)（Wang et al., 2025）实现，比手写 CUDA 更高 productivity，比 Triton 在 mixed precision 下更灵活。

### 6.3 Recomputing 与最优 block size（Eq. 20）

mHC 在 backward 时 recompute 中间 activations（不存 $\vec{\mathbf{x}}_l$ 在每层）。对 $L_r$ 连续层只存首层 input $\mathbf{x}_{l_0}$，backward 时按需重算。

Memory 公式：
$$\text{Total Memory} \approx nC \cdot \left\lceil \frac{L}{L_r} \right\rceil + (n+2)C \cdot L_r$$

- 第一项：persistent storage，每 $L_r$ 层存一次 $nC$。
- 第二项：transient recomputation buffer，$(n+2)C \times L_r$。

取导数为零：
$$L_r^* = \arg\min_{L_r} [\cdots] \approx \sqrt{\frac{nL}{n+2}}$$

n=4、L=30 时 $L_r^* \approx \sqrt{4 \cdot 30 / 6} = \sqrt{20} \approx 4.5$，取 5。这与 pipeline stage 的层数通常对齐——所以论文选择 recompute 边界与 pipeline stage 同步，避免跨 stage 的 dependency。

### 6.4 DualPipe 中的 communication overlap

[DeepSeek-V3 DualPipe](https://arxiv.org/abs/2412.19437) 是 bidirectional pipeline schedule，把 forward 和 backward 的 communication 与 computation overlap。mHC 带来的挑战：
- $n$-stream 在 stage boundary 需要 $n$ 倍 communication。
- Recompute mHC kernels 在 stage boundary 引入额外 compute。

解决：
1. MLP 的 $\mathcal{F}_{\text{post,res}}$ 在 **dedicated high-priority stream** 上跑，避免阻塞 communication stream。
2. Attention 不用 persistent kernel——允许 preemption，让 communication 抢占。
3. Recompute 与 pipeline communication 解耦（每个 stage 的 $\mathbf{x}_{l_0}$ 本地缓存）。

---

## 7. 实验结果深度分析

### 7.1 训练稳定性（Fig. 5, 27B 模型）

- Baseline（标准 ResNet-style）：loss 平稳下降，gradient norm 稳定。
- HC：~12k step 出现 loss spike（gap 从 -0.04 跳到 -0.01），gradient norm 同时飙高。这是 large-scale training 中的典型"spike"，往往导致 NaN/Inf，需要 restart 或 skip。
- mHC：loss gap 持续 -0.02 ~ -0.025，gradient norm 与 baseline 相当——稳定性恢复。

### 7.2 Propagation stability（Fig. 7）

- HC composite mapping Amax gain：~3000（前向）和类似数量级（反向）。
- mHC composite mapping Amax gain：单层 ~1，composite ~1.6。
- 三个数量级的改善，与理论预期（doubly stochastic spectral norm = 1，n=4 时有限次 Sinkhorn 引入 ~6% 误差，30 层累积到 ~1.6）完全吻合。

### 7.3 下游 benchmark（Table 4，27B 模型）

| Benchmark | Baseline | HC | mHC | mHC vs Baseline | mHC vs HC |
|---|---|---|---|---|---|
| BBH (3-shot) | 43.8 | 48.9 | **51.0** | +7.2 | +2.1 |
| DROP (3-shot) | 47.0 | 51.6 | **53.9** | +6.9 | +2.3 |
| GSM8K (8-shot) | 46.7 | 53.2 | **53.8** | +7.1 | +0.6 |
| HellaSwag | 73.7 | 74.3 | **74.7** | +1.0 | +0.4 |
| MATH | 22.0 | 26.4 | 26.0 | +4.0 | -0.4 |
| MMLU | 59.0 | 63.0 | **63.4** | +4.4 | +0.4 |
| PIQA | 78.5 | 79.9 | **80.5** | +2.0 | +0.6 |
| TriviaQA | 54.3 | 56.3 | **57.6** | +3.3 | +1.3 |

观察：
- mHC 在 reasoning-heavy 任务上（BBH、DROP、GSM8K）相对 HC 改进最大。论文归因于 stability 让模型能"挖掘"更多 HC 的表达力——HC 因 instability 实际上未能完全发挥。
- MATH 上 mHC 略低于 HC（26.0 vs 26.4），可能是 stability-accuracy trade-off 的副作用（HC 的 instability 在某些场景反而 explore 更激进的表示）。
- 系统开销仅 6.7%（n=4），意味着 throughput 仅下降 ~6.7%。

### 7.4 Scaling laws（Fig. 6）

- Compute scaling（3B/9B/27B）：mHC 优势在 compute 增加时 only marginally attenuated——意味着它不是 small-model 的 lucky artifact，而是真正 scalable 的设计。
- Token scaling（3B on 1T tokens）：trajectory 显示持续 advantage。

这是 [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) (Chinchilla) 风格的 compute-optimal scaling 验证。

### 7.5 Hyper-parameter（Table 5）

27B 模型关键 config：
- Total params: 27.0B，Active params: 4.14B（MoE）
- 30 layers，72 routed experts + 2 shared，6 active experts
- Dim 2560，FFN dim 1536
- MLA attention（DeepSeek-V2 风格），KV rank 512，RoPE dim 64
- mHC expansion rate n=4，gating init α=0.01，Sinkhorn iterations t_max=20
- AdamW betas (0.9, 0.95)，base LR 4.0e-4，weight decay 0.1
- 262B training tokens

α=0.01 的 init 让 dynamic mapping 初始接近零，model 启动时几乎就是 baseline ResNet，逐渐 learn 出 stream mixing。这是 **warmup-style initialization**，类似 [SkipNet](https://arxiv.org/abs/1711.09485) 或 LayerScale [Touvron et al., 2021](https://arxiv.org/abs/2103.17239) 的思想——把 innovation 装到 residual path 上，initially 关到接近 identity。

---

## 8. 更深的 intuition 与联想

### 8.1 与 Optimal Transport 的联系

Sinkhorn-Knopp 是 entropic optimal transport [Cuturi, 2013](https://arxiv.org/abs/1306.0894) 的核心算法。当给定 cost matrix $C$，OT 问题 $\min_T \langle T, C\rangle - \epsilon H(T)$ 的解为 $T = \text{diag}(u) \exp(-C/\epsilon) \text{diag}(v)$，$u, v$ 由 Sinkhorn 迭代求出。

mHC 中没有显式 cost matrix，但 $\exp(\tilde{\mathcal{H}}^{\text{res}})$ 可看作 $\exp(-C/\epsilon)$ with $C = -\tilde{\mathcal{H}}^{\text{res}}, \epsilon=1$。所以 mHC 实际上是在每个 layer 学一个 entropic OT plan，把 streams 间 feature transport 当作 optimal transport 问题。这个视角让 mHC 与 [Sinkformers](https://arxiv.org/abs/2110.01607)、[Neural OT](https://arxiv.org/abs/1901.05946) 等工作建立了 connection。

### 8.2 与 Markov Chain 的联系

Doubly stochastic matrix 是 Markov transition matrix（如果只看 row-stochastic 部分）的特殊情况——它额外保证 stationary distribution 是 uniform。这意味着无论 initial streams 的"mass distribution"如何，经过足够多层的 $\mathcal{H}^{\text{res}}$ mixing 后会趋向 uniform mixing。这是一个 **ergodic prior**，鼓励深层 streams 间信息充分融合。

但同时 doubly stochastic 性质限制了表达力——只能做 convex combination，不能做 cancellation。论文通过 $\mathcal{H}^{\text{post}}$ 的 $2\sigma$（输出 $[0, 2]$）部分恢复了 expressive power：layer output 可以是 0 到 2 倍的 $\mathcal{F}$ 输出，允许"信号放大"在 layer output 路径而非 residual 路径上发生。

### 8.3 与 ReZero / SkipInit / LayerScale 的对比

一族工作用"learnable scalar 初始化为 0"实现 deep network stable training：
- [ReZero](https://arxiv.org/abs/2003.13660) (Bachlechner et al., 2020)
- [SkipInit](https://arxiv.org/abs/2101.08692) (De & Smith, 2020)
- [LayerScale](https://arxiv.org/abs/2103.17239) (Touvron et al., 2021)

mHC 的 $\alpha_l^{\text{res}} = 0.01$ init 借鉴了同一思想。但 mHC 比这些更"hard core"——它不仅 gate residual strength，还把 matrix structure 约束到 Birkhoff polytope，从 parameter space 上 enforce stability。

### 8.4 与 Normalization 层的对比

BatchNorm/LayerNorm/RMSNorm 的本质功能之一是 control signal magnitude。mHC 通过 manifold constraint 实现了 **structural normalization**——不依赖 statistics 而依赖几何约束。这类似 [GroupNorm 的 weight standardization](https://arxiv.org/abs/1911.05920) 与 [Weight Normalization](https://arxiv.org/abs/1602.07868) 把 normalization 放到 weights 上而非 activations 上的思想。mHC 把约束放到 connection matrices 上，是更 macro-level 的 normalization。

### 8.5 与 Highway Networks 的对比

[Highway Networks](https://arxiv.org/abs/1505.00387) (Srivastava et al., 2015) 用 gated $\mathbf{T} \mathbf{x} + (1-\mathbf{T}) \mathcal{F}(\mathbf{x})$ 实现 learnable residual。但 Highway 的 gate $\mathbf{T}$ 是 element-wise，没有 cross-stream mixing，且不保证 stability。

mHC 可以看作 **vector-valued Highway Networks with structural constraint**——$\mathcal{H}^{\text{res}}$ 是 matrix-valued gate，且被 project 到 Birkhoff polytope。

### 8.6 关于 Birkhoff Polytope 的几何

Birkhoff polytope $\mathcal{B}_n$ 是 $(n-1)^2$ 维的多面体（因为 $n^2$ 个元素减去 $2n-1$ 个独立约束）。它的顶点是 $n!$ 个 permutation matrices。在 $n=4$ 时，$\mathcal{B}_4$ 是 9 维多面体，有 $4! = 24$ 个顶点。$\mathcal{H}^{\text{res}}$ 在这个 polytope 内移动，gradient flow 由 Sinkhorn-Knopp 的 backward 提供（与 manifold projection 的 Riemannian gradient 类似）。

直觉上：mHC 让 $\mathcal{H}^{\text{res}}$ 在一个"有界有结构"的空间内自由探索，类似 [Gromov-Wasserstein](https://arxiv.org/abs/1905.09627) 中的 constrained coupling learning。

### 8.7 关于 Birkhoff-von Neumann Decomposition

任意 doubly stochastic matrix 可分解为 permutation matrices 的 convex combination：$\mathcal{H} = \sum_k p_k P_k$, $\sum p_k = 1$。这意味着 mHC 的每层 residual mapping 等价于 **"随机选若干 permutation 然后加权平均"**——一种 implicit ensemble of stream permutations。这与 [DenseNet 的 dense connection](https://arxiv.org/abs/1608.06993) 的 implicit ensemble interpretation 遥相呼应。

### 8.8 关于"恢复 identity mapping"的精确表述

ResNet 论文 [He et al., 2016b](https://arxiv.org/abs/1603.05036) 证明：当 shortcut 是 identity 且 addition 不可扰时，任意深层 $L$ 到浅层 $l$ 的信息流是 $\mathbf{x}_l$ 本身（identity）+ $\sum \mathcal{F}$。这里 identity 是"严格无变换"。

mHC 把这推广为"**doubly stochastic identity**"——单层是 doubly stochastic（含 identity 是其特例），composite 也是 doubly stochastic。所以 mHC 是 ResNet identity mapping property 的 **strict generalization**：$n=1$ 时严格退化到 ResNet。

---

## 9. 限制与开放问题

### 9.1 表达力损失

doubly stochastic 约束让 $\mathcal{H}^{\text{res}}$ 无法做 negative weight 或 scaling >1。当模型需要"压制某条 stream"（赋负权）或"放大某条 stream"（scale >1）时，mHC 必须通过 $\mathcal{H}^{\text{pre}}, \mathcal{H}^{\text{post}}$ 间接实现。这可能是 MATH 上 mHC 略低于 HC（26.0 vs 26.4）的原因。

### 9.2 Sinkhorn-Knopp 有限迭代误差

20 次迭代产生 ~6% 偏差，30 层累积到 ~1.6× 的 composite gain。理论上这仍是 bounded，但若层数进一步增加（百B模型可能 100+层），可能需要更多迭代或更稳定算法。可考虑：
- [Log-stabilized Sinkhorn](https://arxiv.org/abs/2006.02603)
- [Newton-type methods on Birkhoff polytope](https://arxiv.org/abs/1710.02693)
- [Gromov-Wasserstein based alternatives](https://arxiv.org/abs/1905.09627)

### 9.3 Other manifolds 的探索

论文 Section 6 提到"framework accommodates exploration of diverse manifold constraints"。可能的替代：
- **Orthogonal group** $\mathcal{O}(n)$：通过 [Cayley transform](https://arxiv.org/abs/2106.13063) parameterize，保证 $\|\mathcal{H}\|_2 = 1$ 但允许负元素。
- **Symplectic group**：保留 Hamiltonian 结构。
- **PSD cone**：保证 positive semi-definite，与 attention 的 QK^T 类似。
- **Stiefel manifold**：列正交，[broader parameterization](https://arxiv.org/abs/1909.01396)。

这些是论文 Section 6 "future research" 真正指向的方向——**manifold-constrained architecture design** 可能成为 macro-architecture 的 next frontier。

### 9.4 与 MoE 的交互

论文用 DeepSeek-V3 风格 MoE，但没详细讨论 mHC 与 expert routing 的 interaction。开放问题：
- 不同 expert 是否应共享 $\mathcal{H}^{\text{res}}$？
- $\mathcal{H}^{\text{res}}$ 是否可以 expert-specific？
- 在 MoE 的 load balancing 与 mHC 的 stream mixing 之间是否有更深的耦合？

---

## 10. 总结

mHC 是一个**结构精巧、工程扎实**的工作。它的核心贡献：

1. **理论 insight**：识别出 HC 的 instability 源于 unconstrained composite mapping $\prod \mathcal{H}^{\text{res}}$，并提出 doubly stochastic 约束恢复 conservation property。
2. **数学优雅**：Birkhoff polytope 提供了 (i) spectral norm bound，(ii) compositional closure，(iii) permutation ensemble interpretation 的三重 benefit。
3. **工程扎实**：kernel fusion、recompute、DualPipe overlap 三层优化让 n=4 mHC 的开销压到 6.7%。
4. **实验充分**：27B 模型 + scaling law + stability analysis，证明 large-scale 适用性。
5. **范式扩展**：把 identity mapping 从 strict identity 推广到 manifold-constrained doubly stochastic，打开 manifold-constrained architecture design 这扇门。

它继承 HC 的拓扑表达力，同时恢复 ResNet 的稳定性——是 macro-architecture design 在 LLM 时代的一次重要 advance。

---

## References

- [He et al., 2016a - Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [He et al., 2016b - Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05036)
- [Zhu et al., 2024 - Hyper-Connections](https://arxiv.org/abs/2409.19606)
- [Sinkhorn & Knopp, 1967 - Doubly Stochastic Matrices](https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-21/issue-2/Concerning-nonnegative-matrices-and-doubly-stochastic-matrices/p10.2140/pjm.1967.21.343.full)
- [Cuturi, 2013 - Sinkhorn Distances](https://arxiv.org/abs/1306.0894)
- [Birkhoff-von Neumann theorem](https://en.wikipedia.org/wiki/Birkhoff%E2%80%93von_Neumann_theorem)
- [Dao et al., 2022 - FlashAttention](https://arxiv.org/abs/2205.14135)
- [Liu et al., 2024b - DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [Liu et al., 2024a - DeepSeek-V2 / MLA](https://arxiv.org/abs/2405.04434)
- [Wang et al., 2025 - TileLang](https://arxiv.org/abs/2504.17577)
- [Hoffmann et al., 2022 - Chinchilla / Compute-Optimal Scaling](https://arxiv.org/abs/2203.15556)
- [Huang et al., 2017 - DenseNet](https://arxiv.org/abs/1608.06993)
- [Srivastava et al., 2015 - Highway Networks](https://arxiv.org/abs/1505.00387)
- [Touvron et al., 2021 - LayerScale](https://arxiv.org/abs/2103.17239)
- [Bachlechner et al., 2020 - ReZero](https://arxiv.org/abs/2003.13660)
- [Xiao et al., 2025 - MUDDFormer](https://arxiv.org/abs/2502.12170)
- [Mak & Flanigan, 2025 - RMT](https://arxiv.org/abs/2506.22696)
- [Pagliardini et al., 2024 - DenseFormer](https://openreview.net/forum?id=kMnoh7CXrq)
- [Heddes et al., 2025 - DeepCrossAttention](https://openreview.net/forum?id=j3JBfFnGYh)
- [Menghani et al., 2025 - LAurel](https://openreview.net/forum?id=rUDRWP9WvZ)
- [Sinkformers](https://arxiv.org/abs/2110.01607)
- [Gromov-Wasserstein](https://arxiv.org/abs/1905.09627)
- [Cayley transform for orthogonal RNN](https://arxiv.org/abs/2106.13063)
