---
source_pdf: mHC.pdf
paper_sha256: d849f4709ed29bfb2751f51e24dc836403f31271d1764dddfba0768d326a75d9
processed_at: '2026-08-05T18:07:07-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 一句话版本

residual connection 是 LLM 能训深的命根子，因为它让浅层信号无损直达深层。Hyper-Connections 想把这条"单车道"扩成 4 车道来增强表达力，效果确实好，但那个"变道矩阵"是完全自由可学习的，连乘几十层之后信号被放大 3000 倍，训练直接爆炸。mHC 干的事情就是给这个变道矩阵加一条规矩——每行和为 1、每列和为 1、元素全非负——这样无论怎么连乘，信号总量永远守恒，既不会爆也不会塌。

---

# 多说几句的人话版

## 残差连接为什么重要

你从第 1 层往第 100 层传信号。普通网络每层都在改信号，传到 100 层早就面目全非了，梯度也回不去。ResNet 的招数是每层都加一条"原封不动传过去"的旁路，所以第 1 层的信号能直接"看到"第 100 层，梯度也能直接顺着这条路流回去。这条旁路就是 identity mapping，它是深网络能训起来的根本原因。

## HC 想干什么

HC 说：一条路太单调了，我把它扩成 4 条平行的路，然后在每个路口装一个可学习的"变道器"，让信息能在 4 条路之间流动。这个变道器是个 4×4 的小矩阵，几乎不增加 FLOPs。实验下来 loss 确实降了，下游任务也涨点。看起来很美。

## HC 的病

那个变道器是完全自由学习的。一层变道器没事，但你的网络有 30 层甚至更多，信号从第 1 层传到第 30 层要经过 30 个变道器连乘。只要这些矩阵的谱范数稍微大于 1（几乎必然会发生），连乘下来信号就被放大几千倍。他们在 27B 模型上实测：forward 信号增益峰值 3000，训练到 12000 步 loss 突然飙升，gradient norm 同步爆掉。这就是 HC 没法 scale 上去的根本原因——它把 identity mapping 那个"信号守恒"的保证给丢了。

## mHC 的核心 insight

别让变道器完全自由。把它限制成一种特殊矩阵：所有元素非负、每行之和等于 1、每列之和也等于 1。这种矩阵叫 doubly stochastic matrix。

为什么这个限制几乎完美？三个原因：

**第一，信号不爆。** 这种矩阵的谱范数最多是 1，所以一层层连乘下来信号只会维持或收缩，不会膨胀。梯度也一样安全。

**第二，连乘之后还是这种矩阵。** 两个 doubly stochastic 矩阵乘起来还是 doubly stochastic。所以无论网络多深，那个 composite mapping 永远守恒。这是 mHC 的杀手锏——它把"identity mapping 的信号守恒"这个性质从"单车道"泛化到了"多车道"。

**第三，几何上特别干净。** 这种矩阵是所有"排列矩阵"（硬性切换车道顺序）的凸包。所以你的变道器本质上是在"软选择几种排列的混合"——既有多车道的表达力，又不会出现正负系数互相抵消把信号抹掉的尴尬。

## 怎么实现这个限制

用 Sinkhorn-Knopp 算法。先把矩阵所有元素取 exp（保证非负），然后反复做"每行除以行和"、"每列除以列和"，交替 20 次就收敛到一个 doubly stochastic 矩阵。这个算法 1967 年就有了，最优传输里用得很多。

## 系统层面没糊弄

HC 当年被诟病的是 memory access 开销大 6 倍但没人认真处理。mHC 团队（DeepSeek，做过 MLA、DualPipe 那拨人）写了专门的 fused kernel（用 TileLang），把 RMSNorm、三个映射计算、Sinkhorn 迭代、residual merge 全融合；又做了 selective recompute（每几层只存一次输入，反向时重算），还推导了最优 recompute 块大小公式 $L_r^* \approx \sqrt{nL/(n+2)}$；最后扩展了 DualPipe 让 pipeline 通信和 mHC 计算能重叠。最终 n=4 时只增加 6.7% 训练时间。

## 结果

27B MoE 上稳定训练，loss 比 baseline 降 0.021，gradient norm 一直平稳（HC 在 12k 步爆掉）。下游 8 个 benchmark 里 7 个超过 HC，推理类的 BBH 和 DROP 涨得最多（+2.1 / +2.3）。composite mapping 的增益从 HC 的 3000 降到 mHC 的 1.6，三个数量级的改善。

## 最直觉的画面

普通 ResNet 是单车道公路，车从起点到终点速度恒定。HC 把公路扩成 4 车道并装上自由调节的变道器，但变道器可能把 4 条车道的车全挤到 1 条上（信号爆）或者让车互相抵消（信号塌）。mHC 给变道器装上信号灯：每条车道流入总量等于流出总量，且不允许反向车流。无论怎么变道，总流量永远守恒。代价只是 6.7% 的红绿灯运营成本，整个系统能 scale 到 27B+ 还稳定运行。

---

## 为什么我觉得这篇值得认真读

它把"扩展拓扑结构"和"保持信号守恒"这两件过去被认为矛盾的事情用 Birkhoff polytope 这个干净几何工具统一起来了。HC 是个有潜力但有原罪的 idea，mHC 是它的赎罪。更重要的是，DeepSeek 团队没把系统成本当小事糊弄过去，6.7% 这个数字是认真工程出来的。这给后面的人开了一个方向：你只要能找到一个合适的 manifold 把 residual mapping 圈住，就可以放心大胆设计更复杂的跨层拓扑。

---

# mHC: Manifold-Constrained Hyper-Connections — 深度讲解

Andrej，这篇 DeepSeek-AI 的 paper 是对你我都非常关心的一个话题的回应：**residual connection 这个十年老范式该如何被认真扩展**。我会尽量把动机、数学、系统、以及和它的"祖宗们"(ResNet, DenseNet, Highway Networks, DenseFormer, MUDDFormer, RMT) 的关系都铺开讲。

---

## 1. 大背景：为什么 residual connection 是 LLM 的命根子

自 ResNet (He et al., 2016a) 以来，单层结构的形式一直是：

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)
$$

变量说明：
- $\mathbf{x}_l \in \mathbb{R}^C$：第 $l$ 层输入，维度 $C$
- $\mathbf{x}_{l+1}$：第 $l$ 层输出
- $\mathcal{F}(\cdot, \mathcal{W}_l)$：residual function (conv / attention / FFN 等)，参数 $\mathcal{W}_l$

**关键性质** (He et al., 2016b, "Identity Mappings in Deep Residual Networks")：递归展开得到

$$
\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)
$$

这里的 $\mathbf{x}_l$ 这一项就是所谓 **identity mapping** — shallow layer 的信号原封不动地"长程直达" deep layer。这个性质对优化至关重要：前向中信号不被任意缩放，反向中梯度能无损回流。整篇 mHC 论文的所有努力都是在试图保住这个性质的同时引入表达力。

参考:
- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016: https://arxiv.org/abs/1512.03385
- He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016: https://arxiv.org/abs/1603.05027

---

## 2. Hyper-Connections (HC) 想做什么、为什么"想做但做不好"

HC (Zhu et al., 2024) 的核心想法是：**把 residual stream 从 1 维 $C$ 扩成 $n$ 维 $nC$，并引入可学习的跨 stream 混合**。单层形式：

$$
\mathbf{x}_{l+1} = \mathcal{H}_l^{\text{res}} \mathbf{x}_l + \mathcal{H}_l^{\text{post}\top} \mathcal{F}(\mathcal{H}_l^{\text{pre}} \mathbf{x}_l, \mathcal{W}_l)
$$

变量：
- $\mathbf{x}_l \in \mathbb{R}^{n \times C}$：现在 hidden state 是一个 $n$-stream 矩阵，$n$ 是 expansion rate (论文里 $n=4$)
- $\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$：residual stream 内部的可学习混合映射
- $\mathcal{H}_l^{\text{pre}} \in \mathbb{R}^{1 \times n}$：把 $nC$ 流聚成 $C$ 维喂给 layer
- $\mathcal{H}_l^{\text{post}} \in \mathbb{R}^{1 \times n}$：把 layer 输出写回流

直观上 $\mathcal{H}^{\text{res}}$ 是"stream 之间的 router"，$\mathcal{H}^{\text{pre}}$ 是"read head"，$\mathcal{H}^{\text{post}}$ 是"write head"。FLOPs 几乎不变 (因为 $n \ll C$)，但模型多了一个新的 scaling 维度 (除了参数量与数据量之外的)。

**参数化 (Eq. 5)**：每个 mapping 都是 dynamic + static 两部分：

$$
\mathcal{H}_l^{\text{res}} = \alpha_l^{\text{res}} \cdot \tanh(\theta_l^{\text{res}} \tilde{\mathbf{x}}_l^\top) + \mathbf{b}_l^{\text{res}}
$$

其中 $\tilde{\mathbf{x}}_l = \text{RMSNorm}(\mathbf{x}_l)$，$\theta_l^{\text{res}} \in \mathbb{R}^{n \times C}$ 是 dynamic 投影，$\mathbf{b}_l^{\text{res}} \in \mathbb{R}^{n \times n}$ 是 static bias，$\alpha_l^{\text{res}}$ 是 scalar gating (初始化为 0.01 让动态部分起步小)。

**消融研究 (Tab. 1)** 很关键：只开 $\mathcal{H}^{\text{res}}$ 就能拿到 -0.022 的 loss gap，再加 $\mathcal{H}^{\text{pre}}$ / $\mathcal{H}^{\text{post}}$ 只能再降 -0.003 / -0.002。这说明 **residual stream 内部的 mixing 是收益主源**，也暗示了它也是不稳定的根源。

参考:
- Zhu et al., "Hyper-Connections", arXiv 2409.19606: https://arxiv.org/abs/2409.19606

---

## 3. HC 的两个病：numerical instability 和 system overhead

### 3.1 Identity mapping 被破坏 → 信号爆炸

把 HC 递归展开 (Eq. 4)：

$$
\mathbf{x}_L = \underbrace{\left(\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{\text{res}}\right)}_{\text{composite mapping}} \mathbf{x}_l + \sum_{i=l}^{L-1} \left(\prod_{j=1}^{L-1-i} \mathcal{H}_{L-j}^{\text{res}}\right) \mathcal{H}_i^{\text{post}\top} \mathcal{F}(\mathcal{H}_i^{\text{pre}} \mathbf{x}_i, \mathcal{W}_i)
$$

对比 ResNet 的 $x_L = x_l + \dots$，HC 那个 composite mapping $\Pi \mathcal{H}^{\text{res}}$ 不再是单位矩阵。因为 $\mathcal{H}^{\text{res}}$ 是 **无约束可学习矩阵**，连乘后谱范数可能远大于 1 → forward signal 爆炸；或者列和远大于 1 → backward gradient 爆炸。

**论文里用一个非常聪明的度量**："Amax Gain Magnitude"：
- Forward gain = max absolute row sum of composite mapping (对应 forward signal 的 $\ell_\infty$ 范数放大)
- Backward gain = max absolute column sum (对应 backward gradient 的放大)

27B model 实测 (Fig. 3b)：composite mapping 的 Amax Gain Magnitude 峰值 **~3000**。也就是说从浅层到深层，信号被无约束地放大了 3000 倍。Fig. 2 显示在 ~12k step 处 loss 突然飙升，gradient norm 同步爆掉。这是典型的 residual stream divergence。

### 3.2 Memory access overhead

Tab. 2 算了 per-token I/O：
- 普通 residual：read $2C$ / write $C$
- HC：read $(5n+1)C + n^2 + 2n$ / write $(3n+1)C + n^2 + 2n$

对 $n=4$ 这接近 6 倍的 I/O。再叠加 pipeline parallelism 通信量 $\propto n$ → memory wall 加 pipeline bubble 双重打击。

---

## 4. mHC 的核心思想：把 $\mathcal{H}^{\text{res}}$ 投影到 Birkhoff polytope

这是全文最精彩的几何 move。让 $\mathcal{H}_l^{\text{res}}$ 落在 **doubly stochastic matrix** 的 manifold 上：

$$
\mathcal{P}_{\mathcal{M}^{\text{res}}}(\mathcal{H}_l^{\text{res}}) := \left\{ \mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n} : \mathcal{H}_l^{\text{res}} \mathbf{1}_n = \mathbf{1}_n, \ \mathbf{1}_n^\top \mathcal{H}_l^{\text{res}} = \mathbf{1}_n^\top, \ \mathcal{H}_l^{\text{res}} \geqslant 0 \right\}
$$

变量：
- $\mathbf{1}_n$：长度 $n$ 全 1 向量
- 第一条约束：每行和为 1
- 第二条：每列和为 1
- 第三条：元素非负

这个集合就是著名的 **Birkhoff polytope** — 所有 $n \times n$ permutation matrices 的凸包 (Birkhoff-von Neumann 定理)。

为什么这个约束几乎完美？三个性质：

### 4.1 Norm preservation
对 doubly stochastic matrix，谱范数 $\|\mathcal{H}_l^{\text{res}}\|_2 \leq 1$。
证明 sketch: 行和为 1 + 非负 → $\|\mathcal{H}\|_\infty = 1$；列和为 1 + 非负 → $\|\mathcal{H}\|_1 = 1$。再由 $\|\cdot\|_2 \leq \sqrt{\|\cdot\|_1 \cdot \|\cdot\|_\infty} = 1$。
→ 映射是 non-expansive，gradient 不会爆炸，vanishing 也被严格抑制。

### 4.2 Compositional closure
Doubly stochastic 集合对乘法封闭：$\prod_i \mathcal{H}_i^{\text{res}}$ 仍是 doubly stochastic。
证明：两个双随机矩阵乘积的行和 = (A(B1)) = A1 = 1，列和 = (1^T A)B = 1^T B = 1。
→ 这是 mHC 的"杀手锏"：composite mapping 在任意深度都保持稳定，**identity mapping 的"信号守恒"性质被泛化成"信号总量守恒"**。

### 4.3 几何意义：convex combination of permutations
Birkhoff polytope 的顶点是 permutation matrices。所以 $\mathcal{H}^{\text{res}} x_l$ 实际是"对 $n$ 个 stream 做 permutation 的凸组合" — 它在 **不做缩放、不做 cancellation 的前提下做特征融合**。这正好避免了 positive 和 negative 系数互相抵消导致信号坍缩。

直觉上 (这是我对你的直觉讲)：原始 ResNet 的 identity mapping 是 Birkhoff polytope 上的一个顶点 ($n=1$ 退化情形就是标量 1，或者 $n>1$ 时的恒等 permutation)。mHC 把 residual connection 的空间从"那个唯一的顶点"放宽到了"整个多面体内"，但绝不允许越界 — 这就是"manifold-constrained"的精神。

参考:
- Sinkhorn & Knopp, 1967: https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-21/issue-2/Concerning-nonnegative-matrices-and-doubly-stochastic-matrices/pjm/1102993855.full
- Birkhoff-von Neumann theorem: https://en.wikipedia.org/wiki/Birkhoff%E2%80%93von_Neumann_theorem
- Sinkhorn distance / Optimal transport (Cuturi 2013): https://arxiv.org/abs/1306.0895

---

## 5. 参数化与 Sinkhorn-Knopp 投影

### 5.1 参数化 (Eq. 7)

输入 $\mathbf{x}_l \in \mathbb{R}^{n \times C}$ 先 flatten 成 $\vec{\mathbf{x}}_l \in \mathbb{R}^{1 \times nC}$，再算三个 mapping 的 pre-activation：

$$
\tilde{\mathcal{H}}_l^{\text{res}} = \alpha_l^{\text{res}} \cdot \text{mat}(\vec{\mathbf{x}}_l' \varphi_l^{\text{res}}) + \mathbf{b}_l^{\text{res}}
$$

变量说明：
- $\varphi_l^{\text{res}} \in \mathbb{R}^{nC \times n^2}$：dynamic projection，把 $nC$ 维输入投到 $n^2$ 维
- $\text{mat}(\cdot)$：把 $1 \times n^2$ 重塑成 $n \times n$
- $\mathbf{b}_l^{\text{res}} \in \mathbb{R}^{n \times n}$：static bias

**注意一个小细节**: 原始 HC 用 `tanh` 作为 dynamic part 的非线性，mHC **去掉了** `tanh` (直接 $\vec{\mathbf{x}}_l' \varphi_l$)，因为后续 Sinkhorn-Knopp 会做 `exp` 操作，本身已经提供非线性 + positivity。这个改动很微妙，但可能对表达力有影响 (后面会联想)。

### 5.2 Manifold projection (Eq. 8)

$$
\begin{aligned}
\mathcal{H}_l^{\text{pre}} &= \sigma(\tilde{\mathcal{H}}_l^{\text{pre}}) \\
\mathcal{H}_l^{\text{post}} &= 2\sigma(\tilde{\mathcal{H}}_l^{\text{post}}) \\
\mathcal{H}_l^{\text{res}} &= \text{Sinkhorn-Knopp}(\tilde{\mathcal{H}}_l^{\text{res}})
\end{aligned}
$$

注意几个设计选择：
- $\mathcal{H}^{\text{pre}} \in [0,1]$：让 read 是非负的 (避免 cancellation)
- $\mathcal{H}^{\text{post}} \in [0,2]$：让 write 可以放大 (平均增益 ≈ 1)，保证不系统性地压制信号
- $\mathcal{H}^{\text{res}}$ 投影到双随机

**Sinkhorn-Knopp 迭代 (Eq. 9)**：

$$
\mathbf{M}^{(0)} = \exp(\tilde{\mathcal{H}}_l^{\text{res}}), \quad \mathbf{M}^{(t)} = \mathcal{T}_r(\mathcal{T}_c(\mathbf{M}^{(t-1)}))
$$

- $\mathcal{T}_c(\mathbf{M})_{ij} = \mathbf{M}_{ij} / \sum_i \mathbf{M}_{ij}$：列归一化
- $\mathcal{T}_r(\mathbf{M})_{ij} = \mathbf{M}_{ij} / \sum_j \mathbf{M}_{ij}$：行归一化

迭代收敛到 doubly stochastic matrix (Sinkhorn 定理保证正矩阵必然收敛)。论文用 $t_{\max} = 20$。

**为什么 20 步够用**：Sinkhorn 在 entries 数量级差异不大时收敛极快。20 步通常 residual 行列和偏差在 $10^{-3}$ 量级，已经满足稳定性需求。Fig. 7a 显示 backward gradient gain 略偏离 1，就是这个近似误差导致的，但复合 (Fig. 7b) 最大也只到 ~1.6，比 HC 的 3000 小了三个数量级。

---

## 6. Infrastructure：为什么这部分不能省

这是 paper 里我最欣赏的一节 — DeepSeek 把"系统开销"当成一等公民在 paper 里讲清楚。HC 原作者忽略了 memory wall，mHC 必须把账补上。

### 6.1 Kernel fusion (Eq. 10–19)

他们用 TileLang (Wang et al., 2025) 写了 5 个 fused kernel：
1. 把 $\varphi_l^{\text{pre}}$, $\varphi_l^{\text{post}}$, $\varphi_l^{\text{res}}$ 合并成一个大矩阵 $\varphi_l \in \mathbb{R}^{nC \times (n^2+2n)}$，一次 matmul 同时算三个 pre-activation
2. 把 RMSNorm 的 divide-by-norm **挪到 matmul 之后** (数学等价，但避免了在高维 $nC$ 上做 norm)
3. Sinkhorn-Knopp 20 步迭代放在一个 kernel 里
4. Backward pass 自定义 kernel，on-chip recompute 迭代过程
5. 把 $\mathcal{H}^{\text{res}} \mathbf{x}_l$ + $\mathcal{H}^{\text{post}\top} \mathcal{F}$ + residual merge 融合，把 read 从 $(3n+1)C$ 降到 $(n+1)C$，write 从 $3nC$ 降到 $nC$

混合精度策略：$\varphi_l$ 用 tf32 (累加精度)，$\vec{\mathbf{x}}_l$ 用 bfloat16 (访存友好)，$\alpha_l$ 和 $\mathbf{b}_l$ 用 float32 (小张量精度)。

### 6.2 Recomputing 与最优块大小 (Eq. 20)

$nC$ 的 hidden state 全部存下来太贵，所以对每 $L_r$ 个连续层只存第一层的输入 $\mathbf{x}_{l_0}$，其余在 backward 时 on-the-fly recompute。

Memory = resident (固定存储) + transient (重算时占用)：

$$
\text{Memory}(L_r) = nC \cdot \left\lceil \frac{L}{L_r} \right\rceil + (n+2)C \cdot L_r
$$

对 $L_r$ 求导取 0 得到：

$$
L_r^* \approx \sqrt{\frac{nL}{n+2}}
$$

对 $n=4, L=30$ (27B model), $L_r^* \approx \sqrt{4 \cdot 30 / 6} \approx 4.5$。论文选 $L_r$ 对齐 pipeline stage 边界 (因为跨 stage 重算会破坏 pipeline schedule)。

### 6.3 DualPipe 扩展

DualPipe (DeepSeek-V3 的 schedule) 是把 forward / backward / weight-grad 重叠，让 expert parallelism 和 pipeline parallelism 的通信与计算重叠。mHC 的 $n$-stream 让 stage 间通信量增加 $n$ 倍，recompute 又增加 stage 边界的计算开销。他们的处理：
- $\mathcal{F}_{\text{post,res}}$ (MLP/FFN 的) 放在 dedicated high-priority stream 上跑，防止阻塞 communication stream
- Attention 层不用 persistent kernel，允许被 preempt 以便灵活调度
- Recompute 与 pipeline 通信解耦 (因为 $\mathbf{x}_{l_0}$ 已本地缓存)

**最终 overhead**: n=4 时只增加 **6.7%** 训练时间。这个数字很关键 — 说明 mHC 在系统层面"几乎免费"。

参考:
- TileLang (Wang et al., 2025): https://arxiv.org/abs/2504.17577
- DeepSeek-V3 (DualPipe): https://arxiv.org/abs/2412.19437
- FlashAttention (memory wall concept): https://arxiv.org/abs/2205.14135
- Zero Bubble Pipeline Parallelism (Qi et al., 2024): https://arxiv.org/abs/2401.10241

---

## 7. 实验：稳定性 + 性能 + scalability

### 7.1 主结果 (Tab. 4, 27B model)

| Benchmark | Baseline | HC | mHC | mHC vs HC |
|---|---|---|---|---|
| BBH (3-shot) | 43.8 | 48.9 | **51.0** | +2.1 |
| DROP (3-shot) | 47.0 | 51.6 | **53.9** | +2.3 |
| GSM8K (8-shot) | 46.7 | 53.2 | **53.8** | +0.6 |
| HellaSwag | 73.7 | 74.3 | **74.7** | +0.4 |
| MATH (4-shot) | 22.0 | **26.4** | 26.0 | -0.4 |
| MMLU (5-shot) | 59.0 | 63.0 | **63.4** | +0.4 |
| PIQA | 78.5 | 79.9 | **80.5** | +0.6 |
| TriviaQA | 54.3 | 56.3 | **57.6** | +1.3 |

观察：
- mHC 在 8 个中 7 个超过 HC，只在 MATH 略低 0.4 (噪声范围)
- **Reasoning 类 (BBH, DROP) 提升最大** — 作者认为这是因为 stable signal propagation 让长链推理受益
- mHC 相对 baseline 总平均提升 ~3-4%，这是 architecture-only gain (无数据/参数变化)

### 7.2 Scaling (Fig. 6)

- Compute scaling: 3B → 9B → 27B (proportional data)，mHC 的 loss gap 在大 compute 下只是"轻微衰减"，没有掉下去
- Token scaling: 3B on 1T tokens 训练曲线稳定，gap 维持

这强烈提示 mHC **没有 scalability ceiling**。HC 在 27B 上其实就开始不稳定 (Fig. 2)，mHC 把这个 ceiling 推到至少 27B 以上。

### 7.3 Stability 分析 (Fig. 7, 8)

- Single-layer Amax gain：mHC 的 forward gain ≈ 1, backward gain 略偏离 1 (Sinkhorn 20 步近似误差)
- Composite Amax gain：最大 ≈ 1.6 (HC ≈ 3000)
- Fig. 8 可视化：HC 当某个 entry 大时整行/整列都大 (系统性不稳)，mHC 矩阵始终接近某个 permutation matrix 的凸组合

---

## 8. 与相关工作的谱系 — 这才是这篇 paper 的"真正坐标"

我想把 mHC 放在 macro-architecture 演进的整条脉络里看。

### 8.1 第一代：Dense connectivity
- **ResNet** (2016): 加法 + identity mapping 的奠基
- **Highway Networks** (Srivastava et al., 2015): gating $y = g \cdot H(x) + (1-g) \cdot x$ — 第一次破坏 identity 的尝试，但 $g$ 是 sigmoid，长程会 saturate
- **DenseNet** (Huang et al., 2017): 拼接所有前层 — feature reuse 极强但 memory 线性增长
- **FractalNet** (Larsson et al., 2016): 递归分形拓扑
- **DLA** (Yu et al., 2018): 跨 depth/resolution 的递归聚合

这代都"加宽连接"，但都没形成 LLM 时代的标准。

### 8.2 第二代：Transformer 时代的 dense connection 复兴
- **DenseFormer** (Pagliardini et al., 2024): depth-weighted averaging across layers — 简单粗暴的跨层加权平均
- **Highway Transformer** (Chai et al., 2020): self-gating attention
- **Cross-layer retrospective retrieving / Layer Attention** (Fang et al., 2023): 用 attention 做跨层信息流
- **ResiDual** (Xie et al., 2023): dual residual connections
- **MUDDFormer** (Xiao et al., 2025): multiway dynamic dense connections — 非常接近 HC 的精神，都是 per-token dynamic 跨层连接
- **DeepCrossAttention** (Heddes et al., 2025): 跨层 cross-attention 替代 residual
- **RMT** (Mak and Flanigan, 2025): outer-product memory matrix 替代 residual stream
- **LAurel** (Menghani et al., 2025): learned augmented residual layer
- **HC** (Zhu et al., 2024): 集大成者，把"加宽 stream + 可学习跨层 routing"做到 FLOPs-neutral

**这代的关键缺陷** (mHC 的论点)：都破坏了 identity mapping 的"信号守恒"，所以都没法 scale 到 LLM 体量。MUDDFormer / DenseFormer 等都在中小规模验证，没人敢在 27B+ MoE 上严肃训练。

### 8.3 第三代：manifold-constrained (mHC 自己)
mHC 的"立 flag"主张是：**topological complexity 与 identity mapping 不必二选一，只要把 residual mapping 限制在合适的 manifold 上**。

参考:
- Highway Networks: https://arxiv.org/abs/1505.00387
- DenseNet: https://arxiv.org/abs/1608.06993
- DenseFormer: https://openreview.net/forum?id=kMnoh7CXrq
- MUDDFormer: https://arxiv.org/abs/2502.12170
- RMT: https://arxiv.org/abs/2506.22696
- DeepCrossAttention: https://openreview.net/forum?id=j3JBfFnGYh
- LAurel: https://openreview.net/forum?id=rUDRWP9WvZ
- ResiDual: https://arxiv.org/abs/2304.14802

---

## 9. 几个我自己想深挖的 intuition

### 9.1 为什么是 doubly stochastic，而不是 orthogonal 或 low-rank？

我脑子里第一时间想到的替代方案：
- **Orthogonal matrix** ($\mathcal{H}^\top \mathcal{H} = I$)：完美保持 $\ell_2$ 范数，且对乘法封闭。问题：orthogonal 矩阵没有"non-negative + 行和列和为 1"的 convex combination 性质，element-wise 会正负抵消 → 信号 cancellation 风险。Cayley parameterization 也比较麻烦。
- **Stochastic matrix** (只行和为 1)：Markov chain transition matrix。对乘法封闭，但是只保证 forward stability (行和=1), 不保证 backward stability (列和 ≠ 1)。Backward gradient 通过转置传，所以列和也得 = 1。
- **Diagonal scaling** ($\text{diag}(d_1, \dots, d_n)$ 且 $d_i \in [0, 1]$)：最简单，但完全没 stream mixing 能力。
- **Low-rank perturbation** of identity ($I + UV^\top$ with spectral constraint)：表达力可能不够。

Doubly stochastic 同时拿到：**non-negativity (no cancellation)** + **行/列和为 1 (forward/backward 守恒)** + **Birkhoff polytope 几何 (convex combination of permutations = 完美的"混合"语义)** + **对乘法封闭 (composite stability)** + **Sinkhorn 高效投影**。这套组合的均衡性是它赢的原因。

### 9.2 mHC 和 MoE 路由的关系

mHC 的 $\mathcal{H}^{\text{res}}$ 投影到 Birkhoff polytope，让人直接联想到 **Sinkhorn-based MoE routing** (如 Switch Transformer 之后的负载均衡 Sinkhorn 方法)。本质都是"把不均衡的 score 通过 alternating normalization 转成 balanced assignment"。

但 mHC 的 $\mathcal{H}^{\text{res}}$ 是 $n \times n$ (n=4，小)，MoE 的 routing 是 token × expert (大)。mHC 是 **stream-level** 的"软路由"，MoE 是 **token-level** 的"硬选 expert"。两者在 DeepSeek-V3 MoE 架构里同时存在，互不冲突。

参考 Switch Transformer 的 Sinkhorn balancing: https://arxiv.org/abs/2101.03961

### 9.3 mHC 和 layer normalization 的隐含关系

论文里有个我没看到强调但很重要的点：mHC 对 $\mathcal{H}^{\text{pre}}$ 和 $\mathcal{H}^{\text{post}}$ 也加了 non-negativity (sigmoid)。这意味着 read 和 write 都是"非负凸组合"。再加上 RMSNorm 的 scale 不变性，整个 mHC block 在某种意义上是一个 **energy-preserving** 系统。

对比 QK-norm / attention temperature 等技巧 — 都是在不破坏表达力的前提下给信号加"守恒律"。mHC 是把这条线推到了 **macro-architecture** 层面。

### 9.4 Birkhoff polytope 的"柔性表达力"

Birkhoff polytope 的极点 (permutation matrix) 是"硬切换 stream"，内部点是"软混合"。让 $\mathcal{H}^{\text{res}}$ 在训练中可从接近 $I$ (identity permutation) 滑向接近其他 permutation — 这给网络一个 **离散拓扑演化的连续松弛**。可能比 Mixture-of-Depths / Branch-Train-Merge 等"硬拓扑选择"方法更易训。

### 9.5 与 Neural ODE / Continuous-depth 的暗合

如果 $\mathcal{H}^{\text{res}} \approx I$ (Birkhoff polytope 里的恒等顶点)，那 mHC 的递归形式 $\mathbf{x}_{l+1} = \mathcal{H}^{\text{res}} \mathbf{x}_l + \text{small}$ 接近 Euler discretization of $\dot{\mathbf{x}} = f(\mathbf{x})$。**Birkhoff 约束保证了离散化的稳定性** (类似 symplectic integrator 保能量)。这跟 Neural ODE / ResNet-as-flow 那一脉有精神上的连结。

参考:
- Neural ODE (Chen et al., 2018): https://arxiv.org/abs/1806.07366
- Augmented Neural ODE: https://arxiv.org/abs/1904.01681

### 9.6 $\alpha$ 初始化为 0.01 的意义

$\alpha_l^{\text{res}} = 0.01$ 让 dynamic 部分起步非常小，$\mathcal{H}^{\text{res}} \approx \mathbf{b}^{\text{res}}$ (static bias)。$\mathbf{b}^{\text{res}}$ 应该初始化为接近 $I/n$ (uniform doubly stochastic) 或 $I$ (identity permutation)，让 mHC 在训练初期退化成普通 ResNet (n=1 的特殊情形)。这是 **warm-start strategy** — 让模型先学会"用 ResNet 的方式工作"，再渐进引入 stream mixing。和 skip-connection initialization (例如 Fixup, ReZero, SkipInit) 是同一思路。

参考 ReZero: https://arxiv.org/abs/2003.02867  
参考 SkipInit: https://arxiv.org/abs/2006.07926

---

## 10. 几个值得深挖的实验细节 (Appendix A.1)

| 属性 | 3B | 9B | 27B | 3B 1T |
|---|---|---|---|---|
| Total Params | 2.97B | 9.18B | 27.0B | 2.97B |
| Active Params | 612M | 1.66B | 4.14B | 612M |
| Layers | 12 | 18 | 30 | 12 |
| Routed Experts | 64 | 64 | 72 | 64 |
| Active Experts | 6 | 6 | 6 | 6 |
| Shared Experts | 2 | 2 | 2 | 2 |
| Dimension | 1280 | 1920 | 2560 | 1280 |
| FFN Dim | 896 | 1280 | 1536 | 896 |
| Attention | MLA | MLA | MLA | MLA |
| KV Rank | 512 | 512 | 512 | 512 |
| RoPE θ | 10000 | 10000 | 10000 | 10000 |
| RMSNorm ε | 1e-20 | 1e-20 | 1e-20 | 1e-20 |
| **mHC n** | 4 | 4 | 4 | 4 |
| **mHC α init** | 0.01 | 0.01 | 0.01 | 0.01 |
| **Sinkhorn t_max** | 20 | 20 | 20 | 20 |
| Training Tokens | 39.3B | 105B | 262B | 1.05T |

观察：
- ε = 1e-20 (极端小) — 配合 RMSNorm 的"几乎不除"行为，让 mHC 的信号守恒更纯粹
- DeepSeek-V3 全家桶：MLA + MoE + Loss-Free balancing + DualPipe + RoPE
- KV rank 512: MLA 的低秩 KV cache
- 6 active experts + 2 shared: 标准 DeepSeek 路由

### 10.1 关于 Learning rate 的细节
- 3B: 8.6e-4, 9B: 5.9e-4, 27B: 4.0e-4 — 与 Chinchilla scaling law 一致的递减趋势
- Decay schedule: step decay at 0.8×, 0.9× steps with 0.316, 0.1 multipliers
- 3B 1T 用 9.0e-4 (因为只跑 1T，可以更激进)

---

## 11. 我会有的几个质疑 / 后续可问的问题

1. **为什么 tanh 被去掉？** 原始 HC 的 dynamic part 是 $\tanh(\theta \tilde{\mathbf{x}})$，mHC 改成线性 $\vec{\mathbf{x}} \varphi$。$\tanh$ 起到"dynamic range 压缩"作用，去掉后 Sinkhorn 输入的 pre-activation 可能更大。是否 $\exp$ 已经够稳定？需要 ablation。

2. **$n=4$ 是否最优？** 论文没系统扫 $n$。直觉上 $n$ 越大，Birkhoff polytope 的"permutation 凸组合空间"越丰富，但 Sinkhorn 收敛变慢，I/O 也增加。$n=8$ 或 $n=16$ 会不会更好？或是 $n=2$ 已经足够？

3. **Sinkhorn 20 步的 backward** 是怎么求导的？他们提到 "custom backward kernel that recomputes intermediate results on-chip and traverses the entire iteration"。这隐含 **through-time 的精确求导**，对 20 步迭代的 Jacobian 全展开计算。是否有 numerical issue？是否能用 implicit differentiation (类似 Sinkhorn divergence 的 envelope theorem) 更高效？

4. **Manifold 选择的最优性**：Birkhoff 是非负 + 行列和为 1。但其他 manifold (orthogonal group, special linear group, symplectic group) 可能有别的几何性质。Paper 的 "outlook" 部分明确邀请大家探索别的 manifold — 这是个开放问题。

5. **mHC 与 Muon optimizer / Adam 的交互**：DeepSeek 最近推 Muon (球面化 momentum)。$\mathcal{H}^{\text{res}}$ 投影到 Birkhoff 之后，参数空间不再是欧式，用 Muon 这种"球面优化"是否会更好？这是个有趣的方向。

6. **mHC 在 vision / diffusion 上呢？** 论文只在 LLM 上做。Diffusion transformer (DiT) 用 adaLN-zero，本质也是某种 residual gating。mHC 与 adaLN-zero 的关系？U-Net 的 skip connection 是否能换成 mHC？

参考:
- Muon optimizer: https://github.com/KellerJordan/Muon
- DiT (adaLN-zero): https://arxiv.org/abs/2212.09748

---

## 12. 一个最直觉的 mental model

我想用一个比喻给你建立 intuition：

**普通 ResNet** 是一条单车道公路。车 (信号) 每过一个路口 (layer) 都能直接通过，没有变道，所以从起点到终点的速度恒定。这就是 identity mapping。

**HC** 把公路扩成 $n$ 条车道，并在每个路口装上"任意可调节变道器"。问题是变道器是 free-form 矩阵 — 它可能把 4 条车道的车全挤到 1 条上 (信号爆)，或者随机抵消 (信号塌)。一旦车多起来，路口就堵成灾害。

**mHC** 在变道器上装了一个"信号灯约束"：保证 (i) 每条车道流入的总量等于流出的总量 (行列和为 1)，(ii) 不允许"反向车流" (非负)。这样无论怎么变道，**总流量守恒**，且变道是"凸组合式的"软切换。再加 6.7% 的红绿灯运营成本 (infrastructure overhead)，整个系统能 scale 到 27B+ 还稳定运行。

---

## 13. 总结

mHC 的贡献是**两件事**：
1. **理论层面**：识别出 HC 的 instability 来自 unconstrained residual mapping 破坏 identity mapping，提出在 Birkhoff polytope 上做 manifold projection 修复"信号守恒"，并给出严格的 norm bound / closure / 几何解释三连击。
2. **系统层面**：把 6.7% overhead 落到具体工程上 — TileLang kernel fusion + 块状 recompute + DualPipe 重叠 — 让"理论上美好"的东西在 27B MoE 上真能跑。

它的真正意义在于 **"topological complexity ≠ 必须破坏 identity"** 这个论点被实证了。这打开了一个空间：将来 macro-architecture 设计可以更激进地探索拓扑，只要找到合适的 manifold 把 residual mapping 圈住就行。Paper 的 "outlook" 段已经在邀请大家探索别的 manifold (orthogonal, low-rank, symplectic...)。

我个人觉得这是 2024-2025 macro-architecture revival 里最 solid 的一篇，因为它**不回避系统成本、不回避稳定性实证、几何解释干净**。HC 是个有潜力的 idea 但有"原罪"，mHC 是它的"赎罪 + 升华"。考虑到作者团队 (Zhenda Xie 等 DeepSeek-AI) 之前做了 MLA、Loss-Free balancing、DualPipe — 他们对"如何在 MoE LLM 上把抽象数学 trick 落到 fused kernel" 有非常成熟的 muscle memory，这是其他学术组难以复现的护城河。

---

## 主要参考链接汇总

- mHC paper: https://arxiv.org/abs/2507.13791 (推测；arXiv ID 待官方发布)
- HC (Zhu et al., 2024): https://arxiv.org/abs/2409.19606
- ResNet: https://arxiv.org/abs/1512.03385
- Identity Mappings: https://arxiv.org/abs/1603.05027
- Sinkhorn-Knopp 1967: https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-21/issue-2/Concerning-nonnegative-matrices-and-doubly-stochastic-matrices/pjm/1102993855.full
- Birkhoff-von Neumann theorem: https://en.wikipedia.org/wiki/Birkhoff%E2%80%93von_Neumann_theorem
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- DeepSeek-V2 (MLA): https://arxiv.org/abs/2405.04434
- TileLang: https://arxiv.org/abs/2504.17577
- MUDDFormer: https://arxiv.org/abs/2502.12170
- DenseFormer: https://openreview.net/forum?id=kMnoh7CXrq
- RMT: https://arxiv.org/abs/2506.22696
- Highway Networks: https://arxiv.org/abs/1505.00387
- DenseNet: https://arxiv.org/abs/1608.06993
- FractalNet: https://arxiv.org/abs/1605.07648
- DLA: https://arxiv.org/abs/1707.06443
- DeepCrossAttention: https://openreview.net/forum?id=j3JBfFnGYh
- LAurel: https://openreview.net/forum?id=rUDRWP9WvZ
- ResiDual: https://arxiv.org/abs/2304.14802
- Highway Transformer: https://aclanthology.org/2020.acl-main.616/
- Cross-layer retrospective (Fang 2023): https://openreview.net/forum?id=pvgEL1yS3Ql
- Switch Transformer: https://arxiv.org/abs/2101.03961
- FlashAttention: https://arxiv.org/abs/2205.14135
- Zero Bubble Pipeline: https://arxiv.org/abs/2401.10241
- Chinchilla scaling laws: https://arxiv.org/abs/2203.15556
- Neural ODE: https://arxiv.org/abs/1806.07366
- DiT (adaLN-zero): https://arxiv.org/abs/2212.09748
- ReZero: https://arxiv.org/abs/2003.02867
- SkipInit: https://arxiv.org/abs/2006.07926
- Cuturi Sinkhorn distance: https://arxiv.org/abs/1306.0895
- Muon optimizer: https://github.com/KellerJordan/Muon
