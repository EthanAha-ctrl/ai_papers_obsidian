---
source_pdf: Cut Less, Fold More Model Compression through the Lens of Projection Geometry.pdf
paper_sha256: 34190871c0435fc3727865cfc4ba7bf42fb3c16845b86b0d703a65c10158e6fa
processed_at: '2026-08-03T18:10:51-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Cut Less, Fold More

Andrej, 好嘞, 我换个更口语的角度重新捋一遍, 重点是让你抓 intuition 而不陷入公式细节。

---

## 这篇 paper 到底想说什么

一句话版本: **pruning 把权重直接砍成 0 扔掉, folding 把相似权重 merge 到一起共享输出, 后者在数学上严格更优, 实验上也确实更好。**

核心 insight 其实特别简单, 但 paper 用 projection geometry 的语言把它 formalize 了, 还给了 theorem 证明。

---

## 为什么 pruning 是一个很粗暴的操作

想象你有一个 layer, $m$ 个 output channels, 每个 channel 有一个 weight vector $w(i)$, 维度 $p$。你训练完得到 $\mathbf{W} \in \mathbb{R}^{m \times p}$。

Magnitude pruning 的操作: 算每个 row 的 norm $\|w(i)\|_2$, 把 norm 小的 row 直接 zero out, 然后删掉。剩下的 $k$ 个 row 组成新的 layer。

这里有个细节你可能没仔细想过: **被 zero out 的 row, 信息全丢了, 变成 origin $\mathbf{0}$**。

如果你画图, pruning 的 reconstruction error 是:
$$\text{error}_{\text{prune}} = \sum_{i \in \text{pruned}} \|w(i) - 0\|_2^2$$

也就是 pruned rows 到 origin 的距离平方和。

---

## Folding 的核心 idea

Folding 换了个思路: **既然这些 row 要被"压缩", 与其扔到 origin, 不如把它们聚成 cluster, 用 cluster mean 替代**。

举个例子, 假设你 prune 掉 10 个 row, $w(1), \ldots, w(10)$。Pruning 把它们全设为 0。Folding 把它们聚成 1 个 cluster, 用 $\mu = \frac{1}{10}\sum_i w(i)$ 替代这 10 个 row, 然后通过下一层的合并重组来复用这一个 channel。

Reconstruction error:
$$\text{error}_{\text{fold}} = \sum_{i=1}^{10} \|w(i) - \mu\|_2^2$$

这是一个经典的统计学事实: **对一个集合, 最小化到所有点距离平方和的 representative point, 就是 mean**。Origin 永远不会比 mean 更好 (除非 mean 恰好是 origin)。

所以 folding 的 error 永远 ≤ pruning 的 error。这就是 Theorem 2.1。

---

## 为什么这个直觉可以 generalize

上面我只讲了一个 cluster 的情况。Folding 实际可以做 $k$ 个 cluster, 用 k-means 找最优分配。

Theorem 2.2 说: 对任何 pruning rank $k_p$, 总存在一个 $k_f = k_p + 1$ 个 cluster 的 k-means folding, 它的 reconstruction error 严格不大于 pruning 的。

理由很简单: k-means folding 是一个更大的 operator class 的最优解, pruning 只是其中一种特殊的 degenerate cluster assignment (每 row 自己一个 cluster + 多余的 row 全归到一个"被 zero out"的 cluster, 但这个 cluster 的 mean 不是 origin 而是它们自己的 mean)。

Folding 用 mean 替代 zero, 多了一个 cluster, 但保留更多信息。

---

## 那 "+1 rank" 的 slack 不公平吗

这是最容易被 reviewer 攻击的点。Paper 的回应:

**理论层面**: 这个 slack 只是证明工具, 不是 empirical protocol。实验都是严格 matched budget。

**实验层面** (Appendix F.3, Figure 23): 他们量化了两件事:
- $\Delta_{\text{rank}}(k)$: pruning 从 $k$ 增到 $k+1$ 个 retained row, Frobenius error 改善多少
- $\Delta_{\text{method}}(k)$: 同 rank 下从 pruning 换到 folding, error 改善多少

结果: $\Delta_{\text{method}}$ 比 $\Delta_{\text{rank}}$ 大 **1-2 个数量级**。也就是说, folding 的优势**主要不是来自多一个 cluster**, 而是来自 cluster projection 本身比 axis-aligned projection 更 expressive。

而且实际数字看, 256 channels 保留 50%, $k_p=128$ vs $k_f=129$, 相对开销 0.78%。ViT 768 宽保留 50%, 相对开销 0.26%。完全可以忽略。

---

## Projection geometry 视角的真正威力

Paper 把两者统一为 orthogonal projection:

$$\mathbf{W}_{\text{compressed}} = \mathbf{C} \mathbf{W}$$

- Pruning 的 $\mathbf{C}_p = \mathrm{diag}(1, \ldots, 1, 0, \ldots, 0)$, projection subspace 由 standard basis 张成 (axis-aligned)
- Folding 的 $\mathbf{C}_f = \mathbf{U}_f (\mathbf{U}_f^\top \mathbf{U}_f)^{-1} \mathbf{U}_f^\top$, 其中 $\mathbf{U}_f$ 是 binary cluster indicator, projection subspace 由 cluster 结构张成

关键差别: **pruning 的 subspace 是 fixed coordinate-aligned, folding 的 subspace 是 cluster-structured, 可以看作在 row space 里 "旋转" 来更好地 fit 数据**。

这个视角让你立刻看到: pruning 是 folding 的一个特例 (用 $\mathbf{0}$ 当 cluster mean)。Folding 是 richer operator class, 必然不差。

---

## 从 parameter error 到 loss perturbation

Pruning/Folding 改变 weight, 自然改变 loss。Paper 假设 loss 是 Lipschitz:
$$|\mathcal{L}(\mathbf{W}) - \mathcal{L}(\mathbf{W}')| \leq \kappa \|\mathbf{W} - \mathbf{W}'\|_F$$

- $\mathcal{L}$: training loss
- $\mathbf{W}'$: 压缩后的 weight
- $\kappa$: Lipschitz constant, 衡量 loss 对 weight 扰动的敏感度
- $\|\cdot\|_F$: Frobenius norm

所以 Frobenius error 小 ⟹ loss 扰动 bound 小。结合 Theorem 2.2, folding 给出 provably tighter loss perturbation bound。

**但是**: $\kappa$ 在 sharp minima 会非常大 (loss landscape 像针尖)。这种情况下, 即使 Frobenius error 差异小, 实际 loss 扰动可能很大, 理论保证失效。

这就是为什么 paper 在 Adam 高 lr 下偶尔观察到 folding 反而输给 pruning - Adam 高 lr 倾向于找到 sharp minima (Wilson 2018, https://arxiv.org/abs/1705.08292; Zhou 2021, https://arxiv.org/abs/2010.05627), $\kappa$ 极大, Frobenius error 不再预测 accuracy。

---

## Sharpness 分析: 机制层面的解释

Appendix F.1 (Figure 19-22) 做了 sharpness 度量:
$$\text{Sharpness}(\rho) = \max_{\|\epsilon\|_\infty \le \rho} \frac{\mathcal{L}(\mathbf{W} + \epsilon) - \mathcal{L}(\mathbf{W})}{\rho^2}$$

- $\rho$: 扰动半径
- $\epsilon$: weight perturbation
- $\|\cdot\|_\infty$: max-norm

发现:
1. Compression ratio 越高, sharpness 越高 (直到 basin 被打破)
2. **FOLD 的 sharpness 系统性低于 MAG1**
3. **Adam 训练下, $\Delta$ sharpness 和 $\Delta$ accuracy 强负相关** (Pearson -0.93 at 40% compression)

直觉解释: Folding 沿 cluster 方向投影, 保留 directional structure, 模型还在原来的 basin 里。Pruning 直接 zero out, 把 model 从 basin 里踢出来, sharpness 急剧上升。

Adam 高 lr 下, basin 本来就窄, 任何 projection 都可能踢出 basin, folding 的 sharpness 优势消失, 所以 accuracy 优势也消失。

---

## 实验细节的几个关键点

### 规模

>1,000 checkpoints, 横扫 hyperparameters (optimizer, lr, augmentation, SAM, regularization)。这个 scale 在 compression paper 里非常罕见 - 大多数 pruning paper 只跑一个 recipe 几个 seed。

### 严格 matched budget

Table 9, 10 显示: **FOLD 和 MAG 在每个 layer 上的 params, FLOPs, activations, non-zero activations 完全相同**。也就是说, 压缩后两个模型的推理 cost 一模一样, 性能差异纯粹来自 weight 质量。

这点很重要, 否则 reviewer 会质疑 "是不是 folding 保留了更多 capacity"。

### Runtime 成本

FOLD 的 compression time 比 MAG 慢 5x 左右 (ResNet18: 9.48s vs 1.77s, CLIP ViT-B/32: 92.83s vs 2.63s)。因为 k-means 是迭代的, pruning 是 one-pass scoring。

但这是一次性成本, 推理 latency 完全一样 (3.17 vs 3.15 ms/batch for ResNet)。

---

## Failure cases: 什么时候 folding 输

Paper 诚实地列了几个 folding 不占优的 regime:

1. **Adam + 极高 learning rate**: sharp minima, $\kappa$ 巨大, Lipschitz bound 失效
2. **极低 lr + 长 warmup** (LLaMA-60M, Table 1 个别行): training 太稳定, weight 已经很集中, clustering 没什么可 cluster 的
3. **极低 compression ratio (<10%)**: pruning 也没删多少 channel, 差距本来就小, 有时噪音反转
4. **强 data augmentation + CNN**: augmentation 已经让 CNN 变得很 robust, axis-aligned removal 不那么致命, folding 优势缩窄

ViT 的行为有意思: RandAugment 反而扩大 FOLD 优势 (Figure 7d)。解释: ViT 本来没有 strong inductive bias, augmentation 帮助 ViT 形成 directional structure, 给 folding 提供了 clustering 的素材。

---

## SAM 和 folding 的协同效应

SAM (Sharpness-Aware Minimization, Foret 2021, https://arxiv.org/abs/2010.01412) 训练时找 flat minima:
$$\min_w \max_{\|\epsilon\| \le \rho} \mathcal{L}(w + \epsilon)$$

Figure 6 显示: SAM 提升 FOLD 和 MAG1 两者, 但 FOLD 提升更多, 尤其是 SAM ρ 较小时。

直觉: SAM 把 model 拉到 flat basin, 此时 folding 的 cluster projection 路径更平滑地保留 basin 结构, pruning 直接 zero out 仍然有破坏性。SAM 让 folding 的几何优势更显著。

但 ρ 过大时, SAM 把 basin 拉得太平, 所有 projection 路径都在 robustness ball 内, folding 优势缩窄。

---

## 跟其他 compression 方法的关系

### Quantization

改精度不改结构, calibration 通常必需。和 folding 完全正交, 可以叠加 (paper 说这是 future work)。

### Low-rank factorization (SVD, CP-decomposition)

Low-rank 是 continuous subspace projection (任意 basis), 表达力比 folding (discrete cluster indicator) 更强。但需要 fine-tuning 恢复, folding 是 calibration-free。可以这样理解: **folding 是 low-rank projection 的 discrete, calibration-free 近似**。

Lebedev CP-decomposition: https://arxiv.org/abs/1412.6553
Horvath Maestro: https://arxiv.org/abs/2308.14929
Han SLTrain: https://arxiv.org/abs/2406.02214

### SparseGPT / Wanda

SparseGPT (Frantar 2023, https://arxiv.org/abs/2301.00734) 用 calibration 数据 + second-order 信息做 unstructured pruning, Wanda (Sun 2024, https://arxiv.org/abs/2306.11695) 用 activation magnitude。这两个都还是 axis-aligned pruning 的变体, 只是 selection criterion 更聪明。Paper 的几何框架可以推广到这些方法 - 它们的 projection subspace 仍然是 axis-aligned, 只是 selector 不同。

### Model merging / Git Re-Basin

Model Soups (Wortsman 2022a, https://arxiv.org/abs/2203.05482) average 多个 fine-tuned model。Git Re-Basin (Ainsworth 2023, https://arxiv.org/abs/2209.04836) 通过 neuron permutation alignment 合并。这些是 inter-network 融合, folding 是 intra-network 融合压缩。但都依赖同一个核心现象: 神经网络有 neuron permutation symmetry 和 redundancy, 可以被 merge 而不失功能。

Entezari 2022 (https://arxiv.org/abs/2110.06296) 显示 permutation invariance 导致 linear mode connectivity。Folding 可以看作在单网络内部利用这个 symmetry 的 self-merging。

---

## Folding 的局限和我的几个疑虑

### 1. k-means 最优解 NP-hard

Theorem 2.2 假设 optimal k-means。但 k-means 在 high-dim 上 NP-hard (Mahajan et al. 2009), 实际用 Hartigan-Wong (Hartigan & Wong 1979, https://doi.org/10.2307/2346830) 的局部解。当 $p$ 很大 (LLaMA hidden dim 4096+), clustering 质量可能下降, 理论保证有 gap。

Paper 没讨论这个, 但我猜实际影响有限, 因为 $m$ (cluster 的对象数量) 不大, k-means 在中等 $m$ 上通常收敛到接近全局最优。

### 2. Hardware 友好性

Folding 后 weight 在内存里仍是 dense matrix, 只是某些 row 相同。PyTorch CUDA kernel 不会自动 dedup。所以推理 latency 和 pruning 一样, 但内存占用没省。理论上可以 custom kernel 去重, 但工程复杂度不低。

### 3. 跨层依赖

Paper 的理论是 single-layer 的。但 folding 一个 layer 后, 下一层的 input distribution 变了 (因为多个 row 输出相同), 下一层 folding 的 optimal cluster 也会变。Paper 的实验是 full-network 的, 但理论没建模跨层 interaction。

### 4. Attention layer 没做

ViT 和 LLaMA 实验只折叠 FFN block, attention 的 Q/K/V projection 没折叠。理由可能是 attention 的 structure 更复杂 (head 维度, multi-head interaction), k-means clustering 不直接适用。Paper 说这是 future work。

---

## 一些更深的联想

### Folding 和神经科学里的 redundancy

神经科学里, "neuron redundancy" 是个经典现象 - 大量 neuron 做相似的事情, 失去一部分不影响功能。Folding 本质上是在 explicit 地利用这个 redundancy, 把 functionally equivalent 的 neuron merge。

Pruning 则是"减员", 利用 redundancy 但通过删除。两种策略都依赖 redundancy 存在, 但 folding 保留了 redundancy 的"代表", pruning 扔掉。

### Folding 和 information bottleneck

如果把 layer output 看作 random variable $Y = \mathbf{W}X$, folding 把 $Y$ 限制在一个低维 cluster manifold 上 (每个 cluster 输出相同)。这有点像 information bottleneck (Tishby 2015), 但目标不同 - IB 是最大化 mutual information compression, folding 是最小化 weight reconstruction error。

### Folding 和 weight tying

Weight tying (embedding 和 output projection 共享, 见 Press & Wolf 2017, https://arxiv.org/abs/1608.05859) 是 hand-coded 的 weight sharing。Folding 是 data-driven, learned 的 weight sharing (通过 clustering 发现哪些 weight 可以 share)。两者都利用 redundancy, 但 folding 自动发现。

### Folding 和 Mixture of Experts

MoE (Shazeer 2017, https://arxiv.org/abs/1701.06538) 是反向操作: 把一个 dense layer 拆成多个 expert, 每个 input 只激活部分 expert。Folding 是把多个相似 expert merge 成一个 dense。某种意义上 MoE 和 folding 是同一个 spectrum 的两端 - "展开" vs "折叠"。

### 和 LLM compression 的关系

LLM pruning 最近很火 (SparseGPT, Wanda, ShortGPT), 但都是 unstructured 或 semi-structured, 需要 calibration。Folding 是 structured, calibration-free。但 paper 只在 LLaMA-60M/130M 上做, 没在 7B+ 上验证。在大 LLM 上, folding 的 k-means 成本会更高, 但仍是一次性。

如果 folding 在大 LLM 上有效, 它可能成为 SparseGPT/Wanda 的 structured 替代品, 无需 calibration data。

### Lottery Ticket Hypothesis 的视角

Lottery Ticket (Frankle & Carbin 2019, https://arxiv.org/abs/1803.03635) 说 pruned network 可以 retrain 回到原性能。Folding 的视角补充: 如果 lottery 是"找到一个 sub-network", 那 folding 是"找到一个 sub-network + 共享 representative"。Folding 的"彩票"更容易找到, 因为它有更大的 solution space (cluster assignment 灵活)。

---

## 一个具体的 mental model

想象你训练了一个 ResNet, 某个 conv layer 有 256 个 filter。你想压缩到 128 个。

**Magnitude pruning**: 算每个 filter 的 norm, 删掉 128 个最小的, 留下 128 个 norm 最大的。剩下的 128 个 filter 完全不动, 删掉的 128 个 filter 的计算能力归零。

**Folding**: 用 k-means 把 256 个 filter 聚成 129 个 cluster (注意是 129, 不是 128, 但相对开销 0.78%)。每个 cluster 的所有 filter 用 cluster mean 替代。然后下一层的 conv 把来自同一个 cluster 的 input channel 相加合并。

为什么 folding 更好? 因为**被"压缩"的 128 个 filter 并非完全独立 - 它们之间可能有大量 directional overlap**。Pruning 把它们全设为 0, 浪费了它们的公共方向。Folding 把它们 collapse 到 mean, 保留了公共方向, 只丢了 intra-cluster variance。

这个 intra-cluster variance 就是 pruning 额外付出的 cost, 在 paper 里量化为 $(m-k_p)\|\mu\|_2^2$。

---

## 整体评价

这篇 paper 的核心贡献其实就两个 theorem 和一堆实验。理论部分很 clean, 但限于 single-layer Lipschitz bound。实验部分非常 thorough, 在 compression paper 里 scale 罕见。

我觉得最有意思的 insight 是: **compression operator 的几何性质 (axis-aligned vs cluster-structured) 直接影响压缩后的 weight 质量和 loss landscape 行为**。这给 compression method design 提供了一个新的设计维度 - 不只是 "prune 什么" 或 "factorize 成什么", 而是 "project 到什么形状的 subspace"。

未来的方向可能是: 探索更丰富的 projection geometry (比如低秩 + cluster hybrid), 或者 calibration-aware folding (用 activation 信息指导 clustering, 类似 Wanda 但保持 cluster structure)。

希望这版"人话"帮你 build intuition, Andrej! 核心就是 mean 比 0 更好这么个朴素道理, 被 projection geometry 的语言包装后变得可证明可量化。

---

# Cut Less, Fold More: 从 Projection Geometry 视角讲 Model Compression

Andrej, 这篇 paper 给了一个相当 elegant 的理论视角来对比两种 calibration-free post-training compression 方法。核心 insight 是: **structured pruning 和 model folding 都可以 cast 为 parameter space 上的 orthogonal projection**, 但是它们的 projection geometry 本质不同 - 一个是 axis-aligned, 一个是 cluster-structured。下面我从直觉到公式到实验逐层剖析。

---

## 1. 核心直觉

考虑一个 layer 的 weight matrix $\mathbf{W} \in \mathbb{R}^{m \times p}$, $m$ 行, 每行是一个 parameter vector $w(i) \in \mathbb{R}^p$。

- **Magnitude pruning** 直接把某些 row 设为 0, 然后删除。直观上: 把 weight matrix 投影到保留的 axis-aligned coordinate subspace 上。被删除 row 信息完全丢失。
- **Model folding** 把相似 row 聚类成 $k$ 个 cluster, 每个 cluster 用 cluster mean 替换, 然后 tie 在一起共享输出, 通过下一层的合并重组维持近似功能。直观上: 投影到 cluster-structured subspace, 保留了 directional information。

关键直觉是: **如果两个 pruned row 本身有大量 directional overlap, 直接 zero out 浪费了它们的平均结构**, 而 folding 把它们 collapse 到均值上, 还保留了它们的公共部分。这一点在 Frobenius norm 上是可以严格证明的。

---

## 2. 公式解析: 两种 projection operator

### 2.1 Orthogonal Projection 基础

给定一个 $k$-维 subspace 的 basis $\mathbf{U} \in \mathbb{R}^{m \times k}$ (column orthogonal 或一般 column rank $k$), 对应的 orthogonal projection matrix 是:

$$\mathbf{C} = \mathbf{U}(\mathbf{U}^\top \mathbf{U})^{-1}\mathbf{U}^\top$$

- 上标 $\top$: matrix transpose
- $\mathbf{U}^\top \mathbf{U}$: $k \times k$ 的 Gram matrix
- $(\mathbf{U}^\top \mathbf{U})^{-1}$: inverse Gram matrix, 保证 projection 是 orthogonal
- 满足 $\mathbf{C} = \mathbf{C}^\top = \mathbf{C}^2$, 即 symmetric 且 idempotent

几何意义: $\mathbf{C}y$ 是 $y$ 到 $\mathrm{Range}(\mathbf{U})$ 的 Euclidean closest point。

### 2.2 Pruning 作为 axis-aligned projection

假设我们保留前 $k$ 个 row, prune 掉后 $m-k$ 个, 那么:

$$\mathbf{U}_p = \binom{\mathbf{I}_k}{\mathbf{0}} \in \mathbb{R}^{m \times k}$$

- $\mathbf{I}_k$: $k \times k$ identity matrix
- 下方的 $\mathbf{0}$ 是 $(m-k) \times k$ zero block
- 每列是一个 standard basis vector $e_i \in \mathbb{R}^m$

因此 $\mathbf{C}_p = \mathbf{U}_p \mathbf{U}_p^\top = \mathrm{diag}(1, 1, \ldots, 1, 0, \ldots, 0)$, 作用在 $\mathbf{W}$ 上:

$$\mathbf{W}_p = \mathbf{C}_p \mathbf{W}$$

后 $m-k$ 行变 0。直观几何: 这是投影到一个由 $k$ 个 coordinate axis 张成的 subspace, axis 之间的夹角被强制为 90°, 无法旋转。

### 2.3 Folding 作为 cluster-structured projection

Folding 用 $k$ 个 cluster, $\mathbf{U}_f \in \{0,1\}^{m \times k}$, 每行恰好一个 1:

$$u_f(i,j) = 1 \iff i \in S_j$$

- $S_j$: 第 $j$ 个 cluster 的 index 集合
- $\sum_j u_f(i,j) = 1$ for all $i$ (硬分配约束)

注意 $\mathbf{U}_f^\top \mathbf{U}_f = \mathrm{diag}(|S_1|, |S_2|, \ldots, |S_k|)$, 是 cluster 大小的对角阵。所以:

$$\mathbf{C}_f = \mathbf{U}_f (\mathbf{U}_f^\top \mathbf{U}_f)^{-1} \mathbf{U}_f^\top$$

具体 entry: $C_f(i, i') = \frac{1}{|S_j|}$ if $i, i' \in S_j$, else $0$。投影效果:

$$w_f(i) = \mu_j = \frac{1}{|S_j|} \sum_{i' \in S_j} w(i') \quad \forall i \in S_j$$

即每个 row 被替换成 cluster mean。这是经典的 k-means clustering objective 等价形式 (Bauckhage 2015, https://arxiv.org/abs/1512.07548)。

**几何差异**: Pruning 的 subspace 由 axes 决定 (固定), folding 的 subspace 由 cluster assignment 决定 (可旋转 - cluster 可以任意选择 weights 的子集, 等价于在 row space 中形成 hyper-tetrahedron 风格的 basis)。Folding 是 richer operator class。

---

## 3. 两个核心 Theorem

### 3.1 Theorem 2.1: 存在性结果 (folding + 1 rank 必胜 pruning)

**陈述**: 给定任何 pruning $\mathbf{U}_p$ rank $k_p$ ($0 \le k_p \le m-1$, 至少 prune 一个 row), 存在 folding $\mathbf{U}_f'$ rank $k_f = k_p + 1$ 使得:

$$\|\mathbf{W} - \mathbf{W}_p\|_F^2 \geq \|\mathbf{W} - \mathbf{W}_f'\|_F^2$$

- 下标 $p$: pruning
- 下标 $f'$: special folding (构造性的)
- $\|\cdot\|_F$: Frobenius norm, $\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$

**构造**: 把被 pruning 删掉的 $m - k_p$ 行全部合并到一个 cluster, 其余保留 row 各自一个 cluster。

**证明** (我把每一步写细一点):

不失一般性排列 $\mathbf{W}$, 让 pruned rows 在前: $w(1), \ldots, w(m-k_p)$。

**Pruning 误差**:
$$\mathbf{W} - \mathbf{W}_p = \begin{pmatrix} w(1) \\ \vdots \\ w(m-k_p) \\ \mathbf{0} \\ \vdots \\ \mathbf{0} \end{pmatrix}$$
$$\|\mathbf{W} - \mathbf{W}_p\|_F^2 = \sum_{i=1}^{m-k_p} w(i)^\top w(i)$$

(每一行 squared L2 norm 加起来)

**Folding 误差** (用 cluster mean $\mu = \frac{1}{m-k_p} \sum_{i=1}^{m-k_p} w(i)$):
$$\mathbf{W} - \mathbf{W}_f' = \begin{pmatrix} w(1) - \mu \\ \vdots \\ w(m-k_p) - \mu \\ \mathbf{0} \\ \vdots \\ \mathbf{0} \end{pmatrix}$$

$$\|\mathbf{W} - \mathbf{W}_f'\|_F^2 = \sum_{i=1}^{m-k_p} (w(i) - \mu)^\top (w(i) - \mu)$$

展开:
$$= \sum_{i=1}^{m-k_p} \bigl[ w(i)^\top w(i) - 2 w(i)^\top \mu + \mu^\top \mu \bigr]$$

$$= \sum_{i=1}^{m-k_p} w(i)^\top w(i) - 2 \Bigl(\sum_{i=1}^{m-k_p} w(i)\Bigr)^\top \mu + (m-k_p)\mu^\top \mu$$

由 $\mu$ 的定义 $\sum_i w(i) = (m-k_p)\mu$, 代入:
$$= \sum_i w(i)^\top w(i) - 2(m-k_p)\mu^\top \mu + (m-k_p)\mu^\top \mu$$
$$= \sum_i w(i)^\top w(i) - (m-k_p)\mu^\top \mu$$
$$\leq \sum_i w(i)^\top w(i) = \|\mathbf{W} - \mathbf{W}_p\|_F^2$$

最后一步用了 $(m-k_p)\mu^\top \mu \geq 0$。

**直觉解释**: pruning 把 row 设为 0 等价于用 origin $\mathbf{0}$ 替代这些 row, 而 folding 用它们的 mean 替代。Mean 到各 row 的距离平方之和总是严格小于等于 origin 到各 row 的距离平方之和 (mean 是 L2-optimal representative)。差值正好是 $(m-k_p) \|\mu\|_2^2$, 即 cluster 的 between-cluster variance。Pruning 完全浪费了这部分方差信息。

### 3.2 Theorem 2.2: k-means folding 是 cluster-structured projection 中的最优

**陈述**: 令 $\mathbf{U}_f^*$ 是 k-means 最优解 (在 $k_f$ 个 cluster 中最小化 within-cluster sum of squares)。则对任何 pruning rank $k_p = k_f - 1$:

$$\|\mathbf{W} - \mathbf{W}_p\|_F^2 \geq \|\mathbf{W} - \mathbf{W}_f^*\|_F^2$$

**证明**: k-means 的 objective 可以写为约束矩阵分解:
$$\min_{\mathbf{U}} \|\mathbf{W} - \mathbf{U}(\mathbf{U}^\top\mathbf{U})^{-1}\mathbf{U}^\top \mathbf{W}\|_F^2$$
约束: $u(i,j) \in \{0,1\}$, $\sum_j u(i,j) = 1$。

这正好是 folding 的 orthogonal projection (Eq. 2 + Eq. 4)。由 Theorem 2.1, $\mathbf{U}_f'$ 是可行解, 但 k-means 的 $\mathbf{U}_f^*$ 给出最小误差, 所以:
$$\|\mathbf{W} - \mathbf{W}_p\|_F^2 \geq \|\mathbf{W} - \mathbf{W}_f'\|_F^2 \geq \|\mathbf{W} - \mathbf{W}_f^*\|_F^2$$

**重要含义**: 这给出严格三层 ordering: pruning error ≥ special folding error ≥ optimal k-means folding error。

### 3.3 从 reconstruction error 到 functional perturbation

假设 loss $\mathcal{L}$ 是 Lipschitz continuous (Eq. 1):
$$|\mathcal{L}(\mathbf{W}_1) - \mathcal{L}(\mathbf{W}_2)| \leq \kappa \|\mathbf{W}_1 - \mathbf{W}_2\|_F$$

- $\kappa > 0$: Lipschitz constant, 衡量 loss 对 parameter 扰动的局部敏感度
- 越小 $\kappa$ 越平坦, 越大 $\kappa$ 越尖锐

直接套用:
$$|\mathcal{L}(\mathbf{W}) - \mathcal{L}(\mathbf{W}_f^*)| \leq \kappa \|\mathbf{W} - \mathbf{W}_f^*\|_F \leq \kappa \|\mathbf{W} - \mathbf{W}_p\|_F$$

所以 folding 给出 provably tighter loss perturbation bound。

**重要 caveat** (paper 在 Appendix F.3 也讨论了): 当 local $\kappa$ 极大 (sharp minima, 如 Adam 高 lr 训练时), 即使 Frobenius 误差小, loss 扰动也可能极大, 理论保证失效。这是为什么 paper 发现极端高 lr 下 folding 偶尔反超 pruning 失败 - 不是理论错, 是 Lipschitz 假设不成立。

---

## 4. 一秩 slack 的问题

Theorem 2.1/2.2 比较 pruning rank $k_p$ vs folding rank $k_p+1$, 多了一个 cluster。这看似不公平, 但 paper 论证:

**理论**: slack 仅是证明技术工具, 实验协议严格 matched budget。

**实验验证** (Appendix F.3, Figure 23):
- $\Delta_{\text{rank}}(k) = \frac{\|\mathbf{W} - \mathbf{W}_p^{(k)}\|_F - \|\mathbf{W} - \mathbf{W}_p^{(k+1)}\|_F}{\|\mathbf{W}\|_F}$: 增加 1 个保留 rank 的相对误差改进
- $\Delta_{\text{method}}(k) = \frac{\|\mathbf{W} - \mathbf{W}_p^{(k)}\|_F - \|\mathbf{W} - \mathbf{W}_f^{*(k)}\|_F}{\|\mathbf{W}\|_F}$: 同 rank 下从 pruning 换到 folding 的相对误差改进

经验结果: $\Delta_{\text{method}}$ 比 $\Delta_{\text{rank}}$ 大 1-2 个数量级。所以 folding 的优势来自 cluster projection 本身的丰富性, 不是 +1 rank 的"作弊"。

**实际数字**: ResNet-18 stage 256 channels 50% 保留, $k_p=128, k_f=129$, 相对增加 0.78%; ViT-B/32 768 宽 50% 保留, $k_p=384, k_f=385$, 相对增加 0.26%。

---

## 5. 实验数据深入分析

### 5.1 规模

>1,000 checkpoints:
- 216 ResNet18 Adam + 576 ResNet18 SGD on CIFAR-10
- 50 PreActResNet18 + 200 ViT-B/32 from Andriushchenko et al. 2023 (https://arxiv.org/abs/2302.07011)
- 72 CLIP ViT-B/32 from Wortsman et al. 2022 Model Soups (https://arxiv.org/abs/2203.05482)
- 36 LLaMA-60M/130M on C4

### 5.2 Table 1 (LLaMA-60M) 关键观察

| Sparsity | MAG2 baseline | FOLD baseline | MAG2 advantage case |
|---|---|---|---|
| 20% | FOLD 通常更好 (e.g. 47.17 vs 54.51) | ✓ | 极低 lr + 长 warmup (32.20 baseline: 46.57 vs 47.54) |
| 50% | FOLD 大幅领先 (e.g. 221.32 vs 398.62) | ✓ | 极少 |

注意: PPL 越低越好。50% sparsity 下 FOLD 的 PPL 通常是 MAG2 的 ~50-60%, 巨大提升。20% 下差距小但 FOLD 仍占优。

**反例**: 当 weight_decay=0.01, warmup=2200, max_lr=0.001 (非常低 lr + 长 warmup) 时, baseline PPL 32.20, MAG2(20%) = 46.57 < FOLD(20%) = 47.54。这种极端 stable training 让 weight 已经很集中, folding 的 clustering 优势消失。

### 5.3 Figure 1 视觉化直觉

Scatter plot 横轴 MAG1 accuracy, 纵轴 FOLD accuracy, 颜色 = layer-wise compression ratio。点落在对角线上方 = FOLD 胜。Bar plot 显示 $\Delta = \text{Acc}(\text{FOLD}) - \text{Acc}(\text{MAG1})$ vs compression ratio。

**关键趋势**: compression ratio 越高 (50-90%), $\Delta$ 越正且越大。低 compression (<20%) 时差距小且有时反转。这与直觉吻合: 在高 compression 下, 每一个被影响的 channel 都更关键, 此时 folding 的 directional preservation 优势显著; 而在低 compression 下, 移除少数 channel 影响有限, pruning 的简单粗暴反而足够。

### 5.4 训练 hyperparameter ablation

**Learning rate** (Figure 5):
- Adam: FOLD 优势在 moderate-low lr 最大, 在 very high lr 缩窄或反转, 在极小 lr 消失 (两个都崩)
- SGD: 依赖更弱, 偶尔反转 (ViT-B/32)
- 解释: 高 lr + Adam → sharp minima → κ 极大 → Lipschitz bound 失效

**SAM** (Figure 6, Sharpness-Aware Minimization, https://arxiv.org/abs/2010.01412):
- SAM 提升 FOLD 和 MAG1 两者, 但 FOLD 提升更多
- SAM ρ 较小时差距大, ρ 较大时差距缩小 (太多 flatten 把所有 projection 都拉进 robustness ball)
- 解释: SAM 引导到 flat minima, folding 的几何优势在 flat 区域更显著

**RandAugment** (Figure 7):
- CNN (ResNet18, PreActResNet18): RAUG 缩小 FOLD 优势
- ViT-B/32: RAUG 扩大 FOLD 优势
- 解释: augmentation 引入 input perturbation invariance, 等价于 parameter-space 平滑 (Yoo & Yoon 2025, https://arxiv.org/abs/2505.24592)。CNN 已经被 RAUG 平滑很多, axis-aligned removal 不那么致命; ViT 反而需要 augmentation 才让 directional 信息有聚类价值

### 5.5 Sharpness 分析 (Appendix F.1, Figure 19-22)

Worst-case $\ell_\infty$ sharpness 定义:
$$\text{Sharpness}(\rho) = \max_{\|\epsilon\|_\infty \le \rho} \frac{\mathcal{L}(\mathbf{W} + \epsilon) - \mathcal{L}(\mathbf{W})}{\rho^2}$$

- $\rho$: perturbation radius ($10^{-4}, 5\times10^{-4}, 10^{-3}$)
- $\epsilon$: 同形 weight perturbation

**关键发现**:
1. Sharpness 随 compression ratio 上升直到中等 compression, 极端 compression 时下降 (跳出 basin)
2. FOLD 平均 sharpness 低于 MAG1
3. Adam 下 $\Delta$ sharpness 与 $\Delta$ accuracy 强负相关 (Pearson -0.93 at 40%)
4. SGD 下关系弱且分散

**Intuition**: Folding 沿 cluster-structured direction 投影, 保留了 weight 的 directional structure, 所以 loss landscape 的 basin 结构被保持, sharpness 增长慢。Pruning 直接 zero out row, 把 model 踢出 basin, sharpness 急剧上升。

### 5.6 Runtime 分析 (Table 7, 8)

PreActResNet18 64.1% compression:
- Compression time: FOLD 9.48s vs MAG 1.77s (FOLD 慢约 5x, 一次性成本)
- Inference latency 几乎相同 (3.17 vs 3.15 ms/batch) - 因为压缩后拓扑完全一样
- FLOPs 完全一样 (199.05 MFLOPs/img)

CLIP ViT-B/32 7.47% compression:
- Compression time: FOLD 92.83s vs MAG 2.63s
- 推理 latency 一样

**重要**: Table 9, 10 显示 FOLD 和 MAG 在所有 layer 上产生**完全相同**的参数量, FLOPs, activation 大小, 非 zero activations。这证明对比是严格 fair 的 - 性能差异纯粹来自 weight quality, 不来自拓扑差异。

---

## 6. 与其他 compression 方法的关系

### 6.1 Quantization vs Folding

Quantization (Darvish Rouhani et al. 2020, https://arxiv.org/abs/2005.05922) 修改数值精度, 不改结构, 通常需 calibration。
Folding 改结构, 不改精度。两者正交, 可组合 (paper 提到这是 future work)。

### 6.2 Low-rank Factorization vs Folding

Low-rank decomposition (Lebedev 2015, https://arxiv.org/abs/1412.6553; Horvath Maestro 2024, https://arxiv.org/abs/2308.14929) 把 $\mathbf{W}$ 写成 $\mathbf{A}\mathbf{B}$, $\mathbf{A} \in \mathbb{R}^{m \times r}, \mathbf{B} \in \mathbb{R}^{r \times p}$, 是 continuous subspace projection (任意 basis, SVD-like)。Folding 是 discrete cluster projection (basis 向量限制为 binary indicator)。Low-rank 表达力更强但需 fine-tuning, folding 是 calibration-free 的 (suboptimal but lightweight low-rank)。

### 6.3 SparseGPT / Wanda

SparseGPT (Frantar & Alistarh 2023, https://arxiv.org/abs/2301.00734) 用 calibration 数据做 second-order 稀疏化, 是 axis-aligned 但 sensitivity-aware。
Wanda (Sun 2024, https://arxiv.org/abs/2306.11695) 用 activation magnitude 加权。
都仍属于 axis-aligned pruning 范畴, paper 的理论框架指出它们几何上受限。

### 6.4 Model Merging / Git Re-Basin

Model merging (Wortsman 2022a Model Soups, https://arxiv.org/abs/2203.05482; Ainsworth 2023 Git Re-Basin, https://arxiv.org/abs/2209.04836; Entezari 2022, https://arxiv.org/abs/2110.06296) 跨网络融合, 不压缩。Folding 是 intra-network 合并压缩, 共享 projection 几何 but 不同 objective。

---

## 7. 限制和 Open Questions

Paper 自己列的限制:
1. 只在 FFN block 折 attention (LLaMA), attention layer 没做
2. 没和 quantization/distillation 组合
3. 没在 large LLM (7B+) 上做, 因为训练成本
4. 主要对比 magnitude pruning, 没对比 calibration-aware (SparseGPT/Wanda)

我的几点观察:

**问题 1: k-means 在 high-dim weight 的 clustering 质量**。当 $p$ 很大 (e.g. 4096 for LLM hidden dim), k-means 在 $m$ 个 $p$-dim 向量上找最优解是 NP-hard。实际用 Hartigan-Wong (https://doi.org/10.2307/2346830) 的局部最优。理论 Theorem 2.2 假设 optimal k-means, 实际只有近似。Gap 可能比理论大。

**问题 2: Folding 的 hardware 友好性**。虽然 paper Table 9/10 说 FOLD 和 MAG 推理 FLOPs 完全相同, 但实际部署中, 是否真的能像 pruning 一样得到 dense matrix 乘法的 hardware 加速? Folding 后 weight 仍 dense, 但是有大量 row 共享相同值 - 理论上可以 dedup, 但 PyTorch CUDA kernel 不一定优化。

**问题 3: 与 neuron permutation symmetry 的关系**。Git Re-Basin (https://arxiv.org/abs/2209.04836) 显示神经网络有 neuron permutation symmetry, folding 本质上是在 intra-layer 上利用这个 symmetry - 把相似的 neuron merge。所以 folding 可以看作 implicit 的 self-merging, 利用单网络内部的 neuron 冗余。

**问题 4: 折叠后 fine-tuning 的 sharpness 依赖**。Figure 3 显示 folding 后 fine-tune 1-5 epoch 仍领先。但 5 epoch 是否够长?长期 fine-tuning 后 basin 是否会回到同样位置, 拉平 folding 的初始优势?

---

## 8. 我对这篇 paper 的整体评价

**优点**:
- 理论非常 clean, projection 视角统一两个看似不同的方法
- 实验规模庞大 (>1000 checkpoints, 多 hyperparameter sweep)
- 诚实承认 failure case (高 lr Adam 反转, 极低 lr + 长 warmup)
- 实验设计严谨 (严格 matched budget, REPAIR 统一, 推理拓扑相同)
- Sharpness 分析给出机制解释

**潜在弱点**:
- 理论只在 single-layer 级别, 跨层交互效应没建模 (虽然实验包含 full-network)
- Lipschitz 假设在 sharp minima 失效, paper 承认但没给修正
- LLaMA 实验只在 60M/130M, 没有真正大模型
- 对 attention layer 折叠未探索, ViT/LLaMA 都是 FFN-only

**未来方向猜测**: folding 可以自然扩展到 attention - 把相似的 Q/K/V projection 合并; 与 LoRA 组合 (low-rank adaptation on folded backbone); pruning+folding hybrid (先 fold 再 prune singleton clusters)。

---

## 9. 参考 web links

- 主 paper GitHub: https://github.com/osaukh/folding_as_projection
- LLaMA reproduction: https://github.com/nanguoyu/simple_model_folding_public
- 原 folding paper (Wang et al. 2025): https://openreview.net/forum?id=W2Wkp9MQsF
- Andriushchenko 2023 (sharpness benchmark): https://arxiv.org/abs/2302.07011
- Model Soups (Wortsman 2022): https://arxiv.org/abs/2203.05482
- SAM (Foret 2021): https://arxiv.org/abs/2010.01412
- Git Re-Basin (Ainsworth 2023): https://arxiv.org/abs/2209.04836
- SparseGPT (Frantar 2023): https://arxiv.org/abs/2301.00734
- Wanda (Sun 2024): https://arxiv.org/abs/2306.11695
- k-means = matrix factorization (Bauckhage 2015): https://arxiv.org/abs/1512.07548
- REPAIR (Jordan 2023): https://arxiv.org/abs/2211.08403
- LLaMA (Touvron 2023): https://arxiv.org/abs/2302.13971
- C4 (Raffel 2020): https://arxiv.org/abs/1910.10683
- AdaSAP sharpness-aware pruning (Bair 2024): https://arxiv.org/abs/2306.14306
- Pruning fundamental limit (Zhang 2025): https://arxiv.org/abs/2306.05857
- Augmentation & flat minima (Yoo 2025): https://arxiv.org/abs/2505.24592
- Vision Transformers without pretraining (Chen 2022): https://arxiv.org/abs/2106.01548
- SLTrain low-rank (Han 2024): https://arxiv.org/abs/2406.02214
- Maestro trainable decomposition (Horvath 2024): https://arxiv.org/abs/2308.14929
- CP-decomposition (Lebedev 2015): https://arxiv.org/abs/1412.6553

---

## 10. 一个简化版公式 cheatsheet

```
Pruning:    W_p = C_p W,  C_p = diag(I_k, 0)
            error = Σ pruned ||w_i||^2

Folding:    W_f = C_f W,  C_f = U_f (U_f^T U_f)^{-1} U_f^T
            U_f(i,j) = 1[i ∈ S_j]
            error = Σ_j Σ_{i∈S_j} ||w_i - μ_j||^2

Dominance:  error_pruning(k_p) ≥ error_folding'(k_p+1) ≥ error_folding*(k_p+1)
            where folding' uses 1 cluster for all pruned, rest singleton

Loss bound: |L(W) - L(W_compressed)| ≤ κ ||W - W_compressed||_F
            smaller Frobenius error ⟹ tighter loss bound (when κ moderate)
```

核心 take-away: **Pruning 是 coordinate-aligned subspace projection (axes 固定), folding 是 cluster-structured subspace projection (basis 可旋转)**。后者严格 richer, 给定 +1 rank (相对开销可忽略) 必胜前者。这个 geometric insight 解释了为什么 model folding 在 calibration-free compression 中持续胜出 magnitude pruning, 也预测了在 sharp minima (高 lr Adam) 时 folding 优势可能消失 - 因为 Lipschitz bound 不再控制 loss。

希望这个 walkthrough 帮你 build 起对 pruning vs folding 几何关系的 intuition!
