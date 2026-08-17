---
source_pdf: SiameseNorm Breaking the Barrier to Reconciling PrePost-Norm.pdf
paper_sha256: a4f371830d5bed3edc9b075d14e078d45d0c1a34e2de3fd99c60fa7c54c323e9
processed_at: '2026-08-12T05:54:05-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 SiameseNorm

好，刚才那版太 academic 了，我用大白话重新过一遍。

---

## 这篇 paper 在纠结什么

Transformer 里有个老问题：LayerNorm 到底放哪。

放 residual 之前叫 **Pre-Norm**，放之后叫 **Post-Norm**。

你可以这么想：residual connection 就像一条高速公路，信号一路往前冲。LayerNorm 就像收费站。

**Pre-Norm** 的做法是：收费站建在岔路口上，主路本身不收费。好处是信号一路畅通无阻，深网络也能训。坏处是主路上的车越积越多，到后面几层，新来的车相对于已经堆积如山的车流来说，简直就是一滴水进大海。所以深层的 contribution 被严重稀释，你 prune 掉几层发现性能几乎不掉 —— 这就是 "deeper layers ineffective" 现象。

**Post-Norm** 的做法是：收费站建在主路上，每过一层就强制把车流规模压回固定值。好处是每层 contribution 都有效，深层真正在干活。坏处是反传梯度时每层都要乘一次 LN 的 Jacobian，这个 Jacobian 的 spectral norm 对信号很敏感，乘多了就 explode 或者 vanish。大模型训着训着就炸了。

现在所有大模型都用 Pre-Norm，因为稳。但你其实一直在付一个隐性代价：你的模型深是深了，真正 "工作" 的深度没那么深。

---

## 为什么没人搞定这件事

有人试过把 Pre-Norm 和 Post-Norm 混着用。比如浅层用 Pre-Norm，深层用 Post-Norm。或者像 HybridNorm 那样 attention 之后做 LN，MLP 之前做 LN。

这些方法的问题是：它们都在同一条流上做手脚。你只要还在单流架构里，就有一个根本矛盾 ——

你想让梯度有一条干净的 identity path（这是 Pre-Norm 的核心），那主路上就不能 normalize。你想让主路 magnitude 受控（这是 Post-Norm 的核心），主路上就必须 normalize。这两个东西在几何上是 mutually exclusive 的，你怎么折中都只能拿一半。

paper 把这个叫 **structural incompatibility**。不是 hyperparameter 调不好，是结构上就没法两全。

---

## SiameseNorm 的核心思路

既然一条流上搞不定，那就开两条流。

一条流走 Pre-Norm 路线，专门保留 identity gradient，不管 magnitude 增长。叫 **Y-stream**。

一条流走 Post-Norm 路线，专门维持 bounded representation，不管梯度失稳。叫 **X-stream**。

两条流 **share 同一套参数**。也就是说，Attention 和 MLP 的 weight 是共享的，只是 hidden state 不同。每层的 residual block 同时处理两个 stream 融合后的输入。

这是关键 —— share 参数意味着这并不是在增加模型容量，而是强迫同一套参数同时受两种 normalization 范式的约束。两个 stream 起到 mutual regularization 的作用。

最后在输出端把两条流融合，得到最终 representation。

类比一下：你派两支队伍去探同一片地图。一支队伍走稳健路线，不冒进但也不图快，负责把路径打通（Y-stream）。另一支队伍走激进路线，每走一步就校准位置，保证探索精度（X-stream）。两支队伍 share 同一份地图笔记，互相参考对方的信息。

---

## 具体怎么实现的

每层做这些事：

1. 先把 Y-stream 做一次 LN，normalize 到 unit scale，准备喂给 residual block
2. residual block 的输入是 $X_i + Y_i'$，也就是 bounded stream 当前态加上 normalize 过的 unbounded stream，这是 fuse 的地方
3. residual block 输出 $O_i$ 同时更新两个 stream：
   - X-stream: $X_{i+1} = \text{LN}(X_i + O_i)$，走 Post-Norm 路径
   - Y-stream: $Y_{i+1} = Y_i + O_i$，走 Pre-Norm 路径

最后输出：$X_{\text{output}} = X_N + \text{LN}_{\text{final}}(Y_N)$

参数量几乎不增加，只多了几个 LN 操作。Overhead 可以忽略。

---

## 光有双流还不够，得加两个 trick

paper 很诚实地说，裸的双流架构还不 work，需要两个额外机制。

**Trick 1: Normalized Input**

虽然 $X_i$ 已经被 Post-Norm normalize 过，$Y_i'$ 也被 LN normalize 过，但两者相加之后 ($X_i + Y_i'$) 的分布可能漂移。喂给 $F_i$ 之前还得再过一次 LN。

你可以这么理解：两个 stream 的 scale 即使各自都 OK，加起来之后分布形状可能变形。Transformer block 对输入分布很敏感，所以加一道 "interface normalization" 保证 $F_i$ 看到的输入是稳定的。

ablation 显示这个 trick 贡献 0.08 PPL。差距不大，但 consistent。

**Trick 2: Depth-wise Scaling**

这个更 subtle，但也更关键。

问题：Y-stream 是 Pre-Norm 路线，它的 magnitude 会随深度近似 $\sqrt{l}$ 增长（这是 residual 累加的中心极限定理结果）。X-stream 是 Post-Norm 路线，magnitude 始终被压在 unit scale。

结果就是：深层两个 stream 的 scale ratio 越来越大。同一个 $O_i$ 要同时喂给两个 stream，对 Y-stream 来说可能太小（相对它的巨大 magnitude 是噪声），对 X-stream 来说可能太大（LN 要剧烈缩放，破坏稳定）。

fix: 在 X-stream 那侧把 $O_i$ 缩放 $1/\sqrt{l+1}$：
$$X_{i+1} = \text{LN}\left(X_i + \frac{1}{\sqrt{l+1}} O_i\right)$$

为什么是 $\sqrt{l+1}$？因为这正是 Pre-Norm stream magnitude 增长的速率。用 $1/\sqrt{l+1}$ 缩放相当于把 $O_i$ 校准到一个 "两个 stream 都能健康吸收" 的尺度。

这个 trick 贡献 0.25 PPL，而且它也是单流 HybridNorm 能不能收敛的关键 —— 没它直接 diverge，有它 PPL 10.65。

---

## 梯度上到底发生了什么

paper 给了个 block Jacobian 矩阵，两个 stream 的梯度动力学看对角块：

- Y-stream 的对角块是 $\mathbf{I} + \mathbf{J}_{F_j}\mathbf{J}_{\text{LN}_j^Y}$，跟 Pre-Norm 一模一样，那个 $\mathbf{I}$ 保证了梯度高速公路
- X-stream 的对角块是 $\mathbf{J}_{\text{LN}_j^X}(\mathbf{I} + \mathbf{J}_{F_j})$，跟 Post-Norm 一模一样，LN Jacobian 强制 bounded scale

非对角块是两个 stream 互相影响的 cross terms。这很关键，没有这两块两个 stream 就完全独立了，不会有 synergy。

直白说：梯度空间里同时跑两个 regime，Pre-Norm regime 负责把梯度稳定地传回早期层，Post-Norm regime 负责让 representation 维持合理 scale。两个 regime 通过 cross terms 互相校准。

还有一个 nice property：SiameseNorm 严格 generalize 已有范式。把 $\text{LN}_i^X$ 的参数 zero 掉就退化成 Pre-Norm，把 $\text{LN}_i^Y$ 的参数 zero 掉就退化成 Post-Norm。所以已有最好的方法都是 SiameseNorm 的特例，SiameseNorm 的性能下界就是它们里头最好的那个。

---

## 实验怎么做的

1.3B 参数 OLMo 架构，从 scratch 训练，FineWeb-Edu 数据集。三个学习率：4e-4, 1e-3, 2e-3。每个跑 100B tokens。最激进的 2e-3 还延伸到 350B tokens 测长期稳定性。

baseline 包括 Pre-Norm, Post-Norm, DeepNorm, ResiDual, HybridNorm, Hyper-Connections-2×DHC。很全。

总计算量超过 50,000 A100 小时，不是 toy experiment。

---

## 结果怎么样

Table 1 是核心数据。挑几个 striking 的点：

**低学习率 (4e-4)**：所有方法都收敛。HybridNorm 10.91 > Pre-Norm 11.21，说明 Post-Norm paradigm 上限确实更高。SiameseNorm 10.57，又比 HybridNorm 低 0.34。在 Post-Norm 已经能收敛的 regime 下还能进一步改善。

**中学习率 (1e-3)**：分水岭。Post-Norm 和 HybridNorm 都 diverge。Pre-Norm 稳定在 10.84。SiameseNorm 10.43，比 Pre-Norm 低 0.41。

**高学习率 (2e-3)**：DeepNorm 也 diverge。ResiDual 频繁 loss spike。Pre-Norm 10.89, Hyper-Connections 10.77。**SiameseNorm 10.48, Arithmetic accuracy 39.6%**。

这个 39.6% 要特别说一句。Pre-Norm 在 Arithmetic 上是 28.1%，随机 baseline 是 25%。其他所有方法都在 28-31% 区间。SiameseNorm 直接跳到 39.6%，相对 Pre-Norm 提升 40.9%。这不是 marginal improvement，是质的飞跃。

**长期 (350B tokens, 2e-3)**：SiameseNorm PPL 9.42, Arithmetic 43.4, Avg 58.70。Pre-Norm PPL 9.67, Arithmetic 36.2, Avg 57.17。优势持续放大，没有衰减。

---

## 为什么 Arithmetic 提升这么大

Arithmetic 是个 sequential reasoning 任务，真正需要每一层都干活，对 effective depth 最敏感。

Pre-Norm 的问题是深层 contribution 被稀释，深层其实没在做什么。SiameseNorm 通过 Post-Norm stream 恢复了深层的有效作用，所以 Arithmetic 提升特别大。

这跟 Gromov 那篇 "Unreasonable Ineffectiveness of Deeper Layers" 完全对得上 —— 他发现能 prune 掉 Pre-Norm 模型的很多深层，性能几乎不掉。SiameseNorm 从另一头证明：如果深层真的 work 了，任务性能能大幅提升。

---

## 两个 stream 各自在干什么

paper 做了个 Logit Lens 分析，看每个 stream 的 final hidden state 投影到 vocabulary space 后跟最终 prediction 对不对得上：

- HybridNorm stream (X-stream): 42.6% 匹配最终输出
- Pre-Norm stream (Y-stream): 16.2% 匹配

也就是说，**inference 时主要是 Post-Norm stream 在做决定**，Pre-Norm stream 在 final output 那里的权重只有 Post-Norm stream 的 40% 左右。

但 Pre-Norm stream 在训练时是不可或缺的，它提供稳定的梯度让深层能学。

我的直觉：Pre-Norm stream 是 "optimizer's anchor"，训练时稳住梯度。Post-Norm stream 是 "expressor"，inference 时真正 drive prediction。这种 training/inference 角色分离的设计很 elegant。

---

## 跟之前类似工作的区别

最像的两个工作是 ResiDual 和 Hyper-Connections。

**ResiDual** 也是双流，但它的 Pre-Norm stream 不参与 residual block 的输入，只是个 global shortcut 聚合每层输出到最终 output。从 Jacobian 看，它的 cross term 是 0，Pre-Norm stream 完全不接收后续 $F_j$ 的梯度信息。所以它的 Pre-Norm stream 梯度稳定但 uninformative。这解释了为什么 ResiDual 不会完全 diverge 但频繁 loss spike —— 它根本没真正 "用上" Pre-Norm stream 的 expressivity。

SiameseNorm 的 cross term 让两个 stream 互相 influence，这是本质区别。

**Hyper-Connections** 也是 dual-branch，但它本质上还在 Pre-Norm paradigm 里做扩展，通过 widening connections 和 learnable mixing 来扩展 Pre-Norm。它依赖 Pre-Norm-biased initialization 来稳定。SiameseNorm 强制两个 stream 初始等贡献，更严格地测试 intrinsic stability。

实验上 SiameseNorm 在所有 setting 下都优于 Hyper-Connections，Arithmetic 上差距特别大（39.6 vs 30.6）。

---

## 我的直觉

让我把核心直觉压缩成几条：

**1. Normalization 位置是表示空间的几何约束，不是 trick。** Pre-Norm 让 representation 自由增长，Post-Norm 每层把 representation 投影回 unit sphere。这两个几何操作在一条流上 mutually exclusive，双流把它 decouple 是 "对" 的方向。

**2. Share 参数是关键。** 如果两个 stream 独立参数就是 ensemble，增加 capacity。Share 参数意味着同一个 transformation 被两种 constraint 同时训练，起到 mutual regularization 作用。有点 contrastive learning 里同一 encoder 处理不同 augmentation 的味道。

**3. 训练和推理时两个 stream 角色分离。** Pre-Norm stream 是 training stabilizer，Post-Norm stream 是 inference expressor。这种设计哲学类似 BatchNorm 在 train/test 不同行为，但这里在 stream 层面做。

**4. Depth-wise Scaling 揭示了一个 deep 几何问题。** Pre-Norm stream 的 $\sqrt{l}$ 增长是 residual sum 的中心极限定理结果，不是 bug。但这个增长让两个 stream scale ratio 越来越大。$1/\sqrt{l+1}$ scaling 把 $O_i$ 重新校准到 "双方都能健康吸收" 的尺度。这让我想到 He init 里 $\sqrt{1/d}$ 的角色 —— 都是在处理 residual 累加的 scale 问题。

**5. Arithmetic 的巨大提升暗示 effective depth 真的被恢复了。** 28.1 → 39.6 不是 marginal improvement，是质的飞跃。如果这个 finding 在更大模型上 hold，可能改变我们对 LLM scaling 的理解 —— 也许不需要堆更深，而是让已有的深度真正 work。

---

## Limitations 和我想到的问题

paper 自己提了两个：
1. 某些 downstream task 提升没 PPL 提升那么显著
2. "Massive activations" 现象比 baseline 更明显

我的额外想法：

**Downstream vs PPL gap**: 我怀疑原因是 evaluation benchmark 都是 multiple-choice 浅层 reasoning，对 effective depth 敏感度不如 generation task。如果换成 free-form generation 或 long-context reasoning，提升可能更显著。Arithmetic 这个真需要 depth 的任务大幅提升支持这个猜测。

**Scale-up**: 1.3B 太小。7B, 70B, 405B 上还 hold 吗？Post-Norm 在大模型上更不稳定，SiameseNorm 能 scale 到那个 size 吗？Depth-wise scaling 的 $\sqrt{l+1}$ 在 100+ 层时还合理吗？

**Efficiency**: 双流增加了一些 LN 和 add 操作。长序列 + 大 batch 训练时这些小操作可能不那么小，需要更详细 profiling。

**跟其他架构的组合**: SiameseNorm 跟 MoE, linear attention, sparse attention 是否 orthogonal？我直觉是 yes，因为它只改 residual block 的 normalization 拓扑，不动 attention/FFN 内部。

**理论**: paper 的 gradient analysis 是 chain rule 推导，没有 spectral analysis。能否证明 Jacobian 的 spectral radius 在更宽条件下 bounded？能否解释为什么 $\sqrt{l+1}$ 是 optimal scaling？

---

## 相关工作的 links

整理一下 paper 提到的关键相关工作：

- SiameseNorm 代码: [github.com/Qwen-Applications/SiameseNorm](https://github.com/Qwen-Applications/SiameseNorm)
- Gromov et al. 2025, "Unreasonable Ineffectiveness of Deeper Layers": [arXiv:2403.17887](https://arxiv.org/abs/2403.17887)
- Sun et al. 2025, "Curse of Depth in LLMs": [arXiv:2502.05795](https://arxiv.org/abs/2502.05795)
- Sun et al. 2024, "Massive Activations in LLMs": [arXiv:2402.18562](https://arxiv.org/abs/2402.18562)
- Wang et al. 2024, DeepNorm: [arXiv:2203.00555](https://arxiv.org/abs/2203.00555)
- Xie et al. 2023, ResiDual: [arXiv:2304.14802](https://arxiv.org/abs/2304.14802)
- Zhu et al. 2025, Hyper-Connections: [OpenReview](https://openreview.net/forum?id=7qU0KQh4qT)
- Zhuo et al. 2025, HybridNorm: [arXiv:2503.04598](https://arxiv.org/abs/2503.04598)
- Li et al. 2025, Mix-LN: [OpenReview](https://openreview.net/forum?id=3p3c4nFlhJ)
- Kim et al. 2025, Peri-LN: [arXiv:2502.02732](https://arxiv.org/abs/2502.02732)
- Geva et al. 2021, Logit Lens: [ACL Anthology](https://aclanthology.org/2021.emnlp-main.446/)
- Groeneveld et al. 2024, OLMo: [ACL Anthology](https://aclanthology.org/2024.acl-long.725/)
- Penedo et al. 2024, FineWeb-Edu: [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)
- Xiong et al. 2020, "On Layer Normalization in Transformer": [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- Vaswani et al. 2017, Attention is All You Need: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## 一句话总结

我们过去十年把 Pre-Norm 当成 obvious best practice，其实是在 "稳定但上限低" 和 "上限高但不稳定" 之间被动选了前者。SiameseNorm 告诉你这个二选一是 single-stream 架构强加给你的，不是问题本身的性质。开两条流，让 Pre-Norm 专心稳梯度，Post-Norm 专心管 representation scale，share 参数让它们互相校准。Arithmetic 28.1 → 39.6 这个数字让我重新相信 effective depth 是个真实可恢复的 quantity。

我会很期待看到这个方法在更大 scale 上的验证，特别是跟 MoE、long-context、reasoning-oriented training 结合时。如果 SiameseNorm 真的恢复了 deep layers 的有效贡献，reasoning 任务上的提升可能比 arithmetic 还要显著。

---

# SiameseNorm: 用双流架构消解 Pre-Norm 与 Post-Norm 的结构性对立

Andrej, 这篇 paper 我读完之后,第一感觉是它把一个被社区默认为 "已经解决" 的问题重新打开了 —— Transformer 里 LayerNorm 到底放在哪。我们这些年在 LLM 里几乎无脑用 Pre-Norm, 觉得这就是 best practice, Post-Norm 是早期 Vaswani 那批人的 "naive" 选择。但这篇 paper 用一个非常干净的 argument 告诉你: 我们其实一直在付一个没意识到的代价, 而这个代价可以通过一个简单的双流结构拿回来。

我把核心思路、数学细节、实验数据、和我的直觉理解都拆开来讲。

---

## 1. 问题的真正根源: structural incompatibility, 不是 hyperparameter tuning 问题

先回到基本。一个 residual block 的两种 normalization 范式:

**Pre-Norm**:
$$X_{i+1} = X_i + F_i(\text{LN}_i(X_i))$$
这里 $X_i \in \mathbb{R}^d$ 是第 $i$ 层的 hidden state, $F_i$ 是 residual transformation (Attention 或 MLP), $\text{LN}_i$ 是 LayerNorm 或 RMSNorm。LN 只作用在 residual branch 的输入上, 主路径 $X_i$ 原样累加。

**Post-Norm**:
$$X_{i+1} = \text{LN}_i(X_i + F_i(X_i))$$
LN 作用在 residual addition 之后, 强制重新 normalize 整个 main path。

社区默认选 Pre-Norm 的理由是 backprop 时梯度有一条 clean identity path。从公式 (5) 看 Pre-Norm 的梯度:
$$\nabla_{\theta_i} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial X_N} \left[ \prod_{j=N-1}^{i+1} \left(\mathbf{I} + \mathbf{J}_{F_j} \mathbf{J}_{\text{LN}_j}\right) \right] \frac{\partial X_{i+1}}{\partial \theta_i}$$

这里 $\mathbf{I}$ 是单位矩阵 (来自 skip connection), $\mathbf{J}_{F_j} = \frac{\partial F_j}{\partial X_j}$ 是 residual block 的 Jacobian, $\mathbf{J}_{\text{LN}_j}$ 是 LN 的 Jacobian。乘积从第 $N-1$ 层反传到第 $i+1$ 层。关键: $\mathbf{I}$ 这一项永远存在, 即使 $\mathbf{J}_{F_j}\mathbf{J}_{\text{LN}_j}$ 很小, 梯度仍能顺着 identity path 流过去, 所以深网络也能训练。

Post-Norm 的梯度从公式 (7) 看:
$$\nabla_{\theta_i} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial X_N} \left[ \prod_{j=N-1}^{i+1} \mathbf{J}_{\text{LN}_j}(\mathbf{I} + \mathbf{J}_{F_j}) \right] \frac{\partial X_{i+1}}{\partial \theta_i}$$

注意 $\mathbf{J}_{\text{LN}_j}$ 被乘在最外层。LN 的 Jacobian 的 spectral norm 对信号很敏感, 多层连乘之后梯度要么 vanish 要么 explode。这就是 Post-Norm 训不动的根因。

但是, Pre-Norm 付出了一个隐性代价。看 paper 的 Figure 2a: 主路径 $X_i$ 的 $\ell_2$ norm 随深度 near-exponential 增长。原因是 $X_i$ 永远不被 normalize, 每层的 residual update $F_i(\text{LN}(X_i))$ 持续累加进去。结果是深层看到的输入 magnitude 越来越大, 但每层 residual block 自己的输出是被 LN 限制在固定 scale 的。这导致一个严重 imbalance: 深层要影响一个 huge magnitude 的主路径, 但只能贡献一个 fixed-scale 的 update, 相对贡献被严重稀释。

这就是 Gromov et al. (2025) "The Unreasonable Ineffectiveness of the Deeper Layers" ([paper](https://arxiv.org/abs/2403.17887)) 和 Sun et al. (2025) "The Curse of Depth in LLMs" ([paper](https://arxiv.org/abs/2502.05795)) 指出的现象: 你能 prune 掉 Pre-Norm Transformer 的很多深层, 性能几乎不掉。这暗示了 effective depth 严重不足。

Post-Norm 这边相反, 它每层都把信号强制 normalize 回 unit scale, 所以深层依然有效, 但代价是梯度爆炸。两个范式各自的 "病":

- **Dilution Problem (Pre-Norm)**: identity path 保留得好, 但主路径 magnitude 失控, 深层贡献被稀释
- **Distortion Problem (Post-Norm)**: magnitude 控制得好, 但反复的 scale contraction 破坏了梯度几何, 导致 compounding instability

paper 在 Section 2.3 给出的核心 claim: 这两个目标在 single-stream 设计下数学上不可兼得。要保持 strict identity gradient path 就不能在主路径上 normalize, 要 enforce bounded scale 就必须 normalize 主路径, 二者几何上冲突。

---

## 2. SiameseNorm 的核心 idea: decouple, 不要 compromise

paper 的解法非常优雅。既然两个目标在一条流上冲突, 那就开两条流, 一条专门保留 identity gradient, 一条专门维持 bounded representation, 然后 share 参数让它们融合。

设 $X_i$ 是 bounded stream (Post-Norm-like), $Y_i$ 是 unbounded stream (Pre-Norm-like)。初始:
$$X_0 = Y_0 = \text{input}$$

每层更新:
$$Y_i' = \text{LN}_i^Y(Y_i)$$
$$O_i = F_i(X_i + Y_i')$$
$$X_{i+1} = \text{LN}_i^X(X_i + O_i)$$
$$Y_{i+1} = Y_i + O_i$$

最终输出:
$$X_{\text{output}} = X_N + \text{LN}_{\text{final}}(Y_N)$$

让我逐项解释:

- $Y_i'$: 对 unbounded stream 先做 LN, 把它 normalize 到 unit scale, 准备喂给 residual block。这等价于 Pre-Norm 里 $\text{LN}(X_i)$ 的角色
- $O_i = F_i(X_i + Y_i')$: 关键一步。Residual block 的输入是 $X_i + Y_i'$, 即 bounded stream 的当前态 加上 normalized 的 unbounded stream。两个 stream 在这里 fuse
- $X_{i+1} = \text{LN}_i^X(X_i + O_i)$: X-stream 走 Post-Norm 路径, residual addition 后再 LN, 强制 bounded
- $Y_{i+1} = Y_i + O_i$: Y-stream 走 Pre-Norm 路径, residual addition 后不 LN, 保留 identity path

注意 $F_i$ 的参数 $\theta_i$ 是共享的 —— 两个 stream 用的是同一个 transformation, 只是 hidden state 的"视角"不同。这就是 "Siamese" (孪生) 的含义。参数量几乎不增加, 只多了几个 LN 操作。

---

## 3. Gradient analysis: 为什么这个架构 work

paper 在 Section 3 给了 block Jacobian transition matrix 的推导, 这是整个 paper 的理论核心。设 $S_i = [X_i, Y_i]^\top$, 上一层到下一层的 Jacobian:

$$\frac{\partial S_{j+1}}{\partial S_j} = \begin{bmatrix} \mathbf{J}_{\text{LN}_j^X}(\mathbf{I} + \mathbf{J}_{F_j}) & \mathbf{J}_{\text{LN}_j^X}\mathbf{J}_{F_j}\mathbf{J}_{\text{LN}_j^Y} \\ \mathbf{J}_{F_j} & \mathbf{I} + \mathbf{J}_{F_j}\mathbf{J}_{\text{LN}_j^Y} \end{bmatrix}$$

逐块解读:

**右下角 block** $\mathbf{I} + \mathbf{J}_{F_j}\mathbf{J}_{\text{LN}_j^Y}$: 这正是 Pre-Norm 的梯度动力学 (对照公式 5)。那个 $\mathbf{I}$ 项保证了 Y-stream 有一条 identity gradient highway, 不会被 vanish。所以 Y-stream 训练稳定。

**左上角 block** $\mathbf{J}_{\text{LN}_j^X}(\mathbf{I} + \mathbf{J}_{F_j})$: 这正是 Post-Norm 的梯度动力学 (对照公式 7)。X-stream 受 LN Jacobian 调制, 强制 bounded representation scale。

**非对角 block** $\mathbf{J}_{\text{LN}_j^X}\mathbf{J}_{F_j}\mathbf{J}_{\text{LN}_j^Y}$ 和 $\mathbf{J}_{F_j}$: 这两个 cross terms 是两 stream 之间的信息流。X-stream 通过 $F_j$ 影响 Y-stream (左下), Y-stream 通过 normalize 后再经 $F_j$ 影响 X-stream (右上)。这是 fuse 的关键 —— 没有这两块, 两个 stream 就完全独立了, 不会有 synergy。

我的直觉: 这个 Jacobian 的结构说明 SiameseNorm 在梯度空间里同时跑了两个 "regime", Pre-Norm regime 和 Post-Norm regime, 然后通过 cross terms 让它们互相校准。Y-stream 提供 "锚" —— 即使 X-stream 的梯度因为 LN 而失稳, 仍能通过 Y-stream 把有效梯度传回早期层。X-stream 提供 "shape" —— 强制 representation 维持一个合理的 scale, 不让 deep layer 的有效贡献被稀释。

paper 在 Section 3 末尾还指出一个 nice property: SiameseNorm 严格 generalize 了已有范式:
- 把 $\text{LN}_i^X$ 的参数 zero 掉 → X-stream 消失, 退化成 Pre-Norm
- 把 $\text{LN}_i^Y$ 的参数 zero 掉 → Y-stream 与 X-stream 脱钩, 退化成 Post-Norm
- 中间配置可以涵盖 Mix-LN ([Li et al., 2025](https://arxiv.org/abs/2409.09093)) 等 hybrid 设计

所以已有的最优方法 (无论 Pre-Norm, Post-Norm, 还是 hybrid) 都是 SiameseNorm 的特例, SiameseNorm 的性能下界就是这些特例里最好的那个。这是一个很强的理论保证。

---

## 4. 两个 "看似微小但必不可少" 的 mechanism

paper 在 Section 4.2 提到, 裸的 Siamese 架构还不足以 work, 需要两个额外的 mechanism。这两个 mechanism 我觉得是工程上很关键的发现, 单独拿出来讲。

### 4.1 Normalized Input

虽然两个 sub-stream 各自的输出都已经被 normalize 过了 ($X_i$ 来自 Post-Norm, $Y_i'$ 来自 $\text{LN}_i^Y$), 但 paper 发现 fuse 之后 ($X_i + Y_i'$) 还需要再过一次 LN 才能喂给 $F_i$。这个观察在 ablation Table 3 里 row 5 vs row 6: 去掉这个 input normalization, PPL 从 10.51 退化到 10.43。

我的解读: $X_i$ 和 $Y_i'$ 各自的 scale 不一定匹配, 简单相加可能产生分布漂移。Transformer block 对输入分布很敏感 (尤其 attention 的 softmax 和 MLP 的激活函数), 多加一次 LN 等于给 $F_i$ 一个稳定的 "interface"。这呼应了 Kim et al. (2025) Peri-LN ([paper](https://arxiv.org/abs/2502.02732)) 和 Sandwich-Norm 的思路: LN 在 residual block 入口的重要性被低估了。

### 4.2 Depth-wise Scaling

这个是更 subtle 的发现。问题: Pre-Norm stream 的 $Y_i$ magnitude 会随深度 $\sim\sqrt{l}$ 增长 (残差累加的统计规律), 而 Post-Norm stream 的 $X_i$ 始终是 unit scale。两者 scale mismatch 越来越严重。

考虑 $O_i = F_i(X_i + Y_i')$ 这个 update:
- 喂给 Y-stream: $Y_{i+1} = Y_i + O_i$。如果 $O_i$ 太小, 相对 $Y_i$ 的大 magnitude 就是噪声, Y-stream 学不到东西
- 喂给 X-stream: $X_{i+1} = \text{LN}_i^X(X_i + O_i)$。如果 $O_i$ 太大, LN 会被 forced to 剧烈缩放, 破坏稳定性

paper 的 fix: 在 HybridNorm stream (X-stream) 那侧, 把 residual update 缩放 $1/\sqrt{l+1}$:
$$X_{i+1} = \text{LN}_i^X\left(X_i + \frac{1}{\sqrt{l+1}} O_i\right)$$

其中 $l$ 是 layer index (从 0 开始计数, 所以 $l+1$ 避免除零)。

物理意义: $\sqrt{l+1}$ 大致就是 Pre-Norm stream magnitude 增长的速率 (中心极限定理下, $l$ 个独立同分布的残差和的 norm 是 $\sqrt{l}$ 量级)。用 $1/\sqrt{l+1}$ 缩放相当于把 $O_i$ 调节到一个 "双方都能健康吸收" 的尺度。这跟 DeepNorm ([Wang et al., 2024](https://arxiv.org/abs/2203.00555)) 用 depth-dependent residual scaling 的思路同源, 但目的不同: DeepNorm 是为了稳定 Post-Norm 单流, SiameseNorm 是为了平衡双流。

ablation 在 Table 3:
- HybridNorm + Depth-Scaling (单流): PPL 10.65 (原本 diverge)
- SiameseNorm without Depth-Scaling: PPL 10.68
- SiameseNorm with Depth-Scaling: PPL 10.43

Depth-wise Scaling 贡献了 0.25 的 PPL 改进, 同时也是 HybridNorm 单流能收敛的必要条件。

---

## 5. 实验设置: 严格控制在 1.3B OLMo

paper 用 OLMo ([Groeneveld et al., 2024](https://aclanthology.org/2024.acl-long.725/)) 1.3B 架构, 从 scratch 训练, 数据是 FineWeb-Edu ([Penedo et al., 2024](https://arxiv.org/abs/2406.17557))。配置细节在 Table 4: 16 层, hidden 2048, 16 heads, FFN intermediate 8192, SwiGLU, RoPE, RMSNorm ($\epsilon = 10^{-5}$), QK-Norm, Mitchell truncated normal init, AdamW ($\beta_1=0.9, \beta_2=0.95$), weight decay 0.1, cosine schedule with 2000 warmup steps, BF16 AMP, gradient clip 1.0。

学习率扫了三档: $4 \times 10^{-4}$, $1 \times 10^{-3}$, $2 \times 10^{-3}$, 每个 setting 训 100B tokens。最激进的 $2 \times 10^{-3}$ 又延伸到 350B tokens 测 long-term stability。总计算量超过 50,000 A100 hours。

baseline 很全: Pre-Norm, Post-Norm, DeepNorm, ResiDual, HybridNorm, Hyper-Connections-2×DHC。

特别注意一个细节: SiameseNorm 初始化时把所有 LN scale 和 mixing vector $\alpha$ 都设成 1.0, 强制两个 stream 初始等贡献。这跟 Hyper-Connections 用 Pre-Norm-biased init 不同, 是更严格的 "intrinsic stability" 测试。

---

## 6. 主结果: 高学习率下 SiameseNorm 全面胜出

Table 1 是核心数据。我挑几个最 striking 的观察:

**Setting A (LR=4e-4, conservative)**: 所有方法都收敛。HybridNorm PPL 10.91 > Pre-Norm 11.21, 验证了 Post-Norm 范式确实有更高上限。SiameseNorm 10.57, 比 HybridNorm 又低了 0.34。这说明 SiameseNorm 不只是 "stabilize Post-Norm", 它在 Post-Norm 已经能收敛的 regime 下还能进一步改善。

**Setting B (LR=1e-3)**: 这是分水岭。Post-Norm 和 HybridNorm 都 diverge。DeepNorm 勉强收敛到 11.47。Pre-Norm 稳定在 10.84。SiameseNorm 10.43, 比 Pre-Norm 低 0.41。Arithmetic accuracy 从 Pre-Norm 的 27.0 提升到 29.4。

**Setting C (LR=2e-3, aggressive)**: DeepNorm 也 diverge。ResiDual 频繁 loss spike, PPL 13.66。Pre-Norm 10.89, Hyper-Connections 10.77。SiameseNorm 10.48, **Arithmetic 39.6** (Pre-Norm 28.1, 相对提升 40.9%)。这个数字非常惊人。

**Setting D (LR=2e-3, 350B tokens)**: Pre-Norm PPL 9.67, Arithmetic 36.2, Avg 57.17。SiameseNorm PPL 9.42, Arithmetic 43.4, Avg 58.70。Long-term 稳定性验证通过, 优势持续放大。

**为什么 Arithmetic 提升这么大**: paper 给的解读是 arithmetic reasoning 强依赖 effective depth, Pre-Norm 因为 magnitude dilution 让深层失效, SiameseNorm 的 Post-Norm stream 恢复了深层作用。这个解读跟 Gromov 的 "deeper layers ineffective" 现象完全对得上。我的额外直觉: arithmetic 是个 "sequential" 任务, 需要 model 真正用上每一层的 transformation 而不是靠几个浅层 shortcut, 所以对 effective depth 最敏感。

**高学习率本身带来增益**: paper 在 4.3 节指出 "Higher Learning Rates Boost Model Performance"。Qiu et al. (2025) Gated Attention ([paper](https://arxiv.org/abs/2505.06708)) 也观察到类似现象 —— 在稳定的前提下, 大 LR 系统性提升性能。SiameseNorm 因为稳定性好, 能用上大 LR, 因此能 fully unlock 这个增益。Pre-Norm 也稳定但上限低, Post-Norm 上限高但不稳定, SiameseNorm 把两者优势都拿到了。

---

## 7. Ablation 拆解: 哪些组件真的 matter

Table 3 的 ablation (LR=1e-3):

| Normalized Input | Depth-Scaling | Topology | PPL |
|---|---|---|---|
| ✓ | ✗ | Original (HybridNorm) | diverge |
| ✓ | ✓ | Original (HybridNorm+DS) | 10.65 |
| ✓ | ✗ | ResiDual | 11.68* (spike) |
| ✓ | ✗ | Siamese | 10.68 |
| ✗ | ✓ | Siamese | 10.51 |
| ✓ | ✓ | Siamese | 10.43 |

几个结论:

1. **Siamese topology 本身就有用**: 即使没有 Depth-Scaling, SiameseNorm (10.68) 也击败了 diverge 的 HybridNorm 和 spiking 的 ResiDual。这验证了双流解耦本身的有效性。

2. **Depth-Scaling 对单流也有效**: 单流 HybridNorm 加上 Depth-Scaling 能从 diverge 变成 10.65, 说明 scale mismatch 是 Post-Norm 不稳定的另一个 root cause, 跟 gradient distortion 是两个独立的 pathology。

3. **Normalized Input 必要**: row 5 去掉它, PPL 10.51, 比 row 6 的 10.43 差 0.08。差距不算巨大但 consistent。我的解读是 fuse 后的 distribution drift 在长期训练中会累积。

4. **Sub-stream 的选择 crucial** (Table 2): 用 HybridNorm 作为 sub-stream 比用 vanilla Post-Norm 好 0.41-0.64 PPL。这说明 SiameseNorm 是一个 "框架", 它的上限取决于你塞进去的 sub-stream 有多强。如果未来有比 HybridNorm 更强的 Post-Norm variant, 套进 SiameseNorm 应该能进一步突破。

---

## 8. Stream 贡献分析: 谁在做决定

paper 在 Section 5.2 做了一个很有意思的 Logit Lens ([Geva et al., 2021](https://aclanthology.org/2021.emnlp-main.446/)) 分析, 看哪个 stream 实际驱动模型输出。

**LN scale parameters (Figure 6)**: 在大多数 layer, 两个 stream 的 LN scaling 都保持显著比例 (没有一边 collapse 到 0)。说明两个 stream 都在参与 feature extraction, 不是某个 stream 退化成 "dummy"。

**Final fusion layer 权重**: HybridNorm stream 的 weight 收敛到 1.05, Pre-Norm stream 收敛到 0.42。也就是说 Post-Norm stream 在输出端占主导 (大约 2.5:1 的比例)。

**Logit Lens 对比**: 把每个 stream 的 final hidden state 直接投影到 vocabulary space, 看哪个匹配最终 prediction:
- HybridNorm stream: 42.6% 匹配最终输出
- Pre-Norm stream: 16.2% 匹配
- 在 model 预测分歧的 case 中: 41.2% 对齐 HybridNorm, 14.3% 对齐 Pre-Norm

这个数据很有启发。Post-Norm stream 真正在做 "expressive decision", Pre-Norm stream 更多是在 optimization 过程中提供稳定的梯度锚, 而不是在 inference 时直接 drive prediction。这跟 paper 的 theoretical motivation 完全一致: Pre-Norm stream 的角色是 "stabilizer", Post-Norm stream 是 "expressor"。

我的额外思考: 这有点像 ensemble, 但跟 ensemble 不同的是两个 stream share 参数, 所以不是在增加 capacity, 而是在同一个 capacity 下提供两种 "inductive bias" 的融合。某种意义上, 这是一种 implicit regularization —— 让模型同时受两种 normalization 范式的约束。

---

## 9. 与 ResiDual 和 Hyper-Connections 的对比

paper 在 Appendix A.1 详细对比了最像的两个 prior work。

### ResiDual ([Xie et al., 2023](https://arxiv.org/abs/2304.14802))

ResiDual 看起来也是双流, 但关键区别: 它的 Pre-Norm stream (Y-stream) 不参与 residual block 的输入, 只是一个 global shortcut 聚合每层输出到最终 output。

ResiDual 的 Jacobian:
$$\frac{\partial S_{j+1}}{\partial S_j} = \begin{bmatrix} \mathbf{J}_{\text{LN}_j^X}(\mathbf{I} + \mathbf{J}_{F_j}) & \mathbf{0} \\ \mathbf{J}_{F_j} & \mathbf{I} \end{bmatrix}$$

右上角是 $\mathbf{0}$, 意味着 Pre-Norm stream 完全不接收后续 $F_j$ 的梯度信息。它的梯度稳定但 uninformative。这解释了为什么 ResiDual 不会完全 diverge 但频繁 loss spike —— 它没有真正 "用上" Pre-Norm stream 的 expressivity, 只是把它当 gradient highway。

SiameseNorm 的 cross term $\mathbf{J}_{\text{LN}_j^X}\mathbf{J}_{F_j}\mathbf{J}_{\text{LN}_j^Y}$ 让两个 stream 互相 influence, 这是本质区别。

### Hyper-Connections ([Zhu et al., 2025a](https://arxiv.org/abs/2502.19878))

Hyper-Connections 也是 dual-branch, 但它本质上更偏向 Pre-Norm paradigm, 通过 widening connections 和 learnable mixing 来扩展 Pre-Norm。它依赖 Pre-Norm-biased initialization 来稳定, 不像 SiameseNorm 强制等贡献初始。mHC ([Xie et al., 2025](https://arxiv.org/abs/2502.06788)) 是 Hyper-Connections 的改进版, 明确指出原版有训练不稳定问题。

paper 在 Table 1 显示 SiameseNorm 在所有 setting 下都优于 Hyper-Connections-2×DHC。Setting C (LR=2e-3): SiameseNorm 10.48 vs HC 10.77。Setting D: 9.42 vs 9.57 (HC 还有 loss spike, 标 *)。Arithmetic 上差距更大: 39.6 vs 30.6 (Setting C), 43.4 vs 33.6 (Setting D)。

我觉得 Hyper-Connections 的设计哲学是 "在 Pre-Norm 框架内做扩展", 而 SiameseNorm 是 "把 Pre-Norm 和 Post-Norm 真正 decouple 然后融合", 后者更触及问题本质。

---

## 10. 我的直觉总结: 为什么 SiameseNorm 是 "对" 的设计

让我把我的理解压缩成几个核心 intuition:

**Intuition 1: normalization 位置不是一个 "trick", 它是表示空间的几何约束**。Pre-Norm 等价于在 main path 上不做约束, 让 representation 自由增长; Post-Norm 等价于每层把 representation 投影回 unit sphere。这两个几何操作是 mutually exclusive 的, 在一条流上只能选一个。SiameseNorm 的 "对" 在于它承认了这个 mutual exclusivity, 然后用双流把它 decouple。

**Intuition 2: share parameters 是关键**。如果两个 stream 各自独立参数, 那只是 ensemble, 增加了 capacity。Share 参数意味着两个 stream 用同一个 transformation 处理不同的 "view" (bounded vs unbounded), 这强迫 transformation 同时被两种 constraint 训练, 起到 mutual regularization 的作用。这有点像 contrastive learning 里同一个 encoder处理不同 augmentation 的味道。

**Intuition 3: Pre-Norm stream 是 "optimizer's anchor", Post-Norm stream 是 "expressor"**。Logit Lens 数据表明, 训练时 Pre-Norm stream 提供稳定梯度让深层能学, 但 inference 时是 Post-Norm stream 在做决定。这种 "training/inference 角色分离" 的设计很 elegant, 类似于 BatchNorm 在 train/test 不同行为的设计哲学, 但这里是在 stream 层面做的。

**Intuition 4: Depth-wise Scaling 揭示了一个 deep 的 scale 几何问题**。Pre-Norm stream 的 $\sqrt{l}$ 增长不是 bug, 是 residual sum 的中心极限定理结果。但这个增长让两个 stream 之间的 scale ratio 越来越大。$1/\sqrt{l+1}$ 的 scaling 不是 ad-hoc, 它对应的是把 $O_i$ 重新校准到一个 "双方都能健康吸收" 的尺度。这让我想到 He init 里 $\sqrt{1/d}$ 的角色 —— 都是在处理 residual 累加的 scale 问题。

**Intuition 5: Arithmetic 上的巨大提升暗示 effective depth 真的被恢复了**。28.1 → 39.6 (40.9% 相对提升) 不是 marginal improvement, 是质的飞跃。这跟 "deeper layers ineffective" 现象的解决方案高度一致。如果这个 finding 在更大模型上 hold, 它可能改变我们对 LLM scaling 的理解 —— 也许我们不需要堆更深, 而是要让已有的深度真正 "work"。

---

## 11. Limitations 和我想到的 open questions

paper 自己提了两个 limitation:
1. 某些 downstream task 的提升没 PPL 提升那么显著
2. "Massive activations" ([Sun et al., 2024](https://arxiv.org/abs/2402.18562)) 现象比 baseline 更明显

我的额外思考:

**关于 massive activations**: 这其实是 Pre-Norm stream 的 unbounded growth 在 final fusion 处暴露出来。$Y_N$ 的 magnitude 很大, 即使过 LN_final, 它的 distribution shape 可能仍带 "outlier" 特征。这可能是 paper 提到的 limitation 2 的根因。一个可能的 fix 是在 Y-stream 周期性插入 "soft normalization" (不强制 unit scale 但抑制极端 outlier), 既保留 identity path 又控制极端值。

**关于 downstream vs PPL gap**: PPL 改善 0.4 但下游 task 平均提升 ~1 个百分点, 不成比例。我怀疑原因是 evaluation 用的 benchmark 都是 multiple-choice 形式的浅层 reasoning, 它们对 effective depth 的敏感度不如 generation task。如果换成 free-form generation 或 long-context reasoning, 提升可能更显著。Arithmetic 这个 "真需要 depth" 的任务大幅提升支持这个猜测。

**关于 scale-up**: 1.3B 是个很小的模型。SiameseNorm 在 7B, 70B, 405B 上还 hold 吗? Post-Norm 在大模型上更不稳定, SiameseNorm 是否能 scale 到那个 size? Depth-wise scaling 的 $\sqrt{l+1}$ 在 100+ 层时还合理吗? 这些都需要验证。

**关于 efficiency**: paper 说 overhead negligible, 但双流确实增加了一些 LN 和 add 操作。在长序列 + 大 batch 的训练里, memory bandwidth bound 的情况下, 这些 "小" 操作可能不那么小。需要更详细的 profiling。

**关于与其他 architecture 的组合**: SiameseNorm 跟 MoE, linear attention, sparse attention 这些其他架构改进是否 orthogonal? 我直觉是 yes, 因为它只改 residual block 的 normalization 拓扑, 不动 attention/FFN 内部。但需要实验确认。

**关于理论**: paper 的 gradient analysis 是 chain rule 推导, 没有 spectral analysis。能否证明 SiameseNorm 的 Jacobian 的 spectral radius 在更宽条件下 bounded? 能否解释为什么 $\sqrt{l+1}$ 是 optimal scaling 而不是其他形式? 这些是 future work 可以深挖的方向。

---

## 12. 相关工作的 web links

我把 paper 里提到的关键相关工作 link 整理一下, 方便你进一步阅读:

- 原论文 (这篇): [SiameseNorm GitHub](https://github.com/Qwen-Applications/SiameseNorm)
- Gromov et al. 2025, "Unreasonable Ineffectiveness of Deeper Layers": [arXiv:2403.17887](https://arxiv.org/abs/2403.17887)
- Sun et al. 2025, "Curse of Depth in LLMs": [arXiv:2502.05795](https://arxiv.org/abs/2502.05795)
- Sun et al. 2024, "Massive Activations in LLMs": [arXiv:2402.18562](https://arxiv.org/abs/2402.18562)
- Wang et al. 2024, DeepNorm: [arXiv:2203.00555](https://arxiv.org/abs/2203.00555)
- Xie et al. 2023, ResiDual: [arXiv:2304.14802](https://arxiv.org/abs/2304.14802)
- Zhu et al. 2025a, Hyper-Connections: [OpenReview](https://openreview.net/forum?id=7qU0KQh4qT)
- Zhuo et al. 2025, HybridNorm: [arXiv:2503.04598](https://arxiv.org/abs/2503.04598)
- Li et al. 2025, Mix-LN: [OpenReview](https://openreview.net/forum?id=3p3c4nFlhJ)
- Kim et al. 2025, Peri-LN: [arXiv:2502.02732](https://arxiv.org/abs/2502.02732)
- Xie et al. 2025, mHC (Manifold-constrained Hyper-Connections): [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)
- Geva et al. 2021, Logit Lens / FFN as key-value memories: [ACL Anthology](https://aclanthology.org/2021.emnlp-main.446/)
- Ba et al. 2016, Layer Normalization: [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Zhang & Sennrich 2019, RMSNorm: [NeurIPS 2019](https://papers.neurips.cc/paper/2019)
- Groeneveld et al. 2024, OLMo: [ACL Anthology](https://aclanthology.org/2024.acl-long.725/)
- Penedo et al. 2024, FineWeb-Edu: [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)
- Xiong et al. 2020, "On Layer Normalization in Transformer": [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- Wang et al. 2019, Pre-Norm for deep Transformers: [ACL Anthology](https://aclanthology.org/N19-1423/)
- Vaswani et al. 2017, Attention is All You Need: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## 13. 最后的直觉总结

如果让我用一段话总结这篇 paper 给我的最大启发:

我们过去十年把 Pre-Norm 当成 "obvious best practice", 但其实是在 "稳定但上限低" 和 "上限高但不稳定" 之间做了一个被动选择。SiameseNorm 告诉我们, 这个二选一是 single-stream 架构强加给我们的, 不是问题本身的性质。通过双流 + 共享参数, 我们可以让两个 normalization 范式 "各司其职": Pre-Norm stream 专门维护梯度高速公路, Post-Norm stream 专门维护 representation 的 bounded expressivity, cross terms 让它们互相校准。这不仅是 normalization 位置的改进, 是一个 "用结构 decoupling 解决 inductive bias 冲突" 的设计范例。Arithmetic task 上 28.1 → 39.6 这个数字, 让我重新相信 "effective depth" 是个真实可恢复的 quantity, 而不是必须接受的 pathology。

我想我会很期待看到这个方法在更大 scale 上的验证, 尤其是跟 MoE、long-context、reasoning-oriented training 结合时的表现。如果 SiameseNorm 真的恢复了 deep layers 的有效贡献, 那 reasoning 任务上的提升可能比 arithmetic 还要显著。
