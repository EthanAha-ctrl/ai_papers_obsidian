---
source_pdf: SiameseNorm.pdf
paper_sha256: a4f371830d5bed3edc9b075d14e078d45d0c1a34e2de3fd99c60fa7c54c323e9
processed_at: '2026-08-12T05:57:16-07:00'
target_folder: LLM-Training/nanogpt
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 SiameseNorm

Andrej，我换一种方式讲，抛开公式，先说故事。

---

## 一句话版本

Pre-Norm 和 Post-Norm 打了这么多年，一个稳但深层没用，一个强但容易炸。大家一直想把两个的优点合起来，但怎么合都是"顾此失彼"。这篇 paper 说：别在一条路上较劲了，开两条路，让它们各跑各的，最后再汇合——两条路共享同一套参数，所以不增加模型大小，但既稳又强。

---

## 用一个比喻说清楚

想象你在修一条从北京到深圳的高速公路，中间有 100 个收费站。

**Pre-Norm 的做法**：收费站只检查旁边那条辅路，主路一路畅通无阻。好处是车（gradient）跑得飞快，永远不堵。坏处是主路上车越积越多，到后面几十个收费站的时候，新加进来的车相对于主路上已有的车海，简直就是沧海一粟——后面的收费站形同虚设，设了等于没设。这就是 Pre-Norm 的 "Dilution Problem"——深层网络等于摆设，你 prune 掉一半 layer performance 几乎不掉（Gromov et al., 2025 的实验直接证明了这点）。

**Post-Norm 的做法**：每个收费站都把主路上的车全部清点一遍，强制重新排队，保持车流密度恒定。好处是每个收费站的发言权是平等的，深层也能发挥作用。坏处是每次清点都要花时间，而且 gradient 传回来的时候，每过一个收费站都要被"压缩"一次，传了 100 层之后 gradient 要么消失要么爆炸。这就是为什么 Post-Norm 在大模型上动不动就 diverge。

**之前的 hybrid 尝试**：有人想，那前 50 层用 Pre-Norm，后 50 层用 Post-Norm，行不行？或者奇数层用这个，偶数层用那个？结果发现，只要你还在同一条路上，这两个机制就是互斥的——你要么让车流自由累积（Pre-Norm），要么让车流被周期性 reset（Post-Norm），没有中间态。所有 hybrid 方案本质都在这两个极端之间摇摆，永远在 trade-off。

**SiameseNorm 的做法**：干脆修两条平行的路。一条路（Y-stream）不设收费站，车流自由累积——这是 Pre-Norm 的 spirit，保证 gradient 畅通。另一条路（X-stream）每个收费站都 reset——这是 Post-Norm 的 spirit，保证 representation bounded。关键 trick 是：两条路在每个收费站都汇聚一次，共用同一个"收费站设备"（$F_i$，也就是 attention 或 MLP block），所以你不需要修两套收费站，只是多修了一条路而已。最后到深圳的时候，两条路的车流再合在一起出城。

---

## 为什么这个 idea 能成立？核心 insight

这里最关键的洞察是 paper 第 2.3 节那个 "Structural Incompatibility" 论证，用大白话说就是：

**Pre-Norm 的本质是"别碰主路"——主路是 gradient 的高速通道，一碰就破坏了 identity path。**

**Post-Norm 的本质是"必须碰主路"——不碰的话 representation scale 就会爆炸，深层没法工作。**

这两个操作在单条路上是直接互斥的。你可以想象成一个人既要"完全不节食"又要"严格控制体重"——在同一个身体上这是矛盾的。但如果你 clone 出两个人，一个人随便吃（Y-stream），一个人严格 diet（X-stream），两个人共享一套消化系统（$F_i$），那就各取所需了。

这就是为什么叫 **Siamese**Norm——Siamese twin（暹罗双胞胎），共享身体但有两个头。

---

## 公式层面，用大白话解释 Eq (8) 那个 Jacobian

paper 里那个 block matrix 看着吓人，其实意思很直白。把两条路的状态拼在一起 $S_i = [X_i, Y_i]^{\top}$，然后看 gradient 怎么从第 $j+1$ 层传回第 $j$ 层。

那个 2×2 的矩阵：
- **左上角** $\mathbf{J}_{\mathrm{LN}_j^X}(\mathbf{I} + \mathbf{J}_{F_j})$：这条是 X-stream 自己内部的 gradient 传播——和标准 Post-Norm 一模一样，每层都被 $\mathbf{J}_{\mathrm{LN}}$ 压一下
- **右下角** $\mathbf{I} + \mathbf{J}_{F_j}\mathbf{J}_{\mathrm{LN}_j^Y}$：这条是 Y-stream 自己内部的 gradient 传播——和标准 Pre-Norm 一模一样，$\mathbf{I}$ 这个 identity 项就是那条畅通无阻的高速路
- **非对角线**：两条路之间的交叉影响——X-stream 的 gradient 有一部分会"漏"到 Y-stream，反之亦然

所以这个矩阵告诉你：**两条 paradigm 在各自的 diagonal block 里纯净运行，互不打架**。这就是 decoupling 的数学体现。

相比之下，ResiDual 的矩阵右下角是纯 $\mathbf{I}$，没有 $\mathbf{J}_{F_j}\mathbf{J}_{\mathrm{LN}_j^Y}$ 那一项——意思是 Y-stream 完全不参与每层的 transform，只是个 global accumulator。这就解释了为什么 ResiDual 经常 spike：梯度稳是稳了，但信息量不够，网络学不好。

参考 ResiDual: https://arxiv.org/abs/2304.14802

---

## 实验结果用大白话怎么说

Table 1 那堆数字，核心就三件事：

### 1. Learning Rate 是架构比较的照妖镜

在保守 LR（4e-4）下，HybridNorm 比 Pre-Norm 好——PPL 10.91 vs 11.21。这说明 Post-Norm family 的 representation capacity 确实更高。

但你把 LR 加到 1e-3，HybridNorm 直接 diverge，Pre-Norm 稳稳的。这就是为什么业界普遍用 Pre-Norm——在你能稳定训练的 LR regime 下，Pre-Norm 往往是赢家，因为 Post-Norm 压根跑不了 aggressive LR。

**但这不代表 Pre-Norm 本身更强，只代表 Pre-Norm 能承受更大的 LR，而大 LR 意味着更好的 performance**。这是个 confounding factor，之前很多 paper 没搞清楚。

### 2. SiameseNorm 两头通吃

在所有 LR 设置下，SiameseNorm 都最优：
- 4e-4：10.57（比 HybridNorm 的 10.91 还好）
- 1e-3：10.43（HybridNorm diverge 了，Pre-Norm 是 10.84）
- 2e-3：10.48（依然最优）

它既不怕大 LR（像 Pre-Norm），又在 representation 上发挥出 Post-Norm 的 capacity。这就是"breaking the barrier"的意思——以前你必须选一个，现在不用了。

### 3. Arithmetic 任务的 39.6 vs 28.1 是最硬核的证据

这是让我最 excited 的数字。基础算术任务（比如 3 位数加法）对 sequential reasoning 和 effective depth 非常敏感。Pre-Norm 只做到 28.1（几乎接近 random baseline 25%），SiameseNorm 做到 39.6，相对提升 40.9%。

这说明什么？**SiameseNorm 真的让深层"活"过来了**。之前 Pre-Norm 的深层是摆设，现在深层真的在做事了。这才是这篇 paper 最大的价值——不只是 PPL 降了零点几个点，而是网络的表达能力质变了。

参考 Gromov 的 pruning 实验: https://openreview.net/forum?id=O5dogEyYsO

---

## 两个工程 trick 的大白话

paper 4.2 节末尾提到两个 tricks，看着不起眼，其实缺一不可。

### Normalized Input

两条路各自都是 normalized 的，但它俩加在一起 $X_i + Y_i'$ 就不 normalized 了。直接喂给 $F_i$ 的话，$F_i$ 看到的 input distribution 不稳定。所以加了一个 LN 再喂进去。

这就像两个人各自穿好衣服出门，但搂在一起走的时候，还得再整理一下仪容。

### Depth-wise Scaling

深层的时候，Y-stream 的 magnitude 越来越大（因为它不 normalize），X-stream 的 magnitude 恒定。共享的 $O_i$（residual block 输出）面对一个问题：量级该匹配谁？

匹配 Y-stream 吧，X-stream 承受不住会炸；匹配 X-stream 吧，Y-stream 觉得 update 太小没感觉。

解决方法：在送入 X-stream 之前，把 $O_i$ 除以 $\sqrt{l+1}$（$l$ 是 layer index）。深层除以更大的数，让 X-stream 在深层接收更 gentle 的 update。这个 trick 和 DeepNorm 的 depth-dependent scaling 思路相通，但用法不同。

参考 DeepNorm: https://arxiv.org/abs/2203.00555

---

## 哪条路在"做主"？Logit Lens 的发现

Section 5.2 做了一个很 clever 的分析。用 Logit Lens 把每条路最后层的 hidden state 直接投影到 vocabulary space，看哪条路的预测和最终 output 对齐。

结果：
- X-stream（Post-Norm variant）和最终 output 匹配 42.6%
- Y-stream（Pre-Norm）只匹配 16.2%
- 分歧时 X-stream 也以 41.2% vs 14.3% 主导

**大白话**：X-stream 是"发言人"，负责产出最终预测。Y-stream 是"幕后顾问"，它不直接说话，但它的存在让 X-stream 能被有效训练——它维护着 gradient highway，让 optimization 顺畅。

这有点像公司里 CEO 和 COO 的关系——CEO（X-stream）对外发布决策，COO（Y-stream）保证内部运营不崩。两个人共享同一套资源（$F_i$ 的参数），缺一不可。

参考 Logit Lens: https://arxiv.org/abs/2103.01657

---

## 我觉得最 elegant 的地方

这篇 paper 最让我欣赏的是它的设计哲学：**遇到两难，不要在单点上 compromise，而是 structural decouple**。

这和 ResNet 当年的精神一脉相承——ResNet 解决 degradation problem 不是靠更好的 initialization 或 activation function，而是用一个 structural change（skip connection）直接绕过问题。SiameseNorm 解决 Pre/Post-Norm 矛盾也不是靠更 clever 的 hybrid schedule，而是用一个 structural change（双流共享参数）直接化解矛盾。

这种"用结构换 optimization landscape"的思路，在 neural network design 里是最有生命力的。trick 会过时，但 structural insight 会持续启发后续工作。

参考 ResNet: https://arxiv.org/abs/1512.03385

---

## 最大的悬念

paper 只做了 1.3B。在这个规模上 SiameseNorm 完胜，但：

**7B、70B 会怎样？**

Post-Norm 的 instability 是随 scale 加剧的——1.3B 能稳定的 Post-Norm variant，到 7B 可能就炸了。SiameseNorm 的 Y-stream identity path 在更大规模上能否还能 hold 住 aggressive LR？X-stream 的 Post-Norm dynamics 在更深网络（比如 80 层）里会不会重新出现 gradient 问题？

这些都是 open questions。paper 作者在 Limitations 里也承认 massive activations 问题更严重了，这在 scale up 的时候可能变成 real bottleneck。

但不管怎样，这篇 paper 的核心 idea——**双流共享参数来 decouple 矛盾机制**——是一个可以被反复借鉴的设计 pattern。即使 SiameseNorm 本身在更大规模需要 further modification，这个 pattern 的影响力会持续。

代码在这里: https://github.com/Qwen-Applications/SiameseNorm

---

## 如果只记一句话

Pre-Norm 和 Post-Norm 的矛盾在单流里无解，SiameseNorm 用双流共享参数把它们拆开各跑各的，既稳又强，深层真的活了。

---

# SiameseNorm: 用双流架构化解 Pre-Norm 与 Post-Norm 的根本矛盾

Andrej 你好，这篇 paper 我读了好几遍，核心 idea 其实非常 elegant，但里面藏了很多 subtle 的设计决策，值得拆开来仔细讲。我会按你的要求，把数学、架构、实验和直觉都串起来，尽可能多覆盖相关联想。

---

## 1. 核心矛盾的根因：单流架构里的代数不可能三角

这篇 paper 最 sharp 的贡献是**第 2.3 节的 "Structural Incompatibility" 论证**——它把一个工程上大家都隐约感觉到的问题，提升到了一个 almost 数学层面的论断。

考虑一个单 residual stream $X_i$，你想同时满足两个性质：

**(A) Clean identity gradient path**：反向传播时，梯度可以通过 $\mathbf{I}$ 项无衰减地穿过任意多层。这是 Pre-Norm 的本质——它在 Eq (1) 里把 LN 放在 residual branch 内部：

$$X_{i+1} = X_i + F_i(\mathrm{LN}_i(X_i))$$

它的 forward Jacobian 是 $\mathbf{I} + \mathbf{J}_{F_j}\mathbf{J}_{\mathrm{LN}_j}$，$\mathbf{I}$ 这一项保证了梯度 highway（Eq 5）。

**(B) Bounded representation scale**：每层之后 hidden state magnitude 被严格 reset 到固定尺度（比如 unit variance）。这是 Post-Norm 的本质（Eq 2）：

$$X_{i+1} = \mathrm{LN}_i(X_i + F_i(X_i))$$

其 Jacobian 是 $\mathbf{J}_{\mathrm{LN}_j}(\mathbf{I} + \mathbf{J}_{F_j})$，其中 $\mathbf{J}_{\mathrm{LN}_j}$ 是一个 contractive operator（spectral norm 通常 < 1，且对输入信号敏感）。

**矛盾点**：在单流里，$X_i$ 既要作为 identity path 无衰减地累积，又要被 LN 周期性 reset。这两个操作是直接互斥的——"无衰减累积"和"周期性收缩"在代数上无法共存。任何 hybrid 方案（Mix-LN、HybridNorm、Sandwich-Norm 等）本质上都在这两个极端之间 oscillate，不可能同时拿到两个性质的 full benefit。

这一论证让我想到 **Highway Networks (Srivastava et al., 2015)** 的 gate 设计——它用 $g(x) \cdot H(x) + (1-g(x)) \cdot x$ 来让网络 learn 多少信息走 transform 多少走 identity，但本质上还是单流，gate 和 LN 的作用有功能重叠。SiameseNorm 的选择是直接拆成两流，让两个机制在各自的 stream 里纯净运行。

参考：
- Highway Networks: https://arxiv.org/abs/1505.00387
- Sandwich-Norm (CogView): https://arxiv.org/abs/2105.13290
- Mix-LN: https://openreview.net/forum?id=ujGhM4H3Iz

---

## 2. Pre-Norm 的 "Dilution Problem"：为什么深层越来越没用

这个 paper 引用了一个我觉得非常关键的 recent finding：**Gromov et al. (2025) 的 "unreasonable ineffectiveness of deeper layers"**——对深层 Pre-Norm Transformer 做 layer pruning，performance 几乎不掉。这意味着 Pre-Norm 网络的 effective depth 远小于 nominal depth。

paper 第 2.2 节把这个现象用 **magnitude scaling imbalance** 来解释，公式上：

- residual branch 的输入被 LN 限制在 fixed scale $\approx \sqrt{d}$
- 但 main path $X_i$ 的 magnitude 在 depth 上不断累积，到深层 $|X_i|_2$ 变得很大

要让第 $i$ 层（深层）的输出 $O_i = F_i(\mathrm{LN}(X_i))$ 对 main path $X_i$ 产生 meaningful relative contribution，网络必须 learn 一个很大的 scaling factor 把 $O_i$ 放大到匹配 $|X_i|$ 的量级。这在 optimization 上非常困难。

**Figure 2a 的实验**非常有说服力：他们试了一个叫 PreNorm-EmbedNorm 的变体——在 embedding 后加一个 parameter-free RMSNorm，把 $X_0$ 的 magnitude 从 ~2 放大到 $\sqrt{d} \approx 45$，目的是强行压制早期层的 magnitude 增长。结果 magnitude profile 确实变 flat 了，但 PPL 反而 degrade 了 0.4。这说明"在 Pre-Norm 框架内调节 magnitude"是个死胡同——dilution 不是 implementation bug，而是架构的内在 property。

这让我联想到 **Curse of Depth (Sun et al., 2025)** 和 **Peri-LN (Kim et al., 2025)** 的相关工作，他们也都观察到了 Pre-Norm 的 magnitude 爆炸问题，但解决方案都是单流内的修补，无法根本性解决。

参考：
- Unreasonable ineffectiveness of deeper layers: https://openreview.net/forum?id=O5dogEyYsO
- Curse of Depth: https://arxiv.org/abs/2502.05795
- Peri-LN: https://arxiv.org/abs/2502.02732

---

## 3. SiameseNorm 的架构：双流 + 共享参数

### 3.1 状态更新公式

定义两个 stream：
- $X_i$：bounded stream（Post-Norm-like），每层做 LN
- $Y_i$：unbounded stream（Pre-Norm-like），无 LN 累积

初始化 $X_0 = Y_0 = \text{input embedding}$。

每层 $i$ 的更新（paper 里这部分公式是核心）：

$$
\begin{aligned}
Y_i' &= \mathrm{LN}_i^Y(Y_i) \\
O_i &= F_i(X_i + Y_i') \\
X_{i+1} &= \mathrm{LN}_i^X(X_i + O_i) \\
Y_{i+1} &= Y_i + O_i
\end{aligned}
$$

**关键变量解释**：
- $Y_i'$：unbounded stream 经过 normalize 后的版本，作为 $F_i$ 的输入之一
- $O_i$：residual block 的输出，被两个 stream 共享
- $\mathrm{LN}_i^X$：X-stream 的层归一化，强制 bounded
- $\mathrm{LN}_i^Y$：Y-stream 的层归一化，只为 $F_i$ 提供 normalized input，但 Y 本身不被 normalize

最终输出在第 $N$ 层 fuse：

$$X_{\text{output}} = X_N + \mathrm{LN}_{\text{final}}(Y_N)$$

**关键设计点**：$F_i$ 只有一份参数，作用在 $X_i + Y_i'$ 的 fused representation 上。这意味着 computational overhead 仅来自额外的 LN，paper 里说"negligible"——确实，LN 的 FLOPs 相对 attention 和 MLP 可忽略。

### 3.2 它是 Pre/Post/Hybrid 的 strict generalization

这点很重要，paper 在 Section 3 强调了：
- 若把 $\mathrm{LN}_i^X$ 的 scale parameter 置 0 → X-stream 不再贡献，退化为 Pre-Norm
- 若把 $\mathrm{LN}_i^Y$ 置 0 → Y-stream 不贡献 normalized input，退化为 Post-Norm
- 中间状态涵盖 Mix-LN、HybridNorm 等 hybrid 方案

所以 SiameseNorm 的 performance lower bound 就是这些 special cases 的 best。这是一个很强的理论性 argument——它在 optimization landscape 上是一个 superset。

---

## 4. 梯度动力学分析：Eq (8) 是整篇 paper 的灵魂

这是我最喜欢的部分。定义 $S_i = [X_i, Y_i]^{\top}$（concatenated state）。对参数 $\theta_i$ 的梯度是：

$$
\nabla_{\theta_i}\mathcal{L} = \frac{\partial\mathcal{L}}{\partial S_N}\left(\prod_{j=N-1}^{i+1}\frac{\partial S_{j+1}}{\partial S_j}\right)\left[\mathbf{J}_{\mathrm{LN}_i^X}\right]\frac{\partial O_i}{\partial \theta_i}
$$

中间的 block Jacobian transition matrix（Eq 8）：

$$
\frac{\partial S_{j+1}}{\partial S_j} = \begin{bmatrix}
\mathbf{J}_{\mathrm{LN}_j^X}(\mathbf{I} + \mathbf{J}_{F_j}) & \mathbf{J}_{\mathrm{LN}_j^X}\mathbf{J}_{F_j}\mathbf{J}_{\mathrm{LN}_j^Y} \\
\mathbf{J}_{F_j} & \mathbf{I} + \mathbf{J}_{F_j}\mathbf{J}_{\mathrm{LN}_j^Y}
\end{bmatrix}
$$

**变量含义**：
- 对角块 (1,1)：$\mathbf{J}_{\mathrm{LN}_j^X}(\mathbf{I}+\mathbf{J}_{F_j})$ —— 正好是 Post-Norm 的 Jacobian（对应 Eq 7）
- 对角块 (2,2)：$\mathbf{I} + \mathbf{J}_{F_j}\mathbf{J}_{\mathrm{LN}_j^Y}$ —— 正好是 Pre-Norm 的 Jacobian（对应 Eq 5），其中 $\mathbf{I}$ 提供 identity highway
- 非对角块：stream 之间的交叉耦合

**直觉**：这个矩阵告诉你两件事在并行运行：
1. **Top-left block**：X-stream 把 Post-Norm 的 bounded representation dynamics 完整保留了——每层被 $\mathbf{J}_{\mathrm{LN}}$ modulate，spectral norm < 1，但 representation scale 严格 bounded
2. **Bottom-right block**：Y-stream 把 Pre-Norm 的 identity gradient highway 完整保留了——$\mathbf{I}$ 项直接出现在对角线上，gradient 无衰减

这就是 paper 反复强调的 "decoupling"：两个 paradigm 在各自的 stream 里纯净运行，互不打断。其他 multi-path 设计（比如 ResiDual）的关键缺陷就在这里——它的 Y-stream（Pre-Norm stream）没有连接回 $F_i$ 的输入，所以对角块 (2,2) 退化为纯 $\mathbf{I}$（没有 $\mathbf{J}_{F_j}\mathbf{J}_{\mathrm{LN}_j^Y}$ 这一项），意味着 Y-stream 只是个 global accumulator，不参与迭代 transform。这就是为什么 ResiDual 在 Figure 4 里频繁 spike 但不 diverge——梯度 stable 但 uninformative。

参考 ResiDual: https://arxiv.org/abs/2304.14802

### 4.1 数值上验证 gradient stability

Figure 5 是关键证据。在 $\eta = 10^{-3}$ 下：
- HybridNorm（Post-Norm 变体）：gradient norm 反复 spike 到 >100，最终 diverge
- Pre-Norm：稳定在 <0.5
- SiameseNorm：和 Pre-Norm 几乎重叠的 stable trajectory，<0.5

也就是说，SiameseNorm 在 high LR 下完全继承了 Pre-Norm 的 stability，但表达能力又解锁了 Post-Norm 的容量。

---

## 5. 工程上的两个关键 trick：Normalized Input 和 Depth-wise Scaling

paper 第 4.2 节末尾提到两个 mechanisms，ablation study（Table 3）证明它们不可或缺。这部分 paper 写得有点低调，但其实是工程实现成败的关键。

### 5.1 Normalized Input

虽然 $X_i$ 和 $Y_i'$ 各自都是 normalized 的，但它们的 sum $X_i + Y_i'$ **不是** normalized 的。在送进 $F_i$ 之前必须再加一个 LN：

$$O_i = F_i(\mathrm{LN}(X_i + Y_i'))$$

这一点 paper 里写得很隐晦，但 ablation Table 3 的 Row 5 vs Row 6 显示，去掉这个 LN 会让 PPL 从 10.43 退化到 10.51（其实差距没那么大，但确实 significant）。这呼应了 Xiong et al. (2020) 的经典分析——LN before attention 是防止 attention output magnitude explosion 的关键。

参考 Xiong et al.: https://arxiv.org/abs/2002.04745

### 5.2 Depth-wise Scaling

随着深度增加，$Y_i$（unbounded）的 magnitude 趋向增长，而 $X_i$（bounded）保持 unit scale。共享的 residual update $O_i$ 面临 dilemma：
- 如果 $O_i$ 量级适配 $Y_i$，那对 $X_i$ 来说太大了 → X-stream 不稳定
- 如果 $O_i$ 量级适配 $X_i$，那对 $Y_i$ 来说太小了 → Y-stream 更新不动

解决方案：在送入 X-stream 之前对 $O_i$ 做 depth-dependent scaling：

$$X_{i+1} = \mathrm{LN}_i^X\left(X_i + \frac{1}{\sqrt{l+1}} O_i\right)$$

其中 $l$ 是 layer index。这个 $1/\sqrt{l+1}$ 让 X-stream 在深层接收较小的 update，类似 DeepNorm 的 depth-dependent scaling 思路，但作用方向不同——DeepNorm 是在 Post-Norm 里让 deep layer 的 residual contribution 衰减以稳定训练，SiameseNorm 是在双流之间 rebalance。

参考 DeepNorm: https://arxiv.org/abs/2203.00555

Table 3 的 Row 1 vs Row 2 显示，depth-wise scaling 让 HybridNorm 单流从 diverge 变成 10.65 PPL；Row 4 vs Row 6 显示加上 scaling 后 SiameseNorm 从 10.68 提升到 10.43，提升 0.25 PPL。

---

## 6. 实验结果的核心 takeaways

### 6.1 Table 1 的 Setting A vs B vs C vs D

我重新整理一下关键数字（用 PPL 衡量，越低越好）：

| Setting | LR | Pre-Norm | HybridNorm | SiameseNorm |
|---------|-----|----------|------------|-------------|
| A (100B) | 4e-4 | 11.21 | 10.91 | 10.57 |
| B (100B) | 1e-3 | 10.84 | diverge | 10.43 |
| C (100B) | 2e-3 | 10.89 | diverge | 10.48 |
| D (350B) | 2e-3 | 9.67 | – | 9.42 |

**三个观察**：

1. **LR sensitivity 决定 architecture 比较**：在 conservative LR（4e-4）下，HybridNorm 比 Pre-Norm 强（10.91 vs 11.21），印证 Post-Norm family 的 capacity 上限更高。但在 aggressive LR 下，HybridNorm 直接 diverge，Pre-Norm 反而胜出。这说明历史文献里 "Pre-Norm 优于 Post-Norm" 的结论很多是 LR regime 的 artifact——两者 optimal LR 不同，比较时如果不做 grid search 就不公平，但 grid search 计算上 prohibitive。

2. **Higher LR 普遍提升 performance**：所有能 converge 的方法，从 4e-4 → 2e-3 都获得显著 PPL 下降。这是 Qiu et al. (2025) 在 Gated Attention 里也观察到的现象。所以 architecture 的真正考验是"能不能承受 aggressive LR"——SiameseNorm 在 2e-3 下还能稳定 improve，这是它的核心竞争力。

3. **Arithmetic task 的 39.6 vs 28.1**：在 Setting C，SiameseNorm 在基础算术任务上达到 39.6 accuracy，相对 Pre-Norm 的 28.1 提升 40.9%。这个数字非常 striking——arithmetic 是 sequential reasoning 的代表，对 effective depth 高度敏感。Pruning 实验显示 Pre-Norm 的深层没用，而 SiameseNorm 通过恢复 effective depth 让深层真正参与 reasoning。这是 paper 标题"breaking the barrier"最直接的证据。

参考 Gated Attention: https://arxiv.org/abs/2505.06708

### 6.2 Setting D 的 long-term stability

350B tokens 训练（vs 100B）的结果：
- Pre-Norm：9.67 PPL
- SiameseNorm：9.42 PPL
- Arithmetic accuracy：SiameseNorm 达到 43.4 vs Pre-Norm 36.2

这说明 long-term 训练下 SiameseNorm 优势继续放大，没有 late-stage instability。这个 long-horizon stability 对大规模预训练至关重要——很多看起来 early training 稳定的方法在 trillion-token 规模会突然崩。

---

## 7. 与其他 multi-path 设计的对比（Appendix A.1）

### 7.1 vs ResiDual

ResiDual 的 Jacobian：

$$
\frac{\partial S_{j+1}}{\partial S_j} = \begin{bmatrix}
\mathbf{J}_{\mathrm{LN}_j^X}(\mathbf{I} + \mathbf{J}_{F_j}) & \mathbf{0} \\
\mathbf{J}_{F_j} & \mathbf{I}
\end{bmatrix}
$$

注意 bottom-right 是纯 $\mathbf{I}$，没有 $\mathbf{J}_{F_j}\mathbf{J}_{\mathrm{LN}_j^Y}$ 项。这意味着 Y-stream 完全不接收来自后续 residual transform 的梯度——它只是个 "global shortcut"。这导致梯度 stable 但 uninformative，对应 Figure 4 里 ResiDual 频繁 spike 但不 diverge 的现象。

SiameseNorm 的关键差异在于 $F_i$ 作用在 $X_i + Y_i'$ 的融合上，让两个 stream 都参与每层 transform，从而 bottom-right block 有完整的 Pre-Norm 形式。

### 7.2 vs Hyper-Connections

Hyper-Connections (Zhu et al., 2025a) 是另一个重要的 multi-path 设计，它用 learnable mixing matrix 在不同 stream 之间做信息交换。paper 里观察到 Hyper-Connections 也有 training instability（在 aggressive LR 下），mHC (Xie et al., 2025) 是其改进版，更偏向 Pre-Norm。

SiameseNorm 的优势在于：
- 更简单——没有 $H_{res}$ 这种 cross-stream mixing，只通过 shared $F_i$ 自然耦合
- 更稳定——Table 1 显示 SiameseNorm 在所有 LR 设置下都优于 Hyper-Connections-2×DHC

paper Section A.1 也承认 Hyper-Connections 框架可以和 SiameseNorm 结合，未来可能找到 unified multi-path perspective。

参考 Hyper-Connections: https://openreview.net/forum?id=Bpf7ijuFQB
参考 mHC: https://arxiv.org/abs/2512.24880

---

## 8. Analysis 部分：哪个 stream 主导？

Section 5.2 的 Logit Lens 分析很有意思。

**Input contribution（Figure 6）**：在大部分 layer，X-stream（HybridNorm）和 Y-stream（Pre-Norm）对 $F_i$ 输入的贡献都有显著比例，不是某一个完全 dominate。这说明双流不是 redundant 的，它们提供互补的 representation。

**Output contribution**：在 final fusion layer：
- X-stream 的 learned LN weight: 1.05
- Y-stream 的 learned LN weight: 0.42

Logit Lens 投影到 vocabulary space 后：
- X-stream 与 final output 匹配 42.6% 的时间
- Y-stream 仅 16.2%
- 在分歧预测中，X-stream 也以 41.2% vs 14.3% 主导

**直觉解释**：X-stream（Post-Norm variant）承担 representation 的"主输出"角色，因为它经过严格 normalization，hidden state 直接映射到 vocabulary space 时分布稳定。Y-stream（Pre-Norm）更多是优化 stabilizer——它保证 gradient highway 通畅，让 X-stream 能够被有效训练，但它的 raw hidden state 由于 magnitude 爆炸不适合直接 decode。

这有点像 **Mixture of Experts 的 routing**——一个 expert 负责 generation，另一个 expert 负责 optimization stability。但这里两个 stream 完全共享参数，不增加模型容量。

参考 Logit Lens: https://arxiv.org/abs/2103.01657

---

## 9. Limitations 与 future work

paper Section 7 诚实承认两个 limitation：

1. **下游任务提升不如 PPL 提升显著**：PPL 从 10.84 → 10.43 是 4% 相对提升，但下游 benchmark（ARC、HellaSwag 等）提升通常只有 1-2 个点。这暗示 PPL 提升主要来自 representation quality，不一定 transfer 到 reasoning task（除 arithmetic 外）。

2. **Massive activations 问题更严重**：Sun et al. (2024) 观察到 LLM 里少数维度有 extreme activation，SiameseNorm 里这个现象更严重。这可能因为 Y-stream 不做 normalize，累积导致某些维度 magnitude 极大。这可能是未来优化的空间——比如在 Y-stream 里加一个 lightweight regulation（不是 full LN，而是某种 sparsity-aware scaling）。

参考 Massive Activations: https://arxiv.org/abs/2402.17762

---

## 10. 我对这篇 paper 的整体直觉与联想

### 10.1 类比：SiameseNorm 像 GRU 里的 update gate

Pre-Norm 是简单的 residual accumulation，类似 vanilla RNN——信息无衰减累积但容易爆。Post-Norm 是严格的 reset，类似 RNN 每步都 normalize 但梯度难传。SiameseNorm 让两个机制并行——这有点像 GRU 的 update gate $z_t$：

$$h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

但 SiameseNorm 不是 soft gate，而是硬分流——Y-stream 完全累积（$z_t = 0$），X-stream 完全 normalize（$z_t = 1$）。这种"硬分流"在数学上更纯净，避免了 gate 的 optimization difficulty。

### 10.2 联想：和 Mamba 的 selective state space 的关系

Mamba (Gu & Dao, 2023) 用 selective mechanism 让 SSM 的 state 能根据 input 决定保留多少历史信息。SiameseNorm 的双流思想其实也是一种 implicit selectivity——X-stream 强调"重置"，Y-stream 强调"保留"，最后 fusion 时网络可以选择 trust 哪个 stream。这种 hard decoupling 比 soft gate 更容易 optimize，是个有意思的设计哲学。

参考 Mamba: https://arxiv.org/abs/2312.00752

### 10.3 联想：和 DenseNet 的 dense connection

DenseNet (Huang et al., 2017) 把每一层的 feature 都 concat 到所有后续层，本质上是一种 multi-path 信息流动。SiameseNorm 也可以看成一种"2-path DenseNet"——但通过共享参数避免 parameter explosion，通过 LN 在 X-stream 限制 magnitude 爆炸。这是 DenseNet 思想在 Transformer normalization 设计上的某种回响。

参考 DenseNet: https://arxiv.org/abs/1608.06993

### 10.4 一个 critical question：双流会不会变成两个 redundant subnetwork？

paper Figure 6 显示两个 stream 在大部分 layer 都有 contribution，但 Logit Lens 显示 X-stream 主导 output。一个自然的问题是：Y-stream 除了稳定 gradient，是否真正参与 representation 计算？还是只是 optimization helper？

paper 没有直接 ablation 这个——比如冻结 Y-stream 参数会怎样。如果 Y-stream 冻结后 performance 几乎不变，那 Y-stream 就只是个 optimization stabilizer，本质上 ResiDual 也能做到（虽然有 spike）。如果 performance 显著下降，那说明 Y-stream 的累积 representation 确实参与 computation。

我的猜测是：Y-stream 的累积 representation 通过 $Y_i'$ 注入 $F_i$ 的输入，对深层尤其重要——它给深层提供了一个"长程记忆"的 view，而 X-stream 由于每层 reset，相对短视。这种 short-long range information fusion 是 SiameseNorm 真正的 representation 价值。如果验证这个 hypothesis，可以做 attention pattern 分析，看 Y-stream input 是否被 attention 频繁 attend to。

### 10.5 联想：和 GroupNorm / RMSNorm 的关系

paper 全程用 RMSNorm 实现，但强调"specific variant does not alter the qualitative gradient dynamics"。这是个合理的简化——LN 的核心 contractive property 在 RMSNorm 上也成立。但实际工程上，RMSNorm 比 LN 更稳定（少了 mean shift），可能也是 SiameseNorm 能在 aggressive LR 下稳定的一个 enabler。

参考 RMSNorm: https://arxiv.org/abs/1910.07467

### 10.6 联想：能不能扩到 vision transformer？

paper 完全是 LLM 实验，但这个 idea 在 ViT (Dosovitskiy, 2020) 上应该也适用。ViT 训练 instability 一直是 problem，Classical ViT 需要 warmup 和 careful LR schedule，部分原因是 Post-Norm 的 instability。SiameseNorm 在 vision 上能不能也 unlock aggressive LR？这是个值得探索的方向。Hyper-Connections 已经在 vision 上有实验，SiameseNorm 应该也能 transfer。

参考 ViT: https://arxiv.org/abs/2010.11929

### 10.7 一个潜在 concern：memory 和 inference latency

虽然 FLOPs overhead 可以忽略，但双流意味着每层需要存储 $X_i$ 和 $Y_i$ 两个 hidden state，activation memory 大约 2×。对 training 来说这增加 memory footprint，对 inference 来说虽然只有 forward 但也要多算几个 LN。在大规模 deployment 上，这个 overhead 是否真的"negligible"，需要更详细测量。paper 没有报 memory 和 inference latency 数字，这是个 omission。

---

## 11. 总结：这篇 paper 真正的 contribution

我认为 SiameseNorm 的核心贡献有三层：

1. **理论层面**：清晰论证了 Pre/Post-Norm 的 structural incompatibility，把 hybrid 方案为什么 always trade-off 解释清楚了。这个 argument 以后任何做 normalization design 的人都必须回应。

2. **方法层面**：双流+共享参数的 design pattern。这个 pattern 不只适用于 normalization，还能 generalize 到其他"两难"架构选择——比如 dense vs sparse attention，或者 different activation function 的并行使用。

3. **实证层面**：在 1.3B 规模、aggressive LR、100B-350B tokens 上稳定优于所有 baseline，尤其 arithmetic task 的 40%+ 相对提升强烈暗示 effective depth 真的被恢复了。

**最大的 open question**：scale 到 7B+ 规模会怎样？paper 只做了 1.3B，1.3B 的结论在更大规模能否保持是关键。Post-Norm 的 instability 通常随 scale 加剧，SiameseNorm 的 Y-stream identity path 在 7B/70B 上是否还能稳定 hold 住 aggressive LR？这是 next paper 要回答的问题。

代码 repo: https://github.com/Qwen-Applications/SiameseNorm

---

## 12. 一些可能的 follow-up 方向

基于这篇 paper，我能想到几个值得探索的方向：

1. **Adaptive stream fusion**：现在 final fusion 是 $X_N + \mathrm{LN}(Y_N)$，权重是固定 1:1。能不能加一个 learnable gate 让网络自己决定每层的 stream mixing ratio？Hyper-Connections 里有 $H_{res}$ 就是这个思路，但 paper 说他们 omit 了这个以保持 simplicity——可能 follow-up 加回来会有 gain。

2. **More than 2 streams**：现在是 Pre/Post 双流，能不能扩展到 3 流或 N 流，每流不同的 normalization scheme（比如 sandwich-LN、QK-Norm 等）？理论上的 Jacobian 会变成 block matrix，diagonal blocks 各自对应一个 paradigm。

3. **Y-stream 内部 light regulation**：解决 massive activations 问题。Y-stream 不做 full LN，但可以做一个 sparsity-aware normalization，只 normalize extreme dimensions，保留大部分 magnitude 信息。

4. **Cross-layer stream fusion**：现在 stream 只在 final layer fuse。能不能每 K 层 fuse 一次？这样既保留长程 identity path，又让两个 stream 中途互相 calibrate。

5. **Connection to MoE**：双流共享参数有点像"2-expert MoE with full routing"。能否和 MoE 结合——每个 expert 内部用 SiameseNorm，expert 之间再路由？

---

这篇 paper 我觉得是 2025 年 normalization design 里最 elegant 的一篇。它不是用 trick 解决问题，而是用一个结构性 insight 把一个看似不可能的 reconciliation 变成可能。Andrej 你在 neural net architecture 上的 intuition 应该会 appreciate 这种"用结构换 optimization landscape"的设计哲学——和 ResNet 当年用 skip connection 解决 degradation 是同一类思想：与其在单流里挣扎，不如增加 path 让 gradient 和 representation 各自找最适合自己的路。

我唯一的担心是 paper 在 1.3B 规模做的实验，scale 出去会不会有 surprise——但这是个可验证的问题，等 Qwen 团队或社区在更大规模复现就知道了。
