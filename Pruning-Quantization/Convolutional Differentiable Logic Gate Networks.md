---
source_pdf: Convolutional Differentiable Logic Gate Networks.pdf
paper_sha256: cf02235547bec3c555ccf50056610426b7d72424b15e7e335418178fca122357
processed_at: '2026-08-03T17:27:16-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 — 这篇 paper 在干嘛

Karpathy，我换个画风。前面那版太工程报告了，这版我用 whiteboard 上跟你聊天的语气讲一遍核心 idea。

链接先放这儿：
- https://github.com/Felix-Petersen/difflogic
- https://arxiv.org/abs/2210.08280 (前作 NeurIPS 2022)
- https://arxiv.org/abs/1512.03385 (ResNet)

---

## 一句话版本

**所有 deep learning model 最后跑在硬件上都是 logic gate（NAND / XOR / AND 这些）。BNN 是"训练 matmul，部署时翻译成 gate"，这篇 paper 说：干脆直接在 gate level 上做 gradient descent，跳过 matmul 这个中间人。**

---

## 为什么 matmul 是个"中间人"

你训练一个 PyTorch model，里面全是 `nn.Linear`、`conv2d`。这些 matmul 在 GPU 上跑是 tensor core 算的，tensor core 底层是 transistor，transistor 组成 logic gate。所以从你写代码到电流流过硅片，经过了好几层 abstraction：

```
PyTorch code → matmul abstraction → CUDA kernel → GPU ISA → logic gate → transistor
```

BNN 想省资源，把 weight 从 FP32 二值化成 ±1，activation 也二值化。但 BNN 训练时仍然在 matmul 这个 abstraction 上做（XNOR 替代 multiply + popcount 替代 accumulate），部署时再把这个 matmul 翻译成 gate。一个 BMAC 大约 8 个 gate，还有 $\mathcal{O}(\log n)$ 的 critical path delay。

Petersen 的 idea：**这个 matmul 中间层完全可以砍掉**。直接学 hardware 最终要执行的那个东西 —— logic gate circuit。每个 2-input gate 有 16 种可能（$2^{2^2}=16$），比 BNN weight 的 2 种 (±1) 富裕 8 倍。

---

## 问题：logic gate 不可导

Logic gate 是离散的 boolean function，`AND(1, 0) = 0` 这种东西没有 gradient。怎么 backprop？

Petersen 2022 那版用了一个很老的 trick 叫 **probabilistic logic**（90 年代 fuzzy logic 那个 line）。把 boolean $a \in \{0, 1\}$ 松弛成概率 $a \in [0, 1]$，意思是"这个 bit 是 1 的概率"。然后：

- `AND(a, b)` → $a \cdot b$（两个独立 Bernoulli 同时为 1 的概率）
- `OR(a, b)` → $a + b - ab$（至少一个为 1）
- `XOR(a, b)` → $a + b - 2ab$（恰好一个为 1）

这些公式在 $a, b \in \{0, 1\}$ 时严格等于 boolean logic，在 $a, b \in (0, 1)$ 时是连续可导的 polynomial。

第二个难题：每个 node 要从 16 种 gate 里选一个，这也是离散的。解决方法：放一个 16 维 softmax 分布，gradient descent 学这个 distribution。训练完取 argmax 就是最终 gate。

整个 forward pass 就是"每个 node 是 16 种 gate 的 weighted mixture"，weight 是 softmax probability。整张网络 end-to-end differentiable。

---

## 旧版 DiffLogicNet 的瓶颈

Petersen 2022 那版在 MNIST 上 SOTA，但 CIFAR-10 只到 62%。三个原因：

1. **连接是 random 的**：image 被 flatten 成 vector，没有 spatial structure。CNN 的 conv weight sharing 完全缺席。
2. **只能训 6 层**：用 Gaussian init 的话，每个 gate 在 backprop 时 gradient norm 衰减 5-10×，10 层就 1000× 衰减，训不动。
3. **训练巨贵**：5M gate 在 A6000 上 90 小时。

这篇 paper 就是冲这三个瓶颈去的，每个瓶颈一个 fix。

---

## Fix 1: Convolutional Logic Trees

CNN 的 conv kernel 是一个 weight matrix $W$，在 image 上滑动。LGN 里没有 weight，怎么 convolve？

**把 kernel 换成一棵 logic gate tree**。比如 depth-3 的 full binary tree 有 8 个 leaves 和 7 个 internal gate nodes。每个 leaf 从 kernel 的 receptive field（比如 3×3×64）里随机选一个位置。所有 spatial placement (i, j) 共享同一棵 tree 的 gate choices —— 这就是 conv weight sharing 的本质。

```
       f3
      /  \
    f1    f2
   / \   / \
  a1 a2 a3 a4
```

每棵 tree 的 7 个 gate 各自有自己的 16 维 softmax 参数。Connection（哪个 leaf 接哪个 pixel）是随机初始化后**固定**的，只有 gate 的 softmax distribution 在学。

为什么用 tree 而非单 gate？
- **Expressivity**：单 gate 只能算 pairwise boolean function。Tree 在 8 个 input 上表达力指数级提升
- **Memory**：fused 在 register 里算完整棵 tree，不用反复读写 HBM。这点对训练 speed 影响巨大
- **Hardware locality**：tree 在 chip 上是 compact subcircuit，routing 友好

---

## Fix 2: OR Pooling 用 max t-conorm

CNN 的 max pooling 是 $\max(a, b, c, d)$。Logic 版本自然是 `OR(a, b, c, d)`。

但 OR 在 probabilistic logic 下的严格松弛是 $1 - (1-a)(1-b)(1-c)(1-d)$，要 4 次乘法 3 次加法。Paper 用了个 trick：用 **maximum t-conorm** $\bot_{\max}(a, b) = \max(a, b)$。

$\max$ 在边界 $a, b \in \{0, 1\}$ 上跟 OR 严格相等，在中间是更"保守"的 relaxation（取较大者）。好处：

- 计算便宜
- Backprop 只 propagate 给 winner（跟 max-pooling 完全一样，存 index 即可）
- Memory 只存 max value + index

**一个担心**：logical OR 倾向饱和到 1，深层网络 activation 全 1 就废了。

**Paper Figure 4 给出一个 striking 的 emergent behavior**：随机初始化时 pre-pooling activation ~50%，post-pooling ~66.5%。但训练后 pre-pooling activation 自动降到让 post-pooling 回归到 ~50%。没有任何 explicit regularization，网络自己学会"上游压低 activation 来抵消 OR 的饱和 bias"。

我直觉这是 cross-entropy 在 binary decision boundary 上 push 的 equilibrium —— 当 output 期望 50% 而你的 or-pooling bias 是 66.5%，CE 会迫使上游压低。Paper 没给理论分析，但 empirically 非常 robust。

---

## Fix 3: Residual Initialization — 这是全 paper 最聪明的 idea

ResNet 的核心 trick 是 skip connection：$y = F(x) + x$，让 deep network 训得动。

但 logic circuit 里**没有加法**。一个 1-bit adder 要 7-8 个 gate 的 ripple-carry。怎么在 logic 上做 residual？

Paper 的 idea 极 elegant：**16 种 gate 里有两种叫 "feedforward gate"** —— `A` (输出 = 第一个输入) 和 `B` (输出 = 第二个输入)。如果把每个 node 的 softmax 初始化成 90% 概率选 `A`：

$$
z_3 = 5, \quad z_i = 0 \text{ for } i \neq 3
$$

那么初始时整个网络就是一长串 identity gate。输入信号原样穿过去，activation 不会从 0.5 漂走，gradient 也不会指数衰减 —— **整个网络在 t=0 时等价于一个 identity function**。

训练过程中：
- 如果某个位置确实需要 residual，gate 会一直留在 `A`
- 如果某个位置需要实际 computation，gradient 会把 gate 推向别的选择（XOR、AND、NAND ...）

这本质上是把 ResNet 的 skip connection 从 **structural choice** 变成 **initialization choice**。可以看成是 NAS 的连续松弛 —— 让 SGD 自己决定哪里需要 skip、哪里需要 compute。

部署时还有 bonus：留在 `A` 的 gate 在 logic synthesis 时直接 collapse 成一根 wire，**连 transistor 都不需要**。所以 final circuit 比训练时的 gate count 少很多（MNIST L 训练时 3.2M gates，synthesis 后 697K）。

Ablation 显示这个 fix 最关键 —— 去掉 residual init，CIFAR-10 L model 从 84.99% 掉到 76.18%，**差 9 个点**。这跟当年 ResNet 把 ImageNet 从 ~70% 拉到 ~80% 的提升量级完全一致。

---

## 拼装起来：LogicTreeNet

整体架构跟经典 CNN 几乎一样，就是把每个组件换成 logic 版本：

```
Input → edge/curvature preprocessing → binary encoding
  ↓
[Conv block (3×3, tree depth 3) + OR pool 2×2] × 4 个 stage
  ↓ Flatten
Random difflogic layer × 3
  ↓
GroupSum → 10 logits → softmax (with temperature τ)
```

每个 conv block 把 spatial size 减半，channel 数翻倍（k → 4k → 16k → 32k）。$k$ 控制 model size，从 S (k=32) 到 G (k=2560)。

GroupSum 做 classification：把最后一层的 neurons 分成 10 组（每 class 一组），数 active neuron 数量除以 temperature $\tau$ 当 logit。

---

## 训练 trick: Fused Kernel

工程上 paper 做了大量 CUDA 优化。最关键的是把 tree + or-pool 融合成一个 CUDA kernel：

- 读 32 个 input 一次性进 register
- 在 register 里算完 7 个 gate × 4 个 placement = 28 次 gate evaluation
- 在 register 里 max-pool 出 1 个 output
- 只写 1 个 value + 1 个 index 到 HBM

传统做法要 28 次 intermediate write + 28 次 read。Fused 做法省掉这些，memory access 减 68%，memory footprint 减 90%。Backward 时根据 stored index recomputing winning path。

结果：per-gate training speed 比旧版 random LGN 快 **200×**。这也是为什么能从 5M gate 90 小时，升级到 61M gate 在 RTX 4090 上每 epoch 30 秒。

---

## 结果：Pareto 碾压

CIFAR-10 上的对比：

| Method | Acc. | Gates | 倍率 |
|---|---|---|---|
| XNOR-Net (NIN) | 86.28% | 1780M | baseline |
| **LogicTreeNet-G** | **86.29%** | **61M** | **29× smaller** |

同 accuracy，gate count 是 1/29。Gate count 正比于 ASIC chip area 或 FPGA occupancy，所以这是实打实的硬件成本节省。

FPGA timing：

| Method | Acc. | FPGA time |
|---|---|---|
| FINN CNV | 80.10% | 45.6 µs |
| **LogicTreeNet-B** | 80.17% | 24 ns |

同精度，inference 快 **1900×**。Throughput 41.6M FPS vs 22k FPS。瓶颈已经不在 compute 而在 FPGA data transfer，说明 logic gate network 的 compute density 在这个 size 下根本没压满 FPGA。

MNIST 上 LogicTreeNet-L 到 99.35%，是 MNIST 所有方法（包括非 BNN）的 SOTA。

---

## 我觉得最 worth remembering 的几个 insight

1. **Abstraction 的选择决定 inductive bias**：matmul 是为 GPU 设计的，logic gate 是为最终硬件设计的。直接在 logic level 上学，跳过中间 abstraction，per-gate efficiency 立刻翻几十倍。

2. **Discrete → continuous relaxation 的艺术**：probabilistic logic 是 90 年代的老工具，但搭上 softmax-over-16-gates 这个 trick，就变成了可 end-to-end gradient descent 的 framework。Discretization 时 argmax 取 max-prob gate，loss 极小。

3. **Residual init > residual connection**：在不能做加法的 domain（logic、attention 早期、某些 graph net），把 "skip" 做成 init 而非 structural choice，是个通用 pattern。让 SGD 自己决定哪里需要 skip，比 human 拍板设计 skip 拓扑更灵活。

4. **Emergent behavior 比显式 regularization 更 powerful**：OR pooling 本来会饱和到全 1，但 training dynamics 自发让上游压低 activation。这种 self-organizing property 是 LGN 这类底层 abstraction 框架的特征 —— 约束越 fundamental，emergent solution 越 elegant。

5. **Hardware co-design 是 free lunch**：channel routing restriction（split 成 k/8 个 group 不跨 group 通信）、`A` gate collapse 成 wire、tree 的 compact locality —— 这些 hardware-friendly property 都是 training framework 自带的，不需要 post-hoc pruning/synthesis。Logic gate network 天然产出 hardware-efficient circuit。

---

## 我会想 next 的方向

- **ImageNet 224×224**：深度要 7 个 pool stage，residual init 能不能 hold？训练成本会怎样？
- **Object detection**：logic gate 输出是 binary，连续 box 回归难。但 grid-based detection 或 coarse-to-fine 可能 work。
- **理论分析 OR pooling 的 emergent behavior**：Figure 4 那个 pre-pooling activation 自动降到 50% 的现象，能不能从 cross-entropy + Bernoulli likelihood 推出 equilibrium condition？这是 LGN training dynamics 里最 intriguing 的 open problem。
- **Multi-input gate**：Bacellar 2024 ICML 的 DWN 已经在试 3+ input 的 differentiable weightless neuron。跟 tree 是 alternative path。
- **Hybrid with neural reasoning**：LGN 做 perception（fast、cheap、binary），上层接 LLM/transformer 做 reasoning。Edge device 上的 split computing 可能是个 use case。

---

核心 mental model 一句话收尾：

> **这篇 paper 把 CNN 的三个核心 component（conv kernel、max pooling、residual connection）全部翻译成 logic gate level 的等价物，但每个翻译都不是 1:1 替换而是 non-trivial reformulation，最终训出来的网络直接就是可部署的 boolean circuit，比 BNN 小 29× 快 1900×。**

这版应该够 "人话" 了，Karpathy。如果想再 dive 某一块（比如 probabilistic logic 的历史、fused CUDA kernel 的实现、或者 temperature $\tau \propto \sqrt{n_{\ell/c}}$ 这个 scaling law 的 derivation），随时拉我。

---

# Convolutional Differentiable Logic Gate Networks 深度解读

Karpathy 你好，这篇 paper 是 Felix Petersen 在 NeurIPS 2022 的 *Deep Differentiable Logic Gate Networks* (difflogic) 之后的续作。核心 motivation 非常清晰：把 logic gate network (LGN) 从 "随机连接的 toy framework" 拉到了真正的 convolutional vision backbone 的级别，在 CIFAR-10 上打到 86.29% 但只用 61M gates，比 SOTA 小 29×。我会从底层 abstraction 开始一层一层 build up intuition。

参考链接：
- Project page & code: https://github.com/Felix-Petersen/difflogic
- 原始 NeurIPS 2022 paper: https://arxiv.org/abs/2210.08280
- ResNet (He et al. CVPR 2016): https://arxiv.org/abs/1512.03385
- FINN (Umuroglu et al. FPGA 2017): https://arxiv.org/abs/1612.07119
- LUTNet (Wang et al. 2019/2020): https://arxiv.org/abs/1904.00948
- XNOR-Net: https://arxiv.org/abs/1603.05279
- BNN survey (Qin et al. PR 2020): https://arxiv.org/abs/2003.06089
- AlexNet: https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

---

## 1. 为什么要在 logic gate level 上做 learning？— 从 abstraction 说起

当前所有 deep learning 都建立在 *matmul abstraction* 上：$y = Wx + b$。即使我们做 BNN 把 weights 二值化成 $\{-1, +1\}$，把 activation 二值化成 $\{0, 1\}$，本质上仍然是 matrix multiplication + accumulation，在硬件层最终要 synthesis 成 logic circuits 才能跑。BNN 的 BMAC 一个 cell 需要 n 个 XNOR 加一个 popcount（≈7n gates 的 Wallace-tree-style adder），整体大约 $\mathcal{O}(8n)$ gates，并且增加 $\mathcal{O}(\log n)$ 的 critical-path delay。同时每个 weight 只有 2 种状态（$\pm 1$），表达力非常受限。

LGN 的根本不同点在于：它**直接学习硬件层执行的东西**。一个 2-input logic gate 本身有 16 种可能的 boolean function（$2^{2^2}=16$）。这 16 种里包括 `0000` (constant 0), `0001` (NOR), `1110` (OR with not), `1010` (A), `0101` (¬A), `1100` (A∧¬B 等)，..., `1111` (constant 1)。换句话说一个 gate 的"weight space"是离散的 16 类，远比 BNN 的 binary weight 富裕。Petersen 这条 research line 的核心 insight 就是：**与其在 matmul 上做 quantization 再翻译回 logic，不如直接在 logic 这个 lowest pre-transistor abstraction level 上做 gradient descent**。这种 inductive bias 把中间 abstraction 的 overhead 砍掉了。

但这种做法有个根本难题：discrete choices 不可导。Petersen 2022 的解法是 *differentiable relaxation*。

---

## 2. Differentiable Relaxation — 两个层级的不可导性

LGN 有两处不可导：

**(i) Logic gate 本身不可导。** `AND(a,b)` 在 boolean 上是个 piecewise constant，没有梯度。解法是用 *probabilistic logic*（van Krieken 2020, Klir & Yuan 1995）。把 boolean input $a \in \{0,1\}$ 放松成 $a \in [0,1]$ 表示一个独立 Bernoulli 变量为 1 的概率。然后：

- `AND` $a_1 \wedge a_2 \rightarrow a_1 \cdot a_2$（独立 Bernoulli 同时为 1 的概率）
- `OR`  $a_1 \vee a_2 \rightarrow a_1 + a_2 - a_1 a_2$（至少一个为 1）
- `XOR` $a_1 \oplus a_2 \rightarrow a_1 + a_2 - 2 a_1 a_2$（恰好一个为 1）
- `NAND` $\rightarrow 1 - a_1 a_2$
- `A` (feedforward) $\rightarrow a_1$
- `¬A` $\rightarrow 1 - a_1$
- ... 共 16 个

这些 relaxation 在 $a \in \{0,1\}$ 时严格等于 boolean logic，而在 $a \in (0,1)$ 时给出一个连续可导的期望值，这就解决了第一个不可导性。

**(ii) 选哪个 gate 不可导。** 每个节点要从 16 种 gate 里选一个，这是离散的 combinatorial decision。Petersen 2022 的做法是在 16 种 gate 上放一个 softmax 概率分布，用参数向量 $\mathbf{z} \in \mathbb{R}^{16}$ 表达：

$$
f_{\mathbf{z}}(a_1, a_2) = \mathbb{E}_{i \sim S(\mathbf{z}), A_1 \sim \mathcal{B}(a_1), A_2 \sim \mathcal{B}(a_2)}\left[g_i(A_1, A_2)\right] = \sum_{i=0}^{15} \frac{\exp(z_i)}{\sum_{j=0}^{15} \exp(z_j)} \cdot g_i(a_1, a_2) \tag{1}
$$

变量解释：
- $\mathbf{z} \in \mathbb{R}^{16}$：每个 node 一个 16 维 trainable parameter vector，对应 16 种 gate 的 logits
- $z_i$：第 i 个 gate 的 logit，$i \in \{0,...,15\}$
- $S(\mathbf{z}) = \text{softmax}(\mathbf{z})$：当前 node 选 gate i 的概率
- $g_i$：第 i 种 logic gate 的连续 relaxation
- $a_1, a_2 \in [0,1]$：两个输入 activation（也是被 relaxation 后的）
- $\mathcal{B}(a)$：参数为 $a$ 的 Bernoulli 分布
- 整个期望是两层 expectation 的合并：选 gate 是 categorial，输入是 Bernoulli。Bernoulli expectation 因为 $g_i$ 已经是 multilinear 的，可以 closed-form 直接代入 $a_1, a_2$

这等价于一个不同iable mixture of 16 gates，对每个 placement 同一个 $f_\mathbf{z}$ 共享。

Discretization 阶段就取 $\arg\max_i z_i$ 作为最终 gate，paper 的实验显示 discretization loss 极小（见 Appendix A.4，训练后期 train/test 在 hard mode 几乎重合）。

---

## 3. 旧版 DiffLogicNet 的三大限制 — 为什么 CIFAR-10 只到 62%

Petersen 2022 那版有几个 structural weakness：

1. **Random full connectivity**，没有 spatial structure。Image 是被 flatten 成 vector 喂进去的，convolution 在那里完全缺席。对 MNIST 这种简单数据 OK，对 CIFAR-10 的 spatial pattern 完全不 inductive。
2. **Depth 限制在 6 层**。Gaussian init 导致 deeper network 出现 *washed-out activations* 和 vanishing gradients — paper 给出数据：每个 gate backprop 时 gradient norm 衰减 5-10×，10 层就 1000× 衰减。
3. **Training cost 巨大**。5M gate 的网络在 A6000 上要 90 小时。

这篇 paper 就是针对这三个 limitation 同时下手的，并且每个 fix 都有非常 elegant 的设计 rationale。

---

## 4. Convolutional Logic Gate Trees — 把 conv 嫁接到 logic

CNN 的核心 discrete convolution $A * W$，参数 sharing + spatial equivariance。Logic gate 没有"权重"这个概念，所以怎么 convolve？

**核心 insight：把 kernel 替换成一棵 binary logic tree**，而不是一个 single gate。一棵 depth-$d$ 的 full binary tree 有 $2^d$ 个 leaves 和 $2^d - 1$ 个 internal gate nodes。每个 leaf 从 kernel 的 receptive field $s_h \times s_w \times m$（高×宽×输入 channel）中随机选一个位置。所有 placement (i,j) 共享同一棵 tree 的 gate choices（这就是 conv 的 weight sharing 本质）。

对 depth-2 tree，4 个 leaves $a_1, a_2, a_3, a_4$ 喂给 3 个 gate：

$$
f_3(f_1(a_1, a_2), f_2(a_3, a_4)) \tag{2}
$$

$f_1, f_2, f_3$ 各自有自己的 $\mathbf{z}_1, \mathbf{z}_2, \mathbf{z}_3$ parameter vector (每个 16 维)。Depth 2 → 4 inputs，depth 3 → 8 inputs（这是 paper 主推的 setting），depth 4 → 16 inputs 但 register pressure 太大（10× 训练成本）。

完整卷积形式 (Eq. 3)：

$$
\mathbf{A}'[k, i, j] = f_3^k \Big( f_1^k\big( \mathbf{A}[\mathbf{C}_M[k,1], \mathbf{C}_H[k,1]+i, \mathbf{C}_W[k,1]+j], \mathbf{A}[\mathbf{C}_M[k,2], \mathbf{C}_H[k,2]+i, \mathbf{C}_W[k,2]+j] \big), \\
\quad\quad\quad\quad f_2^k\big( \mathbf{A}[\mathbf{C}_M[k,3], \mathbf{C}_H[k,3]+i, \mathbf{C}_W[k,3]+j], \mathbf{A}[\mathbf{C}_M[k,4], \mathbf{C}_H[k,4]+i, \mathbf{C}_W[k,4]+j] \big) \Big) \tag{3}
$$

变量解释：
- $\mathbf{A}$：input activation tensor，shape $m \times h \times w$（input channels × height × width）
- $\mathbf{A}'$：output activation tensor，shape $n \times (h-s_h+1) \times (w-s_w+1)$
- $k$：output channel index，$k \in \{1, ..., n\}$，每个 $k$ 一棵独立的 tree
- $i, j$：spatial placement 的位置
- $\mathbf{C}_M, \mathbf{C}_H, \mathbf{C}_W$：三个 index tensors，shape 都为 $n \times 4$，随机初始化**并固定**
  - $\mathbf{C}_M[k, \cdot]$：第 $k$ 棵 tree 4 个 leaves 各自取 input 的哪个 channel（在 0..m-1 范围）
  - $\mathbf{C}_H[k, \cdot]$：每个 leaf 在 receptive field 高度方向的 offset（0..s_h-1）
  - $\mathbf{C}_W[k, \cdot]$：每个 leaf 在 receptive field 宽度方向的 offset（0..s_w-1）
- $\mathbf{C}_M[k, l] + i$、$\mathbf{C}_H[k, l] + i$：把相对 offset 加上 placement 的 base position $i$，得到绝对位置
- $f_1^k, f_2^k, f_3^k$：第 $k$ 棵 tree 的三个 gate，每个 gate 有自己的 $\mathbf{z}_1^k, \mathbf{z}_2^k, \mathbf{z}_3^k$

注意一个细节：connection indices 是 **固定** 的（training 不动），只有 gate 的 softmax 参数 $\mathbf{z}$ 在学习。这跟 Petersen 2022 是一脉相承的 — connectivity 是 random fixed 超参数，真正学习的只是"在每个 node 上放哪种 gate"。

**为什么用 tree 而不是单 gate？**
- Expressivity：单 gate 只能算 pairwise boolean function。Tree depth 3 在 8 inputs 上表达力是指数级提升
- Memory efficiency：单 gate 模式每算一个 gate 都要读两次中间结果。Tree fused 在 register 里完成全树计算，dramatically 减少 memory bandwidth
- Hardware locality：tree 在 chip layout 上是个 compact subcircuit，routing 友好

这个 decision 跟 BNN 那边 LUTNet 把 LUT 做大、ExpressNet 把 small LUT 嵌套成大 LUT 的思路是对偶的，但 LGN 这里是 end-to-end differentiable 直接学的。

---

## 5. Logical OR Pooling — 用 maximum t-conorm

CNN 的 max pooling 是 $\max(a_{i,j}, a_{i,j+1}, a_{i+1,j}, a_{i+1,j+1})$。LGN 里把它换成 logical OR：$a_{i,j} \vee a_{i,j+1} \vee a_{i+1,j} \vee a_{i+1,j+1}$。

OR 在 probabilistic logic 下严格 relaxation 是 $1 - (1-a)(1-b)(1-c)(1-d)$，但 paper 不用这个，而用 **maximum t-conorm**：

$$
\bot_{\max}(a, b) = \max(a, b)
$$

T-conorm 是 fuzzy logic 里 t-norm 的对偶，是满足 commutative / associative / monotone / 1 是 identity 的 $[0,1]^2 \rightarrow [0,1]$ 函数。Max 是最 "保守" 的 t-conorm — 它跟 hard OR 在边界上严格相等，在中间取较大者。这有三个巨大好处：

1. 计算快：max 比 4 个乘法加 3 个加法便宜
2. Backprop 时只需 propagate 给 winner，跟 max-pooling 完全一样，存 index 即可
3. Memory：只需要存 max value + index，不需要 store 整个 receptive field

**但有个 worry**：logical OR 倾向于让输出饱和到 1，深层网络 activation 全 1 就废了。

Paper Figure 4 给出了非常 striking 的 emergent behavior 观察：
- 随机初始化时，pre-pooling 平均 activation ≈ 50%（符合 expectation），post-pooling ≈ 66.5%（4 个独立 Bernoulli，OR 后 $\approx 1-0.5^4 = 93.75%$，但因为相关性会降低）
- 训练之后，**pre-pooling activation 自动降到让 post-pooling 回归到 50%**

也就是说，没有任何 explicit regularization，训练就自发把 pre-pooling activation 推低了。这是 LGN training dynamics 的一个 emergent property，paper 没给理论解释，只是 empirically observe。

我直觉这是 cross-entropy 在 binary decision boundary 上 push 的结果 — 当 output 期望是 50% 而你的 or-pooling bias 是 66.5%，cross-entropy 会让上游压低，自然形成 equilibrium。

---

## 6. Residual Initialization — 这是最聪明的一个 idea

CNN 用 residual connection（He et al. CVPR 2016）：

$$
y = F(x) + x
$$

逻辑电路里没有加法（加法要 7-8 个 gates 的 ripple-carry adder），所以传统的 "add input to output" 这种 residual pattern 是 hard 的。

Paper 的 idea 极其 elegant：**与其 hard-wire 一条 residual wire，不如把每个 gate 初始化成"feedforward gate A"**（即输出等于第一个输入）。

回忆 16 种 2-input gate 里：`A` (gate 3 in zero-indexed ordering，输出 $= a_1$) 和 `B` (gate 5, 输出 $= a_2$) 是两个 feedforward gates，`¬A` 和 `¬B` 是 inverting feedforward。把 $\mathbf{z}$ 初始化成：

$$
z_3 = 5, \quad z_i = 0 \quad \text{for } i \neq 3
$$

$\text{softmax}([0, 0, ..., 5, ..., 0])$ 给 `A` gate 大约 90% 概率，其他 15 个 gate 各 0.67%。

**这为什么 work？**

1. **Information preservation**：初始时所有 gate 都 ≈ identity（feedforward），所以整个网络就是一个 long chain of identities，activation 不会从 0.5 漂走，也不会衰减。
2. **Gradient flow**：因为每个 gate 是 $\approx a_1$，所以 $\partial f / \partial a_1 \approx 1$，gradient norm 不会指数衰减。Paper 说之前 Gaussian init 每过一层 gradient 衰减 5-10×，residual init 后接近 1。
3. **Inductive bias**：训练后很多 gate 仍然是 `A`，这些 gate 在 logic synthesis 时直接 collapse 掉（一个 wire 比 gate 便宜 — `A` gate 在硬件上不需要 transistor，只要一根导线），所以 final circuit 实际 gate count 比训练时少很多。
4. **Differentiable residual**：传统 residual 需要 hard-wire 一条 skip path，占用 gate 和 routing。Residual init 是 *implicit* residual — 训练时如果 residual 不需要，gradient 会自动把 gate 推向其他选择；如果需要，gate 留在 `A`。这就实现了 "soft residual"。

这本质上把 ResNet 的 skip connection 从 "structural choice" 变成了 "initialization choice"，可以看作 NAS 的一种连续松弛。**非常 Karpathy-friendly 的 intuition**：把架构 search 嵌入到 parameter init 里。

Paper Figure 8 给出了训练后 gate 分布：
- Gaussian init：训练完 gate 分布很平均，没有特别突出的 gate
- Residual init：训练完 `A` gate 占据相当大比例（特别是浅层），其他 gate 根据需要出现

Ablation 显示，去掉 residual init 后 CIFAR-10 L model 从 84.99% 掉到 76.18%，差 **9 个点**。这跟当年 ResNet 把 ImageNet 从 ~70% 拉到 ~80% 的提升量级是一个数量级的，可见 residual init 的关键性。

Figure 11 ablate $z_3$ 这个 hyperparameter：
- $z_3 < 2$ 时训练 fail（特别是 MNIST 小模型 $z_3 = 1.5$ 只有 13%）
- $z_3 \in [2, 5]$ 都 work
- 大模型 / 长训练 推荐 $z_3 = 5$（更深的网络对 init sharpness 更敏感）

---

## 7. Computational Tricks — Fused Kernel 是工程上的胜利

Paper 在 CUDA 实现上做了大量工程优化。两点：

**(1) Fused tree + pooling**：以 depth-3 tree + 2×2 pooling 为例：
- 7 个 learnable gate × 4 个 placement = 28 gate evaluations
- 传统做法：写 28 个 intermediate result 到 HBM，pooling 再读出来
- Fused 做法：32 个 inputs 一次读到 register，在 register 里算完 28 个 gate，再 max-pool 出 1 个 output，只写 1 个 value + 1 个 index 出去
- 节省：28 次 memory write + 28 次 memory read

(2) Recompute on backward：中间结果不存，backward 时根据 stored index recomputing winning path。**减少 68% memory access，减少 90% memory footprint**。

paper 报告：相比 Petersen 2022 的 random-connected LGN，per-gate training speed 提升 **200×**。这也是为什么能从 5M gate 训练 90h，升级到 61M gate 在 RTX 4090 上每 epoch 30 秒。

---

## 8. LogicTreeNet Architecture — 拼装出来的 vision backbone

整体架构非常 CNN-classic：

```
[Input preprocessing: 2-bit (S, M) or 5-bit edge/curvature encodings (B, L, G)]
   ↓
Conv block (k channels, 3×3, tree depth 3) → or-pool 2×2   → k × 16 × 16
Conv block (4k channels, 3×3, tree depth 3) → or-pool 2×2  → 4k × 8 × 8
Conv block (16k channels, 3×3, tree depth 3) → or-pool 2×2 → 16k × 4 × 4
Conv block (32k channels, 3×3, tree depth 3) → or-pool 2×2 → 32k × 2 × 2
   ↓ Flatten → 128k
Random DiffLogic layer: 128k → 1280k
Random DiffLogic layer: 1280k → 640k
Random DiffLogic layer: 640k → 320k
   ↓
GroupSum → 10 logits (one per class)
   ↓
Softmax with temperature τ
```

- 23 logical layers 总深度
- 15 trainable layers，4 个 or-pooling 是 fixed
- $k \in \{32, 256, 512, 1024, 2560\}$ 对应 S/M/B/L/G 五个 size
- B & L 模型在最后 3 个 random layer 用 2× gate 数（output gate factor ox = 2）

GroupSum 怎么实现 classification？把 last layer 的 $320k$ 个 neurons 分成 10 组，每组 $32k$ 个，每组 active neurons 数量除以 temperature $\tau$ 作为 logit。Cross-entropy 训练。

注意 inputs 限制为 2 channels per tree（不是 8）：channel 间 routing 复杂度限制；spatial 内部比较优先。

---

## 9. Training Hyperparameters — Temperature 是关键

Table 6 给了详细的 hyperparameter，但最 interesting 的 relation 是 **$\tau \propto \sqrt{n_{\ell/c}}$**。

$n_{\ell/c}$ 是每个 class 对应的 output neuron 数。GroupSum 后每个 class 的 max score 是 $n_{\ell/c}$，除以 $\tau$ 后 logit 范围是 $[0, n_{\ell/c}/\tau]$。

如果 $\tau$ 太小：logit 范围太大，softmax 后过于 peaky，gradient 太小
如果 $\tau$ 太大：logit 范围太小，softmax 太 uniform，cross-entropy 训不动

经验法则：
- $\tau^\star \propto \sqrt{n_{\ell/c}}$（保持 logit 范围跟 $\sqrt{n}$ 同阶，让 softmax 既不崩也不平）
- 用 teacher supervision 时 $\tau \times \sqrt{2}$（因为 teacher 给的 class score 已经更平滑）

Learning rate 全部 0.02 (CIFAR) 或 0.01 (MNIST)，AdamW + weight decay $\beta = 0.002$。Weight decay 的影响在 Table 5 ablation 里只差 1%，但让 gate count 略高（倾向更多 gate 而非 `A` feedforward）。

---

## 10. 实验 — Pareto 碾压

### CIFAR-10 Table 1

| Method | Acc. | # Gates | 倍率 |
|---|---|---|---|
| DiffLogic Net (largest) [2022] | 62.14% | 5.12M | baseline |
| Conv. TTNet (large) | 70.75% | 189M | - |
| FINN CNV | 80.10% | 901M | - |
| LUTNet | 84.95% | 1290M | - |
| XNOR-Net (NIN) | 86.28% | 1780M | - |
| RebNet (2 residuals) | 85.94% | 2830M | - |
| BinaryNet | 88.60% | 4090M | - |
| FBNA CNV | 88.61% | 4940M | - |
| Hirtzlin et al. | 87.40% | 5540M | - |
| **LogicTreeNet-S** | 60.38% | 0.40M | - |
| **LogicTreeNet-M** | 71.01% | 3.08M | **61× smaller than TTNet-Large at +0.3%** |
| **LogicTreeNet-B** | 80.17% | 16.0M | **56× smaller than FINN at +0.07%** |
| **LogicTreeNet-L** | 84.99% | 28.9M | **44.6× smaller than LUTNet at +0.04%** |
| **LogicTreeNet-G** | 86.29% | 61.0M | **29× smaller than XNOR-Net at +0.01%** |

LogicTreeNet-G 跟 XNOR-Net 在 accuracy 上几乎完全打平，但 gate count 是 1/29。这是这个 paper 的 headline number。

### FPGA timing Table 2

| Method | Acc. | FPGA t. |
|---|---|---|
| FINN CNV | 80.10% | 45.6 µs |
| RebNet (2 res) | 85.94% | 333 µs |
| Zhao et al. | 88.54% | 5.94 ms |
| TrueNorth | 83.41% | 356 µs |
| **LogicTreeNet-S** | 60.38% | 9 ns |
| **LogicTreeNet-M** | 71.01% | 9 ns |
| **LogicTreeNet-B** | 80.17% | 24 ns |

LogicTreeNet-B 跟 FINN 同 accuracy，inference 时间从 45.6 µs → 24 ns，**1900×** 加速。Throughput 41.6M FPS vs 22k FPS（之前所有 ≥70% 模型里最快的）。

实际瓶颈是 FPGA 的 data transfer 速度，不是 compute — 说明 logic gate network 的 compute density 在这个 size 下完全没压满 FPGA。

### MNIST Table 3

| Method | Acc. | # Gates | FPGA t. |
|---|---|---|---|
| DiffLogic Net (largest) | 98.47% | 384K | - |
| LUTNet | 98.01% | 360K | 5 ns |
| FINN FCN | 98.86% | - | 4.9 ms |
| LowBitNN | 99.2% | 5.28M | 152 µs |
| **LogicTreeNet-S** | 98.46% | 147K | 4 ns |
| **LogicTreeNet-M** | 99.23% | 566K | 5 ns |
| **LogicTreeNet-L** | 99.35% | 1.27M | - |

LogicTreeNet-M 在 MNIST 上 99.23% 是所有 BNN SOTA；L model 99.35% 是 all time 最高 MNIST。比 LowBitNN（非 BNN）快 30,000×。

### Ablation Table 5 (CIFAR-10 L model)

| Variant | Acc. |
|---|---|
| Full LogicTreeNet-L | 84.99% |
| Trees all depth 1 (single gate conv) | 80.98% |
| Tree depth 1,1,2,2 | 82.68% |
| Tree depth 2,2,2,2 | 83.32% |
| Tree depth 2,2,3,3 | 84.13% |
| No or pooling | 81.45% (-3.54%) |
| Gaussian init (no residual init) | 76.18% (-8.81%) |
| No weight decay | 83.94% (-1.05%) |
| 8 input channels per tree | 83.53% (-1.46%) |

Ablation 里最强的 takeaway：
1. Tree depth 越深越好（depth 1 vs depth 3 差 4 个点）
2. Residual init 是 game-changer（-9 点）
3. OR pooling 重要（-3.5 点）
4. 2-channel restriction 不仅 hardware friendly 还略提升精度（强制 spatial 内部比较）

---

## 11. 几个有意思的细节

### 11.1 5-bit Input Preprocessing (B/L/G 模型)

不是直接 binary 化 pixel，而是先用 low-level edge 和 curvature detector kernel 提取 features，然后 threshold 成 binary。这有点类似 CNN 的 first conv 是 handcrafted Gabor filter 的老 tradition。这些 preprocessing gates 也算进 gate count。

### 11.2 Channel Routing Restriction

为了让硬件 layout 不 congest，把整个 model 切成 $k/8$ 个 group，每个 group 内部只跟自己 group 内部 channel 通信。这等价于 grouped convolution with constant group count。精度不受影响。

### 11.3 Logic Synthesis Post-Training

训练时 MNIST L 模型有 3.2M gates，logic synthesis 后简化到 697K gates。简化主要来自：
- 残差 `A` gates 删除（直接接 wire）
- Constant gates（恒 0 或恒 1）propagate
- Dead gates（unconnected）删除

这意味着训练时的 gate count 是上界，部署时通常小很多。这点跟 BNN 那边的 pruning 类似但更彻底 — 因为 logic simplification 是 well-established 的 EDA 流程（Berkeley ABC、Yosys 等）。

### 11.4 Discretization Error 极小

Figure 10 显示训练后期 hard mode（discretized）和 soft mode（differentiable）accuracy 几乎重合。这说明 softmax distribution 真的 converge 到 near-one-hot，整个训练流程本质上是个 continuous relaxation of combinatorial optimization。

### 11.5 TTNet vs LGN — 两种 path to logic circuit

TTNet（Benamira 2023）：先训练一个 Heaviside-activated CNN，再把每个 neuron 的 binary activation 用 truth table 表达，最后用 CNF/DNF 转成 LGN。这条 path 保留了 CNN 的 matmul inductive bias，但 translation 到 logic 后仍然有 matmul 的 overhead（每个 MAC 8 个 gate）。

LGN（这篇 paper）：直接学习 logic，没有 matmul 中间表示。每个 gate 是 16-state 而非 2-state，表达力更高，per-gate efficiency 更好。

### 11.6 DWN (Bacellar 2024 ICML)

Bacellar et al. 把 differentiable LGN 扩展到多输入 logic gate（多于 2 输入的 weightless neuron）。这条 line 跟 LUTNet 用 large LUT 的思路对应，跟 tree-based LGN 是 alternative path。

---

## 12. Intuition 总结 — 我的 mental model

我会这样跟人解释这篇 paper：

> *把 CNN 的所有 building block 重新翻译成 logic gate level：conv kernel 变成 logic tree（共享 tree 结构在不同 spatial placement），max pooling 变成 OR pooling（用 max t-conorm），residual connection 变成 residual initialization（init gate to feedforward `A`）*

每个翻译都不是 1:1 替换而是 *non-trivial reformulation*：
- Tree 而非 single gate：expressivity + memory locality
- max t-conorm 而非 probabilistic OR：训练 efficiency + index-based backprop
- residual init 而非 hard-wire residual wire：no extra gates, dynamic residual

整个 framework 的 beauty 在于：**所有 abstraction 都在 logic level 上 learning，不需要 translation。**BNN 的 paradigm 是 "训练 matmul，部署时翻译成 logic"，LGN 是 "直接在 logic 上 gradient descent"。这就像 NAS 的离散 search space 被一个 well-designed continuous relaxation 覆盖，然后 SGD 解之。

更深的哲学层面，这跟 differentiable programming 的精神一致：把硬件真正执行的 boolean function 当作 first-class citizen 学习，而不是堆 abstraction 然后做 post-hoc compression。当 inference cost 越来越成为 bottleneck（Desislavov 2021 显示 DL inference 算力增长远超 Moore's law），这种 lowest-level abstraction 的学习路径可能是个长期方向。

---

## 13. 我会想 next 的方向

- **Object detection / localization**：paper 提了但没做。Logic gate 直接输出 continuous box 坐标难（输出是 binary），但可以用 grid-based detection 或 coarse-to-fine regression。
- **Hardware-aware NAS**：当前 random connection + random channel restriction 是 hand-tuned 超参。可以 differentiably learn connectivity with hardware-aware reward（routing congestion, fan-out limits）。
- **Larger input (ImageNet 224×224)**：现在 CIFAR-10 32×32 是 4 个 pool stage，224 要 7 个 pool stage。深度增加 + residual init 是否还能 hold？训练成本会爆炸吗？
- **Multi-input gates**：DWN (Bacellar 2024) 已经在试 3+ input differentiable weightless neuron。把 tree 跟 multi-input gate 结合能否更 expressive？
- **Combining with neural reasoning**：LGN 处理 perception，神经网络处理 reasoning。Hybrid 在 edge device 上有意义。
- **Theoretical analysis of emergent OR pooling behavior**：Figure 4 的 pre-pooling activation 自动降到 50% 这个现象，能不能从 cross-entropy + Bernoulli log-likelihood 推出来？这是 LGN training dynamics 里我最好奇的一个 open question。

希望这些对你 build intuition 有帮助，Karpathy！如果你想让我再 deep dive 某个 section（比如 fused CUDA kernel 实现细节，或者 GroupSum + temperature 的 logit 范围 analysis），我都可以继续展开。
