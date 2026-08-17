---
source_pdf: Parallel Training in Spiking Neural Networks.pdf
paper_sha256: 0fd71c4bb57d3b7e4bd7aff1f17fb20589828ee57cf7406dcef407572712e574
processed_at: '2026-08-06T02:08:38-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

Karpathy，咱们喝杯咖啡聊聊这篇paper。

---

## 一句话说清楚

**SNN训练慢是因为有个"reset"操作逼着你必须一步一步算，作者把这个reset干掉，换成一个"动态遗忘因子"，既能并行训练，又能保持SNN的好处。**

---

## 为啥SNN训练慢？

LIF neuron的工作流程是这样：

1. 输入来了 → 膜电位往上涨
2. 超过阈值 → 发spike，然后**reset清零**
3. 没超过 → 继续累积

问题就在第2步的reset。因为你**必须先知道上一时刻有没有发spike，才能算这一时刻的膜电位**。这种"必须等上一步完成才能算下一步"的依赖，GPU最讨厌——GPU喜欢的是"给我一堆数据我一次性全算完"。

打个比方：你在做一道菜，必须等前一步炒完才能切下一刀。GPU想同时帮你切100刀，但你不让它切，因为"这一刀切不切取决于上一刀炒没炒好"。

---

## 之前别人怎么解决？

**路线A（PSN他们）**：直接把reset删了，换成一个大的可学习矩阵。

问题：这个矩阵大小是 $T \times T$（$T$是序列长度）。序列越长，矩阵越大，参数越多。而且训练时参数跟序列长度绑死了——你用2048长度训练，推理时遇到30000长度直接崩。

**路线B（SpikingSSM他们）**：保留reset，但用近似方法绕过。

问题：近似再怎么搞也超不过原来的LIF neuron。

---

## 这篇paper的核心洞察

作者说：**别盯着"reset"这个具体机制，想想它到底在干嘛。**

reset其实干两件事：

**第一件事：引入非线性**

没有reset的话，膜电位就是个纯粹的加权求和——线性滤波器，表达力很弱。reset让"输出"能反过来影响"状态"，这就非线性了。

**第二件事：控制膜电位别爆炸**

输入一直来，膜电位一直涨，不reset就炸了。reset保证不管输入多大，膜电位都在合理范围内。

---

## 作者的解决方案

**用一个"动态decay因子"$\alpha_t$ 替代reset。**

核心公式就一行：

$$H_t = \alpha_t \cdot H_{t-1} + (1-\alpha_t) \cdot X_t$$

- $\alpha_t$ 接近0 → 等于hard reset（清零）
- $\alpha_t$ 接近1 → 完全保留历史
- $\alpha_t$ 取中间值 → 部分遗忘

关键：**这个 $\alpha_t$ 不是固定的，是由当前输入通过一个小的causal convolution算出来的。**

所以它叫"dynamic decay"——遗忘程度是input-dependent的，自适应的。

---

## 为啥这样就能并行？

因为去掉reset后，膜电位变成了**线性递推**：

$$H_t = \sum_{i=1}^{t} \left(\prod_{j=i+1}^{t} \alpha_j\right)(1-\alpha_i) X_i$$

虽然每个 $\alpha$ 依赖于input，但一旦input给定，$\alpha$ 就确定了，整个序列的膜电位可以用一个**矩阵乘法**一次性算出来。

这就是Mamba、Linear Attention、RetNet那套"parallel scan"的思路——**SNN终于跟主流sequence modeling在数学上接上了。**

作者直接借用了 flash-linear-attention 库里的Triton kernel来加速。

---

## 为啥还能保持serial inference？

因为递推形式 $H_t = \alpha_t H_{t-1} + (1-\alpha_t) X_t$ 依然成立。推理的时候你就一步步算，只需要存 $H_{t-1}$ 和一个小的conv state，内存 $O(1)$。

**训练时用parallel form（快），推理时用serial form（省内存）**，两全其美。

---

## 效果怎么样？

**训练速度**：16k长度序列，比PSN快25.6倍。

**泛化能力**：用2k长度训练，30k长度推理还稳定。PSN超过训练长度直接崩。

**任务覆盖**：图像分类、事件相机、时序预测、语言建模、强化学习，全都SOTA或接近SOTA。

**能耗**：因为dynamic decay让spike发放更稀疏（该发才发，不该发就decay掉），总能耗比原始LIF还低一点。

---

## 这篇paper真正牛在哪？

**不是技术多花哨，是思考问题的层次更高。**

之前的人纠结"reset要不要保留""怎么保留""怎么近似"，都是在具体机制层面打转。

作者站高一层：**reset只是实现某些功能的手段，功能本身才是本质。把功能识别清楚，找一个更好的实现方式就行了。**

这个思路跟你以前讲"Transformer为什么替代RNN"是一个味道——不是RNN不好，是attention这种实现方式更好地满足了"长程依赖建模"这个功能需求。

---

## 一句话总结

**把SNN的reset换成input-dependent dynamic decay，训练快25倍，能外推，能耗更低，还跟Mamba/Linear Attention数学上同构。SNN终于有了scalable的并行训练范式。**

---

# Parallel Training in Spiking Neural Networks — 深度解读

Karpathy你好，这篇paper来自中科院自动化所Guoqi Li组，核心idea非常优雅。我把它拆成几个层次来build你的intuition。

---

## 1. 核心问题：Reset是SNN并行训练的"原罪"

SNN的主流神经元是 **LIF (Leaky Integrate-and-Fire)**，离散迭代形式如下（Eq.1-3）：

$$H_t = \beta V_{t-1} + (1-\beta) X_t \quad \text{(charging)}$$
$$S_t = \Theta(H_t - V_{\text{th}}) \quad \text{(firing)}$$
$$V_t = \begin{cases} H_t(1-S_t) + V_{\text{reset}} S_t & \text{hard reset} \\ H_t - V_{\text{th}} S_t & \text{soft reset} \end{cases}$$

变量解释：
- $H_t$：t时刻**充电后、reset前**的membrane potential
- $V_t$：t时刻**reset后**的membrane potential（给下一时刻用）
- $X_t \in \mathbb{R}^C$：t时刻的输入
- $\beta = 1 - 1/\tau_m$：decay factor，$\tau_m$是membrane time constant
- $S_t$：spike（0或1）
- $V_{\text{th}}$：发放阈值
- $\Theta(\cdot)$：Heaviside step function

**问题在哪？** 注意到 $V_t$ 依赖于 $S_t$，$S_t$ 依赖 $H_t$，$H_t$ 又依赖 $V_{t-1}$——这是典型的 **sequential recurrence**。GPU最讨厌这种串行依赖，BPTT (Backprop Through Time) 的时间和显存都是 O(T)，长序列（语言建模、长时序预测）根本训不动。

参考PSN原paper的对比：https://arxiv.org/abs/2309.13727

---

## 2. 作者的核心洞察：Functional View（功能性视角）

作者的关键insight是：**不要执着于"reset"这个具体机制，要识别它实现的功能**。Reset实际承担两个function：

### Function 1: Introducing Nonlinearity

Definition 3.1：若 $H_t = g(X_1, X_2, \ldots, X_t)$ 关于输入不是线性方程，则hidden state非线性。

去掉reset，Eq.1展开成纯粹的线性卷积：

$$H_t = \sum_{i=1}^{t} \beta^{t-i}(1-\beta) X_i$$

这只是一个exponential moving average，**纯线性滤波器**，表达力极弱。加上reset之后（hard reset举例）：

$$H_t = \beta(1 - f(H_{t-1})) H_{t-1} + (1-\beta) X_t$$

注意 $f(\cdot)$ 是firing function嵌进recursion，这种"output影响state"的反馈就是非线性的来源。

### Function 2: Controlling Membrane Potential

作者巧妙地形式化了两个概念：

**Definition 3.2 (∆-short control)**: 存在 $\Delta \in \mathbb{N}^+$，对任意 $t > \Delta$，若 $H_{t-\Delta} \geq V_{\text{th}}$ 且后续输入都小于 $V_{\text{th}}/\Delta$，则 $H_t < V_{\text{th}}$。

直觉：**大输入的影响在 $\Delta$ 步内被消化掉**，防止持续发放。

**Definition 3.3 (long control)**: 输入有上界 $C$，则 membrane potential 序列也有上界 $C_H$。

直觉：**有界输入不会让膜电位爆炸**。

Appendix A.1-A.4 给出了严格证明：
| | hard reset | soft reset |
|---|---|---|
| IF neuron | $\Delta$-short ✓ + long ✓ | 两者都 ✗ |
| LIF neuron | $\Delta$-short ✓ + long ✓ | $\Delta$-short ✗, long ✓ |

注意 soft reset 在 IF 上连 long control 都没有——这意味着有界输入也能让膜电位爆炸，需要decay来救命。

### Reset本身的局限

作者进一步argue reset并不是这两个function的最优实现：

- **Hard reset** 的 $\Delta \equiv 1$，无论输入多大都一刀切清零，**spatial discriminability差**（阈值以上的输入差异被抹平）
- **Soft reset** 减固定值 $V_{\text{th}}$，大输入要多个timestep消化，**temporal discriminability差**（spike持续依赖过去输入）

→ 这就给了作者motivation：**用一个更好的机制同时实现两个function**。

---

## 3. PTSI的三个Condition（理论骨架）

作者给出parallel training + serial inference兼容的三个structural condition，我觉得这是paper最漂亮的部分：

### Condition 1: Prefix Summarizability
$$\forall t \geq 1, \quad H_t = \phi_\theta(X_{1:t}), \quad S_t = g_{\theta'}(H_t)$$

输出只依赖prefix（causal），且参数 $\theta$ 是 **time-invariant** 的——所有timestep共享同一组参数。

### Condition 2: Online Updatability
$$\forall t \geq 1, \quad H_t = u_\theta(H_{t-1}, X_t)$$

可以**online递推**，每步只用 $(H_{t-1}, X_t)$ 就能算出 $H_t$。这保证了**serial inference**的能力。

### Condition 3: Offline Parallelizability
$$\exists p, \forall t \in [1, T], \quad H_t = p_\theta(X_{1:t})$$

给定固定长度窗口 $X_{1:T}$，存在一个**与递推顺序无关**的计算图（如conv / matmul / parallel scan）能并行算出全部 $H_{1:T}$。

**Table 1 的对比很说明问题**：

| Method | Func 1 | Func 2 | Cond 1 | Cond 2 | Cond 3 |
|---|---|---|---|---|---|
| LIF | ✓ | ✓ | ✓ | ✓ | ✗ |
| PSN | ✗ | ✗ | ✗ | ✗ | ✓ |
| Sliding PSN | ✗ | ✓ | ✓ | ✓ | ✓ |
| IPSU | ✓ | ✗ | ✗ | ✗ | ✓ |
| **DSN (Ours)** | ✓ | ✓ | ✓ | ✓ | ✓ |

LIF卡在Cond 3（无法并行）；PSN卡在Cond 1,2（参数与训练长度耦合，破坏causality，且参数量 $O(T^2)$）；只有DSN同时满足5个格子。

参考flash-linear-attention库对类似parallel form的实现：https://github.com/fla-org/flash-linear-attention

---

## 4. DSN的核心设计

### 4.1 Serial form

$$\mathbf{H}_t = \boldsymbol{\alpha}_t \odot \mathbf{H}_{t-1} + (1 - \boldsymbol{\alpha}_t) \odot \mathbf{X}_t \quad \text{(Eq.9)}$$
$$\mathbf{S}_t = \text{Clip}[\text{Round}(\mathbf{H}_t), 0, N] \quad \text{(Eq.10)}$$
$$\boldsymbol{\alpha}_t' = \text{CausalConv1D}(\mathbf{X}_{t-k+1:t}) \quad \text{(Eq.11)}$$
$$\boldsymbol{\alpha}_t = \text{Sigmoid}(\boldsymbol{\alpha}_t')^{1/\tau} \quad \text{(Eq.12)}$$

变量解释：
- $\mathbf{X}_t \in \mathbb{R}^{C \times 1}$：t时刻输入（$C$个channel）
- $\boldsymbol{\alpha}_t \in \mathbb{R}^{C \times 1}$：**dynamic decay**，每个channel一个值，由近期输入决定
- $k$：causal conv的kernel size（论文取 $k=4$）
- $\tau$：超参，控制 $\alpha_t$ 的"陡峭度"（论文取 $\tau=0.25$）
- $N$：integer spike的上限（论文取 $N=4$）
- $\text{Round}(\cdot)$：round to nearest integer
- $\odot$：element-wise product

### 4.2 关键insight：Dynamic decay如何替代reset

直觉：**reset是"硬性清零"，dynamic decay是"软性遗忘"**。$\alpha_t \to 0$ 等价于hard reset，$\alpha_t \to 1$ 等价于完全保留历史，$\alpha_t$ 取中间值就是部分遗忘——而且这个"遗忘程度"是由input自适应决定的。

**Proposition 4.1**（Appendix A.2证明）：dynamic decay可以同时实现nonlinearity和 $\Delta$-short / long control，且比reset更灵活。

证明的关键（Eq.A.17）：要保证 $\Delta$-short control，只需让 $\alpha$ 满足
$$\alpha_{t-\Delta+1} < \frac{V_{\text{th}} - X_{t-\Delta+1}}{H_{t-\Delta} - X_{t-\Delta+1}} \in (0, 1]$$

而且dynamic decay能generalize到任意 $\tau \in [1, \Delta]$ 的duration（Eq.A.21），这是reset做不到的——reset只能 $\Delta=1$（hard）或"无限"（soft）。

### 4.3 为什么用CausalConv而不是别的

Table 2 的ablation：
| Design | Accuracy |
|---|---|
| Causal Conv (论文选用) | 90.10% |
| Fully Connected | 89.28% |
| Low-rank mapping | 86.72% |
| Inter-channel conv | 86.76% |
| w/o conv (退化为static decay) | 84.53% |

Causal conv最简洁有效，参数少且捕获short-term dependency（Mamba和Griffin都用类似结构 https://arxiv.org/abs/2312.00752 , https://arxiv.org/abs/2402.19427）

### 4.4 Integer-valued spike firing

Eq.10 用 Clip+Round 替代 Heaviside。这里参考了作者组前作（https://arxiv.org/abs/2410.22644）的integer-spike训练技巧。

Table 3 显示：
| N | Surrogate | Accuracy |
|---|---|---|
| 4 | Rect | 90.10% |
| 3 | Rect | 89.56% |
| 2 | Rect | 89.28% |
| 1 | Rect | 86.74% |
| 1 | ATan | 87.45% |

$N=4$ 时性能最好，等价于把 $T=4$ 个binary timestep压缩到1步——既减少训练步数又增加表达力。

---

## 5. Parallel Form推导（最精彩的部分）

把Eq.9展开：
$$\mathbf{H}_t = \sum_{i=1}^{t} \left(\prod_{j=i+1}^{t} \boldsymbol{\alpha}_j\right)(1 - \boldsymbol{\alpha}_i) \odot \mathbf{X}_i \quad \text{(Eq.14)}$$

这是 **input-dependent coefficients 的 linear recurrence**，和 **Gated Linear Attention / Mamba** 的parallel form同构。

定义矩阵（Eq.15-16）：
$$\mathbf{W}_{ij} = \begin{cases} \left(\prod_{k=i+1}^j \boldsymbol{\alpha}_k\right)(1-\boldsymbol{\alpha}_i) & j \geq i \\ 0 & j < i \end{cases}$$

$$\mathbf{P}_j = \prod_{k=1}^j \boldsymbol{\alpha}_k \quad \text{(cumulative product, lower-triangular)}$$
$$\mathbf{A}_i = \boldsymbol{\alpha}_i$$
$$\mathbf{M}_{ij} = \mathbb{1}[j \geq i] \quad \text{(causal mask)}$$

最终parallel form（Eq.17）：
$$\mathbf{H} = \mathbf{X} \left( \left( \frac{\mathbf{1}-\mathbf{A}}{\mathbf{P}} \right)^T \mathbf{P} \odot \mathbf{M} \right)$$

直觉解释：
- $\mathbf{P}$ 是 $\alpha$ 的cumulative product，对应"历史衰减累积"
- $(1-\mathbf{A})/\mathbf{P}$ 是把每个位置的"该保留的输入比例"归一化
- 外积 $\mathbf{u}^T \mathbf{P}$ 形成 $T \times T$ 矩阵
- $\odot \mathbf{M}$ 强制causality

整个计算复杂度从 $O(T^2)$（PSN）降到 $O(T)$，且能用 **Triton kernel** 加速。作者用了 flash-linear-attention 库里的HGRN operator实现，参考：https://github.com/fla-org/flash-linear-attention , https://triton-lang.org/

参考Parallel Linear Recurrence原paper：https://arxiv.org/abs/1702.04695

---

## 6. 实验数据深度解析

### 6.1 Training Efficiency（Fig.5 left）

| Sequence length | PSN | Sliding PSN | DSN |
|---|---|---|---|
| 1k | baseline | similar | similar |
| 16k | 1× | ~12× faster than PSN | **25.6× faster than PSN** |

DSN在16k序列上：forward 21.7×加速，backward 28.0×加速。Sliding PSN只在forward快，但backward慢3.2×，所以总体DSN还是2.2×更快。

### 6.2 Extrapolation（Fig.5 right）

在WikiText-103上训练2k序列，测试1k-30k的PPL：

- **Masked PSN**: 2k以后直接崩溃（参数与训练长度绑定）
- **Sliding PSN**: 能到~10k但PPL上升
- **LIF**: 自然支持任意长度但PPL略高
- **DSN**: 30k依然稳定，且PPL最低

这验证了**Condition 1, 2的实际价值**——参数time-invariant + online update = 长序列泛化。

### 6.3 多Task SOTA

| Task | Dataset | Architecture | DSN vs 次优 |
|---|---|---|---|
| Image Classification | Sequential CIFAR10 | Conv SNN | 90.10% vs 88.45% (PSN) |
| Image Classification | Sequential CIFAR100 | Conv SNN | 64.70% vs 62.21% (PSN) |
| Image Classification | ImageNet | SEW ResNet18 | 68.21% vs 67.63% (PSN) |
| Event Processing | CIFAR10-DVS | VGGSNN | 85.30% vs 85.30% (Sliding PSN) |
| Time-series | Metr-la/Pems-bay/Solar | Spikformer | RSE↓ 0.566 vs 0.603 |
| Language Modeling | WikiText-103 | SpikingSSM | PPL 28.50 vs 32.25 |
| RL | Hopper-v4 | Spiking Actor | 3565 vs 3446 (MDC-SAN) |
| RL | Walker2d-v4 | Spiking Actor | 4436 vs 4340 (TD3 ANN!) |

注意Walker2d-v4上DSN(4436)甚至超过了ANN TD3(4340)——这是SNN很少能在连续控制上超过ANN的成就。

参考：Spikformer (https://arxiv.org/abs/2209.02076), SpikingSSM (https://arxiv.org/abs/2407.01222)

### 6.4 Energy Consumption（Table 9）

| Method | S-CIFAR10 SFR | Energy (mJ) | S-CIFAR100 SFR | Energy (mJ) |
|---|---|---|---|---|
| LIF | 0.1499 | 107.80 | 0.1697 | 121.78 |
| PSN | 0.2143 | 235.87 | 0.2226 | 242.03 |
| Sliding PSN | 0.1820 | 170.39 | 0.1900 | 176.22 |
| **DSN** | 0.1238 | **102.89** | 0.1324 | **108.94** |

DSN的spike firing rate **比LIF还低**（0.1238 vs 0.1499），这弥补了causal conv引入的额外FLOPs，总能耗甚至略低于LIF。这背后的intuition：dynamic decay让membrane potential更智能地自适应，避免了不必要的spike发放。

参考：SpikingJelly框架 https://github.com/fangwei123456/spikingjelly

---

## 7. Appendix的Approximation Experiment（很妙的设计）

作者在 Appendix B.7 做了一个验证dynamic decay表达力的实验：

构造6种LIF神经元（hard/soft reset × 3种 $\tau_m$），用DSN去拟合它们的membrane potential行为。

**Table B.15 结果**：
- 平均拟合精度 92.97% (Dataset A, normal noise) / 95.05% (Dataset B, structured signals)
- 用integer spike后提升到 98.46% / 98.32%

这证明了：**dynamic decay能够"模拟"reset机制的所有行为**，且integer-valued训练让拟合更精确（因为integer spike等价于多timestep的binary spike累积）。

---

## 8. 我对这篇paper的intuition总结

1. **抽象层次提升**：从"reset vs no-reset"的binary争论，上升到"function vs implementation"的functional view。这是真正research taste的体现——好的工作往往是在更高抽象层次上重新组织问题。

2. **与Linear RNN/State Space Model的统一**：DSN的parallel form与Mamba、RetNet、Gated Linear Attention本质上同构，都是 **input-dependent decay linear recurrence**。SNN社区终于和主流sequence modeling社区在数学上汇合了。这是SNN大规模化的关键。

3. **Causality是核心约束**：Condition 1 (Prefix Summarizability) 是作者特别强调的——参数必须time-invariant且causal。PSN/Masked PSN用 $T \times T$ 学习矩阵破坏了这个性质，所以不能extrapolate。

4. **Biological plausibility与computational efficiency的trade-off被重构**：传统观点认为reset是生物机制必须保留。作者证明reset的功能（nonlinearity + potential control）可以用更"软"的dynamic decay实现，且更灵活。这给neuromorphic hardware设计提供了新的方向。

5. **Integer-valued training的协同效应**：integer spike + dynamic decay 的组合在approximation experiment上表现互补——dynamic decay提供连续可微的"软控制"，integer spike提供离散的"硬编码"。这两者结合在Appendix B.7的拟合精度98%+上得到验证。

6. **未解决的开放问题**：
   - Dynamic decay引入的causal conv和Sigmoid在neuromorphic chip上还是较昂贵（作者在Section 5.3讨论了ternary weight和shift-based sigmoid的可能适配，参考 https://arxiv.org/abs/2402.17764）
   - Appendix B.7显示当 $\tau_m$ 增大（长程依赖增强）时拟合精度下降（85.87%），说明DSN对**超长程依赖**仍有提升空间
   - Causal conv的kernel size $k=4$ 是固定的，更long-range的input-dependency可能需要attention或更大kernel

参考相关的follow-up工作：
- Multiplication-free Parallelizable Spiking Neurons: https://arxiv.org/abs/2410.05175
- SpikingSSM: https://arxiv.org/abs/2407.01222
- P-SSM: https://openreview.net/forum?id=x6Yyv5jzJF

---

## 9. 与你早期工作的可能联想

你之前在https://karpathy.ai/zero-to-hero.html 讲过RNN的parallelization难题（BPTT显存爆炸）和Transformer如何用attention破解。DSN的故事是**SNN版本的"用parallel form替代sequential recurrence"**，但保留了SNN的event-driven/biological structure。如果未来要把SNN scaling law做出来，这篇paper是奠基性的——它给了SNN community一个和Mamba/RetNet同样 expressive 且 hardware-friendly 的 parallel primitive。

可以考虑的实验方向：
1. 把DSN放到更大模型（>100M参数）上做scaling law
2. 验证DSN在in-context learning上的能力（SNN的spike sparsity是否帮助reasoning）
3. Causal conv换成selective SSM（Mamba block）会怎样？
4. 在neuromorphic chip（如Loihi 2, Speck）上做实际deployment benchmark

希望这个解读对你build intuition有帮助！如果对某个部分想深入讨论（比如parallel form的Triton实现细节、Appendix A的证明trick、或者跟Mamba/linear attention的更精确对应关系），我可以再展开。
