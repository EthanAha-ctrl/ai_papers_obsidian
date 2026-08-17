---
source_pdf: Causal Forcing Autoregressive Diffusion Distillation Done Right for.pdf
paper_sha256: 2d5631d346f2b1e5a957ee027db8cf30c915ba93ae0a2ae329d2913465b9e1e3
processed_at: '2026-08-03T15:13:23-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话版本

你想把一个"一次性看全部frames才能生成视频"的慢模型，蒸馏成一个"逐帧生成、可实时交互"的快模型。现有SOTA方法（Self Forcing）的做法里藏着一个理论bug——**teacher选错了**，导致student学出来的是模糊平均而不是真实分布。修复方法简单得令人意外：**换一个AR的teacher就好了**。

---

## 故事怎么来的

### 场景设定

你手里有个Wan2.1这种bidirectional video diffusion model，质量很高，但一次生成整个视频要103秒，0.78 FPS。你想让它变成实时（17 FPS, 0.69s latency），还能让用户边看边干预后续frames。

这就要做两件事：
1. 把multi-step变成few-step（采样步数从几十步降到4步）
2. 把bidirectional attention改成causal attention（只能看过去frames，不能偷看future）

合起来叫**asymmetric distillation**：用bidirectional多步teacher蒸馏出AR少步student。

### 现有SOTA（Self Forcing）怎么做的

两阶段pipeline：
- **Stage 1 (ODE distillation)**: 用bidirectional teacher跑PF-ODE采样一堆(noisy, clean)对，让AR student学习从noisy映射到clean
- **Stage 2 (DMD)**: 用score distillation进一步打磨分布

听起来没毛病。但实验上，即使这样折腾完，AR student还是显著不如直接DMD蒸馏的bidirectional student（Fig. 1）。这中间差的那部分performance，作者称为**architectural gap**——"把full attention换成causal attention"这个动作本身带来的loss。

### 作者的第一个关键实验：DMD救不了architectural gap

作者做了一个clever的controlled experiment（Fig. 2）：
- 先用standard DMD把bidirectional teacher蒸馏成一个few-step bidirectional model（这一步消除了"多步到少步"的gap）
- 再给它套上causal mask变成AR model（这一步只引入architectural gap，没有sampling-step gap）

结果：性能依然远不如standard DMD。

**这证明什么**：architectural gap在DMD阶段根本补不回来，必须在更早的ODE initialization阶段就处理好。

### 作者的第二个关键发现：injectivity被破坏

这是全文最核心的insight，我用一个比喻讲。

**比喻**：假设你是个学生（AR student），老师（bidirectional teacher）给你出题。题目形式是"给你一张noisy frame $x_t^i$，告诉你它前面的frames，请还原出clean frame $x_0^i$"。

**正常情况（bidirectional teacher + bidirectional student）**：老师给你整个视频的noisy版本$x_t^{1:N}$，你还原整个clean视频。每个noisy视频对应唯一clean视频，一对一映射，学起来没问题。

**Self Forcing的情况（bidirectional teacher + AR student）**：老师还是用bidirectional方式生成(noisy, clean)对，但学生只能看到当前frame的noisy版$x_t^i$和过去frames。问题是：**同一个$x_t^i$，在不同的future frames $x_t^{>i}$配合下，可以对应完全不同的clean $x_0^i$**。

因为bidirectional model在denoise第$i$个frame时，会利用所有frames的信息（包括未来的）。固定当前frame，改变未来frames，当前frame的denoise结果就变了。

所以学生看到"同一个输入对应多个不同答案"，MSE loss下它会学什么？**学平均值**。这就是为什么Self Forcing出来的视频发糊——它在做regression to the mean。

这个"一个输入对应多个输出"的问题，数学上叫**违反injectivity**。在AR setting下，要求的是**frame-level injectivity**：每个noisy frame（加上历史）必须映射到唯一clean frame。

作者严格证明了（Lemma 3.2, Proposition 3.3）：用bidirectional teacher蒸馏AR student，**必然违反frame-level injectivity**，且collapse到conditional mean。

### 解决方案：换teacher

既然问题出在teacher不是AR的，那就**先训练一个AR的teacher**，再用它做ODE distillation。

但这里又冒出一个subtle问题：训练AR teacher有两种范式——

**Teacher Forcing (TF)**：训练时第$i$个frame attend到**干净的**历史frames $x_0^{<i}$
**Diffusion Forcing (DF)**：训练时第$i$个frame attend到**加噪的**历史frames $x_t^{<i}$

业界一直觉得DF更"高级"（Chen et al., 2024的Diffusion Forcing paper是NeurIPS 2024的spotlight），因为能独立给每frame加不同noise level，看起来更灵活。

但作者发现：**DF用在AR training上有train-inference mismatch**。训练时conditioning on noisy prefix，推理时conditioning on clean prefix（因为前一帧已经生成干净了），两个distribution不一样。理论证明（Proposition 3.4）这个KL divergence严格大于0。

实验验证（Tab. 2）：TF的VisionReward是3.343，DF只有1.583，差一倍。DF的Dynamic Degree看起来高（60 vs 50），但那是video collapse导致motion metric虚高。

所以AR teacher要用TF训练。

### 完整pipeline

三阶段（叫Causal Forcing）：
1. **Stage 1**: 用TF训练一个AR diffusion model（2K steps on 3K synthetic data from Wan2.1）
2. **Stage 2**: 用这个AR teacher采样PF-ODE轨迹，做causal ODE distillation初始化student（1K steps）
3. **Stage 3**: 用asymmetric DMD进一步优化（750 steps on Vid-ProM）

关键：Stage 2的teacher是AR的，它的PF-ODE天然满足frame-level injectivity（因为它本身只用$x_t^i$和prefix，不碰future），所以student能正确学到flow map而不是conditional mean。

---

## 为什么这个工作有价值

### 1. 诊断精准

很多distillation paper发现"AR蒸馏效果差"就把锅甩给"架构差异"这种模糊概念，然后用各种engineering trick补救。这篇paper第一次从理论层面定位到根因：**injectivity violation**。这个根因是可证明的（Lemma 3.2用measure theory严格证明），不是empirical observation。

### 2. 解决方案简洁

没有新架构、没有新loss、没有新optimizer。只是把teacher从bidirectional换成AR的，pipeline其他部分完全不动。这种"找到正确问题，方案自然浮现"的工作很优雅。

### 3. 有一系列controlled experiment佐证

- Fig. 2隔离sampling-step gap，证明DMD补不了architectural gap
- Fig. 9隔离initialization：用AR teacher的paired data但student从bidirectional model初始化，结果一样好。证明问题在paired data而不在initialization
- Tab. 4对比multi-step AR直接初始化 vs causal ODE初始化：multi-step AR在few-step下还有额外mismatch（conditioning的prefix质量退化），所以causal ODE是必要的

这种"每一步都cleanly isolate一个变量"的实验设计是顶配水准。

### 4. 揭示了一个被忽视的mismatch

**Diffusion Forcing用在AR training是错的**——这个结论反直觉。Diffusion Forcing是NeurIPS spotlight工作，被广泛引用。但作者证明它只在bidirectional continuation场景下正确（continuation时把clean tail和noise concat，这与训练匹配）。把DF的"conditioning on noisy prefix"照搬到AR training，就有train-inference gap。这澄清了社区一个常见的误解。

---

## 我的intuition与几个延展思考

### Regression to the mean是diffusion distillation的核心敌人

这个paper让我重新理解了一件事：diffusion model本质上是在建模**score function**（log probability density的gradient），而不是直接回归data。这是因为multi-modal分布下，直接回归必然collapse到mean。

ODE distillation想做的事是把multi-step采样压缩成few-step，本质是学习从任意noisy state直接跳到clean endpoint的mapping。这个mapping必须injective，否则student又会退化成回归mean。

所以injectivity不是ODE distillation的"额外要求"，而是**它的基本存在条件**。Self Forcing违反这个条件还能跑，纯粹是因为DMD阶段部分补救了，但补救不彻底。

### 这个insight能推广到哪

Frame-level injectivity本质说的是：**distillation的teacher和student必须有匹配的信息流**。Teacher用到的信息student也要能用，否则injectivity破坏。

推广一下：
- 蒸馏有因果约束的student（只能看过去），teacher也必须有因果约束
- 蒸馏稀疏attention的student，teacher也应该是稀疏attention的（或者至少信息流要匹配）
- 蒸馏量化model，teacher也应该是量化的（否则precision mismatch破坏injectivity）

这给未来的distillation工作提供了一个普适的设计原则。

### Causal CD还有很大空间

作者承认causal CD只是vanilla LCM的简单实现（VisionReward 1.798 vs causal ODE+DMD的6.326）。但理论上CD和ODE distillation是等价的（ODE distillation是CD的简化版）。用sCM (Lu & Song, 2024, https://arxiv.org/abs/2410.11081)或Mean Flows (Geng et al., 2025, https://arxiv.org/abs/2505.13447)做causal extension，可能把CD的performance提上来，而且CD不需要大规模ODE paired data generation，省存储省时间。这是个明确的future direction。

### Chunk-wise vs frame-wise

Frame-wise setting下Self Forcing完全崩溃（Dynamic Degree只有2），chunk-wise下还能用（Dynamic 24）。原因是chunk-wise每chunk 3 frames内部有bidirectional interaction，某种程度上缓解了non-injectivity。但Causal Forcing在两种setting下都强（frame-wise Dynamic 64, chunk-wise 68）。这说明Causal Forcing的方法论是"原则上正确"的，不依赖chunk-size的trick。

---

## 几个值得关注的reference

- **Self Forcing** (https://arxiv.org/abs/2506.08009): 被本文"打脸"的SOTA baseline
- **CausVid** (https://arxiv.org/abs/2504.20332): 另一个AR distillation工作，同样有non-injectivity问题
- **Diffusion Forcing** (https://arxiv.org/abs/2407.01392): NeurIPS 2024 spotlight，本文证明其AR usage是suboptimal的
- **Consistency Models** (https://arxiv.org/abs/2303.00948): ODE distillation的理论基础
- **Rectified Flow** (https://arxiv.org/abs/2209.03003): Flow matching的injectivity相关理论
- **Mean Flows** (https://arxiv.org/abs/2505.13447): future work可能用来改进causal CD
- **sCM** (https://arxiv.org/abs/2410.11081): 同上
- **Project page**: https://thuml.github.io/CausalForcing.github.io/

---

## 总结成给Karpathy听的一句话

这篇paper的intellectual core是：**asymmetric distillation里，teacher和student的信息流必须匹配，否则injectivity破坏，student退化成conditional mean**。表现上就是bidirectional teacher蒸馏AR student必然产生模糊视频，修复方法是用AR teacher（通过TF训练）采样PF-ODE做ODE initialization。理论+controlled experiment+最终performance三重证据链很完整，是distillation领域一个clean的"诊断+处方"工作。

---

# Causal Forcing: Autoregressive Diffusion Distillation 深度解析

## 1. 总览：这篇paper在解决什么问题

这是一篇来自Tsinghua University的工作，核心贡献是修复了autoregressive (AR) video diffusion distillation中一个被忽视的理论缺陷。

**场景**: 你有一个pretrained bidirectional video diffusion model（如Wan2.1），它一次能生成整段视频但很慢。你希望把它distill成few-step AR model，实现real-time interactive video generation。

**现状**: SOTA方法如Self Forcing (Huang et al., 2025a, https://arxiv.org/abs/2506.08009) 和CausVid (Yin et al., 2025, https://arxiv.org/abs/2504.20332)采用两阶段pipeline：
1. ODE distillation初始化AR student
2. DMD (Distribution Matching Distillation, Yin et al., 2024, https://arxiv.org/abs/2310.13367)进一步优化

但即使如此，AR distilled model仍然显著落后于标准DMD蒸馏的bidirectional student（见Fig. 1）。

**本文发现**: 现有方法在ODE distillation阶段违反了一个关键条件——**frame-level injectivity**，导致student学到的是conditional expectation而非真实flow map，产生blurry videos。本文提出Causal Forcing：用AR teacher（而非bidirectional teacher）做ODE distillation。

**结果**: 在相同throughput (17 FPS)和latency (0.69s)下，相比Self Forcing：
- Dynamic Degree +19.3% (57→68)
- VisionReward +8.7% (5.820→6.326)
- Instruction Following +16.7% (48→56)

---

## 2. 背景知识：理解distillation pipeline的几个building blocks

### 2.1 Diffusion models与PF-ODE

Diffusion models通过forward process $x_t = \alpha_t x_0 + \sigma_t \epsilon$ 扰动数据，其中：
- $x_0 \sim p_{data}(x_0)$ 是clean data
- $t \in [0, T]$ 是diffusion timestep
- $\alpha_t, \sigma_t$ 是noise schedule
- $\epsilon \sim \mathcal{N}(0, I)$ 是Gaussian noise

在flow matching参数化（Lipman et al., 2022, https://arxiv.org/abs/2210.02727）下，采用$\alpha_t = 1-t, \sigma_t = t, T=1$，velocity为：
$$v_t := \frac{dx_t}{dt} = \epsilon - x_0$$

Sampling通过求解PF-ODE（probability flow ODE, Song et al., 2020, https://arxiv.org/abs/2011.13456）：
$$dx_t = v_\theta(x_t, t)dt, \quad x_T \sim \mathcal{N}(0, I), \quad t: T \to 0 \quad (Eq. 1)$$

**Intuition**: PF-ODE定义了一个deterministic mapping，从noise $x_T$ 到data $x_0$。这个mapping的flow map $\phi: (x_t, t) \mapsto x_0$ 是后续distillation的核心。

### 2.2 Autoregressive video diffusion

Full-sequence diffusion models一次生成所有frames，无法interactive。AR video diffusion采用frame-wise factorization：
$$p_\theta(x_0^{1:N}) = \prod_{i=1}^N p_\theta(x_0^i | x_0^{<i})$$

其中上标$i$表示frame index，$x_0^{<i} = (x_0^1, \ldots, x_0^{i-1})$是历史frames。

两种训练范式：
- **Teacher Forcing (TF)**: 第$i$个noisy frame $x_t^i$ attend到clean prefix $x_0^{<i}$
- **Diffusion Forcing (DF, Chen et al., 2024, https://arxiv.org/abs/2407.01392)**: $x_t^i$ attend到noisy prefix $x_t^{<i}$，每frame独立加噪

实践上，TF通过concatenating clean video和noisy counterpart + causal attention mask实现。

### 2.3 ODE distillation (Consistency Distillation的简化版)

Consistency Distillation (CD, Song et al., 2023, https://arxiv.org/abs/2303.00948)学习flow map $G_\theta: (x_t, t) \mapsto x_0$，boundary condition $G_\theta(x, 0) \equiv x$。Loss：
$$\mathbb{E}_{x_0, \epsilon, t}[w(t) d(G_\theta(x_t, t), G_{\theta^-}(\hat{x}_{t-\Delta t}, t-\Delta t))]$$

其中$\hat{x}_{t-\Delta t}$通过teacher求解一步ODE得到，$\theta^-$是stop-gradient的running average。

本文采用的简化variant直接回归clean target：
$$\theta^* = \min_\theta \mathbb{E}_{t, x_t}[\|G_\theta(x_t, t) - x_0\|^2]$$

要求$(x_t, x_0)$在同一PF-ODE轨迹上。

### 2.4 DMD (Distribution Matching Distillation)

DMD通过最小化student分布$p_\theta(\tilde{x})$与data分布的KL divergence实现distillation。梯度（Eq. 2）：
$$\nabla_\theta \mathbb{E}_t[D_{KL}(p_{\theta,t} \| p_{data,t})] = -\mathbb{E}_{\tilde{x}, t, \tilde{x}_t}[(s_{real}(\tilde{x}_t, t) - s_{fake}(\tilde{x}_t, t))\frac{\partial \tilde{x}}{\partial \theta}]$$

- $s_{real}$: frozen model预测data分布的score
- $s_{fake}$: online trainable model预测student分布的score
- $\tilde{x}_t \sim q_{t|0}(\tilde{x}_t | \tilde{x})$: 加噪的student sample

---

## 3. 问题诊断：Architectural Gap的根源

### 3.1 现象：DMD无法弥补architectural gap

作者做了一个controlled experiment（Fig. 2）：用standard DMD初始化AR student（即先用DMD蒸馏一个bidirectional few-step model，再施加causal mask）。这消除了sampling-step gap，只剩architectural gap。结果显示性能仍远不如standard DMD。

**结论**: architectural gap不能在DMD阶段解决，必须在更早的ODE initialization阶段处理。

### 3.2 核心理论：Frame-level injectivity

这是本文最关键的insight。

**Injectivity requirement for ODE distillation**: MSE regression要well-defined，paired data必须injective——每个noisy sample对应唯一clean sample（Liu et al., 2022, https://arxiv.org/abs/2209.03003）。

对于bidirectional teacher蒸馏bidirectional student，injectivity在video level天然成立：
$$x_0^{1:N} = \phi^{Bi}(x_t^{1:N}, t)$$
其中$\phi^{Bi}$是bidirectional model的flow map。任意noisy video $x_t^{1:N}$对应唯一clean video。

但对于AR student，injectivity要求shift到frame level：

**Definition 3.1 (Frame-level injectivity)**: 对于AR flow map $\phi^{AR}: (x_t^i, t) \mapsto x_0^i$，frame-level injectivity成立当：
$$\forall t \in (0, 1], \forall \{x_t^j\}_{j=1}^N, \{y_t^j\}_{j=1}^N: \forall i \in [N], x_t^i = y_t^i \Rightarrow \phi^{AR}(x_t^i, t) = \phi^{AR}(y_t^i, t) \quad (Eq. 4)$$

即给定noisy frame $x_t^i$，clean frame $x_0^i$唯一确定。

### 3.3 Self Forcing为什么违反frame-level injectivity

**Lemma 3.2 (Frame-level non-injectivity of bidirectional PF-ODE)**: 

设$x_t^{1:N}$满足bidirectional model的PF-ODE，$x_t^i$是第$i$个frame，$x_t^{other} := x_t^{[N]\backslash\{i\}}$是其余frames。如果$\phi^{Bi}(x_t^{1:N}, t)^i$关于$x_t^{other}$不是a.e. constant（这个假设由DiT attention的non-constancy保证，见Xi et al., 2025, https://arxiv.org/abs/2502.01776; Zhao et al., 2025d），则：
$$\forall t \in (0,1], \forall x_t^{1:N}, \exists y_t^{1:N}: y_t^i = x_t^i \text{ and } \phi^{Bi}(x_t^{1:N}, t)^i \neq \phi^{Bi}(y_t^{1:N}, t)^i \quad (Eq. 6)$$

且$\mathbb{P}(\text{Var}(\phi^{Bi}(x_t^{1:N}, t)^i | x_t^i, t) > 0) > 0$。

**Intuition**: Bidirectional model用所有frames（包括future frames$x_t^{>i}$）去denoise第$i$个frame。固定$x_t^i$，不同的$x_t^{>i}$会产生不同的$x_0^i$。但AR student在推理时没有$x_t^{>i}$，因此信息丢失，injectivity被破坏。

**Proposition 3.3 (Distribution mismatch)**: 在这种情况下，MSE regression的最优解是conditional mean（Bishop, 2006）：
$$G_\theta^*(x_t^i, t) = \mathbb{E}[x_0^i | x_t^i, t] \approx p_{data}(x_0^i) \quad (Eq. 7)$$

**关键观察**: 这个conditional mean **不等于**真实的flow map，也不等于data分布。从$L^2$正交投影：
$$\mathbb{E}\|Y\|^2 = \mathbb{E}\|\hat{Y}\|^2 + \mathbb{E}\|Y - \hat{Y}\|^2 = \mathbb{E}\|\hat{Y}\|^2 + \mathbb{E}[\text{Var}(Y|U, t)]$$

由Lemma B.1，$\mathbb{E}[\text{Var}(Y|U, t)] > 0$，所以$\mathbb{E}\|\hat{Y}\|^2 < \mathbb{E}\|Y\|^2$，即conditional mean的分布与data分布不同。视觉上表现为blurring。

---

## 4. 方法：Causal Forcing三阶段

### 4.1 Stage 1: Teacher Forcing AR diffusion training

作者发现一个反直觉的结果：**TF优于DF用于AR diffusion training**，无论理论上还是实验上。

**Proposition 3.4 (Distribution mismatch in DF)**: 
$$\mathbb{E}_{y \sim p_{data}(x_0^{<i})}[D_{KL}(p_{DF}(x_0^i | y) \| p_{data}(x_0^i | y))] > 0$$

证明思路（Appendix B.2）：
- 设$Y := x_0^{<i}$, $X := x_0^i$, $Z := x_t^{<i}$（对$Y$独立加噪）
- Markov chain: $X \to Y \to Z$ under $p_{data}$
- DF训练使$p_{DF}(x|z) = p_{data}(x|z)$
- 推理时query clean prefix $y$，得$p_{DF}(x|y) = p_{data}(x|Z=y)$
- 用反证法：若$D_{KL} = 0$则$p_{data}(X|Z=y) = p_{data}(X|Y=y)$ a.e.
- 由tower property: $p_{data}(X \in A | Z=y) = \mathbb{E}[f_A(Y)|Z=y]$，其中$f_A(y) := P(X \in A | Y=y)$
- 在regularity conditions (A3)下，conditional expectation operator的不动点必为constant
- 推出$X \perp Y$，与假设(A2)矛盾

**Intuition**: DF训练时conditioning on noisy prefix $x_t^{<i}$，但推理时conditioning on clean prefix $x_0^{<i}$，存在train-inference distribution mismatch。TF训练和推理都conditioning on clean prefix，gap为0。

实验对比（Tab. 2）：
- TF: VisionReward 3.343
- DF: VisionReward 1.583（-111.2%）

### 4.2 Stage 2: Causal ODE distillation

用Stage 1的AR diffusion model作为teacher，sample PF-ODE轨迹$\{x_t^i\}_{t \in S \cup \{0\}}$。

**采样过程**:
1. 从real dataset采样clean prefix $x_{gt}^{<i}$
2. 从Gaussian noise $x_T^i \sim \mathcal{N}(0, I)$开始
3. 用AR teacher condition on $x_{gt}^{<i}$，ODE solver求解轨迹

**训练目标**（Eq. 8）：
$$\theta^* = \min_\theta \mathbb{E}_{x_{gt}^{<i}, t \in S, i}[\|G_\theta(x_t^i, x_{gt}^{<i}, t) - x_0^i\|^2]$$

- $x_{gt}^{<i}$: 来自real dataset的clean prefix（teacher forcing）
- $S$: predefined timesteps set，论文用$\{1, 0.9375, 0.8333, 0.625\}$
- $i$: uniform采样frame index

**关键点**: AR teacher的PF-ODE天然满足frame-level injectivity，因为AR模型本身只用$x_t^i$和prefix，对每个frame独立denoise。这避免了Proposition 3.3的collapse。

### 4.3 Stage 3: Asymmetric DMD

用Stage 2的causal ODE-distilled model初始化asymmetric DMD，protocol完全follow Self Forcing：
- $s_{real}$: Wan2.1-14B（frozen）
- $s_{fake}$: Wan2.1-1.3B（online trainable）

### 4.4 Extension: Causal Consistency Models

本文还提出causal CD（Eq. 9）：
$$\theta^* = \min_\theta \mathbb{E}_{x_{gt}, \epsilon, t, i}[w(t) d(G_\theta(x_t^i, x_{gt}^{<i}, t), G_{\theta^-}(\hat{x}_{t-\Delta t}^i, x_{gt}^{<i}, t-\Delta t))]$$

其中$\hat{x}_{t-\Delta t}^i$通过AR teacher求解ODE一步得到。

采用LCM (Luo et al., 2023a, https://arxiv.org/abs/2310.04378) setting，48 discretized timesteps，UniPC solver，EMA rate 0.99。

对于flow matching的v-prediction参数化，boundary condition $G_\theta(x^i, x_{gt}^{<i}, 0) \equiv x^i$自动满足（Eq. 31）：
$$G_\theta(x^i, x_{gt}^{<i}, t) = x^i - t v_\theta(x^i, x_{gt}^{<i}, t)$$

因为$t=0$时$G_\theta(x^i, x_{gt}^{<i}, 0) = x^i - 0 = x^i$。

---

## 5. 实验结果详解

### 5.1 主实验（Table 1）

| Model | Throughput | Latency | Total | Quality | Semantic | Dynamic | VisionReward | Instruct | Rating |
|-------|------------|---------|-------|---------|----------|---------|--------------|----------|--------|
| Wan2.1-1.3B (bidirectional) | 0.78 | 103 | 83.37 | 84.30 | 79.65 | 61 | 5.275 | 42 | 2.29 |
| Self Forcing | 17.0 | 0.69 | 83.74 | 84.48 | 80.77 | 57 | 5.820 | 48 | 2.87 |
| **Causal Forcing (Ours)** | **17.0** | **0.69** | **84.04** | **84.59** | **81.84** | **68** | **6.326** | **56** | **1.64** |

关键观察：
- 相比Wan2.1 bidirectional，throughput提升2079%（0.78→17.0 FPS），且质量持平甚至略优
- 相比Self Forcing，所有指标提升，Rating从2.87降到1.64（越低越好）
- 相比其他AR models（NOVA, Pyramid Flow, SkyReels-V2, MAGI-1），大幅领先

### 5.2 Ablation studies（Table 2）

**AR training策略**:
- DF: VisionReward 1.583, Dynamic 60（高但病态，因为collapse放大motion metric）
- TF: VisionReward 3.343, Dynamic 50

**ODE initialization + DMD (chunk-wise)**:
- Self Forcing's ODE + DMD: VisionReward 3.330, Dynamic 24, Instruct 38
- Causal ODE + DMD: VisionReward 6.326, Dynamic 68, Instruct 56
- 改进：VisionReward +90.0%, Dynamic +183.3%, Instruct +47.4%

**ODE initialization + DMD (frame-wise)**:
- Self Forcing's ODE + DMD: VisionReward 1.951, Dynamic 2
- Causal ODE + DMD: VisionReward 6.204, Dynamic 64
- 改进：VisionReward +218.0%, Dynamic +3100%（frame-wise setting下gap更显著）

**CD对比**:
- Asymmetric CD: VisionReward -7.983, Instruct -42（极差）
- Causal CD: VisionReward 1.798, Instruct 18

### 5.3 Initialization不是bottleneck（Appendix C.3, Fig. 9）

作者做了一个精巧的controlled experiment：用AR teacher生成paired data $\mathcal{D}_{Causal}$，但student从bidirectional model初始化。结果与从AR model初始化相当，都远好于Self Forcing's ODE distillation。

**结论**: 性能gap主要由paired data construction决定，而非student initialization。这强化了frame-level injectivity的理论分析。

### 5.4 为什么multi-step AR diffusion不能直接用作DMD initialization（Appendix C.2, Fig. 7-8, Tab. 4）

Multi-step TF AR diffusion model + DMD: VisionReward 5.863, Dynamic 66
Causal ODE + DMD: VisionReward 6.326, Dynamic 68

Multi-step AR model在multi-step sampling下能较好narrow gap，但few-step sampling下存在额外mismatch：第$i$个frame condition的prefix $x_0^{<i-1}$在few-step下质量退化，与training时condition的high-quality ground-truth prefix不符。Error accumulate across chunks，造成inter-frame abrupt transitions（Fig. 7 top）。

Causal ODE-distilled model在few-step下更稳定（Fig. 7 bottom），是更合适的DMD initialization。

---

## 6. 我的Intuition building与思考

### 6.1 为什么frame-level injectivity如此关键

考虑一个极端例子：fixed noisy frame $x_t^i$，对应两个不同clean frames $x_0^i$ 和 $y_0^i$（分别由两个不同future $x_t^{>i}$和$y_t^{>i}$决定）。MSE loss的最优解是$(x_0^i + y_0^i)/2$——一个平均化的blurry frame。

这其实是**regression to the mean**现象（Bishop, 2006），在multi-modal conditional distribution下MSE必然collapse to mean。Diffusion model之所以能生成sharp samples，是因为它建模score function而非直接回归mean。ODE distillation把multi-step过程压缩到few-step，必须保证每个intermediate state对应唯一endpoint，否则就退化成回归问题。

### 6.2 与consistency model理论的关系

Consistency models (Song et al., 2023)的boundary condition $G_\theta(x, 0) \equiv x$隐含了ODE轨迹上的自洽性：同一轨迹上任意点都映射到同一endpoint。这本质上就是injectivity。

本文的frame-level injectivity是把这个概念从sample level推广到frame level。在AR setting下，每个frame独立的ODE轨迹是必要条件，否则cross-frame信息污染injectivity。

### 6.3 为什么Self Forcing的作者没发现这个问题

Self Forcing的pipeline是：
1. Bidirectional teacher采样PF-ODE轨迹
2. AR student学习从noisy intermediates回归clean video
3. DMD进一步优化

直觉上这看起来合理——毕竟bidirectional teacher质量更高。但问题在于：bidirectional teacher的ODE轨迹是**video-level injective**，每个$(x_t^{1:N}, t)$对应唯一$x_0^{1:N}$。但AR student只接收$x_t^i$和prefix，丢弃了$x_t^{>i}$的信息。这个information loss导致同一个$(x_t^i, \text{prefix})$对应多个$x_0^i$。

作者用controlled experiment（Fig. 9）证明：即使student从bidirectional model初始化，只要用AR teacher采样paired data，性能就恢复。这证明问题不在初始化，而在paired data的injectivity。

### 6.4 与历史工作的联系

**Diffusion Forcing (Chen et al., 2024)**原本是为long-video generation设计的：continuation时concatenate clean prefix（视频尾部）和noise，这与训练匹配。本文指出，把DF用于AR training（conditioning on noisy prefix）是misapplication，因为推理时conditioning on clean prefix。

**Mean Flows (Geng et al., 2025, https://arxiv.org/abs/2505.13447)** 和 **sCM (Lu & Song, 2024, https://arxiv.org/abs/2410.11081)** 改进了consistency model的训练。本文的causal CD是vanilla LCM的简单instantiation，作者承认"rudimentary"，留待future work。

**Score Distillation (Wang et al., 2023, https://arxiv.org/abs/2209.14988)** 和 **Diff-Instruct (Luo et al., 2023b)** 是DMD的precursor。DMD通过两个score model ($s_{real}$, $s_{fake}$)的差分估计KL gradient，避免了mode collapse。

### 6.5 Architectural gap与future directions

本文解决了一个specific的architectural gap——bidirectional→causal在distillation中的问题。但更广义的architectural gap问题在其他场景也存在：
- **Sparse attention → full attention**: 稀疏attention model蒸馏
- **MoE → dense**: mixture-of-experts到dense model
- **Quantized → full precision**: 量化模型蒸馏

每个gap可能都有自己的"injectivity-like"必要条件。这个工作提供的方法论：先分析optimal regression solution的性质，检查它是否匹配teacher的flow map，可能具有普适意义。

### 6.6 实验设计的亮点

1. **Controlled experiments**: Fig. 2隔离sampling-step gap，Fig. 9隔离initialization，Tab. 4隔离multi-step vs few-step。每个ablation都clean地isolate一个变量。

2. **公平比较**: 所有方法用相同的3K ODE initialization steps，相同prompts，相同dataset size。作者特别强调"$\mathcal{D}_{Bi}$和$\mathcal{D}_{Causal}$都是内部synthetic data用相同prompts，data quality一致"。

3. **多维度evaluation**: VBench (Huang et al., 2024, https://arxiv.org/abs/2311.17935), VisionReward (Xu et al., 2024, https://arxiv.org/abs/2412.21059), Dynamic Degree, Instruction Following, user study。特别考虑motion-rich prompts（100-prompt custom set）。

### 6.7 局限性与潜在问题

1. **Causal CD仍然weak**: Tab. 2显示causal CD (VisionReward 1.798)远不如causal ODE + DMD (6.326)。作者承认这是vanilla LCM的简单实现，未来用sCM/Mean Flows可能改进。

2. **Frame-wise vs chunk-wise**: Frame-wise setting下Self Forcing崩溃（Dynamic 2），但chunk-wise下还能用（Dynamic 24）。这说明chunk-wise某种程度上缓解了non-injectivity（每chunk 3 frames内部有interaction）。本文method在两种setting下都强。

3. **Scale**: 基于1.3B Wan2.1，81 frames，832×480。Scaling到更大模型和更长视频是否保持优势未验证。

4. **理论假设**: Lemma 3.2的"not a.e. constant"假设依赖DiT attention的non-constancy，这是empirical observation而非定理。其他架构（如Mamba-based）可能不同。

---

## 7. 关键references

- Self Forcing: https://arxiv.org/abs/2506.08009
- CausVid: https://arxiv.org/abs/2504.20332
- DMD: https://arxiv.org/abs/2310.13367
- Consistency Models: https://arxiv.org/abs/2303.00948
- Flow Matching: https://arxiv.org/abs/2210.02727
- Rectified Flow: https://arxiv.org/abs/2209.03003
- Wan2.1: https://arxiv.org/abs/2503.20314
- Diffusion Forcing: https://arxiv.org/abs/2407.01392
- DiT: https://arxiv.org/abs/2212.09748
- Score Distillation (SDS): https://arxiv.org/abs/2209.14988
- VBench: https://arxiv.org/abs/2311.17935
- VisionReward: https://arxiv.org/abs/2412.21059
- Mean Flows: https://arxiv.org/abs/2505.13447
- sCM: https://arxiv.org/abs/2410.11081
- LCM: https://arxiv.org/abs/2310.04378
- PF-ODE (Score SDE): https://arxiv.org/abs/2011.13456
- Bishop PRML: 经典regression to mean理论
- Project page: https://thuml.github.io/CausalForcing.github.io/

---

## 8. 总结

这篇paper的核心insight是：**distillation的paired data必须满足injectivity，否则MSE regression会collapse to conditional mean**。在AR video setting下，这意味着需要frame-level injectivity，而bidirectional teacher天然不满足。解决方案简单优雅：用AR teacher（通过teacher forcing训练）采样PF-ODE轨迹。

这个工作的价值在于：
1. **诊断精准**: 用controlled experiment和理论分析定位到architectural gap的根源
2. **方案简洁**: 不需要复杂新架构，只是改变teacher选择
3. **理论扎实**: Lemma 3.2, Proposition 3.3, 3.4都有严格证明
4. **实验全面**: 主实验、ablation、controlled experiments俱全
5. **启发深远**: injectivity原则可能推广到其他architectural gap场景

对real-time interactive video generation领域，这提供了一个clean baseline，未来工作可以在更强的CD variants (sCM, Mean Flows)上做causal extension，可能进一步突破性能。
