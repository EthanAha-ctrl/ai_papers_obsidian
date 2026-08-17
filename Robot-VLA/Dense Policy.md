---
source_pdf: Dense Policy.pdf
paper_sha256: 6accd3d463a5724815d196ffd31518b3848732c95517e1c24a266453f54499e2
processed_at: '2026-08-03T19:51:11-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用人话来说，这篇 paper 的核心就是在讲：**怎么教机器人像人一样“打草稿”，然后慢慢“描细节”。**

为了 build your intuition，我们可以把机器人生成 action（动作轨迹）这件事，想象成画画。目前主流的画法有两种，但都有毛病：

1.  **Diffusion Policy（DP）**：就像是拿一块橡皮，从一团乱码噪声里一点点擦出整幅画。它每次都要同时处理整张画布（整个 horizon 的 action），算力消耗大，而且画出来的轨迹有时候比较僵硬。
2.  **Next-Token Prediction（ICRT 等）**：就像是从左到右、一个像素一个像素地画。这对语言文字很有效，因为语言有强因果逻辑。但 action 不一样，你下一步手要往哪移，不仅取决于前面怎么移，还取决于终点在哪。只看左边画右边，很容易画歪。

人类是怎么做事的？比如你要伸手拿杯子。你的大脑绝不会去规划“第一秒肱二头肌肉收缩 10%，第二秒收缩 15%”。你的大脑是先定两个**keyframes**（关键帧）：手抬起、手抓到杯子。然后再下意识地把中间的动作“补齐”。这就叫 **coarse-to-fine**（从粗到细）。

Dense Policy 就是把这个直觉写成了算法。

---

### 算法机制的技术直觉

Dense Policy 的网络架构极其简单，是一个只有 encoder 的 BERT。没有 decoder。为什么能用 encoder 做 generation？这正是这篇 paper 最巧妙的地方。

它通过一个叫 **Dense Process** 的递归循环来做“补齐”的动作。假设我们要生成未来 $T=16$ 步的 action sequence。

**Step 1: 起草最粗的 keyframes**
初始状态 $A^0 = \mathbf{0}$，就是一个全零的向量。把它丢进 BERT encoder，跟当前的 observation $O$ 做 cross-attention。网络吐出几个极度稀疏的 keyframe（比如就 2 个点，代表开头和结尾）。

**Step 2: Upsampling（线性插值补齐）**
有了 2 个点，怎么变出 4 个点？paper 里用了一个极度简单的公式（公式 4）。
本质上就是**取平均值做线性插值**。

来看看公式 4 的细节：
$$
\tilde{a}_{t+j}^n = 
\begin{cases} 
a_{t+j} & \text{if } j \mod \frac{T}{2^n} = 0 \\ 
\frac{1}{2}(a_{t+j - \frac{T}{2^{n+1}}} + a_{t+j + \frac{T}{2^{n+1}}}) & \text{if } j \mod \frac{T}{2^n} \neq 0 \\ 
a_{t+T - \frac{T}{2^n}} & \text{if } j = T - \frac{T}{2^{n+1}}
\end{cases}
$$
变量解释：
-   $n$：当前所在的层级。
-   $T$：总时间步数（比如 16）。
-   $j$：当前要计算的时间索引。
-   $a$：具体的 action 向量（比如末端执行器的 XYZ 位置和姿态）。

这个公式翻译成人话就是：
-   如果 $j$ 刚好落在上一层的 keyframe 上，直接保留原值不动。
-   如果 $j$ 落在两个 keyframe 中间，就把前后两个 keyframe 的 action 加起来除以 2，作为粗略的猜测。
-   如果 $j$ 是序列末尾的新增点，直接用最后一个 keyframe 的值来填充边界。

**Step 3: Cross-attention Refinement（看图修改）**
把刚才插值出来的、有点粗糙的 4 个点，再丢回 BERT encoder，让它再看一遍当前的 observation $O$。因为有了更密集的 action 作为 context，网络这时就可以对中间那些插值的点进行微调，让它们更符合物理环境。

**Step 4: 循环**
2 变 4，4 变 8，8 变 16。直到长度达到 horizon $T$。这就是公式 5 里说的 $A^{n+1} = \mathbf{Enc}(A_{up}^n, O)$。

这种“插值 -> 微调 -> 插值 -> 微调”的递归过程，把一个长度为 16 的序列生成，压缩成了 $\log_2(16) = 4$ 次网络 forward pass。这就是 paper 里吹的 **logarithmic time inference**。不仅快，而且因为每次插值都有上一层的结构做先验，生成的轨迹极其连贯。

---

### 为什么 Encoder-only 能 work？

传统做 generation 一定要用 decoder，因为 decoder 有 causal mask（只能看前面的 token），能保证自回归生成。BERT encoder 是双向的，所有 token 互相都能看见。

Dense Policy 用 encoder 能做生成，核心就在于它**跳出了“左到右”的生成逻辑**。
在它的每一层处理中，所有 action 位置（无论是 keyframe 还是插值出来的点）同时看彼此，同时看 observation，然后同时输出 refinement。这完美契合了 action 的物理特性：action 在时间上是 **bidirectional**（双向相关）的。中间的动作依赖于首尾，首尾的动作也受中间动作的物理约束。用双向 attention 去处理双向依赖，这非常 make sense。

---

### 实验数据的 intuition

为什么 Dense Policy 比 Diffusion Policy 好那么多？看实验数据表 1。

paper 做了一个非常公平的 test。把 Diffusion Policy 里的 action head 砍掉，换成 Dense Policy 的 action head，视觉 backbone（ResNet18 或 Sparse Conv）完全保持一致。这样可以完全隔离出 action head 的功效。

在 MetaWorld 的 **Shelf Place** 任务上，3D Dense Policy 达到了 77% 的成功率，而 DP3 只有可怜的 17%。相差 60%！
Shelf Place 是一个对物体位姿估计要求极高的任务。Diffusion Policy 在去噪过程中，很容易在某个局部极小值里陷进去，导致抓取位姿差了几毫米，任务直接失败。而 Dense Policy 一开始就规划了跨越整个轨迹的 keyframes，相当于它先“瞄准”了终点，然后再慢慢把中间动作对齐。这种全局视野让它对误差的容忍度极高。

在真实世界的 **Pour Balls（倒球）** 任务里（Table 2），Dense Policy 能把 10 个球全倒进去的成功率是 60%，而 RISE（SOTA 3D diffusion policy）只有 25%。
倒球是一个需要 6-DoF 极度平滑旋转的任务。Diffusion 生成的轨迹往往有微小的抖动，导致球在杯子里乱晃，最后倒歪了。Dense Policy 的 linear upsampling 机制保证了它的底色是极度平滑的，BERT 只是在平滑的基础上做微调。这种“先天平滑”的归纳偏置让它非常适合流体或软体操作。

---

### 我的一些发散联想

这篇 paper 的直觉让我联想到很多其他领域的东西：

1.  **类似于 B-spline 或 Catmull-Rom spline**
    传统计算机图形学里画曲线，就是先定几个 control points，然后算插值。Dense Policy 简直就是用神经网络强化过的 spline 插值。第一层 keyframes 是 control points，神经网络的作用是在给定 observation 的情况下，把 control points 摆对位置，并且对插值点做环境自适应的微调。

2.  **Chain of Thought (CoT) 在 action 领域的映射**
    大模型里 CoT 是把一个复杂的推理拆成几步。Dense Policy 这里的 keyframe 就像是 action 的 CoT。先想“手伸到哪”，再想“中间怎么伸”。层级结构天然蕴含了 planning 的过程。

3.  **Multi-modal Distribution 的致命伤**
    我必须指出 Dense Policy 可能存在的一个硬伤。L2 loss 加上 deterministic 的 encoder refinement，意味着它本质上是在做 condition expectation $E[A|O]$。
    如果一个任务有 multiple modes（比如遇到障碍物，你可以从左边绕，也可以从右边绕），Dense Policy 很可能会把这两个 mode 平均一下，结果就是直直地撞上障碍物。这就是 ACT（基于 VAE 的 CVAE）和 Diffusion Policy 为什么强大的原因——它们能建模 multi-modal distribution。paper 里没有讨论这一点，可能在他们的实验 task 里 multimodality 不严重，但如果是复杂的长视野导航或者避障，这可能是个大坑。

4.  **和 Consistency Models 的对比**
    最近 Image generation 领域很火的 Consistency Models 也是把 multi-step 的 diffusion 压缩到 1-step。Dense Policy 用 4 步搞定了 horizon=16 的生成。未来如果把 consistency model 的自一致性约束引入到 Dense Policy 的每一层级 refinement 中，也许能把这 4 步进一步压成 1 步，达到 ACT 的速度，同时保留规划的精度。

5.  **Scaling 到 VLA (Vision-Language-Action) 的潜力**
    paper 最后说要扩展到 general-purpose VLA。如果要把这个 action head 接到 LLM 后面，是非常顺滑的。LLM 输出 language token，最后几个 output position 作为初始的 $A^0$ latent，然后在这个 LLM backbone 上做 $\log_2 T$ 次 bidirectional attention 的 refinement。这相当于在 LLM 里内嵌了一个层级式的 policy planner。对比 $\pi_0$ 用 Flow Matching 做 action head，Dense Policy 的 action head 参数量极小（paper Figure 9 显示比 ACT 还少一半参数），极其适合 plug-and-play 接入大模型。

总结一下，这篇 paper 告诉我们：**别跟风用 next-token 硬套机器人控制，也别死磕 diffusion 的去噪。action 生成本质上是时空结构的补全，利用 coarse-to-fine 的双向插值，用最小的算力、最简单的 encoder，就能拿到最好的效果。**

相关参考 link：
*   [Dense Policy Project Page](https://selen-suyue.github.io/DpNet/)
*   [Diffusion Policy (对比 baseline)](https://diffusion-policy.cs.columbia.edu/)
*   [3D Diffusion Policy / DP3 (对比 baseline)](https://3d-diffusion-policy.github.io/)
*   [VAR: Visual Autoregressive Modeling (启发了 coarse-to-fine 思想)](https://arxiv.org/abs/2404.02905)
*   [RISE (real-world 3D baseline)](https://arxiv.org/abs/2402.10847)
*   [MAR: Masked Autoregressive (图像领域的 continuous AR)](https://arxiv.org/abs/2406.11838)
*   [OpenVLA (未来可能的整合方向)](https://openvla.github.io/)

---

# Dense Policy 深度解析

## 核心问题与动机

这篇 paper 来自 SJTU 的团队，包括 Lixin Yang 和 Cewu Lu 等人。他们想解决一个非常 fundamental 的问题：autoregressive policy 在 robot manipulation 上为何一直打不过 diffusion policy 这类 holistic generation 方法？这是一个很有意思的现象，因为 NLP 领域 autoregressive 是主流，但 robot action 生成却相反。

paper 提到的核心痛点：
- Action space 是 continuous 的且 high-dimensional，不像 token 那样有 discrete codebook
- Robot demonstration 数据 sparse，比 language corpus 小得多
- Next-token prediction 在 action 上难以捕捉 long-term dependencies，因为 action 之间是 **bidirectional** 相关的（后面的 action 会约束前面的，反之亦然）
- CARP 借鉴 VAR 用 multi-scale VQ-VAE，但 discrete codebook 破坏了 action 的精度

直觉来源：人类操作时并不是 step-by-step 规划，而是先想象几个 keyframes 跨越整个 task execution，然后再 refine 中间过程——这就像视觉中的 receptive field，是 coarse-to-fine 的过程。

参考 link：[Project page](https://selen-suyue.github.io/DspNet/)

---

## 方法架构解析

### Problem Formulation

设 observation $O_t$，需要预测 horizon $T$ 上的 action sequence：
$$A_{t:t+T} = \{a_t, a_{t+1}, \dots, a_{t+T-1}\}$$

其中 $a_t$ 是 end-effector 的 TCP pose。

Dense Policy 的核心思想是将 $A$ 分解为一个层次结构 $A^1, A^2, \dots, A^{\log_2 T}$：

**公式 (1) 详解**：
$$A^n = \{a_{t+i}^n \mid i \mod \frac{T}{2^n} = 0, \; i \in \mathbb{N}_{<T}\}$$

变量含义：
- $A^n$：第 $n$ 层级（level）的 sparse action sequence
- $a_{t+i}^n$：在 time $t+i$ 处第 $n$ 层的 action 表示
- $T$：total horizon 步数
- $n$：level 索引，从 1 到 $\log_2 T$
- $i \mod \frac{T}{2^n} = 0$：每 $\frac{T}{2^n}$ 步采样一个 keyframe

举例：若 $T = 16$，则
- $A^1$：2 个 keyframe（间隔 8）
- $A^2$：4 个 keyframe（间隔 4）
- $A^3$：8 个 keyframe（间隔 2）
- $A^4 = A$：16 个 keyframe（完整序列）

注意 $A^0 = \emptyset$，初始化为零向量 $A^0 = 0$，提供 unbiased starting point。

**公式 (2) 详解**：
$$P(A|O) = \prod_{i=1}^{n} P(A^i | A^{i-1}, A^{i-2}, \dots, A^0, O)$$

这是一个 chain-rule factorization，每一层的生成 conditioned on 所有之前的层和 observation。这本质上是 multi-scale coarse-to-fine 的 autoregressive factorization，和 VAR (Visual Autoregressive Modeling, Tian et al. NeurIPS 2024) 思想类似，但应用在 continuous action space 上，没有 VQ 离散化。

参考：[VAR paper](https://arxiv.org/abs/2404.02905)

### Dense Process

每一层之间的 transition 经历 "Dense Process"：

**公式 (3) 和 (4) - Upsampling**：

upsampled action sequence：
$$A_{up}^n = \{\tilde{a}_{t+j}^n \mid j \mod \frac{T}{2^{n+1}} = 0, \; j \in \mathbb{N}_{<T}\}$$

具体每个 $\tilde{a}_{t+j}^n$ 的生成有三种情况：

$$\tilde{a}_{t+j}^n = 
\begin{cases} 
a_{t+j} & \text{if } j \mod \frac{T}{2^n} = 0 \\ 
\frac{1}{2}(a_{t+j - \frac{T}{2^{n+1}}} + a_{t+j + \frac{T}{2^{n+1}}}) & \text{if } j \mod \frac{T}{2^n} \neq 0 \\ 
a_{t+T - \frac{T}{2^n}} & \text{if } j = T - \frac{T}{2^{n+1}}
\end{cases}
$$

变量含义详解：
- $j$：时间索引
- $\frac{T}{2^n}$：当前层的 sampling interval
- $\frac{T}{2^{n+1}}$：当前层到下一层之间新增的 sample interval（更细粒度）
- 第一种情况：$j$ 落在 keyframe 上，直接保留上一层的结果
- 第二种情况：$j$ 落在两个 keyframe 中间，取**前后最近 keyframes 的算术平均**作为插值
- 第三种情况：边界处理，最后一个新增点用最后一个 keyframe 的值填充（避免越界）

这个 upsampling 实际上就是 linear interpolation，非常简单高效，但提供了 coarse prior。

**公式 (5) - Cross-attention refinement**：
$$A^{n+1} = \text{Enc}(A_{up}^n, O)$$

用 4 层 BERT encoder 对 upsampled actions 和 observation features 做 cross-attention refinement。注意是 encoder-only 架构，没有 decoder，这很关键——因为 bidirectional attention 允许所有 action positions 之间互相 attend，捕捉 bidirectional dependencies。

### Observation Encoder

- 2D：ResNet18 + GroupNorm（默认，跟 Diffusion Policy 一致）
- 3D：Sparse convolutional network（Minkowski，跟 DP3/RISE 一致）
- Proprioception：MLP，训练时随机 mask 部分 end-effector pose 防止 memorization

---

## Inference 复杂度对比

这是 paper 一个重要 selling point。让我详细分析：

| Method | Generation Paradigm | Complexity |
|--------|---------------------|------------|
| ACT | Single-step variational inference | $O(1)$ steps，但 single step 内含 CVAE sampling |
| Diffusion Policy | $K$ diffusion steps | $O(K)$，通常 $K=10\sim20$ |
| ICRT (Next-Token) | 逐 token 生成 | $O(T)$ linear |
| ARP (Next-Chunk) | 分 chunk 生成 | $O(T/C)$ linear |
| CARP (VAR-style) | Multi-scale residual | $O(\log T)$ 但每层做 VQ lookup |
| Dense Policy | Bidirectional expansion | $O(\log T)$，$\log_2 T$ 次递归 |

对于 $T=16$，Dense Policy 只需 4 次递归，每次内含 4 层 encoder，所以 inference step 数极少。

---

## 实验数据深度分析

### Simulation 结果（Table 1）

11 个任务覆盖 3 个 benchmark：

| Method | Adroit (Door) | Adroit (Pen) | DexArt (Laptop) | DexArt (Toilet) | MetaWorld (Bin Picking) | MetaWorld (Box Close) | MetaWorld (Hammer) | MetaWorld (Peg Insert Side) | MetaWorld (Disassemble) | MetaWorld (Shelf Place) | MetaWorld (Reach) | **Avg** |
|--------|--------------|--------------|------------------|------------------|-------------------------|------------------------|---------------------|------------------------------|----------------------------|----------------------------|---------------------|--------|
| 3D Dense Policy | **72±3** | **61±0** | 85±4 | **74±3** | **47±10** | **69±8** | **100±0** | **82±4** | **98±1** | **77±4** | **31±3** | **72±4** |
| DP3 | 62±4 | 43±6 | **81±2** | 71±3 | 34±30 | 42±3 | 76±4 | 69±7 | 69±4 | 17±10 | 24±1 | 53±7 |
| 2D Dense Policy | **59±8** | **65±1** | 28±7 | **36±8** | **25±2** | **51±3** | **86±4** | **60±7** | **71±6** | **59±6** | **27±4** | **52±5** |
| DP | 37±21 | 13±2 | **31±4** | 26±8 | 15±4 | 30±5 | 15±6 | 34±7 | 43±7 | 11±3 | 18±2 | 25±5 |

**关键观察**：
- 3D Dense Policy 平均 72%，比 DP3 (53%) 高 **19 个百分点**
- 2D Dense Policy 平均 52%，比 DP (25%) 高 **27 个百分点**
- Pen task 上 3D Dense Policy (61) vs DP3 (43)，差距 18%——这是 high-DoF dexterous manipulation
- Peg Insert Side（contact-rich）上 3D Dense Policy (82) vs DP3 (69)
- Laptop (DexArt) 上 DP3 反而略高（81 vs 85），可能因为 DexArt 任务对 3D 几何更敏感而非时序一致性

参考 DP3: [3D Diffusion Policy](https://3d-diffusion-policy.github.io/)  
参考 DP: [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)

### Real-World 实验（Table 2）

4 个 task，涵盖 soft-body / articulated / 6-DoF / long-horizon：

| Method | Put Bread Succ (%) | Open Drawer Succ (%) | Pour Balls Poured (%) | Pour Balls Balls | Pour Balls Complete (%) | Flower Arr. Succ (%) | Flower Arr. Flowers |
|--------|--------------------|------------------------|------------------------|--------------------|---------------------------|------------------------|------------------------|
| 3D Dense Policy | **85** | **45** | 85 | **7.30/10** | **60** | **70** | **1.0/3.0** |
| RISE | 75 | 40 | **95** | 6.85/10 | 25 | 50 | 0.6/3.0 |
| 2D Dense Policy | 55 | 20 | 35 | 3.30/10 | 25 | — | — |
| DP | 40 | 20 | 30 | 2.35/10 | 20 | — | — |
| ACT | 35 | 10 | 30 | 2.75/10 | 20 | — | — |

**亮点**：
- Put Bread into Pot 上 3D Dense Policy 比 RISE 高 10%，体现对 soft-body deformation 的鲁棒性
- Flower Arrangement 上 3D Dense Policy 比 RISE 高 20%——long-horizon multi-object manipulation 的优势尤其明显
- Pour Balls 上 Dense Policy "Complete" metric (60%) 远超 RISE (25%)，说明 smooth fluid motion 减少了 pouring deviation
- 2D 整体表现都比 3D 差很多，尤其在 Flower Arrangement 上完全无法完成——印证 2D 缺乏 complex spatial reasoning

参考 RISE: [RISE paper](https://arxiv.org/abs/2402.10847)

### Ablation：Bidirectional vs Unidirectional

paper 在 4 个 challenging tasks (Door, Bin Picking, Shelf Place, Box Close) 上对比了：
- Bidirectional (Dense Policy)
- Unidirectional Next-Token
- Unidirectional Next-Chunk (chunk size = 2)

结果显示 bidirectional 在 learning efficiency 和最终 ceiling 上都明显更优。这印证了 paper 的核心 hypothesis：action 之间的 bidirectional dependency 是 next-token prediction 难以捕捉的。

### Efficiency 分析

**Training 效率**：
- ACT 训练 unstable（CVAE 的 ELBO 优化难）
- DP 比 Dense Policy 收敛慢
- Dense Policy 训练 1000 epochs；ACT 需要 2000 epochs

**Inference 与参数量**（Figure 9）：
- ACT 和 Dense Policy 推理速度相当
- Dense Policy 比 ACT 参数量少一半以上
- Dense Policy 比 DP 多 9.19M 参数，但 inference 快 10 倍
- 优势来源：logarithmic recursion ($\log_2 T$ 次而非 $T$ 或 $K$ 次)

---

## 直觉构建与关联思考

### 1. 与 Human Motor Control 的类比

paper 引用 Botvinick (2008) 和 Wolpert & Ghahramani (2000) 的 cognitive science 工作，提到人类运动控制是 hierarchical 的——prefrontal cortex 构建 abstract action plans，basal ganglia 和 motor cortex 逐步 refine。Dense Policy 的 coarse-to-fine 结构很好地映射了这个理论。

### 2. 与 Visual Autoregressive Modeling (VAR) 的关系

Dense Policy 明显受 VAR (Tian et al. NeurIPS 2024) 启发，但有重要差异：
- VAR 用 VQ-VAE 先把 image 编成 multi-scale discrete tokens，然后 next-scale prediction
- Dense Policy **不**做离散化，直接在 continuous action space 上做 hierarchical expansion
- VAR 的 residual prediction 在 image 上 work，但 action 需要更高精度，discretization 误差不可接受

### 3. 与 MAR (Masked Autoregressive) 的关系

MAR (Li et al. 2024) 在 image 上用 diffusion loss + continuous representation + random mask order。Dense Policy 没用 mask prediction，而是用 bidirectional expansion，这是 action 模态的特殊需求——actions 有 temporal causality，不能完全 random order。

参考 MAR: [MAR paper](https://arxiv.org/abs/2406.11838)

### 4. 与 BERT 的关系

paper 用 BERT encoder 而非 GPT-style decoder-only，这是关键设计选择：
- BERT 的 bidirectional self-attention 自然支持 "看到前后 action" 的需求
- 这与 NLP 中 BERT 比 GPT 在 understanding 任务上更强，但 generation 上更难相对应
- Dense Policy 通过 hierarchical upsampling 解决了 BERT 难以做 generation 的问题——每一层做 in-place refinement，不依赖 causal mask

### 5. 推测的局限性

paper 自己提到未探索 VLA 大模型 scaling，可能的问题：
- Logarithmic recursion 假设 horizon 是 2 的幂次，非 2 的幂次需要 padding 或 truncation
- Upsampling 用 linear interpolation，可能不适合 action 速度变化的 task（比如快速到慢速过渡）
- BERT encoder 的 $O(L^2)$ attention 对超长 horizon 不友好
- 与 VLM backbone（如 OpenVLA, $\pi_0$）的整合尚未验证，可能存在 representation mismatch

参考 OpenVLA: [OpenVLA](https://openvla.github.io/)  
参考 $\pi_0$: [π0 paper](https://arxiv.org/abs/2410.24164)

### 6. 与 Diffusion Policy 的对比直觉

Diffusion Policy 是从 noise 开始 iterative denoise，每一步都是同维度的 refinement。Dense Policy 是从 1 frame 开始 iterative expansion，每一步增加维度并 refinement。这是两种完全不同的 "iterative generation" 范式：
- Diffusion: 在 fixed-dimension latent 上做 denoising trajectory
- Dense: 在 growing-dimension action sequence 上做 coarse-to-fine hierarchy

Diffusion 的每一步提供 global correction（整个序列都在变），但 step 之间没 explicit hierarchy；Dense 的每一步只 expand 到 next granularity，且 previous level 作为 strong prior，可能提供更好的 conditioning signal。

### 7. 与 Diffusion for Image Generation 的类比

DDPM 在 image 上的成功靠的是 $O(1000)$ steps；近期 Consistency Models / Rectified Flow 把这个降到 1-4 steps。Dense Policy 的 $\log_2 T$ steps 提供了一个介于 1-step 和 1000-step 之间的中间路线，可能 future work 可以借鉴 consistency model 思路进一步压缩到 1-step generation。

### 8. Inverse Dynamics 视角

考虑 action 生成本质上是在学 inverse dynamics $P(A | O)$，Dense Policy 的 factorization 可以看作在学：
$$P(A | O) = P(A^1 | O) \cdot P(A^2 | A^1, O) \cdot P(A^4 | A^2, A^1, O) \cdots$$

每一层都是 conditional inverse dynamics 在不同 temporal resolution 上的 instance。这和 hierarchical RL 中的 options / MAXQ framework 思想相通，但用 supervised learning 实现，避免了 RL 的 exploration 难题。

参考 MAXQ: [Dietterich 2000](https://link.springer.com/article/10.1023/A:1007670830556)

---

## 我的 Critical Thoughts

1. **Linear interpolation 的局限性**：upsampling 用 simple averaging 可能在 sharp transitions（比如 gripper close/open）处产生不一致的中间 action，未来可以考虑 learnable upsampling（如 transposed conv 或 learnable spline）。

2. **对 horizon 的 hard assumption**：$\log_2 T$ 必须是整数，否则需要非对称 expansion，paper 没讨论这种情况。

3. **Action representation 单一性**：paper 只用 TCP pose，没考虑 joint torques 或 finger actions（Adroit Shadow hand 用了 multi-fingered，但似乎 action 还是 end-effector parametric）。对于 full-body humanoid 这种 action 维度远超 TCP 的场景，Dense Policy 是否仍然有效未知。

4. **Multi-modal action distribution**：Diffusion Policy 在 multi-modal action distribution 上有天然优势（可以采样不同 mode），Dense Policy 是 deterministic refinement + L2 loss，可能 struggle 在 multi-modal demonstration 上。这是 ACT 也有的问题，paper 没明确讨论。

5. **Cross-attention 的必要性**：paper 用 BERT encoder 同时处理 observation 和 action，cross-attention 实现可能比 concatenation 更高效，但也限制了 action refinement 的灵活性——因为 observation 是 fixed signal，每层都 attend 同样的 $O$ 似乎冗余。或许 early layers 学 abstract geometric relation，late layers 学 fine-grained spatial adjustment。

6. **与 LLM 范式整合的潜力**：paper 提到未来想 extend 到 VLA，一个自然思路是把 BERT encoder 替换成 causal LM decoder，让 language tokens 和 action tokens 在同一个 transformer 里 coarse-to-fine 联合生成。这能和 $\pi_0$、OpenVLA 等 VLA 范式整合，前景诱人。

---

## 总结

Dense Policy 提出了一个 elegant 的 autoregressive paradigm for action generation：用 bidirectional encoder-only architecture + hierarchical coarse-to-fine expansion + continuous action space (no VQ)。它在 11 个 simulation task 和 4 个 real-world task 上都打败了 diffusion-based 和 ACT-based baselines，同时 inference 快 10 倍、参数少一半。这给 robot learning 社区提供了一个 strong baseline，也给 VLA 大模型的 action head 设计提供了新的候选方案。

关键 intuition takeaways：
1. **Bidirectional > Unidirectional for actions**：因为 action 序列天然有 bidirectional dependency
2. **Coarse-to-fine 是 hierarchical prior 的强形式**：比 VQ residual 更适合 continuous action
3. **Encoder-only 可以做 generation**：通过 in-place refinement 而非 causal decoding
4. **Logarithmic complexity 而非 linear**：是 next-scale prediction 的天然优势

这是一篇 methodologically clean、experimentally thorough、intuitively satisfying 的工作，值得关注后续的 scaling 与 VLA 整合。

相关参考：
- [Dense Policy Project Page](https://selen-suyue.github.io/DspNet/)
- [VAR: Visual Autoregressive Modeling](https://arxiv.org/abs/2404.02905)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [3D Diffusion Policy](https://3d-diffusion-policy.github.io/)
- [RISE: 3D Perception for Robot Imitation](https://arxiv.org/abs/2402.10847)
- [ACT: Bimanual Manipulation](https://tonyzhaozh.github.io/aloha/)
- [MAR: Masked Autoregressive](https://arxiv.org/abs/2406.11838)
- [OpenVLA](https://openvla.github.io/)
- [π0 by Physical Intelligence](https://arxiv.org/abs/2410.24164)
- [ICRT: In-Context Imitation via Next-Token](https://arxiv.org/abs/2408.15980)
- [CARP: Coarse-to-Fine Autoregressive Prediction](https://arxiv.org/abs/2412.06782)
- [BERT](https://arxiv.org/abs/1810.04805)
- [DexArt Benchmark](https://dexart.github.io/)
- [MetaWorld](https://meta-world.github.io/)
- [Adroit](https://arxiv.org/abs/1709.10087)
