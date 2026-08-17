---
source_pdf: Continuous Thought Machines.pdf
paper_sha256: 671fd295f6362b3910d5def7e78d5eb08f8a99d3866c4808f79a26c685dec171
processed_at: '2026-08-03T17:25:34-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话拆解 Continuous Thought Machines (CTM)

如果用一句话来概括这篇 paper 的核心：CTM 把现代神经网络里那种简单粗暴的“瞬间激活”神经元，换成了有记忆、有节奏、能互相“对暗号”的动态神经元，让 AI 可以像人一样“多想几秒”再给出答案。

下面为你把 CTM 的核心机制和直觉彻底拆开。

## 1. 为什么要搞 CTM？现在的 AI 缺了什么？

现在的深度学习模型（比如 Transformer、CNN）处理信息基本都是“一锤子买卖”。看一张图，前馈网络一层层传到底，直接吐出结果。就算有 residual connection，神经元本身也只是一个静态的数学函数 $y = \sigma(Wx+b)$。这跟人类大脑完全不同。

生物学大脑的神经元是有时间动态的，会震荡，会记住前几毫秒的输入。大脑在处理复杂问题时，会通过神经元的同步放电来“绑定”不同特征，并且会“想一会儿”（System 2 慢思考）。现在的 DL 模型为了 scalability 和 GPU 友好，把这些 temporal dynamics 全抽象掉了。CTM 的目的就是把“时间动态”和“神经元同步”重新放回神经网络的核心位置。

## 2. 核心创新 1：给每个神经元发个“私人小本本”

以前的神经元，看到输入就立刻输出，看完就忘。CTM 给每个神经元发了一个专属的 MLP（多层感知机），还给它配了一个长度为 $M$ 的记忆小本本。

### 技术细节与公式

每个神经元 $d$ 不再是一个简单的激活函数，它变成了一个有私有权重 $\theta_d$ 的微型网络。它会处理自己过去 $M$ 个 tick 的输入历史 $\mathbf{A}_d^t$。

公式 (3)：
$$\mathbf{z}_d^{t + 1} = g_{\theta_d}(\mathbf{A}_d^t)$$

变量解释：
- $\mathbf{z}_d^{t+1}$：神经元 $d$ 在下一个内部时间步 $t+1$ 的激活值。
- $\mathbf{A}_d^t \in \mathbb{R}^M$：该神经元过去 $M$ 个时间步的输入历史记录。
- $g_{\theta_d}$：神经元 $d$ 专属的私有 MLP（Neuron-Level Model, NLM）。
- $\theta_d$：只属于神经元 $d$ 的私有权重参数。

### 直觉构建

因为每个神经元的 MLP 权重不一样，在训练过程中，有的神经元可能学会了做“累加器”，有的学会了做“震荡器”，有的学会了做“微分器”。这就产生了极其丰富的 neural dynamics。传统 RNN（如 LSTM）里的 gate 是所有神经元共享同一个 sigmoid 函数，CTM 彻底打破了这个限制，让神经元从 scalar function 升级成了 dynamic system。

## 3. 核心创新 2：用“对暗号”来作为主表达

传统网络用“哪些神经元亮了”（activation vector，维度是 $D$）来表示信息。CTM 用“哪两个神经元在一段时间内一起亮”（synchronization matrix，维度是 $D \times D$）来表示信息。

### 技术细节与公式

CTM 计算两个神经元历史激活值的内积，并引入了可学习的指数衰减率，让模型自己决定该记住多久以前的事。

公式 (10)：
$$\mathbf{S}_{ij}^t = \frac{\sum_{\tau=1}^t e^{-r_{ij}(t-\tau)} z_i^\tau z_j^\tau}{\sqrt{\sum_{\tau=1}^t e^{-r_{ij}(t-\tau)}}}$$

变量解释：
- $\mathbf{S}_{ij}^t$：神经元 $i$ 和 $j$ 在时间 $t$ 的同步度。
- $z_i^\tau, z_j^\tau$：神经元 $i$ 和 $j$ 在历史时间步 $\tau$ 的激活值。
- $r_{ij} \geq 0$：属于这对神经元的私有学习参数，控制遗忘的速度。
- $e^{-r_{ij}(t-\tau)}$：时间衰减系数，越早的激活 $\tau$ 对当前 $t$ 的影响越小。
- 分母：归一化项，保证同步度 magnitude 稳定。

### 直觉构建

这相当于把 representation space 从 $D$ 维放大到了 $D^2$ 维。如果 $D=4096$，表达空间大了几千倍。更重要的是，它编码的是“时间上的相关性”。生物视觉里，处理边缘的神经元和处理颜色的神经元同步放电，你才知道这是一个完整的物体（Binding problem 的解法）。传统 NN 只能靠 high-dimensional vector 的 magic 硬挤，CTM 直接把 temporal correlation 作为 first-class citizen 拿出来用。

## 4. 机制设计：给 AI 留出“发呆思考”的时间

CTM 解耦了“输入数据的时间”和“大脑内部思考的时间”。就算输入是一张静态图片，CTM 内部也会自己跑 $T$ 个 tick 的循环。在每个 tick 里，它通过 synchronization 算出一个 query，去图片上看一眼（attention），然后把看到的信息和内部状态混合，进入下一个 tick。

这就是 test-time compute scaling 的一种原生实现。简单图片它想 5 个 tick 确定了就停，复杂图片它想 40 个 tick。

## 5. 如何实现“想到哪停到哪”？绝妙的 Loss 设计

没有专门加一个 halting module 去判断该不该停（像 PonderNet 那样），CTM 靠一个非常简单的 loss function 实现了 native adaptive compute。

### 技术细节与公式

公式 (11)：
$$L = \frac{\mathcal{L}^{t_1} + \mathcal{L}^{t_2}}{2}$$

变量解释：
- $t_1 = \arg\min(\mathcal{L})$：整个思考过程中，cross-entropy loss 最低的那个 tick。
- $t_2 = \arg\max(\mathcal{C})$：整个思考过程中，模型最“确信”的那个 tick。Certainty $\mathcal{C}$ 定义为 $1 - \text{normalized entropy}$。
- $\mathcal{L}^{t_1}, \mathcal{L}^{t_2}$：分别在 $t_1$ 和 $t_2$ 时刻计算的 loss。

### 直觉构建

这个 loss 告诉模型：“我不逼你非要在最后一个 tick 给出答案，你在哪个 tick 觉得最有把握、loss 最小，我就用那个 tick 的结果算梯度。” 这就自动实现了 adaptive compute，模型会自我对齐 certainty 和 correctness，简单数据在 early tick 就能出结果并主导 gradient，困难数据会一直演化到 late tick。

## 6. 工程奇迹：如何让 $D^2$ 的计算不爆炸？

算几百万个神经元对的内积，算 50 个 tick，naive 的计算复杂度是 $\mathcal{O}(D^2 t)$，GPU 直接冒烟。Sakana AI 用了一个 rank-1 的递归公式解决了这个问题。

### 技术细节与公式

公式 (16) & (17)：
$$\alpha_{ij}^{t+1} = e^{-r_{ij}} \cdot \alpha_{ij}^t + z_i^{t+1} z_j^{t+1}$$
$$\beta_{ij}^{t+1} = e^{-r_{ij}} \cdot \beta_{ij}^t + 1$$

变量解释：
- $\alpha_{ij}^t$：带衰减的历史内积和（公式 10 的分子）。
- $\beta_{ij}^t$：衰减系数的累加和（公式 10 的分母平方项）。
- $z_i^{t+1}, z_j^{t+1}$：当前 $t+1$ 时刻两个神经元的激活值。

### 直觉构建

在 $t+1$ 步，根本不需要重新把过去所有的历史拿出来算内积。只需要把上一时刻的 $\alpha_{ij}^t$ 乘以衰减系数 $e^{-r_{ij}}$，加上当前这一步的内积 $z_i^{t+1} z_j^{t+1}$ 就行了。这就是一阶递归。每个 tick 只需 $\mathcal{O}(1)$ 的更新，整体降到 $\mathcal{O}(D_{\text{sub}})$，这才是 CTM 能跑起来的关键，让它既 biologically plausible 又 computationally tractable。

## 7. 三个惊艳的实验现象

### 7.1 2D Mazes (迷宫寻路)
实验故意不给 positional encoding，逼模型自己建地图。CTM 学会了先在脑子里顺着路“看”过去（Episodic future thinking），然后再输出动作。LSTM 在长 tick 下直接崩溃，CTM 稳如老狗，甚至能泛化到比训练集大得多的 99x99 迷宫。

### 7.2 ImageNet-1K (看图分类)
模型自己学会了“四处乱看”。没有任何 loss 教它扫描图片，但在 50 个 tick 里，它的 attention head 会自然地在图片上移动，寻找线索，像极了人类的扫视。并且画出的 UMAP 图里出现了大脑皮层里那种 traveling waves（行波）。

### 7.3 Parity (奇偶校验)
64 位长度的 01 序列算前缀奇偶。CTM 学到了两种可解释的算法：有的 random seed 训出来是从前往后扫，有的是从后往前扫。模型在内部自己发明并执行了排序算法，展示了真正的 algorithmic reasoning。

## 8. 发散联想

1. **NLMs 与 KAN 的共鸣**：Kolmogorov-Arnold Networks (KAN) 把权重换成可学习的 splines，CTM 把 activation function 换成可学习的 NLM。两者都在微观尺度上增加 model 的 expressivity，这是对抗 Scaling Law 瓶颈的另一个维度。
2. **Synchronization 是天然的 World Model 载体**：LeCun 一直推 JEPA，强调预测未来的 latent space。Synchronization 通过内积捕获 temporal correlation，本质上就是一个高阶的 temporal kernel，这可能是构建 world model 的理想底层结构。
3. **LLM 的 System 2 缺失**：如果把 NLMs 和 Synchronization 放到 LLM 里，token 预测可能就变成一种有节律的推理过程。目前的 Quiet-STaR 只是隐式地藏了 rationale，CTM 则是显式地在 internal ticks 里循环展开。未来的 long-horizon agent 可能极度依赖这种 native 的 temporal dynamics。

## Reference Links

- [Sakana AI - Continuous Thought Machines Official Blog](https://sakana.ai/ctm/)
- [Universal Transformer (Dehghani et al.)](https://arxiv.org/abs/1807.03819)
- [PonderNet (Banino et al.)](https://arxiv.org/abs/2107.05407)
- [Adaptive Computation Time (Graves)](https://arxiv.org/abs/1603.08983)
- [Liquid Time-Constant Networks (Hasani et al.)](https://arxiv.org/abs/2006.04439)
- [World Models (Ha & Schmidhuber)](https://arxiv.org/abs/1803.10122)
- [Neural Synchrony in Cortical Networks (Uhlhaas et al.)](https://www.frontiersin.org/articles/10.3389/fnint.2009.00005/full)
- [Cortical Travelling Waves (Muller et al.)](https://www.nature.com/articles/s41583-018-0017-4)
- [Kolmogorov-Arnold Networks (Liu et al.)](https://arxiv.org/abs/2404.19756)
- [Quiet-STaR (Zelikman et al.)](https://arxiv.org/abs/2403.09629)
- [A Path Towards Autonomous Machine Intelligence (LeCun)](https://openreview.net/pdf?id=BZ5a1r-kVsf)

---

# Continuous Thought Machines (CTM) 深度解析

## 1. Paper 概览

**Title**: Continuous Thought Machines  
**Authors**: Luke Darlow, Ciaran Regan, Sebastian Risi, Jeffrey Seely, Llion Jones  
**Affiliation**: Sakana AI (Tokyo), University of Tsukuba, IT University of Copenhagen  
**Core thesis**: 现代 NN 为了 efficiency 抽象掉了 neuron-level temporal dynamics 和 neural timing，但 biological brain 的复杂 neural activity 和 synchrony 正是 cognition 的关键。CTM 重新引入这些元素，作为 first-class computational primitive。

两个核心 innovation：
- **Neuron-Level Models (NLMs)**: 每个 neuron 拥有 private weight parameters 处理 incoming pre-activation histories
- **Neural synchronization 作为 latent representation**: 通过 temporal correlations between neuron-level activity 实现 observation 和 prediction

paper 不追求 SOTA,而是 demonstrate neural dynamics 作为 core operating principle 时 unlock 的 capabilities。

---

## 2. Architecture 详解（对应 Figure 3 的 ①-⑩）

### 2.1 Internal Tick Dimension（decoupled from data）

CTM 有一个 internal sequence dimension $t \in \{1, ..., T\}$,与 data 的 sequence dimension 解耦。这允许 model 对 static data（如 ImageNet image）进行 iterative refinement,模拟 "thought steps"。该思想与 [Universal Transformer](https://arxiv.org/abs/1807.03819), [Perceiver](https://arxiv.org/abs/2103.03206), [Looped Transformers](https://arxiv.org/abs/2311.12424) 同源。

### 2.2 Synapse Model ①

公式 (1):
$$\mathbf{a}^t = f_{\theta_{\text{syn}}}(\text{concat}(\mathbf{z}^t, \mathbf{o}^t)) \in \mathbb{R}^D$$

变量解释:
- $\mathbf{z}^t \in \mathbb{R}^D$: 上一个 internal tick 的 post-activations,即 latent state
- $\mathbf{o}^t \in \mathbb{R}^{d_{\text{input}}}$: 当前 tick 的 attention output
- $\mathbf{a}^t \in \mathbb{R}^D$: pre-activations,作为 NLMs 的输入
- $f_{\theta_{\text{syn}}}$: synapse model,采用 [U-Net](https://arxiv.org/abs/1505.04597) style MLP,depth $k=16$ (8 层 down + 8 层 up),bottleneck 宽度 16

**Intuition**: synapse model 把 "current internal state + external observation" 转化为 "next pre-activations",类似 biological synapse 传递 pre-synaptic activity 到 post-synaptic 输入端。U-Net 结构保留了 multi-scale 信息流。

### 2.3 Pre-activation History ②

公式 (2):
$$\mathbf{A}^t = [\mathbf{a}^{t-M+1}, \mathbf{a}^{t-M+2}, ..., \mathbf{a}^t] \in \mathbb{R}^{D \times M}$$

变量解释:
- $M$: memory length,采用 FIFO rolling window
- $\mathbf{A}^t$: 每个 neuron d 对应一个 $\mathbf{A}_d^t \in \mathbb{R}^M$ 向量
- 初始 $\mathbf{A}^{t=1}$ 和 $\mathbf{z}^{t=1}$ 是 learnable parameters
- 实验中 $M \approx 10-100$ 有效

**Intuition**: 每个 neuron 有 short-term memory,记住自己最近的 $M$ 个 inputs。这是 stateless ReLU neuron 所没有的能力。

### 2.4 Neuron-Level Models (NLMs) ③④

公式 (3):
$$\mathbf{z}_d^{t+1} = g_{\theta_d}(\mathbf{A}_d^t)$$

变量解释:
- $g_{\theta_d}$: neuron d 的私有 MLP,depth 1,width $d_{\text{hidden}}$
- $\theta_d$: neuron d 的 private weights(不共享)
- $\mathbf{z}_d^{t+1}$: neuron d 在 $t+1$ tick 的 post-activation

实现 (Listing 2) 用 einsum 高效计算:
```python
# inputs shape: (b, d, M), weights_1 shape: (M, d_hidden, d_model)
out = einsum('bdM,Mhd->bdh', inputs, weights_1) + bias_1  # (b, d, h)
out = einsum('bdh,hd->bd', out, weights_2) + bias_2       # (b, d)
```

**Intuition**: 这是 CTM 与传统 RNN 最关键的区别。LSTM 的 gate 是 fixed sigmoid function,所有 neurons 共享相同 functional form。CTM 中每个 neuron 有自己的 learnable MLP,可以学到不同的 temporal processing pattern:有的 neuron 可能 integrate,有的 oscillate,有的 decay。这是 neural dynamics diversity 的来源。类比:biological brain 中不同 neuron 类型(pyrnamidal, interneuron, etc.)有不同 dynamics。

### 2.5 Post-activation History ⑤

公式 (4):
$$\mathbf{Z}^t = [\mathbf{z}^1, \mathbf{z}^2, ..., \mathbf{z}^t] \in \mathbb{R}^{D \times t}$$

变量解释:
- $\mathbf{Z}^t$: 非固定长度的 post-activation history
- $\mathbf{Z}_d^t \in \mathbb{R}^t$: neuron d 的完整 activation trace

### 2.6 Neural Synchronization ⑥

公式 (5):
$$\mathbf{S}^t = \mathbf{Z}^t \cdot (\mathbf{Z}^t)^T \in \mathbb{R}^{D \times D}$$

每个元素:
$$\mathbf{S}^t_{ij} = \sum_{\tau=1}^t z_i^\tau \cdot z_j^\tau$$

变量解释:
- $\mathbf{S}^t_{ij}$: neuron i 和 neuron j 的 post-activation histories 的内积
- 编码了两个 neuron 在时间上的 **co-activation** 程度

**Intuition**: 这是 CTM 最 deep 的 idea。传统 NN 的 representation 是 D 维 activation vector。CTM 的 representation 是 $D \times D$ synchronization matrix,cardinality 高了 D 倍。更关键的是,它编码的不是 "哪个 neuron active",而是 "哪两个 neurons co-active over time"。

生物学 inspirations ([Uhlhaas et al., 2009](https://www.frontiersin.org/articles/10.3389/fnint.2009.00005/full)):neural synchrony 被认为是 binding problem 的 solution,即不同 features 如何被 integrated 成 coherent perception。Singer 等人的工作表明 synchronized oscillations 在 visual cortex 中编码 object boundaries。

数学上:
- $S_{ij}$ 大且正 → 两 neurons 同步激活
- $S_{ij}$ 大且负 → 两 neurons anti-correlated 激活
- $S_{ij}$ 接近 0 → 两 neurons 独立激活

这提供了 D^2 个 "relationship features",远多于 D 个 "magnitude features"。

### 2.7 Neuron Pair Sub-sampling ⑦⑧

公式 (6)(7):
$$\mathbf{y}^t = \mathbf{W}_{\text{out}} \cdot \mathbf{S}_{\text{out}}^t$$
$$\mathbf{q}^t = \mathbf{W}_{\text{in}} \cdot \mathbf{S}_{\text{action}}^t$$

变量解释:
- $\mathbf{S}_{\text{out}}^t \in \mathbb{R}^{D_{\text{out}}}$: 用于 output projection 的 subsampled synchronization
- $\mathbf{S}_{\text{action}}^t \in \mathbb{R}^{D_{\text{action}}}$: 用于 attention query 的 subsampled synchronization
- $\mathbf{W}_{\text{out}}, \mathbf{W}_{\text{in}}$: projection matrices

三种 sampling 策略 (Appendix C.2):
1. **Dense pairing**: 选 $J$ neurons,计算所有 $\binom{J}{2} + J$ pairs,$D_{\text{out}} = J(J+1)/2$。Strong bottleneck,所有 gradient 都流过 selected neurons
2. **Semi-dense pairing**: 选 $J_1$ 和 $J_2$ 两套 neurons,左 vector 来自 $J_1$,右 vector 来自 $J_2$。Bottleneck 宽度 2 倍
3. **Random pairing**: 随机选 $D_{\text{out}}$ pairs,允许 overlap,加上 $n_{\text{self}}$ 个 self-pairs $(i, i)$ 以保留 snapshot 依赖。开放 bottleneck

**Intuition**: $D \times D$ matrix 太大 (D=4096 时是 16M entries),必须 subsample。Dense pairing 强制 gradient 流过少数 neurons,适合需要 strong inductive bias 的任务 (mazes);Random pairing 允许更自由的 gradient flow,适合大数据集。

### 2.8 Attention ⑨⑩

公式 (8):
$$\mathbf{o}^t = \text{Attention}(Q = \mathbf{q}^t, KV = \text{FeatureExtractor}(\text{data}))$$

- FeatureExtractor: task-specific encoder(如 ResNet, MLP)
- $\mathbf{o}^t$ 与 $\mathbf{z}^{t+1}$ concat 后送入下一 tick 的 synapse model

**Intuition**: attention query 是从 synchronization representation 投影出来的。这意味着 CTM 用 "neuron co-activation patterns" 来决定 attend 哪里。传统 attention 中 Q 来自 hidden state;CTM 中 Q 来自 pairwise neural synchrony,这是一种更 abstract 的 representation。

### 2.9 Temporal Decay Scaling

公式 (9):
$$\mathbf{R}_{ij}^t = [\exp(-r_{ij}(t-1)), \exp(-r_{ij}(t-2)), ..., \exp(0)]^T \in \mathbb{R}^t$$

公式 (10):
$$\mathbf{S}_{ij}^t = \frac{(\mathbf{Z}_i^t)^T \cdot \text{diag}(\mathbf{R}_{ij}^t) \cdot \mathbf{Z}_j^t}{\sqrt{\sum_{\tau=1}^t [\mathbf{R}_{ij}^t]_\tau}}$$

变量解释:
- $r_{ij} \geq 0$: 每对 neurons 的可学习 exponential decay rate
- $\mathbf{R}_{ij}^t$: 时间衰减系数向量,越早的 tick 衰减越多
- 分子:weighted inner product
- 分母:decay 的 AUC (Area Under Curve),用于归一化保证 magnitude 稳定
- $r_{ij} = 0$: 无衰减,所有 ticks 等权
- $r_{ij}$ 大: 偏向 recent ticks,近似 sliding window

**Intuition**: 不同 neuron pairs 可以在不同时间尺度上工作。例如,某些 pair 可能用于 short-term correlation (high $r$),其他 pair 用于 long-term integration (low $r$)。这模仿了 biological brain 中不同 neural circuits 在不同时间尺度运作的特性。

### 2.10 Recursive Computation (Appendix H)

naive 计算 $\mathbf{S}^t$ 是 $O(D^2 t)$,不可行。Paper 给出 first-order recursion:

定义:
$$\alpha_{ij}^t := \sum_{\tau=1}^t e^{-r_{ij}(t-\tau)} z_i^\tau z_j^\tau, \quad \alpha_{ij}^1 = z_i^1 z_j^1$$
$$\beta_{ij}^t := \sum_{\tau=1}^t e^{-r_{ij}(t-\tau)}, \quad \beta_{ij}^1 = 1$$

那么 $S_{ij}^t = \alpha_{ij}^t / \sqrt{\beta_{ij}^t}$,且:
$$\alpha_{ij}^{t+1} = e^{-r_{ij}} \cdot \alpha_{ij}^t + z_i^{t+1} z_j^{t+1} \quad \text{(Eq. 16)}$$
$$\beta_{ij}^{t+1} = e^{-r_{ij}} \cdot \beta_{ij}^t + 1 \quad \text{(Eq. 17)}$$

**Intuition**: 这是 rank-1 update,每个 pair per tick 只需 O(1) computation。从 $O(D^2 t)$ 降到 $O(D_{\text{sub}}) = O(D_{\text{out}} + D_{\text{action}})$ per tick。这是工业级实现的关键 - 否则 D=4096, T=50 的 forward pass 会爆炸。

### 2.11 Loss Function ⑪

公式 (11):
$$L = \frac{\mathcal{L}^{t_1} + \mathcal{L}^{t_2}}{2}$$

变量解释:
- $\mathcal{L}^t = \text{CrossEntropy}(\mathbf{y}^t, y_{\text{true}})$: tick t 的 loss
- $\mathcal{C}^t = 1 - \text{normalized entropy}$: tick t 的 certainty
- $t_1 = \arg\min(\mathcal{L})$: 最低 loss 的 tick
- $t_2 = \arg\max(\mathcal{C})$: 最高 certainty 的 tick

**Intuition**: 这是 native adaptive computation 的实现。简单 sample 可能在 tick 5 就达到 high certainty & low loss,loss 主要从 tick 5 计算;难 sample 需要 tick 40,loss 主要从 tick 40 计算。无需 explicit halting module(vs [PonderNet](https://arxiv.org/abs/2107.05407), [ACT](https://arxiv.org/abs/1603.08983))。argmin 和 argmax 的组合确保 certainty 与 correctness 对齐。

---

## 3. Experimental Results 详细数据

### 3.1 2D Mazes - Sequential Reasoning

**Setup**:
- 39×39 mazes, predict up to 100 steps (left/right/up/down/wait)
- 75 internal ticks, D=2048, M=25, $d_{\text{hidden}}=32$, $k=16$
- ResNet-34 backbone, **no positional encoding**
- 31.9M parameters
- 自动 curriculum:只 loss 已经正确路径 + 5 步

**Results** (Figure 4):
| Model | Params | Path Length=20 | Path Length=100 | Solve Rate |
|-------|--------|-----------------|-----------------|------------|
| FF (1-layer) | 54.8M | low | overfit | - |
| LSTM 1-layer T=50 | 42.3M | low | struggles | - |
| LSTM 3-layer T=75 | 110M | low | unstable | - |
| **CTM** | 31.9M | **high** | **high** | strong |

**Generalization**:
- 训练在 39×39 mazes,测试在 99×99 mazes
- 通过 sequential re-application,CTM 学会 generalize 到更大 mazes (Figure 1c)
- 学会 "look ahead" 路径,然后 follow,类似 episodic future thinking in humans

**Emergent behaviors** (Section I):
- "Double take" 早期训练:rapid approximate solve,然后 restart 慢慢 solve
- 能 "change mind":错误路径 → 正确路径
- Different attention heads 分工:有的全局观察,有的 follow path

**Intuition**: no positional encoding 是关键 constraint - CTM 必须建立 internal world model 来追踪当前位置。LSTM 在长 internal ticks 下 unstable,CTM 通过 NLMs + synchronization 保持 stable dynamics。

### 3.2 ImageNet-1K - Adaptive Processing

**Setup**:
- ResNet-152 backbone (constrained, 3×3 first conv)
- D=4096, T=50, M=25, $d_{\text{input}}$=1024, $n_{\text{heads}}$=16
- $D_{\text{out}}$=8196, $D_{\text{action}}$=2048, $n_{\text{self}}$=32
- Random pairing
- 8 H100 GPUs, 500K iterations, batch size 64
- AdamW, lr=5e-4, cosine annealing

**Results**:
- **72.47% top-1, 89.89% top-5** (uncropped data, 50 internal ticks)
- Adaptive computation (Figure 5a):certainty threshold 0.8 → 大部分 instances 在 10 个 ticks 内可以 halt
- Excellent calibration (Figure 5b)
- 学会 "look around" 图像 (Figure 2b) without any training signal
- 出现 low-frequency traveling waves 在 UMAP-projected neuron activations 上 (Figure 11)

**Intuition**: 72.47% top-1 不如 SOTA (现代 transformer-based models 80%+),但 paper 强调这是 preliminary,目标是 demonstrate mechanism 而非 push performance。"look around" 行为是 emergent 的,没有 explicit loss 引导。Traveling waves 类似 cortical traveling waves (Muller et al., 2018),也是 emergent 的。

### 3.3 Parity - Learning Sequential Algorithms

**Setup**:
- 64-length binary sequence (1 或 -1)
- Predict cumulative parity at each position
- $d_{\text{model}}$=1024, M=25, $d_{\text{hidden}}$=4, k=1
- Semi-dense pairing

**Results** (Figure 6):
| Model | T | Final Accuracy |
|-------|---|----------------|
| LSTM | 10 | unstable |
| LSTM | 25 | unstable |
| CTM | 10 | ~50% sequence |
| CTM | 50 | ~85% |
| CTM | 75 | perfect (some seeds) |
| CTM | 100 | perfect (some seeds) |

**Learned Strategies**:
- **Strategy 1**: attention 从 sequence 开头扫到结尾,逐步更新 parity prediction (Figure 6d)
- **Strategy 2**: attention 从结尾反向扫到开头,在最后 ticks 同时 predict 多个 positions

不同 random seeds 学到不同 strategies (Figure 19, 20)。Run 1 学 backward scan,Run 3 学 forward scan,Run 2 stuck at suboptimal solution。

**Intuition**: parity 是 algorithmic task,需要 model 学到内部 algorithm。LSTM 即使参数匹配也 struggle。CTM 的 NLMs + synchronization 组合提供了学 algorithm 的 sufficient flexibility。

### 3.4 Sorting Real Numbers

- Sort 30 numbers from $\mathcal{N}(0, I_{30})$
- Output sorted indices 用 [CTC loss](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- Wait times pattern (Figure 27a): initial wait time 高,中间低,最后有小 bump
- Wait times 与 "data delta" (相邻 values 差) 相关 (Figure 27b):大 delta → 长 wait time
- Generalize 到不同 normal distributions (Figure 27c)

**Intuition**: wait time 反映 internal algorithm。大 delta 需要更多 processing 时间,类似 humans 排序时遇到"距离远的数字"会停顿一下。

### 3.5 Q&A MNIST - Memory & Arithmetic

**Setup**:
- Observe N digits (1-4), then interwoven index+operator embeddings
- Modular arithmetic (mod 10)
- T=1 or T=10 per input, M=3 or M=30

**Results**:
| Model | Repeats/Input | Accuracy (4 digits, 4 ops) |
|-------|---------------|----------------------------|
| LSTM | 1 | high |
| CTM | 1 | lower |
| LSTM | 10 | 21% |
| **CTM** | 10 | **96%+** |

**Generalization** (Figure 31):CTM 能 generalize 到更多 digits 和 operations than training。CTM 学会 incremental computation (Figure 32):每个 index+operator tuple 后输出 intermediate result,而不是 wait for answer flag。

**Intuition**: digit observations 必须在 memory window 之外被 recall。CTM 通过 synchronization 实现 long-term memory - 不是 stored activations,而是 activation correlations。LSTM 在 multi-tick setting 不稳定,CTM 反而更稳定。

### 3.6 Reinforcement Learning

**Environments**: CartPole, Acrobot, MiniGrid Four Rooms (POMDPs)
**Method**: PPO with sliding window M (rather than full history)
**Results**: Performance 与 LSTM 相当,但 neuron dynamics 更丰富 (Figure 35)
**Implementation details**:
- 用 sliding window 而非 full history(防止 history 无限增长)
- 用 learned initial state trace
- Two-layer feedforward synapse (UNet 在 RL 中 too heavy)

**Intuition**: CTM 作为 stateful RNN,在 POMDP 中处理 partial observability 通过保留 activation history。RL 中 input 是 sequential 的,internal ticks 嵌套在 environment steps 中。

### 3.7 Ablations

**Maze ablation** (Table 5):
| Model | Test Accuracy | Solve Rate |
|-------|---------------|------------|
| **CTM** | 94.6% ± 0.7% | 65.9% ± 5.7% |
| CTM (No NLMs) | 82.9% ± 4.4% | 35.0% ± 7.2% |
| CTM (No Synch) | 85.1% ± 0.5% | 37.5% ± 0.7% |
| LSTM + Synch | 82.4% ± 0.9% | 33.8% ± 3.3% |

**CIFAR-100 width ablation** (Figure 24):
- Wider models → 更 diverse neural activity (cosine similarity 分布更集中在 0 附近)
- Performance 先提升后下降(可能 overfitting 或需要更多 training)

**CIFAR-100 tick ablation** (Figure 25):
- T=50 最 performant
- 出现 **two regions of high certainty**:early ticks 和 later ticks
- 暗示 two-phase processing (可能 fast intuition + slow deliberation)

**Intuition**: ablation 清晰说明:
- NLMs 单独不够(没有 synchronization 作为 representation)
- Synchronization 单独不够(没有 NLMs 产生复杂 dynamics)
- LSTM + synchronization 失败 → synchronization 不能简单加到现有 RNN 上,必须与 NLMs 配合
- 两个 components 是 complementary 的,不是 redundant 的

---

## 4. Intuition 构建 - 从多个角度

### 4.1 为什么 NLMs 重要?

传统 NN 的 neuron: $z = \sigma(Wx + b)$ - stateless scalar function,没有时间维度。Complexity 通过 layer 堆叠获得。

CTM 的 neuron: $z_d^{t+1} = g_{\theta_d}(\mathbf{A}_d^t)$ - 每个 neuron 是一个 dynamic system,有 memory (M 步 history) 和 processing (私有 MLP)。

这使得单个 neuron 可以产生:
- Oscillation (类似 biological neuron)
- Integration / accumulation
- Decay
- Differentiation (sensitivity to change)
- Pattern matching (specific input sequences)

类比:传统 NN 是 feedforward circuit(逻辑门);CTM 是 dynamical system(振动电路)。NLMs 是把 neuron 从 scalar function 提升为 temporal processor。

### 4.2 为什么 synchronization 重要?

考虑 D=4096 的 model:
- 传统 representation: 4096 维 vector
- Synchronization representation: $D^2/2 \approx 8M$ 个 unique pairs

Cardinality 高 2000 倍。更重要的是 representation 的 nature 不同:
- Activation vector: "which neurons are active now"
- Synchronization matrix: "which neurons co-active over time"

后者是 temporal relationship,前者是 instantaneous state。Binding problem(Singer 等)认为 synchrony 是 brain 解决 feature binding 的 mechanism - 不同 features (颜色、形状) 通过 synchronized firing 被 bound 成 coherent perception。

数学上, $S_{ij}$ 是 inner product,所以 $D \times D$ matrix 是 positive semi-definite 的,可以看作 kernel matrix。这给 representation 提供了 rich structure。

### 4.3 为什么 internal tick decouple from data?

这允许 CTM 对 static data "think"。一个 image 不是一次 forward pass 处理完,而是 CTM 通过 attention 反复观察,每次 update internal state。

这与 [Kahneman's System 1 vs System 2](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) dichotomy 相关:
- 传统 feedforward NN: System 1 (fast, automatic, parallel)
- CTM: 引入 System 2 元素 (slow, deliberative, sequential)

也呼应 [Test-Time Compute scaling](https://arxiv.org/abs/2502.05171):inference 时多 think → better performance。但 CTM 的 multi-tick 是 native 而非 post-hoc。

### 4.4 为什么 loss function 这样设计?

argmin(loss) + argmax(certainty) 双重 anchor:
- argmin(loss): 鼓励 model 在某个 tick 达到 best prediction
- argmax(certainty): 鼓励 certainty 与 correctness 对齐(否则 high certainty + wrong prediction 会被 penalize)

vs alternatives:
- Final-tick loss only: 强制 model 必须在最后 tick 解决,无法利用 early ticks
- All-tick loss (e.g., Universal Transformer): 平均所有 ticks,信号稀释
- ACT/PonderNet: 需要 explicit halting module,gradient path 复杂

CTM 的设计简单且有效:每个 sample 找到自己的"最佳思考时刻",loss 自动从那里计算。

### 4.5 为什么 emergent properties 重要?

paper Section I 列出 11 个 emergent properties,这些都没有 explicit training signal:

1. **Periodic dynamics** emerge during training (不是 initialization 时)
2. **Dead neurons** 可视化 - 新的 utilization 诊断
3. **"Double take"** in maze solving - early training 的 approximate-then-refine 行为
4. **"Change mind"** - 错路径 → 对路径
5. **Attention head specialization** - global vs local heads
6. **"Look around"** degree 增加 with training on ImageNet
7. **Attention shifts** between broad and narrow views
8. **Directional attention** without positional encoding
9. **Unusual strategies** under constraints (e.g., backward maze solving)
10. **Forward vs backward** parity strategies (seed-dependent)
11. **Incremental computation** in Q&A MNIST

**Intuition**: 这些 behaviors 都不是 hard-coded inductive bias,而是 model 自己 discover 的。这暗示 CTM 的 architecture 提供了 sufficient flexibility 让 model explore solution space。这也是 paper 的 core message:好的 architecture 应该让 capability emerge,而非 impose。

### 4.6 与 Biological Plausibility 的关系

CTM 不是 faithful biological model:
- 不是 spiking networks
- 没有 dendrites/axons/chemical synapses
- 没有 different neuron types (excitatory/inhibitory)
- 用 differentiable continuous activations

但 CTM 保留了关键的 biological principles:
- **Neuron-level temporal processing**: 单个 neuron 有 dynamics
- **Neural synchrony as representation**: 时间相关性是 computation 基础
- **Decoupled internal time**: thinking 与 input 时间分离

[Liquid State Machines](https://link.springer.com/chapter/10.1007/978-3-642-19492-8_13) 也是 temporal dynamics 但通常 non-differentiable;[SNNs](https://www.nature.com/articles/s41583-022-00613-3) 用 spikes 但 training 困难。CTM 的贡献是把 biological inspiration 与 modern deep learning tractability 结合。

---

## 5. Related Work 关系图

### 5.1 Adaptive Computation
- [PonderNet](https://arxiv.org/abs/2107.05407): learnable halting module with ponder probability
- [ACT](https://arxiv.org/abs/1603.08983): Graves 的 halting distribution
- [AdaTape](https://arxiv.org/abs/2305.03765): dynamic input sequence extension
- [Sparse Universal Transformer](https://arxiv.org/abs/2310.07096): recurrent + halting + MoE
- **CTM**: adaptive compute 是 emergent,无 halting module

### 5.2 Iterative Reasoning
- [Quiet-STaR](https://arxiv.org/abs/2403.09629): hidden rationale generation in LLMs
- [RIMs](https://arxiv.org/abs/1909.10893): modular asynchronous sub-networks
- [RAM](https://arxiv.org/abs/1406.6247): recurrent visual attention
- **CTM**: temporal patterns of synchronization 作为 primary representation

### 5.3 Biologically Inspired
- [Liquid Time-Constant Networks](https://arxiv.org/abs/2006.04439): ODE-based time-varying neurons
- [Spiking Neural Networks](https://www.nature.com/articles/s41583-022-00613-3): discrete spikes
- [Artificial Kuramoto Oscillatory Neurons](https://arxiv.org/abs/2410.13821): oscillator-based NNs
- **CTM**: continuous, differentiable, GPU-friendly abstraction

### 5.4 Synchronization
- [Reichert & Serre 2013](https://arxiv.org/abs/1312.6115): synchrony as gating in complex-valued NNs
- [Complex-valued NNs survey](https://ieeexplore.ieee.org/document/9842802): control-theoretic synchrony
- **CTM**: synchrony as learned latent representation (first at this scale)

### 5.5 Recurrence
- [Universal Transformer](https://arxiv.org/abs/1807.03819): iterative refinement with shared weights
- [Perceiver](https://arxiv.org/abs/2103.03206): iterative attention for general perception
- [Looped Transformers](https://arxiv.org/abs/2311.12424): learn algorithms via loops
- [Latent Reasoning](https://arxiv.org/abs/2502.05171): recurrent depth for test-time compute
- **CTM**: recurrence + NLMs + synchronization (新增 dynamic neuron + temporal correlations)

---

## 6. Limitations & Open Questions

- **Training time extended**: internal sequence 多次 forward pass
- **Parameter count**: NLMs 比 scalar activations 参数多 (D × M × d_hidden × d_hidden × 2)
- **Comparison breadth > depth**: 没有与 SOTA models head-to-head
- **No language modeling experiments**: 文本任务未探索
- **No self-supervised learning**: 仅 supervised RL 和 classification
- **Synchronization sampling is heuristic**: random/dense pairing 不是 principled

---

## 7. Reference Links

**Paper & Code**:
- [Sakana AI - Continuous Thought Machines](https://sakana.ai/ctm/)
- [arXiv version (if available)](https://arxiv.org/abs/2505.05522) - 推测的 arxiv ID
- [GitHub repository](https://github.com/SakanaAI/CTM) - 推测的 repo

**Related Work**:
- [Universal Transformer (Dehghani et al.)](https://arxiv.org/abs/1807.03819)
- [Perceiver (Jaegle et al.)](https://arxiv.org/abs/2103.03206)
- [PonderNet (Banino et al.)](https://arxiv.org/abs/2107.05407)
- [Adaptive Computation Time (Graves)](https://arxiv.org/abs/1603.08983)
- [Liquid Time-Constant Networks (Hasani et al.)](https://arxiv.org/abs/2006.04439)
- [World Models (Ha & Schmidhuber)](https://arxiv.org/abs/1803.10122)
- [Spiking Neural Networks review](https://www.nature.com/articles/s41583-022-00613-3)
- [Neural Synchrony (Uhlhaas et al.)](https://www.frontiersin.org/articles/10.3389/fnint.2009.00005/full)
- [Traveling Waves in Cortex (Muller et al.)](https://www.nature.com/articles/s41583-018-0017-4)
- [Looped Transformers (Yang et al.)](https://arxiv.org/abs/2311.12424)
- [Latent Reasoning (Geiping et al.)](https://arxiv.org/abs/2502.05171)
- [Quiet-STaR (Zelikman et al.)](https://arxiv.org/abs/2403.09629)
- [Recurrent Independent Mechanisms (Goyal et al.)](https://arxiv.org/abs/1909.10893)
- [RAM (Mnih et al.)](https://arxiv.org/abs/1406.6247)
- [AdaTape (Xue et al.)](https://arxiv.org/abs/2305.03765)
- [U-Net (Ronneberger et al.)](https://arxiv.org/abs/1505.04597)
- [Attention is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
- [ResNet (He et al.)](https://arxiv.org/abs/1512.03385)
- [CTC Loss (Graves et al.)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [AdamW (Loshchilov & Hutter)](https://arxiv.org/abs/1711.05101)
- [UMAP (McInnes et al.)](https://arxiv.org/abs/1802.03426)
- [Kuramoto Oscillatory Neurons (Miyato et al.)](https://arxiv.org/abs/2410.13821)
- [CIFAR-10H (Peterson et al.)](https://arxiv.org/abs/1810.07281)
- [CIFAR-10D (Ho-Phuoc)](https://arxiv.org/abs/1811.07270)
- [Gymnasium RL environments](https://github.com/Farama-Foundation/Gymnasium)
- [MiniGrid](https://arxiv.org/abs/2306.13831)
- [PPO (Schulman et al.)](https://arxiv.org/abs/1707.06347)
- [RNNs Learning Algorithms (Schwarzschild et al.)](https://arxiv.org/abs/2106.08927)
- [Kahneman System 1/2](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow)
- [Building Machines that Learn Like People (Lake et al.)](https://arxiv.org/abs/1604.00289)
- [Path Towards Autonomous Machine Intelligence (LeCun)](https://openreview.net/pdf?id=BZ5a1r-kVsf)

---

## 8. Final Intuition 总结

从 Karpathy 视角的直觉构建:

1. **CTM 把 neuron 从 scalar function 提升为 dynamic system**。传统 NN 的 "depth" 来自 layers;CTM 的 depth 来自 internal ticks + 每 neuron 的 temporal processing。这是一个新的 scaling dimension。

2. **Synchronization 是 D^2 维的 representation**,而 activation vector 是 D 维。这意味着 CTM 在 representation capacity 上有 D 倍 advantage。这个 "alphabet size" 的提升是关键的 representation innovation。

3. **Adaptive compute 是 emergent 而非 designed**。Loss function 选择 argmin(loss) + argmax(certainty),adaptive 行为自然涌现。这是 paper 的 elegance - 简单机制产生复杂行为。

4. **Maze 任务的设计精妙**:no positional encoding 强制 model 建立 internal spatial representation。这比 standard maze tasks 更 probing - 测试 model 能否 form world model,而非 memorize spatial patterns。

5. **ImageNet "look around" 行为是 emergent**,没有 explicit loss 引导。这暗示 CTM 的 architecture 让 visual processing 自然产生 sequential exploration,类似 human eye movements。

6. **Parity 任务展示 CTM 学到 interpretable algorithms**(forward scan vs backward scan)。这是 algorithmic reasoning 的 evidence。

7. **Ablations 清晰表明 NLMs 和 synchronization 是 synergistic**。LSTM + synchronization 失败证明不能 simple "add synchronization" to existing architectures。

8. **Recursive computation (Eq. 16-17) 是工业级实现的关键**,把 $O(D^2 t)$ 降到 $O(D_{\text{sub}})$ per tick。没有这个 optimization,CTM 不可行。

9. **Limitations 诚实**:paper 明确说不追求 SOTA,而是 demonstrate mechanism。72.47% ImageNet 不是终点,而是 starting point。

10. **Future directions**:language modeling, self-supervised video, lifelong learning, multi-modal - 所有这些都需要 CTM 的 temporal dynamics + adaptive compute。NLMs + synchronization 是 general-purpose primitive。

paper 的更大 contribution 可能不在 performance,而在于 **重新引入 neural timing 作为 first-class computational primitive** 这一 conceptual shift。从 long arc of AI research 看,把时间 dynamics 放回 model 的核心,可能是 missing piece towards more general intelligence。
