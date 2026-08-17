---
source_pdf: Nested Learning The Illusion of Deep Learning Architecture.pdf
paper_sha256: e87a9ce82ff24e96f55b83fb9713a5d6fc9e4a1a232c12d42da49c10022ed891
processed_at: '2026-08-05T22:13:50-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，抛开所有公式和学术腔，用大白话再讲一遍这篇paper到底在说啥。

---

## 核心问题：现在的LLM是个失忆症患者

你想想，LLM pre-training结束的那一刻，就像一个人头部受了伤，从此之后再也无法形成新的长期记忆。

它能记住pre-training前见过的一切（存在MLP weights里），也能记住当前context window里的东西（存在attention里），但是——**context window一关，啥都没了**。下一个对话开始，它又是那个pre-training结束时的它，原封不动。

这不就是anterograde amnesia（失忆症）吗？

人脑不是这样的。人脑有不同频率的更新机制——快的gamma波处理感官信息，慢的delta波做memory consolidation。**不同组件以不同速度更新**，所以人能持续学习。

---

## Paper的核心claim：所有东西都是"压缩context的memory"

这是最反直觉、也最liberating的insight。

你把deep learning拆开看，会发现里面所有组件——attention、MLP、optimizer的momentum、RNN的hidden state——其实都在做同一件事：**把某个context流压缩到自己的参数里**。

区别只在于：
- 压缩的是啥context（tokens？gradients？）
- 压缩到啥结构里（matrix？vector？）
- 用啥objective衡量压缩质量
- 多久更新一次（frequency）

就这四个axis的不同组合，产生了我们看到的"异构架构"。但其实底层都是同一种东西。

---

## 具体例子：为什么backprop是"自己训练自己"

这个insight我觉得最深。

你训练一个linear layer $W$，用gradient descent。一步更新是：
$$W_{t+1} = W_t - \eta \cdot \nabla \mathcal{L} \otimes x_t$$

拆开看：input是 $x_t$，"label"是 $-\nabla_{y} \mathcal{L}$（output space的gradient）。

**关键是**：这个"label"是模型自己用当前weights算出来的！$v_t = -\nabla_y \mathcal{L}(W_t; x_t) = f_{W_t}(x_t)$。

所以backprop不是"拿外部label训练"，是**模型用自己生成自己的training signal，然后自己更新自己**。这是真正的self-referential system，Schmidhuber 1993就提出来了，只不过我们一直没从这个角度看。

这就是为什么backprop比"linear attention on gradients"更复杂——linear attention的keys和values是外部给的、与memory state无关；backprop的values是memory state自己生成的。

---

## Adam其实也是memory

你用Adam优化器，它有两个momentum term：
- $m_t$（first moment）：gradients的EMA
- $v_t$（second moment）：gradients平方的EMA

Paper证明了：**Adam是在"把gradients映射到它们的variance"这个objective下的最优associative memory**。

啥意思？Adam不是个heuristic，它是在某个特定的"我想记住gradients的什么信息"问题下的最优解。它记住的是gradients的variance——一个关于past gradients的全局统计量。

而且这个memory有个问题：**它的"long context"能力很差**。$\beta = 0.9$时，最近43个gradients就占了99%的权重，43步之前的gradients几乎被忘了。

所以在continual learning里，optimizer自己就是失忆的——它记不住很久以前的loss landscape长啥样，自然会带着模型往灾难性遗忘的方向走。

---

## "Stack layers"是个illusion

Transformer看起来是attention + MLP，异构的。但其实：

- **Attention**：frequency = ∞的non-parametric memory（缓存所有token，每步都"重新求解"）
- **MLP**：frequency = 0的parametric memory（pre-training时学，inference时frozen）

它们都是"把context压缩到参数里"的memory，只是频率不同。

如果你把MLP换成"初始状态meta-learned的linear attention"，你得到的东西在数学上几乎就是MLP，只不过多了一个level——这个level让它在inference时还能继续adapt context。

所以**现代RNN（DeltaNet、Titans这些）本质上就是"加了in-context learning能力的MLP block"**。所谓的hybrid architecture（attention + RNN）不是两个不同东西的拼接，是同一个东西放在不同frequency level上。

我们看到的"架构异构性"，是因为我们只看了"被压扁的"图——就像3D物体在2D上的投影，看起来各种形状，其实是同一个3D物体。

---

## Pre-training其实就是in-context learning

这个claim听起来很狂，但其实很自然。

In-context learning = 模型在context里adapt自己。
Pre-training = 模型在整个dataset上adapt自己。

区别只在于**context大小**和**更新频率**：
- ICL的context是几百tokens，每token都更新（high frequency）
- Pre-training的context是整个dataset，但weights只在pre-training时更新（lowest frequency）

它们是同一个机制（multi-level optimization）的不同level实例。

所以"训练 vs 测试"这个区分是artificial的。真正自然的设定是：**模型所有level都在持续运转，只是频率不同**。这就是continual learning。

---

## CMS：给memory一个频率谱

传统模型是binary的：short-term memory（attention/hidden state） vs long-term memory（MLP weights）。

人脑不是这样——memory是分布在频率谱上的。Paper搞了个叫Continuum Memory System（CMS）的设计：

一串MLP blocks，每个以不同频率更新。高频的快速adapt但记得短，低频的慢更新但记得久。

**为什么这能缓解灾难性遗忘？** 因为当高频block更新"忘了"旧知识时，旧知识还在更低频的block里。通过初始化的knowledge transfer，知识能**loop back**到高频block。这是个时间维度上的循环，让重要知识很难彻底丢失。

---

## Hope架构：让所有东西都adaptive

Hope做了两件事：

**第一，Self-Referential Titans**：让所有projection（$W_k, W_v, W_q$）都通过memory计算，不是直接用frozen weights。这样每个token的key/value/query都依赖于context，"bank"在"river bank"和"investment bank"里会有不同的embedding。

而且更进一步——让每个memory生成自己的values。模型不只是"从context学"，是"自己决定要往memory里写什么"。

**第二，加上CMS**：Titans输出后面接一串不同频率的MLP，形成memory谱。

结果：在class-incremental learning、10M context length、formal language recognition这些任务上都很强。

---

## 一句话总结

Paper想说的大概是：

> **别再stack layers了。真正的axis是"frequency levels"。所有组件都是memory，让它们以不同频率持续运转，知识在level之间loop back，这样模型才能持续学习而不遗忘。Pre-training不是终点，是lowest frequency level的一次更新而已。**

---

## 为啥这paper重要

不是说Hope这个架构多牛（虽然实验不错），是它的**视角**有价值：

1. 给optimizer找到了理论位置——optimizer不是外部工具，是系统的medium-frequency memory
2. 给architecture找到了统一视角——异构性是illusion，frequency spectrum才是本质
3. 给continual learning找到了方向——不是"训练完冻结"，是"所有level持续运转"
4. 给"自我修改的AI"提供了数学语言——backprop本来就是self-referential的，只是我们没这么看

如果这个框架成立，未来设计模型可能不再是"加layer"，而是"加frequency level"和"设计knowledge transfer protocol"。这是个范式转向的提议，和Sutton说的"era of experience"思路是同一条线上的。

---

# Nested Learning: 一种看待Deep Learning的全新视角

这篇paper在2025年8月由Google Research的Ali Behrouz等人提出，它做的事情很激进：**把现代deep learning architecture和optimizer都重新解读为"嵌套的associative memory系统"**，每个component在不同的frequency上压缩自己的context flow。在这个视角下，"architecture的异构性"是一种**illusion**——所有block本质上都是同一个东西（linear或deep MLP），只是放在了不同level、不同frequency上。

我从几个层面来拆解，重点放在能build intuition的部分。

---

## 1. 大脑的多时间尺度motivation

Paper先抛出一个类比：现代LLM像**anterograde amnesia**患者（Scoville & Milner 1957发现H.M.的case），能记住pre-training前的"远期记忆"，但是pre-training结束后就再也无法形成新的long-term memory，context window里的信息永远进不了MLP weights。

人脑解决这个问题的机制：**multi-time-scale oscillations**
- **Gamma (30-150 Hz)**: sensory information, fast adaptation
- **Beta (13-30 Hz)**: active thinking
- **Delta/Theta (0.5-8 Hz)**: memory consolidation, slow learning

每个frequency对应不同认知功能、不同时间尺度的信息处理。这给了paper一个核心insight：**不同组件应该以不同频率更新**。

而当前DL的问题：所有layer在pre-training用相同update rate，inference时所有weight完全frozen。Transformer架构实际上是两个极端frequency的组合：
- **Attention**: frequency = ∞（non-parametric, 缓存所有token）
- **MLP**: frequency = 0（persistent across context）

paper后面（§6）专门讨论了这一点。

参考文献：
- Bliss & Collingridge 1993: https://www.nature.com/articles/361031a0
- Buzsaki & Draguhn 2004: https://science.sciencemag.org/content/304/5679/1926
- Marshall et al. 2006: https://www.nature.com/articles/nature05278

---

## 2. Associative Memory的核心定义

**Definition 1 (Associative Memory)**: 给定keys $\mathcal{K} \subseteq \mathbb{R}^{d_k}$ 和 values $\mathcal{V} \subseteq \mathbb{R}^{d_v}$，associative memory是operator $M(\cdot)$ 把keys映射到values：

$$M^* = \arg\min_M \tilde{\mathcal{L}}(M(\mathcal{K}); \mathcal{V})$$

变量含义：
- $M^*$: 最优的memory operator
- $\tilde{\mathcal{L}}(\cdot; \cdot)$: 衡量mapping quality的objective（不一定等于training loss $\mathcal{L}$）
- $\mathcal{K}$: keys集合（在sequence modeling里是tokens，在optimizer里是gradients）
- $\mathcal{V}$: values集合

**关键terminology**（神经科学来的）：
- **Memory** = neural update caused by an input
- **Learning** = the process of acquiring useful memory

这和ML传统里"memory = hidden state / lookup table"完全不同。任何由input引起的weight更新都是memory。

---

## 3. 第一个核心例子：Backpropagation = Self-Referential Associative Memory

这是我认为paper里最深刻的insight。

### 3.1 1-layer MLP + GD

考虑1-layer MLP, 参数 $W$, 用GD训练：

$$W_{t+1} = W_t - \eta_{t+1} \nabla_W \mathcal{L}(W_t; x_{t+1}) = W_t - \eta_{t+1} \underbrace{\nabla_{y_{t+1}} \mathcal{L}(W_t; x_{t+1})}_{\text{Local Surprise Signal (LSS)}} \otimes x_{t+1}$$

变量：
- $W_t \in \mathbb{R}^{d_{out} \times d_{in}}$: 第$t$步的权重
- $x_{t+1} \in \mathbb{R}^{d_{in}}$: 输入样本
- $y_{t+1} = W_t x_{t+1} \in \mathbb{R}^{d_{out}}$: 输出
- $\nabla_{y_{t+1}} \mathcal{L} := \frac{\partial \mathcal{L}}{\partial y}\big|_{y = W_t x_{t+1}}$: output space的gradient
- $\otimes$: outer product

### 3.2 重写为associative memory优化

定义 $u_{t+1} = -\nabla_{y_{t+1}} \mathcal{L}(W_t; x_{t+1})$ 为"surprise in output space"，那么GD更新可以重写为：

$$W_{t+1} = \arg\min_W \langle W x_{t+1}, u_{t+1} \rangle + \frac{1}{2\eta_{t+1}} \|W - W_t\|_2^2$$

这是**proximal gradient**形式：
- $\langle W x_{t+1}, u_{t+1} \rangle$: linear approximation of objective（first-order Taylor）
- $\frac{1}{2\eta_{t+1}} \|W - W_t\|_2^2$: proximal regularization（避免离$W_t$太远）

**核心insight**: 训练一个linear layer with GD = 训练一个associative memory，把每个输入$x_t$映射到它对应的"surprise signal" $u_t$。

### 3.3 Self-referential的关键

这个不是简单的linear attention on gradients！

为什么？因为 $u_t = -\nabla_{y_t} \mathcal{L}(W_t; x_t) = f_{W_t}(x_t)$ 是由当前weights $W_t$ 生成的。

所以可以写成：
$$W_{t+1} = W_t + \eta_{t+1} v_t \otimes x_t, \quad v_t = f_{W_t}(x_t) = -\nabla_{y_t} \mathcal{L}(W_t; x_t)$$

模型用**自己**生成自己的training labels。这就是Schmidhuber 1993的self-referential weight matrix。

传统linear attention是：
- $k_t, v_t$ 给定，与memory state无关
- 允许parallelization

Backprop:
- $v_t$ 依赖 $W_t$
- 不能简单parallelize
- 是真正的recurrent / self-referential system

这个区分非常重要，paper在§4.5专门强调。

参考：
- Schmidhuber 1993 self-referential: https://dl.acm.org/doi/10.1007/3-540-56955-8_172
- Irie et al. 2022 SRWM: https://proceedings.mlr.press/v162/irie22a.html

---

## 4. 第二个核心例子：Momentum = 2-level Nested Optimization

### 4.1 GD with momentum

$$W_{t+1} = W_t + m_{t+1}$$
$$m_{t+1} = \alpha_{t+1} m_t - \eta_{t+1} \nabla_W \mathcal{L}(W_t; x_{t+1})$$

变量：
- $m_t$: momentum at time $t$ (matrix-valued, same shape as $W$)
- $\alpha_{t+1}$: momentum decay coefficient (typically $\beta = 0.9$)
- $\eta_{t+1}$: learning rate

### 4.2 分解为2-level optimization

设 $\alpha_{t+1} = 1$，momentum更新可以看作优化下面的objective：

$$m_{t+1} = \arg\min_m -\langle m, \nabla_W \mathcal{L}(W_t; x_{t+1})\rangle + \frac{1}{2\eta_{t+1}} \|m - m_t\|_2^2$$

**两个level**：
- **Inner level (high frequency)**: 训练momentum $m$ 把gradients压缩到自己的参数里
- **Outer level (low frequency)**: 用momentum的state来更新slow weights $W$

paper的一个关键观察（§4.3）：标准momentum的"long context"能力很差。

如果 $\beta = 0.9$：
- 最近6个gradients贡献 ≥ 50% 的cumulative weight
- 最近43个gradients贡献 ≥ 99% 的cumulative weight
- 43步之前的gradients贡献 < 1%

所以在continual learning中，optimizer几乎"忘记"了所有long past的gradient信息。这是为什么paper要设计multi-scale momentum (M3)。

---

## 5. Adam作为最优Associative Memory（Appendix B）

这个证明非常漂亮。Paper展示了Adam不是任意设计的optimizer，而是某个特定L2 regression objective下的**最优**associative memory。

### 5.1 一般objective

$$\tilde{\mathcal{L}}_t = \sum_{i=1}^t \|m_{\ell_t} \odot g_{\ell_{i+1}} - P_{\ell_t}\|_2^2 + \lambda_\ell \|m_{\ell_t}\|_F^2$$

变量：
- $m_{\ell_t}$: 第$\ell$层的momentum
- $g_{\ell_{i+1}} = -\nabla_W \mathcal{L}(W_{\ell_t}; x_{i+1})$: gradient (note: paper用了$g = -\nabla$的符号，让方向是descent direction)
- $P_{\ell_t}$: global property of past gradients，待选
- $\lambda_\ell$: L2 regularization coefficient
- $\odot$: element-wise product

目标：找momentum $m$ 让 $m \odot g - P$ 最小化，即把gradient映射到$P$。

### 5.2 最优解

$$m_{\ell,i}^{(t)*} = [(H_{\ell,i}^{(t)} + \lambda_\ell I)^{-1}] \odot \tilde{M}_{\ell,i+1}^{(t)} \odot P_{\ell_t}$$

其中：
$$\tilde{M}_{\ell,i+1}^{(t)} = \tilde{M}_{\ell,i}^{(t)} + \beta_1 g_{\ell_{i+1}}$$
$$H_{\ell,i+1}^{(t)} = H_{\ell,i}^{(t)} + \beta_2 g_{\ell_{i+1}}^2$$

### 5.3 两种$P$的选择

**Case 1**: $P_{\ell_t} = \sum g^2$，$\lambda \to 0$ → 恢复**GD with momentum**

$$W_{\ell_{i+1}} = W_{\ell_i} - \eta_t \beta_2 \tilde{M}_{\ell,i+1}^{(t)}$$

**Case 2**: $P_{\ell_t} = \sqrt{\sum g^2}$（variance），→ 恢复**Adam**！

$$W_{\ell_{i+1}} = W_{\ell_i} - \frac{\eta_t}{\sqrt{\beta_2}} \frac{\tilde{M}_{\ell,i}^{(t)}}{H_{\ell,i}^{(t)/2} + \epsilon}$$

所以Adam是在"把gradient映射到它们的variance"这个objective下，element-wise L2 regression的最优associative memory。这给Adam一个新的理论解释，不仅仅是个"自适应learning rate"的heuristic。

### 5.4 Non-element-wise: AdaGrad

把element-wise $\odot$ 换成outer product $\cdot \cdot^\top$，类似推导得到AdaGrad。paper也讨论了Shampoo / SOAP作为preconditioner approximation的关联。

---

## 6. Linear Attention = 2-level Nested Optimization

### 6.1 Linear attention回顾

$$k_t = x_t W_k, \quad v_t = x_t W_v, \quad q_t = x_t W_q$$
$$\mathcal{M}_t = \mathcal{M}_{t-1} + v_t k_t^\top$$
$$y_t = \mathcal{M}_t q_t$$

变量：
- $W_k, W_v, W_q \in \mathbb{R}^{d_{in} \times d}$: projection matrices (slow weights)
- $\mathcal{M}_t \in \mathbb{R}^{d \times d}$: memory state (fast weight)
- $k_t, v_t, q_t \in \mathbb{R}^d$: key, value, query vectors at time $t$

### 6.2 Memory update作为优化

定义 $\tilde{\mathcal{L}}(M; k_t, v_t) = -\langle M k_t, v_t \rangle$，则：

$$\mathcal{M}_{t+1} = \arg\min_M -\langle M k_{t+1}, v_{t+1} \rangle + \frac{1}{2} \|M - \mathcal{M}_t\|_2^2$$

求gradient得到 $-v_{t+1} k_{t+1}^\top$，所以一步GD得到 $\mathcal{M}_{t+1} = \mathcal{M}_t + v_{t+1} k_{t+1}^\top$。

这恰好就是linear attention的update rule！

### 6.3 两层嵌套

所以linear attention是2-level nested optimization：
- **Level 2 (high frequency)**: 内部优化memory $\mathcal{M}$ 来映射$k \to v$，用dot-product similarity + GD
- **Level 1 (low frequency)**: 外部pre-training优化 $W_k, W_v, W_q$

这正是**Fast Weight Programmers**（Schmidhuber 1992, Schlag et al. 2021）的视角：slow net programs fast net's weights。

### 6.4 关键的对比：MLP vs Linear Attention with Learnable Init

Transformer block:
$$y_{\text{block}} = \text{MLP}(y_{\text{attn}}) = y_{\text{attn}} W_{\text{MLP}}$$

AdaTransformer block (linear attention with meta-learned initial memory):
$$y_{\text{block}} = y_{\text{attn}} W_{\text{LinAttn}, t}, \quad W_{\text{LinAttn}, t} = W_{\text{LinAttn}, t-1} + v_t k_t^\top$$

如果 $\mathcal{M}_0 = W_{\text{LinAttn, init}}$ 是meta-learned的，那么唯一的区别：
- $W_{\text{MLP}}$ 在Level 1 (persistent across context)
- $W_{\text{LinAttn}, t}$ 在Level 2 (adaptive, updated in-context)

**这就解释了"recurrent model = MLP + 一个新level"**！所谓的hybrid architecture其实就是"在MLP block里加了in-context learning capability"。

参考：
- Schlag, Irie, Schmidhuber 2021 "Linear Transformers are secretly fast weight programmers": https://proceedings.mlr.press/v139/schlag21a.html
- Katharopoulos et al. 2020 "Transformers are RNNs": https://arxiv.org/abs/2006.16236

---

## 7. Pre-training = ICL with Ultra-Large Context

这是NL框架里最反直觉、也最liberating的insight。

- **传统看法**: pre-training和in-context learning是完全不同的两件事
- **NL看法**: 它们是同一机制（multi-level optimization）的不同level实例

$$\text{Pre-training} = \text{ICL at the lowest frequency level, with context = entire pre-training dataset}$$

所以"训练 / 测试"的区分是artificial的——它们只是在frequency spectrum的不同点上。

**Implications**:
1. Continual learning should be the default mode, not "train then freeze"
2. Catastrophic forgetting happens when knowledge transfer between levels breaks down
3. "Test-time training"和"test-time memorization"都是parametric ICL的实例，不是新东西

这个视角和Sutton的"Welcome to the era of experience"以及Silver/Sutton的"experience-based AI"非常契合：
- Sutton 2025 (Oak architecture keynote): https://rlj.cs.umass.edu/rlc-2025
- Silver & Sutton 2025: https://deepmind.google/discover/blog/welcome-to-the-era-of-experience/

---

## 8. Nested Learning的形式化（§3.2）

### 8.1 Update Frequency

**Definition 2**: 对任意component $A$，其frequency $f_A$定义为单位时间内更新次数。

排序算子 $\succ$:
- $A \succ B$ 如果 $f_A > f_B$，或者 $f_A = f_B$ 但$A$的state计算需要$B$的state

### 8.2 Nested System

**Definition 3**: 一个nested system有$K$个ordered levels，每个level $k$包含若干optimization problems $\{(\mathcal{L}_i^{(k)}, C_i^{(k)}, \Theta_i^{(k)})\}_{i=1}^{N_k}$，每个用GD优化：

$$\theta_{i,t+1}^{(k)} = \arg\min_{\Phi_i^{(k)}} \langle \Phi_i^{(k)} x_{t+1}, -\nabla \mathcal{L}_i^{(k)}(\theta_{i,t}^{(k)}; x_{t+1}) \rangle + \frac{1}{2\eta_{i,t+1}^{(k)}} \|\Phi_i^{(k)} - \theta_{i,t}^{(k)}\|_2^2$$

变量：
- $\theta_{i,t}^{(k)}$: 第$k$层第$i$个problem在$t$时刻的参数
- $\mathcal{L}_i^{(k)}$: 第$i$个problem的objective
- $C_i^{(k)}$: 第$i$个problem的context（数据）
- $\Theta_i^{(k)}$: 参数feasible set
- $\eta_{i,t+1}^{(k)}$: 第$k$层的learning rate

### 8.3 NSAM (Nested System of Associative Memories)

**Definition 4**: NSAM是nested system，每个optimization problem的context是key-value pairs：

$$\theta_{i,t+1}^{(k)} = \arg\min_{\Phi_i^{(k)}} \langle \Phi_i^{(k)} k_{t+1}^{(i)}, -\nabla \mathcal{L}_i^{(k)}(\theta_{i,t}^{(k)}; k_{t+1}^{(i)}, v_{t+1}^{(i)}) \rangle + \frac{1}{2\eta_{i,t+1}^{(k)}} \|\Phi_i^{(k)} - \theta_{i,t}^{(k)}\|_2^2$$

注意每个optimization problem有自己的gradient flow，所以paper也叫它们"boxes of gradient flow"。

---

## 9. Knowledge Transfer Between Levels (§3.3)

Paper讨论了几种level间的knowledge transfer方式：

### 9.1 Direct Connection (Parametric)

$$\mathcal{M}^{(0)}(\cdot) := \mathcal{M}^{(0)}(\cdot; \Theta^{(1)})$$

低频memory的forward pass依赖于高频memory的参数。

**例子**: Linear Transformer with zero initial memory，fast weight直接决定output。

### 9.2 Direct Connection (Non-Parametric)

$$\mathcal{M}^{(0)}(\cdot) := \mathcal{M}^{(0)}(\cdot; C^{(1)})$$

低频memory的forward pass依赖于高频memory的context（不是参数）。

**例子**: **Softmax Attention**！attention block的output依赖于context（key-value pairs），不是persistent parameters。

### 9.3 Knowledge Transfer via Backpropagation

两个level的参数在同一个gradient flow里：

$$\Theta_{t+1}^{(1)} = \Theta_t^{(1)} - \eta_{t+1}^{(1)} \delta_1 \hat{x}_{t+1}^\top$$
$$\Theta_{t+1}^{(0)} = \Theta_t^{(0)} - \eta_{t+1}^{(0)} \delta_0 x_{t+1}^\top$$

其中 $\delta_{t+1}^{(0)} = J_{\phi^{(0)}}(x_{t+1})^\top (\Theta_{t+1}^{(1)} \delta_{t+1}^{(1)})$

两个block共享gradient flow，但更新频率不同。

### 9.4 Knowledge Transfer via Initialization (MAML)

$$\Theta_0^{(1)} = \arg\min_\Phi \mathbb{E}_{C \sim C^{(0)}} [\ell(M^{(1)}(\cdot; \Phi), C)]$$

外层学习一个最优的initialization，让内层在new context上快速adapt。

**例子**: Meta-learned initial memory state in linear attention (TTT, Titans, Miras)。

### 9.5 Knowledge Transfer via Generation

**Weight generation** (Hypernetworks): 一个level生成另一个level的weights
**Context generation** (Optimizer case): architecture生成optimizer的context（即gradients）

**关键insight**: optimizer和architecture是**inter-connected** system。Architecture决定了optimizer看到的gradient distribution，所以"one optimizer fits all architectures"是不对的，需要architecture-specific optimizers。

---

## 10. Continuum Memory System (CMS, §7)

这是paper的核心新设计。

### 10.1 传统LSM (Long-term/Short-term Memory)的问题

传统模型分两类memory：
- Short-term: attention / hidden state (高frequency)
- Long-term: MLP weights (低frequency, pre-training时frozen)

但人脑memory是**distributed**和**continuum**的，不是二元分类。

### 10.2 CMS定义

**Formulation**: 一串MLP blocks $\text{MLP}^{(f_1)}, \text{MLP}^{(f_2)}, \ldots, \text{MLP}^{(f_k)}$，每个level有不同frequency $f_\ell$。

$$y_t = \text{MLP}^{(f_k)}(\text{MLP}^{(f_{k-1})}(\cdots \text{MLP}^{(f_1)}(x_t)))$$

参数 $\theta^{(f_\ell)}$ 每 $C^{(\ell)} = \max_i C^{(i)} / f_\ell$ 步更新一次：

$$\theta_{i+1}^{(f_\ell)} = \theta_i^{(f_\ell)} - \begin{cases} \sum_{t=i-C^{(\ell)}}^i \eta_t^{(\ell)} f(\theta_t^{(f_\ell)}; x_t) & \text{if } i \equiv 0 \pmod{C^{(\ell)}} \\ 0 & \text{otherwise} \end{cases}$$

变量：
- $f_\ell$: 第$\ell$个block的update frequency
- $C^{(\ell)}$: 第$\ell$个block的chunk size (越大frequency越低)
- $\eta_t^{(\ell)}$: 第$\ell$层的learning rate
- $f(\cdot)$: optimizer的error component (e.g., gradient)

### 10.3 三种CMS变体

**Nested CMS** (Eq 72): 各level通过meta-learning initialization传递knowledge
$$\theta_0^{(f_{s+1})} = \arg\min_\Phi \mathbb{E}_{\mathcal{T} \sim C^{(s)}} [\ell(\Theta, \mathcal{T}; \Phi)]$$

**Sequential CMS** (Eq 73): 所有level的初始化都通过backprop在最低frequency level优化
$$\theta_0^{(f_s)} = \arg\min_\Phi \mathbb{E}_{\mathcal{T} \sim C^{(1)}} [\ell(\Theta, \mathcal{T}; \Phi)]$$

**Independent (Head-wise) CMS** (Eq 74): 各level独立计算，然后aggregate
$$y_t = \text{Agg}(\text{MLP}^{(f_k)}(x_t), \text{MLP}^{(f_{k-1}}(x_t), \ldots, \text{MLP}^{(f_1)}(x_t))$$

### 10.4 CMS帮助continual learning的机制

关键insight：当MLP$^{(f_s)}$更新时，它可能"忘记"旧knowledge。但这个knowledge还在更低frequency的MLP$^{(f_{s'})}$, $s' < s$ 中。

通过backprop初始化的knowledge transfer，遗忘的knowledge可以**loop back**到MLP$^{(f_s)}$！这是一个时间维度的循环，让重要知识很难完全丢失。

### 10.5 为什么CMS不慢

- 每次只更新一小部分block（接近scheduled update time的）
- 平均update cost: $O\left(\frac{1}{\hat{f}} \times \frac{L_{\text{layer}}}{5} \times d_{in}^2\right)$
- 非更新时刻可以sequence parallelize

---

## 11. Self-Referential Titans (§8.1)

### 11.1 动机：让所有projection都adaptive

传统Transformer的问题：$W_k, W_v, W_q$在pre-training后就frozen。每个token的projection只依赖token本身和position，**不依赖context**。

例如"bank"这个词，它的projection是固定的，但"bank"在"river bank" vs "investment bank"应该有不同的embedding。靠depth来补偿这个bottleneck不够。

### 11.2 Deep Self-Referential Titans formulation

让所有projection通过各自的memory计算：

$$k_t = \mathcal{M}_{k,t-1}(x_t), \quad v_t = \mathcal{M}_{v,t-1}(x_t), \quad q_t = \mathcal{M}_{q,t-1}(x_t)$$
$$\eta_t = \mathcal{M}_{\eta,t-1}(x_t), \quad \alpha_t = \mathcal{M}_{\alpha,t-1}(x_t)$$

变量：
- $\mathcal{M}_{\sharp, t}$: 第$\sharp$种memory在$t$时刻的state，$\sharp \in \{k, v, q, \eta, \alpha, \text{memory}\}$
- $\eta_t$: learning rate (adaptive!)
- $\alpha_t$: forget gate / weight decay (adaptive!)

### 11.3 Self-Modifying: 模型生成自己的values

更进一步：让每个memory的value由模型自己生成：

$$\hat{v}_{\sharp, t} = \mathcal{M}_{\sharp, t-1}(v_t), \quad \sharp \in \{k, v, q, \eta, \alpha, \text{memory}\}$$

这里 $v_t$ 是从输入投影出来的"基础value"，但每个memory用它自己的current state来transform这个value，得到它自己用的value $\hat{v}_{\sharp, t}$。

### 11.4 Update rule

用L2 regression + Delta Gradient Descent (DGD)：

$$\mathcal{M}_{\sharp, t} = \mathcal{M}_{\sharp, t-1} (\alpha_t I - \eta_t k_t k_t^\top) - \eta_t \nabla \mathcal{L}_{\mathcal{M}_{\sharp, t-1}}(\mathcal{M}_{\sharp, t-1}; k_t, \hat{v}_{\sharp, t})$$

变量：
- $\alpha_t I$: forget gate乘以identity（weight decay）
- $-\eta_t k_t k_t^\top$: data-dependent decay（DGD的关键）
- $-\eta_t \nabla \mathcal{L}$: gradient step

Memory architecture:
$$M_\sharp(\cdot) = (\cdot) + W_{\sharp, 1} \sigma(W_{\sharp, 2} (\cdot))$$

2-layer MLP with residual connection。

### 11.5 两种objective的具体形式

**Dot-product similarity** (Hebbian-like):
$$\mathcal{M}_{\sharp, t} = \mathcal{M}_{\sharp, t-1}(\alpha_t I - \eta_t k_t k_t^\top) - \eta_t \hat{v}_{\sharp, t} k_t^\top$$

**L2 regression** (Delta rule):
$$\mathcal{M}_{\sharp, t} = \mathcal{M}_{\sharp, t-1}(\alpha_t I - \eta_t k_t k_t^\top) - \eta_t (\mathcal{M}_{\sharp, C \times \lceil t/c \rceil} k_t - \hat{v}_{\sharp, t}) k_t^\top$$

后者多了 $(\mathcal{M} k_t - \hat{v})$ 项，是prediction error，让update有state-dependent的"correction"。

---

## 12. Delta Gradient Descent (DGD, §4.5)

### 12.1 动机

标准GD用dot-product similarity作为inner objective，update与state无关：
$$W_{t+1} = W_t - \eta \nabla_y \mathcal{L} \otimes x_t$$

每个data sample独立处理。但对**sequence data**，token之间高度相关，i.i.d.假设不成立。

### 12.2 L2 regression extension

$$W_{t+1} = \arg\min_W \frac{1}{2} \|W x_t - u_t\|_2^2 + \frac{1}{2\eta_t} \|W - W_t\|_2^2$$

where $u_t = -\nabla_{y_t} \mathcal{L}(W_t; x_t)$

### 12.3 Closed-form (用Sherman-Morrison)

假设 $\|x_t\|_2 = \lambda$（normalized input），令 $\eta_t' = \eta_t / (1 + \eta_t \lambda^2)$：

$$W_{t+1} = W_t (I - \eta_t' x_t x_t^\top) - \eta_t' \nabla_{y_t} \mathcal{L}(W_t; x_t) \otimes x_t$$

注意 $(I - \eta_t' x_t x_t^\top)$ 这一项——这是**data-dependent weight decay**。

### 12.4 Intuition

当input $x_t$ 与之前的某个$x_{t'}$相似时，$(I - \eta' x_t x_t^\top)$会衰减那个方向的weights。这就是Delta rule（Prados 1989）的核心：**在写入新memory前，先擦除与新input重叠的旧memory**。

这给了sequence modeling一个天然的"管理容量"机制。Linear attention缺少这个，所以paper认为DGD在token space比标准GD更合适。

参考：
- Prados & Kak 1989 (Delta rule original): https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/el%3A19890150

---

## 13. M3 Optimizer (Multi-scale Momentum Muon, §7.2)

把CMS的思想用到optimizer上：

### 13.1 Formulation

两个momentum term + 一个second moment：

$$M_t^{(1)} = M_{t-1}^{(1)} + \beta_1 g_t \quad \text{(fast momentum, every step)}$$
$$M_t^{(2)} = M_t^{(2)} - \beta_2 \begin{cases} \sum_{i=t-\hat{C}}^t g_i & \text{if } t \equiv 0 \pmod{\hat{C}} \\ 0 & \text{otherwise} \end{cases} \quad \text{(slow momentum, every C steps)}$$
$$V_t = V_{t-1} + \beta_2 g_t^2 \quad \text{(second moment, like Adam)}$$

### 13.2 Update

$$\Theta_t = \Theta_{t-1} - \eta \frac{O_t^{(1)} + \alpha O_t^{(2)}}{\sqrt{V_t + \epsilon}}$$

where $O_t^{(i)} = \text{NewtonSchulz}_T(M_t^{(i)})$ (orthogonalize using Newton-Schulz iteration).

变量：
- $\beta_1, \beta_2, \beta_3$: EMA decay rates for三个momentum
- $\alpha$: weight for slow momentum aggregation
- $\hat{C}$: chunk size for slow momentum update
- $\epsilon$: numerical stability
- $T$: Newton-Schulz iteration steps

### 13.3 Intuition

M3 = Adam + Muon + CMS。
- Adam部分: 用$V_t$做per-parameter scaling
- Muon部分: Newton-Schulz正交化gradient direction
- CMS部分: 两个不同frequency的momentum，slow momentum"记住"了long past的gradient信息

### 13.4 实验（Figure 11, ImageNet-21K, ViT）

M3的训练/测试loss都比AdamW和Muon低。但注意，paper也警告M3在更大模型上可能有computational overhead（Figure 12显示M3比Muon慢，和AdaMuon on par）。

---

## 14. Hope Architecture (§8.3)

### 14.1 组合

**Hope = Self-Referential Titans + CMS**

Forward pass:
1. **Self-Referential Titans部分**: 所有projection通过memory计算，所有memory用L2 regression + DGD + 自生成values来update（Section 11）
2. **CMS部分**: Titans输出经过一串不同frequency的MLP blocks

```
Input → [Self-Referential Titans] → [MLP^(f_1)] → [MLP^(f_2)] → ... → [MLP^(f_k)] → Output
         (high frequency)            (low freq)    (lower freq)       (lowest freq)
```

### 14.2 Hope-Attention变体

把Self-Referential Titans换成softmax global attention + CMS。这个变体用于ablation。

### 14.3 训练

用chunk-wise parallelization（类似TTT, Titans）：
- 把sequence分成chunks of size $C$
- 在每个chunk内部，所有tokens可以并行处理
- 在chunk边界更新memory state

设 $f_\sharp = L / C_{\sharp}$ 是每个memory的update frequency，$L$是context length。

---

## 15. 实验结果分析

### 15.1 Class-Incremental Learning (Figure 6)

Datasets: CLINC (150 intent classes), Banking (77 classes), DBpedia (70 classes)

Backbone: Llama3-8B / Llama3-3B + 15B tokens continual pre-training

Baselines: ICL, EWC (Kirkpatrick 2017), InCA (Momeni 2025)

**结果**: Hope在三个数据集上都是best，包括beating InCA（带external learner的方法）。

**Key takeaway**: CMS的多level设计让knowledge在不同frequency level间loop back，缓解catastrophic forgetting。

### 15.2 Effect of Levels (Figure 7)

Tasks: MK-NIAH (RULER), LongHealth, QASPER

比较不同number of levels和不同lowest frequency的Hope变体。

**Key observations**:
1. 更多level → 更好的in-context learning
2. 更高的lowest frequency → 更adaptive但更弱的persistent memory → 性能下降
3. "Lowest Frequency = 2K"是efficiency和performance的sweet spot

### 15.3 NIAH (Table 1, multi-key/multi-query/multi-value)

| Model | MK-NIAH-1 4K | 8K | 16K | MQ-NIAH 4K | 8K | 16K |
|---|---|---|---|---|---|---|
| Transformer | 79.4 | 83.0 | 61.4 | 58.9 | 48.0 | 29.8 |
| RWKV-7 | 21.4 | 18.8 | 9.6 | 20.4 | 14.8 | 8.6 |
| Comba | 21.4 | 19.4 | 8.2 | 21.8 | 15.2 | 6.4 |
| Titans | 26.4 | 23.6 | 8.2 | 22.8 | 19.8 | 9.4 |
| **HOPE** | **29.4** | **24.8** | **14.8** | **31.7** | **24.8** | **14.2** |

Hope在attention-free模型里最好，但都还远低于Transformer。这显示softmax attention在multi-key retrieval任务上还是霸主。

### 15.4 BABILong (Figure 9)

Hope能维持性能到**10M context length**！Titans和ARMT在1M之后开始掉，GPT4/GPT4o在128K-256K就崩了。

这显示CMS的long-context能力非常强。但注意：需要fine-tuning，zero-shot效果会掉很多（paper明确说）。

### 15.5 Language Modeling (Table 2)

| Model | Wiki ppl | LMB ppl | Avg (760M) |
|---|---|---|---|
| Transformer++ | 24.18 | 24.27 | 50.11 |
| Samba | 21.07 | 22.85 | 51.46 |
| RetNet | 25.77 | 24.19 | 48.19 |
| RWKV-7 | 23.75 | 23.08 | 50.55 |
| Comba | 22.41 | 22.19 | 50.89 |
| Titans | 20.08 | 21.52 | 51.68 |
| **HOPE** | **18.68** | **20.07** | **52.28** |

在760M / 30B tokens和1.3B / 100B tokens两个scale上，Hope都是SOTA attention-free model。

### 15.6 Formal Language Recognition (Table 5)

| Model | Parity Bin0 | Parity Bin1 | (aa)* Bin0 | (aa)* Bin1 | a^n b^n |
|---|---|---|---|---|---|
| LSTM | 100 | 100 | 100 | 100 | 100 |
| Transformer | 46.4 | 0.0 | 0.0 | 0.0 | 100 |
| Linear | 78.1 | 0.0 | 0.0 | 0.0 | 100 |
| DeltaNet | 98.2 | 10.1 | 0.0 | 0.0 | 100 |
| **HOPE** | **100** | **100** | **100** | **100** | **100** |

Hope在所有formal language任务上完美。Transformer在state-tracking任务（parity）上失败，因为它的"state"是non-parametric的、被context bound的。Hope通过non-linear recurrence和self-referential update获得state-tracking能力。

### 15.7 Ablation (Table 6)

| Variant | LM ppl | Reasoning acc |
|---|---|---|
| HOPE | 12.24 | 58.1 |
| w/o DGD | 13.41 | 56.5 |
| w/o Momentum | 13.58 | 56.9 |
| w/o weight decay | 13.71 | 57.2 |
| w/o CMS | 13.04 | 57.3 |
| w/o inner-projection k | 13.77 | 56.9 |
| w/o inner-projection v | **13.90** | 55.1 |
| w/o inner-projection q | 12.19 | 57.4 |

**Key insights**:
- 移除inner-projection $v$ 最伤（13.90）→ self-generated values非常关键
- 移除inner-projection $q$ 反而略好（12.19）→ $q$可能不需要adapt?
- DGD比standard GD重要
- CMS单独贡献明显

---

## 16. 几个关键Intuition总结

### Intuition A: Architecture是"被压扁的"NL图

当我们看Transformer，我们看到attention + MLP，似乎是异构的。但在NL视角下，我们看到的都是同一个东西：linear或deep MLP，被放在不同的level、不同的frequency上。

这就像3D物体在2D上的投影——表面看很多样，本质是同一个3D物体的不同视角。

### Intuition B: 所有参数都是memory，只是频率不同

传统区分"learnable parameters / hidden states / optimizer states"是artificial的。NL说：
- Pre-training parameters: lowest frequency memory
- Optimizer momentum: medium-low frequency memory
- Hidden states / RNN memory: high frequency memory  
- Attention scores: highest frequency (∞, non-parametric)

CMS就是这个insight的直接应用——用frequency spectrum替代binary分类。

### Intuition C: Pre-training = ICL

Pre-training是lowest level的ICL，context是整个dataset。
ICL是highest level的learning。
"训练/测试"区分是artificial的。

### Intuition D: Backprop是self-referential associative memory

这个insight很深：backprop不是简单的"minimize loss"，是模型用自己生成的surprise signal训练自己。

$$v_t = f_{W_t}(x_t) = -\nabla_{y_t} \mathcal{L}(W_t; x_t)$$

模型自己生成自己的training labels，控制自己的learning。这是Schmidhuber self-referential weight matrix的现代形式。

### Intuition E: Catastrophic forgetting是compression的必然

NL说：catastrophic forgetting不是bug，是有限capacity下做compression的必然结果。CMS通过multi-level + loop back缓解，但不能完全消除——因为有限的capacity总要做trade-off。

### Intuition F: Optimizer和Architecture是inter-connected system

Optimizer看到的gradients是architecture生成的，所以optimizer的选择依赖architecture。这suggest了**architecture-specific optimizers**的方向。

---

## 17. 批判性思考

### 17.1 描述性 vs 预测性框架

NL能漂亮地统一existing methods，但它的**预测能力**如何？Hope是一个证据，但Hope的很多设计选择（2-layer MLP for memory, DGD vs other updates, number of levels, frequency schedule）并没有从NL原理严格推导出来。

### 17.2 Computational cost

Self-Referential Titans有6个memory modules + CMS有多个MLP blocks = 很多参数。Paper只到1.3B，scaling behavior不清楚。M3 optimizer已经显示有computational overhead。

### 17.3 CMS的更新trigger

"每C步更新一次"是个简单粗暴的设计。更principled的方式应该是adaptive trigger，比如基于surprise level。

### 17.4 Adam的最优性是element-wise L2 regression下的

paper自己也承认这个limitation。Element-wise更新本身就限制了表达力。Non-element-wise的AdaGrad也是associative memory，但element-wise下的"最优"不等价于general最优。

### 17.5 NL和Bayesian / PAC-Bayes的关系

NL的"multi-level context compression"和information bottleneck / minimum description length有connection，paper没有深入探讨。把这些放在一起可能能更严格地分析。

### 17.6 Pre-training = ICL的实用性

这个insight很美，但实际做continual pre-training，naive地继续训练会catastrophic forgetting + instability。Paper的Hope实验用了15B tokens continual pre-training，但这小scale。真正billion-scale的continual pre-training需要什么架构支持？paper没回答。

---

## 18. 相关work链接

- Titans (Behrouz et al. 2025): https://openreview.net/forum?id=8GjSf9Rh7Z
- Miras (Behrouz et al. 2025b): https://arxiv.org/abs/2504.13173
- Atlas (Behrouz et al. 2025): https://arxiv.org/abs/2505.23735
- TTT (Sun et al. 2024): https://arxiv.org/abs/2407.04620
- RWKV-7 (Peng et al. 2025): https://arxiv.org/abs/2503.14456
- Muon (Jordan et al. 2024): https://kellerjordan.github.io/posts/muon/
- AdEMAMix (Pagliardini et al. 2025): https://openreview.net/forum?id=jj7b3p5kLY
- Schmidhuber self-referential 1993: https://link.springer.com/chapter/10.1007/3-540-56955-8_172
- Akyürek et al. 2022 (ICL as meta-learning): https://arxiv.org/abs/2211.15661
- Von Oswald et al. 2023 (mesa-optimization in transformers): https://arxiv.org/abs/2309.05858

---

## 19. 个人评价

这篇paper在我看过的2025年的paper里属于**最ambitious**的一类。它不是做一个incremental的architecture或optimizer改进，而是给整个deep learning一个新的unified lens。

**真正的贡献**：
1. 把backprop + GD重新解读为self-referential associative memory
2. 把Adam作为某个特定L2 regression的最优associative memory
3. 把architecture和optimizer统一为NSAM的不同实例
4. CMS和Hope作为proof-of-concept

**可能的影响**：
- Continual learning: 从"训练完冻结"转向"所有level持续运转"
- Optimizer设计: architecture-specific optimizers成为新方向
- Architecture设计: 新的"depth" axis（levels / frequencies），不仅仅是layers
- Self-modifying systems: 模型生成自己的values / learning rules

**最大风险**：
- 实验scale有限（1.3B），不确定scaling laws
- 框架很多设计选择仍是heuristic
- Computational cost vs 效益的trade-off不清楚

如果NL框架真的成立，未来的deep learning可能不再是"stack more layers"，而是"design multi-frequency nested systems"。这是个方向性的转向，和Sutton的"era of experience"思路呼应。

对build intuition而言，最有用的几个mental model：
1. **Optimizer = memory of gradients**
2. **Backprop = self-referential surprise-based memory**
3. **Architecture = flattened nested optimization graph**
4. **Pre-training = ultra-long context ICL**
5. **Continual learning = multi-frequency memory management with loop-back**

希望这个详细解析对你build intuition有帮助，Andrej。如果你想深入某个具体section（比如Newton-Schulz和Muon的connection，或者chunk-wise parallelization的具体实现），可以再问。
