---
source_pdf: Equivariant Action Sampling for.pdf
paper_sha256: 8b2ff7a35f8bf42f47a384d9befda1a30cdc433125d4e05856919b7514cb0b7a
processed_at: '2026-08-04T04:55:36-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，我们抛开复杂的数学公式，用最直白的人话来 build up the intuition。

### 1. 核心痛点：瞎蒙会破坏对称性

想象你在玩飞镖，靶子是一个完美的圆形。这个靶子具有 rotation symmetry（旋转对称性），也就是说，你把靶子转90度，玩飞镖的物理规律完全没变。

在 Reinforcement Learning (RL) 里面，Continuous Control 任务（比如控制机械臂、控制小车）经常具备这种 symmetry。为了利用这种 symmetry 加速学习，我们会用 Equivariant Network，这就像是给 AI 装上了一副“对称眼镜”，让它明白“靶子转了90度，你的投掷动作也跟着转90度就行了”。

但是，我们在连续空间里选 action 时，因为动作有无数种可能，通常没法用公式直接求极值，只能用 sampling（采样）的办法去试。比如 Cross-Entropy Method (CEM) 或 MPPI，就是让 AI 先瞎蒙出一堆随机动作，然后挑效果最好的那个。

**问题就出在这个“瞎蒙”上。** 
假设 AI 蒙眼睛随机往靶子上扔了10个飞镖。就算靶子是对称的，这10个飞镖落点的几何分布绝对是对称的吗？显然不是。这10个点完全是乱七八糟的。如果此时把靶子旋转90度，你再重新蒙眼扔10个飞镖，新的10个落点绝对不会刚好是刚才那10个落点旋转90度的结果。

这就导致了一个尴尬的局面：你的神经网络虽然懂 symmetry，但你的 sampling 算法破坏了 symmetry。有限次采样带来的随机噪声，把好不容易建立的 equivariance 给打破了。这就叫做 **Weak Equivariance**（弱等变）——理论上平均下来是对称的，但实际只要采样的数量有限，就是不对称的。

### 2. G-Augmented Sampling：强制对称的“影分身之术”

为了解决采样带来的不对称，这篇 paper 提出了一个极其直觉又优雅的方法：**G-Augmented Sampling**。

既然随机采样的点不对称，那我们就强行让它对称。
假设你随机采了1个 action $a$。由于你知道这个环境有 $D_4$（90度旋转和镜像）的 symmetry group $G$，你就可以使用“影分身之术”，把这个 action $a$ 乘上 group 里的所有元素 $g$。

如果你采了1个点，经过 group $G$ 的变换，你瞬间得到了 $|G|$ 个点（比如 $D_4$ 有8个元素，你就得到8个点）。这8个点形成了一个完美的对称轨道。你把原本的 $N$ 个采样点，扩充成了 $N \times |G|$ 个点，并且在这个扩充后的集合里找最优解。

**用公式讲讲变量：**
$$ G\mathbb{A} = \{g \cdot a_i \mid g \in G\}_{i=1}^N $$
*   $\mathbb{A}$：你一开始瞎蒙出来的 $N$ 个随机动作集合。
*   $G$：环境的 symmetry group（比如旋转群）。
*   $g$：Group 里面的某个具体操作（比如转90度）。
*   $G\mathbb{A}$：扩充后的完美对称采样集合。

**Intuition:** 因为你的搜索空间被你强行“对称化”了，无论输入的 state 怎么旋转，你手里的这把“飞镖”在空间里的分布形态是一模一样的（因为轨道的封闭性）。这就在有限次采样的情况下，硬生生凑出了 **Strong Equivariance**（强等变）。由于搜索空间变小了（你只需要在 1/8 的基本空间里搜索，剩下的 7/8 靠 symmetry 白嫖），Sample efficiency 直接提升了 2-3 倍。

### 3. 在 TD-MPC 里的应用：规划未来也要对称

这篇 paper 把这个思想塞进了 TD-MPC 这个 Model-based RL 算法里。TD-MPC 的核心是用 Model Predictive Path Integral (MPPI) 来做 planning。

Planning 就像是下棋，你要往后看好几步。算法会采样出很多条 trajectory（轨迹），然后评估哪条轨迹回报最高。Paper 证明了一件事：只要你把 G-Augmented Sampling 用在 action sequence 上，同时你的 dynamics model（预测未来状态的模型）和 reward model 都是 equivariant/invariant 的，那么你整条轨迹的评估过程就是完美对称的。

如果 state 被旋转了，你采样出来的最优轨迹也会跟着旋转，毫无偏差。

### 4. 实验里的一个“坑”：物理约束比对称更重要

Paper 里有一个极其精彩的 Reacher（两连杆机械臂）实验，这能极大地 build up 我们的 intuition。

Reacher 任务有两个关节。第一个关节的角度是全局的，如果你旋转整个环境，第一个关节的角度也该跟着旋转。但是第二个关节是连在第一个关节上的，它的角度是**相对的**，它是物理上的一个 kinematic constraint（运动学约束）。在全局旋转下，第二个关节的相对角度是不变的。

一开始，研究人员想追求极致的 symmetry，把第二个关节的角度也当作“会随全局旋转而旋转”的特征输入给网络。结果发现性能大幅下降！
**直觉解释：** 把相对角度当作全局旋转量，相当于强行要求网络去学习“当世界旋转时，机械臂的骨头也要扭曲”这种违背物理常识的事情。这就告诉我们，**Geometric Deep Learning 必须向物理 Kinematics 低头**。Symmetry 的 representation assignment 错了，比没有 symmetry 还糟糕。

### 总结

这篇 paper 的核心 intuition 就是：**靠神经网络架构硬凑出来的 symmetry 是不够的，靠采样算法强行凑出来的 symmetry 才是铁打的**。通过给随机采样的动作加上“影分身之术”，我们在 continuous control 的 planning 里实现了真正的强等变，既节省了算力，又提升了泛化性。

### References for further reading
如果你想看看具体的实验结果图或者源码，可以参考以下链接：
1. **Equivariant Action Sampling Paper**: [arXiv:2307.08226](https://arxiv.org/abs/2307.08226)
2. **TD-MPC Base Algorithm**: [arXiv:2203.04955](https://arxiv.org/abs/2203.04955)
3. **Equivariant Network Library (escnn)**: [GitHub - QUVA-Lab/escnn](https://github.com/QUVA-Lab/escnn)
4. **Implicit Behavioral Cloning (Coordinate Regression origin)**: [arXiv:2109.00137](https://arxiv.org/abs/2109.00137)

---

Hello Andrej！这篇 paper 《Equivariant Action Sampling for Reinforcement Learning and Planning》 的核心 intuition 极其精妙，它精准击中了 model-based RL 和 sampling-based planning 中的一个盲点：**random sampling 会破坏 physical symmetry**。

在 continuous control 中，我们常常通过 Cross-Entropy Method (CEM) 或 Model Predictive Path Integral (MPPI) 在 action space 里做 sampling。如果 environment 本身具备 symmetry（例如 2D 平面旋转 $SO(2)$ 或 dihedral group $D_4$），传统 sampling 方法即使搭配了 equivariant neural network，也无法在 finite sample 数量下保持严格的 equivariance。这篇 paper 提出了 G-augmented sampling，通过在 group orbit 上强行扩充 samples，实现了 strong equivariance，从而把 sampling-based planning 的 sample efficiency 提升了 2-3 倍。

下面我为你详细拆解其中的 technical details 和 mathematical formulations，并 build up the intuition。

### 1. 为什么 Random Sampling 会 Break Symmetry？

在 Geometric MDP (GMDP) 中，state $s$ 和 action $a$ 都在 Euclidean space 中，且受到 symmetry group $G$ 的作用。MDP 的 transition 和 reward 满足严格的 symmetry 约束：

$$
P(s' \mid s, a) = P(g \cdot s' \mid g \cdot s, g \cdot a) \tag{1}
$$
$$
R(s, a) = R(g \cdot s, g \cdot a) \tag{2}
$$

**公式变量解释：**
*   $P$: Transition probability function。
*   $R$: Reward function。
*   $s, a, s'$: Current state, action, next state。
*   $g$: 任意一个属于 symmetry group $G$ 的元素（例如一个 90 度旋转矩阵）。
*   $g \cdot s$: Group element $g$ 作用在 state $s$ 上的映射（通过 group representation $\rho_S(g)$ 实现）。

传统做法是设计一个 equivariant Q-network $Q_\theta(s, a)$。在单步的 action selection 中，我们要通过采样来寻找最优 action：$a_0 = \arg\max_a Q(s_0, a)$。通常我们用 CEM 从一个 Gaussian distribution $\mathcal{N}(\mu, \sigma^2 I)$ 里采样出 $m$ 个 actions $\mathbb{A} = \{a_i\}_{i=1}^m$。

**Intuition behind the failure:** 尽管 Gaussian distribution $\mathcal{N}(\mu, \sigma^2 I)$ 本身在连续空间下是 isotropic 的，理论上具有 $O(d)$ invariance。但是，你具体抽出来的 finite sample set $\{a_1, a_2, ..., a_m\}$ 是一堆离散的、带有随机噪声的点。如果你把 state $s_0$ 旋转 $g$，由于采样随机性，你从同一个 Gaussian 里抽出来的新样本集 $\{a_1', a_2', ..., a_m'\}$ 绝对无法刚好等于 $\{g \cdot a_1, g \cdot a_2, ..., g \cdot a_m\}$。这就导致了有限步的 sampling approximation 打破了网络原本拥有的 equivariance 特性。这种破坏使得算法在测试时的表现极度不稳定，也无法 generalize 到 unseen 的 symmetric configurations。

### 2. Weak Equivariance vs. Strong Equivariance

Paper 中非常 mathematically 严格地区分了这两种概念。假设我们要估计一个函数 $f(x) = \mathbb{E}_\omega[q(x, \omega)]$，其中 $\omega$ 是随机变量。我们已知 $f(x)$ 满足 $G$-equivariance 约束：$f(g \cdot x) = \rho(g^{-1}) f(x)$。

**Weak Equivariance (弱等变):**
$$
\forall g \in G, \quad \mathbb{E}_\omega[q(g \cdot x, \omega)] = \rho(g^{-1}) \mathbb{E}_\omega[q(x, \omega)] \tag{3}
$$
**公式解释:** 期望层面上满足 equivariance，但是在有限样本下 $\hat{f}(x) = \frac{1}{m} \sum_{i=1}^m q(x, \omega_i)$ 不满足。传统使用 equivariant network 加上 naive sampling 的方法就是 weak equivariance。

**Strong Equivariance (强等变):**
$$
\forall g \in G, \quad q(g \cdot x, \omega) = \rho(g^{-1}) q(x, \omega) \tag{4}
$$
**公式解释:** 这意味着对于任何一个具体的样本 $\omega$，这个映射都严格满足 equivariance。无论你采了多少个样本，求平均后的结果必然也是 equivariant 的。

### 3. G-Augmented Sampling: 核心方法解析

为了达到 Strong Equivariance，作者提出了一种极其直观但 mathematically elegant 的方法：G-augmented sampling。

假设我们在 single step 下采样 $N$ 个 actions，记为集合 $\mathbb{A} = \{a_i\}_{i=1}^N$。我们不直接在这个集合上做 argmin，而是把每个 action 乘以所有的 group elements $g \in G$，生成 augmented sample set：

$$
G\mathbb{A} = \{g \cdot a_i \mid g \in G\}_{i=1}^N
$$

然后在这个扩充后的集合上寻找最优 action：

$$
a_0 = \arg\min_{a \in G\mathbb{A}} E(s_0, a) \tag{5}
$$

**为什么这保证了 Strong Equivariance？**
如果我们将 state $s_0$ 旋转 $g$，变成 $g \cdot s_0$。因为我们的 sample set $G\mathbb{A}$ 已经包含了 group $G$ 的所有 orbit，所以对于新的 state，我们用的 sample set 其实还是 $G\mathbb{A}$（因为 group 具有封闭性，对已经在 orbit 里的点再作用 $g$，集合还是那个集合）。
由于 energy function $E$ 是 $G$-invariant 的（即 $E(g \cdot s, g \cdot a) = E(s, a)$），我们有：

$$
g \cdot a_0 = g \cdot \arg\min_{a \in G\mathbb{A}} E(s_0, a) = \arg\min_{a \in G\mathbb{A}} E(g \cdot s_0, a) \tag{5推导}
$$

**Intuition:** 这相当于在 action space 上施加了一个 group convolution。我们强行把 random Gaussian noise 沿着 symmetry group 的流形“拉扯”成了完美的 symmetric distribution。这让我联想到 Statistical Mechanics 里面的 Haar measure，如果你想让一个积分保持 group invariance，你的 measure 必须在 group orbit 上是 uniform 的。G-augmented sampling 本质上就是构建了一个离散的、在 group orbit 上 uniform 的经验测度。

### 4. Equivariant TD-MPC 架构解析

Paper 将这个思想扩展到了 multi-step planning，具体实现了一个 Equivariant TD-MPC。TD-MPC 使用 MPPI 算法在 latent space 里 rollout trajectories。

一条轨迹定义为 $\tau_i = (s_t, a_t, s_{t+1}, a_{t+1}, \dots, s_{t+H})$。轨迹的 return 计算公式为：

$$
\mathbf{return}(\tau) = \mathbb{E}_\tau \left[ \gamma^H Q_\theta(s_H, a_H) + \sum_{t=0}^{H-1} \gamma^t R_\theta(s_t, a_t) \right] \tag{10}
$$

**公式变量与上下标解释：**
*   $\tau$: 一条采样得到的 trajectory。
*   $\mathbb{E}_\tau$: 对 trajectory 分布求期望。
*   $\gamma$: Discount factor（折扣因子），通常在 $(0, 1)$ 之间（如 0.99）。
*   $H$: Planning horizon，即我们向前预测的步数。作为上标，$\gamma^H$ 表示第 $H$ 步的 discount 权重。
*   $t$: Time step，作为下标，$s_t$ 表示第 $t$ 步的 state，$a_t$ 表示第 $t$ 步的 action。
*   $Q_\theta$: Parameterized value function（由参数 $\theta$ 组成的 Q-network）。
*   $R_\theta$: Parameterized reward function。

为了使整个 planning 过程 equivariant，TD-MPC 里的所有 modules 都必须满足 symmetry 约束：
1.  **Dynamics model** $f_\theta$: 需要 $G$-equivariant。
    $$ \rho_S(g) \cdot f_\theta(s_t, a_t) = f_\theta(\rho_S(g) \cdot s_t, \rho_A(g) \cdot a_t) \tag{6} $$
2.  **Reward model** $R_\theta$ 和 **Value model** $Q_\theta$: 需要 $G$-invariant（因为它们输出标量，对应 trivial representation $\rho_{tri}(g) = 1$）。
    $$ Q_\theta(s_t, a_t) = Q_\theta(\rho_S(g) \cdot s_t, \rho_A(g) \cdot a_t) \tag{8} $$
3.  **Policy model** $\pi_\theta$: 需要 $G$-equivariant。
    $$ \rho_A(g) \cdot \pi_\theta(\cdot \mid s_t) = \pi_\theta(\cdot \mid \rho_S(g) \cdot s_t) \tag{9} $$

在 MPPI 的 trajectory sampling 阶段，作者把 single-step 的 G-augmented sampling 推广到了 sequence 采样上。对一条 sampled action sequence $(a_t, \dots, a_{t+H})$，使用 group $G$ 对整条 sequence 进行作用，生成 $G \cdot \tau$。由于 $Q_\theta$ 和 $R_\theta$ 都是 invariant 的，且 dynamics $f_\theta$ 是 equivariant 的，因此 return 也是 invariant 的：
$$ \mathbf{return}(g \cdot \tau) = \mathbf{return}(\tau) \tag{12} $$
在 MPPI 选取 top-K trajectories 的时候，如果用 G-augmented sampling，K=1 时严格满足 equivariance。这就保证了 planning 阶段的完美 symmetry preservation。

### 5. 实验直觉与 Reference Frame 的陷阱

Paper 里有一个实验非常能 build intuition：Coordinate Regression problem。
作者用一个 EBM (Energy-Based Model) 预测图像中 marker 的 $(x, y)$ 坐标。如果只在图像的第一象限训练，传统的 CEM 在测试第二、三、四象限时完全 fail，因为它的 energy landscape 是 lumpy 且 asymmetry 的，无法 extrapolate。
但是，如果使用 equivariant EBM 配合 G-augmented sampling，即使只在第一象限训练，模型也能在其他三个象限给出完美的 prediction！因为 G-augmented sampling 强制把 CEM 的搜索过程绑定在了 $D_4$ group 的 orbit 上。

**另外一个极其深刻的实验是关于 Reacher 的 Reference Frame 选择：**
在 Reacher task 中，机械臂有两个 joint。第一个 joint 的绝对角度 $\theta_1$ 在全局坐标系下随旋转改变，属于 standard representation $\rho_{std}(g)$。而第二个 joint 相对于第一个 joint 的相对角度 $\theta_2$ 是一个 **kinematic constraint**，在全局旋转下是不变的，它应该属于 trivial representation $\rho_{tri}(g) = 1$。
如果强行用 global reference frame，把第二个 joint 的位置也当作 standard representation（即随旋转改变），算法性能反而会大幅下降！
**Intuition:** 这告诉我，equivariance 不是盲目套用 group theory 就能奏效的。它必须与 physical kinematics 精确对齐。Representation assignment 错误会破坏网络 respect 物理定律的能力，导致 generalization 下降。这是 geometric deep learning 在 robotics 应用中最容易踩坑的地方。

### 6. 扩展联想与 Reference Frame 的直觉

这种 G-augmented sampling 的思想本质上与 Steerable CNNs 和 Equivariant Graph Neural Networks 中的 kernel constraint 求解有异曲同工之妙。在 E(2)-Equivariant CNNs (如 *escnn* library) 中，为了满足 equivariance，网络第一层的 filter 必须被 parameterized 为特定的 Bessel functions 或 harmonics，等价于在 filter space 上做 group orbit 上的 projection。

G-augmented sampling 则是在 **sampling space** 上做 group projection。如果 $G$ 是一个 continuous group（例如 $SO(2)$），理论上我们无法穷举 orbit，但 paper 中采用 discrete subgroups（如 $D_4, D_8, C_8$, 或者 3D 下的 icosahedral group with order 60）来近似。这相当于在 continuous Haar measure 上做了 Monte Carlo discretization。从 Path Integral Control 的角度看，这种 augmentation 就相当于把系统的 Lagrangian 显式地在 group space 里平均化，得到了一个 symmetrized path integral。

### References & Web Links for Deep Dive

如果你想进一步深挖其中的 math 和 implementation details，这里有一些相关的 references：

1.  **The Paper itself (arXiv preprint)**:
    [Equivariant Action Sampling for Reinforcement Learning and Planning (arXiv)](https://arxiv.org/abs/2307.08226)
2.  **TD-MPC (Base algorithm used in the paper)**:
    [TD-MPC: Temporal Difference Learning for Model Predictive Control (arXiv)](https://arxiv.org/abs/2203.04955)
3.  **Equivariant Neural Networks Library (used for implementation)**:
    [escnn: General E(2)-Equivariant Steerable CNNs (GitHub)](https://github.com/QUVA-Lab/escnn)
4.  **Implicit Behavioral Cloning (The coordinate regression testbed origin)**:
    [Implicit Behavioral Cloning (arXiv)](https://arxiv.org/abs/2109.00137)
5.  **Geometric Deep Learning fundamentals**:
    [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges (arXiv)](https://arxiv.org/abs/2104.13478)
6.  **Symmetric Embeddings for Equivariant World Models (Related theory on latent space symmetry)**:
    [Learning Symmetric Embeddings for Equivariant World Models (arXiv)](https://arxiv.org/abs/2204.11371)

总结来说，这篇 paper 给了我们一个很强的 insight：在涉及 sampling 的 model-based RL 中，仅仅让 network equivariant 是不够的，sampling 的 procedure 本身也必须被 algebraically 约束。只有 strong equivariance 才能在 finite sample regime 下真正激发出 symmetry 带来的 generalization power。
