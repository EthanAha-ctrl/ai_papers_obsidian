---
source_pdf: Equivariant Diffusion Policy.pdf
paper_sha256: 3e4ed85a4bae7b5322f5e2d8fb00461ee436a31a574d030c68d4aed3c9593c1f
processed_at: '2026-08-04T04:58:58-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的 Andrej，我们来剥开学术包装，用最直白的 human language 聊聊这篇 paper 到底在搞什么鬼，以及它为什么 work。

### 1. 核心痛点：Diffusion Policy 太吃数据了

现在的机器人 imitation learning 基本都在用 Diffusion Policy。为什么用 Diffusion？因为人类示教数据存在 multimodality（同一个状态下，你可以往左抓，也可以往右抓）。传统的 regression 模型遇到这种情况会把动作平均化，导致机器人啥也干不了。Diffusion model 通过学习一个 noise prediction function $\varepsilon_\theta(\mathbf{o}, \mathbf{a}^k, k)$，本质上是在学一个 score field（梯度场），能把多模态的 action distribution 完美拟合出来。

但是，学习这个 score field 比学一个简单的 mapping 难多了。网络需要搞明白：在任意给定的 observation $\mathbf{o}$ 下，对于任意加了噪的 action $\mathbf{a}^k = \mathbf{a} + \varepsilon^k$，它该往哪个方向去噪。这导致标准 Diffusion Policy 是个 data hog，没几百条示教数据根本训不出来。

### 2. 核心直觉：把物理常识焊死在网络里

机器人的 world 是有物理对称性的。如果桌子上的物体、机器人的 base，连同摄像头整体绕 z 轴（重力轴）旋转了 90 度，那么专家示教的轨迹也应该整体旋转 90 度。这就是 SO(2) symmetry。

普通的神经网络不懂这个。它看到 0 度的抓取轨迹和 90 度的抓取轨迹，会当成两件完全不同的事情去学。它必须看遍各个角度的示教数据，才能在 weight space 里勉强拟合出这种旋转规律。

**这篇 paper 的核心 intuition 就是：既然我们知道这个 symmetry 一定存在，为什么要让网络去 data 里 hard-learn？直接用 equivariant network 把它硬编码进网络架构里。**

如果 expert policy $\pi(\mathbf{o}) = \mathbf{a}$ 是 SO(2)-equivariant 的（即输入旋转 $g\mathbf{o}$，输出也旋转 $g\mathbf{a}$），那么 paper 证明了，diffusion model 要学的那个 noise prediction function $\varepsilon(\mathbf{o}, \mathbf{a}^k, k)$ 也必然是 equivariant 的：
$$ \varepsilon(g\mathbf{o}, g\mathbf{a}^k, k) = g\varepsilon(\mathbf{o}, \mathbf{a}^k, k) $$
**公式变量解析：**
*   $g \in \mathrm{SO}(2)$: 绕 z 轴的旋转矩阵。
*   $\mathbf{o}$: Observation（图像/Voxel + 机器人状态）。
*   $\mathbf{a}^k$: 加了噪的 action，上标 $k$ 表示 diffusion 的第 $k$ 步去噪。
*   $\varepsilon$: 网络预测的 noise。

这从数学上保证了：如果你把输入图像和 noisy action 旋转一下，网络预测出来的去噪方向（score field）也会跟着刚性旋转。网络直接免修了“旋转对应规律”这门课，0成本获得 generalization，sample efficiency 爆炸式提升。

### 3. 技术难点：6DoF Action 怎么转？

要把这个 intuition 落地，必须定义清楚 6DoF 的 action $\mathbf{A}_t \in \mathbb{R}^{4\times4}$ 在旋转 $g$ 时怎么变。Paper 分了两种情况：

**Absolute Control (Position Control)：**
下一帧 pose 直接等于 action：$T_{t+1} = \mathbf{A}_t$。世界旋转 $g$，就是左乘 $T_g$：$g\mathbf{A}_t = T_g \mathbf{A}_t$。
如果把矩阵按列展开成向量 $\mathbf{a}_t = \mathrm{Vec}_c(\mathbf{A}_t)$，它的 group action 就是：
$$ g\mathbf{a}_t = (\rho_1 \oplus \rho_0^2)^4(g)\mathbf{a}_t $$
**公式拆解：**
*   $\rho_1$: 频率为 1 的 irreducible representation，本质上就是那个 2x2 的旋转矩阵 $\begin{pmatrix} \cos g & -\sin g \\ \sin g & \cos g \end{pmatrix}$，作用在 x,y 坐标上。
*   $\rho_0^2$: 频率为 0 的 representation（就是常数 1），作用在 z 坐标和齐次项上（因为它们不随平面旋转改变）。
*   $\oplus$: Direct sum，把这些变换拼成一个 block diagonal matrix。
*   $^4$: 因为 4x4 矩阵有 4 列，所以这个组合重复 4 次。

**Relative Control (Velocity Control)：**
下一帧 pose 是当前 pose 乘上 action：$T_{t+1} = \mathbf{A}_t T_t$。这里有个坑：世界旋转 $g$ 时，当前的 base pose $T_t$ 也变了，所以 $\mathbf{A}_t$ 的变换变成了 conjugation（共轭）：$g\mathbf{A}_t = T_g \mathbf{A}_t T_g^{-1}$。
按行展开后，求解这个线性变换 $\rho_{\mathbf{A}}$ 会发现里面包含了 $\cos(2g)$ 和 $\sin(2g)$ 的项（因为矩阵乘法引入了二倍角公式）。必须找到一个 change-of-basis matrix $P$，把它 block diagonal 化为纯洁的 irreducible representations：
$$ g\mathbf{a}_t = P^{-1} \left[ (\rho_0^6 \oplus \rho_1^4 \oplus \rho_2)(g) \right] P\mathbf{a}_t $$
*   $\rho_2$: 频率为 2 的 representation，也就是旋转角变成了 $2g$。这非常 elegant，它揭示了在 relative pose 下的 conjugation 本质上激发了二倍频的旋转特征。网络内部在纯数学的 irreducible 空间里学习，输出时再通过 $P^{-1}$ 变回现实世界的 SE(3)。

### 4. 架构的巧妙：Regular Representation 的 Hack

怎么把这个 equivariance 约束塞进复杂的 1D Temporal U-Net 里？如果强行把卷积核换成 steerable filter，工程极其难写。

作者用了个非常聪明的 hack：**Regular Representation**。
他们把网络限制在离散子群 $C_8$ 上（8 个离散旋转，每 45 度一个）。在 $C_8$ 下，Regular Representation 本质上就是 **cyclic permutation（循环移位）**。

网络把 observation 和 action 编码后，特征图的 shape 是 $u \times d$（$u=8$）。你可以把这 8 个 channel 看作是 8 个不同旋转角度下的“视角特征”。
当输入图像旋转了 45 度时，这 8 个 channel 的特征仅仅是在 channel 维度上 cyclic shift 了一位！

在 Denoising 阶段，作者不用改写复杂的 U-Net。他直接把特征按 group element slice 开：
$$ z^g = U(e_{\mathbf{o}}^g, e_{\mathbf{a}^k}^g, k) $$
让这 8 个 slice 共享同一个标准的 1D Temporal U-Net $U$ 跑一遍。因为 weight sharing 是在 group orbit 上发生的，所以网络输出天然 equivariant。这就跟 CNN 在 pixel grid 上 share weight 保证 translation equivariance 是一个道理，只不过这里是 group orbit 上的 weight sharing。

### 5. 实验数据与直觉验证

在 MimicGen 的 12 个仿真任务上，这个架构的威力彻底爆发：

| Method | Obs | 100 Demos Avg Success | 1000 Demos |
| :--- | :--- | :--- | :--- |
| **EquiDiff (Voxel)** | Voxel | **63.9% (+21.9)** | 77.9% |
| DiffPo-C (Baseline) | RGB | 42.0% | 71.4% |
| DP3 | Point Cloud | 23.9% | 56.8% |

**直觉解读：**
1.  **Low-data regime 屠杀：** 仅用 100 个 demos，EquiDiff 比原版 Diffusion Policy 成功率高 21.9%。
2.  **Data Efficiency 逆天：** EquiDiff 用 200 个 demos 训出来的效果（72.6%），比原版用 1000 个 demos 训出来的效果（71.4%）还要好。因为原版网络要花大量参数去 learn 旋转不变性，而 EquiDiff 把这部分 capacity 全省下来去学 task 的 temporal logic 了。
3.  **Voxel 优于 RGB：** 为什么强调 Voxel？因为 RGB 相机是 perspective projection（透视投影），如果相机不是正交正上方往下看，物体转了 90 度，它在 2D 图像上的形变不是严格的 2D 旋转。这叫 symmetry mismatch。而 Voxel 是在 metric 3D space 下构建的，物体绕 z 轴转，Voxel grid 就是刚性旋转，严格满足 SO(2) symmetry。所以要 fully exploit equivariance，输入也要是 geometrically correct 的。

在真实机器人实验中，58 个 demos 就教会了机械臂完成“开烤箱 -> 拉托盘 -> 拿 bagel -> 放进去 -> 关托盘 -> 关烤箱”这种极长 horizon 的复杂任务，成功率 80%，而 baseline 只有 10%。

### 6. Hallucinations 与更深的联想

这篇 paper 给我很多发散的联想，对于你 Andrej 这种搞 foundation model 的人，这些点可能更有意思：

**Data Augmentation vs Equivariant Architecture 的本质区别**
Data Augmentation 是一种 soft inductive bias。你给网络看旋转后的数据，告诉它 loss 要小，网络通过梯度下降去拟合。但这只是 statistical enforcement，网络依然可以在庞大 hypothesis space 里找到不平滑的解。Equivariant Architecture 是 hard inductive bias，如果输出不满足 equivariance，在数学上根本无法从网络里 generate 出来。在极小数据集下，Data Augmentation 容易 overfit 到 augmented distribution，而 Equivariant Network 能做到 theoretically guaranteed generalization。Table 9 的 ablation study 证明了这点：CNN + Aug 比 plain CNN 好，但依然被 Equivariant Net 吊打。

**AlphaFold 2 与 SE(3)-Equivariance**
这篇 paper 处理的是 SO(2)（绕 z 轴转），因为桌面上有重力，z 轴是特殊的。但在蛋白质结构预测里，空间是各向同性的。AlphaFold 2 和 Equivariant Diffusion for Molecule Generation 用的是 full SE(3)-equivariance。这引出一个直觉：**物理空间的对称性决定了网络架构的最优解**。如果未来搞 mobile manipulator（机器人到处跑，没有固定的桌面和重力方向约束），这种 SO(2) 就不够了，可能需要 frame-conditioned SE(3) 或者 gauge equivariant networks，把 local reference frame 的变换也内化进网络结构。

**VLA (Vision-Language-Action) Models 的 Geometric Head**
现在大家都在搞 VLA（比如 RT-2, Octo），用 LLM/VLM 当 backbone 处理 language 和 vision。Language 是离散的，没有 spatial equivariance。但是 action 输出端依然受物理空间约束。
如果纯靠 LLM 的 token 去回归 action coordinates，它对 spatial generalization 是很弱的。如果把 LLM 当作提取 invariant semantics 的 trunk，在 action head 接一个 Equivariant Diffusion Head，把语言指令和 spatial coordinates 解耦，语言管“做什么”，equivariant head 管“在什么朝向下怎么做”。这可能极大地加速 VLA 在 few-shot manipulation 上的收敛。

**Latent Diffusion 与 Score Field 的几何结构**
Diffusion 学的是 score function $\nabla \log p(\mathbf{a}|\mathbf{o})$。这个 vector field 在高维空间里非常复杂。Equivariant constraints 其实是在告诉网络：这个 high-dimensional vector field 在 group action 下是刚性闭合的。这就像在一个极度复杂的流形上，我们先验地知道它具有某种对称性，那么用对称性约束的基函数去拟合它，比用无约束的 MLP 去拟合要高效无数倍。如果我们能在 latent space 里也定义出严格的 group representations，做 Equivariant Latent Diffusion，计算开销和对数据量的需求可能会进一步指数级下降。

**Incorrect Equivariance 的边界**
Paper 里提了一嘴 limitation：如果 task 本身不满足对称性，硬套 equivariance 会损害性能（参考他们另一篇 paper [57]）。比如倒水 task，杯子倾斜的角度跟重力强相关，绕 z 轴转 90 度，倒水的物理过程完全变了。如果这时候强行用 SO(2)-equivariant network，网络会被物理现实打脸。未来的方向可能是 Adaptive Equivariance，网络自己学一个 gating 来决定当前 task 到底有多 equivariant，在保持 inductive bias 和 fitting physical reality 之间动态平衡。

---

你好 Andrej，非常荣幸能与你探讨这篇 paper。这篇 Equivariant Diffusion Policy 把 Geometric Deep Learning 的 group equivariance 理论非常优雅地注入到了当前 robotics imitation learning 领域极具统治力的 Diffusion Policy 框架中。从直觉上讲，diffusion model 本质上是在学习一个 score function（即 data manifold 上的 gradient field），如果 task 本身具备 spatial symmetry，那么这个 gradient field 也必然具备对应的 symmetry。通过 equivariant neural network 在网络架构层面硬编码这种 symmetry，可以极大缩小 hypothesis space，从而在极少数据的情况下实现极强的 generalization。

下面我将从 core intuition、mathematical foundations、architecture dissection、experimental data 以及 broader related associations 进行详尽的剖析。

---

### 1. Core Intuition 与 Problem Formulation

在 behavior cloning (BC) 中，我们的目标是学习一个 expert policy $\pi: \mathbf{o} \mapsto \mathbf{a}$。由于 human demonstration 往往存在 multimodality（对于同一个 observation，可能有多个合理的 action），传统的 MSE regression 会导致 averaging effect。Diffusion Policy 通过 DDPM (Denoising Diffusion Probabilistic Models) 学习一个 noise prediction function $\varepsilon_\theta(\mathbf{o}, \mathbf{a}^k, k)$ 来拟合分布，巧妙解决了 multimodality 问题。

但是，学习这个 denoising function 比学习一个 explicit policy 复杂得多。对于任意 state-action pair $(\mathbf{o}, \mathbf{a})$，网络需要为所有可能的 diffusion step $k$ 和 Gaussian noise $\varepsilon^k$ 建立映射。在 3D robotic manipulation 中，world 存在极强的 SO(2) symmetry（即绕 gravity axis / z-axis 的旋转不变性）。如果桌子上的物体和机器人整体绕 z 轴旋转了角度 $g$，那么 expert 采取的 action 也应该相应地旋转 $g$。

**Proposition 1** 给出了核心理论保障：
如果 expert policy $\pi$ 是 SO(2)-equivariant 的，即 $\pi(g\mathbf{o}) = g\pi(\mathbf{o})$，那么 ground truth noise prediction function $\varepsilon(\mathbf{o}, \mathbf{a}^k, k)$ 也是 SO(2)-equivariant 的：
$$ \varepsilon(g\mathbf{o}, g\mathbf{a}^k, k) = g\varepsilon(\mathbf{o}, \mathbf{a}^k, k) $$

**公式变量与上下标解析：**
*   $\varepsilon$: Ground truth noise prediction function。
*   $g \in \mathrm{SO}(2)$: Group element，代表绕 z 轴的一个旋转矩阵。
*   $\mathbf{o}$: Observation（包含 visual input 和 gripper pose）。
*   $\mathbf{a}^k = \mathbf{a} + \varepsilon^k$: 加噪后的 action sequence，其中 $\mathbf{a}$ 是 clean action，$\varepsilon^k$ 是 step $k$ 对应的 Gaussian noise。
*   $k$: Diffusion step (上标 $k$ 表示该变量是在 diffusion step $k$ 时的状态)。

**Proof Intuition:** 
观察推导过程：$\varepsilon^k = \varepsilon(\mathbf{o}, \pi(\mathbf{o}) + \varepsilon^k, k)$。当输入 $g\mathbf{o}$ 时，由于 $\pi$ 是 equivariant 的，clean action 变为 $g\pi(\mathbf{o}) = g\mathbf{a}$。由于 noise $\varepsilon^k$ 也是按照同样的 group representation 进行变换的，利用 linearity $g\mathbf{a} + g\varepsilon^k = g(\mathbf{a} + \varepsilon^k) = g\mathbf{a}^k$，可以直接得出 $\varepsilon$ 对 $g\mathbf{o}$ 和 $g\mathbf{a}^k$ 的预测结果必然是 $g\varepsilon^k$。这从数学上保证了 diffusion 的 score field 旋转后完全对应。

---

### 2. Mathematical Dissection: SO(2) Representation on 6DoF Action

要把上述理论落地，最大的挑战是如何定义 action $\mathbf{a}_t$ 在 SO(2) 作用下的变换规则。机器人的 6DoF action 是一个 SE(3) pose matrix $\mathbf{A}_t \in \mathbb{R}^{4 \times 4}$。Paper 中探讨了两种 control mode 下的 representation 分解：

#### Absolute Control (Position Control)
在 absolute control 下，下一时刻的 pose 直接等于 action：$T_{t+1} = \mathbf{A}_t$。
当 world 绕 z 轴旋转 $g$ 时，$\mathbf{A}_t$ 的变换为左乘 $T_g$：$g\mathbf{A}_t = T_g \mathbf{A}_t$。
将 $\mathbf{A}_t$ 按列展开 $\mathbf{a}_t = \mathrm{Vec}_c(\mathbf{A}_t) = [\mathbf{A}_t^{1T}, \mathbf{A}_t^{2T}, \mathbf{A}_t^{3T}, \mathbf{A}_t^{4T}]^T$。
由于 $T_g$ 对 x,y 坐标做 2D rotation，对 z 坐标做恒等变换，其 irreducible representation 为：
$$ \rho_1(g) = \begin{pmatrix} \cos g & -\sin g \\ \sin g & \cos g \end{pmatrix}, \quad \rho_0(g) = 1 $$
因此，对于按列展开的 $\mathbf{a}_t$，其 group action 为：
$$ g\mathbf{a}_t = (\rho_1 \oplus \rho_0^2)^4(g)\mathbf{a}_t $$
**公式解析：**
*   $\rho_1$: 频率为 1 的 irreducible representation，作用在 x,y 两个维度上。
*   $\rho_0^2$: 频率为 0（trivial）的 representation 重复 2 次，作用在 z 坐标和齐次坐标的常数项 1 上。
*   $(\dots)^4$: 表示这种 block diagonal representation 重复 4 次，因为按列展开有 4 列。
*   $\oplus$: Direct sum，将不同的 representation block 拼成一个 block diagonal matrix。

为了简化计算并保证网络输出能被映射回合法的 SE(3) 空间，作者采用了 6D rotation representation [49]（去掉齐次项常数）。最终的 action vector 维度为 10（6D rotation + 3D translation + 1D gripper width）：
$$ g\mathbf{a}_t = (\rho_1^3 \oplus (\rho_1 \oplus \rho_0) \oplus \rho_0)(g)\mathbf{a}_t $$

#### Relative Control (Velocity Control)
在 relative control 下，$T_{t+1} = \mathbf{A}_t T_t$。此时 world 旋转 $g$ 同时作用于 current pose $T_t$ 和 relative pose $\mathbf{A}_t$，导致 $\mathbf{A}_t$ 的变换为 conjugation（共轭）：
$$ g\mathbf{A}_t = T_g \mathbf{A}_t T_g^{-1} $$
由于按行展开 $\mathbf{a}_t = \mathrm{Vec}_r(\mathbf{A}_t)$，我们需要找到一个 16x16 的 matrix $\rho_{\mathbf{A}}$ 使得 $\rho_{\mathbf{A}}(g) \mathrm{Vec}_r(\mathbf{A}_t) = \mathrm{Vec}_r(T_g \mathbf{A}_t T_g^{-1})$。
通过代数求解，$\rho_{\mathbf{A}}$ 会包含 $\cos(2g)$ 和 $\sin(2g)$ 的项（因为矩阵乘法中引入了双角公式，例如 $c^2 - s^2 = \cos(2g)$）。
通过寻找 change-of-basis matrix $P$，可以将其 block diagonal 化为 irreducible representations：
$$ g\mathbf{a}_t = P^{-1} \left[ (\rho_0^6 \oplus \rho_1^4 \oplus \rho_2)(g) \right] P\mathbf{a}_t $$
**公式解析：**
*   $\rho_2$: 频率为 2 的 irreducible representation。这非常 elegant，它说明在 relative pose 下的 conjugation operation 本质上蕴含了二倍频的旋转特征。如果将 SO(3) 分解到 SO(2) 子群下，这种 conjugation 对应于 SO(3) 的 adjoint representation。
*   $P$: 固定的 basis 变换矩阵。网络在内部学习 irreducible representation 空间下的 feature，输出时再通过 $P^{-1}$ 变换回 SE(3) 的 vector space。

---

### 3. Architecture Dissection

Paper 在 Section 4.3 和 Figure 3 中阐述了网络结构。整体分为 Encoding、Denoising 和 Decoding 三部分，基于 `escnn` library [50] 实现，约束在离散子群 $C_8$（8 个离散旋转，每 45 度一个）上。

1.  **Equivariant Encoders (White boxes)**
    *   **Observation Encoder:** 
        *   对于 agent view（RGB 或 Voxel），使用 Equivariant ResNet-18 (2D) 或 8-layer 3D Equivariant CNN。输出为 regular representation，shape 为 $\mathbb{R}^{u \times d_{\mathbf{o}}}$（其中 $u=8$）。这意味着 feature 被显式拆分成 8 个 group orbit channels。
        *   对于 eye-in-hand image（视角随机械臂动，不满足全局 equivariance），使用 standard ResNet，输出为 invariant feature（trivial representation $\rho_0$）。
        *   Gripper state：position 用 $\rho_1 \oplus \rho_0$，orientation (6D) 用 $\rho_1^3$，finger position 用 $\rho_0^2$。
    *   **Action Encoder:** 
        *   Noisy action $\mathbf{a}^k$ 通过 Equivariant Linear Layer 映射为 regular representation，shape 为 $\mathbb{R}^{u \times d_{\mathbf{a}}}$。

2.  **Denoising Network (Yellow box)**
    *   这是 paper 的一个工程亮点。如何让 1D Temporal U-Net (来自 Diffuser [15] 和 Diffusion Policy [1]) 处理 equivariant feature？
    *   作者没有设计一个极其复杂的 equivariant U-Net，而是利用了 regular representation 的特性。Regular representation 本质上是 group element 对 feature 的 permutation。
    *   令 $e_{\mathbf{o}}^g \in \mathbb{R}^{d_{\mathbf{o}}}$ 和 $e_{\mathbf{a}^k}^g \in \mathbb{R}^{d_{\mathbf{a}}}$ 为 group $C_8$ 中某个 element $g$ 对应的 partial embedding。
    *   网络对这 8 个 orbit **共享同一个 1D Temporal U-Net $U$**：
        $$ z^g = U(e_{\mathbf{o}}^g, e_{\mathbf{a}^k}^g, k) $$
    *   **Intuition:** 因为如果输入旋转了一个 group element，regular representation 内部只是在 channel 维度上发生了 cyclic permutation。把每个 group element 的 feature slice 出来分别过同一个网络，天然保证了 equivariance。这比在卷积层内部做复杂的 steerable kernel 卷积要简单得多，尤其适合需要捕捉 long-horizon temporal dependency 的 1D U-Net。

3.  **Equivariant Decoder (Gray box)**
    *   将输出的 regular representation noise embedding 通过 Equivariant Linear Layer 映射回原始的 action representation space（即前面推导的 $\rho_1^3 \oplus (\rho_1 \oplus \rho_0) \oplus \rho_0$）。随后通过 orthogonalization 将 vector 还原为合法的 SE(3) pose matrix。

---

### 4. Experimental Data Analysis

#### 4.1 Simulation Performance (MimicGen)
Paper 在 MimicGen [11] 的 12 个 tasks 上进行了实验。Table 1 和 Table 2 的数据极具说服力。

| Method | Ctrl | Obs | 100 Demos Avg Success | 200 Demos | 1000 Demos |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EquiDiff (Vo) | Abs | Voxel | **63.9% (+21.9)** | **72.6% (+14.8)** | 77.9% (+6.5) |
| DiffPo-C [1] | Abs | RGB | 42.0% | 57.8% | 71.4% |
| DiffPo-T [1] | Abs | RGB | 29.0% | 43.0% | 64.9% |
| DP3 [20] | Abs | PCD | 23.9% | 35.1% | 56.8% |
| ACT [51] | Abs | RGB | 21.3% | 38.2% | 63.3% |

**数据解读：**
1.  **Low-Data Regime Dominance:** 在 100 demos 的情况下，EquiDiff (Vo) 相比 baseline Diffusion Policy 提升了惊人的 21.9%。这直接验证了 equivariance inductive bias 缓解了 diffusion model 高数据开销的痛点。
2.  **Data Efficiency:** EquiDiff 用 200 demos 训练的效果（72.6%）甚至超越了原版 Diffusion Policy 用 1000 demos 训练的效果（71.4%）。
3.  **Voxel vs RGB:** Paper 发现 RGB 输入的 agent view 存在 perspective distortion，导致对称性被破坏（因为相机不是正交投影）。而 Voxel grid 是在 metric space 下构建的，严格保持 SO(2) symmetry。因此 Voxel 版本 (Vo) 效果最好。

#### 4.2 Real-World Robot Experiments
在 Section 5.3 中，作者在真实 Franka Emika + fin-ray fingers 上测试了 6 个长视野任务。
*   **Bagel Baking**（开烤箱 -> 拉托盘 -> 拿 bagel -> 放入托盘 -> 关托盘 -> 关烤箱），仅用 58 个 demos，EquiDiff 达到 80% 成功率。而 baseline 仅有 10%。
*   这证明了在真实世界的低维数据场景下，equivariant structure 使得网络无需再“浪费”参数去学习旋转不变性，而是专注于学习 task 的 temporal logic 和 fine-grained manipulation。

---

### 5. Broader Context, Hallucinations 与 Related Associations

这篇 paper 触及了当前 AI 领域几个非常深刻的命题，我想在这里为你做一些延伸和联想。

#### 5.1 Data Augmentation vs Equivariant Architecture
Paper 在 Appendix I 中对比了 CNN + Rotation Data Augmentation 与 Equivariant Network。结论是：Data Augmentation 能提升性能，但依然不如 Equivariant Net。
*   **Intuition:** Data Augmentation 是一种 statistical enforcement，它告诉网络“旋转后的输入应该对应旋转后的输出”，但网络依然需要在参数空间中通过梯度下降去拟合这组关系。而 Equivariant Network 是一种 structural enforcement（hard constraint），它在 hypothesis space 中直接划出了一个 invariant subspace。如果 network 输出不符合 equivariance，在数学上根本无法产生。对于极小数据集，前者容易 overfit 到 augmented distribution，而后者具有 theoretically guaranteed generalization。

#### 5.2 Connection to AlphaFold 2 与 SE(3)-Equivariance
这篇 paper 探讨的是 SO(2) Equivariance（绕重力轴旋转）。在 structural biology 领域，AlphaFold 2 [Reference](https://www.nature.com/articles/s41586-021-03819-2) 和 Equivariant Diffusion for Molecular Generation [4] 普遍采用完整的 SE(3) Equivariance (全 3D 旋转平移不变)。
*   **为什么 robotics 很少做 full SE(3) closed-loop？** 重力方向破坏了 3D 空间的各向同性。物体掉落、摩擦力、桌面 normal vector 都依赖于 z-axis 必须朝上。因此，SO(2) (绕 z 轴) + Z-translation (平移) 往往是 table-top manipulation 最合理的 symmetry group。
*   **Open Question:** 如果机器人是 mobile manipulator（比如 Spot 满地跑），则坐标系不再固定，full SE(3) equivariance 或者更复杂的 gauge equivariance [Walters et al.](https://arxiv.org/abs/2310.12232) 将是必要的。可以考虑将 paper 中的 SO(2) representation 升级为 frame-conditioned SE(3) representation，类似 DP3 [20] 的变体。

#### 5.3 The "Score Field" Perspective in Diffusion
Diffusion Model 学习的 $\varepsilon_\theta$ 本质上是 $\nabla_{\mathbf{a}^k} \log p(\mathbf{a}^k | \mathbf{o})$ 的近似。这个 vector field 在 high-dimensional action space 中指向 high-density region。
*   当 visual input $\mathbf{o}$ 旋转时，这个 vector field 应该刚性旋转。传统的 UNet 没有内置这种 geometric prior，因此它必须通过大量数据“记住”每一种旋转姿态下的 vector field 形状。
*   Equivariant Diffusion Policy 使得 vector field 的结构由网络架构直接保证。这让我联想到 Vector Quantized Diffusion 或者 Latent Diffusion 中对 latent space 的结构化约束。实际上，如果能将 action space 映射到一个完全由 irreducible representations 构成的 latent space 中进行 diffusion，理论上可以实现完美的 equivariant generative process。

#### 5.4 Implementation detail: Regular Representation 的本质
在实现中，$C_8$ 的 regular representation 是 $8 \times 8$ 的 permutation matrix。网络的特征图形状为 $u \times d$。
*   当 group element $g$ 作用时，特征图仅仅是沿着 $u$ 这个维度做了一次 cyclic shift。这意味着只要在推理时，如果输入图像旋转了 45 度，网络中间的 feature 永远会做 1 位的 shift，最后输出的 action 也自动旋转 45 度。
*   这和 Convolutional Neural Networks 的 weight sharing 有着异曲同工之妙：CNN 在 pixel grid 上 share weights 保证了 translation equivariance；而 Steerable CNN / Equivariant CNN 在 group orbit 上 share weights 保证了 rotation equivariance。

#### 5.5 Future Direction: Equivariance in VLA Models
当前 robotics 领域正在向 Vision-Language-Action (VLA) 模型发展，例如 RT-2 [Google DeepMind](https://robotics-transformer2.github.io/) 或 Octo [Octo Model](https://octo-models.github.io/)。
*   Language 是 discrete 且 modality 极其复杂的，它本身对 spatial rotation 没有 explicit equivariance。但是，action 输出端依然需要满足 spatial constraints。
*   如果将这篇 paper 的思想融入 VLA：可以把 LLM 视为处理 invariant features（语言指令、object semantics）的 trunk，而在 policy head 部分显式接入这种 Equivariant Diffusion Head。这就像在 Universal Transformer 上面加了一个 geometric inductive bias head，可能极大加速 VLA 在 few-shot manipulation task 上的收敛。

#### 5.6 Out-of-Plane Rotation Limitation
Paper 在 Appendix Table 4 提到了 max out of plane rotation。由于理论限制在 SO(2)，如果 demonstration 中包含了大量绕 x/y 轴的倾斜动作（比如倒水、插拔水平方向的 USB），SO(2) equivariance 会变成 incorrect equivariance [57] 损害性能。
*   Coffee Preparation D1 是 paper 中一个很有意思的 task，它的 max out of plane rotation 达到了 59 度。即便如此，EquiDiff 依然取得了巨大的提升。这说明只要 task 的主体逻辑在 SO(2) projection 下是可分的，partial equivariance 依然具有很强的 regularization 作用。

### References
*   Project Page: [https://equidiff.github.io](https://equidiff.github.io)
*   Diffusion Policy: [https://diffusion-policy.cs.columbia.edu/](https://diffusion-policy.cs.columbia.edu/)
*   MimicGen: [https://mimicgen.github.io](https://mimicgen.github.io)
*   escnn library: [https://github.com/QUVA-Lab/escnn](https://github.com/QUVA-Lab/escnn)
*   Equivariant Diffusion for Molecule Generation (Hoogeboom et al.): [https://arxiv.org/abs/2202.02976](https://arxiv.org/abs/2202.02976)
*   AlphaFold 2: [https://www.nature.com/articles/s41586-021-03819-2](https://www.nature.com/articles/s41586-021-03819-2)

总结来说，这篇 paper 不仅是 Diffusion Policy 的一次成功改进，更是 Geometric Deep Learning 在 robotics 领域的一次漂亮落地。它证明了在生成式模型中注入精确的物理对称性，是突破 data scaling law 瓶颈的一个极其有效的 pathway。希望这些分析对你的 intuition building 有所帮助！
