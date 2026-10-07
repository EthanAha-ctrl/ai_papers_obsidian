---
source_pdf: MONGE-AMPERE ` FLOW FOR GENERATIVE MODELING.pdf
paper_sha256: c14bb87f6e0736c6763b1e1fe4b4a4b238b0f4f29b09614d8e1b463b0eddc5c8
processed_at: '2026-08-05T20:17:19-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话再讲一遍

Karpathy你想听人话，那我把数学符号全剥掉，讲讲这paper到底在干嘛，以及为什么我觉得它clever。

---

## 一句话版本

**与其学一个复杂的变换函数，不如学一个"风场"，让Gaussian噪声像水一样被风吹成你想要的形状。**

---

## 问题从哪来

Flow-based models（RealNVP、Glow那些）的核心难题就一个字：**Jacobian行列式**。

你想把 $z$ 变成 $x$，得知道变换的"体积缩放比例"才能算概率密度。这个比例就是Jacobian matrix的行列式。问题是 $N$ 维matrix的行列式计算量是 $O(N^3)$，对images这种高维数据直接跪了。

所以RealNVP搞了coupling layer，把Jacobian人为限制成block-triangular，行列式瞬间变成 $O(N)$。代价是你的network architecture被严重束缚了——只能split成两半，一半不变另一半变。expressive power打折扣。

这paper说：**我不要这个束缚，但我还是要 $O(N)$**。

---

## 核心trick：把"一步走"变成"走无穷多步"

这是整个paper最beautiful的idea，我慢慢讲。

### 离散 vs 连续

传统flow：$\mathbf{x} = f(\mathbf{z})$，一次性变换完。Jacobian是 $\partial\mathbf{x}/\partial\mathbf{z}$，要算 $\det$。

Monge-Ampère flow的思路：不要一步到位，走很多很多小步。每步只挪一点点。

你想象从 $z$ 到 $x$ 的路径是一条曲线 $\mathbf{x}(t)$，$t$ 从 $0$ 到 $T$。每一瞬间有一个velocity field告诉你"往哪挪、挪多快"：

$$\frac{d\mathbf{x}}{dt} = \mathbf{v}(\mathbf{x})$$

这就是一个ODE。你用RK4或Euler积分 $T$ 步，就得到最终位置 $\mathbf{x}(T)$。

### 为什么连续化能省钱？

关键在于：每一步的变换是infinitesimal的，$\mathbf{x}_{t+\epsilon} = \mathbf{x}_t + \epsilon\mathbf{v}(\mathbf{x}_t)$。

Jacobian of one step: $J = I + \epsilon\frac{\partial\mathbf{v}}{\partial\mathbf{x}}$

那么 $\ln\det(J) = \text{Tr}\ln(I + \epsilon H) \approx \epsilon\,\text{Tr}(H) + O(\epsilon^2)$

这里 $H = \partial\mathbf{v}/\partial\mathbf{x}$ 是velocity field的Jacobian。

**$\text{Tr}(H)$ 是什么？** 就是 $H$ 的对角元之和 $\sum_i \partial v_i/\partial x_i$，也就是 $\nabla\cdot\mathbf{v}$（divergence）！

所以一步的log-Jacobian就是 $\epsilon(\nabla\cdot\mathbf{v})$，累加起来就是积分：

$$\ln\frac{p(\mathbf{x},T)}{p(\mathbf{x},0)} = -\int_0^T \nabla\cdot\mathbf{v}(\mathbf{x}(t))\,dt$$

**从 $O(N^3)$ 的determinant，变成 $O(N)$ 的divergence。** 因为trace只看对角元，不需要算cofactor expansion。

这就像：**你不需要知道整个matrix长啥样，只需要知道对角线之和。**

---

## 为什么velocity field必须是gradient？

这里Brenier定理出场了。Optimal transport理论说，在quadratic cost下，最优的transport map是某个convex function $u$ 的gradient：$\mathbf{x} = \nabla u(\mathbf{z})$。

翻译成人话：**最优的搬运方式是"从高势能滚向低势能"，不能有旋涡。**

为什么不能有旋涡？因为旋涡意味着两个particle绕圈，最终交换位置，这在transport中是waste——你可以不走弯路直接到。所以optimal transport的velocity field必须是curl-free的。

数学上，curl-free意味着 $\mathbf{v} = \nabla\varphi$（某个scalar potential的gradient）。这就自动满足：

$$\nabla\times\mathbf{v} = \nabla\times\nabla\varphi \equiv 0$$

**所以我们不直接学 $\mathbf{v}$，而是学 $\varphi$，然后取gradient。** 好处：

1. 只需要学一个scalar function（1个output），不是vector field（$N$个output）
2. 自动curl-free，对应optimal transport
3. Symmetry超级好impose（后面讲）

---

## 流体力学图像（这是最intuitive的部分）

把probability distribution想象成一盆水。

- 初始状态：Gaussian分布 = 水面平静、均匀铺开
- Target状态：data distribution = 你想要的水面形状（比如MNIST digits那种multimodal shape）

你怎么把平静的水面deform成target shape？**用风吹。**

风场就是 $\mathbf{v} = \nabla\varphi$。Potential $\varphi$ 的landscape决定了风往哪吹：

- $\varphi$ 高的地方，风往外吹（流体被推开，density降低）
- $\varphi$ 低的地方，风往里吸（流体聚集，density升高）
- $\nabla^2\varphi > 0$（convex区域）：local source，density dilute
- $\nabla^2\varphi < 0$（concave区域）：local sink，density concentrate

Density的变化由**continuity equation**控制：

$$\frac{\partial p}{\partial t} + \nabla\cdot(p\mathbf{v}) = 0$$

这就是**质量守恒**：水不会凭空产生或消失，只是从一处流到另一处。某处density增加，必然是别处的水流过来了。

**和diffusion model的本质区别**：
- Diffusion model = 热传导（加噪声+去噪声），有随机性，最终收敛到steady state
- Monge-Ampère flow = 对流（advection），完全deterministic，有限时间完成

Diffusion像往咖啡里倒牛奶等它自己扩散匀；Monge-Ampère像用勺子搅咖啡把牛奶推到位。

---

## Training：其实是在做optimal control

这里要build一个非常重要的intuition。

你有一个dynamical system（ODE），你想要它最终的状态符合target。你能control的是potential $\varphi$。这就是optimal control problem。

**类比**：你开一辆车，想让它最终停在北京。你能control的是方向盘和油门（$\varphi$）。车的运动方程是物理定律（ODE）。你通过调整control来让车到达目的地。

Training过程：
1. 跑一遍forward simulation（积分ODE）
2. 看终点density和target差多少（loss）
3. Backprop through ODE integrator，算 $\varphi$ 的gradient
4. 更新 $\varphi$ 的参数

**Backprop through ODE** 本质上就是Pontryagin's maximum principle——控制论里的经典结论。神经网络的反向传播其实是这个原理的离散化版本。

两个具体task：

### Density Estimation (MNIST)
你有一堆data samples $\pi(\mathbf{x})$，想知道它们的density。做法是**backward**积分：从data出发，倒着走回Gaussian。途中累积log-likelihood的变化，就知道data的likelihood了。

$$\text{NLL} = -\mathbb{E}_{\mathbf{x}\sim\pi}[\ln p(\mathbf{x},T)]$$

### Variational Free Energy (Ising)
你知道energy function $E(\mathbf{x})$ 但不知道normalization constant $Z$。做法是**forward**积分：从Gaussian出发，顺着走到target，让model去拟合Boltzmann distribution。

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}\sim p(\mathbf{x},T)}[\ln p(\mathbf{x},T) + E(\mathbf{x})]$$

这两项一项是entropy（$\ln p$ 越大越concentrated），一项是energy（越低越好）。**Model要在"集中到低能量区域"和"保持spread out"之间找平衡**，这就是free energy $F = E - TS$ 的variational principle。

---

## Symmetry：这paper最elegant的部分

物理系统的symmetry很重要。Ising model有：
- Spin翻转 $Z_2$：$\mathbf{x} \to -\mathbf{x}$
- 平移symmetry
- $D_4$旋转reflection（正方形的8个symmetry操作）

你要让generative model respect这些symmetry，否则生成的samples会有bias。

**传统方法的痛**：vector-valued map $f: \mathbb{R}^N \to \mathbb{R}^N$ 要impose symmetry，得设计equivariant architecture，很复杂。

**Monge-Ampère flow的解**：只要让scalar potential $\varphi$ 对称就行！

$$\varphi(\mathbf{x}) = \frac{1}{|G|}\sum_{\mathbf{g}\in G}\tilde\varphi(\mathbf{g}\mathbf{x})$$

翻译成人话：**把input做所有symmetry变换，过同一个network，取平均。**

为什么work？因为gradient和Laplacian都是linear operator，symmetric potential的gradient自然就是equivariant的。你不需要设计fancy的group equivariant CNN，只需要在input上做group augmentation。

**Stochastic trick**：每步积分只随机sample一个 $g$，不是全算 $|G|$ 个。on average对称性恢复，计算量不增加。这像data augmentation，但是是"online"的。

---

## 1D Gaussian toy example的intuition

Appendix A那个例子很helpful。设 $\varphi(x) = \lambda x^2/2$。

- $\lambda > 0$：potential是个碗，风从中心往外吹。Gaussian被"撑大"，variance指数增长。
- $\lambda < 0$：potential是个山，风往中心吹。Gaussian被"压缩"，variance指数减小。

具体来说，$p(x,t) \propto \exp(-\alpha(t)^2 x^2/2)$，其中 $\alpha(t) = e^{-\lambda t}$。

**Intuition**：potential的curvature（$\lambda$）直接控制density的scaling rate。正curvature = expand，负curvature = compress。

推广到高维：**Hessian eigenvalues的正负决定了每个principal direction上是expand还是compress。**

---

## 和后来发展的关系

这篇paper是2018年的，当时Neural ODE也刚出来。后续发展：

### FFJORD (2019)
Monge-Ampère flow要exact算Laplacian $\nabla^2\varphi$，对high-dim还是贵。FFJORD用Hutchinson trace estimator：

$$\nabla^2\varphi = \text{Tr}(H_\varphi) \approx \mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}[\epsilon^T H_\varphi \epsilon]$$

只需要Hessian-vector product（$O(N)$），不需要full Hessian。**这是把Monge-Ampère flow scale up到image dataset的关键。**

### Flow Matching (2023)
更激进的做法：不care OT了，直接regress velocity field到target vector field。训练更简单，不需要backprop through ODE integrator。现在的Stable Diffusion 3、视频生成模型都在用。

### Continuous Normalizing Flows family
这paper + Neural ODE + FFJORD一起，构成了CNF这条线。后来发现和score-based diffusion、Schrödinger bridge都有深层联系。

---

## 我觉得这paper真正clever的地方

1. **Linearization trick**：把discrete的determinant问题变成continuous的trace问题。这是数学上的"降维打击"。

2. **Potential parameterization**：用scalar function + gradient代替vector field。少N倍参数，还自动curl-free。

3. **Symmetry的优雅处理**：scalar potential的group average，比equivariant network简单太多。

4. **Unifying view**：density estimation和variational inference用同一个framework，只是loss不同。物理学家和ML researcher都能用。

5. **Conceptual clarity**：fluid dynamics + optimal transport + control theory三个领域自然交汇，不是强行拼接。

---

## 局限也很明显

1. **ODE integration慢**：100步RK4 = 400次NN evaluation，比RealNVP一次forward慢多了
2. **没用adjoint method**：memory consumption大
3. **Single hidden layer**：minimalist但限制表达力
4. **只测了MNIST和Ising**：没有CIFAR、ImageNet实验，scalability存疑
5. **Time-independent potential**：suboptimal，真正OT需要time-dependent

---

## 给你的take-away

如果只记一件事：**连续时间极限把Jacobian行列式（$O(N^3)$）变成divergence（$O(N)$），代价是要积分ODE。**

这trade-off值不值？2018年看着一般，但后来Neural ODE的adjoint method解决了memory，FFJORD解决了Laplacian计算，Flow Matching解决了训练难度。这条路线现在已经是generative modeling的主流之一。

所以这paper的价值更多在**conceptual framework**——把optimal transport的数学结构带进了normalizing flow，为后来整个CNF family铺路。

---

Reference links:
- [Paper on arXiv](https://arxiv.org/abs/1809.00088)
- [Neural ODE](https://arxiv.org/abs/1806.07366)
- [FFJORD](https://arxiv.org/abs/1810.01367)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Benamou-Brenier Dynamic OT](https://link.springer.com/article/10.1007/s002110050002)
- [Weinan E - ML via Dynamical Systems](https://arxiv.org/abs/1611.07422)
- [Villani - Optimal Transport](https://link.springer.com/book/10.1007/978-3-540-71050-9)
- [Onsager Ising Solution](https://journals.aps.org/pr/abstract/10.1103/PhysRev.65.117)

---

# Monge-Ampère Flow for Generative Modeling 深度解析

Karpathy你好，这篇paper我读得非常仔细，给你做一个尽量完整的technical walkthrough，顺便build一些intuition。

## 1. 背景：Flow-based Generative Models的痛点

Flow-based models（NICE, RealNVP, Glow, MAF）的核心是找一个diffeomorphism $\mathbf{x} = f(\mathbf{z})$，把simple base distribution $p(\mathbf{z})$（通常是Gaussian）变换成复杂的data distribution $q(\mathbf{x})$。Change of variable formula给出：

$$p(\mathbf{z}) = q(\mathbf{x})\left|\det\left(\frac{\partial \mathbf{x}}{\partial \mathbf{z}}\right)\right|$$

这里 $\frac{\partial \mathbf{x}}{\partial \mathbf{z}}$ 是Jacobian matrix，$|\det(\cdot)|$ 是其行列式的绝对值。

**痛点**：计算 $\det(J)$ 是 $O(N^3)$ 的，对高维数据不可行。所以NICE用triangular Jacobian（$O(N)$），RealNVP用block-triangular，MAF用autoregressive结构。这些architectural constraints牺牲了expressive power。

这篇paper的核心insight：**用连续时间极限把Hessian determinant（$O(N^3)$）简化为Laplacian（$O(N)$）**。

Reference: [Normalizing Flows tutorial](https://blog.evjang.com/2018/01/nf1.html)

---

## 2. 从Brenier定理到Monge-Ampère方程

### 2.1 Brenier定理

在optimal transport theory中，给定quadratic cost $c(\mathbf{z},\mathbf{x}) = \|\mathbf{z}-\mathbf{x}\|^2/2$，Brenier (1991) 证明了optimal transport map是某个convex function $u$ 的gradient：

$$\mathbf{x} = \nabla u(\mathbf{z})$$

这里 $u: \mathbb{R}^N \to \mathbb{R}$ 是convex的Brenier potential，$\nabla u$ 是其gradient（一个vector-valued map）。

**Intuition**：与其直接学一个vector field $\mathbf{x}(\mathbf{z})$，不如学一个scalar potential $u(\mathbf{z})$，然后取gradient。这自动保证了map是monotone的（convexity），对应optimal transport的"无交叉"性质。

### 2.2 Monge-Ampère方程

把 $\mathbf{x} = \nabla u(\mathbf{z})$ 代入change of variable formula：

$$\frac{p(\mathbf{z})}{q(\nabla u(\mathbf{z}))} = \det\left(\frac{\partial^2 u}{\partial z_i \partial z_j}\right)$$

变量解释：
- $p(\mathbf{z})$：latent space的density（已知，通常是Gaussian）
- $q(\mathbf{x})$：target density（data distribution，只有samples或unnormalized form）
- $\nabla u(\mathbf{z})$：generative map
- $\frac{\partial^2 u}{\partial z_i \partial z_j}$：$u$ 的Hessian matrix，记作 $H_u$
- $\det(H_u)$：Hessian的行列式，即Jacobian of $\nabla u$

**三个挑战**：
1. $\det(H_u)$ 对 $u$ 是非线性的（highly non-convex optimization）
2. 计算complexity是 $O(N^3)$
3. 实际中往往只有一端的samples，或者只有unnormalized density

Reference: [Villani - Optimal Transport](https://link.springer.com/book/10.1007/978-3-540-71050-9)

---

## 3. 核心技巧：Linearization与连续时间极限

### 3.1 Infinitesimal transformation

把Brenier potential写成perturbation形式：

$$u(\mathbf{z}) = \frac{\|\mathbf{z}\|^2}{2} + \epsilon\varphi(\mathbf{z})$$

变量解释：
- $\|\mathbf{z}\|^2/2$：identity map的potential（$\nabla(\|\mathbf{z}\|^2/2) = \mathbf{z}$）
- $\epsilon$：infinitesimal小参数
- $\varphi(\mathbf{z})$：perturbation potential（这是我们要学的）

于是：

$$\mathbf{x} = \nabla u(\mathbf{z}) = \mathbf{z} + \epsilon\nabla\varphi(\mathbf{z})$$

即 $\mathbf{x} - \mathbf{z} = \epsilon\nabla\varphi(\mathbf{z})$，变换是infinitesimal的。

### 3.2 Log-Jacobian的Taylor展开

Jacobian是：

$$J = I + \epsilon H_\varphi$$

其中 $H_\varphi = \frac{\partial^2\varphi}{\partial z_i\partial z_j}$ 是 $\varphi$ 的Hessian。

利用matrix logarithm的展开 $\ln(I+A) = A - A^2/2 + A^3/3 - \cdots$：

$$\ln\det(I + \epsilon H_\varphi) = \text{Tr}\ln(I + \epsilon H_\varphi) = \epsilon\,\text{Tr}(H_\varphi) + O(\epsilon^2)$$

而 $\text{Tr}(H_\varphi) = \sum_i \frac{\partial^2\varphi}{\partial z_i^2} = \nabla^2\varphi$ 就是Laplacian！

所以：

$$\ln q(\mathbf{x}) - \ln p(\mathbf{z}) = -\ln\det(J) = -\epsilon\nabla^2\varphi(\mathbf{z}) + O(\epsilon^2)$$

**关键简化**：$O(N^3)$ 的determinant变成 $O(N)$ 的Laplacian。

### 3.3 连续时间极限

取 $\epsilon \to 0$，把infinitesimal steps累加成连续时间演化。把 $\epsilon$ 替换成 $dt$，得到两个coupled ODEs：

$$\boxed{\frac{d\mathbf{x}}{dt} = \nabla\varphi(\mathbf{x})} \quad (2)$$

$$\boxed{\frac{d\ln p(\mathbf{x},t)}{dt} = -\nabla^2\varphi(\mathbf{x})} \quad (3)$$

变量解释：
- $\mathbf{x}(t) \in \mathbb{R}^N$：time-dependent的particle位置
- $p(\mathbf{x},t)$：time-dependent的概率密度
- $\varphi(\mathbf{x}): \mathbb{R}^N \to \mathbb{R}$：learnable scalar potential（neural network参数化）
- $t \in [0,T]$：连续时间
- $T$：total integration time
- $\nabla\varphi$：velocity field（gradient of potential）
- $\nabla^2\varphi = \sum_i \partial^2\varphi/\partial x_i^2$：Laplacian，控制density的local变化率

**Boundary conditions**：
- $\mathbf{x}(0) = \mathbf{z} \sim \mathcal{N}(\mathbf{z})$
- $p(\mathbf{x},0) = \mathcal{N}(\mathbf{x})$
- $p(\mathbf{x},T) = q(\mathbf{x})$（target）

**Intuition**：把generative map从"one-shot function"变成"dynamical system"。就像ResNet把plain network拆成residual blocks，这里把transport map拆成无穷多个infinitesimal steps，每步只需要学一个简单的velocity field。

---

## 4. 流体力学Interpretation（这是paper的精华）

### 4.1 Lagrangian vs Eulerian视角

Equation (2) 是**Lagrangian描述**：跟随fluid parcel的trajectory。

Equation (3) 中的 $d/dt$ 是**material derivative**：

$$\frac{d}{dt} = \frac{\partial}{\partial t} + \frac{d\mathbf{x}}{dt}\cdot\nabla$$

第一项 $\partial/\partial t$ 是local变化率（fixed point观察），第二项 $\frac{d\mathbf{x}}{dt}\cdot\nabla$ 是convective变化率（跟着parcel移动看到的spatial gradient）。

### 4.2 Continuity Equation

把material derivative展开，代入equation (2)：

$$\frac{\partial p(\mathbf{x},t)}{\partial t} + \nabla\cdot[p(\mathbf{x},t)\,\mathbf{v}] = 0 \quad (5)$$

其中 $\mathbf{v} = \nabla\varphi(\mathbf{x})$。

这就是**compressible fluid的continuity equation**！物理意义：probability mass守恒，density的变化完全由flux $p\mathbf{v}$ 的divergence决定。

### 4.3 关键性质

1. **Irrotational flow**：$\nabla\times\mathbf{v} = \nabla\times\nabla\varphi \equiv 0$。Velocity field无旋，这是gradient flow的特征。

2. **Compressible**：$\nabla\cdot\mathbf{v} = \nabla^2\varphi$ 可以非零，fluid可以压缩或膨胀。

3. **Deterministic**：没有stochastic force（区别于diffusion models）。

4. **Reversible**：ODE可以forward/backward积分，complexity对称。

**Deep intuition**：想象一盆水，初始是Gaussian分布（平静的水面）。我们用一个curl-free的"风场"$\mathbf{v} = \nabla\varphi$ 吹这盆水，让它deform成target shape。风场的"源"和"汇"（$\nabla^2\varphi$ 的正负）控制哪里density聚集、哪里分散。

为什么curl-free重要？因为Brenier定理说optimal transport map是convex potential的gradient，本质上就是curl-free condition。这保证了transport是"无涡旋"的，particle轨迹不交叉。

Reference: [Fluid Dynamics in ML](https://arxiv.org/abs/1804.04272)

---

## 5. Training：Optimal Control视角

### 5.1 目标函数

Training变成optimal control problem：

$$\min_\varphi I[p(\mathbf{x},T), q(\mathbf{x})] \quad (4)$$

两个具体场景：

**场景1: Density Estimation**（已知data samples $\pi(\mathbf{x})$，maximize likelihood）

$$\text{NLL} = -\mathbb{E}_{\mathbf{x}\sim\pi(\mathbf{x})}[\ln p(\mathbf{x},T)] \quad (6)$$

这等价于minimize $D_{\text{KL}}(\pi(\mathbf{x}) \| p(\mathbf{x},T))$。

**场景2: Variational Free Energy**（已知unnormalized Boltzmann distribution $e^{-E(\mathbf{x})}/Z$）

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}\sim p(\mathbf{x},T)}[\ln p(\mathbf{x},T) + E(\mathbf{x})] \quad (7)$$

这等价于minimize $D_{\text{KL}}(p(\mathbf{x},T) \| e^{-E}/Z)$，是 $\ln Z$ 的variational upper bound。

**Intuition for (7)**：Loss有两项
- $\ln p(\mathbf{x},T)$：negative entropy of model（越concentrated越大）
- $E(\mathbf{x})$：physical energy（越低越favorable）
- 平衡：model想concentrate在低能量区域，但太concentrated会损失entropy。这正是统计力学free energy $F = E - TS$ 的variational形式。

### 5.2 计算流程

**Density estimation**（backward integration）：
1. 从data $\mathbf{x} \sim \pi(\mathbf{x})$ 出发
2. Backward integrate equations (2,3) 从 $t=T$ 到 $t=0$
3. Accumulate $\int_T^0 d\ln p(\mathbf{x}(t),t) = -\int_0^T \nabla^2\varphi(\mathbf{x}(t))\,dt$
4. 得到 $\ln p(\mathbf{x},T) = \ln p(\mathbf{x},0) - \int_0^T \nabla^2\varphi\,dt = \ln\mathcal{N}(\mathbf{x}(0)) - \int_0^T \nabla^2\varphi\,dt$

**Sampling**（forward integration）：
1. 从 $\mathbf{z} \sim \mathcal{N}(\mathbf{z})$ 出发
2. Forward integrate equations (2,3) 从 $t=0$ 到 $t=T$
3. 得到samples $\mathbf{x}(T)$ 和它们的likelihoods

Reference: [Weinan E - Machine Learning via Dynamical Systems](https://arxiv.org/abs/1611.07422)

---

## 6. 架构与实现细节

### 6.1 Potential function参数化

$\varphi(\mathbf{x})$ 用一个**单hidden layer densely connected NN**参数化：

$$\varphi(\mathbf{x}) = W_2 \cdot \text{softplus}(W_1\mathbf{x} + \mathbf{b}_1) + b_2$$

变量：
- $W_1 \in \mathbb{R}^{h\times N}$：input-to-hidden weight
- $W_2 \in \mathbb{R}^{1\times h}$：hidden-to-output weight
- $\mathbf{b}_1 \in \mathbb{R}^h$, $b_2 \in \mathbb{R}$：biases
- $\text{softplus}(x) = \ln(1+e^x)$：保证higher-order differentiability

为什么用softplus而不是ReLU？因为ReLU的二阶导数几乎处处为零（除了kink处），而我们需要Laplacian $\nabla^2\varphi$，所以potential必须 $C^2$ smooth。

**Gradient和Laplacian通过automatic differentiation计算**。

### 6.2 ODE Integration：RK4

用4th order Runge-Kutta积分。RK4的每个step需要evaluate derivative 4次：

对于 $\dot{\mathbf{x}} = \nabla\varphi(\mathbf{x})$：

$$\mathbf{k}_1 = \nabla\varphi(\mathbf{x}_t)$$
$$\mathbf{k}_2 = \nabla\varphi(\mathbf{x}_t + \epsilon\mathbf{k}_1/2)$$
$$\mathbf{k}_3 = \nabla\varphi(\mathbf{x}_t + \epsilon\mathbf{k}_2/2)$$
$$\mathbf{k}_4 = \nabla\varphi(\mathbf{x}_t + \epsilon\mathbf{k}_3)$$
$$\mathbf{x}_{t+\epsilon} = \mathbf{x}_t + \frac{\epsilon}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)$$

同样对 $\ln p$ 积分（用 $\nabla^2\varphi$）。

### 6.3 与ResNet的等价性

```
┌─────────────────────────────────────────────────┐
│  RK4 step (4 NN evaluations)                    │
│  x_{t+ε} = x_t + ε/6 (k1+2k2+2k3+k4)            │
└─────────────────────────────────────────────────┘
              ↑ repeat d times
┌─────────────────────────────────────────────────┐
│  d=100 steps → 400 "layers"                     │
│  但所有layers share 同一个 φ 的参数              │
└─────────────────────────────────────────────────┘
```

**Parameter efficiency**：100个RK4 steps = 400层"network"，但参数只有一个单hidden layer NN。这解释了为什么只用MAF约1/10的参数。

### 6.4 Hyperparameters

| Problem | $\epsilon$ (step size) | $d$ (steps) | $T=\epsilon d$ | $h$ (hidden) | $B$ (batch) |
|---------|------------------------|-------------|-----------------|--------------|-------------|
| MNIST | 0.1 | 100 | 10.0 | 1024 | 100 |
| Ising | 0.1 | 50 | 5.0 | 512 | 64 |

**Trade-off**： Longer $T$ → deeper effective network → 每步学更简单的transform，但integration error累积；Shorter $T$ → 每步需要学更complex的velocity field。

**Variable depth at inference**：训练时用小 $\epsilon$、大 $d$；inference时可以用大 $\epsilon$、小 $d$ 加速（牺牲精度）。

Reference: [Neural ODE](https://arxiv.org/abs/1806.07366)

---

## 7. Application 1: MNIST Density Estimation

### 7.1 预处理

1. **Dequantization**：把integer pixel values映射到continuous space（加uniform noise）
2. **Logit transformation**：$\mathbf{x} \mapsto \text{logit}(\lambda + (1-2\lambda)\mathbf{x})$，$\lambda = 10^{-6}$

Logit变换把 $[0,1]$ 区间映射到 $(-\infty, +\infty)$，使data分布更接近Gaussian，便于flow学习。

### 7.2 结果

| Model | Test NLL (↓ better) |
|-------|---------------------|
| MADE | 1380.8 ± 4.8 |
| Real NVP | 1323.2 ± 6.6 |
| MAF | 1300.5 ± 1.7 |
| **Monge-Ampère Flow** | **1255.5 ± 2.0** |

**亮点**：
- NLL最低（最好）
- Variance小（2.0 vs MAF的1.7，但绝对值低45 nats）
- Parameter count约为MAF的1/10

### 7.3 可视化理解

Figure 3(b) 展示了MNIST images如何被backward flow"溶解"成Gaussian noise。过程是**连续的Gaussianization**：
- 早期steps：remove fine-grained texture
- 中期steps：remove digit shape
- 晚期steps：变成pure Gaussian noise

这让人联想到**coarse-to-fine的hierarchical structure**：flow先处理high-frequency细节，再处理low-frequency结构。

Reference: [Gaussianization](https://papers.nips.cc/paper/2000/hash/96d7ff740b5ae5b5a0b1c9c7b2c4c5f3.html)

---

## 8. Application 2: Ising Model Variational Free Energy

### 8.1 Ising Model背景

2D Ising model on square lattice，spins $s_i \in \{-1, +1\}$，partition function：

$$Z_{\text{Ising}} = \sum_{\mathbf{s}\in\{\pm1\}^{\otimes N}} \exp\left(\frac{1}{2}\mathbf{s}^T K \mathbf{s}\right) \quad (10)$$

变量：
- $\mathbf{s}$：spin configuration
- $K$：coupling matrix，$K_{ij} = (1+\sqrt{2})/2$ for nearest neighbors（critical temperature）
- $T_c = 2/\ln(1+\sqrt{2}) \approx 2.269$：Onsager critical temperature

**为什么critical point难？** Critical fluctuations是long-range的，correlation length发散。Variational method必须capture这些long-range correlations，对model表达力要求高。

### 8.2 Continuous Formulation（Appendix B）

用Hubbard-Stratonovich transformation把discrete spins变成continuous variables：

**Step 1**: Offset coupling $K \to K + \alpha I$（$\alpha$ 使最小eigenvalue = 0.1，确保正定）

**Step 2**: Gaussian integration trick：
$$\exp\left(\frac{1}{2}\mathbf{s}^T(K+\alpha I)\mathbf{s}\right) = \int d\mathbf{x}\,\exp\left(-\frac{1}{2}\mathbf{x}^T(K+\alpha I)^{-1}\mathbf{x} + \mathbf{s}^T\mathbf{x}\right)$$

**Step 3**: Trace out spins（$\sum_{s_i=\pm1} e^{s_i x_i} = 2\cosh(x_i)$）：

$$\boxed{E(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T K^{-1}\mathbf{x} - \sum_i \ln\cosh(x_i)} \quad (8)$$

变量：
- $\mathbf{x} \in \mathbb{R}^N$：continuous auxiliary variables
- $K^{-1}$：inverse coupling matrix（第一项是Gaussian prior）
- $\ln\cosh(x_i)$：从tracing out spins得到的nonlinear term
- $N = 16^2 = 256$（lattice size）

**Free energy解析关系**：
$$\ln Z = \ln Z_{\text{Ising}} - \frac{1}{2}\ln\det(K+\alpha I) + \frac{N}{2}[\ln(2/\pi) - \alpha]$$

$Z_{\text{Ising}}$ 的exact value来自Onsager (1944) / Kaufman (1949)。

Reference: [Onsager Solution](https://journals.aps.org/pr/abstract/10.1103/PhysRev.65.117)

### 8.3 对称性处理（这是paper的技术亮点）

Ising model on periodic square lattice有symmetries：
- $Z_2$: spin inversion $\mathbf{x} \to -\mathbf{x}$
- Translation: lattice translations
- $D_4$: 8个rotational/reflection symmetries of square

物理上 $E(\mathbf{x}) = E(g\mathbf{x})$ for $g \in G$。要generate symmetric configurations，model must respect这些symmetries。

**传统方法的困难**：vector-valued generative map很难impose symmetry，因为需要设计equivariant architecture。

**Monge-Ampère flow的优雅解**：因为map是scalar potential的gradient，只需让potential symmetric：

$$\varphi(\mathbf{x}) = \frac{1}{|G|}\sum_{\mathbf{g}\in G}\tilde\varphi(\mathbf{g}\mathbf{x})$$

变量：
- $G$：symmetry group（如 $D_4$ 有8个elements）
- $|G|$：group order
- $\mathbf{g}$：group element（symmetry operation）
- $\tilde\varphi$：shared neural network（所有terms共享参数）

**为什么work**：gradient和Laplacian都是linear operators：
$$\nabla\varphi(\mathbf{x}) = \frac{1}{|G|}\sum_{\mathbf{g}} \mathbf{g}^{-1}\nabla\tilde\varphi(\mathbf{g}\mathbf{x})$$

**Stochastic approximation**：每步integration只sample一个 $\mathbf{g}$ 来evaluate，on average恢复对称性。计算成本 $\times 1$ 而非 $\times |G|$。

### 8.4 结果

Figure 4 显示variational loss收敛到exact free energy（红色水平线），精度与Li & Wang (2018)的specialized 2D network相当。生成的Ising configurations展现出各种domain shapes，且respect physical symmetries。

Reference: [Neural Network Renormalization Group](https://arxiv.org/abs/1802.02840)

---

## 9. Appendix A: 1D Gaussian解析解

考虑 $\varphi(x) = \lambda x^2/2$，初始 $p(x,0) = \mathcal{N}(x)$。

**Ansatz**：保持Gaussian form，允许variance变化：

$$p(x,t) = \frac{\alpha(t)}{\sqrt{2\pi}}\exp\left(-\frac{\alpha(t)^2 x^2}{2}\right)$$

变量：
- $\alpha(t)$：time-dependent的inverse width
- $\alpha(0) = 1$：初始标准Gaussian

ODE (2): $\dot{x} = \nabla\varphi = \lambda x$

ODE (3) with material derivative：

$$-\lambda = \frac{d\ln p}{dt} = \underbrace{\frac{d\ln\alpha}{dt}}_{\partial/\partial t} - \underbrace{\left(\frac{d\ln\alpha}{dt} + \lambda\right)\alpha^2 x^2}_{\dot{x}\cdot\nabla\ln p}$$

**Matching powers of $x$**：
- Constant term: $\frac{d\ln\alpha}{dt} = -\lambda$
- $x^2$ term: $\frac{d\ln\alpha}{dt} + \lambda = 0$（consistent）

**Solution**: $\alpha(t) = e^{-\lambda t}$

**物理图像**：
- $\lambda > 0$：potential是"山谷"，particle加速远离原点（$\dot{x}=\lambda x$），Gaussian宽度指数增长
- $\lambda < 0$：potential是"山峰"（但非convex！），particle被推向原点，Gaussian收缩
- 远离原点的particle移动更快（velocity ∝ $x$）

这个toy example说明：**potential的curvature直接控制density的scaling**。

---

## 10. 与Related Work的对比

### 10.1 Normalizing Flows (NICE, RealNVP, Glow)

| Aspect | Traditional NF | Monge-Ampère Flow |
|--------|----------------|-------------------|
| Jacobian computation | Triangular/block structure | Continuous-time → Laplacian |
| Complexity | $O(N)$ (with constraints) | $O(N)$ (no architectural constraint) |
| Expressive power | Limited by coupling layers | Flexible (any smooth potential) |
| Invertibility | Explicit inverse needed | ODE backward integration |

### 10.2 Autoregressive Flows (MAF, IAF)

| Aspect | Autoregressive | Monge-Ampère |
|--------|----------------|--------------|
| Forward complexity | $O(N)$ (sequential) | $O(N)$ (parallel) |
| Inverse complexity | $O(N)$ but sequential | Same as forward |
| Reversibility | Implicit (solve nonlinear eq) | Exact (ODE reversible) |

**Key insight**: MAF适合density estimation（forward快），IAF适合sampling（inverse快）。Monge-Ampère flow两者对称。

### 10.3 Diffusion Models (Sohl-Dickstein, DDPM)

| Aspect | Diffusion | Monge-Ampère |
|--------|-----------|--------------|
| Dynamics | Stochastic (Langevin-like) | Deterministic (advection) |
| Steady state | Asymptotic $t\to\infty$ | Finite time $T$ |
| Noise | Essential | None |
| Reversibility | Stochastic reverse | Deterministic reverse |

**Conceptual difference**: Diffusion是"加noise再去noise"，Monge-Ampère是"用风场吹流体"。前者像热传导，后者像advection。

Reference: [Diffusion Models](https://arxiv.org/abs/1503.03585)

### 10.4 Neural ODE (Chen et al. 2018)

同期work，用adjoint method做memory-efficient backprop。Monge-Ampère flow没有用adjoint，但作者在Discussion中提到可以adopt。

**Adjoint method的核心**：不存储中间states，backward时重新solve一个adjoint ODE来计算gradients。Memory complexity从 $O(d)$ 降到 $O(1)$。

Reference: [Neural ODE paper](https://arxiv.org/abs/1806.07366)

---

## 11. Building Intuition: 几个关键insights

### Insight 1: Linearization作为"Jacobian trick"

传统flow的Jacobian determinant是 $O(N^3)$。这篇paper的trick：

$$\ln\det(I + \epsilon H) \approx \epsilon\,\text{Tr}(H) + O(\epsilon^2)$$

把determinant（multiplicative, nonlinear）变成trace（additive, linear）。然后accumulate over infinitely many steps。

这本质上就是**把discrete map变成continuous flow，用微分代替差分**。类似地，ResNet的 $h_{l+1} = h_l + f(h_l)$ 在continuous limit变成 $\dot{h} = f(h)$。

### Insight 2: Potential-based parameterization的优势

学vector field $\mathbf{v}(\mathbf{x})$ vs 学potential $\varphi(\mathbf{x})$ 然后 $\mathbf{v} = \nabla\varphi$：

1. **自动curl-free**：保证transport的"无交叉"性质（optimal transport的特征）
2. **Symmetry友好**：scalar function的symmetrization比vector field简单
3. **Parameter efficient**：$N$维vector field需要 $N$ outputs，potential只需1个
4. **物理可解释**：potential landscape直接对应"势能地形"

### Insight 3: Control Theory视角

Training = optimal control：
- **State**: $(\mathbf{x}(t), p(\mathbf{x},t))$
- **Control**: $\varphi(\mathbf{x})$（参数化的NN）
- **Dynamics**: ODEs (2,3)
- **Objective**: terminal cost $I[p(\mathbf{x},T), q(\mathbf{x})]$

这联系到Pontryagin's maximum principle和Hamilton-Jacobi-Bellman equation。Backpropagation本质上就是Pontryagin的adjoint equation。

Reference: [Han & E - Deep Learning for Stochastic Control](https://arxiv.org/abs/1611.07422)

### Insight 4: Symmetry作为inductive bias

$$\varphi(\mathbf{x}) = \frac{1}{|G|}\sum_{\mathbf{g}\in G}\tilde\varphi(\mathbf{g}\mathbf{x})$$

这个**group averaging** trick在physics中很常见（构造symmetric observables）。在ML中，类似思想出现在：
- Deep Sets (Zaheer et al. 2017): permutation invariance
- Equivariant CNNs (Cohen et al. 2018): translation/rotation equivariance
- Symmetric normalizing flows

**Deep insight**: Symmetry约束在scalar potential上是"免费"的（只需symmetrize input），在vector field上很expensive（需要设计equivariant architecture）。

### Insight 5: Time-independent vs Time-dependent Potential

本文用 $\varphi(\mathbf{x})$（time-independent），所有integration steps共享参数。

**Extension**: $\varphi(\mathbf{x}, t)$（time-dependent）可以induce更rich的flow。Benamou-Brenier (2000) 证明optimal transport的dynamic formulation是pressureless flow with constant velocity：

$$\mathbf{v}(\mathbf{x},t) = \frac{T-t}{T}\mathbf{v}_0(\mathbf{x}) + \frac{t}{T}\mathbf{v}_T(\mathbf{x})$$

这对应linear interpolation between initial and final velocity fields。

**Practical implication**: Time-independent potential是suboptimal transport（更constrained），但parameter efficient。Time-dependent更接近true OT，但需要更多参数（或用hypernetwork参数化 $\varphi(\mathbf{x},t)$）。

Reference: [Benamou-Brenier Dynamic OT](https://link.springer.com/article/10.1007/s002110050002)

---

## 12. Limitations与Future Directions

### 12.1 当前limitation

1. **Integration cost**: 100步RK4 = 400次NN evaluation，比单次forward pass慢
2. **Memory**: 没用adjoint method，存储所有中间states
3. **Single hidden layer**: minimalist但可能限制expressive power
4. **Time-independent potential**: suboptimal transport

### 12.2 论文提到的improvements

1. **CNN potential** for spatial/temporal data
2. **Symplectic integrators** 保证time-reversal symmetry
3. **Adjoint backprop** (Neural ODE style) 减少memory
4. **Wasserstein loss** 替代KL
5. **Batch normalization** during integration
6. **Time-dependent potential** via hypernetwork

### 12.3 后续发展（paper之后的field演进）

1. **FFJORD** (Grathwohl et al. 2019): 用Hutchinson's trace estimator近似Laplacian，避免exact computation
   $$\text{Tr}(H_\varphi) \approx \mathbb{E}_{\mathbf{v}\sim\mathcal{N}(0,I)}[\mathbf{v}^T H_\varphi \mathbf{v}]$$

2. **Continuous Normalizing Flows**: 这篇paper的思路被广泛采纳

3. **Equivariant Flows**: symmetry处理发展成独立方向

4. **Stochastic Interpolants** (Albergo et al. 2023): 统一deterministic和stochastic flows

5. **Flow Matching** (Lipman et al. 2023): 简化CNF training，直接regress velocity field

Reference: [FFJORD](https://arxiv.org/abs/1810.01367), [Flow Matching](https://arxiv.org/abs/2210.02747)

---

## 13. 公式速查表

| Equation | Meaning | Variables |
|----------|---------|-----------|
| $\mathbf{x} = \nabla u(\mathbf{z})$ | Brenier map | $u$: convex potential |
| $\frac{p(\mathbf{z})}{q(\nabla u)} = \det(H_u)$ | Monge-Ampère eq | $H_u$: Hessian of $u$ |
| $u = \|\mathbf{z}\|^2/2 + \epsilon\varphi$ | Perturbation | $\epsilon$: small param |
| $\dot{\mathbf{x}} = \nabla\varphi$ | Velocity ODE | $\varphi$: learnable potential |
| $\dot{\ln p} = -\nabla^2\varphi$ | Density ODE | $\nabla^2$: Laplacian |
| $\partial_t p + \nabla\cdot(p\mathbf{v}) = 0$ | Continuity eq | $\mathbf{v}=\nabla\varphi$ |
| $\text{NLL} = -\mathbb{E}[\ln p(\mathbf{x},T)]$ | Density estimation | $T$: final time |
| $\mathcal{L} = \mathbb{E}[\ln p + E]$ | Variational free energy | $E$: energy function |
| $\varphi = \frac{1}{|G|}\sum_g \tilde\varphi(g\mathbf{x})$ | Symmetrization | $G$: symmetry group |

---

## 14. 总结：这篇paper的真正贡献

这篇paper在2018年提出了一个**概念上非常clean**的framework：

1. **理论motivation**: 从optimal transport的Brenier定理出发，有solid数学基础
2. **计算trick**: 连续时间极限把Hessian determinant简化为Laplacian
3. **物理interpretation**: compressible fluid的gradient flow，intuitive且visualizable
4. **Symmetry处理**: scalar potential的symmetrization，elegant且general
5. **Unifying view**: density estimation和variational inference用同一个framework

虽然后来的FFJORD、Flow Matching等在scalability上做得更好，但这篇paper的**conceptual clarity**仍然非常有价值。它把optimal transport、fluid dynamics、dynamical systems、control theory这几个看似不相关的领域用一个simple ODE system联系起来。

对于你build intuition，我建议重点关注：
- **Linearization trick** (Section 3): 理解为什么连续时间能简化Jacobian
- **Continuity equation** (Section 4): 理解probability mass的"流体"图像
- **Symmetrization** (Section 8.3): 理解scalar potential的symmetry优势
- **Control perspective** (Section 5): 理解training作为optimal control

Reference links:
- [Paper on arXiv](https://arxiv.org/abs/1809.00088)
- [Weinan E's homepage](https://web.math.princeton.edu/~weinan/)
- [Lei Wang's homepage](http://www.cphy.ac.cn/)
- [Neural ODE](https://arxiv.org/abs/1806.07366)
- [FFJORD](https://arxiv.org/abs/1810.01367)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Optimal Transport - Villani](https://link.springer.com/book/10.1007/978-3-540-71050-9)
- [Dynamic OT - Benamou-Brenier](https://link.springer.com/article/10.1007/s002110050002)

如果你想深入某个部分（比如symplectic integrator的细节、或者与后来Flow Matching的关系），我可以继续展开。
