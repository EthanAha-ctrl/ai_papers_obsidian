---
source_pdf: EMPMEmbodiedMPMforModeling and Simulation of Deformable Objects.pdf
paper_sha256: 25452280bba5b8ee5dfb53e1cb9946f6ae62c1c2553c65bd36975e2068cea922
processed_at: '2026-08-04T04:05:19-07:00'
target_folder: Physics
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好嘞 Karpathy，咱们撇开那些学术套话，用最直白的“人话”把这个 EMPM 系统从里到外拆解一下。这篇 paper 本质上就是教你**怎么让机器人通过看视频，自动搞懂面团、绳子、布料这些东西的物理脾气，然后在电脑里做一个一模一样的数字替身，最后用它来指挥机器人干活。**

---

### 1. 为什么我们要抛弃 Spring-Mass，拥抱 MPM？

想象一下你要让机器人捏面团。以前大家做 real-to-sim 通常有两种套路：

第一种是 **Spring-Mass system**（比如 PhysTwin）。它把物体简化成一大堆点，点之间用弹簧连着。这种模型有个致命弱点：弹簧拉长了会缩回去，但你把面团捏扁了，它是不会弹回来的。Spring-mass 完全没有 permanent deformation 的概念。而且弹簧拉断了就是断了，很难模拟撕开 pita bread 那种复杂的断裂面。

第二种是 **Learning-based neural dynamics**（比如 PGND）。直接拿一个 graph neural network 去学物体的运动。这玩意儿是个 black box，你需要喂海量的数据，而且换个稍微不一样的面团它就 Overfitting 崩掉了。

所以 EMPM 选择了第三条路：**Material Point Method (MPM)**。这是 continuum mechanics 里的经典方法。它不发明新物理，它就是用一种极其聪明的离散化方式，把最正宗的偏微分方程解出来。MPM 能统一处理 elastic（弹性）、elastoplastic（弹塑性）、fracture（断裂）和 contact（接触）。所以它才能在捏橡皮泥、撕面饼这种任务上秒杀 spring-mass。

---

### 2. MPM 到底在干什么？（建立 Intuition）

咱们直接看 MPM 是怎么跑的。你可以把 MPM 想象成“粒子”和“背景网格”的交替游戏。
为什么搞这么复杂？因为纯用粒子，你算每个粒子受周围多大的力非常费劲，要不断做 neighbor search。纯用网格，物体一运动，网格里的物质就混了，产生数值耗散。

所以 MPM 结合了两者的优点：**粒子携带物质信息，网格用来算力。**

咱们看论文里的公式，我把每个变量拆开讲：

#### Step 1: P2G (Particle to Grid) - 把质量泼到网格上
$$m_i^n = \sum_p w_{ip}^n m_p$$
$$m_i^n \boldsymbol{v}_i^n = \sum_p w_{ip}^n m_p (\boldsymbol{v}_p^n + C_p^n (\boldsymbol{x}_i - \boldsymbol{x}_p^n))$$

*   $p$ 是 particle 的下标，$i$ 是 grid node 的下标，$n$ 是第 $n$ 个时间步。
*   $w_{ip}^n$ 是 B-spline weight。你把一个粒子扔在网格里，它会对周围 27 个网格点（3x3x3）产生权重影响，离得越近 $w$ 越大。
*   $m_p$ 是粒子质量，$m_i^n$ 就是周围所有粒子把质量“泼”到网格点 $i$ 上的总和。
*   $\boldsymbol{v}_p^n$ 是粒子速度。$C_p^n$ 是 APIC (Affine Particle-in-Cell) 引入的 affine velocity term，它记录了粒子局部的速度梯度。有了 $C_p$，粒子不仅把速度传给网格，还把“速度怎么随空间变化”也传给了网格。这保证了动量守恒，避免系统能量凭空耗散。

#### Step 2: Grid Update - 在网格上算力
$$\boldsymbol{v}_i^{n+1} = \boldsymbol{v}_i^n + \frac{\Delta t}{m_i} f_i(\boldsymbol{x}_i^n; \theta)$$

*   $\Delta t$ 是时间步长。
*   $f_i$ 是网格点上的力。这里面大有玄机，包含了弹性力、重力，还有机器人推它的力。
*   $\theta = \{E, \nu, \rho, y\}$ 就是我们要优化的物理参数。$E$ 是 Young's modulus（硬度），$\nu$ 是 Poisson's ratio（抗压缩性），$\rho$ 是密度，$y$ 是 plastic yield stress（屈服应力，超过这个力就会永久变形）。

**Intuition: 算力靠的是 Deformation Gradient**
每个粒子都有一个 $3 \times 3$ 的矩阵 $F_p$，这叫 deformation gradient，记录了粒子从初始形态被捏成了什么样。论文用了 Fixed Corotated elasticity model：
$$\Psi(F) = \mu \|F - R(F)\|_F^2 + \frac{\lambda}{2}(\text{tr}(F) - 3)^2$$
$R(F)$ 是把 $F$ 做极分解提纯出来的旋转矩阵。因为如果你把一块橡皮泥整体转个 90 度，它内部是不应该有应力的。把 $R(F)$ 减掉，剩下的就是纯粹的拉伸或挤压。这个 $\Psi$ 算出来的能量求导就是 force。

#### Step 3: G2P (Grid to Particle) - 把速度拿回粒子
$$\boldsymbol{v}_p^{n+1} = \sum_i w_{ip}^n \boldsymbol{v}_i^{n+1}, \quad \boldsymbol{x}_p^{n+1} = \boldsymbol{x}_p^n + \Delta t \, \boldsymbol{v}_p^{n+1}$$

网格更新完速度 $\boldsymbol{v}_i^{n+1}$ 后，利用同样的 B-spline weight $w_{ip}^n$，把速度插值回粒子上。然后粒子用这个新速度更新自己的位置 $\boldsymbol{x}_p$。因为更新位置是 Lagrangian 的（直接相加），所以边界清晰，没有 numerical diffusion。

#### Step 4: Elastoplasticity - 怎么保留永久变形？
$$F_p^{\text{trial}, n+1} = \text{updateF}(F_p^{E,n}), \quad F_p^{E,n+1} = \text{returnMap}(F_p^{\text{trial}, n+1})$$

这里用了 continuum mechanics 里的经典操作：$F = F^E F^P$。
$F^P$ 是塑性形变（不可逆的），$F^E$ 是弹性形变（可恢复的）。当你捏面团时，先用 $F_p^{E,n}$ 算出一个 trial（试探）形变 $F_p^{\text{trial}, n+1}$。如果这个 trial 形变算出来的 von Mises stress 超过了 yield stress $y$，return mapping 算法就会把超出的部分“挤”到 $F^P$ 里去，只保留合法的弹性部分。所以面团被捏扁后，$F^P$ 记住了新的形状，放手后面团也不会弹回原样。

---

### 3. Differentiable Simulation: 怎么从视频里“学”物理？

MPM 算法是 50 年前就有的。这篇 paper 的灵魂在于 **Differentiable MPM**。
机器人拿着 RGB-D 相机拍了一堆视频，我们怎么知道面团的 $E$ 和 $\nu$ 到底是多少？

答案是把整个 MPM simulator 变成一个 differentiable layer。论文用了 NVIDIA 的 **Warp** 框架。
你随便瞎猜一个 $\theta$，让 MPM 跑 $T$ 步，得到预测的 particle 轨迹 $\hat{X}_{1:T}$。然后拿你从 RGB-D 视频里重建出来的 3D 点云 $\tilde{X}_{1:T}$ 跟它对比，算 Loss：

$$L_{\text{offline}} = \lambda_{\text{dist}} \sum_{t=1}^T \text{Chamfer}(\hat{X}_t, \tilde{X}_t) + \lambda_{\text{trk}} \sum_{t=1}^T \sum_{j \in \mathcal{T}_t} \|\hat{X}_{t,j} - \tilde{X}_{t,j}^{\text{trk}}\|_2^2$$

*   **Chamfer Distance**: 我不管你哪个点对应哪个点，只要两个点云的外形轮廓越近越好。这保证了 global shape 对齐。
*   **Tracking Loss**: 这是论文里极其关键的 supervision。我用 point tracking model（比如 Co-Tracker）追踪视频里某个特定的点 $j$，得到它的真实 3D 轨迹 $\tilde{X}_{t,j}^{\text{trk}}$。然后我要求 MPM 里对应的 particle $\hat{X}_{t,j}$ 必须沿着这个轨迹走。这就是 strong supervision，防止 MPM 为了骗过 Chamfer distance 把面团整体反转或者扭成麻花。

有了 Loss，因为 MPM 每一步的 P2G, Grid Update, G2P 都是可微的，你可以直接用 chain rule 把 gradient $\nabla_\theta L_{\text{offline}}$ 传回最开始的物理参数 $\theta$！然后用 AdamW 更新参数。跑几轮梯度下降，物理参数就收敛了。这就是 Differentiable Physics 的魔力。

---

### 4. Online Adaptation: 边捏边学

如果离线学完，面团放久了变干变硬了怎么办？EMPM 搞了个 online 优化。
实时跑的时候，因为没有时间做长时间的 point tracking，tracking loss 就失效了。于是他们用了这个 loss：
$$L_{\text{online}} = \lambda_{\text{dist}} L_{\text{dist}} + \lambda_{\text{mask}} L_{\text{mask}}$$
$L_{\text{dist}}$ 还是 Chamfer distance。$L_{\text{mask}}$ 是用 Grounded SAM2 在 2D 图像上分割出来的 mask，把模拟的粒子投影到图像上算 IoU。

**这里有个极具 Engineering 巧思的 trick：Quasi-static Assumption（准静态假设）**
如果机器人正在快速甩绳子，这时候你在 online 跑 simulation 去算 gradient，gradient 会非常噪杂甚至发散，因为 dynamic state 太复杂了。
作者就很鸡贼地让机器人**“捏住物体停顿一下”**。在物体受力平衡的 quasi-static 状态下，从当前状态往前跑 10 步 simulation，如果物理参数是对的，物体应该几乎不动（$\check{X}_{t+H} \approx \hat{X}_t$）。如果参数不对，模拟的面团会塌陷或者鼓起来。这时候算 loss 并 backprop，更新出来的参数非常稳定。这就是用静态的平衡态去 bypass 动态梯度的 chaos。

---

### 5. Rendering: 凑齐最后一块拼图 (3D Gaussian Splatting)

物理引擎算出来的只是一堆灰色的点。为了让渲染出来的画面好看，EMPM 整合了 3D Gaussian Splatting (3DGS)。
具体怎么 attach 呢？他们用了 **Linear Blend Skinning (LBS)**。
你在初始状态建一堆漂亮的 3D Gaussians，然后把每个 Gaussian 绑定到它最近的几个 MPM particles 上。当 MPM particles 动了，Gaussian 的中心坐标就根据周围 particles 的运动进行插值移动。这样 Gaussian 的运动完全受到 MPM physics 的约束。渲染出来的 pita bread 被撕开时，Gaussian 的纹理也随之撕裂，看起来 photorealistic。

---

### 6. 实验结果分析

看 Table 1 的数据，非常直观：
*   **Elastic Objects (绳子、布料)**: EMPM 和 PhysTwin 打平（Dist 0.0222 vs 0.0227）。因为弹性的东西 spring-mass 确实够用。
*   **Elastoplastic Objects (面团、橡皮泥)**: EMPM 碾压 PhysTwin（Dist 0.0082 vs 0.0177，IoU 0.7768 vs 0.6918）。因为 spring-mass 根本无法表达橡皮泥被捏扁后的 permanent shape change。
*   **Training Time**: EMPM 训练只要 161.8 秒，PhysTwin 要 496.8 秒，PGND（纯学习）竟然要 50675 秒！因为 EMPM 是把所有 frame 打包做 batch differentiable simulation 优化，而 PGND 要在那慢慢 train neural network。这就是 model-based prior 的降维打击。

---

### 7. 与你 Karpathy 的 Ideas 联结 (Broader Context)

这篇 paper 其实触及了你经常提的几个核心话题：

1.  **Software 2.0 meets Physics**: 我们在 differentiable physics 里看到的，本质上就是把一段显式的 C++ 代码（MPM 算法）变成了一个可以用 autograd 求导的 computation graph。Warp 和 Taichi 干的就是这个事。这就是 Software 2.0 在物理引擎上的完美体现——不学神经网络的 weights，直接学物理公式的 constants。
2.  **Model-based World Model**: 如果把 EMPM 看作一个 World Model，它是极其 sample efficient 的。只需要看一段 10 秒的视频，它就能把面团“想明白”。相对的，Sora 那种生成式 world model 需要看几百万小时的视频，而且物理一致性还不好。EMPM 的 inductive bias 是连续介质力学方程，这种强 prior 保证了它不会违反物理常识。
3.  **Soft-body MuJoCo**: 在 rigid body 时代，你做机器人 learning 离不开 MuJoCo。但在 deformable object 时代，MuJoCo 无能为力。EMPM 展示了一个潜在的 “Soft-body MuJoCo” 范式：用 MPM 模拟，用 Differentiable simulation 做 system identification，最后做 MPC 或者 RL。如果能做成一个像 MJX (MuJoCo XLA) 那样的 GPU-accelerated differentiable MPM 库，那 robotic manipulation 的 dataset 和 benchmark 会大爆发。

---

### 8. 吐槽与 Limitation

虽然这套系统很漂亮，但是坑也不少：
*   **Point Tracking 的脆弱性**: 论文在 Limitation 里也承认了。如果面团被严重遮挡，或者形变太大，Co-Tracker 这种模型在几秒后输出的点全都是错的。Tracking loss 一废，系统就只能靠 Chamfer distance 死撑，很容易掉进 local minima。这正是目前 vision 社区要去解的长时序、大形变 tracking 问题。
*   **Warp 显存爆炸**: Differentiable simulation 通过几百个 timestep 做 backprop，你需要把每一步的中间状态全存在显存里。如果粒子数多了（比如百万级），单张 A6000 绝对 OOM。所以 paper 里只敢做小物体。这也是为什么他们提供了 CMA-ES 的 zero-order 优化作为 fallback。
*   **接触摩擦的简化**: Coulomb friction 在快速抓取或滑动时其实误差挺大的，真正的 gripper 有橡胶软垫，摩擦力非常 non-linear，用单一参数去拟合它肯定不够准。

---

### 9. 参考链接

如果想深挖，可以直接看这些 source：

*   **EMPM Project Page**: https://embodied-mpm.github.io
*   **NVIDIA Warp (Differentiable Engine 核心)**: https://github.com/NVIDIA/warp
*   **PhysTwin (Spring-Mass Baseline 对比)**: https://github.com/phys-twin/PhysTwin
*   **3D Gaussian Splatting**: https://github.com/graphdeco-inria/gaussian-splatting
*   **Taichi (另一个极强大的 Differentiable MPM 框架)**: https://github.com/taichi-dev/taichi
*   **Grounded SAM 2 (Real-time Segmentation)**: https://github.com/IDEA-Research/Grounded-SAM-2
*   **APIC (Affine Particle-in-Cell) 原论文**: https://www.math.ucla.edu/~cffjiang/research/apic/apic.pdf
*   **PhysGaussian (同类的 GS + MPM 工作)**: https://github.com/yanqinJiang/PhysGaussian

总结一下，EMPM 证明了在 embodied AI 领域，扎实的连续介质力学 + differentiable programming 比 spring-mass 玩具模型和 black-box 神经网络都要好。它本质上是用最经典的物理公式，在 GPU 上构建了一个可以“对齐视觉”的数字双胞胎。这种把 first principles physics 和 visual reconstruction 无缝缝合的 pipeline，绝对是接下来机器人操作 deformable objects 的主流打法。

---

# EMPM: Embodied MPM 论文深度解读

好嘞 Karpathy！这篇 paper 我来给你彻底拆解一下，重点放在 build intuition 上，把 MPM 的物理本质、differentiable simulation 的 trick、loss 设计的思路、还有 real-to-sim-to-real pipeline 都串起来讲清楚。

## 1. Core Intuition: 为什么是 MPM，而不是 spring-mass 或者纯 learning-based

这篇 paper 的核心 thesis 可以这么理解：之前 deformable object 的 real-to-sim-to-real pipeline 基本走两条路——要么是 **spring-mass system**（PhysTwin 那一套），要么是 **neural dynamics**（PGND、ParticleNet 那一套）。两者都有致命问题：

- **Spring-mass** 把 continuum material 简化成离散弹簧网络，elastic spring 的本构关系本质上 cannot capture plastic flow（永久变形），也无法自然 accommodate fracture（弹簧不能断）。pita bread 撕裂、plasticine 捏扁这种 elastoplastic + fracture 场景它就崩了。
- **Neural dynamics** 是 black-box statistical approximator，需要海量训练数据，OOD generalization 差，而且物理参数不可解释。

EMPM 选择第三条路：用 **MPM (Material Point Method)**，这是 continuum mechanics 里的 hybrid Lagrangian-Eulerian 方法。particle 携带 material state（mass、momentum、deformation gradient $F$），background grid 负责 force computation。这种 hybrid 设计的好处是：Lagrangian 的 advection 自然避免 Eulerian 的 numerical diffusion，Eulerian grid 又避免了纯 Lagrangian (SPH 那种) 的 neighbor search 痛点。最关键的是 MPM 能统一处理 **hyperelastic + elastoplastic + fracture + contact**，这是 spring-mass 永远做不到的。

## 2. MPM 数学细节：每一步在干什么

我把 paper 里的 equation 串起来讲，把变量都解释清楚。

### 2.1 State representation

每个 particle $p$ 在 time step $n$ 携带：
- $\boldsymbol{x}_p^n \in \mathbb{R}^3$: position
- $\boldsymbol{v}_p^n \in \mathbb{R}^3$: velocity  
- $F_p^n \in \mathbb{R}^{3\times 3}$: deformation gradient（这是 continuum mechanics 的核心，描述 material element 从 rest shape 到 current shape 的线性映射）
- $C_p^n \in \mathbb{R}^{3\times 3}$: APIC affine velocity term（用来保留 velocity field 的局部 affine 结构，比 PIC 的零阶更精确，又比 FLIP 的 noise 更干净）
- $m_p$: mass

Material parameters $\theta = \{E, \nu, \rho, y\}$：
- $E$: Young's modulus（弹性模量，描述 stiffness，paper 里 normalize 到 $[0,1]$ by $10^6$，相当于单位是 MPa）
- $\nu$: Poisson's ratio（横向变形/纵向变形，不可压缩材料 $\nu \to 0.5$）
- $\rho$: density
- $y$: plastic yield stress（超过这个 stress 就发生 permanent deformation）

### 2.2 P2G (Particle to Grid) — Eq 2

$$m_i^n = \sum_p w_{ip}^n m_p, \quad m_i^n \boldsymbol{v}_i^n = \sum_p w_{ip}^n m_p \big(\boldsymbol{v}_p^n + C_p^n (\boldsymbol{x}_i - \boldsymbol{x}_p^n)\big)$$

变量解释：
- $w_{ip}^n$: B-spline weight，描述 particle $p$ 对 grid node $i$ 的贡献权重。quadratic B-spline 的 support 是相邻 3×3×3 = 27 个 grid nodes
- $m_i^n$: node $i$ 的累积质量
- $\boldsymbol{v}_i^n$: node $i$ 的累积速度（通过 APIC，速度不是常数，而是 affine function of position，$\boldsymbol{v}(\boldsymbol{x}) = \boldsymbol{v}_p + C_p (\boldsymbol{x} - \boldsymbol{x}_p)$）

**Intuition**: 这一步是把 Lagrangian particles 的 mass 和 momentum "溅" 到 Eulerian grid 上。B-spline 是 $C^1$ 连续的，保证 P2G/G2P 往返不丢能量也不引入数值耗散。

### 2.3 Grid update — Eq 3

$$\boldsymbol{v}_i^{n+1} = \boldsymbol{v}_i^n + \frac{\Delta t}{m_i} f_i(\boldsymbol{x}_i^n; \theta)$$

$f_i$ 是 grid node $i$ 上的合力，内部包含：
1. **Elastic force**：从 $F_p$ 计算 stress $\sigma_p$，再 transfer 到 grid。这里用的是 **Fixed Corotated (FC) elasticity model**：
   $$\Psi(F) = \mu \|F - R(F)\|_F^2 + \frac{\lambda}{2}(\text{tr}(F) - 3)^2$$
   其中 $R(F)$ 是 $F$ 的 polar decomposition 的旋转部分（$F = R \hat{F}$，$R \in SO(3)$，$\hat{F}$ symmetric positive definite）。$\mu, \lambda$ 是 Lamé parameters，由 $E, \nu$ 转换得到：$\mu = \frac{E}{2(1+\nu)}$, $\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}$。
   
   **Intuition**: FC 模型的好处是 rotation-invariant，物体整体旋转不会产生 spurious stress，比 linear elasticity (Saint-Venant Kirchhoff) 稳定得多，比 Neo-Hookean 又简单得多（不需要 SVD）。

2. **Plasticity via von Mises return mapping** — Eq 6：
   $$F_p^{\text{trial}, n+1} = \text{updateF}(F_p^{E,n}), \quad F_p^{E,n+1} = \text{returnMap}(F_p^{\text{trial}, n+1})$$
   
   $F = F^E F^P$ 分解，$F^P$ 是 "rest shape memory"（plastic 部分），$F^E$ 是 elastic 偏离这个 rest shape 的部分。trial $F^E$ 计算后，检查 von Mises equivalent stress $\sigma_{vm} = \sqrt{\frac{3}{2} \|\text{dev}(\sigma)\|_F^2}$ 是否超过 yield stress $y$。超过就 project 回 yield surface，剩余的 deformation 进入 $F^P$。
   
   **Intuition**: 这是 classic radial return mapping 算法，弹塑性 decomposition 的 key insight 是 plastic flow 不改变 volume（对金属、dough 这种），所以 $F^P$ 的 determinant 通常是 1。

3. **Boundary conditions**: Robot gripper / hand 速度作为 Dirichlet condition 强制施加在 contact nodes。Coulomb friction $f_t \leq \mu_c f_n$ 处理 table / gripper 接触。

### 2.4 G2P (Grid to Particle) — Eq 4

$$\boldsymbol{v}_p^{n+1} = \sum_i w_{ip}^n \boldsymbol{v}_i^{n+1}, \quad \boldsymbol{x}_p^{n+1} = \boldsymbol{x}_p^n + \Delta t \, \boldsymbol{v}_p^{n+1}$$

把 grid 上的新速度 transfer 回 particles，然后 advection。这里 advection 是 Lagrangian 的（直接 $\boldsymbol{x}_p \mathrel{+}= \Delta t \, \boldsymbol{v}_p$），没有 numerical diffusion。

### 2.5 Deformation gradient update — Eq 5

$$F_p^{n+1} = \big(I + \Delta t \sum_i \boldsymbol{v}_i^{n+1} (\nabla w_{ip}^n)^\top\big) F_p^n$$

这里 $\nabla w_{ip}^n$ 是 B-spline 的空间梯度，$\sum_i \boldsymbol{v}_i^{n+1} (\nabla w_{ip}^n)^\top$ 实际上就是 grid 上 velocity field 的 gradient（通过 B-spline 插值），加上 identity 后就是 incremental deformation $\Delta F$，乘以 $F_p^n$ 就是 updated deformation gradient。

**Intuition**: 这个公式是 MPM 的灵魂——它告诉你 particle 的 deformation 在这一步被 grid velocity field 怎么 stretch 了。$F$ 的演化是 purely kinematic 的（只跟 velocity field 有关），force 计算反过来用 $F$ 通过 constitutive model 得到 stress。

## 3. Differentiable MPM: 怎么 backprop through simulation

这是 EMPM 的 key technical contribution。传统 MPM simulator 不可微，你没法 gradient-descend material parameters。EMPM 用 **NVIDIA Warp** [32]，它提供：
- Auto-differentiation through simulation steps
- PyTorch integration（forward 用 Warp kernel，backward 自动）
- GPU acceleration

Loss 是 Eq 8：
$$L_{\text{offline}} = \lambda_{\text{dist}} \sum_{t=1}^T \text{Chamfer}(\hat{X}_t, \tilde{X}_t) + \lambda_{\text{trk}} \sum_{t=1}^T \sum_{j \in \mathcal{T}_t} \|\hat{X}_{t,j} - \tilde{X}_{t,j}^{\text{trk}}\|_2^2$$

- $\hat{X}_t$: MPM 预测的 particle 位置
- $\tilde{X}_t$: 从 RGB-D 重建的 point cloud
- $\tilde{X}_{t,j}^{\text{trk}}$: 第 $j$ 个 tracked point 的位置（通过 point tracking model 算的 3D trajectory）
- $\mathcal{T}_t$: time $t$ 上有效的 tracked points（被 occlusion mask 过滤）

**Two-term loss 的设计 intuition**：
- Chamfer distance 是 global shape alignment，但它不知道哪个 particle 对应哪个 observation point（correspondence-free），容易陷入 local minima（比如把整个物体翻转）
- Tracking loss 是 point-wise correspondence，给每个 particle 一个明确的 target，是 strong supervision
- 两者结合：Chamfer 防止 tracking 失败时崩盘，tracking 防止 Chamfer 退化到 trivial solution

**Backprop 的核心**: $\nabla_\theta L_{\text{offline}}$ 通过 chain rule 穿过整个 MPM rollout。每个 time step 的 P2G、grid update、G2P、F update 都是 differentiable operation，Warp 会自动 accumulate gradient。Memory cost 大致是 $O(\text{particles} \times \text{steps} \times \text{state size})$，对 1000 particles、100 steps 大概几个 GB。

Paper 也提供 zero-order alternative：**CMA-ES** on forward simulator。好处是不需要 memory（forward-only），可以更 aggressive 探索 parameter space。坏处是 sample efficiency 差。这种 hybrid 策略在 system identification 里很常见。

## 4. Online Adaptation: quasi-static trick

Online optimization 是这 paper 的另一个 contribution。挑战是 streaming video 里 point tracking 不可靠（occlusion 多），所以 tracking loss 用不了。Eq 9：

$$L_{\text{online}} = \lambda_{\text{dist}} L_{\text{dist}} + \lambda_{\text{mask}} L_{\text{mask}}$$

- $L_{\text{dist}}$: 3D Chamfer distance
- $L_{\text{mask}}$: 2D segmentation mask loss（用 Grounded SAM2 [36] 算的 mask），把 simulated particles render 到 image plane，跟 segmentation mask 算 IoU 或 cross-entropy

**Key insight — quasi-static assumption**:
当物体被 gripper 夹住处于 equilibrium 时（balanced forces），如果 material parameters 正确，forward simulate 几步应该几乎不动：$\check{X}_{t+H} \approx \hat{X}_t$。如果 parameters 错了，比如 stiffness 太小，simulation 会"陷"或者"鼓"，跟真实 observation 偏离。

所以每 5 个 streaming step 做一次 optimization，每次 simulate 10 步 forward，compute $L_{\text{online}}$，backprop update $\theta$。这个 trick 把 online optimization 从 unstable dynamic state 转移到 stable equilibrium state，gradient 噪声小很多。

## 5. 3DGS Integration: appearance 怎么 attach 到 MPM

Geometry 上 MPM particles 提供位置，但 appearance 需要 Gaussian Splatting [2]。Paper 用 **Linear Blend Skinning (LBS)** [8] 把 MPM particle motion 插值到 Gaussian centers：

$$\boldsymbol{x}_{\text{Gauss}} = \sum_k w_k(\boldsymbol{x}_{\text{Gauss}}^{\text{rest}}) \, \boldsymbol{x}_{p_k}^{\text{current}}$$

每个 Gaussian center 在 rest shape 里 bind 到几个最近的 MPM particles（$w_k$ 是 barycentric-like weight），运动时 follow particles 的 affine transform。这样 Gaussian 的形变跟 MPM physics 一致，render 出来既 photorealistic 又 physics-consistent。

类似工作有 PhysGaussian [14]、SpringGaus [25]，EMPM 跟它们的区别是用 MPM 而不是 spring-mass 或 graph，所以能处理 elastoplastic + fracture。

## 6. Experimental Analysis

Table 1 给出 quantitative 对比，几个关键 takeaways：

| Metric | EMPM | PhysTwin | PGND |
|---|---|---|---|
| Elastic Dist ↓ | **0.0222** | 0.0227 | 0.0618 |
| Elastoplastic Dist ↓ | **0.0082** | 0.0177 | 0.0245 |
| Elastoplastic IoU ↑ | **0.7768** | 0.6918 | 0.5069 |
| Elastoplastic PSNR ↑ | **27.82** | 27.01 | 21.26 |
| Training time (elastic) ↓ | **161.8s** | 496.8s | 50675.5s |

**Intuition**:
- **Elastic object** 上 EMPM 和 PhysTwin 差不多（都是 0.022 左右），因为 elastic deformation spring-mass 也能建模
- **Elastoplastic object** 上 EMPM 大幅领先（0.0082 vs 0.0177，~2x 改进；IoU 0.78 vs 0.69），因为 MPM 的 elastoplastic decomposition 自然支持 permanent deformation，spring-mass 没有这个 mechanism
- **PGND** 在所有 metric 上都差很多，因为 learning-based 需要 large training set，generalization 不好。但 inference 最快（5s），因为纯 feedforward
- **Training efficiency**: EMPM 比 PhysTwin 快 3x（161.8s vs 496.8s），比 PGND 快 300x。原因是 EMPM 是 batch optimization over all frames，而 PhysTwin 是 frame-by-frame，PGND 是 train 10 episodes 的 neural network

Table 2 的 online optimization 结果：
- Rope: $L_{\text{mask}}$ 0.0456 → 0.0428（~6% 改进）
- Bread dough: $L_{\text{mask}}$ 0.0031 → 0.0024（~23% 改进）

改进幅度不大但 consistent，说明 online correction 确实 work。Bread dough 改进更大可能是因为 plastic deformation 对 stiffness 更 sensitive。

## 7. Failure Modes & Limitations

Paper 自己承认的 limitation：**point tracking under occlusion**。tracked points 在几秒后经常全部失效，online 优化时只能 rely on Chamfer + mask。这是 tracking-based loss 的 fundamental 问题。Emerging tracking models (Co-Tracker3 [19], MegaSaM [18]) 可能能缓解，但 deformable object 的 long-range tracking 本身就是 open problem。

潜在的其他 limitations（paper 没明说但我推断）：
- **MPM resolution vs Gaussian resolution 的 mismatch**: MPM particles 通常几千个，3DGS 需要 ~100k Gaussians for photorealistic rendering，LBS 插值可能 smooth 掉 fine detail
- **Coulomb friction 的 contact model 简化**: 真实 gripper 接触有 compliance、friction anisotropy，paper 用简单 Coulomb 可能不够准
- **Single set of material parameters $\theta$**: 假设物体 homogeneous，但 dough 这种 object 内部 density 可能 vary
- **Quasi-static assumption 限制了 online optimization 的适用场景**: 高速 dynamic interaction 下 online 优化可能不稳定

## 8. Broader Context: 这 paper 在 landscape 里的位置

EMPM 处在几个 trend 的交汇点：

1. **Differentiable physics simulation**: ChainQueen [29]、DiffTaichi [31]、Warp [32]、Brax、MJX 这条线。核心 insight 是把 physics simulator 当作 differentiable layer，用 gradient descent 做 system identification 或 optimal control。这是 "Software 2.0" 思路 applied to physics。

2. **Radiance field + physics**: PhysGaussian [14]、PAC-NeRF [15]、PhysDreamer [16]、PIE-NeRF [24]、SpringGaus [25]。EMPM 的 differentiation 是用 MPM 而不是 FEM 或 spring-mass，trade-off 是 MPM 对 large deformation + fracture 更 friendly，FEM 对精确 stress 更准。

3. **Real-to-sim-to-real for manipulation**: PhysTwin [8]、Real-is-Sim [7]、PEGG [6]。EMPM 的 contribution 是把 digital twin 从 elastic-only 扩展到 elastic + elastoplastic + fracture，覆盖了更多 manipulation scenario（dough kneading, tearing, squeezing）。

4. **Embodied AI with physics prior**: 跟你 Karpathy 之前讲的 "world model" 思路有 connection。EMPM 是 model-based world model，用 physics simulator 当 prior，parameters 通过 sensory data identify。vs learning-based world model (Dreamer、GAIA、Genie) 是 model-free 的，纯 data-driven。EMPM 的优势是 sample efficiency（只需一段 video）和 interpretability（$E, \nu, y$ 是物理量），劣势是 model class 受限于 MPM 假设的 continuum mechanics。

## 9. 公式再深一点：APIC 的 $C_p$ 怎么 update

Paper 里没写 APIC 的 $C_p$ update，但这是 implementation detail 里 important 的部分。APIC 的 $C_p^{n+1}$ 是从 grid velocity gradient 算的：

$$C_p^{n+1} = \sum_i \boldsymbol{v}_i^{n+1} (\nabla w_{ip}^n)^\top \cdot (\boldsymbol{x}_i - \boldsymbol{x}_p^n)^\top$$

或者更精确的 APIC formulation 是 $C_p^{n+1} = \frac{1}{4 \Delta x^2} \sum_i \boldsymbol{v}_i^{n+1} (\boldsymbol{x}_i - \boldsymbol{x}_p^n)^\top$（quadratic B-spline 的 specific constant）。

**Intuition**: $C_p$ 捕获 local velocity field 的 affine 部分（线性 gradient + translation）。下次 P2G 时把这个 affine 信息传回 grid，避免了 PIC 的 energy dissipation 和 FLIP 的 noise。APIC 的 momentum conservation 是 discrete-level exact 的，这对 long rollout 稳定性 critical。

## 10. Implementation 细节再聊

- **GPU**: 单张 Nvidia A6000 (48GB) 就够训练
- **Optimizer**: AdamW, lr = $10^{-4}$
- **E normalization**: $E \in [0,1]$ 通过 $10^{-6}$ scaling，相当于 $E$ 的真实单位是 MPa（橡胶 ~1 MPa, dough ~0.1 MPa, 塑料 ~1000 MPa）
- **Camera rig**: 3× RealSense D455, multi-view RGB-D
- **Segmentation**: Grounded SAM2 [36] (text prompt → mask)
- **Tracking**: off-the-shelf point tracking model (paper 没指明，可能是 Co-Tracker 或 TAPIR 类)
- **Robot**: bimanual Franka arms with parallel-jaw grippers

## 11. 你可能的 next questions

你可能想问：

1. **为什么 MPM 比 FEM 好？** FEM mesh 在 large deformation 下会 invert，需要 remeshing，复杂且破坏 differentiability。MPM 是 meshless（particles），deformation 全在 $F$ 里，没有 mesh topology 问题。FEM 在 small deformation linear regime 更精确，MPM 在 large deformation + topology change 更鲁棒。

2. **Differentiable simulation 的 memory bottleneck？** 是的，forward pass 的所有 intermediate state 都要存。Warp 有 checkpointing/gravity-Kill 选项，但 paper 没提。这是为什么他们也提供 CMA-ES 作为 zero-order alternative。

3. **跟你的 μ-PPL 或者 nanoGPT 类比？** 这 paper 像是 "nanoMPM"——minimal but correct implementation，把 differentiable MPM + RGB-D + 3DGS 拼成 usable pipeline。它不是 push frontier of physics（MPM 是 50 年的方法），而是 push frontier of **integration**: 把成熟 physics + 成熟 vision + 成熟 differentiable programming 拼成 real-to-sim-to-real loop。

4. **能不能扩展到 fluid？** 原则上可以，MPM 对 fluid (WCMPM、Implicit APIC) 有成熟 formulation。但 fluid 的 $F$ update 和 elastic 不同，需要 equation of state 而不是 constitutive model。Paper 没做，留作 future work。

5. **能不能 integrate 到 model predictive control (MPC)？** Paper 在 Section 3.5 提到了这个 application，但没实现。理论上 EMPM 提供 action-conditioned rollout，可以 query cost function (target geometry, collision) over candidate trajectories $\boldsymbol{u}_{t:t+H}$，用 trajectory optimization 或 sampling-based planner。Challenges 是 MPM forward pass 10-22ms (per Table 3 testing time / frames)，对 real-time MPC 可能太慢，但可以 offline plan。

## 12. 参考链接

- **Project page**: https://embodied-mpm.github.io
- **3D Gaussian Splatting**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **gsplat library**: https://github.com/nerfstudio-project/gsplat
- **NVIDIA Warp**: https://github.com/NVIDIA/warp
- **Taichi (alternative differentiable MPM)**: https://github.com/taichi-dev/taichi
- **SAM2**: https://github.com/facebookresearch/sam2
- **PhysTwin (baseline)**: https://github.com/phys-twin/PhysTwin （ICCV 2025）
- **PhysGaussian (related)**: https://github.com/yanqinJiang/PhysGaussian
- **PhysDreamer (related)**: https://github.com/yanqinJiang/PhysDreamer
- **APIC paper**: https://www.math.ucla.edu/~cffjiang/research/apic/apic.pdf
- **MPM course (Jiang et al. 2016)**: https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf
- **Co-Tracker3**: https://github.com/facebookresearch/co-tracker
- **RealSense D455**: https://www.intelrealsense.com/depth-camera-d455/
- **Franka Emika Panda**: https://www.franka.de/

---

整体上，EMPM 是个 well-engineered system paper，把 differentiable MPM、3DGS、RGB-D perception、robot telemetry 串起来，提供 elastic + elastoplastic + fracture 的 unified framework。它的 contribution 主要在 integration 和 online adaptation，physics 本身没有突破，但 engineering polish 很高，bimanual manipulation demo 也很 compelling。对 robotic manipulation of deformable objects 这个领域，这是个 solid step forward，尤其是把 elastoplastic material 第一次带进 real-to-sim-to-real pipeline。

如果你 want 更深挖某一块（比如 APIC 的 derivation、FC elasticity 的 polar decomposition、return mapping 算法、或 Warp 的 autodiff internals），我可以展开讲。
