---
source_pdf: TacEx GelSight Tactile Simulation in Isaac Sim –.pdf
paper_sha256: 6ee37818d248cc27a050da4a8ca3bfb2f9d8ecba2e068d0626dd25b965740da3
processed_at: '2026-08-12T12:09:32-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 TacEx

## 这篇 paper 在干嘛？

想象你要训练一个 robot 用手指头去摸东西、抓东西。Robot 手指头上贴了一层**软软的 gel pad**，里面藏着个 camera，能"看到" gel 被压扁的样子——这就是 **GelSight sensor**，一个 vision-based tactile sensor。

问题来了：**你不想每次都拿真 robot 去试**，摔坏了心疼，而且慢。你想在 simulation 里训练，然后迁移到真 robot 上（Sim2Real）。

但是 simulate 触觉这件事**特别难**，因为：

1. Gel pad 是软的，被压会变形，你得 simulate 这个变形——这是 **soft body physics**
2. Camera 看到的不是真实物体，是 gel 表面被压出来的"地形"，你得把这个"地形"渲染成 RGB 图像——这是 **optical simulation**
3. Gel 表面还印着一堆 marker dots，它们会跟着 gel 一起动，你得算它们怎么动——这是 **marker simulation**

已有的 tactile simulators 各搞各的：有的用 PyBullet，有的用 MuJoCo，有的自己写 FEM，sensor 型号也五花八门。你**没法公平比较**，也没法把它们组合起来用。

TacEx 就是来收拾这个烂摊子的。

---

## TacEx 的核心 idea：把三个东西拆开，塞进 Isaac Sim

直觉上，TacEx 说：**触觉仿真其实是三件正交的事**，为啥非要绑在一起？

```
[Physics]  →  gel 怎么变形
[Optical]  →  变形后 camera 看到啥 RGB 图
[Marker]   →  marker dots 怎么跟着动
```

这三件事**可以独立选型**。你想要快？用 rigid body 近似。你想要准？用 FEM soft body。你想换 sensor 型号？改 optical calibration 就行，physics 不用动。

然后，把这三层全部塞进 **NVIDIA Isaac Sim** 这个生态。为啥选 Isaac Sim？因为它自带：

- Photorealistic rendering（camera 渲染质量高）
- GPU 并行 physics（PhysX）
- ROS 支持（跟真 robot 对接方便）
- Isaac Lab 提供 RL training framework

参考链接：[Isaac Sim](https://developer.nvidia.com/isaac/sim)、[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)

---

## Physics：三种选择，一个比一个贵

### 选项 1：Rigid body + compliant contact（最便宜）

把 gel pad 当成**硬邦邦的 rigid body**，接触的时候用一个"弹簧"模型模拟软的感觉。快到飞起——1024 个并行环境每帧只要 0.0093 ms。

**缺点**：不准。Gel pad 真的是软的，rigid body 模型根本捕捉不到 deformation 的细节。但如果你只是想快速 prototype 一个 RL environment，这就够用了。

### 选项 2：PhysX 内置 soft body（中等）

Isaac Sim 自带 FEM-based soft body simulation。听起来很美好，论文试了一下发现**根本没法用**：

> 用两个 soft gel pad 去抓东西，**物体永远滑掉**，怎么调参数都没用。

原因：**PhysX soft body 不支持 static friction**。没有静摩擦力，你抓不住任何东西。这是个 deal-breaker。

所以这篇 paper 其实给 Isaac Sim soft body 模拟器打了个**差评**——目前还不能用于 contact-rich manipulation。

### 选项 3：GIPC（最准但最贵）

GIPC 是 **IPC（Incremental Potential Contact）** 的 GPU 加速版。参考：[GIPC paper](https://doi.org/10.1145/3643028)、[IPC original](https://doi.org/10.1145/3386569.3392425)

IPC 的核心 idea 用人话讲就是：

> 在每一步，找一个让 gel pad "最舒服"的位置——总能量最低的位置。但是有一条铁律：**任何两个东西不能互相穿透**。如果快要穿透了，就有一个"barrier"能量爆炸式增长，把物体弹开。

数学上就是每一步求解：

$$\min_{x^{t+1}} \underbrace{\frac{1}{2\Delta t^2}\|x - \tilde{x}^t\|_M^2}_{\text{惯性}} + \underbrace{E_{elastic}(x)}_{\text{弹性}} + \underbrace{E_{barrier}(x)}_{\text{防穿透}} + \underbrace{E_{friction}(x)}_{\text{摩擦}}$$

变量解释：
- $x$：所有 vertex 的新位置（要求解的未知数）
- $\tilde{x}^t = x^t + \Delta t \cdot v^t$：根据当前速度预测的位置
- $\Delta t$：时间步长
- $M$：mass matrix

**Barrier function** 是关键，长这样：

$$b(d) = -\kappa(d - \hat{d})^2 \ln\left(\frac{d}{\hat{d}}\right), \quad 0 < d < \hat{d}$$

- $d$：两个 surface 之间的距离
- $\hat{d}$：barrier 开始生效的距离（比如 0.1 mm）
- $\kappa$：stiffness，自动调节

当 $d \to 0$（快要穿透）时，$b(d) \to +\infty$，物理上就不可能穿透。

**为什么这个对 RL 特别重要？** 因为 RL 训练经常用 **domain randomization**——你故意把 material parameters、friction coefficient 随机扰动，让 policy 鲁棒。如果 physics engine 不 robust，参数一乱 simulation 就爆炸，RL 直接崩。IPC 保证**无论你怎么乱调参数，simulation 都不会 blow up**，这是巨大的工程优势。

GIPC 的代价：**VRAM 占用巨大**，论文里只能在 single environment 下跑，mesh 分辨率 12k vertices 时每帧 221 ms。跟 PhysX rigid body 的 0.0093 ms 比差了 4-5 个数量级。

---

## Optical：怎么把 gel 变形变成 RGB 图

这一层用的是 **Taxim**，思路很直觉：

1. 在 gel pad 内部放个 camera，朝外看
2. Camera 渲染 depth map——就是物体表面离 camera 多远
3. 算出物体压入 gel 多深（indentation depth）
4. 用 Gaussian kernel 平滑这个 depth map，模拟 gel 的弹性变形传播
5. 从变形场算出 surface normal（每个点的法向量）
6. 用一个**预先标定好的 polynomial lookup table**，把 normal 向量映射成 RGB 值

$$I_c(u,v) = f_c(n_x, n_y, n_z), \quad c \in \{r, g, b\}$$

- $n = (n_x, n_y, n_z)$：pixel $(u,v)$ 处的 surface normal
- $f_c$：calibration 阶段拟合的多项式

为啥这个方法好？因为它**只需要少量真实 tactile 图像做 calibration**，不像 GAN 或 diffusion model 那样需要海量数据。而且每一步都有物理意义，可解释。参考：[Taxim](https://arxiv.org/abs/2109.04027)

---

## Marker：怎么算那些小点点的运动

GelSight 表面印着一堆小 marker dots，它们随 gel 一起变形，运动轨迹反映了 gel 受到什么力。FOTS 用了一个特别聪明的简化：

$$d_{normal}(r) = A_n \exp(-\alpha_n r)$$

- $r$：marker 到 contact center 的距离
- $A_n$：normal force 大小决定的 amplitude
- $\alpha_n$：spatial decay（离接触中心越远，位移越小）

类似的还有 shear 和 twist 的 exponential model。总位移是三者叠加。

**为什么这个设计聪明？** 因为它**不依赖准确的 FEM simulation**。你用 rigid body 也好，soft body 也好，只要能给出 contact center 和 contact area，FOTS 就能算 marker flow。这给了 TacEx 极大的灵活性——marker simulation 跟 physics decoupled 了。

参考：[FOTS](https://arxiv.org/abs/2305.03429)

---

## 工程实现：怎么把 GIPC 塞进 Isaac Sim

这是论文里最 hidden engineering value 的部分。

**问题**：GIPC 是个独立的 C++ solver，Isaac Sim 有自己的 physics loop（PhysX），怎么让它们协同工作？

**TacEx 的方案**：

1. **Gel pad 的"背面"（贴在 sensor case 上的部分）由 PhysX 管运动**——robot 动了，sensor case 动了，gel pad 背面跟着动（kinematic）
2. **Gel pad 的"前面"（接触物体的部分）由 GIPC 算变形**——根据背面位置和物体接触，解 FEM
3. **Attachment points 怎么找？** 用 PhysX 的 sphere ray casting，在 gel pad tet mesh 的每个 vertex 处发射一个极小 sphere，如果 hit 到 sensor case 的 collider，就标记为 attachment point
4. **每一步更新顺序**：PhysX step → 查 sensor case pose → 算 attachment point 新位置 → GIPC solve → 更新 USD mesh → render

**USDRT API** 用来快速更新 mesh vertex positions，绕过 USD 的序列化开销。参考：[USDRT docs](https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usd_fabric_usdrt.html)

这个设计让 GIPC 成为一个"插件式"的 soft body solver，而 Isaac Sim 仍然负责 scene management、rendering、robot control。

---

## 实验告诉我们什么

### Ball Rolling（球滚动）

- Rigid body：能跑 18 个并行环境，然后 VRAM 爆了
- GIPC soft body：只能跑 1 个环境

**Insight**：bottleneck 不是 physics，是 **camera simulation**——每个 environment 需要一个 USD camera 渲染 height map，VRAM 占用巨大。

### Object Lifting（抓东西）

- PhysX soft body：**抓不住**，物体永远滑掉
- 这是 negative result，但很有价值

**Insight**：Isaac Sim 内置的 soft body 目前不能用于 grasping，因为缺 static friction。

### Beam Twisting（拧梁）

- 用 GIPC 把一根软梁拧成麻花还能弹回来
- 展示了 IPC 在 extreme deformation 下的 robustness

### RL Environments

论文坦白说：**还没训出成功的 tactile-feedback policy**，只是验证 pipeline 能跑。

这个诚实很重要。说明 tactile RL 目前**真的很难**，即使 simulation infrastructure 搭好了，high-dimensional tactile observation 对 RL 仍然是 challenge。

---

## 性能数据的直觉

| 方案 | 速度 | 能跑多少 env |
|---|---|---|
| PhysX rigid | 0.0093 ms/frame @ 1024 envs | 1024+ |
| PhysX soft | 0.1267 ms/frame @ 128 envs | 256 OOM |
| GIPC soft | 24-221 ms/frame @ 1 env | 1 |

**残酷的现实**：如果你想大规模 RL 训练（需要上千 envs），只能用 rigid body approximation。Soft body（尤其 GIPC）太贵了。

Table 1 的 tactile simulation 速度也很有意思：16 envs 之前 GPU parallelization 有效，16 envs 之后 VRAM thrashing 导致性能暴跌。这说明**VRAM 是核心 bottleneck**，不是 compute。

---

## 我的判断

### 这篇 paper 的真正贡献

**不是算法创新，是系统集成 + 工程实现**。它做了三件事：

1. **第一次把 GIPC（GPU IPC）集成进 Isaac Sim**，这个工程量不小
2. **证明了 modular tactile simulation 是可行的**——physics、optical、marker 三层可以解耦
3. **暴露了 Isaac Sim soft body 的 limitation**——缺 static friction，不能 grasping

### 它揭示的 bigger picture

Tactile RL 目前面临一个 **fundamental trade-off**：

```
Fidelity (GIPC)  ←——————→  Speed (PhysX rigid)
     ↑                              ↑
  单环境，慢但准              多环境，快但糙
  无法大规模 RL               可以大规模 RL 但失真
```

目前的 practical solution 可能是：**rigid body 大规模预训练 + soft body fine-tuning**。类似 curriculum learning 的思路。

### 跟 Sim2Real 的关系

论文承认**没有做 Sim2Real 实验**，这是个 big caveat。但 IPC 的 robustness 保证理论上对 Sim2Real 友好——你可以 aggressive domain randomization 而不担心 simulation 爆炸。

### 跟 TacSL 的对比

TacSL（NVIDIA + Meta 的工作，[link](https://arxiv.org/abs/2408.06506)）也做类似的事，但用 simplified soft contact model 代替 FEM。TacEx 用 GIPC 更准，TacSL 更轻量。这反映了一个经典 tension：**fidelity vs speed**。

---

## 一句话总结

TacEx 是一个**把触觉仿真的三件事（physics + optical + marker）解耦、塞进 Isaac Sim 生态**的工程框架，核心贡献是 GIPC 集成和 modular design，但暴露了 tactile RL 当前面临的 fidelity-speed trade-off 和 Isaac Sim soft body 的 limitation。它更像是一个 **infrastructure paper**，为后续 tactile RL 研究铺路，而不是提出一个新算法。

参考链接汇总：
- [TacEx 项目主页](https://sites.google.com/view/tacex)
- [GIPC paper](https://doi.org/10.1145/3643028)
- [IPC original](https://doi.org/10.1145/3386569.3392425)
- [Taxim](https://arxiv.org/abs/2109.04027)
- [FOTS](https://arxiv.org/abs/2305.03429)
- [TacSL](https://arxiv.org/abs/2408.06506)
- [DiffTactile](https://arxiv.org/abs/2306.03146)
- [TacIPC](https://arxiv.org/abs/2403.06115)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)

---

# TacEx: GelSight Tactile Simulation in Isaac Sim 深度解析

## 1. 论文核心动机与定位

TacEx 解决的是 **tactile-rich manipulation** 中一个关键的工程瓶颈：当前 tactile simulators 各自为政，使用不同的 physics engine、不同的 sensor model、不同的 robotics framework，导致 **interoperability 极差**，研究者很难做 fair comparison。

论文的定位非常清晰——做一个 **modular、extensible、easy-to-use** 的 framework，embedded 在 NVIDIA Isaac Sim / Isaac Lab 生态中。这个选择很聪明，因为 Isaac Sim 提供了 photorealistic rendering、ROS support、GPU-accelerated physics，Isaac Lab 则提供 teleoperation、GPU-parallelized training、各种 RL library 集成。

项目主页：https://sites.google.com/view/tacex

---

## 2. 整体架构解析

TacEx 的 simulation pipeline 可以分解为 **三个正交的子模块**：

### 2.1 Physics Simulation（物理仿真层）
论文提供三种选择：
- **PhysX rigid body + compliant contact**：最快，适合 prototyping
- **PhysX FEM soft body**：内置方案，但发现缺少 static friction 导致无法 grasp
- **GIPC**（GPU-accelerated IPC）：最准确，支持 soft-to-soft contact

### 2.2 Optical Simulation（光学仿真层）
- 使用 **Taxim** 的 GPU 加速实现 [Taxim-GPU](https://git.ias.informatik.tu-darmstadt.de/tactile-sensing/taxim-gpu)
- 基于 height map → surface normal → polynomial RGB mapping

### 2.3 Marker Simulation（标记点仿真层）
- 使用 **FOTS** 的 exponential displacement model
- 不依赖准确的 gelpad deformation，只需要 height map

这种模块化设计的精髓在于：**marker flow 可以在 rigid body gelpad 上也能仿真**，这比 Chen et al. [10] 那种依赖准确 FEM deformation 的方案灵活得多。

---

## 3. Physics Simulation 技术深入

### 3.1 为什么选择 IPC / GIPC？

**IPC（Incremental Potential Contact）** [original paper](https://doi.org/10.1145/3386569.3392425) 是一种 **barrier method**，其核心数学形式是时间离散化的 energy minimization：

$$\min_{x^{t+1}} \quad E_{total}(x^{t+1}) = E_{inertia}(x^{t+1}) + E_{elastic}(x^{t+1}) + E_{barrier}(x^{t+1}) + E_{friction}(x^{t+1})$$

其中各 energy term 定义：

- **Inertia term**（时间积分）：
$$E_{inertia}(x) = \frac{1}{2 \Delta t^2} \| x - x^t - \Delta t v^t \|^2_M = \frac{1}{2} \sum_i m_i \| x_i - \tilde{x}_i^t \|^2$$
  - $x_i$：vertex $i$ 在 $t+1$ 时刻的位置
  - $\tilde{x}_i^t = x_i^t + \Delta t \, v_i^t$：predicted position
  - $m_i$：vertex $i$ 的 mass
  - $\Delta t$：时间步长

- **Elastic energy**（Saint-Venant Kirchhoff 或 Neo-Hookean）：
$$E_{elastic}(x) = \sum_e V_e \Psi(F_e(x))$$
  - $V_e$：tetrahedron $e$ 的 rest volume
  - $F_e = \partial x / \partial X$：deformation gradient
  - $\Psi$：strain energy density

- **Barrier energy**（防止 intersection 的关键）：
$$E_{barrier}(x) = \sum_{(i,j) \in C} b(d_{ij}(x))$$
其中 barrier function：
$$b(d) = \begin{cases} -\kappa (d - \hat{d})^2 \ln(d / \hat{d}), & 0 < d < \hat{d} \\ 0, & d \geq \hat{d} \end{cases}$$
  - $d_{ij}$：point-triangle 或 edge-edge pair 之间的 distance
  - $\hat{d}$：barrier 激活距离（dhat）
  - $\kappa$：barrier stiffness（自动 adaptive）

- **Friction energy**（frictional contact）：
$$E_{friction}(x) = \sum_{(i,j) \in C} \mu \lambda_{ij} \| T_{ij} (x_i - x_j) \|$$
  - $\mu$：friction coefficient
  - $\lambda_{ij}$：normal contact force（Lagrange multiplier）
  - $T_{ij}$：tangent plane projection

**GIPC** [paper](https://doi.org/10.1145/3643028) 是 IPC 的 GPU 加速版本，使用 **Gauss-Newton optimization** 求解上述 minimization 问题，在 GPU 上实现 massive speedup。

### 3.2 IPC 的核心保证

IPC 的两个关键理论保证：
1. **Intersection-free**：barrier energy 在 $d \to 0$ 时趋向无穷，保证不会发生 penetration
2. **Inversion-free**：通过 line search 确保 $det(F_e) > 0$，即 tetrahedron 不会翻转

这两个保证对 RL 的 **domain randomization** 至关重要——可以随意调参数而不用担心 simulation blow up。

### 3.3 GIPC 集成到 Isaac Sim 的工程细节

这是论文最工程化的部分。pipeline 如下：

**Initialization 阶段**：
1. 在 Isaac Sim 中 spawn assets（不带物理属性）
2. 提取 USD mesh 的 triangle mesh data（world position + triangle indices）
3. 用 **Wildmeshing** [link](https://doi.org/10.1145/3386569.3392385) 生成 tetrahedra mesh
4. 用 tet mesh 的 surface vertices/triangles 更新 Isaac Sim USD mesh 的 topology
5. 计算 attachment points（见下）

**Per-step 阶段**：
1. **PhysX step**：更新 rigid body（robot、sensor case）的位置
2. **GIPC step**：
   - 查询 sensor case 的 current pose $T_{case}^t = (p_{case}^t, R_{case}^t)$
   - 更新 attachment points 位置：
   $$p_{attach,i}^{t+1} = p_{case}^t + R_{case}^t \cdot offset_i$$
     - $offset_i$：vertex $i$ 相对 sensor case 的 precomputed offset
   - 调用 GIPC solver 求解所有非 attachment vertices 的新位置
   - 计算 object position = mean of vertex positions（用于 RL observation）
3. 用 **USDRT API** [link](https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usd_fabric_usdrt.html) 快速更新 USD mesh vertices
4. Isaac Sim renderer 渲染

**Attachment Point 计算**：
- 用 **sphere ray casting**（PhysX scene query）检测 tet mesh 顶点是否在 sensor case rigid body 内部
- sphere radius 很小，max distance 很小，命中即认为该 vertex 是 attachment point
- 这种方法很巧妙，不需要显式的几何 boolean 操作

### 3.4 三种 Physics 方案对比实验

**Ball Rolling 实验**（Table 2）：
| num_envs | 1 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| rigid | 3.6930 | 0.2426 | 0.1286 | 0.0673 | 0.0361 | 0.0212 | 0.0143 | 0.0093 |
| soft (PhysX) | 4.7069 | 0.4496 | 0.2718 | 0.1798 | 0.1267 | OOM | - | - |

注意：rigid body 在 1024 envs 时每帧只要 0.0093ms，这是 **GPU parallelization** 的威力。PhysX soft body 在 256 envs 时 OOM。

**GIPC 性能**（Table 3）：
| num_vert | num_tetra | GIPC time |
|---|---|---|
| 1029 | 3717 | 24.95 ms |
| 7900 | 40370 | 110.47 ms |
| 12509 | 66563 | 221.61 ms |

GIPC 时间与 tetrahedra 数量大致线性，目前只能在 **single environment** 下运行（VRAM 限制）。这是 TacEx 的主要瓶颈。

---

## 4. Optical Simulation（Taxim）

### 4.1 Pipeline

Taxim [paper](https://arxiv.org/abs/2109.04027) 的光学仿真分四步：

**Step 1: Height Map 生成**
- 在 Isaac Sim 的 gelpad 内部放置一个 USD camera
- camera 朝向 gelpad 表面，渲染 depth 得到 object surface 的 height map $H(u, v)$
  - $u, v$：image pixel coordinates
  - $H(u, v)$：object 表面在该像素处的高度

**Step 2: Gelpad Deformation 近似**
- 计算 indentation depth：
$$D(u, v) = \max(0, H_{gel} - H(u, v))$$
  - $H_{gel}$：gelpad 原始厚度
- 用 **pyramid Gaussian kernels** 平滑 $D$，得到 deformation field $\delta(u, v)$

**Step 3: Surface Normal 计算**
- 从 deformation field 计算 surface normal：
$$n(u, v) = \frac{(-\partial_u \delta, -\partial_v \delta, 1)}{\sqrt{(\partial_u \delta)^2 + (\partial_v \delta)^2 + 1}}$$
  - $\partial_u, \partial_v$：image plane 上的梯度

**Step 4: Polynomial RGB Mapping**
- 用 precomputed **polynomial lookup table** $f_{r,g,b}$ 映射 normal 到 RGB：
$$I_c(u, v) = f_c(n_x, n_y, n_z), \quad c \in \{r, g, b\}$$
- $f_c$ 是通过 calibration 拟合的多项式（通常是 degree-5 或更高）

**Step 5: Shadow 添加**
- 基于 light source 方向和 surface normal 计算 shadow，增强真实感

### 4.2 为什么 Taxim 比 GAN-based 方法好？

对比 Higuera et al. [30] 的 diffusion model 或 Kim et al. [32] 的 GAN，Taxim 优势：
- **数据效率**：只需要少量真实 tactile images 做 calibration
- **可解释性**：每个步骤都有物理意义
- **可微性**：理论上可以 differentiable（虽然 TacEx 没用这个特性）
- **速度快**：GPU 加速版只需几 ms

---

## 5. Marker Simulation（FOTS）

### 5.1 FOTS 模型

FOTS [paper](https://arxiv.org/abs/2305.03429) 用 **exponential functions** 建模 marker displacement：

对于 normal load（法向压入）：
$$d_{normal}(r) = A_n \exp(-\alpha_n r)$$
- $r$：marker 到 contact center 的距离
- $A_n$：normal load amplitude
- $\alpha_n$：spatial decay rate

对于 shear load（切向滑动）：
$$d_{shear}(r) = A_s \exp(-\alpha_s r) \hat{d}_{shear}$$
- $\hat{d}_{shear}$：shear direction unit vector

对于 twist load（旋转）：
$$d_{twist}(r) = A_t \exp(-\alpha_t r) \cdot (r \times \hat{n})$$
- $(r \times \hat{n})$：tangent direction at radius $r$

总 displacement：
$$d_{total}(r) = d_{normal} + d_{shear} + d_{twist}$$

### 5.2 为什么 FOTS 比 FEM-based marker 仿真好？

Chen et al. [10] 和 DiffTactile [13] 直接从 FEM mesh deformation 计算 marker 位置：
- 需要 marker 到 tetrahedra face 的 mapping
- 如果 marker 在某个 tetrahedron 内，根据该 tet 的 4 个 vertex 位置做 barycentric interpolation

**FOTS 的优势**：
- 不依赖准确的 FEM simulation，可以配合 rigid body 使用
- 计算极快（exponential 函数）
- 容易扩展到不同 sensor 型号
- 只需要 height map 和 contact center 信息

### 5.3 Contact Center 提取

从 height map 提取 contact center：
$$c = \frac{\sum_{(u,v)} D(u, v) \cdot (u, v)}{\sum_{(u,v)} D(u, v)}$$
这是 indentation depth 加权的 centroid，很 robust。

z-rotation（twist angle）直接从 Isaac Sim 的 object pose 获取，不需要从 height map 估计，这是个简化但实用的设计。

---

## 6. 实验分析

### 6.1 Ball Rolling

- 用 **differential inverse kinematics** 控制 end-effector 到 goal positions
- Rigid body 配置：可同时仿真 18 个 robots（VRAM 限制来自 camera simulation）
- Soft body (GIPC) 配置：只能 1 个 robot

这说明 **camera-based tactile simulation 是 VRAM bottleneck**，而不是 physics simulation。

### 6.2 Object Lifting（失败案例）

这是论文诚实的部分：
- PhysX soft body **无法 grasp**：objects 总是 slip away
- 原因：**PhysX soft body simulation 缺少 static friction**
- 即使一个 rigid + 一个 soft gelpad 也不可靠

这个 negative result 很有价值——它说明 Isaac Sim 内置 soft body 还不能用于 contact-rich manipulation。这也是 TacEx 引入 GIPC 的 motivation。

### 6.3 Beam Twisting

- Beam 作为 soft body，attached to a plate
- Robot 用两个 soft gelpad twist + stretch beam 直到 snap back
- 展示 GIPC 在 **extreme deformation** 下的 stability
- 验证 friction simulation 合理

### 6.4 RL Environments

三个 Isaac Lab 环境：
1. **Object Pushing**
2. **Object Lifting**
3. **Pole Balancing**

使用 PPO [Schulman et al.](https://arxiv.org/abs/1707.06347)，observation 用 **marker displacements**。

论文坦白说："we have validated that the training pipeline works, and we are currently working towards obtaining successful policies"——这意味着 **还没有成功训练出利用 tactile feedback 的 policy**，这是当前 limitation。

---

## 7. 性能数据深度解读

### 7.1 Tactile Simulation Speed（Table 1）

| num_envs | height map (ms) | optical sim (ms) | marker sim (ms) |
|---|---|---|---|
| 1 | 1.3718 | 5.9015 | 4.4863 |
| 2 | 0.8508 | 3.8886 | 2.8838 |
| 4 | 0.5988 | 3.0424 | 2.1184 |
| 8 | 0.4323 | 2.5773 | 1.7587 |
| 16 | 2.8827 | 5.7314 | 5.0450 |
| 18 | 3.5149 | 5.931 | 5.52343 |

**关键观察**：
1. 1→8 envs 时，**per-env time 下降**（GPU parallelization 有效）
2. 16 envs 时 **突然变慢**——VRAM 接近上限导致 thrashing
3. FOTS marker sim 在 CPU 上无 parallelization，是 bottleneck

### 7.2 Physics Speed 对比

- PhysX rigid：1024 envs 时 0.0093ms/frame（极快）
- PhysX soft：256 envs OOM
- GIPC：single env，24-221ms（取决于 mesh 分辨率）

**结论**：如果要大规模 RL 训练，只能用 PhysX rigid + compliant contact。GIPC 适合 high-fidelity 单环境仿真或 small-scale 研究。

---

## 8. 与 Concurrent Work 对比

### TacSL [link](https://arxiv.org/abs/2408.06506)
- 也集成到 Isaac Sim
- 用 **simplified soft contact model** 代替 FEM
- TacEx 优势：GIPC 提供真实 soft-to-soft contact

### TACTO [link](https://ieeexplore.ieee.org/document/9680187)
- 用 PyBullet rigid body
- 速度快但准确性差

### DiffTactile [link](https://arxiv.org/abs/2306.03146)
- Differentiable FEM-based
- 适合 gradient-based optimization，但 RL 用不上
- 用 MLP 代替 polynomial lookup table（数据需求大）

### TacIPC [link](https://arxiv.org/abs/2403.06115)
- 也用 IPC，但 CPU 版本
- GIPC 的 GPU 加速是 TacEx 的优势

---

## 9. 局限与未来方向

论文诚实列出 limitation：
1. **只有 qualitative experiments**，缺少 Sim2Real 验证
2. **RL policies 还没成功**利用 tactile feedback
3. **GIPC VRAM 限制**导致只能 single env

未来工作：
- Quantitative Sim2Real experiments
- 更多 RL environments
- Benchmarking platform for tactile simulators

---

## 10. 我的 Intuition 总结

从 engineering 角度看，TacEx 的核心 insight 是 **decouple physics、optical、marker 三个仿真层**，让用户可以根据 task 需求权衡 speed vs accuracy。这种 modularity 比"one-size-fits-all"的方案更适合 research。

从 RL 角度看，论文揭示了一个 **uncomfortable truth**：即使用最先进的 GIPC soft body simulation，也无法在合理 VRAM 内做大规模 RL training。这意味着 tactile RL 目前只能在 **rigid body approximation** 下做大规模训练，再用 soft body 做 fine-tuning 或 evaluation。这种 **two-stage training** 可能是未来的 practical solution。

从 simulation fidelity 角度看，IPC 的 barrier method 提供了 **robustness guarantee**，这对于 domain randomization 至关重要——你可以任意调 material parameters 而不用担心 simulation blow up，这对 sim-to-real 是 huge win。

相关参考链接：
- [Isaac Sim](https://developer.nvidia.com/isaac/sim)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [GIPC paper](https://doi.org/10.1145/3643028)
- [IPC original paper](https://doi.org/10.1145/3386569.3392425)
- [Taxim](https://arxiv.org/abs/2109.04027)
- [FOTS](https://arxiv.org/abs/2305.03429)
- [TacSL](https://arxiv.org/abs/2408.06506)
- [DiffTactile](https://arxiv.org/abs/2306.03146)
- [TacIPC](https://arxiv.org/abs/2403.06115)
- [Wildmeshing](https://doi.org/10.1145/3386569.3392385)
- [PPO](https://arxiv.org/abs/1707.06347)
