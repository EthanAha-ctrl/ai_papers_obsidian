---
source_pdf: SimTac A Physics-Based Simulator for Vision-Based Tactile.pdf
paper_sha256: e53e7d18cfc4563177ef3be326842340081f5c1f2665d026f9161a7f10ce0b4e
processed_at: '2026-08-12T06:43:48-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SimTac 人话版

## 一句话总结

**做一个"触觉传感器的设计器+测试器"——你给它一个奇形怪状的传感器设计图（比如想做一个像大象鼻子的触觉传感器），它就能在电脑里告诉你：按下去皮肤怎么变形、摄像头看到什么图像、接触力多大。**

---

## 为什么需要这东西？

现在的 vision-based tactile sensor（就是那种"摄像头+硅胶皮"的触觉传感器，比如 GelSight、GelTip）几乎都是平的或者半球形的。但大自然里的触觉器官都是奇形怪状的——人手指、猫爪、章鱼触手、大象鼻子。形状跟功能是绑定的。

想做一个仿生形状的触觉传感器，传统做法是捏硅胶、塞摄像头、反复试。一次试错就要几周，而且你不知道设计到底行不行，做出来才发现"哦这个形状光路不对，图像看不清"。

SimTac 想做的就是：**把"做硬件试错"变成"在电脑里试"**。你给它 sensor 的 3D shape、LED 怎么排、camera 怎么放、material 用什么硅胶，它吐出来触觉图像 + 力场。然后你 sim-to-real 直接把 policy 训了，sim 里训的模型 zero-shot 放到 real sensor 上还能用。

---

## 难在哪？三个 bottleneck

### 1. Deformation：曲面怎么算变形？

平的 sensor 你直接当 heightmap 做 smoothing 就行。曲面 sensor 你要做物理模拟，问题是：
- **FEM**（有限元）准但 mesh 密了就慢死，大变形时 mesh 还会扭曲到不能算
- **Heightmap** 简单但忽略 material properties，曲面下更不靠谱

### 2. Rendering：曲面里光怎么传播？

平的 sensor 里，LED 发的光基本是直线打到皮肤表面。曲面 sensor 里，光会沿着弯曲表面传播（想象一下光纤里的全反射），路径不是直线。传统 path tracing 能算但太慢。

### 3. Force：怎么实时给力？

FEM 算 dense force field 准但 real-time 跑不动。MPM 算 deformation 快但 force prediction 不准。

---

## 三个核心 trick 是怎么解决的

### Trick 1: MPM 算 deformation

**Intuition**：把硅胶想象成一大堆小颗粒，每个颗粒有 mass、velocity。再放一个虚拟的 3D grid 在空间里。每个时间步：
- Particles 把自己的 mass 和 momentum "撒" 到附近的 grid nodes 上（P2G）
- Grid nodes 算 velocity、施加 boundary condition（rigid body velocity 或者 fixed = 0）
- Grid 把 velocity "吸" 回 particles（G2P）
- Particles 更新位置

**为什么不用 FEM？** FEM 的 mesh 在大变形下会扭曲、tangle，需要 remeshing。MPM 的 particles 是无 mesh 的，怎么变形都不会坏。而且 grid 是固定的 Eulerian，不需要跟随物质。

**Analogy**：像沙子堆。你按一下沙堆，沙粒们挤来挤去但每颗沙子还是沙子，不会"mesh 坏掉"。grid 只是用来算力传导的临时工具。

### Trick 2: Light field rendering

这是最 clever 的部分。

**问题**：在弯曲的手指形 sensor 里，LED 到皮肤上某个点的光路是沿着曲面走的，不是直线。怎么算这个光路？

**他们的 trick**：
1. 对每个 skin 上的目标点 T，找一个"过 LED 和 T、并且垂直于 T 处 surface normal 的平面"
2. 这个平面跟 sensor mesh 求交，得到一条 curve
3. 这条 curve 就是"光从 LED 出发沿着曲面走到 T"的近似路径

**为什么这样行得通？** 因为这个 plane 同时包含 LED 方向和 surface normal 方向，所以它跟曲面相交得到的 curve 既从 LED 出发、又 tangent 到 surface（在 T 点）。这是一个把 3D surface-tangential propagation 问题降维成 2D plane-mesh intersection 的 trick。

**然后**：offline 算一遍 light field（所有目标点的光路），存起来。Online 渲染时只跑 Phong's model——每个点的亮度就是 diffuse + specular + ambient 三个 term 加一下，GPU 上极快。

**Analogy**：像预先画好一张"光路地图"——"如果我要照亮曲面上的某个点，LED 的光得绕这个路径走"。地图预先画好，渲染时只查地图然后算亮度，不用每帧重新算光路。

### Trick 3: STN 当 surrogate 学 FEM

**问题**：MPM 算 deformation 快，但算 force 不准。FEM 算 force 准但慢。

**解法**：
1. Offline 用 FEM 跑几千次 contact，生成 (deformation, force) pairs 当训练数据
2. 训练一个 Sparse Tensor Network（STN）学 deformation → force 的 mapping
3. Online 时只 forward pass STN，速度跟 MPM 一样快但精度逼近 FEM

**STN 是什么？** 把 particle cloud voxel 成稀疏 4D tensor，用 sparse conv 处理。Sparse conv 只在 non-empty voxel 上算，跳过 empty space，所以 efficiency 跟 active voxel 数成正比，跟 full grid 大小无关。网络结构是 Minkowski UNet，encoder-decoder 对称，skip connection。

**为什么不用 PointNet？** PointNet 在每个 point 上独立 MLP，丢 local structure。STN 的 sparse conv 能捕捉邻域 spatial correlation，对 force prediction 重要（力跟 neighbor particles 的相对位移强相关）。

**Material 改了怎么办？** freeze encoder，只 fine-tune decoder。Intuition：encoder 学的是 geometric features（哪里有 contact、哪里变形大），这是 material-invariant 的；decoder 学的是 deformation → force mapping，这跟 material constitutive 关系强相关。所以换 material 只动 decoder 就行，不需要重训。

**Analogy**：像请了个会算力的老师（FEM）离线出几千道题，让一个学生（STN）刷题学会"看 deformation 猜 force"。学生 inference 快，老师只管出题不用上场。换 material 的话学生只需要复习一下"新材料下答案怎么变"，encoder 学的"看图能力"还能复用。

---

## Pipeline 全貌

```
你给: sensor shape (mesh) + LED 排布 + camera 位置 + material (E, ν)
         ↓
    [MPM 跑 deformation]  → deformed particle cloud
         ↓
    ┌────────────┬─────────────────┐
    ↓            ↓                 ↓
[Occlusion     [Light field      [STN
 removal +      rendering]         surrogate]
 camera proj    → tactile RGB      → dense force
 → depth map]    image              field]
```

三个模块各司其职，通过 representation transformations（particle cloud ↔ depth map、deformation → force via STN）解耦。这种"每个模块用最合适的方法"的 design 比"end-to-end 一个大 MLP"靠谱得多。

---

## Real-time 性能

| 模块 | 规模 | FPS |
|---|---|---|
| MPM | 40K particles | 250 |
| MPM | 300K particles | 33 |
| Rendering | 320×240 | 100 |
| STN | 1K points | 100 |

40K particles 的 MPM 跑 250 FPS，比 real-time 30 FPS 快 8 倍——足够在 sim 里大量 rollout 训 policy。硬件是 RTX 4060，consumer 级别。

---

## 实验里干了啥

### Accuracy
- 6538 张 sim vs real tactile image 对比，SSIM 高、MSE/MAE 低、PSNR 高
- 9980 个 sim vs FEM dense force field 对比，MAE = $2.77 \times 10^{-4}$ mm（deformation）和 $8.6 \times 10^{-6}$ N（force）

### Sim2Real 三个 task
1. **Object classification**（ResNet50）：Sim2Sim 100%，**Sim2Real zero-shot 91.3%**
2. **Slip detection**（VGG-19 + LSTM）：Sim2Sim 97.89%，**Sim2Real 92.06%**
3. **Contact safety**（ResNet50 regression）：Sim2Sim MAE 0.028，**Sim2Real MAE 0.105**

最 impressive 的是 slip detection zero-shot 92%——你在 sim 里训的"判断有没有滑"的 model 直接放到 real sensor 上还 92% 准。

### Flexibility
- 测了 cat paw / octopus tentacle / human thumb / DigiTac 四种 biomorphic shape，都能 sim
- Material 改硬度（soft/medium/hard），fine-tune decoder 就能适配，MAE 全部低于 $4.37 \times 10^{-4}$ mm

### Elephant trunk case study
最 cool 的 demo：在 sim 里设计一个大象鼻子形状的触觉 sensor，tip 有两个 protrusions 能 pinch 抓东西。然后**真的 3D 打印做出来了**，对比 sim 和 real 的 tactile image 一致性高。这是从 design 到 fabrication 的 complete pipeline 演示。

---

## Limitations（paper 自己承认 + 我加的）

- 训 STN 需要 FEM ground truth，全新 shape 要跑几天 FEM
- Phong's model 没建模 inter-reflection、subsurface scattering、多次内部反射
- 不是 differentiable simulator（对比 DiffTactile），不能 gradient-based co-design sensor shape + policy
- Light field 是 undeformed state 预计算的，deformation 改变光路这件事是 approximation
- STN fine-tune strategy 只验证了 material 改变，shape 改变可能 encoder 也要重训

---

## 这个 paper 为什么有意思（Karpathy 视角）

最让我觉得 elegant 的是 **modular expert design**：MPM 解决"任意几何 deformation"、light field 解决"曲面光路"、STN surrogate 解决"力计算快又准"。三个模块各用最合适的方法，然后通过 representation transformation（particle cloud ↔ depth map、deformation → force）解耦。

这跟你 "Software 2.0" 的 framing 很契合：STN surrogate 是典型 Software 2.0（learned mapping from data），MPM 和 light field 是 Software 1.0（explicit physics algorithm）。SimTac 是 hybrid 1.0/2.0 系统——用 1.0 处理 physics-correct 的部分（deformation、light propagation），用 2.0 处理 learned approximation 的部分（FEM → STN）。这种 pattern 在 physics simulation 里越来越常见，例如 differentiable fluid sim、learned contact model 等。

另一个值得注意的点：light field 的 trick 本质是把一个 3D 球面搜索问题（"光在曲面上怎么走"）降维成 2D plane-mesh intersection。这种"找对的几何 parametrization 把问题简化"的 engineering taste，跟 classic graphics 里的 trick 一脉相承（比如 normal mapping、shadow volume）。

---

## 一句话再总结

SimTac = MPM（deformation）+ Light field + Phong（rendering）+ STN（force surrogate），三个模块各司其职，第一次让 physics-based tactile simulation 从 flat 推广到 arbitrary biomorphic geometry，而且 real-time 跑得动，sim-to-real zero-shot 还能用。Engineering taste 满分，limitation 也清楚——下一步应该是 differentiable version 来支持 sensor design optimization。

---

# SimTac: Physics-Based Simulator for Biomorphic Vision-Based Tactile Sensing 深度解析

## 1. Paper 核心动机与定位

这篇 paper 来自 King's College London 的 Shan Luo 课题组 (EP/T033517/2 "ViTac" 项目),第一作者是 Xuyang Zhang。核心要解决的问题是:现有 vision-based tactile sensors (如 GelSight, DIGIT, GelTip) 几乎全部局限于 planar 或 hemispherical 几何,而 biological organisms (human fingers, cat paws, octopus tentacles, elephant trunks) 的 tactile sensing 深度 intertwined with morphological form。设计 biomorphic sensors 靠 trial-and-error 硬件迭代极其 painful,因为复杂曲面下的 deformation modeling、内部 light path 控制、camera/LED 集成都很棘手。

SimTac 的 contribution 是把 simulator 从 flat shape 推广到 arbitrary biomorphic geometry,关键 trick 是把 simulation 拆成三个可解耦的模块,每个模块都用一个能在 GPU 上跑实时的方法。

**Related work 对比直觉**:
- Depth-based methods (FOTS [29], TACTO [30], TacSl [31]): 快但忽略 material properties,deformation 只是 heightmap smoothing。
- MPM-based methods (Tacchi [33], [34-37]): 物理 accurate 但一直局限于 flat sensors。
- FEM-based methods (Taxim [41], DiffTactile [42], TacIPC [40]): 精度高但 mesh density 上去后 computational cost 爆炸,无法 real-time。
- Data-driven rendering (TactGen [46], GAN-based [44]): 需要 real sensor data,generalize 到新 sensor 几何困难。
- Path tracing (Agarwal et al. [47,48]): 物理真实但慢。

SimTac 的选择:MPM 做 deformation (快、能处理任意 geometry) + Light field rendering 做光学 (offline 预计算 light path,online 只跑 Phong) + STN 做 force prediction (用 FEM offline 生成的 ground truth 训练神经网络,online 推理快)。这个组合的 intuition 是:**offline 慢但 accurate 的部分用 FEM 预跑生成训练数据,online 实时部分用快的方法逼近**。

**Reference links**:
- Tacchi (MPM-based): https://ieeexplore.ieee.org/document/10018778
- Taxim (FEM-based): https://ieeexplore.ieee.org/document/9721188
- DiffTactile (Differentiable FEM): https://diff-tactile.github.io/
- TACTO (PyBullet-based): https://ieeexplore.ieee.org/document/9686270
- GelTip (sensor being simulated): https://ieeexplore.ieee.org/document/9348226

---

## 2. System Architecture 总览

SimTac 的 pipeline 拆成三块,可以理解为三个" Expert modules" 串联:

```
Input: sensor shape (mesh) + marker pattern + optical system + material props
        ↓
[Module 1] MPM Particle-based Deformation
        → outputs: deformed particle cloud P
        ↓
   ┌────────────────────────┬───────────────────────┐
   ↓                        ↓                       ↓
[Module 2a] Occlusion      [Module 2b] Light Field   [Module 3] Sparse Tensor
  removal + camera         rendering (Phong)         Networks → force/deformation
  projection + depth       → tactile RGB image       field (FEM-level accuracy)
  map interpolation
```

关键设计 intuition:
- **Discretization uniformity**: 用 Structured Mesh Algorithm [Bern & Plassmann, 2000] 做 mesh partitioning,把 mesh nodes 直接当 particles,这样能保留 particle indices,后续可以指定哪些 particle 是 actuator、哪些是 boundary condition、哪些是 contact surface。这对 biomorphic shape 至关重要,因为你要在任意 region 上施加 active motion。
- **Deformation → Rendering 解耦**: MPM 输出 deformed particle cloud,后处理转成 depth map,再喂给 light field renderer。这意味着 renderer 不需要知道 deformation 的物理细节,只需要一个 depth map 和 surface normals。
- **Force prediction 用 surrogate model**: FEM 在 dense mesh 下太慢,但 FEM 的输出是 dense force field 的 ground truth。SimTac 用 MPM 的 fast deformation 作为 STN 的 input,STN 学习一个 MPM-deformation → FEM-force 的 mapping。这样 inference 时完全不需要跑 FEM。

---

## 3. Module 1: MPM Deformation Simulation 细节

### 3.1 为什么选 MPM 而不是 FEM 或 SPH?

MPM (Material Point Method) [De Vaucorbeil et al. 2020] 是 hybrid Lagrangian-Eulerian method:
- **Lagrangian part**: 物质信息(mass、momentum、stress、deformation gradient)存在 particles 上,particles 跟随物质运动,所以能追踪大变形、free surface、fracture。
- **Eulerian part**: 计算 stress divergence、momentum update 在固定的 background grid上做,避免了 mesh tangling 问题 (FEM 在大变形下 mesh 会扭曲,需要 remeshing)。

对 tactile sensor 的 elastomer 来说,contact 时 local deformation 可以到 1mm scale,在 0.4mm mesh spacing 下就是大变形。FEM mesh 在这种情况下会 jam,而 MPM 的 particles 可以自由流过 grid。

**Reference**: MPM review paper, https://www.sciencedirect.com/science/article/pii/S0065215620300015

### 3.2 MPM 数学详解

每个 particle p 携带:
- $x_p \in \mathbb{R}^3$ — position
- $v_p \in \mathbb{R}^3$ — velocity  
- $C_p \in \mathbb{R}^{3\times 3}$ — affine velocity matrix (capture local rotational/shear 信息,这是 APIC scheme 的关键 [Jiang et al. 2015])
- $F_p \in \mathbb{R}^{3\times 3}$ — deformation gradient, $F_p = \frac{\partial \varphi_p}{\partial x_p}(x_p)$,其中 $\varphi_p: \mathbb{R}^3 \to \mathbb{R}^3$ 是 deformation map
- $m_p, V_p^0$ — mass 和 initial volume

每个时间步四个阶段 (Algorithm 1):

**Stage 1: P2G (Particle-to-Grid)**

用 quadratic B-spline kernel $w_{jp}$ 把 particle 信息 scatter 到周围 3×3×3=27 个 grid nodes 上:

$$M_i = \sum_{j \in \mathbb{G}_i} \sum_{p \in \mathbb{P}_j} w_{jp} m_p$$

- $M_i$ — i-th grid node 的 mass
- $\mathbb{G}_i$ — 27 个 grid nodes 包含 i 和其邻居
- $\mathbb{P}_j$ — j-th grid cell 里的 particles
- $w_{jp}$ — quadratic B-spline weight,衡量 particle p 对 grid node j 的贡献(类似 trilinear interpolation 但更 smooth)

Grid momentum 分两部分:

$$MG_i = MM_i + ME_i$$

- $MM_i$ — 来自 particle motion 的 momentum:
$$MM_i = \sum_{j \in \mathbb{G}_i} \sum_{p \in \mathbb{P}_j} w_{jp} \left( m_p v_p^{(k)} + C_p^{(k)} (X_j - x_p^{(k)}) \right)$$

- $v_p^{(k)}, C_p^{(k)}$ — particle p 在 time step k 的 velocity 和 affine velocity
- $X_j$ — j-th grid node 的位置
- $C_p^{(k)} (X_j - x_p^{(k)})$ 这一项是 affine 修正,使得 grid 看到的不是 particle 中心的常数速度,而是一阶 Taylor 展开的局部 affine 速度场 — 这让 MPM 比原始 FLIP/PIC 更准确,减少了 numerical dissipation。

- $ME_i$ — 来自 elastic stress 的 momentum:
$$ME_i = -\Delta t \sum_{j \in \mathbb{G}_i} \sum_{p \in \mathbb{P}_j} \frac{4}{\Delta X^2} w_{jp} V_p^0 S_p^{(k)} (X_j - x_p^{(k)})$$

- $\Delta t$ — 时间步长
- $\Delta X$ — grid spacing
- $V_p^0$ — particle 的初始体积 (在 reference config 下)
- $S_p^{(k)}$ — p-th particle 的 first Piola-Kirchhoff stress tensor (PK1),由 $F_p$ 通过 constitutive model 算出。论文里 sensor membrane 用弹性体, Young's modulus $E$, Poisson's ratio $\nu$。
- 系数 $\frac{4}{\Delta X^2}$ 来自 B-spline 的二阶导数离散化 (内部 force = $-\nabla \cdot P$ 在 MPM 里转成 grid momentum)。

**Stage 2: Grid Operation**

$$V_i = \frac{MG_i}{M_i}$$

简单的 momentum / mass = velocity。Grid position 不更新 (Eulerian)。然后施加 boundary conditions:rigid body 的 grid nodes velocity 设为 rigid body velocity,fixed boundary 的 grid nodes velocity 设为 0。

**Stage 3: G2P (Grid-to-Particle)**

把 grid 上的 velocity 和 affine 信息 gather 回 particles:

$$v_p^{(k+1)} = \sum_{i \in \mathbb{G}'_p} w_{ip} V_i^{(k)}$$

- $\mathbb{G}'_p$ — particle p 周围的 27 个 grid nodes
- $w_{ip}$ — 同样的 B-spline weight

$$C_p^{(k+1)} = \frac{\Delta X^2}{\Delta t} \sum_{i \in \mathbb{G}'_p} w_{ip} v_p^{(k+1)} \frac{X_i - x_p^{(k)}}{\Delta X}$$

- 这是从 grid velocity field 的空间梯度估计 affine velocity $C_p$。直觉上,如果 grid velocity 在 particle 周围变化,$C_p$ 就会捕获这种 gradient (类似 least-squares 拟合一个 affine field)。

$$F_p^{(k+1)} = (I + \Delta t \cdot C_p^{(k+1)}) F_p^{(k)}$$

- Deformation gradient 的更新公式。直觉:$\nabla v$ (velocity gradient) 乘上 $\Delta t$ 给出 infinitesimal strain increment,$I + \Delta t C$ 是 exponential map 的一阶近似 (实际 APIC MPM 用这个简化版)。

**Stage 4: Particle Operation**

施加 particle-level boundary conditions (Eq. 18):

$$v_p^{(k+1)} = \begin{cases} v & \text{if } p \in \mathbb{I} \text{ (rigid body)} \\ 0 & \text{if } p \in \mathbb{B} \text{ (fixed boundary)} \end{cases}$$

然后位置更新:

$$x_p^{(k+1)} = x_p^{(k)} + \Delta t \cdot v_p^{(k+1)}$$

**Terminal condition**: 当 indenter 或 sensor 达到目标 pose (例如 indentation depth = 1mm) 时停止迭代。

### 3.3 Post-processing: 从 particles 到 depth map

MPM 输出的 deformed particle cloud 不能直接喂给 renderer,需要:

1. **Occluded particle removal**: 用 ray casting [Decherchi & Rocchia 2013] 检查从 camera position $C_c$ 出发到每个 surface particle $P_i$ 的射线 $\vec{L_i}$ 是否被其他 triangle mesh $M_j$ 挡住。如果被挡住就删掉这个 particle。这对 biomorphic shape 至关重要,因为 finger-shape sensor 的 curved surface 会有 self-occlusion。

2. **Camera projection** (Eq. 1):

$$u_i = \frac{x_i}{z_i} f_u + c_x, \quad v_i = \frac{y_i}{z_i} f_v + c_y, \quad d_i = \sqrt{(x_i - x_c)^2 + (y_i - y_c)^2 + (z_i - z_c)^2}$$

- $(u_i, v_i)$ — particle 在 depth map $D$ 上的 pixel coordinates
- $d_i$ — particle 到 camera 的 Euclidean 距离 (depth value)
- $(x_i, y_i, z_i)$ — particle 在 camera frame 的 3D coordinates
- $(c_x, c_y)$ — image principal point (image center)
- $f_u, f_v$ — focal length in pixel units (Eq. 2):
$$f_u = \frac{D_{width}}{2 \tan(fov/2)}, \quad f_v = \frac{D_{height}}{2 \tan(fov/2)}$$
- $D_{width}, D_{height}$ — depth map 分辨率 (例如 320×240)
- $fov$ — camera 的 angular field of view

3. **2D cubic spline interpolation**: 把离散的 depth map 插值成连续 smooth depth map,因为 MPM particle 在投影后可能稀疏。

**Intuition**: MPM 的输出本质上是一个 deformed 3D point cloud,但 renderer 需要一个 2D depth map 和 normal map。Post-processing 就是做这个 representation transformation,同时处理 visibility。

---

## 4. Module 2: Light Field-Based Optical Rendering

这是这篇 paper 最 innovative 的部分,专门为 biomorphic shape 设计。

### 4.1 问题:Flat sensor vs. biomorphic sensor 的光学差异

在 flat GelSight 上,光从 LED 到 membrane 表面点的路径几乎是直线 (因为 membrane 是 flat slab)。所以 Phong's model 直接用 $\hat{L} = L_s - T$ 作为 incident direction 就行。

但在 finger-shaped GelTip 或 elephant trunk sensor 上,光在 membrane 内部会发生:
1. **Direct line propagation**: LED 发出的光穿过 transparent elastomer 直接打到 membrane 表面点 — 这部分仍然是直线。
2. **Surface-guided propagation**: 光进入 membrane 后沿着 curved surface 的几何路径传播 (类似 optical fiber 中的 total internal reflection,或至少是 surface-tangential propagation)。这部分是非线性的,路径跟随 membrane 几何。

Paper 把这两个分别建模为 $\hat{L}_{linear}$ 和 $\hat{L}_{non-linear}$。

### 4.2 Offline Light Field Generation

**Linear light field** $\hat{L}_{linear}$: 对每个 target point $T(x,y,z)$ 在 membrane 上,计算 $\vec{L} = L_s - T$ (LED 到该点的直线方向)。这给出了所有 target points 的 incident direction field。

Target point cloud $\hat{P}_t$ 由 depth map $D$ 通过 inverse camera projection (Eq. 3) 得到:
$$x = (u - c_x) \frac{z}{f_u}, \quad y = (v - c_y) \frac{z}{f_v}, \quad z = D(u, v)$$

**Non-linear light field** $\hat{L}_{non-linear}$: 这是关键创新。算法是:

1. 对每个 target point $T$ 在 curved surface $z = f(x,y)$ 上,计算 surface normal:
$$\vec{n} = \left(-\frac{\partial z}{\partial x}, -\frac{\partial z}{\partial y}, 1\right)$$

2. 定义 propagation plane $P_l$,这个 plane 包含 light source $L_s$ 和 target point $T$。Plane normal 是:
$$\vec{P}_l = \vec{n} \times \vec{L}$$
- 这是 $\vec{n}$ (surface normal at T) 和 $\vec{L}$ (vector from $L_s$ to $T$) 的 cross product,得到的 plane 同时垂直于 surface normal 和 LED-to-point vector。
- Intuition: 这个 plane 是"光从 LED 出发,沿着 surface 传播到 T 点"的最自然 plane,因为光在 surface 上的传播必须 tangent 到 surface (在 propagation direction 上),而这个 plane 包含 surface normal 和 LED 方向,所以光在这个 plane 内既能 tangent 到 surface 又能从 LED 到 T。

3. 在这个 plane $P_l$ 上,计算 plane 与 membrane mesh $M$ 的 intersection curve。具体做法是检查 mesh $M$ 的每个 triangle face 是否与 $P_l$ 相交,如果有就计算交点,连接起来得到一条连续 curve。这条 curve 就是光在 membrane 表面从 LED 附近到 T 的 propagation path。

4. 在 T 点沿着 curve 的 tangent direction 作为 incident light direction $\hat{L}$。

**Intuition**: 这其实是把 surface-tangential light propagation 简化成了一个 2D 几何问题 — 找出"过 LED 和 T 的、垂直于 surface normal at T 的那个 plane"与 surface 的 intersection curve。这个 trick 把 3D 球面搜索简化成了 2D plane-mesh intersection,可 offline 预计算。

### 4.3 Online Image Rendering: Phong's Model

得到 deformed depth map $D$ 和 offline light field $\hat{L}$ 后,用 Phong's reflection model (Eq. 4) 渲染每个点 $T$:

$$I = k_a i_a + \sum_{m \in \hat{L_s}} \big( k_d (\hat{L}_m \cdot \hat{N}) i_{m,d} + k_s (\hat{R}_m \cdot \hat{V})^\alpha i_{m,s} \big)$$

- $I$ — total illumination intensity at point T
- $k_a$ — ambient reflection coefficient
- $i_a$ — ambient light intensity,这里取 linear light field 渲染的 background image (或 real sensor 的 undeformed image)
- $\hat{L_s}$ — set of light sources (LED ring 通常 4 个 LED)
- $m$ — 索引某个 light source
- $\hat{L}_m$ — incident light direction at T from source m,从 non-linear light field 取
- $\hat{N}$ — normalized surface normal at T,通过 Eq. 6 计算:
$$\hat{N} = \frac{\frac{\partial p}{\partial x} \times \frac{\partial p}{\partial y}}{\left\|\frac{\partial p}{\partial x} \times \frac{\partial p}{\partial y}\right\|}$$
- 偏导通过 Sobel edge detector 在 point cloud $P$ 上估计 (因为 MPM 输出的是离散 particles,不是解析 surface)
- $k_d$ — diffuse reflection coefficient
- $i_{m,d}$ — diffuse light intensity from source m
- $k_s$ — specular reflection coefficient  
- $\hat{R}_m$ — reflected light direction (Eq. 5):
$$\hat{R}_m = 2(\hat{L}_m \cdot \hat{N})\hat{N} - \hat{L}_m$$
- $\hat{V}$ — view direction (从 T 指向 camera)
- $\alpha$ — shininess exponent (specular highlight 锐度)
- $i_{m,s}$ — specular light intensity from source m

每个 RGB channel 独立计算,最后 combine 成 $(I_R, I_G, I_B)$。

**Final composition**:
- Background (undeformed region) 用 linear light field 渲染 (或直接用 real sensor 的 undeformed image 作为 ambient term)
- Foreground (deformed contact region) 用 non-linear light field 渲染
- Overlay 得到 final tactile image

**Intuition**: linear light field 负责 "background illumination"(均匀的整体光照),non-linear light field 负责 "contact deformation 产生的 highlight/shadow pattern"。这个分工让渲染速度很可控:background 可以缓存 ( undeformed state 不变),只有 foreground 需要每帧更新。

### 4.4 为什么这比 path tracing 快?

Path tracing 需要对每个 pixel 跑 Monte Carlo ray tracing,bounce 多次。SimTac 的 trick 是:
1. Light field (light propagation path) 是 offline 预计算的,对每个 sensor 几何只需算一次。
2. Online 时只是 Phong's model 的 closed-form 计算 (dot products + power),GPU 上每个 pixel 并行,极快。

代价是:
- 假设光只在 surface-guided propagation 或 direct line 两种 path 中传播,忽略了多次 internal reflection。
- 假设 light source 是 point light (虽然可以 discretize line/area light)。
- Surface 间互反射 (inter-reflection) 没建模。

但对 tactile sensor 这种 thin elastomer + reflective coating 的结构,这些近似足够好。

---

## 5. Module 3: Sparse Tensor Network for Force Prediction

### 5.1 为什么需要这个 module?

MPM 给出 deformation field (particle displacements $(\Delta x_i, \Delta y_i, \Delta z_i)$),但 tactile perception tasks 还需要 force field。直接从 MPM 算 force 不准确 (MPM 在 contact boundary 的 stress 估计 noise 大),而 FEM 准但慢。

**Solution**: 用 FEM offline 生成 (deformation, force) pairs 作 ground truth,训练一个 neural network 学习 MPM-deformation → FEM-force 的 mapping。Online 时只跑 forward pass,快且准。

### 5.2 为什么用 Sparse Tensor Networks 而不是 PointNet/PointNet++?

- PointNet 在每个 point 上独立 MLP + max pooling,丢失了 local spatial structure。
- PointNet++ 用 ball query 分层,但 tactile sensor 的 particle density 不均匀 (actuator region 密,普通 region 稀),ball query 的尺度难选。
- Sparse Tensor Networks (STN) [Choy et al. 2019, Minkowski Engine] 把 point cloud voxel 成 sparse 4D tensor (3D spatial + 1D feature channel),用 sparse convolution 处理。Sparse convolution 只在 non-empty voxels 上做卷积,跳过 empty space,所以 efficiency 跟 active voxel 数量成正比,不跟 full grid size 成正比。

**Reference**: Minkowski Engine, https://github.com/StanfordVL/MinkowskiEngine

### 5.3 Sparse Tensor 公式与网络结构

**Voxelization** (Eq. 7):

$$C_v = \begin{bmatrix} b^1 & c_x^1 & c_y^1 & c_z^1 \\ \vdots & \vdots & \vdots & \vdots \\ b^N & c_x^N & c_y^N & c_z^N \end{bmatrix}, \quad F_{in} = \begin{bmatrix} f_x^1 & f_y^1 & f_z^1 \\ \vdots & \vdots & \vdots \\ f_x^N & f_y^N & f_z^N \end{bmatrix}$$

- $C_v$ — voxel coordinate matrix, $N \times 4$
- $b^i$ — batch index (i-th particle 属于哪个 batch sample,用于 batched training)
- $\{c_x^i, c_y^i, c_z^i\} \in \mathbb{Z}^3$ — integer voxel coordinates (floor of particle position / voxel size $D_v$)
- $N$ — 总 voxel 数 (取决于 voxel size $D_v$)
- $F_{in}$ — input feature matrix, $N \times 3$,每个 voxel 的 feature 是该 voxel 内所有 particles 的 displacement 平均值 $\{f_x^i, f_y^i, f_z^i\} \in \mathbb{R}^3$

**Sparse convolution** (Eq. 8):

$$\mathbf{f}_{\mathbf{c}}^{out} = \sum_{s \in \mathcal{N}(c, K)} W_s \mathbf{f}_{c+s}^{in}, \quad \mathbf{f}_{\mathbf{c}}^{out} \in F^{out}, \quad \mathbf{c} \in C_v^{out}$$

- $\mathbf{c}$ — output voxel 的 integer coordinate
- $\mathcal{N}(c, K)$ — $\mathbf{c}$ 的 neighborhood,由 kernel size $K$ 定义 (例如 3×3×3 kernel 给 27 neighbors)
- $s$ — offset 用于定位 input voxel (相对 $\mathbf{c}$ 的 offset)
- $\mathbf{f}_{c+s}^{in}$ — input voxel $(c+s)$ 的 feature vector
- $W_s$ — learned weight for offset $s$ (类似 regular conv 的 kernel weight)
- **关键**: input 和 output voxel coordinates 相同 ($C_v^{in} = C_v^{out}$),所以这是 strided-1 conv (类似 ResNet 里的 basic block)。Downsampling/up-sampling 通过 strided conv 和 transpose sparse conv 完成。

**Minkowski UNet14 架构**:
- Encoder-decoder 对称结构 (类似 U-Net)
- 14 layers (大概是 4 downsampling stages + bottleneck + 4 upsampling stages)
- Encoder: 逐层 voxel 数减半 (sparse strided conv)、channel 数加倍
- Decoder: 逐层 voxel 数加倍 (sparse transpose conv)、channel 数减半
- Skip connections 连接 encoder 和 decoder 的对应层
- 输出: $N \times 3$ feature vector (X, Y, Z 方向的 force 或 deformation)
- Inverse voxelization: 把 voxel feature 还原到 particle feature

**Training details**:
- Loss: L1 (MAE)
- Optimizer: Adam, lr=1e-3, lr decay 10× every 10 epochs
- Batch size: 32
- 100 epochs
- Dataset split: 3:1:1 (train:val:test)

### 5.4 Material property fine-tuning strategy

对新的 material stiffness (Young's modulus 改变),不需要从头训练,只需:
1. 用新 material 跑少量 FEM 生成 ground truth (limited set)
2. **冻结 encoder** (encoder 学的是 deformation 的一般 spatial features,跟 material 无关)
3. **只更新 decoder** (decoder 学的是 deformation → force 的 mapping,跟 material constitutive model 强相关)
4. 在新 material 上 evaluate

这个 trick 的 intuition:encoder 提取的是 geometric feature (哪里有 contact、哪里 deformation 大),这是 material-invariant 的;decoder 把 feature map 到 force,这跟 material 的 stress-strain 关系直接相关。

---

## 6. Experiments & Results 详解

### 6.1 Optical Response Accuracy (Fig. 5)

**Setup**: 6,538 data pairs from real GelTip 和 SimTac,14 个不同 indenter shapes,不同 contact positions 和 orientations。

**Metrics**:
- SSIM (Structural Similarity Index) — 越高越好,衡量 structural pattern 相似度
- MSE (Mean Squared Error) — 越低越好,pixel-level difference
- MAE (Mean Absolute Error) — 越低越好
- PSNR (Peak Signal-to-Noise Ratio) — 越高越好,衡量 signal-to-noise

**Results (Fig. 5b)**: 分布显示 SSIM 高、MSE/MAE 低、PSNR 高,quantitatively 验证 simulator 准确。

**Failure modes** (Supplementary Fig. 13):
- Contact depth 越大,similarity 下降 — 因为 deformation region 扩大,non-linear light field 近似误差累积。
- Contact position 从 tip 到 base,similarity 先升后降 — tip region 超出 camera focal range 导致 blur,base region 太靠近 camera 导致 lens distortion。
- Contact angle 变化引起 RGB light distribution variation。

### 6.2 Mechanical Response Accuracy (Fig. 6)

**Setup**: 9,980 data pairs,10 seen objects for training,4 unseen objects for testing。

**Dense field comparison (vs FEM ground truth)**:
- Deformation MAE on test set: $2.77 \times 10^{-4}$ mm
- Force MAE on test set: $8.6 \times 10^{-6}$ N
- 整个 dataset MAE: $2.84 \times 10^{-4}$ mm (deformation), $7.4 \times 10^{-6}$ N (force)

**Sparse field comparison (vs real sensor)**:
- Marker motion 通过 downsample dense deformation field 到 marker sparsity,project 到 camera frame
- Total force MAE:
  - X (shear): 0.021 N (13.18% of actual force)
  - Y (shear): 0.013 N (9.24%)
  - Z (normal): 0.134 N (6.27%)
- Normal force error 绝对值大但 percentage 小 (因为 normal force 量级大)

### 6.3 Efficiency (Section 3.2)

| Module | Particle/Image Count | FPS |
|---|---|---|
| MPM deformation | 40K particles | 250 FPS |
| MPM deformation | 300K particles | 33 FPS |
| MPM deformation | 1.3M particles | 10 FPS |
| Optical rendering | 320×240 | 100 FPS |
| Optical rendering | 640×480 | 25 FPS |
| Optical rendering | 1280×960 | 10 FPS |
| Force prediction (STN) | 1K points | 100 FPS |
| Force prediction (STN) | 5K points | 76 FPS |
| Force prediction (STN) | 25K points | 62 FPS |

**Hardware**: i7-13700HX 16-core CPU + NVIDIA RTX 4060 GPU。

**Intuition**: 40K particles 的 MPM 能跑到 250 FPS,比 real-time (30 FPS) 快 8 倍,足够 sim-to-real training 时大量 rollouts。Optical rendering 在 320×240 能跑 100 FPS。STN 在 1K points 时 100 FPS,跟 deformation 模块的 throughput 匹配。

### 6.4 Flexibility (Fig. 7, 8)

**Sensor shape flexibility**: 测试了 cat paw、octopus tentacle、human thumb、DigiTac (marker-based)四种 biomorphic shapes。每种都能 simulate contact deformation + optical response + mechanical response。DigiTac 还能 simulate physical pin motion (marker-based sensor)。

**Material flexibility (Fig. 8)**:
- Three stiffness levels: Soft ($E = 0.0725$ MPa), Medium ($E = 0.145$ MPa), Hard ($E = 0.29$ MPa)
- Fine-tuning strategy: 冻结 encoder,只 fine-tune decoder,用 small FEM dataset
- MAE across materials:
  - Deformation: < $4.37 \times 10^{-4}$ mm
  - Force: < $1.16 \times 10^{-6}$ N
  - Total force: < 0.042 N

### 6.5 Sim2Real Tasks (Fig. 11)

**Task 1: Object Classification**
- ResNet50 backbone (pretrained on ImageNet)
- 8,016 image pairs,3:1:1 split
- 30 epochs,Adam,lr=0.001,batch=32
- **Sim2Sim accuracy**: 100%
- **Sim2Real zero-shot accuracy**: 91.3%
- Failure: similar-shaped indenters (especially flat indenter 与 curved membrane insufficient contact 时,wide-angle lens distortion 让 features 相似)

**Task 2: Slip Detection**
- VGG-19 (ImageNet pretrained) + LSTM
- 8 个 image sequence 作为 input (temporal)
- 832 sequences,1:1 slip/non-slip balanced
- Slip 定义: FEM-recorded tangential force 从 increasing 转 decreasing 的 turning point 位移作 threshold
- **Sim2Sim accuracy**: 97.89%
- **Sim2Real accuracy**: 92.06%
- Key insight: non-slip 时 object contour 和 markers 同步移动;slip 时 object 继续动但 markers 几乎不动 (relative motion)

**Task 3: Contact Safety Assessment**
- ResNet50 regression model
- Predicts safety coefficient ∈ [0, 1]: 1 = safe,0 = high-risk
- Safety coefficient = normalized peak distributed force from predicted force field
- 2,548 SimTac images + 559 real images
- 50 epochs,Adam,lr=0.001,batch=32,MAE loss
- **Sim2Sim MAE**: 0.028
- **Sim2Real MAE**: 0.105
- Failure reasons (Supplementary Fig. 19):
  1. 3D-printed real objects surface textures 缺失 in simulation
  2. Indentation 深时 silicone deformation 改变 light distribution,real image 整体变暗
  3. 部分 real data indentation 不足,tactile image 只 capture partial contact
  4. Some contact poses 让 object 超出 camera FOV

### 6.6 Elephant Trunk Sensor Case Study (Fig. 9, 10)

这是 paper 最 cool 的 demonstration:从 simulation 到 real fabrication 的 complete pipeline。

**Design (Fig. 9b)**:
- Silicone membrane in elephant trunk shape,tip 有两个 finger-like protrusions
- 120° wide-angle camera at base,1.5mm focal length
- Red-white-blue-green LED ring (1:1:1:1 brightness)
- Actuator surfaces on protrusions (particle velocity controlled for open/close motion)
- Outer surface = reflective layer for optical rendering
- Inner surface = support layer (velocity = 0,boundary condition)
- Membrane mesh: C3D4 tetrahedral elements, $h = 0.4$ mm particle spacing
- Material: $E = 1.45 \times 10^5$ kPa, $\nu = 0.45$ (silicone XP-565,1:15 mix ratio)
- 256 grid nodes,total grid length 70mm
- $10^6$ particles for contact object (rigid)

**Real fabrication (Fig. 9d, e)**:
- 4 modules: actuator, optical, skin, lighting
- Dimensions: 34 × 26 × 50 mm
- Camera: 5MP OV5640 USB, 1920×1080 @ 30Hz
- Lens: 120° distortion-free fixed-focus, 1.5mm focal length
- Support layer: transparent resin, Formlabs Form-3 3D printer, polished
- Elastomer: cast in 3D-printed mold, polished
- Reflective coating: aluminum powder + reflective pigment membrane
- LED ring: WS2812 control chip, 1615 SMD LEDs, programmable RGB

**Validation (Fig. 10)**:
- Sensor 先 vertical contact 感知 object shape
- 然后两个 protrusions 闭合执行 grasp action
- Tactile feedback 整个过程都被 simulate 和 real fabricate,visual comparison 显示 high consistency

---

## 7. Limitations & Discussion

Paper 自己承认的 limitations:
1. **FEM ground truth dependency**: 训练 STN 需要离线 FEM 生成 ground truth,对全新 sensor shape (不是 material change) 要跑几天 FEM (即使 GPU 加速)。
2. **Phong's model 限制**: 没建模 inter-reflection、subsurface scattering、polarization、多次 internal reflection。
3. **MPM 的 contact handling**: Particle-based contact 在 sharp edges 上可能有 self-intersection (没明说,但 MPM 通用问题)。

**My additional critique**:
- **No differentiability**: SimTac 不是 differentiable simulator (对比 DiffTactile [42]),无法直接 gradient-based optimization for sensor design。如果要做 co-design of sensor morphology + control policy,需要 differentiable version。
- **STN 的 generalization 到 new shape**: Fine-tuning strategy 只验证了 material 改变 (encoder frozen,decoder fine-tuned)。如果是 shape 改变,encoder 学到的 geometric features 可能也不 valid,需要重新训练。这跟 limitation 1 是同一问题。
- **No dynamic contact/sliding**: MPM 在持续 sliding、rolling、twisting contact 下的 stability 没充分验证。Paper 里 slip detection task 用 8-frame sequence,但 MPM 时间步长 vs real-time consistency 没讨论。
- **Light field precomputation 限制**: Light field 是 offline 预计算,假设 sensor membrane 在 undeformed state。但实际 deformation 也会改变内部光路 (membrane 变薄/变厚)。Paper 用 non-linear light field 在 deformed depth map 上重新算 incident direction,但 light path 本身 (LED 到 T 的 surface curve) 还是 undeformed 几何算的。这是 approximation。

---

## 8. 与 SOTA 的 Positioning

| Simulator | Deformation | Rendering | Force | Geometry | Differentiable | Speed |
|---|---|---|---|---|---|---|
| TACTO [30] | PyBullet heightmap | PyRender | Penalty | Flat only | No | Fast |
| Taxim [41] | FEM | Data-driven (real image lookup) | FEM | Flat | No | Medium |
| Tacchi [33] | MPM | Phong | MPM stress | Flat | No | Fast |
| DiffTactile [42] | FEM | Diff path tracing | FEM | Flat | Yes | Slow |
| TactGen [46] | GAN | Generative | - | Flat | No | Medium |
| FOTS [29] | Heightmap | Phong | Penalty | Flat | No | Very fast |
| **SimTac** | **MPM** | **Light field + Phong** | **STN (FEM-trained)** | **Biomorphic** | **No** | **Fast** |

SimTac 是第一个真正支持 arbitrary biomorphic geometry 的 physics-based tactile simulator。唯一接近的是 Gomes et al. 2023 [28] "Beyond flat GelSight sensors",但那个的 rendering 还是 data-driven,需要 real sensor data。

---

## 9. 可能的 Future Directions (hallucination-friendly)

基于这篇 paper 的 intuition,我想到几个方向:

1. **Differentiable SimTac**: 把 MPM 改成 differentiable MPM (DiffMPM 已经存在,见 [Hu et al. 2020, DiffTaichi]),把 light field rendering 也做成 differentiable (Phong's model 本身 differentiable,light field 几何计算可能需要 relax 成 soft assignment),STN 已经 differentiable。这样可以 gradient-based co-design sensor shape + perception policy。

2. **Inverse design of biomorphic sensors**: 给定 task (例如 grasp delicate tissue),用 differentiable SimTac + gradient optimization 反推 optimal sensor morphology (finger shape、tip curvature、membrane thickness distribution)。

3. **Multi-modal sensor fusion**: SimTac 输出 tactile image + dense force field,可以跟 visual RGB-D、proprioception 融合,做 ViTac synergy [Shan Luo's ViTac project]。Paper 里提到 ViTac project,但没具体做 multi-modal fusion experiment。

4. **Real-time closed-loop control with SimTac**: 在 sim 里 train RL policy,tactile state = SimTac output,直接 sim-to-real deploy 到 real robot。Paper 只验证了 perception tasks (classification、slip、safety),没做 closed-loop control。

5. **Differentiable light field via neural representation**: 用 NeRF/MipNeRF 学 sensor 内部 light field,替代 geometric light field computation。NeRF 可以学到多次 reflection、subsurface scattering。这能放松 Phong's model 限制。

6. **MPM + Differentiable FEM hybrid**: 用 MPM 做 forward simulation (快),用 differentiable FEM 做 force gradient (准)。STN 学习 inverse mapping 而不只是 forward。

7. **Soft robot simulation**: 把 SimTac 的 MPM deformation 扩展到 entire soft robot body,不只是 sensor skin。Elephant trunk sensor 本身就是 soft robot,如果整个 gripper 都是 soft body + tactile skin,可以做 fully soft robotic simulation。

8. **GPU acceleration of light field generation**: 当前 light field 是 offline 预计算。如果 sensor shape 在 simulation 中变化 (例如 active morphing sensor),需要 online recompute light field。用 CUDA 加速 plane-mesh intersection 可以做到。

9. **Uncertainty quantification**: STN 给 point estimate,加 Bayesian STN (例如 Deep Ensemble 或 MC Dropout) 给 uncertainty,这对 contact safety assessment 重要 (high uncertainty 时 conservative behavior)。

10. **SimTac for medical robotics**: Biomimetic tactile sensors + SimTac 可以用于 minimally invasive surgery sensor design,例如 finger-shaped tactile sensor for palpation。

---

## 10. Bottom Line Intuition

SimTac 的核心 insight:**biomorphic tactile sensor simulation 的 bottleneck 不在单一模块,而在三个模块的协同**。MPM 解决 "任意几何下 deformation" 的问题,light field rendering 解决 "complex geometry 下光路" 的问题,STN surrogate 解决 "FEM 准但慢" 的问题。三个模块都用各自最适合的方法,然后通过 representation transformations (particle cloud ↔ depth map、deformation field → force field through STN) 解耦。这种"modular expert"设计哲学是工程化的好范例,跟 end-to-end differentiable 但 each module 都用差强人意的方法 (例如 all-MLP) 形成对比。

对 Andrej Karpathy 来说,这个 paper 让我想到 e.g. 你的 "Software 2.0" essay 里的观点:STN surrogate model 是典型 Software 2.0 (learned mapping),而 MPM 和 light field rendering 是 Software 1.0 (explicit algorithm)。SimTac 把它们组合起来,用 Software 1.0 解决 physics-correct 部分,用 Software 2.0 解决 learned approximation 部分 (FEM → STN),这是 hybrid 1.0/2.0 的 pattern,在 physics simulation 里越来越常见。

**Reference links for further reading**:
- Shan Luo's lab page: https://www.kcl.ac.uk/people/shan-luo
- ViTac project: https://www.kcl.ac.uk/research/vitac-visual-tactile-synergy-for-handling-flexible-materials
- GelTip sensor: https://ieeexplore.ieee.org/document/9348226
- Minkowski Engine (STN): https://github.com/StanfordVL/MinkowskiEngine
- Material Point Method review: https://www.sciencedirect.com/science/article/pii/S0065215620300015
- DiffTactile (differentiable FEM sim): https://diff-tactile.github.io/
- Tacchi (MPM-based sim): https://ieeexplore.ieee.org/document/10018778
- TACTO simulator: https://github.com/facebookresearch/tacto
- Taxim: https://github.com/CMURoboTouch/Taxim
- YCB dataset (used in experiments): https://www.ycbbenchmarks.com/
- Phong's model original: https://dl.acm.org/doi/10.1145/358669.358688
- Differentiable MPM (DiffTaichi): https://github.com/taichi-dev/difftaichi
- Open3D (used for sampling): http://www.open3d.org/
- Biomimetic soft robots review: https://www.nature.com/articles/s41598-022-22192-6
