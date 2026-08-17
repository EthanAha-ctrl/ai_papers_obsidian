---
source_pdf: SplatSim Zero-Shot Sim2Real Transfer of RGB Manipulation Policies.pdf
paper_sha256: 0c8741dcb67b1cdb5afa3ee85bb2dd185d4b8de583519d96e1b501c42242d247
processed_at: '2026-08-12T10:09:09-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SplatSim 用人话说

## 一句话版本

在 simulator 里用 Gaussian Splatting 代替传统 mesh 渲染，让 sim 里的图看起来跟 real 几乎一样，policy 在 sim 里学好之后直接扔到 real robot 上就能用，不需要 fine-tune。

---

## 为什么要搞这个

robot manipulation 现在最大的痛点就是 **data 太贵了**。你想要 policy 学会"把苹果放到盘子里"，得人在旁边 teleoperation 收几百条 demonstration，一个任务几个小时。四个任务就是 20.5 小时人时，谁能扛得住。

那 sim 不是免费的吗？在 sim 里你想要多少 data 就有多少，motion planner 自动跑，3 小时搞定。问题是 **sim 里的图跟 real 的图差太远了**。

PyBullet 这类 simulator 默认的 renderer 是 mesh-based，渲染出来的图就是那种塑料感、没阴影、没 texture 细节、光照假的图。你在这种图上训 policy，policy 学到的是"这种塑料世界里的特征"，一放到 real 里，光照、颜色、反射全变了，policy 直接懵逼。

这就是 RGB Sim2Real gap 的本质：**分布不一样**。深度、点云、触觉这种 modality 在 sim 和 real 里数值几乎一致，所以 gap 小。但 RGB 承载的视觉信息太丰富了，sim 渲染跟不上。

---

## SplatSim 的思路

聪明的地方在于：**与其费力去让 mesh renderer 变得更真实，不如直接把真实世界"拍"下来重建一个 photorealistic 的世界，然后让 robot 在这个世界里活动**。

具体怎么做？

### 第一步：扫一遍场景

你拿个 RGB camera 把工作台、robot、object 都拍一圈视频，用 Gaussian Splatting 重建出一个 3D 场景 $S_{real}$。这个场景是 photorealistic 的，因为它就是从真实图像重建的，texture、color、reflectance 全都在里面。

Gaussian Splatting 的好处是它不是 implicit function（像 NeRF），而是一堆 explicit 的 3D Gaussian primitive，每个有 mean $\mu$ 和 covariance $\Sigma$。你可以直接操作每个 Gaussian，这是关键。

### 第二步：把 robot 和 object 分割出来

你需要知道哪些 Gaussian 属于 robot base、哪些属于 robot link 1、link 2、哪些属于 object。用 robot 的 CAD model 给的 bounding box 来切，articulated gripper 这种 bounding box 不好使的就用 KNN classifier 来分类。

### 第三步：对齐坐标系

simulator 有自己的坐标系 $\mathcal{F}_{sim}$，Gaussian Splat 重建出来有自己的坐标系 $\mathcal{F}_{splat}$，俩不一样。用 ICP 算法对齐，得到一个变换矩阵 $T^{\mathcal{F}_{splat}}_{\mathcal{F}_{robot}}$。

### 第四步：跑 sim，渲染图

在 PyBullet 里跑 physics，每一步给出 joint angles $q_t$ 和 object pose $x^k_t$。用 forward kinematics 算出每个 link 在 sim frame 里的 transformation $T^l_{fk}$，然后用这个 conjugation 公式（这是 paper 最 elegant 的一步）：

$$T = (T^{\mathcal{F}_{splat}}_{\mathcal{F}_{robot}})^{-1} \cdot T^l_{fk} \cdot T^{\mathcal{F}_{splat}}_{\mathcal{F}_{robot}}$$

直觉就是：把 Gaussian 从 splat frame 搬到 sim frame，在 sim frame 里做 forward kinematics 变换，再搬回 splat frame。就是 change of basis。然后对每个 Gaussian 应用：

$$\mu' = R\mu + t$$
$$\Sigma' = R\Sigma R^T$$

mean 就是平移加旋转，covariance 因为描述形状，左右各乘一个 $R$ 和 $R^T$（conjugation），保持体积不变。这就是 Gaussian Splatting 适合 rigid body manipulation 的根本原因 —— 每个 Gaussian 可以独立 transform，干净利落。

然后调用标准 Gaussian Splatting rasterizer 渲染出这一帧的 RGB 图。

### 第五步：训 policy

把 sim 跑出来的 trajectory $\tau_{\mathcal{E}} = \{(s_1, a_1), ..., (s_T, a_T)\}$ 每一步都渲染成 photorealistic RGB $I^{sim}_t$，得到 $\tau_{\mathcal{G}} = \{(I^{sim}_1, a_1), ..., (I^{sim}_T, a_T)\}$。

用 Diffusion Policy 训，input 是 RGB + end-effector pose，output 是 next action。

### 第六步：部署

freeze policy，直接扔到 real robot 上。real camera 出来的 RGB 图直接喂进去，policy 出 action，robot 执行。整个过程 zero-shot，不需要 real data fine-tune。

---

## 关键 trick：Augmentation 必不可少

SplatSim 渲染虽然 photorealistic，但还是有系统差异：
- **没 shadow**：Gaussian Splatting 是 radiance field，不显式建模 light source，每个 Gaussian 只学一个 view-dependent color，不会投影 shadow
- **Cable 不动**：robot 的 power cable 在 real 里会晃，sim 里按 rigid body 处理，cable 僵硬
- **Reflectance 不变**：joint 变化时 view-dependent 反射理论上应该变，但 splat 重建时没学到这种动态

所以 training 时要加 augmentation：gaussian noise、color jitter、random erasing。

效果多惊人？**不加 augmentation 21%，加了 86.25%**。65 个百分点的 gap。这说明 SplatSim 渲染虽然大幅缩小了 domain shift，但残留的 shift 依然致命，需要 augmentation 补齐。这其实就是一种温和的 domain randomization。

---

## 实验结果说话

| Task | Sim2Sim | Real2Real | Sim2Real (SplatSim) |
|------|---------|-----------|---------------------|
| T-Push | 100% | 100% | 90% |
| Pick-Up-Apple | 100% | 100% | 95% |
| Orange-On-Plate | 97.5% | 95% | 90% |
| Assembly | 85% | 90% | 70% |
| **Total** | **95.62%** | **97.5%** | **86.25%** |

86.25% zero-shot Sim2Real，对比 real data 训练的 97.5%，只差 11 个百分点。这在 RGB manipulation 上是 SOTA 级别。传统 domain randomization 方法在 RGB 上往往 0-30% 成功率，差距巨大。

Assembly 最差（70%）是因为需要精确 placement，cube 叠 cube，没 shadow 导致深度感知模糊，加上 contact dynamics 敏感。

数据收集时间 3 小时 vs 20.5 小时，省 85% 人力。

Rendering quality：PSNR 22.62 dB，SSIM 0.7845，在 robot 新 pose 下与 real 对比，相当不错。

---

## 为什么用 Gaussian Splatting 不用 NeRF

1. **Explicit primitive**：每个 Gaussian 独立可操作，能直接 segment、transform。NeRF 是 implicit function $F_\theta(x,d) \to (\sigma, c)$，要把 transform 嵌进 MLP 输入，分割 link 几乎不可能。
2. **Real-time rendering**：Gaussian Splatting tile-based rasterizer 1080p >100 FPS，NeRF 逐像素 MLP query 慢几个数量级。生成大规模 data 时这是 bottleneck。
3. **几何精度**：Gaussian 有显式 3D 位置，ICP 直接可用。NeRF 没显式点云，需要额外导出。

---

## 跟其他工作的区别

- **RialTo**：也走 Real2Sim2Real，但 policy input 是 point cloud，test time 需要 depth camera。SplatSim 只要 RGB。
- **Maniwhere**：大规模 RL，但 test time 仍需 depth。
- **Embodied Gaussians**：学 forward model for robot-object interaction，每个新 robot/object 都要 real data。SplatSim 把 dynamics offload 给 physics engine，只需一段 static scene video。
- **RoboStudio**：关注 system identification，不是 policy data generation。
- **GS Navigation**：做 navigation，agent 不与环境交互。SplatSim 做 manipulation，需要精确接触渲染。

---

## Limitations

1. **只能 rigid body**：cloth、liquid、plants 不行。Gaussian rigid transformation 假设每个 Gaussian 跟 link 一起运动，柔性物体需要 deformation model。
2. **Shadow 缺失**：这是最大的 visual gap 来源。
3. **Cable 失真**：rigid body 假设下 cable 不会自然摆动。
4. **Assembly 任务 70%**：精细 placement 对 shadow 和 contact 渲染敏感。

---

## 我的直觉总结

SplatSim 给我的最大启发：**Sim2Real gap 的本质是 distribution shift，源头 90% 来自 rendering pipeline 的近似误差**。

传统两条路：
- Domain randomization：把误差 randomize 掉，代价是 policy 保守、sample efficiency 低
- Collect real data：昂贵

SplatSim 第三条路：**用 SOTA 3D reconstruction 把 rendering 误差降到 policy 容忍范围内，再用 mild augmentation 兜底**。

公式 (3) 的 conjugation $T = A^{-1}BA$ 是整个方法的代数灵魂 —— 把 simulator 的 physics 表达翻译到 splat 的视觉表达。change of basis，简洁、通用、可扩展。

剩下的 11% gap 主要来自 shadow、cable、precise contact，这些都是 Gaussian Splatting 当前 limitation，也是未来 1-2 年明确改进路径。如果 shadow 问题能解决（比如额外训个 shadow map network 或把 shadow 作为附着 Gaussian），Assembly 任务应该能从 70% 提到 85%+。

这个工作的更大意义在于它验证了一个范式：**"photorealistic sim + mild augmentation + behavior cloning" 可以达到接近 real data 训练的效果**。如果这个范式能 scale 到更多场景、更多任务，real-world data collection 的角色会从"训练数据"退化成"场景扫描 + 最终评测"，中间的 policy learning 完全在 sim 里完成。这对整个 robotics field 的 data efficiency 意义重大。

---

# SplatSim: 用 Gaussian Splatting 消解 RGB Sim2Real Gap

## 1. 论文核心直觉 (Build Intuition First)

Sim2Real 在 RGB modality 上的根本痛点可以一句话总结：**simulator 生成的 RGB 图像分布 与 real-world RGB 图像分布 之间存在巨大的 domain shift**。深度、点云、触觉这类 modality 的 Sim2Real gap 较小，因为它们的数值在 sim 和 real 中几乎一致。但 RGB 图像承载了 color、texture、lighting、specular reflectance、shadow 等高维视觉信息，传统 mesh-based rendering pipeline（如 PyBullet / MuJoCo / Isaac Sim 默认 renderer）根本无法复现这些细节，导致 policy 在 sim 上学到的视觉特征对 real-world 不具备 generalization 能力。

SplatSim 的核心 insight 是：**把 rendering primitive 从 mesh 替换成 3D Gaussian Splats，同时保留 simulator 作为 physics backend**。这样既享受 Gaussian Splatting 的 photorealism，又享受 simulator 的 scalability、parallelization、cost-efficiency 与 safety。本质上是一个 "rendering 替换" 操作，physics engine 不变，policy 输入 modality 不变（仍然只是 RGB），但训练数据的视觉分布大幅接近真实世界。

项目主页：https://splatsim.github.io

参考 Gaussian Splatting 原文：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

## 2. 方法架构详解

### 2.1 整体 Pipeline (Fig. 2 解析)

Pipeline 分上下两阶段：

**上半部分（Data Generation）:**
- (a) 在 PyBullet physics simulator 中用 expert 收集 demonstrations。Expert 有两种来源：human teleoperation via Gello [50] 或 privileged-information motion planner（直接读取 simulator 中 rigid body 的 ground truth pose 来规划轨迹）。
- (b) 把 simulator 中的 trajectory $\tau_{\mathcal{E}} = \{(s_1, a_1), \dots, (s_T, a_T)\}$ 输入到 simulator-aligned 的 splat models，通过 rigid body transformation 渲染出 photorealistic RGB 序列 $I^{sim}_t$。
- 这些 $(I^{sim}_t, a_t)$ pairs 作为 Diffusion Policy [51] 的训练数据。

**下半部分（Deployment）:**
- 训练完成后 freeze policy，直接 zero-shot 部署到 real robot。Policy 在 test time 只接收 real-world RGB 图像 $I^{real}$，不依赖 simulator 或 splat。

### 2.2 问题形式化 (Sec. IV-A)

记号定义：
- $S_{real}$：从多个 RGB viewpoints 采集的真实场景的 Gaussian Splat（含 robot）
- $S^k_{obj}$：第 k 个 object 的 splat，从多视角采集
- $I^{sim}$：从 splat 渲染出的 photorealistic 图像
- $E$：expert（human 或 motion planner）
- $\tau_{\mathcal{E}} = \{(s_1, a_1), \dots, (s_T, a_T)\}$：一条 episode 轨迹

状态 $s_t = (q_t, x^1_t, \dots, x^n_t)$ 其中：
- $q_t \in \mathbb{R}^m$：robot joint angles（m 个 joint）
- $x^k_t = (p^k_t, R^k_t)$：第 k 个 object 的 position $p^k_t \in \mathbb{R}^3$ 与 orientation $R^k_t \in SO(3)$

Action $a_t = (p^e_t, R^e_t)$：end-effector 的 position 与 orientation。

### 2.3 坐标系定义 (Sec. IV-B) — 这是理解方法的关键

四个坐标系之间的关系决定了整个变换的代数结构：

| Frame | 含义 |
|-------|------|
| $\mathcal{F}_{real}$ | 真实世界参考 frame（primary） |
| $\mathcal{F}_{sim}$ | simulator frame，aligned with $\mathcal{F}_{real}$ |
| $\mathcal{F}_{robot}$ | 真实 robot base frame，aligned with $\mathcal{F}_{real}$ |
| $\mathcal{F}_{splat}$ | robot base 在 Gaussian Splat 场景中的 frame |
| $\mathcal{F}_{k\text{-}obj,sim}$ | 第 k 个 object 在 simulator 中的 frame（初始位于原点，无旋转） |
| $\mathcal{F}_{k\text{-}obj,splat}$ | 第 k 个 object 在其 splat 中的 frame |

关键 insight: $\mathcal{F}_{sim}$ 与 $\mathcal{F}_{robot}$ 都和 $\mathcal{F}_{real}$ 对齐，意味着 simulator 中 robot base 和 real-world robot base 共享同一个坐标系。但是 splat point cloud 是从 SfM/MVS 重建出来的，它的坐标系 $\mathcal{F}_{splat}$ 与 $\mathcal{F}_{real}$ 不一致，需要标定一个 transformation $T^{\mathcal{F}_{splat}}_{\mathcal{F}_{robot}}$ 来 bridge。

### 2.4 Rigid Body 变换的数学 (Eq. 1, 2)

3D Gaussian Splatting 中每个 primitive 是一个 anisotropic 3D Gaussian，由两组参数描述：
- $\mu \in \mathbb{R}^3$：mean（中心位置）
- $\Sigma \in \mathbb{R}^{3\times 3}$：covariance matrix（描述 Gaussian 的形状、方向、scale）

当对一个 Gaussian 应用 rigid transformation $T = (R, t)$ 时：
$$\mu' = R\mu + t \tag{1}$$
$$\Sigma' = R \Sigma R^T \tag{2}$$

变量解释：
- $\mu'$：变换后的 mean position
- $R \in SO(3)$：rotation matrix
- $t \in \mathbb{R}^3$：translation vector
- $\mu$：原始 mean
- $\Sigma'$：变换后的 covariance
- $\Sigma$：原始 covariance
- $R^T$：R 的转置（对正交矩阵即逆）

**直觉**：mean 是 affine 变换（先旋转再平移）；covariance 是 conjugation $R \Sigma R^T$ 而非简单的 $R\Sigma$，这是因为 covariance 描述的是分布的形状，刚体变换保持形状体积不变，需要左右各乘一个 $R$ 与 $R^T$（这与 $SO(3)$ 的保距性质一致）。这正是 Gaussian Splatting 适合做 rigid body manipulation 的根本原因 —— explicit primitive 使得每个 Gaussian 可独立 transform，不像 NeRF 那样要把变换嵌进 MLP 输入中，无法干净地分割 rigid parts。

### 2.5 Robot Splat Models (Sec. IV-C) — 三步法

参考 Fig. 3：

**Step 1: ICP Alignment**

从 $S_{real}$ 中手动 segment 出 robot 的 3D Gaussians，取它们的 means 组成 point cloud $P_{splat}$。再从 simulator 中获取 robot 在 home pose 下的 ground truth point cloud $P_{sim}$。用 Iterative Closest Point (ICP) 算法求最优 rigid transformation:

$$T^{\mathcal{F}_{splat}}_{\mathcal{F}_{robot}} = \arg\min_T \sum_i \| T \cdot P_{splat}^{(i)} - P_{sim}^{(nearest(i))} \|^2$$

ICP 参考：https://en.wikipedia.org/wiki/Iterative_closest_point

**Step 2: Robot Link Segmentation**

利用 robot CAD model 提供的 ground truth axis-aligned bounding boxes (AABB)，把每个 link 的 3D Gaussians 分割出来，记为 $\bar{S}^l_{real}$，其中 $l$ 是 link index。

**Step 3: Forward Kinematics Transformation**

这一步是整篇 paper 最 elegant 的代数操作。给定 simulator 在 time $t$ 给出的 joint angles $q_t$，PyBullet 的 forward kinematics 函数给出 link $l$ 在 $\mathcal{F}_{sim}$ 中的 transformation $T^l_{fk}$。需要把它"翻译"到 splat frame 中去应用，公式 (3)：

$$T = (T^{\mathcal{F}_{splat}}_{\mathcal{F}_{robot}})^{-1} \cdot T^l_{fk} \cdot T^{\mathcal{F}_{splat}}_{\mathcal{F}_{robot}} \tag{3}$$

变量解释：
- $T$：最终作用于 link $l$ 的 Gaussians 的 transformation
- $T^{\mathcal{F}_{splat}}_{\mathcal{F}_{robot}}$：splat frame 到 robot(sim) frame 的变换
- $(T^{\mathcal{F}_{splat}}_{\mathcal{F}_{robot}})^{-1}$：其逆，即 robot frame 到 splat frame 的变换
- $T^l_{fk}$：link $l$ 在 simulator frame 中的 forward kinematics 变换

**直觉**：这是一个 conjugation $A^{-1} B A$ 结构。物理意义是：在 splat frame 中描述的 Gaussians → 变换到 sim frame → 在 sim frame 中应用 forward kinematics → 再变换回 splat frame。这是 change of basis 的标准操作，等价于把 $T^l_{fk}$ 这个"sim frame 内的运动"通过 similarity transform 表达成"splat frame 内的运动"。这在 Lie group 理论里就是 adjoint action。

应用 $T$ 到 $\bar{S}^l_{real}$ 中每个 Gaussian 上用 Eq. 1 与 Eq. 2，再调用 Gaussian Splatting 的标准 rasterizer 渲染。

### 2.6 Object Splat Models (Sec. IV-D) — Eq. 4

Object 的处理多一步，因为 object 在 splat 中的初始姿态未必在原点：

$$T = (T^{\mathcal{F}_{splat}}_{\mathcal{F}_{robot}})^{-1} \cdot T^{k\text{-}obj}_{fk} \cdot T^{\mathcal{F}_{k\text{-}obj,splat}}_{\mathcal{F}_{k\text{-}obj,sim}} \tag{4}$$

变量解释：
- $T^{k\text{-}obj}_{fk}$：object $k$ 在 sim frame 中的当前 pose（simulator 在 time $t$ 输出）
- $T^{\mathcal{F}_{k\text{-}obj,splat}}_{\mathcal{F}_{k\text{-}obj,sim}}$：object $k$ 从 sim frame 到 splat frame 的 alignment（通过 ICP 得到）

直觉：先在 sim frame 中得到 object 当前的 pose，再用 ICP 给出的 splat-sim 偏移把 object 摆回 splat scene 的"自然位置"，最后用整体 splat-robot 变换嵌回 splat 世界。

### 2.7 Articulated Object Segmentation (Sec. IV-E)

对于 parallel jaw gripper 这类 articulated object，axis-aligned bounding box 不足以分割 link（因为 gripper finger 沿非标准方向运动）。Solution：训练一个 KNN classifier，用 URDF 标注的 simulator point cloud 作为 ground truth labels，对每个 3D Gaussian 推断其所属 link class。参考 Fig. 4。

KNN 选择的原因猜测：splat 与 simulator 点云都密集，KNN 在这种 dense 对应问题上稳定且无需训练；同时 KNN 的 distance metric 可以选用欧氏距离，因为 ICP 已经把两个点云对齐过。

### 2.8 数据生成与 Policy Training (Sec. IV-F, IV-G)

把上面三块（robot、object、articulated object）合在一起，给定一条 sim trajectory $\tau_{\mathcal{E}}$，每一步 $s_t$ 都能渲染出 photorealistic RGB $I^{sim}_t$，得到 demonstration set $\tau_{\mathcal{G}} = \{(I^{sim}_1, a_1), \dots, (I^{sim}_T, a_T)\}$。

Policy 用 Diffusion Policy [51, 52]，state-of-the-art behavior cloning 方法。输入：
- RGB observation（来自 SplatSim rendering）
- End-effector position + orientation

输出：action（next end-effector pose）。

**Augmentation 极其重要**：SplatSim 渲染缺少 shadows、cable 等柔性部件的形变、动态反射变化。Training 时加入：
- Gaussian noise injection
- Random erasing
- Color jitter (brightness / contrast)

Augmentation 把性能从 21% 提到 86.25%（Sec. V-D）。这个 gap 巨大，说明尽管 SplatSim 大幅减小了 Sim2Real gap，残留的 distribution shift 依然显著，需要 augmentation 补齐。

Diffusion Policy 论文：https://diffusion-policy.cs.columbia.edu/

---

## 3. 实验数据表解析 (Table I)

| Task | Sim2Sim | Real2Real | Sim2Real (SplatSim) | Sim hrs | Real hrs |
|------|---------|-----------|---------------------|---------|----------|
| T-Push | 100% | 100% | 90% | 3.0 | 3.5 |
| Pick-Up-Apple | 100% | 100% | 95% | 0.0* | 3.5 |
| Orange-On-Plate | 97.5% | 95% | 90% | 0.0* | 6.0 |
| Assembly | 85% | 90% | 70% | 0.0* | 7.5 |
| **Total** | **95.62%** | **97.5%** | **86.25%** | **3.0** | **20.5** |

\* 表示该任务的 sim demonstrations 完全由 motion planner 自动生成，无 human effort。

**关键观察：**

1. **Total Sim2Real 86.25% vs Real2Real 97.5%**：gap 仅约 11 个百分点。考虑到完全 zero-shot，这个数字非常强。对比之下，传统 domain randomization 方法在 RGB manipulation 上往往 0-30% 成功率。

2. **Assembly 任务最差（70%）**：Assembly 需要精确 placement（cube 叠 cube），对 visual alignment 与 contact dynamics 极敏感。SplatSim 渲染缺少 shadows 可能导致深度感知模糊，加上 rigid body 假设使 cable 等柔性元素失真，对精细任务影响更大。

3. **数据收集时间 3h vs 20.5h**：节省 85% 人力。Pick-Up-Apple / Orange-On-Plate / Assembly 三任务完全自动化生成（标 *）。

4. **Sim2Sim 与 Real2Real 差距很小**：说明 Diffusion Policy 在数据分布一致时本就接近 ceiling。Sim2Real 下降主要来自 residual visual gap，而非 policy capacity 限制。

5. **每个任务 40 trials**：sample size 偏小但作者选了 contact-rich 任务，方差较高，70-95% 的区间可以接受。

---

## 4. Rendering Quality 量化 (Sec. V-C)

在 300 个不同 joint angles 下对比 rendered robot 与 real robot 的图像：
- **PSNR = 22.62 dB**
- **SSIM = 0.7845**

PSNR 解读：22.62 dB 对应 pixel-domain MSE 约为 $10^{-22.62/10} \approx 0.0055$（归一化到 [0,1]）。这个值在 NeRF/Gaussian Splatting 文献里属于中等偏上水平（典型场景 25-30 dB），但考虑到这里是 robot 在新 joint pose 下渲染且与真实对比，22.62 已经相当不错。

SSIM 0.7845：1.0 是完全一致，0.78 表示结构相似度良好，细节略有差异（常见于 reflective surfaces 与 shadows）。

PSNR/SSIM 参考：https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

---

## 5. Augmentation 影响 (Sec. V-D)

- 无 augmentation：21% 平均成功率
- 有 augmentation（gaussian noise + color jitter + random erasing）：86.25% 平均成功率

**65 个百分点的提升**。这说明 SplatSim 渲染虽然 photorealistic，但仍有系统性差异：
- 缺 shadows（Gaussian Splatting 不直接建模 lighting，只是 view-dependent color）
- Rigid body assumption 导致 cable 等柔性元素错位
- View-dependent reflectance 在 joint 变化时可能不一致

Augmentation 在 pixel-level 上扰动，强迫 policy 学到 invariant feature，避免对 spurious cue（如 sim 特有的高光模式）过拟合。这呼应了 domain randomization 的精神，但因为 base fidelity 高，augmentation 强度可以较温和，不需要 randomize texture/geometry。

参考 NeRF2Real [53] 也是用 NeRF 渲染 + augmentation 来做 bipedal motion 的 Sim2Real：
https://arxiv.org/abs/2304.06809

---

## 6. 与 Related Work 的差异化分析

### 6.1 与 RialTo [33] 对比

RialTo (RSS 2024) 也走 Real2Sim2Real 路径，但 policy input 是 **point cloud**，需要 depth sensing at test time。
- RialTo paper: https://manifold-ml.berkeley.edu/projects/rialto/

SplatSim 的关键差异：**test time 只需 RGB**，不需要 depth camera。在许多农业、service robotics 场景中 depth camera 在 outdoor / 强光下不可靠，RGB-only 部署更鲁棒。

### 6.2 与 Maniwhere [34] 对比

Maniwhere 做大规模 RL，但 test time 仍需 depth。
- Maniwhere: https://openreview.net/forum?id=jart4nhCQr

### 6.3 与 Embodied Gaussians [47] 对比

Embodied Gaussians 学一个 forward model for robot-object interaction，**每个新 robot / object 都要 real-world data**。
- Embodied Gaussians: https://openreview.net/forum?id=AEq0onGrN2

SplatSim 把 dynamics 完全 offload 给 physics engine，**只需一段 static scene video**（含 robot home pose），即可渲染任意 trajectory。

### 6.4 与 RoboStudio [48] 对比

RoboStudio 关注 system identification，不是 policy 数据生成。
- Robo-GS: https://arxiv.org/abs/2408.14873

### 6.5 与 Gaussian Splatting Navigation [49] 对比

Quach et al. 把 Gaussian Splatting + simulator 用于 drone navigation，但 agent 不与环境交互。
- GS navigation transfer: https://openreview.net/forum?id=ubq7Co6Cbv

SplatSim 解决的 manipulation 比 navigation 更难，因为需要精确的 robot-object 接触渲染。

### 6.6 与 PhysGaussian [43] 对比

PhysGaussian 把 physics 嵌入 Gaussian Splatting 做 generative dynamics。
- PhysGaussian: https://arxiv.org/abs/2311.12198

SplatSim 不做 generative dynamics，只做 rigid body transformation + physics engine for contact。

---

## 7. Limitations 与 Failure Modes

论文明确列出的限制：

1. **只能处理 rigid body**：cloth、liquid、plants 无法处理。这是因为 Gaussian Splatting 的 rigid transformation (Eq. 1, 2) 假设每个 Gaussian 跟着 rigid link 一起运动。柔性物体需要 deformation model，如 DeformGS [46] (https://openreview.net/forum?id=DeformGS) 或 4D-GS [40]。

2. **Shadow 缺失**：Gaussian Splatting 是 radiance field 而非完整光照模型，每个 Gaussian 只学一个 view-dependent color $c(\theta)$，不显式建模 light source 与 occlusion。这意味着 sim 渲染中 robot 与 object 都不投影 shadow 到桌面，real-world 中却有强烈 shadow，造成 distribution shift。这是为什么 augmentation 必不可少。

3. **Cable 等柔性 component**：robot 的 power cable 在 real 中会随机摆动，但在 sim 中按 rigid body 处理，渲染中 cable 不会随 robot motion 自然摆动。

4. **Assembly 任务 70%**：可能就是 shadow 缺失 + 精细 contact 渲染不准导致的。Cube 叠 cube 时，shadow 提供深度线索，缺失会让 policy 难以判断 cube 之间相对高度。

---

## 8. 我的延伸思考与相关联想

### 8.1 为什么 Gaussian Splatting 比 NeRF 更适合 Sim2Real

1. **Explicit representation**：每个 Gaussian 是独立 primitive，可以直接 segment、transform、compose。NeRF 是 implicit function $F_\theta(x, d) \to (\sigma, c)$，要把 rigid transform 嵌入 MLP 输入或用 inverse warping，分割 link 几乎不可能。
2. **Real-time rendering**：Gaussian Splatting 用 tile-based rasterizer 实时渲染（>100 FPS at 1080p），NeRF 需要逐像素 MLP query，慢几个数量级。生成大规模 demonstration 数据时这是 bottleneck。
3. **Editable**：可以删除某些 Gaussian、给某些 Gaussian 加透明度、改变颜色，方便做 augmentation 与 object insertion。
4. **几何精度**：Gaussian 显式表征 3D 位置，ICP alignment 直接可用。NeRF 没有显式点云，需要额外导出。

参考 3DGS 原文：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

### 8.2 与 Domain Randomization 的关系

传统 domain randomization [30, 31] 思路：把 sim 渲染弄得"足够多样"，强迫 policy 学 invariant feature。
- Domain Randomization: https://arxiv.org/abs/1709.07857

SplatSim 思路：把 sim 渲染弄得"足够真实"，缩小 domain gap 本身。两者互补 —— SplatSim 之后仍用 mild augmentation（一种温和 randomization）兜底 residual gap。可以看作 **"fidelity-first, randomization-second"** 范式。

### 8.3 与 Real2Sim2Real 范式

SplatSim 属于 Real2Sim2Real：先 real 扫描得到 splat（Real→Sim），在 sim 中生成数据训练 policy，再部署回 real（Sim→Real）。这条 pipeline 越来越成为主流，相关工作有：
- RialTo: https://manifold-ml.berkeley.edu/projects/rialto/
- NeRF2Real: https://arxiv.org/abs/2304.06809
- RoboStudio: https://arxiv.org/abs/2408.14873

### 8.4 与可微渲染 + RL 的结合

未来方向之一：把 Gaussian Splatting 渲染做 differentiable，结合 RL 让 policy 直接在 photorealistic sim 中学习，而不是先生成 dataset 再 behavior clone。相关工作：
- PhysGaussian: https://arxiv.org/abs/2311.12198
- Gaussian Splatting RL: https://gaussian-splatting.github.io/

### 8.5 与 Latent Policy / World Model 的联想

SplatSim 渲染的 photorealistic image 也可以作为 latent world model 的预训练数据。例如 Genie / DreamerV3 类方法在 sim 中学习 world dynamics，再 transfer 到 real。
- DreamerV3: https://arxiv.org/abs/2301.04104

### 8.6 与 Agriculture Robotics 的天然契合

论文 future work 提到 pruning 与 harvesting。这些任务特点：
- Outdoor 强光下 depth camera 不可靠，RGB 是更稳健 modality
- Fruit / leaf 是非 rigid 但接近 rigid（短时序内）
- Field data collection 困难，sim-first 范式高效

但 outdoor 场景的 Gaussian Splatting 重建挑战大（光照变化、动态背景），需要结合 outdoor 3DGS 工作。

### 8.7 与 Embodied AI Foundation Models 的关系

SplatSim 可视为一种 **"visual data engine"**：给一个场景 splat → 无限生成 photorealistic interaction 数据。这与 Open-X-Embodiment [https://robotics-transformer-x.github.io/] 与 RT-2 [https://robotics-transformer2.github.io/] 等 vision-language-action model 的数据饥渴天然契合。可以想象：用 SplatSim 大规模生成 photorealistic manipulation 数据，加入 VLA 训练集。

### 8.8 与"Sim is all you need"哲学

SplatSim 在某种意义上强化了"只要 sim 足够好，sim is all you need"的论点。如果渲染质量足够高、physics 足够准、augmentation 足够强，real-world data collection 可能主要价值在"扫描场景建 splat"与"最终评测"两步，中间 training 完全在 sim 中完成。这与 Tesla Bot、Figure AI 等公司的 sim-heavy strategy 思路一致。

### 8.9 与 Implicit Scene Reconstruction 的下一步

SplatSim 假设有静态场景 splat 与单独 object splat。实际上可以扩展到：
- Dynamic scene splat（4D-GS [40]: https://arxiv.org/abs/2311.16479）
- Self-supervised object segmentation（SAM3D [https://segment-anything.com/])
- Joint optimization of splat + physics parameters（system identification）

---

## 9. 一些可能改进方向 (Hallucinated Extensions)

### 9.1 Shadow Rendering via Gaussian Splatting

可以额外训练一个 shadow map network：给定 light source direction + Gaussian scene，预测每个 pixel 的 shadow intensity。或者把 shadow 作为额外一组 anisotropic Gaussians 附着在 link 上，跟 link 一起变换。

### 9.2 Soft Gaussian for Cables

把 cable 用一组串联的 Gaussian chain 建模，用 mass-spring physics 模拟其形变，渲染时把 spring 节点位置作为 Gaussian means。这可以解决 cable 渲染失真问题。

### 9.3 Differentiable Splatting + RL

让 splat rendering 可微，直接用 photorealistic image 作为 RL observation，policy gradient 通过渲染回传到 geometry。需要 differentiable Gaussian rasterizer：
- gsplat: https://github.com/nerfstudio-project/gsplat

### 9.4 Multi-Scene Generalization

当前每个场景需要单独 scan + segment + ICP。可以训练一个 foundation model 一次性 segment robot link、object、gripper across scenes，类似 SAM 在图像上的成功。

### 9.5 Tactile + RGB Fusion

虽然 SplatSim 主打 RGB-only，但结合 tactile simulation（如 TACTO [https://arxiv.org/abs/2012.08456]）可以做 multi-modal policy，进一步减小接触丰富任务的 Sim2Real gap。

---

## 10. 总结直觉

SplatSim 给我的最大启发是：**Sim2Real gap 的本质是 distribution shift，而 distribution shift 的源头 90% 来自 rendering pipeline 的近似误差**。传统思路要么 randomize 掉这些误差（domain randomization，代价是 sample efficiency 低、policy 保守），要么去 collect real data（昂贵）。SplatSim 选第三条路：**用 SOTA 3D reconstruction（Gaussian Splatting）把 rendering 误差降到 policy 容忍范围内**，再用 mild augmentation 兜底。

公式 (3) 的 conjugation 结构 $T = A^{-1} B A$ 是整个方法的代数灵魂 —— 它把 simulator 的 physics 表达翻译到 splat 的视觉表达。这个 change-of-basis 思路简洁、通用、可扩展，是 paper 最 elegant 的贡献。

实验结果 86.25% zero-shot Sim2Real 在 RGB manipulation 上是 SOTA 级别（对比经典方法 0-30%）。剩下的 11% gap 主要来自 shadow、cable、precise contact，这些都是 Gaussian Splatting 当前 limitation，也是该方向未来 1-2 年的明确改进路径。

---

## 关键参考链接汇总

- SplatSim 项目主页: https://splatsim.github.io
- 3D Gaussian Splatting 原始论文: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Gello Teleoperation: https://arxiv.org/abs/2309.13037
- RialTo: https://manifold-ml.berkeley.edu/projects/rialto/
- Maniwhere: https://openreview.net/forum?id=jart4nhCQr
- Embodied Gaussians: https://openreview.net/forum?id=AEq0onGrN2
- Robo-GS: https://arxiv.org/abs/2408.14873
- PhysGaussian: https://arxiv.org/abs/2311.12198
- 4D Gaussian Splatting: https://arxiv.org/abs/2311.16479
- DeformGS: https://openreview.net/forum?id=DeformGS
- Splat-MOVER: https://openreview.net/forum?id=8XFT1PatHy
- GraspSplats: https://openreview.net/forum?id=pPhTsonbXq
- GS Navigation Transfer: https://openreview.net/forum?id=ubq7Co6Cbv
- NeRF2Real: https://arxiv.org/abs/2304.06809
- Domain Randomization: https://arxiv.org/abs/1709.07857
- PyBullet: https://pybullet.org
- gsplat (differentiable rasterizer): https://github.com/nerfstudio-project/gsplat
- DreamerV3: https://arxiv.org/abs/2301.04104
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- RT-2: https://robotics-transformer2.github.io/
- TACTO (tactile sim): https://arxiv.org/abs/2012.08456
- SAM (Segment Anything): https://segment-anything.com/
- On Sim2Real Transfer (blog by Haonan Yu): https://www.haonanyu.blog/post/sim2real/
