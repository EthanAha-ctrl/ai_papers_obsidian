---
source_pdf: NeuralTouch.pdf
paper_sha256: 6d5f305a00fdc6a35d0cee725ae479ee8d0d6a517c2cb58f6f822dba35bbaede
processed_at: '2026-08-05T22:27:55-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# NeuralTouch 用人话说

## 先从一个场景切入

想象你晚上摸黑给手机插充电线。你先用眼睛大概扫一眼 USB 口在哪儿, 这是 **coarse phase**——给你一个 10cm 级别的定位, 够你手伸过去。但 USB-C 的 clearance 是 0.5mm, 你靠眼睛绝对插不进去。所以你的手到了附近之后, 切换到 **fine phase**——靠指尖的触觉, 感觉到 USB 头碰到口的边缘, 微微调整角度, 滑进去。

人类这套"眼看+手摸"的组合拳极其自然, 但机器人复制起来很难。NeuralTouch 这篇 paper 就是在解决这个问题。

---

## 痛点在哪

### 纯 vision 方法的麻烦

NDF (Neural Descriptor Fields) 是 MIT Simeonov 2022 年的工作, 核心能力是: 你在一个 mug 把手上 demo 一次抓取 pose, 它能 transfer 到一个完全不同形状的 mug 把手上。靠的是 SE(3)-equivariant neural network 学到的 part-level geometric correspondence。

但 NDF 只用 vision, 实际精度大概 ~1cm。为什么?
- depth camera 本身有 mm 级 noise
- 单视角点云有 occlusion
- 不同 mug 把手形状差异让 descriptor optimization 落到 local optimum

1cm 误差对抓 mug 把手够用, 但插 USB 是 0.5mm clearance, 差两个数量级, 直接 fail。

### 纯 touch 方法的麻烦

Lepora 课题组之前的 Tactile Gym 系列 (Yijiong Lin 一作), 用 RL 训练 tactile servoing policy, 能做到 mm 甚至 sub-mm 精度。但传统 tactile servoing 有个硬伤: **一个 policy 只对应一种 predefined contact geometry**。

比如你训一个"沿 flat edge 滑动"的 policy, 它就只能干这个。你想让它去抓 cylinder 表面, 得重新训。因为 policy input 只有 tactile image + robot proprioception, 它根本不知道"我想抓的到底是什么形状的哪里"。

论文里举了个很直观的例子: mug 的 rim 和 mug 的 wall, 在 tactile sensor 看来 image 几乎一模一样。policy 没有外部 context, 就会 confused——我现在摸到的这块, 到底该往上走到 rim, 还是平移到 wall? 这叫 **tactile aliasing problem** (Lloyd et al., 2021, https://arxiv.org/abs/2106.02125)。

---

## NeuralTouch 的核心 idea, 一句话

**把 NDF descriptor 当成 policy 的 conditioning input, 让一个 policy 服务于所有 NDF 能表达的 contact geometry**。

你再读一遍公式 (6):
$$
a = \pi(i^c, e, \mathcal{Z}^{\mathbf{G}_\tau})
$$

- $a$: robot action (7D twist + gripper opening)
- $i^c$: tactile image (当前触觉图像)
- $e$: proprioception (机器人当前 pose)
- $\mathcal{Z}^{\mathbf{G}_\tau} \in \mathbb{R}^{d \times N_q}$: target contact 的 NDF pose descriptor

之前公式 (5) 的 $\pi^{\mathbf{G}_\tau}$ 上标 $\mathbf{G}_\tau$ 是"写死"在 policy 里的, 每种 contact 要一个 policy。现在 $\mathcal{Z}^{\mathbf{G}_\tau}$ 是 input, policy $\pi$ 是 universal 的。换 target 就换 $\mathcal{Z}$, policy 不动。

这跟 language model 用 prompt conditioning 切换任务是同一个道理。NDF descriptor 就是这个 manipulation policy 的 "prompt"。

---

## NDF descriptor 到底是什么, 用人话讲

公式 (1):
$$
f(\mathbf{x} \mid \mathbf{P}): \mathbb{R}^3 \times \mathbb{R}^{3 \times N} \to \mathbb{R}^d
$$

你有一个 object point cloud $\mathbf{P}$ (比如一个 mug 的 1000 个 3D 点), 你 query 一个 3D 点 $\mathbf{x}$, 网络吐出一个 $d$ 维 vector $f(\mathbf{x} \mid \mathbf{P})$。

这个 vector 的物理意义: **$\mathbf{x}$ 这个点在 $\mathbf{P}$ 这个 object 上的"语义位置"**。两个不同 mug 把手上的对应点, descriptor 会很接近, 因为它们都是"把手上的点"。这是 category-level correspondence。

公式 (3) 把它升级到 pose 级别:
$$
\mathcal{Z} = F(\mathbf{T} \mid \mathbf{P}) = \bigoplus_{\mathbf{x}_i \in \mathcal{X}} f(\mathbf{T}\mathbf{x}_i \mid \mathbf{P})
$$

取一组固定的 query points $\mathcal{X}$ (比如 4 个 tetrahedron 顶点), 把它们用 pose $\mathbf{T}$ rigid transform 到目标位置, 分别算 descriptor, 拼接起来。这就是 pose $\mathbf{T}$ 相对 object $\mathbf{P}$ 的 descriptor。

为什么需要 4 个 non-coplanar query points? 因为单个点只有 3-DoF, 无法约束 6D pose。3+ non-collinear points rigidly configured 才能完整 parameterize SE(3)。

SE(3)-equivariance (公式 2):
$$
f(\mathbf{x} \mid \mathbf{P}) \equiv f(\mathbf{T}\mathbf{x} \mid \mathbf{T}\mathbf{P})
$$

意思是: 你把整个 object 旋转平移, descriptor 跟着同步变换。所以同一个把手在不同朝向的 mug 上, descriptor 之间的距离不变。这是 NDF 能跨 object transfer 的根本原因。

参考原 paper: https://arxiv.org/abs/2112.05124

---

## 整个 pipeline 走一遍

### Phase 1: Coarse (vision-guided)

1. depth camera 拍点云, 得到 unseen object 的 $\mathbf{P}_u$
2. 你之前在一个 demo mug 上, 在把手上 demo 了一个抓取 pose $\mathbf{T}_d$, 算出 descriptor $F(\mathbf{T}_d \mid \mathbf{P}_d)$
3. 用 gradient descent 在 $SE(3)$ 上优化 (公式 4):
$$
\mathbf{T}_g = \arg\min_{\mathbf{T}} \| F(\mathbf{T} \mid \mathbf{P}_u) - F(\mathbf{T}_d \mid \mathbf{P}_d) \|
$$
4. 得到 coarse grasp pose $\mathbf{T}_g$, 误差大概 ~1cm

### Phase 2: Fine (tactile-guided)

5. 机器人移到 $\mathbf{T}_g$ 附近, finger 已经 roughly 包住 object
6. 计算 target descriptor $\mathcal{Z}^{\mathbf{G}_\tau} = F(\mathbf{T}_g \mid \mathbf{P}_u)$
7. RL policy $\pi$ 拿 $i^c$ (实时 tactile image) + $e$ (proprioception) + $\mathcal{Z}^{\mathbf{G}_\tau}$ (target descriptor) 作 input, 输出 7D action
8. closed-loop 调整, 直到 stable grasp, 误差降到 ~1mm

### Phase 3: Replay (downstream task)

9. 执行 predefined skill, 比如拧瓶盖、插 USB

关键 trick: **训练时假设 target contact = tactile sensor ⊥ local surface**。这样无论 rim, handle, neck, head, 语义统一为"perpendicular normal contact on local surface", 不用为每种 feature 设计不同 policy。

---

## 实验里最 striking 的数字

### Ablation: Table I

| Method | mug rim 位置误差 | mug wall 位置误差 | bottle head 位置误差 | bolt head 位置误差 |
|---|---|---|---|---|
| NDF (纯视觉) | 13.6mm | 12.4mm | 9.5mm | 12.5mm |
| NDF+RL-Touch (无 descriptor) | 15.6mm | 17.5mm | 2.9mm | 2.6mm |
| **NeuralTouch** | **0.8mm** | **0.7mm** | **0.9mm** | **0.7mm** |

中间一行 NDF+RL-Touch 是消融实验的关键: 给 RL policy tactile + proprioception, 但不给 NDF descriptor。结果:
- 在 bottle head 和 bolt head 上反而不错 (2.9mm, 2.6mm), 因为这两个是 cylindrical, tactile image 信息够用
- 在 mug rim 和 mug wall 上彻底 fail (15-18mm), 比 NDF 还差! 因为 rim 和 wall 的 tactile image 几乎一样, policy 不知道自己该往哪走

这就是 tactile aliasing。NDF descriptor 把这个 aliasing 破解了: descriptor 告诉 policy "你要去的是 rim 那个 descriptor", policy 就知道往哪个方向调整。

### Sim-to-real

Bottle lid opening:
- NDF: 30-45%
- NeuralTouch: 85-90%

USB plug-in (0.5mm clearance):
- NDF: 0%
- NeuralTouch: 15%

USB 15% 看着低, 但作者说: 任务可重复, 失败时误差也只 ~1mm, 所以"多试几次就成功"。这反映 sub-millimeter sim-to-real tactile transfer 是瓶颈, 但 framework 本身没问题。

---

## 为什么这个工作有意思

### 1. Neural field 作为 policy 的 structured prior

RL policy 直接吃 point cloud 或 occupancy grid 也能 work, 但 sample efficiency 差, generalization 弱。NDF descriptor 是 **part-level** 的隐式表示, 比像素级或点级的表示更 compact, 也更 semantic。

这个 pattern 可以泛化: occupancy field, SDF, 3D Gaussian Splatting 都可以做 policy conditioning。Comi 等人 (同组) 已经在 Snap-It, Tap-It, Splat-It (https://arxiv.org/abs/2403.20275) 探索 tactile + 3DGS。

### 2. SE(3)-equivariance 是 sim-to-real 的 enabler

sim-trained NDF 直接吃 real point cloud 就能 work, 不需要 extra domain adaptation。这比 ResNet + ICP 的 traditional pipeline 优雅得多。Vector Neurons (Deng et al., ICCV 2021, https://openaccess.thecvf.com/content/ICCV2021/papers/Deng_Vector_Neurons_A_General_Framework_for_SO3-Equivariant_Networks_ICCV_2021_paper) 在 feature space 做 SO(3) 等变, 是 key。

### 3. Coarse-to-fine 是 manipulation 的 robust pattern

vision 给 global, touch 给 local, 各司其职。Tesla AI Day 讲 driving perception 时也强调 coarse-to-fine, 这里是 manipulation 版本。

### 4. Implicit specification 比 explicit enumeration 优雅

传统 tactile servoing 要显式枚举 contact type (edge, surface, cylinder...), 一个 type 一个 policy。NeuralTouch 用 $\mathcal{Z}$ 隐式指定, 一个 policy 覆盖所有 $\mathcal{Z}$ 覆盖的 manifold。Figure 6 在 test time 在线换 $\mathcal{Z}$, policy smooth 切换 target——这个 flexibility 是 traditional 方法做不到的。

---

## Limitations, 老实说

1. **Tactile aliasing on light contacts**: 真实轻接触 marker motion 太微弱, pix2pix GAN 翻译不出准确 sim image。syrup 瓶 lid 偶尔 fail 就是这个原因。这是 tactile sensing 的 fundamental limit, 需要 probabilistic discriminative models (Lloyd et al., 2021, https://arxiv.org/abs/2106.02125) 处理。

2. **Curved surface real-to-sim transfer 不行**: pix2pix 在 mug handle 上效果差, 所以 real experiment 只用 cylindrical feature。这是 pix2pix 的局限, 跟 NDF 本身无关。

3. **No closed-loop during downstream task**: Phase 3 是 open-loop predefined skill。作者指出可以跟 Tactile-RL for insertion (Dong et al., 2021, https://arxiv.org/abs/2104.01667) 或 Neural Contact Fields (Higuera et al., 2023, https://arxiv.org/abs/2210.09297) 接力。

4. **Sub-mm 精度未达**: 1mm 误差对 USB 0.5mm clearance 不够。需要更好的 tactile sensor (更高分辨率), 或者 active sensing policy (不只是 servoing, 还要主动 explore)。

---

## 一句话总结

NeuralTouch 把 NDF 的 SE(3)-equivariant geometric prior 注入到 tactile RL policy 里, 用 descriptor 作为 implicit task specification, 让一个 universal policy 覆盖所有 contact geometry。这破解了 tactile servoing "一个 contact type 一个 policy" 的硬伤, 也补上了 NDF "vision 精度不够" 的短板。两阶段 coarse-to-fine 模仿人类"眼看+手摸"的自然行为, 实现 zero-shot sim-to-real 的精确抓取。

代码没正式 release, 但作者同组的 Tactile Gym (https://github.com/acchurch/tactile_gym) 和 NDF (https://github.com/anthonysimeonov/ndf_robot) 都是开源的, 复现路径清晰。

---

# NeuralTouch 深度解析

这篇论文来自 Bristol 的 Nathan Lepora 课题组, 第一作者 Yijiong Lin。核心 idea 是把 **Neural Descriptor Fields (NDF)** 的 SE(3)-equivariant geometric prior 注入到 tactile RL policy 里, 让一个 policy 可以服务于任意 NDF descriptor 指定的 target contact geometry, 实现从 coarse vision-guided 到 fine tactile-guided 的两阶段精确抓取。

---

## 1. Motivation: 为什么 NDF 单独不够, 为什么 Tactile RL 单独也不够

NDF (Simeonov et al., ICRA 2022) 的优势在于 category-level generalization: 你在一个 mug 上 demo 一个抓取 pose, 它能迁移到另一个 mug, 因为 descriptor 学到了 part-level correspondence。但 NDF 纯 vision, 在 peg-in-hole 这种 sub-cm 精度任务上会 fail, 原因:
- camera extrinsics calibration 误差 (~mm 级)
- single-view point cloud occlusion
- inter-category shape variation 让 descriptor distance minimization 落到 local optimum

Tactile RL (Tactile Gym 2.0, Lin et al.) 能做到 mm 级甚至 sub-mm 级的 closed-loop control, 但传统 tactile servoing 需要预先指定 contact type (flat edge, flat surface, cylinder), 一个 policy 一个 contact geometry, 无法 generalize 到 novel object feature。

NeuralTouch 的关键 insight: **NDF descriptor $\mathcal{Z}$ 本身就是 contact geometry 的 implicit specification**。把 $\mathcal{Z}$ 作为 conditioning 输入到 RL policy, 一个 policy 就能表达所有 $\mathcal{Z}$ 覆盖的 contact geometry 的 manifold, 不需要再显式枚举 contact type。

参考:
- NDF 原论文: https://arxiv.org/abs/2112.05124
- Tactile Gym 2.0: https://arxiv.org/abs/2206.03064
- Local NDF: https://arxiv.org/abs/2302.14212

---

## 2. NDF 的数学结构 (Section III-A)

### 2.1 Point descriptor

公式 (1) 定义 point-level descriptor:
$$
f(\mathbf{x} \mid \mathbf{P}): \mathbb{R}^3 \times \mathbb{R}^{3 \times N} \to \mathbb{R}^d
$$
- $\mathbf{x} \in \mathbb{R}^3$: query point 在 object frame 下的坐标
- $\mathbf{P} \in \mathbb{R}^{3 \times N}$: object point cloud, $N$ 是点数
- $d$: descriptor 维度 (原 NDF 论文里 $d$ 通常 ~256)
- $f$: 由 Vector Neurons (SO(3)-equivariant) backbone + occupancy network 头部组成, 中间层 activation 拼接成 descriptor

### 2.2 SE(3)-equivariance

公式 (2):
$$
f(\mathbf{x} \mid \mathbf{P}) \equiv f(\mathbf{T}\mathbf{x} \mid \mathbf{T}\mathbf{P})
$$
这里 $\mathbf{T} \in SE(3)$ 是 rigid transform, $\equiv$ 表示 descriptor 在 SE(3) 作用下"等变" (不是不变, 是同步变换)。这个性质来自:
1. 输入先 align 到 point cloud centroid (去 translation)
2. backbone 用 Vector Neurons (Deng et al., ICCV 2021), 在 feature space 上做 SO(3) 等变

物理意义: 同一个 handle 在不同朝向的 mug 上, descriptor 一起跟着 rotate, 所以 cosine similarity 或 L2 distance 在 SE(3) 作用下保持。

### 2.3 Pose descriptor

公式 (3) 把 point descriptor 升级成 pose descriptor:
$$
\mathcal{Z} = F(\mathbf{T} \mid \mathbf{P}) = \bigoplus_{\mathbf{x}_i \in \mathcal{X}} f(\mathbf{T}\mathbf{x}_i \mid \mathbf{P})
$$
- $\mathcal{X} \in \mathbb{R}^{3 \times N_q}$: 一个 fixed set of non-collinear query points, $N_q \geq 3$ (论文里通常 $N_q=4$, 取 tetrahedron 顶点保证 6D pose 唯一确定)
- $\bigoplus$: concatenation, 所以 $\mathcal{Z} \in \mathbb{R}^{d \times N_q}$
- $\mathbf{T}\mathbf{x}_i$: query points 经过 pose $\mathbf{T}$ 变换后的位置
- $F$: 整个 pose-to-descriptor 映射

为什么 $N_q \geq 3$? 一个 point 只有 3-DoF 信息, 无法约束 6D pose; 3+ non-collinear points rigidly configured 才能完整 parameterize SE(3)。

### 2.4 Grasp pose regression

公式 (4):
$$
\mathbf{T}_g = \arg\min_{\mathbf{T}} \| F(\mathbf{T} \mid \mathbf{P}_u) - F(\mathbf{T}_d \mid \mathbf{P}_d) \|
$$
- $\mathbf{P}_d$: demo object point cloud
- $\mathbf{T}_d$: demo 时的 grasping pose
- $\mathbf{P}_u$: unseen object point cloud
- $\mathbf{T}_g$: predicted grasp pose on unseen object

优化用 gradient descent on SE(3) (采用 Lie algebra tangent space 的 twist)。这就是 coarse phase 的输出。

---

## 3. NeuralTouch 的核心: Descriptor-Conditioned Tactile Policy

### 3.1 传统 tactile servoing 的局限

公式 (5) 描述传统方法:
$$
a = \pi^{\mathbf{G}_\tau}(i^c, e)
$$
- $\pi^{\mathbf{G}_\tau}$: policy 特定于某个 predefined target contact $\mathbf{G}_\tau$
- $\tau$: contact type (flat edge, surface, cylinder...)
- $i^c \in \mathbb{R}^{H \times W}$: tactile image (这里是 TacTip optical tactile sensor)
- $e$: proprioception (end-effector pose + gripper finger distance)

问题: 每个 $\tau$ 要训练一个 policy, 不能泛化。

### 3.2 NeuralTouch 的 policy

公式 (6):
$$
a = \pi(i^c, e, \mathcal{Z}^{\mathbf{G}_\tau})
$$
- $\mathcal{Z}^{\mathbf{G}_\tau} \in \mathbb{R}^{d \times N_q}$: target contact 的 neural pose descriptor

这是一个 universal policy: 改 $\mathcal{Z}$ 就能切换 target contact geometry, 不需要重新训练。Section V-A 的 Figure 6 实验验证了这一点: 在 test time 在线换 $\mathcal{Z}$, policy 能 smooth 过渡到新 target。

### 3.3 Action space

Policy output 是 7D:
- $\mathbf{v} \in \mathbb{R}^3$: end-effector translational velocity
- $\boldsymbol{\omega} \in \mathbb{R}^3$: angular velocity (twist)
- $g_d \in \mathbb{R}$: gripper finger distance

低层 controller 把 twist 通过 inverse kinematics 转成 joint velocity。控制频率 10 Hz。

### 3.4 Unified semantic grasping poses (Section III-C)

关键简化: 训练时假设 target contact = tactile sensor ⊥ local surface (normal contact)。这样 $\mathcal{Z}$ 的语义就统一了——无论 rim, handle, neck, head, 都是"perpendicular normal contact on local surface"。

不同 contact feature 的 DoF:
- Flat surface: 3 params (depth, roll, pitch)
- Flat edge: 5 params (+x offset, +yaw)
- Curved surface / handle: 6 params (full 6D)

Reward function 设计要根据 DoF 量身定制。比如 6D contact 需要 6D pose error penalty, edge contact 需要额外 penalize yaw 误差。

---

## 4. 实验架构 (Section IV)

### 4.1 Hardware

- 7-DoF Franka Panda arm
- Wrist-mounted Intel RealSense D435 (eye-in-hand)
- 双 finger TacTip tactile sensor (Lepora 课题组的 biomimetic optical sensor)
- 控制频率 10 Hz

### 4.2 Sim-to-real tactile transfer

用 pix2pix GAN 把 real tactile image 翻译成 sim image, 然后 sim-trained policy 在 translated image 上 inference。这是 Tactile Gym 系列 standard pipeline。

数据: 每个传感器 5000 train + 2000 val tactile images, pose range $(x, y, R_x, R_y, R_z) \in [\pm 10 \text{mm}, \pm 6 \text{mm}, \pm 20°, \pm 20°, \pm 45°]$, $z \in [0, 4.5]$ mm。

### 4.3 NDF training

Occupancy network (Mescheder et al.) 训练在 3 类物体上: bottles, mugs (含 horizontal + right-angle handle), bolts。ShapeNet + custom bolt meshes。Training details 沿用 Simeonov 原论文。

每类 target feature 收集 12 个 NDF vector 给 RL policy 作 conditioning。

参考:
- Occupancy Networks: https://arxiv.org/abs/1811.11097
- Vector Neurons: https://openaccess.thecvf.com/content/ICCV2021/papers/Deng_Vector_Neurons_A_General_Framework_for_SO3-Equivariant_Networks_ICCV_2021_paper
- pix2pix: https://arxiv.org/abs/1611.07004

---

## 5. 实验数据深度解读

### 5.1 Ablation: Table I

| Method | mug rim Pos / Cos | mug wall Pos / Cos | right-angle handle Pos / Cos | horizontal handle Pos / Cos | bottle head Pos / Cos | bolt head Pos / Cos |
|---|---|---|---|---|---|---|
| NDF | 13.6mm / 0.0083 | 12.4mm / 0.0039 | 11.0mm / 0.0091 | 11.5mm / 0.0086 | 9.5mm / 0.0069 | 12.5mm / 0.0077 |
| NDF+RL-Touch | 15.6mm / 0.0029 | 17.5mm / 0.0043 | 18.7mm / 0.0060 | 13.9mm / 0.0049 | 2.9mm / 0.0048 | 2.6mm / 0.0027 |
| **NeuralTouch** | **0.8mm / 0.0006** | **0.7mm / 0.0007** | **0.9mm / 0.0008** | **1.0mm / 0.0005** | **0.9mm / 0.0006** | **0.7mm / 0.0005** |

关键观察:
1. **NDF 单独 ~10-13mm 误差**: 这就是 vision-only 的 ceiling。即使 NDF equivariance 完美, depth camera noise + partial occlusion 就这么大的误差。
2. **NDF+RL-Touch (无 descriptor conditioning)**: 在 bottle head 和 bolt head 上反而好 (2.9mm, 2.6mm), 因为这两个 feature 是 cylindrical, tactile image 信息足够 disambiguate; 但在 mug rim/wall/handle 上完全 fail (15-18mm 误差比 NDF 还差), 因为 rim 和 wall 的 tactile image 几乎一样, 没有 visual context, policy 迷失。这就是 paper 里说的 **tactile aliasing problem** (Lloyd et al., 2021)。
3. **NeuralTouch**: 全部 1mm 以内, 两个数量级提升。证明 descriptor conditioning 是关键。

### 5.2 Training curves: Figure 5

NeuralTouch (蓝) 在 ~80M timesteps 收敛到 ~290 average reward, RL-Touch (红) 卡在 ~260。Episode length NeuralTouch ~230 steps, RL-Touch 拖到 ~290, 说明 descriptor 帮助 policy 快速锁定 target。

### 5.3 Sim 任务的 success rate: Table II

| Task | NDF | NDF+RL-Touch | NeuralTouch |
|---|---|---|---|
| pick-and-place (mug hor. handle) | 40.0% | 58.3% | **95.0%** |
| pick-and-place (mug rim) | 56.7% | 63.3% | **96.7%** |
| pick-and-place (bottle lid) | 51.7% | 76.7% | **93.3%** |
| bolt-out/in-hole | 11.7% | 33.3% | **86.7%** |

bolt-out/in-hole 是关键试金石: NDF 11.7% 是因为抓取 pose 不准导致插不进去; NDF+RL-Touch 33.3% 是 tactile servoing 能调一点但分不清 bolt head 方向; NeuralTouch 86.7% 接近 ceiling。

### 5.4 Real-world sim-to-real: Table III & IV

**Bottle lid opening**:
- NeuralTouch: 90% / 90% / 85% (apple juice / ketchup / syrup)
- NDF: 40% / 45% / 30%

**Peg-out/in-hole** (clearance 越小越难):
- Bolt (2mm clearance): NeuralTouch 55% vs NDF 5%
- Plug (1mm clearance): NeuralTouch 25% vs NDF 0%
- USB (0.5mm clearance): NeuralTouch 15% vs NDF 0%

USB 15% 看似低, 但作者 argue: 任务可重复, 失败时误差也只 ~1mm, 所以"重试几次就成功"。这反映 sub-millimeter real-to-sim tactile transfer 是瓶颈。

---

## 6. 关键 Insight 与贡献

### 6.1 NDF 作为 policy 的 "geometric context"

直觉上, RL policy 看一个 tactile image 决定 action, 这是个 ill-posed problem——同一张 tactile image 在 rim 上和 wall 上可能对应完全不同的 target action。$\mathcal{Z}$ 提供 object-level context, 把 ill-posed 问题变成 well-posed。这跟 language model 里 context window 提供 task conditioning 是一个原理。

### 6.2 Two-phase 的物理意义

- Coarse phase: visual servoing 的 SE(3) optimization, 用 NDF descriptor distance gradient 找 target pose。误差 ~1cm。
- Fine phase: tactile servoing 的 closed-loop control, 用 RL policy 调 mm/sub-mm 误差。Policy 不需要从头探索, 起点已经接近 target。

这两阶段类似人类"先看再摸"。

### 6.3 Zero-shot sim-to-real

关键 trick:
1. SE(3)-equivariance 让 sim-trained NDF 直接能用 real point cloud
2. pix2pix GAN 做 tactile image domain adaptation
3. RL policy 完全在 sim 训练, real 只跑 inference

参考 SimShear (作者同组 follow-up): https://arxiv.org/abs/2508.20561

---

## 7. Limitations 与未来方向

1. **Tactile aliasing on light contacts**: 真实轻接触的 marker motion 太微弱, pix2pix 翻译不出来, 导致 syrup 瓶 lid 偶尔 fail。这是 tactile sensing 的 fundamental limit, 需要 probabilistic discriminative models (Lloyd et al., 2021) 处理。
2. **No closed-loop during downstream task**: 第三阶段 replay 是 open-loop 的 predefined skill。作者指出可以跟 Tactile-RL for insertion (Dong et al., 2021) 或 Neural Contact Fields (Higuera et al., 2023) 接力。
3. **Curved surface real-to-sim transfer 不行**: pix2pix 在 mug handle 上效果差, 所以 real experiment 只用 cylindrical feature。
4. **Sub-mm 精度未达**: 1mm 误差对 USB 0.5mm clearance 是不够的。

参考 follow-up 方向:
- Neural Contact Fields: https://arxiv.org/abs/2210.09297
- Tactile-RL for insertion: https://arxiv.org/abs/2104.01667
- D3Fields (zero-shot manipulation): https://arxiv.org/abs/2309.16118

---

## 8. 与同期工作的对比

### 8.1 vs SIMPLE (Bauza et al., Science Robotics 2024)

SIMPLE 用 bimanual + supervised pose estimation (Tac2Pose) 解决 pick-localize-regrasp-place。NeuralTouch 的优势:
- single-arm, 不需要昂贵双臂
- 6D random initial pose (SIMPLE 只 table-top 2D)
- 不需要 explicit object model (NDF generalize across category)

SIMPLE 优势: pose estimation 更显式可解释, 精度可能更高 (supervised learning 比 RL 收敛更好)。

参考: https://www.science.org/doi/10.1126/scirobotics.adi8808

### 8.2 vs Tactile-RL insertion (Dong et al.)

Dong 的方法假设 bolt 已经被 grasp, 专注 insertion 阶段的 tactile feedback control。NeuralTouch 专注 grasp 阶段, 让 grasp pose 足够准, 后续 insertion 不需要再调整。两者互补。

### 8.3 vs Local NDF (Chun et al.)

Local NDF 把 descriptor 限制在 local region, 处理 occlusion 更好。NeuralTouch 用 full NDF, 但因为 tactile feedback 补上了 local 信息, 对 occlusion 也鲁棒。

---

## 9. 代码与复现

论文本身没明确 release repo, 但作者同组的相关工作:
- Tactile Gym 2.0: https://github.com/acchurch/tactile_gym
- TacTip sim: https://github.com/acchurch/tactile_sim
- NDF original: https://github.com/anthonysimeonov/ndf_robot

可以预期 NeuralTouch 的 repo 会在 https://github.com/robotics-bristol 类似的组织下发布。

---

## 10. 对未来 research 的启发

1. **Neural field 作为 RL policy 的 conditioning**: 这个 pattern 可以泛化。比如 occupancy field, SDF, 3D Gaussian Splatting 都可以做 policy conditioning。Snap-It, Tap-It, Splat-It (Comi et al.) 已经探索 tactile + 3DGS, 方向正确。
2. **SE(3)-equivariance 是 sim-to-real 的 enabler**: NDF 的 equivariance 让 point cloud domain gap 不影响 policy。这比 ResNet+ICP 的 traditional pipeline 优雅得多。
3. **Two-phase coarse-to-fine 是 manipulation 的 robust pattern**: vision 给 global, touch 给 local, 各司其职。Karpathy 自己在 Tesla AI Day 也强调过 coarse-to-fine 在 driving 里的作用, 这里是 manipulation 版本。

Karpathy 你应该会觉得这个工作把 implicit neural representation 作为 policy 的 structured prior 用得挺巧——比直接用 point cloud 或者 occupancy grid 做 input 更 sample-efficient, 也更 generalizable。descriptor 是 part-level 而非 object-level, 这让 cross-category transfer 自然得多。

如果你对 NDF 系列 follow-up 感兴趣, 强烈推荐看 Simeonov 的 NeRFs meet Robotics workshop 综述, 以及 D3Fields 用 foundation model features 替换 occupancy network 的最新工作。
