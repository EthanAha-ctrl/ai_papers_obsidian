---
source_pdf: PointWorld.pdf
paper_sha256: 368ee87a0aeb2b784026d4eb0bd47e38937759473a2762f0ac338f9ad06a4b08
processed_at: '2026-08-06T05:15:02-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PointWorld 用人话说

## 一句话版本

让 robot 学会"想象"——给它看一张 RGB-D 图片，告诉它"我要这么动"，它能脑补出来接下来 1 秒钟这个 3D 世界会怎么变。

## 为什么这件事难

你看到桌上有个杯子，我让你闭上眼睛推它一下，你能猜出来它会往哪滚、会不会翻、碰到别的东西会怎样。这个能力对人来说 trivial，对 robot 来说极其困难。

传统做法有两派：

**一派是 physics engine**（MuJoCo, Isaac Sim 这种）。你给它建个 model，告诉它物体的 mass、friction、shape，它算给你看。问题是现实世界太 messy 了，你不可能给每个 random 场景都建模，sim-to-real gap 也很烦。

**一派是 video prediction**（Sora, Cosmos 这种）。给它大量视频，让它学 "看图预测未来"。问题是这些 model 预测的是 pixel，没有 explicit 的 action，physics consistency 也不靠谱——你让它推一个杯子，它可能给你生成一个 "看起来像在推杯子" 的视频，但杯子的运动轨迹在物理上是错的。

PointWorld 走了第三条路。

## PointWorld 的核心 idea

**不要预测 pixel，预测 3D 点的移动。**

你想啊，robotic manipulation 真正关心的就是 geometry：gripper 抓到哪了、物体被推到哪去了、接触点在哪。Appearance（颜色、纹理、光照）对 manipulation 来说基本不重要。那就别预测 appearance 了，只预测 3D geometry 怎么变。

具体来说：
- Input：一张 RGB-D 图（变成 3D 点云）+ robot 接下来要做什么动作
- Output：每个 3D 点在未来 10 步（1 秒）会移动到哪里

这个 formulation 特别 clean。state 是 3D 点，action 也是 3D 点，model 就是学一个 "3D 点怎么动" 的函数。

## 最聪明的一个 trick：action 也用 3D 点表示

这是 paper 里我觉得最 elegant 的地方。

不同 robot 的 action space 是不一样的：Franka 是 7 个 joint angle，humanoid 可能是 20+ 个 joint，mobile manipulator 还有 base motion。你怎么用一个 model 处理所有这些 embodiment？

PointWorld 的解法：**把 robot 本身也变成 3D 点云**。

你有个 URDF（robot 的 blueprint），知道每个 joint 要怎么动。那就 sample 几百个点在 gripper 表面，用 forward kinematics 算出来这些点在未来 10 步每一步的 3D 位置。这就是你的 "action"——一堆 3D 点的轨迹。

这样一来：
- Franka 的 action 是 gripper 上几百个点的轨迹
- Humanoid 的 action 也是 gripper 上几百个点的轨迹
- 从 model 的角度看，action 就是 "有一坨点要这么动，scene 里的点会怎么 respond"

**Embodiment 差异被 absorb 进了 geometry 里**。这就像 LLM 把所有语言的语法差异 absorb 进了 token sequence 里一样——representation 统一了，你就可以 scale。

## 为什么只 sample gripper 的点

paper 做了个 ablation，发现用整个 robot body 的点反而不如只用 gripper 的点。

直觉上：大部分 robot 表面永远不接触 scene，给它们算 feature 是浪费的。而且 real-world data 本来就很 noisy，太多 robot 点会 "淹没" 本来就 sparse 的 scene learning signal。

Gripper-only 是个 sweet spot：足够表达 contact geometry，又不至于太 noisy。

这个 finding 有点反直觉——你会觉得更多 info 总是更好，但其实 signal-to-noise ratio 比 signal 总量更重要。

## 数据是怎么搞的

这是 paper 的大工程。

他们用 DROID（一个 large-scale real robot dataset），但 DROID 只有 RGB video，没有 3D annotation。要训 3D world model 需要 3D ground truth，怎么办？

他们搭了一个 pipeline：

1. **Depth**: 用 FoundationStereo（最近的 stereo matching model）从 stereo RGB 算 metric depth，比 sensor depth 准很多
2. **Camera pose**: DROID 给的 camera extrinsics 不准。他们用 VGGT 初始化，然后用一个很聪明的方法 refine——robot 的 URDF 是已知的，把 robot mesh 渲染出来和 observed depth 对齐，optimize camera pose 让它对齐
3. **Point tracking**: 用 CoTracker3 在 2D 上做 dense point tracking，然后 lift 到 3D

这个 pipeline 让他们从 DROID 里 recover 了约 60% episodes 的可靠 3D point flows，大约 200 小时。

再加 BEHAVIOR-1K 的 300 小时 simulation data（这个有 ground truth 3D，好处理），总共约 500 小时，2M trajectories。

这是目前最大的 3D dynamics dataset。

## 训练的几个关键设计

**Backbone**: 用 PTv3（Point Transformer V3），从 50M scale 到 1B 参数。PTv3 的好处是 memory efficient，1B 模型 inference 只要 0.12 秒，能做 real-time MPC。

**Loss**: 这个有讲究。naïve L2 loss 不行，因为 scene 里只有 1-5% 的点真的在动，剩下都是 static，loss 被 static 点主导。他们加了：
- **Movement weighting**: 根据真实位移给每个点加权，动的点多给 loss
- **Aleatoric uncertainty**: 模型额外 predict 一个 per-point uncertainty，noisy 的点自动 down-weight。这个超 cool——模型自己学会哪些点的 ground truth 不可靠
- **Huber loss**: 对 outlier robust

**DINOv3 features**: frozen DINOv3 给 scene points 打 feature。DINOv3 学到的 dense feature 有 strong objectness prior，相当于免费送你一个 "这大概是个物体" 的 segmentation hint，不用 explicit segment。

**Chunked prediction**: 一次 forward pass 预测 10 步，比 autoregressive（一步一步预测）效果好且快。原因：training 和 inference 一致，没有 distribution shift。

## Scaling law 成立

这是 paper 最 exciting 的 result 之一。

他们试了 50M 到 1B 参数，5% 到 100% data，发现 error 随 data 和 model size 都是 log-linear 下降。

这意味着什么？意味着 3D world model 和 LLM 一样服从 scaling law。你现在投入更多 compute 和 data 会持续变好，没有 saturation。这条路可以一直走下去。

## Real-world deployment

最 impressive 的 demo：一个 pre-trained checkpoint，zero-shot 部署到 real Franka robot 上，能做：
- 推 rigid 物体（tissue box, book）
- 折 deformable（scarf, pillow）
- 开 articulated（microwave, drawer）
- 用 tool（duster, broom）

**不需要任何 demonstration 或 finetuning**。就一个 model，一张 RGB-D 图，MPPI planner 搜 action。

而且他们用的 gripper 是 3D-printed fin ray，训练数据里从来没见过这个 gripper geometry——model 能 generalize 到 unseen gripper shape。

## 这个工作在整个 field 里的位置

我觉得 PointWorld 标志着 robot world modeling 进入了一个新阶段。

之前 robot dynamics model 基本上是两类：
1. Physics simulator（精确但需要建模，domain specific）
2. Small learned model（如 Yunzhu Li 的 particle dynamics，在 controlled lab 环境工作）

PointWorld 是第一个把 learned 3D dynamics model scale 到 in-the-wild real-world deployment 的工作。它证明了几件事：

1. **3D world model 可以 scale**——scaling law 成立
2. **Zero-shot real deployment 可行**——不需要 per-task training
3. **Cross-embodiment 可以统一**——point flow representation 是个好的 abstraction layer
4. **Data engineering 是关键 enabler**——最新 3D vision models 让 large-scale 3D annotation 成为可能

这和 GPT-3 之于 NLP 的角色很像。Idea 不全是新的（particle dynamics, action chunking, MPC 都有 prior work），但 scale + engineering 把它 push 到了 new effectiveness level。

## 和 video world model 的关系

Sora、Cosmos 那些是 appearance-first，PointWorld 是 geometry-first。对 manipulation 来说 geometry 更重要——你 push 一个杯子，不关心它的 texture 变没变，只关心它移到哪了。

但 long-term 这两个会 merge。你想要一个既能预测 geometry 又能预测 appearance 的 world model。PointWorld 解决了 geometry 部分，appearance 部分可以接 Gaussian Splatting 或 NeRF 之类。

我猜 future direction 就是 PointWorld + appearance model 的 hybrid。

## 我觉得最 cool 的几个点

1. **Aleatoric uncertainty 的 emergent behavior**：模型没见过 uncertainty ground truth，但自发学会在布的边缘 predict 高 uncertainty——因为布边缘的运动物理上就更 variable。这说明 objective 设计对了，model 真的在学物理。

2. **Robot-depth alignment 做 camera calibration**：利用 robot 是 known rigid body 这个事实，把 rendered robot mesh 和 observed depth 对齐来 refine camera pose。这招太 elegant 了。

3. **Distance-to-robot feature**：给每个 scene point 一个 "robot 离你多远" 的 hint，帮 model attention 到 relevant region。简单但有效。

4. **Real + sim joint training 的 noise 问题**：sim data 太干净会让 uncertainty head collapse，他们用 batchwise constant variance 来 fix。这种 subtle issue 只有真正做 large-scale real+sim training 才会遇到。

## 几个 open question

1. **Long horizon**: 目前 1 秒 prediction，长程任务怎么办？Replanning 可能 work 但需要验证。
2. **Cost function specification**: 现在手动画 mask + 指定 target，怎么 automate？VLM 是个方向（参考 ReKep）。
3. **Fine objects**: pen、cable 这种细物体 3D annotation 还是很难。
4. **Causality**: model 学的是 correlation，exogenous factor（别人动了一下环境）会 confuse 它。
5. **Deformable robot**: fin ray 这种 soft gripper 本身会变形，rigid body assumption 会 break。

## Big picture

PointWorld 让我对 robot learning 的 future 更乐观了。

之前大家觉得 manipulation 需要针对每个 task 收集 demo、训练 policy。PointWorld 展示了另一种 path：训一个大 world model，用 MPC 在 inference 时搜 action。这更像 "model-based control" 的路线，但 model 是 learned 的、large-scale 的、generalizable 的。

如果这条路 scale 下去，也许未来 robot 就是：一个大 world model + 一个 planner，不需要 per-task policy。World model 本身就是 "skill library"——它知道物体怎么动、contact 怎么 propagate、articulation 怎么 work，planner 只需要指定 goal。

这和 LeCun 的 JEPA 哲学也呼应：predict in representation space 而非 pixel space。3D point flow 就是一个 "physics-grounded representation"，比 pixel sample-efficient，比 abstract latent interpretable。

总之，PointWorld 是一个 milestone，告诉我们 3D world model 可以 scale、可以 zero-shot deploy、可以 unify embodiment。接下来的故事应该会更精彩。

参考链接：
- https://point-world.github.io
- https://droid-dataset.github.io  
- https://behavior.stanford.edu/
- https://arxiv.org/abs/2312.10035 (PTv3)
- https://arxiv.org/abs/2508.10104 (DINOv3)
- https://co-tracker3.github.io
- https://rekep-robot.github.io
- https://arxiv.org/abs/2503.17973 (PhysTwin, hybrid 方向)

---

# PointWorld: 3D World Model for Robotic Manipulation 深度讲解

## 1. 核心思想与设计哲学

PointWorld 的核心 insight 可以浓缩为一句话：**unification for scaling**。把 state 和 action 统一到同一个 modality——3D point flows。这和 LLM 的 "next-token prediction" 哲学类似：一个统一的、容易 scale 的 objective，让模型从海量 heterogeneous 数据中学习一个 "single source of truth of the physical world"。

这种 formulation 的关键好处：
- **Embodiment-agnostic**：不同 robot（Franka single-arm, bimanual humanoid, mobile manipulator）的 action 都被转成 3D point flows，通过 geometry of interaction 统一起来，不需要为每种 robot 设计专门的 action space
- **Modality alignment**：state 和 action 在同一个 3D space 里，模型只需要学一个 mapping $\mathcal{F}_\theta: \mathbf{S} \times \mathbf{A} \to \mathbf{S}$，inductive bias 干净
- **Implicit objectness/articulation/material**：不需要 explicit segmentation 或 material specification，所有这些 property 都通过 interaction geometry 隐式 emerge

paper 在 Figure 3 中画了一个很漂亮的图：从 RGB-D + robot point flows，模型需要 implicitly 完成 object segmentation, material property estimation, shape completion, contact reasoning, object-object dynamics propagation, gravity reasoning——全部在一个 forward pass 里。这就像 LLM 通过 next-token prediction 隐式学会了 syntax, semantics, world knowledge 一样。

项目主页：https://point-world.github.io

## 2. State-Action Representation 的设计

### 2.1 State: Scene Point Flows

State $\mathbf{s}_t = \{(\mathbf{p}_{t,i}, \mathbf{f}_i^S)\}_{i=1}^{N_S}$ 包含 $N_S$ 个点，每个点有：
- 位置 $\mathbf{p}_{t,i} \in \mathbb{R}^3$（3D coordinates）
- 时间常量 feature $\mathbf{f}_i^S \in \mathbb{R}^{D_S}$（dimension $D_S$）

从 RGB-D 获得 state 的流程：
1. 用 forward kinematics 和 URDF 算出 robot pixels 并 mask 掉
2. 把剩余 pixels back-project 到 3D 得到 $\mathbf{p}_{t,i}$

关键 design choice：**correspondence 只在 model forward pass 内保持**（"imagination" 内）。这意味着不需要外部 point tracker，每个 forward pass 的 point count 都可以变。这是一个很优雅的设计：state 是一个 "static snapshot"，dynamics 完全由 model 内部预测。

### 2.2 Action: Robot Point Flows

这是 paper 最核心的创新之一。给定 joint configuration sequence $\{\mathbf{q}_{t+k}\}_{k=0}^H$：

1. 在 time $t$ 时 sample robot surface points **一次**
2. 每个 point attach 到对应的 link
3. 用 forward kinematics 传播到每个 timestep $t+k$，得到有序的 robot points $\{(\mathbf{r}_{t+k,j}, \mathbf{f}_{t+k,j}^R)\}_{j=1}^{N_R}$

其中 $\mathbf{r}_{t+k,j} \in \mathbb{R}^3$ 是点 $j$ 在 time $t+k$ 的位置，$\mathbf{f}_{t+k,j}^R \in \mathbb{R}^{D_R}$ 是 time-varying feature。

**为什么是 fully observable robot points？** 这是 paper 一个非常聪明的设计：
- 在 occluded regions（比如 egocentric view 抱一个大箱子），partial observable 的 action 会让模型 confused
- Robot URDF 是 a priori known 的，所以可以 "imagine" 完整的 robot geometry
- 这样 action 在 representation 层面就是 fully observable 的，只有 scene 是 partial observable

**实践中只 sample gripper 的 points**（每个 gripper 300-500 个点），因为大部分 robot surface points 永远不接触 scene。这是 efficiency 和 effectiveness 的 sweet spot，Section 5.2 的 ablation 专门验证了这一点。

### 2.3 Per-Point Input Features (Table 4 详解)

Robot features stack：
$$\phi_{t,j}^{\text{robot}} = [p_{t,j}^{\text{robot}}, c_j^{\text{robot}}, n_{t,j}^{\text{robot}}, \tilde{g}_t, v_{t,j}^{\text{robot}}, a_{t,j}^{\text{robot}}]$$

- $p_{t,j}^{\text{robot}}$: 3D position（随时间变化）
- $c_j^{\text{robot}}$: 固定 magenta color (1,0,1)，标识 robot identity
- $n_{t,j}^{\text{robot}}$: 从 URDF 算出的 surface normal
- $\tilde{g}_t$: 标量 gripper openness，broadcast 到所有 robot points
- $v_{t,j}^{\text{robot}}, a_{t,j}^{\text{robot}}$: 用 mid-point finite difference 算出的 velocity 和 acceleration，在 horizon 边界用 zero-velocity 假设

Scene features 只在 $t=0$ 算：
$$\phi_i^{\text{scene}} = [x_{0,i}, c_{0,i}^{\text{scene}}, n_{0,i}^{\text{scene}}, g_{0:T-1}, d_{0:T-1,i}]$$

- $x_{0,i}$: 第一帧的 3D 坐标
- $c_{0,i}^{\text{scene}}, n_{0,i}^{\text{scene}}$: 第一帧的 RGB color 和 estimated normal
- $g_{0:T-1} \in \mathbb{R}^T$: 整个 horizon 的 gripper openness sequence，broadcast 到每个 scene point
- $d_{0:T-1,i} \in \mathbb{R}^T$: 每个 timestep scene point $i$ 到最近 robot point 的距离——这是一个 explicit 的 "interaction field" hint

那个 distance field $d_{t,i} = \min_j \|x_{0,i} - r_{t,j}\|_2$ 很有意思，它给 model 一个 explicit 的 "robot 要过来接触这个 point 了" 的 prior signal，帮助模型 attention 到 relevant regions。

## 3. Architecture 详解

### 3.1 整体数据流（Figure 2）

```
RGB-D + Robot URDF + Joint Actions
        ↓
   Scene Point Cloud (mask robot pixels, backproject)
   Robot Point Flows (forward kinematics from URDF)
        ↓
   Concatenate scene + time-stacked robot points
        ↓
   Scene points: frozen DINOv3 features (multi-layer)
   Robot points: temporal embeddings
        ↓
   PTv3 backbone (PointTransformerV3)
        ↓
   Shared MLP head → per-point displacements for H steps
```

### 3.2 PTv3 Backbone 为什么 work

paper Table 1 的 backbone comparison 是一个关键 ablation：

| Backbone | Params | Mem | FLOPs | Latency (ms) | $\ell_2$ mover |
|----------|--------|-----|-------|--------------|---------------|
| GBND (baseline) | 1.00x | 1.00x | 1.00x | 13.46 | 0.0390 |
| PointNet | 1.03x | 0.34x | 0.04x | 5.93 | 0.0369 |
| PointNet++ | 1.07x | 0.67x | 0.06x | 327.08 | 0.0368 |
| SparseConv | 33.31x | 7.18x | 1.32x | 17.70 | 0.0396 |
| Transformer | 41.06x | 0.31x | 3.38x | 30.43 | 0.0339 |
| PTv3-50M | 49.14x | 0.30x | 0.34x | 59.60 | 0.0331 |
| PTv3-132M | 127.22x | 0.69x | 1.04x | 69.60 | 0.0324 |
| PTv3-411M | 398.67x | 1.89x | 1.90x | 102.47 | 0.0315 |
| PTv3-1B | 957.71x | 4.30x | 3.57x | 123.65 | 0.0312 |

GBND（Graph-Based Neural Dynamics）有两个 fundamental 问题：
1. **Memory scaling**：为 scene 中所有 points 维持 high-dim features 太贵
2. **Partial observability**：纯 local message passing，long-range effects 要穿过 noisy hops

PTv3 的设计哲学：
- **Point serialization**：把 points 按 space-filling curve 排序，local grouping 效果类似 GBND 的 relational bias
- **U-Net hierarchy**：在 progressively coarser point sets 上做 attention，实现 long-range modeling
- **Massive parameter growth**：从 50M 到 1B，memory 只从 0.30x 涨到 4.30x，latency 从 59.6ms 涨到 123.6ms（仍然 real-time-ish）

Table 6 给出 1B 模型的具体 PTv3 config：
- Grid size: 1.5 cm
- Encoder depth: (4, 4, 8, 8, 12, 12, 4) —— 7 个 stage
- Encoder channels: (256, 384, 384, 512, 512, 768, 1024) —— 最后 stage 1024 channels
- Encoder heads: (8, 12, 12, 16, 16, 24, 32) —— multi-head attention
- Encoder stride: (1, 2, 2, 2, 2, 2, 2) —— 5 次 downsampling
- Decoder 是对称的 U-Net 结构

这个架构选择让 model 能 scale 到 1B 参数还能保持 0.12s latency，是 real-time MPC 的关键。

PTv3 paper: https://arxiv.org/abs/2312.10035

### 3.3 DINOv3 Scene Featurization

这是 paper 的一个关键 trick：用 frozen DINOv3 ViT-L/16 的 multi-layer features 给 scene points 打 feature。

具体做法：
1. 把 3D scene point $x_{0,i}$ 投影到每个 camera $c$：
$$\tilde{u}_{c,i} = K_c(R_c x_{0,i} + t_c)$$
其中 $K_c$ 是 intrinsics，$(R_c, t_c)$ 是 extrinsics，$\tilde{u}_{c,i}$ 是 homogeneous pixel coordinate

2. 归一化得到 pixel coordinate $u_{c,i}$
3. 在 DINOv3 patch-token grid 上 bilinear interpolation 得到 $f_{c,i} \in \mathbb{R}^{D_{\text{patch}}}$
4. 用 depth-consistency mask $m_{c,i}$ 过滤掉投影不准的 view
5. 跨 camera 聚合：
$$f_i = \frac{1}{\max(1, \sum_c m_{c,i})} \sum_c m_{c,i} f_{c,i}$$

然后 $f_i$ 通过 learned projection 映射到 backbone width (256)，和 raw scene features fusion 后送入 backbone。

**为什么这个 work？** DINOv3 学到的 dense features 包含 strong objectness prior，不需要 explicit segmentation。这和近期一些工作（如 FoundationStereo, VGGT 用 frozen 2D features 增强 3D models）的思路一致：**2D pre-trained features 是当前最好的 "implicit objectness" 来源**，比 3D self-supervised pre-training（如 Sonata）在 fine-grained 场景下还差一截。

DINOv3: https://arxiv.org/abs/2508.10104

## 4. Training Objective 深度解析

### 4.1 公式 (1) 详解

$$\frac{1}{2} \sum_{k,i}^{H,N_S} \underbrace{w_{k,i}}_{\text{movement}} \Big( \underbrace{\rho_\delta(\hat{\mathbf{P}}_{t+k,i} - \mathbf{P}_{t+k,i})}_{\text{Huber loss}} \underbrace{e^{-s_{k,i}}}_{\text{aleatoric}} + \underbrace{s_{k,i}}_{\text{uncertainty}} \Big)$$

逐项解释：
- $H$: prediction horizon（=10 steps）
- $N_S$: scene point 数量
- $k$: timestep index（0 到 H-1）
- $i$: point index（1 到 $N_S$）
- $w_{k,i}$: movement weight，归一化的 movement likelihood
- $\rho_\delta$: elementwise Huber loss，参数 $\delta$ 控制 quadratic vs linear 的切换点
- $\hat{\mathbf{P}}_{t+k,i}, \mathbf{P}_{t+k,i}$: 预测和 ground-truth 的 point $i$ 在 timestep $t+k$ 的 3D position
- $s_{k,i}$: 预测的 log-variance，$\sigma_{k,i}^2 = e^{s_{k,i}}$

### 4.2 Movement Weighting

$$m_{k,i} = \sigma(\kappa(\delta_{k,i} - \tau))$$
$$w_{k,i} = m_{k,i} / \sum_{k,i} m_{k,i}$$

- $\delta_{k,i} \geq 0$: ground-truth displacement vector 的 norm
- $\sigma$: logistic sigmoid
- $\tau$: displacement threshold（点的位移要多大才算 "moving"）
- $\kappa$: temperature（控制 sigmoid 的 sharpness）

**为什么需要这个？** 在 full-scene prediction 中，只有 1-5% 的 points 真的在 move。naïve $\ell_2$ loss 会被 static points 主导，training signal 极其 sparse。Movement weighting 把 loss 集中到 moving points 上。

但单纯 movement weighting 会 overemphasize noisy signals（一些点 "看起来 moving" 是因为 depth noise），所以需要 uncertainty regularization 来 temper 这些 weights。

### 4.3 Aleatoric Uncertainty Regularization

这个 idea 来自 Kendall & Gal (2017) 的 Bayesian deep learning work。模型 predict 一个 per-point log-variance $s_{k,i}$，loss 里有两项：
- $e^{-s_{k,i}} \cdot \text{residual}$：residual 被 uncertainty down-weight
- $s_{k,i}$：防止 model 把所有点的 uncertainty 都推到无穷大

这是一个自动的 robustness mechanism：对于 noisy 点，模型 predict 高 uncertainty（大 $s_{k,i}$），residual 被压低；对于 reliable 点，模型 predict 低 uncertainty，residual 正常贡献 gradient。

paper Figure 4 展示了一个很 intuitive 的例子：robot release 一块黄布，uncertainty 在布的边缘最大——因为布边缘的 motion 有更大的物理 variability。这是 model **没有 ground-truth uncertainty** 的情况下自发学到的，说明它真的 capture 了 physical property 的 uncertainty。

### 4.4 Huber Loss

$$\rho_\delta(x) = \begin{cases} \frac{1}{2}x^2 & |x| \leq \delta \\ \delta(|x| - \frac{1}{2}\delta) & |x| > \delta \end{cases}$$

paper 用 $\delta = 5.0$。Huber loss 对 outlier 更 robust（线性而非二次惩罚大 residual），这对 real-world noisy data 很重要。

### 4.5 Simulation Data 上的特殊处理

这是一个很 subtle 但关键的 trick。在 simulation 上，residual 趋于 0，会导致 $s_{k,i} \to \log \rho_\delta(\cdot) \to -\infty$，于是 $e^{-s_{k,i}} \to \infty$，一点点 numerical discrepancy 就产生巨大 gradient，destabilize joint training。

paper 的解法：在 simulation domain 上，把 log-variance 替换成一个 batchwise constant，匹配 real data 上的 average variance。这样 heteroscedastic weighting 只在 real noisy data 上有效，不会 collapse。

这种细节体现了 real+sim joint training 的 difficulty：两个 domain 的 noise level 不同，naïve joint training 会让模型 exploit 干净 domain 的 gradient。

参考：Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NeurIPS 2017, https://arxiv.org/abs/1703.04977

## 5. Dataset Curation Pipeline

这是 paper 的另一个核心 contribution。要 scale 3D world model，需要 large-scale, high-quality 3D dynamics data。Existing dataset（DROID）虽然有 diverse real-world interaction，但没有 3D annotation。

### 5.1 DROID 3D Annotation Pipeline

DROID: https://droid-dataset.github.io

paper 用一个三阶段 pipeline 从 DROID 的 raw RGB video 恢复 3D point flows：

**Stage 1: Depth Estimation with FoundationStereo**

不用 DROID 的 sensor depth（在 open-world 环境 noise 很大），改用 FoundationStereo 算 stereo depth。FoundationStereo 在 manipulation 典型的 close working distance 上特别 effective。

但 FoundationStereo 也不是完美的：远距离 texture-less 区域（如墙壁）depth 不准。所以 clamp 到 [0, 4]m 并 produce per-pixel validity mask。

FoundationStereo: https://arxiv.org/abs/2505.05415 (Wen et al., CVPR 2025)

**Stage 2: Camera Pose Estimation**

这是最 tricky 的部分。DROID 提供的 extrinsics 不准（imperfect calibration），VGGT 直接估的 pose 又 deviate 几十厘米。

paper 的解法是两阶段：
1. **VGGT initialization**: 多视角 pose estimator 给一个 initial estimate $T_{E_0 B}$
2. **Robot-depth refinement**: 利用 robot 的 known geometry（URDF + joint states），optimize camera extrinsics 让 rendered robot mesh 和 observed depth 对齐

具体 optimization 的 loss：
$$L_{\text{robot-depth}} = \frac{1}{K} \sum_{i,t,k} |d_{i,t,k}^{\text{obs}} - d_{i,t,k}^{\text{pred}}|$$

- $i$: camera index
- $t$: timestep
- $k$: valid robot pixel index（过滤后）
- $d^{\text{obs}}$: observed depth
- $d^{\text{pred}}$: predicted depth（把 robot surface 点从 base frame 投影到 camera 得到）

每个 camera optimize 一个 6-DoF update，用 first-order optimizer，100 iterations，lr $10^{-3}$。

这个 idea 很 elegant：robot 是 known rigid body，它的 depth 应该和 rendered URDF 完美对齐。如果不对齐，就是 extrinsics 错了。这是一个 self-supervised calibration 的思路，利用 known geometry 做 anchor。

**Stage 3: 2D Point Tracking with CoTracker3**

有了 depth 和 camera pose 后，用 CoTracker3 在 workspace + non-robot mask 上做 dense 2D tracking，得到 2D trajectories 和 visibility map。然后 back-project 到 3D 得到 3D point flows。

CoTracker3: https://arxiv.org/abs/2410.11825

**Postprocessing**:
- DBSCAN outlier removal（multi-scale, $\epsilon \in \{0.02, 0.05\}$m）
- Per-frame normal estimation + temporal consistency

最终 pipeline 能 recover 60% DROID episodes 的 reliable 3D point flows（近 200 小时）。Quality 上 dominate DROID 原始 extrinsics 和纯 VGGT 方案（Figure 5）。

VGGT: https://arxiv.org/abs/2503.11651

### 5.2 BEHAVIOR-1K Simulation Data

BEHAVIOR-1K: https://arxiv.org/abs/2403.09227

B1K 提供 photorealistic home-scale simulation，1100 小时 bimanual humanoid teleoperation。paper 的处理：

1. **Replay**: 用 recorded state + action 重放 episode，每步 attach 3 个 virtual camera（left/right shoulder + head）
2. **Clip filtering**: 切 11-frame clips，用一组 logical condition（公式 3）筛选有意义的 interaction
3. **3D point flows from sim**: 利用 rigid link decomposition + per-link trajectory，efficiently reconstruct per-point 3D trajectories

公式 3 的 clip acceptance criterion：
$$\neg C_t \wedge \big( (M_o \wedge M_j) \vee (M_o \wedge C_f) \vee (\neg M_o \wedge M_g \wedge M_j) \big)$$

- $C_t$: trunk/arm collision（discard 这种）
- $M_o$: object motion
- $M_j$: non-base joint motion
- $M_g$: gripper motion/state change
- $C_f$: gripper finger contact

三个 disjunct 分别对应：
1. Object motion + joint motion（正常 manipulation）
2. Object motion + finger contact（finger 接触导致的 object motion）
3. No object motion + gripper + joint motion（"negative" clips，监督 background dynamics）

这个 filtering 逻辑很精细，目的是去掉无意义的 static clips 同时保留 diverse interaction supervision。

### 5.3 数据规模

Total: ~2M trajectories, ~500 hours
- DROID: ~200 hours real-world single-arm
- B1K: ~300 hours simulated bimanual/whole-body/mobile

这是目前最大的 3D dynamics modeling dataset，paper 全部 open-source。

## 6. Scaling Roadmap（Section 5.1）

paper Figure 7 展示了一个逐步改进的 roadmap，从 baseline [5] 的 $\ell_2$ mover 0.0390 到 final 0.0312：

1. **Modern backbone (PTv3)**: 0.0390 → 0.0331（50M）→ 0.0312（1B）
2. **Stabilized training objective**: movement weighting + uncertainty + Huber
3. **Pre-trained 2D features (DINOv3)**: substantial boost
4. **Model scaling**: 50M → 1B，log-linear gains

**Scaling Law（Figure 9）**：在 data 和 model size 两个 axis 上，$\ell_2$ error 大致 log-linear 下降。这和 LLM/Vision 的 scaling law 一致（Kaplan et al. 2020, Chinchilla, Flamingo）。

这个 result 很 important：它说明 3D world modeling 也服从 scaling law，意味着投入更多 data 和 compute 会持续带来 gain，没有 saturation 的迹象。

参考：
- Kaplan et al., "Scaling Laws for Neural Language Models", https://arxiv.org/abs/2001.08361
- Hoffmann et al. (Chinchilla), "Training Compute-Optimal Large Language Models", https://arxiv.org/abs/2203.15556

## 7. Chunked Prediction（Section 5.2 Ablation）

paper 用 chunked prediction：一次 forward pass 预测 H=10 步（1 秒）。对比两个 baseline：
- Teacher-forcing（每步用 GT 输入）
- Self-feeding（10k warmup 后 autoregressive）

还对比 sliding-window inference（W=1, 5）用同一个 chunked model。

Result（Figure 12）：**Training 和 inference 都用 chunked prediction 最好**。W=1 推理（相当于 self-feeding）degradation 最严重；W=5 部分恢复但超出训练 window 后 degrade。

这个 result 说明：
1. Training-inference consistency 很重要（类似 RL 中的 train-test distribution match）
2. Chunked prediction 在 compute 上更高效（1 次 forward vs 2-10 次 autoregressive）
3. 减少 rollout drift（autoregressive 的 error 会累积）

这和 ACT (Action Chunking with Transformers, Zhao et al. 2023) 的思路类似，但在 world model 而非 policy 上。

ACT: https://arxiv.org/abs/2304.13705

## 8. Partial Observability Ablation（Section 5.2）

paper 训练了 4 个 variant：1, 2, 3, 或 random up to 3 个 camera。然后在不同 test setting 下评估。

Key findings（Figure 13）：
- 更多 camera 训练 → test error 一致降低
- 固定 camera count 训练的 model 在更多 camera test 时反而更好（很 intuitive：更多 info = 更准）
- **Random-view model 最 robust**，在所有 test camera count 下都表现好

这个 result 说明 model 能学会 "infer objectness and physical properties under partial observability"。Exposure to varied observability during training 让 model 学会 robust reasoning，而非 overfit 到某个 view count。

这和 LLM 的 random masking 思路类似：通过 random degradation 让 model 学会 robust inference。

## 9. Generalization 实验（Section 5.3, Table 2）

| Setting | D→D | B→B | D→B | B→D | D→H | B→H | D+B→H | Scratch |
|---------|-----|-----|-----|-----|-----|-----|-------|---------|
| Zero-Shot mover | 0.0315 | 0.0087 | 0.1460 | 0.0558 | 0.0305 | 0.0531 | 0.0300 | 0.0293 |
| Finetuned mover | - | - | 0.0107 | 0.0378 | 0.0271 | 0.0299 | 0.0272 | 0.0293 |

- D = DROID, B = B1K, H = held-out real (CLVR lab)
- "From Scratch" = 在 held-out lab 数据上从头训练的 specialist

Key findings：
1. **In-domain**: sub-centimeter error on B1K held-out
2. **Cross-domain zero-shot**: 困难（D→B 0.1460），但 finetune 5% iterations 就能接近 from-scratch 20x updates
3. **Held-out real zero-shot**: D→H 0.0305 vs Scratch 0.0293，几乎 on-par！
4. **Finetune held-out**: 0.0271 < 0.0293，**surpass specialist with 20x fewer updates**
5. **D+B joint training**: zero-shot 比 D-only 稍好

这个 result 很 strong：一个 pre-trained model 在未见过的 real-world lab 上 zero-shot 就能达到 specialist 水平，finetune 少量就超过。这证明 PointWorld 学到的是 transferable interaction dynamics，而非 memorize。

## 10. Real-World Robot Experiments（Section 5.4, Figure 8）

部署 setup：
- Franka arm + 3D-printed fin ray gripper（注意：**fin ray gripper 是 unseen geometry**，训练数据用 Robotiq 2F-85 和 Galexea R1 Pro，这是 cross-gripper geometry generalization）
- Wheeled base（in-the-wild deployment）
- 1 个 RealSense D435
- FoundationStereo 估 depth
- MPPI planner

### 10.1 MPPI Integration

公式 (2) 的 trajectory optimization：
$$\arg\min \sum_{k=1}^T [c_{\text{task}}(\mathbf{s}_k) + c_{\text{ctrl}}(\mathbf{E}_k)]$$
$$\text{s.t. } \mathbf{s}_{1:T} = \mathcal{F}_\theta^T(\mathbf{s}_0, \mathbf{a}_{1:T}), \mathbf{E}_0 = \mathbf{E}_{\text{measured}}$$

- $T$: planning horizon（=30 steps = 3 autoregressive forward passes）
- $c_{\text{task}}$: task cost，把 task-relevant points 推到 target positions
  $$c_{\text{task}}(\mathbf{s}_k) = \frac{1}{|\mathcal{T}_{\text{task}}|} \sum_{i \in \mathcal{T}_{\text{task}}} \|\mathbf{p}_{k,i} - \mathbf{g}_i\|_2^2$$
  其中 $\mathcal{T}_{\text{task}}$ 是 task-relevant point set，$\mathbf{g}_i$ 是 target position
- $c_{\text{ctrl}}$: SE(3) path length + reachability regularization
- $\mathbf{E}_k$: end-effector pose at step $k$

MPPI 具体 config（Appendix A.6.1）：
- Cubic spline noise，$n_{\text{knots}}=4$, degree 3
- $\sigma_{\min}=0.05, \sigma_{\max}=0.50$
- 256 samples per iteration
- Temperature $\beta=0.05$
- EMA = 0.9
- 20 refinement iterations
- Planning time: 几秒

Task cost 是一个统一 interface：手动通过 GUI 用 SAM2 选 object mask + 指定 target positions。这个简单的 cost function 跨 rigid pushing, deformable, articulated, tool use 都 work，因为 model 本身 capture 了 interaction dynamics。

### 10.2 实验 Result（Figure 8）

Task + success rate：
- Rigid pushing（tissue box, book）
- Deformable（scarf fold, pillow place）
- Articulated（microwave open, drawer close）
- Tool use（duster sweep, broom sweep）

每个 task 10 个 random initial configuration。**所有任务都 zero-shot**，pre-trained checkpoint 直接部署，不需要 demo 或 post-training。

这证明 PointWorld capture 了 transferable interaction dynamics：
- Contact reasoning under partial observability（rigid pushing）
- Implicit articulation/deformation inference
- Object-object interaction（tool use）

SAM2: https://arxiv.org/abs/2408.00714
MPPI: Williams et al., "Model Predictive Path Integral Control: From Theory to Parallel Computation", https://arc.aiaa.org/doi/10.2514/1.G002670

## 11. Limitations 和 Future Work（Appendix A.1）

paper 很诚实地列了 limitations：

1. **Static initial state**: 假设 observation instant 是 static，没有 prior velocity。要 support dynamic initial conditions 需要 recurrent state 或 tracked trajectories。

2. **Reward/cost specification**: MPPI 需要显式 cost function，paper 用 manual GUI。未来可以用 VLM 自动指定（参考 ReKep, https://arxiv.org/abs/2409.01652）或从 demo 用 IRL infer。

3. **Fine-scale objects**: pen、cable 这种 thin objects 难以准确 annotate 3D，calibration noise 和 object thickness 相当。

4. **Correlation vs causation**: 模型学的是 training distribution 中的 correlation，不能 disentangle exogenous factors（其他 agent、环境变化）。这是所有 predictive world model 的 fundamental limitation。

5. **Lack of photometric dynamics**: 只输出 geometry displacement，不预测 appearance change（light on/off）。可以和 Gaussian Splatting 或 NeRF 结合。

6. **Rigid-body robot assumption**: URDF 假设 rigid links，ignore soft/tendon-driven/compliant 结构（如 fin ray gripper 本身会变形）。

7. **Actuation/tracking assumptions**: 把 robot trajectory 当作 known sequence，不 model 控制器 tracking error、contact-induced deviation。

8. **Lack of explicit physics priors**: 纯 data-driven，没有 Newtonian mechanics 或 conservation law。可以加 physics-informed regularization（参考 PhysTwin, https://arxiv.org/abs/2503.17973）。

这些 limitations 都指向 future work：recurrent state for dynamic init, VLM for reward, NeRF/Gaussian for appearance, deformable link modeling, closed-loop robot-scene co-modeling, physics priors。

## 12. 更广的 Connections

### 12.1 和 Video World Models 的对比

Cosmos (NVIDIA), Genie (Google), V-JEPA 2 (Meta) 等 video world model 在 photorealism 上很强，但：
- 缺 explicit action conditioning
- 物理 consistency 弱
- Inference 慢（diffusion 要几秒）

PointWorld 的 3D point flow representation 牺牲 appearance 换来：
- Explicit 3D action conditioning
- 物理一致性（geometry of interaction）
- Real-time inference（0.1s）

这是 geometry-first vs appearance-first 的 trade-off。对 manipulation 这种 contact-rich task，geometry 更 important。

Cosmos: https://arxiv.org/abs/2501.03575
V-JEPA 2: https://arxiv.org/abs/2506.09985

### 12.2 和 Particle-based Dynamics 的对比

Yunzhu Li 的 particle dynamics 系列（Learning particle dynamics, ParticleGrid, AdaptiGraph）是 PointWorld 的直接 predecessor。但：
- 多在 small-scale, controlled environment
- 需要 objectness prior 或 material specification
- 没 scale 到 in-the-wild

PointWorld 的 contribution 是把这些 idea scale up：用 modern 3D vision pipeline 标 real-world data，用 PTv3 + DINOv3 做 backbone，scale 到 1B 参数 + 2M trajectories。

Particle dynamics: https://arxiv.org/abs/1810.01566
ParticleGrid: https://arxiv.org/abs/2506.15680

### 12.3 和 Neuro-Symbolic / Hybrid World Models 的对比

一些工作（Neuro-Symbolic Concept Learner, Mao et al.）把 symbolic reasoning 和 neural perception 结合。PointWorld 走的是 pure neural scaling 路线，类似 GPT 之于 neuro-symbolic NLP。这是"unification for scaling"哲学的体现。

未来 hybrid 方向（如 PhysTwin 把 physics simulator 和 neural model 结合）可能解决 PointWorld 的 extrapolation 问题。

### 12.4 和 Action Representation 文献的对比

paper 对比了 4 种 action representation（Figure 11）：
1. Whole-body point flows, same #points（sparse coverage）
2. Whole-body point flows, 2000 points（similar density）
3. 6-DoF end-effector pose + gripper openness（low-dim）
4. Joint positions + gripper openness（low-dim）

Result：**Gripper-only flows（300-500 points）最好**。

Why？在 B1K 上，spatial contact representation 优于 low-dim。在 DROID 上，whole-body flows 反而 underperform low-dim，因为 extensive robot points obscure sparse learning signal from noisy real data。Gripper-only 是 sweet spot：spatial contact reasoning + 不被 noisy points 主导。

这个 result 很重要：它说 action representation 不是越 expressive 越好，要平衡 expressiveness 和 signal density。类似 LLM 中 tokenization 的 trade-off。

### 12.5 和 Recent Embodied World Models 的对比

近期一些工作探索类似 idea：
- Tesseract (https://arxiv.org/abs/2504.20995): 4D embodied world model
- Dream2Flow: bridging video generation 和 manipulation
- ParticleFormer: multi-object multi-material manipulation

PointWorld 的独特之处：
- Scale（1B params, 2M trajectories）
- Real+sim joint training
- Zero-shot real-world deployment
- Open-source data + model

## 13. 我对 PointWorld 的整体评价

### 13.1 Strengths

1. **Clean formulation**: state-action unified as 3D point flows，philosophically elegant
2. **Scaling works**: log-linear gains in data 和 model size
3. **Zero-shot deployment**: 在真实 robot 上 work，不需要 finetune
4. **Data engineering**: 3D annotation pipeline 是 tour de force，利用最新的 3D vision models
5. **Cross-embodiment**: single-arm 和 bimanium 都能 train

### 13.2 What I find most interesting

1. **Robot point flows as fully-observable action**: 这个 design choice 太聪明了。Robot URDF 是已知的，可以 "imagine" 完整 geometry，规避 partial observability。Scene 是 partial observable，action 是 fully observable，这个 asymmetry 很合理。

2. **Distance-to-robot feature**: $d_{0:T-1,i}$ 给 model 一个 explicit "interaction field" hint，帮助 attention 到 relevant regions。这是 simple but effective 的 inductive bias。

3. **Aleatoric uncertainty emergent behavior**: Figure 4 中布边缘的 uncertainty 更大，这是 model 自发学到的物理 variability，没有 ground-truth uncertainty supervision。这说明 objective design 对了。

4. **Chunked > autoregressive**: 在 world model 中 chunked prediction 比 autoregressive 好，这和 policy 中的 ACT 类似。这是一个 general principle：reduce distribution shift between train and test。

5. **Real + Sim joint training**: paper 揭示了一个 subtle 的 issue——sim data 的 noise-free 特性会让 uncertainty head collapse。用 batchwise constant variance 解法很 pragmatic。这种 real+sim 的 noise level mismatch 是 general problem。

### 13.3 Open Questions

1. **Long-horizon**: 当前 H=10 (1s)，long-horizon planning 怎么办？Replanning 可能 work，但需要实验验证。
2. **Multi-step tasks**: 当前 task 都是 single-step（push, fold, open）。Long-horizon task（如做三明治）如何 specify cost？
3. **Closed-loop**: 当前是 open-loop planning（30 steps 一次 plan），closed-loop replanning 效果如何？
4. **Generalization to new object categories**: 未见过的 articulated object（如 scissors）能 generalize 吗？
5. **Sample efficiency**: 2M trajectories 是很多数据。能不能用更少 data + 更多 structure（physics priors）达到类似效果？

### 13.4 Big Picture

PointWorld 代表了 robotic world modeling 的一个重要 milestone：
- 证明 3D world model 可以 scale
- 证明 zero-shot real-world deployment 可行
- 提供 open-source data + model 给社区

它和 video world model（Cosmos, V-JEPA 2）走的是不同路线：geometry-first vs appearance-first。对 manipulation，geometry 可能更重要。但 long-term，两个路线应该 merge：一个既有 geometry consistency 又有 photorealism 的 world model。

这让人想起 LeCun 的 JEPA philosophy：predict in latent space 而非 pixel space。PointWorld 的 3D point flow space 可以看作一个 "physics-grounded latent space"，比 pixel space 更 sample-efficient，比 abstract latent space 更 interpretable。

JEPA: https://arxiv.org/abs/2301.08243

## 14. Implementation Details 我觉得值得注意

### 14.1 Data Preprocessing（Table 3）

- Voxel downsampling: 1.5 cm
- Max scene points: 12,000
- Max robot points: 500
- Random yaw, uniform scaling [0.9, 1.1], reflection
- Chromatic auto-contrast, translation, jitter

这些 augmentation 是 standard point cloud training recipe，但 scale 到 1B 模型时细节很重要。

### 14.2 Training Config（Table 5）

- AdamW, lr $1 \times 10^{-4}$
- 300 epochs
- Global batch size 1920 sequences
- Gradient clipping global $\ell_2$ norm 5
- 128 H100 GPUs
- 20 days training

1B 模型在 128 H100 上训练 20 天，这是一个相当大的 compute investment。Batch size 1920 很大，说明 data pipeline 要很 efficient 才能 feed 这个 batch size。

### 14.3 Real-Robot Control Details

- 6-DoF end-effector pose prediction
- Position control at 20 Hz
- 每个目标 pose linearly interpolated（5mm translation, 1° rotation 步长）
- PyBullet IK
- Deoxys joint-impedance controller

这个 control stack 是 standard manipulation setup，没有特殊硬件。重要的是 model 本身 zero-shot work。

## 15. 总结

PointWorld 是一个重要的工作，它把 3D world modeling scale 到了前所未有的程度，并证明了 zero-shot real-world manipulation 的可行性。核心 insight 是 "unification for scaling"——把 state 和 action 统一到 3D point flows，然后像 LLM 一样 scale data 和 model size。

Key takeaways for building intuition：
1. **Representation matters more than architecture**: 3D point flows 这个 representation 选择 enable 了 cross-embodiment training 和 stable regression objective
2. **Data quality is bottleneck**: 3D annotation pipeline 利用 FoundationStereo + VGGT + CoTracker3 是关键 enabler
3. **Real+sim joint training 有 subtle issues**: noise level mismatch 需要 careful handling
4. **Chunked > autoregressive for world models**: 减少 train-test distribution shift
5. **Scaling laws hold in 3D world modeling**: log-linear gains in data 和 model size

这个工作让我想起 GPT-3 之于 NLP：不是 idea 全新，而是 scale + engineering 把已有 idea push 到新的 effectiveness 水平。下一个 GPT-4 级别的 world model 可能是 PointWorld + appearance generation + long-horizon reasoning 的结合。

更多参考：
- Project page: https://point-world.github.io
- DROID dataset: https://droid-dataset.github.io
- BEHAVIOR-1K: https://behavior.stanford.edu/
- PTv3: https://arxiv.org/abs/2312.10035
- DINOv3: https://arxiv.org/abs/2508.10104
- FoundationStereo: https://research.nvidia.com/publication/2025_FoundationStereo
- VGGT: https://vgg-t.github.io
- CoTracker3: https://co-tracker3.github.io
- MPPI: https://arc.aiaa.org/doi/10.2514/1.G002670
- ReKep: https://rekep-robot.github.io
- PhysTwin: https://arxiv.org/abs/2503.17973
