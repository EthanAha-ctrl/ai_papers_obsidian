---
source_pdf: Aim My Robot.pdf
paper_sha256: fc7424d942356b6ea64e4a90a7282ed84b8923a1c2c7353f4bdfaa5f4beabad5
processed_at: '2026-07-18T07:01:57-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Aim My Robot (AMR) 深度技术解析

你好 Andrej，这篇paper触及了一个非常fundamental的问题：existing navigation research长期被一个宽松的success criterion（1m radius）所束缚，导致整个领域在"approach phase"的精度上几乎是空白。AMR填补了从coarse navigation到precision manipulation之间这段关键的"last 10 meters"问题。下面我从intuition出发，把每个技术决策拆解清楚。

---

## 1. Problem Positioning: 为什么这是一个被忽视的important问题

Object-goal navigation（[ObjectNav](https://arxiv.org/abs/2006.11453)）和instance-goal navigation（[InstanceImageNav](https://arxiv.org/abs/2211.15876), [Goat](https://arxiv.org/abs/2311.06430)）普遍采用1m success radius，这个convention继承自[PointNav](https://arxiv.org/abs/1806.02764)。但是真实的downstream task——forklift插入pallet、机械臂抓取drawer handle、机器人dock到charging station——通常需要<5cm的base positioning精度，否则manipulation policy会因workspace超出reachable范围而失败。这一点在[OKRobot](https://arxiv.org/abs/2401.12202)中有明确量化：base pose error每增加几厘米，grasp success rate急剧下降。

AMR将问题重新formulate为**local, object-centric, high-precision maneuver**：robot已经在object的vicinity（<10m，约一个room size），需要reach一个特定的SE(2) relative pose。这相当于把navigation和visual servoing之间的鸿沟用一个unified learning system填上。

---

## 2. Goal Specification: 为什么用 reference image + mask + (S, d, θ)

### 2.1 Formulation

$$G = \{I_R, M, \mathbf{C}\}, \quad \mathbf{C} = \{S, d, \theta\}$$

变量说明：
- $I_R$：reference image（可以是robot memory中的历史frame，也可以是任意视角的scene image）
- $M$：object mask（二值或soft mask，标注$I_R$中的target object）
- $S \in \{\text{front, back, left, right}\}$：approach side，由$I_R$中object most visible的side定义"front"，剩下三个side按几何关系派生
- $d \in [0.0\text{m}, 1.0\text{m}]$：approach distance（robot base到object的最终距离）
- $\theta \in \{0^\circ, \pm 15^\circ, \pm 30^\circ\}$：approach angle（robot heading相对object front的偏角）

### 2.2 Intuition

这个parametrization有几个精妙之处：

**(a) 避免CAD model依赖**。传统的precision navigation（如[FoundationPose](https://arxiv.org/abs/2312.08344) + planning）需要object的3D model，并要求object在initial observation中visible。这在open-world setting中几乎不可行——robot遇到的多数object没有available CAD。

**(b) 避免close-up reference image**。[InstanceImageNav](https://arxiv.org/abs/2211.15876)通常要求提供target的近景图作为goal。AMR允许reference image来自任意视角，甚至object在reference中部分遮挡也能work（见Fig.9 fridge实验）。

**(c) 与high-level planner天然接口**。SAM（[Segment Anything](https://arxiv.org/abs/2304.02643)）可以提供mask，LLM（[Code as Policies](https://arxiv.org/abs/2302.01601)）可以输出$(S, d, \theta)$。这样AMR就成为一个"execution primitive"，可以被hierarchical system调用。

**(d) 离散化θ的设计**。θ只取$\{0^\circ, \pm15^\circ, \pm30^\circ\}$五个值，这是precision和flexibility之间的折衷。太细的离散会让training data稀疏；太粗又无法表达"docking at 30° to read a side gauge"这样的task。

---

## 3. Data Pipeline: 仿真生成的photorealistic demonstrations

### 3.1 Asset和Simulator选择

- **[HSSD-200](https://arxiv.org/abs/2306.11290)**：100+ room layouts，10,000+ objects，每个object都有PBR (Physics-Based Rendering) texture
- **[Isaac Sim](https://developer.nvidia.com/isaac-sim)**：NVIDIA的simulator，支持ray tracing rendering，配合PBR能产生photorealistic images（见Fig.3）
- 总共54个scene转成USD格式，49 train / 5 test
- 500k trajectories，7.5M frames

这里的关键intuition是：**sim2real的瓶颈不在rendering fidelity本身，而在asset diversity**。HSSD的优势是object和room组合的long-tail分布，让model学到的是"objectness"和"geometric reasoning"而非特定texture的memorization。

### 3.2 Trajectory Generation

robot model为cylindrical rigid body，radius $R \in [0.1, 0.5]$m（uniform sampled，用于让model aware of footprint）。规划器：

- **[AIT\*](https://arxiv.org/abs/2006.16685)** (Adaptively Informed Trees)：asymptotically optimal sampling-based planner
- **[Reeds-Shepp](https://en.wikipedia.org/wiki/Dubins_path#Reeds%E2%80%93Shepp_curve) state space**：允许backward motion的car-like曲线，turning radius=0表示differential-drive可以原地转向
- Cost function鼓励camera朝向goal，惩罚excessive backward motion（小量backward允许）

每条path按0.2m或5°的间隔render observation。Camera tilt angle的设置很巧妙：**让object mesh的最低vertex出现在image底部1/4处**。这样object在view中始终可见，即使robot离object较远或object在view外（object在frame外时tilt会假设一个虚拟object位置）。

### 3.3 为什么不用teleoperation

paper直接说"humans are poor at estimating distances and angles from images"——这确实是imitation learning中一个under-discussed的问题。人类teleop出来的trajectory在final pose上会有系统性偏差（humans tend to stop 20-30cm too far），用这种noisy supervision训练的model永远学不到cm-level precision。仿真中planner提供的near-optimal demonstrations才是precision的来源。

### 3.4 DAgger augmentation

训练后deploy回simulation，识别failure（collision / tracking loss / out of tolerance），用[DAgger](https://arxiv.org/abs/1011.0684)补充这些edge case。从Table II看，去掉DAgger collision rate从3.8%上升到5.5%——大约45%的relative increase，说明DAgger主要helps with long-tail。

---

## 4. Model Architecture: 多模态融合的细节

整个architecture分三stage：multi-modal sensor encoding → goal-aware fusion → motion generation。

### 4.1 RGB-D Encoding：Depth作为positional encoding

**这是我认为paper中最elegant的设计**。RGB image $I_t$ ($224 \times 224 \times 3$) 通过frozen [MAE-Base](https://arxiv.org/abs/2111.06377)得到$14 \times 14 \times 512$ feature map（196 tokens）。

depth image resize到$14 \times 14$，然后每个pixel的3D location通过back-projection计算：

$$[x', y', z']^T = \mathbf{R} \mathbf{K}^{-1} [x, y, d]^T + \mathbf{t}$$

变量说明：
- $[x, y]$：pixel coordinate in depth image
- $d$：depth value（该pixel的metric distance）
- $\mathbf{K}$：camera intrinsics（$3 \times 3$ upper triangular matrix）
- $\mathbf{K}^{-1}[x, y, d]^T$：把pixel homogeneous coordinate反投影到camera frame的3D ray，乘以$d$得到3D point
- $[\mathbf{R} | \mathbf{t}]$：camera extrinsics（$6$-DoF transform），把camera frame的point转换到robot base frame
- $[x', y', z']^T$：该depth pixel在robot egocentric coordinate frame下的3D location

然后对$x', y', z'$分别施加sinusoidal positional encoding $f$，concatenate得到depth patch的position embedding：

$$\text{PE}_{\text{depth}} = [f(x'); f(y'); f(z')]$$

加到对应的RGB patch feature上。

**Intuition**：传统做法是把RGB和depth各encode一遍然后concatenate（token数翻倍），或者用depth直接concat到RGB channel上。AMR的做法本质上是**用metric 3D coordinate作为每个visual token的"position tag"**——告诉transformer"这个patch在robot frame的$(x', y', z')$位置"。这有几点好处：

1. Token数量不变（仍是196），节省compute
2. 3D信息直接注入到attention的Q/K中，transformer可以学习"这个3D位置的物体和那个3D位置的物体之间的spatial relation"
3. History frame的depth可以通过修改$[\mathbf{R}|\mathbf{t}]$（设为$t-i$时刻camera到$t$时刻base的transform，从odometry得到）直接对齐到当前base frame——**这等价于显式做了geometric warping**

这一点让我想到[NeRF](https://arxiv.org/abs/2003.08934)中的positional encoding和[BEVFormer](https://arxiv.org/abs/2203.17270)中的spatial cross-attention——都是用3D coordinate作为query的positional information，让transformer学会geometric reasoning。

### 4.2 LiDAR Encoding

256个LiDAR point重采样，分成32个directional bin，每个bin内8个$(x, y)$ point过一个MLP得到1个token。共32个LiDAR tokens。

**Intuition**：这种radial binning保留了LiDAR的物理结构（360°扫描，每个angular sector内有多个range reading）。比起把所有point flatten到一个sequence再过attention，binning + MLP更像[PointNet](https://arxiv.org/abs/1612.00593)的local feature aggregation，compute efficient且structure-preserving。

### 4.3 Goal-aware, Robot-aware Fusion

- Reference image $I_R$也过frozen MAE
- Mask $M$过shallow conv network（类似[SAM](https://arxiv.org/abs/2304.02643)的mask encoder）
- 所有visual token flatten后送入**Vision Context Encoder** $F$
- $F$的output中对应robot observation的token送入**mask decoder** $G$，回归object mask in all history frames（auxiliary loss $L_{\text{mask}}$，pixel-wise L2）
- Goal $(d, \theta, S)$和robot footprint $R$分别通过独立MLP tokenize
- 所有token送入**Multi-modal Context Encoder** $H$

**Mask decoder的作用**：表面上是auxiliary task，实际是让Vision Context Encoder学到"track target object across frames"的能力。从ablation看，去掉mask decoder精度只略微下降（Fig.8），但作者发现model prediction错误时mask tracking也错误，提供了**model interpretability**——这点对debugging很valuable。

### 4.4 Motion Generation: Autoregressive Waypoint Decoding

这是另一个核心设计。Base trajectory表示为$T$个waypoint，每个waypoint是egocentric polar coordinate：

$$(\psi_i, r_i, \phi_i), \quad i = 0, \ldots, T-1$$

变量说明：
- $\psi_i \in [0, 2\pi]$：direction（waypoint相对robot当前heading的角度）
- $r_i \in [0, 0.2\text{m}]$：distance（waypoint到robot的距离，每个waypoint最大20cm）
- $\phi_i \in [0, 2\pi]$：heading（robot到达waypoint时应该朝向的角度）

**为什么用polar不用cartesian**：polar天然适配"robot向前走一段、转一个角度"的运动模式，且每个waypoint的range bounded，便于discretize。

**Multi-token classification with residual**：每个waypoint需要预测$(\psi, r, \phi)$三个量。如果直接分类，bin数量是$30 \times 32 \times 12 = 11520$——太稀疏，每个bin的training sample极少。AMR采用autoregressive factorization：

$$p(\psi, r, \phi) = p(\psi) \cdot p(r|\psi) \cdot p(\phi|\psi, r)$$

每个变量单独分类（30, 32, 12 bins），再通过residual MLP弥补分类的离散性：

$$z' = C(z) + R(z, C(z))$$

- $C(z)$：classifier输出的discrete class center
- $R(\cdot, \cdot)$：MLP，input是token $z$和class center $C(z)$，output是连续residual
- $z'$：final continuous prediction

**Intuition**：纯分类容易训练（CE loss）但精度受限，纯回归精度高但训练不稳定（multi-modal target分布会让MSE loss confused）。Residual classification是两者折衷——classifier处理"大方向"，MLP refine"小偏移"。这和[RT-1](https://arxiv.org/abs/2212.06817)中的token discretization思路一致，也和[BiT](https://arxiv.org/abs/2206.11239)（即引用[25]的Behavior Transformers）的k-means classification思路类似。

**为什么autoregressive而非action chunking**：navigation trajectory天然multi-modal——同一goal可以有多条feasible path（绕左边或绕右边）。ACT（[Action Chunking Transformer](https://arxiv.org/abs/2304.13705)）用diffusion或VAE处理multi-modality，但在long horizon waypoint sequence上效果一般。Autoregressive decoding通过sequential sampling天然支持multi-modal distribution：不同的早期decision（先左转或先右转）会lead to完全不同的后续trajectory。Table II显示autoregressive decoder collision rate 3.8% vs ACT 5.8%。

### 4.5 Camera Tilt Decoding

Tilt angle $\alpha$是uni-modal的（每个时刻一个值），直接L2 regression。和base trajectory异步预测：tilt每个timestep更新一次，trajectory每8个timestep重新predict一次。这种异步设计避免了trajectory prediction被tilt jitter干扰。

### 4.6 Loss

$$L = L_{\text{mask}} + L_{\text{base}} + L_{\text{tilt}}$$

- $L_{\text{mask}}$：pixel-wise L2 on predicted mask in history frames
- $L_{\text{base}}$：每个waypoint的classification loss + regression loss
- $L_{\text{tilt}}$：L2 regression loss on tilt angle

---

## 5. Experimental Results

### 5.1 Simulation: AMR vs Classical Pipeline

Baseline：[FoundationPose](https://arxiv.org/abs/2312.08344)估计object 6D pose → 由CAD model的bounding box计算goal pose → ground truth occupancy map上跑planner。这个baseline用了AMR不需要的三种extra info：CAD model、initial visibility、perfect map。

Fig.6显示：
- Object initially visible: AMR和baseline都work，AMR的angular error更小，尤其object远时
- Object initially invisible: baseline完全无法处理（pose estimator要求object visible），AMR自然handle

**一个interesting observation**：baseline在object太近（0-2m）时angular error反而增大——因为large object（如furniture）部分out-of-view时bounding box估计不稳。AMR作为**active approach**通过移动robot主动改善perception，这点很像[IBVS](https://en.wikipedia.org/wiki/Visual_servoing)的active perception philosophy。

### 5.2 Generalization to Unseen Objects

Table I对比seen vs unseen objects的median error：

| Category | Seen MAE (°) | Seen MDE (m) | Unseen MAE (°) | Unseen MDE (m) |
|----------|--------------|--------------|-----------------|------------------|
| Chair | 1.45 | 0.05 | 1.08 | 0.04 |
| Drawers | 0.77 | 0.02 | 0.55 | 0.03 |
| Couch | 0.82 | 0.02 | 0.57 | 0.04 |
| Picture | 0.84 | 0.03 | 0.64 | 0.04 |
| Shelves | 0.65 | 0.07 | 0.68 | 0.02 |
| Tables | 7.14 | 0.04 | 1.20 | 0.04 |
| Unlabeled | 0.78 | 0.03 | 0.74 | 0.04 |

unseen的误差和seen几乎一样（甚至更小），说明model学到的是generic geometric reasoning而非object-specific template matching。Tables在seen category上error异常高（7.14°）可能是某个training scene中table的特殊放置造成distribution skew，unseen反而更uniform。

### 5.3 Ablation: 各component贡献

Table II（collision rate, 越低越好）：

| Config | Collision % |
|--------|-------------|
| Full | 3.8 |
| No mask decoder | 7.1 |
| ACT decoder | 5.8 |
| No DAgger | 5.5 |
| No depth | 17.7 |
| No footprint | 26.2 |
| No LiDAR | 25.6 |

排序：**LiDAR ≈ Footprint > Depth > DAgger > Mask decoder > ACT**。

- 去掉LiDAR或footprint token，collision率从3.8%暴涨到25%+。LiDAR提供360° coverage，footprint让model"知道自己的身体大小"。两者缺一不可，因为avoid collision需要"知道周围障碍 + 知道自己多大"。
- Depth去掉collision率17.7%，说明depth的3D信息对obstacle avoidance和approach precision都critical。
- Fig.8的error distribution显示所有ablation的median error相近（~3cm），但**long-tail failure case差异巨大**。这是large-scale evaluation才能揭示的——单一scene或小dataset evaluation会miss掉robustness的差异。

### 5.4 Real-World Kitchen Experiment

Table III对比hist=0和hist=4在6个object上的表现：

| | Fridge | Sink | Stove |
|---|--------|------|-------|
| hist=0 | 1/2.4cm/2.3° | 3/14.8cm/0.1° | 3/2.0cm/0.8° |
| hist=4 | 3/3.1cm/0.4° | 3/7.9cm/0.2° | 3/2.4cm/0.8° |

| | Cabinet | Spam | Cup |
|---|---------|------|-----|
| hist=0 | 2/3.1cm/1.7° | 1/1.8cm/0.6° | 3/9.6cm/0.8° |
| hist=4 | 3/2.1cm/0.6° | 2/2.0cm/0.1° | 3/8.7cm/0.9° |

格式：(#completed / distance error / orientation error)，每个object 3次trial。

**关键发现**：simulation中hist=0和hist=4差不多，但real-world中hist=4明显更robust（更多completed trial）。作者的解释是simulation observation clean，history没必要；real-world有遮挡、motion blur、perceptual noise，history让model能"持续track target across disturbances"。这点和[ViNT](https://arxiv.org/abs/2306.14833)中history在real deployment更重要的观察一致。

**Sink是最差的object**：14.8cm error（hist=0）。因为sink是concave to tabletop（嵌入台面），没有large visible surface，model难以稳定track。**Cup也较大**（~9cm）：small object在远距离时只占几个pixel，mask prediction噪声大。

**整体精度**：除sink和cup外，real-world error在1.8-3.1cm，~1°——达到cm-level precision，和simulation几乎一致，sim2real transfer成功。

### 5.5 Forklift Demo

Ackermann steering的forklift通过~500 demonstrations fine-tune后能load pallet——这证明architecture的kinematic agnosticity。base model在differential-drive上训练，少量data fine-tune就能迁移到完全不同的kinematic model，说明model学的是high-level navigation strategy而非specific motion primitive。

---

## 6. 与相关工作的对比和定位

### 6.1 vs ObjectNav / InstanceImageNav

[ObjectNav](https://arxiv.org/abs/2006.11453), [InstanceImageNav](https://arxiv.org/abs/2211.15876), [Goat](https://arxiv.org/abs/2311.06430)都定义success为1m radius，没有precise pose concept。AMR填补了approach phase的precision gap。

### 6.2 vs FoundationPose + Planning

[FoundationPose](https://arxiv.org/abs/2312.08344)是SOTA的6D pose estimator，但需要CAD model + initial visibility。AMR完全不需要CAD，且能handle object initially invisible（通过active search + memory of reference mask）。但注意FoundationPose的精度本身更高（mm级），AMR的cm级足够mobile manipulation但还不足以做高精度assembly。

### 6.3 vs IBVS

[IBVS](https://en.wikipedia.org/wiki/Visual_servoing)是classical active vision方法，通过image feature error feedback控制robot。AMR类似IBVS的active nature（refine prediction as robot approaches），但更sophisticated：reason about object instance（mask tracking），adhere to robot kinematics（trajectory parametrization），handle collision avoidance（LiDAR + footprint awareness）。可以理解为一个learned, object-aware, kinematics-aware IBVS。

### 6.4 vs GNM / ViNT / Nomad

[GNM](https://arxiv.org/abs/2209.11353), [ViNT](https://arxiv.org/abs/2306.14833), [Nomad](https://arxiv.org/abs/2310.07896)是foundation model for visual navigation，但都coarse（1m success）。AMR是complementary的——可以想象一个hierarchical system：GNM/ViNT做coarse global navigation到达object vicinity，AMR接管last 10m做precision approach。

### 6.5 vs Mobile Manipulation Systems

[Mobile ALOHA](https://arxiv.org/abs/2401.02117), [OKRobot](https://arxiv.org/abs/2401.12202), [SPIN](https://arxiv.org/abs/2405.07991)等mobile manipulation system都implicit地assume base positioning已经解决，重点放在arm manipulation。AMR的precision navigation能直接作为这些system的前置module。

---

## 7. Limitations和我的思考

paper自己提到：
- Short-term memory（local navigation only）
- 只在household training，factory和outdoor unstructured环境泛化未验证
- Cylindrical robot + centered camera假设，难以直接apply到legged robot或non-trivial body shape

我额外想到的几点：

**(a) Goal condition的ambiguity**。当object从45° angle被观察时没有dominant side（Fig.7c failure case 4），S会ambiguous。这可能需要更rich的goal specification（如直接给出desired robot pose relative to world frame）或者让model输出confidence并请求clarification。

**(b) Repetitive object confusion**。Fig.7c failure case 3显示当scene中有多个相同object时model可能lock到错的一个。可以用[DETR](https://arxiv.org/abs/2005.12872)-style的query机制或加入spatial context（"the chair next to the window"）。

**(c) Dynamic environment**。paper假设static environment。如果target object被移动（如人推chair），model可能fail。Memory mechanism需要upgrade到dynamic object tracking。

**(d) Polar coordinate的representation limit**。$\phi_i$作为absolute heading在long sequence上会accumulating discretization error。可以换成relative heading（$\Delta \phi_i = \phi_i - \phi_{i-1}$），减少error accumulation。

**(e) Closed-loop vs open-loop execution**。预测12个waypoint但每8步replan——这相当于MPC-style的receding horizon。可以explore more aggressive的closed-loop formulation（每步replan），trade off compute和reactivity。

**(f) Diffusion policy alternative**。base trajectory的multi-modal nature也可以用[Diffusion Policy](https://arxiv.org/abs/2303.04137)处理。paper选autoregressive可能是为了inference speed（10-12Hz），diffusion的iterative denoising可能更慢。但最近[Consistency Policy](https://arxiv.org/abs/2310.17525)等加速方法可以一试。

---

## 8. 几个可能的发展方向

1. **3D object-centric representation**：现在用mask + 2D reference image，可以升级到3D（如[Nerf](https://arxiv.org/abs/2003.08934)或[3D Gaussian Splatting](https://arxiv.org/abs/2308.14737)的object representation），让goal specification更精确。
2. **Language-conditioned precision goal**：把$(S, d, \theta)$替换成自然语言spec（"approach the drawer from the front, 80cm away, slightly to the left"），让[VLM](https://arxiv.org/abs/2304.08485)做spec parsing。
3. **Active perception for goal ambiguity resolution**：当model uncertain时主动explore（移动到不同视角）以disambiguate。
4. **Multi-robot extension**：forklift的fine-tune是per-kinematic的。能否用一个unified architecture + kinematics descriptor token处理多种robot platform？这有点像[RT-X](https://arxiv.org/abs/2310.12984)的cross-embodiment idea。
5. **Integration with manipulation**：当base到达precision goal后，arm manipulation可以assume base pose fixed，简化manipulation policy的input space。这能显著提升[Mobile ALOHA](https://arxiv.org/abs/2401.02117)这类system的success rate。

---

## References

- [AMR Project Page](https://sites.google.com/view/aimmyrobot)
- [HSSD-200](https://arxiv.org/abs/2306.11290)
- [Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [MAE](https://arxiv.org/abs/2111.06377)
- [AIT\*](https://arxiv.org/abs/2006.16685)
- [FoundationPose](https://arxiv.org/abs/2312.08344)
- [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643)
- [DAgger](https://arxiv.org/abs/1011.0684)
- [Code as Policies](https://arxiv.org/abs/2302.01601)
- [Goat: Go to Any Thing](https://arxiv.org/abs/2311.06430)
- [InstanceImageNav](https://arxiv.org/abs/2211.15876)
- [GNM](https://arxiv.org/abs/2209.11353)
- [ViNT](https://arxiv.org/abs/2306.14833)
- [Nomad](https://arxiv.org/abs/2310.07896)
- [OKRobot](https://arxiv.org/abs/2401.12202)
- [Mobile ALOHA](https://arxiv.org/abs/2401.02117)
- [SPIN](https://arxiv.org/abs/2405.07991)
- [Behavior Transformers (BiT)](https://arxiv.org/abs/2206.11239)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Action Chunking Transformer](https://arxiv.org/abs/2304.13705)
- [PointNet](https://arxiv.org/abs/1612.00593)
- [DETR](https://arxiv.org/abs/2005.12872)
- [BEVFormer](https://arxiv.org/abs/2203.17270)
- [NeRF](https://arxiv.org/abs/2003.08934)
- [3D Gaussian Splatting](https://arxiv.org/abs/2308.14737)
- [RT-1](https://arxiv.org/abs/2212.06817)
- [RT-X](https://arxiv.org/abs/2310.12984)
- [Reeds-Shepp curves (Wikipedia)](https://en.wikipedia.org/wiki/Dubins_path#Reeds%E2%80%93Shepp_curve)
- [Visual Servoing (IBVS)](https://en.wikipedia.org/wiki/Visual_servoing)

---

总结一句intuition给你：AMR本质上是把"object pose estimation"这个passive perception problem，重新cast成"closed-loop active control with learned perceptual representation"这个active problem——用deep multi-modal encoder替代pose estimator，用autoregressive waypoint decoder替代geometric planner，用simulation-generated near-optimal demonstrations替代human teleop。整个系统的precision来自simulation planner的optimality，robustness来自multi-modal fusion和large-scale data diversity。这是一个把"precision"从classical pipeline的privilege变成learned system的capability的important step。

如果让我pointer to future work，我会建议关注**dynamic object tracking**和**cross-embodiment generalization**——这两个方向的突破会让AMR从"local precision navigation"进化成真正的"general precision mobile manipulation primitive"。
