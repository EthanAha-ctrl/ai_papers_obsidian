---
source_pdf: OmniRetarget Interaction-Preserving Data Generation for Humanoid Whole-Body
  Loco-Manipulation and Scene Interaction.pdf
paper_sha256: dd7cdf341d4c0a69127ee93fb51f4286ad648d7c8aa90579c5b2bbff7c672528
processed_at: '2026-08-05T23:23:36-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 OmniRetarget

## 这篇 paper 到底在干嘛

想象你是一个 humanoid robot 的教练, 你想教 Unitree G1 学会搬箱子、爬台阶、甚至跑酷。最直接的办法就是让人类先表演一遍, 录下来, 然后让 robot 照着学。

问题来了: 人的 body 和 robot 的 body 长得不一样。人有 200 多块骨头, 灵活得很; G1 就那么几个 joint, 还有一双 rigid 的脚板。你直接把人的 motion 套到 robot 身上, 会出现各种尴尬:

- **Foot skating**: 人的脚明明踩在地上不动, robot 的脚却在地上滑来滑去 (因为 IK 算出来的 joint angle 让脚没法精确停在原位)
- **Penetration**: 人的手可以 "穿过" 想象中的箱子表面去抓另一面, robot 的手就直接插进箱子里了
- **Interaction 丢失**: 人抓箱子时, 手和箱子表面是有 specific 的 relative angle 的, 你只 match "手的位置", 这个 angle 信息就丢了

这些 artifact 看起来是小问题, 但对 downstream RL 是灾难。RL policy 看到 reference 里 foot 在滑, 它就学到了 "原来脚可以滑", 于是真机上脚真的滑, robot 摔倒。或者 reference 里手穿进箱子, RL 学到 "手可以穿进物体", 真机上直接撞箱子。

prior work 怎么解决? 加一堆 reward 来 compensate: foot air time penalty、contact schedule reward、collision penalty... 调参调到怀疑人生, 而且每个 task 要单独调。

OmniRetarget 的核心 insight 很简单: **与其在 RL 阶段擦屁股, 不如一开始就把 reference 弄干净**。Reference 干净了, RL 就只需要很少的 reward, 不用调参, 还能 zero-shot 上真机。

---

## 它怎么把 reference 弄干净的

### Core idea: Interaction Mesh

关键 trick 是搞一个 "interaction mesh"。把人的 joint 位置、箱子表面采的点、地面采的点, 全扔进一个 3D mesh 里 (Delaunay tetrahedralization, 就是把空间切成一堆 tetrahedron)。

这个 mesh 的 vertices 之间的 connectivity 编码了 "手离箱子表面多远"、"脚离地面多高" 这种 spatial relationship。数学上用 Laplacian coordinate 来表达: 每个 vertex 相对于它 neighbors centroid 的 offset。

当你把 human motion 搬到 robot 上时, 只要 keep 这个 Laplacian coordinate 尽量不变, "手和箱子的相对关系" 就自动 preserve 了。手抓箱子边缘这个 action, 不管箱子多大、放哪, 关系都一样。

### Hard constraint: 直接消灭 artifact

Prior work 多是 "soft penalty": collision 大了惩罚一下, foot 滑了惩罚一下。问题是 penalty 之间会打架, 你调一个 weight 就影响另一个, 最后妥协出一个 "都不太严重" 的结果。

OmniRetarget 用 **hard constraint**:

- Collision: signed distance 必须 ≥ 0, 直接 forbid penetration
- Foot sticking: stance phase 的脚, 位置必须 = 上一帧位置, 直接 forbid skating
- Joint limit, velocity limit: 都 hard bound

用 Sequential SOCP solver 来解这个 constrained optimization。每帧 warm start 上一帧的解, 迭代 10 次左右收敛。

结果就是 Table II 里看到的: penetration duration = 0.00, foot skating = 0。干干净净。

### Data augmentation: 一个 demo 变一堆

Teleoperation 采数据很贵, 一个人穿 MoCap suit 搬箱子搬一天也采集不了多少 variation。OmniRetarget 可以从一个 demo 自动生成大量 variation:

- 改箱子 initial pose (位置、朝向)
- 改箱子 shape (尺寸 scale)
- 改 terrain 高度

怎么做的? 直接在 interaction mesh 里把箱子的点 transform 一下, 重新解 optimization, robot 的 motion 就自动适应了。

这里有个 trick: mesh 必须在 **object local frame** 里构造, 不能用 world frame。否则箱子一转, Laplacian coordinate 全变了, optimization 就乱了。用 object frame 的话, "手抓箱子边缘" 这个关系是 rotation-invariant 的。

还有个 trick: 单纯移箱子, optimization 可能直接把整个 robot rigid transform 一下就完事了, 这没意义。所以加一个 constraint 锁住脚的位置, 逼着 upper body 重新协调, 生成真正 diverse 的数据。

---

## 为什么 RL 可以这么简单

Prior work 的 RL 通常要十几个 reward: body tracking、foot contact、air time、collision penalty、joint acceleration、energy、reach target、object height、object orientation... 每个都要 tune weight。

OmniRetarget 只用 5 个:
1. Body tracking (DeepMimic-style)
2. Object tracking (when applicable)
3. Action rate (平滑 action)
4. Soft joint limit
5. Self-collision

全部用 BeyondMimic 的默认 hyperparameter, 一个都不调。Domain randomization 也只 4 项 (torso COM、joint default、random push、obs noise), 对比 prior work 的 RFI、motor PD、action delay 等十几项, 极简。

Observation 纯 proprioceptive, robot 是 blind 的。Scene understanding 的责任完全由 reference motion 承担, RL 只负责 dynamics realization: 把 kinematic trajectory 变成 physically realizable 的 action。

这印证了 BeyondMimic 的 philosophy: **reference 质量决定 RL 难度**。Clean reference + minimal RL > Noisy reference + heavy reward engineering。

---

## 效果如何

### Kinematic quality

Table II 数字很直观。OmniRetarget penetration duration = 0.00±0.01, max depth 1.34cm (SQP linearization 的 small violation, RL 能 fix)。Foot skating = 0。Contact preservation 0.96 (和 GMR 的 0.99 差不多, 但 GMR 是把 hand 塞进箱子里的 "假 contact")。

### RL downstream success rate

- Robot-object (OMOMO): OmniRetarget 82.20% ± 9.74%, PHC 71.28% ± 22.55%, GMR 50.83% ± 23.89%, VideoMimic 3.85% ± 8.41%
- Robot-terrain: OmniRetarget 94.73%, PHC 52.63% ± 49.93% (variance 巨大, 说明很多 motion 直接挂了)

Terrain climbing 这种 agile motion 对 reference precision 要求极高, baseline 有一半 motion 完全失败 (0% success), OmniRetarget 只有一个失败。

### 真机 demo

- 30 秒 parkour: 搬椅子 → 用椅子当踏脚石爬 platform → 跳下做 roll 吸收冲击
- 0.9m platform climbing (70% robot height)
- Wall-flip: 0.5s 完成, peak angular velocity 15 rad/s, 真机 5/5 成功
- Slope crawling, box carrying

全部 zero-shot sim-to-real, 没有真机 fine-tuning。

---

## 为什么这篇 paper 重要

1. **Data-centric paradigm**: 把 humanoid learning 的 bottleneck 从 "调 RL reward" 移到 "生成 clean data"。这和 LLM 的 data-centric 趋势一致, 与其花力气 fix model, 不如花力气 fix data。

2. **Interaction-preserving**: 第一次把 human-object-terrain 的 interaction 显式建模进 retargeting。Prior work 只 match keypoint, 丢失了 interaction structure。

3. **Scalable data generation**: 一个 demo 能 augment 出大量 variation, 解决 teleoperation 的 scalability 问题。Object frame mesh 的 trick 让 augmentation 是 "几何上 meaningful" 的。

4. **Minimal RL**: 5 reward、4 DR、proprioceptive only、no curriculum、no tuning。这证明 "reference 质量" 是 RL complexity 的主要 driver, 而不是 "task 难度"。

5. **开源**: 全部 code、retargeted dataset (8+ 小时)、trained policy 都会 open-source。对 humanoid community 是巨大的资源。

---

## 一句话

OmniRetarget 告诉你: humanoid 学复杂 skill, 关键不在 RL 多聪明, 而在 reference motion 多干净。把 motion generation 做 principled, RL 就 minimal, 就能 zero-shot 上真机跑 30 秒 parkour。

---

参考:
- Project page: https://omniretarget.github.io/
- BeyondMimic: https://arxiv.org/abs/2508.04614
- OMOMO dataset: https://jasonqsy.github.io/OMOMO/
- Drake: https://drake.mit.edu/
- Unitree G1: https://www.unitree.com/g1/

---

# OmniRetarget 论文深度解析

## 1. 核心问题与 Motivation

Humanoid robot 学习复杂 whole-body skill 的最大 bottleneck 是 **data scarcity** 和 **embodiment gap**。当我们将 human motion capture 数据 retarget 到 humanoid robot (如 Unitree G1) 时, 现有方法存在三大问题:

1. **Foot skating**: 支撑脚在 contact phase 期间发生滑动
2. **Penetration**: 机器人 body 穿入 object 或 terrain
3. **Interaction loss**: 丢失了 human-object、human-terrain 之间的 spatial relationship

这些 artifact 导致下游 RL policy 需要大量 ad-hoc reward engineering (例如 air time penalty、contact schedule reward) 来 compensate, 这正是 BeyondMimic [33] 所揭示的 "garbage in, garbage out" 问题。

OmniRetarget 的核心 insight: **与其在 RL 阶段用复杂 reward 弥补低质量 reference, 不如在 retargeting 阶段就用 principled optimization 生成 artifact-free 的 trajectory**。

参考: 
- BeyondMimic: https://arxiv.org/abs/2508.04614 (BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion)
- DeepMimic: https://xbpeng.github.io/projects/DeepMimic/index.html

---

## 2. Interaction Mesh: 核心 Mathematical Tool

### 2.1 什么是 Interaction Mesh

Interaction mesh 最早由 Ho et al. (SIGGRAPH 2010) [14] 提出, 是一个 volumetric tetrahedral mesh, 其 vertices 包括:
- Human/robot 的 anatomical keypoints (joint positions)
- 从 object surface 和 environment surface 采样的点

通过 **Delaunay tetrahedralization** [48] 构造这个 mesh (3D 的 Delaunay triangulation)。Key idea 是: 这个 mesh 的 **Laplacian coordinates** 编码了 body parts 之间以及 body 与 environment 之间的 relative spatial relationship。当我们 warp human mesh 到 robot mesh 时, 只需保持 Laplacian coordinate 尽量不变, 就能 preserve 这些 interaction。

参考:
- Interaction Mesh original paper: https://dl.acm.org/doi/10.1145/1778765.1778834
- Delaunay tetrahedralization: Si & Gärtner (2005)

### 2.2 Laplacian Coordinate 的定义

公式 (1):
$$L(p_{t,i}) = p_{t,i} - \sum_{j \in \mathcal{N}(i)} w_{ij} \cdot p_{t,j}$$

变量解释:
- $p_{t,i} \in \mathcal{P}_t$: 第 $t$ 帧第 $i$ 个 keypoint 的 3D position
- $\mathcal{N}(i)$: 在 tetrahedralization 中与 vertex $i$ 相连的所有 neighbors
- $w_{ij}$: 权重, 论文用 uniform weight $w_{ij} = 1/|\mathcal{N}(i)|$

Laplacian coordinate $L(p_{t,i})$ 本质上是 vertex $i$ 相对于其 neighborhood **centroid** 的 offset vector, 它表达了 local geometric structure (类似 graph signal processing 中的 Laplacian operator)。如果 body part 和 environment 在 human demo 中保持某种 spatial relationship, 这个 relationship 就编码在 Laplacian coordinate 中, retargeting 时只要最小化 $L$ 的 change, 就能 preserve 这种 relationship。

### 2.3 Laplacian Deformation Energy

公式 (2):
$$E_L = \sum \left\| L(p_{t,i}^{\text{source}}) - L(p_{t,i}^{\text{target}}) \right\|^2$$

求和遍历所有对应 keypoint pair。这是 mesh deformation 的 standard objective (Alexa 2003 [49], Zhou et al. SIGGRAPH 2005 [50]), 用于驱动 human motion 到 robot motion 的 warping, 同时 preserve interaction structure。

---

## 3. Constrained Optimization Formulation

### 3.1 主目标 (3a)

$$q_t^* = \arg\min_{q_t} \sum_i \left\| L(p_{t,i}^{\text{source}}) - L(p_{t,i}^{\text{target}}(q_t)) \right\|^2 + \left\| q_t - q_{t-1} \right\|_Q^2$$

变量含义:
- $q_t$: robot configuration (floating base pose 的 quaternion + translation, 加上所有 joint angles)
- $p_{t,i}^{\text{target}}(q_t) = f_i(q_t)$: 通过 forward kinematics 从 $q_t$ 计算得到的 robot keypoint position
- $Q$: cost matrix, 鼓励 temporal smoothness, 防止 trajectory 抖动

第一项是 Laplacian deformation (核心), 第二项是 smoothness regularization。

### 3.2 Hard Constraints (3b-3e)

这是 OmniRetarget 与 PHC、GMR、VideoMimic 的核心差异 (Table I):

**(3b) Collision avoidance**: $\phi_j(q_t) \geq 0, \forall j$
- $\phi_j$ 是第 $j$ 个 collision pair 的 **signed distance function (SDF)**
- 保证 robot body、object、terrain 之间无 penetration

**(3c) Joint position limits**: $q_{\min} \leq q_t \leq q_{\max}$

**(3d) Joint velocity limits**: $\nu_{\min} \cdot dt \leq q_t - q_{t-1} \leq \nu_{\max} \cdot dt$
- 保证 motion 在 motor 物理能力内

**(3e) Foot sticking constraint**: $p_t^F = p_{t-1}^F, \forall \text{stance foot}$
- Stance phase 定义: source motion 中脚的 horizontal velocity < 1 cm/s
- 这条 hard constraint **直接消灭 foot skating** (Table II 显示 OmniRetarget 的 foot skating Duration 和 Max Velocity 均为 0)

### 3.3 Sequential SOCP Solver

论文 Appendix D 给出详细求解过程。对每帧 $t$ 迭代求解一个 **Second-Order Cone Program**:

$$dq_n^* = \arg\min_{dq_n} \| L^{\text{source}} - (J_L^n \cdot dq_n + \bar{L}_n^{\text{target}}) \|^2 + \|\bar{q}_n + dq_n - q_{t-1}\|_Q^2$$

subject to:
- $J_j^n \cdot dq_n + \phi_j(\bar{q}_n) \geq 0$ (linearized collision)
- Joint limits (linearized)
- Velocity limits (linearized)
- $p_t^F(\bar{q}_n) + J_F^n \cdot dq_n = p_{t-1}^F$ (linearized foot sticking)
- $\|dq_n\|_2 \leq \varepsilon$ (trust region, $\varepsilon = 0.2$)

变量含义:
- $dq_n$: 第 $n$ 次 SQP iteration 的 configuration increment
- $\bar{q}_n = q_{t-1}^* + \sum_{k<n} dq_k$: 当前 iterate
- $J_L^n = \partial L^{\text{target}} / \partial q |_{q=\bar{q}_n}$: Laplacian 对 configuration 的 Jacobian
- $J_j^n, J_F^n$: collision SDF 和 foot position 的 Jacobian

关键技术细节:
1. **Quaternion 微分几何**: floating base 是 $S^3$ manifold (单位 quaternion), 用 Drake [51] 的 automatic differentiation 正确处理 rotation 的 differential geometry (Planning with Attitude [52] 思路)
2. **Warm start**: 每帧用上一帧 $q_{t-1}^*$ 作为初始 guess, 加速收敛
3. **Trust region** $\varepsilon = 0.2$ 保证 linearization 有效

参考:
- Drake: https://drake.mit.edu/
- Planning with Attitude: https://arxiv.org/abs/2103.02447

---

## 4. Data Augmentation: 从单一 demo 到大规模 dataset

这是论文的一大亮点, 解决了 teleoperation 数据采集 labor-intensive 的问题。

### 4.1 Object Pose & Shape Augmentation

公式 (14): Exponential decay offset
$$\tilde{p}_{\text{obj}}(t) = \begin{cases} \Delta p_{\text{obj}} + p_{\text{obj}}(0) & \text{if } t < t_m \\ \Delta p_{\text{obj}} e^{-(t-t_m)/\tau_p} + p_{\text{obj}}(t) & \text{if } t \geq t_m \end{cases}$$

变量:
- $\Delta p_{\text{obj}}, \Delta\theta_{\text{obj}}$: 对 object 初始 pose 的 perturbation (translation + rotation)
- $t_m$: object 开始运动的时间
- $\tau_p, \tau_\theta$: decay time constants
- $\oplus$: quaternion composition

这个 formulation 很精妙: perturbation 在 $t_m$ 之前是 constant offset (改变 object 的 initial pose), 之后 exponential decay 让 object trajectory 平滑回归到原轨迹, 这样 robot 既看到了不同的 initial configuration, 又能复用后续的 manipulation motion。

### 4.2 防止 trivial augmentation

简单转动物体后, 整个 robot 跟着做 rigid transform 是无意义的。论文引入:
- 公式 (4): $\|q_t - \bar{q}_t^*\|_W$, 其中 $W$ heavily penalize 下肢 deviation (锁定 foot 位置)
- 公式 (5): $p_0^F = \bar{p}_0^{F*}$ (强制 initial foot pose 与 nominal 一致)

这迫使 upper body 重新协调来 pick up 不同 pose 的 box, 生成真正有意义的多样性。

### 4.3 Object Frame Interaction Mesh

Section VI-C.2 和 Fig. 8 的关键 insight:

如果用 world frame 计算 Laplacian coordinate, 当 object 旋转 180° 时, $L_W$ 从 (0,1) 变到 (0,-1), 但 $L_O$ (object frame) 保持不变。因此 **mesh 必须在 object local frame 中构造**, 才能 preserve robot-object 的相对几何关系 (例如 "手抓 box 的把手"这种 relation 不应随 box 整体 rotation 而改变)。

---

## 5. Minimal RL Formulation

论文 Section IV 声称: 由于 reference motion 质量 high-fidelity, RL 只需要 5 个 reward term, 不需要 curriculum。

### 5.1 Observation Space (纯 proprioceptive)

- Reference motion: joint position/velocity, pelvis position/orientation error
- Proprioception: pelvis linear/angular velocity, joint position/velocity
- Previous action

**Agent 是 blind** 的, 完全不知道 scene 和 object 的 explicit 信息, 必须严格 follow reference trajectory。这等价于: 把感知 scene 的责任完全交给 reference motion generation 阶段 (OmniRetarget 负责), RL 只负责 dynamics realization。

### 5.2 五个 Reward Term

1. **Body tracking**: DeepMimic-style position/orientation/velocity tracking
2. **Object tracking** (where applicable): DeepMimic-style for object pose
3. **Action rate**: $\|a_t - a_{t-1}\|^2$, 鼓励平滑 action
4. **Soft joint limit penalty**: 当 joint 接近 limit 时惩罚
5. **Self-collision penalty**: binary, 当 self-collision force > 1N 时触发

**Hyperparameter 完全使用 BeyondMimic [33] 的默认值, 不调参**, 这是 minimal formulation 的精髓。

### 5.3 Domain Randomization (极简, 只 4 项)

对比 prior work 通常有 RFI、motor PD、action delay 等十几项, OmniRetarget 只用:
- Torso COM position: ±2.5/5/7.5 cm
- Joint default position: ±0.01 rad
- Random push: 0.3 m/s, 0.78 rad/s for 1-3s
- Observation noise

Object 物理 randomization (mass 0.1-2kg, COM ±8cm, inertia 50-150%, shape ±10%) 用于 generalization。

---

## 6. 实验 Results 详解

### 6.1 Kinematic Quality Benchmark (Table II)

**Robot-Object Interaction (OMOMO dataset)**:

| Method | Penetration Duration | Max Depth (cm) | Foot Skating Duration | RL Success |
|--------|---------------------|----------------|----------------------|-----------|
| PHC | 0.68 ± 0.21 | 5.11 ± 3.09 | 0.05 ± 0.05 | 71.28% ± 22.55% |
| GMR | 0.83 ± 0.14 | 8.50 ± 3.94 | 0.02 ± 0.01 | 50.83% ± 23.89% |
| VideoMimic | 0.60 ± 0.27 | 7.48 ± 4.95 | 0.12 ± 0.07 | 3.85% ± 8.41% |
| **OmniRetarget** | **0.00 ± 0.01** | **1.34 ± 0.34** | **0** | **82.20% ± 9.74%** |

观察:
1. OmniRetarget penetration 几乎为 0 (因为 hard constraint), 但偶尔有 1.34cm max depth, 这是 SQP linearization 引入的 small violation, 论文说 "RL 可以 easily fix"
2. Foot skating 严格为 0 (hard constraint)
3. Contact preservation 0.96 与 GMR 0.99 相当, 但 GMR 用 keypoint scaling 把 hand 塞进 object 内 (Fig. 7b), 是 "假" contact
4. RL success rate: OmniRetarget 比 baseline 高 10%+, variance 也显著小 (9.74% vs 22.55%)

**Robot-Terrain (In-House MoCap)**:

OmniRetarget success rate 94.73% vs PHC 52.63% (variance 49.93%!), 差距巨大。Fig. 10b histogram 显示: PHC 和 VideoMimic 有近一半 motion **完全失败** (0% success), 因为 terrain climbing 需要极高 precision reference, 低质量 reference 无法被 RL recover。

### 6.2 Wall-Flip 实验 (Fig. 6)

这是一个高 dynamic 极限 case:
- 完成时间 0.5s
- Peak angular velocity 15 rad/s
- Peak linear velocity 3.5 m/s

技术细节:
- Robot foot 是 rigid 的, 而 human foot 有 arch 可以 flex 维持 contact
- 因此 retargeting 时 robot 必须 align 更靠近 wall 才能获得足够 contact area
- RL training 时放松了 end-effector position error threshold 到 0.5m (其他 motion 0.25m), 并移除了 foot joint orientation tracking term
- 其他 reward 保持不变

**5/5 真机 success rate**, 说明 OmniRetarget + minimal RL 的鲁棒性。

### 6.3 30-second Parkour Sequence (Fig. 1)

完整 pipeline 展示:
1. 搬运 4.6kg chair 到 platform
2. 用 chair 作为 stepstone 爬上去
3. 从 platform 跳下并做 parkour roll 吸收冲击

这个 multi-stage task 展示了 OmniRetarget 在 long-horizon 复杂场景的能力, 也是对 Boston Dynamics Atlas tool-use demo [53] 的致敬。

---

## 7. 与 Prior Retargeting Methods 的技术对比 (Table III)

| Method | Optimization Type | Objective | Preprocessing | Data Format |
|--------|------------------|-----------|---------------|-------------|
| PHC [10] | Trajectory-wise | Keypoint position match | Model fitting (SMPL scale) | SMPL only |
| GMR [9] | Per-frame IK | Position+orientation match | Direct scaling | SMPL, BVH |
| VideoMimic [11] | Trajectory-wise | Pairwise distance preservation | Model fitting | SMPL only |
| IMMA [22] | Multi-stage | Interaction mesh + IK | Unknown | Unknown |
| **OmniRetarget** | Per-frame | Interaction mesh deformation | Direct scaling | SMPL, BVH |

关键差异:
1. **Per-frame vs trajectory-wise**: OmniRetarget 用 per-frame (像 GMR), 但 objective 是 interaction mesh (像 IMMA)。Trajectory-wise (PHC, VideoMimic) 更全局但难收敛, IMMA 用 multi-stage 导致 sub-optimal
2. **Hard constraint vs soft penalty**: VideoMimic 用 soft penalty (Eq. 11 中 $\lambda_c \mathcal{L}_{\text{collision}}$ 等), 需要 tune 6+ 个 $\lambda$ 权重, 而 OmniRetarget 用 hard constraint, 无需 tune
3. **Direct scaling vs model fitting**: OmniRetarget 用 $p_{t,i}^{\text{source}} = \alpha \cdot M_i(q_t^{\text{demo}}; \beta^{\text{demo}})$, $\alpha = h_{\text{robot}}/h_{\text{demo}}$, 不需要拟合 SMPL 到 robot morphology, 兼容 BVH 数据更友好

参考:
- PHC: https://github.com/ZhengyiLuo/PHC
- OmniH2O: https://omni-h2o-translation.github.io/
- HumanPlus: https://humanoid-ai.github.io/
- VideoMimic: https://videomimic.github.io/

---

## 8. Architecture Flow 解析

```
┌─────────────────────────────────────────────────────────┐
│  Input: Human MoCap (SMPL / BVH)                       │
│  e.g., OMOMO, LAFAN1, in-house MoCap                   │
└────────────────────────┬────────────────────────────────┘
                         │ Direct scaling: α = h_robot/h_demo
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Source keypoints P_t^source                            │
│  + Object/terrain surface points (sampled)              │
└────────────────────────┬────────────────────────────────┘
                         │ Delaunay tetrahedralization
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Interaction Mesh (source)                              │
│  - Vertices: anatomical keypoints + scene points        │
│  - Edges encode spatial relationship                    │
└────────────────────────┬────────────────────────────────┘
                         │ Augmentation (object pose/shape, terrain)
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Constrained Optimization (per-frame)                   │
│  min: Laplacian deformation + smoothness                │
│  s.t.: collision SDF ≥ 0, joint/velocity limits,       │
│        foot sticking                                    │
│  Solver: Sequential SOCP with Drake autodiff            │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Kinematically feasible robot trajectory q_0:T          │
│  - No penetration, no foot skating                     │
│  - Preserved interaction structure                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  RL Training (BeyondMimic-style minimal formulation)   │
│  - 5 rewards, 4 DR terms, proprioceptive only           │
│  - No curriculum, no reward tuning                     │
└────────────────────────┬────────────────────────────────┘
                         │ Zero-shot sim-to-real
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Real humanoid (Unitree G1)                             │
│  - 30s parkour, box carrying, platform climbing,       │
│    wall-flip, slope crawling                            │
└─────────────────────────────────────────────────────────┘
```

---

## 9. 我对这篇 paper 的 Intuition 总结

### 9.1 为什么 Interaction Mesh 是对的关键

Humanoid retargeting 最大的失败模式是 "keypoint matching" 思维: 把 hand 抓 box 的某点 retarget 到 robot hand 的某点。这种 reduction 丢失了 "hand 和 box 的 surface 是 how 相对 oriented" 的信息。OmniRetarget 通过 mesh (包含 box 的 surface points) + Laplacian coordinate, **同时**保持 hand 与 box surface 的相对 geometry, 这是 interaction-preserving 的数学本质。

### 9.2 为什么 Minimal RL 可行

Prior work 之所以需要大量 reward engineering, 是因为 reference 有 artifact:
- Foot skating → 需要 air time / contact schedule reward
- Penetration → 需要 collision penalty
- Interaction loss → 需要 task-specific reward (e.g., box height tracking)

OmniRetarget 在 **kinematic layer** 就解决了这些, 把 "scene understanding" 的 burden 移到上游的 optimization。RL 只剩 "dynamics realization" 任务, 5 个 reward 足够。

这与 BeyondMimic [33] 的 philosophy 完全一致: **当 reference 是 clean 的, minimal formulation 就 sufficient**。OmniRetarget 把这个 insight 推广到 scene-interaction 任务。

### 9.3 Data Augmentation 的"几何不变性"思想

Object 改变 pose 时, 如果 mesh 在 world frame, Laplacian 会变化 (与 global rotation耦合), 导致 optimization 错乱。Object frame 的 mesh 让 Laplacian **invariant to rigid transform of object**, 这是数学上的一个 elegant trick, 让 augmentation 自动产生合理的 robot motion。

### 9.4 未来方向

论文 Section VI 提到 frame-by-frame optimization 对 noisy source (如 video) 鲁棒性不足, 可以做 trajectory-wise joint optimization。另一个方向是 learning autonomous visuomotor policy (目前是 proprioceptive only, 完全依赖 reference motion 作为 "scene prior")。

潜在延伸:
- 用 OmniRetarget 生成数据训练 **generative model** (如 diffusion policy over humanoid whole-body)
- 结合 **vision-language model** 做 high-level task planning + OmniRetarget 做 low-level reference generation
- 把 interaction mesh 推广到 **multi-agent** (人-人 interaction, 人-工具-物体)

---

## 10. 重要 Links

- **Project page**: https://omniretarget.github.io/
- **BeyondMimic (核心引用)**: https://arxiv.org/abs/2508.04614
- **OMOMO dataset**: https://jasonqsy.github.io/OMOMO/
- **LAFAN1 dataset**: https://github.com/tyiyu/LAFAN1-Retargeting-Dataset
- **Unitree G1**: https://www.unitree.com/g1/
- **Drake**: https://drake.mit.edu/
- **DeepMimic**: https://xbpeng.github.io/projects/DeepMimic/index.html
- **OmniH2O**: https://omni-h2o-translation.github.io/
- **ASAP**: https://arxiv.org/abs/2502.01143 (参考 [13], humanoid agile skill)
- **Atlas tool-use demo**: https://www.youtube.com/watch?v=-e1QhJ1EhQ
- **Interaction Mesh (Ho et al.)**: https://dl.acm.org/doi/10.1145/1778765.1778834
- **Mink IK library**: https://github.com/kevinzakka/mink

---

## 11. 一句话总结

OmniRetarget 把 humanoid loco-manipulation 的 data bottleneck 从 **"downstream RL reward engineering"** 移到了 **"upstream principled optimization"**, 通过 interaction mesh + hard constrained SOCP 生成 interaction-preserving、artifact-free 的 reference, 让 minimal RL formulation (5 reward, 4 DR, proprioceptive only) 即可实现 zero-shot sim-to-real 的复杂 30s parkour sequence。这是 BeyondMimic philosophy 在 scene-interaction 场景下的重要推广, 也是 data-centric AI 在 humanoid robotics 领域的典范应用。
