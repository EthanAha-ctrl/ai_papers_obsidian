---
source_pdf: HyperSim A Holistic Sim-To-Real Framework For Robust Robotic Manipulation.pdf
paper_sha256: f17379c87b20c29e282aa621be6a8bf8261dd518b0d29783a7dd7a9686388809
processed_at: '2026-08-05T08:56:19-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HyperSim 用人话说

---

## 一句话说清楚

你在 simulation 里训了个 robot policy，搬到真实世界就拉胯。这篇 paper 说：**sim 和 real 之间有三道坎**，你得一起迈过去，光迈一道没用。

哪三道坎？

1. **长得不像**——sim 里的画面太干净太假，real world 又脏又乱
2. **练得太少**——policy 只见过"正常路径"，没见过意外情况，一碰到扰动就懵
3. **不懂真实物理**——sim 的动力学和 real 不完全一致，policy 学到的 mapping 有偏差

HyperSim 就是三管齐下，同时治这三个病。

---

## 三道坎分别怎么治

### 第一道坎：长得不像

**问题**: 传统 sim 就是"一张 floating table + void background"，跟真实世界差太远。robot 在 sim 里学的 visual feature 到 real world 全失效。

**药方**: 用 **3D Gaussian Splatting (3DGS)** 重建真实环境。

具体做法分两块：

**Background**（不碰的区域）: 拿个 LiDAR + RGB camera 扫一遍真实环境，用 GPGS 方法生成一堆 Gaussian primitives。每个 Gaussian 就是空间里的一个"发光小椭球"，有位置、形状、颜色、透明度。几百上万个这样的小椭球叠加起来，就能 render 出 photorealistic 的画面。

公式长这样：

$$\mathcal{G}(\mathbf{p}) = \exp\left(-\frac{1}{2}(\mathbf{p}-\mathbf{p}_k)^T \Sigma^{-1} (\mathbf{p}-\mathbf{p}_k)\right)$$

人话翻译：
- $\mathbf{p}$: 你想 query 的那个 3D 点
- $\mathbf{p}_k$: 第 $k$ 个小椭球的中心
- $\Sigma$: 椭球的形状矩阵，决定它往哪个方向扁、哪个方向长
- 整个公式就是算"这个点离椭球中心有多远"，越近值越大，越远越快衰减到 0

渲染 pixel 颜色的时候：

$$\mathbf{c}(\mathbf{x}) = \sum_{k=1}^{K} \mathbf{c}_k \cdot o_k \cdot \mathcal{G}^{2D}(\mathbf{x}) \cdot \prod_{j=1}^{k-1}\left(1 - o_j \cdot \mathcal{G}^{2D}(\mathbf{x})\right)$$

人话翻译：
- 从相机射出一条光线，依次穿过一堆按深度排好序的 Gaussian
- 每穿过一个，光线被"吸收"一部分（opacity 决定吸多少）
- 最终 pixel 颜色 = 所有 Gaussian 贡献的加权平均
- 前面的 Gaussian 越不透明，后面的越被"挡住"（transmittance term）

**Foreground**（要碰的区域）: 不能用 Gaussian，因为 Gaussian 的 mesh 噪声太大，做不了精确的 collision detection。所以用 18 个 spatial constraint solver 来 procedural 生成 foreground，比如 `place_on_surface`、`place_left_edge`、`with_obstacles` 这些。

然后关键一步：用 TSDF 把 Gaussian render 出来的 color+depth 融合成一个 mesh，这个 mesh 和 Gaussian 严格对齐。物理引擎用 mesh 做 collision，rendering 用 Gaussian 做 visual，两者共享同一套 geometry。

**为什么这么设计**: Gaussian 擅长 "看起来真"，mesh 擅长 "碰撞算得准"。拆开各干各的活，比硬要一个东西同时干两件事好。

---

### 第二道坎：练得太少

**问题**: 传统 trajectory generation 只生成"成功路径"——从 A 到 B 一路顺畅。policy 学完只会 "背课文"，稍微有点变化就傻眼。

**药方**: **Adversarial perturbation and recovery**。

核心 idea 特别简单：

> 当 robot 的手快要碰到目标物体的时候（这个时刻叫 bottleneck pose），突然把目标物体挪走/转一个角度，逼 robot 重新找位置。

具体流程：

1. 把 task 拆成 subtask（比如 "抓杯子" 是一个 subtask，"放杯子" 是另一个）
2. 每个 subtask 分两段：approaching（手飞过去）+ interaction（手碰到物体开始操作）
3. approaching 和 interaction 的分界点就是 **bottleneck pose**——定义为 TCP 进入以物体中心为圆心、半径 $d$ 的 hemisphere 的那一刻
4. robot 刚到 bottleneck pose，啪一下把物体位置随机扰动（position 扰动 0.02-0.2m，orientation 全范围 ±180°）
5. motion planner 重新算一条 recovery 路径，robot 追过去
6. 一条 trajectory 最多扰动 3 次

**为什么有效**:

第一个原因是 **state coverage 扩大了**。BaseSim 的物体位置全堆在 workspace 中心，orientation 集中在 $[0°, 180°]$。ADSim 打散到整个 workspace，orientation 均匀分布。policy 见过的 configuration 多了，泛化自然好。

第二个原因是 **强制学会 closed-loop**。如果 trajectory 全是顺畅的，policy 本质上在做 open-loop replay——记住一开始看到什么就执行什么动作。但加了 perturbation 后，policy 必须靠 real-time visual feedback 来纠正，否则完不成 task。等于训练时就逼它学会"看一眼 → 调整动作"的闭环控制。

这个 idea 看着简单，但比 domain randomization 聪明——不是全局乱随机，只在 critical moment 扰动，既有 "正常路径" 也有 "recovery 路径"，数据更 structured。

---

### 第三道坎：不懂真实物理

**问题**: sim 里的摩擦系数、质量、接触动力学都和 real 有微妙的差异。policy 在 sim 里学到的 "看到这个画面就执行这个动作" 的 mapping，搬到 real 会有偏差。

**药方**: **Sim-and-real co-training**。

做法很朴素：

$$\mathcal{L}_{\mathcal{D}^\alpha} = \alpha \cdot \mathcal{L}_{\mathcal{D}_s} + (1-\alpha) \cdot \mathcal{L}_{\mathcal{D}_r}$$

人话：
- $\mathcal{D}_s$: sim 数据（几千条）
- $\mathcal{D}_r$: real 数据（35 条 human demo）
- $\alpha$: 训练时从 sim 采样的概率，设成接近 1（比如 0.9）
- 每个 training step，90% 概率从 sim 采 batch，10% 概率从 real 采
- 两个 loss 加起来一起 backprop

**为什么有效**:

Sim data 量大、覆盖广，提供丰富的 supervision signal。Real data 量少但物理真实，把 policy "锚" 到真实动力学上。混合训练，policy 学到的 representation 是 **domain-invariant** 的——不管 sim 还是 real，同样 visual pattern 映射到同样 action。

对比传统方法：
- Domain randomization: 在 sim 里疯狂随机化参数，希望覆盖 real 的参数。缺点是调参靠经验，而且牺牲 fidelity
- System identification: 精确测量 real 的物理参数，填回 sim。缺点是 tedious 且不完美
- Co-training: 啥都不调，直接混数据，让 policy 自己找出两个 domain 共有的 invariant feature

---

## 实验怎么验证的

### Task

**Deep-bin picking**: 从一个很深的 bin 里抓一个 red plug，放到旁边的 bin 里。

为什么选这个 task？因为难。深 bin 意味着机械臂要伸进去，joint limit 和 collision 问题严重。物体靠 bin 壁的时候更麻烦。这种 task 对 sim-only 训练的 policy 来说就是地狱难度。

### Metrics

传统就一个 binary success rate，但这个 metric 太粗。"完全没靠近目标" 和 "抓到了但放歪了" 都算 failure，但显然 capability 不一样。所以 paper 设计了三个 metric：

| Metric | 含义 | 衡量什么 |
|---|---|---|
| **TAR** | Target Alignment Rate，成功到达 bottleneck pose 的比例 | perception + approach 能力 |
| **SR1** | First-Attempt Success Rate，一次 attempt 就成功的比例 | 开环执行能力 |
| **SR3** | 3 次内成功的比例 | 闭环恢复 + 整体完成能力 |

这三个 metric 有 hierarchy 关系：SR1 成功必然 TAR 成功，SR3 ≥ SR1。

### 实验结果

**Zero-shot**（纯 sim 训练，直接部署）：

| Data | Policy | TAR | SR1 | SR3 |
|---|---|---|---|---|
| BaseSim | ACT | 10% | 5% | 5% |
| ADSim | ACT | 45% | 10% | 15% |
| 3DGS-ADSim | ACT | 55% | 20% | 25% |
| BaseSim | π0 | 45% | 45% | 55% |
| ADSim | π0 | 75% | 60% | 70% |
| 3DGS-ADSim | π0 | 80% | 60% | 75% |

看这个表的几个 takeaway：

1. **每个 module 都有贡献**。BaseSim → ADSim（加对抗扰动）TAR 涨 35%，ADSim → 3DGS-ADSim（加高保真渲染）再涨 5-10%。三个模块是 additive 的。

2. **π0 碾压 ACT**。同样数据，π0 比 ACT 高 25-55%。π0 是 VLA foundation model，pre-trained 见过海量数据，有强大的 visual prior。这说明 foundation model 的 prior + 高质量 synthetic data 是 **乘法关系**，互相放大。

3. **SR1 和 SR3 的 gap 说明 retry 有用**。比如 3DGS-ADSim + π0，SR1=60% 但 SR3=75%，说明 15% 的 trial 是靠 retry 救回来的。这反映 policy 有一定的 error recovery 能力。

**Few-shot co-training**（加 35 条 real demo）：

| Data | Policy | TAR | SR1 | SR3 |
|---|---|---|---|---|
| Real35 only | π0 | 85% | 70% | 70% |
| Real35&ADSim | π0 | 90% | 65% | 85% |
| Real35&3DGS-ADSim (HyperSim) | π0 | 95% | 75% | 95% |

最反直觉的结果：**加 sim data 比只用 real data 还好**。Real35 only SR3=70%，HyperSim SR3=95%。

这说明 sim data 没有稀释 real data，反而提供了 complementary information——更广的 state coverage、recovery behavior。Real data 提供物理 grounding，sim data 提供规模和多样性，1+1>2。

**Dynamic robustness**（inference 时人为扰动物体）：

| Data | Policy | TAR | SR1 |
|---|---|---|---|
| Real35&BaseSim | π0 | 30% | 25% |
| Real35&ADSim | π0 | 80% | 60% |
| Real35&3DGS-ADSim | π0 | 80% | 60% |

没用 adversarial 的，SR1 只有 25%。用了 adversarial 的，SR1 60%。**35% 的提升**。

这直接验证了 adversarial training 的价值——训练时学的 recovery skill 迁移到了 inference 时的真实扰动处理。

注意 ADSim 和 3DGS-ADSim 在 robustness 上没差异（都 60%），说明 visual fidelity 对 dynamic robustness 贡献不大，robustness 主要来自 adversarial perturbation 机制。这合理，dynamic perturbation 考验的是 closed-loop control，不是 perception。

---

## 整体 takeaway

### 1. Sim-to-real 是系统工程，不是单点突破

三个 module 单独用都不够：
- 只加 3DGS（视觉真）但没 adversarial（coverage 窄）→ policy overfit nominal path
- 只加 adversarial（coverage 广）但 sim 画面太假 → visual feature 对不上
- 只加 co-training 但 sim data 质量差 → 混进来反而污染

三个一起上，才能从 5% SR3 到 95% SR3。

### 2. Foundation model 和 synthetic data 互相成就

π0 在 BaseSim 上 55%，在 HyperSim 上 95%。同样 model，data 质量决定上限。Foundation model 提供通用 prior，high-quality synthetic data 激活这个 prior。

这和 LLM 里 pre-train + instruction tuning 的 synergy 一模一样：pre-train 给你世界知识，instruction tuning 告诉你怎么用这些知识完成具体 task。

### 3. Adversarial training 教会 policy "recovery"

传统 trajectory 只教 policy "正常怎么做"，adversarial perturbation 教它 "出错了怎么办"。这两个 skill 是正交的，policy 需要同时具备。35% 的 robustness 提升说明 recovery skill 能从 sim 迁移到 real。

### 4. Synthetic data 是 real data 的放大器

35 条 real demo 只有 70% SR3。加一堆 sim data，SR3 到 95%。Sim data 不是替代 real data，是 **放大** real data 的效用——用 real data 提供物理 grounding，用 sim data 提供规模和多样性，最后效果 > 纯 real。

---

## Reference

- 3D Gaussian Splatting: https://doi.org/10.1145/3592433
- GPGS (geometry-aware GS, IROS 2025): paper 里 reference [24]
- MimicGen (data augmentation 对比): https://arxiv.org/abs/2310.17596
- RoboTwin 2.0 (piecewise generation 对比): https://arxiv.org/abs/2506.18088
- Sim-and-real co-training (RSS 2025): https://arxiv.org/abs/2503.22634
- π0.5 VLA model: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- O3DE engine: https://docs.o3de.org
- RoboCasa (大规模 sim 数据): RSS 2024

---

# HyperSim 深度解析

这篇 paper 来自 Huawei CloudRobo Lab， tackling 一个 robotics 里面的经典难题——sim-to-real gap。让我从架构直觉出发，逐层拆解。

---

## 1. Overall Framework 直觉

HyperSim 的核心 design philosophy 是 **two-layer architecture**：

- **Base Layer**: 标准 data-to-policy pipeline（合成数据 → 训练 → 部署）
- **Enhancement Layer**: 三个可插拔模块，分别针对 sim-to-real gap 的三个维度
  - Visual fidelity gap → Geometry-aware 3D Gaussian Splatting
  - State-action coverage gap → Adversarial perturbation-and-recovery
  - Representation gap → Sim-and-real co-training

这里有一个关键的 insight：sim-to-real failure 不是单一原因造成的，是 environment complexity mismatch + data distribution skew + visual/physics discrepancy 三者耦合的结果。所以 paper 强调 holistic framework，避免单独优化某一维度时被其他维度 bottleneck。

---

## 2. High-Fidelity Environment Construction

### 2.1 Hybrid Scene Decomposition

这里采用 **foreground + background decoupling** 的策略：

**Foreground (manipulation zone)**: 用 18 个 spatial relation constraints 生成物理可交互的场景。这 18 个 solver 分三类：
- Unary geometric priors: `scale`, `pose2D`, `pose3D`
- Explicit pairwise relations: `place_on_surface`, `place_left_edge`, `place_to_left` 等 12 个
- Implicit multi-object formations: `random_placement`, `no_overlapping`, `with_obstacles`

这种 constraint-based 生成比纯 randomization 更可控，能精确建模物体间的空间关系。

**Background (non-interactive)**: 用 GPGS [24] 重建 photorealistic 背景。这里的关键是 GPGS 利用 fused LiDAR scans 提供的 geometric priors 来约束 Gaussian primitives 的空间分布。

### 2.2 3D Gaussian Splatting 数学细节

公式 (1) 定义了单个 Gaussian primitive：

$$\mathcal{G}(\mathbf{p}) = \exp\left(-\frac{1}{2}(\mathbf{p}-\mathbf{p}_k)^T \Sigma^{-1} (\mathbf{p}-\mathbf{p}_k)\right)$$

变量解释：
- $\mathbf{p} \in \mathbb{R}^3$: 3D 空间中任意 query point 的坐标
- $\mathbf{p}_k \in \mathbb{R}^3$: 第 $k$ 个 Gaussian primitive 的中心位置 (mean/position)
- $\Sigma \in \mathbb{R}^{3\times 3}$: 协方差矩阵，正定，控制 Gaussian ellipsoid 的形状和朝向。可以分解为 $\Sigma = R \cdot S \cdot S^T \cdot R^T$，其中 $R$ 是 rotation matrix（朝向），$S$ 是 scaling matrix（沿各轴的尺度）
- $\Sigma^{-1}$: 协方差矩阵的逆，即 precision matrix
- $\frac{1}{2}$ 系数: 来自 multivariate Gaussian density 的标准形式

这个形式本质是各向异性 3D Gaussian，在空间中表现为一个 ellipsoid。

公式 (2) 是 volumetric rendering 的核心——alpha blending：

$$\mathbf{c}(\mathbf{x}) = \sum_{k=1}^{K} \mathbf{c}_k \cdot o_k \cdot \mathcal{G}^{2D}(\mathbf{x}) \cdot \prod_{j=1}^{k-1}\left(1 - o_j \cdot \mathcal{G}^{2D}(\mathbf{x})\right)$$

变量解释：
- $\mathbf{x} \in \mathbb{R}^2$: 像素坐标
- $\mathbf{c}(\mathbf{x}) \in \mathbb{R}^3$: 像素 $\mathbf{x}$ 最终的 RGB 颜色值
- $\mathbf{c}_k$: 第 $k$ 个 Gaussian 的 view-dependent 颜色（由 spherical harmonics coefficients 表示，通常用 3 阶 SH，共 48 维：16 coefficients × 3 channels）
- $o_k \in [0,1]$: 第 $k$ 个 Gaussian 的 opacity
- $\mathcal{G}^{2D}(\mathbf{x})$: 3D Gaussian 投影到 image plane 后在像素 $\mathbf{x}$ 处的值（通过 Jacobian of projective transformation 近似）
- $K$: 沿该 pixel 的 viewing ray 排序后影响该 pixel 的 Gaussian 数量
- $\prod_{j=1}^{k-1}(1 - o_j \mathcal{G}^{2D}(\mathbf{x}))$: **transmittance term**，表示光线穿过前 $k-1$ 个 Gaussian 后剩余的能量比例

这个公式的物理直觉：从相机发出光线，依次穿过按 depth 排序的 Gaussian primitives，每经过一个 Gaussian，光线能量被部分吸收（opacity × Gaussian 值），剩余能量继续传播。最终 pixel 颜色是所有 Gaussian 贡献的加权求和，权重由 opacity 和 transmittance 共同决定。

### 2.3 Gaussian-Mesh Hybrid Representation

这是 paper 的一个技术亮点。单纯 3DGS 有一个问题：geometric mesh 噪声大、scale 估计不准，对 contact-rich simulation 不友好。HyperSim 的解决方案：

1. 用 GPGS 生成 Gaussian representation（负责 photorealistic rendering）
2. 用 rendered color + depth maps 跑 **TSDF (Truncated Signed Distance Function)** fusion，生成 colorized mesh
3. Gaussian 和 mesh 严格 spatially aligned

这样物理引擎（如 O3DE）可以用 mesh 做 collision detection 和 contact dynamics，同时用 Gaussian 做 visual rendering，两者共享同一套几何 backbone。

**Intuition**: 这种 design 解耦了 "looking real" 和 "feeling real" 两个目标。Gaussian 擅长前者，mesh 擅长后者，hybrid 让它们各司其职。

---

## 3. Adversarial Trajectory Generation

### 3.1 Piecewise Decomposition via Bottleneck Pose

核心概念是 **bottleneck pose**：定义为 TCP (Tool Center Point) 进入以 target object center $\mathcal{O}_A$ 为中心、半径为 $d$ 的 hemisphere 时的 configuration。

坐标系定义：
- $\mathcal{F}_A$ (object frame): origin 在 target object 的 center of mass，xy-plane 平行于 workspace surface
- $\mathcal{F}_T$ (tool frame): origin 在 TCP
  - x-axis: 指向 fingertips 方向
  - y-axis: 沿 gripper stroke 连接两个 fingers
  - z-axis: 右手定则确定

Bottleneck pose 把一个 subtask 分成两个 phase：
1. **Approaching primitive**: 从 initial state 到 bottleneck pose，用 motion planner 生成 collision-free 路径
2. **Interaction primitive**: 从 bottleneck pose 开始的 contact 和 manipulation，用 IK solver + gripper controller

这种 decomposition 的好处：approaching 阶段是 free-space motion，可以用 sampling-based planner（如 RRT*）高效解决；interaction 阶段需要精确 control，用 IK + controller 更合适。

### 3.2 Adversarial Perturbation Mechanism

关键 design：当 gripper 到达 bottleneck pose 时，**突然对 target object 的 state（translation + rotation）施加扰动**，迫使 motion planner 重新计算 recovery trajectory。

扰动参数（来自 Section IV-A）：
- 2D position 扰动: $\Delta p \sim \mathcal{U}[0.02, 0.2]$ m（每个分量独立采样）
- Orientation 扰动: $\Delta \theta \sim \mathcal{U}[-180°, 180°]$（全范围）
- 每条 trajectory 最多 3 次 interventions（平衡 diversity 和 trajectory length/stability）

这个机制有**双重作用**：

**作用 1: 扩展 state-action coverage**

从 Fig. 7 可以看到，BaseSim 的 2D pose 分布集中在 workspace center，形成一个 elongated pattern；orientation 分布偏向 $[0, 180°]$。ADSim 把分布扩展到整个 workspace，orientation 也更 uniform。

数学上，假设 BaseSim 的 target state 分布是 $p_{\text{base}}(s)$，ADSim 通过扰动引入了一个 proposal distribution $q(s|s_0)$，使得最终分布变为：

$$p_{\text{AD}}(s) = \int p_{\text{base}}(s_0) \cdot q(s|s_0) \, ds_0$$

由于 $q$ 是 uniform-like 的，$p_{\text{AD}}$ 比 $p_{\text{base}}$ 更平坦，coverage 更广。

**作用 2: 强制 closed-loop visuo-motor learning**

纯 demonstration 训练的 policy 容易 overfit 到 initial state，变成 open-loop replay。Adversarial perturbation 在 bottleneck pose 处突然改变 target，policy 必须 **利用 real-time visual feedback** 重新定位。这相当于在训练时就强制 policy 学会 closed-loop correction。

Fig. 5 用 cup-placement 任务展示了这个流程：subtask 1 (grasp cup) → bottleneck → perturbation → recovery → subtask 2 (place cup) → bottleneck → perturbation → recovery → 完成。

**Intuition**: 这是一种 **data augmentation via environmental dynamics**，类似于 domain randomization，但更 targeted——只在 critical moment (bottleneck) 扰动，而不是全局 randomize。这样生成的 trajectory 既有 "正常" 的部分（learning nominal behavior），又有 "recovery" 的部分（learning robustness），比纯 randomization 更 sample-efficient。

---

## 4. Sim-And-Real Co-Training

### 4.1 Formulation

公式 (3):

$$\mathcal{L}_{\mathcal{D}^\alpha} = \alpha \cdot \mathcal{L}_{\mathcal{D}_s} + (1-\alpha) \cdot \mathcal{L}_{\mathcal{D}_r}$$

变量解释：
- $\mathcal{D}_s$: simulation dataset，$|\mathcal{D}_s| \gg |\mathcal{D}_r|$（数量级差异）
- $\mathcal{D}_r$: real-world dataset
- $\mathcal{L}_{\mathcal{D}_s}$, $\mathcal{L}_{\mathcal{D}_r}$: 分别在两个 dataset 上的 behavioral cloning loss
- $\alpha \in [0, 1]$: **co-training ratio**，表示从 simulation data 采样的概率
- $\alpha = 1$: 纯 simulation training（zero-shot deployment）
- $\alpha < 1$（接近 1）: few-shot，混入少量 real data

实现上，每个 training step 以概率 $\alpha$ 从 $\mathcal{D}_s$ 采样一个 batch，以概率 $1-\alpha$ 从 $\mathcal{D}_r$ 采样一个 batch，分别计算 loss 并 backprop。

### 4.2 为什么 Co-Training 有效

Co-training 的哲学和 domain randomization / system identification 不同。后两者试图 **显式消除** sim-real 差异，co-training 则把 sim 和 real data 当作 generic data 混合采样，让 policy 自己 **learn domain-invariant features**。

这种 implicit alignment 的好处：
- 不需要 domain expertise 调 randomization range
- 不需要精确的 dynamics parameter identification
- 自然处理 visual + dynamics 双重 gap

从 information theory 角度，sim data 提供大量 diverse supervision signal，real data 提供 ground-truth physical dynamics。混合训练让 policy 在 sim 的 rich supervision 下学到 generalizable representation，同时被 real data "anchored" 到真实 dynamics。

---

## 5. Experiment Design 详解

### 5.1 三个 Fine-Grained Metrics

这是 paper 的一个 methodological contribution。传统 binary success rate 把所有 failure 等同对待，但实际上 "抓到目标但放置失败" 和 "完全没接近目标" 的 capability level 完全不同。

- **TAR (Target Alignment Rate)**: end-effector 成功从 initial state 导航到 target 的 bottleneck pose 的比率。衡量 **perception + approach** 能力。
- **SR1 (First-Attempt Success Rate)**: 单次 continuous attempt 成功完成的比率。衡量 **开环执行能力**。
- **SR3 (Overall Success Rate)**: 允许 3 次 retry 内成功的比率。衡量 **闭环恢复能力 + 整体 task completion**。

这三个 metric 形成 capability hierarchy: TAR > SR1 > SR3 严格包含关系（SR1 成功必然 TAR 成功，SR3 成功必然 SR1 或通过 retry 成功）。

### 5.2 Task: Deep-Bin Picking

Task design 很关键：把 target object (如 red plug) 从 central deep bin 转移到 adjacent bins。相比 flat-surface manipulation，deep-bin 有两个挑战：
1. **Kinematic constraints**: 机械臂需要深入 bin 内部，joint limits 和 singularity 问题更严重
2. **Collision risks**: 尤其 object 靠近 bin 壁的 corner cases

这解释了为什么 BaseSim + ACT 只有 5% SR3——这种 task 对 sim-only 训练的 policy 来说太难了。

### 5.3 Evaluation Protocol

固定 20 个 trial，每个 trial 有不同的 target pose (translation + rotation) 和 visual distractors。所有 policy 用 **完全相同** 的 evaluation set，确保公平比较。总共 400+ real-world trials。

---

## 6. 实验结果深度分析

### 6.1 Zero-Shot 结果 (Table I)

| Training Data | Policy | TAR | SR1 | SR3 |
|---|---|---|---|---|
| BaseSim | ACT | 10% | 5% | 5% |
| ADSim | ACT | 45% | 10% | 15% |
| 3DGS-ADSim | ACT | 55% | 20% | 25% |
| BaseSim | π0 | 45% | 45% | 55% |
| ADSim | π0 | 75% | 60% | 70% |
| 3DGS-ADSim | π0 | 80% | 60% | 75% |

**关键发现 1: 渐进式 module 贡献**

BaseSim → ADSim：TAR 提升巨大（ACT: 10%→45%，π0: 45%→75%）。这印证了 Section 3.2 的分析——adversarial perturbation 强制 policy 学会 closed-loop visuo-motor alignment，从 open-loop replay 变成 closed-loop control。

ADSim → 3DGS-ADSim：所有 metric 提升 ~10%。这是 visual fidelity 的贡献。BaseSim/ADSim 的 "clean background" 和 real-world 的 cluttered background 有巨大 visual gap，3DGS 重建的 photorealistic background 缩小了这个 gap。

**关键发现 2: Foundation Model 的 synergy**

π0 比 ACT 高 25-55%。π0 是 vision-language-action foundation model，pre-trained on internet-scale data，有强大的 visual generalization prior。这说明：

$$\text{Real Performance} \propto f(\text{Data Quality}) \times g(\text{Model Capacity})$$

两者不是简单叠加，是 **乘法关系**。高质量 synthetic data 给 foundation model 的 prior 提供了 task-specific 的 grounding，foundation model 的 prior 反过来让 synthetic data 更容易被 "理解"。

### 6.2 Few-Shot Co-Training (Table II & III)

加入 35 条 human demonstrations 后：

| Training Data | Policy | TAR | SR1 | SR3 |
|---|---|---|---|---|
| Real35 | π0 | 85% | 70% | 70% |
| Real35&ADSim | π0 | 90% | 65% | 85% |
| Real35&3DGS-ADSim (HyperSim) | π0 | 95% | 75% | 95% |

**关键发现**: HyperSim (SR3=95%) > Real35 only (SR3=70%)。

这是一个 **counter-intuitive** 的结果：加入 synthetic data 后，性能 **超过** 只用 real data 训练的 policy。这说明 synthetic data 不是简单 "稀释" real data，而是提供了 **complementary information**——更多样的 state coverage、recovery behavior、visual context。

从 Table VI 可以看到 scaling behavior:
- Real10: SR3=45% → Real10&ADSim: SR3=55%
- Real20: SR3=45% → Real20&ADSim: SR3=65%
- Real35: SR3=70% → Real35&ADSim: SR3=85%

Co-training 的增益随 real data 增加而扩大（10%→20%→15%），说明 co-training 不仅在小 data regime 有效，在 data scaling 时也保持 synergy。

### 6.3 Dynamic Robustness (Table IV)

这是验证 H3 的实验——在线 inference 时人为扰动 target object。

| Training Data | Policy | TAR | SR1 |
|---|---|---|---|
| Real35&BaseSim | π0 | 30% | 25% |
| Real35&ADSim | π0 | 80% | 60% |
| Real35&3DGS-ADSim | π0 | 80% | 60% |

**关键发现**: Adversarial training 带来 **35% SR1 提升**（25%→60%）。

这个结果直接验证了 Section 3.2 的 design intent——训练时注入的 perturbation-and-recovery behavior 迁移到了 inference 时的 dynamic perturbation handling。Policy 学到了一种 **recovery skill**，而不仅仅是 nominal task execution。

注意 ADSim 和 3DGS-ADSim 在 dynamic robustness 上 **没有差异**（都是 SR1=60%），说明 visual fidelity 对 dynamic robustness 贡献不大——这是合理，因为 dynamic perturbation 主要考验 closed-loop control，而 visual gap 影响的是 perception accuracy。

---

## 7. 整体 Intuition 与更广的思考

### 7.1 为什么 Holistic Framework 重要

传统 sim-to-real 方法往往 **局部优化**：
- Domain randomization 解决 visual gap，但牺牲 fidelity
- System identification 解决 dynamics gap，但需要 expertise
- Data augmentation 解决 coverage gap，但可能产生 non-smooth trajectory

HyperSim 的贡献在于认识到这些 gap 是 **耦合** 的。比如只解决 visual gap（用 3DGS）但不解决 coverage gap（没有 adversarial），policy 依然 overfit 到 nominal trajectory；只解决 coverage gap 但 visual gap 大，policy 无法 ground visual features。

从 Table I 的 ablation 可以清晰看到这种 **additive + synergistic** 效应：
- BaseSim → ADSim: +40% TAR (coverage 贡献)
- ADSim → 3DGS-ADSim: +10% TAR (visual 贡献)
- 加上 co-training: 再 +15% TAR (representation 贡献)

### 7.2 与相关工作对比

- **MimicGen [9]**: data augmentation via spatial transformation，容易产生 non-smooth trajectory。HyperSim 用 piecewise generation + adversarial perturbation，更可控。
- **RoboTwin [12]**: piecewise generation，但没有 adversarial mechanism。HyperSim 在此基础上加了 perturbation-recovery。
- **RoboGen [8]**: LLM-driven task decomposition，但 scene fidelity 有限。HyperSim 用 GPGS 解决 visual fidelity。
- **Sim-and-real co-training [22]**: HyperSim 直接采用这个 idea，但配合 high-fidelity synthetic data，让 co-training 更有效。

### 7.3 Limitations

Paper 自己承认：
1. 只在 single humanoid embodiment (Galaxea R1) 上验证
2. 只测了一个 task suite (deep-bin picking)
3. Hardware safety constraints 限制了更激进的实验

潜在 extension 方向：
- Multi-embodiment: 不同 morphologies 的 robot
- More complex tasks: assembly, tool use, deformable object manipulation
- 自动化 perturbation scheduling: 当前是 fixed 3 次，可以做成 curriculum learning
- Learned perturbation: 用 RL 或 bandit 学习在哪里、怎么 perturb 最 effective

### 7.4 我的 Intuition 总结

这篇 paper 给我的最大启发是：**sim-to-real 不是单一技术问题，是系统工程问题**。三个模块各自不是 novel（3DGS、adversarial training、co-training 都有 prior work），但 **组合起来** 在 challenging task (deep-bin) 上达到 95% SR3，这个 systematic effect 是单独模块无法实现的。

另一个 insight 是 foundation model 和 synthetic data 的 **multiplier effect**。π0 在 BaseSim 上只有 55% SR3，在 HyperSim 上达到 95%——40% 的提升说明 foundation model 的 potential 需要 high-quality data 来 unlock。这和 LLM pre-training + instruction tuning 的 synergy 异曲同工：pre-trained model 提供通用 prior，高质量 task-specific data 激活这种 prior。

---

## Reference Links

- 3D Gaussian Splatting 原始 paper: https://doi.org/10.1145/3592433
- GPGS (Geometry-aware Gaussian Splatting): referenced in IROS 2025
- MimicGen: https://arxiv.org/abs/2310.17596
- RoboGen: https://arxiv.org/abs/2311.01455
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- Sim-and-real co-training: https://arxiv.org/abs/2503.22634
- π0.5 VLA model: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- RDT-1B: https://arxiv.org/abs/2410.07864
- O3DE Engine: https://docs.o3de.org
- InternData-A1: https://arxiv.org/abs/2511.16651
- RoboCasa: referenced in RSS 2024
- Coarse-to-fine imitation: Johns, ICRA 2021
