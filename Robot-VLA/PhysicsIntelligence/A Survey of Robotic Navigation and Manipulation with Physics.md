---
source_pdf: A Survey of Robotic Navigation and Manipulation with Physics.pdf
paper_sha256: 9ca3c45eba1066a7663a6ba9e8d3d0dda00ec27c6e0caeaf7b9489ba08aac7a4
processed_at: '2026-07-17T21:32:17-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# A Survey of Robotic Navigation and Manipulation with Physics Simulators in the Era of Embodied AI — 深度解读

这篇 paper 是 City University of Hong Kong / University of Melbourne / Universität Hamburg / Agile Robots / KU Leuven / TUM 几个组合作写的 survey，核心论点是：**在 Embodied AI (EAI) 的 navigation 和 manipulation 两条主线里，physics simulator 是 sim-to-real transfer 的 bottleneck，同时 method 上正在从 RL → IL → Diffusion Policy → VLA 的方向快速演化**。下面我把里面所有技术细节展开讲，并且补一些 paper 没明说但我认为对 build intuition 很关键的物理/数学动机。

---

## 1. Why this survey, why now

Karpathy 你肯定熟悉这种 pattern：一个领域积累到一定密度，就会有人出来做 taxonomy。这篇 survey 的差异点在于它**把 simulator 当 first-class citizen**，而不只是把方法列一遍。

传统 survey 比如 Hwang et al. [70] (1992), Yasuda et al. [190] (2020), Zhu et al. [208] (2021) 这些是 LLM/world model 出现之前的，所以视野完全是 RL-centric。Liu et al. [107] (2024) 的 "Aligning cyber space with physical world" 是范围更广的 EAI survey，但 navigation 和 manipulation 不深。Zheng et al. [204] 只做 manipulation methods，不碰 differentiable physics。

这篇 survey 的真正价值在于**把 simulator 的 property（physics engine, rendering, dynamics, GPU 加速）和 task 的复杂度绑定**：grasping 用 2D + gripper 就够，dexterous manipulation 必须 3D + multi-finger hand + MuJoCo 这类能算 multi-point contact 的引擎。这种"task complexity → perception granularity → simulator 选择"的耦合关系是这篇 paper 最 useful 的 framing。

参考链接：
- Survey 项目页（如果有）: https://github.com/likheng/Awesome-Embodied-AI-with-Physics-Simulators
- Liu et al. EAI survey: https://arxiv.org/abs/2407.06886
- Zheng et al. manipulation survey: https://arxiv.org/abs/2408.11537

---

## 2. Navigation 部分

### 2.1 Navigation 的两个 sim-to-real gap

这是整篇 paper 最关键的概念区分。Figure 3 把 navigation 拆成四步：**Perception → Memory Building → Decision Making → Action Execution**，每一步都暴露不同的 gap：

1. **Visual sim-to-real gap**: simulator 渲染的 RGB-D 和真实相机有差距（光照、exposure、lens distortion、sensor noise）。这影响 perception module 的迁移。
2. **Physics sim-to-real gap**: 不平地形、friction、collision dynamics 不准，locomotion policy 在 sim 里学到的 control command 到 real 就翻车。这影响 action execution module。

直觉上可以这样想：visual gap 是"看见"的问题，physics gap 是"踩上去/撞上去"的问题。Navigation 这两件事都得对，manipulation 主要 bottleneck 是 physics gap（因为 manipulation 本质就是 physics interaction）。

### 2.2 Navigation Simulators 分类

按 environment scale 分三类：

**Indoor simulators**：
- **Matterport3D Simulator** [18]: 用真实 3D scan，10,800 panoramic views，194,400 RGB-D images，90 scenes。**没有 physics engine**，只能在 precomputed viewpoint 之间离散跳（平均 2.25m 间距），靠 walkable path 隐式做 collision detection。这种 simulator 适合做 visual grounding 研究，但 locomotion research 完全用不了。
  
- **Habitat-Sim** [114, 132, 161]: 渲染速度 10,000+ FPS on GPU。Bullet physics engine 做 rigid-body dynamics。集成 RGB-D noise model（distortion 之类）。和 Habitat-Lab 配合支持 PointGoal / ObjectNav / ImageNav / VLN / EQA。这个是 navigation 圈事实标准的 simulator。https://aihabitat.org/

- **AI2-THOR** [87]: Unity3D 的 Physically-Based Rendering (PBR)，支持 domain randomization（material、lighting 抖动）。Unity physics engine。ProcTHOR [36] 扩展程序化生成 10,000 个 house。https://ai2thor.allenai.org/

- **iGibson** [92, 176, 177]: PBR + BRDF（Bidirectional Reflectance Distribution Function，描述光在 surface 的 angle-dependent reflectivity）。Bullet engine。15 个 indoor scene，可扩展 CubiCasa5K。iGibson 的特色是 **Interactive Navigation**：agent 要 push 物体才能通过 cluttered scene。http://svl.stanford.edu/igibson/

**Outdoor simulators**：
- **CARLA** [38]: 自动驾驶专用。Unreal Engine 4 的 PhysX，ray tracing + PBR，CARLA2Real 工具增强 photorealism。https://carla.org/
- **AirSim** [146]: 微软出品，drone + 车。Unreal PhysX + 自定义 fast physics engine。内置 IMU / GPS sensor model。

**General-purpose simulators**：
- **ThreeDWorld (TDW)** [51]: Unity3D + PBR + HDRI lighting。PhysX 支持 cloth + fluid。Transport Challenge 在这里。
- **Isaac Sim** [124]: NVIDIA 的旗舰。RTX ray-traced rendering，PhysX 5。集成 Isaac Lab。warehouse → outdoor。https://developer.nvidia.com/isaac-sim

Table 2 里有个细节很重要：Matterport3D 那行连 dynamics 都没标，因为本质上它是 dataset viewer 不是 simulator。Habitat 只支持 rigid，iGibson 支持 R/S/C（rigid/soft/cloth）。CARLA/AirSim 标了 R° 表示只 focus on rigid body dynamics for autonomous driving。

### 2.3 Navigation Benchmark Datasets

Table 1 是核心。我重点 highlight 几个：

**Goal-driven**:
- **HM3D** [137]: 1,000 scenes，Habitat 平台。PointNav 上 DD-PPO [172] 跑出 97% SR（这其实是 saturated 任务了）。https://aihabitat.org/datasets/hm3d/
- **HM3D-OVON** [192]: 181 scenes，open-vocabulary ObjectNav。DAgRL+OD 37.1-39.0% SR。open-vocab 才是真正 unsolved 的方向。
- **MultiON** [53]: 50,000 train episodes，要按顺序访问多个 object。OracleMap 94% SR (1-ON), 48% SR (3-ON)。可见 multi-goal 难度断崖式上升。
- **DivScene** [169]: 4,614 scenes, 81 scene types，用 GPT-4 驱动的 HOLODECK [188] 程序化生成，BFS 产生 expert trajectory。NATVLM 54.94% SR。这个 dataset 体现了 LLM-as-environment-generator 的新范式。

**Task-driven**:
- **R2R** [4]: 90 scenes, 21,567 instructions，Matterport3D 平台。SEQ2SEQ 20.4% SR。这是 VLN 的奠基 dataset。https://bringmeaspoon.org/
- **VLN-CE** [88, 89]: 把 R2R 的离散环境搬到 continuous Habitat。Cross-modal attention 33% SR。
- **VLN-CE-Isaac** [24]: VLN-CE 适配到 Isaac Sim 给 legged robot，加 LiDAR terrain adaptation。NaVILA [24] 54.0% SR。
- **ALFRED** [149]: 120 scenes, 8k demos, 25k directives, 428k image-action pairs。SEQ2SEQ 4% SR（任务太难）。https://askforalfred.com/
- **TEACh** [126]: 120 scenes, 3,047 sessions，带 dialogue + segmentation。E.T. 9% SR。https://teach-benchmark.github.io/
- **Robo-VLN** [72]: 90 scenes, 3177 trajectories，专门给 robot embodiment。HCM 46% SR。

注意 SR 数字差距巨大：PointNav 接近满分，VLN/EQA 才 30-50%，household long-horizon 任务才 9%。这告诉你 navigation field 的真实 frontier 在哪。

### 2.4 Evaluation Metrics — 公式逐变量拆解

**SPL (Success weighted by Path Length)** [3], Equation (1):

$$\mathrm{SPL} = \frac{1}{N} \sum_{i=1}^{N} S_i \frac{\ell_i}{\max(p_i, \ell_i)}$$

变量含义：
- $N$: 总 episode 数
- $S_i \in \{0, 1\}$: 第 $i$ 个 episode 的 success indicator（达到 goal 为 1）
- $\ell_i$: 第 $i$ 个 episode 的 shortest path length（geodesic distance，参考路径）
- $p_i$: 第 $i$ 个 episode agent 实际走的 path length

直觉：单纯算 SR 太宽松，agent 可以绕一大圈最后到达。SPL 用 $\frac{\ell_i}{\max(p_i, \ell_i)}$ 作为 penalty factor：如果 $p_i \leq \ell_i$（不可能，shortest 是下界），如果 $p_i \gg \ell_i$ 这个 ratio 趋近 0，成功也被惩罚掉。$\max$ 是为了防止 agent 找到比 reference path 更短的捷径时 ratio > 1。

**EQA Efficiency** [113], Equation (2):

$$E = \frac{1}{N} \sum_{i=1}^{N} \frac{(\sigma_i - 1)}{4} \times \frac{\ell_i}{\max(\rho_i, \ell_i)} \times 100\%$$

变量：
- $N$: episode 数
- $\sigma_i \in [1, 5]$: LLM-Match score，给答案打分 1-5
- $\ell_i$: shortest path
- $\rho_i$: agent 实际 path

直觉：$\frac{\sigma_i - 1}{4}$ 把 1-5 分归一化到 [0,1]，再乘 path efficiency ratio $\frac{\ell_i}{\max(\rho_i, \ell_i)}$。这个 metric 的设计哲学是：答得对 + 走得短 才算好。如果一个 agent 答对但 explore 半天，分数被压。这种把 correctness 和 efficiency 乘起来的设计在 EQA 这种 "answer after exploration" task 上很自然。

**CLS (Coverage weighted by Length Score)** [73]:

$$\mathrm{CLS}(P, R) = \mathrm{PC}(P, R) \cdot \mathrm{LS}(P, R)$$

- $P$: predicted path
- $R$: reference path
- $\mathrm{PC}$: Path Coverage，预测路径覆盖 reference 的比例
- $\mathrm{LS}$: Length Score，长度匹配度

**nDTW (Normalized Dynamic Time Warping)** [71]: 把 DTW 用在 path 上，输出 [0,1]，越高表示路径在 location + order 上越对齐。

**ISR (Independent Success Rate)** [154]: 把 VLN 的 multi-step instruction 拆成 subtask，每个 subtask 独立打分。

**IVR (Instruction Violation Rate)** [28]:

$$\mathrm{IVR} = \frac{\text{Number of episodes with violations}}{\text{Total number of episodes}}$$

intuition: 测常识违规（jaywalking 之类），这是 safety dimension。Figure 6 用四个 dimension 评估 metric：Scale Invariance / Order Invariance / Safety Compliance / Energy Efficiency。SPL 在 Order Invariance 上弱（它只看终点），nDTW 强（看 path 序列）。

### 2.5 Navigation Methods — Memory 的二分法

这篇 paper 把 navigation method 按 memory type 分，这是很 clean 的 framing。

#### Explicit Memory

**Metric Map-Based**: 把 environment 离散成 grid / point cloud / voxel / mesh。比如 occupancy grid [157] 每个 cell 标 free/occupied。Fu et al. [48] 用 occupancy grid 算 geodesic distance，steepest descent 优化。Chen et al. [22] 用 CNN fuse semantic feature from occupancy map + RGB，再用 RNN 集成时序。Huang et al. [67] 把 CLIP embedding back-project 到 semantic grid map 上，做 open-vocabulary landmark localization（cosine similarity between grid embeddings and CLIP text queries）。

intuition: metric map 显式存空间信息，fine-grained control 容易，但 scale 不行——10km² outdoor 你 grid cell 数爆炸。

**Graph-Based**: topological graph，node = landmark（doorway, intersection），edge = traversable path。Savinov et al. [144] SPTM 用 CNN encode image 成 node feature，算 similarity 规划。Chaplot et al. [152] Neural Topological SLAM，panorama image 作 node。Beeching et al. [11] 用 neural planner 估 node 之间的 connection probability。Yang et al. [187] 用 knowledge graph，node = object category，edge = "next to" / "in" 关系，能 infer unseen object location（mango 大概率在 apple 旁边的 fridge）。

intuition: graph 抽象掉了 precise geometry，只保留 topology，所以 scale 友好，但 fine-grained control 弱。Dijkstra / A* 这种经典算法直接能跑。

#### Implicit Memory

**Latent Representation-Based**: 不存显式数据结构，把 observation + action 历史 encode 成 latent vector，直接 infer action。Zhu et al. [209] cross-modal attention + LSTM 处理 dialog history。Hong et al. [61] VLN-BERT 用 transformer 的 [CLS] token 当 state representation。

这种方法的致命问题：**累积误差**。一步一步从历史 infer，错误 propagate。Moghaddam et al. [117] 的 trick 是先 predict 成功的 future state（goal 或 critical sub-goal），再 plan 到那里——这相当于把 open-loop 转成 closed-loop with sub-goal anchor。

**Foundation Model-Based**: NavGPT [205] 用 GPT-4 zero-shot，把 visual observation 转成 text 描述喂 LLM，LLM autoregressive 输出 action。NaviLLM [201] 用 VLM 直接处理 visual feature + text instruction。MapNav [197] 把 VLM + Annotated Semantic Map 结合，缓解 LLM hallucination 问题。

intuition: LLM 的 pre-trained knowledge 是巨大的 implicit memory，但 fine-grained spatial grounding 差。MapNav 的 trick 是把 explicit metric map 当成"空间事实约束"喂给 VLM，相当于给 hallucination 加 ground truth 锚。

**World Model-Based**: 这是 navigation 的 frontier。Bar et al. [9] Navigation World Model (NWM) 是 video diffusion model，generate potential future frames 用来 evaluate navigation path——从单张 image extrapolate unseen route。X-Mobility [106] latent world model forecast environment dynamics。NVIDIA Cosmos Predict [2] + Cosmos Reason 双模型：Predict 是 video world model，Reason 是 CoT reasoning model，把 video forecast 解释成 navigation advice（比如 Predict 预测有行人横穿，Reason 输出"减速"）。https://arxiv.org/abs/2501.03575

intuition: world model = "imagined simulator"。如果 world model 足够准，agent 可以在想象里 rollout，不用真 simulator。同时 world model 还能 generate synthetic data 缓解 robotics data scarcity。这是 navigation 下一个 paradigm 的候选。

---

## 3. Manipulation 部分

### 3.1 Manipulation Tasks 按 complexity + DoF 排序

Figure 7 是核心 schematic：

**Tasks**（复杂度递增）：
1. **Grasping**: pick-and-place。Planar grasping 3 DoF，full 3D grasping 6 DoF (x,y,z,roll,pitch,yaw)。
2. **Dexterous manipulation**: multi-finger hand，finger gaiting / rolling / pivoting [56, 125]。要建模 multi-point contact + friction，MuJoCo 这类引擎的核心战场。
3. **Deformable object manipulation**: cloth, rope。state space 巨大因为点间相对距离不固定。
4. **Mobile manipulation**: arm mounted on wheeled / quadruped / humanoid base，要 navigation + manipulation 联合。
5. **Open-world manipulation**: "infinite variability problem" [163]，unseen object in unstructured environment。
6. **Fragile object manipulation**: soft gripper (pneumatic [155], hydraulic [206], tendon-driven [168])，precise force control。
7. **Bi-manual manipulation**: ALOHA [199] / Mobile ALOHA [50]，dual-arm coordination。https://mobile-aloha.github.io/

**Hardware**（DoF 递增）：grippers → dexterous hands → soft hands → bimanual → arm on wheeled → quadruped → humanoids。

### 3.2 Physics Engines — Classical vs Differentiable

这是 manipulation 部分最关键的对比。

**Classical Engines**:
- **Gazebo** [86]: ROS 紧密集成，支持 DART / ODE / Bullet 多引擎。OGRE 渲染，没 ray tracing。http://gazebosim.org/
- **PyBullet** [33]: Bullet engine，GPU 加速 + continuous collision detection。LCP (Linear Complementarity Problem) contact model——这个有问题，friction cone 估计不准，影响 physics sim-to-real gap。OpenGL rasterization only。https://pybullet.org/
- **MuJoCo** [165]: contact dynamics 精度之王。generalized coordinates 建模 multi-joint system。soft contact model 有 interpenetration 问题。multi-threading 加速 RL。OpenGL rasterization。**dexterous manipulation 的事实标准**。https://mujoco.org/
- **Isaac Sim** [124] + **SAPIEN** [19, 116, 178]: 这俩是 photorealism 双雄。GPU rasterization + real-time ray tracing。PhysX 5 支持 rigid/soft/cloth/fluid。SAPIEN 还有 built-in depth noise simulation（基于 distance, edge, material property 生成 realistic noise）。https://sapien.ucsd.edu/
- **CoppeliaSim** [140]: 多引擎支持 (MuJoCo, Bullet, ODE, Newton, Vortex)，但没 GPU 加速，渲染 OpenGL + partial ray tracing。http://www.coppeliarobotics.com/

**Differentiable Engines** — 这是 paper 重点 push 的方向：

- **Dojo** [62]: 把 contact simulation formulate 成 optimization problem，用 implicit function theorem 提供 smooth differentiable gradient。直觉上：传统 engine 算 contact 是个 non-smooth 事件（碰撞瞬间力突变），Dojo 把它 relax 成 smooth optimization，这样能 backprop。https://github.com/dojo-sim/Dojo.jl

- **DiffTaichi** [64, 65]: 不同iable编程语言。megakernel approach 把多个 computation stage merge 进一个 CUDA kernel，最大化 GPU utilization。https://github.com/taichi-dev/taichi

- **Genesis** [6]: 基于 DiffTaichi 构建，fully differentiable simulation。**速度比现有 GPU 加速 simulator 快 10-80x**，同时不损失 physical fidelity。还带 ray-tracing rendering 和 generative data engine（natural language → multimodal data 自动生成 training environment）。https://github.com/Genesis-Embodied-AI/Genesis

为什么 differentiable physics 这么重要？intuition 如下：

传统 RL 在 simulator 里通过 trial-and-error 学 policy，gradient 是通过 policy gradient / Q-learning 间接估计的。Differentiable simulator 让你能 **直接 backprop through physics**：从 final state 的 loss 到 initial action 的 gradient，经过 contact / friction / collision 这些物理过程都是 well-defined。这意味着 sample efficiency 巨大提升，sim-to-real gap 也小（因为 gradient 直接对齐物理规律）。

但 differentiable physics 有个 hidden 问题：现实物理本身是 non-smooth 的（碰撞瞬间冲量无穷大）。所以 differentiable simulator 必然要在物理精度上做妥协（soft contact）。这是精度 vs gradient 的 trade-off。

### 3.3 Manipulation Benchmark Datasets

Figure 10 是可视化对比。我 highlight 几个：

**Rigid body**:
- **Meta-World** [193]: 50 个 distinct manipulation task。https://meta-world.github.io/
- **RLBench** [74]: 100 个 task。https://sites.google.com/view/rlbench

**Deformable**:
- **SoftGym** [100]: 10 个 env (Pour Water, Fold Cloth, Straighten Rope)。https://sites.google.com/view/softgym
- **PlasticineLab** [69]: 用 DiffTaichi 做 differentiable soft-body。
- **GRIP** [110]: 1,200 objects，soft + rigid gripper，基于 IPC (Incremental Potential Contact) simulator，给 detailed deformation + stress distribution data。

**Mobile manipulation**:
- **OVMM** [191]: AI Habitat 上，200 scenes, 7,892 objects, 150 categories，任务是把 object 从 start place 搬到 goal place。https://ovmm.github.io/
- **Behavior-1K** [93]: Omnigibson + PhysX 5。1,000 个 household activity，50 scenes，1,900+ object types，9,000+ object models（含 liquid, deformable, transparent）。https://behavior.stanford.edu/
- **ManiSkill-Hab** [150]: ManiSkill3 平台 [162]，30,000+ FPS photorealistic，三个 long-horizon task: Tidy House / Prepare Groceries / Set Table。https://maniskill.github.io/
- **BRMData** [198]: bimanual mobile manipulation，10 个 household task。

**Language-conditioned**:
- **CALVIN** [115]: 34 long-horizon task，每个配 multi-step instruction。https://calvinrobot.github.io/
- **RoboTwin** [119]: LLM 生成 manipulation environment + task，object 从 demo video 重建。
- **RoboMind** [175]: 55,000 real trajectory, 279 tasks, 61 objects, 10,000 linguistic annotations，多 robot embodiment。https://github.com/RoboMinds/RoboMind
- **DROID** [83]: 76,000 real trajectory = 350 小时，564 scenes, 86 tasks。https://droid-dataset.github.io/

**Multi-robot embodiment**:
- **Open X-Embodiment** [31]: **22 种 robot**，527 skills, 160,266 tasks, 1M+ trajectory。**最大的开源 real robot dataset**。https://robotics-transformer-x.github.io/

**Visual perception**:
- **GraspNet-1Billion** [42, 44]: 97,280 images, 88 objects, 1.1B+ grasp pose，每个 image 标 6D pose + grasp point。https://graspnet.net/

注意 Figure 10 的 log scale：Open X-Embodiment 的 trajectory 数（百万级）远超其他，这就是为什么 RT-X / OpenVLA 这种 generalist policy 训得起来。

### 3.4 Manipulation Methods

#### 3.4.1 Perception Representation

Table 3 列了所有公式。我逐个拆：

**Voxel Map**, Equation (3):

$$V(x, y, z) = \begin{cases} 1 & \text{if } V[x, y, z] \text{ is occupied} \\ 0 & \text{if } V[x, y, z] \text{ is unoccupied} \end{cases}$$

intuition: 3D 空间离散成 occupancy grid，每个 voxel 标 0/1。VoxPoser [68] 用 VLM 解析 language instruction 生成 voxel map，highlight task-relevant region（比如"抓杯子把手"就 highlight 把手 voxel）。VoxAct-B [103] 把这思路扩展到 bimanual。https://voxposer.github.io/

**Pose Estimation**, Equation (4):

$$\hat{p} = \arg\max_{\hat{p} \in P} L_{\text{pose}}(\hat{p})$$

- $P$: pose hypothesis 集合
- $L_{\text{pose}}$: pose likelihood function
- $\hat{p}$: 最优 pose 估计

Pix2Pose [129] pixel-wise coordinate regression 从 RGB 直接估 3D coordinate，不需要 textured 3D model。FoundationPose [16] 统一 6D pose estimation + tracking，支持 model-based (CAD) 和 neural implicit (novel view synthesis)。https://nvlabs.github.io/FoundationPose/

**Grasp Proposal**, Equation (5):

$$\hat{g} = \arg\max_{g \in G} L_{\text{grasp}}(g)$$

- $G$: grasp candidate 集合
- $L_{\text{grasp}}$: grasp quality / likelihood
- $\hat{g}$: 最优 grasp

LERF [81] (Language Embedded Radiance Fields) 把 VLM 和 3D scene rep 结合，zero-shot task-specific grasp proposal。LERF-TOGO [138] 用 natural language prompt ("mug handle") query 任务相关 region。F3RM [148] 把 CLIP feature distill 进 3D representation 做 few-shot grasping。GraspSplats [76] 用 Gaussian splatting 实时 grasp proposal。https://lerf.io/

**SO(3)-equivariant**, Equation (6):

$$f(Rx; \theta) = R f(x; \theta), \quad \forall R \in \mathrm{SO}(3)$$

- $x$: input 3D point cloud
- $R \in \mathrm{SO}(3)$: 任意 3D rotation matrix（special orthogonal group, det=1, $R^T R = I$）
- $f(\cdot; \theta)$: neural network with parameter $\theta$
- $f(x; \theta)$: output representation

intuition: 输入旋转 $R$，输出也跟着旋转 $R$。**注意这跟 invariant 不同**——invariant 是 $f(Rx) = f(x)$，输出对旋转无变化。Equivariant 是输出跟着变。这对 manipulation 极其重要：如果杯子转了 30°，最优 grasp point 也应该跟着转 30°，而不是不变。

Vector Neuron Networks (VNNs) [37] 实现这点的核心 trick：把 neuron 从 scalar 扩展到 3D vector。线性层要满足 $W(xR) = (Wx)R$，这等价于 $W$ 在 rotation 下 commute。非线性层（VNN 的 Eq. 9）：

$$v' = \begin{cases} q & \text{if } \langle q, k \rangle \geq 0 \\ q - \langle q, \frac{k}{\|k\|} \rangle \frac{k}{\|k\|} & \text{otherwise} \end{cases}$$

- $q = WV_i$: linear-transformed feature
- $k = UV_i$: 学到的 direction vector
- $\langle \cdot, \cdot \rangle$: inner product
- $\frac{k}{\|k\|}$: unit direction of $k$

intuition: 这是 ReLU 的 rotation-equivariant 推广。普通 ReLU 是 $\max(0, x)$，对 scalar 操作。VNN 把它变成"沿 $k$ 方向 clip"。关键性质：$\langle qR, kR \rangle = \langle q, k \rangle$（inner product rotation-invariant），所以 clipping 的决策不依赖 rotation，整个操作 equivariant。https://github.com/dnlkrm/Vectors

**SE(3)-equivariant**, Equation (7):

$$f(Rx + t; \theta) = R f(x; \theta) + t, \quad \forall (R, t) \in \mathrm{SE}(3)$$

- $(R, t)$: 旋转 + 平移，组成 SE(3) = Special Euclidean group
- $x$: input
- $f(x; \theta)$: output

SE(3) 比 SO(3) 多了 translation。Neural Descriptor Fields (NDFs) [151] 实现：用 VNN 保证 SO(3)-equivariance，再加 mean-center shift:

$$x_n - \frac{1}{N} \sum_{i=1}^{N} x_i$$

intuition: 把 point cloud 减去 centroid。这样不论 object 在哪（translation 任意），相对坐标不变，function 只看 relative transformation。这就把 SE(3) 问题降回 SO(3) 问题 + centroid offset。

Equivariant Descriptor Fields (EDFs) [143] 扩展成 "bi-equivariant"：object 和 placement target 都在动，两边都要 equivariant。Useek [180] 检测 SE(3)-equivariant keypoint 处理 arbitrary 6-DoF pose object。Equi-GSPR [78] 和 SURFELREG [80] 把 SE(3)-equivariant feature 用到 point cloud registration。

**SIM(3)-equivariant**, Equation (8):

$$f(\alpha R x + t; \theta) = \alpha R f(x; \theta) + t, \quad \forall (\alpha, R, t) \in \mathrm{SIM}(3)$$

- $\alpha > 0$: scale factor
- $R$: rotation
- $t$: translation
- SIM(3): similarity group，多了 scale 维度

intuition: 比 SE(3) 多了 scale invariance——大杯子和小杯子结构上一样，grasp point 应该按比例缩放。EFEM [91] 用 SDF encoder-decoder：

$$\Theta = \Phi(P), \quad \hat{v}(x) = \Psi(\Theta, x)$$

- $P$: input point cloud
- $\Phi$: VN-based encoder，输出 latent embedding $\Theta$
- $\Psi$: decoder，预测 query position $x$ 处的 SDF
- $\hat{v}(x)$: predicted Signed Distance Function value

rotation + translation equivariance 靠 VNN + mean-centering，**scale equivariance 靠 channel-wise normalization**（每个 channel 归一化掉 magnitude）。https://github.com/leijy11/EFEM

**Visuo-Tactile Perception**: tactile sensor 补 RGB-D 看不见 occluded object 的不足。NeuralFeels [158] + DIGIT 360 [90] 把 vision + touch 集成进 multi-finger hand，即使 visual occlusion 也能估 object pose + shape。

#### 3.4.2 Policy Learning

Table 4 的核心公式：

**Model-Free RL (Q-learning update)**, Equation (10):

$$Q(\mathbf{s}_t, \mathbf{a}_t) \leftarrow Q(\mathbf{s}_t, \mathbf{a}_t) + \alpha \left[ r_{t+1} + \gamma \max_{\mathbf{a}'} Q(\mathbf{s}_{t+1}, \mathbf{a}') - Q(\mathbf{s}_t, \mathbf{a}_t) \right]$$

- $\mathbf{s}_t$: 当前 state
- $\mathbf{a}_t$: 当前 action
- $r_{t+1}$: 收到的 reward
- $\gamma \in [0, 1]$: discount factor（未来 reward 的折扣）
- $\alpha \in [0, 1]$: learning rate
- $\max_{\mathbf{a}'} Q(\mathbf{s}_{t+1}, \mathbf{a}')$: 下一 state 的最优 Q-value 估计
- 方括号内: TD error (Temporal Difference error)

intuition: Q-function 估计"在 state $s$ 做 action $a$ 的累计未来 reward"。Bellman equation 说 $Q(s, a) = r + \gamma \max_{a'} Q(s', a')$，update rule 就是把当前 estimate 往 Bellman target 推。OpenAI [125] 用 PPO [145] + LSTM policy 做 dexterous manipulation（旋转物体到指定 orientation），训练时大量 randomize friction 等物理参数。

**Model-Based RL value function**, Equation (11):

$$V(\mathbf{s}) = \max_{\mathbf{a}} \left[ R(\mathbf{s}, \mathbf{a}) + \gamma \sum_{\mathbf{s}'} P(\mathbf{s}' \mid \mathbf{s}, \mathbf{a}) V(\mathbf{s}') \right]$$

- $V(\mathbf{s})$: state value
- $R(\mathbf{s}, \mathbf{a})$: reward function
- $P(\mathbf{s}' \mid \mathbf{s}, \mathbf{a})$: transition probability（state 转移模型）
- $\gamma$: discount factor

intuition: 显式建模 $P(\mathbf{s}' \mid \mathbf{s}, \mathbf{a})$ 这个 dynamics model，然后用它做 planning（比如 MPC）。Nagabandi et al. [120] 用神经网络近似 $\hat{p}_\theta(\mathbf{s}' \mid \mathbf{s}, \mathbf{a})$ 做 dynamics model，配合 MPC online planning。

**Behavior Cloning (BC)**, Equation (12):

$$\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} \|\pi_\theta(\mathbf{o}_i) - \mathbf{a}_i\|^2$$

- $N$: demonstration 数
- $\pi_\theta$: policy with parameter $\theta$
- $\mathbf{o}_i$: 第 $i$ 个 demonstration 的 observation
- $\mathbf{a}_i$: 对应的 expert action
- $\|\cdot\|^2$: MSE loss

intuition: supervised learning，让 policy 模仿 expert。简单暴力，但有 **compounding error** 问题：早期 timestep 的小误差让 agent 偏离 training distribution，之后没见过这种 state，越走越偏。

**Action Chunking with Transformers (ACT)** [199], Equation (13):

$$\mathbf{a}_t = \frac{\sum_{i=0}^{k-1} w_i \cdot \hat{\mathbf{a}}_t^{(i)}}{\sum_{i=0}^{k-1} w_i}, \quad w_i = \exp(-m \cdot i)$$

- $k$: action chunk 长度
- $\hat{\mathbf{a}}_t^{(i)}$: 第 $i$ 个 overlapping chunk 在 timestep $t$ 的 prediction
- $w_i = \exp(-m \cdot i)$: decay weight，$m$ 是 decay 参数
- $\mathbf{a}_t$: final action，是所有 overlapping chunk 的加权平均

intuition: BC 一次只 predict 1 个 action 容易累积误差。ACT 一次 predict $k$ 个 action 组成 chunk，多个 chunk overlap 的时候加权平均（最近 chunk 权重最高）。这相当于 **temporal ensemble**，平滑掉单次 prediction 的 noise，同时让 action 在 chunk 边界过渡自然。$m$ 控制衰减速度，$m$ 大 → 只信最新 chunk。https://tonyzhaozh.github.io/aloha/

**Learning from human video**, Equation (14):

$$\mathbf{a} = f_{\text{retarget}}(f_{\text{pose}}(\mathbf{o}))$$

- $\mathbf{o}$: human video observation
- $f_{\text{pose}}$: 提取 human pose
- $f_{\text{retarget}}$: 把 human pose 重定向到 robot pose
- $\mathbf{a}$: robot action

intuition: human video 数据海量，robot demonstration 稀缺。通过 pose retargeting 把 human action 转成 robot action，扩 dataset。HumanPlus [49], OmniH2O [58], OKAMI [94], ACE [186] 都用这思路。

**Diffusion Policy** [26], Equation (15):

$$\mathbf{a}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{a}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{a}_t, t, \mathbf{o}) \right) + \sigma_t \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(0, I)$$

- $\mathbf{a}_t$: 第 $t$ 步 diffusion 的 noisy action
- $\mathbf{a}_{t-1}$: 去噪一步后的 action
- $\alpha_t, \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$: noise schedule 参数
- $\epsilon_\theta(\mathbf{a}_t, t, \mathbf{o})$: neural network 预测的 noise，以 observation $\mathbf{o}$ 和 timestep $t$ 为 condition
- $\sigma_t$: stochastic noise 的 std
- $\mathbf{z} \sim \mathcal{N}(0, I)$: 标准 Gaussian noise

intuition: 这是 DDPM [59] 的 reverse process 一步。$\mathbf{a}_t$ 是完全 noise 的 action，每步去一点 noise，最终得到 clean action $\mathbf{a}_0$。$\epsilon_\theta$ 学的是"这张 noisy image 里有多少 noise"，减掉它就 cleaner 一点。**为什么 diffusion 比 BC 好？** 因为 manipulation 的 action distribution 是 multimodal 的（同一个 observation 可以有多种合理 action）。BC 用 MSE loss 假设 unimodal Gaussian，遇到 multimodal 就 average 出 invalid action。Diffusion 天然支持 multimodal distribution。RDT-1B [105] 把 diffusion policy 输出 generalize 到不同 robot hardware platform。3D Diffusion Policy [195] 用 point cloud conditioning 代替 2D image。https://diffusion-policy.cs.columbia.edu/

**VLM policy**, Equation (16):

$$\pi(l_{\text{act}} \mid \mathbf{o}, l_{\text{ins}})$$

- $\mathbf{o}$: observation
- $l_{\text{ins}}$: language instruction
- $l_{\text{act}}$: language form 的 action description（"move arm forward"）

EMMA [189] 把 visual observation 转 text 描述，LLM 生成 text action description。PaLM-E [39] 把 image embedding 直接 embed 进 LLM prompt。OK-Robot [104] 先扫 home 建 navigation map（CLIP embedding），用 A* 导航到 object，再用 AnyGrasp [43] 生成 grasp。AlignBot [200] fine-tune LLaVA [101, 102] 把 user preference 转成 instruction cue，喂 GPT-4 [1] 生成 task plan，再调 ACT / AnyGrasp 执行。

**VLA policy**, Equation (17):

$$\pi(\mathbf{a} \mid \mathbf{o}, l_{\text{ins}})$$

- 直接输出 low-level action $\mathbf{a}$（不再经过 language）

RT-1 [17] 是 VLA 鼻祖，把 action token 化成 language token 格式，end-to-end transformer。RT-2 [210] 把 PaLM-E（high-level semantic）+ RT-1（low-level control）融合。RT-H [13] 引入 language-motion hierarchy：$\pi_h(l_{\text{act}} \mid \mathbf{o}, l_{\text{ins}})$ 和 $\pi_l(\mathbf{a} \mid \mathbf{o}, l_{\text{ins}}, l_{\text{act}})$，分层决策。RT-X [31] 在 Open X-Embodiment 上训 generalist policy 跨 22 种 robot。OpenVLA [84] 是 RT-X 的开源版。https://openvla.github.io/

**π0** [15] 用 **flow matching** 生成 continuous motor action in action chunk，支持高频执行。Flow matching 是 diffusion 的 cousin，但用 ODE 而不是 SDE，更高效。https://www.physicalintelligence.company/blog/pi0

**GR00T N1** [14] NVIDIA 的 humanoid foundation model，dual-system：System 1 是 Eagle-2 VLM [97] backbone 做 semantic reasoning @ 10Hz，System 2 是 Diffusion Transformer (DiT) [131] 做 action generation @ 120Hz。用 flow-matching loss 训练，数据三合一：web-scale human video + physics simulator synthetic trajectory + real teleop。还用 Inverse Dynamics Model (IDM) 给 unannotated video 伪标 action，实现 cross-embodiment（tabletop arm + humanoid 一起训）。https://developer.nvidia.com/groot

**Equivariant Policy Learning**: EquivAct [184] 用 EFEM encoder [91] 学 SIM(3)-equivariant visuomotor policy，输入 point cloud，VNN-based action network 输出 end-effector command。Equibot [183] 同样 SIM(3)-equivariant encoder + SIM(3)-equivariant diffusion policy。这俩是 manipulation generalization 的重要方向。https://equivact.github.io/

---

## 4. Future Directions

1. **Efficient Learning**: Lin et al. [99] 显示 scaling data 让 single-task policy generalize。但 biological system 用极少数据就 adapt。Continual learning [5, 202] 是关键方向。

2. **Continual Learning**: catastrophic forgetting 是 VLN 大问题 [98]。EWC [85] / Meta-learning [46] / Titans neural memory [12]。NeSyC [29] neuro-symbolic continual learner。Zheng et al. [203] 给 LLM-based agent 的 lifelong learning roadmap。

3. **Neural ODEs** [21]: 处理 continuous dynamics（pouring liquid 这种）。Liquid network [57] 处理 irregular input。

4. **Evaluation Metrics**: 现在太 goal-oriented（SR, path length）。需要 procedural quality：Energy efficiency / Smoothness。Jiang et al. [77] exploration-aware EQA benchmark。

---

## 5. 我的 critical insights（给你 build intuition 用）

Karpathy 你肯定有自己的判断，我列几个我读完这篇 paper 的 take：

**(1) Navigation 和 manipulation 的 paradigm 走向完全不同**。Navigation 的 frontier 是 world model——因为 navigation 是 long-horizon, partial observability, state space 巨大，需要 "imagined rollout"。Manipulation 的 frontier 是 VLA + diffusion policy hybrid——因为 manipulation 是 high-frequency, multimodal action distribution, 需要 foundation model 的 semantic + diffusion 的 multimodality。Pan et al. [127] 和 Diffusion-VLA [171] 这种 "VLA 做高层规划 + Diffusion 做低层执行" 的 hybrid 是很自然的 decoupling。

**(2) Equivariance 是 manipulation generalization 的数学 key**。VLA 的 generalization 靠 data scale（Open X-Embodiment 1M trajectory），是 brute force。Equivariant policy 靠数学 prior——输入转 $\alpha R x + t$，输出自动 $\alpha R f(x) + t$，这样不需要数据覆盖所有 pose/scale 组合，sample efficiency 几个数量级提升。但 equivariance 也是 inductive bias，limit model 表达能力。两者最终可能 converge：用 equivariant architecture 当 backbone，再叠 foundation model 知识。

**(3) Differentiable physics 是 sim-to-real 的真正解决方向**。传统 simulator 你只能 randomize physics parameter (friction, mass) 让 policy robust，但这是被动防御。Differentiable simulator 让你能 gradient-optimize policy directly through physics，这是主动 align。Genesis 10-80x 速度提升是 game changer，让 differentiable simulation 终于 practical。

**(4) Dataset 的两个维度 — scale 和 diversity — 在分化**。Open X-Embodiment 1M trajectory 是 scale 极致，但跨 22 robot 其实 diversity 有限。Behavior-1K 1000 activity 是 task diversity 极致。理想 dataset 应该两者都有：large-scale + diverse task + diverse embodiment + diverse scene。现在的 bottleneck 是 real-world data collection cost。

**(5) Evaluation metric 落后于 method**。SR / SPL / Success Rate 这些 metric 是 2018 年设计的，反映不出 VLA 时代的 nuance：energy efficiency, smoothness, safety compliance, generalization to unseen object, robustness to perturbation。这个 gap 是 opportunity。

**(6) Tactile perception 是 under-explored frontier**。vision-only manipulation 在 transparent / reflective / occluded object 上 fundamental fail。NeuralFeels [158] / DIGIT 360 [90] 这种 visuo-tactile 融合才刚开始。下一个 dexterous manipulation breakthrough 我赌在 tactile。

参考综合资源：
- This survey: https://arxiv.org/abs/2502.13125 (推测)
- Habitat: https://aihabitat.org/
- Isaac Sim: https://developer.nvidia.com/isaac-sim
- Genesis: https://github.com/Genesis-Embodied-AI/Genesis
- MuJoCo: https://mujoco.org/
- SAPIEN: https://sapien.ucsd.edu/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- DROID: https://droid-dataset.github.io/
- Mobile ALOHA: https://mobile-aloha.github.io/
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- π0: https://www.physicalintelligence.company/blog/pi0
- GR00T N1: https://developer.nvidia.com/groot
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- 3D Diffusion Policy: https://3d-diffusion-policy.github.io/
- Vector Neurons: https://github.com/dnlkrm/Vectors
- NDF: https://yilundu.github.io/ndf/
- EquivAct: https://equivact.github.io/
- VoxPoser: https://voxposer.github.io/
- FoundationPose: https://nvlabs.github.io/FoundationPose/
- LERF: https://lerf.io/
- GraspNet-1Billion: https://graspnet.net/
- CALVIN: https://calvinrobot.github.io/
- Meta-World: https://meta-world.github.io/
- RLBench: https://sites.google.com/view/rlbench
- SoftGym: https://sites.google.com/view/softgym
- Behavior-1K: https://behavior.stanford.edu/
- ManiSkill: https://maniskill.github.io/
- ALFRED: https://askforalfred.com/
- R2R: https://bringmeaspoon.org/
- HM3D: https://aihabitat.org/datasets/hm3d/
- NavGPT: https://github.com/GengzeZhou/NavGPT
- Cosmos: https://arxiv.org/abs/2501.03575
- Dojo: https://github.com/dojo-sim/Dojo.jl
- DiffTaichi: https://github.com/taichi-dev/taichi
- Simulately wiki (作者维护的 simulator 对比): https://simulately.wiki

---

要不要我 deep dive 某个具体方向？比如 differentiable physics 的数学（implicit function theorem 怎么 apply 到 contact）、VLA 的 action tokenization 细节、equivariant network 的 group theory foundation、或者 world model 在 navigation 里的具体 rollout 机制？我可以再展开讲，因为这种 survey 信息密度很高，每个 sub-section 背后都有大量 paper 可以挖。
