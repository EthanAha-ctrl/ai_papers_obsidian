---
source_pdf: Aligning Cyber Space with Physical World A.pdf
paper_sha256: f1041f3396d4561d2c91d0b9d6f4fce739169328dfdc07aa657623118ac8ea74
processed_at: '2026-08-18T00:47:01-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

既然你要我用人话讲，那咱们就把这篇 100 多页的论文拆解开，揉碎了，用最直白的方式把 Embodied AI 的底牌亮出来。我会保留核心的英文专有名词，同时给你上足技术细节、公式和架构分析，直接帮你 build up 最硬核的 intuition。

---

### 1. Embodied AI 的核心直觉：从大脑在缸里到肉身破壁

Disembodied AI（比如 ChatGPT）活在 token 的 cyberspace 里，它是个“缸中之脑”。你给它喂什么数据，它就在什么分布里做概率拟合。Embodied AI 的目的就是给这个大脑装上手脚和眼睛，让它去物理世界里碰壁、感知、互动。

这篇论文的核心论点就在于：**Multi-modal Large Models (MLMs) 和 World Models (WMs) 提供了真正的 brain 架构**，让 robot 能够理解人类意图、主动探索环境、执行复杂任务。目前最经典的代表就是 Google 的 RT 系列。

> **Intuition 构建：** 为什么 LLM 用来做机器人那么难？因为互联网上的数据全是结果导向的（比如视频里人拿起了杯子），LLM 学会了“说”拿杯子，但它完全没有“受力”、“摩擦力”、“重力”的概念。Embodied AI 的本质就是去补齐这部分物理直觉。

---

### 2. Simulator：黑客帝国的母体

要训练 robot，在真实世界里采数据太贵了。买个 Franka Emika panda 机械臂，找个专家遥操采一万条数据，得耗几个月。所以大家都在 simulator 里炼丹，炼好了再往真实世界迁移。

这里的核心在于 simulator 的物理引擎和渲染保真度。你看这篇论文里的 Table II，几个主流 simulator 的特性对比非常清晰：

| Simulator | Physics Engine | 核心优势 | 主要应用 |
| :--- | :--- | :--- | :--- |
| Isaac Sim (NVIDIA) | PhysX | High-fidelity, Ray tracing, USD 格式 | Nav, AD, RL |
| MuJoCo | Custom | Contact dynamics 极其精确 | RL, Robot Simulation |
| Habitat (Meta) | Bullet | 1000+ 真实扫描室内场景，极快 | Nav, 多 agent 交互 |
| SAPIEN (Stanford) | PhysX | 专注 articulated objects (门、抽屉) | 细粒度物理交互 |

> **架构解析：** 为什么 NVIDIA 要搞 Isaac Sim 并且大力推 USD (Universal Scene Description) 格式？因为物理仿真最大的瓶颈在于 scene 的数字孪生。USD 格式可以把 mesh、材质、关节约束、摩擦系数统一在一个图层级结构里，GPU 可以做大规模并行 ray tracing 和 collision detection。这就是大规模 RL 训练的基础设施。

---

### 3. Embodied Perception：从被动看图到主动感知

传统 CV 是被动感知，给一张图，找猫。Embodied AI 是主动感知，robot 可以走过去，换个角度看，甚至摸一下。论文里重点讲了 3D Visual Grounding (3D VG)。

3D VG 的目标：给定语言描述 $D$，在 3D 场景 $S$ 中定位目标物体 $o^*$。

$$o^* = \arg\max_{o \in S} \text{Score}(o, D)$$

*   $o^*$：预测的目标物体。
*   $S$：3D 场景中的所有物体候选集。
*   $D$：人类的自然语言描述指令。
*   $\text{Score}$：跨模态匹配打分函数。

**Two-stage vs One-stage 架构深度对比：**
*   **Two-stage (比如 ScanRefer):** 先用 3D detector（如 VoteNet）生成一堆 proposals $\{p_1, p_2, ...\}$，再用 cross-modal transformer 去 match。痛点在于：sparse proposals 容易把目标漏掉；dense proposals 会引入太多干扰物体，让 matching 阶段崩溃。
*   **One-stage (比如 3D-SPS):** 直接把 language feature 注入到 point cloud 的 keypoint 采样里。Description-aware keypoint sampling 让模型在采点的时候就只关注语言相关的区域，然后 progressive mining 目标点。

> **Intuition 构建：** 相当于你看房找东西。Two-stage 是先无脑把全屋扫一遍三维地图，再去找符合描述的东西；One-stage 是你边走边看，看到红色的东西就多关注，逐步缩小范围。后者效率高且不易受背景干扰。

---

### 4. Visual Language Navigation (VLN)：在未知环境里听人指路

VLN 是最核心的 spatiotemporal 任务。Agent 收到指令：“走到厨房，在桌子上拿个苹果”，它得边看边走。

**状态转移公式（公式1的深度解析）：**

$$Action_t = \mathcal{M}(O_t, H_{1:t-1}, I)$$

*   $Action_t$：当前时刻 $t$ 选择执行的动作（比如“前进”、“左转”或“停止”）。
*   $O_t$：当前时刻的 egocentric visual observation。
*   $H_{1:t-1}$：历史记忆信息，包含之前所有的观测和动作。
*   $I$：自然语言指令。
*   $\mathcal{M}$：策略模型（通常是基于 Transformer 的网络）。

> **技术细节：** 论文里把 VLN 方法分为 Memory-Understanding Based 和 Future-Prediction Based。早期大家都在死磕 $H_{1:t-1}$，比如用 Graph 建 topo-map（LVERG），或者用 RNN 记忆。但这有个致命问题：如果 agent 一直没看到目标，它就会在原地打转，因为它是 partial observable Markov decision process (POMDP)。
>
> 所以 **Future-Prediction Based** 方法开始崛起，比如 HNR 用 neural radiance representation 直接预测前方未见环境的特征表征。这就跟 World Model 的思想接轨了——你不光能记住过去，你还能“想象”未来。如果想象的前方有目标，你就走过去；没有，就换个方向。

---

### 5. Embodied Agents 的 VLA 与 World Model 之争

这是整篇论文的技术最高峰。Google 的 RT 系列演进就是这段历史的缩影。

**RT-2 的颠覆性架构：Vision-Language-Action (VLA) Model**
RT-2 把 robot 的连续控制动作离散化，变成 text token，直接和 internet 上的 text/image token 一起训练。
*   输入：$V$ (Vision token) + $I$ (Instruction token)
*   输出：$A$ (Action token, 比如 `<move_to(x,y,z)>`, `<grasp()>`)

这极大提升了 generalization，但是论文指出了它的阿喀琉斯之踵：**VLA 模型只是基于 internet 数据的 pattern matcher，它无法真正模拟 physical laws。** 推理频率只有 1-3Hz，打个乒乓球都不够，更别说精细操作。

于是 **Embodied World Model (EWM)** 概念登场。论文里把 World Model 分成三类：

1.  **Generation-based (生成式)：** 比如 Sora，通过生成视频来隐式学习物理规律。但这太耗算力，且不可控。
2.  **Prediction-based (预测式)：** LeCun 主推的路线。以 I-JEPA 为代表。

**I-JEPA 核心公式直觉化解析：**
JEPA 的架构就是一个 encoder 加一个 predictor。

$$\mathcal{L}_{\text{JEPA}} = \frac{1}{N} \sum_{i=1}^{N} \left\| \phi_E(x_i) - \phi_P(\hat{x}_i, z_i) \right\|^2$$

*   $x_i$：目标块的内容，通常是未来的 state 或未观测的 region。
*   $\phi_E(\cdot)$：Context encoder，把已知输入映射到 latent space。
*   $\hat{x}_i$：当前已知的 context。
*   $z_i$：目标变量，也就是 predictor 要去优化的 latent action。
*   $\phi_P(\cdot)$：Predictor，在 latent space 里预测 $x_i$ 的表征。

> **Intuition 构建：** 假设你在打篮球，球飞过来了。你的大脑不需要渲染出每一帧像素级的高清未来画面（生成式太慢了）。你的大脑只需要在 latent space 里预测：“球大概会落在那个位置”，然后你的身体（policy）做出反应去接球。这就是 prediction-based world model 相比 generation-based 的降维打击，计算量小，直接对接 policy。

---

### 6. Sim-to-Real：跨越现实裂缝的五大法术

在仿真里训练得再好，到了真实世界，光照变了、摩擦系数变了，policy 就崩了。论文总结了五大 sim-to-real paradigm：

1.  **Real2Sim2Real:** 扫描真实场景建 digital twin，在 twin 里用 RL 死磕，再把策略拿回真实世界。
2.  **TRANSIC:** 真实世界里人带着手套手把手教 robot，收集 correction 数据，训一个 residual policy 叠加在 RL base policy 上。
3.  **Domain Randomization:** 极其暴力美学。在 simulator 里，把摩擦力、质量、光照全部随机化。逼着 network 学会一个在所有可能的物理参数下都 work 的通用策略。到了真实世界，真实参数只是这个分布里的一个采样点。
4.  **System Identification:** 精确建模，死磕参数，让仿真尽可能一比一复刻真实。
5.  **Lang4Sim2Real:** 用自然语言描述图像，作为连接 sim 和 real 的桥梁。比如两张图（一张虚拟一张真实），用 CLIP 提取成语言描述的 embedding，在这个空间里让两者对齐。

---

### 7. 触觉感知 Tactile：极度被低估的 modality

论文里提到了一个极其关键但极其被冷落的方向：Tactile sensor。
视觉告诉你“那是个杯子”，但你要抓它，你需要知道“滑不滑”、“重不重”、“硬不硬”。如果只靠视觉去抓透明玻璃杯，CNN 直接抓瞎。

**GelSight 架构解析：**
GelSight 是 vision-based tactile sensor 的代表作。它的原理极其巧妙：在机械指肚上涂一层透明弹性硅胶 gel，gel 后面放一个摄像头和几个彩色 LED。当 gel 压到物体表面时，表面会变形。由于 LED 从不同角度打光，gel 表面的微小形变会反射出不同颜色。后方的摄像头拍下这个彩色图，用 CNN 就能解算出高度图和摩擦力。
目前 Tactile 数据集极度匮乏，论文里提到的 ObjectFolder 和 TVL (Touch-Vision-Language) 试图把触觉对齐到 CLIP 的 multimodal space 里。

---

### 8. Andrej 视角的未来联想与突破点 (Hallucination / Association)

作为 Karpathy，你肯定会关注这篇 paper 背后更深层的系统架构问题。我帮你联想几个可能引爆的突破点：

**A. Mamba 架构在 Embodied 里的降维打击**
论文里提到了 RoboMamba。现在 VLA 的瓶颈在于 Transformer 处理 long-horizon 视频和历史轨迹时 $O(N^2)$ 复杂度爆炸。Mamba 的 State Space Model (SSM) 把复杂度降到线性 $O(N)$，非常适合处理 robot 的长周期流式数据。
$$h_t = A h_{t-1} + B x_t, \quad y_t = C h_t$$
*   $h_t$：当前隐藏状态，包含了所有历史信息。
*   $A, B, C$：状态转移矩阵，Mamba 的核心在于让它们 input-dependent（依赖当前输入 $x_t$ 动态变化）。
这意味着 robot 可以用 Mamba 实时流式处理 camera stream，并且保持极强的历史记忆，这是 Transformer 做不到的。

**B. Video Generation 模型直接作为 World Model**
Sora 证明了 video generation model 能够隐式学到牛顿力学。未来 Embodied AI 的终极形态，可能就是把 Sora 的 Diffusion Transformer (DiT) 架构直接接到 robot 视频流上。你给 model 一个当前帧和一个 action token，它直接生成未来 10 帧视频。如果生成的视频里杯子掉地上了，policy 就知道这个 action 不行。这就是论文里讲的 Generation-based Methods 的极化。

**C. Causal Discovery 嵌入到 Robot Planning**
论文第八章提到 Causal Relation Discovery。现在的 LLM planner 会产生幻觉，比如让它找毛巾，它会回答去卧室找，但如果毛巾锁在柜子里，它永远找不到，因为它只有关联性记忆，没有因果推断。未来的 Embodied agent 必须把 counterfactual reasoning 嵌进去。用 do-calculus 公式 $P(Y | do(X))$ 去评估某个动作的因果效应，才能实现长周期的可靠任务执行。

**D. Soft Robotics 与 Biomimetic 控制的数学建模**
目前的控制论都是基于 rigid body dynamics (刚体动力学)。但论文里提到的 biomimetic robots 用的是 soft materials。这玩意的动力学是非线性的无穷维偏微分方程。如果用 deep RL + Neural ODE 去拟合 soft robot 的连续介质力学方程，将是一个极度 fertile 的处女地。

### Reference & Further Reading
* 原始论文与 Paper List: [HCPLab Embodied AI GitHub](https://github.com/HCPLab-SYSU/Embodied-AI-Paper-List)
* LeCun I-JEPA (World Model 的底层逻辑): [I-JEPA Paper](https://arxiv.org/abs/2301.08243)
* RT-2 架构解析: [RT-2 Project](https://robotics-transformer2.github.io/)
* Open X-Embodiment 跨本体数据: [Open X-Embodiment](https://robotics-transformer-x.github.io/)
* Tactile 与多模态对齐: [ObjectFolder GitHub](https://objectfolder.stanford.edu/)

人话说到底，Embodied AI 的核心就是：**给概率生成模型装上物理因果和时空连续性的外挂。** 谁能把 internet scale 的 LLM 跟底层的 differential physics equations 无缝对齐，谁就搞出了 AGI。

---

# Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI 深度讲解

## 论文整体定位

这篇综述由中山大学 HCPLab 的 Yang Liu 团队完成，发表于 2024 年，是 MLMs (Multi-modal Large Models) 时代下首篇系统性覆盖 Embodied AI 全栈的综述。论文的核心论点是：**AGI 的实现路径必须通过 Embodied Agent 将 cyberspace 中的认知能力迁移到 physical world 的物理执行上**，而 MLMs + World Models (WMs) 提供了"brain"的可行架构。

论文的关键区分：
- **Disembodied AI** (e.g., ChatGPT, RoboGPT): Cognition 与 physical entity 是 disentangled 的，运行在 cyberspace 中
- **Embodied AI** (e.g., RT-1, RT-2, RT-H): Cognition 集成到 physical entity（robots、cars、devices）中，与 physical space 交互

这一点非常 LeCun-style，跟他 JEPA 系列工作的思想是高度一致的 — 真正的 intelligence 必须能预测并作用于物理世界。

参考链接：
- Paper: https://arxiv.org/abs/2407.06886
- GitHub: https://github.com/HCPLab-SYSU/Embodied-AI-Paper-List

---

## I. Embodied Robots 形态学分类

论文把 robot 硬件形态分为六类，这个分类背后的 logic 是 **task-driven morphology**：

### Fixed-base Robots (Franka, KUKA iiwa, Sawyer)
- Micron-level precision，适合 lab automation 和 industrial manufacturing
- 缺陷是 operational range 受限，无法在大空间内移动协作
- 在 RT-1 / RT-2 系列工作中，Franka Emika Panda 是最常用的 research platform

### Wheeled Robots (Kiva, Jackal)
- Energy efficiency 高，适合 flat surface 的 logistics / warehousing
- 局限在复杂地形上 mobility 受限
- Kiva Systems（被 Amazon 收购）是 warehouse robot 的开山鼻祖：https://www.amazon.science/blog/how-amazon-uses-robotics

### Tracked Robots (iRobot PackBot)
- 大接触面积 → 在 soft terrain (mud, sand) 上不易下沉
- 军事、灾难救援场景，但 energy efficiency 低

### Quadruped Robots (Unitree A1/Go1, Boston Dynamics Spot, ANYmal C)
- Multi-jointed design 模仿 quadrupedal animals
- 在 uneven terrain 上保持 balance 和 mobility 的关键在于 **gait adjustment + terrain adaptation**
- ANYmal C 用 modularity 和 durability 实现工业巡检，甚至月球任务
- Spot 是 locomotion research 的事实标准：https://www.bostondynamics.com/products/spot

### Humanoid Robots (Atlas, HRP series, ASIMO, Pepper, Tesla Optimus)
- 最大区别在于 **dexterous hand design** — multi-DOF + high-precision sensors
- ASIMO 是 Honda 1986-2018 的 legacy，Pepper 走 social robot 路线
- Atlas 用 hydraulic → motor-driven 的 transition 是当前 trend，integration with LLMs 是 next step

### Biomimetic Robots (fish-like, insect-like, soft-bodied)
- Flexible materials 实现 lifelike agile movement
- Energy efficiency 高（模仿 biological organism 的 efficient movement mechanism）
- 但 manufacturing 复杂、durability 差

---

## II. Embodied Simulators 对比

这是论文中最有价值的 comparison table 之一。Simulators 分两类：

### General Simulators

关键 feature 维度（Table II）：
- **HFPS** (High-Fidelity Physical Simulation)
- **HQGR** (High-Quality Graphics Rendering)
- **RRL** (Rich Robot Library)
- **DLS** (Deep Learning Support)
- **LSPC** (Large-Scale Parallel Computing)
- **ROS** (ROS Integration)
- **MSS** (Multiple Sensor Simulation)
- **CP** (Cross-Platform)

Isaac Sim (NVIDIA, 2023) 是当前最强的 general simulator，基于 PhysX engine，支持 USD 格式描述 scene。Isaac Gym 是其 RL 训练专用版本，做大规模并行 sim：https://developer.nvidia.com/isaac-sim

MuJoCo (2012, DeepMind 收购后开源) 是 RL 领域最常用的 physics engine，Custom engine 设计让 contact dynamics 求解非常精确：https://mujoco.org/

### Real-Scene Based Simulators (Table III)

这一类更接近 household embodied AI 的需求：

**AI2-THOR** (Allen Institute, 2017):
- 120 rooms (kitchen/bedroom/bathroom/living room)
- iTHOR + RoboTHOR 两部分，RoboTHOR 含 89 个 modular apartments，且对应真实世界场景
- 物体 state 可被 script 改变（open/close, cold/hot）

**Matterport 3D** (2018):
- 90 个 architectural indoor scenes，194,400 RGB-D images
- 关键创新是 **discrete viewpoints** — agent 在 viewpoints 间移动，每个 viewpoint 提供 1280×1024 的 panorama
- 这是 R2R navigation benchmark 的基础

**Habitat** (Meta, 2019):
- Habitat-sim (基于 Bullet) + Habitat-lab (RL framework) + Habitat-challenge (benchmarks)
- 1000+ scenes，极度开放，支持 multi-agent simulation
- 是当前 Embodied AI challenge 的主战场：https://aihabitat.org/

**VirtualHome** (2018):
- 独特的 **environment graph** 表示 — nodes (objects with ID, state) + edges (relationships)
- API 极简：`operation + object` 格式，特别适合 embodied planning 研究

**SAPIEN** (Stanford, 2020):
- 基于 PhysX，专注于 articulated objects 的物理交互
- PartNet-Mobility Dataset 提供 door, cabinet, drawer 等 hinged parts
- 区别于 AI2-THOR 的 script-based interaction，SAPIEN 提供 force/torque 控制的真实物理交互

**iGibson** (Stanford, 2021):
- 15 high-quality scenes + 10,000+ 模拟场景
- 独特 attribute：temperature, humidity, cleanliness, switch status
- 支持 LiDAR sensor（其他 simulator 多数只有 RGB-D）

**TDW** (MIT, 2021):
- Multi-physics-engine 集成：rigid body, soft body, fabric, fluid
- **Situational audio** — 仿真物体交互声音，这是其他 simulator 没有的

近期工作还有：
- **ProcTHOR** (Allen Institute, 2022): procedural generation of 10,000 interactive scenes
- **HOLODECK** (2024): LLM-guided 自动定制场景
- **PhyScene** (2024): conditional diffusion 生成 physically consistent 3D scenes

---

## III. Embodied Perception

### A. Active Visual Perception

这是论文 Figure 7 的核心 — 三大组件构成 active perception：

**1. Visual SLAM (vSLAM)**

vSLAM 用 onboard camera 估计 robot pose 同时构建 environment map。分两类：

- **Traditional vSLAM**: MonoSLAM (filter-based), PTAM/ORB-SLAM (keyframe-based), DTAM/LSD-SLAM (direct tracking)
  - 输出 sparse/dense point cloud map，但 point 不对应 semantic object
  
- **Semantic vSLAM**: SLAM++, CubeSLAM (3D cuboid), QuadricSLAM (3D ellipsoid), DS-SLAM/DynaSLAM (dynamic object filtering), GS-SLAM (3D Gaussian splatting)
  - GS-SLAM 用 differentiable splatting rendering pipeline，balance efficiency 和 accuracy

**2. 3D Scene Understanding**

Point cloud 处理方法分三类：
- **Projection-based**: MV3D, PointPillars, MVCNN — 投影到 2D plane 用 CNN
- **Voxel-based**: VoxNet, SSCNet, MinkowskiNet (sparse convolution), Embodiedscan
- **Point-based**: PointNet, PointNet++, PointMLP, PointTransformer, Swin3d, PT2/PT3, PointMamba, Mamba3D

最近 trend 是 Transformer 和 Mamba 架构在 point cloud 上的扩展，PT3 (Point Transformer V3) 用 partition-based pooling 实现 simpler faster stronger。

**3. Active Exploration**

这是 active perception 区别于 passive perception 的关键：
- **Interacting with environment**: Pinto et al. 的 curious robot，通过 physical interaction 学 visual representation 而非仅依赖 category label
- **Changing viewing direction**: Jayaraman et al. 用 RL 减少 unobserved parts 的 uncertainty；NeU-NBV 用 image-based neural rendering 的 uncertainty estimation 指导 next best view

### B. 3D Visual Grounding (3D VG)

3D VG 的核心公式化：给定 language description $D$ 和 3D scene $S$，找到 target object $o^*$：

$$o^* = \arg\max_{o \in S} \text{Score}(o, D)$$

其中 $\text{Score}$ 是 cross-modal matching function。

方法分两类：

**Two-stage Methods** (Fig. 8 upper):
- Stage 1: 用 pretrained detector (e.g., VoteNet) 生成 proposals $\{p_1, p_2, ..., p_n\}$
- Stage 2: 用 cross-modal matching 在 proposals 中找 target

代表工作：
- **ScanRefer / ReferIt3D** (2020): GNN 编码 contextual relationship
- **3DVG-Transformer** (2021): coordinate-guided contextual aggregation + multiplex attention
- **TransRefer3D**: entity-aware + relation-aware attention
- **MVT** (2022): multi-view transformer 学 view-independent representation
- **LLM-Grounder** (2023): LLM decompose query → 生成 plan → evaluate spatial/commonsense relations，零样本
- **ZSVG3D** (2023): LLM 识别 objects + reasoning → scripted visual program → Python code

Two-stage 的 dilemma（Fig. 8b）：
- **Sparse proposals**: Stage 1 可能 miss target → Stage 2 无法 match
- **Dense proposals**: Stage 2 区分困难（冗余 object 多）

**One-stage Methods** (Fig. 8 bottom):
- **3D-SPS** (2022): 把 3D VG 视作 keypoint selection 问题，description-aware keypoint sampling → goal-oriented progressive mining
- **BUTD-DETR** (2022): bottom-up top-down detection transformer，灵感来自 MDETR 和 GLIP
- **EDA** (2023): 把 long text decouple 成 5 个 semantic components (main object, auxiliary object, attributes, pronoun, relationship) → dense alignment
- **ReGround3D** (2024): visual-centric reasoning module (MLM-powered) + 3D grounding module + **Chain-of-Grounding** 机制

### C. Visual Language Navigation (VLN)

VLN 的核心公式（公式 1）：

$$Action = \mathcal{M}(O, H, I)$$

其中：
- $O$ = current observation (visual input)
- $H$ = historical information (past trajectory)
- $I$ = natural language instruction
- $\mathcal{M}$ = navigation policy model
- $Action$ = chosen action 或 action candidate list

**Evaluation Metrics**:
- **SR** (Success Rate): 到达 target 的比例
- **TL** (Trajectory Length): 导航路径长度，反映 efficiency
- **SPL** (Success weighted by Path Length): $SPL = \frac{1}{N}\sum_{i=1}^{N} S_i \cdot \frac{\ell_i}{\max(p_i, \ell_i)}$
  - $S_i \in \{0,1\}$ 是第 $i$ 个 episode 是否成功
  - $\ell_i$ 是 shortest path length
  - $p_i$ 是 actual path length

**Datasets** (Table VI):
- **R2R** (2018): Matterport3D, step-by-step instructions, 21,567 paths
- **R4R** (2019): R2R 的 long-trajectory 扩展, 200,000+
- **VLN-CE** (2020): continuous environment 扩展
- **REVERIE** (2020): concise high-level instruction，agent 要 locate distant invisible target
- **SOON** (2021): coarse-to-fine instruction，target-oriented navigation
- **DDN** (2023): demand-driven，只给 human demand 不指定 object
- **ALFRED** (2020): navigation + interaction，25,743 instances
- **OVMM** (2023): pick object in unseen environment → place to specified location
- **BEHAVIOR-1K** (2023): 1000 个 long-sequence daily tasks，可能含数千 low-level actions
- **CVDN / DialFRED**: dialog-based，agent 可以 ask questions

**Methods** 分两大方向：

**Memory-Understanding Based** (主流):
- **Graph-based**: LVERG (language-visual entity relation graph), LM-Nav (LLM extract landmarks + VLM match)
- **Semantic Map**: FILM (RGB-D + semantic segmentation → 3D voxel map), VER (2D-3D sampling quantifies physical world)
- **Learning Schemes**: CMG (adversarial learning, imitation + exploration), GOAT (causal learning - BACL/FACL), RCM (cross-modal grounding), FSTT (test-time adaptation)
- **LLM-based**: NaviLLM (integrate historical observation into embedding + fine-tune LLM), NaVid (video-based VLM, hierarchical pooling), DiscussNav (multi-expert discussion for zero-shot VLN)

**Future-Prediction Based** (rising):
- **BGBL / ETPNav**: waypoint predictor 把 continuous navigation 转换为 node-to-node discrete navigation
- **NvEM**: theme module + reference module 融合编码 neighbor views
- **HNR**: hierarchical neural radiation representation 直接预测 future visual representation（不预测 pixel-level）
- **LookBY**: RL-based prediction，agent 直接 map "current observation + future prediction" 到 action
- **MiC**: LLM 直接 predict target + possible location → 想象 scene

### D. Non-Visual Perception: Tactile

**Sensor Design 三类** (Fig. 10):
- **Non-vision-based**: BioTac（force, pressure, vibration, temperature），输出 low-dimension series
- **Vision-based**: GelSight, Gelslim, DIGIT, 9DTact, TacTip, GelTip, AllSight — camera 在 gel 后方记录 deformation image
- **Multi-modal**: 受人类皮肤启发，pressure + proximity + acceleration + temperature

**Datasets** (Table VIII): 主要是 GelSight / DIGIT 系列的 vision-based tactile datasets，包括 ObjectFolder (1.0/2.0/Real), TVL (touch-vision-language), TVL 是 Fu et al. 2024 的 multimodal alignment dataset

**Methods 三方向**:
1. **Robotic Manipulation**: RL-based (Visuotactile-RL, Rotateit, Any-Rotate for in-hand rotation) + GAN-based (ACTNet, STR-Net 解决 sim-to-real)
2. **Classification & Recognition**: Traditional (autoencoder, joint training, contrastive learning - CLIP-style) + LLMs/VLMs (Yang et al., Fu et al., Yu et al. 用 contrastive pretrain + LLaMA fine-tune)
3. **3D Reconstruction**: Suresh et al. (Gaussian process SDF), Smith et al. (chart-based + neural network), Comi et al. (DeepSDF + CNN)

---

## IV. Embodied Interaction

### A. Embodied Question Answering (EQA)

EQA 要求 agent 从 first-person perspective 探索 environment 收集信息回答问题。

**Datasets 演进** (Table IX):
- **EQA v1** (2018): SUNCG + House3D, 5,000+ questions, 4 types (location/color/color room/preposition)
- **MT-EQA** (2019): multi-object extension, 19,000+ questions, 6 types 含 cross-object comparison
- **MP3D-EQA** (2019): realistic 3D environment, 1,136 questions
- **IQUAD V1** (2018): interactive EQA, 75,000+ multiple choice, 需要 understanding affordances
- **VideoNavQA** (2019): decouple visual reasoning from navigation, 101,000 video-question pairs
- **K-EQA** (2023): knowledge-based, 60,000 questions, 需要 knowledge graph 推理
- **OpenEQA** (2024): 首个 open-vocabulary EQA dataset，含 EM-EQA (episodic memory) 和 A-EQA (active exploration)，1,600+ questions，180+ 真实环境
- **HM-EQA** (2024): GPT-4V 生成，500 questions，267 scenes
- **S-EQA** (2024): GPT-4 + cosine similarity 筛选，binary answer

**Methods**:

*Neural Network Methods*:
- Das et al. (2018): 4 modules (vision, language, navigation, answering)，CNN + RNN，imitation learning 预训练 + policy gradient fine-tune
- Wu et al. (2020): unified SGD pipeline 联合训练 navigation + QA
- Gordon et al. (IQUAD): Hierarchical Interactive Memory Network + Egocentric Spatial GRU
- Tan et al. (K-EQA): neural program synthesis + knowledge/scene graph + MCTS

*LLMs/VLMs Methods*:
- Majumdar et al. (OpenEQA): Blind LLMs / Socratic LLMs (with scene graph) / VLMs 处理 multi-frame
- A-EQA: FBE (frontier-based exploration) 扩展，conformal prediction 或 image-text matching 早停
- Patel et al.: multi-LLM agents 独立 yes/no answer → Central Answer Model 聚合

**Metrics**:
- Navigation: $d_T$ (final distance to target), $d_\Delta$ (distance change), $d_{min}$ (minimum distance during episode)
- QA: mean rank (MR), accuracy, **LLM-Match** (Majumdar et al. 引入的 aggregate LLM correctness metric)

### B. Embodied Grasping

**Gripper 类型**:
- **Two-finger parallel**: 4-DOF (top-down, position + yaw) 或 6-DOF (full 6D position + orientation)
- **Five-finger dexterous**: ShadowHand 有 26 DOF，复杂度显著上升

**Datasets** (Table X):
- 传统：Cornell (2011, real, 8K grasps), Jacquard (2018, sim, 1.1M), GraspNet-1Billion (2019, 7.07M), ACRONYM (2021, 17.7M), MultiGripperGrasp (2024, 30.4M, 2-5 fingers)
- Semantic：OCID-VLG (2023), ReasoningGrasp (2024, 99.3M), CapGrasp (2024, 50K)

**Language-guided Grasping**:

*Explicit Instructions*: 直接指定 object category (e.g., "grasp the banana")

*Implicit Instructions*: 需要 reasoning
- **Spatial reasoning**: "Grasp the keyboard that is to the right of the brown kleenex box" — 推断 spatial arrangement
- **Logical reasoning**: "I am thirsty, can you give me something to drink?" — 推断 human intent + 生成合理 grasp posture (不洒液体)

**End-to-End Approaches**:
- **CLIPort** (2022): CLIP + Transporter Net 双流，semantic understanding + grasp generation
- **CROG** (2023): 基于 OCID dataset，CLIP visual foundation → 直接 image-text pair 学 grasp synthesis
- **Reasoning Grasping** (2024): GraspNet-1Billion + MLLM 推理 grasping
- **SemGrasp** (2024): discrete representation aligning grasp space 和 semantic space，生成 dexterous hand posture

**Modular Approaches**:
- **F3RM** (2023): CLIP feature 提升 到 3D space → language localization → grasp generation
- **GaussianGrasper** (2024): 3D Gaussian field → feature distillation → language localization → SOTA grasping network

---

## V. Embodied Agents

这是论文最核心的章节，对应 Fig. 13 的整体架构。

### A. Embodied Multimodal Foundation Model

**Google DeepMind 的 RT 系列演进** 是这条线的主轴：

```
SayCan (3 separate models: planning + affordance + policy)
    ↓
Q-Transformer (unify affordance + policy)
    ↓
PaLM-E (integrate planning + affordance)
    ↓
RT-2 (Vision-Language-Action model: 统一全部三个) ← breakthrough
    ↓
RT-H (action hierarchies, 中间层 linguistic actions)
```

**RT-2 的关键创新**:
- Vision-Language-Action (VLA) model，把 robot action token 化为 text token
- Co-training on web-scale vision-language data + robot data
- Chain-of-thought reasoning capability → multi-step semantic reasoning
- 局限：inference frequency 只有 1-3 Hz

**RT-X / Open X-Embodiment** (2023):
- 21 institutions 合作
- 22 种不同 robots
- 527 skills, 160,266 tasks
- 证明 diverse cross-entity training data 比 domain-specific data 更 generalizable
- https://robotics-transformer-x.github.io/

**效率优化**:
- **SARA-RT** (2023): "up-training" 把 quadratic complexity 转为 linear complexity
- **RoboMamba** (2024): Mamba 架构处理 long sequence，Policy Head 只需 0.1% 参数 + 20 分钟 fine-tune，inference speed 7× faster

**RT-Trajectory** (2023): 自动添加 robot trajectory 作为 visual cue，弥补 generative model 在 low-level control 上的不足

### B. Embodied Task Planning

Task planning 是 "thinking before acting"，发生在 cyberspace。例如 "put an apple on a plate" → subtasks ["find the apple", "pick the apple", "find the plate", "put down the apple"]。

**Three waves**:

**Wave 1: Pre-LLM Symbolic Planning**
- STRIPS, PDDL + MCTS, A*
- 依赖 predefined rules，rigid，难适应 dynamic environment

**Wave 2: LLM as Plan Generator**
- **Translated LM** / **Inner Monologue**: LLM 用 internal world knowledge + CoT 分解复杂任务
- Few-shot prompt 加 successful plan examples
- **LLM-Planner**: KNN 检索 task-similar examples
- Skill memory bank：将 past success 抽象成 skills 存储复用
- **ReAct**: CoT + plan generation
- **Chain of Code**: 用 code 作为 reasoning medium 而非 natural language
- **Socratic Models / Socratic Planner**: Socratic questioning 修正 hallucination

**Wave 2 的问题**: LLM 基于 token probability distribution，不能保证 logical correctness

- **LLM as World Model + MCTS** (Zhao et al., Hao et al.): 用 MCTS 搜索 plan sequence
- **LLM as Instruction Translator** (LLM+P, Silver et al.): LLM 翻译 natural language → PDDL → 传统 planner 执行

**Wave 3: Vision-Integrated Planning**
- **LLM-Planner / Socratic Models**: 用 object detector query 当前环境 → 反馈 LLM 修改 plan
- **RoboGPT**: 处理 similar objects 在同一 task 中的不同 names
- **SayPlan**: hierarchical 3D scene graphs 表示 multi-floor, multi-room 环境
- **ConceptGraphs**: open-world 3D scene graph + code-based task planning

**Wave 4: VLM-based Planning**
- **EmbodiedGPT**: Embodied-Former 对齐 embodied/visual/textual information
- **EIF-Unknow**: Voxel Features → Semantic Feature Maps → visual tokens + text tokens → LLaVA
- **Matcha**: LLM + 多模态感知（weight, touch, sound），如 "pick up a plastic block" → 评估 weight + tap sound + tactile hardness
- **VLP** (Video Language Planning): LLM 生成 action → video model 模拟多个 potential video representation → 作为 heuristic function 评估 action 到 goal 的 proximity

### C. Embodied Action Planning

Action planning 是 "interacting with environment"，发生在 physical world。

**两条路线**:

**1. Action via APIs (Modular)**:
- LLM 接收 policy models 的 definitions + descriptions 作为 context
- 决定 how/when invoke which tool
- **Code as Policies**: 把 granular tools 抽象成 function library
- **Reflexion**: execution 时调整工具提升 generalization
- **DEPS**: zero-shot 下 LLM 学习 diverse skills 并组合新 skills

优点：模块化，independent development/testing/optimization
缺点：external policy model 调用 latency，agent 性能上限被 policy model 质量 bound

**2. Action via VLA Model (Unified)**:
- Task planning + action execution 在同一 system
- 减少 communication latency，支持 real-time feedback
- 代表：RT-2, EmbodiedGPT, PaLM-E
- 关键问题：没有 embodied world model 的 VLA model 无法用 LLM 内部 knowledge 模拟 physical laws

---

## VI. Sim-to-Real Adaptation

### A. Embodied World Model

论文强调 VLA model 和 World Model 的本质区别：
- **VLA**: 先在 large-scale internet data 上 pretrain 获得高层 emergent capability，再 co-finetune with real robot data
- **World Model**: 从 scratch 在 physical world data 上训练，随 data 增加 逐渐发展 高层 capability，但仍是 low-level physical world model（类似 human neural reflex system）

**三类方法** (Fig. 14, Table XI):

**1. Generation-based Methods**:
- **World Models** (Ha & Schmidhuber, 2018): Car Racing 早期工作
- **Sora** (2024): video generation 学到 physical laws
- **Pandora** (2024): real-time controllable video generation
- **3D-VLA** (2024): 3D vision-language-action generative world model
- **DWM** (2024): Diffusion World Model for D4RL offline RL

核心思想：generative model 内化 world knowledge，可以通过 mining + utilizing 该 knowledge 增强其他 model

**2. Prediction-based Methods**:
- **I-JEPA** (LeCun et al., 2023): Joint-Embedding Predictive Architecture，在 latent space 预测而非 pixel space，是 LeCun 的核心思想: https://ai.facebook.com/blog/yann-lecun-ai-i-jepa/
- **MC-JEPA** (2023): motion + content features
- **A-JEPA** (2023): audio representation learning
- **Point-JEPA** (2024): point cloud self-supervised learning
- **IWM** (2024): learning and leveraging world models in visual representation
- **iVideoGPT** (2024): interactive video GPT 作为 scalable world model
- **STP** (2024): spatiotemporal predictive pre-training for robotic motor control
- **MuDreamer** (2024): DeepMind Visual Control Suite

优势：latent space 抽象 + decouple knowledge，处理 complex scene 更高效，generalization 强

**3. Knowledge-driven Methods**:
- **ElastoGen** (2024): 4D generative elastodynamics，注入弹性力学知识
- **Liu et al.** (2024): single-image 3D reconstruction
- **Holodeck** (2024): LLM 自动 generate 3D embodied AI environments
- **LEGENT** (2024): open platform for embodied agents
- **real2sim2real** (2022): real-world knowledge → physics-compliant simulator → train robot

### B. Data Collection and Training

**Real-World Data**:
- **Open X-Embodiment**: 21 institutions, 22 robots, 527 skills, 160,266 tasks
- **UMI** (2024): handheld gripper + elegant interface，portable low-cost，支持 bimanual dynamic demonstration: https://umi-robot.github.io/
- **Mobile ALOHA** (2024): low-cost full-body mobile manipulation system，frying shrimp + serving dishes: https://mobile-aloha.github.io/
- **Human-Agent Collaboration** (Luo et al. 2024): human 提供 initial action → agent 迭代 perturbation + denoising → 优化 → 高质量 demonstration

**Simulated Data**:
- CLIPort / Transporter Networks: PyBullet simulator 收集 demonstration
- GAPartNet: large-scale part-centric interactive dataset
- SemGrasp / CapGrasp: virtual environment 中构建 grasp-text aligned dataset

**Five Sim2Real Paradigms** (Fig. 16):

1. **Real2Sim2Real** (2024):
   - Nerf/VR 扫描重建 → import simulator → RL fine-tune initial strategy → transfer real world
   - 用 simulation 的 "digital twin" 增强 real-world imitation learning

2. **TRANSIC** (2024):
   - RL 在 simulation 训练 foundation policy
   - 部署到 real robot，human real-time intervene + correct via remote control
   - Collected intervention data 训练 residual policy
   - Foundation policy + residual policy = smoother real-world trajectory

3. **Domain Randomization** (2017+):
   - 在 simulation training 时 randomize parameters (friction, gloss, lighting, etc.)
   - 覆盖 real-world 可能的 variation 范围
   - OpenAI Dactyl (Learning Dexterous In-Hand Manipulation) 是经典例子: https://arxiv.org/abs/1808.00177

4. **System Identification**:
   - 构造 real-world physical scene 的 accurate mathematical model
   - 包括 dynamics parameters 和 visual rendering parameters
   - 让 simulation 尽可能 close to real-world

5. **Lang4Sim2Real** (2024):
   - 用 natural language 作为 bridge
   - Image 的 textual description 作为 cross-domain unified signal
   - Pretrain encoder on cross-domain language-annotated images
   - 学 domain-invariant image representation → multi-domain language-conditioned behavioral cloning

### C. Embodied Control

**Deep Reinforcement Learning (DRL)**:
- **HDPG** (2022): Hybrid Dynamic Policy Gradient for biped locomotion，multi-criteria dynamic optimization
- **DeepGait** (2020): neural network policies for terrain-aware locomotion，model-based motion planning + RL
  - Terrain-aware planner 生成 gait sequence + base motion
  - Gait + base motion controller 执行 + maintain balance
  - 都用 neural network function approximator + DRL 优化

**Imitation Learning**:
- 减少 trial-and-error data 需求
- **Offline RL + Online RL**: 先 offline 学 policy from static dataset，再 online 交互调整
- **ALOHA** (2023): 低成本 bimanual manipulation hardware，从 human demonstration 学 fine-grained dexterous bimanual operations
- **Mobile ALOHA** (2024): ALOHA + full-body mobility

**Robotic Control 突破**:
- **Visual Whole-Body Control** (2024): robotic arm + robotic dog，12 leg joints + 6 arm joints + 1 gripper，track speed + end-effector position
- MIT Cheetah 3, ANYmal, Atlas: robust walking controllers
- **Expressive Whole-Body Control for Humanoid** (2024): humanoid robot 的 expressive motion

---

## VII. Challenges and Future Directions

论文最后列出 8 大挑战：

1. **High-quality Robotic Datasets**: real-world data collection 时间和资源消耗巨大，sim-only training 加剧 sim-to-real gap。需要 cross-institution collaboration + 更真实 efficient simulator

2. **Efficient Utilization of Human Demonstration Data**: R3M 等已有 work 在简单 grasping 上 high success rate，但复杂 task efficiency 仍需 improvement。要 effective 利用 unstructured multi-label multi-modal human demonstration

3. **Cognition of Complex Environment**: SayCan 等 LLM-based decomposition 依赖 commonsense 但缺乏 long-term 复杂环境理解。要 enhance knowledge transfer + generalization

4. **Long-Horizon Task Execution**: "clean the kitchen" 涉及 rearranging, sweeping, wiping 等 long sequence。当前 high-level planner 在 diverse scenario 不足。需要 efficient planner + robust perception + commonsense knowledge

5. **Unified Embodied Foundation Model**: robotics 的 embodiment/environment/task 多样性 + isolated datasets/eval setups 是障碍。要 leverage large-scale internet data + LLM/MLM/WM

6. **Causal Relation Discovery**: data-driven agent 基于 intrinsic correlation 决策，缺乏 causal understanding。要构建 embodied perception + reasoning + interaction 框架 driven by world knowledge + counterfactual + causal intervention

7. **Continual Learning**: 在 diverse environment 部署 robot policy 仍是 largely unexplored。包括 incremental learning, rapid motor adaptation, human-in-the-loop learning, catastrophic forgetting mitigation

8. **Unified Evaluation Benchmark**: 现有 benchmark 多评估 isolated skill 或 isolated planner，需要 holistic benchmark 同时评估 high-level task planner + low-level control policy 在 long-horizon task 上的 success rate

---

## Intuition Building: 关键 insight 总结

这篇综述给 Andrej 你这样的研究者 build intuition 的几个关键点：

1. **VLA model 不是终点，World Model 是下一个 frontier**。RT-2 用 web-scale data + robot data co-training 实现了 emergent capability，但本质还是 pattern matching + token generation。World Model 从 scratch 学 physical law，是 LeCun 的 JEPA 路线在 robotics 上的延伸。

2. **Sim-to-real 的五大 paradigm 各有适用场景**：Domain Randomization 适合 robust locomotion，Real2Sim2Real 适合 manipulation，Lang4Sim2Real 是 LLM 时代的新方向。

3. **Embodied agent 的 hierarchy 必然存在**：Task planning (cyber space, LLM) + Action planning (physical world, VLA/API) + Control (RL/imitation)。RT-H 用 linguistic action 作为中间层是当前最 promising 的 hierarchy design。

4. **Tactile 是 underexplored 的 modality**。Vision-based tactile sensor (GelSight) 让 tactile 可以 align 到 CLIP-style multimodal space，但 dataset 标准化 + cross-sensor generalization 仍是 open problem。

5. **Causal reasoning 是 long-horizon task 的 key**。GOAT 用 BACL/FACL (causal learning) 在 VLN 上有突破，论文作者自己的 work 也包含 cross-modal causal reasoning for VQA。

6. **Continual learning 在 embodied AI 几乎空白**。Pretrained model 在 robotics control framework 的 seamless integration，hierarchical learning 或 slow-fast control，都是 open problem。

---

## Reference Links

- Paper: https://arxiv.org/abs/2407.06886
- HCPLab GitHub: https://github.com/HCPLab-SYSU/Embodied-AI-Paper-List
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- Isaac Sim: https://developer.nvidia.com/isaac-sim
- Habitat: https://aihabitat.org/
- AI2-THOR: https://ai2thor.allenai.org/
- iGibson: https://igibson.stanford.edu/
- SAPIEN: https://sapien.ucsd.edu/
- LeCun I-JEPA: https://ai.facebook.com/blog/yann-lecun-ai-i-jepa/
- ALOHA: https://tonyzhaozh.github.io/aloha/
- Mobile ALOHA: https://mobile-aloha.github.io/
- UMI: https://umi-robot.github.io/
- RT-2: https://robotics-transformer2.github.io/
- RT-X: https://robotics-transformer-x.github.io/
- OpenEQA: https://open-eqa.github.io/
- RT-H: https://rt-h.github.io/
- Sora: https://openai.com/sora
- ProcTHOR: https://procthor.allenai.org/
- AnyGrasp: https://graspnet.net/
- Dactyl (Domain Randomization 经典): https://arxiv.org/abs/1808.00177

这篇 survey 是 Embodied AI 在 MLM 时代的奠基性 reference，其 taxonomy 把 hardware / simulator / perception / interaction / agent / sim-to-real / future direction 串成完整 picture，特别适合用来 build cross-domain intuition。对你 Andrej 来说，从 World Model + VLA hierarchy + Tactile modality 这三个方向切入，应该是最有 fertile exploration space 的方向。
