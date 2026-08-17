---
source_pdf: Learning Embodied Intelligence from Physical Simulators and World Models.pdf
paper_sha256: 81c1f3a7a5d526ce3450e818445c89365832c161f41cc7ffe1b7cddc6d023b54
processed_at: '2026-08-05T12:54:49-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Survey

## 一句话总结

这篇 paper 说的事情其实很简单：**要让 robot 真正聪明，光靠 explicitly 写物理方程的 simulator 不够，得让 robot 自己"脑补"一个世界模型（world model），在脑子里推演未来会发生什么。** 这两条路线一直在并行走，现在开始 merge，这会催生真正的 embodied intelligence。

---

## 为什么需要这两条路线

先讲个最直观的故事。

假设你是一个 baby，刚学会走路。你不会先在脑子里解一个 Lagrangian equation 算重心位置，再决定迈哪只脚。你靠的是**反复 trial-and-error + 脑子里有个粗糙的"如果我这样动会怎样"的预测模型**。这个"脑补模型"就是 world model 的雏形。

而传统的 robotics 走的是另一条路：把所有 physics 写死，用 MuJoCo 这种 simulator 精确算每根 joint 的 torque。这条路在工业机器人时代很好用，因为环境是 structured 的。但一旦放到野外，simulator 的几个致命问题就暴露了：

- **Friction 永远不准**：real wood 和 sim wood 的摩擦系数差 20% 很正常
- **Contact dynamics 是 nightmare**：两个物体接触瞬间发生什么，simulator 用 penalty method 或 LCP 简化，真实世界是 elastic + plastic + fracture 的混合
- **Sensor noise 模型太理想**：真实 LiDAR 有 multipath、bloom、motion distortion，simulator 里都是 ideal Gaussian

所以 sim-to-real gap 是**结构性**的，不是调参能解决的。

paper 在 Section 4.5 把这个讲得很清楚：

> Simulators face challenges: Accuracy, Complexity, Data dependency, Overfitting

而 world model 走的是 opposite philosophy：**不要 explicit 建模，让 model 从 data 里自己学 dynamics**。这相当于让 robot 像 human baby 一样，通过观察和互动，自己 build 一个 internal model of the world。

这两条路现在开始 converge：
- Simulator 提供 high-fidelity data 训练 world model
- World model 反过来作为 implicit physics engine，可以 generate OOD scenarios，弥补 simulator 的局限

这就是 paper 的核心 thesis。

---

## IR-L0 到 IR-L4：机器人版本的"自动驾驶 L0-L5"

paper 提了个 5 级分类。我觉得最 interesting 的维度是 **Societal Cognition Ability**——这是 robotics 比 autonomous driving 多出来的维度。

**IR-L0**：工厂里的 welding robot，PLC 控制，按预编路径走。完全没 intelligence。

**IR-L1**：扫地机器人，碰到墙就转，红外避障。有最 basic 的反应能力，但只是 if-else rule。

**IR-L2**：餐厅 service robot，能听"送水"指令，能 SLAM 导航，能避障。开始有 contextual understanding，但还是 task-specific。

**IR-L3**：eldercare robot，能识别老人情绪，能多轮对话，能在 dynamic 环境做 decision。这一级开始需要 emotional intelligence 和 ethical governance——paper 在 Section 2.3.4 明确说要 "embedded ethical governance systems"。这是 humanoid robot 比 autonomous vehicle 多出来的需求：car 不需要理解你今天心情不好。

**IR-L4**：fully autonomous，能在任何环境 self-evolve。paper 说需要 "AGI frameworks integrating meta-learning, generative AI, embodied intelligence"。这一级现在是科幻，但 paper 暗示 world model 是通往这级的 key enabler——因为只有学会在脑内 simulate 任意 future，才能做 open-ended innovation。

我的 take：这个分级体系最有用的地方是把 "Societal Cognition" 显式纳入。现在大多数 robotics paper 只 talk about dexterity 和 locomotion，但其实 deployment 到人类社会，social intelligence 才是 bottleneck。一个 mechanical perfect 但 socially awkward 的 robot 在 real world 会处处碰壁。

---

## 物理模拟器的演进：从精确到可微

### 三个时代

**时代 1 (1998-2012): 精确但慢**
- Webots、Gazebo 这一代，CPU 跑，single environment
- MuJoCo 2012 出现，用 convex optimization 处理 contact，比 penalty method 干净很多
- 这个时代的 simulator 主要用于 verification，不能 scale

**时代 2 (2017-2023): GPU 加速**
- Isaac Gym 2021 是分水岭：**single GPU 上 parallel simulate thousands of environments**
- 这是 RL training 的 enable——以前一个 robot 学走路要 sim 几个月，现在几天
- PhysX 5 支持 deformable body、soft contact、fluid (particle-based)
- 但 rendering 还是大问题，直到 Isaac Sim 集成 Omniverse RTX 才有 ray tracing

**时代 3 (2024-2025): 可微物理**
- MuJoCo XLA (JAX backend)、Newton (NVIDIA+DeepMind+Disney)、Genesis
- 核心能力：**gradient 可以 through simulation backpropagate**
- 这意味着你能直接 optimize "让 robot 跳得更高"，gradient 告诉你怎么调 joint stiffness

可微物理为什么是 big deal？传统 simulator 是 black box，你只能 sample-based 优化（像 CMA-ES、random search）。可微之后，你能做 gradient-based optimization，sample efficiency 提升 orders of magnitude。

### MuJoCo 的核心技术 intuition

MuJoCo 为什么快又稳？因为它把 contact 当作**凸优化问题**：

$$\min_v \frac{1}{2} v^T M v + v^T c \quad \text{s.t.} \quad Av + b \geq 0$$

- $v$ 是所有 joint 的 generalized velocity
- $M$ 是 mass matrix（稀疏结构，computable in $O(n)$）
- $c$ 包含 Coriolis、centrifugal、gravity 这些 bias force
- $Av + b \geq 0$ 是 contact 不穿透的 constraint

这是 convex QP，能 globally solve。而 Bullet、ODE 用的是 iterative Gauss-Seidel，可能 oscillate。这就是为什么 MuJoCo 在 humanoid 上特别 popular。

参考：https://mujoco.org/

### Isaac 系列的 GPU 加速

Isaac Gym 的 trick：
- PhysX GPU 用 warp-based parallel solver
- 每个 CUDA thread 处理一个 contact constraint
- Block-level reduction 做 Jacobian assembly
- 1000+ environments 同时跑，FPS 数百万

Isaac Sim 在此基础上集成 Omniverse RTX renderer，做 ray-traced LiDAR simulation，精度到 millimeter 级。这对 sim-to-real 至关重要，因为 LiDAR noise model 在传统 simulator 里太 idealized。

Isaac Lab 进一步加 tiled rendering，multi-camera throughput 提升 1.2×。它还支持 HDF5 demonstration data 直接 train imitation learning policy。

参考：https://developer.nvidia.com/isaac/sim

### Genesis：all-in-one

Genesis 2024 出来，野心更大：把所有 physics solver 统一在一个 framework 里。

- **MPM (Material Point Method)**: 处理大变形、流体、颗粒物
- **SPH (Smoothed Particle Hydrodynamics)**: 流体
- **FEM (Finite Element Method)**: 弹性体
- **PBD (Position-Based Dynamics)**: 实时 cloth、rope

Genesis benchmark：在 512-32768 environments 上 throughput 比 Isaac Gym 快 2.70×-11.79×。这主要是因为它的 architecture 从一开始就为 massive parallelism 设计。

而且 Genesis 内置 generative data engine，用 natural language prompt 就能 generate scenario。这是 simulator + world model convergence 的雏形。

参考：https://github.com/Genesis-Embodied-AI/Genesis

### Newton：2025 年新玩家

Newton 是 NVIDIA + DeepMind + Disney Research 联合开发的，2025 年 3 月才发布。亮点：

- 基于 NVIDIA Warp framework，**GPU 70× speedup**
- Differentiable physics，gradient 能 backprop
- OpenUSD scene construction
- 与 MuJoCo Playground、Isaac Lab 兼容

Newton 想做的是"physics engine for the AI era"——传统 physics engine 是为 graphics 和 engineering simulation 设计的，Newton 是为 robot learning 设计的，differentiable 是 first-class citizen。

参考：https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/

---

## World Model：三种角色，一个本质

这是 paper 最 valuable 的部分。它把 world model 在 embodied AI 里的角色清晰地分成三个：

### Role 1: Neural Simulator（数据生成器）

world model 当作"会做梦的 simulator"。给它一个初始画面和一个 action，它 generate 接下来几秒会发生什么。

**GAIA-1** 是开山之作。Wayve 用 9B transformer 在 4700 小时 driving data 上训练，发现 emergent behavior：模型能"推理"agent 交互。比如你给它一个左转 action，它会生成"车左转 + 对向车避让 + 行人停下"的合理 video。这没 explicit 编进 model，是从 data 里学到的。

**GAIA-2** 从 autoregressive 转向 diffusion，加 structured conditioning：ego-vehicle state、road layout、weather、time of day。这让 model 可以 controllable 生成各种 corner case：暴雨夜 + 高速 + 突然并道。

**NVIDIA Cosmos** 想做 general-purpose world foundation model。架构是 Diffusion + Autoregressive 混合：

- Video 通过 Cosmos-Tokenize1-CV8×8×8-720p encode 到 latent
- Latent 加 Gaussian noise
- 3D patchification 后过 self-attention + cross-attention (text condition) + MLP
- Decoder 重建 video

Cosmos-Transfer1 进一步加 spatially conditioned control：你可以传 segmentation map、depth、edges，model generate 对应的 photorealistic video。

**为什么 neural simulator 重要？** 因为传统 simulator 生成数据有两个死结：
1. Asset 有限：你要 1000 种不同的 chair 模型，传统 simulator 要建模 1000 个 3D mesh
2. Lighting、texture 太"完美"：simulator 里的材质都是 PBR idealized，real world 有 scratch、dust、stain

Neural simulator 从 internet video 学到真实材质分布，generate 的数据天然有 realism。

### Role 2: Dynamic Model（规划引擎）

这是 model-based RL 的核心。agent 学一个 dynamic model $f: (s_t, a_t) \rightarrow s_{t+1}$，然后在 model 内"想象"未来，做 planning。

**Dreamer 系列** 是这条路线的代表。它的 RSSM 架构是 key innovation。

为什么 RSSM 厉害？因为它把 deterministic 和 stochastic 结合了：

- **Deterministic path** $h_t = f(h_{t-1}, s_{t-1}, a_{t-1})$：保证 long-term memory 不丢
- **Stochastic path** $s_t \sim q(s_t | h_t, o_t)$：handle multi-modal future（同一个 state 可以有多种 future）

这解决了两个传统方法的痛点：
- Pure RNN：deterministic，无法 represent multiple futures，会被 planner exploit
- Pure SSM：stochastic 但 long-term memory 差，因为每个 step 都要 sample

Dreamer 的训练 objective 是 ELBO 的变体：

$$\mathcal{L} = \sum_t \left[ \underbrace{\log p(o_t|s_t, h_t)}_{\text{reconstruction}} + \underbrace{\log p(r_t|s_t, h_t)}_{\text{reward pred}} - \underbrace{D_{KL}(q(s_t|h_t, o_t) \| p(s_t|h_t))}_{\text{posterior-prior gap}} \right]$$

DreamerV3 在 2023 年达到 single configuration 跨 150+ tasks SOTA，包括 Atari、DMC、Minecraft diamond collection。这是 unified world model 的 proof of concept。

参考：https://danijar.com/project/dreamerv3/

**PlaNet** 是 Dreamer 的前作，2018 年提出 latent dynamics planning。它的 reward 是 explicitly learned predictor：从 latent state 预测 reward，而不是像 Dreamer 那样 implicit 通过 value function。

**DayDreamer** 把 Dreamer 拿到 real robot 上跑，证明 latent world model 能 sim-to-real transfer。

### Role 3: Reward Model（implicit reward）

RL 最痛苦的是 reward design。手动设计 reward function 容易 reward hacking，又容易 miss 真正想要的 behavior。

**VIPER** 提了个 elegant idea：**用 expert video 训一个 video prediction model，这个 model 的 prediction likelihood 就是 reward**。

intuition 很 simple：如果 agent 的 action 让 trajectory 看起来"很 expert-like"，那 prediction model 会觉得这个 trajectory 很 predictable，likelihood 高，reward 高。

公式：
$$r_t = \log p_\theta(o_t | o_{<t}, a_{<t})$$

这相当于把 expert demonstration 当 prior，agent 的 reward = 与 prior 的 alignment。

为什么 powerful？因为：
1. 不需要设计 reward function
2. expert video 容易获得（YouTube 上大量 human activity video）
3. Cross-embodiment generalization：用 human hand video 训的 model，能给 robot hand policy 当 reward

V-JEPA 2 是 LeCun 路线的最新作品，1.2B 参数，在 1M+ video hours 上 actionless pretrain，再 62 小时 robot data fine-tune，novel environments 上 65%-80% 成功率。这暗示了 self-supervised pretrain + small-scale fine-tune 的范式可能 work。

参考：https://ai.meta.com/blog/v-jepa-2-world-model-background-knowledge-enables-robotic-planning/

---

## 三种架构的 intuition

paper 识别出 5 种 architecture，但我觉得最 fundamental 的区分是三个：

### 1. Reconstruction-based (Dreamer 系列)

**思路**：学一个 encoder + decoder，把 observation encode 成 latent，predict latent 演化，再 decode 回 observation。

**Pros**: 显式 model observation distribution，sample efficient
**Cons**: pixel reconstruction 浪费 capacity，可能学到无关 detail（比如 background 纹理）

### 2. Predictive (JEPA 系列)

**思路**：在 latent space predict，不 decode 回 pixel。学一个 energy function $E(x, \hat{x}) = \|\phi(x) - \psi(\hat{x}_{context})\|^2$ 衡量"target representation" 和 "predicted representation" 的距离。

intuition：你不需要 predict 每个像素，只需要 predict "high-level 语义会怎么变"。比如 ball 在飞，你不需要 predict 球表面每个像素怎么动，只需要 predict "ball 位置 + 速度"。

**Pros**: 不浪费 capacity 在 irrelevant detail，pretraining 效率高
**Cons**: 不能直接 generate observation（这是缺点也是优点，看 task）

### 3. Generative (Diffusion / Autoregressive)

**思路**：直接 model $p(o_{t+1}|o_t, a_t)$，generate 下一帧或下一段 video。

**Diffusion** 优势：高 fidelity，能 generate multi-modal distribution
**Autoregressive** 优势：long-horizon consistency（因为是 sequential）
**Hybrid** (Vid2World, Epona)：combine both

**Pros**: 直接 output 可用 video，能当 neural simulator
**Cons**: sample 慢（diffusion 要几十步 denoise）；训练 expensive

---

## Autonomous Driving 和 Robotics 的 Convergence

paper 在 Section 6 开头讲了个有意思的 observation：

> "Autonomous driving vehicles can be seen as an intelligent robot with four wheels together with a smaller action space comparing to humanoid robots."

这暗示了 embodied intelligence 的 unified framework 可能存在。

实际上 Tesla 在做这个事——FSD 和 Optimus 共享 visual encoding backbone。

但两者差异显著：

| 维度 | Autonomous Driving | Articulated Robot |
|------|--------------------|-------------------|
| Action space | 3-DoF (steering, accel, brake) | 7-DoF arm 或 30+ DoF humanoid |
| Horizon | 秒级 (8-12s planning) | 分钟级 (manipulation task) |
| Safety criticality | open road, human life | controlled environment |
| Observation | multi-camera + LiDAR | multi-modal + tactile |
| Contact | 几乎无（除非撞了） | core challenge |
| Generalization | 地理、天气 | object、layout、embodiment |

这解释了为什么 driving 的 world model 主要是 video generation（neural simulator role），而 robotics 的 world model 主要是 latent dynamics model（dynamic model role）——driving 不太需要精细 contact reasoning，robotics 必须做 contact。

paper 列了大量 driving world model：GAIA-1/2、DriveDreamer 1/2/4D、MagicDrive 1/3D/V2、OccWorld、DriveWorld、Cosmos-Drive、InfinityDrive、ReconDreamer 等等。但 robotics world model 的 dynamic model role 工作更多：Dreamer 系列、PlaNet、DayDreamer、SWIM、DWL、Puppeteer、Surfer、TWIST、WMP、RWM、SSWM、WMR、PIN-WM、MoDem-V2、V-JEPA 2 等。

这个差异本身就是 insight：driving 的核心 problem 是 scene generation（neural simulator），robotics 的核心 problem 是 control via internal simulation（dynamic model）。

---

## 关键 Open Problems

paper 列了 9 个 challenge，我认为最 critical 的是这三个：

### 1. Causal Reasoning vs Correlation

现在的 world model 学 correlations。比如 model 学到"红灯 + 车停"，但不知道是红灯导致停车，还是停车导致红灯。这导致 OOD 时 model 会 hallucinate。

Causal reasoning 需要什么？需要 model 能回答 counterfactual："如果我刚才不刹车会怎样？"

这是当前 generative model 的 fundamental limitation。可能的解法：
- Structural causal model + neural network hybrid
- Interventional training：在 data 里 augment "if action=A instead of B" 的对比
- Causal discovery methods (NOTEARS、CAM) 集成到 WM training

### 2. Compositional Generalization

Human 学会 "cup" 和 "table" 后，立刻理解 "cup on table"。Current WM 需要看大量 "X on Y" examples 才能 generalize。

可能的解法：
- Object-centric representations（FOCUS、DreamerPro 已经在做）
- Slot attention
- Disentangled latent spaces
- Symmetry-aware architectures（equivariant networks）

### 3. Systematic Benchmarking

现在 world model 评测用 FID、FVD、LPIPS 这些 generative metrics。但一个 FID 低的 model 不一定 planning performance 好。

需要的 benchmark：
- Downstream task performance（end-to-end evaluate）
- Counterfactual reasoning（"if action changed, what would happen?"）
- Long-horizon consistency（10s+ 后还合理吗？）
- OOD robustness

NAVSIM、Bench2Drive 是 driving 方向的尝试。Robotics 方面 WorldEval、SeaWave、DreamGen Bench 是 emerging efforts。

---

## 我的几个 Personal Take

1. **Differentiable physics + World Model 是 future**。Newton 和 Genesis 代表 simulator 走向 differentiable，V-JEPA 2 和 DreamerV3 代表 world model 走向 general-purpose。这两条路最终会 merge：differentiable physics 提供 inductive bias，world model 提供 data-driven flexibility。

2. **V-JEPA 2 的 paradigm 可能是 robotics 的 GPT moment**。1M+ video hours pretrain + 62 hours robot fine-tune = 65-80% success on novel env。如果这个 scaling law 成立，robotics 会从"每个 task 重新 train"进入"pretrain + fine-tune"时代。

3. **Diffusion Policy 的成功暗示 diffusion 在 action space 也 work**。这 opens up diffusion-based planning（像 Diffuser、Decision Diffuser）替代 model predictive control。π₀ 已经在做这个——用 flow matching 生成 action。

4. **Tactile world model 是 under-explored frontier**。paper 在 future direction 提到但没深入。Vision-only world model 处理不了"筷子夹豆腐"这种 fine-grained manipulation。Tactile sensing + world model 是 next big thing。

5. **Sim-to-real gap 可能根本 close不了，但可以 manage**。Domain randomization、teacher-student distillation、residual physics、digital cousins 都是 management 策略。最终可能是 simulator 提供 scaffold，world model 提供 real-world refinement。

6. **The "bitter lesson" of robotics**：long-term 看，general method（large-scale learning）会 beat specialized method（hand-crafted control）。MPC、WBC 这些 model-based 方法短期内还在，但 world model pretrain + RL fine-tune 的范式会 dominate。

参考：
- World Models original paper: https://arxiv.org/abs/1803.10122
- DreamerV3: https://danijar.com/project/dreamerv3/
- V-JEPA 2: https://ai.meta.com/blog/v-jepa-2-world-model-background-knowledge-enables-robotic-planning/
- NVIDIA Cosmos: https://www.nvidia.com/en-us/glossary/world-models/
- Genesis: https://github.com/Genesis-Embodied-AI/Genesis
- Newton: https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/
- Survey repo: https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey
- Sora: https://openai.com/research/video-generation-models-as-world-simulators
- GAIA-1: https://arxiv.org/abs/2309.17080
- VIPER: https://arxiv.org/abs/2305.14343
- LeCun JEPA position paper: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- π₀: https://arxiv.org/abs/2410.24164

---

# Learning Embodied Intelligence from Physical Simulators and World Models 深度讲解

## Paper 整体定位

这篇 paper 是南京大学 Long Group 团队在 2025 年发布的 comprehensive survey，核心目标是**系统性地梳理 Embodied Intelligence 的两大 enabler：Physical Simulators 和 World Models**，并揭示它们之间的 complementary 关系。这个 survey 的独特之处在于**把 simulator 和 world model 放在同一个框架下讨论**，而以前的 survey 通常只聚焦其中一方。

paper 的核心 thesis：传统 robotics 发展依赖 explicit physics-based simulator（如 MuJoCo、Isaac Sim），但这些 simulator 存在 accuracy、complexity、data dependency 和 overfitting 的瓶颈；而 world model 作为 implicit learned representation，可以提供更 flexible、更 adaptable 的 environment modeling，从而 bridge sim-to-real gap。

paper 的 GitHub repo：https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey

---

## IR-L0 到 IR-L4：Intelligent Robot 分级标准

paper 提出了一个 5 级分类体系，从四个维度评估：
- **Autonomy**: Human Control → Full Autonomy
- **Task Handling Ability**: Basic Tasks → Innovation
- **Environmental Adaptability**: Controlled Only → Universal Flexibility
- **Societal Cognition Ability**: No Social Cognition → Advanced Social Intelligence

### 各级别详细技术要求

**IR-L0 (Basic Execution Level)**:
- Hardware: PLC/MCU-based motion controllers，高精度 servomotors
- Perception: 极度有限，仅 limit switches 和 encoders
- 单向 closed-loop："command input - mechanical execution"

**IR-L1 (Programmatic Response Level)**:
- 引入 rule-based reactive capabilities（FSM、random walk）
- Basic sensors: infrared、ultrasonic、pressure
- 适用于 closed-task environments with clearly defined rules

**IR-L2 (Basic Perception and Adaptation)**:
- 关键跃迁：开始具备 contextual understanding
- Multimodal sensor arrays（cameras、LiDAR、microphone arrays）
- Behavior Trees + SLAM + path planning + obstacle avoidance
- 典型任务：service robot 执行 "water delivery" 或 "navigation guidance"

**IR-L3 (Humanoid Cognition and Collaboration)**:
- Autonomous decision-making in complex、dynamic environments
- Multimodal fusion: vision + speech + tactile
- Affective computing 用于 emotion recognition
- Deep learning architectures (CNNs、Transformers、RL)
- 例：eldercare robot 分析 speech patterns 和 facial expressions 检测情绪变化

**IR-L4 (Fully Autonomous Level)**:
- Self-evolving ethical reasoning
- AGI frameworks integrating meta-learning、generative AI、embodied intelligence
- Cloud-edge-client collaborative systems
- Multi-agent collaboration

**Intuition**: 这个分级体系让我想到 SAE 的自动驾驶分级 L0-L5，但增加了 Societal Cognition 这个维度，这是 robotics 独有的挑战。IR-L3 已经需要 emotional intelligence 和 ethical governance，这在目前的自动驾驶系统中是缺失的，但在 humanoid robot 中是必要的，因为 robot 需要 close collaboration with humans。

---

## Robotic Mobility、Dexterity 和 Interaction

### 3.1 核心技术方法

#### Model Predictive Control (MPC)

MPC 的核心是 optimization-based control strategy。在每个 time step，求解以下优化问题：

$$\min_{u_{t:t+N}} \sum_{k=0}^{N-1} \ell(x_{t+k}, u_{t+k}) + \ell_f(x_{t+N})$$

subject to:
- $x_{t+k+1} = f(x_{t+k}, u_{t+k})$ (dynamic model)
- $x_{t+k} \in \mathcal{X}$ (state constraints)
- $u_{t+k} \in \mathcal{U}$ (input constraints)

其中：
- $x_{t+k}$ 是 time step $t+k$ 的 state vector
- $u_{t+k}$ 是 input vector
- $N$ 是 prediction horizon
- $\ell(\cdot, \cdot)$ 是 stage cost
- $\ell_f(\cdot)$ 是 terminal cost
- $f(\cdot, \cdot)$ 是 system dynamics

关键里程碑：2015 年 Koenemann 等人首次在 HRP-2 humanoid robot 上实现 real-time whole-body MPC。

参考: https://ieeexplore.ieee.org/document/7353406

#### Whole-Body Control (WBC)

WBC 通过 prioritized task hierarchy 解决 redundant manipulator 的 control 问题。典型 formulation：

$$\min_{\dot{q}} \|J_1 \dot{q} - \dot{x}_1^*\|^2 + \sum_{i=2}^{n} w_i \|J_i \dot{q} - \dot{x}_i^*\|^2$$

或者 hierarchical formulation 用 nullspace projection：
$$\dot{q}_i = \dot{q}_{i-1} + (J_i N_{i-1})^+ (\dot{x}_i^* - J_i \dot{q}_{i-1})$$

其中：
- $J_i$ 是 task $i$ 的 Jacobian
- $N_{i-1}$ 是 nullspace projector of priority $i-1$
- $()^+$ 是 Moore-Penrose pseudoinverse
- $\dot{q}$ 是 joint velocity
- $\dot{x}_i^*$ 是 desired task-space velocity

Khatib 的 operational space formulation 是这里的奠基工作。

#### Reinforcement Learning

paper 提到 1998 年 Morimoto 和 Doya 的工作，用 RL 让 simulated 2-joint 3-link robot 自主学习 standing-up。现代 RL 应用包括：

- **DeepLoco (2017)**: hierarchical deep RL for bipedal tasks
- **Xie et al. (2019)**: iterative RL + Deterministic Action Stochastic State (DASS) tuples 让 Cassie bipedal robot 实现稳健 walking

DASS tuple 的核心 insight：在 sim-to-real 中，确定性 action 配合 stochastic state 表示，可以让 policy 在分布 mismatch 下仍然 robust。

#### Visual-Language-Action (VLA) Models

2023 年 Google DeepMind 的 RT-2 开创了这个范式。核心思路：将 robot action 离散化为 language-like tokens，从而利用 internet-scale visual-language pretraining。

RT-2 的 tokenization 方式：将 7-DoF end-effector action 离散成 256 bins，每个维度一个 token，总共 7 个 action tokens，与 text tokens 拼接输入 LLM backbone。

参考: https://robotics-transformer2.github.io/

### 3.2 Robotic Locomotion

#### Unstructured Environment Adaptation

关键技术演进：

1. **早期 position-controlled robots**: 高 gear-ratio 导致高 impedance，contact 时容易 damage
2. **Force-controlled joints (low gear ratio)**: 提供更好的 compliance 和 smooth response
3. **Cassie bipedal robot 的 full-body dynamic controller (Reher et al.)**: 显式建模 passive spring mechanisms
4. **DCM (Divergent Component of Motion) + WBC**: Mesesan et al. 在 TORO robot 上实现 soft mat 上的 dynamic walking

DCM 定义：
$$\xi = x_{CoM} + \frac{\dot{x}_{CoM}}{\omega}$$

其中 $\omega = \sqrt{g/l}$ 是 LIPM 的 natural frequency，$g$ 是 gravity，$l$ 是 leg length。DCM 的 divergent 特性使得 capture point control 成为可能。

5. **Learning-based methods**: 
   - 2020 Lee et al.: 首次成功在 outdoor 环境 real-world 应用 RL 到 legged locomotion
   - Siekmann et al.: blind stair traverse with domain randomization
   - Perceptive Internal Models (PIM): 利用 depth camera + LiDAR 构建 height maps

#### High Dynamic Movements

Simplified models:
- **SLIP (Spring-Loaded Inverted Pendulum)**: $\ddot{z} = -g + \frac{k}{m}(L_0 - z)$
- **LIPM (Linear Inverted Pendulum Model)**: $\ddot{x} = \frac{g}{z_0} x$（$z_0$ 是 constant CoM height）
- **SRBM (Single Rigid Body Model)**: 将整个 robot 当作 single rigid body

CDM-MPC (He et al. 在 KUAVO humanoid 上) 用 Centroidal Dynamics Model 配合 MPC 实现 continuous jumping。

AMP (Adversarial Motion Priors) 的核心：
$$r_{style}(s, s') = -\log(1 - D(s, s'))$$

其中 $D$ 是 discriminator，从 motion capture data 学习 motion style reward。

#### Fall Protection 和 Recovery

- **UKEMI**: 控制 falling posture 分布 impact forces
- **HiFAR**: multi-stage curriculum learning for fall recovery
- **HoST**: Unitree G1 上 robust standing-up across diverse environments
- **Embrace Collisions**: 扩展 robot 的 contact 交互能力，模仿 human 的 roll-and-stand、side-lying

### 3.3 Robotic Manipulation

#### Unimanual Manipulation

**Gripper-based manipulation** 演进：
1. Early: precise physical models + pre-programming + Visual Servoing
2. Learning-based perception: PoseCNN (instance-level 6D pose)、NOCS (category-level)、AffordanceNet、Where2Act
3. Imitation learning: Neural Descriptor Fields、Diffusion Policy、RT-2

Diffusion Policy 的核心 forward process：
$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ 是 cumulative product of noise schedule。Reverse process 学习 denoise action sequence。

**Dexterous hand manipulation**:
- **Two-stage methods**: 先生成 grasp pose，再控制 hand 实现
  - UGG: diffusion model unify pose + object geometry generation
  - SpringGrasp: model uncertainty in partial observations
- **End-to-end methods**:
  - RL: DexVIP、GRAFF、DextrAH-G、DextrAH-RGB
  - Imitation Learning: DexCap、SparseDFF、Neural Attention Field
  - **DexGraspVLA**: vision-language-action + diffusion action controller，1287 unseen object/lighting/background combinations 上 90.8% zero-shot success rate

#### Bimanual Manipulation

- **BUDS**: decompose into stabilizer + executor roles
- **SIMPLe**: Graph Gaussian Processes 表示 bimanual motion primitives
- **ALOHA 系列**: low-cost hardware，efficient large-scale demonstration data
- **ACT (Action Chunking with Transformer)**: action chunking + Conditional VAE (CVAE)

ACT 的 CVAE objective：
$$\mathcal{L} = \mathbb{E}_{q_\phi(z|a, o)} [\log p_\theta(a|z, o)] - D_{KL}(q_\phi(z|a, o) || p(z))$$

10 minutes 的 demonstration data 就能 train 有效 policy。

- **RDT-1B**: diffusion DiT architecture 的 bimanual manipulation foundation model，across heterogeneous multi-robot systems unified action representation

#### Whole-Body Manipulation Control

- **TidyBot**: LLM 学习 personalized household tidying preferences
- **MOO**: VLMs 将 object description 从 language 映射到 visual observations
- **HARMON**: human motion generation priors + VLM editing
- **OKAMI**: object-aware redirection from single RGB-D video
- **OmniH2O**: reinforcement learning Sim-to-Real + VR teleoperation universal kinematic interface
- **HumanPlus**: Transformer-based low-level control + visual imitation，仅需 monocular RGB camera 学习 whole-body skills
- **WB-VIMA**: autoregressive action denoising 建模 hierarchical structure

#### Foundation Models 在 Humanoid Manipulation 中

两种范式：
1. **Hierarchical**: FM 作为 high-level planner + 低层 expert policy
   - Helix (Figure AI): dexterous manipulation 协同
   - GR00T N1 (NVIDIA): general humanoid foundation model
   - π₀: vision-language + flow matching for universal control
2. **End-to-End VLA**: 直接从 multimodal inputs 到 action outputs
   - RT series (Google DeepMind)

### 3.4 Human-Robot Interaction

三个维度：
1. **Cognitive Collaboration**: bidirectional cognitive alignment
   - multimodal intention learning
   - 语义理解 + dynamic context analysis
   - L3mvn、Sg-Nav、Trihelper、CogNav、UniGoal 利用 LLM 模拟 human cognitive states
2. **Physical Reliability**: 力、时间、距离的 coordination
   - PRM、RRT、CHOMP、STOMP、TrajOpt 等 motion planning
   - Impedance control、admittance control for safe physical contact
   - HandoverSim、GenH2R、MobileH2R 利用 simulation 训练 handover policy
3. **Social Embeddedness**: 
   - Peripersonal space 理解
   - 行为理解（gesture、gaze、emotional expressions）
   - Cross-cultural adaptation

---

## 4. General Physical Simulators

### 主流 Simulator 演进时间线

| Simulator | Year | Physics Engine | Key Feature |
|-----------|------|----------------|--------------|
| Webots | 1998 | ODE | 教育导向，2018 open-source |
| Gazebo | 2002 | DART (default) | ROS 集成，modular plugin |
| MuJoCo | 2012 | MuJoCo | convex optimization contact |
| PyBullet | 2017 | Bullet | Python wrapper，lightweight |
| CoppeliaSim | 2010 | Bullet/ODE/Vortex/Newton | distributed control architecture |
| Isaac Gym | 2021 | PhysX + FleX (GPU) | GPU-accelerated parallel training |
| Isaac Sim | 2021+ | PhysX 5 + Omniverse RTX | ray tracing + USD standard |
| Isaac Lab | 2024+ | PhysX (GPU) | modular RL framework on Isaac Sim |
| SAPIEN | 2020 | PhysX | part-level interactive objects |
| Genesis | 2024 | Custom | unified physics solvers + generative data engine |
| Newton | 2025 | NVIDIA Warp | differentiable + 70× speedup |

### MuJoCo 的核心技术

MuJoCo 的核心 innovation 是把 contact constraints 表述为 convex optimization problem：

$$\min_v \frac{1}{2} v^T M v + v^T c$$

subject to:
$$A v + b \geq 0$$ (contact constraints)

其中：
- $v$ 是 generalized velocity
- $M$ 是 inertia matrix
- $c$ 是 bias forces (Coriolis、centrifugal、gravity)
- $A$、$b$ 是 contact Jacobian 和 offset

这个 formulation 保证了 numerical stability 和 computational efficiency，即使在大 time step 下也 maintain accuracy。

### Isaac 系列的 GPU 加速

Isaac Gym 的核心：在 single GPU 上 parallel simulate thousands of environments。PhysX GPU implementation 用 warp-based parallel solver：

1. 每个 thread 处理一个 constraint
2. Block-level reduction 用于 Jacobian assembly
3. Iterative solver (Gauss-Seidel 或 Projected Gauss-Seidel)

Isaac Sim 集成 Omniverse RTX renderer，提供 millimeter-level LiDAR precision。

Isaac Lab 的 tiled rendering 技术：multi-camera input 处理 throughput 提升 1.2×。

### Genesis 的 Generative Data Engine

Genesis 集成多种 physics solvers:
- Rigid body dynamics
- **MPM (Material Point Method)**: 适合 large deformation、fluid、granular materials
- **SPH (Smoothed Particle Hydrodynamics)**: fluid simulation
- **FEM (Finite Element Method)**: soft body
- **PBD (Position-Based Dynamics)**: real-time cloth、rope

Genesis 的 benchmark: 在 512 到 32768 environments 上 throughput 比 Isaac Gym 提升 2.70× 到 11.79×。

### Newton (2025, NVIDIA + DeepMind + Disney Research)

Newton 基于 NVIDIA Warp framework:
- 70× simulation speedup via GPU acceleration
- Differentiable physics engine，支持 backpropagation
- OpenUSD-based scene construction
- 与 MuJoCo Playground、Isaac Lab 兼容

### 物理特性对比 (Table 2)

paper 详细对比了 suction、random external forces、deformable objects、soft-body contacts、fluid mechanism、DEM simulation、differentiable physics 这 7 个维度。

关键 insight：
- **Differentiable physics** 是 frontier：MuJoCo XLA (JAX)、PyBullet (Tiny Differentiable Simulator)、Genesis (MPM solver 已实现，rigid body 在 roadmap) 都在朝这个方向走
- **Fluid mechanism** 仍是大缺口：只有 Webots、Gazebo（basic）、Isaac Sim（particle-based）、Genesis（high-fidelity）支持

### Rendering Capabilities (Table 3)

| 维度 | 关键差异 |
|------|---------|
| Rendering Engine | OpenGL (legacy) → Vulkan (modern) → Omniverse RTX (photoreal) |
| Ray Tracing | 仅 Isaac Sim/Lab、SAPIEN、Genesis (LuisaRender) 支持 |
| PBR | Webots WREN、Gazebo Ogre、Isaac Sim/Lab、SAPIEN、Genesis 支持 |
| Parallel Rendering | Isaac 系列、SAPIEN (ManiSkill3, 30000+ FPS)、Genesis 优化 |

### Sensor 和 Joint Support (Table 4)

- 大部分主流 simulator 都支持 RGB Camera、IMU、Force contact
- LiDAR: 仅 Isaac Sim/Lab、Gazebo、Webots、CoppeliaSim、PyBullet、Genesis 支持
- Helical joint: 仅 Gazebo 和 CoppeliaSim 原生支持

---

## 5. World Models

NVIDIA 对 World Model 的定义："generative AI models that understand the dynamics of the real world, including physics and spatial properties"

### 5.1 Architecture 演进

paper 识别出 5 种主要 architecture：

#### 1. Recurrent State Space Model (RSSM)

Dreamer 系列的核心架构。RSSM 同时 maintain deterministic state 和 stochastic state：

$$h_t = f_\theta(h_{t-1}, s_{t-1}, a_{t-1})$$
$$\tilde{s}_t \sim q_\theta(s_t | h_t, o_t)$$
$$\hat{s}_t \sim p_\theta(s_t | h_t)$$

其中：
- $h_t$ 是 deterministic recurrent state (GRU output)
- $s_t$ 是 stochastic latent state
- $o_t$ 是 observation
- $a_{t-1}$ 是 previous action
- $q_\theta$ 是 posterior (encoder)
- $p_\theta$ 是 prior (transition predictor)

这个 hybrid 设计平衡了 multi-modal future prediction（stochastic 部分）和 long-term memory（deterministic 部分）。

Dreamer 的训练 objective:
$$\mathcal{L} = \mathbb{E}_{q(s_{1:T}|o_{1:T})} \left[ \sum_t \underbrace{\log p(o_t | s_t, h_t)}_{reconstruction} + \underbrace{\log p(r_t | s_t, h_t)}_{reward} + \underbrace{\log p(s_t | h_t)}_{prior} - \underbrace{\log q(s_t | h_t, o_t)}_{posterior} \right]$$

DreamerV1 在 DMC 上表现 strong，DreamerV2 用 discrete latent variables 在 Atari 达到 human-level，DreamerV3 加 normalization 机制，single configuration 跨 150+ 任务 SOTA。

参考: https://danijar.com/project/dreamerv3/

#### 2. Joint-Embedding Predictive Architecture (JEPA)

LeCun 提出，核心 idea：在 abstract latent space 预测 representations，不 reconstruct 原始 pixels。

I-JEPA 的 energy function:
$$E(x, \hat{x}) = \| \phi(x) - \psi(\hat{x}_{context}) \|^2$$

其中：
- $\phi(\cdot)$ 是 target encoder
- $\psi(\cdot)$ 是 context encoder + predictor
- $x$ 是 masked image patches
- $\hat{x}_{context}$ 是 visible context patches

V-JEPA 扩展到 spatiotemporal domain，V-JEPA 2 是 1.2B 参数 model，在 1M+ video hours 上预训练，再用 62 hours robot data fine-tune，在 novel environments 上 65%-80% 成功率。

V-JEPA 2 的训练:
1. **Stage 1**: actionless pretraining on 1M+ video hours → learn physical intuition
2. **Stage 2**: action-conditioned fine-tuning with minimal robot data

参考: https://ai.meta.com/blog/v-jepa-2-world-model-background-knowledge-enables-robotic-planning/

#### 3. Transformer-based State Space Models

Trans-Dreamer、TWM、Genie 用 Transformer 替换 RNN，捕获 long-range dependencies。

Genie (DeepMind) 是 interactive environment foundation model，用 autoregressive video generation + latent action model 实现 controllable environment generation。

参考: https://sites.google.com/view/genie-2024/

#### 4. Autoregressive Generative World Models

类似 LLM 的 next-token prediction，把 video 当作 token sequence。

GAIA-1 是 representative work：9B parameter transformer，在 4700 hours proprietary driving data 上训练。

Video tokenization: VQ-VAE 把每帧 video encode 成 discrete tokens:
$$z = \arg\min_k \| x - e_k \|^2$$

其中 $\{e_k\}_{k=1}^{K}$ 是 codebook entries。然后 transformer autoregressive predict next tokens。

**Limitation**: discrete token quantization 导致 high-frequency detail 丢失，影响 visual quality。

#### 5. Diffusion-based Generative World Models

Recent dominant architecture。从 noise $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ 出发，iterative denoise:
$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$$

其中：
- $\alpha_t$ 是 noise schedule
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$
- $\epsilon_\theta$ 是 noise prediction network
- $\sigma_t$ 是 stochastic noise variance

Latent diffusion (VDM、Imagen Video、VideoLDM、SVD、Sora、Veo3) 把 diffusion 移到 latent space，降低 computational cost。

DriveDreamer、Vista、GAIA-2 是 autonomous driving 中的 representative diffusion-based world models。

**Emerging trend**: Vid2World、Epona 探索 diffusion + autoregressive hybrid，结合 visual expressiveness 和 temporal modeling。

### 5.2 World Models 的三种核心角色

#### Role 1: Neural Simulator

World model 作为 controllable、high-fidelity synthetic data generator，取代传统 simulator。

**Cosmos 系列** (NVIDIA):
- Cosmos foundation video model: 统一 platform 用于 general-purpose world simulator
- Cosmos-Transfer1: spatially conditioned multimodal video generator
  - Adaptive fusion control
  - Structured inputs: segmentation maps、depth、edges
  - 用途：sim-to-real transfer、data augmentation、robot perception

**GAIA-1**: 9B transformer，4700 hours driving data，emergent behaviors 包括 reasoning about agent interactions。

**GAIA-2**: 从 autoregressive 转向 diffusion，structured conditioning inputs (vehicle states、road layout、scene semantics)，high-resolution multi-camera consistent video。

**3D-structured neural simulators**:
- **DriveWorld**: city-scale traffic simulator with causal agent interactions
- **DOME**: diffusion-based，predicts 3D occupancy frames，long-horizon
- **AETHER**: geometry-aware framework，preserve scene geometric consistency
- **DeepVerse**: 4D autoregressive video generation

**Robotics domain**:
- **WHALE**: behavior-conditioning + retracing-rollout for OOD generalization
- **Whale-X**: 414M parameters
- **RoboDreamer**: compositional world model，language compositionality generalize 到 unseen object-action combinations
- **DreMa**: Gaussian Splatting + physics simulation，one-shot policy learning on Franka robot
- **DreamGen**: 4-stage pipeline for generalizable robot policies，zero-shot generalization with minimal real-world data
- **EnerVerse**: autoregressive video diffusion + Free Anchor Views (FAVs) for 3D world modeling + 4D Gaussian Splatting
- **WorldEval**: Policy2Vec + latent action conditioning，scalable policy ranking
- **Huawei Pangu World Model**: high-fidelity digital environments (camera videos + LiDAR point clouds)
- **RoboTransfer**: geometry-consistent video diffusion for sim-to-real policy transfer

#### Role 2: Dynamic Model

用于 model-based reinforcement learning (MBRL)。Agent 学习 dynamic model $f: (s_t, a_t) \rightarrow s_{t+1}$ 和 reward model $r: (s_t, a_t) \rightarrow r_t$，用来 simulate interactions。

**PlaNet** (Hafner et al., 2018):
- Latent dynamic model for pixel-based planning
- Combination of deterministic 和 stochastic transitions
- Latent overshooting for multi-step prediction

**Plan2Explore** (Sekar et al., 2020):
- Self-supervised RL agent
- Uses model-based planning for active novelty-seeking
- Zero/few-shot adaptation to unseen tasks

**Dreamer 系列** 详细演进:
- Dreamer (2020): RSSM + actor-critic on DMC
- DreamerV2 (2021): discrete latent variables，Atari human-level
- DreamerV3 (2023): normalization + training stabilization，150+ diverse tasks SOTA with single config
- DayDreamer (2023): Dreamer-style model on physical robots

**ContextWM** (Wu et al., 2023):
- Pretraining on natural videos
- Context modulation mechanism selectively attends to predictable spatial-temporal regions
- Sample-efficient fine-tuning in downstream robotics

**iVideoGPT** (Wu et al., 2024):
- VQ-VAE tokenize videos、actions、rewards
- Transformer autoregressive predict future tokens
- Bypasses explicit state construction，sample full video rollouts

**Recent dynamics models for articulated robots**:
- **SWIM**: affordance-space world model，trained on human videos，<30min adaptation
- **DWL**: end-to-end RL for humanoid locomotion，zero-shot sim-to-real
- **Surfer**: decoupling action 和 scene prediction，54.74% on SeaWave benchmark
- **GAS**: surgical robotic manipulation，69% success on unseen objects
- **Puppeteer**: hierarchical world model，56-DoF humanoid，8 tasks without reward engineering
- **TWIST**: teacher-student world model distillation
- **PIVOT-R**: primitive-driven waypoint-aware world model，19.45% improvement on SeaWave
- **HarmonyDream**: task harmonization，10%-69% performance gains，new Atari 100K record
- **SafeDreamer**: Lagrangian + world model planning for safe RL
- **WMP**: world model-based perception for legged locomotion
- **RWM**: dual-autoregressive mechanisms for long-horizon dynamics
- **RWM-O**: offline + explicit epistemic uncertainty estimation
- **SSWM**: state-space world models，10× faster WM training，4× MBRL speedup
- **WMR**: end-to-end world model reconstruction for blind humanoid locomotion，3.2km hikes on ice/snow
- **PIN-WM**: physics-informed WM for non-prehensile manipulation，Gaussian Splatting + differentiable simulation
- **LUMOS**: language-conditioned imitation learning with world models，zero-shot real robot transfer
- **OSVI-WM**: one-shot visual imitation learning
- **FOCUS**: object-centric world model
- **FLIP**: flow-centric model-based planning
- **EnerVerse-AC**: action-conditional WM，multi-level action-conditioning + ray map encoding
- **FlowDreamer**: RGB-D world model with explicit 3D scene flow，7-11% improvement
- **HWM**: lightweight video-based WM for humanoid robotics，33%-53% size reduction
- **MoDem-V2**: real-world contact-rich manipulation，first successful direct real-world training of vision-based MBRL
- **V-JEPA 2**: 1.2B parameters，two-stage training (1M+ video hours pretrain + 62 hours robot fine-tune)

#### Role 3: Reward Model

利用 world model 的 prediction likelihood 作为 implicit reward signal。Key insight: 如果 agent 的 behavior 产生的 trajectory 容易被 model predict，那么 likely aligns with expert demonstrations。

**VIPER** (Escontrela et al., 2023):
- 训练 autoregressive video world model on expert demonstrations
- 在线 agent behavior 的 reward = model 的 prediction likelihood
- 在 DMC、Atari、RLBench 上 strong performance
- 支持 cross-embodiment generalization

Reward formulation:
$$r_t = \log p_\theta(o_t | o_{<t}, a_{<t})$$

高 prediction likelihood = 与 expert demonstrations 分布一致 = high reward。

**PlaNet 的 Reward Predictor**: explicitly learned reward predictor 从 latent state 预测 reward，minimize error between predicted 和 true reward。

参考: https://arxiv.org/abs/2305.14343

---

## 6. World Models for Intelligent Agents

### 6.1 Autonomous Driving

paper 识别出三个核心角色，对应 Fig. 21。

#### 6.1.1 WMs as Neural Simulators for Autonomous Driving

详细列举了 20+ representative works:

**GAIA-1** (Hu et al., 2023):
- First to treat WM as sequence prediction in autonomous driving
- Multi-modal inputs: video、text、action
- Emergent behaviors: predicting diverse futures from same context
- Controllable generation via action conditioning 和 text prompts (weather、time of day)

**GAIA-2** (Russell et al., 2025):
- Transition from autoregressive to diffusion
- Structured conditioning: ego-vehicle dynamics、multi-agent interactions、environmental factors
- External latent embeddings from proprietary driving models
- High-resolution multi-camera consistent video across UK、US、Germany

**DriveDreamer** series:
- DriveDreamer: two-stage training，first learn structured traffic constraints，then anticipate future states
- DriveDreamer-2: LLM interface converts text queries → agent trajectories → HDMap → driving videos (Unified Multi-View Model UniMVM)
- DriveDreamer4D: world model priors synthesize novel trajectory videos with explicit spatiotemporal consistency control

**MagicDrive** series:
- MagicDrive: camera poses、road maps、3D bounding boxes、textual descriptions + cross-view attention
- MagicDrive3D: "generation-first, reconstruction-later" pipeline with Deformable Gaussian Splatting
- MagicDrive-V2: DiT architecture + 3D VAE，MVDiT block，848×1600 resolution，241 frames

**Panorama和多视角**:
- **WoVoGen**: explicit 4D world volumes，intra-world consistency + inter-sensor coherence
- **Panacea**: panoramic video with multi-view consistency + super-resolution

**Occupancy-based**:
- **Occ-Sora**: 4D occupancy generation，16-second videos
- **Drive-World**: Memory State-Space Models (MSSM) with Dynamic Memory Bank + Static Scene Propagation
- **Drive-OccWorld**: occupancy forecasting + end-to-end planning，memory module accumulating semantic/dynamic info
- **GaussianWorld**: 4D occupancy forecasting via Gaussian representations
- **DFIT-OccWorld**: decoupled dynamic flow + image-assisted training
- **OccLLama**: occupancy-language-action generative WM
- **OccWorld** (Zheng et al.): vector-quantized VAE learn discrete scene tokens from 3D occupancy，GPT-like spatial-temporal modeling

**Long-horizon 和 reconstruction**:
- **InfinityDrive**: indefinitely long driving sequences
- **ReconDreamer**: online restoration + progressive data update，first to render multi-lane shifts up to 6 meters

**Other notable**:
- **ADriver-I**: general WM
- **DrivingWorld**: video GPT
- **DualDiff+**: dual-branch diffusion with reward guidance
- **GeoDrive**: 3D geometry-informed driving WM
- **Cosmos-Transfer1**: adaptive fusion control
- **BEVWorld**: BEV latent space
- **HoloDrive**: holistic 2D-3D multi-modal street scene
- **GEM**: generalizable ego-vision multimodal WM
- **DriveArena**: closed-loop generative simulation platform
- **ACT-Bench**: action controllable WM benchmark
- **Epona**: autoregressive diffusion WM
- **DrivePhysica**: physics-informed driving WM
- **Cosmos-Drive**: scalable synthetic driving data generation
- **RenderWorld**: self-supervised 3D label WM

#### 6.1.2 WMs as Dynamic Models

- **MILE**: model-based imitation learning for urban driving，joint learning of predictive WM 和 driving policy
- **TrafficBots**: multi-agent traffic simulation with configurable personalities via CVAE
- **UniWorld**: 4D geometric occupancy prediction as pre-training task
- **OccWorld** (Zheng et al.): VQ-VAE discrete scene tokens + GPT-like modeling
- **GaussianWorld**: 4D occupancy forecasting with Gaussian representations
- **DFIT-OccWorld**: decoupled dynamic flow + image-assisted training
- **MUVO**: spatial voxel representations for camera + LiDAR fusion
- **ViDAR**: visual point cloud forecasting as pre-training
- **LAW**: self-supervised learning without perception labels
- **Think2Drive**: efficient RL in latent space，expert-level in complex urban scenarios
- **HERMES**: unified 3D scene understanding + generation，BEV + world queries with causal attention
- **Cosmos-Reason1**: physical common sense + embodied reasoning
- **Doe-1**: next-token generation with multi-modal tokens (observation、description、action)
- **DrivingGPT**: driving world modeling + trajectory planning，unified "driving language"
- **CarFormer**: self-driving with learned object-centric representations
- **Copilot4D**: unsupervised WM via discrete diffusion
- **UnO**: unsupervised occupancy fields for perception + forecasting
- **NeMo**: neural volumetric WM
- **Imagine-2-Drive**: high-fidelity WM in CARLA

#### 6.1.3 WMs as Reward Models

- **Vista**: generalizable reward functions via simulation capability，self-assessment without manual reward engineering
- **WoTE**: BEV world models for real-time safety assessment，SOTA on NAVSIM 和 Bench2Drive
- **Drive-WM**: multi-future trajectory exploration with image-based reward evaluation
- **Iso-Dream**: controllable vs non-controllable dynamics separation，better long-horizon planning

#### 6.1.4 技术趋势

paper 总结出 4 个 trend:

1. **Generative Architecture 演进**: autoregressive → diffusion → hybrid (DiT)
2. **Multi-modal Integration**: text、image、LiDAR、trajectory、HDMap 多模态 input，controllable scenario generation
3. **3D Spatial-Temporal Understanding**: 从 RGB 到 4D occupancy grids，Gaussian Splatting 增强 geometric fidelity
4. **End-to-End Integration**: world model 不再是 standalone tool，而是 autonomous driving pipeline 的一部分

### 6.2 World Models for Articulated Robots

详细 comparison 见 Table 6 和 Table 7。这里列举 paper 提到的所有 works:

#### Neural Simulators for Articulated Robots

- **WHALE** (2024): behavior-conditioning + retracing-rollout
- **RoboDreamer** (2024): compositional WM，factorizing video generation into primitives
- **DreMa** (2024): Gaussian Splatting + physics simulation，one-shot policy learning
- **DreamGen** (2025): 4-stage pipeline，zero-shot generalization with minimal real data
- **EnerVerse** (2025): autoregressive video diffusion + FAVs + 4D Gaussian Splatting
- **WorldEval** (2025): Policy2Vec + latent action conditioning
- **Cosmos** (NVIDIA, 2025): unified platform for foundation video models
- **Pangu** (2025): high-fidelity digital environments
- **RoboTransfer** (2025): geometry-consistent video diffusion for sim-to-real
- **TesserAct** (2025): 4D embodied world models
- **3DPEWM** (2025): 3D persistent embodied WMs
- **SGImageNav** (2025): imaginative WM with scene graphs
- **EmbodieDreamer** (2025): real2sim2real transfer via embodied WM

#### Dynamics Models for Articulated Robots

- **PlaNet** (2018): RSSM latent dynamic model for pixel-based planning
- **Plan2Explore** (2020): self-supervised RL，model-based planning + novelty seeking
- **Dreamer 系列** (2020-2023): RSSM-based，DMC + Atari + real robots
- **DayDreamer** (2024): Dreamer on physical robots
- **Dreaming 系列**: likelihood-free InfoMax contrastive objective
- **DreamerPro** (2022): prototypical representations for visual distraction robustness
- **TransDreamer** (2024): Transformer-based SSM
- **LEXA** (2021): unified unsupervised goal-reaching
- **FOWM** (2023): offline pretraining + online finetuning，epistemic uncertainty regularization
- **SWIM** (2023): affordance-space WM from human videos
- **ContextWM** (2023): pretraining on natural videos
- **iVideoGPT** (2023): autoregressive Transformer
- **DWL** (2024): humanoid locomotion，zero-shot sim-to-real
- **Surfer** (2024): progressive reasoning，SeaWave 54.74%
- **GAS** (2024): surgical robotic manipulation，69% on unseen
- **Puppeteer** (2024): hierarchical WM，56-DoF humanoid
- **TWIST** (2024): teacher-student WM distillation
- **PIVOT-R** (2024): primitive-driven waypoint-aware WM，28× efficiency
- **HarmonyDream** (2024): task harmonization，new Atari 100K record
- **SafeDreamer** (2024): Lagrangian + WM planning for safe RL
- **WMP** (2024): world model-based perception for legged locomotion
- **RWM** (2025): dual-autoregressive mechanisms
- **RWM-O** (2025): offline + epistemic uncertainty estimation
- **SSWM** (2025): state-space WMs，10× faster training
- **WMR** (2025): end-to-end WM reconstruction for blind humanoid，3.2km hikes
- **PIN-WM** (2025): physics-informed WM，Gaussian Splatting + differentiable simulation
- **LUMOS** (2025): language-conditioned imitation learning + WMs
- **OSVI-WM** (2025): one-shot visual imitation
- **FOCUS** (2025): object-centric WM
- **FLIP** (2025): flow-centric model-based planning
- **EnerVerse-AC** (2025): action-conditional WM
- **FlowDreamer** (2025): RGB-D WM with explicit 3D scene flow，7-11% improvement
- **HWM** (2025): lightweight video-based WM for humanoid，33%-53% size reduction
- **MoDem-V2** (2024): first direct real-world training of vision-based MBRL
- **V-JEPA 2** (2025): 1.2B parameters，two-stage training
- **MoSim** (2025): neural motion simulator
- **DALI** (2025): dynamics-aligned latent imagination
- **AdaWorld** (2025): adaptable WMs with latent actions

#### Reward Models for Articulated Robots

- **VIPER** (2023): video prediction models as rewards，cross-embodiment generalization
- **GWM** (2025): Gaussian WMs for robotic manipulation
- **PlaNet** (2018): explicit reward predictor

#### 技术趋势

paper 识别出 4 个方向:

1. **Tactile-Enhanced WMs for Dexterous Manipulation**: high-resolution contact modeling + visuo-tactile fusion
2. **Unified WMs for Cross-Hardware and Cross-Task Generalization**: hardware-agnostic dynamics encoding，object-centric representations，sim-to-real bridges via residual physics
3. **Hierarchical WMs for Long-Horizon Tasks**: goal-conditioned latent spaces，memory-augmented transformers，self-supervised skill discovery
4. **Compositional Generalization**: disentangled abstract representations of entities、relations、physical properties

---

## 6.3 Challenges 和 Future Perspectives

paper 列出 9 个核心 challenge:

1. **High-Dimensionality 和 Partial Observability**: camera、LiDAR、radar 等 high-dim input；partial observability 需要信念状态维护
2. **Causal Reasoning vs Correlation Learning**: 当前 WM 学习 correlations 而非 causal relationships，无法 counterfactual reasoning
3. **Abstract 和 Semantic Understanding**: 需要超越 pixel prediction，理解 traffic laws、pedestrian intent、object affordances
4. **Systematic Evaluation 和 Benchmarking**: MSE on future predictions 不足以衡量 downstream task performance
5. **Memory Architecture 和 Long-Term Dependencies**: compounding prediction errors、stochastic nature of real world
6. **Human Interaction 和 Predictability**: agent behavior 需要 legible、predictable、socially compliant
7. **Interpretability 和 Verifiability**: deep learning-based WMs 是 black box，safety-critical applications 需要 audit 和 formal verification
8. **Compositional Generalization 和 Abstraction**: disentangled、abstract representations of entities、relations、physical properties
9. **Data Curation 和 Bias**: "long tail" of rare but safety-critical events

---

## 我的 Intuition 和 Insights

### 关于 Physical Simulator 的 evolution

从 paper 可以看出 simulator 的演进轨迹：从**精确 physics-based**（Webots、Gazebo、MuJoCo）到 **GPU-accelerated**（Isaac Gym、Isaac Sim）再到 **differentiable**（MuJoCo XLA、Newton、Genesis）。Differentiable physics 是关键 breakthrough，因为它 enables end-to-end optimization，gradient 可以通过 simulation backpropagate 到 policy。

但 simulator 的核心 limitation 在于：**永远存在 sim-to-real gap**。原因包括：
1. **Contact dynamics** 简化：real world 的 friction、compliance、wear 难以精确建模
2. **Sensor noise** 不完全建模
3. **Distribution shift**：训练 distribution 之外的 scenario

这就是 world model 作为 implicit learned representation 的价值——它从 data 中直接学习 dynamics，避免了 explicit modeling 的偏差。

### 关于 World Model 的三角色

paper 最 valuable 的 contribution 是清晰区分了 WM 的三种角色：

1. **Neural Simulator**: WM 作为 **data generator**。这个 role 上 WM 相比 traditional simulator 的优势是 (a) 可以 generate OOD scenarios，(b) 可以利用 internet-scale video data pretrain，(c) 生成 controllable content。

2. **Dynamic Model**: WM 作为 **planning engine**。这是 model-based RL 的核心。WM 学到 environment 的 dynamics，agent 在 WM 内做 imagination rollout，optimize policy。

3. **Reward Model**: WM 作为 **reward inference**。当 reward function 难以设计时，expert demonstration 训练的 WM 可以提供 implicit reward。Key insight 是 prediction likelihood 等价于 trajectory 与 expert distribution 的 alignment。

这三个角色互相强化：好的 neural simulator 可以 generate data 训练 dynamic model；好的 dynamic model 可以做 planning 和 reward inference。

### 关于架构选择

paper 清晰展示了 architecture evolution:
- **RSSM** (2018-2023): 同时 maintain deterministic + stochastic state，适合 multi-modal future prediction 和 long-term memory
- **JEPA** (2023+): 在 latent space predict，不 reconstruct pixels，更 efficient 和 sample-efficient，适合 large-scale pretraining
- **Transformer-based**: long-range dependencies，但 quadratic complexity
- **Autoregressive**: 类似 LLM，但 discrete token quantization 丢失 high-freq detail
- **Diffusion-based**: 高 fidelity，但 sampling speed 是 bottleneck
- **Hybrid** (emerging): diffusion + autoregressive 结合

V-JEPA 2 的成功值得关注——1.2B parameters，先在 1M+ video hours 上 actionless pretrain，再用 62 hours robot data fine-tune，65%-80% success rate on novel environments。这印证了 LeCun 的 thesis：通过 predictive learning 在 latent space 学习 world dynamics，然后用少量 labeled data fine-tune，可以大大降低 robot data 需求。

### 关于 Autonomous Driving 和 Robotics 的 convergence

paper 在 section 6 开头指出："Autonomous driving vehicles can be seen as an intelligent robot with four wheels together with a smaller action space comparing to humanoid robots"。这是一个重要的 conceptual insight。

Tesla 也在 research 中共享 visual encoding architecture for autonomous driving 和 robots。这暗示了 embodied intelligence 的 unified framework：world model 学到的 visual dynamics、physical reasoning 是 cross-domain transferable 的。

但 autonomous driving 和 articulated robots 也有显著差异：
- **Action space**: driving 是低维 (steering、acceleration、brake)，robotics 是高维 (joint torques、end-effector poses)
- **Horizon**: driving 需要秒级 lookahead，robotics 可能需要分钟级
- **Safety criticality**: driving 是 open road，robotics 是 controlled environment
- **Multi-modal observation**: driving 是 multi-camera + LiDAR，robotics 是 multi-modal + tactile

### 关于 Sim-to-Real Gap

paper 提到 several bridges:
1. **Domain randomization**: vary physics parameters、visual appearance
2. **Sim-to-Real via residual learning**: learn residual physics on top of simulator
3. **Digital cousins** (PIN-WM): physics-aware digital cousins bridge sim-to-real
4. **Teacher-student distillation** (TWIST): state-privileged teacher supervise image-based student
5. **Online restoration** (ReconDreamer): world model-based component mitigates artifacts

### 关于 Future Direction

paper 的 9 个 challenges 中，我认为最关键的是:

1. **Causal Reasoning**: 当前 WM 学 correlations，无法做 counterfactual。这限制了 OOD generalization。Causal discovery methods 需要集成到 WM 训练。

2. **Compositional Generalization**: humans 可以从 "cup" + "table" 立即理解 "cup on table"，但 current WMs 需要大量 compositional examples。Disentangled representations 和 structured latent spaces 是 promising direction。

3. **Systematic Benchmarking**: MSE on future prediction 与 downstream task performance 脱节。需要新的 metrics 衡量 WM 的 planning utility 和 safety relevance。

4. **Long-horizon Memory**: compounding errors + stochastic world 让 long-term prediction 困难。State-space models (Mamba、RWKV) 可能提供 efficient long-range modeling。

### 关于一些没有充分讨论的方向

paper 没有充分覆盖的几个 frontier:

1. **Tactile world models**: paper 在 future direction 提到，但 current work 很少。Tactile sensing 对于 fine-grained dexterous manipulation 至关重要。

2. **Active perception**: robot 应该主动 control sensors 探索 environment，减少 partial observability。这与 WM 的 planning 能力紧密相关。

3. **Multi-agent world models**: multi-robot collaboration、human-robot interaction 需要 model other agents 的 behavior。

4. **Self-supervised embodied learning**: 类似 V-JEPA 2 的 paradigm，如何大规模利用 internet videos、Ego4D 等数据 pretrain WM，然后 transfer 到 robot。

5. **Interpretability of world models**: deep WMs 是 black box。理解它们的 internal representations 和 failure modes 对于 safety-critical applications 必要。

---

## 总结

这篇 survey 是 embodied intelligence 领域的重要 reference，systematically 梳理了 physical simulators 和 world models 两大 enabler。它的核心 contribution 在于：

1. **IR-L0 到 IR-L4 分级标准**: 提供 robot intelligence 的 unified evaluation framework
2. **Simulator 对比**: 详细分析 10+ 主流 simulator 在 physical、rendering、sensor 维度的能力
3. **World Model 三角色**: 系统识别 neural simulator、dynamic model、reward model 三种 role
4. **Architecture evolution**: 从 RSSM 到 JEPA 到 diffusion 的清晰演进路径
5. **Application coverage**: 全面覆盖 autonomous driving 和 articulated robots 的 recent works

paper 的 limitation:
- 没有深入讨论 quantitative comparison between simulators 和 WMs
- 没有充分讨论 computational efficiency trade-offs
- Causal reasoning、interpretability 等 future direction 仅 conceptual sketch

作为 Karpathy 你可能最感兴趣的方向:
1. **V-JEPA 2** 的 self-supervised pretraining paradigm 是 LeCun 思路的 representative work
2. **Cosmos** 作为 foundation WM platform 的 scalability
3. **Differentiable physics** (Newton、Genesis) 在 sim-to-real bridge 上的潜力
4. **Diffusion Policy** 在 manipulation 中的成功，可以延伸到 locomotion 和 whole-body control
5. **VLA models** (RT-2、OpenVLA、π₀) 的 end-to-end integration 与 WM 的结合

参考链接：
- Paper GitHub: https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey
- DreamerV3: https://danijar.com/project/dreamerv3/
- V-JEPA 2: https://ai.meta.com/blog/v-jepa-2-world-model-background-knowledge-enables-robotic-planning/
- NVIDIA Cosmos: https://www.nvidia.com/en-us/glossary/world-models/
- RT-2: https://robotics-transformer2.github.io/
- Sora technical report: https://openai.com/research/video-generation-models-as-world-simulators
- GAIA-1: https://arxiv.org/abs/2309.17080
- MuJoCo: https://mujoco.org/
- Isaac Sim: https://developer.nvidia.com/isaac/sim
- Genesis: https://github.com/Genesis-Embodied-AI/Genesis
- NVIDIA Newton: https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/
- LeCun JEPA paper: https://openreview.net/pdf?id=BZ5a1r-kVsf
- VIPER: https://arxiv.org/abs/2305.14343
- Dreamer series: https://dreamerv3.github.io/
- ALOHA: https://tonyzhaozh.github.io/aloha/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- OpenVLA: https://openvla.github.io/
- π₀: https://arxiv.org/abs/2410.24164
