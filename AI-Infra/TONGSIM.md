---
source_pdf: TONGSIM.pdf
paper_sha256: bddfbd0fa51c5b135e6d65799570836535cb41be6b9f7f2301ab3ee79ef59a39
processed_at: '2026-08-12T16:35:48-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# TongSIM 人话版

好，咱们抛开学术腔，用 Karpathy 式的 first principles 思维重新捋一遍这 paper。

---

## 这帮人到底想干嘛

BIGAI (Song-Chun Zhu 那拨人) 的核心 belief：**AGI 必须在 physical + social environment 里 train 出来，光靠 text 不行**。他们之前搞了 Tong Test (一个 AGI 评估 framework，https://doi.org/10.1016/j.eng.2024.06.008)，TongSIM 本质就是 Tong Test 的 **infrastructure layer** — 你得先有个 world 让 agent 住进去，才能测它有没有 general intelligence。

现在 embodied AI 仿真圈的现状很割裂：
- Habitat (https://aihabitat.org/) 跑得快，thousands FPS，但 scene 是 scan 来的，interaction 弱
- Isaac Sim / OmniGibson (https://behavior.stanford.edu/omnigibson) 物理超精确，但慢，compute cost 高
- VirtualHome (http://virtualhome.org/) 走 symbolic program 路线，fidelity 低
- GRUtopia (https://github.com/OpenRobotLab/GRUtopia) 想做 city-scale 但比较新

没人能同时干好 low-level navigation + mid-level household task + high-level multi-agent social simulation。TongSIM 想当一个 **universal substrate**，用 UE5 当 rendering base，通过 plug-in 方式接 MuJoCo / Isaac Lab 做精确物理，覆盖整个 task spectrum。

---

## 架构直觉

最核心的设计 decision：**rendering 和 control 解耦**。

```
UE5 Server (rendering + scene + light physics)
    ↕ gRPC / TCP
Python Controller (agent logic + task definition + eval)
    ↕ stream
Web / VR Client (human observer / teleoperation)
```

这个设计为什么 clever？因为 visual data 是 high-bandwidth（RGB 1080p @ 60fps），control command 是 low-bandwidth（"move to (x,y,θ)" 就几个 float）。如果把这俩塞进同一个 pipe，bottleneck 永远在 visual transmission。分开之后，Python SDK 可以轻量地 control，render stream 走另一条路给 human 或 vision model 看。

跟 Isaac Lab 的 client-server 设计思路一致，跟 ROS2 的 topic 分发也类似。Habitat 3.0 用的是 batched vectorized rendering，路线不同 — Habitat 牺牲了 fidelity 换 throughput，TongSIM 牺牲 throughput 换 fidelity。

---

## Scene 这块怎么搞的

115 个 indoor scene，方法是 **expert seed + procedural expansion + human filter**，三步走：

**Step 1**: 专业设计师手工 curate 一批 seed scene，保证 semantic realism（沙发不会出现在厨房中间，床不会挡门）。

**Step 2**: 自动 expansion pipeline：
- **Coarse phase**: 把 seed scene 拆成 functional units (bedroom, kitchen, study)，以 door frame 作为 alignment anchor 随机重组。door frame 这个 anchor 选择很聪明——所有 room 都必须有门，用门对齐就天然保证拓扑合理，不会出现 "厨房通向厕所但没有走廊" 这种 weird layout。
- **Fine phase**: 对每个 unit 内部做 micro-perturbation，object pose 加 noise，同 category asset swap（把这张 sofa 换成 library 里另一张 sofa）。

**Step 3**: 设计师 human-in-the-loop filter 掉不合理的 sample。

跟 ProcTHOR (https://procthor.allenai.org/) 的区别：ProcTHOR 完全自动 generate，scale 大但偶尔出 weird scene。TongSIM 牺牲一些 scale 换 fidelity，最后那道 human filter 保证 quality。

Outdoor 是个 **holistic metropolis** — 不是 isolated fragments 拼起来，是 contiguous 的城市。教育区、住宅区、商业区、医疗区、road network、traffic 全部连在一起。这个 contiguous 设计的关键意义：**long-horizon context preservation**。agent 从咖啡馆走到对面书店，中间这段路上的 sensory observation、social interaction、scene transition 都是连续的，agent 能学到 "我从 A 到 B 路上环境怎么渐变" 的 mental map。isolated fragments 训练不出这种 long-horizon spatial reasoning。

---

## Physics 这块的 trade-off

UE5 自带 Chaos engine，能做 rigid body / fluid / cloth / destruction，但它是 **game-grade** 不是 research-grade。对 household task 够用，对 robot manipulation precision 不够。

他们的解法：**pluggable physics backend**。

**Route 1**: 把 UE5 物理换成 MuJoCo (https://mujoco.org/)。所有 scene entity 的 physics delegate 给 MuJoCo，state update sync 回 UE5 渲染。MuJoCo 在 manipulation 研究里是 gold standard，contact dynamics 精确，constraint solver 快。

**Route 2**: 直接迁移到 Isaac Lab (https://github.com/isaac-sim/IsaacLab)，物理 + 渲染全用 NVIDIA Isaac Sim，保留 TongSIM 的 task architecture 和 agent interface。

这俩 route 反映一个 honest 取舍：与其自己造个高保真物理引擎（成本高、生态小），不如 plug in 现有 best-in-class。让 TongSIM 专注于 task definition + scene diversity + benchmark design，把 physics 当可替换 component。这种 layer 分离可能成为 next-gen embodied AI simulator 的标准范式。

还在 integrate NVIDIA Flex — 用 unified particle representation 表示所有 material，让 fluid 和 rigid body seamless interact。pour liquid / mop floor / spill water 这类 task 需要 fluid simulation，rigid-body-only simulator 做不了。

---

## Interactive Object 设计里最 clever 的点

**双层 control mechanism**：

每个 electronic device 有两个独立 state：
- **Powered state**: 物理电源连接状态（插没插电）
- **Activation state**: 逻辑开关状态（按钮按没按）

设备 operational = powered AND activated。

这意味着 microwave 没插电，即使你按了开关也不会工作。这种 decomposition 在 real-world robotics 里 critical — robot 需要诊断 "为什么这设备不工作"，是没电还是开关没开？这种 causal structure 在 single-state simulator 里学不到，agent 学不到 "check power first, then check switch" 这种 diagnostic reasoning。

**Spatial Interaction Anchors** + **Placement Points**：

每个 interactable object 预定义 semantic anchor 点（handle, button, spout），end-effector 通过 anchor 精确定位。Placement Points 是预定义的 spatial docking slot，自动 calibrate relative pose — 比如 cup 必须 align 到 water dispenser 出水口下方。

这个设计把 manipulation 问题降维成 "navigate to anchor + trigger action"，大大降低 MLLM agent 的 action space complexity。对 multi-object synergy（cup + dispenser, plate + fork + table）特别友好。

---

## Procedural Animation + Text-Driven Motion

**Procedural animation** 用 Control Rig + IK real-time 合成 gait，适应不同 skeletal structure。好处是不需要 animation asset，换 avatar 不用重做 animation。代价是 motion 自然度比 hand-crafted 差。TongSIM 把它做成 unified locomotion layer，跟 navmesh / physics / perception 深度集成。

**Text-Driven Motion Generation** 是最 ambitious 的部分。架构：

```
Text prompt + target position + voxel map
    ↓
Intent parser (natural language → executable constraint)
    ↓
Diffusion model (motion synthesis)
    ↓
Bi-directional gRPC streaming (generate-while-play)
    ↓
Local navigation + root motion fusion
```

Voxel input 怎么构造：围绕 character 当前位置 + trajectory 前两个 waypoint 采样 voxel grid，并行采样 geometric occupancy，生成 bitmap byte stream 跟 request 一起传给 diffusion model。

**Controllable root motion fusion** 把 motion 拆成两层：
- Vertical motion（起伏、footstep timing）— server 端控制 rhythm
- Horizontal displacement + steering — local navigation module 实时规划

这样 motion 既有 diffusion 的 diversity 又有 goal-directed 的 guarantee。跟 MDM (Motion Diffusion Model, https://guytevet.github.io/mdm-page/) 思路类似，但 voxel-aware 在 embodied AI 场景里很 rare。

---

## Crowd Simulation: Hierarchical SFM + VLM

**Low level**: Social Force Model (SFM, Helbing 1995, https://arxiv.org/abs/cond-mat/9805244)

核心公式：
$$f_{ij} = A \exp\left(-\frac{r_{ij} - d_{ij}}{B}\right) \mathbf{n}_{ij}$$

- $f_{ij}$: agent i 受到 j 的 social repulsion force
- $A$: interaction strength（force 强度）
- $B$: interaction range（衰减距离）
- $r_{ij} = r_i + r_j$: 两 agent 半径之和
- $d_{ij}$: 实际距离
- $\mathbf{n}_{ij}$: 从 j 指向 i 的单位向量

距离近的时候 force 大，距离远 force 衰减。加 A* feasible region sampling 做 path planning。

**High level**: VLM 模拟 human agent 的 semantic behavior。SFM 只能产生 emergent crowd dynamics（瓶颈处自然形成 lane、panic 时 turbulence），但 NPC 不会 "去买咖啡"——它只会避障。加 VLM 后 NPC 有 task-relevant behavior。

这种 hybrid 既保 SFM 的 emergent dynamics，又加 VLM 的 semantic controllability。Virtual Community 限制 15-25 并发，SimWorld 用 discrete waypoint。TongSIM 宣称稳定 100+ pedestrian，因为 SFM 是 O(N²) pairwise force，N=100 时还行。

---

## Parallel Training

UE5 一个 instance 内 load 多个 mutually independent sub-level，agent 同时从多个 environment 采数据。实验在 i9-13900KF + RTX 4090 上跑，near-linear scaling 直到 saturation，之后 inter-process communication overhead 占主导。

跟 Habitat 3.0 的 vectorized rendering 比 scale 不一样——Habitat 能跑 thousands FPS，TongSIM 在 UE5 上是 process-level parallel，受 OS scheduling 限制。但 TongSIM 每个 environment 是 full-fidelity UE5 渲染，Habitat 是 simplified mesh + fast rendering。Trade-off 还是 speed vs fidelity。

---

## Benchmarks 人话解读

### Single-Agent Navigation: paper ball cleanup

Agent 在 multi-room cluttered environment 里捡 scattered paper balls。完全 randomize target 数量、分布、agent initial pose。预计算 traversable free space 保证 task 可解（target 不会 spawn 在墙里）。

Baseline 是 PPO + 19×19 occupancy grid (208cm × 208cm physical area)。

**Metric**:

Success Rate:
$$\mathrm{SR} = \frac{N_{\mathrm{success}}}{N_{\mathrm{total}}} \times 100\%$$

- $N_{\mathrm{total}}$: 评估总 episode 数
- $N_{\mathrm{success}}$: 成功 episode 数

Efficiency:
$$\mathrm{Efficiency} = \frac{1}{N_{\mathrm{success}}} \sum_{i=1}^{N_{\mathrm{success}}} \left(\frac{T_{\max} - S_i}{T_{\max}}\right)$$

- $T_{\max}$: 每 episode 最大允许步数
- $S_i$: 第 i 个成功 episode 实际步数
- 步数越少 efficiency 越接近 1

**结果**: PPO SR=0.6, Efficiency=0.34；Human SR=1.0, Efficiency=0.54

PPO 在 local obstacle avoidance 基本可以，但跨 room navigation 困难——这反映 PPO + local grid 缺 global spatial memory。跨 room 需要 cognitive map，long-horizon credit assignment 是 PPO 弱项。可以考虑加 recurrent state 或 explicit memory module（NeRF-based memory, semantic map）。

### Multi-Agent Cooperative Search (MACS)

Post-flood search scenario，partial observability + 动态 hazards + static obstacles。5 个 agent 协作收集 supplies（需要 2 人协作才能拿），同时避开动态 hazards。

Action space: continuous 2D Box(-1, 1)。Observation: 30 个 radial ray-casting sensor，返回 surrounding entity 的 relative distance / orientation / velocity。

Reward design 有意思：local_reward ratio = 0.9，90% reward 来自 individual，10% 来自 team。既鼓励 individual agency 又保 team incentive。

**Metric**:
$$\bar{R} = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} r_t^{(i)}$$

- $N$: agent 数量
- $T$: episode horizon
- $r_t^{(i)}$: agent i 在 t 时刻 reward
- Team total reward normalize by agent 数量让不同 N 可比较

**结果**: MAPPO 19.24 > IPPO 14.75 > Random -6.51

MAPPO 的 CTDE (centralized training decentralized execution) 在 cooperative setting 下确实有优势。Centralized critic 能学 joint state value，协调多 agent 一起 collect supply（n_coop=2，单个 agent 触碰 reward 只有 0.01，真正协作才 +10）。

### Robot Social Navigation: human-robot hybrid

Robot (Unitree Go2) 在 30 个 SFM-driven pedestrian 的 dynamic crowd 里导航到 target。Sensor: RGB-D + 3D LiDAR + GPS。ROS2 接口。

**Metric**:

Efficiency:
$$\mathrm{EFF} = 1 - \frac{T_{\mathrm{actual}} - T_{\min}}{T_{\max} - T_{\min}}$$

- $T_{\mathrm{actual}}$: 实际完成时间
- $T_{\min}$: 理论最短时间（无障碍直行）
- $T_{\max}$: 最大允许时间

Social Norm Compliance (SNC): 不侵入 personal space 的时间比例
- Type-1 intrusion: $d < 0.45$ m（intimate space）
- Type-2 intrusion: $0.45 \leq d \leq 1.2$ m（personal space）

Total:
$$\mathrm{Total} = 100 \times (0.2 \cdot \mathrm{EFF} + 0.2 \cdot \mathrm{SRT} + 0.3 \cdot \mathrm{SAF} + 0.3 \cdot \mathrm{SNC})$$

权重设计：SAF 和 SNC 各 0.3 比 EFF 和 SRT (各 0.2) 高，benchmark 优先考虑 safety + sociality 而非 pure efficiency。

**结果**: Human 92.7 >> MPPI 43.1 >> DWA 10.4

DWA 在 dynamic crowd 下 SR=0.1，因为它假设 obstacle 静态。MPPI sampling-based 好一些但还是远低于 human。核心结论：**pure geometric planning 不够，需要 social cognition**。这跟 CrowdNav、SocialGAN、DSARL 等 social navigation 文献一致——需要 modeling human intent。

### Household Composite Tasks: 测 MLLM 的真实能力

8 类 task 3 大 domain：
- **Object Understanding**: counting, gift selection
- **Spatial Intelligence**: building blocks, jigsaw puzzle, understanding buttons
- **Social Activity**: setting tables, tidying rooms, preparing baggage

MLLM 通过 prompt 封装成 embodied agent，perception-reasoning-action loop。不 retrain，不加 external module，纯 prompt engineering。直接测 model intrinsic capability。

**17 个 MLLM 结果**：

| Model | Mean |
|-------|------|
| Gemini-2.5-Pro | 24.53 |
| GPT-5 | 21.54 |
| Claude-3.7-Sonnet | 20.52 |
| Llama-4-Maverick | 14.48 |
| Llama-3.2 | 2.81 |

**核心 insight**：
1. **Object Understanding 最强**：GPT-5 在 Gift Selection 拿 69.06。MLLM perception + classification 基本 OK。
2. **Spatial Intelligence 最弱**：Building blocks 普遍 0-12 分，jigsaw puzzle 5-8 分。Spatial reasoning + manipulation planning 严重不足。
3. **Social Activity 中等偏弱**：最高 20-30 分。

这组数据揭示 critical gap：当前 MLLM 的 **spatial intelligence 是 weakest link**。Building blocks 要求 3D mental simulation（这个 block 放上去会不会倒）、multi-step physical reasoning（先放 base 再叠哪些）、fine-grained manipulation planning（抓哪里放哪里）。这些能力 text training data 里几乎学不到，需要 embodied training 或 synthetic spatial data。

### S³IT: Spatially Situated Social Intelligence Test

最 advanced 的 benchmark。Task：给 room layout + 一组 NPC（各有 identity, preference, interpersonal relationship），agent 要安排座位让所有人满意。

Setup: 5 个 dynamic layout scene，59 个 NPC，复杂 family + social 关系网，7,000 个 problem，每个 problem 有 3-Likert intensity weighting（preference 强弱）。

**T-Agent 3-phase pipeline**:
1. **Phase I**: 跟每个 NPC 对话构建 preference profile
2. **Phase II**: 探索 3D room 构建结构化表示
3. **Phase III**: integrate 信息，iterative refine seating plan

**Prioritization Gap (PG)**:
$$\mathrm{PG} = S_{\mathrm{high}} - S_{\mathrm{low}}$$

- $S_{\mathrm{high}}$: high-weighted preference 满足率
- $S_{\mathrm{low}}$: low-weighted preference 满足率
- PG > 0 说明 agent 能区分 priority

**结果**：

| Model | Embodied | Social | Conflict | PG | Average |
|-------|----------|--------|----------|-----|---------|
| Gemini-2.5-pro | 40.6 | 56.2 | 85.7 | 8.8 | 47.8 |
| GPT-5 | 29.0 | 56.9 | 86.1 | 15.4 | 42.7 |
| Claude-4.5 | 19.1 | 37.6 | 46.0 | 4.8 | 23.1 |

**关键发现**：所有 model **Embodied 维度最低**，**Conflict 维度最高**。Conflict resolution 容易（text training 里有大量 corpus），但 spatial reasoning + social preference integration 是真正 bottleneck。GPT-5 的 PG=15.4 最高，adaptive reasoning architecture 在 priority weighting 上最强。

核心 thesis 验证：**spatial intelligence 是 embodied social reasoning 的 cornerstone**。没有 spatial reasoning，social reasoning 只能在 text 层面 simulate，没法 ground 到 physical environment。

---

## 跟其他 simulator 横向对比

| Feature | TongSIM | GRUtopia | OmniGibson | Habitat 3.0 | VirtualHome | Virtual Community |
|---------|---------|----------|------------|-------------|-------------|-------------------|
| Engine | UE5.6 | Isaac Sim | Isaac Sim | Bullet | Unity3D | Genesis |
| Scene | 115 | 100 | 50 | 211 | 6 | 35 urban |
| Indoor | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| Outdoor | ✓ | ✓ | ✓ | × | × | ✓ |
| City-level interaction | ✓ | ✓ | × | × | × | ✓ |
| Parallel training | ✓ | ✓ | ✓ | ✓ | × | × |
| Task-oriented fidelity | ✓ | ✓ | × | ✓ | × | ✓ |
| Sim-to-Real support | ✓ | ✓ | ✓ | ✓ | × | ✓ |
| Multi-agent | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Human-robot teaming | ✓ | ✓ | × | ✓ | × | ✓ |

TongSIM 的 sweet spot：**broad task coverage + medium fidelity + medium speed**。Habitat 3.0 throughput 完胜，Isaac Sim/OmniGibson 物理 fidelity 更高。TongSIM 走的是 coverage 路线。

---

## 我的 intuition & 联想

### 1. 设计哲学根源
TongSIM 体现 BIGAI 团队长期的 AGI 四元论：**perception + cognition + action + social interaction**。这跟 Karpathy 你自己强调的 "AI 要 ground 到 physical world" 思路一致。TongSIM 的 benchmark suite 就是按这四维度设计的。

### 2. Benchmark 揭示的本质问题
- **Spatial reasoning 是 MLLM bottleneck**：Building blocks 5-12 分，jigsaw 5-8 分。这不是 model scale 问题，是 training data + architecture 问题。Text-only training 学不到 3D physical reasoning，需要 embodied training + spatial inductive bias。
- **Long-horizon planning 误差累积**：PPO 跨 room navigation 失败，MLLM composite task 失败。Error accumulates exponentially。
- **Social cognition 需要 model-based 而非 reactive**：A*+DWA 在 crowd 里 SR=0.1，human SR=1.0。Pure geometric planner 不可行，需要 intent modeling。

### 3. World Model 方向的关联
最近 World Model (DreamerV3, Genie 2, Genesis) 兴起，核心思想是 latent dynamics model 替代 explicit physics simulation。TongSIM 走 explicit physics + rendering 路线，是 "传统 simulator" 阵营。Paper 也提到 "Embodied World Models is becoming increasingly vital"。

如果 TongSIM 5.0 加 latent world model pretraining（用 TongSIM 生成大量 trajectory 训 world model），然后 world model 做 planning，这会是 killer feature。**"simulator + learned world model" hybrid** 可能是 embodied AI 下一阶段的 dominant paradigm。

### 4. 跟 Symmetrical Reality 的关联
Zhang et al. 的 Symmetrical Reality (https://arxiv.org/abs/2403.17019) 提出 physical-virtual 双向 interaction paradigm。TongSIM 的 VR Client + Audio2Face + NPC 双向 dialogue 是这个 vision 的 partial implementation。未来 embodied AI 训练很可能走这条路——human-in-the-loop 跨 reality 数据收集。

### 5. 对 Eureka Labs 的 relevance
TongSIM 在 embodied education（agent 通过 perception-reasoning-action loop 学复杂 task）的 design philosophy，跟 Eureka Labs 的 "AI tutor + embodied learning" 方向有 conceptual overlap。S³IT 那种 social intelligence test 也可以作为 next-gen AI education agent 的 evaluation framework。

### 6. Missing pieces
- MuJoCo / Isaac Lab integration 的具体 API surface 没详述
- Procedural generation 的 quantitative diversity metric 没给（跟 ProcTHOR 的 scene diversity analysis 比起来缺）
- Sim-to-real 的实际 robot experiment 还没做
- Crowd simulation 的 SFM 参数（A, B 系数）没披露

---

## Reference Links

**TongSIM & BIGAI**:
- TongSIM GitHub: https://github.com/bigai-ai/tongsim
- Tong Test paper: https://doi.org/10.1016/j.eng.2024.06.008
- Symmetrical Reality: https://arxiv.org/abs/2403.17019

**Simulator 对比**:
- Habitat 3.0: https://aihabitat.org/
- OmniGibson / BEHAVIOR-1K: https://behavior.stanford.edu/omnigibson
- GRUtopia: https://github.com/OpenRobotLab/GRUtopia
- SAPIEN: https://sapien.ucsd.edu/
- iGibson: http://svl.stanford.edu/igibson/
- VirtualHome: http://virtualhome.org/
- AI2-THOR: https://ai2thor.allenai.org/
- ProcTHOR: https://procthor.allenai.org/
- Virtual Community: https://arxiv.org/abs/2508.14893

**Physics engine**:
- MuJoCo: https://mujoco.org/
- NVIDIA Isaac Sim: https://developer.nvidia.com/isaac-sim
- Isaac Lab: https://github.com/isaac-sim/IsaacLab
- Genesis: https://genesis-embodied-ai.github.io/

**Algorithm & method**:
- PPO: https://arxiv.org/abs/1707.06347
- MAPPO: https://arxiv.org/abs/2103.19752
- IPPO: https://arxiv.org/abs/2011.09533
- DWA: https://ieeexplore.ieee.org/document/580373
- MPPI: https://arxiv.org/abs/1509.01149
- A* (Hart, Nilsson, Raphael 1968): https://ieeexplore.ieee.org/document/4082128
- Social Force Model: https://arxiv.org/abs/cond-mat/9805244
- MDM (Motion Diffusion Model): https://guytevet.github.io/mdm-page/

**Benchmark**:
- ALFRED: https://askforalfred.com/
- R2R (Vision-Language Navigation): https://aimarkup.org/

---

## 一句话总结

TongSIM 是 BIGAI 团队用 UE5 搭的 universal embodied AI platform，核心 contribution 是 **scope** — 把 single-agent navigation → multi-agent coop → human-robot social → household composite → spatial social intelligence 五层 task 整合到统一 platform。Paper 最大的 scientific finding 是 **MLLM spatial reasoning bottleneck**：SOTA MLLM 在 building blocks/jigsaw 等 spatial task 平均 5-10 分 (满分 100)，远低于 object understanding 的 60+ 分。这揭示 text-only training 学不到 3D physical reasoning，验证了 embodied training + spatial inductive bias 的必要性。Pluggable physics backend (MuJoCo / Isaac Lab) 的 layer 分离设计可能成为 next-gen embodied AI simulator 的标准范式。

---

# TongSIM: 通用 Embodied AI 仿真平台深度解析

## 1. Motivation: 这篇 paper 想解决什么问题

Embodied AI 当前最核心的痛点是**仿真平台的碎片化**。你看 Habitat 系列擅长 fast RL training 但场景是 scan-based 的、interaction 弱; OmniGibson / Isaac Sim 物理保真度极高但 cost 大; VirtualHome 走 program symbolic 路线; GRUtopia 想做 city-scale 但是是 new entrant。BIGAI 这帮人 (Song-Chun Zhu 课题组 + Yujia Peng + Zhenliang Zhang) 想搞一个**统一平台**同时覆盖:
- Low-level: navigation, locomotion, manipulation
- Mid-level: household composite tasks
- High-level: multi-agent social simulation, human-robot collaboration

这种 hierarchical coverage 的野心让我想起他们的 Tong Test (Peng et al., *Engineering* 2024, https://doi.org/10.1016/j.eng.2024.06.008), TongSIM 本质上是 Tong Test 的 implementation substrate。

**核心 thesis**: 单一 simulator 通过 task-adaptive fidelity + scene diversity + 多模态 agent interface, 可以同时训练 low-level RL policy 和 high-level MLLM agent, 而不需要在多个平台之间切换。

---

## 2. 系统架构 (Section 3.1)

架构是经典的 **Client-Server 分离设计**:

```
┌────────────────────────────────────────────────────┐
│  TongSIM UE Server (UE5.6 + Chaos physics)         │
│  ├── Multimodal Sensors (RGB/Depth/Voxel/Seg)       │
│  ├── High-fidelity Simulation (rigid/fluid/cloth)  │
│  ├── Large-scale NPC System (SFM + VLM)            │
│  └── Parallel Training (sub-level instantiation)   │
│           ▲                                         │
│           │ gRPC / TCP                              │
│           ▼                                         │
│  TongSIM Python Controller                          │
│  ├── Scene Manager (metadata, navmesh, spawn)       │
│  ├── Agent API (kinematics, target, interaction)    │
│  └── Benchmark Evaluator                            │
│           ▲                                         │
│           │ stream                                  │
│           ▼                                         │
│  Web Client / VR Client / Audio2Face               │
└────────────────────────────────────────────────────┘
```

关键设计点: **rendering 与 control 解耦**。Python SDK 通过 API 操控 scene/object/NPC/agent, render 结果 stream 到 web。这跟 Isaac Lab / Habitat 3.0 的设计哲学一致——把 high-bandwidth 视觉输出和 low-bandwidth control command 分开传输, 避免一个 bottleneck 卡住整个 pipeline。

**28 个 interaction primitives** 覆盖: pick-and-place, toggle door, sit/stand, pour liquid, mop, wipe, read, cut food, consume, sleep/wash routine... 这种 granularity 介于 VirtualHome 的 program-level 和 SAPIEN 的 part-level 之间, 适合 long-horizon composite task。

---

## 3. Scene 系统: 115 室内 + 连续室外城市

### 3.1 Indoor scenes
115 个场景由两条路线生成:
1. **Expert-designed seeds**: 专业设计师手工 curate layout, 保证 semantic realism (沙发不放在厨房, 床不挡门)
2. **Automated expansion pipeline**: coarse-to-fine

**Coarse phase**: 把已存在 scene 拆成 functional units (bedroom, study, kitchen), 然后以 **door frame 作为 alignment anchor** 随机重组。这里有个 clever 的点——门框是天然 的 alignment primitive, 因为所有 room 都必须有门, 用门做 anchor 就保证拓扑合理性。

**Fine phase**: 对每个 unit 内部做 micro-perturbation
- 物体 pose 加 noise
- 同 category asset 替换 (swap sofa with another sofa from library)

**Human-in-the-loop validation**: 设计师 filter 掉 semantically implausible 的 sample。这是 ProcTHOR (Deitke et al., NeurIPS 2022, https://procthor.allenai.org/) 那种 procedural generation 没做的——ProcTHOR 完全自动, 但 TongSIM 在最后加了一道人工关, 牺牲一些 scale 换取 fidelity。

Style coverage: modern apartment, villa, medieval castle, Japanese garden, classical Chinese architecture。这种 stylistic diversity 对训练 robust vision policy 重要——避免 agent overfit 到某一种 rendering style。

### 3.2 Outdoor: holistic metropolis
**关键设计**: **spatially contiguous**, 不是 isolated fragments。教育区、住宅区、商业区、医疗区、road network、traffic simulation 全部连在一起。这点跟 Virtual Community (Zhou et al., https://arxiv.org/abs/2508.14893) 类似但 TongSIM 强调 contiguous navigation 的 long-horizon context preservation。

对 embodied agent 训练有什么意义?**Contextual continuity**。如果 agent 从咖啡馆出来走到对面书店, 中间这段路径上的 sensory observation、social interaction、scene transition 都是真实连续的, agent 可以学到 "我从 A 到 B 一路上环境怎么变化" 的 mental map。isolated fragments 训练不出这种 long-horizon spatial reasoning。

---

## 4. Agent 系统

TongSIM 的 agent 既可以是 AI-driven embodiment, 也可以是 NPC。NPC 控制是 **hybrid**: rule-based + LLM。

**Action space 三层 granularity**:
1. **Kinematic primitives**: nod, wave, turn — atomic motion
2. **Target-driven**: gaze at coordinate, point-to-point navigation — closed-loop control
3. **Composite activities**: consume item, pour liquid, mop, read — multi-step sequenced

这种分层让我联想到 ROS 的 hierarchy (move_base > local planner > velocity command), 也跟 "LLM-Brain + Controller-Cerebellum" 这个 hybrid architecture 思路吻合。

---

## 5. Platform Features 深度解析

### 5.1 Physics: Chaos + Flex
UE5 自带 **Chaos physics engine** 处理 rigid body / fluid / destruction / cloth。Chaos 是 UE5 时代替换 PhysX 的 engine, 比 PhysX 4 更适合 large-scale destruction simulation。

他们还在 integrate **NVIDIA Flex** — Flex 用 unified particle representation 表示所有 material, 让 fluid 和 rigid body 可以 seamless interact。这对 pour liquid / spill water / mop floor 这类 task 是必须的, 因为 rigid-body-only simulator 做不了 fluid。

**对比**: OmniGibson 直接用 PhysX 5 + Isaac Sim 的 particle system, 是 GPU-native 的。TongSIM 在 UE5 上集成 Flex 是 CPU-based particle, 规模受限但跟 UE5 渲染 pipeline 衔接好。

### 5.2 Interactive Objects: 双层 control + spatial anchors

这是 paper 里我觉得最 clever 的设计:

**Electromechanical Logic (双层 control)**:
- **Powered state**: 物理电源连接状态
- **Activation state**: 逻辑开关状态

设备的最终 operational state = powered AND activated。这意味着 microwave 没插电, 即使你按了开关也不会工作。

这种 decomposition 在 real-world robotics 里 critical——robot 需要先诊断 "为什么这个设备不工作": 是没电, 还是开关没开? 这种 causal structure 在 single-state simulator 里学不到。

**Spatial Interaction Anchors**:
对每个 interactable object 预定义 semantic anchor 点 (handle, button), end-effector 通过 anchor 精确定位。还有 **Placement Points**——例如水杯必须 align 到 water dispenser 出水口下方, 系统自动 calibrate relative pose。

这个设计实际上把 manipulation 问题降维成了 "navigate to anchor + trigger action", 大大降低了 MLLM agent 的 action space complexity。GRUtopia 的 approach 比较类似, 但 TongSIM 显式定义了 placement point 概念, 对 multi-object synergy (cup + dispenser) 更友好。

### 5.3 Procedural Animation (Control Rig + IK)

这是 experimental feature。传统 UE pipeline 是: animator 制作 animation clip → state machine 切换 clip。问题是 skeleton-specific, 换个 avatar 就要重做。

TongSIM 用 **Control Rig + IK** real-time 合成 gait, 适应不同 skeletal structure。Foot planting 通过 opposite foot phase 检测——一只脚支撑时另一只脚摆动, 通过 ground alignment IK 让脚贴地。这种 procedural approach 在 game engine 里不算新 (GTA 系列早就在用), 但 TongSIM 把它做成 **unified locomotion layer**, 跟 navmesh / physics / perception pipeline 深度集成。

好处: 用户不需要改 skeleton asset 就能 deploy agent 到任意 scene。代价: motion 自然度比 hand-crafted animation 差。

### 5.4 Text-Driven Motion Generation (Diffusion-based)

这模块是 paper 里最 ambitious 的部分之一。架构:

```
Text Prompt + Target Position + Voxel Map
                ↓
        Intent Parser (NL → executable constraint)
                ↓
        Diffusion Model (motion synthesis)
                ↓
   Bi-directional gRPC streaming (generate-while-play)
                ↓
        Local Navigation + Root Motion Fusion
```

**Voxel input 怎么构造**:
- 围绕 character 当前位置 + trajectory 前两个 waypoint 采样 voxel grid
- 并行采样 geometric occupancy, 生成 bitmap byte stream
- 跟 request 一起传给 diffusion model

**Controllable root motion fusion**:
- **Vertical motion** (起伏、footstep timing): server 端 control rhythm
- **Horizontal displacement + steering**: local navigation module 实时规划

这种 decomposition 把 "运动风格" (server) 和 "导航目标" (local) 解耦, 让 motion 既有 diffusion 的多样性, 又能保证 goal-directed。

跟 MDLM (masked diffusion motion models) 或者 MDM (Motion Diffusion Model, https://research.nvidia.com/labs/toronto-ai/mdm/) 思路类似, 但 voxel-aware 的 motion generation 在 embodied AI 场景里很 rare。

### 5.5 Large-Scale Crowd Simulation (Hierarchical)

两层架构:

**Low-level (motion control)**:
- **Social Force Model (SFM)**: Helbing 1995 的经典 pedestrian model (https://en.wikipedia.org/wiki/Social_force_model)
  基本公式 (paper 没明写, 但 SFM 的核心):
  $$f_{ij} = A \exp\left(-\frac{r_{ij} - d_{ij}}{B}\right) \mathbf{n}_{ij}$$
  其中:
  - $f_{ij}$: agent i 受到 j 的 social repulsion force
  - $A$: interaction strength
  - $B$: interaction range
  - $r_{ij} = r_i + r_j$: 两 agent 半径之和
  - $d_{ij}$: 实际距离
  - $\mathbf{n}_{ij}$: 从 j 指向 i 的单位向量
  
- **A\*-based feasible region sampling**: 在 free space 里 sample 下一帧位置

**High-level (decision)**:
- **VLM** 模拟 human agent 的 semantic behavior
- 支持 robot agent 协作

这种 hybrid 让 crowd simulation 既有 SFM 的 emergent crowd dynamics (瓶颈处自然形成 lane、panic 时产生 turbulence), 又有 VLM 的 task-relevant behavior (NPC 会去买咖啡而不是随机游走)。

**对比**: Virtual Community 限制 15-25 并发, SimWorld 用 discrete waypoint。TongSIM 宣称稳定支持 100+ pedestrian, 这是 SFM 的优势——O(N²) 的 pairwise force 在 N=100 时还行, GPU 加速后到 N=1000 也可能。

### 5.6 Sim-to-Real: MuJoCo / Isaac Lab Integration

两条路线:

**Route 1: MuJoCo backend**
- UE5 物理引擎换成 MuJoCo (Todorov et al., IROS 2012, https://mujoco.org/)
- Scene entity 的物理 simulation 全部 delegate 给 MuJoCo
- State update 同步回 UE5 渲染

MuJoCo 在 robot manipulation training 上的优势: contact dynamics 精确, constraint solver 快, soft contact model 适合 grasping 研究。UE5 Chaos 对 robotic manipulation 不够精确——它是 game-grade 不是 research-grade。

**Route 2: Isaac Lab native**
- 物理 + 渲染全部迁移到 Isaac Sim
- 保留 TongSIM task architecture 和 agent interface

Isaac Lab (https://github.com/isaac-sim/IsaacLab) 是 NVIDIA 的 robot learning framework, GPU-native physics, 跟 ROS2 兼容好。TongSIM 这条路线相当于把 Isaac Lab 当 backend, 把 TongSIM 当 task definition layer。

**我的 intuition**: 这两条路线反映 TongSIM 团队的 honest 取舍。UE5 优势是 rendering quality 和 asset ecosystem, 但物理保真度不够 robot learning 用。与其重新造一个高保真物理引擎, 不如 plug in 现有 best-in-class 的 MuJoCo / Isaac Lab。这种 layer 分离让 TongSIM 可以同时服务 "social simulation (不需要精确物理)" 和 "robot manipulation (需要精确物理)" 两种用户。

### 5.7 Parallel Training (Sub-level Instantiation)

UE5 一个 instance 内 load 多个 mutually independent sub-level, agent 同时从多个 environment 采数据。

实验配置:
- CPU: Intel Core i9-13900KF (24 cores / 32 threads, 3.0 GHz)
- GPU: NVIDIA GeForce RTX 4090
- Task: spatial exploration & navigation

结果: **near-linear scaling** 直到 saturation, 之后 inter-process communication + scheduling overhead 占主导。

这跟 Habitat 3.0 的 parallel simulation (https://aihabitat.org/) 比较起来 scale 不一样——Habitat 用 batching + scene vectorization 能跑到数千 FPS, TongSIM 在 UE5 上的 parallel 是 process-level, scale 受限于 OS scheduling。

但 TongSIM 的优势是每个 environment 是 full-fidelity UE5 渲染, 而 Habitat 是 simplified mesh + fast rendering。trade-off 还是 speed vs fidelity。

---

## 6. Benchmarks 详细解析

### 6.1 Single-Agent: Spatial Exploration & Navigation

**Task**: 在 multi-room cluttered environment 里收集 scattered paper balls。完全 randomize:
- paper ball 数量
- 空间分布
- agent initial pose

**Feasibility guarantee**: 预计算 traversable free space (考虑 agent collision geometry), 保证 agent 和 target 都 spawn 在 reachable region。这个细节很重要——很多 RL benchmark 失败 case 其实是 task 不可解 (target 在墙里), 不是 agent 笨。

**Observation**: egocentric RGB + depth + voxel grid。Baseline 用 19×19 occupancy grid, 对应 208cm × 208cm physical area。

**Action space**: 移动到 coordinate, 旋转到 orientation (discrete goal-conditioned)

**Metric**:

Success Rate:
$$\mathrm{SR} = \frac{N_{\mathrm{success}}}{N_{\mathrm{total}}} \times 100\%$$

- $N_{\mathrm{total}}$: 评估总 episode 数
- $N_{\mathrm{success}}$: 成功 episode 数 (在 $T_{\max}$ 步内完成)

Efficiency:
$$\mathrm{Efficiency} = \frac{1}{N_{\mathrm{success}}} \sum_{i=1}^{N_{\mathrm{success}}} \left(\frac{T_{\max} - S_i}{T_{\max}}\right)$$

- $T_{\max}$: 每 episode 最大允许步数
- $S_i$: 第 i 个成功 episode 实际用的步数
- 越少步数完成, efficiency 越接近 1

**结果**:

| Agent | Success Rate | Efficiency |
|-------|--------------|-----------|
| PPO | 0.6 | 0.34 |
| Human | 1.0 | 0.54 |

**Failure analysis**:
1. **Obstacle avoidance in clutter**: RL agent 在 narrow / obstacle-dense region 卡住
2. **Long-horizon navigation**: 跨 room traversal 困难

**我的 takeaway**: PPO + 19×19 local grid 训出的 policy 缺乏 global spatial memory。Local observation → action mapping 学得好 (避障基本可以), 但跨 room 需要构建 cognitive map, 这种 long-horizon credit assignment 是 PPO 的弱项。可以考虑加 recurrent state (LSTM/Transformer) 或 explicit memory module (NeRF-based memory, semantic map)。

### 6.2 Multi-Agent Cooperative Search (MACS)

**Scenario**: post-flood search, partial observability, stochastic hazards + static obstacles

**Objectives**:
- Collaboration: 多 agent 协作收集 supplies (要求 multi-agent manipulation)
- Safety: evade 动态 hazards
- Efficient navigation: 仅靠 local sensory 规划 path

**Action space**: continuous 2D, $\text{Box}(-1.0, 1.0)$

**Observation**: 30 个 radial ray-casting sensors, 每个 sensor 返回 surrounding entity 的 relative distance / orientation / velocity

这种 radial sensor 设计让我想到 Autonomous Driving 里的 LiDAR 表示, 也跟 MAgent / Particle MPE 类似, 是 multi-agent RL 经典 observation。

**Default config**:

| Parameter | Value |
|-----------|-------|
| n_agents | 5 |
| n_supplies | 10 |
| n_hazards | 10 |
| n_coop (supplies 需要几人协作) | 2 |
| n_sensors | 30 |
| sensor_range | 500.0 |
| max_cycles | 500 |
| supply_reward | +10.0 |
| hazard_reward | -1.0 |
| encounter_reward | +0.01 |
| thrust_penalty | -0.01 |
| localization (local/global reward ratio) | 0.9 |

**Reward formulation**: 这里有个有意思的设计——`localization = 0.9` 意味着 90% reward 来自 local, 10% 来自 global team performance。这种设计既鼓励 individual agency, 又保持 team incentive。

**Metric**:

Mean episodic return per agent:
$$\bar{R} = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} r_t^{(i)}$$

- $N$: agent 数量
- $T$: episode horizon
- $r_t^{(i)}$: agent i 在 t 时刻 reward
- 总 team reward normalize by agent 数量, 让不同 N 的实验可比较

**Baseline**:
- **IPPO** (Independent PPO, De Witt et al. 2020, https://arxiv.org/abs/2011.09533): 每个 agent 独立 policy, 不知道 joint state
- **MAPPO** (Yu et al., NeurIPS 2022, https://arxiv.org/abs/2103.19752): CTDE paradigm, centralized critic + decentralized actor

**结果**:

| Method | Mean Step Reward | Mean Episodic Return per Agent |
|--------|------------------|-------------------------------|
| MAPPO (CTDE) | 0.0380 | 19.24 |
| IPPO (Independent) | 0.0295 | 14.75 |
| Random | -0.013 | -6.51 |

MAPPO 比 IPPO 高 30%。CTDE 的 centralized critic 在 cooperative setting 下确实有优势, 能学到 joint state value, 协调多个 agent 一起 collect supply (因为 n_coop=2, 单个 agent 触碰 supply reward 只有 0.01, 真正协作收集才有 +10)。

### 6.3 Human-Robot Hybrid: Robot Social Navigation

**Scenario**: 城市街道 + 动态 crowd, robot 导航到 target

**Crowd simulation**: SFM 驱动 30 个 random pedestrian, 100+ 并发稳定

**Robot**: Unitree Go2, ROS2 接口

**Sensor suite**: RGB-D camera + 3D LiDAR + GPS

**Baseline**:
1. Human Teleoperation (keyboard)
2. A* global + DWA local (Fox et al., 1997, https://ieeexplore.ieee.org/document/580373)
3. A* global + MPPI local (Williams et al., 2015, https://arxiv.org/abs/1509.01149)

**Metrics**:

Efficiency:
$$\mathrm{EFF} = 1 - \frac{T_{\mathrm{actual}} - T_{\mathrm{min}}}{T_{\max} - T_{\min}}$$

- $T_{\mathrm{actual}}$: 实际完成时间
- $T_{\min}$: 理论最短时间 (无障碍直行)
- $T_{\max}$: 最大允许时间

Success Rate (SRT): 成功完成 episode 比例

Safety (SAF): discrete scoring
- 0 collision: 1.0
- 1-3 collision: 0.5
- >3 collision: 0.0

Social Norm Compliance (SNC): 不侵入 personal space 的时间比例
- Type-1 intrusion: $d < 0.45$ m (intimate space)
- Type-2 intrusion: $0.45$ m $\leq d \leq 1.2$ m (personal space)

**Total Score**:
$$\mathrm{Total} = 100 \times (0.2 \cdot \mathrm{EFF} + 0.2 \cdot \mathrm{SRT} + 0.3 \cdot \mathrm{SAF} + 0.3 \cdot \mathrm{SNC})$$

权重设计有意思——SAF 和 SNC 各 0.3, 比 EFF 和 SRT (各 0.2) 高, 说明 benchmark 优先考虑 safety + sociality 而非 pure efficiency。

**结果**:

| Baseline | Robot | EFF | SRT | SAF | SNC | Total |
|----------|-------|-----|-----|-----|-----|-------|
| Human Teleoperation | Go2 | 0.89 | 1.0 | 0.95 | 0.88 | 92.7 |
| A* + DWA | Go2 | 0.42 | 0.1 | 0 | 0 | 10.4 |
| A* + MPPI | Go2 | 0.73 | 0.6 | 0.25 | 0.31 | 43.1 |

**关键发现**: 传统 geometric planner (A*+DWA) 在 dynamic crowd 下 success rate 只有 10%, 因为 DWA 假设 obstacle 静态, 在 stochastic 多 agent 场景里 collision 频发。MPPI 通过 sampling-based 优化好一些, 但还是远低于 human (43.1 vs 92.7)。

**核心结论**: pure geometric planning 不够, 需要 social cognition。这点跟 recent social navigation 文献一致——比如 CrowdNav, SocialGAN, DSARL 都证明需要 modeling human intent。

### 6.4 Primary Composite: Household Benchmark

**核心 motivation**: 现有 benchmark (ImageNet, COCO, VQA, MMBench) 是 isolated task, 缺乏 embodied interaction evaluation。TongSIM 把 MLLM 封装成 agent 在 3D household 环境里执行 composite task。

**8 类 task, 3 大 domain**:
- **Object Understanding**: counting, gift selection
- **Spatial Intelligence**: building blocks, jigsaw puzzle, understanding buttons
- **Social Activity**: setting tables, tidying rooms, preparing baggage

**Perception-Reasoning-Action loop**:
```
Perception (multi-view RGB + JSON scene description)
        ↓
Reasoning (ReAct-style prompting, output API calls)
        ↓
Action (MoveToObject, PickUp, ...)
        ↓
   Environment Update
        ↓
   Loop until done / timeout
```

**关键设计**: 不 retrain, 不加 external module, 仅通过 prompt 把 MLLM 封装成 embodied agent。这种 "纯 prompting evaluation" 直接测模型 intrinsic capability, 但也意味着模型得会 tool use / API calling, 这本身是个 skill。

**实验结果 (17 个 MLLM)**:

| Model | Mean |
|-------|------|
| Gemini-2.5-Pro | 24.53 |
| Gemini-2.5-Flash | 23.05 |
| o3 | 22.88 |
| GPT-5 | 21.54 |
| Claude-3.7-Sonnet | 20.52 |
| Claude-4-Sonnet | 20.51 |
| Doubao-1.5-vision-pro | 19.15 |
| ... | ... |
| Llama-3.2 | 2.81 |

**Domain-specific insights**:
1. **Object Understanding 最强**: GPT-5 在 "Gift Selection" 拿 69.06 — MLLM 的 perception + classification 能力 OK
2. **Spatial Intelligence 最弱**: Building blocks 普遍 0-12 分, jigsaw puzzle 5-8 分 — spatial reasoning + manipulation planning 严重不足
3. **Social Activity 中等偏弱**: 不同 model 各有强项, 但最高也才 20-30 分

**我的 intuition**: 这组数据揭示了一个 critical gap — 当前 MLLM 的 spatial intelligence 是 weakest link。Building blocks 任务要求:
- 3D 空间 mental simulation (这个 block 放上去会不会倒)
- Multi-step physical reasoning (先放哪个 base, 再叠哪些)
- Fine-grained manipulation planning (具体抓哪里, 放哪里)

这些能力 text training data 里几乎学不到。需要 embodied training 或 synthetic spatial data。

### 6.5 Advanced Composite: S³IT (Spatially Situated Social Intelligence Test)

**Task**: seat arrangement。给一个 room layout + 一组 NPC (各有 identity, preference, interpersonal relationship), agent 要安排座位让所有人满意。

**Setup**:
- 5 个 dynamic layout scene
- 59 个 NPC, 各有 background
- 复杂 family + social 关系网
- 7,000 个 problem
- 每个 problem: 1 room + 几个 NPC + 特定 preferences + conflicts
- 3-Likert intensity weighting

**T-Agent pipeline (3 phases)**:
1. **Phase I**: NPC Preference Extraction & Summarization — T-Agent 跟每个 NPC 对话, 构建 preference profile
2. **Phase II**: Environmental Cognition — 探索 3D room, 构建结构化表示
3. **Phase III**: Multi-Constraint Decision-Making — integrate 信息, iterative refine seating plan

**Prioritization Gap (PG) metric**:
$$\mathrm{PG} = S_{\mathrm{high}} - S_{\mathrm{low}}$$

- $S_{\mathrm{high}}$: high-weighted preference 满足率
- $S_{\mathrm{low}}$: low-weighted preference 满足率

PG > 0 说明 agent 能区分 priority, 高权重 preference 优先满足。

**结果**:

| Model | Embodied | Social | Conflict | PG | Average |
|-------|----------|--------|----------|-----|---------|
| Gemini-2.5-pro | 40.6 | 56.2 | 85.7 | 8.8 | 47.8 |
| o3 | 32.9 | 53.8 | 89.0 | 12.7 | 43.1 |
| GPT-5 | 29.0 | 56.9 | 86.1 | 15.4 | 42.7 |
| o4-mini | 29.0 | 54.5 | 89.5 | 6.8 | 41.4 |
| GPT-4.1 | 23.3 | 43.2 | 55.4 | 3.8 | 29.3 |
| Doubao-1.5 | 24.6 | 43.0 | 62.5 | 3.8 | 28.3 |
| GPT-4o | 24.6 | 43.0 | 51.7 | 3.3 | 28.2 |
| GPT-4.1-mini | 22.8 | 39.3 | 42.5 | 3.7 | 26.8 |
| Claude-4.5 | 19.1 | 37.6 | 46.0 | 4.8 | 23.1 |
| GPT-4o-mini | 16.7 | 34.7 | 45.8 | 1.6 | 19.3 |

**关键 insight**:
- 所有 model **Embodied 维度最低**, **Conflict 维度最高**
- GPT-5 的 PG=15.4 最高 — adaptive reasoning architecture 在 priority weighting 上最强
- Gemini-2.5-pro 总分 47.8, 唯一 >40

**核心 thesis 验证**: spatial intelligence 是 embodied social reasoning 的 cornerstone。Conflict resolution 容易 (text training 里有大量相关 corpus), 但 spatial reasoning + social preference integration 是 model 真正的 bottleneck。

---

## 7. 与其他 Simulator 对比

| Feature | TongSIM | GRUtopia | OmniGibson | Habitat 3.0 | VirtualHome | Virtual Community |
|---------|---------|----------|------------|-------------|-------------|-------------------|
| Engine | UE5.6 | Isaac Sim | Isaac Sim | Bullet | Unity3D | Genesis |
| Scene | 115 | 100 annotated | 50 | 211 | 6 | 35 urban |
| Indoor | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| Outdoor | ✓ | ✓ | ✓ | × | × | ✓ |
| City-level interaction | ✓ | ✓ | × | × | × | ✓ |
| Parallel training | ✓ | ✓ | ✓ | ✓ | × | × |
| Task-oriented fidelity | ✓ | ✓ | × | ✓ | × | ✓ |
| NPC control | ✓ | ✓ | × | ✓ | ✓ | ✓ |
| Sim-to-Real support | ✓ | ✓ | ✓ | ✓ | × | ✓ |
| Single-agent | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Multi-agent | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Human-robot teaming | ✓ | ✓ | × | ✓ | × | ✓ |

TongSIM 在 task 覆盖广度上是最大优势。但 Habitat 3.0 在 throughput (thousands FPS) 上完胜 TongSIM, Isaac Sim/OmniGibson 在物理 fidelity 上更精确。TongSIM 的 sweet spot 是 **broad task coverage + medium fidelity + medium speed**。

---

## 8. 我的 Intuition & Takeaways

### 8.1 设计哲学
TongSIM 体现 BIGAI 团队 (Song-Chun Zhu 系) 长期倡导的 **AGI = perception + cognition + action + social interaction** 四元论 (Tong Test, https://arxiv.org/abs/2406.07341)。TongSIM 是这一理论 framework 的 implementation:
- **Perception**: multimodal sensor (RGB/Depth/Voxel/Seg)
- **Cognition**: MLLM reasoning (ReAct loop)
- **Action**: 28 interaction primitives + composite activity
- **Social**: crowd SFM + VLM-driven NPC + S³IT

### 8.2 技术取舍
UE5 作为 base engine 是个有趣的 trade-off:
- **优势**: rendering quality > Isaac Sim, asset ecosystem > custom engine, character animation pipeline 成熟
- **劣势**: physics 不如 MuJoCo 精确, parallel throughput < Habitat

他们的解法是 pluggable backend (MuJoCo / Isaac Lab), 把 UE5 当 rendering + scene composition layer。这是 pragmatic 的选择。

### 8.3 Benchmark 揭示的本质问题
1. **Spatial reasoning 是 MLLM bottleneck**: Building blocks 5-12 分, jigsaw 5-8 分 — MLLM 缺 3D mental simulation 能力, 这不是 prompt engineering 能补的, 需要 architectural change 或 embodied training
2. **Long-horizon planning 误差累积**: PPO 在跨 room navigation 失败, MLLM 在 composite task 失败 — error accumulates exponentially
3. **Social cognition 需要 model-based 而非 reactive**: A*+DWA 在 crowd 里 SRT=0.1, human SRT=1.0 — pure geometric planner 不可行, 需要 intent modeling

### 8.4 跟 Symmetrical Reality 的关联
Zhang et al. 的 Symmetrical Reality (VR 2019, https://ieeexplore.ieee.org/document/8794523) 提出 physical-virtual 双向 interaction paradigm。TongSIM 的 VR Client + Audio2Face + NPC 双向 dialogue, 是这个 vision 的 partial implementation。未来 embodied AI 训练很可能走这条路——human-in-the-loop 跨 reality 数据收集。

### 8.5 Missing pieces
- Paper 没有详述 MuJoCo / Isaac Lab integration 的具体 API surface
- Procedural generation pipeline 的 quantitative diversity metric 没给出 (跟 ProcTHOR 那种 scene diversity analysis 比起来)
- Sim-to-real 的实际 robot experiment 还没做 (paper 提到 "experimental development work has been basically completed")
- Crowd simulation 的 SFM 参数没披露 (A, B 系数)

### 8.6 联想: 这跟 World Model 方向的关联
最近 World Model (DreamerV3, Genie, Genie 2, Genesis) 兴起, 核心思想是 latent dynamics model 替代 explicit physics simulation。TongSIM 走 explicit physics + rendering 路线, 是 "传统 simulator" 阵营。但 paper 里也提到 "Embodied World Models is becoming increasingly vital", 说明他们意识到 latent world model 方向的重要性。

如果 TongSIM 5.0 加 latent world model pretraining (用 TongSIM 生成大量 trajectory 训 world model), 然后用 world model 做 planning, 这会是个 killer feature。这种 "simulator + learned world model" hybrid 可能是 embodied AI 下一阶段的 dominant paradigm。

### 8.7 对 Karpathy 的 relevance
作为曾经 Tesla Autopilot 负责人 + 当前 Eureka Labs 创始人, 你应该对 "general-purpose embodied AI platform" 这个 vision 共鸣很强。TongSIM 在 embodied education (训 agent 通过 perception-reasoning-action loop 学复杂 task) 上的 design philosophy, 跟 Eureka Labs 的 "AI tutor + embodied learning" 方向其实有 conceptual overlap。TongSIM 的 benchmark suite (尤其是 S³IT 那种 social intelligence test) 也可以作为 next-gen AI education agent 的 evaluation framework。

---

## 9. Reference Links

- **TongSIM GitHub**: https://github.com/bigai-ai/tongsim
- **Habitat 3.0**: https://aihabitat.org/
- **OmniGibson / BEHAVIOR-1K**: https://behavior.stanford.edu/omnigibson
- **GRUtopia**: https://github.com/OpenRobotLab/GRUtopia
- **SAPIEN**: https://sapien.ucsd.edu/
- **iGibson**: http://svl.stanford.edu/igibson/
- **VirtualHome**: http://virtualhome.org/
- **AI2-THOR**: https://ai2thor.allenai.org/
- **ProcTHOR**: https://procthor.allenai.org/
- **ALFRED**: https://askforalfred.com/
- **R2R (Vision-Language Navigation)**: https://aimarkup.org/
- **MuJoCo**: https://mujoco.org/
- **NVIDIA Isaac Sim**: https://developer.nvidia.com/isaac-sim
- **Isaac Lab**: https://github.com/isaac-sim/IsaacLab
- **Genesis Engine**: https://genesis-embodied-ai.github.io/
- **Tong Test paper (Peng et al.)**: https://doi.org/10.1016/j.eng.2024.06.008
- **Virtual Community paper**: https://arxiv.org/abs/2508.14893
- **Symmetrical Reality (Zhang et al., VR 2024)**: https://arxiv.org/abs/2403.17019
- **MAPPO paper**: https://arxiv.org/abs/2103.19752
- **IPPO (StarCraft Multi-Agent Challenge)**: https://arxiv.org/abs/2011.09533
- **DWA original paper**: https://ieeexplore.ieee.org/document/580373
- **MPPI paper (Williams et al.)**: https://arxiv.org/abs/1509.01149
- **A* algorithm (Hart, Nilsson, Raphael 1968)**: https://ieeexplore.ieee.org/document/4082128
- **Social Force Model (Helbing & Molnár, 1995)**: https://arxiv.org/abs/cond-mat/9805244
- **PPO algorithm (Schulman et al., 2017)**: https://arxiv.org/abs/1707.06347
- **MDM - Human Motion Diffusion Model**: https://guytevet.github.io/mdm-page/
- **Genesis - Generative physics engine**: https://genesis-embodied-ai.github.io/

---

## 10. 总结

TongSIM 是 BIGAI 在 embodied AI simulator 领域的 comprehensive attempt, 核心贡献不是单点技术突破, 而是 **scope**: 把 single-agent navigation → multi-agent coop → human-robot social → household composite → social intelligence test 五层 task 整合到统一平台。

Paper 最大的 scientific contribution 是 benchmark result 揭示的 **MLLM spatial reasoning bottleneck**: 在 building blocks, jigsaw puzzle 等 spatial task 上, SOTA MLLM 平均 5-10 分 (满分 100), 远低于 object understanding task 的 60+ 分。这不是 model scale 问题, 是 **training data + architecture** 问题——text-only training 学不到 3D physical reasoning, 需要 embodied training + spatial inductive bias。

MuJoCo / Isaac Lab 集成是关键战略选择, 让 TongSIM 既保 UE5 渲染优势, 又能接入 robot learning 生态。这种 layer 分离设计可能成为 next-gen embodied AI simulator 的标准范式。

未来值得关注的演进方向: latent world model pretraining on TongSIM trajectories, human-in-the-loop symmetrical reality 训练, scene diversity quantification, social physics model (VLM + SFM + game theory 集成)。
