---
source_pdf: RoboFactory.pdf
paper_sha256: 802941cb245dba969658f88f08619efc933cad095ddcfd3f30ef1d7da62088d6
processed_at: '2026-08-12T00:53:04-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RoboFactory 用人话说

Hey Andrej，好，我换个方式讲，假装咱俩喝咖啡聊天。

---

## 这帮人到底在干什么

一句话：**想让多个 robot arm 一起干活，但不知道怎么自动生成 training data。**

你看，现在 single robot arm 的 data generation 已经有成熟 pipeline 了——用 LLM 帮你 plan，然后 call 一些 predefined motion primitives，trajectory 就出来了。[RoboTwin](https://robotwin.github.io/) 这类工作已经搞定了。

但一旦你从 1 个 arm 变成 2 个、3 个、4 个 arm，事情就崩了。不是变难了一点，是**指数级变难**。

---

## 为什么 multi-agent 这么难

想象你让 4 个 robot arm 完成一个任务："拿一块牛排，用相机拍下来"。

听起来简单，拆开看：
- $a_1$ 抓牛排
- $a_2$ 和 $a_3$ 抬相机
- $a_4$ 按快门

但每个 arm 不能只管自己，会出三种问题：

**第一种：逻辑搞错了**。$a_3$ 去抓相机的镜头，不是相机机身。好家伙，镜头碎了。

**第二种：空间撞了**。$a_2$ 和 $a_3$ 同时往中间抬相机，路径没规划好，两个 arm 直接撞上，hardware 损坏。

**第三种：时间浪费了**。$a_4$ 在那傻等，因为它以为 $a_2$ 会撞到它，但其实 $a_2$ 早走开了。efficiency 极低。

这三个问题——logical、spatial、temporal——就是 multi-agent 的核心 pain point。Single-arm 时代根本不用想这些。

---

## 他们的 solution：加 constraints

核心 idea 特别朴素：**既然 LLM 生成的 trajectory 会出问题，那我给你加规则，不让你犯错。**

他们提了三类 constraints，对应上面三个问题：

### Logical Constraints $\mathcal{C}_l$
告诉你"该干什么、怎么干"。比如抓相机必须抓机身那个点，按快门手指得正对着快门方向。这是 high-level 的 interaction rule。

### Spatial Constraints $\mathcal{C}_s$  
告诉你"哪儿能去、哪儿不能去"。核心实现是把场景搞成 3D voxel grid，每个 voxel 5cm × 5cm × 5cm，然后看谁占了哪个 voxel。如果两个 arm 的 voxel 重叠了，就 violation。类似于把空间离散化做 collision check。

### Temporal Constraints $\mathcal{C}_t$
告诉你"什么时候干、按什么顺序干"。比如 $a_2$ 先把空间用了，$a_4$ 等会儿再用同一块空间，这样就能并行，效率上去了。或者"必须 $a_5$ 开盖之后 $a_4$ 才能放东西"，这是 ordering constraint。

---

## 系统怎么跑的：两个 module 配合

整个 framework 叫 RoboFactory，核心是两个 module 互相 call：

### RoboBrain（大脑）
用 GPT-4o 当 brain。Input 是 global task description + 多视角 RGB images + 上一步干了啥 + 上一步有没有 violation 的 feedback。Output 是下一步每个 agent 的 subgoal + textual constraints。

公式就是：

$$\mathcal{G}^{\mathrm{next}}, \mathcal{C} = \mathcal{F}_{\mathrm{VLM}}(\mathcal{O}, \mathcal{G}_{\mathrm{global}}, \mathcal{G}^{\mathrm{pre}}, f^{\mathrm{pre}})$$

- $\mathcal{G}^{\mathrm{next}}$：下一个 subgoal，比如 $a_1$ 去抓牛排
- $\mathcal{C}$：constraints，比如"$a_2$ 和 $a_3$ 别撞"
- $\mathcal{O}$：RGB observations
- $\mathcal{G}_{\mathrm{global}}$："拍牛排"这个 task
- $\mathcal{G}^{\mathrm{pre}}$：上一轮的 subgoal
- $f^{\mathrm{pre}}$：上轮有没有 violation

然后 RoboBrain 用 visual programming 调 motion primitives 生成 raw trajectory。但这个 trajectory 是 unconstrained 的，可能有问题。

### RoboChecker（质检员）
GPT-4o 生成的 constraints 是文字，比如"$a_2$ 别和别的 agent 撞"。文字怎么 constrain trajectory？得有个 interface。

RoboChecker 干的就是这个：**把文字 constraint 翻译成能和 physical world 交互的 representation**。

具体有四种 validation function：
- `Validate_Direction()`：检查 gripper 方向对不对
- `Validate_Interaction()`：检查 contact point 对不对  
- `Validate_Spatial_Occupancy()`：检查 voxel 有没有 overlap
- `Validate_Scheduling()`：检查 temporal order 对不对

跑一遍，如果 violation，return False + 失败原因，打回 RoboBrain 重新 plan。如果 pass，trajectory 存下来当 training data。

---

## Benchmark：11 个 task

他们搞了个 benchmark，从 1 个 arm 到 4 个 arm，11 个 task：

- 1-agent：Pick Meat、Stack Cube、Strike Cube
- 2-agent：Lift Barrier（一起抬）、Pass Shoe（接力）、Place Food（一个开盖一个放）、Two Robots Stack Cube
- 3-agent：Camera Alignment（对相机）、Three Robots Stack Cube（叠三个）
- 4-agent：Take Photo（拍牛排那个）、Long Pipeline Delivery（四段接力）

每个 task 用 random seed 随机化初始 asset placement，pre-collect 了 150 条 expert demo。Simulator 用的 [ManiSkill3](https://github.com/haosulab/ManiSkill)，robot 是 [Franka Panda](https://franka.de/) 7-DoF arm。

---

## 实验结果：数字说话

### Diffusion Policy 跑这些 task

他们用 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 当 baseline，分别用 50、100、150 条 demo 训练：

| Agent 数 | 50 demo | 100 demo | 150 demo |
|---------|---------|----------|----------|
| 1 | 25% | 47% | **49%** |
| 2 | 13% | 32.5% | 27.5% |
| 3 | 7.5% | 6% | 20.5% |
| 4 | 2.5% | 4% | **10%** |

看到没？从 1 个 arm 到 4 个 arm，success rate 从 49% 掉到 10%。这 cliff 太陡了。**说明现成的 single-agent imitation learning 方法根本撑不起 multi-agent**。

特别是 Long Pipeline Delivery 这个 task，不管多少 demo，全是 0%。因为这个 task 需要长期 temporal dependency，DP 学不到。这暗示当前 VLA / diffusion-based policy 在 long-horizon 上有 fundamental limitation。

### 四种 multi-agent 架构对比

他们还探索了 multi-agent imitation learning 的架构设计。两个维度：
- **View**：Global（看全场）vs Local（看自己的 ego-view）
- **Policy**：Shared（一个 policy 管所有 arm）vs Separate（每个 arm 一个 policy）

| Arch | View | Policy | Lift Barrier | Place Food |
|------|------|--------|--------------|------------|
| 1 | Global | Shared | 49% | 5% |
| 2 | Local | Shared | 4% | 0% |
| 3 | Global | Separate | 26% | 17% |
| 4 | Local | Separate | **58%** | **20%** |

Arch2 直接崩了，为什么？因为 shared policy 从不同 ego-view 学的时候，它得先搞清楚"我现在是哪个 arm"再决定动作。相当于让 model 同时学 agent ID classification + action generation，capacity 被占满了。

Arch4 最好：local view 提供更 fine-grained 的视觉信息，separate policy 让每个 arm 专攻自己的 skill。直觉上特别 make sense。

### Constraint 的 ablation

最有说服力的是这个 ablation——把三类 constraint 逐个关掉看效果：

**Success Rate：**

| Logical | Spatial | Temporal | Take Photo |
|---------|---------|----------|------------|
| ✓ | ✗ | ✗ | 37.1% |
| ✓ | ✓ | ✗ | 53.8% |
| ✓ | ✗ | ✓ | 62.2% |
| ✓ | ✓ | ✓ | **88.2%** |

**Episode Length（越短越好）：**

| Logical | Spatial | Temporal | Take Photo |
|---------|---------|----------|------------|
| ✓ | ✗ | ✗ | 407 |
| ✓ | ✓ | ✗ | 325 |
| ✓ | ✗ | ✓ | 238 |
| ✓ | ✓ | ✓ | **204** |

三个 constraint 全开，success rate 从 37.1% 飙到 88.2%，episode length 从 407 降到 204。**这就是 compositional constraints 的价值**——每个 constraint 解决一类问题，缺了哪个都不行。

---

## 我觉得有意思的几个点

### 1. Neural + Symbolic 的分工
RoboBrain 是 neural（GPT-4o），负责 generalization 和 reasoning；RoboChecker 是 symbolic（hand-crafted validation rules），负责 verification 和 safety。这个分工在 safety-critical domain 可能是范式。纯 neural 有 hallucination，纯 symbolic 没法 generalize。两个结合正好互补。

### 2. Constraints 是 inductive bias
Compositional constraints 本质上是把人类对"安全协作"的 prior knowledge encode 成 verification rules。类似 CNN 把 translation invariance 编进架构，这里把 collaboration prior 编进 data generation pipeline。问题是这些 prior 应该 hand-craft 还是 learn？Paper 现在是 hand-craft，未来应该是 learned verifier。

### 3. Data quality >> Data quantity
Episode length 减半意味着同样训练时间能覆盖 2 倍场景。在 robotics data 极度稀缺的当下，这种 data efficiency 提升比 success rate 数字更重要。高质量的 short trajectory 比 garbage long trajectory 值钱多了。

### 4. LLM-as-orchestrator 范式
这个 paper 验证了一个 pattern：**LLM propose, verifier check, executor act**。LLM 的 reasoning 能力负责 high-level planning，symbolic verifier 负责 reliability，motion primitives 负责 low-level execution。未来可能看到更多这种三段式 robotics system。类似于 OpenAI o1 在 robotics 领域的应用——search-based planning，不是一步到位，而是 propose + verify + iterate。

### 5. Multi-agent 是 VLA 的下一个 frontier
49% → 10% 的 success rate cliff 暴露了 fundamental gap。现在所有 VLA model（[OpenVLA](https://openvla.github.io/)、[pi0](https://www.physicalintelligence.company/blog/pi0)）都是 single-agent。未来需要 architectural innovation：agent communication、shared memory、joint planning。这个 paper 只是揭开了 multi-agent embodied AI 的冰山一角。

### 6. 5cm voxel 是 hacky 但 effective
Spatial constraint 用 5cm × 5cm × 5cm voxel 做 occupancy check，听起来粗糙，但实际 effective。这暗示在 robotics 里，coarse discretization + symbolic check 可能比 dense continuous representation 更实用。类似 [3D Diffusion Policy](https://3d-diffusion-policy.github.io/) 用 point cloud 但其实 coarse voxel 也够用。

---

## Limitation 我觉得他们没说的

1. **RoboChecker 是 hand-crafted rules**，四种 validation function 都是 predefined 的。遇到没见过的 constraint type 就挂了。未来应该用 VLM 自己做 constraint checking，让 verifier 也是 learned 的。

2. **GPT-4o 是 black box**，如果它 hallucinate 一个错误 constraint，整个 pipeline 就崩了。而且你没法 fine-tune GPT-4o，error propagation 控制不住。

3. **Motion primitives 是 predefined 的**，限制了 task diversity。新 task 如果需要 unseen motion，直接死路。

4. **没有 agent communication**，每个 arm 只看自己的 ego-view，没法显式告诉别的 arm "我要往左走"。在 [Dec-POMDP](https://arxiv.org/abs/2103.01955) 理论里这是 sub-optimal 的。Future work 应该加 implicit communication（通过 observation 推断 intent）或 explicit communication channel。

5. **Sim-to-real gap 没验证**，全在 ManiSkill 里跑的。Real world 里 depth estimation 误差 + voxel 离散化误差会放大。

---

## Reference Links

- [RoboFactory Project Page](https://iranqin.github.io/robofactory/)
- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [RoboTwin](https://robotwin.github.io/)
- [ManiSkill3 GitHub](https://github.com/haosulab/ManiSkill)
- [PartNet-Mobility Dataset](https://sapien.ucsd.edu/)
- [Depth Anything V2](https://depth-anything.com/)
- [Code as Policies](https://code-as-policies.github.io/)
- [Code-as-Monitor (CaM)](https://jianqy.github.io/Code-as-Monitor/)
- [OpenVLA](https://openvla.github.io/)
- [pi0 Blog](https://www.physicalintelligence.company/blog/pi0)
- [3D Diffusion Policy](https://3d-diffusion-policy.github.io/)
- [ACT / ALOHA](https://tonyzhaozh.github.io/aloha/)
- [GPT-4o Tech Report](https://arxiv.org/abs/2303.08774)
- [RoboCasa](https://robocasa.github.io/)

---

总结一句：这 paper 用的方法不复杂，就是 LLM plan + symbolic check，但踩中了一个 important problem（multi-agent data generation），并且用 ablation 量化证明了 compositional constraints 的价值。对想搞 multi-agent embodied AI 的人来说是个 solid 的 baseline 和 benchmark。Build intuition 的角度，记住三个 constraint type + RoboBrain/RoboChecker 分工 + 49%→10% success cliff 就够了。

---

# RoboFactory: Embodied Multi-Agent Collaboration with Compositional Constraints

Hey Andrej, 这篇 paper 是 Shanghai AI Lab 团队(Yiran Qin, Li Kang 等)的工作, tackles 一个非常实际且 underexplored 的 problem: 如何为 **multi-embodied agent** 系统**自动生成** safe 且 efficient 的 training data。让我从 intuition 层面 decompose 一下整个 pipeline。

---

## 1. Motivation: 从 single-agent 到 multi-agent 的 scalability 瓶颈

当前 robotics 的数据生成 pipeline 大量 leverage LLMs 的 reasoning capability (e.g., [RoboTwin](https://robotwin.github.io/), [RoboCodeX](https://arxiv.org/abs/2402.16117)) 来 automate single-agent data collection。但当 number of agents 从 1 scaling 到 N 时, decision space complexity 爆炸式增长, 出现三个新的 failure modes:

1. **Logical inconsistency**: agent $a_3$ 去 grab camera lens 而非 camera body, 直接破坏 hardware
2. **Spatial collision**: $a_2$ 与 $a_3$ 在 trajectory 执行时碰撞, catastrophic hardware damage
3. **Temporal inefficiency**: $a_4$ 在 nonexistent collision worry 下 unnecessary waiting, waste 时间

这三个 failure modes 分别对应 **logical / spatial / temporal** 三个 orthogonal dimensions。Single-agent 系统的 simple adaptation 无法 cover 这些。

---

## 2. Compositional Constraints: 三类约束的形式化

### 2.1 Logical Constraints $\mathcal{C}_l$

定义 **permissible actions 和 interaction rules**, 聚焦 high-level logic: interaction objects, contact points, movement directions。Examples:

- **Usage permissions**: only specific tools can process certain materials
- **Contact point restrictions**: agents must grasp objects from designated points
- **Directional consistency**: multiple agents transporting object 时, applied forces 必须保持 aligned

### 2.2 Spatial Constraints $\mathcal{C}_s$

定义 agents 可以 operate 的位置和 physical interactions 的 structure:

- **Geometric boundaries**: no agent may enter 1-meter radius around active machinery
- **Workspace partitioning**: 把 construction site 分成 exclusive zones prevent collisions
- **Task-specific placement**: components 必须在 2cm tolerance 内才算 valid assembly
- **Adaptive behaviors**: dynamic rerouting around obstacles, gripper orientation adjustment for narrow apertures

### 2.3 Temporal Constraints $\mathcal{C}_t$

Regulate **when** 以及 **in what order** actions 必须执行:

- **Synchronization**: Agent C must wait 5 seconds after Agent D finishes welding
- **Parallel execution windows**: two agents must lift object simultaneously within 0.5-second tolerance
- **Dynamic adjustment**: extend task duration 响应 environmental delays, reschedule 当 prior steps overrun

### Compositional nature

这三类 constraints **不是 orthogonal 隔离**, 而是 integrate 起来:

$$\mathcal{C} = \{\mathcal{C}_l, \mathcal{C}_s, \mathcal{C}_t\}$$

Logical 定义 interaction protocols + shared objectives, temporal synchronize actions with task dependencies, spatial encode geometric + semantic boundaries。这种 unified framework 让 local decisions converge 成 robust, efficient, executable collaborative behaviors。

---

## 3. RoboFactory 框架: RoboBrain + RoboChecker 双模块架构

### 3.1 整体 pipeline

![](https://iranqin.github.io/robofactory/static/images/teaser.png)

Input:
- Global task instruction $\mathcal{G}_{\mathrm{global}}$ (e.g., "Grab the steak and use the camera to photograph it with 4 Embodied Agents")
- RGB observations $\mathcal{O} = \{o_{\mathrm{global}}, o_1, ..., o_n\}$, 包括 1 个 global view + n 个 ego-centric views
- n agents $\{a_1, a_2, ..., a_n\}$

### 3.2 RoboBrain 的核心公式

$$\mathcal{G}^{\mathrm{next}}, \mathcal{C} = \mathcal{F}_{\mathrm{VLM}}(\mathcal{O}, \mathcal{G}_{\mathrm{global}}, \mathcal{G}^{\mathrm{pre}}, f^{\mathrm{pre}}) \quad (1)$$

Variable breakdown:
- $\mathcal{G}^{\mathrm{next}} = \{g_1^{\mathrm{next}}, ..., g_n^{\mathrm{next}}\}$: 下一步每个 agent 的 subgoal set
- $\mathcal{C} = \{\mathcal{C}_l, \mathcal{C}_s, \mathcal{C}_t\}$: 三类 textual compositional constraints
- $\mathcal{F}_{\mathrm{VLM}}$: VLM function, 实际使用 GPT-4o
- $\mathcal{O}$: multi-view RGB observations
- $\mathcal{G}_{\mathrm{global}}$: text task instruction
- $\mathcal{G}^{\mathrm{pre}} = \{g_1^{\mathrm{pre}}, ..., g_n^{\mathrm{pre}}\}$: 上一轮的 subgoal set (history)
- $f^{\mathrm{pre}}$: Constraint Violation Feedback from RoboChecker, 如果上一轮有 violation, 这里会包含 failure reason 的 string

RoboBrain 还通过 **visual programming** invoke 预定义 motion primitives (类似 [Code as Policies](https://code-as-policies.github.io/)) 生成每个 agent 的 unconstrained trajectory $traj_1, ..., traj_r$。

### 3.3 RoboChecker: 从 text constraints 到 physical interfaces

核心 challenge: RoboBrain 输出的 constraints 是 textual space 的, 无法直接 constrain 真实 trajectory data。RoboChecker 的作用是 **text-to-physical interface transformation**。

对每个 constraint $c_i$ (e.g., "Avoid collision between Agent 2 and other Agents"), 构建 corresponding interface $h_i$, 加上 trajectory 和 robot states $\mathcal{S} = \{s_1, ..., s_n\}$, 生成 evaluation protocol (check code for trajectory)。

Protocol 输出:
- **boolean**: 是否违反 constraint
- **string**: violation reason, 作为 feedback $f^{\mathrm{pre}}$ 返回给 RoboBrain re-plan

### 3.4 四种 Constraint Interface 的 physical representation

#### (a) Interaction Position (Logical)

对每个 3D asset 标注 interaction positions。**同一物体不同 interaction 形式对应不同 positions**: grasping a camera vs using a camera 有 distinct interaction positions。

#### (b) Interaction Direction (Logical)

对每个 3D asset 标注 interaction directions。e.g., pressing camera shutter 要求 gripper 沿 shutter 的 **direct-facing direction** 移动。

#### (c) 3D Occupancy Interface (Spatial)

具体 pipeline:
1. 用 depth camera 或 [Depth Anything V2](https://depth-anything.com/) 做 depth estimation of current 3D scene
2. 基于 current robotic arm states 计算每个 joint point 的 absolute coordinates
3. Integrate depth information 获得 robotic arm + objects 的 occupancy information
4. **Voxel size: 5cm × 5cm × 5cm** 作为基本 discrete occupancy unit (trade-off between precision 和 computational cost)

#### (d) Temporal Occupancy Interface (Temporal)

对 sub-goal set $\mathcal{G}^{\mathrm{next}}$ 下所有需要移动的 agents 做 **dynamic occupancy modeling**。通过分析 temporal occupancy 信息 detect irrational scheduling + temporal logic violations。

### 3.5 RoboChecker 的 Validation Functions

形式化定义:

```
(Agent_ID, Target_Object, Required_Orientation) -> Validate_Direction()
(Agent_ID, Target_Object, Contact_Point) -> Validate_Interaction()
(Agent_IDs) -> Validate_Spatial_Occupancy()
(Agent_IDs, Task_Dependency_Type) -> Validate_Scheduling()
```

其中 `Task_Dependency_Type` ∈ {"Sequential", "Simultaneous"}。这种 design 直接对应到 paper 的核心 insight: **temporal constraint 在 trajectory 上的本质就是 ordering + synchronization**。

---

## 4. RoboFactory Benchmark: 11 tasks 的 multi-agent manipulation suite

### 4.1 Benchmark 对比 (Table 1)

| Benchmark | Single-agent | Multi-agent | Task Level |
|-----------|--------------|-------------|------------|
| EgoPlan-Bench | ✓ | ✗ | Plan |
| MMWorld | ✓ | ✗ | Plan |
| VAB | ✓ | ✗ | Plan |
| RoboCasa | ✓ | ✗ | Plan |
| RoboTwin | ✓ | ✗ | Plan & Control |
| **RoboFactory** | **✓** | **✓** | **Plan & Control** |

RoboFactory 是 **第一个** 同时具备 multi-agent + plan & control 的 benchmark。

### 4.2 Tasks 详情

基于 [ManiSkill3](https://github.com/haosulab/ManiSkill) simulator, 使用 [Franka Emika Panda Arm](https://franka.de/) (7-DoF + 1D gripper = 8-dim action per agent)。3D assets 来自 [PartNet-Mobility Dataset](https://sapien.ucsd.edu/)。

| Task | Agent # | Description |
|------|---------|-------------|
| Pick Meat | 1 | Pick meat, lift to height |
| Stack Cube | 1 | Stack blue on red |
| Strike Cube | 1 | Strike cube with hammer |
| Lift Barrier | 2 | Lift barrier from both ends (synchronized) |
| Pass Shoe | 2 | Pass shoe from one arm to another |
| Place Food | 2 | Open lid + place food (sequential) |
| Two Robots Stack Cube | 2 | Two-arm cube stacking |
| Camera Alignment | 3 | Pick object + align camera |
| Three Robots Stack Cube | 3 | Sequential 3-cube stacking |
| Take Photo | 4 | Pick object + align camera + press shutter |
| Long Pipeline Delivery | 4 | 4-arm shoe pipeline |

每个 task pre-collected 150 sets of data, 包含 camera RGB observations + joint actions。

---

## 5. 实验: Diffusion Policy baseline 的细致分析

### 5.1 Training Setup

采用 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 作为 base:
- CNN-based backbone
- Prediction horizon: **8** (model 同时预测未来 8 步)
- Observation steps: **3** (用过去 3 步观测)
- Action steps: **6** (实际执行 6 步, receding horizon)
- Batch size: 128
- Optimizer: AdamW, lr=$1.0 \times 10^{-4}$, betas=[0.95, 0.999], eps=$1.0 \times 10^{-8}$
- Warmup: 500 steps
- Epochs: 300
- Hardware: single RTX 4090
- Training time: ~5 hours for 150 demos (avg episode length 205)

### 5.2 Success Rate by Agent Number (Table 2)

| Agent # | 50 Demo | 100 Demo | 150 Demo |
|---------|---------|----------|----------|
| 1-Agent | 25% | 47% | **49%** |
| 2-Agent | 13% | 32.5% | 27.5% |
| 3-Agent | 7.5% | 6% | 20.5% |
| 4-Agent | 2.5% | 4% | **10%** |

**关键观察**:

1. **Performance cliff with agent count**: 1→2→3→4 agents, success rate 从 49% 暴跌到 10%。这是 **multi-agent coordination difficulty** 的直接体现, 而非单纯的 model capacity 问题。

2. **Long Pipeline Delivery = 0%**: 这个 task 需要长期 temporal dependency, DP 学不到。这暗示当前 diffusion-based policy 在 **long-horizon reasoning** 上的 fundamental limitation, 类似于 [pi0](https://www.physicalintelligence.company/blog/pi0) 等 VLA model 在 long-horizon task 上的困境。

3. **2-agent 在 100 demo 达 peak, 1/3/4-agent 在 150 demo 达 peak**: 2-agent task 中 individual agent action 较简单, 150 demo 出现 overfitting, 学到不 generalize 的 spurious patterns。

### 5.3 Multi-agent Imitation Learning Architectures (Table 3)

设计四种架构, 基于 **view scope** (global vs local) × **policy sharing** (shared vs separate):

| Arch | View | Policy | Lift Barrier | Place Food |
|------|------|--------|--------------|------------|
| Arch1 | Global | Shared | 49% | 5% |
| Arch2 | Local | Shared | 4% | 0% |
| Arch3 | Global | Separate | 26% | 17% |
| Arch4 | Local | Separate | **58%** | **20%** |

**关键分析**:

**为什么 Arch2 (Local + Shared) 完全崩溃 (4%, 0%)?**

当 shared policy 从不同 ego-views 学习时, 必须 **infer agent ID currently executing + generate corresponding action**。这相当于让 model 同时学 "我是谁" + "我该做什么", 模型 capacity 被 agent ID classification 任务占据, 严重 degrade performance。

**为什么 Arch4 (Local + Separate) 最好?**

- **Local view** 提供 richer, more detailed information 适合 fine-grained manipulation
- **Separate policy** 让每个 agent 专攻自己的 skill, 在 Place Food 这种两个 arm 技能 distinct 的 task 上尤其有效

**为什么 Arch3 (Global + Separate) 比 Arch1 (Global + Shared) 在 Place Food 上更好 (17% vs 5%)?**

Place Food 需要 distinct skills (open lid vs place food), shared policy 必须同时 handle 两种 skills, 容易互相干扰。Separate policy 让每个 agent 特化。

**直觉**: Multi-agent 系统的 optimal design = **local observations for spatial richness** + **separate policies for skill specialization**。这与 [MARL](https://arxiv.org/abs/2103.01955) 领域中 **centralized training decentralized execution (CTDE)** 的思路有相通之处。

---

## 6. Ablation Study: Constraints 的 contribution 量化

### 6.1 Success Rate Ablation (Table 4)

| Logical | Spatial | Temporal | Lift Barrier | Three Robots Stack | Take Photo |
|---------|---------|----------|--------------|--------------------| -----------|
| ✓ | ✗ | ✗ | 80.2 | 62.5 | 37.1 |
| ✓ | ✗ | ✓ | 85.4 | 84.2 | 62.2 |
| ✓ | ✓ | ✗ | 95.2 | 92.7 | 53.8 |
| ✓ | ✓ | ✓ | **97.5** | **98.9** | **88.2** |

### 6.2 Episode Length Ablation (Table 5)

| Logical | Spatial | Temporal | Lift Barrier | Three Robots Stack | Take Photo |
|---------|---------|----------|--------------|--------------------|------------|
| ✓ | ✗ | ✗ | 123 | 685 | 407 |
| ✓ | ✗ | ✓ | 92.8 | 452 | 238 |
| ✓ | ✓ | ✗ | 115 | 652 | 325 |
| ✓ | ✓ | ✓ | **80.7** | **424** | **204** |

**关键 insights**:

1. **Spatial constraints 对 success rate 提升最关键**: 在 Three Robots Stack 上, 加 spatial 后 62.5 → 92.7 (+30.2%)。没有 spatial constraints, robot arm 频繁碰撞, 没有纠正 spatial feedback。

2. **Temporal constraints 对 episode length 缩短最关键**: 在 Take Photo 上, 加 temporal 后 407 → 238 (-41.5%)。Temporal constraints 通过 detect parallel execution opportunities 大幅提升 efficiency。

3. **Three constraints 缺一不可**: Full constraint setup 在 Take Photo 上达到 88.2%, 相比 logical-only 37.1% 提升 51.1 个百分点。**这量化证明了 compositional 的必要性**。

4. **Episode length 减半 = training/inference 时间减半**: 这对 real-world deployment 至关重要。

---

## 7. RoboChecker 的实际运行示例

Paper supplementary 给出了 Take Photo task 的完整 RoboChecker 输出 (Figure 6):

```
CheckCode composition:
- VI (Validate Interaction): 检查 agent 与 object 的 contact point
- VD (Validate Direction): 检查 gripper orientation
- VSO (Validate Spatial Occupancy): 检查 voxel collision
- VS (Validate Scheduling): 检查 temporal ordering
```

CheckCode 返回 true 当且仅当所有 interface 通过 validation, 否则 identify failed interfaces 并把 feedback 发给 RoboBrain re-plan。这种设计本质上是 **neuro-symbolic reasoning**: LLM 做高层 planning + symbolic rules 做 verification。

---

## 8. 与 broader landscape 的关系

### 8.1 与 VLA Models ([OpenVLA](https://openvla.github.io/), [pi0](https://www.physicalintelligence.company/blog/pi0))

当前 VLA models 几乎全部是 single-agent。RoboFactory 揭示了一个 critical bottleneck: 当从 single → multi-agent 时, 即使 150 demos 也只能达到 10% success rate (4-agent)。这暗示未来 VLA 需要 **explicit multi-agent training** + **agent communication** 机制。

### 8.2 与 [CaM (Code-as-Monitor)](https://jianqy.github.io/Code-as-Monitor/)

CaM 引入 constraint-based elements for program synthesis。RoboFactory 把这个思路 extend 到 multi-agent 场景, 通过 constraint interface 实现 reactive + proactive failure detection。

### 8.3 与 [3D Diffusion Policy](https://3d-diffusion-policy.github.io/)

3D DP 用 point cloud observations 提升 geometric understanding。RoboFactory 的 spatial constraint interface 也依赖 3D occupancy modeling, 但用于 **verification** 而非 policy input。未来工作可能 merge 这两个方向: 用 3D representation 既做 policy input 又做 constraint verification。

### 8.4 与 Hierarchical RL

RoboBrain ↔ motion primitives 的关系本质上是 **hierarchical control**: high-level planner (LLM) + low-level controller (predefined primitives)。这与 [ACT](https://tonyzhaozh.github.io/aloha/) 等 chunking-based 方法形成对比: ACT 在单一 policy 内部做 hierarchical, 而 RoboFactory 在 system 层面做 hierarchical。

### 8.5 与 World Models

RoboChecker 在某种意义上扮演 **verifier world model** 的角色: 它 predicts trajectory 是否会 violate constraints, 并提供 feedback。这暗示一种 **search-based planning** 范式: LLM 提议, world model 验证, 类似于 OpenAI 的 [o1 思路](https://openai.com/o1/) 在 robotics 上的应用。

---

## 9. Limitations 与未来方向

### 9.1 Paper 自己指出的 limitation

> "The constraints may struggle to accurately model intricate physical phenomena, potentially limiting their applicability in tasks requiring precise interactions."

e.g., soft body deformation, fluid dynamics, friction-dependent tasks 无法用 5cm voxel 或 simple contact point 建模。

### 9.2 我能想到的更深 limitations

1. **RoboChecker 是 hand-crafted rules, 没有 learning**: 四种 validation functions 都是 predefined 的, 无法 generalize 到 unseen constraint types。未来应该用 learned verifier (类似 [VLM-based reward model](https://arxiv.org/abs/2402.04764)) 替代。

2. **GPT-4o 的 reasoning 是 black box**: 生成 subgoal 和 constraints 的 process 无法 fine-tune, error propagation 严重。如果 GPT-4o hallucinate 一个错误的 constraint, 整个 pipeline 失败。

3. **Motion primitives 是 predefined 的**: 限制了 task diversity。如果新 task 需要 unseen motion, 无法处理。这类似于 [BC-Z](https://arxiv.org/abs/2102.09298) 的 limitations。

4. **没有 agent communication 机制**: 每个 agent 只看自己的 ego-view, 无法显式 communicate intentions。这在 [Multi-agent POMDP](https://arxiv.org/abs/2301.01307) 理论中是 sub-optimal 的。

5. **ManiSkill simulator 的 sim-to-real gap**: 虽然 paper 没有讨论, 但所有 experiments 在 simulation 中。Real-world deployment 时, depth estimation 误差 + voxel discretization 误差会放大。

### 9.3 可能的 future directions

- **Learned constraint verifier**: 用 VLM 自己做 constraint checking, 不依赖 hand-crafted interfaces
- **Multi-agent VLA**: 端到端训练一个 VLA 处理 multi-agent planning + control
- **Implicit communication**: agents 通过 observation 互相推断 intentions, 不需要 explicit communication channel
- **Differentiable constraint interface**: 把 constraints 编入 policy training的 loss function, 让 policy 在 training 时就学会 respect constraints
- **Real-world deployment**: 通过 sim-to-real transfer 验证 framework 的实际可用性

---

## 10. 给你的几个思考点 (Karpathy-style intuition)

1. **Multi-agent 是 VLA 的下一个 frontier**: 这个 paper 用 pure engineering (LLM + symbolic checker) 暴露了 multi-agent embodied AI 的 fundamental difficulty。Success rate 从 49% → 10% 的 cliff 表明, 仅靠 scaling single-agent methods 不够, 需要 architectural innovation。

2. **Constraints 是 inductive bias**: Compositional constraints 本质上是把 human prior knowledge about safe collaboration encode 成 verification rules。这类似于 CNN 的 translation invariance prior, 但是 multi-agent collaboration domain 的 prior。问题是: 这些 prior 应该 hand-craft 还是 learn?

3. **Neural + Symbolic 在 robotics 的 sweet spot**: RoboBrain (neural, LLM) + RoboChecker (symbolic, rules) 的二分法非常 elegant。Neural 负责 generalization + reasoning, symbolic 负责 verification + safety。这可能是 AGI 在 safety-critical domain 的范式。

4. **Data quality >> Data quantity**: Ablation 显示加 constraints 让 episode length 减半, 意味着同样训练时间能 cover 2x scenarios。在 robotics data scarcity 的当下, 这种 data efficiency 提升可能比 success rate 数字本身更重要。

5. **LLM-as-orchestrator + symbolic-verifier 是范式**: 这个 paper 验证了一个 pattern, 未来我们可能看到更多 "LLM propose, verifier check, executor act" 的 robotics system。LLM 的 reasoning capability + symbolic verifier 的 reliability 是互补的。

---

## References

- [RoboFactory Project Page](https://iranqin.github.io/robofactory/)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [RoboTwin](https://robotwin.github.io/)
- [ManiSkill3](https://github.com/haosulab/ManiSkill)
- [PartNet-Mobility Dataset](https://sapien.ucsd.edu/)
- [Depth Anything V2](https://depth-anything.com/)
- [Code as Policies](https://code-as-policies.github.io/)
- [Code-as-Monitor (CaM)](https://jianqy.github.io/Code-as-Monitor/)
- [OpenVLA](https://openvla.github.io/)
- [pi0](https://www.physicalintelligence.company/blog/pi0)
- [3D Diffusion Policy](https://3d-diffusion-policy.github.io/)
- [ACT / ALOHA](https://tonyzhaozh.github.io/aloha/)
- [GPT-4o Technical Report](https://arxiv.org/abs/2303.08774)
- [RoboCasa](https://robocasa.github.io/)
- [EgoPlan-Bench](https://arxiv.org/abs/2312.16172)

希望这个 breakdown 帮你 build intuition about how multi-agent embodied AI 的 data generation pipeline 应该 design。如果你想 deep dive 某个 specific aspect (e.g., temporal occupancy 的具体实现, 或者 multi-agent DP 的架构对比), 我可以进一步 expand。
