---
source_pdf: RoboFactory Exploring Embodied Agent Collaboration with Compositional
  Constraints.pdf
paper_sha256: 802941cb245dba969658f88f08619efc933cad095ddcfd3f30ef1d7da62088d6
processed_at: '2026-08-12T00:51:36-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 RoboFactory

## 一句话版本

这篇paper想干的事情：让好几个robotic arm一起干活，但不让它们互相打架、不让它们干蠢事、不让它们磨蹭。

---

## 问题到底在哪

想象4个robotic arm要完成"拍牛排照片"这个任务。一个抓牛排，两个抬camera，一个按shutter。听起来简单，但实际上会出各种幺蛾子：

**第一种蠢事**：a₃去抓camera的lens而不是handle。这是**逻辑搞错了**——它不知道该碰哪里、从哪个方向碰。

**第二种蠢事**：a₂和a₃同时往中间伸，撞了。这是**空间搞错了**——它们不知道彼此占了哪些位置。

**第三种蠢事**：a₄非要等a₂、a₃完全停下来才动，但其实它们可以同时进行。这是**时间搞错了**——白白浪费了parallel execution的机会。

single-agent的时候这些问题不存在，因为就一个arm，自己跟自己不会冲突。一到multi-agent，decision space爆炸式复杂化。

---

## 作者的解法：三类constraint

作者把所有可能出的幺蛾子归成三类：

### Logical Constraint（逻辑约束）
**"谁该碰哪里、从哪个方向碰"**

每个3D object上都预先annotate好了合法的interaction point和合法的direction。比如camera上有两个点：一个grip point（用来抓），一个shutter point（用来按）。grip point的合法方向是横向，shutter point的合法方向是垂直向下。

如果trajectory里gripper要去碰lens，或者方向不对，RoboChecker直接判定违规。

技术细节：对每个object $o_k$，定义合法接触点集合 $P_k$ 和合法方向集合 $D_k$。在timestep $t$，如果gripper接触object $o_k$，需要满足：

$$p_{\text{contact}}(t) \in P_k \quad \text{且} \quad \hat{n}_{\text{gripper}}(t) \cdot \hat{d}_k > \cos(\theta_{\text{tol}})$$

这里 $p_{\text{contact}}(t)$ 是gripper末端的接触位置，$\hat{n}_{\text{gripper}}(t)$ 是gripper的法向量，$\hat{d}_k$ 是合法方向，$\theta_{\text{tol}}$ 是容差角度。如果点乘结果大于 $\cos(\theta_{\text{tol}})$，说明gripper方向和合法方向夹角在容差范围内。

### Spatial Constraint（空间约束）
**"别撞车"**

把整个3D scene切成5cm×5cm×5cm的小方块（voxel）。每个arm的每个关节、scene里每个object都映射到这些voxel上。如果两个arm在同一时刻占了同一个voxel，就是collision，违规。

技术细节：定义occupancy function $\mathcal{V}(x,y,z,t)$，表示位置 $(x,y,z)$ 在时刻 $t$ 是否被某个arm占据。对任意两个agent $i, j$ 和任意时刻 $t$：

$$\text{Vol}(\mathcal{V}_i(\cdot, t) \cap \mathcal{V}_j(\cdot, t)) = 0$$

这里 $\mathcal{V}_i(\cdot, t)$ 是agent $i$ 在时刻 $t$ 占据的所有voxel的集合，$\text{Vol}(\cdot)$ 是体积函数。交集体积为零意味着没有重叠。

5cm voxel是精度和计算量的trade-off——太小计算量爆炸，太大检测不到细小碰撞。

### Temporal Constraint（时间约束）
**"什么时候该干嘛"**

分两种：
- **Sequential**：a₅先开锅盖，a₄才能放食物
- **Simultaneous**：a₂和a₃必须在0.5秒内同时lift camera

技术细节：定义temporal relation $R$，每个元素是 $(g_i, g_j, r)$，其中 $r \in \{\text{seq}, \text{sync}\}$：
- 若 $r = \text{seq}$，要求 $t_{\text{start}}(g_j) \geq t_{\text{end}}(g_i)$，即 $g_j$ 的开始时间不早于 $g_i$ 的结束时间
- 若 $r = \text{sync}$，要求 $|t_{\text{start}}(g_j) - t_{\text{start}}(g_i)| \leq \delta_{\text{tol}}$，即两个subgoal开始时间差不超过容差 $\delta_{\text{tol}}$

---

## 两个核心模块怎么配合

### RoboBrain（大脑）

输入：global task描述 + RGB图像（一个global view + 每个agent的ego view）+ 之前做了什么 + 之前的违规反馈

输出：每个agent的下一个subgoal + 三类textual constraint

公式1长这样：
$$\mathcal{G}^{\text{next}}, \mathcal{C} = \mathcal{F}_{\text{VLM}}(\mathcal{O}, \mathcal{G}_{\text{global}}, \mathcal{G}^{\text{pre}}, f^{\text{pre}})$$

变量含义：
- $\mathcal{O}$：observation集合，包括global view $o_{\text{global}}$ 和 $n$ 个ego view $o_1, ..., o_n$
- $\mathcal{G}_{\text{global}}$：全局任务指令（比如"4个agent拍牛排照片"）
- $\mathcal{G}^{\text{pre}}$：上一轮的subgoal集合 $\{g_1^{\text{pre}}, ..., g_n^{\text{pre}}\}$
- $f^{\text{pre}}$：上一轮RoboChecker返回的违规反馈
- $\mathcal{G}^{\text{next}}$：这一轮生成的subgoal集合 $\{g_1^{\text{next}}, ..., g_n^{\text{next}}\}$
- $\mathcal{C} = \{\mathcal{C}_l, \mathcal{C}_s, \mathcal{C}_t\}$：logical、spatial、temporal三类constraint的集合
- $\mathcal{F}_{\text{VLM}}$：VLM函数，具体实现是GPT-4o

RoboBrain不直接输出joint angle，而是输出motion primitive的调用代码，比如 `MOVE(agent_id=1, target_position=[0.3, 0.2, 0.5])`。这样LLM只管high-level逻辑，不用管low-level控制信号。

### RoboChecker（质检员）

把RoboBrain输出的text constraint变成可以check的Python代码，实时监控trajectory执行。四个validation function：

1. **Validate_Direction(agent_id, target_object, required_orientation)**：检查gripper方向对不对
2. **Validate_Interaction(agent_id, target_object, contact_point)**：检查接触点对不对
3. **Validate_Spatial_Occupancy(agent_ids)**：检查有没有collision
4. **Validate_Scheduling(agent_ids, task_dependency_type)**：检查时序对不对

任何一个function返回False，trajectory立刻halt，把violation reason反馈给RoboBrain重新plan。这是closed-loop的关键。

---

## Benchmark：11个任务

基于ManiSkill 3 simulator，Franka Emika Panda 7-DoF arm。任务从1个agent到4个agent递进：

| Task | Agents | 干什么 |
|------|--------|--------|
| Pick Meat | 1 | 抓牛排抬起来 |
| Stack Cube | 1 | 蓝cube叠红cube |
| Strike Cube | 1 | 抓锤子敲cube |
| Lift Barrier | 2 | 两个arm同时抬杆子两端 |
| Pass Shoe | 2 | 一个arm传鞋给另一个 |
| Place Food | 2 | 一个开锅盖一个放食物 |
| Two Robots Stack Cube | 2 | 一个放蓝一个叠红 |
| Camera Alignment | 3 | 一个持object两个align camera |
| Three Robots Stack Cube | 3 | 三个arm依次叠cube |
| Take Photo | 4 | 抓牛排+抬camera+按shutter |
| Long Pipeline Delivery | 4 | 4个arm接力传鞋 |

---

## 实验结果告诉我们什么

### 结果1：当前方法在multi-agent上很拉胯

| Agent数 | 150 demo success rate |
|---------|----------------------|
| 1 agent | 49% |
| 2 agent | 27.5% |
| 3 agent | 20.5% |
| 4 agent | 10% |

Long Pipeline Delivery（4个arm接力）直接0%。这说明当前Diffusion Policy在multi-agent + long-horizon上是**fundamentally broken**的。

### 结果2：最佳架构是Local View + Separate Policy

| 架构 | Lift Barrier | Place Food |
|------|-------------|------------|
| Shared Policy + Global View | 49% | 5% |
| Shared Policy + Local View | 4% | 0% |
| Separate Policy + Global View | 26% | 17% |
| Separate Policy + Local View | **58%** | **20%** |

**Intuition**：

Shared Policy + Local View最差（4%、0%）。原因：shared policy要从ego view猜"我是哪个agent"，再决定output什么action。两个agent的ego view可能长得很像，但需要的action完全不同，policy根本学不出来。这就好比两个人长得差不多，但一个要往左一个要往右，model分不清。

Lift Barrier用Shared Policy + Global View反而不错（49%），因为它是symmetric task，两个arm做一样的事，shared policy可以leverage symmetry。但Place Food是asymmetric task（一个开盖一个放食物），shared policy就不行了（5%）。

**结论**：multi-agent不一定要share backbone，每个agent独立policy + ego-centric view反而更robust。这对未来设计multi-agent VLA model有直接指导意义——别盲目追求shared model。

### 结果3：三类constraint缺一不可

Success rate：

| Logical | Spatial | Temporal | Take Photo |
|---------|---------|---------|------------|
| ✓ | ✗ | ✗ | 37.1% |
| ✓ | ✓ | ✗ | 53.8% |
| ✓ | ✗ | ✓ | 62.2% |
| ✓ | ✓ | ✓ | **88.2%** |

Episode length（越短越好）：

| Logical | Spatial | Temporal | Take Photo |
|---------|---------|---------|------------|
| ✓ | ✗ | ✗ | 407 steps |
| ✓ | ✓ | ✗ | 325 steps |
| ✓ | ✗ | ✓ | 238 steps |
| ✓ | ✓ | ✓ | **204 steps** |

**Intuition**：

Spatial constraint对success rate影响最大（37.1→53.8）。没有它，arm频繁collision直接失败。Spatial是"能不能做成"的bottleneck。

Temporal constraint对episode length影响最大（325→204）。没有它，agent全部串行执行，浪费时间。Temporal是"做得快不快"的bottleneck。

三者compositional效果（88.2%）远超任何两者组合，说明它们是complementary的，不是redundant的。

---

## 这篇paper真正的价值

1. **第一次把multi-agent embodied collaboration的constraint形式化成三个orthogonal维度**，这是conceptual contribution
2. **第一个multi-agent manipulation benchmark**，11个task覆盖1-4 agent，这是infrastructure contribution
3. **揭示了当前imitation learning在multi-agent上的fundamental limitation**，4-agent只有10%，这是empirical contribution
4. **发现Local View + Separate Policy > Global View + Shared Policy**，对未来multi-agent VLA设计有直接指导意义

## 潜在问题

- RoboBrain和RoboChecker都依赖GPT-4o，API cost高，academic lab复现门槛高
- Constraint是LLM显式生成的text，可能难以建模deformable object、fluid dynamics这种complex physical phenomena
- Closed-loop反馈需要反复call LLM重新plan，long-horizon task的latency可能explode
- 目前在simulation里，sim-to-real gap未知，5cm voxel在real camera noise下可能失效

---

## 相关链接

- 项目主页：https://iranqin.github.io/robofactory/
- arXiv：https://arxiv.org/abs/2502.13131
- ManiSkill 3：https://github.com/haosulab/ManiSkill
- Diffusion Policy：https://diffusion-policy.cs.columbia.edu/
- RoboTwin：https://robotwin-benchmark.github.io/
- Code-as-Monitor：https://arxiv.org/abs/2412.04455
- Depth Anything V2：https://depth-anything-v2.github.io/
- PartNet-Mobility：https://sapien.ucsd.edu/browse
- GPT-4o：https://openai.com/index/hello-gpt-4o/
- π₀ VLA model：https://www.physicalintelligence.company/blog/pi0
- OpenVLA：https://openvla.github.io/

---

# RoboFactory 深度讲解

## 1. 核心动机：从 Single-Agent 到 Multi-Agent 的鸿沟

Andrej，这篇paper要解决的核心问题是 **scaling from single-agent embodied systems to multi-agent embodied systems** 时遇到的 data bottleneck。当前的 LLM/VLM 驱动的 single-agent data generation pipeline（比如 RoboTwin、RoboCodeX）通过 motion primitives 让 LLM 来 generate trajectory，已经 work 得不错。但一旦扩展到 multi-agent，decision space 的复杂度爆炸：

- **Logical 维度**：哪个 agent 抓 steak？哪个 agent 按 shutter？
- **Spatial 维度**：两个 arm 同时去 lift camera，trajectory 不能碰撞
- **Temporal 维度**：是 sequential 还是 parallel execution？什么时候同步？

如果只是简单地把 single-agent pipeline 复制 N 份，每个 agent 各干各的，就会产生 Figure 1 里的各种 failure：a₃ 抓 camera lens 损坏硬件、a₂ 和 a₃ trajectory collision、a 串行等待浪费时间。这就是 paper 想要 fix 的痛点。

参考链接：
- RoboFactory 项目页：https://iranqin.github.io/robofactory/
- 原始 arXiv：https://arxiv.org/abs/2502.13131
- ManiSkill 3 simulator：https://github.com/haosulab/ManiSkill

---

## 2. Compositional Constraints：三个维度的形式化

这是 paper 的核心 conceptual contribution。作者把 multi-agent collaboration 中的约束 decompose 成三个 orthogonal 的维度，然后 compose 起来。

### 2.1 Logical Constraints $\mathcal{C}_l$

定义 agent 与 environment 之间的 **interaction protocol**。具体包含：

- **Interaction Position**：每个 3D asset 上 annotate 了合法的 interaction point。比如 camera 的 grip point 和 shutter press point 是不同的；steak 的 grasp point 在中间而不是边缘。
- **Interaction Direction**：每个 interaction point 上的合法方向。比如按 shutter 需要 gripper perpendicular to shutter surface。

形式化：对于每个 object asset $o_k$，定义 interaction position set $P_k = \{p_k^1, p_k^2, ...\}$ 和 direction set $D_k = \{d_k^1, d_k^2, ...\}$。Trajectory $\tau_i$ 在 timestep $t$ 接触 object $o_k$ 时，需要满足：

$$p_{\text{contact}}(t) \in P_k \quad \text{and} \quad \hat{n}_{\text{gripper}}(t) \cdot \hat{d}_k > \cos(\theta_{\text{tol}})$$

其中 $p_{\text{contact}}(t)$ 是 gripper 末端 contact point，$\hat{n}_{\text{gripper}}(t)$ 是 gripper normal vector，$\hat{d}_k$ 是合法 interaction direction，$\theta_{\text{tol}}$ 是方向容差角度。

### 2.2 Spatial Constraints $\mathcal{C}_s$

定义 agent 在 physical space 中的合法行为。Paper 用 **3D occupancy grid** 来实现：

- 对当前场景做 depth estimation（用 Depth Anything V2 [44] 或者深度相机）
- 基于 robotic arm 当前 joint states，计算每个 joint 的 absolute world coordinate
- 把 scene + arm 都 voxel 化，voxel size 是 $5 \times 5 \times 5$ cm$^3$
- 对每个 agent $a_i$ 的 trajectory $\tau_i$，检查 voxel occupancy 是否与其他 agent 或 obstacle 重叠

形式化：定义 occupancy function $\mathcal{V}: \mathbb{R}^3 \to \{0, 1\}$。对 timestep $t$：

$$\mathcal{V}_i(p, t) = 1 \iff \exists \text{ part of } a_i \text{ at position } p \text{ at time } t$$

Spatial constraint 要求：

$$\forall t, \forall (i, j), i \neq j: \quad \text{Vol}(\mathcal{V}_i(\cdot, t) \cap \mathcal{V}_j(\cdot, t)) = 0$$

5cm voxel size 是 computation cost 和 collision detection precision 之间的 trade-off。

### 2.3 Temporal Constraints $\mathcal{C}_t$

定义 agent 之间的 **timing relationship**。这一类很关键，因为单看 spatial constraint 会导致过度保守——比如 a₂ 在 $t_0$ 用了某 voxel，a₄ 在 $t_1$ 才需要同一 voxel，但如果 strict spatial occupancy 就会 block。

Paper 把 temporal constraint 分成两种：

- **Sequential**：$a_5$ 必须先 open lid，$a_4$ 才能 place food
- **Simultaneous**：$a_2$ 和 $a_3$ 必须在 0.5 秒 tolerance 内同步 lift camera

形式化：对 subgoal 集合 $\mathcal{G}^{\text{next}}$，定义 temporal relation $R \subseteq \mathcal{G}^{\text{next}} \times \mathcal{G}^{\text{next}} \times \{\text{seq}, \text{sync}\}$。对每个 $(g_i, g_j, r) \in R$：

- 若 $r = \text{seq}$：$t_{\text{start}}(g_j) \geq t_{\text{end}}(g_i)$
- 若 $r = \text{sync}$：$|t_{\text{start}}(g_j) - t_{\text{start}}(g_i)| \leq \delta_{\text{tol}}$

这种 decomposition 让 RoboChecker 可以做 fine-grained spatial awareness at each timestep（"this voxel is occupied at $t_0$ but free at $t_1$"），从而允许 temporal-spatial sharing。

### 2.4 Compositional 之所以叫 Compositional

三种 constraints 是 **orthogonal 但 composable** 的：
- Logical 不关心 timing，只关心 "can this action happen on this object in this direction"
- Spatial 不关心 timing 和 logic，只关心 geometry
- Temporal 不关心 geometry，只关心 ordering

但 real-world task 同时受三者约束。比如 "a₄ press shutter" 这个 action 需要：
- Logical：a₄ 的 gripper 必须对准 shutter contact point，方向 perpendicular
- Spatial：a₄ 的 arm 不能撞到 a₂、a₃ 在 lift 的 camera
- Temporal：a₄ press 必须在 a₁ 抓住 steak 且 a₂、a₃ 完成对齐之后

---

## 3. RoboFactory 框架架构

### 3.1 整体 Pipeline

核心是两个模块：**RoboBrain**（planner）+ **RoboChecker**（validator）。这是经典的 **generate-then-validate** 范式，类似 Code-as-Monitor [51] 的思路，但扩展到 multi-agent setting。

#### 输入

- Global task instruction $\mathcal{G}_{\text{global}}$（"Grab steak and use camera to photograph it with 4 agents"）
- Observation $\mathcal{O} = \{o_{\text{global}}, o_1, ..., o_n\}$，包括 1 个 third-person global view 和 $n$ 个 ego-centric view（每个 agent 一个）
- Previous subgoals $\mathcal{G}^{\text{pre}} = \{g_1^{\text{pre}}, ..., g_n^{\text{pre}}\}$
- Previous feedback $f^{\text{pre}}$（来自 RoboChecker 的 violation reason）

#### 核心公式（公式 1）

$$\mathcal{G}^{\text{next}}, \mathcal{C} = \mathcal{F}_{\text{VLM}}(\mathcal{O}, \mathcal{G}_{\text{global}}, \mathcal{G}^{\text{pre}}, f^{\text{pre}})$$

变量解释：
- $\mathcal{G}^{\text{next}} = \{g_1^{\text{next}}, ..., g_n^{\text{next}}\}$：每个 agent 的下一个 subgoal（自然语言描述，如 "Agent_1: grasp the steak"）
- $\mathcal{C} = \{\mathcal{C}_l, \mathcal{C}_s, \mathcal{C}_t\}$：三类 textual constraint
- $\mathcal{F}_{\text{VLM}}$：VLM 函数（具体用 GPT-4o [1]）
- 上下标：上标 "next"/"pre" 表示时序前后；下标 $l, s, t$ 分别表示 logical/spatial/temporal

#### Trajectory Generation

RoboBrain 不直接输出 joint angles，而是用 **visual programming** 调用 motion primitives（Python functions）。比如：

```python
MOVE(agent_id=1, target_position=[x, y, z])
GRASP(agent_id=1, object_id="steak", grasp_point=...)
```

这种 abstraction 让 LLM 只关注 high-level logic，避免 low-level control 信号计算（trick 来自 RoboTwin [29]）。

#### Validation Loop

RoboChecker 把 textual constraint 转成 **CheckCode**（一组 Python function），对 trajectory 实时检查。如果某个 interface 返回 False，立刻 halt，并把 violation reason 作为 $f^{\text{pre}}$ 反馈给 RoboBrain 重新规划。这是 **closed-loop** 的关键。

### 3.2 Constraint Interface：从 Text 到 Physics

RoboChecker 的核心创新是把 textual constraint 转成可以 interact with physics 的 representation。Paper 设计了 4 类 validation function：

#### Function 1: `Validate_Direction(agent_id, target_object, required_orientation)`

检查 agent 在 interaction 时的 gripper orientation。比如 constraint 是 "The gripper of Agent_1 must be perpendicular to {Object}"，则从 trajectory 中提取 grasp moment，compute gripper normal vector，verify dot product with object surface normal > threshold。

#### Function 2: `Validate_Interaction(agent_id, target_object, contact_point)`

检查 gripper 末端 contact point 是否在 annotated 的 legal interaction position set 内。比如 "Agent_3 must grasp Object_B at its left point"，则从 trajectory 提取 gripper position at grasp event，检查是否在 object_B 的 left_point 邻域。

#### Function 3: `Validate_Spatial_Occupancy(agent_ids)`

对每个 agent 的 trajectory 做 voxelization，构建 4D occupancy tensor $\mathcal{V}(x, y, z, t)$，检查 agent 之间 voxel 不重叠。

#### Function 4: `Validate_Scheduling(agent_ids, task_dependency_type)`

检查 subgoal 之间的 temporal relation。从 trajectory 中提取每个 subgoal 的 start time 和 end time，verify sequential 或 simultaneous constraint。

Figure 6 给出了一个完整的 Take Photo task 的 CheckCode 例子，组合了 VI（Validate Interaction）、VD（Validate Direction）、VSO（Validate Spatial Occupancy）、VS（Validate Scheduling）。只有所有 interface 都 pass，CheckCode 才返回 True。

---

## 4. RoboFactory Benchmark

### 4.1 设计

基于 ManiSkill 3 [35] simulator，Franka Emika Panda 7-DoF arm + parallel gripper。3D assets 来自 PartNet-Mobility Dataset [42]。共 11 个 task，覆盖 1/2/3/4 agent 场景：

| Task | # Agents | Description |
|------|----------|-------------|
| Pick Meat | 1 | Pick steak 到指定高度 |
| Stack Cube | 1 | 蓝色叠红色 cube |
| Strike Cube | 1 | Pick hammer 然后敲 cube |
| Lift Barrier | 2 | 两个 arm 同时 lift 长杆两端 |
| Pass Shoe | 2 | 一个 arm 抓 shoe 传给另一个 |
| Place Food | 2 | 一个开锅盖，另一个放食物 |
| Two Robots Stack Cube | 2 | 一个放蓝 cube，另一个叠红 cube |
| Camera Alignment | 3 | 一个 arm 持 object，另两个 align camera |
| Three Robots Stack Cube | 3 | 三个 arm 依次叠 cube |
| Take Photo | 4 | 1 抓 steak + 2 持 camera + 1 按 shutter |
| Long Pipeline Delivery | 4 | 4 个 arm 接力传递 shoe |

每个 task 预采集 150 expert demo（RGB + joint action）。每个 agent 配 ego-centric camera，外加 1 个 global camera。

### 4.2 与其他 benchmark 对比

Table 1 显示，RoboFactory 是第一个同时具备 **multi-agent + plan & control** 双重特性的 benchmark。EgoPlan-Bench、MMWorld 只做 plan 不做 control；RoboCasa、RoboTwin 做 plan + control 但只有 single-agent。

---

## 5. 实验结果详解

### 5.1 Diffusion Policy 基线（Table 2）

用 CNN-based Diffusion Policy [6]，prediction horizon = 8，obs steps = 3，action steps = 6。每个 agent 独立训练一个 policy（Local View + Separate Policy），input 是 ego-view RGB (320×240)。

| Task Level | 50 Demo | 100 Demo | 150 Demo |
|-----------|---------|----------|----------|
| 1-Agent avg | 25% | 47% | 49% |
| 2-Agent avg | 13% | 32.5% | 27.5% |
| 3-Agent avg | 7.5% | 6% | 20.5% |
| 4-Agent avg | 2.5% | 4% | 10% |

**关键观察**：
1. **Scaling law for data**：1-agent 和 3-agent、4-agent 在 150 demo 达到 peak，说明 multi-agent 任务对 data 量需求更大。
2. **Overfitting 现象**：2-agent 在 100 demo 达到 peak (32.5%)，150 demo 反而下降到 27.5%。作者归因于 individual agent action 相对简单，150 demo 产生 overfit。这个 observation 很有意思——说明 multi-agent 的 data efficiency 和 task complexity 之间有 non-monotonic 关系。
3. **Agent 数量与 success rate 负相关**：从 1-agent (49%) 到 4-agent (10%) 急剧下降，说明 collaboration 难度指数级增长。
4. **Long Pipeline Delivery 0%**：4-agent 接力任务完全失败，暴露 diffusion policy 在 long-horizon temporal dependency 上的弱点。

这印证了一个重要 intuition：**当前 imitation learning 方法在 multi-agent + long-horizon 上是 fundamentally broken 的**。

### 5.2 Multi-Agent 架构对比（Table 3）

这是 paper 里我觉得最有 insight 的实验。作者设计了 4 种 architecture，在 Lift Barrier 和 Place Food 两个 2-agent task 上对比：

| Policy | View Scope | Lift Barrier | Place Food |
|--------|-----------|--------------|------------|
| Shared | Global (Arch1) | 49% | 5% |
| Shared | Local (Arch2) | 4% | 0% |
| Separate | Global (Arch3) | 26% | 17% |
| Separate | Local (Arch4) | 58% | 20% |

**Intuition 分析**：

**Arch1 vs Arch2（Shared Policy 下 Global vs Local）**：从 49% → 4% 和 5% → 0%，灾难性下降。原因：shared policy 需要从 ego-view 推断 "我是哪个 agent"，再 generate action。这个 ID inference 是 ambiguous 的——两个 agent 的 ego-view 可能很相似，但需要的 action 不同，policy 学不出来。Shared policy 在 global view 下反而 OK，因为 global view 包含所有 agent 信息，policy 不需要猜 ID。

**Arch1 vs Arch3（Shared vs Separate，Global View）**：Lift Barrier 49% → 26%（shared 更好），Place Food 5% → 17%（separate 更好）。**关键 insight**：Lift Barrier 是 symmetric task（两个 arm 做一样的事），shared policy 可以 leverage symmetry，效果更好；Place Food 是 asymmetric task（一个开盖，一个放食物），需要不同 skill，separate policy 可以 specialize。

**Arch3 vs Arch4（Separate Policy 下 Global vs Local）**：26% → 58% 和 17% → 20%。Local view 提供了更 fine-grained 的 manipulation 信息。这说明对于 low-level control，ego-centric view 比 third-person view 更 informative。

**综合最佳：Arch4（Local View + Separate Policy）**。这个结论对未来 multi-agent VLA model 设计有指导意义——multi-agent 不一定要 share backbone，独立 policy + local view 反而更 robust。

### 5.3 Constraint Ablation（Table 4 & Table 5）

在 Lift Barrier、Three Robots Stack Cube、Take Photo 三个任务上做 constraint 消融。

**Success Rate（越高越好）**：

| Logical | Spatial | Temporal | Lift Barrier | Three Robots Stack | Take Photo |
|---------|---------|---------|--------------|---------------------|------------|
| √ | × | × | 80.2 | 62.5 | 37.1 |
| √ | × | √ | 85.4 | 84.2 | 62.2 |
| √ | √ | × | 95.2 | 92.7 | 53.8 |
| √ | √ | √ | **97.5** | **98.9** | **88.2** |

**Average Episode Length（越短越好）**：

| Logical | Spatial | Temporal | Lift Barrier | Three Robots Stack | Take Photo |
|---------|---------|---------|--------------|---------------------|------------|
| √ | × | × | 123 | 685 | 407 |
| √ | × | √ | 92.8 | 452 | 238 |
| √ | √ | × | 115 | 652 | 325 |
| √ | √ | √ | **80.7** | **424** | **204** |

**Intuition 分析**：

1. **Spatial constraint 对 success rate 影响最大**：从 80.2 → 95.2（Lift Barrier）。没有 spatial constraint，arm 频繁 collision，直接失败。Spatial constraint 是 data generation 成功率的 bottleneck。

2. **Temporal constraint 对 episode length 影响最大**：从 115 → 80.7（Lift Barrier），从 685 → 424（Three Robots Stack）。没有 temporal constraint，agent 全部串行执行，浪费时间。Temporal constraint 让 agent 可以 parallel execute，大幅缩短 episode。

3. **Compositional > Single**：三者一起用 (97.5%) 显著优于任意两者组合（最高 95.2%）。三种 constraint 是 complementary 而非 redundant 的。

4. **Take Photo task** 受益于 temporal constraint 最大（37.1 → 88.2），因为 4-agent 任务需要精细的同步调度。

---

## 6. Limitation & Future Direction

Paper 自己承认的 limitation：constraint 可能难以精确建模 complex physical phenomena，比如 deformable object manipulation、fluid dynamics、contact-rich 任务中的 friction modeling。

**我从这篇 paper 看到的几个 future direction**：

1. **Learning-based constraint discovery**：目前 constraint 是 RoboBrain 用 LLM 显式 generate 的 text。未来可以让 model 从 demonstration 中 **隐式学习 constraint**，类似 differentiable optimization 的思路。

2. **Multi-agent VLA model**：当前是 per-agent 独立 Diffusion Policy。如果有一个 unified VLA（像 π₀ [2] 那样）直接 input global view + instruction，output joint action，会怎样？这个 paper 的 Arch1 实验暗示这条路有挑战，但值得探索。

3. **Constraint hierarchy**：现在的 constraint 是 flat 的。未来可以引入 hierarchical constraint（task-level → subtask-level → motion-level），类似 HTN planning。

4. **Real-world transfer**：现在在 ManiSkill simulation 里。如果 sim-to-real gap 大，spatial constraint 的 5cm voxel 在 real camera noise 下可能失效。

5. **Communication overhead**：RoboChecker 的 closed-loop 反馈需要反复 call LLM 重新 plan。对 long-horizon task（如 Long Pipeline Delivery），这个 latency 可能 explode。未来需要 offline constraint validation 或 learned checker。

---

## 7. 相关工作链接

为了 build intuition，这里列几个关键 reference 的 link：

- **Diffusion Policy**（paper 的 base model）：https://diffusion-policy.cs.columbia.edu/
- **RoboTwin**（trajectory generation inspiration）：https://robotwin-benchmark.github.io/
- **ManiSkill 3**（simulator）：https://github.com/haosulab/ManiSkill
- **Code-as-Monitor**（constraint-aware visual programming 思路来源）：https://arxiv.org/abs/2412.04455
- **Depth Anything V2**（spatial constraint 用的 depth estimation）：https://depth-anything-v2.github.io/
- **PartNet-Mobility Dataset**（3D assets 来源）：https://sapien.ucsd.edu/browse
- **GPT-4o**（RoboBrain 和 RoboChecker 的 backbone）：https://openai.com/index/hello-gpt-4o/
- **π₀**（Physical Intelligence 的 VLA model，参考 multi-agent VLA 可能性）：https://www.physicalintelligence.company/blog/pi0
- **OpenVLA**（open-source VLA）：https://openvla.github.io/
- **3D Diffusion Policy**（point cloud 版 DP）：https://3d-diffusion-policy.github.io/

---

## 8. 总结：这篇 paper 的真正贡献

Andrej，从你的视角来看，这篇 paper 的核心贡献可以归纳为：

1. **Formalization**：把 multi-agent embodied collaboration 中模糊的 "constraint" 概念 formalize 成三个 orthogonal 维度，这是 **conceptual contribution**。
2. **Infrastructure**：构建了第一个 multi-agent manipulation benchmark with 11 tasks across 1-4 agents，这是 **infrastructure contribution**。
3. **Empirical finding**：揭示了当前 imitation learning 在 multi-agent + long-horizon 上的 fundamental limitation（4-agent 10% success，Long Pipeline Delivery 0%），这是 **empirical contribution**。
4. **Architectural insight**：Local View + Separate Policy > Global View + Shared Policy，对 future multi-agent VLA 设计有指导意义。

**Potential critique**：RoboBrain 和 RoboChecker 都依赖 GPT-4o，意味着 data generation 的 cost 很高（每个 task 的 expert demo 都需要大量 LLM call）。对 academic lab 来说，复现这个 pipeline 的 API cost 不低。未来如果开源 model 能替代 GPT-4o 做 constraint generation，会大幅降低门槛。

希望这个深度讲解帮你 build 起对 multi-agent embodied collaboration 的 intuition。如果你想深入讨论某个 specific aspect（比如 constraint interface 的实现细节、或 multi-agent VLA 的架构设计空间），我们可以继续聊。
