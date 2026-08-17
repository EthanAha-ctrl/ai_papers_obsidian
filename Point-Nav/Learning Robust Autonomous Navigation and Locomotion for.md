---
source_pdf: Learning Robust Autonomous Navigation and Locomotion for.pdf
paper_sha256: 1084638f40dd9ee600fd7bb8d6da9bb57c612828b95c7c0e2a2b05258c9653ea
processed_at: '2026-08-05T13:43:35-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 这篇paper在干嘛

一句话: **给一个有轮子又有腿的机器狗，教它在城市里自己送快递。**

这个机器狗是ANYmal quadruped加了4个actuated wheels。长这样: 四条腿，每条腿末端是个轮子。flat路面上轮子滚，遇到台阶腿就迈。

goal是在Zurich和Seville的real urban environment里，给一个GPS goal，自己走过去，8.3km级别mission，少人工干预。

参考: https://www.science.org/doi/10.1126/scirobotics.adi9641

---

## 为什么这事难

三件事搅在一起，每件单独都不简单:

### 难点1: 什么时候滚、什么时候迈

纯轮子robot上不了台阶。纯腿robot（比如ANYmal）慢且耗电——SubT challenge里ANYmal只能跑1小时，平均速度2.2 km/h，只有人类walking speed的一半。

wheeled-legged的好处是flat路面driving（fast + efficient），遇到台阶stepping。问题是怎么决定何时切换。

之前的方法要么用heuristic（"看到台阶就迈腿"），要么用pre-defined gait sequence（trot 2秒再drive 5秒这种）。这些都很脆。

trajectory optimization方法能算出复杂motion，但算力贵且要good initialization。

### 难点2: 高速robot的navigation

传统navigation = path planning + path following。path planning用sampling-based planner（RRT*那种），算一条path要几秒。

但这个robot能跑到5 m/s。几秒前算的path，几秒后环境可能变了（有行人、有动态obstacle）。而且传统planner用traversability cost map，这个map是symmetric的——上台阶和下台阶cost一样。实际上下台阶比上台阶容易多了。

### 难点3: 怎么把locomotion和navigation接起来

DARPA SubT challenge里Team Cerberus观察到的问题: robot经常在path中途停下来re-plan，或者走zigzag。module之间用heuristic通信，衔接处出问题。

---

## 他们的核心招数: 两层RL policy

### 整体结构

```
GPS goal
   ↓
Dijkstra on navigation graph → global path
   ↓
Waypoint selection (anchor pursuit, 3m lookahead)
   ↓
WP1, WP2 (两个waypoint, 间距5-20m)
   ↓
HLC (high-level controller) @ 10 Hz
   ↓ 输出 velocity command (vx, vy, ωz)
LLC (low-level controller) @ 50 Hz
   ↓ 输出 joint positions (12) + wheel velocities (4)
PD controllers → actuators
```

这个hierarchy的intuition像公司管理: 老板（HLC）10Hz想"往哪个方向走、走多快"，员工（LLC）50Hz想"具体怎么动腿动轮子实现这个velocity"。

为什么10Hz vs 50Hz? temporal abstraction。老板不需要每秒想50次方向，想太频繁反而探索效率低。后面ablation证实: 50Hz的HLC效果差很多。

参考HRL: https://arxiv.org/abs/1909.10618

### 为什么不end-to-end

他们也试了end-to-end（一个policy直接从sensor到joint command），效果很差（Table S1: 长path只有4.6% success rate）。

原因: locomotion和navigation是两个time scale的任务，混在一个MDP里reward shaping极难。一个reward要同时encourage"跟上velocity command"和"reach waypoint"，两个objective互相打架。

分开后: LLC专注velocity tracking（reward清楚），HLC专注waypoint reaching（reward也清楚），HLC的reward里加一个term让LLC的effort小（通过 $w_l \cdot (r_l + r_r)$ 实现），这就能让HLC知道什么样的command对LLC是舒服的。

---

## LLC怎么训的: Privileged Learning

### 核心idea

Privileged learning的intuition像老师带答案教学生。

**Teacher policy**: 在simulation里训练，能看到real world不可能直接测量的"作弊"信息:
- true foot contact state
- terrain normal at each foot  
- foot contact force
- true robot velocity
- gravity vector in base frame
- friction coefficient

这些privileged info让teacher学得快、学得好。Teacher是plain 3-layer MLP。

**Student policy**: 部署时用的。看不到privileged info，只能看noisy IMU、joint encoder、noisy height scan。

怎么训? 用DAgger imitation learning。Student是GRU-based RNN，从noisy observation的temporal sequence里infer出teacher看到的privileged info的equivalent representation。

Loss function:

$$\mathcal{L} := \mathbb{E}_{(s_t, o_t) \sim \mathcal{D}} \left\{ (\pi^{\text{teacher}}(s_t) - \pi^{\text{student}}(o_t, h_t))^2 \right\}$$

变量解释:
- $s_t$: teacher看到的full state（含privileged info）
- $o_t$: student看到的noisy observation，是 $s_t$ 去掉privileged info $x_t$ 后加noise的版本
- $h_t$: student GRU的hidden state
- $\mathcal{D}$: 数据分布
- $\pi^{\text{teacher}}, \pi^{\text{student}}$: 两个policy网络

直觉: student被迫从history里"猜"出terrain properties、contact state这些它看不到的东西。这个猜测能力就是部署时robust的来源。

参考DAgger: https://arxiv.org/abs/1011.0686
参考Lee et al. 2020: https://www.science.org/doi/10.1126/scirobotics.abb4743

### 一个关键改动: 去掉CPG

之前Miki et al. 2022的ANYmal controller，action space里包含CPG（Central Pattern Generator）参数——预定义gait的频率、duty factor、phase。这相当于"教robot怎么走"。

这篇完全去掉CPG。Action就是16D raw vector: 12个joint position target + 4个wheel velocity target。Gait完全从reward structure里emerge出来。

这个bet的收益: emergent behavior远超handcrafted gaits。实验里看到:
- 大discrete obstacle上: asymmetric creeping + driving混合gait
- 台阶/陡坡: trot（像point-foot quadruped）
- bumpy terrain: pure driving，腿当active suspension
- 下坡: 主动降低body height防翻
- 60cm台阶下: 前腿stretch down + 后腿crouch，前轮着地后roll forward
- 40cm block中间all wheels in air时: 用knees crawl forward until wheel regains contact

这些behavior没人写规则，全是从reward里自己长出来的。

### LLC的reward里几个关键的

**Velocity tracking** (最核心):
$$r_{lv} := \begin{cases} 2.0 \exp(-2.0 \cdot ||v_{xy}^{body}||^2), & \text{if } |v_{des}| < 0.05 \\ \exp(-2.0 ||v_{xy}^{body} - v_{des}||^2) + v_{des} \cdot v_{xy}^{body}, & \text{otherwise} \end{cases}$$

变量:
- $v_{xy}^{body} \in \mathbb{R}^2$: robot base在body frame下的horizontal velocity
- $v_{des} \in \mathbb{R}^2$: HLC下发的desired horizontal velocity

第一行: command接近0时，鼓励robot完全静止（reward peaked at 0 velocity）
第二行: Gaussian tracking term + linear shaping term。linear term的作用是tracking error大时reward仍然有incentive去减小error，避免Gaussian在远处gradient消失

**Joint torque penalty** (efficiency关键):
$$r_\tau := -\sum_{i \in joints} ||\tau_i||^2$$

$\tau_i$: joint $i$的torque

这个直接关联到heat loss（torque正比于current，current²正比于heat）。所以 $\sum \tau^2$ 小 = 省电 = COT低。

**Joint smoothness** (penalize target的1st & 2nd difference):
$$r_s = -c_k \sum_{i=1}^{12} ((q_{i,t,des} - q_{i,t-1,des})^2 + (q_{i,t,des} - 2q_{i,t-1,des} + q_{i,t-2,des})^2)$$

变量:
- $q_{i,t,des}$: joint $i$在time step $t$的target position
- $c_k$: scaling constant

第一项penalize target的jerk（1st difference）
第二项penalize target的acceleration（2nd difference）

这让joint trajectory平滑，减少hardware wear，也有助于sim-to-real。

**Body contact penalty** (除wheel外):
$$r_{bc} := -|I_{c,body} \setminus I_{c,wheel}|$$

- $I_{c,body}$: 所有body contact的index set
- $I_{c,wheel}$: wheel contact的index set
- $\setminus$: set difference
- $|\cdot|$: set cardinality

惩罚wheel以外的身体部位碰地（比如knee撞地），除非真的需要（前面那个40cm block用knee crawl是不得已）。

### Actuator modeling (sim-to-real的关键细节)

这帮人对actuator建模很认真，因为sim-to-real的gap很大程度来自actuator dynamics。

**Joint actuator** (SEA, Series Elastic Actuator): 用actuator network [Hwangbo 2019]学习inverse dynamics，从真实torque measurement训练神经网络。

**Wheel actuator** (pseudo-direct drive，没accurate torque measurement): 学一个mapping从velocity command + history到motor current:

$$I_t = f(\dot{\phi}_{target} | \dot{\phi}_{t-1}, \dot{\phi}_{t-2}, \cdots)$$

然后torque:
$$\tau_t = K_\tau \cdot GR \cdot I_t$$

变量:
- $I_t$: motor current at time $t$
- $K_\tau$: torque constant
- $GR$: gear ratio
- $\dot{\phi}_{target}$: commanded wheel velocity
- $\dot{\phi}_{t-1}, \dot{\phi}_{t-2}, \cdots$: wheel velocity history

再加friction:
$$\tau_{friction,C} = -C_1 \dot{\phi} \quad \text{(Coulomb)}$$
$$\tau_{friction,S} = -C_2 \text{sgn}(\dot{\phi}) \quad \text{(stick)}$$

$C_1, C_2$是randomized constants，放进privileged observation让teacher看见。部署时student不知道真实friction，但训练时见过各种friction，所以robust。

参考actuator network: https://www.science.org/doi/10.1126/scirobotics.aau5872

---

## HLC怎么训的: Mobility-aware Navigation

### 核心insight: HLC能"看到"LLC在想什么

这是这篇paper最clever的点之一。

HLC的observation包含:
1. **LLC的hidden state**（RNN的belief state）——这个latent captures了terrain properties、disturbances等信息
2. 3张时间戳的height scan（现在 + 0.1s前 + 0.2s前，处理dynamic obstacles）
3. **Position buffer**: 最近20个visited positions，每50cm采一个，覆盖10m
4. Waypoint history + action history

第1点是关键。HLC读LLC的hidden state，等于HLC知道LLC"感觉到的环境"是什么样。这比HLC自己从raw sensor重新infer terrain properties强多了——LLC已经做了一遍inference，HLC直接用结论。

这就是**mobility-aware**的本质: HLC知道LLC的capability和当前状态，所以能avoid commands that LLC can't track。这是0 collision的根源。

### Position buffer + exploration bonus

这个设计我觉得很巧妙。

Position buffer存: 每隔0.5m记录一个position + visitation time（停留多少个time step）。最近20个。

Exploration bonus reward:
$$r_{h,exp} := \sum_{P_{buf}} C(s_t, wp_t^1, p_{buf}^i)$$

$$C(p_{robot}, wp^1, p_{buf}^i) := \begin{cases} 0.0 & |p_{robot} - wp^1| < 0.75 \\ -n_{buf}^i & |p_{robot} - p_{buf}^i| < 1.0 \end{cases}$$

变量:
- $p_{robot}$: robot当前位置
- $wp^1$: 最近waypoint
- $p_{buf}^i$: buffer里第$i$个历史position
- $n_{buf}^i$: 在$p_{buf}^i$位置停留的time step数

含义: 
- 离waypoint很近（<0.75m）时不penalize
- 否则，如果robot在1m范围内某个**之前访问过的位置**，penalty正比于在那里停留的时间

intuition: 在地上撒面包屑标记去过的地方，如果robot在去过的地方停留，就惩罚。这鼓励robot主动explore新区域，避免local minima里打转。

Fig. 5A展示了这个behavior: robot遇到blocked path，倒退，沿墙探索，找到楼梯上去。没有position buffer的话（Fig. 7D-ii）robot就在原地打转出不来。

Ablation (Table S1): 去掉memory，10-20m path的SPL从0.689掉到0.526。

### Bounded action space: Beta distribution

不用Gaussian，用Beta distribution作为action distribution。

Bounds:
- $v_x \in [-1.0, 2.0]$ m/s（向前偏，align with camera朝向）
- $v_y \in [-0.75, 0.75]$ m/s
- $\omega_z \in [-1.25, 1.25]$ rad/s

为什么Beta distribution:
1. Hard limits，safety和interpretability
2. 容易regularize motion
3. 避免Gaussian在边界处gradient问题

Beta PDF: $f(x; \alpha, \beta) = x^{\alpha-1}(1-x)^{\beta-1}/B(\alpha, \beta)$
- $\alpha, \beta$: shape parameters
- $B(\alpha,\beta)$: Beta function
- 期望 $E[X] = \alpha/(\alpha+\beta)$
- $\alpha+\beta$ 越大variance越小

**Parameterization trick**: 不直接output $\alpha, \beta$，而是output mean和sum:
- $a_1, a_2 = \pi_{hi}(s_t)$
- $\alpha = a_1 \cdot a_2$
- $\beta = a_2 - a_1 \cdot a_2$
- mean $= \alpha/(\alpha+\beta) = a_1$
- $\alpha + \beta = a_2$ (控制variance)

这样mean直接是网络output，无需post-hoc计算。很实用的工程技巧。

参考: https://proceedings.mlr.press/v70/chou17a.html

### HLC的reward结构

HLC reward = $r_h + w_l \cdot (r_l + r_r)$

- $r_h$: 高层任务reward
- $r_l$: LLC task reward（让HLC感知LLC的effort）
- $r_r$: regularization
- $w_l$: scaling，让 $r_h$ 和 $r_l + r_r$ 的expected sum magnitude相当

这个设计很关键: HLC不仅要reach goal，还要让LLC的effort小。这就是**mobility-aware**的本质——HLC知道什么样的velocity command对LLC是舒服的。

具体 $r_h$ 包含:

**Sparse goal reward**:
$$r_{h,goal} := \begin{cases} 1.0 & |p_{robot} - wp^1| < 0.75 \\ 0.0 & \text{otherwise} \end{cases}$$

**Dense reward** (训练初期加速):
$$r_{h,dense} := \begin{cases} 1.0 & |e_{wp^1}| < 0.75 \\ \text{clip}(v \cdot \widehat{e_{wp^1}}, 0.0, v_{thres})/v_{thres} & \text{otherwise} \end{cases}$$

变量:
- $e_{wp^1} = p_{robot} - wp^1$ (位置误差向量)
- $\widehat{e_{wp^1}}$: 单位向量
- $v$: robot当前velocity
- $v_{thres} = 0.5$

reward正比于velocity在指向waypoint方向的投影。鼓励robot朝waypoint方向走。

**Near-goal stability**:
$$r_{h,stability} := \begin{cases} \exp(-2.0 ||v||^2) & |p_{robot} - wp^1| < 0.75 \\ 0.0 & \text{otherwise} \end{cases}$$

到达waypoint后鼓励robot停下来，避免到了还在乱动。

### HLC network architecture

- **Position history**: 1D CNN + max pooling (像PointNet)，permutation-invariant——因为position buffer里position顺序不影响语义
- **Height scan**: 3-layer 2D CNN + MLP
- **其他inputs + output**: plain MLP
- **Output layer**: Sigmoid（输出Beta参数）

HLC inference time: 平均0.34ms。对比baseline sampling planner超过1秒。这是1000× speedup，10Hz control frequency让robot能对dynamic obstacles响应。

参考PointNet: https://arxiv.org/abs/1612.00593

---

## 训练环境怎么造: WFC + Navigation Graph

### Wave Function Collapse (WFC)

WFC是procedural content generation算法，来自indie game开发。intuition类似Sudoku的constraint propagation。

输入: example tile map + tile adjacency规则
输出: 新的N×N tile map，遵守同样adjacency约束

这里定义3种tiles:
- **Stair**: 只能沿x轴连接到Floor
- **Floor 0**: flat
- **Floor 1**: flat with features

WFC生成terrain + connectivity graph。这个graph就是navigation graph。

参考WFC: https://github.com/mxgmn/WaveFunctionCollapse

### Navigation graph-guided training

这个设计从game development借鉴的（CryEngine, Unreal Engine的navigation system）。

每个episode:
1. WFC生成terrain + graph
2. 用Dijkstra在graph上random选两个node算shortest path
3. 沿path sample两个waypoints，lookahead distance均匀采样$[5, 20]$m
4. Path末端把last node复制两次作为两个waypoints

为什么这么设计:
- 提供solvable yet challenging problems（保证有feasible path）
- Agent学到的是graph-following behavior，不是random goal-seeking
- Path包含detours、tight gaps、sharp turns

这个设计对长path特别关键。Ablation (Table S1): no WFC，10-20m path的SPL从0.689暴跌到0.302。

### Dynamic obstacles training

White boxes，random number/position/velocity，向robot移动，speed在$[0.1, 0.5]$m/s。

部署时robot能应对pedestrians（Fig. 5E），因为training时见过dynamic obstacles。ZED 2i camera做human detection，在elevation map里给50cm半径加height offset，让HLC"看到"人。

### Terrain curriculum (filtering)

用genetic algorithm + Minimal Criterion filter terrain parameters:

$$f(c_\mathcal{T}, \pi) = \begin{cases} \mathbb{E}\{\nu(s_t | c_\mathcal{T})\} & \text{if } t_l < \mathbb{E}\{\nu(s_t | c_\mathcal{T})\} < t_h \\ 0.0 & \text{otherwise} \end{cases}$$

变量:
- $c_\mathcal{T}$: terrain parameter
- $\pi$: 当前policy
- $\nu(s_t | c_\mathcal{T})$: success score（velocity tracking error < 20% command speed时取1.0）
- $t_l, t_h$: success rate上下界

只保留success rate在$[t_l, t_h]$之间的terrain。太简单的浪费training，太难的学不会。Filtered terrain params复用到HLC training的tile generation。

参考Minimal Criterion coevolution: https://dl.acm.org/doi/10.1145/3377930.3389824

---

## 结果有多impressive

### Kilometer-scale missions

**Zurich Glattpark**: 
- Area: 245m × 345m
- Handheld laser scan: 90分钟
- 13个goal points
- Total distance: 8.3 km
- Manual intervention次数: 极少（只有3种情况: 有小孩、waypoint在untraversable区域、long corridor localization失败）

**Seville**: 另一个urban environment

### Efficiency metrics

Mechanical Cost of Transport:
$$COT_{mech} = \frac{\sum_{\text{all joints}} [\tau \dot{\theta}]^+}{mg |v_{xy}^b|}$$

变量:
- $[\cdot]^+ = \max(\cdot, 0)$: 只算正功
- $\tau$: joint torque
- $\dot{\theta}$: joint speed
- $mg$: robot总重量
- $|v_{xy}^b|$: base的horizontal speed

| Robot | Avg speed | COT_mech |
|-------|-----------|----------|
| Our wheeled-legged | 1.68 m/s | 0.16 |
| ANYmal (SubT) | ~0.56 m/s | ~0.34 |

- 3× speed
- 53% lower COT
- Driving mode下joint COT ≈ 0.01（几乎为零，因为腿static）
- $\sum \tau^2$ for leg joints低16%，despite重12kg且更快

Wheeled-legged的efficiency优势巨大，因为driving时joint几乎不耗能。

### vs Baseline (sampling-based planner)

Baseline: Wellhausen et al. 2021的sampling-based planner，SubT用的。

| Method | Failure rate | Collision rate | Planning time | Tracking error |
|--------|--------------|----------------|---------------|----------------|
| Ours | 0/10 | 0/10 | 0.34ms | 0.24 m/s |
| Baseline | high | high | >1s | 0.45 m/s |

Baseline两个failure mode:
1. **Occlusion handling**: sampling-based planner在occluded区域assumed traversable，实际是墙
2. **Tracking error**: 假设perfect path following，实际LLC有delay，远处waypoint导致overshoot collision

HLC的两个核心优势:
- 0 collision: HLC和LLC co-trained，HLC知道LLC的capability
- 1000× faster: 神经网络inference vs sampling planner

### Ablation (Table S1)

SPL (Success weighted by Path Length):
$$\text{SPL} = \frac{1}{N} \sum_{i=1}^N S_i \frac{l_i}{\max(p_i, l_i)}$$

变量:
- $N$: episodes数
- $S_i \in \{0,1\}$: episode $i$ success indicator
- $l_i$: shortest path distance
- $p_i$: 实际traversed path length

| Policy | 5-10m SPL | 10-20m SPL |
|--------|-----------|------------|
| Ours | 0.897 (90.1%) | 0.689 (76.3%) |
| No path sampling | 0.858 (84.0%) | 0.497 (55.9%) |
| No WFC | 0.865 (87.1%) | 0.302 (30.5%) |
| Memoryless | 0.873 (89.7%) | 0.526 (57.3%) |
| No temporal abstraction (50Hz) | 0.798 (82.3%) | 0.370 (39.7%) |
| End-to-end | 0.304 (31.8%) | 0.045 (4.6%) |

关键insights:
- **End-to-end最差**: locomotion + navigation一个MDP，reward shaping难
- **No temporal abstraction差**: 50Hz HLC探索效率低
- **No WFC对长path致命**: 缺diverse challenges，长path navigation不行
- **Memory对长path重要**: 短path可reactive解决，长path要exploration

---

## 几个我觉得最clever的设计

### 1. HLC读LLC的hidden state

这是mobility-aware的根源。传统HRL用latent sub-goal或pure velocity command通信，信息流单向。这里HLC能读LLC的hidden state，等于HLC知道LLC"感觉到的环境"和"当前状态"。所以HLC能avoid commands that LLC can't track。

这是0 collision的根源。也是direction-dependent traversability的来源——HLC通过LLC的hidden state理解了"上台阶难、下台阶易"。

### 2. Position buffer + exploration bonus

在地上撒面包屑的intuition。惩罚原地打转，鼓励探索。这是Fig. 5A exploratory behavior的来源。

比RNN implicit memory更interpretable——能直接看到robot记得哪些位置。Ablation证实: 去掉memory，长path SPL从0.689掉到0.526。

### 3. 去掉CPG让gait emerge

不预定义gait template，让gait从reward里emerge。这个bet的收益: emergent behavior远超handcrafted gaits。asymmetric creeping + driving混合、knee crawling、主动body lowering，这些没人写规则，全是从reward structure自己长出来的。

这是model-free RL的本质优势: 不受human prior限制。

### 4. WFC + navigation graph的training environment

从game development借鉴的。WFC生成solvable yet challenging terrain + graph，Dijkstra采样path，沿path采样waypoint。

这个设计对长path特别关键。Ablation: no WFC，长path SPL从0.689暴跌到0.302。

### 5. Privileged learning的teacher-student框架

Teacher用"作弊"信息训练，Student通过RNN从noisy history里infer出equivalent representation。比domain randomization更精细——domain randomization让policy对各种disturbance robust，privileged learning让policy主动infer disturbance是什么。

参考: https://www.science.org/doi/10.1126/scirobotics.abb4743

---

## Limitations和未来方向

paper自己列的:
1. **Semantic info缺失**: 主要用geometric info，没用semantic（pavement detection, visual traversability）
2. **有限FOV**: 3m前方感知，限制了最大speed deployment（hardware能6.2m/s但没法safely deploy）。未来: 去掉elevation mapping，直接用raw sensory stream
3. **Map creation需要human labor**: 90分钟handheld scan + 人工graph设计

我自己的观察:
- Long corridor localization failure（geometric degeneracy）没解决，靠recovery
- Position buffer只有20个positions，10m距离，更长detour可能overflow（Fig. 7D-i就出现stuck）
- Dynamic obstacles训练只用了0.1-0.5m/s的boxes，真实pedestrians更复杂
- 5G router和GPS antenna装了但没用上（localization靠pre-scanned point cloud）

---

## 整体评价

这是一篇system paper，不是纯algorithm paper。贡献在于integration:

- 把privileged learning, hierarchical RL, WFC, navigation graph, elevation mapping, SLAM, human detection等组装成一个working system
- 在real urban environment做了kilometer-scale validation
- 通过careful ablation证明每个component都necessary

技术novelty层面:
- HLC读LLC hidden state是新的（之前HRL多是用latent sub-goal或pure velocity command）
- Position buffer + exploration bonus是clever设计
- WFC用于robotics training environment是新的应用
- 去掉CPG让gait完全emerge，实验结果很好

但fundamental algorithm都是existing的: PPO, DAgger, Beta distribution actions, hierarchical decomposition。这篇的价值在于**showing that carefully combining these components can achieve kilometer-scale autonomous urban navigation**，这是legged robot领域的milestone。

ETH RSL这个组在legged robot RL这个方向已经积累了5年: Lee 2020 → Miki 2022 → 这篇2024。每篇都是前一篇的extension + integration，逐渐往real-world deployment靠近。这篇终于做到了city-scale autonomous mission，是这条路线的一个重要节点。

参考:
- PPO: https://arxiv.org/abs/1707.06347
- DAgger: https://arxiv.org/abs/1011.0686
- Beta distribution actions: https://proceedings.mlr.press/v70/chou17a.html
- PointNet: https://arxiv.org/abs/1612.00593
- HRL survey: https://arxiv.org/abs/1909.10618
- Open-ended learning: https://arxiv.org/abs/2107.12808
- SPL metric: https://arxiv.org/abs/1807.06757
- Wave Function Collapse: https://github.com/mxgmn/WaveFunctionCollapse
- Lee et al. 2020: https://www.science.org/doi/10.1126/scirobotics.abb4743
- Miki et al. 2022: https://www.science.org/doi/10.1126/scirobotics.abk2822
- Actuator network: https://www.science.org/doi/10.1126/scirobotics.aau5872

如果你对某个具体部分想更深入聊（比如DAgger的data collection细节、WFC的tile adjacency怎么定义、或者gait emergence的mechanism），告诉我，我可以再展开。

---

# Learning Robust Autonomous Navigation and Locomotion for Wheeled-Legged Robots 深度解读

这篇paper来自ETH Zurich的Robotic Systems Lab，发表在Science Robotics 2024，是Marco Hutter组的工作。Joonho Lee主导，延续他们之前ANYmal系列的工作（Science Robotics 2020 Lee et al. 和 2022 Miki et al.）。这次的核心是给quadruped加上actuated wheels，用hierarchical RL让它在city里做last-mile delivery级别的autonomous navigation。

参考链接:
- paper: https://www.science.org/doi/10.1126/scirobotics.adi9641
- ETH RSL lab: https://rsl.ethz.ch/
- Swiss-Mile Robotics: https://www.swiss-mile.com/
- 之前Lee et al. 2020: https://www.science.org/doi/10.1126/scirobotics.abb4743
- Miki et al. 2022: https://www.science.org/doi/10.1126/scirobotics.abk2822
- Wave Function Collapse: https://github.com/mxgmn/WaveFunctionCollapse
- Open3D SLAM: https://github.com/PRBonn/open3d_slam
- Elevation mapping GPU: https://github.com/leggedrobotics/elevation_mapping_cupy

---

## 1. Motivation和核心问题

### 1.1 为什么需要wheeled-legged robot

纯legged robot（如ANYmal）有两个硬伤:
- **Speed**: ANYmal平均2.2 km/h，只有人类walking speed的一半
- **Endurance**: DARPA SubT challenge里ANYmal只能operate 1小时左右

纯wheeled robot则在stairs和uneven terrain上无能为力。

Wheeled-legged robot结合两者优势: flat terrain上driving（高speed、低COT），遇到obstacles时stepping。关键挑战是**何时step、何时drive**——传统方法用heuristic或pre-defined gait sequence，不够robust。

### 1.2 三个核心Challenge

paper明确列出三个:

**Challenge 1: Hybrid locomotion**
- 之前的方法: heuristic决定step/drive [Bjelonic 2021]，或pre-defined gait sequence [Geilinger 2018]
- Trajectory optimization方法 [Bjelonic 2022] 算力贵，依赖good initialization
- 问题: speed和efficiency依赖gait和direction of motion，没有biological inspiration可借鉴

**Challenge 2: Navigation planning忽略dynamic robot characteristics**
- 传统sampling-based planner [Wellhausen 2023, Frey 2022]用explicit traversability cost map，没考虑whole-body state
- 没考虑tracking error随terrain/velocity/gait变化
- 速度高时planning要几秒，导致collision

**Challenge 3: System integration**
- Sub-modules孤立开发，靠heuristic inter-module communication
- Team Cerberus在SubT里观察到的path midway pause、zigzag motion等病态行为

---

## 2. 系统架构总览

### 2.1 双层policy结构

```
Global Path (graph nodes) 
    ↓
Waypoint Selection (anchor pursuit, 3m lookahead)
    ↓
WP1, WP2 (两个waypoint)
    ↓
[HLC: High-Level Controller] @ 10 Hz
    ↓ velocity command (vx, vy, ωz)
[LLC: Low-Level Controller] @ 50 Hz
    ↓ joint positions (12) + wheel velocities (4)
PD controllers → actuators
```

关键设计选择: **explicit sub-goal**而非learned latent sub-goal。他们尝试过end-to-end和latent sub-goal，但explicit velocity command更好——因为:
1. 允许independent development of controllers
2. 符合legged robotics的传统practice
3. Pre-trained LLC可以复用到不同high-level applications

### 2.2 Robot hardware

- 基础: ANYmal quadruped + 4个actuated wheels
- Sensors: 3个LiDAR (1个Velodyne VLP-16在顶部用于localization，2个Robosense RS-Bpearl在前后用于elevation mapping)，1个ZED 2i RGB stereo camera，5G router，GPS antenna，delivery box
- Wheel hardware limit: max 6.3 m/s (joint speed 45 rad/s × wheel radius 0.14 m)
- 实测peak: 5.0 m/s on flat terrain

---

## 3. LLC: Low-Level Locomotion Controller

### 3.1 Privileged Learning框架

LLC用两阶段训练:

**Stage 1: Teacher policy training**
- 用PPO在simulation训练
- Input: **privileged observation** $x_t$ + normal observation
- Privileged info包括: noiseless joint states, foot contact state, terrain normal at each foot, foot contact force, robot velocity, gravity vector in base frame
- 这种teacher在simulation里"作弊"——用了real world不可能直接测量的信息
- Teacher是plain 3-layer MLP

**Stage 2: Student policy training (DAgger)**
- 用imitation learning从teacher学
- Student input: noisy IMU measurements, joint states, noisy height scans
- 关键: 用GRU-based RNN encoder处理temporal sequence，让student能从noisy history里infer出teacher privileged info的equivalent representation
- Loss function:

$$\mathcal{L} := \mathbb{E}_{(s_t, o_t) \sim \mathcal{D}} \left\{ (\pi^{\text{teacher}}(s_t) - \pi^{\text{student}}(o_t, h_t))^2 \right\}$$

变量说明:
- $s_t$: teacher看到的full state（含privileged info）
- $o_t$: student看到的noisy observation，是 $s_t \setminus x_t$ 的noisy version
- $h_t$: student GRU的hidden state
- $\mathcal{D}$: 数据分布

### 3.2 关键修改: 去掉CPG

相比之前Miki et al. [2022]，最大的改动是**移除CPG (Central Pattern Generator)**。Miki的方法里action space包含CPG参数控制gait频率和phase。这里完全去掉，让gait从reward structure里emerge出来。

这是model-free RL的核心bet: 让agent自己决定何时抬腿、何时roll wheel，而不是predefine一个gait template。

### 3.3 Observation和Action space

**Observation** (student, deployment时):
- Exteroceptive: 围绕4个wheels的circular height scan pattern
- Proprioceptive: 直接用raw IMU (linear acceleration + angular velocity)，不用state estimator! joint angles, joint velocities
- Command: 3D vector (target vx, vy, yaw rate)

**Action**: 16D vector
- 12 joint position commands (PD controller的target)
- 4 wheel velocity commands

**为什么直接用raw IMU**: 传统state estimator在wheel slippage或discrete height change时high error。Movie S4里展示了state estimator error导致的locomotion failure。直接用raw measurement减少heuristic filtering，消除对准确orientation/velocity estimate的依赖。

### 3.4 LLC的Reward function

LLC reward = $r_l + r_r$，其中 $r_l$ 是task reward，$r_r$ 是regularization。

**Linear velocity tracking**:
$$r_{lv} := \begin{cases} 2.0 \exp(-2.0 \cdot ||v_{xy}^{body}||^2), & \text{if } |v_{des}| < 0.05 \\ \exp(-2.0 ||v_{xy}^{body} - v_{des}||^2) + v_{des} \cdot v_{xy}^{body}, & \text{otherwise} \end{cases}$$

变量:
- $v_{xy}^{body} \in \mathbb{R}^2$: robot base在body frame下的horizontal velocity
- $v_{des} \in \mathbb{R}^2$: HLC下发的desired horizontal velocity
- 第一行: 当command接近0时，鼓励robot完全静止
- 第二行: Gaussian tracking term + linear shaping term，linear term确保reward在tracking error大时仍然有incentive去减小它

**Yaw rate tracking**:
$$r_{av} := \exp(-2.0 (\omega_z^{body} - \omega_{des})^2)$$

**Base motion penalty** (penalize uncommanded directions):
$$r_{bm} := -1.25 (v_z^{body})^2 - 0.4 |\omega_x^{body}| - 0.4 |\omega_y^{B}|$$

**Orientation** (keep body level):
$$r_{ori} = \arccos(R_b(3,3))^2$$

这里 $R_b(3,3)$ 是body rotation matrix的第(3,3)元素，即body z轴在世界z轴方向的投影。$\arccos$ 给出tilt angle，平方后penalize大tilt。

**Base height** (target 0.55m):
$$r_h = \max(0.0, |h_{base} - 0.55| - 0.05)$$

tolerance 0.05m，dead zone。

**Regularization**里关键的几个:

Joint torque penalty (proportional to electric current → heat loss):
$$r_\tau := -\sum_{i \in joints} ||\tau_i||^2$$

这个直接关联到COT的evaluation: $\sum \tau^2$ 越小，heat loss越小，efficiency越高。

**Joint smoothness** (penalize 1st & 2nd order finite difference):
$$r_s = -c_k \sum_{i=1}^{12} ((q_{i,t,des} - q_{i,t-1,des})^2 + (q_{i,t,des} - 2q_{i,t-1,des} + q_{i,t-2,des})^2)$$

变量:
- $q_{i,t,des}$: joint $i$ 在time step $t$ 的target position
- 第一项penalize jerk（实际是1st difference of targets）
- 第二项penalize acceleration of targets（2nd difference）
- $c_k$: scaling constant

**Body contact penalty** (only non-wheel contacts):
$$r_{bc} := -|I_{c,body} \setminus I_{c,wheel}|$$

$I_{c,body}$: 所有body contact的index set
$I_{c,wheel}$: wheel contact的index set
$\setminus$: set difference
$|\cdot|$: cardinality

意思: penalize wheel之外的任何身体部位接触地面（比如knee撞地）。

**Knee joint constraint** (prevent knee flipping):
$$r_{jc,i} = \begin{cases} -(q_i - q_{i,th})^2, & \text{if } q_i > q_{i,th} \\ 0.0 & \text{otherwise} \end{cases}$$

只对knee joint设threshold $q_{i,th}$。

### 3.5 Emergent gaits观察

实验中观察到的emergent behaviors（Fig. 6）:

| Terrain | Emergent gait |
|---------|---------------|
| Large discrete obstacle | Asymmetric: creeping + driving混合 |
| Stairs / steep uphill | Trot (像point-foot quadruped) |
| Bumpy terrain (height ~ wheel radius) | Pure driving, active suspension |
| Downhill | 降低body height，driving |
| 60cm step down | 前腿stretch down + 后腿crouch，前轮着地后roll forward |
| 40cm block (中间all wheels in air) | 用knees crawl forward until wheel regains contact |

**非对称traversability**: 下台阶比上台阶能traverse更高。这直接反映在Fig. 5C-D，HLC会主动避开高台阶的ascent但允许descent。传统cost-map方法用symmetric traversability（与motion direction无关），HLC理解的是**direction-dependent traversability**。

### 3.6 Actuator modeling (sim-to-real关键)

**Joint actuator**: Series Elastic Actuator (SEA)，用actuator network [Hwangbo 2019]学习inverse dynamics，从torque measurement训练。

**Wheel actuator** (pseudo-direct drive，没有accurate torque measurement):
学一个mapping从velocity command和history到motor current:
$$I_t = f(\dot{\phi}_{target} | \dot{\phi}_{t-1}, \dot{\phi}_{t-2}, \cdots)$$

然后torque:
$$\tau_t = K_\tau \cdot GR \cdot I_t$$

变量:
- $I_t$: motor current at time $t$
- $K_\tau$: torque constant
- $GR$: gear ratio
- $\dot{\phi}_{target}$: commanded wheel velocity
- $\dot{\phi}_{t-1}, \dot{\phi}_{t-2}, \cdots$: wheel velocity history

再加friction model:
$$\tau_t = K_\tau \cdot GR \cdot I_t + \tau_{friction}$$

其中:
$$\tau_{friction,C} = -C_1 \dot{\phi} \quad \text{(Coulomb)}$$
$$\tau_{friction,S} = -C_2 \text{sgn}(\dot{\phi}) \quad \text{(stick)}$$

$C_1, C_2$ 是randomized constants，放入privileged observation。这是domain randomization的思路——让policy在训练时见过各种friction特性，部署时robust。

---

## 4. HLC: High-Level Navigation Controller

### 4.1 核心insight: mobility-aware

传统navigation = path planning + path following + inter-module communication。这里HLC**替代了这三个module**，直接从observation到velocity command。

关键insight: HLC能感知LLC的capability。怎么做到的? 通过**直接读LLC的hidden state**。

HLC observation包含:
1. **LLC的hidden state** (RNN belief state)——这个latent captures了terrain properties, disturbances等信息，是LLC对环境的internal representation
2. 3张时间戳的height scan around robot (现在 + 0.1s前 + 0.2s前，处理dynamic obstacles)
3. **Position buffer**: 最近20个visited positions，每50cm采一个，覆盖10m距离
4. Waypoint history + action history (smoother trajectories)

### 4.2 Position buffer和exploration bonus

这是这篇paper一个很巧妙的设计。

Position buffer存: 每隔0.5m记录一个position + visitation time（停留多少个time step）。

Exploration bonus reward:
$$r_{h,exp} := \sum_{P_{buf}} C(s_t, wp_t^1, p_{buf}^i)$$

$$C(p_{robot}, wp^1, p_{buf}^i) := \begin{cases} 0.0 & |p_{robot} - wp^1| < 0.75 \\ -n_{buf}^i & |p_{robot} - p_{buf}^i| < 1.0 \end{cases}$$

变量:
- $p_{robot}$: robot当前位置
- $wp^1$: 最近waypoint
- $p_{buf}^i$: buffer里第$i$个历史position
- $n_{buf}^i$: 在$p_{buf}^i$位置停留的time step数

含义:
- 如果robot离waypoint很近（<0.75m），不penalize
- 否则，如果robot在1m范围内某个**之前访问过的位置**，penalty正比于在那里停留的时间

这鼓励agent: 不要在同一个地方反复转圈（local minima），要主动explore新区域。这就是Fig. 5A的exploratory behavior的来源——遇到blocked path会倒退沿墙走找出口。

Ablation (Table S1): 去掉memory，10-20m path的SPL从0.689掉到0.526。

### 4.3 Bounded action space: Beta distribution

不用Gaussian，用Beta distribution作为action distribution [Chou 2017]。

Bounds:
- $v_x \in [-1.0, 2.0]$ m/s (向前偏，align with camera朝向)
- $v_y \in [-0.75, 0.75]$ m/s
- $\omega_z \in [-1.25, 1.25]$ rad/s

**为什么Beta**: 
1. Hard limits，safety和interpretability
2. 容易regularize motion
3. 避免Gaussian在边界处的gradient问题

**Parameterization trick**: 不直接output $\alpha, \beta$，而是output mean和sum:
- $a_1, a_2 = \pi_{hi}(s_t)$
- $\alpha = a_1 \cdot a_2$
- $\beta = a_2 - a_1 \cdot a_2$
- mean $= \alpha/(\alpha+\beta) = a_1$
- $\alpha + \beta = a_2$ (控制variance)

这样mean直接是网络output，无需post-hoc计算。

Beta PDF: $f(x; \alpha, \beta) = x^{\alpha-1}(1-x)^{\beta-1}/B(\alpha, \beta)$
- $B(\alpha,\beta)$: Beta function（normalization）
- 期望 $E[X] = \alpha/(\alpha+\beta)$
- $\alpha+\beta$ 越大variance越小

### 4.4 Waypoint reward structure

**Sparse goal reward**:
$$r_{h,goal} := \begin{cases} 1.0 & |p_{robot} - wp^1| < 0.75 \\ 0.0 & \text{otherwise} \end{cases}$$

**Dense reward** (training初期):
$$r_{h,dense} := \begin{cases} 1.0 & |e_{wp^1}| < 0.75 \\ \text{clip}(v \cdot \widehat{e_{wp^1}}, 0.0, v_{thres})/v_{thres} & \text{otherwise} \end{cases}$$

变量:
- $e_{wp^1} = p_{robot} - wp^1$ (位置误差向量)
- $\widehat{e_{wp^1}}$: 单位向量
- $v$: robot当前velocity
- $v_{thres} = 0.5$
- $\text{clip}$: 截断到$[0, v_{thres}]$

意思: reward正比于velocity在指向waypoint方向的投影，clamp在$[0,1]$。鼓励robot朝waypoint方向走。

**Near-goal stability**:
$$r_{h,stability} := \begin{cases} \exp(-2.0 ||v||^2) & |p_{robot} - wp^1| < 0.75 \\ 0.0 & \text{otherwise} \end{cases}$$

到达waypoint后鼓励robot停下来。

### 4.5 HLC reward structure总结

HLC训练reward = $r_h + w_l \cdot (r_l + r_r)$

- $r_h$: 高层任务reward（goal reaching, dense, exploration, stability）
- $r_l$: LLC task reward（让HLC感知LLC的effort）
- $r_r$: regularization
- $w_l$: scaling，让 $r_h$ 和 $r_l + r_r$ 的expected sum magnitude相当

这个设计很关键: HLC不仅要reach goal，还要让LLC的effort小。这就是**mobility-aware**的本质——HLC知道什么样的velocity command对LLC是舒服的。

### 4.6 Network architecture

- **Position history**: 1D CNN + max pooling (像PointNet [Qi 2017])，permutation-invariant
- **Height scan**: 3-layer 2D CNN + MLP
- **其他inputs + output**: plain MLP
- **Output layer**: Sigmoid（输出Beta参数）

HLC inference time: 平均0.34ms（vs baseline sampling planner超过1秒）。这是巨大优势——10Hz control frequency，可对dynamic obstacles响应。

---

## 5. Training Environment: WFC + Navigation Graph

### 5.1 Wave Function Collapse (WFC)

WFC是procedural content generation算法 [Gumin 2016]，来自indie game开发。原理:
- 输入: example tile map + tile adjacency rules
- 输出: 新的N×N tile map，遵守同样的adjacency约束
- 类似Sudoku的constraint propagation

这里定义3种tiles:
- **Stair**: 只能沿x轴连接到Floor
- **Floor 0**: flat
- **Floor 1**: flat with features

WFC生成terrain + connectivity graph。

### 5.2 Navigation graph-guided training

这个设计是从game development借鉴的（CryEngine, Unreal Engine的navigation system）。

每个episode:
1. WFC生成terrain + graph
2. 用Dijkstra在graph上random选两个node算shortest path
3. 沿path sample两个waypoints，lookahead distance均匀采样$[5, 20]$m
4. Path末端把last node复制两次作为两个waypoints

为什么这么设计:
- 提供solvable yet challenging problems（保证有feasible path）
- Agent学到的是graph-following behavior，不是random goal-seeking
- Path包含detours、tight gaps、sharp turns

### 5.3 Dynamic obstacles training

White boxes，random number/position/velocity，向robot移动，speed在$[0.1, 0.5]$m/s。这是为了training时见过dynamic obstacles，部署时能应对pedestrians。

### 5.4 Terrain curriculum (filtering)

用genetic algorithm + Minimal Criterion [Brant 2017] filter terrain parameters:

$$f(c_\mathcal{T}, \pi) = \begin{cases} \mathbb{E}\{\nu(s_t | c_\mathcal{T})\} & \text{if } t_l < \mathbb{E}\{\nu(s_t | c_\mathcal{T})\} < t_h \\ 0.0 & \text{otherwise} \end{cases}$$

变量:
- $c_\mathcal{T}$: terrain parameter
- $\pi$: 当前policy
- $\nu(s_t | c_\mathcal{T})$: success score（这里定义为velocity tracking error < 20% command speed时取1.0）
- $t_l, t_h$: success rate上下界

只保留success rate在$[t_l, t_h]$之间的terrain，太简单太难都丢掉。这避免policy wasted在impossible terrains或stuck在trivial ones。Filtered terrain params复用到HLC training的tile generation。

---

## 6. Experimental Results

### 6.1 Kilometer-scale missions

**Zurich Glattpark**: 
- Area: 245m × 345m
- Handheld laser scan: 90分钟
- 13个goal points
- Total distance: 8.3 km
- Manual intervention次数: 极少

**Seville**: 另一个urban environment

### 6.2 Efficiency metrics

Mechanical Cost of Transport:
$$COT_{mech} = \frac{\sum_{\text{all joints}} [\tau \dot{\theta}]^+}{mg |v_{xy}^b|}$$

变量:
- $[\cdot]^+$: $\max(\cdot, 0)$，只算正功
- $\tau$: joint torque
- $\dot{\theta}$: joint speed
- $mg$: robot总重量
- $|v_{xy}^b|$: base的horizontal speed

**结果对比**:
| Robot | Avg speed | COT_mech |
|-------|-----------|----------|
| Our wheeled-legged | 1.68 m/s | 0.16 |
| ANYmal (SubT) | ~0.56 m/s | ~0.34 |

- 3× speed
- 53% lower COT
- Driving mode下joint COT ≈ 0.01（几乎为零，因为腿static）
- 相比ANYmal: wheels exert 1.2× total mechanical power while achieving 3.4× speed
- $\sum \tau^2$ for leg joints低16%，despite重12kg且更快

这数据非常impressive。Wheeled-legged的efficiency优势巨大，因为driving时joint几乎不耗能。

### 6.3 Comparison with baseline (Fig. 7)

Baseline: Wellhausen et al. [2021]的sampling-based planner，SubT用的。

Setup: point-goal navigation across complex obstacle (stairs + wall)，10 trials。

| Method | Failure rate | Collision rate | Planning time | Avg tracking error |
|--------|--------------|----------------|---------------|---------------------|
| Ours (full) | 0/10 | 0/10 | 0.34ms | 0.24 m/s |
| Ours (no memory) | high | - | 0.34ms | - |
| Baseline | high | high | up to >1s | 0.45 m/s |

Baseline的两个failure mode:
1. **Occlusion handling**: sampling-based planner在occluded区域assumed traversable，但实际上是墙
2. **Tracking error**: 假设perfect path following，实际LLC有delay和error，远处waypoint导致overshoot collision

HLC两个核心优势:
- 0 collision: HLC和LLC co-trained，HLC知道LLC的capability
- 1000× faster: 神经网络inference vs sampling planner

### 6.4 Ablation studies (Table S1)

SPL (Success weighted by Path Length):
$$\text{SPL} = \frac{1}{N} \sum_{i=1}^N S_i \frac{l_i}{\max(p_i, l_i)}$$

变量:
- $N$: episodes数
- $S_i \in \{0,1\}$: episode $i$ success indicator
- $l_i$: shortest path distance
- $p_i$: 实际traversed path length

| Policy | 5-10m SPL | 10-20m SPL |
|--------|-----------|------------|
| Ours | 0.897 (90.1%) | 0.689 (76.3%) |
| No path sampling | 0.858 (84.0%) | 0.497 (55.9%) |
| No WFC | 0.865 (87.1%) | 0.302 (30.5%) |
| Memoryless | 0.873 (89.7%) | 0.526 (57.3%) |
| No temporal abstraction (50Hz) | 0.798 (82.3%) | 0.370 (39.7%) |
| End-to-end | 0.304 (31.8%) | 0.045 (4.6%) |

关键insights:
- **End-to-end最差**: locomotion + navigation一个MDP，reward shaping难
- **No temporal abstraction差**: 50Hz HLC探索效率低
- **No WFC对长path致命**: 缺diverse challenges，长path navigation不行
- **Memory对长path重要**: 短path可reactive解决，长path要exploration

---

## 7. 几个有意思的设计细节

### 7.1 Anchor pursuit (waypoint selection)

传统pure pursuit在path上interpolate固定距离sub-waypoints。这里改为:
- 如果next node < 3m: 直接用node作waypoint
- 否则: project robot到path上，选3m lookahead的waypoint

Anchor points（global graph的nodes）必须approached，robot不能take shortcut绕过它们。但sub-waypoints随robot移动，给obstacle avoidance留freedom。

### 7.2 Localization

用Open3D SLAM [Jelavic 2022]: ICP-based，但用IMU + joint encoder odometry作prior，scan-to-map matching而非scan-to-scan。在高speed和geometrically degenerate环境（长走廊）更robust。

Pre-scanned point cloud只用localization，不用于navigation。Navigation完全靠onboard elevation mapping。

### 7.3 Human detection

ZED 2i SDK的Spatial Object Detection，detect人后给elevation map在50cm半径内加height offset。HLC training时见过dynamic obstacles，能保持距离overtake。

### 7.4 Asymmetric traversability (Fig. 5D, 6C)

HLC学到direction-dependent traversability。下台阶可以更高，上台阶低就避让。这是cost-map方法做不到的——cost map是symmetric的，与direction无关。

这是mobility-aware navigation的核心体现: HLC通过读LLC的hidden state，理解了LLC的asymmetric capability。

---

## 8. Limitations和未来方向

paper自己列的:
1. **Semantic info缺失**: 主要用geometric info，没用semantic（pavement detection, visual traversability）
2. **有限FOV**: 3m前方感知，限制了最大speed deployment（虽然hardware能6.2m/s，但没法safely deploy）。未来方向: 去掉elevation mapping，直接用raw sensory stream
3. **Map creation需要human labor**: 90分钟handheld scan + 人工graph设计

补充我自己的观察:
- Long corridor localization failure（geometric degeneracy）没解决，靠recovery
- Position buffer只有20个positions，10m距离，更长detour可能overflow（Fig. 7D-i就出现stuck）
- Dynamic obstacles训练只用了0.1-0.5m/s的boxes，真实pedestrians更复杂

---

## 9. 直觉性总结

这篇工作的核心insight我归纳为三点:

**Insight 1: Privileged learning让sim-to-real可信**
Teacher用"作弊"信息（true contact, friction, terrain normal）训练，Student通过RNN从noisy history里reconstruct等效representation。这比domain randomization更精细——domain randomization让policy对各种disturbance robust，privileged learning让policy主动infer disturbance是什么。

**Insight 2: Hierarchical decomposition的temporal abstraction**
HLC 10Hz, LLC 50Hz。这不是engineering convenience，而是temporal abstraction——HLC在更慢的时间尺度上做"high-level决策"，让exploration更efficient。Ablation证实: 50Hz HLC效果差。这和HRL literature [Nachum 2018]的结论一致。

**Insight 3: Mobility-aware的关键是policy间通信**
传统HRL用latent sub-goal通信，这里用explicit velocity command。但更重要的是HLC能读LLC的hidden state——这是双向信息流。HLC知道LLC"在想什么"，所以能avoid commands that LLC can't track。这是0 collision的根源。

**Insight 4: Environment design比algorithm重要**
WFC + navigation graph + terrain curriculum的组合，比单纯random goal + random obstacle强一个数量级。这和open-ended learning [DeepMind 2021]的insight一致: agent能力受限于training environment的complexity。

**Insight 5: 去掉inductive bias (CPG)的代价和收益**
去掉CPG是bet——让gait完全从reward emerge。代价: 训练可能更难，没有biological prior。收益: emergent behavior远超handcrafted gaits（asymmetric creeping + driving混合，knee crawling，主动body lowering）。这是model-free RL的本质优势。

---

## 10. 一些延伸阅读和思考

### 10.1 跟相关工作的关系

- **Miki et al. 2022 (Science Robotics)**: 这篇的直接前身，perceptive locomotion with privileged learning + RNN encoder。这篇把method扩展到wheeled-legged，并加HLC。
- **Lee et al. 2020 (Science Robotics)**: 最早的terrain curriculum + privileged learning框架。所有后续工作的基础。
- **Rudin et al. 2022 (CoRL)**: Massively parallel RL training。这篇LLC的training hyperparameters参考了这个。
- **Hoeller et al. 2021**: State representation + navigation in dynamic environment。Memory mechanism的inspiration。
- **Ji et al. 2022**: Concurrent training of control policy + state estimator。这篇直接用raw IMU的insight来源。

### 10.2 跟DeepMind相关工作的对比

- **Open-Ended Learning [DeepMind 2021]**: Automatic task generation让agent general capability。这篇的WFC + terrain curriculum是简化版。
- **Emergence of maps in memories [Wijmans 2023]**: Memory在navigation中的作用。这篇用explicit position buffer替代RNN implicit memory，更interpretable。

### 10.3 跟autonomous driving的对比

- **End-to-end vs modular**: 这里也讨论了，modular + hierarchical胜过end-to-end。
- **Beta distribution actions [Chou 2017]**: 来自autonomous driving RL，bounded action对safety关键。

### 10.4 我对这篇工作的整体评价

这是一篇system paper，不是纯algorithm paper。它的贡献在于integration:
- 把privileged learning, hierarchical RL, WFC, navigation graph, elevation mapping, SLAM, human detection等组装成一个working system
- 在real urban environment做了kilometer-scale validation
- 通过careful ablation证明每个component都necessary

技术novelty层面:
- HLC读LLC hidden state是新的（之前HRL多是用latent sub-goal或者pure velocity command）
- Position buffer + exploration bonus是clever设计
- WFC用于robotics training environment是新的应用

但fundamental algorithm都是existing的: PPO, DAgger, Beta distribution actions, hierarchical decomposition。这篇的价值在于**showing that carefully combining these components can achieve kilometer-scale autonomous urban navigation**，这是legged robot领域的milestone。

参考:
- PPO: https://arxiv.org/abs/1707.06347
- DAgger: https://arxiv.org/abs/1011.0686
- Beta distribution actions: https://proceedings.mlr.press/v70/chou17a.html
- PointNet: https://arxiv.org/abs/1612.00593
- HRL survey: https://arxiv.org/abs/1909.10618
- Open-ended learning: https://arxiv.org/abs/2107.12808
- SPL metric: https://arxiv.org/abs/1807.06757
- Wave Function Collapse: https://github.com/mxgmn/WaveFunctionCollapse

如果你对某个具体部分（比如privileged learning的DAgger细节，或者WFC如何定义tile adjacency）想深入讨论，可以告诉我，我可以再展开。这篇supplementary materials还有很多细节（actuator network训练、不同action space的ablation、gait selection的RealNVP action space等）。
