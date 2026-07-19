---
source_pdf: BEHAVIOR-1K A Human-Centered, Embodied AI.pdf
paper_sha256: ab4033f3e8a8187ad2cfdf11f1979d33996ecce1e81fa3d62af33a92bbb17a2b
processed_at: '2026-07-18T14:55:14-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark — 深度技术讲解

## 1. 高层直觉: 为什么需要这个 Benchmark

在 embodied AI 领域存在一个 long-standing 的 **diversity-realism tradeoff**:

- **Diversity-first benchmarks** (VirtualHome, ALFRED): 几百个 activities, 几十个 scenes, 但是 physical realism 弱 — 物体只是 "graph nodes", 没有 realistic physics, grasping 只是 state transition
- **Realism-first benchmarks** (Habitat 2.0, ManiSkill, RoboSuite): 物理/控制真实, 但是只有 1-7 个 tasks, 容易 overfit

BEHAVIOR-1K 同时突破这两个 axis: **1000 个 activities + 全物理仿真 (rigid, deformable, fluid, cloth, thermal)**, 并且关键在于 activity selection 来自 **1,461 个真实人类** 的 preference survey, 而不是 researcher 凭直觉挑选。

核心 thesis: 如果你想 build general-purpose home robot, 你的 benchmark 应该 **from human needs, for human needs**, 而不是 from researcher imagination。

项目网站: https://behavior.stanford.edu

---

## 2. Survey Methodology 的细节直觉

### 2.1 Activity Sourcing Pipeline

Activity pool 来自两个互补 source:

1. **Time-Use Surveys** (540 activities): 
   - American Time Use Survey (ATUS, https://www.bls.gov/tus/)
   - Harmonised European Time Use Surveys (https://ec.europa.eu/eurostat/web/time-use-surveys)
   - Multinational Time Use Study (https://www.timeuse.org/mtus)
   
   这些 surveys 记录人们 daily 怎么 spend time — 频繁但耗时的 activities

2. **WikiHow articles** (180,000+ → filtered): https://www.wikihow.com
   - 涵盖不 frequent 但 important 的 activities (e.g., "clean rusty garden tools")
   - 人类 seek guidance 的活动, 即使不 frequent

### 2.2 Feasibility Filtering

合并后用以下 criteria filter (因为 OMNIGIBSON 的 simulation constraints):

| Filtering Principle | Example Filtered Out |
|---|---|
| Physics/chemistry 不支持 | SteamingClothes, MakingSoap |
| 涉及 media creation/consumption | ReadingABook |
| > 1 day real time | DryingSeedsOvernight |
| 需要 non-visual perception | SweeteningFood |
| BDDL 表达不了的 fine geometry | SettingUpANativityScene |
| Branded items | SprayingWindex |

最终 2,090 activities 进入 survey。

### 2.3 Survey 执行

- **Platform**: Amazon Mechanical Turk (https://www.mturk.com)
- **1,461 个 respondents**, demographics 接近 MTurk population
- 每个 activity 50 个 responses
- **10-point Likert scale** (1=less beneficial, 10=most beneficial)
- 每个 respondent 回答 50 个 activity questions + 4 个 attention check repeats
- 重复 responses 差距 > 2 → 视为 failed attention check; 2+ failures → 拒绝

### 2.4 Pilot Experiments

Question wording 三选一 (robot / assistant / automation): 30 个 pairwise T-tests + 10 个 ANOVAs → 没有统计显著差异 → 选 "robot"

Format 二选一 (10-point Likert vs. best-worst scaling): Kendall's tau 显示强相关 → 选 resource-efficient 的 Likert

### 2.5 Survey 结果直觉

- **Gini index = 0.158** (high statistical dispersion) — 人类 preferences 高度 diverse, 不集中
- 平均得分 5.16, 范围 1.9-9.3
- Top scores: laborious tasks (scrubbing bathroom floor, cleaning bathtub)
- Bottom scores: recreational (game-play)
- ~200 cleaning activities + ~200 cooking activities
- 最终 BEHAVIOR-1K = top 909 from survey + 91 from BEHAVIOR-100 = 1000 activities

**直觉**: 人类 delegate 给 robot 的本质是 **physical effort + low enjoyment** 的组合 — 这给 benchmark 设定了清晰的 difficulty profile (长 horizon + 复杂 manipulation)。

---

## 3. BEHAVIOR-1K DATASET: Knowledge Base 的架构

### 3.1 Object Space 构建 Pipeline

```
1000 activities 
   → 5 WikiHow articles/activity (Upwork crowdworkers)
   → Noun phrase extraction (FlairNLP chunking model)
   → Manual filtering into tangible objects
   → 2,964 leaf synsets (1,538 WordNet + 1,426 custom)
```

### 3.2 Property Annotation 的混合策略

Table A.1 列出 33 个 object properties, 用 4 种 annotation 方法:

| Method | Properties Examples | 决策 criteria |
|---|---|---|
| **GPT-3** | cookable, sliceable, fireSource, liquid, particleRemover | Hamming distance + FPR < 10% vs human ground truth |
| **Human** | breakable, flammable, openable, toggleable | GPT-3 不够 reliable 的 |
| **Manual** | assembleable, cloth, fillable, meltable, rigidBody, rope | Simulator-implementation sensitive |
| **Programmatic** | deformable, diceable, drapeable, foldable, freezable, heatable, substance, unfoldable | 从其他 properties 推导 |

**关键直觉**: GPT-3 当成 "weak human annotator", 只在它已经够准的 properties 上用。Quality 验证: 5 个 expert annotators 抽检 → accuracy > 96.8%, F1 > 91%, FDR/FPR 在 2-3%。

### 3.3 Property Parameters (object-property tuples)

只标 binary property 不够 — 比如 apple 和 chicken 都是 cookable, 但 cook temperature 不同。所以 BEHAVIOR-1K 加了 continuous parameters:

```
cookTemperature(crab.n.05) = 63°C
cookTemperature(squash.n.02) = 58°C
cookTemperature(meatball.n.01) = 63°C
cookTemperature(chicken_leg.n.01) = 74°C

heatGenerated(toaster_oven.n.01) = 204°C
heatGenerated(ember.n.01) = 1093°C
heatGenerated(hand_blower.n.01) = 45°C
```

这些 parameters 让 OMNIGIBSON 能 simulate realistic thermal dynamics。

### 3.4 Transition Rules (跨过 PhysX 的物理限制)

很多 real-world processes (blending, baking, removing rust) 物理引擎 simulate 不了。Transition Machine = **modular rule-based bypass**:

```
Input:  strawberry.n.01 + ice.n.01 + lemon_juice.n.01 + agave.n.01
Machine: blender.n.01 (toggledOn)
Output: smoothie.n.01
```

```
Input:  paint.n.01 covering object
Required: particleRemover saturated with solvent.n.01 OR acetone.n.01
Output: paint removed
```

```
Input:  rust.n.01 covering object  
Required: emery_paper.n.01 OR whetstone.n.01
Output: rust removed
```

**直觉**: 这是 simulation 的 "structured hallucination" — 不真 simulate chemistry, 但产生 visually consistent state transition。是经典 symbolic-physical hybrid 思路。

---

## 4. BDDL: Predicate Logic 的三个新特性

### 4.1 Substance Representation

传统 PDDL 把每个 object 当 distinct instance。但 water 这种 substance 没有清晰 instance boundary — 如果有两个 bottle 的 orange juice (`oj_1`, `oj_2`) 各装半瓶, 倒进 glass 后, goal `filled(glass, oj_1)` 永远不会满足, 因为 particles 混了。

**Solution**: 一个 definition 里同一个 substance 最多一个 instance。如果想表达 "多瓶 juice", 用 **container synset** (e.g., `orange_juice__bottle.n.01`) — 有 bottle.n.01 的 properties 但没有 liquid particles 的开销。

### 4.2 Three-Valued Predicates

问题: `not open` 在 PDDL 里是 binary 切换, 但 crack window 一点点就从 `not open` 跳到 `open`, 不符合人类直觉。

**Solution**: 对某些 predicates 用 **paired Booleans**:
- `filled` / `empty` (不是 `not filled`)
- `open` / `closed` (不是 `not open`)
- `folded` / `unfolded` (不是 `not folded`)

Annotator 看到 `filled`, negate 就 swap 成 `empty`。底层用 De Morgan's Law 把 negations 推到 atomic formulae 后再 swap。

### 4.3 Composition/Decomposition via `future` Predicate

PDDL 假设 `:objects` 全程 persist。但 baking sugar cookies 需要创造 `sugar_cookie.n.01`, 这个 object 在 `:init` 里不存在。

**Solution**: 在 `:init` 用 `(future sugar_cookie.n.01_1)` 标记。约束:
- 所有 `future` objects 必须出现在 `:objects`
- 不能出现在 `:init` 其他 literals 里

灵感来自 Minecraft construction-planning (Wichlacz et al., 2019, https://ojs.aaai.org/index.php/ICAPS/article/view/18114)。

### 4.4 BDDL Example 解析 (Listing 1: BakingSugarCookies)

```lisp
(:objects
   flour.n.01_1 - flour.n.01
   ...
   sugar_cookie.n.01_1 ... sugar_cookie.n.01_6 - sugar_cookie.n.01
   oven.n.01_1 - oven.n.01
   ...
)
(:init
   (filled flour__sack.n.01_1 flour.n.01_1)   ; sack 装着 flour
   (ontop flour__sack.n.01_1 countertop.n.01_1)
   ...
   (inroom oven.n.01_1 kitchen)                ; scene-relevant constraint
   (future sugar_cookie.n.01_4)                ; 还没存在, 由 transition machine 创建
   ...
)
(:goal
   (and
      (real ?sugar_cookie.n.01_1)              ; "real" = 物理存在 (已 transition 完成)
      ...
      (forall (?sugar_cookie.n.01 - sugar_cookie.n.01)
         (and
            (cooked ?sugar_cookie.n.01)        ; temperature 达到 cook 阈值
            (ontop ?sugar_cookie.n.01 ?cookie_sheet.n.01_1)
         )
      )
   )
)
```

**直觉**: 这把 "long-horizon cooking task" 表达成 logical goal, 而不是 trajectory imitation。Agent 可以用任何 sequence 达成 — 这是 symbolic task specification 的强大之处。

---

## 5. OMNIGIBSON 架构深度解析

### 5.1 Software Stack

```
OMNIGIBSON
   ├── BDDL parser / sampler
   ├── Extended object state system (Temperature, SoakedLevel, etc.)
   ├── Transition Machine
   ├── Action primitive interface
   └── Sensor simulation (RGB, Depth, LiDAR, Segmentation)
       
Nvidia Omniverse (ray-traced rendering, RTX)
Nvidia PhysX 5 (rigid body, articulation, deformable, fluid particles)
```

### 5.2 Extended Object States: 选择性更新

Table A.5 显示 property → required states 的 mapping:

| Object Property | Required Extended States |
|---|---|
| cookable | MaxTemperature, Temperature |
| overcookable | MaxTemperature, Temperature |
| freezable | Temperature |
| flammable | Temperature |
| heatable | Temperature |
| soakable | SoakedLevel |
| toggleable | ToggledState |
| sliceable | SlicedState |
| breakable | BrokenState |
| heatSource / fireSource / coldSource / waterSource | ToggledState |

**关键性能直觉**: 不是所有 object 都 track 所有 state — 比如 table 不 track Temperature。这是 **property-driven selective state tracking**, 类似 ECS (Entity-Component-System) 的设计。

### 5.3 Logical Predicates 的两层 API

OMNIGIBSON 对每个 predicate 同时实现:
- **Checking function**: physical state → Boolean (for goal evaluation)
- **Sampling function**: Boolean → physical state (for initialization)

Examples (Table A.8, A.9):

```
OnTopOf(o1, o2): 
  o2 ∈ InSameNegativeVerticalAxisObjs(o1) 
  ∧ o2 ∉ InSamePositiveVerticalAxisObjs(o1) 
  ∧ InContactWith(o1, o2)
  
  Sampling: ray-cast from above o2, sample valid pose, ensure collision-free

Cooked(o):
  T_cooked ≤ T_o^max < T_burnt
  (历史最大温度在 cook 阈值和 burnt 阈值之间)
  
  Sampling True:  T_o^max ← max(T_o^max, T_cooked)
  Sampling False: T_o^max ← min(T_o^max, T_cooked - 1)

Open(o):  joint state q > 0.05(q_upper - q_lower) + q_lower
  (相对 threshold, 不是绝对)

Sliced(o): 接触 SlicingTool 且 contact force > F_sliced (default 10N)
  Sampling True: 用两个 half objects 替换 whole, 继承 Temperature 等 states

Soaked(o, l): SoakedLevel w for liquid l ≥ w_soaked (default 50 particles)
Filled(o, l): ContainerVolume 内 l particle 数 ≥ 50% volume

Covered(o, s): CoveredLevel c for substance s ≥ c_covered (default 50 particles)
```

### 5.4 关键 Object Model Properties (Table A.6)

每个 object model 需要的 semantic annotations:

- **HeatSourceLink**: 虚拟 link 产生 heat (不 collide)
- **FireSourceLink**: 产生 fire
- **CleaningToolLink**: 接触 dirt particles 才能 clean
- **WaterSourceLink / WaterSinkLink**: 水源/水槽位置
- **TogglingLink**: 接触时切换 toggled state
- **SlicingLink**: 接触时切片 (需 force threshold)
- **RelevantJoints** (for openable): 哪些 joints 决定 "open" — 比如 microwave 的 door 但不包括 buttons
- **AttachmentKeypoints** (for assembleable): 连接 key points
- **ClothKeypoints** (for foldable): 判断 folded = key points 距离足够近
- **ContainerVolume** (for fillable): 内部容积几何
- **StableOrientations**: 用 3D geometry library 计算 (放置物体时用)

**直觉**: 这是 **semantic-physical bridge** — 把 human-readable predicates (`open`, `filled`, `folded`) 映射到 specific physical structures。

### 5.5 Visual Realism 评估

60 个 AMT participants 对 5 个 simulator 各 50 张 1280x720 图片排序:

- **OMNIGIBSON**: 3.20 ± ? (最高, ray-traced)
- AI2-Thor: 1.69
- TDW: 1.65
- iGibson 2.0: 1.74
- Habitat 2.0: 1.73

Ray-tracing 让 OMNIGIBSON 在 transparency, reflection, lighting 上显著领先。

### 5.6 Performance Benchmark (Table A.10)

| Eval Condition | Rs_int (81 objs) | house_single_floor (621 objs) |
|---|---|---|
| Full Feature Set | 24 SPS | 11 SPS |
| - Fluid and Cloth | 58 | 26 |
| - Object State Update | 77 | 55 |
| - Robot | 90 | 60 |

**关键直觉**: 
- Cloth/fluid 是最大 performance hit (3x slowdown)
- Object state update 中等 hit
- Robot presence 是最小 hit (因为 robot 是 fixed cost)
- 600 objects 还能跑 11 SPS — 可用但不够快做大规模 RL
- 当前 iGibson 2.0 是 ~100 SPS, OMNIGIBSON 是 ~60 FPS (因为 ray-tracing)

Setup: Intel i7-10700K, RTX 3080, $t_a = t_s = 1/60$ s

---

## 6. Experiments: Baselines 和技术细节

### 6.1 三个 paradigmatic activities

| Activity | Scene | Required Skills | Min Primitive Steps |
|---|---|---|---|
| **CollectTrash** | mockup_apt | rigid body manipulation | 16 |
| **StoreDecoration** | Rs_int | articulated object (drawer) | ~10 |
| **CleanTable** | restaurant_hotel | cloth + fluid (dip, wipe) | 6 |

**直觉**: 这些是 BEHAVIOR-1K 里相对简单的 activities, 但 horizon 已经远超 typical RL benchmark (MetaWorld 等通常 50-200 low-level steps; 这里需要 hundreds-thousands low-level commands)。

### 6.2 三个 Baselines

#### RL-VMC (Visuomotor Control via SAC)

SAC objective (Eq. 2):

$$\pi^* = \arg\max_\pi \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty}\gamma^t\left(R(s_t, a_t, s_{t+1}) + \alpha H(\pi(\cdot|s_t))\right)\right]$$

变量解释:
- $\pi^*$: optimal stochastic policy
- $\tau$: trajectory $s_0, a_0, s_1, a_1, \ldots$ sampled from $\pi$
- $\gamma \in [0,1)$: discount factor (0.99 in paper), exponentially down-weights future rewards
- $R(s_t, a_t, s_{t+1})$: reward function (在 BEHAVIOR-1K 里是 sparse BDDL goal signal)
- $\alpha$: entropy temperature coefficient — 控制 exploration vs exploitation 的 tradeoff
- $H(\pi(\cdot|s_t)) = -\int \pi(a|s_t)\log\pi(a|s_t) da$: policy entropy at state $s_t$

**直觉**: SAC = expected return + entropy bonus, 鼓励 stochastic, off-policy (用 replay buffer), sample efficient vs PPO 但需要 continuous action space。

#### RL-Prim. (Discrete action primitives via PPO)

PPO-Clip objective (Eq. 1):

$$C^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

变量解释:
- $\theta$: policy network parameters
- $\hat{\mathbb{E}}_t$: empirical expectation over collected timesteps
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$: importance sampling ratio between new and behavior policy
- $\hat{A}_t$: GAE (Generalized Advantage Estimation) estimate, 用 $\gamma = 0.99$, $\lambda_{GAE} = 0.99$ (here)
- $\epsilon = 0.2$: clipping range, 限制 policy update per step 防止 destructive large updates
- $\min$: 取 clipped 和 unclipped 的较小值 → pessimistic bound

**直觉**: PPO 通过 clipping 实现 trust region, 不用 KL constraint 计算, 比 TRPO 简单。On-policy, sample efficiency 低但 stable。

#### RL-Prim.Hist. (PPO + 3-step history)

加 3 帧 history observations, 帮 disambiguate aliased states。

### 6.3 Network Architecture (Fig A.11)

**RL-Prim. architecture**:
```
128x128x3 RGB → Conv-ReLU-MaxPool-Flatten → feature
[proprioception: grasping or not] → MLP
              ↓
            MLP (128-dim)
              ↓
       ┌──────┴──────┐
   Value head     Action head (discrete over primitives)
```

**RL-VMC architecture**:
```
128x128x3 RGB → Conv-ReLU-MaxPool-Flatten → feature
[proprioception] → MLP
              ↓
            Actor (continuous actions)
            Critic (Q value)
       Target Critic (soft update, τ=0.005)
```

### 6.4 Action Primitives 设计 (Fig A.10)

6 个 primitives, 都用 sampling-based motion planner (RRT-Connect, Kuffner & LaValle 2000, https://ieeexplore.ieee.org/document/844730):

| Primitive | Composition |
|---|---|
| **navigate** | Robot base 全身 collision-free trajectory to a location |
| **pick** | untuck arm → pre-grasp pose → Cartesian line down (interrupt on contact) → close gripper → retract up → tuck arm |
| **place** | untuck → pre-place pose → open gripper → retract to untuck |
| **push** | untuck → pre-push → Cartesian line down → line toward robot → retract up → tuck |
| **dip** | untuck → pre-dip → Cartesian down into liquid → up → retract |
| **wipe** | untuck → pre-wipe → Cartesian down → horizontal left-right → toward robot → up → tuck |

**Assistive grasping**: 当 gripper fingers 都 contact object 时, 创建 rigid joint (类似 StickyMitten)。这是 simplification — Table 4 显示去掉这个 simplification 性能暴跌。

### 6.5 主结果 (Table 2)

| Method | Primitives | History | StoreDec | CollectTrash | CleanTable |
|---|---|---|---|---|---|
| RL-VMC | ✗ | ✗ | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 |
| RL-Prim. | ✓ | ✗ | 0.48±0.06 | 0.42±0.02 | 0.77±0.08 |
| RL-Prim.Hist. | ✓ | ✓ | 0.55±0.05 | 0.63±0.03 | 0.88±0.02 |

**直觉 chain**:
1. **RL-VMC 0%**: sparse reward + extreme long-horizon → credit assignment 问题 + deep exploration 难 + vanishing gradient on recurrent signals。End-to-end visual → joint control 跨度太大
2. **RL-Prim. 巨大提升**: action abstraction 把 horizon 从 ~1000 steps 压到 ~10-16 steps, PPO 能学。CleanTable 最好 (6 steps), CollectTrash 最差 (16 steps)
3. **History 再提升**: CollectTrash 上从 42% → 63% (+50% relative), 因为 "looking at bin" 是 aliased state — 不知道哪些 locations 已经清理过。Memory 解 partial observability

### 6.6 Efficiency Metrics (Table 3)

CollectTrash 上的 efficiency:

| Method | Dist. Nav. [m] | Sim. Time [s] | Kin. Dis. [m] |
|---|---|---|---|
| RL-VMC | 27.58±5.95 | 16.67±0.00 | 0.00±0.00 |
| RL-Prim. | 17.98±2.35 | 13.95±5.14 | 12.34±5.01 |
| RL-Prim.Hist. | 15.33±2.70 | 12.48±3.68 | 10.82±3.90 |

- **Kin. Dis.** = kinematic disarrangement = 物体因 robot motion 被意外 displacement 总距离
- RL-VMC 数字虚高 — 因为没成功, robot 一直 random walk
- History 帮减少 redundant actions, 提升所有 efficiency 指标 ~10-15%

### 6.7 Ablation: Simplification 的影响 (Table 4)

| Phys. Realism | Grasping Full Motion | StoreDec | CollectTrash | CleanTable |
|---|---|---|---|---|
| ✓ | ✓ | 0.00 | 0.00 | 0.00 |
| ✗ | ✓ | 0.46 | 0.36 | 0.73 |
| ✗ | ✗ (training setting) | 0.48 | 0.42 | 0.77 |

**关键直觉**:
- **Fully physics-based grasping**: 性能 100% 跌到 0% — grasping 是 manipulation 的 bottleneck
- **Full motion trajectory execution**: 几乎没影响 — free-space motion planning 在 evaluation 时基本能找到 path, 所以训练时直接 set final config 是合理的加速 trick

**重要 caveat**: 在 simulation 里训练时用 assistive grasp 简化, 评估时切到 fully physics — policy 完全 collapse。这是 **sim-to-sim gap**, 不只是 sim-to-real gap。

---

## 7. Sim-to-Real 实验深度分析

### 7.1 Setup

- **Real scene**: mockup apartment in lab (bedroom + living + dining)
- **Digital twin**: 用手机扫描 + 投影 texture + 替换 objects 为 BEHAVIOR-1K 3D models
- **Robot**: Tiago++ (PAL Robotics) — omnidirectional base, 2x 7-DoF arms, 1-DoF prismatic torso, SICK LiDAR x2, ASUS Xtion RGB-D
- **Perception**: YOLOv3 (https://arxiv.org/abs/1804.02767) for object detection, depth map for 3D centroid
- **Navigation**: particle filter (https://mitpress.mit.edu/9780262201629/probabilistic-robotics/) with 2 LiDAR + map
- **Sim-to-real technique**: SECANT-style image-based data augmentation (https://arxiv.org/abs/2106.09678)

### 7.2 三个 Conditions 的结果

| Setting | Runs | Success |
|---|---|---|
| Simulation | 50 | ~40% |
| Real + Optimal Policy | 27 | ~22% |
| Real + Trained Policy (RL-Prim.) | 26 | 0% |

### 7.3 Failure Source Analysis (Fig 5 right)

**Simulation failures**:
- 主要是 visual policy 选错 primitive
- Place primitive 的 stochasticity + motion planner noise
- 没有 grasping failures (用了 assistive grasp)

**Real-world with trained policy**:
- 44% perception errors (visual policy 选错 action — sim-real visual gap)
- ~40% grasping failures (real physics grasp 难)
- 剩余是 motion planning + navigation noise

**Real-world with optimal policy**:
- Grasping 仍 ~40% (actuation issue)
- 剩余是 perception — 主要是 YOLOv3 detection errors
- 没有 policy errors (因为 optimal)

### 7.4 Sim-Real Gap 的 Sources (Fig A.12)

**RGB gap**: 
- Real camera poor dynamic range
- Wooden texture mismatch
- Surface reflectivity 差异
- Lighting 条件变化

**Depth gap**:
- Reflective surfaces 在 real depth sensor 上失效
- "Shadow" effects (projected light mechanism artifacts)
- Sim 里 depth 干净

**LiDAR gap**: 最小 — LiDAR 是 active sensor, robust to lighting

### 7.5 Compounding Error 的洞察

**Real-world 特有**: navigation inaccuracies 导致 robot base placement 不利 → 后续 manipulation 失败。Simulation 里假设 perfect localization, 这种 compounding 不存在。

**直觉**: 这是 hierarchical system 的 classic 问题 — 上一层的 error 会 cascade 到下一层。在 sim 里分层 isolate 简化训练, 在 real 里需要 joint consideration。

---

## 8. 与 Related Benchmarks 的精确定位

Table 1 的关键 dimension 对比:

| Benchmark | Activities | Scenes | Objects | Fluid | Cloth | Deformable | Thermal |
|---|---|---|---|---|---|---|---|
| **BEHAVIOR-1K** | **1000** | 50 | 9318 | ✓ | ✓ | ✓ | ✓ |
| BEHAVIOR-100 | 100 | 15 | 300+ | ✗ | ✗ | ✗ | partial |
| Habitat 2.0 HAB | 3 | 1 | ~162 | ✗ | ✗ | ✗ | ✗ |
| ALFRED | 7 | 120 rooms | 92+YCB | ✗ | ✗ | ✗ | ✗ |
| VirtualHome | 549 | 1 | UNK | ✗ | ✗ | ✗ | ✗ |
| SAPIEN ManiSkill | 5 | 1 | 162 | ✗ | ✗ | partial | ✗ |
| SoftGym | 1 task family | 1 | few | ✓ | ✓ | ✓ | ✗ |
| RFUniverse | 1 | 1 | 73+ | ✓ | ✓ | ✓ | ✗ |
| TDW Transport | 1 | 1 | 112 | ✗ | ✗ | ✗ | ✗ |

**关键观察**: 
- **没有其他 benchmark 同时支持 fluid + cloth + deformable + thermal + 1000 activities**
- Habitat 2.0 只能支持 23% 的 BEHAVIOR-1K activities (Fig 4 left)
- Fluids 和 flexible materials 是 top object synsets (Fig 4 right) — 没 OMNIGIBSON 这类 features 根本 simulate 不了 BEHAVIOR-1K 的 50%+ activities

References:
- BEHAVIOR-100: https://arxiv.org/abs/2105.10067
- Habitat 2.0: https://arxiv.org/abs/2106.05375
- ALFRED: https://arxiv.org/abs/1912.01734
- VirtualHome: https://arxiv.org/abs/1806.07011
- SAPIEN: https://arxiv.org/abs/2003.08515
- SoftGym: https://arxiv.org/abs/2011.07215
- RFUniverse: https://arxiv.org/abs/2202.00199

---

## 9. 关键公式补充

### 9.1 GAE (Generalized Advantage Estimation, 用于 PPO 的 $\hat{A}_t$)

虽然 paper 没展开, 但 PPO 的 advantage estimate 用 GAE (https://arxiv.org/abs/1506.02438):

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD error, $\lambda \in [0,1]$ 是 bias-variance tradeoff 参数 (paper 里 $\lambda_{GAE} = 0.99$)。

### 9.2 Soft Update (SAC 的 target critic)

$$\theta_{target} \leftarrow \tau\theta + (1-\tau)\theta_{target}$$

$\tau = 0.005$ (paper Table A.11) — small polyak averaging rate 让 target 缓慢跟随 current。

### 9.3 Success Score Q (BEHAVIOR-100 metric)

Table A.13 报告 Q — 综合考虑 success + efficiency 的 scalar。From BEHAVIOR-100 paper:

$$Q = \text{success\_rate} \cdot \text{efficiency\_factor}$$

具体 efficiency factor 用 distance / time / disarrangement 的 normalized scores。

### 9.4 Probability Ratio in PPO

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

如果 $\pi_\theta$ 偏离 $\pi_{\theta_{old}}$ 太多, $r_t$ 远离 1, clipping 在 $[1-\epsilon, 1+\epsilon]$ 限制 update magnitude。

---

## 10. 局限和未来方向

### 10.1 Paper 承认的 Limitations

1. **Rendering speed**: 60 FPS (vs iGibson 2.0 的 100 FPS) — ray-tracing 牺牲 speed 换 realism
2. **No human simulation**: 不 simulate humans 在场 — 真实 home 机器人需要 human-robot interaction
3. **Sim2real gap**: 还没 incorporate sensor noise models, actuation noise models

### 10.2 隐含 Limitations (我从 paper 推断)

1. **Grasping bottleneck**: Assistive grasp 是大 simplification, 真实 fully physics grasp 让所有 baseline 跌到 0% — 说明当前 manipulation RL 在 realistic grasping 上根本没法 work, 需要 new methods (tactile sensing, force control, diffusion policies)
2. **Long-horizon planning**: 即使 action primitives + history 才 50-88% success 在 6-16 步任务上 — 真正 BEHAVIOR-1K 的复杂 cooking 任务 (50+ steps) 还远超当前能力
3. **Activity coverage**: Experiments 只测了 3 个最简单 activity (都是 manipulation 类), 1000 里的 997 个 untested — covering cooking, cleaning with chemicals, multi-room activities 都是开放挑战
4. **Symbolic-physical gap**: BDDL 给了 logical goal, 但 agent 怎么从 visual observation infer logical state (e.g., "is this cooked?") 是 huge perception challenge
5. **Reward sparsity**: 全靠 BDDL goal signal, 没 intermediate reward — 1000 activities 上设计 dense reward 不可行, 但 sparse reward 训练 intractable。Hierarchical RL / intrinsic motivation / LLM-guided reward shaping 是 promising direction

### 10.3 与当前 frontier 的 connection (2026 年视角)

- **Vision-Language-Action models (VLAs)**: RT-2, Open-X-Embodiment, $\pi_0$ 等 — 可以直接 map (image, instruction) → action sequence, 跳过 explicit primitive 设计
- **Diffusion policies**: Diffusion Policy, $\pi_0$ 在 manipulation 上比 RL-VMC 强很多 — 在 BEHAVIOR-1K 上重做 baseline 会很有意思
- **World models**: DreamerV3, Genie — 可以在 latent space 里 plan long-horizon
- **LLM planners + VLA executors**: SayCan, VoxPoser, Code as Policies — LLM 选 primitive, VLA 执行 — 接近 RL-Prim. 但用 foundation model 替换 learned policy
- **Foundation world models for sim**: 1X World Model, Genie 2 — 可能 future 不用 PhysX 而用 learned simulator
- **Sim-to-real via domain adaptation**: Dexterity from Touch (OpenAI), ALOHA sim2real — BEHAVIOR-1K 当前 sim2real 0% success 说明需要更强 domain randomization + tactile modalities

---

## 11. 你可能想知道的 Intuition 总结

1. **为什么这个 benchmark 难**: 因为它把 human preference (tedious → robot) 直接映射到 task difficulty (long horizon + complex manipulation)。短 horizon 的 RL玩具 (MetaWorld) 解决不了真人类需求。

2. **为什么 RL-VMC 0%**: Sparse reward + 1000+ steps horizon + visuomotor direct mapping = credit assignment 和 exploration 双重 fail。End-to-end 从 pixel 到 torque 在长 horizon 上是死路。

3. **为什么 RL-Prim. 能 work**: Action abstraction 把 horizon 压缩一个数量级, 让 PPO 在 16-step MDP 上能学。但 primitive 设计本身需要 domain knowledge — 没解决 general problem, 只是 reframe。

4. **为什么 history 关键**: 因为 embodied activities 有 aliased states — same observation 可以对应 different internal state (e.g., "bin in view, but what I already cleaned?")。Partial observability 是 embodied AI vs Atari 的核心差别。

5. **为什么 sim-to-real 0%**: 多重 gap 叠加:
   - Visual gap (camera dynamic range, texture) → perception fail (44%)
   - Grasping gap (assistive sim grasp vs real contact-rich grasp) → actuation fail (40%)
   - Compounding from navigation noise
   当前 sim 不 model 这些就强行 transfer 等于盲目。需要 (a) better sim realism, (b) domain randomization, (c) system identification, (d) tactile sensing。

6. **BEHAVIOR-1K vs Habitat 2.0 的本质区别**: Habitat 强调 photo-realistic scene scans + navigation + 简单 rearrangement; BEHAVIOR-1K 强调 **task diversity grounded in human preference + complex state transitions (cooking, cleaning, fluid, cloth)**。两者互补, 但 BEHAVIOR-1K 更接近 general home robot。

7. **BDDL 的哲学**: 不是 "trajectory imitation" 而是 "goal specification" — agent 可以用任何 path 达到 goal state。这是 symbolic AI 和 RL 的 sweet spot: symbolic 描述 what, RL 学习 how。Plan-and-execute paradigm 的 modern 版本。

8. **OMNIGIBSON 的本质创新**: 把 extended object states (Temperature, SoakedLevel, ToggledState, CoveredLevel) 加进 physics simulator。PhysX 5 给 rigid/deformable/fluid 的 physics, OMNIGIBSON 加 "semantic physics" — 物体根据 heat/cold source 更新 temperature, 根据 contact 更新 sliced/broken state。这是 hybrid simulator 的设计模式。

9. **Transition Machine 的意义**: 是 "structured hallucination" — 不真 simulate chemistry (blending strawberries → smoothie), 但用 rule-based state transition 在 trigger conditions 满足时 fire。这是 symbolic-physical hybrid 的 pragmatic 解 — 完整 physics simulate 太贵, 完全 symbolic 不真实, transition machine 是中间路径。

10. **未来方向直觉**: 如果有 VLA + BEHAVIOR-1K, 可以把 1000 activities 的 BDDL goal 当 instruction, 让 VLA 从 (image, BDDL goal) → primitive selection 或 end-effector action。这等于在 BEHAVIOR-1K 上做 RT-2-style training, 数据规模 1000 tasks × 50 scenes × 9000 objects 已经接近有意义规模。

---

## 12. 进一步阅读建议

- **BEHAVIOR-100** (前作): https://arxiv.org/abs/2105.10067
- **iGibson 2.0**: https://arxiv.org/abs/2108.07228
- **SAC**: https://arxiv.org/abs/1801.01290
- **PPO**: https://arxiv.org/abs/1707.06347
- **GAE**: https://arxiv.org/abs/1506.02438
- **Habitat 2.0**: https://arxiv.org/abs/2106.05375
- **SoftGym**: https://arxiv.org/abs/2011.07215
- **SAPIEN**: https://arxiv.org/abs/2003.08515
- **VirtualHome**: https://arxiv.org/abs/1806.07011
- **ALFRED**: https://arxiv.org/abs/1912.01734
- **TDW Transport Challenge**: https://arxiv.org/abs/2103.14025
- **Nvidia Omniverse**: https://www.nvidia.com/en-us/omniverse/
- **Nvidia PhysX 5**: https://developer.nvidia.com/physx-sdk
- **RRT-Connect**: https://ieeexplore.ieee.org/document/844730
- **SECANT (sim2real augmentation)**: https://arxiv.org/abs/2106.09678
- **YOLOv3**: https://arxiv.org/abs/1804.02767
- **Probabilistic Robotics (Thrun)**: https://mitpress.mit.edu/9780262201629/probabilistic-robotics/
- **BEHAVIOR 项目网站**: https://behavior.stanford.edu
- **BEHAVIOR-1K GitHub**: https://github.com/StanfordVL/behavior
- **OMNIGIBSON GitHub**: https://github.com/StanfordVL/OmniGibson

如果你想 deep dive 某个具体方向 (BDDL compiler 实现 / extended state 更新算法 / sim-to-real 数据 pipeline / 用 VLA 在 BEHAVIOR-1K 上做 baseline), 告诉我, 我可以再展开。
