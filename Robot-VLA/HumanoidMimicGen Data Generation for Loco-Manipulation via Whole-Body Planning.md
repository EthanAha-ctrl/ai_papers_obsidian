---
source_pdf: HumanoidMimicGen Data Generation for Loco-Manipulation via Whole-Body
  Planning.pdf
paper_sha256: 4cc79994d93a90f65a1e66c4c273b0b000a2bad85d375b1487dc5be4a5d6b653
processed_at: '2026-08-05T08:14:23-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HumanoidMimicGen: 人话版

## 一、The Story 一句话版

让humanoid robot学会"边走边抓东西"，最难的不是algorithm，是**数据**。NVIDIA这帮人发现：你给机器人看1个人类demo，它能自动生成1000条类似但场景不同的训练数据，policy success rate从26%飙到89%。

Project page: https://humanoidmimicgen.github.io/

## 二、Why this is hard? Build your intuition

### 2.1 先建立baseline intuition

想象你训一个static机械臂policy。你给它100条demo，policy差不多能学会pick一个object。Easy。

现在换成humanoid。它需要先**走过去**，再**弯腰**，再**伸手**，再**抓**。每一步都有误差，误差会累积。你走到box前面偏了5cm，手就够不到handle。

所以humanoid loco-manipulation本质上是一个**error propagation game**：locomotion的小误差会被manipulation阶段放大成大失败。

### 2.2 传统data generation为啥fail

之前MimicGen系列（MimicGen [Mandlekar et al. CoRL 2023] https://openreview.net/forum?id=dk-2R1f_LR, DexMimicGen [Jiang et al. 2024] https://arxiv.org/abs/2410.24185）做data generation的核心idea很优雅：

> 我有一条demo，robot在state A抓object。现在object换到state B，我把end-effector trajectory做SE(3)变换，从object A的坐标系转到object B的坐标系。

公式是：
$$a'[e] = s'[f] \cdot s_0^\psi[f]^{-1} \cdot a^\psi[e]$$

变量解释：
- $a^\psi[e]$：source demo里end-effector $e$ 在某一帧的target pose（SE(3)刚体变换）
- $s_0^\psi[f]$：source demo开始时object $f$ 的pose
- $s'[f]$：新场景下object $f$ 的pose
- $a'[e]$：adapt到新场景的end-effector target pose

直觉：把source action"attach"到object frame上，object搬到哪，action跟到哪。

**但这在humanoid上炸了**。原因：

传统方法假设每个limb都能通过OSC (Operational Space Control, Khatib 1987)独立控制end-effector pose。humanoid做不到——腿在走路时必须时刻maintain balance，你没法对leg说"你独立track这个foot pose"，它要先保证不摔倒。

Reference: OSC原paper https://ieeexplore.ieee.org/document/1087247

## 三、Their Key Insight

Paper的核心insight可以用一句话概括：

> **Decouple dynamic stability from static precision.**

具体讲：humanoid在做manipulation时大部分时间是**站着不动**的，这时候腿只要freeze住maintain balance就行，不需要动态调整。而locomotion阶段虽然腿在动，但手臂可以闲着。

所以他们把整个task切成两类phase：
- **Dynamic phase**：腿在走，手臂跟着摆，用RL controller
- **Static phase**：站定操作，腿freeze，手臂做precision manipulation

这种decoupling让upper body和lower body的control problem**正交化**——你可以用classical IK + motion planning处理手臂的precision问题，用RL处理腿的dynamic问题。

这个insight其实echo了HOVER [He et al. ICRA 2025] https://arxiv.org/abs/2503.06281 和Homie [Ben et al. 2025] https://arxiv.org/abs/2502.13013 的whole-body control philosophy，但HumanoidMimicGen把它用在了**data generation**上。

## 四、The Method: Step by Step

### 4.1 Hybrid Action Space

他们设计的action space是hybrid的：

$$a = \langle a[J_{upper}], a[l] \rangle$$

- $a[J_{upper}]$：upper body joints（arms + hands + torso）的joint position commands
- $a[l] = [\dot{x}, \dot{y}, \dot{\theta}, z]$
  - $\dot{x}, \dot{y}$：pelvis的平面速度
  - $\dot{\theta}$：yaw角速度  
  - $z$：torso目标高度

RL controller接收$a[l]$输出leg joint commands。这个design的巧妙之处：**上层planner不需要知道怎么balance，只要会说"往前走0.5m/s"就行**。

### 4.2 Skill DAG

每个demo被切成skills，每个skill是 $\psi = \langle e, f, d^\psi \rangle$：
- $e$：哪只手
- $f$：抓哪个object
- $d^\psi$：这段demo的subsequence

skills之间有constraints：
- **Precedence**：pick必须在place之前
- **Coordination**：两只手必须同时pick

这些constraints构成DAG（有向无环图），paper用topological sort决定执行顺序。

Example: Table-to-Shelf任务
- 左手pick box + 右手pick box（并行）
- 左手place on shelf + 右手place on shelf（并行）
- 两层DAG

### 4.3 Three-Phase Planning（核心算法）

这是paper最clever的部分。给定新场景下object的pose，他们这么生成一条trajectory：

**Step 1: 求target configuration $q''$**

用whole-body IK求解"机器人最终姿态"，让active end-effectors都reach到adapted target pose $T[e]$。

公式：$T[e] = s[f] \cdot s_0^\psi[f]^{-1} \cdot s_0^\psi[e]$

这里 $s_0^\psi[e]$ 是source demo开始时EE pose，$s[f]$ 是当前object pose，$s_0^\psi[f]$ 是source demo开始时object pose。

**Step 2: 构造switch configuration $q'$**

$$q'[J_{upper}] = q[J_{upper}] \quad \text{(保持当前上半身)}$$
$$q'[J_{lower}] = q''[J_{lower}] \quad \text{(用target的腿位)}$$

这是个"hybrid姿态"——上半身还是当前state，下半身已经摆到manipulation该有的leg placement。从dynamic phase到static phase的桥梁。

**Step 3: Decoupled execution**

1. Plan locomotion trajectory $\tau_l$：$q \to q'$，用RL controller执行
2. Plan manipulation trajectory $\tau_m$：$q' \to q''$，用upper-body joint controller执行
3. Replay skill demo $\tau_{\Psi_i}$：逐帧IK跟踪adapted EE poses

**Why this works**：RL controller走路时velocity tracking不精确，会 drift。但如果在manipulation开始前先把腿freeze在 $q'$，后续的手臂planning就基于一个确定的base pose，误差不再propagate。

### 4.4 Motion Noise: 让policy学会recover

生成trajectory时加noise：
$$a' = a + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

**执行 $a'$ 但store $a$ 作为label**。

这个trick的intuition：BC policy容易过拟合到"expert trajectory的exact replay"。真实部署时control有误差，policy会进入training时没见过的state。motion noise让training data本身就包含"off-trajectory但recover回来"的behavior，policy学到的是recovery skill而非memorization。

这个idea来自MimicGen系列的传统 [Mandlekar et al. 2023] https://openreview.net/forum?id=dk-2R1f_LR，但在humanoid上效果尤其显著——ablation显示去掉motion noise，success从0.89掉到0.49。

### 4.5 Collision Handling的clever hack

Humanoid的mesh太复杂（尤其dexterous hand），没法手动放collision spheres。他们用自动化的shrinking sphere algorithm [Inui et al. 2016] https://www.cadanda.com/v13n2/199-207在mesh表面采样点，每个点求minimum tangent sphere，再greedy选subset最大化coverage。

但sphere over-approximation导致一个问题：grasped object和end-effector在initial/target state都"碰撞"（sphere包络比真实mesh大），IK直接infeasible。

Hack：每次IK前检测collision pairs，把colliding spheres shrink到out-of-collision。target EE pose也做同样处理，但只在end-effector所在rigid connected component内shrink。

## 五、Experiments: The Money Table

### 5.1 主结果

| Data source | Avg Success Rate |
|---|---|
| 1 human demo | 0.26 |
| 100 human demos | 0.48 |
| DexMimicGen+ (extended baseline) | 0.33 |
| **HumanoidMimicGen (1 demo → 1000 sim)** | **0.89** |

Key observations:
- **100 human demos还不如1 demo + auto generation**！人类teleop humanoid本身带noise，100条demo里的inconsistency反而让policy confused。
- DrillLiftObstacle这种long-horizon + navigation任务，100 human demos是**0%**！说明人类根本没法consistently teleop这种复杂task，data generation是唯一出路。

### 5.2 Architecture Ablation

| Architecture | Avg PSR |
|---|---|
| VLA (GR00T N1.6 finetune) | 0.89 |
| Flow Matching (AdaFlow) | 0.86 |
| Diffusion Policy | 0.51 |

VLA最强因为有pretrain prior。Diffusion Policy弱是因为1000条demo对diffusion model的multimodal action distribution建模不够，且长horizon下iterative denoising累积误差。

Reference: GR00T N1 https://arxiv.org/abs/2503.14734, AdaFlow https://arxiv.org/abs/2402.04292, Diffusion Policy https://diffusion-policy.cs.columbia.edu/

### 5.3 Sim-and-Real Co-training

Real-world tasks:
| Task | Real-only | Co-train |
|---|---|---|
| ThrowBottle | 0.60 | 0.75 |
| BoxToCart | 0.35 | 0.60 |
| PickCanister | 0.50 | 0.75 |
| PickCanisterWithObstruction | 0.60 | 0.75 |

20% relative improvement。Real data补sim-to-real gap，sim data补state coverage，两者互补。

### 5.4 Embodiment Generalization

他们还测了floating-base G1（去掉腿）：
| Embodiment | Avg |
|---|---|
| G1 legged | 0.89 |
| G1 floating | 0.90 |

surprising：floating-base略好！但细看individual task：
- PushShelfForward：legged 1.00 → floating 0.53（floating没法用whole-body force推东西）
- BoxTableToShelf：legged 0.53 → floating 1.00（floating精确控制base pose，没leg error）

intuition：legs是双刃剑。需要force interaction时是优势，需要precision时是累赘。这恰恰证明了hybrid action space的必要性——该用腿的时候用腿，该freeze的时候freeze。

## 六、Bigger Picture: 这篇paper的真正贡献

### 6.1 Data is the bottleneck, not algorithms

这篇paper最深刻的启示：**humanoid robot的scale瓶颈不在policy architecture，在data**。VLA模型已经够强（GR00T N1.6），问题是没有loco-manipulation数据来finetune。

100 human demos的0.48 vs 1000 sim demos的0.89——这个gap说明人类teleop humanoid本身是low-quality data source，而自动化generation能产生更consistent、更高coverage的数据。

这与LLM的发展轨迹parallel：GPT的突破不在transformer architecture，而在web-scale data。Robotics正在走同样的路，只是data collection更难，所以需要HumanoidMimicGen这样的"synthetic data engine"。

### 6.2 Classical planning + Learning的hybrid philosophy

paper拒绝pure end-to-end learning，保留了classical IK、collision checking、motion planning的structure。Why？

因为humanoid的action space虽然高维，但**structured**——skills之间有logical order，每个skill有object-relative geometry，whole-body IK有well-defined feasibility constraints。这些structure用classical方法处理更reliable，用learning从头学要大量data且不interpretable。

Learning的部分被限制在：1) RL locomotion controller处理dynamic balance，2) VLA policy处理perception到action的reactive mapping。Classical planning处理高层 sequencing 和 feasibility。

这种"structured learning"哲学和LeCun的JEPA idea有共鸣：不是所有knowledge都从raw pixels学，要有architectural prior。

Reference: LeCun on world models https://openreview.net/forum?id=BZ5a1r-kVsf

### 6.3 Sim-and-real co-training成为standard recipe

Reference: Maddukuri et al. https://arxiv.org/abs/2503.24361

这篇paper进一步confirm了sim-and-real co-training的effectiveness。recipe很清晰：
1. Sim里用data generation产生large-scale consistent demonstrations
2. Real里collect少量demos校准sim-to-real gap
3. Co-train，让policy既学到general skill manifold又在real distribution上fine-tuned

这个recipe正成为robotics manipulation的新baseline，类似于LLM里pretrain + SFT的范式。

## 七、Intuition for Karpathy

如果你要take away one thing：

> **Humanoid robot control的本质是error management。** Locomotion引入误差，manipulation放大误差。好的system design是**让误差在phase transition时reset**——dynamic phase允许loose tracking，static phase强制precise control，中间通过switch configuration做hard reset。

这个principle其实超越了robotics。在neural network training里，gradient noise在early training是feature（帮助escape sharp minima），late training是bug（破坏fine-tuning）。Stochastic depth、dropout这些technique本质上也是"phase-aware noise injection"。

HumanoidMimicGen的motion noise trick：execute noisy action but store clean label——这简直就是BC版的"learn from perturbed states, supervise by expert action"。和DPO里"negative sample with positive gradient"有结构上的相似。

Reference: DPO https://arxiv.org/abs/2305.18290

更深一层：**all control is about managing the gap between intended and actual**。Humanoid把这个gap暴露得最明显，所以解决humanoid loco-manipulation的method，往往蕴含着control learning的一般性principle。

## 八、Reference汇总

核心paper:
- HumanoidMimicGen project: https://humanoidmimicgen.github.io/
- MimicGen (predecessor): https://openreview.net/forum?id=dk-2R1f_LR
- DexMimicGen: https://arxiv.org/abs/2410.24185
- SkillGen: https://openreview.net/forum?id=YOFrRTDC6d

Architecture references:
- GR00T N1: https://arxiv.org/abs/2503.14734
- AdaFlow: https://arxiv.org/abs/2402.04292
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

Whole-body control:
- HOVER: https://arxiv.org/abs/2503.06281
- Homie: https://arxiv.org/abs/2502.13013
- SONIC: https://arxiv.org/abs/2511.07820
- OmniRetarget: https://arxiv.org/abs/2509.26633

Sim-real co-training:
- Maddukuri et al.: https://arxiv.org/abs/2503.24361
- Point Bridge: https://arxiv.org/abs/2601.16212

Motion planning infrastructure:
- cuRobo: https://arxiv.org/abs/2310.17274
- MuJoCo: https://mujoco.org/
- robosuite: https://robosuite.ai/

Theoretical context:
- OSC: https://ieeexplore.ieee.org/document/1087247
- CP-Gen: https://openreview.net/forum?id=KSKzA1mwKs

---

# HumanoidMimicGen: 深度技术解析

## 一、Core Problem & Motivation

HumanoidMimicGen 解决的核心问题：如何从**极少量人类遥操作演示**（single source demo per task）自动合成数千条高质量的loco-manipulation训练数据。这背后有一个关键张力——humanoid robot的action space是高维复合空间（arms + legs + torso + hands），传统的MimicGen这类data generation方法假设每个limb都有stable且independent的task-space control，但humanoid在locomotion时必须做whole-body coordination来保持balance。

Reference: 原始MimicGen [Mandlekar et al., CoRL 2023] https://openreview.net/forum?id=dk-2R1f_LR

## 二、Hybrid Action Space 的设计哲学

### 2.1 为什么不能直接用task-space OSC

传统的MimicGen / DexMimicGen系列方法假设每个limb通过Operational Space Control (OSC, Khatib 1987)独立控制end-effector pose。但humanoid有以下根本性困难：

1. **动态稳定性约束**：双足机器人locomotion时必须时刻保持ZMP（Zero Moment Point）落在support polygon内
2. **contact-rich**：腿与地面持续接触，不能像固定机械臂那样假设free space
3. **高度耦合**：torso motion同时影响两个end-effector的reachability

### 2.2 Hybrid Action Space的formulation

paper采用分层hybrid action space（灵感来自Homie [Ben et al. 2025] https://arxiv.org/abs/2502.13013）：

$$a = \langle a[J_{upper}], a[l] \rangle$$

其中：
- $a[J_{upper}]$ 是upper body（arms, hands, torso $J_t$）的joint position commands
- $a[l] = [\dot{x}, \dot{y}, \dot{\theta}, z]$ 是base motion command
  - $\dot{x}, \dot{y}$：planar base velocity（机器人骨盆的平面速度）
  - $\dot{\theta}$：yaw rate（绕z轴角速度）
  - $z$：desired torso height（躯干目标高度）

RL locomotion controller接收上述指令后输出dynamically feasible的leg joint position commands。这个decoupling把balance问题deleg给了预训练的RL policy，让上层planning无需关心dynamic stability。

## 三、Skill Planning DAG

### 3.1 Skill representation

每个source demonstration $d$ 被segment成object-centric skills：
$$\psi = \langle e, f, d^\psi \rangle$$

- $e \in \mathcal{E}$：end-effector frame（左/右手）
- $f \in \mathcal{F}$：reference object frame（如box、shelf）
- $d^\psi \subseteq d$：demonstration的连续子序列

### 3.2 Constraint types

paper定义两种约束：
- **Precedence** $\mathcal{P} = \{\langle \psi_{i_1}, \psi_{i_2} \rangle, \dots\}$：$\psi$必须先于$\psi'$完成
- **Coordination** $\mathcal{C} = \{\langle \psi_{j_1}, \psi_{j_2} \rangle, \dots\}$：两个skill必须同时启动

关键reduce：通过transitivity把coordination约减为precedence：
$$\mathcal{P} \leftarrow \mathcal{P} \cup \{\langle \psi^*, \psi' \rangle \in \Psi^2 \mid \exists \psi \in \Psi. \langle \psi, \psi' \rangle \in \mathcal{C} \land \langle \psi^*, \psi \rangle \in \mathcal{P}\}$$

这样skill plan就构成了DAG $\langle \Psi, \mathcal{P} \rangle$，顶点是skills，有向边是precedence。

### 3.3 Topological sort式的执行

每轮iteration greedy选择当前无依赖的skills：
$$\Psi_i = \{\psi \in \Psi \mid \neg \exists \psi' \in \Psi. \langle \psi', \psi \rangle \in \mathcal{P}\}$$

这相当于online topological sort，把incomparable skills分组同时执行。例如Table-to-Shelf任务两轮：先两个pick skills并行，再两个place skills并行。

## 四、Whole-Body Data Generation的完整算法

### 4.1 空间不变性adaptation公式

这是MimicGen的核心insight，paper用SE(3)的rigid transformation实现object-frame relative的end-effector pose transfer：

$$a'[e] = s'[f] \cdot s_0^\psi[f]^{-1} \cdot a^\psi[e]$$

变量解析：
- $a^\psi[e] \in SE(3)$：source demonstration中end-effector $e$ 的target pose
- $s_0^\psi[f] \in SE(3)$：skill开始时刻reference object $f$ 的pose
- $s'[f] \in SE(3)$：当前新state下object $f$ 的pose
- $s_0^\psi[f]^{-1}$：把source action变换到object frame
- $s'[f] \cdot (\cdot)$：把object-frame的action变换到新世界坐标

intuition：这个公式computes end-effector target pose relative to the initial object pose, then re-applies it to the new object pose。只要skill相对于object的几何关系不变，就能在新的object位置replay同样的contact-rich manipulation。

### 4.2 Three-Phase Whole-Body Planning

paper的关键创新是把planning拆成三阶段：

**Phase 1: Whole-Body IK求解target configuration $q''$**

调用 `whole-inv-kinematics(\mathcal{I}, T, s)` 返回batch $Q$，每个$q'' \in Q$满足active end-effector都reach到 $T[e]$。

**Phase 2: Switch configuration $q'$**

构造$q'$：upper joints = $q$（当前），lower joints = $q''$（target的腿部）。这是"从locomotion切到manipulation"的桥接姿态。

**Phase 3: Decoupled trajectory planning**

- Locomotion trajectory $\tau_l$：从 $q$ 到 $q'$，用RL controller执行
- Manipulation trajectory $\tau_m$：从 $q'$ 到 $q''$，用upper-body joint-space controller执行
- Skill adaptation trajectory $\tau_{\Psi_i}$：通过`adapt-skill-demos`逐帧IK跟踪adapted end-effector poses

### 4.3 Why decoupled?

paper的insight：RL locomotion controller虽然能维持balance，但**无法精确跟踪velocity commands**。如果在manipulation过程中还试图让腿动态调整，会出现：
- End-effector pose抖动
- Upper-body tracking误差累积

decoupled approach把"动态稳定"和"静态精确"分离：locomotion阶段允许velocity tracking不完美，manipulation阶段则把腿冻结在switch configuration $q'$。

### 4.4 Pseudocode核心循环（Algorithm 1）

```
while |Ψ| ≠ 0:
    Ψ_i ← 当前无依赖的skills
    Ψ ← Ψ \ Ψ_i
    for each skill ψ in Ψ_i:
        T[e] = s[f] · s_0^ψ[f]^(-1) · s_0^ψ[e]  # 目标EE pose
    Q ← whole_body_IK(J, T, s)
    for q'' in Q:                                # batch尝试
        q' ← copy(q); q'[J_l] ← q''[J_l]          # switch config
        τ_l ← plan_motion(J_l, q, q', s)
        if τ_l exists:
            s ← control_locomotion(τ_l)
            break
    q' ← s[J]   # 实际achieved switch config
    τ_m ← plan_motion(J_t ∪ J_{a_l} ∪ J_{a_r}, q', q'', s)
    s ← control_manipulation(τ_m)
    τ_Ψi ← adapt_skill_demos(s, Ψ_i)
    s ← control_manipulation(τ_Ψi)
return check_success(s)
```

## 五、Motion Planning细节（Appendix C）

### 5.1 Spherical Collision Representation

paper在cuRobo [Sundaralingam et al., ICRA 2023] https://arxiv.org/abs/2310.17274 之上构建humanoid collision model。挑战：humanoid（特别是dexterous hand）的mesh过于复杂，手动放置spheres不现实。

自动化流程：
1. 在每个mesh表面采样大量点
2. 对每个采样点 $p$，求解minimum volume sphere tangent to $p$ and another face on mesh（用shrinking sphere algorithm [Inui et al. 2016]）
3. inflate sphere radius by $\epsilon \approx 0.01m$（覆盖更多表面）
4. greedy combinatorial optimization：在计算budget内选 $K$ 个spheres maximize surface coverage

### 5.2 Lazy sphere shrinking for contact states

关键问题：sphere over-approximation导致初始配置和目标EE pose都处于collision state（与grasped object），使IK和planning infeasible。

解决方案：每次IK/planning调用前，**检测当前collision pairs并shrink spheres直到out of collision**。对target end-effector poses，仅在same rigid connected component（given planning joints）内shrink。

### 5.3 Free joint order for IK

paper用iterative optimization最小化$L_0$ distance：
$$\min ||q'' - q||_0$$
weighted by joint group。

具体free joint顺序：
$$[J_a, \quad J_a \cup J_t, \quad J_a \cup J_l, \quad \mathcal{I}]$$

对应4层fallback：
1. 只动arms（最cheap，risk最低）
2. arms + torso（torso motion同时影响两个EE）
3. arms + legs（需要重新规划locomotion）
4. 全自由度（最后兜底）

这个设计的intuition：尽量保留当前configuration，减少不必要的torso和leg motion，避免引入control error。

## 六、G1 Loco-Manipulation Benchmark

### 6.1 9个task的设计维度

paper沿三个axis设计tasks：

| Task | Loco | Nav | 1-arm | 2-arm | Vert | Contact | Long |
|------|------|-----|-------|-------|------|---------|------|
| PushButton | √ | | √ | | | | |
| DrillLift | √ | | √ | | | | |
| BoxLift | √ | | | √ | | | |
| BoxLiftFloor | √ | | | √ | √ | | |
| PickDrillFromHolder | √ | | √ | | | √ | |
| PushShelfForward | √ | | | √ | | √ | |
| DrillLiftObstacle | √ | √ | √ | | | | √ |
| DrillPnP | √ | | √ | | | | √ |
| BoxTableToShelf | √ | | | √ | √ | | √ |

Capabilities含义：Loco=locomotion; Nav=obstacle-free navigation; 1-arm=single-arm; 2-arm=bimanual; Vert=vertical reach; Contact=contact-rich; Long=long-horizon。

### 6.2 为什么这个benchmark重要

paper明确强调：**small base-placement errors can make downstream manipulation unreachable**。这正是loco-manipulation的根本难点——locomotion误差会propagate到manipulation阶段。例如BoxLift中，如果base yaw error积累几度，end-effector就够不到box handle。

## 七、实验结果深度分析

### 7.1 主结果（Table 1）

| 设置 | Avg PSR |
|------|---------|
| 1 Human Demo | 0.26 |
| 100 Human Demos | 0.48 |
| DexMimicGen+ | 0.33 |
| HumanoidMimicGen (Ours) | 0.89 |

intuition：
- 1 human demo就训policy只能学到single trajectory的memorization，generalization极差
- 100 human demos虽然覆盖更多states，但human teleop本身带noise，且loco-manipulation很难一致地perform
- DexMimicGen+虽然extended到locomotion，但缺乏whole-body planning + collision checking，导致很多navigation失败
- HumanoidMimicGen从1 demo生成1000条，每条都经whole-body IK + motion planning验证feasibility，因此成功率飞升

值得注意：**DrillLiftObstacle**这个long-horizon + navigation任务上：
- 1 human demo: 0.04
- 100 human demos: 0.00
- DexMimicGen+: 0.00
- Ours: 0.87

100个human demos居然是0%！说明这类需要精确navigation + manipulation的长程任务，human teleop难以produce consistent demonstrations，data generation才是唯一可行路径。

### 7.2 Ablation: Motion noise vs Init noise (Table 5)

| Setup | Avg |
|-------|-----|
| Ours (full) | 0.89 |
| w/o Motion Noise | 0.49 |
| w/o Init. Noise | 0.51 |

motion noise的形式：
$$a' = a + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$
关键：执行 $a'$ 但store $a$ 作为label。这让policy看到off-nominal state分布但supervision signal仍是clean expert action。

这个ablation的intuition：BC policy容易"过拟合到expert trajectory exact replay"，但真实rollout时control有error，policy必须能在perturbed state下recover。motion noise在training时就把这种recovery behavior编码进data。

### 7.3 Policy architecture ablation (Table 4)

| Architecture | Avg PSR |
|--------------|---------|
| VLA (GR00T N1.6 finetune) | 0.89 |
| Flow Matching (AdaFlow) | 0.86 |
| Diffusion Policy | 0.51 |

Reference: AdaFlow [Hu et al. 2024] https://arxiv.org/abs/2402.04292; Diffusion Policy [Chi et al. 2023] https://diffusion-policy.cs.columbia.edu/

Diffusion Policy表现差的原因（猜测）：1000条demo对diffusion model来说不够覆盖multimodal action distribution，且loco-manipulation的action horizon很长，diffusion的iterative denoising引入累积误差。

VLA略胜Flow Matching的原因：GR00T N1.6 base model已经预训练在large robotic manipulation datasets上，自带strong prior，finetune时只需adapt到loco-manipulation distribution。

### 7.4 Sim-and-Real Co-training (Figure 5)

Real-world tasks:
| Task | Real Only | Co-training |
|------|-----------|-------------|
| ThrowBottle | 0.60 | 0.75 |
| BoxToCart | 0.35 | 0.60 |
| PickCanister | 0.50 | 0.75 |
| PickCanisterWithObstruction | 0.60 | 0.75 |
| Avg | 0.51 | 0.71 |

20% relative improvement。intuition：sim data提供broad state coverage和consistent expert behavior，real data提供sim-to-real gap的calibration。两者co-train时policy既学到general skill manifold又在real distribution上fine-tuned。

## 八、Embodiment Generalization (Table 3)

paper还测试了floating-base variant（去掉legs的G1）：

| Task | G1 (legged) | G1 w/o Legs |
|------|-------------|-------------|
| Avg | 0.89 | 0.90 |

surprising result：floating-base平均性能略高！但深入看：
- PushShelfForward：legged 1.00 → floating 0.53（floating-base无法提供push所需的whole-body force）
- BoxTableToShelf：legged 0.53 → floating 1.00（floating-base精确控制base pose，避免了leg placement误差）
- DrillPnP：legged 0.70 → floating 0.57

intuition：legged humanoid在需要force interaction的任务上有优势，但precision manipulation任务反而受locomotion error拖累。这正是hybrid action space存在的根本motivation。

## 九、Limitations与Future Direction

paper坦诚的局限：
1. **Manual skill annotation**：需要人工标注skill segments和precedence/coordination constraints。未来可用foundation models自动化（如VLM做video segmentation）
2. **Fixed skill sequence**：当前假设fixed set of object-centric skills和fixed skill sequence structure，不能 generalize到需要新skill组合的任务
3. **Rigid object-frame assumption**：不能处理large intra-category geometric variation或ambiguous contact affordances。引用CP-Gen [Lin et al., CoRL 2025] https://openreview.net/forum?id=KSKzA1mwKs 作为可能解决方案

## 十、技术贡献的更大图景

HumanoidMimicGen处在几个trend的交叉点：

1. **VLA scaling bottleneck**：GR00T N1 [Bjorck et al. 2025] https://arxiv.org/abs/2503.14734 这类VLA需要海量data，但humanoid teleop成本极高。data generation是唯一可行scale路径。

2. **Classical planning meets learning**：与OmniRetarget [Yang et al. 2025] https://arxiv.org/abs/2509.26633 这类纯RL方法不同，HumanoidMimicGen保留了classical motion planning的reliability（IK、collision checking）和RL的dynamic feasibility（leg controller），是hybrid systems的典范。

3. **Sim-and-real co-training becoming standard**：与Maddukuri et al. https://arxiv.org/abs/2503.24361 这类工作呼应，sim data补足real data的coverage问题成为主流recipe。

4. **Whole-body control的decoupled design**：HOVER [He et al. ICRA 2025] https://arxiv.org/abs/2503.06281 和SONIC [Luo et al. 2025] https://arxiv.org/abs/2511.07820 等whole-body controller工作提供了lower-level interface，HumanoidMimicGen则在上层做planning。两者正交且互补。

## 十一、可以深挖的若干方向

1. **Skill discovery自动化**：当前依赖human annotation。可以用inverse RL或VLM-based video understanding自动segment skill boundaries。

2. **Adaptive skill sequence**：DAG是fixed的，但real-world execution可能需要contingency-aware replanning。可结合LLM做online skill selection。

3. **Contact-aware IK**：当前sphere shrinking是greedy heuristic。可以用differentiable contact model做joint optimization。

4. **Cross-embodiment transfer**：Table 3只测了G1的legged/floating两种variant。如果能把HumanoidMimicGen的source demo cross-transfer到不同humanoid（如Unitree H1 vs G1），将极大降低per-robot data collection成本。

5. **Failure-aware data generation**：当前discard失败的planning attempts。可以分析failure modes，做active data generation prioritize hard states。

## 十二、对Karpathy可能的intuition启发

从teaching的角度，HumanoidMimicGen的**核心insight**可以这样概括：

> Data quality > Data quantity。但data quality的定义在loco-manipulation中是"经过feasibility验证的、state coverage充分的、带recovery behavior的" demonstrations。

1 human demo的0.26 → 1000 sim demos的0.89，差距不是来自"看到更多object positions"，而是来自：
- **Whole-body planning的约束satisfaction**（每个generated trajectory都通过IK和collision check）
- **Motion noise注入的distributional robustness**
- **Init state randomization带来的state coverage**

这与LLM pretraining的"quality > quantity"趋势是一致的：carefully curated data远胜于海量low-quality data。

另一个深层insight：**humanoid robot的control problem本质上是structured的**——不是end-to-end learning一个flat policy，而是发现skills之间的combinatorial structure（DAG），用classical planning处理feasibility，用learning处理perception和reactive control。这种"structured learning"哲学可能就是robotics从pure RL走向practical deployment的关键路径。

Reference汇总：
- Project page: https://humanoidmimicgen.github.io/
- MimicGen: https://openreview.net/forum?id=dk-2R1f_LR
- DexMimicGen: https://arxiv.org/abs/2410.24185
- SkillGen: https://openreview.net/forum?id=YOFrRTDC6d
- Homie: https://arxiv.org/abs/2502.13013
- GR00T N1: https://arxiv.org/abs/2503.14734
- cuRobo: https://arxiv.org/abs/2310.17274
- HOVER: https://arxiv.org/abs/2503.06281
- SONIC: https://arxiv.org/abs/2511.07820
- OmniRetarget: https://arxiv.org/abs/2509.26633
- Sim-and-real co-training: https://arxiv.org/abs/2503.24361
- CP-Gen: https://openreview.net/forum?id=KSKzA1mwKs
- AdaFlow: https://arxiv.org/abs/2402.04292
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
