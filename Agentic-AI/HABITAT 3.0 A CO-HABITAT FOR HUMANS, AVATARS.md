---
source_pdf: HABITAT 3.0 A CO-HABITAT FOR HUMANS, AVATARS.pdf
paper_sha256: 6383018901acbf7677050585da9a30707437cf7119def797be500f33a68021af
processed_at: '2026-08-19T10:16:31-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说Habitat 3.0

## 一句话版本

以前的embodied AI模拟器里机器人像个隐士在空房子里晃悠，Habitat 3.0终于把"人"塞进去了，而且速度还快得离谱。

---

## 为什么需要这东西

你想想，assistive robot的终极目标是啥？是帮人干活。但之前所有simulator里，robot都是一个人在空荡荡的虚拟房子里navigation，从来没有"人"这个变量。

这就像你想训练一个服务生，但从来不让他见到客人。这不扯淡吗？

所以Meta的人说：**够了，我们得让人和robot能在一个模拟器里共处**。

---

## 三大难题，三个解法

### 难题1：怎么在模拟器里造个"真人"

人体很难搞——皮肤会变形、关节会弯、body shape千差万别。用SMPL-X那种参数化模型可以生成realistic的人，但query一次就慢得要死，根本跑不了RL需要的上亿次step。

**他们的trick**：offline先把12种身材（4男4女4中性）的skeleton rig和blendshape全算好cache住，sim时只做最便宜的LBS skinning。

代价是啥？皮肤形变不再随pose动态变化，某些姿势会有skinning artifact。但换来的是：

- 两个robot：1345 FPS
- 一个robot + 一个humanoid：1190 FPS

**就掉了11%**。这几乎是免费的realism。

### 难题2：怎么让真人进来玩

搞了个client-server架构。server跑simulation和AI inference（重活），client只管rendering和input（轻活，能port到浏览器甚至VR）。

这样researcher可以用keyboard/鼠标做小规模test，也可以拉30个人做大规模user study。

### 难题3：设计什么任务来研究协作

两个任务：

**Social Navigation**：robot要找到人、跟上人、保持1-2米安全距离。不是简单的navigation——目标是动态的，障碍物（人）也是动态的。

**Social Rearrangement**：robot和人一起把两个东西搬到指定位置。真正的协作，要分工、要让路、要不撞在一起。

---

## 最有意思的发现

### 发现1：RL自己学出了"让路"

没有人在reward function里写"请礼让行人"。但RL policy自己学到了：
- 在窄走廊里做three-point turn给人让路
- 预测人的走向提前后退
- 在门口主动yield

这就是emergent behavior的魔力。你给它一个distance-based reward，它自己搞出了social navigation的"潜规则"。

### 发现2：训练时见过多种partner，才能泛化

这是social rearrangement里最核心的实验。

他们试了几种training population：

- **Learn-Single**：只跟一个partner训练 → train时98%成功率，换partner暴跌到50%
- **Plan-Pop3,4**：训练时见过3-4种不同partner → train时77%，换partner还能保持71%
- **Learn-Pop**：8个随机初始化的partner一起train → 以为能产生diversity，结果失败，泛化很差

**关键insight**：随机初始化产生的behavioral diversity不够。你得**explicitly设计**不同的partner behavior（比如让不同partner负责搬不同的object）才能逼出泛化能力。

这跟AlphaStar的population play思路一脉相承，但在embodied setting里是第一次。

### 发现3：真人的反应跟ZSC agent不一样

30个真人参与user study。结果有个反直觉的事：

在自动评估里，Plan-Pop3的ZSC泛化最好。但到了真人study里，**Learn-Single反而让人更高效**（RE 134% vs 123%）。

猜为啥？真人是**reactive**的——你会看robot在干嘛，然后自己调整计划。但ZSC里的humanoid agent没那么灵活。

所以Learn-Single虽然只会跟一种partner配合，但这种partner的行为模式恰好跟reactive的真人更像。**真人更像"单一partner"而不是"diverse population"**。

这个发现其实挺重要的：ZSC的assumptions可能不完全适用于real human-robot interaction。

### 发现4：自动评估的ranking在真人里保持了

虽然绝对数字对不上，但**相对排序对上了**：Learn-Single比Plan-Pop3让人更快完成任务。这说明simulator的fidelity够用，可以在里面做human-robot collaboration的研究。

---

## 我的几个takeaway

**1. Caching是王道。** 别追求每帧都perfect，offline算好cache住，runtime只做最便宜的操作。这个思路在很多地方都work。

**2. Population diversity要explicit。** 靠random init产生diversity是不靠谱的。你需要design diversity的axes（谁搬哪个object、速度多快、风格如何）。

**3. Emergent social behavior是真的。** 没人教robot让路，但reward shaping下它自己学出来了。这说明social behavior可能没那么难emerge。

**4. Sim-to-real的gap可能没那么大。** 至少在ranking这个层面，自动评估能预测真人的体验。

**5. ZSC跟real human有gap。** 真人更reactive、更adaptive，跟simulated humanoid行为模式不同。future work应该让humanoid agent更reactive。

---

## 这paper的真正贡献

不是某个algorithm多novel，而是infrastructure。他们造了个能让人和robot在同一个高速模拟器里coexist的平台，然后证明了上面能跑出emergent collaborative behavior，并且自动评估跟真人评估对得上。

**这就是打开embodied social AI这个方向的infra paper。** 以后做human-robot collaboration的人不用再造轮子了，直接在这上面跑。

---

# Habitat 3.0: 深度技术解析

Hi Andrej, 让我深入拆解这篇paper的技术细节，帮你build intuition。

## 1. Core Positioning

Habitat 3.0本质上是Meta的Embodied AI模拟器系列的第三版，核心thesis是：**以前的embodied AI agent都是"hermits"（隐士），独自在虚拟世界中导航，现在要引入humanoid avatars，让robot学会与人collaborate**。这是从single-agent embodied AI迈向social embodied AI的关键一步。

项目主页: http://aihabitat.org/habitat3
视频overview: https://tinyurl.com/msywvsnz
GitHub (Habitat系列): https://github.com/facebookresearch/habitat-sim

---

## 2. Human Simulation - 技术核心

### 2.1 SMPL-X 参数化人体模型

人体表示基于 SMPL-X (Pavlakos et al., 2019)，这是一个data-driven的parametric body model。关键参数：

$$\theta \in \mathbb{R}^{109}, \quad \beta \in \mathbb{R}^{10}$$

- $\theta$ (theta): pose参数，109维，编码hands + body + face的joint rotations
- $\beta$ (beta): shape参数，10维，通过mesh vertex displacements的principal components捕获body shape variation

Paper里有个typo把 $\theta$ 写成了 $\dot{J}$，从context看应该是pose参数 $\theta$。

SMPL-X reference: https://smpl-x.is.tue.mpg.de/

### 2.2 Skeleton + Skinned Mesh 双重表示

这是graphics领域的经典方法 (Kavan & Žára, 2005; Kavan et al., 2007)：

- **Articulated skeleton**: 骨骼通过rotational joints连接，用于physics collision checking
- **Surface skinned mesh**: 附着在skeleton上的mesh，通过Linear Blend Skinning (LBS)更新vertices

**关键设计哲学**: physics用简化skeleton，rendering用skinned mesh，二者解耦。这样collision detection快，但visual fidelity不损失。

### 2.3 Linear Blend Skinning (LBS) 公式

LBS是skinning的标准方法，vertex $v$ 的新位置：

$$v' = \sum_{i=1}^{n} w_i \cdot M_i \cdot v$$

其中：
- $w_i$: vertex $v$ 受第 $i$ 个joint影响的权重
- $M_i$: 第 $i$ 个joint的transformation matrix
- $n$: 影响该vertex的joint数量

### 2.4 Caching策略 - 性能关键

**这是全文最聪明的设计**：offline计算12个body models (4 male + 4 female + 4 neutral)的skeleton rig和blend shapes，simulation时只需要做LBS。

**Trade-off**: 失去pose-dependent blend shapes（皮肤形变不再随pose动态变化），但换取了：
- 1190 FPS (robot + humanoid)
- 1345 FPS (two robots)
- 性能损失仅 ~11%

对比：Habitat 2.0只有robots时1400 FPS，加了humanoid基本不掉帧，这是巨大的工程成就。

### 2.5 Motion Generation - Hierarchical Behavior

**Walking**: 从AMASS dataset (Mahmood et al., 2019)取motion clip，trim成单个walking cycle，cyclic播放。

AMASS: https://amass.is.tue.mpg.de/

**Reaching**: 用VPoser (SMPL-X的VAE pose prior) offline预计算一组reaching poses，indexed by 3D hand position relative to humanoid root。runtime时通过interpolation获取新位置，避免query VPoser。

**Pick/Place motion**: 从current hand position到target画3D line，沿线采样intermediate points，retrieve对应poses。当hand到达object position时，kinematically attach/detach object。

**Failure mode**: 当hand太近或太远于body时，超出VPoser的valid range，motion会失败。

---

## 3. Social Navigation: Find & Follow People

### 3.1 Task Setup

- Robot: Boston Dynamics Spot
- Sensors:
  - Arm depth camera (224×171, hFOV 55°)
  - Arm RGB camera (480×640, hFOV 47°)
  - Binary humanoid detector (1-dim)
  - Humanoid GPS: relative distance + heading (2-dim, polar coordinates)

### 3.2 Reward Function 深度解析

Distance reward:
$$r_t^{distance} = \begin{cases} 
\Delta(b_t, h_t) - \Delta(b_{t-1}, h_{t-1}), & \text{if } \Delta(b_t, h_t) \leq 1 \\
2, & \text{if } 1 < \Delta(b_t, h_t) \leq 2 \\
\Delta(b_{t-1}, h_{t-1}) - \Delta(b_t, h_t), & \text{otherwise}
\end{cases}$$

变量解释：
- $b_t$: robot在time $t$的位置
- $h_t$: humanoid在time $t$的位置
- $\Delta(\cdot, \cdot)$: geodesic distance

**Intuition**: 三个regime
1. **太近 (<1m)**: 鼓励远离（reward正比于远离量）
2. **甜蜜区 (1-2m)**: 固定reward +2，这是"safe following distance"
3. **太远 (>2m)**: 鼓励靠近

Orientation reward:
$$r_t^{orientation} = \begin{cases} 
(h_t - b_t) \cdot v_t^{forward}, & \text{if } \Delta(b_t, h_t) \leq 3 \\
0, & \text{otherwise}
\end{cases}$$

- $v_t^{forward}$: robot在世界坐标系中的normalized forward vector
- $(h_t - b_t)$: 也normalized，指向humanoid的单位向量

**Intuition**: 鼓励robot面对humanoid（点积最大为1，当两向量同向时）

Total reward:
$$r_t^{social-nav} = 10 \cdot \mathbb{1}^{success} + r_t^{distance} + 3 \cdot r_t^{orientation} - 0.1$$

- $\mathbb{1}^{success}$: 维持1-2m距离且面对humanoid至少400 steps时为1
- $-0.1$: slack penalty，鼓励尽快找到humanoid

### 3.3 Metrics

**Finding Success (S)**: 是否在max steps内找到humanoid（1-2m且facing）

**SPS (Success Weighted by Path Steps)**:
$$\text{SPS} = S \cdot \frac{l}{\max(l, p)}$$

- $l$: oracle minimum steps to find humanoid
- $p$: agent's actual path steps

**Following Rate (F)**:
$$\text{F} = \frac{w}{\max(E - l, w)}$$

- $w$: agent成功跟随humanoid的steps数
- $E$: episode总长度
- $E - l$: oracle能跟随的steps

### 3.4 Oracle Minimum Steps 计算

动态目标的最短路径问题：
$$l = \arg\min_i (r_i < i)$$

- $r_i$: 从robot起点到humanoid第 $i$ 步位置 $h_i$ 的min steps
- 找最早的 $i$ 使得 $r_i < i$（即robot能在humanoid到达该位置前先到）

### 3.5 Training Details

- DDPPO (Wijmans et al., 2019) - Distributed Distributional PPO
- 4× NVIDIA A100 GPUs
- 24 parallel envs per GPU, 128 steps per update
- LSTM + ResNet-18, 8517k params
- Learning rate: $10^{-4}$, max grad norm: 0.2
- 200M env steps (~4 days training)
- 3 random seeds

DDPPO paper: https://arxiv.org/abs/1911.00357

### 3.6 Results Analysis

| Method | S | SPS | F | CR |
|--------|---|-----|---|-----|
| Heuristic Expert | 1.00 | 0.97 | 0.51 | 0.52 |
| End-to-end RL | 0.97±0.00 | 0.65±0.00 | 0.44±0.01 | 0.51±0.03 |

**Key insight**: Heuristic有map privileged info，所以finding success更高。但RL policy学到emergent behaviors:
- Anticipating humanoid motion
- Backing up to avoid collisions  
- Making way in narrow spaces (3-point turn)

**Collision Rate 0.51看似高**: 因为episode termination定义为collision OR max steps，所以CR其实是"以collision结束的episode比例"，不是"每步碰撞概率"。

### 3.7 Ablations - Sensor Importance

| Ablation | S | SPS | F | CR |
|----------|---|-----|---|-----|
| - humanoid GPS | 0.76 | 0.34 | 0.29 | 0.48 |
| - humanoid detector | 0.98 | 0.68 | 0.37 | 0.64 |
| - arm depth | 0.94 | 0.54 | 0.19 | 0.71 |
| - arm depth + arm RGB | 0.96 | 0.61 | 0.38 | 0.55 |

**Phase-dependent sensor importance**:
- **Before finding**: humanoid GPS最重要（定位）
- **After finding**: arm perception最重要（avoid collision during following）

---

## 4. Social Rearrangement - 协作任务

### 4.1 Task Setup

- Robot + humanoid协作移动2个objects到target位置
- Object位置已知（3D coordinates in robot's start frame）
- Robot sensors: egocentric depth, proprioceptive state, humanoid relative distance + heading
- **No access to**: humanoid actions, intents, full states

### 4.2 Two-Layer Policy Architecture

```
High-level policy (learned)
    ↓ selects skill from library
Low-level skills (oracle OR learned)
    - Navigate to X
    - Pick object Y
    - Place at Z
    + 4 primitive actions (forward/back/turn left/turn right)
```

**Oracle skills**: privileged info (map for nav, instantaneous pick/place)
**Learned skills**: pre-trained, frozen, realistic (no privileged info)

### 4.3 Reward Function

$$r_t^{social-rearrange} = 10 \cdot \mathbb{1}^{success} + 5 \cdot \mathbb{1}^{subgoal} - 5 \cdot \mathbb{1}^{collision} - 0.005$$

- $+10$: 整个task success
- $+5$: 完成任何subgoal（pick一个target OR place一个object到goal）
- $-5$: collision penalty + episode terminate
- $-0.005$: per-timestep penalty，鼓励快速完成

### 4.4 Population-Based Training - 核心创新

参考Jaderberg et al., 2019 (AlphaStar): https://arxiv.org/abs/1904.05754

**三种baselines**:

**Learn-Single**: 单个humanoid policy，低diversity
**Plan-Pop_p**: privileged planner驱动的population，size $p \in \{1,2,3,4\}$
**Learn-Pop**: 8个random init的humanoid policies，jointly trained

### 4.5 ZSC-Pop-Eval (Zero-Shot Coordination)

10个evaluators组成ZSC population:
- 3 from Learn-Single checkpoints
- 3 from Learn-Pop checkpoints  
- 4 planner-based (1 picks both, 2 pick one, 1 stays still)

每个baseline训练时见到约1/3的ZSC collaborators，需要generalize到2/3。

### 4.6 Results - The Generalization Story

| Method | Train-pop SR | ZSC-pop SR | Train-pop RE | ZSC-pop RE |
|--------|--------------|------------|--------------|------------|
| Learn-Single | 98.50 | 50.94 | 159.2 | 106.02 |
| Plan-Pop1 | 91.2 | 50.44 | 152.4 | 109.75 |
| Plan-Pop2 | 66.89 | 70.23 | 110.06 | 102.13 |
| Plan-Pop3 | 77.79 | **71.79** | 118.95 | 101.99 |
| Plan-Pop4 | 72.42 | 71.32 | 105.49 | 103.53 |
| Learn-Pop | 92.20 | 48.52 | 135.32 | 99.80 |

**Critical insight**: 
- Learn-Single和Plan-Pop1在train-pop上最好(>90%)，但ZSC-pop暴跌到~50%
- Plan-Pop3,4在ZSC-pop上最好(~71%)
- **Learn-Pop失败**: 说明random init不能产生足够diversity的population

**RE metric intuition**: 
$$\text{RE} = \frac{L^{human}}{\max(L^{joint}, E)}$$

- $L^{human}$: humanoid单独完成任务steps
- $L^{joint}$: humanoid-robot team完成steps
- $E$: max allowed steps

RE > 100%意味着robot让humanoid更高效。Learn-Single在train-pop达到159.2%，说明robot让humanoid效率提升59%。

### 4.7 为什么Learn-Pop失败？

Random init的8个humanoid policies没有产生足够behavioral diversity。对比Plan-Pop通过设计不同的object preferences强制diversity。

**Future direction**: 需要更好的population diversity sources，如humanoid speed, motion style等。

### 4.8 Ablations

| Ablation | Train-pop SR | ZSC-pop SR |
|----------|--------------|------------|
| + learned skills | 41.09 | 21.44 |
| - depth + RGB | 76.70 | 70.89 |
| - Humanoid-GPS | 76.45 | 68.70 |
| - Primitive actions | 85.71 | 76.80 |

**Learned skills大幅下降**: 77.79% → 41.09%，因为high-level policy没见过low-level failures。

**Primitive actions去除反而SR提升** (85.71% vs 77.79%)，但CR暴涨(0.12 → 0.32)，说明agent不浪费时间让路，但safety下降。

---

## 5. Low-Level Skills 训练细节

### 5.1 Navigation Skill

$$r_t^{nav} = 10 \cdot \mathbb{1}^{success} + r_t^{distance} + 0.05 \cdot r_t^{orientation} - 0.005 \cdot \mathbb{1}^{collision} - 0.01$$

- $\mathbb{1}^{success}$: distance < 1.5m AND forward·object vector > 0.5
- 300M steps (~6 days), 90% success rate

### 5.2 Pick Skill

$$r_t^{pick} = 2 \cdot \mathbb{1}^{success} + r_t^{move} + r_t^{retract} - 0.005$$

- $r_t^{move}$: end-effector靠近object
- $r_t^{retract}$: pick后retract arm到resting position
- Grasp通过teleport模拟(< 0.15m)
- 100M steps (~1 day)

### 5.3 Place Skill

$$r_t^{place} = 10 \cdot \mathbb{1}^{success} + r_t^{bonus} + r_t^{move} + r_t^{retract} - 0.005$$

- $r_t^{bonus} = +5$: 放对位置
- 50M steps (~0.5 day)

---

## 6. Human-in-the-Loop Evaluation

### 6.1 Architecture - Client-Server

- **Server**: simulation + agent inference + avatar controls (heavy compute)
- **Client**: platform-specific rendering + input handling (portable to web browser, VR)

### 6.2 User Study

30 participants, 3 conditions:
1. Solo (human alone)
2. With Learn-Single robot
3. With Plan-Pop3 robot

Each: 10 episodes per condition, Latin-square counter-balanced ordering

### 6.3 Statistical Analysis

用GLMM (Generalized Linear Mixed-Effect Model)分析：
- Dependent var: Task Steps (TS) - Poisson distribution
- Independent var: method/condition
- Random effects: participants + scenes (random intercepts)

**Why GLMM?** Human有learning effect，不能简单平均。GLMM控制participant和scene的random effects，估计每个condition的mean TS。

### 6.4 HITL Results

| Method | CR | TS | RC | RE |
|--------|----|----|----|----|
| Solo | 0.0 | 1253.17 | - | 100.0 |
| Learn-Single | 0.12 | 936.60 | 0.36 | 133.80 |
| Plan-Pop3 | 0.13 | 1015.05 | 0.44 | 123.46 |

**Key findings**:
1. Both robots让human更高效 (RE > 100%)
2. Learn-Single更高效但Plan-Pop3 offload更多task (RC 0.44 vs 0.36)
3. **Automated eval的relative ordering在HITL中保持** - 重要validation
4. Humans更reactive than ZSC agents，SR=1.0 across all episodes

### 6.5 Post-hoc Pairwise Comparisons

| Comparison | Mean diff in TS | p-value |
|------------|-----------------|---------|
| Plan-pop vs Learn-Single | 78.45 | 0.0533 |
| Solo vs Learn-Single | 316.58 | <0.0001 |
| Solo vs Plan-pop | 238.12 | <0.0001 |

**Learn-Single vs Plan-Pop3差异不显著** (p=0.0533)，但都比solo显著好。

---

## 7. Simulator Performance - 工程奇迹

### 7.1 Benchmarking Setup

- Single Nvidia V100 GPU
- 300 steps per agent, 10 runs
- Scene sizes:
  - Small: 68.56 m² (1 bed, 1 bath)
  - Medium: 136.11 m² (3 bed, 2 bath)
  - Large: 846.15 m² (4 bed, 4 bath + den + office)

### 7.2 Key Performance Numbers

**Single environment**: 140-250 FPS
- 1 Spot: 245±19 FPS
- 2 Spots: 150±13 FPS
- Spot + Humanoid: 136±8 FPS (基本与2 robots持平!)

**16 environments**: 1100-2290 FPS
- Robot-humanoid: 1191±3 FPS

### 7.3 与其他模拟器对比

| Simulator | Robot | Humanoid | HITL | Speed (steps/s) |
|-----------|-------|----------|------|------------------|
| VirtualHome-Social | ✗ | ✓ | MK | 10 |
| VRKitchen | ✗ | ✓ | MK+VR | 15 |
| SAPIEN | ✓ | ✗ | ✗ | 200-400 |
| AI2-THOR | ✓ | ✗ | ✗ | 90-180 |
| Habitat 2.0 | ✓ | ✗ | ✗ | 1400 |
| TDW | ✗ | ✓ | VR | 5-168 |
| SEAN 2.0 | ✗ | ✓ | MK | 3-60 |
| iGibson | ✓ | ✓ (rigid) | ✗ | 100 |
| **Habitat 3.0** | ✓ | ✓ (SMPL-X) | MK+VR | 140-250 (single) |

**Habitat 3.0优势**:
1. 唯一同时支持real robot + SMPL-X humanoid + HITL
2. 200个authored scenes (最多)
3. 多种control interfaces (Jt, IK, HL)

---

## 8. Emergent Behaviors - 最exciting的部分

### 8.1 Social Navigation Emergent Behaviors

- **Three-point turn**: 在narrow corridor中，robot做3-point turn让路给humanoid
- **Backing up**: 预测humanoid未来路径，提前back up避免collision
- **Yielding space**: 在doorway前主动yield

Backup-Yield Rate (BYR): ~0.15-0.24 across RL variants，说明这些行为是RL学出来的，不是explicitly programmed。

### 8.2 Social Rearrangement Emergent Behaviors

- **Task splitting**: 即使unseen partner，robot学会split task (RC ≈ 0.5)
- **Reactive replanning**: 当humanoid在1.5m内时abort当前skill，high-level policy replan
- **Backward motion to let pass**: Fig. 10 frame 2展示robot后退让humanoid通过hallway

---

## 9. Limitations & Future Directions

### 9.1 Human Simulation Limits

- 只有walking + reaching behaviors
- 在place rotation有visual artifacts
- 固定LBS weights → 某些pose有skinning artifacts
- 无法处理opening cabinet, sitting down等复杂motion

### 9.2 HITL Gap

- Human通过HITL tool获得的observations vs automated humanoid的observations有gap
- 提供visual cues (markers)弥补，但observation space仍然不同

### 9.3 Task Limits

- No communication between robot and human
- Ground truth object locations已知
- Objects都在open receptacles上
- 不处理articulated objects (drawers等)

---

## 10. 我的Intuition Building

### 10.1 为什么这篇paper重要？

1. **Paradigm shift**: Embodied AI从single-agent到multi-agent social
2. **Scale matters**: 1190 FPS意味着可以跑大规模RL训练
3. **SMPL-X + caching**: 让realistic human simulation在RL训练中可行
4. **Population-based ZSC**: 提供了human-robot collaboration的training paradigm
5. **HITL validation**: 自动eval的ranking在real human中保持，说明simulator fidelity够用

### 10.2 与相关工作的联系

- **SMPL** (Bogo et al.): 同一parametric body family
- **DDPPO** (Wijmans): Habitat系列一直用的RL算法
- **Population Play** (Jaderberg, AlphaStar): 把game AI的技术用到embodied AI
- **Overcooked-AI** (Carroll): 类似ZSC setup但是2D grid
- **VirtualHome** (Puig): 之前的humanoid simulator但没有robot
- **Habitat 2.0** (Szot): rearrangement task的predecessor
- **Adaptive Coordination** (Szot 2023): social rearrangement的前身，但是两个robots

### 10.3 真正的技术亮点

1. **Caching strategy**: 看似简单但是1190 FPS的关键
2. **Hierarchical skills**: 让high-level policy不需要学low-level control
3. **Plan-Pop设计**: 通过object preferences强制diversity，比random init好
4. **GLMM statistical analysis**: 严肃处理human study的learning effect

### 10.4 可以深入思考的问题

1. **Learn-Pop为什么失败？** Random init不够，需要explicit diversity dimensions (speed, style等)
2. **Learn-Single在HITL反而比Plan-Pop3好** - 为什么ZSC-pop最差的在real human中最好？可能因为real human更reactive，更像Learn-Single的single partner。
3. **Communication的潜力**: 没有communication的ZSC已经能达到71% SR，加上language应该更高 - 联系到TEACH (Padmakumar 2021)等工作
4. **Sim-to-real gap**: Spot robot已经real deployment，Habitat 3.0训练的policy能否transfer到real Spot + real human？
5. **Foundation model integration**: VIMA, RT-2等robot foundation model能否在这个setup中work？

---

## 11. Reference Links

- Paper: http://aihabitat.org/habitat3
- Habitat-sim GitHub: https://github.com/facebookresearch/habitat-sim
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- AMASS: https://amass.is.tue.mpg.de/
- DDPPO: https://arxiv.org/abs/1911.00357
- Population Play (AlphaStar): https://arxiv.org/abs/1904.05754
- HSSD dataset: https://huggingface.co/datasets/hssd/hssd-hab
- Spot robot: https://www.bostondynamics.com/products/spot
- Overcooked-AI: https://arxiv.org/abs/1910.05790
- TEACH: https://arxiv.org/abs/2110.00500
- VPoser (SMPL-X VAE): included in SMPL-X
- YCB dataset: http://www.ycbbenchmarks.com/

---

希望这个深度解析帮你build了对Habitat 3.0的intuition。最让我excited的是population-based ZSC training + HITL validation的组合，这为future human-robot collaboration research提供了solid的infrastructure。Learn-Pop失败是个interesting negative result，提示我们需要更principled的diversity generation方法。HITL中Learn-Single反超Plan-Pop3也是个puzzle，可能与human的reactivity特性有关。
