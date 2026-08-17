---
source_pdf: DexHandDiff.pdf
paper_sha256: 18f96a26dbf11149d5677ce72bdbecc8162536e18f795dd69e368013d98447b8
processed_at: '2026-08-03T20:31:01-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲DexHandDiff

## 这篇paper到底在搞啥

想象你教机器人用手开门。你给它看了100遍开门到90度的演示，然后说："现在给我开到30度。"

**Diffusion Policy（现在最火的方法）的反应**：懵了。它只会replay动作，不知道"30度"是啥意思。动作序列是从头到尾一条龙，中间没有"状态"这个概念可以介入。就像你只会照着菜谱做菜，换个食材就不会了。

**Diffuser（更早的方法）的反应**：它能理解"30度"这个目标，因为它生成的是状态序列。但问题来了——它生成的画面里，门自己就转起来了，手还没碰到门把手，门已经开了30度。这在物理上根本不可能。这就是paper说的"ghost states"（幽灵状态）。

**DexHandDiff的做法**：同时生成手的状态+动作+门的状态，但用物理规律把它们绑在一起。手没碰到门把手之前，门不许动；碰到了之后，手和门一起被引导到目标。还会先让手对准门把手，再开始拧。

## 为什么这个"幽灵状态"问题难

核心矛盾在于：

**你想要灵活性** → 需要在"状态空间"里操作，这样可以直接说"目标是30度"
**但状态空间里有你控制不了的东西** → 门的角度、笔的朝向，这些不是你能直接操纵的DoF

在简单的gripper任务里（比如抓个方块），所有状态都是你能直接控制的——机械臂位置就是方块位置。所以纯state diffusion没问题。

但灵巧手不一样。门的角度是通过手指接触门把手、施加力矩间接改变的。如果你直接在状态空间里generate"门到30度"，模型根本不管手的动作，就把门的状态硬拉过去了。render出来看就像闹鬼。

## 三个关键设计的直觉

### 1. 为什么要joint state-action diffusion

单独diffusion state → 灵活但会闹鬼
单独diffusion action → 不闹鬼但不灵活

那就两个一起diffusion呗。状态告诉我"要去哪"，动作告诉我"怎么去"，两个在同一个trajectory里互相约束。

但光joint还不够，你得保证生成的state和action在物理上是一致的。这就是dynamics model的作用——它学了一个 $s_{t+1} = \mathcal{T}(s_t, a_t)$ 的预测器，如果生成的state-action pair违反这个物理规律，就penalize。

### 2. 为什么要分两个阶段

你开门的过程本质上是两段：
- **阶段一**：手伸向门把手，门不动
- **阶段二**：手抓住把手开始转，手和门一起动

这两个阶段的物理完全不同。阶段一里门是独立的，手是独立的，手只要reach到门把手就行。阶段二里手和门变成一个耦合系统，你动门把手，门才动。

如果用single-phase guidance，要么在阶段一就过早影响门的状态（又闹鬼了），要么在阶段二guide得不够（够不到目标）。

DexHandDiff用一个简单的距离判断来切换阶段：手掌和门把手的距离小于0.1就认为接触了。接触前只guide手reach，接触后guide手和门一起到目标，同时约束门的变化不能太突然。

### 3. 为什么用LLM生成guidance function

每个任务的guidance function都不一样：
- Door: 对准门把手 → 门转到目标角度 → 约束门角速度
- Pen: 笔的朝向对准目标 → 鼓励手指动 → 约束笔的pose变化
- Hammer: 对准锤柄 → 敲钉子到指定深度 → 约束锤子运动

手写这些function很烦，每个任务大概要试20次才能调好。用LLM的话，你给它环境的结构化描述（哪些维度是门角度、哪些是手位置、哪些是锤子pose），加上task instruction，它就能写出来。大概5次就能调好。

效果上LLM比human略差一点（Door 40% vs 70%），但考虑到省了大量人力，这是值得的trade-off。而且随着LLM更强，这个gap会缩小。

## 效果到底多好

最核心的数字：

**Goal adaptation（训练只见过90度开门）**：
- 开30度：DexHandDiff 70% vs 其他方法最高16.7% → **4倍**
- 平均：59.2% vs 次优29.5% → **2倍**

**跨任务（10个task平均）**：
- DexHandDiff 70.7% vs Diffusion Policy 58.0% vs Diffuser 34.0%

**幽灵状态减少**：predicted state和真实simulation的L2距离，DexHandDiff几乎比baseline小一半

**速度**：5-7 Hz，用DPM-Solver++加速能到36 Hz，够用

## 代价是什么

1. **In-domain性能略降**：90度任务上，DexHandDiff 90%，而Diffusion Policy 100%。因为多了一堆adaptability guidance，在training distribution内反而不如纯action replay精确。去掉这些guidance能达到96.7%。这是合理的trade-off。

2. **需要训dynamics model**：额外的一个MLP，但很轻量（总模型3.96M参数）。

3. **需要手写/LLM写guidance function**：虽然LLM自动化了，但还是要调prompt、跑几次iteration。

## 我的intuition总结

这篇paper的漂亮之处在于它**精确诊断了问题**：

1. Ghost states不是bug，是state-only diffusion的inherent limitation——你把不可控DoF当成可控的来generate，物理必然不对
2. Action-only的adaptability差也不是bug，是缺少中间状态representation的必然结果——纯action sequence没法在中间步介入goal condition
3. Solution就是把两个拼起来，但关键是**保持物理因果性**：手先到接触点，然后通过接触影响物体

这个dual-phase的设计特别elegant，因为它mirror了物理本身——manipulation就是有pre-contact和post-contact两个regime，硬要混在一起就是不对。

LLM guidance generation是bonus，让方法更实用，但核心insight还是joint state-action diffusion + interaction-aware guidance。

---

想看具体visualization的话，https://dexdiffuser.github.io/ 有那些逐帧对比，特别是ghost state的对比图，看一眼就知道我在说什么了。

---

# DexHandDiff: Interaction-aware Diffusion Planning for Adaptive Dexterous Manipulation 深度解析

Andrej, 这篇paper来自HKU和UC Berkeley的团队, 解决的是dexterous manipulation中diffusion-based planning的两个核心痛点: **ghost states** 和 **goal adaptability**。让我从intuition层面给你拆解。

## 1. 核心问题: 为什么现有diffusion planner在灵巧手上会失败?

### 1.1 Ghost States的本质

Table 1 对比了四种方法, 关键区别在于diffusion的对象和conditioning方式:

| Method | Diffusion on | Condition Type | Action Gen | Goal Adapt | No Ghost | Interaction Aware |
|--------|-------------|----------------|------------|------------|----------|-------------------|
| Diffuser | State | Classifier-Guided | Inverse Dyn | ✓ | ✗ | ✗ |
| Decision Diffuser | State | Classifier-Free | Inverse Dyn | ✗ | ✗ | ✗ |
| Diffusion Policy | Action | Classifier-Free | Direct | ✗ | ✓ | ✗ |
| **DexHandDiff** | **State & Action** | **Classifier-Guided** | **Direct** | **✓** | **✓** | **✓** |

Ghost states的根源在于**causality break**。在contact-rich interaction中, object state不是直接可控的DoF, 必须通过hand state的transition间接影响。当你用state-only diffusion生成所有states(包括object), 模型会generate physically impossible的sequences——比如pen自己旋转, door自动打开。

Figure 2 展示的pen reorientation案例特别直观: state-based diffusion生成trajectory中, pen看起来autonomously旋转到target pose, 手指最后才move到grasp position。这在物理上impossible, 因为pen的运动必须由finger contact驱动。

### 1.2 Action-only的adaptability瓶颈

Diffusion Policy这类action-only方法虽然在in-domain task上precision高(Door 90°达到100%), 但面对goal shift就崩了。Table 2 显示:

- Open 30°: DP只有16.7%
- Open 50°: DP只有3.3%
- Close Door: DP直接0%

原因很清晰: action trajectory是end-to-end mapping, 没有explicit state representation可供conditioning。你想把90°的data adapt到30°, 没有intermediate state guidance告诉model "到30°就停"。

### 1.3 Classifier-free vs Classifier-guided的trade-off

Classifier-free把task variation encode在model里, 好处是不需要external classifier, 坏处是**bound by training data distribution**。你想zero-shot adapt到training没见过的goal configuration, 它做不到。

Classifier-guided通过gradient-based guidance直接conditioning reward/goal, 理论上可以adapt到任意新goal。代价是guidance function需要manual design——这正是DexHandDiff用LLM解决的部分。

## 2. 核心方法: Joint State-Action Diffusion

### 2.1 为什么joint diffusion是正解

DexHandDiff的key insight: **state和action必须jointly diffuse, 同时maintain causal coupling**。

trajectory定义为:
$$\tau = [(a_0, s_0), (a_1, s_1), ..., (a_T, s_T)]$$

其中state $s$ 包含hand state (24 joint angles + 3 position offsets) 和object state (door hinge angle, pen pose等), action $a$ 只包含controllable DoFs (hand joints和positions)。

这个设计同时解决两个问题:
1. State在diffusion里 → 可以explicit conditioning和goal specification
2. Action也在diffusion里 → 避免inverse dynamics的误差累积, 同时state和action的coupling通过guidance维护

### 2.2 Extended Behavior Model和Energy Function

这是理论框架的核心。标准conditional diffusion是:
$$\tilde{p}_\theta(\tau) \propto p_\theta(\tau) p(\mathcal{O}_{1:T}=1 | \tau)$$

DexHandDiff generalize为product of experts:
$$\tilde{p}_\theta(\tau) \propto p_\theta(\tau) \prod_{i=1}^{n} h_i(\tau)$$

每个expert $h_i$ 对应一个energy function $\varepsilon_i$:
$$h_i(\tau, c) = \frac{1}{\int e^{-\varepsilon_i(\tau, c)} d\tau} e^{-\varepsilon_i(\tau, c)}$$

其中 $c$ 是task-specific condition。这个formulation允许把多个guidance signals加性组合。

guidance gradient $g$ 分解为:
$$g = \nabla_\tau \log \prod_{i=1}^{n} h_i(\tau) = -\sum_{i=1}^{n} \nabla_\tau \varepsilon_i(\tau, c)$$

这意味着每个guidance term独立设计, 最终gradient是各项之和。在reverse diffusion step (Eq. 7)中, mean shift为 $\mu_\theta + \Sigma g$。

### 2.3 Dynamics-aware Consistency

Joint state-action diffusion的最大风险是state和action不一致。DexHandDiff用learned dynamics model作为soft constraint:

$$\varepsilon_{\text{dyn}}(\tau) = |s_{t+1} - \mathcal{T}(s_t, a_t)|^2$$

其中 $\mathcal{T}(s, a)$ 是单独训练的dynamics model。这个energy term在training时作为additional loss, 在inference时作为guidance term。

intuition: diffusion model可能generate任何看起来plausible的state-action pair, 但只有满足 $\mathcal{T}(s,a)$ 的才是physically realizable的。通过penalize violation, 强制generated trajectory符合observed dynamics。

## 3. Dual-Phase Guidance: 反映物理interaction的本质

这是DexHandDiff最elegant的设计, 直接mirror了dexterous manipulation的物理结构。

### 3.1 Phase transition的物理依据

$$\epsilon = \begin{cases} 
\epsilon_{\text{pre}} = \epsilon_{\text{align}} + \epsilon_{\text{dyn}} & \text{if } |s_{\text{hand}} - s_{\text{contact}}| > \delta_1 \\
\epsilon_{\text{post}} = \epsilon_{\text{succ}} + \epsilon_{\text{dyn}} + \epsilon_{\text{penalty}} & \text{otherwise}
\end{cases}$$

其中 $s_{\text{hand}}$ 是palm position, $s_{\text{contact}}$ 是object上的contact point (door latch, hammer handle等), $\delta_1$ 是小threshold。

**Pre-contact phase**: hand和object是decoupled的。hand只需要reach到contact point, object state不应该变化。guidance只focus on alignment。

**Post-contact phase**: hand和object变成coupled system。hand action通过contact影响object state, guidance需要jointly control两者。

这种分阶段设计直接反映了manipulation的motor control策略——先reach再manipulate, 跟human grasping的two-phase策略一致。

### 3.2 Physical Constraint Guidance

$$h_{\text{penalty}} \triangleq 1 - H(|s_{\text{obj}}^{t+1} - s_{\text{obj}}^t| - \delta_2)$$

其中 $H(\cdot)$ 是Heaviside step function, $\delta_2$ 是小threshold。

这个term的作用: penalize object state的abrupt变化。当 $|s_{\text{obj}}^{t+1} - s_{\text{obj}}^t| > \delta_2$ 时, $H=1$, $h_{\text{penalty}}=0$, energy为无穷大(实际实现用large penalty)。这强制object state连续变化, 防止ghost states。

通过Eq. 11转换, $\epsilon_{\text{penalty}}$ 变成Dirac delta function, 直接在constraint violation时set value。

### 3.3 In-hand Manipulation的简化结构

对于pen spinning这类in-hand task, object已经在手里, 没有明确的pre-contact phase。guidance简化为:

$$\epsilon = \epsilon_{\text{goal}} + \epsilon_{\text{finger}} + \epsilon_{\text{dyn}} + \epsilon_{\text{penalty}}$$

其中finger motion guidance:
$$h_{\text{finger}}(\tau, t) = H(|s_{\text{finger-joints}}^{t+1} - s_{\text{finger-joints}}^t| - \delta_3)$$

这要求finger joints有active movement, 防止object自己move而finger不动的ghost state。

## 4. LLM-based Guidance Generation

### 4.1 Text-to-Reward for Diffusers

这是paper的另一个重要contribution, 把Eureka/Text2Reward的paradigm extend到diffusion planning。

两阶段流程:
1. **Prompt generation**: 6-part template (function purpose, guidance structure, environment description, function prototype, task instruction, few-shot hints) + simulation documents → task-specific prompts
2. **Code generation**: LLM根据prompt写guidance function code

### 4.2 Environment Abstraction

paper设计了comprehensive Pythonic environment abstraction, 包括:
- `BaseEnv`: core components (hand, objects) + observation space
- `AdroitHand`: 28-DOF joint specification
- Supporting classes: Door, Handle, Pen, Hammer等, 含physical properties和state representations

这种structured abstraction让LLM能generate physically consistent的guidance function。

### 4.3 Few-shot Hints vs Examples

paper用few-shot hints而非direct examples。每个hint展示specific technique:
- Door: soft interpolation for targets (`interpolated_angle = (1-alpha)*current_angle + alpha*target_angle`)
- Pen: orientation similarity via normalized dot product
- Hammer: nail insertion displacement作为measure, constraint hammer qpos changes

### 4.4 实际效果

Table 5 的ablation:
- Naïve Guide (直接guide object到goal): Door 0%, Pen 20%, Hammer 20%
- Human Craft: Door 70%, Pen 40%, Hammer 46.7%
- LLM Gen: Door 40%, Pen 26.7%, Hammer 43.3%

LLM generation接近human expert在hammer上, 在door和pen上略低但still reasonable。更重要的是, human trial-and-error从~20次降到~5次。

## 5. 实验数据深度分析

### 5.1 Goal Adaptability (Table 2)

| Method | Open 30° | Open 50° | Open 70° | Open 90° | Open 110° | Close Door | Avg |
|--------|----------|----------|----------|----------|-----------|------------|-----|
| Diffuser (Inpaint) | 16.7 | 16.7 | 6.7 | 56.7 | 10.0 | 0 | 17.8 |
| Diffuser (Guided) | 10.0 | 26.7 | 10.0 | 63.3 | 6.7 | **60.0** | 29.5 |
| Decision Diffuser | 0 | 3.3 | 16.7 | **100** | 30.0 | 0 | 25.0 |
| Diffusion Policy | 16.7 | 3.3 | 13.3 | **100** | 3.3 | 0 | 22.8 |
| **DexHandDiff** | **70.0** | **56.7** | 53.3 | 90.0 | 26.7 | 58.3 | **59.2** |

关键观察:
- **30°任务**: DexHandDiff 70% vs 次优16.7%, **4x提升**。这说明classifier-guided的gradient guidance能有效steer到OOD goal。
- **Close Door**: 58.3%, 这是task reversal。Diffuser Guided 60%略高, 可能因为state inpainting对reversal更直接。
- **90° trade-off**: DexHandDiff 90% < DP/DD的100%。paper解释这是adaptability的代价——去掉adaptability guidance后能达到96.7%。这是reasonable trade-off, 因为in-domain性能略降换取显著OOD提升。
- **110°较低**: 26.7%, 可能因为超出training data range太远, learned dynamics override guidance。

paper还提到一个counterintuitive现象: goals closer to training data不一定higher success rate。70°任务有8/14 failures opened to 90° instead, 说明learned bias比distant goal更难correct。这暗示diffusion model的data prior很强, 需要更强guidance override。

### 5.2 Cross-task Performance (Table 3)

| Task | Diffuser | Conditional DP | DexHandDiff |
|------|----------|----------------|-------------|
| Door Open 90° | 56.7 | 100 | 90.0 |
| Door Open 30° | 16.7 | 16.7 | **70.0** |
| Pen Full Re-orien | 10.0 | 80.0 | **93.3** |
| Pen Half-side | 3.3 | 23.3 | **40.0** |
| Hammer Full Drive | 53.3 | 76.7 | **90.0** |
| Hammer Half Drive | 23.3 | 33.3 | **46.7** |
| Relocate Full | 56.7 | 96.7 | 96.7 |
| Relocate Half-side | 53.3 | 86.7 | **93.3** |
| Block Rotate-Z | 36.7 | 40.0 | **50.0** |
| Block Half-side | 30.0 | 26.7 | **36.7** |
| **Average** | 34.0 | 58.0 | **70.7** |

DexHandDiff在9/10个task上达到最高或并列最高, 平均70.7% vs DP 58.0% vs Diffuser 34.0%。

特别值得注意的是Pen Half-side: training只有right-hemisphere orientations, test要求left-hemisphere。DexHandDiff 40% vs DP 23.3%, 利用diffusion的multi-modality和anisotropy实现hemisphere跨越。

### 5.3 Ghost State Reduction (Table 4)

| Adapt Task | Diffuser | DexHandDiff | Reduction |
|------------|----------|-------------|-----------|
| Door 30° | 4.19 | 2.92 | 30% |
| Door 70° | 4.03 | 2.38 | 41% |
| Pen Half | 5.23 | 2.76 | 47% |
| Hammer Half | 4.01 | 2.41 | 40% |
| Relocate Half | 5.48 | 3.22 | 41% |

测量的是predicted state和simulated state的L2 distance (per dimension normalized)。DexHandDiff几乎halve了baseline的gap, 直接验证ghost state reduction效果。

### 5.4 Framework Ablation (Table 6)

| Method | Goal Guide | Dyn Guide | Joint S&A | Interact Mech | Overall SR |
|--------|------------|-----------|-----------|---------------|------------|
| No-guide | ✗ | ✗ | ✗ | ✗ | 24.1 |
| Diffuser | ✓ | ✗ | ✗ | ✗ | 27.5 |
| Dyn-guide | ✓ | ✓ | ✗ | ✗ | 27.5 |
| Joint S&A | ✓ | ✗ | ✓ | ✗ | 30.8 |
| Dyn+Joint | ✓ | ✓ | ✓ | ✗ | 31.7 |
| **DexHandDiff** | ✓ | ✓ | ✓ | ✓ | **67.5** |

关键insight:
- **Goal guidance alone**: 27.5%, 微提升, 说明naïve goal guidance不够
- **+Dynamics guidance**: 27.5%, 没提升! 说明dynamics model单独用没用, 必须配合joint structure
- **+Joint S&A**: 30.8%, 小提升, joint structure本身有帮助
- **+Interaction mechanism**: 31.7% → 67.5%, **2x提升!** 这是dual-phase guidance和physical constraint的核心贡献

这个ablation清楚显示: interaction-aware design是关键, 其他components是supporting role。

### 5.5 Efficiency (Table 7)

| Task | Door | Pen | Hammer | Relocate | Block |
|------|------|-----|--------|----------|-------|
| Freq (Hz) | 5.04 | 5.88 | 5.86 | 5.78 | 6.92 |

RTX 3090上, receding horizon 8 (door 32)。模型只有3.96M params, 3.27 GFLOPS。用DPM-Solver++可4x加速, command interpolation可达36Hz, 满足real robot control。

## 6. 技术细节补充

### 6.1 Network Architecture

- **Temporal U-Net**: 6 residual blocks, dual temporal convolutions + group norm + Mish activation
- **Timestep injection**: linear embedding, added after first conv in each block
- **Dynamics model**: 3-layer MLP, batch norm, ReLU, hidden dim 512

### 6.2 Training Configuration

- Adam optimizer, lr=2e-4, batch size 256
- 5×10^5 steps for all tasks
- **Predict denoised trajectory τ_0 directly** (而非noise ε), 这对classifier-free方法更友好
- Planning horizon: train T=32, inference T=8 (door/block) or T=32 (hammer/pen)
- Diffusion steps K=20
- Guidance scale α ∈ {500, 1000, 2000}, task-dependent

### 6.3 Guidance Function实现细节

以Door task为例, generated guidance function包含:

```python
# Phase 1: Pre-interaction
reaching_reward = -torch.mean(torch.norm(palm_pos - handle_pos, p=2, dim=2), dim=1)

# Phase 2: Post-interaction  
door_angle_diff = torch.norm(door_hinge_angle - target_door_angle, p=2, dim=2)
door_reward = -torch.mean(door_angle_diff, dim=1)

# Velocity constraint (smooth door movement)
door_velocity = (door_hinge_angle[:, 1:, 0] - door_hinge_angle[:, :-1, 0]) / self.dt
velocity_reward = -torch.norm(door_velocity, p=2, dim=1)

# Dynamics consistency
dyn_reward = self.cal_dyn_reward(state=normed_obs, action=normed_actions)

# Phase combination via grasp mask
total_reward = (1 - grasp_mask[:, 0]) * reaching_reward + \
               grasp_mask[:, 0] * (door_reward + velocity_reward + dyn_reward)
```

Adaptive scaling确保各reward term初始magnitude合理: reaching ~12, door ~30, dynamics ~1.2。

### 6.4 Pen Task的Orientation Similarity

```python
# Normalize vectors
pen_rotation_norm = pen_rotation / (torch.norm(pen_rotation, p=2, dim=-1, keepdim=True) + 1e-6)
target_rotation_norm = target_rotation / (torch.norm(target_rotation, p=2, dim=-1, keepdim=True) + 1e-6)

# Dot product similarity (higher = more aligned)
orientation_similarity = torch.sum(pen_rotation_norm * target_rotation_norm, dim=-1)
```

这比直接L2 distance on rotation更robust, 因为它measure方向一致性而非绝对距离。

### 6.5 Hammer Task的Multi-constraint

```python
# Nail insertion target (halfway = 0.04m)
target_insertion = 0.04 * torch.ones_like(nail_insertion)
insertion_reward = -torch.norm(nail_insertion - target_insertion, p=2, dim=1)

# Hammer joint smoothness
hammer_joint_pos_changes = torch.norm(obs[:, 1:, 27:33] - obs[:, :-1, 27:33], p=2, dim=2)

# Hammer position smoothness  
hammer_pos_changes = torch.norm(hammer_pos[:, 1:, :] - hammer_pos[:, :-1, :], p=2, dim=2)

# All gated by grasp mask
total_reward = total_reward + phase2_reward * grasp_mask.float()
```

## 7. Intuition总结与Limitations

### 7.1 核心Intuition

1. **Causal coupling matters**: object state不能独立generate, 必须通过hand state transition。Joint state-action diffusion + dynamics constraint maintain这个causality。

2. **Phase transition is physical**: manipulation有天然的pre/post contact分界, guidance design应该mirror这个physics。强行用single-phase guidance要么premature influence object, 要么insufficient goal steering。

3. **Gradient guidance enables OOD adaptation**: classifier-guided通过gradient直接steer到任意goal, 不受training distribution限制。这是vs classifier-free的根本优势。

4. **LLM automates domain knowledge encoding**: guidance function设计需要大量domain knowledge (哪个state维度对应什么物理量, 怎么组合reward terms)。LLM + structured environment description能reasonable自动化这个过程。

5. **Soft constraints > hard constraints**: physical consistency通过energy function softly enforce, 而非hard simulation constraint。这让diffusion process保持differentiable和efficient。

### 7.2 Limitations和Future Directions

1. **Vision-based perception**: paper提到future work需要vision model估计hand state和object pose。目前assume state直接observable, real deployment需要bridge这个gap。

2. **Inference speed**: 虽然DPM-Solver++加速, 但door task 5Hz可能不够high-frequency control。Real robot可能需要further optimization或model distillation。

3. **LLM reliability**: LLM guidance generation虽然reduce human effort, 但质量仍略低于human expert (Table 5)。More structured prompt或更powerful LLM可能improve。

4. **Generalization scope**: 只在5个task上验证, 更diverse的task和环境需要more evaluation。特别是contact更rich的任务如assembly, deformable manipulation。

5. **Learned bias override**: paper提到70°任务有8/14 failures opened to 90°, 说明learned data prior很强。可能需要adaptive guidance scale或更sophisticated的guidance schedule。

### 7.3 对field的影响

DexHandDiff标志着diffusion-based planning从simple gripper task向contact-rich dexterous manipulation的关键一步。几个重要implications:

1. **Joint state-action modeling成为standard**: 未来diffusion planner likely adopt这个structure, 特别是contact-rich场景。

2. **LLM+Diffusion synergy**: text-to-reward paradigm extend到diffuser guidance, 开启automated policy design的新方向。

3. **Interaction-aware design philosophy**: phase-based guidance design可能inspire更多physics-informed generative model。

4. **Benchmark for goal adaptation**: Door 30°/50°/70°/110°/close的setup提供systematic evaluation of OOD goal adaptation, 可能成为standard benchmark。

paper的code和visualization在 https://dexdiffuser.github.io/ , 值得看看那些visualization frame-by-frame, 特别是ghost state对比和goal adaptation的连续motion。

相关工作的reading list:
- [Diffuser](https://arxiv.org/abs/2205.09991) - Janner et al., ICML 2022
- [Decision Diffuser](https://arxiv.org/abs/2211.05707) - Ajay et al., ICLR 2023  
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) - Chi et al., IJRR 2023
- [Eureka](https://arxiv.org/abs/2310.12931) - Ma et al., 2023
- [Text2Reward](https://arxiv.org/abs/2309.11489) - Xie et al., ICLR 2024
- [Code as Policies](https://arxiv.org/abs/2209.07753) - Liang et al., ICRA 2023

这篇paper的beauty在于它精准诊断了diffusion planning在dexterous manipulation上的failure mode, 然后用principled的方法(addressing causality, phase transition, physical consistency)解决, 而非纯empirical hacking。LLM automation是pragmatic的bonus, 让方法更accessible。期待看到real robot deployment的follow-up work。
