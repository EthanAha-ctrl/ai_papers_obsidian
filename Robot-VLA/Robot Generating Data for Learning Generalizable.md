---
source_pdf: Robot Generating Data for Learning Generalizable.pdf
paper_sha256: 9887e02d43c828d663e0d2a22306d680201c9df43cc880969e7d90f784e5c3e3
processed_at: '2026-08-12T01:42:49-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 RST

## 一句话版本

让机器人自己给自己出题、自己解题、越做越难，最后把这套"自学笔记"蒸馏成一个能看图干活的视觉策略。

---

## 为什么需要这个

机器人学习现在想走 foundation model 路线——海量数据 pretrain，下游 zero-shot 泛化。但 robot data 跟 image/text data 有个根本区别：**它必须包含 valid action**。

收集一张猫的图片，网上扒就行。收集一段"机器人把水壶放到微波炉里再开灯"的 trajectory，你得有人 teleoperation，或者搞个昂贵的系统。而且组合爆炸——6 个积木能搭出来的结构是天文数字，人类不可能挨个演示。

所以大家都在想：能不能让机器人自己 generate 数据？难点在于：**你不能随便让 robot 乱动，乱动出来的 trajectory 没有训练价值**。推一块积木到桌上随机位置，是 valid trajectory，但对学"搭积木"毫无帮助。

RST 的 answer 是：让 robot 自己 decide "下一步该学什么 task"，而且这个 task 必须 **刚好在它能力边界上**——比已经会的难一点，但不是难到做不出来。

---

## 核心直觉：value 差 = "刚好够得着"

想象你在攀岩。你站在当前位置 $s_T$，想找下一个抓点 $g$。好的抓点应该满足两个条件：

1. **从你现在位置能够到**（reachable）
2. **从地面出发够不到**（challenging）

如果只满足条件 1，你选到的可能是地面就能直接够到的地方，没进步。如果只满足条件 2，你选到的可能是天花板，根本够不到，白费力气。

RST 用的 metric 就是这两个条件的 "代数差"：

$$g^* = \arg\max_g \underbrace{V_\varphi(s_T, g)}_{\text{从当前位置能到吗}} - \underbrace{V_\varphi(s_0, g)}_{\text{从起点能到吗}}$$

这里 $V_\varphi$ 是 universal value function，在 sparse reward goal-conditioned MDP 里，它近似 "从 state $s$ 出发到 goal $g$ 的成功概率"。所以这个差值大，意味着 "当前位置给我的 advantage 大"——我必须先走完原 trajectory，才有机会到 $g$。

这个 idea 其实就是 **asymmetric self-play** 的简化版：setter 给 solver 出题，题要 "刚好难到 solver 紧张但能做"。POET 用进化算法做这件事，RST 用 value function 的代数差做这件事，更直接更 cheap。

参考：
- POET: https://arxiv.org/abs/1901.01753
- Asymmetric self-play: https://arxiv.org/abs/1703.05407
- ACL (Automatic Curriculum Learning, Graves): https://arxiv.org/abs/1710.04852

---

## 三步走，人话版

### Step 1: 给个起点

先从一个小 dataset 起步，只包含 **最 basic 的 skill**：
- Block domain：scripted policy 移动单块积木
- Kitchen domain：把 D4RL 的 kitchen-partial 切成单组件交互的 short clip，relabel 成 $(s_i, g=s_j)$ 的 demo

用 BC + PPO 训一个 state-based goal-conditioned policy。这个 policy 只会做最简单的事，但它有 universal value function——这就够了，后面靠它自己 generate 越来越复杂的数据。

### Step 2: 自己给自己加难度

核心 loop：

```
repeat N rounds:
    拿出之前所有 successful trajectories
    对每条 trajectory (s0 → sT):
        在 sT 附近 perturb 一个 object 的状态，得到 candidate goals
        用 value 差 metric 选出 best goal g*
        让 policy 从 sT 往 g* rollout
        if 成功:
            concat 成新 trajectory (s0 → g*)
    用新 trajectories BC + PPO fine-tune policy
    把 rollout 存进 dataset
```

几个关键 design choice，讲讲为什么：

**为什么只 perturb 一个 object？** 因为这是 incremental 难度。一次只动一个东西，保证新 task 是 "旧 task + 一个新 skill" 的 composition，而不是 "完全无关的全新 task"。Paper 里的 ablation（Table II "w/o restr."）证明：random 采样 goal space，success rate 从 70% 暴跌到 6%。这个 domain knowledge bias 极其关键。

**为什么 BC + PPO 两段式？** BC 提供 good starting point（避免 PPO 从 random 探索 sparse reward），PPO 修正 BC 的 minor errors。每一轮都重置 PPO 起点，避免 catastrophic forgetting。

**为什么 value 差 metric 比 single-term metric 好？** Table II 消融：
- 只用 $V(s_T, g)$（end2new）：选到容易的 goal，没 challenge，数据没信息增益
- 只用 $-V(s_0, g)$（init2new）：选到不可达的 goal，policy rollout 失败，数据没用
- 两者差：刚好在 capability frontier 上，既 solvable 又 informative

### Step 3: 蒸馏成视觉策略

State-based policy 是 "老师"，它有 ground truth state 这个 privileged information。Visual policy 是 "学生"，只有 camera image。

老师在教学环境里自学了一堆复杂 skill，但 deployment 时学生只有眼睛看。所以把老师的 (state, action, goal) 换成，用 IL 训学生。

不同 domain 用不同 visual encoder：
- Block stacking：Slot Attention（object-centric representation，正好匹配 task 的 object-centric 结构）+ Transformer
- Kitchen：R3M（pretrained visual representation for robotics）
- Real world：Diffusion Policy + ResNet18 + heavy domain randomization

参考：
- Slot Attention: https://arxiv.org/abs/2006.11555
- R3M: https://arxiv.org/abs/2203.12601
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Learning by Cheating (teacher-student paradigm): https://arxiv.org/abs/1912.02988

---

## 实验结果，人话解读

### Block Stacking

6 个长方体积木，goal 是 target pose（可以在空中，需要底下有支撑）。Action 是 teleport（简化 setting，选一个 block 直接移到 target pose，参考 https://arxiv.org/abs/2103.07541）。Sparse reward：所有 target 都稳定达到才给 1。

3 轮 task expansion 后，visual policy zero-shot 测试三种人设计的结构（"I"、"3T"、"Y'"），success rate 28% / 42% / 15%。

对比 **Direct RL**：用同样 seeding dataset BC warm-start，然后直接 PPO 优化 evaluation goal distribution。137M timesteps 训练，0% 成功。Sparse reward + multi-object compositional task 是 hard exploration，直接优化根本起不来。

RST 的本质是把不可直接优化的 hard problem，拆成 agent 自己 generate 的渐进 curriculum，平滑掉 exploration 难度。

Figure 4 展示了 task 难度分布随 round 右移——agent 自动 discover 越来越高的 stack。Figure 5 可视化：round 1 单 "T" shape，round 2 multiple towers，round 3 复杂 stack。这是 open-ended complexity growth 的直接证据。

Real-world deployment 用 Franka Panda，visual policy 能在真实环境搭各种结构（Figure 3），包括桌面 rearrangement 和空中 stack。

### Franka Kitchen

D4RL kitchen-partial，7 个 component（M/K/L/B/T/S/H），9-DoF action。Seeding dataset 是单组件交互 short clip。

8 轮 task expansion 后，数据从 133k 涨到 1.799M（13.5× 扩张）。

Visual policy zero-shot 测试 2/3/4-stage compositional task：
- 2-stage（HM, MT）：99%, 93%
- 3-stage（HLM, BLM, LMST）：51%, 52%, 72%
- 4-stage（BKT, BLST, HLMT）：45%, 46%, 52%

对比 **FLAP**（planning-based offline RL，learn affordance + plan subtask）：
- 2-stage 有一些 success（HM 0%, MT 9%）
- 4-stage 全部 0%

FLAP 的 bottleneck 是 planner 能力——它得把长 horizon task 拆成 subtask，但 seeding dataset 太小，planner 学不好。RST 把 planning 问题转化成 flat policy 问题，policy 自己 internalize 了 long-horizon structure。

有意思的发现：visual policy 在某些 task（BKT）上比 state policy 还好（0.45 vs 0.00）。作者解释是 multi-task training over broad dataset 的 regularization effect——BC on broad dataset 平滑了 state policy 的 specific failure mode。

参考：
- FLAP: https://arxiv.org/abs/2211.11787
- D4RL: https://arxiv.org/abs/2004.07219

---

## 为什么这个 framework 本质上是 "software 2.0 的 self-improving compiler"

你（Karpathy）讲过 software 2.0 的视角：NN 是 differentiable program，我们用 gradient descent "编译" 数据进 weights。RST 在这个框架下是 **self-improving compiler**——它 generate 的不是 code，是 task + trajectory，用 differentiable metric（value 差）驱动 open-ended behavior generation。

更深一层：RST 是 "learning to learn" 的 instantiation。Policy 不只学 task，还学 "如何扩展自己能解决的 task space"。这接近 open-ended learning 的 ultimate goal——agent 持续 self-improve，没有 fixed target。

类比 biological evolution：生命从单细胞到多细胞到复杂器官，没有 external designer，靠 "刚好够得着的变异 + 自然选择"。RST 的 task expansion 是 "刚好够得着的新 task + value-based selection"，mechanism 不同但 spirit 相似。

参考 open-ended learning：
- POET: https://arxiv.org/abs/1901.01753
- Enhanced POET: https://arxiv.org/abs/2010.04704
- PAIRED: https://arxiv.org/abs/2106.04823

---

## 和其他思路的对比，人话版

### vs HER (Hindsight Experience Replay)
HER 是 "走完之后回头看，假装我本来就想去那"。RST 是 "往前看，预测我下一步该去哪"。HER 最大化数据利用，RST 主动探索 new task。理论上可结合——HER relabel RST 的失败 rollout。
- HER: https://arxiv.org/abs/1707.01495

### vs LLM-grounded methods (SayCan, Inner Monologue)
LLM methods 把复杂 task 拆成简单 skill，用 LLM 当 planner。问题：planner 能力有限，4-stage task 拆不好就全崩。RST 把 planning 内化进 flat policy，policy 自己就 handle long-horizon。
- SayCan: https://say-can.github.io/
- Inner Monologue: https://arxiv.org/abs/2207.05608

### vs BOSS (Bootstrap Your Own Skills)
BOSS 用 LLM chain language-conditioned task。RST 用 visual goal，更适合 stacking 这种 language 难描述的精细 task。
- BOSS: https://arxiv.org/abs/2310.18215

### vs Curriculum Learning (CL)
传统 CL 有 target task prior，从易到难 sample subgoal。RST 没有 target prior，open-ended discover task complexity frontier。RST 是 unsupervised CL。
- Reverse Curriculum (Florensa): https://arxiv.org/abs/1709.02759
- Automatic Goal Generation: https://arxiv.org/abs/1805.04880

### vs Foundation Model philosophy
Foundation model 靠 scale data + scale model。RST 靠 scale task via self-generation。两者未来会 converge——VLA foundation model 提供 seeding policy，RST-style task expansion 在 deployment 时持续 self-improve。
- RT-2: https://robotics-transformer2.github.io/
- RT-X: https://arxiv.org/abs/2310.08864

---

## 当前 limitations，不回避

1. **只解决 compositional generalization**：其他 aspect（光照变化、object shape variation）需要额外 data augmentation。作者自己承认这点。
2. **Object-centric 假设**：task expansion 的 "perturb one object" bias 假设环境是 object-centric。Deformable object、fluid、articulated object 需要新 mechanism。
3. **Block domain 的 teleport action 是简化**：不直接 transfer 到 continuous control robot。Paper 里 real-world 部分用 Diffusion Policy 替代，但 task expansion 仍在 simulator 中跑。
4. **Seeding dataset 仍是 cold-start bottleneck**：虽然小，但还得人 collect。
5. **Value function approximation 误差**：UVF 在 long-horizon 下误差大，可能选错 goal。Paper 没讨论 sensitivity。
6. **Simulator 依赖**：task expansion 需要 reset，所以必须在 simulator 中跑。Real-world task expansion 是 open problem。

---

## 如果让我赌未来方向

**RST + VLM-grounded goal proposal + diffusion-based policy distillation** 是下一代 robot foundation model 的关键 recipe。

当前 RST 的 "random one-object perturbation" 是最 manual 的版本。未来用 VLM 提出 "语义上合理但 policy 还不会做" 的 goal（比如 "把水壶放进微波炉再开底火"），再用 RST 的 UVF-metric 筛选——VLM 提供 semantic imagination，RST 提供 capability-grounded filtering，两者互补。

更激进一点：**real-world task expansion**。当前 task expansion 在 sim 里跑因为需要 reset。如果用 world model predict $s_T$ 后的 rollout（Dreamer-style），减少 sample cost，就有可能在 real environment 里跑 task expansion loop。
- Dreamer: https://arxiv.org/abs/1912.01603
- SuSIE (semantic imagination for robot data): https://arxiv.org/abs/2302.11550

---

## 一句话 takeaway

RST 把 "robot data collection" 这个 expensive 问题，转化成 "robot self-curriculum" 这个 cheap 问题，用 value function 的代数差当 progression metric，让 robot 自己 compose 越来越复杂的 task。它是 open-ended learning 在 robotics 的第一个 practical instantiation，也是 robot foundation model 自我进化的 prototype。

核心 insight：**最好的训练数据不是人 collect 的，是 robot 在自己能力边界上自己 generate 的**。

---

# Robot Self-Teaching: 让机器人自己生成训练数据

## 1. 这篇 paper 在解决什么问题

Robot learning 想做 pretraining-based foundation model，但 robot trajectory 数据采集成本远高于 image/text，因为它必须包含 valid control actions。当前主流方案是 human teleoperation 收集数据，遇到 long-horizon compositional tasks（如多物体堆叠、kitchen 多组件操作）时，组合爆炸使 human demonstration 不可行。

RST (Robot Self-Teaching) 提出：让机器人自己 generate increasingly complex tasks 和对应 trajectories，最后把这些自生成数据蒸馏成 generalizable visual policy。

核心直觉类似 **POET (Wang et al., 2019)** 中的 open-ended environment co-evolution，但用 universal value function 的"代数差"作为 progression metric，替代了进化算法：
- https://arxiv.org/abs/1901.01753 (POET)
- https://arxiv.org/abs/2010.04704 (Enhanced POET)

---

## 2. 三阶段框架解析

### Stage A: Warm-Start from Seeding Dataset

Seeding dataset 只包含 **basic skills**：
- Block-stacking domain：scripted policy transport 单个 cuboid
- Franka kitchen：把 "kitchen-partial" 切成单组件交互的 short chunks，relabel 成 $(s_i, g = s_j)$ demo

训练流程：
1. BC 预训练 policy $\pi_\theta(a|x, g^{(x)})$ 得到 good starting point
2. PPO 在 seeding tasks 上 refine，robustify policy

数学形式（公式 1）：
$$L_{\mathrm{bc}} = \mathbb{E}_{(x, a, g^{(x)}) \sim \mathcal{D}}[-\log \pi_\theta(a | x, g^{(x)})]$$

变量解释：
- $x$：input，state $s$ 或 image $o$
- $a$：action
- $g^{(x)}$：goal in input space（state-goal $g^{(s)}$ 或 image-goal $g^{(o)}$）
- $\mathcal{D}$：goal-conditioned demonstration dataset
- $\theta$：policy 参数
- $\pi_\theta(a|x, g^{(x)})$：goal-conditioned stochastic policy

PPO reference: https://arxiv.org/abs/1707.06347

### Stage B: Task Expansion（核心创新）

给定一个 successful trajectory $\tau = (s_0, \ldots, s_T)$，目标是把它 extend 成更复杂的 trajectory $(s_0, \ldots, s_T, \ldots, g^*)$。

#### 关键公式（公式 2）

$$g^* = \arg\max_g V_\varphi(s_T, g) - V_\varphi(s_0, g)$$

变量解释：
- $g^*$：选出的 best new goal
- $g$：候选 goal state（从 goal space 中采样）
- $V_\varphi(\cdot, \cdot)$：universal value function，参数 $\varphi$，输入 state-goal pair，输出 expected discounted return
- $s_T$：原成功 trajectory 的 terminal state
- $s_0$：原 trajectory 的 initial state

#### 这个 metric 为什么 work——intuition building

在 sparse 0/1 reward goal-conditioned MDP 中，$V_\varphi(s, g)$ 近似"从 $s$ 出发到 $g$ 的可达概率（折扣期望回报）"。所以 $V_\varphi(s_T, g) - V_\varphi(s_0, g)$ 有清晰语义：

- $V_\varphi(s_T, g) \uparrow$：$g$ 从原 trajectory 终点 **容易到达**（reachable）
- $V_\varphi(s_0, g) \downarrow$：$g$ 从原 trajectory 起点 **难以到达**（challenging）

最大化两者差 = 选一个"先得走完原轨迹才能解决"的新目标，正是 compositional extension 的定义。

直觉上可以把它理解为 value gradient 上的 "advantage for continuation"：当前 trajectory 给我创造了到达 $g$ 的 advantage，而这个 advantage 在 $s_0$ 处还不存在。这与 **HER (Hindsight Experience Replay)** 中的 relabeling 哲学相反——HER 是 forward 后 backtrack relabel goal；RST 是 forward predict 哪个 goal 该被 compose 上去。
- HER: https://arxiv.org/abs/1707.01495

#### 为什么不能只用单边 metric

Table II 的消融极其关键：

| Method | I sr. | 3T sr. | Y' sr. |
|---|---|---|---|
| RST w/o restr. | 6.6±3.8 | 0.7±0.4 | 0.2±0.1 |
| RST init2new ($\arg\min_g V_\varphi(s_0, g)$) | 14.5±8.1 | 0.1±0.1 | 0.4±0.3 |
| RST end2new ($\arg\max_g V_\varphi(s_T, g)$) | 13.4±9.7 | 2.5±4.2 | 1.0±0.9 |
| **RST full** | **70.8±18.6** | **44.3±6.7** | **8.0±7.0** |

- **init2new**：只选"从起点难到"的 goal，可能选到完全不可达的 goal，policy rollout 失败，数据没用。
- **end2new**：只选"从终点可达"的 goal，可能选到从起点也很容易到达的 goal（如把水壶推回去），task expansion 没有 "增加难度"，没有信息增益。
- **Full metric**：两个条件 jointly 满足，task 既 solvable 又 informative。

这印证了 **asymmetric self-play / setter-solver** 的核心教训：setter 必须给 solver "恰到好处"的 challenge，过易或过难都 collapse。
- Asymmetric self-play (Sukhbaatar et al.): https://arxiv.org/abs/1703.05407
- ACL Graves et al.: https://arxiv.org/abs/1710.04852

#### Domain-knowledge 偏置（关键 implementation trick）

"When sampling candidate goal states during task expansion, we incorporate minimal domain knowledge of object-centric environments, by only perturbing the states of one object in the original terminal state to create goal states."

这个 bias 极其重要——它强制 task 是 **incremental**（每次只动一个 object/component），保证 composability。如果不限制（"w/o restr." 消融），random 采样 goal space 会产生几乎不可解的多物体扰动 task，导致 success rate 暴跌（Table II 第一行）。

#### Iterative 扩展循环

```
for round in 1..N:
    for each successful trajectory (s0 → sT):
        sample candidate goals g (one-object perturbation)
        select g* = argmax V(sT, g) - V(s0, g)
        rollout policy from sT to g*
        if success:
            concat → new trajectory (s0 → g*)
    BC on expanded trajectories
    PPO fine-tune on expanded tasks
    collect rollouts into dataset D_round
```

每轮 policy 都变得更 skillful，下轮能 generate 更 complex task，形成 **open-ended curriculum**。

### Stage C: Visual Policy Distillation

把 state-based policy 当 **teacher**（privileged information），visual policy 当 **student**。Teacher-student / privileged learning paradigm：
- https://arxiv.org/abs/1912.02988 (Learning by Cheating)

不同 domain 用不同 visual encoder：
- **Block-stacking**：Slot Attention（object-centric）+ Transformer + BC
  - Slot Attention 用 unsupervised reconstruction loss 在所有 generated images 上预训练并 freeze
  - https://arxiv.org/abs/2006.11555
- **Kitchen**：R3M 预训练 encoder + IL
  - https://arxiv.org/abs/2203.12601
- **Real-world deployment**：Diffusion Policy + ResNet18 + intensive domain randomization
  - https://arxiv.org/abs/2303.04137

---

## 3. 实验数据深度解读

### Block-Stacking Domain

Task 设定：
- 最多 6 个 cuboid blocks
- Goal = target poses of subset of blocks（可在空中，需 stack 支撑）
- Action = 直接 teleport 一个 object 到 desired pose（simplified discrete setting，类似 https://arxiv.org/abs/2103.07541）
- Sparse reward：only when all targets reached stably

#### Table I：累积轮数对 zero-shot 的影响

| Data source | I sr. | 3T sr. | Y' sr. |
|---|---|---|---|
| round 1 | 0.7±1.2 | 4.0±1.7 | 0.0±0.0 |
| round 2 | 0.7±0.6 | 2.0±1.0 | 0.0±0.0 |
| round 1,2 | 9.7±1.5 | 19.0±4.4 | 1.6±0.6 |
| round 3 | 10.0±3.6 | 25.3±5.8 | 1.3±0.6 |
| **round 1,2,3** | **28.0±1.7** | **41.7±4.0** | **15.0±3.0** |
| Direct RL | 0.0 | 0.0 | 0.0 |

几个关键观察：
1. **单轮数据不够**：仅 round 1 或 round 2 单独训练效果差，因为数据复杂度太低（多数只是单 block 移动）。
2. **累积效应**：round 1+2 比 round 1 或 round 2 单独大一个数量级，因为数据多样性 + curriculum signal 联合起作用。
3. **Direct RL 完全 fail**：137M timesteps PPO 训练 0% 成功。说明 sparse reward + multi-object compositional task 是 hard exploration problem。RST 的本质是 **automatic curriculum smoothing**，把不可直接优化的 hard problem 拆解成 agent 自己能 generate 的 sub-task 序列。

#### Figure 4：生成 task 难度分布

随 round 增加，maximum height of target positions 分布右移——agent 自动 discover 越来越高的 stack 结构。这是 open-endedness 的直接证据。

#### Figure 5：可视化生成 task 复杂度递增

- Round 1：单 "T" shape
- Round 2：multiple towers
- Round 3：更复杂 stack 结构

这非常像 **POET 的 environmental complexity growth**，但 mechanism 是 value-based 而非 evolution-based。

### Franka Kitchen Domain

Environment：D4RL "kitchen-partial"
- 7 components: M (microwave), K (kettle), L (light), B (bottom burner), T (top burner), S, H (cabinets)
- 9-DoF action (robot joints)
- 8 rounds of task expansion
- https://arxiv.org/abs/2004.07219 (D4RL)

#### Table III：Compositional Generalization

| Method | HM | MT | HLM | BKT | BLM | LMST | BLST | HLMT |
|---|---|---|---|---|---|---|---|---|
| **RST (visual)** | **0.99±0.01** | **0.93±0.03** | **0.51±0.02** | **0.45±0.10** | **0.52±0.04** | **0.72±0.05** | **0.46±0.22** | **0.52±0.15** |
| FLAP | 0.00 | 0.09±0.01 | 0.00 | 0.00 | 0.01±0.01 | 0.00 | 0.00 | 0.00 |
| State @ 1st rd. | 0.31±0.43 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| State @ 4th rd. | 0.45±0.33 | 0.99±0.01 | 0.24±0.26 | 0.00 | 0.07±0.10 | 0.33±0.20 | 0.47±0.30 | 0.41±0.22 |
| State @ 8th rd. | 0.97±0.02 | 0.99±0.00 | 0.43±0.23 | 0.00 | 0.91±0.06 | 0.70±0.15 | 0.50±0.24 | 0.80±0.09 |

非常 interesting 的几个点：

1. **FLAP collapse on 4-stage tasks**：FLAP 是 planning-based method（learn affordance + plan subtasks），在 4-stage task 上 0%。这印证了 "planner capability bottleneck"——LLM-based 或 affordance-based planner 无法精确分解长 horizon 问题。
   - FLAP: https://arxiv.org/abs/2211.11787
2. **State policy 也有 curriculum 效应**：1st round 只能做 HM，4th round 能做更多，8th round 几乎全开。
3. **Visual policy 在某些任务上超过 state policy**：BKT 任务 visual 0.45 vs state @ 8th 0.00。作者解释为 "multi-task training over the generated broad dataset"。这是 distillation 的 regularization bonus——BC on broad dataset 平滑了 state policy 的 specific failure modes。
4. **8 轮 vs 1 轮的数据增长**：Table IV 显示从 133k 增长到 1.799M，约 13.5× 扩张。

#### Figure 7：Task complexity growth

随 round 增加，需要 manipulate 4+ components 的 task 数量从 0 增长到几百。Open-ended curriculum 自动 emerge。

---

## 4. 与其他方法的对比

### RST vs HER (Hindsight Experience Replay)
- HER：trajectory rollout 后 backtrack relabel goal = achieved state，最大化数据利用
- RST：forward predict which goal to compose next，主动探索 new task
- Complementary：理论上可结合，HER relabel RST rollouts

### RST vs LMP (Latent Motor Plans, Lynch et al.)
- LMP：从大量 play data 学 latent plan，goal-conditioned at test time
- RST：generate structured curriculum data，不需要海量 unstructured play
- https://arxiv.org/abs/1910.07569 (LMP)

### RST vs BOSS (Bootstrap Your Own Skills)
- BOSS：LLM 链接 language-conditioned tasks，自然语言 caption
- RST：visual goal specification（更精确，适合 stacking 这种 language 难描述的 task）
- https://arxiv.org/abs/2310.18215 (BOSS)

### RST vs BOSS / LLM-grounded methods (SayCan, Inner Monologue)
- LLM methods：bypass 复杂 policy learning，用 LLM 做高层 planning + 简单 skill policy
- RST：self-augment data into complex policy，最终得到 flat policy（不需 test-time planning）
- https://say-can.github.io/ (SayCan)
- https://arxiv.org/abs/2207.05608 (Inner Monologue)

### RST vs Go-Explore
- Go-Explore：archive promising states，回访后继续 explore
- RST：archive successful trajectories，从 terminal state continue to compose new task
- 类似 spirit，不同 mechanism
- https://arxiv.org/abs/1901.10995 (Go-Explore)

### RST vs Curriculum Learning (CL)
- CL：有 prior target task，sample sub-goal/initial state 由易到难
- RST：open-ended，没有 target task prior，agent 自己 discover task complexity frontier
- RST 是 unsupervised CL，与 Florensa et al. (Automatic Goal Generation) 类似但更激进
- https://arxiv.org/abs/1709.02759 (Reverse Curriculum)
- https://arxiv.org/abs/1805.04880 (Automatic Goal Generation)

### RST vs POET / PAIRED
- POET：进化 algorithm co-evolve environment & agent
- PAIRED：adversarial environment design
- RST：value-based metric 做 environment/task design
- https://arxiv.org/abs/2010.04704 (Enhanced POET)
- https://arxiv.org/abs/2106.04823 (PAIRED)

---

## 5. 关键技术细节与未明说的设计选择

### Universal Value Function 的关键性
UVF (Schaul et al., 2015) 让 RST 能 query **arbitrary** $(s, g)$ pair 的 reachability，无需 retrain。这是 task expansion 的前提——没有 UVF，每 generate 一个 candidate goal 都得重新 evaluate reachability，计算量爆炸。
- https://proceedings.mlr.press/v37/schaul15.html (UVF)

### State-space vs Visual-space for task expansion
为什么 task expansion 在 state space 而非 image space？因为：
1. State 低维，value function 学得准
2. State 是 object-centric，可逐 object perturb
3. Image space 中 "perturb one object" 需要 generative model，引入额外误差

这正是 teacher-student / privileged learning 的标准操作。

### BC + PPO 的两段式 fine-tune
- BC 先 warm-start（避免 PPO 从随机 start 探索）
- PPO 再 refine（修正 BC 的 minor errors）

这种 staged fine-tune 在 robotics 中是 standard practice，但 RST 在每一轮 task expansion 后都重复这个流程，**不断重置 PPO 起点**——这避免了 PPO catastrophic forgetting 之前 task 的能力。

### Domain randomization for sim2real
Real-world deployment 用 intensive texture/lighting/camera-pose randomization，类似 RCAN / RL with randomization。
- https://arxiv.org/abs/1801.00604 (RCAN, Tobin et al.)

---

## 6. Limitations 与 Future Directions

### 当前 limitations
1. **Compositional generalization only**：作者明确指出 RST focus on compositional generalization，其他 aspect（光照、object shape variation）需要额外 data augmentation（如 [51] semantic imagination）。
2. **Object-centric assumption**：task expansion 的 "perturb one object" bias 假设环境是 object-centric。Non-object-centric tasks（如 deformable object、fluid）需要新 mechanism。
3. **Discrete teleport action** in block-stacking：简化 setting，不直接 transfer 到 continuous control robot。
4. **Seeding dataset 仍需人类/scripted**：虽然规模小，但仍是 cold-start bottleneck。
5. **Value function approximation 误差**：UVF 在 long-horizon 下误差大，可能选错 goal。Paper 未讨论这个 sensitivity。

### 自然 extension 方向

1. **Foundation model integration**：用 VLM / video generation model propose candidate goals，替代 random sampling + one-object perturbation。比如 SuSIE (Semantically Imagined Experience) 框架。
   - https://arxiv.org/abs/2302.11550
2. **3D / point cloud task expansion**：从 object-centric 到 scene-centric，扩展到 deformable / articulated objects。
3. **Multi-robot self-teaching**：多机器人协作 generate curriculum，类似 multi-agent POET。
4. **LLM-guided task description**：把 visual goal 用 VLM caption 成 language，配合 BOSS-style chaining。
5. **Real-world task expansion**：当前 task expansion 在 simulator 中跑（因为需要 reset）。Sim2real 自我改进 loop（real data → sim task expansion → real distillation）是 natural next step。
6. **Continual learning integration**：避免 expanded policy 忘记 seeding tasks，用 EWC / replay buffer 之类。
7. **Intrinsic motivation / curiosity**：当前 metric 是 value-based，可结合 ICM/RND 提供 exploration bonus。
   - https://arxiv.org/abs/1802.12894 (RND)
8. **Diffusion-based goal generation**：用 diffusion model 替代 random sampling，generate 在 UVF 上 advantage 高的 goal，类似 decision transformer / diffusion policy。
9. **World model integration**：用 world model predict $s_T$ 后的 rollout，减少 sample cost，类似 Dreamer / DayDreamer。
   - https://arxiv.org/abs/1912.01603 (Dreamer)
10. **Asymmetric self-play formalization**：把 task expansion formalize 为 two-agent game（setter = task generator, solver = policy），引入 game-theoretic analysis。

---

## 7. 我的整体 intuition 总结

RST 的核心 insight 可以一句话概括：**用 universal value function 的"代数差"作为 open-ended curriculum 的 progression metric，让 robot 自己 compose 越来越复杂的 task**。

它本质上是把 **Asymmetric Self-Play** 中 "setter vs solver" 的二元对抗换成 "current policy vs expanded policy" 的 self-composition，metric 用 UVF 而非赢/输信号。这避免了 zero-sum game 的 instability，同时保留了 open-ended complexity growth。

更深层看，RST 是 **"learning to learn"** 的某种 instantiation：policy 不只学 task，还学 "如何扩展自己能解决的 task space"。这接近 **Open-Ended Learning / AutoCurriculum** 的 ultimate goal。

**与 foundation model 哲学的对比**：foundation model 靠 scale data + scale model；RST 靠 scale task via self-generation。两者未来很可能 converge——VLA foundation model 提供 seeding policy，RST-style task expansion 在 deployment 时持续 self-improve。

**与你的工作（Karpathy）的连接**：你讲过的 "software 2.0" / "neural network as a differentiable program" 视角下，RST 是用 NN (UVF) 当 "program synthesis" 的 controller——它 generate 的不是 code，而是 task + trajectory。这本质是用 differentiable metric 驱动 open-ended behavior generation，非常软件 2.0 的 spirit。

---

## 8. 关键 references 汇总

### Core method references
- Project page: https://sites.google.com/view/robot-self-teaching
- PPO: https://arxiv.org/abs/1707.06347
- UVF (Schaul et al., 2015): https://proceedings.mlr.press/v37/schaul15.html
- Slot Attention: https://arxiv.org/abs/2006.11555
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- R3M: https://arxiv.org/abs/2203.12601
- ResNet: https://arxiv.org/abs/1512.03385
- D4RL: https://arxiv.org/abs/2004.07219
- Learning to design bridge without blueprint (teleport setting): https://arxiv.org/abs/2103.07541

### Baseline & related method
- FLAP: https://arxiv.org/abs/2211.11787
- BOSS: https://arxiv.org/abs/2310.18215
- SayCan: https://say-can.github.io/
- Inner Monologue: https://arxiv.org/abs/2207.05608
- LMP (Play data): https://arxiv.org/abs/1910.07569
- Bridge Data: https://arxiv.org/abs/2109.13396
- BC-Z: https://arxiv.org/abs/2202.02005

### Open-ended & curriculum
- POET: https://arxiv.org/abs/1901.01753
- Enhanced POET: https://arxiv.org/abs/2010.04704
- PAIRED: https://arxiv.org/abs/2106.04823
- Curriculum Learning (Bengio 2009): https://dl.acm.org/doi/10.1145/1553374.1553380
- Reverse Curriculum (Florensa): https://arxiv.org/abs/1709.02759
- Automatic Goal Generation: https://arxiv.org/abs/1805.04880
- ACL (Graves): https://arxiv.org/abs/1710.04852
- Asymmetric Self-Play: https://arxiv.org/abs/1703.05407
- Go-Explore: https://arxiv.org/abs/1901.10995

### Goal-conditioned RL & HER
- HER: https://arxiv.org/abs/1707.01495
- Actionable Models (Chebotar et al.): https://arxiv.org/abs/2104.11477
- HiQL: https://arxiv.org/abs/2307.11949
- Decision Transformer: https://arxiv.org/abs/2106.01345

### Exploration & representation
- RND: https://arxiv.org/abs/1802.12894
- ICM: https://arxiv.org/abs/1705.05363
- Dreamer: https://arxiv.org/abs/1912.01603

### Sim2real & teacher-student
- Learning by Cheating: https://arxiv.org/abs/1912.02988
- RCAN (domain randomization): https://arxiv.org/abs/1801.00604
- Teacher-Student sim2real (Loquercio et al.): https://arxiv.org/abs/2104.04704

### Data augmentation for robotics
- SuSIE (Semantically Imagined Experience): https://arxiv.org/abs/2302.11550

### Foundation model for robotics
- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://robotics-transformer2.github.io/
- RT-X: https://arxiv.org/abs/2310.08864
- VIMA: https://arxiv.org/abs/2210.03094

---

如果让我赌一个方向：**RST-style task expansion + VLM-grounded goal proposal + diffusion-based policy distillation** 是 robot foundation model 下一代的关键 recipe。当前 RST 用的 random one-object perturbation 是手工程度最低的版本，未来用 VLM 提出 "语义上合理但 policy 还不会做" 的 goal，再用 RST 的 UVF-metric 筛选——就是 RST-VLM-fusion 的 sweet spot。
