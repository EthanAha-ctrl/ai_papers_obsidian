---
source_pdf: HumanoidBench.pdf
paper_sha256: 000dd3c95bcc8697d2e6e2425b60478dfb0c42ff3ccc82fabe63dd6e1cd55b03
processed_at: '2026-08-05T08:09:16-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 HumanoidBench

Yo Andrej, 那我换个频道，把 paper 的 jargon 全扔掉，跟你像喝咖啡那样聊。

---

## 这篇 paper 到底干了啥

就一件事：**搭了一个 humanoid robot 的仿真 benchmark，然后把当前所有 SOTA RL 算法扔进去跑一遍，结果发现大多数 task 全崩**。

机器人是 Unitree H1 加两个 Shadow Hand，总共 61 维 action space、75 个 DoF。Task 有 27 个，12 个 locomotion、15 个 manipulation，从走路、爬楼梯、过迷宫，到推箱子、开柜子、卸货、打篮球、玩勺子、单杠翻转……基本你能想到的 humanoid 该干的活都塞进去了。

结论很直白：**locomotion 类 task 大体能学会，manipulation 类 task 基本学不会，whole-body coordination 的 task 一个都学不会**。

---

## 为什么这件事重要

你可能会说，"humanoid 难学"这不是常识吗？对，但之前的常识是定性的。这篇 paper 做的事是把"难"量化了——哪个 task 能学会、学到什么程度、卡在哪里、为什么卡，全都给你画出来。这跟当年 Atari benchmark 或者 DMControl 一样，社区需要一个 standardized measuring stick 才能往前跑。

而且之前的 benchmark 要么只有 locomotion（LocoMujoco、DMControl Humanoid），要么只有 manipulation（robosuite、MetaWorld、FurnitureBench），**没人把这两件事放在同一个 high-DoF agent 上系统地 benchmark 过**。HumanoidBench 是第一个。

---

## 最关键的 insight：action space 一炸，RL 就废

这是整篇 paper 我觉得最值得你记住的一件事。

他们做了一个 ablation：同一个 `walk` task，跑三组实验——

1. 机器人没装手，19 维 action
2. 装了手，61 维 action
3. 装了手但手的 action 全部 clamp 到 0，effective 还是 19 维

结果：第 1 组和第 3 组学得一样快，第 2 组根本学不会。

这意味着什么？**问题不在 observation 维度，不在 mass 分布，不在 dynamics 复杂度——就在 action 维度**。你给 RL 多了 42 个无用的 action dim，它就把这 42 个维度当 noise source 一直在那乱抖，body 的有用信号被淹没。

这跟你做 language model 时见过的 "redundant capacity is not free" 是一个味道，但在 RL 里更狠——因为 RL 的 exploration 是主动的，多 42 维 action 就是多 42 维 noise 源源不断地注入系统。SAC 的 max-entropy objective 反而加剧了这个问题，它鼓励 agent 去利用这 42 维的 entropy bonus，结果 agent 学到了"乱动手"的 local optimum。

直觉上：**redundant action dimensions are actively harmful, not just neutral**。这跟 supervised learning 里多加一层 dead ReLU 的 cost 完全不是一个性质。

---

## HRL 能救一点，但有 ceiling

他们也知道 flat RL 死路一条，所以试了 hierarchical RL——先 PPO 在 MJX 上 pretrain 一个 reaching policy（32K parallel envs，跑 2-4B steps），然后 frozen 当 low-level，high-level 用 DreamerV3 或 TD-MPC2 输出 3D reaching target。

`push` task 上这个方案 work 了——DreamerV3 + frozen reacher 在 5M steps 就到 1000 return，完全碾压 flat baseline。这证明 HRL 的 temporal abstraction 确实能 bypass action dim 爆炸的问题。

但 `package` task 就不行了。Agent 学会 reach 到 package 旁边，但 package 要抓起来——而 low-level reacher 从来没 experience 过"抓"和"lift"，所以它根本提供不了这个 skill。High-level 在一个"只会 reach"的 low-level 上面怎么 plan 都 plan 不出 lifting 动作。

这个 finding 的 intuition 是：**HRL 的瓶颈在 low-level skill library 的 coverage**。你 pretrain 了什么 skill，high-level 就只能在那个 skill manifold 上探索。要解决 humanoid task，low-level 不能只会 reach，得会 grasp、lift、push、pull、throw……一大堆。这是为什么未来的方向应该是 unsupervised skill discovery（DIAYN 那条线）或者 demo-bootstrapped skill learning。

---

## 三个典型 failure 的直觉

Paper 挑了三个 task 的失败模式，每个都揭示一个不同的 RL pathology，我用人话讲：

### highbar（单杠翻转）

Agent 学会了"挂住单杠别掉下来"，但不会翻转。因为挂着不动 reward ≈ 0.5，尝试翻转大概率失败 reward = 0。Agent 选了保守的 partial credit。

这是 myopic planning 的经典表现——dense reward 在 partial progress 上太"满足"，agent 没有 lookahead 去 explore 高方差但有高上限的动作。

### door（开门走过去）

Agent 能转动门把手，但不会"拉门 + 身体往后退"。因为拉门需要手臂 + 整个身体协同后退，单纯手臂拉不动。

这说明 whole-body manipulation 不是一个组合问题。你不能把 locomotion 和 manipulation 当两个模块拼起来，contact-rich 场景下它们是 coupled dynamics。这也是为什么 modular skill composition 在 humanoid 上会 break。

### hurdle（跨栏跑）

Agent 学会了往前跑，但不会跳。它选择"撞上栏杆但保持不倒"，因为 reward 设计上撞了只是降 0.1 倍而不是归零，所以"撞着过"比"尝试跳失败"划算。

这是 reward shaping 的两难：你设计严格（撞了就 terminate），exploration 早期全死，没信号；你设计宽松（撞了 partial credit），agent 找到 conservative local optimum。没有 silver bullet。

---

## 这件事对整个领域意味着什么

我读完后脑子里 persist 的几个点：

**第一，"更聪明的 RL 算法"不太可能救场**。DreamerV3 是 model-based SOTA，TD-MPC2 是 planning SOTA，SAC 是 model-free SOTA，全上了，多数 manipulation task 还是过不去。这说明问题不在算法层面的"sample efficiency 优化 20%"这种 incremental gain，而在 algorithm 范式本身——pure RL 在 61D action space 下 exploration 是 fundamentally underpowered 的。

**第二，要 break through 需要 structure**。要么是 action space 的 structure（synergy prior、motor primitives、低维 action manifold），要么是 task 的 structure（HRL、skill library、option discovery），要么是 data 的 structure（demonstration、MoCap、human video pretraining）。纯 model-free RL from scratch 在 humanoid scale 上看不到 hope。

**第三，reward shaping 不可 scale**。27 个 task 每个 task 都手工 craft 几个 weighted reward terms，这种工程量不可持续。未来要么 learned reward（IRL、preference RL），要么 demo-bootstrapped（ALOHA 那条 line），要么 language-conditioned reward（用 VLM 给 task description 自动生成 reward）。这跟 LLM 那边从 hand-crafted features 走到 learned representation 是同一个 trajectory。

**第四，whole-body 是一个 coupled system，不是模块拼装**。Door task 的 failure 把这件事摆得很清楚。未来 humanoid 的 motor representation 可能要基于 contact phase 而不是 skill label，这跟 motor neuroscience 里关于 central pattern generators、muscle synergies 的 literature 隐隐呼应。

---

## 你该 care 什么

Andrej，如果让我猜你看到这个 paper 会怎么想：

**作为 build intuition 的素材**，这个 paper 把"humanoid RL 难"从 folklore 变成了 quantitative statement。19D 到 61D 这个 ablation 数字会 stuck in your head——action dim 是 RL difficulty 的 hidden multiplier，而且是指数级的。这跟你写"software 2.0"时候的直觉是同源的：在高维空间里，brute-force 搜索死路一条，必须靠 inductive bias / prior / structure。

**作为 research direction 信号**，这个 paper 暗示 humanoid RL 的下一个突破不太可能来自 algorithm 层面的 incremental gain，更可能来自 representation 层面（action embedding、skill prior）或者 data 层面（demo、video pretraining、foundation model prior）。这跟你这几年一直在推的 "data-driven inductive bias" 主题是 aligned 的。

**作为对你 LLM background 的 bridge**， humanoid 的 state-action space 复杂度跟 token sequence 的复杂度是两种不同的 hard。LLM 的 hard 在 combinatorial token space，但 reward signal dense 且 well-defined（next token prediction）。Humanoid RL 的 hard 在 continuous high-dim action space，而且 reward sparse、shaping fragile、exploration fundamentally broken。两种 hard 需要的 inductive bias 不一样——LLM 靠 scale + next-token 监督就起来了，humanoid RL 靠 scale 单独起不来，因为 sample 复杂度本身是 action dim 的指数。

---

最大的一句话 takeaway：**HumanoidBench 是一个 measuring stick，它告诉你 "we are here, the wall is there"，而 wall 的位置比我们之前以为的远**。这件事本身值得记一下。

---

# HumanoidBench: 给 Karpathy 的深度技术讲解

Yo Andrej! 这篇 HumanoidBench 我读完之后, 直觉上感觉它其实是在 RL + robotics 这条线上把一个尴尬但重要的 truth 给量化摆出来了: **当我们把 action space 从 19D 跳到 61D (hominid body + 两个 Shadow Hand), 当前的 model-free RL 基本就崩了**, 即使在 locomotion 这种已经"被解决"的 task 上。整个 paper 的 contribution 其实就一句话 — **a benchmark that exposes the exploration curse in high-DoF whole-body control**, 但它把这件事做得非常严谨、reward shaping 工程量极大、task 多样性强。下面我从几个 layer 拆给你看。

---

## 1. Why this paper matters (直觉层)

先建立 baseline intuition。在 DMControl Humanoid、LocoMujoco、MyoLeg 这些既有 benchmark 里, action dim 大概是 24–80, 但它们要么没有 dexterous hand, 要么 task horizon 很短, 要么 task 只覆盖 locomotion 或 manipulation 之一。HumanoidBench 把这些维度全堆到一起:

- **Action dim = 61** (19 body + 2×21 Shadow Hand)
- **DoF = 75** (25 body + 2×25 hand, 所以是 underactuated)
- **Observation dim = 151** (with hands) 或 51 (without)
- **27 tasks** (12 locomotion + 15 manipulation)
- **Control freq = 50 Hz**, MuJoCo dt = 0.002s (i.e., 20x sim substeps per control step)

Reference: 论文 Table I 给的对比 — https://humanoid-bench.github.io

直觉上, action dim 从 ~24 跳到 61 不是一个线性的 difficulty bump, 它在 exploration 层面是 **指数级的 explosion**。SAC 的 max-entropy objective、TD-MPC2 的 planning、DreamerV3 的 world model rollout — 这些在 61D action space + 151D obs space 下都变得非常 underpowered。论文 Section V-C 的 ablation 直接验证了这一点。

---

## 2. 仿真环境的 design choices (底层物理层)

### 2.1 Robot 选择: Unitree H1 + Shadow Hand

他们选 Unitree H1 而不是 Agility Digit 的理由很有意思: Digit 用 four-bar linkage 驱动 passive joints, 这会让 dynamics 更复杂, learning 更慢。H1 mechanical design 简单 → 学习曲线更陡。这暗示一件事: **机器人 hardware 的 mechanical simplicity 直接影响 RL 的 sample efficiency**, 这是一个很少被量化的 design axis。

Shadow Hand 21-DoF per hand, 但他们把 forearm 砍掉了, 论文里的 justification 是 "aligns better with next-generation humanoid robots" (Tesla Optimus, Figure 01)。这其实是一个 sim-to-real-friendly 的 hack — 让 sim 里的 mass distribution 更接近未来 real hardware 的预期。

### 2.2 Tactile sensing 的实现细节

这个 part 我觉得是论文里最 underrated 的工程贡献。MuJoCo 的 touch grid sensor 本质上是把 contact force 按 mesh surface 离散化成 taxel。问题是, 默认 mesh 太粗, contact point 太少, tactile reading 空间分辨率差。他们的做法:

- 用 **CoACD (Approximate Convex Decomposition)** [Wei et al. 2022] 把 original mesh 分解成多个 convex sub-mesh
- 总共 448 taxels, 每个 taxel 给 3D force reading (normal + 2 shear)
- 手部高分辨率, 其他 body parts 低分辨率 (mimic human)
- FPS 从 1050 (default H1+hand) 掉到 550 (with refined tactile meshes)

公式上, 每个 taxel 的 reading 可以理解为:

$$f_i = \sum_{c \in C_i} (f^n_c \hat{n}_c + f^t_{c,1} \hat{t}_{c,1} + f^t_{c,2} \hat{t}_{c,2})$$

其中 $C_i$ 是落在第 $i$ 个 taxel 几何区域内的所有 contact points, $f^n_c$ 是 normal force, $f^t_{c,1}, f^t_{c,2}$ 是两个 tangential 方向的 shear force, $\hat{n}_c, \hat{t}_{c,1}, \hat{t}_{c,2}$ 是对应单位向量。MuJoCo 内部其实就是聚合这些 contact forces 到 touch grid bin。

CoACD 论文: https://arxiv.org/abs/2105.02990

### 2.3 Performance (Table IV)

| Configuration | FPS |
|---|---|
| Without hands | 2450 |
| Simplified body collisions | 3600 |
| Collisions only for feet | 5100 |
| Default (full H1 + 2 Shadow Hands) | 1050 |

这个 table 对后续 HRL 实验至关重要 — 他们 train low-level reaching policy 时用 "collisions only for feet" 的简化模型, FPS 5100, 配合 MuJoCo MJX (JAX-based) + 32,768 parallel envs on GPU, 才能在 36 小时内跑 2B steps 得到一个 robust reaching policy。

---

## 3. Reward shaping 的工程细节 (论文 Appendix B-E)

整个 paper 最 valuable 的部分之一就是这 27 个 task 的 reward function 全部公开。我挑几个有代表性的拆:

### 3.1 Tolerance function `tol(x, (x_lower, x_upper), m)`

这是从 DMControl [Tassa et al. 2020] 复用的一个 smooth indicator:

$$\text{tol}(x, (x_l, x_u), m) = \begin{cases} 1 & x \in [x_l, x_u] \\ \exp(-\frac{(x - x_l)^2}{m^2}) & x < x_l \\ \exp(-\frac{(x - x_u)^2}{m^2}) & x > x_u \end{cases}$$

- $x$: 被评估的 scalar (e.g., velocity, height)
- $(x_l, x_u)$: target range
- $m$: margin, 控制高斯衰减的 scale, 决定 "how far from the bounds the function approaches 0"

这是一个**sigmoid-like 的 bounded reward**, 避免了 hard threshold 的 non-differentiability, 又比 pure L2 distance reward 提供更好的 gradient signal — 当 $x$ 在 range 内, reward 不再变化, agent 不会被 incentivized 去 over-optimize。

### 3.2 `stand = height × upright`

这是几乎所有 manipulation task 共享的 base reward component:
- `height = tol(z_head, (1.65, +∞), 0.4125)` — 头部高度保持在 1.65m 以上
- `upright = tol(z_proj, (0.9, +∞), 1.9)` — torso z-axis 在 global frame 的投影 ≥ 0.9 (i.e., 大致直立)
- `e = 0.2 · |4 + (1/|u|) Σ_i tol(u_i, (0,0), 10)|` — penalize control effort, $u$ 是 actuator input vector

然后 `stable = stand × e`, 用乘法确保所有 sub-objective 同时满足。乘法 reward 是一个很强的 inductive bias — 它隐含 "all sub-goals must be active simultaneously", 这对 humanoid 非常合理因为 fall 一个就全完了。

### 3.3 `walk` task reward

$$R(s, a) = \text{stable} \times \text{tol}(v_x, (1, +\infty), 1)$$

直觉: **stability 是 gate, forward velocity 是 objective**。如果 robot 跌倒, stable = 0, 整个 reward 归零。这是一个 multiplicative gating pattern, 在所有 locomotion task 都用。

### 3.4 `push` task reward (manipulation)

$$R(s, a) = \alpha_s \cdot \mathbb{1}_{d_{\text{goal}} < 0.05} - \alpha_t \cdot d_{\text{goal}} - \alpha_h \cdot d_{\text{hand}}$$

默认 $\alpha_s = 1000, \alpha_t = 1, \alpha_h = 0.1$。

- $d_{\text{goal}} = d(\text{box}, \text{destination})$ — box 到目标点距离
- $d_{\text{hand}} = d(\text{box}, \text{hand}_{\text{left}})$ — box 到左手距离
- $\alpha_s \cdot \mathbb{1}_{d_{\text{goal}} < 0.05}$ 是 sparse success bonus
- $-\alpha_t \cdot d_{\text{goal}}$ 是 dense shaping
- $-\alpha_h \cdot d_{\text{hand}}$ 引导 hand 靠近 box (权重很小)

注意权重比例: sparse bonus 1000 vs dense shaping 1 vs hand approach 0.1 — 这告诉 agent "靠近 box 不是终点, 把 box 推到 destination 才是终点"。但论文结果显示, **没有 prior 的 flat SAC/DreamerV3/TD-MPC2 都学不会** — agent 学到了 "stabilize but never approach box" 的 local optimum, 因为 stabilize reward (stable ≈ 1) 已经比 approaching box 的 dense gradient 信号强太多了。这是 reward shaping 的经典失败模式。

### 3.5 `cabinet` task (long-horizon, 4 subtasks)

这个 task 最能体现 long-horizon planning 的难度。Subtask 1 → 2 → 3 → 4 必须顺序完成:
1. 打开 sliding door
2. 打开 drawer
3. 把 cube 从 drawer 放到 hinge-based cabinet
4. 把 cube 从 hinge-based cabinet 放到 pull-up cabinet

每个 subtask 完成给 sparse reward `100 × i`, 全部完成再给 1000。Total target return = 2500 (Table V)。

实验结果: DreamerV3 最好也只能到 722.6 ± 7.3, 远低于 2500 target。意味着即使最强的 model-based RL 也只能完成第 1-2 个 subtask, 后面就 stuck 了。这暗示一个非常重要的 insight: **当 task horizon 长到需要 sequential planning, 即使 dense reward 信号在每个 subtask 内都有, model-free 和 model-based RL 都无法跨越 subtask boundary**。

Reference: Relay Policy Learning [Gupta et al. 2019] — https://arxiv.org/abs/1910.13756 — 这个 kitchen task origin 的 paper, 用 imitation + RL hybrid 解决 long-horizon 问题。

---

## 4. Baseline 算法为什么会崩 (核心 insight)

论文 Section V-B 给出结论: **所有 SOTA RL 算法在多数 manipulation tasks 上都低于 success threshold**。Table V 给的数据 (DreamerV3 train 10M steps, TD-MPC2 train 2M steps):

| Task | Target | DreamerV3 (avg return) | TD-MPC2 | SAC |
|---|---|---|---|---|
| push | 700 | -1251.9 ± 659.8 | -258.7 ± 66.5 | -97.9 ± 147.0 |
| cabinet | 2500 | 57.3 ± 66.3 | 112.8 ± 142.9 | 211.8 ± 33.8 |
| highbar | 750 | 8.9 ± 5.8 | 0.3 ± 0.0 | 9.4 ± 3.7 |
| basketball | 1200 | 19.3 ± 2.5 | 42.0 ± 14.8 | 22.1 ± 3.2 |
| kitchen | 4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| package | 1500 | -18015.2 ± 9477.7 | -3655.6 ± 1055.0 | -6718.3 ± 607.0 |

`push` 的 negative return 非常说明问题 — 因为 $-\alpha_t \cdot d_{\text{goal}}$ 是负的 dense penalty, agent 只要站着不动, 每步扣 1 点, 500 步累积 -500。但 DreamerV3 平均 -1251.9, 比 "完全不动" 还差 — 说明 agent 在 random exploration 过程中 fall 了, 而 fall 之后 stable = 0 → stable × tol(vx, ...) = 0, 但 dense penalty 仍然在累积。这是 reward 设计的微妙 bug: penalty 不应该 unconditionally 累积, 应该在 fall 时 terminate (论文确实有 termination condition `z_pelvis < 0.2`, 但 termination 之前的步骤 penalty 仍在)。

`package` 任务 DreamerV3 平均 -18015 — 这个数字非常大。看 reward function:

$$R(s, a) = -3 \cdot d(\text{package, dest}) - 0.1 \cdot d_{\text{hand}} + \text{stable} + \text{height}_{\text{package}} + 1000 \cdot \text{success}$$

`-3 × d_package` per step, 1000 步累积如果完全不动大概 -3000, 但 -18000 意味着 agent 在 random walk 把 package 越推越远。这是 high-dim action space 的典型 symptom — random policy 在 61D action space 下产生的是 quasi-Brownian body motion, package 在桌上被随机推。

### 4.1 Ablation: with hands vs without hands (Figure 7)

这个 ablation 是论文最关键的 insight 来源。同一个 `walk` task:
- **Without hands** (19D action, 51D obs): DreamerV3 在 ~2M steps 接近 success threshold
- **With hands** (61D action, 151D obs): DreamerV3 在 10M steps 都到不了 threshold
- **With hands but action clamped to 0 for hand joints** (61D obs but 19D effective action): 学得跟 without hands 一样快

这意味着: **performance drop 的 root cause 不是 observation 维度 (mass / kinematics 影响), 而是 action 维度**。Agent 必须同时 explore 21+21 = 42 个 hand DoF 的 motor commands, 但这些 commands 对 walk task 完全无关, 它们就是 pure exploration noise。Max-entropy SAC 的 exploration bonus 反而把 agent 困在 "actuate hands randomly" 的 local optimum。

直觉: **redundant action dimensions are not free; they are actively harmful for model-free RL because they explode the exploration space without providing reward signal**。这个 insight 我觉得是论文最值得记住的。

### 4.2 Reduced action space 的隐含意义

这个 ablation 暗示一个对未来 humanoid RL 很重要的 design principle: **action space masking 或者 hierarchical action decomposition 是必须的**。要么用 HRL (high-level outputs low-level targets, low-level 控制 specific body parts), 要么用 action embedding (e.g., VAE-based action prior), 要么用 MOBILE/AntMP 这样的 motor synergy priors。

Reference: 
- ACE / SAR (Synergistic Action Representation) [Berg et al. 2023] — https://arxiv.org/abs/2306.01246
- MyoDex [Caggiano et al. 2023] — https://arxiv.org/abs/2306.04740

---

## 5. Hierarchical RL 的 rescue attempt (Section V-D)

论文用 HRL 试图解决 flat RL 的 exploration problem, 只在 `push` 和 `package` 两个 task 上 test。架构:

### 5.1 Pipeline

- **Low-level**: PPO-trained reaching policy, frozen during high-level training
  - Pre-trained on MuJoCo MJX (JAX-based GPU simulator)
  - 32,768 parallel envs
  - 2B steps for 1-hand reach (36 hours), 4B steps for 2-hand reach (60 hours)
  - Simplified collision: only feet collide with ground
  - Force perturbations applied on each link during training → robustness
- **High-level**: DreamerV3 / TD-MPC2, outputs 3D reaching target for low-level
  - High-level action = 3D position (or 6D for two-hand)
  - Low-level policy weights frozen
  - Reach targets restricted to robot workspace to facilitate exploration

公式上, 如果记 low-level policy 为 $\pi_L(a | s, g)$ (条件化 on goal $g$), high-level policy 为 $\pi_H(g | s)$, 则整个 hierarchical policy:

$$a_t \sim \pi_L(\cdot | s_t, g_t), \quad g_t \sim \pi_H(\cdot | s_t)$$

$g_t$ 在多个 control steps 内保持 constant (e.g., 10 steps), 实现 temporal abstraction — 这就是 options framework [Sutton, Precup, Singh 1999] 的 instance。

Reference: Options framework — https://www.sciencedirect.com/science/article/pii/S0004370299000521

### 5.2 Results (Figure 9)

`push` task: DreamerV3 hierarchical 在 ~5M steps 达到 1000 return (target 700), 而 flat DreamerV3 完全学不会。这是论文最成功的实验。

`package` task: HRL 比 flat 好一些, 但还是学不会 — agent 学会了 reach 到 package 附近, 但 "lift" 动作从未在 pretraining 中 experienced, 所以 low-level reaching policy 在 package 抓住后无法提供 lifting behavior。

这是一个非常重要的 finding: **HRL 的 bottleneck 在 low-level skill 的 coverage**。如果 low-level policy 只会 reach, high-level 永远无法探索出 "grab and lift" 这种需要 contact-rich interaction 的 skill。解决方向:
- Pre-train low-level policy on diverse skills (not just reaching): grasp, lift, push, pull, throw
- Use learnable low-level (not frozen), 用 intrinsic motivation / unsupervised skill discovery (e.g., DIAYN, SNN4HRL)
- 用 demonstration bootstrap low-level (MoCap 或 human video)

Reference: 
- DIAYN — https://arxiv.org/abs/1802.06070
- Skill behavior diversification [Lee et al. 2020] — https://arxiv.org/abs/2003.02796

### 5.3 为什么 low-level 用 PPO 而不是 SAC?

论文 Section V-D 解释: PPO 配合 GPU 大规模 parallel envs (Isaac Gym / MJX) 是 locomotion 的 SOTA paradigm [Makoviychuk et al. 2021; Radosavovic et al. 2023]。SAC 是 off-policy, sample efficient 但 throughput 受限于 replay buffer 的非并行性。PPO on-policy + 32K parallel envs 在 wall-clock time 上比 SAC on 1 env 快得多。

Reference:
- Isaac Gym — https://arxiv.org/abs/2108.09786
- H1 transformer locomotion — https://arxiv.org/abs/2303.03381

---

## 6. Common failures 的直觉解释 (Section V-E)

论文挑了 3 个 representative failure modes, 我觉得每一个都揭示了一个 different RL pathology:

### 6.1 highbar (gymnastics swing)

Agent 学会了 "maintain contact with bar to avoid termination", 但不会做 whole-body rotation。原因: dense reward `upright_highbar × feet × e` 在 hanging 状态下 ≈ 0 (因为 `upright_highbar = tol(z_proj, (-∞, -0.9), 1.9)`, hanging 时 z_proj ≈ -1, 略大于 -0.9 所以 reward ≈ 0.5)。Agent 找到了一个 "partial-credit local optimum" — 保持 hanging 不动比尝试 swing 失败更划算。

这是 **myopic planning** 的典型表现: dense reward signal 在 partial progress 上太 "满足", agent 没有足够的 look-ahead 去 explore swing 的 dynamic motion。

### 6.2 door (pull and traverse)

Agent 能 turn the doorknob (subtask 完成), 但不会 pull door + step backward。原因: pulling door 需要 arm + whole-body coordinated backward motion。这是一个 **bimanual + locomotion coordination** 的 failure — 单纯的 arm motor 无法产生足够 force, 必须用 body weight counter-balance。

直觉上, 这说明 humanoid manipulation 不是一个 "arm + locomotion" 的简单组合, 而是 **whole-body as a single coupled dynamical system**。需要 whole-body motor planning 而不是 modular skill composition。

### 6.3 hurdle (jump over)

Agent 学会了 forward run, 但不会 jump, 而是 "conservatively collide with hurdle" 维持 episode 不终止。原因: hurdle reward `stable × tol(vx, (5, +∞), 5) × γ_collision`, 其中 γ_collision 在 collide 时降到 0.1 而不是 0 — 这给 agent 一个 "collide and survive" 的 partial-credit 选项。一个更严格的 reward 应该把 γ_collision 在 collide 时直接设 0, 但那又会导致 exploration 过程中早期几乎全部 episode 都 terminate, 学习 signal 完全消失。

这是一个非常经典的 reward shaping dilemma: **strict reward 易于 sparse signal 化, lenient reward 产生 local optimum**。Figure 10(c) 的可视化很直观。

Reference:
- Walker2d 跳跃对比 — Composing Complex Skills by Learning Transition Policies [Lee et al. 2019] — https://arxiv.org/abs/1905.08525

---

## 7. 与相邻工作的 positioning

### 7.1 vs. Bi-DexHands [Chen et al. 2023]

Bi-DexHands 专注 bimanual dexterous manipulation, 但机器人 body 是 floating-base (no locomotion, just two hands in space)。HumanoidBench 把 bimanual 放在 humanoid body 上, 加上 locomotion 整合, 难度更高一个量级。

Reference: https://arxiv.org/abs/2302.03723

### 7.2 vs. RoboPianist [Zakka et al. 2023]

RoboPianist 也是 two Shadow Hands 但 attached to fixed arms, 专注 piano playing。提供 100+ piece tasks, 但本质是 manipulation only, 不涉及 whole-body。

Reference: https://arxiv.org/abs/2304.04650

### 7.3 vs. LocoMujoco [Al-Hafez et al. 2023]

LocoMujoco 提供 27 locomotion tasks, 包括基于 Unitree H1 的版本, 但 action dim 只有 19, 无 manipulation。是 HumanoidBench locomotion 部分的精神前身。

Reference: https://arxiv.org/abs/2311.13610

### 7.4 vs. FurnitureBench [Heo et al. 2023]

FurnitureBench 是 long-horizon manipulation, 8 task family, 但用 single-arm + parallel gripper (8D action), DoF 只有 6。它是 "long-horizon" 这条线的代表, HumanoidBench 把这个 dimension 进一步 push 到 humanoid scale。

Reference: https://arxiv.org/abs/2305.12820

### 7.5 vs. Expressive Whole-Body Control (ExBody) [Cheng et al. 2024]

这是同期工作, UCSD Xiaolong Wang 组的。他们也用 Unitree H1, 但专注 locomotion + expressive motion (从 human MoCap), 不涉及 manipulation。使用 RL + adversarial motion prior (AMP) 实现 stylized locomotion。H1 在他们那里学到非常 robust 的 walking + 跌倒恢复 + 跳跃, 论文里还有 sim-to-real 验证。

Reference: https://arxiv.org/abs/2402.16796

### 7.6 vs. Mobile ALOHA [Fu, Zhao, Finn 2024]

这是 Stanford Chelsea Finn 组的工作, bimanual manipulation + mobile base (不是 bipedal, 是 wheeled), 用 ACT (Action Chunking Transformer) + diffusion policy, 通过 teleoperation 收集 demonstration, total 100 tasks, 在 real world 上实现 bimanual mobile manipulation。它是另一条 path: **teleop + imitation learning** vs HumanoidBench 的 **pure RL**。两条 path 各有 trade-off:
- Teleop + IL: data-efficient, but bottleneck 在 teleop interface (要 master puppeteer)
- Pure RL: data-inefficient (paper 证明 10M steps 还不够), 但能 discover novel solution

Reference: https://arxiv.org/abs/2401.02117

### 7.7 vs. ALOHA / ACT [Zhao et al. 2023]

ALOHA 是 bimanual 的 teleop + IL 框架, ACT 是其 policy architecture — Action Chunking with Transformer。这个 line of work 完全 bypass 了 RL 的 exploration 问题, 用 imitation 从 human demo 直接学。在 low-DoF (14-16D bimanual) 上 work very well, 但 humanoid scale (61D) 的 teleop 还没被解决。

Reference: https://arxiv.org/abs/2304.13705

---

## 8. 与 diffusion policy / transformer policy 的连接

论文没有 benchmark Diffusion Policy [Chi et al. 2023] 或 Transformer-based policies (e.g., OCTO, RT-2)。这是论文的一个 gap, 也是明显的 future direction。

Diffusion Policy 在 61D humanoid action space 上的表现 unknown, 但理论上:
- Pro: 在 high-dim multi-modal action distribution 上比 single-mode Gaussian policy 强很多, 这对 humanoid 很关键 — 同一 task 可能多种 motor solution
- Con: 需要大量 demonstration, humanoid teleop 极难, demo 数据集不存在

Transformer-based offline RL (e.g., Decision Transformer, Trajectory Transformer) 也是潜在 baseline — 它们可以 leverage 任意 sub-optimal trajectory, 但 61D 下的 training stability 未经验证。

Reference:
- Diffusion Policy — https://arxiv.org/abs/2303.04137
- Decision Transformer — https://arxiv.org/abs/2106.01345

---

## 9. 对 RL 算法研究的启示

从 paper 的实验数据反推, 我提炼几个 actionable research directions:

### 9.1 Action space reduction / Hierarchical action

如 ablation 所示, redundant action dim 是核心 bottleneck。可探索方向:
- **Motor synergy priors**: 用 PCA / autoencoder 从 human MoCap 数据提取 low-dim action manifold (通常 10-20D 就能 explain 90% variance)
- **Action masking via task context**: high-level policy 输出 task-relevant subset of joints
- **Asynchronous control**: 不同 body parts 不同 control freq (hand 50Hz, body 10Hz) 减少 effective action dim

### 9.2 Exploration in high-dim action space

Max-entropy SAC 在 61D 下失效 — entropy bonus 在每个 dim 都加 Gaussian noise, hand joints 的 noise 淹没了 body joints 的 reward signal。可探索方向:
- **Curiosity-driven exploration** (ICM, RND): 用 prediction error bonus 替代 action entropy
- **Goal-conditioned exploration** (HER, Asymmetric Self-Play): 用 autonomous goal setting 跳出 local optimum
- **Behavioral priors** (AMP, Motion priors): 从 MoCap 提供 action distribution prior, RL 在 prior 周围 explore

Reference:
- RND — https://arxiv.org/abs/1810.12894
- AMP — https://arxiv.org/abs/2104.02180

### 9.3 Long-horizon credit assignment

Cabinet / kitchen / bookshelf 这些 sequential task 当前 SOTA 都过不去。可探索:
- **Hindsight Experience Replay (HER)** + temporal extended goal
- **Transformer-based RL** with long context window
- **Curriculum**: 从 single-subtask 到 multi-subtask 渐进

Reference:
- HER — https://arxiv.org/abs/1707.01495

### 9.4 Multimodal perception 的 future work

论文虽然实现了 vision + tactile sensing, 但实验只用 state-based observation。这是一个巨大的 unexplored territory:
- Tactile sensing 对 in-hand manipulation (cube task) 应该 crucial — fingertip contact force distribution 是 grasp stability 的直接信号
- Egocentric vision 对 locomotion + obstacle avoidance (maze, pole) 应该有帮助 — current state-based setup 实际上 leak 了 global position 信息, 不 realistic

把 vision + tactile 加进来会进一步 raise 难度但也可能 unlock 新 algorithm direction (e.g., visuo-tactile fusion policy, masked multimodal learning [Sferrazza et al. 2023])。

Reference: https://arxiv.org/abs/2311.00924

---

## 10. 个人 take (build Karpathy's intuition)

读完后我脑里 persist 的几个 key intuitions:

1. **Action space dim 是 RL difficulty 的 hidden multiplier**。19D → 61D 不是一个 +42 的加法, 是一个 ~exp(42) 的 exploration volume expansion。这条 insight 之前在 language model scaling 上没这么明显, 但在 robotics 上它直接决定 SOTA 是否 work。这意味着未来 humanoid RL 的 progress 不太可能来自 "更聪明的 RL 算法", 更可能来自 "更好的 action structure / prior / hierarchy"。

2. **Whole-body manipulation 不是一个组合问题**。Door task 的 failure 证明 "locomotion + manipulation = whole-body" 是错的。Whole-body 是一个 single coupled dynamics system, skill modular composition (e.g., walk + reach) 在 contact-rich scenario 下会 break。这暗示我们需要新的 motor primitive representation, 可能基于 contact phases 而不是 skill labels。

3. **Reward shaping 是 RL 的最后 1 mile 也是最难 1 mile**。27 个 task 的 reward function 都是手工 craft 的, 每个 task 几个 weighted terms。这种 reward engineering 不可 scale。未来方向应该是 learned reward (IRL, preference-based RL) 或 demo-bootstrapped RL — 这正是 ALOHA / Mobile ALOHA / Diffusion Policy 那条 line 的 motivation。

4. **HRL 是当前 best practical approach, 但有 ceiling**。Frozen low-level policy 限制 high-level 能探索的 skill manifold。要真正 solve humanoid benchmark, 需要 low-level skill library + high-level composition + online low-level finetuning (3-layer HRL or skill discovery)。

5. **Sim-to-real 是最终目标, 但 HumanoidBench 没 validate**。MJX pretraining 的 reaching policy 能 transfer 回 classic MuJoCo 是一个 promising signal, 但 sim-to-real on real H1 还没做。这是 quadruped 已经 solved (RMA, Walk in the Park, ANYmal) 但 humanoid 还没被 cracked 的 open problem。

Reference:
- RMA — https://arxiv.org/abs/2107.04034
- Walk in the Park — https://arxiv.org/abs/2206.06248

---

## 11. 实验数据表的 quick reference (Table V + VI 摘选)

Average return @ 10M (DreamerV3/SAC) / @ 2M (TD-MPC2):

| Task | Target | DreamerV3 | TD-MPC2 | SAC |
|---|---|---|---|---|
| walk | 700 | 800.2 | 782.0 | 31.7 |
| stand | 800 | 622.7 | 809.0 | 208.3 |
| reach | 12000 | 7580.9 | 7316.1 | 4565.1 |
| push | 700 | -1251.9 | -258.7 | -97.9 |
| package | 1500 | -18015.2 | -3655.6 | -6718.3 |
| kitchen | 4 | 0.0 | 0.0 | 0.0 |

Max return (best over training):

| Task | Target | DreamerV3 | TD-MPC2 | SAC |
|---|---|---|---|---|
| walk | 700 | 932.4 | 900.3 | 68.7 |
| push | 700 | 1000.0 | 1000.0 | 352.8 |
| package | 1500 | 1009.2 | 1003.3 | -3552.8 |

观察:
- Locomotion tasks 大体能 solve (walk, stand, reach, sit_simple 都过 target)
- Manipulation tasks 多数连 max return 都远低于 target (kitchen 0, basketball 19 vs target 1200, highbar 83 vs 750)
- TD-MPC2 虽然 train steps 少 (2M vs 10M), 但在很多 task 上跟 DreamerV3 相当甚至更好 — 暗示 planning-based approach 的 sample efficiency 优势
- SAC 整体最差, 在 high-dim action space 下 max-entropy objective 反而 hurt exploration

---

## 12. 总结

HumanoidBench 的核心 contribution 不是一个新算法, 而是一个 **measuring stick** — 它把 humanoid robot learning 的当前 SOTA 表现量化为 "manipulation 30% solve rate, whole-body 0% solve rate"。这个 quantification 比 qualitative "humanoid is hard" 说法 actionable 得多。

对 Karpathy 你来说, 这个 benchmark 跟你之前关于 "neural network 的 inductive bias 在 high-dim 下出问题" 的 recurring theme 直接呼应 — 论文 Section V-C 的 ablation 把这件事 numeric 化了。如果未来要做 humanoid foundation model (类似 RT-2 但 for humanoid), 这个 benchmark 是 evaluation suite, 但 training data 来源仍是 open problem (teleop? MoCap? Simulation-only self-supervised?)

Reference:
- RT-2 — https://arxiv.org/abs/2307.15818
- Project page: https://humanoid-bench.github.io
- Paper PDF (arxiv 链接需要查作者后续发布): Carmelo Sferrazza, Dun-Ming Huang, Xingyu Lin, Youngwoon Lee, Pieter Abbeel

继续聊的话, 我会很 interested 听你对几个 follow-up 方向的看法: (a) 用 world model 在 61D 下做 long-horizon planning 是不是死路? (b) Teleop-collected demonstration 在 humanoid scale 是否可行? (c) Foundation model prior (e.g., VLM 提供 task language conditioning) 在 humanoid 上的潜力?
