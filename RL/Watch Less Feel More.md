---
source_pdf: Watch Less Feel More.pdf
paper_sha256: d86ad945734f8e3ccb5c669a11f13b2a86bd2df8958c51644c7090d832a8a53e
processed_at: '2026-08-13T03:40:56-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

## 一句话总结

**让 robot 学会在黑暗中开门开抽屉 —— 只看一眼，剩下全靠手感。**

---

## 核心问题：为什么开门这么难？

你以为开门很简单？对 robot 来说简直噩梦。

你面对一扇门，你**看**它，觉得它就是个门。但你一**抓**住把手开始拉，门才会告诉你它的秘密：

- 旋转轴在哪？你不知道，得靠手感判断
- 阻力多大？你不知道，得拉一下试试
- 铰链紧不紧？你不知道，得持续拉才能感觉到

两扇**看起来一模一样**的门，可能一个是新的顺滑无比，一个是老锈门拉都拉不动。Vision 在这里完全失效 —— 你眼睛看到的信息和真实物理属性根本不相关。

更要命的是，如果你不管门的实际运动轨迹，硬按你脑子里的"理想开门轨迹"去拉，门会卡住，robot arm 会被巨大的反作用力怼到触发 emergency stop。这不是软件 bug，这是物理定律。

---

## 作者的 Intuition：你怎么在黑屋子里开门？

想象一下半夜停电，你要出门。你能做到吗？大概率能。

你怎么做的？
1. 你大概知道门把手在哪
2. 你摸过去，手碰到把手
2. 你一拉，发现门不动 —— 哦，铰链有点紧，得用力点
3. 你拉开门，感觉到门在画弧线 —— 哦，铰链在左边
4. 你顺着弧线继续拉，门开了

全程没开灯。靠的是什么？**你对动作和反馈的记忆**。你做了一个 action（拉），得到了一个 observation（门动了/没动/动了但方向不对），你脑子里就在 update 对这个门的 model，然后决定下一步怎么拉。

这就是 "Watch Less, Feel More" 的字面意思 —— 别老盯着看，多感受。

---

## 方法的三件套

### 件一：只看一眼，之后纯手感

传统方法有两种极端：

**Open-loop 派**：看一眼，算好完整 trajectory，闭眼执行。问题是门动起来之后你不知道，trajectory 跟门的实际运动对不上，累积 error 越来越大，最后门没开成 robot 还撞上了。

**Vision-closed-loop 派**：每一步都看一眼，根据最新 vision 调整。问题是 vision sim-to-real gap 巨大 —— sim 里的点云渲染得完美无瑕，real 里你的 RealSense 拍出来一堆 noise，而且手抓住把手之后把手被挡住了根本看不见。

**作者的折中**：只在 t=0 那一帧用 RGBD 拍一张照，跑 SAM 分割 + GSNet 预测一个 grasp pose（手该往哪抓），之后整个 episode 再也不看 vision，纯靠 proprioception（关节位置、end-effector pose、gripper 抓没抓住）和 action history 跑闭环。

这等于把 vision 的 sim-to-real gap **局部化**到了一个 isolated module，而不是让它污染整个 policy。

### 件二：Robot 要学会"软硬兼施"

这是这篇 paper 的核心 contribution。

传统 position control 的逻辑是："我让你去 x 位置，你一定要去到 x，不管路上遇到什么"。遇到门把手的反作用力，它会产生巨大的 joint torque 硬怼。

Impedance control 的逻辑是："我让你去 x 位置，但你要像弹簧一样，遇到外力可以适当让一让"。弹簧的刚度 $K$ 决定了它有多"倔" —— $K$ 大就硬，$K$ 小就软。

关键 insight：**不同阶段应该用不同刚度**。
- Reaching 阶段（手离门还远）：应该硬，快速精确地 reach 到把手附近
- Opening 阶段（已经抓住把手在拉门）：应该软，让 hand 跟着门的实际弧线走，而不是硬把门往自己的 trajectory 上拽

作者让 RL policy 自己输出这个 $k_p$（stiffness gain），范围 clip 在 $[60, 140]$。结果 super cool —— **完全没给 $k_p$ 任何 reward signal**，policy 自己学到了 reaching 时调高、opening 时调低。这就是 emergent behavior，RL 最让人着迷的地方。

为什么这件事重要？因为 position control 在 sim 里能"作弊" —— sim 的 physics engine 允许你施加任意大的 torque，没有 emergency stop。所以 sim 里看起来 success rate 0.84，一切美好。一旦部署到 real Franka 上，大 torque 直接触发安全停机，success rate 跌到 0.40。Impedance control 这时候就救命了 —— 它物理上限制了 force 的施加，保证 real-world 不炸。

### 件三：Sim-to-Real 的 Information Asymmetry

这是从 locomotion 借来的老 trick，但用在 manipulation 上很合适。

Sim 里你知道一切 ground truth：pivot center 在哪、object mass 多少、stiffness 多少、joint 现在开了多少度。

Real 里你一个都不知道。你只知道：我刚才做了什么 action，然后 robot 的 joint 现在在哪，end-effector 现在在什么 pose。

作者的方案：**训练两个 network，同步训**。

- **Teacher (Privileged Encoder $\phi$)**：吃 sim 的 ground truth physics parameter，输出一个 20 维 latent $z$
- **Student (Adaptation Module $\sigma$)**：吃过去 10 步的 (observation, action) history，输出一个 20 维 latent $\tilde{z}$

两者之间用一个双向 stop-gradient 的 L2 loss 拉近：

$$\lambda \|z - \text{sg}[\tilde{z}]\|_2 + \|\text{sg}[z] - \tilde{z}\|_2$$

这个 loss 看起来奇怪，其实就是在说："让 teacher 的 $z$ 和 student 的 $\tilde{z}$ 互相靠近，但梯度不要乱流 —— teacher 别为了迁就 student 而变笨，student 别为了抄 teacher 而塌缩到 trivial solution"。

部署到 real 时，teacher 扔掉，只用 student。Student 从 history 里 implicit 地 infer 出了 object 的物理属性。

为什么要 online 训而不是先训 teacher 再 distill？因为 two-stage 容易出现 realizability gap —— student 永远追不上 teacher。同步训的话，teacher 知道 student 能做到什么程度，会 adjust 自己的 representation。

---

## Reward 怎么搞？

这是个 multi-stage task：reach → grasp → open。End-to-end 训一个 policy 处理所有 stage，reward 设计就成了炼丹的关键。

作者的思路是 **task-aware + motion-aware** 两类 reward：

### Task-aware（保证正确顺序）

不能让 policy 学会"不 grasp 就硬掰门"这种 cheating behavior。所以：
- 成功开门的 reward 只有在"已经 grasp"的时候才给满
- 距离 reward 在 grasp 之后会衰减（已经抓到了，就别再追距离了）
- 门开了多少度的 reward 也只有在 grasp 之后才激活

核心 trick：用 $0.5^{\mathbb{1}_g}$ 这种指数 decay 来 modulate reward —— 没 grasp 时 weight 是 1，grasp 后变 0.5。这比硬 if-else 要 smooth，对 RL 友好。

### Motion-aware（保证可部署）

这一组是 paper 的 engineering 重点，也是 sim-to-real 的关键：
- **Energy penalty**：惩罚大的 $\tau \dot{q}$（joint power），避免 sim 里学到"用 brute force 解决一切"
- **Tracking reward**：让 policy 输出的 command 是 impedance controller 能跟得上的，别提出 physics-infeasible 的 target
- **Smoothness penalty**：惩罚 action 方向反复横跳
- **Axis regularization**：reaching 阶段惩罚 y 轴乱动（别横向 drift），opening 阶段惩罚 z 轴乱动（开门不该有垂直运动）

这些 term 看起来琐碎，但每一个都对应一个 sim-to-real 会爆的具体 failure mode。Ablation 里把 motion-aware 拿掉，real-world SR 从 0.80 跌到 0.64，动作变得 jerky，grasp 经常掉。

---

## Domain Randomization 怎么搞？

作者明确说：**focus 在 physics gap，不是 vision gap**。

Randomize 的东西：
- Object 位置和 yaw rotation
- Joint friction、stiffness、mass
- Grasp pose 在 y/z 轴上加 noise + spherical cone 的 rotation noise

为什么要 randomize physics？因为 policy 必须学到"遇到不同 friction/stiffness 的门，靠 history 推断出来并 adapt"，而不是记住"friction=0.3 时的最优 action"。

---

## 实验结果里最值得关注的几点

### 1. Open-loop 方法在 OpenDoor+ 上全线崩溃

OpenDoor 是开 15%（halfway），OpenDoor+ 是开 80%（near full）。Open-loop 方法（Where2Act, GAPartNet）在 OpenDoor 上还能看，到 OpenDoor+ 上直接跌到 0.02。

为什么？因为 80% 开度意味着 long-horizon circular motion。Open-loop 预测的几个 waypoint 在 revolute joint 的弧线上累积 error，根本对不上。Closed-loop 才能 correct 这些 error。

### 2. 最关键的 Ablation：W/o Impedance Control

Sim 里 SR 还是 0.84/0.90，real 里跌到 0.40/0.44。**这是整篇 paper 最重要的 single result**。

它直接证明：position control 在 sim 里是"作弊"通过的，real-world 的 safety stop 会把这个 cheating 暴露无遗。Impedance control 不是锦上添花，是 sim-to-real 的必要条件。

### 3. Failure Mode 分解

OpenDoor+ real-world 50 次失败：
- 6 次是 Grasping Stage 失败（SAM/GSNet 的 vision module 没预测好 grasp pose）
- 4 次是 Opening Stage 失败（manipulation policy 自己出错）

意味着如果 grasp 成功，opening policy 的 SR 是 40/44 = **90.9%**。

这个 decomposition 非常有 insight —— paper 的 manipulation policy 本身已经很强了，bottleneck 转移到了 upstream vision。Future work 的方向应该是改进 grasp prediction，而不是 manipulation policy。

---

## 几点 Karpathy 你可能会有的思考

### 1. 为什么不用 tactile sensor？

Paper 标题叫 "Feel More"，但 feel 的来源是 proprioception，不是真正的 tactile。如果加上 DIGIT 之类的 tactile sensor 作为额外的 history channel，policy 应该能更 fine-grained 地感知 contact 状态，可能能进一步 push SR。

### 2. 第一帧 vision 的 strong assumption

整个 pipeline 假设第一帧能成功预测 grasp pose。如果 object 被 occlude 或者 GSNet 挂了，后面全崩。一个更 robust 的设计是 active vision —— 当 policy 检测到 uncertainty 高时主动 request vision input。

### 3. Reward engineering 的 scalability

Table I 的 reward 有 10 个 term，每个都有 weight 要 tune。从 door/drawer scale 到 microwave、oven、scissors 这种新 category，可能要重新 design reward。这条路走到头，自然就是 LLM-generated reward (Eureka 那条线)。

### 4. Online Distillation 的 Generalization

这种 teacher-student with privileged info 的 pattern 可以 generalize 到很多 sim-to-real task。任何"sim 中知道 ground truth、real 中不知道"的 information 都可以这样处理：
- Tactile force distribution
- Multi-view observation
- Object 的 future trajectory（如果 sim 里 scripted）

这是一个 framework-level 的 insight，不局限于 articulated object。

### 5. 为什么 PPO 就够了？

Paper 用的就是 plain PPO，没有任何 RL algorithm 创新。所有 performance gain 来自 system engineering —— observation design、architecture、low-level controller、reward shaping、domain randomization。

这又一次印证：**applied RL 里 engineering >> algorithm**。好的 engineering + PPO 能 beat 花哨的新算法。这跟你 "Software 2.0" 的观点一致。

---

## 我的判断

这篇 paper 不是 fundamental breakthrough，但是一个 **engineering quality 极高的 system paper**。它把 locomotion 领域的 distillation trick、control theory 的 impedance control、manipulation 的 reward shaping 三个东西组合起来，解决了一个具体且重要的 open problem —— articulated object manipulation 的 sim-to-real。

84% real-world success rate 是 strong empirical evidence。最有价值的 insight 是 **impedance control 对 sim-to-real 的 necessity**（而非 enhancement），以及 **vision 的 task-localized use** 这两个 design pattern。

如果你想在这个方向继续做，我赌 tactile + active vision + LLM reward 这三件事任意一个加上去都能发 follow-up。

参考链接：
- 项目主页: https://watch-less-feel-more.github.io/
- PartNet-Mobility: https://partnet.mpi-inf.mpg.de/
- SAM: https://arxiv.org/abs/2304.02643
- Eureka (LLM reward): https://arxiv.org/abs/2310.12931
- RMA (locomotion distillation 原版): https://arxiv.org/abs/2107.04034
- Deep Whole-Body Control (online distillation 出处): https://arxiv.org/abs/2210.10044

---

# Watch Less, Feel More 论文深度解析

## 1. 论文核心动机：Articulated Object Manipulation 的本质困境

Articulated object (铰接物体) manipulation 与传统 rigid object manipulation 有一个根本性的不同 —— **物体本身就是一个 dynamic environment**。当你推一个杯子时，杯子只是被动的rigid body；当你开门时，门是一个有internal joint、friction、stiffness的动态系统，它会通过物理contact"反馈"给你它的内在属性。

论文指出两个核心难点：

1. **Articulation characteristics 只能通过contact识别**：两扇外观完全相同的门，pivot center可能偏移几厘米，friction可能差一个数量级，stiffness可能完全不同。Vision-only方法在这点上本质上information incomplete。

2. **Joint constraint问题**：如果robot action不tolerate object joint motion而强行执行command，会产生large force，损坏物体和robot。这是rigid manipulation里不存在的问题。

Karpathy你会注意到，这个问题的structure其实和**locomotion task**非常类似 —— terrain friction、slope、compliance这些environment parameter也是hidden的，agent必须通过proprioception history来infer。这大概是作者选择借鉴locomotion领域teacher-student distillation pipeline的直觉来源。

参考链接：
- 项目主页：https://watch-less-feel-more.github.io/
- PartNet-Mobility dataset: https://partnet.mpi-inf.mpg.de/
- IsaacGym: https://developer.nvidia.com/isaac-gym

---

## 2. 核心Intuition：人在黑暗中如何开门？

论文用了一个非常漂亮的intuition：**人类可以在完全黑暗中开门**。

假设你知道：
- 门把手的大致位置
- 门是left-hinged还是right-hinged

你完全可以不靠视觉完成开门动作。你的策略是：
1. 伸手朝estimated handle position移动
2. 一旦contact上，通过tactile和proprioception感受handle的位置和orientation
3. 施加pull force，同时感受门是否在动 —— 如果没有动，可能friction大，需要更用力；如果在动但偏离了expected circular trajectory，说明你对pivot center的估计有误
4. 持续根据"我做了什么action"和"实际发生了什么motion"的discrepancy来调整下一步action

这就是 **"Watch Less, Feel More"** 的字面含义。Vision只在第一帧用来获取high-level task specification（handle在哪、什么类型的门），之后整个manipulation过程都是proprioception-driven的closed-loop。

这个intuition有几个重要implication：
- Vision sim-to-real gap被**局部化**到了第一帧的grasp pose prediction，而不是每一步都需要vision feature
- History information implicitly encode了object的physical properties
- Action的compliance比precise tracking更重要

---

## 3. 方法架构详解

### 3.1 Action Space 设计

$$a^t \in \mathbb{R}^{11}$$

包含：
- $\Delta_{xyz}^t \in \mathbb{R}^3$：target delta position，end-effector的增量位置target
- $R^t \in \mathbb{R}^6$：target 6D orientation，用6D rotation representation而不是quaternion或euler，这是从Zhou et al.的工作继承的好习惯，避免topological discontinuity
- $G^t \in \mathbb{R}^1$：gripper action
- $k_p^t \in \mathbb{R}^1$：**impedance control的stiffness gain**，这是论文的核心创新之一

最终通过action scaler转换成 $c^t \in \mathbb{R}^9$ 的robot command。

**关键设计哲学**：single dexterous action prediction at a time，而不是short-horizon primitive。这避免了open-loop waypoint execution的rigidity。

### 3.2 Observation Space

$$o^t = [g^t, q^t, \delta^t, ee^t, \mathbb{1}_{grasp}^t, \tilde{r}_{pivot}^t, \tilde{r}_{radius}^t, \tilde{r}_{rh}^t] \in \mathbb{R}^{30}$$

变量解释：
- $g^t \in \mathbb{R}^7$：desired grasping pose (3D position + 4D quaternion)，由simulation的bounding box或real-world的GSNet预测
- $q^t \in \mathbb{R}^7$：robot joint configuration (7 DoF Franka)
- $\delta^t \in \mathbb{R}^1$：robot-object relative distance
- $ee^t \in \mathbb{R}^9$：end-effector pose (3D position + 6D rotation)
- $\mathbb{1}_{grasp}^t \in \mathbb{R}^1$：graspability signal，是distance-based和contact-aware的condition，**不是直接的open/close command**
- $\tilde{r}_{pivot}^t \in \mathbb{R}^3$：noisy pivot center，带noise的estimated旋转中心
- $\tilde{r}_{radius}^t \in \mathbb{R}^1$：noisy pivot radius
- $\tilde{r}_{rh}^t \in \mathbb{R}^1$：right-hinged boolean（哪边铰链）

**Privileged Observation**（只在sim中训练时用）：
$$o_{priv}^t = [r_{pivot}^t, r_{radius}^t, r_m^t, r_{stiff}^t, q_{obj}^t, \mathbb{1}_{grasped}^t] \in \mathbb{R}^8$$

- $r_{pivot}^t$：真实pivot center (ground truth)
- $r_{radius}^t$：真实pivot radius
- $r_m^t$：object mass
- $r_{stiff}^t$：object stiffness
- $q_{obj}^t$：object joint position（当前门/抽屉打开多少）
- $\mathbb{1}_{grasped}^t$：handle是否被抓取

这个design的核心思想是 **asymmetric information access during training**：teacher network能看到ground truth的physical parameters，从而学到"如果我知道object的真实属性，最优action是什么"；student network只能看history，但被supervised去infer这些privileged information的latent representation。

参考：
- 6D rotation representation: https://arxiv.org/abs/1812.07035 (Zhou et al.)
- GSNet grasp prediction: https://arxiv.org/abs/2102.09556

---

## 4. Online Policy Distillation：关键架构创新

### 4.1 传统Teacher-Student Pipeline的问题

传统做法（如Kumar et al.的RMA, Rapid Motor Adaptation）：
1. Stage 1: 训练teacher policy with privileged observation
2. Stage 2: Freeze teacher，训练student module去predict privileged latent from history

问题：**Realizability gap** —— Student可能无法完全reconstruct teacher看到的privileged information，两阶段训练导致student始终在追赶teacher，难以超越。

### 4.2 本文的Online Distillation

作者采用Fu et al. (Deep Whole-Body Control, CoRL 2023) 的思路，同时训练：

- **Privileged Observation Encoder $\phi$**：shallow MLP，把 $o_{priv}^t$ 编码成 $z^t \in \mathbb{R}^{20}$
- **Adaptation Module $\sigma$**：temporal architecture，从 $H=10$ 个历史的 $(o^t, a^{t-1})$ pairs 中infer出 $\tilde{z}^t \in \mathbb{R}^{20}$

Loss function包含：
$$\mathcal{L} = \mathcal{L}_{PPO} + \lambda \|z - \text{sg}[\tilde{z}]\|_2 + \|\text{sg}[z] - \tilde{z}\|_2$$

其中：
- $\text{sg}[\cdot]$：stop gradient operator，阻止梯度回传
- $\lambda$：linear schedule的权重，开始时小（避免policy过于conservative），逐渐增大
- 第一项 $\|z - \text{sg}[\tilde{z}]\|_2$：让teacher的latent $z$ 去匹配student的预测 $\tilde{z}$，但不让梯度流回student
- 第二项 $\|\text{sg}[z] - \tilde{z}\|_2$：让student的 $\tilde{z}$ 去匹配teacher的 $z$，但不让梯度流回teacher

这种**双向stop gradient**的设计避免了teacher和student相互"塌缩"到trivial solution。如果只用单向supervision，可能出现student给出garbage但teacher被迫去match的情况。

### 4.3 Actor Input构造

$$\text{Actor input} = [p^t \oplus z^t]$$

其中 $p^t = (o^t \oplus a^{t-1})$。在real-world deployment时，把 $z^t$ 替换为 $\tilde{z}^t$（Adaptation Module的输出），整个pipeline就是end-to-end closed-loop。

### 4.4 History Buffer的精简

Adaptation Module $\sigma$ 只保留action history中的：
- position command $\Delta_{xyz}^t$
- gripper command $G^t$  
- controller gain $k_p^t$

这个design很有意思 —— 这三个量恰好是"agent做出的explicit选择"，而observation部分保留full state。这种asymmetric design减少了input dimensionality，同时保留了最能体现"action-observation因果关系"的信息。

参考：
- RMA (Rapid Motor Adaptation): https://arxiv.org/abs/2107.04034
- Deep Whole-Body Control: https://arxiv.org/abs/2210.10044
- Teacher-student distillation in locomotion: https://arxiv.org/abs/2405.01402

---

## 5. Variable Impedance Control：物理层面的Compliance

### 5.1 Impedance Control的基础方程

$$M(\ddot{x_c} - \ddot{x_d}) + D(\dot{x_c} - \dot{x_d}) + K(x_c - x_d) = F_{ext}$$

变量解释：
- $M$：robot的mass-inertia matrix（对角positive definite）
- $D$：damping matrix（对角positive definite）
- $K$：stiffness matrix（对角positive definite）
- $x_d$：desired trajectory (RL policy输出的target)
- $x_c$：impedance controller输出的实际command trajectory
- $F_{ext}$：external force，来自robot-object interaction
- $\ddot{}, \dot{}$：分别表示对时间的二阶和一阶导数

这个方程的物理意义：robot的行为被建模为一个**mass-spring-damper系统**，目标轨迹 $x_d$ 是spring的rest position，$F_{ext}$ 是外部扰动。当 $F_{ext}$ 大时，实际command $x_c$ 会偏离 $x_d$，偏离程度由 $K$ 决定。

**Impedance control vs Position control vs Force control**：
- Position control：强制 $x_c = x_d$，忽略 $F_{ext}$，可能产生infinitely large force
- Force control：直接控制 $F_{ext}$，但需要explicit force sensor
- Impedance control：trade-off，$K$ 大时偏position control，$K$ 小时偏compliant

### 5.2 论文的具体实现

Actor输出 $a_{k_p}^t \in [-1, 1]$（通过tanh激活的RL standard），然后scaled：

$$c_{k_p}^t = \text{clip}(a_{k_p}^t, -1, 1) \times 40 + 100$$

这个scaling的intuition：
- $k_p$ 范围是 $[60, 140]$
- 中点100，对应 $a_{k_p}=0$
- 这个范围是empirically tuned，能在sim和real中都产生reasonable motion

然后 $K = \text{diag}(c_{k_p}^t, c_{k_p}^t, c_{k_p}^t, c_{k_p}^t, c_{k_p}^t, c_{k_p}^t)$（6D Cartesian），所有维度用同一个gain，简化了learning space。

**Critical damping condition**：
$$D = 2\sqrt{MK}$$

这是控制理论中的经典结果 —— critical damping保证系统在response to perturbation时最快回到稳态而不oscillate。这个analytic relation减少了一个需要learn的参数。

### 5.3 Stiffness的Adaptive Behavior

Figure 4展示了非常有趣的结果 —— **即使没有直接针对 $k_p$ 的reward**，policy自己学到了stage-dependent gain：
- **Reaching stage**（gripper离object远）：$k_p$ 高 → 接近position control，快速精确reach
- **Opening stage**（gripper已经抓住handle）：$k_p$ 低 → 柔软compliant，允许 $x_c$ 偏离 $x_d$，跟随object的实际joint motion

这种emergent behavior是RL最迷人的地方 —— 你设计了mechanism（variable impedance + articulated object physics），reward signal是task success，policy自己发现了stage-aware gain tuning是最优策略。

参考：
- Hogan impedance control original paper: https://ieeexplore.ieee.org/document/1087098
- Variable impedance learning review: https://arxiv.org/abs/2103.12996
- Sim-to-real with impedance: https://arxiv.org/abs/2305.17110 (IndustReal)

---

## 6. Reward Design：多阶段End-to-End Manipulation

这是论文中engineering最heavy的部分。Articulated object manipulation本质是multi-staged task（reach → grasp → open），但作者想训成single end-to-end policy而不是hierarchical options。

### 6.1 Task-Aware Rewards（确保正确顺序）

定义几个key indicator：
- $\mathbb{1}_d = \mathbb{1}[\delta \leq 0.05]$：距离够近
- $\mathbb{1}_{dy} = \mathbb{1}[0.02 \leq \delta \leq 0.08]$：在"接近但还没到"的range
- $\mathbb{1}_g = \mathbb{1}[\delta \leq 0.015 \wedge \mathbb{1}_{contact}]$：成功grasp

**Success reward**:
$$r_{success} = \mathbb{1}_d \times 0.5^{\mathbb{1}_g} \times \mathbb{1}_s \times 40$$

其中 $\mathbb{1}_s$ 是task success indicator（door开了>15%/80% 或 drawer开了>20%/80%）。$0.5^{\mathbb{1}_g}$ 这个term很巧妙：如果没grasp（$\mathbb{1}_g=0$），weight是1；如果grasp了（$\mathbb{1}_g=1$），weight是0.5。这避免了policy"不grasp就硬掰"的cheating behavior。

**Distance reward**:
$$r_{dist} = \exp(-10 \times 2\delta^{0.5}) \times 0.8^{\mathbb{1}_g}$$

exponential shaping function，$0.8^{\mathbb{1}_g}$ 让grasp之后的distance reward衰减（因为已经grasp了，distance不再是主要目标）。

**Object state reward**:
$$r_{obj} = q_{obj} \times 0.5^{\mathbb{1}_g} \times 0.5^{\mathbb{1}_d} \times w_{len}$$

$q_{obj}$ 是joint position（开了多少），$w_{len}$ 是episode length weight。鼓励"已经grasp且distance够近"时才鼓励opening。

**Grasp reward**:
$$r_{grasp} = 0.2 \times \mathbb{1}_g$$

小的constant reward for achieving grasp，引导exploration。

### 6.2 Motion-Aware Rewards（确保平滑可部署）

**Energy penalty**:
$$r_{energy} = -0.05 \times \sum (\tau \dot{q})^{0.5} \times \mathbb{1}_{\varepsilon}$$

$\tau$ 是joint torque，$\dot{q}$ 是joint velocity，$(\tau \dot{q})$ 是instantaneous power，开根号是sublinear penalty。$\mathbb{1}_{\varepsilon}$ 是某种condition mask（论文没明确，推测是"已grasp"之后才penalize energy，避免reaching阶段被penalty拖累）。

**Tracking rewards**:
$$r_{pos} = \exp(-4|c_{pos} - ee_{pos}|) \times \mathbb{1}_d$$
$$r_{ori} = \exp(-4\Delta(c_{ori} - ee_{ori})) \times \mathbb{1}_d$$

$c_{pos}, c_{ori}$ 是impedance controller output，$ee_{pos}, ee_{ori}$ 是actual end-effector pose。这个reward让policy倾向于输出"能被impedance controller准确track"的command，避免提出物理上infeasible的target。

**Smoothness penalty**:
$$r_{smooth} = -0.001 \times \sum \mathbb{1}_{[\text{sgn}(a_t) \neq \text{sgn}(a_{t-1})]} \times (a_t - a_{t-1})$$

这个term惩罚action方向的反复变化，避免oscillation。

**Axis regularization**:
$$r_y = -0.005 \times \mathbb{1}_{dy} \times (a_t[y] \times 15)^2$$
$$r_z = -0.07 \times \mathbb{1}_g \times (a_t[z] \times 15)^2$$

这两个是task-specific的prior knowledge注入：
- $r_y$：在"接近但还没grasp"阶段，惩罚y轴动作（避免横向drift）
- $r_z$：在"已经grasp"阶段，惩罚z轴动作（开门开抽屉不应该有垂直方向motion）

### 6.3 Reward设计的Intuition总结

这个reward system的complexity反映了一个deep issue in RL for manipulation：**multi-staged task的reward shaping本质上是个program synthesis问题**。每个term对应一个task structure的inductive bias，weights的tuning是manual的。这也是为什么很多recent工作开始用LLM自动generate reward（Eureka, text2reward等）。

Karpathy你可能会想，这种heavy engineering的reward design是否还能scale？我的看法是这篇paper的reward design虽然复杂，但每个term都有clear physical interpretation，是"principled engineering"而非blind tuning。但确实，如果要去generalize到更多task categories，这种手工design会become bottleneck。

参考：
- Eureka (LLM-generated reward): https://arxiv.org/abs/2310.12931
- text2reward: https://arxiv.org/abs/2310.09116
- Reward shaping theory: https://arxiv.org/abs/1606.06152 (Ng et al. classic)

---

## 7. Domain Randomization策略

论文的randomization分两类：

### 7.1 Geometric Randomization
- Object position：覆盖合理workspace
- Object yaw rotation：real-world中object放置朝向有variation
- Desired grasping pose：从bounding box推断后，加入y/z axis的noise + spherical cone的rotation noise

### 7.2 Physical Randomization
- Joint friction
- Joint stiffness  
- Object mass

这些randomization的核心目的：**让policy学到"通过contact和history infer物理参数"的能力**，而非记住specific physics parameter下的optimal action。这与system identification literature的思路一脉相承。

注意：作者特别强调"physics gap"而非"vision gap"是他们的主要focus。这从侧面说明，对于已经decided不用vision feature as policy input的方法来说，sim-to-real的核心challenge转移到了physics fidelity上。

参考：
- Domain randomization classic: https://arxiv.org/abs/1703.06907 (OpenAI Hand)
- Automatic domain randomization: https://arxiv.org/abs/1910.07113

---

## 8. 实验结果深度分析

### 8.1 Simulation Baselines Comparison (Table II)

| Method | Type | OpenDoor Train/Test | OpenDrawer Train/Test | OpenDoor+ Train/Test | OpenDrawer+ Train/Test |
|--------|------|---------------------|----------------------|---------------------|----------------------|
| PPO | Closed-loop | 0.04/0.05 | 0.09/0.11 | 0.02/0.02 | 0.03/0.02 |
| Where2Act | Open-loop | 0.22/0.14 | 0.31/0.27 | 0.02/0.02 | 0.01/0.01 |
| RGBManip | Closed-loop | 0.62/0.59 | 0.63/0.67 | 0.38/0.41 | 0.49/0.42 |
| GAPartNet | Open-loop | 0.70/0.75 | 0.51/0.59 | 0.40/0.44 | 0.45/0.49 |
| PartManip | Closed-loop | 0.75/0.70 | 0.83/0.77 | 0.68/0.57 | 0.62/0.59 |
| **Ours** | Closed-loop | **0.96/0.95** | **0.97/0.96** | **0.96/0.93** | **0.97/0.96** |

几个关键观察：

1. **Open-loop methods (Where2Act, GAPartNet)在OpenDoor+/OpenDrawer+上dramatically fail**。因为这些任务要求达到80% joint limit，意味着long-horizon circular/prismatic motion。Open-loop预测的waypoint在physics execution时会累积error，特别是revolute joint的circular trajectory上。

2. **PPO baseline惨烈失败**（success rate < 5%）。这说明plain PPO + state observation完全无法处理multi-staged articulated manipulation task。这从侧面印证了reward shaping和architectural choices的重要性。

3. **本文方法train/test gap极小**（OpenDoor+从0.96→0.93，OpenDrawer+从0.97→0.96）。这种generalization gap接近0的表现，要么是method真的robust，要么是test set不够distribution shift。考虑到作者用的PartNet-Mobility数据集，train/test split应该是category-level，物体几何差异可能有限。

### 8.2 Real-World Ablation (Table III)

| Method | OpenDoor+ Real | OpenDrawer+ Real |
|--------|---------------|------------------|
| W/o Distillation | 0.62 | 0.60 |
| W/o Imp. Control | 0.40 | 0.44 |
| W/o Regularization | 0.64 | 0.70 |
| W/o Randomization | 0.66 | 0.64 |
| **Ours** | **0.80** | **0.84** |

**最dramatic的ablation是W/o Impedance Control**：sim中success rate还是0.84/0.90，但real-world跌到0.40/0.44。这是paper的核心claim之一 —— **position control在sim中"作弊"通过**，因为sim的physics引擎允许large torque且没有safety stop；real-world中large torque会触发Franka的emergency stop。这是典型的"sim掩盖了method的缺陷"。

这个ablation是整个paper最重要的result，它直接论证了impedance control对sim-to-real transfer的necessity，而非仅仅method performance enhancement。

### 8.3 Failure Mode分析

作者做了一个很有价值的分析：把real-world failure分解为：
- **Grasping Stage failure**：grasp pose prediction失败（来自SAM + GSNet的vision pipeline）
- **Opening Stage failure**：manipulation policy本身失败

OpenDoor+：50次中6次Grasping Stage失败，4次Opening Stage失败。这意味着如果grasp成功，opening policy的SR是 40/44 ≈ **90.9%**。

OpenDrawer+：50次中7次Grasping Stage失败，1次Opening Stage失败。如果grasp成功，SR是 42/43 ≈ **97.7%**。

这个分析非常有insight —— 说明paper的核心method（manipulation policy with history + impedance）本身已经非常robust，real-world的bottleneck转移到了上游vision module。这是一个positive signal，意味着future work应该focus on更好的grasp prediction，而不是manipulation policy本身。

---

## 9. 与相关工作Positioning

### 9.1 vs Affordance-based Methods (Where2Act, VAT-MART, GAPartNet)

这类方法predict visual affordance heatmap → choose contact point → predict action/trajectory → open-loop execute。

**根本问题**：open-loop execution忽略了contact后的physical interaction。对于short-horizon task（开15%）可能勉强work，但long-horizon task（开80%）会累积error。

### 9.2 vs Vision-based RL (PartManip, RGBManip)

这类方法用pointcloud或RGB作为policy input，closed-loop执行。

**根本问题**：
1. Vision sim-to-real gap很大（渲染domain gap + sensor noise gap）
2. 每一步都run vision inference，computational cost高
3. Manipulation过程中actionable part被occlude时vision失效

本文的方法只在第一帧用vision（获取grasp pose），之后纯proprioception。这本质上是**task structure decomposition**：vision负责"where is what"，proprioception负责"how to interact"。

### 9.3 vs Locomotion Adaptation Methods (RMA, Deep WBC)

本文的online distillation架构直接借鉴locomotion领域。差异点：
- Locomotion的privileged info主要是terrain parameters（friction, slope, mass of robot）
- Manipulation的privileged info是object parameters（pivot center, radius, mass, stiffness, joint position）
- Locomotion的action是joint torque/position，essentially unconstrained
- Manipulation的action还要考虑rigid grasp constraint和joint limit

本文的contribution之一是show这种distillation架构在fine-manipulation上也works，且通过impedance control解决了locomotion里不存在的compliance issue。

参考：
- Where2Act: https://arxiv.org/abs/2101.07973
- VAT-MART: https://arxiv.org/abs/2106.14440
- PartManip: https://arxiv.org/abs/2305.12785
- RGBManip: https://arxiv.org/abs/2310.09169

---

## 10. Karpathy视角的几点思考

### 10.1 Information Bottleneck设计哲学

这篇paper有一个深层设计哲学：**task-specific information bottleneck**。Vision信息量极大但redundant也多；proprioception信息量小但直接task-relevant。通过把vision限制在第一帧的grasp pose prediction，作者人为构造了一个information bottleneck，forcing policy去exploit history information。

这种思路让我想到VAE的latent bottleneck —— 通过限制capacity，force representation学到most informative的structure。RL里的类似思想见数据效率相关文献，比如DrQ-v2的image augmentation。

### 10.2 Privileged Information的Asymmetric Training

Teacher-student with privileged information是recent RL的一个powerful pattern。核心insight：**ground truth information在simulation中是free的，在real-world中是expensive的**。Training时充分利用free information，deployment时只用cheap information，通过distillation bridge gap。

这种pattern的generalization空间很大 —— 任何sim-to-real task都可以考虑"什么信息sim中知道但real中不知道"，然后design privileged encoder + adaptation module。例如：
- Tactile sensing: sim中可以access contact normal force distribution，real中只有low-res tactile sensor
- Multi-camera: sim中可以从任意角度观察，real中只有wrist camera
- Future trajectory: sim中知道object即将怎么动（如果 scripted），real中不知道

### 10.3 Reward Engineering vs RL Algorithm

这篇paper用的是plain PPO，没有任何RL algorithm层面的innovation。所有的performance gain来自：
1. Observation/action space design
2. Architecture (adaptation module + privileged encoder)
3. Low-level controller (impedance)
4. Reward shaping
5. Domain randomization

这印证了一个observation —— **在applied RL中，system engineering >> algorithm innovation**。PPO加上好的engineering能beat花哨的新算法，这跟Karpathy你在"Software 2.0"和很多talk中提到的观点一致。

### 10.4 Limitations和Future Directions

Paper没explicitly discuss的几个limitation：

1. **第一帧vision的strong assumption**：假设第一帧object可见且grasp pose可预测。如果object被occlude或grasp prediction失败，整个pipeline崩溃。Real-world failure mode也证实了这一点。

2. **Articulation parameter的noise assumption**：$\tilde{r}_{pivot}, \tilde{r}_{radius}, \tilde{r}_{rh}$ 是noisy但unimodal的。如果object类型未知（比如不知道是door还是drawer），这个prior可能inadequate。

3. **Single grasp throughout episode**：如果opening过程中gripper slip了，policy没有re-grasp的能力。Long-horizon manipulation可能需要multiple grasp attempts。

4. **No tactile sensing**：Paper title叫"Feel More"，但实际"feeling"来源是proprioception (joint position, ee pose)而非tactile sensor。Real tactile sensing能提供更rich的contact information。

5. **Reward design的task-specificity**：从door/drawer generalization到其他articulated object（如scissors, pliers, microwave）可能需要重新design reward。

Future direction的几个猜想：
- **Tactile modality integration**：把tactile sensing作为additional history channel，真正的"feel more"
- **LLM-guided reward generation**：用LLM自动generate类似Table I的reward function，scale到更多task categories
- **Active vision**：不是只第一帧用vision，而是在uncertainty高时主动trigger vision（比如grasp failure detection）
- **Bi-manual manipulation**：很多articulated object需要两只手（如打开jam jar），这个framework可以extend

参考：
- Software 2.0: https://karpathy.medium.com/software-2-0-a64152b37c35
- Tactile sensing for manipulation: https://arxiv.org/abs/2203.00155 (DIGIT)
- LLM for reward design: https://arxiv.org/abs/2310.12931 (Eureka)
- Active vision manipulation: https://arxiv.org/abs/2310.09069 (ImageManip)

---

## 11. 总结：这篇paper真正教给我们什么

1. **Vision sim-to-real gap可以通过task decomposition局部化**：只在high-level task specification时用vision，low-level execution用proprioception。

2. **Impedance control不是icing on cake，而是sim-to-real的必要条件**：Position control在sim中能"作弊"通过，real-world的safety mechanism会暴露这个cheating。

3. **History information可以implicit encode object physical properties**：不需要explicit system identification，temporal model + supervised regularization就够了。

4. **Online distillation > two-stage distillation**：joint training避免realizability gap。

5. **Multi-staged task的end-to-end training需要stage-aware reward design**：纯task success reward不足以引导multi-staged behavior。

6. **Real-world failure mode分析极其有价值**：把失败归因到upstream vision vs downstream policy，能inform future work方向。

这篇paper在applied RL for manipulation领域是a solid piece of work，engineering quality很高，每个design choice都有clear justification。虽然不fundamentally novel任何一个单独component，但把distillation + impedance + reward shaping + domain randomization组合在一起解决了articulated object manipulation的open problem，并提供了84% real-world success rate的strong empirical evidence。

最终，这个工作的最大价值在于show了 **proper physical grounding (impedance control) + proper information flow (history-based adaptation) + proper task decomposition (vision as proxy)** 三者结合能实现reliable sim-to-real for contact-rich manipulation —— 这对整个robot learning community都是valuable lesson。
