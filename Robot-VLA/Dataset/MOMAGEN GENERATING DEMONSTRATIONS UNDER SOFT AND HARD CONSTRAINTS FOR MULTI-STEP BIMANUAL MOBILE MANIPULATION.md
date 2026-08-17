---
source_pdf: MOMAGEN GENERATING DEMONSTRATIONS UNDER SOFT AND HARD CONSTRAINTS FOR
  MULTI-STEP BIMANUAL MOBILE MANIPULATION.pdf
paper_sha256: a719fb702505eeec2bfbd13dda2c6a7008131c075b003bff654d44938d45ce9f
processed_at: '2026-08-05T20:07:40-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 MOMAGEN

## 一句话版本

你给人演示一次怎么做家务（比如拿杯子），MOMAGEN 就能在仿真里自动变出上千种不同的做法，让机器人学会在各种乱七八糟的场景下都能干这件事。

---

## 为什么要搞这个

假设你想教一个带轮子的双臂机器人（bimanual mobile manipulator，比如 Galaxea R1）干家务。最直接的方法是 teleoperation——你拿个 VR 手柄控制它走过去、抓东西、放东西。但这玩意儿控制起来是真累：base 要走、torso 要转、两只 arm 各 7 个 DoF、还要看 camera……人类操作员的认知负荷直接爆表。收集一条像样的 demo 要 1-3 分钟，而且质量还不稳定。

如果你要训一个 visuomotor policy（从 RGB 图像直接预测 joint action 的神经网络），至少要上千条 demo。纯靠人 teleop，collect 一个 task 的数据可能要几十个小时。这在 scale 上根本走不通。

所以大家早就想到了一个偷懒办法：**我能不能只收集 1 条 demo，然后在仿真里自动变出 1000 条？** 这就是 X-Gen 家族（MimicGen / SkillMimicGen / DexMimicGen / DemoGen / PhysicsGen）干的事。

参考：MimicGen https://arxiv.org/abs/2310.17596

---

## 以前的方法怎么 work 的（在 tabletop 上）

想象你有一条人操作的 demo：机器人走过去抓杯子。X-Gen 的核心 trick 是：

1. 把这条 demo 切成 subtask（navigate → pregrasp → grasp → lift）
2. 对每个 contact-rich subtask，记录 gripper 相对 object 的位姿——比如 "gripper 在杯子把手正前方 5cm，朝向 30°"
3. 现在把杯子挪到桌上另一个位置，甚至转个方向
4. 关键假设：**gripper 相对 object 的几何关系不变**，所以新的 gripper world 位姿 = 新 object world 位姿 × 原来的相对位姿
5. 用 IK 把 gripper 位姿还原成 joint angle，丢给 motion planner 检查可行性
6. 仿真里跑一遍，成功的就留下来当训练数据

这个 trick 在 fixed-base tabletop 上 work 得很好。MimicGen 能把 1 条 demo 扩成 200-300 条，DexMimicGen 处理 dexterous bimanual 也能 work。

---

## 一旦机器人会走路，这个 trick 就崩了

Mobile manipulator 多了一个 base。以前的方法对 base 的处理是——**直接 replay 原来的 base 轨迹**。这在 object 只挪了 15cm（D0 randomization）时勉强 work，但只要 object 挪远一点（D1，整个 countertop 范围随机），replay 的 base pose 就让 arm 根本够不到 object。

这叫 **reachability 问题**。

还有一个更隐蔽的问题：base 会动，head camera 跟着动。原来 demo 里 camera 一路看着杯子走过来，现在你 replay base 轨迹但杯子已经挪到另一边了，camera 可能全程对着空气拍。policy 训练的时候输入图像里根本没有 target object，学到的就是 garbage。

这叫 **visibility 问题**。

Paper 里 Table 2 和 Table 3 用数字说话：baseline 在 D1/D2 下 data generation success rate 直接归零（因为 reachability 挂了），在 D0 下即使能 generate 出来，visibility ratio 也只有 40%（MOMAGEN 是 86%）。

---

## MOMAGEN 的核心 idea：让 base 也参与采样

以前的方法把 base trajectory 当成"神圣不可侵犯"的原样回放对象。MOMAGEN 的核心 insight 是：**base pose 也应该被重新采样**。

具体来说，对每个 subtask，MOMAGEN 做这件事：

1. 看看当前 base 位置能不能让 arm 够到新位置的 object（IK 检查）
2. 看看当前 camera 朝向能不能看到 object（visibility 检查）
3. 如果不行——**别硬撑，挪车**
4. 在 object 附近 sample 一个新的 base 位置 + torso 角度 + camera 朝向
5. 再检查 IK + visibility，不行就继续 sample
6. 找到 valid 的配置后，用 motion planner 从当前 base 位置 plan 一条路过去（这一路 camera 尽量盯着 object 看）
7. 到位后，arm 做 motion planning 到 pregrasp，然后 task-space replay 那条 contact-rich 轨迹
8. 最后 retraction——arm 收回来贴紧身体，方便下一阶段 navigation

听起来简单，但这就是为什么 MOMAGEN 能处理 D1（整个 furniture 范围随机）而 baseline 全挂——因为 baseline 根本不会挪车。

Algorithm 1 的 line 8 那一句 "Sample new T_base" 是整篇 paper 的灵魂。

---

## Hard constraint vs Soft constraint 的直觉

Paper 把 constraint 分成两类：

**Hard constraint（做不到就废）**：
- Reachability：arm 够不到 object，task 不可能完成
- Visibility during manipulation：抓取那一刻 camera 必须看到 object，否则 policy 学不到 visual servoing

**Soft constraint（做到了更好）**：
- Visibility during navigation：导航过程中能看到 object 是好事，policy 能提前锁住目标，但不强求
- Retraction：manipulation 完了把 arm 收回来，让 base 走起来更安全

这个划分非常 intuition-driven。你想想，policy 训练的时候如果关键帧（grasp 前那一帧）看不到 object，那 action 和 image 之间根本没有 correspondence，policy 只能瞎猜。但 navigation 过程中偶尔看不到 object 几帧，policy 还有 history context 可以 rely on，所以 soft 就够了。

实验上 Figure 6d 把这个验证得很清楚：Tidy Table D0 上，去掉 visibility constraint 的 policy success rate 只有 0.05，加上 full visibility constraint 是 0.40。8 倍差距，visibility constraint 是 game-changer。

---

## 公式 1 的直觉翻译

Paper 的公式 1 看起来吓人，其实就是在说：

"我要找一个 action sequence，让机器人从起点走到终点完成任务。这个过程中：
- 物理得对（dynamics）
- 关节不能超限（kinematic）
- 不能撞东西（collision）
- 关键帧得看到 object（visibility）
- 抓取那一刻 gripper 相对 object 的姿势得和 source demo 一样（contact transform）
- 最后得真的完成任务（success）"

然后在这个前提下，尽量让 navigation 过程也能看到 object，尽量让 arm 收回来。

**就这么回事**。所有 X-Gen 方法都可以套进这个模板，只是它们用了更弱/更少的 constraint。MimicGen 只检查 success；SkillMimicGen 加了 collision-free；DexMimicGen 加了 bimanual temporal coordination；MOMAGEN 第一次加了 visibility。

---

## 实验上到底有多 impressive

### Data generation success rate

D0（小扰动）下 MOMAGEN 平均 63%，baseline 也能做到差不多。但 D1（大范围随机）下 baseline 直接 0%，MOMAGEN 还能 20-66%。D2（加 obstacles）下 MOMAGEN 还有 7-47%。

这个数字的意义是：**MOMAGEN 能 generate 出 baseline 根本 generate 不出来的数据**。不是"多一点"，是"从 0 到有"。

### Visibility ratio

Tidy Table D0 下 MOMAGEN 的 object visibility 是 86%，baseline 是 40%。这意味着同样 1000 条 demo，MOMAGEN 的数据里 target object 出现在 camera 里的帧数是 baseline 的 2 倍多。policy 学的时候见过的"有用的 visual observation"多了一倍，自然学得好。

### Policy performance

WB-VIMA（37M 参数的小 model，from scratch 训练）在 Tidy Table D0 上：
- MOMAGEN data: 0.40 success
- Baseline data: 0.05 success

π₀（3.3B 参数的大 VLA model，LoRA fine-tune）在 Pick Cup D1 上：
- MOMAGEN data: 能学到 0.25
- Baseline: 0（因为 baseline 在 D1 上根本 generate 不出数据）

### Sim-to-real（最香的部分）

Pick Cup D0 task，用 Galaxea R1 真机：
- WB-VIMA：1000 条 MOMAGEN synthetic data pretrain + 40 条 real demo fine-tune → 10% success；纯 40 条 real demo → 0%
- π₀：1000 条 synthetic + 40 条 real → 60% success；纯 40 条 real → 0%

**注意 π₀ 已经在 10k+ hours 的多 robot 数据上预训练过了**，按理说它的 prior 应该很强。但面对一个全新的 robot embodiment + 全新的 task setup，40 条 real demo 根本不够 fine-tune 出能用的 policy（0% success）。加上 1000 条 MOMAGEN synthetic data 之后跳到 60%。

这说明 MOMAGEN 的 data 不是在"补数据量"，是在"补 task-specific 的 visual-action prior"。这 1000 条 synthetic data 告诉 π₀："这个机器人在这张桌子前抓这个杯子大概长什么样、动作怎么走"，然后 40 条 real data 再把 sim-to-real gap 补上。

这个结果的 practical 意义巨大——意味着你拿到一个新机器人，只需要 teleop 1 次 + 采集 40 次真实数据，就能部署一个能用的 policy。相比传统几百条 real teleop 的成本，这是数量级的降低。

参考：π₀ https://arxiv.org/abs/2410.24164

---

## 为什么 visibility 这么重要（我的直觉理解）

Karpathy 你教过神经网络，应该比我更清楚：policy 学习的本质是建立 $s \to a$ 的 mapping。如果 state $s$ 里的图像根本看不到 target object，那这个 mapping 就退化成"从 proprioception 猜 action"——因为视觉通道是 garbage。

更糟的是，如果数据里有的帧看到 object、有的帧没看到，policy 会学到一种"看到就抓、没看到就乱动"的混乱策略。它没法学会"持续 visual servoing 跟踪 object"这种行为模式。

MOMAGEN 的 visibility constraint 本质上是在 guarantee 训练数据的"视觉信号质量"——每条 trajectory 里 target object 都大量出现在 camera 里，policy 能学到"从远到近持续 track object → 调整 arm 朝向 object → grasp"这种连贯的 visuomotor behavior。

这和人类学动作的方式其实有点像：你教小孩抓东西，也会让他眼睛盯着东西看，而不是让他瞎摸。

---

## 1 个 source demo 这个 claim 要打折看

Paper 反复强调 "single source demo"（$N_{src}=1$），这个在 teleop 层面是真的——人只 teleop 了一次。但 Algorithm 1 之前有个 **annotation 步骤**：人要手动标注每个 subtask 的 target object、pregrasp timestep、end timestep、retraction type。

Figure 2 里那个 annotated subtask 的示意图，背后是人手动标的。对 4 个 task，每个 task 1-3 分钟的 demo，可能要拆成 5-10 个 subtask，每个 subtask 标几个 annotation point。这个工作量不算大（可能 10-30 分钟），但也不是 zero。

所以实际的 human cost 是：**1 次 teleop + 1 次 subtask annotation + 1 次 scene setup**。相比 teleop 1000 次，仍然是巨大节省，但"single demo"这个 marketing 要理解对——它指的是"single teleop session"，不是"single human interaction"。

---

## 我觉得这篇 paper 最大的贡献

不是具体的 algorithm，是那个 **formulation**。

Paper 把所有 X-Gen 方法统一写成 constrained optimization，然后指出每个方法只是用了不同的 constraint 组合。这意味着：

1. **后续工作不需要发明新 trick，只需要设计新 constraint**。比如你想处理 deformable object，就加一个 "deformation energy" constraint；想处理 multi-object interaction，就加一个 "object-object contact" constraint。
2. **Constraint 的 hard/soft 划分是 design choice**，应该和下游 policy 的 property 共同设计。Short-history policy 需要 hard visibility，long-history policy 可能 soft 就够。
3. **整个 data generation 领域有了一个 principled foundation**，不再是"each paper 自己造一个轮子"。

这种统一框架的贡献，长期看比单个 algorithm 更有价值。就像 RL 有了 Bellman equation 之后，所有 RL algorithm 都是它的不同 approximation。

---

## 延伸：这对 VLA 时代意味着什么

π₀ / π₀.5 / OpenVLA / RT-2 这些 VLA model 都在追求"一个 model 跨所有 robot"。但现实是，每换一个 robot embodiment，还是要 fine-tune。问题是：

- Real teleop data 极贵（尤其 bimanual mobile manipulator）
- Pretrained VLA 的 prior 很强但不够 task-specific

MOMAGEN 这种 "1 real demo → 1000 synthetic demos + 40 real demos fine-tune → 可用 policy" 的 pipeline，可能就是 VLA 时代的**标准 embodiment adaptation 范式**。你拿到新机器人，先用 MOMAGEN 在仿真里生成一堆"这个机器人干这个任务"的合成数据，pretrain 一下，再用少量 real data fine-tune 填补 sim-to-real gap。

这比纯 sim-to-real 更现实（因为 sim-to-real gap 总是在），比纯 real teleop 更便宜（因为 1 vs 1000）。MOMAGEN 找到了一个 sweet spot。

参考：
- π₀.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://openvla.github.io/
- RT-2: https://arxiv.org/abs/2307.15818

---

## 如果我是 reviewer 会 push 的点

1. **$\mathcal{G}_{\mathrm{vis}}$ 的具体数学形式没给**。是 object bounding box 投影面积 > threshold？还是 frustum check + ray casting 检查 occlusion？这影响 reproducibility。
2. **Base sampling 的 "heuristics" 没写清楚**。Line 8 说 "sample new T_base"，但在 object 周围什么范围 sample？uniform？gaussian？rejection sampling？这直接决定 data generation efficiency。
3. **Soft visibility cost 的 weight 怎么设的**？太大会让 navigation 变僵硬（camera 死锁在 object 上），太小没用。Paper 没说这个 hyperparameter。
4. **Cross-embodiment 只在 Pick Cup 上 demo 了**，这个 task 太简单。在 Clean Frying Pan 这种 contact-rich bimanual task 上能不能 cross-embodiment？怀疑。
5. **Sim-to-real 只在 Pick Cup 上做**，而且 success rate 只有 10%（WB-VIMA）/ 60%（π₀）。更复杂的 task（Tidy Table, Clean Pan）能不能 sim-to-real？paper 没试。

这些都不影响 paper 的核心 contribution，但说明 MOMAGEN 离"通用 solution"还有距离。

---

## 最后的人话总结

**这篇 paper 在说**：教带轮子的双臂机器人干家务，以前的方法只能处理"东西稍微挪一点"的情况，东西挪远一点或者加了障碍物就崩了。MOMAGEN 让机器人学会"东西够不着就挪车过去，眼睛看不到就转个头看"，然后在仿真里自动生成上千种不同场景的练习数据。用这些数据训出来的 policy，只需要 40 条真实数据 fine-tune 就能在真机上干活。

**核心 insight**：训 visuomotor policy 的数据里，object 必须在 camera 里看得见，否则 policy 学不到东西。这个听起来像废话，但以前的方法根本没保证这一点。MOMAGEN 用 visibility constraint 强制保证，效果立竿见影。

**最重要的贡献**：把整个"自动生成 demonstration data"领域统一到一个 constrained optimization 框架里，后续工作可以在这个框架里设计新的 constraint，而不是各搞各的。

Project page: https://momagen.github.io/

---

# MOMAGEN: 单 Demo 驱动的 Bimanual Mobile Manipulation 数据生成

Karpathy 你好，这篇 paper 来自 Stanford (Li Fei-Fei lab) 和 UT Austin (Roberto Martín-Martín) 的合作，第一作者 Chengshu Li 和 Mengdi Xu 等共同贡献。它本质上是把 X-Gen 家族的"seed demo → 大规模合成"范式，从 tabletop / fixed-base 推到真正难的场景：**bimanual + mobile + multi-step**。下面我会把它彻底拆开，包括 formulation 的每个变量、Algorithm 1 的逻辑、constraint 设计的动机、以及实验里的关键数字。

---

## 1. 为什么 X-Gen 家族在 mobile manipulation 上崩了

X-Gen 的核心思路（MimicGen → SkillMimicGen → DexMimicGen → DemoGen → PhysicsGen）是：
1. 让人 teleop 几条 source demo
2. 把 demo 拆成 subtask，每个 subtask 拿到 end-effector (EEF) 相对于 object 的相对位姿 $\mathbf{T}_{o_i}^{E_k}$
3. 在新的随机化场景里，把 source demo 的 EEF 轨迹按新 object 位姿做 rigid transform，再做 motion planning 还原成 joint 轨迹
4. 在仿真里 validate，保留成功的，作为训练数据

这套方法对 fixed-base bimanual 工作良好，但在 mobile manipulator 上有两个深层 failure mode：

**(a) Reachability**：之前的方法对 base 轨迹的处理方式是 **naive replay**——把 source demo 的 base 轨迹原样回放。在 D0 这种小扰动下还能 work，但只要 object 在 countertop 上稍微挪远一点，replay 的 base pose 就让 EEF 完全够不到 object。Paper 里 Table 2 里 baseline 在 D1/D2 下 success rate 直接归零，就是这个原因。

**(b) Visibility**：mobile base 让 head camera 变成 active camera，naive replay base 轨迹，object 极易飘出 FoV，导致后续 visuomotor policy 看不到目标，action 就退化为乱猜。Paper Table 3 显示 baseline 在 Tidy Table 上 visibility ratio 只有 0.40，而 MOMAGEN 是 0.86——这种 visibility 差距直接影响下游 policy 性能。

这两点合起来意味着：**mobile manipulation 的 data generation 不是简单加一维 base action，而是要同时优化 base pose、torso/camera pose、arm pose 三者，并且这三个变量被 reachability / visibility 这种 task-relevant 几何约束强烈耦合**。这是 paper 的核心 insight。

参考链接：
- MimicGen: https://arxiv.org/abs/2310.17596
- SkillMimicGen: https://openreview.net/forum?id=YOFrRTDC6d
- DexMimicGen: https://arxiv.org/abs/2503.05652 (Behavior Robot Suite)
- DemoGen: https://arxiv.org/abs/2502.16932
- PhysicsGen: https://arxiv.org/abs/2502.20382

---

## 2. 核心 Formulation：Constrained Optimization（公式 1）

Paper 把所有 data generation 方法统一写成一个 constrained optimization。这个 formulation 是整篇文章的"骨架"，后面所有方法（包括 X-Gen 家族）都是它的不同实例化。

$$
\underset{a_{t \in [T]}}{\arg\min} \quad \mathcal{L}(\cdot) \quad \text{s.t.} \quad 
\begin{cases}
s_{t+1} = f(s_t, a_t), & \forall t \in [T] \quad \text{(dynamics)} \\
\mathcal{G}_{\mathrm{kin}}(s_t, a_t) \leq 0, & \forall t \in [T] \quad \text{(kinematic feasibility)} \\
\mathcal{G}_{\mathrm{coll}}(s_t, a_t) \geq 0, & \forall t \in [T] \quad \text{(collision-free)} \\
\mathcal{G}_{\mathrm{vis}}(s_t, a_t, o_{i(t)}) \leq 0, & \forall t \in [T] \quad \text{(visibility)} \\
\mathbf{T}_W^{E_k} = \mathbf{T}_W^{o_i} (\mathbf{T}_W^{o_{i,src}})^{-1} \mathbf{T}_W^{E_k}, & \forall \text{contact } \tau_i, \forall k \in [K_i] \quad \text{(contact-rich transform)} \\
s_t \in D_{\mathrm{success}}, & \exists t \in [T] \quad \text{(task success)}
\end{cases}
$$

**变量逐项解释**：

- $s_t \in \mathcal{S}$：MDP 的 state，包括 base 位姿、torso joint、arm joint、gripper、object pose
- $a_t \in \mathcal{A}$：action，paper 里取的是 target joint position（target joint position 是 absolute 而非 delta，这是模仿 WB-VIMA 和 π₀ 的接口约定）
- $f(s_t, a_t)$：system dynamics，这里通过仿真器 OmniGibson 推进
- $T$：trajectory 总长度
- $\mathcal{L}(\cdot)$：soft constraint cost 的总和（visibility during navigation、retraction 紧凑度等）
- $\mathcal{G}_{\mathrm{kin}}$：kinematic feasibility constraint，e.g. joint limit $\leq 0$（违反则 $\leq 0$）
- $\mathcal{G}_{\mathrm{coll}}$：collision-free constraint，$\geq 0$ 表示 signed distance 大于 0
- $\mathcal{G}_{\mathrm{vis}}$：visibility constraint，比如 "object 是否投影到 camera frustum 内"
- $o_{i(t)}$：在时刻 $t$ 当前 subtask 关心的 target object
- $\mathbf{T}_W^{E_k}$：EEF 在 world frame $W$ 下时刻 $k$ 的 6D pose（SE(3)）
- $\mathbf{T}_W^{o_i}$：当前 target object $o_i$ 在 world frame 下的 pose（随机化后的新 pose）
- $\mathbf{T}_W^{o_{i,src}}$：source demo 里 target object 的 pose
- $K_i$：subtask $i$ 的步数

**关键 intuition——contact-rich 那条等式**：

$$
\mathbf{T}_W^{E_k} = \mathbf{T}_W^{o_i} \cdot (\mathbf{T}_W^{o_{i,src}})^{-1} \cdot \mathbf{T}_W^{E_k}
$$

这条等式是说：**EEF 在 object 坐标系下的相对位姿保持不变**。把 source demo 的 EEF world pose $\mathbf{T}_W^{E_k}$ 先乘以 $(\mathbf{T}_W^{o_{i,src}})^{-1}$ 转到 object frame（这是"contact-relative frame"，比如 grasping 时 gripper 相对 cup 把手的位姿），然后用新 object 的 world pose $\mathbf{T}_W^{o_i}$ 再 transform 回 world frame。这是整个 X-Gen 范式的"坐标变换不变性"假设——只要 EEF 和 object 的相对几何关系对，运动就能复现。Free-space subtask 不需要这条约束，直接用 motion planner 在 joint space 或 task space 采样即可。

**这个 formulation 的统一性**：MimicGen 只用了 Succ 一条 hard constraint；SkillMimicGen 加了 Kin / C-Free；DexMimicGen 加了 Temporal（bimanual 协调）；PhysicsGen 加了 Dyn（动力学）和 Trac（trajectory tracking）作为 soft；MOMAGEN 在它们基础上补了 **Vis（hard + soft）** 和 **Ret（soft）**，并且第一次让 base pose 成为被采样的变量，而不是从 source 直接 replay。

---

## 3. MOMAGEN 的四个核心 constraint 设计

Paper Section 4.1 给出四个 constraint，按 hard / soft 划分。这是整篇文章最值得反复推敲的部分。

### 3.1 Reachability 作为 hard constraint
对每个 subtask，base pose 必须 sampled 得让所有 contact-rich segment 所需的 EEF pose 落在 arm workspace 内。具体做法是：对候选 base pose $\mathbf{T}^{\mathrm{base}}$，对 subtask 关键帧（pregrasp / contact frames）解 IK，IK 解不出来就 reject。

### 3.2 Object Visibility during Manipulation 作为 hard constraint
对每个 subtask 的 manipulation phase，必须保证 task-relevant object 在 head camera 的 FoV 内。这是 paper 的核心 claim：visuomotor policy 训练时如果关键帧看不到 object，policy 根本学不到 visual-grounding 的 action。Paper 通过 torso articulation 和 camera pose 的 joint 采样来满足这个约束。

### 3.3 Object Visibility during Navigation 作为 soft constraint
导航过程中"最好"能看到 object，但不是必须。通过在 motion planning 的 cost function 里加一项 "camera 朝向 target object 的角度差"，bias base/torso motion 让 camera 尽量 look at object。这是图 5 里 MOMAGEN 显著超过 ablation 的原因——soft constraint 让 ~80% 的 navigation 帧 object 都可见，而 w/o soft vis 只有 ~20%。

### 3.4 Retraction 作为 soft constraint
每个 manipulation 结束后，torso 和 arm 应该 retraction 到一个 compact 配置，让后续 base motion 更安全、更不容易撞东西。这条对应 Algorithm 1 line 14。

**intuition**：Hard / Soft 的区分体现了 paper 对 "什么 truly 限制 policy 学习" 的判断。Hard 是"做不到就 policy 训不出来"，soft 是"做到更好但不是必需"。Reachability 是 hard 因为 IK 解不出来 trajectory 根本不存在；visibility during manipulation 是 hard 因为 final approach 阶段如果看不到 object，policy 学不到 close-loop visual servoing；visibility during navigation 是 soft 因为它只是给 policy 多一些"提前看到 object"的 context，并不是必需的。

---

## 4. Algorithm 1 详解

Algorithm 1 是 MOMAGEN 的实际执行流程，建议你配合 Figure 2 一起看。我把它逐行展开：

```
Input: source demo, new initial state s_0
Output: generated demo

For each subtask (line 1):
  Get current T_base, T_cam, q_torso, q_arm (line 2)
  if held object not in hand: abort (line 3)  # grasp 失败检测
  Compute T_eef via new target object pose (line 4)  # 公式1 的 contact transform
  
  Check visibility of target object with T_cam (line 5)
  Solve IK for arm trajectory with current T_base, T_cam (line 6)
  
  while not visible or no IK exists (line 7):
    Sample new T_base (line 8)        # 关键：base pose 不是 replay
    Sample new T_cam (line 9)          # 关键：camera pose 也是采样
    Solve IK for arm + torso with sampled T_base, T_cam (line 10)
  
  Plan base + torso motion with soft visibility cost (line 11)
  Plan arm motion from prev T_eef to pregrasp T_eef (line 12)
  Task-space control to follow transformed T_eef (line 13)
  Attempt retraction (line 14)
```

**关键设计 choice**：

1. **Line 8 的 base pose sampling**：这是与所有 prior X-Gen 工作的根本区别。Base pose 不是 source replay，而是在 target object 附近采样。Paper Section 4.2 提到用 "heuristics" 来 sample，比如在 object 周围的某个 angular range 内 sample，然后做 conditional sampling——把 robot configuration 分成 torso subspace 和 arm subspace 分别 sample，类似 task-motion planning (Garrett et al., 2021) 的思想。这能大幅缩小 sample space。

2. **Line 6 / 10 的 IK 优先**：paper 提到一个 efficiency trick——先用快的 IK check 做 preemptive filtering，再调贵的 motion planning。cuRobo (Sundaralingam et al., 2023) 是 GPU 加速的 motion generator，能并行解多个 IK，这让 dense sampling 变得可行。

3. **Line 13 的 task-space control**：contact-rich phase 不是用 motion planning（因为 motion planner 处理 contact 不稳定），而是直接 task-space replay——把 source demo 的 EEF pose 序列（已经 transform 到新 object frame）作为 task-space reference，PD 控制器跟踪。

参考链接：
- cuRobo: https://arxiv.org/abs/2310.17274
- Integrated Task and Motion Planning (PDDLStream): https://arxiv.org/abs/2010.01083

---

## 5. 实验设计：Task + Domain Randomization + Baselines

### 5.1 四个 task（Figure 3）
| Task | 描述 | 关键挑战 |
|---|---|---|
| Pick Cup | 导航到桌子抓起 cup | 简单 baseline |
| Tidy Table | cup 从 countertop 移到 sink | long-range navigation + manipulation |
| Put Dishes Away | 两个 plate 用两个 arm 独立 stack 到 shelf | bimanual uncoordinated |
| Clean Frying Pan | 双臂用 brush 擦 pan | contact-rich bimanual coordinated |

四个 task 覆盖了 pick-and-place / long-range / bimanual-uncoordinated / contact-rich 这四个 axis，设计上是有代表的。

### 5.2 三级 domain randomization
- **D0**: object 位置 ±15cm、±15° on same furniture
- **D1**: object anywhere on same furniture, unrestricted orientation
- **D2**: D1 + distractors on furniture + obstacles on floor

D2 是 paper 的重要 selling point——prior X-Gen 在 D0 之外完全失效（Table 2 baseline 在 D1/D2 全部 N/A），MOMAGEN 在 D2 还能拿到 Pick Cup 0.47、Clean Frying Pan 0.16 的 success rate。这说明 base sampling + motion planning with obstacles 让方法有真正的 generalization 能力。

### 5.3 Baselines
- **SkillMimicGen** (Garrett et al., 2024): single-arm + motion planning + task-space control
- **DexMimicGen** (Jiang et al., 2025b): dexterous bimanual data gen

两个 baseline 都被 paper "extend" 了——加上 base trajectory replay from source demo，类似 MimicGen 处理 mobile 的方式。这是 fair comparison 的关键，因为原版 SkillMimicGen / DexMimicGen 根本不支持 mobile。

---

## 6. 关键实验数据解读

### 6.1 Data Generation Success Rate (Table 2)

D0 下：
| Method | Pick Cup | Tidy Table | Put Dishes | Clean Pan |
|---|---|---|---|---|
| MOMAGEN | 0.86 | 0.80 | 0.38 | 0.51 |
| SkillMimicGen | 1.00 | 0.69 | 0.38 | 0.40 |
| DexMimicGen | 1.00 | 0.72 | 0.38 | 0.35 |
| MOMAGEN w/o hard vis | 0.97 | 0.59 | 0.29 | 0.24 |

**有趣观察**：
- Pick Cup 上 baseline 反而更高（1.00 vs 0.86），因为简单 task 上 constraint 少反而让 sampling 容易，MOMAGEN 加 hard vis constraint 反而 reject 了一些采样
- Clean Frying Pan 上 MOMAGEN (0.51) 显著优于 baseline (0.40/0.35) 和 ablation (0.24 w/o hard vis)，说明 contact-rich bimanual task 上 hard visibility constraint 实际帮 motion planning 找到 better torso configuration
- D1/D2 baseline 全部失败，MOMAGEN 在 D1 拿到 0.20–0.66，D2 拿到 0.07–0.50

### 6.2 Object Visibility Ratio (Table 3)

D0 下 Tidy Table：
- MOMAGEN: 0.86
- SkillMimicGen: 0.40
- DexMimicGen: 0.39
- w/o soft vis: 0.63
- w/o hard vis: 0.63
- w/o any vis: 0.46

**intuition**：Hard visibility constraint 主要作用在 manipulation 阶段（关键帧），soft visibility 主要作用在 navigation 阶段（密集帧）。两者都去掉，visibility ratio 从 0.86 跌到 0.46。Figure 5 的分布图更直观——MOMAGEN 的 trajectory visibility 分布集中在 80-100% 区间，ablation 分布散在 0-80%。

### 6.3 Policy Performance (Figure 6)

WB-VIMA on Tidy Table D0:
- MOMAGEN data: 0.40 success
- SkillMimicGen data: ~0.05
- DexMimicGen data: ~0.05
- Ablations (w/o vis const): 0.05–0.15

WB-VIMA on Pick Cup D1:
- MOMAGEN: 0.25
- Baselines: 0 (因为 baseline 在 D1 上完全 gen 不出来 data)

**关键 takeaway**：Visibility ratio 从 0.40 提到 0.86，policy success rate 从 0.05 提到 0.40——这是 8 倍提升，visibility 直接决定 visuomotor policy 的成败。Paper 在 Figure 6d 的 ablation 里专门验证了这一点：Pick Cup D0 上 w/o vis const 只能到 0.45-0.65，full MOMAGEN 到 0.75；Tidy Table D0 上 ablation 上限 0.05，MOMAGEN 到 0.40。

### 6.4 Data Scaling (Figure 7)
π₀ fine-tune 在 500 / 1000 / 2000 demos 上，D1 难度下 success rate 有明显 scaling 趋势。这是 paper 暗示 MOMAGEN 可以继续 scale 的 evidence——传统 X-Gen 因为 gen 不出来 D1 数据，根本没法 scaling。

### 6.5 Sim-to-real (Section 5.4 + Appendix A.1)
Setup: Galaxea R1 robot, Pick Cup D0, 1000 synthetic demos + 40 real demos
- WB-VIMA: sim pretrained → 10% real success; baseline → 0%
- π₀: sim pretrained → 60% real success; baseline → 0%

**这个结果非常重要**：π₀ baseline 已经在 10k+ hours robot data 上预训练，但在 40 个 real demos 上 fine-tune 仍然 0%，加 MOMAGEN 的 1000 synthetic demos 后跳到 60%。这说明 sim synthetic data 的价值不是"补足数据量"，而是"提供 task-specific 的 visual-action prior"，让 fine-tune 能从正确的"基础动作分布"出发。

参考链接：
- π₀: https://arxiv.org/abs/2410.24164
- π₀.5: https://arxiv.org/abs/2504.16054
- WB-VIMA / Behavior Robot Suite: https://arxiv.org/abs/2503.05652
- Galaxea R1: https://www.galaxea.ai/

---

## 7. Cross-Embodiment 与 Articulated Object 的延伸

### 7.1 Cross-embodiment (Appendix B.2)
Paper 做了一个有意思的实验：用 Galaxea R1 上的 1 个 source demo，给 TIAGo 生成 Pick Cup demos。两个机器人虽然都是 dual 7-DoF arm + holonomic base，但 torso 设计和 arm workspace 差异很大。能 work 的原因是 MOMAGEN 在 task-space replay EEF 轨迹，而 task-space trajectory 是 embodiment-agnostic 的。但 paper 也承认 limitation——gripper size 差异会导致 confined space 的 collision。

这个 cross-embodiment 能力其实是 MOMAGEN formulation 的副产品：因为 contact transform 公式只关心 EEF 相对 object 的几何，不关心 joint configuration，所以换 robot 本质上是换一个 IK solver。

### 7.2 Articulated Object (Appendix B.1)
Get Bottle task：navigate to fridge → open door with one arm → retrieve bottle inside. 这里 fridge 先是 task-relevant object（for door-opening subtask），door 打开后 bottle 变成新的 task-relevant object。这展示了 MOMAGEN 的 subtask 切换机制——target object 在 subtask 之间动态变化。

这让我想到 BEHAVIOR-1K 的设计哲学：所有 task 都分解成 subtask + subgoal，MOMAGEN 在这个分解上 operate。

参考链接：
- BEHAVIOR-1K: https://arxiv.org/abs/2403.09227
- OmniGibson: https://behavior.stanford.edu/omnigibson

---

## 8. Training Details：WB-VIMA vs π₀

### 8.1 WB-VIMA (Appendix C.2)
- Input: egocentric colored point cloud (4096 points, fused from 3 RGB-D cameras) + proprioception (21-dim)
- Proprioception: $v^{\mathrm{base}} \in \mathbb{R}^3$, $q^{\mathrm{torso}} \in \mathbb{R}^4$, $q^{\mathrm{left}} \in \mathbb{R}^6$, $q^{\mathrm{grip-left}} \in \mathbb{R}^1$, $q^{\mathrm{right}} \in \mathbb{R}^6$, $q^{\mathrm{grip-right}} \in \mathbb{R}^1$，加起来 21
- Architecture: PointNet (4096→256) + proprioceptive MLP (21→256) → Transformer (4 layers, 8 heads, 512 embed) → Diffusion head (UNet, DDIM, 16 denoise steps)
- 2-step history
- Model size: 37.1M
- 训练 1M steps，2× RTX 3090，~40 hours

### 8.2 π₀ (Appendix C.3)
- Input: 3× RGB (224×224) + same 21-dim proprioception
- LoRA rank 32, fine-tune 50k steps, batch 64
- PaliGemma VLM backbone (3B, 18 layers, 18 heads) + 300M action expert
- Flow matching head (10 inference steps)
- 4× H200, ~7 hours
- Action chunk: 50 future steps
- Action/proprio zero-padded to 32-dim

**直觉对比**：WB-VIMA 是 from-scratch 训练的小 model，验证 data quality；π₀ 是大 pretrained model + LoRA fine-tune，验证 data 能不能提升 strong baseline。两个都正向收益，说明 MOMAGEN 的 data 不挑 model。

### 8.3 Data Cleaning (Appendix C.1)
一个值得注意的细节：仿真 teleop 会产生 "frozen segments"——operator 在 grasp 前犹豫，gripper 几乎静止。Paper 用一个简单 heuristic：如果 step $i$ 到 $i+5$ 的 joint position diff 在所有维度都 $<10^{-3}$，就丢弃这段。这对短 history 的 policy（WB-VIMA 2-step、π₀ 1-step）尤其重要，因为 frozen segment 会让 policy 学到"什么都不做"。

参考链接：
- PaliGemma: https://arxiv.org/abs/2407.07726
- VIMA: https://arxiv.org/abs/2210.03094

---

## 9. Limitations 与延伸思考

Paper 自己列了三个 limitation：
1. **依赖 full scene knowledge**：生成 demo 需要 ground-truth object pose，仿真里 trivial，real-world 难。Paper 提到可以用 SAM2 来 estimate object pose relative to robot，但这是 future work。
2. **只支持 alternating navigation + manipulation**：不支持 whole-body manipulation（比如一边走一边开门）。Paper 说 framework "easily extensible"，但实际实现上 navigation 和 manipulation 是分段的。
3. **GPU 资源密集**：每个 successful demo 0.1–1.3 GPU hours，主要瓶颈是 cuRobo motion planning。

### 我自己的延伸思考

**(a) Constraint 选择的主观性**：Paper 承认 soft constraint 的选择是"highly dependent on specific application"。比如"visibility during navigation 是 soft"这个判断，对 short-history policy 很对，但对 long-history policy（比如有 16-step context）可能 hard 也没问题。这暗示 MOMAGEN 的 constraint 应该和下游 policy 的 receptive field 共同设计。

**(b) 与 RL 的关系**：MOMAGEN 是 pure imitation，没用 RL。但 formulation 公式 1 看起来很像 constrained MDP / primal-dual RL。完全可以在 MOMAGEN generated data 上做 RL fine-tune（类似 TransIC, https://arxiv.org/abs/2405.10315），把 hard constraint violation 作为 RL 的 safety constraint。

**(c) 与 VLA 的关系**：π₀ 在 MOMAGEN data 上 fine-tune 跳到 60%，这暗示了一个有趣方向——**synthetic data 作为 VLA 的 embodiment-specific adapter**。π₀ pretrained 在 10k+ hours 多 robot data 上，但每个新 robot 还是要 fine-tune。MOMAGEN 这种"1 demo → 1000 synthetic"的 pipeline 可能是 VLA 时代的"低成本 embodiment adaptation"标准范式。

**(d) 与 GenAug / RoboGen 的对比**：GenAug (https://arxiv.org/abs/2302.06671) 用 generative model 做 image-level augmentation；RoboGen (https://arxiv.org/abs/2311.01455) 用 LLM 自动生成 task + simulation。MOMAGEN 处于中间——task 是 human-specified，simulation 是 human-set-up，但 trajectory 是自动生成的。三者其实可以结合：RoboGen 生成 task setup，MOMAGEN 生成 trajectory，GenAug 进一步 augment 视觉。

**(e) "1 source demo" 是不是 over-fit claim**：Paper 强调 $N_{src}=1$，但仔细看 Algorithm 1，annotation 阶段需要人标注 subtask、target object、pregrasp timestep、retraction type——这本身是非常重的 human-in-the-loop 工作。所以实际 human cost 不是"1 个 demo"，是"1 个 demo + 详细的 subtask annotation"。这点 paper 没明说，但实操上重要。

**(f) Hard visibility 的数学形式没明确**：Paper 没给 $\mathcal{G}_{\mathrm{vis}}$ 的具体数学表达。可能是 "object bounding box 投影到 image plane 的面积 ≥ threshold" 或 "object center 在 frustum 内 + 无 occlusion"。这影响 reproducibility，需要看 code。

**(g) 与 SafeMimic 的关系**：SafeMimic (Bahety et al., 2025, https://arxiv.org/abs/2506.15847) 是同作者群的另一个工作，做 safe human-to-robot imitation for mobile manipulation。MOMAGEN 是 offline data generation，SafeMimic 是 online safe imitation，两者互补。

---

## 10. 整体评价与 Position

**贡献**：
1. 第一个把 X-Gen 推到 bimanual mobile manipulation，并且是 single source demo
2. 把整个 X-Gen 家族统一到 constrained optimization framework 下
3. 实验上 demo 了 sim-to-real 的可行性（40 real demo + 1000 synthetic → 60% success）

**我会关注的下一步**：
- 把 LLM 接入 annotation，自动从 task description 推 subtask + target object
- 把 hard constraint 从 simulation-grounded 推到 perception-grounded（用 SAM2 / FoundationPose）
- 把 navigation + manipulation 的 alternating 假设打破，支持 whole-body simultaneous motion
- 在更多 embodiment 上做 cross-embodiment，尤其是人形机器人（HumanPlus, OmniH2O, Mobile ALOHA 这类）

参考链接：
- Mobile ALOHA: https://arxiv.org/abs/2401.02117
- HumanPlus: https://arxiv.org/abs/2406.10454
- OmniH2O: https://openreview.net/forum?id=oL1WEZQal8
- SafeMimic: https://arxiv.org/abs/2506.15847
- GenAug: https://arxiv.org/abs/2302.06671
- RoboGen: https://arxiv.org/abs/2311.01455
- TransIC: https://arxiv.org/abs/2405.10315

---

## 一句话总结

MOMAGEN 把"用 1 个 human demo 生成 1000 条 bimanual mobile manipulation trajectory"这件事，本质化为一个 **base + torso + arm 联合采样 + reachability/visibility 约束的 constrained optimization 问题**，并且证明 hard visibility constraint 对 visuomotor policy 训练是 game-changer——这点对 sim-to-real 价值连城。整个 formulation 的 elegance 在于：它把 X-Gen 家族从"trick 集合"升级为"constraint 选择的 design space"，后续工作可以在这个 framework 里探索新的 constraint 组合。
