---
source_pdf: Planning-Guided Diffusion Policy Learning for Generalizable Contact-Rich
  Bimanual Manipulation.pdf
paper_sha256: 98648054505b093446d6aa9876a1f480e273981f581d1f8a0491a8f5c3431e1f
processed_at: '2026-08-06T04:47:25-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 GLIDE

好，我抛开那些学术腔，用最直白的方式跟你聊聊这篇 paper 到底在干嘛。

---

## 这篇 paper 在解决什么问题

想象你有一个双臂机器人，桌上放个大盒子，你跟它说"把这个盒子转 90 度"。听起来简单，但你想想机器人要干嘛：

首先它得用两条胳膊去夹住这个盒子。但盒子太大，手指头根本捏不住，得用胳膊的几个 link 去卡住它。然后开始转，转着转着发现 joint 快到极限了，得松开、换个姿势重新夹、再继续转。可能要折腾好几轮才能转到 90 度。

这过程中全是 contact——胳膊碰盒子、盒子碰桌子、不同 link 同时碰。contact 这东西在物理上是非光滑的，要么碰要么不碰，没有中间态。你要规划这么一条轨迹，本质上是在一个 huge 的离散-连续混合空间里搜索，计算量爆炸。

这就是所谓的 **contact-rich bimanual manipulation**，一直是 robotics 里的硬骨头。

---

## 传统方法为啥不行

有两条路，两条都有硬伤。

**第一条路：model-based planning**。你把机器人物理写成一个优化问题，让 solver 去找最优轨迹。问题是 contact mode 太多了——两个胳膊每条有 7 个 joint，加上 link 表面任何一点都可能 contact，组合数是指数级的。求解器跑半天算不出来一条，而且它需要知道盒子的精确 shape 和 pose，real world 你哪有这个信息。

**第二条路：imitation learning**。你让人去 teleop 收集一堆 demo，然后学一个 policy。问题是双臂 contact-rich 任务的 teleop 难得要死，你用 VR 手柄或者 leader-follower 设备去控制两个胳膊做这种精细的 contact 操作，一条 demo 可能要录几十分钟还经常失败。你要 12000 条这样的 demo？人力成本根本不现实。

所以你卡在这：planner 能跑但太慢且需要特权信息，learning 需要数据但数据太贵。

---

## GLIDE 的核心 idea：让 planner 当老师，policy 当学生

这篇 paper 的思路其实很朴素：**既然 real world 收数据太难，那我在仿真里让 planner 去跑，跑出来的成功轨迹当 demo，然后用这些 demo 训练一个 neural network policy**。

部署的时候不用 planner，只用 policy。policy 接收的是 point cloud（不是 privileged state），输出的是 joint command，closed-loop 地跑。

这样你就绕开了两个问题：planner 的实时性问题和 real world 数据采集问题。planner 慢没关系，反正它在仿真里离线跑；policy 部署时不需要知道 object shape，因为它从 point cloud 里推断。

这个 idea 本身不新，[Guided Policy Search](https://proceedings.mlr.press/v28/levine13.html) 十年前就在做类似的事。但 GLIDE 把它在 contact-rich bimanual 这个 hard setting 下真正跑通了，而且跑出了不错的 generalization。关键在于一堆细节做对了。

---

## Planner 怎么跑快了

他们用的 planner build on [Pang et al. 2023](https://arxiv.org/abs/2206.08502) 和 [Suh et al. 2025](https://arxiv.org/abs/2203.08532) 的工作。这两篇的核心 trick 是：**别去求解 long-horizon trajectory optimization，把 contact dynamics 在局部做 linear approximation，然后每步只解一个小 QP**。

具体说就是每一步解这么个东西：

$$\min_{q_+^u, a} \quad (q_+^u - q_{\text{goal}}^u)^T Q (q_+^u - q_{\text{goal}}^u) + (a - q^a)^T R (a - q^a)$$

翻译成人话：我要找一个 action $a$（两条胳膊的 joint command），让执行完之后 object 的 pose $q_+^u$ 尽量靠近 goal，同时 action 不要偏离当前 joint position $q^a$ 太多（smoothness）。$Q$ 和 $R$ 就是两个权重矩阵，调你多在乎 goal distance vs action 平滑。

$q_+^u$ 怎么来的？用一个 local linear model $f_{\text{local}}$ 近似 contact dynamics。这个近似只在局部成立，所以每步只能走一小步，但每步的 QP 秒解。

再加上一个 greedy strategy：每次 contact phase 都尽量把 object 往 goal 方向推。作者说这个 greedy 改动基本不影响 success rate 但大幅提速。为什么？因为这里 planner 的目的不是找 global optimal，而是 **大量生成成功轨迹给 BC 用**。Greedy 让轨迹 distribution 更一致，BC 反而好学。

---

## 但 raw planner 数据不能直接用

这是 paper 里一个很容易被忽略但特别重要的点。

Planner 跑出来的轨迹有很多问题：
- $f_{\text{local}}$ 是近似的，在真实 dynamics 里 rollout 可能根本到不了 goal
- RRT 有随机性，有时候规划出来的轨迹绕一大圈
- 有些轨迹卡在 joint limit 走不动

如果你直接拿这些 garbage 去做 BC，policy 会学一坨乱七八糟的东西。

所以 GLIDE 做了一步 **filtered BC**：把每条 planner 轨迹在 Drake simulator 里高保真地 rollout 一遍，**只保留真的能到 goal 而且时间合理的轨迹**。然后把保留下来的轨迹按 object 做均匀 rebalance，最后渲染成没有颜色的 point cloud。

这一步把数据质量拉上来了。Table V 显示 500 demo 时 Fixed 45° 只有 33% 成功率，12000 demo 时 80%。而且还没饱和，继续加数据应该还能涨。

---

## 为什么用 Diffusion Policy

Contact-rich 任务有个麻烦：同一个状态可能有多种 valid action。比如你要把盒子往左转，你可以先动左臂推、也可以先动右臂拉，两种 action sequence 都能 work。

如果你用普通的 MSE regression 去 predict action，它会把这个 multimodal distribution 平均掉，得到一个两个 mode 之间的无效 action。这就像你让一个网络预测"猫或狗的图片"，它输出一只猫狗混合体。

[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 通过 score matching 学的是 distribution 的 mode 而不是 mean。它从 noise 开始，一步步 denoise 出一个 action sequence，自然能 express multimodal distribution。

GLIDE 在这个基础上做了三个关键改动，这三个改动才是 sim-to-real 能 work 的真正原因。

---

## 三个关键 design choices

### Trick 1: Flying Point Augmentation

Real world 的 RGBD sensor 在深度边界处会产生 "flying pixel"——就是一些深度值完全错误的噪点。仿真里没有这个 artifact，policy 没见过，一到 real 就崩。

GLIDE 的解法简单粗暴：训练时以 0.5% 的概率随机给某些点加大的 Gaussian noise，模拟 flying pixel。Table III 显示这个 trick 在仿真里几乎不影响（0.780 vs 0.740），但 real world 里是从 0.320 到 0.800 的生死差别。

这种 "仿真里看不出差别、real 里生死攸关" 的 trick 是 robotics sim-to-real 最有意思的地方。你光看仿真 ablation 会觉得它没用，真到 real 上才发现它是命门。

### Trick 2: Residual Action Prediction

原版 Diffusion Policy 直接预测 absolute joint angle。GLIDE 改成预测 **residual**：$a = q_{\text{future}} - q_{\text{current}}$。

为什么这个重要？因为 absolute joint angle 的 range 太大了——不同初始配置下，同样的 "转 10 度" 对应的 absolute target 完全不同，但 residual 都是 "10 度"。Residual 让 action 的 distribution 在不同 trajectory、不同 object 上更 compact、更 consistent，policy 好学。

Table III 里去掉 residual，real world 从 0.800 掉到 0.520。又一次 sim 几乎无差别、real 致命。

### Trick 3: Shape-Agnostic Task Representation

这是我觉得最巧的一点。

原版 diffusion policy 每个 task 训一组参数。GLIDE 要 single policy 处理任意 target pose，而且不知道 object shape。

怎么表示 task？用 keypoint tracking：

1. 第一帧用 [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) 做 open-vocabulary segmentation，把 target object 分出来
2. 在 segmentation 里用 Farthest Point Sampling 选几个 keypoint（空间上均匀分布的 anchor 点）
3. 用 [TAPIR](https://deepmind-tapir.github.io/) 或 [CoTracker](https://co-tracker.github.io/) 实时跟踪这些 keypoint 的 3D 位置
4. 每步用 keypoint 的 current position vs initial position 算一个 SE(2) delta，当 task spec $c_t$ 喂给 policy

这个的 beauty 在于：你不需要知道 object 是盒子还是碗还是袋子。你只要知道 "这几个点现在在哪、目标让它们到哪" 就够了。policy 学的是 "怎么让这些点到达目标" 的通用 skill，自然 generalizes 到 OOD object。

这就是为什么 Table II 里 OOD container 还能有 65% 成功率——policy 根本不在乎 object shape，它只在乎 keypoint 的几何变换。

---

## Action Horizon $T_a$ 的 trade-off

Diffusion Policy 每次推理不是只预测一步 action，而是预测一整个 chunk of $T_a$ 步。训练时用 $T_a=64$，推理时用 $T_a=20$。

Table IV 的 ablation 很有意思：

- $T_a=8$：太短，轨迹不连续，性能差（0.44）
- $T_a=20$：sweet spot（0.74）
- $T_a=64$：太长，chunk 内没有 visual feedback，contact 中途发生 slip 来不及反应（0.20 on Random）

这个 trade-off 的 intuition 是：contact-rich 任务需要足够的 action horizon 保证平滑（太小会 jitter），但又需要足够的 closed-loop frequency应对 contact 中的 unexpected perturbation。$T_a=20$ 是这两个力的平衡点。

---

## 结果里几个值得注意的点

**Policy 比 Planner 成功率高**。Planner 在 Sim 上 Fixed 45° 只有 33.7%，Policy 是 74%。这反直觉但合理：policy 经过了 filtered BC，只学了成功轨迹的 pattern；policy 有 visual feedback 做 closed-loop，能补偿 planner 的 open-loop 误差；planner 本身是 stochastic RRT，policy 是 deterministic 的。

**Real 比 Sim 略高**（80% vs 74% on Fixed）。这个我有点存疑，因为 real world 只跑了 25 个 trial，confidence interval 很宽。但至少说明 sim-to-real gap 被跨过去了，不是在 real 上崩盘。

**OOD 表现不错但 Hard case 崩**。Random Hard 只有 20%，因为需要 multi-round contact phase，长 horizon 下 error 累积。Failure analysis 显示 52% 的 failure 是 robot 卡在 poor joint configuration 走不动。

---

## 这篇 paper 的 limitation 和我的看法

**Planner 还是太慢**。2 days on 96 CPU 才生成 12k 条。如果要 scale 到百万级数据，planner 得再快几个数量级。可能的方向是用 learned value function 加速 RRT，或者用 GPU 并行的 differentiable simulator。

**只训练了 box**。OOD 到 container 能 work 是因为 container 还是有点 box-like。真的到任意 shape（剪刀、衣架、螺丝刀）可能崩。keypoint representation 对拓扑变化大的 object 可能不够。

**只测了 SE(2)**。SE(3) reorientation（把盒子立起来、翻转）contact mode 空间会再爆炸一个量级，这篇没碰。

**没学 recovery**。52% 的 failure 是 stuck，说明 policy 只会 happy path，不会从 bad state recover。一个自然的 extension 是在数据生成时加入 near-failure scenario 和 recovery trajectory，让 policy 见过 failure 长什么样。

**整体 takeaway**：这篇 paper 的价值不在 GLIDE 这个方法本身多 novel，而在于它把 "planner 生成数据 → filter → diffusion policy + 几个 sim-to-real trick" 这条 pipeline 在一个真正 hard 的任务上跑通了，并且给出了哪些 trick 重要的清晰 ablation。这套 recipe 在其他 contact-rich manipulation 任务上应该能复用。

更深层的启示是：**manipulation 领域下一个阶段的竞争是 data scaling 的竞争，而 data scaling 的 bottleneck 是 demonstration 来源**。Real world teleop 太慢，real world RL 太危险，simulation + planning 是目前最 viable 的大规模 data engine。GLIDE 验证了这条路在 contact-rich bimanual 这种 hard setting 下也走得通。

参考资料：
- [GLIDE 项目主页](https://glide-manip.github.io/)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [DP3 3D Diffusion Policy](https://3d-diffusion-policy.github.io/)
- [Pang et al. Global Planning for Contact-Rich Manipulation](https://arxiv.org/abs/2206.08502)
- [Drake Simulator](https://drake.mit.edu/)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [TAPIR](https://deepmind-tapir.github.io/)
- [CoTracker](https://co-tracker.github.io/)
- [Guided Policy Search](https://proceedings.mlr.press/v28/levine13.html)
- [ALOHA](https://tonyzhaozh.github.io/aloha/)
- [Open X-Embodiment](https://robotics-transformer-x.github.io/)

---

# GLIDE: Planning-Guided Diffusion Policy for Contact-Rich Bimanual Manipulation 深入解析

Andrej，这篇 paper 我觉得是一个挺漂亮的 "scaling via simulation + filtered BC + careful inductive biases" 的范例。让我从 intuition 出发，把整条 pipeline 拆开讲，包括公式里每个上下标在说什么、为什么这么做、ablation 想告诉我们什么。

---

## 1. 任务本质：为什么这个问题难

Contact-rich bimanual manipulation 跟我们平时玩的 single-arm grasping/pushing 完全不是一个量级。核心难点有三层：

1. **Hybrid dynamics**：robot link 和 object 之间的 contact 是 non-smooth 的，contact mode 集合随 link 数和 horizon 指数爆炸。MIP formulation 在 full-arm bimanual 上不可解。
2. **Long horizon multi-phase**：要把一个 bulky box 旋转 150°，单个 grasp 角度受 joint limit 约束，必须做 "approach → manipulate → release → re-approach → manipulate" 这种 multi-round 序列。
3. **Sim-to-real & generalization**：planner 需要 privileged state（object pose, geometry），real world 给你的只有 depth point cloud。

GLIDE 的核心 insight 是：**用 model-based planner 生成海量仿真 demo 当作 BC 的 supervision，再用 diffusion policy 的 multimodal 表达能力 + 几个关键 inductive bias 跨越 sim-to-real 和 OOD 的 gap**。

任务形式化：
- 状态 $s_t \in S$，但 policy 看不到 $s_t$，只看 $o_t \in O$（point cloud + proprioceptive joint states）
- 任务 spec $c \in C$，定义成 current object pose → target pose 的 SE(2) transformation
- 要学一个 single policy $\pi_\theta(a | o, c)$，覆盖任意 target pose 和未见过的 object

关键决策：**不假设 object shape 已知**，所以 $c_t$ 每一步都要从 visual observation 重新估计，这就需要 keypoint tracking 这条链路。

---

## 2. 数据合成：把 planner 当成 oracle，但要做 filtered BC

### 2.1 Planner 的核心：linearized local contact dynamics

GLIDE 直接 build on Pang et al. (CoRL 2023, [https://arxiv.org/abs/2206.08502](https://arxiv.org/abs/2206.08502)) 和 Suh et al. 2025 的 contact trust region 工作 ([https://groups.csail.mit.edu/robotics-large/publications/papers/2025/suh2025contact.pdf](https://groups.csail.mit.edu/robotics-large/publications/papers/2025/suh2025contact.pdf))。这两篇工作的核心 idea 是：把 local contact dynamics 做 **smoothed linear approximation**，把一个 long-horizon trajectory optimization 问题降解为单步 QP，避免 MIP 的 combinatorial explosion。

Planner 的单步优化目标：

$$
\min_{q_+^u,\, a} \;\; \underbrace{(q_+^u - q_{\mathrm{goal}}^u)^T \, \mathbf{Q} \, (q_+^u - q_{\mathrm{goal}}^u)}_{\text{state cost}} \;+\; \underbrace{(a - q^a)^T \, \mathbf{R} \, (a - q^a)}_{\text{action regularizer}}
$$

subject to $q_+^u = f_{\mathrm{local}}(q^u, q^a, a)$

变量含义：
- $q^a$：robot 的 joint configuration（a = "agent/arm"），当前时刻
- $q^u$：object 的 configuration（u 通常表示 "unknown object"，是 planner 内部维护的 privileged state）
- $q_+^u$：执行 action $a$ 后 object 的下一步 configuration，"+" 表示 next step
- $q_{\mathrm{goal}}^u$：object 的目标 pose
- $a$：commanded joint angles（两臂同时）
- $\mathbf{Q}$：state cost matrix，penalize object 离 goal 的距离（SE(2) 上一般是对 position + yaw angle 加权）
- $\mathbf{R}$：action cost matrix，penalize joint command 偏离当前 $q^a$，相当于一个 trust region / smoothness regularizer
- $f_{\mathrm{local}}$：linear approximation of local contact dynamics（这是 [4] 的核心贡献，把 contact 的 complementary constraint 用 smoothed linearization 来近似）

### 2.2 Greedy planning vs sampling-based

原版 Pang et al. 是 sampling-based global planner，GLIDE 改成 **greedy**：每次 contact phase 都尽量把 object 往 goal 推。作者说这个改动基本不影响 success rate，但大幅提速。这里有个 intuition：作为 BC 的 oracle，你不需要 globally optimal trajectory，你需要的是 **consistent + diverse + 大规模** 的 suboptimal-but-successful trajectory。Greedy 让数据 distribution 更集中，反而让 BC 学得更稳。

### 2.3 Algorithm 1 解读

```
while q^u ≠ q_goal^u:                          # 外层 multi-phase loop
    q_grasp^a ← SAMPLECONTACT(q^u)             # 1. IK 生成 grasp pose
    while q^a ≠ q_grasp^a:                     # 2. collision-free approach
        q^a, a ← PLANCOLLISIONFREE(...)        #    bidirectional RRT + shortcut
    while q^a not at joint limit and q^u ≠ q_goal^u:  # 3. contact phase
        q^a, q^u, a ← PLANCONTACT(...)         #    单步 QP, greedy move
```

三个子模块：
- **Contact sampler**：用 IK 生成 distal links 能 pinch 住 object 的 grasp config $q_{\mathrm{grasp}}^a$
- **Bidirectional RRT with shortcutting**：经典 collision-free motion planning，参考 LaValle 的 textbook ([https://planning.cs.uiuc.edu/](https://planning.cs.uiuc.edu/))
- **Contact planner**：上面那个 QP，每步贪心

### 2.4 Filtered Behavior Cloning：这个细节很重要

直接拿 planner 输出做 BC 会 fail，因为：
1. $f_{\mathrm{local}}$ 是近似，rollout 到 high-fidelity simulator 里可能不收敛
2. RRT 的 stochasticity 导致 trajectory 长度 variance 巨大
3. 部分 trajectory 会卡死

所以 GLIDE 在 Drake simulator ([https://drake.mit.edu/](https://drake.mit.edu/)) 里把每条轨迹 rollout，**过滤掉不到 goal 或太慢的轨迹**，再按 object 维度做 uniform rebalance。最后渲染成 **uncolored point cloud**（去掉 color channel 让 policy 不过拟合 texture）。

这个 filter 的效果在 Table V 里隐含体现：500 demo 时 Fixed 45° 只有 0.33，12000 demo 时 0.80。更关键的是 **没有 plateau**，说明还可以继续 scaling。

数据成本：**2 days on 96-CPU machine** 生成 12,000 条 trajectory，这比真人 teleop 便宜几个数量级。

---

## 3. Diffusion Policy 的几个关键 design choices

这部分是 paper 的精华，也是 sim-to-real 能 work 的原因。基于 Diffusion Policy ([https://diffusion-policy.cs.columbia.edu/](https://diffusion-policy.cs.columbia.edu/)) 和 DP3 ([https://3d-diffusion-policy.github.io/](https://3d-diffusion-policy.github.io/))。

### 3.1 Point cloud preprocessing

- **Workspace clipping**：只保留 robot workspace 内的点，去掉 background。这一步对 sim-to-real 关键，因为 real world 背景复杂、训练 sim 里干净。
- **Flying Point Augmentation**：以小概率（0.5%）随机给某些点加 large Gaussian noise。这个名字来自 RGBD sensor 的 "flying pixels" 现象——depth 边界处会有无效深度点。这个 augmentation 让 policy 对 real sensor noise robust。

Table III 显示这个 trick 在 sim 里几乎无效（0.780 vs 0.740），但 real world 里是生死线（0.520 vs 0.800）。这是典型的 "sim 不在乎但 real 必死" 的 design choice。

### 3.2 Task representation：keypoint tracking without known object shape

这是 paper 里我觉得最巧的一段。原版 diffusion policy 假设任务 fixed，每个 task 一组参数。GLIDE 要 single policy 处理任意 target pose，并且 **不知道 object shape**。

解法：
1. 在 initial frame $o_0$ 上用 **Grounding DINO** ([https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)) + EfficientViT ([https://github.com/mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)) 做 open-vocabulary segmentation，分割出 target object
2. 在 segmentation 内用 **Farthest Point Sampling** 选 keypoints（FPS 是点云处理经典算法，保证 spatial 覆盖均匀）
3. 用 **TAPIR** ([https://deepmind-tapir.github.io/](https://deepmind-tapir.github.io/)) 或 CoTracker ([https://co-tracker.github.io/](https://co-tracker.github.io/)) 在 3D 空间里实时跟踪这些 keypoints
4. 每步 $t$ 由 keypoints 的当前 3D 位置 vs 初始位置计算 $c_t$（current → target 的 SE(2) 变换）

Intuition：你不需要知道 object 是盒子还是碗，你只要知道 "这几个 anchor 点现在在哪、目标在哪" 就够 policy 推 action 了。这种 shape-agnostic task representation 是 OOD generalization 能 work 的根本。

### 3.3 Residual action prediction

原 diffusion policy 直接预测 absolute end-effector pose 或 absolute joint angle。GLIDE 改成预测 **residual**：

$$
a_{t+1:t+T_a} = \{q_i - q_t\}_{i=t+1}^{t+T_a}
$$

其中 $q_t$ 是当前 joint positions，$q_i$ 是未来第 $i$ 步的 joint positions，$T_a$ 是 prediction horizon。

为什么这招 work：
- Absolute joint angle 在不同初始 config 下 scale 和 shift 差异巨大，policy 很难学
- Residual 在不同 trajectory、不同 object 上 distribution 更 compact、scale-consistent
- 类似于 control theory 里的 velocity form 或者 incremental MPC

Table III 里去掉 residual action，real world 从 0.800 掉到 0.520，sim 几乎不变。又一次 "sim 不暴露但 real 致命"。

### 3.4 Action horizon $T_a$

训练用 $T_a = 64$，推理用 $T_a = 20$。Table IV：

| $T_a$ | Fixed 45° | Random Rotation |
|-------|----------|----------------|
| 8     | 0.440    | 0.270          |
| 20    | **0.740**| **0.400**      |
| 40    | 0.760    | 0.340          |
| 64    | 0.770    | 0.200          |

Intuition：$T_a$ 越大，预测的 action chunk 越长，单次 inference 覆盖的 horizon 越多，**但** 也意味着 policy 在 chunk 内不接收新 observation，失去 closed-loop feedback。对于 contact-rich 任务，中间发生 slip 或 unexpected contact，$T_a=64$ 就跟不上。$T_a=20$ 是 sweet spot。

这跟 Diffusion Policy 原文里 $T_a = 8$ 的结论不同——contact-rich 任务对 closed-loop reactivity 要求更高，但又需要足够 horizon 保证平滑。

---

## 4. 实验结果解读

### 4.1 In-Distribution（Table I）

| Task | Planner(Sim) | Policy(Sim) | Policy(Real) |
|------|--------------|-------------|--------------|
| Fixed 45° | 0.337 | 0.740 | **0.800** |
| Random Easy | 0.227 | 0.610 | 0.600 |
| Random Medium | 0.141 | 0.410 | 0.360 |
| Random Hard | 0.099 | 0.180 | 0.200 |
| Random Overall | 0.156 | 0.400 | 0.387 |

**注意三个反直觉的点**：

1. **Policy 比 Planner 成功率高**。Planner 在 Sim 上 Fixed 只有 0.337，Policy 在 Sim 上 0.740，real 0.800。这是因为：(a) Policy 经过了 filtered BC，只学了 successful trajectories 的 pattern；(b) Policy 用 visual feedback 做 closed-loop，能补偿 planner 的 open-loop 误差；(c) Planner 是 stochastic RRT，policy 是 deterministic neural net。

2. **Real 比 Sim 略高**（Fixed 0.800 vs 0.740）。这通常意味着 sim-to-real gap 没有想象的那么大，而且 real world 评估的 trial 数少（25 次）variance 大。但也说明 design choices 真的把 sim-to-real 跨过去了。

3. **Hard task 退化严重**。Random Hard 只有 0.20，因为需要 multi-round contact phase，long horizon 下 error 累积，joint limit 卡死概率高。

### 4.2 OOD（Table II + Fig 6）

OOD container：
| Task | Empty | Overfilled | Overall |
|------|-------|------------|--------|
| Fixed 45° | 0.688 | 0.625 | 0.657 |
| Random | 0.250 | 0.313 | 0.282 |

Inflatable toys：Fixed 52%, Random 28%

这里有个很 surprising 的点：**overfilled container 比 empty container 在 Random task 上反而略高**（0.313 vs 0.250）。我猜可能是因为 overfilled 增加了 weight，contact 更稳定，slippage 减少。这暗示 policy 学到的是 contact mechanics 而不是 surface texture。

但 Fixed task 上 overfilled 略低，可能是 initial approach phase 时 irregular 顶部边界让 keypoint tracking 不稳。

### 4.3 Failure Mode（Sec IV-F）

Random Hard 的 failure 分布：
- **52%：joint config stuck**——robot 卡在一个 poor configuration，无法继续 rotate object。这是 planner 本身的 limitation，因为 greedy + joint limit。Fix idea：在数据生成时加入 "near-failure recovery" trajectories。
- **20%：object slippage**——contact force 不够或角度不对
- **16%：torque limit exceeded**——squeeze too hard

这三个 failure mode 全部指向 **数据 diversity 不足**：我们只学 successful trajectories，policy 不知道 failure 长什么样、不知道怎么 recover。这正是 [RoboTurk](https://roboturk.stanford.edu/) 和 recent recovery learning 工作想解决的。

---

## 5. 我的几点 intuition / 联想

1. **Filtered BC > raw BC**。Planner 是 noisy oracle，必须 filter。这个 lesson 在 [MPC-Net](https://arxiv.org/abs/1811.06406)、[Motion Policy Networks](https://arxiv.org/abs/2306.17833) 里都有体现。Scaling data 的同时必须 scaling data quality。

2. **Greedy planner + learned policy > global optimal planner**。Policy 能 closed-loop 补偿 planner 的 suboptimality，而 planner 提供了 large-scale supervision。这其实就是 [Guided Policy Search](https://proceedings.mlr.press/v28/levine13.html) 的精神：用 model-based method 当 "teacher"，但最终部署的是 reactive student。

3. **Residual action 是 bimanual / high-DoF 任务的通用 trick**。类似 [Diffusion Policy 的 proprioception conditioning](https://diffusion-policy.cs.columbia.edu/)、[ACT 的 transformer action chunking](https://tonyzhaozh.github.io/aloha/)。本质上 high-DoF action space 太大，absolute prediction 难收敛，relative prediction 借用了 current state 作为 anchor。

4. **Flying Point Augmentation 这种 "tiny trick" 在 real world 是分水岭**。这让我想起 [Domain Randomization](https://arxiv.org/abs/1703.06907) 和 [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) 都不是大架构创新，但实战中 work。Real world deployment 的瓶颈往往是 sensor artifact 而不是 high-level reasoning。

5. **OOD generalization 来自 shape-agnostic task representation**。Keypoint tracking + SE(2) delta 把 "把盒子转 45°" 和 "把碗转 45°" 映射成同一个 task spec。这跟 [Keyframe-based manipulation](https://arxiv.org/abs/2209.12959) 和 [Transporter Networks](https://transporternets.github.io/) 的思路一致。

6. **Diffusion policy 的 multimodal 表达能力对 contact-rich 任务关键**。Contact mode 本质是 multimodal：你可以用 left link push from left，也可以用 right link push from right，两种 action sequence 都能 work。MSE regression 会 average 这两种 mode 得到无效 action；diffusion 通过 score matching 学到 distribution 的 mode 而不是 mean。参考 [Decision Diffuser](https://arxiv.org/abs/2211.15657)。

7. **Scale 还没饱和**。Table V 没看到 plateau，意味着把 demo 数从 12k 推到 100k 应该还能涨。结合 [Open X-Embodiment](https://robotics-transformer-x.github.io/) 的大规模 cross-embodiment 数据思路，未来这条路 scaling 空间巨大。

8. **Failure mode 指向 recovery learning**。52% 的 failure 是 stuck，说明 policy 缺乏 "探索性 escape"。一个直接 extension 是用 [RoboGen](https://robogen-website.github.io/) 风格自动生成 near-failure 场景，或者加 [value function 做 recovery](https://arxiv.org/abs/2310.12263)。

---

## 6. 一些可以质疑的地方

- **Planner 太慢（2 days / 96 CPU for 12k traj）**。如果要 scale 到 1M traj，需要 planner 进一步加速。可能的 path 是用 [JAX-based differentiable simulator](https://github.com/erikfrey/brax) 或者直接用 learned value function 当 heuristic 加速 RRT。
- **只用了 box primitive 训练**。OOD 到 container work 是因为 container 还是有点 box-like。真的到 arbitrary shape（比如螺丝刀、衣架）可能崩。
- **只用 SE(2) target**。SE(3) 的 generalization（比如翻转、立起）没测，那个 contact mode 空间会爆炸。
- **Real world eval 样本少**（25 trials in-dist, 16+16 OOD）。0.800 vs 0.740 这种差异在 25 个样本下 confidence interval 很宽。

---

## 7. 总结：这条路线的本质

GLIDE 不是单纯的 "diffusion policy 论文"，也不是单纯的 "planning 论文"，它的核心 thesis 是：

> **在 contact-rich 这种 dynamics 复杂、数据昂贵的任务上，最优策略是用 model-based planner 在仿真里大规模生成 data，再用 expressive learned policy（diffusion）+ 几个 task-specific inductive bias 去 learn 一个 closed-loop、shape-agnostic、sim-to-real 的 reactive controller。**

这跟 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)、[DP3](https://3d-diffusion-policy.github.io/)、[ALOHA](https://tonyzhaozh.github.io/aloha/)、[UMI](https://umi-spinal.github.io/) 这些工作的共同 pattern 是：**manipulation 的下一个 decade 是 data scaling 战争，而 data scaling 的关键 bottleneck 是 demonstration 来源**。Teleop 太慢，real-world RL 太危险，**simulation + planning 是目前最 viable 的大规模 data engine**。

参考资料：
- GLIDE 项目主页: [https://glide-manip.github.io/](https://glide-manip.github.io/)
- Diffusion Policy: [https://arxiv.org/abs/2303.04137](https://arxiv.org/abs/2303.04137)
- DP3 (3D Diffusion Policy): [https://arxiv.org/abs/2403.03954](https://arxiv.org/abs/2403.03954)
- Pang et al. Global Planning for Contact-Rich Manipulation: [https://arxiv.org/abs/2206.08502](https://arxiv.org/abs/2206.08502)
- Drake (simulator): [https://drake.mit.edu/](https://drake.mit.edu/)
- Grounding DINO: [https://arxiv.org/abs/2303.05499](https://arxiv.org/abs/2303.05499)
- TAPIR: [https://arxiv.org/abs/2306.08637](https://arxiv.org/abs/2306.08637)
- CoTracker: [https://arxiv.org/abs/2307.07635](https://arxiv.org/abs/2307.07635)
- ALOHA / Mobile ALOHA: [https://tonyzhaozh.github.io/aloha/](https://tonyzhaozh.github.io/aloha/)
- UMI: [https://umi-spinal.github.io/](https://umi-spinal.github.io/)
- Guided Policy Search: [https://proceedings.mlr.press/v28/levine13.html](https://proceedings.mlr.press/v28/levine13.html)

整篇 paper 我觉得最值得 take away 的不是 GLIDE 本身，而是 **filtered BC + residual action + flying point aug + keypoint task spec 这一套 sim-to-real recipe**——这套 recipe 在很多 contact-rich 任务上应该都能复用。
