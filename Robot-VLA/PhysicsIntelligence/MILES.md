---
source_pdf: MILES.pdf
paper_sha256: 6b6d76e296455f06428efc74f229ed9df2724154b2fb4318fc9e5916d1ab6c39
processed_at: '2026-08-05T18:09:50-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲 MILES

## 一句话版本

你给机器人演示一遍怎么开锁，机器人自己围着那个锁瞎溜达半小时，记录下"我每次溜达完怎么走回演示里的那个位置"，拿这堆数据训练一个 policy，就完事了。

---

## 为什么要搞这个

教机器人干活现在有三种办法，每种都有要命的毛病：

**第一种：给一个 demo，直接模仿。** 机器人记住了"这条路径怎么走"，但它只会走这条路径。你把它手往左挪 3 厘米，它看到的画面就不一样了，它就懵了，下一步乱走，再下一步更乱，越来越偏。这叫 covariate shift。

**第二种：给几百个 demo。** 这下覆盖广了，policy 鲁棒了。但谁愿意演示几百遍开锁？累死。

**第三种：给一个 demo，然后让机器人自己用 RL 乱试。** 理论上行，但 RL 在 6-DoF action space 里 random exploration 基本是在大海捞针。更糟的是机器人一乱动就把东西撞翻了，你得跑过去把锁摆正重新开始。这个 reset 的人工成本比直接多给几个 demo 还累。

MILES 说：我来个第四种。

---

## MILES 的核心 idea

你演示一遍开锁，比如 30 个 waypoint。机器人记住这 30 个位置。

然后机器人自己开始干活。它在第 1 个 waypoint 附近随便偏一点（4 厘米、4 度以内），然后试图直线走回第 1 个 waypoint。走的过程中它记录看到的画面、感受到的力、以及它实际做的动作。

这条"偏出去再走回来"的轨迹叫一条 **augmentation trajectory**。

机器人对 30 个 waypoint 的每一个都重复 10 次这种"偏出去再走回来"。

完事之后，它手上有几百条这样的轨迹。每条轨迹的结尾都落在 demo 的某个 waypoint 上，所以可以把"走回来"这部分和"从那个 waypoint 接着执行 demo"这部分拼起来，变成一条完整的新 demo。

几百条新 demo 拿来做个 behavioral cloning，训个 LSTM，收工。

---

## 为什么这个 idea work

**为什么能解决 covariate shift：** 原来你只有一条 1D 的 demo 轨迹，policy 一偏就 OOD。现在机器人在 demo 周围 4cm 范围内密密麻麻铺了一堆"我偏了 2cm 怎么走回去"、"我偏了 3cm 怎么走回去"的轨迹，policy 见过的 state 空间从一条线变成一个管状区域，test 时偏一点根本不慌。

**为什么不需要 reset：** RL 之所以要 reset 是因为 random action 会把 environment 搞乱。MILES 的 action 不是 random 的，是"直线走回 demo waypoint"。这个动作很温和，一般不会把东西撞翻。而且万一真撞翻了，它有个检测机制会立即停手（下面讲），不会继续在错误状态上瞎采数据。

**为什么对 contact-rich 任务特别有效：** 这是最聪明的部分。机器人试图走直线回 demo waypoint，但用 impedance control，是软的。如果中间有个锁挡着，它走不了直线，会被迫绕一下。这个"绕一下"的曲线轨迹本身就是极其珍贵的训练数据 —— 它教会 policy "碰到锁的时候该怎么 regulate force 绕过去"。Environment 自动给它当老师，完全 free。

---

## 两个关键的停止条件

机器人采数据时有两个检查，任何一个触发都要处理：

**检查 1：Reachability（到底有没有走回去）**

机器人试图回到 demo waypoint $w_k$，结果因为卡住了没回到。通过 proprioception 直接读 EE pose 就知道。这种轨迹直接扔掉，因为它的终点不在 demo 上，没法 fuse。

**检查 2：Environment Disturbance（环境有没有被搞乱）**

机器人回到 $w_k$ 了，但它路上把锁推歪了。这时虽然 EE pose 对了，但从 wrist camera 看到的画面和 demo 第 k 步的画面不一样了。

怎么判断"画面不一样"？用 DINO ViT 提特征，算两张图 patch 级 cosine similarity，低于 0.94 就算被搞乱了。

一旦检测到 environment disturbance，整个 data collection 立即停。记录停在第几个 waypoint，叫 $R$。

**为什么必须停：** 如果锁被推歪了，后续 demo 的动作就不对了（demo 是基于锁在原位教的），继续采的数据全是垃圾。

---

## 部分开环的妥协

如果 disturbance 发生在第 $R$ 个 waypoint，那 $R$ 到 $N$ 这段没采到数据。怎么办？

部署时：policy 闭环跑到第 $R$ 个 waypoint（通过检测 policy 一直输出 identity action 判断到了），然后剩下那段直接 open-loop replay demo 的动作。

这是个小妥协，但实用。多数任务 disturbance 发生在后面（比如 Twist Screw 任务拧螺丝时螺丝刀被拧转了），前面大部分是 closed-loop 的，足够鲁棒。

---

## 结果好不好

7 个 real-world task，每个 20 trials：

| 方法 | 平均成功率 |
|------|-----------|
| 直接 replay demo | 6% |
| RL-based (FISH, no reset) | 10% |
| Pose estimation + replay | 71% |
| **MILES** | **87%** |

最 impressive 的是 Lock with key 90%。这个任务 tolerance 极小，要插钥匙再转 90 度，pose estimation + replay 只能 50% 因为开环误差累积。MILES 闭环到最后所以碾压。

USB 插入 70% 是最差的，因为 USB port tolerance < 1mm，对视觉精度要求极高。

---

## 几个我 think about 的点

**1. 这个 idea 看起来简单，但其实很巧妙的地方在于"target 是 well-defined 的"。** RL exploration 难是因为没有 target，random action 不知道往哪走。MILES 的 target 永远是"demo 的某个 waypoint"，这个 target 是 proprioception 可测的、environment-independent 的 anchor。所以"走回 target"这个 action 可以自动 label，不需要人标。

**2. DINO features 当 disturbance detector 是个 nice trick。** 用 pixel-wise 比较对光照太敏感，用 object detector 太重。DINO 是 self-supervised 学的 semantic feature，对光照 robust 对 object pose 敏感，正好是要的性质。

**3. 顺序采数据这件事很关键。** Ablation 里把"按 demo 顺序"换成"随机顺序"成功率从 87% 掉到 43%。因为 robot 必须在 $w_k$ 执行 demo action $a_k$ 才能到 $w_{k+1}$，跳着来 robot 根本不在正确的 task state 上。

**4. 根本局限：single demo 没法处理 task variation。** Demo 里钥匙正面朝上，policy 不会知道钥匙翻倒怎么办。MILES 解决了 covariate shift，没解决 task ambiguity。真要 deploy 到 messy 现实世界，多 demo 还是要的。

**5. 部分开环是个 graceful degradation 但不是根本方案。** 如果 deployment 时 object 被撞歪了，replay 那段还是 会崩。但这个妥协比 RL 需要人一直 reset 好太多。

---

## 最直白的类比

想象你教人开车从 A 到 B，只演示了一遍。

- **BC from 1 demo**：学生只记住"先直走 100 米再右转"。结果你把他丢在偏离路线 5 米的地方，他就不会了，因为"先直走 100 米"这个指令已经不对了。
- **DAgger**：你在旁边坐着，学生一偏你就喊"往左打方向盘"。累死你。
- **RL**：学生自己在停车场乱开，撞了你就去把车摆正重新开始。撞 1000 次他可能学会了，你也累瘫了。
- **MILES**：学生自己把车开偏几米，然后尝试开回你演示的路线上，记下"我偏了 3 米的时候是怎么打方向盘回来的"。重复几百次，他现在知道偏一点怎么回正了。你回家喝茶了。

这就是 MILES。

---

# MILES: Making Imitation Learning Easy with Self-Supervision 深度讲解

## 1. 一句话直觉 (The Core Intuition)

MILES 的核心 idea 可以这样理解：**给定一条人类 demonstration，让 robot 自己去"探索"demonstration 周围的 state space，但探索的目的不是为了 random trial-and-error，而是为了收集"如何回到 demonstration 上"的轨迹**。这样就把 single demo 自动扩展成了一个 dense 的 dataset，从而绕开了 covariate shift 和 RL 的 reset 问题。

类比一下：传统 BC from one demo 就像是只给学生一张"标准答案卷"，他记住了答案路径但不知道从别的起点怎么走回去；DAgger 是老师在旁边随时纠错；RL 是让学生自己乱试但每次搞砸了都要重新摆桌子。MILES 让学生在答卷附近随机摆一个起点，然后让他自己走回答卷上，并记录下"我是怎么走回来的"，自动积累成 dataset。

论文链接：https://www.robot-learning.uk/miles
arXiv: https://arxiv.org/abs/2410.12720 (作者主页)

---

## 2. Problem Setup 与 Motivation 的精细拆解

### 2.1 三个范式的失败模式 (对应 Figure 1)

| 范式 | 数据需求 | Reset 需求 | 主要失败模式 |
|------|---------|-----------|-------------|
| BC from 1 demo | 1 demo | 1 reset | covariate shift — policy 遇到 demo 外的 state 就崩 |
| BC from N demos | N demos | 1 reset (per demo) | 人工成本高，几百条 demo 很累 |
| RL-based IL (e.g. GAIL, FISH) | 1 demo | 频繁 reset (training 期间) | random exploration 效率极低，environment 被扰动后没法 recover |
| **MILES** | 1 demo | 1 reset (initial) | 自监督收集"返回 demo"的轨迹，BC 训练 |

关键 insight：**RL-based IL 之所以需要 reset，是因为 exploration policy 会扰动 environment**（撞翻物体、掉落 gripper 抓住的东西）。MILES 通过两点避免：(1) 用 augmentation trajectory 这个结构化的 "go-to-demo-waypoint" 而非 random action；(2) 一旦检测到 environment disturbance 就立即停止 data collection。

### 2.2 为什么 single demo + covariate shift 是根本矛盾

Behavior cloning 假设 train 和 test 的 state distribution 一致 (iid)。单条 demo 只覆盖一条 1D manifold 在 state space 中的轨迹。Policy 在 test 时一旦偏离这条 manifold（哪怕几毫米），下一步的 observation 就 OOD，action 就错了，下一步偏得更远 — 这是 compounding error。MILES 的 augmentation trajectories 恰好把 demo 周围 4cm / 4° 范围内的 state 都填充了，policy 就有了"偏离后如何回正"的能力。

---

## 3. 形式化定义 (Formal Definitions)

### 3.1 Demonstration

$$\zeta := \{(w_n^\zeta, o_n^\zeta, a_n^\zeta)\}_{n=1}^N$$

变量解释：
- $\zeta$：单条 human demonstration
- $N$：demonstration 中的 waypoint 总数（例如 lock with key 任务里 N=32）
- $w_n^\zeta \in SE(3)$：第 $n$ 个 timestep 的 end-effector (EE) 6-DoF pose，通过 proprioception 测得
- $o_n^\zeta$：observation，包含 wrist camera RGB image + force-torque measurement (6 维)
- $a_n^\zeta$：action，包含 gripper state + 6-DoF EE pose target（相对于 timestep $n-1$ 的 EE pose，用 impedance controller 执行）

这里 action 用 **relative pose** 是关键设计 — 让 policy 自然 generalize 到不同 object pose。

### 3.2 Augmentation Trajectory

$$\tau_k := \{(w_m^{\tau_k}, o_m^{\tau_k}, a_m^{\tau_k})\}_{m=1}^M$$

约束：
$$\text{terminal state: } (w_M^{\tau_k}, o_M^{\tau_k}) = (w_k^\zeta, o_k^\zeta)$$

也就是说，$\tau_k$ 的终点必须落在 demonstration 的第 $k$ 个 waypoint 上。$M$ 是轨迹长度，不同 $\tau_k$ 的 $M$ 可以不同。

### 3.3 Fused Trajectory (新 demo)

$$\zeta_k := \{(o_m^{\tau_k}, a_m^{\tau_k})\}_{m=1}^M \cup \{(o_n^\zeta, a_n^\zeta)\}_{n=k}^R$$

注意这里把 $w$ (proprioception) 丢掉了，因为 policy 输入只用 vision + force，这是 spatial generalization 的必要条件。

最终 dataset：
$$\mathcal{D}_{new} = \{\zeta, \zeta_1^1, \zeta_1^2, \ldots, \zeta_1^Z, \ldots, \zeta_R^{Z-1}, \zeta_R^Z\}$$

- $Z$：每个 demo waypoint 收集的 augmentation trajectory 数量（实验中 $Z=10$）
- $R$：data collection 停止时的 demo timestep（如果没 disturbance 则 $R=N$）

---

## 4. Self-Supervised Data Collection 的精细流程

### 4.1 单条 augmentation trajectory 怎么采 (对应 Algorithm 2)

```
1. 当前 robot 在 demo waypoint w_k^ζ
2. 在 w_k^ζ 附近随机 sample 一个 pose w_1^τ_k（4cm, 4°范围内）
3. 用 straight line 移动到 w_1^τ_k（这一段不记录数据）
4. 从 w_1^τ_k 用 straight line 移回 w_k^ζ
   4.1 移动过程中 impedance controller 会让轨迹被 environment "reshape"
       —— 如果撞到 lock，轨迹会弯，但最终能到
   4.2 同时记录 RGB, force-torque, action (relative EE pose)
5. 到达 w_k^ζ 后，检查 reachability 和 environment disturbance
```

**直觉**: 这里的关键 trick 是用 impedance controller + compliance。robot "尝试"走直线，但 environment 的几何会迫使它绕过障碍，这种"被迫绕过的曲线"本身就是 contact-rich manipulation 的珍贵 training data。这相当于让 environment 自动 supervise "如何 regulate force 来 overcome friction/collision"。

### 4.2 Reachability 检查 (Algorithm 3)

$$\text{Reachable} := (w_M^{\tau_k} == w_k^\zeta)$$

通过 proprioception 直接读 EE pose，判断是否到达 target waypoint（在 controller 精度范围内）。

**为什么重要**: 如果 trajectory 没回到 $w_k^\zeta$，那它就没法 fuse 进 demonstration（fuse 的前提是终点 = demo waypoint）。这种情况下 augmentation trajectory 被丢弃，robot 被显式 re-commanded 回到 $w_k^\zeta$（Algorithm 4）继续下一轮采样。

### 4.3 Environment Disturbance 检查 (Algorithm 5)

这是 MILES 最巧妙的设计之一。

$$\text{sim} = \frac{1}{P} \sum_{x=1}^{P} \cos(f_{I_k^\zeta}^x, f_{I_M^{\tau_k}}^x)$$

- $P$：image patch 数量
- $f_{I_k^\zeta}^x$：demonstration 第 $k$ 步 RGB image 的第 $x$ 个 patch 的 DINO-ViT feature
- $f_{I_M^{\tau_k}}^x$：augmentation trajectory 终点 RGB image 的第 $x$ 个 patch 的 DINO-ViT feature
- $\cos(\cdot, \cdot)$：cosine similarity
- 阈值 $\theta = 0.94$（论文用 RLBench in CoppeliaSim 自动调出来的 universal 值）

**判断**: 如果 $\text{sim} < \theta$，说明 environment 被扰动了（比如 gripper 推歪了 object），整个 data collection 立即停止，记录 $R = k$。

**为什么用 DINO 而不用 pixel-wise / template matching**: 论文在 supplementary A.2 提到，pixel-wise Euclidean distance 和 template matching 对 lighting 变化和 noise 极其 sensitive。DINO ViT features 是 self-supervised 学到的 semantic features，对光照、轻微 noise robust，但对 object pose 变化敏感 — 这正好是想要的性质。

参考：
- DINO 原论文: https://arxiv.org/abs/2104.14294 (Caron et al., ICCV 2021)
- DINO features as dense descriptors: https://arxiv.org/abs/2203.10334 (Amir et al., ECCVW 2022)

### 4.4 Data Collection 的高层循环 (Algorithm 1)

外层 for loop 遍历 demo waypoints $k = 1, 2, \ldots, N$，内层 while loop 采 $Z$ 条 augmentation trajectory。**关键顺序约束**：必须按 demo 顺序进行（ablation 里 "No Sequence" 验证了这点，去掉后成功率从 87% 掉到 43%）。

**为什么顺序重要**: 因为 robot 需要在每个 $w_k^\zeta$ 执行 demo action $a_k^\zeta$ 才能 progress 到 $w_{k+1}^\zeta$。如果跳过中间 waypoint 直接到后面，robot 就不在正确的 task state 上，augmentation trajectory 起点就不对了。

---

## 5. Policy 架构解析

### 5.1 网络结构 (对应 supplementary A.4.1)

```
RGB image (H × W × 3)
     │
     ▼
ResNet-18 backbone (pretrained)
     │
     ▼ feature vector
     │
     ├── concat ──┐
                 │
Force-torque (6D) ── MLP ── 100-dim embedding
                 │
                 ▼
              concat
                 │
                 ▼
              LSTM
                 │
                 ▼
       6-DoF action + 1-dim gripper action
```

训练目标 (standard BC likelihood)：
$$\mathcal{L}(\psi) = -\sum_{(o, a) \in \mathcal{D}_{new}} \log p_\psi(a \mid o)$$

### 5.2 Deployment 时两种模式

**Case A: No environment disturbance ($R = N$)**

完全 closed-loop，policy $f_\psi$ 一直运行到 task 完成。

**Case B: Environment disturbance occurred ($R < N$)**

$$\pi = \{f_\psi, \zeta_{remaining}\}, \quad \zeta_{remaining} := \{a_n^\zeta\}_{n=R}^N$$

部署流程：
1. $f_\psi$ closed-loop 控制直到它持续输出 identity transformation（表示 "我已经到达 $R$ 状态了"）
2. 然后开环 replay $\zeta_{remaining}$

**如何检测 $f_\psi$ 收敛到 identity**: 论文 A.4.4 说得简单粗暴 — 持续预测 identity transformation 时切换。这是个很 pragmatic 的设计。

**LSTM hidden state reset**: 每 $2R$ timesteps reset 一次。直觉是 policy 不需要记住比一个完整任务还长的 history。

---

## 6. 实验 table 精细解读

### 6.1 主实验 (Table 1, 7 个 real-world tasks, 20 trials each)

| Method | Lock | USB | Socket | Power | Screw | Bread | Lid | Mean |
|--------|------|-----|--------|-------|-------|-------|-----|------|
| Demo Replay | 0 | 0 | 0 | 0 | 0 | 15 | 25 | 6 |
| Reset Free Residual RL | 0 | 15 | 35 | 0 | 0 | 0 | 0 | 7 |
| Reset Free FISH | 0 | 30 | 25 | 15 | 0 | 0 | 0 | 10 |
| Pose Est + Demo Replay | 50 | 10 | 85 | 80 | 70 | 100 | 100 | 71 |
| **MILES** | **90** | **70** | **85** | **85** | **85** | **95** | **100** | **87** |

**关键观察**:

1. **Contact-rich 任务 (Lock, USB, Screw)**: MILES 大幅领先 Pose Est + Demo Replay。原因：这些任务 tolerance < 1mm，pose estimation 的小误差经过 open-loop replay 会 compound 成失败。MILES 在这些任务的 $R=N$ 部分 closed-loop，避免了 compounding。

2. **Non-contact-rich 任务 (Bread, Lid)**: Pose Est + Demo Replay 表现也很好 (100%)。因为这些任务 tolerance 大，open-loop replay 的小误差可以容忍。MILES 在这里没显著优势。

3. **RL baselines 全面崩盘**: Reset Free FISH 只有 10% mean。原因论文里说得很清楚 — RL exploration 在 6-DoF action space 下极其 inefficient，且 random action 扰动 environment 后没法 recover。FISH 原本在原 paper 里是有 reset 的，这里去掉 reset 性能就崩，恰恰证明了 RL-based IL 的 reset 依赖问题。

4. **USB 任务是 MILES 的短板 (70%)**: 论文归因于 USB port tolerance < 1mm。这是 contact-rich manipulation 的根本难题 — 即使 closed-loop policy 也需要极高的视觉精度和 force feedback 解读能力。

参考 FISH: https://arxiv.org/abs/2303.01497

### 6.2 Ablation (Table 2)

| Ablation | Mean | 主要影响 |
|----------|------|---------|
| No Sequence (随机 waypoint 顺序) | 43 | ↓44 — 最严重 |
| No Env Disturbance check | 47 | ↓40 — non-contact 任务崩 |
| No Reachability check | 73 | ↓14 — contact-rich 任务崩 |
| No Memory (去掉 LSTM) | 74 | ↓13 — history 重要 |
| Full MILES | 87 | — |

**直觉解读**:
- **No Sequence 最严重**: 因为 robot 不按 demo 顺序走， augmentation trajectory 的起点就和 demo 的 task progression 不一致，dataset 里充满了 "task state 错乱" 的样本。
- **No Env Disturbance 对 non-contact 任务毁灭性**: 想象 Open Lid 任务，robot 在 lid 已经被打开后还在采 augmentation trajectory，这些 trajectory 根本不是"如何开 lid"，而是"已经开了 lid 后再乱动"，BC 学到的就是垃圾。
- **No Reachability 对 contact-rich 任务毁灭性**: Lock 任务里，key 卡住了 trajectory 没回到 $w_k^\zeta$，但 trajectory 还被 fuse 进 dataset，policy 学到 "卡住了也要假装回到 demo waypoint" — 这是 hallucinated training data。
- **No Memory 影响适中**: LSTM 能记住 "我已经在返回 demo 的哪一步了"，对 multi-step trajectory 很重要。

### 6.3 Modality Ablation (Figure 5)

- Vision only: 在 USB / Socket 上更好（vision 提供 precise spatial info）
- Force only: 几乎全崩（force 在 free space 是 0，且有 symmetry ambiguity）
- Vision + Force: 整体最均衡

**直觉**: Force feedback 对"已经接触"的状态有用，但 vision 是判断"还没接触、要往哪接触"的主要信号。Force 信号噪声大，在某些任务上反而引入 confusion。

### 6.4 Dataset Size Ablation (Figure 6)

- High tolerance (Open Lid): 25% data (8 分钟) 还能 100%
- Low tolerance (Lock, USB, Screw): 50% data 开始崩

**直觉**: Contact-rich 任务需要密集覆盖 "collision escape" 的轨迹空间，data 越多越能 cover 各种卡住情形。Non-contact 任务 free space 几乎所有 augmentation trajectory 都是直线，policy 学个 "看到 lid 就往上拉" 就够了，几条 data 就行。

### 6.5 Simulation 实验 (Table 4, RLBench 5 tasks)

Mean: MILES 88 vs Pose Est+Replay 80 vs FISH 4 vs Residual RL 0。和 real-world 几乎一致，证明 real-world 优势不是 hardware-specific。

---

## 7. Limitations 和我的 critical thoughts

论文自己列了三个 limitations，但我想补充几个更深的：

### 7.1 论文承认的
1. **Wrist camera only**: FOV 限制，没法处理大 workspace。External camera 接近后切换是个 workaround。
2. **Workspace setup 人工**: 需要 human 把 workspace 清理成只有 task-relevant object。
3. **Per-task policy**: 没验证 multi-task generalization。

### 7.2 我补充的 critical points

**A. Reachability check 是 binary 的，丢弃了大量有用 data**

当 $w_M^{\tau_k} \neq w_k^\zeta$ 时 augmentation trajectory 被整个丢弃。但其实 "卡住" 这个 trajectory 本身就是 contact-rich 任务最珍贵的数据 — 它告诉你 "遇到这种 collision 怎么办"。一个改进方向是：在 trajectory 终点继续让 robot 尝试 recover（比如小幅度 wiggle），把 recover 后的最终 pose 作为 target waypoint，重新 label 整条 trajectory 指向一个 "approximate" 的 demo waypoint。这相当于让 MILES 处理 "partial reachability"。

**B. Environment disturbance 检测的 global threshold $\theta=0.94$ 不可微调**

这个 threshold 是用 RLBench 自动调的，但 real-world 不同 task 的 "可接受 disturbance" 不一样。Lock 任务里 key 转几度可能 OK，但 USB 任务里 USB 翻倒就是 fatal。一个改进是 per-task learnable threshold，或者用 binary classifier（disturbance vs no disturbance）替代 cosine similarity threshold。

**C. Augmentation trajectory 起点是 uniform random 4cm 范围**

这意味着大量 sample 浪费在 "free space 里随便走"上。一个改进是用 active sampling — 在 policy 当前 uncertainty 高的区域多采样。这接近于 DAgger 的 idea，但用 self-supervised uncertainty estimate 而非 human expert。

**D. Single demo 的根本限制：无法覆盖 task variation**

比如 USB 任务，如果 demo 里 USB 是正面朝上，policy 不会知道 USB 翻倒过来怎么处理。这是 single-demo IL 的 fundamental limitation。MILES 缓解了 covariate shift 但没解决 task variation。Multi-demo 仍然是必要的 for realistic deployment。

**E. 部分开环 (Case B) 是个妥协**

当 disturbance 检测到后，$\zeta_{remaining}$ 是开环 replay。这部分如果遇到新的 perturbation（比如 object 在 deployment 时被撞歪），policy 没法 recover。这是个 graceful degradation 而非根本解决方案。

**F. DINO ViT 的计算成本**

每个 augmentation trajectory 终点都要跑 DINO ViT 算 cosine similarity。对 real-time data collection 来说，每条 trajectory 终点 image 都 forward 一次 ViT，data collection 速度受限。论文没提 inference time，但实际部署可能需要 lightweight feature extractor。

---

## 8. 与相关工作的对比 intuition

### 8.1 vs DAgger (Ross, Gordon, Bagnell, 2011)

DAgger: https://arxiv.org/abs/1011.0686

DAgger 需要 interactive expert 在 policy 偏离时提供 correct action。MILES 用 "go-back-to-demo" 的 self-supervised mechanism 替代了 expert query — robot 自己知道 target waypoint (是 demo 的某个 state)，所以 "correct action" 自动被定义为 "指向 demo waypoint 的 relative EE movement"。

### 8.2 vs Coarse-to-Fine Imitation (Johns, 2021)

Paper: https://arxiv.org/abs/2103.15575

这是 MILES 的前作系列。Coarse-to-Fine 用 pose estimation + replay，正是 MILES 的 baseline "Pose Est + Demo Replay"。MILES 在 contact-rich 任务上的提升来自 closed-loop policy 替代 open-loop replay。

### 8.3 vs One-shot Imitation via Pose Estimation (Vitiello, Dreczkowski, Johns, 2023)

Paper: https://arxiv.org/abs/2310.22143

这个工作用 retrieval-based pose estimation，和 MILES 共享 lab (Imperial College London Robot Learning Lab, Edward Johns 组)。MILES 可以看作是这条线的进化版 — 从 pure pose estimation 走向 self-supervised closed-loop policy。

### 8.4 vs MimicGen (Mandlekar et al., 2023)

Paper: https://arxiv.org/abs/2310.01069

MimicGen 用 trajectory transform 自动生成新 demo，但需要 source demos 和精确的 object pose。MILES 不需要 object pose，因为 wrist camera + relative action 已经 implicit 了对齐。

### 8.5 vs FISH (Haldar, Pari, Rai, Pinto, 2023)

Repo: https://github.com/siddhanthaldar/FISH

FISH 用 value function 学 "how good is this trajectory matching the demo"，是 RL-based。Reset-Free FISH 在 MILES 实验里崩到 10%，正是 paper 想证明的：**RL 在 single-demo + no-reset 设定下根本不可行**。但要注意，FISH 原设定里是有 reset 的，去掉 reset 不公平 — 不过这恰恰是 MILES 想要的对比，证明 reset 才是 RL 的 bottleneck。

### 8.6 vs Diffusion Policy (Chi et al., 2023)

Paper: https://arxiv.org/abs/2303.04137

Diffusion Policy 用 multi-demo 数据训练 multi-modal action distribution。MILES 用 single demo + self-supervised augmentation，但 action distribution 是 uni-modal 的 (LSTM regression)。如果 task 本身需要 multi-modal action (e.g. 从左边或右边 grasp 都可以)，MILES 的 LSTM regression 会 average out。这是一个 fundamental 的 limitation，论文没讨论。

### 8.7 vs R+X (Papagiannis et al., 2024)

Paper: https://arxiv.org/abs/2407.12957

这是同一作者后续工作，从 everyday human video 检索 + 执行。说明作者在 MILES 之后往 video retrieval 方向走，解决 MILES 提到的 "per-task policy 没法 generalize 到新任务" 的 limitation。

---

## 9. 数据流 (Data Flow) 整体图解

```
Human demo (N waypoints, 1 reset)
        │
        ▼
┌────────────────────────────────┐
│  For k = 1 to N:               │
│    For j = 1 to Z=10:          │
│      Sample random pose near   │
│      w_k^ζ (4cm, 4°)           │
│      Move there (no record)    │
│      Move back to w_k^ζ        │
│      (record RGB, F/T, action) │
│      Check reachability        │
│      Check env disturbance     │
│        (DINO cosine sim <0.94?)│
│        If yes: R=k, STOP       │
│      Execute a_k^ζ to go to    │
│      w_{k+1}^ζ                 │
└────────────────────────────────┘
        │
        ▼
D = {τ_1, τ_2, ..., τ_R} (R = N if no disturbance)
        │
        ▼ Fuse each τ_k with demo segment [k, R]
        │
D_new = {ζ} ∪ {ζ_k^j : k=1..R, j=1..Z}
        │
        ▼
Train f_ψ (ResNet-18 + force MLP + LSTM) via BC
        │
        ▼
Deploy: f_ψ until identity action, then replay ζ_remaining (if R<N)
```

---

## 10. Time Complexity 估算

假设 demo $N=40$ waypoints（取平均），每 waypoint 采 $Z=10$ 条 augmentation trajectory，每条 trajectory 假设 30 timesteps（4cm 移动约几秒）：

- Total trajectories = $10 \times 40 = 400$
- Total timesteps = $400 \times 30 = 12000$ samples
- Data collection time: 论文 Table 3 显示约 20-40 min per task，符合每 waypoint 1 min 的估算
- Training: 标准 BC，12000 samples 在单 GPU 上几分钟

对比 RL-based IL: 通常需要 millions of environment steps，外加 resets，hours 到 days。

---

## 11. 关键公式的物理直觉汇总

### 11.1 Reachability
$$w_M^{\tau_k} == w_k^\zeta$$

物理意义：robot 真的回到了 demo 指定的 spatial location。这是 data "validity" 的硬约束 — 没有 reachability，augmentation trajectory 就是个 random motion，不能 teach "如何回到 demo"。

### 11.2 Environment Disturbance
$$\frac{1}{P}\sum_{x=1}^P \cos(f_{I_k^\zeta}^x, f_{I_M^{\tau_k}}^x) < \theta$$

物理意义：从 robot 视角看，environment "看起来"和 demo 那时候不一样了。DINO features encode semantic content（object 在哪、什么 pose），所以 cosine similarity 低 = object 被移动了 = task state 变了 = 后续 demo 不能继续 replay 了。

### 11.3 Fused Trajectory
$$\zeta_k = \tau_k \cup \{(o_n^\zeta, a_n^\zeta)\}_{n=k}^R$$

物理意义：一条完整的 "新 demo" = "从某个偏移位置回到 demo 第 k 步" + "从 demo 第 k 步执行到第 R 步"。这条新 demo 完整解决了 task，且起点在 demo manifold 附近，dense 覆盖了 state space。

### 11.4 Action 表示
$$a_n^\zeta = (\Delta p, \Delta R, g)$$

其中 $\Delta p \in \mathbb{R}^3$ 是 EE 位置增量，$\Delta R \in SO(3)$ 是 EE 旋转增量，$g \in \{0, 1\}$ 是 gripper state。**全部 relative to 上一 timestep 的 EE frame**，这是 spatial generalization 的 key — policy 学到的是 "看到这个视觉 + force pattern，往这个 relative 方向移动"，而非绝对位置。

---

## 12. 我的整体评价

**Strengths**:
1. Idea 简单优雅，self-supervision 的 "go-back-to-demo" 是个非常 natural 的 formulation
2. Real-world 实验扎实，7 个 task 跨 contact-rich / non-contact，20 trials each
3. DINO features 用于 disturbance detection 是个 nice trick，解决传统 pixel-wise brittle 问题
4. 两个 validity condition (reachability + disturbance) 的设计让 method gracefully handle early stopping

**Weaknesses**:
1. Single demo 是个 fundamental limitation，无法处理 multi-modal task variation
2. 部分开环 (Case B) 是个妥协
3. 4cm / 4° 范围 arbitrary，大 workspace 不适用
4. Wrist camera only 限制了 spatial awareness
5. LSTM regression 对 multi-modal action 无能为力
6. Per-task policy，没 multi-task 实验

**Future directions I'd explore**:
1. Active sampling for augmentation trajectory 起点（用 ensemble uncertainty）
2. Diffusion policy 替代 LSTM，handle multi-modal action
3. Learnable disturbance threshold per task
4. "Partial reachability" — 把没回到的 trajectory 用 recover motion 延伸，重新 label target
5. Multi-task MILES: 共享 backbone + task-conditioned policy
6. External camera + wrist camera fusion for larger workspace
7. Foundation model (e.g. VLM) 提供 "go-back-to-demo" 的 high-level guidance 替代 random sampling

---

## 13. 相关 reference 链接汇总

- MILES 主页: https://www.robot-learning.uk/miles
- DINO (Caron et al. ICCV 2021): https://arxiv.org/abs/2104.14294
- DINO as dense descriptors: https://arxiv.org/abs/2203.10334
- DAgger (Ross et al. 2011): https://arxiv.org/abs/1011.0686
- FISH (Haldar et al. 2023): https://arxiv.org/abs/2303.01497
- Coarse-to-Fine Imitation (Johns 2021): https://arxiv.org/abs/2103.15575
- One-shot IL via Pose Estimation (Vitiello et al. 2023): https://arxiv.org/abs/2310.22143
- MimicGen (Mandlekar et al. 2023): https://arxiv.org/abs/2310.01069
- Diffusion Policy (Chi et al. 2023): https://arxiv.org/abs/2303.04137
- R+X (Papagiannis et al. 2024): https://arxiv.org/abs/2407.12957
- YODO (Wen et al. 2022): https://arxiv.org/abs/2201.12716
- DART (Laskey et al. 2017): https://arxiv.org/abs/1703.06900
- RLBench (James et al. 2019): https://arxiv.org/abs/1909.12271
- SERL (Luo et al. 2024): https://arxiv.org/abs/2401.12987
- Diffusion Policy code: https://diffusion-policy.cs.columbia.edu/
- Franka Emika Panda: https://www.franka.de/

---

## 14. 一句话总结 intuition

MILES 把 imitation learning 的核心矛盾 — single demo 的 covariate shift vs multi-demo / RL 的人工成本 — 重新 frame 成 "**让 robot 自己生成 'how to get back to the demo' 的训练数据**"。通过两个简单但关键的 validity condition (reachability via proprioception, disturbance via DINO features)，它实现了 single demo + single reset + 30 min self-supervised data collection 的极简 IL pipeline，在 contact-rich 任务上显著超越 RL-based baselines。方法的美感在于 self-supervision 的 target 是 "demo waypoint" — 一个 well-defined, environment-independent 的 anchor — 这让 BC 训练变得 trivial，避开了 RL 的 sample inefficiency 和 reset dependency。
