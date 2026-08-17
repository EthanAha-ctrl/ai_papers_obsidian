---
source_pdf: ReKep.pdf
paper_sha256: 6820e718a66f3c52f27996d3ee2623260dd442eddeab2df098212eb6eb769334
processed_at: '2026-08-11T22:30:02-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 ReKep

## 一句话版

你给 GPT-4o 看一眼厨房的照片，说"把茶倒进杯子"。GPT-4o 看完就写了几段 Python 代码，代码里规定了"茶壶把手在哪儿""茶壶嘴要对准杯口""茶壶要保持直立别洒了"这些**几何关系**。然后一个数学求解器拿着这些代码，实时算出机器人手臂每一刻该怎么动。整个过程不需要给机器人任何物体的 3D 模型，也不需要训练数据。

---

## 为什么之前的做法不行

### 旧路子一：给机器人 CAD 模型

传统做法是给机器人每个物体的精确 3D model，然后算 rigid-body transform。问题：

- 你拿个茶壶还得先 3D scan 一遍？换一件衣服就完蛋——衣服没有固定形状
- 厨房里几十种东西都得提前建模？根本 scale 不起来
- [MegaPose](https://arxiv.org/abs/2212.06870)、[FoundationPose](https://arxiv.org/abs/2312.08344) 这类 6-DoF pose estimation 工作做得很好，但仍然要 CAD model

### 旧路子二：模仿学习

收集 100 段人类演示，训一个 transformer 策略，端到端从像素到动作。问题：

- 100 个 demo 在见过的场景里能到 93% 成功率（论文 A.12 simulation 实验）
- 换个没见过的茶壶，掉到 14%
- 数据 collected 不动，新任务又得重头训
- [RT-2](https://arxiv.org/abs/2307.15818) 这种 VLA model 每帧调一次大模型，只能跑 3 Hz，反应慢

### 旧路子三：让 VLM 直接预测 6-DoF 动作

[MOKA](https://arxiv.org/abs/2403.03174) 和 [PIVOT](https://arxiv.org/abs/2402.07872) 都尝试过让 VLM 直接输出 6 个自由度的 pose。但他们自己都报告："current VLMs are not capable of reliably predicting 6-DoF motions"。VLM 看图很在行，但要它输出一个精确的 (x, y, z, roll, pitch, yaw) 数值——它懵了。

ReKep 的 insight：**别让 VLM 干它不擅长的事**。VLM 擅长看图 + 写代码 + 理解语义关系，那就让它干这个；6-DoF 数值求解让数学 solver 干。

---

## ReKep 三个核心 trick

### Trick 1：用 keypoints 当"锚点"

[DINOv2](https://arxiv.org/abs/2304.07193) 是个 self-supervised vision transformer，它的 feature 对物体部位边界特别敏感。配上 [SAM](https://arxiv.org/abs/2304.02643) 做 segmentation，能在场景里自动挑出 5 个左右"语义上 meaningful"的点，比如：

- 茶壶的把手中心
- 茶壶的嘴尖
- 杯口的中心
- 杯子的底部

每个点就是 $\mathbb{R}^3$ 坐标，从 RGB-D 相机投影到 world frame。这些点是 **锚点**——后面所有约束都建立在这些点上。

为什么是 keypoint 而不是 dense feature field？因为 VLM 需要**用文字指代**它要约束的东西。"keypoint 3 要离 keypoint 7 大约 10cm"——这种表达 VLM 写得出来。"dense feature volume 在第 1347 个体素的位置要满足某个 manifold"——VLM 写不出来。Keypoint 是为 VLM 量身定制的 representation。

### Trick 2：把"约束"写成 Python 函数

ReKep 的核心：每个 constraint 是一个 Python 函数 $f:\mathbb{R}^{K\times 3}\rightarrow\mathbb{R}$，输入 $K$ 个 3D 点坐标，输出一个 cost。当 $f(k)\le 0$ 时约束满足。

具体例子（pour tea 的 stage 2，align 阶段）：

```python
def stage2_subgoal_constraint1(end_effector, keypoints):
    """茶壶嘴要在杯口正上方 10cm"""
    spout = keypoints[3]      # 茶壶嘴
    cup_opening = keypoints[7] # 杯口中心
    dist = np.linalg.norm(spout - cup_opening)
    # 要求 z 方向高于 10cm，且 x,y 对齐
    z_diff = spout[2] - cup_opening[2]
    xy_dist = np.linalg.norm(spout[:2] - cup_opening[:2])
    return max(xy_dist - 0.02, 0) + max(0.10 - z_diff, 0)
```

```python
def stage2_path_constraint1(end_effector, keypoints):
    """茶壶要保持直立，避免洒出来"""
    handle = keypoints[2]
    spout = keypoints[3]
    vector = spout - handle  # 茶壶的"轴向"
    # 这个 vector 应该接近水平（即 z 分量小）
    return abs(vector[2]) - 0.05
```

注意 path constraint 的妙处：它**全程每一步**都要满足，所以茶壶在 transport 过程中不会倾斜。sub-goal constraint 只在 stage **结束**时需要满足。这种 "end-state + transition" 的二分法本质上就是 [TAMP](https://www.annualreviews.org/doi/10.1146/annurev-control-071020-104040) 里的 precondition/effect，但用连续不等式重写了。

VLM 写这种 code 非常自然——它不需要懂 quaternion，只要用 NumPy 写"L2 距离""dot product""叉乘"这些它本来就熟的操作。**3D rotation 隐式表达在 vector 关系里**，这是 ReKep 跟 MOKA / PIVOT 的本质区别。

### Trick 3：让 solver 实时算动作

VLM 写完 constraint 后就下班了。后续全靠 numerical solver：

**Sub-goal problem**：解下一个 end-effector 该到哪儿（单个 6-DoF pose，单臂 $\mathbb{R}^6$ 或双臂 $\mathbb{R}^{12}$），满足当前 stage 的所有 sub-goal constraint，同时 minimize collision、reachability 等 auxiliary cost。

**Path problem**：从当前 pose 走到 sub-goal 的 dense trajectory，每一步都满足 path constraint。

求解器配方：
- 第一帧用 Dual Annealing（global search）+ SLSQP（local refinement），~1 秒
- 后续帧只用 SLSQP warm-start，~10 Hz

10 Hz 意味着机器人能实时反应——人把杯子挪一下，机器人下一帧就重新 plan，重新对齐。

---

## 整个 pipeline 怎么 run

```
[RGB-D 相机] → [DINOv2 + SAM] → 自动挑 keypoints
                                    ↓
                              [GPT-4o] ← 语言指令"pour tea"
                                    ↓
                          生成 N 个 stage 的 Python 约束函数
                                    ↓
                          [SciPy solver] ← 实时 keypoint tracking
                                    ↓
                          SE(3) end-effector trajectory
                                    ↓
                              [Franka arm 执行]
                                    ↑
                              20 Hz 视觉闭环更新 keypoints
```

VLM 只在 task 一开始调用一次，写完 code 就退出 loop。后续整个 execution 全靠 DINOv2 feature matching 跟踪 keypoints（20 Hz）+ SciPy solver 算动作（10 Hz）。这个 "expensive VLM once + cheap loop many" 的设计是关键——它让系统能 reactive 而又不需要每帧调 GPT-4o。

---

## Backtracking：让系统真的 reactive

人突然把杯子拿走会怎么样？论文 Algorithm 1 的 backtracking 机制：

1. 当前在 stage 2（align），path constraint 是"茶壶保持直立"——这个没违反
2. 但 sub-goal constraint"茶壶嘴在杯口上方"——现在杯口没了，违反
3. 检查 stage 1（grasp）的 path constraint——还是满足（茶壶还在手里）
4. **回退到 stage 2 重新 plan**：等 cup 出现新位置，重新 align
5. 如果连 grasp 都失效（茶壶被抢走），回退到 stage 1 重新 grasp

这种"违反哪条约束就回到哪儿"的机制让 ReKep 能处理人机协作和外部干扰。论文 Table 2 显示在有干扰的设定下，annotated ReKep 仍有 46.7% 成功率（VoxPoser 只有 6.7%）。

---

## 实验结果人话版

### 主实验（Table 1）

- **VoxPoser 总成功率 10%**，因为它输出 2D affordance map，无法表达 6-DoF 旋转，更别说 12-DoF 双臂协调
- **ReKep Auto 44.3%**：完全自动，VLM 自己写约束
- **ReKep Annotated 68.6%**：人工写约束（keypoint 还是自动提的）

Auto 和 Annotated 之间的 24% gap 就是"VLM 写约束还差 24%"——这是 VLM reasoning 的提升空间。

### 双臂任务 VoxPoser 全 0

Fold Garment、Pack Shoes、Collaborative Folding 这三个双臂任务，VoxPoser 全 0/10。因为 VoxPoser 的 2D value map 根本没法表达"左手抓左袖，右手抓右袖，同时往中间对折"这种 12-DoF 协调。ReKep 自然支持双臂——只要给 VLM 一组 keypoint，它写双臂的约束跟写单臂一样自然。

### 衣物泛化（Fig. 5）

8 种衣物：sweater / shirt / hoodie / vest / dress / pants / shorts / scarf。GPT-4o 在每个类别上生成的 fold 策略**很多跟人类常识一致**。比如 sweater：先双臂同时折两个袖子到中间，再用单臂把领子拉到底部——这就是人叠毛衣的常见手法。这种策略没在 prompt 里教，GPT-4o 内化的 world knowledge 直接产出。

Strategy Success 52.5%，Execution Success 73.8%。意思是只要 VLM 写对了约束，执行 73.8% 能成；52.5% 是 VLM 写对的概率。

### Simulation 对比（A.12）

OmniGibson 模拟器，Pour Tea 任务：

| Method | Seen Poses | Unseen Poses | Unseen Objects |
|---|---|---|---|
| RVT (100 demos 训练) | 93% | 31% | 14% |
| ReKep (零样本) | 75% | 68% | 72% |

**这个表是 ReKep 的核心 value proposition**。imitation learning 在训练分布内 93%，但分布外掉到 14%；ReKep 没有 demo，靠 VLM + solver，unseen objects 仍然 72%。**generalization 来自结构而非数据**——只要 VLM 能写出对的约束，solver 能解，就 work。

---

## 哪里还很烂

### Tracker 是最大瓶颈

Fig. 4 error breakdown：point tracker 占 ~40% 错误。DINOv2 feature matching 在物体被严重 occlusion 时经常失效。后续工作 [SpatialTracker](https://arxiv.org/abs/2403.23499)、[CoTracker3](https://arxiv.org/abs/2410.11831)、[TAPVid-3D](https://arxiv.org/abs/2407.11362) 都在攻这个问题，直接 plug-in 替换就能提升 ReKep。

### Long-horizon 直接 GG

A.14 "prepare breakfast tray" 10 个 stage：拿桌布→放盘里→拿面包→放桌布上→双手拿杯和茶壶→对齐→倒茶→放杯→双手拿托盘→递给人。自动 pipeline **完全失败**：keypoint proposal 在多物体场景里爆炸，VLM 写 10 个 temporally-dependent constraint 错误率高，tracker 在 10 stage 里频繁 occlusion。最后作者只能人工标注 keypoints + 每个 stage 重新 detect（不做持续 tracking）。这是 ReKep 当前 ceiling。

### Articulated object 没碰

抽屉、剪刀、柜门这种 articulated object 论文完全没做。作者在 A.11 给了思路：把 joint 也表达成 keypoint relation——prismatic joint = "某点只能沿某 line 移动"，revolute joint = "某点只能沿某 circle 移动"。但这需要 VLM 理解 articulation，[A3-VLM](https://arxiv.org/abs/2406.07549) 那种 articulation-aware 模型可能能补上。

### Contact-rich 任务 rigidity 假设失效

倒茶还行，但 peg-in-hole、tight insertion、推箱子这种 contact-rich 任务，"end-effector 动一下，object 跟着 rigidly 动"的假设错误。需要 learned dynamics model 或 [MuJoCo MJX](https://mujoco.readthedocs.io/) 这种 differentiable physics。论文 [Model-based Control with Sparse Neural Dynamics](https://arxiv.org/abs/2402.17070) 给了方向。

### Task-space planning 的 kinematic 问题

Solver 在 task space 优化 end-effector pose，不显式考虑 robot joint limit。有些 pose 即使 IK 有解，motion 也很难（接近 singularity）。作者承认 joint-space planning 更好但更慢，留作未来工作。[cuRobo](https://arxiv.org/abs/2310.17274) 这种 GPU 加速 kinematic solver 是直接解决方案。

---

## 我的 takeaways

### 1. Representation 比算法重要

ReKep 之所以 work，本质上找到了一个 **VLM 能写、solver 能解、perception 能 track** 的 representation sweet spot——keypoints 之间的 Python 不等式。当年 [AlphaGo](https://www.nature.com/articles/nature16961) 用 policy+value network 而非纯 search，也是 representation-first 思路。

### 2. Declarative > Imperative for robot control

传统 robot stack 是 imperative：写 trajectory、写 controller。ReKep 走 declarative：写 constraint，让 solver 算 trajectory。这跟 [CLP](https://en.wikipedia.org/wiki/Constraint_logic_programming)、[model checking](https://en.wikipedia.org/wiki/Model_checking) 思路一致。LLM 写 declarative spec 比写 imperative code 容易太多——这跟软件工程里 [spec-driven development](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/sdp04-bickford.pdf) 趋势对齐。

### 3. "VLM as spec writer, solver as executor" 范式

ReKep 提供了一个 blueprint：VLM 输出不是直接 action 或直接 text，而是 **structured code that an external solver can consume**。这种范式可以推广到 code generation、circuit design、game level generation 等领域。这跟 [Eureka](https://arxiv.org/abs/2310.12931) 让 LLM 写 reward function、[Code as Policies](https://arxiv.org/abs/2209.07753) 让 LLM 写控制代码同源，但 ReKep 写的是 **constraint**，更 declarative、更 expressive、更容易让 solver exploit。

### 4. Keypoint 是 universal interface

Keypoint 同时是 perception output（DINOv2）、VLM reference（numerical mark）、optimization variable、human inspection handle。这种 "one representation multiple consumers" 是好的 system design 标志。对比 [NDF](https://arxiv.org/abs/2112.05124) 那种 implicit dense descriptor field，VLM 没法用文字直接指——这是 ReKep 选择 explicit keypoints 的妥协。

### 5. 闭环频率是 foundation model 时代 critical resource

ReKep 把 VLM 调用降到 task start 一次，剩下全靠 20Hz tracker + 10Hz solver。这种"贵 VLM 调一次 + 便宜循环跑很多次"的设计是 VLA 系统的核心 trade-off。对比 [RT-2](https://arxiv.org/abs/2307.15818) 每帧调 VLA 模型只能跑到 ~3 Hz。ReKep 的 reactive 能力直接来自这个架构选择。

### 6. Sim-to-real 通过结构而非数据

A.12 simulation 实验显示 zero-shot ReKep 在 unseen 上 72%，远超 100 demos 的 monolithic 14%。原因不是 ReKep 学到更好 representation，而是它**不依赖分布假设**——只要 VLM 写对约束 + solver 能解，就 work。这暗示 foundation-model-driven manipulation 的 generalization 范式可能从 "more data" 转向 "better structure"。跟 [FunSearch](https://www.nature.com/articles/s41586-023-06647-8)、[OPRO](https://arxiv.org/abs/2310.03720) 那种"LLM as optimizer"思路呼应。

### 7. 后续工作直接利好的方向

- 用 [SpatialTracker](https://arxiv.org/abs/2403.23499) 替换 DINOv2 tracker → 打掉最大 error source
- 用 GPT-4o vision 或 [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet) 替换 VLM → 提升 constraint quality
- 用 [cuRobo](https://arxiv.org/abs/2310.17274) 替换 PyBullet IK → 解决 task-space kinematic 问题
- 整合 [SayCan](https://arxiv.org/abs/2204.01691) 类 hierarchical planner → 解决 long-horizon ceiling

---

## 一个具体例子走通整个 pipeline

Pour tea 这个任务从头到尾：

**Step 1：Keypoint Proposal**

RGB-D 相机拍一张厨房照片。DINOv2 ViT-S/14 提 features，SAM 分割出 teapot mask 和 cup mask。每个 mask 内做 PCA + k-means（k=5），取 cluster centroids。最后得到 keypoints（编号 0-7 大概）：

- 0, 1, 2：teapot handle 区域（grip 候选）
- 3：teapot spout 尖端
- 4, 5：teapot body 上几个点
- 6：cup 开口中心
- 7：cup 底部

**Step 2：VLM 生成 ReKep**

GPT-4o 看到 overlay 了数字的图 + 指令"(single-arm) pouring tea from teapot"，输出：

```python
num_stages = 3

# Stage 1: grasp teapot handle
def stage1_subgoal_constraint1(end_effector, keypoints):
    """end-effector 到 teapot handle 中心"""
    handle_center = (keypoints[0] + keypoints[1] + keypoints[2]) / 3
    return np.linalg.norm(end_effector - handle_center) - 0.02

# Stage 2: align spout with cup opening
def stage2_subgoal_constraint1(end_effector, keypoints):
    """spout 在 cup opening 正上方 10cm"""
    spout = keypoints[3]
    cup = keypoints[6]
    xy_dist = np.linalg.norm(spout[:2] - cup[:2])
    z_diff = spout[2] - cup[2]
    return max(xy_dist - 0.02, 0) + max(0.10 - z_diff, 0)

def stage2_path_constraint1(end_effector, keypoints):
    """teapot 保持直立"""
    handle = (keypoints[0] + keypoints[2]) / 2
    spout = keypoints[3]
    vector = spout - handle
    return abs(vector[2]) - 0.05  # z 分量应该小（水平）

# Stage 3: pour
def stage3_subgoal_constraint1(end_effector, keypoints):
    """spout 在 cup 上方 5cm 且倾斜"""
    spout = keypoints[3]
    cup = keypoints[6]
    z_diff = spout[2] - cup[2]
    handle = (keypoints[0] + keypoints[2]) / 2
    vector = spout - handle
    # 要倾斜约 90 度，即 vector 接近垂直
    return max(0.05 - z_diff, 0) + abs(abs(vector[2]) / np.linalg.norm(vector) - 1.0)
```

**Step 3：Solver 执行**

- Sub-goal solver 解 stage 1：找到 end-effector 该去 teapot handle 中心，~1 秒（Dual Annealing + SLSQP）
- Path solver 解从当前 pose 到 handle 的 trajectory，~10 Hz
- Robot 执行，DINOv2 tracker 20 Hz 更新 keypoint 位置
- 到达 grasp pose → 抓住 → 进入 stage 2
- Sub-goal solver 解 stage 2：spout 要到 cup 上方 10cm
- 每步检查 path constraint：teapot 保持直立（不洒）
- 到达 align pose → 进入 stage 3
- 解 stage 3 sub-goal：tilt 90 度倒水
- 完成

**Step 4：如果中途人动了杯子**

- Stage 2 执行中，tracker 发现 cup keypoint 位置突变
- Sub-goal constraint"spout 在 cup 上方 10cm"重新违反
- Solver 重新 plan，让 spout 追上新的 cup 位置
- 全程 path constraint"teapot 直立"保持

---

## 最终一句话

ReKep 的 elegance 在于：**让大模型只做它擅长的事（看图 + 写代码 + 推理关系），让数学求解器做它擅长的事（实时算 6-DoF 数值），让 DINOv2 做它擅长的事（视觉跟踪），三者通过 keypoints 这个 universal interface 串起来**。这种 "modular + 各取所长" 的设计哲学，比任何 monolithic end-to-end 模型都更 robust、更 interpretable、更 generalizable。

论文主页：[rekep-robot.github.io](https://rekep-robot.github.io)

References：
- [ReKep arXiv](https://arxiv.org/abs/2409.01652)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [GPT-4o](https://arxiv.org/abs/2303.08774)
- [SAM](https://arxiv.org/abs/2304.02643)
- [Sequence-of-Constraints MPC](https://ieeexplore.ieee.org/document/9982296)
- [VoxPoser](https://arxiv.org/abs/2307.05973)
- [MOKA](https://arxiv.org/abs/2403.03174)
- [PIVOT](https://arxiv.org/abs/2402.07872)
- [Eureka](https://arxiv.org/abs/2310.12931)
- [Code as Policies](https://arxiv.org/abs/2209.07753)
- [KPAM](https://arxiv.org/abs/1903.00606)
- [NDF](https://arxiv.org/abs/2112.05124)
- [SpatialTracker](https://arxiv.org/abs/2403.23499)
- [CoTracker3](https://arxiv.org/abs/2410.11831)
- [cuRobo](https://arxiv.org/abs/2310.17274)
- [RT-2](https://arxiv.org/abs/2307.15818)
- [AnyGrasp](https://github.com/graspnet/anygrasp_sdk)
- [nvblox](https://arxiv.org/abs/2311.00626)
- [SayCan](https://arxiv.org/abs/2204.01691)
- [FunSearch](https://www.nature.com/articles/s41586-023-06647-8)
- [OPRO](https://arxiv.org/abs/2310.03720)

---

# ReKep: 用关键点关系约束表达 robotic manipulation 任务

## 1. TL;DR 与核心 intuition

ReKep 把 manipulation task 表达为一组 **Python 函数** $f:\mathbb{R}^{K\times 3}\rightarrow\mathbb{R}$，输入是场景里 $K$ 个 3D keypoints 的坐标，输出是 cost；当 $f(k)\le 0$ 时约束满足。整套系统有三个解耦层：

- **Keypoint layer**：DINOv2 + SAM 自动提议 semantically meaningful 3D points（无需 CAD model）
- **Constraint layer**：GPT-4o 通过 visual prompting 把自然语言指令翻译成 Python 函数（stateless NumPy 运算）
- **Optimization layer**：SciPy solver 实时（~10 Hz）求解 SE(3) end-effector pose trajectory

这个 representation 之所以好用，可以用一句话概括 intuition：**让 VLM 只负责"指出哪些点之间应该有什么几何关系"，让 numerical solver 负责"算出具体 6-DoF pose"**。MOKA 和 PIVOT 这些工作都报告 VLM 直接预测 6-DoF rotation 不可靠（参见 [MOKA](https://arxiv.org/abs/2403.03174)、[PIVOT](https://arxiv.org/abs/2402.07872)），ReKep 用 keypoints + dot product/cross product 在 3D Cartesian 空间里 implicitly 表达 SO(3)，绕开了 explicit rotation representation 的难题。

项目主页：[rekep-robot.github.io](https://rekep-robot.github.io)；论文 arXiv 链接 [ReKep](https://arxiv.org/abs/2409.01652)。

---

## 2. ReKep 表示的精妙之处

### 2.1 单 keypoint 的限制 vs 多 keypoint 的表达能力

单点只有 $\mathbb{R}^3$，但论文里反复强调：**multiple keypoints + rigidity assumption 可以表达 line / surface / volume / full SO(3) rotation**。具体地：

- **Line**：两个 keypoints $k_i, k_j$ 形成 vector $v = k_j - k_i$
- **Angle between vectors**：用 dot product $v_1 \cdot v_2 = \|v_1\|\|v_2\|\cos\theta$，所以 $\cos\theta - \cos\theta^* \le 0$ 就是一个 ReKep constraint
- **Surface normal alignment**：三点叉乘 $n = (k_j-k_i)\times(k_l-k_i)$，再和目标 normal 做 dot product
- **Position alignment**：L2 distance $\|k_i - k_j\|_2 - d^* \le 0$

公式里每个符号的含义：
- $k_i \in \mathbb{R}^3$：第 $i$ 个 keypoint 在 world frame 的 Cartesian 坐标
- $K$：scene 里 keypoints 总数
- $\boldsymbol{k} \in \mathbb{R}^{K\times 3}$：所有 keypoints 拼成的数组
- $f$：单条 ReKep constraint，stateless Python function

### 2.2 Sub-goal constraints vs Path constraints

每个任务被分解为 $N$ 个 stage，每个 stage $i$ 有两类约束：

- $\mathcal{C}_{\text{sub-goal}}^{(i)}=\{f_{\text{sub-goal},1}^{(i)},\dots,f_{\text{sub-goal},n}^{(i)}\}$：stage **结束**时必须满足
- $\mathcal{C}_{\text{path}}^{(i)}=\{f_{\text{path},1}^{(i)},\dots,f_{\text{path},m}^{(i)}\}$：stage **全程**每一步都必须满足

Pour tea 的例子很形象：
- Stage 1 (grasp)：sub-goal = end-effector 到 handle；无 path
- Stage 2 (align)：sub-goal = spout 在 cup opening 上方 10cm；path = teapot 保持 upright
- Stage 3 (pour)：sub-goal = spout 上方 5cm 且 tilt；path = spout 始终在 cup opening 正上方

这种 "end-state + transition" 的二分法本质上就是把 TAMP 的 logical precondition/effect 用连续不等式重写——这是受 [Sequence-of-Constraints MPC (Toussaint et al. 2022)](https://ieeexplore.ieee.org/document/9982296) 启发的。

---

## 3. 完整 Optimization Formulation（Eq.1 详解）

完整问题写为：

$$
\min_{\mathbf{e}_{1:T}, g_{1:N}} \sum_{i=1}^{N}\left[\lambda_{\text{sub-goal}}^{(i)}(\mathbf{e}_{g_i}) + \sum_{t=g_{i-1}}^{g_i} \lambda_{\text{path}}^{(i)}(\mathbf{e}_t)\right]
$$

$$
\text{s.t.}\begin{cases}
\mathbf{e}_1 = \mathbf{e}_{\text{init}},\; g_0=1,\; 0<g_i<g_{i+1}\\
f(k_{g_i})\le 0,\;\forall f\in\mathcal{C}_{\text{sub-goal}}^{(i)}\\
f(k_t)\le 0,\;\forall f\in\mathcal{C}_{\text{path}}^{(i)},\; t=g_{i-1},\dots,g_i\\
k_{t+1} = h(k_t, \mathbf{e}_t),\; t=1,\dots,T-1
\end{cases}
$$

变量逐个解释：
- $\mathbf{e}_t \in SE(3)$：时刻 $t$ 的 end-effector pose，单臂 6-DoF，双臂拼成 12-DoF
- $g_i \in \{1,\dots,T\}$：从 stage $i$ 切到 stage $i+1$ 的时刻，是 auxiliary decision variable，意味着 solver 也要决定 stage 切换时机
- $T$：trajectory 总长度
- $N$：stage 总数
- $h$：keypoint forward model，给定 $\mathbf{e}_t$ 怎么更新 $k_t$
- $\lambda_{\text{sub-goal}}^{(i)}, \lambda_{\text{path}}^{(i)}$：auxiliary cost（collision avoidance, reachability, regularization 等）

这是 **direct shooting** 形式的 trajectory optimization（参考 [Tedrake, Underactuated Robotics](https://underactuated.csail.mit.edu/)）：决策变量直接是 trajectory 本身，不需要 LQR 那种 feedback structure。

---

## 4. Hierarchical Decomposition（为什么能跑 10 Hz）

直接解 Eq.1 太慢，论文做了两层分解：

### 4.1 Sub-goal problem (Eq.2)

$$
\arg\min_{\mathbf{e}_{g_i}} \lambda_{\text{sub-goal}}^{(i)}(\mathbf{e}_{g_i}) \quad \text{s.t.}\quad f(k_{g_i})\le 0,\;\forall f\in\mathcal{C}_{\text{sub-goal}}^{(i)}
$$

只求下一个 sub-goal 的 **单个** end-effector pose。决策变量：单臂 $\mathbb{R}^6$（3 位置 + 3 Euler angle），双臂 $\mathbb{R}^{12}$。Bounds 是 workspace 范围 + 朝下的 hemisphere（Franka 关节限位）。

### 4.2 Path problem (Eq.3)

$$
\arg\min_{\mathbf{e}_{t:g_i}, g_i} \lambda_{\text{path}}^{(i)}(\mathbf{e}_{t:g_i}) \quad \text{s.t.}\quad f(k_{\hat{t}})\le 0,\;\forall f\in\mathcal{C}_{\text{path}}^{(i)},\;\hat{t}=t,\dots,g_i
$$

从当前 pose 走到 sub-goal 的 dense trajectory。决策变量数 = 距离 / step size（20cm 位置 + 45° 旋转），spline 拟合后 dense sample 送低层 controller。

### 4.3 Solver 配方

- **第一次求解**：Dual Annealing（[Xiang et al. 1993](https://www.sciencedirect.com/science/article/pii/037596019390655J) 的 generalized simulated annealing）做 global search，再用 SLSQP（[Kraft 1988](https://www.science.gov/sites/default/files/citation_files/a-software-package-for-sequential-quadratic-programming.pdf)）做 local refinement。~1s。
- **后续帧**：只用 SLSQP warm-start，~10 Hz。
- 决策变量归一化到 $[0,1]$，对 numerical optimizer 稳定性至关重要。

### 4.4 Backtracking

```
while i ≤ N:
    if ∃ f ∈ C_path^(i) s.t. f(k_t) > 0:  # path constraint 违反
        i ← i - 1                          # 回退到上一 stage
        continue
    if distance(e_t, e_{g_i}) < ε:        # 到达 sub-goal
        i ← i + 1                          # 前进到下一 stage
        continue
    solve sub-goal (Eq.2)
    solve path (Eq.3)
    execute next m actions
    t ← t + m + 1
```

这个机制让系统 reactive：人把 cup 拿走 → stage 2 的 path constraint "teapot 在 transport 中保持 upright" 不违反，但 stage 2 sub-goal "spout 在 cup opening 上方" 不成立 → robot 重新进入 stage 2 求解，重新 align。如果 cup 被从 gripper 抢走 → stage 1 的 grasping path constraint 失效 → 回退到 stage 1 重新 grasp。

---

## 5. Forward Model $h$ 的 "Local Rigidity" 技巧

要解 Eq.2/Eq.3 必须知道 $\Delta\mathbf{e}$ 如何引起 $\Delta k$。论文做"local rigidity"假设：

$$
k'_{\text{[grasped]}} = \mathbf{T}_{\Delta\mathbf{e}} \cdot k_{\text{[grasped]}}
$$

- $k_{\text{[grasped]}}$：与 grasped object 同属一个 mask 的 keypoints（由 SAM 提供）
- $\mathbf{T}_{\Delta\mathbf{e}} \in SE(3)$：end-effector pose 变化对应的 rigid transform
- 其他非 grasped keypoints 假设 static

**关键**：这个假设只在 0.1s 求解窗口内成立。真实 $k_t$ 由 DINOv2 feature matching 在 20Hz 视觉追踪得到，每帧重新塞回 optimizer。这是经典的 "model-predict + observation-correct" 思路，跟 EKF 的 predict-update 结构异曲同工。

对 deformable object（衣服、毛毯）这个假设破坏，所以 prompt 里专门写了 "Avoid using path constraints when manipulating deformable objects"。

---

## 6. Keypoint Proposal Pipeline 的细节

### 6.1 完整流程（Appendix A.5）

1. 用 DINOv2 with registers (ViT-S/14) 提 patch features $\mathbf{F}_{\text{patch}}\in\mathbb{R}^{h'\times w'\times d}$
   - $h', w'$：patch grid size（image / 14）
   - $d$：feature dim，ViT-S 是 384
2. 双线性插值上采样到 $\mathbf{F}_{\text{interp}}\in\mathbb{R}^{h\times w\times d}$
3. SAM 提所有 mask $\mathbf{M}=\{\mathbf{m}_1,\dots,\mathbf{m}_n\}$
4. **对每个 mask**：先 PCA 降到 3 维（$\mathbf{F}_{\text{PCA}}=\text{PCA}(\mathbf{F}_{\text{resized}}[\mathbf{m}_i], 3)$）。PCA 的作用是去 texture 干扰，让 cluster 更聚焦于 part boundary
5. 在 mask 内做 k-means（$k=5$，Euclidean distance）
6. 取 cluster centroid 作为 candidates
7. 投影到 world frame（用 calibrated RGB-D）
8. Mean Shift（bandwidth 8cm）去重

### 6.2 为什么是 DINOv2 + SAM

Appendix A.13 做了消融：DINOv2 vs CLIP vs ViT（supervised ImageNet）± SAM。

| Method | Cup handle 区分 | Box panel 区分 | Top/side panel 对比 |
|---|---|---|---|
| SAM + DINOv2 | 锐利 cyan | 清晰 | blue/green 对比强 |
| SAM + CLIP | 有区分但 saturating | 中等 | 中等 |
| SAM + ViT | 弱 | 弱 | 弱 |
| DINOv2 only | 强 | 强 | 强，但跑到背景上 |

SAM 的 objectness prior 至关重要：没有 SAM，DINOv2 keypoints 会出现在 background。这与 [Deep ViT Features](https://arxiv.org/abs/2112.05814)、[D³Fields](https://arxiv.org/abs/2309.16118)、[CoTracker](https://arxiv.org/abs/2307.07635) 的观察一致——DINOv2 特征对 object part 边界非常敏感。

### 6.3 Keypoint Tracker（Appendix A.7）

简化版 [D³Fields](https://arxiv.org/abs/2309.16118)：
1. 初始化：聚合每个 keypoint 2cm 范围内所有像素的 DINOv2 特征，取均值作为 reference feature
2. 跟踪：每帧算 cosine similarity，top-100 matches（cutoff 0.6），用 median deviation (m=2) reject outlier，最后 uniform filter window=10
3. 频率 20 Hz

**这是系统最大错误源**（见后文 error breakdown），原因：intermittent occlusion 让 DINOv2 特征匹配失败。这也是为什么后续工作 [SpatialTracker](https://arxiv.org/abs/2403.23499)、[TAPIR](https://arxiv.org/abs/2306.13831) 这类专门的 3D point tracker 会显著提升整体性能。

---

## 7. VLM Constraint Generation 的 prompt 工程

Prompt 在 Appendix A.6 完整给出。几个值得注意的设计：

1. **No image-text in-context examples**：只给 text-only examples。这迫使 VLM 用 internalized world knowledge 而不是模仿
2. **强制 structured output**：固定 Python code block 格式，每个 stage 一组 `stage{i}_subgoal_constraint{j}` 和 `stage{i}_path_constraint{j}`
3. **Avoid `if` statements**：保持 constraint 是 smooth function，对 gradient-based optimizer 友好
4. **Grasping 单独成 stage**：保证 VLM 写出的 constraint 在 grasp 时只涉及 end-effector + grasp point，避免逻辑混乱
5. **Deformable object 警告**：明示 "avoid path constraints for deformable objects"
6. **Rotation 隐式表达**："You may use two keypoints to form a vector, which can be used to specify a rotation (by specifying the angle between the vector and a fixed axis)"

最后一个点的妙处：VLM 不需要懂 quaternion、axis-angle、Euler，只要在 3D Cartesian 空间用 dot product 表达 "这个向量应该和那个向量夹角是多少度"。具体 6-DoF rotation 由 solver 算出。这是 ReKep 跟 MOKA / PIVOT 的本质区别——后两者逼 VLM 直接 output rotation 参数，ReKep 让 VLM output **关系**，由 solver 求 **数值**。

---

## 8. Sub-goal Solver 和 Path Solver 的 cost 组成

### Sub-goal solver 的 cost term：

- **Constraint violation**：ReKep cost × 大权重
- **Scene collision**：用 [nvblox](https://arxiv.org/abs/2311.00626) 算 ESDF，gripper + grasped object 用 farthest point sampling 降到 30 个点，threshold 15cm
- **Reachability**：用 [PyBullet](https://pybullet.org/) IK 残差作 proxy（占 40% objective 评估时间）
- **Pose regularization**：靠近当前 pose
- **Consistency**：靠近上次 solution（perception noise 缓冲）
- **(Dual-arm) Self-collision**：两个点集 pairwise distance
- **Grasp metric**（仅 grasping stage）：用 [AnyGrasp](https://github.com/graspnet/anygrasp_sdk)

### Path solver 额外：

- **Path length**：dense samples 的差分和
- **Table clearance**：惩罚穿桌面
- **Collision 忽略起点 5cm 范围**：避免 high-frequency noise 导致抖动

---

## 9. 实验数据精读

### 9.1 主实验（Table 1）

| Task | VoxPoser | ReKep Auto | ReKep Annot. |
|---|---|---|---|
| Pour Tea | 3/10 | 8/10 | 8/10 |
| Recycle Can | 3/10 | 6/10 | 8/10 |
| Stow Book | 0/10 | 3/10 | 6/10 |
| Tape Box | 4/10 | 7/10 | 8/10 |
| Fold Garment (双臂) | 0/10 | 5/10 | 6/10 |
| Pack Shoes (双臂) | 0/10 | 3/10 | 5/10 |
| Collab. Folding (双臂) | 0/10 | 4/10 | 7/10 |
| **Total** | **10.0%** | **44.3%** | **68.6%** |

关键观察：
- **双臂任务 VoxPoser 全 0**：因为 VoxPoser 用 2D affordance map，无法表达 12-DoF 协调
- **Auto vs Annot. gap ~24%**：这 24% 是 VLM + keypoint proposal 的错误贡献
- **Pour Tea 即使 Auto 也 80%**：因为 pour 是相对简单的几何关系（spout 在 cup 上方 + tilt）
- **Stow Book 最难**：15cm bookshelf opening 要求 6-DoF 精确插入，VLM 写 constraint 难

### 9.2 Disturbance 实验（Table 2）

| Task (Dist.) | VoxPoser | ReKep Auto | ReKep Annot. |
|---|---|---|---|
| Pour Tea | 0/10 | 2/10 | 4/10 |
| Tape Box | 2/10 | 3/10 | 5/10 |
| Collab. Folding | 0/10 | 3/10 | 5/10 |
| **Total** | **6.7%** | **26.7%** | **46.7%** |

干扰条件下 Annot. 从 68.6% → 46.7%，掉 22%。这反映 backtracking 机制虽然 work，但 keypoint tracker 在物体被突然移动时经常失效。

### 9.3 Garment folding 泛化（Table in Fig. 5）

8 种衣物，**Strategy Success 52.5%**（keypoints+constraints 对不对），**Execution Success 73.8%**（给定正确 strategy 能执行成功）。

最 striking 的发现：**VLM 自己泛化出人类常识的 fold 策略**。比如 sweater：先双臂 fold 两个 sleeve 到中间，再用单臂把 neck 拉到底部——这正是人叠毛衣的常见手法。这种 strategy 没在 prompt 里教，GPT-4o 内化的 world knowledge 直接产出。

### 9.4 Simulation 对比（Appendix A.12）

OmniGibson 的 Pour Tea：

| Method | Seen Poses | Unseen Poses | Unseen Objects |
|---|---|---|---|
| Monolithic policy (RVT, 100 demos) | 0.93 | 0.31 | 0.14 |
| ReKep (Zero-shot) | 0.75 | 0.68 | 0.72 |

这是 ReKep 的核心价值主张：**训练分布外，zero-shot ReKep 远超 imitation learning**。Monolithic 在 seen poses 上靠 100 demos 学到 93%，但 unseen objects 掉到 14%。ReKep 没有 demo，靠 VLM 写 constraint + solver 算 motion，unseen objects 仍有 72%。

---

## 10. Error Breakdown（Fig. 4）

模块错误占比（按 pipeline 时序依赖加权）：
1. **Point tracker**：最大（~40%）—— occlusion 是核心瓶颈
2. **Keypoint proposal**：次大（~25%）—— 漏掉关键 part（例如 fold 时 missing collar）
3. **VLM**：~20%—— 指错 keypoint 或写错 constraint
4. **Optimization**：很小（<10%）—— 因为 constraint 通常很多解
5. **Segmentation, 3D reconstruction, low-level controller**：都很小

这个 breakdown 对未来工作很有指导意义：**tracker > proposal > VLM > solver** 是改进优先级。后续工作如 [SpatialTracker](https://arxiv.org/abs/2403.23499)、[TAPVid-3D](https://arxiv.org/abs/2407.11362)、[CoTracker3](https://arxiv.org/abs/2410.11831) 直接利好 ReKep。

---

## 11. 与相关工作的关系网

### 11.1 Structural representation 谱系

- **Rigid-body pose**：[Kaelbling & Lozano-Perez](https://ieeexplore.ieee.org/document/6297148)、[MegaPose](https://arxiv.org/abs/2212.06870)、[FoundationPose](https://arxiv.org/abs/2312.08344)——需要 CAD model，deformable 失效
- **Keypoint-based**：[KPAM (Manuelli et al.)](https://arxiv.org/abs/1903.00606)、[Neural Descriptor Fields](https://arxiv.org/abs/2112.05124)、[D³Fields](https://arxiv.org/abs/2309.16118)——keypoints 但 constraint 通常 manual
- **ReKep**：keypoints 但 constraint 由 VLM 自动生成

### 11.2 Foundation model for robotics 谱系

- **VLM 直接 output action**：[RT-2](https://arxiv.org/abs/2307.15818)、[ManipLLM](https://arxiv.org/abs/2405.07036)——受限于 2D 像素输出
- **VLM output value map**：[VoxPoser](https://arxiv.org/abs/2307.05973)、[RoboPoint](https://arxiv.org/abs/2406.10721)——2D affordance
- **VLM output code/constraint**：[Eureka (reward design)](https://arxiv.org/abs/2310.12931)、[Code as Policies](https://arxiv.org/abs/2209.07753)、[ReKep]——code 是更 expressive 的 output modality
- **VLM output keypoints/marks**：[MOKA](https://arxiv.org/abs/2403.03174)、[PIVOT](https://arxiv.org/abs/2402.07872)、[CoPa](https://arxiv.org/abs/2403.08248)——ReKep 在此基础上加了 constraint optimization

### 11.3 TAMP 谱系

- **Classic TAMP**：[PDDLStream](https://github.com/mit-drll/pddlstream)、[Logic-Geometric Programming (Toussaint 2015)](https://www.ijcai.org/Proceedings/2015/0666)——symbolic planning + geometric solver
- **Sequence-of-Constraints MPC**：[Toussaint et al. 2022](https://ieeexplore.ieee.org/document/9982296)——ReKep 直接借鉴的 hierarchical optimization formulation
- **Learned feasibility**：[Driess et al.](https://arxiv.org/abs/2010.00101)、[Diffusion solver](https://arxiv.org/abs/2309.00966)——data-driven

ReKep 在 TAMP 谱系里相当于 **"symbolic precondition 用 VLM 写成 Python constraint"**，把 TAMP 从手工 symbolic predicate 推到 vision-grounded 自动生成。

---

## 12. 与一些底层直觉的连接

### 12.1 与 Differentiable Simulation 的关系

ReKep 的 forward model $h$ 是非 learned、non-differentiable 的（用 rigidity 假设）。如果换上可微 forward model（如 [Brax](https://github.com/google/brax)、[DiffSim](https://arxiv.org/abs/2106.01398)），整个 pipeline 可以 end-to-end 反向传播。但论文选择非 differentiable 是有理由的：实时性 + 不需要训练数据。这是工程 trade-off。

### 12.2 与 MPC 的关系

Eq.1 本质是 long-horizon MPC，stage timing $g_i$ 是 switching time。Decomposition 把它简化成 short-horizon MPC + waypoint tracking，是 [Toussaint SoC-MPC](https://ieeexplore.ieee.org/document/9982296) 的直接对应。

### 12.3 与 Program Synthesis 的关系

VLM 写 ReKep function 等价于 **program synthesis from natural language**。这跟 [Code as Policies](https://arxiv.org/abs/2209.07753) 思路同源，但 ReKep 写的不是直接调 API 的 code，而是 **constraint function**——一种 declarative program，让 solver 决定 imperative motion。这是从 imperative to declarative 的抽象升级。

### 12.4 与 Implicit Representation (NeRF/DINO descriptor fields) 的关系

Keypoint 是 explicit 3D coordinate，跟 [NDF](https://arxiv.org/abs/2112.05124)、[D³Fields](https://arxiv.org/abs/2309.16118) 那种 implicit dense descriptor field 互补。ReKep 选择 explicit 是为了让 VLM 能用 natural number index 引用（keypoint 0, 1, 2...），implicit field 没法用文字直接指。这是 representation 选择为了适配 VLM 输出而做的妥协。

### 12.5 与 "Llms as optimizers" 的关系

最近 [Large Language Models as Optimizers (OPRO)](https://arxiv.org/abs/2310.03720)、[FunSearch](https://www.nature.com/articles/s41586-023-06647-8) 都在用 LLM 做 optimization。ReKep 是这种思路在 manipulation 上的实例：LLM 写 objective function，外部 solver 求 argmin。

---

## 13. Limitations 深度讨论

### 13.1 Long-horizon 退化

Appendix A.14 的 "prepare breakfast tray" 10-stage task：自动 pipeline 完全失败。原因：
- Keypoint proposal：多个 object 同时出现，candidates 爆炸
- VLM：写 10 个 stage 的 temporally-dependent constraint 出错率高
- Tracker：10 stage 中多物体频繁 occlusion

最终作者只能 manually annotate keypoints + 每个 stage 重新 detect（不持续 track）。这是 ReKep 现在的 ceiling——long-horizon 仍需人工。后续工作 [SayCan](https://arxiv.org/abs/2204.01691)、[Inner Monologue](https://arxiv.org/abs/2207.05608) 的 hierarchical planner 可能可以整合。

### 13.2 Articulated object 缺失

论文不做 drawer、cabinet、scissors 这类 articulated object。作者在 A.11 提出扩展思路：把 joint 也表达成 keypoint relation，例如 prismatic joint = 某点只能沿某 line 移动，revolute joint = 某点只能沿某 circle 移动。但这需要 VLM 理解 articulation，参考 [A3-VLM](https://arxiv.org/abs/2406.07549)、[Kinematic-aware prompting](https://arxiv.org/abs/2311.02847) 的工作。

### 13.3 Task-space planning 的 kinematic 问题

Solver 优化 task space pose，不显式考虑 robot joint limit / kinematics。Reachability 用 IK 残差 proxy，但有些 pose 即使 IK 有解，motion 也很难（接近 singularity）。Joint-space planning 更优但更慢，作者留作未来工作。[cuRobo](https://arxiv.org/abs/2310.17274) 这类 GPU 加速 kinematics solver 是直接解决方案。

### 13.4 Rigidity forward model 在 contact-rich 场景失效

Pour tea 还行，但 peg-in-hole、tight insertion、推 box 这种 contact-rich 任务，rigidity 假设错误。需要 learned dynamics model 或 [MuJoCo MJX](https://mujoco.readthedocs.io/) 类 differentiable physics。论文 [Model-based Control with Sparse Neural Dynamics](https://arxiv.org/abs/2402.17070) 给了思路。

---

## 14. 个人 takeaways 和延伸联想

1. **Representation > Algorithm**：ReKep 的成功本质是找到了一个 VLM-friendly、solver-friendly、perception-friendly 的 representation sweet spot。这跟当年 [AlphaGo](https://www.nature.com/articles/nature16961) 用 policy+value network 而非纯 search 是同一种 representation-first 思路。

2. **Declarative vs Imperative for robot control**：传统 robot stack 是 imperative（写 trajectory、写 controller）。ReKep 走 declarative（写 constraint，让 solver 算 trajectory），更接近 [CLP](https://en.wikipedia.org/wiki/Constraint_logic_programming)、[model checking](https://en.wikipedia.org/wiki/Model_checking) 那种 CS 传统。LLM 写 declarative spec 比写 imperative code 容易得多——这与软件工程里 [spec-driven development](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/sdp04-bickford.pdf) 趋势一致。

3. **Multi-modal grounding 的范式**：ReKep 提供了一个 blueprint——VLM 输出不是直接 action 或直接 text，而是 **structured code that an external solver can consume**。这种 "VLM as spec writer, solver as executor" 范式可以推广到 code generation、circuit design、game level generation 等领域。

4. **Keypoint as universal interface**：Keypoint 同时是 perception output（DINOv2）、VLM reference（numerical mark）、optimization variable、human inspection handle。这种 "one representation multiple consumers" 是好的 system design 标志。

5. **Closed-loop 频率是 foundation model 时代的 critical resource**：ReKep 把 VLM 调用降到 task start 一次，剩下全靠 20Hz tracker + 10Hz solver。这种 "expensive VLM once + cheap loop many" 是 VLA (vision-language-action) 系统设计的核心 trade-off。对比 [RT-2](https://arxiv.org/abs/2307.15818) 每帧调 VLA 模型只能跑到 ~3 Hz。

6. **Sim-to-real 通过结构而非数据**：A.12 simulation 实验显示 zero-shot ReKep 在 unseen 上 72%，远超 100-demos 的 monolithic 14%。原因不是 ReKep 学到更好 representation，而是它 **不依赖分布假设**——只要 VLM 能写对 constraint + solver 能解，就 work。这暗示 foundation-model-driven manipulation 的 generalization 范式可能从 "more data" 转向 "better structure"。

7. **后续工作的方向**：
   - 用 [SpatialTracker](https://arxiv.org/abs/2403.23499) 替换 DINOv2 tracker → 直接打掉最大 error source
   - 用 [GPT-4o vision fine-tune](https://openai.com/index/gpt-4o/) 或 [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet) 替换 VLM → 提升 constraint quality
   - 用 [cuRobo](https://arxiv.org/abs/2310.17274) 替换 PyBullet IK → 解决 task-space kinematic 问题
   - 整合 [VLM-as-a-service](https://arxiv.org/abs/2407.08003) 做 long-horizon replanning → 解决 A.14 ceiling

---

## 15. 一句话总结

ReKep 把 manipulation 任务表达为 **"3D keypoints 之间的 Python 不等式约束"**，让 VLM 写约束、让 numerical solver 算动作、让 DINOv2 tracker 闭环更新——三层各取所长，在 zero-shot、in-the-wild、bimanual、reactive 任务上取得 44.3% 自动 / 68.6% 标注的成功率，远超 VoxPoser 的 10%。Limitation 集中在 tracker、long-horizon、articulated object，但 representation choice 本身非常 elegant，预计会激发一系列后续工作。

---

## References

- [ReKep 论文与项目主页](https://rekep-robot.github.io)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [GPT-4o technical report](https://arxiv.org/abs/2303.08774)
- [SAM (Segment Anything)](https://arxiv.org/abs/2304.02643)
- [Sequence-of-Constraints MPC (Toussaint et al.)](https://ieeexplore.ieee.org/document/9982296)
- [KPAM (Manuelli et al.)](https://arxiv.org/abs/1903.00606)
- [VoxPoser](https://arxiv.org/abs/2307.05973)
- [MOKA](https://arxiv.org/abs/2403.03174)
- [PIVOT](https://arxiv.org/abs/2402.07872)
- [D³Fields](https://arxiv.org/abs/2309.16118)
- [AnyGrasp](https://github.com/graspnet/anygrasp_sdk)
- [cuRobo](https://arxiv.org/abs/2310.17274)
- [SpatialTracker](https://arxiv.org/abs/2403.23499)
- [CoTracker3](https://arxiv.org/abs/2410.11831)
- [Code as Policies](https://arxiv.org/abs/2209.07753)
- [Eureka](https://arxiv.org/abs/2310.12931)
- [nvblox](https://arxiv.org/abs/2311.00626)
- [Toussaint Logic-Geometric Programming](https://www.ijcai.org/Proceedings/2015/0666)
- [Neural Descriptor Fields](https://arxiv.org/abs/2112.05124)
- [FoundationPose](https://arxiv.org/abs/2312.08344)
- [Tedrake Underactuated Robotics](https://underactuated.csail.mit.edu/)
- [SayCan](https://arxiv.org/abs/2204.01691)
- [RT-2](https://arxiv.org/abs/2307.15818)
