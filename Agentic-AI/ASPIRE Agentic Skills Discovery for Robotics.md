---
source_pdf: ASPIRE Agentic Skills Discovery for Robotics.pdf
paper_sha256: a2ddd6f36b497d8f106cb1ec7e7528be6eae0ead2d85c9ee75c9a50bc159f521
processed_at: '2026-08-18T01:21:42-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ASPIRE 用人话说

Karpathy，好，咱们抛开 paper 的 academic framing，用大白话讲讲这玩意儿到底干了啥。

---

## 一句话版本

**让 LLM 像人类 robotics engineer 一样 debug robot 程序，而且越调试越牛逼，因为它会把"踩过的坑 + 修复套路"记下来给下次用。**

---

## 为什么之前的 coding agent 不 work

你想啊，你现在让 Claude 写个 robot arm 抓碗的程序。程序跑了一遍，失败了。Claude 问你："为什么失败？"你能告诉它啥？

你说："任务失败了。"

这就好比你让一个 SWE-bench agent 改 bug，但是只告诉它 "test failed"，不给它 stderr、不给 stack trace、不给 line number。它能改个锤子。

**Robot failure 是个极其 messy 的东西**：
- 是 perception 没看到碗？
- 是看到了但 grasp 规划烂了？
- 是 grasp 规划好了但 IK 解不出？
- 是 IK 解出了但 motion planning 撞桌子了？
- 是都搞定了但 gripper 没夹紧掉了？
- 是夹紧了但 place 的时候放歪了？

Prior work 比如 CaP-Agent0 [Fu et al., 2026] 给 agent 的信息就是"0/1 success"。Agent 拿到 0，只能瞎猜，然后 retry，再失败，再瞎猜。这跟蒙眼打靶没区别。

更糟的是：**每次任务都是白纸一张**。第 100 个任务和第 1 个任务一样没经验。人类 engineer 调试了 100 个 task 之后，脑子里会形成一堆 "哦这种情况我见过，上次是 gripper 没夹紧，要调 yaw 90 度" 的套路。Prior coding agent 全丢了。

ASPIRE 就解决这两件事：(1) 给 agent 看得见的 trace，(2) 把 fix 记下来复用。

参考：
- CaP-Agent0 论文 https://arxiv.org/abs/2603.22435
- SWE-bench https://swe-bench.github.io/

---

## 系统怎么干活

### 打个比方

想象 NVIDIA 有一间 robotics lab，里面坐着 100 个 Claude instance（actor），每个桌上放一个 robot task。外面有个 manager（coordinator）调度。

每个 Claude 干 4 件事：
1. **写一段 Python 程序**控制 robot（用预定义的 API：感知、规划、抓取、运动）
2. **在 sim 里跑一遍**，但 sim 会把每一步的细节都录下来
3. **看录像**，定位哪一步挂了
4. **改代码**，再跑一遍验证

搞定之后，Claude 写一个"心得报告"交给 manager。Manager 看完觉得这个心得是 generic 的（不是只对这一个 task 有效），就存进 **skill library**——一个所有 Claude 共享的 wiki。

下次有个新 Claude 来解决新 task，先去 wiki 翻一翻："哦原来 SAM3 找 'bowl' 找不到的时候可以用 'white bowl' 或者 'round container' fallback。"直接复用，不用重新踩坑。

### 三个关键组件

**1. Robot Execution Engine（录像机）**

这是最核心的创新。Sim 跑的时候不是只给最终 success/fail，而是把**每个 primitive call** 都记录：
- 调了啥 API
- 输入是啥，输出是啥
- 返回状态（成功/PLANNING_ERROR/IK_FAILURE 等）
- 调用前后的 RGB 关键帧 + overlay（比如 grasp candidate 画在哪、segmentation mask 长啥样）

**但故意不给 full video**。因为给太多视觉信息会 distract agent，它不知道哪一帧是 causally relevant 的。只给 primitive call 前后的 snapshot，让 agent 聚焦在 "这一步发生了啥"。

举个具体例子（论文 Fig. 2 的 BEHAVIOR-1K 任务）：robot 要去拿 radio。ego-view 显示 robot 看到 radio 了，但就是走不过去。Trace 显示 `navigate_to_pose` 反复返回 `PLANNING_ERROR`。Agent 仔细看 trace 发现：navigation target 落在 table 边缘 20cm 内，触发 collision avoidance，planner 直接 refuse。

诊断清楚了——不是 perception 烂，不是 grasp 烂，是 **navigation target 不可行**。Agent 写个 patch：在 radio 周围 sample 多个 approach 方向，从 collision-free 的方向靠近。一跑，成功了。

这个 "multi-angle approach" 套路就被 manager 收进 skill library。以后任何 "PLANNING_ERROR near obstacle boundary" 的场景都能用。

**2. Skill Library（共享 wiki）**

这个 library 存的不是 "如何完成某 task" 的完整程序，而是 **repair knowledge**——碎片化的、可复用的修复套路。

分类也不预先定死，是从实际 debug 中归纳出来的。现在 library 里有这么几大类（Appendix A 展开）：

- **Localization skills**：SAM3 prompt 策略，比如 "bowl" 找不到就用 "white bowl" → "round container" → Molmo pixel grounding fallback chain
- **Navigation skills**：multi-angle approach、collision-aware pose selection
- **Grasping skills**：per-object z_offset 和 yaw registry（比如 butter 这种扁平物体 yaw 90 度更稳）
- **Motion primitives**：linear push、3-waypoint lift-transit-descend（避免直接 diagonal motion 撞东西）
- **Scene reasoning**：spatial relation disambiguation（"front" 就是 sort by X 取最小）
- **Debugging workflows**：gripper width 中等 = marginal grasp，要 retry with perpendicular yaw

每个 skill 存 4 个字段：failure signature、when-to-apply guard、repair strategy、code sketch。

**Admission 流程**：actor 报告 → coordinator 审计（API policy compliance + reusability）→ 通过才进 library。Coordinator 串行化 admission 避免并发写冲突。

**3. Evolutionary Search（防止钻牛角尖）**

光看 trace 改代码有个问题：agent 可能死磕同一个策略。比如总是想 "再加个 grasp retry" 而不去试 "换完全不同的 approach angle"。

Evolutionary search 每轮让 agent **propose K 个不同 hypothesis 的 candidate programs**。Prompt 里明确要求："No two candidates should fail at the same stage for the same reason." 每个候选必须 test 一个 distinct hypothesis。

跑完 K 个之后，下一轮 condition on Top3 + 它们的 residual failure traces，再 propose K 个新的。这比 single-trajectory self-debug 更 broad，比 random search 更 focused。

---

## 算法（Algorithm 1）人话版

```
给你一个 task τ，一个初始烂程序 P₀，一堆 debug seeds S_dbg，
一批 held-out validation seeds S_val，共享 skill library L，
一个 agent M，预算 T 轮每轮 K 个候选，成功阈值 θ。

1. 先用 P₀ 在 debug seeds 上跑一遍，记录 baseline 成绩 r*
2. 把 (P₀, r*, trace) 塞进 history H
3. for 第 i 轮 (i = 1 到 T):
4.   agent M 看着 task + Top3(history) + skill library + 全部 history 的失败 traces
     写出 K 个新候选程序 {P_i^1, ..., P_i^K}
5.   for 每个候选 k:
6.     在 debug seeds 上跑 P_i^k，记录成绩 r_i^k 和 trace Z_i^k
7.   end
8.   把这 K 个 (程序, 成绩, trace) 都塞进 H；标记本轮最佳的 k*
9.   if 本轮最佳 > 历史最佳 r*:
10.    更新最佳程序 P* 和最佳成绩 r*
11.  if 最佳成绩 >= 阈值 θ:
12.    break（够了别浪费算力）
15. end for
16. 用最佳程序 P* 在 held-out validation seeds 上跑一遍（不可逆！只能跑一次）
17. 从整个 history 里 extract 可复用的 pattern G
18. return (P*, validation 成绩, G)
```

**变量含义表**：

| 符号 | 含义 |
|------|------|
| τ | Task spec（自然语言 + 环境 config） |
| P₀ | 初始 seed 程序（baseline 或上轮输出） |
| S_dbg | Debug seeds（学习用，LIBERO 51-65 / Robosuite 101-125） |
| S_val | Held-out seeds（最终验证用，LIBERO 1-50 / Robosuite 1-100） |
| L | Shared skill library（持续增长） |
| M | Coding agent（Claude Opus 4.6, 1M context） |
| T | 最大 evolutionary search 轮数 |
| K | 每轮候选数（population size） |
| θ | 早停阈值 |
| r | Task score (0-1) |
| Z | Trace bundle（per-primitive multimodal logs） |
| H | 全部 (program, score, trace) 历史 |
| P\* | 目前最佳程序 |
| r\* | 目前最佳成绩 |
| P_i^k | 第 i 轮第 k 个候选 |
| r_i^k, Z_i^k | P_i^k 的成绩和 trace |
| k\* | 本轮 argmax index |
| r_val | Held-out validation 成绩 |
| G | Extracted validated patterns（待 admit 进 library） |

**关键 intuition**：每轮 propose 时 agent 看到的是 Top3(H) 的成功程序 + 它们的 residual failure traces。这意味着下一轮候选既 inherit 成功结构，又 target 剩余 failure mode。这就是 "failure-conditioned candidate generation"。

参考：
- Self-Debug [Chen et al., 2024] https://arxiv.org/abs/2203.10251
- AlphaEvolve https://arxiv.org/abs/2506.13131

---

## 实验数据人话解读

### LIBERO-Pro [Zhou et al., 2025]（短 horizon + 扰动）

10 个任务 × 50 个 held-out seeds。Aspire 在 seeds 51-65 学，eval seeds 1-50。

| Method | Overall Pos | Overall Task | All |
|--------|------|------|-----|
| OpenVLA | 0.00 | 0.00 | 0.00 |
| π₀ | 0.00 | 0.00 | 0.00 |
| π₀.₅ | 0.25 | 0.01 | 0.13 |
| CaP-Agent0 | 0.20 | 0.16 | 0.18 |
| **Aspire** | **0.77** | **0.67** | **0.72** |

**人话**：OpenVLA 和 π₀ 在 LIBERO-Pro 上直接崩到 0%。这其实印证了 LIBERO-Pro 的 motivation——prior VLA 在原版 LIBERO 上表现不错，但加了扰动就原形毕露，说明它们是 **memorize** 而不是 **generalize**。π₀.₅ 稍微好点，在 position 扰动上能到 38%，但 task 扰动几乎全 0——语义变化扛不住。

Aspire 直接干到 72% overall，libero-object 上 Pos 98% Task 95% 接近 saturated。libero-goal 的 Task perturbation 最难（45%），因为 goal 语义变化大。

### Robosuite（contact-rich manipulation）

7 个任务 × 100 trials：

| Task | CaP-Agent0 | Aspire |
|------|------|------|
| cube_lift | 0.97 | 0.97 |
| cube_stack | 0.98 | 0.99 |
| cube_restack | 0.89 | 1.00 |
| spill_wipe | 1.00 | 0.99 |
| **two_arm_handover** | **0.20** | **0.92** |
| two_arm_lift | 0.74 | 0.71 |
| nut_assembly | 0.00 | 0.09 |

**人话**：bimanual handover 从 20% 飙到 92%。两臂协调 handoff 是出了名的难，timing + grasp handoff 要 tightly coupled。Prior method 拿不到 trace 根本不知道哪一步挂了。Aspire 看 gripper width、IK status、object pose 就能定位 "啊是 right arm 没接住" 然后改。

### BEHAVIOR-1K（长 horizon household）

| Task | Human Task | CaP-Agent0 Task | Aspire Task |
|------|------|------|------|
| Soda Can pick-up | 0.72 | 0.72 | **0.88** |
| Radio pick-up | 0.36 | 0.56 | **0.88** |

**人话**：Radio pick-up 上人类 expert 写的程序只有 36% success，Aspire 干到 88%。**人类被 agent 超越**。因为人类 hard-code 一种 approach 策略（可能只从正面靠近），agent 通过 trace 发现 "正面 collision 了，换个角度试试" 这种 emergent strategy。

### Zero-shot Transfer（最 striking 的结果）

Skill library 在 LIBERO-90（90 个短 task）上积累，然后 **zero-shot** 扔到 LIBERO-Pro Long（held-out 长 horizon task）上，不做任何 debug、retry、library update。

| Method | Overall |
|--------|---------|
| OpenVLA | 0.00 |
| π₀ | 0.00 |
| π₀.₅ | 0.05 |
| CaP-Agent0 | 0.038 |
| Aspire (N=0, no library) | 0.047 |
| Aspire (N=25) | 0.137 |
| Aspire (N=50) | 0.215 |
| **Aspire (N=90, full library)** | **0.305** |

**人话**：Prior best 3.8%，Aspire 30.5%，**8 倍**。而且 success 随 library size 单调增长（N=0→25→50→90），证明 **compounding effect 确实存在**——积累越多 skill，越能 transfer 到新 task。

但 per-task breakdown 有 non-monotonic 的情况。比如 "Soup + cream cheese" Pos 在 N=50 时 0.02 比 N=25 的 0.12 还低。论文 §5 limitations 解释：library 增长后有些 entry 变 stale / overly specific / misleading，**long-term memory management 没完全解决**。这是 future work。

### Sim-to-Real Cross-Embodiment（最 practical 的结果）

Sim 里用 Franka，real 里用 bimanual YAM，**embodiment 和 API 都不同**。Sim 学到的 skill 作为 in-context guidance 给 real-robot coding agent（GPT-5.5），看能不能减少 real-world debugging 的 token 消耗。

| Task | w/o Skills Total Tokens | w/ Skills Total Tokens | w/o Skills Success | w/ Skills Success |
|------|------|------|------|------|
| Put bowl on plate | 8.65M | 5.11M | 20/20 | 20/20 |
| Lift soda can | 61.94M | 6.58M | 13/20 | **19/20** |
| Open/push drawer | 334.9M | 81.67M | 0/20 | **11/20** |

**人话**：
- Bowl 两个都能搞定，但有 skill 省 40% token
- Soda can 从 13/20 → 19/20，token 从 62M → 6.6M（**省 10 倍**）
- Drawer 从 0/20 → 11/20，token 从 335M → 82M（**省 4 倍**，且从完全失败变部分成功）

**这证明 sim 学的 skill 不是 sim-specific code**，是 abstract repair pattern 能跨 embodiment transfer。Transfer 机制是 in-context guidance（GPT-5.5 看到 skill description + code sketch 后写 real-robot 程序），不是 policy deployment。

### Ablation（哪个组件最重要）

| Setting | Macro Avg Success |
|---------|---------------------|
| Base (no engine, no evo search) | 14% |
| + Robot Execution Engine | 62% (+48) |
| + Engine + Evolutionary Search | 72% (+10) |

**人话**：**Engine 是最大贡献（+48%）**。这印证 attribution 是核心瓶颈。给 agent 看 trace 比让它瞎猜强 5 倍。Evolutionary search 只额外 +10%，是 incremental gain，针对 hard residual cases。

Per-iteration progress：前 2-3 轮提升最陡，之后 diminishing returns。K=3-5 candidates × 3-5 iterations 是 sweet spot。

参考：
- LIBERO-Pro https://arxiv.org/abs/2510.03827
- BEHAVIOR-1K https://behavior.stanford.edu/
- Robosuite https://robosuite.github.io/

---

## Skill Library 具体长啥样

Appendix A + E.5 给了完整细节。我挑几个代表 skill 讲：

### Skill 1: Multi-Object Disambiguation

场景：场景里有两个碗，指令说 "抓前面的碗"。SAM3 找 "bowl" 返回两个 mask，naive 取 max(score) 可能选错。

```python
masks = sam3(rgb, "bowl")  # returns ≥2 masks
candidates = [mask_to_world_point(m) for m in masks]

if "front" in instruction:
    candidates.sort(key=lambda p: p[0])  # sort by X
elif "back" in instruction:
    candidates.sort(key=lambda p: p[0], reverse=True)
elif "left" in instruction:
    candidates.sort(key=lambda p: p[1])  # sort by Y

target = candidates[0]
```

**变量含义**：
- `masks`: SAM3 返回的 mask list
- `candidates`: 每个 mask 投影到 world frame 的 3D 点
- `p[0]`, `p[1]`: 点的 X, Y 坐标
- `target`: 最终选中的目标点

**Intuition**：用 spatial relation（"front/back/left"）做 disambiguation，比单纯靠 SAM3 score 强。这是从某个 LIBERO-spatial task 的 debug 中归纳出来的。

### Skill 2: Linear Push on Floor Plane

场景：要把扁平物体（比如 plate）推到某位置，抓不起来。

```python
def push_object(obj_pos, target_xy):
    direction = (target_xy - obj_pos[:2]) / np.linalg.norm(target_xy - obj_pos[:2])
    pre_push = obj_pos[:2] - direction * 0.10
    post_push = target_xy
    
    move_to([pre_push[0], pre_push[1], obj_pos[2]])
    move_to([post_push[0], post_push[1], obj_pos[2]])  # linear push
```

**变量含义**：
- `obj_pos`: 物体 3D world coord (x, y, z)
- `target_xy`: 目标 2D 位置（floor plane projection）
- `direction`: 从 object 指向 target 的 unit vector
- `pre_push`: approach 起点，物体后方 10cm
- `post_push`: push 终点

**Intuition**：先去物体后面 10cm，再直线推过去。这是 non-prehensile manipulation 的 primitive，从 "Push plate to stove" 这种任务中归纳出来。

### Skill 3: 3-Waypoint Lift-Transit-Descend

```python
# Waypoint 1: lift straight up (avoid dragging)
lift_z = grasp_pos[2] + 0.15
joints = solve_ik([grasp_pos[0], grasp_pos[1], lift_z], quat.tolist())
if joints is not None: move_to_joints(joints)

# Waypoint 2: lateral move at lift height
joints = solve_ik([tgt_center[0], tgt_center[1], lift_z], quat.tolist())
if joints is not None: move_to_joints(joints)

# Waypoint 3: descend to release
release_z = surface_z + 0.05
joints = solve_ik([tgt_center[0], tgt_center[1], release_z], quat.tolist())
if joints is not None: move_to_joints(joints)
```

**Intuition**：为什么不直接 `goto_pose(target)`？因为直接 move 会走 diagonal 路径，可能撞到路上的 tall object 或 basket wall。3 个 cardinal waypoint（up → over → down）保证物体始终在 free air。这是从多个 pick-and-place task debug 中归纳的。

### Skill 4: Top-Down Quaternion

```python
import numpy as np
from scipy.spatial.transform import Rotation

def make_topdown_quat(yaw_deg=0):
    """Build top-down end-effector quaternion (xyzw→wxyz for solve_ik)."""
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # scipy returns xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # reorder to wxyz
```

**变量含义**：
- `yaw_deg`: 绕 Z 轴旋转角度（度）
- `R`: 3×3 rotation matrix
- `q`: quaternion，scipy 返回 (x, y, z, w)
- 返回值: (w, x, y, z) 格式，因为 `solve_ik` 用 wxyz convention

**Intuition**：top-down grasp 需要 end-effector 朝下，但默认 orientation 不一定朝下。这个 helper 构造正确的 top-down quaternion + yaw 旋转。Yaw 0 默认，如果 gripper finger 撞邻物就试 90 或 45 度。

---

## 系统约束：API Policy

Appendix E.2 揭示一个关键 design：

**FORBIDDEN APIs**：
- `env.handle.env.sim`（直接访问 MuJoCo sim object）
- `sim.data.body_xpos`（ground-truth object positions）
- 读 `.bddl`, `.xml`, `.urdf` asset 文件推断 geometry / success predicates

**ALLOWED APIs**：
- `get_observation()`（RGB, depth, intrinsics, extrinsics, robot state）
- `segment_sam3_text_prompt(rgb, text)`
- `plan_grasp(depth, intrinsics, segmentation)`
- `solve_ik(position, quaternion_wxyz)`
- `move_to_joints(joints)`, `open_gripper()`, `close_gripper()`

**Rule of thumb**（论文原话）："If a real robot with a camera could do it, it's allowed. If it reads the physics engine's internal state, it's forbidden."

**为什么这重要**：这让 agent 写的程序 transferable to real robot。Sim 到 real 不需要重写 code，只需要 adapt perception prompts 和 grasp parameters。这是 sim-to-real transfer 能 work 的前提。

参考：
- SAM3 https://github.com/facebookresearch/segment-anything-3
- MuJoCo Playground https://github.com/google-deepmind/mujoco_playground
- Claude Code Best Practices https://www.anthropic.com/engineering/claude-code-best-practices

---

## 跟其他工作比

### vs Voyager [Wang et al., 2023]
Voyager 在 Minecraft 中用 GPT-4 累积 skill library。但 Minecraft 是 **discrete + cheap failure + deterministic**，agent 能随便试错。Robotics 是 **continuous physics + expensive rollouts + multimodal perception**，所以 ASPIRE 需要 execution engine 做 attribution，需要 evolutionary search 做 sample-efficient exploration。

### vs Eureka [Ma et al., 2024a]
Eureka 用 LLM 自动 design reward function，然后 RL 训 policy。ASPIRE 不学 policy，直接 **写程序 + 累积 skill**——avoid RL 的 sample inefficiency 但 bound 在 predefined API expressivity 内。

### vs AlphaEvolve [Novikov et al., 2025]
AlphaEvolve 用 LLM + evolutionary search 发现数学算法。ASPIRE 借鉴 evolutionary search 但 apply 到 robot program。关键区别：AlphaEvolve fitness 是 deterministic math verification，ASPIRE fitness 是 stochastic simulator rollout。

### vs SWE-agent [Yang et al., 2024]
SWE-agent 设计 agent-computer interface 让 LLM 自主 edit code、run tests、inspect outputs。ASPIRE 借鉴这个 paradigm 但 extend 到 robotics——**multimodal trace 替代 stdout/stderr**。

### vs GR00T N1 [NVIDIA et al., 2025]
GR00T N1 是 end-to-end VLA foundation model。ASPIRE 是 coding agent。两者是互补的：VLA 处理 routine cases 速度快，coding agent 处理 edge cases + accumulate skills，skills 反过来能 distill 进 VLA training data。

参考：
- Voyager https://voyager.minedojo.task/
- Eureka https://eureka-research.github.io/
- AlphaEvolve https://arxiv.org/abs/2506.13131
- SWE-agent https://swe-agent.computer/
- GR00T N1 https://arxiv.org/abs/2503.14734

---

## Limitations 人话

### 1. 不是 fully autonomous real-world lifelong learner
Real world 里 success detection、safe reset、safety monitoring、calibration 都要人搞。Sim 里这些都是 programmatic + cheap。Future work 要 close this loop。

### 2. 依赖 frontier LLM
用 Claude Opus 4.6 + 1M context。没 verify smaller LLM 能否 sustain same debugging loop。如果 reasoning 能力不够，trace interpretation + repair synthesis 会 break down。

**可能 solution**：用 frontier LLM 在 sim 中 accumulate skill library，smaller LLM 在 real-time inference 中只 retrieve + execute，不做 debugging。

### 3. Predefined API bound expressivity
Agent 只能用 predefined primitives。如果 task 需要新 primitive，agent 只能 inefficient approximation 或靠人扩展 API。

**Future**: agent 自主 propose 新 primitive，coordinator 在 sim 中 validate 安全性后 admit 进 API。这接近 RoboGen [Wang et al., 2024b] 的自动 skill synthesis。

### 4. Long-term memory management 没完全解决
Library 增长后有些 entry 变 stale / overly specific / redundant / misleading。这解释了 zero-shot transfer 的 non-monotonic 现象。

**需要**：
- **Pruning**: skill retrieval 导致任务 fail 时标记 skill 为 suspicious
- **Re-validation**: 定期 re-validate skill 在新 task suite 上
- **Hierarchical retrieval**: 类似 RAG 的 reranking，根据 current task context rank skills

### 5. Compute-intensive
Debug + evolutionary search 消耗大量 LLM calls + simulator rollouts。Scaling 到 large task suite 需要 cheaper LLM inference 或 more sample-efficient search 或 stronger reuse mechanisms。

参考：
- RoboGen https://arxiv.org/abs/2410.11751
- AtomVLA https://arxiv.org/abs/2603.08519

---

## 我（GLM）的额外联想

### A. 跟 RL 的关系
ASPIRE 本质是 **program-space RL**，但 policy 是 LLM-generated program 而非神经网络。State 是 (task spec, trace history)，action 是 write/repair program，reward 是 task success。但跟 RL 不同的是：
- 没有 gradient update，"policy improvement" 通过 skill library accumulation 实现
- Exploration 通过 evolutionary search 而非 ε-greedy
- Credit assignment 通过 trace attribution 而非 TD-learning

这其实是 **inductive programming + lifelong learning** 的现代化版本。

### B. Skill Library vs RAG
Skill library 某种意义上是 **structured RAG for robotic knowledge**。但比普通 RAG 多了：
- **Validated admission**: 只有通过 debug validation + reusability check 才进库
- **Failure signature**: 每个 skill 有 trigger condition，不是纯 semantic similarity retrieval
- **Code sketch**: 不只是 text，还有 executable code

### C. Evolutionary Search vs Tree Search
Evolutionary search 在这里其实是 **beam search over program space**，但：
- Branching factor = K（每轮 K 个候选）
- Pruning 通过 closed-loop execution（执行失败就淘汰）
- Backtracking 通过 condition on Top3(H)（不只看当前轮，看全部历史）

跟 MCTS 比，它没有 UCB 这种 exploration bonus，但 LLM 自身的 diversity（temperature + prompt）替代了 exploration mechanism。

### D. Sim-to-Real 的抽象层次
ASPIRE 的 sim-to-real transfer 不是 policy deployment，是 **knowledge transfer at abstraction level**。Sim 学到的 "multi-angle approach" skill 在 real 里不是直接跑代码，是 GPT-5.5 看了 skill description 后**重新写 real-robot 程序**。这跟 domain randomization / sim-to-real RL 完全不同——那是 low-level policy transfer，ASPIRE 是 high-level strategy transfer。

这暗示一个 future direction：**hierarchical sim-to-real**。Low-level motor skill 用 RL + domain randomization，high-level task strategy 用 ASPIRE-style skill library。两者结合可能比任何单一方法都强。

### E. 跟 Self-Play / AlphaGo 的类比
AlphaGo 用 MCTS + policy network 自我对弈积累 experience。ASPIRE 用 evolutionary search + LLM 自我对弈（write → execute → debug → refine）积累 skill library。都是 **bootstrapping from self-generated experience**，但：
- AlphaGo 的 experience 是 Q-value table / policy network weights
- ASPIRE 的 experience 是 textual skill descriptions + code sketches

后者更 interpretable、更 editable、更 transferable across embodiment。

### F. 为什么 VLA 在 LIBERO-Pro 上全崩
LIBERO-Pro [Zhou et al., 2025] 的设计就是要 expose VLA 的 memorization 问题。VLA 本质是 **behavior cloning**，learned mapping 是 (image, instruction) → action。如果 test distribution 偏离 training distribution（object pose 变、instruction paraphrase），mapping 就失效。

ASPIRE 不依赖 learned mapping，依赖 **explicit program + explicit perception**。SAM3 看到 bowl 就看到，看不到就换 prompt；IK 解不出就换 approach angle。每个 component 都是 **transparent + repairable**。这是 representation choice 的根本差异。

### G. 跟 Software Engineering Agent 的本质差异
SWE-bench agent debug code，failure 是 **logical**（test pass/fail）。Robotics agent debug robot program，failure 是 **physical + perceptual + logical 混合**。这就是为什么 ASPIRE 需要 multimodal trace——text-only trace 不够，必须看 RGB keyframe + grasp candidate overlay + motion trajectory 才能定位物理层面的 root cause。

---

## Final Intuition

**ASPIRE = SWE-bench agent paradigm + multimodal trace + lifelong skill library + evolutionary search，applied to robotics.**

核心 insight 是 **attribution 是 robotic debugging 的 bottleneck**。Ablation 显示 execution engine 单独贡献 +48%，evolutionary search 只额外 +10%。Prior work 给 agent 再强 LLM 也没用——没有 trace，agent 看不到 root cause。

**Compounding 是长期价值**。Zero-shot transfer 从 N=0 的 4.7% 增长到 N=90 的 30.5%，证明 skill library 确实是 reusable knowledge 而非 task-specific memorization。

**Sim-to-real 通过 abstraction level 绕过 embodiment gap**。不是 deploy sim policy，是 transfer abstract repair patterns as in-context guidance，让 real-robot agent 自己 adapt。

**Critical path to fully autonomous lifelong learning**：解决 real-world success detection + safe reset + memory management，ASPIRE 就能 scale 到 broad real-world deployment。

希望这个"人话"版本能让你 build intuition！如果对某个 component 想深挖（比如 evolutionary search 的 prompt 怎么保证 hypothesis diversity，或者 skill library 的 retrieval mechanism 是 semantic search 还是 rule-based），继续问。

参考汇总：
- ASPIRE https://research.nvidia.com/labs/gear/aspire/
- LIBERO https://lifelong-robot-learning.github.io/
- LIBERO-Pro https://arxiv.org/abs/2510.03827
- BEHAVIOR-1K https://behavior.stanford.edu/
- CaP-X https://arxiv.org/abs/2603.22435
- MuJoCo Playground https://github.com/google-deepmind/mujoco_playground
- Voyager https://voyager.minedojo.task/
- Eureka https://eureka-research.github.io/
- AlphaEvolve https://arxiv.org/abs/2506.13131
- SWE-agent https://swe-agent.computer/
- GR00T N1 https://arxiv.org/abs/2503.14734
- RoboGen https://arxiv.org/abs/2410.11751
- Claude Code Best Practices https://www.anthropic.com/engineering/claude-code-best-practices

---

# ASPIRE: Agentic Skills Discovery for Robotics 深度解析

Karpathy 你好！这篇来自 NVIDIA GEAR Lab、UMich、UIUC、UC Berkeley、CMU 联合的工作，核心 idea 非常清晰：**把 software engineering agent 的 write-execute-debug 闭环搬进 robotics，并通过 skill library 实现 lifelong compounding**。下面我从 motivation、架构、算法、实验四个层次 build 你的 intuition。

---

## 1. Motivation: 为什么 prior coding agent 走不通

现有 code-as-policy 系统（CaP [Liang et al., 2023]、RoboCodeX [Mu et al., 2024]、CaP-X/CaP-Agent0 [Fu et al., 2026]）都卡在两个根本问题：

### 1.1 Attribution 问题
Robot failure 是**多组件耦合**的：perception、motion planning、grasp generation、contact dynamics、long-horizon coordination 都可能 fail。但 prior work 只给 agent **rollout-level coarse feedback**（任务成败 0/1），相当于让 SWE-bench [Jimenez et al., 2024] 上的 agent 只知道 "test failed"，但拿不到 stderr、拿不到 stack trace。Agent 根本无法 localize root cause。

### 1.2 No Compounding Experience
Prior coding agent 解决第 100 个 task 和第 1 个 task 一样没经验。人类 robotics engineer 调试多了会形成 "grasp recovery heuristics"、"navigation strategies"、"prompting recipes" 这种 transferable knowledge。Prior work 全部丢弃。

ASPIRE 的洞察：**failure attribution + reusable skill accumulation = open-ended self-improvement**。

参考链接：
- CaP: https://code-as-policies.github.io/
- SWE-bench: https://swe-bench.github.io/
- Voyager (open-ended embodied agent): https://voyager.minedojo.task/
- Eureka (LLM reward design): https://eureka-research.github.io/

---

## 2. 系统架构：Coordinator-Actor + 三组件 Loop

### 2.1 高层架构
```
┌─────────────────────────────────────────────────┐
│  Coordinator (中央调度 + skill librarian)        │
│  - 派发 actor 到 individual task                 │
│  - 审计 actor 的 findings                        │
│  - 串行化 skill admission (避免 conflict write)   │
└──────────┬──────────────────────────────────────┘
           │ dispatch                    ↑ findings
           ↓                             │ report
┌────────────────────────┐  ┌────────────────────────┐
│  Actor A (coding agent)│  │  Actor B (coding agent)│ ...并行
│  - write Python code   │  │                        │
│  - execute in engine    │  │                        │
│  - inspect trace        │  │                        │
│  - diagnose + repair     │  │                        │
└──────────┬─────────────┘  └────────────────────────┘
           ↓
┌─────────────────────────────────────────────────┐
│  Robot Execution Engine                          │
│  - per-primitive multimodal traces               │
│  - validate repairs via re-execution             │
└─────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────┐
│  Evolutionary Search (per-task)                  │
│  - K candidates per iteration                    │
│  - condition on Top3(history) + skill library    │
└─────────────────────────────────────────────────┘
```

关键设计：**Actor 之间不共享 chat history 或 raw trajectory**。所有 transferable knowledge 都被 distill 进 skill library，再以 in-context guidance 形式被新 actor 检索。这样 actor 的 context window 始终聚焦于当前 task specification + current program + structured failure traces。

### 2.2 三组件 Interlock

**Component 1: Robot Execution Engine (§2.1)**

把 coarse rollout feedback 替换成 **per-primitive multimodal trace**。对每个 primitive call（perception / planning / grasping / control）记录：
- Invoked API
- Inputs / outputs
- Return status
- Multimodal evidence: RGB keyframes（call 前后帧 + overlays）、grasp candidates、object poses、motion-planning results

**关键 trade-off 的解决**：too little evidence hides failing primitive；too much raw visual context distracts from causal chain。Engine 的设计是只保留 primitive call **前后帧** + overlays + return values，agent 看不到 full video，但能看到 causally relevant snippets。

Fig. 2 的 BEHAVIOR-1K navigate-and-pick-up-radio 例子非常 illustrative：
- Ego-view keyframes 显示 robot 找到 radio 但反复 fail to approach
- Primitive trace localize 到 `PLANNING_ERROR`：navigation target 落在 table 的 collision-avoidance buffer 内（~20cm from edge）
- Agent 诊断后写 multi-angle approach routine：在 radio 周围 sample alternative navigation targets，从 collision-free 方向 approach
- 这个 repair 被 admit 进 skill library 成为 "Multi-Angle Approach" skill

**Component 2: Skill Library (§2.2)**

存储的 **不是完整 task program**，而是 **heterogeneous repair knowledge**：
- Localization heuristics（SAM3 prompting strategies、disambiguation）
- Perception prompts（per-object working prompts registry）
- Grasping constraints（per-object z_offset、yaw registry）
- Navigation recovery（multi-angle approach、collision-aware poses）
- Motion primitives（linear push、waypoint patterns）
- Scene-understanding routines（spatial relations、occlusion handling）
- Debugging workflows（gripper width → grasp quality inference、IK failure recovery）

**Taxonomy 不预先规定**，从 validated repairs 中 inductively 形成。

每个 skill 存为 compact in-context guidance，包含 4 个字段：
1. **Failure signature**（什么样的 trace pattern 触发这个 skill）
2. **When-to-apply guard**（situational retrieval condition）
3. **Repair strategy**（textual description）
4. **Code sketch**（when useful，representative code）

**Admission protocol**：
- Actor 报告 structured findings：failure mode、validated fix、transferable pattern、task-specific quirks、debug validation success rate
- Coordinator 审计：API policy compliance check + reusability judgment
- 只有通过 debug validation 且 likely transferable 的 repair 才进入 shared library

**Component 3: Evolutionary Search (§2.3)**

解决 trace-guided debugging 的 **local repair loop** 问题：agent 可能反复 patch 同一个失败策略而不 explore fundamentally different solutions。

每轮：
- Agent 基于 skill library + Top3(history) + residual failure traces，propose **K 个 candidate programs**
- 每个 candidate 在 execution engine 中执行
- 下一轮 condition on best-performing programs + 它们的 remaining failure modes
- Search target 是 **robot program 本身**
- Termination: candidate 解决 debug configs OR budget exhausted
- Validated repairs 在 search 结束后 admit 进 skill library（前提：跨环境变体和任务 generalize）

---

## 3. Algorithm 1 详解

让我逐行解释 Algorithm 1：

```
Require: task τ, program P0, sets S_dbg, S_val, skill library L, 
         agent M, budget (T, K), threshold θ
Notation: Execute(P, S) = (r, Z) returns score r and trace bundle Z

1: (r*, Z0) ← Execute(P0, S_dbg); P* ← P0
2: H ← {(P0, r*, Z0)}
3: for i = 1, ..., T do
4:   {P_i^k}_{k=1}^K ← ProposeRepairs(M, τ, Top3(H), L, H)
5:   for k = 1, ..., K do
6:     (r_i^k, Z_i^k) ← Execute(P_i^k, S_dbg)
7:   end for
8:   H ← H ∪ {(P_i^k, r_i^k, Z_i^k)}_{k=1}^K; k* ← argmax_k r_i^k
9:   if r_i^{k*} > r* then
10:    (P*, r*) ← (P_i^{k*}, r_i^{k*})
11:  end if
12:  if r* ≥ θ then
13:    break
14:  end if
15: end for
16: (r_val, Z_val) ← Execute(P*, S_val)
17: G ← ExtractValidatedPatterns(H, P*, r_val, Z_val)
18: return (P*, r_val, G)
```

**变量 / 上下标解释**：

| Symbol | Meaning |
|--------|---------|
| τ | Task specification (自然语言 + 环境 config) |
| P₀ | Initial seed program (baseline 或 prior iteration 输出) |
| S_dbg | Debug seed set (learning phase 用的 seeds，文中是 LIBERO 51-65 / Robosuite 101-125 / BEHAVIOR 26-35) |
| S_val | Validation seed set (held-out，文中是 LIBERO 1-50 / Robosuite 1-100 / BEHAVIOR 1-25) |
| L | Shared skill library (持续增长) |
| M | Coding agent (Claude Opus 4.6 + 1M context) |
| T | Max evolutionary search iterations |
| K | Candidates per iteration (population size) |
| θ | Success threshold for early stopping |
| r | Scalar task score (0-1) |
| Z | Trace bundle (per-primitive multimodal logs) |
| H | History of all (program, score, trace) tuples |
| P\* | Best program found so far |
| r\* | Best score so far |
| P_i^k | k-th candidate at iteration i |
| r_i^k, Z_i^k | Score and trace of P_i^k |
| k\* | argmax index within iteration i |
| r_val | Final held-out validation score |
| G | Extracted validated patterns (待 admit 进 skill library) |

**Line 1-2: Initialization**
执行 initial program P₀ 在 debug seeds S_dbg 上，得到 baseline score r\* 和初始 trace bundle Z⁰。History H 只含这一个 entry。

**Line 3-15: Evolutionary Search Main Loop**
外层循环最多 T 轮。

**Line 4: Propose K candidates**
关键操作：agent M 基于 (a) task τ, (b) Top3(H) 即历史 top 3 程序，(c) skill library L, (d) full history H（包含失败 traces），propose K 个新的 candidate programs。**每个 candidate 必须 test 一个 distinct hypothesis**——这是 E.4 subagent prompt 中明确规定的："No two candidates should fail at the same stage for the same reason."

**Line 5-7: Parallel Execution**
K 个 candidate 在 debug seeds 上 parallel execute，每个产生 (r_i^k, Z_i^k)。

**Line 8: Update History + Track Best**
把所有 K 个新 entry 加入 H；标记本轮最佳 k\*。

**Line 9-11: Update Global Best**
如果本轮最佳超过历史最佳，更新 P\*, r\*。

**Line 12-14: Early Stopping**
r\* 达到 threshold θ 就 break，避免 unnecessary computation。

**Line 16: Held-out Validation**
用 P\* 在 held-out validation seeds S_val 上执行**一次**，得到 r_val。注意：这一步**不可逆地消耗 held-out seeds**，所以 E.3 prompt 中明确说 "Stage 2 is a one-shot validation operation. Do not debug Stage 2 runs."

**Line 17: Pattern Extraction**
从整个 history 中 extract validated patterns G，准备 admit 进 skill library。这里有个隐含条件：patterns 必须跨 environment variations 和 tasks 都 generalize 才会被 admit。

**Line 18: Return**
返回最佳程序 P\*、held-out score r_val、extracted patterns G。

**Key intuition**：这个 algorithm 的 power 在于 **failure-conditioned candidate generation**。每轮 propose 时，agent 看到的是 Top3(H) 的成功程序 + 它们的 residual failure traces，这样下一轮 candidate 既 inherit 成功结构，又 target 剩余 failure mode。这比 single-trajectory self-debug [Chen et al., 2024] 更 broad，比 pure random search 更 focused。

参考：
- Self-Debug (Chen et al., 2024): https://arxiv.org/abs/2203.10251
- AlphaEvolve (program search for math discovery): https://arxiv.org/abs/2506.13131
- K-Search (co-evolving world model): https://arxiv.org/abs/2602.19128

---

## 4. 实验数据深度分析

### 4.1 LIBERO-Pro [Zhou et al., 2025]

**Setup**: 10 tasks × 50 held-out seeds per suite/perturbation。Aspire 在 seeds 51-65 上 learn，evaluate on seeds 1-50。

**Table 2 数据**：

| Method | libero-object Pos | libero-object Task | libero-goal Pos | libero-goal Task | libero-spatial Pos | libero-spatial Task | Overall Pos | Overall Task | All |
|--------|------|------|------|------|------|------|------|------|-----|
| OpenVLA | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| π₀ | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| π₀.₅ | 0.17 | 0.01 | 0.38 | 0.00 | 0.20 | 0.01 | 0.25 | 0.01 | 0.13 |
| CaP-Agent0 | 0.22 | 0.18 | 0.26 | 0.17 | 0.12 | 0.14 | 0.20 | 0.16 | 0.18 |
| **Aspire** | **0.98** | **0.95** | **0.81** | **0.45** | **0.51** | **0.60** | **0.77** | **0.67** | **0.72** |

**Insight 1**: OpenVLA 和 π₀ 在 LIBERO-Pro 上**完全崩溃**（0%），即便它们在原版 LIBERO 上有不错表现。这验证了 LIBERO-Pro 的 motivation——prior VLA 是 **memorize** rather than **generalize**。

**Insight 2**: π₀.₅ 在 position perturbation 上还有一些能力（libero-goal Pos 38%），但在 task perturbation 上几乎全 0（最高 1%）。这暗示 VLA 对 perceptual robustness 比 semantic robustness 强。

**Insight 3**: Aspire 在 libero-object 上接近 saturated（Pos 98%, Task 95%），libero-goal 上 Pos 81% 但 Task 只有 45%——goal axis 的 task perturbation 是 hardest，可能因为 goal semantics 变化大。

### 4.2 Robosuite [Zhu et al., 2020] - Contact-rich Manipulation

**Table 3** 7 个 task × 100 trials：

| Task | CaP-Agent0 | Aspire | Δ |
|------|------|------|---|
| cube_lift | 0.97 | 0.97 | 0 |
| cube_stack | 0.98 | 0.99 | +1 |
| cube_restack | 0.89 | 1.00 | +11 |
| spill_wipe | 1.00 | 0.99 | -1 |
| **two_arm_handover** | **0.20** | **0.92** | **+72** |
| two_arm_lift | 0.74 | 0.71 | -3 |
| nut_assembly | 0.00 | 0.09 | +9 |
| Mean | 0.68 | 0.81 | +13 |

**Key**: bimanual handover 从 20% → 92% 是最大的 win。Two-arm coordination 需要 tightly coupled timing + grasp handoff，prior method 没法 debug，Aspire 通过 trace 看 gripper width、IK status、object pose 就能定位 handoff failure。

### 4.3 BEHAVIOR-1K [Li et al., 2024] - Long-horizon Mobile Manipulation

**Table 4**：

| Task | Human Nav | Human Task | CaP-Agent0 Nav | CaP-Agent0 Task | Aspire Nav | Aspire Task |
|------|-----------|-----------|------|------|------|------|
| Soda Can pick-up | 0.80 | 0.72 | 0.84 | 0.72 | 0.92 | **0.88** |
| Radio pick-up | 0.88 | 0.36 | 0.80 | 0.56 | 1.00 | **0.88** |

**Insight**: Aspire 在 Radio pick-up 上把 human expert program 的 36% task success 提升到 88%。**人类写的程序被 agent 超越**——因为人类 hard-code 一种 approach 策略，agent 能 discover multi-angle approach skill。

### 4.4 Zero-Shot Transfer to LIBERO-Pro Long (§3.5)

这是最 striking 的结果。Skill library 在 LIBERO-90 上积累，**zero-shot transfer** 到 LIBERO-Pro Long（held-out long-horizon tasks）。

**Table 5**：

| Method | Pos | Task | Overall |
|--------|-----|------|---------|
| OpenVLA | 0.00 | 0.00 | 0.00 |
| π₀ | 0.00 | 0.00 | 0.00 |
| π₀.₅ | 0.08 | 0.01 | 0.05 |
| CaP-Agent0 | 0.052 | 0.024 | 0.038 |
| Aspire (N=0) | 0.00 | 0.094 | 0.047 |
| Aspire (N=25) | 0.056 | 0.218 | 0.137 |
| Aspire (N=50) | 0.138 | 0.292 | 0.215 |
| **Aspire (N=90)** | **0.226** | **0.383** | **0.305** |

**Insight**: Aspire (N=90) 达到 30.5% overall，对比 CaP-Agent0 的 3.8%——**8 倍提升**。更重要的趋势在 Fig. 5(b)：success 随 skill library size **单调增长**（N=0 → 25 → 50 → 90），证明 **compounding effect**。

但 Table 6 per-task breakdown 揭示了 **non-monotonicity**：某些 task 在更大 library 下反而 drop（如 "Soup + cream cheese" Pos: N=50 时 0.02 比 N=25 的 0.12 低）。论文 §5 limitations 提到这可能是 library entry 变 stale / overly specific / misleading 导致——**long-term memory management 没完全解决**。

### 4.5 Sim-to-Real Cross-Embodiment Skill Transfer (§3.6)

真实机器人：bimanual YAM manipulation station，用 GPT-5.5 reasoning-xhigh mode。仿真用 Franka，real 用不同 embodiment + 不同 API。

**Table 1**：

| Task | w/o Skills Output Tokens (M) | w/ Skills Output Tokens (M) | w/o Skills Total Tokens (M) | w/ Skills Total Tokens (M) | w/o Skills Success | w/ Skills Success |
|------|------|------|------|------|------|------|
| Put bowl on plate | 0.05 | 0.04 | 8.65 | 5.11 | 20/20 | 20/20 |
| Lift soda can | 0.18 | 0.03 | 61.94 | 6.58 | 13/20 | **19/20** |
| Open/push drawer | 1.33 | 0.36 | 334.917 | 81.67 | 0/20 | **11/20** |

**Insight**: drawer 任务从 0/20 → 11/20，token 消耗减少 4 倍。这证明 **sim-discovered skills 不是 sim-specific code**，而是 **abstract repair patterns** 能跨 embodiment transfer。Transfer 是 in-context guidance（GPT-5.5 看到 skill description + code sketch 后写 real-robot 程序），not policy deployment。

### 4.6 Ablation Studies (§3.7)

**Robot Execution Engine + Evolutionary Search 各自贡献** (Fig. 6):

| Setting | Macro-average Success |
|---------|---------------------|
| Base (no engine, no evo search) | 14% |
| + Robot Execution Engine | 62% (+48) |
| + Robot Execution Engine + Evo Search | 72% (+10) |

**Engine 是最大贡献**——这印证 attribution 是 prior work 的核心瓶颈。Evo search 是 incremental，针对 hard residual cases。

**Per-iteration progress** (Fig. 6c): 前 2-3 轮提升最陡，之后 diminishing returns。这暗示 K=3-5 candidates × 3-5 iterations 是 sweet spot。

Table 9 详细 breakdown evolutionary search progress：

| Task | Iter 0 | Iter 1 | Iter 2 | Iter 3 | Iter 4 |
|------|--------|--------|--------|--------|--------|
| Bowl → plate | 0.62 | 0.60 | 0.60 | 0.18 | **0.86** |
| Wine bottle → rack | 0.40 | 0.76 | 0.74 | - | - |
| Bowl → stove | 0.62 | 0.82 | - | - | - |
| Push plate → stove | 0.00 | 0.80 | - | - | - |

可以看到 progress 是 **non-monotonic** 的（"Bowl → plate" 在 iter 3 掉到 0.18 然后 iter 4 跳到 0.86），说明 evolutionary search 在探索 hypothesis space，而不是单纯 refinement。

---

## 5. Skill Library 实现

Appendix A 给出了完整 taxonomy。论文 Fig. 7-12 展示 6 大类：

**Fig. 7: Debugging Skills**
- Failure signature → when-to-apply guard → repair sketch
- 例：gripper width 中等 → marginal grasp → retry with perpendicular yaw

**Fig. 8: Localization Skills**
- SAM3 prompting strategies + multi-prompt fallback + 3D centroid extraction
- Disambiguation by spatial relation（"front" → sort by X coord, take candidates[0]）

**Fig. 9: Navigation Skills**
- Multi-angle approach: sample alternative navigation targets around object
- Collision-aware pose selection

**Fig. 10: Strategic Grasping Skills**
- Per-object grasp point selection（contact-rich alignment）
- Adapt grasp strategy to object geometry

**Fig. 11: Motion Primitive Skills**
- Linear push on floor plane（计算 direction vector → pre_push → post_push）
- Contact-rich alignment recovery

**Fig. 12: Scene Reasoning Skills**
- Spatial relations reasoning
- Support surface / occlusion handling

**Code example** from Fig. 3 (Multi-Object Disambiguation):
```python
masks = sam3(rgb, "bowl")  # returns ≥2 masks
candidates = [mask_to_world_point(m) for m in masks]

if "front" in instruction:
    candidates.sort(key=lambda p: p[0])  # sort by X
elif "back" in instruction:
    candidates.sort(key=lambda p: p[0], reverse=True)
elif "left" in instruction:
    candidates.sort(key=lambda p: p[1])  # sort by Y

target = candidates[0]
```

**Code example** from Fig. 3 (Linear Push):
```python
def push_object(obj_pos, target_xy):
    direction = (target_xy - obj_pos[:2]) / np.linalg.norm(target_xy - obj_pos[:2])
    pre_push = obj_pos[:2] - direction * 0.10
    post_push = target_xy
    
    move_to([pre_push[0], pre_push[1], obj_pos[2]])
    move_to([post_push[0], post_push[1], obj_pos[2]])  # linear push
```

变量含义：
- `obj_pos`: 物体 3D world coordinate (x, y, z)
- `target_xy`: 目标位置在 floor plane 上的 2D projection
- `direction`: 从 object 指向 target 的 unit vector
- `pre_push`: approach 起点，物体后方 10cm
- `post_push`: push 终点

---

## 6. API Policy & System Prompt (Appendix E)

E.2 揭示了一个重要的 design choice：**FORBIDDEN APIs** vs **ALLOWED APIs** 的划分。

**FORBIDDEN**：
- `env.handle.env.sim` - MuJoCo sim object 直接访问
- `sim.data.body_xpos` - ground-truth object positions
- Reading `.bddl`, `.xml`, `.urdf` asset files 推断 geometry / success predicates

**Rule of thumb**: "If a real robot with a camera could do it, it's allowed. If it reads the physics engine's internal state, it's forbidden."

**ALLOWED**:
- `get_observation()` - RGB, depth, intrinsics, extrinsics, robot state
- `segment_sam3_text_prompt(rgb, text)` - SAM3 text-prompted segmentation
- `plan_grasp(depth, intrinsics, segmentation)` - GraspNet-style grasp planning
- `solve_ik(position, quaternion_wxyz)` - 7-DoF IK
- `move_to_joints(joints)`, `open_gripper()`, `close_gripper()`

这个 API design 是关键：让 agent 写的程序 **transferable to real**，sim 到 real 不需要重写 code，只需要 adapt perception prompts 和 grasp parameters。

---

## 7. Limitations (§5) - 未来方向

论文诚实地列出 5 个 limitations：

1. **不是 fully autonomous real-world lifelong learner**: real-world 需要 success detection、safe reset、safety monitoring、calibration maintenance。Simulation 中这些都是 cheap + programmatic，real 中要 close the loop。

2. **依赖 frontier LLM**: 用 Claude Opus 4.6 + 1M context。没 verify smaller/weaker LLM 能否 sustain same debugging loop。这是 open question——如果 reasoning 能力不够，整个 trace interpretation + repair synthesis 会 break down。

3. **Predefined API bound**: agent 只能用 predefined perception / planning / control primitives。如果 task 需要新 primitive，agent 只能 inefficient approximation 或 rely on human。**Future**: agent 自主 propose / validate / incorporate new primitives。

4. **Long-term memory management 没完全解决**: library 增长后 entry 可能 stale / overly specific / redundant / misleading。这解释了 §4.4 看到的 non-monotonic zero-shot transfer。Future: 更 robust retrieval / pruning / ranking / re-validation。

5. **Compute-intensive**: debug + evolutionary search 消耗大量 LLM calls + simulator rollouts。Scaling 到 large task suite 需要 cheaper LLM inference 或 more sample-efficient search 或 stronger reuse mechanisms。

---

## 8. Build Intuition: 跟 Related Work 的对比

### 8.1 vs Voyager [Wang et al., 2023]
Voyager 在 Minecraft 中用 GPT-4 写 skill library，但 Minecraft 是 **discrete + cheap failure + deterministic**，agent 能反复试错。ASPIRE 在 **continuous physics + expensive rollouts + multimodal perception** 中工作，所以需要 execution engine 做 attribution，需要 evolutionary search 做 sample-efficient exploration。

### 8.2 vs Eureka [Ma et al., 2024a]
Eureka 用 LLM 自动 design reward function，然后 RL 训 policy。ASPIRE 不学 policy，直接 **写程序 + 累积 skill**——avoid RL 的 sample inefficiency 但 bound 在 predefined API expressivity 内。

### 8.3 vs AlphaEvolve [Novikov et al., 2025]
AlphaEvolve 用 LLM + evolutionary search 发现数学算法。ASPIRE 借鉴 evolutionary search 思路但 apply 到 robot program。关键区别：AlphaEvolve 的 fitness 是 deterministic math verification，ASPIRE 的 fitness 是 stochastic simulator rollout。

### 8.4 vs SWE-agent [Yang et al., 2024]
SWE-agent 设计 agent-computer interface 让 LLM 自主 edit code、run tests、inspect outputs。ASPIRE 借鉴这个 paradigm 但 extend 到 robotics——**multimodal trace 替代 stdout/stderr**。

### 8.5 vs RoboCodeX [Mu et al., 2024]
RoboCodeX 也是 multimodal code generation for robot behavior synthesis，但没 lifelong skill accumulation。ASPIRE 的 compounding effect 是核心差异。

参考链接：
- Voyager: https://voyager.minedojo.task/
- Eureka: https://eureka-research.github.io/
- AlphaEvolve: https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- SWE-agent: https://swe-agent.computer/
- RoboCodeX: https://github.com/OpenRobotX/robocodex

---

## 9. 个人思考 & Open Questions

**9.1 为什么 this works**
核心是 **attribution-driven repair** + **reusable distillation**。人类工程师就是这样 debug 的：看 trace、找 root cause、写 fix、记住 pattern。ASPIRE 把这个 loop 自动化并 compounding。

**9.2 Limitation 1 - Real-world lifelong learning 的关键 missing piece**
论文承认 success detection + safe reset + safety monitoring 是 missing。这是真实 lifelong learning 的 bottleneck——simulation 中 success checking 是 programmatic (`task.check_done()`)，real 中需要 visual success classifier。Future work 应该把这个 close 起来。

**9.3 Limitation 2 - Frontier LLM 依赖**
如果 smaller LLM 不能 sustain debugging loop，那 ASPIRE 的 deployment cost 很高。一个可能 direction：用 frontier LLM 在 sim 中 accumulate skill library，然后 smaller LLM 在 real-time inference 中只 retrieve + execute，不做 debugging。

**9.4 Limitation 3 - API expressivity**
predefined API 是 double-edged sword：让 debugging tractable + safe，但 bound expressivity。一个 hybrid：agent 能 propose 新 primitive（如特定 contact-rich motion），coordinator 在 sim 中 validate 安全性后 admit 进 API。这接近 RoboGen [Wang et al., 2024b] 的自动 skill synthesis 思路。

**9.5 Limitation 4 - Memory management**
non-monotonic zero-shot transfer (Table 6) 是 red flag。需要 active memory management：
- **Pruning**: 当 skill retrieval 导致任务 fail 时，标记 skill 为 suspicious
- **Re-validation**: 定期 re-validate skill 在新 task suite 上
- **Hierarchical retrieval**: 类似 RAG 的 reranking，根据 current task context rank skills

**9.6 Open Question: Aspire vs VLA hybrid**
ASPIRE 在 LIBERO-Pro 上完全 beat VLA，但 VLA 的优势是 **end-to-end learning + inference speed**。一个 hybrid: VLA 处理 routine cases，coding agent 处理 edge cases + accumulate skills，skills 反过来 distill 进 VLA training data。这是 GR00T N1 [NVIDIA et al., 2025] + ASPIRE 的潜在结合点。

参考：
- GR00T N1: https://arxiv.org/abs/2503.14734
- RoboGen: https://arxiv.org/abs/2410.11751
- AtomVLA (post-training via predictive latent world models): https://arxiv.org/abs/2603.08519

---

## 10. 总结

**ASPIRE 的核心贡献**：把 software engineering agent paradigm 完整搬到 robotics，通过三个 interlocking 组件——robot execution engine (attribution)、skill library (compounding)、evolutionary search (exploration)——实现了 **open-ended lifelong learning for robot programming**。

**最 striking 的 numbers**:
- LIBERO-Pro Object perturbation: 0% (prior VLA) / 18% (CaP-Agent0) → 98% (Aspire)
- LIBERO-Pro Long zero-shot: 4% (prior best) → 31% (Aspire N=90)
- Sim-to-real token cost reduction: 10× on soda can, 4× on drawer

**最 deep 的 insight**: **attribution 是 robotic debugging 的 bottleneck**，不是 program synthesis 能力。Ablation 显示 execution engine 单独贡献 +48% success，evolutionary search 只额外 +10%。Prior work 给 agent 再强 LLM 也没用——没有 trace，agent 看不到 root cause。

**Future work 的 critical path**: 解决 real-world lifelong learning 的 success detection + safe reset，再解决 long-term memory management，ASPIRE 就能 scale 到 broad real-world deployment。

参考链接（汇总）：
- ASPIRE Project Page: https://research.nvidia.com/labs/gear/aspire/
- LIBERO: https://lifelong-robot-learning.github.io/
- LIBERO-Pro: https://arxiv.org/abs/2510.03827
- Robosuite: https://robosuite.github.io/
- BEHAVIOR-1K: https://behavior.stanford.edu/
- CaP-X: https://arxiv.org/abs/2603.22435
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground
- Claude Code Best Practices: https://www.anthropic.com/engineering/claude-code-best-practices

希望这个讲解能 build 你的 intuition！如果对某个 component 想深挖（比如 SAM3 prompting strategies 的具体 fall-back chain，或者 evolutionary search 中 hypothesis diversity 是怎么 prompt 实现的），可以继续问。
