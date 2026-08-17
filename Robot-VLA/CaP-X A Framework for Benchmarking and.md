---
source_pdf: CaP-X A Framework for Benchmarking and.pdf
paper_sha256: 9e2bb993b1fdd619117bfe546693665e93336bdb6a85a41d45556c8f28bb11f2
processed_at: '2026-08-03T14:58:26-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 CaP-X

## 1. 这篇 paper 到底在搞啥

一句话：**让 LLM/VLM 直接写 Python code 控制机器人，然后系统研究这事儿到底行不行。**

听起来简单，但里面有个一直被回避的核心问题。之前的 "Code as Policy" 工作（比如 Liang 2023 那篇 ICRA: https://arxiv.org/abs/2209.07753 ）其实有点"作弊"——它们给 LLM 一堆现成的高层 API，比如 `stack_objs_in_order()`、`pick_and_place(obj, target)`，把 perception、几何推理、control 全打包好了。LLM 只需要做"点菜"——把任务拆成几个 function call 的序列。

这就好比给一个小孩一套已经组装好一半的乐高，他只需要把几个大块拼起来，就觉得自己"会搭乐高"了。但如果你给他一袋子散装零件，让他自己看图纸拼，他就傻眼了。

这篇 paper 的核心 motivation 就是：**拆掉脚手架，看看 LLM 裸奔能跑成啥样**。

---

## 2. 为啥这是个真问题

机器人控制这事有两套传统 paradigm：

**经典控制派**：工程师手写软件，把 high-level goal 拆成 subtask，compose perception 和 control module，处理 edge case。好处是 interpretable、有几何精度保证，坏处是 scalability 瓶颈——每个 task 都要 human expert 手动设计，没法 generalize 到 open-ended environment。

**VLA 派**（Vision-Language-Action，像 RT-2: https://arxiv.org/abs/2307.15818 、π₀: https://arxiv.org/abs/2410.24164 、OpenVLA: https://arxiv.org/abs/2406.09246 ）：直接从大规模 visual-motor data 学，端到端 imitation learning。好处是 contact-rich task（叠衣服、用工具）表现好，坏处是 lack interpretability，新 embodiment 或 long-horizon task 需要重新采数据 retrain。

**Code as Policy 派**想当中间派：用 LLM 写 executable code 来 compose perception 和 control primitive。这继承了经典控制的 interpretability 和 composability，但把"human engineer"换成"LLM agent"。

但问题是，之前 CaP 工作大多用 high-level primitive，我们不知道 performance 到底来自 LLM 的 reasoning 能力，还是来自 human 设计的 API。CaP-X 就是来填这个坑的。

---

## 3. CaP-Gym：一个让 agent 写代码控制机器人的环境

### 3.1 双 loop 架构

核心设计是 **dual-loop**：

```
┌────────────────────────────────────┐
│  Code Executor Loop (有状态 REPL)   │
│  - agent 写一段 Python              │
│  - 这段 code 可以调用 primitives    │
│  - 一次 "turn" = 一段完整 program   │
└──────────────┬─────────────────────┘
               │ 调用 API
               ▼
┌────────────────────────────────────┐
│  Low-Level Environment Loop         │
│  - 物理仿真器 或 real robot         │
│  - RoboSuite / LIBERO-PRO / BEHAVIOR│
│  - 一段 code 可能触发多次 sim step  │
└────────────────────────────────────┘
```

这个设计的关键 insight：**把"思考"和"执行"解耦**。agent 在 symbolic/cartesian level 推理（"把 gripper 移到 (x, y, z)"），具体怎么转成 joint angle 由 IK solver 搞定。这让 agent 不用同时学高层规划和底层运动学。

### 3.2 Perception 和 Control Primitive

Perception 端用了几个现成的强 model：
- **SAM3**（https://arxiv.org/abs/2511.16719 ）：text-conditioned segmentation。你给它一张图和 "red cube"，它返回 red cube 的 mask
- **Molmo 2**（https://arxiv.org/abs/2601.10611 ）：open-vocabulary pointing。你问"brown spill 在哪"，它返回一个 (x, y) 像素坐标
- **OpenCV / Open3D**：标准 vision library

Control 端用了：
- **PyRoki**（https://arxiv.org/abs/2505.03728 ）：IK solver 和 motion planner
- 处理 collision checking、reachability

关键：所有 primitive 实现为 **stateless service**，可以并行高吞吐评估（这对跑 100 trial × 12 model × 8 tier 的 benchmark 必不可少）。

### 3.3 三个 simulator 整合

总共 187 个任务，来自：
- **RoboSuite**（https://arxiv.org/abs/2009.12293 ）：tabletop manipulation
- **LIBERO-PRO**（https://arxiv.org/abs/2510.03827 ）：加了 perturbation 的 LIBERO，测鲁棒性
- **BEHAVIOR-1K**（https://arxiv.org/abs/2403.09227 ）：1000 个日常活动，long-horizon mobile manipulation

---

## 4. CaP-Bench：8 个 tier 的系统化评估

这是 paper 最核心的贡献之一。benchmark 沿三个 axis 切：

| Axis | 选项 |
|------|------|
| **Abstraction Level** | High-level API vs Low-level primitive |
| **Temporal Interaction** | Single-turn vs Multi-turn |
| **Perceptual Grounding** | 无感知 / Raw RGB / Visual diff 文本 |

具体 8 个 tier：

### 4.1 Single-Turn (S1-S4)

| Tier | 感知 | Primitive | Examples | 说明 |
|------|-----|----------|----------|------|
| S1 | GT state | High | - | reasoning upper bound，屏蔽感知噪声 |
| S2 | Noisy (RGB-D) | High | - | prior CaP 工作默认 setting |
| S3 | Noisy | Low | 有 examples | 接近人类 developer 用的 API |
| S4 | Noisy | Low | 无 examples | 最难，纯从 signature 推理 |

S1 vs S2 能告诉你 perception noise 贡献了多少 failure；S3 vs S4 能告诉你 in-context example 贡献了多少；S2 vs S3/S4 能告诉你 abstraction 贡献了多少。

### 4.2 Multi-Turn (M1-M4)

| Tier | 反馈类型 | 说明 |
|------|---------|------|
| M1 | stdout/stderr | 文字执行反馈，能 print 调试 |
| M2 | Raw RGB | 直接把图像塞回 context window |
| M3 | VDM text | VLM 把图像转成结构化文字描述 |
| M4 | VDM + Low-level | M3 的 VDM 加 S3 的 low-level primitives |

VDM (Visual Differencing Module) 是这 paper 的一个关键创新，下面会详细讲。

### 4.3 完整对比表

```
       Perception  Primitives   Examples  Multi-turn  VDM
S1     State       High          -         No         No
S2     Noisy       High          -         No         No
S3     Noisy       Low          Yes        No         No
S4     Noisy       Low          No         No         No
M1     Noisy       High          -         Yes(text)  No
M2     Noisy       High          -         Yes(RGB)  No
M3     Noisy       High          -         Yes        Yes
M4     Noisy       Low          Yes        Yes        Yes
```

---

## 5. 三个核心发现（Takeaway）

### 5.1 Takeaway 1: Single-turn 下，LLM 仍不如 human expert

Figure 1 的数据很 striking：12 个 frontier model 在 S4 setting 下 zero-shot Pass@1 都没达到 human expert 水平。即使 GPT-5.2、Gemini-3-Pro、Claude Opus 4.5 这些最强 model 也明显落后。

这是个重要 reality check。LLM 在 SWE-Bench（https://arxiv.org/abs/2310.06770 ）、GPQA（https://arxiv.org/abs/2311.12622 ）等任务上接近人类，但在 robot control code generation 上还有显著 gap。

**Intuition**: 机器人代码不只是"逻辑正确"，还要"几何正确"、"物理可行"。比如你写 `move_to([0.5, 0.0, 0.3])`，这个 pose 可能在 singularity 上，可能 collide，可能 out of reach。软件代码错了抛 exception，机器人代码错了撞坏东西。

### 5.2 Takeaway 2: High-level abstraction 提升 performance 但限制 expressivity

Figure 3 显示 **单调上升**：S4 < S3 < S2 < S1，abstraction 越高 success rate 越高。这印证了 prior CaP 工作的"强 zero-shot performance"其实大量来自 designer 提供的 scaffolding。

但这里有个 trade-off：

**Abstraction 好处**：
- 把 perception + geometry + control 折叠进 single primitive
- 缩小 search space
- Model 只需做 task sequencing

**Abstraction 代价**：
- Action space 被 human prior 约束
- 有个 generality ceiling
- 表达不了 hierarchical perception fallback（见 Appendix E.2.2）这类复杂行为

这给了一个 scalable middle ground：**让 agent 自己从 low-level primitive 中 recover structure**，而不是 hardcode high-level API。这正是 CaP-Agent0 的 skill library 想做的。

### 5.3 Takeaway 3: Multi-turn + Visual Grounding 显著提升 performance

Figure 5 的结果有几个 surprising findings：

**a) M1 (text-only multi-turn) 一致提升所有模型**

execution feedback (stdout/stderr) 对 debugging 和 recovery 极其重要。这印证了 self-debug 思路（Chen et al., 2024: https://arxiv.org/abs/2304.05128 ）。

**b) M2 (raw RGB multi-turn) 反而 degrade performance！**

原文说："Counter-intuitively, directly interleaving raw RGB observations at each turn (M2) degrades performance relative to the text-only M1 baseline."

**为啥会这样？** 我的理解：foundation model 很少被训练来 jointly reason over software coding 和 physical task execution 的图像。当你往一个 coding LLM 的 context 里塞张图，它反而被干扰了——图里那些光照、阴影、背景的 noise 对代码生成毫无帮助，反而稀释了 attention。

**c) M3 (VDM) 是最优解**

VDM 用一个 VLM 把图像转成 structured natural language，比如：
- Turn 1: "场景里有一个红色 cube 在桌子左侧，绿色 cube 在右侧，目标是把红色叠到绿色上"
- Turn N: "自上次以来，红色 cube 已经被 grasp，gripper 正在上升，但还没到达绿色 cube 上方"

这种 text grounding 比 raw image 好得多，比纯 stdout 反馈也好。

**Intuition**: 把视觉信息"咀嚼"成 text 再喂给 coding agent，比直接塞图像更有效。因为：
1. Text 是 coding LLM 的 native language
2. VDM 帮忙 filter 掉 irrelevant visual detail
3. Text 容易和 code 中 variable 名对应

**d) M4 (Low-Level + VDM + Multi-turn) 达到 High-Level multi-turn parity**

Figure 6 显示，low-level primitive + multi-turn feedback 能超过 S2 (high-level single-turn)，甚至达到 M3 (high-level multi-turn) 的水平。

这支持一个核心 hypothesis: **robustness 可以在 runtime 通过增加 agent 的 reasoning/verification/self-correction capacity 来 synthesize**，即使只有 atomic primitive。这是 test-time compute scaling 在 embodied agent 上的体现。

---

## 6. CaP-Agent0: Training-Free Framework

基于 CaP-Bench 的 failure mode 分析，作者设计三个对应组件：

### 6.1 Multi-turn Visual Differencing (VDM)

解决 cross-modal alignment failure（M2 暴露的问题）。具体流程：

**Turn 1**:
```
VLM 输入: task instruction + initial RGB
VLM 输出: scene description + task-relevant attributes
```

**Turn N (N > 1)**:
```
VLM 输入: task instruction + previous RGB + current RGB
VLM 输出: 
  - previous → current 的视觉差异
  - task 完成状态
  - 结构化文字 summary
```

VDM 输出作为 coding agent observation 的一部分。这相当于给 coding agent 配了个"视觉翻译官"。

### 6.2 Auto-Synthesized Skill Library

作者发现 capable model 在 S3 经常自己写辅助函数（3D 坐标变换、grasp filtering 等）。受 Voyager（https://arxiv.org/abs/2305.16291 ）启发，CaP-Agent0 把这些自动提取成持久 skill library。

**Synthesis pipeline**：
1. 收集 S3 successful rollout 的 code
2. 用 regex 提取 function definition
3. 用 LLM 分析这些 function，识别频繁出现的 task-agnostic logic
4. Promote 成 reusable skill

最终 9 个 skill（Appendix G.1）包括：

**`rotation_matrix_to_quaternion(R)`**: 3x3 旋转矩阵转 quaternion [w, x, y, z]，用 Sheppard's method 避免数值不稳定

```python
tr = np.trace(R)  # trace = R00 + R11 + R22
if tr > 0:
    S = np.sqrt(tr + 1.0) * 2  # = 4w
    w = 0.25 * S
    x = (R[2, 1] - R[1, 2]) / S
    y = (R[0, 2] - R[2, 0]) / S
    z = (R[1, 0] - R[0, 1]) / S
# ... 其他分支处理 trace 接近 0 的情况
```

变量含义：
- `tr`: 旋转矩阵的 trace，等于 $1 + 2\cos\theta$（θ 是旋转角）
- `S`: 中间变量，等于 $4w$，用于归一化
- `w, x, y, z`: quaternion 四个分量

**`depth_to_point_cloud(depth_img, intrinsics)`**: depth → 3D 点云，用针孔相机逆变换

```python
fx = intrinsics[0, 0]  # x 方向 focal length
fy = intrinsics[1, 1]  # y 方向 focal length
cx = intrinsics[0, 2]  # principal point x
cy = intrinsics[1, 2]  # principal point y

# 对每个像素:
x_cam = (u - cx) * z / fx  # u 是 pixel column, z 是 depth
y_cam = (v - cy) * z / fy  # v 是 pixel row
```

**Intuition**: pixel coordinate $(u, v)$ 减去 principal point $(c_x, c_y)$ 得到相对光轴的 offset，乘以 $z/f$ 转成物理距离。$z$ 越大，相同 pixel offset 对应更大物理距离（perspective effect）。

**`select_top_down_grasp(grasps, scores, cam_to_world, vertical_threshold=0.8)`**: 选 top-down grasp

```python
world_z = np.array([0, 0, 1])  # 世界 up 方向
for g_camera in grasps:
    g_world = cam_to_world @ g_camera  # 变换到世界系
    R = g_world[:3, :3]  # 3x3 旋转
    gripper_approach = R[:, 2]  # gripper Z 轴在世界系方向
    alignment = -np.dot(gripper_approach, world_z)
    if alignment > vertical_threshold:  # cos θ > 0.8, θ < 37°
        # 这个 grasp 够 vertical
```

变量含义：
- `R[:, 2]`: 旋转矩阵第三列，gripper local Z 轴在 world frame 中的方向
- `world_z = [0, 0, 1]`: 世界 up
- `-dot(gripper_approach, world_z)`: gripper 朝下时 dot = -1，取负 = 1
- `vertical_threshold = 0.8`: $\cos(37°) \approx 0.8$

**Intuition**: top-down grasp 时 gripper Z 轴 = -world Z，所以 `-dot` 让 perfect top-down 时值为 1，方便 threshold。

**关键哲学**: 这些 skill 不同于 fixed high-level API。它们是 **discovered** 的，从 successful execution 中涌现，保留 low-level interface 的 expressivity 同时通过 reuse 提升 robustness。

### 6.3 Parallel Reasoning (Ensemble)

每个 turn 并行采样多个 candidate:

**Single-model**: 一个 model 9 次查询，temperature 0.1, 0.2, ..., 0.9
**Multi-model**: GPT-5.2 + Claude Opus 4.5 + Gemini-3-Pro 各 3 次，temperature 0.1, 0.5, 0.9

中心 coding agent 再 synthesize 这些 candidate 成最终 code。Synthesis prompt 的关键规则：
1. 批判性分析，假设没有 candidate 完全正确
2. 优先 explicit check 而非 assumption
3. 适当时候 combine 多个 candidate 的 best idea
4. 如果 candidate 根本分歧，选更 robust 方案

**为啥这有效**: Appendix E.2.4 给了个精彩 case study：

- **Single query** (M4): 失败后才 retroactively 加 fallback，只针对刚遇到的 failure case
- **Ensemble**: 初始 generation 就 preemptively 预期 failure case，预先实现 fallback

**Deep insight**: ensemble 不只是投票，它在 distribution 上覆盖了更多 failure mode，把 "after-the-fact fix" 转化为 "before-the-fact prevention"。Appendix G.5 显示 ensemble 还减少 average turn count——因为 preemptive 比 retroactive 高效。

### 6.4 CaP-Agent0 整体效果

Figure 8 的 ablation：
- Base (S3 low-level) → M4 (VDM) → +SL (skill library) → +1M (single-model ensemble) → +3M (multi-model ensemble)
- 每步都有提升
- 7 个 task 中 4 个达到或超过 human expert

Figure 23 显示 ensemble 还减少 average turn count，因为 preemptive handling 比 retroactive debugging 高效。

### 6.5 LIBERO-PRO 对比 VLA

Table 2 的对比很关键。LIBERO-PRO 有两种 perturbation:
- **Pos**: 物体初始位置交换
- **Task**: 指令中目标物体改变（如 "put moka pot on stove" → "put frypan on stove"）

| Method | libero-object Pos | libero-object Task | libero-goal Pos | libero-goal Task | libero-spatial Pos | libero-spatial Task |
|--------|------------------|-------------------|-----------------|------------------|-------------------|---------------------|
| OpenVLA | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| π₀ | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| π₀.₅ | 0.17 | 0.01 | 0.38 | 0.00 | 0.20 | 0.01 |
| **CaP-Agent0** | **0.22** | **0.18** | **0.26** | **0.17** | **0.12** | **0.14** |

**关键 insight**: VLA 在 task perturbation 几乎完全失败（0.00-0.01），而 CaP-Agent0 仍 robust。

**为啥**: VLA 训练数据来自固定 instruction distribution，遇到没见过的 instruction 就傻眼。CaP-Agent0 通过 runtime code generation 处理 instruction variation——它把"put X on Y"翻译成具体 perception + grasp + place 代码，X 和 Y 是什么不影响这个 process。

**但要注意公平性**: VLA 是 training-based，需要数据；CaP-Agent0 是 training-free，依赖 frontier VLM 的 prior knowledge。而且 CaP-Agent0 inference cost 远高于 VLA（multi-turn + ensemble 调用 frontier model 好多次），latency 也长得多。两者适合不同场景。

### 6.6 BEHAVIOR 真实世界任务

Table 3 是两个 mobile manipulation 任务（R1Pro wheel-based humanoid）：

| Task | Method | Nav Success | Task Success |
|------|--------|------------|--------------|
| Pick up Radio | Human | 88% | 36% |
| Pick up Radio | S3 | 72% | 24% |
| Pick up Radio | CaP-Agent0 | 80% | **56%** |
| Pick up Soda Can | Human | 80% | 72% |
| Pick up Soda Can | S3 | 52% | 32% |
| Pick up Soda Can | CaP-Agent0 | 84% | 72% |

Radio 任务中 CaP-Agent0 **超过 human expert**！原因：robot 视野受限，靠近物体会丢 sight。Human policy 是固定脚本，CaP-Agent0 能主动 reposition 获取更好视角。

**Intuition**: agentic framework 在 perception-limited 场景下可能比 hand-crafted policy 更优，因为它能主动调整策略。

---

## 7. CaP-RL: 用 RL Post-Train Coding Agent

### 7.1 为啥要 RL

CaP-Agent0 已经很强，但依赖 frontier model（GPT-5.2 等）。能不能让一个小 model（Qwen2.5-Coder-7B）也达到类似水平？这就是 RL post-training 的目标。

### 7.2 GRPO 算法

用 Group Relative Policy Optimization（来自 DeepSeek-R1: https://arxiv.org/abs/2501.12948 ），核心思想是 group-relative advantage estimation。

对每个 prompt $q$，采样一组 $\{o_1, o_2, ..., o_G\}$ 共 $G$ 个 outputs，每个 output $o_i$ 对应 reward $r_i$。

**Advantage 计算**:
$$A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

其中：
- $A_i$: 第 $i$ 个 output 的 advantage
- $\text{mean}(\{r_1, ..., r_G\})$: group 内 reward 均值
- $\text{std}(\{r_1, ..., r_G\})$: group 内 reward 标准差

这避免了训练 critic network，用 group statistics 代替 baseline。

**Policy gradient loss**:
$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right) \right] - \beta \cdot \text{KL}[\pi_\theta || \pi_{\text{ref}}]$$

其中：
- $\rho_i = \frac{\pi_\theta(o_i | q)}{\pi_{\text{old}}(o_i | q)}$: importance sampling ratio
- $\epsilon$: PPO clip 参数
- $\beta$: KL penalty 系数
- $\pi_{\text{ref}}$: reference policy（初始 model）
- KL 项防止 policy 偏离太远

**CaP-RL 的 reward**: physics simulation 的 binary task success（成功/失败），是 verifiable signal。

### 7.3 训练设置的关键选择

**用 S1 (privileged state-based) 训练，S2 (noisy) 评估**。

为啥用 S1 训练？在 S2 tier，perception 和 control error 会复合，导致 "correct programs" 也可能 fail execution，引入 credit assignment ambiguity。这跟 G1（https://arxiv.org/abs/2505.13426 ）在 VLM RL 中观察到的问题类似——RL signal 太 noisy 时学不动。

这是个 division of labor：
- **RL 负责 strategic reasoning** 提升
- **Pretraining 负责 perception** robustness

### 7.4 训练效果

Table 4 数据非常 striking:

| Method | Cube Lift (sim) | Cube Stack (sim) | Spill Wipe (sim) | Cube Lift (real) | Cube Stack (real) |
|--------|----------------|------------------|------------------|------------------|-------------------|
| Human Expert | 93% | 73% | 100% | 92% | 84% |
| Qwen 2.5 Coder 7B (base) | 25% | 4% | 30% | 24% | 12% |
| Qwen w/ CaP-RL | 80% | 44% | 93% | 84% | 76% |

Cube Stack 从 4% → 44%（11x 提升！），Spill Wipe 从 30% → 93%。

### 7.5 Sim-to-Real 的反常结果

| Task | Sim (CaP-RL) | Real (CaP-RL) | Gap |
|------|-------------|---------------|-----|
| Cube Lift | 80% | 84% | +4% |
| Cube Stack | 44% | 76% | **+32%** |

**Cube Stack 在 real world 反而比 sim 高 32 个百分点！** 这违反直觉。

作者解释：agent 优化的是 abstract perception API reasoning，不是 raw visual feature，所以 policy transfer 不需要面对 visual gap。这跟传统 VLA 的 visual feature transfer 困难形成鲜明对比。

**Deep insight**: 如果 agent reasoning 发生在 symbolic/abstract level，sim-to-real gap 可以极小。

但我对这个结果有点保留——real 只跑了 25 trial，statistical noise 大。sim 的 physics stochasticity 也可能让 sim 更难。需要更多 trial 确认。

### 7.6 RL 前后行为变化

Appendix D 的 qualitative analysis 很 insightful。

**Pre-RL failure mode**: "Step Skipping" + "Hallucinated State"

```python
# Pre-RL (失败)
green_cube_position, _, _ = get_object_pose("green cube")
placement_position = np.array(green_cube_position) + np.array([0, 0, 0.05])
goto_pose(placement_position, (0, 0, 1, 0), z_approach=0.1)
open_gripper()  # 没有 grasp red cube！直接 open
```

模型正确识别了 placement position，但 **跳过 grasp step**，似乎幻觉自己已经持有物体。

**Post-RL improvement**:

1. **Causal Sequencing**: 学会完整 manipulation chain
   - Identify → Grasp → Transport → Release
   - 通过环境交互学到"必须 close gripper 才能 place"的因果依赖

2. **Dynamic Geometric Reasoning**:

```python
# Post-RL
red_cube_pos, _, red_cube_extent = get_object_pose("red cube", return_bbox_extent=True)
green_cube_pos, _, green_cube_extent = get_object_pose("green cube", return_bbox_extent=True)
placement_pos = green_cube_pos + np.array([0, 0, red_cube_extent[2]/2 + green_cube_extent[2]/2])
```

变量含义：
- `red_cube_extent[2]`: red cube 沿 Z 轴的 extent（高度）
- `green_cube_extent[2]`: green cube 沿 Z 轴的 extent
- `red_cube_extent[2]/2 + green_cube_extent[2]/2`: 两个 cube 半高之和

**Intuition**: RL 让模型从 memorization（hard-coded 0.05 offset）转向 grounded geometric reasoning（基于实际物体尺寸计算 stacking 高度）。

---

## 8. 真实世界 Demo

Appendix A 和 E 展示了多个 zero-shot real-world 任务，全在 Franka Panda 或 AgiBot G1 上跑的，no post-training。

### 8.1 Needle in a Haystack

Cluttered scene 里找 auto pencil refill holder（uncommon object）。VLA 在 uncommon object 上常失败，CaP-Agent0 通过 pretrained VLM (Molmo 2) 成功 localize。

### 8.2 Mechanical Search

三个倒扣 cups，一个下面藏 lime。CaP-Agent0 系统性探索每个 cup。这呼应了 Huang et al. 2022 的 mechanical search 工作（https://arxiv.org/abs/2204.11577 ）。

### 8.3 Multimodal Symbolic Reasoning

用木块拼出 "59 + 8 = ?"，robot 感知 equation、reasoning、grasp 正确数字 block 放到正确位置。第一次 attempt 就成功。

### 8.4 Learning from Human Feedback

Pick apple 任务，第一次 grasp 太高。Human feedback: "grasped the apple too high"。CaP-Agent0 修改 code，第二次成功。展示了 human-in-the-loop 的灵活性。

### 8.5 Embodied Reasoning (Common Sense Physics)

"Stack objects as high as possible"。Scene 有 square 和 round objects。CaP-Agent0 推理出 **round on top of square** 的稳定 stacking 策略。

VDM agent 给出关键 context: "The cubes appear to have flat surfaces suitable for stacking, while the tennis ball is spherical and would likely need to be placed on top or handled carefully."

### 8.6 Tool Generalization with Domain Knowledge

"Take the elevator downstairs"。Robot 不直接面对电梯，需要计算 wall normal 来决定 push button 方向。CaP-Agent0 调用 **SciPy RANSAC** 算法对 segmented wall pointcloud 拟合平面，计算 surface normal。

**这非常 powerful**: coding agent interface 让 robot 可以利用整个 Python ecosystem，不局限于 predefined robotics API。这是 VLA 完全做不到的。

### 8.7 Implicit Multi-Step Reasoning for Obstructed Goals

"Place blue cube on top of yellow cube"，但初始状态是 blue 在 bottom，yellow 在 middle，green 在 top。CaP-Agent0 通过 VDM 推理出复杂 strategy：
1. Remove green cube
2. Remove yellow cube to access blue
3. Pick blue
4. Place blue on yellow

这种 task decomposition 传统上需要大量 imitation learning data 或 manual state-machine design。

---

## 9. 一些 Critical 思考

### 9.1 VDM 为啥比 raw image 好

我的理解：

1. **Training distribution mismatch**: VLM 训练数据中，"coding + image" 的联合分布很稀疏，而 "text description + code" 是常见 pattern
2. **Information bottleneck**: text 是 VLM 已经 compress 过的 representation，去除了 irrelevant visual detail
3. **Compositionality**: text 容易和 code 中 variable 名对应，image pixel 难直接 map 到 code

但 VDM 有 single-point-of-failure——如果 VDM 本身 hallucinate，错误会 propagate 到 coding agent。

### 9.2 Skill Library 的局限

9 个 skill 看起来 reasonable，但有几个问题：

1. **Generalization**: 这些 skill 从 7 个核心 task 提取，能 generalize 到 BEHAVIOR 这种 long-horizon mobile manipulation 吗？
2. **Maintenance**: primitive API 变了，skill library 要 resync
3. **Quality**: 用 regex 提取 function definition 比较 brittle

### 9.3 CaP-RL 的 S1 训练选择

用 S1 (privileged) 训练但 S2 (noisy) 评估是 sim-to-sim transfer。RL 学到的是 strategic reasoning，perception robustness 仍依赖 base model prior。这是个合理的 division of labor，但也意味着 RL 的提升有上限——如果 base model perception 不行，RL 也救不了。

### 9.4 LIBERO-PRO 对比的公平性

虽然 CaP-Agent0 在 instruction perturbation 上完胜 VLA，但要注意：

1. **Cost**: CaP-Agent0 每个 trial 调用 frontier VLM 多次，inference cost 远高于 VLA
2. **Latency**: VLA 可以 real-time control（~10-50Hz），CaP-Agent0 周期是秒级
3. **Capability ceiling**: CaP-Agent0 依赖 VLM code generation 能力，contact-rich task（pouring, insertion）可能仍是 VLA 强项

作者在 Future Work 中承认："Programmatic control performs well on long-horizon, reasoning-heavy tasks, but remains brittle for contact-rich behaviors that require tight visual servoing and continuous feedback."

### 9.5 跟 Eureka / Text2Reward 的区别

Eureka（https://arxiv.org/abs/2310.12931 ）和 Text2Reward（https://arxiv.org/abs/2309.11489 ）也用 LLM 生成 code，但区别 fundamental：

- **Eureka/Text2Reward**: LLM 生成 reward function code，训练 separate policy。LLM 是 frozen 的 code generator
- **CaP-RL**: RL 直接 fine-tune LLM 本身，LLM 是被训练的 policy

这是 "LLM as tool" vs "LLM as agent" 的区别。

### 9.6 Ensemble 减少 Turn Count 的 deep insight

M4 (single query) 平均 turn count 比 CaP-Agent0 (ensemble) 高。Reason：

- Single query: retroactive debugging，每次失败修一个 bug
- Ensemble: preemptive prevention，初始就考虑多种 failure mode

这呼应 AlphaCode（https://arxiv.org/abs/2203.07807 ）思路——大量采样 + 过滤比单次生成更有效。但 CaP-Agent0 用 synthesis 而非 filter，可能更适合 coding 这种需要 composition 的任务。

### 9.7 跟 Voyager 的关系

CaP-Agent0 的 skill library 明显受 Voyager（https://arxiv.org/abs/2305.16291 ）启发。区别：

1. **Task scope**: Voyager 是 open-ended exploration，CaP-Agent0 是 task-driven
2. **Skill verification**: Voyager 用 GPT-4 verify skill，CaP-Agent0 用 environment execution outcome
3. **Skill format**: Voyager 用 JavaScript code，CaP-Agent0 用 Python robotics API

---

## 10. 公式与 Intuition 汇总

### 10.1 针孔相机逆变换（depth → 3D point）

```python
x_cam = (u - cx) * z / fx
y_cam = (v - cy) * z / fy
```

- $(u, v)$: pixel coordinate
- $z$: depth
- $f_x, f_y$: focal length（像素单位）
- $(c_x, c_y)$: principal point（光轴与图像平面交点）

**Intuition**: pixel $(u, v)$ 减去 principal point 得到相对光轴的 offset，乘 $z/f$ 转物理距离。$z$ 越大，相同 pixel offset 对应更大物理距离（perspective effect）。

### 10.2 Frame Transformation

```python
points_cam_hom = np.hstack([points_cam, np.ones((N, 1))])  # (N, 4)
points_world_hom = (extrinsics @ points_cam_hom.T).T  # (N, 4)
```

- `extrinsics`: 4x4 camera-to-world matrix
- Homogeneous coordinates 让 affine 变换可以用单一 matrix 乘法表示

### 10.3 Top-down Grasp 几何约束

```python
world_z = np.array([0, 0, 1])
R = g_world[:3, :3]
gripper_approach = R[:, 2]
alignment = -np.dot(gripper_approach, world_z)
```

- `R[:, 2]`: 旋转矩阵第三列，gripper local Z 轴在 world frame 方向
- `world_z`: world up direction
- `-dot`: gripper 朝下时 dot = -1，取负 = 1（perfect top-down）
- `vertical_threshold = 0.8`: $\cos(37°) \approx 0.8$

### 10.4 Quaternion Convention

代码中反复出现 `wxyz` ↔ `xyzw` 转换：

```python
quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
```

- `viser`, Open3D, ROS: `[w, x, y, z]` (scalar-first)
- SciPy, PyTorch3D: `[x, y, z, w]` (scalar-last)

这种 convention mismatch 是 robotics code 常见 bug source，skill library 把它封装成 helper 很合理。

### 10.5 GRPO Group Advantage

$$A_i = \frac{r_i - \mu}{\sigma}$$

- $A_i$: 第 $i$ 个 output 的 advantage
- $\mu$: group 内 reward 均值
- $\sigma$: group 内 reward 标准差

**Intuition**: group 内 reward 高于平均 = "good"，低于平均 = "bad"。避免 absolute reward scale 问题，不需要 critic。代价是 sample efficiency 低（每 prompt 需要 $G$ 个 sample）。

---

## 11. 实验数据再分析

### 11.1 In-Context Examples 的效果

Appendix C 和 Figure 18: 加入 API usage examples 让几乎所有模型提升约 +20%。

**Insight**: 当前 LLM 的 API reasoning 能力仍依赖 in-context scaffolding。纯 zero-shot 从 signature 推理仍困难，需要 example 来"激活"相关知识。

### 11.2 LIBERO-PRO Task-level 分析

看 Table 6 细节：

- `Place(bbq_sauce, basket)`: π₀.₅ Pos=1.00, CaP-Agent0 Pos=0.12
- `Place(tomato_sauce, basket)`: π₀.₅ Pos=0.00, CaP-Agent0 Pos=0.16

**Insight**: VLA 在 in-distribution task 上仍很强，CaP-Agent0 的优势在 OOD generalization。两者适合不同场景。

### 11.3 Sim-to-Real 反常结果

Cube Stack: sim 44% → real 76%。作者解释是 abstract API reasoning 避免 visual gap。但 real 只 25 trial，可能 statistical noise。需要更多 trial 确认。

---

## 12. Future Work 与 Open Questions

### 12.1 Hybrid CaP-VLA Policy

最有前途方向：CaP 管理 high-level task logic 和 recovery，VLA 处理 low-level execution。这能 combine 两者优势：
- CaP: long-horizon reasoning, compositionality, debugging
- VLA: contact-rich control, continuous feedback, fast execution

类似 TRI LBM（https://arxiv.org/abs/2507.05331 ）和 Gemini Robotics（https://arxiv.org/abs/2510.03342 ）的 hybrid 思路。

### 12.2 Optimization-based Control Primitives

作者指出当前依赖 IK 可能 suboptimal。未来引入 trajectory optimization (trajopt) 作为 primitive，让 agent 指定 constraints 而非 just goal pose。

### 12.3 Test-Time Compute Scaling

CaP-Agent0 的 ensemble + multi-turn 是 test-time compute scaling。Snell et al. 2024（https://arxiv.org/abs/2408.03314 ）系统化研究了 LLM test-time compute。CaP-X 把这扩展到 embodied agent，但仍有很多 unexplored：

- Tree search (ToT, https://arxiv.org/abs/2305.10601 )
- Self-consistency (https://arxiv.org/abs/2203.11171 )
- Learned decoding strategies

### 12.4 Open-Ended Skill Accumulation

当前 skill library 是 batch-synthesized。能否实现 online, open-ended accumulation？类似 Voyager 的 lifelong learning。挑战：skill quality verification, pruning, conflict avoidance。

### 12.5 RL with Multi-Modal Rewards

CaP-RL 用 binary task success 作 reward。能否用 dense reward（distance-to-goal）或 process reward（code correctness）？可能加速 convergence 但引入 reward engineering。

---

## 13. 总结

CaP-X 用最简单的方式说：**系统研究"让 LLM 写代码控制机器人"这套到底行不行**。

核心发现：

1. **之前 CaP 工作有水分**：high-level primitive 给 LLM 太多 scaffolding，拆掉后裸奔确实差
2. **Test-time compute 可以弥补 abstraction removal**：low-level primitive + multi-turn + ensemble 能 match high-level single-turn
3. **Text grounding > raw image grounding**：当前 VLM 在 coding + image joint reasoning 上有 alignment gap，把图像转 text 更有效
4. **Symbolic reasoning 极小化 sim-to-real gap**：agent 在 abstract API level reasoning，避免 visual feature transfer 困难
5. **RL 可以修复 causal reasoning failure**：post-RL 模型学会 grasp → place 的因果依赖，从 memorization 转向 dynamic geometric reasoning

整个工作为 embodied coding agent 奠定了 benchmark 和方法论基础。结合 hybrid CaP-VLA、optimization-based primitive、lifelong skill accumulation，这可能是通往 generalist robot 的 viable path。

参考链接汇总：
- Project page: https://capgym.github.io
- Code as Policies: https://arxiv.org/abs/2209.07753
- Voyager: https://arxiv.org/abs/2305.16291
- DeepSeek-R1 (GRPO): https://arxiv.org/abs/2501.12948
- SWE-Bench: https://arxiv.org/abs/2310.06770
- LIBERO-PRO: https://arxiv.org/abs/2510.03827
- BEHAVIOR-1K: https://arxiv.org/abs/2403.09227
- RoboSuite: https://arxiv.org/abs/2009.12293
- π₀: https://arxiv.org/abs/2410.24164
- π₀.₅: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
- SAM3: https://arxiv.org/abs/2511.16719
- Molmo 2: https://arxiv.org/abs/2601.10611
- PyRoki: https://arxiv.org/abs/2505.03728
- Eureka: https://arxiv.org/abs/2310.12931
- Text2Reward: https://arxiv.org/abs/2309.11489
- Test-time compute scaling: https://arxiv.org/abs/2408.03314
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Self-Consistency: https://arxiv.org/abs/2203.11171
- AlphaCode: https://arxiv.org/abs/2203.07807
- Qwen2.5-Coder: https://arxiv.org/abs/2409.12186
- Gemini Robotics: https://arxiv.org/abs/2510.03342
- TRI LBM: https://arxiv.org/abs/2507.05331
- G1 (RLVR for VLM): https://arxiv.org/abs/2505.13426

---

# CaP-X: 用 Coding Agent 控制机器人的系统研究

## 1. 核心问题与 Motivation

这篇 paper 来自 Stanford、Berkeley、NVIDIA 等团队（包括 Ken Goldberg、Fei-Fei Li、Linxi "Jim" Fan 等大佬），核心研究一个根本性问题：**当 LLM/VLM 直接通过生成 executable code 控制机器人时，它们到底有多强？**

之前的 Code-as-Policy 工作（如 Liang et al., 2023 ICRA: https://arxiv.org/abs/2209.07753 ）有个隐藏问题——它们依赖大量 human-crafted 高层 primitives，比如 `stack_objs_in_order()` 这种函数，把 perception、geometry reasoning、control 全打包好了。我们无法判断：performance 究竟来自 agent 的 reasoning 能力，还是来自 designer 提供的 scaffolding？

CaP-X 提供了一个系统化的 framework 来回答这个问题。整个 framework 包含四个组件：
- **CaP-Gym**: 交互式环境
- **CaP-Bench**: benchmark
- **CaP-Agent0**: training-free agent framework  
- **CaP-RL**: 用 RL post-train coding agent

参考链接：https://capgym.github.io

---

## 2. CaP-Gym 架构详解

CaP-Gym 是一个 hierarchical control framework，基于 Gymnasium interface（Brockman et al., 2016: https://arxiv.org/abs/1606.01540 ）。关键设计是 **dual-loop architecture**：

```
┌─────────────────────────────────────────────┐
│  Code Executor Loop (stateful REPL)         │
│  - Agent 生成的 Python code 在这里执行       │
│  - 一个 "turn" = 一段完整 program 执行       │
└─────────────┬───────────────────────────────┘
              │ 调用 primitives
              ▼
┌─────────────────────────────────────────────┐
│  Low-Level Environment Loop                  │
│  - 物理仿真器 或 real robot                  │
│  - RoboSuite / LIBERO-PRO / BEHAVIOR         │
│  - 一个 code turn 可能触发多个 sim steps     │
└─────────────────────────────────────────────┘
```

### 2.1 Perception Primitives

所有计算密集的 perception primitive 实现为 **stateless services**，支持高吞吐并行评估：

- **SAM3** (Carion et al., 2025: https://arxiv.org/abs/2511.16719 ): language-conditioned segmentation
- **Molmo 2** (Clark et al., 2026: https://arxiv.org/abs/2601.10611 ): open-vocabulary pointing
- **OpenCV** (Bradski, 2000)
- **Open3D** (Zhou et al., 2018: https://arxiv.org/abs/1801.09847 )

### 2.2 Control Primitives

agent 不直接 emit joint-space actions，而是调用：
- **PyRoki** (Kim et al., 2025 IROS: https://arxiv.org/abs/2505.03728 ): IK solver 和 motion planner
- 处理 collision checking, reachability, action-space transformation
- 让 agent 在 task-oriented Cartesian space 推理，把 feasibility 委托给 controller

这个设计哲学很关键：agent 在 "thinking" 层面用 Cartesian pose，execution 层面由 IK solver 保证可行性。这避免了 agent 同时学高层规划和底层运动学。

### 2.3 整合的任务套件

CaP-Gym 整合了 **187 个任务** 来自三个标准 simulator：
- **RoboSuite** (Zhu et al., 2020: https://arxiv.org/abs/2009.12293 ): tabletop manipulation
- **LIBERO-PRO** (Zhou et al., 2025: https://arxiv.org/abs/2510.03827 ): 鲁棒性测试版的 LIBERO
- **BEHAVIOR** (Li et al., 2024: https://arxiv.org/abs/2403.09227 ): 1000 个日常活动的 embodied AI benchmark

---

## 3. CaP-Bench: 8 个 Tier 的系统化评估

这是 paper 最有价值的部分之一。CaP-Bench 沿三个 axis 系统化评估：

| Axis | 选项 |
|------|------|
| Abstraction Level | High-level vs Low-level primitives |
| Temporal Interaction | Single-turn vs Multi-turn |
| Perceptual Grounding | No perception / Raw RGB / Visual diff text |

具体 8 个 tier：

### 3.1 Single-Turn Tiers (S1-S4)

**S1 - Privileged + High-Level**: 使用 ground-truth sim state（masks, object poses），human-crafted 高层 API。这是 reasoning 的 upper bound，能区分 algorithmic failure 和 perception error。

**S2 - Noisy + High-Level**: 用 real perception modules 处理 RGB-D 输入。大多数 prior CaP 工作的默认 setting。

**S3 - Noisy + Low-Level + Examples**: human-designed abstraction 被替换为底层 primitives（如 `solve_ik()`, `sam3_text_prompt()`），但 docstring 包含 usage examples。

**S4 - Noisy + Low-Level, No Examples**: 移除 usage examples，agent 只能从 function signature 和 docstring 推理。

### 3.2 Multi-Turn Tiers (M1-M4)

**M1 - Text-Only Multi-turn**: agent 收到 stdout/stderr，可以做 state introspection，主动 inject `print()` 暴露 hidden symbolic variables。

**M2 - Multimodal**: 每个 turn 把当前 RGB observation pipe 回 context window。仅 multimodal foundation model 可用。

**M3 - Visual Differencing Module (VDM)**: 用 VLM 把 visual observation 转成 structured natural language。第一 turn 描述场景和 task-relevant attributes，后续 turn 描述 previous→current 的差异和任务完成状态。

**M4 - Low-Level + VDM**: 同时有 VDM 和 S3 的 low-level primitives + usage examples。

### 3.3 完整对比表

| Tier | Perception | Primitives | Examples | Multi-turn | VDM |
|------|-----------|-----------|----------|-----------|-----|
| S1 | State-based | High | - | No | No |
| S2 | Noisy | High | - | No | No |
| S3 | Noisy | Low | Yes | No | No |
| S4 | Noisy | Low | No | No | No |
| M1 | Noisy | High | - | Yes (text) | No |
| M2 | Noisy | High | - | Yes (RGB) | No |
| M3 | Noisy | High | - | Yes | Yes |
| M4 | Noisy | Low | Yes | Yes | Yes |

---

## 4. 三个关键 Takeaway

### Takeaway 1: Single-turn 下，frontier models 仍不如 human expert

Figure 1 的数据非常 striking。12 个模型在 S4 setting 下，zero-shot Pass@1 都没达到 human expert 的水平。即使是 GPT-5.2、Gemini-3-Pro、Claude Opus 4.5 这些 frontier 模型也明显落后。

这是个重要的 reality check：在 software engineering (SWE-Bench, https://arxiv.org/abs/2310.06770 )、math (GPQA, https://arxiv.org/abs/2311.12622 ) 等领域，LLM 已经接近或超越人类，但在 robot control code generation 上还有大 gap。

### Takeaway 2: High-Level abstraction 提升 performance 但限制 expressivity

Figure 3 显示一个 **单调上升** 趋势：S4 → S3 → S2 → S1，abstraction 越高，success rate 越高。这印证了 prior CaP 工作的 "强 zero-shot performance" 其实大量来自 designer 提供的 scaffolding。

但这里有个 trade-off：

**Abstraction 的好处**：
- 把 low-level perception + geometric reasoning + control 折叠进 single primitive
- 缩小 effective search space
- Model 只需做 task sequencing

**Abstraction 的代价**：
- Action space 被 human prior 约束
- Imposes a generality ceiling
- 无法表达如 hierarchical perception fallback（见 Appendix E.2.2）这类 expressive behavior

这个 insight 直指一个 scalable middle ground：**agent 应该自己从 low-level primitives 中 recover structure**，而不是依赖 human 设计的 high-level API。这正是 CaP-Agent0 的 skill library 想做的。

### Takeaway 3: Multi-turn + Visual Grounding 显著提升 performance

Figure 5 的结果有几个 surprising findings：

**a) M1 (text-only multi-turn) 一致提升所有模型**：execution feedback (stdout/stderr) 对 debugging 和 recovery 极其重要。这印证了 software agent 的 self-debug 思路（Chen et al., 2024 ICLR: https://arxiv.org/abs/2304.05128 ）。

**b) M2 (raw RGB multi-turn) 反而 degrade performance**：
> "Counter-intuitively, directly interleaving raw RGB observations at each turn (M2) degrades performance relative to the text-only M1 baseline."

作者假设这是 cross-modal alignment gap：foundation models 很少被训练来 jointly reason over software coding 和 physical task execution 的图像。这呼应了 Hu et al. (2025: https://arxiv.org/abs/2505.15146 ) 和 Wang et al. (2026: https://arxiv.org/abs/2601.16973 ) 的观察：text-grounded feedback 比 raw image 表现更好。

**c) M3 (VDM) 是最优解**：通过把视觉转成 structured natural language，M3 显著超过 M1 和 M2。这给了我们一个重要 intuition：**在当前 VLM 能力下，把视觉 grounding 转成 text 比直接输入 image 更有效**。

**d) M4 (Low-Level + VDM + Multi-turn) 达到 High-Level multi-turn 的 parity**：Figure 6 显示，低层 primitives + multi-turn feedback 不仅能超过 S2 (high-level single-turn)，还能达到 M3 (high-level multi-turn) 的水平。

这支持一个核心 hypothesis：**robustness 可以在 runtime 通过增加 agent 的 reasoning/verification/self-correction capacity 来 synthesize**，即使只有 atomic primitives。这正是 test-time compute scaling 的体现。

---

## 5. CaP-Agent0: Training-Free Agentic Framework

CaP-Agent0 是 paper 的核心贡献之一，基于 CaP-Bench 的三个 failure mode 设计三个对应组件：

### 5.1 Multi-turn Visual Differencing (VDM)

VDM 解决 cross-modal alignment failure（M2 tier 暴露的问题）。具体流程：

**Turn 1**:
```
VLM Input: task instruction + initial RGB observation
VLM Output: structured scene description + task-relevant visual attributes
```

**Turn N (>1)**:
```
VLM Input: task instruction + previous RGB + current RGB
VLM Output: 
  - Differences between previous and current image
  - Whether task is completed
  - Structured text summary
```

VDM 的输出作为 coding agent observation context 的一部分。

### 5.2 Auto-Synthesized Skill Library

这是最有意思的设计之一。作者发现 capable models 在 S3/S4 经常 synthesizes auxiliary utility functions（如 3D 坐标变换、grasp filtering）。受 Voyager (Wang et al., 2023: https://arxiv.org/abs/2305.16291 ) 启发，CaP-Agent0 引入自动合成、跨 trial 持久的 skill library。

**Synthesis pipeline**:
1. 收集 S3 tier successful rollouts 的 code outputs
2. 用 regex 提取 function definitions
3. 用 LLM 分析这些 functions，识别频繁出现的 task-agnostic logic
4. 把这些 logic promote 成 reusable skills

完整 9 个 synthesized functions（Appendix G.1）包括：
- `rotation_matrix_to_quaternion(R)`: 用 Sheppard's method 转 3x3 旋转矩阵到 quaternion
- `decompose_transform(T)`: 分解 4x4 homogeneous matrix
- `depth_to_point_cloud(depth_img, intrinsics)`: depth → 3D 点云
- `mask_to_world_points(mask, depth, intrinsics, extrinsics)`: mask 像素 → 世界坐标点
- `pixel_to_world_point(u, v, z, intrinsics, extrinsics)`: 单像素 → 3D 点
- `transform_points(points, transform_matrix)`: 应用 homogeneous transform
- `interpolate_segment(p1, p2, step)`: 生成线段 waypoints
- `normalize_vector(v)`: 单位化
- `select_top_down_grasp(grasps, scores, cam_to_world, vertical_threshold)`: 选择 top-down grasp

让我详解 `select_top_down_grasp` 的几何逻辑：

```python
def select_top_down_grasp(grasps, scores, cam_to_world, vertical_threshold=0.8):
    """
    Args:
        grasps: (N, 4, 4) - N 个 grasp pose, camera frame
        scores: (N,) - grasp confidence
        cam_to_world: (4, 4) - camera-to-world extrinsic
        vertical_threshold: cos θ 阈值, 1.0 = 完全垂直
    """
    world_z = np.array([0, 0, 1])  # 世界 Z 轴
    for i, g_camera in enumerate(grasps):
        g_world = cam_to_world @ g_camera  # 变换到世界系
        R = g_world[:3, :3]                # 提取 3x3 旋转
        gripper_approach = R[:, 2]         # 第 3 列 = approach vector
        # 对 Franka, gripper Z 轴朝下时 dot(gripper_z, world_z) = -1
        alignment = -np.dot(gripper_approach, world_z)
        if alignment > vertical_threshold:  # cos θ > 0.8, θ < 37°
            ...
```

这里 `R[:, 2]` 是旋转矩阵第三列，代表 gripper 的 Z 轴在世界系中的方向。`-np.dot(gripper_approach, world_z)` 计算 approach vector 与世界 -Z 方向的 alignment。如果 gripper 朝下（gripper Z = -world Z），alignment = 1。

**关键设计哲学**：skill library 不同于 fixed high-level API。这些 skill 是 **discovered** 的，从 successful execution 中涌现，保留 low-level interface 的 expressivity 同时通过 reuse 提升 robustness。

### 5.3 Parallel Reasoning (Ensembled Code Generation)

受 Pan et al. (2025: https://arxiv.org/abs/2504.15466 ) 和 Jin et al. (2025: https://arxiv.org/abs/2502.11517 ) 启发，CaP-Agent0 在每个 turn 并行采样 candidates：

**Single-model**: 同一模型 9 次查询，temperature 0.1, 0.2, ..., 0.9
**Multi-model**: GPT-5.2, Claude Opus 4.5, Gemini-3-Pro 各 3 次，temperature 0.1, 0.5, 0.9

中心 coding agent 再 synthesize 这些 candidates 成最终 code snippet。Synthesis prompt（Appendix G.3）的关键规则：
1. 批判性分析，假设没有 candidate 完全正确
2. 优先 explicit checks 而非 assumptions
3. 适当时候 combine 多个 candidates 的最佳 idea
4. 如果 candidates 根本分歧，选更 robust 的方案

**为什么这有效**：从 S2 和 S3 的结果看，failure 经常来自 insufficient test-time exploration，而非 capability 不足。Appendix E.2.4 给出了一个精彩 case study：

- **Single query** (M4): 失败后才 retroactively 加 fallback，只针对刚刚遇到的 failure case
- **Ensemble**: 在 initial generation 就 preemptively 预期 failure case，预先实现 fallback

这是一个非常 deep 的观察：**ensemble 不只是投票，它在 distribution 上覆盖了更多 failure mode**，从而把 "after-the-fact fix" 转化为 "before-the-fact prevention"。

### 5.4 CaP-Agent0 在 CaP-Bench 的表现

Figure 8 显示 ablation：
- Base (S3 low-level) → M4 (VDM) → +SL (skill library) → +1M (single-model ensemble) → +3M (multi-model ensemble)
- 每一步都有提升
- 在 7 个任务中的 4 个，CaP-Agent0 达到或超过 human expert

### 5.5 LIBERO-PRO 对比

Table 2 的对比非常重要。LIBERO-PRO (https://arxiv.org/abs/2510.03827 ) 通过两种 perturbation 测试鲁棒性：
- **Pos perturbation**: 物体初始位置交换
- **Task perturbation**: 指令中的目标物体改变

| Method | libero-object Pos | libero-object Task | libero-goal Pos | libero-goal Task | libero-spatial Pos | libero-spatial Task |
|--------|------------------|-------------------|-----------------|------------------|-------------------|---------------------|
| OpenVLA | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| π₀ | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| π₀.₅ | 0.17 | 0.01 | 0.38 | 0.00 | 0.20 | 0.01 |
| **CaP-Agent0** | **0.22** | **0.18** | **0.26** | **0.17** | **0.12** | **0.14** |

关键 insight：**VLA 在 task perturbation 上几乎完全失败（0.00-0.01），而 CaP-Agent0 仍然 robust**。这是因为 VLA 训练数据来自固定 instruction distribution，而 CaP-Agent0 通过 runtime code generation 处理 instruction variation。

但要注意公平性：VLA 是 training-based，需要数据；CaP-Agent0 是 training-free，依赖 frontier VLM 的 prior knowledge。这是不同 paradigm 的 trade-off。

### 5.6 BEHAVIOR 真实世界长视野任务

Table 3 展示了两个 mobile manipulation 任务：

| Task | Method | Nav Success | Task Success |
|------|--------|------------|--------------|
| Pick up Radio | Human | 88% | 36% |
| Pick up Radio | S3 | 72% | 24% |
| Pick up Radio | CaP-Agent0 | 80% | 56% |
| Pick up Soda Can | Human | 80% | 72% |
| Pick up Soda Can | S3 | 52% | 32% |
| Pick up Soda Can | CaP-Agent0 | 84% | 72% |

Radio 任务中 CaP-Agent0 (56%) **超过 human expert (36%)**！这非常 striking。原因：robot 视野受限，靠近物体会丢失 sight。CaP-Agent0 能主动 reposition 获取更好视角，而 human policy 是固定脚本。

这给我一个重要 intuition：**agentic framework 在 perception-limited 场景下可能比 hand-crafted policy 更优**，因为它能主动调整策略。

---

## 6. CaP-RL: 用 RL Post-Train Coding Agent

这是 paper 最技术性的部分之一。CaP-RL 把 RLVR (Reinforcement Learning with Verifiable Rewards) 范式从 reasoning/coding 扩展到 robot manipulation。

### 6.1 方法论

**算法**: Group Relative Policy Optimization (GRPO)
- 来自 DeepSeekMath (Shao et al., 2024: https://arxiv.org/abs/2402.03300 ) 和 DeepSeek-R1 (Guo et al., 2025: https://arxiv.org/abs/2501.12948 )
- Base model: Qwen2.5-Coder-7B-Instruct (Hui et al., 2024: https://arxiv.org/abs/2409.12186 )
- 训练任务: Cube Lift, Cube Stack, Spill Wipe
- 训练 tier: **S1 (privileged state-based)**，避免 S2 的 noisy reward

**为什么用 S1 训练？** 这是个 critical 设计选择。在 S2 tier，perception 和 control error 会复合，导致 "correct programs" 也可能 fail execution，引入 credit assignment ambiguity。这和 G1 (Chen et al., 2025: https://arxiv.org/abs/2505.13426 ) 在 VLM RL 中观察到的问题类似。

### 6.2 GRPO 公式详解

GRPO 的核心思想是 group-relative advantage estimation。对每个 prompt $q$，采样一组 $\{o_1, o_2, ..., o_G\}$ 共 $G$ 个 outputs，每个 output $o_i$ 对应 reward $r_i$。

**Advantage 计算**:
$$A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

其中：
- $A_i$ 是第 $i$ 个 output 的 advantage
- $\text{mean}(\{r_1, ..., r_G\})$ 是 group 内 reward 均值
- $\text{std}(\{r_1, ..., r_G\})$ 是 group 内 reward 标准差

这避免了训练 critic network，用 group statistics 代替 baseline。

**Policy gradient loss**:
$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right) \right] - \beta \cdot \text{KL}[\pi_\theta || \pi_{\text{ref}}]$$

其中：
- $\rho_i = \frac{\pi_\theta(o_i | q)}{\pi_{\text{old}}(o_i | q)}$ 是 importance sampling ratio
- $\epsilon$ 是 PPO clip 参数
- $\beta$ 是 KL penalty 系数
- $\pi_{\text{ref}}$ 是 reference policy（通常初始 model）
- KL 项防止 policy 偏离太远

**CaP-RL 的特殊之处**: reward 来自 physics simulation 的 execution outcome，是 binary verifiable signal（task 成功/失败），而非中间 reward。

### 6.3 训练效果

Table 4 展示了 dramatic improvement:

| Method | Cube Lift (sim) | Cube Stack (sim) | Spill Wipe (sim) | Cube Lift (real) | Cube Stack (real) |
|--------|----------------|------------------|------------------|------------------|-------------------|
| Human Expert | 93% | 73% | 100% | 92% | 84% |
| Qwen 2.5 Coder 7B (base) | 25% | 4% | 30% | 24% | 12% |
| Qwen w/ CaP-RL | 80% | 44% | 93% | 84% | 76% |

Cube Stack 从 4% → 44%（11x 提升！），Spill Wipe 从 30% → 93%。

### 6.4 Sim-to-Real Transfer 的 Minimal Gap

这是个非常 striking 的结果：

| Task | Sim (CaP-RL) | Real (CaP-RL) | Gap |
|------|-------------|---------------|-----|
| Cube Lift | 80% | 84% | +4% |
| Cube Stack | 44% | 76% | +32% |

**Cube Stack 在 real world 反而比 sim 高！** 这违反直觉。作者解释：因为 agent 优化的是 abstract perception API reasoning 而非 raw visual features，policy transfer 不需要面对 visual gap。

这给了一个 deep insight: **如果 agent reasoning 发生在 symbolic/abstract level，sim-to-real gap 可以极小**。这和传统 VLA 的 visual feature transfer 困难形成鲜明对比。

### 6.5 RL Post-Training 的行为变化

Appendix D 提供了 qualitative analysis，非常 insightful：

**Pre-RL failure mode**: "Step Skipping" + "Hallucinated State"
```python
# Pre-RL code (失败)
green_cube_position, _, _ = get_object_pose("green cube")
placement_position = np.array(green_cube_position) + np.array([0, 0, 0.05])
goto_pose(placement_position, (0, 0, 1, 0), z_approach=0.1)
open_gripper()  # 没有 grasp red cube！直接 open
```

模型正确识别 placement position，但 **跳过 grasp step**，似乎幻觉自己已经持有物体。

**Post-RL improvement**:

1. **Causal Sequencing**: 学会完整 manipulation chain
   - Identify → Grasp → Transport → Release
   - 通过环境交互学到 "must close gripper before place" 的因果依赖

2. **Dynamic Geometric Reasoning**:
```python
# Post-RL code
red_cube_pos, _, red_cube_extent = get_object_pose("red cube", return_bbox_extent=True)
green_cube_pos, _, green_cube_extent = get_object_pose("green cube", return_bbox_extent=True)
placement_pos = green_cube_pos + np.array([0, 0, red_cube_extent[2]/2 + green_cube_extent[2]/2])
```

变量含义：
- `red_cube_extent[2]`: red cube 沿 Z 轴的 extent（高度）
- `green_cube_extent[2]`: green cube 沿 Z 轴的 extent
- `red_cube_extent[2]/2 + green_cube_extent[2]/2`: 两个 cube 半高之和，即 stacking 时中心点的高度 offset

这表明 RL 让模型从 memorization（hard-coded 0.05 offset）转向 grounded geometric reasoning。

---

## 7. 真实世界 Demo 解析

Appendix A 和 E 展示了多个 zero-shot real-world 任务，非常 impressive：

### 7.1 Needle in a Haystack
在 cluttered scene 中找到 auto pencil refill holder（uncommon object）。VLA 在 uncommon object 上常失败，CaP-Agent0 通过 pretrained VLM (Molmo 2) 成功 localize。

### 7.2 Mechanical Search
三个倒扣 cups，一个下面藏 lime。CaP-Agent0 系统性探索每个 cup。这呼应了 Huang et al. (2022: https://arxiv.org/abs/2204.11577 ) 的 mechanical search 工作。

### 7.3 Multimodal Symbolic Reasoning
用木块拼出 "59 + 8 = ?"，robot 感知 equation、reasoning、grasp 正确数字 block 放到正确位置。第一次 attempt 就成功。

### 7.4 Learning from Human Feedback
Pick apple 任务，第一次 grasp 太高。Human feedback: "grasped the apple too high"。CaP-Agent0 修改 code，第二次成功。这展示了 human-in-the-loop 的灵活性。

### 7.5 Embodied Reasoning (Common Sense Physics)
"Stack objects as high as possible"。Scene 有 square 和 round objects。CaP-Agent0 推理出 **round on top of square** 的稳定 stacking 策略。

VDM agent 给出关键 context: "The cubes appear to have flat surfaces suitable for stacking, while the tennis ball is spherical and would likely need to be placed on top or handled carefully."

### 7.6 Tool Generalization with Domain Knowledge
"Take the elevator downstairs"。Robot 不直接面对电梯，需要计算 wall normal 来决定 push button 方向。CaP-Agent0 调用 **SciPy RANSAC** 算法对 segmented wall pointcloud 拟合平面，计算 surface normal。

这非常 powerful：**coding agent interface 让 robot 可以利用整个 Python ecosystem**，而不局限于 predefined robotics API。这是 VLA 完全做不到的。

### 7.7 Implicit Multi-Step Reasoning for Obstructed Goals
"Place blue cube on top of yellow cube"，但初始状态是 blue 在 bottom，yellow 在 middle，green 在 top。CaP-Agent0 通过 VDM 推理出复杂 strategy：
1. Remove green cube
2. Remove yellow cube to access blue
3. Pick blue
4. Place blue on yellow

这种 task decomposition 传统上需要大量 imitation learning data 或 manual state-machine design。

---

## 8. 技术细节深挖

### 8.1 S3 Primitives 完整 API（Appendix F）

让我分析 S3 tier 提供的完整 primitive set：

**Perception APIs**:
- `get_observation()`: 返回 dict 包含 RGB, depth, intrinsics, extrinsics
- `segment_sam3_text_prompt(rgb, text_prompt)`: 文本引导分割
- `segment_sam3_point_prompt(rgb, point_coords)`: 点引导分割
- `point_prompt_molmo(image, text_prompt)`: Molmo 指点
- `get_oriented_bounding_box_from_3d_points(points)`: 3D 点 → OBB
- `plan_grasp(depth, intrinsics, segmentation)`: Contact-GraspNet grasp planning

**Control APIs**:
- `solve_ik(position, quaternion_wxyz)`: 求解 IK
- `move_to_joints(joints)`: 阻塞式移动到 joint configuration
- `open_gripper()` / `close_gripper()`: 夹爪控制

**Bimanual variants**: `solve_ik_arm0`, `solve_ik_arm1`, `move_to_joints_both`, etc.

### 8.2 `plan_grasp` 的返回值结构

```python
grasp_poses: np.ndarray  # shape (K, 4, 4), camera frame homogeneous transforms
grasp_scores: np.ndarray  # shape (K,), confidence scores
```

注意：grasp 是 **camera frame**，caller 需要自己 transform 到 world frame。这种设计选择让 primitive 保持 stateless，但也增加 agent 的认知负担——它必须理解 frame transformation。

### 8.3 `solve_ik` 的 TCP offset 处理

```python
def solve_ik(self, position, quaternion_wxyz):
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot = SciRotation.from_quat(quat_xyzw)
    offset_pos = pos + rot.apply(self._TCP_OFFSET)  # TCP offset 旋转到 world frame
    # ... 调用 IK solver
```

变量解释：
- `quaternion_wxyz`: $[w, x, y, z]$ 格式的单位 quaternion
- `quat_xyzw`: SciPy 的 $[x, y, z, w]$ 格式
- `self._TCP_OFFSET`: 默认 `[0.0, 0.0, -0.107]`，TCP 到 end-effector 的 offset
- `rot.apply(self._TCP_OFFSET)`: 把 TCP offset 旋转到 world frame

这里有个 subtle 点：用户给的 `position` 是 TCP 目标位置，IK solver 求解的是 end-effector joint configuration，所以需要加 offset。

### 8.4 Contact-GraspNet 的 z_range 配置

```python
grasp_sample, grasp_scores, _ = self.grasp_net_plan_fn(
    depth, intrinsics, segmentation,
    1,
    z_range=[0.2, 3.5] if self.is_handover else [0.2, 2.0],
    forward_passes=1 if self.is_handover else 3,
)
```

- `z_range`: grasp 深度范围（米），handover 任务用更大范围
- `forward_passes`: forward pass 次数，更多 = 更准确但更慢

### 8.5 GRASP pose 的 TCP adjustment

```python
grasp_sample_tf = (
    vtf.SE3.from_matrix(grasp_sample) @ 
    vtf.SE3.from_translation(np.array([0, 0, 0.12]))
).as_matrix()
```

这是用 `viser.transforms` (Yi et al., 2025: https://arxiv.org/abs/2507.22885 ) 做 SE3 变换：
- `vtf.SE3.from_matrix(grasp_sample)`: 从 4x4 matrix 构造 SE3
- `vtf.SE3.from_translation([0, 0, 0.12])`: 沿 grasp approach 方向平移 12cm
- `@`: SE3 乘法（先平移再旋转的复合变换）

这是为了让 grasp pose 退后 12cm，给 pre-grasp approach 留空间。

---

## 9. 一些 Critical 思考

### 9.1 关于 VDM 的有效性

为什么 text grounding 比 raw image 更好？我认为有几个原因：

1. **Training distribution mismatch**: VLM 训练数据中，"coding + image" 的联合分布很稀疏，而 "text description + code" 是常见 pattern
2. **Information bottleneck**: text 是 VLM 已经 compress 过的 representation，去除了 irrelevant visual detail
3. **Compositionality**: text 容易和 code 中 variable 名对应，而 image pixel 难以直接 map 到 code

但这也有局限：VDM 依赖一个 strong VLM，如果 VDM 本身 hallucinate，错误会 propagate 到 coding agent。这是个 single-point-of-failure。

### 9.2 关于 Skill Library 的局限

9 个 synthesized skill 看起来很 reasonable，但有几个潜在问题：

1. **Generalization**: 这些 skill 是从 7 个核心 task 的 successful rollouts 提取的，能否 generalize 到 BEHAVIOR 这种 long-horizon mobile manipulation？
2. **Maintenance**: 当 primitive API 变化时，skill library 需要 resync
3. **Quality**: 用 regex 提取 function definition 比较 brittle，更复杂 pattern 可能遗漏

### 9.3 关于 CaP-RL 的 S1 训练选择

用 S1 (privileged) 训练但 S2 (noisy) 评估是个有趣的 sim-to-sim transfer。这避免了 perception noise 在 RL 中的 credit assignment 问题。但这也意味着：

- RL 学到的是 **strategic reasoning**，不是 perception robustness
- Perception robustness 仍依赖 base model 的 prior

这是一个 division of labor：RL 负责 strategic improvement，pretraining 负责 perception。

### 9.4 LIBERO-PRO 对比的公平性

虽然 CaP-Agent0 在 instruction perturbation 上完胜 VLA，但要注意：

1. **Cost**: CaP-Agent0 每个 trial 调用 frontier VLM 多次（multi-turn + ensemble），inference cost 远高于 VLA
2. **Latency**: VLA 可以 real-time control（~10-50Hz），CaP-Agent0 的 code generation + execution 周期是秒级
3. **Capability ceiling**: CaP-Agent0 依赖 VLM 的 code generation 能力，contact-rich task（pouring, insertion）可能仍是 VLA 强项

作者在 Future Work 中也承认这点："Programmatic control performs well on long-horizon, reasoning-heavy tasks, but remains brittle for contact-rich behaviors that require tight visual servoing and continuous feedback."

### 9.5 与 Eureka 和 Text2Reward 的区别

Related work 中提到 Eureka (Ma et al., 2024 ICLR: https://arxiv.org/abs/2310.12931 ) 和 Text2Reward (Xie et al., 2024 ICLR: https://arxiv.org/abs/2309.11489 )。区别在于：

- **Eureka/Text2Reward**: LLM 生成 reward function，训练 separate policy。LLM 是 frozen 的 code generator。
- **CaP-RL**: RL 直接 fine-tune LLM 本身，LLM 是被训练的 policy。

这是 fundamental 的区别：CaP-RL 把 LLM 视为可优化的 agent，而非 frozen tool。

### 9.6 关于 Multi-turn 平均 turn count

Appendix G.5 和 Figure 23 显示 ensemble 减少 average turn count。这有个 interesting implication：

- Single query: retroactive debugging，每次失败后修一个 bug
- Ensemble: preemptive prevention，初始就考虑多种 failure mode

这呼应了 AlphaCode (Li et al., 2022: https://arxiv.org/abs/2203.07807 ) 的思路——大量采样 + 过滤比单次生成更有效。但 CaP-Agent0 用 synthesis 而非 filter，可能更适合 coding 这种需要 composition 的任务。

### 9.7 Voyager 的灵感

CaP-Agent0 的 skill library 明显受 Voyager (Wang et al., 2023: https://arxiv.org/abs/2305.16291 ) 启发。Voyager 在 Minecraft 中通过 iterative skill library accumulation 实现 open-ended learning。CaP-Agent0 把这思路迁移到 robot manipulation，但有几个不同：

1. **Task scope**: Voyager 是 open-ended exploration，CaP-Agent0 是 task-driven
2. **Skill verification**: Voyager 用 GPT-4 verify skill，CaP-Agent0 用 environment execution outcome
3. **Skill format**: Voyager 用 JavaScript code，CaP-Agent0 用 Python robotics API

---

## 10. 公式汇总与 Intuition Building

### 10.1 3D 点云反投影（depth → point cloud）

```python
# From skill library
fx = intrinsics[0, 0]  # focal length x
fy = intrinsics[1, 1]  # focal length y
cx = intrinsics[0, 2]  # principal point x
cy = intrinsics[1, 2]  # principal point y

x_cam = (u - cx) * z / fx
y_cam = (v - cy) * z / fy
```

这是针孔相机模型的逆变换：
- $(u, v)$: pixel coordinate
- $z$: depth
- $f_x, f_y$: focal length
- $(c_x, c_y)$: principal point
- $(x_{cam}, y_{cam}, z)$: camera frame 3D point

**Intuition**: pixel coordinate 减去 principal point 得到相对于光轴的 offset，乘以 $z/f$ 转换为物理距离。$z$ 越大，相同 pixel offset 对应更大物理距离（perspective effect）。

### 10.2 Frame Transformation

```python
points_cam_hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])  # (N, 4)
points_world_hom = (extrinsics @ points_cam_hom.T).T  # (4, N) → (N, 4)
points_world = points_world_hom[:, :3]
```

- `extrinsics`: 4x4 camera-to-world matrix
- `points_cam`: (N, 3) camera frame points
- `points_cam_hom`: (N, 4) homogeneous coordinates
- `points_world`: (N, 3) world frame points

Homogeneous coordinates 让 affine transformation（旋转+平移）可以用单一 matrix 乘法表示。

### 10.3 Top-down Grasp 几何约束

```python
world_z = np.array([0, 0, 1])
g_world = cam_to_world @ g_camera  # grasp pose 转 world frame
R = g_world[:3, :3]
gripper_approach = R[:, 2]  # gripper Z axis in world frame
alignment = -np.dot(gripper_approach, world_z)
```

变量含义：
- `R[:, 2]`: 旋转矩阵第三列，代表 gripper local Z 轴在 world frame 中的方向
- `world_z = [0, 0, 1]`: world up direction
- `-dot(gripper_approach, world_z)`: 如果 gripper Z 朝下（= -world_z），dot = -1，取负 = 1
- `vertical_threshold = 0.8`: cos(37°) ≈ 0.8，允许 ~37° 倾斜

**Intuition**: 我们希望 gripper 从上往下抓（top-down），即 gripper Z 轴 = -world Z。`-dot` 让 perfect top-down 时值为 1，方便 thresholding。

### 10.4 Quaternion-Quaternion 顺序问题

代码中反复出现 `wxyz` 和 `xyzw` 转换：

```python
quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
```

这是因为不同库的 convention：
- `viser`, Open3D, ROS: `[w, x, y, z]` (scalar-first)
- SciPy, PyTorch3D: `[x, y, z, w]` (scalar-last)

这种 convention mismatch是 robotics code 中常见的 bug source，CaP-Agent0 的 skill library 把这封装成 helper function 很合理。

### 10.5 GRPO 的 Group Advantage

回到 RL，让我再深入 GRPO 的 intuition：

传统 actor-critic:
$$A(s, a) = r + \gamma V(s') - V(s)$$

需要训练 $V(s)$（critic）。Critic 不准时 advantage 估计就有 bias。

GRPO 的替代方案：
- 对同一 prompt $q$ 采样 $G$ 个 outputs $\{o_1, ..., o_G\}$
- 计算 group 内 mean 和 std
- 标准化每个 reward：$A_i = (r_i - \mu) / \sigma$

**Intuition**: 如果一个 output 在 group 内 reward 高于平均，它就是 "good"，反之 "bad"。这避免了 absolute reward scale 问题，也不需要 critic。代价是 sample efficiency 较低（每个 prompt 需要 $G$ 个 samples），但 verifier 可得时这是 acceptable trade-off。

---

## 11. 实验数据再分析

### 11.1 Figure 18: In-Context Examples 的效果

Appendix C 和 Figure 18 显示，加入 API usage examples 让几乎所有模型提升约 +20%。这暗示当前 LLM 的 API reasoning 能力仍依赖 in-context scaffolding，纯 zero-shot 从 signature 推理仍困难。

### 11.2 Figure 23: Ensemble 减少 Turn Count

M4 (single query) 平均 turn count 比 CaP-Agent0 (ensemble) 高。这说明 ensemble 不只提升 success rate，还提升 efficiency。Reason：ensemble preemptive 处理 failure mode，减少 retroactive debugging cycles。

### 11.3 LIBERO-PRO Task-level 分析

看 Table 6 的细节：

- `Place(bbq_sauce, basket)`: π₀.₅ Pos=1.00, CaP-Agent0 Pos=0.12。π₀.₅ 在这个 task 的位置鲁棒性远超 CaP-Agent0
- `Place(tomato_sauce, basket)`: π₀.₅ Pos=0.00, CaP-Agent0 Pos=0.16

这表明 VLA 在 in-distribution task 上仍很强，CaP-Agent0 的优势在 OOD generalization。

### 11.4 Sim-to-Real 的反常结果

Cube Stack: sim 44% → real 76%。作者解释是 abstract API reasoning 避免 visual gap。但我怀疑还有其他因素：

1. **Real trial count**: 只 25 次，statistical noise 大
2. **Real environment setup**: 可能更可控（固定 lighting, 物体 placement）
3. **Sim 的 physics stochasticity**: sim 中 grasp 可能更 random

需要更多 trial 才能确认这个反常现象的 robustness。

---

## 12. Future Work 与 Open Questions

### 12.1 Hybrid CaP-VLA Policy

作者提到最有前途的方向：CaP 管理 high-level task logic 和 recovery，VLA 处理 low-level execution。这能 combine 两者的优势：
- CaP: long-horizon reasoning, compositionality, debugging
- VLA: contact-rich control, continuous feedback, fast execution

这是一个 hierarchical 系统，类似 TRI LBM (https://arxiv.org/abs/2507.05331 ) 和 Gemini Robotics (https://arxiv.org/abs/2510.03342 ) 的 hybrid 思路。

### 12.2 Optimization-based Control Primitives

作者指出当前依赖 IK 可能 suboptimal：
> "incorporating optimization-based control primitives that allow agents to specify task-level constraints and account for collision avoidance during motion planning, rather than relying solely on inverse kinematics solutions that may be suboptimal when directly interpolated to in joint space."

这意味着引入 trajectory optimization (trajopt) 作为 primitive，让 agent 可以指定 constraints 而非 just goal pose。

### 12.3 Test-Time Compute Scaling

CaP-Agent0 的 ensemble + multi-turn 是一种 test-time compute scaling。Snell et al. (2024: https://arxiv.org/abs/2408.03314 ) 系统化研究了 LLM test-time compute。CaP-X 把这扩展到 embodied agent，但仍有很多 unexplored：

- Tree search (ToT, Yao et al., 2023: https://arxiv.org/abs/2305.10601 )
- Self-consistency (Wang et al., 2023: https://arxiv.org/abs/2203.11171 )
- Learned decoding strategies

### 12.4 Open-Ended Skill Accumulation

当前 skill library 是 batch-synthesized。能否实现 online, open-ended accumulation？类似 Voyager 的 lifelong learning。挑战：

- Skill quality verification
- Skill library pruning
- Avoiding skill conflicts

### 12.5 RL with Multi-Modal Rewards

CaP-RL 用 binary task success 作为 reward。能否用 dense reward（如 distance-to-goal）或 process reward（code correctness）？这可能加速 RL convergence 但也引入 reward engineering 问题。

---

## 13. 总结

CaP-X 是一个 comprehensive framework，系统地探索了 "LLM/VLM 通过 code 控制 robot" 这个 paradigm 的能力和局限。核心贡献：

1. **CaP-Gym**: 提供统一环境，支持 sim 和 real，覆盖 tabletop/bimanual/mobile manipulation
2. **CaP-Bench**: 8 个 tier 的系统化评估，揭示 abstraction/multi-turn/grounding 的影响
3. **CaP-Agent0**: training-free framework，通过 VDM + skill library + ensemble 达到 human-level performance
4. **CaP-RL**: 用 GRPO post-train coding agent，实现 minimal sim-to-real gap

最 deep 的 insights：
- **Test-time compute 可以 compensate abstraction removal**：low-level primitives + multi-turn + ensemble 可以 match high-level single-turn
- **Text grounding > raw image grounding**：当前 VLM 在 coding + image joint reasoning 上有 alignment gap
- **Symbolic-level reasoning 极小化 sim-to-real gap**：agent 在 abstract API level reasoning，避免 visual feature transfer 困难
- **RL 可以修复 causal reasoning failure**：post-RL 模型学会 grasp → place 的因果依赖，从 memorization 转向 dynamic geometric reasoning

整个工作为 embodied coding agent 这个方向奠定了 benchmark 和方法论基础。结合 hybrid CaP-VLA、optimization-based primitives、lifelong skill accumulation，这可能是通往 generalist robot 的 viable path。

参考链接汇总：
- Project page: https://capgym.github.io
- Voyager: https://arxiv.org/abs/2305.16291
- GRPO/DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Code as Policies: https://arxiv.org/abs/2209.07753
- SWE-Bench: https://arxiv.org/abs/2310.06770
- LIBERO-PRO: https://arxiv.org/abs/2510.03827
- BEHAVIOR-1K: https://arxiv.org/abs/2403.09227
- RoboSuite: https://arxiv.org/abs/2009.12293
- π₀: https://arxiv.org/abs/2410.24164
- π₀.₅: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- SAM3: https://arxiv.org/abs/2511.16719
- Molmo 2: https://arxiv.org/abs/2601.10611
- PyRoki: https://arxiv.org/abs/2505.03728
- Eureka: https://arxiv.org/abs/2310.12931
- Text2Reward: https://arxiv.org/abs/2309.11489
- Test-time compute scaling: https://arxiv.org/abs/2408.03314
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Self-Consistency: https://arxiv.org/abs/2203.11171
- AlphaCode: https://arxiv.org/abs/2203.07807
- Qwen2.5-Coder: https://arxiv.org/abs/2409.12186
- Gemini Robotics: https://arxiv.org/abs/2510.03342
- TRI LBM: https://arxiv.org/abs/2507.05331
