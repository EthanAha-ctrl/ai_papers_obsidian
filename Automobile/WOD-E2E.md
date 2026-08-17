---
source_pdf: WOD-E2E.pdf
paper_sha256: 553db25eb2190eda3450672000b80a4841bbf06d1b9b614333cbe351ddf3e0b3
processed_at: '2026-08-13T04:44:09-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# WOD-E2E 用人话说

## 1. 这篇 paper 在讲什么 story

Waymo 自己有 6.39 million miles 的真实驾驶日志。绝大多数是无聊的——上下班、去超市、红灯停绿灯行。但里面藏着那么 0.03% 的"怪场景"：鸟群突然飞过来、滑板车从桥上掉下来、施工区有人指挥交通、雪天对向摩托车冲过来、应急车辆挡路……

这些 0.03% 才是自动驾驶真正的考验。你 99.97% 的场景做得再丝滑，撞上一次 0.03% 就上新闻了。

所以 Waymo 做的事很朴素：**把这 0.03% 挖出来，做成一个专门考 E2E driving model 的"难题集"，同时发明一个能合理评分的 metric**。这两件事加起来就是 WOD-E2E。

---

## 2. 挖数据这件事本身就很 engineering

6.39M miles 怎么变成 0.03%？两步漏斗：

**第一步**：rule-based + MLLM 自动 mining，从 6.39M miles 筛到 6,888 miles（0.1%）。rule 靠 Waymo 内部的 auto-labels——3D detection、mapping、tracking、prediction——做 heuristics。比如"前方 N 米内有 construction cone 且 ego velocity 在 X 范围"这种规则。MLLM（具体是 Gemini）辅助做 semantic 理解，比如 Spotlight cluster 专门用 Gemini 去 open-vocabulary 搜"长尾物体"。

**第二步**：human filtering，30% 通过率。也就是说人看了之后 70% 觉得"这其实不算 long-tail"给剔了。最后留下 0.03%。

这个 conversion rate 数据其实挺有意思的——说明自动 mining 的 precision 大概 30%。如果 recall 也很低，那意味着真实 long-tail 可能比 0.03% 还多，但 Waymo 选择高 precision 路线保证质量。

11 个 cluster 分类很 Waymo 风格：construction、intersection、pedestrians、cyclists、cut-ins、FOD（foreign object debris，就是路上乱七八糟的东西）、special vehicles、spotlight（人工挑的）、others。每个 cluster 都有详细的 mining criteria 在 Table 1 里。

---

## 3. 为什么传统 metric 是 broken 的

这里其实是 paper 最有价值的地方，需要好好讲。

### 3.1 ADE 的问题

ADE（Average Distance Error）就是：你 model 猜的 trajectory 和 ground truth trajectory 的 L2 距离。

听起来合理，但实际 driving 是 **multi-modal** 的。比如前方有个倒地的滑板车，人类驾驶员可能：
- A: 紧急刹车
- B: 向左 nudge 绕过去
- C: 向右 nudge 绕过去

三个都合理。但 dataset 里 ground truth 只记录了当时那个 driver 选的那一条，比如 A。如果你的 model 选了 B（也很合理），ADE 会狠狠扣分——因为它和 A 的 L2 距离很大。

这就 broken 了。model 学到的反而是"猜中那条 specific trajectory"，而不是"理解场景做出合理决策"。

### 3.2 PDMS 的问题（NAVSIM 用的）

PDMS（Predictive Driver Model Score）更复杂，考虑 comfort、TTC（time to collision）、ego progress 等。但 paper 指出几个问题：

- **TTC 对 amorphous obstacles 无解**：鸟群怎么算 collision？鸟不是刚性 box，你 detection 都做不出来，更别说 TTC。
- **Off-road 高惩罚不合理**：紧急情况下 partial off-road（比如右轮上人行道避开 falling object）是合理的，但 PDMS 会狠狠扣分。
- **Comfort 在 safety-critical 场景下是次要的**：紧急避让时舒适度本来就要牺牲，PDMS 仍然把 comfort 算进 score。

### 3.3 RFS 的思路

RFS（Rater Feedback Score）的核心 idea：**找 3 个人类专家，让他们对每个场景给 3 条 candidate trajectory 打 0-10 分，然后看 model 猜的轨迹和哪条高分 trajectory 对齐**。

这个设计的妙处在于：
1. **Multi-modal ground truth**：3 条 trajectory 覆盖不同的合理驾驶 mode，model 只要对齐其中一条高分 trajectory 就行，不再被 single ground truth 绑架。
2. **Human judgment implicitly 处理了 edge case**：rater 看到鸟群不会去算 TTC，他会判断"这个轨迹是否合理避开"。
3. **Off-road 行为不再机械扣分**：rater 会根据场景判断 off-road 是否合理。

---

## 4. RFS 公式怎么算的（人话版）

设你 model 猜了一条 trajectory（蓝色），dataset 里有 3 条 rater-rated trajectory，分别打了分 $s_{rater,1}, s_{rater,2}, s_{rater,3}$（0-10）。

**Step 1**: 对每条 rater trajectory，画一个 "trust region"——一个长方形，longitudinal 4 米、lateral 1 米（在 t=3s 时刻）。这个 region 的大小会随速度 scale。

**Step 2**: 看 model prediction 在 t=3s 时落在哪条 rater trajectory 的 trust region 里。如果在某条里面，就给那条的 $s_{rater}$ 分。如果在外面，按超出比例做指数衰减：$s_{rater} \times 0.1^{\text{exceed ratio}}$。

**Step 3**: 对 3 条 rater trajectory 各算一个 score，取 max（最宽容的那条）。

**Step 4**: 对 t=3s 和 t=5s 两个时间点各算一次，取平均。

**Step 5**: Floor 到 4 分（避免极端低分）。

直觉上：**RFS 就是"你的 prediction 和最匹配的那条人类认可 trajectory 有多近"**。近的话拿满分，远的话指数衰减，太远就 floor 到 4 分。

### 4.1 Speed scaling 的细节

$$\text{scale}(v) = \begin{cases} 0.5, & v < 1.4 \\ 0.5 + 0.5 \times \frac{v - 1.4}{9.6}, & 1.4 \leq v < 11 \\ 1, & v \geq 11 \end{cases}$$

- $v < 1.4$ m/s（≈5 km/h，步行速度）：region 缩到一半。低速时该精确——你在停车场慢行还偏离 1 米就不合理。
- $v > 11$ m/s（≈40 km/h，城市道路速度）：region 全尺寸。高速时不确定性大，宽容些。
- 中间线性过渡。

这个设计其实是 **physically grounded adaptive threshold**，比固定 L2 距离合理得多。控制论里本来就有"速度越高，控制不确定性越大"的常识，RFS 把这个常识编进了 metric。

### 4.2 为什么 longitudinal 是 lateral 的 4 倍

$\bar{\tau}_{lng} = 4 \bar{\tau}_{lat}$，这是因为：
- Longitudinal 方向（前后）不确定性主要来自速度估计误差，速度一差，3 秒后位置可能差几米
- Lateral 方向（左右）不确定性主要来自 steering，量级小很多

所以 trust region 是个长方形，长边沿 driving direction。这比圆形 region 更符合 driving physics。

---

## 5. NaiveEMMA baseline 的刻意简化

NaiveEMMA 是 EMMA 的简化版，paper 明确说"crucially, it does not use past camera frames"。只看当前一帧的 8 个 camera（拼成 768×768 一张大图）+ 3s ego status history + high-level routing command。

这是 **deliberate**。目的是做公平 baseline，隔离 dataset 和 metric 的贡献，而不是靠 model architecture 刷分。EMMA 原版有 CoT、有 generalist task training、有 test-time scaling，NaiveEMMA 都砍掉了。

参数量也没特别大，fine-tune 自 Gemini Flash，只在 WOD-E2E training split 上训。

---

## 6. Leaderboard 上最有意思的发现

### 6.1 Poutine 赢麻了

Poutine 是 MLLM-based（Qwen2.5 3B），加 CoT reasoning，加 RL post-training，**关键是用 RFS 作为 RL reward**。RFS 7.986，比第二名高 0.21。

对比 AutoVLA：也是 MLLM + RL，但用 ADE 作为 RL reward，RFS 只有 7.556。

这两个对比直接回答了"RL 在 E2E driving 里有没有用"——**有用，但 reward 必须和 target metric 对齐**。用 ADE 做 reward 等于让 model 优化错误目标，最后 RFS 上不去。

这是个很强的 lesson。Karpathy 你自己老说"reward is everything"，这就是 reward design 在 autonomous driving 上的实证。

### 6.2 MLLM vs Diffusion 在 cross-dataset transfer 上的差异

- **MLLM-based** 加额外 dataset 有明显 gain（Poutine 加 CoVLA，AutoVLA 加 nuPlan + nuScenes）
- **Diffusion-based** 加额外 dataset gain 很小（DiffusionLTF 加了 4 个 dataset 反而比只加 1 个的 UniPlan 差）

Paper 的 hypothesis：MLLM 的 CoT reasoning 能学抽象 driving knowledge，对 visual / geometric distribution shift 鲁棒。Diffusion 依赖 dense pixel-level prediction，对 visual shift 敏感。

这个观察让我想到一个更深的点：**MLLM 可能是 E2E driving 的 better inductive bias**，因为 driving 本质上是"理解场景 → 推理决策 → 执行"，而不是"pixel → pixel prediction"。CoT 给了 model 一个 explicit 的 reasoning scaffold，让它能 leverage 预训练的 world knowledge。

### 6.3 ADE 和 RFS 弱相关

Paper 画了 19 个 submission 的 RFS vs ADE 散点图（Table 2 右），只有 mild positive correlation。

具体例子：
- WayNet: ADE 2.8（顶尖）但 RFS 显著低于多数 submission
- HMVLM: ADE 比多数差，但 RFS 接近 top

这说明 ADE 会骗你。一个 model 可以 ADE 很好（猜中 ground truth trajectory 的 L2 距离小）但 RFS 很差（选错了 mode）。反之亦然。

**RFS 的存在是必要的**——ADE 单独无法告诉你 model 在 long-tail 上是真的好还是 over-fit 到 single ground truth。

---

## 7. 几个值得深挖的 design choice

### 7.1 RFS 为什么是 3 条 trajectory 而不是 5 条或 10 条

Paper 没明说，但可以推断：
- 3 条覆盖 left/straight/right 三个 mode 够用了
- 更多条会增加 labeling 成本（每个场景要更多 rater 评分）
- 3 条是 multi-modal coverage 和 cost 的 sweet spot

### 7.2 Critical frame 的"已经开始行动"约束

Labeler 选 critical frame 时被要求"选 ego 已经开始 take action 的 frame"，避免 reaction bias。这个细节很重要：

如果你选 ego 还没动的 frame，model 可能学到"看到鸟群 → 等 N 秒再刹车"，因为 history motion 信息暗示了时机。但真实驾驶里 model 不知道人类何时会行动。选 ego 已开始动的 frame，相当于告诉 model"这是决策时刻，你现在就该动"。

这是 open-loop evaluation 里一个常见的 pitfall，WOD-E2E 显式处理了。

### 7.3 Trajectory scoring 的 base-10 扣分制

- Major infraction（safety/reaction/legality）: -2
- Minor infraction（braking/efficiency）: -1

从 10 分起扣。强制至少一条 trajectory > 6（视为 safe and feasible 下限）。

从 Figure 6 看，Rank 1 trajectory 最低就是 6 分，Rank 3 分布在 0-6。这说明 rater 之间是有共识的：最优 trajectory 一定 ≥6，sub-optimal 的可能很低。

这个 6 分阈值的设计巧妙：它给 RFS 的 max 操作提供了一个 implicit baseline——如果 model 对齐的 trajectory 只有 4 分，那即使对齐得再准，RFS 也上不去。model 必须对齐到那条 ≥6 分的 trajectory 才能拿到高分。

---

## 8. 和 NAVSIM 的对比再深一点

NAVSIM 是 NeurIPS 2024 的工作，建立在 nuPlan 过滤版上。它的两个核心 limitation 被 WOD-E2E 直接针对：

1. **NAVSIM 是 filter 而非 collect**：从已有数据里过滤出"有挑战"的，但没有 targeted collection。WOD-E2E 是 targeted mining from 6.39M miles，能覆盖更 nuanced 的 long-tail。

2. **PDMS 优先 comfort 和 ego progress**：在 safety-critical 场景下这个优先级是错的。紧急避让时 comfort 不重要，ego progress 也不重要（停下来避开 bird flock 是合理的，但 PDMS 会扣 progress 分）。RFS 通过 human judgment 隐式处理这些 trade-off。

这里有个更深的哲学问题：**到底该用 programmatic metric 还是 human judgment？**

- Programmatic metric（ADE、PDMS）：可复现、可大规模计算、无主观性
- Human judgment（RFS）：能处理 edge case、符合直觉、但贵且可能有 rater bias

WOD-E2E 的答案是：**在 long-tail 场景下，human judgment 不可替代**。因为 long-tail 的本质就是"规则覆盖不到"，programmatic metric 必然 broken。

---

## 9. Open-loop 的 limitation

Paper 自己承认：open-loop setup 是 limitation。原因是没有 affordable realistic sensor simulation。Closed-loop 需要重新 render camera、模拟其他 agent 反应，计算成本太高。

但 paper 提了一个有意思的点：**long-tail real-world scenarios 可以用来测试 high-fidelity simulator 的 generalizability**。也就是说，simulator 如果能在 WOD-E2E 这些场景上"看起来像真的"，那它就可以用作 closed-loop testbed。

这其实暗示了未来的方向：**neuro-simulator + real long-tail scenarios = closed-loop evaluation**。用真实 long-tail 数据 validate simulator，再用 simulator 做 closed-loop test。Waymo 内部肯定在搞这个。

---

## 10. 一些联想和延伸

### 10.1 RFS 和 RLHF 的相似性

RFS 本质上是 **trajectory-level preference learning**。3 条 rated trajectory 类似 RLHF 里的人类 preference pair，只是这里用绝对评分 0-10 而非 pairwise comparison。

未来可以想象：
- 用 RFS labels 训一个 reward model
- 这个 reward model 在 RL 里给 model 的 trajectory 打分
- 不需要再人工标注，可以 scale

这就是 Poutine 已经做的事（用 RFS 做 RL reward），但如果训一个 RFS predictor，可以 further scale。

### 10.2 Long-tail mining 的 meta-pattern

WOD-E2E 的 mining pipeline（rule + MLLM + human filter）其实是一个通用 pattern：

```
大 corpus
  → cheap automatic filter (rule / classifier)
  → MLLM semantic filter
  → human quality filter
  → 高质量小 dataset
```

这个 pattern 在 data curation 里越来越常见。Anthropic、OpenAI 做 RLHF data 也是类似流程。Waymo 这篇 paper 把这个 pattern 应用到 autonomous driving data engine，效果显著。

### 10.3 MLLM 在 E2E driving 里到底为什么 work

Poutine 的成功（RFS 7.986）让我思考 MLLM 的优势本质是什么：

- **Not 参数量**：Qwen2.5 3B 比 DiffusionDrive 60M 大，但不是决定性因素
- **Not 多模态 fusion**：Diffusion 也能 fuse image + state
- **是 CoT reasoning**：MLLM 能在 predict trajectory 前做 explicit reasoning，把场景理解、风险评估、决策逻辑显式化

这暗示 E2E driving 可能本质上是个 **reasoning task** 而非 **pattern matching task**。Long-tail 场景之所以难，是因为它们无法被训练数据覆盖，需要 model 在线推理。MLLM 的 reasoning capability 恰好填补这个 gap。

Diffusion models 擅长 multi-modal generation（生成 diverse trajectory），但不擅长 reasoning。这是它们在 long-tail 上被 MLLM 拍的根本原因。

### 10.4 为什么 floor 是 4 而不是 0

Paper 没明说，但从 Figure 6 推断：Rank 3 trajectory 众数在 4-6 之间。4 大概是 "很差但不算完全错" 的下限。

如果 floor 是 0，那 off-distribution prediction 会把 average 拉到很低，noise 太大。floor=4 保证 RFS 有意义的高分区间在 [4, 10]，6 分是 "safe and feasible" 阈值，这样 model 间的差异更明显。

---

## 11. 最核心的 intuition 总结

用一句话讲 WOD-E2E：**用 0.03% 的 real long-tail 数据 + human-rated multi-modal trajectory preference 来 forcing E2E driving model 学真正的 reasoning，而不是 pattern matching**。

三个关键 insight：

1. **Long-tail 的 scarcity**：6.39M miles 里只占 0.03%，但决定 system 安全性。需要 targeted mining 而非 random collection。

2. **Metric alignment 决定 RL 成败**：用 ADE 做 reward 训 RL，model 学错；用 RFS 做 reward 训 RL，model 学对。这和 InstructGPT 用 human preference 训 RM 再 RL 是同一个 lesson。

3. **MLLM + CoT 是 long-tail 的 better inductive bias**：因为 long-tail 本质是"训练数据覆盖不到，需要在线推理"，而 CoT 给了 model 一个 reasoning scaffold。

最后一点 personal opinion：这篇 paper 的 contribution 表面上是 dataset + metric，但深层的 contribution 是**证明了 human preference 在 E2E driving evaluation 里不可替代**。所有试图用 programmatic metric 逃避 human judgment 的努力，在 long-tail 上都会 broken。这和 RLHF 在 LLM 上的 story 几乎一模一样。

参考链接：
- WOD-E2E Challenge: https://waymo.com/open/challenges/
- EMMA paper: https://arxiv.org/abs/2410.23265
- NAVSIM: https://github.com/autonomousvision/navsim
- Poutine: https://arxiv.org/abs/2506.11234
- AutoVLA: https://arxiv.org/abs/2506.13757
- DiffusionDrive: https://github.com/hustvl/DiffusionDrive

---

# WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios 深度解析

## 1. Paper 核心定位

这篇 paper 来自 Waymo 团队，瞄准的是当前 E2E driving 研究的一个核心痛点：**现有 dataset 都是 nominal scenarios，无法真正考验 E2E 系统在 long-tail 事件上的 robustness**。三个核心 contribution：

1. **WOD-E2E dataset**：4,021 个 driving segments，~12 小时，所有 segments 均为 long-tail scenarios，在 6.39M miles 原始数据中只占 **0.03%**
2. **Rater Feedback Score (RFS)**：human-aligned open-loop metric，解决 ADE / PDMS 的 multi-modal 缺陷
3. **Baseline NaiveEMMA + Leaderboard 分析**：覆盖 MLP-based / Diffusion-based / MLLM-based 三类方法

参考链接：
- arXiv: https://arxiv.org/abs/2508.15125 (推测)
- Waymo Open Dataset: https://waymo.com/open/
- WOD-E2E Challenge: https://waymo.com/open/challenges/

---

## 2. Dataset 详细解析

### 2.1 规模与 Split

| 项目 | 数值 |
|------|------|
| Total segments | 4,021 |
| Total duration | ~12 hours |
| Train split | 2,037 |
| Validation split | 479 |
| Test split | 1,505 |
| Segment length | 20 seconds |
| Camera frequency | 10 Hz |
| Camera count | 8 (360° coverage) |
| Long-tail frequency | < 0.03% |

Train 数据含 20s 完整数据，Test 数据仅含 12s + 隐藏的 8s future trajectory 用于 evaluation。

### 2.2 Coordinate System

**Vehicle Coordinates**（所有 trajectory 数据的参考系）：
- 原点：ego vehicle center
- x-axis: forward
- y-axis: left
- z-axis: upward

**Sensor Frame**（每相机独立）：
- 原点：lens center
- x-axis: out from lens
- y/z plane: parallel to image plane
- 右手系

通过 extrinsic transformation 与 vehicle frame 对齐。这个设计的关键意义是：**任何 3D trajectory 都可以 project 到 8 个 camera 图像上做可视化或 training**，这对 MLLM-based methods 尤其重要。

### 2.3 输入模态

**Camera Data**：
- 8 cameras: front, front-left, front-right, side-left, side-right, rear, rear-left, rear-right
- 提供 intrinsics + extrinsics
- 注意 NaiveEMMA 把 8 个 camera 拼接成单张 768×768 image，而非 temporal stacking

**Routing Information**：
- 高级 command: {GO_STRAIGHT, GO_LEFT, GO_RIGHT}
- 由 10s 未来 driven route 推导
- 仅表示在 decision point 的分支方向
- **不**包含 micro maneuvers（lane change / nudge）
- **不**包含 speed profile

**Ego Status**：
- Past trajectory: 4s 历史，4Hz 频率，waypoints $[(x_1, y_1), (x_2, y_2), ...]$
- Velocity + Acceleration（aligned with past trajectory）
- Future trajectory: 5s future（仅 train/val 可见）

### 2.4 Long-tail Mining 流程

这是 dataset 的关键工程贡献。流程：

```
6,391,012 miles 原始日志
   ↓ Rule-based + MLLM 自动 mining
6,888 miles (0.1%) — 自动 mining 阈值通过
   ↓ Human filtering (30% conversion rate)
~2,066 miles (0.03%) — 最终 WOD-E2E
```

**11 个 scenario cluster**：

1. **Construction**: road closure, uniformed personnel, abnormal road surface
2. **Intersection**: unprotected maneuvers, stop sign interactions, traffic-violating agents, rails/cable cars
3. **Pedestrians**: low-visibility crossings, unexpected behaviors, unsafe maneuvers
4. **Cyclists**: losing control, group interactions
5. **Multi-Lane Maneuvers**: nudge to overtake, freeway merging, hazards from neighboring lane
6. **Single-Lane Maneuvers**: overtake on narrow roads, open-door vehicle interactions
7. **Cut-ins**: oncoming cut, aggressive lane change
8. **Foreign Object Debris (FOD)**: animals, large debris (box/glass/metal), abnormal road conditions (flood/fire)
9. **Special Vehicles**: emergency vehicles blocking road, pull-over required
10. **Spotlight**: 用 Gemini 搜索特定 long-tail objects
11. **Others**: 不属于以上类别

**Mining 的关键洞察**：依靠 Waymo 内部 rich auto-labels（3D detection、mapping、tracking、prediction）做 rule-based heuristic，再叠加 MLLM 做 semantic 理解。Spotlight cluster 专门留给 Gemini 做 open-vocabulary 搜索——这是 MLLM 在 data engine 中的一个具体应用。

### 2.5 数据分布

**City distribution**：主要来自 cities L, K, J。Test set 含额外 cities 以测试 generalization。

**Scenario cluster × road type**：
- Intersection, FOD, Pedestrians 占最大份额
- Freeway 在 Cut-ins cluster 中占主导（high-speed 安全关键）
- Freeway 在 Intersections 中也存在（ freeway on/off-ramp 交互）

**Driving behaviors**：
- 直行（含 hard braking / swerving）: 主导
- 左右转: ~30%（左右比例相近）
- Lane change: 10.3%
- On-ramp: 1.7%

---

## 3. Rater Feedback Score (RFS) — 深入技术细节

这是 paper 最核心的方法学创新。让我们拆解每一步。

### 3.1 Labeling Pipeline

**Step 1: Critical Moment Selection**

Labeler 三步流程：
1. **High-level Understanding**: 扫描整段 20s 视频，识别 critical event 和正确决策
2. **Moment Selection**: 找最早出现 critical event 视觉线索的 frame。关键约束：**选择 ego vehicle 已经开始行动的 frame**，避免反应偏差（reaction bias）来自历史运动信息
3. **Reasoning Documentation**: 简短记录选择该 frame 的理由

**Step 2: Trajectory Sampling**

使用 Wayformer（motion forecasting model）生成至多 64 条 diverse trajectories：
- 输入：perception detections, mapping elements, predicted behaviors of other agents
- 输出：candidate trajectories

Trajectory selection 两步：
1. **Automatic bucketing**: 按 velocity, acceleration, lane change 分桶
2. **Human sampling**: labeler 从中选 < 12 条，覆盖 leftmost / middle / rightmost 的 lateral 谱

**Step 3: Trajectory Scoring**

Rater 选 3 条 diverse trajectories 进行评分（强制至少一条 optimal）。

5 个评分维度：
- **Safety**: collision, near-miss, unsafe conditions
- **Legality**: traffic laws, emergency vehicle 行为
- **Reaction Time**: 响应及时性
- **Braking Necessity**: 是否有非必要 / 过保守 braking
- **Efficiency**: 进展效率，避免不必要 lane change / hesitation

**Scoring mechanism**:
- 初始 base score: 10 分
- Major infraction（safety / reaction time / legality）: -2 分
- Minor infraction（braking necessity / efficiency）: -1 分
- Multiple violations 可叠加 discretionary deduction
- 强制至少一条 trajectory 评分 > 6（视为 safe and feasible 下限）

### 3.2 RFS 公式详解

RFS 设计目标是衡量 **model predicted trajectory 与 3 条 human-rated reference trajectories 的 alignment**。

#### 3.2.1 Trust Region 定义

对每个 rater trajectory，在其周围定义 rectangular trust region。两个时间点：$t \in \{3, 5\}$ 秒。

**Base thresholds**（沿用 WOMD）：
- $t = 3$: $\bar{\tau}_{lat} = 1.0$ m, $\bar{\tau}_{lng} = 4.0$ m
- $t = 5$: $\bar{\tau}_{lat} = 1.8$ m, $\bar{\tau}_{lng} = 7.2$ m

设计直觉：
- $\bar{\tau}_{lng} / \bar{\tau}_{lat} = 4$：longitudinal 容忍度是 lateral 的 4 倍（因为速度方向上的不确定性更高）
- $t = 5$ 时阈值大于 $t = 3$：未来越远，不确定性越大

#### 3.2.2 Speed-dependent Scaling

**关键创新**：base threshold 根据 rater trajectory 的初始速度 $v$（m/s）做 piece-wise linear scaling：

$$
\text{scale}(v) = 
\begin{cases} 
0.5, & v < 1.4 \\
0.5 + 0.5 \times \frac{v - 1.4}{11 - 1.4}, & 1.4 \leq v < 11 \\
1, & v \geq 11 
\end{cases}
$$

**变量解释**：
- $v$: rater trajectory 的初始速度（m/s）
- $1.4$ m/s ≈ 5 km/h（典型步行速度，低速阈值）
- $11$ m/s ≈ 40 km/h（典型城市道路速度，高速阈值）
- $0.5$：低速时 trust region 缩小一半（精确性更重要）
- $1.0$：高速时 trust region 全尺寸（不确定性更大）

**Piece-wise linear design**:
- $v < 1.4$: 常数 0.5
- $1.4 \leq v < 11$: 线性从 0.5 上升到 1
- $v \geq 11$: 常数 1

Final thresholds:
$$
\tau_{lng} = \text{scale}(v) \times \bar{\tau}_{lng}, \quad \tau_{lat} = \text{scale}(v) \times \bar{\tau}_{lat}
$$

#### 3.2.3 单条 Rater Trajectory 的 Score 计算

对 prediction 与某条 rater trajectory 之间的 longitudinal / lateral 距离误差 $\Delta_{lng}, \Delta_{lat}$：

$$
s_{rater} \times 0.1^{\max\left\{\max\left\{\frac{\Delta_{lng}}{\tau_{lng}}, \frac{\Delta_{lat}}{\tau_{lat}}\right\} - 1, 0\right\}}
$$

**变量**：
- $s_{rater}$: 该 rater trajectory 的人工评分 $[0, 10]$
- $\Delta_{lng}$: prediction 与 rater trajectory 在 longitudinal 方向上的距离误差
- $\Delta_{lat}$: lateral 方向上的距离误差
- $\tau_{lng}, \tau_{lat}$: 经过 speed scaling 的 trust region 边界
- $0.1$: exponential decay base

**核心直觉分析**：
- 当 $\max\{\Delta_{lng}/\tau_{lng}, \Delta_{lat}/\tau_{lat}\} \leq 1$（即 prediction 在 trust region 内）
  - $\max\{... - 1, 0\} = 0$
  - $0.1^0 = 1$
  - Score = $s_{rater}$（flat score）

- 当 $\max\{.../\tau\} > 1$（prediction 在 trust region 外）
  - exponent $> 0$
  - $0.1^{\text{exponent}} < 1$
  - Score 指数衰减
  - 例如：当 ratio = 2（在 region 边界 2 倍外），score 衰减为 $s_{rater} \times 0.1^1 = 0.1 \times s_{rater}$

这种设计 **smoothly penalize** 离 rater trajectory 越远的 prediction。

#### 3.2.4 Final RFS 组合

最终 RFS 通过三步组合：

1. **Max over 3 rater trajectories**: 取 prediction 与 3 条 rater trajectories alignment 中最高的 score
2. **Average over $t \in \{3, 5\}$**: 对 3s 和 5s 两个时间点的 score 取平均
3. **Floor = 4**: 最低不低于 4 分

设计直觉：
- **Max**: 容忍 multi-modal behavior——只要 prediction 对齐到至少一条合理 rater trajectory 即可
- **Average**: 同时考虑 short-term（3s）和 long-term（5s）的 alignment
- **Floor = 4**: 防止极端低分，为 off-distribution prediction 提供基线

### 3.3 RFS vs 传统 Metrics 对比

| Metric | 优点 | 缺点 |
|--------|------|------|
| **ADE** | 简单 | 只与单条 ground truth 比较；忽略 multi-modal |
| **PDMS** (NAVSIM) | 考虑 comfort, TTC | 难以处理 amorphous obstacles (flock of birds)；off-road 高惩罚不利 safety-critical；需要 agent annotation |
| **RFS** | Human-aligned; multi-modal; speed-adaptive | 需要人工评分成本高；依赖 rater quality |

RFS 的关键优势：
- **不依赖 agent annotation**：FOD scenarios 中的 flock of birds 等难以检测的 objects 不影响 RFS 评估
- **容许 off-road 行为**：在 emergency maneuver 中，AV 可能合理地偏离 road，RFS 通过 rater judgment 自然处理这种情况
- **多模态 ground truth**：3 条 trajectory 捕捉驾驶行为的多解性

---

## 4. Baseline NaiveEMMA 架构解析

### 4.1 架构特征

NaiveEMMA 是 EMMA [14] 的简化版本：

**Input**:
- 8 cameras 拼接成单张 768×768 image（不做 temporal stacking）
- 3s past ego-status history
- High-level routing command (GO_STRAIGHT/LEFT/RIGHT)

**Output**:
- 5s future trajectory

**Backbone**:
- Fine-tune from Gemini Flash [6]
- **不**使用 internal driving datasets，仅 WOD-E2E training split

**Omitted components**（vs original EMMA）:
- Generalist task training mixtures
- Chain-of-Thought (CoT) reasoning
- Test-time scaling methods

### 4.2 设计哲学

NaiveEMMA 的简化是 deliberate choice——**作为公平 baseline，隔离 WOD-E2E dataset 和 RFS metric 的贡献**。它不依赖过去 camera frames（只用 current frame），简化了 architectural complexity。

---

## 5. Experimental Results 详解

### 5.1 RFS Validation (Table 3)

| Configuration | RFS |
|--------------|-----|
| Baseline | 7.14 |
| + WOD-E2E finetuning | 7.22 |
| + Multi-camera inputs | 7.30 |
| + Test-time scaling | 7.39 |

**Intuition**:
- Long-tail exposure: +0.08（暴露 long-tail scenarios 显著提升）
- Multi-camera: +0.08（360° 理解对 long-tail 重要）
- Test-time scaling: +0.09（处理 scenario ambiguity 的 multi-sampling 有效）

这表明 RFS **确实能区分**不同 capability 的 model，符合直觉。

### 5.2 Leaderboard 提交（Table 2）

| Method | RFS ↑ | ADE ↓ | Training | Backbone | Params |
|--------|-------|-------|----------|----------|--------|
| **MLP-based** | | | | | |
| Swin-Trajectory | 7.543 | - | 1 SFT | Swin Transformer | 36M |
| **Diffusion-based** | | | | | |
| DiffusionLTF | 7.717 | 2.977 | 4 SFT | DiffusionDrive | 60M |
| UniPlan | 7.779 | 2.986 | 2 SFT | DiffusionDrive | 60M |
| **MLLM-based** | | | | | |
| Baseline | 7.528 | 3.018 | 1 SFT | Gemini Nano | 3B |
| AutoVLA | 7.556 | 2.958 | 3 SFT+RL | Qwen2.5 | 3B |
| HMVLM | 7.736 | 3.071 | 1 SFT | Qwen2.5 | 3B |
| Poutine | **7.986** | 2.741 | 2 SFT+RL | Qwen2.5 | 3B |

**关键观察**：
1. **Poutine (MLLM + RL with RFS reward) 取得最高 RFS 7.986**
2. Diffusion-based models 在 RFS 上略优于 baseline MLLM (UniPlan 7.779 > Baseline 7.528)
3. 参数量并非决定性因素（Swin-Trajectory 36M 仍能达 7.543）

### 5.3 三个 Research Questions 深度分析

#### Q1: Extra data source helpful for E2E Driving?

**Answer: It depends on architecture.**

**MLLM-based (Poutine, AutoVLA)**：添加外部 dataset 显著提升
- Poutine: WOD-E2E + CoVLA
- AutoVLA: WOD-E2E + nuPlan + nuScenes
- 两者均获明显 RFS gain

**Diffusion-based (UniPlan, DiffusionLTF)**：改进有限
- DiffusionLTF: WOD-E2E + CARLA + NAVSIM + WOD-Perception (4 个 dataset)
- UniPlan: WOD-E2E + nuPlan (2 个 dataset)
- DiffusionLTF (4 dataset, RFS 7.717) < UniPlan (2 dataset, RFS 7.779)

**Hypothesis**: CoT reasoning 让 MLLM 能利用跨 dataset 的 abstract world knowledge，对 geometric / visual distribution shift 鲁棒。Diffusion models 依赖 dense pixel-level prediction，对 visual distribution shift 敏感。

**Intuition**: 这暗示 MLLM 的 reasoning 是 transfer learning 的关键 mechanism，diffusion models 的 representation 仍是 local / dataset-specific。

#### Q2: Better ADE → Better RFS?

**Answer: No.**

证据：
- WayNet: ADE = 2.8 (top-tier) 但 RFS 显著低于多数提交
- HMVLM: ADE > 多数提交，但 RFS 接近 top

**直觉解释**：
- ADE 衡量与单条 ground truth 的 L2 距离——可能 over-fitting 到一条 specific trajectory
- RFS 衡量与 human preference 的 alignment——容忍 multi-modal 解决方案
- 一个 model 可以 "ADE 低" 但选错 mode（保守但错），或者 "ADE 高" 但选对 mode（合理但与 ground truth 速度不同）

这强烈支持了 RFS 的必要性。

#### Q3: RL effective in E2E Driving?

**Answer: Yes, when reward is aligned with target metric.**

| Model | RL Reward | RFS Gain |
|-------|-----------|----------|
| AutoVLA | ADE | +0.028 (vs Baseline 7.528) |
| Poutine | RFS | +0.250 (vs HMVLM 7.736) |

**Key insight**: RL with RFS reward (Poutine) 比 RL with ADE reward (AutoVLA) 提升**显著更大**。这进一步证明：
- Reward signal alignment 极其重要
- ADE 作为 reward 会导致 model 优化错误目标
- RFS 作为 reward 直接对齐 evaluation metric，提供 better gradient direction

---

## 6. 与现有 Dataset 的对比

| Dataset | Long-tail focus | Multi-modal metric | Camera | Limitation |
|---------|-----------------|-------------------|--------|------------|
| nuScenes [2] | No | ADE only | Multi | Simple extrapolation 即达 strong performance [16, 39] |
| NAVSIM [8] | No (filtered) | PDMS | Multi | Simulation-based; PDMS 不充分 for safety-critical |
| WOMD [10] | No | ADE | Embeddings only (无 full image) | Motion prediction 为主 |
| CoVLA [1] | General diversity | - | Multi | 无 specific long-tail over-sampling |
| **WOD-E2E** | **Yes (0.03%)** | **RFS** | **8 cameras** | **Open-loop only** |

### 6.1 NAVSIM 的两个关键 limitation

1. **数据采集无 long-tail 特化**：NAVSIM 是 nuPlan 的过滤版本，依赖 filtering 而非 targeted collection。可能错过 nuanced diversity of long-tail scenarios。

2. **PDMS 不足**：
   - 优先 ego progress 和 comfort（safety-critical 时 comfort 是次要的）
   - TTC 对 amorphous obstacles（flock of birds）难以计算
   - Off-road 行为高惩罚，但 emergency 时 partial off-road 可能合理

### 6.2 WOMD 的限制

WOMD 是 motion prediction dataset，**只提供 image embeddings 而非 raw camera images**，外部 researcher 难以做 full E2E research。

---

## 7. Rarity Comparison 量化

用 Gemini 2.5 Pro [6] 对各 dataset test set 做统一 rarity scoring：
- 输入：front camera sequence + scoring prompt
- 输出：rarity_score (0-100), rare_factors, reasoning trace
- Rarity tiers: 4 个 based on complexity, risk, long-tail factors

**Results (Figure 3 left)**：
- WOD-E2E 在所有 percentage tiles 上显著高于其他 dataset
- 最极端 10% 数据：WOD-E2E 平均 rarity score ~93
- Full dataset average 也保持高 rarity

这是**首次用 LLM 做统一 rarity benchmark**——MLLM 作为 universal evaluator 在 dataset analysis 中的应用。

---

## 8. Critical Intuitions 总结

### 8.1 Long-tail 的 0.03% 数据究竟意味着什么？

6.39M miles 中只有 0.1% 通过自动 mining，再经过 human filtering 30% 转化率 → 0.03%。这暗示：
- 自动 mining 是 high-precision low-recall 的（先过滤到 0.1%）
- Human filtering 进一步保证 quality（70% 被剔除）
- 最终 long-tail 在真实驾驶中是 **极稀有但极重要** 的事件

### 8.2 RFS 为什么用 piece-wise linear 而非线性？

Piece-wise linear 在 $v \in [1.4, 11]$ 之间过渡的设计考虑：
- 低速（$v < 1.4$）时 trust region 不再缩小：避免极端低速时过严苛
- 高速（$v \geq 11$）时不再扩大：避免高速时容忍度过大
- 中间区域线性过渡：smoothness + interpretability

### 8.3 NaiveEMMA 为何不用 past camera frames？

Paper 强调 "Crucially, it does not use past camera frames." 这是 deliberate simplification：
- 隔离 single-frame reasoning 能力的评估
- 减少 temporal modeling complexity
- 测试 model 在 minimal input 下的 long-tail robustness

### 8.4 RFS Floor = 4 的设计

为什么 floor 是 4 而非 0？可能原因：
- 4 是 rank-3 trajectory 的合理下限（从 Figure 6 看，rank-3 分布主要在 0-6）
- 避免 single outlier example 完全 dominate average
- 为完全 off-distribution prediction 提供 non-zero baseline
- 与 0-10 scale 保持有意义的相对差距

### 8.5 MLLM + CoT + RL with RFS 是最优组合？

从 Poutine 的成功推断 optimal recipe：
1. **Backbone**: Qwen2.5 3B（中等规模 MLLM）
2. **Training data**: WOD-E2E + CoVLA（long-tail + diversity）
3. **SFT**: 2 轮（先 base task，再 task specialization）
4. **CoT**: 生成 reasoning caption 再 predict trajectory
5. **RL**: GRPO with **RFS as reward**

关键点是 **reward 必须与 target metric 对齐**。这是 RL 在 E2E driving 中 scaling 的关键 lesson。

---

## 9. Limitations & Future Directions

Paper 明确承认：**Open-loop setup** 是 limitation，因为 realistic sensor simulation 计算成本过高。但作者强调 WOD-E2E 可用于：
- 测试 high-fidelity simulator 的 generalizability
- 作为 closed-loop evaluation 的 bridge dataset
- 推进 open-loop benchmark SOTA

未来可能的扩展方向：
1. **Closed-loop variant**: 用 neuro-simulator 或 reconstructed scenes 做 closed-loop
2. **RFS refinement**: 引入更多 rater 或 active learning
3. **Cross-dataset RFS**: 验证 RFS 在其他 dataset 上的 transferability
4. **Long-horizon RFS**: 扩展到 10s+ horizon
5. **RFS with reasoning**: 评估 model 的 CoT 是否与 rater reasoning 一致

---

## 10. 关键 Reference 链接

- **WOD-E2E Dataset & Challenge**: https://waymo.com/open/challenges/2025/e2e-driving/
- **EMMA Paper (Hwang et al., TMLR 2025)**: https://arxiv.org/abs/2410.23265 (相关)
- **NAVSIM (Dauner et al., NeurIPS 2024)**: https://github.com/autonomousvision/navsim
- **DiffusionDrive**: https://github.com/hustvl/DiffusionDrive
- **nuScenes**: https://www.nuscenes.org/
- **WOMD**: https://waymo.com/open/data/motion/
- **CoVLA**: https://arxiv.org/abs/2408.15803
- **VLP (CVPR 2024)**: https://arxiv.org/abs/2401.05777
- **DriveVLM (CoRL 2024)**: https://arxiv.org/abs/2402.12289
- **S4-Driver (CVPR 2025)**: https://arxiv.org/abs/2412.03522
- **AutoVLA**: https://arxiv.org/abs/2506.13757
- **Poutine**: https://arxiv.org/abs/2506.11234
- **HMVLM**: https://arxiv.org/abs/2506.05883
- **Wayformer (ICRA 2023)**: https://arxiv.org/abs/2207.05844
- **Gemini 2.5**: https://arxiv.org/abs/2507.06261

---

## 11. 总结与 Intuition Building

WOD-E2E 是一个**面向 long-tail 的 dataset + metric bundle**，核心 insight 可以浓缩为：

1. **Dataset contribution**: 用 rule + MLLM mining pipeline 把 6.39M miles → 0.03% long-tail，证明 long-tail 的 practical scarcity
2. **Metric contribution**: RFS 用 3 条 human-rated trajectories + speed-adaptive trust region + exponential decay 解决 multi-modal evaluation 问题
3. **Empirical contribution**: 证明 MLLM + CoT + RL (with RFS reward) 是当前 SOTA 组合，且 ADE 与 RFS 弱相关

**最深刻的 takeaway**: 在 E2E driving 的 long-tail evaluation 中，**reward / metric alignment 决定 RL 的有效性**。如果用 ADE 训练 RL，model 会优化错误目标；用 RFS 训练 RL，model 直接优化 human-aligned objective。这是 reward design lesson 在 autonomous driving 领域的明确实证。

RFS 的 piece-wise linear speed scaling 是一个 elegant 工程选择：低速严格、高速宽松，符合 control theory 中"速度越高，控制不确定性越大"的常识。这暗示好的 autonomous driving metric 需要 **physically grounded adaptive thresholds**，而非 fixed L2 distance。
