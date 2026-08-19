---
source_pdf: HumanTracker Towards Comprehensive and Human-Aligned Motion Tracking Benchmark.pdf
paper_sha256: 59e2f8815f611ed66184352e7477176533732da99a30c3410801e4bb06b8060f
processed_at: '2026-08-19T11:58:49-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲

## 一句话版本

让机器人模仿人类动作这件事, 以前大家只会用"逐帧关节角度差"来打分, 但这个分数跟人眼看着像不像完全对不上 — 于是他们做了两件事: 1) 搞了 153 小时的专业演员动作库, 分成 4 大类方便诊断失败; 2) 用 RLHF 那套办法, 找 6 个博士生看视频 pairwise 对比, 训了一个叫 HumanScore 的 reward model 去预测人类偏好, 发现它跟人判断的一致率 90.8%, 远超 MPJPE 的 80.5%。

---

## 痛点是什么

想象你训了一个 humanoid policy 去跟踪人类动作。你怎么知道它好不好?

**老办法 (MPJPE)**: 逐帧, 对比机器人每个关节角度和 reference 差多少 radian, 然后所有 joint 所有 frame 求平均。数字越低越好。

**问题 1 — Averaging 把关键 artifact 平均没了**: 假设有两个 rollout:
- Rollout A: 每帧每个 joint 都差 0.05 rad, 全程稳定
- Rollout B: 大部分 joint 大部分时间几乎完美, 但中间脚滑了 0.3 秒, 期间脚速度突然爆掉, 然后猛地一下顿住恢复

这两个 MPJPE 可能差不多, 甚至 B 更低。但人看视频一眼就觉得 B 不对劲 — 脚滑、顿挫、节奏被打断。这种 "temporal event" 类的失败, 被 per-frame averaging 完全 wash out 掉。

这就像用 BLEU score 评 LLM — 相关性有, 但当 LLM 已经"语法都对"之后, BLEU 区分不出哪个更 helpful, 更是一模一样的问题。Karpathy 你自己讲过 RLHF 为什么要取代 next-token NLL, 同一个故事在 motion tracking 上重演。

**问题 2 — Test set 太小且没分类**: 圈内还在用 AMASS 的 140-sequence test set ([Mahmood et al. ICCV 2019](https://doi.org/10.1109/ICCV.2019.00554))。140 个 motion 太少, 没法 stress test "脚踩到桌子底下多 contact 恢复"、"单脚支撑非对称平衡"、"高速跳跃落地"这种 long-tail 行为。而且最后报一个 aggregate score, 你根本不知道 tracker 是死在 Daily 上还是 Ground 上。

---

## 他们做了什么

### 1. 搞了个大数据集: HumanTracker Benchmark

- **153 小时, 25K clips**, 24 个 professional performers (dance teacher, fitness coach, tennis coach, 全职 mocap actor)
- 多相机 optical mocap system 抓取, 然后 retarget 到 29-DoF humanoid (用 [GMR](https://arxiv.org/abs/2510.02252))
- 手工 inspect 移除"看着像人但 retarget 到机器人就破"的片段 (floating, ground penetration, discontinuous contact)
- 每 clip 含: family label + NL description + SMPL + robot qpos trajectory
- 9:1 train/test split, 同源 motion 不跨 partition

跟其他数据集对比:

| Dataset | Hours | Categories | Text Label |
|---|---|---|---|
| AMASS | 40 | No | No |
| HumanML3D | 28.6 | No | Yes |
| PHUMA | 73 | No | No |
| **HumanTracker** | **153** | **4** | **Yes** |

### 2. 四大类 (按失败模式分, 不是按活动类型分)

这是 taxonomy 设计的精髓。不是分成 running / jumping / walking 这种 activity-based 分类 (那样诊断价值低), 而是按 **"哪种 failure regime 会暴露"** 分:

- **Daily (89h)**: 走路转身这种平稳动作。暴露 steady-state stability + residual drift
- **Highly Dynamic (11h)**: 跳、踢、杂技、快舞步。暴露 rapid support switch + impact timing
- **Interaction (48h)**: 涉及物体/环境的上半身协调。暴露 hand/arm/whole-body coordination (还能给 manipulation 提供 prior)
- **Ground (5h)**: 跪、坐、滚、爬起来。低 COM + 多点 contact, 对 contact geometry & friction 极敏感

这个分类的 insight 是: **不同 motion 对 controller 的考验维度不同, 用同一个 aggregate score 评会掩盖 trade-off**。结果证明这个判断是对的 (后面会讲)。

### 3. 标准化 evaluation protocol

以前你说"我的 tracker MPJPE 0.05", 他说"我 0.08", 但其实你们 reference indexing 不一样, termination rule 不一样, simulator 不一样 — apples-to-oranges。

他们做法: 每个 tracker 保留 native policy obs + action decoder (不 hack 它的 internal representation), 但 **reference representation, rollout accounting, metric 全部标准化**。统一 29-DoF humanoid + 共享 MuJoCo entry point + 50 Hz。这才能 fair 比较。

Tracking metric (沿用 [SONIC](https://arxiv.org/abs/2511.07820) 的 whole-body termination):
- 测 pelvis / 两 ankle / 两 wrist 的 vertical position error + pelvis rotation error
- 任一 vertical error > 0.25m 或 pelvis rotation > 1 rad 就判 episode fail
- 报 Succ (完成率) 和 MPJPE (29 joint mean abs error)

### 4. 训 HumanScore: 用 RLHF 思路做 preference-aligned metric

这是最核心的部分。完整 pipeline 对标 [Christiano et al. NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) + [InstructGPT](https://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html), 只是把 LLM 换成 humanoid motion tracking policy。

#### Step 1: 生成 rollout pair
- 用 HumanTracker **training split** (绝不碰 test set)
- 4 个 SOTA tracker 跑同一 reference: GMT, TWIST2, SONIC, Humanoid-GPT
- 切成 250-frame (5 秒) window
- 6 种 tracker combination 平均分配, 每个 tracker 出现频率相同, 防 display position 偏向

#### Step 2: 找人标
- **6 个 humanoid robotics 博士生** (domain expert, 不是 Mechanical Turk)
- 决策流程: 先看是否任一 rollout 失败 → 若都还活着, 顺序比较 jitter → foot sliding → locomotion consistency → whole-body naturalness
- 标签: Left better / Right better / Similar / Cannot compare
- 总共 6000 original pair, bilateral mirror 翻倍到 12000 records
- 80/20 split by motion_id 防 leakage

#### Step 3: Frame feature (539-dim per frame)
- 70 dim current reference (root pose, joint pos/vel, foot contact)
- 126 dim robot state & action (root/IMU pose, action, motor target, joint pos/vel)
- 20 dim measured contact dynamics (foot contact, force, velocity)
- 15 dim root motion (pelvis/root velocities)
- 308 dim current keypoints (14 个 4×4 pose + 6-dim spatial velocity)

**关键设计选择**: 输入是 simulator state **不是 rendered video**。人从 video 给 preference, 但 reward model 直接吃 simulator state, 避免 camera viewpoint 依赖。这跟 [RoboReward](https://arxiv.org/abs/2601.00675), [RoboMeter](https://arxiv.org/abs/2603.02115), [LIV](https://proceedings.mlr.press/v202/ma23b.html) 那条 vision-based reward model 的线不同 — HumanScore 用 privileged state。

**另一个设计**: **不用 future reference residual**。Reward model 不知道未来 reference 是什么, 只能基于当前 + 历史评估。这样它学的是"tracking 质量怎么样"而不是"未来 motion plausibility"。Ablation 证实加 future info 反而略差, 说明这个信号难 exploit。

#### Step 4: 架构
- Frame vector → linear projection → layer norm → sinusoidal pos encoding
- 4-layer bidirectional Transformer (model dim 256, 8 heads, FFN 1024, dropout 0.1)
- Masked mean pooling (尾部短 segment 用 right zero pad + validity mask)
- MLP head → scalar unbounded reward $r_\theta(\tau)$

#### Step 5: Loss

**Strict pair (chosen vs rejected 明确)**, 经典 Bradley-Terry:

$$\mathcal{L}_{\mathrm{diff}}^{(i)} = -\log \sigma(\Delta_i), \quad \Delta_i = r_{\mathrm{chosen}}^{(i)} - r_{\mathrm{rejected}}^{(i)}$$

这里 $r_{\mathrm{chosen}}^{(i)}$ 和 $r_{\mathrm{rejected}}^{(i)}$ 是 reward model 给 chosen / rejected trajectory 的标量 reward, $\sigma$ 是 sigmoid, $\Delta_i$ 是 reward gap, 想让 chosen 的 reward > rejected 的 reward, sigmoid 把 gap 转成概率, $-\log$ 是 NLL — 跟 LLM DPO / RLHF 完全一样的形式。

**Similar pair (两个等价)**, symmetric 约束:

$$\mathcal{L}_{\mathrm{similar}}^{(j)} = -\frac{1}{2} \log \sigma(\Delta_j) - \frac{1}{2} \log \sigma(-\Delta_j), \quad \Delta_j = r_a^{(j)} - r_b^{(j)}$$

$\sigma(\Delta_j)$ 和 $\sigma(-\Delta_j)$ 都大 ⟺ $|\Delta_j|$ 都小 ⟺ 两个 reward 应该接近。等价于 push $r_a \to r_b$。

**总 loss**: 两种 pair 各贡献一次, weighted by 总 pair count。

#### Step 6: Inference 计算 HumanScore

一个 rollout $\tau$ 含 $F$ 帧, 切成 $N = \lceil F/250 \rceil$ 个 window。Window $s_i$ 实际 $L_i$ 帧。短尾 window 右 zero-pad + validity mask 处理。

**Bounded reward per window**:

$$\rho_\theta(s_i) = \sigma(r_\theta(s_i)) \in (0, 1)$$

把 unbounded reward → (0, 1), 防止 outlier window 主导总分。

**Trajectory-level score**:

$$\mathrm{HumanScore}(\tau) = \frac{100}{F} \sum_{i=1}^{N} L_i \rho_\theta(s_i)$$

按实际帧数加权平均, 乘 100 normalize 到 0-100。实证 training 分布上 $\rho_\theta$ 从 $4 \times 10^{-8}$ 到 0.99, dynamic range 足够区分。

---

## 实验结果 — 最有意思的发现

### Table 3 - Zero-shot Evaluation

| Method | Daily (Succ/MPJPE/HS) | Highly Dynamic | Interaction | Ground |
|---|---|---|---|---|
| GMT | 17.0 / 0.250 / 2.4 | 36.2 / 0.196 / 7.0 | 81.4 / 0.205 / 11.7 | 0.0 / 0.456 / 4.0 |
| TWIST2 | 60.1 / 0.105 / 10.1 | 39.9 / 0.112 / 16.9 | 91.3 / 0.111 / 28.3 | 0.0 / 0.341 / 4.5 |
| SONIC | 93.8 / 0.102 / 49.5 | 82.1 / 0.118 / 41.0 | 97.6 / 0.128 / 54.6 | 20.1 / 0.231 / 26.5 |
| Humanoid-GPT | **94.4 / 0.046 / 54.7** | **86.9 / 0.047 / 49.2** | 97.2 / 0.070 / **56.8** | 32.9 / 0.216 / 24.9 |

**关键观察**: Humanoid-GPT 整体最强, 但在 **Ground family 上, SONIC 的 HumanScore (26.5) 反而高于 Humanoid-GPT (24.9)**, 尽管 Humanoid-GPT 在 Succ 和 MPJPE 上都更好。

这就是 paper thesis 的最强 evidence: **MPJPE 第一的方法 ≠ Preference 第一的方法**。Humanoid-GPT 跟得更准, 但 SONIC 在 Ground (低姿态多 contact) 上 perceived naturalness 更高。一个 tracker 在不同 family 上的强弱 trade-off, 被 single aggregate score 完全掩盖。

### Table 4 - Preference Alignment Rate

| Metric | Align Rate |
|---|---|
| **HumanScore** | **0.9083** |
| MPJPE | 0.8049 |
| MPJVE | 0.8404 |
| KPT Position MAE | 0.8405 |
| Foot Contact Accuracy | 0.7882 |
| Avg Joint Accel | 0.6933 |
| Avg Joint Jerk | 0.7232 |

HumanScore 超过任何单一 analytic diagnostic 10+ 个百分点。Foot Contact Accuracy 居然最低, 说明光测"foot contact agreement"还不够 — timing 和 slide magnitude 都重要, contact 准 ≠ 视觉上自然。

### Table 9 - Bootstrap CI

20000 次 bootstrap, stratified by family & clustered by source motion:

| Metric | 95% CI |
|---|---|
| HumanScore | [87.36, 93.83] |
| MPJPE | [75.95, 84.76] |

HumanScore CI 下界 (87.36) 还高于 MPJPE 上界 (84.76) — 优势 statistically meaningful。

### Figure 5 - Sensitivity

- **5a**: 移除 measured contact features → 主要在 Ground family degrade (因为 Ground 多 contact transition); 加 future reference info → slightly worse (说明信号难 exploit)
- **5b**: Context 从 1s 增到 5s, Align Rate 单调上升 — 短片段只 capture instantaneous pose error, 长 context 才能揭示 evolving artifact (foot sliding, repeated jitter, progressive drift, recovery)

---

## 为什么 HumanScore 不能被 simple rule 还原

一个自然问题: 既然 foot skating 是关键 artifact, 为什么不直接测"contact 时 foot velocity"做个 rule-based metric?

**原因 1 — 人类判断是多维度同时考量**: Foot Contact Accuracy 单独只有 0.7882, 比 MPJPE 还低。人类同时考虑 pose + velocity + contact + smoothness + stability + error accumulation, 任何单一 diagnostic 都漏掉几维。

**原因 2 — 失败是 temporal event, 不是 per-frame quantity**: Foot sliding 不是某一瞬间 velocity > threshold, 而是"持续 N 帧的 sliding pattern + 后续 recovery 是否 smooth + 整体 rhythm 是否被打破"。这种 temporal structure 没法 single-frame diagnostic 公式化, 但 Transformer 学得到。

**原因 3 — 隐含 quality 难 formalize**: Annotator 会考虑 human-likeness, naturalness, jitter rhythm 这些难 explicit formalize 的 perceptual quality。Reward model 从 raw state + contact + velocity 能 implicit 学到, 因为这些 quality 在训练分布里和 preference 是 correlated 的。

这跟 [MotionCritic, Wang et al. ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c129741a2451e5fefe447591e39de30e-Abstract-Conference.html) 学 motion generation perceptual metric 思路一致, 只是 HumanScore 针对 robot tracking rollout。

---

## Intuition 总结

**Paradigm shift 1 — 从 per-frame averaging 走向 trajectory-level preference**: 失败的 perceptual unit 是 temporal event (slide, impact, support switch, recovery), 不是 isolated pose。Avg over time 会 wash out 关键 artifact。这跟 LLM 里从 token-level NLL 走向 trajectory-level RLHF reward 是同一个故事, 只不过在 continuous control / motion tracking 上重演。

**Paradigm shift 2 — 从 single aggregate 走向 family-level diagnosis**: Single number 排行榜掩盖 "tracker A 在 Ground 上完全崩, tracker B 在 Highly Dynamic 上更稳"这种 trade-off。Family-level 报告才能 pinpoint failure mode, 才能驱动下一轮 research。这也是为什么 SONIC 在 Ground 上 HumanScore 反超 Humanoid-GPT 这种细节才有意义 — 否则全被 aggregate score 抹平。

**Paradigm shift 3 — 从 rule-based diagnostic 走向 learned preference model**: 人类感知 quality 的复杂度超过任何 single rule 能 capture。Learned reward model 能 implicitly 学到 multi-dimensional perceptual quality, 90.83% align 远超任何 single diagnostic 的 70-84%。

**Paradigm shift 4 — Metric 是 proxy, 不是 ground truth**: 传统 kinematic metric 是 quality 的 proxy, HumanScore 是更 aligned 的 proxy, 但仍是 proxy。最终判断还是 human judgment。Paper 明确说 HumanScore 只用作 evaluation, 不建议直接当 RL reward — 直接优化一个 learned score 会 trigger reward hacking, 跟 [Christiano et al. 2017](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) 早就指出的 "deep RL reward model 会被 exploit" 是一回事。要用作 RL reward 必须加 regularization + independent human eval。

放到更大图景里, 这篇 paper 是 RLHF 思想在 robotics 的一次干净落地: domain expert annotator + pairwise comparison + Bradley-Terry reward model + temporal Transformer。它没发明新 RLHF 算法, 但 demonstrate 了 RLHF-aligned reward 在 humanoid tracking 上的 strong utility, 而且 153h 数据 + 开源 [code](https://github.com/GalaxyGeneralRobotics/HumanTracker) + [project page](https://dairuliu.github.io/humantracker) 让 reproducibility 变得可行。

相邻方向联想:
- [BeyondMimic, Liao et al. 2025](https://arxiv.org/abs/2508.08241) 把 motion tracking 推向 versatile control via guided diffusion — 未来 HumanScore 还要测 "能不能 extrapolate 到 reference 之外的 versatile behavior"
- [Switch-JustDance, Kim et al. 2025](https://arxiv.org/abs/2511.17925) 用 commercial game 做 whole-body benchmark, 是 complementary 思路 — 用人类设计好的 game level 做 stress test
- [Motion Turing Test, Li et al. 2026](https://arxiv.org/abs/2603.06181) 测 motion 是否 human-like, 是更 broad 的 metric, 跟 HumanScore 互补
- [KungFuBot, Xie et al. 2025](https://arxiv.org/abs/2506.12851) 专做 highly dynamic skill, 在 Highly Dynamic family 上同维度
- RLHF 在 LLM 有 DPO [Rafailov et al.](https://arxiv.org/abs/2305.18290) 直接优化 policy 的变体, Robotics 等价物还在探索 — HumanScore 当前明确限制 "不用作 RL reward", 但这是 next step, 跟 [Robo-Dopamine, Tan et al. 2025](https://arxiv.org/abs/2512.23703) 这条 process reward model for manipulation 方向平行

---

# HumanTracker: 深度技术解析

## Core Motivation

这篇 paper 直击 humanoid motion tracking evaluation 的两大痛点。让我先把 intuition 铺清楚:

**痛点 1 - Metric Misalignment**: 传统的 kinematic metrics 比如 MPJPE 把 tracking 退化成 per-frame pose matching 问题, 然后对 joints 和 time 做 averaging。这种 averaging 会 wash out 掉人类最敏感的 physical artifacts — foot skating, mistimed touch-downs, contact break-and-reattach, drift accumulation under closed-loop control。两个 rollout MPJPE 几乎相同, 但一个稳定自然, 一个抖得像帕金森。这就像用 BLEU score 评价 LLM 的 helpfulness 一样, correlation 是有的, 但 alignment gap 很大。

**痛点 2 - Test Scope 太窄**: 圈内还在用 AMASS 的 140-sequence test set [Mahmood et al., ICCV 2019](https://doi.org/10.1109/ICCV.2019.00554)。这个 set 既小又没分类, 没法 stress test long-tail behaviors (contact transitions, asymmetric balancing, complex recoveries), 而且所有 motion 被 collapse 成 single aggregate score, 没法 diagnose failure mode 在哪。

---

## HumanTracker Benchmark

### Data Scale & Collection

| Dataset | Clips | Hours | Categories | Text Label |
|---|---|---|---|---|
| AMASS [26] | >11K | >40 | No | No |
| HumanML3D [2](https://arxiv.org/abs/2405.16909) | 14.6K | 28.6 | No | Yes |
| PHUMA [12](https://arxiv.org/abs/2510.26236) | 76K | 73 | No | No |
| **HumanTracker** | **25K** | **153** | **4** | **Yes** |

- 24 professional performers: dance teachers, fitness coaches, tennis coaches, full-time mocap actors
- 多摄像头 optical mocap system, 然后用 GMR (General Motion Retargeting) [Araujo et al. 2025](https://arxiv.org/abs/2510.02252) retarget 到 29-DoF humanoid
- 因为 visually valid human recording ≠ valid robot reference, 手工 inspect 移除 floating / ground penetration / discontinuous contact 的 segments
- 每个 clip 含: top-level family label + NL description + fitted SMPL [Loper et al.](https://dl.acm.org/doi/10.1145/3115408.3115415) + robot-space qpos trajectory
- 9:1 train/test split, 同一 source motion 不跨 partition (防 leakage)

### Four Motion Families (Diagnostic Taxonomy)

| Family | Hours | #Clips | Challenges |
|---|---|---|---|
| Daily | 89 | 9.7K | steady locomotion, mild contacts |
| Highly Dynamic | 11 | 2.7K | impacts, aerial phases, fast footwork |
| Interaction | 48 | 10.9K | hands-body coordination |
| Ground | 5 | 1.6K | low posture, multi-contact transitions |
| **Total** | **153** | **25K** | diverse |

这个 taxonomy 不是按 activity 类型 (running/jumping) 分的, 是按 **failure regime** 分的:
- **Daily**: 暴露 steady-state stability + residual drift
- **Highly Dynamic**: 暴露 phase/timing errors under rapid support switch
- **Interaction**: 暴露 hand/arm/whole-body coordination (也能给 manipulation 提供 kinematic prior)
- **Ground**: 低 COM + 多点 contact, 对 contact geometry & friction 极敏感

---

## Standardized Tracker Evaluation Protocol

核心思路: 把 evaluation 当作 controlled experiment。每个 tracker 保留 native policy observation + action decoder (即不 hack 它们的 internal representation), 但 **reference representation, rollout accounting, reported metrics 都标准化**。这样避免 "我比你 MPJPE 低" 其实是因为你用了不同的 reference indexing 或不同的 termination rule 这种 apples-to-oranges 比较。

**统一接口**:
- 29-DoF humanoid qpos representation
- 共享 MuJoCo evaluation entry point
- 50 Hz 控制频率
- 每步存储: simulated generalized position/velocity, policy action, motor target, foot contacts & forces, foot/pelvis velocities, 14 keypoints pose & spatial velocity

**Tracking metric** (沿用 SONIC [24](https://arxiv.org/abs/2511.07820) 的 whole-body termination criterion):
- 测量: pelvis vertical position error, 两 ankle vertical position error, 两 wrist vertical position error, pelvis rotation error
- Termination 触发: 任一 vertical error > 0.25 m, 或 pelvis rotation error > 1 rad, 或 qpos/qvel 出现 nonfinite value
- 报告: **Succ** = 完成的 episode 比例; **MPJPE** = 29 actuated joint 的 mean absolute error (radians) over executed portion

---

## HumanScore: Preference-Aligned Reward Model

这是 paper 的核心贡献, 思路直接对标 RLHF [Christiano et al., NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) 和 InstructGPT [Ouyang et al., NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html), 只不过用在 humanoid motion tracking 而不是 LLM。

### Step 1: Preference Data Construction

**Rollout generation**: 用 HumanTracker training split (绝不碰 test set), 跑 GMT [4](https://arxiv.org/abs/2506.14770), Humanoid-GPT [29](https://arxiv.org/abs/2606.03985), SONIC [24](https://arxiv.org/abs/2511.07820), TWIST2 [44](https://arxiv.org/abs/2511.02832) 四个 tracker, 对同一 robot-space reference 产生 aligned rollout。

**Segmentation**: 50 Hz 下, 每个 rollout 切成 consecutive 250-frame window (即 5 秒)。尾部不足 250 的短 window 保留。

**Uniform pair sampling**: 把所有 window 按 (family, source motion, temporal position) 排序得到 global index, 然后在这个 list 上均匀采样 unique indices。这一步刻意避免 manual 偏向 "看起来难" 的样本 — 让采样覆盖整个 training distribution 而不被 particular family 主导。

**Balanced tracker pairing**: 6 种 unordered tracker combination (C(4,2)) 平均分配, 所以每个 pairing 贡献相同数量的 comparison, 每个 tracker 出现频率相同。candidate order 在每个 pairing 内 alternate, 最终 task order 确定性 shuffle — 防 display position 偏向某个 tracker。

**Human annotation**:
- 6 个 humanoid robotics 方向的博士研究者 (domain expert, 不是 Mechanical Turk random annotator)
- 决策流程 (顺序很关键):
  1. 先判断是否任一 rollout 失败完成或失平衡
  2. 若都还存活: 比较 jitter → foot sliding → locomotion consistency → whole-body naturalness (这个顺序)
  3. "Similar": 没有一个明显更好
  4. "Cannot compare": 都不可用或证据不足
- Display order randomized, 记录 underlying candidate indices (切断 display side ↔ tracker identity 的固定关联)
- 6000 original pairs, bilateral mirror 翻倍到 12000 records, 即 24000 motions
- 80/20 split by motion_id (seed 42) — 同源 motion 整体在一个 partition, 防 leakage
- Cannot compare pairs 排除出 reward model optimization

### Step 2: Frame Feature Representation (539-dim per frame)

| Group | Contents | Dim |
|---|---|---|
| Current reference | root pose, navigation velocity, joint position/velocity, foot contact | 70 |
| Robot state & action | root/IMU pose, action, motor target, joint position/velocity | 126 |
| Measured contact dynamics | foot contact, force, velocity | 20 |
| Root motion | pelvis/root velocities in local & navigation frames | 15 |
| Current keypoints | 14 个 4×4 pose + 6-dim spatial velocity | 308 |
| **Total** | | **539** |

注意一个设计选择: **不用 future reference residual**。意思是 reward model 看不到 "未来 reference 是什么", 只能基于当前 reference state + rollout 的历史 + 当前 → 评估 quality。这让 reward 学的是 "tracking 做得好不好", 不是 "未来 motion plausibility 怎么样"。Figure 5a 的 ablation 也证实加 future info 反而略差, 说明这个信号难 exploit。

另一个设计选择: 输入是 **simulator trajectories 不是 rendered videos**。人类从 rendered video 给 preference, 但 reward model 直接吃 simulator state, 避免 camera viewpoint / rendering choice 的依赖。这跟 RoboReward [13](https://arxiv.org/abs/2601.00675), RoboMeter [18](https://arxiv.org/abs/2603.02115), LIV [25](https://proceedings.mlr.press/v202/ma23b.html) 这条 vision-based reward model 的线不同 — HumanScore 用 privileged state。

### Step 3: Architecture (Temporal Transformer)

1. Frame vector (539-dim) → linear projection → layer normalization → sinusoidal positional encoding
2. 4-layer bidirectional Transformer encoder (model dim 256, 8 heads, FFN 1024, dropout 0.1)
3. 每个 attention layer 都用 padding mask
4. **Masked mean pooling**: 对 valid output token 做 elementwise mean → trajectory representation

公式 (Appendix A.1):

$$\bar{h}_j = \frac{\sum_{t=1}^{250} m_t h_{t,j}}{\sum_{t=1}^{250} m_t}$$

其中 $m_t \in \{0, 1\}$ 是 validity mask (1 表示真实帧, 0 表示 padding), $h_{t,j}$ 是第 $t$ 帧 token 在 feature index $j$ 上的 encoder 输出, $\bar{h}_j$ 是 pooled trajectory representation 的第 $j$ 维。这样尾部短 segment 也能处理而不让 padding 污染 attention 或 pooling。

5. MLP head (3 个 linear layer) → scalar unbounded reward $r_\theta(\tau) \in \mathbb{R}$

### Step 4: Preference Objective (Bradley-Terry + Similar Constraint)

**Strict pair** $i \in \mathcal{D}$ (chosen vs rejected 明确):

$$\mathcal{L}_{\mathrm{diff}}^{(i)} = -\log \sigma(\Delta_i), \quad \Delta_i = r_{\mathrm{chosen}}^{(i)} - r_{\mathrm{rejected}}^{(i)}$$

其中 $r_{\mathrm{chosen}}^{(i)}$ 和 $r_{\mathrm{rejected}}^{(i)}$ 分别是 reward model 给 chosen trajectory 和 rejected trajectory 的标量 reward, $\sigma$ 是 sigmoid, $\Delta_i$ 是 reward gap。这就是经典 Bradley-Terry [3](https://doi.org/10.1093/biomet/39.3-4.324) preference loss, 在 LLM 的 DPO / RLHF 里到处都是。

**Similar pair** $j \in \mathcal{S}$ (两个 trajectory 等价):

$$\mathcal{L}_{\mathrm{similar}}^{(j)} = -\frac{1}{2} \log \sigma(\Delta_j) - \frac{1}{2} \log \sigma(-\Delta_j), \quad \Delta_j = r_a^{(j)} - r_b^{(j)}$$

这是 symmetric 约束: $\sigma(\Delta_j)$ 和 $\sigma(-\Delta_j)$ 都大 ⟺ $|\Delta_j|$ 都小 ⟺ 两个 reward 应该接近。等价于 push $r_a \to r_b$, 比简单用 MSE 更 well-behaved 因为它通过 sigmoid 控制了梯度 scale。

**总 loss**:

$$\mathcal{L} = \frac{1}{|\mathcal{D}| + |\mathcal{S}|} \left( \sum_{i \in \mathcal{D}} \mathcal{L}_{\mathrm{diff}}^{(i)} + \sum_{j \in \mathcal{S}} \mathcal{L}_{\mathrm{similar}}^{(j)} \right)$$

每 pair 只贡献一次, weighted by total pair count。

### Step 5: Inference (HumanScore Computation)

给定一个 rollout $\tau$ 含 $F$ 帧, 切成 $N = \lceil F/250 \rceil$ 个 consecutive window。window $s_i$ 实际帧数 $L_i \leq 250$, $\sum_i L_i = F$。短尾 window 右侧 zero-pad + validity mask。

**Bounded reward per window**:

$$\rho_\theta(s_i) = \sigma(r_\theta(s_i)) \in (0, 1)$$

把 unbounded reward $\to (0,1)$, 防止某个 outlier window 主导 trajectory 总分。

**Trajectory-level score**:

$$\mathrm{HumanScore}(\tau) = \frac{100}{F} \sum_{i=1}^{N} L_i \rho_\theta(s_i)$$

按实际帧数加权平均所有 window 的 bounded reward, 然后乘 100 把范围 normalize 到 0-100。Padding 帧既不进入 window 内部 pooling, 也不进入 trajectory average。

实证: 在 training data distribution 上, $\rho_\theta(s_i)$ 从 $4 \times 10^{-8}$ 到 0.99 都有, dynamic range 足够区分 trajectory 质量。

---

## Experiments & Results

### Table 3 - Zero-shot Evaluation (HumanTracker test split)

 trackers 都没在 test set 上训练/finetune。

| Method | Daily (Succ/MPJPE/HumanScore) | Highly Dynamic | Interaction | Ground |
|---|---|---|---|---|
| GMT [4](https://arxiv.org/abs/2506.14770) | 17.0 / 0.250 / 2.4 | 36.2 / 0.196 / 7.0 | 81.4 / 0.205 / 11.7 | 0.0 / 0.456 / 4.0 |
| TWIST2 [44](https://arxiv.org/abs/2511.02832) | 60.1 / 0.105 / 10.1 | 39.9 / 0.112 / 16.9 | 91.3 / 0.111 / 28.3 | 0.0 / 0.341 / 4.5 |
| SONIC [24](https://arxiv.org/abs/2511.07820) | 93.8 / 0.102 / 49.5 | 82.1 / 0.118 / 41.0 | 97.6 / 0.128 / 54.6 | 20.1 / 0.231 / 26.5 |
| Humanoid-GPT [29](https://arxiv.org/abs/2606.03985) | **94.4 / 0.046 / 54.7** | **86.9 / 0.047 / 49.2** | 97.2 / 0.070 / **56.8** | 32.9 / 0.216 / 24.9 |

关键发现:
- Humanoid-GPT 在 Daily / Highly Dynamic 上领先 (大多数 metric 最佳, MPJPE 显著低于 SONIC, 几乎一半), Interaction 也领先 HumanScore
- 但在 **Ground** family, **SONIC 的 HumanScore (26.5) 高于 Humanoid-GPT (24.9)**, 尽管 Humanoid-GPT 在 Succ 和 MPJPE 上更好
- 这揭示了一个重要事实: **Humanoid-GPT 整体跟踪更准, 但 SONIC 在 Ground 上 perceived naturalness 更高** — 即 Succ 和 MPJPE 不能完全 capture human preference, 尤其在 multi-contact regime 下

这个 case 本身就是论文 thesis 的最强 evidence: kinematic metric 排第一的方法 ≠ preference 排第一的方法。

### Table 4 - Preference Alignment Rate

| Metric | Align Rate with Human |
|---|---|
| **HumanScore** | **0.9083** |
| MPJPE (rad) | 0.8049 |
| MPJVE (rad/s) | 0.8404 |
| KPT Position MAE (m) | 0.8405 |
| Foot Contact Accuracy | 0.7882 |
| Avg Joint Accel (rad/s²) | 0.6933 |
| Avg Joint Jerk (rad/s³) | 0.7232 |

HumanScore 超过任何单一 analytic diagnostic 10 个百分点以上。Foot Contact Accuracy 居然最低 (0.7882), 说明光测 foot contact agreement 还不够 — timing 和 slide magnitude都重要, contact 准 ≠ 视觉上自然。

### Table 9 - Bootstrap Uncertainty

20000 次 bootstrap, stratified by family & clustered by source motion:

| Metric | 95% CI |
|---|---|
| HumanScore | [87.36, 93.83] |
| MPJPE | [75.95, 84.76] |

HumanScore CI 下界 (87.36) 仍高于 MPJPE 上界 (84.76) — 即使有不确定性, 优势是 statistically meaningful 的。

### Figure 5a - Input Feature Ablation

- 移除 measured contact features (20-dim block): 主要在 **Ground** family 上 degrade, 因为 Ground 有复杂 multi-contact transition
- 加 future reference info: slightly worse, 说明这个信号难 exploit, 当前 + 历史 state 已足够

### Figure 5b - Temporal Context Ablation

- Context 从 1s 增到 5s, Align Rate 单调上升
- 解释: 短片段只能 capture instantaneous pose error, 长 context 才能揭示 evolving artifacts (foot sliding, repeated jitter, progressive drift, recovery from instability)
- 这直接验证了 paper 的核心 claim: **perceptual unit of failure 是 event (slide / impact / support switch / recovery), 不是 isolated pose**

---

## Why HumanScore Beats Rule-Based Diagnostics

一个自然问题: 既然 foot skating 是关键 artifact, 为什么不直接测 foot velocity when in contact 然后做一个 rule-based metric? Paper 论证了 HumanScore 不能被这种 simple rule 还原。

**原因 1**: 单一 diagnostic 都是局部信号。Foot Contact Accuracy (0.7882) 单独看比 MPJPE (0.8049) 还低。人类判断时同时考虑 pose + velocity + contact + smoothness + stability + error accumulation over time, 任何单一 metric 都漏掉其中几维。

**原因 2**: 失败是 temporal event, 不是 per-frame quantity。Foot sliding 不是某一瞬间 velocity > threshold, 而是 "持续 N 帧的 sliding pattern + 后续 recovery 是否 smooth + 整体 rhythm 是否被打破"。这种 temporal structure 没法用 single-frame diagnostic 公式化, 但 Transformer 学得到。

**原因 3**: Annotator 会考虑 human-likeness, naturalness, jitter rhythm 这些难以 formalize 的 perceptual quality。Reward model 从 raw state + contact + velocity 能 implicitly 学到这些, 因为这些 quality 在训练分布里和 preference 是 correlated 的。

这跟 [Wang et al., MotionCritic, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c129741a2451e5fefe447591e39de30e-Abstract-Conference.html) 学 motion generation 的 perceptual metric 思路一致, 只是 HumanScore 针对 robot tracking rollout 不是 generated motion。

---

## Limitations (Appendix G 总结 + 我加一些思考)

1. **Distribution scope**: HumanScore 在 HumanTracker training motions + 4 个 tracker 的 rollout 上训练。它的 test split 是 motion-disjoint, 测的是 unseen motion 的 generalization, 不是 unseen robot / simulator / controller family 的 generalization。换 embodiment 就得 retrain。

2. **Privileged state dependency**: 539-dim input 包含 simulator privileged state (contact forces, IMU 等), 这些在 hardware 上不一定 observable。要用在 real-robot trajectory 上得换 observable feature set 或加 state estimator。

3. **Single primary judgment per pair**: 每个 pair 只有一个 primary label, 没有 ICC (inter-annotator agreement) 的多 rater 重复打分。无法量化 label noise 上的 uncertainty。

4. **Diagnostic composite 比较有限**: 只对比了 single diagnostic, 没对比 fitted linear / nonlinear diagnostic composite。未来要 delimit "到底什么必须从 trajectory-level preference 学, 什么能从 diagnostic 推断"。

5. **Family imbalance**: Ground 只有 1.6K clips, 而 Daily / Interaction 各近 10K。这种 distribution 反映 natural activity frequency, 但对 statistical inference 不到全人群行为。

6. **No RL use case**: Paper 明确说 HumanScore 只用于 evaluation。直接用作 RL reward 可能 trigger reward hacking — score 高 ≠ 真的好, 因为 reward model 有自己的 blind spot。这跟 [Christiano et al.](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) 早就指出的 "deep RL reward model 会被 exploit" 是一回事。要用作 RL reward 必须加 regularization 和 independent human eval。

---

## Intuition 总结

这篇 paper 给 humanoid motion tracking evaluation 带来的 paradigm shift:

1. **从 per-frame averaging 走向 trajectory-level preference**: 失败的 perceptual unit 是 temporal event, 不是 isolated pose。Avg over time 会 wash out 关键 artifact。这跟 LLM 里从 token-level NLL 走向 trajectory-level RLHF reward 是同一个故事, 只不过在 continuous control / motion tracking 上重演。

2. **从单一 aggregate score 走向 family-level diagnosis**: Single number 排行榜掩盖了 "tracker A 在 Ground 上完全崩, tracker B 在 Highly Dynamic 上更稳" 这种关键 trade-off。Family-level 报告才能 pinpoint failure mode, 才能驱动下一轮 research。

3. **从 rule-based diagnostic 走向 learned preference model**: 人类感知 quality 的复杂度超过任何 single rule (foot slip, joint jerk, contact agreement...) 能 capture。Learned reward model 能 implicitly 学到 multi-dimensional perceptual quality, 而且 90.83% align rate 远超任何 single diagnostic 的 70-84%。

4. **从 "metric 等于 success" 走向 "metric 是 proxy"**: 传统 kinematic metric 是 proxy for quality, HumanScore 是更 aligned 的 proxy, 但仍然是 proxy。最终判断还是 human judgment。这是 RLHF 在 LLM 上学到的一课 — reward model 不是 ground truth, 它只是一个更 aligned 的 proxy, 用作 RL reward 还需谨慎。

放到更大的图景里看, 这篇 paper 是 RLHF 思想在 robotics 领域的一次干净落地: domain expert annotator + pairwise comparison + Bradley-Terry reward model + temporal transformer。它没发明新 RLHF 算法, 但它 demonstrate 了 RLHF-aligned reward 在 humanoid tracking 上的 strong utility, 而且提供了 reproducible benchmark + 153h 数据 + 开源 [code](https://github.com/GalaxyGeneralRobotics/HumanTracker) + [project page](https://dairuliu.github.io/humantracker)。

联想到几个相邻方向:
- [BeyondMimic, Liao et al. 2025](https://arxiv.org/abs/2508.08241) 把 motion tracking 推向 versatile humanoid control via guided diffusion — 这意味着未来 tracker 的 evaluation 还要测 "能不能 extrapolate 到 reference 之外的 versatile behavior", 当前 HumanScore 还没 cover
- [Switch-JustDance, Kim et al. 2025](https://arxiv.org/abs/2511.17925) 用 commercial motion game 做 whole-body skill benchmark, 是另一种 complementary benchmark 思路 — 用人类设计好的 game level 而不是 mocap 数据做 stress test
- [Motion Turing Test, Li et al. 2026](https://arxiv.org/abs/2603.06181) 测 humanoid motion 是否看起来 human, 是更 broad 的 human-likeness metric, 不限定 tracking — 跟 HumanScore 互补, 可以测 "tracker 跟得很准但 motion 本身不像人" 这种 case
- [KungFuBot, Xie et al. 2025](https://arxiv.org/abs/2506.12851) 专门做 highly dynamic skill, 在 Highly Dynamic family 上和 HumanTracker 是同一挑战维度
- RLHF 在 LLM 上有 DPO [Rafailov et al.](https://arxiv.org/abs/2305.18290) 这种直接优化 policy 的变体, Robotics 等价物还在探索 — HumanScore 当前明确限制 "不用作 RL reward", 但这是 next step, 跟 Robo-Dopamine [Tan et al. 2025](https://arxiv.org/abs/2512.23703) 这条 process reward model for manipulation 的方向是平行的
