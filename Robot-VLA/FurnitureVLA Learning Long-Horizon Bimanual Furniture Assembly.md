---
source_pdf: FurnitureVLA Learning Long-Horizon Bimanual Furniture Assembly.pdf
paper_sha256: 14c3e1955e7731a19cd6f63f1d901c9631e1427ed197f0f070ceda9ead48397a
processed_at: '2026-08-04T11:18:35-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲FurnitureVLA

## 一句话总结

让机器人组装IKEA家具这件事，以前都是toy-scale单臂玩玩，这篇paper第一次做real-scale双臂，核心trick是让VLA自己学会"我这个subtask做到百分之几了"，到了95%就自动切下一个subtask。

## 为什么这事难

想象你组装一把IVAR椅子，要7个步骤，1550个control steps。每一步都要精确对齐，tolerance才1-2cm + 4度。这中间任何一个tiny error，后面就雪崩了。

这本质是 **DAgger problem** 的经典困境：policy在training时只见过expert trajectory附近的状态，一旦execution时有个小抖动偏出去，policy就进了没见过的state，output更烂，下一步偏得更远，compound下去就废了。

VLA model比如 $\pi_{0.5}$ 在short-horizon task上很猛，但放到这种1550步的长任务上，直接finetune一把梭，只能做到48% success rate。KALLAX更是惨到11%，因为它parts最重最d大，drift一旦起来直接撞singularity。

## 核心idea: 分段 + progress signal

### 第一步: 把长任务切成subtask

不要一个monolithic policy搞定全把椅子，而是切成 "grasp left frame" → "attach rails" → "attach right frame" → "lift and rotate" 这种semantic subtask。每个subtask内部horizon短，distribution narrow，policy容易学。

### 第二步: 关键insight — boundary选在哪

这是个非常clever的design choice。直觉上你会想，subtask边界应该是part装完的那个瞬间。但paper反其道而行，boundary选在 **retreat之后** (手臂撤回去、脱离接触之后)。

为什么? Part刚装完那个state是contact-rich的，稍有误差就incomplete insertion，state variance巨大。下一个subtask拿到这种乱七八糟的initial state，更难学。

Retreat之后手臂撤开了，contact-free，小误差不会amplify，state分布narrow，下一个subtask的起点consistent。本质是在 **decouple subtask之间的误差传播**。

### 第三步: progress信号怎么设计

最naive的想法: 给每个subtask assign一个discrete label (subtask 1 = 0.1, subtask 2 = 0.3, ...)。Paper试了，**完全失败，0% success**。

原因很有意思: "part快装好" 和 "part装好了" 在视觉上几乎一样，policy根本分不出来该不该jump到下一个label，卡住了。

所以progress必须 **continuous**。但continuous怎么assign? Paper的方案是 **基于action primitive分段线性**。

每个subtask由几个primitive组成 (pickup, place, retreat等)。每个primitive分到uniform的progress range，比如3个primitive就是 [0, 0.33], [0.33, 0.66], [0.66, 1.0]。primitive内部按时间linear interpolate。

这个设计的妙处: progress和semantic stage对齐了。不是简单按时间normalize (那样短primitive会跳太快)，而是每个semantic stage贡献相等的progress range。Policy可以学到 "我还在pickup阶段，progress应该还在0-0.33之间" 这种mapping。

### 第四步: 怎么trigger transition

Policy每个timestep predict一个 $\hat{p}_t \in [0, 1]$。当 $\hat{p}_t \geq 0.95$ 持续两步，或者历史上曾经high过但现在短暂dip，都允许transition。Isolated spike直接filter掉。

这个逻辑很intuitive: progress信号要 **persistent** 才可信，偶尔一个noise spike不能触发切换。但又不能太迟钝，否则subtask早做完了还在等。

## 其他engineering细节

### Temporal ensembling

$\pi_{0.5}$ 一次predict 50步action，但不直接执行50步，而是maintain一个rolling buffer，对当前timestep的多个历史prediction做weighted average。权重是exponential decay，$\lambda = -0.1$ 时大概70%给最新prediction，30%给历史。

这相当于 **trajectory smoothing**。Real-scale bimanual操作大heavy part时，抖一下grip就slip了，smoothing对stability至关重要。KALLAX这种最重的furniture偏好更aggressive的smoothing ($\lambda = 0.25$)。

### Rear camera

双臂操作大件时，front view经常被挡。加一个rear camera，去掉的话LACK从98%暴跌到45%。这不是marginal gain，是enabler。

### Resolution 448 vs 224

从224升到448，average从0.60升到0.80。对齐magnet hole这种sub-cm precision任务，pixel-level detail决定成败。实现上就是把PaliGemma-SigLIP的positional embedding从24×24 bicubic interpolation到32×32。

## 实验结果的人话解读

Simulation: FurnitureVLA从baseline的48%拉到80%，主要靠两块 — progress-enhanced贡献+32%，design factor study再贡献+21%。两块同等重要，说明 **algorithm和engineering在precision task里都不能省**。

Real-world IVAR chair: full assembly 40%，相比simulation的56%只drop 16%。这个gap主要来自error accumulation (per-part success ~78%，但串联起来就掉到40%)，没有任何single subtask是catastrophic failure，失败是cumulative的。

最有意思的是 **emergent self-correction**: policy会regrasp (grip不稳时自动松开重抓)、会对齐magnet (插入前做small corrective motion)。这些behavior是从teleoperated demo里学到的，teleoperator自己也会这么correct，policy学到了corrective behavior的distribution。

## 我的takeaway

这篇paper的core insight其实很简单: **long-horizon VLA的failure主要是distribution drift，解法是切短 + 自动detect transition**。但execution细节非常多 — boundary选在哪、progress怎么assign、discrete vs continuous、smoothing怎么配、viewpoint怎么设计 — 每一个都踩过坑才得到最终方案。

Progress signal这个idea我觉得可以推广。本质上它是一个 **implicit stage detector**，不需要external module，policy自己学会判断进度。这个pattern在很多long-horizon task里都应该work，比如cooking (切菜→炒菜→装盘)、cleaning (扫→拖→整理)。

Post-retreat boundary的idea更generic: **在contact-free、error不会amplify的state做task boundary**。这个principle适用于任何contact-rich manipulation的long-horizon decomposition。

参考:
- Paper: https://arxiv.org/abs/2504.16054 ($\pi_{0.5}$ backbone)
- FurnitureBench: https://furniturebench.github.io/
- ACT (temporal ensembling): https://tonyzhaozh.github.io/aloha/

---

# FurnitureVLA: Real-Scale Bimanual Furniture Assembly with VLA

这篇paper来自MERL (Mitsubishi Electric Research Laboratories), 解决了一个非常实际且困难的问题: 用VLA model做real-scale双臂家具组装。让我深入剖析其核心insight。

## 1. Task的核心难度

Furniture assembly的难点是 **long-horizon + high-precision + bimanual coordination** 的复合:

| Furniture | Subtasks | Skill executions | Control steps | Duration |
|-----------|----------|------------------|---------------|----------|
| LACK side table | 4 | 12 | 650 | ~65s |
| KALLAX shelf | 4 | ~15 | 850 | ~85s |
| IVAR chair | 7 | 25 | 1550 | ~155s |

对比prior work如CALVIN (通常<200 steps) 或FurnitureBench (toy-scale, single-arm), 这个setting难度大幅提升。关键challenges:
- **Compounding error**: small deviation → out-of-distribution state → cascading failure (Ross et al. DAgger problem [13])
- **Tight geometric tolerance**: translation $\epsilon \in [1, 2]$ cm, rotation $\delta = 0.998$ (~4°/axis)
- **Bimanual singularity avoidance**: 大件需双臂同时lift+rotate+align

参考: FurnitureBench https://furniturebench.github.io/

## 2. System Architecture Overview

整个system分为两部分:

### 2.1 Simulation pipeline (Isaac Gym)
- 基于 FurnitureBench codebase扩展
- Motion planning生成expert demos: single-arm直接plan EEF pose trajectory; bimanual则disable physics, plan object trajectory, 双臂作为rigid end-effectors跟随
- **关键trick**: magnet simulation用pose-resetting (Isaac Gym不支持runtime weld constraint), 当relative pose落入tolerance内时, 每个tick kinematic teleport到rigid relative pose

### 2.2 VR teleoperation system
硬件: Meta Quest 3 + Quest2ROS (72Hz streaming), dual Kinova Gen3 (7-DoF), Robotiq Hand-E (left) + 2F-85 (right)

三大设计principle:
1. **Decoupled translation/rotation control**: index trigger = translation only, middle trigger = rotation only, both = full 6-DoF
2. **Pre-defined grasp primitives**: 90° orientation variants via button, snap-to-preset while保持position
3. **Synchronized bimanual mode**: 左臂mirror右臂的translation/rotation, 用于lift+rotate IVAR chair frame这种heavy task

参考: Quest2ROS https://github.com/rvp-group/quest2ros

## 3. Progress-Enhanced VLA (核心方法)

这是paper的核心创新。让我从intuition出发讲解。

### 3.1 为什么naive finetune不够

直接在full-length bimanual demo上finetune $\pi_{0.5}$ (monolithic finetuned) 只达到48% average success。原因: VLA在long-horizon任务上effective regime是short-horizon, long trajectories导致distribution drift, small error放大成catastrophic failure。

### 3.2 Subtask decomposition + continuous progress

**核心idea**: 把长任务分解成semantically grounded subtasks, 在每个subtask内finetune VLA, 同时joint predict一个continuous progress signal $p_t \in [0, 1]$ 来trigger subtask transition。

Action representation扩展:
$$\tilde{a}_t = [a_t^\top, p_t]^\top \in \mathbb{R}^{15}$$

其中 $a_t = [a_t^L, a_t^R] \in \mathbb{R}^{14}$, $a_t^L, a_t^R \in \mathbb{R}^7$ 分别是左右臂的 $[x, y, z, u, v, w, \gamma]$ (EEF pose + gripper state)。

### 3.3 Progress信号的数学公式

这是paper最精妙的设计。Progress不是简单的linear interpolation over time, 而是基于 **action primitives** 的分段线性映射:

$$p_t = \frac{i}{N_k} + \frac{1}{N_k} \cdot \frac{t - s_i}{s_{i+1} - s_i}, \quad s_i \leq t < s_{i+1}$$

**变量解释**:
- $i$: 当前action primitive的index, $i \in \{0, 1, ..., N_k - 1\}$
- $N_k$: 第$k$个subtask包含的action primitive总数 (例如pick up, place, retreat等)
- $s_i$: 第$i$个primitive的起始timestep
- $s_{i+1}$: 第$(i+1)$个primitive的起始timestep (对最后一个primitive, 是subtask结束)
- $t$: 当前timestep

**Intuition**: 每个primitive占用 $[i/N_k, (i+1)/N_k]$ 的uniform interval。在primitive内部, progress随时间linearly增加。这样progress从0单调递增到1, 且每个segment对应一个fixed control objective。Linear interpolation之所以natural, 是因为within-primitive的motion是smooth的, 时间参数化progress与motion evolution一致。

**为什么不直接用normalized time $t / T_{subtask}$**: 因为不同primitive持续时间差异大 (retreat通常短, insertion通常长), 简单normalize会让progress在short primitive里跳太快, 在long primitive里跳太慢, 失去semantic alignment。Primitive-based的分段保证每个semantic stage贡献相等的progress range。

### 3.4 Post-Retreat Subtask Boundaries (关键insight)

这是一个非常subtle但重要的设计。Subtask boundary定义在 **retreat之后** (contact-free state), 而是在part assembly完成时 (contact-rich state)。

**原因分析**:
- Post-assembly state是contact-rich的, small execution error会导致incomplete insertion或unstable contact, 状态variance很大
- 这会widening next subtask的initial state distribution, 加剧cross-subtask distribution shift
- Post-retreat state是contact-free的, 无force constraint, small error不会amplify, 状态分布narrower且consistent

这个设计本质上是在 **减少subtask之间的coupling**, 让每个subtask成为一个self-contained problem。

### 3.5 Discrete vs Continuous Progress (ablation的关键发现)

Paper做了一个ablation: 用discrete progress (每个subtask $k$ assigned固定值 $(2k-1)/(2K)$, 所有observations within subtask共享同一个constant progress)。

**结果**: Discrete progress **完全失败** (0% success across all furniture)!

**原因**: 在subtask completion时, model无法advance progress signal, 卡住了。"part nearly assembled" 和 "part assembled" 的visual similarity太高, discrete transition无法detect。Continuous progress提供smooth, unambiguous的supervision signal throughout整个subtask, 这点至关重要。

## 4. Inference: Subtask Transition Logic

Predicted progress $\hat{p}_t$ 触发subtask transition的逻辑:

$$h_t = (\hat{p}_t \geq \tau_p), \quad \tau_p = 0.95$$

$$\text{TRANSIT} = \begin{cases} 1, & h_t \wedge h_{t-1} \\ 1, & h_t \wedge \neg h_{t-1} \wedge \neg h_{t-2} \wedge \exists \Delta \geq 3: h_{t-\Delta} \\ 0, & \text{otherwise} \end{cases}$$

**逻辑解析**:
- Case 1: 连续两步都high → 确认transition (persistent signal)
- Case 2: 当前high, 上一步low, 上上步low, 但历史某步 $\Delta \geq 3$ 之前曾high → 允许transition (容忍短暂dip)
- Case 3: 其他情况 → 不transition (filter isolated spikes)

**Intuition**: 这是一个lightweight temporal filter, 平衡 **responsiveness** (对persistent signal快速响应) 和 **robustness** (过滤noise spike)。Progress signal总是用most recent prediction, 不受temporal ensembling buffer影响。

## 5. Design Factors Study (precision的关键)

Paper系统地研究了4个perception/control design factor, 这些在real-scale bimanual assembly中至关重要。

### 5.1 Temporal Ensembling

维持rolling buffer of $B$ overlapping action chunks, executed action是weighted average:

$$\hat{a}_t = \frac{\sum_{i=0}^{B-1} w_i a_t^{[t-i]}}{\sum_{i=0}^{B-1} w_i}, \quad w_i = e^{\lambda i}$$

**变量解释**:
- $B$: buffer大小
- $a_t^{[t-i]} \in \mathbb{R}^{14}$: $i$步之前对timestep $t$的预测 (从overlapping chunks中取)
- $w_i = e^{\lambda i}$: 权重, $\lambda$是温度参数
- $\lambda < 0$: emphasize recent predictions; $\lambda = 0$: uniform averaging

**结果**: $\lambda = -0.1$ 最优。此时 ~70%权重给most recent prediction, 其余分散到prior predictions。这balance了 **responsiveness** (recent prediction主导) 和 **stability** (prior predictions提供smoothing)。

**为什么real-scale bimanual特别需要smoothing**: 大heavy parts和coordinated bimanual motion对instability敏感, smoothing能减少jitter导致的grip slip或misalignment。KALLAX (最heavy) 偏好 $\lambda = 0.25$ (更aggressive smoothing), 印证这点。

参考: ACT paper https://tonyzhaozh.github.io/aloha/

### 5.2 Action Horizon

$\pi_{0.5}$ predicts chunk of $H = 50$ actions, 但只execute前 $k \in \{5, 10, 25\}$ 个再replan。

**Trade-off**:
- Fewer steps (e.g., 5): 更频繁replan, 更好error recovery, 但higher overhead, trajectory less smooth
- More steps (e.g., 25): less replanning, smoother, 但error积累

**结果**: 10和25各自在不同furniture上最优。KALLAX偏好25 (heavy parts需要smoothing), LACK偏好10。

### 5.3 Viewpoint

4 cameras: 1 front, 2 wrist-mounted, 1 **rear** (新增)。

**Rear camera的必要性**: Large furniture parts和frequent bimanual interactions经常occlude frontal observation。Removing rear camera: LACK从0.98降到0.45! KALLAX从0.85降到0.57! 可见rear view对bimanual occlusion至关重要。

Replacing rear with front-view depth: 效果也不好 (average 0.50)。RGB的fine-grained texture cue对precision assembly比depth更重要。

### 5.4 Image Resolution

224 → 300 → 448, 性能monotonically提升 (0.60 → 0.72 → 0.80)。

**实现细节**: PaliGemma-SigLIP vision backbone原pretrained在224×224, paper将其upscale到448×448, 得到32×32 token grid (448/14 = 32, 14是SigLIP patch size)。Pretrained positional embeddings (24×24) 通过bicubic interpolationresize到32×32。End-to-end finetune。

**Intuition**: Contact-rich bimanual assembly需要fine-grained visual cue (magnet hole alignment, edge matching), low resolution会丢失这些detail。

## 6. Backbone: $\pi_{0.5}$

- **Vision encoder**: PaliGemma-2B + SigLIP (448×448 input)
- **Action expert**: Gemma-300M
- **Action decoder**: Flow matching (10 denoising steps at inference)
- **Total params**: ~2.6B (2.3B PaliGemma + 311M action expert + <1M projection heads)
- **Training**: 40K steps, 8×L40S, batch 64, AdamW, LR 2.5e-5 cosine decay, full finetune (no LoRA)

**Flow matching**: $\pi_\theta(a_{t:t+H-1} | o_t, g)$ 在single forward pass解码 $H=50$ future actions。Flow matching相比diffusion效率更高, 10步denoising足够。

参考: $\pi_{0.5}$ paper https://arxiv.org/abs/2504.16054

## 7. 实验结果深度分析

### 7.1 Simulation结果

| Method | LACK | KALLAX | IVAR | Average |
|--------|------|--------|------|---------|
| $\pi_{0.5}$ (zero-shot) | 0.00 | 0.00 | 0.00 | 0.00 |
| $\pi_{0.5}$ (monolithic finetuned) | 0.91 | 0.11 | 0.41 | 0.48 |
| FurnitureVLA | 0.98 | 0.85 | 0.56 | 0.80 |

**Zero-shot失败**完全expected, task out-of-distribution。

**Monolithic finetune的KALLAX极低 (0.11)**: KALLAX有largest heaviest parts, 对long-horizon drift和singularity最敏感。FurnitureVLA提升到0.85, 说明progress-enhanced subtask decomposition对heavy bimanual task特别有效。

**IVAR最难 (0.56)**: 7 subtasks, 包含双臂lift+rotate整个partial assembled chair frame这种极复杂操作。Subtask 5是bottleneck (Fig. 5), 需要双臂grasp+lift+attach chair frame。

### 7.2 Design factor的累积效果

Default config (无temporal ensembling, action horizon 25, no rear cam, 224 res): average ~0.59 (从Table II推算)

Optimal config (λ=-0.1, horizon 10, full view, 448 res): 0.80

**Design factor study贡献 +21%**, 几乎和progress-enhanced VLA本身贡献 (+32%) 相当! 这说明对precision task, engineering design factor和algorithmic innovation同等重要。

### 7.3 Real-world结果 (IVAR chair)

| Metric | S1 | S2 | S3 | S4 | S5 | S6 | S7 |
|--------|----|----|----|----|----|----|-----|
| Full Assembly SR | 0.80 | 0.73 | 0.60 | 0.53 | 0.47 | 0.47 | 0.40 |
| Per-Part SR | 0.80 | 0.80 | 0.73 | 0.80 | 0.67 | 0.87 | 0.80 |

**Real-world full assembly = 0.40**, vs simulation 0.56, **drop 16%** (paper标题强调)。这个drop相对合理, 考虑到real-world noise, friction, magnet alignment error。

**Per-part vs Full assembly的gap**: Per-part average ~0.78, Full assembly 0.40。这38%的gap量化了 **error accumulation across subtasks**, 印证long-horizon的根本困难。没有任何single subtask是catastrophic failure mode, 失败是cumulative的。

**Emergent self-correction** (Fig. 6): 
- Regrasping: gripper insufficient contact时, 自动reopen+adjust pose+regrasp
- Magnet alignment: attachment前做small corrective motion对齐magnets

这些behavior是从teleoperated demo中学到的, teleoperator在demo时也有类似correction, 说明policy学到了 **corrective behavior的distribution**, 而非只是open-loop trajectory。

## 8. 关键Insight总结

1. **Continuous > Discrete progress**: Visual similarity between "nearly done" 和 "done" state让discrete transition无法detect。Continuous progress提供smooth, unambiguous supervision, 是long-horizon VLA的关键design choice。

2. **Post-retreat boundaries减少cross-subtask coupling**: Contact-free state的narrow distribution让每个subtask成为self-contained problem, 大幅减少distribution drift。

3. **Progress作为implicit subtask detector**: 不需要external stage estimator, VLA自己predict progress并trigger transition, 系统更简洁。

4. **Design factor和algorithm同等重要**: Temporal ensembling, action horizon, viewpoint, resolution这些"工程"choices累积贡献+21%, 在precision task中不可忽视。

5. **Bimanual需要dedicated viewpoint design**: Rear camera对bimanual occlusion至关重要, 移除导致~50%性能drop。

6. **Real-world sim gap主要在error accumulation**: Per-part success ~0.78 vs Full assembly 0.40, gap来自cumulative error而非single catastrophic mode。

## 9. Limitations和未来方向

Paper坦诚承认:
- **Fixed-base setup**: 限制在kinematic workspace内, mobile bimanual platform (如Mobile ALOHA) 可处理更大item
- **Magnet代替screwing**: 真正的tool use (screwdriver)需要significantly higher precision, 是orthogonal challenge

**我个人的延伸思考**:
- Progress信号能否用learned (而非primitive-based)的方式assign? 例如用contrastive learning从visual similarity推断progress
- Post-retreat boundary的idea能否推广到其他long-horizon task? 核心是在contact-free, error不会amplify的state做boundary
- Self-correction behavior的emergence暗示VLA学到了distribution over corrective policies, 能否显式encourage这种behavior (例如data augmentation with perturbation)?

参考链接:
- FurnitureBench: https://furniturebench.github.io/
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- ACT (temporal ensembling origin): https://tonyzhaozh.github.io/aloha/
- OpenVLA: https://openvla.github.io/
- Mobile ALOHA: https://mobile-aloha.github.io/
- Isaac Gym: https://developer.nvidia.com/isaac-gym
- DROID dataset: https://droid-dataset.github.io/

这篇paper在long-horizon VLA领域是一个solid的system contribution, progress-enhanced的idea简洁且effective, design factor study对precision manipulation community有实用价值。Real-world 16% drop在如此complex task上算convincing的validation。
