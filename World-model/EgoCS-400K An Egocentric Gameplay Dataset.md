---
source_pdf: EgoCS-400K An Egocentric Gameplay Dataset.pdf
paper_sha256: 0360d2559722bde4d282e9d269f6bed9e93c5cc11dfcfd5650135dd7ade12b85
processed_at: '2026-08-18T07:29:57-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 EgoCS-400K

## 一句话版本

他们用 Counter-Strike 的 replay 文件，搞出一个 10,000 小时、40 万条第一人称视频的数据集，每条视频都带完整 keyboard/mouse 输入、per-tick state、action timeline 和 VLM 生成的 caption，专门给 world model 训练用。

## 为什么这件事有价值

现在 world model 遇到一个尴尬的瓶颈：大家都在喊"视频不是被动看的，是 action 驱动的"，但真要训 action-conditioned 视频，数据从哪来？

- Web video（YouTube 那种）：只有画面 + 弱 caption，没有 action signal
- Ego4D：第一人称，但 action label 是"切洋葱"这种语义级，底层 control signal 早丢了
- Robotics（DROID、Open X-Embodiment）：action/state 全有，但机器人采一条 trajectory 要几十分钟真金白银，scene 还单调
- Minecraft（MineRL、PLAICraft）：便宜，但视觉太糙，像玩具

CS demo 文件的妙处在于：**它不是录屏，是录像带+操作日志**。HLTV 上有成千上万个 pro match 的 `.dem` 文件，每个文件记录了整场比赛每一 tick 的所有 player state、input、event。你可以把它重新 render 成任意视角的视频，同时把每个 frame 对应回权威的 tick timeline。这种"replay file 是 ground truth timeline，video 是渲染产物"的解耦，让你可以无成本地批量造出 observation-action-state 三者严格对齐的数据。

10 个 player 同一 round 的视角都有，这是 multi-agent reasoning 的天然监督——同一战术场景，10 条相互关联的 egocentric stream。

## Pipeline 的精髓

整条 pipeline 的核心思想其实就一句话：**所有 supervision 从 demo 文件硬解析出来，video 只负责提供外观信息**。

这个设计很反直觉。传统 caption dataset 是反过来——caption 是 ground truth，video 是被描述对象。EgoCS-400K 里，caption 不是 free-form 文字，是 replay-derived facts 约束下的结构化输出。

具体三步：

1. **Demo → Video**：拉公开 pro match demos，用 CS client 渲染成第一人称视频。CS2 是 64 ticks/s，视频用 32 FPS，所以每个 frame 精确对应 2 个 ticks。

2. **Parse + Segment**：解析出 per-tick 的所有 state（位置、视角、按键、武器状态），用 rule-based detector 切成 atomic action spans，然后用 dynamic programming 把长视频切成 2-6.5 秒的 training segments。

3. **Prior-guided captioning**：把 segment 内的 action/movement/camera priors 和视频一起喂给 VLM，生成结构化 caption。

## 几个我觉得很 clever 的设计

### 1. Protected chain + DP segmentation

最朴素的问题：一段 90 秒的视频怎么切？

如果固定 4 秒一刀，刀会落在 reload 中间、grenade 飞行中途、scope 开镜过渡——这些 action 被切断了，caption 就会描述"player 拔枪... 然后视频没了"。

他们把这种"不能切"的 interval 标成 protected，merge 成 protected chains，然后要求所有 segment 边界只能落在 protected interval 之外。这退化成一个 boundary selection 问题，用 DP 解。

Cost vector 是 lexicographic ordering（不是标量加权）：
1. 覆盖率（gap penalty）—— 优先级最高
2. pre-action context shortfall
3. segment 数量偏好
4. 偏离 target duration 的程度
5. boundary preference

这种优先级排序的好处是：每一项只在前一项 tie 时才起作用，不需要 tune 权重。

### 2. Pre-action context 这一项

这是整个 DP 设计最 elegant 的地方。

考虑投弹这个 action：玩家先调整视角找到投弹方向 → 切到 grenade → 拉环 → 投出。如果 segment 起点 = protected chain 起点，caption 模型只看到"player threw a grenade"，但看不到"为什么投这里"。让 segment 提前 0.5s，模型能看到 approach motion 和 aim setup，caption 就变成"player 在拐角处预瞄，拔出 HE grenade 投向 doorway"。

这一个项就把 caption 从"动作描述"提升到"动机描述"。

### 3. Prior filtering 的双 failure mode

给 VLM 喂 priors 时，他们没有简单"全喂"或"不喂"，而是做了精细过滤。两种 failure mode：

- **不喂 priors**：VLM 容易漏掉短而机械的 action（weapon switch、recoil recovery），因为视觉上不明显。
- **全喂 priors**：spurious grounding。比如 caption 前一刻有个 leftward jitter，紧接着明显 right turn 露出 doorway，dense prior 会让 VLM 把 doorway 错误关联到 leftward jitter。

所以过滤规则是：保留确实产生 visual change 的 action，移除 noEffect action（比如 weapon-switch 动画期间按 fire，state machine 阻塞，实际没射）。

Movement priors 还有个细节：玩家按住 W 但撞墙没动，被检测为 ineffective motion。这区分了 **input** 和 **outcome**——world model 要学的是"按 W → 前进"，但 demo 里大量存在"按 W 但没动"的反例，不过滤会让模型学到错误因果。

### 4. Mouse proxy

Demo 文件里没有"鼠标硬件位移"，但记录了 player 的 pitch/yaw。他们用 view-angle delta 作为 mouse motion 的连续 proxy：

$$\Delta\theta_t = \theta_t - \theta_{t-1}, \quad \Delta\psi_t = \text{wrap}_{[-180,180)}(\psi_t - \psi_{t-1})$$

对 world model 训练来说，模型要学的是"我转了多少视角 → 场景怎么变"，sensitivity 具体值不重要。这个 proxy 恰好是 invariance 的。

## 局限性

CS 是 game environment，几个 domain gap：

- Action 是 keyboard/mouse，不是 robotic 7-DoF end effector
- Physics 是 game state machine，不是 Newtonian physics
- 没有 tactile feedback、deformable object、articulated manipulation
- Caption 是 VLM-generated，不是 gold label
- Behavioral distribution 偏 combat，没有日常行为

但作为 intermediate testbed，价值在于：让 world model 在"dense supervision + 高保真视觉 + 复杂 multi-agent 场景"三者齐全的环境下先验证 scaling law，再迁移到真实 embodied 场景。这和 ImageNet 之于 visual recognition 的角色类似——不是最终目标，但提供了必要的 pretraining signal。

## 我的延伸联想

1. **Replay file 作为 audit path 的思想可以推广**。任何能 record trajectory 的环境（赛车 replay、StarCraft replay、ROS bag）都可以套用这套 pipeline。这给 embodied AI 提供了比 robotic logging 便宜得多的数据范式。

2. **Prior-guided VLM captioning 对自动驾驶也适用**。nuScenes、Waymo 都有 raw sensor + CAN-bus signal，但 caption 是 free-form。如果把 CAN signal 转成 action/movement/camera priors 约束 VLM，可以得到和 EgoCS-400K 类似的 grounded driving caption。

3. **Pre-action context 的思想可以反过来用于 RL**。在 imitation learning 里，expert demonstration 的 pre-action state（approach、aim、setup）往往比 action 本身更有信息量。

4. **Lexicographic cost vector 的思想**。在多目标 optimization 里，scalar weighting 总要 tune weight，而 lexicographic ordering 只需确定 priority。这套设计在数据切分、RL reward shaping、curriculum learning 里都适用。

## 相关参考

- Project: https://EgoCS-400K.github.io
- Genie (Bruce et al. 2024, ICML): https://arxiv.org/abs/2408.14837
- GameNGen (Valevski et al. 2024): https://arxiv.org/abs/2408.14837
- Oasis (Decart): https://oasis-model.github.io/
- PLAICraft (He et al. 2025): https://arxiv.org/abs/2505.12707
- π0 (Black et al. 2024): https://arxiv.org/abs/2410.24164
- OpenVLA (Kim et al. 2025): https://arxiv.org/abs/2406.09246
- DROID (Khazatsky et al. 2024): https://arxiv.org/abs/2403.12945
- Ego4D (Grauman et al. 2022): https://arxiv.org/abs/2110.07058
- MineDojo (Fan et al. 2022): https://arxiv.org/abs/2206.08853
- MineRL (Guss et al. 2019): https://arxiv.org/abs/1904.08208
- Hunyuan-GameCraft 2 (Tang et al. 2025): https://arxiv.org/abs/2511.23429
- HunyuanWorld (2025): https://arxiv.org/abs/2507.21809
- The Matrix (Feng et al. 2025): https://arxiv.org/abs/2412.03568
- WorldScape: https://worldscape.io/

---

# EgoCS-400K: 用 Counter-Strike Demos 构建 Egocentric World Model 的训练数据

## 一、核心 insight: 为什么是 CS demo

World model 训练真正缺的，是"observation ← action ← state"三者的因果绑定。web video（HowTo100M、WebVid、Kinetics）只有被动视觉流；Ego4D/EPIC-KITCHENS 有第一人称，但 action label 是语义级别（"切洋葱"），低层 control signal 丢失；robotics datasets（DROID、Open X-Embodiment）有 action supervision，但 scene 多样性低、collection 成本高。

CS demo 文件的特殊性在于它是一个 **replay-grounded trajectory**，而 recording。每个 `.dem` 文件保存的不是 pixel stream，而是 per-tick 的 player state（位置、view angle、velocity、weapon state、button bitmask），server-side event（fire、grenade_thrown、detonation）以及 round-level context。这意味着给定同一个 demo，可以重新 render 出完全对齐的第一人称视频，同时把每个 frame 反向映射回 authoritative tick timeline。这种 **"replay file 作为 ground truth timeline，video 作为渲染产物"** 的解耦，是整个 dataset 的设计根基。

让我把规模先量化（Table 2）：

| 项 | 数值 |
|---|---|
| Matches | >1,000 |
| Rounds | >40,000 |
| Round-player videos | >400,000 |
| Total video | >10,000 hours |
| Avg length | ≈90 s |
| Maps | 13 |

10 个 player viewpoint per round 的设计意味着同一战术场景有 10 个相互关联的 egocentric stream，这对 multi-agent reasoning 是天然的监督。

Project page: https://EgoCS-400K.github.io

---

## 二、Pipeline 总览

Figure 2 展示的 pipeline 可以分成三个 phase：

**Phase 1 — Demo → Video**：从 HLTV 拉公开 pro match demos，用 CS Demo Manager + CS client 渲染。关键设计：CS2 是 64 ticks/s，渲染用 32 FPS，所以每个 video frame 对应 2 个 ticks（deterministic mapping）。这一步建立了"frame ↔ tick"的双向 lookup，所有后续 annotation 都从这个映射出发。

**Phase 2 — Parsing + Segmentation**：用 DemoParser2 抽取 per-tick trace，rule-based detector 生成 atomic action spans，构造 protected chains，DP 切分 training segments。

**Phase 3 — Prior-guided VLM captioning**：把 segment-local 的 action/movement/camera priors 和 clipped video 一起喂给 VLM，输出结构化 JSON caption。

整条 pipeline 的精髓在于：**所有 supervision 从 demo 文件"硬解析"出来，video 是 soft evidence**。Video 提供 appearance（光照、几何、烟雾、手部），demo 提供 control/state 事实，二者在同一个 temporal frame 内 reconcile。这是和普通 caption dataset 的本质区别。

---

## 三、Parsing 细节：replay → per-tick supervision

### 3.1 Keyboard / Mouse Reconstruction

Keyboard 端是离散的：从 button bitmask 直接 decode W/A/S/D、jump、duck、walk、fire、right-click、reload、use。

Mouse 端有意思。作者**用 view-angle delta 作为 mouse motion 的连续 proxy**，而非 raw hardware mouse counts。这是合理的：在 demo 文件里不存在"鼠标硬件位移"这个量，但是 game state 中存在 player 朝向 $\theta_t$（pitch，俯仰）和 $\psi_t$（yaw，偏航）。

公式 (1)：

$$\Delta\theta_t = \theta_t - \theta_{t-1}, \quad \Delta\psi_t = \text{wrap}_{[-180,180)}(\psi_t - \psi_{t-1})$$

- $t$：tick index
- $\theta_t$：tick $t$ 时刻的 pitch（垂直方向视角）
- $\psi_t$：tick $t$ 时刻的 yaw（水平方向视角）
- $\Delta\theta_t$、$\Delta\psi_t$：连续两 tick 之间的视角位移
- $\text{wrap}_{[-180,180)}(\cdot)$：把 yaw 差值 wrap 到半开区间，避免 $\psi$ 从 179° 跳到 -179° 时被算成 358° 的伪位移

公式 (2)：

$$\text{mouse\_dy}_t = \Delta\theta_t, \quad \text{mouse\_dx}_t = \Delta\psi_t$$

所以 `mouse_dx/dy` 在这里**测的是视角位移**，单位是角度/tick。这个 proxy 假设 mouse sensitivity 设置稳定，但对 world model 来说够用——模型要学的是"我转了多少视角"，而不是"我手挪了多少距离"。

### 3.2 Atomic Action Extraction

不靠人工标注，纯 rule-based detector 把 raw signals 映射成 temporally bounded action spans。每种 action 用不同 source signal：

- Weapon switch ← active weapon 变化
- Reload ← 持续的 reload-state flag
- Inspect ← weapon animation state
- Duck ← duck-state flag + duck amount
- Fire ← fire button + weapon-fire event（联合验证）

Grenade 这种 multi-stage action 被拆成 temporal chain：preparation（按键 hold）→ release（投出）→ flight（弹道）→ effect（爆炸 / 烟雾消散）。每一段是独立 span 但保留 high-level 关联。这种结构对 caption 极重要：让 VLM 描述"投弹-飞行-爆炸"的因果序列，而不是把爆炸单独抽出来当孤立事件。

---

## 四、Protected Chains 与 DP Segmentation

### 4.1 什么是 protected interval

切 video 时，有些 action span **不能被边界切断**：

- weapon draw（拔枪动画）
- reload（换弹）
- grenade preparation / flight
- scope transition（开镜过渡）

切断这些会让 caption 出现 "weapon 出现一半"、"弹道中途消失"、"镜开一半" 这种 visual / semantic discontinuity。

但有些 span 是 state-only 的，切了无所谓：

- lingering grenade effect（爆炸后烟雾继续弥漫）
- sustained scoped（持续开镜状态）
- sustained crouched（持续蹲下）

把 overlapping/adjacent 的 protected intervals merge 成 **non-overlapping protected chains**，每个 chain 是"不可分割的最小连续单元"。

### 4.2 为什么不能用固定长度切分

固定 4 秒切分会在 reload 中间落刀，导致：
1. caption 描述一个不完整的 reload
2. world model 训练时看到 action 中途被截断的样本
3. downstream action-conditioned generation 学到错误的因果终止

所以 segmentation 退化成 **constrained boundary selection**：边界只能落在 protected interval 之外的位置。

### 4.3 DP formulation

Video timeline 被建模成一个 ordered set $V = \{t_0, t_1, \ldots, t_N\}$，其中每个 $t_i$ 是一个 valid boundary node（落在 protected interval 之外的时间点）。

每个 candidate segment 是一条 edge $i \to j$，合法的条件是 duration $t_j - t_i$ 落在 $[L_{\min}, L_{\max}]$ 内（默认 2.0s / 6.5s，target 4.0s）。

DP recurrence（公式 3）：

$$D[j] = \text{lexmin}_{i<j, (i,j) \in \mathcal{E}} \big(D[i] + C(i,j)\big)$$

- $D[j]$：从起点到 valid node $j$ 的最优累积 cost
- $\mathcal{E}$：所有合法 segment edge + gap edge 的集合
- $\text{lexmin}$：lexicographic minimum（按优先级字典序比较，不是标量加权）
- $C(i,j)$：edge cost vector

Cost vector（公式 4）：

$$C(i,j) = \big(G(i,j),\; P(i,j),\; N(i,j),\; |t_j - t_i - T|,\; B(i,j)\big)$$

各分量含义（**优先级从高到低**）：

1. $G(i,j)$ — gap penalty：未覆盖时间区间的总量，最高优先，避免漏掉 gameplay
2. $P(i,j)$ — **pre-action context shortfall**：关键的 insight 项
3. $N(i,j)$ — segment count preference，倾向于更少的 segment
4. $|t_j - t_i - T|$ — 偏离 target duration 的程度，$T$ 默认 4.0s
5. $B(i,j)$ — boundary preference，软偏好（比如倾向于在 tick 边界而非 mid-tick 切）

### 4.4 Pre-action context 这一项的 intuition

$P(i,j)$ 是整个 DP 设计中最值得品味的一项。论文原话：

> "the visual evidence needed to describe an action often begins before the protected interval itself: view direction, approach motion, hand or weapon preparation, and surrounding context make the subsequent action more interpretable"

考虑投弹这个 action：玩家先调整视角找到投弹方向 → 切到 grenade → 拉环 → 投出。如果 segment 起点 = protected chain 起点，caption 模型看不到"为什么投这里"，只能描述"扔了一个雷"。让 segment 提前 0.5s（默认 desired pre-action context），模型能看到 approach motion 和 aim setup，从而生成"玩家在拐角处预瞄，然后拔出 HE grenade 投向 doorway"这种 grounded caption。

这就是 lexicographic ordering 的妙处：先保证覆盖率（G），再保证 caption quality（P），再保证 segment 数量（N），再保证长度（|·-T|），最后才考虑 boundary preference（B）。每一项只在前一项 tie 时才发挥作用。

---

## 五、Prior-Guided VLM Captioning

### 5.1 为什么 prior filtering 必要

这是 paper 里最 elegant 的设计。两种 failure mode：

**Mode A — 无 priors**：VLM 容易漏掉短而机械的 action（weapon switch、recoil recovery、小幅 camera turn）。这些在 visual 上不显眼，但是是 player input 的关键事实。

**Mode B — priors 过密**：spurious visual grounding。Paper 举的例子特别生动：如果 caption 前一刻有一个小幅 leftward view jitter，紧接着一个明显 right turn 露出 doorway，dense prior 会让 VLM 把 doorway 错误关联到 leftward jitter 上。

所以 prior filtering 不是简单 thresholding，是 **noise vs signal 的因果归因**：保留确实产生 visual change 的 action，抑制 minor noisy motion。

### 5.2 Action priors 的过滤规则

保留：weapon switching、firing、reloads、inspections、grenade prep/flight、scope transitions、melee、short posture/airborne transitions。

移除：`noEffect` actions，即 parsed input 存在但 visual 上没有可靠 effect 的。比如 weapon-switch 动画期间按下 fire，由于 state machine 阻塞，实际不会发射 bullet，所以不能描述为 firing。这一步把 game state machine 的约束 hard-code 进 captioning pipeline。

### 5.3 Movement priors

把 contiguous W/A/S/D 状态 group 成 temporal runs $r = [a, b]$。对每个 run：

公式 (5)：

$$d_r = \sqrt{(x_b - x_a)^2 + (y_b - y_a)^2}, \quad \bar{v}_r = \frac{1}{b-a+1} \sum_{t=a}^{b} v_t^{2D}$$

- $x_a, y_a$ / $x_b, y_b$：run 起止 tick 的 player planar position
- $d_r$：planar displacement（直线位移量）
- $v_t^{2D}$：tick $t$ 的 2D 速度
- $\bar{v}_r$：run 内 mean planar speed
- $b - a + 1$：run 的 tick 数

判定规则：如果同时 $d_r < \tau_d$ 且 $\bar{v}_r < \tau_v$，即使按键 pressed，也视为 **ineffective motion**（玩家按住 W 但被墙挡住）。

这个判定非常关键——它区分了 **input** 和 **outcome**。World model 要学的是"按 W 后会前进"，但 demo 里大量存在"按 W 但没动"的反例（撞墙、被卡住）。如果不过滤，caption 会描述"player moving forward"，但 video 显示静止，模型会学到错误的因果。

剩下的 runs 按 displacement 分桶：
- 低 displacement → 抑制
- 中 displacement → "small position adjustments"
- 高 displacement → "forward / backward / strafing"

### 5.4 Camera priors

先把 per-tick view-angle 位移 aggregate 到 temporal bin $b$（粗粒度时间窗）：

公式 (6)：

$$\Delta_b^\psi = \sum_{t \in b} \Delta\psi_t, \quad A_b^\psi = \sum_{t \in b} |\Delta\psi_t|$$

公式 (7)：

$$\Delta_b^\theta = \sum_{t \in b} \Delta\theta_t, \quad A_b^\theta = \sum_{t \in b} |\Delta\theta_t|$$

- $\Delta_b^\psi, \Delta_b^\theta$：bin $b$ 内 yaw / pitch 的净位移（有符号）
- $A_b^\psi, A_b^\theta$：bin $b$ 内 yaw / pitch 的累计绝对位移
- $\Delta\psi_t, \Delta\theta_t$：来自公式 (1) 的 per-tick 位移

这里已经有 insight：$|\Delta_b| / A_b$ 类似 "sign consistency" ratio。如果 player 在 bin 内左右摇摆（jitter），$\Delta_b \approx 0$ 但 $A_b$ 大；如果稳定向右转，$\Delta_b \approx A_b$。

把 contiguous active bins merge 成 candidate view event $e$：

公式 (8)：

$$\Delta_e^u = \sum_{b \in e} \Delta_b^u, \quad A_e^u = \sum_{b \in e} A_b^u, \quad \rho_e^u = \frac{|\Delta_e^u|}{A_e^u}$$

- $u \in \{\psi, \theta\}$：yaw 或 pitch 轴
- $\Delta_e^u$：event $e$ 内某轴净位移
- $A_e^u$：event $e$ 内某轴累计绝对位移
- $\rho_e^u \in [0, 1]$：方向一致性 ratio

Event 保留条件：$A_e^u$ 和 $|\Delta_e^u|$ 都超过 angular-motion threshold，且 $\rho_e^u$ 显示一致方向。

Yaw events 映射成 "left turn / right turn"，pitch events 映射成 "looking down / raising view"。**关键约束**：camera prior 描述的是 viewpoint continuity，不强制 caption 重复 mechanical input label（避免"鼠标向右移动"这种描述，而是"player turns right to reveal a doorway"）。

### 5.5 VLM 输出 schema

输出强制 JSON，四个 top-level field：

- `scene_draft`：first-person visual details / environment progression / visible effects / chronological events
- `long_prompt`：把 scene_draft 转成 coherent video-generation caption
- `confidence`：模型自评
- `flags`：审计 / 过滤 flag

`scene_draft` + `long_prompt` 的分层是 dual-purpose 设计：scene_draft 给下游 reasoning model 用（结构化、可解析），long_prompt 给 video generation model 用（自然语言）。这种 decoupling 我觉得是未来 caption 数据集的趋势。

---

## 六、和 Related Work 的对比直觉

| Dataset | 视觉 | Action supervision | State supervision | Scene diversity | Limit |
|---|---|---|---|---|---|
| HowTo100M / WebVid | ✓ | 弱（caption） | ✗ | 高 | passive |
| Ego4D / EPIC-KITCHENS | ✓ egocentric | semantic action label | ✗ | 中 | 缺低层 control |
| DROID / Open X-Embodiment | ✓ | low-level action | ✓ | 低 | 真实但贵 |
| MineRL / MineDojo / PLAICraft | ✓ | low-level | ✓ | 中（单游戏） | Minecraft 视觉简单 |
| **EgoCS-400K** | ✓ egocentric, high-fidelity | low-level keyboard/mouse + view-angle | per-tick full state | 中（13 maps） | CS 限定 |

CS 比 Minecraft 的优势：高保真渲染（photo-realistic level）、复杂战术行为（multi-agent、partial observability、utility management）。和 PLAICraft (He et al. 2025, https://arxiv.org/abs/2505.12707) 的关键区别是：PLAICraft 用 instrumented live logging（边玩边记），EgoCS-400K 用 **replay reconstruction**（事后从 demo 解析），所以可以爬公开 pro match 而无需自己采集。

和 GameNGen (Valevski et al. 2024, https://arxiv.org/abs/2408.14837)、Oasis (Decart, https://oasis-model.github.io/)、Genie (Bruce et al. 2024)、Hunyuan-GameCraft (Li et al. 2025, https://arxiv.org/abs/2506.17201) 等 world model 工作相比，EgoCS-400K 是 **数据侧贡献**，而不是模型。它要填的坑是：这些 world model 现在大多在狭窄数据上训（DOOM、Minecraft screenshot、单游戏自采 trace），缺乏大规模、高保真、多模态对齐的公开训练集。

---

## 七、Limitations 我自己的解读

Paper Section 4.3 提到几个 domain gap，我想更细化：

1. **Action space 是离散 keyboard + 连续 view-angle**，不是 robotic 的 7-DoF end-effector。直觉上 mouse → view-angle 这个 mapping 比 keyboard → joint torque 简单得多，但 world model 学到的 "view turn → scene rotate" 这种 egocentric 几何 prior 是可迁移的。
   
2. **Physics 是 game physics，不是 Newtonian physics**。弹道、碰撞、烟雾扩散都遵循 game state machine。这意味着模型学到的 "投弹 → 抛物线 → 落地 → 爆炸" 是 game-tuned 的，迁移到真实物理环境需要 fine-tune。

3. **没有 tactile feedback 和 deformable object**。CS 里不能"按压"、"揉捏"、"撕开"，所有 object 都是 rigid。
   
4. **Caption 是 VLM-generated 不是 gold**。虽然有 prior 约束和 confidence/flag 字段，但 hallucination 风险存在。downstream 使用者应该把 `confidence < threshold` 的样本过滤掉。
   
5. **Behavioral distribution偏 combat**。没 cooking、cleaning、conversation 等日常行为。

但作为 **intermediate testbed**，它的价值在于：让 world model 在"dense supervision + 高保真视觉 + 复杂 multi-agent 场景"三者齐全的环境下先验证 scaling law，再迁移到真实 embodied 场景。这和 ImageNet 之于 visual recognition 的角色类似——不是最终目标，但提供了必要的 pretraining signal。

---

## 八、几个我自己的延伸联想

1. **Replay file 作为 audit path 的思想可以推广**。任何能 record trajectory 的环境（赛车 replay、StarCraft replay、ROS bag）都可以套用这套 pipeline：replay 作为 ground truth timeline，render / extract 作为多模态 supervision。这给 embodied AI 提供了一个比 robotic logging 便宜得多的数据范式。

2. **Pre-action context $P(i,j)$ 的设计可以反过来用于 RL**。在 imitation learning 里，expert demonstration 的 pre-action state（approach、aim、setup）往往比 action 本身更有信息量。这套 DP segmentation 思想可以直接迁移到 robot learning 的 clip 切分。

3. **Prior-guided VLM captioning 范式对自动驾驶也适用**。nuScenes / Waymo 都有 raw sensor + CAN-bus signal，但 caption 通常是 free-form。如果把 CAN signal 转成 action/movement/camera priors，再约束 VLM caption，可以得到和 EgoCS-400K 类似的、grounded 的 driving caption。

4. **Lexicographic cost vector 的思想**。在多目标 optimization 里，scalar weighting 总是要 tune weight，而 lexicographic ordering 只需要确定 priority。这套设计在数据集切分、RL reward shaping、curriculum learning 里都适用。

5. **Mouse proxy 的局限**：用 view-angle delta 代替 raw mouse counts 假设了 sensitivity 不变，但 pro player 经常动态调 sensitivity。不过对 world model 训练而言，这恰好是 invariance——我们要学的是"视角变了 → 场景变了"，sensitivity 的具体值不影响这条因果链。

References for further reading:
- Project: https://EgoCS-400K.github.io  
- Genie (Bruce et al. 2024): https://arxiv.org/abs/2408.14837 推到 ICML  
- GameNGen (Valevski et al. 2024): https://arxiv.org/abs/2408.14837  
- Oasis (Decart): https://oasis-model.github.io/  
- PLAICraft (He et al. 2025): https://arxiv.org/abs/2505.12707  
- π0 (Black et al. 2024): https://arxiv.org/abs/2410.24164  
- OpenVLA (Kim et al. 2025): https://arxiv.org/abs/2406.09246  
- DROID (Khazatsky et al. 2024): https://arxiv.org/abs/2403.12945  
- Ego4D (Grauman et al. 2022): https://arxiv.org/abs/2110.07058  
- MineDojo (Fan et al. 2022): https://arxiv.org/abs/2206.08853  
- MineRL (Guss et al. 2019): https://arxiv.org/abs/1904.08208  
- Hunyuan-GameCraft 2 (Tang et al. 2025): https://arxiv.org/abs/2511.23429  
- HunyuanWorld (2025): https://arxiv.org/abs/2507.21809  
- The Matrix (Feng et al. 2025): https://arxiv.org/abs/2412.03568  
- WorldScape: https://worldscape.io/

如果你接下来想深入某个方向，我建议优先看：(a) DP segmentation 的 cost vector 在你自己 dataset 上的迁移可能性；(b) prior filtering 的双 failure mode 框架在 VLM hallucination 控制中的泛化；(c) replay-grounded 这个思想在你自己关心领域（robotics? autonomous driving?）有没有可对应的 audit-able source of truth。这三点都是从 paper 里最直接可迁移的 methodology contribution。
