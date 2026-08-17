---
source_pdf: EGOTRAJ REAL-WORLD EGOCENTRIC HUMAN TRAJECTORY DATASET FOR MULTIMODAL
  PREDICTION.pdf
paper_sha256: c6789c21ac3754c8b4f240cc614ab7712284681a8b02fa63e9ec08d9dbcfea82
processed_at: '2026-08-04T02:51:05-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 EgoTraj

好的，用更口语的方式重新讲一遍，重点是 build intuition，让你能 get 到这个工作真正在解决什么问题、为什么这么设计、实验数据背后的故事是什么。

---

## 这篇 paper 到底在干嘛

一句话：**给人戴上一个 AR 头显（Meta Quest Pro），让他在真实城市街道里随便走，同时把他的头怎么动、眼睛看哪里、看到的画面是什么全部同步录下来，然后用这些信号去预测他接下来 3.5 秒会走到哪里。**

就这么简单。但背后的 motivation 很深。

---

## 为什么这件事难，为什么之前没人做好

传统的 trajectory prediction dataset（ETH、UCY、Stanford Drone、nuScenes、Waymo 这些）都是从天上往下看，或者固定在墙上的 camera 拍的。你只能看到"这个人在 BEV 坐标系下从 A 走到 B"，但你不知道他**为什么**这么走 — 他看到了什么、他在注意什么、他脑子里在 plan 什么。

这就像你只看一个 chess player 的棋谱，但看不到他在看哪个棋子、在 think 什么下一步。你能预测他下一步走哪吗？能，但很 limited。

Egocentric 视角给你的是完全不同的 information：你能直接 access 到这个人**感知世界的方式**。他头转哪里、眼睛看哪里，这些 gaze signal 本质上是**意图的 leading indicator**。

这里有个超级关键的心理学发现，是 Land 在 2006 年发表的（ref [17]）：**人在执行 motor action（比如转弯）之前 1-2 秒，眼睛已经先 fixate 到目标位置了。**

这个 1-2 秒的时间常数是整篇 paper 的物理根基。它意味着：如果你想预测一个人 3.5 秒后的位置，而你只能观测他过去 1.5 秒的运动学，那你观测窗口里**刚好包含**他 gaze 信号，但**可能还没看到** motor 信号。gaze 比 motion 提前 1-2 秒，所以 gaze 给了你一个"提前量"，让你能在 motor action 还没发生时就预判到。

这是为什么这篇 paper 的 ablation study 里，gaze 单独加进去就能把 ADE 从 0.19 m 降到 0.15 m — 降幅 21%。这个降幅不是 magic，是 Land 2006 的心理学规律在数据上的直接体现。

---

## Dataset 设计里真正聪明的地方

### 1. waypoints 是 landmark 不是 GPS point

paper 里 7 个 waypoints 是 UT Austin 校园 + 周边商业区的 7 个 landmark（比如某个 building、某个 bus stop、某个 plaza）。参与者从 7 个里选 2 个，自由走过去。不是 scripted route。

为什么这重要？因为如果 waypoints 是精确 GPS point，参与者会走"最短路径"，路径自由度很低，dataset 会变成"template replay"。用 landmark 的话，同一段从 waypoint A 到 waypoint B 的路径，不同人会走完全不同的线。

paper 用 DTW（Dynamic Time Warping）距离验证了这点：同一对 waypoint 之间，103 次 within-pair 比较，median 距离 122.8 米，range 从 19.3 到 379.4 米。这个 spread 说明路径选择是真实的"自由变量"，不是 templated。

### 2. Meta Quest Pro 而不是 Project Aria

Aria 是 Meta 的研究专用设备，sensor 更丰富，但部署成本高、参与者需要专门培训。MQPro 是商用 MR 头显，passthrough 模式下用户能看到真实世界，所以能"自然行走"。

trade-off 是：MQPro 的 passthrough camera 分辨率 1024×1024，比 Aria 的 1408×1408 小；FOV ~90°，比 Aria 130° 窄。但 75 个 participant 用 MQPro 是可复制的，用 Aria + Xsens mocap suit 就不太可能大规模 deploy。

这个 trade-off 本质是：**sensing richness vs deployment cost**。EgoTraj 选了后者，因为它要让"大规模 in-the-wild 采集"成为可能。

### 3. 75 participants, 每人恰好一个 session, 严格 subject-disjoint

这点很重要。如果同一个 participant 贡献多个 session，那 train/test split 里 test set 可能包含"已见 participant 的新 session"，模型会过拟合到 participant-specific gait pattern。EgoTraj 强制每人一个 session，所以 train/test split 是严格 subject-disjoint，model 必须 generalize 到**完全没见过的 walker**。

### 4. 31/75 participant 对该区域不熟悉

paper 专门设计了 "Unfamiliar" split（8 个 unfamiliar participant 的 session 全 hold out 到 test）。结果发现：纯运动学 baseline 在 unfamiliar split 上 ADE 0.23 m，比 random split 的 0.19 m 高 21% — 说明人对环境熟悉时 navigation 行为更果断，运动学更平稳，反而更难预测（因为没有 hesitation signal）。

但加了 full multimodal（V+P+S+G）之后，unfamiliar split 的 ADE 降到 0.14 m，跟 random split 的 0.12 m 差距收缩到 16%。这是 multimodal 信号 generalize 到新环境的直接证据。

---

## 传感器同步的工程难点

MQPro 内部有两个进程同时跑：
- Unity app @ 50 Hz 记录 head pose + gaze + IMU，写 CSV
- Python 脚本 @ 30 fps 录 RGB video，写 MP4

这两个进程用 file-based signal 同步：Unity 写 `start_signal` 文件 → Python 检测到后启动 video capture → Python 写 `video_ready` → Unity 开始正式记录 sensor → 参与者按 B 按钮 → Unity 写 `stop_signal` → 两进程同时关闭。

同步精度上限是 frame interval，大约 5.893 ms（1/30 fps 的一半左右）。

后处理阶段统一时间轴：
- position 用 linear interpolation（position 在 R³，是线性空间，linear interp 没问题）
- orientation quaternion 用 SLERP（quaternion 在 S³ 流形上，linear interp 会破坏 unit norm 约束，必须用 SLERP 保证 geodesic 路径）
- gaze 用 normalized linear interp（gaze 是 direction vector，interp 后要 renormalize 到单位长度）

SLERP 公式直觉：
$$
\text{SLERP}(q_0, q_1; t) = \frac{\sin((1-t)\theta)}{\sin(\theta)} q_0 + \frac{\sin(t\theta)}{\sin(\theta)} q_1
$$
$t$ 是插值参数（0 到 1），$\theta = \arccos(q_0 \cdot q_1)$ 是两个 quaternion 之间的夹角。这个公式保证插值结果始终在 $S^3$ 单位球面上，不会"塌陷"到内部。

---

## Gaze calibration — 一个被 paper 轻描淡写但其实很 tricky 的细节

MQPro 的 eye tracking 给你的是 3D gaze origin + 3D direction vector（相对于 headset 坐标系）。但 RGB passthrough camera 跟 eye 不是同一个位置 — 眼睛在 headset 内部偏上，camera 在鼻梁前方，两者有大约 2 cm 的 translation offset，加上 lens 的 radial distortion。

所以直接用 pinhole camera model 投影 gaze 到 image plane 会有系统性误差。

paper 的解法是 per-session quadratic calibration：
$$
u = a_0 + a_1 \theta + a_2 \phi + a_3 \theta \phi + a_4 \theta^2 + a_5 \phi^2
$$
$$
v = b_0 + b_1 \theta + b_2 \phi + b_3 \theta \phi + b_4 \theta^2 + b_5 \phi^2
$$
其中 $\theta$ 是 gaze yaw 角，$\phi$ 是 gaze pitch 角。系数 $(a_i, b_i)$ 对每个 session 单独 fit。

直觉：
- linear 项 $a_1\theta, b_2\phi$ 对应理想 pinhole 投影
- quadratic 项 $a_4\theta^2, b_5\phi^2$ 吸收 lens radial distortion（barrel/pincushion）
- cross term $a_3\theta\phi$ 吸收 tangential distortion + eye-to-camera translation offset

paper 没给 calibration error 数字（这部分其实应该补），但 Figure 4 显示投影 gaze 红点跟实际注视点 alignment 不错。

---

## Scene annotation pipeline — VLM 已经能做 human-annotator 级别的标注了

paper 用 Qwen2.5-VL-7B-Instruct 给 38,606 帧自动生成 scene annotation。CoT prompting 5 步：

1. 环境上下文（crosswalk / sidewalk / intersection）
2. 动态 agent（pedestrian / vehicle / cyclist）
3. traffic signal + obstacle
4. gaze fixation target
5. inferred short-term intent

输出 JSON sidecar file，indexed by session + frame ID。

质量评估（Table 6）：
- Environmental context: Cohen's κ = 0.89, VLM accuracy 0.91
- Dynamic agents: κ = 0.87, accuracy 0.89
- Traffic signals: κ = 0.96, accuracy 0.92
- Gaze target: κ = 0.95, accuracy 0.98（最高，因为 gaze 已投影成红点，VLM 只需 object detection at marker location）
- Short-term intent: κ = 0.83, accuracy 0.84（最低，因为 inherently subjective）
- Structural compliance: 96%（after up to 2 retries）

κ > 0.8 在 social science 里就是 strong agreement，κ > 0.93 几乎是 perfect。这说明 VLM 在 navigation context annotation 上已经达到 human-annotator 可靠度。未来的 dataset annotation pipeline 会越来越自动化，这是 trend。

失败模式 audit（50 帧）：
- 漏检 pedestrian 4%
- 误读 traffic light 3%
- gaze 目标 mismatch 2%
- 总错误率 < 10%，集中在视觉模糊场景（远距离、occlusion、boundary case）

---

## Baseline — 5 个模型对比

paper 不得不自己 re-implement 所有 baseline，因为：
- EgoNav / LookOut / EgoCogNav 都没公开代码（只 arXiv preprint）
- Social-LSTM / TUTR 是 BEV dataset 训练的，不兼容 egocentric 6DoF pose

所以 paper 选了 5 个 representative baseline：

### 1. Const_Vel（纯运动学外推）
用最后几步的 linear velocity $\vec{v}$ 和 angular velocity $\vec{\omega}$ 做 body-frame 匀速外推：
$$
\hat{p}_{t+\Delta t} = p_t + R_t \vec{v} \Delta t
$$
$$
\hat{R}_{t+\Delta t} = \exp(\Delta t \cdot \vec{\omega}^\wedge) R_t
$$
$\vec{\omega}^\wedge$ 是 angular velocity 的 skew-symmetric matrix（so(3) Lie algebra），$\exp$ 是 matrix exponential（Lie algebra → Lie group SO(3) 的指数映射）。

这个 baseline 简单但合理 — 行人短时间内确实接近匀速。ADE 0.24 m 就是这个"匀速假设"在 3.5 s 窗口下的天然下限。

### 2. Lin_Ext（per-axis linear regression）
对每个轴 $(x, y, z)$ 和每个 quaternion 分量 $(q_w, q_x, q_y, q_z)$ 单独做 linear regression。

这个 baseline 在 head rotation 上灾难性失败（$L_{1\text{head}} = 1.39$），因为 quaternion 不是线性空间，axis-wise linear regression 会破坏 unit norm 约束，给出非单位 quaternion，orientation 直接 garbage。

ADE 0.26 m 反而比 Const_Vel 的 0.24 m 还差 — linear regression 对 1.5 s 短序列过拟合，不如简单 velocity 外推。

### 3. M_Transformer（vanilla Transformer, early fusion）
把所有 modality embedding 直接 concatenate，过 temporal Transformer encoder + linear head 输出。ADE 0.20 m, $L_{1\text{head}}$ 0.74。

这是 LookOut paper 的标准 baseline，EgoTraj 直接复用。

### 4. CXA-Transformer（cascaded cross-attention, Qiu et al. RA-L 2022, ref [30]）
多 stream 架构，ego-translation 和 ego-rotation 各走独立 transformer encoder，然后 cascaded cross-attention fusion：ego stream 作为 query，social/scene stream 作为 key/value，sequential 融合。

ADE 0.19 m, $L_{1\text{head}}$ 0.69（head rotation 最优）。这个架构是 paper ablation study 的 base architecture，因为 cascade 结构适合"逐步加 modality"的 ablation 实验。

### 5. EgoCast（WACV 2025, ref [5]）
原设计是 full-body pose forecasting，paper 把它的 Transformer forecasting module 改造，直接吃 past ego-translation + head rotation，去掉 pose-estimation stage。

ADE 0.16 m（最优 trajectory 误差），但 $L_{1\text{head}}$ 0.78（head rotation 不如 CXA）。说明 EgoCast 的 temporal head 更擅长 position prediction，而 CXA 的 cross-attention 对 orientation 的 SO(3) 流形结构更友好。

---

## Ablation Study — 这是全 paper 信息密度最高的部分

base architecture: CXA-Transformer。每次只加一种 modality，看 ADE 变化。

modality 定义：
- $\mathcal{V}$: ego-trajectory（position + rotation）
- $\mathcal{C}$: nearby people center points（YOLOv8-Pose 的 torso centroid）
- $\mathcal{B}$: nearby people bounding boxes
- $\mathcal{P}$: nearby people full body pose（17 个 COCO keypoint）
- $\mathcal{S}$: semantic segmentation（OneFormer）
- $\mathcal{D}$: relative depth（Depth Anything V2）
- $\mathcal{G}$: gaze 投影到 $(u, v)$

### 单 modality 增量效果排序（vs $\mathcal{V}$ baseline 的 ADE 0.19 m）

| 加什么 | ADE | 降幅 | 直觉解读 |
|---|---|---|---|
| + $\mathcal{S}$ | 0.16 | -15.8% | segmentation 直接告诉你"哪里可走"，把无穷解空间约束到 sidewalk，prior 很强 |
| + $\mathcal{G}$ | 0.15 | -21.1% | gaze 领先 motor 1-2 s，anticipatory signal 最强 |
| + $\mathcal{P}$ | 0.17 | -10.5% | body pose 包含 turn-anticipation（肩膀先转），比 center/bbox 好 |
| + $\mathcal{D}$ | 0.18 | -5.3% | outdoor 平面环境，depth 主要是远处建筑，携带 walkable 信息少 |
| + $\mathcal{C}$ | 0.18 | -5.3% | center point 只给位置，不给姿态，信息量低 |
| + $\mathcal{B}$ | 0.18 | -5.3% | bbox 给 spatial extent 但不直接给"对面行人朝哪走"，反而干扰 head rotation（$L_{1\text{head}}$ 从 0.69 恶化到 0.81） |

**关键 insight**: gaze 单加效果最强，验证了 Land 2006 的心理学规律。scene segmentation 第二强，因为它提供的是"拓扑约束"（哪里可走），比几何约束（depth）更 fundamental。body pose 比 center/bbox 好，因为 fine-grained limb motion 包含 turn-anticipation signal。

### gaze + social 组合的 synergy

| 组合 | ADE | 解读 |
|---|---|---|
| $\mathcal{V} + \mathcal{P} + \mathcal{G}$ | 0.12 | 强 synergy — 行人姿态 + ego 注意力双重确认 |
| $\mathcal{V} + \mathcal{B} + \mathcal{G}$ | 0.16 | 弱 synergy — bbox 不给姿态信息 |
| $\mathcal{V} + \mathcal{C} + \mathcal{G}$ | 0.14 | 中等 synergy |

直觉：如果 ego 在 gaze 一个正在转弯的行人，模型可以从该行人的 body pose 推断 ego 的避让方向。gaze + pose 给了"ego 在看谁 + 那个人在做什么"的双重信号。

### 最优组合 $\mathcal{V} + \mathcal{P} + \mathcal{S} + \mathcal{G}$

ADE 0.12 m, FDE 0.23 m, $L_{1\text{head}}$ 0.58 — 全部最优。

这意味着：在 1.5 s 观测 + 3.5 s 预测这个时间尺度下，ego-motion + social pose + scene segmentation + gaze 这四种 modality 在数学上接近互补。gaze 在已有 social + scene 的基础上还能把 ADE 从推测的 ~0.13 降到 0.12（paper 没列 $\mathcal{V} + \mathcal{P} + \mathcal{S}$ 单独行，但可以从 $\mathcal{V} + \mathcal{P}$ 0.17 + $\mathcal{V} + \mathcal{S}$ 0.16 推测）。

---

## Generalization across splits — 验证模型不是过拟合到路径模板

paper 设计了三个 split：

1. **Random Participant** (n=8): 标准 random split
2. **Waypoint Held-Out** (n=10): 3/21 origin-destination 对完全 hold out
3. **Unfamiliar** (n=8): 31 名 unfamiliar participant 中 8 个 hold out

Table 4 数据（ADE，带 95% bootstrap CI from 1000 resamples）：

| Modality | Random | Waypoint Held-Out | Unfamiliar |
|---|---|---|---|
| $\mathcal{V}$ | 0.19±.014 | 0.21±.018 | 0.23±.019 |
| $\mathcal{V} + \mathcal{P}$ | 0.17±.011 | 0.19±.015 | 0.20±.013 |
| $\mathcal{V} + \mathcal{S}$ | 0.16±.013 | 0.18±.012 | 0.18±.016 |
| $\mathcal{V} + \mathcal{G}$ | 0.15±.009 | 0.16±.014 | 0.16±.010 |
| $\mathcal{V} + \mathcal{P} + \mathcal{S} + \mathcal{G}$ | **0.12±.008** | **0.14±.010** | **0.14±.012** |

**关键观察**：
- $\mathcal{V}$ only 上，Unfamiliar ADE (0.23) 比 Random (0.19) 高 21% — 纯运动学对环境熟悉度敏感
- 加了 full multimodal 后，三个 split 的 ADE 是 0.12 / 0.14 / 0.14，差距收缩到 16% — multimodal 信号 generalize 到新环境和新 waypoint 对
- 95% CI 不重叠（$\mathcal{V}$ only 的 Random CI [.176, .204] vs full multimodal 的 [.112, .128]）— 统计上显著改进
- Waypoint held-out 和 Unfamiliar 上的 ADE 都是 0.14 — "没见过的 waypoint 对"和"不熟悉环境的参与者"对模型挑战程度类似

---

## Active-transition windows — gaze 价值的严格测试

paper 定义了一个子集："active transition" — turning 行为开始于观测窗口 $T_{\text{obs}}$ 最后 0.5 s 内。这意味着大部分方向变化落在预测窗口，观测窗口里 motor signal 还没表达 turn intent。

这是 gaze 价值的最严格测试场景：如果 gaze 真的 lead motor 1-2 s，那 active-transition window 里 gaze 应该已经在过去 1.5 s 观测中 fixate 了 turn target，而 motion signal 还平稳。

结果：full multimodal CXA-Transformer 在 active-transition subset 上 ADE 0.23±0.022, FDE 0.38±0.028。

这个数字比 random split 的 ADE 0.12 高，但比 motion-only baselines 在 active-transition 上"drift outward"的灾难情形好得多（Figure 8 显示 Const_Vel / Lin_Ext / position-only 沿 pre-turn heading 飘出，偏离 ground truth 几米）。

直觉：gaze 帮模型"提前知道"要转弯，但 sharp turn 本身仍然是 deterministic prediction 的硬上限。

---

## Failure case — 90° 急转弯

Figure 6 的 Window-Slice 2512 是典型失败案例：参与者在 traffic light 转绿时突然 90° 转弯。

所有 baseline 都不能正确预测曲率：
- Const_Vel / Lin_Ext: 继续沿 pre-turn heading 飘
- 神经网络模型: 部分预判方向变化，但低估 turn magnitude

原因：
- 1.5 s 观测窗口里 linear velocity 和 angular velocity 都在 normal walking range
- 没有 strong anticipatory cue 表明马上急转
- gaze 虽然已 fixate 新方向，但模型对 gaze → sharp turn 的 mapping 还没学到
- dataset 里这种 abrupt transition 样本太少

paper 末尾暗示 future work 方向：**intent-aware + uncertainty-aware probabilistic forecasting**。deterministic prediction 在 multimodal intent 下 inherently 难 — 同一个观测可能对应多个未来轨迹（直行 vs 转弯），必须输出 distribution 而非 point estimate。

---

## Cascaded cross-attention fusion 架构（Figure 13）

四个独立 transformer encoder streams：
1. Ego-motion stream: 输入 45 timestep × 7 DoF (3 pos + 4 quat)，输出 $E_V \in \mathbb{R}^{45 \times d}$
2. Social stream: 输入 $J_n \in \mathbb{R}^{T_{\text{obs}} \times 34}$ per pedestrian（17 COCO keypoint × 2 坐标），输出 $E_P$
3. Scene stream: 输入 segmentation feature $S \in \mathbb{R}^{T_{\text{obs}} \times k}$，输出 $E_S$
4. Gaze stream: 输入 $G = \{(u_t, v_t)\}$，输出 $E_G$

Cascade fusion 顺序：V → P → S → G

Stage 1:
$$
F_1 = \text{CrossAttn}(Q=E_V, K=E_P, V=E_P)
$$

Stage 2:
$$
F_2 = \text{CrossAttn}(Q=F_1, K=E_S, V=E_S)
$$

Stage 3:
$$
F_3 = \text{CrossAttn}(Q=F_2, K=E_G, V=E_G)
$$

$F_3$ 过 decoder + linear head 输出未来轨迹。

**为什么 cascade 比 concat 好**：
- concat 是"所有 modality 被动塞给 model"，model 要自己学会怎么 weight
- cascade 是"主 modality 主动 select 它需要的上下文"，ego-motion 始终作为 query
- cascade 顺序按 abstraction level 递增：motion (最 raw) → pose (peer-level) → scene (environment-level) → gaze (cognitive-level)
- 这跟 Perceiver / Flamingo 的 "Q from main modality, K/V from auxiliary" 设计哲学一致

Cross-attention 公式：
$$
\text{CrossAttn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$
$Q \in \mathbb{R}^{N \times d_k}$（ego stream tokens），$K \in \mathbb{R}^{M \times d_k}$（auxiliary modality tokens），$V \in \mathbb{R}^{M \times d_v}$，$\sqrt{d_k}$ scaling 防止 softmax 饱和。Output 是 $N \times d_v$，每个 ego token 对应一个加权平均的 auxiliary 信号。

---

## 我的 takeaway（build intuition）

如果让我从这篇 paper 提炼几个最核心的 insight：

1. **gaze leads motor by 1-2 s** 是 egocentric trajectory prediction 的物理根基。这个 1-2 s 时间常数决定了 $T_{\text{obs}}$ 至少要 ≥1 s 才能让 gaze 进入观测窗口。paper 用 1.5 s 观测 + 3.5 s 预测，刚好覆盖这个 lead time。

2. **scene segmentation > depth > pose > bbox** 在这个 task 上的边际收益排序。这反映了 outdoor pedestrian navigation 的本质：拓扑约束（哪里可走）> 几何约束（距离）> agent 互动（行人姿态）> 局部外观（bbox）。

3. **cascaded cross-attention 比 simple concat 强**，因为 multimodal transformer 的难点在"异构信号融合"。cascade 让主 modality 主动 select 上下文，不是被动 concat。Table 2 里 CXA 的 $L_{1\text{head}}$ 0.69 比 M_Transformer 的 0.74 好，就是 cascade 对 orientation SO(3) 流形更友好的直接证据。

4. **急转弯是 deterministic prediction 的硬上限**。必须走向 probabilistic multimodal forecasting — 输出 distribution 而非 point estimate。这是 paper 留给 future work 的核心方向。

5. **AR headset 比 research-grade sensor suite 更适合大规模采集**。MQPro 是商用设备，75 个 participant 用它采集是可复制的。用 Aria + Xsens mocap suit 就不太可能 scale 到这个量级。trade-off 是 sensing richness vs deployment cost，EgoTraj 选了后者。

6. **数据集设计的"自由度"很关键**。waypoints 是 landmark 不是 GPS point，DTW 距离证明 path 自由度真实存在（median 122.8 m，range [19.3, 379.4] m）。这是 dataset 不是 scripted trajectory replay 的关键证据。

7. **VLM annotation 已经达到 human-annotator 可靠度**。Qwen2.5-VL-7B + CoT 在 5 个 annotation field 上 Cohen's κ > 0.83，structural compliance 96%。未来 dataset annotation pipeline 会越来越自动化。

8. **80/10/10 session-level + subject-disjoint + 95% bootstrap CI** 是这类 dataset paper 的统计严谨度新标准。之后做 dataset benchmark 都应跟进。

---

## 延伸联想（宁可 hallucinate 也不漏）

- **与 Ego-Exo4D 的联系**（CVPR 2024, ref [9]）：Ego-Exo4D 是 ego + exo dual perspective 的 skilled activity dataset，关注 manipulation；EgoTraj 关注 navigation。两者未来可能合并成通用 egocentric behavior dataset。

- **与 Humanoid robot navigation**：ARMOR（ref [16]）用 egocentric perception 做 humanoid collision avoidance。EgoTraj 直接给这类方法提供训练数据。作者是同一个人（Yehia），前作 XR-DT 和 ARCAS 都是 humanoid/AR 相关。

- **与 Social-Implicit**（ECCV 2022, ref [23]）：本文 co-author Abduallah Mohamed 的 prior work，提出 IMLE（Implicit Maximum Likelihood Estimation）做多模态 trajectory sampling。完全可以套到 EgoTraj 上做 multimodal intent distribution prediction — 这正好解决 paper 末尾提出的 sharp turn 难题。

- **与 Diffusion-based trajectory prediction**：MID（Mao et al. CVPR 2023）和 LED（Shi 2024）用 diffusion model 做 multi-future trajectory sampling。EgoTraj 的 90° 急转弯 case 是 diffusion-based 方法的 natural target — 用 gaze 作为 conditioning signal，diffusion 可以 sample 出"直行 vs 左转 vs 右转"的 multi-future distribution。

- **与 PDE-based traffic flow modeling**：Christian Claudel（co-author）的本职研究是 Lighthill-Whitham-Richards (LWR) traffic PDE。paper 没明显结合，但理论上 pedestrian flow 也可以做 continuum PDE 建模。EgoTraj 的 head pose 数据可以做"个体 fluid particle trajectory"采样源，用 stochastic PDE 拟合 crowd flow。

- **与 Tesla FSD 的 pedestrian prediction**：自动驾驶里 pedestrian intent prediction 跟 egocentric pedestrian trajectory prediction 是 dual problem — vehicle-ego predicting other-pedestrian vs pedestrian-ego predicting self-trajectory。两者数据可能 cross-fertilize。

- **与 Apple Vision Pro**：MQPro 是这篇 paper 的 hardware，AVP 没被使用但同样有 eye tracking + 6DoF tracking + passthrough。未来在 AVP 上复现 EgoTraj 协议是明显的 follow-up。

- **与 NVidia Isaac Sim + synthetic data**：EgoTraj 是真实数据，75 subject。可以用 Isaac Sim + procedurally generated urban scenes 做数据增强，pretrain on synthetic + finetune on EgoTraj。这正好是 synthetic-to-real transfer 的标准配方。

- **与 CityGML / digital twin**：作者前作 XR-DT 用 AR 增强 digital twin。EgoTraj 可以视为 digital twin 里 pedestrian agent 的"behavioral ground truth"采集 pipeline。

---

## Reference links

- 项目主页与代码: https://github.com/yehiahmad/EgoTraj
- 作者前作 ARCAS: https://arxiv.org/abs/2512.05299
- 作者前作 XR-DT: https://arxiv.org/abs/2512.05270
- LookOut (ICCV 2025): https://arxiv.org/abs/2506.21450
- EgoCogNav: https://arxiv.org/abs/2511.17581
- EgoNav: https://arxiv.org/abs/2403.19026
- EgoCast (WACV 2025): https://openaccess.thecvf.com/content/WACV2025/papers/Escobar_EgoCast_Forecasting_Egocentric_Human_Pose_in_the_Wild_WACV_2025_paper.pdf
- Project Aria: https://www.projectaria.com/
- Land 2006 (gaze leads motor): https://www.sciencedirect.com/science/article/pii/S1350946205000562
- Nymeria (ECCV 2024): https://arxiv.org/abs/2406.05823
- Ego4D (CVPR 2022): https://arxiv.org/abs/2110.06091
- Ego-Exo4D (CVPR 2024): https://arxiv.org/abs/2404.04458
- Social-Implicit (ECCV 2022): https://arxiv.org/abs/2209.13024
- Qwen2.5-VL: https://arxiv.org/abs/2409.12191
- Depth Anything V2: https://arxiv.org/abs/2406.09463
- OneFormer (CVPR 2023): https://arxiv.org/abs/2211.06586
- YOLOv8: https://github.com/ultralytics/ultralytics
- EgoBlur: https://arxiv.org/abs/2309.04076

---

总结一句人话：**这篇 paper 给人戴上 Meta Quest Pro 在真实城市里走，同时录下头怎么动、眼睛看哪、看到什么画面，然后证明"眼睛看哪里"这个信号能让你提前 1-2 秒知道这个人接下来要往哪走，比单纯看运动轨迹准 21%。这是第一个在 outdoor + 75 人 + gaze + 6DoF + scene annotation 五维度同时满足的 AR-headset-based trajectory dataset，给未来"意图感知的概率轨迹预测"铺了路。**

---

# EgoTraj 深度解析

这份paper是 UT Austin 的 MASSlab 与 Honda Development & Manufacturing of America (HDMA) 协作完成的成果, 由 Christian Claudel (教授, civil engineering 方向, 专长 traffic sensing, queue theory, PDE-based traffic modeling) 和 Junfeng Jiao (建筑学院, 专长 urban informatics, VR/AR for built environment) 联合指导。一作 Ahmad Yehia 此前有 XR-DT 与 ARCAS (AR collision avoidance for VRU) 的 arXiv 工作, 可视为这篇 EgoTraj 的技术铺垫。通讯作者 Abduallah Mohamed 现在在 Meta Reality Labs, 也参与过 Social-Implicit (ECCV 2022) 等行人轨迹预测经典工作, 所以这篇 paper 把"egocentric sensing + AR headset + trajectory prediction"三者缝合起来并不奇怪。

参考链接：
- 项目主页与代码: https://github.com/yehiahmad/EgoTraj
- 作者前作 ARCAS (arXiv:2512.05299): https://arxiv.org/abs/2512.05299
- 作者前作 XR-DT (arXiv:2512.05270): https://arxiv.org/abs/2512.05270
- LookOut (ICCV 2025, 文中主要参考): https://arxiv.org/abs/2506.21450
- EgoCogNav (arXiv:2511.17581): https://arxiv.org/abs/2511.17581
- EgoNav (arXiv:2403.19026): https://arxiv.org/abs/2403.19026
- Project Aria 系列 (Nymeria, ADT, AEA): https://www.projectaria.com/
- EgoCast (WACV 2025): https://openaccess.thecvf.com/content/WACV2025/papers/Escobar_EgoCast_Forecasting_Egocentric_Human_Pose_in_the_Wild_WACV_2025_paper.pdf

---

## 1. 为什么这篇 paper 重要 (build intuition)

人类导航 = perception + planning + control, 其中 perception 在第一人称视角下天然高效。传统的 pedestrian trajectory prediction datasets (ETH/UCY/Stanford Drone/inD/nuScenes/Waymo/JRDB) 都是 BEV 或固定相机视角, 只记录"where people move", 不记录"how they perceive", 缺少 gaze/head-pose 这类 intention cue。EgoTraj 把这一缺口补上, 把 trajectory prediction 从"外观察运动学"推到"内省视觉注意力 + 运动学"。

直觉上关键 insight: pedestrian 在转弯前 1-2 秒会先 fixate 转弯点 (Land 2006, 文中 ref [17])。所以 gaze 是 heading change 的 leading indicator, 比 velocity 的导数提前一两秒。这是整篇 paper 的物理直觉基础, 也是 ablation study 里 gaze 单独加进去就能把 ADE 从 0.19 m 降到 0.15 m 的根本原因。

---

## 2. Dataset 的硬件与采集协议

### 2.1 硬件: Meta Quest Pro (MQPro)

MQPro 关键传感器:
- Passthrough RGB camera (1024×1024, 30 fps, 满足 MR full-color passthrough)
- 2× infrared eye-tracking cameras (用于估计 binocular gaze)
- 4× wide-angle monochrome tracking cameras (用于 inside-out VIO SLAM)
- 6-axis IMU

这是与 Aria (Project Aria) 最大的差异: Aria 是研究专用设备, 体积小但定制性强, 而 MQPro 是商用 MR 头显, passthrough 模式下使用者可以"看到"真实世界, 因此更适合"in-the-wild 自然行走"的实验范式。代价是: passthrough 视频画质 (1024×1024) 比 Aria 的 1408×1408 小, 视角 FOV 也更窄 (~90° vs Aria 130°), 而且存在 eye-to-camera 几何 misalignment (眼睛位置 ≈ 2 cm 高于 passthrough camera, 这个 disparity 没在 paper 里显式建模, 而是用 quadratic calibration 吸收掉)。

### 2.2 录制协议细节 (paper Sec 3.1 + Appendix A)

- 30 Hz RGB passthrough 视频 (H.264 MP4)
- 50 Hz head pose + gaze (CSV), 实际公布版本 resample 到 30 Hz 与视频对齐
- 7 个 outdoor waypoints 分布在 UT Austin 周边校园 + 商业混合街区, 含 34 个 bus stop + signalized intersections + 弯道 + 窄廊道
- 21 个 origin-destination 对 (7 选 2)
- 75 名参与者, 每人恰好 1 个 session, 队列年龄 18-38, 14 国籍, 性别均衡, 31/75 对该区域不熟悉
- 单次 session 5-15 分钟, 平均 8 分钟
- DTW (Dynamic Time Warping) 距离证明路径非 templated: 同一对 waypoints 之间 103 次 within-pair 比较, median 122.8 m, range [19.3, 379.4] m, 这个分布是路径自由度的有效 sanity check
- IRB 批准, EgoBlur (ref [32]) 自动 blur 人脸与车牌, 不公开 raw video

DTW 公式直觉 (不是 paper 公式, 是通用定义):
$$
\text{DTW}(X, Y) = \min_{\pi} \sum_{(i,j) \in \pi} \|X_i - Y_j\|_2
$$
其中 $X, Y$ 是两条轨迹序列, $\pi$ 是 monotonic alignment path。median 122.8 m 说明同对 waypoint 间, 实际 walking path 差异远大于 GPS 误差量级, 因此路径选择是有效的"自由变量"。

### 2.3 多模态传感器同步 (Appendix A.2)

这是这篇 paper 工程上最关键、最容易出 bug 的地方。协议是:

- Unity app 跑在 MQPro 上, 50 Hz 记录 head pose / gaze / IMU (写入 CSV)
- Python 脚本跑在 headset 环境, 30 fps 录制 RGB video (写 MP4)
- 两者之间用 file-based signal 同步: `start_signal`, `video_ready`, `stop_signal`
- 同步精度上限 ≈ frame interval = 5.893 ms

同步流程:
1. Unity 启动 → 初始化 session dir + CSV files → 等待 controller A 按钮
2. participant 按 A → Unity 写 `start_signal` 文件
3. Python 检测到 `start_signal` → 启动 screen capture, 录视频, 写 `video_ready`
4. Unity 检测到 `video_ready` → 开始正式记录 sensor stream (此时 video 已就绪)
5. participant 按 B → Unity 写 `stop_signal` → 两个进程同时关闭

后处理阶段统一时间轴: 
- position 用 linear interpolation
- orientation quaternion 用 SLERP (Spherical Linear Interpolation)
- gaze 用 normalized linear interpolation
- 全部 resample 到 30 Hz 对齐 RGB video

SLERP 公式直觉 (for quaternions $q_0, q_1$ with angle $\theta = \arccos(q_0 \cdot q_1)$):
$$
\text{SLERP}(q_0, q_1; t) = \frac{\sin((1-t)\theta)}{\sin(\theta)} q_0 + \frac{\sin(t\theta)}{\sin(\theta)} q_1
$$
$t \in [0,1]$ 是插值参数, $\theta$ 是两个 quaternion 之间的角度。SLERP 保证插值路径在 $S^3$ 流形上是 geodesic, 不会因为 linear interp 引入非单位 norm 的 orientation。

### 2.4 Dataset 统计与 published 格式

最终每 session 一个 HDF5 文件, 含三个 group:
- `pose`: timestamp, position $(x, y, z)$, rotation $(q_w, q_x, q_y, q_z)$, linear velocity, angular velocity
- `gaze`: origin, direction (binocular 3D vectors, 投影后变成 $(u, v)$ 像素坐标)
- `video`: segment index, frame index

Dataset 总量:
- 75 sessions, 10.7 hours, 1.15M RGB frames
- 累计行走 46.73 km
- 平均 walking speed 1.25 m/s, 与 typical pedestrian 文献值 1.2-1.4 m/s 吻合 (Paper 3.3 节)
- 7% 帧 near-stationary (< 0.3 m/s), 抓到 stop-and-go dynamics

HDF5 选择直觉: 跟 Ego4D/Nymeria 一致选 HDF5, 因为 (a) hierarchical structure 适合 multimodal (b) chunked 压缩让 frame-level random access 便宜 (c) 一个文件便于 reproducibility。

### 2.5 与同类 dataset 的横向对比

Table 1 (paper 中的核心对比表) 关键差异:

| Dataset | Year | Hours | Frames | Subj. | Device | Gaze | 6DoF | Scene Ann. |
|---|---|---|---|---|---|---|---|---|
| KrishnaCam | 2016 | 70 | 7.6M | 1 | Google Glass | ✗ | ✗ (GPS only) | ✗ |
| EgoMotion | 2016 | 9.1 | 65.5K | N/P | GoPro Stereo | ✗ | ✗ (SfM offline) | ✗ |
| FPL | 2018 | 4.5 | 162K | N/P | Chest Cam | ✗ | ✗ | ✗ |
| Nymeria | 2024 | 300 | 32.4M | 264 | Aria + Xsens | ✓ | ✓ | ✓ |
| EgoNav | 2024 | 3.3 | 237.6K | N/P | RealSense | ✗ | ✓ | ✗ |
| LookOut | 2025 | 4.0 | 288K | N/P | Aria | ✓ | ✓ | ✗ |
| EgoCogNav | 2025 | 6.0 | 432K | 17 | Tobii + Aria | ✓ | ✓ | ✗ |
| **EgoTraj** | **2026** | **10.7** | **1.15M** | **75** | **Quest Pro** | **✓** | **✓** | **✓** |

直觉解读: Nymeria 是体量天花板 (300 小时), 但是 collaborative multi-actor 室内外混合, 不是 pedestrian-focused outdoor; EgoCogNav 是 gaze + 6DoF 最接近的, 但是 17 个 subject 而且室内外混合; LookOut 没记录 gaze。EgoTraj 在"outdoor + 75 subjects + gaze + 6DoF + scene annotation + AR headset"五维度同时满足这一点上是 unique selling proposition, 算是 AR device-centric trajectory dataset 的"first at scale"。

---

## 3. Scene annotation pipeline (paper Sec 3.4 + Appendix D)

### 3.1 VLM-based annotation

- 模型: Qwen2.5-VL-7B-Instruct (ref [39])
- 输入: 253 个 privacy-blurred video segments, 1 fps 采样, 共 38,606 帧
- Prompt: example-driven + Chain-of-Thought (CoT), 每帧推理 5 步: 环境上下文 → 动态 agent → 交通信号/障碍 → gaze fixation → 短期意图
- 输出: JSON sidecar 文件, indexed by session + frame ID

CoT 阶段 (Appendix D.1):
1. environmental context (crosswalk/sidewalk/intersection...)
2. dynamic agents (pedestrian/vehicle/cyclist...)
3. traffic signals + obstacles
4. projected gaze → attention target
5. inferred short-term intent

### 3.2 质量评估

Table 6 (per-field Cohen's κ + VLM accuracy):

| Field | Cohen's κ | VLM accuracy |
|---|---|---|
| Environmental context | 0.89 | 0.91 |
| Dynamic agents | 0.87 | 0.89 |
| Traffic signals | 0.96 | 0.92 |
| Gaze target | 0.95 | 0.98 |
| Short-term intent | 0.83 | 0.84 |
| Structural compliance | 96% (after ≤2 retries) | — |

直觉:
- κ > 0.8 表示 strong agreement, ≥0.93 几乎是 perfect
- gaze target 这一项 VLM 准确率最高 (0.98), 因为 gaze 已经被投影成红色 marker 在图像上, VLM 几乎只需做 object detection at marker location
- short-term intent 这项 κ 与 VLM 都最低 (0.83/0.84), 因为这一项 inherently subjective, 即使两个 human annotator 也难以完全一致
- 失败模式 audit (50 帧): 错过 pedestrian 4%, 误读 traffic light 3%, gaze 目标不匹配 2%, 累计 <10%, 大多集中在视觉模糊场景 (远距离行人、遮挡 traffic light、gaze near object boundary)

---

## 4. Gaze-to-pixel calibration (paper Sec 3.5)

这是从 MQPro raw 3D gaze 到 image coordinate 的关键映射。

模型: per-session quadratic calibration
$$
u = a_0 + a_1 \theta + a_2 \phi + a_3 \theta \phi + a_4 \theta^2 + a_5 \phi^2
$$
$$
v = b_0 + b_1 \theta + b_2 \phi + b_3 \theta \phi + b_4 \theta^2 + b_5 \phi^2
$$
其中 $\theta$ 是 gaze yaw 角, $\phi$ 是 gaze pitch 角, $\theta^2/\phi^2/\theta\phi$ 项吸收镜头径向畸变 + eye-to-camera 几何非共心带来的耦合。$(a_i, b_i)$ 系数对每个 session 单独拟合。

直觉: 
- linear 项 $a_1 \theta, b_2 \phi$ 对应理想针孔相机投影
- quadratic 项吸收 radial distortion (lens barrel/pincushion)
- cross term $\theta\phi$ 吸收 tangential distortion + eye-to-camera translation (双眼瞳距 ≈ 6.4 cm, 但 camera 在鼻梁上方, 几何上是非共心)

---

## 5. Benchmarking 评估协议

### 5.1 指标定义

- ADE (Average Displacement Error): 预测序列与 ground truth 序列在所有预测时间步上欧氏距离的平均
$$
\text{ADE} = \frac{1}{T_{\text{pred}}} \sum_{t=1}^{T_{\text{pred}}} \|\hat{p}_{t_0+t} - p_{t_0+t}\|_2
$$
$\hat{p}$ 是预测的 3D 位置, $p$ 是 GT 3D 位置, $t_0$ 是观测序列最后时刻, $T_{\text{pred}}$ 是预测窗口长度。

- FDE (Final Displacement Error): 终点处的欧氏距离
$$
\text{FDE} = \|\hat{p}_{t_0+T_{\text{pred}}} - p_{t_0+T_{\text{pred}}}\|_2
$$

- $L_{\text{rot}}$ (head rotation error, ref LookOut): 用预测 quaternion 与 GT quaternion 之间的 L1 距离
$$
\mathcal{L}_{\text{rot}} = \frac{1}{T_{\text{pred}}} \sum_{i=1}^{T_{\text{pred}}} \left\|\hat{R}_{t+i} R_{t+i}^{\top} - I\right\|_1
$$
其中 $\hat{R}_{t+i}, R_{t+i} \in SO(3)$ 是 predicted / GT 旋转矩阵, $\hat{R}R^\top$ 给出 relative rotation matrix, 减去 identity $I$ 的 L1 范数度量两个 rotation 的差距。这个度量比直接看 quaternion 角度差更稳健, 因为 relative rotation matrix 直接显示 misalignment 的所有分量。

### 5.2 观测/预测窗口设置

- $T_{\text{obs}} = 1.5$ s (即 45 帧 @ 30 Hz)
- $T_{\text{pred}} = 3.5$ s (即 105 帧 @ 30 Hz)

这个 1.5 s → 3.5 s 配置比 ETH/UCY 常见的 3.2 s → 4.8 s 短, 因为 egocentric trajectory 在长时窗口下不确定性增长更快 (BEV 可以靠 social force 约束, egocentric 一旦转弯就不准了)。1.5 s 大致对应 gaze 领先 motor action 的 lead time (Land 2006), 所以 1.5 s 观测窗口刚够让 gaze signal 进入观测, 3.5 s 预测窗口刚够覆盖下一秒转弯动作 + 2.5 秒稳态行走。

数据划分: 80/10/10 session-level split (train/val/test), 严格 subject-disjoint (每人恰好一个 session)。

---

## 6. Baseline 模型 (paper Sec 4.1)

EgoTraj 把 5 个 baseline 全部 in-house re-implement, 因为 EgoNav/LookOut/EgoCogNav 没公开代码, 而 Social-LSTM/TUTR 是 BEV dataset 训练, 与 egocentric 6DoF pose 不兼容。

### Baseline 1: Const_Vel

Const_Vel (LookOut paper 里提出) 是纯运动学外推: 用最后若干 step 的 linear velocity $\vec{v}$ 与 angular velocity $\vec{\omega}$ 做 body-frame 匀速外推:
$$
\hat{p}_{t+\Delta t} = p_t + R_t \vec{v} \Delta t
$$
$$
\hat{R}_{t+\Delta t} = \exp(\Delta t \cdot \vec{\omega}^\wedge) R_t
$$
$\vec{\omega}^\wedge$ 是 angular velocity 的 skew-symmetric matrix (so(3) Lie algebra 元素), $\exp$ 是 matrix exponential (从 Lie algebra 到 Lie group SO(3) 的指数映射)。这个 baseline 简单但合理, 因为行人短时间内确实接近匀速。

### Baseline 2: Lin_Ext

Lin_Ext 对每个轴 $(x, y, z)$ 与每个 quaternion 分量 $(q_w, q_x, q_y, q_z)$ 单独做 linear regression:
$$
\hat{p}^{(k)}_{t+i} = \beta_0^{(k)} + \beta_1^{(k)} (t+i), \quad k \in \{x,y,z\}
$$
$\beta_0, \beta_1$ 用最小二乘拟合过去 1.5 s 的数据。这个 baseline 失败在 head rotation ($L_{1\text{head}} = 1.39$) 严重, 因为 quaternion 不是线性空间 (它在 $S^3$ 上), 直接 axis-wise linear regression 会破坏 unit norm 约束, 给出非正常数 orientation。

### Baseline 3: M_Transformer

vanilla Transformer baseline, early fusion: 把所有 modality embedding 直接 concatenate, 过一个 temporal Transformer encoder + linear head 输出未来轨迹。这是 LookOut paper 里的标准 baseline。

### Baseline 4: CXA-Transformer (Qiu et al. RA-L 2022, ref [30])

Cascaded Cross-Attention Transformer: 多 stream 架构, ego-translation 与 ego-rotation 各自走独立 transformer encoder, 然后 cascaded cross-attention fusion, 每层 fusion 让 ego stream 作为 query, social/scene stream 作为 key/value。这是 paper 里 ablation study 的 base architecture。

Cascaded cross-attention 直觉 (ref Figure 13):
- 不是一次性 concat 所有 modality, 而是 stream-wise 分层融合
- 每一层 fusion 让"主信号" (ego-motion) 主动"提取"它需要的上下文, 而不是被动接收
- 与 simple concatenation 相比, 这种 design 在多 modality 异构数据上表现更稳 (Table 2 中 CXA 在 head rotation 上 $L_{1\text{head}} = 0.69$, 比 M_Transformer 的 0.74 还好)

### Baseline 5: EgoCast (WACV 2025)

EgoCast 原设计是 full-body pose prediction, paper 把它的 Transformer forecasting module 改造, 直接吃 past ego-translation + head rotation, 去掉 pose-estimation stage。在 Table 2 上 EgoCast 取得最低 ADE (0.16 m) 与 FDE (0.28 m), 但 head rotation 反而比 CXA 差 (0.78 vs 0.69), 说明 EgoCast 的 temporal head 更擅长位置预测, 而 CXA 的 cross-attention 结构对 orientation 的 SO(3) 流形结构更友好。

### 主结果表 (Table 2) 完整数据:

| Model | ADE↓ | FDE↓ | $L_{1\text{head}}$↓ |
|---|---|---|---|
| Const_Vel | 0.24 | 0.35 | 0.82 |
| Lin_Ext | 0.26 | 0.39 | 1.39 |
| M_Transformer | 0.20 | 0.32 | 0.74 |
| CXA-Transformer | 0.19 | 0.29 | 0.69 |
| EgoCast | 0.16 | 0.28 | 0.78 |

直觉解读:
- 1.5 s 观测 + 3.5 s 预测这个时间尺度上, Const_Vel 的 ADE 0.24 m 已经不错, 因为 3.5 s 内行人基本不会急转弯
- Lin_Ext 反而比 Const_Vel 还差 (ADE 0.26 vs 0.24), 因为线性回归对短期序列过拟合, 还破坏 quaternion 单位性, 给 head rotation 灾难性 1.39
- 神经网络整体把 ADE 从 ~0.24 降到 ~0.16, 收益 ~33%, 这是 multimodal sensing 的"上限收益"
- 0.16 m ADE 在 3.5 s 预测窗口下, 对 AR collision avoidance 是有意义的精度, 但对盲人导航还需要更准

---

## 7. Ablation Study (paper Sec 4.1, Table 3) — 这是全 paper 信息密度最高的部分

Base architecture: CXA-Transformer, 因为其 cascaded cross-attention 适合"逐步加 modality"。

Modality 定义:
- $\mathcal{V}$: ego-trajectory (position + rotation)
- $\mathcal{C}$: nearby people center points (torso keypoint centroid, via YOLOv8-Pose)
- $\mathcal{B}$: nearby people bounding boxes (via YOLOv8-Pose)
- $\mathcal{P}_s$: nearby people full body pose (via YOLOv8-Pose)
- $\mathcal{S}$: semantic segmentation (via OneFormer, ref [13])
- $\mathcal{D}$: relative depth (via Depth Anything V2, ref [43])
- $\mathcal{G}$: gaze projected to $(u, v)$

Table 3 完整数据:

| Modality | ADE↓ | FDE↓ | $L_{1\text{head}}$↓ |
|---|---|---|---|
| $\mathcal{V}$ | 0.19 | 0.29 | 0.69 |
| $\mathcal{V} + \mathcal{C}$ | 0.18 | 0.29 | 0.79 |
| $\mathcal{V} + \mathcal{B}$ | 0.18 | 0.30 | 0.81 |
| $\mathcal{V} + \mathcal{P}$ | 0.17 | 0.27 | 0.77 |
| $\mathcal{V} + \mathcal{S}$ | 0.16 | 0.26 | 0.74 |
| $\mathcal{V} + \mathcal{D}$ | 0.18 | 0.29 | 0.78 |
| $\mathcal{V} + \mathcal{G}$ | 0.15 | 0.26 | 0.69 |
| $\mathcal{V} + \mathcal{C} + \mathcal{G}$ | 0.14 | 0.25 | 0.67 |
| $\mathcal{V} + \mathcal{B} + \mathcal{G}$ | 0.16 | 0.26 | 0.70 |
| $\mathcal{V} + \mathcal{P} + \mathcal{G}$ | 0.12 | 0.24 | 0.63 |
| $\mathcal{V} + \mathcal{S} + \mathcal{G}$ | 0.12 | 0.25 | 0.65 |
| $\mathcal{V} + \mathcal{D} + \mathcal{G}$ | 0.15 | 0.27 | 0.71 |
| $\mathcal{V} + \mathcal{P} + \mathcal{S} + \mathcal{G}$ | **0.12** | **0.23** | **0.58** |

### 关键观察

**观察 1: 单 modality 增量效果排序 (vs $\mathcal{V}$ baseline)**

- 加 $\mathcal{S}$: ADE 0.19 → 0.16 (-0.03, 减 15.8%)
- 加 $\mathcal{G}$: ADE 0.19 → 0.15 (-0.04, 减 21.1%)
- 加 $\mathcal{P}$: ADE 0.19 → 0.17 (-0.02, 减 10.5%)
- 加 $\mathcal{D}$: ADE 0.19 → 0.18 (-0.01, 减 5.3%)
- 加 $\mathcal{C}$: ADE 0.19 → 0.18 (-0.01, 减 5.3%)
- 加 $\mathcal{B}$: ADE 0.19 → 0.18 (-0.01, 减 5.3%) 但 $L_{1\text{head}}$ 反而恶化 0.69 → 0.81

直觉解读:
- **gaze 单加效果最强**, 验证 Land 2006 的 "gaze leads motor by 1-2 s" 假设
- **scene segmentation 第二强**, 因为它直接告诉模型"哪里可走哪里不可走" — 这是 walkable area prior, 把无穷解空间约束到 sidewalk 路径
- **body pose 比 center/bbox 好**, 因为 fine-grained limb motion 包含 turn-anticipation 信号 (eg. 行人转身前肩膀先转), 而 center/bbox 只给 spatial extent
- **relative depth 增量最小**, 因为在 outdoor 平面环境, depth 主要是远处建筑物 → 不携带 walkable 信息
- **bounding box 甚至伤害 head rotation**, 因为 bbox 不直接给"对面行人朝哪个方向走"信息, 反而引入额外噪声到 attention, 干扰 ego-rotation stream

**观察 2: gaze 与 social modality 组合**

- $\mathcal{V} + \mathcal{P} + \mathcal{G}$: ADE 0.12 (减 36.8% vs $\mathcal{V}$) ← strong synergy
- $\mathcal{V} + \mathcal{B} + \mathcal{G}$: ADE 0.16 (减 15.8% vs $\mathcal{V}$) ← weak synergy
- $\mathcal{V} + \mathcal{C} + \mathcal{G}$: ADE 0.14 (减 26.3% vs $\mathcal{V}$) ← medium synergy

直觉: body pose 与 gaze 的 synergy 来自"行人姿态 + ego 注意力"双重确认 — 如果 ego 在 gaze 一个正在转弯的行人, 模型可以从该行人的 body pose 推断 egocentric 的避让方向。

**观察 3: 最优组合 $\mathcal{V} + \mathcal{P} + \mathcal{S} + \mathcal{G}$**

ADE 0.12 m, FDE 0.23 m, $L_{1\text{head}}$ 0.58 — 全部最优。这说明在 1.5 s + 3.5 s 这个时间尺度, 三种 modality (ego-motion + social + scene) 在数学上接近互补, gaze 在已有 social + scene 的基础上还能把 ADE 从 0.13 (推测 $\mathcal{V} + \mathcal{P} + \mathcal{S}$ 的) 降到 0.12 (虽然 paper 没列 $\mathcal{V} + \mathcal{P} + \mathcal{S}$ 单独的行, 但从 $\mathcal{V} + \mathcal{P}$ 0.17 + $\mathcal{V} + \mathcal{S}$ 0.16 推测 $\mathcal{V} + \mathcal{P} + \mathcal{S}$ 大约在 0.13-0.14 量级)。

---

## 8. Generalization across splits (paper Sec 4.1, Table 4)

为检验模型是否过拟合到"路径模板", paper 设计了三个 split:

1. Random Participant (n=8 sessions, 标准 random split)
2. Waypoint Held-Out (n=10 sessions, 3/21 origin-destination 对完全 hold out)
3. Unfamiliar (n=8 sessions, 31 名对该区域不熟悉的参与者中 8 个)

Bootstrap 95% CI 来自 1000 次重采样。

| Modality | Random ADE/FDE | Waypoint Held-Out ADE/FDE | Unfamiliar ADE/FDE |
|---|---|---|---|
| $\mathcal{V}$ | 0.19±.014 / 0.29±.021 | 0.21±.018 / 0.32±.024 | 0.23±.019 / 0.34±.027 |
| $\mathcal{V} + \mathcal{P}$ | 0.17±.011 / 0.27±.019 | 0.19±.015 / 0.29±.022 | 0.20±.013 / 0.31±.020 |
| $\mathcal{V} + \mathcal{S}$ | 0.16±.013 / 0.25±.014 | 0.18±.012 / 0.28±.018 | 0.18±.016 / 0.29±.023 |
| $\mathcal{V} + \mathcal{G}$ | 0.15±.009 / 0.26±.017 | 0.16±.014 / 0.26±.013 | 0.16±.010 / 0.29±.018 |
| $\mathcal{V} + \mathcal{P} + \mathcal{S} + \mathcal{G}$ | **0.12±.008** / **0.23±.012** | **0.14±.010** / **0.25±.011** | **0.14±.012** / **0.26±.014** |

直觉解读:
- 在 $\mathcal{V}$ only 上, Unfamiliar ADE (0.23) 比 Random (0.19) 高 21%, 说明纯运动学模型对环境熟悉度敏感 (因为行人熟悉环境时 navigation 行为更果断, 预测更难)
- 加入 $\mathcal{V} + \mathcal{P} + \mathcal{S} + \mathcal{G}$ 之后, 三个 split 的 ADE 0.12 / 0.14 / 0.14, 差距收缩到 16%, 说明多模态确实 generalize 到不熟悉环境的参与者与没见过的 waypoint 对
- 95% CI 不重叠 ($\mathcal{V}$ only 与 full multimodal 在 Random split 上 CI 区间 [.176, .204] vs [.112, .128] 不重叠), 这是统计上显著改进
- Waypoint held-out 与 Unfamiliar 上的 ADE 都是 0.14, 这个数字接近, 说明"没见过的 waypoint 对" 与 "不熟悉环境的参与者"对模型挑战程度类似

---

## 9. Active-transition windows (paper Sec 4.1)

这是 paper 最有 insight 的子实验之一。定义: turning 行为开始于 $T_{\text{obs}}$ 最后 0.5 s 内, 即大部分方向变化落在预测窗口而非观测窗口的样本。这是 gaze 应该 lead motor 1-2 s 假设的严格测试。

结果: full multimodal CXA-Transformer 在 active-transition subset 上 ADE 0.23±0.022, FDE 0.38±0.028。比 $\mathcal{V}$ only 在 random split 上的 ADE 0.19 高, 但比 motion-only baselines 在 active-transition 上 drift-outward 灾难情形下大幅好 (从图 8 可见 Const_Vel/Lin_Ext 直接沿 pre-turn heading 飘出, 偏离 ground truth 几米)。

直觉: 这就是 paper Sec 4.1 Failure Cases 提到的 ~90° 急转弯难问题的核心。在 $T_{\text{obs}}$ = 1.5 s 末尾开始 turn, 模型观测到的运动学序列还很平稳, 没有方向变化先兆; 唯一的 anticipatory signal 来自 gaze — 行人已经在过去 1-2 s 内 fixate 了目标方向。Full multimodal 用 gaze 把这个信号纳入, 但仍然 ADE 0.23 比 random 场景下 0.12 高, 说明 active-transition 是硬案例。Paper 末尾留这个 regime 给 future work 的 uncertainty-aware 与 intent-aware prediction。

---

## 10. 架构图解析 (Figure 13, Appendix E)

paper 在 Appendix E 给出 multimodal fusion 的详细图示:

### 10.1 Modality encoder streams

四个独立 transformer encoder streams, 每个吃一种 modality:
1. **Ego-motion stream**: 输入 $\mathcal{V} = \{(p_t, o_t)\}_{t=t_0-T_{\text{obs}}+1}^{t_0}$, 即 1.5 s × 30 Hz = 45 timestep × 7 DoF (3 position + 4 quaternion) = 45×7 输入, 编码后得到 $E_V \in \mathbb{R}^{45 \times d}$
2. **Social stream**: 输入 $J_n \in \mathbb{R}^{T_{\text{obs}} \times d}$ per pedestrian $n$, $d$ 是 pose 关键点维度 (YOLOv8-Pose 17 个 COCO keypoint × 2 坐标 = 34), 编码后 $E_P \in \mathbb{R}^{T_{\text{obs}} \times d'}$
3. **Scene stream**: 输入 $S \in \mathbb{R}^{T_{\text{obs}} \times k}$, $k$ 是 segmentation feature 维度, 编码后 $E_S \in \mathbb{R}^{T_{\text{obs}} \times d'}$
4. **Gaze stream**: 输入 $G = \{(u_t, v_t)\}_{t=t_0-T_{\text{obs}}+1}^{t_0}$, 每帧 2D 坐标, 编码后 $E_G \in \mathbb{R}^{T_{\text{obs}} \times d'}$

### 10.2 Cascaded cross-attention fusion

不是一次性 concat, 而是 sequential:

Stage 1: $E_V$ 作为 query, $E_P$ 作为 key/value
$$
F_1 = \text{CrossAttn}(Q=E_V, K=E_P, V=E_P)
$$
得到 ego-motion 已经"看过"附近行人姿态的融合特征 $F_1$

Stage 2: $F_1$ 作为 query, $E_S$ 作为 key/value
$$
F_2 = \text{CrossAttn}(Q=F_1, K=E_S, V=E_S)
$$
得到再融入 scene semantics 的融合特征 $F_2$

Stage 3: $F_2$ 作为 query, $E_G$ 作为 key/value
$$
F_3 = \text{CrossAttn}(Q=F_2, K=E_G, V=E_G)
$$
最终融合 gaze 信号得到 $F_3$, 过 decoder + linear head 输出未来轨迹 $\{(\hat{p}_t, \hat{o}_t)\}_{t=t_0+1}^{t_0+T_{\text{pred}}}$

直觉解读:
- ego-motion 始终作为 query, 让模型"主动选择"它需要的上下文
- cascade 顺序 V → P → S → G 不是任意, 而是按照 modality 的 abstraction level 递增 (motion 是最 raw, pose 是 peer-level, scene 是 environment-level, gaze 是 cognitive-level)
- 这跟 general multimodal transformer 的 "Q from main modality, K/V from auxiliary" 模式一致, 类似 Perceiver / Flamingo 的设计哲学

Cross-attention 公式直觉:
$$
\text{CrossAttn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$
$Q \in \mathbb{R}^{N \times d_k}$ (ego stream tokens), $K \in \mathbb{R}^{M \times d_k}$ (auxiliary modality tokens), $V \in \mathbb{R}^{M \times d_v}$, $d_k$ 是 key 维度 (用于 scaling 防止 softmax 饱和)。Output $N \times d_v$, 即每个 ego token 对应一个加权平均的 auxiliary 信号。

---

## 11. EgoViz Dashboard (paper Sec 3.6, Figure 5)

paper 配套发布了一个 interactive inspection interface 叫 EgoViz, 同步显示四个 view:
1. **2D trajectory plot**: 30 Hz pose samples + gaze-direction vector (white arrow), 可 zoom
2. **BEV of full session path**: 当前位置高亮
3. **egocentric RGB frame**: 投影 gaze 红点
4. **VLM-generated scene annotation**: 文本描述

底部 status bar 显示 timestamp, 3D position, speed, frame index。这个 dashboard 是 paper 的 reproducibility 灵魂, 让 reviewer / 后续 user 可以 frame-level 验证 multimodal alignment。github.com/yehiahmad/EgoTraj 上公开。

---

## 12. Paper 跟相邻工作的"intellectual lineage"

### 12.1 与 LookOut (ICCV 2025, ref [25])

- LookOut 也是 AR headset (Aria) + outdoor navigation + 6DoF head pose trajectory prediction
- 但 LookOut **没有 gaze** 数据, 仅 4 小时 + N/P 个 subject
- LookOut 给出 Const_Vel / Lin_Ext / M_Transformer 三个 baseline, EgoTraj 直接复用
- LookOut 把 trajectory prediction problem 变成 head pose prediction problem (因为 headset 没有 body, 只有 head, 6DoF head pose 就是 navigation signal)
- EgoTraj 把这个 framing 接过来, 加 gaze, 在 head rotation L1 metric 上扩展

### 12.2 与 EgoCogNav (arXiv 2025, ref [31])

- EgoCogNav 6 小时, 17 subject, Tobii + Aria
- 关注 cognition-aware modeling + perceived navigational uncertainty
- EgoCogNav 室内外混合, EgoTraj 纯 outdoor
- EgoCogNav 也是 gaze + 6DoF + indoor/outdoor, 但是没公开代码

### 12.3 与 Nymeria (ECCV 2024, ref [20])

- Nymeria 300 小时, 264 subject, Aria + Xsens (额外的 IMU mocap suit)
- 规模最大, 但是 collaborative multi-actor 室内场景, 不是 outdoor pedestrian navigation focus
- 数据格式上 Nymeria 也是 HDF5 + pose + gaze, paper 复用其格式约定

### 12.4 与作者自己的 ARCAS (ref [44])

- ARCAS 用 SLAM-based tracking + AR overlay 做 VRU (vulnerable road user) collision avoidance
- 是 EgoTraj 的"下游应用"概念验证
- EgoTraj 数据可视为给 ARCAS 类系统提供训练 + benchmark 的数据基础

---

## 13. 失败模式与未来方向 (paper Sec 4.1, Sec 5)

### 13.1 Sharp ~90° turn failure (Window-Slice 2512, Figure 6 right)

直觉: 在 1.5 s 观测末尾 + 0.5 s 内 90° 急转弯, 所有 baseline 都不能正确预测曲率。原因:
- 观测窗口 1.5 s 内 linear velocity 与 angular velocity 都在 normal walking range
- 没有 strong anticipatory cue 表明马上要急转
- gaze 虽然已经在 fixate 新方向, 但 full multimodal 模型对 gaze → motion 的 mapping 还没学到对 sharp transition 的鲁棒性
- paper Sec 4.1 末尾说这种 deterministic trajectory forecasting 在 multimodal intent 下 inherently 难, 需要 intent-aware + uncertainty-aware 预测

### 13.2 Future work hint

paper Sec 5 + Sec 4.1 末段暗示方向:
- **Multimodal intent prediction**: 不只输出未来轨迹, 还输出 multimodal distribution (eg. 20 个 sample with probability)
- **Uncertainty quantification**: 用 probabilistic forecasting (eg. cVAE / diffusion-based trajectory sampling) 给出置信区间
- **Long-horizon egocentric forecasting**: 现在 3.5 s, 推到 10 s + 用 scene graph 而非 segmentation
- **End-to-end embodied navigation**: 把 prediction 接到 robot/humanoid control loop (类似 SCAND dataset ref [15] 或 ARMOR ref [16])
- **Gaze + scene + motion joint generative model**: 用 diffusion / flow matching 在 gaze conditional distribution 上采样未来轨迹

---

## 14. 个人 takeaway (build intuition)

如果让我从这篇 paper 提炼几个最核心的 insight:

1. **gaze leads motion by 1-2 s** 是 egocentric trajectory prediction 的物理根基, 这个 1-2 s 时间常数决定了 $T_{\text{obs}}$ 至少要 ≥1 s 才能让 gaze 进入观测窗口

2. **scene segmentation > depth > pose > bbox** 在这个 task 上的边际收益排序, 反映了 outdoor pedestrian navigation 的本质是"拓扑约束 > 几何约束 > agent 互动 > 局部外观"

3. **cascaded cross-attention 比 simple concat 强**, 因为 multimodal transformer 的难点在"异构信号融合", 而 cascade 让主 modality 主动 select 上下文, 不是被动 concat

4. **急转弯是 deterministic prediction 的硬上限**, 必须走向 probabilistic multimodal forecasting

5. **AR headset (MQPro) 比 Aria + Xsens 组合更适合大规模采集**, 因为商用设备便宜、deployable, 而且 passthrough 让参与者能自然行走 — 这个 trade-off (sensing richness vs deployment cost) 是 EgoTraj 的工程核心

6. **数据集设计的"自由度"很关键**: waypoints 是 landmark 不是 GPS point, DTW 距离证明 path 自由度真实存在, 这是 dataset 不是 scripted trajectory replay 的关键证据

7. **scene annotation 用 Qwen2.5-VL-7B + CoT + VLM 给出 κ > 0.83**, 说明 VLM 在 navigation context 上已达到 human-annotator 可靠度, 未来 dataset 的 annotation pipeline 会越来越自动

8. **80/10/10 session-level + subject-disjoint split + 95% bootstrap CI** 是这类 dataset paper 的统计严谨度新标准, 之后做 dataset benchmark 都应跟进

---

## 15. 可能的延伸联想 (宁可 hallucinate 也不漏)

- **与 Ego-Body / Ego-Exo4D 的联系**: Ego-Exo4D (CVPR 2024, ref [9]) 是 ego + exo dual perspective 的 skilled activity dataset, 与 EgoTraj 互补。EgoTraj 关注 navigation, Ego-Exo4D 关注 manipulation, 两者未来可能合并成通用 egocentric behavior dataset
- **与 Ego4D 的 future prediction benchmark**: Ego4D (CVPR 2022, ref [8]) 也有 future-hand-object prediction 子任务, 但 metric 是 2D bounding box; EgoTraj 把它推到 6DoF trajectory + gaze conditional, 更适合 AR navigation
- **与 Humanoid robot navigation**: ARMOR (ref [16]) 用 egocentric perception 做 humanoid collision avoidance, EgoTraj 直接给这类方法提供训练数据
- **与 Social navigation in robotics**: SCAND dataset (ref [15]) 是 teleoperated robot 社交导航, 与 EgoTraj 形成对比 — 一个是 robot perspective, 一个是 human perspective; 未来可做 cross-embodiment transfer learning
- **与 Implicit Maximum Likelihood Estimation**: Abduallah Mohamed (本文 co-author) 的 Social-Implicit (ECCV 2022, ref [23]) 提出 IMLE 做多模态 trajectory sampling, 完全可以套到 EgoTraj 上做 multimodal intent distribution prediction
- **与 Diffusion-based trajectory prediction**: MID (Mao et al. CVPR 2023) / LED (Shi 2024) 用 diffusion model 做 multi-future trajectory sampling, EgoTraj 的 90° 急转弯 case 是 diffusion-based 方法的 natural target
- **与 PDE-based traffic flow modeling**: Christian Claudel (co-author) 的本职研究方向是 Lighthill-Whitham-Richards (LWR) traffic PDE, paper 没明显结合, 但理论上 pedestrian flow 也可以做 continuum PDE 建模, EgoTraj 的 head pose 数据可以做"个体 fluid particle trajectory"采样源
- **与 CityGML / digital twin**: paper 提到 XR-DT (作者前作) 用 AR 增强 digital twin; EgoTraj 可以视为 digital twin 里 pedestrian agent 的"behavioral ground truth"采集 pipeline
- **与 Tesla FSD 的 pedestrian prediction**: 自动驾驶里 pedestrian intent prediction 跟 egocentric pedestrian trajectory prediction 是 dual problem — 一个是 vehicle-ego predicting other-pedestrian, 一个是 pedestrian-ego predicting self-trajectory; 两者数据可能 cross-fertilize
- **与 Apple Vision Pro**: MQPro 是这篇 paper 的 hardware, AVP 没被使用, 但 AVP 同样有 eye tracking + 6DoF tracking + passthrough, 未来 AVP 上复现 EgoTraj 协议是明显 follow-up
- **与 NVidia Isaac Sim + synthetic data**: EgoTraj 是真实数据, 但仅 75 subject; 可以用 Isaac Sim + procedurally generated urban scenes 做数据增强, pretrain on synthetic + finetune on EgoTraj

---

## 16. 推荐进一步阅读清单

为了 build 更完整 intuition, 这些 references 值得一并看:

1. **Land 2006** (ref [17]) - "Eye movements and the control of actions in everyday life" - gaze leads motor 1-2 s 的实验心理学基础: https://www.sciencedirect.com/science/article/pii/S1350946205000562

2. **LookOut (ICCV 2025)** (ref [25]) - egocentric humanoid navigation dataset 的奠基 paper: https://arxiv.org/abs/2506.21450

3. **EgoCogNav** (ref [31]) - gaze-aware cognition-aware navigation 的另一份 work: https://arxiv.org/abs/2511.17581

4. **EgoNav** (ref [41]) - scene-aware egocentric trajectory prediction: https://arxiv.org/abs/2403.19026

5. **EgoCast (WACV 2025)** (ref [5]) - egocentric human pose forecasting 的 transformer baseline: https://openaccess.thecvf.com/content/WACV2025/papers/Escobar_EgoCast_Forecasting_Egocentric_Human_Pose_in_the_Wild_WACV_2025_paper.pdf

6. **Project Aria documentation** - 与 MQPro 直接对比: https://www.projectaria.com/

7. **Social-Implicit (ECCV 2022)** (ref [23]) - 本文 co-author Abduallah Mohamed 的 prior work: https://arxiv.org/abs/2209.13024

8. **Ego4D (CVPR 2022)** (ref [8]) - 大规模 egocentric video dataset 的奠基: https://arxiv.org/abs/2110.06091

9. **Ego-Exo4D (CVPR 2024)** (ref [9]) - dual perspective skilled activity: https://arxiv.org/abs/2404.04458

10. **Nymeria (ECCV 2024)** (ref [20]) - 大体量 multimodal egocentric daily motion: https://arxiv.org/abs/2406.05823

11. **Qwen2.5-VL technical report** (ref [39]) - VLM annotation 的 backbone: https://arxiv.org/abs/2409.12191

12. **Depth Anything V2** (ref [43]) - relative depth 估计: https://arxiv.org/abs/2406.09463

13. **OneFormer (CVPR 2023)** (ref [13]) - semantic segmentation backbone: https://arxiv.org/abs/2211.06586

14. **YOLOv8** (ref [10]) - pose detection: https://github.com/ultralytics/ultralytics

15. **EgoBlur** (ref [32]) - privacy de-identification pipeline: https://arxiv.org/abs/2309.04076

---

总结一句: EgoTraj 把 AR headset (MQPro) 的 multimodal sensing 能力 (RGB + 6DoF + gaze) 与 outdoor pedestrian navigation 的现实 challenge 缝合起来, 通过 75-participant 严格 subject-disjoint benchmark + cascaded cross-attention fusion 验证了 gaze + scene + social pose 的 synergy, 给未来 intent-aware probabilistic egocentric trajectory prediction 提供了第一个高质量 outdoor 数据基础与可复现 baseline。
