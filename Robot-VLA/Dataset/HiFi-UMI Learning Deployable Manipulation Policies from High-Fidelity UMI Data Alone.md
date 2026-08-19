---
source_pdf: HiFi-UMI Learning Deployable Manipulation Policies from High-Fidelity
  UMI Data Alone.pdf
paper_sha256: 5ec3eb818a43b88382985dcf1aaeafcdcf3ad1648dfdbec3c09e0f6631c944f1
processed_at: '2026-08-19T11:06:39-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 HiFi-UMI

## 一句话版

之前大家觉得，用手持夹爪采数据便宜是便宜，但质量不够 deploy，所以真实机器人那点 teleop 数据少不了。这篇 paper 说：**只要硬件做得够精，teleop 数据真的可以一个都不要**。

## 背景为啥这事儿难

robot learning 现在卡在数据上，这个所有人都知道。teleop 数据质量好，但是贵得离谱——AgiBot World 为了搞 2976 小时，动用 100 台双臂人形机器人，专门盖了个 4000 平米的场子 (https://arxiv.org/abs/2503.06669)。这个 scale 量级你要复现，钱和场地都跟不上。

UMI (https://arxiv.org/abs/2402.10329) 出来之后，大家觉得有救了——一个手持夹爪，谁都能采，成本降一个数量级。但用了两年下来，社区形成了一个默契：**UMI 数据只能做 pre-training**，真正要 deploy 到具体机器人上，还是得在 evaluation scene 里采一小撮 teleop 数据 "anchor" 一下，把 policy 拉回 embodied reality。ActiveUMI (https://arxiv.org/abs/2510.01607)、XRZero-G0、RDT2 deploy 版本全都保留这个 anchor。

这个 anchor 的存在，本质上是大家在说："UMI 数据的 fidelity 我信不过，差那临门一脚"。

## 他们干了啥

HiFi-UMI 的 claim 是：**之前信不过，是因为 fidelity 真的没做到位，做够了就不需要 anchor 了**。

他们把 handheld 采集设备的四个维度全部拉满：

**第一，pose 精度。** 之前 UMI 把 SLAM 装在手腕上，手腕在 manipulation 时被手、物体、self-occlusion 疯狂遮挡，长 horizon 下 drift 严重。HiFi-UMI 把 stereo camera 装头上——头部动作远比手部稳定——然后手上的 marker cube 由头部相机直接观测。这样两只手的 relative pose 是 native 测量的，不用后处理 reconstruct。精度做到 3mm，和 VR controller tracking、base station tracking 一个量级，但不需要任何外部设备。

**第二，同步。** 之前用 software timestamp 对齐，ms 级误差。快速手部动作时，image 是 t 时刻的、pose 是 t+5ms 的，action label 直接被污染。他们用一个 GPIO hardware trigger 把所有 sensor 拉到同一根线上，跨 sensor 偏差 < 40 微秒。

**第三，视野。** 之前一个 155° wrist fisheye，盲区大、depth cue 弱。他们每只手装两个**非平行** fisheye，加上头部 stereo，共 6 个 camera，水平垂直覆盖都 ~200°。两个 non-parallel camera 有 baseline，stereo cue 在 gripper 周围很 dense。

**第四，gripper form factor。** 之前 trigger 款触觉对应感弱，finger-sleeve 款太窄不适合重物。他们做了一个非对称两指 glove，指尖窄适合精细操作，根部宽适合承重。

这四样合起来，加上一套处理 flywheel（offline SLAM、simulation replay 验证、AI 标注、人工 verify），20,000 小时数据里 96% 能通过 robot-executability 验证。这个 yield rate 本身就说明 fidelity 是 deployment-grade 的——你 teleport 都不见得 96% trajectory 能 replay 成功。

## 实验结果有多硬

他们在同一台 bimanual 机器人上，用三种完全不同架构的 policy backbone 测：

- StarVLA-QwenPI（modular VLA，Qwen3-VL + flow-matching DiT）
- OpenPI-π0.5（公开 checkpoint VLA，PaliGemma + Gemma action expert）
- LingBot-VA（WAM，先预测 future video 再反解 action）

每个 backbone 都跑两组：一组只用 HiFi-UMI 数据 post-train，一组只用 teleop 数据 post-train。其他全部 freeze——架构、init、recipe、deploy stack 都一样。

结果：

| Backbone | UMI | Teleop | 差 |
|---|---|---|---|
| StarVLA-QwenPI | 51.3% | 53.8% | -2.5 pp |
| OpenPI-π0.5 | 77.5% | 74.4% | +3.1 pp |
| LingBot-VA | 56.9% | 57.5% | -0.6 pp |

差值正负都有，全部在 sampling noise 内（40 rollouts/task，一个 rollout 就 2.5 pp）。三个完全不同的架构都给出 parity，说明这是 data 性质，不是某个 model 的 trick。

**最 striking 的不对称性**：teleop 数据是在 evaluation scene 里采的，UMI 数据完全不在 evaluation scene——背景、灯光、桌面外观全不同。也就是说 UMI 是在 scene-level distribution shift 下打平 in-distribution teleop。这还不算完——UMI 一侧用了 3,200 条轨迹/task，teleop 用 300 条。所以这比较的是 practical pipeline throughput，不是 per-trajectory efficiency。但这个 asymmetry 对 teleop 是 favorable 的，UMI 在更难的条件下打平，才更说明问题。

最强 policy 在 Remote Insertion（精度插入任务）上做到 **85%**，这个数字是 zero teleop + scene shift 条件下达成的。

## Pre-training 的 bonus

除了 post-training parity，他们还在 StarVLA-QwenPI 上做了 4000 小时 UMI 数据的 pre-training：

- held-out action error 在 one pass 内降 61%，power-law fit $\alpha=0.268$, $R^2=0.993$。这是 exposure scaling，power law hold 得很好
- 10 个**完全没在 pre-training 里出现**的 task 上 action error 降 41%
- post-training 时只需要 800 条 task-specific 数据就能超过从 scratch 训 3,200 条的 baseline——pre-training 给的是真实可用的 visual-motor prior，不是 task-specific memorization
- 4 个 benchmark task 上 aggregate real-robot success 比 scratch init 高 **+18.1 pp**

这里有个**很 informative 的细节**：OOD transfer 的速率强依赖 pre-training 里覆盖了哪种 **interaction dynamics**。Rigid pick-and-place 类任务 OOD 改善最快（pre-training 里 1/3 frames 是这类），cloth folding 改善最慢（<1% frames）。这说明 transfer 看的是 dynamics coverage，不是 object identity。这对未来数据采集方向是个 actionable insight——要采的是 interaction type 的多样性，不是 task 数量的多样性。

## 直觉上的 takeaway

**robot learning 之前那个共识——"robot-free 数据 seed 但不能 finish"——可能是个 fidelity 问题，不是 robot-free setting 本身的限制。**

如果你类比 LLM 的发展：早期大家觉得 human annotation 不可替代，后来 web-scale self-supervised pre-training 做大了，RLHF 那点 human data 只剩 align 作用，再后来连 align 都开始自动化。robot 这边，teleop 数据扮演的就是 "human annotation" 的角色——embodied、精准、但贵。如果 HiFi-UMI 这条路线 scale 得起来，teleop 可能会退到只承担 evaluation 的角色，训练全走 robot-free data。

当然，这个 paper 也有 caveats：

- **没做 fidelity ablation**——证明 "整体高 fidelity 够"，但没拆开 3mm vs 6mm、GPIO vs software sync 各自贡献多少。这是把 fidelity 当 design principle 整体打包验证，paper 自己 acknowledge 这个 limitation。
- **Task scope 有限**——4 个 tabletop bimanual task，long-horizon、mobile、dexterous 都没测
- **Pre-training evidence 只在 StarVLA-QwenPI 上**，没在 OpenPI 和 LingBot 上复现
- **Sample 不 matched**——UMI 3200 vs teleop 300，是 practical throughput 比较不是 efficiency 比较

但这些 caveats 不改变核心 claim：**第一次有 controlled 证据表明，robot-free handheld 数据的 fidelity 可以高到支撑 zero-robot post-training，且跨多个 policy family 复现**。

如果这个结果 hold 得住并在更广 task scope 上 generalize，robot learning 数据生产的成本曲线会被重画——teleop 的 $/hour 和 UMI 的 $/hour 差一个数量级，AgiBot World 那种 100 robots + 4000m² facility 的 setup 可能不是唯一路径。这对整个 field 的 democratization 意义很大——任何一个 lab 都能拿一套 HiFi-UMI 设备，在自家厨房采几千小时数据，训出一个 deploy 得了的 bimanual policy。

这大概是这篇 paper 在长叙事里的真正位置：**它在 robot learning 的数据金字塔里，把 UMI 这一层从 "中间层 + pre-training only" 推到了 "deployment-grade"**。

参考链接：

- Paper & dataset: https://cloud.simpleai.tech/simple-world-lab/hifi-umi/ , https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K
- UMI original: https://arxiv.org/abs/2402.10329
- ActiveUMI: https://arxiv.org/abs/2510.01607
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- AgiBot World: https://arxiv.org/abs/2503.06669
- Scaling laws for imitation learning: https://arxiv.org/abs/2410.18647

---

# HiFi-UMI: 从 High-Fidelity Robot-Free 数据到 Deployable Manipulation Policy

## 1. 论点与动机 — 一个对 UMI 共识的正面挑战

UMI 这个方向从 Chi et al. 在 RSS 2024 提出 (https://arxiv.org/abs/2402.10329) 以来已经形成了一个相对稳定的"分层共识"：handheld gripper 数据**可以 scale**，但是因为 SLAM drift、单目 depth cue 弱、software 同步、reconstructed relative pose 这些缺陷，它只能承担 **pre-training** 的角色；post-training 那一步，社区几乎默认必须有一个 **real-robot "anchor"**——少量 teleop 数据把 policy "ground" 到具体机器人上。ActiveUMI (https://arxiv.org/abs/2510.01607)、XRZero-G0、甚至 RDT2 在 deploy-grade 时都保留了这个 anchor。

HiFi-UMI 提的核心 hypothesis 非常 sharp，叫做 **zero-robot post-training**：

> 如果 robot-free 数据的 fidelity 高到一定阈值，就可以**完全替代** teleop 数据做 post-training，policy 直接 deploy 到真实机器人。

这是一个**强假设**，因为 paper 把 fiduciary burden 完全放在"data 本身的质量"上，而不是 mixing ratio 上。prior work 的逻辑是"减少 real-robot fraction"，HiFi-UMI 的逻辑是"提高 robot-free 数据的 fidelity，让 anchor 本身不必要"。

Paper 自己也坦白：他们**没有做 fidelity 各维度的 ablation**（这是 limitations 里明确写的），而是把 fidelity 当 design principle 整体打包验证。这是一个 strategic 的选择——证明 "sufficient" 不证明 "necessary per-axis"。这是 build intuition 时要记住的一个 caveat。

---

## 2. 硬件系统：四个 Fidelity Axes 的 Co-Design

这是 paper 的硬件核心。把 Table 1 (与 UMI/FastUMI/ActiveUMI/TacUMI/RDT2/FastUMI Pro/XRZero-G0 的对比) 翻译成 design rationale，本质上是攻击 prior handheld capture 的四个 fidelity 缺陷。

### 2.1 Pose Acquisition: Head-Mounted Offline Stereo-Inertial SLAM

这是整套设计最 subtle 的一步。先看 prior art 的 spectrum：

| 方法 | 精度 | 限制 |
|---|---|---|
| Wrist VIO (UMI, DAS fingers) | ~6mm | 腕部视角易被手/物体遮挡，长 horizon drift |
| VR inside-out (ActiveUMI, XRZero-G0) | ~4mm | 贵，online tracking 没有 offline optimization的好处 |
| Base station (TacUMI, RDT2, FastUMI Pro) | ~3mm | 需要 instrumented environment，杀掉 in-the-wild |
| **Head stereo-inertial SLAM (HiFi-UMI)** | **~3mm** | **portable + offline optimization + native bimanual** |

关键 insight 来自一个 simple observation：**头部视角比腕部稳定得多**。在 manipulation 中，手附近的视角被 own hand、object、self-occlusion 持续 corrupt；而头部 motion 通常远小于 hand motion。这意味着 SLAM 的视觉约束更稳定，accumulated drift 也小。

更进一步，**bimanual 的 relative pose 是 natively measured 的**：两个 marker cube 同时出现在 head camera 的同一帧里，所以 inter-gripper relative pose 直接继承 per-gripper pose 的精度，不需要 cross-camera co-visibility 后处理重建。这恰好是 coordinated bimanual task（比如 shirt folding）最在乎的 axis。

这里有个**放弃 global loop closure** 的关键工程妥协。SLAM 通常靠 loop closure 来 bound global drift，但 manipulation 持续改变场景——抓起一个物体、移动它、放下它——这违反了 standard loop closure 背后的 static-world assumption (DynaSLAM https://arxiv.org/abs/1806.05620 处理过类似问题)。HiFi-UMI 的处理：用 **dynamic sliding window 的 local-consistency constraint** 替代 global loop closure，结果是 long-horizon global drift bound 在 centimeter 级，但 workspace-local accuracy 仍然是 mm 级。这是一个聪明的 trade-off——deployment 关心的本来就是 workspace-local 轨迹，而不是整段 video 的全局一致性。

精度数字 (Table 2)：

| Metric | Value |
|---|---|
| Local end-effector error (~2m workspace) | 3 mm |
| Cross-sensor timing offset | < 40 μs |
| Dropped frames (6 cameras @ 25 fps) | <1 per 270,000 frames |
| SLAM reconstruction success | 98% |
| Gripper opening-angle error | < 0.1° |

3mm 这个数字要 contextualize：它和 VR controller tracking、base station tracking 在同一量级，但**不需要任何外部 instrumented infrastructure**。这是和 base-station-based 系统（RDT2 等）的本质差别。

### 2.2 Gripper Morphology: Full-Palm Glove

Trigger-based UMI gripper 直接 emulate parallel-jaw command，但 tactile 对应感弱；finger-sleeve 触觉好但通常窄长，limit heavy/large object 任务。HiFi-UMI 设计了一个**非对称两指 glove**：narrow fingertip (precision) + wide proximal (load)。这覆盖 task spectrum 更广。

### 2.3 Cameras: 6 Views, ~200° FoV

每个手两个**非平行** fisheye camera（top + bottom），加上 head stereo，共 6 cameras。这里关键的 design choice 是 **non-parallel**——两个相机有 baseline，stereo cue 在 gripper 周围 dense。horizontal + vertical coverage 都 ~200°，相对 UMI 的 155° 单 wrist fisheye 有大幅提升。

deployment 时 policy **只接收 4 个 wrist views**，head stereo pair 只用于 offline trajectory reconstruction。这意味着 policy inference cost 不被 head 数据增加，但 training data 的 pose 质量受益于 head SLAM。这是个 asymmetric usage——capture 时用全部，deploy 时只用子集。

### 2.4 Synchronization: Single GPIO Hardware Trigger

<40μs 跨 sensor 同步，相对 software timestamp alignment 的 ms 级。这个差距看起来小，但在快速 hand motion 时，ms 级 timestamp 错位会直接污染 action label——image 是 t 时刻的，但 pose 是 t+5ms 的。GPIO trigger 从硬件层面消除这个 noise source。

---

## 3. Data Processing Flywheel

不是单纯采集，而是六阶段 flywheel：

1. **Collection & upload** — 在线 quality 监控，实时 voice feedback，Wi-Fi cloud streaming
2. **Trajectory reconstruction & auto cleaning** — offline SLAM + 异常轨迹自动 recompute
3. **Simulation retargeting** — whole-body motion control algorithm replay validation
4. **AI-assisted annotation** — multi-view joint reasoning，跨视角解决单视角遮挡
5. **Human verification** — 采样式，focus 在 low-confidence 样本
6. **Analysis & export** — statistical balancing，versioned release

cumulative yield 98% × 98% ≈ **96%** raw captures 成为 robot-executable data。这个数字和 AgiBot World (https://arxiv.org/abs/2503.06669) 这类大规模 robot 数据集的 curation 标准相当，是个非常高的 yield。

**Simulation retargeting validation** 这步是 paper 的一个 under-appreciated 设计。每条 trajectory 都要在仿真里用 WBC (whole-body controller, 类似 UMI on legs https://arxiv.org/abs/2407.10353) replay 一次，kinematically/dynamically infeasible 的轨迹直接丢弃。这相当于在数据进入 training 之前就做了 robot-executability 的硬约束验证——这是为什么 HiFi-UMI 数据可以直接 deploy 的一个 silent reason。

---

## 4. Action Representation: Chunk-Anchored Relative Target

这是一个所有 backbone 共享的物理约定，公式 (2)：

$$
\Delta \mathbf{T}_{t_0, h}^{j, (m)} = \left(\mathbf{T}_{t_0}^{j}\right)^{-1} \mathbf{T}_{t_0 + \delta_h^{(m)}}^{j}
$$

变量分解：

- $j$ — arm 索引，$\{L, R\}$ 之一（bimanual）
- $m$ — backbone 索引（StarVLA-QwenPI / OpenPI-π0.5 / LingBot-VA）
- $t_0$ — chunk 起点 time step，是 query 时的 measured anchor
- $h$ — chunk 内 step 索引（0 到 $H_m - 1$）
- $\delta_h^{(m)}$ — backbone $m$ 在 index $h$ 处的 native future offset（不同 backbone 有不同 temporal stride）
- $\mathbf{T}_{t_0}^j$ — arm $j$ 在 $t_0$ 时刻的 end-effector pose（SE(3) 4×4 matrix）
- $\Delta \mathbf{T}_{t_0, h}^{j,(m)}$ — 相对于 anchor 的 relative pose

**关键 design choice**：chunk 内每一行 target 都相对于**同一个 measured $t_0$ anchor**，而不是递归相对于前一个 action target。这避免了 error accumulation——如果第 $h$ 步是相对于第 $h-1$ 步的预测，那么早期 step 的小 error 会被 chunk 后续 step 放大。

Translation 在 end-effector frame，orientation 用 **Rotation6D** (https://arxiv.org/abs/1812.07035, Zhou et al. CVPR 2019)——取 rotation matrix 前两行 6 维表示，避免 quaternion 的 double-cover 不连续性和 Euler 角的 gimbal lock。Gripper target 是 absolute（不取 relative）。

每个 arm 10 channels (3 translation + 6 rotation + 1 gripper)，bimanual 20 channels。三个 backbone 各自把这 20 channels 塞进 native tensor：

- **StarVLA-QwenPI**: 直接用 20 channels
- **OpenPI-π0.5**: pad 进 32-dim action tensor
- **LingBot-VA**: 映射进 30-dim tensor，未用 channel mask 掉

这种"common physics, native tensorization"设计是 paper 评估策略的精髓——同一份 supervision，三种不同的 tensorization，如果三种都达到 parity，那是 data 的功劳，不是任何一种 architecture 的 trick。

---

## 5. 三个 Backbone 的技术细节

### 5.1 StarVLA-QwenPI (Modular VLA)

Backbone: **Qwen3-VL-4B-Instruct** (https://arxiv.org/abs/2505.09388, 36 transformer layers, hidden width 2560)。Action head: **π-style conditional flow-matching DiT** (借鉴 π0 https://arxiv.org/abs/2410.24164, DiT https://arxiv.org/abs/2212.09748, flow matching https://arxiv.org/abs/2210.02757)。

每个 DiT block cross-attend 到对应 layer 的 Qwen feature。Paper 自己加了一个 modification：**attention-only self-attention residual after every odd DiT block**——couple $H=20$ action steps 但不重复 feed-forward 计算。这是 paper 对 StarVLA QwenPI 原 path 的一个改进，原 path 缺 action-side self-attention，限制了 chunk 内 token mixing。

公式 (3) 的 flow matching 训练目标：

$$
\begin{aligned}
a^{\tau} &= (1 - \tau) \epsilon + \tau a, \\
\mathcal{L}_{\mathrm{FM}} &= \mathbb{E}\Big[\| v_\theta(a^\tau, \tau, z_t) - (a - \epsilon) \|_2^2\Big].
\end{aligned}
$$

变量：

- $\tau$ — flow time（注意这里是上标 $\tau$，不是时间 $t$），sample 自 $u \sim \text{Beta}(1.5, 1.0)$，然后 $\tau = (s - u)/s$，$s = 0.999$。这个 sigmoid-ish 时间采样是 flow matching 的常见技巧：Beta(1.5, 1.0) 偏向小 $u$，意味着 $\tau$ 偏向 1（接近 data），让训练更多关注 data-rich 区域
- $\epsilon \sim \mathcal{N}(0, I)$ — Gaussian noise
- $a$ — ground-truth action chunk
- $a^\tau$ — interpolated sample，在 noise 和 data 之间
- $v_\theta$ — 网络预测的 vector field
- $z_t$ — conditioning，来自 Qwen 的 layer-wise features

推理时从 Gaussian noise 用 **8 explicit-Euler steps** 积分 learned vector field。结果 chunk $H=20$，执行前 $H_{\text{exec}}=10$ 步（receding horizon, ACT-style https://arxiv.org/abs/2304.13705 / Diffusion Policy https://arxiv.org/abs/2303.04137）。

### 5.2 OpenPI-π0.5 (Public VLA Checkpoint)

JAX 实现的 OpenPI-π0.5 (https://arxiv.org/abs/2504.16054)。Backbone: **PaliGemma** (https://arxiv.org/abs/2407.07726) visual-language stream (SigLIP So400m/14 visual encoder + 18-layer Gemma-2B LM) + **Gemma-300M action expert** (18 layers)。Streams 用 modality-specific parameters + **Mixture-of-Transformers joint attention**（来自 π0 https://arxiv.org/abs/2410.24164）。

Proprioceptive state **discretized and serialized together with instruction**，不是作为 continuous action-side token 注入。Action tokens attend to full prefix + to each other，整个 chunk 联合预测。

公式 (4) flow matching：

$$
\mathcal{L}_{\mathrm{FM}} = \mathbb{E}\left[\left\| v_\theta(x_t, t \mid c) - (\epsilon - a) \right\|_F^2\right]
$$

变量：

- $t$ — flow time
- $x_t = (1-t)a + t\epsilon$ — interpolation（注意是 $t \to 1$ 趋向 noise，跟 StarVLA 公式里 $\tau$ 的方向定义略不同，这是为什么 vector field target 是 $\epsilon - a$ 而不是 $a - \epsilon$）
- $c = (o, q, \ell)$ — conditioning context（observation, proprio, language）
- $\|\cdot\|_F$ — Frobenius norm

Paper 明确说这是**sole fine-tuning objective**，omit 了 full π0.5 recipe 的 knowledge insulation、autoregressive subtask generation、auxiliary text loss。这让 comparison 更干净——只测 raw flow-matching BC，不测 π0.5 的 fancy extras。推理用 10 Euler steps。

### 5.3 LingBot-VA (Causal WAM)

WAM (World-Action Model) 的代表 (https://arxiv.org/abs/2601.21998)。和 VLA 的根本区别：action 不直接从 current observation 解码，而是**先预测 future visual states，再从 future latents 反解 action**——inverse dynamics 形式。

公式 (5) factorization：

$$
\begin{aligned}
& p_\theta(a_{t:t+H-1}, z_{t+1:t+K} \mid h_t, \ell) \\
&= p_\theta(a_{t:t+H-1} \mid z_{t+1:t+K}, h_t, \ell) \cdot p_\theta(z_{t+1:t+K} \mid h_t, \ell)
\end{aligned}
$$

变量：

- $z_t = E_{\text{VAE}}(o_t)$ — multi-view observation 经过 causal VAE 的 latent
- $h_t = (z_{\leq t}, a_{<t})$ — video-action history
- $\ell$ — language instruction
- $K$ — video horizon
- $H$ — action horizon
- 第二个因子 $p_\theta(z_{t+1:t+K} \mid h_t, \ell)$ — forward dynamics，生成 future video latents
- 第一个因子 $p_\theta(a_{t:t+H-1} \mid z_{t+1:t+K}, h_t, \ell)$ — inverse dynamics，从 future latents 解码 action chunk

Loss：$\mathcal{L}_{\text{LingBot}} = \mathcal{L}_{\text{video}} + \mathcal{L}_{\text{action}}$，等权。Block-causal mask 保证 interleaved video-action chunks 的因果序。LingBot-VA 用 chunk-anchored convention，20 active bimanual dimensions 通过 fixed channel map 进 native 30-dim tensor。Rotation6D 取 rotation matrix 前两行。

**部署**：bounded rolling KV cache，video/action guidance scales $5/1$，$8/16$ denoising steps，attention window 24。Episode reset 后 first call 生成 12 actions，之后每次生成 24-action 两-block chunk。Source-time stride 3 + VAE downsampling factor 4 → 12 native action slots per latent video frame。

---

## 6. 实验设计 — Frozen Benchmark + Asymmetric Data

### 6.1 评估协议

- **Hardware**: Tianji Robotics Marvin M6 bimanual platform，2 × 7-joint force-controlled arms，125 Hz IK，1 kHz EtherCAT
- **Decoupled evaluators**: policy operator + scene operator 分离，前者跑 policy，后者独立构 sample test instance
- **Pre-frozen benchmark**: task definitions, object sets, language instructions, checkpoints, initial-condition bank, timeouts, safety rules 在评估开始前 freeze
- **Randomized policy order** 减少 temporal/operator bias
- **40 rollouts per task-policy pair**

### 6.2 数据 asymmetry (关键 design)

| 条件 | UMI | Teleop |
|---|---|---|
| 轨迹数 / task | 3,200 | 300 |
| 小时数 / task | 10–20h | 3–7h |
| 收集场景 | evaluation scene **外** | evaluation scene **内** |

**这个 asymmetry 故意偏向 teleop**——teleop data 在 evaluation scene 收集，所以 visual context 完全 in-distribution；UMI data 完全是 scene-level distribution shift（background、illumination、tabletop 全不同）。这是 conservative test。Paper 强调这是 **practical data-production pipeline 比较**，不是 per-trajectory efficiency 比较——300 条 teleop 实际收集 wall-clock 时间是 3,200 条 UMI 的几倍，因为 teleop 需要 robot execution、environment reset、safety checks、recovery。

### 6.3 七个 training conditions (Table 4)

| ID | Backbone | Init / Pre-train | Post-train data |
|---|---|---|---|
| C1 | StarVLA-QwenPI | Qwen3-VL, scratch action head | HiFi-UMI |
| C2 | StarVLA-QwenPI | Qwen3-VL, scratch action head | Teleop |
| C3 | OpenPI-π0.5 | pi05_base | HiFi-UMI |
| C4 | OpenPI-π0.5 | pi05_base | Teleop |
| C5 | LingBot-VA | lingbot-va-base | HiFi-UMI |
| C6 | LingBot-VA | lingbot-va-base | Teleop |
| C7 | StarVLA-QwenPI | Qwen3-VL → HiFi-UMI pre-train | HiFi-UMI |

两个 comparison：
- **Data-source comparison**: C1 vs C2, C3 vs C4, C5 vs C6 — 固定 backbone/init/recipe/deploy，只换 post-train data source
- **Initialization comparison**: C1 vs C7 — 固定 post-train data，只换 init（Qwen3-VL scratch vs + HiFi-UMI 4000h pre-train）

---

## 7. 核心结果

### 7.1 UMI vs Teleop 的 Parity (zero-robot post-training 假设验证)

| Backbone | UMI | Teleop | Diff (pp) |
|---|---|---|---|
| StarVLA-QwenPI | 51.3% (82/160) | 53.8% (86/160) | **−2.5** |
| OpenPI-π0.5 | 77.5% (124/160) | 74.4% (119/160) | **+3.1** |
| LingBot-VA | 56.9% (91/160) | 57.5% (92/160) | **−0.6** |

观察：

1. **方向正负都有**，没有系统性 bias
2. **gap 都在 sampling noise 内**——40 rollouts per task-policy pair 意味着 1 个 success = 2.5 pp。−2.5/+3.1/−0.6 等价于大约 1 个 rollout 的差异
3. **跨 backbone 一致**，说明结论是 data 性质，不是某个 architecture 的 artifact
4. 最强 policy 在 Remote Insertion 上达到 **85%**——这是个 precision insertion task，且 teleop baseline 是 in-scene 收集，UMI 不在 evaluation scene——所以 85% 是在 scene shift + zero teleop 条件下达成的

Task-level 细节（Fig. 9, 11）：

- **Stain Wiping**: 大致持平，UMI 在 OpenPI 上 tie (65.0%)；LingBot 上 UMI 略胜 (62.5% vs 60.0%)
- **Shirt Folding**: teleop 略胜，StarVLA 60.0% vs 52.5%；OpenPI 80.0% vs 77.5%
- **Remote Insertion**: UMI 略胜，StarVLA 52.5% vs 50.0%；OpenPI 85.0% vs 77.5%（这个 gap 较大，但仍是 task-level noise 范围）；LingBot teleop 胜 50.0% vs 42.5%
- **Produce Sorting**: OpenPI UMI 82.5% vs 75.0%

### 7.2 UMI 数据 scaling (Fig. 10a)

Remote Insertion 上 OpenPI-π0.5 训练 UMI 数据 scaling：400 / 800 / 1600 / 3200 / 6400 episodes → 37.5% / 65.0% / 70.0% / 85.0% / 82.5%。

观察：

- 低数据 regime 增长快——400→800 几乎翻倍 (37.5→65.0)
- ~3,200 后饱和，6,400 略降（noise 范围）
- 这是 ICLR 2025 scaling law (https://arxiv.org/abs/2410.18647) 风格的 saturation 现象，但在 task-specific post-training 上而非 full pre-training

### 7.3 LingBot-VA 的 Qualitative Tempo 差异

Paper 报告了一个 qualitative observation：UMI-post-trained policy 倾向 **larger-amplitude, more continuous, more natural-looking motions**；teleop 倾向 incremental corrections。在 Remote Insertion 上 UMI 经常 decisive initial grasp 但 retry 多，所以成功率反而略低。这暗示一个未来方向：**broad coverage of contact correction + regrasp behaviors** 仍然重要——UMI 数据可能 nominal execution 更好，但 recovery robustness 上还需要数据多样性补足。

### 7.4 Oracle Ground-Truth-Video Analysis (Fig. 12)

这是 WAM 评估的 diagnostic，把 LingBot-VA 的 future-video latent 替换成 ground-truth cached latent，isolate action decoder 性能。

公式 (7) XYZ RMSE：

$$
E_{\mathrm{XYZ}} = 10^3 \sqrt{\frac{1}{6H} \sum_{t=1}^{H} \sum_{b \in \{L, R\}} \|\hat{\mathbf{p}}_{t, b} - \mathbf{p}_{t, b}\|_2^2}
$$

变量：

- $H$ — chunk length，这里 = 12
- $t$ — chunk 内 step，1 到 H
- $b$ — arm index，$\{L, R\}$
- $\hat{\mathbf{p}}_{t,b}, \mathbf{p}_{t,b}$ — predicted / ground-truth 3D position
- $6H$ — total count：$H$ steps × 2 arms × 3 (XYZ) = $6H$ values
- $10^3$ — meters → mm 转换

公式 (8) Rotation error：

$$
E_{\mathrm{rot}} = \frac{1}{2(H-1)} \sum_{t=2}^{H} \sum_{b \in \{L, R\}} d_{\mathrm{SO}(3)}\Big(\widehat{\Delta \mathbf{R}}_{t, b}, \Delta \mathbf{R}_{t, b}\Big)
$$

变量：

- $\Delta \mathbf{R}_{t,b} = \mathbf{R}_{t-1, b}^\top \mathbf{R}_{t, b}$ — adjacent-frame rotation increment
- $d_{\mathrm{SO}(3)}$ — 测地角（geodesic angle），$\arccos\left(\frac{\text{tr}(R_1^\top R_2) - 1}{2}\right)$
- $t=2$ 开始因为 $t=1$ 的前驱是 external chunk anchor，不是预测序列里的
- $2(H-1)$ — total count：$(H-1)$ 增量 × 2 arms。这就是为什么不是 $H$ 而是 $H-1$——adjacent increment count 比 step count 少 1
- $\widehat{\Delta\mathbf{R}}$ 是预测的相邻增量，$\Delta\mathbf{R}$ 是 ground-truth 的相邻增量

为什么要用 adjacent-frame increment 而不是 anchor-relative？为了避免 accumulated anchor-relative drift 主导 metric——你想测的是 **local rotation direction/magnitude** 的 fidelity，不是 long-horizon drift。

结果：

| Condition | XYZ RMSE | SO(3) error |
|---|---|---|
| Teleop→Real | 21.64 mm | 0.46° |
| UMI→Real | 24.33 mm | 0.65° |
| UMI→UMI | 21.13 mm | 0.88° |
| Random→Real | 117.57 mm | 126.47° |
| Random→UMI | 123.80 mm | 126.49° |

观察：

1. UMI→Real vs UMI→UMI 跨域差只有 **3.20 mm XYZ, 0.23° SO(3)**——这意味着 UMI-trained decoder **跨域 generalize 几乎无损**
2. UMI→Real vs Teleop→Real 差 **2.69 mm XYZ, 0.19° SO(3)**，且 random 基线差 ~95 mm / 125°，所以 UMI 和 Teleop 在 oracle 条件下都接近 noise floor
3. UMI→Real 相对 Random→Real 降 **79.3% translation, 99.5% rotation**
4. 这说明 WAM 的 inverse dynamics 部分（action decoder）从 UMI 学到的 pose 解码能力是 deployment-grade 的——closed-loop 残留的失败主要来自 future-video generation，不是 action decoding

---

## 8. Pre-Training Scaling (StarVLA-QwenPI only, 4,000 hours)

### 8.1 Held-Out Action Error (Fig. 13)

公式 (9) power-law fit：

$$
\mathcal{L}_{\mathrm{heldout}}(S) = \mathcal{L}_\infty + A \cdot S^{-\alpha}
$$

变量：

- $S$ — cumulative UMI action chunks processed globally
- $\mathcal{L}_\infty$ — asymptotic lower bound（不可约 error floor）
- $A$ — prefactor
- $\alpha$ — scaling exponent，控制曲线陡峭程度

拟合：$\alpha = 0.268$, $R^2 = 0.993$（在 LR decay 之前）。One pass through 4000h = 180k steps，held-out action error 降 **61%**。

注意这个 fit 测的是 **exposure scaling**（同一 corpus 多次 pass），不是 dataset-size scaling（增加 corpus 大小）。这是 paper 自己 explicit 区分的——因为 corpus fixed，所以这是 compute/data exposure 维度的 scaling 而非 "more diverse data" 维度的 scaling。

### 8.2 OOD Transfer (Fig. 14) — 最 informative 的分析之一

10 个**未在 pre-training 出现**的 task，平均 action error 降 **41%**。每个 task 都改善。但**改善速率按 task family 显著不同**：

- **Rigid utensil/tableware**: 改善最快（pre-training 1/3+ frames 覆盖 pick-and-place 类）
- **Granular transfer**: 中间
- **Cloth folding**: 改善最慢（pre-training <1% frames 覆盖 textile folding）

OOD scaling power-law exponent $\alpha = 0.095$——比 in-distribution 的 0.268 小很多，说明 OOD 改善更慢但仍存在。

**这个分析的 deep insight**：transfer depends more on **interaction dynamics coverage** than on **object identity**。如果你 pre-training 见过很多 pick-and-place rigid objects，你对新 rigid objects 的 transfer 很好；但如果你 pre-training 没见过 cloth folding 的 dynamics，你对新 cloth folding 的 transfer 慢——哪怕 testing object 本身不同。

这给了一个 actionable 的数据采集 priority：未来 UMI 数据采集应该 balance interaction dynamics（rigid pick-place / deformable / granular / contact-rich / regrasp），不只是 task diversity。这呼应 paper 在 limitations 里说的"characterizing which interaction dynamics a deployable policy most depends on, and when that coverage saturates, remains a central open question"。

### 8.3 Pre-Training 对 Post-Training 的 Benefit (Fig. 10b, 15)

Remote Insertion 的 init 比较：

- Qwen-VL scratch baseline + 3,200 demos → ~73% success（Fig. 10b 虚线）
- HiFi-UMI pre-trained + **800 demos** → 超过 scratch 3,200 baseline
- + 1,600 demos → 80%
- + 3,200 demos → 仍 above baseline

四个 benchmark 综合（Fig. 15）：

| Init | Stain Wiping | Shirt Folding | Remote Insertion | Produce Sorting | Aggregate |
|---|---|---|---|---|---|
| Qwen3-VL scratch | ~45% | ~52% | ~52% | ~55% | 51.3% |
| + HiFi-UMI 4000h pre | ~70% | ~75% | ~73% | ~70% | 72.4% |
| OpenPI-π0.5 reference | 65% | 80% | 85% | 82.5% | 77.5% |

Aggregate 提升 **+18.1 pp**。最 striking 的是用 1/4 数据（800 vs 3,200）就能超过 scratch baseline——这是 pre-training 给的 **data efficiency** 上的 gain，不只是 ceiling 上的 gain。

---

## 9. 数据集 Release

**HiFi-UMI-2K** (https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K)：

- 2,000 小时 curated subset
- 482,100+ episodes，110+ scenes
- 6 camera views per episode（synchronized）
- CC BY 4.0 license
- Human faces masked

Full corpus 实际有 20,000+ 小时，4.32M+ episodes，480+ scenes。2K 是 curated release。

---

## 10. 批判性思考与 Limitations

Paper 自己列了几个 limitations，但作为 reader 还可以多挖几个：

1. **No fidelity ablation**: 这是最大的 caveat。Paper 证明 "high fidelity as a whole" 够，但不证明 3mm vs 6mm、GPIO vs software sync、200° vs 155° FoV 各自的 marginal contribution。一个 systematically degrade 每个 axis 的 ablation 会把"high fidelity helps"变成 actionable spec。Paper 自己 acknowledge 这点留 future work。

2. **Sample 不 matched**: UMI 用 3,200 条 / task，teleop 用 300 条 / task。Paper 解释这是 practical pipeline throughput 比较，不是 per-trajectory efficiency 比较。但读者应该意识到——这个比较 favorable 给 UMI 数据 volume 优势。如果 matched sample (300 vs 300)，结果可能不同。

3. **Task scope 有限**: 4 个 tabletop bimanual task。Long-horizon、mobile manipulation、high-precision industrial task 都没测。Paper 承认 generality 未测。

4. **Pre-training evidence 只在 StarVLA-QwenPI 上**: OpenPI 和 LingBot 都没做 pre-training ablation（用 publicly released base checkpoints）。如果 pre-training benefit 跨 backbone 复现，会更强化。但 paper 明确说这是 narrowed evidence。

5. **Fidelity 测的是 whole-pipeline output**：3mm 是 pipeline 处理后的 trajectory accuracy，不是 capture-time raw accuracy。Simulation retargeting 和 auto-cleaning 这两步可能"修复"了一些 capture-time error。所以严格说，paper 测的是 "processed fidelity"，不是 "raw fidelity"。

6. **Bimanual relative pose 的 native measurement 依赖 head camera 观察 marker cube**：如果 head camera 视角被遮挡（比如 operator 低头看桌面），marker cube 可能丢失。Paper 没详细讨论这个 failure mode 的频率。

7. **Power-law 拟合 α=0.268 in-domain vs α=0.095 OOD**：这个差距很大，意味着 in-distribution scaling 的收益远高于 OOD。但 OOD scaling 仍然存在——如果未来 10x corpus，OOD error 还能再降一截。但是 α=0.095 是 shallow exponent，可能需要 corpus 量级跃迁才能看到显著 OOD 收益。这呼应 LLM scaling literature 里 in-domain vs OOD 的常见 pattern。

8. **WAM 的 oracle 分析揭示 bottleneck 在 video generation**：UMI→Real 24.33mm vs UMI→UMI 21.13mm 几乎无损，说明 action decoder 跨域 generalize 好。但 closed-loop LingBot-VA 只有 56.9% success——gap 来自 future-video generation。这说明 WAM 在 UMI data 上要 deploy-grade，瓶颈在 video generation 的 fidelity，不是 action supervision。

---

## 11. 与相关工作的 Landscape

- **UMI original** (https://arxiv.org/abs/2402.10329, RSS 2024): handheld gripper 奠基。Wrist VI-SLAM，~6mm 误差，155° 单 fisheye，software sync，reconstructed relative pose。HiFi-UMI 把每个 axis 都升级。

- **FastUMI / FastUMI Pro** (https://arxiv.org/abs/2409.19499): 用 dedicated tracker (T265) 替代 bespoke SLAM，更 robust。Pro 加 external lighthouse + wrist VIO fusion，~3mm 但需要 fixed infrastructure。

- **VISTA** (https://arxiv.org/abs/2606.04708, 2026): 最接近的 baseline——也是 UMI-only post-training on bimanual。但所有 baseline 都在同一 handheld corpus 上训，只 isolate model + curation design，不 isolate data source vs teleop。HiFi-UMI 是第一个把 UMI post-training 和 teleop post-training 在 same robot 上直接比较的。

- **ActiveUMI** (https://arxiv.org/abs/2510.01607): VR controller + 真实 gripper 复制。~4mm，native relative pose，3 views，ms software sync。Shrink real-robot fraction to small share，不 eliminate。

- **XRZero-G0** (https://arxiv.org/abs/2604.13001): VR headset + dual grippers + closed-loop quality inspection。Same as ActiveUMI 在 fidelity 维度上。

- **RDT2** (https://arxiv.org/abs/2602.03310, 2026): 10,000h UMI 数据，zero-shot cross-embodiment transfer on simple tasks。但 deploy-grade 时仍 mix 少量 real-robot data。Base-station tracking。

- **DexCap / DexUMI / DexWild** (https://arxiv.org/abs/2403.07788, https://arxiv.org/abs/2505.21864): Dexterous hand 数据采集。

- **AirExo-2** (https://arxiv.org/abs/2503.03081): Paper 自己 explicit 归因 UMI-style 设备的两个 fidelity 瓶颈——SLAM-based pose + limited FoV。HiFi-UMI 正好 attack 这两点。

- **ARCap** (https://arxiv.org/abs/2410.08464): AR feedback 保证 kinematic validity。

- **EgoMimic** (https://arxiv.org/abs/2410.24221) / **H-RDT** (https://arxiv.org/abs/2507.23523): 用 human/egocentric data 但仍 co-train / fine-tune with robot data。HiFi-UMI 是第一个明确说 robot-free 数据**单独**够的。

- **π0 / π0.5** (https://arxiv.org/abs/2410.24164, https://arxiv.org/abs/2504.16054): VLA flow-matching + DiT action expert，HiFi-UMI 直接借用这个 recipe。

- **Diffusion Policy** (https://arxiv.org/abs/2303.04137, RSS 2023): receding-horizon chunk execution 范式源头。

- **FAST** (https://arxiv.org/abs/2501.09747) / **OpenVLA-OFT** (https://arxiv.org/abs/2502.19645, RSS 2025): action tokenization / chunked decoding。HiFi-UMI action representation 借鉴。

- **Scaling laws for imitation learning** (https://arxiv.org/abs/2410.18647, ICLR 2025): task-specific data scaling 的 saturation 现象。HiFi-UMI 的 Fig. 10a 在 Remote Insertion 上复现这个 pattern。

- **GR-1 / GR-2 / WorldVLA / DreamZero / VPP / UniPi** (https://arxiv.org/abs/2312.13139, https://arxiv.org/abs/2410.06158, https://arxiv.org/abs/2506.21539, https://arxiv.org/abs/2602.15922, https://arxiv.org/abs/2412.14803, https://arxiv.org/abs/2302.00111): WAM family 的代表。

- **DROID** (https://arxiv.org/abs/2403.12945), **BridgeData V2** (https://arxiv.org/abs/2308.12952), **Open X-Embodiment** (https://arxiv.org/abs/2310.08864), **AgiBot World** (https://arxiv.org/abs/2503.06669), **RoboMIND** (https://arxiv.org/abs/2412.13877): teleop corpus 代表，HiFi-UMI 直接对标它们的数据规模但成本更低。

- **Qwen3-VL** (https://arxiv.org/abs/2505.09388): StarVLA-QwenPI backbone

- **PaliGemma** (https://arxiv.org/abs/2407.07726): OpenPI-π0.5 backbone

- **Rotation6D** (https://arxiv.org/abs/1812.07035, Zhou et al. CVPR 2019): 连续 rotation 表示

- **AprilTag** (https://april.eecs.umich.edu/papers/details.php?id=8, Olson ICRA 2011): fiducial marker 系统

- **ORB-SLAM3** (https://arxiv.org/abs/2007.11898): offline stereo-inertial SLAM 后端

- **VINS-Mono** (https://arxiv.org/abs/1708.05776, Qin et al.): VIO 经典工作

- **Flow Matching** (https://arxiv.org/abs/2210.02757, Lipman et al. ICLR 2023): generative modeling 框架

- **DiT** (https://arxiv.org/abs/2212.09748, Peebles & Xie ICCV 2023): scalable diffusion transformers

- **DynaSLAM** (https://arxiv.org/abs/1806.05620): dynamic scene SLAM

- **UMI on legs** (https://arxiv.org/abs/2407.10353): WBC replay validation 借鉴

- **RoboVQA** (https://arxiv.org/abs/2311.00858): AI-assisted annotation

- **AdamW** (https://arxiv.org/abs/1711.05101): optimizer

- **StarVLA community** (https://github.com/starVLA/starVLA): modular VLA ecosystem

- **"A careful examination of large behavior models"** (https://www.science.org/doi/10.1126/scirobotics.adea6201, Science Robotics 2026): real-robot evaluation protocol reference

---

## 12. 论文在 robot learning 长叙事中的位置

如果看 robot learning 数据的"金字塔"层级（paper Sec 2.1 自己用这个 framing）：

```
                Egocentric video (Ego4D, Ego-Exo4D)
                       ↓ scale, no action
            UMI-style handheld (HiFi-UMI, UMI, FastUMI)
                   ↓ scale + actionable, robot-free
        Teleop on real robot (DROID, AgiBot World, RoboMIND)
              ↓ fully embodied, expensive
        Simulation (大量但 sim-to-real gap)
```

之前 UMI 这一层被定位为"中间层 + pre-training only"。HiFi-UMI 的工作 essentially 把这一层**向上推到了 deployment-grade**——它证明了只要 fidelity 够，UMI 数据可以承担 deployment role，不需要往下走到 teleop 层。

如果这个结果 hold得住并 generalize，意味着：

1. **Robot learning 的数据生产成本曲线会被重新画**：teleop 的 $X/hour vs UMI 的 $X/10hour 这种数量级差距。AgiBot World 100 robots × 4000 m² facility vs portable handheld 设备。

2. **Pre-training + Post-training 的 dataset 都可以是 robot-free**：如果未来 VLA foundation model 全部在 UMI 数据上 pre-train + post-train，teleop 真的会变成只在 evaluation 时才出现的角色。

3. **数据 fidelity 而不是 data volume 成为主要瓶颈**：HiFi-UMI 3mm 是因为 hardware co-design；如果只是简单 scaling UMI hardware 而不提升 fidelity，scale 出来的数据可能仍然只能 pre-training。这给硬件创新重新注入 legitimacy。

4. **Cross-task transfer 的 "interaction dynamics coverage" insight** 改变数据采集的优先级：不再是"采更多 task"，而是"采更广 interaction dynamics family"。这对 dexterous、contact-rich、deformable manipulation 尤其重要。

5. **WAM 在 UMI data 上 deploy-grade 的 bottleneck 是 video generation**：这是 oracle 分析暴露的——action decoder 已经够好，video generation 跨域 generalize 是 WAM 走向 deploy 的下一个 hill。

---

## 13. Open Questions（paper 自己提的 + 我的联想）

1. **Fidelity 各 axis 的 marginal contribution**：3mm vs 6mm pose error 对 deployment success rate 的边际影响？GPIO sync vs software sync 在哪些 task 上有差？ultra-wide FoV 在 contact-rich task 上的具体 contribution？

2. **Sample-matched UMI vs Teleop**：300 vs 300 而不是 3200 vs 300，结果会怎样？这个 paper 没测，但是回答"per-trajectory efficiency"的关键。

3. **Power-law 在 corpus 扩展时是否持续**：α=0.268 是 exposure scaling（同 corpus 多 pass）。如果 corpus 10x（更多 scene、更多 interaction dynamics），power-law 会保持还是 saturate？这关系到 robot-free data 的 ultimate ceiling。

4. **Pre-training benefit 是否在 teleop post-training 上也成立**：HiFi-UMI pre-trained init 在 teleop post-training 上是否也 +18 pp？如果是，那是 universal visual-motor prior；如果只在 UMI post-training 上有效，那是 matched-domain benefit。Paper 明确说这是 open question。

5. **Long-horizon tasks**：4 个 tabletop task 都是分钟级。Multi-stage、半小时级任务（做饭、整理房间）的 zero-robot post-training 是否仍然成立？

6. **Mobile manipulation / locomotion-manipulation coupled**：HiFi-UMI 是 stationary bimanual。Mobile manipulation（wheel + arm）的 UMI 数据采集和 deploy 是另一个 fidelity frontier。

7. **Dexterous hand（多指）**：DexCap/DexUMI 路线 + HiFi-UMI 风格 fidelity co-design 的结合，是个未探索 territory。

8. **WAM 的 future-video generation fidelity**：oracle 分析说 action decoder 够好，那 video generation 怎么提升？用 UMI 的多视角 observation 做 future latent prediction 的 fidelity ceiling 在哪？

---

## 14. 一个 build-intuition 的总结

如果你只记一件事，记这个：**在 robot learning 里，"robot-free data 不够 deploy" 的共识可能不是 robot-free setting 本身的限制，而是 prior UMI 数据 fidelity 不够高**。HiFi-UMI 通过 hardware co-design（head SLAM + native bimanual pose + GPIO sync + ultra-wide FoV + glove gripper）+ 软件pipeline（offline SLAM + sim retargeting validation + AI annotation + 人工 verify）把 fidelity 推到 deployment-grade，然后用三个不同 family 的 backbone 在 4 个 task 上证明 zero-robot post-training 和 in-domain teleop 在 sampling noise 内 parity。

而且 pre-training 在同 corpus 上给 +18 pp real-robot success，800 demos 就超 scratch 3,200 demos baseline，提示 robot-free pre-training 给的 visual-motor prior 是 reusable 的。

这是 robot learning 数据生产范式可能从 "teleop-centric + UMI-supplement" 转向 "UMI-centric + teleop-evaluation-only" 的一个 evidence-based 推力。如果未来 VLA foundation model 都在 HiFi-UMI 风格 corpus 上 pre-train + post-train，teleop 会退到只承担 evaluation 的角色——这跟 LLM 里 "human annotation 退到只做 eval, 训练全靠 self-supervised/web data" 的演化路径有结构上的相似性。

Reference web links:

- Paper & dataset: https://cloud.simpleai.tech/simple-world-lab/hifi-umi/ , https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K
- UMI: https://arxiv.org/abs/2402.10329
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- DROID: https://arxiv.org/abs/2403.12945
- BridgeData V2: https://arxiv.org/abs/2308.12952
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- AgiBot World: https://arxiv.org/abs/2503.06669
- RoboMIND: https://arxiv.org/abs/2412.13877
- FastUMI: https://arxiv.org/abs/2409.19499
- ActiveUMI: https://arxiv.org/abs/2510.01607
- AirExo-2: https://arxiv.org/abs/2503.03081
- DexCap: https://arxiv.org/abs/2403.07788
- ARCap: https://arxiv.org/abs/2410.08464
- EgoMimic: https://arxiv.org/abs/2410.24221
- H-RDT: https://arxiv.org/abs/2507.23523
- Scaling laws for imitation learning: https://arxiv.org/abs/2410.18647
- FAST: https://arxiv.org/abs/2501.09747
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- Rotation6D: https://arxiv.org/abs/1812.07035
- AprilTag: https://april.eecs.umich.edu/papers/details.php?id=8
- ORB-SLAM3: https://arxiv.org/abs/2007.11898
- Flow Matching: https://arxiv.org/abs/2210.02757
- DiT: https://arxiv.org/abs/2212.09748
- PaliGemma: https://arxiv.org/abs/2407.07726
- Qwen3: https://arxiv.org/abs/2505.09388
- DynaSLAM: https://arxiv.org/abs/1806.05620
- UMI on legs: https://arxiv.org/abs/2407.10353
- RoboVQA: https://arxiv.org/abs/2311.00858
- WorldVLA: https://arxiv.org/abs/2506.21539
- GR-1: https://arxiv.org/abs/2312.13139
- GR-2: https://arxiv.org/abs/2410.06158
- Ego4D: https://arxiv.org/abs/2110.07058
- AdamW: https://arxiv.org/abs/1711.05101
- StarVLA code: https://github.com/starVLA/starVLA
- Science Robotics eval protocol: https://www.science.org/doi/10.1126/scirobotics.adea6201
