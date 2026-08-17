---
source_pdf: Robotic Manipulation by Imitating Generated Videos.pdf
paper_sha256: 8fb3e2c4d70c61c34b0d3257f68935a5bdcdc24b5da6de240407e01143b8a09f
processed_at: '2026-08-12T01:51:04-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Karpathy，没问题，刚刚那篇太学术了，我用人话重新捋一遍这篇 paper 到底在搞什么飞机。

---

## 一句话总结

这篇 paper 说：**现在 AI 生成视频已经足够逼真了，机器人只要照着 AI 生成的视频学动作，就能在现实里把活干了，完全不需要真人手把手教它。**

---

## 1. 他们想解决什么痛点？

以前让机器人学干活，通常得人亲自操作机器人记录数据，或者找一堆真人视频让机器人学。又费钱又费力，换个机器人型号可能还得重新采数据。

这帮人就问了一个很狂的问题：**既然 Sora、Kling 这些模型能生成“人倒水”的视频，那机器人能不能直接看这个生成的视频，自己悟出怎么倒水？**

---

## 2. 他们到底怎么做的？

整个流程其实很直白，像个流水线：

1. **拍个照，下个指令**：告诉系统“给植物浇水”，输入一张当前桌面照片。
2. **生成视频**：拿这张照片和指令，喂给 Kling，生成一段大概 5 秒的“倒水”视频。
3. **质检过滤**：因为生成的视频经常跑偏，比如水壶乱飞或者没倒进盆里。他们用 GPT-4o 抽几帧看看，如果觉得这视频不对劲，就扔掉重生成。
4. **提取物体运动轨迹**：视频有了，得搞清楚里面的水壶到底是怎么动的。他们先估测视频每一帧的深度，然后用一个叫 FoundationPose 的 tracker，把水壶每一帧的 6D pose（三维位置+三维旋转）抓出来。
5. **照葫芦画瓢**：机器人先抓住水壶，然后因为抓住了就不变了，它只要让自己的机械臂跟着水壶的 6D pose 轨迹走就行。

---

## 3. 听起来简单，凭啥能发顶会？

因为结果太让人意外了。他们证明了三件事：

1. **生成的视频竟然跟真人演示一样管用**。他们拿 Kling v1.6 生成并通过 GPT-4o 过滤的视频去跑，成功率达到了 85%，而用真人录的视频跑，成功率是 100%。这说明现在的 video diffusion model 已经足够懂物理规律了。
2. **生成视频比让 VLM 直接写“解题步骤”强多了**。他们对比了 ReKep 这个方法，就是让 GPT-4o 输出几个关键点的关系，然后解方程。结果 ReKep 只有 50% 成功率，而直接生成视频的 RIGVid 有 85%。这说明现在的 VLM 直接去抽象提取关键点还不靠谱，生成密集的视频像素反而更准。
3. **6D pose tracking 完爆其他的提取方法**。有人用光流，有人用特征点追踪，都不太行。只有直接用 mesh 做 6D pose tracking 才稳。因为物体一转，点就被遮住了，但你有 mesh，就算背面被遮住，你也能算出它怎么转的。

---

## 4. 什么时候会拉胯？

他们很坦诚，这套系统现在主要卡在两个地方：

1. **算深度老算错**。所有的失败案例，几乎全是因为估测视频深度时出了偏差。比如勺子明明靠近摄像头了，但深度模型说没动。
2. **得提前建模型**。FoundationPose 要 track 物体，得提前知道物体的 3D mesh。虽然现在能靠扫一下物体建出来，但在真正的 open-world 里，随手捡个没见过的东西就没法搞了。

---

## 5. 我的直觉总结

这篇 paper 传递的最强信号是：**Robotics 的瓶颈正在从 robot 端转移到 generative model 端**。

以前是机器人太笨，怎么教都不会。现在是 generative model 进化太快，只要视频生成模型再迭代几代，把物理规律和深度搞准了，robot 这边甚至都不用改代码，就能免费升级，变得更厉害。这就是我最喜欢这篇 paper 的地方。

---

# RIGVid: Robots Imitating Generated Videos — 技术深度剖析

Karpathy 你好, 这篇 paper 我非常认真地读完了, 因为它触及了一个我一直关心的核心问题: **生成模型能否作为 robot policy 的 supervision source**. 我会从 motivation, system architecture, 数学公式, 实验数据, failure mode 几个维度展开, 并且加入大量我自己的延伸联想.

---

## 1. Top-Level Motivation: 为什么这件事重要

这篇 paper 想回答的核心 question 是: 给一个 state-of-the-art video diffusion model (Sora / Kling), 加一个 state-of-the-art 6D pose tracker (FoundationPose), 我们能不能让 robot 完全 **zero-physical-demonstration** 地完成 manipulation task?

之前 robotics from video 的工作分两派:
- **Large-scale video pretraining** (RT-X, VPT, GenRM 等): 用 internet video 学 affordance / latent action, 但是有 domain gap, 还要 fine-tune 到 specific robot embodiment.
- **Paired video imitation** (Human-to-Robot Imitation, MimicPlay, RobotTelekinesis): 需要在 controlled setting 下采集 demonstration video, viewpoint / morphology 要匹配.

RIGVid 走的是 **第三条路**: 让 generative model 自己 synthesize 一个 task-conditioned, scene-conditioned video, 然后从里面 extract trajectory. 这本质上是一种 **"world model as policy"** 的范式 — video diffusion model 隐式地编码了 (a) 物体 affordance, (b) 物理常识, (c) task semantics, 然后我们只需要把 visual pixel 信息 decode 成 executable SE(3) trajectory.

我的 intuition: 这就像让 LLM 来做规划, 但规划的"语言"是 raw video pixels, 而不是 text tokens. Video generation 的 computation cost 看似 wasteful, 但实验证明它恰恰比 sparse keypoint / affordance representation 提供更 dense, 更 reliable 的 supervision.

---

## 2. System Architecture: 完整 Pipeline 深度拆解

整个 RIGVid pipeline 由 8 个模块串起来 (Fig 1 + Fig 2):

```
[Initial RGB-D image]
        |
        v
[Language command: "pour water on the plant"]
        |
        v
[Video Diffusion (Kling v1.6)] -- candidate generated video
        |
        v
[VLM Filter (GPT-4o)] -- accept / reject (up to 5 retries)
        |
        v
[Monocular Depth Estimator (Ke et al.)] -- per-frame depth
        |
        v
[Affine depth alignment to real RGBD]
        |
        v
[Active Object Mask: GPT-4o -> Grounding DINO -> SAM-2]
        |
        v
[6D Pose Tracking: FoundationPose + BundleSDF mesh]
        |
        v
[Smoothing filter (moving average on rotation vectors)]
        |
        v
[AnyGrasp: grasp active object]
        |
        v
[Retargeting: T_ee_t = T_ee_obj · T_obj_t]
        |
        v
[Closed-loop Execution: real-time FoundationPose tracking
                          + deviation threshold (3cm / 20deg) backtrack]
```

### 2.1 Video Diffusion 模块

输入是 initial RGB image + language command, 输出是 ~5 秒的 generated video. Paper 试了 3 个 model:

- **Sora**: 视觉效果好但完全不遵守 prompt, 100% fail rate. 会改变 camera viewpoint, swap object, alter scene layout. 这其实是 video diffusion 训练时 prioritized cinematic aesthetics 导致的 side effect.
- **Kling v1.5**: 更遵循 prompt 但物理合理性差 (e.g., water 从 kettle 顶部倒出而不是 spout).
- **Kling v1.6**: 最 reliable, pass rate 分别 83% / 66% / 55% / 45% (pour / lift / place / sweep).

关键 hyperparameter: `relevance factor = 0.7`, negative prompt = "fast motion". 这两个 settings 在 Appendix A 里提到, 我觉得是 empirical 经验, 估计是避免 object 在 video 里 motion 太快导致 tracking 失败.

### 2.2 VLM Filtering 模块 (这是非常 clever 的设计)

为什么不直接用生成视频? 因为 Kling v1.6 仍有 ~50% 的视频不遵守 prompt. 怎么过滤? Paper 的 trick 是:

1. 从 video 里均匀采样 4 帧
2. 垂直 concatenate 成一张图
3. 喂给 GPT-4o, 问 "does this video perform the action in the command?"
4. GPT-4o 回答 Yes/No

Table 1 显示这个 filter 与 human judgment 的 Pearson correlation **平均 0.84** (pour 0.91 / lift 0.91 / place 0.91 / sweep 0.66). 注意它**几乎不会有 false positive** (错的视频通过), 主要 error 是 false negative (好视频被丢掉).

我自己的联想: 这种 "4-frame vertical stack" 的小 trick 其实是一种 cheap temporal summary. 类似地, 在 LLM eval 里我们常用 "first/middle/last frame" 做 video QA. 这里用 4 帧是为了 capture 到 start / mid-action / end state, 让 VLM 能判断 task 是否完成.

对照 Table 1, paper 还试了 VBench++ 的标准 metrics:
- **Video-text consistency**: 平均 Pearson 0.34 (基本没用)
- **I2V subject consistency**: 平均 0.37 (基本没用)

这说明: **off-the-shelf video eval metrics 完全不能预测下游 robot task success**, VLM-as-judge 才是 reliable 的. 这是一个相当 surprising 的 finding.

### 2.3 Depth Alignment 的数学

Monocular depth estimator (Ke et al. "Video Depth without Video Models") 输出的 depth 存在 **scale-shift ambiguity**: $\hat{D}_{pred} = s \cdot D_{true} + t$, 其中 $s$ 是 scale, $t$ 是 shift.

Paper 用 active object 区域的 real RGBD depth 做 calibration. 设 $M$ 是 active object mask, $D_{real}$ 是真实 depth. 最小化:

$$\min_{s, t} \sum_{p \in M} \| s \cdot \hat{D}_{pred}(p) + t - D_{real}(p) \|^2$$

这是 standard linear least squares, closed-form:

$$s^* = \frac{\text{Cov}_M(\hat{D}_{pred}, D_{real})}{\text{Var}_M(\hat{D}_{pred})}, \quad t^* = \mathbb{E}_M[D_{real}] - s^* \cdot \mathbb{E}_M[\hat{D}_{pred}]$$

其中 $\text{Cov}_M$ 和 $\text{Var}_M$ 都是只在 mask $M$ 内计算. 这个 $s^*, t^*$ 然后被 apply 到整个 video 的所有 frame.

我的 intuition: 这个 alignment 只在 first frame 的 active object 上做, 假设 video 内 depth scale 一致. 这是个 strong assumption, 但因为 video diffusion model 本身是 temporal consistent 的, 所以 OK.

### 2.4 Active Object Segmentation

三步 pipeline:
1. GPT-4o 看 image + command, 输出 object category (e.g., "watering can")
2. Grounding DINO 把 text category grounding 成 bounding box
3. SAM-2 把 bbox refine 成 pixel mask

这个 stack 已经是 robotics VLM pipeline 的 de facto standard. 我之前在 RT-2 / VoxPoser 都见过类似的组合.

### 2.5 6D Pose Tracking: 为什么 FoundationPose 是对的选

Paper 比较了多种 tracking 方案 (Section 4.4), RIGVid 选 **FoundationPose** 是关键决定. Why?

- **Model-based 6D tracker**: 需要预计算 object mesh (用 BundleSDF 扫一遍 object 得到), 但 tracking 极其 stable.
- **MegaPose** (Appendix J): translation jitter 0.0045m, rotation jitter 37.47°
- **FoundationPose**: translation jitter 0.0029m, rotation jitter 14.31°

Jitter 公式 (Appendix J):

$$\text{jitter}_{trans} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} \| \Delta \mathbf{t}_t \|^2}, \quad \text{jitter}_{rot} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} \theta_t^2}$$

其中:
- $N$ = total frames
- $\Delta \mathbf{t}_t = \mathbf{t}_t^{raw} - \mathbf{t}_t^{smooth}$ 是第 $t$ 帧的 translation residual (raw pose 与 Gaussian-smoothed pose 之差, σ=2)
- $\theta_t$ 是第 $t$ 帧的 angular residual, 通过 relative rotation $R_{smooth}^{-1} \cdot R_{raw}$ 计算得到, 用 axis-angle 表示取 angle, 单位 rad -> 转 degree

FoundationPose 的 jitter 比 MegaPose 小一倍多, 而且**支持 real-time inference**, 这对 closed-loop execution 是 critical 的.

### 2.6 Motion Retargeting: 核心 Math

这是 paper 最 elegant 的部分. 设:

- $T_{obj}^{grasp} \in SE(3)$: robot 在 grasp 那一刻 object 在 robot world frame 的 pose
- $T_{ee}^{grasp} \in SE(3)$: robot 在 grasp 那一刻 end-effector 的 pose
- $T_{obj}^{t} \in SE(3)$: 从 generated video 中提取的, 第 $t$ 帧 object 的 6D pose (已经 align 到 robot world frame)

那么 grasp 时 end-effector 到 object 的 fixed transformation:

$$T_{ee \to obj} = (T_{ee}^{grasp})^{-1} \cdot T_{obj}^{grasp}$$

注意这里 $T_{ee \to obj}$ 是一个 constant rigid transform, 因为 grasp 之后 object 不再相对 gripper 移动. 这是个 strong assumption, 但配合 AnyGrasp 选 high-quality grasp point, 在 paper 的 4 个 task 上都 work.

之后每一帧的 end-effector target pose:

$$T_{ee}^{t, target} = T_{obj}^{t} \cdot (T_{ee \to obj})^{-1} = T_{obj}^{t} \cdot (T_{obj}^{grasp})^{-1} \cdot T_{ee}^{grasp}$$

这就是 Fig 2 里画的"橙色 object pose trajectory 转换成蓝色 end-effector trajectory"的公式.

**Key insight**: 这个 retargeting **完全 robot-agnostic**. 换一个 robot (paper 试了 ALOHA bimanual), 只需要重新算 $T_{ee \to obj}$, 其他部分 (video generation, depth, tracking) 都不变. 这就是 paper 在 Fig 11 里展示的 ALOHA 80% success vs xArm 100% success 的来源 — 唯一区别就是 embodiment 的 kinematic limit.

我的延伸思考: 这个 formulation 假设 grasp 是 rigid 的. 对 **deformable object** (毛巾、绳结) 就会 break. Paper 在 Fig 11 里展示了 wiping (擦桌子) 和 ironing (熨烫), 但这两个 task 看起来 object 本身还是 rigidly grasped, 只是末端执行 complex trajectory. 真正的 deformable manipulation 还需要别的方法.

### 2.7 Closed-Loop Execution: 工程上的 critical design

这是 paper 让我 impressed 的另一个地方. 它**不是** open-loop replay trajectory, 而是做 real-time feedback control:

1. Robot 执行 trajectory 时, FoundationPose 实时 track 真实物体 pose $\hat{T}_{obj}^{t}$
2. 比较 $\hat{T}_{obj}^{t}$ 与 planned $T_{obj}^{t}$
3. 如果 position deviation > 3 cm OR orientation deviation > 20°
4. **Robot backtracks 到上一个 successfully executed trajectory point**, 然后从那里 resume

Fig 3 展示了 human push robot 的 recovery 过程. Appendix H 还展示了 grasp 失败 (object 在 gripper 里 rotate 了) 的自动 re-orient.

我的 intuition: 这本质上是一种 **"trajectory-level servoing"**, 而不是 standard visual servoing 的 pixel-level feedback. 它的 robustness 来自 6D pose 的 structured representation — 你能 cleanly 地比较两个 SE(3) pose 的差异, 但你很难 cleanly 比较 two optical flow fields.

---

## 3. 实验数据深度分析

### 3.1 Main result: Generated video ≈ Real video (Section 4.2)

Fig 7 的关键数据:

| Video Source | Pour | Lift | Place | Sweep | Avg |
|---|---|---|---|---|---|
| Sora (unfiltered) | 0% | 0% | 0% | 0% | 0% |
| Kling v1.5 (unfiltered) | low | low | low | low | low |
| Kling v1.6 (unfiltered) | 80% | 60% | 50% | 20% | 52.5% |
| **Kling v1.6 (filtered)** | **100%** | **80%** | **90%** | **70%** | **85%** |
| Real human demo | 100% | 100% | 100% | 100% | 100% |

(Tasks 从左到右难度递增)

**Takeaway**: 
1. Video quality 与 success rate **strongly correlated**
2. Filtered Kling v1.6 generated video 已经 almost match real demo, 差距只有 15% on the hardest task (sweeping)
3. 这意味着: **随着 video diffusion model 变强, RIGVid 会自动变强**, 这是个非常 attractive 的 scaling law.

### 3.2 vs ReKep (VLM-based sparse representation)

Fig 8: RIGVid **85% vs ReKep 50%**, gap 35%.

ReKep 是 Stanford 的方法, 让 VLM 输出 relational keypoint constraints (e.g., "keypoint 8 above keypoint 15"), 然后 solver 解出 trajectory.

Appendix F 显示 ReKep 在 pouring task的 failure mode:
- VLM 让 grasp keypoint 1
- Transport 时让 keypoint 8 above 15, 高度 above 7
- Pouring 时 keypoint 8 above 15 and above 4

但 robot 执行失败. 原因 (Appendix F, Fig 14): VLM 输出的 keypoint 位置本身就不对 — lid image 上没有 keypoint 在 handle 上, placing task 上 keypoints 都 cluster 在 pan 角落.

**我的核心 intuition**: VLM 直接输出 sparse symbolic representation (keypoints + constraints) 缺少 **dense geometric supervision**. Video generation implicit 编码了 spout 的方向、倾斜轴、water flow 的物理 — 这些都很难用几个 keypoint 表示. 所以 video 是 **richer representation**, 哪怕 compute cost 高.

### 3.3 vs 其他 trajectory extraction 方法

这是 paper 最 informative 的 ablation (Fig 9):

| Method | Representation | Avg Success |
|---|---|---|
| Track2Act | 2D point tracks (start->goal) | 7.5% |
| AVDC | Optical flow (dense) | 32.5% |
| 4D-DPM | 3D feature field (Gaussian splatting) | 35.0% |
| Gen2Act | Sparse 2D point tracks on generated video | 67.5% |
| **RIGVid** | **6D object pose trajectory** | **85.0%** |

为什么 RIGVid 完胜? 几个层次的原因:

1. **Track2Act (7.5%)**: 只用 start + goal image, 没有 intermediate frames. 它的 track prediction network 在 OOD scene 上 generalize 差. 这是 information-theoretically 不足.

2. **AVDC (32.5%)**: dense optical flow 听起来好, 但**累积误差**严重. 每帧 flow 误差小, 但 sum 起来 trajectory physically implausible. 这是 dense representation 的 numerical instability.

3. **4D-DPM (35%)**: 3D feature field 看起来 elegant, 但 tracking unstable and jerky. 因为它依赖 per-frame feature field alignment, 没有 temporal smoothing prior.

4. **Gen2Act (67.5%)**: 在 generated video 上跑 BootsTAP point tracking, 然后 PnP. 这是 RIGVid 最强的 baseline. 失败模式 (Appendix G, Fig 15): **object 大幅 rotate 时所有 visible point 都被 occlude**, PnP 没法解. 这是 correspondence-based method 的 fundamental limitation.

5. **RIGVid (85%)**: 用 object mesh + FoundationPose, **即使大部分 object 被 occlude**, 也能用 mesh 渲染 + ICP-like matching 维持 stable 6D pose. 这是 model-based tracking 的 inherent advantage.

我的延伸思考: 这其实是个 **representation selection** 的问题. 6D object pose 是 **structured, low-dimensional, geometrically constrained** representation. Point tracks 是 unstructured, high-dimensional, 容易 drift. 这就像 MPC vs end-to-end policy 的对比 — 前者用结构化 prior 换 sample efficiency, 后者用 flexibility 换 generalization. RIGVid 选了前者, 在当前 video quality 下 work.

### 3.4 Failure Mode Analysis (Appendix I)

Filter Kling v1.6 video 上失败的 trial, 除了 1 例 gripper slip, **全部归因于 monocular depth estimation error**. Appendix I Fig 17 做了一个 beautiful control experiment:

| Video Source | Depth Source | Success |
|---|---|---|
| Real video | Real depth (RGBD camera) | **100%** |
| Real video | Generated depth | 85% |
| Kling v1.6 video | Generated depth | 85% |

这说明: **当前 RIGVid 的 bottleneck 不是 video generation, 是 depth estimation**. 

两种 failure mode (Fig 18, 19):
1. **Inaccurate depth**: spatula 明显靠近 camera, 但 depth 只变 6.8 cm
2. **Temporal flickering**: 3 帧 0.066 秒内 depth 从 40.1cm 跳到 38.2cm, 物理不可能

我的联想: 如果未来有 **video depth model** (而不是 per-frame depth estimator), temporal flickering 应该能解决. Ke et al. 那篇 paper 叫 "Video Depth without Video Models", 反讽的是它不用 video model, 所以有 flickering. 真正的 video diffusion-based depth model (e.g., DepthCrafter, NVDS) 应该 better.

---

## 4. Limitations & My Critical Take

### 4.1 Mesh Pre-computation

FoundationPose 需要 object mesh, 用 BundleSDF 扫一遍物体得到. 这在 lab setting OK, 但在 truly open-world deployment (e.g., 捡到没见过的物体) 不 work.

Appendix C 显示 mesh-free version (BundleSDF 同时 tracking + reconstruction) 也 work, pouring task 90% success, **但 inference 要 30 minutes per video**, 不能 real-time. 这是 model-free tracking 的 speed-accuracy tradeoff.

未来方向: **3D Gaussian Splatting-based real-time model-free 6D tracker**, 比如 BundleSDF 的优化版, 或者 SplaTAM 的 tracking mode.

### 4.2 Grasp Rigidity Assumption

$T_{ee \to obj}$ 假设 grasp 后 object 与 end-effector 是 rigid connection. 这对硬物 OK, 对 deformable / articulated object (e.g., 衣服、剪刀) break. 

Paper Fig 11 显示了 wiping / ironing, 但我看 video demo 上, 这些 task 的 object 本身是 rigidly held (毛巾被拉紧后近似 rigid). 真正挑战是 pouring water 这种 fluid simulation, paper 没碰.

### 4.3 Single-Object Manipulation

整个 pipeline 只 track 一个 active object. 对 bimanual manipulation with two interacting objects (e.g., one hand holds bowl, other stirs) 需要 extend. Fig 11 bottom left 展示了 bimanual putting shoes in box, 但那是 two independent single-arm tasks, 不是真正 coordinated bimanual.

### 4.4 Video Generation Cost

Sora / Kling 生成一个 5 秒 video 要几十秒到几分钟, 不能 real-time. 所以 RIGVid 是 **offline planning** + online closed-loop execution. 对 reactive task (e.g., 抓苍蝇) 不适用.

### 4.5 Camera Viewpoint

Generated video 的 camera viewpoint 必须**与 robot camera viewpoint一致**, 否则 retargeting 会出错. Paper 在 Appendix A 强调 "viewed from a natural, human-like perspective". 这意味着 video diffusion model 知道从 input image 的 viewpoint 续生成. 如果 camera viewpoint 异常 (e.g., top-down), 生成质量会 drop.

---

## 5. 我的延伸联想 & 与 Related Work 的技术比较

### 5.1 RIGVid vs Gen2Act

Gen2Act (Bharadhwaj et al. 2024) 思路几乎一样, 但需要 **1,822 human-collected robot demonstrations** 训 policy. RIGVid 把这 1,822 demos 全砍掉, 只用 video + tracking + retargeting. 这是 "embodiment-agnostic" 的胜利.

但 Gen2Act 有一个 RIGVid 没有的能力: 它的 learned policy 可以 generalize到 training distribution 之外的场景. RIGVid 完全 rely on FoundationPose 的 mesh matching, mesh 没有 / object 大幅变形都会失败.

我的直觉: RIGVid 是 **"zero-shot inference"** 范式, Gen2Act 是 **"few-shot policy learning"** 范式. 前者更 data-efficient, 后者更 robust to distribution shift.

### 5.2 RIGVid vs Dreamitate

Dreamitate (Liang et al. 2024) 也是 video generation for robot, 但它在 robot data上 fine-tune video diffusion model. RIGVid 用 off-the-shelf Kling, 不 fine-tune. 这是 scaling 的胜利 — 通用 video model 比 robot-specific model 强太多.

### 5.3 RIGVid vs ReKep 的 representation debate

ReKep 是 symbolic, RIGVid 是 perceptual. 这让我联想到经典的 "neural vs symbolic" 辩论. Paper 的实验证明: **在当前 VLM 能力下, symbolic keypoints representation 太 sparse**, video 的 dense pixel representation 更 work. 但 future VLM 可能能输出更 rich 的 symbolic representation (e.g., mesh + constraint graph), 那 ReKep-like 方法可能 catch up.

### 5.4 联想到 World Models

这篇 paper 让我强烈联想到 **World Models for Robotics** 这条线:
- GAIA-1 (Wayve): video generation as world model for autonomous driving
- GenRM: video generation as reward model
- UniSim: video generation as universal simulator
- JEPA (LeCun): latent prediction as world model

RIGVid 本质上是在说: **video diffusion model 已经 implicit 学到了 manipulation 的 world dynamics**, 我们只需要 extract 出来. 这是一种 "world model as policy" 的 special case, 其中 policy = video_generator + tracker + retargeter.

如果未来 video model 能生成 physically accurate video (e.g., NVIDIA 的 Cosmos world foundation model), 那 RIGVid 这种方法会变得极其 powerful.

### 5.5 联想到 Latent Action / World Model

最近 Latent Action Pretraining (LAPIE, Ye et al. 2024) 从 video 中 extract latent action, 用来 pretrain robot policy. 与 RIGVid 思路类似 — 都是把 video 当 supervision. 区别: LAPIE 学 latent representation 再用 BC fine-tune, RIGVid 直接用 tracking extract explicit trajectory.

我的想法: 也许 hybrid 方法最好 — 用 latent action 做 pretraining, 用 explicit 6D pose 做 fine-tuning. 这是 **latent + symbolic** 的两阶段.

### 5.6 关于 "Video as Policy" 范式的 scaling law

Paper Fig 7 显示的 trend 让我兴奋: video model 越好, robot 越好. 这意味着 robotics 的 progress 现在 **decoupled from robot data collection**, 只 depends on video generation progress. 如果 Sora-2 / Kling v2 进一步提升 physical plausibility, RIGVid 会免费升级.

这跟 LLM-as-policy (e.g., Code as Policies, VoxPoser) 的 trend 一致: **generative model 进步 -> robot 能力进步**. 但 RIGVid 比这些方法多了一层 — 它 generate 的是 **spatial-temporal pixel tensor**, 不是 text / code, 所以能 capture fine-grained motion dynamics.

---

## 6. Web References

- Project page: https://rigvid-robot.github.io/
- FoundationPose (Wen et al.): https://arxiv.org/abs/2312.08344, code: https://github.com/NVlabs/FoundationPose
- BundleSDF (Wen et al.): https://arxiv.org/abs/2303.05886, code: https://github.com/NVlabs/BundleSDF
- AnyGrasp (Fang et al.): https://github.com/graspnet/anygrasp_html
- Grounding DINO: https://arxiv.org/abs/2303.05499, code: https://github.com/IDEA-Research/GroundingDINO
- SAM-2 (Meta): https://arxiv.org/abs/2408.00714, code: https://github.com/facebookresearch/sam2
- Kling AI: https://klingai.com/
- Sora (OpenAI): https://openai.com/sora
- ReKep (Huang et al.): https://rekep.github.io/, arxiv: https://arxiv.org/abs/2409.01652
- Gen2Act (Bharadhwaj et al.): https://homangab.github.io/gen2act/, arxiv: https://arxiv.org/abs/2409.16283
- Track2Act (Bharadhwaj et al.): https://homangab.github.io/track2act/, arxiv: https://arxiv.org/abs/2405.01527
- AVDC (Ko et al.): https://arxiv.org/abs/2310.08576
- 4D-DPM (Kerr et al.): https://robotseerobotdo.github.io/, arxiv: https://arxiv.org/abs/2409.18121
- Video Depth without Video Models (Ke et al.): https://arxiv.org/abs/2411.06148, code: https://github.com/huggingface/spaces-depth-anything
- VBench++ benchmark: https://arxiv.org/abs/2411.13503, code: https://github.com/Vchitect/VBench
- MegaPose (Labbe et al.): https://megapose6d.github.io/, arxiv: https://arxiv.org/abs/2206.05903
- GPT-4o technical report: https://arxiv.org/abs/2303.08774
- Open X-Embodiment (RT-X): https://robotics-transformer-x.github.io/, arxiv: https://arxiv.org/abs/2310.08864
- MimicPlay (Wang et al.): https://mimic-play.github.io/, arxiv: https://arxiv.org/abs/2302.12422
- Latent Action Pretraining (LAPIE): https://latentactionpretraining.github.io/, arxiv: https://arxiv.org/abs/2410.11758
- Dreamitate (Liang et al.): https://arxiv.org/abs/2406.16862
- ALOHA / Mobile ALOHA: https://tonyzhaozh.github.io/aloha/, arxiv: https://arxiv.org/abs/2304.13705
- VoxPoser (Huang et al.): https://voxposer.github.io/, arxiv: https://arxiv.org/abs/2307.05973

---

## 7. 我的 Final Take

这篇 paper 是 **robotics + generative model** 这条线最近最有 insight 的工作之一. 它的核心贡献不是某个 module (FoundationPose / AnyGrasp / Kling 都是 off-the-shelf), 而是证明了:

1. **Generated video 已经达到 "as good as real demo"** 的 critical threshold
2. **Dense video supervision > sparse VLM abstraction** (打脸 ReKep 这条线)
3. **6D object pose > point tracks / optical flow / feature field** 作为 trajectory representation (打脸 Gen2Act / AVDC / 4D-DPM)
4. **Embodiment-agnostic retargeting** 通过 simple rigid transform 就能做到
5. **Closed-loop SE(3) tracking** 让方法 robust to perturbation

它的 limitation 也清晰: 依赖 mesh, 依赖 rigid grasp, 依赖 depth estimator, 生成 cost 高. 这些都是 **engineering limitation, 不是 conceptual limitation** — 随着周边技术进步都会解决.

如果你 (Karpathy) 在关心 "neural nets learn world model" 这条大 narrative, 那 RIGVid 给了一个 concrete 的 evidence: **video diffusion model 确实学到了足够的 manipulation physics**, 以至于我们可以从它生成的 video 里直接 extract executable robot trajectory. 这跟你在 [Intro to LLM 4 - State of LLM Learning] 里讲到的 "next-token prediction learns to model the world" 是同一个 theme, 只是在 visual-manipulation domain.

接下来我会 watch 的方向:
- Video diffusion model 加入 **3D awareness** (e.g., NVDS, ReconX, WorldConsistentVideoDiffusion) — 会直接帮助 RIGVid
- **Model-free 6D tracker** 速度提升 — 解除 mesh 依赖
- **Deformable / articulated object tracking** — 解除 rigid grasp 假设
- **Bimanual coordinated manipulation** via dual video generation

希望这个回答能 build up 你的 intuition on this work. 如果你想深入某个 module (比如 FoundationPose 内部如何做 render-and-compare, 或者 BundleSDF 的 neural field formulation), 我可以继续展开.
