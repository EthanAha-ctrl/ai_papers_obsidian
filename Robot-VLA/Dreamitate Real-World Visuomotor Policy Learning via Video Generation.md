---
source_pdf: Dreamitate Real-World Visuomotor Policy Learning via Video Generation.pdf
paper_sha256: 8a0e4298529c71bab4ac0c45ee1b6c93063def07ba3f353b81471a5a02c0bd95
processed_at: '2026-08-03T23:27:15-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Dreamitate 用人话说

## 一句话版本

**让 SVD 先 "做梦" 想象一个人怎么用工具完成任务, 然后从梦里把工具的 6D pose 抠出来, 让 robot 照着做。**

Project: <https://dreamitate.cs.columbia.edu/>

---

## 为什么这么搞? 核心痛点

做 robot manipulation 最烦的事就是 **collect data**。Diffusion Policy 这种 SOTA 方法, 训练 data 全靠人 teleop robot, 一个 demo 录起来又慢又贵, 几百条 data 训出来的 model 见到新 object 就傻眼。

但是 internet 上有海量 "人干活" 的 video — 舀汤、扫地、推东西, 这些 video SVD 都看过, 它脑子里已经有 "物理 common sense" 了。问题是, 你怎么把 SVD 脑子里的 prior 抠出来给 robot 用?

两个 naive 方案都有坑:

**方案 A**: 直接让 video model 生成 robot 动作的视频。坑: internet 上 robot video 太少, SVD 根本没见过多少 robot, 生成出来不像。

**方案 B**: 让 video model 生成人手动作的视频, 然后想办法转成 robot action。坑: 人手和 robot gripper 长得完全不一样, embodiment gap 巨大。

Dreamitate 的 insight 特别简单也特别 elegant: **让人和 robot 都拿同一个工具**。工具的 6D pose 就是 action, 人手怎么拿、robot 怎么拿, 在 task level 上完全等价。工具成了 human 和 robot 共享的 "API"。

这个 idea 本质上是说: manipulation task 里真正重要的 embodiment 就只有 end-effector (工具) 那一块, arm 和 torso 就是执行 IK 的肉, 不用学。

---

## Method 具体怎么跑的

整体 pipeline 三步, 公式就一行:

$$a_t = \mathcal{T}(\hat{v}_t), \quad \{\hat{v}_t\} = f_\theta(v_0)$$

变量解释:
- $v_0$: 初始 scene image (实际是 stereo pair, 两个相机拍的)
- $f_\theta$: video generation model (fine-tuned SVD), 参数 $\theta$
- $\hat{v}_t$: 生成的第 $t$ 帧, 也是 stereo pair $(\hat{v}_t^1, \hat{v}_t^2)$
- $\mathcal{T}$: tracking 函数, 从帧里抠出 tool 的 6D pose
- $a_t \in \mathrm{SE}(3)$: tool 相对相机的 6D pose, 直接当 robot action

### Step 1: 录 data

人手持 3D-printed 工具, 在桌上干活, 两个 Intel RealSense D435i 相机 45° 角同步录。**完全不需要 teleop robot**, 拿起工具就开始录, 一个 task 几百条 demo 一个下午就搞完。工具都是 3D 打印的, 有 CAD model, 长得奇形怪状 (勺子、刷子、推杆、夹子)。

相机 setup:
- 两相机相距 660 mm
- 距桌面 760 mm
- 45° 夹角, 一个偏 top view, 一个偏 side view

为什么要 stereo? 因为单目没法恢复 depth, tool 的 z 坐标估不准, robot 就会捅穿桌子或者悬空。

### Step 2: Fine-tune SVD

base model 是 Stable Video Diffusion (SVD, <https://arxiv.org/abs/2311.15127>), 生成 25 帧 video。

**Stereo 的工程 trick**: SVD 输出 25 帧 (奇数), 作者把前 13 帧对应 view 1, 后 12 帧对应 view 2。第一帧因为是 input conditioning image, test 时丢弃。对每帧的 image embedding 根据 "这是哪个视角" 做 modification, 让 model 知道现在在生成哪只眼睛的画面。

Fine-tune objective (Eq. 1):

$$\min_\theta \mathbb{E}_{v \in \mathcal{V}} \left[ \sum_{t=1}^{T} \| \hat{v}_t^1 - v_t^1 \|_2 + \| \hat{v}_t^2 - v_t^2 \|_2 \right]$$

变量:
- $\mathcal{V}$: 训练 video 集合
- $v_t^1, v_t^2$: ground truth 第 $t$ 帧的 view 1 和 view 2
- $\hat{v}_t^1, \hat{v}_t^2$: 模型生成的对应帧
- $\|\cdot\|_2$: L2 norm, 像素重建误差

注意这里 paper 写的是 pixel-level L2, 但 SVD 实际是 latent diffusion, 真实训练 loss 是 VAE latent space 上的 noise prediction MSE。这里简化了表述。

**冻结策略**: 参考 Generative Camera Dolly (<https://arxiv.org/abs/2405.14868>), freeze VAE encoder + decoder, 只 fine-tune spatial 和 temporal attention layers。这是为了保住 SVD 预训练学到的视觉 prior, 不被小数据 fine-tune 破坏。本质上是 LoRA-like 高效微调。

Hyperparameters (Table 4):
- Resolution: $768 \times 448$
- Learning rate: $1e-5$
- Batch size: 4
- Training steps: 16384 (rotation/scooping/sweeping), 17408 (push-shape)
- Clip duration: 2-3 秒
- FPS: 5-6
- Motion score: 200 (SVD 的运动幅度参数)
- Hardware: 4 张 A100, 40 小时 per task
- Inference: 30 denoising steps, CFG = 1.0

每个 task 训一个 model, 没有跨 task 共享。

### Step 3: Track then Act

生成的 stereo video 用 MegaPose (<https://arxiv.org/abs/2212.06870>) 做 6D pose tracking。MegaPose 是 render-and-compare 方法, 给 CAD model 就能估 novel object pose, 慢但准。

**Stereo 3D 恢复细节**:
- 每个相机独立估 tool 6D pose
- translation: 两个相机的 tool center 各自反投影成 3D 射线, 求两射线 closest midpoint (因为噪声, 不严格相交)
- rotation: 两个视角的 rotation 做 quaternion averaging

CAD-aware 的 trick:
- Scooping: 只 track 勺子 handle (头端被 particles 遮)
- Sweeping / Push-Shape: 只 track 工具本体, 不 track handle (被人手遮)

最后 robot 用 impedance control 把离散 6D keyframe 插值成连续 trajectory, **open-loop 执行 12 步, 中间不 replan**。

---

## Experiments 抓重点

### 四个任务

| Task | Train Objects | Demos | Test Objects | Test Trials |
|---|---|---|---|---|
| Rotation | 31 | 371 | 10 | 40 |
| Scooping | 17 bowls + 8 particles | 368 | 8 bowls + 4 particles | 40 |
| Sweeping | 6 particles | 356 | 6 particles | 40 |
| Push-Shape | 26 letters | 727 | 8 shapes | 32 |

**训练和测试的 object set 完全不重叠**, 测试还在不同桌子、不同光线下做, 专门考 generalization。

### 主结果 (Table 2)

| Model | Rotation | Scooping | Sweeping | Push-Shape mIoU | Push-Shape Rot Err |
|---|---|---|---|---|---|
| ACT | 5/40 | 10/40 | 3/40 | 0.527 | 44.1° |
| VQ-BeT | 4/40 | 11/40 | 0/40 | 0.477 | 47.4° |
| Diffusion Policy | 22/40 | 22/40 | 5/40 | 0.550 | 48.2° |
| **Dreamitate** | **37/40** | **34/40** | **37/40** | **0.731** | **8.0°** |

**Sweeping 差距最夸张** (37 vs 5): sweeping 是 multi-modal task (可以扫任意一颗, 多条路径都行)。BC 类方法在 multi-modal action distribution 上会 mode collapse 或者 average 出无意义的 mean trajectory。Video generation 是 conditional sampling, 天然 capture multi-modality — 每次 sample 都能从 valid mode 里选一个。

**Push-Shape rotation error 差距 8° vs 44°**: 这是个 long-horizon 任务, 需要 model 隐式理解 "推一个物体的角, 它会转" 这种物理动力学。SVD 预训练时见过海量 "推东西" video, 这种 forward model 已经在 weights 里了。BC 直接 regress action, 学不到这种 forward model。

**Rotation 上 Diffusion Policy 也不差 (55%)**: Rotation 是 short-horizon 单一动作, BC 局限性还不大, 但 Dreamitate 还是 92.5%, 说明在 grasp point 选择上 video prior 帮了大忙。

### Ablation (Table 3) — 最 informative

| Variant | Rotation Success |
|---|---|
| Full Model | 37/40 |
| w/o Pretraining | 18/40 |
| Stereo In, Mono Out (SIMO) | 30/40 |
| Mono In, Mono Out (MIMO) | 14/40 |

**关键 insight**:

- **Pretraining 贡献约 50% 性能** (37 → 18)。没 pretrain, SVD 输出模糊 tool 变形, 但 stereo projection 还能把扭曲 pose 投回 3D 救一部分。说明 internet video prior 真的进了 model。

- **SIMO (30) vs MIMO (14)**: 单视角输入就让 video model disambiguate depth 失败, 即使输出双视角帮助也有限。**Stereo input 提供的几何信号比 output 形式更重要**。

这跟 human vision 类比: 双眼立体视在 fine motor task (穿针) 上至关重要, 这里似乎也对应了 robotics 版的 binocular cue 必要性。

### Data Scaling Curve (Fig. 8)

把训练 data 降到 1/3:
- Diffusion Policy: success 大幅下降
- Dreamitate: 几乎持平

这条曲线形状很经典 — pre-trained foundation model 在 low-data regime 优势最大, 因为 prior 替代了 missing data。跟 R3M (<https://arxiv.org/abs/2203.12601>)、Voltron 在低数据 manipulation 上的曲线形状一致。

---

## 为什么 work — 直觉上的解释

### 1. SVD 见过几百万个 "人干活" 视频

它脑子里已经 encode 了:
- 物体不能穿模
- 勺子要伸进碗里才能舀
- 推一个物体的角它就转
- 障碍物要绕开

这些 prior 在 BC 里完全缺失。BC 只见过 371 个 demo, SVD 见过 10M+ video。**数量级差异决定 generalization 差异**。

### 2. Video 是可解释的中间 representation

传统 end-to-end BC 是 black box, 失败了不知道为啥。Dreamitate 中间产物是 video, 人可以直接看 "model 想做啥", 错了能 debug。这对 HRI 安全性是巨大 feature。

### 3. Tool-as-API 的 elegance

人和 robot 都拿同一个工具, 工具 6D pose 就是共享 action space:
- 人手 vs. robot gripper 在 task level 等价
- 不同 robot (xArm 7 vs UR5) 共享同一个 policy
- 数据收集无需 teleop

代价: 每个任务需要一个 CAD-known 3D-printed tool, 通用性弱一点。

### 4. Action 放在 "哪里 prior 多" 哪里就 work

Diffusion Policy 在 action space (12 维 SE(3) + gripper) 做 diffusion, prior 来自 ResNet-18 ImageNet pretrain (1M image)。
Dreamitate 在 pixel space ($768 \times 448 \times 3 \times 25$) 做 diffusion, prior 来自 SVD web-scale video (100M+ clip)。

**Prior 的数量级差异决定了泛化能力的差异**。这是 "where to put the prior" 的设计选择 — 你把 action 编进哪个 space, 就能 leverage 那个 space 的 pretrain prior。

---

## Limitations

Paper 自陈:
1. **需要 visually trackable tool**: 透明、反光、太小的 object tracking 会 fail
2. **需要 CAD model**: 没法用任意现成工具
3. **Task-specific model**: 每 task fine-tune 一次, 没跨任务 text conditioning
4. **只能 rigid tool**: 没法做 force control, 没法用 compliant tool (sponge, cloth)
5. **慢得离谱**: A100 上 33.5s 生成 video + 7.5s tracking, 一个 12-step action 要 40 秒。完全 offline planning, realtime 不可能

我补充几个更深的:

- **Open-loop 12 步很脆弱**: 中间 object 滑了、掉了, 没法纠正。Fix 应该是 receding horizon + partial video inpainting, 但延迟更爆炸。

- **Tool pose 跨帧 jitter**: video diffusion 不天然 enforce temporal pose consistency, 即使每帧 pose 还原, 跨帧 trajectory 可能跳。Paper 用 impedance control 平滑掉一部分, 但本质问题没解。

- **6D pose 对称性歧义**: 圆柱刷子绕轴对称, MegaPose rotation 估计有歧义。

- **Stereo 几何一致性不严格**: SVD 在 latent space 生成, stereo pair 的几何一致性没硬约束。Paper 没测 triangulation 误差有多大。

- **25 帧硬上限**: 长任务必须 chunk, chunk 之间没平滑衔接机制。Push-Shape 本质上是分段规划。

---

## 相关联想

### 同类 "video-as-policy" 工作

- **Sora as world simulator** (<https://openai.com/research/video-generation-models-as-world-simulators>): 同样把 video generation 当 physical world prior, 但 Sora 没接 robot。Dreamitate 是 Sora 思想在 manipulation 上的可落地版本。

- **Du et al. text-guided video policy** (<https://arxiv.org/abs/2302.14135>): 用 Imagen Video 生成 robot video, 再用 inverse dynamics 模型估 action。Du 生成 robot video 数据稀少, Dreamitate 生成 human+tool video 数据丰富。

- **UniSim** (<https://arxiv.org/abs/2310.06680>): video model 当 closed-loop sensor simulator, "world model" 路线, 与 Dreamitate 的 "policy" 路线正交。

- **Video Language Planning** (<https://arxiv.org/abs/2310.12925>): video model 当 long-horizon planner, 每节点再由 short-horizon controller执行。

- **MimicPlay** (<https://arxiv.org/abs/2310.17565>): human demo → robot action, 但中间用 latent subgoal 而非 pixel-level video。Dreamitate 更 "硬核", 直接在 pixel space 工作。

- **Black et al. image-editing policy** (<https://arxiv.org/abs/2310.10639>): 用 image editing diffusion model 当 policy, "language/image as action" 范式的早期探索。

### Tracker 替代选项

MegaPose 慢 (7.5s/frame)。新一代:
- **FoundationPose** (<https://arxiv.org/abs/2311.13619>, CVPR 2024): 神经网络 + render-and-compare 混合, 比 MegaPose 快很多, 支持 novel object
- **BundleSDF** (RSS 2023): real-time joint tracking + neural SDF
- **CoTracker** (<https://arxiv.org/abs/2307.07635>): dense pixel tracking, 不输出 6D pose
- **TAPIP** (2024): 长序列 feature tracking

换 FoundationPose 整条 pipeline 可能进 5-10 秒级, 配 receding horizon 就接近可用。

### Action representation 演化路径

宏观视角, robot action representation 经历:
1. Joint angle / torque (low-level)
2. End-effector pose (BC, ACT, Diffusion Policy)
3. Latent action (VQ-BeT, ACT 的 latent)
4. **Pixel / video as action** (Dreamitate, RT-2 的 "language as action")

每往上一层, 信号更 "人类可读", 能 leverage 的 internet prior 越多, 但控制精度更依赖下游 perception pipeline。这是 representation trade-off: 抽象层次越高, prior 越强, "解码" 成精确控制越难。Dreamitate 的 stereo + CAD + MegaPose 就是它解码的代价。

### 长期演化猜想

1. **统一 multi-task model**: text prompt condition, 一个 model 处理所有 tool 和 task。需要解 tool 切换的 tracking 问题。

2. **Closed-loop video planning**: 每 N 步 re-observe 重新 generate, 配合 video inpainting 做 partial conditioning。Sora 的长视频生成技术里有苗头。

3. **Force-aware extension**: video model + force-torque 数据, 用 multimodal VLM 推理接触。最难, 因为 video model 本质 geometry-only。

4. **End-to-end co-training**: video model 和 tracking model 联合训练, 让生成 video 本身就 "track-friendly" (tool 区域高分辨率、强纹理)。目前是 sequential, tracking 鲁棒性全靠 MegaPose 自己。

5. **On-robot video data + internet human video joint pretraining**: 解 robot-specific visual distribution mismatch。

---

## 一句话再总结

Dreamitate 把 visuomotor policy 拆成 **"video 生成 → 6D tool tracking → robot IK"** 三段。SVD 的 internet prior 弥补 BC 在 small-data regime 的 generalization 不足, known CAD tool 桥接 human-robot embodiment gap, 4 个真机任务 success rate 从 BC baseline 的 12.5%-55% 提升到 85%-92.5%, 但代价是 open-loop、task-specific、慢 (40s/inference)。

最值得记住的 contribution 是 **"video as interpretable action representation"** — action 不再是 latent 向量, 是人可以直接观看、可以 critique 的视频。这对未来 HRI 安全性、对 RL reward shaping (video 当对比基线)、对 human-in-the-loop intervention 都打开了新方向。

---

Reference:
- Dreamitate project: <https://dreamitate.cs.columbia.edu/>
- SVD: <https://arxiv.org/abs/2311.15127>
- MegaPose: <https://arxiv.org/abs/2212.06870>
- Diffusion Policy: <https://arxiv.org/abs/2303.04137>
- ACT / ALOHA: <https://arxiv.org/abs/2304.13705>
- VQ-BeT: <https://arxiv.org/abs/2403.03181>
- MimicPlay: <https://arxiv.org/abs/2310.17565>
- Generative Camera Dolly: <https://arxiv.org/abs/2405.14868>
- FoundationPose: <https://arxiv.org/abs/2311.13619>
- Du et al. text-guided video policy: <https://arxiv.org/abs/2302.14135>
- UniSim: <https://arxiv.org/abs/2310.06680>
- Sora as world simulator: <https://openai.com/research/video-generation-models-as-world-simulators>
- Black et al. image-editing policy: <https://arxiv.org/abs/2310.10639>
- R3M: <https://arxiv.org/abs/2203.12601>
- CoTracker: <https://arxiv.org/abs/2307.07635>

---

# Dreamitate: 通过 Video Generation 学习 Real-World Visuomotor Policy

## 一、核心 idea 与 motivation

这篇 paper 来自 Carl Vondrick 组 (Columbia) 联合 Toyota Research Institute (Achal Dave, Pavel Tokmakov) 与 Stanford (Shuran Song)。核心 idea 可以一句话概括：**把 visuomotor policy 重新表述成 "先 video generation, 再 3D tracking" 的两阶段过程**, robot action 不是直接由神经网络 regress 出来的, 而是从生成的 human-tool 视频 "提炼" 出 tool 的 6D trajectory, 再交给 robot 执行。

为什么这么做? 三个 motivation:

1. **Behavior Cloning 的 data scaling 困境**：传统 BC 需要大量 teleop data, 每 demo 都要操作 robot, 成本高且 visual diversity 有限。SVD 这类 video model 预训练在 internet-scale video 上, 已经 encode 了海量 human manipulation prior。

2. **Human video vs. Robot video 的两难**：
   - 生成 human 行为: diversity 高, 但存在 embodiment gap (人手 vs. robot gripper)。
   - 直接生成 robot 行为 (如 UniSim, Du et al. 2023): 无 gap, 但 robot video data 比 human video 少了几个数量级。

3. **Tool 是最小可行 embodiment**：作者的关键 insight 是, manipulation 中真正重要的 "embodiment" 其实是 end-effector 与 object 的 contact interaction; 剩余的 arm/torso 完全可以用 inverse kinematics 重建。如果一个 known CAD model 的 rigid tool 同时被 human 和 robot 夹持, 则 tool 的 6D pose 就是 robot 要 follow 的 SE(3) action, human hand 与 robot gripper 在 task level 上等价。

这其实是一种 **task-specific tool-as-interface** 的设计, 类似 MimicPlay (Yin et al., RSS 2024) 把人手轨迹作为 robot 的 latent subgoal, 但 Dreamitate 把它推到了 pixel-level video generation + 3D tracking 的程度。

Paper link: <https://dreamitate.cs.columbia.edu/>
arXiv: <https://arxiv.org/abs/2406.16862>

---

## 二、Method 解析

### 2.1 整体公式

给定初始观测 $v_0$, 目标是输出 SE(3) action 序列:

$$
a_t = \mathcal{T}(\hat{v}_t), \quad \{\hat{v}_t\}_{t=1..T} = f_\theta(v_0)
$$

变量含义:
- $v_0$: 初始 scene image (实际是 stereo pair, 因为要恢复 3D)。
- $f_\theta(\cdot)$: video generative model, 参数 $\theta$, 初始化自 Stable Video Diffusion 的预训练权重。
- $\hat{v}_t$: 生成的第 $t$ 帧 (stereo pair, 即 $(\hat{v}_t^1, \hat{v}_t^2)$)。
- $\mathcal{T}(\cdot)$: tool tracking 函数, 把 video frame 中的 tool 提取出 6D pose, 是已知 CAD 的 model-based tracking。
- $a_t \in \mathrm{SE}(3)$: tool 相对 camera 的 6D pose, 直接作为 robot end-effector target。

注意一个 subtle 点: action space 不是 raw joint torque 或 joint position, 而是 end-effector pose, 然后由 robot controller (impedance control) 做 inverse kinematics 与轨迹 smoothing。这让 method 对不同 robot hardware (xArm 7 / UR5) 是 agnostic 的。

### 2.2 Video Generation 模块

**Base model**: Stable Video Diffusion (SVD, Blattmann et al. 2023, <https://arxiv.org/abs/2311.15127>), 25 frames per clip。

**Stereo 适配**: 这是 paper 的一个工程 trick。SVD 默认输出 25 帧 (奇数), 作者把前 13 帧对应 view 1, 后 12 帧对应 view 2, 第一帧因为是 input image 本身, 在 test 时丢弃。对 per-frame image embedding 根据 viewing angle 做 conditioning modification, 让 model 知道 "现在生成的是哪只眼睛看到的"。

**Fine-tuning objective** (Eq. 1):

$$
\min_\theta \mathbb{E}_{v \in \mathcal{V}} \left[ \sum_{t=1}^{T} \| (\hat{v}_t^1 - v_t^1) \|_2 + \| (\hat{v}_t^2 - v_t^2) \|_2 \right]
$$

注意 Eq. 1 写的是 $L_2$ 像素 reconstruction, 但 SVD 是 latent diffusion model, 实际训练 loss 是在 VAE latent space 上的 noise prediction MSE (与 SVD 原文一致)。这里的 $L_2$ 写法是为了 paper 表述简洁, 把 encoder + scheduler 隐含掉了。

**冻结策略**: 参考 Generative Camera Dolly (Van Hoorick et al. 2024, <https://arxiv.org/abs/2405.14868>) 的做法, freeze encoder 和 decoder (即 VAE), 只 fine-tune spatial/temporal attention layers。这是 LoRA-style 的高效微调, 避免破坏 pretrain prior。

**Hyperparameters** (Table 4):
- Resolution: $768 \times 448$。
- Learning rate: $1e-5$。
- Batch size: 4。
- Steps: 16384 (rotation/scooping/sweeping), 17408 (push-shape)。
- Clip duration: 2–3 秒, fps 5–6, motion score 200。
- 4 张 A100, 40 小时训练 (每 task 一个 model)。
- Inference: 30 denoising steps, CFG = 1.0 (无 classifier-free guidance 加强, 因为 task 已经被 fine-tune 进去)。

### 2.3 Track then Act 模块

**Tracker**: MegaPose (Labbe et al. 2022, <https://arxiv.org/abs/2212.06870>), 一种 render-and-compare 的 6D pose estimator, 对 novel object 也能 work, 只要给 CAD model。

**Stereo 3D 恢复**:
- 每个相机独立估计 tool 的 6D pose。
- translation: 把两个相机估计的 tool center 各自反投影成 3D 空间中的射线 (camera ray), 求 两条射线的 closest midpoint (因为噪声, 不严格相交)。
- rotation: 两个视角的 rotation 用 quaternion averaging 合并。

**CAD-aware simplification**:
- Scooping 任务: 只 track 勺子的 handle (头端会被 particles occlude)。
- Sweeping / Push-Shape: 只 track 工具本体, 不 track handle (handle 被人手遮挡)。

**Robot 执行**: 用 impedance control 做 trajectory smoothing, 把离散 6D pose keyframes 插值成连续运动。整条 pipeline 是 open-loop (12 步 horizon), 没有 replan。

### 2.4 Data collection

- 顶部 + 侧面双 Intel RealSense D435i, 相距 660 mm, 45° 夹角, 距桌面 760 mm。
- Human demonstrator 直接手持 3D-printed tool 做 demo, 不需要 teleop, 极大降低数据收集成本。
- Tool 设计: 每个任务一个独特 tool (gripper, scoop, brush, push rod), 都有 CAD。

这是对比 Diffusion Policy 训练 setup 的一个重要优势: Diffusion Policy 训练数据必须包含 robot 真实轨迹 (teleop), Dreamitate 只需要人手持工具的 RGB 录像。

---

## 三、Experiments 详解

### 3.1 四个任务 (Table 1)

| Task | Train Objects | Demos | Test Objects | Test Trials |
|---|---|---|---|---|
| Rotation | 31 | 371 | 10 | 40 |
| Scooping | 17 bowls + 8 particles | 368 | 8 bowls + 4 particles | 40 |
| Sweeping | 6 particles | 356 | 6 particles | 40 |
| Push-Shape | 26 letters | 727 | 8 shapes (digits/polygons) | 32 |

强调一点: 训练和测试的 object set 完全不重叠, 测试在不同桌子和光照条件下做, 这要求 strong generalization。

### 3.2 主结果 (Table 2)

| Model | Rotation | Scooping | Sweeping | Push-Shape (mIoU) | Push-Shape (Rot Err) |
|---|---|---|---|---|---|
| ACT | 5/40 (12.5%) | 10/40 (25%) | 3/40 (7.5%) | 0.527 | 44.1° |
| VQ-BeT | 4/40 (10%) | 11/40 (27.5%) | 0/40 (0%) | 0.477 | 47.4° |
| Diffusion Policy | 22/40 (55%) | 22/40 (55%) | 5/40 (12.5%) | 0.550 | 48.2° |
| **Dreamitate** | **37/40 (92.5%)** | **34/40 (85%)** | **37/40 (92.5%)** | **0.731** | **8.0°** |

几个 intuition:

1. **Sweeping 任务差距最大** (37 vs 5): sweeping 是 multi-modal 的 (可以扫任意一个 particle, 多条路径), BC 类方法在 multi-modal action distribution 上容易 mode collapse 或 average 出无意义轨迹。Video generation 是 conditional sampling, 天然能 capture multi-modality。

2. **Push-Shape 的 rotation error 差距 8° vs 44°**: 这是 long-horizon 任务, 需要 push 之后预测 object 的 sliding + rotation 动力学。Video model 通过 pretrain 学到了大量物理常识 (推东西会转), 而 BC 直接 regress action 学不到这种 forward model。

3. **Rotation 上 Diffusion Policy 也不差**: Rotation 是 short-horizon 单一动作, BC 的局限性还不大; 但 Dreamitate 仍领先 92.5% vs 55%, 说明它在 grasp point 选择上更稳。

### 3.3 Ablation (Table 3) — 这是最 informative 的部分

| Variant | Rotation |
|---|---|
| Full Model | 37/40 |
| w/o Pretraining | 18/40 |
| Stereo In Monocular Out (SIMO) | 30/40 |
| Monocular In Monocular Out (MIMO) | 14/40 |

**关键 insight**:

- **Pretraining 贡献 ~50% 性能** (37 → 18)。没 pretrain 的 SVD 输出模糊, tool 几何变形, 但 stereo projection 还能把扭曲的 tool pose 投回 3D, 说明 tracking 鲁棒性也救了一部分。
  
- **Stereo 双输入单输出 (SIMO) 30/40 vs 全 monocular (MIMO) 14/40**: 单视角输入就让 video model 无法 disambiguate depth, 即使输出两视角也帮助有限。Stereo input 本身提供的几何信号比 output 形式更重要。

可以对比一下: in human vision, 双眼立体视觉在 fine motor task (比如 threading needle) 上至关重要, 这里似乎也对应了一个 robotics 版的 binocular cue 必要性。

### 3.4 Scaling Curve (Fig. 8)

把训练数据降到 1/3:
- Diffusion Policy: success 大幅下降
- Dreamitate: 几乎保持

这个曲线的形状很经典 — pre-trained foundation model 在 low-data regime 优势最大, 因为 prior 替代了 missing data。这与 R3M (Nair et al. 2022, <https://arxiv.org/abs/2203.12601>)、Voltron (Karamcheti et al. 2023) 在低数据量 manipulation learning 上观察到的曲线形状一致。

---

## 四、为什么这个 approach work — intuition

### 4.1 Video prior 提供了 "物理 common sense"

SVD 见过几百万个 "人扫东西" "人舀水" "人推东西" 的视频, 它隐式学到了:
- 物体不能穿模
- 勺子要伸到碗里才能舀
- 推一个角会让物体转
- 障碍物要绕开

这些 prior 在 BC 里完全缺失。BC 只见过 371 个 demo, 而 SVD 见过 10M+ 视频。

### 4.2 Action 通过 video "显式化" 成可解释 representation

传统 end-to-end BC 是 black box, 失败不可调试。Dreamitate 中间产物是 video, 人类可以直接看 model "想做啥", 这是 HRI 场景的关键安全特性。这让我想到 Robotic Control via Pretrained Image Editing (Black et al. 2023, <https://arxiv.org/abs/2310.10639>) 和 RT-2 的 "language as action" 范式 — 把 action 编进一个人类可读的 modal space (language / video), 让模型 prior 和可解释性都受益。

### 4.3 Tool 是 "task-specific action interface"

这是我觉得最 elegant 的设计。Dreamitate 不要求 model 直接输出 robot joint, 而是输出 tool pose。这意味着:
- 人手和 robot gripper 共享同一个 "action space" (tool pose)
- 不同 robot (xArm 7 vs UR5) 共享同一个 policy
- 数据收集无需 teleop

代价是每个任务需要一个 CAD-known tool, 通用性弱一些。

---

## 五、Limitations 与开放问题

Paper 自陈的 limitations:

1. **需要 visually trackable tool**: 透明、反光、过小 object 会 fail。MegaPose 在小 object 上 pose 估计不稳。
2. **需要 CAD model**: 没法直接用任意 houseware 工具。
3. **Task-specific model**: 每个 task fine-tune 一次, 没有跨任务的 text conditioning。
4. **只能 rigid tool**: 没法 force control, 没法 compliant tool (sponge, cloth)。
5. **Real-time 不可行**: A100 上 33.5s/video + 7.5s/tracking, 一共 ~40s 才能生成一次 12-step action, 完全是 offline planning。

我会补充几个更深的开放问题:

- **Open-loop 的脆弱性**: 12 步 horizon 不 replan, 中间出现意外 (object 滑了) 没办法纠正。一个 fix 是 iterative replan + partial video conditioning (类似 diffusion policy 的 receding horizon), 但会进一步增加延迟。
- **Tool pose 在生成 video 中的 jitter**: 即使单帧 pose 还原, 跨帧的 trajectory 可能跳变, robot 跟着抖动。Paper 用 impedance control 平滑掉了一部分, 但本质上 video diffusion 不天然 enforce temporal pose consistency。
- **6D pose tracking 的 ambiguity**: 当 tool 有对称性 (e.g. 圆柱刷子绕轴对称), MegaPose 的 rotation 估计会有歧义, 影响 action 精度。
- **SVD 的 25 帧硬限制**: 长任务必须 chunk, 但 chunk 之间没有平滑衔接机制, 这导致 Push-Shape 之类 long-horizon 任务本质上还是分段规划。
- **3D 一致性 vs video prior**: 模型在 latent space 生成, stereo 一致性不严格保证。MIMO ablation 显示 input 必须是 stereo, 但 paper 没测量 stereo 输出几何上的一致性有多强 (是否两条射线的 triangulation 误差很小)。

---

## 六、相关联想与延伸

### 6.1 同类 "video-as-policy" 工作

- **Sora as world simulator** (OpenAI 2024, <https://openai.com/research/video-generation-models-as-world-simulators>): 同样把 video generation 当 physical world prior, 但 Sora 没接 robot。Dreamitate 可以看作 Sora 思想在 manipulation 上的可落地版本。

- **Learning Universal Policies via Text-Guided Video Generation** (Du et al. 2023, <https://arxiv.org/abs/2302.14135>): 用 Imagen Video 生成 robot video, 然后 inverse dynamics 模型估 action。区别: Du 生成 robot video, 数据稀少; Dreamitate 生成 human+tool video, 数据丰富。

- **UniSim** (Yang et al. 2023, <https://arxiv.org/abs/2310.06680>): 用 video model 做 closed-loop sensor simulation, 是 "world model" 路线, 与 Dreamitate 的 "policy" 路线正交。

- **Video Language Planning** (Du et al. 2023, <https://arxiv.org/abs/2310.12925>): video model 当 long-horizon planner, 每个节点再由 short-horizon controller 执行, 与 Dreamitate 的单层结构不同。

- **MimicPlay** (Yin et al. RSS 2024, <https://arxiv.org/abs/2310.17565>): 也是 human demo → robot action, 但中间用 latent subgoal 而非 pixel-level video。Dreamitate 更 "硬核", 直接在 pixel space 工作。

- **GenSim** (Kim et al. 2023): language-conditioned 仿真生成, 用于 multi-task policy pretraining, 数据来源是仿真而非 internet。

### 6.2 Tracker 替代选项

MegaPose 是 render-and-compare, 慢 (7.5s/frame)。新一代选择:
- **FoundationPose** (Wen et al. CVPR 2024, <https://arxiv.org/abs/2311.13619>): 神经网络 + render-and-compare 混合, 比 MegaPose 快很多, 而且支持 novel object。
- **BundleSDF** (Wen et al. RSS 2023): real-time joint tracking + neural SDF。
- **CoTracker** (Doersch et al. 2023, <https://arxiv.org/abs/2307.07635>): dense pixel tracking, 但不输出 6D pose。
- **TAPIP** (Zheng et al. 2024): 长序列 feature tracking。

如果换成 FoundationPose, 整条 pipeline 实时性可能进入 5–10 秒级, 配合 receding horizon 就接近可用。

### 6.3 Action representation 的演化路径

这是更宏观的视角。Robot action representation 经历:
1. Joint angle / torque (low-level)
2. End-effector pose (BC, ACT, Diffusion Policy)
3. Latent action (VQ-BeT, ACT 的 latent)
4. **Pixel/video as action** (Dreamitate, RT-2 用 language as action)

每往上一层, 信号更 "人类可读", 但控制精度更依赖下游的 perception pipeline。这是一个 representation trade-off: 抽象层次越高, 越能 leverage internet prior, 但 "解码" 成精确控制越难。Dreamitate 的 stereo + CAD + MegaPose 就是它解码的代价。

### 6.4 与 Diffusion Policy 的本质对比

Diffusion Policy 也是 diffusion, 也 handle multi-modal action distribution。区别:
- Diffusion Policy 在 action space 做 diffusion (12 维 SE(3) + gripper)。
- Dreamitate 在 pixel space 做 diffusion (768×448×3×25 frame)。

Pixel space diffusion 的 advantage: 可以 leverage SVD 的海量视觉 prior。Disadvantage: action 必须从像素里 "再提取", 引入 tracking 误差。

这其实是一个 "where to put the prior" 的设计选择。Diffusion Policy 的 prior 来自 ResNet-18 visual encoder (ImageNet pretrain, 1M image); Dreamitate 的 prior 来自 SVD (web-scale video, 100M+ clip)。数量级差异决定了泛化能力的差异。

### 6.5 长期演进猜想

Dreamitate 这种 "video model 作为 policy backbone" 的范式, 可能演化方向:

1. **统一 multi-task model**: 用 text prompt condition, 一个 model 处理所有 tool 和 task。需要解决 tool 切换的 tracking 问题。
2. **Closed-loop video planning**: 每 N 步 re-observe 并重新 generate, 配合 video inpainting 做 partial video conditioning。这一点在 Sora 的长视频生成技术里有苗头。
3. **Force-aware extension**: 把 video model 与 force-torque 数据结合, 用 multimodal VLM 推理接触。这一步最难, 因为 video model 本质是 geometry-only。
4. **End-to-end co-training**: 把 video model 和 tracking model 联合训练, 让生成的 video 本身就 "track-friendly" (tool 区域高分辨率、纹理强)。目前 Dreamitate 是 sequential, tracking 鲁棒性完全靠 MegaPose 自己。
5. **On-robot video data 加 internet human video 的 joint pretraining**: 解决 robot-specific visual distribution mismatch。

---

## 七、一句话总结

Dreamitate 把 visuomotor policy 拆成 "video 生成 → 6D tool tracking → robot IK" 三段, 通过 SVD 的 internet-scale prior 弥补 BC 在 small-data regime 下的 generalization 不足, 用 known CAD tool 桥接 human-robot embodiment gap, 在 4 个真机任务上把 success rate 从 BC baseline 的 12.5%–55% 提升到 85%–92.5%, 但代价是 open-loop、task-specific、慢 (40s/inference)。

它最值得记住的 contribution 是 **"video as interpretable action representation"** 这个范式 — action 不再是 latent 向量, 而是人类可以直接观看、可以 critique 的视频。这对未来 HRI 安全性、对 RL 的 reward shaping (video 可以作为对比基线)、对 human-in-the-loop intervention 都打开了新方向。

Reference:
- Project page: <https://dreamitate.cs.columbia.edu/>
- SVD: <https://arxiv.org/abs/2311.15127>
- MegaPose: <https://arxiv.org/abs/2212.06870>
- Diffusion Policy: <https://arxiv.org/abs/2303.04137>
- ACT / ALOHA: <https://arxiv.org/abs/2304.13705>
- VQ-BeT: <https://arxiv.org/abs/2403.03181>
- MimicPlay: <https://arxiv.org/abs/2310.17565>
- Generative Camera Dolly: <https://arxiv.org/abs/2405.14868>
- FoundationPose: <https://arxiv.org/abs/2311.13619>
- Du et al. text-guided video policy: <https://arxiv.org/abs/2302.14135>
- UniSim: <https://arxiv.org/abs/2310.06680>
- Sora as world simulator: <https://openai.com/research/video-generation-models-as-world-simulators>
- Black et al. image-editing policy: <https://arxiv.org/abs/2310.10639>
