---
source_pdf: GRAIL Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors.pdf
paper_sha256: 6e125c17f5fc148c1f3c4cbb2bbcefbc889dedfc9881e40042b9f21554ad1cae
processed_at: '2026-08-19T09:51:06-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 GRAIL

好，咱换个方式聊。抛开那些花里胡哨的术语，我用最直白的话给你捋一遍这事儿。

---

## 一句话说清楚这是个啥

**人形机器人想学会又走又抓东西（边走边拿杯子、爬楼梯同时搬箱子），但教它做事的数据太难弄了。**

老办法是让真人遥控机器人去抓一遍录下来，或者给人穿 mocap 衣服录动作。每换一个新物体、换一个新场景就得重新来一遍，累死人也贵死你。

GRAIL 说：**咱别用真人了。我用 AI 生成一段"人跟东西交互"的视频，再从视频里反推出机器人能用的动作数据。全流程在电脑里跑，不用碰真机器人，直到最后部署那一下才上真机。**

就这么简单。

---

## 那为啥不直接用现成的视频？

你看 YouTube 上有的是人搬箱子爬楼梯的视频。直接拿来用不就完了？

问题是：**单看一段视频，你不知道相机在哪儿、人离镜头多远、箱子多大、手到底有没有碰到箱子。** 这就是 paper 里反复说的 "depth ambiguity" —— 你看到一个手指头在画面里挨着箱子，但实际在 3D 空间里手指头可能离箱子还有 20 厘米，也可能已经穿进箱子里面了。

所以直接从野视频重建 4D 动作，就跟闭着眼听声辨位差不多——方向大概对，距离全靠猜。

---

## GRAIL 的招：先搭台再唱戏

GRAIL 干的事儿是**倒过来**。

它不是"先有视频再猜 3D"，而是"先造好一个完全已知的 3D 场景，再让 AI 在这个场景里演一段视频"。

你想想这个区别：

- **老办法**：拿着一个黑盒视频，里面啥都不知道，你要猜相机、猜尺寸、猜人的体型、猜物体形状……每一个都是未知数。
- **GRAIL**：相机参数 $C_K$（intrinsics）、$C_E$（extrinsics）是你自己设的，object 的 3D mesh 是你下载的，人的体型是 pre-fitted 到 Unitree G1 比例的，场景的 depth map 是你渲染出来的。这些**全是已知量**。

然后你让 VFM（Kling，快手的视频生成模型）在这个场景里生成一段"人去拿桌上的苹果"的视频。VFM 只负责一件事：**告诉你"怎么拿"**——先伸手、再握住、然后抬起来。这个行为逻辑是 VFM 提供的 prior。至于几何细节，VFM 管不了也不需要它管。

---

## 视频有了，怎么变成机器人动作？

这段是 paper 里最 technical 的部分，但 intuition 其实很朴素。

VFM 给你的视频虽然画面上看着流畅，但 3D 信息里全是毛病。手穿模进物体里了，或者悬浮在物体表面上方 10 厘米处。你直接拿这个去驱动机器人，机器人要么把自己手指头怼进苹果里面，要么在空中瞎抓一通。

所以 GRAIL 做了一个 **joint optimization**——同时优化人和物体的轨迹，让它们满足一堆约束条件：

### 三个核心约束（用大白话）

1. **$L_{\mathrm{kp}}$（Keypoint Alignment）**：你从视频里检测到手在画面的某个位置。你优化出来的 3D 人，投影回画面，手得在差不多的位置。这保证你的动作跟视频"看起来一致"。

2. **$L_{\mathrm{depth}}$（Depth Alignment）**：这是最关键的一步。你用 MoGe-2 从视频估一个 depth map，然后跟你一开始渲染场景时就知道的 ground truth 背景 depth 对齐。对齐之后你就有了 metric scale——你知道"手离桌面是 15 厘米"而不是"大概有个 15 厘米吧"。然后你把人的 mesh 顶点和物体 mesh 顶点跟这个 depth point cloud 对齐。

   公式是 Chamfer Distance：
   $$L_{\mathrm{depth}} = \frac{1}{T} \sum_{t=1}^{T} \mathcal{CD}(V_t^{\mathcal{H}, \mathrm{vis}}, \mathbf{P}_t^{\mathcal{H}}) + \mathcal{CD}(V_t^{\mathcal{O}, \mathrm{vis}}, \mathbf{P}_t^{\mathcal{O}})$$
   - $V_t^{\mathcal{H}, \mathrm{vis}}$：人的 mesh 上可见的顶点
   - $\mathbf{P}_t^{\mathcal{H}}$：从 depth map 反投影回来的 point cloud
   - $\mathcal{CD}$：双向 Chamfer Distance，就是两组点之间互相找最近邻然后算距离

3. **$L_{\mathrm{cont}}$（Contact Alignment）**：你用 VLM（GPT-4o）去看视频帧，判断"这个时刻左手在碰苹果"。然后你只在这些 contact frame 上，强制人的手部顶点和苹果的顶点在 depth 方向上靠拢。

   $$L_{\mathrm{cont}} = \frac{1}{|\mathcal{T}_c|} \sum_{t \in \mathcal{T}_c} \mathcal{CD}_z(V_t^{\mathcal{H}, \mathrm{cont}}, V_t^{\mathcal{O}, \mathrm{cont}})$$
   - $\mathcal{T}_c$：检测到 contact 的帧集合
   - $\mathcal{CD}_z$：只在 z 轴（viewing direction）算 Chamfer Distance
   - $V_t^{\mathcal{H}, \mathrm{cont}}$：SMPL-X 上 contact 区域的顶点
   - $V_t^{\mathcal{O}, \mathrm{cont}}$：object 上投影到 contact 区域内的顶点

**直觉**：前两个 loss 管"看起来对"，这个 loss 管"碰得对"。它只在检测到接触的帧上起作用，只管 depth 方向，不碰 2D 投影，所以不会跟 keypoint loss 打架。

---

## 4D 动作有了，怎么教机器人？

现在你有了一条干净的、metric scale 的、手和物体接触合理的 4D 轨迹。接下来要把它变成机器人的 joint command。

先用 GMR 把 SMPL-X 的人体动作 retarget 到 Unitree G1 的 joint space 上。G1 有 29 个 body DoF + 每只手 7 个 finger DoF = 43 DoF。

然后的问题是：**怎么让机器人 track 这个 reference motion？**

### 基座：SONIC

他们用了一个已有的 pretrained whole-body controller 叫 SONIC（也是 NVIDIA 的工作）。SONIC 干的事是：输入一个 kinematic target（关节角度），encode 成一个 discrete latent token $z_t$，再 decode 成 joint action。用的是一个叫 Finite Scalar Quantization（FSQ）的技术把连续值量化成离散 token。

### 两种任务两种策略

**Manipulation（抓东西）**：你不想破坏 SONIC 原本的 locomotion 能力，所以 encoder、quantizer、decoder 全部 freeze。你只训一个小 adaptor $\pi_\phi$（3 层 MLP），它输出两个东西：

1. 一个 64 维的 latent residual $\Delta z_t$
2. 一个 2 维的 hand grasp primitive（左手开/合，右手开/合）

然后把 $\Delta z_t$ 乘以 $\lambda=0.1$ 加到 SONIC 的 latent token 上，再过 FSQ 和 decoder：
$$\mathbf{a}_t^{\mathrm{body}} = \mathcal{G}(z_t + 0.1 \cdot \Delta z_t)$$

这个设计非常聪明。你想想——SONIC 的 latent space 是它花了大量数据学出来的"怎么走路保持平衡"的 representation。你不去动它，只在上面叠一个小的 residual 来注入"怎么抓东西"的信息。相当于 SONIC 负责底盘稳定，adaptor 负责上层操作。这跟 LoRA 在 LLM 里的思路一脉相承——freeze 大模型，只训一个小 adapter。

**Terrain Traversal（爬楼梯/坐椅子）**：这种场景 SONIC 的 flat-ground 假设不适用了，所以不能只加 adaptor，得 fine-tune 整个 controller。但为了不让它忘了平地走路，他们加了一个 11×11 的 local height map 作为额外输入，用一个 3 层 CNN encode 进去，跟 proprioception 一起喂给 controller。

---

## 数据规模

这是最 impressive 的部分。

- **1,000 个 object assets**（来自 Robocasa, ComAsset, OMOMO, Hunyuan3D）
- **1,000 个 procedural terrain configurations**（台阶、斜坡、路缘石）
- **4 大类任务**：pick-up（捡起来）、whole-body manipulation（搬推拉）、sitting（坐下）、terrain traversal（地形行走）
- **总共 20,000+ 条 sequences**
- 全程不用真人遥控、不用 mocap、不用碰真机器人

---

## 实验结果

### Table 1：4D HOI 生成质量对比

| Method | Contact↓ | Penetration↓ | Inter. Score↑ | Tracking SR↑ | Body Dev↓ | Obj Dev↓ |
|--------|----------|-------------|--------------|-------------|-----------|----------|
| HOIDiff | 0.012 | 2.07% | 1.79 | 15.8% | 0.212 | 0.335 |
| CHOIS | 0.034 | 3.74% | 2.47 | 10.5% | 0.256 | 0.364 |
| DAViD | 0.246 | 1.46% | 2.74 | 24.0% | 0.472 | 0.583 |
| **GRAIL** | **0.008** | **0.90%** | **3.58** | **88.9%** | **0.091** | **0.085** |

Tracking Success Rate 88.9% vs 第二名 24.0%——这不是提升一点点，是降维打击。原因很简单：别的方法在视频生成之后才去猜 3D，GRAIL 在生成之前就把 3D 钉死了。

### Table 2：Tracking Policy 对比

| Method | SR↑ | ObjPos↓ | MPJPE-L↓ |
|--------|-----|---------|---------|
| HDMI | 48.5% | 0.283 | 122.3 |
| ResMimic | 49.2% | 0.393 | 80.9 |
| Ours w/o SONIC | 45.0% | 0.395 | 243.5 |
| Ours w/o $\pi_\phi$ | 39.7% | 0.303 | 37.1 |
| Ours w/o Rel. Obs. | 57.9% | 0.257 | 43.0 |
| **Ours (Full)** | **81.4%** | **0.135** | **41.8** |

几个关键 ablation：
- **去掉 SONIC**（从 scratch 训）→ SR 从 81.4% 掉到 45.0%。pretrained locomotion prior 是命根子。
- **去掉 adaptor $\pi_\phi$**（退化为 vanilla SONIC）→ SR 从 81.4% 掉到 39.7%。光会 track body 不够，得有 manipulation-specific 的 module。
- **去掉 relative observation**（用 absolute 代替 delta）→ SR 从 81.4% 掉到 57.9%。"未来 object pose 减当前 simulated pose"这个 delta observation 对泛化至关重要。

### Table 3：真机部署

| | Cube | Apple | Tea Box | Carrot | Wet Wipes | **Avg** |
|---|------|-------|---------|--------|-----------|---------|
| Seen | 100% | 60% | 100% | 70% | 90% | **84%** |

| | Spray Can | Lint Roller | Peach | Flashlight | Med Bottle | **Avg** |
|---|-----------|-------------|-------|------------|-----------|---------|
| Unseen | 100% | 50% | 90% | 80% | 80% | **80%** |

- **Stair-climbing: 90%**
- **Pick-up (seen): 84%**
- **Pick-up (unseen): 80%**

注意这些全是在 **10Hz** 推理频率下跑的（因为外接 RTX 5090 + streaming overhead）。能在 10Hz 下做到这种 success rate，说明 latent action space 本身就在做 temporal smoothing——FSQ 的 discretization 起到了类似 low-pass filter 的作用。

---

## 跟其他方案的本质区别

| 方案 | 数据来源 | 成本 | Scale 上限 | 问题 |
|------|---------|------|-----------|------|
| Teleoperation (ALOHA, Homie) | 真人遥控 | 高 | 低 | 每个场景都得重弄 |
| Motion Capture (GRAB, Humoto) | 穿 mocap 衣 | 高 | 中 | 只能录动作不能录场景 |
| In-the-wild video (VideoMimic, HumanX) | YouTube 视频 | 低 | 高 | 3D 重建全是歧义 |
| **GRAIL** | VFM + 3D assets | 中 | **高** | 需要 3D mesh + VLM 靠谱 |

GRAIL 的 sweet spot 是：**用 VFM 提供行为先验（behavior prior），用 3D assets 提供几何确定性（geometric certainty），把两者的优势叠加。**

---

## 我的几个 takeaway

**1. Privileged information 是 key。** 这个 paper 最核心的 insight 就是——如果你能把 3D scene 在 generation 之前就 specify 好，那 4D reconstruction 的 ill-posedness 就被大幅消除了。这不是什么新算法，是一种 problem formulation 的转变。从 "given video, infer 3D" 变成了 "given 3D, generate video, then lift back to 3D"。后者 trivially easier。

**2. Latent residual adaptation 会成为标配。** $\Delta z_t$ 这个设计太干净了。pretrained controller 的 latent space 是一个 rich representation of "how to not fall over"。你在上面叠一个小 residual 来注入 task-specific 信息，这跟 NLP 里 LoRA / prefix tuning 的思路完全一致。未来 humanoid 的 policy learning 大概率都会长这样：一个 general locomotion base + per-task latent adaptor。

**3. BPS encoding 让 policy 跨形状泛化。** Basis Point Set 把任意形状的 object encode 成固定维度的 descriptor，所以同一个 $\pi_\phi$ 能 track 从苹果到喷雾罐的各种物体。这跟 PointNet 的 global feature 思路类似，但更 lightweight（10 维 vs 1024 维）。

**4. Contact-gated reward 很关键。** $R_t^{\mathrm{grasp}}$ 只在 contact frame 上激活，这避免了 policy 在还没碰到物体时就开始奖励 grasp pose。$\mathcal{H}\{C_t\}$ 这个 indicator 来自 reconstruction 阶段的 contact label，是 4D HOI trajectory 传下来的 privileged information。

**5. 14 分钟一条 sequence，20000 条 = ~4.7 天单卡。** 而且大部分时间花在 joint optimization（~8 min）上。如果未来 VFM 能直接 output 3D-consistent video（比如 4D Gaussian Splatting based generation），这个 bottleneck 可以大幅压缩。

---

## 相关 links

- **GRAIL project page**: https://research.nvidia.com/labs/dair/grail/
- **SONIC (base controller)**: https://arxiv.org/abs/2511.07820
- **FoundationPose (object tracking)**: https://arxiv.org/abs/2312.00783
- **Kling (VFM used)**: https://klingai.com/
- **Infinigen (scene generation)**: https://arxiv.org/abs/2306.10629 (Procedural photorealistic worlds)
- **Unitree G1 (robot)**: https://www.unitree.com/g1/
- **InterMimic (4D HOI tracking eval)**: https://arxiv.org/abs/2503. (InterMimic: Towards universal whole-body control for physics-based human-object interactions)
- **GMR (retargeting)**: https://arxiv.org/abs/2510.02252
- **HDMI (baseline)**: https://arxiv.org/abs/2509.16757
- **ResMimic (baseline)**: https://arxiv.org/abs/2510.05070
- **DAViD (baseline)**: https://arxiv.org/abs/2503.20118

---

简而言之，GRAIL 就是**让 AI 在虚拟世界里演戏给机器人看，机器人看完就学会了**。不用人教，不用遥控，场景随便造，数据随便扩。然后学到的东西直接往真机上扔，84% 抓东西成功，90% 爬楼梯成功。

这是 data scaling for humanoid robotics 的一条全新路线——从 human-limited 变成 compute-limited。

---

Andrej, 很高兴能与你深入探讨这篇来自 NVIDIA 与 UCLA 的 paper 《GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors》。这篇工作非常契合你一直倡导的 "Software 2.0/3.0" 以及 data-driven robotics 的理念。它本质上是在构建一个极其强大的 data engine，试图跨越 teleoperation 和 motion capture 的物理可扩展性瓶颈。

为了 build your intuition，我将从宏观的 pipeline 设计到微观的公式推导、网络架构以及实验数据，为你进行全维度的拆解，并且会串联起相关的前沿工作。

---

### 1. 宏观 Intuition: 为什么需要 GRAIL？

Humanoid loco-manipulation 面临的核心痛点是 **data scaling barrier**。如果依赖 teleoperation (如 ALOHA 2 或 HumanPlus) 或者 motion capture，每一个新 object、每一种新 terrain 都需要重建物理场景并且重新采集，这受限于 human effort 和 hardware setup。

近期 Video Foundation Models (VFMs) 展现出了惊人的物理世界理解能力。一个直觉的做法是：直接从 in-the-wild 的 human video 中重建 4D Human-Object Interaction (HOI)。但是，从 monocular video 恢复 metric scale、object geometry、human shape 以及 contacts 是一个极度 ill-posed 的 inverse problem，存在巨大的 depth ambiguity 和 morphology mismatch。

GRAIL 的核心 insight 是：**与其在生成后去痛苦地猜测 3D 信息，不如在生成前就完全指定 3D configuration。** 
因此，GRAIL 保持 fully digital，直到最后 deployment 阶段才接触真实物理世界。它利用 3D assets 搭建好拥有绝对 metric scale 的场景，用 VFM 生成 video 作为 "interaction prior" (提供行为逻辑)，然后利用这个 fully specified 的 3D 环境作为强约束，去解出物理上 executable 的 4D trajectories，最后通过 RL 训练 tracking policy 并 distillation 到 visual policy。

---

### 2. Pipeline 核心拆解

整个 GRAIL pipeline 可以分为三个主要阶段加上一个 sim-to-real 部署阶段：

#### 2.1 Robot-Centric Human Video Generation
这一步解决的是 "行为从哪里来" 的问题。
为了确保后续 retargeting 到 Unitree G1 时 morphology mismatch 最小，GRAIL 没有直接生成 robot video，而是生成了 human video，但是这个 human asset 是经过 pre-fitted 的（与 G1 的 proportions 对齐）。

*   **Scene Assembly**: 使用 Infinigen (Raistrick et al., 2023) 生成 procedural environments，并且使用 rigid body simulation (XPBD) 让 object 落到一个稳定且 collision-free 的初始状态 $\Theta_1^{\mathcal{O}}$。
*   **VFM Generation**: 通过 VLM (GPT-4o) 根据渲染出的第一帧生成 interaction prompt，然后输入给 VFM (Kling 2.5 Turbo Pro) 生成 5 秒、24 fps、1080p 的 static-camera video。
*   **Intuition**: 这里的 static-camera 设置非常关键。因为相机不动，且我们完全知道相机的 intrinsics $C_K \in \mathbb{R}^{3\times3}$ 和 extrinsics $C_E = (r^{\mathcal{C}}, t^{\mathcal{C}})$，后续的 2D-to-3D lifting 就有了绝对的锚点。

#### 2.2 Interaction-Aware HOI Reconstruction
这是 paper 中最 math-heavy 的部分。VFM 生成的 video 在视觉上很流畅，但是在几何上充满了噪声（比如手穿模到物体内部，或者悬浮在物体上方）。GRAIL 通过一个 joint optimization 来强制实现 physical plausibility。

首先，通过独立的模块获取初始估计：
*   **Human Motion**: 使用 GENMO 提取 SMPL-X body pose，使用 WiLoR 提取 MANO hand pose。对于 missing detection，用 temporal linear interpolation 填补并用 Savitzky-Golay filter 平滑。
*   **Object Pose**: 使用 FoundationPose，在 RGB-only (zeroed depth channels) 设定下进行 6-DoF tracking。因为物体的 3D mesh 和初始位姿完全已知，tracking 极其精准。

接着进入联合优化阶段。优化目标不是绝对的 poses $\Theta$，而是 residual motions $\Delta \Theta$，最终位姿 $\Theta_t = \hat{\Theta}_t \oplus \Delta \Theta_t$。这里使用 6D rotation representation 以保证神经网络的连续性。

**公式解析 (1): 联合优化目标**
$$
L = \lambda_{\mathrm{kp}} L_{\mathrm{kp}} + \lambda_{\mathrm{proj}} L_{\mathrm{proj}} + \lambda_{\mathrm{depth}} L_{\mathrm{depth}} + \lambda_{\mathrm{cont}} L_{\mathrm{cont}} + \lambda_{\mathrm{reg}} L_{\mathrm{reg}}
$$
这里 $\lambda$ 是各项 loss 的权重。我们重点看几个创新的 loss：

**公式解析 (2): Keypoint Alignment**
$$
L_{\mathrm{kp}} = \frac{1}{T} \sum_{t=1}^T \| \mathcal{K}^{\mathcal{H}}(\Theta_t^{\mathcal{H}}) - p_t \|
$$
*   $T$: 总帧数。
*   $\Theta_t^{\mathcal{H}}$: 优化中的 SMPL-X 参数。
*   $\mathcal{K}^{\mathcal{H}}(\cdot)$: 已知 camera 的 projection function。
*   $p_t \in \mathbb{R}^{J \times 3}$: 从 video 中检测出的 2D body 和 hand keypoints。
这个 loss 保证了优化后的 3D motion 在 image space 上不偏离 VFM 生成的原始视频。

**公式解析 (4): Depth Alignment**
$$
L_{\mathrm{depth}} = \frac{1}{T} \sum_{t=1}^T \mathcal{CD}(V_t^{\mathcal{H}, \mathrm{vis}}, \mathbf{P}_t^{\mathcal{H}}) + \mathcal{CD}(V_t^{\mathcal{O}, \mathrm{vis}}, \mathbf{P}_t^{\mathcal{O}})
$$
*   $V_t^{\mathcal{H}, \mathrm{vis}}, V_t^{\mathcal{O}, \mathrm{vis}}$: 重建出的 human 和 object mesh 的 visible vertices。
*   $\mathbf{P}_t^{\mathcal{H}}, \mathbf{P}_t^{\mathcal{O}}$: 使用 MoGe-2 估计 metric depth，并用 SAM2 分割出 human 和 object，unproject 回 3D space 形成的 point clouds。
*   $\mathcal{CD}$: Bidirectional Chamfer Distance。
**Intuition**: VFM 生成的视频没有真实的 metric scale，直接用它会导致轨迹在 depth 轴上漂移。这里利用第一步渲染出的 3D 环境 ground truth 背景 depth map，将 MoGe-2 的 relative depth align 到 metric scale。然后强制 mesh 顶点去贴合这些 point clouds，这就把 VFM 的行为强行拉入了具有真实尺度的 3D 物理空间。

**公式解析 (5): Contact Alignment**
$$
L_{\mathrm{cont}} = \frac{1}{|\mathcal{T}_c|} \sum_{t \in \mathcal{T}_c} \mathcal{CD}_z(V_t^{\mathcal{H}, \mathrm{cont}}, V_t^{\mathcal{O}, \mathrm{cont}})
$$
*   $\mathcal{T}_c$: 通过 VLM 预测出存在 contact 的 frames 集合。
*   $V_t^{\mathcal{H}, \mathrm{cont}}$: SMPL-X part segmentation 找出的 contact body region 的 vertices。
*   $V_t^{\mathcal{O}, \mathrm{cont}}$: 通过 filter $\mathcal{F}$ 筛选出的在 screen space 上投射到 contact region 内的 object vertices。
*   $\mathcal{CD}_z$: 仅在 viewing direction 的 z 轴上计算 Chamfer Distance。
**Intuition**: 图像空间的 loss 无法解决手是在物体前面还是后面的问题。Contact loss 只在 detected contact 帧生效，并且只惩罚 depth 方向的偏差，这就巧妙地解决了手与物体交互时的穿透或悬浮问题，且不会破坏 2D 投影的准确性。

#### 2.3 Task-General Loco-Manipulation Tracking
重建出 motion 后，使用 GMR (Araújo et al., 2025) 将 SMPL-X motion retarget 到 Unitree G1 上。接下来不是对每一个 trajectory 训一个 policy，而是训练 task-general policies。这里基于了一个 pretrained whole-body controller: **SONIC** (Luo et al., 2025)。SONIC 使用 Finite Scalar Quantization (FSQ) 将 kinematic targets 编码为 discrete latent token $z_t = \mathcal{E}(\tilde{q}_t)$，再通过 decoder $\mathcal{G}(z_t)$ 解出 joint-level actions。

GRAIL 设计了两个互补的 tracker：

**A. Object-Aware Latent Adaptor (用于 manipulation)**
为了不破坏 SONIC 原有的 locomotion 能力，encoder、quantizer、decoder 全部 frozen。只训练一个 adaptor policy $\pi_\phi$。

**公式解析 (6): Latent Residual Modulation**
$$
(\Delta z_t, \mathbf{a}_t^{\mathrm{hand}}) = \pi_\phi(\mathbf{s}_t, \mathbf{o}_t), \quad \mathbf{a}_t^{\mathrm{body}} = \mathcal{G}(z_t + \lambda \Delta z_t)
$$
*   $\pi_\phi$: 3-layer MLP (512, 256, 128 dims, SiLU)。
*   $\mathbf{s}_t$: Proprioception (joint pos, vel, base ang vel 等)。
*   $\mathbf{o}_t$: Object reference (包含 object pos, hand-to-object transforms, BPS shape encoding, 以及 critical delta observations: reference future object pose - current simulated pose)。
*   $\Delta z_t$: 64-dim latent residual。
*   $\lambda$: 0.1 scaling factor。在 FSQ quantization 之前加入这个 residual。
*   $\mathbf{a}_t^{\mathrm{hand}}$: 2-dim binary signal (sigmoid + threshold)，控制左右手的 open/close grasp，映射到每只手 7 个 finger joints。
**Intuition**: 这种 residual latent modulation 极其优雅。它类似于在 LLM 中做 parameter-efficient fine-tuning，只在 latent action space 注入 manipulation-specific 的扰动，既赋予了这个 frozen controller 抓取能力，又保住了它的平衡能力。BPS (Basis Point Set) encoding 提供了 object-shape awareness，使得同一个 policy 可以泛化到不同形状的物体。

**B. Scene-Aware Tracker (用于 terrain traversal 和 sitting)**
在这个场景下，flat-ground prior 失效。因此需要 fine-tune 整个 controller (包括 encoder 和 decoder)，并且加入一个 height-map encoder $\epsilon_h$。

*   **Height-map Encoding**: 构建一个 11x11 的 grid，覆盖 robot 周围 1.5m 范围 (0.15m resolution)。向下 raycast 获取 terrain height，转换到 robot yaw-aligned local frame，形成 $[11, 11, 3]$ tensor。
*   **CNN Projector**: 3-layer CNN (channels [64, 128, 256], kernel 3x3, stride 2, LeakyReLU)，flatten 后得到 1,024-dim feature，与 proprioception 和 tokenizer feature concat 后输入 fusion MLP。
*   **Auxiliary Loss**: 训练了一个 parallel kinematic decoder $\mathcal{G}_{\mathrm{rec}}$ 重建 input motion targets，提供 MSE loss 防止 latent representation 崩溃。

#### 2.4 RL Reward 设计
**公式解析 (7) & (8) & (9):**
$$
R_t^{\mathrm{motion}} = \sum_i w_i \exp\left(-\frac{\|\tilde{\mathbf{x}}_{i,t} - \mathbf{x}_{i,t}\|^2}{\sigma_i^2}\right)
$$
$R_t^{\mathrm{motion}}$ 是 tracking reward，$\tilde{\mathbf{x}}$ 和 $\mathbf{x}$ 分别是 reference 和 simulated quantities。使用 exponential kernel 而不是纯 L2 distance，因为 exponential reward 会随着 error 增大而 saturate，防止 RL 为了追求不可能达到的完美匹配而产生振荡行为。

$$
R_t^{\mathrm{obj}} = w_p \exp(-\alpha_p \|\tilde{p}_t^{\mathcal{O}} - p_t^{\mathcal{O}}\|) + w_r \exp(-\alpha_r \|\tilde{r}_t^{\mathcal{O}} \ominus r_t^{\mathcal{O}}\|)
$$
Object tracking reward，分别惩罚 position 和 rotation 偏差。被 simulated finger-object contact indicator gate 住，只有在接触时才激活。

$$
R_t^{\mathrm{grasp}} = w_c \min\left(\frac{N_t^{\mathrm{contact}}}{N_{\min}}, 1\right) + w_d [-\cos(d_t^{\mathrm{thumb}}, d_t^{\mathrm{index}})]^+ + w_f \exp\left(-\gamma \frac{1}{N_f} \sum_j \|f_{j,t} - c_t\|\right)
$$
这个 grasp reward 设计得极其精妙：
1.  $w_c$ 项鼓励持续的 finger contact，saturating at $N_{\min}$。
2.  $w_d$ 项惩罚 thumb 和 index finger 的方向，$[-\cos(\cdot)]^+$ 促使它们从对侧接近，形成稳定 pinch grasp。$d_t$ 是从 object center 到 fingertip 的向量。
3.  $w_f$ 项将所有 fingertips $f_{j,t}$ 拉向 object contact centroid $c_t$。

---

### 3. 实验数据深度解析

#### Table 1: HOI Generation 对比
对比了 HOIDiff, CHOIS, DAViD 等 baselines。
*   GRAIL 的 Tracking Success Rate (SR) 达到了 **88.9%**，而第二名 DAViD 只有 24.0%。
*   Body Deviation (0.0913) 和 Object Deviation (0.0851) 比 baselines 低了一个数量级。
*   **Intuition**: 为什么差距这么大？因为 HOIDiff 和 CHOIS 是纯 generative 模型，缺少物理约束；DAViD 虽然用了 VFM，但是它的 3D scene 是在生成后才去 infer 的，因此存在极大的 depth-scale drift。GRAIL 的 privileged setup 彻底消灭了这种 drift。

#### Table 2: Task-General Tracking 对比与 Ablation
*   对比 HDMI 和 ResMimic，GRAIL 在 SR (81.4% vs 48.5%/49.2%) 和 Object Position Error (0.135 vs 0.283/0.393) 上碾压。
*   **Ours w/o SONIC**: 从 scratch 训练，SR 暴跌到 45.0%，证明 pretrained latent locomotion prior 极其关键。
*   **Ours w/o $\pi_\phi$**: 退化为 vanilla SONIC，SR 降到 39.7%。说明仅靠 body tracking 无法学会 manipulation。
*   **Ours w/o Rel. Obs.**: 将 relative object observations 换成 absolute，SR 降到 57.9%。说明预测 future object pose 与 current pose 的 delta 对 RL 策略泛化至关重要。

#### Table 3: Sim-to-Real 部署
在 Unitree G1 上，stair-climbing 达到 90% 成功率，object pick-up 在 seen objects 上 84%，unseen objects 上 80%。
部署频率只有 10Hz (由于外接 desktop with RTX 5090 并 streaming 视觉和 proprio 数据)。能在 10Hz 这么低的频率下实现如此 robust 的动态控制，说明 latent action space 具有极强的 temporal smoothing 能力。

---

### 4. 发散性联想与未来推演

1.  **VFM 与 Physics Simulation 的闭环**: GRAIL 目前把 VFM 当作开环的 "behavior prior"。未来如果能让 VFM 直接在 3D latent space 生成 (例如基于 3D Gaussian Splatting 或 Neural Radiance Fields 的 video generation)，就可以跳过痛苦且耗时的 joint optimization 阶段 (目前占据 14 分钟/序列中的 8 分钟)。
2.  **Closing the Sim-to-Real Gap for Manipulation**: 虽然 pick-up 达到了 84%，但 paper 也承认在 severe occlusion 或 fast motion 下 reconstruction 会 fail。Future direction 可能是引入 tactile sensing 或者将 VFM 直接 finetune 成能够理解 robot morphology 的 model。
3.  **LLM/VLM as Reward Designer**: 这里的 contact labels 是通过 VLM 查询视频帧获得的。如果让 VLM 实时观察 RL rollout 并提供 dense reward feedback (类似 Eureka 或 VLM-RM)，可能能够绕过需要重建精确 4D trajectory 的需求。
4.  **Latent Space 的 combinatorial structure**: $\Delta z_t$ 作为 latent residual 能够成功叠加在 FSQ token 上，暗示了 discrete latent action space 具有类似于自然语言中的 "word embedding + compositionality" 的特性。你可以加上一个 "grasp" vector，加上一个 "lift" vector，decoder 会自然将其组合成 "grasp and lift" 的动作，这或许是你一直追求的 neural computer architecture 在 robotics 领域的具象化。

### 相关参考链接

*   **GRAIL Project Page**: https://research.nvidia.com/labs/dair/grail/
*   **SONIC (Base Controller)**: https://arxiv.org/abs/2511.07820 (Supersizing motion tracking for natural humanoid whole-body control)
*   **GENMO (Human Motion Estimator)**: https://arxiv.org/abs/2501.07253 (虽然 paper 中引用待定，但指向此方向)
*   **FoundationPose (Object Tracking)**: https://arxiv.org/abs/2312.00783 (Unified 6D pose estimation and tracking of novel objects)
*   **GMR (Retargeting)**: https://arxiv.org/abs/2510.02252 (Retargeting matters: General motion retargeting for humanoid motion tracking)
*   **DAViD (Baseline VFM HOI)**: https://arxiv.org/abs/2503.20118 (Modeling dynamic affordance of 3d objects using pre-trained video diffusion models)
*   **Kling VFM**: https://klingai.com/

总结来说，GRAIL 通过极其严谨的 engineering 和清晰的 insight，把 VFM 的 "hallucination" 约束在了 physics-specified 的 3D cage 里，为 humanoid data scaling 提供了一条 non-teleoperated 的康庄大道。这种 asset-conditioned generative data 的思路，绝对会成为接下来 robot learning 的主流范式。
