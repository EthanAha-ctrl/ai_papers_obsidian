---
source_pdf: UniDex A Robot Foundation Suite for Universal Dexterous Hand Control from
  Egocentric Human Videos.pdf
paper_sha256: 2ccb2df92e35428fd2d1d3949366ff7161c0ca0ef4fd5498c9b03e7e62530975
processed_at: '2026-08-12T19:24:42-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Yo Andrej, 没问题。如果我们要用最直白的人话来聊 UniDex，本质上的故事非常简单：**dexterous hand 的 teleoperation data 极度昂贵且难收集，于是他们去白嫖了海量的人类第一人称操作视频，通过一套精妙的转换 pipeline，把人类的动作硬生生“翻译”成各种机器手的动作，并在这个庞大的合成数据集上预训练出了一个能跨手型泛化的 3D VLA foundation model。**

为了 build your intuition，我们把这篇 paper 拆解成几个核心的 engineering decisions，并且我不会漏掉任何背后的 technical depth。

### 1. The Data Bottleneck & The "Translation" Problem (Kinematic Retargeting)

**Intuition:** 人的手有 27 DoFs，机器手（比如 Inspire）只有 6 active DoFs，Shadow Hand 有 24 DoFs。骨骼结构完全不同，你不可能直接 copy joint angles。但是，manipulation 的本质在于 **fingertip 在 3D 空间中划过的轨迹**。所以，只要让机器手的 fingertip 去追赶人的 fingertip，就能实现 functional equivalent 的动作。

**Technical Deep Dive:**
为了完美对齐 fingertip 轨迹，他们引入了一个 6-DoF 的 dummy base offset $T_{\text{offset}}$。为什么需要这个？因为如果你把机器手的 base 死死钉在人类手腕的位置，由于 link lengths 不同，机器手可能会穿透物体，或者根本碰不到物体边缘。你需要一个全局的 "wrist tweak"。

Forward kinematics 公式定义了机器手 fingertip 的位置：
$$x _ { i } ( q ; T _ { \mathrm { o f f s e t } } ) = \mathrm { T r a n s } \Big ( T _ { \mathrm { w o r l d } } ^ { \mathrm { d u m m y } } T _ { \mathrm { o f f s e t } } T _ { i } ( q ) \Big ) \in \mathbb { R } ^ { 3 }$$
- $q$ 是机器手的 joint configuration。
- $T _ { \mathrm { w o r l d } } ^ { \mathrm { d u m m y } }$ 是 dummy base 在世界坐标系下的 pose，被固定为人类手的 pose $T_{\text{hand}}$。
- $T _ { \mathrm { o f f s e t } }$ 是从 dummy base 到真实 robot base 的 rigid transform（这就是 human 可以调的 6 个自由度）。
- $T _ { i } ( q )$ 是从 robot base 到第 $i$ 个 fingertip 的 homogeneous transform。
- $\mathrm{Trans}(\cdot)$ 提取 4x4 矩阵中的 translation 部分（3D 坐标）。

IK 求解器要最小化的 error vector 是：
$$e ( q , T _ { \mathrm { o f f s e t } } ) = \left[ \begin{array} { c } { x _ { 1 } ( q ; T _ { \mathrm { o f f s e t } } ) - x _ { 1 } ^ { \star } } \\ { \vdots } \\ { x _ { m } ( q ; T _ { \mathrm { o f f s e t } } ) - x _ { m } ^ { \star } } \end{array} \right] \in \mathbb { R } ^ { 3 m }$$
- $x _ { i } ^ { \star } \in \mathbb{R}^3$ 是人类第 $i$ 个 fingertip 的目标位置。
- $m$ 是机器手的手指数量。

对于有 mimic joints（耦合关节，比如 Inspire 手指联动）的手，主关节和从属关节之间的关系是：
$$q _ { j _ { s } } = k q _ { j _ { m } } + c$$
- $q _ { j _ { s } }$ 是从属关节角度。
- $q _ { j _ { m } }$ 是主关节角度。
- $k$ 和 $c$ 是硬件约束常数。通过迭代求解保证物理合法性。

### 2. Visual Alignment: Photoshop in 3D

**Intuition:** 如果直接把人类操作视频喂给 VLA，网络会看到一只毛茸茸的人手，这和机器手的外观差距太大，导致 visual domain gap。他们的做法极其暴力且有效：把人手从 pointcloud 里“抠掉”，然后把渲染好的机器手 3D mesh 塞进去。

**Technical Details:**
利用 WiLoR 和 SAM2 检测并 mask 掉 RGB-D 数据中的人类手部 points。接着将 retargeted 的 robot hand mesh 渲染进 scene pointcloud。最后，通过 pinhole camera model 把 fused pointcloud reproject 回 2D RGB frame，这样能保证 depth ordering 正确，避免单视角下的 occlusion 问题。这就给 policy 提供了近乎完美的 robot-centric visual supervision。

### 3. FAAS: A Universal Action Vocabulary

**Intuition:** 不同的机器手，URDF 里 joint index 0 可能代表大拇指，也可能代表手腕旋转。如果直接用 raw joint angles 做 action space，cross-embodiment 的 foundation model 根本无法训练。FAAS 的思想是把动作按 **functional role** 对齐。

**Technical Breakdown:**
FAAS 是一个 82 维的 vector：
- **前 18 维 (Wrist Poses):** 每只手 9 维（双手共 18 维）。9 维包含 6 维 continuous rotation representation（用两个 3D vectors 表示 local x 和 y 轴，避免 quaternion 的不连续性导致学习困难）+ 3 维 translation。
- **后 64 维 (Hand Joints):** 每只手 32 slots。其中 21 个 base slots 给通用手指功能（比如 thumb flexion, index curling 等），剩下 11 个 slots 给特定 hand 的额外 DoFs 或留给未来的手。

**Conceptual Mapping Table:**
| FAAS Dimension Slot | Functional Role | Allegro Hand Mapping | Inspire Hand Mapping | Wuji Hand Mapping |
| :--- | :--- | :--- | :--- | :--- |
| Slot 0-4 | Thumb Joints | Thumb Roll, Flexion... | Thumb Abduction, Flexion | Thumb Base, Middle, Tip |
| Slot 5-9 | Index Joints | Index Abduction, Flexion | (Mimic) Index Flexion | Index Base, Middle, Tip |
| ... | ... | ... | ... | ... |
| Slot 25-26 | Extra Wrist (Shadow)| N/A | N/A | Wrist Roll/Pitch |

通过这种 functional alignment，policy 学到的是 "bend the thumb to pinch" 这个概念，而不是 "set motor 3 to 0.5 radians"。这就是为什么 Inspire 手上训练的 policy 可以 zero-shot 迁移到 20-DoF 的 Wuji 手上。

### 4. UniDex-VLA: 3D Vision-Language-Action Flow Model

**Intuition:** 对于 tool-use，你必须精确知道工具手柄的 3D 几何形状和 contact affordance。2D image 缺乏深度信息，所以必须用 pointcloud。

**Architecture & Flow Matching:**
他们把 $\pi_0$ 架构里的 SigLIP 2D encoder 换成了 **Uni3D**（一个用 2D ViT 预训练权重初始化的 3D pointcloud encoder）。

训练目标采用 conditional flow-matching loss：
$$L ^ { \tau } ( \theta ) = \mathbb { E } _ { p ( A _ { t } \mid o _ { t } ), q ( A _ { t } ^ { \tau } \mid A _ { t } ) } \left[ \left\| v _ { \theta } ( A _ { t } ^ { \tau } , o _ { t } ) - u ( A _ { t } ^ { \tau } \mid A _ { t } ) \right\| \right]$$
- $o_t$ 是 observation (pointcloud + language + proprioception)。
- $A_t$ 是 ground truth 的 H-step action chunk。
- $\tau \in [0, 1]$ 是时间步参数。
- $A _ { t } ^ { \tau } = \tau A _ { t } + ( 1 - \tau ) \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。这是一个 linear-Gaussian 插值，从纯噪声平滑过渡到 ground truth action。
- $u ( A _ { t } ^ { \tau } \mid A _ { t } ) = A _ { t } - \epsilon$ 是 target vector field，直接指向 ground truth。
- $v _ { \theta }$ 是神经网络预测的 vector field。

Inference 时用 Forward Euler 积分去噪：
$$A _ { t } ^ { \tau + \delta } = A _ { t } ^ { \tau } + \delta v _ { \theta } ( A _ { t } ^ { \tau } , o _ { t } )$$
- $\delta = 0.1$ 是 step size，意味着只需要 10 步就能生成动作，比 standard diffusion 快很多。

### 5. UniDex-Cap: The Data Economics of Human-Robot Co-training

**Intuition:** Pretrain 之后，fine-tune 还是需要 real robot demos。为了进一步降本，他们搞了个头显+深度相机的便携设备，记录人在真实世界里的操作，转换成 robot data 和 robot demos 混着喂给模型。

**Technical Deep Dive:**
硬件上把 Apple Vision Pro 和 Intel RealSense L515 物理绑定，通过 GUI 标定两者的 extrinsic matrix $T_{\mathrm{RS}}^{\mathrm{VP}}$。坐标转换公式：
$$P _ { \mathrm { R S } } = T _ { \mathrm { R S } } ^ { \mathrm { V P } } P _ { \mathrm { V P } }$$
- $P _ { \mathrm { V P } }$ 是 Vision Pro 坐标系下的 hand/head pose (homogeneous coordinates)。
- $P _ { \mathrm { R S } }$ 是转换到 RealSense 相机坐标系下的 pose。

**The "2:1 Exchange Rate" Magic:**
实验得出了一个惊人的结论：**2 个 transformed human demos 大约能替代 1 个 real robot demo 的训练效果**。因为人类收集 demos 的速度是 teleoperation 的 5.2 倍，这相当于把 data collection 的效率提升了 2.5 倍左右。没有 robot data 全靠 human data 依然成功率为 0（因为视觉和本体感觉的 gap 无法完全消除），所以两者是互补关系。

### 6. Extended Intuitions & Hallucinations for Andrej

1. **FAAS is essentially BPE for Actions:** 就像 LLM 里用 Byte-Pair Encoding 把不同语言的高频组合映射到同一个 token 一样，FAAS 把硬件异构性抽象成了功能语义。如果未来有了 30-DoF 的仿生手，只要按照 functional role 往那个 82 维向量的 reserved slots 里填值，foundation model 就能直接 inference。这种 action representation 的 design 甚至比模型架构本身更重要。
2. **The Next Frontier: Action-Free Egocentric Pretraining:** 论文 limitation 里提到还没用 action-free 的视频。直觉上，如果 UniDex 能在数百万小时的 Ego4D 视频上做一个类似于 V-JEPA 的 3D world model pretraining，学到 physical dynamics 和 affordance，然后再用 UniDex-Dataset 做 action alignment，dexterous manipulation 的 data wall 就真的被打破了。
3. **Pointcloud vs NeRF/3DGS:** UniDex 用的是 raw pointcloud (Uni3D)。如果在高度 cluttered 或 occluded 的场景下，raw pointcloud 会有很多 holes。未来的 VLA 可能会直接 ingest 3D Gaussian Splatting 或者 NeRF 的特征，这样能给 policy 提供一个 dense 的 3D feature field，对于精细的 in-hand manipulation contact reasoning 会有质的飞跃。
4. **The Flow Matching "Trick":** 为什么用 flow matching 不用 DDPM？因为 high-DoF action space (82维) 的 multi-modal distribution 极其复杂。Flow matching 通过 straight-line probability paths 生成更加 stable 的 vector field，在 inference 时只需要 10 步 Euler integration。对于 real-time robot control (通常需要 10-50Hz)，这种低延迟的生成式 policy 是必需的。

### Web Links for Reference

- UniDex Project Page: [https://unidex-ai.github.io/](https://unidex-ai.github.io/)
- $\pi_0$ Flow Matching VLA: [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
- Uni3D Pointcloud Encoder: [https://arxiv.org/abs/2310.06773](https://arxiv.org/abs/2310.06773)
- Flow Matching Theory (Lipman et al.): [https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)
- SAM2 (Visual Masking): [https://arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714)
- DemoGen (Spatial Data Aug): [https://arxiv.org/abs/2502.16932](https://arxiv.org/abs/2502.16932)

---

Hey Andrej, 这篇 paper 的 core idea 非常符合你对于 scaling up robot data 的直觉。解决 dexterous manipulation 数据稀缺的路径，往往涉及 cross-embodiment training 和 leveraging human data。UniDex 的切入点正是 egocentric human videos，将海量的人类第一人称操作视频转化为 robot-executable trajectories，并设计了一套能跨越不同 dexterous hand morphology 的 unified action space (FAAS)，最后辅以一个便携的 data capture setup (UniDex-Cap) 来 bridge the sim-to-real / human-to-robot gap。

下面我为你详细拆解这篇 paper 的技术细节，试图 build your intuition about why this pipeline works。

### 1. UniDex-Dataset: 从 Human Video 到 Robot Trajectory 的转换

Intuition: 人类视频是 God-given 的 massive manipulation dataset，但 human hand 和 robot dexterous hand 在 kinematics 和 visual appearance 上存在巨大的 domain gap。直接让 robot 模仿 human joint angles 会失败，因为 link lengths 和 joint limits 完全不同。UniDex 的核心洞察是：**fingertip trajectories and contact semantics** 才是 manipulation 的 invariant features。

#### 1.1 Kinematic Retargeting (Human-in-the-loop)
为了解决运动学差异，他们提出了一个两阶段的 retargeting pipeline。

首先，定义 human fingertip targets:
$$X ^ { \star } = \left[ x _ { 1 } ^ { \star } , \ldots , x _ { m } ^ { \star } \right] \in \mathbb { R } ^ { 3 \times m }$$
这里 $X ^ { \star }$ 是一个 $3 \times m$ 的矩阵，$m$ 代表 robot hand 的 finger 数量，$x _ { i } ^ { \star } \in \mathbb{R}^3$ 是第 $i$ 个 fingertip 在 world frame 下的 3D target position。

接着，为了允许全局的 hand base 调整以保证 physical plausibility (比如避免穿透物体)，他们引入了一个 6-DoF 的 dummy base offset $T_{\text{offset}}$。Forward kinematics 公式如下：
$$x _ { i } ( q ; T _ { \mathrm { o f f s e t } } ) = \mathrm { T r a n s } \Big ( T _ { \mathrm { w o r l d } } ^ { \mathrm { d u m m y } } T _ { \mathrm { o f f s e t } } T _ { i } ( q ) \Big ) \in \mathbb { R } ^ { 3 }$$
- $q$ 是 robot joint configuration。
- $T_{\mathrm{world}}^{\mathrm{dummy}}$ 是 dummy base 在 world frame 的 pose，固定为 human hand 的 pose $T_{\text{hand}}$。
- $T_{i}(q)$ 是从 robot base 到第 $i$ 个 fingertip 的 homogeneous transform。
- $\mathrm{Trans}(\cdot)$ 提取 translation 部分。

IK 的优化目标就是最小化 fingertip residual：
$$e ( q , T _ { \mathrm { o f f s e t } } ) = \left[ \begin{array} { c } { x _ { 1 } ( q ; T _ { \mathrm { o f f s e t } } ) - x _ { 1 } ^ { \star } } \\ { \vdots } \\ { x _ { m } ( q ; T _ { \mathrm { o f f s e t } } ) - x _ { m } ^ { \star } } \end{array} \right] \in \mathbb { R } ^ { 3 m }$$
这是一个 $3m$ 维的 error vector。通过 PyBullet 的 multi-end-effector IK solver 求解 $q$。

对于 mimic joints (比如 Inspire, Oymotion 这类带耦合关节的手)，他们用迭代修正：
$$q _ { j _ { s } } = k q _ { j _ { m } } + c$$
- $q_{j_s}$ 是 slave joint (mimic joint) 的角度。
- $q_{j_m}$ 是 master joint 的角度。
- $k$ 和 $c$ 是 kinematic constraints 常数。

**Human-in-the-loop 部分**：完全自动的 IK 有时候会产生不合理的 contact，作者用一个 GUI 让 human 调整 $T_{\text{offset}}$ 的 6 个 DoF (3 translation + 3 rotation)。直觉上，这相当于给 robot hand 一个全局的 "wrist tweak"，让它更好地 wrap around 物体。这种 semi-automatic 方式在 scaling up 数据和保证质量之间取得了平衡。

#### 1.2 Visual Alignment
Visual gap 的处理非常粗暴有效：
1. 使用 WiLoR 和 SAM2 检测并 mask out human hand 的 pointcloud。
2. 将 retargeted 的 robot hand mesh 渲染并 insert 到 scene pointcloud 中。
3. 通过 pinhole camera model 把 fused pointcloud reproject 回 2D RGB-D frame，确保 depth ordering 正确，避免 visual occlusion 造成的问题。

### 2. FAAS: Function-Actuator-Aligned Space

Intuition: 不同的 dexterous hands (从 6 DoF 到 24 DoF) 拥有不同的 URDF 结构。如果直接用 raw joint angles 作为 action space，跨具身泛化是不可能的，因为 index 0 在 Hand A 可能是 thumb flexion，在 Hand B 可能是 wrist roll。作者构建了 FAAS，将 functionally similar actuators 映射到 shared coordinates。

FAAS 是一个 82 维的 action vector：
- **前 18 维**: Wrist poses (双手，每只手 9 维)。9 维 pose 包含 6 维 continuous rotation representation (两个 3D vectors 表示 local x 和 y axes，这是常规的 rotation 处理技巧，避免 quaternion discontinuity) + 3 维 translation。
- **后 64 维**: Joint commands (双手，每只手 32 slots)。
  - 21 个 base actuator slots 跨所有 hand 共享。
  - 11 个 extra slots 给 hand-specific DoFs (比如 Shadow Hand 的额外 wrist joints) 或留给未来的 hand。

**Mapping 逻辑**：FAAS 按照 functional roles (thumb-index pinch, finger curling, lateral ab-/adduction) 对齐 actuators。例如，FAAS indices {0, 1, 3, 5, 6} 在 Oymotion, Allegro, Inspire, Wuji 中都对齐到功能相似的 joints。这种 design 让 policy network 学到的是 "如何执行一个 pinch" 而不是 "如何驱动 joint index 3"。

### 3. UniDex-VLA: 3D Vision-Language-Action Model

Intuition: Dexterous tool-use 需要精确的 3D geometric reasoning，特别是单视角 egocentric view 下。2D image encoders 难以捕捉 depth 和 contact affordance。

#### 3.1 Architecture
- Base model: 基于 $\pi_0$ 架构修改。
- Vision Encoder: 替换了 PaliGemma 中的 SigLIP 2D encoder，改用 **Uni3D**。Uni3D 是一个 vanilla ViT，用 2D pretrained weights 初始化，将 pointcloud features 与 image-text-aligned features 对齐。这是一个非常聪明的 3D representation injection 方式。
- Inputs: $o_t = [P_t, \ell_t, q_t]$
  - $P_t$: Single-view colored pointcloud (从 RGB-D 获取，cropped + downsampled)。
  - $\ell_t$: Language instruction。
  - $q_t$: Robot proprioception (在 FAAS 空间中表示)。
- Outputs: $A_t = [a_t, \dots, a_{t+H-1}]$，H-step action chunk。Wrist pose 使用 relative pose (相对于 action chunk 第一帧)，hand joints 使用 abstracted representation。

#### 3.2 Flow-Matching Objective
为了生成 multi-modal action distributions，他们使用了 conditional flow-matching loss：
$$L ^ { \tau } ( \theta ) = \mathbb { E } _ { p ( A _ { t } \mid o _ { t } ), q ( A _ { t } ^ { \tau } \mid A _ { t } ) } \left[ \left\| v _ { \theta } ( A _ { t } ^ { \tau } , o _ { t } ) - u ( A _ { t } ^ { \tau } \mid A _ { t } ) \right\| \right]$$
- $\tau \in [0, 1]$ 是 time step。
- $q ( A _ { t } ^ { \tau } \mid A _ { t } ) = \mathcal { N } ( \tau A _ { t } , ( 1 - \tau ) I )$ 是 linear-Gaussian probability path。这意味着 $A_t^\tau$ 是 ground truth action $A_t$ 和 noise 的 linear interpolation。
- 采样 $A _ { t } ^ { \tau } = \tau A _ { t } + ( 1 - \tau ) \epsilon$，其中 $\epsilon \sim \mathcal { N } ( 0 , I )$。
- Target vector field $u ( A _ { t } ^ { \tau } \mid A _ { t } ) = A _ { t } - \epsilon$。直觉上，这是指向 ground truth 的 vector。
- Network $v_\theta$ 预测这个 vector field。

Inference 时用 Forward Euler integration 去噪：
$$A _ { t } ^ { \tau + \delta } = A _ { t } ^ { \tau } + \delta v _ { \theta } ( A _ { t } ^ { \tau } , o _ { t } )$$
- $\delta = 0.1$ 是 step size。
- $A _ { t } ^ { 0 } \sim \mathcal { N } ( 0 , I )$ 是初始纯噪声。

### 4. UniDex-Cap: Human-Robot Data Co-training

Intuition: 即使 pretrain 了，finetune 依然需要 real robot demos，这很贵。如果能把 human in-the-wild 的操作直接拿来 co-train 就能极大降低成本。

Setup:
- Apple Vision Pro: 捕获 hand/head poses。
- Intel RealSense L515: 捕获高质量的 RGB-D。
- 3D-printed mount: 物理耦合两者，固定 relative transform。

Calibration: 通过 GUI 调整 Vision Pro 坐标系到 RealSense 坐标系的 rigid transform $T_{\mathrm{RS}}^{\mathrm{VP}}$：
$$P _ { \mathrm { R S } } = T _ { \mathrm { R S } } ^ { \mathrm { V P } } P _ { \mathrm { V P } }$$
- $P_{\mathrm{VP}}$ 和 $P_{\mathrm{RS}}$ 是 homogeneous coordinates。

**惊人的实验结论 (Human-Robot Exchange Rate)**:
在 Make Coffee task 上，作者发现 human demos 和 robot demos 的 exchange rate 大约是 **2:1**。即 2 个 transformed human demos 大致能替代 1 个 real robot demo 的训练效果。考虑到 human demos 的收集速度是 robot demos 的 ~5.2 倍，这种 co-training 极大地提高了 data efficiency。

### 5. Experiments & Results Analysis

在 5 个 challenging tool-use tasks (Make Coffee, Sweep Objects, Water Flowers, Cut Bags, Use Mouse) 跨 2 种 hands (Inspire, Wuji) 上测试。

- **Performance**: UniDex-VLA 达到了 81% 的 average task progress，远超 $\pi_0$ (38%)、DP 和 DP3。特别是在 "Use Scissors to Cut Bags" 这种高难度 in-hand reconfiguration 任务上，相比 baseline 有 84.6% 的相对提升。这证明了 pretraining 在 UniDex-Dataset 上注入了强大的 dexterous motion priors。
- **Spatial Generalization**: 通过 DemoGen 对 pointcloud 做 geometric editing (平移物体) + TAMP 生成新 poses 的 robot states，实现了全工作空间的高成功率。3D perception 天然支持这种空间泛化。
- **Object Generalization**: 换了不同 color, size, handle/spout 的 kettle，依然 work。
- **Zero-shot Cross-Hand Generalization**: 在 Inspire Hand (6 DoF) 上训练 Make Coffee，直接 zero-shot deploy 到 Wuji (20 DoF) 和 Oymotion (6 DoF, kinematics 不同)。UniDex-VLA 达到了 60% (Oymotion) 和 40% (Wuji) 的成功率，baselines 几乎全挂。这是 FAAS 最有力的证明。

### 6. 联想与发散

1. **Scaling Laws for Dexterous Priors**: 这篇 paper 本质上是在做 dexterous manipulation 的 "ImageNet moment"。通过将 heteroembodiment 数据统一到 FAAS，并在 9M frames 上 pretrain，模型学到了 universal tool-use affordance。下一步可能是把 action-free 的 egocentric videos (比如 Ego4D) 也利用起来，做 causal predictive modeling，类似 V-JEPA 但针对 3D manipulation。
2. **Human-in-the-loop vs Fully Automatic Retargeting**: 虽然作者把 human effort 降到了最低 (只调 6 个 DoF slider)，但要 scale 到 millions of videos，可能需要学习一个 diffusion model 来自动 predict $T_{\text{offset}}$，或者引入 physics simulation 验证 contact plausibility 来自动 reject bad retargets。
3. **Pointcloud vs 2D+Depth**: UniDex 证明了对于 tool-use，explicit 3D pointcloud (Uni3D) 比 2D RGB 更好。因为 3D perception 对于 contact 估计至关重要。未来可能会走向 larger 3D foundation models (如 Point-E or Sora 类的 3D generative models) 作为 perception backbone。
4. **Active Vision for Dexterous Hands**: 论文用的是 single-view egocentric RGB-D。在 occlusion 严重时 (比如 hand wrap around object)，single-view 可能丢失关键 contact 信息。结合 active vision (head/wrist 微调) 或者 tactile sensing 会是自然的 extension。
5. **FAAS limitations**: 82 维可能对于 future biologically inspired hands (比如 30 DoF) 不够。一个 dynamic length 的 token-based action space (类似 RT-2 的 action tokens) 结合 cross-attention 可能是更 flexible 的 unified action space，但 FAAS 的 simplicity 是其 zero-shot transfer 成功的关键。

### Web Links for Reference

- Project Page: [https://unidex-ai.github.io/](https://unidex-ai.github.io/)
- $\pi_0$ (Base VLA architecture): [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
- Uni3D (3D Vision Encoder): [https://arxiv.org/abs/2310.06773](https://arxiv.org/abs/2310.06773)
- DP3 (3D Diffusion Policy): [https://arxiv.org/abs/2403.03954](https://arxiv.org/abs/2403.03954)
- OpenTeleVision (Teleoperation): [https://arxiv.org/abs/2407.01512](https://arxiv.org/abs/2407.01512)
- WiLoR (Hand detection for masking): [https://arxiv.org/abs/2502.16420](https://arxiv.org/abs/2502.16420) (actually WiLoR is CVPR 2025, link might vary, but this is related to hand tracking)
- SAM2 (Masking): [https://arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714)
- DemoGen (Data Augmentation): [https://arxiv.org/abs/2502.16932](https://arxiv.org/abs/2502.16932)
