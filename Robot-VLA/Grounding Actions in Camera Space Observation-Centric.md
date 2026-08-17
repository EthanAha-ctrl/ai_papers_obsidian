---
source_pdf: Grounding Actions in Camera Space Observation-Centric.pdf
paper_sha256: f908d234e4e5c5c8e823ed8a3469e40fac4319bdca3df10fdf4f544965ac4851
processed_at: '2026-08-04T22:26:06-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲 OC-VLA

## 一句话版本

现在所有 VLA model 都让机器人"闭着眼睛猜"自己该怎么动，OC-VLA 把机器人的眼罩摘了。

---

## 问题出在哪

想象你教一个机器人抓杯子。你看的是**摄像头画面**（2D 图像），但你让机器人输出的动作是**相对于机器人自己底座的坐标**（"往前 30 厘米、往上 10 厘米"）。

问题来了：摄像头装在哪，机器人底座在哪，这俩位置关系每次都不一样。同一个"抓杯子"动作，换个摄像头角度，画面完全变，但机器人底座坐标里的目标动作**一模一样**。

这就好比：你让一个人看照片认路，但给他的答案格式是"从你家门口往东走 50 米"。换个人看同一张照片，他家在不一样的地方，答案完全不同，但他只能看到照片。

**模型被逼着从 2D 画面反推 3D 空间关系，而且每个摄像头角度都要反推一套不同的对应规则。** 1417 个摄像头角度 = 1417 套规则，数据不够，模型学不过来，就糊了。

---

## OC-VLA 干了啥

**把动作的"答案格式"从机器人底座坐标，换成摄像头坐标。**

就这么简单。训练前用摄像头标定矩阵把答案转换一下，推理时预测完了再转回来发给机器人。模型内部架构一行代码不改。

---

## 为什么这么简单的东西 work

关键直觉：

- 视觉模型（DINOv2、CLIP）是在图像上训练的，它的"母语"是摄像头坐标
- 机器人动作以前用机器人底座坐标，等于让视觉模型说外语
- 现在动作也用摄像头坐标，**模型在自己母语里做 prediction**，舒服了

更深的直觉：摄像头动了，画面怎么变，摄像头坐标里的动作也跟着怎么变，**两者同步变化**，模型学到的规律就稳定。以前摄像头动了，画面变但机器人底座动作不变，模型很懵——同样的输入对应不同的输出，或者说不同的输入对应同样的输出，梯度互相打架。

---

## 效果多好

- **模拟实验**：成功率涨 8~14 个点，离散动作空间涨得更多（13.8%）
- **真实机器人固定摄像头**：比 OpenVLA-OFT（7B 参数）还高 5 个点，而 OC-VLA 只有 334M 参数。**小模型靠对齐坐标系打败大模型。**
- **换没见过的摄像头角度**：别人掉 20 个点，OC-VLA 只掉 14 个点
- **训练数据里摄像头有轻微扰动**：OC-VLA 优势更大（+12.5%）

特别值得注意的是**长序列任务**（16 步 trajectory）：robot frame 崩到 16.4%，camera frame 39.4%，差 23 个点。每步小误差，累积起来就是灾难，坐标系对齐把误差源头掐了。

---

## 凭啥这么简单的东西没人早点做

我觉得有三个原因：

1. **惯性思维**：机器人控制历来在 base frame，数据集也这么标，大家没质疑过
2. **标定数据缺失**：需要每个 trajectory 带 camera extrinsic，Droid 之前大规模数据集不一定有
3. **跨领域盲区**：做 vision 的人不关心 robot frame，做 robotics 的人不深究 vision encoder 的 supervision space，两边各自默认自己的坐标系是对的

OC-VLA 的作者同时懂 vision pretraining 和 robotics control，才发现了这个 misalignment。

---

## 什么时候管用、什么时候不管用

**管用**：
- 多摄像头、多视角数据训练（Droid 这种）
- 想泛化到新视角部署
- 跨机器人迁移（不同 setup 不同 camera 位置）

**不管用 / 需要小心**：
- 单摄像头固定不动，优势很小
- 任务依赖绝对世界方向（"往北走"、"对准重力方向"），摄像头坐标会丢这个信息
- 摄像头标定不准，误差会直接传到动作上
- 多摄像头同时输入，anchor 到哪个 camera frame 需要额外设计

---

## 我的判断

这工作不会拿 best paper，但**应该成为 VLA 的默认 preprocessing**，就像图像输入要 normalize 一样自然。未来所有 VLA model 出厂就该默认 camera frame action，除非有特殊理由。

它揭示的教训比方法本身更重要：**当你 model 学不好，先别急着堆参数、改架构，先检查你的输入和监督信号是不是在同一个"语言"里。**

---

# OC-VLA 深度讲解

这篇 paper 解决了 VLA (Vision-Language-Action) model 里一个长期被忽视但非常本质的问题：**perception space 和 action space 的 coordinate frame misalignment**。我读完后的直觉是,这个工作虽然方法极简(本质就是一个坐标变换),但揭示的 insight 极其深刻,值得仔细拆解。

---

## 1. 核心问题：为什么 Robot Base Coordinate 是错的 Prediction Target

### 1.1 Misalignment 的本质

考虑现有 VLA model 的 pipeline：

- **Perception 侧**：image 经过 DINOv2 / CLIP 等 vision encoder,这些 backbone 在 web-scale image data 上预训练,**supervision signal 全部定义在 image / camera coordinate**(2D bbox、segmentation mask、depth、optical flow 等等)。Latent representation 天然 align 到 camera viewpoint。
- **Action 侧**：end-effector pose 的 ground truth 几乎全部定义在 **robot base coordinate**（world frame）,因为这是 robot controller 的 native frame。

这两个 frame 之间通过 extrinsic matrix **T** 隔开,而 **T 是 per-setup 不同的**。这意味着 model 必须隐式地从 2D image 反推 3D world action,等价于隐式 estimate **T**。这是一个 ill-posed 的 inverse problem,尤其在 single-view 设置下。

### 1.2 为什么在大规模 Pretraining 下问题被放大

Droid dataset (https://droid-dataset.github.io/) 包含 **1417 个不同的 third-person camera viewpoints**。如果 action target 在 robot base frame,那么：

- 同一个 robot action（例如把杯子从 A 移到 B）,在 robot base frame 是**唯一**的 target。
- 但从 1417 个不同相机看,这个 action 在 image 上的 appearance 完全不同。
- Model 被迫把 1417 种不同的 2D observation 映射到**同一个** robot-base action,**这等价于强制 model 内部学会 1417 套不同的 T^-1 inverse mapping**。

这是典型的 **supervision conflict**：同一 label 对应分布差异极大的 input,梯度互相打架,generalization 必然差。这正是 OC-VLA 想解决的核心矛盾。

---

## 2. 方法：把 Action Anchor 到 Camera Frame

### 2.1 核心数学

给定两个相邻时刻 end-effector 在 world frame 的 pose：

$$\mathbf{P}_{\text{world}_1}, \mathbf{P}_{\text{world}_2} \in \mathbb{R}^{4 \times 4}$$

这是 4×4 homogeneous transformation matrix,形式为：

$$\mathbf{P} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{bmatrix}$$

其中 $\mathbf{R} \in SO(3)$ 是 3×3 rotation matrix,$\mathbf{t} \in \mathbb{R}^3$ 是 translation。下标 `world_1`, `world_2` 表示在 robot base (= world) frame 下,timestep 1 和 timestep 2 的 pose。

**Relative action in world frame**（公式 1）：

$$\mathbf{A}_{\text{world}} = \mathbf{P}_{\text{world}_2} \mathbf{P}_{\text{world}_1}^{-1}$$

这是 SE(3) 上的右乘 composition,表示 "从 pose 1 到 pose 2 的相对变换"。为什么用 relative 而不是 absolute？因为 absolute pose 对 robot 起始位置敏感,relative action 更具 invariant 性,且与模仿学习的 chunking 思想一致（参考 Diffusion Policy https://diffusion-policy.cs.columbia.edu/）。

**Camera frame 变换**（公式 2）：

$$\mathbf{P}_{\text{cam}_2} = \mathbf{T} \mathbf{P}_{\text{world}_2}, \quad \mathbf{P}_{\text{cam}_1} = \mathbf{T} \mathbf{P}_{\text{world}_1}$$

其中 $\mathbf{T} \in \mathbb{R}^{4 \times 4}$ 是 world-to-camera extrinsic matrix：

$$\mathbf{T} = \begin{bmatrix} \mathbf{R}_{wc} & \mathbf{t}_{wc} \\ \mathbf{0} & 1 \end{bmatrix}$$

$\mathbf{R}_{wc}$：world→camera 的 3D rotation
$\mathbf{t}_{wc}$：world 原点在 camera frame 下的坐标

**Relative action in camera frame**（公式 3）：

$$\mathbf{A}_{\text{cam}} = \mathbf{P}_{\text{cam}_2} \mathbf{P}_{\text{cam}_1}^{-1}$$

代入展开（公式 4）：

$$\boxed{\mathbf{A}_{\text{cam}} = \mathbf{T} \mathbf{A}_{\text{world}} \mathbf{T}^{-1}}$$

这是 group theory 里的 **conjugation**（共轭变换）。直觉：同一个 abstract 的 relative motion,在 world frame 表达为 $\mathbf{A}_{\text{world}}$,在 camera frame 表达为 $\mathbf{T}\mathbf{A}_{\text{world}}\mathbf{T}^{-1}$。几何上是同一个 transformation,只是 base frame 换了。

最终把 $\mathbf{A}_{\text{cam}} \in \mathbb{R}^{4\times4}$ 转成 7-dim action $\langle x, y, z, \text{roll}, \text{pitch}, \text{yaw}, \text{gripper}\rangle$ 作为 model 的预测 target。

### 2.2 为什么这样 Work —— Geometric Intuition

这里是我觉得 paper 最深刻的地方。来看 perception 的本质：

Camera 的 image formation（公式 6, 7）：

$$u = \frac{f_x \cdot X_{\text{cam}}}{Z_{\text{cam}}} + c_x$$
$$v = \frac{f_y \cdot Y_{\text{cam}}}{Z_{\text{cam}}} + c_y$$

变量解释：
- $(X_{\text{cam}}, Y_{\text{cam}}, Z_{\text{cam}})$：3D point 在 camera frame 下的坐标
- $(u, v)$：投影到 image plane 的 pixel coordinate
- $f_x, f_y$：x、y 方向的 focal length
- $c_x, c_y$：principal point（通常接近 image center）

Key insight:
- **Camera intrinsic K 只依赖相机型号**,同款相机 K 完全一样。
- **Camera extrinsic T 依赖安装位置**,每个 setup 不同。

所以：image observation → camera coordinate 的反推**只需要 K**（fixed,可以 hard-coded 或 pretrained）,但 camera coordinate → robot base 需要 **T**（per-setup 变化,model 必须隐式学）。

把 action target 放在 camera frame,model 只需学 **observation → camera-frame action**,这个 mapping 对 camera 位置变化是 **covariant** 的（camera 移动时,target 跟着旋转,与 observation 一致变化）。如果 target 放在 robot frame,model 需要学 observation → robot action,这个 mapping 对 camera 位置变化是 **invariant** 的（target 不变,observation 变）,这是更难学的。

一句话总结:**covariance 比 invariance 容易学,因为信息没有 lost**。

---

## 3. 模型架构详解

### 3.1 Backbone 选择

基于 Dita（https://arxiv.org/abs/2503.19757）的 300M lightweight VLA architecture。整体 pipeline：

```
Language instruction ──→ CLIP text encoder (frozen) ──→ text tokens
                                                              │
RGB image (224×224) ──→ DINOv2 ──→ image features ──→ Q-Former (4 layers) + FiLM
                                                              │
                                                       ──→ LLaMA2-style Transformer (12 layers, 768 hidden, causal)
                                                              │
                                                       predicted action (7-dim)
```

细节：
- **CLIP text encoder**（https://arxiv.org/abs/2103.00020）：frozen,encode language instruction。
- **DINOv2**（https://arxiv.org/abs/2304.07193）：self-supervised vision encoder,输出 dense image features。
- **Q-Former**（来自 BLIP-2 https://arxiv.org/abs/2301.12597）：4 层,把 image token 数从 196 压缩到 **32**，控制 model size。
- **FiLM conditioning**（https://arxiv.org/abs/1709.07871）：每个 Q-Former block 注入 FiLM layer,用 text embedding 做 conditioning,引导 image feature 的 selection 和 compression。直觉：让 language 决定 "看哪里"。
- **Transformer decoder**：LLaMA2-style,12 层,hidden 768,causal mask。总参数 ~334M。

### 3.2 两种 Action Space 实现

**Continuous action space**：
- Transformer 作为 **Diffusion Transformer (DiT)**（https://arxiv.org/abs/2212.09748）
- Training: DDPM（https://arxiv.org/abs/2006.11239）100 timesteps,对 ground truth action 加 Gaussian noise,model 预测 noise（epsilon-prediction）
- Loss: MSE
- Inference: DDIM（https://arxiv.org/abs/2010.02502）10 timesteps 加速

**Discrete action space**：
- Action normalize 到固定 range 后 quantize 成 discrete bins
- Transformer **non-autoregressive** 一次 forward 预测全部 action token（虽然用 causal mask,但 output 一次性产生）
- Loss: cross-entropy
- 优势：token 间 semantic consistency 更好,推理效率高（参考 OpenVLA https://openvla.github.io/）

### 3.3 Training vs Inference 的 Asymmetry

这是 OC-VLA 的关键工程细节：

| 阶段 | Input | Output | 坐标变换方向 |
|------|-------|--------|------------|
| Training | dataset 里的 robot-base pose | camera-base action label | World → Camera (用 T) |
| Inference | image + language | predicted camera-base action | Camera → World (用 T^-1) 再发给 robot controller |

所以 **T 在训练时是 label 生成器,inference 时是 action 反变换器**。整个 model 内部永远不知道 T 存在,只在 data preprocessing 和 postprocessing 用。这就是 paper 强调的 **plug-and-play**。

---

## 4. 实验数据深度解读

### 4.1 Simulation: ManiSkill2 (https://maniskill2.github.io/)

Setup：
- 5 个 task families：PickCube, StackCube, PickSingleYCB, PickClutterYCB, PickSingleEGAD
- 300,000 个 random third-person camera viewpoints pool
- 每个 trajectory 随机采样 20 个 camera 渲染
- 共 ~40,000 trajectories,19:1 train/val split
- 500 个 evaluation trajectory (每 task 100)

**Table I 关键数据**：

| Coord | Continuous | All | PickC | StackC | SingleYCB | ClutterYCB | EGAD |
|-------|-----------|-----|-------|--------|-----------|------------|------|
| Robot | √ | 45.2% | 71.0% | 62.0% | 30.0% | 15.0% | 48.0% |
| **Camera** | √ | **53.2%** | **88.0%** | 65.0% | **46.0%** | 19.0% | 48.0% |
| Robot | × | 38.6% | 61.0% | 51.0% | 28.0% | 8.0% | 45.0% |
| **Camera** | × | **52.4%** | **80.0%** | 65.0% | **48.0%** | 19.0% | 50.0% |

直觉观察：
- Continuous action space：camera frame 提升 **+8.0%** 整体
- Discrete action space：camera frame 提升 **+13.8%** 整体（52.4% vs 38.6%）
- **Discrete 提升更大**的原因猜测：discrete bin 把 action space 离散化后,model 必须精确选对 bin,坐标 misalignment 会让相邻 bin 的语义混乱,coordinate 对齐带来的收益更明显。
- PickCube 上提升最大（+17% continuous,+19% discrete）,因为 PickCube 对 grasp position 精度敏感,coordinate 对齐直接 fix 了 grasp localization error。
- ClutterYCB 提升最小,因为 clutter 场景下 occlusion 是主要瓶颈,coordinate 对齐帮助有限。

### 4.2 Real Robot: Franka + 3 cameras

Setup：
- Franka Emika Panda 7-DoF + Robotiq 2F-85 gripper
- 3 个 RealSense D435i（https://www.intelrealsense.com/depth-cameras/）:Cam1, Cam2 用于训练,Cam3 留作 zero-shot novel view test
- 15 tasks (Cam1) + 8 tasks (Cam2 with slight perturbation)
- 10-shot setting（每 task 10 demos）

**Table II 关键数据**（10-shot fixed camera + novel camera var）：

| Method | Avg | Avg (var) | Drop |
|--------|-----|-----------|------|
| OpenVLA-OFT | 63.3% | 42.0% | -21.3% |
| π0 | 50.7% | 34.7% | -16.0% |
| Robot Base (Dita-based) | 58.0% | 41.3% | -16.7% |
| **Camera Base (OC-VLA)** | **68.0%** | **54.0%** | **-14.0%** |

直觉：
- **Fixed camera**:OC-VLA 68.0% vs OpenVLA-OFT 63.3%,超越 4.7%,说明 coordinate alignment 比单纯堆参数更有效（OpenVLA-OFT 7B vs OC-VLA 334M）
- **Novel camera**:OC-VLA 54.0% vs OpenVLA-OFT 42.0%,超越 12%,**drop 只有 14% vs 21.3%**
- 这说明 OC-VLA 学到的是 **view-invariant 的 perception-action mapping**,泛化到 unseen viewpoint 的能力本质更强
- π0 (https://arxiv.org/abs/2410.24164) 表现最差,可能因为它是 flow matching 大模型,在小数据 fine-tune 时不适应,且其 action space 设计假设了 robot base frame

**Table III 关键数据**（camera perturbation during training）：

| Method | Avg | 
|--------|-----|
| Robot Base (fixed cam) | 66.3% |
| **Cam Base (fixed cam)** | **77.5%** |
| Robot Base (perturbed cam) | 61.3% |
| **Cam Base (perturbed cam)** | **73.8%** |

直觉：
- 即使训练时 camera 有 perturbation,OC-VLA 仍 73.8% vs 61.3%,优势 +12.5%
- 而 fixed cam 下优势是 +11.2%
- **Perturbation 下优势更大**,印证 paper 的论点：当 data 里 viewpoint 多样时,camera frame supervision 的优势被进一步放大（因为 robot frame 下 model 必须把多种 observation 映射到同一个 action,gradient 冲突更严重）

### 4.3 Ablation（Table IV）

| Coord | #Obs | #Traj | Freeze ViT | All |
|-------|------|-------|------------|-----|
| Robot | 2 | 2 | × | 38.6% |
| Camera | 2 | 2 | × | 52.4% |
| Robot | 2 | 2 | √ | 16.6% |
| Camera | 2 | 2 | √ | 27.8% |
| Robot | 2 | 16 | × | 16.4% |
| Camera | 2 | 16 | × | 39.4% |
| Robot | 3 | 3 | × | 33.0% |
| Camera | 3 | 3 | × | 51.8% |

关键 intuition：
- **Freeze ViT**：性能大幅下降,但 camera frame 优势**保持**（27.8% vs 16.6%,+11.2%）。说明 coordinate alignment 不依赖 vision encoder fine-tune,即 plug-and-play 真的 plug-and-play。
- **Long trajectory (#Traj=16)**：robot baseline 崩到 16.4%,camera 39.4%,**+23% 的巨大优势**。Long horizon task 累积误差放大,coordinate misalignment 导致每个 step 的小误差累积成大失败,OC-VLA 把误差源头消除,优势更明显。
- **More obs (#Obs=3)**：51.8% vs 33.0%,优势 +18.8%。多 frame observation 给 model 更多 viewpoint 线索,但 robot frame 下 model 还是要把多视角 fold 到同一 robot action,camera frame 则直接利用多视角信息。

---

## 5. 联想与延伸思考

### 5.1 与 Embodied Foundation Model 的关系

这个工作让我想到 **embodied pretraining 的一个更深层问题**：foundation model 的 pretraining objective 和 downstream task 的 supervision frame 必须一致。

- CLIP / DINOv2 的 contrastive objective 都在 image/camera space
- 但 VLA 的 imitation loss 在 robot space
- 两个 space 的 "language" 不通,即使 backbone 强,transfer 也打折扣

OC-VLA 把 action 也搬到 camera space,**等于把 imitation loss 翻译成了 vision encoder 母语**,这是本质改进。

类似思路：
- **Diffusion Policy** (https://diffusion-policy.cs.columbia.edu/) 用 visual chunk 但 action 仍在 robot frame
- **3D Diffusion Policy** (https://3d-diffusion-policy.github.io/) 用 point cloud,但 point cloud 本身是 camera frame,所以 action 也是 camera frame,本质上和 OC-VLA 同源
- **SpatialVLA** (https://arxiv.org/abs/2501.15830) 引入 spatial representation,也是想 align perception-action

### 5.2 与 SE(3) Equivariance Network 的关系

公式 4 的 conjugation $\mathbf{T}\mathbf{A}\mathbf{T}^{-1}$ 是 SE(3) 上的 group action。这暗示一个更深的设计：**理想的 VLA 应该是 SE(3)-equivariant 的**。

相关工作：
- **SE(3)-Transformer** (https://arxiv.org/abs/1806.02512)
- **Equiformer** (https://arxiv.org/abs/2106.02916)
- **EquiBot** (https://arxiv.org/abs/2307.09734) 已经在 robot 上验证 equivariance 的好处

OC-VLA 没有显式做 equivariant architecture,只是**通过 data label 变换隐式实现 equivariance**。一个 follow-up 方向：在 transformer 里显式做 SE(3) equivariant,可能进一步提升 cross-view generalization。

### 5.3 Camera Extrinsic 的鲁棒性问题

OC-VLA 假设 camera extrinsic **T 已知且精确**。这在 real-world deployment 是一个隐藏的 fragility：

- 手动 calibration 误差（典型 1-5cm translation, 1-3 度 rotation）
- RealSense 自带的 factory calibration 不一定准
- Camera 被碰歪后 T 失效

可能的解决方向：
- **Self-calibration**：从 image 内容在线估计 T（参考 https://arxiv.org/abs/1904.02067）
- **Joint learning**：把 T 当 latent variable,model 隐式 infer T（但这又回到原来的问题）
- **Camera Pose Estimation as auxiliary task**：让 model 额外预测 T,作为 regularization

### 5.4 与 World Model 的联系

OC-VLA 把 action anchor 到 observation frame 的思想,和 world model 的 next-frame prediction 有哲学上的相似：world model 在 image space 预测,action 也在 image-aligned space 预测,两者**共享 representation**,迁移更顺畅。

参考：
- **DreamerV3** (https://arxiv.org/abs/2301.04104) 在 latent space 预测
- **GAIA-1** (https://arxiv.org/abs/2309.17080) video world model
- **Sora** (https://openai.com/sora) video generation as world simulator

如果 VLA 未来融合 world model,camera-frame action 几乎是必须的,因为 world model 的 prediction 在 image space,action 必须和 image space 兼容。

### 5.5 Limitation 我自己想到的

Paper 没明确讨论的几个点：

1. **Single-camera assumption**:如果 VLA 同时用 wrist camera + third-person camera,两个 camera 的 T 不同,action 该 anchor 到哪个？需要 multi-camera extension,可能 weighted average 或 attention over cameras。

2. **Moving camera**:如果 camera 在 robot 上（wrist-mounted）,T 随 robot pose 变化,这时 camera frame action 等价于 end-effector frame action,反而可能更优。Paper 没讨论这种情况。

3. **Gripper action**:7-dim action 里 gripper 是 1-dim scalar,坐标变换不影响它。但如果 action 包含 finger joint angles（dexterous hand）,这些 joint angle 不在 SE(3) 里,无法用 conjugation 变换,需要分头处理。

4. **Coordinate frame 不变性的极限**:如果 task 本身依赖 absolute world direction（例如 "往北走" "对准地球重力方向"）,camera frame 反而丢失信息。这种情况需要 task-specific 设计。

5. **Numerical stability**:conjugation $\mathbf{T}\mathbf{A}\mathbf{T}^{-1}$ 在 4×4 matrix 上做 inversion,如果 T 接近 singular 会数值不稳定。实际 T 来自 calibration 一般 well-conditioned,但理论上是 risk。

### 5.6 一个 toy experiment 建议验证 intuition

如果你想自己 build intuition,我建议跑一个 minimal experiment：

- 在 ManiSkill2 一个 task 上,只用 single camera fixed view 训练两个 model（robot frame vs camera frame action）,预期两者差距很小
- 然后换一个 train 中没见过的 camera view test,预期 camera frame 显著更好
- 进一步,故意 corrupt T（加 5cm noise）,看 camera frame model 的 graceful degradation 程度

这个 minimal experiment 能直接验证 paper 的核心 claim:**advantage 主要来自 cross-view generalization,不是 single-view fitting**。

---

## 6. 总结：什么时候该用 OC-VLA

| 场景 | 是否推荐 OC-VLA | 原因 |
|------|----------------|------|
| Single camera, fixed setup | 推荐但优势小 | Coordination 不影响,但也不会 hurt |
| Multi-camera, viewpoint diverse data | **强烈推荐** | 直接解决 supervision conflict |
| Cross-embodiment pretraining | **强烈推荐** | 不同 robot 不同 camera,frame 对齐是必须 |
| Wrist camera only | 推荐替代为 end-effector frame | 等价于 camera frame,无 T 变换 |
| Task 依赖 absolute world direction | 谨慎使用 | 可能丢失 world 信息 |
| Dexterous hand with finger joints | 需要扩展 | finger action 不在 SE(3) |

---

## 7. 我的整体评价

**优点**：
- Insight 深刻,直击 VLA 的本质 misalignment 问题
- 方法极简,plug-and-play,无 GPU overhead,无架构改动
- 实验充分,simulation + real + ablation + cross-view 都做了
- 数学严谨,SE(3) conjugation 表达清晰

**改进空间**：
- 没讨论 multi-camera 融合
- 没讨论 wrist camera 等特殊情况
- Camera extrinsic 鲁棒性没分析
- 没和 SE(3) equivariant network 对比

**直觉总结**：OC-VLA 是 VLA 领域一个"小而美"的工作,用一个 4×4 matrix 变换把 perception-action alignment 问题从 architecture level 降到 data preprocessing level,这体现了作者对 robotics learning 的深刻理解。它不会成为 VLA 的银弹,但**应该是未来所有 VLA model 的 default preprocessing step**,就像 image normalization 一样自然。

---

## 8. Reference 链接汇总

**Paper & Code**：
- Droid Dataset: https://droid-dataset.github.io/
- Dita (paper base): https://arxiv.org/abs/2503.19757

**Baseline VLA Models**:
- OpenVLA: https://openvla.github.io/
- π0: https://arxiv.org/abs/2410.24164
- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://arxiv.org/abs/2307.15818
- Octo: https://arxiv.org/abs/2405.12213
- SpatialVLA: https://arxiv.org/abs/2501.15830

**Backbones**:
- DINOv2: https://arxiv.org/abs/2304.07193
- CLIP: https://arxiv.org/abs/2103.00020
- BLIP-2 (Q-Former): https://arxiv.org/abs/2301.12597
- FiLM: https://arxiv.org/abs/1709.07871
- LLaMA2: https://arxiv.org/abs/2302.13971

**Diffusion**:
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- DiT: https://arxiv.org/abs/2212.09748
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

**Benchmark**:
- ManiSkill2: https://maniskill2.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- BridgeData V2: https://rtd-ros.github.io/

**Related Equivariance Work**:
- SE(3)-Transformer: https://arxiv.org/abs/1806.02512
- Equiformer: https://arxiv.org/abs/2106.02916
- 3D Diffusion Policy: https://3d-diffusion-policy.github.io/

希望这些讲解能 build 你对 coordinate frame alignment 在 robot learning 中关键作用的 intuition。这个 paper 的核心 takeaway：**当你发现 model 学不好,先检查 supervision signal 和 input 是不是在同一个 space**。
