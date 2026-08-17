---
source_pdf: DextrAH-RGB.pdf
paper_sha256: 8a595ac9e72c9556511b058ef498737879b022ae91e9aa80c7306f8dd30d322a
processed_at: '2026-08-03T20:57:32-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DextrAH-RGB: 用人话再讲一遍

好，我把刚才那一堆公式和表格"翻译"成更像聊天的语言，重点讲 **intuition**，不堆术语。技术细节还是保留，但用更直白的方式说。

---

## 一句话总结

这篇 paper 做的事：**让机器人用两只"眼睛"（两个普通彩色摄像头）看一眼，就能直接伸手去抓各种没见过的东西，整个过程端到端，不靠 depth camera，不靠预先知道物体形状。**

之前的同类工作基本都得用 depth camera（像 Intel RealSense 的红外模式），这篇第一次纯用 RGB 把这件事干成了，还能 sim2real。

Paper link: https://dextrah-rgb.github.io/

---

## 为什么这事难？先讲背景

Dexterous grasping（灵巧抓取）就是让一个**多指机械手**（这篇用的是 16 个关节的 Allegro Hand）去抓各种东西。这件事难在哪？

**第一难：机械手自由度太高**。16 个手指关节 + 7 个手臂关节 = 23 维 action space。RL 在这么高维空间里探索，很难 converge。

**第二难：要 reactive，不能只 plan 一次**。经典方法（比如 GraspIt!、UniDexGrasp）是"算出一个 static grasp pose 然后执行"，但物体滑了、位姿变了就崩。你要 closed-loop，物体动我也跟着调整。

**第三难：要 generalizable**。不能只对训练集那几个物体 work，要对没见过的物体也 work。

**第四难：sim2real gap**。你在仿真里训得再好，真实世界光照、摩擦、材质全不一样，policy 一下就崩。

DextrAH-RGB 的解法是：**用一堆 trick 组合起来，把这几个难全解决**。下面逐个讲 trick。

---

## Trick 1: 别让 RL 直接控制 23 个关节，而是让它"指挥"一个安全的底层控制器

这是整篇 paper 最关键的 idea，叫 **geometric fabrics**。

Reference: https://arxiv.org/abs/2405.02250

打个比方：你让一个小孩开车，你不会让他直接控制油门和方向盘的每个细节（那样容易撞墙），而是给他一个"导航系统"，他只说"往那开"，系统自己处理避障、限速、保持车道。

Fabric 就是这个导航系统。RL policy 输出的 action 不是一个 raw joint acceleration，而是一个**高层指令**：palm 想到哪去（6-DoF pose）+ 手指想摆成什么 shape（5 维 PCA）。然后 fabric 系统 + 一个 QP（quadratic program）把这个高层指令翻译成真实的 joint acceleration，**同时保证不撞墙、不超 joint limit、不超 jerk 上限**。

这样 RL 只用学 11 维 action（6 palm + 5 PCA），维度从 23 降到 11，探索容易多了。而且即使 RL 输出 garbage，fabric 也会兜底，robot 不会 self-destruct。

### 那 5 维 PCA 是什么？

16 个手指关节太多了，但人手抓东西其实就那么几种"shape"：捏、包握、侧捏、三指捏……这些叫 **eigengrasps**。Reference: https://api.semanticscholar.org/CorpusID:6853822

作者把人手抓取的 motion capture 数据 retarget 到 Allegro Hand 上，然后做 PCA，发现前 5 个主成分能 cover 绝大部分 human-like grasp shape。所以 RL 只控制这 5 维，相当于"选个 grip type + 微调"。

这个 idea 直接继承自前作 DextrAH-G: https://arxiv.org/abs/2407.02274

---

## Trick 2: 两阶段训练——先让 teacher 偷看答案，再让 student 只看图

RL 直接从 RGB 学，sample inefficient 到没法训。所以分两步：

### Stage 1: Teacher FGP

Teacher 在 sim 里用 PPO 训，**输入是 privileged state**：robot joint state + **ground truth object pose** + object identity one-hot + fabric state。

因为 teacher 知道物体在哪（特权信息），RL 很快能学会"怎么抓"。这是在学 **"how to grasp"** 这个 skill。

### Stage 2: Student FGP

Student 输入是 **stereo RGB 图像 + proprioception**，**没有 object pose**。用 DAgger（online imitation learning）把 teacher distill 过来。这是在学 **"how to see"**——怎么从像素推断物体在哪、怎么抓。

Reference DAgger: https://arxiv.org/abs/1011.0686

这就是典型的 **privileged learning** 范式，OpenAI Rubik's Cube（https://arxiv.org/abs/1910.07113）、DexExtreme（https://arxiv.org/abs/2210.13702）都用这个套路。思想就是：让有信息的 agent 先学会 skill，再把这个 skill "翻译"给没信息的 agent。

---

## Trick 3: Reward 设计得很 "诱导"

Teacher 的 reward 一共 4 项，每项都有讲究：

$$
r = w_{hand\_obj} r_{hand\_obj} + w_{obj\_goal} r_{obj\_goal} + w_{lift} r_{lift} + w_{curl} r_{curl}
$$

### $r_{hand\_obj}$：手靠近物体

$$
d_{hand\_obj} = \max_{i \in \{palm, fingertips\}} \|x^i - x^{obj}\|
$$
$$
r_{hand\_obj} = \exp(-10 \cdot d_{hand\_obj})
$$

变量解释：
- $i$：palm 和 4 个 fingertip，共 5 个关键点
- $x^i$：第 $i$ 个关键点 3D 位置
- $x^{obj}$：物体重心位置
- $\|\cdot\|$：欧氏距离
- $d_{hand\_obj}$：5 个关键点里**离物体最远**那个的距离

**关键在 max**：用 max 而不是 min，是逼着"整只手"都靠近，防止 policy 只伸一根手指头去戳物体刷分。

$\exp(-10d)$ 形式的好处：距离远时 gradient 小（不浪费 effort），距离近时 gradient 大（精修）。

### $r_{obj\_goal}$：物体到目标

$$
r_{obj\_goal} = \exp(-\beta_{obj\_goal} \cdot \|x^{obj} - x^{goal}\|)
$$

变量：
- $x^{goal}$：freespace 里一个目标点
- $\beta_{obj\_goal}$：温度系数，ADR 从 15 涨到 20（越大越 sharp）

这个 reward 是"把物体搬到某个位置"。光靠近没用，得搬走。

### $r_{lift}$：物体抬起来

$$
r_{lift} = \exp\left(-\beta_{lift} (x_z^{obj} - x_z^{goal})^2\right)
$$

变量：
- $x_z^{obj}$：物体当前高度
- $x_z^{goal}$：目标高度
- $\beta_{lift}$：温度系数

平方形式：太高太低都不好，得刚好到目标高度。这个 reward 验证"真的抓稳了"，不是滑了一下又掉。

### $r_{curl}$：别提前 curl 手指

$$
r_{curl} = -\beta_{curl} \|q_{hand} - q_{curl}\|^2
$$

变量：
- $q_{hand}$：当前手指关节配置
- $q_{curl}$：某个"张开"的 nominal 配置
- $\beta_{curl}$：从 0.01 涨到 0.05

这是个 **penalty**（负号），防止 policy 还没到物体就把手 curl 起来。相当于 prior："先张开手，到了再 curl"。

---

## Trick 4: Automatic Domain Randomization (ADR) —— 渐进式加难度

Reference: https://arxiv.org/abs/2210.13702

如果一开始就把 sim 里所有 physics 参数 randomize 到最大范围，policy 根本学不动（信号太乱）。如果一直用 fixed 参数，sim2real 就崩。

ADR 的解法：**policy 表现好了就线性加大 randomization range**。

举个 paper Table II 的例子：

| Parameter | Initial | Terminal |
|-----------|---------|----------|
| Robot Static Friction | U(1,1) = 1.0 | U(0.3, 1.2) |
| Object Mass Scaling | U(1,1) = 1.0 | U(0.5, 3) |
| Object Spawn Width | U(0,0) = 0 | U(0, 0.8) |
| Object Measured Position Noise | U(0,0) | U(0, 0.3) |
| $\beta_{obj\_goal}$ | 15 | 20 |

一开始 friction 恒定 1.0（容易学），末期 friction 在 [0.3, 1.2] 随机（逼 robustness）。一开始物体总在固定位置，末期 spawn 范围越来越大。一开始没有观测噪声，末期加 0.3 的 position noise。

**直觉**：像打游戏从 easy 模式慢慢调到 hard 模式，policy 逐步 generalize。

DexExtreme 那篇是 per-parameter 渐进，这篇改成 **所有参数一起 ramp**，更简单，避免某个参数 ramp 太快导致 policy 卡住。

---

## Trick 5: Student 网络设计——为什么不用 ResNet/ViT？

Student 架构：

```
Left RGB (320×240)  ──┐
                      ├─→ CNN [16,32,64,128] + AvgPool → flatten → 32-dim emb
Right RGB (320×240) ──┘                                              ↓
                                              concat(L emb, R emb, proprio) → LSTM(512) → MLP[512,512,256] → action
                                                                                              ↓
                                                                              aux head MLP[512,256] → object pos
```

变量：
- CNN encoder：4 层 conv，filter 数 [16, 32, 64, 128]，ReLU 激活
- Embedding dim：**32**（超级 compact）
- LSTM：512 units
- MLP：3 层 [512, 512, 256]
- 全部用 ELU 激活

为什么不用 ResNet/ViT 这种 pre-trained backbone？paper 给两个理由：

**理由 1**：frozen backbone 学的是 ImageNet 语义特征，不一定是 grasping 最相关特征。小 CNN 从头训能学 task-specific 特征。

**理由 2**：finetune 大 backbone 太贵。gradient tracking 一个 ResNet50 顶几十个这个小 CNN，并行 env 数量大幅下降，训练慢 10 倍以上。

为什么 32 维 embedding 够？因为任务简单——"物体大概在哪、什么 shape 大概如何"，不需要细粒度语义。Stereo 隐式提供 depth，更不需要 monocular depth estimation。

**DenseNet-style skip**：LSTM input 和 output concat 后再喂给 MLP。Reference: https://arxiv.org/abs/1608.06993 和 https://arxiv.org/abs/2010.09163

直觉：gradient 能直接 flow 回去，policy 在 long horizon 任务上更稳。D2RL paper 验证了 dense architecture 在 RL 中显著提升 performance。

---

## Trick 6: KL loss 比 L2 loss 好——因为 variance 自带权重

Student loss：

$$
\mathcal{L} = \mathcal{L}_{action} + \mathcal{L}_{aux}
$$

### $\mathcal{L}_{action}$：KL divergence

$$
\mathcal{L}_{action} = D_{KL}(\pi_{student} \| \pi_{teacher})
$$

因为 teacher 和 student 都是 **diagonal Gaussian**，且 variance 是 fixed 的（不是 learned），KL 展开后常数项 drop，只剩 quadratic 项：

$$
\mathcal{L}_{action} = \sum_i \frac{1}{\sigma_i^2} (\mu_{student}^i - \mu_{teacher}^i)^2
$$

变量解释：
- $\mu_{student}^i$：student 在第 $i$ 维 action 输出的 mean
- $\mu_{teacher}^i$：teacher 在第 $i$ 维 action 输出的 mean
- $\sigma_i^2$：teacher 在第 $i$ 维 action 的 variance（fixed）
- $i$：action 维度 index

**这个公式就是 variance-weighted L2**！

直觉：
- 在 teacher **uncertain** 的维度（$\sigma_i^2$ 大），权重 $\frac{1}{\sigma_i^2}$ 小，student 不用精确 match
- 在 teacher **confident** 的维度（$\sigma_i^2$ 小），权重大，student 必须精确 match

为什么比 plain L2 好？因为 plain L2 把所有维度同等对待，但 teacher 在某些维度上本来就 random（探索噪声），强行 match 这些噪声反而 hurt student。KL 自动告诉 student："teacher 哪一维是 confident 的照抄，哪一维是 noisy 的忽略"。

paper 说 4 个 seed 实验都 confirm KL > L2，这不是偶然，是 variance-weighting 的功劳。

### $\mathcal{L}_{aux}$：辅助预测物体位置

$$
\mathcal{L}_{aux} = \|\hat{x}_{obj} - x_{obj}\|
$$

变量：
- $\hat{x}_{obj}$：student aux head 预测的物体位置
- $x_{obj}$：sim 里 ground truth 物体位置（privileged）

这个 aux head 强迫 student encoder **explicitly 学到物体在哪**。相当于把 teacher 的 privileged input "蒸馏" 到 student latent 里。不加这个 aux head，student 可能学到 shortcut feature（比如某背景颜色对应某物体），加了就逼它学真正的 localization。

---

## Trick 7: Episode 长度的 trick——别让"抓"被"举"稀释

Teacher 训 10 秒 episode（让 RL 充分 explore + 确保稳抓）。但如果 student 也 10 秒，那大部分 timestep 物体已经在空中了，**抓取阶段被稀释**，student 学不好"how to grasp"。

Student 解法：**物体在空中超过 2 秒就 timeout**。

直觉：让 student 在 "trying to grasp" 阶段 spend 更多 fraction of timesteps。但又不能太短，否则 student 没机会学 "grasp failed, retry" 的 recovery behavior。2 秒是个 sweet spot。

---

## Trick 8: Sim2real 全靠 domain randomization + 渲染质量

这是 RGB sim2real 的核心。Isaac Lab 提供 **ray-traced tiled rendering**（实时光线追踪），比传统 rasterizer 渲染质量高一个量级。Reference: https://github.com/isaac-sim/IsaacLab

Randomization 包括：

**Lighting**：
- HDRI 背景 30% 概率换
- 强度 U(1000, 4000)
- 旋转 U(SO(3))

**Object 材质**：
- Texture 来自 Omniverse Asset Library 的 random everyday object（即使 UV mapping 错也行，就是逼 student 学 robust feature）
- Albedo tint, roughness, metallic, specular 全 U(0,1)
- Texture scale U(0.7, 5)

**Robot & Table**：同样 randomize

**Data augmentation**：
- Random background：p=0.5
- Color jitter：p=1.0（每次都做）
- Random blur：p=0.1

直觉：sim 渲染再 real 也比真实 world "太干净"。Randomization 弥补 gap。Color jitter + random background 让 student 不能依赖颜色做识别，必须学 shape。Blur 模拟 motion blur，让 student 对快速运动也 robust。

Reference Synthetica: https://arxiv.org/abs/2410.21153

---

## Trick 9: 部署要 60Hz——CUDA Graph 是关键

Real-world hardware：
- Kuka LBR iiwa 7-DoF arm
- Allegro Hand v4 16-DoF
- 2× Intel RealSense D415（stereo，rigid mount 到 table）
- 单 NVIDIA Jetson Orin

频率：
- Kuka PD control：1 kHz
- Allegro PD control：333 Hz
- Camera stream：60 Hz
- **Policy：60 Hz**（用 CUDA graph capture）

为什么 60Hz 这么重要？Dexterous grasping 是 contact-rich 任务，物体在指间滑了几毫秒 policy 还没反应过来就掉了。30Hz 不够 reactive。

怎么做到 60Hz？用 **CUDA graph capture**：把整个 CNN + LSTM forward 录制成一个 graph，每次 inference 只 launch 一次 GPU kernel，避免每个 op 单独 launch 的 overhead。这是 deployment 工程上的关键 trick。

paper 实测：60Hz 远比 30Hz 表现好。

---

## 实验结果人话版

Table I success rate（11 个 YCB 物体，每个 5 个 pose）：

| Object | DextrAH-RGB | DextrAH-G (depth) |
|--------|-------------|-------------------|
| Pitcher | 20% | 80% |
| Pringles | 100% | 100% |
| Coffee | 80% | 100% |
| Container | 100% | 100% |
| Cup | 60% | 80% |
| Cheezit | 40% | 100% |
| Cleaner | 80% | 100% |
| Brick | 100% | 100% |
| Spam | 60% | 100% |
| Pot | 100% | 100% |
| Airplane | 0% | 60% |

DextrAH-RGB 平均 ~58%，DextrAH-G 平均 ~93%，差 35 个点。

**为什么差这么多？** RGB 比 depth 难学，这是 fundamental tradeoff。但 RGB 优势在 generalizable：透明物体、IR 噪声场景，depth 直接废了，RGB 还能用。

**Failure pattern**：
- **Airplane 0%**：thin wings，stereo 在 thin structure 上 depth estimate 不可靠
- **Pitcher 20%**：handle thin，同样问题
- **Brick/Pot/Container 100%**：大块、容易 wrap，stereo 够用

所以 RGB policy 在 **thin structure** 上有明显短板，这是 stereo vision 的 fundamental limit，paper 也承认。

---

## Limitations 人话版

1. **PCA action space 限制 dexterity**：5 维 PCA 抓住 human grasp modes，但做不了 in-hand manipulation、tool use
2. **Table collision 在 fabric 里硬编码**：小物体贴近 table 时难抓
3. **Two-stage pipeline 麻烦**：teacher → student 训练流程长，未来 single-stage end-to-end RGB RL 是方向
4. **Grasp 不 functional**：pot 抓 base 不抓 handle，要 functional grasp 需要 task-specific reward
5. **Single object scene**：不能处理 clutter

我补充的 limitation：
6. **Stereo baseline 固定**：RealSense D415 baseline ~55mm，对人手大小物体 depth 估计有限
7. **Camera fixed to table**：不能 eye-in-hand，没法 active view
8. **Sim2real 靠 brute force randomization**：未来 differentiable rendering 或 real2sim2real 可能更 efficient
9. **Reward 简单**：只有 lift，没有 grasp quality / force closure 等精细指标
10. **没跟 monocular RGB baseline 对比**：paper 说 stereo > mono 但没数据表

---

## 我会建议的 future direction

1. **Active vision**：camera 装到 wrist 上，policy 主动调整视角
2. **Multi-view fusion**：3+ cameras with different baselines，弥补 thin structure
3. **Foundation model feature**：frozen DINOv2 / ViT feature + 小 task head，或者 LoRA finetune
4. **Functional grasp**：reward 加 grasp affordance 项（handle 朝上、grip axis 对齐）
5. **In-hand manipulation**：当前只 grasp，下一步是 grasp 后 reorient
6. **Multi-object scene**：加 attention mechanism 处理 clutter
7. **Tactile sensing**：加 DIGIT 等 tactile sensor 弥补 vision occlusion
8. **Diffusion policy**：LSTM 换成 diffusion head，可能更好处理 multimodal action distribution
9. **World model**：DreamerV3 风格在 latent space 学 dynamics，sample efficiency 可能更好
10. **VLM guidance**：GPT-4V / Gemini 给语言指令 "grasp the pot by the handle"，policy 条件化在 language 上做 functional grasp

---

## 整篇 paper 的 takeaway

如果只记一句话：**用 fabric 把 action space 安全化 + privileged teacher 蒸馏到 RGB student + 大量 visual randomization + CUDA graph 部署到 60Hz = 第一个能 work 的 end-to-end RGB dexterous grasping sim2real。**

RGB 比 depth 难，tradeoff 是 generalization 好（透明物体、IR 噪声不受影响），但 thin structure 上差。这是 RGB vision-based dexterous manipulation 真正要突破的点。

Paper 本身: https://dextrah-rgb.github.io/
前作 DextrAH-G: https://arxiv.org/abs/2407.02274
Geometric Fabrics: https://arxiv.org/abs/2405.02250
Isaac Lab: https://github.com/isaac-sim/IsaacLab
DAgger: https://arxiv.org/abs/1011.0686
DenseNet: https://arxiv.org/abs/1608.06993
DexExtreme: https://arxiv.org/abs/2210.13702
Synthetica: https://arxiv.org/abs/2410.21153
OpenAI Rubik's Cube: https://arxiv.org/abs/1910.07113
YCB Object Set: http://dx.doi.org/10.1109/MRA.2015.2448951

---

# DextrAH-RGB: Pixels-to-Action Dexterous Grasping with Stereo RGB

非常好的 paper，Andrej。我来深入拆解这篇 NVIDIA + UC Berkeley 的工作，重点是 build intuition about **为什么这个 pipeline 能 work**，以及每个设计选择背后的 rationale。

---

## I. Big Picture: 这篇 paper 解决什么问题？

Dextrous grasping 一直被三类方法 dominate：
1. **Static grasp synthesis**（如 GraspIt!, UniDexGrasp）：找一个 grasp pose，但反应式（reactivity）差，遇到 unseen object 容易崩；
2. **Depth-based visuomotor policy**（如 DextrAH-G, DexPoint）：依赖 depth camera，对透明 / 半透明 / IR 噪声敏感；
3. **CAD-model-dependent methods**（如 DexRepNet）：需要 object CAD model 做 pointcloud registration，generalization 受限。

DextrAH-RGB 的核心 claim：**第一个 end-to-end stereo RGB-based dexterous grasping policy，成功 sim2real transfer 到 complex, dynamic, contact-rich 任务**。绕开 depth 的依赖，直接 pixels-to-action。

Paper link: https://dextrah-rgb.github.io/
ArXiv: https://arxiv.org/abs/2407.02274 (DextrAH-G，前作)
GitHub Isaac Lab: https://github.com/isaac-sim/IsaacLab

---

## II. 整体 Pipeline 架构图解析

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Teacher FGP Training (Simulation)                  │
│  ─────────────────────────────────────────────                │
│  Inputs: robot state + privileged object pose + one-hot obj  │
│  RL algo: PPO                                                │
│  Action space: geometric fabric (6-DoF palm + 5-DoF PCA)     │
│  Reward: hand_obj + obj_goal + lift + curl                   │
│  Domain Randomization: ADR (progressive)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓ distill via online DAgger
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Student FGP Training (Simulation)                  │
│  ─────────────────────────────────────────────                │
│  Inputs: proprioception + stereo RGB (320×240 ×2)            │
│  Supervision: KL(π_student ‖ π_teacher) + aux object pos     │
│  Rendering: Isaac Lab ray-traced tiled renderer               │
│  Randomization: HDRI + materials + data aug                   │
└─────────────────────────────────────────────────────────────┘
                            ↓ zero-shot sim2real
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Real-world Deployment                              │
│  Kuka iiwa 7-DoF + Allegro Hand v4 (16-DoF)                  │
│  2× Intel RealSense D415 (stereo)                            │
│  Jetson Orin @ 60Hz policy (CUDA graph capture)              │
└─────────────────────────────────────────────────────────────┘
```

为什么两 stage？因为 **RL 直接从 RGB 学非常 sample inefficient**，pixel-level exploration signal 太稀疏。先让 teacher 看到特权信息（object pose），把"how to grasp"学到位，然后 student 只需学"how to see"。这是典型的 privileged learning 思路，和 [OpenAI Rubik's cube](https://arxiv.org/abs/1910.07113)、[DexPoint](https://arxiv.org/abs/2211.09423)、[DexExtreme](https://arxiv.org/abs/2210.13702) 一脉相承。

---

## III. Geometric Fabrics: Safe Reactive Action Space

这是从 DextrAH-G 继承的核心，不搞清楚 fabric 你就理解不了为什么 policy 能 safe。

### 1.1 什么是 Geometric Fabric？

Reference: [Van Wyk et al., Geometric Fabrics, 2024](https://arxiv.org/abs/2405.02250) 和 [Fabrics (Ratliff & Van Wyk, 2023)](https://arxiv.org/abs/2309.07368)

Fabric 是一个 **artificial second-order dynamical system** $\ddot{q} = f(q, \dot{q})$，设计成对应某种"几何行为"。它由两部分组合：

- **Geometric term** $f_{geo}(q, \dot{q})$：定义 nominal behavior。关键性质是 **speed-independent**，即系统沿同一条路径运动，速度大小不影响轨迹形状。这意味着即使 RL policy 输出抖动，path 仍然稳定。
- **Forcing term** $f_{force}(q, \dot{q}, t)$：用来 perturb 系统离开 nominal path，常用于 safety（joint limits）或 task guidance（RL signal）。

最终通过 torque law（PD control）映射到真实 robot。

直觉：**Fabric 像一个弹簧曲面 / 势能场**，RL policy 提供推力，但推力被曲面 shape 约束。当推力消失，robot 自然回到 nominal 路径。这避免了 RL 直接输出 joint acceleration 时容易撞 joint limit / 撞桌子的危险。

### 1.2 DextrAH-RGB 中的具体 Fabric 设计

| Component | 类型 | 作用 |
|-----------|------|------|
| Collision avoidance (geometry) | geometric | 默认保持距离，避免碰撞 |
| Collision avoidance (forcing) | forcing | 仅近距离激活，推离碰撞 |
| Elbow-out, fingers-curled attractor | geometric | nominal rest pose，避免 perturbation 被 cancel |
| Joint position limits | forcing | safety critical，硬约束 |
| Palm pose forcing | forcing | RL 输出，目标 6-DoF palm pose |
| PCA finger forcing | forcing | RL 输出，目标 5-dim PCA hand config |

### 1.3 Action Space: 6-DoF palm + 5-DoF PCA

为什么是 16-DoF hand 的 PCA 到 5 维？因为 16 维 raw joint action space 太高，PPO 探索困难。PCA basis 来自 **retargeted human grasp data**，所以 5 维 PCA 抓住了"human-like grasp shape"的主要 modes。这是从 [Ciocarlie et al. eigengrasps](https://api.semanticscholar.org/CorpusID:6853822) 那里继承来的思想。

最后通过一个 **quadratic program (QP)** 把 fabric acceleration 映射到 robot acceleration，同时 respect 加速度和 jerk 上限：

$$
\min_{\ddot{q}} \quad \|\ddot{q} - \ddot{q}_{fabric}\|^2 \quad \text{s.t.} \quad \ddot{q}_{min} \leq \ddot{q} \leq \ddot{q}_{max}, \quad \dddot{q}_{min} \leq \dddot{q} \leq \dddot{q}_{max}
$$

变量解释：
- $\ddot{q}$：robot joint acceleration vector
- $\ddot{q}_{fabric}$：fabric 系统输出的目标 acceleration
- $\ddot{q}_{min/max}, \dddot{q}_{min/max}$：robot 物理上下限

---

## IV. Teacher FGP: State-based RL Training

### 2.1 Network Architecture

- MLP: 2 层 × 512 units
- LSTM: 512 units（带 skip connection，类似 DenseNet 思想）
- Final readout

输入向量包括：
- Measured robot joint position + velocity
- Measured fingertip + palm points position + velocity
- **Object pose**（privileged!）
- Object position goal
- One-hot encoding of object
- Last FGP action
- Fabric 的 position, velocity, acceleration

注意 teacher 看到的是 ground truth object pose，这是关键 privileged 信息，student 看不到。

### 2.2 Reward Function 详解

总 reward：
$$
r = w_{hand\_obj} \cdot r_{hand\_obj} + w_{obj\_goal} \cdot r_{obj\_goal} + w_{lift} \cdot r_{lift} + w_{curl} \cdot r_{curl}
$$

各项含义：

**(1) Hand-object proximity reward**

$$
d_{hand\_obj} = \max_{i \in \{palm\_pos, fingertips\}} \|x^i - x^{obj}\|
$$
$$
r_{hand\_obj} = \exp(-10 \cdot d_{hand\_obj})
$$

变量解释：
- $i$：index over 5 个 hand 关键点（1 个 palm + 4 个 fingertip）
- $x^i$：第 $i$ 个关键点的 3D 位置
- $x^{obj}$：object 的 3D 位置
- $\|\cdot\|$：Euclidean norm
- $d_{hand\_obj}$：**最远**那个 keypoint 到 object 的距离（注意是 max，不是 min！）

为什么用 max 而不是 min？因为 max 鼓励**所有** finger 都靠近 object，而不是只有 1 个 finger 接近。如果用 min，policy 可能只用一根手指头去戳 object 就拿到 reward。用 max 强迫"整只手包围 object"。

指数 $\exp(-10 d)$ 形式的好处是：在 $d$ 小时 gradient 大（精细靠近时强化），在 $d$ 大时饱和（远距离不浪费 gradient）。

**(2) Object-to-goal reward**

$$
r_{obj\_goal} = \exp(-\beta_{obj\_goal} \cdot \|x^{obj} - x^{goal}\|)
$$

- $\beta_{obj\_goal}$：温度系数，ADR 从 $-15$ 线性增到 $-20$（注意 paper 表里写的是负值，可能是 typo 应该是正值，因为 reward 越接近 1 越好，$\beta$ 越大越 sharp）
- $x^{goal}$：freespace 中目标位置

**(3) Lifting reward**

$$
r_{lift} = \exp\left(-\beta_{lift} \cdot (x_z^{obj} - x_z^{goal})^2\right)
$$

- $x_z^{obj}$：object 的 z（垂直方向）坐标
- $x_z^{goal}$：目标高度
- 平方形式：偏离目标高度（无论上下）都惩罚

**(4) Curl regularization**

$$
r_{curl} = -\beta_{curl} \cdot \|q_{hand} - q_{curl}\|^2
$$

- $q_{hand}$：当前 hand joint 配置
- $q_{curl}$：某个 nominal "stretched open" 配置
- $\beta_{curl}$：ADR 从 $-0.01$ 到 $-0.05$

为什么 penalty 让手指头别 curl 太多？因为 hand 在还没 grasp 之前如果就 curl 起来，policy 就没法 wrap around object。这个 reward 相当于 prior："默认张开手指头，等到了 object 附近再 curl"。

### 2.3 Automatic Domain Randomization (ADR)

Reference: [Handa et al., DexExtreme](https://arxiv.org/abs/2210.13702)

ADR 的关键 idea：**不一开始就把所有 randomization 开到最大**，否则 policy 学不动。而是随 policy performance 提升，**linearly ramp** 各项 physics 参数的 range。

Table II 的关键参数初始 / 终值：

| Parameter | Initial | Terminal |
|-----------|---------|----------|
| Robot Static Contact Friction | U(1,1) = 1.0 | U(0.3, 1.2) |
| Robot Dynamic Contact Friction | U(1,1) | U(0.2, 1) |
| Object Mass Scaling | U(1,1) | U(0.5, 3) |
| Object Spawn Width | U(0,0) | U(0, 0.8) |
| Object Spawn Height | U(0,0) | U(0, 1) |
| Object Measured Position Noise | U(0,0) | U(0, 0.3) |
| Robot Joint Friction Coefficient | U(0,0) | U(-10, 10) |
| $\beta_{obj\_goal}$ | -15 | -20 |
| $\beta_{curl}$ | -0.01 | -0.05 |

直觉：初期 friction 是 deterministic 1.0（容易学），末期是 noisy [0.3, 1.2]（逼 robustness）。Object mass 初期正常，末期 [0.5, 3] 倍变化（让 policy 不能依赖精确 mass）。

DexExtreme 之前是 **per-parameter** increment，这篇 paper 改成 **all parameters shifted in tandem**（一起 ramp）。这个 trick 更简单，避免某个参数 randomization 太快导致 policy 卡住。

---

## V. Student FGP: RGB Distillation

### 3.1 Student Network Architecture（DenseNet-style）

```
Left  RGB (320×240×3) ──┐
                       ├──→ CNN Encoder [16,32,64,128] + AvgPool
Right RGB (320×240×3) ──┘      ↓
                        flatten → project to 32-dim embedding (each)
                                  ↓
                        concat(stereo emb L, stereo emb R, proprioception)
                                  ↓
                        LSTM (512 units)
                                  ↓
                        concat(input to LSTM, LSTM output)  ← DenseNet skip
                                  ↓
                        MLP [512, 512, 256]
                                  ↓
                        → action output (palm pose + PCA fingers)
                                  ↓
                        + auxiliary head MLP [512, 256] → object position
```

变量细节：
- **CNN encoder**: 4 conv layers, filter counts `[16, 32, 64, 128]`, ReLU
- **Embedding dim**: 32（非常 compact！）
- **LSTM**: 512 units
- **MLP**: 3 layers `[512, 512, 256]`
- **All activations**: ELU（不是 ReLU，ELU 在 negative region 有非零 gradient，对 policy smoothness 好）

为什么这么 compact 的 CNN encoder，不用 ResNet/ViT pre-trained backbone？

paper 给出两个 reason：
1. **End-to-end gradient 让 encoder 学 task-specific features**，frozen ResNet 学到的 features 是 ImageNet 任务的，不一定是 grasping 最 relevant 的；
2. **如果 finetune 大 backbone，gradient tracking 太贵**，并行 env 数量大幅下降，slow down training。

CNN 这么小（128 个 filter 最多）够吗？答案是够，因为：
- 输入只有 320×240，不需要感受野很大
- Task 是 "see object position + shape 大概"，不需要 fine-grained 语义识别
- Stereo 隐式提供 depth，不需要 monocular depth estimation

### 3.2 DenseNet-style Skip Connections

Reference: [Huang et al., DenseNet](https://arxiv.org/abs/1608.06993) 和 [Sinha et al., D2RL](https://arxiv.org/abs/2010.09163)

paper 说 LSTM 的 input 和 output **concat** 后再喂给 MLP。这个 dense connection 让 gradient flow 更直接，policy 在 long horizon 任务上更稳。在 D2RL paper 里已经验证 dense connections 在 RL 中显著提升 performance。

### 3.3 Loss Function 深度解析

总 loss：
$$
\mathcal{L} = \mathcal{L}_{action} + \mathcal{L}_{aux}
$$

**Imitation loss** (KL divergence)：
$$
\mathcal{L}_{action} = D_{KL}(\pi_{student} \| \pi_{teacher})
$$

由于 teacher 和 student 都是 **diagonal Gaussian**，且 variance 是 fixed（不是 learned）：

$$
D_{KL}(\mathcal{N}(\mu_s, \Sigma_s) \| \mathcal{N}(\mu_t, \Sigma_t)) = \frac{1}{2}\left[\log\frac{|\Sigma_t|}{|\Sigma_s|} - d + \text{tr}(\Sigma_t^{-1}\Sigma_s) + (\mu_s - \mu_t)^T \Sigma_t^{-1} (\mu_s - \mu_t)\right]
$$

由于 $\Sigma_s = \Sigma_t = \Sigma$ (fixed)，前几项是常数，只剩 quadratic 项：

$$
\mathcal{L}_{action} = (\mu_{student} - \mu_{teacher})^\top \Sigma_{teacher}^{-1} (\mu_{student} - \mu_{teacher})
$$

对角高斯进一步简化：
$$
\mathcal{L}_{action} = \sum_i \frac{1}{\sigma_i^2} (\mu_{student}^i - \mu_{teacher}^i)^2
$$

变量解释：
- $\mu_{student}^i$：student policy 第 $i$ 维 action 的 mean
- $\mu_{teacher}^i$：teacher policy 第 $i$ 维 action 的 mean
- $\sigma_i^2$：teacher policy 第 $i$ 维 action 的 variance
- $i$：action dimension index

**关键 insight**：这个 loss 是 **variance-weighted L2**！

- 在 teacher **high variance**（uncertain）的 dimension 上，weight $\frac{1}{\sigma_i^2}$ **小**，student 不需要精确 match
- 在 teacher **low variance**（confident）的 dimension 上，weight **大**，student 必须精确 match

为什么比 plain L2 好？因为 plain L2 把所有 dimension 同等对待，但 teacher 在某些 action dimension 上可能本身就 random（exploration 噪声），强行 match 这些噪声反而 hurt student。KL 自动告诉 student："teacher 在哪一维是 confident 的，照抄；哪一维是 noisy 的，忽略"。

paper 说 4 个 seed 实验都 confirm KL > L2，这不是巧合，是 variance-weighting 的功劳。

**Auxiliary loss**:
$$
\mathcal{L}_{aux} = \|\hat{x}_{obj} - x_{obj}\|
$$

- $\hat{x}_{obj}$：student auxiliary head 预测的 object position
- $x_{obj}$：ground truth object position（sim 中 privileged）

auxiliary loss 强迫 student encoder **explicitly 学到 object position**。这相当于把 teacher 的 privileged input "蒸馏" 到 student 的 latent 里。如果不加这个 aux head，student 可能学到 "shortcut features"（比如某背景颜色对应某 object），加上 aux head 强迫它学真正的 object localization。

### 3.4 Online DAgger

Reference: [Ross, Gordon, Bagnell, DAgger](https://arxiv.org/abs/1011.0686)

DAgger 的精髓：**student 在 rollout 过程中产生的 state 分布**和 teacher 不一样（covariate shift）。如果只做 offline behavioral cloning，student 一旦偏离，看到没见过的 state 就崩。

Online DAgger：student 自己 rollout，**on-policy** 收集 (state, teacher action) 对，然后用 KL loss 训 student。这样 student 见到的 state 分布是它自己的，teacher 只是个 oracle 提供 target。

### 3.5 Episode Length Trick

Teacher 训 10 秒（让 RL 充分 explore + 确保稳抓），但 student 如果也 10 秒，那大部分 timestep object 已经在空中了，**抓取阶段被稀释**。

Student 训练时：**object 在空中超过 2 秒就 timeout**。

直觉：让 student 在 "trying to grasp" 阶段 spend 更多 fraction of timesteps。但又不能太短，否则 student 没机会学 "grasp failed, retry" 的 recovery behavior。2 秒是个 sweet spot。

### 3.6 Visual Randomization + Data Augmentation

Isaac Lab 提供 **ray-traced tiled rendering**，比传统 rasterizer 渲染质量高很多，这是 RGB sim2real 的基础。Reference: [Mittal et al., Orbit](https://arxiv.org/abs/2303.03700), [Synthetica](https://arxiv.org/abs/2410.21153)

Table III 关键参数：
- **HDRI backgrounds**：30% 概率 randomize，强度 U(1000, 4000)
- **Object texture**: 来自 Omniverse Asset Library 的 random everyday object textures（即使 UV mapping 不对也 OK，因为只是逼 student 学 robust features）
- **Material properties**: albedo tint, roughness, metallic, specular 全部 U(0,1)
- **Robot/Table**: 同样 randomize

Data augmentation (Table IV)：
- Random background: p=0.5
- Color jitter: p=1.0（每次都做）
- Random blur: p=0.1（模拟 motion blur）

直觉：sim 渲染再 real 也比真实 world "太干净"。Randomization 弥补这个 gap。Color jitter + random background 让 student 不能依赖颜色做识别，必须学 shape。Blur 让 student 对快速运动也有 robustness。

---

## VI. Experiments 深度分析

### 6.1 Hardware Setup

- **Arm**: Kuka LBR iiwa 7-DoF
- **Hand**: Allegro Hand v4 16-DoF
- **Cameras**: 2× Intel RealSense D415（rigid mount 到 table，形成 stereo config）
- **Compute**: 单 NVIDIA Jetson Orin
- **Frequencies**:
  - Kuka PD: 1 kHz
  - Allegro PD: 333 Hz
  - Camera stream: 60 Hz
  - Policy: 60 Hz（用 **CUDA graph capture** 降低 kernel launch overhead）

CUDA graph 是关键 trick：把整个 CNN+LSTM forward 录制成 graph，每次 inference 只需 launch 一次，避免每个 op 单独 launch 的 overhead。让 60Hz 真正能跑。

paper 实测：60Hz 远比 30Hz 表现好。原因：dexterous grasping 是 contact-rich 任务，频率高才能 reactive，30Hz 时 object 在指间滑了 policy 还没反应过来。

### 6.2 Success Rate Table I 仔细看

| Object | DextrAH-RGB | DextrAH-G (depth) | DexDiffuser | ISAGrasp | Matak |
|--------|-------------|-------------------|-------------|----------|-------|
| Pitcher | 20% | 80% | - | - | 67% |
| Pringles | 100% | 100% | 60% | 60% | 100% |
| Coffee | 80% | 100% | - | - | 67% |
| Container | 100% | 100% | - | 40% | - |
| Cup | 60% | 80% | 60% | 1 | 0% |
| Cheezit | 40% | 100% | 80% | 80% | 0% |
| Cleaner | 80% | 100% | 100% | - | 100% |
| Brick | 100% | 100% | - | - | 100% |
| Spam | 60% | 100% | - | 1 | 0% |
| Pot | 100% | 100% | - | 80% | - |
| Airplane | 0% | 60% | 20% | - | - |

平均一下（忽略 missing）：
- **DextrAH-RGB**: ~58%（11 个 object）
- **DextrAH-G**: ~93%
- **DexDiffuser**: ~64%（5 个 object 上有数）
- **ISAGrasp**: ~57%（5 个 object 上有数）
- **Matak**: ~48%

DextrAH-RGB 比 DextrAH-G 平均低 35 个点，**但是用了 RGB 而非 depth**。这是 core tradeoff：RGB 更 general（不依赖 IR，能处理透明 object）但更难学。

几个观察：
1. **Airplane** 0%：airplane 有 thin wings，stereo vision 在 thin structure 上 depth estimation 不可靠，policy 抓不到。
2. **Pitcher** 20%：pitcher 有 handle，handle thin，同样问题。
3. **Brick / Pot / Container** 100%：这些是大块、容易 wrap 的 object，stereo 就够。

**Failure mode pattern**：RGB policy 在 **thin structures**（handle, wing, edge）上明显不如 depth policy。这是因为 monocular/stereo depth 在 thin structures 上有 ambiguity。

### 6.3 Continuous Running 的妙处

paper 强调 policy 可以 **continuously run** 直到 success 或 unrecoverable failure。这是 recurrent architecture 的好处：LSTM 隐状态可以 accumulate evidence，progressively adapt。

不像 grasp synthesis 方法（一次 plan 失败就重 plan），这里 policy 一直在 reactive control，对象滑了就调整 finger force，看到 object 偏了就 re-approach。这种 closed-loop behavior 才是真正 dexterous。

---

## VII. Limitations 自我剖析

paper 自陈几个 limitation，我觉得分析得挺到位：

1. **PCA action space 限制 dexterity**：5 维 PCA 抓 human-like grasp modes，但没法做 in-hand manipulation、tool use 等更复杂动作。
2. **Table collision avoidance 在 fabric 里硬编码**：导致小 object（贴近 table）难抓。paper 建议未来让 policy 通过 sensor 学这个。
3. **Two-stage pipeline 麻烦**：teacher → student 训练流程长。如果 exploration algorithm 更好（比如 curiosity-driven, 或更好的 intrinsic reward），可能 single-stage end-to-end RL 直接从 RGB 学。
4. **Grasp 不 functional**：pot 是抓 base 而不是 handle。要 functional grasp，需要 task-specific reward，而不只是 "lift object"。
5. **Single object scene**：不能处理 clutter。Clutter 需要 attention mechanism 或者 multi-object reasoning。

我补充几点 paper 没说但很重要的 limitation：

6. **Stereo baseline 是 fixed**：RealSense D415 的 baseline ~55mm，对人手大小 object 的 depth 估计有限。可能 wider baseline 或者多视角 camera 会更好。
7. **Camera mount 是 fixed to table**：不能 eye-in-hand。如果 camera 在 wrist 上，policy 可以 active view，可能解决 occlusion 问题。
8. **Sim2real gap 仍主要靠 domain randomization**：如果未来用 differentiable rendering 或者 real2sim2real，可能更 efficient。
9. **Reward 比较简单**：只有 lift，没有 "grasp quality", "stable grasp", "force closure" 等更精细指标，所以抓法不优雅。

---

## VIII. 跨 paper 的关联联想

### 8.1 与 DextrAH-G 的关系

[DextrAH-G](https://arxiv.org/abs/2407.02274) 是直接前作，几乎完全一样 pipeline，差别只在 student 输入是 depth 还是 RGB。DextrAH-G 用 depth，所以 sim 渲染快（depth 不需要 texture/lighting），domain randomization 简单。DextrAH-RGB 必须 ray-traced rendering + 大量 visual randomization，工程上难很多。

### 8.2 与 DexExtreme 的关系

[DexExtreme](https://arxiv.org/abs/2210.13702) 也是 NVIDIA 的工作，做 in-hand cube manipulation，用 ADR + sim2real。DextrAH-RGB 继承了 ADR 的 idea，但 task 从 in-hand manipulation 变成 arm-hand grasping。

### 8.3 与 OpenAI Rubik's Cube 的关系

[OpenAI Rubik's Cube](https://arxiv.org/abs/1910.07113) 是 dexterous sim2real 的经典，用 5 个 camera + 随机化 + LSTM policy。DextrAH-RGB 用 2 个 stereo camera + randomization + LSTM，思路类似但更 modern（geometric fabrics, Isaac Lab ray-tracing）。

### 8.4 与 Visual Dexterity 的关系

[Visual Dexterity (Chen et al.)](https://www.science.org/doi/10.1126/scirobotics.adc9244) 是 in-hand reorientation，object set 也是 DextrAH-RGB 用的来源之一。

### 8.5 与 Hand-Object Pretraining from Videos 的关系

[Singh et al.](https://arxiv.org/abs/2409.08273) 用 human video pretrain policy，然后 sim2real finetune。这跟 DextrAH-RGB 的 sim-only training 形成对比：human video 提供 prior，sim 提供 scale。

### 8.6 与 Get a Grip 的关系

[Get a Grip](https://arxiv.org/abs/2410.23701) 做 multi-finger grasp evaluation at scale，evaluate 抓得稳不稳。DextrAH-RGB 也可以用类似 metrics。

### 8.7 与 DexDiffuser 的关系

[DexDiffuser](https://arxiv.org/abs/2402.02989) 用 diffusion model 生成 grasp pose，是 generative 方法。和 DextrAH-RGB 的 reactive policy 思路不同，diffusion 适合 offline plan，reactive 适合 closed-loop。

### 8.8 与 Implicit Shape Augmentation (ISAGrasp) 的关系

[ISAGrasp](https://arxiv.org/abs/2210.13638) 用 implicit shape augmentation 让 policy 对 unseen shape robust。DextrAH-RGB 的 object texture randomization 思路类似但更 brute force。

---

## IX. 我对这篇 paper 的整体评价

**Strengths**:
1. 第一个真正 work 的 end-to-end RGB dexterous grasping sim2real，破除"必须用 depth"的迷思；
2. Fabric action space 设计精巧，让 RL policy 既 reactive 又 safe；
3. KL loss with fixed variance 的 variance-weighting 解释很 clean；
4. Episode length trick 解决 distillation 阶段 grasp phase dilution 问题，是个工程细节但很 important；
5. CUDA graph + Jetson Orin 部署，真正 60Hz 跑起来；
6. Visual randomization 全面，从 HDRI 到 material 到 augmentation。

**Weaknesses**:
1. 平均 success rate 比 DextrAH-G 低 35 点，"RGB 比 depth 难" 代价大；
2. Thin structure object（pitcher, airplane）几乎完全 fail，stereo vision 在这些 case 上 fundamental limit；
3. No functional grasp（抓 pot 不抓 handle），泛化到 tool use 还远；
4. Two-stage training 复杂，未来 single-stage end-to-end RGB RL 是方向；
5. 没有跟 monocular RGB baseline 对比（虽然 paper 提了 stereo > mono，但没数据表）；
6. 没讨论 inference latency breakdown（CNN encoder 多耗时？LSTM 多耗时？60Hz 是不是极限？）。

**Future Directions 我会建议**:
1. **Active vision**: 把 camera 装到 wrist 上，policy 可以主动调整视角；
2. **Multi-view fusion**: 3+ cameras with different baselines，弥补 thin structure；
3. **Foundation model feature**: 在 CNN encoder 前加一个 frozen DINOv2 / ViT 做 feature extraction，然后只 finetune 小 task head。或者用 LoRA finetune backbone；
4. **Functional grasp**: reward 加入 "grasp affordance" 项，比如让 handle 朝上、grip axis 对齐；
5. **In-hand manipulation**: 当前只 grasp，下一步是 grasp 后 manipulate，比如 reorient object；
6. **Multi-object scene**: 加入 attention mechanism 处理 clutter；
7. **Tactile sensing**: 加 tactile sensor (e.g., DIGIT) 弥补 vision 在 occlusion 下的不足；
8. **Diffusion policy**: 把 LSTM 换成 diffusion policy head，可能更好的 multimodal action distribution；
9. **World model**: 用 world model (DreamerV3 风格) 在 latent space 学 dynamics，可能 sample efficiency 更好；
10. **VLM guidance**: 用 GPT-4V / Gemini 这种 VLM 给语言指令 "grasp the pot by the handle"，policy 条件化在 language 上做 functional grasp。

---

## X. 关键 Reference 汇总

- DextrAH-RGB project: https://dextrah-rgb.github.io/
- DextrAH-G (前作): https://arxiv.org/abs/2407.02274
- Geometric Fabrics: https://arxiv.org/abs/2405.02250
- Fabrics (Ratliff): https://arxiv.org/abs/2309.07368
- DexExtreme: https://arxiv.org/abs/2210.13702
- Isaac Lab: https://github.com/isaac-sim/IsaacLab
- Orbit (Isaac Lab paper): https://arxiv.org/abs/2303.03700 (note: paper 实际 link 是 IEEE RA-L，但 arxiv 上有)
- Synthetica: https://arxiv.org/abs/2410.21153
- DAgger: https://arxiv.org/abs/1011.0686
- DexPoint: https://arxiv.org/abs/2211.09423
- UniDexGrasp: https://arxiv.org/abs/2303.00938
- UniDexGrasp++: https://arxiv.org/abs/2304.00464
- DexGraspNet: https://arxiv.org/abs/2210.02697
- DexRepNet: https://arxiv.org/abs/2303.09806
- DexDiffuser: https://arxiv.org/abs/2402.02989
- ISAGrasp: https://arxiv.org/abs/2210.13638
- DenseNet: https://arxiv.org/abs/1608.06993
- D2RL: https://arxiv.org/abs/2010.09163
- OpenAI Rubik's Cube: https://arxiv.org/abs/1910.07113
- Visual Dexterity: https://www.science.org/doi/10.1126/scirobotics.adc9244
- Hand-Object Pretraining from Videos: https://arxiv.org/abs/2409.08273
- Agarwal et al. functional grasping: https://arxiv.org/abs/2312.02975
- Get a Grip: https://arxiv.org/abs/2410.23701
- Grasp'D-1M: https://arxiv.org/abs/2306.08132
- YCB Object Set: http://dx.doi.org/10.1109/MRA.2015.2448951
- ViT: https://arxiv.org/abs/2010.11929
- ResNet: https://arxiv.org/abs/1512.03385
- Matak & Hermans (visuotactile): https://arxiv.org/abs/2212.08604
- Eigengrasps (Ciocarlie): https://api.semanticscholar.org/CorpusID:6853822

---

总结一句：这篇 paper 是 **"把 depth 换成 RGB"** 这一 idea 的严肃实现，工程上做得很扎实（fabric action space、KL variance-weighting、ADR、ray-traced sim、CUDA graph 部署），但 RGB 相对 depth 的 fundamental limit（thin structure 上的 ambiguity）依然没解，这是后续 vision-based dexterous manipulation 真正要突破的点。
