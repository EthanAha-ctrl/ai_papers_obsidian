---
source_pdf: RoboTron-Mani All-in-One Multimodal Large Model for Robotic Manipulation.pdf
paper_sha256: 390e221946b22ff574577ef9bc8bf511254eb45e46206cff57e2bdb4c5e5d109
processed_at: '2026-08-12T02:01:22-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 RoboTron-Mani

---

## 这篇论文到底在解决什么问题？

embodied AI 圈子这两年有个尴尬的现象：大家都想做 generalist robot model，一个模型干所有活。Google 搞了个 [Open X-Embodiment](https://arxiv.org/abs/2310.08864)，把好多数据集堆一起训，结果 **RT-1-X 性能比 RT-1 还差**。数据越多反而越差，这在 deep learning 里是反直觉的。

问题出在哪？每个数据集用自己的坐标系、自己的相机、自己的 action 表示方式。RLBench 里机器人往下动是 `z=-0.1`，ManiSkill2 里同样往下动变成 `z=+0.1`，因为它 Z 轴朝下。模型看到这些矛盾信号，学到的就是一锅浆糊。

所以核心矛盾是：**数据量上去了，但数据没对齐，等于没上**。

---

## 他们的核心 insight

同一张桌子、同一个杯子，你用 iPhone 拍和用单反拍，2D 像素完全不一样。但这个东西在 3D 空间里就是同一个东西。

如果模型能直接在 3D 表示上做决策，那不管哪个数据集、哪个相机角度，同一个场景的表示就是一致的。这就从根本上解决了跨数据集对齐问题。

---

## 他们怎么做的

### 第一件事：把 2D 多视角特征"抬"到 3D

UVFormer 这个模块干的活类似自动驾驶里的 [BEVFormer](https://arxiv.org/abs/2203.17270)。它知道每个相机的内外参，通过 cross-attention 把多张 2D 图像的特征"反投影"到一个 $80 \times 80 \times 40$ 的 3D grid 上。

直觉上就是：模型不再看"这张照片里杯子在哪"，而是直接知道"3D 空间里杯子在 $(x=0.3, y=0.1, z=0.5)$"。换相机、换视角、换数据集，这个 3D 表示都不变。

### 第二件事：数据对齐

光有 3D 表示还不够，得把所有数据集的坐标系和 action 表示统一：

- **坐标系**：全部转成 X-right, Y-forward, Z-up
- **Action 表示**：全部用 CRMM（旋转矩阵复合法），避开 Euler 角的 gimbal lock 问题
- **缺失数据补全**：很多数据集没存相机参数，他们花了几百人天去 replay 仿真环境重新渲染出来

这部分很脏很累，但这是让联合训练能 work 的地基。

### 第三件事：多模态输出当辅助任务

模型不仅输出 action，还要预测下一帧图像和 occupancy map。

听起来像多此一举——你又不靠生成的图像做决策。但 Table 2 的 ablation 说明白了：加了这些辅助任务，CALVIN 的平均完成序列长度从 2.37 涨到 3.31。

为什么？因为**强迫模型预测未来会发生什么，模型就必须理解物理世界在怎么变化**。这是在逼模型学 world model 的 representation，而不只是学个 state→action 的映射。

而且 Figure 7 显示生成的图像其实挺糊的，但这不重要。重点是学 representation 的过程，不是生成质量。

### 第四件事：MIM（Modality Isolation Mask）

这个设计很巧妙。不同数据集有的有 depth、有的没 wrist camera、有的有 5 个视角。如果用传统方式训，缺失 modality 会破坏整个 attention pattern。

MIM 在 attention mask 层面把 text、image、action 这些 modality 隔开。训练时某个 modality 缺了，其他 modality 不受影响。推理时只想输出 action，可以把 image 和 occupancy 的输出关掉。

这让模型能在**残缺的数据上训练，在完整的 modality 上推理**，或者反过来。这是 generalist model 必须具备的灵活性。

---

## 结果怎么样

一个 4B 参数的模型，在 5 个数据集上同时评估：

- LIBERO: 91.7%（超过 expert 89.8%）
- RoboCasa: 47.4%（超过 GR00T-N1 的 40.9%）
- CALVIN: 93.8%（持平 MDT）
- Meta-World: 80.1%（持平 PRISE）
- RT-1: 60.0%（接近 RT-2-X 55B 的 60.7%）

要知道所有对比的模型都是**专门在单个数据集上训的 expert**，而 RoboTron-Mani 是一个模型同时干所有事。

CALVIN 上 average sequence length 从 baseline 的 1.77 提到 3.51，翻了一倍。这个 benchmark 是让机器人连续完成 5 个语言指令，能平均做到 3.5 个已经很猛了。

---

## 最让我印象深刻的是什么

**Table 3 的 alignment 对比实验**。

RoboFlamingo（2D 模型）用对齐后的数据训练，性能几乎没变化（LIBERO 69.1%→72.1%）。因为它压根没法在 feature level 利用 alignment。

RoboTron-Mani 用对齐后的数据，LIBERO 从 64.2% 飙到 90.7%，涨了 26 个点。

这说明：**数据对齐本身没用，得有 3D perception 能力才能吃到对齐的红利**。这俩是配套的，缺一不可。这是一个很干净的 causal story。

---

## 局限性也很明显

1. **训练规模偏小**：2.1M samples, 10 epochs, 32 卡 50 小时。跟 LLM 的 scale 没法比。
2. **Real-world 只有 100 个 episode**：10 个任务 × 10 次试验，统计意义有限。
3. **occupancy 监督依赖 point cloud**：仿真里容易拿，真实世界要标 occupancy map 成本高。
4. **主要是 Franka 机器人**：cross-embodiment 能力还没充分验证。

---

## 一句话总结

这篇论文的核心 message：**embodied AI 要走向 generalist，不能只靠堆数据，得让数据在 3D representation level 真正统一**。UVFormer 是实现统一的技术手段，data alignment 是地基，多模态辅助任务是 representation 的催化剂。整个 story 是自洽的。

参考：
- [论文 GitHub](https://github.com/EmbodiedAI-RoboTron/RoboTron-Mani)
- [RoboUniView (UVFormer 来源)](https://arxiv.org/abs/2406.18977)
- [BEVFormer (类似思想)](https://arxiv.org/abs/2203.17270)
- [Open X-Embodiment (对比基线)](https://arxiv.org/abs/2310.08864)

---

# RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation 深度解析

Andrej，这篇来自 Meituan 的论文非常有意思，它试图解决 embodied AI 领域一个根本性的矛盾：**如何让一个 generalist policy 在多个异构数据集上联合训练，同时达到甚至超过 expert models 的性能**。让我从直觉出发，逐层拆解。

---

## 1. 核心问题与动机

当前 robot learning 领域存在两个痛点：

### Pain Point 1: 2D Multimodal Models 不够用
现有的 MLLMs（如 LLaVA, Flamingo）主要处理 2D image-text，但 robot 需要在 **3D physical space** 中交互。直接把 2D 模型套用到 robotics，忽略了 camera parameters、depth、多视角等关键 3D 信息。参考 [OpenFlamingo](https://arxiv.org/abs/2308.01390) 和 [LLaVA](https://arxiv.org/abs/2304.08485)。

### Pain Point 2: Data Heterogeneity
[Open X-Embodiment](https://arxiv.org/abs/2310.08864) 虽然聚合了多个数据集，但：
- 缺少 multi-view images, camera intrinsics/extrinsics, depth maps
- 没有做 space alignment，导致 6D pose 在不同数据集间不一致
- 结果 RT-1-X 性能反而不如 RT-1（见 RT-X paper Table 1）

作者的核心 insight：**3D scene 是不变的，但 2D visual features 会因相机参数而变化**。如果能把多视角 2D 特征统一到 3D 表示，就能实现真正的跨数据集对齐。

---

## 2. RoboTron-Mani 架构详解

### 2.1 整体公式

$$
(O_A, [O_I, O_O]) = \text{RoboTron-Mani}(T, I, Cam)
$$

**变量解释**：
- $O_A$: 输出 action（必需）
- $O_I$: 输出 image（可选，包括 static image $O_{simg}$ 和 wrist image $O_{gimg}$）
- $O_O$: 输出 occupancy map（可选）
- $T$: language instruction
- $I \in \mathbb{R}^{H \times N \times H_{img} \times W \times 3}$: 多视角多帧图像
  - $H$: 时间步数（论文中 $H=12$）
  - $N$: 视角数（$N=3$）
  - $H_{img}, W$: 图像高宽（$256 \times 256$）
- $Cam$: camera parameters（intrinsics + extrinsics）

这个公式的关键在于 **multimodal output 是可选的**，这通过 MIM 机制实现，后面详细讲。

### 2.2 四大核心组件

#### Component 1: Vision Encoder
提取 $H$ 个时间步、$N$ 个视角的观测特征 $F_I^{h,n}$。通常使用 CLIP 预训练的 vision encoder。

#### Component 2: 3D Perception Adapter (UVFormer)

这是从 [RoboUniView](https://arxiv.org/abs/2406.18977) 借来的关键模块：

$$
U_I^h = \text{UVFormer}(Q, X^h, Cam^h)
$$

**变量解释**：
- $Q = \{Pos, Emb\}$: learnable unified view queries
  - $Pos \in \mathbb{R}^{L \times B \times 3P}$: 3D grid 位置编码
  - $Emb \in \mathbb{R}^{L \times B \times C}$: learnable features
- $X^h = \{F_I^{h,n}\}_{n=1}^N$: 第 $h$ 个时间步的 N 个视角图像特征
- $Cam^h = \{Cam^{h,n}\}_{n=1}^N$: 对应的 camera parameters
- $U_I^h \in \mathbb{R}^{L \times B \times C}$: unified view representation

**关键参数**：$L=80, B=80, P=40, C=1024$

这意味着 3D 工作空间被 discretize 成 $80 \times 80 \times 40$ 的 grid，每个 pillar cell 由 $Emb_{l,b} \in \mathbb{R}^C$ 负责。

**Intuition**: UVFormer 本质上是在做 **multi-view feature lifting**，类似 [BEVFormer](https://arxiv.org/abs/2203.17270) 在自动驾驶中的做法。通过 camera parameters 把 2D image features 反投影到 3D space，用 cross-attention 让 3D queries 聚合多视角信息。这样即使不同数据集用不同相机，同一 3D 场景的特征表示是一致的。

#### Component 3: Feature Fusion Decoder (基于 OpenFlamingo)

为什么选择 OpenFlamingo 而不是 LLaVA？因为 **LLaVA 用 auto-regressive 机制处理多帧图像效率低**，而 OpenFlamingo 的 cross-attention 更适合 video/multi-frame 输入。

**文本序列构造**：
$$
T' = \{[T_{img}, T^h, T_{simg}, T_{gimg}, T_{occ}, T_{act}]\}_{h=1}^H
$$

**Token 解释**：
- $T_{img}$: 标记原始图像位置
- $T^h$: 第 $h$ 步的文本指令（长度 $L^h$）
- $T_{simg}$: static image read-out tokens（8 个）
- $T_{gimg}$: wrist image read-out tokens（8 个）
- $T_{occ}$: occupancy read-out tokens（8 个）
- $T_{act}$: action read-out token（1 个）

序列长度：$\sum_h^H (1 + L^h + 8 \times 3 + 1) = \sum_h^H (L^h + 26)$

然后：
$$
F_T = \text{WE}(T')
$$

**Attention Fusion**:
- Query: $F_T^h$（文本特征）
- Key, Value: $U_I^h$（视觉特征）

#### Component 4: Modality-Isolation-Mask (MIM) —— 这是关键创新

MIM 是一个 **KQ mask matrix**，控制不同 modality tokens 之间的 attention 连接。看 Figure 5 Left，dark squares 表示允许 attention，white squares 表示禁止。

**Intuition**: 传统 multimodal model 中，所有 token 都能互相 attend，这导致：
1. 训练时如果某个 modality 缺失，整个序列会受影响
2. 推理时如果想省略某个 output modality，会破坏 attention pattern

MIM 的做法是 **isolate** 不同 modality，让 text、image、action tokens 各自有独立的 attention scope。这样：
- 训练时可以灵活加入/去除 auxiliary modality supervision
- 推理时可以只保留 action output，省略 image/occupancy 生成

这个设计让我想到 [Flamingo](https://arxiv.org/abs/2204.14141) 中的 perceiver resampler 和 gated cross-attention，但 MIM 更进一步，直接在 attention mask 层面做 modality isolation。

### 2.3 Multimodal Decoders

#### (a) Image Decoder
结构简单：2 个 attention decoder layers → image patches → 组装成完整图像。

输出：$O_{simg}^h$（static image）或 $O_{gimg}^h$（wrist image）。

#### (b) Occupancy Decoder
```
LLM 输出特征 → reshape → upsample → 3D conv → 3D occupancy
```

输出：$O_o^h = \{o_{pos}^h, o_{rgb}^h\}$，包含位置和 RGB 颜色。

**有趣的设计**: Occupancy 有两条生成路径：
- $O_{o_v}^h$: 通过 UVFormer 从多视角特征直接生成（visual path）
- $O_{o_m}^h$: 通过 LLM 输出（对应 $T_{occ}$ token）

论文实验发现两者效果相近，默认用 $O_{o_v}^h$。这暗示 **LLM 内部的 3D understanding 和显式的 3D reconstruction 可能是等价的表示**。

#### (c) Action Decoder
用 MLP 或 [DiT](https://arxiv.org/abs/2411.19650) blocks。

输出：
$$
a_{pose}^h = \{\Delta pos_x^h, \Delta pos_y^h, \Delta pos_z^h, \Delta rot_x^h, \Delta rot_y^h, \Delta rot_z^h\}
$$
$$
a_g^h: \text{1-DoF gripper action}
$$

这是 delta 6D pose + 1-DoF gripper，共 7 维 action space。

---

## 3. Training Objective

### 总 Loss
$$
l = l_a + \lambda_{image}(l_{simg} + l_{gimg}) + \lambda_{occ} l_o
$$

**参数**: $\lambda_{image} = 0.1, \lambda_{occ} = 0.1$

**关键**: $l_{simg}, l_{gimg}, l_o$ 可以在对应 modality 不可用时排除，这得益于 MIM。

### Action Loss
$$
l_a = \sum_h \left(\text{MSE}(a_{pose}^h, \hat{a}_{pose}^h) + \lambda_g \text{BCE}(a_g^h, \hat{a}_g^h)\right)
$$

**参数**: $\lambda_g = 0.01$

这里 pose 用 MSE（回归），gripper 用 BCE（分类），$\lambda_g$ 很小说明 gripper loss 权重低。

### Image Loss (L2)
$$
l_{simg} = \sum_h \sum_{pixels} \|O_{simg}^h - \hat{I}_{simg}^{h+1}\|_2^2
$$

预测**下一帧**图像，这是 next-frame prediction 的思想，类似 video prediction。

### Occupancy Loss
$$
l_o = \sum_h \sum_{points} l_{o'}
$$
$$
l_{o'} = \|o_{pos}^h - \hat{o}_{pos}^h\|_2^2 + \lambda_{rgb} \|o_{rgb}^h - \hat{o}_{rgb}^h\|_2^2
$$

**参数**: $\lambda_{rgb} = 0.5$

Occupancy 包含 position 和 RGB，position 权重更高。

---

## 4. RoboData: 数据集融合的工程艺术

这是论文的另一个核心贡献。包含的数据集：

| Dataset | Platform | Robot | Coordinate | Views | Action Repr | Episodes |
|---------|----------|-------|------------|-------|-------------|----------|
| CALVIN | PyBullet | 7-DOF Franka | Right-Forward-Up | Static, Gripper | EADM | 20K |
| Meta-World | MuJoCo | 4-DOF Sawyer | Right-Forward-Up | 6 views | None | 5K |
| LIBERO | MuJoCo | 7-DOF Franka | Forward-Left-Up | 4 views | CRMM | 6.5K |
| RoboMimic | MuJoCo | 7-DOF Franka | Forward-Left-Up | 2 views | CRMM | 1.6K |
| RoboCasa | MuJoCo | 12-DOF Franka | Forward-Left-Up | 5 views | CRMM | 5K |
| ManiSkill2 | SAPIEN | 7-DOF Franka | Forward-Right-Down | 2 views | PCM | 30K |
| RoboCAS | SAPIEN/Isaac | 7-DOF Franka | Forward-Left-Up | 3 views | Absolute | 7.3K |
| RLBench | V-REP | 7-DOF Franka | Forward-Left-Up | 4 views | Absolute | 1.8K |
| Colosseum | PyRep | 7-DOF Franka | Forward-Left-Up | 4 views | Absolute | 2K |

**总计**: 70,000 episodes, 7 million samples

### 4.1 3D Space Alignment

统一目标坐标系：**X-right, Y-forward, Z-up**

不同数据集的变换矩阵，例如 RoboMimic:
$$
W_{Robomimic} = \begin{bmatrix} 0 & 1 & 0 & 0 \\ -1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0.4 \\ 0 & 0 & 0 & 1 \end{bmatrix} W_{Robomimic}^{ori}
$$

这个矩阵做了：绕 Z 轴旋转 90° + Z 方向平移 0.4m。

Workspace 统一限制：$[-0.5, -0.5, 0]$ 到 $[0.5, 0.5, 1]$

**Intuition**: Figure 6 展示了一个经典问题——RLBench 和 ManiSkill2 中机器人同样做"向下运动"，但 action 表示完全相反：
- RLBench: $a_{pose} = [0, 0, -0.1, 0, 0, 0]$
- ManiSkill2: $a_{pose} = [0, 0, 0.1, 0, 0, 0]$

这是因为坐标系定义不同（Z-up vs Z-down）。如果不做对齐，模型会学到矛盾的 action mapping。

### 4.2 Action Representation Alignment

三种 action 表示方法：

#### EADM (Euler Angle Difference Method)
$$
A_t = (p^{t+1} - p^t, r_{euler}^{t+1} - r_{euler}^t)
$$

**问题**: Euler 角有 gimbal lock，大角度旋转时不稳定。

#### CRMM (Composite Rotation Matrix Method) —— 论文选用
$$
A_t = (p^{t+1} - p^t, r_{matrix}^{t+1} \cdot \text{Inv}(r_{matrix}^t))
$$

**优势**: 避免 gimbal lock，可以处理任意复杂旋转组合。

#### PCM (Pose Composition Method)
$$
A_t = (\text{Inv}(R_{matrix}^t) \cdot (p^{t+1} - p^t), \text{Inv}(R_{matrix}^t) \cdot R_{matrix}^{t+1})
$$

**区别**: PCM 是在 robot body frame 下表示 action，CRMM 是在 world frame 下。论文统一用 CRMM，参考 [SRT](https://arxiv.org/abs/2407.12998)。

### 4.3 Missing Data Imputation

很多数据集没有 camera intrinsics/extrinsics，论文的做法是：**replay 原始仿真，重新渲染获取这些参数**。这需要重建仿真环境，工作量巨大（"hundreds of person-days"）。

---

## 5. 实验结果深度分析

### 5.1 Table 1: 多数据集性能对比

| Dataset | Best Expert | RoboTron-Mani | 差距 |
|---------|-------------|---------------|------|
| LIBERO | QueST 89.8% | **91.7%** | +1.9% |
| RoboCasa | GR00T-N1 40.9% | **47.4%** | +6.5% |
| CALVIN | MDT 93.7% | 93.8% | +0.1% |
| Meta-World | PRISE 80.4% | 80.1% | -0.3% |
| RT-1 | RT-2-X(55B) 60.7% | 60.0% | -0.7% |

**关键发现**: RoboTron-Mani 是**唯一一个在 5 个数据集上同时评估的 generalist policy**，其他都是 expert models。在 LIBERO 和 RoboCasa 上甚至超过了 expert，这非常 impressive。

### 5.2 Table 2: Ablation Study (CALVIN ABCD)

| ID | FFA | Image | UVFormer | OCC | Avg Len |
|----|-----|-------|----------|-----|---------|
| 1 | × | × | × | × | 1.77 |
| 2 | √ | × | × | × | 2.37 |
| 3 | √ | √ | × | × | 3.13 |
| 4 | √ | × | √ | × | 2.88 |
| 5 | √ | √ | √ | × | 3.21 |
| 6 | √ | × | √ | √ | 3.18 |
| 7 | √ | √ | √ | √ | 3.31 |
| 7* | √ | √ | √ | √ (DiT) | **3.51** |

**逐行分析**：

1. **Baseline → +FFA** (1.77 → 2.37): Frame-by-Frame Action 让模型每帧都输出 action，而不是只在最后一帧。这几乎提升了 34%，说明 **dense supervision** 非常关键。

2. **+FFA → +Image** (2.37 → 3.13): 加入 next-frame image prediction，提升 32%。这印证了 **predictive auxiliary task** 能学到更好的 representation。

3. **+FFA → +UVFormer** (2.37 → 2.88): 3D perception 带来 21% 提升。注意这里 Image 和 UVFormer 是 alternatives，Image 提升更大（3.13 vs 2.88）。

4. **+Image+UVFormer** (3.21): 两者结合有叠加效果，但不是简单相加。

5. **+OCC** (3.18 vs 2.88): 对比 row 4 和 row 6，OCC 带来 10% 提升。

6. **All + DiT** (3.51): DiT action head 比 MLP 更强，能建模 action distribution 的多模态性。

**最重要的 insight**: Figure 7 显示生成的 image 和 occupancy 质量并不理想（模糊），但对 action performance 帮助巨大。这说明 **auxiliary modality 的价值在于 representation learning，而不是 generation quality**。

### 5.3 Table 3: Data Alignment 的重要性

| Model | Aligned? | LIBERO | RoboCasa | CALVIN | Meta-World |
|-------|----------|--------|----------|--------|------------|
| RoboFlamingo* | No | 69.1% | 15.8% | 41.7% | 63.2% |
| RoboFlamingo* | Yes | 72.1% | 15.6% | 43.8% | 65.3% |
| RoboTron-Mani | No | 64.2% | 27.0% | 74.7% | 79.3% |
| RoboTron-Mani | Yes | **90.7%** | **30.6%** | **91.0%** | 78.6% |

**关键观察**：

1. **RoboFlamingo 对 alignment 不敏感**（LIBERO +3%, RoboCasa -0.2%）：因为它用 2D features，无法真正做 input space alignment。

2. **RoboTron-Mani 对 alignment 极其敏感**：
   - LIBERO: 64.2% → 90.7% (+26.5%)
   - CALVIN: 74.7% → 91.0% (+16.3%)
   - RoboCasa: 27.0% → 30.6% (+3.6%)

3. **Meta-World 例外**：alignment 前后几乎无变化（79.3% vs 78.6%），因为 Meta-World 只有 3 个 position 变化，没有旋转，且坐标系本就一致。

**Intuition**: 3D perception 能力是 alignment 的前提。没有 3D input，alignment 无法在 feature level 体现；有了 3D input，alignment 能让多数据集的 3D 表示真正统一。

### 5.4 Table 4: Real-world 实验

| Model | Easy | Medium | Hard |
|-------|------|--------|------|
| RoboTron-Mani | **82.5%** | **70.0%** | **46.6%** |
| RoboUniView* | 75.0% | 36.6% | 23.3% |
| RoboFlamingo* | 35.0% | 6.6% | 6.6% |

**Hardware setup**:
- Robot: Dalu mobile base + UR3 arm
- Gripper: Robotiq two-finger
- Cameras: Intel D435 (wrist) + Orbbec Gemini Pro (static)
- Compute: NVIDIA RTX 3090
- Communication: ROS1

10 个任务分三个难度，RoboTron-Mani 在 hard tasks 上优势最明显（46.6% vs 23.3%），说明 **3D perception + multimodal output** 对复杂任务特别重要。

### 5.5 Table 6: 与 OpenVLA 对比 (CALVIN)

| Method | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg Len |
|--------|--------|--------|--------|--------|--------|---------|
| OpenVLA (LoRA) | 78% | 55% | 29% | 17% | 8% | 1.86 |
| RoboTron-Mani | 81% | 54% | 37% | 25% | 16% | 2.15 |

**公平条件**: window size = 1，RoboTron-Mani 从 scratch 训练，OpenVLA 用官方权重 fine-tune。

**关键**: Task 3-5 的差距越来越大（29→37, 17→25, 8→16），说明 RoboTron-Mani 能捕捉**更长时序依赖**。Avg Len 2.15 vs 1.86，提升 15.6%。

---

## 6. 训练细节

从 supplementary material Section 8：

```
Model: 4B parameters, bf16 precision
Training: 10 epochs, 2.1M samples
Hardware: 32 × 80G-A100 GPUs
Time: ~50 hours
Optimizer: AdamW
LR schedule: cosine annealing, 1e-4 → 1e-6
```

**参数配置**:
- $H=12$ (time steps)
- $N=3$ (views)
- Image size: $256 \times 256$
- 3D grid: $L=80, B=80, P=40$
- Feature dim: $C=1024$
- Loss weights: $\lambda_{image}=0.1, \lambda_{occ}=0.1, \lambda_g=0.01, \lambda_{rgb}=0.5$

---

## 7. 与相关工作的定位

### vs. Open X-Embodiment
- OXE 缺 3D info，RoboData 补全
- OXE 无 space alignment，RoboData 做 alignment
- OXE 的 RT-1-X < RT-1，RoboTron-Mani > experts

### vs. OpenVLA / Octo / HPT
这些是 VLA models，主要用 2D features，需要 fine-tune 才能在特定数据集上 work。RoboTron-Mani 是 zero-shot generalist，不需要 fine-tune。

### vs. RT-2
RT-2 是 55B 参数的大模型，RoboTron-Mani 只有 4B，但在 RT-1 benchmark 上 60% vs 60.7%，几乎持平。

### vs. π0 / GR00T-N1
这些是更新的 VLA models，但仍然主要基于 2D features。RoboTron-Mani 的 3D perception 是差异化优势。

---

## 8. 我的思考与 Intuition

### 8.1 为什么 3D Perception 是关键？
论文反复强调：**同一 3D 场景用不同相机拍，2D features 不同，但 3D features 一致**。UVFormer 本质上是在做 **view-invariant representation learning**，这和 [NeRF](https://arxiv.org/abs/2003.08934) 的思想类似——把多视角信息融合到统一的 3D 表示。

### 8.2 为什么 Multimodal Output 有用？
Table 2 显示，即使生成的 image 和 occupancy 质量很差（Figure 7 模糊），但对 action 帮助巨大。我的解释：

1. **Representation bottleneck**: 强制模型预测下一帧 image 和 occupancy，迫使 LLM 内部学到 **world model** 的 representation，而不只是 action mapping。

2. **Multi-task learning effect**: Auxiliary tasks 起到 regularizer 作用，防止模型对 action 过拟合。

3. **3D grounding**: Occupancy prediction 直接监督模型理解 3D geometry，这对 manipulation 至关重要。

### 8.3 MIM 的深层意义
MIM 不只是技术 trick，它解决了一个根本问题：**multimodal data 的 incompleteness**。不同数据集有不同 modality：
- RT-1 无 depth
- ManiSkill2 无 wrist camera
- RoboCasa 有 5 个视角

MIM 让模型能**在有缺失的数据上训练，在完整的 modality 上推理**，或反之。这是迈向真正 generalist 的关键。

### 8.4 局限性
1. **训练规模有限**: 2.1M samples, 10 epochs，相比 LLM 的 scale 仍然小
2. **Real-world 实验规模小**: 10 tasks × 10 trials = 100 episodes
3. **Sim-to-real gap**: 主要验证在仿真，真实世界只有 UR3 单臂
4. **OCC supervision 依赖**: 需要 point cloud，真实世界获取成本高

### 8.5 未来方向
1. **Scaling**: 增大数据和模型规模
2. **More embodiments**: 目前主要是 Franka，需要扩展到更多 robot types
3. **Self-supervised OCC**: 减少对标注 occupancy 的依赖
4. **Real-world 3D supervision**: 用 depth camera 自动生成 occupancy labels

---

## 9. 总结

RoboTron-Mani 的核心贡献是**将 3D perception 融入 MLLM 框架，并通过 data alignment 实现真正的跨数据集联合训练**。关键设计：

1. **UVFormer** 做 3D feature unification
2. **MIM** 做 flexible multimodal training/inference
3. **Multimodal output** 做 representation enhancement
4. **RoboData** 做 space + action alignment

这个工作让我看到 embodied AI 的一个重要趋势：**不是简单堆数据，而是让数据在 representation level 真正统一**。3D perception 是实现这一统一的关键桥梁。

参考链接：
- [论文 GitHub](https://github.com/EmbodiedAI-RoboTron/RoboTron-Mani)
- [OpenFlamingo](https://arxiv.org/abs/2308.01390)
- [RoboUniView](https://arxiv.org/abs/2406.18977)
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)
- [CALVIN benchmark](http://calvin.cs.uni-freiburg.de/)
- [CogACT (DiT action head)](https://arxiv.org/abs/2411.19650)
- [BEVFormer (类似的 3D lifting)](https://arxiv.org/abs/2203.17270)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [π0](https://arxiv.org/abs/2410.24164)
- [GR00T N1](https://arxiv.org/abs/2503.14734)
