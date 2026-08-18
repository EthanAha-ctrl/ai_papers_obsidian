---
source_pdf: ACE-Ego-0 Unifying Egocentric Human and Robotic Data for.pdf
paper_sha256: a072e85857c536bfd691dd915f6bd8d8fc98ad5f421e1ac225a541ff46ff681f
processed_at: '2026-08-17T23:46:17-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ACE-EGO-0: 给 Karpathy 的人话版深度解读

Andrej, 这篇 paper 本质上是在解决 VLA model 的 data scaling 瓶颈。目前 robot teleoperation data 收集成本极高，而 YouTube 或者 Ego4D 这种 egocentric human video 数据量极其庞大且免费。直接把 human video 和 robot data 混在一起训练会崩掉，因为空间坐标系不同、机械臂形态不同、控制频率不同，最关键的是 human hand 重建出来的 action label 杂讯极大。ACE-EGO-0 的核心贡献就是把这两类异质数据通过 Spatial/Structural/Temporal 三个维度的对齐放到同一个 representation space 里，同时用 Reliability-Aware loss 把 noisy 的人类 data 降级为 auxiliary supervision，让 clean 的 robot data 主导 action expert 的学习。

下面我为你拆解其中的硬核技术细节，并 build 一下这背后的 intuition。

---

## 1. 硬核拆解：四个维度的对齐

### 1.1 Spatial Alignment: Camera-Space Action (公式 1, 11)

**人话**: 机器人的 action 通常定义在它自己的 base frame (比如底盘中心)。这里强行把所有 action 转换到 head-camera 的坐标系下。Policy 直接预测“相机视角下的相对位移”。

**技术细节**:
$$p_{\mathrm{cam}} = R_{\mathrm{cam} \leftarrow s} p_s + t_{\mathrm{cam} \leftarrow s}$$
- $p_s$: end-effector 在 source frame (robot base) 的 position
- $R_{\mathrm{cam} \leftarrow s}, t_{\mathrm{cam} \leftarrow s}$: 从 source frame 到 head-camera frame 的 rotation 和 translation (通过相机标定获得)
- $p_{\mathrm{cam}}$: 转换后用于训练的 camera-space position

**Intuition**: 视觉输入本身就是相机拍的，如果在相机坐标系下预测 action，perception 和 action 就处于同一个空间。VLM backbone 不需要去隐式学习复杂的 world-to-camera 变换。部署新 robot 时，只需要替换一下相机外参矩阵就能把预测值转回 robot base frame。

### 1.2 Structural Alignment: Morphology Token (公式 2, 17-19)

**人话**: 不同的 robot (比如单臂、双臂、人形) 骨骼结构完全不同，甚至人类的手也没有物理 end-effector。这里给 action expert 注入一个 morphology token 告诉它“你现在是谁”。

**技术细节**:
$$h_{\mathrm{morph}} = \begin{cases} P_{\mathrm{morph}}(E_{\mathrm{urdf}}(\mathcal{G}_r)), & \text{robot source } r \\ P_{\mathrm{surr}}(e_d), & \text{human source } d \end{cases}$$
- $\mathcal{G}_r$: robot $r$ 的 URDF kinematic graph
- $E_{\mathrm{urdf}}$: 对 URDF graph 做 message passing (公式 17) 提取出的全局 body summary $z_{\mathrm{body}}$ 和左右臂 chain summary $z_{\mathrm{chain}}$
- $e_d$: 为 human video dataset $d$ 随机初始化并端到端学习出的 surrogate embedding
- $P_{\mathrm{morph}}, P_{\mathrm{surr}}$: 投影网络，把 robot 和 human 的结构信息映射到同一个共享的 morphology token 空间

**Intuition**: Morphology token 只在 action decoding 阶段注入，VLM backbone 完全 embodiment-agnostic。这样 VLM 学到的“看到杯子要去抓”的 visual-language 语义是跨 robot 共享的，只有在最后输出 action 时才根据结构特征进行条件约束。

### 1.3 Temporal Alignment: Time-Aligned Action Chunking (公式 3, 4, 5)

**人话**: 不同数据集的控制频率不一样 (10Hz 到 30Hz)。如果都预测未来 40 个 step，10Hz 数据预测的是 4 秒，30Hz 数据预测的是 1.33 秒，物理时间跨度根本对不上。这里统一按照物理时间来切 chunk。

**技术细节**:
$$H_d = \mathrm{round}(f_d T^\star)$$
- $f_d$: dataset $d$ 的控制频率
- $T^\star = 2$ 秒: 目标物理时间窗口
- $H_d$: dataset $d$ 的 action chunk 步数

为了解决变长 chunk 在同一个 batch 里的 padding 开销和 gradient 不稳定，按照 task cluster、归一化的 episode phase $\phi$ (公式 4) 和 horizon bucket $b_H$ 进行分桶采样 (公式 5)。

**Intuition**: 必须让模型预测的“未来”在物理时间上是等长的，否则 policy 在不同 dataset 之间切换时会面临时间尺度错乱的灾难。

### 1.4 Quality Alignment: Reliability-Aware Loss (公式 6, 8, 20) —— 最核心的贡献

**人话**: Human video 重建出来的 action label 充满杂讯 (wrist 旋转漂移、遮挡导致的手抖)。如果用标准的 L2 loss 去硬拟合，这些杂讯会直接污染 action expert。这篇 paper 的核心 insight 是：把人类 data 当作“不可靠的弱监督”，只让它监督最可靠的维度（如 wrist xyz position），并且用 Huber loss 钝化 outlier。

**技术细节**:
时空可靠性权重分解为三层 (公式 20):
$$W_{t,j} = \rho_j \cdot w_{\mathrm{data}}(d, h(j)) \cdot w_{\mathrm{step}}(t, h(j))$$
- $\rho_j \in [0,1]$: **静态 channel-level 先验**。Position channels $\rho=1.0$ (可靠)，Rotation/gripper channels $\rho=0.001$ (极其不可靠，相当于被 mask 掉)
- $w_{\mathrm{data}} \in [0.25, 1.0]$: **Dataset-level 先验**。根据该 dataset 通过 sanity filter 的比例和 median normalized jerk 估计
- $w_{\mathrm{step}}(t, h)$: **Step-level 动态平滑因子**。通过计算一阶差分 $\Delta p_t$ (speed) 和二阶差分 $\Delta^2 p_t$ (jerk) (公式 22)，如果超过该 dataset 95th percentile 阈值，则用指数衰减 $\max\{w_{\mathrm{min}}, \exp[-\alpha(q-1)]\}$ 进行 down-weight (公式 24)。

人类辅助损失 (公式 8):
$$\mathcal{L}_{\mathrm{hux}} = \mathbb{E}_{s, \epsilon} \frac{1}{Z} \sum_{t, j} M_{t,j} W_{t,j} \mathrm{Huber}_\beta\big(\hat{v}_\theta(\mathbf{a}_s, s)_{t,j} - (\tilde{\mathbf{a}} - \epsilon)_{t,j}\big)$$
- $\tilde{\mathbf{a}}$: 经过 $W_{\mathrm{smooth}}=3$ 帧 window 时序平滑后的人类 action target
- $Z = \sum_{t,j} M_{t,j} W_{t,j}$: 归一化因子，使得 loss 对有效帧数量 scale-invariant
- $\mathrm{Huber}_\beta$: $\beta=1.0$ 的 Huber loss，对 outlier 比 L2 更 robust
- 总 loss 中 $\lambda_{\mathrm{hux}} = 0.1$，保证高保真 robot data (公式 7 的 primary flow-matching loss) 主导模型。

**Intuition**: 人类 data 的价值在于提供广泛的行为覆盖，这就像在低维空间里画出了 action 的边界。只要 wrist position 大致是对的，模型就能学到“看到咖啡杯要伸手过去”的语义先验。你不需要强迫模型去学习人类手指抖动的杂讯。让 clean robot data 去锚定精度，让 noisy human data 去拓展广度。

---

## 2. Egocentric Video-to-Action 转换 Pipeline

**人话**: 怎么从 YouTube 视频里提取出可以用来训练机器人的 action label？这里设计了一个五阶段 pipeline，把 2D 视频变成 22 维的 camera-space action vector。

**流程解析**:
1. **Dataset Curation**: 筛选 4-30 秒的 clip，包含交互动作。
2. **Video Selection**: 用 face detection 滤掉非第一人称视角，用 image captioning 滤掉没有 manipulation verb 和 object noun 的片段。
3. **3D Hand Reconstruction**: 
   - 用 SAM3 tracker 拿到连续的 hand bounding box。
   - 用 HaMeR 重建 MANO 参数 (手部 mesh)。
   - 因为逐帧重建有 depth ambiguity 和 jitter，进行两阶段 L-BFGS 全局轨迹优化 (公式 10): $\mathcal{L}_{\mathrm{smooth}} = \mathcal{L}_{\mathrm{reproj}} + \lambda_{\mathrm{tv}} \sum_t \|\mathbf{t}_{t+1}^{\mathrm{global}} - 2\mathbf{t}_t^{\mathrm{global}} + \mathbf{t}_{t-1}^{\mathrm{global}}\|_2^2$。这里 $\lambda_{\mathrm{tv}}$ 控制时序平滑度，惩罚二阶差分。
4. **Action Parameterization**: 取 wrist 作为 origin，palm 法向量构建 hand-centric frame，转成 6D continuous rotation (公式 12, 13)。Gripper openness 用 thumb-to-palm 距离线性归一化到 $[0.04, 0.10]$ m 匹配真实机械臂夹爪。
5. **Quality Control**: NaN filter、Static filter (无运动能量的丢弃)、Spike filter (瞬移超 3$\sigma$ 的丢弃)、Bimanual filter (双臂行为异常的丢弃)。

---

## 3. 实验数据与 Ablation 深度解读

### 3.1 Pretraining Data Pool (Table 1)

数据池总计 **6,013.7+ 小时**：
- **Human video**: 1,478.9 hours (1.49M episodes) —— 来自 Ego4D, EgoExo4D, EPIC-KITCHENS-100, HOI4D, EgoDex, Xperience-10M。
- **Robot data**: 4,534.8+ hours —— 包含 AgiBot Alpha/Beta, Galaxea R1Lite, AgiBot DigitalWorld, RoboCasa Tabletop, 以及自采的 1800+ hours Galbot。

### 3.2 Ablation: 证明 Reliability-Aware 的绝对重要性 (Figure 5b)

在 RoboCasa GR1 TableTop 上做 component ablation:
- Full ACE-EGO-0: **72.8%**
- 去掉 Morphology tokens: 70.9% (-1.9%)
- 去掉 Time-aligned chunking: 71.7% (-1.1%)
- **去掉 Reliability-aware human loss**: **69.2% (-3.6%)** —— 降幅最大！

这证实了前面的 intuition：如果你把 noisy pseudo-action 当成 clean robot label 一样去用标准 MSE/Flow matching loss 硬拟合，模型会被杂讯带偏，效果反而最差。

### 3.3 Data-Scarce Fine-Tuning (Section 5.5, Figure 6)

这是最能体现 human data 价值的实验。
在 Sweep Cubes 任务上，只有 34 条 robot demo (45.8K frames)：
- 仅用 robot data 微调: **10%** success rate。
- 加入 419 条 task-matched human video (117.5K frames): **40%** success rate (4× 提升)。

**Workspace 覆盖度可视化**:
- 34 条 robot demo 的 end-effector 凸包面积: **0.062 m²**
- 419 条 human video 的凸包面积: **0.296 m²** (4.8× 更大)

**Intuition**: Robot teleoperation 数据往往集中在非常狭窄的空间区域内，模型极易过拟合。Human video 提供了广阔的 workspace coverage，即使 label 有杂讯，这种“广覆盖”作为正则化项极大地缓解了过拟合。这就像用很低分辨率的广角镜头去看世界，虽然看不清细节，但能知道世界的大致轮廓，避免模型在局部死胡同里打转。

### 3.3 仿真与真实世界结果 (Table 3, 4, Figure 5a)

- **RoboCasa GR1 TableTop (24 tasks)**: ACE-EGO-0 达到 **72.8%** avg success，超越 GR00T-N1.6 (47.6%), JoyAI-RA (63.2%), DIAL (70.2%)。
- **RoboTwin 2.0 (50 tasks, bimanual)**: Easy split **91.12%**, Hard split (强 domain randomization) **90.62%**。Hard split 相比 Easy 仅下降 0.5%，而基线 $\pi_{0.5}$ 下降了 6% (82.74% -> 76.76%)。说明 camera-space action representation 对视觉扰动极其鲁棒。
- **Real-Robot (ARX bimanual, 6 tasks)**: ACE-EGO-0 平均 **78.3%**，超越 $\pi_{0.5}$ (71.7%) 和 GR00T-N1.7 (35.6%)。在 Scoop Coffee 这种需要双臂紧密协作的任务上，达到 86.7%，比 $\pi_{0.5}$ 高出 16.7%。

---

## 4. 总结：构建 Intuition

Andrej, 这篇 paper 的核心逻辑可以总结为一个经典的 ML 课题：**如何有效利用便宜但 noisy 的数据来增强昂贵但 clean 的数据？**

它的解法非常工程化且 elegant：
1. **Representation 层面**: 消除一切坐标系、形态、频率的物理不一致性，统一投射到一个无量纲的、以自我观察为中心的 camera-space 中。
2. **Supervision 层面**: 绝对不把弱监督和强监督等权对待。通过 channel prior、dataset prior、step smoothness 三级降维，把 noisy human data 剥离到只负责提供大致的 position 轨迹先验。让 robot data 去负责锚定 precise action。

这种 design pattern 非常适合未来 foundation model 的 scaling。当我们要把 web-scale 的 video data 喂给 robot model 时，这种 reliability-aware objective 是不可或缺的 safety guard。

---

## 5. 参考链接

- **ACE-EGO-0 Project**: https://acerobotics-vla.github.io/ACE-Ego/
- **ACE-EGO-0 Code**: https://github.com/ACERobotics-VLA/ACE-Ego
- **$\pi_0$ (Flow Matching VLA baseline)**: https://arxiv.org/abs/2410.24164
- **$\pi_{0.5}$ (Open-world VLA)**: https://proceedings.mlr.press/v305/black25a.html
- **GR00T N1 (NVIDIA Humanoid Model)**: https://arxiv.org/abs/2503.14734
- **HaMeR (3D Hand Reconstruction)**: https://arxiv.org/abs/2306.15419
- **Ego4D Dataset**: https://arxiv.org/abs/2110.07058
- **DIAL (Latent World Model Baseline)**: https://arxiv.org/abs/2603.29844
- **Zhou et al. (6D Rotation Representation)**: https://arxiv.org/abs/1812.07035
- **RoboTwin 2.0 Benchmark**: https://arxiv.org/abs/2506.18088

---

# ACE-EGO-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining 深度解析

## 1. 背景与动机

这篇 paper 来自 ACE Robotics + CUHK MMLab + SJTU + THU 团队, 项目主页: https://acerobotics-vla.github.io/ACE-Ego/ , 代码: https://github.com/ACERobotics-VLA/ACE-Ego 。核心要回答的问题是: VLA (Vision-Language-Action) model 如何同时利用大规模 egocentric human video 和 multi-embodiment robot demonstration 进行 pretraining。

当前 VLA 领域的 scaling 受限于 robot data 收集成本 (teleoperation 慢且贵), 而 human egocentric video 极其便宜且覆盖广。但把这两类异质数据直接混合训练会遇到两个根本性难题:

**Problem 1: Representation Heterogeneity**
- Spatial: robot end-effector pose 在 base/world frame, MANO hand pose 在 local frame
- Structural: 不同 robot 有不同 kinematic chain (GR1 humanoid, Galaxea wheeled, Galbot mobile bimanual)
- Temporal: 控制频率从 10Hz 到 30Hz 不等, 同样的 40 step chunk 覆盖的物理时间不同
- Quality: robot sensor 是 high-fidelity, human video reconstruction 是 noisy pseudo-action

**Problem 2: Supervision-Quality Mismatch**
Prior work (EgoMimic, EgoVLA, HumanPlus) 把 reconstructed hand trajectory 直接当作 robot action label 喂进 BC/diffusion loss, 这相当于强迫 policy 拟合重建 pipeline 的 artifact 和 jitter。

ACE-EGO-0 的核心 insight: 这两个 problem 需要分别解决, representation 用统一 interface, quality 用 reliability-aware weighting 把 human data 降级为 auxiliary supervision。

---

## 2. 整体架构解析

参考 Figure 2 的架构图, 整个 pipeline 可以分解为:

```
[Multi-view Images] + [Language Instruction]
        ↓
[VLM Backbone (Qwen3-VL-4B-Instruct)]
        ↓ (shared representation, embodiment-agnostic)
[Action Expert (Flow-matching DiT, ~600M params)]
        ↑
[Morphology Token] (only injected here, NOT in VLM)
        ↓
[Time-aligned Camera-space Action Chunk]
```

**关键 design choice**: morphology token 只在 action expert 的 decoding 阶段注入, VLM backbone 完全 embodiment-agnostic。这保证了 VLM 学到的 visual-language understanding 是跨 embodiment 共享的, 只有 action execution 路径是 embodiment-specific 的。

---

## 3. Unified Action Representation 详解

这是 paper 的第一个核心贡献, 通过三个维度对齐 heterogeneous data。

### 3.1 Spatial Alignment: Canonical Action Space (公式 1)

$$p_{\mathrm{cam}} = R_{\mathrm{cam} \leftarrow s} p_s + t_{\mathrm{cam} \leftarrow s}, \quad R_{\mathrm{cam}, ee} = R_{\mathrm{cam} \leftarrow s} R_{s, ee}$$

**变量含义**:
- $p_s \in \mathbb{R}^3$: end-effector position 在 source frame (robot base/world frame)
- $R_{s, ee} \in \mathrm{SO}(3)$: end-effector orientation 在 source frame
- $R_{\mathrm{cam} \leftarrow s} \in \mathrm{SO}(3)$: source frame 到 head-camera frame 的 rotation (来自标定)
- $t_{\mathrm{cam} \leftarrow s} \in \mathbb{R}^3$: 对应 translation
- $p_{\mathrm{cam}}$: position 在 camera frame
- $R_{\mathrm{cam}, ee}$: orientation 在 camera frame

**Intuition**: 把所有 action 投影到 head-camera coordinate frame, 这样 perception (image) 和 action 在同一个 coordinate system 里。Policy 不需要学习 implicit 的 world-to-camera transform, 这是一个 implicit inductive bias。部署新 robot 时, 只需要替换一个 camera extrinsic matrix (公式 11):

$$\hat{p}_s = R_{\mathrm{cam} \leftarrow s}^\top (\hat{p}_{\mathrm{cam}} - t_{\mathrm{cam} \leftarrow s}), \quad \hat{R}_{s, ee} = R_{\mathrm{cam} \leftarrow s}^\top \hat{R}_{\mathrm{cam}, ee}$$

### 3.2 Human Hand-Centric Frame Derivation (公式 13)

由于 human hand 没有物理 end-effector, paper 定义一个 hand-centric frame 作为 proxy:

$$\mathbf{x} = \frac{\mathbf{p}_{\mathrm{palm}} - \mathbf{p}_{\mathrm{wrist}}}{\|\mathbf{p}_{\mathrm{palm}} - \mathbf{p}_{\mathrm{wrist}}\|_2}, \quad \mathbf{z} = \hat{n}(\mathbf{p}_{\mathrm{wrist}}, \mathbf{p}_{\mathrm{thumb}}, \mathbf{p}_{\mathrm{middle}}), \quad \mathbf{y} = \mathbf{z} \times \mathbf{x}$$

**变量含义**:
- $\mathbf{p}_{\mathrm{wrist}}$: wrist joint position (作为 origin, 因为 HaMeR 重建 wrist 最稳定)
- $\mathbf{p}_{\mathrm{palm}}$: palm centroid, 是 index/middle/ring fingertip 的均值
- $\mathbf{p}_{\mathrm{thumb}}, \mathbf{p}_{\mathrm{middle}}$: thumb 和 middle fingertip positions
- $\hat{n}(\mathbf{a}, \mathbf{b}, \mathbf{c})$: 三点定义平面的 unit normal, 符号选择远离 palm 方向
- $\mathbf{x}, \mathbf{y}, \mathbf{z}$: hand frame 的三个轴, $R_{\mathrm{cam}, \mathrm{hand}} = [\mathbf{x}, \mathbf{y}, \mathbf{z}] \in \mathrm{SO}(3)$

**Intuition**: 这个 frame 构造利用 palm 几何的稳定性, 即使 wrist 有 yaw drift, $\mathbf{x}$ 轴 (wrist-to-palm) 和 $\mathbf{z}$ 轴 (palm normal) 仍然稳定。Gripper openness 用 thumb-to-palm 距离 $d_t = \|\mathbf{p}_{\mathrm{thumb}, t} - \mathbf{p}_{\mathrm{palm}, t}\|_2$ 线性归一化到 $[0.04, 0.10]$ m (匹配 robot gripper stroke)。

### 3.3 Continuous 6D Rotation (公式 12)

为避免 quaternion 和 Euler 的 discontinuity, 用 Zhou et al. [45] 的 continuous 6D 表示:

$$\mathrm{rot6d}(R_{\mathrm{cam}, ee}) = [R_{\mathrm{cam}, ee}^{(:, 1)}; R_{\mathrm{cam}, ee}^{(:, 2)}] \in \mathbb{R}^6$$

取 rotation matrix 的前两列拼接。这个表示在 SO(3) 上是连续的, 避免 quaternion 的 double-cover 问题和 Euler 的 gimbal lock。

### 3.4 Unified 22-Dimensional Action Layout (公式 14, 15)

$$\mathbf{a} = [\mathbf{a}_{\mathrm{left}}; \mathbf{a}_{\mathrm{right}}] \in \mathbb{R}^{22}$$

每个 arm 11D:
$$\mathbf{a}_{\mathrm{arm}} = [\underbrace{p_x, p_y, p_z}_{\text{Position (3D)}}, \underbrace{r_1, \dots, r_6}_{\text{Continuous Orientation (6D)}}, \underbrace{g}_{\text{Gripper (1D)}}, \underbrace{\alpha}_{\text{Activity Flag (1D)}}]$$

**关键设计**: activity flag $\alpha \in \{0, 1\}$ 表示该 arm 是否 active, 让 policy 同时处理 single-arm 和 bimanual embodiment。On-disk 存储 16D (Euler), 训练时转 22D (6D rotation)。

### 3.5 Structural Alignment: Morphology Conditioning (公式 2)

$$h_{\mathrm{morph}} = \begin{cases} P_{\mathrm{morph}}(E_{\mathrm{urdf}}(\mathcal{G}_r)), & \text{robot source } r \\ P_{\mathrm{surr}}(e_d), & \text{human source } d \end{cases}$$

**变量含义**:
- $\mathcal{G}_r$: robot $r$ 的 URDF kinematic graph (joint-centric)
- $E_{\mathrm{urdf}}$: URDF encoder (message passing over kinematic tree)
- $e_d \in \mathbb{R}^D$: human source $d$ 的 learned surrogate embedding (随机初始化, 端到端训练)
- $P_{\mathrm{morph}}, P_{\mathrm{surr}}$: projection 到共享 morphology token space

**URDF Encoder 细节** (公式 17-19):
每个 joint 用 29D descriptor 描述 (kinematic attributes + range/actuation + graph topology + chain relation)。Encoder 跑 $L$ 层 residual message passing:

$$H^{(0)} = \phi_{\mathrm{in}}(X_r), \quad H^{(\ell+1)} = H^{(\ell)} + \phi_\ell([H^{(\ell)}; \bar{A}_r H^{(\ell)}])$$

其中 $\bar{A}_r = D_r^{-1}(A_r + I)$ 是 row-normalized adjacency (加 self-loop)。然后 pool 成两个 summary:

$$z_{\mathrm{body}}^r = \rho_{\mathrm{body}}(\mathrm{mp}(\mathcal{T}_r)), \quad z_{\mathrm{chain}}^r = \rho_{\mathrm{chain}}([\mathrm{mp}(\mathcal{C}_L^r); \mathrm{mp}(\mathcal{C}_R^r)])$$

- $z_{\mathrm{body}}^r$: 全局 embodiment summary (所有 joint 的 mean pool)
- $z_{\mathrm{chain}}^r$: 左右 manipulation chain 的 summary (end-effector 相关的 kinematic path)
- $\mathrm{mp}(\mathcal{S}) = \frac{1}{|\mathcal{S}|} \sum_{j \in \mathcal{S}} H_j^{(L)}$

**Intuition**: body summary 捕获全局结构 (例如 humanoid vs wheeled), chain summary 聚焦于 manipulation 关节路径 (joint limit, axis, type)。Human source 没有 URDF, 用一个 per-dataset learned embedding 吸收 source-level factor (camera placement, visual domain, annotation quality, action statistics)。这个 embedding 端到端学习, 比手动设计 feature 更灵活。

### 3.6 Temporal Alignment: Time-Aligned Action Chunking (公式 3, 4, 5)

$$H_d = \mathrm{round}(f_d T^\star)$$

**变量含义**:
- $f_d$: dataset $d$ 的控制频率
- $T^\star = 2$ s: target physical duration (固定)
- $H_d$: dataset $d$ 的 step horizon

例如 10Hz dataset $H = 20$, 20Hz dataset $H = 40$, 30Hz dataset $H = 60$, 但都覆盖 2 秒物理时间。

为处理 variable-horizon chunk 在同一个 batch 里的 padding 问题, 用 composite key 分桶:

$$\phi = \mathrm{clip}\left(\frac{t + \frac{1}{2}H_d}{L_e}, 0, 1\right)$$

**变量**:
- $t$: sample 起始 index
- $L_e$: episode 总长度
- $\phi$: normalized episode phase (0=开始, 1=结束), 跨 dataset 可比

$$k = (c_{\mathrm{task}}, b_\phi, b_H)$$

按 task cluster + phase bucket + horizon bucket 分组, 同组进同一 mini-batch。

**Intuition**: 同一个 batch 内的 sample 物理 phase 相似, horizon 相同, padding 最少。这同时稳定 gradient update (避免不同物理时间尺度的 action chunk 混在一个 batch)。

---

## 4. Reliability-Aware Training Objective 详解

这是 paper 的第二个核心贡献, 解决 supervision-quality mismatch。

### 4.1 Hierarchical Reliability Decomposition (公式 6, 20)

$$W_{t,j} = \rho_j \cdot w_{\mathrm{data}}(d, h(j)) \cdot w_{\mathrm{step}}(t, h(j))$$

**三层结构**:
1. **Channel-level prior $\rho_j$** (静态): position channels $\rho = 1.0$, rotation/gripper channels $\rho = 0.001$
   - Intuition: HaMeR 重建 wrist position 相对可靠, 但 wrist rotation 和 thumb-to-palm 距离 (proxy gripper) 在 occlusion 下非常 noisy
2. **Dataset-level prior $w_{\mathrm{data}} \in [0.25, 1.0]$**: 每个 human source 的全局质量 ceiling
   - 估计方法: 聚合该 source 通过 sanity filter 的 frame 比例 + retained trajectory 的 median normalized jerk
   - 高生存率 + 低 jerk → $w_{\mathrm{data}}$ 接近 1.0
3. **Step-level smoothness $w_{\mathrm{step}}(t, h)$** (动态, 公式 22-24): 

$$\Delta p_t^h = \|p_t^h - p_{t-1}^h\|_2, \quad \Delta^2 p_t^h = \|p_{t+1}^h - 2p_t^h + p_{t-1}^h\|_2$$

$$q_{t,h} = \max\left(\frac{\Delta p_t^h}{\tau_{\mathrm{jump}}(d, h)}, \frac{\Delta^2 p_t^h}{\tau_{\mathrm{jerk}}(d, h)}\right)$$

$$w_{\mathrm{step}}(t, h) = \begin{cases} 1, & q_{t,h} \leq 1 \\ \max\{w_{\mathrm{min}}, \exp[-\alpha(q_{t,h} - 1)]\}, & q_{t,h} > 1 \end{cases}$$

**变量**:
- $\Delta p_t^h$: inter-frame speed (一阶差分)
- $\Delta^2 p_t^h$: jerk (二阶差分)
- $\tau_{\mathrm{jump}}(d, h), \tau_{\mathrm{jerk}}(d, h)$: per-dataset, per-hand 的 95th percentile 阈值 (预先计算)
- $\alpha = 1.5$: attenuation sharpness
- $w_{\mathrm{min}} = 0.2$: 最小 step weight, 避免完全 mask 掉

**Intuition**: $q_{t,h} > 1$ 表示 jump 或 jerk 超过该 dataset 的 95th percentile, 通常意味着 tracking failure 而非真实快速运动。用 exponential soft attenuation, 比 hard threshold 更平滑。$w_{\mathrm{min}} = 0.2$ 避免信号完全消失。

### 4.2 Robot Primary Loss (公式 7)

标准 conditional flow-matching loss:

$$\mathcal{L}_{\mathrm{action}} = \mathbb{E}_{s, \epsilon} \sum_{t, j} M_{t,j} \|\hat{v}_\theta(\mathbf{a}_s, s)_{t,j} - (\mathbf{a} - \epsilon)_{t,j}\|^2$$

**变量**:
- $\mathbf{a}$: clean robot action target (sensor-grounded, 22D)
- $\epsilon \sim \mathcal{N}(0, I)$: Gaussian noise
- $s \sim \mathcal{U}(0, 1)$: flow time parameter
- $\mathbf{a}_s = s\mathbf{a} + (1-s)\epsilon$: flow interpolant (线性插值 noise 和 target)
- $\hat{v}_\theta$: predicted velocity field (DiT action expert)
- $M_{t,j} \in \{0, 1\}$: action mask (validity, 处理 padding 和无效 frame)

这是 π0 风格的 flow matching, 用 delta action chunk (相对 head-camera frame)。注意 $M_{t,j}$ 对 robot data 是 mask, 不是 reliability weight, 表示该 channel 是否有效 (例如单臂 dataset 的另一只 arm 的 channel 是 masked 而非 down-weighted)。

### 4.3 Human Auxiliary Loss (公式 8, 9)

$$\mathcal{L}_{\mathrm{hux}} = \mathbb{E}_{s, \epsilon} \frac{1}{Z} \sum_{t, j} M_{t,j} W_{t,j} \mathrm{Huber}_\beta(\hat{v}_\theta(\mathbf{a}_s, s)_{t,j} - (\tilde{\mathbf{a}} - \epsilon)_{t,j})$$

**变量**:
- $\tilde{\mathbf{a}}$: temporally smoothed human target (window $W_{\mathrm{smooth}} = 3$ frames, 抑制 per-frame jitter)
- $Z = \sum_{t,j} M_{t,j} W_{t,j}$: normalization factor (使 loss 对 valid entry 数量 scale-invariant)
- $\mathrm{Huber}_\beta$: Huber regression, $\beta = 1.0$ (transition point), 对 outlier 更 robust
- $\lambda_{\mathrm{hux}} = 0.1$: 总 loss 平衡权重

$$\mathcal{L} = \mathcal{L}_{\mathrm{action}} + \lambda_{\mathrm{hux}} \mathcal{L}_{\mathrm{hux}}$$

**核心 intuition**:
1. **Huber loss** 替代 L2: 对 reconstruction noise 的 outlier 更 robust, 不会让单个 noisy frame 拖垮整个 trajectory
2. **Normalization $Z$**: 保证 human loss 不受 batch 内 valid frame 数量影响
3. **Per-channel $\rho_j$**: 实际效果是把 human supervision 几乎完全集中在 6 个 position channel (wrist xyz × 2 hands), rotation 和 gripper channel 几乎被 mask 掉 ($\rho = 0.001$)
4. **$\lambda_{\mathrm{hux}} = 0.1$**: human loss 比 robot loss 小一个量级, 保证高保真 robot data 主导 action expert

这个设计让 human video 的角色变成 "提供 broad visual+behavioral coverage 的 weak auxiliary supervision", 而不是 "替代 robot demonstration 的等价监督源"。

---

## 5. Egocentric Video-to-Action Pipeline 详解

参考 Figure 4, 五阶段 pipeline 把 raw video 转成 camera-space pseudo-action。

### Stage 1: Dataset Curation
- 输入: 6 个 public dataset (Ego4D, EgoExo4D, EPIC-KITCHENS-100, HOI4D, EgoDex, Xperience-10M)
- Clip 时长 filter: 4s - 30s (太短不够完整 manipulation primitive, 太长稀释)
- 统一 storage format: clip id, frame index, camera intrinsic, narration, license

### Stage 2: Video Selection
两个 filter:
1. **Ego-interaction filter**: face detection confidence > 0.5 的 clip 被丢弃 (说明是 observer view, 不是 egocentric)
2. **Caption filter**: 保留 narration 包含至少一个 manipulation verb 和一个 manipulable object noun 的 clip

### Stage 3: 3D Hand Reconstruction
三个 sub-stage:
1. **2D tracking**: SAM3 [49] tracker 获得 temporally consistent hand bounding box + segmentation mask
   - 丢弃: keypoint confidence < $\tau_{\mathrm{kp}} = 0.4$, track length < $\ell_{\mathrm{min}} = 15$ frames
2. **Local pose estimation**: HaMeR [13] 重建 MANO 参数 $\{\beta, \theta_t, \mathbf{t}_t^{\mathrm{local}}\}_{t=1}^T$ per frame
   - $\beta$: MANO shape parameter (手掌大小)
   - $\theta_t$: pose parameter (joint angle)
   - $\mathbf{t}_t^{\mathrm{local}}$: local root translation
3. **Global trajectory optimization** (公式 10): 两阶段 L-BFGS
   - Stage 1 ($N_{\mathrm{root}} = 30$ iter): 估计 globally consistent root translation 和 orientation
   - Stage 2 ($N_{\mathrm{smooth}} = 200$ iter): 联合 minimize reprojection error + temporal smoothness

$$\mathcal{L}_{\mathrm{smooth}} = \mathcal{L}_{\mathrm{reproj}} + \lambda_{\mathrm{tv}} \sum_t \|\mathbf{t}_{t+1}^{\mathrm{global}} - 2\mathbf{t}_t^{\mathrm{global}} + \mathbf{t}_{t-1}^{\mathrm{global}}\|_2^2$$

**变量**:
- $\mathcal{L}_{\mathrm{reproj}}$: 2D keypoint reprojection loss (从 estimated 3D hand 投影回 2D, 与 detected keypoint 比)
- $\mathbf{t}_t^{\mathrm{global}}$: frame $t$ 的 global hand root translation
- $\lambda_{\mathrm{tv}} = 1.0$: total variation regularization strength (二阶差分, penalize jerk)

Camera pose $(\mathbf{R}_t^{\mathrm{cam}}, \mathbf{t}_t^{\mathrm{cam}})$ 由 VIPE [51] 估计, 让 local 重建可以转到 shared world frame 做全局优化, 但最终 pseudo-action 还是转回 head-camera frame 用于训练。

### Stage 4: Action Parameterization
- Wrist origin + palm-plane orientation + thumb-to-palm gripper proxy (Sec 3.1.1)
- Gripper normalization: $d_t$ 线性归一化到 $[0.04, 0.10]$ m
- Degenerate filter: 如果 $d_{90} - d_{10} < \tau_{\mathrm{grip}} = 1.5$ cm (10-90 percentile 范围), 视为 closed-fist 无 grasp transition, 赋 constant neutral gripper state

### Stage 5: Quality Control
四个 filter:
1. **Completeness filter**: 无 NaN/Inf, frame index 连续, $||q|| - 1| \leq \tau_{\mathrm{quat}} = 10^{-3}$
2. **Static filter**: 两只手都满足 per-second motion energy < $\tau_{\mathrm{static}}$ (source-specific) 则丢弃
3. **Spike filter**: inter-frame 位置变化超过 $\kappa_{\mathrm{spike}} = 3\sigma$ 的 frame 比例超过 $\rho_{\mathrm{spike}} = 5\%$ 则丢弃整个 episode
4. **Bimanual filter**: 基于 inter-hand distance 统计 + 两手时间相关性, 移除异常 dual-arm 行为

最终产出 1,478.9 hours 的 pseudo-action-labeled human video (Table 1)。

---

## 6. 实验数据与结果分析

### 6.1 Pretraining Data Pool (Table 1)

| Source | Episodes | Frames | Hours | Supervision |
|---|---|---|---|---|
| **Human video subtotal** | 1,494,195 | 144,032,114 | 1,478.9 | Pseudo-action |
| - Ego4D | 948,683 | 23,396,157 | 216.6 | Pseudo-action |
| - EgoDex | 327,317 | 83,894,075 | 776.8 | Pseudo-action |
| - Xperience-10M | 99,027 | 31,370,900 | 435.7 | Pseudo-action |
| **Robot subtotal** | ~268,585 | ~460M | 4,534.8+ | Robot action |
| - AgiBot Alpha/Beta | 116,013 | 209,284,239 | 1,937.8 | Robot action |
| - Galbot self-collected | ~60,000 | ~194M | 1,800+ | Robot action |
| **Total** | ~1,762,780 | ~604M | 6,013.7+ | Mixed |

**Intuition**: 6K+ hours 的混合数据, human video 占 ~25% 小时数但提供远超 robot 的行为多样性 (long-tail manipulation skill)。Robot data 提供 high-fidelity supervision anchor。

### 6.2 RoboCasa GR1 TableTop Results (Table 3)

24 个 task, 50 rollouts/task:

| Method | Avg Success (%) |
|---|---|
| GR00T-N1.6 | 47.6 |
| Qwen3PI | 43.9 |
| FLARE | 55.0 |
| ABot-M0 | 58.3 |
| JoyAI-RA | 63.2 |
| DIAL | 70.2 |
| **ACE-EGO-0** | **72.8** |

ACE-EGO-0 比 DIAL 高 2.6%, 比 GR00T-N1.6 高 25.2%。值得关注的是 PlacematToTieredshelf 这种困难 task, ACE-EGO-0 是 44%, 而 GR00T-N1.6 只有 28.5%, 提升 54% 相对增益。PlateToPlate ACE-EGO-0 达到 98% (近乎完美)。

### 6.3 RoboTwin 2.0 Results (Table 4)

50 个 bimanual task, 100 trials/task:

| Method | Easy (%) | Hard (%) |
|---|---|---|
| $\pi_{0.5}$ | 82.74 | 76.76 |
| Motus | 88.66 | 87.02 |
| LingBot-VLA | 88.56 | 86.68 |
| ABot-M0 | 86.06 | 85.08 |
| JoyAI-RA | 90.48 | 89.28 |
| Hy-VLA | 90.9 | 90.1 |
| **ACE-EGO-0** | **91.12** | **90.62** |

Hard setting (强 domain randomization) 下 ACE-EGO-0 只比 Easy 降 0.5%, 比 $\pi_{0.5}$ 的 Easy→Hard 降 6% 小得多, 说明 camera-space action representation 对 visual perturbation 更 robust。

### 6.4 Real-Robot Results (Figure 5a)

ARX bimanual platform, 6 task × 30 trials:

| Method | Pick Tea | Scoop Coffee | Category Sorting | Sweep Cubes | Stack Bowls | Pack Shoes | Avg |
|---|---|---|---|---|---|---|---|
| $\pi_{0.5}$ | - | 70.0 | 80.0 | - | - | - | 71.7 |
| GR00T-N1.7 | - | 36.7 | 83.3 | 6.7 | 73.3 | - | 35.6 |
| **ACE-EGO-0** | - | **86.7** | **90.0** | - | - | - | **78.3** |

**关键观察**:
- Scoop Coffee (contact-rich bimanual): ACE-EGO-0 86.7% vs $\pi_{0.5}$ 70% (+16.7%) vs GR00T-N1.7 36.7% (+50%)
- Sweep Cubes (horizontal trajectory): GR00T-N1.7 仅 6.7%, ACE-EGO-0 明显更好
- Pack Shoes (longest sequence): 所有 method 都 degradation, 说明 long-horizon compounding drift 仍是 open challenge

### 6.5 Ablation Studies

**Component Ablation (Figure 5b)**, RoboCasa 上:

| Configuration | Success (%) | $\Delta$ |
|---|---|---|
| Full ACE-EGO-0 | 72.8 | - |
| - Morphology tokens | 70.9 | -1.9 |
| - Time-aligned chunking | 71.7 | -1.1 |
| - Reliability-aware human loss | 69.2 | **-3.6** |

**最大下降来自移除 reliability-aware human loss**, 说明如果不区分 noise quality, 直接把 noisy pseudo-action 当成 robot label, 会污染 action expert。这验证了 paper 的核心 hypothesis: supervision quality 比 representation 对齐更重要。

**Data Source Ablation (Table 5)**:

| Pretraining Config | Success (%) |
|---|---|
| From Qwen (no embodied pretrain) | 65.4 |
| Robot Only | 68.3 (+2.9) |
| Robot + Human (full) | 72.8 (+4.5) |

Human video 贡献 +4.5% 增益 (最大单源增益), Robot 贡献 +2.9%, 都显著大于 0。从纯 vision-language pretrain 到 embodied pretrain 是 +9.6% 总增益。

### 6.6 Human Data for Augmented Fine-Tuning (Section 5.5)

Sweep Cubes task, 只用 34 robot demo (45.8K frames): 10% success rate。
加入 419 episodes 的 task-matched human video (117.5K frames): 40% success rate, **4× 提升**。

参考 Figure 6 的可视化:
- 34 robot demo workspace coverage: 0.062 m² (convex hull area)
- 419 human video episodes workspace coverage: 0.296 m² (**4.8× 更大**)
- Robot cluster 嵌入 human distribution 内, human data 提供 action space 的 broader coverage

**Intuition**: 这就是 human video 作为 auxiliary supervision 的真正价值 — 即使 label noisy, 它覆盖的 action distribution 比 robot demo 宽得多, 在 data-scarce regime 下能极大缓解 overfitting 到 narrow demonstration 的问题。

---

## 7. Architecture 和 Training Hyperparameters (Table 7, 8)

**Architecture**:
- VLM backbone: Qwen3-VL-4B-Instruct (~4B params, 36 layers, hidden 2560)
- Vision encoder: 24 layers, patch 16×16
- Action expert: Flow-matching DiT (36 layers, 1024 hidden, 16 heads, ~600M params)
- Input: 256×256 (head + wrist images)
- Inference: 4 flow-matching decoding steps

**Training**:
- 128×A800 (80GB) GPUs
- AdamW (β1=0.9, β2=0.95, ε=1e-8)
- VLM lr: 2e-5, Action expert lr: 1e-4
- Cosine schedule, min lr 5e-7, warmup 5000 steps
- Weight decay: 1e-8, gradient clip: 1.0
- Batch size per device: 8
- Pretraining: 200K steps
- Fine-tuning: 16×A800, dataset-specific $H_d$ with $T^\star = 2$s

---

## 8. 关键 Insight 总结

### 8.1 为什么 Camera-Space 而非 World-Space?
- Camera-space 让 perception 和 action 共享 coordinate frame, VLM 学到的 spatial understanding 可以直接 transfer 到 action
- World-space 需要每个 robot 学习自己的 base-to-world transform, 这违反了 generalist 的初衷
- 部署只需替换一个 camera extrinsic, 实现 embodiment-agnostic perception

### 8.2 为什么 Morphology Token 隔离在 Action Expert?
- VLM backbone 应该学到 embodiment-agnostic 的 visual-language understanding
- Morphology 是 execution-specific 的 (kinematic chain, joint limit), 只在 action decoding 时需要
- 如果 morphology 进 VLM, 会让 visual feature 被 embodiment id 污染

### 8.3 为什么 Time-Aligned 而非 Step-Aligned?
- 10Hz dataset 的 40 step = 4 秒, 30Hz dataset 的 40 step = 1.33 秒, 这是完全不同的 physical horizon
- 时间对齐让所有 dataset 监督同一个 future physical window, 这是 cross-embodiment learning 的基础
- 分桶采样解决 variable-horizon padding 问题, 否则 mini-batch 内会有大量 pad token 浪费 compute

### 8.4 为什么 Reliability-Aware 而非 Equal Weight?
- Naive joint training 让 noisy human pseudo-action 和 clean robot label 平等竞争, action expert 会被 noise 拖向平均
- Channel prior $\rho_j$ 把 human supervision 几乎完全限制在 6 个 position channel, 这正是 HaMeR 重建最可靠的部分
- Step-level weight 进一步 mask 掉 tracking failure 的具体 frame
- $\lambda_{\mathrm{hux}} = 0.1$ 确保 robot data 主导, human data 只提供 broad coverage 的 weak supervision

### 8.5 为什么 Human Data 在 Data-Scarce Fine-Tuning 最有价值?
- Robot demo 通常只覆盖 narrow action distribution (34 demo 在 0.062 m²)
- Human video 覆盖 4.8× 更广的 workspace
- 即使 label noisy, 更广的 coverage 能 prevent overfitting 到 narrow demo
- 这是 robot learning 的 "data augmentation via human video" 范式

---

## 9. 局限性与未来方向

Paper 自己列出 (Section 7):
1. 只测了 tabletop, 没测 mobile manipulation, whole-body humanoid, deformable object
2. Pretraining pool 没有 dexterous hand data 和 force/torque sensing
3. Human-video scaling 和 pseudo-action fidelity (尤其 rotation 和 fine-grained finger motion) 还有提升空间, 未来如果 rotation channel 也可靠, reliability-aware objective 可以监督更多 dimension

---

## 10. 相关工作链接

- **π0**: https://arxiv.org/abs/2410.24164 (Flow matching VLA)
- **π0.5**: https://proceedings.mlr.press/v305/black25a.html (Open-world VLA)
- **OpenVLA**: https://arxiv.org/abs/2406.09246
- **GR00T N1**: https://arxiv.org/abs/2503.14734 (NVIDIA humanoid foundation model)
- **Open X-Embodiment**: https://arxiv.org/abs/2310.08864
- **Ego4D**: https://arxiv.org/abs/2110.07058
- **EgoMimic**: https://arxiv.org/abs/2410.03661 (Egocentric imitation learning)
- **EgoVLA**: https://arxiv.org/abs/2507.12440 (从 human video 学 VLA)
- **HaMeR**: https://arxiv.org/abs/2306.15419 (3D hand reconstruction)
- **MANO**: https://arxiv.org/abs/1710.06519 (hand model)
- **Zhou 6D rotation**: https://arxiv.org/abs/1812.07035 (continuous rotation representation)
- **RoboCasa**: https://arxiv.org/abs/2406.10965
- **RoboTwin 2.0**: https://arxiv.org/abs/2506.18088

---

## 11. 我的批判性思考

这篇 paper 在工程上做得很扎实, 几个值得 deep dive 的点:

**Strengths**:
1. **三层 alignment 的系统性**: spatial/structural/temporal 三个 axis 系统对齐, 没有 hand-wave 任何一个
2. **Reliability-aware objective 的 hierarchical decomposition**: channel + dataset + step 三层, 每层都有明确的物理意义
3. **5-stage pipeline 的可扩展性**: 把 noisy video processing 拆成可独立调优的阶段, 每个阶段都有 quantitative threshold
4. **Data-scarce fine-tuning 的实证**: Figure 6 的 workspace coverage 可视化是最有说服力的 insight, 说明 human data 的真正价值是 coverage 而非 fidelity

**Open Questions**:
1. **$\lambda_{\mathrm{hux}} = 0.1$ 的 sensitivity**: Paper 说在 Appendix A.5 有 sensitivity analysis, 但 main text 没展开。这个 weight 太小 human 信号可能被淹没, 太大可能污染 robot signal, 是个 delicate balance
2. **$\rho_j = 0.001$ for rotation channels 实际上是 mask**: 这等于几乎完全放弃从 human video 学 rotation, 那 human video 的真正贡献只是 position channel 的 broad coverage。如果未来 HaMeR 之类的方法 rotation 更可靠, 这部分还能解锁
3. **Long-horizon 的根本 limitation**: Pack Shoes task 所有 method 都 degradation, 说明现在的 action chunk 4-step flow matching 在 long sequence compounding drift 上仍有问题, 可能需要 hierarchical planning 或 world model 辅助
4. **Morphology token 的 generalization**: URDF encoder 在 pretraining 见过的 robot 上工作, 但 zero-shot 到未见过的 embodiment 怎么样? Paper 没测这个
5. **Activity flag $\alpha$ 的语义**: 当一个 dataset 是单臂时, 另一只 arm 的 $\alpha = 0$, 那 VLM 是否会学到 "另一只 arm inactive" 这个 prior, 在 bimanual task 上反而成为 bias? 这点没讨论

**与 π0 / π0.5 的对比**:
ACE-EGO-0 用 Qwen3-VL-4B 作为 backbone (vs π0 用 PaliGemma 3B), action expert 是 600M DiT (π0 也是 flow matching)。架构上很接近 π0 family, 主要 contribution 在 data pipeline 和 training objective 而非 architecture novelty。这也是合理的, 因为 embodied AI 的瓶颈在 data 不在 architecture。

**与 DIAL 的对比**:
DIAL 用 latent world model 解耦 intent 和 action, ACE-EGO-0 直接用 explicit pseudo-action + reliability weighting。DIAL 在 RoboCasa 70.2%, ACE-EGO-0 72.8%, 差距不大, 但 ACE-EGO-0 的方法更简单直接, 工程上更容易 reproduce。

**Intuition 总结**: 这篇 paper 的真正贡献是把 "如何用 noisy human video 监督 VLA" 这个问题拆成了两个正交的 sub-problem — representation heterogeneity 用 unified interface 解决, supervision quality mismatch 用 reliability-aware weighting 解决。两者缺一不可, ablation 也证实 reliability-aware 是最大 contributor (-3.6% without it)。这是 mixed-source VLA pretraining 的一个 solid baseline, 后续工作大概率会在这个 framework 上 scale up human video portion 和 pseudo-action fidelity。
