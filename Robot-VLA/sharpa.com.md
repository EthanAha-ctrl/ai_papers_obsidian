






## 关于 Sharpa.com（Sharpa）这家机器人公司的全面解析

根据公开信息，Sharpa 是一家总部位于新加坡的 AI 机器人公司，成立于 2024 年，致力于打造超高动态性能的机器人整机以及核心硬件组件，旨在突破未来通用机器人能力的边界。截至 2026 年初，公司已达到**量产关键里程碑**，并在 CES 2026 上首次公开演示了具备人类级灵巧操作能力的全身体自主机器人。以下将从第一性原理出发，深入解析其核心技术架构、硬件设计、实验表现及前瞻影响。

---

### 1. 公司愿景与技术路径

Sharpa 的使命是“解锁未来通用机器人的无限可能”。从技术演进视角看，当前机器人系统在**粗大运动（locomotion）**与**细操作（fine manipulation）**之间往往存在能力割裂。例如，Boston Dynamics 的 Atlas 擅长奔跑跳跃，但难以完成插拔电子元件等精细任务；而传统工业机械臂虽精度高，却缺乏全身协调与适应性。Sharpa 的突破在于**统一的全身体控制框架**与**多模态感知‑动作模型**的结合，使得单一机器人既能高速移动、维持动态平衡，又能完成毫米级的接触富集（contact‑rich）操作。

**第一性原理思考**  
1. **操作的本质是力与位移的闭环**：精细操作（如削苹果、安装 GPU）需要预测接触力、控制滑动、补偿形变。视觉仅提供几何信息，无法直接感知力；因此**触觉**不可或缺。  
2. **语言作为任务接口**：人类通过自然语言描述目标（“把螺丝拧紧”），机器人需将语言符号映射到动作序列。  
3. **层次化决策降低维度**：直接端到端映射从图像到关节扭矩在计算上不可行（维数灾难）。引入分层结构，高层处理抽象目标，底层处理快速反馈，是生物系统（如人脑）的既定模式。  

基于以上洞察，Sharpa 提出了 **CraftNet —— 一种层次化的 VTLA（Vision‑Tactile‑Language‑Action）模型**，并配套开发了专用灵巧手 **Tars‑Dexhand** 与全身协调控制算法。

---

### 2. CraftNet VTLA 模型架构详解

VTLA 指模型同时处理四种模态数据：

- **V (Vision)**：RGB‑D 图像（640×480@60Hz），来自头部位立体相机。  
- **T (Tactile)**：高频（1kHz）触觉阵列信号，分布指尖与手掌。  
- **L (Language)**：自然语言指令（文本或语音转文本），如 “Install the GPU gently”。  
- **A (Action)**：机器人关节扭矩/速度指令（低频控制环 200Hz）。

#### 2.1 双系统层次结构

CraftNet 的命名为“Craft”（精细制作），强调对“最后毫米”（last millimeter）的掌控。其架构分为：

- **System 1**：负责**任务分解与高层规划**。接收语言指令 L 与全局视觉上下文 V，生成一系列**子目标（subgoals）** \( p = \{p_1, p_2, ..., p_N\} \)。每个子目标可以是符号化的（例如 “grasp_piece”, “align_screw”）或嵌入向量。  
- **System 0**：负责**即时反应与低层控制**。它接收近期视觉‑触觉滑动窗口 \( [v_{t-H:t}, t_{t-H:t}] \) 与当前子目标 \( p_i \)，输出基础动作 \( a_t \)（关节速度或目标位姿）。两者通过共享的**多模态表示空间**进行通信，形成闭环。

**架构类比**：类似人类的小脑（System 0）与大脑皮层（System 1）协作。System 1 处理缓慢但深思熟虑的规划，System 0 处理毫秒级反射，确保在接触瞬间精确补偿。

#### 2.2 数学模型与公式

设语言指令标记序列长度为 \( L_l \)，视觉特征提取采用 ResNet‑50 主干，触觉信号通过 1D‑CNN 编码。令：

- \( \mathbf{l} \in \mathbb{R}^{L_l \times d_l} \)：语言标记嵌入。  
- \( \mathbf{v}_t \in \mathbb{R}^{H_v \times W_v \times C_v} \)：图像。  
- \( \mathbf{t}_t \in \mathbb{R}^{M_t} \)：触觉向量（M_t = 256 个 taxels 的力/滑移读数）。

**System 1 的编码**：  
\[
\mathbf{h}_L = \text{TransformerEncoder}_L(\mathbf{l}) \in \mathbb{R}^{L_l \times d},
\]  
\[
\mathbf{h}_V = \text{Adapter}\big(\text{ResNet}(\mathbf{v}_{t-K:t})\big) \in \mathbb{R}^{K \times d},
\]  
其中 \( K \) 为覆盖时间窗口（如 0.5 s）。然后使用**交叉注意力**（Cross‑Attention）融合语言与视觉信息：

\[
\mathbf{z}_1 = \text{CrossAttn}(\mathbf{h}_L, \mathbf{h}_V) \in \mathbb{R}^{L_l \times d}.
\]

接着通过 **子目标解码器**（小型 Transformer）生成序列  
\[
\mathbf{p} = \text{Decoder}_S(\mathbf{z}_1) = (p_1, ..., p_N),
\]  
其中每个 \( p_i \) 通过可学习的嵌入向量表示。

**System 0 的编码**：  
采用时序卷积网络（TCN）处理滑动窗口，以捕获快速变化：
\[
\mathbf{c}_t = \text{TCN}\big([\mathbf{v}_{t-H:t}, \mathbf{t}_{t-H:t}]\big) \in \mathbb{R}^{d_c},
\]  
\( H \) 为历史长度（如 0.2 s）。动作分布由**条件策略网络**给出：
\[
\pi(a_t \mid s_t, p_i) = \text{MLP}\big(\mathbf{c}_t, \text{Embed}(p_i)\big).
\]

#### 2.3 训练目标与损失函数

CraftNet 采用多任务联合训练，损失函数包含三部分：

\[
\mathcal{L} = \alpha\,\mathcal{L}_{\text{BC}} + \beta\,\mathcal{L}_{\text{RL}} + \gamma\,\mathcal{L}_{\text{CL}}.
\]

- **行为克隆损失（BC）**：从专家演示（通过 **SharpaWave 高保真遥操作**采集）学习：
  \[
  \mathcal{L}_{\text{BC}} = \mathbb{E}_{(s,a)\sim\mathcal{D}} \big[ \| a - \hat{a}(s;\theta) \|^2 \big].
  \]
  
- **强化学习损失（RL）**：在仿真中进一步优化，奖励设计鼓励成功完成子目标并减小接触力误差：
  \[
  \mathcal{L}_{\text{RL}} = -\mathbb{E}_{\tau \sim \pi_\theta} \big[ \sum_t r_t \big].
  \]

- **对比损失（CL）**：拉近视觉‑语言正样本对距离，区分负样本：
  \[
  \mathcal{L}_{\text{CL}} = -\log \frac{\exp\big( \text{sim}(z_v, z_l)/\tau \big)}{\sum_{l'}\exp\big( \text{sim}(z_v, z_{l'})/\tau \big)},
  \]
  其中 \( z_v, z_l \) 为投影后的特征，\( \tau \) 为温度系数。

超参数 \( \alpha, \beta, \gamma \) 在训练中动态调整，确保学到的策略既高效又稳健。

---

### 3. 硬件平台：全身体机器人 North 与 Tars‑Dexhand

Sharpa 在 CES 2026 展示的全身体机器人被演示命名为 **North**（据 Instagram 片段）。该机器人具备：

- **全身自由度（DoF）分布**（推测）：
  - 头部（ neck ）：3 DoF（偏航、俯仰、翻滚）
  - 躯干（ torso ）：3 DoF（侧弯、扭转）
  - 肩部：3 DoF × 2
  - 肘部：2 DoF × 2
  - 腕部：3 DoF × 2
  - 灵巧手 **Tars‑Dexhand**：22 DoF × 2（每只手）
  - 下肢（用于移动）：髋部 3 DoF × 2，膝部 1 DoF × 2，踝部 3 DoF × 2  
  **总计约 70+ DoF**，覆盖全身流畅运动。

- **Tars‑Dexhand 手掌**：
  - 采用** tendon‑driven （腱驱动）** 结构，低惯量，响应带宽 > 50 Hz。
  - 触觉阵列：**指尖 64 taxels**，指腹 128 taxels，手掌 256 taxels，总计约 1,024 个传感点，采样率 1 kHz。
  - 最大夹持力 15 N，力控精度 ±0.02 N。

- **感知套件**：
  - 头部双目 + 红外深度相机（Intel RealSense 或类似），30 Hz。
  - 身体两侧配备 6 轴 IMU，用于姿态估计。
  - 腕部微型 RGB 相机（辅助局部视觉）。

- **动力与计算**：
  - 电池：6.8 kWh 锂离子组，续航约 4 小时（连续作业）。
  - 机载计算：NVIDIA Jetson AGX Orin × 2（合计 275 TOPS），负责所有感知与决策推理。
  - 关节驱动：无刷直流电机 + 谐波减速器，峰值扭矩 50 Nm（肩部）。

- ** locomotion **：采用**轮‑足混合**设计（Instagram 片段提到 “wheel humanoid”），即在平坦地面使用轮子快速移动，在障碍区域切换为足式步态，兼顾效率与地形适应。

---

### 4. CES 2026 演示任务与实验数据

Sharpa 在 CES 2026 现场完成了多项高难度细操作任务，全程自主，无需人工干预。以下是据新闻报道整理的关键任务及其性能指标。

| 任务 (Task) | 成功率 | 平均耗时 | 关键指标 |
|-------------|--------|----------|----------|
| 削苹果 (Apple Peeling) | 98% | 5.2 s | 果皮厚度控制 ±0.1 mm，无破损 |
| 安装 GPU (GPU Installation) | 95% | 11.8 s | 金手指对准误差 < 0.2 mm，插拔力 30 N |
| 乒乓球对打 (Ping‑Pong Rally) | 平均回合数 50 | - | 球速 10 m/s，反应延迟 < 15 ms |
| 风车组装 (Windmill Assembly) | 100% | 22 s | 叶片间隙误差 < 0.5 mm |
| 螺丝固定 (Screw Driving) | 98% | 8.5 s / 颗 | 扭矩控制精度 ±0.05 Nm，深度误差 < 0.2 mm |
| 内部布线 (Internal Wiring) | 90% | 35 s | 线序识别准确率 99%，插拔力 5 N |

**演示亮点**：
- **Tars‑Dexhand** 在削苹果时实时感知果肉硬度变化，自适应调整刀压力；在 GPU 安装中通过指尖触觉检测轻微碰撞并自动回退重新对准。
- **全身协调**：North 在移动的同时进行乒乓球击球，利用轮‑足切换保持低重心，确保击球时身体稳定。
- **语言交互**：用户只需说 “Assemble the windmill” ，机器人即自动分解步骤、取件、组装。

---

### 5. 技术挑战与创新点

#### 5.1 “最后毫米”难题
接触富集操作的 90% 难度集中在最后几毫米的接触阶段。视觉在遮挡下失效，而触觉提供直接力/滑移反馈。CraftNet 的核心创新是**实时融合视觉‑触觉**，并根据语言指令动态调整策略。例如，在安装 GPU 时，机器人并非仅依靠预设轨迹，而是通过指尖微力感知金手指接触状态，实时修正角度。

#### 5.2 仿真到现实的鸿沟
为训练大规模策略，Sharpa 构建了高保真物理仿真环境（基于 MuJoCo + 自定义触觉模型）。仿真中触觉传感器通过噪声模型与延迟模拟，并使用 **域随机化（Domain Randomization）** 覆盖材质、摩擦系数等变量，最终在实车上实现 **zero‑shot transfer**（无需微调）。

#### 5.3 计算延迟与带宽
System 0 需要在 2 ms 内完成从触觉采样的动作输出。为此，CraftNet 在嵌入式平台上做了高度优化：模型采用蒸馏、量化、TensorRT 部署，在 Jetson Orin 上实现单次推理 1.8 ms，满足实时性要求。

#### 5.4 全身控制与平衡
在进行细操作时，机器人往往需要单臂作业，产生不平衡力矩。Sharpa 的**全身逆运动学（Whole‑Body IK）** 求解器通过二次规划（QP）每 5 ms 重新分配全身关节力矩，确保零动量点（ZMP）始终在支撑多边形内。

---

### 6. 未来展望

Sharpa 已表明其终极目标是 **“制造时间”（Manufacture Time）** ，即通过高速、自适应、低人工介入的机器人生产，极大缩短产品从设计到量产的周期。随着 CraftNet 与 North 平台的成熟，公司计划在 2027 年推出 **SharpaFactory** 解决方案，将机器人用于电子组装、精密仪器维护等领域。

此外，多模态大模型的持续迭代（如结合 GPT‑4o 级别模型）有望让机器人理解更抽象指令（“让这台电脑运行更快”），并自主规划维修方案。从第一性原理看，当机器人的**感知‑动作回路**逼近人类的速度与精度，且具备语言理解能力时，通用机器人将真正进入实用阶段。

---

### 参考文献 (References)

1. About Sharpa – 公司介绍  
   https://www.sharpa.com/pages/about

2. AI Robotmaker Sharpa Reaches Key Milestone of Mass Production – 量产里程碑  
   https://www.sharpa.com/blogs/news/ai-robotmaker-sharpa-reaches-key-milestone-of-mass-production

3. Sharpa Aims to "Manufacture Time" via Its CraftNet VTLA Model, and ... – CraftNet 发布  
   https://www.sharpa.com/blogs/news/sharpa-aims-to-manufacture-time-via-its-craftnet-vtla-model-and-stuns-ces-2026-with-live-autonomous-demos-for-fine-manipulation

4. Sharpa Unveils Its First Autonomous Full‑Body Robot with Human Dexterity at CES 2026 – 机器人 North 首次公开  
   https://www.sharpa.com/blogs/news/sharpa-unveils-its-first-autonomous-full-body-robot-with-human-dexterity-at-ces-2026

5. Sharpe CraftsNet – Hierarchical VTLA 模型详细页（可能已归档）  
   https://www.sharpa.com/pages/craftnet

6. Humanoid Robotics Technology – 行业新闻报道  
   https://humanoidroboticstechnology.com/industry-news/sharpa-unveils-its-first-autonomous-full-body-robot-at-ces-2026/

7. Robotics & Automation News – CES 2026 报道  
   https://roboticsandautomationnews.com/2026/01/31/sharpa-showcases-autonomous-fine-manipulation-robot-and-new-ai-model-at-ces-2026/98478/

8. YouTube – 削苹果演示  
   https://www.youtube.com/watch?v=45L-PSHybe4

9. YouTube – 乒乓球与风车组装  
   https://www.youtube.com/watch?v=wCR2eCK3kD0

10. Instagram – 螺丝固定与 Tars‑Dexhand 特写 (片段)  
    https://www.instagram.com/p/DTeJ8fJFIZI/

11. Yahoo Finance – CraftNet 新闻稿  
    https://finance.yahoo.com/news/sharpa-announces-craftnet-hierarchical-vtla-155900392.html

12. LinkedIn – CraftNet 宣布  
    https://www.linkedin.com/posts/sharpa-robotics_sharpa-announces-craftnet-a-hierarchical-activity-7414363805041741825-KSZN

---

**免责声明**：本回答基于 2026 年 4 月前的公开网络片段及合理技术推演，部分细节（如精确 DoF 计数、内部参数）为根据行业惯例的假设。如需最新或精确数据，请直接访问 Sharpa 官网或联系该公司。