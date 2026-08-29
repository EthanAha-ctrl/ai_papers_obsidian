---
source_pdf: Vision-Language-Action ModelsforAutonomous Driving Past, Present, andFuture.pdf
paper_sha256: fbe3dcc6ff10a44fd3f6ff0e328c40346af905ead96e707a3bb876abd5a784af
processed_at: '2026-08-13T01:44:22-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---
论文给了一个特别简洁的公式：

$$\mathbf{a}_t = H(F(\mathbf{x} | \boldsymbol{\theta}))$$

翻译成人话就是：**输入一堆东西（图、激光雷达点云、文字指令、车的状态），经过一个 VLM 大脑处理，再通过一个 action head 输出动作。**

$\mathbf{x}$ 是输入，包括：
- 多个摄像头的 RGB 图（6-8 个相机）
- LiDAR 点云（3D 几何信息）
- 语言指令（"下个路口左转"）
- 车自己的状态（速度、加速度、方向盘角度）

$F(\cdot)$ 是 VLM backbone，就是类似 Qwen2.5-VL、InternVL、Gemini 这种多模态大模型。它负责把图和文字理解了，做 reasoning。

$H(\cdot)$ 是 action head，负责把 VLM 的理解转成实际能开车的输出。

---

## Action 有几种表达方式？

这是 VLA 设计里最核心的选择——**你怎么表达"车要怎么动"这件事**。

### 第一种：直接吐控制信号
就是直接输出 $(\delta_t, \tau_t, \beta_t)$——方向盘角度、油门、刹车。好处是直接能开，坏处是没有 future planning，容易抖。

### 第二种：吐 trajectory waypoints
输出未来 $\Phi$ 步的坐标 $\{(x_i, y_i)\}_{i=1}^{\Phi}$。比如未来 3 秒内车应该到哪些位置。这是最主流的做法，好处是有 planning horizon，坏处是 downstream 还要个 controller 把 waypoints 转成实际控制信号。

### 第三种：吐语言
就是用文字表达动作，比如 "turn left at intersection" 或者把 trajectory 坐标写成文字 tokens。这是 VLA 最 unique 的设计——**action 直接从 LLM 的 language head 出来**。好处是 interpretability 极强，坏处是文字和连续控制之间有 gap。

### 第四种：连续函数
用 $v(t)$ 速度曲线和 $\kappa(t)$ 曲率曲线表达未来运动。物理上最自然，但训练起来比较 tricky。

---

## 两大架构流派

论文把 VLA 分成两大类，这个 taxonomy 是整篇 paper 的骨架：

### 流派一：End-to-End VLA（单系统）

一个 model 干所有事——看图、读文字、推理、输出动作。优点是 tight coupling，缺点是 VLM 太大太慢，real-time 要求下扛不住。

这里面又分两种 action generator：

**Textual Action Generator**——用语言表达动作。代表是 DriveMLM（输出 meta-action 比如 "change lane"）、EMMA（用 Gemini 直接输出 trajectory as text）、DriveGPT4（输出解释+控制信号 as text）。

**Numerical Action Generator**——在 VLM backbone 上接一个 MLP 或者 diffusion head，直接输出数字 trajectory。代表是 LMDrive（ResNet+LLaMA+MLP 输出 control）、ORION（EVA-02+Vicuna+diffusion）、SimLingo（InternViT+Qwen2+解耦 MLP）。

### 流派二：Dual-System VLA（双系统）

灵感来自 Kahneman 的《Thinking, Fast and Slow》——人脑有两个系统：System 1 快速直觉反应，System 2 慢速深度思考。

VLA 版本就是：**VLM 当 System 2（慢思考，给 high-level guidance），传统 planner 当 System 1（快执行，输出 low-level control）。**

这个流派又分两种：

**Explicit Guidance**——VLM 直接输出 meta-action 或者粗 waypoints，planner 再细化。比如 DriveVLM 就是 VLM 通过 chain-of-thought 给出 "减速+往右变道" 这种 decision，然后传统 planner 把它变成详细 trajectory。

**Implicit Transfer**——训练时 VLM 当 teacher，把 reasoning 能力蒸馏到 compact E2E network 里，runtime 时不需要 VLM。比如 VLP 就是把 BEV features 和 planning queries 跟 language embeddings 对齐。

---

## 从 VA 到 VLA 的演化路线

论文讲了个故事，从最早的 VA model 一路怎么走到 VLA 的：

**第一阶段：Action-Only (1988-2020)**
ALVINN 用一个简单 neural network 从图像直接预测方向盘。ChauffeurNet 用 behavior cloning 从大量驾驶数据学。TransFuser 用 transformer 做 multimodal fusion。这些都是直接 pixels → actions。

**第二阶段：Perception-Action (2022-2024)**
UniAD 把 perception（检测、跟踪、预测）和 planning 统一到一个 network 里。VAD 用 vectorized scene representation 提升安全性和效率。DriveTransformer 用 sparse query 架构做到 scalable。这一阶段开始有 perception supervision，但还没 language。

**第三阶段：World Models (2024-2025)**
OccWorld、Vista、DriveDreamer 这些开始建模"如果车这么开，未来场景会怎么变"。这是从 reactive 到 proactive 的转变——不再只是看当下，而是能"想象"未来。

**第四阶段：VLA (2024-now)**
DriveMLM、DriveLM、LMDrive 开始引入 language。到 AutoVLA、SimLingo、ORION 已经能做到 closed-loop driving + language reasoning。

---

## 数据集演化

VA 时代的数据集就是图+轨迹：
- **BDD100K**: 120M frames，美国各种天气路况
- **nuScenes**: 1.4M frames，6 个相机+LiDAR，1000 个场景
- **Waymo Open Dataset**: 200M frames，高分辨率+长轨迹+HD map

VLA 时代需要图+语言+轨迹三元组：
- **BDD-X** (2018): 最早的，给 BDD100K 加了 26K 条 human rationale——人标注"司机为啥这么开"
- **DriveLM** (2024): 把驾驶建模成 graph-structured QA，445K 条问答对，覆盖 perception→prediction→planning 全链路
- **ImpromptuVLA** (2025): 从 8 个公开数据集聚合，2M frames + 80K corner-case 标注，专门 focus 长尾场景
- **CoVLA** (2025): 6M frames + 6M captions，规模最大的 VLA 数据集之一

---

## Benchmark 上表现咋样？

### nuScenes (Open-Loop)
- 老的 VA model 比如 UniAD: L2 error = 0.69m
- 好的 VLA model 比如 Reasoning-VLA: L2 = 0.22m，collision rate 从 12% 降到 7%
- **结论：VLA 在 open-loop 上已经超过 VA**

### WOD-E2E (Waymo, Long-tail)
这个 benchmark 特别重要，因为它用了 **Rater Feedback Score**——不是跟 expert trajectory 比，而是跟 human preference 比。

- Waymo 自己的 baseline: RFS = 7.53
- AutoVLA: RFS = 7.99（已经超过 baseline）
- OpenEMMA (纯 Qwen2-VL): RFS = 5.16（很差，ADE 5s = 12.74m）
- **结论：不是用个大 VLM 就行，architecture 和 training strategy 比模型大小重要**

### NAVSIM (Closed-Loop on nuPlan)
- 最好的 VA model (NaviHydra): PDMS = 92.7
- 最好的 VLA model (ReflectDrive): PDMS = 94.7
- **结论：VLA 在 closed-loop 上也开始领先**

### Bench2Drive (CARLA, 最严格)
- UniAD: DS = 45.81, SR = 16.36%
- SimLingo (VLA): DS = 85.94, SR = 67.27%
- **结论：好的 VLA 设计在 closed-loop 上大幅领先 VA**

---

## 几个关键 Model 的核心想法

### AutoVLA — 动作离散化 + RL
把连续 trajectory 用 VQ-VAE 量化成 discrete action tokens，跟 reasoning tokens 一起 autoregressive 生成。然后用 GRPO（DeepSeekMath 那套）做 RL fine-tuning，penalize 冗长无用的 reasoning。核心 insight：**让 reasoning 和 action 在同一个 token sequence 里统一**。

### SimLingo — Action Dreaming
把 speed waypoints 和 path waypoints 解耦——速度和路径分别预测，这样控制更精细。还有个 "action dreaming" 机制，把语言指令和 control sequence 对齐。核心 insight：**trajectory 的速度维度和空间维度有不同特性，应该分开建模**。

### ORION — Diffusion Action Head
不用 deterministic head，用 diffusion model 输出 trajectory。好处是能建模 multi-modal distribution——同一场景下有多种合理开法。核心 insight：**驾驶本质是 multi-modal 的，deterministic prediction 会 average out 多种可能**。

### DriveVLM — 慢快分工
VLM 做 chain-of-thought reasoning 给出 "减速+变道" 这种 high-level decision，传统 planner 把它变成 detailed trajectory。核心 insight：**让 VLM 做 it 最擅长的 reasoning，让 planner 做它最擅长的 control，各司其职**。

---

## 当前最大的坑

**坑一：Latency**
VLM 太慢了。Qwen2.5-VL 7B 模型 inference 要几百毫秒，但驾驶要求 sub-50ms。现在各种 hack：FastDriveVLA 做 token pruning，ETA 做 asynchronous reasoning，但还是没完全解决。

**坑二：Hallucination**
VLA 能生成很流畅的 reasoning text，但这 text 不一定真的对应它的 internal decision。可能它 action 对了但 explanation 是编的，也可能 explanation 听起来合理但 action 是错的。这在 safety-critical 场景下很危险。

**坑三：Long-tail**
遇到训练数据里没见过的场景（奇怪路况、罕见交通情况），VLA 还是会 fail。ImpromptuVLA 专门搞 80K corner-case 数据来缓解，但远没解决。

**坑四：Temporal Coherence**
Transformer 的 context window 有限，长时间序列的 reasoning 会断片。开车需要持续几秒甚至几十秒的 planning horizon，现在的架构还是 short-term conditioning。

**坑五：没有 domain-specific foundation model**
大家都在用 Qwen2.5-VL、InternVL 这种通用 VLM，但这些模型不是为开车设计的。需要专门预训练的 driving foundation model。

---

## 未来往哪走？

**方向一：VLA + World Model 统一**
现在 VLA 是 reactive——看当下，决定怎么开。未来应该是 proactive——能"想象"如果这么开，未来会怎样，然后选最好的。就是把 World Model（能预测未来场景演化）和 VLA（能 reasoning + action）合起来。

**方向二：更好的 Multimodal Fusion**
现在很多 VLA 只用 camera，但 LiDAR、Radar 的 3D 几何信息对安全至关重要。Language 提供 semantic grounding，geometry 提供 spatial precision，两者要 tight fusion。

**方向三：Continual Learning**
现在模型训完就固定了，但道路会变、驾驶习惯会变。需要能在线持续学习，又不 catastrophic forgetting。

**方向四：更好的 Evaluation**
现在 benchmark 主要看 trajectory 精度，但 VLA 特有的风险——reasoning 错误、instruction following 失败、hallucination——这些没有被 benchmark 覆盖。需要新的评估协议。

---

## 我的核心 takeaway

这篇 paper 读下来，我的 intuition 是：

**VLA 不是简单的"给自动驾驶加个大模型"，而是范式转变——从"系统架构设计"转向"agent 能力构建"。**

老的 AD 是 engineer 去设计 perception module、prediction module、planning module 之间的接口。VLA 是让 model 自己学会 perception、reasoning、action 的统一。

这个转变跟 LLM 改变 NLP 的方式很像——从 task-specific pipeline 到 unified foundation model。但 AD 比 NLP 难得多，因为：
1. **Safety 是硬约束**，不能像 LLM 那样 hallucinate
2. **Real-time 是硬约束**，不能像 LLM 那样几秒生成一个回答
3. **Physical grounding 是硬约束**，不能像 LLM 那样只处理符号

最 promising 的方向，我觉得是 **Unified Vision-Language-World Model**——一个模型能看懂当下、能用语言推理、能想象未来、能输出动作。这接近于一个真正"会开车的人"的认知架构。

Dual-System 是现在的 pragmatic compromise，但 long-term 看，随着 hardware 加速和 model compression，End-to-End VLA 应该会 dominate，因为 tight coupling 带来的 joint optimization 优势是 architecture hack 没法替代的。

最值得关注的 open question：**language reasoning 和 action 之间到底有没有 causal link？** 如果没有，那 VLA 的 interpretability 就是 illusion——它只是在 post-hoc 编故事。如果有，我们怎么 measure 和 enforce 这个 link？这个问题可能比模型本身更难，但也是 VLA 能不能真正 deploy 的关键。

---

# Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future — 深度解读

## 1. High-level 定位与核心 Motivation

这篇 paper 是 WorldBench team (HKUST + ZJU + NUS + NTU 等) 在 2025 年底发布的一篇系统综述，专门聚焦 VLA 范式在自动驾驶领域的演化。它的核心价值在于 three-fold：(1) historical continuity，从 VA 追溯到 VLA；(2) domain-specific focus，不同于 robotics 通用 VLA survey；(3) fine-grained architectural taxonomy，区分 End-to-End vs Dual-System。

Project page: https://worldbench.github.io/vla4ad
GitHub: https://github.com/worldbench/awesome-vla-for-ad
HuggingFace Leaderboard: https://huggingface.co/spaces/worldbench/vla4ad

传统自动驾驶系统的核心问题是 modular "Perception-Decision-Action" pipeline 存在 cross-stage error propagation：perception 噪声被 downstream reasoning 和 control 放大。VA models (ALVINN, ChauffeurNet, UniAD, VAD, TransFuser, DriveTransformer) 通过 end-to-end mapping 缓解了这个问题，但仍然存在四个 fundamental limitations：
- **Limited Interpretability**: black box，safety-critical 场景下不可审计
- **Weak Generalization**: 缺乏 world knowledge，long-tail scenario 下脆弱
- **No Chain-of-Thought Reasoning**: pixels → actions 直接映射，无显式推理
- **No Language Understanding**: 无法 incorporate human instructions

VLA 通过 coupling VLM backbone with action-prediction head，将 perception、language reasoning、decision-making 联合建模，本质上是在 build 一个 human-compatible autonomous agent。

## 2. 核心数学形式化

论文给出的统一 formulation 非常简洁：

$$\mathbf{a}_t = H(F(\mathbf{x} | \boldsymbol{\theta})) \tag{1}$$

变量解析：
- $\mathbf{a}_t$: timestamp $t$ 时刻的 action 输出，可以是 trajectory waypoints、control signals 或 language tokens
- $\mathbf{x}$: multimodal input，包括 sensor observations (RGB, LiDAR)、latent representations (BEV, occupancy)、language instructions、ego-vehicle state
- $F(\cdot)$: VLM backbone，参数化为 $\boldsymbol{\theta}$，典型实现是 ViT + LLM decoder + bridge network
- $H(\cdot)$: action generation head，决定输出 formulation

这个公式看起来简单，但它 unifying 了所有 VLA 变体——不同的设计选择就体现在 $F$ 的架构和 $H$ 的形式上。

### Action Space 的四种范式

**(1) Discrete Trajectory** (Eq 2):
$$\mathbf{a}_t = \{(x_i, y_i)\}_{i=1}^{\Phi}$$
- $\Phi$: prediction horizon (未来步数，典型 3s 内的 waypoint 数)
- $(x_i, y_i) \in \mathbb{R}^2$: 第 $i$ 步的 2D Cartesian 坐标

代表方法：UniAD, VAD, DriveTransformer

**(2) Continuous Trajectory** (Eq 3):
$$\mathbf{a}_t = (v(t), \kappa(t)), \quad t \in [0, T]$$
- $v(t)$: speed profile over future horizon
- $\kappa(t)$: curvature profile
- $T$: future time horizon

这种表示 inherent captures vehicle dynamics 的连续性，适合 MPC-style planners。

**(3) Direct Control** (Eq 4):
$$\mathbf{a}_t = (\delta_t, \tau_t, \beta_t)$$
- $\delta_t$: steering angle, $\delta_t \in [\delta_{\min}, \delta_{\max}]$
- $\tau_t$: throttle input
- $\beta_t$: brake input

代表方法：LMDrive, DriveGPT4-V2。直接送 actuator，latency 最低，但缺少 trajectory-level reasoning。

**(4) Language Representation** (Eq 5):
$$\mathbf{a}_t = \{w_1, w_2, \ldots, w_T\}, \quad w_i \in \mathcal{V}$$
- $\mathcal{V}$: vocabulary
- $T$: sequence length
- $w_i$: vocabulary 中的 token

代表方法：DriveMLM, DriveGPT4, EMMA, AutoVLA。这是 VLA 最 distinctive 的设计——action 通过 LLM 的 language head 自回归生成。

## 3. 架构 Taxonomy 深度解析

论文的核心 taxonomy 是二维分类：

### Dimension 1: End-to-End VLA vs Dual-System VLA

**End-to-End VLA** (Section 4.1): 单一 model 同时承担 perception + reasoning + action。优点是 joint optimization、low latency interface、tight coupling；缺点是 VLM backbone 的 compute cost 与 real-time constraint 之间的张力。

**Dual-System VLA** (Section 4.2): 灵感来自 Kahneman 的 *Thinking, Fast and Slow*。VLM 作为 System 2 (slow deliberation) 生成 high-level guidance；specialized driving module 作为 System 1 (fast execution) 负责 real-time trajectory generation。这种 design 在 interpretability 和 safety-critical reactivity 之间取得 balance。

### Dimension 2: Textual vs Numerical Action Generator

在 End-to-End 内部进一步细分：

**Textual Action Generator** (Section 4.1.1):
- Meta-Actions: 离散 semantic decisions ("accelerate", "change lane", "turn left")
- Trajectory Waypoints as Text: 将 future coordinates 表达为 language tokens

代表方法：DriveMLM (用 LLM 输出 behavioral planning states), EMMA (Gemini-VLM unified pipeline), AlphaDrive (GRPO + RL refine meta-actions), DriveAgent-R1 (CoT + RL with trajectory/meta-action rewards), ImpromptuVLA (80K corner-case dataset pretraining), Drive-R1 (CoT alignment via RL)。

**Numerical Action Generator** (Section 4.1.2):
- Additional Action Head: 在 VLM backbone 上 attach MLP / GRU / diffusion head
- Additional Action Tokens: 将连续 actions discretize 成 codebook tokens，与 reasoning tokens 一起 autoregressive 生成

代表方法：
- LMDrive: ResNet + LLaMA/Vicuna + MLP head → closed-loop control signals
- ORION: EVA-02 + Vicuna-1.5 + diffusion-based predictor，modeling multi-modal trajectory distributions
- SimLingo: InternViT + Qwen2 + disentangled MLP (解耦 speed waypoints 和 geometric path waypoints)
- AutoVLA: discretize trajectory 成 action codebook + GRPO fine-tuning
- DriveMoE: Mixture-of-Experts，dynamic activate experts for "lane following" / "overtaking"

## 4. Dual-System VLA 详解

Dual-System 进一步分为：

### Explicit Action Guidance (Section 4.2.1)

VLM 输出 structured action guidance，下游 planner 转化为 low-level control。

**Meta-Action Guidance**:
- FasionAD: VLM issues meta-actions + learned switching mechanism
- LeapVAD: analytic branch (memory bank) + heuristic branch (retrieve prior meta-actions)
- Senna: Senna-VLM (commonsense VLM) + Senna-E2E (executor)
- DiffVLA: VLM-generated lateral/longitudinal decisions as one-hot priors → diffusion planner denoising
- DME-Driver: VLM Decision-Maker + dedicated Executor
- ReAL-AD: 三层 hierarchy (strategy → decision → operation)

**Waypoint Supervision**:
- DriveVLM: hierarchical reasoning-to-planning，VLM 通过 CoT 生成 meta-actions + coarse waypoints
- SOLVE: shared vision encoder + Trajectory CoT module，iteratively refine candidate waypoints

### Implicit Representations Transfer (Section 4.2.2)

VLM 在训练时作为 teacher，runtime 时不直接参与，将 reasoning 能力 distill 到 compact E2E network。

**Knowledge Distillation**:
- VLP: align BEV features + planning queries with pretrained language embeddings via contrastive + supervisory objectives
- VLM-AD: VLM 生成 free-form textual justifications + structured behavior labels，distill into planner via alignment head + action classification head
- VERDI: align perception, prediction, planning outputs with VLM-generated CoT explanations across all stages
- ALN-P3: full-stack co-distillation (perception tokens + predicted motions + planned trajectories)

**Multimodal Feature Fusion**:
- InsightDrive: language-guided scene representations，VLM-generated descriptions 通过 cross-attention modulate BEV features
- VLM-E2E: explicit modeling driver attention，fuse textual attention cues with BEV features via learnable gating
- NetRoller: extract latent reasoning variables from VLMs，adapt 成 compact features
- ReCogDrive: align linguistic priors with diffusion-based planner，RL refine trajectories
- ETA: asynchronous VLM reasoning + action-mask mechanism，ensure guidance without real-time cost

## 5. Input Modalities 数学表示

论文将 input $\mathbf{x}$ 分为四类，每类有 precise mathematical formulation：

**Sensor Inputs**:
- Visual Images: $\mathbf{x}_{img} \in \mathbb{R}^{N_c \times H \times W \times 3}$
  - $N_c$: camera 数量 (典型 6-8 for surround view)
  - $H, W$: image height/width
- LiDAR Point Clouds: $\mathbf{x}_{lidar} \in \mathbb{R}^{N_p \times D}, D \geq 4$
  - $N_p$: point 数量
  - $D$: 维度，包括 $x, y, z$, velocity, intensity

**Latent Representations**:
- BEV Features: $\mathbf{x}_{bev} \in \mathbb{R}^{C \times H_{bev} \times W_{bev}}$
  - $C$: feature channels
  - $H_{bev}, W_{bev}$: BEV grid spatial dimensions
- Occupancy Grids: $\mathbf{x}_{occ} \in \mathbb{R}^{C_{occ} \times X \times Y \times Z}$
  - $X, Y, Z$: 3D grid resolution
  - $C_{occ}$: occupancy feature channels (occupancy, flow, semantics)

**Language Inputs**:
$$\mathbf{x}_{lang} \in \mathbb{Z}^T \text{ or } \mathbf{x}_{lang} \in \mathbb{R}^{T \times D_{emb}}$$
- $T$: sequence length
- $D_{emb}$: embedding dimension (if token embeddings)

**Vehicle State**: $\mathbf{x}_{state} \in \mathbb{R}^{D_{state}}$
- $D_{state}$: state vector 维度，包括 speed, acceleration, steering angle, yaw rate, turn indicator status

## 6. Action Head 四种类型对比

| Type | 机制 | 代表方法 | 优势 | 劣势 |
|------|------|----------|------|------|
| **LH** (Language Head) | VLM language modeling head 直接输出 free-form text 或 discretized action tokens | DriveMLM, DriveGPT4, EMMA, AutoVLA | 强 interpretability，native reasoning | discrete language ↔ continuous control gap |
| **REG** (Regression) | Decoder + MLP head，直接预测 continuous values | LMDrive, DriveGPT4-V2, CoVLA-Agent | 兼容 planner/actuator，no quantization | 牺牲一些 interpretability |
| **SEL** (Trajectory Selection) | 评估 candidate trajectories set，选择 optimal | WoTE, SeerDrive | 保证 kinematic constraints | candidate set 设计依赖 |
| **GEN** (Trajectory Generation) | Diffusion / VAE，从 noise 迭代 refine | ORION, DiffVLA, DiffusionDrive, DriveMoE | capture multi-modality + uncertainty | 计算开销大 |

## 7. Datasets & Benchmarks 深度分析

### VA Datasets (Vision-Action)
- **BDD100K** (2020): 120M frames, 美国 diverse conditions, behavioral cloning 基础
- **nuScenes** (2020): 1.4M frames, 1000 scenes, 6-camera surround + LiDAR + radar + 3D boxes + trajectories
- **Waymo Open Dataset** (2020): 200M frames, high-resolution + long trajectories + HD maps
- **nuPlan** (2021): 4.6M frames, long-horizon ego trajectories + dense map context + closed-loop simulation interface
- **Argoverse 2** (2021): 300K scenes
- **Bench2Drive** (2024): 2M frames, CARLA V2, closed-loop evaluation
- **WOD-E2E** (2025): 800K frames, long-tail safety-critical scenarios + human preference annotations

### VLA Datasets (Vision-Language-Action)
- **BDD-X** (2018): 8.4M frames + 26K captions, 时间对齐 human rationales
- **Talk2Car** (2022): 400K frames + 12K captions, natural language → trajectory grounding
- **DriveLM** (2024): 4.8K real + 64K sim, 445K + 3.76M QA pairs, graph-structured QA targeting conditional reasoning
- **LMDrive** (2024): 3M frames + 64K instructions, language-guided closed-loop driving
- **ImpromptuVLA** (2025): 2M frames + 80K captions/instructions/QA, aggregated from 8 public datasets, focus on corner cases
- **CoVLA** (2025): 6M frames + 6M captions, real-world driving videos paired with behavior descriptions
- **MetaAD** (2025): 120K frames + 30K reasoning/plan/QA
- **DriveAction** (2025): 2.6K frames + 16.18K QA

### Benchmarks 实验数据分析

#### nuScenes Open-Loop (Table 5)
关键数据点：
- UniAD (VA baseline): L2=0.69m, CR=0.12
- VAD (VA): L2=0.37m, CR=0.14
- DriveTransformer (VA): L2=0.40m, CR=0.11
- Drive-R1 (VLA, InternVL2 + CoT alignment via RL): L2=0.31m, CR=0.09
- AutoVLA (VLA, Qwen2.5-VL + GRPO): L2=0.33m, CR=0.10
- Reasoning-VLA: L2=0.22m (best!), CR=0.07
- EMMA (Gemini-VLM): L2=0.32m, CR=未报告

观察：VLA models 在 open-loop 上已经接近或超越 VA models，但要小心 overfitting nuScenes 的 closed distribution。

#### WOD-E2E (Table 6) — 最关键 benchmark
WOD-E2E 引入了 **Rater Feedback Score (RFS)**，这是基于 human preference annotations 而非 logged expert trajectories 的评估，更接近 real-world deployment 评估。

关键数据：
- Waymo Baseline: RFS(Overall)=7.53, RFS(Spotlight)=6.60, ADE 5s=3.02, ADE 3s=1.32
- AutoVLA: RFS(Overall)=7.99, RFS(Spotlight)=6.94, ADE 5s=2.74, ADE 3s=1.21
- Poutine: RFS(Overall)=7.99, RFS(Spotlight)=6.89, ADE 5s=2.96, ADE 3s=1.35
- dVLM-AD (LLaDA-V + controllable reasoning): RFS(Overall)=7.63, ADE 5s=3.02
- OpenEMMA (Qwen2-VL): RFS(Overall)=5.16, ADE 5s=12.74 (明显落后)
- LightEMMA (Qwen2.5-VL): RFS(Overall)=6.52, ADE 5s=3.73

观察：VLA models 在 RFS 上表现 diverse，strong VLA (AutoVLA, Poutine, HMVLM) 接近或超越 VA baselines，但 weak VLA (OpenEMMA) 显著落后。这表明 architecture + training strategy 比 backbone scale 更重要。

#### NAVSIM (Table 7) — Closed-Loop on nuPlan
NAVSIM 的 PDMS (Predictive Driver Model Score) 聚合了多个 sub-metrics:
- NC (No Collision)
- DAC (Driving Admissibility Check)
- TTC (Time-To-Collision)
- C (Comfort)
- EP (Ego Progress)

关键数据：
- TransFuser (VA, 2022): PDMS=84.0
- UniAD (VA): PDMS=83.4
- DiffusionDrive (VA, GEN): PDMS=88.1
- WoTE (VA, SEL + BEV world model + reward-guided): PDMS=88.3
- AD-R1 (VA, RL): PDMS=91.9
- NaviHydra (VA, SEL): PDMS=92.7
- AutoVLA (VLA, LH + Best-of-N oracle): PDMS=92.1, NC=99.1, EP=87.6
- ReflectDrive (VLA, GEN + LLaDA-V): PDMS=94.7 (current SOTA)
- AdaThinkDrive (VLA, InternVL3 + adaptive thinking RL): PDMS=93.0

观察：VLA models 在 NAVSIM 上已经 systematic 超越 VA models，ReflectDrive 达到 94.7 PDMS。TTC 和 EP 是 more discriminative indicators。

#### Bench2Drive (Table 8) — CARLA Closed-Loop
最严格的 closed-loop evaluation，metrics 包括 DS (Driving Score), SR (Success Rate), Efficiency, Comfort。

关键数据：
- UniAD-Base (VA): DS=45.81, SR=16.36%
- VAD (VA): DS=42.35, SR=15.00%
- DriveTransformer (VA): DS=63.46, SR=35.01%
- Raw2Drive (VA, RL): DS=71.36, SR=50.24%
- GuideFlow (VA, GEN): DS=75.21, SR=51.36%
- ORION (VLA, GEN): DS=77.74, SR=54.62%
- AutoVLA (VLA, LH+REG): DS=78.84, SR=57.73%
- SimLingo (VLA, REG + action dreaming): DS=85.94, SR=67.27% (current SOTA)
- CoReVLA (VLA, LH, dual-stage collect-and-refine): DS=72.18, SR=50.00%
- ReasonPlan (VLA, LH): DS=64.01, SR=34.55%

观察：SimLingo 的 action dreaming mechanism (align natural language instructions with control sequences) 在 closed-loop 上取得显著领先。VLA models 整体上已经超越 VA baselines，但 variance 很大，architecture design 至关重要。

## 8. Evolutionary Narrative: VA → VLA

### Vision-Action Models (Section 3)

**End-to-End Models** 分两类：

**(1) Action-Only Model**:
- Imitation Learning: ALVINN (1988), ChauffeurNet, TransFuser, NEAT, TCP, BEV-Planner, Urban-Driver
- Reinforcement Learning: Latent-DRL, LSD, LBC, WoR, Roach, Think2Drive (MBRL + latent world model), Raw2Drive (dual-stream MBRL), RAD (3DGS-based closed-loop RL)

**(2) Perception-Action Model**:
- Dense BEV-Based: ST-P3, UniAD, VAD, OccNet, Para-Drive, GenAD, DiffusionDrive, GuideFlow
- Sparse Query-Based: SparseAD, SparseDrive, DiFSD, DriveTransformer, GaussianAD

### World Models (Section 3.2)

按 prediction modality 分三类：

**(1) Image-Based World Models**:
- Diffusion-based: GenAD, Vista, Imagine-2-Drive, DriveDreamer, Drive-WM
- Autoregressive: DrivingWorld (GPT-style), DrivingGPT (interleaved image + action tokens), Epona (AR + diffusion)

**(2) Occupancy-Based World Models**:
- OccWorld, RenderWorld, OccVAR, T³Former (triplanes), Drive-OccWorld, DFIT-OccWorld, NeMo

**(3) Latent-Based World Models**:
- LAW (self-supervised future feature prediction), World4Drive, Echo-Planning (CFC cycle), Covariate-Shift, SeerDrive

### VA → VLA 的 four key motivations:
1. **Interpretability**: VA 是 black box，VLA 可以 articulate reasoning via language
2. **Generalization**: VA 缺乏 world knowledge，VLA leverage large-scale pretraining
3. **Chain-of-Thought**: VA 直接 pixels → actions，VLA 支持 step-wise reasoning
4. **Language Understanding**: VA 无法 incorporate human instructions，VLA native 支持

## 9. Key Models 深度技术解析

### AutoVLA (Section 4.1.2, NeurIPS 2025)
[Paper](https://arxiv.org/abs/2506.13757)

核心创新：
- **Discretize continuous trajectories into action codebook**: 用 VQ-VAE 将 trajectory waypoints 量化成 discrete tokens
- **Unified autoregressive generation**: reasoning tokens + action tokens 在同一 sequence 中 autoregressive 生成
- **GRPO fine-tuning**: penalize redundant reasoning，improve token efficiency
- **Fast/Slow thinking**: adaptive reasoning depth based on scene complexity

实验结果：
- nuScenes: L2=0.33m, CR=0.10
- WOD-E2E: RFS(Overall)=7.99, ADE 3s=1.21
- NAVSIM: PDMS=92.1, NC=99.1
- Bench2Drive: DS=78.84, SR=57.73%

### SimLingo (CVPR 2025)
[Paper](https://arxiv.org/abs/2410.23262)

核心创新：
- **Disentangled MLP head**: 解耦 temporal speed waypoints 和 geometric path waypoints，enable finer-grained control
- **Action Dreaming**: align natural language instructions with control sequences，open-loop evaluated via success rate
- **Vision-only closed-loop**: 不依赖 LiDAR，纯视觉 + language

实验结果：
- Bench2Drive: DS=85.94 (SOTA), SR=67.27%, Efficiency=244.18, Comfort=25.49

### ORION (ICCV 2025)
[Paper](https://arxiv.org/abs/2503.19755)

核心创新：
- **Diffusion-based predictor**: 替代 deterministic head，modeling multi-modal trajectory distributions under uncertainty
- **Vision-Language instructed action generation**: EVA-02 (vision) + Vicuna-1.5 (language) + diffusion action head

实验结果：
- nuScenes: L2=0.34m, CR=0.37
- Bench2Drive: DS=77.74, SR=54.62%

### DriveVLM (CoRL 2024)
[Paper](https://arxiv.org/abs/2402.12289)

核心创新 (Dual-System 代表):
- **Hierarchical reasoning-to-planning**: VLM 通过 CoT 生成 meta-actions + coarse waypoints
- **Conventional planners**: 将 coarse waypoints transform 成 detailed trajectories
- 分工：VLM 负责 "what to do" (high-level)，planner 负责 "how to do" (low-level)

### DriveLM (ECCV 2024)
[Paper](https://arxiv.org/abs/2312.06585)

核心创新：
- **Graph-structured VQA**: 把 autonomous driving 建模为 graph-structured visual question answering
- **Conditional reasoning**: multi-stage perception → prediction → planning
- **Textualized trajectory waypoints**: future coordinates 通过 language tokens 生成

### AlphaDrive (arXiv 2025)
[Paper](https://arxiv.org/abs/2503.07608)

核心创新：
- **GRPO (Group Relative Policy Optimization)**: refine meta-actions via RL
- **Multi-objective rewards**: trajectory quality + decision correctness + format consistency

## 10. Action Space 设计哲学

这是 paper 中最值得深入思考的部分。不同 action space 的选择反映了不同 design philosophy：

### Discrete Trajectory vs Continuous Trajectory
- **Discrete**: explicit geometric path planning，适合 open-loop evaluation (L2, CR metrics)，但 waypoint 之间的 interpolation 依赖 downstream controller
- **Continuous**: captures vehicle dynamics 的连续性，适合 MPC-style planners，但 training data 要求更高

### Direct Control vs Trajectory
- **Direct Control**: $(\delta_t, \tau_t, \beta_t)$ 直接送 actuator，latency 最低，但缺少 trajectory-level reasoning，容易导致 jittery behavior
- **Trajectory**: 提供 future planning horizon，allow smoother control，但需要 downstream controller (PID, MPC) 转化为 control signals

### Language Representation 的双刃剑
将 actions 表达为 language tokens 是 VLA 最 distinctive 的设计，但也带来 fundamental tension：
- **优势**: leverage LLM 的 reasoning ability，native interpretability，unified sequence modeling
- **劣势**: discrete language ↔ continuous control 的 quantization gap，可能引入 precision limits，extreme cases 下 trajectory collapse

这也是为什么 AutoVLA, OpenDriveVLA 等方法探索 action codebook discretization——试图在 language token 的统一性和 trajectory 的连续性之间找到 balance。

## 11. Challenges & Future Directions 深度思考

### Current Challenges (Section 6.1)

**Model Architecture & System Efficiency**:
1. **Real-time Processing & Latency**: VLM backbone 的 compute cost 巨大，high-resolution + high-frame-rate camera 产生 long visual-token sequences。Sub-50ms inference 仍是 unmet requirement。FastDriveVLA (token pruning), ETA (asynchronous reasoning) 是 promising directions。

2. **Lack of Domain-Specific Foundation Models**: general-purpose VLMs (Qwen2.5-VL, InternVL, Gemini) 不是为 driving-specific perception / physics / multi-sensor fusion 优化。需要 dedicated driving foundation models (类似 MindVLA from Li Auto)。

**Data & Generalization**:
3. **Long-tail Scenarios**: misbehaving traffic agents, unusual road layouts, unpredictable weather 仍然是 failure points。ImpromptuVLA 的 80K corner-case dataset 是 promising direction。

4. **Cost of High-Quality Data**: vision-action-language triplets at scale 非常 expensive。Synthetic environments (3DGS, Cosmos) 有 sim-to-real gap。

**Core Capabilities & Trustworthiness**:
5. **Interpretability & Hallucination**: VLA 产生的 natural-language rationales 是 generated artifacts，不一定 faithful 反映 underlying causal reasoning。Language hallucination 是新风险——model 可能用 confident 但 spurious 的 narrative justify 错误 decision。

6. **Long-Horizon Temporal Coherence**: transformer-based VLA 受限于 context window 和 short-term conditioning，temporal fragmentation 导致 inconsistent decisions。

### Future Directions (Section 6.2)

**Next-Generation Model Paradigms**:
1. **Unified Vision-Language-World Models**: integrate VLA with predictive world models，simulate future scene evolution conditioned on candidate actions，enable proactive planning。这其实是将 Section 3.2 的 World Models 与 Section 4 的 VLA 统一起来。

2. **Richer Multimodal Fusion**: early + tight fusion of LiDAR, Radar, event cameras, HD maps。Language 提供 semantic grounding，但 robust 3D geometry 对 safe decision-making 不可或缺。

**Advancing Intelligence & Adaptation**:
3. **Socially Aware, Knowledge-Grounded Driving**: deeper commonsense reasoning — intent, conventions, causal relationships。Leverage large-scale video-language corpora + external knowledge bases + structured reasoning modules。

4. **Continual & Onboard Learning**: static, offline-trained models 无法 capture evolving road infrastructures 或 regional driving customs。Safe incremental learning while avoiding catastrophic forgetting 是 essential。

**Ecosystem for Safe Deployment**:
5. **Standardized Evaluation & Safety Guarantees**: 当前 benchmarks 不 capture VLA-specific risks (reasoning failures, instruction-following errors, cross-modal inconsistencies)。需要 multi-step instruction execution + ambiguous language robustness + hallucination resistance benchmarks。Formal verification tools 提供 theoretical safety guarantees。

6. **Human-Centric Interaction & Personalization**: natural language enable drivers specify goals/constraints/preferences ("drive cautiously", "avoid unprotected left turns")。Personalization modules adapt driving styles to different users。Challenge 是 balance personalization with strict safety/regulatory requirements。

## 12. 我的延伸思考与 Open Questions

### 12.1 Scaling Laws in VLA for AD
一个 fundamental question: VLA in AD 是否存在类似 LLM 的 scaling laws? 从 Table 5-8 的数据看，Qwen2.5-VL (3B/7B) 系列在多个 benchmarks 上表现 stable，但 Gemini-VLM (EMMA) 在 nuScenes 上 L2=0.32 表现优秀，却在 WOD-E2E 上 RFS 未报告。这暗示 driving-specific 的 scaling law 可能与 general VLM 不同——data quality > model scale。

### 12.2 The Language-Action Alignment Problem
论文中反复提到 language reasoning 和 action generation 之间的 alignment 问题。RDA-Driver, Drive-R1, AlphaDrive 都试图通过 RL 来 enforce consistency。但 deep question: language explanation 和 action 之间是否真的存在 causal link，还是 post-hoc rationalization?

这是 AI safety 中 **deference to internal state** 的核心问题。如果 VLA 产生的 rationale 只是 post-hoc narrative，那 interpretability 就是 illusion。可能的解决方案：
- **Causal interventions**: 改变 rationale 应该 causal 改变 action
- **Counterfactual probing**: "如果场景是 X 而非 Y，rationale 应该改变吗？"
- **Mechanistic interpretability**: 直接 inspect internal representations

OmniDrive 已经探索 counterfactual reasoning，但这个方向还远未成熟。

### 12.3 World Models + VLA 的统一
论文 Section 6.2.1 提出 Unified Vision-Language-World Models 作为 future direction。这其实是将 model-based RL (Think2Drive, Raw2Drive) 与 VLA 结合。核心 insight: 如果 VLA 能 simulate future scene evolution conditioned on candidate actions，就能：
- Proactive planning (而非 reactive)
- Uncertainty estimation (multi-modal futures)
- Counterfactual reasoning (what-if analysis)

实现路径可能：
- **Latent world model + VLA**: 在 latent space 中 simulate futures，避免 pixel-level generation 的 compute cost
- **Diffusion-based world model + VLA**: 用 diffusion model 生成 multi-modal futures，VLM 提供 language-conditioned guidance
- **Autoregressive world model + VLA**: 类似 DrivingGPT，interleave image + action tokens

### 12.4 The "Slow Thinking" Bottleneck
Dual-System VLA 的核心 insight 是 VLM 作为 System 2 (slow thinking) + planner 作为 System 1 (fast execution)。但 VLM 的 inference latency 通常 100-500ms，远超 driving 的 real-time requirement (sub-50ms)。

ETA 的 solution (asynchronous reasoning + action-mask) 是 promising direction。另一个可能 direction: **speculative decoding for VLA** — fast planner 先 propose action，VLM 在后台 verify/refine，类似 ARM 芯片的 speculative execution。

### 12.5 From Closed-Loop to Open-World
当前 benchmarks (Bench2Drive, NAVSIM) 都是 closed-loop simulation，与 real-world deployment 仍有 gap。WOD-E2E 引入 human preference annotations 是重要 step，但仍限于 fixed scenarios。真正的 open-world evaluation 需要：
- **Continual learning benchmarks**: scenarios 随时间 evolve
- **Adversarial robustness**: other agents 的 adversarial behaviors
- **Distribution shift metrics**: train/test distribution divergence 的 quantitative measures

### 12.6 The Hallucination Problem in Safety-Critical Systems
VLA 的 hallucination 比 LLM hallucination 更 dangerous——一个 confident 但错误的 rationale 可能导致 fatal accident。可能的 mitigation:
- **Confidence calibration**: VLA 应该知道自己的 uncertainty
- **Safety envelopes**: hard constraints (collision avoidance) independent of VLM reasoning
- **Redundancy**: 多个 independent systems vote (类似 airplane 的 triple modular redundancy)

## 13. 实用工程 Insights

### 13.1 Backbone Choice
从 Table 3 可以看出 backbone 分布：
- **Qwen2.5-VL** 是 most popular (AutoVLA, FastDriveVLA, AutoDrive-R2, CoReVLA, VDRive, SpaceDrive, etc.)
- **LLaMA-2/3** 系列: DriveMLM, DriveGPT4, OccLLaMA, OmniDrive, WKER
- **InternVL/InternViT**: SimLingo, AdaThinkDrive, Percept-WAM, ReCogDrive
- **Gemini**: EMMA (Google Waymo)
- **LLaVA-1.5**: DiMA, SOLVE, LMAD, OmniReason
- **EVA-02**: ORION, OmniDrive, WKER, OmniReason

Qwen2.5-VL 占据主流的原因：strong performance + permissive license + efficient inference + good vision-language alignment。

### 13.2 Action Head Choice Heuristics
- **Open-loop benchmarks (nuScenes)**: LH + REG 表现接近，GEN 在某些 case 更好 (Reasoning-VLA L2=0.22)
- **Closed-loop benchmarks (Bench2Drive)**: REG (SimLingo DS=85.94) > LH > GEN，因为 continuous output 兼容 controller
- **Real-time deployment**: REG + token pruning (FastDriveVLA) > LH (autoregressive overhead) > GEN (diffusion iterative overhead)

### 13.3 Data Efficiency
从 Table 4 可以看出 dataset scale 差异巨大：
- BDD100K: 120M frames (VA)
- CoVLA: 6M frames + 6M captions (VLA)
- ImpromptuVLA: 2M frames + 80K captions (VLA, focus on corner cases)
- DriveLM: 4.8K real scenes + 445K QA (VLA, graph-structured)

ImpromptuVLA 的 insight: **corner-case focused data** 比 scale 更重要。80K corner-case clips 比 6M generic clips 对 long-tail generalization 更有效。

## 14. 相关 References 与进一步阅读

### Core Papers
- **UniAD** (CVPR 2023): https://arxiv.org/abs/2212.10156
- **VAD** (CVPR 2023): https://arxiv.org/abs/2303.12077
- **DriveTransformer** (ICLR 2025): https://arxiv.org/abs/2410.22061
- **TransFuser** (PAMI 2022): https://arxiv.org/abs/2105.05977
- **DriveLM** (ECCV 2024): https://arxiv.org/abs/2312.06585
- **DriveVLM** (CoRL 2024): https://arxiv.org/abs/2402.12289
- **LMDrive** (CVPR 2024): https://arxiv.org/abs/2312.07988
- **EMMA** (TMLR 2025): https://arxiv.org/abs/2410.23262
- **AutoVLA** (NeurIPS 2025): https://arxiv.org/abs/2506.13757
- **SimLingo** (CVPR 2025): https://arxiv.org/abs/2410.23262
- **ORION** (ICCV 2025): https://arxiv.org/abs/2503.19755
- **AlphaDrive** (arXiv 2025): https://arxiv.org/abs/2503.07608
- **ImpromptuVLA** (NeurIPS 2025): https://arxiv.org/abs/2505.23757
- **CoVLA** (WACV 2025): https://arxiv.org/abs/2407.07726

### Datasets & Benchmarks
- **nuScenes**: https://www.nuscenes.org
- **Waymo Open Dataset**: https://waymo.com/open
- **nuPlan**: https://www.nplan.io
- **Bench2Drive**: https://github.com/Thinklab-SJTU/Bench2Drive
- **NAVSIM**: https://github.com/autonomousvision/navsim
- **WOD-E2E**: https://arxiv.org/abs/2510.26125
- **DriveLM**: https://github.com/OpenDriveLab/DriveLM
- **BDD-X**: https://github.com/explain-away/ddx
- **ImpromptuVLA**: https://impromptu-vla.github.io
- **CoVLA**: https://github.com/CVLAB-Unibo/VLA-Dataset

### Surveys & Related
- **End-to-End AD Survey** (PAMI 2024): https://arxiv.org/abs/2306.16927
- **3D/4D World Models Survey**: https://arxiv.org/abs/2509.07996
- **VLA for Embodied AI Survey**: https://arxiv.org/abs/2405.14093
- **VLA Models Survey**: https://arxiv.org/abs/2502.06851

### Open-source Implementations
- **Awesome VLA for AD**: https://github.com/worldbench/awesome-vla-for-ad
- **VLA4AD Leaderboard**: https://huggingface.co/spaces/worldbench/vla4ad
- **OpenEMMA**: https://github.com/realsa-org/OpenEMMA
- **OpenDriveVLA**: https://github.com/OpenDriveLab/OpenDriveVLA

## 15. 总结性 Intuition

这篇 paper 给我的 core intuition:

1. **VLA 是 AD 的 paradigm shift**: 从 modular "Perception-Decision-Action" pipeline 到 joint reasoning + action 的 holistic agent。这类似 LLM 改变 NLP 的方式——从 task-specific architectures 到 unified foundation models。

2. **Dual-System 是 practical compromise**: 纯 End-to-End VLA 在 real-time constraint 下 struggle，Dual-System (VLM slow + planner fast) 是近期 practical solution。但 long-term 看，hardware acceleration + model compression 可能让 End-to-End VLA 变得 feasible。

3. **Language 是 interface, not just output**: VLA 的真正 power 在于 language 作为 human-AI interaction 的 unified interface——可以是 instructions, explanations, rationales, 或 actions。这 enable 了 personalization + interpretability + instruction-following 的统一。

4. **World Models 是 missing piece**: 当前 VLA 是 reactive (perceive → reason → act)，future VLA 应该 be proactive (perceive → simulate futures → reason → act)。World Models 提供 "imagination" capability。

5. **Evaluation 需要革命**: 当前 benchmarks 主要是 trajectory-based metrics (L2, CR) 或 closed-loop simulation (DS, SR)。VLA-specific risks (hallucination, instruction-following errors, reasoning failures) 需要 new benchmarks。WOD-E2E 的 RFS 是重要 step，但还不够。

6. **Safety 是 fundamental**: VLA 在 safety-critical domain deployment，hallucination + interpretability illusion + long-tail failure 是 real risks。需要 formal verification + redundancy + confidence calibration 的综合 solution。

这篇 survey 的最大价值在于提供 systematic taxonomy 和 historical context，让研究者能 quickly identify open problems 和 promising directions。对 Karpathy 这样的 researcher 来说，最 exciting 的方向可能是 Unified Vision-Language-World Models + causal reasoning + continual learning——这些是通向 truly human-compatible autonomous agents 的关键。
