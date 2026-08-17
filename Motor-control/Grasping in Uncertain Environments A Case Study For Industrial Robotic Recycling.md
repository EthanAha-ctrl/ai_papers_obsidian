---
source_pdf: Grasping in Uncertain Environments A Case Study For Industrial Robotic
  Recycling.pdf
paper_sha256: 99fbab5f7b2970c245896d0f50048306a7ba78bdc695585230bb14498a2da9b8
processed_at: '2026-08-04T22:20:51-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

## 从一个场景讲起

想象 你站在 Athens 的一个 WEEE recycling plant 里。面前堆着 old microwave、broken PC tower、emergency lamp。你的 job 是把这些东西拆开，把里面的 copper、aluminum、steel 分出来回收。问题是这些东西：

- Dirty，上面全是 dust 和 grease
- Damaged，外壳变形，screws 生锈
- Unidentified，没有 CAD model，没有标签
- 照明条件差，camera 看不清

传统 approach 是 vision pipeline：用 camera 拍照 → object detection → pose estimation → plan grasp → execute。但在这个 plant 里，vision 给你的 position estimate 可能偏了 2-3 cm，orientation 可能差了 15-20 度。如果你直接信 vision 的数据去 grasp，十有八九会 miss 或者撞坏 object。

这篇 paper 的 insight 很简单：**既然 vision 不靠谱，那就让 robot "摸"着找**。就像你闭着眼睛在桌上找手机，手碰到东西了就知道在哪了。

---

## 核心问题用大白话讲

Vision system 告诉 robot 三件事：

$$
\tilde{\boldsymbol{x}}_{\mathrm{obj}} = \boldsymbol{x}_{\mathrm{obj}} + \boldsymbol{\sigma}_{\boldsymbol{x}} \in \mathbb{R}^{3}
$$

$\boldsymbol{x}_{\mathrm{obj}}$ 是 object 真实中心位置（不知道），$\boldsymbol{\sigma}_{\boldsymbol{x}}$ 是 vision 误差（也不知道分布），$\tilde{\boldsymbol{x}}_{\mathrm{obj}}$ 是 vision 报上来的带噪声的位置。同理 $\tilde{\pmb{n}}_{\mathrm{obj}}$ 是带噪声的法向量（告诉你 object 朝哪个方向），$\tilde{\pmb{d}}_{\mathrm{obj}}$ 是带噪声的尺寸。

关键 constraint：robot 拿到这些 noisy data 后，vision 就断线了。整个 grasp 过程里 robot 只能靠 wrist 上的 FT sensor 感知。这就像给你一张模糊的地图，然后把你扔到野外，只给你一个指南针。

---

## 三种 gripper 对应三种"摸"法

### Gripper A：Tactile Finger（带 pressure sensor array 的手指）

**场景**：抓 emergency lamp 里的 light bulb。Glass 很脆，用力不对就碎。

**核心 trick**：计算 Center of Pressure (CoP)。

$$
^{\mathrm{EE}} p_{\mathrm{CoP}} = [0, 0, ^{\mathrm{EE}} z_{\mathrm{CoP}}]^{\top}
$$

这里的 $^{\mathrm{EE}} z_{\mathrm{CoP}}$ 是从 End-Effector 到压力中心的 z 轴距离。它等于 $^{\mathrm{EE}} z_{\mathrm{TCP}} - ^{\mathrm{TCP}} z_{\mathrm{CoP}}$，也就是 EE 到 TCP 的距离减去 TCP 到 CoP 的距离。

**为什么 care CoP？** 想象你拧灯泡。如果你的手指捏在灯泡正中间，旋转时几乎没有 torque 浪费。如果你捏在灯泡边缘，旋转时会产生很大的 lever arm，glass 可能碎。CoP 就是告诉你 "你实际捏在哪了" 的量。

Algorithm 流程：
1. Approach 直到 contact
2. Compliant close（手指在 x/y 方向 compliant，z 方向 force control）
3. 如果 CoP 在 fingertip 附近（说明捏偏了），手指微张，往前挪 1 cm，再闭
4. 尝试 lift
5. 如果 lift 时 force 很大（说明被卡住），就绕 CoP 旋转
6. 重复直到 success 或 timeout

这整套 flow 本质上是在做 **contact-based localization**：用 force feedback 不断 refine grasp 的 position 和 orientation，直到物理上 feasible。

### Gripper B：Slim Finger（3D printed narrow fingers）

**场景**：抓 emergency lamp 里的 battery。Battery 藏在 housing 旁边，空间很窄。

**核心 trick**：Human-like blind search。

因为 slim finger 的 opening width $l_{width}$ 很小，vision 误差可能比 opening width 还大。直接闭合肯定 miss。

Algorithm：
1. 移动到 estimated position 旁边 $1.2 l_{width}$ 处
2. 慢慢朝 estimated position 方向移动
3. Contact 了就说明找到 object 边缘了
4. 闭合 grasp
5. 如果没 contact，换另一侧再来

这就是 blind search 的精髓：用 contact event 作为 binary detector，把 continuous position uncertainty 降到一个 point。Cost 是时间，但 success rate 极高。

### Gripper C：Vacuum Gripper（suction cups）

**场景**：抓 PC tower 的 cover、microwave 的 cover。大面积 flat 表面。

**核心 trick**：45-degree rotation heuristic。

Vacuum grasp 的 failure mode 是 suction cup 没 seal 上（因为 surface 不平或者 angle 不对）。如果你 grasp 失败了，gripper 旋转 45 度再试。最多旋转 7 次（360 / 45 = 8，但第一次不算）。

**为什么 45 度？** 这是个 heuristic。Industrial parts 的 edges 通常是 orthogonal 的（90 度对称），所以 45 度 rotation 保证你最多 2 次就能 align 到一个 edge 方向。如果 surface 有 multiple features，45 度 grid search 能 cover 所有可能的 principal orientations。

这个 heuristic 的 beauty 在于：不需要任何 machine learning，不需要 tactile image processing，纯靠 geometry + counting。Engineering 上极致 pragmatic。

---

## Control 的底层逻辑

Eq 4 是整个 system 的 control law：

$$
\pmb{u \nu}_{\mathrm{EE, des}} = S_{\mathrm{vel}} \pmb{s} \pmb{\mathcal{V}}_{\mathrm{EE, max}} + S_{\mathrm{frc}} K_P (\pmb{\mathcal{F}}_{des} - \pmb{\mathcal{F}})
$$

用人话拆解：
- $\pmb{u \nu}_{\mathrm{EE, des}}$：最终发给 robot 的 velocity command
- $S_{\mathrm{vel}}$：在哪些 axis 上用 velocity control（比如沿 x 移动）
- $S_{\mathrm{frc}}$：在哪些 axis 上用 force control（比如沿 z 压）
- $\pmb{\mathcal{V}}_{\mathrm{EE, max}} \cdot \pmb{s}$：最大 velocity 乘以 scaling factor
- $K_P (\pmb{\mathcal{F}}_{des} - \pmb{\mathcal{F}})$：force error 的 P control

**Intuition**：想象你往下按东西，你希望 z 方向保持一个恒定的力（比如 10N），但 x/y 可以自由移动。那 $S_{\mathrm{frc}}$ 在 z 上是 1，$S_{\mathrm{vel}}$ 在 z 上是 0。反过来 x/y 上 $S_{\mathrm{vel}}$ 是 1，$S_{\mathrm{frc}}$ 是 0。这就是 hybrid control 的本质——**不同 axis 上用不同 control modality**。

Eq 5 的 Adjoint map 是 frame transformation：

$$
\pmb{\mathcal{V}}_{\mathrm{EE, des}} = \mathbf{Ad}_{\mathrm{EE}}^{\mathsf{T}} T_{\mathrm{TCP}} \pmb{\mathcal{V}}_{\mathrm{TCP, des}}
$$

因为 FT sensor 装在 EE 上，但 gripper 的 contact point 在 TCP，中间隔着一个 rigid transformation。Adjoint map $\mathbf{Ad}$ 就是把 TCP frame 的 wrench/twist 转换到 EE frame 的数学工具。从 Lie group 角度，$SE(3)$ 的 twist 属于 Lie algebra $se(3)$，wrench 属于 dual space $se^*(3)$，Adjoint map 描述了它们在 frame 变换下的 covariance。

---

## 为什么不用 Deep Learning？

这是 paper 的 implicit choice，但值得深究：

1. **Data scarcity**：WEEE recycling 的 real-world data 极少。每种 device 形状不同，damage pattern 不同。你没法 collect 一个 large dataset 来 train CNN 或 RL policy。

2. **Interpretability**：Industrial deployment 要求 predictability。如果一个 learned policy fail 了，你不知道为什么。但一个 force-based heuristic fail 了，你 log 一下 force profile 就能 debug。

3. **Compute constraint**：Factory floor 的 controller 可能就是个 PLC 或 embedded PC，跑不动 large neural network。Force control 的 computation cost 是 $O(1)$ per timestep。

4. **Safety certification**：Force control 有 Lyapunov stability proof，可以 certify。Learned policy 很难 certify。

但这不代表 learning 没用。未来的方向可能是：
- 用 RL learn optimal search strategy（比如 B 的 $1.2 l_{width}$ 这个数字，RL 可以 learn 出最优 ratio）
- 用 Diffusion Policy 从 tactile image 直接 predict grasp success probability
- 用 VAE compress tactile data into latent space，在 latent space 做 planning

---

## 扩展联想：跟最新 Robotics Research 的 connection

### 1. Tactile SLAM

这篇 paper 的 blind search (Strategy B) 本质上是在做 **1D tactile SLAM**：用 contact event 作为 landmark，gripper pose 作为 state，estimation 通过 binary detection 更新。如果 generalize 到 3D，就是 tactile SLAM problem。

参考：[Tactile SLAM with GelSight](https://www.science.org/doi/10.1126/scirobotics.abf6061)

### 2. Diffusion Policy + Tactile

2023-2024 年 Diffusion Policy 在 manipulation 领域大火。如果把这篇 paper 的 force feedback 替换成 Diffusion Policy：

- Input：FT sensor 6D wrench history + current gripper pose
- Output：next EE velocity command
- Training data：human demonstration 或 self-supervised exploration

理论上 Diffusion Policy 可以 learn 出比 45-degree heuristic 更聪明的 search trajectory。但代价是需要 thousands of demonstrations。

参考：[Diffusion Policy (Chi et al. 2023)](https://diffusion-policy.cs.columbia.edu/)

### 3. Foundation Models for Grasping

Google 的 RT-2、Octo 等 vision-language-action models 可以 zero-shot grasp unseen objects。但它们依赖清晰 vision。如果把 FT sensor data 也 tokenize 进 VLM 的 context window，理论上可以做 vision-tactile fusion 的 foundation model。

参考：[Octo: Generalist Robot Policy](https://octo-models.github.io/)

### 4. Variable Impedance Control + Learning

Paper 里 $K_P$ 是固定的。但 optimal $K_P$ 应该 depend on uncertainty level：uncertainty 大就 soft，uncertainty 小就 stiff。这可以用 meta-learning 或 Bayesian optimization online tune。

参考：[Variable Impedance Control Review](https://www.annualreviews.org/doi/10.1146/annurev-control-072220-095931)

### 5. Sim-to-Real for Tactile

最大的 pain point 是 tactile simulation。现成的 simulator（Isaac Gym, MuJoCo）对 soft contact 的 simulation 很不准确。如果用 FEM-based tactile simulator（如 TACTO, Taxim），可以在 sim 里 train force-based policy，然后 sim-to-real。

参考：[Taxim: An Optical-based Tactile Simulator](https://arxiv.org/abs/2109.08027)

---

## Summary：这篇 paper 的真正 contribution

撇开 具体 algorithm，这篇 paper 的 philosophical contribution 是：

**在 uncertainty 极大的 industrial scenario 下，simple force-based heuristics + cheap hardware 可以 outperform complex vision-only pipeline。**

这跟当前 Robotics 领域 "end-to-end learning 解决一切" 的 narrative 形成对比。它提醒我们：real industrial deployment 关心的是 reliability、cost、maintainability，benchmark 上的 SOTA number 反而 secondary。

从 engineering intuition 角度，这篇 paper 教会我们几个 design principle：

1. **Modality-specific control**：不同 axis 用不同 control mode（force vs velocity）
2. **Contact as localization**：用 contact event 做 binary detection，比 continuous estimation 更 robust
3. **Heuristic search over continuous space**：45-degree grid search 是 discrete approximation of continuous orientation space
4. **CoP-based rotation**：物理量直接 parametrize control policy，比 learned latent space 更 interpretable

这些 principle 可以 transfer 到其他 uncertain manipulation tasks：food handling、agricultural harvesting、underwater manipulation 等等。

---

因为 WEEE (Waste Electrical and Electronic Equipment) recycling industry 面临 harsh environment 且 target object 存在 damage 和 dirt，所以 vision data 的 noise 极大。paper 提出 利用 inexpensive grippers 配合 FT (Force-Torque) sensor 执行 tactile feedback，从而 overcome vision 的 uncertainty。这篇 paper 的 core insight 在于：利用 simple, low-cost hardware 配合 deterministic, model-based control heuristics 来 solve uncertainty。这种 approach 区别于 依赖 massive neural networks 或 end-to-end RL 的 方法。

### Problem Formulation 与 Uncertainty 建模

vision system 提供 的 object estimates 存在 noise，如 Eq 1 描述了 geometric uncertainty：
$$
\tilde{\boldsymbol{x}}_{\mathrm{obj}} = \boldsymbol{x}_{\mathrm{obj}} + \boldsymbol{\sigma}_{\boldsymbol{x}} \in \mathbb{R}^{3} \tag{1a}
$$
$$
\tilde{\pmb{n}}_{\mathrm{obj}} = \pmb{n}_{\mathrm{obj}} + \pmb{\sigma}_n \in \mathbb{R}^{3} \tag{1b}
$$
$$
\tilde{\pmb{d}}_{\mathrm{obj}} = \pmb{d}_{\mathrm{obj}} + \pmb{\sigma}_d \in \mathbb{R}^{2} \tag{1c}
$$
其中 $\boldsymbol{x}_{\mathrm{obj}}$ 是 object 的 exact center (unknown)，$\pmb{n}_{\mathrm{obj}}$ 是 object 的 exact orientation (以 normal vector 形式表示)，$\pmb{d}_{\mathrm{obj}}$ 是 object 的 symmetric planar dimensions。$\boldsymbol{\sigma}_{\boldsymbol{x}}, \pmb{\sigma}_n, \pmb{\sigma}_d$ 分别 代表 location, orientation, dimensions 的 uncertainty。他们的 probability distribution 是 unknown 的。加 $\tilde{}$ 的变量 则是 vision unit 输出的 noisy measurement。

为了 grasp object，需要 计算 desired pose $^B T_{\mathrm{TCP, des}}$ (Eq 2)：
$$
{ ^ B T }_{\mathrm{TCP, des}} = \left[ \begin{array}{cc} ^B R_{\mathrm{TCP, des}} & p_{\mathrm{trans}} \\ 0 & 0 \end{array} \right] \in \mathbb{R}^{4 \times 4}
$$
其中 $^B R_{\mathrm{TCP, des}} \in \mathbb{R}^{3 \times 3}$ 是 rotation matrix，$p_{\mathrm{trans}} \in \mathbb{R}^{3}$ 是 translation vector。$p_{\mathrm{trans}}$ 直接 取值为 $\tilde{\boldsymbol{x}}_{\mathrm{obj}}$。

对于 rotation matrix $^B R_{\mathrm{TCP, des}}$ (Eq 3)，由 approach vector $\hat{\mathbf{a}}$ 和 orientation vector $\hat{\mathbf{\omega}}$ 决定：
$$
{ }^{\mathrm{B}} R_{\mathrm{TCP, des}} = \left[ \begin{array}{lll} n_x & o_x & a_x \\ n_y & o_y & a_y \\ n_z & o_z & a_z \end{array} \right]
$$
这里 $\hat{\mathbf{a}} = [ \hat{a}_x, \hat{a}_y, \hat{a}_z ]^{\mathsf{T}}$ 是 approach vector (tool 的 z-axis)，在 这个 case 中 $\hat{\mathbf{a}} = -\tilde{\pmb{n}}_{\mathrm{obj}}$，代表 tool 沿 着 object 的 normal vector 反方向 接近 object。$\hat{\mathbf{\omega}} = [ \hat{o}_x, \hat{o}_y, \hat{o}_z ]^{\mathsf{T}}$ 是 orientation vector，基于 object dimensions $\tilde{\pmb{d}}_{\mathrm{obj}}$ 推导，perpendicular 于 $\hat{\mathbf{a}}$。然后 normal vector $\pmb{n} = \hat{\pmb{o}} \times \hat{\pmb{a}}$ (cross product)，并且 $\pmb{o} = \pmb{a} \times \pmb{n}$。这种 frame 定义方式 确保了 gripper 能够 准确对齐 object 的 main axes。

### System Architecture 与 Hybrid Control

Fig 3 展示了 system architecture。Task planner 作为 state machine，调用 vision unit 获取 object info，然后传递给 robotic grasping skills。值得注意的 是，robotic unit 和 vision unit 之间 没有 feedback loop。Robot 在 execution 阶段 purely 依赖 wrist 上的 FT sensor 进行 closed-loop control。这种 design 降低了 computation overhead，同时 避免 vision occlusion 导致的 failure。

因为 COMAU Racer 5 是 industrial robot，通常 不支持 joint torque control 或 impedance control，所以 authors 定制了 Cartesian hybrid force-velocity controller (Eq 4)：
$$
\pmb{u \nu}_{\mathrm{EE, des}} = S_{\mathrm{vel}} \pmb{\mathcal{V}}_{\mathrm{EE, des}} + S_{\mathrm{frc}} K_P (\pmb{\mathcal{F}}_{des} - \pmb{\mathcal{F}}) \tag{4a}
$$
$$
= S_{\mathrm{vel}} \pmb{s} \pmb{\mathcal{V}}_{\mathrm{EE, max}} + S_{\mathrm{frc}} K_P (\pmb{\mathcal{F}}_{des} - \pmb{\mathcal{F}}) \tag{4b}
$$
这里 $\pmb{u \nu}_{\mathrm{EE, des}} \in \mathbb{R}^6$ 是 发送给 robot 的 Cartesian velocity command (twist，包含 linear velocity 和 angular velocity)。
$S_{\mathrm{vel}}$ 和 $S_{\mathrm{frc}}$ 是 selection matrices，用于 在 不同 axes 上 激活 velocity control 或 force control。举例来说，在 approach direction (z-axis) 使用 force control 以避免 collision，在 x/y axes 使用 velocity control 进行 search。
$\pmb{s}$ 是 scaling vector，$\pmb{\mathcal{V}}_{\mathrm{EE, max}}$ 是 maximum EE twist。
$K_P$ 是 positive definite proportional control gain matrix，用于 force tracking。
$\pmb{\mathcal{F}}_{des} - \pmb{\mathcal{F}}$ 是 wrench error，即 desired wrench 与 FT sensor 测量值 的 difference。

为了 将 TCP frame 的 wrench 和 twist 转换到 EE frame，使用了 Adjoint map (Eq 5)：
$$
\pmb{\mathcal{V}}_{\mathrm{EE, des}} = \mathbf{Ad}_{\mathrm{EE}}^{\mathsf{T}} T_{\mathrm{TCP}} \pmb{\mathcal{V}}_{\mathrm{TCP, des}} \tag{5a}
$$
$$
\mathcal{F}_{\mathrm{EE, des}} = \mathrm{Ad}_{\mathrm{EE}}^{\mathsf{T}} T_{\mathrm{TCP}} \mathcal{F}_{\mathrm{TCP, des}} \tag{5b}
$$
从 Lie group 角度 理解，$SE(3)$ 的 Lie algebra $se(3)$ 包含 twist $\nu$，而 dual space 包含 wrench $\mathcal{F}$。Adjoint map $\mathrm{Ad}$ 描述了 coordinate frame 变换下 twist 和 wrench 的 transformation rule。这种 mathematical foundation 保证了 force-velocity 控制 在 不同 coordinate frames 下 的 consistency。

### Grasping Strategies 细节与 Intuition

针对 不同 object properties，paper 提出 三种 gripper 及 对应 strategy。

**Strategy A: Tactile Finger Gripper**
hardware 包含 pressure sensor array。通过 sensor 数据 计算 Center of Pressure (CoP)：
$$
^{\mathrm{EE}} p_{\mathrm{CoP}} = [0, 0, ^{\mathrm{EE}} z_{\mathrm{CoP}}]^{\top}
$$
其中 $^{\mathrm{EE}} z_{\mathrm{CoP}} = ^{\mathrm{EE}} z_{\mathrm{TCP}} - ^{\mathrm{TCP}} z_{\mathrm{CoP}}$，代表 EE 到 CoP 沿 z-axis 的 total distance。
**Intuition**: 当 grasping light bulb 时，如果 grip 偏离 CoP，rotation 会 导致 large torque 破坏 glass。通过 动态 计算 CoP 并以 CoP 作为 pivot 进行 rotation，可以 minimize 残余 torque。Algorithm 流程是：approach -> compliant close -> 如果 CoP 位于 fingertip 则 微调 -> 尝试 lift -> 如果 resistance 大则 rotate around CoP。这 模仿了 人类 拧灯泡时 的 intuitive 动作。现代 Robotics RL (如 SAC, PPO) 在 sim-to-real 时 常面临 contact dynamics 不可微 的 问题，而这个 paper 使用 hybrid force-velocity control，直接 将 force feedback 编码进 closed-loop dynamics，提供 stability guarantees。

**Strategy B: Slim-Fingered Gripper**
hardware 是 pneumatic gripper with 3D printed fingers，适合 narrow spaces。
**Intuition**: 因为 vision 误差 大，直接 grasp 极易 miss。Algorithm 采用 "human-like search"：将 gripper 移动到 距离 estimated position $1.2 l_{width}$ 处（$l_{width}$ 是 opening width），然后 slow approach 直到 contact。如果 失败 则 尝试 另一侧。这种 blind search 利用 force feedback 作为 binary trigger，极大 提高 了 在 cluttered environment 中 的 success rate。

**Strategy C: Vacuum Gripper**
hardware 包含 multiple suction cups。适合 large flat objects。
**Intuition**: Vacuum grasp 对 surface 平整度 和 orientation 敏感。如果 orientation 误差 导致 suction cup 无法 seal，grasp 就会 fail。Algorithm 采用 45-degree rotation heuristic。如果 grasp 失败，gripper 绕 center 或 single cup 旋转 45 度重试（最多 7 次）。这是一种 discrete grid search over orientation space，计算 cost 极低，在 industrial setting 中 objects (如 PC Tower cover) 的 edges 通常是 orthogonal 的，所以 45 度 search 能够 快速 align 到 optimal angle。

### Experiments 与 Table II 分析

Table II 比较 了 standard grippers 和 proposed grippers 在 Accurate Vision (AV) 和 Inaccurate Vision (IV) 下 的 performance。虽然 paper 中 的 Table II 存在 OCR garbled problem，但 从 context 可以 推断出 其 structure 和 meaning：
- **Standard 2-Finger**: 在 AV 下 可行，在 IV 下 完全 fail。因为 缺乏 compliance 和 search 机制。
- **Proposed Strategy A, B, C**: 在 IV 下 依然 能 achieve high success rate。

Experiments 在 lab 和 Athens 的 industrial recycling plant 进行。证明 了 tactile methods 能 tolerate 几毫米甚至 几厘米 的 vision error。

### 扩展 Intuition 与 Future Direction

未来 如果 结合 Tactile sensing (如 GelSight) 和 Diffusion Policy，可以将 CoP 的计算 和 45-degree search 替换为 learned latent space exploration。举例 来说，使用 VAE 将 tactile image 压缩 为 latent vector，通过 latent dynamics model 预测 optimal rotation angle。

另外，Eq 4 中 的 $K_P$ 是 固定 的 proportional gain。如果 使用 Variable Impedance Control (VIC)，gain 可以 根据 estimated uncertainty $\sigma_x, \sigma_n$ 动态调整：当 uncertainty 大 时，降低 stiffness 以增加 compliance，避免 damage；当 uncertainty 小 时，增加 stiffness 以提高 precision。

### Web Links Reference

1. 机器人控制理论基础 (Modern Robotics): http://hades.mech.northwestern.edu/index.php/Modern_Robotics
2. Hybrid Force/Position Control 经典 Paper (Khatib 1986): https://ieeexplore.ieee.org/document/1087035
3. HR-Recycler EU Project (实验背景): https://cordis.europa.eu/project/id/820842
4. COMAU Racer 5 工业机器人规格: https://www.comau.com/en/our-competences/robotics-products/racer-family
5. Tactile Sensing in Robotics (GelSight 传感器): https://www.gelsight.com/
