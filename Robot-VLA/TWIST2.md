---
source_pdf: TWIST2.pdf
paper_sha256: 7c627455b7b8d4a86c321002c757150f36a6de5fccda7f4d3a017df18352dd11
processed_at: '2026-08-12T18:43:14-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# TWIST2 用人话说一遍

## 一句话概括

这帮人把 Stanford 之前搞的 TWIST 系统（humanoid 遥操作 + 数据采集）做了一个关键升级：**把昂贵笨重的 MoCap studio 换成一个随身携带的 VR 头显**，而且保留了"全身控制"的能力。听起来简单，背后是一堆工程细节和算法选择。

---

## 要解决的问题：humanoid 的 data 哪儿去了

你看 GPT、看 ALOHA、看 π0、π0.5，这些 system 全都是靠大规模 data 喂出来的。ALOHA 之所以 work，是因为有人花时间做了整套 cheap hardware + teleoperation pipeline，让两个人对坐在桌前就能迅速产几百条 demonstration。

**humanoid 这边没这个 luxury**。

之前能做全身遥操作的就两条路：
1. **MoCap studio**（TWIST 用的方案）：精度高，但需要一整个光学动捕房间，机器没法搬到外面，setup 巨慢，一台机器几十万美元
2. **VR + 简化控制**（AMO、CLONE 用的方案）：便宜、便携，但下半身只能给"往前走 0.5 m/s"这种粗略指令，**没法控制腿干细活**

TWIST2 做的事就是**两边的好处都拿**：便携 + 全身控制 + 单人操作 + 第一人称视野。

---

## 硬件上他们干了什么

### 1. PICO VR 替代 MoCap

PICO 4U VR 头显 + 两个 motion tracker 绑在小腿上，整套大约 1000 美元。头显做 inside-out tracking，tracker 补下肢，100 Hz 输出全身 pose。Setup 大约 1 分钟。

**这跟 HTC Vive Tracker 不一样**——Vive 需要 base station，还要 third-person camera 辅助，BD Atlas demo 就用这套。PICO 全部自包含，更 portable。

代价：PICO 在 elbow、knee 这些没 tracker 的关节上 estimation 精度差，global pose 有时会跳。

### 2. 加一个便宜的 2-DoF 脖子

Unitree G1 出厂是没有真正可用的脖子的。但他们发现做 long-horizon mobile manipulation，**脖子是必需品**，不是奢侈品。

理由很直觉：robot 要弯腰拿地上的篮子，如果有脖子能低头看，上身就保持直立；没脖子就要整个 torso 弯下去，又累又容易不稳。

所以他们 3D 打印了一个 2-DoF（yaw + pitch，故意不做 roll，因为人类日常脖子 roll 用得少）的附加 neck，用两个 Dynamixel motor，成本 250 美元。关键是**不用拆原装的 head LiDAR**，直接 bolt-on。这就让任何 G1 用户都能复现。

### 3. Dex31 hand 当 gripper 用

Unitree Dex31 表面是 dexterous hand，实际只有 3 根手指、DoF 有限，操作起来更像 parallel-jaw gripper。作者直接放弃做五指 retargeting（这是个 open research problem），用 PICO 手柄的按钮控制 $\alpha \in [0, 1]$，在两个预设 pose（open 和 close）之间线性插值。简单粗暴但 reliable。

---

## 算法上几个关键选择

### A. Command vector 怎么设计

这是整篇 paper 最 subtle 的点之一。他们给 low-level controller 的指令是：

$$\mathbf{p}_{\mathrm{cmd}} = [\dot{x}_{\mathrm{ref}}, \dot{y}_{\mathrm{ref}}, z_{\mathrm{ref}}, \phi_{\mathrm{ref}}, \theta_{\mathrm{ref}}, \dot{\psi}_{\mathrm{ref}}, \mathbf{q}_{\mathrm{ref}}]$$

意思是：横向移动用 **velocity**（相对量），但高度 $z$、roll、pitch 用 **absolute**，yaw 角速度也用 velocity。全身关节位置直接给绝对值。

为什么这么混着用？因为 $x, y$ 这种全局位置在 long-horizon 下会累积 drift，relative velocity 让 controller 自己消化误差；$z$ 和 roll/pitch 是局部物理量，IMU 直接量得准，给绝对值精度更高。

更关键的是 $\mathbf{q}_{\mathrm{ref}}$ **直接给全身关节位置**。这就是和 AMO、CLONE 的本质区别——他们只给 root velocity，腿怎么走由 RL locomotion policy 自动决定，所以做不了踢球、跳蹲、legged manipulation。TWIST2 给了全身 joint reference，腿能动起来做精细动作。

### B. GMR Retargeting：上下身分开处理

把 human pose 映射到 robot joint 的公式里，他们做了一个非常聪明的事——**上身只 match rotation，下身 match rotation + position**。

为什么？因为 PICO global pose 有时会突然跳（teleportation、tracking 丢失）。如果上身也跟 position，pose 一跳整个 robot arm 就飞起来。只跟 rotation 的话，global pose 跳不影响 local rotation，手臂保持稳定姿态。

但下身不行，下身要保证 foot placement，foot sliding 会让 walking 完全失败，所以 position 必须跟。这是把 sensor noise 的特性考虑到 optimization 里的典型操作。

### C. Low-level RL controller：one-stage training

之前 TWIST 用复杂的 teacher-student pipeline，TWIST2 直接 one-stage PPO 训练。他们用 20k motion clips 训练，其中只有 73 条来自 PICO 自己采集。

73 条听起来少，但这是关键——这 73 条覆盖了 daily movement 的 manifold（walk、crouch、reach），policy 学到之后就能 generalize 到 sim-to-real。**少但 in-distribution 比 多但 out-of-distribution 重要**。

Reward 用 $r = e^{-\alpha\|\mathbf{p}_{\mathrm{cmd}} - \mathbf{p}_{\mathrm{cur}}\|}$ 这种 exponential kernel，会 saturate，agent 不会为了零误差抖 motor。

### D. 上层 visuomotor policy 用 Diffusion Policy

数据采集回来后训一个 autonomous policy。Observation 是 ZED Mini 给的 224×224 RGB + 历史 command $\mathbf{p}_{\mathrm{cmd}}$。Action 也是 $\mathbf{p}_{\mathrm{cmd}}$，输出 64-step chunk。

**这里有个反直觉的选择**：proprioception 用历史 command 而不是 raw robot state $\mathbf{s}$。

理由有二：
1. 让 high-level 和 low-level 解耦，high-level 不需要知道 robot joint encoder noise
2. 避免 overfit 到 noisy state——$\mathbf{p}_{\mathrm{cmd}}$ 相当于"intended action"，更 stable

外加 10% Gaussian noise 注入到 proprioception，逼 policy 多依赖 vision。

---

## 效果数字

**遥操作效率**（Table II）：
- 双臂操作：18.5 分钟收 98 条 demo，100% success，平均 11 秒一条
- 移动操作：19.5 分钟收 46 条，25 秒一条

这个 throughput 在 humanoid 领域是惊艳的，已经接近 ALOHA 在桌面双臂上的效率。

**消融实验**（Table III）：
- 去掉脖子：novice 成功率从 83% 跌到 41%，时间翻倍 → **脖子是 essential**
- 去掉 stereo：成功率不变，时间增加 45% → depth 主要影响 grasp 精度
- 去掉 egocentric view：expert 反而最快（站在 robot 旁边），但没法做 mobile task

**自主 policy**：
- WB-Dex（弯腰抓 cup 放进 box）：170 条 demo，抓取精度受限
- Kick-T（踢 T 型盒子到目标）：50 条 demo，6/7 成功率

---

## 我的直觉判断

这篇 paper 的价值不在算法创新，每个 piece 都不新——VR 遥操作、GMR retargeting、PPO motion tracker、Diffusion Policy——但**组合在一起第一次让 humanoid data collection 达到 ALOHA 级别的 efficiency 和 usability**。

这跟 ImageNet 之于 vision、ALOHA 之于 bimanual 是同一种意义上的 contribution：infrastructure breakthrough。

关键的是**他们 open-source 所有东西**，包括 3D 打印文件、MuJoCo XML、retargeting pipeline、RL controller、data、policy。任何有 Unitree G1 的 lab 都可以复现。这才是 scaling 真正能开始的前提。

**局限也很诚实**：
1. PICO 精度不如 MoCap，特别是没 tracker 的关节
2. 高度 dynamic motion（sprint）track 不好
3. visuomotor policy 还是在固定场景测，generalization 没验

要 scale humanoid data，下一步要么是 cross-embodiment（让 data 在不同 humanoid 间复用），要么是大规模多人采集（这需要 TWIST2 这种低成本 system 普及）。

简单说：**他们在 humanoid 上把 ALOHA 那一套复刻了一遍，可复现、可移植、便宜**。这就够了。

---

# TWIST2 深度解析：Portable Humanoid Teleoperation 与 Whole-Body Visuomotor Policy

## 一、 Motivation 与 Design Philosophy

这篇 paper 解决 humanoid robotics 长期存在的 data bottleneck 问题。观察 bimanual manipulation 领域，ALOHA / GELLO / π0 / π0.5 已经形成了 scalable data pipeline，但 humanoid 这边还在依赖 MoCap studio。作者团队（Stanford + Berkeley + CMU + USC + Amazon FAR 的 dream team，Karen Liu、Pieter Abbeel、Angjoo Kanazawa、Jiajun Wu 等共同 advising）在 TWIST [1] 之后做了关键升级——把 MoCap 替换成 VR-based 系统，同时保留 full whole-body control 这个核心特性。

Table I 中作者把现有 humanoid teleop 系统分为三类：decoupled (HOMIE)、partial (AMO, CLONE)、full (TWIST)。TWIST2 是第一个同时满足 portable + full whole-body control + single operator + egocentric teleop + foot control + wrist control 的系统。这个组合并不是 trivial 的工程拼装，每个组合都对应一个深层次的设计 trade-off。

参考链接：
- TWIST (predecessor): https://arxiv.org/abs/2505.02833
- AMO: https://arxiv.org/abs/2503.14301
- CLONE: https://arxiv.org/abs/2506.08931
- HOMIE: https://arxiv.org/abs/2502.13013
- HumanPlus: https://humanoid-robot.github.io/
- OmniH2O: https://arxiv.org/abs/2406.08858

---

## 二、 System Architecture 整体架构

参考 Fig.2，TWIST2 是一个 hierarchical two-level control 系统：

**Low-level controller** $\pi_{\mathrm{low}}$：task-agnostic 的 whole-body motion tracker，RL 训练，50Hz 输出 PD target。

**High-level controller** $\pi_{\mathrm{high}}$：两个变体——teleop policy（human + retargeter）与 visuomotor policy（Diffusion Policy），输出 command vector $\mathbf{p}_{\mathrm{cmd}}$。

数据流：PICO 4U (100Hz human pose) → GMR retargeting → $\mathbf{p}_{\mathrm{cmd}}$ → Redis → $\pi_{\mathrm{low}}$ (50Hz) → PD controller → motor torques。同时 ZED Mini stereo → PICO display (h265 via GStreamer) + JPEG via ZMQ 到 data collection process。

### 2.1 Command Interface 设计 intuition

公式 (1) 定义了 command vector $\mathbf{p}_{\mathrm{cmd}} = [\dot{x}_{\mathrm{ref}}, \dot{y}_{\mathrm{ref}}, z_{\mathrm{ref}}, \phi_{\mathrm{ref}}, \theta_{\mathrm{ref}}, \dot{\psi}_{\mathrm{ref}}, \mathbf{q}_{\mathrm{ref}}]$。

- $\dot{x}_{\mathrm{ref}}, \dot{y}_{\mathrm{ref}}$：root 在 world frame 中 x、y 轴的 translational velocity，相对量，意味着不需要全局定位
- $z_{\mathrm{ref}}$：root 的绝对高度（用于蹲下、踮脚等动作）
- $\phi_{\mathrm{ref}}, \theta_{\mathrm{ref}}$：root 的 roll、pitch 绝对角度（保持平衡用）
- $\dot{\psi}_{\mathrm{ref}}$：root 的 yaw angular velocity，相对量
- $\mathbf{q}_{\mathrm{ref}}$：whole-body joint positions

**这里有一个非常 subtle 的设计直觉**：root 的 translations 和 yaw 用 relative（velocity 形式），而 $z_{\mathrm{ref}}$ 和 roll/pitch 用 absolute。为什么？因为 $z$ 和 roll/pitch 是局部物理量，IMU 可以直接测量，没有 drift；而 $x, y, \psi$ 在 long-horizon 下会因为 state estimation drift 累积误差，用 velocity command 可以让 low-level controller 自己消化掉这些误差。这一思路借鉴了 legged robot 领域的指令接口设计，比如 ANYmal、Cassie 等 quadruped / biped 控制器。

而且 $\mathbf{q}_{\mathrm{ref}}$ 直接包含全身关节位置，这意味着 legged manipulation、dancing、踢球这种需要精确下肢控制的任务都可以表达。AMO/CLONE 把下半身简化成 root velocity command，这就丢失了大量 whole-body dexterity 信息。

### 2.2 State observation

公式 (2) $\mathbf{s} = [\boldsymbol{\omega}, \dot{\boldsymbol{\omega}}, \mathbf{q}, \dot{\mathbf{q}}]$。

- $\boldsymbol{\omega}$：IMU 给出的 root orientation（通常用 quaternion 或 6D 表示）
- $\dot{\boldsymbol{\omega}}$：IMU 给出的 angular velocity
- $\mathbf{q}$：所有 joint encoder 读数
- $\dot{\mathbf{q}}$：joint velocity（数值差分或 motor 自带）

注意这里没有 root linear velocity——这是 legged robot RL 中常见的 domain randomization 训练目标：让 policy 不依赖 noisy 的 linear velocity 估计。训练时用 privileged information（包括 linear velocity），部署时用 history encoder 推断。

### 2.3 PD 控制

公式 (4) $\boldsymbol{\tau} = K_P(\mathbf{q}_{\mathrm{tgt}} - \mathbf{q}) - K_D \dot{\mathbf{q}}$ 是经典的 PD 控制，$K_P$ 是 position gain（决定刚度），$K_D$ 是 damping gain。这里 $\mathbf{q}_{\mathrm{tgt}}$ 由 $\pi_{\mathrm{low}}$ 给出，相当于 RL policy 在 cartesian-like task space 学一个 kinematic mapping，然后由 PD 在 joint space 做 tracking。这种 decomposition 让 RL 不需要直接学 torque，sample efficiency 高很多。

参考 ETH 的 legged robot 控制综述：
- Learning-based Control: https://arxiv.org/abs/2104.09267

---

## 三、 TWIST2 Neck：低成本 Active Vision

### 3.1 Hardware

Fig.3 中的 TWIST2 Neck 是 2-DoF 设计，yaw + pitch，用两个 Dynamixel XC330-T288 motor，U2D2 接口，从 G1 onboard 12V/5A 取电。所有结构件 3D 打印，总成本 $250（不含 ZED Mini $400）。Fig.15 显示他们已经制造了 3 个 unit，证明 reproducibility。

**关键 intuition**：为什么不加 roll？因为 human daily activity 中 neck roll 使用频率很低（图5 显示 yaw + pitch 已经覆盖 80% 人类颈部功能），且 2-DoF 让 control 简单、可靠、便宜。这跟 ToddlerBot [24] 的设计思路一脉相承。

### 3.2 Neck Retargeting 数学

公式 (8)：$R_{\mathrm{rel}} = R_{\mathrm{spine}}^\top R_{\mathrm{head}}$。

- $R_{\mathrm{head}}, R_{\mathrm{spine}} \in SO(3)$：human head 和 spine 在 world frame 的全局旋转矩阵
- $R_{\mathrm{rel}}$：head 相对 spine 的相对旋转，这就去掉了 human torso rotation 的影响，只保留 neck joint 贡献

公式 (9)：
- $q_{\mathrm{neck}}^{\mathrm{yaw}} = \psi = \arctan2(r_{21}, r_{11})$：从旋转矩阵 $R_{\mathrm{rel}} = [r_{ij}]$ 提取 yaw
- $q_{\mathrm{neck}}^{\mathrm{pitch}} = \theta = \arcsin(-r_{31})$：提取 pitch

这是经典的 ZYX Euler angle 分解，先 yaw（绕 z）再 pitch（绕 y），跳过 roll。$\arctan2(r_{21}, r_{11})$ 处理 gimbal lock 边界情况。这种提取方式在 robot locomotion 控制中非常常见，比如 MIT Cheetah 的躯干姿态估计。

参考：
- ToddlerBot: https://arxiv.org/abs/2502.00893
- ZYX Euler conventions: https://en.wikipedia.org/wiki/Euler_angles

---

## 四、 PICO 4U 替代 MoCap

PICO 4U VR 头显 + 2 个 PICO Motion Tracker 绑在 calf 上，整套 ~$1000，比 optical MoCap 便宜 1-2 个数量级。PICO 通过 inside-out tracking 给出全身 pose 估计，不需要 external camera，所以 portable。100Hz 输出，setup 只需 1 分钟。

跟 HTC Vive Tracker 对比，PICO 不需要 base station，也不需要 third-person view camera，更灵活。Boston Dynamics 最近的 Atlas demo [28] 用的就是 Vive Tracker 方案。

**关键 limitation**：PICO 在 elbow、knee 这种没有 tracker 的关节上 estimation 精度低。这也是为什么作者在 Section V Limitations 里提到这个问题。

参考：
- PICO 4 Ultra: https://www.picoxr.com/global/products/pico4-ultra
- PICO Motion Tracker: https://www.picoxr.com/global/products/pico-motion-tracker
- XRoboToolkit: https://arxiv.org/abs/2508.00097
- Boston Dynamics Atlas blog: https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/

---

## 五、 Holistic Human-to-Humanoid Retargeting (GMR)

### 5.1 Body Retargeting 详解

公式 (6) 是 GMR [1, 29] 的修改版：

$$\mathbf{q}^* = \arg\min_{\mathbf{q}} \sum_{i \in \mathcal{L}_{\mathrm{up}} \cup \mathcal{L}_{\mathrm{low}}} w_i^R \|R_i^{\mathrm{human}} - R_i^{\mathrm{robot}}(\mathbf{q})\|_F^2 + \lambda_{\mathrm{pos}} \sum_{k \in \mathcal{P}_{\mathrm{low}}} w_k^p \|p_k^{\mathrm{human,pelvis}} - p_k^{\mathrm{robot}}(\mathbf{q})\|_2^2$$

- $\mathcal{L}_{\mathrm{up}}, \mathcal{L}_{\mathrm{low}}$：upper-body 和 lower-body link 集合
- $\mathcal{P}_{\mathrm{low}}$：选定的 lower-body 关键点（feet/ankles/pelvis）
- $R_i^{\mathrm{human}}, R_i^{\mathrm{robot}}(\mathbf{q})$：link $i$ 在 human 和 robot 上的 global orientation
- $p_k^{\mathrm{human,pelvis}}$：human keypoint $k$ 在 **pelvis frame** 中的位置（**核心 trick**：把所有 position 转到 pelvis frame，消除 global pose 误差）
- $p_k^{\mathrm{robot}}(\mathbf{q})$：robot 上对应 keypoint 位置（forward kinematics 算出）
- $w_i^R$：每个 link 的 rotation weight
- $w_k^p$：每个 keypoint 的 position weight
- $\lambda_{\mathrm{pos}}$：rotation vs position 的 trade-off
- $\|\cdot\|_F$：Frobenius norm，对 rotation matrix 之间的距离度量
- $\|\cdot\|_2$：L2 norm，对 position 距离

**这个公式背后有一个核心 insight**：upper body **只优化 rotation**，不优化 position；lower body **同时优化 rotation 和 position**。为什么？因为 PICO 的 global pose estimation 有跳跃（teleportation 时，或 tracking drift 时），如果 upper body 也优化 position，global pose 跳变会导致 robot arm 飞起来。让 upper body 只跟 rotation，teleport 时 robot 手仍然保持自然姿态，只是 torso 不动。Lower body 因为涉及 foot placement，必须做 position tracking，否则 foot sliding 严重。

### 5.2 Hand Retargeting

公式 (7)：$\mathbf{q}_{\mathrm{hand}} = (1-\alpha)\mathbf{q}_{\mathrm{open}} + \alpha \mathbf{q}_{\mathrm{close}}$

- $\alpha \in [0, 1]$：scalar grasp command
- $\mathbf{q}_{\mathrm{open}}, \mathbf{q}_{\mathrm{close}}$：两个 canonical configurations

这里有一个 pragmatic 的工程判断：Unitree Dex31 虽然号称 dexterous hand，实际只有 3 根手指，DoF 有限，更接近 parallel-jaw gripper。所以作者直接绕开 5-finger pose estimation，用 PICO controller 上的 button 直接控制 $\alpha$，更 reliable。这也避开了 dexterous hand retargeting 这个还没解决好的 research problem。

针对 power grasp 和 fine-grained pinch 用两套不同的 $\mathbf{q}_{\mathrm{open}}, \mathbf{q}_{\mathrm{close}}$，对应不同 task 类型。

参考：
- GMR GitHub: https://github.com/YanjieZe/GMR
- GMR paper: https://arxiv.org/abs/2510.02252
- Dex31 (Unitree): https://www.unitree.com/

---

## 六、 Low-Level Motion Tracker RL 训练

### 6.1 训练数据

20k motion clips：
- 7k 通过 GMR retarget
- 13k 来自 TWIST original dataset（AMASS + OMOMO + in-house MoCap）
- 73 个 PICO collected motions（关键！用于 domain gap bridging）

**关键 intuition**：73 个 motion 听起来少，但这正反映了作者对 sim-to-real 的理解——只要覆盖 daily movement 的 manifold（walking、crouching、reaching、grasping），policy 就能 generalize。这种 "small but in-distribution" 的 data 对 sim-to-real 极其重要。

### 6.2 Reward 设计

公式 (10)：$r_{\mathrm{track}} = e^{-\alpha\|\mathbf{p}_{\mathrm{cmd}} - \mathbf{p}_{\mathrm{cur}}\|}$

- $\mathbf{p}_{\mathrm{cur}}$：当前实际状态（root velocity + root pose + joint positions）
- $\alpha$：sharpness parameter，控制 reward 的指数衰减速度。$\alpha$ 越大，对 tracking error 越敏感

Total reward $r = r_{\mathrm{track}} + r_{\mathrm{reg}}$，$r_{\mathrm{reg}}$ 包括 action change penalty（smoothness）、energy 等。这种 exponential kernel reward 在 humanoid imitation 中很流行，相比 L2 reward，它 saturates，不会让 agent 过度追求零误差而 actuation jitter。

### 6.3 Network Architecture

Actor 是 convolutional history encoder + MLP backbone。这跟 TWIST、OmniH2O 等 work 一样，把 history proprioception 和 history reference command 压缩到 latent vector，类似 transformer 的 token embedding 但用 1D conv。PPO 训练。

**为什么不用 teacher-student？** TWIST 用了复杂的 teacher-student pipeline，TWIST2 直接 one-stage。这是工程上的简化——经过 TWIST 的实验，作者发现 one-stage 训练只要 data 和 reward 设计得当，效果不差，且部署不需要 RNN hidden state 管理，方便很多。

参考：
- PPO: https://arxiv.org/abs/1707.06347
- Exponential kernel reward (similar to DeepMimic): https://arxiv.org/abs/1804.05621

---

## 七、 Visuomotor Policy Learning（Diffusion Policy）

### 7.1 Observation & Action Space

**Observation**：
- Visual：360×640 RGB → 224×224 downsample（ZED Mini stereo，左目或右目？paper 没明说，从 Fig.7 看是单目或同时用两个）
- Proprioception：**历史 command $\mathbf{p}_{\mathrm{cmd}}$ 而不是 raw robot state $\mathbf{s}$**

这里有个很重要的 design choice。用 $\mathbf{p}_{\mathrm{cmd}}$ 作为 proprioception input 有两个理由：
1. **Modularity**：high-level policy 只需要输出 $\mathbf{p}_{\mathrm{cmd}}$，不依赖 $\mathbf{s}$ 的具体含义，可以独立训练和 deploy
2. **Error accumulation mitigation**：raw robot state $\mathbf{s}$ 是 noisy 的（IMU drift、encoder noise），如果 high-level 直接见 $\mathbf{s}$，可能会 overfit 到 noise 上；用 $\mathbf{p}_{\mathrm{cmd}}$ 等于见 "intended action"，更 stable

**Action**：同样是 $\mathbf{p}_{\mathrm{cmd}}$，与 teleop 一致。

### 7.2 Network Architecture

- **Diffusion Policy** [35] backbone
- 1D convolutional blocks 做 temporal modeling
- **64 action chunks**，sample-based prediction，对应 ~2 秒（64 / 30Hz ≈ 2.13s）
- Visual encoder：ResNet-18 + R3M [37] pretrained

**Diffusion Policy intuition**：相比 VAE 或 autoregressive，diffusion 在 multi-modal action distribution 上有天然优势。humanoid action 是高度 multi-modal 的——同一个视觉观测下，可能有多种 reasonable motion。64-step chunk + receding horizon control 让 policy 有 temporal consistency，避免 jitter。

R3M pretrained visual encoder 是从 Ego4D 等数据集学的 representation，迁移到 robot task上，相当于给 policy 一个 "human activity understanding" 的 prior。这比 from-scratch 训练 visual encoder sample efficiency 高很多。

### 7.3 Augmentation

- Proprioception noise：10% Gaussian，强制 policy 依赖 vision 而不是 overfit 到 state
- Visual augmentation：random crop, rotation, color jitter

这个 10% noise 是个 simple but effective trick，类似 domain randomization 思想，让 policy 不依赖 perfect state estimation。

### 7.4 Deployment

- ONNX 转换
- RTX 4090 上 20Hz inference
- 预测 64 步，执行 48 步（receding horizon，每 48 步重新预测）
- 30Hz 执行（与 data collection 频率一致）

**为什么 64 预测 48 执行？** 经典 MPC receding horizon 思想。预测多步保证 temporal consistency，少执行几步避免 stale action。48 步执行约 1.6s，比 prediction horizon 短，正好让 policy 在 chunk 末段重新感知环境。

参考：
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- R3M: https://arxiv.org/abs/2203.12601
- 3D Diffusion Policy (DP3): https://arxiv.org/abs/2403.03954
- Action chunking (ACT): https://arxiv.org/abs/2304.13705

---

## 八、 Experimental Results 深度解读

### 8.1 Long-Horizon Teleoperation (Fig.10)

两个标志性任务：
1. **Folding towels**：locate → move to front → grasp → shake to spread → pinch corner → fold in half → repeat → fold thirds/quarters → press crease → place。连续 fold 3 个 towel，bottleneck 是 motor overheating，不是 system 本身。
2. **Transporting baskets through door**：adjust position via foot placement → bend down → pick left/right basket → approach door → push door with arm → walk through → place on shelf。

这两个 task 都需要：
- Long-horizon planning（>30s 连续操作）
- Fine-grained wrist control（pinch towel corner）
- Active vision（找 basket、看门、看 shelf）
- Whole-body coordination（bend down + reach + walk）

**为什么 egocentric active vision 是关键？** 作者在 Fig.12 中给了一个直觉性的图示：human 无法 grasp 超出 FOV 的 object，这意味着如果 camera 固定在 head 上（无 neck），mobile task 中 robot 看不到地面物体就要弯腰，非常 inefficient。Neck 让 robot 可以低头看物体同时保持上身直立。

### 8.2 Data Collection Efficiency (Table II, III)

Table II：Bimanual Manip 1 用 18.5 min 收集 98 episodes，100% success，11s/episode；Mobile Manip 用 19.5 min 收集 46 episodes，25s/episode。这种 throughput 是惊人的——ALOHA 在 bimanual 任务上大概是这个数量级，但 humanoid 上能做到这个效率是 TWIST2 的重要贡献。

Table III：User study 对比 4 个 ablation：

| Config | Novice Success | Expert Success | Avg Time (s) |
|---|---|---|---|
| TWIST2 (full) | 10/12 | 10/11 | 67.8 |
| w/o Stereo | 10/12 | 10/15 | 98.3 |
| w/o Neck | 7/17 | 9/12 | 112.3 |
| w/o Egocentric View | 10/13 | 10/10 | 68.7 |

**关键 finding**：
- **w/o Neck 是 worst**：novice success rate 从 83% 跌到 41%，time cost 翻倍。这证明 neck 对 long-horizon task 是 essential，不是 nice-to-have
- **w/o Stereo**：success rate 不变（因为 task 不强依赖 depth），但 time cost 增加 45%。Depth perception 主要影响 grasp precision
- **w/o Egocentric View**：novice 几乎没影响，但 expert 极快（43s for 10 demos）——因为 expert 站在 robot 旁边，可以同时看 robot 和 VR pass-through。但 mobile task 不可能这样操作

### 8.3 Autonomous Policy Results (Fig.11, 14)

**WB-Dex**（Whole-Body Dexterous pick & place）：
- 170 demonstrations 训练
- Cup 太轻，grasp 失败率高（Fig.14 显示大部分 failure 是 grasp slip）
- Policy 能 reliable reach，说明 visuomotor policy 学到了 visual servoing 能力

**Kick-T**（Kick T-shape box to target）：
- 50 demonstrations 训练
- 6/7 success rate
- 数据中 action pattern consistent：left foot kick → right foot step to maintain balance
- 这是 legged manipulation 的 showcase，证明 whole-body control 的价值

**为什么 170 demos 还不够？** Cup 太轻 + Dex31 finger 精度有限，这种 fine-grained task需要更多 data 或更好的 hand。作者在 limitations 里没明说，但 50 demos for Kick-T 已经够，说明 dynamic task 反而比 static fine-grained task 容易 learn，因为 dynamic task 的 action manifold 更 constrained。

参考：
- ALOHA 2: https://arxiv.org/abs/2405.02292
- ACT (Action Chunking Transformer): https://arxiv.org/abs/2304.13705

---

## 九、 Discussion on Scaling Humanoid Data

作者在 Section VI 提了三个 scaling 关键点：

1. **Hardware standardization**：Unitree G1 现在是事实标准，作者提倡先标准化再 scale。G1 提供 80% core human functionality with neck add-on
2. **Democratization**：MoCap-free, low-cost, portable
3. **Data sharing**：开放 dataset on HuggingFace，https://twistdata.github.io

**这里延伸一个重要联想**：humanoid data scaling 跟 LLM data scaling 有本质区别。LLM 的 data 是 text，format 统一；humanoid data 是 proprioception + vision + action，每个 robot embodiment 的 action space 不同。要 scale humanoid data，需要：
- Cross-embodiment policy（如 RT-X, π0.5）
- Action tokenization / discretization（让 data 可移植）
- Standardized observation format

参考：
- RT-X: https://robotics-transformer-x.github.io/
- π0.5 cross-embodiment: https://arxiv.org/abs/2504.16054
- Open X-Embodiment: https://arxiv.org/abs/2310.08864

---

## 十、 跟相关工作的关系网络

### 10.1 TWIST 谱系

TWIST2 是 TWIST 的直接 successor。TWIST 用 MoCap，TWIST2 用 VR。两者共享 GMR retargeting、PPO motion tracker、whole-body control philosophy。TWIST2 的核心 novelty 是把 cost 从 ~$50k MoCap 降到 ~$1.5k，同时让 single operator 成为可能。

### 10.2 跟 AMO/CLONE 的关系

AMO、CLONE 都用 VR 但只做 partial whole-body control。他们的 command interface 是 root velocity（root linear + angular velocity），下肢由 RL locomotion policy 自动生成。这适合 walking + reaching 这种简单 task，但做不了 dancing、kicking、legged manipulation。

TWIST2 通过保留 $\mathbf{q}_{\mathrm{ref}}$ 在 command vector 中，让 teleop 直接控制下肢 joint position，unlocks 这类 task。

### 10.3 跟 OmniH2O、HumanPlus 的关系

OmniH2O [14] 也用 VR 做 whole-body teleop，但用 Quest + 1 个 tracker，全身 pose estimation 精度不如 PICO 2-tracker 模式。HumanPlus [12] 用 RGB camera + pose estimation（如 SLAM 模型），no wearable，但 latency 高、occlusion 敏感。

### 10.4 跟 Bimanual Manipulation 的对比

ALOHA、GELLO 是 bimanual arm teleop 标杆。TWIST2 在 humanoid 上的 throughput 跟 ALOHA 相当（11s/episode vs ALOHA 在 bimanual task 上 10-20s），但 humanoid多了 whole-body coordination，复杂度更高。这说明 VR-based teleop 的 efficiency 可以逼近机械臂级别的 teleop。

### 10.5 跟 VideoMimic、HEAD 等 Visual Humanoid Control 对比

VideoMimic [18] 做 real2sim2real，让 robot 学 sitting 这种 static interaction。HEAD [20] 做 keypoint-based hierarchical nav。TWIST2 是第一个用 egocentric vision 做 whole-body loco-manipulation 的 visuomotor policy。

PDC [21] 在 sim 里做 visual RL，sim-to-real gap 大。TWIST2 通过 imitation learning 绕过 sim-to-real visual gap，直接从 human demo 学。

### 10.6 跟 Diffusion Policy 生态

DP3 [36] 是 3D Diffusion Policy，作者 Yanjie Ze 也是 DP3 第一作者。TWIST2 的 visuomotor policy 用 2D ResNet-18 + R3M，没用 3D point cloud。这可能是因为 humanoid 的 observation 主要是单目 ego view，3D 益处不大。但作者后续可能会用 DP3 思路升级。

### 10.7 跟 Large Behavior Models 的关系

Boston Dynamics 最近的 Atlas Large Behavior Models [28] 跟 TWIST2 思路类似——用 VR teleop 收集 data，然后 train policy。但 BD 用 Vive Tracker + 第三方相机，TWIST2 用 PICO inside-out，更 portable。这是 humanoid data scaling 的 industry trend。

参考：
- BeyondMimic: https://arxiv.org/abs/2508.08241
- Vision in Action (VIP): https://arxiv.org/abs/2506.15666
- BeamDojo: https://arxiv.org/abs/2411.14386

---

## 十一、 Critique 和 Open Questions

虽然 paper 写得很 solid，但有几个可以深挖的方向：

1. **PICO global pose drift**：作者承认 PICO 在 elbow/knee 上 estimation 精度低。如果 PICO 未来支持更多 tracker（手腕、上臂、大腿），可以显著提升 retargeting 精度。作者提到 "2-tracker mode provides more stable pose estimation"，但这是 stability-accuracy trade-off

2. **Dynamic motion 限制**：sprinting 等 highly dynamic motion track 不好。这跟 reward shaping 和 data distribution 有关。如果加入 more aggressive motion data + curriculum learning，可能突破

3. **Generalization**：visuomotor policy 只在固定场景（如 cup、T-box）上 demo。未见物体、未见环境的 generalization 没测。这需要更大规模 data + VLA-style architecture

4. **Action chunk 长度 64-step 是不是最优**？作者没做 ablation。ACT、π0 都对 chunk length 敏感，64 步可能针对 humanoid long-horizon task 太短

5. **Cross-embodiment 迁移**：TWIST2 完全绑定 Unitree G1。如果要在 Booster T1、Figure 02 上 deploy，需要重新 train low-level controller。Retargeting 公式部分通用，但 RL policy 需要重训

6. **Stereo utilization**：paper 没明说 visuomotor policy 用的是 stereo 还是 mono。如果是 mono，stereo 的 depth 信息没充分利用。可以用 stereo depth 作为额外 observation

---

## 十二、 Intuition Summary

我把 TWIST2 的核心 intuition 总结为三个层次：

**Layer 1: Engineering**
- VR + 2 trackers 替代 MoCap → cost 降 30x，portability 提升
- Neck add-on 不拆原 head → backward compatible
- Hand 退化成 gripper → 避开 dexterous hand retargeting 难题
- Single operator + joystick control center → workflow 关键

**Layer 2: Algorithmic**
- Command interface 用 relative + absolute 混合 → 平衡 long-horizon stability 和 control precision
- GMR retargeting 分上下身 → upper body immune to global pose drift, lower body 保证 foot placement
- Low-level RL 用 exponential kernel reward + history encoder → sample efficient + 通用
- Diffusion Policy + 64 action chunk → multi-modal action + temporal consistency
- 用 $\mathbf{p}_{\mathrm{cmd}}$ 作为 proprioception input → decoupling + noise robustness

**Layer 3: Philosophical**
- Full whole-body control > partial control：unlocks legged manipulation, dancing, kicking
- Egocentric active vision > third-person or fixed camera：mobile manipulation 的必须条件
- Imitation from human demo > pure RL：绕过 reward engineering 和 sim-to-real gap
- Portable + reproducible > lab-only MoCap：scaling 的前提条件
- Hardware standardization >百花齐放：先 converge 到 G1，再 scale

TWIST2 的真正贡献在于把 "full whole-body teleop + egocentric vision + single operator + portable hardware + visuomotor policy" 这五个 pieces 整合成一个 end-to-end reproducible pipeline，每个 piece 都不新，但组合在一起第一次让 humanoid data collection 达到了 ALOHA 级别的 efficiency 和 usability。这跟当年 ImageNet 之于 vision、ALOHA 之于 bimanual 的意义类似——不是算法突破，是 infrastructure breakthrough。

参考链接汇总：
- TWIST2 project: https://yanjieze.com/TWIST2
- Dataset: https://twistdata.github.io
- TWIST (predecessor): https://arxiv.org/abs/2505.02833
- GMR: https://github.com/YanjieZe/GMR
- XRoboToolkit: https://arxiv.org/abs/2508.00097
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- DP3: https://arxiv.org/abs/2403.03954
- R3M: https://arxiv.org/abs/2203.12601
- ALOHA: https://arxiv.org/abs/2304.13705
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- Boston Dynamics Atlas LBM: https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/
- OmniH2O: https://arxiv.org/abs/2406.08858
- HumanPlus: https://humanoid-robot.github.io/
- AMO: https://arxiv.org/abs/2503.14301
- CLONE: https://arxiv.org/abs/2506.08931
- HOMIE: https://arxiv.org/abs/2502.13013
- MobileTV: https://arxiv.org/abs/2412.07773
- Open Television: https://arxiv.org/abs/2407.01512
- ToddlerBot: https://arxiv.org/abs/2502.00893
- AMASS: https://amass.is.tue.mpg.de/
- OMOMO: https://arxiv.org/abs/2305.02775
- PPO: https://arxiv.org/abs/1707.06347
- DeepMimic: https://arxiv.org/abs/1804.05621
- RT-X: https://robotics-transformer-x.github.io/

如果你想深入某个具体方向（比如 GMR retargeting 的数值稳定性、Diffusion Policy 在 humanoid 上的 chunk length 选择、或 PICO pose estimation 的误差模型），我可以继续展开。
