---
source_pdf: NaVILA Legged Robot Vision-Language-Action Model for Navigation.pdf
paper_sha256: e02fcccae8aad56cf33234882de06b97256b76aad86a011171b83db40790bc8c
processed_at: '2026-08-05T22:00:25-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej, 用最直白的人话讲，这篇 paper 核心就是一件事：**把大模型的“嘴”和机器人的“腿”解耦。**

以前的 End-to-End VLA (Vision-Language-Action model) 搞法很贪心，恨不得把人类指令和摄像头画面塞进大模型，然后让大模型直接吐出 12 个关节的电机扭矩。这非常反直觉。因为大语言模型天生是在文本分布上预训练的，你逼它做微积分级别的物理连续控制，它的 reasoning 能力被严重稀释，而且大模型推一次要一秒钟，但机器人腿足控制需要 50 帧每秒，频率完全错配。

所以 NaVILA 的 intuition 极度符合生物学常识：**构建“大脑”与“小脑”的双层架构。**

**大脑** 慢速运转，大概 1 Hz。它看 RGB 图像，听人话，但它不直接算关节扭矩，而是输出一个“人类也能听懂的自然语言中级指令”，比如 "move forward 75cm"。
**小脑** 快速运转，大概 50 Hz。它拿这个中级指令，结合自己的 LiDAR 感知，算出 12 个关节该怎么动，同时负责避障和防摔。

这个 decoupling 一做，整个系统就丝滑了。大模型只用管它最擅长的语言推理，不用过拟合到特定机器人的关节上；Low-level 控制只管在物理世界里打怪升级。所以你换机器人的时候，Unitree Go2 换成 Booster T1 人形机器人，大脑完全不用重训，只换个“小脑”就行。

为了 build your intuition，我下面把两层的技术细节彻底拆开。

### 1. 大脑：怎么调教 VLM 做空间推理？

大脑用的是 NVIDIA 的 VILA (一个 image-based VLM)。他们拒绝用 Video VLM，因为高质量 video-text 预训练数据太少。VILA 因为在预训练时用了图文交错语料，对多图推理很强。

**Prompt 设计的直觉：**
在 VLN (Vision-Language Navigation) 里，当前帧 和 历史帧 的语义完全不同。当前帧是“我眼前有堵墙”，历史帧是“我从厨房走出来的”。NaVILA 在 prompt 里用文本明确区分：把最新帧单独拎出来，历史帧做均匀采样且强制包含第一帧（起点）。这样 LLM 就有了 episodic memory。

**从 YouTube 旅游 Vlog 偷数据：**
这是 paper 最天马行空的一笔。直接拿 2K 个人类第一人称旅游视频，切成 20K 条轨迹。但是视频没有 action label 怎么办？他们用 MASt3R (一个 3D 重建模型) 去估计相机的 6-DoF 位姿。
公式直觉：假设从图像 $I_t$ 到 $I_{t+1}$，MASt3R 估计出平移向量 $T \in \mathbb{R}^3$ 和旋转矩阵 $R \in SO(3)$。
我们只提取地面平面的平移分量 $T_{xy}$ 算距离：$d = \|T_{xy}\|_2$。
提取 yaw 轴旋转角：$\theta = \arctan2(R_{21}, R_{11})$。
这样，每一帧人类步伐，就变成了 "forward $d$ meters, turn $\theta$ degrees" 这种纯语言 action label。把大模型推回它最舒服的自然语言分布里。

### 2. 小脑：怎么训练腿足 RL Policy？

Low-level action space 定义为期望关节位置 $a_t = q^d_t \in \mathbb{R}^{12}$。下标 $t$ 是时间步，上标 $d$ 代表 desired（期望值）。然后通过 PD controller 转成扭矩：
$$\tau = K_p (q^d - q) + K_d (\dot{q}^d - \dot{q})$$
其中 $K_p$ 是 stiffness（刚度），$K_d$ 是 dampness（阻尼），$q$ 和 $\dot{q}$ 是当前关节的角度和角速度。

**Reward 设计的直觉 (Table X)：**
为了让机器人走得像样且安全，他们用 PPO 训练，Reward 各有深意：
- **线速度追踪**：$r_{lin} = \exp(-\|v^{cmd}_{xy} - v_{xy}\|_2^2)$，权重 1.5。$v^{cmd}_{xy}$ 是大脑给的 0.5 m/s 目标速度，$v_{xy}$ 是实际速度。用 $\exp$ 包住平方误差，当误差大时 reward 饱和不爆炸，误差小时近似线性，非常稳定。
- **脚打滑惩罚**：$-0.05 \cdot \|v_{feet}\| \cdot \mathbb{1}[\dot{F}_{feet} > 1]$。下标 $feet$ 代表脚部，$v_{feet}$ 是脚的线速度，$\dot{F}_{feet}$ 是脚底接触力的变化率。$\mathbb{1}[\cdot]$ 是指示函数，当接触力变化大说明脚踩地了。这段公式意思是：如果脚踩在地上有接触力，但此时脚还在快速滑动，就狠狠惩罚。防止机器人在冰面或草地上打滑。

**单阶段训练：**
以前搞腿足机器人喜欢搞 Teacher-Student 蒸馏，先训一个有上帝视角的 teacher，再让只看传感器的 student 模仿。NaVILA 直接 single-stage 训练。Actor 只看真实的 LiDAR height map 和 proprioception (本体感觉)，Critic 在训练时偷看 simulator 的真实地形。这导致 collision rate 暴降 74% (Table V: 0.81 vs 3.09)。直觉上，自己在泥地里摸爬滚打学出来的避障，比看老师录像学出来的强得多。

### 3. 实验数据表与量化部署

**Benchmark 大跃迁 (Table I)：**
在 R2R-CE Val-Unseen 上，NaVILA 只用单目 RGB，Success Rate (SR) 达到 54.0%。那些用全景图+深度图+里程计的传统方法，即便上了 simulator pre-trained 的 waypoint predictor，最好也才 57.0% 左右。VLM 的 common sense 推理直接补足了传感器硬件的短板。

**VLN-CE-Isaac 新基准 (Table IV)：**
Habitat simulator 里 agent 是个幽灵，能穿过 10cm 的沙发缝。NaVILA 在 Isaac Sim 里建了考虑真实物理碰撞的新基准。Go2 机器人在这里盲走 SR 只有 36.2%，加上 LiDAR 视觉后 SR 提到 50.2%。

**量化 (Table XIII)：**
这点很实用。用 AWQ 量化把大模型从 FP16 压到 W4A16 (权重 4-bit，激活 16-bit)。
GPU 显存：18.5GB $\rightarrow$ 8.6GB。
延迟：594ms $\rightarrow$ 368ms (快了 40%)。
Success Rate 几乎没掉 (48.2% vs 49.7%)。
这意味着整个大脑大模型可以塞进机器狗背上的 RTX 4090 里跑，彻底摆脱云端图传的网络延迟。

### 4. 狂想与关联

看这个架构，我立刻联想到生物学里的 Central Pattern Generator (CPG)。低级神经中枢负责高频的节律运动（走路、避障），而大脑皮层负责低频的高级意图（去哪、干嘛）。NaVILA 的 two-level framework 简直就是 CPG 理论在现代 Deep RL 里的复现。

更 further 的是，语言作为一种 bottleneck representation，极度类似人类自身的“内隐语言”思考过程。当我们走路时，大脑并不会去想收缩哪块肌肉，大脑想的是“去冰箱拿瓶水”。NaVILA 把这种 cognitive hierarchy 用极简的工程手段实现了。

未来的极致演化，我猜会引入 **World Model**。现在的 VLA 是 reactive 的，看一步走一步。如果大脑里内嵌一个像 Sora 那样的视频生成世界模型，大模型可以在输出 "move forward 75cm" 之前，先在脑内“播放”一下往前走 75cm 会看到什么，如果脑内预测会撞墙，就换一个 action。这就是 DreamWalker 探索的方向，把它和 NaVILA 的 real-world 躯体结合，就是具身智能的终局形态。

### Web Links for Reference
- NaVILA Project: https://navila-bot.github.io
- VILA (Base VLM): https://arxiv.org/abs/2312.07533
- MASt3R (3D Pose estimator for YouTube videos): https://arxiv.org/abs/2406.09756
- Isaac Lab (Simulator): https://github.com/isaac-sim/IsaacLab
- AWQ (Quantization): https://arxiv.org/abs/2306.00978
- NaVid (Previous SOTA similar approach): https://arxiv.org/abs/2402.15852
- DreamWalker (Mental planning idea): https://arxiv.org/abs/2310.05560

---

# NaVILA Paper Deep Dive

Andrej, 这篇paper的核心贡献是把 Vision-Language Navigation (VLN) 从 discrete/continuous 的 simulator 设定推到真实的 legged robot 上, 而且提出了一个非常 elegant 的 two-level hierarchy。下面我尽量把每个技术细节都讲透, 让你 build intuition。

## 1. 问题动机:为什么 legged robot + VLN 难

传统 VLN 在 Habitat 这类 simulator 里做, agent 在 navigation graph 上 teleport, 或者最多在 continuous space 里用一个 simplified kinematic model 移动。这种 setup 根本不考虑 legged robot 的 physics: 12 个 joint motor 的 torque limit, 地面 friction, 机器人 base 的 wobble, 透明玻璃感知失效, 强光下 RGB 失效等等。Legged robot (quadruped/humanoid) 的好处是可以走 cluttered scene、爬楼梯、过草地, 但坏处是 low-level control 极其复杂。

直接用 end-to-end VLA (像 RT-2, OpenVLA) 把 VLM 的输出压成 quantized low-level joint command 有几个 fundamental 问题:
- LLM/VLM 的 pretraining 分布是 natural language, 你逼它输出 12 维 torque, 它的 reasoning 能力会被 dilute
- 不同 robot 的 joint configuration 不同, 模型不能跨 robot 迁移
- VLM 推理慢 (1 FPS), 但 low-level control 要 50-100 Hz, 频率严重 mismatch
- 没有 obstacle avoidance 的 closed-loop 反应能力

NaVILA 的核心 insight: **把 action 也保持在 language domain**, 让 VLM 输出像 "move forward 75cm" / "turn left 30 degrees" 这样的 mid-level command, 然后 low-level RL policy 把它翻译成 joint torque。这个 decoupling 一举解决了上面所有问题。

## 2. 整体架构 Intuition

```
Human Instruction (language)
        ↓
   ┌─────────────┐  low frequency (~1 Hz)
   │   VLA (VILA) │  ← RGB frames + history + instruction
   └─────────────┘
        ↓
  mid-level action (language: "move forward 75cm")
        ↓
   ┌─────────────┐  high frequency (~50 Hz)
   │ Locomotion  │  ← LiDAR height map + proprioception
   │   Policy    │
   └─────────────┘
        ↓
  joint position q^d ∈ R^12
        ↓
   PD controller → torque → robot
```

这个 hierarchy 有两个 timescale, 这其实是 control theory 里经典的 singular perturbation 思想: VLM 在 "slow manifold" 上做 planning, locomotion policy 在 "fast manifold" 上做 tracking + obstacle avoidance。这种 decoupling 让两个 subsystem 各自可以独立训练、独立改进。

## 3. High-level VLA: Taming VILA for Navigation

### 3.1 为什么选 VILA 而不是 video VLM

VILA 是 NVIDIA 的一族 image-based VLM (https://nvila-6.github.io/)。Paper 里明确说不用 video encoder, 原因是 video-text pretraining data 不够。Image-based VLM 的 generalization 更强。VILA 的特殊之处在于它的 pretraining 阶段就用了 image interleaved corpus (比如 MMC4), 所以 multi-image reasoning 能力强, 这对 VLN 处理历史帧非常关键。

VILA 的三阶段训练:
1. **Connector alignment**: 冻住 LLM 和 vision encoder, 只训 projector (MLP), 用 alignment data
2. **Visual-language pretraining**: 训 connector + LLM, 用 interleaved image-text corpus
3. **Instruction tuning**: 全部 unfreeze, 用 instruction tuning data

NaVILA 从 stage 2 的 checkpoint 开始 fine-tune。

### 3.2 Navigation Prompt 设计: 关键创新

这是我觉得最 clever 的地方。传统 VLM 处理 video 是 uniform sample frames, 然后全部堆在 text 前面。但 VLN 里, **当前帧** 和 **历史帧** 的语义角色完全不同:
- 当前帧 t: 用来做 immediate decision (是否要转弯, 是否到达目标)
- 历史帧 0..t-1: 用来做 memory bank (记住起点, 已访问的地方, 推理下一步)

NaVILA 的 frame sampling:
1. 强制提取最新帧 t 作为 current observation
2. 从前 t-1 帧里 uniform sample, 但保证第一帧 (起点) 一定包含
3. 用 textual cue 区分: "a video of historical observations: ⟨frame1⟩...⟨frame_k⟩ current observation: ⟨frame_t⟩"

这个设计避免了引入 special token (像 NaVid 那样), 保持 LLM 输入输出都在 language domain, 能完全 leverage pre-trained LLM 的 reasoning 能力。Paper 里测试了 8 到 64 frames, 在 R2R-CE 上 8 frames 已经够了, 增加更多 frames 收益有限 (Table IX)。

Intuition: 这其实就是 "working memory" 的设计模式。current observation 是 sensory input, historical frames 是 episodic memory。用 textual prefix 区分, 让 LLM 自己学会 weight 它们。

### 3.3 从 Human Touring Videos 学习

这是 paper 的另一个亮点, 第一次证明直接用 human video 训练能改进 continuous navigation。Pipeline:
1. 2K YouTube egocentric touring videos
2. Entropy-based sampling [26] (Lin et al., ICCV 2023, https://arxiv.org/abs/2307.08009) 切成 20K trajectories
3. 用 MASt3R [27] (https://arxiv.org/abs/2406.09756) 做 metric camera pose estimation, 提取 step-wise 6-DoF action
4. 用 VLM caption + LLM rephrase 生成 natural language instruction

MASt3R 是关键: 它能在 wild 场景下估计 metric scale 的 camera pose, 这意味着可以提取 "前进 X 米, 转弯 Y 度" 这种 metric action label, 而不只是 discrete "go to next node"。

Intuition: 以前 VLN 用 human video 都是做 pretraining 帮助 landmark understanding, NaVILA 第一次把 human video 当成直接的 supervision signal 训练 continuous navigation。这本质上是把 "人类旅游视频" 当成一种 weakly supervised robot demonstration。

### 3.4 SFT Data Blend: 四类数据混合

这是防止 catastrophic forgetting + 增强 generalization 的关键:

1. **Navigational data from real videos** (上面讲的)
2. **Navigational data from simulations**: R2R-CE 和 RxR-CE, 用 Habitat 的 shortest path follower 生成 step-wise video。一个 trick: 把连续 actions 合并 (两个 "forward 25cm" 合成 "forward 50cm"), 最多合并 3 个, 这样 action 分布更多样, 减少 overfitting。还有 label rebalancing, 因为 stop action 太少。
3. **Auxiliary navigational data**: EnvDrop 的 augmented instructions + trajectory summarization task (给定 trajectory video, 让 LLM 描述路径) + ScanQA (3D scan QA, 用 multi-view RGB)
4. **General VQA**: ShareGPT4V, Video-ChatGPT 等, 保持 general capability

Intuition: 这是经典的 "multi-task learning 防止 overfitting 到 specific action distribution" 思路。如果你只训 navigation data, VLM 会退化成一个 narrow navigation model, 失去 general reasoning 能力。混合 general VQA 是一个 regularizer。

## 4. Low-level Locomotion Policy

### 4.1 为什么 single-stage 而不是 teacher-student

大多数 perceptive locomotion 工作 (Lee et al. Science Robotics 2020, Miki et al. 2022) 用两阶段:
1. Stage 1: 训一个 privileged policy, 输入是 privileged terrain height map + proprioception
2. Stage 2: 训一个 student policy, 用 privileged policy 做 distillation, 输入只有 sensor data (proprioception + height map)

NaVILA 用 single-stage PPO, 直接让 actor 在真实 sensor 输入上训练。好处:
- 省一个 distillation stage, 时间效率高
- Policy 直接和 environment 交互, 能 explore 到 distillation 学不到的策略
- Isaac Lab 的 ray-casting 让训练 throughput 达到 60K FPS on RTX 4090

但 critic 仍然用 privileged observation (包括 terrain height scan around robot + linear velocity)。这是 asymmetric actor-critic 的标准 trick: critic 只在 training 时用, deployment 时只有 actor。

### 4.2 Observation / Action Space 详解

**Critic observation** $\mathbf{o}^c$ (training only):
- Proprioception: linear/angular velocity, orientation (roll/pitch/yaw), joint position $q \in \mathbb{R}^{12}$, joint velocity $\dot{q} \in \mathbb{R}^{12}$, previous action $a_{t-1}$
- Velocity command (from VLA)
- **Privileged terrain height scan** around robot (从 simulator 直接读)

**Actor observation** $\mathbf{o}^a$ (deployment):
- Proprioception history (注意: linear velocity 被排除! 因为真实 robot 测不准)
- Velocity command
- **LiDAR height map** (从 raw point cloud 重建)

Action: $a_t = q^d_t \in \mathbb{R}^{12}$ (desired joint position), 通过 PD controller 转 torque:
$$\tau = K_p (q^d - q) + K_d (\dot{q}^d - \dot{q})$$
其中 $K_p$ 是 stiffness, $K_d$ 是 dampness, 是 manufacturer 提供的。

### 4.3 LiDAR Height Map 重建

为什么选 LiDAR 不选 depth camera:
- Depth camera 在强光下失效
- 透明玻璃会"看穿", depth 全是 noise
- LiDAR (Unitree L1) 360°×90° FoV, 15Hz, 对透明物体和强光 robust

重建流程:
1. Raw point cloud 从 LiDAR ray-casting
2. 在 robot body frame 下建 voxel grid, X range [-0.8, 0.2]m, Y range [-0.8, 0.8]m, Z range [0.05, 0.5]m, voxel size 0.06m
3. 每个 voxel 取最低 height value (这样能检测到 overhang obstacle)
4. 对最近 5 帧 LiDAR 做 max filter 平滑 (temporal smoothing, 抑制 noise)

Intuition: "取最低值" 是 conservative 策略, 因为 robot 要从下面钻过去, 高处的障碍不影响。但如果有 overhang (低矮桌子), 取最低值会让 robot 知道这里有东西要避开。Temporal max filter 是为了防止 flickering。

### 4.4 Reward Function 数学解析

Paper 的 Table X 给了 reward:

**Linear velocity tracking**:
$$r_{lin} = \exp(-\|v^{cmd}_{xy} - v_{xy}\|_2^2), \quad w = 1.5$$
其中 $v^{cmd}_{xy} \in \mathbb{R}^2$ 是 desired linear velocity (从 VLA command 来), $v_{xy} \in \mathbb{R}^2$ 是 actual base velocity。exp kernel 让 reward 在误差大时 saturate, 误差小时线性, 比 quadratic penalty 更 stable。

**Angular velocity tracking**:
$$r_{ang} = \exp(-(\omega^{cmd}_{yaw} - \omega_{yaw})^2), \quad w = 1.5$$
只 track yaw, 因为 roll/pitch 不应该被 command 控制。

**Linear velocity penalty (z)**: $-2.0 \cdot |v_z|$, 惩罚垂直方向运动 (跳跃), 保持稳定

**Angular velocity penalty (xy)**: $-0.05 \cdot \|\omega_{xy}\|_2$, 惩罚 roll/pitch rate

**Flat orientation**: $-2.0 \cdot (\text{some orientation error})$, 让 base 保持水平

**Joint acceleration**: $-2.5 \times 10^{-7} \cdot \|\ddot{q}\|_2^2$, 极小权重, 平滑动作

**Energy**: $-2 \times 10^{-5} \cdot \sum |\tau_i \dot{q}_i|$, mechanical power penalty

**Body height**: $-5.0 \cdot (h^{target} - h)^2$, 让 base height 保持目标值 (Go2 大约 0.35m)

**Feet slipping**: $-0.05 \cdot \|v_{feet}\| \cdot \mathbb{1}[\dot{F}_{feet} > 1]$, 当脚有 contact force ($\dot{F}_{feet}$ 是 contact force rate) 但还在滑动时惩罚, 这是 standard quadruped locomotion trick 防止脚打滑

### 4.5 Domain Randomization

Sim-to-real 的关键:
- Body mass: $\pm 3$ kg uniform noise
- Static/dynamic friction: $[0.4, 4.0]$
- Motor strength: $[0.9, 1.1]$
- System delay: $[\Delta_t, \Delta_t]$ (这个范围好像写错了, 应该是 $[-\Delta_t, \Delta_t]$ 之类)

## 5. VLN-CE-Isaac Benchmark: 新贡献

现有 VLN-CE benchmark 在 Habitat 上, agent 是简化 kinematic model, 能穿过 10cm 缝隙, 这对 quadruped 不现实。NaVILA 提了新 benchmark 基于 Isaac Sim (Isaac Lab, https://github.com/isaac-sim/IsaacLab):
- 用 R2R 的 1839 trajectories, 筛出 1077 traversable high-quality mesh
- 考虑 robot 的真实 physics, joint collision, foot contact
- 兼容多 robot (Go2, H1)

Table IV 结果显示:
- Go2 vision policy 比 blind policy SR 高 14% (50.2 vs 36.2)
- H1 vision policy 比 blind 高 21% (45.3 vs 24.4)
- Oracle policy (perfect command execution) 在 Go2 上比 NaVILA vision 高 15%, 说明 low-level 仍有改进空间

Intuition: H1 是 humanoid, 重心高, 容易摔, 所以 vision 帮助更大 (更需要感知避障), 但绝对 SR 比 Go2 低, 因为 humanoid 控制本身难。

## 6. 实验结果关键点

### 6.1 VLN-CE Benchmark (Table I)

NaVILA 在 R2R-CE Val-Unseen: SR 54.0%, SPL 49.0%, **只用 single-view RGB**, 而 SOTA 方法大多用 panoramic + depth + odometry + simulator-pretrained waypoint predictor。

关键观察: NaVILA 是第一个只用 single-view RGB 就能匹配甚至超过用 panoramic + waypoint predictor 的方法。这说明 VLM 的 reasoning 能力可以补偿 sensor input 的缺失。

### 6.2 Cross-dataset (Table II)

只在 R2R 上训, 在 RxR-CE 上 zero-shot 测试: NaVILA SR 34.3% vs NaVid 23.8%, 提升 10 个点。说明 mid-level language action 的 generalization 比 discrete action 强。

### 6.3 ScanQA (Table III)

64 frames 配置下 CIDEr 102.7, 超过 LEO (101.4) 等 3D LMM, 而 NaVILA 只用 multi-view RGB, 不用 3D scan 或 camera pose。这说明 NaVILA 学到的 spatial reasoning 是 general 的, 不依赖 3D representation。

### 6.4 Low-level Policy (Table V)

NaVILA (single-stage) vs ROA (teacher-student distillation):
- Linear vel error: 0.066 vs 0.161 (降低 59%)
- Angular vel error: 0.113 vs 0.152 (降低 26%)
- Collision rate: 0.81 vs 3.09 (降低 74%!)

Collision rate 大幅降低是关键, 说明 single-stage 直接和环境交互学到的 obstacle avoidance 比 distillation 强很多。

### 6.5 Real World (Table VI)

25 instructions × 3 repeats:
- NaVILA overall 88% SR (paper 摘要里说的)
- Complex instructions 75% SR
- 比 GPT-4o (作为 VLM baseline) 显著好
- 在 Booster T1 humanoid 上不用 retraining 直接用, SR 也很高, 验证 cross-embodiment

### 6.6 Quantization (Table XIII)

用 AWQ [111] (https://arxiv.org/abs/2306.00978) 把 FP16 → W4A16:
- GPU memory: 18.5GB → 8.6GB (减半)
- Latency: 594ms → 368ms (快 40%)
- SR 几乎不变 (48.2 vs 49.7)

这意味着可以 onboard deploy 到 robot, 不用传图到云端, 大幅减少 latency。这是 future work。

## 7. 相关联想与延伸思考

### 7.1 与 RT-2, OpenVLA 的对比

RT-2 (https://arxiv.org/abs/2307.15818) 和 OpenVLA (https://arxiv.org/abs/2406.09246) 都是 end-to-end VLA, 输出 quantized joint action。它们在 manipulation 上 work, 但 navigation 不一样:
- Manipulation 的 action space 相对低维 (7-DoF arm), 模型容易学
- Navigation 是 long-horizon, 需要 reasoning + memory, end-to-end 容易 catastrophic
- Navigation 的 action distribution 更连续, quantization 损失大

NaVILA 的 mid-level language action 是一个 generalizable 设计, 完全可以反过来用在 manipulation: "grasp the cup from the left side" 这种 mid-level skill command, 然后 low-level skill policy 执行。这其实就是 SayCan (https://arxiv.org/abs/2204.01691) 的思路, 但 NaVILA 把它做得更细粒度, 而且不需要 predefined skill library。

### 7.2 Hierarchical RL 的回归

NaVILA 本质上是 hierarchical RL 的 modern 版本:
- Old: Options framework (Sutton, Precup, Singh 1999, https://web.eecs.umich.edu/~baveja/Papers/options.pdf)
- New: VLM 做 high-level option selection, RL policy 做 low-level execution

但 NaVILA 的 high-level 不是选 option, 而是生成 continuous mid-level command in language。这避免了 predefined option set 的限制, 也避免了 end-to-end 训练的 instability。

### 7.3 Spatial Reasoning 的瓶颈

Paper 的 limitation 提到 error correction 弱 (Figure 13 失败案例)。这其实是所有 VLM-based navigation 的通病: VLM 没有 explicit 的 "backtracking" 机制。可能的改进方向:
- Chain-of-Thought prompting 让 VLM 先 reason 再 act
- 加入 explicit 的 self-localization (像 SpatialRGPT, https://arxiv.org/abs/2406.14196)
- 用 visual memory module 显式存储 explored map

### 7.4 Long-context 的潜力

VILA 已经支持 1024 frames (sequence parallel training)。如果未来 long-context LLM (像 Gemini 1.5 Pro 2M tokens) 能高效处理, VLN agent 可以有更强的 episodic memory。但 paper 里 Table IX 显示 8 frames 已经够, 说明 R2R-CE 的 instruction horizon 不长。Real-world long-horizon 任务可能需要更多。

### 7.5 与 EgoExo4D, MMScan 等 3D 数据集的关联

NaVILA 用 YouTube touring video, 但更结构化的数据集像 EgoExo4D (https://egoexo4d-data.org/) 提供 ego + exo 双视角, 可能有更丰富的 spatial supervision。MMScan (https://arxiv.org/abs/2406.09401) 提供 3D scan + language grounding, 可以增强 spatial understanding。

### 7.6 World Model 的角度

NaVILA 的 VLA 不预测未来 state, 是 pure reactive。如果加入 world model (像 Sora, https://openai.com/sora), VLA 可以 "imagine" 不同 action 的后果, 做 model-based planning。DreamWalker (https://arxiv.org/abs/2310.05560) 已经探索这个方向, 但还在 simulator 里。结合 NaVILA 的 real-world locomotion, 可能是下一步。

### 7.7 Inference 时间的 Engineering

VLA 1 FPS, locomotion 50 Hz, 这中间的 buffer 和 command interpolation 很关键。Paper 没细讲, 但实际部署需要:
- VLA 输出 mid-level command 时附带 duration (forward 75cm → 在 0.5m/s 下 1.5 秒)
- Locomotion policy 在这 1.5 秒里持续 track velocity
- VLA 在 locomotion 执行时异步生成下一个 command
- 如果 VLA 没生成完, locomotion 用上一条 command 直到完成或超时

这是一个 producer-consumer pattern, 在 robotics 里很常见。

### 7.8 Safety 和 Verification

NaVILA 的 low-level policy 用 LiDAR 做 obstacle avoidance, 这其实是一个 implicit safety layer。但 VLA 输出可能不合理 (比如 "move forward 100m" 直接撞墙)。Paper 用 regular expression parser 提取 action, 但没有 verification module。未来可以加:
- Action clipping (limit max distance/angle)
- Semantic safety filter (VLM 输出后 LLM check 一遍)
- Runtime monitoring (locomotion policy 检测到危险, 触发 stop)

## 8. 我的直觉总结

NaVILA 的核心贡献其实是一个"representation choice": 把 mid-level action 放在 language domain。这个 choice 引发了一系列好处:

1. **Modularity**: VLA 和 locomotion 独立训练、独立改进、独立 swap
2. **Generalization**: Language 是 broad distribution, 不会 overfit 到 specific robot
3. **Data efficiency**: Human video 可以直接用 (因为 human action 也能用 language 表达)
4. **Reasoning preservation**: VLM 的 reasoning 不被 action prediction 任务 dilute
5. **Cross-embodiment**: 同一个 VLA 跑 Go2 和 T1, 只换 locomotion policy

代价:
1. Latency: VLA 慢, 不能做高频 reactive planning
2. Precision: "forward 75cm" 没有 "joint torque [0.3, -0.2, ...]" 精确
3. Action space 限制: 只能表达 language 能描述的 action (前进/转弯/停止), 复杂 manipulation 不行

这个 trade-off 在 navigation 上是合适的, 因为 navigation 本身就是 high-level + low-frequency decision。如果要做 fine manipulation (插入 USB), 这个 hierarchy 可能不够, 需要 finer-grained mid-level (像 "approach to 2cm, align to 1 degree")。

Future work 方向我猜是:
- Finer mid-level action vocabulary (含 manipulation skills)
- 加入 world model 让 VLA predict 后果
- Onboard deployment (用 AWQ quantization)
- 更长 context 支持 longer-horizon 任务
- Error recovery 机制 (backtracking, re-planning)

## 参考 Web Links

- NaVILA 项目主页: https://navila-bot.github.io
- VILA: https://nvila-6.github.io/
- VLN-CE benchmark: https://jacobkrantz.github.io/vlnce_benchmark/
- Habitat: https://aihabitat.org/
- Isaac Lab: https://github.com/isaac-sim/IsaacLab
- MASt3R: https://github.com/naver/mast3r
- PPO: https://arxiv.org/abs/1707.06347
- Unitree Go2: https://www.unitree.com/go2
- R2R原始 paper: https://arxiv.org/abs/1711.11492
- NaVid (related work): https://arxiv.org/abs/2402.15852
- AWQ quantization: https://arxiv.org/abs/2306.00978
- SpatialRGPT: https://arxiv.org/abs/2406.14196
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- SayCan: https://arxiv.org/abs/2204.01691
- DreamWalker: https://arxiv.org/abs/2310.05560
- EgoExo4D: https://egoexo4d-data.org/
- Learning from YouTube videos (Lin et al.): https://arxiv.org/abs/2307.08009
- MMScan: https://arxiv.org/abs/2406.09401

如果你想深入某个具体方面 (比如 reward shaping 细节, 或 VLA 的 frame sampling 策略 ablation), 我可以再展开。
