---
source_pdf: NaVILA.pdf
paper_sha256: e02fcccae8aad56cf33234882de06b97256b76aad86a011171b83db40790bc8c
processed_at: '2026-08-05T22:02:30-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# NaVILA 用人话说

## 这篇paper想干啥？

想象你跟一只机器狗说："出门右转，到那个黄色消防栓左转，沿着坡走下去，停在门口前。"

机器狗得听懂这话，还得真迈开腿走过去，别撞墙别摔跤。

这就是 NaVILA 想解决的问题——**让腿式机器人听人话导航**。

---

## 为啥这事儿难？

以前大家搞 VLN (Vision-Language Navigation) 都是这么玩的：在仿真器里放一个agent，告诉它"去厨房"，它就在预设的导航图节点之间 **传送**。听起来像打游戏开挂，完全不考虑机器人怎么迈腿。

后来有人搞 VLN-CE (Continuous Environment)，让agent真的得一步步移动，但大多数还是用轮式机器人，或者假设完美执行命令。

到了腿式机器人（四足机器狗、人形机器人），问题就炸了：

1. 你不能直接让 VLM 吐出"左前腿关节角度0.3弧度"——VLM 是用自然语言训练的，逼它输出这种数值就废了它的reasoning能力
2. 腿式机器人得实时避障，VLM 又大又慢，跑不动高频控制
3. 不同机器人腿不一样，换个机器人就得重训

---

## NaVILA的核心思路

**让 VLM 说人话，让RL policy听人话。**

VLM 输出类似 "move forward 75cm" 这样的语言指令，locomotion policy 把它翻译成关节运动。

就这么简单一个解耦，但解决了一堆问题：

- VLM 待在它擅长的语言领域，reasoning能力保留
- 换机器人？换个 locomotion policy 就行，VLM 不用动
- VLM 慢就慢点（1秒1次），locomotion policy 高频跑避障，各干各的
- 训练数据好搞——YouTube旅游视频都能用，因为"往前走2米"这种action能从视频里估计出来

---

## 两个模块拆开看

### 高层 VLA：VLM 改造成导航agent

基于 VILA（NVIDIA的VLM家族），架构就是 Vision Encoder → Projector → LLM 老三样。

关键设计在 **prompt 怎么组织**：

VLN任务里，视频帧有两种角色：
- **最新一帧**：当前看到的，用来做即时决策
- **历史帧**：记忆库，用来追踪进度

NaVILA 在 prompt 里明确区分：
```
a video of historical observations: <frame1><frame3><frame5>
current observation: <frame_t>
```

不搞特殊token，全用自然语言cue，最大化保留预训练prior。

**训练数据**是个亮点，四类混着喂：
1. YouTube旅游视频（2K视频处理成20K轨迹）——真实世界泛化
2. R2R-CE/RxR-CE仿真轨迹——标准VLN任务
3. ScanQA、轨迹摘要等辅助任务——空间理解
4. 通用VQA数据——保持世界知识不退化

YouTube视频那条线很巧妙：用 MASt3R 估计相机位姿提取action，用VLM+GPT-4o生成指令描述。这是以前做不到的，因为连续action label难获取，现在metric pose estimation成熟了才enable。

### 低层 locomotion：单阶段RL策略

Unitree Go2，12个腿部关节，LiDAR做感知。

**为什么用LiDAR不用depth camera？**
- 玻璃depth相机瞎了，LiDAR能看见
- 强光下RGB和depth都退化
- Go2自带L1 LiDAR，不用额外装

Height map构造：把LiDAR点云体素化，每个grid取最低值，最近5帧做max filter平滑。

**训练上的关键选择**：single-stage，不做teacher-student distillation。

传统做法是两阶段：先训个privileged encoder吃height scan等特权信息，再蒸馏出个student encoder。NaVILA直接单阶段PPO，actor吃真实传感器数据，critic吃特权信息只在训练时用。

结果是collision rate从3.09降到0.81，4倍降低。直觉上single-stage让policy直接和环境交互，能探索到distillation dataset里没有的策略。

---

## 实验结果说啥

### VLN-CE benchmark

只用单视角RGB，没有depth、没有全景、没有odometry、没有simulator预训练的waypoint predictor。

R2R-CE上SR=54.0，比之前SOTA NaVid高17个点。跨数据集（只在R2R训，零样本测RxR）SR=34.3，比NaVid高10个点。

这说明mid-level language action让VLM学到的representation更transferable，不是overfit到特定数据集的action分布。

### VLN-CE-Isaac 新benchmark

Habitat允许agent穿过10cm沙发缝，腿式机器人根本过不去。所以用Isaac Sim建了个新benchmark，物理保真度高，考虑机器人关节和环境的真实交互。

Go2上vision policy比blind policy高14个点（SR 50.2 vs 36.2），证明height map的关键作用。

### 真实世界

25条指令，每条3次，分Workspace/Home/Outdoor三类环境。

几个看点：

1. **碾压GPT-4o**：GPT-4o在outdoor几乎全失败
2. **Human video救命**：没有YouTube视频训练的版本，outdoor simple任务SR=0；加上后SR=1.00。因为仿真数据indoor-dominated，human video补了outdoor泛化
3. **跨机器人**：同一个VLA直接用在Booster T1人形机器人上，camera高度和视角都不同，照样work。这验证了mid-level action的解耦价值

### 量化部署

用AWQ把8B模型从FP16量化到W4A16：
- 显存18.5GB→8.6GB
- 延迟594ms→368ms
- SR几乎无损（49.7→48.2）

意味着能直接在机器人上跑，不用传图到云端。

---

## 我读出来的intuition

**好的systems research是找到对的abstraction boundary。**

NaVILA没有发明新模型、新算法、新loss。它就是找到了"language action"这个边界，让VLM和RL policy各干各的。

这个边界之所以对：
- VLM在语言域reasoning最强，就让它输出语言
- RL policy在连续控制最强，就让它处理执行
- 两者通过语言接口通信，接口本身就是个compositional、robot-agnostic的representation

对比end-to-end VLA（如RT-2直接预测joint actions），NaVILA牺牲了一点端到端的优雅，换来了：
- 更好的reasoning（VLM不被数值输出污染）
- 更好的data efficiency（能用人视频训练）
- 更好的transferability（跨机器人零成本）
- 更好的engineering（双频率，各模块独立优化）

在2024-2026这个时间点，foundation model + specialized policy的modular路线，比"magic end-to-end model"更有生命力。因为前者可以复用所有VLM进步的成果，后者每次VLM升级都得重训。

---

## 局限和未来方向

Paper自承：
- Error recovery弱，偏离指令后不知道怎么纠正
- VLM计算密集（虽然量化缓解了）

我想到的延伸：
1. **更丰富的action space**：现在只有forward/turn left/turn right/stop四种，可以加"climb stairs""avoid obstacle on left"等semantic action
2. **Active perception**：现在被动看前方，可以加主动转头看左右
3. **DAgger式continual learning**：部署失败case反馈回训练数据
4. **Multi-agent**：language action接口天然支持多机器人通信

---

## 一句话总结

**VLM说人话，RL policy听人话，中间用语言当接口——简单粗暴有效，跨机器人能用，真实世界能跑。**

参考：
- 项目页：https://navila-bot.github.io
- VILA: https://arxiv.org/abs/2312.07533
- AWQ: https://arxiv.org/abs/2306.00978
- MASt3R: https://arxiv.org/abs/2406.09681
- VLN-CE: https://arxiv.org/abs/2006.08999
- R2R: https://arxiv.org/abs/1711.07280
- NaVid: https://arxiv.org/abs/2402.13752
- Isaac Lab: https://arxiv.org/abs/2307.10745

---

# NaVILA: Legged Robot Vision-Language-Action Model for Navigation 深度解读

## 一、整体定位与动机

这篇paper把 **Vision-and-Language Navigation (VLN)** 从传统的离散导航图设置，推到了 **legged robot (quadruped + humanoid) 真实部署** 的层面。核心问题非常Karpathy-style：如何把人类语言指令一路翻译到低层的leg joint actions？

传统路线有几个关键痛点：

1. **End-to-end VLA (如 RT-2, OpenVLA)** 直接让VLM吐出量化后的low-level actions，但这违背了LLM/VLM的训练本质——它们是 natural language first 的，强迫它们输出精确的non-verbal joint torques 会让 reasoning 能力退化，而且 severe overfitting 到特定robot morphology。
2. **Discrete VLN (R2R/RxR 原始版本)** 假设agent可以在nav-graph节点之间teleport，完全忽略了 low-level motion planning，到真实硬件上无法部署。
3. **VLN-CE + waypoint predictor** (如 Hong et al. 的 waypoint model) 用simulator预训练的predictor预测附近候选点，依赖simulator-specific数据，泛化性差，且候选点只覆盖nearby locations，无法处理low-level obstacle avoidance。

NaVILA 的核心insight：**在VLM和locomotion policy之间插入一个"语言形式的mid-level action"作为接口**。VLM输出 "move forward 75cm" / "turn right 30 degrees" 这样的language action，locomotion RL policy把它当成velocity command来执行。

这样的好处是：
- VLM保持在语言域内，充分利用预训练的reasoning能力
- 同一个VLA可以跨robot部署（Go2, H1, T1都验证了），只需要换low-level policy
- 双频率运行：VLM低频(1 FPS)做高层planning，locomotion policy高频做实时避障
- 可以用 human touring videos 直接训练（因为mid-level action可以从camera pose估计出来）

项目页：https://navila-bot.github.io

---

## 二、Two-Level Framework 架构解析

### 2.1 高层 VLA (Vision-Language-Action Model)

基于 **VILA** family (Lin et al. CVPR 2024, https://arxiv.org/abs/2312.07533) ，包含三个组件：

**Vision Encoder** → **MLP Projector** → **LLM**

- Vision encoder把input image转成visual token sequence
- Projector做下采样并映射到language domain
- LLM做auto-regressive generation

VILA的3-stage训练：
1. **Stage 1**: 冻结LLM和vision backbone，只训connector (alignment data)
2. **Stage 2**: 训练connector + LLM (image-text interleaved corpus)
3. **Stage 3**: 全部unfreeze做instruction tuning

NaVILA从stage 2的checkpoint出发，用自己设计的SFT data blend做fine-tuning。

### 2.2 关键设计：Navigation Prompts

普通video VLM (VILA) 处理视频时是均匀采样frames，然后放在所有text之前。但VLN中frames有两种不同role：

- **Current observation (frame t)**：做即时决策（如turn right at intersection, stop at goal）
- **Historical frames (frame 1 ~ t-1)**：作为memory bank，track progress, plan next step

NaVILA的采样策略：
- 始终抽取最近的 frame t 作为 current observation
- 从前面的 t-1 帧中均匀采样历史帧，并 **强制包含第一帧**
- 在prompt中用 textual cue 区分两类：
  - `a video of historical observations: <frame1><frame3><frame5>...`
  - `current observation: <frame_t>`

这里有个很重要的设计哲学：**不引入额外special tokens**（不像NaVid [12]）。因为新special token会增加LLM学习负担，破坏预训练language prior。保持input/output都在language domain，最大化利用预训练reasoning。

### 2.3 Learning from Human Touring Videos

这是个非常聪明的设计，也是paper的highlight之一。流程：

1. 从YouTube收集 **2K egocentric touring videos**
2. 用 **entropy-based sampling** (Lin et al. ICCV 2023, https://arxiv.org/abs/2305.09825) 处理成 **20K trajectories**
3. 用 **MASt3R** (Leroy et al. ECCV 2024, https://arxiv.org/abs/2406.09681) 做 metric camera pose estimation，提取step-wise actions
4. 用 **VLM** (VILA) captioning + **LLM** (GPT-4o) rephrasing 生成natural language instructions

关键insight：以前用human video只敢做pre-training或者discrete navigation，因为continuous setting下action label很难获取。但 **metric-pose estimation in the wild** 的成熟使得从视频中提取连续action变得可行——这是技术enabling work。

### 2.4 SFT Data Blend 四类数据

| 类别 | 来源 | 作用 |
|------|------|------|
| Navigational data from real videos | YouTube touring videos (20K trajectories) | 真实世界generalization |
| Navigational data from simulations | R2R-CE, RxR-CE (shortest path follower生成) | 仿真轨迹学习 |
| Auxiliary navigational data | EnvDrop augmented instructions, ScanQA | scene understanding, trajectory summarization |
| General VQA datasets | ShareGPT4V, Video-ChatGPT等 | 保持broad world knowledge |

仿真数据处理有几个重要细节：
- **Action merging**：把连续多个相同方向的action合并（如两个forward 25cm → 一个forward 50cm），最多合并3个连续action。这减少了数据规模，同时引入action diversity，避免overfitting到特定步长。
- **Label rebalancing**：stop action严重underrepresented，做rebalancing让分布更均匀。这个ablation在Table VII体现得很明显——不做rebalancing会让SR从49.7掉到30.0。

### 2.5 Inference时的action parsing

LLM输出text后，用 **regular expression parser** (Kearns 1991) 提取action type (forward/turn left/turn right/stop) 和 arguments (distance/angle)。Paper实验性地观察到所有实验中actions都成功matched——这印证了LLM输出格式可控性其实比想象中好。

---

## 三、Low-level Visual Locomotion Policy

### 3.1 Go2 平台配置

- **LiDAR**：Unitree L1，360°×90° FoV，15Hz point cloud
- **18 DoFs**：6 base DoF (unconstrained) + 12 leg joint motors
- Policy只控制12个leg joints

### 3.2 Action/Observation Space

**Action space**:
$$a = q^d \in \mathbb{R}^{12}$$

其中 $q^d$ 是12个leg joint的desired position，通过stiffness/dampness转换成torque输入simulator。

**Critic observation** $o^c$ (privileged):
- proprioception: linear/angular velocity, orientation, joint positions, joint velocities, previous action
- velocity command at time t
- privileged terrain height scan around robot

**Actor observation** $o^a$ (real-world deployable):
- 同上但 **去掉linear velocity**（真实世界不可靠获取）
- 用 **proprioception history** 隐式估计linear velocity
- LiDAR-based height map

这种actor/critic asymmetric design是 **teacher-student distillation** 的常见pattern，但NaVILA的关键区别是：**single-stage training**，不需要做distillation！Critic只用于训练时引导actor，部署时只actor上场。

### 3.3 Height Map from LiDAR Point Cloud

为什么用LiDAR而不用depth camera？paper给了三个理由：
1. **Transparent object detection**：玻璃表面depth camera完全失效，LiDAR可以检测
2. **Strong sunlight robustness**：强光下RGB和depth都退化
3. **Manufacturer provided**：直接用Go2自带的L1，不需要额外传感器

Height map构造过程：
- 32 channels LiDAR，vertical range (0, 90°)，horizontal (-180, 180°)，horizontal resolution 4°
- Voxel size 0.06m，X range [-0.8, 0.2]m，Y range [-0.8, 0.8]m，Z range [0.05, 0.5]m
- 对每个voxel grid取范围内 **最低值**
- 对最近5帧LiDAR point cloud应用 **maximum filter** 平滑height map

### 3.4 Single-stage RL Training

传统方法（Lee et al. Science Robotics 2020, Miki et al. Science Robotics 2022）用 **two-stage teacher-student**：
- Stage 1: 训privileged encoder处理height scan等privileged observation
- Stage 2: 训adaptation encoder模仿privileged encoder的输出

NaVILA的 **single-stage** 优势：
- 不需要policy distillation，time-efficient
- Policy直接和环境交互，可以发现novel strategies
- 在Isaac Lab ray-casting加持下，RTX 4090上能跑到 **60K FPS**，训练效率很高

**PPO** (Schulman et al. 2017, https://arxiv.org/abs/1707.06347) 算法训练。

### 3.5 Reward Function 解析 (Table X)

| Reward项 | Expression | Weight |
|----------|-----------|--------|
| Linear velocity tracking | $\exp(-\|v_{xy}^{cmd} - v_{xy}\|_2^2)$ | 1.5 |
| Angular velocity tracking | $\exp(-(\omega_{yaw}^{cmd} - \omega_{yaw})^2)$ | 1.5 |
| Linear velocity penalty (z) | $-\|v_z\|_2^2$ | -2.0 |
| Angular velocity penalty (xy) | $-\|\omega_{xy}\|_2^2$ | -0.05 |
| Flat orientation | $-\|\text{up vector}\|_2^2$ | -2.0 |
| Joint accelerations | $-\|\ddot{q}\|_2^2$ | $-2.5\times 10^{-7}$ |
| Energy | $-\|\tau \cdot \dot{q}\|$ | $-2\times 10^{-5}$ |
| Body height | $-(h^{target} - h)^2$ | -5.0 |
| Feet slipping | $-\|v_{feet} \cdot \mathbb{1}[\dot{F}_{feet} > 1]\|_2$ | 0.05 |

变量含义：
- $v_{xy}^{cmd}$: commanded linear velocity (XY plane)
- $v_{xy}$: actual linear velocity
- $\omega_{yaw}^{cmd}$: commanded yaw angular velocity
- $\omega_{yaw}$: actual yaw angular velocity
- $v_z$: vertical linear velocity (penalize bouncing)
- $\omega_{xy}$: roll/pitch angular velocity (penalize tilting)
- $\ddot{q}$: joint accelerations (penalize jitter)
- $\tau$: joint torque, $\dot{q}$: joint velocity (energy = torque × velocity)
- $h^{target}$: target body height
- $\dot{F}_{feet}$: foot contact force, $\mathbb{1}[\dot{F}_{feet} > 1]$: indicator that foot is in contact
- $v_{feet}$: foot velocity (slipping = contact but moving)

这个reward design是相当标准的quadruped locomotion recipe，但几个细节值得注意：
- **tracking reward用exp包**：这比linear penalty更平滑，奖励接近command时gradient大、远离时gradient小
- **Body height penalty weight -5.0 很重**：防止robot蹲下或跳起
- **Energy和joint acceleration权重极小**：因为这些量值很大，乘小系数才不会dominate

### 3.6 Domain Randomization (Table XI)

| Parameter | Value |
|-----------|-------|
| Body Mass | [-3.0, 3.0] kg |
| Static Ground Friction | [0.4, 4.0] |
| Dynamic Ground Friction | [0.4, 4.0] |
| Motor Strength | [0.9, 1.1] |
| System Delay | $[\Delta_t, \Delta_t]$ |

这是sim-to-real的关键，让policy在训练时见过各种物理参数变化，部署时对真实硬件参数不敏感。

---

## 四、实验结果深度分析

### 4.1 VLN-CE Benchmarks (Table I)

NaVILA只用 **single-view RGB**（无depth, 无panoramic, 无odometry），在 R2R-CE Val-Unseen 上：

- **SR = 54.0**，比NaVid (37.0) 高17个点
- **SPL = 49.0**，比NaVid (35.0) 高14个点
- 甚至超过了很多用panoramic + depth + odometry + waypoint predictor的方法（如CMA*, VLNOBERT*）

Table II的cross-dataset实验更有说服力：只在R2R训练，零样本测RxR-CE：
- NaVid: SR = 23.8
- NaVILA: SR = 34.3

**+10.5个点的提升**！这说明mid-level language action让VLM学到了更transferable的navigation representation，而不是overfit到特定数据集的action分布。

### 4.2 ScanQA 空间理解 (Table III)

64 frames的NaVILA在ScanQA Validation上：
- CIDEr = 102.7，超过LEO (101.4, 用3D scene)
- 比NaviLLM高约27个点

这表明NaVILA不仅是navigation specialist，scene understanding也强。原因是 SFT data blend里加了ScanQA和trajectory summarization任务，multi-task学习促进了spatial reasoning。

### 4.3 VLN-CE-Isaac Benchmark (Table IV)

这是paper的另一个contribution——基于Isaac Sim构建的新benchmark。

为什么需要新benchmark？Habitat-based VLN-CE允许agent穿过10cm的sofa缝隙，但quadruped/humanoid根本过不去。Isaac Sim的高保真物理simulation能真实反映robot-environment interaction。

从R2R Val-Unseen的1,839条trajectories中，选了1,077条mesh质量好且traversable的。

关键对比：
| 设置 | Go2 SR | H1 SR |
|------|--------|-------|
| Oracle (完美执行) | 51.3 | - |
| NaVILA-Blind (只用proprioception) | 36.2 | 24.4 |
| NaVILA-Vision (加LiDAR height map) | 50.2 | 45.3 |

**Vision vs Blind 提升14个点**，验证height map的关键作用。

**Oracle vs NaVILA-Vision差15个点**，说明low-level policy的execution误差确实会累积，但同时说明还有提升空间。

**H1 robot比Go2表现差**：因为humanoid尺寸更大，在相同scene中可通行性更低。这点很真实，反映出benchmark对robot morphology敏感。

### 4.4 Low-level Policy Performance (Table V)

| 方法 | Linear Vel. Error | Angular Vel. Error | Collision Rate |
|------|------------------|--------------------|--------------| 
| ROA (with BC loss) | 0.189 | 0.152 | 3.25 |
| ROA | 0.161 | 0.152 | 3.09 |
| NaVILA | 0.066 | 0.113 | 0.81 |

**Collision Rate从3.09降到0.81，4倍降低**！这是single-stage training相对于two-stage distillation的显著优势——direct environment interaction让policy学到更好的避障策略。

### 4.5 Real World Evaluation (Table VI)

25条instruction，每条重复3次，分Workspace/Home/Outdoor三类环境，分Simple/Complex两种难度。

关键观察：
1. **NaVILA vs GPT-4o**：在所有环境所有难度上全面碾压。GPT-4o在outdoor几乎失败。
2. **Human video的作用**：
   - NaVILA† (无human video) outdoor simple SR = 0
   - NaVILA (有human video) outdoor simple SR = 1.00
   
   这是惊人的差别！Human touring video对outdoor generalization至关重要，因为仿真数据indoor-dominated。
3. **跨embodiment generalization**：Booster T1 humanoid用同一个VLA，camera height和view angle都不同，依然能work。这印证了mid-level language action的解耦优势——high-level VLA与robot morphology无关。

### 4.6 Ablation Studies

**Memory size (Table IX)**：8 frames已经够用，更多frames收益有限（甚至有时还下降，可能是attention dilution）。Real-world用8 frames是因为latency约束。

**Quantization (Table XIII)**：用 **AWQ** (Lin et al. MLSys 2024, https://arxiv.org/abs/2306.00978) 把FP16量化到W4A16：
- GPU memory: 18.5GB → 8.6GB (减半)
- Latency: 594.58ms → 367.80ms (快40%)
- SR只从49.7掉到48.2，几乎无损

这意味着NaVILA可以直接在robot上部署，不需要offload到远端服务器，省去image transmission latency。

---

## 五、Key Intuitions and Design Lessons

### 5.1 为什么mid-level language action是好的action representation？

1. **LLM prior alignment**：VLM的reasoning能力是在language data上训练出来的，强迫它输出"joint torque [0.3, -0.1, ...]"会让它退化为一个lookup table。让它输出"move forward 75cm"，是在它熟悉的semantic space内运作。

2. **Compositionality**：语言actions天然compositional——"forward 75cm" + "turn right 30°"可以组合出复杂路径，而量化low-level actions则失去了这种结构。

3. **Cross-robot transfer**：mid-level action是robot-agnostic的，同一个VLA可以直接给Go2和H1用。如果end-to-end预测joint positions，必须为每个robot单独训练。

4. **Data scaling**：可以用YouTube touring videos、reasoning QA等多样化数据源训练VLA，因为不需要把它们对齐到specific low-level action space。

### 5.2 为什么single-stage RL比teacher-student distillation好？

直觉上distillation应该更稳定（有teacher监督），但实验显示NaVILA的single-stage更好。原因：
- **Distillation信息损失**：student encoder的capacity有限，privileged信息可能被压缩损失掉
- **Direct exploration**：single-stage让policy直接和环境交互，可以发现distillation dataset里没出现的策略
- **Simplicity**：少一个stage，少一类hyperparameter，少一个potential bug

### 5.3 双频率架构的engineering value

- VLA 1 FPS：跑large VLM做高层planning
- Locomotion policy 高频实时：处理obstacle avoidance

这种decoupling让两个模块各自优化：
- VLA可以做得更大（8B参数），不用考虑实时性
- Locomotion policy可以训练得超robust，处理各种dynamic obstacle

如果end-to-end，要么VLA做得小（牺牲reasoning），要么做cascaded inference（增加系统复杂度）。

---

## 六、与相关工作的联系

### 6.1 Robot Foundation Models谱系
- **RT-2 / OpenVLA / RT-X** (Brohan et al. 2023, Kim et al. 2024)：end-to-end manipulation VLA，但navigation任务上overfit严重
- **GNM / Nomad** (Shah et al. ICRA 2023, ICRA 2024)：general navigation model，但只处理short description或target image，不做instruction following
- **Quar-VLA** (Ding et al. ECCV 2024)：quadruped VLA但专注low-level action
- **NaVILA的差异化**：明确区分mid-level action和low-level execution，同时支持VLN instruction following

### 6.2 VLN领域演化
- **离散时代** (Anderson et al. CVPR 2018)：teleport between graph nodes
- **VLN-CE** (Krantz et al. ECCV 2020)：continuous but still mid-level action
- **Waypoint predictor时代** (Hong et al. CVPR 2022, Krantz et al. CVPR 2021)：simulator pre-trained predictor显著提分，但泛化性受限
- **NaVILA**：跳过waypoint predictor，直接让VLM输出mid-level language action，更generalizable

### 6.3 Legged Locomotion Learning
- **Lee et al. Science Robotics 2020** (https://www.science.org/doi/10.1126/scirobotics.abc4742)：two-stage teacher-student，LiDAR + depth
- **Miki et al. Science Robotics 2022** (https://www.science.org/doi/10.1126/scirobotics.abk2822)：训练时用预定义terrain height，部署时用外部工具
- **NaVILA**：single-stage，训练时就用真实LiDAR input，避免train-test discrepancy

### 6.4 VLM基础设施
- **VILA** (Lin et al. CVPR 2024, https://arxiv.org/abs/2312.07533)：NaVILA的backbone，多image推理能力强
- **LongVILA** (Xue et al. 2024, https://arxiv.org/abs/2409.01963)：sequence parallel training支持1024 frames
- **AWQ** (Lin et al. MLSys 2024, https://arxiv.org/abs/2306.00978)：W4A16量化让NaVILA能上robot
- **MASt3R** (Leroy et al. ECCV 2024, https://arxiv.org/abs/2406.09681)：metric pose estimation enabling human video learning

---

## 七、Limitations和未来方向

Paper自承的限制（Section V和Appendix E）：
1. **Error recovery弱**：Figure 13展示了一个failure case——robot一开始follow prompt但偏离后无法纠正。原因是训练数据里没有explicit reasoning about error correction。
2. **Image-based VLM计算密集**：虽然AWQ缓解了，但long-context LLM可能更efficient。

我自己想到的几个延伸方向：

### 7.1 Action空间扩展
当前mid-level action只有4种：{move forward, turn left, turn right, stop}。可以扩展到更丰富的semantic action space：
- `climb stairs`
- `avoid obstacle on the left`
- `traverse rough terrain`
- `approach the door`

这需要更精细的language grounding，但能让VLA表达更复杂intention。

### 7.2 Hierarchical RL with language subgoals
当前low-level policy只接收简单的velocity command。可以做成hierarchical RL，让low-level policy也理解language subgoal：
- High level: VLA outputs "go to the chair"
- Mid level: locomotion policy内部做path planning
- Low level: joint control

这样VLA只需要做semantic planning，不需要精确到cm级别的距离估计。

### 7.3 Active perception
当前NaVILA是passive observation——固定camera看着前方。可以加active head control，让robot主动look around收集信息：
- 接近intersection时主动转头查看左右
- 看不到goal时主动explore

### 7.4 Continual learning from deployment
真实部署时遇到的failure case可以反馈回训练数据。可以用 **DAgger** (Dataset Aggregation) 或者online RL的方式continually improve。

### 7.5 Multi-agent coordination
当前是单robot。可以扩展到multi-robot：
- 多个quadruped协同search
- Humanoid + quadruped team work

Language action接口天然支持multi-agent communication。

---

## 八、Overall Takeaway

NaVILA给我最深的impression是它的 **工程美学**：

> 用一个简单的解耦——mid-level language action——同时解决了VLM prior alignment、cross-robot transfer、data diversity、dual-frequency execution四个问题。

这其实是好的systems research的标志：**找到一个关键的abstraction boundary**，让两个复杂模块各司其职，整体效果超过end-to-end。

而且这个工作展示了 **2024-2026机器人研究的范式转变**：
- 不再是end-to-end from pixels to torques
- 而是leverage foundation model的reasoning能力 + specialized RL policy的execution能力
- 接口设计becomes the key research question

从engineering perspective，NaVILA的可复现性也很强：
- VILA是open的
- AWQ是open的
- MASt3R是open的
- Isaac Lab是open的
- R2R-CE/RxR-CE是open的

这意味着这个工作可以被快速reproduce和extend。在embodied AI这个fast-moving领域，这种modular approach比"magic end-to-end model"更有生命力。

Karpathy可能会感兴趣的几个angle：
1. **Mid-level abstraction as a general principle**：不仅是navigation，manipulation也可以用mid-level language action（如"grasp the cup handle" + low-level grasp policy）
2. **Data curation matters more than model architecture**：NaVILA的VLM backbone就是VILA，但SFT data blend的design才是性能关键
3. **Sim-to-real的"sim"需要更real**：VLN-CE-Isaac benchmark说明Habitat-style simulator已经不够，Isaac Sim的物理保真度是新的floor
4. **Quantization让大模型上robot**：W4A16让8B VLM在单卡4090上跑，这是robot部署的enabling technology

参考链接汇总：
- 项目页：https://navila-bot.github.io
- VILA paper: https://arxiv.org/abs/2312.07533
- LongVILA: https://arxiv.org/abs/2409.01963
- AWQ: https://arxiv.org/abs/2306.00978
- MASt3R: https://arxiv.org/abs/2406.09681
- VLN-CE (Krantz et al.): https://arxiv.org/abs/2006.08999
- R2R original: https://arxiv.org/abs/1711.07280
- NaVid: https://arxiv.org/abs/2402.13752
- Learning from YouTube videos (Lin et al.): https://arxiv.org/abs/2305.09825
- PPO: https://arxiv.org/abs/1707.06347
- Isaac Lab (ORBIT): https://arxiv.org/abs/2307.10745
- Lee et al. quadruped locomotion: https://www.science.org/doi/10.1126/scirobotics.abc4742
- Miki et al. perceptive locomotion: https://www.science.org/doi/10.1126/scirobotics.abk2822
- SpatialVLM: https://arxiv.org/abs/2401.12168
- SpatialRGPT: https://arxiv.org/abs/2406.09189
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- GNM: https://arxiv.org/abs/2303.00671
