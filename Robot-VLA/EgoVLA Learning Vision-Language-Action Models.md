---
source_pdf: EgoVLA Learning Vision-Language-Action Models.pdf
paper_sha256: ff3126eaa1f8c1d73c33ae7ed322285bf78eefa477fb32f22306229ddbef4162
processed_at: '2026-08-04T02:55:16-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 EgoVLA

---

## 一句话总结

**既然地球上已经有 80 亿个 "robot"（就是人）天天在各种环境里干各种活，还自带第一人称摄像头（眼睛），那直接从人的视频里学机器人操作不就完了吗？**

---

## 核心直觉

你想想现在 robot learning 的痛点是什么——缺 data。要收 robot data，你得有 robot hardware，得有 expert operator 去遥操作，一个 task 收几百条 demonstration 就累死累活。Open X-Embodiment (https://robotics-transformer-x.github.io/) 那个大数据集，多少个 lab 一起搞才搞出几十万条。

但人呢？全球 80 亿人，每天做饭、开抽屉、拧瓶盖、组装家具，全都是免费的 manipulation demonstration。而且人去的地方、操作的物体、用的工具，比任何 robot lab 都丰富几个数量级。

EgoVLA 的作者就问了一个特别朴素的问题：**人的动作和机器人的动作，到底差多少？**

答案 surprisingly 是：差得没你想的那么多。

人手有 15 个 PCA 维度的 MANO 参数（https://arxiv.org/abs/1711.01324），机器人手 Inspire 也是 6 个 active DOF。人的 wrist 在空间里的 6D pose，机器人的 end-effector 也是 6D pose。中间差的就是几个 geometric transform：
- Wrist 位置 → end-effector 位置：就是坐标系变换 + inverse kinematics
- MANO hand params → robot hand joint commands：就是一个 MLP 做 retargeting

所以如果一个 VLA model 能从第一人称视频预测出"人手接下来怎么动"，那这个预测结果稍微转一下就是 robot action。**你本质上已经有一个 robot policy 了，只是它现在是在人手坐标系里说话。**

这就是 EgoVLA 的 whole story。

---

## 整个 pipeline 的三步走

### Step 1: 收集人类第一人称视频数据

从四个已有 dataset 拼出来的：

| Dataset | 啥玩意 | 多大规模 | 有啥用 |
|---------|--------|----------|--------|
| **HOI4D** (https://arxiv.org/abs/2111.06077) | 单手操作各种物体 | 4000 videos | pick-and-place、开柜门这种基本操作 |
| **HOT3D** (https://arxiv.org/abs/2406.09598) | 33 个 rigid object 的交互 | 833 分钟 | 有非常精确的 3D hand pose 标注，可惜没 language |
| **HoloAssist** (https://arxiv.org/abs/2310.01962) | 复杂双手任务 | 166 小时 | 组装家具、换电池这种 long-horizon task，但 hand pose 比较噪 |
| **TACO** (https://arxiv.org/abs/2401.08399) | tool-action-object 组合 | 2317 sequences | 用工具的操作 |

总共搞出 ~500K image-action pairs。

**关键 trick**: egocentric video 里 camera 一直在动，如果你直接让 model 预测"未来 wrist 在哪"，目标会随 camera 晃。所以他们用 world-frame 的 camera pose 把 future wrist position 投影回当前 camera frame——这样 supervision signal 稳定，model 学的是"相对于当前视角的运动"。

采样率 3 FPS，比较稀疏，但够用。

### Step 2: 训一个 VLA 预测人类动作

Backbone 用 **NVILA-2B** (https://arxiv.org/abs/2412.04468)，NVIDIA 的 efficient VLM，2B 参数。为啥选这么小？因为要在 32 卡 A100 上 fine-tune 整个 model（包括 visual encoder），太大搞不动。

输入有四样：
- **6 帧 RGB**：当前 + 5 帧 history，间隔 0.2 秒，覆盖 1 秒。分辨率 384×384。
- **Language instruction**：描述 immediate skill（"Put can on saucer"），不是 high-level plan
- **Action query tokens**：用 vocabulary 最后 30 个 word ID 当 query token
- **Human proprioception**：wrist pose + hand pose

Action head 是个 300M 的 transformer（6 layer, hidden 1536），output 是未来 1 秒、30 Hz、共 30 步的 action chunk $A_t = [a_t, a_{t+1}, \dots, a_{t+30}]$。

这个 action chunking 的思路跟 ALOHA (https://arxiv.org/abs/2304.13705) 一脉相承——一次预测未来一段比一步步预测更平滑、更有 temporal consistency。

### Step 3: 迁移到 robot

这里有个 chicken-and-egg 问题：你想用 robot data fine-tune，但 robot data 是 robot action space 的，model 只懂 human MANO action space。怎么办？

**两步解决：**

**训练时（robot demo → human representation）**:

给定 robot 的 fingertip 位置 $\mathbf{J}_{\text{obs}} \in \mathbb{R}^{5\times3}$（5 个手指头），去找 MANO 参数 $\Theta \in \mathbb{R}^{15}$ 使得 MANO forward kinematics 算出来的 fingertip 位置 $\mathbf{J}_{\text{pred}}(\Theta)$ 跟观测对齐：

$$\mathcal{L}(\Theta) = \frac{1}{5}\sum_{i=1}^{5} \text{SmoothL1}(\mathbf{J}_{\text{pred}}(\Theta)_i, \mathbf{J}_{\text{obs},i})$$

变量解释：
- $\Theta \in \mathbb{R}^{15}$: MANO 的 15 个 PCA 系数，就是要优化的目标
- $\mathbf{J}_{\text{pred}}(\Theta) \in \mathbb{R}^{5\times3}$: 把 $\Theta$ 喂进 MANO forward kinematics 得到的 5 个 fingertip 3D 位置
- $\mathbf{J}_{\text{obs}} \in \mathbb{R}^{5\times3}$: 从 robot 示范里读到的 5 个 fingertip 实际位置
- $i \in \{1,...,5\}$: thumb, index, middle, ring, pinky 五个手指头
- SmoothL1: Huber loss，对 outlier 比纯 L2 鲁棒

优化完每个 robot demo 就有了对应的 MANO 表示，就能直接 fine-tune EgoVLA 了，architecture 都不用改。

**推理时（human prediction → robot action）**:

EgoVLA 吐出 wrist pose + MANO params 后：
1. Wrist pose 做 3D 坐标变换 → robot end-effector pose → IK 解出 arm joint angle
2. MANO params 喂进 MANO forward kinematics 得 3D hand keypoints → 一个小 MLP ([64,128,64]) 映射到 robot hand 的 DOF commands

这个 retargeting MLP 训了 2000 epoch，batch 2048，最终 fingertip position error $5 \times 10^{-5}$ 米——基本可以忽略。

---

## Loss Function 长啥样

总 loss 三个项加权：

$$\mathcal{L} = \lambda_{\text{wrist trans}} \mathcal{L}_{\text{wrist trans}} + \lambda_{\text{wrist rot}} \mathcal{L}_{\text{wrist rot}} + \lambda_{\text{joint}} \mathcal{L}_{\text{joint}}$$

权重（Table 3 里查的）：
- $\lambda_{\text{wrist trans}} = 20.0$（wrist 平移最重要）
- $\lambda_{\text{wrist rot}} = 5.0$
- $\lambda_{\text{joint}} = 5.0$

各项具体形式：

$$\mathcal{L}_{\text{wrist trans}} = \|\mathbf{T}_{\text{pred}} - \mathbf{T}_{\text{gt}}\|_2^2$$
- $\mathbf{T}_{\text{pred}} \in \mathbb{R}^3$: 预测的 wrist 3D 平移（camera frame）
- $\mathbf{T}_{\text{gt}} \in \mathbb{R}^3$: ground truth

$$\mathcal{L}_{\text{wrist rot}} = \|\mathbf{R}_{\text{pred}} - \mathbf{R}_{\text{gt}}\|_2^2$$
- $\mathbf{R}_{\text{pred}} \in \mathbb{R}^{3\times3}$: 把预测的 rot6D（https://arxiv.org/abs/1812.07035）通过 Gram-Schmidt 正交化恢复的 rotation matrix
- $\mathbf{R}_{\text{gt}} \in \mathbb{R}^{3\times3}$: ground truth rotation matrix

为啥不用 quaternion？因为 quaternion 有 antipodal symmetry（$q$ 和 $-q$ 表示同一个 rotation），网络学起来会卡。rot6D 取 rotation matrix 前两列做 6 维表示，在 SO(3) 上是连续的，没有这种 degenerate 问题。

$$\mathcal{L}_{\text{joint}} = \|\Theta_{\text{pred}} - \Theta_{\text{gt}}\|_2^2$$
- $\Theta_{\text{pred}} \in \mathbb{R}^{15}$: 预测的 MANO PCA 系数
- $\Theta_{\text{gt}} \in \mathbb{R}^{15}$: ground truth

Translation 权重是其他两项的 4 倍，说明 wrist 位置精度对 task 成功更关键，hand 姿态宽容度高一些——你抓个杯子手稍微歪一点没事，但 wrist 位置偏 10cm 就抓空了。

---

## Action Space 选择的 intuition

为什么 action space 这样设计？来拆解一下。

**MANO PCA 15 dims**:

MANO 原始有 15 个 ball joint × 3 DOF = 45 个自由度。但人手其实动不了那么自由，joint 之间有强相关——你弯食指的时候中指也会跟着动一点，这叫 tendon coupling。MANO 通过 PCA 发现前 15 个主成分就能 capture 绝大多数 hand pose 变异。所以 15 dims 是一个 sweet spot：表达力够，维度低，模型好学。

这跟 LLM 里用 byte-pair encoding 把字符压缩成 token 的精神类似——找数据内在的低维结构。

**Rot6D 而非 quaternion/euler**:

这个 Zhou et al. 2019 (https://arxiv.org/abs/1812.07035) 证明过：rotation 表征的连续性对网络学习至关重要。
- Euler angle: gimbal lock，还有不连续
- Quaternion: 看起来连续，但 $q$ 和 $-q$ 是同一个 rotation，网络 output 在两个等价点之间会跳变
- Rot6D: 取 rotation matrix 的前两列（6 个数），再通过 Gram-Schmidt 正交化恢复第三列。在 SO(3) 上是连续映射，学起来最稳

---

## Benchmark 设计

他们搞了个仿真 benchmark 叫 **Ego Humanoid Manipulation Benchmark**，基于 NVIDIA Isaac Lab (https://arxiv.org/abs/2308.12960)。

**Hardware setup**:
- Unitree H1 humanoid (https://github.com/unitreerobotics)
- 两只 Inspire dexterous hand，每只 12 DOF（6 active + 6 mimic joint）
- 30 Hz control frequency

**12 个 task 分两类**:

**Short-horizon（atomic skill, 7 个）**: Push-Box, Flip-Mug, Pour-Balls, Close-Drawer, Open-Drawer, Open-Laptop, Stack-Can

**Long-horizon（multi-stage, 5 个）**: Sort-Cans, Insert-Cans, Unload-Cans, Insert-And-Unload-Cans, Stack-Can-Into-Drawer

Long-horizon task 难多了，比如 Insert-And-Unload-Cans 是"先把右手的 can 插进 slot，再把左手的 can 插进 slot，然后把右边的卸下来，再把左边的卸下来"——这种 4 阶段串行任务对 policy 的 long-range planning 要求很高。

**Visual diversity**: 5 种房间 texture × 5 种桌子 texture = 25 种 visual configuration。Training 用 3 种，evaluation 在 22 种 unseen 上测。

**Demonstration 收集**: 用 OpenTelevision (https://arxiv.org/abs/2407.01512) + Meta Quest 3 遥操作，每 task 收 100 条 successful demo，episode 长度 100-500 帧。

Action space 是 36 维：
- Arm: end-effector control（IK 求解）
- Hand: direct joint actuation, PD control

Simulation 的 physics timestep $dt$ 根据 task 调：
- Contact-rich task（Flip-Mug, Pour-Balls, Drawer）用 $dt = 1/240$
- 其他用 $dt = 1/120$

Decimation 公式：
$$\text{simulation decimation} = \frac{1}{30 \cdot dt}$$

---

## 实验结果讲人话

### 短任务（Table 1）

Seen visual 上：

| 方法 | 平均成功率 |
|------|-----------|
| ACT (specialist, per-task 训) | 24.87% |
| EgoVLA-NoPretrain (只 fine-tune VLM, 没用人类视频) | 64.55% |
| EgoVLA (50% robot data) | 48.15% |
| **EgoVLA (full)** | **84.92%** |

Unseen visual 上：

| 方法 | 平均成功率 |
|------|-----------|
| ACT | 24.89% |
| EgoVLA-NoPretrain | 51.28% |
| **EgoVLA** | **69.11%** |

**看点**:
1. EgoVLA 比 ACT 高 60 个点——generalist 把多个 task 一起学共享了 low-level skill，specialist 从 scratch 学每个 task 太累
2. EgoVLA 比 EgoVLA-NoPretrain 高 20 个点——人类视频 pretraining 给了 manipulation prior
3. Unseen visual 上 EgoVLA 掉 15 个点，NoPretrain 掉 13 个点——pretraining 还是有 generalization 优势的

### 长任务（Table 2）

Seen visual:

| 方法 | 平均 SR | 平均 PSR (progress) |
|------|---------|---------------------|
| ACT | 2.22% | 26.47% |
| EgoVLA-NoPretrain | 26.67% | 54.93% |
| **EgoVLA** | **45.93%** | **80.78%** |

Unseen visual:

| 方法 | 平均 SR | 平均 PSR |
|------|---------|----------|
| ACT | 0.61% | 23.51% |
| EgoVLA-NoPretrain | 11.21% | 36.20% |
| **EgoVLA** | **28.79%** | **69.11%** |

**看点**:
1. ACT 在 long-horizon 几乎全崩（2.22%）——specialist 要从 scratch 学 multi-stage planning 太难
2. EgoVLA 比 NoPretrain 高 20 个点 SR——pretraining 主要是给 long-horizon 加 buff，因为 multi-stage task 需要把 atomic skill 组合起来，pretraining 提供了 skill library
3. **PSR 比 SR 高很多**：比如 EgoVLA 在 unseen 上 SR 28.79% 但 PSR 69.11%——意思是大部分 episode 完成了 70% 的子任务但最后一步失败了。这说明 pretraining 让 model 学会了 skill sequence，只是 execution 精度差一截。

### 关键 ablation

**EgoVLA (50% robot data)**: short-horizon 从 84.92 掉到 48.15，long-horizon 从 45.93 掉到 7.41——long-horizon 对 robot data scale 极其敏感。

**Data mixture ablation (Fig.7)**: 即使 HoloAssist 的 hand pose noisy、HOT3D 没 language、TACO 视觉单一，**加进去还是 transfer 正向**。这与 LLM pretraining 经验一致：noisy but diverse > clean but narrow。

### Spatial generalization (Fig.8)

Long-horizon task 在 object spawn position 上画成功率热图，发现**两个 peak**——一个偏左一个偏右，对应左手操作区/右手操作区。这是 bimanual task 的天然结构，很 intuitive。

---

## 几个有意思的细节

### 1. Zero-shot deployment 完全失败

Paper 自己承认：直接把 EgoVLA 部署到 robot 上不加 fine-tuning，所有 task 0% success。即使 task 在 pretraining 数据里出现过（比如 Pour-Balls）也失败。

原因：
- Human 头戴 camera vs robot torso-mounted camera → 视角不一样
- Human hand vs robot hand morphology 不一样
- Human environment vs robot sim environment appearance 不一样

所以 pretraining 学的是 **manipulation skill prior**，不是可直接 deploy 的 policy。必须 fine-tune 一下。

### 2. Language-conditioned behavior 验证

Figure 6 做了个很 elegant 的实验：固定 visual input，改 language instruction，看 predicted trajectory 变不变。

例子：visual 里有人在柜子前。
- Instruction "Put it in the drawer" → 预测手往 drawer 里伸
- Instruction "Take it out of the drawer" → 预测手往柜子表面方向

这证明 model 学到的不是 fixed stimulus-response mapping，是真正把 language semantics 与 environment grounding 起来了。

### 3. Pretraining 20 epoch + Fine-tune 115 epoch

Pretraining 只跑 20 epoch（500K image-action pairs），但 fine-tune 跑 115 epoch（每个 task 100 demos）——这个比例有点反直觉。

直觉上你可能会想 pretraining 应该跑久一点。但作者 fine-tune 跑这么久可能是因为：
- Robot demo 数量少（每 task 100 条），需要多 epoch 才能 overfit 出 task-specific precision
- LR 在 epoch 100 后从 2e-5 降到 2e-6，说明后期主要是 refinement

### 4. Pretraining 的 LR 是 1e-4，fine-tune 是 2e-5

差 5 倍。Pretraining 阶段 model 从 random init 学 manipulation prior，需要大 LR 探索；fine-tune 阶段要 preserve pretraining knowledge，用小 LR。

### 5. Human wrist prediction 误差

Paper 报告：未来 wrist 位置预测平均误差 ~8cm，2D image plane 上 normalized error ~0.13。

这个精度对人手运动 forecasting 来说已经是 SOTA 水平（跟 HOI-forecast 论文 https://arxiv.org/abs/2204.09429 打平）。8cm 看着挺大，但放到 robot action 上 fine-tune 会修正掉。

---

## 与其他工作的关系

### vs OpenVLA / Octo / RT-2 / π0

这几个都是大 VLA，但都依赖大规模 robot data：
- **OpenVLA** (https://openvla.github.io/): Open X-Embodiment 数据训练
- **Octo** (https://octo-models.github.io/): 同上
- **RT-2** (https://robotics-transformer2.github.io/): Co-fine-tune PaLI-X + robot data
- **π0** (https://www.physicalintelligence.company/blog/pi0): 用大规模 robot data 训 flow model

EgoVLA 的不同之处：用 human egocentric video 替代 robot data 作为 primary signal，robot data 只用于 fine-tune。

### vs EgoMimic / Humanoid Policy

- **EgoMimic** (https://projectegl.github.io/EgoMimic/): 同样从 egocentric video 学，但 task-specific，不是 generalist
- **Humanoid Policy** (https://arxiv.org/abs/2503.13441): UCSD 同组前作，思路类似但没做 VLA 框架

EgoVLA 把这两条线揉起来：egocentric pretraining + VLA generalist。

### vs Video Pretraining 工作

- **R3M** (https://arxiv.org/abs/2203.12601): 用 Ego4D 学 visual representation
- **MVP** (https://arxiv.org/abs/2212.05349): MAE pretrain
- **LAPA** (https://arxiv.org/abs/2410.11758): latent action pretraining

这些都是 pretrain encoder/presentation，EgoVLA 是 pretrain 整个 policy（with explicit action prediction）。

---

## 我的几点思考

### 为什么这条路会 work？

本质上是因为 manipulation skill 在 embodiment 之间有 transferable structure。你抓杯子的时候，"手要靠近杯子、合拢手指、往上抬"这个 abstraction 跟你是人手还是 robot hand 关系不大。Pretraining 学的就是这个 abstraction。

Fine-tune 解决的是 embodiment-specific 的执行细节——视角偏差、morphology 偏差、appearance 偏差。

### 这个范式能 scale 多远？

Paper 用的 pretraining data 是 ~500K pairs，跟 LLM 动辄 trillion token 比小巫见大巫。但 egocentric video 的潜力大得多：
- Ego4D (https://arxiv.org/abs/2110.01667) 已经有 3000 小时
- Ego-Exo4D (https://arxiv.org/abs/2404.04989) 加入第三人称视角
- 未来 AR 眼镜普及后数据会爆炸

瓶颈在 hand pose annotation 的精度。Meta Aria、Quest 3、Vision Pro 这些设备 hand tracking 越来越好，annotation 成本会下降。

### 还有什么没解决？

1. **Zero-shot deployment 仍失败**——paper 自己说 limitation。如果未来能做到 human pretraining 后 zero-shot 到 robot，那才是真正的 embodiment-agnostic policy。可能需要更 embodiment-invariant 的 action representation（比如 latent action, 参考 LAPA）。

2. **只在 simulation 评估**——real robot 上能否 work 未知。不过 paper 强调 benchmark 设计 philosophy 就是 reproducible evaluation，类似 LIBERO (https://arxiv.org/abs/2306.03310)。

3. **Long-horizon 仍远不够好**——unseen 上 SR 28.79%，离实用还远。需要更好的 high-level planner 或者 hierarchical VLA 架构。

4. **每 task 100 demos 还是不少**——如果能降到 10 demos/task 就更实用。few-shot fine-tuning 可能是 next step。

### 可能的延伸方向

- 把 action head 换成 diffusion policy (https://arxiv.org/abs/2303.04137) 处理 multi-modal action distribution
- 结合 world model (DreamerV3, https://arxiv.org/abs/2301.04104) 做 model-based planning
- 扩展到 mobile manipulation（人走动+操作）
- 把 VLM 的 reasoning 能力用来做 high-level task planning，与 EgoVLA 的 low-level skill 组合 hierarchical system
- 用 Gaussian Splatting / NeRF 把 egocentric video 扩展成多视角训练 data

---

## TL;DR

EgoVLA 证明了一件事：**robot manipulation skill 可以脱离 robot hardware 来学**。人在视频里怎么动手，这个 signal 经过几个 geometric transform 就能变成 robot action。人类视频做 pretraining，少量 robot demo 做 fine-tune，就能 train 出一个比 specialist 强得多的 generalist policy。

如果这条 scale 下去，robot learning 的 data 瓶颈可能真的会被打破。

Reference:
- Project: https://rchalyang.github.io/EgoVLA/
- Paper: https://arxiv.org/abs/2507.18140 (具体编号请核对)
- NVILA: https://arxiv.org/abs/2412.04468
- ALOHA/ACT: https://arxiv.org/abs/2304.13705
- OpenVLA: https://openvla.github.io/
- EgoMimic: https://projectegl.github.io/EgoMimic/
- Humanoid Policy: https://arxiv.org/abs/2503.13441
- π0: https://www.physicalintelligence.company/blog/pi0
- LAPA: https://arxiv.org/abs/2410.11758
- DreamerV3: https://arxiv.org/abs/2301.04104

---

# EgoVLA: 从第一人称人类视频学习 Vision-Language-Action 模型

这篇 paper 来自 UC San Diego Xiaolong Wang 组，与 MIT、NVIDIA、UIUC 合作。核心 insight 非常 elegant：**human 本身就是一种 "robot"**，全球有 80 亿个这样的 "robot" 在各种 environment 中持续 operating。如果能从 egocentric human video 训练 VLA，那么 data scale 与 task diversity 的瓶颈就同时被打破了。

Paper link: https://rchalyang.github.io/EgoVLA/
Arxiv: https://arxiv.org/abs/2507.18140 (实际编号请核对)

---

## 1. High-level Intuition: 为什么这条路是 work 的？

传统 robot learning 的 pipeline 是：teleoperation 收集 real robot data → behavior cloning / VLA training。瓶颈在于 robot hardware 与 expert operator 的存在，data 规模本质上被限制。

EgoVLA 的关键观察：**human action space 与 humanoid robot action space 之间的 gap 是可以被 geometric transform 逼近的**。具体来说：
- Human wrist 位置 → robot end-effector 位置（通过 IK）
- Human hand joints (MANO) → robot hand joints（通过 retargeting）

所以一个在 human egocentric video 上训练的 VLA，本质上已经是一个 robot policy，只需要少量 robot demonstration 来 correct appearance/kinematic mismatch。

这种思路与几个近期工作有思想关联：
- **EgoMimic** (https://arxiv.org/abs/2410.24221): 同样从 egocentric video 学习，但更聚焦于 specific task
- **Humanoid Policy** (https://arxiv.org/abs/2503.13441): UCSD 组的前作，从 human video 学习 humanoid policy
- **π0 / π0.5** (https://arxiv.org/abs/2410.24164, https://arxiv.org/abs/2504.16054): Physical Intelligence 的 VLA flow model，action space 设计上有相似 spirit
- **OpenVLA** (https://arxiv.org/abs/2406.09246): Stanford/Berkeley 的开源 VLA
- **RT-2** (https://arxiv.org/abs/2307.15818): Google DeepMind 的 VLA，将 web knowledge 迁移到 robot control

---

## 2. Dataset 构建: Ego-Centric Human Manipulation Dataset

### 2.1 数据源

整合 4 个 source，构建 ~500K image-action pairs：

| Dataset | 规模 | 特点 |
|---------|------|------|
| **HOI4D** (https://arxiv.org/abs/2111.06077) | 4,000 videos | 单手操作，pick-and-place, articulated object interaction |
| **HOT3D** (https://arxiv.org/abs/2406.09598) | 833 min | 33 rigid objects，精确 3D hand + camera pose，但无 language label |
| **HoloAssist** (https://arxiv.org/abs/2310.01962) | 166 hours | 复杂 bimanual task：battery replacement, furniture assembly。Hand pose noisy，所以采样 1/10 |
| **TACO** (https://arxiv.org/abs/2401.08399) | 2,317 sequences | 151 tool-action-object triplets |

### 2.2 关键处理

Egocentric video 的 challenge 在于 camera 一直在动，如果直接 supervise future wrist position，目标会随 camera 移动而漂移。解决方案：

**用 world-frame camera poses 把 future wrist positions 投影到 current camera frame**。这样 supervision signal 始终在同一个 reference frame 下，模型学到的 motion 是相对于当前 camera 的相对运动。

Visual observation 采样率 3 FPS，平衡 computational efficiency 与 temporal continuity。

### 2.3 Data mixture 的 ablation

Figure 7 的 ablation 很 informative：即使 HoloAssist 的 hand annotation noisy、HOT3D 没有 language label、TACO 视觉多样性有限，**加入它们依然带来 positive transfer**。这说明 pretraining data 的 diversity 比 purity 更重要。这点与 LLM pretraining 的经验一致：noisy 但 diverse 的 data 通常优于 clean 但 narrow 的 data。

---

## 3. Model Architecture: EgoVLA

### 3.1 整体架构

Backbone 选用 **NVILA-2B** (https://arxiv.org/abs/2412.04468)，NVIDIA 出品的 efficient VLM，2B 参数。这个 size 的选择是为了：
- 既有足够 capacity 进行 vision-language reasoning
- 又能在 32× A100 上 feasible 地 fine-tune 整个 model（包括 visual encoder）

输入：
- **6 帧 RGB observation**：当前帧 + 5 帧 history，间隔 0.2 sec，覆盖 1 sec 时间窗口。Resolution 384×384。
- **Language instruction**：描述 immediate desired behavior（不是 high-level plan，是 skill execution level）
- **Action query tokens**：用 vocabulary 最后 30 个 word IDs 作为 query tokens
- **Human proprioception**：wrist translation/rotation + hand pose

### 3.2 Action Head

Action head 是一个 300M 的 transformer：
- 6 encoder layers
- Hidden size 1536
- Input: proprioception state + action query token 的 latent embedding
- Output: 序列 $A_t = [a_t, a_{t+1}, \dots, a_{t+H}]$，其中 $H=30$，对应 1 秒 horizon，30 Hz 频率

Action chunking 与 ALOHA (https://arxiv.org/abs/2304.13705) 的设计一致，预测 future action sequence 而非单步 action，有 temporal consistency 优势。

### 3.3 Action Space 设计

这是 paper 最 clever 的设计之一。

**每个 predicted action 包含：**
1. **Wrist pose**: 3D translation + rotation (rot6D representation, 参考 https://arxiv.org/abs/1812.07035)
2. **Hand joint angles**: MANO model 的 top 15 PCA components

**为什么选 MANO PCA 15 dims？**

MANO (https://arxiv.org/abs/1711.01324) 是一个 parametric hand model，原本有 15 ball joints × 3 DOF = 45 DOF。但 human hand 实际上没有这么多 flexibility，joint 之间有 strong correlation。MANO 通过 PCA 降维，取前 15 principal components 就能 capture 绝大多数 hand pose variation。

**为什么用 rot6D 而非 quaternion 或 Euler？**

Zhou et al. 2019 (https://arxiv.org/abs/1812.07035) 证明 6D rotation representation 在神经网络中是 continuous 的，避免 quaternion 的 antipodal symmetry 问题和 Euler 的 gimbal lock 问题。具体来说，rot6D 取 rotation matrix 的前两列，然后通过 Gram-Schmidt 正交化恢复完整 rotation matrix。

---

## 4. Loss Function 详解

总 training objective：

$$\mathcal{L} = \lambda_{\text{wrist trans}} \mathcal{L}_{\text{wrist trans}} + \lambda_{\text{wrist rot}} \mathcal{L}_{\text{wrist rot}} + \lambda_{\text{joint}} \mathcal{L}_{\text{joint}}$$

各项的具体形式：

**Wrist translation loss (L2):**
$$\mathcal{L}_{\text{wrist trans}} = \|\mathbf{T}_{\text{pred}} - \mathbf{T}_{\text{gt}}\|_2^2$$

- $\mathbf{T}_{\text{pred}} \in \mathbb{R}^3$: predicted wrist translation in camera frame
- $\mathbf{T}_{\text{gt}} \in \mathbb{R}^3$: ground-truth wrist translation
- $\|\cdot\|_2^2$: squared L2 norm

**Wrist rotation loss:**
$$\mathcal{L}_{\text{wrist rot}} = \|\mathbf{R}_{\text{pred}} - \mathbf{R}_{\text{gt}}\|_2^2$$

- $\mathbf{R}_{\text{pred}} \in \mathbb{R}^{3\times3}$: 把 predicted rot6D 通过 Gram-Schmidt 转回的 rotation matrix
- $\mathbf{R}_{\text{gt}} \in \mathbb{R}^{3\times3}$: ground-truth rotation matrix

注意这里对 rotation matrix 元素直接做 L2 loss，而不是 geodesic distance。这种简化在 rot6D 已经连续的前提下是可行的。

**Hand joint loss (L2):**
$$\mathcal{L}_{\text{joint}} = \|\Theta_{\text{pred}} - \Theta_{\text{gt}}\|_2^2$$

- $\Theta_{\text{pred}} \in \mathbb{R}^{15}$: predicted MANO PCA coefficients
- $\Theta_{\text{gt}} \in \mathbb{R}^{15}$: ground-truth MANO PCA coefficients

**Weight 选择 (Table 3):**
- $\lambda_{\text{wrist trans}} = 20.0$
- $\lambda_{\text{wrist rot}} = 5.0$
- $\lambda_{\text{joint}} = 5.0$

Translation weight 是 rotation/joint 的 4 倍，说明 wrist 位置精度对 task success 更关键，hand pose 容忍度更高。

---

## 5. Embodiment Transfer: Unified Action Space

这是 paper 最核心的技术贡献。

### 5.1 从 Robot Data 到 Human Representation (训练时)

给定 robot demonstration，需要把它 align 到 human MANO space 才能 fine-tune EgoVLA。

**End-effector pose alignment**: 通过 3D transformation 对齐 robot 与 human 的 coordinate system。

**Hand configuration alignment** via optimization:

$$\mathcal{L}(\Theta) = \frac{1}{5}\sum_{i=1}^{5} \text{SmoothL1}(\mathbf{J}_{\text{pred}}(\Theta)_i, \mathbf{J}_{\text{obs},i})$$

- $\Theta \in \mathbb{R}^{15}$: 待优化的 MANO hand parameters
- $\mathbf{J}_{\text{pred}}(\Theta) \in \mathbb{R}^{5\times3}$: 通过 MANO forward kinematics 计算出的 5 个 fingertip 位置
- $\mathbf{J}_{\text{obs}} \in \mathbb{R}^{5\times3}$: observed robot fingertip 位置
- $i$: fingertip index (thumb, index, middle, ring, pinky)
- SmoothL1: Huber loss，对 outlier 鲁棒

这个 optimization 在 dataset preprocessing 阶段做一次，把所有 robot demonstration 的 hand state 转成 MANO 表示。

### 5.2 从 Human Prediction 到 Robot Action (推理时)

EgoVLA 预测出 wrist pose + MANO parameters 后，需要 map 回 robot：

**Wrist → end-effector**:
1. 3D transformation 转换 coordinate frame
2. Inverse Kinematics (IK) 求解 arm joint angles

**MANO parameters → robot hand actuation**:
1. MANO forward kinematics 算出 3D hand keypoints
2. 训练一个 lightweight MLP，input 是 3D hand keypoints (wrist frame, both hands), output 是 robot hand 所有 DOF 的 actuation values

MLP 结构：
- 4 layer network
- Hidden layers: [64, 128, 64]
- Training: 2000 epochs, batch 2048, LR 0.001

这个 retargeting MLP 的 fingertip position error 是 $5 \times 10^{-5}$ m，基本可忽略。Replay 原始 demonstration 经过这个 pipeline 后 task validity 保留，说明 retargeting error 不破坏 task semantics。

---

## 6. Ego Humanoid Manipulation Benchmark

### 6.1 设计哲学

Paper 强调这个 benchmark 不为 sim-to-real，而是作为 reproducible 的 evaluation testbed，类似 LIBERO (https://arxiv.org/abs/2306.03310) 和 SIMPLER (https://arxiv.org/abs/2405.05941)。

硬件配置：
- **Unitree H1** humanoid (https://github.com/unitreerobotics)
- **Inspire dexterous hands** (https://www.inspire-robots.com/)，每只手 12 DOF (6 active + 6 mimic)
- NVIDIA Isaac Lab (https://arxiv.org/abs/2308.12960)

### 6.2 12 个 Task 分类

**Short-horizon (atomic skill, 7 tasks):**
1. Push-Box
2. Flip-Mug
3. Pour-Balls
4. Close-Drawer
5. Open-Drawer
6. Open-Laptop
7. Stack-Can

**Long-horizon (multi-stage, 5 tasks):**
1. Sort-Cans
2. Insert-Cans
3. Unload-Cans
4. Insert-And-Unload-Cans
5. Stack-Can-Into-Drawer

### 6.3 Action Space

Total 36-dim action space:
- Arm: end-effector control（通过 IK 转换）
- Hand: PD joint control，direct actuation
- Control frequency: 30 Hz

### 6.4 Visual Diversity

5 room textures × 5 table textures = 25 visual configurations。
Training 用 Room 1/2/3 + Table 1，evaluation 在 unseen 上做。

### 6.5 Demonstration Collection

用 **OpenTelevision** (https://arxiv.org/abs/2407.01512) + Meta Quest 3 teleoperation，每 task 收集 100 个 successful demos。Episode 长度 100-500 frames。

### 6.6 Simulation 物理

关键公式:
$$\text{simulation decimation} = \text{render interval} = \frac{1}{30 \cdot dt}$$

- $dt$: physics simulation time step
- 30 Hz: control frequency

不同 task 用不同 $dt$：Push-Box 用 $1/120$，Flip-Mug/Pour-Balls/Drawer 用 $1/240$。这是因为 contact-rich task 需要更细的 physics step 来稳定。

---

## 7. Experimental Results 详解

### 7.1 Short-Horizon Results (Table 1)

**Seen visual configuration:**

| Method | Stack-Can SR | Push-Box SR | Open-Drawer SR | Close-Drawer SR | Flip-Mug SR | Pour-Balls SR | Open-Laptop SR | Mean SR |
|--------|--------------|-------------|----------------|------------------|--------------|----------------|----------------|---------|
| ACT | 22.22 | 11.11 | 18.52 | 48.15 | 7.40 | 3.70 | 62.96 | 24.87 |
| EgoVLA-NoPretrain | 55.56 | 51.85 | 59.26 | 100.00 | 3.70 | 85.19 | 96.30 | 64.55 |
| EgoVLA (50%) | 44.44 | 33.33 | 22.22 | 100.00 | 22.22 | 77.78 | 37.04 | 48.15 |
| **EgoVLA** | **77.78** | **70.37** | 59.26 | 100.00 | **59.26** | 92.59 | **100.00** | **84.92** |

**Unseen visual configuration:**

| Method | Mean SR |
|--------|---------|
| ACT | 24.89 |
| EgoVLA-NoPretrain | 51.28 |
| EgoVLA (50%) | - |
| **EgoVLA** | **69.11** |

### 7.2 Long-Horizon Results (Table 2)

**Seen:**
| Method | Insert-And-Unload SR | Stack-Can-Into-Drawer SR | Sort-Cans SR | Unload-Cans SR | Insert-Cans SR | Mean SR | Mean PSR |
|--------|----------------------|---------------------------|--------------|-----------------|----------------|---------|----------|
| ACT | 0.00 | 11.11 | 0.00 | 0.00 | 0.00 | 2.22 | 26.47 |
| EgoVLA-NoPretrain | 7.41 | 0.00 | 51.85 | 62.96 | 11.11 | 26.67 | 54.93 |
| EgoVLA (50%) | 0.00 | 29.63 | 0.00 | 0.00 | 7.41 | 7.41 | 39.70 |
| **EgoVLA** | **44.44** | 40.74 | 55.56 | **66.67** | 22.22 | **45.93** | **80.78** |

**Unseen:**
| Method | Mean SR | Mean PSR |
|--------|---------|----------|
| ACT | 0.61 | 23.51 |
| EgoVLA-NoPretrain | 11.21 | 36.20 |
| **EgoVLA** | **28.79** | **69.11** |

### 7.3 Key Findings 解读

**Finding 1: Pretraining 必需，zero-shot transfer 失败**

Paper 明确说：no fine-tuning 时所有 task 0% success，即使 task 在 pretraining 中出现过（如 Pour-Balls）。原因：
- Camera pose mismatch（human 头戴 camera vs robot torso-mounted）
- Hand morphology mismatch
- Visual appearance mismatch

这说明 pretraining 学的是 manipulation skill 的 prior，不是直接的 policy transfer。

**Finding 2: Pretraining 主要帮助 long-horizon**

EgoVLA vs EgoVLA-NoPretrain 在 long-horizon 上的 gap (~20% SR) 大于 short-horizon。这是因为 long-horizon 需要 combine 多个 atomic skill，pretraining 提供 skill library。

**Finding 3: Generalization 优势显著**

Unseen visual 上：
- EgoVLA short-horizon SR 从 84.92 → 69.11（下降 ~16%）
- EgoVLA-NoPretrain short-horizon SR 从 64.55 → 51.28（下降 ~13% absolute，~20% relative）

EgoVLA 的 absolute drop 与 NoPretrain 类似，但 NoPretrain 本身 seen 性能就低。

**Finding 4: Data scale 仍重要**

EgoVLA (50%) 的 long-horizon SR 从 45.93 → 7.41，剧烈下降。说明 100 demos/task 是 sweet spot，50 demos 不足以 fine-tune 出可靠 policy。这与 OpenVLA/Octo 的观察一致：pretraining 不能完全替代 in-domain data。

**Finding 5: Spatial generalization 双峰分布**

Figure 8 显示 long-horizon task 在 object spawn position 上有 two peaks，对应 left hand / right hand 各自的高成功率区域。这是 bimanual task 的 natural structure。

---

## 8. Instruction Following 验证

Figure 6 是一个非常有意思的 qualitative experiment：固定 visual input，改 language instruction，看 predicted wrist trajectory 是否相应变化。

例子：
- Visual: 有人在 cabinet 前
- Instruction "Put it in the drawer" → trajectory 朝 drawer 内部
- Instruction "Take it out of the drawer" → trajectory 朝 cabinet 表面

这证明 EgoVLA 学到的不是 stimulus-response mapping，而是真正理解 language semantics 与 environment 的 grounding。

---

## 9. Ablation Studies 深度解读

### 9.1 Pretrain Data Mixture (Figure 7)

测试不同 dataset 组合对 unseen visual generalization 的影响。结论：**多样性越强，generalization 越好**。即使每个单独 dataset 有缺陷（HoloAssist noisy、HOT3D 无 language、TACO 视觉单一），加在一起依然 positive transfer。

这与 CLIP、LLM pretraining 的 scaling law 思想一致：data diversity 是 generalization 的关键驱动力。

### 9.2 Robot Data Scale (Table 1, 2)

EgoVLA (50%) 用 50 demos/task：
- Short-horizon: 61.73 vs 84.92 (full)
- Long-horizon: 7.41 vs 45.93 (full)

Long-horizon 对 data scale 更敏感，因为 multi-stage task 需要 sufficient coverage 来 learn 各 stage 之间的 transition。

---

## 10. Training Details 详解

### 10.1 Hyperparameters (Table 3)

| 参数 | Pretraining | Post-training |
|------|-------------|---------------|
| Epoch | 20 | 115 |
| Batch size | 16×8×4=512 | 512 |
| LR | 1e-4 | 2e-5 (2e-6 after 100 epochs) |
| LR schedule | Cosine | Constant |
| GPUs | 32× A100 | 32× A100 |

Post-training 用 constant LR with manual decay at epoch 100，避免 cosine 在后期 LR 太小导致 underfitting。

### 10.2 ACT Baseline (Table 4)

ACT (https://arxiv.org/abs/2304.13705) 用 DinoV2 (https://arxiv.org/abs/2304.07193) 作为 visual backbone，50000 epochs，batch 50，3× A4000 GPU。Action chunk size 64 (Close-Drawer) 或 128 (其他)。

ACT 作为 specialist baseline（per-task 单独训练），与 EgoVLA 的 generalist setting 对比。EgoVLA 在 long-horizon 上碾压 ACT (45.93 vs 2.22)，因为 ACT 需要从 scratch 学习 multi-stage planning，而 EgoVLA pretraining 已经提供 skill prior。

---

## 11. 与 Related Work 的 Positioning

### 11.1 vs 其他 VLA

- **OpenVLA** (https://openvla.github.io/): 在 Open X-Embodiment (https://robotics-transformer-x.github.io/) 上训练，需要大规模 robot data
- **Octo** (https://octo-models.github.io/): 同样依赖 Open X-Embodiment
- **RT-2** (https://robotics-transformer2.github.io/): 用 large VLM (PaLI-X) + robot data co-fine-tuning
- **π0** (https://www.physicalintelligence.company/blog/pi0): VLA flow model，用大规模 robot data
- **EgoVLA**: 用 human egocentric video 替代 robot data 作为 primary source

### 11.2 vs Video Pretraining for Robotics

- **R3M** (https://arxiv.org/abs/2203.12601): 用 Ego4D pretrain visual representation
- **MVP** (https://arxiv.org/abs/2212.05349): MAE pretraining with video
- **VPT** (https://arxiv.org/abs/2210.03893): Minecraft video pretraining with implicit action
- **LAPA** (https://arxiv.org/abs/2410.11758): Latent action pretraining

这些工作主要 pretrain representation，EgoVLA 直接 pretrain policy（with explicit action prediction）。

### 11.3 vs Egocentric Learning

- **EgoMimic** (https://projectegl.github.io/EgoMimic/): 同样从 egocentric video，但做 task-specific
- **H2O / DexPilot** (https://arxiv.org/abs/2109.06721): hand pose 估计 + retargeting

---

## 12. Limitations 与 Future Direction

Paper 自己承认的：
1. 需要 hand/wrist pose annotation，限制 data 可用性。但 Quest 3、Vision Pro、Aria Glasses 等 AR/VR 设备普及会缓解
2. Zero-shot deployment 失败，仍需 moderate robot data fine-tune

我推测的潜在 future direction：
- **更多 embodiment**: 不只 humanoid，还可以是 gripper-based arm（通过更复杂 retargeting）
- **Latent action pretraining**: 如果 pose annotation 仍稀缺，可以学 latent action space (类似 LAPA)
- **Hierarchical VLA**: EgoVLA 学 low-level skill，加上 high-level planner 做 long-horizon reasoning
- **Diffusion policy head**: 把 transformer action head 换成 diffusion policy (https://arxiv.org/abs/2303.04137) 可能提升 multi-modal action distribution
- **Real-world deployment**: 当前只在 simulation 评估，real robot 上是否 work 仍待验证

---

## 13. Final Thoughts: 这篇 paper 的真正贡献

EgoVLA 的深刻意义在于：它证明了 **manipulation skill 是 embodiment-agnostic 的**，可以脱离具体 robot hardware 来学习。这与 LLM 中 "knowledge 与 reasoning 是 modality-agnostic" 的 spirit 一致。

如果这个方向继续 scale，未来 robot learning 的 data 瓶颈可能完全消失：80 亿人每天产生的 egocentric video 就是无限的 manipulation supervision signal。

一些可能的延伸联想：
- 把 EgoVLA 扩展到 mobile manipulation（human 走动 + 操作）
- 结合 first-person view 与 third-person view 的 Ego-Exo4D (https://arxiv.org/abs/2404.04989)
- 用神经渲染 (Gaussian Splatting / NeRF) 把 egocentric video 扩展成更多视角，提升 robustness
- 把 VLA 与 world model (DreamerV3, https://arxiv.org/abs/2301.04104) 结合，做 model-based planning

这些方向都值得深入探索。

---

## Reference Links

- EgoVLA project: https://rchalyang.github.io/EgoVLA/
- NVILA: https://arxiv.org/abs/2412.04468
- MANO: https://arxiv.org/abs/1711.01324
- rot6D: https://arxiv.org/abs/1812.07035
- ALOHA / ACT: https://arxiv.org/abs/2304.13705
- OpenVLA: https://arxiv.org/abs/2406.09246
- Octo: https://arxiv.org/abs/2405.12213
- π0: https://www.physicalintelligence.company/blog/pi0
- RT-2: https://arxiv.org/abs/2307.15818
- EgoMimic: https://arxiv.org/abs/2410.24221
- HOI4D: https://arxiv.org/abs/2111.06077
- HoloAssist: https://arxiv.org/abs/2310.01962
- HOT3D: https://arxiv.org/abs/2406.09598
- TACO: https://arxiv.org/abs/2401.08399
- OpenTelevision: https://arxiv.org/abs/2407.01512
- Isaac Lab: https://arxiv.org/abs/2308.12960
- LIBERO: https://arxiv.org/abs/2306.03310
- SIMPLER: https://arxiv.org/abs/2405.05941
- DinoV2: https://arxiv.org/abs/2304.07193
- R3M: https://arxiv.org/abs/2203.12601
- LAPA: https://arxiv.org/abs/2410.11758
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Ego-Exo4D: https://arxiv.org/abs/2404.04989
- DreamerV3: https://arxiv.org/abs/2301.04104
- Humanoid Policy: https://arxiv.org/abs/2503.13441
