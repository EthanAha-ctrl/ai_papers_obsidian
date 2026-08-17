---
source_pdf: OmniVLA Physically-Grounded Multimodal VLA with Unified.pdf
paper_sha256: 5b073f4871a2210e109358f2e0834b69be04d9acce8c22ec39b4eb3508a483a0
processed_at: '2026-08-05T23:32:10-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 OmniVLA

## 一句话版本

把 sensor 信号"贴"到 RGB 图上 task-relevant 的区域，让一个只懂 RGB 的 VLA 模型"免费"获得 thermal/mmWave/acoustic 感知能力。

---

## 为什么需要这东西

现在的 VLA 模型（π₀、SmolVLA、OpenVLA）只看 RGB 摄像头。但有一类任务 RGB 根本搞不定：

- "给我那杯**冷的**饮料"——RGB 看不出温度
- "开里面**有东西**的那个盒子"——RGB 看不穿纸盒
- "把响着的手机从衣服堆里**找出来**"——RGB 看不到衣服底下

这些任务需要 thermal camera、mmWave radar、microphone array 这些"超能力" sensor。问题是：怎么把这些 sensor 塞进一个 RGB-pretrained 的 VLA 里？

---

## 直接把 sensor 数据喂进去会怎样

直觉做法是给每个 sensor 训一个 encoder，跟 RGB encoder 并联。但有几个麻烦：

1. **VLA 的 vision encoder 认识 RGB 的统计**——它是在 web-scale image-text 上 pretrain 的。你给它一张 mmWave 的 beamforming heatmap，它看到的是一个完全 OOD 的 input，学习效率很低。实验里 VLA-RAW 只有 56% success rate，就是栽在这。
2. **每种 sensor 长得都不一样**——thermal 是 raster，mmWave 是 1D 复数 array，mic array 也是 1D 复数 array。给每种 sensor 训一个专用 encoder 不可扩展。
3. **Sensor 数据稀缺**——image-text pair 有 LAION 这种 web-scale corpus，mmWave+action 的 demonstration 只能自己采集，100-200 episodes 已经很吃力。

---

## OmniVLA 的核心 insight

作者从人脑借鉴了一个 insight：**人天然把其他感官 anchor 到视觉上**。你听到声音会下意识地把声源 location 投影到当前 visual scene 中，看到 thermal image 会自动把温度和 image 中的物体对应起来。

OmniVLA 就是把这种 cross-modal grounding **explicit 化**，让模型不用再学 "sensor 在 RGB 哪里" 这个 mapping。

---

## 怎么做的：三步走

### Step 1: 把所有 sensor 变成 image-like 的 2D heatmap

Thermal camera 天然输出 raster image，已经 OK。

mmWave radar 和 microphone array 输出的是每根 antenna/mic 的复数采样 $x_{i,k} = A_{i,k} e^{j\psi_{i,k}}$。要 transform 到 angular domain 才能像 image 一样被理解。这里用了经典的 **delay-and-sum beamforming**：

$$
\mathrm{I}_i(\theta,\phi) = 20\log_{10}\left\|\sum_{k=1}^{K} A_{i,k} e^{j\psi_{i,k}} e^{-j\Phi_{i,k}}\right\|
$$

$$
\Phi_{i,k} = \frac{2\pi}{\lambda_i}\big(x_{i,k}\cos\phi\sin\theta + y_{i,k}\sin\phi\big)
$$

变量解释：
- $i \in \{\text{mmWave, acoustic}\}$：sensor 索引
- $k \in \{1, \ldots, K\}$：array 中第 $k$ 个 element（antenna 或 microphone）的索引
- $A_{i,k}$：第 $k$ 个 element 收到的 signal amplitude（实数）
- $\psi_{i,k}$：第 $k$ 个 element 收到的 signal phase（弧度）
- $\theta$：azimuth angle（水平方位角）
- $\phi$：elevation angle（垂直俯仰角）
- $\lambda_i$：sensor $i$ 工作波长（mmWave 几 mm 量级，acoustic 几 cm 量级）
- $(x_{i,k}, y_{i,k})$：第 $k$ 个 element 在 array 平面上的 2D position
- $\Phi_{i,k}$：从 $(\theta, \phi)$ 方向入射的 wave 到达第 $k$ 个 element 相对于 array 中心的 phase delay

**Intuition**：steering phase $e^{-j\Phi_{i,k}}$ 是补偿从 $(\theta, \phi)$ 方向到第 $k$ 个 element 的 path difference。如果信号真的来自这个方向，所有 element 补偿后 phase 对齐，复数相加是 coherent 的，amplitude 大；如果不是，相加 incoherent，amplitude 小。$20\log_{10}\|\cdot\|$ 把 magnitude 转 dB scale，得到一张 azimuth-elevation heatmap。本质上这就是 phased array 信号到 angular domain 的 FFT 等价物，输出长得像一张 2D image。

参考资料：
- Delay-and-sum beamforming 综述：https://arxiv.org/abs/1705.05816
- mmWave radar imaging for robotics：https://dl.acm.org/doi/10.1145/3485730.3485926

### Step 2: 在 RGB 上找 task-relevant 的 mask

用 VLM + SAM2 在 RGB 上切出 task-relevant 物体的 mask：

$$
l = \mathrm{VLM}(\mathrm{T}_{\text{task}}, \mathrm{I}_{\text{RGB}})
$$

$$
\text{mask} = \mathrm{SAM2}(l, \mathrm{I}_{\text{RGB}})
$$

- $\mathrm{T}_{\text{task}}$：task 的 natural language 描述，比如 "Give me the cold drink"
- $\mathrm{I}_{\text{RGB}}$：当前帧 RGB image
- $l$：VLM (GPT-4o) 生成的 segmentation prompt，例如 "red drink", "cardboard boxes", "black phone"
- $\text{mask}$：0/1 matrix，1 表示 task-relevant 物体 region

具体实现是 **Grounded SAM 2**（SAM 2 + Grounding DINO）：
- Grounded SAM 2 repo: https://github.com/IDEA-Research/Grounded-SAM-2
- SAM 2 paper: https://arxiv.org/abs/2408.00714
- Grounding DINO paper: https://arxiv.org/abs/2303.05499

**关键工程 trick**：VLM 调用慢且贵，作者只在 task 开始时调用一次，之后异步 low-frequency update。robot 的 action loop 跑在本地 RTX 4090，15 Hz。VLM 的延迟不进入 critical path。这个异步设计让整个系统在 real-time 内闭环。

### Step 3: 在 mask 区域 overlay sensor heatmap

$$
\mathrm{I}_i^c = \mathrm{Calibration}(\mathrm{I}_i)
$$

$$
\mathrm{I}_i^m = \text{mask} \odot \big(\alpha \mathrm{I}_i^c + (1-\alpha) \mathrm{I}_{\text{RGB}}\big) + (1-\text{mask}) \odot \mathrm{I}_{\text{RGB}}
$$

- $i \in \{\text{mmWave, acoustic, thermal}\}$
- $\mathrm{I}_i^c$：经过一次性 rotation+crop calibration 的 sensor heatmap，和 RGB frame 大致对齐
- $\odot$：Hadamard product，element-wise multiplication
- $\alpha \in [0,1]$：blending 系数，论文默认 $\alpha=1$
- $\mathrm{I}_i^m$：最终 sensor-masked image

$\alpha=1$ 时公式退化成：

$$
\mathrm{I}_i^m = \text{mask} \odot \mathrm{I}_i^c + (1-\text{mask}) \odot \mathrm{I}_{\text{RGB}}
$$

也就是说：**mask 内部完全用 sensor heatmap 替换 RGB pixel，mask 外部保留 RGB pixel**。最终图像统计上仍然 mostly RGB（只有少数 task-relevant region 被 sensor 替换），这正是为什么 frozen RGB vision encoder 还能 work 的关键。

**Calibration 不需要精确到 pixel level**，作者明确说："high-precision pixel matching is not strictly required for the model to learn effectively." 这其实有点 surprising——因为 VLA backbone 对 spatial misalignment 敏感才对。我猜这里能容忍误差的原因是：mask 本身是从 RGB 上分割出来的，所以 mask 的形状是 RGB-aligned 的，sensor heatmap 在 mask 内的"位置"其实没那么重要，重要的是 sensor heatmap 的 intensity pattern 进入到了正确的 mask 区域。模型学到的是 "mask 区域里这张 pattern 表示 sensor 信号强度"，而非 "sensor heatmap 的 pixel (50,50) 对应 RGB 的 pixel (50,50)"。

这个设计可以类比 **prompt as spatial attention prior**：mask 告诉模型 "看这里"，sensor heatmap 告诉模型 "这里有什么物理信号"。和 SAE-style sparse coding 的 intuition 也有点像——把信号集中到 task-relevant 的 basis 上。

---

## 模型架构

### 整体数据流

$$
\mathbf{t}_i = \mathrm{MLP}_i\big(\mathrm{E}_I(\mathrm{I}_i^m)\big)
$$

$$
\mathbf{t}_{\text{task}} = \mathrm{E}_L(\mathrm{T}_{\text{task}})
$$

$$
\text{action} = \mathrm{VLA}\big([\mathbf{t}_1, \mathbf{t}_2, \ldots, \mathbf{t}_m, \mathbf{t}_{\text{task}}]\big)
$$

- $m$：sensor 数量
- $\mathrm{E}_I$：frozen RGB image encoder (SmolVLA 自带的 vision tower，基于 DINO/SigLIP 之类)
- $\mathrm{E}_L$：language embedding layer
- $\mathrm{MLP}_i$：第 $i$ 个 sensor 的 lightweight projector，把 vision encoder output 对齐到 language token space
- $\mathbf{t}_i$：第 $i$ 个 sensor 的 image token embedding
- $\mathbf{t}_{\text{task}}$：task description 的 language token embedding
- $\text{action}$：robot action chunk

**关键设计选择**：所有 sensor 共享同一个 vision encoder $\mathrm{E}_I$！因为 sensor-masked image 已经是 RGB-like，所以同一个 encoder 可以处理。每个 sensor 只需要一个独立的小 MLP，参数量极少。这跟以前 multi-modal fusion 要给每种 modality 训一个大 encoder 完全不同。

Token sequence 是 `[t_thermal, t_mmwave, t_acoustic, t_task]` 这种 concat 形式，送进 LLM backbone。LLM 输出再送进 diffusion/flow-matching based Action Expert，生成 action chunk。

### 训练策略三个 trick

1. **Vision encoder frozen**：保留 RGB pretraining 的 vision statistics 知识，省显存
2. **每个 sensor 的 MLP 用 base model 的 RGB projection layer 初始化**：这是一个非常重要的 prior。相当于告诉 sensor MLP："起步时把 sensor-masked image 当作 RGB 处理"。这个 init 等价于一个 warm start，让 sensor MLP 不需要从头学 "image feature → language token space" 的 mapping
3. **其他 weights (LLM + action expert) 全 trainable**

对比常见做法：很多 multi-modal VLA 给新 modality 从头训一个 encoder，需要大量数据。OmniVLA 用 sensor-masked image 这个表征绕开了这个问题，只需学一个轻量 MLP projector。

---

## 为什么这个 trick work

考虑两个极端情况：

**极端 1：mask = 全 1，整个图都被 sensor 覆盖**。RGB 的 spatial context 全丢失，模型必须从零学 "哪里是物体"，回到 VLA-RAW 的困境。

**极端 2：mask = 全 0，没有 sensor 信息**。退化到 VLA-RGB，只能 25%。

mask 的作用是 **spatial localization prior**：告诉模型 "sensor 信息只在这些 region 内有意义"。这相当于做了一个 strong 的 spatial attention supervision，把 sensor cue 锚定到 RGB semantic region 上。

更深一层：sensor heatmap 本身是 angular domain 的，和 RGB 的 pixel domain 有 projection 关系但不完全 aligned。如果直接让模型学这个 mapping，需要很多数据。Mask 把这个学习问题简化成 "在 RGB 给定的 region 内，sensor heatmap pattern 是什么意思"，问题大大简化。

---

## 实验数据

### 主表

| | Thermal | mmWave | Acoustic | Avg Success | Avg Score |
|---|---|---|---|---|---|
| VLA-RGB | 28% | 8% | 40% | 25% | 0.55 |
| VLA-RAW | 52% | 68% | 48% | 56% | 0.73 |
| OmniVLA | 80% | 84% | 88% | **84%** | **0.90** |

几个关键观察：

- **VLA-RGB 在 mmWave task 上只有 8% success**——这接近 random chance，因为 RGB 根本看不到盒子里的东西，模型完全在猜。这个数字其实是个 sanity check：它证明了这个 task 确实需要 non-RGB perception，而不是 RGB 也勉强能搞。
- **VLA-RAW vs VLA-RGB**: 从 25% → 56%，确实有 raw sensor 信息有效，但 raw sensor heatmap 喂进 frozen vision encoder 效率不高
- **OmniVLA vs VLA-RAW**: 84% vs 56%，绝对提升 28 个百分点。这个 28 个百分点的来源就是 sensor-masked image 这个 representation 的功劳，因为 backbone、训练数据、sensor input 都一样，唯一区别是 input 经过 segmentation+overlay 处理

### Base model 对比

| Base Model | Thermal | mmWave | Acoustic | Avg |
|---|---|---|---|---|
| SmolVLA | 80% | 84% | 88% | 84% |
| π₀ | 68% | 60% | 64% | 64% |

作者归因于 SmolVLA pretrain 数据用了 lerobot dataset，更接近 SO-101 embodiment。π₀ 是 Physical Intelligence 的，pretrain 数据主要是 Franka 等其他 arm。这印证了一个 VLA 领域的常识：**embodiment match 在 pretraining 数据里很重要**。论文这里其实就是想说 OmniVLA approach 对不同 backbone 都适用。

π₀ paper: https://arxiv.org/abs/2410.24164
SmolVLA paper: https://arxiv.org/abs/2506.01844

### Data efficiency

OmniVLA 用 50% 的训练数据（thermal task）能达到 VLA-RAW 用 100% 数据的 success rate。这个对比清晰地说明：**sensor-masked image 让模型 sample-efficient**。因为模型不用再学 "sensor 数据长什么样、怎么分布、怎么和 RGB 对应"，直接拿到 spatially-grounded 的 sensor cue。

### Generalization: 两个互补 effect

这个实验设计得很巧妙。把每个 task 分成两个 stage：
- **Stage 1**：选对要 interact 的物体（"哪个盒子有东西"、"哪个杯子是冷的"、"哪堆衣服下有手机"）
- **Stage 2**：完成 manipulation（"开盒子"、"拿杯子放容器"、"掀开衣服"）

| Model | Thermal S1/S2 | mmWave S1/S2 | Acoustic S1/S2 |
|---|---|---|---|
| OmniVLA-Base (no pretraining) | 100% / 24% | 56% / 40% | 76% / 16% |
| Pretrained VLA-RAW | 76% / 84% | 52% / 76% | 60% / 92% |
| Pretrained OmniVLA | 100% / 92% | 92% / 80% | 92% / 92% |

这个 stage decomposition 给出两个非常 informative 的结论：

- **Sensor-masked image 主要帮助 Stage 1（selection）**：Pretrained OmniVLA 在 Stage 1 全部 ≥92%，而 Pretrained VLA-RAW 在 Stage 1 只有 52-76%。原因是 sensor-masked image 把 sensor cue 限制在 task-relevant mask 上，模型立刻就知道 "哪个区域有信号 = 哪个物体是 target"。
- **Pretraining 主要帮助 Stage 2（manipulation）**：OmniVLA-Base 没有 multi-sensor pretraining，Stage 2 只有 16-40%。Pretrained 版本 Stage 2 提升到 80-92%。原因是 manipulation skill（开盒子、抓物体、掀盖子）是 transferable 的，可以在多个 task 上 share。

这两个 effect 是 **complementary** 的：sensor-masked image 解决 "where to look"，pretraining 解决 "how to act"。这跟 RL 中的 exploration vs exploitation 二分有点像，也对应 robot learning 里 "perception bottleneck" 和 "control bottleneck" 的经典二分。

---

## Intuition Building 与延伸思考

### 跟其他 multi-modal fusion 范式的关系

- **PointVLA / 3D-VLA / SpatialVLA** (https://arxiv.org/abs/2503.07511, https://arxiv.org/abs/2501.15830)：这些是把 depth/3D 注入 VLA。Depth 天然和 RGB pixel-aligned，所以融合简单。OmniVLA 处理的是 angular-domain sensor（mmWave/acoustic），需要 beamforming + mask 才能对齐。
- **VLA-Touch / TactileVLA** (https://arxiv.org/abs/2507.17294)：tactile sensor 在 gripper 上，是 contact-based，没有 spatial correspondence 问题，融合方式完全不同。
- **MultiPLY** (https://arxiv.org/abs/2407.07574)：是 embodied LLM，多感官，但只在 simulator 里，不能 output action。OmniVLA 是 real robot + real sensor + real action。
- **BEVFusion** (https://arxiv.org/abs/2205.13542)：autonomous driving 的 BEV 多 sensor fusion，但 BEV 是 object detection，不输出 action。OmniVLA 输出 action。
- **RFusion / FuseBot** (https://dl.acm.org/doi/10.1145/3485730.3485926, https://roboticsconference.org/2022/program/papers/59/)：MIT Fadel Adib 组的工作，用 RF 找 NLOS 物体然后 grasp。这些是 task-specific pipeline，没有 generalist + instruction following 能力。

### 跟脑科学的联系（作者自己提到的）

Cross-modal plasticity：盲人的 visual cortex 会 process tactile/auditory 信息（通过 Braille 阅读等）。OmniVLA 的 frozen RGB vision encoder 处理 sensor-masked image，本质上让 RGB-trained encoder "reuse" 给非 RGB modality 用，类比 cross-modal plasticity。Neuroscience 综述参考：https://www.nature.com/articles/nn1301

这种类比也提示了一个未来方向：如果 vision encoder 不 frozen，让它在 sensor-masked image 上继续 train，是不是会出现类似 cross-modal plasticity 的 emergent phenomenon？目前作者 frozen 是为了 data efficiency，但如果有更多 data，unfrozen 可能 unlock 更多能力。

### Sensor-masked image 作为 "spatial prompt"

Mask 在 NLP/Vision 领域有两个常见用法：
1. **Attention mask**：屏蔽某些 token
2. **Segmentation mask**：标记 region

OmniVLA 的 mask 同时起了两个作用：
- 对 RGB encoder：相当于 "这里有个 object" 的 spatial prompt
- 对 sensor heatmap：相当于 "sensor 信息只在这个 region 有意义" 的 spatial gating

这种用法和 SAM 自己的 prompt-based segmentation 也有点像：SAM 接受 point/box/mask prompt 来引导分割。OmniVLA 用 mask 来引导 sensor fusion。

### 局限性与潜在问题

- **依赖 SAM2 + VLM**：mask 质量受 segmentation 准确性影响。如果 SAM2 失败（比如透明物体、小物体），整个 fusion 会失败
- **Calibration 误差**：sensor heatmap 和 RGB 的 spatial 对齐如果误差太大，sensor 信息可能溢出到错误 mask 外
- **Mask 更新频率**：异步低频更新，如果场景快速变化（物体被移动），mask 可能 stale
- **VLM cost**：GPT-4o 调用每次 task 一次，长 task 会有 cost
- **Sensor 类型有限**：只验证了 IR + mmWave + acoustic，没验证 tactile, gas, EMG 等
- **Pretraining 数据需要包含 sensor task**：generalization 实验里 pretraining 用了 600 episodes 的 sensor task data，所以不是 zero-shot。这个 setting 实际上是 few-shot + multi-task pretraining，不是真 zero-shot transfer

### 联想到的相关方向

- **Sensor foundation model**：目前 sensor 各自处理。如果有一个 unified sensor encoder pretrained 在 multi-sensor data 上，可以替代 per-sensor MLP
- **Self-supervised sensor pretraining**：beamforming heatmap + mask 作为 pretext task，pretrain 一个 sensor encoder
- **Online mask refinement**：让模型自己 refine mask，不依赖 SAM2，类似 SAM2 的 video mode 但用 sensor cue refine
- **3D scene grounding**：现在 sensor heatmap 是 2D angular，能否 lift 到 3D（mmWave radar 本身有 range 信息）
- **Active sensing**：robot 主动移动 sensor 来获取更多信息，类似 active perception
- **Multi-sensor cross-modal contrastive learning**：thermal/mmWave/acoustic 对同一 scene 的不同观察，可以做 contrastive learning 来 align representation
- **Embodied AI simulator + sensor**：当前 sim 主要支持 RGB/depth，加入 thermal/mmWave/acoustic 的 sim 支持会让大规模 sensor pretraining 可行
- **Habitat-Matterport 的 sensor 扩展**：https://aihabitat.org/ 已经支持 multi-sensor，可以加入 RF 类

### 工程上的一些小观察

- 8x A100 GPU, 14 hours for 50K steps, batch 32。这个训练 cost 对学术 lab 算 reasonable
- RTX 4090 local inference, 15 Hz prediction rate。这个对闭环 control 够用，但比 SmolVLA 原版 50 Hz 还是慢
- VLM async call 让 latency 不进 critical path，但首帧还是有 VLM delay
- 三种 sensor 在硬件上集成得相当紧凑（Fig 4b），看起来是个 small module

### 关于 paper 本身的 meta 观察

- 第一作者 Heyu Guo 来自 Princeton，合作者有 Yasaman Ghasempour（mmWave/RF 方向 expert）和 Omid Abari（wireless/systems）。所以 sensor 端有 wireless 方向 expert 加持，RF 部分很扎实
- 第三作者 Ruichun Ma 等 in Microsoft Research Asia，Baining Guo 和 Lili Qiu 都是 systems/networks veteran。所以这是 Princeton wireless + UCLA systems + MSRA 的合作
- Lili Qiu 之前做 mobile sensing/systems，转向 embodied AI 的一个 path
- 项目开源在 https://github.com/GuoHeyu/OmniVLA

### Take-away intuition

如果只让我用三句话总结这篇 paper：

1. **Sensor-masked image 是 spatially-grounded 的 sensor fusion 表征**：把 sensor heatmap 用 semantic mask 限制到 task-relevant RGB region 上
2. **共享 frozen RGB vision encoder + per-sensor lightweight MLP**：用 RGB pretrain 的 statistical prior 给 sensor-free lunch
3. **Two-stage decomposition 揭示互补 effect**：mask-based representation 帮 Stage 1 (selection)，multi-sensor pretraining 帮 Stage 2 (manipulation)

更抽象地讲，OmniVLA 解决的是 "如何把物理信号（temperature, RF reflection, sound pressure）inject 到 foundation model 中" 这个 general 问题。Sensor-masked image 是一个 elegant 的中间表征，把物理信号 spatially-ground 到 RGB pixel 上，让 pretrained vision encoder 可以直接处理。这个 pattern 应该可以推广到更多 sensor 类型和更多 task domain。

进一步延伸到 robot learning 的更大图景：当前 VLA 主要基于 visual grounding，但 robot 真正需要的是 **physically-grounded** 的 grounding——它需要知道温度、重量、texture、声音、电磁场等。OmniVLA 是这个方向的一个早期 data point，未来会看到更多 sensor 加入 VLA 的 work，比如 IMU+VLA (proprioception 是已有但通常和 vision 一起做), gas sensor+VLA (嗅觉机器人), EMG+VLA (人形机器人的人机交互), 甚至 quantum sensor+VLA (磁场感知)。这个方向我相信会成为一个重要的 research thread。

### 一些可能的 follow-up 项目

1. **Online mask + sensor refinement loop**: 当前 mask 是 SAM2 一次性给定的。如果 mask 可以基于 sensor heatmap 反馈 refine（比如 thermal mask 可以基于 temperature threshold refine），可能更鲁棒
2. **Sensor pretraining corpus**: 目前每个 task 100-200 episodes 太少。可以做一个 multi-task sensor pretraining dataset，让 community 一起贡献
3. **Cross-sensor transfer**: 用 thermal pretrain 的 model 能不能 zero-shot 用 mmWave？因为 sensor-masked image 是统一表征，理论上 cross-sensor transfer 应该比 cross-task transfer 容易
4. **Sensor masking as test-time intervention**: 在 inference 时可以 prompt model "用 thermal 看这里"，类似 visual prompt engineering
5. **Sensor-masked image for VLM (非 VLA)**：把 sensor-masked image 给 GPT-4o 等 VLM 做问答，可能解锁 sensor-aware VLM
6. **Active perception**: robot 主动决定看哪里 / 用什么 sensor 看，结合 active vision 和 active sensing
7. **Sensor fusion as a graph**: 当 sensor 数量增加，可以做成 graph neural network，每个 sensor 是一个 node，mask 是 edge
8. **Foundation sensor encoder**: pretrain 一个 unified sensor encoder 在 multi-sensor data 上，替代 per-sensor MLP

### 一个有趣的哲学问题

OmniVLA 让我想到一个更深层的问题：**机器人的 "perception" 应该多 close to 人类的 perception？** 人类只有 5 种感官（视觉、听觉、触觉、嗅觉、味觉），但 robot 可以有无数种 sensor（mmWave、Lidar、thermal、gas、EMG、quantum magnetometer...）。这些 sensor 给 robot 的 perception capability 远超 human。OmniVLA 的 sensor-masked image 范式让 VLA foundation model 可以接入任意 sensor，相当于让 foundation model 拥有 "超人类" 的 perception。这跟 AR/VR 让人获得 "超人类" visual experience 类似，是 "超人类 perception" 的 embodied AI 版本。

这个方向的 ethical/safety implication 也很值得思考：一个能看到温度、能穿透盒子、能听声定位的 robot 跟一个只有 RGB 的 robot 在 capability 上有本质不同，safety framework 需要重新设计。

---

## 总结性 intuition

OmniVLA 把 "如何让 VLA 理解非 RGB sensor" 这个看似复杂的 multi-modal fusion 问题，简化成 "如何把 sensor 信息 image-native 地 embed 到 RGB 上"。Sensor-masked image 通过三个步骤达成：(1) beamforming 把 sensor 信号变 image-like；(2) SAM2 在 RGB 上找 task-relevant mask；(3) 在 mask 区域 overlay sensor heatmap。结果是一个统计上 mostly-RGB 但 spatially-grounded 了 sensor cue 的 image，可以喂给任何 RGB-pretrained VLA backbone。配合 per-sensor lightweight MLP + multi-sensor pretraining，在 100-200 episodes 量级的数据上就能让 model 学会需要非视觉感知的 manipulation task。

这个工作的核心贡献其实是一个 **representation engineering**：找到一个 unified interface 让 heterogeneous sensor 和 RGB foundation model 兼容。这种 "中间表征" 思路在 ML 历史上反复出现（CNN 是 translation invariance 的 inductive bias，transformer 是 set permutation invariance 的 inductive bias，sensor-masked image 是 spatial grounding 的 inductive bias）。它不一定是最优的，但它是 elegant 且 effective 的。

相关 web link 汇总：
- OmniVLA GitHub: https://github.com/GuoHeyu/OmniVLA
- SmolVLA: https://arxiv.org/abs/2506.01844
- π₀: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- Grounded SAM 2: https://github.com/IDEA-Research/Grounded-SAM-2
- SAM 2: https://arxiv.org/abs/2408.00714
- Grounding DINO: https://arxiv.org/abs/2303.05499
- MultiPLY: https://arxiv.org/abs/2407.07574
- BEVFusion: https://arxiv.org/abs/2205.13542
- RFusion: https://dl.acm.org/doi/10.1145/3485730.3485926
- FuseBot: https://roboticsconference.org/2022/program/papers/59/
- lerobot: https://github.com/huggingface/lerobot
- SO-101: https://github.com/TheRobotStudio/SO-ARM100
- PointVLA: https://arxiv.org/abs/2503.07511
- SpatialVLA: https://arxiv.org/abs/2501.15830
- VLA-Touch: https://arxiv.org/abs/2507.17294
- TactileVLA: https://arxiv.org/abs/2507.09160
- 3D-VLA: https://arxiv.org/abs/2403.09631
- Cross-modal plasticity review: https://www.nature.com/articles/nn1301
- Delay-and-sum beamforming: https://arxiv.org/abs/1705.05816
- mmWave imaging: https://arxiv.org/abs/2010.07947
- EmbodiedScan: https://arxiv.org/abs/2402.14864
- Multi-modal fusion survey: https://arxiv.org/abs/2504.02477

---

# OmniVLA：Beyond-RGB 的 VLA 模型

这篇 paper 的核心 thesis 其实挺干净的：现有的 VLA 模型（OpenVLA, π₀, SmolVLA）都被锁死在 RGB 输入上，但机器人要完成的真实任务里有一大类（"找冷的饮料"、"开有东西的盒子"、"掀开盖着响手机的衣服"）单凭 RGB 根本无法决策。OmniVLA 把 infrared / mmWave radar / microphone array 三类异构 sensor，**通过一个叫 "sensor-masked image" 的中间表征**，统一塞回到 RGB 像素坐标系里，让一个 RGB-pretrained 的 VLA backbone 几乎免费地获得了 multi-modal 感知能力。

我把整篇 paper 拆成几块来讲，重点放在每个设计选择背后的 intuition 上。

---

## 1. Motivation 与挑战拆解

作者把问题概括成三个 challenge：

1. **Heterogeneous sensor 难直接喂给 VLA**。SmolVLA/π₀ 的 vision encoder 是在 web-scale RGB image-text 上 pretrain 的，统计上是 RGB。直接把 beamforming heatmap 或 thermal image 当第二路 input 喂进去，模型相当于看到一个 out-of-distribution 的 channel，学习效率很低（实验里 VLA-RAW 只有 56% 平均 success rate 印证了这点）。
2. **Sensor 硬件异构**。Infrared 是 2D raster；mmWave radar 是 1D complex array；mic array 是 1D complex array。FOV、resolution、coordinate frame 都不同。给每种 sensor 训一个专用 encoder 不可扩展。
3. **Sensor modality 数据稀缺**。不像 image-text pair 有 LAION 这种 web-scale corpus，mmWave+action 的 demonstration episode 只能自己采集，100-200 episodes 已经很吃力。

作者从人类大脑借鉴了一个 insight：**人天然把其他感官 anchor 到视觉上**——我们听到声音会下意识地把声源 location 投影到当前 visual scene 中，看到 thermal image 会自动把温度和 image 中的物体对应起来。所以 sensor-masked image 本质是把这种 cross-modal grounding explicit 化，让模型不用再学 "sensor 在 RGB 哪里" 这个 mapping。

---

## 2. Sensor-masked image：核心 representation

这是整篇 paper 的灵魂。生成 pipeline 三步：

### 2.1 Preprocessing: 把 sensor 变成 image-like 2D 表征

Thermal camera 天然输出 raster image，已经 OK。mmWave radar 和 microphone array 输出的是每根 antenna/mic 的复数采样序列 $x_{i,k} = A_{i,k} e^{j\psi_{i,k}}$，要 transform 到 angular domain 才能像 image 一样被理解。这里用了经典 **delay-and-sum beamforming**，公式 (1)：

$$
\mathrm{I}_i(\theta,\phi) = 20\log_{10}\left\|\sum_{k=1}^{K} A_{i,k} e^{j\psi_{i,k}} e^{-j\Phi_{i,k}}\right\|
$$

$$
\Phi_{i,k} = \frac{2\pi}{\lambda_i}\big(x_{i,k}\cos\phi\sin\theta + y_{i,k}\sin\phi\big)
$$

变量逐项解释：

- $i \in \{\text{mmWave, acoustic}\}$：sensor 索引
- $k \in \{1, \ldots, K\}$：array 中第 $k$ 个 element（antenna 或 microphone）的索引
- $A_{i,k}$：第 $k$ 个 element 收到的 signal amplitude（实数）
- $\psi_{i,k}$：第 $k$ 个 element 收到的 signal phase（弧度）
- $\theta$：azimuth angle（方位角，水平方向）
- $\phi$：elevation angle（俯仰角，垂直方向）
- $\lambda_i$：sensor $i$ 工作波长（mmWave 几 mm 量级，acoustic 几 cm 量级）
- $(x_{i,k}, y_{i,k})$：第 $k$ 个 element 在 array 平面上的 2D position
- $\Phi_{i,k}$：从 $(\theta, \phi)$ 方向入射的 wave 到达第 $k$ 个 element 相对于 array 中心的 phase delay

**Intuition**：steering phase $e^{-j\Phi_{i,k}}$ 是补偿从 $(\theta, \phi)$ 方向到第 $k$ 个 element 的 path difference。如果信号真的来自这个方向，所有 element 补偿后 phase 对齐，复数相加是 coherent 的，amplitude 大；如果不是，相加 incoherent，amplitude 小。$20\log_{10}\|\cdot\|$ 把 magnitude 转 dB scale，得到一张 azimuth-elevation heatmap。

本质上这就是 phased array 信号到 angular domain 的 FFT 等价物，输出长得像一张 2D image，可以直接当 image 用。

参考资料：
- Delay-and-sum beamforming 综述：https://arxiv.org/abs/1705.05816
- mmWave radar imaging for robotics：https://dl.acm.org/doi/10.1145/3485730.3485926

### 2.2 Segmentation：用 VLM + SAM2 找 task-relevant region

公式 (2)：

$$
l = \mathrm{VLM}(\mathrm{T}_{\text{task}}, \mathrm{I}_{\text{RGB}})
$$

$$
\text{mask} = \mathrm{SAM2}(l, \mathrm{I}_{\text{RGB}})
$$

- $\mathrm{T}_{\text{task}}$：task 的 natural language 描述，比如 "Give me the cold drink"
- $\mathrm{I}_{\text{RGB}}$：当前帧 RGB image
- $l$：VLM (GPT-4o) 生成的 segmentation prompt，例如 "red drink", "cardboard boxes", "black phone"
- $\text{mask}$：0/1 matrix，1 表示 task-relevant 物体 region

具体实现是 **Grounded SAM 2**（SAM 2 + Grounding DINO）：
- Grounded SAM 2 repo: https://github.com/IDEA-Research/Grounded-SAM-2
- SAM 2 paper: https://arxiv.org/abs/2408.00714
- Grounding DINO paper: https://arxiv.org/abs/2303.05499

**关键工程 trick**：VLM 调用慢且贵，作者只在 task 开始时调用一次，之后异步 low-frequency update。robot 的 action loop 跑在本地 RTX 4090，跑 15 Hz。VLM 的延迟不进入 critical path。这个异步设计让整个系统在 real-time 内闭环。

### 2.3 Overlay：把 sensor 信息 spatially 塞到 RGB 上

公式 (3)：

$$
\mathrm{I}_i^c = \mathrm{Calibration}(\mathrm{I}_i)
$$

$$
\mathrm{I}_i^m = \text{mask} \odot \big(\alpha \mathrm{I}_i^c + (1-\alpha) \mathrm{I}_{\text{RGB}}\big) + (1-\text{mask}) \odot \mathrm{I}_{\text{RGB}}
$$

- $i \in \{\text{mmWave, acoustic, thermal}\}$
- $\mathrm{I}_i^c$：经过一次性 rotation+crop calibration 的 sensor heatmap，和 RGB frame 大致对齐
- $\odot$：Hadamard product，element-wise multiplication
- $\alpha \in [0,1]$：blending 系数，论文默认 $\alpha=1$
- $\mathrm{I}_i^m$：最终 sensor-masked image

$\alpha=1$ 时，公式退化成：

$$
\mathrm{I}_i^m = \text{mask} \odot \mathrm{I}_i^c + (1-\text{mask}) \odot \mathrm{I}_{\text{RGB}}
$$

也就是说：**mask 内部完全用 sensor heatmap 替换 RGB pixel，mask 外部保留 RGB pixel**。最终图像统计上仍然 mostly RGB（只有少数 task-relevant region 被 sensor 替换），这正是为什么 frozen RGB vision encoder 还能 work 的关键。

**Calibration 不需要精确到 pixel level**，作者明确说："high-precision pixel matching is not strictly required for the model to learn effectively." 这其实有点 surprising——因为 VLA backbone 对 spatial misalignment 敏感才对。我猜这里能容忍误差的原因是：mask 本身是从 RGB 上分割出来的，所以 mask 的形状是 RGB-aligned 的，sensor heatmap 在 mask 内的"位置"其实没那么重要，重要的是 sensor heatmap 的 intensity pattern 进入到了正确的 mask 区域。模型学到的是 "mask 区域里这张 pattern 表示 sensor 信号强度"，而非 "sensor heatmap 的 pixel (50,50) 对应 RGB 的 pixel (50,50)"。

这个设计可以类比 **prompt as spatial attention prior**：mask 告诉模型 "看这里"，sensor heatmap 告诉模型 "这里有什么物理信号"。和 SAE-style sparse coding 的 intuition 也有点像——把信号集中到 task-relevant 的 basis 上。

---

## 3. 模型架构

### 3.1 整体数据流

公式 (4)：

$$
\mathbf{t}_i = \mathrm{MLP}_i\big(\mathrm{E}_I(\mathrm{I}_i^m)\big)
$$

$$
\mathbf{t}_{\text{task}} = \mathrm{E}_L(\mathrm{T}_{\text{task}})
$$

$$
\text{action} = \mathrm{VLA}\big([\mathbf{t}_1, \mathbf{t}_2, \ldots, \mathbf{t}_m, \mathbf{t}_{\text{task}}]\big)
$$

- $m$：sensor 数量
- $\mathrm{E}_I$：frozen RGB image encoder (SmolVLA 自带的 vision tower，基于 DINO/SigLIP 之类)
- $\mathrm{E}_L$：language embedding layer
- $\mathrm{MLP}_i$：第 $i$ 个 sensor 的 lightweight projector，把 vision encoder output 对齐到 language token space
- $\mathbf{t}_i$：第 $i$ 个 sensor 的 image token embedding
- $\mathbf{t}_{\text{task}}$：task description 的 language token embedding
- $\text{action}$：robot action chunk

**关键设计选择**：所有 sensor 共享同一个 vision encoder $\mathrm{E}_I$！因为 sensor-masked image 已经是 RGB-like，所以同一个 encoder 可以处理。每个 sensor 只需要一个独立的小 MLP，参数量极少。这跟以前 multi-modal fusion 要给每种 modality 训一个大 encoder 完全不同。

Token sequence 是 `[t_thermal, t_mmwave, t_acoustic, t_task]` 这种 concat 形式，送进 LLM backbone。LLM 输出再送进 diffusion/flow-matching based Action Expert，生成 action chunk。

### 3.2 训练策略

三个 trick 让 data-efficient 学习成立：

1. **Vision encoder frozen**：保留 RGB pretraining 的 vision statistics 知识，省显存
2. **每个 sensor 的 MLP 用 base model 的 RGB projection layer 初始化**：这是一个非常重要的 prior。相当于告诉 sensor MLP："起步时把 sensor-masked image 当作 RGB 处理"。这个 init 等价于一个 warm start，让 sensor MLP 不需要从头学 "image feature → language token space" 的 mapping
3. **其他 weights (LLM + action expert) 全 trainable**

对比常见做法：很多 multi-modal VLA 给新 modality 从头训一个 encoder，需要大量数据。OmniVLA 用 sensor-masked image 这个表征绕开了这个问题，只需学一个轻量 MLP projector。

---

## 4. 硬件平台

- **Arm**: SO-101 manipulator（lerobot project 的开源机械臂）
- **RGB cameras**: top-down + front + arm-mounted
- **Sensor suite**: depth camera + IR thermal camera + mmWave radar + 6-microphone circular array

lerobot 项目主页：https://github.com/huggingface/lerobot
SmolVLA: https://arxiv.org/abs/2506.01844
SO-101 arm: https://github.com/TheRobotStudio/SO-ARM100

---

## 5. 实验数据详解

### 5.1 主表 (Table I)

| | Thermal | mmWave | Acoustic | Avg Success | Avg Score |
|---|---|---|---|---|---|
| VLA-RGB | 28% | 8% | 40% | 25% | 0.55 |
| VLA-RAW | 52% | 68% | 48% | 56% | 0.73 |
| OmniVLA | 80% | 84% | 88% | **84%** | **0.90** |

几个关键观察：

- **VLA-RGB 在 mmWave task 上只有 8% success**——这接近 random chance，因为 RGB 根本看不到盒子里的东西，模型完全在猜。这个数字其实是个 sanity check：它证明了这个 task 确实需要 non-RGB perception，而不是 RGB 也勉强能搞。
- **VLA-RAW vs VLA-RGB**: 从 25% → 56%，确实有 raw sensor 信息有效，但 raw sensor heatmap 喂进 frozen vision encoder 效率不高
- **OmniVLA vs VLA-RAW**: 84% vs 56%，绝对提升 28 个百分点。这个 28 个百分点的来源就是 sensor-masked image 这个 representation 的功劳，因为 backbone、训练数据、sensor input 都一样，唯一区别是 input 经过 segmentation+overlay 处理

### 5.2 Base model 对比 (Table II)

| Base Model | Thermal | mmWave | Acoustic | Avg |
|---|---|---|---|---|
| SmolVLA | 80% | 84% | 88% | 84% |
| π₀ | 68% | 60% | 64% | 64% |

作者归因于 SmolVLA pretrain 数据用了 lerobot dataset，更接近 SO-101 embodiment。π₀ 是 Physical Intelligence 的，pretrain 数据主要是 Franka 等其他 arm。这印证了一个 VLA 领域的常识：**embodiment match 在 pretraining 数据里很重要**。论文这里其实就是想说 OmniVLA approach 对不同 backbone 都适用。

π₀ paper: https://arxiv.org/abs/2410.24164
SmolVLA paper: https://arxiv.org/abs/2506.01844

### 5.3 Data efficiency (Figure 6)

OmniVLA 用 50% 的训练数据（thermal task）能达到 VLA-RAW 用 100% 数据的 success rate。这个对比清晰地说明：**sensor-masked image 让模型 sample-efficient**。因为模型不用再学 "sensor 数据长什么样、怎么分布、怎么和 RGB 对应"，直接拿到 spatially-grounded 的 sensor cue。

### 5.4 Generalization (Table III)

这个实验设计得很巧妙。把每个 task 分成两个 stage：
- **Stage 1**：选对要 interact 的物体（"哪个盒子有东西"、"哪个杯子是冷的"、"哪堆衣服下有手机"）
- **Stage 2**：完成 manipulation（"开盒子"、"拿杯子放容器"、"掀开衣服"）

| Model | Thermal S1/S2 | mmWave S1/S2 | Acoustic S1/S2 |
|---|---|---|---|
| OmniVLA-Base (no pretraining) | 100% / 24% | 56% / 40% | 76% / 16% |
| Pretrained VLA-RAW | 76% / 84% | 52% / 76% | 60% / 92% |
| Pretrained OmniVLA | 100% / 92% | 92% / 80% | 92% / 92% |

这个 stage decomposition 给出两个非常 informative 的结论：

- **Sensor-masked image 主要帮助 Stage 1（selection）**：Pretrained OmniVLA 在 Stage 1 全部 ≥92%，而 Pretrained VLA-RAW 在 Stage 1 只有 52-76%。原因是 sensor-masked image 把 sensor cue 限制在 task-relevant mask 上，模型立刻就知道 "哪个区域有信号 = 哪个物体是 target"。
- **Pretraining 主要帮助 Stage 2（manipulation）**：OmniVLA-Base 没有 multi-sensor pretraining，Stage 2 只有 16-40%。Pretrained 版本 Stage 2 提升到 80-92%。原因是 manipulation skill（开盒子、抓物体、掀盖子）是 transferable 的，可以在多个 task 上 share。

这两个 effect 是 **complementary** 的：sensor-masked image 解决 "where to look"，pretraining 解决 "how to act"。这跟 RL 中的 exploration vs exploitation 二分有点像，也对应 robot learning 里 "perception bottleneck" 和 "control bottleneck" 的经典二分。

---

## 6. Intuition Building 与延伸思考

### 6.1 为什么 mask 是关键？没有 mask 会怎样？

考虑极端情况：**直接把 sensor heatmap 整张 overlay 到 RGB 上（mask=全 1）**。这会让整张图都被 sensor 信息覆盖，RGB 的 spatial context 全丢失。模型必须从零学 "哪里是物体"，回到 VLA-RAW 的困境。

考虑另一种极端：**用全 0 mask，等于没有 sensor 信息**。退化到 VLA-RGB，只能 25%。

mask 的作用是 **spatial localization prior**：告诉模型 "sensor 信息只在这些 region 内有意义"。这相当于做了一个 strong 的 spatial attention supervision，把 sensor cue 锚定到 RGB semantic region 上。

更深一层：sensor heatmap 本身是 angular domain 的，和 RGB 的 pixel domain 有 projection 关系但不完全 aligned。如果直接让模型学这个 mapping，需要很多数据。Mask 把这个学习问题简化成 "在 RGB 给定的 region 内，sensor heatmap pattern 是什么意思"，问题大大简化。

### 6.2 跟其他 multi-modal fusion 范式的关系

- **PointVLA / 3D-VLA / SpatialVLA** (https://arxiv.org/abs/2503.07511, https://arxiv.org/abs/2501.15830)：这些是把 depth/3D 注入 VLA。Depth 天然和 RGB pixel-aligned，所以融合简单。OmniVLA 处理的是 angular-domain sensor（mmWave/acoustic），需要 beamforming + mask 才能对齐。
- **VLA-Touch / TactileVLA** (https://arxiv.org/abs/2507.17294)：tactile sensor 在 gripper 上，是 contact-based，没有 spatial correspondence 问题，融合方式完全不同。
- **MultiPLY** (https://arxiv.org/abs/2407.07574)：是 embodied LLM，多感官，但只在 simulator 里，不能 output action。OmniVLA 是 real robot + real sensor + real action。
- **BEVFusion** (https://arxiv.org/abs/2205.13542)：autonomous driving 的 BEV 多 sensor fusion，但 BEV 是 object detection，不输出 action。OmniVLA 输出 action。
- **RFusion / FuseBot** (https://dl.acm.org/doi/10.1145/3485730.3485926, https://roboticsconference.org/2022/program/papers/59/)：MIT Fadel Adib 组的工作，用 RF 找 NLOS 物体然后 grasp。这些是 task-specific pipeline，没有 generalist + instruction following 能力。

### 6.3 跟脑科学的联系（作者自己提到的）

Cross-modal plasticity：盲人的 visual cortex 会 process tactile/auditory 信息（通过 Braille 阅读等）。OmniVLA 的 frozen RGB vision encoder 处理 sensor-masked image，本质上让 RGB-trained encoder "reuse" 给非 RGB modality 用，类比 cross-modal plasticity。Neuroscience 综述参考：https://www.nature.com/articles/nn1301

这种类比也提示了一个未来方向：如果 vision encoder 不 frozen，让它在 sensor-masked image 上继续 train，是不是会出现类似 cross-modal plasticity 的 emergent phenomenon？目前作者 frozen 是为了 data efficiency，但如果有更多 data，unfrozen 可能 unlock 更多能力。

### 6.4 Sensor-masked image 作为 "spatial prompt"

Mask 在 NLP/Vision 领域有两个常见用法：
1. **Attention mask**：屏蔽某些 token
2. **Segmentation mask**：标记 region

OmniVLA 的 mask 同时起了两个作用：
- 对 RGB encoder：相当于 "这里有个 object" 的 spatial prompt
- 对 sensor heatmap：相当于 "sensor 信息只在这个 region 有意义" 的 spatial gating

这种用法和 SAM 自己的 prompt-based segmentation 也有点像：SAM 接受 point/box/mask prompt 来引导分割。OmniVLA 用 mask 来引导 sensor fusion。

### 6.5 局限性与潜在问题

- **依赖 SAM2 + VLM**：mask 质量受 segmentation 准确性影响。如果 SAM2 失败（比如透明物体、小物体），整个 fusion 会失败
- **Calibration 误差**：sensor heatmap 和 RGB 的 spatial 对齐如果误差太大，sensor 信息可能溢出到错误 mask 外
- **Mask 更新频率**：异步低频更新，如果场景快速变化（物体被移动），mask 可能 stale
- **VLM cost**：GPT-4o 调用每次 task 一次，长 task 会有 cost
- **Sensor 类型有限**：只验证了 IR + mmWave + acoustic，没验证 tactile, gas, EMG 等
- **Pretraining 数据需要包含 sensor task**：generalization 实验里 pretraining 用了 600 episodes 的 sensor task data，所以不是 zero-shot。这个 setting 实际上是 few-shot + multi-task pretraining，不是真 zero-shot transfer

### 6.6 联想到的相关方向

- **Sensor foundation model**：目前 sensor 各自处理。如果有一个 unified sensor encoder pretrained 在 multi-sensor data 上，可以替代 per-sensor MLP
- **Self-supervised sensor pretraining**：beamforming heatmap + mask 作为 pretext task，pretrain 一个 sensor encoder
- **Online mask refinement**：让模型自己 refine mask，不依赖 SAM2，类似 SAM2 的 video mode 但用 sensor cue refine
- **3D scene grounding**：现在 sensor heatmap 是 2D angular，能否 lift 到 3D（mmWave radar 本身有 range 信息）
- **Active sensing**：robot 主动移动 sensor 来获取更多信息，类似 active perception
- **Multi-sensor cross-modal contrastive learning**：thermal/mmWave/acoustic 对同一 scene 的不同观察，可以做 contrastive learning 来 align representation
- **Embodied AI simulator + sensor**：当前 sim 主要支持 RGB/depth，加入 thermal/mmWave/acoustic 的 sim 支持会让大规模 sensor pretraining 可行
- **Habitat-Matterport 的 sensor 扩展**：https://aihabitat.org/ 已经支持 multi-sensor，可以加入 RF 类

### 6.7 工程上的一些小观察

- 8x A100 GPU, 14 hours for 50K steps, batch 32。这个训练 cost 对学术 lab 算 reasonable
- RTX 4090 local inference, 15 Hz prediction rate。这个对闭环 control 够用，但比 SmolVLA 原版 50 Hz 还是慢
- VLM async call 让 latency 不进 critical path，但首帧还是有 VLM delay
- 三种 sensor 在硬件上集成得相当紧凑（Fig 4b），看起来是个 small module

### 6.8 关于 paper 本身的 meta 观察

- 第一作者 Heyu Guo 来自 Princeton，合作者有 Yasaman Ghasempour（mmWave/RF 方向 expert）和 Omid Abari（wireless/systems）。所以 sensor 端有 wireless 方向 expert 加持，RF 部分很扎实
- 第三作者 Ruichun Ma 等 in Microsoft Research Asia，Baining Guo 和 Lili Qiu 都是 systems/networks veteran。所以这是 Princeton wireless + UCLA systems + MSRA 的合作
- Lili Qiu 之前做 mobile sensing/systems，转向 embodied AI 的一个 path
- 项目开源在 https://github.com/GuoHeyu/OmniVLA

### 6.9 Take-away intuition

如果只让我用三句话总结这篇 paper：

1. **Sensor-masked image 是 spatially-grounded 的 sensor fusion 表征**：把 sensor heatmap 用 semantic mask 限制到 task-relevant RGB region 上
2. **共享 frozen RGB vision encoder + per-sensor lightweight MLP**：用 RGB pretrain 的 statistical prior 给 sensor-free lunch
3. **Two-stage decomposition 揭示互补 effect**：mask-based representation 帮 Stage 1 (selection)，multi-sensor pretraining 帮 Stage 2 (manipulation)

更抽象地讲，OmniVLA 解决的是 "如何把物理信号（temperature, RF reflection, sound pressure）inject 到 foundation model 中" 这个 general 问题。Sensor-masked image 是一个 elegant 的中间表征，把物理信号 spatially-ground 到 RGB pixel 上，让 pretrained vision encoder 可以直接处理。这个 pattern 应该可以推广到更多 sensor 类型和更多 task domain。

进一步延伸到 robot learning 的更大图景：当前 VLA 主要基于 visual grounding，但 robot 真正需要的是 **physically-grounded** 的 grounding——它需要知道温度、重量、texture、声音、电磁场等。OmniVLA 是这个方向的一个早期 data point，未来会看到更多 sensor 加入 VLA 的 work，比如 IMU+VLA ( proprioception 是已有但通常和 vision 一起做), gas sensor+VLA (嗅觉机器人), EMG+VLA (人形机器人的人机交互), 甚至 quantum sensor+VLA (磁场感知)。这个方向我相信会成为一个重要的 research thread。

### 6.10 一些可能的 follow-up 项目

1. **Online mask + sensor refinement loop**: 当前 mask 是 SAM2 一次性给定的。如果 mask 可以基于 sensor heatmap 反馈 refine（比如 thermal mask 可以基于 temperature threshold refine），可能更鲁棒
2. **Sensor pretraining corpus**: 目前每个 task 100-200 episodes 太少。可以做一个 multi-task sensor pretraining dataset，让 community 一起贡献
3. **Cross-sensor transfer**: 用 thermal pretrain 的 model 能不能 zero-shot 用 mmWave？因为 sensor-masked image 是统一表征，理论上 cross-sensor transfer 应该比 cross-task transfer 容易
4. **Sensor masking as test-time intervention**: 在 inference 时可以 prompt model "用 thermal 看这里"，类似 visual prompt engineering
5. **Sensor-masked image for VLM (非 VLA)**：把 sensor-masked image 给 GPT-4o 等 VLM 做问答，可能解锁 sensor-aware VLM
6. **Active perception**: robot 主动决定看哪里 / 用什么 sensor 看，结合 active vision 和 active sensing
7. **Sensor fusion as a graph**: 当 sensor 数量增加，可以做成 graph neural network，每个 sensor 是一个 node，mask 是 edge
8. **Foundation sensor encoder**: pretrain 一个 unified sensor encoder 在 multi-sensor data 上，替代 per-sensor MLP

### 6.11 一个有趣的哲学问题

OmniVLA 让我想到一个更深层的问题：**机器人的 "perception" 应该多 close to 人类的 perception？** 人类只有 5 种感官（视觉、听觉、触觉、嗅觉、味觉），但 robot 可以有无数种 sensor（mmWave、Lidar、thermal、gas、EMG、quantum magnetometer...）。这些 sensor 给 robot 的 perception capability 远超 human。OmniVLA 的 sensor-masked image 范式让 VLA foundation model 可以接入任意 sensor，相当于让 foundation model 拥有 "超人类" 的 perception。这跟 AR/VR 让人获得 "超人类" visual experience 类似，是 "超人类 perception" 的 embodied AI 版本。

这个方向的 ethical/safety implication 也很值得思考：一个能看到温度、能穿透盒子、能听声定位的 robot 跟一个只有 RGB 的 robot 在 capability 上有本质不同，safety framework 需要重新设计。

---

## 7. 总结性 intuition

OmniVLA 把 "如何让 VLA 理解非 RGB sensor" 这个看似复杂的 multi-modal fusion 问题，简化成 "如何把 sensor 信息 image-native 地 embed 到 RGB 上"。Sensor-masked image 通过三个步骤达成：(1) beamforming 把 sensor 信号变 image-like；(2) SAM2 在 RGB 上找 task-relevant mask；(3) 在 mask 区域 overlay sensor heatmap。结果是一个统计上 mostly-RGB 但 spatially-grounded 了 sensor cue 的 image，可以喂给任何 RGB-pretrained VLA backbone。配合 per-sensor lightweight MLP + multi-sensor pretraining，在 100-200 episodes 量级的数据上就能让 model 学会需要非视觉感知的 manipulation task。

这个工作的核心贡献其实是一个 **representation engineering**：找到一个 unified interface 让 heterogeneous sensor 和 RGB foundation model 兼容。这种 "中间表征" 思路在 ML 历史上反复出现（CNN 是 translation invariance 的 inductive bias，transformer 是 set permutation invariance 的 inductive bias，sensor-masked image 是 spatial grounding 的 inductive bias）。它不一定是最优的，但它是 elegant 且 effective 的。

相关 web link 汇总：
- OmniVLA GitHub: https://github.com/GuoHeyu/OmniVLA
- SmolVLA: https://arxiv.org/abs/2506.01844
- π₀: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- Grounded SAM 2: https://github.com/IDEA-Research/Grounded-SAM-2
- SAM 2: https://arxiv.org/abs/2408.00714
- Grounding DINO: https://arxiv.org/abs/2303.05499
- MultiPLY: https://arxiv.org/abs/2407.07574
- BEVFusion: https://arxiv.org/abs/2205.13542
- RFusion: https://dl.acm.org/doi/10.1145/3485730.3485926
- FuseBot: https://roboticsconference.org/2022/program/papers/59/
- lerobot: https://github.com/huggingface/lerobot
- SO-101: https://github.com/TheRobotStudio/SO-ARM100
- PointVLA: https://arxiv.org/abs/2503.07511
- SpatialVLA: https://arxiv.org/abs/2501.15830
- VLA-Touch: https://arxiv.org/abs/2507.17294
- TactileVLA: https://arxiv.org/abs/2507.09160
- 3D-VLA: https://arxiv.org/abs/2403.09631
- Cross-modal plasticity review: https://www.nature.com/articles/nn1301
- Delay-and-sum beamforming: https://arxiv.org/abs/1705.05816
- mmWave imaging: https://arxiv.org/abs/2010.07947
- EmbodiedScan: https://arxiv.org/abs/2402.14864
- Multi-modal fusion survey: https://arxiv.org/abs/2504.02477
