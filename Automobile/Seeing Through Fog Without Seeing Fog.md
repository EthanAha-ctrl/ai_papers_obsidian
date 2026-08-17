---
source_pdf: Seeing Through Fog Without Seeing Fog.pdf
paper_sha256: ff2b466190e313b096ebe930433de5370833bbc30e6024652704c7f0df940973
processed_at: '2026-08-12T04:31:41-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 Paper

## 一句话版本

开车的时候大雾天 sensor 会"各自拉胯"而且拉胯的方式还不一样，现有 fusion 方法假设 sensor 之间互相兜底，结果一个拉了全部跟着拉。这篇 paper 说：让 network 自己看每个 sensor 当前 output 的"信息量"，信息量低的自动降权，信息量高的自动顶上，训练时只见过晴天也能在雾天 work。

---

## 问题到底是什么

想象你开一辆 autonomous car，车顶有 4 只眼睛：

- **RGB camera**: 白天看得很清楚，晚上瞎，雾天灰蒙蒙
- **Lidar**: 激光打出去测距，晴天无敌，雾天激光被雾滴弹回来，20 米外全是白墙
- **Gated camera**: 主动打 NIR laser 但带 time-gated shutter，物理上把近距离 backscatter 挡掉，雾天反而还能看
- **Radar**: 77GHz mmWave，波长比雾滴大太多，雾对它基本透明，但 spatial resolution 很差

晴天时 4 个 sensor 都好，fusion 方法学到 "4 个 sensor 平均一下就行"。雾天来了，lidar 突然变成一堵噪声墙。如果 fusion 方法还是傻乎乎平均，lidar 的垃圾就把别人的好信号污染了。paper Table 2 里有个特别 striking 的数字：**dense fog 下 lidar-only SSD 的 AP 从 73.46 暴跌到 28.98**，AVOD-FPN 这种 SOTA fusion 方法跟着崩到 **33.95**，比 image-only 的 **87.89** 差了一大截。这就是 paper 想解决的问题：**fusion 不如 single sensor**，redundancy assumption 反而帮倒忙。

---

## 为什么现有方法不行

### Late fusion (AVOD, Frustum PointNet)

逻辑是每个 sensor 先独立跑 detector 出 proposal，最后投票。问题是每个 sensor branch 不知道自己现在是不是瞎了，lidar branch 在雾天依然 confidence 满满地输出一堆 backscatter 假目标，最后投票就被它污染。

### Early fusion (Concat SSD)

逻辑是把所有 sensor input 拼一起喂一个 backbone，让 backbone 自己学谁可信。问题晴天训练时所有 sensor 都可信，backbone 学到的最优策略是 "平均权重"，没 incentive 学 "动态降权"。inference 遇到雾，backbone 还是平均，lidar 垃圾 channel 照样进 feature map。

### Domain adaptation (ADDA, CycADA)

理论上可以用 GAN 把晴天图变成雾天图做训练数据。问题是 GAN 只能做 image-domain style transfer，**它建模不了 lidar point cloud 被雾 scatter 成什么样**，也建模不了 gated camera 的 time-gating 物理效应。而且 domain adaptation 需要 target domain unlabeled data，跟 "edge case 永远 rare" 这个前提就矛盾。

---

## Paper 的核心 trick

### Trick 1: 把所有 sensor 投到 image plane

existing fusion 方法要么用 BeV (bird's-eye view) 投影 lidar，要么用 raw point cloud。这两种 representation 和 RGB camera feature 对不齐，所以只能 late fusion。

paper 的做法是**把 lidar, radar, gated 全部 project 到 RGB camera 的 image plane**，pixel-wise 对齐：
- Lidar: 每个 pixel 存 (depth, height, intensity) 三个 channel
- Radar: radar 本来是 2D scan (range-azimuth)，paper 沿 vertical axis 直接 replicate 成 image-like map
- Gated: homography 投影到 RGB plane

这样 4 个 sensor 的 feature map 在每个 spatial location 都对应同一根 ray，**早期就能做 feature exchange**。这是整个 architecture 能 work 的前提。

### Trick 2: Entropy 作为 reliability proxy

paper 不去 classify "现在是 fog 还是 night 还是 snow"，而是直接计算每个 sensor 当前 output 的 local entropy：

$$\rho = \sum_{m,n} \sum_{i=0}^{v} p_i^{mn} \log(p_i^{mn})$$

$$p_i^{mn} = \frac{1}{MN} \sum_{j=1}^{M} \sum_{k=1}^{N} \delta(I(m+j, n+k) - i)$$

人话翻译：
- $I$: 某个 sensor 的 measurement (8-bit，值 0-255)
- $(m,n)$: 当前 pixel 位置
- $M \times N = 16 \times 16$: 以 $(m,n)$ 为中心的 patch
- $i$: pixel 值的 bin (0 到 255)
- $\delta$: 指示函数，pixel 值等于 $i$ 就是 1 否则 0
- $p_i^{mn}$: patch 里像素值等于 $i$ 的比例，就是 empirical probability
- $\rho$: 这个 patch 的 Shannon entropy

**直觉**：如果一块区域是 uniform 的 (天空、墙、fog backscatter)，pixel 值集中在少数几个 bin，entropy 低。如果一块区域有 object edge、texture、structured signal，pixel 值分散在很多 bin，entropy 高。

所以 **entropy 高 = 有 structured signal = sensor 在这块区域 reliable**，**entropy 低 = uniform noise 或 constant = sensor 在这块区域 garbage**。

paper Figure 5 给了 visual evidence：fog chamber 里随 fog 加浓，RGB 和 lidar 的 entropy 明显下降，gated 和 radar 的 entropy 基本不变。night 里 RGB entropy 下降，其他三个 active sensor 不变。这说明 **entropy 是一个 sensor-agnostic、weather-agnostic 的 reliability 指标**，不需要知道具体是什么 weather。

### Trick 3: Entropy-steered Feature Exchange

architecture 是 4 个 parallel VGG backbone，中间穿插 feature exchange blocks。每个 exchange block 的操作：

1. 4 个 stream 的 feature map $F_1, F_2, F_3, F_4$ concatenate 起来
2. 每个 stream 的 entropy map $E_k$ 喂进小 conv → sigmoid → 得到 weight map $W_k \in [0,1]^{H \times W}$
3. 加权融合：$\tilde{F}_k = W_k \odot (F_1 \oplus F_2 \oplus F_3 \oplus F_4)$
4. 再 concatenate 上 entropy map 喂下一层

**直觉**：当前如果 lidar 在某个 region entropy 低 (被 backscatter 淹了)，$W_{lidar}$ 在那个 region 接近 0，lidar 的 feature 被 attenuate，gated 和 radar 的 feature 自然 dominate。这个 weight map 是 **per-pixel、runtime-computed** 的，所以能 spatially adaptive —— 同一张图里可能左半边 lidar 好 (近距离没雾)，右半边 lidar 坏 (远距离雾很重)，weight map 也能区分。

### Trick 4: Train on Clean, Test on Adverse

这是最反直觉的部分。训练时：
- **只用晴天 data**
- **Random dropout**: 每个 sensor 以 50% 概率被 drop (input 和 entropy 都置零)

为什么能 generalize 到没见过的 fog？paper 的逻辑是：

1. Random dropout 等价于告诉 network "任何 sensor 随时可能失效"，所以 network 学到的策略是**通用的 "当某 sensor entropy 低就降权"**，不是 "见到 fog 就降权 lidar" 这种 weather-specific 规则。

2. 晴天 data 里也有 low entropy region (天空、uniform wall)。network 在晴天就学到了 "low entropy → low weight" 这个 mapping。fog 里的 backscatter region 恰好也是 low entropy，所以这个 mapping 直接 transfer。

3. Object appearance 在 fog 里没完全消失 (low-frequency shape 还在)，detector head 学到的 object prior 仍然适用。

这其实是 **implicit domain randomization**。不需要 simulate fog，因为 simulate 也不准 (fog 的物理模型很复杂，特别是 lidar scatter)。直接 random dropout 反而更 general，覆盖所有可能的 sensor failure mode。

---

## 实验数据读法

Table 2 里几个关键数字对比：

**Dense Fog Hard case (最 severe 场景)**:
- Proposed Deep Entropy Fusion: **76.69**
- Deep Fusion (无 entropy steering): 73.93 → entropy 贡献 **+2.76**
- Fusion SSD (late fusion): 63.23 → 比 proposed 低 **13.46**
- Image-only SSD: 74.96 → late fusion 比 single sensor 还差
- Lidar-only SSD: 24.56 → lidar 彻底废了
- AVOD-FPN: 26.17 → SOTA fusion 方法最差
- CycADA (用了 target domain data 的 unfair advantage): 73.38 → 仍然不如 proposed

**Light Fog Hard case**:
- Proposed: 84.90
- Image-only: 70.43
- Lidar-only: 51.91

这里 proposed 比 image-only 高 **14.47**，说明轻度 fog 下 gated/radar 的 redundant 信息被 entropy steering 正确利用了，不像 late fusion 那样被 lidar 拖累。

---

## 为什么这个思路重要

这篇 paper 对我启发最大的是几个点：

**1. Reliability proxy 可以是 unsupervised 的**

不用 label "这个 sensor 现在坏没坏"，用 entropy 这种 information-theoretic measure 就能自动 infer。这个思路可以推广到很多场景：
- Multi-camera 系统，某个 camera 被太阳直射过曝 → entropy 低 → 降权
- IMU + GPS fusion，GPS 进隧道丢信号 → entropy 低 (虽然 GPS 本身不输出 entropy，可以算 covariance)
- Robotics 里 sensor suite 异常检测

**2. Train on clean, generalize to corrupt 是可行的**

前提是 corruption 改变的是 sensor reliability，不是 object appearance 本身。如果 corruption 让 object 本身都看不见 (比如 RGB 在浓雾里完全白茫茫)，那 detector head 学到的 appearance prior 就失效了。但 paper 证明只要还有一个 sensor 能看到 object (gated 在雾里还能看)，fusion 就能 work。

**3. Asymmetric distortion 是 fusion 的真正 hard case**

symmetric distortion (所有 sensor 一起 degrade) 反而好处理，因为 detector confidence 整体下降，threshold 调一下就行。asymmetric distortion 才是 tricky 的，因为 network 会误以为 "还有 sensor 在 confident 输出，应该是真的 object"，结果那个 confident 输出是 garbage。这就需要 runtime adaptive weighting，不能是 training-time fixed weight。

**4. Domain randomization 不一定要在 input level**

OpenAI 那套 sim2real 是 randomize input texture、lighting。paper 这里 randomize 的是 **sensor availability**，是一种更高抽象的 randomization。只要 randomized 的维度和 test-time corruption 共享某个 invariant structure (这里是 entropy)，就能 generalize。

---

## 几个我想吐槽的点

**1. Entropy 是 hand-crafted feature**

16x16 patch 的 Shannon entropy 是人为选的。为什么不是 8x8 或 32x32？为什么不是 learned 的 reliability head？paper 没做这个 ablation。理论上可以让 network 自己 learn 一个 reliability estimator，但需要 failure label，而 failure label 又难定义。Entropy 是个 reasonable 的 unsupervised proxy，但肯定不是最优。

**2. Radar encoding 太粗糙**

直接沿 vertical axis replicate radar scan，这个 encoding 信息损失很大。Radar 其实有 Doppler 信息、有 elevation 信息 (虽然弱)，都没用上。现代 4D radar (比如 Continental ARS540) 有 elevation resolution，可以更精细 encode。

**3. Image-plane projection 丢失被遮挡 region 的 lidar point**

把 lidar 投到 image plane，被前景 object 遮挡的 background lidar point 就丢了。BeV 没这个问题。trade-off 是 image-plane 能 early fusion，BeV 只能 late fusion。paper 选了 early fusion 这一边，但没定量分析这个损失多大。

**4. Dense fog 只有 1k frames**

虽然 paper 喊着 10,000 km driving，但 dense fog labeled frame 只有 1k 左右。hard case 的统计意义可能不够，特别是 Table 2 里 dense fog hard 的数字方差可能很大。不过这是 adverse weather dataset 的通病，采集成本太高。

**5. 没和 weather-aware methods 比**

比如 [51] Sakaridis 的 synthetic fog augmentation，paper 提了一句 "does not affect the reported margins" 但没给详细数字。如果 synthetic fog training + entropy fusion 会不会更好？没 explore。

---

## 对 Andrej 可能有意思的几个角度

**1. 和你的 "Software 2.0" 框架的 connection**

这篇 paper 是典型的 Software 2.0 思路：把 sensor reliability 判断从 explicit rule (weather classifier) 换成 learned function (entropy → weight mapping)。但 entropy 本身是 Software 1.0 的 hand-crafted feature。真正纯 Software 2.0 应该让 network 直接从 raw input learn reliability，但那需要 supervision signal。这反映了一个 deeper 的张力：**unsupervised proxy (entropy) vs learned function (reliability head)** 的 trade-off。

**2. 和 "train on distribution, test out of distribution" 的 connection**

这篇是 OOD generalization 的 case study。当前 OOD generalization 理论 (distribution shift, covariance shift) 大多 assume input distribution shift。这里 shift 的是 **sensor reliability distribution**，更 subtle。network 能 generalize 是因为 entropy 这个 invariant 在两个 distribution 下都成立。这给 OOD generalization 提供了一个思路：**find invariant structure across distributions**。

**3. 和 multi-task learning 的 connection**

fusion 可以看成 multi-task：每个 sensor 是一个 "expert"，fusion 是 "mixture of experts"。MoE 里 gating network 通常 learn from data。这里 gating signal 是 entropy，是 explicit injected。这种 **explicit inductive bias** 在 data scarce 时可能比 learned gating 更 sample efficient。

**4. 后续可能的方向**

- **Learned reliability head**: 用 entropy 作为 self-supervised signal train 一个 reliability estimator，然后 inference 时不用 entropy 用 learned head
- **Sensor control feedback loop**: 根据 reliability 决定 sensor 参数 (lidar power, exposure time, gating delay)，闭环
- **Event camera 加入**: event camera 在 fog 里 behavior 怎样？entropy profile 怎样？
- **Weather-conditional architecture search**: 不同 weather 下最优 fusion architecture 可能不同，用 NAS 自动找

---

## Reference Links

- Paper PDF: https://arxiv.org/abs/1902.08951
- Code & Dataset: https://github.com/princeton-computational-imaging/SeeingThroughFog
- Felix Heide Lab: https://www.computationalimaging.org/
- DENSE EU Project: https://dense2020.eu/
- Gated2Depth (companion): https://www.computationalimaging.org/Publications/gated2depth/
- SSD Paper: https://arxiv.org/abs/1512.02325
- AVOD Paper: https://arxiv.org/abs/1805.03918
- Frustum PointNet: https://arxiv.org/abs/1711.08488
- KITTI Benchmark: http://www.cvlibs.net/datasets/kitti/
- Fog Chamber Setup (Duthon et al.): https://www.sciencedirect.com/science/article/pii/S2352146517306234

---

# Seeing Through Fog Without Seeing Fog - 深度讲解

## Paper Overview

这是 Felix Heide 团队 (Princeton) 和 Mercedes-Benz 合作的工作，发表于 2020 年 CVPR。核心 motivation 是 autonomous driving 中的 multimodal sensor fusion 在 **asymmetric distortion** 场景下的失效问题。paper 名字本身就是一个 paradox：**"Seeing Through Fog Without Seeing Fog"** —— 网络要在没有见过 fog training data 的情况下，inference 时能 robust 处理 fog。

paper 的官方代码和数据：https://github.com/princeton-computational-imaging/SeeingThroughFog

Felix Heide 的 lab page: https://www.computationalimaging.org/

---

## 1. The Core Problem: Asymmetric Sensor Distortion

绝大多数 existing fusion methods (AVOD, MV3D, Frustum PointNet, PointFusion) 都有一个 **隐含假设**：sensor streams 是 **redundant 且 consistent** 的 —— 同一个 object 在一个 sensor 里出现，在另一个 sensor 里也会出现。这个 assumption 在 clear weather 下大致成立，但在 adverse weather 下彻底崩塌。

paper 关键观察 (Figure 3 是核心 illustration)：在 fog 中，不同 sensor 的 degradation 是 **asymmetric** 的：

| Sensor | Clear | Fog (23m vis) | Night |
|--------|-------|---------------|-------|
| RGB camera | good | heavy backscatter + attenuation | noisy in low light |
| Lidar (pulsed 903nm) | good | **backscatter 限制到 <20m range** | unaffected (active illumination) |
| Gated NIR (808nm) | good | **suppresses backscatter via time-gating** | works (active) |
| Radar (77GHz mmWave) | low spatial res | **largely unaffected by fog** | works (active) |
| FIR (thermal) | works | moderate | works |

这意味着：在 fog 中，lidar 几乎废了 (Figure 3 第二行 lidar plot 基本是一堵 wall of backscatter)，但 gated camera 和 radar 仍然 robust。在 night 中，RGB 废了但 lidar/radar 仍好。

**关键 insight**：existing fusion 方法因为训练时 sensor 是 redundant 的，学到的 fusion weight 是 "average out" 的策略。一旦某个 sensor 在 inference 时 suddenly 变成 garbage，network 不知道该 down-weight 它，于是被 garbage 拖累。paper Table 2 里 lidar-only SSD 在 dense fog 下从 73.46 AP 暴跌到 28.98 AP (-44.48)，而 AVOD-FPN 这种 lidar-camera fusion 也跟着崩到 33.95 —— 比 image-only 还差。这说明 **naive fusion 在 asymmetric failure 下比 single sensor 更糟**，因为 redundant assumption 反而误导 network。

---

## 2. Dataset: Multimodal Adverse Weather

paper 的一大 contribution 是采集了真实 adverse weather 的 multimodal dataset。10,000 km 在北欧 (Sweden, Denmark, Finland, Germany) 冬季 driving，2 个月采集，1.4M frames @ 10Hz，最终 100k labels。

### Sensor Stack (Section 3.1)

- **Stereo RGB Camera**: On-Semi AR0230 RCCB (RCCB = Red-Clear-Clear-Bayer，去掉绿色滤色片提升 low-light sensitivity)，1920×1024，12-bit，30Hz，focal 8mm，FOV 39.6°×21.7°，baseline 20.3cm
- **Gated Camera**: BrightwayVision BrightEye，NIR 808nm，1280×720，10-bit，120Hz (实际系统采样 10Hz 多 slice)，FOV 31.1°×17.8°。Gated imaging 用 pulsed laser + time-synchronized shutter，可以 slice 不同 depth range，并 suppress backscatter。
- **Radar**: proprietary FMCW (Frequency-Modulated Continuous Wave) 77GHz，1° angular resolution，200m range，15Hz
- **Lidar**: Velodyne HDL64 S3D + VLP32C，903nm，dual return，10Hz，range 100m/120m
- **FIR (Far-Infrared)**: Axis Q1922 thermal camera，640×480，17μm pixel pitch，NETD < 100mK，30Hz
- **Environmental**: Airmar WX150 weather station + road friction sensor + IMU

### Weather Distribution (Figure 2)

真实 driving data 极度 unbalanced：
- Clear: 多数 frames
- Light fog (visibility 100m-1km): 1k frames
- Dense fog (visibility <100m): 1k frames  
- Snow/Rain: 4k frames
- Clear: 5.5k frames

总 13.5k labeled frames。dense fog 在北美 driving 中只占 0.01% (引自 [61] van Oldenborgh et al.)，即使在 foggy region 每年 dense fog <50m 也只发生 15 次。这就解释了为什么 paper 选择 **train on clean, test on adverse** 的 protocol —— 没法采集足够多 labeled adverse data 来 train。

### Fog Chamber Controlled Recordings

除了 real-world，还有 controlled fog chamber (见 [13, 17])，3 个 visibility level (30m, 40m, 50m) × day/night × 1.5k labeled frames。这个用来做 Figure 5 entropy analysis。

---

## 3. Method: Adaptive Deep Entropy Fusion

### 3.1 Data Representation —— 关键设计 decision

这是 paper 里我认为最 underappreciated 但最重要的 design choice 之一。existing fusion 方法分两类：
- **BeV (Bird's-eye View) projection** (AVOD, MV3D): lidar 投到 top-down 2D plane，camera 是 front view。两者 spatial representation 不一致，只能 late fusion (matching proposals)。
- **Raw point cloud** (VoxelNet, Frustum PointNet): 用 3D voxel 或 pointnet，根本无法和 2D camera feature pixel-wise 对齐。

paper 选择了 **third way**：把 lidar, radar, gated 全部 project 到 **RGB camera image plane**。

具体 input encoding:
- **Camera branch**: 3-plane RGB
- **Lidar branch**: 3-channel image = (depth, height, pulse intensity)，不是 naive depth-only
- **Radar branch**: radar scan 是 2D (range-azimuth)，paper 假设 radar 在 horizontal image dimension 对应，vertical 方向 invariant —— 直接 **replicate** scan 沿 vertical axis 拉伸成 image-like map
- **Gated**: 用 homography mapping 投到 RGB image plane
- **Missing measurement**: zero value 填充

为什么这个 design 关键？因为只有 **pixel-wise aligned** 的 representation 才能在早期 layer 做 feature exchange。BeV 和 image-plane 之间有一个根本 misalignment：image 中一个 pixel 对应 lidar 一个 ray，但 BeV 中一个 pixel 对应 lidar 一根 scan line 投下来的 footprint。早期 fusion 必须在 spatial alignment 的 feature 上做，否则交换 feature 没意义。

### 3.2 Architecture (Figure 4)

整体结构是 **4 个 parallel SSD (Single-Shot Detector) branches** + **deep feature exchange blocks** + **entropy steering**。

Backbone: modified VGG (cited [54] Simonyan & Zisserman VGG)，把 channel 数砍半，截断到 conv4。再从 conv4-10 抽 6 个 feature maps 作为 SSD detection layers (cited [38] FPN-style feature pyramid，[40] SSD)。这是为了 real-time 约束。

```
[RGB]   [Lidar]   [Gated]   [Radar]
  |        |         |        |
VGG-1    VGG-2     VGG-3    VGG-4   ← 4 个独立 backbone，参数不共享
  ↕        ↕         ↕        ↕
[Feature Exchange Block × N]      ← white blocks in Fig 4
  ↓                                  信息双向交换
  ↓ entropy steering (red blocks)
  ↓
SSD detection heads (×6 layers each branch)
```

**Feature Exchange Block** 的核心操作：
1. 把 4 个 stream 的 feature maps $F_1, F_2, F_3, F_4$ concatenate 起来
2. 对应每个 stream 计算一个 entropy map $E_k$
3. 把 $E_k$ 喂进一个小 conv → sigmoid → 得到 weight $W_k \in [0,1]^{H \times W}$
4. **Per-stream 加权**: $\tilde{F}_k = W_k \odot (F_1 \oplus F_2 \oplus F_3 \oplus F_4)$
5. 再 concatenate 上 entropy map 一起喂给下一层

直觉：entropy 高的区域表示该 sensor 信息丰富 → weight 接近 1，特征 pass through；entropy 低的区域 (e.g. lidar 被 fog 全是 backscatter) → weight 接近 0，特征被 attenuate。这样 garbage sensor 的 garbage feature 自动被 down-weight，good sensor 的 feature 自动 dominate。

### 3.3 Entropy Computation (核心公式)

paper Section 4.2 Eq. (1) 给了 local measurement entropy 的定义：

$$\rho = \sum_{m,n} \sum_{i=0}^{v} p_i^{mn} \log(p_i^{mn})$$

with

$$p_i^{mn} = \frac{1}{MN} \sum_{j=1}^{M} \sum_{k=1}^{N} \delta(I(m+j, n+k) - i)$$

我来逐项拆解：

- $I \in [0, 255]^{H \times W}$: 一个 8-bit binarized 单 channel measurement (例如 lidar 的 intensity plane 或 RGB 的某个 channel)
- $i$: 像素值的 bin index，从 0 到 $v=255$
- $(m, n)$: 当前 patch 的中心 pixel 位置
- $M \times N = 16 \times 16$ pixel: patch size
- $\delta(\cdot)$: Kronecker delta，$I(m+j, n+k) - i = 0$ 时返回 1，否则 0
- $p_i^{mn}$: 在以 $(m,n)$ 为中心的 patch 内，像素值等于 $i$ 的比例 (empirical probability)
- $\rho$: 该 patch 的 Shannon entropy (其实是 negative entropy，因为 $p \log p \leq 0$，但 paper 这样写)

所以最终得到一个 $w \times h = 1920 \times 1024 / 16 = 120 \times 64$ 的 entropy map (其实应该是 $H/16 \times W/16$)。然后 upsample 回 full resolution 喂给 fusion block。

**Why entropy as reliability proxy?** paper Section 4.2 解释：与其像 [57, 59] 那样 explicitly infer fog type/strength (这需要 labeled fog data，且泛化差)，不如用一个 sensor-agnostic 的 information-theoretic measure。Entropy 低意味着 measurement 是 constant 或 noise-only (fog backscatter 就是 low-frequency constant-ish noise)，entropy 高意味着有 structured signal (object edges, depth variation)。

Figure 5 给了非常 visual 的 evidence：
- Fog chamber 中 RGB 和 lidar entropy 随 visibility 下降而下降 (backscatter 把 signal 淹没)
- Gated 和 radar entropy 基本不变
- Night 中 RGB entropy 下降，其他三个 active sensor 不变

这就证明 entropy 是一个 **sensor-agnostic, weather-agnostic** 的 reliability proxy —— 不需要知道是 fog 还是 night 还是 snow，只看 measurement 本身的 information content。

### 3.4 Training Strategy —— "Seeing Without Seeing"

这是 paper 标题的来源。training 时：
- **只用 clear weather data**
- **不引入任何 adverse weather sample**
- **Random sensor dropout**: 每个 sensor stream 以 0.5 概率被 drop (input 置零)，entropy 置零
- 训练 4 个 SSD branches + fusion 联合 end-to-end

为什么 random dropout 等价于模拟 asymmetric distortion？因为 adverse weather 本质就是 "某些 sensor 突然变成 garbage"。通过 random dropout，network 被迫学到：
1. 任何一个 sensor 都可能突然失效
2. 必须依赖 entropy 来动态判断当前哪些 sensor 可信
3. 在多个 sensor 同时可用时利用 redundancy，在单 sensor 可用时也能 work

这是一种 **implicit data augmentation**，比 simulating fog (cited [51] Sakaridis synthetic fog) 更 general，因为 dropout 不预设 distortion type。

### 3.5 Loss Functions

Classification loss (Eq. 2): 标准 binary cross-entropy with softmax
$$H(p) = \sum_i \left( y_i \log(p_i) + (1-y_i) \log(1-p_i) \right)$$

- $y_i \in \{0, 1\}$: anchor $i$ 的 ground-truth class label (1 = object, 0 = background)
- $p_i$: network 预测 anchor $i$ 是 object 的 probability

Matching threshold 0.5 (IoU) 区分 positive/negative anchors。

Bounding box regression loss (Eq. 3): Huber loss (smooth L1)
$$H(x) = \begin{cases} x^2 / 2, & \text{if } |x| < 1 \\ |x| - 0.5, & \text{if } |x| \geq 1 \end{cases}$$

- $x$: predicted box coordinate offset relative to anchor
- $|x| < 1$ 时是 quadratic (smooth gradient near 0)，$|x| \geq 1$ 时是 linear (robust to outlier)

Hard negative mining [52]: negative anchors 限制为 positive 数量的 5 倍，防止 background 主导。

Training details:
- From scratch (no ImageNet pretrain)
- Constant learning rate
- L2 weight decay 0.0005

---

## 4. Results Analysis (Table 2)

我来仔细拆 Table 2，这是 paper 最重要的 quantitative evidence。

| Method | Clear (e/m/h) | Light Fog (e/m/h) | Dense Fog (e/m/h) | Snow/Rain (e/m/h) |
|--------|---------------|-------------------|--------------------|--------------------|
| **Deep Entropy Fusion** (proposed) | **89.84 / 85.57 / 79.46** | **90.54 / 87.99 / 84.90** | **87.68 / 81.49 / 76.69** | **88.99 / 83.71 / 77.85** |
| Deep Fusion (no entropy) | 90.07 / 80.31 / 77.82 | 90.60 / 81.08 / 79.63 | 86.77 / 77.28 / 73.93 | 89.25 / 79.09 / 70.51 |
| Fusion SSD (late) | 87.73 / 78.02 / 69.49 | 88.33 / 78.65 / 76.54 | 74.07 / 68.46 / 63.23 | 85.49 / 75.28 / 67.48 |
| Concat SSD (early naive) | 86.12 / 76.62 / 68.61 | 87.98 / 78.24 / 70.17 | 77.99 / 69.16 / 67.07 | 83.63 / 73.65 / 66.26 |
| ADDA [60] | 85.27 / 70.51 / 67.86 | 87.83 / 78.68 / 70.38 | 87.64 / 78.12 / 74.37 | 84.17 / 74.25 / 66.86 |
| CycADA [28] | 88.50 / 77.84 / 69.56 | 89.08 / 79.36 / 75.58 | 87.24 / 77.04 / 73.38 | 85.56 / 74.80 / 67.22 |
| Image-only SSD | 85.43 / 75.75 / 67.79 | 87.76 / 78.52 / 70.43 | 87.89 / 78.25 / 74.96 | 84.33 / 74.38 / 67.01 |
| Gated-only SSD | 77.10 / 61.95 / 58.27 | 80.65 / 69.64 / 61.75 | 75.16 / 66.76 / 61.68 | 77.32 / 61.31 / 57.23 |
| Lidar-only SSD | 73.46 / 57.32 / 54.62 | 68.43 / 54.82 / 51.91 | **28.98 / 25.24 / 24.56** | 67.50 / 52.26 / 46.83 |
| Radar-only SSD | 10.26 / 8.54 / 8.23 | 16.92 / 13.24 / 12.66 | 16.33 / 13.57 / 13.00 | 12.94 / 10.95 / 10.40 |
| AVOD-FPN [35] | 66.47 / 58.71 / 51.63 | 60.40 / 52.51 / 51.92 | 33.95 / 26.29 / 26.17 | 59.55 / 51.91 / 50.54 |
| Frustum PointNet [48] | 80.06 / 75.89 / 67.70 | 84.06 / 76.88 / 75.44 | 76.69 / 73.62 / 68.49 | 78.34 / 74.34 / 66.52 |

**Key observations**:

1. **Lidar 在 dense fog 下崩盘**: 73.46 → 28.98 (easy AP, -44.48 absolute, -60% relative)。这就印证了 asymmetric distortion —— lidar 不是 redundant backup，而是 first casualty。

2. **AVOD-FPN 是 worst fusion**: dense fog easy AP 33.95，比 lidar-only (28.98) 略好但远低于 image-only (87.89)。原因：AVOD 的 proposal 机制依赖 lidar point density 做 importance sampling，fog 中 lidar 全是 backscatter point，proposal 阶段就崩了。

3. **Late fusion (Fusion SSD) 在 dense fog 下 74.07 AP**，比 image-only 87.89 还差。这就是 paper 的核心论点：late fusion 学不到 redundancy，反而被 garbage sensor 拖累。

4. **Domain adaptation (ADDA, CycADA) 反而不如 proposed**: 尽管 ADDA/CycADA 用了 target domain 的 unlabeled data (这是 unfair advantage)，但效果仍不如 deep entropy fusion。CycADA 的 style transfer 在 dense fog 下 87.24 (easy)，但 hard case 只有 73.38 —— 严重 underperform proposed 的 76.69。这说明 weather distortion 不是 simple style transfer 能 capture 的，物理上 asymmetric 的 sensor distortion 不能用 image-domain GAN 解决。

5. **Deep Fusion (no entropy) vs Deep Entropy Fusion**: 这是最 direct ablation。dense fog hard case: 73.93 → 76.69 (+2.76)。说明 entropy steering 在最 severe 场景下贡献最大。

6. **Snow/Rain vs Dense Fog**: snow/rain 对 lidar 影响比 dense fog 小 (67.50 vs 28.98 easy AP)，因为 snow/rain 的 backscatter 没 fog 那么 dense。

7. **Radar-only 性能极低** (10-16 AP) 但稳定。Radar 的 azimuthal resolution 太低，单独用不行，但作为 fusion 中的 "always available" signal 很有价值。

---

## 5. Intuition Building

让我尝试给 Andrej 你 build 一个清晰的 mental model。

### 5.1 为什么 Late Fusion 在 Asymmetric Distortion 下失败

Late fusion (AVOD, Fusion SSD, Frustum PointNet) 的 implicit model 是：
- 每个 sensor branch 独立提取 feature / 生成 proposal
- 在最后 stage 把 proposal 或 high-level feature concatenate / vote

这等价于 ensemble of single-sensor detectors。当 sensors 都 reliable 时，ensemble > single。但当某个 sensor 在 inference 时变成 pure noise，它的 proposal / feature 仍然是 "confident garbage" —— network 没有 mechanism 判断 "我应该 ignore 这个 sensor now"。这就是为什么 Fusion SSD 在 dense fog 下 74.07 < Image-only 87.89：lidar branch 持续输出 fog backscatter 假 detections，污染 final decision。

### 5.2 为什么 Early Fusion (Concat SSD) 也失败

Concat SSD 是 naive early fusion: 把所有 sensor input 在 input layer concatenate，喂给一个 backbone。这种 fusion 完全 rely backbone 自己学到 "which channel to trust"。问题是 clean training data 里所有 sensor 都是 reliable 的，backbone 学到的最优策略是 "average all sensors" —— 没有 incentive 学到 "down-weight specific sensor in specific condition"。在 inference 时遇到 fog，backbone 不知道 lidar channel 现在是 garbage，继续 average，于是被拖累。

### 5.3 为什么 Deep Entropy Fusion 成功

关键在于 entropy map 提供了一个 **explicit, runtime-computed reliability signal** 给 fusion layer。流程是：

```
At inference time, in fog:
- Lidar input → mostly backscatter → low spatial entropy in object regions (uniform noise)
- RGB input → foggy but some structure → moderate entropy
- Gated input → clear structure → high entropy
- Radar input → low res but present → moderate entropy

Entropy steering weights:
- W_lidar ≈ 0 (attenuate)
- W_rgb ≈ 0.5
- W_gated ≈ 1 (pass through)
- W_radar ≈ 0.7

→ Fused feature dominated by gated + radar, lidar suppressed
```

这是一个 **runtime adaptive gating**，且 gate value 是从 input 本身计算出来的，不需要外部 weather classifier。

### 5.4 为什么 Training 只用 Clean Data 也能 Work

这是 paper 最 surprising 的部分。直觉上，要 learn "fog 时 down-weight lidar"，需要 fog training data。但 paper 证明不需要，原因是：

1. **Random dropout 模拟 asymmetric failure**: 训练时 50% 概率 drop 每个 sensor，等价于告诉 network "any sensor can fail at any time"。network 学到的策略不是 "在 fog 时 down-weight lidar"，而是更 general 的 "**当某 sensor entropy 低时 down-weight 它**"。

2. **Entropy 是 weather-agnostic feature**: clean data 里也有 low entropy region (sky, uniform wall)，network 学到 "low entropy → low weight"。这个规律在 fog 中同样适用 —— backscatter region entropy 也低。所以 generalization 是 through **shared entropy structure**，不是 through weather-specific pattern。

3. **Clean data 提供 object appearance prior**: 只要 object appearance 在 clear 和 fog 中没完全变 (foggy image 仍保留 low-frequency object shape)，detector head 学到的 object prior 就 transferable。

这其实是一种 **implicit domain randomization** —— 类似 OpenAI 的 domain randomization for sim2real，但用在 sensor dropout 上。

---

## 6. Connections to Related Work

### 6.1 Gated Imaging

paper 引用了 Gruber et al. [23] Gated2Depth (ICCV 2019): https://www.computationalimaging.org/Publications/gated2depth/

Gated imaging 是 active sensing: pulsed laser + time-gated shutter。Laser 发短脉冲，shutter 在延迟 $\tau$ 后短暂开启，只接收距离 $c\tau/2$ 附近的回波。这从根本上 **物理 suppress backscatter**，因为 backscatter 主要是近距离 (sensor 前几米) 的大气散射，gated shutter 开启时这些早到 backscatter 已经过去了。

paper 里 gated camera 是 key sensor，因为它是唯一在 fog 中仍 high-resolution 的 sensor。

### 6.2 Single Shot Detector (SSD)

Liu et al. SSD: https://arxiv.org/abs/1512.02325

SSD 是 single-stage detector，从多个 feature pyramid level 直接 predict anchor boxes 的 class + offset。比 Faster-RCNN 两阶段快很多。paper 用 SSD 因为 real-time 约束。

### 6.3 AVOD / MV3D / Frustum PointNet

- MV3D (Chen et al.): https://arxiv.org/abs/1611.02179 - 多 view (BeV + front view) proposal fusion
- AVOD (Ku et al.): https://arxiv.org/abs/1805.03918 - 类似 MV3D，feature pyramid + ROI pool
- Frustum PointNet (Qi et al.): https://arxiv.org/abs/1711.08488 - image 先 detect 2D box，再 frustum 投到 3D，PointNet 处理

这些都是 proposal-level fusion，paper 论证它们在 asymmetric distortion 下失败。

### 6.4 DENSE Project

paper acknowledgement 提到 EU H2020 ECSEL DENSE project (contract 692449): https://dense2020.eu/

DENSE 是 European project 专门做 adverse weather 下的 automotive perception，包含了多个 sensor (gated, FIR, radar, lidar) 的 adverse weather benchmark。

### 6.5 后续工作

这篇 paper 之后的相关方向：

- **Entropy-guided fusion** 被后续多个工作引用，例如 probabilistic fusion with uncertainty estimation
- **Gated camera** 在自动驾驶 perception 里逐渐成为研究热点 (BrightwayVision 商业化)
- **Domain generalization** (而不是 domain adaptation) 思路在 weather robustness 里成为 trend，例如:
  - "Robust Learning Through Cross-Task Consistency"
  - "On the Effect of Adversarial Distillation on Asymmetric Robustness"

---

## 7. Limitations & Open Questions

paper 自己提了几个 limitations / future work:
1. **Failure detection + adaptive sensor control**: 例如 lidar 在 fog 时可以 boost power 或者 change pulse pattern。需要 end-to-end model 输出 sensor control signal。
2. **No end-to-end failure detection**: 当前 entropy 只用来 steer fusion，没有显式输出 "this sensor is failing" 信号给 downstream decision making。

我补充几个观察:
1. **Entropy 是 fixed heuristic**: 用 Shannon entropy on 16x16 patch 是 hand-crafted feature。理论上可以让 network 自己 learn 一个 "reliability head"，但那样需要 failure label。Entropy 的好处是 unsupervised。
2. **Image-plane projection 信息损失**: 把 lidar point cloud 投到 image plane 会丢失被遮挡区域的 point。BeV 不会有这个问题。Trade-off 是 image-plane 允许 early fusion，BeV 只能 late fusion。
3. **Radar 表达过于简化**: paper 把 radar scan 直接沿 vertical axis replicate，这是 lossy encoding。更精细的 radar 表达 (例如 Doppler 信息) 没用上。
4. **Weather distribution 仍偏 clear**: dataset dense fog 只有 1k frames，dense fog hard case 的统计意义可能不够。

---

## 8. 总结: Why This Paper Matters

这篇 paper 在 2020 年是多模态 fusion 领域的一个重要 milestone，核心贡献不是 architecture 本身 (VGG + SSD + feature exchange 都不 fancy)，而是:

1. **Systematic expose asymmetric distortion problem**: 用 controlled fog chamber + real-world 10k km driving 给出 quantitative evidence that asymmetric sensor failure 是真实存在且 severe 的，且现有 fusion 方法 handle 不了。

2. **Propose weather-agnostic reliability proxy (entropy)**: 不需要 weather classifier，不需要 adverse weather training data，runtime 计算 entropy 就能 steer fusion。这个 idea 干净且 general。

3. **Train on clean, generalize to adverse**: 这种 "seeing without seeing" 的 paradigm 在 safety-critical system 里特别重要，因为 edge case data 永远 rare。

4. **Dataset release**: 第一次大规模 multimodal adverse weather dataset，对后续研究有持续价值。

对 Andrej 你可能感兴趣的角度:
- 这是一种 **implicit domain randomization** 的 case study，类似你之前讨论 sim2real 时提到的 "structure invariance"
- Entropy 作为 reliability proxy 是 unsupervised 的，可以推广到其他 modality (audio, IMU, event camera)
- "Train on clean, test on corrupt" 这个 paradigm 本质上考验 architecture 的 inductive bias —— 什么 prior 让 network generalize 到 unseen corruption

References:
- Paper: https://arxiv.org/abs/1902.08951
- Code & Data: https://github.com/princeton-computational-imaging/SeeingThroughFog
- Felix Heide Lab: https://www.computationalimaging.org/
- DENSE Project: https://dense2020.eu/
- Gated2Depth (companion work): https://www.computationalimaging.org/Publications/gated2depth/
- Dataset info: https://github.com/princeton-computational-imaging/SeeingThroughFog (includes download links)
- Mercedes-Benz DENSE showcase: https://www.mercedes-benz.com/en/innovation/research/vehicle-safety-research/adverse-weather-perception/
