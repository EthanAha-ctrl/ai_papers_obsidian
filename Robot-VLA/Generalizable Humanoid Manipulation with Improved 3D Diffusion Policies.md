---
source_pdf: Generalizable Humanoid Manipulation with Improved 3D Diffusion Policies.pdf
paper_sha256: 998b2463371dad769fceaa509711a992fffe84f535e0d8e2410456683fed05e9
processed_at: '2026-08-04T13:35:16-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# iDP3 人话版

## 一句话总结

**让robot用"我看到的3D世界"来学动作，rather than用"房间坐标系的3D世界"，这样robot换个地方也能干活。**

---

## 这个paper在解决什么问题？

想象你train了一个robot，在lab里能perfectly抓杯子。然后你把它推到kitchen，它就懵了。Why？

因为之前的vision policy看的是2D image。Lab里的image长这样：白桌子、白墙、日光灯。Kitchen里的image长这样：木桌子、锅碗瓢盆、暖色灯。对CNN来说，这俩image的pixel distribution完全不同，所以features完全不同，所以policy就崩了。

这就像一个人只在白教室里学会抓东西，换了教室就不会了——他学到的是"白色背景里的visual pattern"，而不是"手和物体的空间关系"。

iDP3的insight：**如果robot看的是3D点云，而且是在自己眼睛的坐标系下看的，那这种空间关系不会因为换房间而改变。**

---

## 为什么3D比2D更generalizable？

### 用日常例子build intuition

你闭着眼睛抓杯子的时候，你的brain在算什么？

不是"杯子在房间的(3.2, 1.5, 0.8)位置"，因为如果你挪到另一个房间，这个坐标就变了，但你的brain还是能抓。

你的brain算的是：**"杯子在我眼睛正前方偏下30厘米、手往左伸20厘米能碰到"**。

这就是egocentric representation——以你的眼睛为原点的3D空间关系。

换到kitchen、换到office、换到outdoor，这个relationship不变。杯子还是在你眼睛前方偏下，手还是往左伸20厘米。

**这就是iDP3 view invariance和scene generalization的physical basis。**

### 数学上怎么回事

World frame的point cloud：
$$\mathbf{P}_{world} = \mathbf{R}_{wc} \mathbf{P}_{cam} + \mathbf{t}_{wc}$$

其中：
- $\mathbf{P}_{cam}$ = camera坐标系下的点云，每个点是$(x_c, y_c, z_c)$，原点是camera镜头中心
- $\mathbf{P}_{world}$ = world坐标系下的点云，原点是某个fixed point（比如table角）
- $\mathbf{R}_{wc} \in SO(3)$ = camera到world的rotation matrix
- $\mathbf{t}_{wc} \in \mathbb{R}^3$ = camera到world的translation

DP3用$\mathbf{P}_{world}$，所以必须知道$\mathbf{R}_{wc}$和$\mathbf{t}_{wc}$（这就是calibration）。

iDP3直接用$\mathbf{P}_{cam}$，所以network根本不需要知道camera在world里的pose。

**当robot head转动时**：
- $\mathbf{P}_{cam}$：scene内容变了，但"杯子相对于我眼睛"的关系在local sense是structured的
- $\mathbf{P}_{world}$：如果没recalibrate，coordinates完全错乱

参考：[DP3 paper](https://arxiv.org/abs/2403.03954) | [SE(3) equivariance in robotics](https://arxiv.org/abs/2407.01479)

---

## 但egocentric有个大问题

DP3用world frame是有reason的：可以easy segmentation。

在world frame里，table surface在$z=0$平面，object在$z>0$，很好分割。Policy只看object附近的points。

iDP3用camera frame，没法这么做。Camera看到的整个scene——table、background wall、floor、object——全部混在$\mathbf{P}_{cam}$里。Task-relevant的points（object + gripper）可能只占5%。

### iDP3的解法：暴力scaling

不segment？那就把所有points都喂进去，让network自己figure out哪些重要。

DP3用$N=1024$ points。iDP3用$N=4096$。

| Points | Success |
|--------|---------|
| 1024 (DP3) | 56/129 (43%) |
| 2048 | 65/128 (51%) |
| 4096 (iDP3) | 75/139 (54%) |
| 8192 | 72/132 (55%) |

从1024到4096涨了11个百分点。从4096到8192基本持平，甚至略降。

**Intuition**：太少points，foreground被dilute；太多points，noisy background的signal overwhelm foreground，且compute贵。

这个trade-off说明：**没有free lunch。Egocentric放弃了easy segmentation，用brute-force data volume来补偿。**但这个补偿是有效的，因为3D structure本身是informative的。

### Sampling怎么快

DP3用FPS (Farthest Point Sampling)，$O(N^2)$复杂度，慢。

iDP3用voxel sampling + uniform sampling的cascade：

**Voxel sampling**：把3D space切成小cube（voxel），每个voxel只保留一个代表point。$O(N)$。

**Uniform sampling**：如果voxel后还太多，random subsample到target N。

这比FPS快10倍左右，且spatial coverage更uniform。对real-time deployment重要。

```python
# 简化pseudo-code
points = raw_point_cloud  # ~100k points
points = voxel_downsample(points, voxel_size=0.01)  # ~10k points
points = uniform_sample(points, target=4096)  # exactly 4096
```

---

## Visual Encoder的改进

### DP3的问题

DP3用MLP处理每个point：

$$\mathbf{f}_i = \text{MLP}(\mathbf{p}_i)$$

每个point独立transform，然后用max-pool聚合。这丢了spatial structure。

### iDP3的Conv + Pyramid

Point cloud虽然是unordered set，但在camera frame下，它来自image plane的depth projection。所以implicit地有一个2D spatial structure。

Conv1D能capture相邻points的local pattern：

$$\mathbf{f}^{(l+1)} = \text{Conv1D}_k(\mathbf{f}^{(l)})$$

其中：
- $k$ = kernel size（看几个邻居）
- $l$ = layer index
- $\mathbf{f}^{(0)} = \mathbf{P}_{cam}$（原始坐标）

**Pyramid**：不同layer有不同receptive field。浅层看fine details（gripper fingertip精确位置），深层看coarse structure（object整体在哪）。

$$\mathbf{z} = \text{Pool}(\text{Concat}(\mathbf{f}^{(1)}, \mathbf{f}^{(2)}, \mathbf{f}^{(3)}, \mathbf{f}^{(4)}))$$

把所有layer的features拼起来再pool。

| Encoder | Success |
|---------|---------|
| Linear (DP3 original) | 58/127 (46%) |
| Conv alone | 49/131 (37%) |
| Linear + Pyramid | 66/134 (49%) |
| Conv + Pyramid (iDP3) | 75/139 (54%) |

**有趣的ablation**：Conv alone比Linear还差。但Conv+Pyramid最好。

**Intuition**：Conv的local receptive field太窄，alone时丢失global context。Pyramid的multi-scale aggregation才让conv发挥优势。这就像CNN里深层layer单独拿出来用不如整网络好用，要multi-scale fusion。

参考：[FPN](https://arxiv.org/abs/1612.03144) | [PointNet++](https://arxiv.org/abs/1706.02413)

---

## Prediction Horizon：最striking的result

这个ablation结果让我震惊：

| Horizon | Success |
|---------|---------|
| 4 (DP3) | **0/0 (完全学不会)** |
| 8 | 33/88 (38%) |
| 16 (iDP3) | 75/139 (54%) |
| 32 | 55/130 (42%) |

Horizon=4直接**完全失败**。不是accuracy低，是学都学不会。

### 为什么

Human teleoperation data有三个特点：
1. **Jitter**：人的手会抖，trajectory有high-frequency noise
2. **Sensor noise**：LiDAR点云不perfect
3. **Multi-modality**：同一个state，人可能选slightly不同的action

Horizon=4时，policy只predict未来4步。这4步里，human jitter占很大比重。Network试图fit这些noise，结果学到的是garbage——它overfit到"在这个exact joint configuration下，往这个微小方向抖一下"。

Horizon=16时，policy必须predict较长的trajectory trend。这迫使network忽略短期fluctuation，学习underlying action plan。**这相当于一个temporal low-pass filter。**

数学上，diffusion loss是：

$$\mathcal{L} = \mathbb{E}\left[\sum_{t=1}^{T_p} \|\hat{\mathbf{a}}_t - \mathbf{a}_t^*\|^2\right]$$

其中：
- $T_p$ = prediction horizon
- $\hat{\mathbf{a}}_t$ = predicted action at step $t$
- $\mathbf{a}_t^*$ = ground truth action

$T_p$ 小时，每个step的loss权重相对大，network过分fit每一步。$T_p$ 大时，远端steps的gradient signal弱，但overall trajectory shape被capture。

### 为什么32又下降了

太长horizon有两个问题：
1. **远端prediction uncertainty大**：预测30步后的action，variance太高，loss梯度noisy
2. **Compute cost**：每次inference要denoise更长的sequence

16是sweet spot。

**Connection到ACT**：[ALOHA的ACT paper](https://arxiv.org/abs/2304.13705)也发现action chunking很重要。iDP3的horizon=16本质上是chunk size=16的action prediction。这是behavior cloning的universal insight：**predict chunks, not single steps, to combat noise and multi-modality。**

---

## Diffusion Policy本身是怎么回事

### 用人话说

想象action是一段视频。Diffusion policy是这样工作的：

1. **Training时**：拿一段human demo（clean action sequence），逐步加noise直到变成pure Gaussian noise。Network学习"给定noisy action + observation，预测加了什么noise"。

2. **Inference时**：从pure noise出发，network iteratively去除noise，慢慢"crystallize"出clean action sequence。

### 为什么用diffusion而不是直接regression

**Multi-modality**。同一个observation，可能有多个valid actions（比如抓杯子可以从左边抓也可以从右边抓）。

直接regression用MSE loss会average掉multi-modality，得到一个"从中间穿过去"的无效action。

Diffusion是generative model，能model整个action distribution，sample出其中一个mode。

### 数学

**Forward process**（加噪）：

$$\mathbf{a}_k = \sqrt{\bar{\alpha}_k} \mathbf{a}_0 + \sqrt{1-\bar{\alpha}_k} \epsilon$$

其中：
- $\mathbf{a}_0$ = clean action
- $\mathbf{a}_k$ = noisy action at step $k$
- $\bar{\alpha}_k = \prod_{i=1}^k \alpha_i$ = cumulative noise schedule
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ = random noise

**Training loss**：

$$\mathcal{L} = \mathbb{E}_{\mathbf{a}_0, \epsilon, k}\left[\|\epsilon - \epsilon_\theta(\mathbf{a}_k, \mathbf{o}, k)\|^2\right]$$

Network $\epsilon_\theta$ 学习predict noise $\epsilon$。

**DDIM inference**（10步去噪）：

50步training但只10步inference，靠DDIM的non-Markovian skip。每步：

$$\mathbf{a}_{k-1} = \sqrt{\bar{\alpha}_{k-1}} \hat{\mathbf{a}}_0 + \sqrt{1-\bar{\alpha}_{k-1}} \epsilon_\theta(\mathbf{a}_k, \mathbf{o}, k)$$

其中 $\hat{\mathbf{a}}_0 = \frac{\mathbf{a}_k - \sqrt{1-\bar{\alpha}_k}\epsilon_\theta}{\sqrt{\bar{\alpha}_k}}$ 是predicted clean action。

参考：[Diffusion Policy](https://arxiv.org/abs/2303.04137) | [DDIM](https://arxiv.org/abs/2010.02502) | [DDPM](https://arxiv.org/abs/2006.11239)

---

## 整个System怎么work

### Hardware

- **Robot**: Fourier GR1 humanoid，25 DoF
  - Head: 2 DoF（左右、上下看）
  - Waist: 2 DoF（弯腰、侧弯）
  - Arms: 7 DoF × 2
  - Hands: 6 DoF × 2（Inspire Hands）
- **Camera**: RealSense L515 LiDAR（装在head上）
  - 为什么用L515不用D435？D435是stereo IR，depth noise大。L515是solid-state LiDAR，精度高。DP3的experiments已经证明noisy depth hurt performance。
- **Cart**: height-adjustable，解决table height variation

### Teleoperation

用Apple Vision Pro（AVP）：

1. 人戴AVP
2. AVP track人的head pose、hand pose、wrist pose
3. Relaxed IK把这些human pose retarget到robot joints
4. Robot执行
5. Robot camera画面stream回AVP，人看到robot视角

**Latency**: ~0.5秒。Why这么慢？LiDAR占bandwidth。

**为什么加waist**：之前Open-TeleVision只teleop上半身。加waist后robot能弯腰够低处、能侧身，workspace大很多。这对humanoid很关键——humanoid的优势就是whole-body flexibility。

参考：[Open-TeleVision](https://arxiv.org/abs/2407.01512) | [Relaxed IK](https://roboticsproceedings.org/rss14/p32.pdf) | [Apple Vision Pro teleop](https://github.com/Improbable-AI/VisionProTeleop)

---

## 实验结果的人话解读

### Training scene（lab里）

| Method | Success |
|--------|---------|
| DP (ResNet18, from scratch) | 24/106 (23%) |
| DP (frozen R3M) | 62/138 (45%) |
| DP (finetuned R3M) | **99/147 (67%)** |
| iDP3 (DP3 encoder) | 58/127 (46%) |
| iDP3 (full) | 75/139 (54%) |

**DP+finetuned R3M在training scene上beat iDP3**。

Why？R3M是pre-trained on large-scale human video，有strong 2D visual priors。iDP3是3D from scratch，没有pre-training。

这符合[On Pre-training for Visuo-Motor Control](https://arxiv.org/abs/2212.05749)的发现：pre-training通常beat from-scratch。

**但是**——

### Generalization（换scene/object/view）

| Setting | DP | iDP3 |
|---------|----|----|
| Training scene | 9/10 | 9/10 |
| New Object | 3/10 | **9/10** |
| New View | 2/10 | **9/10** |
| New Scene | 2/10 | **9/10** |

一到generalization，DP直接崩到20-30%，iDP3保持90%。

**这就是paper的punchline**：3D representation的generalization能力是emergent的、inherent的，不需要explicit design。

### 为什么2D方法generalization差

DP+R3M学到的是image appearance features。换scene意味着：
- New background texture
- New lighting
- New viewpoint（camera角度变了）

这些改变image pixel distribution，直接break CNN features。

**Color Jitter augmentation**有用但有限。它augment颜色，但没法augment几何structure变化。

### 为什么3D方法generalization强

Point cloud encode的是geometry。一个cup的3D shape在kitchen和outdoor都一样。Gripper-to-cup的3D spatial relationship不随scene变化。

**Egocentric frame进一步保证**：即使robot head角度变了，camera frame里的relative geometry保持结构相似性。

---

## View Invariance的magic

Figure 8展示：camera视角大幅变化，iDP3还是能抓。

**为什么没explicit equivariance design却有view invariance**？

我的interpretation：egocentric frame里，"object relative to camera"的representation在view change时是smoothly continuous的。Camera转一点，所有points的coordinates同步变换，relative structure preserved。

对比world frame：camera转一点，如果没recalibrate，$\mathbf{P}_{world}$完全错乱。

**数学上**：egocentric representation有一个implicit的SE(3) equivariance property。当camera pose变化$\mathbf{T}$：

$$\mathbf{P}_{cam}' = \mathbf{T}^{-1} \mathbf{T} \mathbf{P}_{cam} = \mathbf{P}_{cam}$$

Wait这个不对。让我重新想。

如果camera自身移动了$\mathbf{T}_{new}$，同一个world point在new camera frame里：

$$\mathbf{p}_{cam}' = \mathbf{T}_{new}^{-1} \mathbf{p}_{world}$$

而old camera frame：
$$\mathbf{p}_{cam} = \mathbf{T}_{old}^{-1} \mathbf{p}_{world}$$

所以：
$$\mathbf{p}_{cam}' = \mathbf{T}_{new}^{-1} \mathbf{T}_{old} \mathbf{p}_{cam}$$

这意味着view change时，所有points经历同一个rigid transformation。Point cloud的internal structure（点之间的relative distance、angle）完全不变。

**Network如果能学会relative geometry（而不是absolute coordinates），就inherently view invariant。**

Conv encoder + pyramid可能help capture这种relative structure，因为conv看local neighborhoods。

对比[EquiBot](https://arxiv.org/abs/2407.01479)的explicit Sim(3) equivariance：iDP3是implicit的，emergent from data + architecture。

参考：[VISTA](https://arxiv.org/abs/2409.03685) | [Equivariant Diffusion Policy](https://arxiv.org/abs/2407.01812)

---

## Limitations

### Teleoperation fatigue

AVP数据采集累人。每个demo要20 rounds successful execution。Paper里每个task只10个demo。

**This is the main bottleneck**。无法scale data = 无法scale skills。

Future方向：
- [UMI-style](https://arxiv.org/abs/2402.10329) handheld data collection（不依赖robot）
- Autonomous data collection via RL refinement
- Shared autonomy（human + robot collaborative）

### Sensor noise

即使L515，点云还是noisy。Paper明确说"even L515 does not produce perfectly accurate point clouds"。

Noisy 3D input限制iDP3性能上限。如果有perfect depth sensor，iDP3可能在training scene上也beat DP+R3M。

### 没用lower body

用cart代替locomotion。Paper说"we believe our approach will perform equally well once whole-body control techniques become mature"——但这还是TODO。

Future: integrate with [humanoid locomotion policies](https://arxiv.org/abs/2402.19469)做loco-manipulation。

### 精细任务困难

螺丝刀这种精细任务teleop困难。Why？AVP tracking精度有限 + 0.5s latency。

Future: haptic feedback、shared autonomy、fine-grained teleop interfaces。

参考：[HumanPlus](https://arxiv.org/abs/2406.10454) | [OmniH2O](https://arxiv.org/abs/2406.08858) | [UMI](https://arxiv.org/abs/2402.10329)

---

## Big Picture: 这篇paper的意义

### 1. 3D policy终于在humanoid上work了

之前3D policies（DP3, PerAct, GNFactor）只在fixed camera setup验证。iDP3是first 3D policy在real humanoid上deploy且generalize。

### 2. Egocentric是paradigm shift

从"robot在environment里"到"environment在robot眼睛里"。这个frame switch unlock了mobile robot deployment。

类比：computer vision从global image features到CNN local features的paradigm shift。Frame of reference matters。

### 3. Data efficiency的implication

1 scene训练 → 多scene部署。

如果每个新scene都要collect data（像[RUM](https://arxiv.org/abs/2410.10154)那样20个scene），data cost巨大。iDP3的generalization意味着data cost可能dramatically降低。

### 4. Inductive bias > model scale

iDP3没有用huge pre-trained model（不像R3M/VLA）。它用small model + right inductive bias（3D egocentric）就beat了2D pre-trained methods in generalization。

**Key lesson**: 对于manipulation，3D geometric structure是比2D appearance更informative的inductive bias。

### 5. 3D pre-training是open frontier

当前iDP3是from scratch。如果有3D-native pre-trained backbone（类似R3M但3D），iDP3可能在training scene上也dominate。

Open questions：
- 能否self-supervised pre-train 3D representations？[Point-MAE](https://arxiv.org/abs/2203.07164), [Point-BERT](https://arxiv.org/abs/2111.14819)是起点
- 能否build 3D foundation model for robotics？
- VLA + 3D hybrid？

参考：[Octo](https://arxiv.org/abs/2405.12213) | [OpenVLA](https://arxiv.org/abs/2406.09246) | [RT-2](https://arxiv.org/abs/2307.15818)

---

## Final Intuition Summary

### Mental Model

想象iDP3是一个"3D-native brain"：

- **Input**：它眼睛看到的3D点云（egocentric camera frame）
- **Processing**：用conv提取multi-scale spatial features
- **Output**：一段action trajectory（16步未来plan）
- **Mechanism**：diffusion process从noise中"crystallize"出coherent plan

### 为什么这些设计synergistic

1. **Egocentric**引入background noise → **Scaling up points**补偿（brute force）
2. **More points**需要better encoder → **Conv + Pyramid**提取structured features
3. **Human data**有jitter → **Long horizon**做temporal smoothing

四个改进互相enable。去掉任何一个都collapse（见ablation）。

### 最deep的insight

**Representation的frame和structure比model size更重要。**

Egocentric 3D是humanoid manipulation的正确abstraction。它match了人类brain的spatial reasoning方式。一旦用对representation，generalization是emergent的，不需要explicit design。

**当前bottleneck是hardware（sensor noise, data scale），不是algorithm。**一旦3D sensors和3D pre-training成熟，3D policies很可能dominate manipulation research。

---

## 关键reference汇总

### Core papers
- [iDP3 website](https://humanoid-manipulation.github.io/)
- [DP3 (predecessor)](https://arxiv.org/abs/2403.03954)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)

### Methods
- [DDIM](https://arxiv.org/abs/2010.02502)
- [AdamW](https://openreview.net/forum?id=Bkg6RiCqY7)
- [R3M](https://arxiv.org/abs/2203.12601)
- [ACT (action chunking)](https://arxiv.org/abs/2304.13705)

### Humanoid & teleop
- [Fourier GR1](https://www.fourierintelligence.com/gr1)
- [Apple Vision Pro teleop](https://github.com/Improbable-AI/VisionProTeleop)
- [Relaxed IK](https://roboticsproceedings.org/rss14/p32.pdf)
- [RealSense L515](https://www.intelrealsense.com/lidar-camera-l515/)
- [HumanPlus](https://arxiv.org/abs/2406.10454)
- [OmniH2O](https://arxiv.org/abs/2406.08858)
- [Open-TeleVision](https://arxiv.org/abs/2407.01512)

### Related 3D & generalization
- [EquiBot (Sim(3) equivariance)](https://arxiv.org/abs/2407.01479)
- [Equivariant Diffusion Policy](https://arxiv.org/abs/2407.01812)
- [VISTA (view synthesis)](https://arxiv.org/abs/2409.03685)
- [Maniwhere](https://arxiv.org/abs/2407.15815)
- [RUM](https://arxiv.org/abs/2410.10154)

### 3D representations
- [PointNet](https://arxiv.org/abs/1612.00593)
- [PointNet++](https://arxiv.org/abs/1706.02413)
- [Point-MAE](https://arxiv.org/abs/2203.07164)
- [Point-BERT](https://arxiv.org/abs/2111.14819)
- [FPN](https://arxiv.org/abs/1612.03144)

### Pre-training & VLA
- [On Pre-training for Visuo-Motor Control](https://arxiv.org/abs/2212.05749)
- [Octo](https://arxiv.org/abs/2405.12213)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [RT-2](https://arxiv.org/abs/2307.15818)

### Humanoid locomotion (future integration)
- [Humanoid locomotion as next token prediction](https://arxiv.org/abs/2402.19469)
- [Real-world humanoid locomotion (Science Robotics)](https://www.science.org/doi/10.1126/scirobotics.adi9579)
- [Expressive Whole-Body Control](https://arxiv.org/abs/2402.16796)

### Data collection
- [UMI (handheld)](https://arxiv.org/abs/2402.10329)
- [DexCap](https://arxiv.org/abs/2403.07788)

---

**TL;DR**: iDP3告诉我们要想给humanoid robot装一个能泛化的大脑，关键是给它正确的"世界观"——以它自己的眼睛为原点的3D空间。用这个世界观，加上几个工程改进（more points, better encoder, longer horizon），robot就能从lab里的数据generalize到真实世界。这比堆model size、用fancy pre-training更有效。当前limit是hardware（sensor、data collection），algorithm层面已经ready。

---

# Improved 3D Diffusion Policy (iDP3) 深度解析

## Paper Core Thesis

这篇paper的核心insight非常elegant：**3D visual representations比2D image representations更inherently generalizable**，但现有3D policies (如DP3)依赖world frame的camera calibration和point cloud segmentation，这阻碍了它们在mobile robots (如humanoids)上的部署。iDP3通过切换到egocentric camera frame并配合几个工程改进，释放了3D representations的generalization潜力。

参考链接：[paper website](https://humanoid-manipulation.github.io/) | [DP3 original paper](https://arxiv.org/abs/2403.03954) | [Diffusion Policy](https://arxiv.org/abs/2303.04137)

---

## Background: Diffusion Policy 的数学框架

先建立intuition。Diffusion Policy把action generation建模成一个iterative denoising process。

### Forward diffusion (加噪)

给定clean action sequence $\mathbf{a}_0 \in \mathbb{R}^{T_a \times D_a}$，其中：
- $T_a$ = action horizon (预测的future steps数量)
- $D_a$ = action dimension (robot DoF数，这里是25)

forward process逐步加Gaussian noise:

$$q(\mathbf{a}_k | \mathbf{a}_{k-1}) = \mathcal{N}(\mathbf{a}_k; \sqrt{\alpha_k} \mathbf{a}_{k-1}, (1-\alpha_k)\mathbf{I})$$

其中：
- $k$ = diffusion timestep，从0到$K$ (这里$K=50$ training, 10 inference with DDIM)
- $\alpha_k$ = noise schedule，控制每步加多少noise
- $\mathbf{I}$ = identity matrix

closed form:

$$q(\mathbf{a}_k | \mathbf{a}_0) = \mathcal{N}(\mathbf{a}_k; \sqrt{\bar{\alpha}_k} \mathbf{a}_0, (1-\bar{\alpha}_k)\mathbf{I})$$

其中 $\bar{\alpha}_k = \prod_{i=1}^{k} \alpha_i$ 是cumulative product。

### Reverse diffusion (去噪，network学习)

训练一个denoising network $\epsilon_\theta$ 预测added noise：

$$\mathcal{L} = \mathbb{E}_{\mathbf{a}_0, \epsilon, k}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_k}\mathbf{a}_0 + \sqrt{1-\bar{\alpha}_k}\epsilon, \mathbf{o}, k)\|^2\right]$$

其中：
- $\mathbf{o}$ = observation (visual + proprioception)
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ = sampled noise
- $\theta$ = network parameters (用AdamW优化)

### Inference (DDIM sampling)

从 $\mathbf{a}_K \sim \mathcal{N}(0, \mathbf{I})$ 出发，用10步DDIM reverse process得到 $\mathbf{a}_0$，然后执行前几个actions，滑窗receding horizon control。

参考：[DDIM paper](https://arxiv.org/abs/2010.02502) | [AdamW](https://openreview.net/forum?id=Bkg6RiCqY7)

---

## DP3 vs iDP3: 核心架构差异

### DP3的3D representation

DP3用world frame point cloud：

$$\mathbf{P}_{world} = \{p_i \in \mathbb{R}^3\}_{i=1}^{N}$$

每个point的coordinates是相对于一个固定的world origin (通常是table corner)。这要求：
1. 精确的camera extrinsic calibration: $[\mathbf{R}_{wc} | \mathbf{t}_{wc}]$
2. Point cloud segmentation去除background

Point encoder用cross-attention:

$$\mathbf{z} = \text{CrossAttn}(\mathbf{Q}, \mathbf{P}_{world})$$

其中 $\mathbf{Q} \in \mathbb{R}^{M \times d}$ 是$M$个learnable query tokens。

### iDP3的egocentric 3D representation

iDP3直接用camera frame：

$$\mathbf{P}_{cam} = \{p_i^{cam} \in \mathbb{R}^3\}_{i=1}^{N}$$

每个point的coordinates是相对于camera optical center。**关键insight**：当robot head转动时，$\mathbf{P}_{cam}$中的spatial relationships (object-to-gripper distance, relative angles)保持结构相似性，而world frame representation会完全改变。

数学上，两者的关系：

$$\mathbf{P}_{world} = \mathbf{R}_{wc} \mathbf{P}_{cam} + \mathbf{t}_{wc}$$

iDP3让policy直接在$\mathbf{P}_{cam}$空间学习，因此network不需要知道$\mathbf{R}_{wc}, \mathbf{t}_{wc}$。这就是为什么不需要calibration。

**Intuition**: 想象你闭上眼睛抓杯子。你的brain用的是"手相对于眼睛看到的杯子"的spatial relationship，而不是"杯子和手相对于房间东北角"的absolute coordinates。iDP3模拟前者。

---

## iDP3的四大改进

### Improvement 1: Egocentric 3D Visual Representations

如上所述。这unlock了humanoid deployment，因为humanoid的head一直在动。

**Trade-off**: 失去了用world frame做easy segmentation的能力。Background points会进入$\mathbf{P}_{cam}$。解决方案见Improvement 2。

### Improvement 2: Scaling Up Vision Input

DP3用 $N=1024$ points (sparse FPS sampling)。iDP3用 $N=4096$。

**为什么有效**: 当不segment时，task-relevant points (object, gripper)只占$\mathbf{P}_{cam}$的一小部分。如果$N$太小，FPS可能采不到足够的foreground points。Scaling up确保foreground points有足够representation。

**Voxel + Uniform sampling替代FPS**:
- FPS: $O(N^2)$复杂度，保证spatial coverage但慢
- Voxel sampling: 把space分成voxel grid，每个voxel取一个代表point。$O(N)$，快且保证coverage
- Uniform sampling: 在voxel sampling后随机subsample到target $N$

从Table II看：
| N | Total Success |
|---|---|
| 1024 (DP3) | 56/129 |
| 2048 | 65/128 |
| 4096 (iDP3) | 75/139 |
| 8192 | 72/132 |

8192反而下降，说明存在sweet spot。太多points会引入更多noisy background，且compute cost上升。

### Improvement 3: Improved Visual Encoder (Conv + Pyramid)

DP3用MLP encoder处理每个point的features。iDP3用pyramid convolutional encoder。

**架构**:

$$\mathbf{f}_1 = \text{Conv1D}(\mathbf{P}_{cam})$$
$$\mathbf{f}_2 = \text{Conv1D}(\mathbf{f}_1)$$
$$\vdots$$
$$\mathbf{f}_L = \text{Conv1D}(\mathbf{f}_{L-1})$$
$$\mathbf{z} = \text{Pool}(\text{Concat}(\mathbf{f}_1, \mathbf{f}_2, ..., \mathbf{f}_L))$$

Pyramid structure: 不同layer的feature map有不同receptive field。浅层capture fine-grained spatial patterns (gripper fingertip位置)，深层capture coarse scene structure (object整体位置)。

**为什么conv比MLP好**: Point cloud虽然unordered，但在camera frame下有implicit spatial ordering (像素grid的projection)。Conv能exploit这个local spatial structure。MLP把每个point独立transform，丢失了spatial context。Table II显示Conv+Pyramid (75/139) > Linear+Pyramid (66/134) > Linear/DP3 (58/127) > Conv alone (49/131)。

注意Conv alone比Linear差，但加Pyramid后最好。这说明pyramid的multi-scale aggregation是关键，conv只是实现它的工具。

参考：[FPN paper](https://arxiv.org/abs/1612.03144) | [PointNet](https://arxiv.org/abs/1612.00593)

### Improvement 4: Longer Prediction Horizon

这是最striking的ablation result:

| Horizon | Total Success |
|---|---|
| 4 (DP3) | 0/0 (完全失败) |
| 8 | 33/88 |
| 16 (iDP3) | 75/139 |
| 32 | 55/130 |

**为什么horizon=4完全失败**: Human teleoperation data有jitter (手抖)和sensor noise。短horizon让policy过分sensitive to这些high-frequency noise。Network学到的是"在这个exact joint configuration下，往这个微小方向动一下"，无法generalize。

**为什么horizon=16最好**: Longer horizon等于temporal low-pass filter。Network必须predict整个trajectory的trend，ignore短期fluctuation。这类似于behavior cloning中的action chunking effect ([ACT paper](https://arxiv.org/abs/2304.13705))。

**为什么horizon=32下降**: 太长会underfit immediate dynamics。远端predictions的gradient signal弱，且action distribution太diffuse。

数学上，prediction horizon影响training loss的有效scope:

$$\mathcal{L} = \sum_{t=1}^{T_p} w_t \cdot \|\hat{\mathbf{a}}_t - \mathbf{a}_t^*\|^2$$

$T_p=16$时，loss权衡了short-term accuracy和long-term planning。$T_p=4$时，loss太local，overfits noise。

---

## System Architecture 全景

### Hardware Stack

- **Robot**: Fourier GR1 humanoid，25 DoF (head + waist + 2 arms × 7 + 2 hands × 6)
- **Camera**: Intel RealSense L515 solid-state LiDAR (比D435深度更准)
- **Hands**: Inspire Hands (dexterous，6 DoF each)
- **Cart**: Height-adjustable，解决table height variation问题

### Teleoperation Pipeline

```
Human movement → Apple Vision Pro tracking
                → Relaxed IK retargeting
                → Robot joint targets
                → Robot execution
                ← Visual feedback (camera stream back to AVP)
```

- AVP tracks: hand pose, wrist pose, head pose
- Relaxed IK: real-time IK solver with smoothness constraints ([paper](https://roboticsproceedings.org/rss14/p32.pdf))
- Waist DoF incorporated (相比Open-TeleVision的改进)
- Latency: ~0.5s (LiDAR占用bandwidth)

### Data Format

每个trajectory包含：
- **Observation**: 
  - Visual: point cloud $\mathbf{P}_{cam} \in \mathbb{R}^{4096 \times 3}$ + RGB image (224×224)
  - Proprioception: 25-dim joint positions $\mathbf{s} \in \mathbb{R}^{25}$
- **Action**: 25-dim target joint positions $\mathbf{a} \in \mathbb{R}^{25}$

注意：尝试过用end-effector pose作为action，效果没显著差异。这暗示在dexterous manipulation中，joint space和task space representation等价。

---

## 实验结果深度解读

### Table I: Training Scene Effectiveness

| Method | Total |
|---|---|
| DP (ResNet18) | 24/106 (22.6%) |
| DP (❄R3M frozen) | 62/138 (44.9%) |
| DP (✶R3M finetuned) | 99/147 (67.3%) |
| iDP3 (DP3 encoder) | 58/127 (45.7%) |
| **iDP3** | **75/139 (54.0%)** |

**关键observation**: DP+finetuned R3M在training scene上beats iDP3。这符合pre-training > from-scratch的常识 ([Hansen et al.](https://arxiv.org/abs/2212.05749))。但R3M是2D image pre-training，iDP3是3D from-scratch。

**"Success/Attempts" metric设计巧妙**:
- Success count = accuracy
- Attempts count = smoothness (jittery policy少尝试)

iDP3的75/139意味着每次attempt成功率高且attempt频率正常，说明policy既准又smooth。

### Table III: Generalization (核心结果)

| Setting | DP | iDP3 |
|---|---|---|
| Training scene | 9/10 | 9/10 |
| New Object | 3/10 | 9/10 |
| New View | 2/10 | 9/10 |
| **New Scene** | **2/10** | **9/10** |

这是paper的punchline。在training scene上两者tie，但一到generalization，DP collapse到20%，iDP3保持90%。

**为什么2D方法generalization差**: 
2D CNN/R3M features encode texture, lighting, viewpoint-specific appearance。New scene意味着new background texture, new lighting, new viewpoint。这些feature shifts直接break policy。

**为什么3D方法generalization强**:
Point cloud encodes几何structure。一个cup的3D shape在kitchen、office、outdoor都相似。Gripper-to-cup的3D spatial relationship不随scene变化。Egocentric frame进一步保证这种invariance。

参考：[R3M](https://arxiv.org/abs/2203.12601) | [VISTA](https://arxiv.org/abs/2409.03685) | [Maniwhere](https://arxiv.org/abs/2407.15815) | [RUM](https://arxiv.org/abs/2410.10154)

---

## Ablation的Intuition总结

| Component | Effect | Intuition |
|---|---|---|
| Egocentric frame | Enables humanoid deployment | 消除calibration依赖，inherent view invariance |
| 4096 points | +19% over 1024 | Brute-force替代segmentation |
| Conv+Pyramid | +17% over Linear | Multi-scale spatial features |
| Horizon=16 | From 0% to 54% | Temporal smoothing of human noise |

这四个改进是synergistic的：egocentric引入background noise → scaling up points补偿 → conv encoder提取structured features → long horizon smooths output。

---

## Limitations & Future Directions

1. **Teleoperation fatigue**: AVP数据采集累人，无法scale。Future: autonomous data collection via RL refinement, 或[UMI-style](https://arxiv.org/abs/2402.10329) handheld data。
2. **Sensor noise**: L515点云仍noisy。Future: better depth sensors, 或neural depth completion。
3. **No lower body**: 用cart替代locomotion。Future: integrate with[humanoid locomotion policies](https://arxiv.org/abs/2402.19469)。
4. **Fine-grained tasks**: 螺丝刀等精细任务teleop困难。Future: haptic feedback, shared autonomy。

**My speculation on scaling**: 如果有large-scale 3D pre-training (类似R3M但3D)，iDP3可能在training scene上也beat DP+R3M。当前3D方法的bottleneck是缺pre-trained backbone。

参考：[HumanPlus](https://arxiv.org/abs/2406.10454) | [OmniH2O](https://arxiv.org/abs/2406.08858) | [Open-TeleVision](https://arxiv.org/abs/2407.01512) | [Expressive Whole-Body Control](https://arxiv.org/abs/2402.16796)

---

## Broader Impact & Open Questions

### 为什么这个工作重要

1. **First demonstration of 3D policy generalization on real humanoid**: 之前3D policies只在fixed setup验证，这是first wild deployment。
2. **Egocentric 3D是paradigm shift**: 类似computer vision从global features到CNN local features的转变。Frame of reference matters。
3. **Data efficiency**: 1 scene训练 → 多scene部署。这对data scaling economics影响巨大。

### Open research questions

1. **3D pre-training**: 能否设计self-supervised 3D representation pre-training (类似MAE for point clouds)？[Point-MAE](https://arxiv.org/abs/2203.07164), [Point-BERT](https://arxiv.org/abs/2111.14819)是起点。
2. **Equivariance理论**: iDP3的view invariance是emergent的，没有explicit SE(3) equivariance design。对比[EquiBot](https://arxiv.org/abs/2407.01479)的explicit Sim(3) equivariance。哪个更好？
3. **Multi-modal 3D**: 融合point cloud + RGB + tactile的3D policy？
4. **Foundation models for 3D robot learning**: 类似[Octo](https://arxiv.org/abs/2405.12213), [OpenVLA](https://arxiv.org/abs/2406.09246)但3D native。

### Connection to VLA trends

当前VLA (Vision-Language-Action) models多用2D images。如果3D representations如此generalizable，为什么VLA不用3D？可能原因：
- Language grounding在2D更成熟 (CLIP, etc.)
- 3D sensors在deployment上cost高
- 3D data scale远小于2D (internet images vs 3D scans)

但iDP3暗示：对于contact-rich manipulation，3D可能是正确inductive bias。Future VLA可能hybrid: 2D for semantic understanding, 3D for spatial reasoning。

参考：[RT-2](https://arxiv.org/abs/2307.15818) | [OpenVLA](https://arxiv.org/abs/2406.09246) | [Octo](https://arxiv.org/abs/2405.12213) | [EquiBot](https://arxiv.org/abs/2407.01479) | [Equivariant Diffusion Policy](https://arxiv.org/abs/2407.01812)

---

## Implementation细节补充

### DDIM 10-step inference

DDIM是非-Markovian reverse process，允许跳步。50 training steps → 10 inference steps的mapping:

$$\mathbf{a}_{k-1} = \sqrt{\bar{\alpha}_{k-1}} \hat{\mathbf{a}}_0 + \sqrt{1-\bar{\alpha}_{k-1}} \epsilon_\theta(\mathbf{a}_k, \mathbf{o}, k)$$

其中 $\hat{\mathbf{a}}_0 = \frac{\mathbf{a}_k - \sqrt{1-\bar{\alpha}_k}\epsilon_\theta}{\sqrt{\bar{\alpha}_k}}$ 是predicted clean action。

10步够用因为action distribution比image distribution简单。

### Voxel Sampling细节

```python
# Pseudo-code
def voxel_sample(points, voxel_size):
    # 1. Assign each point to voxel
    voxel_idx = floor(points / voxel_size).astype(int)
    # 2. Group by voxel
    unique_voxels, indices = unique(voxel_idx, return_index=True)
    # 3. One representative per voxel (e.g., centroid or random)
    return points[indices]

def cascade_sample(points, target_N):
    # Stage 1: voxel sample for coverage
    points = voxel_sample(points, voxel_size=0.01)
    # Stage 2: uniform sample to target N
    if len(points) > target_N:
        points = points[random.choice(len(points), target_N)]
    elif len(points) < target_N:
        points = repeat_or_pad(points, target_N)
    return points
```

这比FPS快~10x，且coverage更uniform。

---

## 总结：构建你的Intuition

**Mental model**: 想象iDP3是一个"3D-native brain"。它看到的不是pixels，而是spatial point cloud。它的"thoughts"是action trajectories。它用diffusion process从noise中"crystallize"出coherent action plan。

**Key inductive biases**:
1. **Egocentric frame**: "我相对于我看到的世界" — view invariant
2. **3D structure**: "几何不变" — scene invariant  
3. **Multi-scale conv**: "既要看细节也要看全局" — robust features
4. **Long horizon**: "要有plan" — noise robust

**为什么这些work together**: 人类manipulation就是3D egocentric的。我们不用world coordinates，不用segmentation，我们用"手相对于眼睛看到的物体"的3D relationship。iDP3让robot用同样的representation，因此inherit了类似的generalization。

**最终take-away**: 3D representations比2D更sample-efficient和generalizable for manipulation，前提是用对的frame (egocentric) 和对的training recipe (long horizon, multi-scale features, sufficient points)。当前bottleneck是sensor noise和data scale，不是algorithm。一旦3D sensors和3D pre-training成熟，3D policies可能dominate manipulation research。

---

## 参考资源汇总

- [Paper website (videos)](https://humanoid-manipulation.github.io/)
- [DP3 (predecessor)](https://arxiv.org/abs/2403.03954)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [DDIM](https://arxiv.org/abs/2010.02502)
- [R3M](https://arxiv.org/abs/2203.12601)
- [Relaxed IK](https://roboticsproceedings.org/rss14/p32.pdf)
- [Apple Vision Pro teleop](https://github.com/Improbable-AI/VisionProTeleop)
- [Fourier GR1](https://www.fourierintelligence.com/gr1)
- [RealSense L515](https://www.intelrealsense.com/lidar-camera-l515/)
- [EquiBot (Sim(3) equivariance)](https://arxiv.org/abs/2407.01479)
- [VISTA (view synthesis)](https://arxiv.org/abs/2409.03685)
- [HumanPlus](https://arxiv.org/abs/2406.10454)
- [OmniH2O](https://arxiv.org/abs/2406.08858)
- [Open-TeleVision](https://arxiv.org/abs/2407.01512)
- [Octo](https://arxiv.org/abs/2405.12213)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [ACT (action chunking)](https://arxiv.org/abs/2304.13705)
- [Humanoid locomotion as next token prediction](https://arxiv.org/abs/2402.19469)
- [Real-world humanoid locomotion with RL (Science Robotics)](https://www.science.org/doi/10.1126/scirobotics.adi9579)
- [Point-MAE](https://arxiv.org/abs/2203.07164)
- [Point-BERT](https://arxiv.org/abs/2111.14819)
- [On pre-training for visuo-motor control](https://arxiv.org/abs/2212.05749)
- [UMI (handheld data collection)](https://arxiv.org/abs/2402.10329)
- [Equivariant Diffusion Policy](https://arxiv.org/abs/2407.01812)
- [Maniwhere](https://arxiv.org/abs/2407.15815)
- [Robot Utility Models](https://arxiv.org/abs/2410.10154)

希望这个深度解析帮你build了关于3D visuomotor policies的intuition。核心message: **representation的frame和structure比model size更重要**。Egocentric 3D是humanoid manipulation的正确abstraction。
