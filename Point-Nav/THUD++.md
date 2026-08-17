---
source_pdf: THUD++.pdf
paper_sha256: 84c91c672bf4714f515223f1e92c55cfcda8783f54bb40a4b1289d04c4da5edc
processed_at: '2026-08-12T15:53:56-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话聊聊 THUD++

Karpathy 你好，这篇 paper 我用大白话跟你拆一下。核心其实就一句话：**现在 mobile robot 圈用的 dataset 基本都是"空房间"，但 robot 真正干活的地方全都是人**。THUD++ 就是为了填这个坑造的。

---

## 一、问题到底出在哪

你想想 ScanNet、SUN RGB-D、Matterport3D、ARKitScenes 这些大家都在用的 dataset——扫描的时候，研究人员要么挑周末没人去扫，要么等人走光了再扫。结果就是：**所有 benchmark 训出来的 perception model，遇到真实部署场景（商场、食堂、走廊有人走）直接懵掉**。

Table I 里那一列 "Dynamic objects" 基本全是 ×，只有 THUD++ 是 √。这本身就是 paper 最有力的论点。

而且更狠的是，他们发现 real scene 里有几类东西对 robot 特别致命但没人标过：
- **Glass door / window**：robot 以为能过，直接撞
- **Stairs**：下去了就上不来
- **Elevator**：robot 不知道能不能进
- **Shopping cart / other robots**：moving obstacle，现有 detector 没见过

这些类别在 ScanNet 的 21 类里根本不存在。THUD++ 扩到了 91 类，就是为了贴近 service robot 真实 deployment。

---

## 二、数据怎么来的——以及为什么 90% 是 synthetic

这里有个很现实的工程问题：**给 dynamic object 标 3D bounding box 是噩梦**。

想象一下，你在食堂录了一段 video，里面一个大爷端着餐盘走过去。你要在每一帧给他标一个 3D bbox——xyz 中心、长宽高、朝向角。每秒 30 帧，走 5 秒就是 150 个 frame，每个 frame 都要 6-DoF 标注。一个 10 秒的 clip 就要 1500 个 3D annotation。一个人手工标要一整天。

所以作者很聪明地做了一个 tradeoff：**real data 少量精标（5,191 frames），synthetic data 大量自动生成（84,984 frames）**。

Unity3D 的好处是：每个 object 的 3D pose、semantic label、instance mask 都从引擎里直接读出来，**precision = 100%，零人工噪声**。这就是为什么 synthetic data 占 90%——标 dynamic 的成本在 real 上根本扛不住。

他们也试过用 SOTA 3D detector（VoteNet、F-PointNet）做自动预标注再人工修，paper 里 honest 地说 "algorithms for 3D objects detection are not very effective"。这其实暴露了当前 3D detection 圈的一个尴尬：**在真实 indoor dynamic scene 上，SOTA 连预标注都做不好**。

---

## 三、Real data 采集的工程坑

这块值得多说两句，因为 Karpathy 你肯定 appreciate 工程细节。

PUDUbot2 + Kinect V2 这个组合有几个典型问题：

**1. Kinect V2 帧率不稳**——标称 30 fps，实际 15-30 fps 飘。原因：time-of-flight 传感器在 motion 中 depth 漂移、IR 散斑受环境光影响。

**2. Pose 和 image 时间戳对不齐**——robot pose 40 fps，image 15-30 fps。直接 nearest-neighbor 匹配会有 0-3 frame 误差，相当于最多 100 ms 的错位，robot 走 1 m/s 就是 10 cm 误差，对 3D 检测足以破坏 frustum 投影。

他们的解法是用 cubic spline 把 40 fps 的 pose 上采到 2000 fps，再找最近邻。这招其实是 SLAM 圈老 trick，EuRoC MAV dataset 也是这么干。公式就是每个区间内：

$$S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3$$

$x$ 是时间，$x_i$ 是控制点，$a_i$ 是该点函数值，$b_i, c_i, d_i$ 通过 $C^2$ 连续性约束解 tridiagonal system。

**3. Depth denoise**——Kinect V2 的 depth map 有大量 hole，他们用 Self-Supervised Deep Depth Denoising (ICCV 2019) 做后处理。这一步很关键，不然下游 3D detector 直接死在缺失的 depth 上。

这些工程细节 paper 里写得很轻描淡写，但实际做 dataset 的人都知道——**多传感器异步是 mobile robot dataset 的头号杀手**。TartanAir、Habitat 这种纯 synthetic 不用面对，但 real robot dataset 必须解决。

参考：https://github.com/edexumo/self-supervised-depth-denoising

---

## 四、Trajectory dataset：为什么 ETH/UCY 在 indoor 上失效

这块我觉得是 paper 最有 insight 的部分之一。

ETH 和 UCY 是 trajectory prediction 圈的 "MNIST"——大家都在用。但这两个 dataset 是 outdoor 的：ETH 是大学门口广场，UCY 是石板路，都是**开阔、无障碍**的场景。Pedestrian 可以走直线、走弧线，trajectory 是 smooth 的。

到了 indoor 就完全不一样了：
- **Office 7.8m × 16.2m**——两个 desk 就把空间挤成 1m 宽的过道
- **Supermarket 19m × 32m**——货架之间 aisle 只有 1.5m
- **Gym 13m × 28m**——最开阔，但还是有器材

Paper 里的实验结果（Table V）很触目惊心：

| Method | ETH (outdoor) ADE | Supermarket ADE | 退化倍数 |
|--------|-------------------|-----------------|----------|
| Social-STGCNN | 0.74 | 1.55 | 2.1x |
| Social-GAN | 1.08 | 1.81 | 1.7x |
| PECNet | 0.91 | 1.88 | 2.1x |

所有方法在 indoor 上 ADE 直接翻倍。

为什么会这样？两个根本原因：

**1. Obstacle 让 trajectory 变成 piecewise linear**

ETH 上 pedestrian 可以从 (0,0) 走到 (10,5)，trajectory 是平滑 Bezier。Indoor 里同样的起终点，中间有桌子，pedestrian 必须 (0,0) → (3,0) → (3,2) → (10,5)，是 sharp turn。Social-STGCNN 的 graph convolution 假设 interaction 是 pairwise smooth repulsion，跟现实不符。

**2. Stop-and-talk 让 velocity 双峰分布**

Outdoor pedestrian 基本一直在走，velocity 近似 Gaussian。Indoor 里人会停下来聊天、看手机、等人——velocity 分布变成两个峰（walking vs. standing）。L2 loss 训出来的 model 预测是 unimodal 均值，自然偏。

Fig. 10 里那个红框特别扎眼——Social-STGCNN 在 obstacle 密集处预测的 trajectory 直接穿墙。这说明**现有方法根本没把 static obstacle 作为 hard constraint**。

这是整个 trajectory prediction 圈的一个盲点：AgentFormer、Social-BiGAT、Y-Net 这些 SOTA 都只建模 human-human interaction，environment 只用一个 occupancy map 或 image feature，没有显式 constraint。THUD++ 把这个问题摆上台面了。

参考：
- Social-STGCNN: https://github.com/abduallahmohamed/Social-STGCNN
- AgentFormer: https://github.com/Khrylx/AgentFormer
- Y-Net: https://github.com/HarshayuGirase/PECNet

---

## 五、5 个 benchmark 暴露的核心 failure mode

每个 task 都有一个"啊原来 SOTA 这么脆弱"的发现，我一个个说：

### 5.1 3D Object Detection：dynamic 下 mAP 砸 15+ 个点

Table III：
- Supermarket（dynamic complexity = 0.94 person/frame）：DeMF 在 dynamic 上 34.51，static 上 38.24，gap 3.73
- Canteen（dynamic complexity = 3.34 person/frame）：DeMF dynamic 28.56，static 45.43，**gap 16.87**

规律很清晰：**dynamic density 越高，dynamic-static gap 越大**。

底层原因：3D detector 普遍做 multi-frame point cloud accumulation（VoteNet 系列），假设 object 在累积时间内不动。Moving person 在 5 frame 累积里变成 5 个 ghost，detector 直接懵掉。这个 insight 暗示**需要 4D-aware 3D detector**（把 time 作为第 4 维，类似 CenterPoint 4D 的设计）。

### 5.2 Semantic Segmentation：对 dynamic 反而不敏感

Table IV 里有个反直觉的发现：ESANet 在 Supermarket 上有/无 dynamic objects 训练，MIoU 78.42% vs 79.63%——**几乎无差异**。

这其实符合直觉：semantic segmentation 是 pixel-level 分类，跟 object 是否移动无关。你标 pedestrian 的 pixel，他在走还是站着，pixel label 都一样。

但 real scene vs synthetic scene 的 gap 暴大：Canteen MIoU 51.85-65.97，Supermarket MIoU 74.83-83.19。20+ 个点的 sim2real gap 说明 **synthetic data 训出来的 segmentation model 在 real 上泛化极差**——玻璃反光、sensor noise、lighting variation 这些 synthetic 模拟不真。

### 5.3 Robot Relocalization：local feature 死得最惨

Fig. 9 展示了 NetVLAD (global feature) 和 FeatLoc (local feature) 随 dynamic complexity 的退化：

- NetVLAD：trans error 从 0.05m 涨到 0.3m，rot error 从 0.02 rad 涨到 0.1 rad——还能用
- FeatLoc：trans error 从 0.1m 涨到 0.8m，rot error 从 0.05 rad 涨到 0.4 rad——直接废了

原因：**local feature (ORB、SIFT 这种) 依赖 keypoint matching**。Moving person 会在 frame 间产生大量 outlier correspondences（人身上的褶皱、纹理都是 false keypoint），RANSAC 都救不回来。Global feature 做的是 image-level aggregated descriptor，对 local perturbation 鲁棒得多。

这跟 SLAM 圈的共识一致：**DROID-SLAM 用 global pose graph 优化比 ORB-SLAM 在 dynamic 下稳得多**。

### 5.4 Trajectory Prediction：见上一节，indoor 直接退化 2x

### 5.5 Robot Navigation：freezing robot problem 重现

这是 paper 最有戏剧性的实验（Table VI）。

三个方法：
- **ORCA**（reaction-based）：经典 reciprocal collision avoidance
- **DS-RNN**（learning-based）：RL 训的 RNN policy
- **AttnGraph**（prediction-based）：先预测 human trajectory，再规划

直觉上 prediction-based 应该最好——你都知道人要往哪走了，绕开不就行了？

**实际结果反过来了**：

| Scene | K=5 SR | K=15 SR | 退化 |
|-------|--------|---------|------|
| Gym AttnGraph | 0.72 | 0.64 | -11.1% |
| Supermarket AttnGraph | 0.46 | 0.35 | -23.9% |
| Gym ORCA | 0.76 | 0.75 | -1.3% |
| Gym DS-RNN | 0.79 | 0.73 | -7.6% |

AttnGraph 退化最严重。Fig. 11 展示了原因——**prediction 的 uncertainty area 被标记为 untraversable**，密集人群下整个空间都是 "uncertain zone"，robot 干脆不动了。

这就是 Trautman & Krause 2010 年提出的 **freezing robot problem**——13 年后 deep learning 时代，prediction-based 方法在密集 indoor 场景下依然没解决。

ORCA 虽然稳，但 ANT（navigation time）爆炸：Gym 从 22.83s 涨到 30.53s（+33.7%）。reaction-based 的代价就是 robot 一直减速避让，效率极低。

DS-RNN 表现最均衡——RL 学到了隐式 human motion pattern，既不 freezing 也不太慢。

这个实验的真正 takeaway：**"predict-then-avoid" paradigm 在 dense dynamic indoor 下失效**。可能的方向是把 prediction uncertainty 直接嵌入 control objective，比如 **diffusion policy + MPC**，让 robot 学会"冒着 uncertainty 也要走"。

参考：
- Freezing Robot Problem 原始 paper: https://en.wikipedia.org/wiki/Freezing_robot_problem
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- DS-RNN: https://github.com/SuiTy/DS-RNN

---

## 六、几个我觉得可以吐槽的点

Karpathy 你肯定能看出几个 paper 没明说的局限：

### 6.1 Real data 占比太小

5,191 real frames vs 84,984 synthetic。Real scene 的 8 个场景每个才 600 多帧，相当于 20 秒 video。这个量级训 detection / segmentation 远远不够，只能做 eval。

如果他们能把 real data 扩到 50k frames 级别（用 HMD 做 hand-held 采集替代 robot，效率高 10x），sim2real 的 value 会大得多。

### 6.2 Sensor suite 太单薄

只有 1 个 RGB-D camera，没 LiDAR、没 multi-cam、没 event camera、没 IMU fusion。

对比 ARKitScenes（LiDAR + RGB-D + IMU）、NuScenes（32-line LiDAR + 6 cam + 5 radar），THUD++ 只能支持 single-sensor 研究。Mobile robot 真实部署基本都 multi-sensor fusion，这个 dataset 用不上。

### 6.3 没提出任何新 algorithm

纯 dataset paper。如果他们顺手提一个 dynamic-aware 3D detector（比如把 object velocity 作为额外的 prediction head）或 obstacle-aware trajectory predictor（把 SDF 作为 input），paper 的 contribution 会厚实一倍。

### 6.4 Navigation test 太小

12m × 12m 的 2D plane，最多 20 humans。真实 deployment 是 100m+ 跨房间 navigation，有电梯、门、stair。这个 testbed 离 real robot 远了。

### 6.5 没有 sim2real transfer 实验

在 synthetic 上 train、real 上 zero-shot eval 的实验完全缺失。这是 dataset paper 应该提供的"用例展示"，否则大家不知道 synthetic data 到底能不能用。

---

## 七、对社区的真正价值

说完吐槽，说回正面价值——**THUD++ 给 mobile robot 圈立了一个 flag**：

> "如果你的 algorithm 在 dynamic complexity = 3+ 的 scene 下 mAP 还掉 20%，那它就还没准备好部署。"

这个 dynamic complexity metric 简单粗暴但好用：

$$D = \frac{1}{N} \sum_{i=1}^{N} n_i^{\text{dyn}}$$

就是每帧平均 dynamic object 数。任何新 algorithm 都可以在 THUD++ 上画一张 "performance vs. dynamic complexity" 曲线，看自己有没有 break point。这个 evaluation paradigm 比 single-number metric 信息量大得多。

而且 Unity3D emulator 开放了，community 可以自己造 scene、造 dynamic pattern，闭环迭代。这点比静态 dataset 强一截。

---

## 八、一句话总结

如果用一句话跟 Karpathy 你说这篇 paper：

**"作者用 90k 帧（90% synthetic）证明了一个 mobile robot 圈一直回避的事实——SOTA perception / prediction / navigation stack 在有人的 indoor 场景下普遍退化 30-50%，freezing robot problem 在 deep learning 时代依然没解决，dynamic scene 是 mobile robot 真正的 hard problem，而不是那些在 static benchmark 上刷点的 task。"**

---

## 参考链接汇总

- THUD++ 主页: https://jackyzengl.github.io/THUD-plus-plus.github.io/
- 原始 ICRA 2024 paper (THUD): https://arxiv.org/abs/2405.10676
- Unity3D: https://unity.com/
- PUDUbot2: https://www.pudurobotics.com/
- Kinect V2 specs: https://developer.microsoft.com/windows/kinect/hardware
- VoteNet / ImVoteNet: https://github.com/facebookresearch/imvotenet
- Social-STGCNN: https://github.com/abduallahmohamed/Social-STGCNN
- AgentFormer: https://github.com/Khrylx/AgentFormer
- DS-RNN / AttnGraph: https://github.com/SuiTy/DS-RNN
- Freezing Robot Problem (Trautman & Krause 2010): https://en.wikipedia.org/wiki/Freezing_robot_problem
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Self-Supervised Depth Denoising: https://arxiv.org/abs/1903.07523

如果你想深入聊 trajectory prediction 怎么 embed obstacle constraint，或者 freezing robot 的 diffusion policy 解法，继续——这俩都是我觉得 paper 留下的最有价值的 open problem。

---

# THUD++ 深度解析：面向移动机器人的大规模动态室内场景数据集

Karpathy 你好！这篇 paper 来自 Tsinghua 团队，是 ICRA 2024 的扩展版本，核心贡献是为 mobile robot 社区提供一个**动态性优先**的 indoor scene understanding benchmark。让我用直觉性的方式拆解这篇 paper 的技术细节、实验设计哲学以及它暴露出的 mobile robot 感知-预测-导航 stack 的深层问题。

---

## 1. Paper 的核心 motivation：为什么需要又一个 indoor dataset？

先看 Table I 的对比逻辑。现有 RGB-D dataset 可以粗略分为两类：
- **2D annotation 阵营**：B3DO (2011, 849 frames)、NYU-Depth v2 (2012, 1449 frames)、SUN3D、Stanford 2D-3D-S、SceneNet RGB-D (5M frames, synthetic)；
- **3D annotation 阵营**：SUN RGB-D、ScanNet (2.5M frames)、SUN-CG、Matterport 3D、InteriorNet (20M frames)、ARKitScenes (5047 scans)、ScanNet++ (1858 scans, 2023)。

一个明显观察：**绝大多数 dataset 的 "Dynamic objects" 列全是 "×"**。也就是说，mobile robot 实际部署场景里**真实存在的 walking people、shopping carts、other moving robots** 这些 entity，在主流 benchmark 里几乎不存在。这正是 THUD++ 想要填的空缺。

THUD++ 的几个关键数字（与现有 dataset 横向比较）：
- 90,175 frames（84,984 synthetic + 5,191 real），平均每帧 **176 个 annotations**（对比 ScanNet 的 10~15，SUN RGB-D 的 20~30），density 极高；
- 20M+ labels，其中 1.2M+ 是 dynamic objects 的 labels；
- 91 个 object classes，包含**长期被忽视的 elevator / glass / stairs / window**等对 robot 危险的类别；
- 8 个 real scenes + 5 个 synthetic scenes，real scene 平均面积 >300 m²；
- 动态对象 mAP 性能显著低于静态对象，验证了 dynamic gap 确实存在。

参考链接：
- THUD++ 项目主页: https://jackyzengl.github.io/THUD-plus-plus.github.io/
- ScanNet: http://www.scan-net.org/
- ScanNet++: https://scannetpp.github.io/
- ARKitScenes: https://github.com/apple/ARKitScenes

---

## 2. Data acquisition pipeline 的工程细节

### 2.1 Real-world 平台：PUDUbot2 + Kinect V2

硬件配置：PUDUbot2 是 Pudu Robotics 的商用 service robot，顶部 V-SLAM 提供 robot pose（xyz 平移 + Euler angle），轮式 odometry 作为补充。pose data 采样 40 fps。

Kinect V2 提供 RGB + depth，**帧率 15-30 fps 波动**（这是 paper 里提到的一个真实痛点，tof 传感器在 motion 中漂移）。

**时间戳对齐的核心 trick**：用 signal board 重叠图像作为 anchor frame，但实测仍有 0-3 frame 的 alignment error。为解决这个问题，作者用 **cubic spline interpolation** 把 40 fps 的 pose data 上采到 2000 fps，然后用最近邻匹配 image timestamp。

Cubic spline 公式（每个区间 [x_i, x_{i+1}] 内）：

$$S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3$$

其中：
- $x$ 是时间戳；
- $x_i$ 是第 i 个控制点的时间；
- $a_i, b_i, c_i, d_i$ 是 spline coefficients，通过相邻区间 $C^2$ 连续性约束求解；
- $a_i = y_i$（function value at $x_i$），$b_i, c_i, d_i$ 通过 tridiagonal linear system 解出。

这个工程细节其实揭示了一个 robot 数据集构建的隐性 cost：**多传感器异步是 mobile robot dataset 的死敌**。这一点在 EuRoC MAV、TartanAir 等数据集里都有类似处理，但 paper 在这里没有强调。

参考链接：
- PUDUbot2: https://www.pudurobotics.com/
- Kinect V2 specs: https://developer.microsoft.com/windows/kinect/hardware
- Self-Supervised Deep Depth Denoising (ICCV 2019): https://arxiv.org/abs/1903.07523

### 2.2 Synthetic 平台：Unity3D + ROS2

模拟器配置（Fig. 3）：
- 虚拟相机高度 1.2 m，pitch = 0°（匹配 real robot 上的 Kinect 位置）；
- RGB/depth resolution = 730 × 530；
- Unity3D 物理引擎驱动 robot motion，同时 ROS2 发布 synthetic data + labels；
- 集成 A* global planner + 个性化 local planner，让 ego-robot 能自主导航采集。

这点很关键：**synthetic data 的 annotation 是从 simulation 引擎直接读取的**，所以每个 frame 的 2D/3D bbox、semantic mask、instance mask 都是完美 ground truth，没有人工标注噪声。这是 synthetic data 的真正优势——**labels 的 precision = 100%**。

集成 Bella robot model 进入虚拟场景做 navigation-based 采集（而不是 random trajectory），这个设计让 synthetic data 的 view distribution 更接近真实 robot deployment。

参考链接：
- Unity3D: https://unity.com/
- Bella robot (AGV): https://www.bellabot.com/

---

## 3. RGB-D 数据集的标注与统计

### 3.1 标注 pipeline

Real data 用**半自动 + 人工修正**：
1. 用 Faster-RCNN（2D）+ VoteNet（3D）做 automatic pre-annotation；
2. 人工 review 修正。

这里隐含一个重要发现：paper 明确说 "algorithms for 3D objects detection are not very effective"——也就是说，**当前 SOTA 3D detector 在 real indoor scene 上预标注都做不到无需人工**，这是 voteNet / ImVoteNet / F-PointNet 在 dynamic 场景下表现不强的早期信号。

Synthetic data 直接从 Unity3D 引擎抽 annotations，包括：
- 2D bounding boxes
- 3D bounding boxes（含 6-DoF pose）
- Semantic segmentation mask
- Instance segmentation mask
- Robot pose
- IMU

### 3.2 数据统计的关键 insight

- **91 个 object classes**——明显比 ScanNet（21）、Matterport3D（40）多，但少于 SceneNet RGB-D（255）；
- 平均每帧 176 labels——密度比 ScanNet 高一个数量级；
- 动态对象 label 总数 1.2M+，包括 pedestrian / robot / shopping cart；
- Fig. 6(a)(c) 显示 real vs synthetic 的 class distribution，可见 synthetic 中 shelf / fridge 类明显多（因为 supermarket scene），real scene 里 chair / table / door 多。

**一个 subtle 点**：paper 没有讨论 synthetic-to-real 的 sim2real gap，只在 canteen / supermarket scene 上分别测试算法（Table III, IV），暗示这两个 distribution 不能直接混训。这是后续工作可扩展的方向。

参考链接：
- VoteNet: https://github.com/facebookresearch/votenet
- Faster-RCNN: https://github.com/rbgirshick/py-faster-rcnn

---

## 4. Trajectory dataset：为什么 ETH/UCY 在 indoor 失效

Table II 列出主流 trajectory dataset：
- **Outdoor**：UCY (786 pedestrians), ETH (750), SDD (11,200)；
- **Indoor**：ATC (shopping centre, 92 days), L-CAS (935 pedestrians, 49 min), THÖR (over 600, 60 min), THUD++ (1257 pedestrians, 60 min, 3 scenes)。

THUD++ 的 3 个 scene：
- Gym: 13 m × 28 m
- Office: 7.8 m × 16.2 m
- Supermarket: 19 m × 32 m

采集方法：Unity3D 中 export pedestrian 坐标，**0.4 秒采样间隔**（即 2.5 Hz，与 ETH/UCY 一致以便对比）。每个 pedestrian 到达目标后立即销毁，避免 stationary 伪 trajectory。

数据格式：每行 `frame_id, pedestrian_id, x, y`，frame_id 步长 10，pedestrian_id 步长 10（这样留出空间插中间的 id）。

**关键直觉**：indoor trajectory 的 difficulty 来自两个 source：
1. **静态障碍物密集** → pedestrian 必须绕行，trajectory 不是 smooth curve；
2. **社交互动频繁** → stop-and-talk、face-to-face 会让 velocity 突变。

这两点在 outdoor dataset（ETH/UCY 的开阔广场）里几乎不存在。所以 paper 在 Table V 里观察：所有 trajectory prediction 方法在 ETH 上 ADE=0.74~1.08，迁移到 Supermarket 上 ADE=1.55~1.81，**性能下降 50-70%**。

参考链接：
- ETH/UCY dataset: https://data.vision.ee.ethz.ch/cvl/aess/dataset/
- SDD: https://cvgl.stanford.edu/projects/uav_data/
- THÖR dataset: https://github.com/srl-freiburg/thor_dataset

---

## 5. 五个 benchmark 任务的实验细节与公式

### 5.1 3D Object Detection（Table III）

测试方法：F-PointNet、ImVoteNet、DeMF。

mAP 定义：
$$\text{mAP} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \text{AP}_c, \quad \text{AP}_c = \int_0^1 P_c(R) \, dR$$

其中 $P_c(R)$ 是 class c 在 recall = R 时的 precision，$|\mathcal{C}|$ 是 class 数。

IoU_3D 阈值通常取 0.25 (indoor standard)。

**实验结论**：
- Supermarket（dynamic complexity = 0.94）：DeMF 在 dynamic 上 34.51 mAP，static 上 38.24，gap = 3.73；
- Canteen（dynamic complexity = 3.34）：DeMF dynamic 28.56，static 45.43，**gap = 16.87**。

这说明：**dynamic complexity 越高，static-dynamic 的 mAP gap 越大**。这背后的根本原因是 3D detector 假设 object 在 frustum 内是 quasi-static 的，而 moving pedestrian 会在 multi-frame aggregation（如 point cloud accumulation）中产生 motion blur / ghost。

DeMF (object-focused image fusion) 表现最好，因为它显式地将 image feature 注入 point cloud branch，对 dynamic object 的 partial observation 更鲁棒。

参考链接：
- F-PointNet: https://github.com/charlesq34/frustum-pointnets
- ImVoteNet: https://github.com/facebookresearch/imvotenet
- DeMF: https://arxiv.org/abs/2207.10589

### 5.2 Semantic Segmentation（Table IV）

测试方法：ACNet (3×R50)、RedNet (2×R34)、ESANet (2×R34)、SA-Gate (2×R101)。

MIoU：
$$\text{MIoU} = \frac{1}{C} \sum_{c=1}^{C} \frac{|P_c \cap G_c|}{|P_c \cup G_c|}$$

其中 $P_c$ 是预测的 class c 像素集合，$G_c$ 是 GT 的 class c 像素集合，$C$ 是 class 总数。

**实验关键观察**：
- Synthetic Supermarket MIoU = 74.83~83.19
- Real Canteen MIoU = 51.85~65.97
- **Real scene 性能普遍低 20+ 点**——sim2real gap 非常明显，主要原因：
  1. Real scene 有 specular reflection（玻璃、地板反光）；
  2. Lighting variation；
  3. Sensor noise（Kinect V2 depth hole）。

但 paper 也做了一个有趣的对照实验：ESANet 在 Supermarket 上有/无 dynamic objects 训练，MIoU 78.42% vs 79.63%——**几乎无差异**。这说明 semantic segmentation 本身对 dynamic 不敏感（pixel-level label 与 motion 无关），dynamic object 影响的是 detection / relocalization / prediction 这种依赖几何一致性的 task。

参考链接：
- ACNet: https://arxiv.org/abs/1905.13545
- RedNet: https://arxiv.org/abs/1806.01054
- ESANet: https://github.com/TUI-NICRobotik/ESANet
- SA-Gate: https://github.com/Xiaoqi-Zhao-SA-Gate/SA-Gate

### 5.3 Robot Relocalization（Fig. 9）

测试方法：NetVLAD（global feature）、FeatLoc（local feature）。

误差度量：
$$e_{\text{trans}} = \|\mathbf{t}_{\text{pred}} - \mathbf{t}_{\text{gt}}\|_2, \quad e_{\text{rot}} = 2 \arccos\left(\frac{\text{tr}(\mathbf{R}_{\text{pred}}^\top \mathbf{R}_{\text{gt}}) - 1}{2}\right)$$

其中 $\mathbf{t}$ 是 3D 平移向量，$\mathbf{R}$ 是 3×3 rotation matrix。

Scene dynamic complexity 定义：
$$D = \frac{1}{N} \sum_{i=1}^{N} n_i^{\text{dyn}}$$

其中 $N$ 是总帧数，$n_i^{\text{dyn}}$ 是第 i 帧中 dynamic pedestrian 数量。

**关键观察（Fig. 9）**：
- 两种方法 error 都随 dynamic complexity 单调上升；
- **FeatLoc (local feature) 退化更严重**，因为它依赖 keypoint matching，而 moving pedestrian 会在 frame 间产生大量 outlier correspondences；
- NetVLAD (global feature) 更鲁棒，因为它做的是 image-level aggregated descriptor，对 local perturbation 不敏感。

这个结论与 SLAM 文献里 ORB-SLAM 在 dynamic scene 上漂移、DROID-SLAM 用 global pose graph 优化的现象一致。

参考链接：
- NetVLAD: https://github.com/RelAR/NetVLAD
- FeatLoc: https://arxiv.org/abs/2203.13770
- Dynamic SLAM survey: https://arxiv.org/abs/2204.08524

### 5.4 Pedestrian Trajectory Prediction（Table V, Fig. 10）

测试方法：Social-GAN、Social-STGCNN、PECNet。

ADE / FDE 定义：
$$\text{ADE} = \frac{1}{T_{\text{obs}} + T_{\text{pred}}} \sum_{t=1}^{T_{\text{pred}}} \|\hat{\mathbf{p}}_t - \mathbf{p}_t\|_2, \quad \text{FDE} = \|\hat{\mathbf{p}}_{T_{\text{pred}}} - \mathbf{p}_{T_{\text{pred}}}\|_2$$

其中 $\hat{\mathbf{p}}_t = (\hat{x}_t, \hat{y}_t)$ 是 t 时刻预测位置，$\mathbf{p}_t$ 是 GT 位置，$T_{\text{pred}}$ 是预测 horizon。

**核心发现（Table V）**：
- ETH (outdoor): Social-STGCNN ADE=0.74, FDE=1.48（最好）
- Supermarket (indoor): Social-STGCNN ADE=1.55, FDE=2.91（退化 2x）
- Office: ADE=1.36（更窄，更难）
- Gym: ADE=1.18（最开阔的 indoor scene）

直觉解释：
1. **Spatial constraint**：indoor 的 narrow corridor / aisle 让 pedestrian 必须绕开 obstacle，trajectory 不是 smooth Bezier，而是 piecewise linear with sharp turns。Social-STGCNN 的 graph convolution 假设 interaction 是 smooth pairwise repulsion，与现实不符。
2. **Interaction density**：indoor pedestrian 之间频率更高的 face-to-face / stop-and-talk 让 velocity 分布双峰化（either walking or stopped），L2 loss 的 unimodal 预测会失败。这是为什么 PECNet（endpoint conditioned）在某些场景反而退化——endpoint 在 indoor 不一定是 destination，可能是中途 stop 点。

Social-STGCNN 在所有 scene 上最好，因为它显式建模 spatio-temporal graph，对 multi-agent interaction 更敏感。但 Fig. 10 的红色框显示，**在 obstacle 密集处预测 trajectory 会直接撞墙**——这说明现有方法没有把 static obstacle 作为 hard constraint 嵌入 prediction。

参考链接：
- Social-GAN: https://github.com/agrimgupta92/sgan
- Social-STGCNN: https://github.com/abduallahmohamed/Social-STGCNN
- PECNet: https://github.com/HarshayuGirase/PECNet

### 5.5 Robot Navigation（Table VI, Fig. 11）

测试方法：ORCA (reaction-based)、DS-RNN (learning-based)、AttnGraph (prediction-based)，均搭配 A* global planner。

评估指标：
- **SR (Success Rate)**：到达目标的比例；
- **CR (Collision Rate)**：与人发生碰撞的 episode 比例；
- **ANT (Average Navigation Time)**：成功 episode 的平均时间（秒）；
- **SPE (Social Path Efficiency)**：
$$\text{SPE} = \frac{L_{\text{actual}}}{L_{\text{A}^*}}$$
其中 $L_{\text{actual}}$ 是 robot 实际走过的 path length，$L_{\text{A}^*}$ 是 A* 算的无障碍 path length。SPE > 1 表示 robot 绕路了。

训练环境：12 m × 12 m 2D plane，最多 20 humans，human 由 ORCA 控制，max speed 0.5-1.5 m/s，radius 0.3-0.5 m。robot 5 m 圆形 sensing range，max speed 1 m/s。

**关键发现（Table VI）**：

| Scene | Pedestrians | ORCA SR | DS-RNN SR | AttnGraph SR |
|-------|-------------|---------|-----------|--------------|
| Gym | K=5 | 0.76 | 0.79 | 0.72 |
| Gym | K=10 | 0.74 | 0.77 | 0.69 |
| Gym | K=15 | 0.75 | 0.73 | 0.64 |
| Supermarket | K=5 | 0.37 | 0.44 | 0.46 |
| Supermarket | K=15 | 0.30 | 0.43 | 0.35 |

**两大 insight**：

1. **Supermarket SR 远低于 Gym**——因为 Supermarket 有货架等 static obstacle，让可通行区域被进一步压缩，robot 几乎无法找到 collision-free path。

2. **AttnGraph (prediction-based) 在高密度下退化最严重**（Gym: 0.72 → 0.64, -11.1%；Supermarket: 0.46 → 0.35, -23.9%）。原因是 trajectory prediction-based 方法会**把 prediction 的 uncertainty area 标记为 untraversable**，导致 robot 陷入 **"freezing robot problem"**（Fig. 11）。这是 Trautman & Krause 2010 在 IJRR 经典 paper 早就提出的问题，THUD++ 在更密集的 indoor 场景下重现了它。

3. **ORCA 的 ANT 增长最显著**（Gym: 22.83 → 30.53, +33.7%）——ORCA 是反应式方法，在高密度下会频繁减速避让，导致 navigation time 爆炸。

4. **DS-RNN 最稳定**——RNN 隐式学到了 human motion pattern，对 dynamic 复杂度有更好的 scaling。

参考链接：
- ORCA (RVO2 library): https://github.com/sybrenstuvel/orca
- DS-RNN: https://github.com/SuiTy/DS-RNN
- AttnGraph: https://github.com/SuiTy/AttnGraph
- Freezing Robot Problem (Trautman): https://en.wikipedia.org/wiki/Freezing_robot_problem

---

## 6. 与相关 dataset 的对比直觉

| Dataset | 主优势 | 主劣势 |
|---------|--------|--------|
| ScanNet / ScanNet++ | High-quality 3D mesh | 几乎全 static |
| Matterport 3D | Panorama + 多视角 | 静态为主 |
| ARKitScenes | Real AR 应用 | low annotation density |
| SceneNet RGB-D | 5M synthetic frames | 无 dynamic |
| InteriorNet | 20M frames + IMU | sim2real gap |
| ETH/UCY | trajectory prediction baseline | outdoor only |
| THÖR | indoor trajectory | 1 scene, 600 pedestrians |
| **THUD++** | **dynamic + multi-task + sim+real** | synthetic 占比高 |

THUD++ 的真正独特性在于**同时覆盖 perception + prediction + navigation 三层 stack**，并用 dynamic complexity 作为统一 axis 量化每个 task 的 degradation。这种**端到端 benchmark 设计**对 robot learning community 有重要价值。

参考链接：
- InteriorNet: https://interiornet.org/
- Matterport 3D: https://matterport.com/

---

## 7. 我对 paper 的几点批评性思考

### 7.1 优点
1. **Dynamic complexity metric** (D = avg #dynamic objs/frame) 是一个简洁好用的 axis，可以推广到其他 dataset；
2. **半自动标注 pipeline** 揭示了 SOTA 3D detector 的弱点，是 dataset paper 里少有的 honest disclosure；
3. **Unity3D emulator + user-friendly interface** 让 community 可以闭环迭代 scene 配置；
4. **5 个 task 的统一 evaluation** 让 dataset 价值最大化。

### 7.2 不足 / 可改进方向

1. **Real data 占比小**：5,191 real frames vs 84,984 synthetic，real scene sim2real 价值被稀释。可以参考 Habitat-Matterport 3D 的 mesh-based real 数据生成 pipeline 增量。

2. **没有提出新的 algorithm**：paper 是 pure dataset paper，没有提出针对 dynamic scene 的新 method。比如一个 dynamic-aware 3D detector、或一个 obstacle-aware trajectory predictor 会让贡献更完整。

3. **Sensor suite 偏简单**：只有 1 个 RGB-D camera，没有 LiDAR / multi-camera / event camera。对比 ARKitScenes (LiDAR + RGB-D), 限制了 sensor fusion 研究价值。

4. **缺乏 long-horizon navigation**：navigation test 只在 12×12 m 区域，没有 long-horizon multi-room navigation 任务，这是 real mobile robot deployment 的关键挑战。

5. **Trajectory dataset 不含 trajectory-conditional context**：只有 (x, y) 坐标，没有 head orientation / gaze / social group label，对 social-aware prediction（如 Social-BiGAT, AgentFormer）支持不足。

6. **没有 sim2real transfer 实验**：在 synthetic 上训练，在 real 上 fine-tune / zero-shot evaluation 的实验缺失，这是 dataset paper 应该提供的"用例展示"。

参考链接：
- Habitat-Matterport 3D: https://aihabitat.org/datasets/hm3d/
- AgentFormer: https://github.com/Khrylx/AgentFormer
- Social-BiGAT: https://github.com/HarshayuGirase/Social-BiGAT

---

## 8. 对 mobile robot 学习研究的启示

Karpathy 你在多个 lecture 里强调过 "data is the bottleneck"。这篇 paper 印证了这一点：

1. **Dynamic scene 是 mobile robot 的真实 deployment regime**——但 SOTA 在 dynamic 下的退化普遍 > 30%，说明当前 perception-prediction-navigation stack 的 robustness 远未饱和。

2. **Synthetic data 的真正价值在 dynamic scene**：因为 real scene 的 dynamic annotation 极其昂贵（要在每帧手工标 moving person 3D bbox），synthetic 是唯一可 scale 的方案。THUD++ 90% 是 synthetic 是合理选择。

3. **Freezing robot problem 在 deep learning 时代仍未解决**——AttnGraph 这种 prediction-based 方法在高密度 indoor 反而更差，说明 naive "predict-then-avoid" paradigm 在 dense dynamic 下失效。一个可能方向是 **learning-based MPC + diffusion policy for crowd navigation**，把 prediction uncertainty 直接嵌入 control objective。

4. **Trajectory prediction 需要 obstacle-aware design**：当前方法把 obstacle 当作 zero-velocity agent 是不对的，需要 explicit environmental encoding（如 occupancy map / signed distance field 作为 input）。可以参考 SoPhie、AgentFormer 的 environment encoding 设计。

5. **3D detection 在 dynamic 下的 degradation** 暗示需要 **temporal-aware 3D detector**——把 multi-frame observation 作为 4D input，类似 4D LiDAR detection (CenterPoint4D, FF-Track) 的设计。

参考链接：
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- CenterPoint 4D: https://arxiv.org/abs/2108.10736
- SoPhie: https://arxiv.org/abs/1910.02308
- AgentFormer: https://arxiv.org/abs/2103.14023

---

## 9. 总结

THUD++ 是一个**定位明确**的 dataset：填补 mobile robot 在 dynamic indoor scene 的 benchmark 空白。它的最大贡献在于：

- 用 dynamic complexity 作为统一 axis，量化 perception / prediction / navigation 三层 stack 的退化；
- 用 Unity3D + ROS2 提供 closed-loop evaluation framework；
- 暴露 SOTA 在 dense dynamic indoor 下的 failure mode（freezing robot、trajectory hitting wall、3D detector mAP 大幅下降）。

它的局限在于：real data 比例小、sensor suite 单一、没有提出新 algorithm、sim2real 实验缺失。

但从 **dataset-as-benchmark** 的角度，THUD++ 给 mobile robot community 提供了一个清晰的 challenge：**如果你的算法在 dynamic complexity = 3+ 的 scene 下 mAP 仍掉 20%，那它就还没准备好部署**。

希望对你 build intuition 有帮助！如果你想深入讨论 trajectory prediction 的 obstacle-aware design 或 freezing robot 的 deep learning 解法，可以继续。

参考链接（综合）：
- THUD++ 项目主页: https://jackyzengl.github.io/THUD-plus-plus.github.io/
- 作者团队 (Tsinghua): https://www.tsinghua.edu.cn/en/
- ICRA 2024 conference: https://www.icra2024.org/
- 之前的 conference paper (THUD ICRA 2024): https://arxiv.org/abs/2405.10676
