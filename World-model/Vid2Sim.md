---
source_pdf: Vid2Sim.pdf
paper_sha256: 83e536ea706ae07d60e9e8d72720209c492933c9b3872071cb78cc2487cfa7a9
processed_at: '2026-08-13T00:31:43-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 Vid2Sim

## 这篇 paper 到底在干什么

一句话：**拿一个手机拍 15 秒视频，吐出一个能跑 RL 的真实感仿真环境，训出来的 policy 直接零样本上真机器人**。

就这么简单。中间的所有公式、loss、trick，都是为了让这件事真的 work。

---

## sim2real gap 为什么这么烦

你训了一个 RL agent，它在 simulator 里跑得贼顺，一到真实世界就废。原因基本就两条：

**第一条：agent 在 sim 里看到的东西跟现实差太远**。传统 simulator（Habitat [Savva et al., ICCV 2019](https://arxiv.org/abs/1904.01201)、MetaDrive [Li et al., PAMI 2022](https://arxiv.org/abs/2109.12674)、Isaac Gym [Makoviychuk et al., 2021](https://arxiv.org/abs/2108.10470)）用的是手画 asset + 传统 rasterization，纹理是程序生成的、光照是 fake 的、植被就是几个绿色三角形。agent 学的是这套"假图像"上的 visual feature，一到真实街道就 OOD。

**第二条：物理对不上**。sim 里的 friction、robot inertia、wheel slip 跟真实机器人不一样，agent 学的动作分布 transfer 不过去。

domain randomization [Tobin et al., IROS 2017](https://arxiv.org/abs/1703.06907) 想解决这两条——把 sim 里的纹理、光照、物理参数全部随机化，逼 agent 学一个 robust policy。问题是 randomize 的范围太广了，agent 学到的是"对任何奇葩情况都凑合"的平庸策略，而不是"对真实世界精确适配"的好策略。system identification [Peng et al., ICRA 2018](https://arxiv.org/abs/1710.06537) 只能搞定物理参数，搞不了 visual gap。

Vid2Sim 的赌注：**与其在假 sim 里随机化，不如直接用真实视频重建 sim**。这样 visual gap 几乎消失，物理 gap 用 mesh + Unity 标准化解决。

---

## 核心 trick：GS 负责"看"，mesh 负责"撞"

3DGS [Kerbl et al., SIGGRAPH 2023](https://arxiv.org/abs/2308.14737) 重建出来的场景很真实，但它就是一堆 Gaussian ellipsoid，没有"墙是硬的"这种概念。你把 agent 放进去，它穿过墙都没人知道。

mesh 有 collision 信息，但 mesh 的纹理渲染质量上不去——Video2Game [Xia et al., CVPR 2024](https://arxiv.org/abs/2312.07534) 试过这条路，PSNR 只有 28.32，跟 GS 的 31.85 差了 3.5 dB，看起来就是塑料质感。

Vid2Sim 说：**干嘛要选？两个都用**。

```
       Agent 看到的画面            Agent 撞到的东西
            │                          │
            ▼                          ▼
      ┌─────────┐                ┌──────────┐
      │   GS    │                │   Mesh   │
      │ raster  │                │ collider │
      │ shader  │                │ (不可见) │
      └─────────┘                └──────────┘
            │                          │
            └──────────┬───────────────┘
                       ▼
                  Unity Physics
```

GS 渲染 RGB 和 depth 给 agent 看，mesh 当 invisible collider 给 Unity 物理引擎算碰撞。agent 感觉上是在真实街道里走，撞墙了 Unity 立刻给它反馈。这套 hybrid 设计是整个 paper 最聪明的地方——把 GS 和 mesh 各自的强项拼起来，避开各自的弱点。

---

## 几个关键 design choice 的 intuition

### 1. 为什么不直接用 monocular depth estimator 的输出做监督？

Depth Anything V2 [Yang et al., 2024](https://arxiv.org/abs/2406.09414) 输出的是 **relative depth**，不是 metric depth。它知道"前面这个点比后面那个点近 2 倍"，但它不知道"前面这个点离我 3.2 米"。

3DGS 经过 SfM 初始化后也有自己的 scale，但这个 scale 是 arbitrary 的——SfM 本来就是 scale-ambiguous 的。

如果你直接 minimize $\|\hat{D} - D\|$，相当于强行让 GS 的 scale 跟 depth estimator 的 scale 对齐。这两个 scale 都是 up-to-scale 的，你强制对齐的结果就是优化在两个 scale 之间来回扯，训不稳。

Vid2Sim 的解法：**用 patch-based NCC**（公式 4）。只看局部 patch 里"高低起伏的模式"对不对，不看绝对数值。NCC 数学上是 scale-invariant 的——你把 GS 的 depth 乘个常数，NCC 不变。这就把 scale 冲突问题彻底绕过去了。

这是非常 elegant 的工程判断。学术上不新（NCC 在 stereo matching 里用了几十年），但放在 GS + monocular prior 的语境里，刚好对症。

### 2. Geometry-Consistent Loss 在干嘛？

公式 6 看起来复杂，其实就一句话：**相邻 pixel 在 depth 平滑区域的 normal 应该一致**。

$$
\mathcal{L}_{\text{geo}} = \frac{\sum w_{ij}(1 - \hat{N}_{ij} \cdot \hat{N}_{i+\Delta x, j+\Delta y})}{\sum w_{ij}}, \quad w_{ij} = 1 - \|\nabla D_{ij}\|
$$

$w_{ij}$ 是个权重——depth 梯度小的地方权重大（接近 1），depth 边缘权重小（接近 0）。这就保证了 loss 只在"平滑区域"生效，不会在墙角、物体边缘乱惩罚。

为什么需要这个？因为 3DGS 训练时容易在平滑表面长出一堆"小噪声 Gaussian"——photometric loss 不关心几何，只关心颜色对不对。加了这个 loss，相邻 pixel 的 normal 被强制一致，那些小噪声 Gaussian 就会被压平。

这个 trick 跟 2DGS [Huang et al., 2024](https://arxiv.org/abs/2403.17888) 的 disk regularization 配合，把 GS 从"一堆球"压成"一堆片"，几何质量大幅提升。

### 3. Screen-Space Covariance Culling 为什么必要？

RL agent 训练时会到处乱跑，camera 位置和角度跟训练 view 偏差很大。这时候原本从训练 view 看起来很正常的 Gaussian，投影到 agent view 里会变成一大块糊在画面上的"floater"。

agent 看到 floater 会以为前面有障碍物，绕着走——policy 就废了。

culling 的逻辑（公式 9）：如果一个 Gaussian 投影到 2D 后覆盖面积超过 image 的 $\alpha$ 倍，直接不渲染。这是一个纯 heuristic——不修改 Gaussian 参数，只在 rasterization 时跳过那些"投影过大"的 splat。

Table 4 显示 FID 从 214.47 降到 191.54，提升 10.7%。这种 trick 写在 paper 里看起来不起眼，但实际工程里是 make-or-break 的细节。

### 4. Hybrid representation 的 ground removal 怎么做的？

从 GS 渲染 depth → TSDF 融合 → marching cubes 出 mesh。但 mesh 里地面会有起伏（因为 monocular depth 在地面区域估计不准），这种起伏会让 robot 在 sim 里乱颠。

直接用 SAM [Kirillov et al., 2023](https://arxiv.org/abs/2304.02643) 删地面不可靠——SAM 对 ground 的 segmentation 模糊。Vid2Sim 用 normal prior（公式 8）：

$$
\mathcal{M}_i = \|\arccos(N_i \cdot \bar{n}_i') < 15°\|
$$

先用 SAM 给个粗的 ground region，算出该 region 的平均法线（基本上是 up 方向），然后任何法线与这个方向夹角小于 15° 的 pixel 都算 ground。这是个很实用的 trick——把 SAM 的粗 mask 用几何 prior 精细化。

---

## 实验结果说明了什么

### Table 1：重建质量

Vid2Sim PSNR 32.41，比 3DGS 高 0.56 dB，比 Video2Game 高 4 dB。更重要的是 Table 1 右半边——只有 Vid2Sim 同时满足 real-time + interactive + RL trainable。其他方法要么不能交互（3DGS、2DGS），要么视觉太差（Video2Game），要么根本不能做 sim（Instant-NGP）。

### Table 2：sim 内 navigation

这是最关键的对比。同一个任务（PointNav），同样用 RGB observation：
- Mesh-based sim（模拟 Video2Game 的 setting）：48.8% SR
- Vid2Sim：81.6% SR

**32.8 个百分点的提升**。这就是 sim2real gap 在 sim 内的量化——photorealistic 渲染对 navigation policy 的影响巨大。

更反直觉的是：加 obstacle 反而提升 SR（No Obj 68.8% → Dynamic 81.6%）。原因是 obstacle 让任务更"密集"，agent 学到的 avoidance 行为更 generalizable，而不是只在空荡荡的街上傻走。

### Table 3：sim2real

这是 paper 最震撼的表。

Mesh-based baseline 在真实世界 **0% SR**——三个任务全废。这就是 sim2real gap 的现实——你的 agent 在 mesh sim 里训得再好，真实世界的视觉分布完全不一样，policy 直接瘫痪。

Vid2Sim 用 30 个 env 训练：Go Straight 85%、Static Obstacle 65%、Dynamic Obstacle 55%。**全部 zero-shot deploy，没有 fine-tune**。

更关键的趋势是 scaling：1 env → 0% / 30% / 0%；5 env → 60% / 40% / 0%；30 env → 85% / 65% / 55%。env 越多，sim2real 越好。这说明 Vid2Sim 的 pipeline 是 scalable 的——如果能搞到 1000 个 web video，可能真的能训出 general navigation foundation policy。

---

## 我的 take

这篇 paper 的 contribution 不是某个单点突破，而是**把一堆已知技术拼成一个完整可用的 pipeline**。GS、TSDF、NCC、SAC、Unity，每个组件都是现成的，但拼起来 work 这件事本身是 contribution。

它的真正价值在于证明了一个 scaling argument：**用真实视频重建 sim 这条路是 work 的，而且 env 越多越好**。这跟 LLM scaling law 的逻辑一样——瓶颈在 data pipeline 的 cost，不在算法。

YouTube 上 walking tour video 海量，如果有人把 Vid2Sim 的 pipeline 工程化到能自动处理 10000 个 video，embodied navigation 的 foundation policy 可能真的就出来了。这跟 Tesla 用行车记录仪数据训 FSD 的逻辑类似，只是多了一个 3D reconstruction 的中间步骤。

几个我觉得值得继续挖的方向：

1. **Metric depth 替代 SSI loss**：Depth Anything V2 的 metric 版本 [Yang et al., 2024](https://arxiv.org/abs/2406.09414) 如果够准，可以直接用 L1 loss，省掉 NCC 的麻烦。前提是 GS 的 SfM 初始化也要对齐 metric scale（可能要加 GPS prior）。

2. **Online reconstruction**：现在 pipeline 是 offline 的——拍完视频、跑 GLOMAP、训 GS、提 mesh、配 Unity。如果能像 SplaTAM [Vedantam et al., 2024](https://arxiv.org/abs/2403.02751) 那样 online 重建，就能边走边建 sim，对 robot navigation 更有意义。

3. **更多 embodiment**：paper 只做了 wheeled robot。legged robot、humanoid 的 dynamics 更复杂，但 hybrid representation 的思路完全可以 extend——GS 给视觉，URDF/mesh 给 collision。

4. **GS 的 rendering 极限**：paper 里 culling 是个 patch，真正解决 extreme view rendering 还是要靠 [Mip-Splatting](https://arxiv.org/abs/2312.00201) 这类 anti-aliasing 工作。Vid2Sim 的 culling 是 cheap fix，不是 fundamental solution。

5. **Generative augmentation 的边界**：现在用 Instruct-4D-to-4D 做风格化，但生成的图像分布跟真实世界还是有 gap。如果用 diffusion model 直接生成 diverse scene 而不依赖重建，可能更 scalable——这其实就是 [LucidSim](https://arxiv.org/abs/2411.17448) 的思路，但它牺牲了 3D consistency。

---

## 总结

用人话讲，Vid2Sim 干了三件事：

1. **把 GS 从"看的东西"变成"能跑 RL 的东西"**——加 geometry supervision 让它重建得准，加 culling 让它在 agent view 下不崩，加 mesh 让它能碰撞。
2. **用真实视频当 sim 的 source**——绕开了传统 simulator 的 asset 限制，scaling 的瓶颈变成"有多少 web video"。
3. **证明这套 pipeline 能 zero-shot sim2real**——30 个 env 就能让 agent 在真实世界 55% 成功率避让行人，这数字不大但足以说明方向对。

工程上很多细节可以挑刺（GLOMAP 慢、30 env 少、只测了 wheeled robot），但作为一个 proof-of-concept，它把"video → interactive sim → real robot"这条 chain 跑通了。接下来就是 scaling 和工程优化的事。

如果你想我再展开某个具体点——比如 NCC 为什么对 monocular depth 监督特别合适、或者 SAC 在 navigation task 上的 reward shaping 为什么这么设计、或者 hybrid representation 在 Unity 里的具体 shader 实现——直接说。

---

# Vid2Sim: 从单目视频到可交互仿真环境的深度解析

## 1. Paper 核心问题与 motivation

这篇 paper 想解决的核心痛点是 **sim2real gap** 在 robot learning 中的长期难题。传统方案如 domain randomization [Tobin et al., IROS 2017](https://arxiv.org/abs/1703.06907) 与 system identification [Peng et al., ICRA 2018](https://arxiv.org/abs/1710.06537) 都受限于 simulator 本身能渲染什么、能模拟什么物理。MetaDrive [Li et al., PAMI 2022](https://arxiv.org/abs/2109.12674)、Habitat [Savva et al., ICCV 2019](https://arxiv.org/abs/1904.01201)、Isaac Gym [Makoviychuk et al., 2021](https://arxiv.org/abs/2108.10470) 这类 simulator 的 asset library 有限，rendering pipeline 又是传统 rasterization，难以复现 real world 中复杂的 urban scene（斑驳的纹理、植被、光照变化、动态物体）。

NeRF [Mildenhall et al., ECCV 2020](https://arxiv.org/abs/2003.08934) 和 3DGS [Kerbl et al., SIGGRAPH 2023](https://repo.samplify.org/kerbl_3dgaussians) 给了一个"从真实视频直接重建 photorealistic 3D 场景"的可能，但这类工作大多止步于 novel view synthesis（NVS），没有 physical interaction。Video2Game [Xia et al., CVPR 2024](https://arxiv.org/abs/2312.07534) 把 NeRF 嫁接到 game engine 上做交互游戏，但它的 visual 来自 textured mesh，fidelity 上不去，更关键的是它面向 game development，没有为 embodied agent 的 closed-loop training 设计。

Vid2Sim 的 core insight：**用 GS 负责"看"，用 mesh 负责"撞"**。一个 monocular video 同时驱动两套表征——GS 给 agent photorealistic RGB/depth observation，mesh 给 physics engine 做 collision detection。这样 RL agent 训练时看到的图像接近 real world，碰撞反馈又来自几何准确的 mesh。

---

## 2. Pipeline 总览（Figure 2 解析）

整个 Vid2Sim 分为三个 stage：

**Stage 1: Geometry-Consistent Scene Reconstruction**
- 输入：handheld monocular video（15s @ 30fps，约 450 帧）
- 用 GLOMAP [Pan et al., ECCV 2024](https://arxiv.org/abs/2407.01991)（比 COLMAP [Schönberger & Frahm, CVPR 2016](https://www.cvlibs.net/publications/Schonberger2016CVPR.pdf) 对 in-the-wild video 更鲁棒）做 SfM 得到 camera pose 与 sparse point cloud
- 用 DEVA tracker [Cheng et al., ICCV 2023](https://arxiv.org/abs/2305.05578) mask 掉 dynamic objects
- 训练 geometry-regularized 3DGS（核心创新点）
- Screen-space covariance culling 清理 floater

**Stage 2: Realistic & Interactive Simulation**
- 从 GS 渲染 depth map，用 KinectFusion/TSDF [Curless & Levoy, 1996](https://www.cs.cmu.edu/~kingr/MDR/615liu/curless-levoy-1996.pdf) 融合出 mesh
- 用 normal prior 剔除 ground plane（公式 8）
- 导入 Unity [Unity Engine](https://unity.com/)，写一个 custom shader 实时 rasterize GS，mesh 设为 invisible 但作为 collider
- 插入 static obstacles（traffic cone、trash bin 等）+ dynamic agents（A* planning 的 pedestrian）
- Scene augmentation：风格化（Instruct-4D-to-4D [Mou et al., CVPR 2024](https://arxiv.org/abs/2306.08904)）、天气粒子（rain/fog/snow，参考 ClimateNeRF [Li et al., ICCV 2023](https://arxiv.org/abs/2303.17926)）

**Stage 3: Sim2Real Validation**
- SAC [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290) 训 1.5M steps，30 个 parallel env，单卡 A5000 大约 15 小时
- 直接 zero-shot deploy 到真实 four-wheeled delivery robot

---

## 3. 3DGS Preliminary（Section 3 公式解析）

### 公式 (1)：3D Gaussian 定义

$$
\mathcal{G}_i(\mathbf{x}) = \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_i)^T \Sigma_i^{-1} (\mathbf{x} - \boldsymbol{\mu}_i) \right)
$$

- $\mathbf{x} \in \mathbb{R}^3$：query 点的空间坐标
- $\boldsymbol{\mu}_i \in \mathbb{R}^3$：第 $i$ 个 Gaussian 的中心（mean）
- $\Sigma_i \in \mathbb{R}^{3 \times 3}$：3D covariance matrix，决定 Gaussian 的"椭球形状"

为了优化时保证 $\Sigma_i$ positive semi-definite，不直接优化 $\Sigma_i$ 而是参数化为 $\Sigma_i = R_i S_i S_i^T R_i^T$，其中 $S_i = \text{diag}(s_1, s_2, s_3)$ 是 scaling，$R_i$ 是 rotation（用 quaternion 表示）。每个 Gaussian 还有 opacity $o_i \in [0,1]$ 和 color $\mathbf{c}_i$（用 spherical harmonics 系数编码 view-dependent 颜色）。

### 公式 (2)：Volumetric Alpha Blending

$$
\mathbf{c}(x) = \sum_{i \in N} T_i \mathbf{c}_i \alpha_i(\mathbf{x}), \quad T_i = \prod_{i=1}^{i-1}(1 - \alpha_i(\mathbf{x}))
$$

- $N$：覆盖该 pixel 的 Gaussian 集合（按 depth 排序）
- $\alpha_i(\mathbf{x}) = o_i \cdot \mathcal{G}_i(\mathbf{x})$：第 $i$ 个 Gaussian 在该 pixel 的 alpha
- $T_i$：transmittance，前面所有 Gaussian 没拦住的比例
- $\mathbf{c}_i$：第 $i$ 个 Gaussian 在该 viewing direction 下的颜色

这本质就是 front-to-back 的 over operator。

### 公式 (3)：Depth 与 Normal Rendering

$$
\hat{\mathbf{D}}(x) = \sum_{i \in N} T_i \mathbf{d}_i \alpha_i(\mathbf{x}), \quad \hat{\mathbf{N}}(x) = \sum_{i \in N} \hat{\mathbf{n}}_i \alpha_i T_i
$$

- $\mathbf{d}_i$：第 $i$ 个 Gaussian 中心到 camera 的距离
- $\hat{\mathbf{n}}_i$：基于 $\Sigma_i$ 最短轴方向的 normal（直觉：扁的 disk 法线就是它的"厚度方向"）

这里有个关键 intuition：3DGS 原版渲染的 depth 是 alpha-weighted 的 median depth，而 normal 是从 covariance 的最短轴推导的——也就是说 Gaussian 越像 2D disk，normal 越准。这正是后面 $\mathcal{L}_{\text{scale}}$ 要惩罚最短轴长度的原因。

---

## 4. Geometry-Consistent Reconstruction（Section 4.1）

### 4.1 为什么需要 Scale-Invariant Loss？

这里有一个非常重要的细节，paper 里没完全讲清楚。一般的 monocular depth estimator（如 Depth Anything V2 [Yang et al., 2024](https://arxiv.org/abs/2406.09414)）输出的是 **relative/affine-invariant depth**，不是 metric depth。而 3DGS 经过 SfM 初始化后有一个**自己的、任意的 scale**（SfM 本身是 scale-ambiguous 的，除非有 GPS/IMU）。

如果直接用 SSI (scale-shift-invariant) loss [Birkl et al., PAMI 2023](https://arxiv.org/abs/2307.08668)，公式上是：

$$
\min_{s, t} \| \hat{\mathbf{D}} - (s \cdot \mathbf{D} + t) \|
$$

这会引入一个全局 affine 变换，但在 GS 训练里会与 photometric loss 的 scale 冲突，导致优化震荡。

Vid2Sim 的解决方案：**patch-based Normalized Cross-Correlation (NCC)**——只看局部结构相似性，丢弃全局 scale 信息。

### 公式 (4)：NCC Depth Loss

$$
\mathcal{L}_{\text{depth}} = 1 - \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \sum_{k=1}^{K^2} \frac{\hat{\mathbf{D}}_{p,k}' \mathbf{D}_{p,k}'}{\hat{\sigma}_p \sigma_p}
$$

- $\mathcal{P}$：所有 patch 集合（如 5×5 patch）
- $K^2$：一个 patch 内的 pixel 数（如 25）
- $\hat{\mathbf{D}}_{p,k}' = \hat{\mathbf{D}}_{p,k} - \bar{\hat{\mathbf{D}}}_p$：rendered depth 减去 patch 内均值（mean-centered）
- $\mathbf{D}_{p,k}'$：predicted depth 同样 mean-centered
- $\hat{\sigma}_p, \sigma_p$：patch 内 rendered/predicted depth 的标准差

直觉：NCC 测量的是"局部相对结构"的一致性，与全局 scale 和 shift 无关。只要 patch 内的"高低起伏模式"对得上就给低 loss，这对 monocular supervision 来说比 MSE 鲁棒得多。

### 公式 (5)：Normal Cosine Loss

$$
\mathcal{L}_{\text{normal}} = 1 - \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} \frac{\hat{\mathbf{N}}_{i,j} \cdot \mathbf{N}_{i,j}}{\|\hat{\mathbf{N}}_{i,j}\| \|\mathbf{N}_{i,j}\|}
$$

- $H, W$：渲染 image 的高宽
- $\hat{\mathbf{N}}_{i,j}$：rendered normal
- $\mathbf{N}_{i,j}$：pseudo-GT normal，由 predicted depth map 反投影成 point cloud 后做 PCA 估计的 normal

为什么用 PCA？因为 monocular depth estimator 不直接输出 normal，但 local point cloud 的最小特征向量方向就是 surface normal——这是 Geometry 101。这一步相当于把 depth prior 转成 normal prior，二者形成互补监督。

### 公式 (6)：Geometry-Consistent Loss（创新点）

$$
\mathcal{L}_{\text{geo}} = \frac{\sum_{i,j} w_{i,j} \cdot (1 - \hat{\mathbf{N}}_{i,j} \cdot \hat{\mathbf{N}}_{i+\Delta x, j+\Delta y})}{\sum_{i,j} w_{i,j}}, \quad w_{i,j} = 1 - \left\| \sqrt{(\nabla_x \mathbf{D}_{i,j})^2 + (\nabla_y \mathbf{D}_{i,j})^2} \right\|
$$

- $(i+\Delta x, j+\Delta y)$：相邻 pixel（右、下，即 $\Delta x, \Delta y \in \{0,1\}$）
- $w_{i,j}$：基于 depth gradient 的权重——depth 变化小的区域权重高（接近 1），depth 边缘权重低
- 分母：归一化项

直觉：这是 **pairwise normal consistency**——在 depth 平滑的区域，相邻 pixel 的 normal 也应该一致。这与 monocular prior 的 SSI 性质相容（局部平面的法线是确定的，与全局 scale 无关），同时压制了 GS 训练里常见的"局部 noisy Gaussian 集群"。

### $\mathcal{L}_{\text{scale}}$：2D Disk 正则

$$
\mathcal{L}_{\text{scale}} = \frac{1}{N} \sum_{i \in N} \|\min(s_1, s_2, s_3)\|
$$

灵感来自 2DGS [Huang et al., SIGGRAPH 2024](https://arxiv.org/abs/2403.17888)。直觉：真实场景里大部分 surface 是 2D 的（墙、地面、桌面），用 disk-shaped Gaussian 比球状更合理，normal 也更可靠。这个 loss 把每个 Gaussian 的最短轴压到接近 0，迫使它从"球"变成"片"。

### 公式 (7)：Total Loss

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rgb}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{normal}} + \mathcal{L}_{\text{geo}} + \mathcal{L}_{\text{scale}}
$$

注意 paper 提到 depth/normal/geo/scale loss 从第 500 iteration 才开始施加——让 GS 先用 photometric loss 摆出大致形状，再用 geometric prior 精修。

---

## 5. Screen-Space Covariance Culling（Section 4.1 末尾）

RL agent 探索时会跑到训练 view 之外的位置，特别是 camera 离 ground 很近时。原本从训练 view 看不见的"长条 Gaussian"（covariance 某一轴特别长）这时会投影成大块模糊 floater，挡住 agent 视线——它会被 agent 误判为 obstacle 而绕开，破坏 policy。

Culling 判定：

$$
\|\Sigma'\|_\infty > \alpha \cdot A_{\text{img}}
$$

- $\Sigma' = J W \Sigma W^T J^T$：投影到 screen space 的 2D covariance
- $\|\Sigma'\|_\infty$：covariance 的最大特征值（直觉上是 Gaussian 在 2D 上覆盖的最大半径）
- $A_{\text{img}} = H \times W$：image 总面积
- $\alpha$：比例阈值（超参）

直觉：如果一个 Gaussian 在 screen 上"摊得太大"（比如超过 image 的 $\alpha$ 倍面积），它就是 artifact，直接不渲染。这是一个 cheap-but-effective 的 heuristic，不改变 GS 参数，只改渲染时的 visibility 决策。

Table 4 显示这个 trick 让 FID 从 214.47 降到 191.54（+10.7% improvement）。注意：FID 在这里是相对指标（agent view vs training view 的分布距离），因为 agent view 没有 GT。

---

## 6. Hybrid Scene Representation（Section 4.2）

这是整个工作的"架构图核心"：

```
┌─────────────────────────────────────────┐
│  Unity Engine (Physics + Rendering)     │
│                                          │
│  ┌────────────┐   ┌──────────────────┐  │
│  │ GS Layer   │   │ Invisible Mesh   │  │
│  │ (visual)   │   │ (collision only) │  │
│  │            │   │                  │  │
│  │ custom     │   │ TSDF from GS     │  │
│  │ shader     │   │ depth rendering  │  │
│  └─────┬──────┘   └────────┬─────────┘  │
│        │                   │             │
│        ▼                   ▼             │
│   ┌──────────┐        ┌───────────┐     │
│   │ Agent    │◀──────▶│ Physics   │     │
│   │ RGB/Depth│        │ Engine    │     │
│   └──────────┘        └───────────┘     │
└─────────────────────────────────────────┘
```

具体实现细节：
1. **GS → Mesh**：渲染每帧的 depth map → KinectFusion 融合 TSDF（voxel size 0.1m）→ marching cubes 出 mesh
2. **Ground removal**：用公式 (8) 的 normal prior 而非纯 SAM，因为 SAM 对 ground 的 segmentation 不可靠

### 公式 (8)：Ground Mask via Normal Prior

$$
\mathcal{M}_i = \|\arccos(\mathbf{N}_i \cdot \bar{\mathbf{n}}_i') < \delta\|
$$

- $\mathbf{N}_i$：第 $i$ 帧的 rendered normal map
- $\bar{\mathbf{n}}_i'$：从 SAM-HQ 给的粗 ground mask 计算的平均 normal 方向（作为 ground plane 的 prior）
- $\delta = 15°$：角度阈值
- $\|\cdot\|$：Iverson bracket（条件成立为 1，否则 0）

直觉：先用 SAM 给一个粗的 ground 区域，算出该区域的平均法线（基本是 up 方向），再用这个法线"放大"——任何 normal 与之夹角小于 15° 的 pixel 都判为 ground。这把 SAM 的"粗略 mask"变成"精确的平面区域"。

3. **Unity Shader**：自己写一个 GS rasterizer shader，在 Unity 里实时渲染（实时性能 OK，因为 GS 原本就是为 real-time rendering 设计的）
4. **Mesh collider**：mesh 材质设为 invisible（不渲染），但作为 Unity 的 MeshCollider 参与物理
5. **Occlusion**：foreground obstacle 和 background GS 之间用 z-buffer 合成（公式上没写，但实现上是 depth test）

---

## 7. Interactive Scene Composition

### Static Obstacles
- Asset pool：traffic cone、trash bin、traffic light、pole
- 每个 episode 随机采样 0–5 个放在 agent 起点和终点之间
- Foreground obstacle 用 mesh 渲染，与 background GS 用 z-buffer 合成 RGB 和 depth

### Dynamic Obstacles (Pedestrians)
- 用 A* [Hart et al., 1968](https://en.wikipedia.org/wiki/A*_search_algorithm) 在 walkable area 上规划 shortest path
- Pedestrian 在 random 点之间走 shortest path
- Agent 必须预测并避让

### Safety-Critical Scenarios
- 倒下的 trash can、roadblock + worker 等 OOD corner case
- 这些场景在 autonomous navigation 研究里是 [Bai et al., ICLR 2023](https://arxiv.org/abs/2210.10792) 提到的 long-tail 难点

---

## 8. Scene Augmentation

### 风格化（Figure 4）
基于 Instruct-4D-to-4D [Mou et al., CVPR 2024](https://arxiv.org/abs/2306.08904)，把 4D scene 当 pseudo-3D 用 2D diffusion 编辑，保持 temporal consistency。可以变 lighting（sunset, night）、语义、季节。

### 天气（Figure 11）
参考 ClimateNeRF [Li et al., ICCV 2023](https://arxiv.org/abs/2303.17926)，在 Unity 里用 particle system 模拟 rain、fog、snow。Rain 是 fall particle，fog 是 volumetric 的，snow 是更慢的 fall particle + wind。

### 多级 augmentation 的意义
让 agent 在同一段 video 重建的 scene 里见到 N 种变体，相当于 **数据扩展**——一个 video 变成多个不同条件下训练环境。这是 Vid2Sim scalability 的关键。

---

## 9. 实验

### 9.1 Reconstruction Evaluation (Table 1)

| Methods | PSNR ↑ | SSIM↑ | LPIPS ↓ | Real Time | Interactive | RL Training |
|---|---|---|---|---|---|---|
| Instant-NGP | 27.50 | 0.827 | 0.240 | ✗ | ✗ | ✗ |
| 3DGS | 31.85 | 0.921 | 0.136 | ✓ | ✗ | ✗ |
| 2DGS | 30.82 | 0.915 | 0.154 | ✓ | ✗ | ✗ |
| Video2Game | 28.32 | 0.834 | 0.275 | ? | ✓ | ✗ |
| **Vid2Sim** | **32.41** | **0.927** | **0.127** | ✓ | ✓ | ✓ |

观察：
- Vid2Sim 的 PSNR 比 3DGS 高 0.56 dB，比 2DGS 高 1.59 dB
- Video2Game 因为用 textured mesh 渲染，PSNR 只有 28.32，差距明显
- **关键**：只有 Vid2Sim 同时满足 real-time + interactive + RL trainable

### 9.2 Urban Navigation Training (Table 2)

| Methods | Obs | PointNav SR↑ | SPL↑ | Cost ↓ | SocialNav SR↑ | SNS↑ | Cost ↓ |
|---|---|---|---|---|---|---|---|
| Mesh† | RGB | 48.8% | 0.496 | 0.34 | 43.2% | 0.991 | 1.04 |
| Vid2Sim (Oracle) | Depth | 92.0% | 0.937 | 0.57 | 85.6% | 0.992 | 0.75 |
| Vid2Sim (No Obj) | RGB | 68.8% | 0.695 | 1.45 | 61.6% | 0.973 | 1.79 |
| Vid2Sim (Static) | RGB | 80.8% | 0.818 | 0.94 | 71.2% | 0.980 | 1.74 |
| **Vid2Sim (Dynamic)** | RGB | **81.6%** | **0.824** | **0.86** | **74.4%** | **0.987** | **1.21** |

关键 insight：
1. **Mesh† vs Vid2Sim**：同样 RGB obs，mesh-based sim 在 PointNav 只有 48.8% SR，Vid2Sim 81.6%——提升 **32.8 个百分点**。这正是 sim2real gap 的量化指标，证明 photorealistic 渲染对 navigation policy 的关键作用。
2. **Oracle depth (92.0%) vs RGB (81.6%)**：depth 显著帮助，但 RGB-only 已经很强，说明 GS-rendered RGB 已经足够 informative
3. **No Obj (68.8%) vs Static (80.8%) vs Dynamic (81.6%)**：加入 obstacle 反而提升 SR！这是因为 obstacle 增加了 task 的"密度"，agent 学到的 avoidance 行为更 generalizable。Cost（撞击次数）也单调下降。
4. **SocialNav 上 Dynamic (74.4%) vs No Obj (61.6%)**：在动态场景里 emergent avoidance behavior 也能 generalize

### 9.3 Generalization vs #Environments (Figure 6)

随 training env 数量从 1 → 30 增加：
- SR 从 ~50% 上升到 ~80%
- SPL 从 ~0.5 上升到 ~0.82
- 方差单调下降

这是 **scaling law 的雏形**：更多 real2sim env → 更 generalizable policy。Paper Section G 也承认 30 env 不够，希望未来扩展到更大规模。这与 LLM scaling 的逻辑一致—— embodied AI 也很可能 follow 类似规律，瓶颈在 data production pipeline 的 cost。

### 9.4 Sim2Real (Table 3)

| Method (Env N) | Go Straight | Static Obstacle | Dynamic Obstacle |
|---|---|---|---|
| Baseline (30) | 0% | 0% | 0% |
| Vid2Sim (1) | 0% | 30% | 0% |
| Vid2Sim (5) | 60% | 40% | 0% |
| **Vid2Sim (30)** | **85%** | **65%** | **55%** |

震撼点：
1. **Mesh-based baseline 完全 0% SR**——这是 sim2real gap 最直接的证明，mesh 渲染的 texture 与真实世界差距太大，policy 完全不能 transfer
2. **Env 数 scaling**：1 env → 0% / 30% / 0%；30 env → 85% / 65% / 55%。Go Straight 的提升尤其显著，说明 agent 在多个 env 学到 generalizable visual feature
3. **Dynamic obstacle 55% SR**：在 zero-shot 真实动态场景下能 55% 成功率——这是非常 impressive 的 sim2real 结果

---

## 10. RL Training Details（Section D）

### Reward Function

$$
R = R_{\text{term}} + c_1 R_{\text{dist}} + c_2 R_{\text{steer}} + c_3 R_{\text{crash}} + c_4 R_{\text{time}}
$$

各项含义：
- $R_{\text{term}} = \pm 10$：到达 +10，失败 -10（sparse）
- $R_{\text{dist}} = d_t - d_{t-1}$：dense reward，鼓励接近 goal，$c_1 = 1$
- $R_{\text{steer}} = -\|s_t - s_{t-1}\| \cdot v_t$：steering 平滑性惩罚，$c_2 = 0.05$
- $R_{\text{crash}} = -\mathbb{1}(c_t)$：碰撞惩罚，$c_3 = 1.0$
- $R_{\text{time}} = -\Delta t \approx -1$：时间惩罚，$c_4 = -0.1$（注意是负的，鼓励快）

终止条件：
- 超出 drivable area
- >3000 steps timeout
- 累计 >3 次碰撞

### SAC Hyper-parameters (Table 5)
- $\gamma = 0.99$（标准 discount）
- $\tau = 0.005$（target network soft update）
- LR $= 3 \times 10^{-4}$
- Batch 256
- 用 SDE（State-Dependent Exploration）——比 action noise 更适合 continuous control

### Observation
- 1280×720 RGB → resize 到 128×72
- Stack 过去 5 帧（temporal context）
- Concatenate 当前帧 + distance to goal + heading angle
- Action：normalized linear + angular velocity ∈ [-1, 1]
- Real-world 通过 system identification 重 scaling 到 real unit

---

## 11. 与 Related Work 的比较

### vs NeRF-based methods
- Mip-NeRF 360 [Barron et al., CVPR 2022](https://arxiv.org/abs/2111.12005)：高质量 NVS 但慢、无 interaction
- NeRF-W [Martin-Brualla et al., CVPR 2021](https://arxiv.org/abs/2008.02268)：unconstrained photo collection，但仍是 NVS only

### vs GS variants
- 2DGS [Huang et al., 2024](https://arxiv.org/abs/2403.17888)：Vid2Sim 借鉴了它的 disk regularization，但 2DGS 没有 interaction
- SuGa [Guédon & Lepetit, CVPR 2024](https://arxiv.org/abs/2403.14049)：surface-aligned GS，做 mesh reconstruction，但不做 simulation
- GS2Mesh [Wolf et al., ECCV 2024](https://arxiv.org/abs/2404.17582)：从 GS 做 stereo-based mesh，质量好但同样无 interaction

### vs Generative methods
- DrivingDiffusion [Li et al., 2023](https://arxiv.org/abs/2310.07771)：multi-view driving video generation，但 no closed-loop physics
- DriveDreamer [Wang et al., ECCV 2024](https://arxiv.org/abs/2309.09777)：world model for driving，2D generation + 物理 prior
- LucidSim [Yu et al., CoRL 2024](https://arxiv.org/abs/2411.17448)：用生成图像训练 locomotion，但是 2D 没有 3D geometry

### vs Data-driven simulator
- MetaUrban [Wu et al., 2024](https://arxiv.org/abs/2407.08725)：urban space simulation，但 asset hand-crafted
- Sim-on-Wheels [Shen et al., 2023](https://arxiv.org/abs/2305.16021)：vehicle-in-the-loop，cost 高
- SimGen [Zhou et al., NeurIPS 2024](https://arxiv.org/abs/2410.23272)：simulator-conditioned generation，但还是 2D image-level

Vid2Sim 的独特位置：**唯一从 monocular video 出发，做 GS+mesh hybrid，能跑 closed-loop RL 的工作**。

---

## 12. Limitations 与未来方向

Paper Section G 承认：
1. **每个 scene 构建慢**：GLOMAP 初始化 + GS 训练 + TSDF 提 mesh + Unity 配置，整个流程可能需要几小时
2. **30 env 不够**：Figure 6 显示 scaling 没饱和，更多 env 会更好
3. **只做了 wheeled robot**：未来扩展到 legged robot、humanoid

潜在改进方向（paper 没提但我推测）：
- 用 [Scaffold-GS](https://arxiv.org/abs/2312.00109) 或 [Mip-Splatting](https://arxiv.org/abs/2312.00201) 改善 extreme view rendering
- 用 metric depth（如 [Metric3D v2](https://arxiv.org/abs/2310.16818)）替代 SSI loss，可能简化 supervision
- 用 [DreamGaussian](https://arxiv.org/abs/2309.16653) 类加速训练
- 用 [SplaTAM](https://arxiv.org/abs/2403.02751) 做 SLAM-style online 重建，避免 offline GLOMAP

---

## 13. Personal Take（build intuition）

Vid2Sim 的真正贡献是把"3DGS 是 NVS 工具"的 framing 推到了"3DGS 是 simulation 的 visual layer"。这件事以前没人做透——很多 paper 都说"GS 可以做 simulation"，但真正闭环到 RL training + sim2real deployment 的这是第一个完整工作。

它的 scaling argument 很 important：30 个 env 就能让 zero-shot sim2real 出现明显 scaling 迹象，意味着如果有人愿意爬 1000 个 web video（YouTube 上 walking tour video 海量），可能真的能训出 general navigation foundation policy。这与 Tesla 用行车记录仪数据训 FSD 的逻辑类似，但多了 3D reconstruction 这一中间步骤。

公式上最 elegant 的部分是 **scale-invariant NCC + geometry-consistent loss 的组合**——把 monocular prior 从"绝对 metric 监督"解放成"局部结构监督"，避开了 SfM 与 depth estimator 之间 scale 不一致的根本矛盾。这种设计思路其实在很多场景都适用：当你有两个 scale-ambiguous 的信号时，patch-based NCC + local consistency 永远比 global MSE 鲁棒。

---

## Reference Links

- **Project Page**: https://metadriverse.github.io/vid2sim/
- **3DGS**: https://repo.samplify.org/kerbl_3dgaussians (paper: https://arxiv.org/abs/2308.14737)
- **2DGS**: https://arxiv.org/abs/2403.17888
- **Video2Game**: https://arxiv.org/abs/2312.07534
- **Depth Anything V2**: https://github.com/DepthAnything/Depth-Anything-V2
- **GLOMAP**: https://arxiv.org/abs/2407.01991
- **COLMAP**: https://colmap.github.io/
- **DEVA Tracker**: https://arxiv.org/abs/2305.05578
- **Instruct-4D-to-4D**: https://arxiv.org/abs/2306.08904
- **ClimateNeRF**: https://arxiv.org/abs/2303.17926
- **SAC**: https://arxiv.org/abs/1801.01290
- **MetaDrive**: https://github.com/metadriverse/metadrive
- **MetaUrban**: https://metadriverse.github.io/metaurban/
- **Unity Engine**: https://unity.com/
- **KinectFusion (TSDF)**: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf
- **LucidSim**: https://arxiv.org/abs/2411.17448
- **DriveDreamer**: https://arxiv.org/abs/2309.09777
- **SimGen**: https://arxiv.org/abs/2410.23272

如果你想我把某一个公式再拆得更细（比如推导一下 NCC 的等价形式、或者 GS 投影 Jacobian $J$ 的具体表达式），或者深入讲 reward shaping 里的 $R_{\text{steer}} = -\|s_t - s_{t-1}\| \cdot v_t$ 为什么乘 $v_t$（这是一个相当 subtle 的 design choice），随时告诉我。
