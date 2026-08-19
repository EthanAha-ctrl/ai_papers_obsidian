---
source_pdf: GS-LIVO Real-Time LiDAR, Inertial, and Visual Multi-sensor.pdf
paper_sha256: 7fb8911abb6baf7bad64515ff717ce296f7716c9217efaf2f1571be7578da649
processed_at: '2026-08-19T10:06:50-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 GS-LIVO

## 一句话故事

把 3D Gaussian Splatting 塞进一个能在 Jetson 上跑 20Hz 的 LIVO SLAM 系统。

---

## 这 paper 想干啥

故事背景是这样:最近 3DGS 大火,大家觉得"诶,这不就是 SLAM 梦寐以求的 map representation 吗"——photorealistic、differentiable、real-time rendering。于是一堆人扑上去做 Gaussian-SLAM [33, 34, 51]。

结果发现一个尴尬事实:

**Pose 跑得飞快,map 优化跟不上。**

具体说,MonoGS 这种 SOTA,在 desktop 上能跑 5-10Hz,但一旦你想 deploy 到 robot 上,RTX-4090 那种 GPU 你塞不进去。Jetson Orin NX 只有 1024 CUDA cores、16GB shared memory,跟本跑不动 millions of Gaussians。

GS-LIVO 团队(就是 FAST-LIVO 那批人)看了一眼说:"嗯,我们用 LiDAR + IMU + Camera 做了十年 LIVO 了,这玩意儿我们对 sensory 很熟。咱们把 Gaussian map 当 visual measurement model 的 reference,顺带把传统 LIVO 的 robustness 带进来。"

于是有了 GS-LIVO。

Reference: https://github.com/hk-mars/FAST_LIO

---

## Intuition:为什么 Gaussian 当 measurement model 很漂亮

传统 LIVO (比如 FAST-LIVO2) 是这样的:

```
reference frame 的 patch → warp 到 current frame → 跟 current frame 比 photometric error
```

这有个麻烦:patch size 大了会有 seams(像 Fig. 4 里那样),patch size 小了对 noise 敏感,而且只能 handle Lambertian surface。

GS-LIVO 的 idea 颠倒过来:

```
current LiDAR/IMU 估计的 pose → 用 Gaussian map render 一张 image → 跟实际 camera 拍到的 image 比
```

Gaussian rendering 天然 differentiable、seamless、能 model non-Lambertian(view-dependent color via spherical harmonics)。Paper 里 Fig. 4 对比 patch size 32/64 的 warping,seams 明显,Gaussian rendering 干干净净。

这给了你一个 intutiion:**map 越好,odometry 越准;odometry 越准,map 越好**。这是 SLAM 的经典 feedback loop,只是用 Gaussian map 把 loop 的质量拉到一个新高度。

---

## 三个 Key Trick 让它能跑

### Trick 1: Hash-Octree + RAM/VRAM 分离(解决 GPU memory bottleneck)

3DGS 通常所有 Gaussians 都塞 VRAM。大场景下 millions 量级,Jetson 根本装不下。

他们的设计很聪明:

- **Global map**:存在 CPU RAM 里,用 hash-indexed octree(non-contiguous,但容量大,还能 swap)
- **当前 FoV 的 Gaussians**:copy 到 contiguous RAM buffer (CGB),再 transfer 到 VRAM (GGB)
- **GPU 只 optimize GGB 里的 Gaussians**,optimize 完写回

为什么这样 work?因为 **consecutive frames 90%+ FoV overlap**。你不需要每 frame reload 全部 map,只 swap out 离开 FoV 的那部分,swap in 新进 FoV 的那部分。

Hash key 公式很简洁:

$$\text{HashKey} = \left\lfloor \frac{{}^W \mathbf{p}_i}{\mathbf{v}_s} \right\rfloor$$

变量解释:
- ${}^W \mathbf{p}_i$:第 i 个 Gaussian center 在 world frame 的 3D position(单位 meter)
- $\mathbf{v}_s$:root voxel edge length(indoor 0.03m,outdoor 0.5-1.0m)
- $\lfloor\cdot\rfloor$:对每个 component 取整,输出 3D integer vector $(k_x, k_y, k_z)$

这个公式就是把连续 3D 空间划成 size 为 $\mathbf{v}_s^3$ 的 voxel grid。每个 root voxel 内部用 octree 递归细分成 2 层(paper 默认),叶子节点存 Gaussians。

Octree 的好处:
- 稀疏空间(空旷区域)用 coarse voxel,细节区域 fine voxel
- 支持 LoD rendering(远处用 root voxel,近处用 leaf voxel)
- $O(1)$ voxel lookup

Reference: https://github.com/octree-gs/octree-gs

---

### Trick 2: LiDAR-Visual Joint Initialization(解决 Gaussian 收敛慢)

原版 3DGS 用 SfM points 初始化,需要 thousands of iterations 才能 converge 到 reasonable representation。SLAM 场景下你没有这个 luxury。

GS-LIVO 的 insight:**LiDAR 已经给了你准确的 surface geometry,为啥不用它直接 init well-structured Gaussians**?

具体怎么做:

**Scaling matrix**(Eq. 2):

$$\mathbf{S}_i = \text{diag}(\mathbf{s}_\delta, \mathbf{s}_y, \mathbf{s}_z)$$

变量解释:
- $\mathbf{s}_\delta$:很小很小的数值(hyperparameter),让 Gaussian 在 surface 法向方向薄得像一张饼
- $\mathbf{s}_y, \mathbf{s}_z$:surface 切平面两个方向的 scale,与 voxel level 相关

**Intuition**:每个 Gaussian 是一张贴在 surface 上的 2D disk,不是一个 3D ball。这跟 surfel-based 方法(FAST-LIVO)思路一致,但用 Gaussian 表示更适合 differentiable rendering。

**Rotation matrix**(Eq. 3):

$$
{}^W \mathbf{R}_i = \begin{pmatrix}
\frac{\mathbf{e}_x \times {}^W \mathbf{n}_i}{\|\mathbf{e}_x \times {}^W \mathbf{n}_i\|} &
\frac{{}^W \mathbf{n}_i \times (\mathbf{e}_x \times {}^W \mathbf{n}_i)}{\|\mathbf{e}_x \times {}^W \mathbf{n}_i\|} &
{}^W \mathbf{n}_i
\end{pmatrix}
$$

变量解释:
- ${}^W \mathbf{n}_i$:surface normal at Gaussian i 的位置(由 LiDAR-inertial SLAM [56] 提供)
- $\mathbf{e}_x = (1,0,0)^T$:x-axis unit vector,作为 reference
- $\times$:cross product

构造一个 rotation matrix,三列分别是 Gaussian local frame 的三个 axes。第三列是 surface normal(thickness 方向),第一列是 x-axis 与 normal 的 cross product(切平面内沿 x 方向),第二列 Gram-Schmidt 正交化出来。

**Covariance**(Eq. 4):

$$\pmb{\Sigma}_{3D,i} = ({}^W \mathbf{R}_i \mathbf{S}_i)({}^W \mathbf{R}_i \mathbf{S}_i)^T$$

这就是 3D-GS 标准 parameterization $\Sigma = R S S^T R^T$。保证 $\Sigma$ symmetric positive semi-definite,便于通过李代数 optimize。

**Color**(Eq. 7):

$$\mathbf{c}(\mathbf{q}_i) = \sum_{j=1}^{4} \mathbf{c}_j \cdot \mathbf{A}_j$$

变量解释:
- $\mathbf{q}_i$:Gaussian 投影到 image plane 的 2D pixel coordinate
- $\mathbf{c}_j$:4 个 neighbor integer pixels 的 RGB
- $\mathbf{A}_j$:bilinear interpolation 的 area weight

只 init spherical harmonic 的 zero order(DC component),higher orders 初始化为 0,后续 photometric optimization 学。

整个 init 流程非常 clean:**有几何(LiDAR)有颜色(camera),直接构出 well-structured Gaussian**,跳过 3D-GS 训练初期那种混乱探索期。

---

### Trick 3: Sliding Window Incremental Update(解决 host-device transfer overhead)

这是 paper 最 clever 的 engineering contribution。

如果不做 sliding window,每 frame 都要:
1. 找当前 FoV 内所有 Gaussians
2. 从 RAM hash-octree 中读取(非连续访问)
3. Copy 到 contiguous buffer
4. Transfer host → device
5. GPU optimize
6. Transfer device → host
7. 写回 hash-octree

O(N) memory copy per frame,N 大了就死。

他们的 sliding window 维护流程(Fig. 3):

| Step | 操作 | 状态变化 |
|------|------|----------|
| 1 | Update to Global Map | FoV 外 voxels 标记 DELETE,optimized params copy 回 global map |
| 2 | Deletion and Compaction | DELETE voxels 与 sliding window 末尾 swap,释放末尾空间 |
| 3 | Overlap and Addition | 用 current LiDAR frame 计算 hash keys,识别 OVERLAP 与 ADD voxels |
| 4 | Appending New Leaf Voxels | ADD voxels append 到 CGB 末尾,更新 SHT,CGB → GGB transfer |
| 5 | GPU Optimization | GGB 上并行 optimize,写回 GGB → CGB → global map |

关键 intuition:**temporal coherence**。连续两帧之间,FoV 重叠 90%+。你只需要处理那 10% 的变化(O(ΔN)),不用重新处理整个 N。

Ablation 数据说话(Fig. 10, 11):

| 指标 | Without Sliding Window | With Sliding Window |
|------|------------------------|---------------------|
| VRAM usage(Outdoor) | 8-22 GB,linearly grow | 2-3 GB,stable |
| Processing time(Outdoor) | 随 map size 超线性增长,可达 seconds/frame | < 100 ms,stable at 10 Hz |
| PSNR(Outdoor) | 25-30 dB 略稳定 | 25-30 dB,view change 时短暂 drop 但快速 recover |

Quality 几乎不损失,memory 与 compute 大幅下降。这就是工程上 win-win 的 trick。

---

## IESKF:Tightly Coupled 的关键

这部分是最 technical 的,但 intuition 很简单。

**问题**:大多数 Gaussian-SLAM 用 gradient descent 算 pose,没 covariance 概念。这意味着你没法把 pose uncertainty propagate 到下一个 IMU/LiDAR measurement,tightly coupled 就无从谈起。

**GS-LIVO 的 approach**:把 photometric loss 对 camera pose 的 Jacobian,通过 chain rule 推到 IMU pose 上,然后塞进 IESKF 框架。

核心 measurement model:

$$
{}^W \mathbf{T}_C^* = \arg\min_{\mathbf{T}(\boldsymbol{\xi})} \sum_{G_i^{3D} \in \theta_{k-1}} \left\| \mathbf{I}_k - \widehat{\mathbf{I}}_k({}^W \mathbf{T}_C; \theta_{k-1}) \right\|
$$

变量解释:
- ${}^W \mathbf{T}_C^*$:要优化的 camera pose in world frame
- $\boldsymbol{\xi} \in \mathbb{R}^6$:camera pose 的 Lie algebra perturbation,$\boldsymbol{\xi} = (\boldsymbol{\omega}, \boldsymbol{\rho})$,$\boldsymbol{\omega}$ 是 rotation 增量(3维),$\boldsymbol{\rho}$ 是 translation 增量(3维)
- $\mathbf{T}(\boldsymbol{\xi}) = \exp(\boldsymbol{\xi}^\wedge)$:Lie algebra 到 Lie group 的 exponential map
- $\mathbf{I}_k$:第 k 帧实际拍到的 image
- $\widehat{\mathbf{I}}_k$:从 Gaussian map 在 pose ${}^W \mathbf{T}_C$ 下 rendered 的 image
- $\theta_{k-1}$:上一帧已 optimized 的 Gaussian parameters(这里 fixed)

注意跟 Eq. (9) 的区别:Eq. (9) optimize Gaussian parameters(pose fixed),Eq. (10) optimize pose(Gaussian parameters fixed)。这是 alternating optimization 经典 pattern。

### Jacobian Chain Rule 推导

Appendix B 给出完整推导,核心是 chain rule:

$$\frac{\partial \mathbf{q}_i}{\partial {}^W \mathbf{R}_I} = \frac{\partial \mathbf{q}_i}{\partial {}^C \mathbf{p}_i} \frac{\partial {}^C \mathbf{p}_i}{\partial {}^C \mathbf{R}_W} \frac{\partial {}^C \mathbf{R}_W}{\partial {}^W \mathbf{R}_I}$$

变量解释:
- ${}^W \mathbf{R}_I$:IMU orientation in world(待优化 state)
- ${}^C \mathbf{p}_i$:Gaussian i 在 camera frame 的 3D position
- ${}^C \mathbf{R}_W$:world-to-camera rotation

三步 chain:
1. $\partial \mathbf{q}_i / \partial {}^C \mathbf{p}_i$:projection 的 Jacobian(pinhole camera $\partial(u,v)/\partial(X,Y,Z)$)
2. $\partial {}^C \mathbf{p}_i / \partial {}^C \mathbf{R}_W$:点在 camera frame 中对 rotation 的 sensitivity
3. $\partial {}^C \mathbf{R}_W / \partial {}^W \mathbf{R}_I$:camera-IMU extrinsic 引入的耦合

Translation 部分(Eq. 12)有 **两项 contributions**:

$$\frac{\partial \mathbf{q}_i}{\partial {}^W \mathbf{t}_I} = \underbrace{\frac{\partial \mathbf{q}_i}{\partial {}^C \mathbf{p}_i} \frac{\partial {}^C \mathbf{p}_i}{\partial {}^C \mathbf{t}_W} \frac{\partial {}^C \mathbf{t}_W}{\partial {}^W \mathbf{t}_I}}_{\text{direct translation chain}} + \underbrace{\frac{\partial \mathbf{q}_i}{\partial {}^C \mathbf{p}_i} \frac{\partial {}^C \mathbf{p}_i}{\partial {}^C \mathbf{R}_W} \frac{\partial {}^C \mathbf{R}_W}{\partial {}^W \mathbf{t}_I}}_{\text{translation 通过 extrinsic rotation 的耦合}}$$

第二项揭示一个 subtle fact:**IMU translation 变化也会通过 extrinsic rotation 影响 camera-to-world rotation**。因为 extrinsic 是 fixed offset,IMU move 时 camera 也 move + 旋转。

### Perturbation Model 推导

Appendix B 给出 perturbation model:

$$(\mathbf{T}(\delta\varphi, \delta t) \cdot {}^C \mathbf{T}_W)^{-1} \cdot {}^C \mathbf{T}_I = {}^W \mathbf{T}_I \boxplus \mathbf{T}(\delta R, \delta \rho)$$

变量解释:
- $(\delta\varphi, \delta t)$:camera pose 上的 perturbation(rotation, translation)
- $(\delta R, \delta \rho)$:IMU pose 上的 perturbation
- $\boxplus$:Lie group 上的 box-plus operation

Rotation 部分推导 key step:

$$(\text{Exp}(\delta\varphi^\wedge) {}^C \mathbf{R}_W)^{-1} {}^C \mathbf{R}_I = {}^W \mathbf{R}_I \text{Exp}(\delta R^\wedge)$$

用 first-order approximation $\text{Exp}(\xi^\wedge) \approx I + \xi^\wedge$:

$${}^W \mathbf{R}_I - {}^W \mathbf{R}_C \delta\varphi^\wedge {}^C \mathbf{R}_I = {}^W \mathbf{R}_I + {}^W \mathbf{R}_I \delta R^\wedge$$

Cancel ${}^W \mathbf{R}_I$:

$$-{}^I \mathbf{R}_C \delta\varphi^\wedge {}^C \mathbf{R}_I = \delta R^\wedge$$

用 skew-symmetric 性质 $a^\wedge b = -b^\wedge a$:

$$(-{}^I \mathbf{R}_C \delta\varphi)^\wedge = \delta R^\wedge$$

得到:

$$\delta\varphi = -{}^C \mathbf{R}_I \delta R$$

所以:

$$\frac{\partial {}^C \mathbf{R}_W}{\partial {}^W \mathbf{R}_I} = -{}^C \mathbf{R}_I$$

Translation 部分类似推导(Appendix B.3),一阶展开后:

$${}^W \mathbf{R}_C {}^C \mathbf{t}_I^\wedge \delta\varphi - {}^W \mathbf{R}_C \delta t = \delta \rho$$

代入 $\delta\varphi = -{}^C \mathbf{R}_I \delta R$ 取系数,得到:

$$\frac{\partial {}^C \mathbf{t}_W}{\partial {}^W \mathbf{t}_I} = -{}^W \mathbf{R}_C^T$$

$$\frac{\partial {}^C \mathbf{R}_W}{\partial {}^W \mathbf{t}_I} = -{}^C \mathbf{t}_I^\wedge {}^C \mathbf{R}_W$$

变量解释:
- ${}^C \mathbf{t}_I^\wedge$:${}^C \mathbf{t}_i$ 的 skew-symmetric matrix,捕捉 extrinsic translation 引入的 rotation-translation coupling

**Intuition**:这是经典 hand-eye calibration 出现的项。IMU-camera extrinsic translation 不为零时,IMU translation 变化会让 camera 也 rotate(因为 camera 跟着 IMU translate,但 extrinsic rotation 让 camera 围绕 IMU 旋转一个 arc)。

### IESKF Update Equations

最后,所有 Jacobian 塞进 IESKF:

$$\mathbf{H} = [\mathbf{J}_1^T, \cdots, \mathbf{J}_m^T]^T$$

$$\mathbf{R} = \text{diag}(\pmb{\Sigma}_{\mathbf{u}_1}, \cdots, \pmb{\Sigma}_{\mathbf{u}_m})$$

$$\mathbf{P} = \mathcal{H}^{-1} \pmb{\Sigma}_{{}^W \hat{\mathbf{T}}_I^-} \mathcal{H}^{-T}$$

$$\mathbf{K} = (\mathbf{H}^T \mathbf{R}^{-1} \mathbf{H} + \mathbf{P}^{-1})^{-1} \mathbf{H}^T \mathbf{R}^{-1}$$

$${}^W \check{\mathbf{T}}_I = {}^W \hat{\mathbf{T}}_I \boxplus T(\boldsymbol{\xi})$$

$$\pmb{\Sigma}_{{}^W \check{\mathbf{T}}_I} = (\mathbf{I} - \mathbf{K}\mathbf{H}) \mathbf{P}$$

变量解释:
- $\mathbf{H}$:stacked Jacobian matrix,$m$ 个 measurements 的 Jacobian 拼起来
- $\mathbf{R}$:measurement noise covariance(对角阵,假设各 measurement 独立)
- $\mathbf{P}$:prior state covariance 经过 Lie group Jacobian $\mathcal{H}$ transform 后的结果
- $\mathbf{K}$:Kalman gain,平衡 measurement noise $\mathbf{R}$ 与 prior covariance $\mathbf{P}$
- ${}^W \hat{\mathbf{T}}_I$:prior state(来自 IMU propagation 或 LiDAR update)
- ${}^W \check{\mathbf{T}}_I$:posterior state(visual update 后)

**Intuition**:这是标准 IESKF update equation,跟 FAST-LIO2、FAST-LIVO2 完全兼容。Paper 重点 highlight 的点:大多数 Gaussian-SLAM 只用 optimizer 算 pose 没 covariance,GS-LIVO 显式 propagate pose covariance 到下一个 sensor update(IMU、LiDAR),form a tightly coupled multi-sensor system。这是与传统 LIO/VIO ecosystem 兼容的关键。

Reference: https://github.com/hk-mars/FAST_LIVO2

---

## 实验数据:GS-LIVO 跟谁比、好在哪

### Rendering Quality(Table II)

| Scene | Method | PSNR (dB)↑ | Dur. (s)↓ | Mem. (GB)↓ |
|-------|--------|------------|-----------|-------------|
| HKU01 (indoor) | 3D-GS(offline) | 26.22 | 2128.6 | 13.8 |
| | SplaTAM | 24.06 | 292.2 | 2.6 |
| | MonoGS | 23.51 | 258.0 | 3.1 |
| | LetsGo(offline) | 24.51 | 3231.3 | 18.1 |
| | **GS-LIVO** | **25.34** | **82.5** | **2.2** |
| CBD03 (indoor) | 3D-GS | 29.54 | 1873.8 | 12.5 |
| | SplaTAM | 26.85 | 265.2 | 4.8 |
| | MonoGS | 27.10 | 278.4 | 4.6 |
| | LetsGo | 25.51 | 3573.6 | 20.4 |
| | **GS-LIVO** | **27.52** | **88.4** | **2.2** |
| HKisland03 (outdoor) | 3D-GS | 17.52 | 3494.1 | 21.6 |
| | SplaTAM | 12.60 | 790.0 | 10.5 |
| | LetsGo | 18.32 | 2803.3 | 17.6 |
| | **GS-LIVO** | 15.32 | **82.8** | **3.2** |

**关键 takeaway**:
- Indoor PSNR 跟 3D-GS offline 接近,但速度快 **25-40x**,memory 少 **6-8x**
- Outdoor PSNR 低于 3D-GS 和 LetsGo(因为 coarse voxel 1.0m),但速度 **40x** faster,memory **6x** less
- 大场景 trade-off 明显:牺牲点细节换 real-time

### Odometry Accuracy

**vs Traditional LIV(Table III)**:

| Method | HKisland03 RMSE (m) | HKairport01 RMSE (m) | Dur. (ms) |
|--------|----------------------|----------------------|-----------|
| FAST-LIVO | 0.51 | 0.56 | 38.9 / 44.5 |
| R3LIVE | 1.71 | 1.22 | 283.3 |
| LVI-SAM | 4.12 | 5.21 | 73.5 / 266.8 |
| **GS-LIVO** | **0.58** | **0.63** | 82.8 / 93.2 |

- Outdoor 比 R3LIVE 好 **3x**,比 LVI-SAM 好 **8x**
- 比 FAST-LIVO 略差(0.58 vs 0.51),但 GS-LIVO 同时 maintain photorealistic dense map
- Processing time 82-93ms 约 12Hz,real-time

**Indoor Playground**:

| Method | Playground01 RMSE (m) | Dur. (ms) |
|--------|----------------------|-----------|
| FAST-LIVO | 0.005 | 10.5 |
| R3LIVE | 0.014 | 60.6 |
| LVI-SAM | 0.023 | 96.6 |
| **GS-LIVO** | **0.006** | 48.5 |

Indoor accuracy 几乎跟 FAST-LIVO 持平,far outperform R3LIVE/LVI-SAM。

**vs Gaussian-based SLAM(Table IV)**:

| Method | Playground01 RMSE (m) | Dur. (ms) | Mem. (GB) |
|--------|----------------------|-----------|-----------|
| SplaTAM | 0.28 | 612.8 | 12.5 |
| MonoGS*(RGBD) | 0.09 | 841.5 | 19.6 |
| MonoGS(mono) | 0.18 | 541.5 | 21.0 |
| **GS-LIVO** | **0.006** | **48.5** | **1.2** |

这是 dramatic 对比:
- **Accuracy**: GS-LIVO 0.006m vs MonoGS 0.09m vs SplaTAM 0.28m → **15-47x improvement**
- **Speed**: 48.5ms vs 541-841ms → **12-17x speedup**
- **Memory**: 1.2GB vs 12.5-21GB → **10-17x reduction**

Outdoor 场景 SplaTAM 和 MonoGS 直接 fail(RMSE = x),因为没 LiDAR/IMU,大尺度环境无能为力。

---

## Jetson Orin NX 上的部署结果

这是 paper 的 first-of-its-kind achievement:

| Metric | Value |
|--------|-------|
| Image resolution | 256×216 |
| Root voxel size | 0.5m |
| Subdivision layers | 2 |
| Sliding window | 20,000 Gaussians |
| PSNR | 23.52 dB |
| Optimization time | 15.3 ms |
| Map maintenance | 18.9 ms |
| **Total pipeline** | **48.3 ms (~20 Hz)** |

Jetson Orin NX 配置:8-core CPU + 1024 CUDA cores + 16GB LPDDR5 shared memory。

Paper 还 demo 了完整 autonomous navigation:Gaussian map → 2D occupancy grid → A* planning → LQR tracking。说明这玩意儿真跑得起来 robot。

Reference: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin-nx/

---

## 为什么这件事重要:Intuition 级别

### 1. Memory locality > Raw GPU power

一般思路:"我 GPU 不够强,换更强的 GPU。"

GS-LIVO 思路:"我 GPU 不强,那就把 working set bounded。"

通过 sliding window,GPU 永远只处理 20K-100K Gaussians,不管 global map 多大。这跟 LLM 中的 KV cache、database 中的 buffer pool、OS 中的 page cache 是同一种思想——**working set 假设 + locality**。

### 2. LiDAR prior 是 Gaussian init 的 game-changer

原版 3D-GS 从 SfM points 初始化,需要 thousands of iterations 才能 converge。原因:SfM points 是 sparse、noisy、unstructured,optimization 要从混乱中找出 structure。

LiDAR 给你的是 dense、accurate、有 surface normal 的 geometry。直接 init 成 well-structured planar Gaussians,跳过混乱探索期。这跟 NeRF 场景下 "deep supervision" 的道理一样——**好 prior 胜过 brute-force optimization**。

### 3. Tightly coupled covariance propagation

大多数 Gaussian-SLAM 用 gradient descent 算 pose,丢了 covariance。GS-LIVO 通过 IESKF 显式 propagate pose covariance 到下一个 sensor update。

为什么这重要?考虑 robot 在隧道里快速运动:
- IMU 200Hz propagate,告诉你 pose 大概在哪
- LiDAR 10Hz update,refine pose
- Visual 10Hz update,再 refine

每次 update 都需要前一个 sensor 的 posterior covariance 作为 prior。没有 covariance,你只能 loosely coupled,每次 visual update 都从头开始优化,robustness 大打折扣。

### 4. Photometric loss on current frame > Patch warping to reference

传统 LIVO(FAST-LIVO2)是 patch warping 到 reference frame。问题:
- Patch 大了有 seams(Fig. 4)
- Patch 小了 noise 敏感
- 只能 handle Lambertian

GS-LIVO 直接 render Gaussian map 到 current frame,跟 captured image 比 photometric loss。优势:
- Seamless(Gaussian blending 天然连续)
- Photorealistic(spherical harmonic model view-dependent color)
- Differentiable(Gaussian rendering 可微分,end-to-end optimization)

---

## Limitations 与 Future Work

### Self-disclosed Limitations

1. **Fixed-level octree**:当前用 fixed subdivision level(2 layers),没有自适应 LoD based on viewing distance、texture richness。作者在 Conclusion 里提 future work。

2. **No Gaussian merging**:homogeneous region(白墙、天空)没合并相似 color 的 Gaussians,可能 redundant storage。

3. **Outdoor PSNR 低于 offline**:coarse root voxel (1.0m) 让 fine details 丢失。

4. **Sliding window 限制 global co-visibility**:只 optimize FoV 内的,无法做 global bundle adjustment。Loop closure 时可能 inconsistency。

5. **Static scene assumption**:dynamic objects(行人、车辆)会被错误 incorporate 进 map。

### Speculative Future Directions

- **Adaptive LoD**:类似 Octree-GS [45] 的 dynamic level selection based on viewing distance
- **Gaussian compression**:类似 Motion-GS [43]、RTG-SLAM [44] 的 compact representation
- **Loop closure**:sliding window 之外的 global consistency mechanism
- **4D-GS for dynamic**:time-varying Gaussians for moving objects
- **Semantic SLAM**:per-category Gaussian optimization

---

## 一句话总结

GS-LIVO = LiDAR-Inertial-Visual 传统 robust LIVO + 3D-Gaussian-Splatting 高质量 map representation + 工程上的 hash-octree + sliding window 让它在 Jetson 上跑 20Hz。

技术核心三个 trick:
1. **Hash-octree + RAM/VRAM 分离**:GPU memory bounded
2. **LiDAR-visual joint init**:Gaussian 收敛快
3. **IESKF + photometric Jacobian**:tightly coupled 与传统 LIO 兼容

数据上三个 highlight:
1. **40x speedup** vs LetsGo rendering
2. **8x memory reduction** vs LetsGo
3. **15-47x accuracy improvement** vs SplaTAM/MonoGS

这是 Gaussian-SLAM 走向实际 robot 部署的关键一步。代码与 hardware CAD 在 https://github.com/HKUST-Aerial-Robotics/GS-LIVO 开源。

---

# GS-LIVO: Real-Time LiDAR-Inertial-Visual Odometry with Gaussian Mapping 深度解析

## 1. Background 与 Motivation

这篇 paper 来自 HKUST Aerial Robotics Group (香港科技大学空中机器人实验室),作者团队 Sheng Hong, Chunran Zheng, Tong Qin, Shaojie Shen 等是 FAST-LIVO 系列的核心开发者。Paper 的核心 motivation 来自一个关键观察:当前 3D-Gaussian Splatting (3DGS) based SLAM 系统 [1, 25, 33-37] 普遍存在一个 **"odometry fast, map slow" 的 bottleneck**。

具体来说:
- **Pose estimation thread** 通常能跑到 near real-time (10-30 Hz)
- **Map optimization thread** 却 lag behind,依赖 separate slower thread (off-line processing or 1-3 Hz update rate)
- 在 dynamic 或 large-scale environments 中,这种 mismatch 导致 robot 无法快速 interpret surroundings

GS-LIVO 想解决的核心问题:**如何在 maintain photorealistic dense Gaussian map 的同时,实现 high-frequency (10 Hz indoor, 3 Hz outdoor) 的 map update**。

Reference: 
- 3D-GS original paper: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- SplaTAM: https://sunglokku.github.io/splatam/
- MonoGS: https://rmurai.co.uk/projects/GaussianSplatterSLAM/

---

## 2. System Architecture 总览

系统包含四个 core modules (参见 Fig. 2):

| Module | 功能 | 关键技术 |
|--------|------|---------|
| Global Gaussian Map | 全局地图存储与索引 | Hash-indexed octree (Sec II-A) |
| Gaussian Initialization & Optimization | 高质量 Gaussian 生成与 photometric refinement | LiDAR-camera joint init + photometric gradient (Sec II-B) |
| Sliding Window Maintenance | 实时 GPU optimization scope 控制 | Incremental update with SHT/CGB/GGB (Sec II-C) |
| IESKF State Estimation | Tightly coupled multi-sensor fusion | Sequential updates with photometric residual Jacobian (Sec II-D) |

**Intuition**: 这个架构本质上是把传统 LIVO (如 FAST-LIVO2 [8]) 的 sparse patch-based map 替换为 dense Gaussian map,同时保留 IESKF 的 tightly coupled fusion 框架。关键 insight 是:**Gaussian map 既是 rendering 用的 photorealistic representation,又直接作为 visual measurement model 的 reference**,这样 mapping 与 localization形成正向反馈回路。

---

## 3. Global Gaussian Map: Hash-indexed Octree

### 3.1 为什么需要 hierarchical structure

直接用 flat hash table 存 Gaussians 的问题:
- 大场景下 Gaussian 数量爆炸 (millions to billions)
- 无法 handle varying Levels of Detail (LoD)
- GPU memory 受限

Octree 的好处:
- 自适应 subdivision,complex region 细分,空旷 region 保持 coarse
- 天然支持 LoD rendering (远距离用 coarse level,近距离用 fine level)
- 与 spatial hashing 结合实现 O(1) voxel lookup

### 3.2 Spatial Hash Key 公式

$$\text{HashKey} = \left\lfloor \frac{{}^W \mathbf{p}_i}{\mathbf{v}_s} \right\rfloor$$

**变量解释**:
- ${}^W \mathbf{p}_i \in \mathbb{R}^3$: 第 i 个 Gaussian 的 mean (center) 在 world frame 中的 3D position
- $\mathbf{v}_s \in \mathbb{R}^+$: root voxel 的 edge length (hyperparameter,indoor 0.03m,outdoor 0.5-1.0m)
- $\lfloor \cdot \rfloor$: floor operation,对每个 component 分别取整
- 输出 HashKey 是 3D integer vector $(k_x, k_y, k_z)$

**Intuition**: 这是一个 uniform spatial discretization,把 continuous 3D space 划分成 root voxels of size $\mathbf{v}_s^3$。每个 root voxel 内部还可以递归 subdivided 成 octree 的 child nodes (paper 中用 2 layers)。

### 3.3 内存架构 (RAM + VRAM 分离)

关键设计 decision (见 Fig. 3):
- **Global Gaussian Map**: 存在 RAM 中,用 non-contiguous hash-octree 结构 (capacity 大,可 swap 扩展)
- **Sliding Window of Gaussians**: FoV 内的 Gaussians 复制到 contiguous RAM region (CGB),再 transfer 到 VRAM (GGB)
- GPU 只 optimize 当前 FoV 内的 Gaussians
- Optimization 完成后,updated Gaussians 写回 global map

这种设计的 insight:**VRAM 是稀缺资源 (Jetson Orin NX 只有 16GB shared memory),RAM 充裕且可扩展**。通过 spatial locality 假设 (consecutive frames share 大部分 FoV),把 working set 限制在 sliding window 内,实现 GPU memory 的 bounded usage。

Reference - Octree-GS hierarchical structure: https://arxiv.org/abs/2403.17898

---

## 4. Gaussian Initialization 与 Optimization

### 4.1 LiDAR-Camera Joint Initialization

这是 paper 的一个 key insight:**不从 random 或 SfM points 初始化 Gaussian,而是用 LiDAR 的 geometric prior 直接 initialize well-structured Gaussians**。

#### 4.1.1 Scaling Matrix

$$\mathbf{S}_i = \text{diag}(\mathbf{s}_\delta, \mathbf{s}_y, \mathbf{s}_z)$$

**变量解释**:
- $\mathbf{s}_\delta \in \mathbb{R}^+$: 极小数值 (paper 中 hyperparameter),让 Gaussian 在 surface 切向方向 thin (slice feature)
- $\mathbf{s}_y, \mathbf{s}_z \in \mathbb{R}^+$: surface 切平面内两个方向的 scale,通常与 voxel level 相关

**Intuition**: Gaussian 被建模为 **2D planar disk attached to object surface**,而不是 isotropic 3D ball。这与 surfel-based 方法 (FAST-LIVO) 的思路一致,但用 Gaussian 表示更适合 differentiable rendering。

#### 4.1.2 Rotation Matrix from Surface Normal

$$
{}^W \mathbf{R}_i = \begin{pmatrix}
\frac{\mathbf{e}_x \times {}^W \mathbf{n}_i}{\|\mathbf{e}_x \times {}^W \mathbf{n}_i\|} &
\frac{{}^W \mathbf{n}_i \times (\mathbf{e}_x \times {}^W \mathbf{n}_i)}{\|\mathbf{e}_x \times {}^W \mathbf{n}_i\|} &
{}^W \mathbf{n}_i
\end{pmatrix}
$$

**变量解释**:
- ${}^W \mathbf{n}_i \in \mathbb{R}^3$: surface 在第 i 个 Gaussian 位置的 normal vector (world frame),由 LiDAR-inertial SLAM [56] 提供
- $\mathbf{e}_x = (1,0,0)^T$: x-axis 的 unit vector
- $\times$: cross product
- $\|\cdot\|$: L2 norm

**Intuition**: 构造一个 rotation matrix,其三列分别是 Gaussian local frame 的三个 axis。第三列是 surface normal ${}^W \mathbf{n}_i$ (Gaussian 的 "厚度方向"),第一列是 $\mathbf{e}_x$ 与 normal cross product 的 normalized 结果 (在切平面内沿 x 方向的 component),第二列由 Gram-Schmidt 正交化得到。

注意 special case: 当 ${}^W \mathbf{n}_i \parallel \mathbf{e}_x$ 时 cross product 为零,需要 fallback 到其他 reference axis (paper 没有详细讨论,这是一个 implementation detail)。

#### 4.1.3 Covariance Matrix Construction

$$\pmb{\Sigma}_{3D,i} = ({}^W \mathbf{R}_i \mathbf{S}_i)({}^W \mathbf{R}_i \mathbf{S}_i)^T$$

**变量解释**:
- $\pmb{\Sigma}_{3D,i} \in \mathbb{R}^{3 \times 3}$: 第 i 个 Gaussian 的 3D covariance matrix
- ${}^W \mathbf{R}_i \in SO(3)$: rotation matrix
- $\mathbf{S}_i = \text{diag}(\mathbf{s}_\delta, \mathbf{s}_y, \mathbf{s}_z)$: scaling matrix

**Intuition**: 这是 3D-GS 的标准 covariance parameterization $\Sigma = R S S^T R^T = R \Sigma_S R^T$,其中 $\Sigma_S = S S^T$ 是 diagonal scaling matrix。这种 parameterization 保证 $\Sigma$ 是 symmetric positive semi-definite (SPSD),且便于通过李代数进行优化。

### 4.2 3D-to-2D Projection (Splatting)

$$\mathbf{q}_i = \pi({}^C \mathbf{T}_W {}^W \mathbf{p}_i)$$

$$\pmb{\Sigma}_{2D,i} = (\mathbf{J}_\pi {}^C \mathbf{R}_W) \pmb{\Sigma}_{3D,i} (\mathbf{J}_\pi {}^C \mathbf{R}_W)^T$$

**变量解释**:
- $\mathbf{q}_i \in \mathbb{R}^2$: Gaussian 在 image plane 上的 2D mean (pixel coordinate)
- $\pi(\cdot)$: pinhole camera projection model
- ${}^C \mathbf{T}_W \in SE(3)$: world-to-camera transformation
- ${}^C \mathbf{R}_W \in SO(3)$: 上述 transformation的 rotation 部分
- $\mathbf{J}_\pi \in \mathbb{R}^{2 \times 3}$: projection model $\pi$ 的 Jacobian (local affine approximation)
- $\pmb{\Sigma}_{2D,i} \in \mathbb{R}^{2 \times 2}$: 2D Gaussian 的 covariance

**Intuition**: 这是 EWA (Elliptical Weighted Average) splatting [57] 的标准公式。核心 idea 是用 $\mathbf{J}_\pi$ 对 nonlinear projection $\pi$ 做 local affine approximation,这样 3D Gaussian 经过 linear transform 后仍然是 Gaussian (closure property of Gaussian under linear map)。

### 4.3 Color Initialization (Bilinear Interpolation)

$$\mathbf{c}(\mathbf{q}_i) = \sum_{j=1}^{4} \mathbf{c}_j \cdot \mathbf{A}_j$$

**变量解释**:
- $\mathbf{c}(\mathbf{q}_i) \in \mathbb{R}^3$: projected pixel $\mathbf{q}_i$ 处的 RGB color
- $\mathbf{c}_j \in \mathbb{R}^3$: 4 个 neighboring integer pixels 的 color
- $\mathbf{A}_j \in \mathbb{R}$: weight,与 $\mathbf{q}_i$ 到 4 个邻居 pixel 的 distance 对应的 area

**Intuition**: 这是标准 bilinear interpolation。Paper 只 initialize spherical harmonic 的 zero order (即 DC component),higher orders 初始化为 0,后续通过 photometric optimization 学习。

### 4.4 Photometric Rendering (Alpha Blending)

$$\widehat{\mathbf{I}}_k(\mathbf{q}_i) = \sum_{i=1}^{M} \left[ \mathbf{c}_i \sigma_i G_i^{2D}(\mathbf{q}_i) \prod_{j=1}^{i-1} (1 - \sigma_j G_j^{2D}(\mathbf{q}_j)) \right]$$

**变量解释**:
- $\widehat{\mathbf{I}}_k(\mathbf{q}_i) \in \mathbb{R}^3$: pixel $\mathbf{q}_i$ 处的 rendered color
- $M$: 影响 pixel $\mathbf{q}_i$ 的 Gaussian 数量 (按 depth 排序)
- $\mathbf{c}_i \in \mathbb{R}^3$: 第 i 个 Gaussian 的 color (SH evaluated at viewing direction)
- $\sigma_i \in [0,1]$: 第 i 个 Gaussian 的 opacity
- $G_i^{2D}(\mathbf{q}_i)$: 2D Gaussian evaluated at $\mathbf{q}_i$,即 $\mathcal{N}(\mathbf{q}_i; \mathbf{q}_i^{\text{center}}, \pmb{\Sigma}_{2D,i})$
- $\prod_{j=1}^{i-1} (1 - \sigma_j G_j^{2D})$: transmittance,前面所有 Gaussian 的"剩余 opacity"

**Intuition**: 这是 front-to-back volumetric alpha blending。每个 Gaussian 贡献 color 的同时也"挡住"后面的 Gaussian。这跟 NeRF 的体积渲染公式同构,但用 explicit Gaussian 代替 implicit MLP。

### 4.5 Photometric Optimization Objective

$$\theta_{k-1}^* = \arg\min_{\theta_{k-1}} \sum_{G_i^{3D} \in \theta_{k-1}} \left\| \mathbf{I}_{k-1} - \widehat{\mathbf{I}}_{k-1}({}^W \mathbf{T}_C; \theta_{k-1}) \right\|$$

**变量解释**:
- $\theta_{k-1}$: 当前 FoV 内所有 Gaussians 的 structure parameters (mean, covariance, opacity) + spherical harmonic coefficients
- $\mathbf{I}_{k-1}$: captured image (ground truth observation)
- $\widehat{\mathbf{I}}_{k-1}({}^W \mathbf{T}_C; \theta_{k-1})$: rendered image,依赖 camera pose ${}^W \mathbf{T}_C$ 和 Gaussian parameters $\theta_{k-1}$

**Intuition**: 这是 photometric loss (L2 norm over pixel differences)。Paper 用 Adam optimizer 迭代更新 Gaussian parameters,使得 rendered image 匹配 observed image。这与 NeRF 的 photometric loss 一致,但 optimize 的是 explicit Gaussian parameters 而非 MLP weights。

---

## 5. Sliding Window of Gaussians: Incremental Maintenance

这是 paper 的另一个 key contribution,解决了 GPU memory bottleneck。

### 5.1 三层 Buffer 架构

| Buffer | 位置 | 作用 | Layout |
|--------|------|------|--------|
| **Spatial Hash Table (SHT)** | CPU RAM | 空间坐标 → memory pointer 的索引 | Hash table |
| **CPU Gaussian Buffer (CGB)** | CPU RAM | 当前 active voxels 的 Gaussian parameters | Contiguous |
| **GPU Gaussian Buffer (GGB)** | GPU VRAM | 供 GPU 并行 optimize 与 render 用的 Gaussian parameters | Contiguous |

### 5.2 五步增量更新流程

**Step 1: Update to Global Map**
- 识别 previous sliding window 中仍在 current FoV 内的 voxels (标记为 OVERLAP)
- FoV 外的 voxels: optimized parameters copy 回 global map,标记为 DELETE

**Step 2: Deletion and Compaction**
- 将 DELETE voxels 与 sliding window 末尾的 voxels swap
- 释放末尾空间,保持 memory continuity

**Step 3: Overlap and Addition**
- 用 current LiDAR frame 计算 spatial hash keys
- 识别与 previous sliding window 的 OVERLAP voxels
- 识别 new areas 需要 ADD 进 sliding window

**Step 4: Appending New Leaf Voxels**
- 将 ADD voxels append 到 CGB 末尾,更新 SHT
- CGB → GGB 直接 transfer (host to device)

**Step 5: GPU Optimization**
- GGB 上的 Gaussians 在 GPU 上做 photometric optimization
- Optimized parameters 写回 GGB → CGB → global map

**Intuition**: 核心 idea 是利用 **temporal coherence** (consecutive frames 90%+ FoV overlap)。如果不做 incremental update,每 frame 都重建 sliding window 需要 O(N) memory copy,这里只需 O(ΔN) = O(N × frame_motion_ratio),大幅减少 host-device transfer overhead。

### 5.3 Ablation Study 数据

从 Fig. 10 (indoor Playground01) 与 Fig. 11 (outdoor HKisland03) 可以看出:
- **With sliding window**: VRAM usage 稳定在 ~2-3 GB,processing time < 100 ms
- **Without sliding window**: VRAM usage 随 map size linearly 增长,processing time 随 Gaussian 数量超线性增长

PSNR 对比:
- Sliding window: PSNR 稳定在 25-30 dB (viewpoint change 时短暂下降,optimization 后恢复)
- Full GPU: PSNR 始终 25-30 dB (理论上略高,因为 optimize 整个 map)
- **关键 finding**: sliding window 几乎不损失 mapping quality,但极大降低 memory 与 computation

---

## 6. IESKF State Estimation with Sequential Updates

### 6.1 与 FAST-LIVO2 的关系

Paper 明确提到 visual update pipeline 是 **redesigned from FAST-LIVO2 [8]**,核心区别:
- FAST-LIVO2: 把 current frame 的 patch warp 到 reference frame,计算 patch-based photometric error
- GS-LIVO: **render Gaussian map 到 current frame pose,直接与 current captured image 比较**

好处 (见 Fig. 4):
- Patch-based 方法在大 patch size 时出现 seams (warping artifacts)
- Gaussian rendering 提供 seamless、photorealistic 的合成,且能 handle non-Lambertian surfaces

### 6.2 Visual Measurement Model

$$
{}^W \mathbf{T}_C^* = \arg\min_{\mathbf{T}(\boldsymbol{\xi})} \sum_{G_i^{3D} \in \theta_{k-1}} \left\| \mathbf{I}_k - \widehat{\mathbf{I}}_k({}^W \mathbf{T}_C; \theta_{k-1}) \right\|
$$

**变量解释**:
- ${}^W \mathbf{T}_C^* \in SE(3)$: optimized camera pose (world frame)
- $\boldsymbol{\xi} \in \mathbb{R}^6$: camera pose 的 Lie algebra (se(3)),即 $\boldsymbol{\xi} = (\boldsymbol{\omega}, \boldsymbol{\rho})$ 其中 $\boldsymbol{\omega}$ 是 rotation 增量、$\boldsymbol{\rho}$ 是 translation 增量
- $\mathbf{T}(\boldsymbol{\xi}) = \exp(\boldsymbol{\xi}^\wedge)$: 从 Lie algebra 到 Lie group 的 exponential map
- $\mathbf{I}_k$: 第 k 帧的 captured image
- $\widehat{\mathbf{I}}_k$: 从 Gaussian map 在 pose ${}^W \mathbf{T}_C$ 下 rendered 的 image
- $\theta_{k-1}$: 已 optimized 的 Gaussian parameters (上一帧的结果)

**Intuition**: 这与 Eq. (9) 的 Gaussian optimization objective 形式相同,但 optimize 的 variable 不同。Eq. (9) 优化 Gaussian parameters $\theta$ (pose fixed),Eq. (10) 优化 camera pose $\boldsymbol{\xi}$ (Gaussian parameters fixed)。这是 **alternating optimization** 的经典 pattern。

### 6.3 Jacobian 推导:从 Photometric Loss 到 IMU Pose

这是 paper 最 technical 的部分,Appendix 给出完整推导。核心 idea 是 **chain rule 把 loss 对 IMU pose 的 Jacobian 拆成若干小 Jacobian 的乘积**。

#### 6.3.1 Mean Value Jacobian

$$\frac{\partial \mathbf{q}_i}{\partial {}^W \mathbf{R}_I} = \frac{\partial \mathbf{q}_i}{\partial {}^C \mathbf{p}_i} \frac{\partial {}^C \mathbf{p}_i}{\partial {}^C \mathbf{R}_W} \frac{\partial {}^C \mathbf{R}_W}{\partial {}^W \mathbf{R}_I}$$

$$\frac{\partial \mathbf{q}_i}{\partial {}^W \mathbf{t}_I} = \frac{\partial \mathbf{q}_i}{\partial {}^C \mathbf{p}_i} \frac{\partial {}^C \mathbf{p}_i}{\partial {}^C \mathbf{t}_W} \frac{\partial {}^C \mathbf{t}_W}{\partial {}^W \mathbf{t}_I} + \frac{\partial \mathbf{q}_i}{\partial {}^C \mathbf{p}_i} \frac{\partial {}^C \mathbf{p}_i}{\partial {}^C \mathbf{R}_W} \frac{\partial {}^C \mathbf{R}_W}{\partial {}^W \mathbf{t}_I}$$

**变量解释**:
- ${}^W \mathbf{R}_I \in SO(3)$: IMU orientation in world frame (要优化的 state)
- ${}^W \mathbf{t}_I \in \mathbb{R}^3$: IMU position in world frame (要优化的 state)
- ${}^C \mathbf{p}_i \in \mathbb{R}^3$: Gaussian i 在 camera frame 的 3D position
- ${}^C \mathbf{R}_W, {}^C \mathbf{t}_W$: world-to-camera transform 的 rotation/translation

**Intuition**: 第一项 $\frac{\partial \mathbf{q}_i}{\partial {}^W \mathbf{R}_I}$ 通过 4 个中间 variable 链式计算:
1. $\frac{\partial \mathbf{q}_i}{\partial {}^C \mathbf{p}_i}$: projection $\pi$ 的 Jacobian (类似 pinhole camera 的 $\partial (u,v)/\partial (X,Y,Z)$)
2. $\frac{\partial {}^C \mathbf{p}_i}{\partial {}^C \mathbf{R}_W}$: 点在 camera frame 中对 rotation 的 sensitivity
3. $\frac{\partial {}^C \mathbf{R}_W}{\partial {}^W \mathbf{R}_I}$: extrinsic ${}^C \mathbf{R}_I$ 引入的耦合

第二项 ${}^W \mathbf{t}_I$ 的 Jacobian 有 **两 contributions**: 第一项是 translation 的直接 chain,第二项揭示一个 subtle fact — **IMU translation 变化也会通过 extrinsic rotation 影响 camera-to-world rotation**(因为 extrinsic 是 fixed offset,IMU move 时 camera 也 move + rotate)。

#### 6.3.2 Covariance Jacobian

$$\frac{\partial \pmb{\Sigma}_{2D}}{\partial {}^W \mathbf{R}_I} = \frac{\partial \pmb{\Sigma}_{2D}}{\partial \mathbf{J}_\pi} \frac{\partial \mathbf{J}_\pi}{\partial {}^C \mathbf{p}_i} \frac{\partial {}^C \mathbf{p}_i}{\partial {}^C \mathbf{R}_W} \frac{\partial {}^C \mathbf{R}_W}{\partial {}^W \mathbf{R}_I} + \frac{\partial \pmb{\Sigma}_{2D}}{\partial {}^C \mathbf{R}_W} \frac{\partial {}^C \mathbf{R}_W}{\partial {}^W \mathbf{R}_I}$$

**Intuition**: 2D covariance $\pmb{\Sigma}_{2D}$ 通过两个 path 依赖 IMU rotation:
1. 通过 $\mathbf{J}_\pi$ (projection Jacobian,因为 $\mathbf{J}_\pi$ 依赖 ${}^C \mathbf{p}_i$ 而 ${}^C \mathbf{p}_i$ 依赖 rotation)
2. 通过 ${}^C \mathbf{R}_W$ 直接耦合 (Eq. 6 中 $\pmb{\Sigma}_{2D}$ 显式含 ${}^C \mathbf{R}_W$)

### 6.4 Camera-IMU Extrinsic Jacobian 推导

Appendix B 给出用 **perturbation method** 推导的关键结果:

$$\frac{\partial {}^C \mathbf{R}_W}{\partial {}^W \mathbf{R}_I} = -{}^C \mathbf{R}_I$$

$$\frac{\partial {}^C \mathbf{t}_W}{\partial {}^W \mathbf{t}_I} = -{}^W \mathbf{R}_C^T$$

$$\frac{\partial {}^C \mathbf{R}_W}{\partial {}^W \mathbf{t}_I} = -{}^C \mathbf{t}_I^\wedge {}^C \mathbf{R}_W$$

**推导 key step**: 从 perturbation model 出发,

$$(\mathbf{T}(\delta\varphi, \delta t) \cdot {}^C \mathbf{T}_W)^{-1} \cdot {}^C \mathbf{T}_I = {}^W \mathbf{T}_I \boxplus \mathbf{T}(\delta R, \delta \rho)$$

其中 $(\delta\varphi, \delta t)$ 是 camera pose 上的 perturbation,$(\delta R, \delta \rho)$ 是 IMU pose 上的 perturbation。

**Rotation 部分推导**:
1. 分解 rotation: $(\text{Exp}(\delta\varphi^\wedge) {}^C \mathbf{R}_W)^{-1} {}^C \mathbf{R}_I = {}^W \mathbf{R}_I \text{Exp}(\delta R^\wedge)$
2. Take inverse 并 substitute: ${}^W \mathbf{R}_C \text{Exp}(-\delta\varphi^\wedge) {}^C \mathbf{R}_I = {}^W \mathbf{R}_I \text{Exp}(\delta R^\wedge)$
3. First-order approximation $\text{Exp}(\xi^\wedge) \approx I + \xi^\wedge$:
   $${}^W \mathbf{R}_I - {}^W \mathbf{R}_C \delta\varphi^\wedge {}^C \mathbf{R}_I = {}^W \mathbf{R}_I + {}^W \mathbf{R}_I \delta R^\wedge$$
4. Cancel ${}^W \mathbf{R}_I$: $-{}^I \mathbf{R}_C \delta\varphi^\wedge {}^C \mathbf{R}_I = \delta R^\wedge$
5. 用 skew-symmetric 性质: $(-{}^I \mathbf{R}_C \delta\varphi)^\wedge = \delta R^\wedge$
6. 得到: $\delta\varphi = -{}^C \mathbf{R}_I \delta R$
7. 最终: $\frac{\partial {}^C \mathbf{R}_W}{\partial {}^W \mathbf{R}_I} = -{}^C \mathbf{R}_I$

**Translation 部分** (类似推导,见 Appendix B.3):
- 一阶展开后得到: ${}^W \mathbf{R}_C {}^C \mathbf{t}_I^\wedge \delta\varphi - {}^W \mathbf{R}_C \delta t = \delta \rho$
- 代入 $\delta\varphi = -{}^C \mathbf{R}_I \delta R$ 后取系数,得到 $\frac{\partial {}^C \mathbf{t}_W}{\partial {}^W \mathbf{t}_I} = -{}^W \mathbf{R}_C^T$ 和 $\frac{\partial {}^C \mathbf{R}_W}{\partial {}^W \mathbf{t}_I} = -{}^C \mathbf{t}_I^\wedge {}^C \mathbf{R}_W$

**Intuition**: ${}^C \mathbf{t}_I^\wedge$ 是 ${}^C \mathbf{t}_I$ 的 skew-symmetric matrix,捕捉 IMU-camera extrinsic translation 引入的 rotation-translation coupling。这是经典的 hand-eye calibration 中出现的项。

### 6.5 IESKF Update Framework

#### 6.5.1 状态与协方差

- **Prior state**: ${}^W \hat{\mathbf{T}}_I$ (来自 IMU propagation 或 LiDAR update 后的结果),covariance $\pmb{\Sigma}_{{}^W \hat{\mathbf{T}}_I^-}$
- **Posterior state**: ${}^W \check{\mathbf{T}}_I$ (visual update 后),covariance $\pmb{\Sigma}_{{}^W \check{\mathbf{T}}_I}$

#### 6.5.2 Measurement Model with Linearization

$$\mathbf{M}({}^W \hat{\mathbf{T}}_I, \mathbf{u}_i) + \mathbf{J}_i \boldsymbol{\xi}$$

**变量解释**:
- $\mathbf{M}({}^W \hat{\mathbf{T}}_I, \mathbf{u}_i)$: measurement function (Gaussian rendering) 在 prior state 处的值
- $\mathbf{u}_i$: 第 i 个 measurement (可以是 pixel intensity 或 patch)
- $\mathbf{J}_i \in \mathbb{R}^{m \times 6}$: 第 i 个 measurement 对 IMU pose perturbation $\boldsymbol{\xi}$ 的 Jacobian
- $\boldsymbol{\xi} \in \mathbb{R}^6$: Lie algebra perturbation

#### 6.5.3 Optimization Objective

$$
\arg\min_{\delta {}^W \hat{\mathbf{T}}_I} \sum_i \left\| \mathbf{M}({}^W \hat{\mathbf{T}}_I, \mathbf{u}_i) + \mathbf{J}_i T(\boldsymbol{\xi}) \right\|_{\pmb{\Sigma}_{\mathbf{u}_i}^{-1}}^2 + \left\| {}^W \hat{\mathbf{T}}_I \boxplus {}^W \bar{\mathbf{T}}_I + \mathcal{H} T(\boldsymbol{\xi}) \right\|_{\pmb{\Sigma}_{{}^W \hat{\mathbf{T}}_I^-}^{-1}}^2
$$

**变量解释**:
- 第一项: visual measurement residual (Mahalanobis norm)
- 第二项: prior state residual (Mahalanobis norm)
- $\pmb{\Sigma}_{\mathbf{u}_i}$: measurement i 的 noise covariance
- $\mathcal{H}$: $\boxplus$ 操作的 Jacobian
- $T(\boldsymbol{\xi})$: Lie algebra 到 group 的 exponential map

#### 6.5.4 Kalman Gain 与 Update

$$\mathbf{H} = [\mathbf{J}_1^T, \cdots, \mathbf{J}_m^T]^T$$

$$\mathbf{R} = \text{diag}(\pmb{\Sigma}_{\mathbf{u}_1}, \cdots, \pmb{\Sigma}_{\mathbf{u}_m})$$

$$\mathbf{P} = \mathcal{H}^{-1} \pmb{\Sigma}_{{}^W \hat{\mathbf{T}}_I^-} \mathcal{H}^{-T}$$

$$\mathbf{K} = (\mathbf{H}^T \mathbf{R}^{-1} \mathbf{H} + \mathbf{P}^{-1})^{-1} \mathbf{H}^T \mathbf{R}^{-1}$$

$${}^W \check{\mathbf{T}}_I = {}^W \hat{\mathbf{T}}_I \boxplus T(\boldsymbol{\xi})$$

$$\pmb{\Sigma}_{{}^W \check{\mathbf{T}}_I} = (\mathbf{I} - \mathbf{K}\mathbf{H}) \mathbf{P}$$

$$T(\boldsymbol{\xi}) = -\mathbf{K}\mathbf{z} - (\mathbf{I} - \mathbf{K}\mathbf{H})(\mathcal{H})^{-1}({}^W \hat{\mathbf{T}}_I \boxplus {}^W \bar{\mathbf{T}}_I)$$

**Intuition**: 这是标准 IESKF (Iterated Error-State Kalman Filter) 的 update equations [2, 3, 7, 8]。关键 feature:
- $\mathbf{K}$ 是 Kalman gain,平衡 measurement noise $\mathbf{R}$ 与 prior covariance $\mathbf{P}$
- $\mathbf{H}^T \math Paper highlight 的点:**unlike 大多数 Gaussian-SLAM 方法 (只用 optimizer 算 pose),GS-LIVO 显式 propagate pose covariance 到下一个 sensor update (IMU propagation, LiDAR update),form a tightly coupled IESKF system**。这是与传统 LIO 系统兼容的关键。

Reference - FAST-LIO2 IESKF: https://github.com/hku-mars/FAST_LIO
Reference - FAST-LIVO2: https://github.com/hku-mars/FAST-LIVO2

---

## 7. Experimental Results 详解

### 7.1 Datasets

| Dataset | Type | Sequences used | Ground Truth |
|---------|------|----------------|---------------|
| FAST-LIVO2 [8] | Public | CBD03, HKU01 | MoCap-like |
| MARS-LVIG [9] | Public | HKairport01, HKisland03 | D-RTK (cm-level) |
| Self-collected | Private | Playground01, Playground02, landmark01 | MoCap |

### 7.2 Rendering Quality (Table II)

#### Indoor (HKU01, CBD03, Playground01, Playground02)

| Method | PSNR (HKU01) | Dur. (s) | Mem. (GB) |
|--------|--------------|----------|-----------|
| 3D-GS [1] | 26.22 | 2128.6 | 13.8 |
| SplaTAM [33] | 24.06 | 292.2 | 2.6 |
| MonoGS [51] | 23.51 | 258.0 | 3.1 |
| S3GS [60] | x | x | x |
| LetsGo [48] | 24.51 | 3231.3 | 18.1 |
| **GS-LIVO (Ours)** | **25.34** | **82.5** | **2.2** |

**Analysis**:
- PSNR 略低于 3D-GS (26.22 vs 25.34),但 3D-GS 是 offline 的,需要 2128s 与 13.8 GB VRAM
- 与 LetsGo (offline LiDAR-assisted): PSNR 略低 (25.34 vs 24.51),**实际上 GS-LIVO 在 HKU01 上 PSNR 更高** (25.34 vs 24.51)
- 在 CBD03 上 GS-LIVO (27.52) **超过** MonoGS (27.10)、SplaTAM (26.85)、LetsGo (25.51)、S3GS (24.92)
- 速度优势巨大: 82.5s vs SplaTAM 292s,vs LetsGo 3231s (40x speedup)
- Memory 优势: 2.2 GB vs LetsGo 18.1 GB (8x reduction)

#### Outdoor (HKisland03, HKairport01)

| Method | PSNR (HKisland03) | Dur. (s) | Mem. (GB) |
|--------|-------------------|----------|-----------|
| 3D-GS | 17.52 | 3494.1 | 21.6 |
| SplaTAM | 12.60 | 790.0 | 10.5 |
| MonoGS | 14.22 | 743.7 | 12.3 |
| LetsGo | 18.32 | 2803.3 | 17.6 |
| **GS-LIVO** | 15.32 | 82.8 | 3.2 |

**Analysis**:
- 在 outdoor 大场景下 GS-LIVO 的 PSNR (15.32) 低于 3D-GS (17.52) 和 LetsGo (18.32)
- 这是 reasonable trade-off: 大场景用 coarse root voxel (1.0m) 导致细节丢失
- 但 speed (82.8s vs 2803s-3494s) 与 memory (3.2 GB vs 17.6-21.6 GB) 优势 dramatic

### 7.3 Odometry Accuracy (Table III, IV)

#### vs Traditional LIV-based (Table III)

| Method | HKisland03 RMSE (m) | HKairport01 RMSE (m) | Dur. (ms) |
|--------|----------------------|----------------------|-----------|
| FAST-LIVO [7] | 0.51 | 0.56 | 38.9 / 44.5 |
| R3LIVE [5] | 1.71 | 1.22 | 283.3 |
| LVI-SAM [54] | 4.12 | 5.21 | 73.5 / 266.8 |
| **GS-LIVO** | **0.58** | **0.63** | **82.8 / 93.2** |

**Analysis**:
- **Outdoor**: GS-LIVO (0.58m) **远好于** R3LIVE (1.71m) 和 LVI-SAM (4.12m)
- 略差于 FAST-LIVO (0.51m),这是 reasonable trade-off,因为 GS-LIVO 同时 maintain photorealistic dense map
- Processing time (82.8 ms ≈ 12 Hz) 仍然 real-time,但慢于 FAST-LIVO (38.9 ms ≈ 25 Hz)

#### Indoor Playground (Table III)

| Method | Playground01 RMSE | Playground02 RMSE | Dur. (ms) |
|--------|-------------------|-------------------|-----------|
| FAST-LIVO | 0.005 | 0.005 | 10.5 / 8.75 |
| R3LIVE | 0.014 | 0.014 | 60.6 / 66.6 |
| LVI-SAM | 0.023 | 0.021 | 96.6 / - |
| **GS-LIVO** | **0.006** | **0.005** | **48.5 / 63.4** |

**Analysis**:
- **Indoor**: GS-LIVO (0.006m) 与 FAST-LIVO (0.005m) 几乎相同,远好于 R3LIVE (0.014m) 与 LVI-SAM (0.023m)
- 在 indoor 环境中 accuracy 几乎不损失,因为 fine root voxel (0.03m) 提供 rich geometry prior

#### vs Gaussian-based SLAM (Table IV)

| Method | Playground01 RMSE | Playground01 Dur. (ms) | Mem. (GB) |
|--------|-------------------|------------------------|-----------|
| SplaTAM [33] | 0.28 | 612.8 | 12.5 |
| MonoGS* (RGBD) [51] | 0.09 | 841.5 | 19.6 |
| MonoGS (mono) [51] | 0.18 | 541.5 | 21.0 |
| **GS-LIVO** | **0.006** | **48.5** | **1.2** |

**Analysis**:
- GS-LIVO 在 accuracy 上 **远超** 所有 Gaussian-based SLAM (0.006m vs 0.09-0.28m,即 15-47x improvement)
- Speed: 48.5ms (20 Hz) vs SplaTAM 612.8ms (1.6 Hz),即 **12x speedup**
- Memory: 1.2 GB vs MonoGS 21.0 GB,即 **17x reduction**
- 关键原因: GS-LIVO 用 LiDAR-Inertial 提供 motion prior,而 SplaTAM/MonoGS 只用 visual + depth,在 rapid motion 时容易 fail

Outdoor 场景下 SplaTAM 与 MonoGS 都 failed (RMSE = x),因为它们没有 LiDAR 与 IMU,在大尺度环境无能为力。**这验证了 LIV sensor configuration 在 large-scale 下的必要性**。

### 7.4 Embedded System Deployment (NVIDIA Jetson Orin NX)

| Metric | Value |
|--------|-------|
| Image resolution | 256×216 |
| Root voxel size | 0.5m |
| Subdivision layers | 2 |
| Sliding window size | 20,000 Gaussians |
| PSNR | 23.52 dB |
| Optimization time | 15.3 ms |
| Map maintenance time | 18.9 ms |
| Total pipeline time | 48.3 ms (~20 Hz) |

**Hardware**: Jetson Orin NX 16GB,8-core CPU + 1024 CUDA cores。

**Intuition**: 这是 paper 的 **first-of-its-kind 成果** — 据作者所知,这是第一个部署在 ARM-based embedded 平台上的 real-time Gaussian-SLAM 系统。关键 enabler 是 sliding window 设计,让 GPU working set bounded in 20K Gaussians,而 global map 可以无限扩展 in CPU RAM。

Paper 还 demo 了 autonomous navigation: Gaussian map → 2D occupancy grid → A* planning → LQR tracking。这显示了 GS-LIVO 不仅是个 SLAM system,更是个完整的 robot perception stack。

Reference - Jetson Orin NX specs: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin-nx/

---

## 8. Ablation Study: Sliding Window 的贡献

从 Fig. 10 (indoor Playground01) 与 Fig. 11 (outdoor HKisland03) 可以详细分析:

### 8.1 PSNR (Fig. 10(a), 11(a))

- **With sliding window**: PSNR 在 25-30 dB 之间,viewpoint change 时有短暂 drop,但 optimization 后快速 recover
- **Without sliding window** (full GPU): PSNR 略高且更稳定,但需要 exponential memory

**Key finding**: Sliding window 几乎不损失 quality,因为 consecutive frames 大部分 FoV overlap,optimizing sliding window 等价于 optimizing 整个 visible map。

### 8.2 Processing Time (Fig. 10(b), 11(b))

- **With sliding window**: < 100 ms consistently (10 Hz update)
- **Without sliding window**: 随 map size linearly 增长,outdoor 场景可达 seconds per frame

### 8.3 VRAM Usage (Fig. 10(e), 11(e))

- **With sliding window**: 稳定 in 2-3 GB
- **Without sliding window**: 8-22 GB,outdoor 场景无法 fit on Jetson Orin NX (16 GB)

### 8.4 Time Breakdown (Fig. 12)

- **Indoor**: 平均 23 ms overhead (window maintenance + transfer)
- **Outdoor**: 平均 71 ms overhead (由于更细 voxel 与更大 sliding window)

---

## 9. 与其他 Gaussian-SLAM 的对比

### 9.1 vs MonoGS [51]

| Aspect | MonoGS | GS-LIVO |
|--------|--------|---------|
| Sensors | RGB-D camera | LiDAR + IMU + Camera |
| Map representation | 3D-GS | Hash-octree 3D-GS + sliding window |
| Pose optimization | Gradient descent on pose | IESKF with sequential updates |
| Covariance propagation | No | Yes (tightly coupled) |
| Motion prior | Constant velocity | IMU propagation |
| Large-scale capability | Limited | Yes (octree + RAM storage) |
| Embedded deployment | No | Yes (Jetson Orin NX) |

### 9.2 vs SplaTAM [33]

| Aspect | SplaTAM | GS-LIVO |
|--------|---------|---------|
| Map growth | Add new Gaussians based on silhouette | Voxel-based densification |
| Memory scaling | Linear with map size | Bounded (sliding window) |
| Pose model | Constant velocity + depth loss | IESKF tightly coupled with IMU |
| Indoor PSNR | Lower (less geometric prior) | Higher (LiDAR geometry) |
| Outdoor | Fails | Works |

### 9.3 vs LetsGo [48]

| Aspect | LetsGo | GS-LIVO |
|--------|--------|---------|
| Mode | Offline | Online real-time |
| Sensors | Handheld polar scanner + LiDAR | Synchronized LIV rig |
| Voxel sizing | Distance-based adaptive | Fixed-level octree |
| LoD | Yes (adaptive) | Yes (fixed-level) |
| Running time | 2803-3573s | 82-93s (40x faster) |

---

## 10. Key Insights 与 Limitations

### 10.1 Key Insights (Build Intuition)

1. **Memory locality > raw GPU power**: 通过 sliding window,把 GPU 限制在 bounded working set,反而能 achieve real-time performance on embedded systems。

2. **LiDAR prior 是 game-changer for Gaussian init**: 不从 random 或 SfM points init,而是用 LiDAR 的 surface normal + position 直接 initialize well-structured planar Gaussians。这避免 3D-GS 训练初期的大量 noise Gaussian,极大加速 convergence。

3. **Photometric loss on current frame 优于 patch warping to reference**: Gaussian rendering 提供 seamless、photorealistic 的合成,且能 handle non-Lambertian surfaces。Patch warping 在大 patch 时出现 seams,小 patch 时 noise sensitive。

4. **IESKF with covariance propagation 是 tightly coupled 的关键**: 大多数 Gaussian-SLAM 只用 optimizer 算 pose,丢失 covariance。GS-LIVO 显式 propagate pose covariance 到 IMU/LiDAR update,form 完整的 tightly coupled 滤波器。

5. **Octree + Hash indexing 支持 LoD + sparse volume coverage**: 这与 hierarchical 3D-GS [26]、Octree-GS [45] 思路一致,但适配 SLAM 的 incremental construction。

### 10.2 Limitations (paper 中 self-disclosed)

1. **Fixed-level octree**: 当前用 fixed subdivision level (paper 用 2 layers),没有自适应 LoD based on viewing distance、texture richness。作者在 Conclusion 中提到 future work。

2. **Homogeneous region Gaussian merging**: 当前没有合并相似 color 的 Gaussians,可能造成 redundant storage。

3. **Outdoor PSNR 低于 offline 方法**: 由于 coarse root voxel (1.0m),细节 (如树叶纹理) 无法 fine-grained capture。

4. **Sliding window 限制 global co-visibility**: Optimize 只在 FoV 内,无法 do global bundle adjustment。Loop closure 时可能需要 re-optimization mechanism (paper 没讨论)。

5. **No dynamic object handling**: Paper 假设 static scene,dynamic objects (行人、车辆) 会被错误 incorporate into map。

### 10.3 Future Directions (推测)

- **Adaptive LoD**: 类似 Octree-GS [45] 的 dynamic level selection based on viewing distance
- **Gaussian compression**: 类似 Motion-GS [43]、RTG-SLAM [44] 的 compact representation
- **Loop closure**: 在 sliding window 之外 maintain global consistency
- **Dynamic scene**: 4D-GS (time-varying Gaussians) for moving objects
- **Semantic segmentation**: Per-category Gaussian optimization

Reference - hierarchical 3D-GS: https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/

---

## 11. 总结

GS-LIVO 的核心贡献可以浓缩为一句话:**通过 hash-octree + sliding window + LiDAR-visual joint init + IESKF tightly coupled fusion,把 3D-Gaussian Splatting 第一次真正部署到 embedded robot 平台,实现 real-time photorealistic SLAM**。

技术上的三个突破:
1. **Hash-octree global map + contiguous sliding window**: 解决大场景 Gaussian 存储 + GPU memory bottleneck
2. **LiDAR-visual joint init with surface normal**: 大幅加速 Gaussian 收敛
3. **IESKF with photometric Jacobian**: 实现 tightly coupled multi-sensor fusion,与 LIO/VIO 传统框架兼容

数据上的三个亮点:
1. **40x speedup** vs LetsGo on rendering
2. **8x memory reduction** vs LetsGo on VRAM
3. **15-47x accuracy improvement** vs SplaTAM/MonoGS on odometry

这是 Gaussian-based SLAM 走向实际 robot 部署的重要一步。Paper 的开源代码与硬件设计在 https://github.com/HKUST-Aerial-Robotics/GS-LIVO 公开,对 community 推动很大。
