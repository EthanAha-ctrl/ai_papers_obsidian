---
source_pdf: Flow4DGS-SLAM Optical Flow-Guided 4D Gaussian Splatting SLAM.pdf
paper_sha256: 0877173f06facf401852126b30395b980f379b783a6b711336a6f48e12c1eb81
processed_at: '2026-08-18T13:33:52-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Flow4DGS-SLAM 用大白话讲

Andrej, 我换个口吻重新讲一遍。这次抛开公式细节，用最直觉的方式聊聊这篇 paper 到底干了啥。

---

## 一句话概括

这篇 paper 把 4DGS-SLAM 那套 SC-GS 风格的 deformation MLP 整个换掉，改用 **optical flow 作为 feedforward prior** 来直接 propagate dynamic Gaussian 的位置，同时用 GMM 来参数化 opacity 和 rotation 的时间演化。一招把 tracking、masking、dynamic reconstruction 三个环节都串起来了，FPS 从 0.04 跳到 0.50，PSNR 在 BONN 上提升 6 dB。

---

## 痛点：4DGS-SLAM 哪里不行

先看看它要替代的 4DGS-SLAM [[1] 4DGS-SLAM, Li et al. 2025](https://github.com/yanyanli/4D-Gaussian-Splatting-SLAM) 有什么问题：

**痛点 1：Deformation MLP 训练太慢**
4DGS-SLAM 用 SC-GS [[2] SC-GS, CVPR 2024](https://github.com/yihua7/SC-GS) 那套——sparse control points + MLP deformation field。每个 mapping step 要训 100 iterations deformation MLP + 100 iterations joint optimization，总共 200 iterations。结果 mapping 一步要 110 秒，FPS 0.04。

**痛点 2：Category-based segmentation 太脆**
它用 segmentation model 来认 dynamic object，只能识别"已知的 dynamic class"（人、车之类）。遇到 balloon、ball 这种 category-agnostic 的 dynamic object 就完全 miss。BONN 数据集 [[3] BONN dataset](https://www.ipb.uni-bonn.de/data/datasets/) 里就有 balloon 场景，4DGS-SLAM 在这种场景上表现很差。

**痛点 3：复杂 dynamic 场景处理弱**
4DGS-SLAM 需要手调一个 "dynamic start time"——什么时候开始分配 dynamic Gaussians。这意味着如果一个人走出 view 再走回来，它就处理不了，因为新出现的时候没有对应的 dynamic Gaussian。

---

## 核心 insight：optical flow 是免费的 prior

这篇 paper 的灵魂 insight 是：**optical flow 这个 prior signal 可以同时干三件事**：

1. **区分 static / dynamic pixel**（mask generation）
2. **提供 camera pose 的粗初始化**（camera init）
3. **propagate dynamic Gaussian 的 3D 位置**（mapping）

这三件事原来要三个 module 分别处理，现在一个 RAFT [[4] RAFT, ECCV 2020](https://github.com/princeton-vl/RAFT) 输出全包了。这就是为啥 paper 叫 "Flow4DGS-SLAM"。

---

## 三个关键 trick

### Trick 1: Camera-Induced Motion Decomposition（Section 3.1）

这个 module 解决"如何 category-agnostic 地认出 dynamic pixel"。

**直觉**: 对 static pixel，它的 optical flow 完全由 camera 6-DoF motion + 该 pixel 的 depth 决定。这是 projective geometry 的经典结论，公式长这样：

$$\mathbf{F}(u,v) = \mathbf{J}(\mathbf{x}) \boldsymbol{\xi}$$

- $\mathbf{F}(u,v) \in \mathbb{R}^2$: pixel $(u,v)$ 处的 optical flow
- $\mathbf{x} = (u, v, Z)^\top$: 该 pixel 对应的 3D point，$Z$ 是 depth
- $\boldsymbol{\xi} \in \mathbb{R}^6$: camera twist（3 维 translation $\boldsymbol{\rho}$ + 3 维 rotation $\boldsymbol{\theta}$，在 Lie algebra $\mathfrak{se}(3)$ 里）
- $\mathbf{J}(\mathbf{x}) \in \mathbb{R}^{2\times 6}$: image Jacobian / interaction matrix，来自 visual servoing 领域 [[5] Chaumette, Hutchinson, Corke - Springer Handbook](https://link.springer.com/referencework/10.1007/978-3-319-32552-1)

image Jacobian 的显式形式：
$$\mathbf{J}(\mathbf{x}) = \begin{bmatrix} -\frac{f_x}{Z} & 0 & \frac{u}{Z} & \frac{uv}{f_y} & -f_x - \frac{u^2}{f_x} & v \\ 0 & -\frac{f_y}{Z} & \frac{v}{Z} & f_y + \frac{v^2}{f_y} & -\frac{uv}{f_x} & -u \end{bmatrix}$$

这里有个超深刻的几何 intuition：
- 前 3 列（translation 部分）每个元素都带 $1/Z$ 因子 → **translation-induced flow scales with 1/depth**（近的物体 apparent motion 大）
- 后 3 列（rotation 部分）与 depth 无关 → **rotation-induced flow is depth-independent**

这就是为啥 epipolar geometry 能解出相对 pose 的本质。

**步骤**:
1. 用 RAFT 算 optical flow $\mathbf{F}$
2. 用 YOLOv9 [[6] YOLOv9](https://github.com/WongKinYiu/yolov9) 拿 semantic mask $\mathcal{M}_s$（拦住已知 dynamic class）
3. 在 $\mathcal{M}_s=0$ 且 depth valid 的 pixels 上，用 IRLS with Cauchy weights 拟合 $\boldsymbol{\xi}$：

$$\hat{\boldsymbol{\xi}} = \arg\min_{\boldsymbol{\xi}} \sum_i w_i \|\mathbf{F}_i - \mathbf{J}_i \boldsymbol{\xi}\|^2$$

- $w_i$: Cauchy weights（iteratively reweighted），让 outlier（dynamic pixel）自然被降权

4. 算 residual $r(u,v) = \|\mathbf{F}(u,v) - \mathbf{J}\hat{\boldsymbol{\xi}}\|_2$
5. 用 robust statistics 阈值：$r > \text{median}(r) + k\cdot\text{MAD}(r)$ 就判为 dynamic

$$\mathcal{M}_{ca}(u,v) = \mathbb{1}\left(r(u,v) > \text{median}(r) + k\,\text{MAD}(r)\right)$$

- $\text{MAD} = \text{Median Absolute Deviation}$，robust 的 spread 度量 [[7] Leys et al. MAD outlier detection](https://www.sciencedirect.com/science/article/pii/S0022103113000668)

6. 最终 dynamic mask $\mathcal{M}_{dy} = \mathcal{M}_s \cup \mathcal{M}_{ca}$——semantic + geometric 互补

**妙在哪**: 这套流程把 "camera motion" 和 "object motion" 在 2D image flow 上 decompose 开来。能被 camera ego-motion 解释的就是 static，解释不了的就是 dynamic。完全不依赖 object category，所以 balloon、ball 这种都能认出来。

**Camera init 的 trick**: 在 clean pixel（$\mathcal{M}_{dy}=0$）上做第二次 weighted least-squares 拿到更准的 $\hat{\boldsymbol{\xi}}^*$，通过 exponential map 变成 SE(3) pose，作为下一步 differentiable rendering tracking 的初值：

$$\mathbf{T}_{cw}^t = \mathbf{T}_{cw}^{t-1} \exp_{\mathfrak{se}(3)}(\hat{\boldsymbol{\xi}}^*)$$

- $\mathbf{T}_{cw}^{t-1}$: 上一帧的 camera-from-world pose
- $\exp_{\mathfrak{se}(3)}(\cdot)$: Lie algebra 到 Lie group 的 exponential map，translation 部分要用 SO(3) 的 left Jacobian $\mathbf{V}(\boldsymbol{\theta}^*)$ 处理与 rotation 的耦合 [[8] Gallego & Yezzi - SE(3) derivative](https://arxiv.org/abs/1902.07220)

还有一个 robustness trick：clamp 住 maximum camera motion，threshold 正比于 inlier ratio。如果 inlier 少（说明 optical flow 噪声大），就别让 init 跑太远。

### Trick 2: Hybrid 4DGS Representation（Section 3.2）

这个 module 解决"dynamic Gaussian 该如何表征时间维度"。

**直觉**: dynamic Gaussian 有四个属性会随时间变：position、opacity、rotation、scale。这篇 paper 的核心设计是 **不同属性用不同方式建模**：

| Attribute | 表征方式 | 为什么 |
|---|---|---|
| Position $\mathbf{x}_i(t)$ | Explicit per-keyframe + linear interp | 要被 optical flow propagate、要被 adaptive insert，必须 explicit |
| Opacity $\sigma_i(t)$ | GMM (K=3) | 要表达"出现-消失"，需要 smooth parametric form |
| Rotation $\mathbf{q}_i(t)$ | GMM (K=3)，与 opacity 共享 temporal kernel | smooth + compact |
| Scale $\mathbf{s}_i$, color $\mathbf{c}_i$ | Static | 假设时间不变 |

**GMM opacity 公式**:
$$m_i(t) = 1 - \exp\left(-A_i \sum_{k=1}^{K} w_{i,k} \mathcal{N}(\hat{t}; \mu_{i,k}, \tau_{i,k}^2)\right)$$
$$\sigma_i(t) = \sigma_i \cdot m_i(t)$$

- $\hat{t} \in [0,1]$: normalized time
- $K=3$: 3 个 Gaussian bump
- $A_i > 0$: amplitude，控制 opacity 峰值
- $w_{i,k}, \mu_{i,k}, \tau_{i,k}$: 每个 component 的 weight / mean / std，全是 learnable
- 外层 $1 - \exp(-\cdot)$ 是 Poisson-style transmittance，保证 $m_i(t) \in [0, 1)$

**直觉**: 用 3 个 Gaussian bump 的加权和来拟合 opacity 时间曲线。一个 bump 在特定时间激活 = 物体出现；多个 overlapping bumps = 持续存在；多个 separated bumps = 间歇出现。非常 flexible。

**GMM rotation 公式**:
$$\mathbf{q}_i(t) = \frac{\sum_{k=1}^{K} w_{i,k} \mathcal{N}(\hat{t}; \mu_{i,k}, \tau_{i,k}^2) \mathbf{q}_{i,k}}{\left\|\sum_{k=1}^{K} w_{i,k} \mathcal{N}(\hat{t}; \mu_{i,k}, \tau_{i,k}^2) \mathbf{q}_{i,k}\right\|}$$

- $\mathbf{q}_{i,k} \in \mathbb{R}^4$: per-component control quaternion
- 分母归一化确保 $\|\mathbf{q}_i(t)\|=1$
- 与 opacity 共享 $\{w, \mu, \tau\}$，节省参数

**Explicit keyframe position**: 在 keyframes $\{t_k\}$ 上学习 $\mathbf{x}_i^k$，中间用 linear interpolation。Keyframe selection: motion mask 差异 + 每 5 帧至少 1 keyframe。

**为什么 hybrid 是 sweet spot**:
- 如果 position 也用 GMM → 失去 optical flow propagation 能力
- 如果 opacity 也用 explicit per-keyframe → storage 爆炸 + 不 smooth
- Hybrid = position explicit + opacity/rotation parametric，刚好平衡

### Trick 3: Scene Flow Propagation + Adaptive Insertion（Section 3.4）

这个 module 解决"如何在 new keyframe 上快速初始化 dynamic Gaussian 的位置"。

**Problem**: SC-GS 风格的 deformation MLP 每个 mapping step 都要从头训，慢。能不能用 optical flow 直接给出 dynamic Gaussian 在新 keyframe 上的位置初始化？

**Scene Flow Propagation**:

给定 keyframe $k-1$ 上的 dynamic Gaussian centers $\{\mathbf{x}_i^{k-1}\}$，要算 keyframe $k$ 上的 centers。

1. **Project**: 把 $\mathbf{x}_i^{k-1}$ 投影到 keyframe $k-1$ 的 image 上
   $$\mathbf{u}_i^{k-1} = \Pi(\mathbf{P}_{k-1}[\mathbf{x}_i^{k-1}])$$
   - $\mathbf{P}_{k-1} = \mathbf{K}[\mathbf{R}_{k-1}|\mathbf{t}_{k-1}]$: projection matrix
   - $\Pi(\cdot)$: homogeneous normalization

2. **Add optical flow**: 
   $$\mathbf{u}_i^k = \mathbf{u}_i^{k-1} + \mathbf{F}^{t_{k-1}, t_k}(\mathbf{u}_i^{k-1})$$

3. **Unproject to 3D**:
   $$\Delta\mathbf{x}_i^k = \mathbf{R}_k^\top\left(D_i^k \mathbf{K}^{-1}\bar{\mathbf{u}}_i^k - \mathbf{t}_k\right) - \mathbf{x}_i^{k-1}$$
   - $D_i^k$: keyframe $k$ 上该 pixel 的 depth
   - $\mathbf{K}^{-1}\bar{\mathbf{u}}_i^k$: camera-space normalized direction
   - 整个本质就是 $\text{unproject}_k(\mathbf{u}_i^k, D_i^k) - \mathbf{x}_i^{k-1}$

4. **KNN smoothing**:
   $$\Delta\hat{\mathbf{x}}_i^k = \sum_{j\in\mathcal{N}(i)} w_{ij}^{knn}\,\Delta\mathbf{x}_j^k$$
   $$w_{ij}^{knn} = \frac{\mathcal{N}\left(\|\mathbf{x}_j^{k-1} - \mathbf{x}_i^{k-1}\|_2; 0, \tau_{knn}^2\right)}{\sum_{l\in\mathcal{N}(i)} \mathcal{N}\left(\|\mathbf{x}_l^{k-1} - \mathbf{x}_i^{k-1}\|_2; 0, \tau_{knn}^2\right)}$$
   - $\mathcal{N}(i)$: nearest neighbors（半径 search）
   - $\tau_{knn}$: smoothing bandwidth
   - 强制 nearby Gaussians 有 similar motion = local rigidity prior

5. 最终 $\mathbf{x}_i^k = \mathbf{x}_i^{k-1} + \Delta\hat{\mathbf{x}}_i^k$

**直觉**: optical flow 给了 2D motion，depth 给了 3D position，unproject 就拿到 3D motion。KNN smoothing 处理 optical flow 噪声 + 强制 local rigidity。这一步完全 feedforward，无需训练。

**Adaptive Gaussian Insertion**:

**Problem**: propagation 只能更新已存在的 dynamic Gaussian。如果 keyframe $k$ 出现了 keyframe $k-1$ 没有的 dynamic region（新物体、previously occluded 部分），怎么办？

**Solution**: 用 backward optical flow 把当前 dynamic mask warp 到前一帧，找"前帧不是 dynamic 但当前是 dynamic"的 pixels：

$$\mathcal{M}_{insert}^{t_k} = \left\{\mathbf{u}_p^k \in \mathcal{M}_{dy}^{t_k} \,\middle|\, \mathbf{u}_p^{k-1} \notin \mathcal{M}_{dy}^{t_{k-1}}\right\}$$

- $\mathbf{u}_p^{k-1} = \mathbf{u}_p^k + \mathbf{F}^{t_k, t_{k-1}}(\mathbf{u}_p^k)$: backward flow warp

然后以密度 $1/D_{init}$ 随机采样 + unproject 初始化新 Gaussians。

**直觉**: 这是 motion-aware 的 Gaussian densification。一个人走出 view 再走回来，他重新出现的时候，对应的 pixel 在前帧肯定不是 dynamic（因为前帧他不在那），所以自动 insert 新 Gaussians。完全不需要 handcraft dynamic start time。

**Mapping loss**:
$$\mathcal{L}_{map} = \lambda_1 \mathcal{L}_c + \lambda_2 \mathcal{L}_d + \lambda_f \mathcal{L}_f + \lambda_m \mathcal{L}_m + \lambda_{iso} \mathcal{L}_{iso}$$

- $\mathcal{L}_c, \mathcal{L}_d$: color + depth L1 loss
- $\mathcal{L}_f$: flow loss（rendered flow vs RAFT flow），只在最后 25 iterations 用
- $\mathcal{L}_m$: binary mask loss，rendered alpha map 与 motion mask 一致
- $\mathcal{L}_{iso}$: isotropic loss，防 Gaussian 退化（继承自 4DGS-SLAM）

每个 mapping step 训 50 iterations（vs 4DGS-SLAM 的 200）。节省来自：① 没有 deformation MLP 训练；② optical flow propagation 给了好的初始化，少 iterations 就收敛。

---

## 实验：用直觉解读

### Tracking Accuracy (Table 1, TUM RGB-D [[9] TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset))

| Method | fr3/sit_st | fr3/walk_st | fr3/walk_xyz | fr3/walk_rpy | Avg. |
|---|---|---|---|---|---|
| MonoGS [[10] MonoGS, CVPR 2024](https://rmurai.io/projects/Gaussian-Splatting-SLAM/) | 0.48 | 21.9 | 30.7 | 34.2 | 15.8 |
| SplaTAM [[11] SplaTAM, CVPR 2024](https://github.com/hermosayhlabs/splatam) | 0.52 | 83.2 | 134.2 | 142.3 | 62.2 |
| 4DGS-SLAM | 0.58 | 0.61 | 2.7 | 3.0 | 2.1 |
| **Ours** | 0.70 | 0.48 | 2.5 | 3.6 | **1.9** |

**直觉解读**:
- MonoGS / SplaTAM 在 walking sequences 上完全崩（130+ cm 误差），因为它们 filter 掉 dynamic 但 tracking drift 严重
- 4DGS-SLAM 已经很好（2.1 cm），Ours 在 dynamic-heavy sequences（walk_st, walk_xyz）上再降一点
- sit_st（near-static）上 Ours 略差（0.70 vs 0.58），因为 dynamic 少，optical flow 噪声反而干扰初始化
- Overall avg 1.9 vs 2.1，**用更少 mapping iterations 取得更好 tracking**

### Rendering Quality (Table 2)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| MonoGS | 17.74 | 0.608 | 0.382 |
| SplaTAM | 19.40 | 0.757 | 0.241 |
| SC-GS [[12] SC-GS](https://github.com/yihua7/SC-GS) | 20.78 | 0.657 | 0.396 |
| 4DGS-SLAM | 22.55 | 0.788 | 0.229 |
| **Ours** | **26.55** | **0.831** | **0.177** |

**+4 dB PSNR 提升 over 4DGS-SLAM**——相当大的跃迁。主要来自 hybrid representation + scene flow propagation 让 dynamic regions 重建质量大幅提升。

### BONN 数据集 (Table 3, 4) - 更难

BONN [[13] BONN](https://www.ipb.uni-bonn.de/data/datasets/) 包含 balloon（category-agnostic）、person_tracking、sync（fast camera motion）等场景。

| Method | PSNR (balloon2) | PSNR (sync) | Avg PSNR |
|---|---|---|---|
| MonoGS | 20.22 | 22.03 | 21.06 |
| 4DGS-SLAM | 22.91 | 24.37 | 23.81 |
| **Ours** | **28.36** | **29.85** | **29.71** |

**+6 dB PSNR 提升**，比 TUM 还猛。原因：BONN 有 category-agnostic dynamic objects (balloon)，4DGS-SLAM 的 category-based segmentation 漏掉，而 Ours 的 residual-based $\mathcal{M}_{ca}$ 捕捉到。

### Runtime (Table 5)

| Method | Dynamic Seg. | Tracking | Mapping | FPS |
|---|---|---|---|---|
| MonoGS | - | 476 ms | 557 ms | 1.93 |
| 4DGS-SLAM | 16 ms | 445 ms | **110562 ms** | 0.04 |
| Ours | 68 ms | 427 ms | **6285 ms** | 0.50 |

**核心 gain 在 mapping**: 6285 ms vs 110562 ms = **17.6x speedup**。

Dynamic Seg. 多花 52 ms 是 RAFT + YOLOv9 的代价，但 mapping 的省时碾压。Tracking 时间基本持平（427 vs 445 ms），说明 tracking iteration 数没变，只是 init 更准让 tracking 收敛快。

### Ablation Study (Table 6)

| Config | walk_xyz ATE / PSNR | balloon2 ATE / PSNR |
|---|---|---|
| w/o Motion Decomp. | 2.7 / 24.40 | 7.4 / 27.59 |
| w/o Flow Propagate | 2.6 / 24.04 | 3.9 / 27.93 |
| w/o Adaptive Insert | 3.4 / 24.47 | 3.7 / 27.91 |
| w/o GMM | 2.7 / 23.91 | 3.5 / 28.14 |
| w/o KNN smooth | 2.5 / 24.47 | 3.5 / 27.93 |
| **Ours** | **2.5 / 24.60** | **3.4 / 28.36** |

**直觉 takeaways**:
- **Motion Decomp** 对 balloon2 影响最大（ATE 7.4 → 3.4 cm），因为 balloon 是 category-agnostic，没有 $\mathcal{M}_{ca}$ 就完全 miss
- **Flow Propagate** 主要影响 PSNR，对 tracking 影响小（因为是给 mapping 用的）
- **Adaptive Insert** 对复杂 dynamic 场景关键（re-entering objects）
- **GMM** 在 balloon2 上 PSNR 提升 0.22 dB（marginal），但 representation capability 上必要
- **KNN smooth** 是辅助 regularizer，影响小但 positive

---

## 整体直觉总结

### "Geometric prior 做 feedforward + gradient 做 refinement" 范式

这篇 paper 体现了一个更大的 trend：**当 explicit geometric priors（optical flow, depth, image Jacobian）可得时，用它们做 feedforward 信号来 bootstrap learning，比从头用 implicit function approximator (MLP) 学要快得多**。

这是 SLAM 圈一直在反思的事情：
- 纯 learning-based 方法（NeRF-SLAM [[14] iMAP](https://academic.oup.com/ijcv/article/130/2/406/6756118), [15] NICE-SLAM](https://github.com/cvglabs/NICE-SLAM), MLP deformation）flexible 但 training 慢
- 传统 geometric 方法（optical flow + IRLS, ICP）快但 limited expressiveness
- Hybrid sweet spot: geometric prior 做初始化 + gradient 做 refinement

### 三层 optical flow 利用是关键 elegant 设计

| 层级 | 用途 | 机制 |
|---|---|---|
| Mask generation | 区分 static / dynamic | Residual after fitting $\mathbf{J}\boldsymbol{\xi}$ |
| Camera init | 提供 coarse pose | $\boldsymbol{\xi}^*$ from IRLS on inlier pixels |
| Gaussian propagation | 初始化新 keyframe 上的 dynamic centers | $\mathbf{F}$ + depth → unproject |

一个 prior signal 服务三个环节，避免了三个独立 module 的开销。这是 paper 最 elegant 的地方。

### 与 SC-GS / 4DGS-SLAM 的根本区别

| 维度 | SC-GS | 4DGS-SLAM | Flow4DGS-SLAM |
|---|---|---|---|
| Deformation model | MLP on sparse control points | 同 SC-GS | Hybrid explicit + GMM |
| Dynamic segmentation | N/A (offline) | Category-based | Category-agnostic (residual) + Semantic |
| New object handling | N/A | Handcrafted start time | Adaptive insertion via backward flow |
| Training speed | Hours | 0.04 FPS | 0.50 FPS |

**核心转变**: 把"用 network 学习 temporal deformation field"分解为"用 optical flow 做 propagation（feedforward，无需训练）+ 少量 gradient refinement"。本质是 inductive bias 的转换——SLAM 中 keyframes sparse，MLP 容易 overfit 且训练慢，explicit + flow propagation 更适合。

---

## 我的 take（一句话版本）

这篇 paper 的真正贡献在于提出了一个 **template**：

> **"Geometric prior 做 feedforward 初始化，gradient 做 refinement"** 是 SLAM + 3DGS 圈的甜蜜点。

未来 SLAM + 4DGS 系工作大概会沿这条路继续探索：
- optical flow 换成更 general 的 foundation model（如 SAM 2 [[16] SAM 2](https://github.com/facebookresearch/sam2) 提供的 mask + flow + feature）
- GMM 换成更 expressive 的 temporal basis（Fourier / Chebyshev polynomials）
- linear interpolation between keyframes 换成 spline / Bézier 处理高阶 motion
- 加 instance segmentation 做 per-object rigidity
- 加 loop closure（dynamic map 直接 align 是难题）

希望这个 version 更 build your intuition, Andrej！如果哪个 trick 还想再深挖（比如 image Jacobian 的推导、GMM K=3 的选择 motivation、KNN $\tau_{knn}$ 的选择），可以再展开。

---

# Flow4DGS-SLAM: Optical Flow-Guided 4D Gaussian Splatting SLAM 深度解析

Andrej, 这篇 paper 来自 NUS Gim Hee Lee 组（Yunsong Wang），核心 contribution 是把 4DGS-SLAM 从"基于 SC-GS deformation MLP 的 offline-style 在线化框架"推进到"optical flow-guided hybrid 表征"，在 tracking accuracy、rendering quality、training efficiency 三个维度都取得 SOTA。下面我按照 motivation → core idea → 每个 module 的技术细节 → 实验数据 → intuition 的顺序展开。

---

## 1. 研究背景与核心 motivation

### 1.1 Dynamic 3DGS-SLAM 的痛点

3DGS-SLAM 系（MonoGS [[1]](https://github.com/muskie82/MonoGS), SplaTAM [[2]](https://github.com/hermosayhlabs/splatam), GS-SLAM [[3]](https://github.com/erikwijmans/gs-slam)）大多把 dynamic objects 当 outliers 滤除，最终 map 只剩 static background。但 dynamic reconstruction 领域已有大量工作用 MLP-based deformation field（[[4] Deformable 3D Gaussians](https://github.com/ingra14t/Deformable-3D-Gaussians), [[5] SC-GS](https://github.com/yihua7/SC-GS), [[6] 4D-Gaussian-Splatting](https://github.com/ingra14t/4DGaussians)）或者直接 parameterize temporal offsets（[[7] Spacetime Gaussians](https://github.com/Andy-Qi/STG), [[8] 4D-rotor GS](https://github.com/Nerf0/4drotor)），但都需要 precomputed multi-view poses + 数小时 offline training。

4DGS-SLAM [[9]](https://arxiv.org/abs/2506.07492)（Li et al. 2025）是首个把 dynamic 3DGS 与 SLAM 联合起来的工作，但有三个核心痛点：

| 痛点 | 原因 | 后果 |
|---|---|---|
| Deformation MLP training expensive | SC-GS 风格的 sparse control points + MLP deformation field 每 mapping step 训 100 + 100 iterations | FPS 仅 0.04 |
| Category-based segmentation | 用 segmentation model 区分 dynamic class | 无法处理 category-agnostic motion（如 balloon、ball） |
| 复杂 dynamics 处理弱 | 需 handcrafted dynamic start time | 无法处理 people leaving & re-entering view |

### 1.2 Flow4DGS-SLAM 的核心 insight

**核心 insight：用 optical flow 作为统一的几何 supervision signal，串起 mask generation → camera initialization → Gaussian propagation → adaptive insertion 四个环节。** 这避免了 SC-GS 风格的 implicit deformation MLP，把 dynamic Gaussian 的 position 表征做 explicit，再用 GMM 表征 smooth 的 opacity / rotation。

---

## 2. 整体架构解析

参考 Figure 2，pipeline 由四个核心 module 构成：

```
RGB-D stream ──┬──► RAFT (optical flow) ──┐
                │                           ├──► Camera-Induced Motion Decomposition
                ├──► YOLOv9 (semantic) ─────┘            │
                │                                       │
                │                                       ▼
                │                              [M_dy, ξ* camera init]
                │                                       │
                ▼                                       ▼
        ┌──────────────────┐              ┌──────────────────────────┐
        │ Static 3DGS      │◄── tracking ─│  Hybrid 4DGS             │
        │ (background map)  │              │  - explicit keyframe xyz │
        └──────────────────┘              │  - GMM opacity / rot     │
                                          │                          │
                                          │ Scene Flow Propagation   │
                                          │ Adaptive Insertion       │
                                          └──────────────────────────┘
```

**Intuition**: 把 "dynamic" 这件事拆解为两个 sub-problem：
1. **几何位置随时间演化** → explicit keyframe positions + optical flow propagation（fast & controllable）
2. **appearance attributes（opacity, rotation）随时间演化** → GMM（smooth, compact, parametric）

这种 hybrid 设计是 paper 的灵魂。

---

## 3. Camera-Induced Motion Decomposition（Section 3.1）

### 3.1 数学模型：image Jacobian 与 ego-motion

核心 idea：对 static point，其 optical flow 完全由 camera 6-DoF motion + depth 决定。基于 small-motion 假设下的 projective geometry：

**Eq. (1)**:
$$\mathbf{F}(u,v) = \mathbf{J}(\mathbf{x})\, \boldsymbol{\xi}$$

- $\mathbf{F}(u,v) \in \mathbb{R}^2$: pixel $(u,v)$ 处的 optical flow（来自 RAFT [[10] RAFT](https://github.com/princeton-vl/RAFT)）
- $\mathbf{x} = (u, v, Z)^\top \in \mathbb{R}^3$: pixel 对应的 3D point（$Z$ 来自 depth map $\mathbf{D}^t$）
- $\boldsymbol{\xi} = [\boldsymbol{\rho}^\top, \boldsymbol{\theta}^\top]^\top \in \mathbb{R}^6$: camera twist，前三维 $\boldsymbol{\rho}$ 是 translation（在 $\mathfrak{se}(3)$ 的 Lie algebra 表示），后三维 $\boldsymbol{\theta}$ 是 rotation
- $\mathbf{J}(\mathbf{x}) \in \mathbb{R}^{2\times 6}$: image Jacobian / interaction matrix（visual servoing 经典公式，[[11] Chaumette et al. Springer Handbook](https://link.springer.com/referencework/10.1007/978-3-319-32552-1)）

**Eq. (2)**: image Jacobian 的显式形式
$$\mathbf{J}(\mathbf{x}) = \begin{bmatrix} -\frac{f_x}{Z} & 0 & \frac{u}{Z} & \frac{uv}{f_y} & -f_x - \frac{u^2}{f_x} & v \\ 0 & -\frac{f_y}{Z} & \frac{v}{Z} & f_y + \frac{v^2}{f_y} & -\frac{uv}{f_x} & -u \end{bmatrix}$$

- 前 3 列对应 translation $\boldsymbol{\rho}$：每个元素含 $1/Z$ 因子，**translation-induced flow scales with $1/Z$**
- 后 3 列对应 rotation $\boldsymbol{\theta}$：**rotation-induced flow is depth-independent**
- $(f_x, f_y)$: camera intrinsics (focal length)

这是一个非常深刻的几何 intuition：camera translation 在 image 上的 effect 受 depth 调制（近处物体 apparent motion 大），而 rotation effect 与 depth 无关（相当于全图统一 rotation）。这正是 why epipolar geometry / Essential matrix 能够解出相对 pose 的本质。

### 3.2 Robust 求解：IRLS with Cauchy weights

**Eq. (3)**:
$$\hat{\boldsymbol{\xi}} = \arg\min_{\boldsymbol{\xi}} \sum_i w_i \|\mathbf{F}_i - \mathbf{J}_i \boldsymbol{\xi}\|^2$$

- $w_i$: Cauchy weights（IRLS, Iteratively Reweighted Least Squares），用于 robustly reject outliers（dynamic pixels）
- 只在 $\mathcal{M}_s(u,v) = 0$（YOLOv9 [[12] YOLOv9](https://github.com/WongKinYiu/yolov9) 认为非 dynamic class）且 $Z > 0$ 的 pixels 上 stack

IRLS with Cauchy 的 intuition：先 OLS 求解 → 计算每个 pixel residual → 给 residual 大的 pixel 小权重（Cauchy: $w_i = \frac{1}{1 + (r_i/c)^2}$）→ 重新求解 → 迭代。最终 dynamic pixels 自然被降权。

### 3.3 Category-agnostic mask via residual thresholding

**Eq. (4)**: residual
$$r(u,v) = \|\mathbf{F}(u,v) - \hat{\mathbf{F}}(u,v)\|_2$$

- $\hat{\mathbf{F}}(u,v) = \mathbf{J}(u,v,Z)\hat{\boldsymbol{\xi}}$: predicted rigid flow（camera-induced）
- 如果 pixel 是 static，则 $r$ 应该接近 0；如果是 dynamic（含 object 自身 motion），$r$ 显著大于 0

**Eq. (5)**: 自适应阈值
$$\mathcal{M}_{ca}(u,v) = \mathbb{1}\left(r(u,v) > \text{median}(r) + k\,\text{MAD}(r)\right)$$
$$\text{MAD}(r) = \text{median}_i |r_i - \text{median}_j(r_j)|$$

- $\text{median}(r)$: residuals 中位数（robust 中心估计，比 mean 更 robust）
- $\text{MAD}$: Median Absolute Deviation，robust 的 spread 度量
- $k$: 灵敏度系数（推测 1.5~3 之间，类似 Z-score 阈值）
- 这是统计学经典的 robust outlier detection（[[13] Leys et al. MAD-based outlier detection](https://www.sciencedirect.com/science/article/pii/S0022103113000668)）

最终 dynamic mask：
$$\mathcal{M}_{dy} = \mathcal{M}_s \cup \mathcal{M}_{ca}$$

semantic mask (YOLOv9) 拦截"已知的 dynamic class"（如 people），residual thresholding 拦截"几何上确实在动的任何 class"（如 balloon、ball）—— 互补覆盖。

### 3.4 Camera pose initialization

拿到 clean 的 mask 后，在 $\mathcal{M}_{dy}=0$ 的 pixels 上做第二次 weighted least-squares，得到 refined twist $\hat{\boldsymbol{\xi}}^*$，然后：

**Eq. (6)**: 把 twist 映射到 SE(3)
$$\exp_{\mathfrak{se}(3)}(\hat{\boldsymbol{\xi}}^*) = \begin{bmatrix} \exp_{\mathfrak{so}(3)}(\boldsymbol{\theta}^*) & \mathbf{V}(\boldsymbol{\theta}^*)\boldsymbol{\rho}^* \\ \mathbf{0}^\top & 1 \end{bmatrix}$$

- $\exp_{\mathfrak{so}(3)}(\boldsymbol{\theta}^*)$: Rodrigues formula，rotation part 的 exponential map
- $\mathbf{V}(\boldsymbol{\theta}^*)$: SO(3) 的 left Jacobian（[[14] Gallego & Yezzi - A Compact Formula for the Derivative of SE(3)](https://arxiv.org/abs/1902.07220)），解决 translation 与 rotation 的耦合问题
- 这是 Lie group / Lie algebra 上的标准 exponential map

**Eq. (7)**: 累积到上一帧 pose 上
$$\mathbf{T}_{cw}^t = \mathbf{T}_{cw}^{t-1} \exp_{\mathfrak{se}(3)}(\hat{\boldsymbol{\xi}}^*)$$

- $\mathbf{T}_{cw}^{t-1}$: previous camera-to-world pose（注意 paper 中是 $cw$ 即 camera-from-world，符号上 $\mathbf{T}_{cw}$ 把 world point 变到 camera frame）

**关键 robustness trick**: paper 提到 clamp maximum camera motion，threshold ∝ inlier pixels ratio。即如果 inlier 比例低（说明 optical flow noise 大），就 clamp 住 motion 幅度，防止初始化跑飞。

### 3.5 Tracking loss（Section 3.3）

**Eq. (11)**: valid mask
$$\mathcal{M}_v = (\neg \mathcal{M}_{dy}) \cap \mathcal{M}_o$$
其中 $\mathcal{M}_o(\mathbf{u}) = \mathbb{1}(\hat{\mathbf{O}}(\mathbf{u}) \geq \alpha)$ 是 opacity mask，过滤掉未观察到的 regions。

**Eq. (12)**: tracking loss
$$\mathcal{L}_{track} = \frac{1}{|\mathcal{V}|}\sum_{\mathbf{u}\in\mathcal{V}} \mathcal{M}_v(\mathbf{u})\left(\lambda_1 L_1(\hat{\mathbf{C}}(\mathbf{u})) + \lambda_2 L_1(\hat{\mathbf{D}}(\mathbf{u}))\right)$$

- $\mathcal{V}$: valid-depth pixels
- $\lambda_1, \lambda_2$: color vs depth 权重
- 这里只在 static regions 上做 photometric + geometric loss，dynamic regions 被 mask 掉

构成 coarse-to-fine pipeline: optical flow → camera init → differentiable rendering refinement。

---

## 4. Hybrid 4DGS Representation（Section 3.2）

### 4.1 与 3DGS 的关系（Eq. 8 - Preliminary）

标准 3DGS [[15] Kerbl et al. 3D Gaussian Splatting](https://repo.acin.tuwien.ac.at/tmp/3dgs/) 的 α-blending:
$$\hat{\mathbf{C}}^s(\mathbf{u}) = \sum_{i=1}^{|\mathcal{G}|} \mathbf{c}_i^s \alpha_i^s(\mathbf{u}) \prod_{j<i}(1-\alpha_j^s(\mathbf{u}))$$

- $\mathbf{c}_i^s$: per-Gaussian color（球谐或 RGB）
- $\alpha_i^s(\mathbf{u}) = \sigma_i^s \cdot \exp\left(-\frac{1}{2}(\mathbf{u}-\boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1}(\mathbf{u}-\boldsymbol{\mu}_i)\right)$: Gaussian footprint × opacity
- $\prod_{j<i}(1-\alpha_j)$: front-to-back 累积的 transmittance

### 4.2 Hybrid 设计的本质

第 $i$ 个 dynamic Gaussian 的 attributes:
- **Static attributes**: $\{\mathbf{s}_i, \sigma_i, \mathbf{c}_i\}$（scale, base opacity, color）—— 与时间无关
- **Dynamic attributes**:
  - (i) **explicit keyframe positions** $\{\mathbf{x}_i^k\}$ at keyframes $\{t_k\}$
  - (ii) **GMM-based temporal opacity** $\sigma_i(t)$ **and rotation** $\mathbf{q}_i(t)$

中间时刻通过 linear interpolation 拿 position，GMM 连续拿到 opacity / rotation。这种 split 设计的核心 intuition 是：

> **Position 是 dynamic Gaussian 中变化最剧烈、最 critical 的属性**（直接决定 rendering correctness），用 explicit 表示允许 optical flow 直接 propagate，避开 MLP 的 implicit bottleneck。
> 
> **Opacity 和 rotation 相对 smooth**，用 GMM 这种 parametric form 既 compact 又 smooth，避免 per-timestamp storage 爆炸。

### 4.3 GMM-based opacity (Eq. 9)

$$m_i(t) = 1 - \exp\left(-A_i \sum_{k=1}^{K} w_{i,k}\, \mathcal{N}(\hat{t}; \mu_{i,k}, \tau_{i,k}^2)\right)$$

- $\hat{t} \in [0, 1]$: normalized time
- $K=3$: mixture components 数量
- $A_i > 0$: learnable activation amplitude（控制 opacity 峰值上限）
- $w_{i,k}, \mu_{i,k}, \tau_{i,k}$: per-component weight / mean / std（learnable）
- $\mathcal{N}(\hat{t}; \mu, \tau^2) = \frac{1}{\sqrt{2\pi}\tau}\exp(-\frac{(\hat{t}-\mu)^2}{2\tau^2})$: Gaussian basis

最终 opacity: $\sigma_i(t) = \sigma_i \cdot m_i(t)$.

**Intuition**: 这相当于用 $K$ 个 Gaussian bump 函数的加权和来拟合 opacity 的时间曲线。$1 - \exp(-\cdot)$ 的外层 wrap 是 Poisson-style blending（类似 volume rendering 的 transmittance），保证 $m_i(t) \in [0, 1)$。这种设计天然能表达"出现-消失"（一个 Gaussian bump 在特定时间激活）、"持续存在"（多个 overlapping bumps）、"间歇出现"（多个 separated bumps）。

### 4.4 GMM-based rotation (Eq. 10)

$$\mathbf{q}_i(t) = \frac{\sum_{k=1}^{K} w_{i,k}\, \mathcal{N}(\hat{t}; \mu_{i,k}, \tau_{i,k}^2)\, \mathbf{q}_{i,k}}{\left\|\sum_{k=1}^{K} w_{i,k}\, \mathcal{N}(\hat{t}; \mu_{i,k}, \tau_{i,k}^2)\, \mathbf{q}_{i,k}\right\|}$$

- $\mathbf{q}_{i,k} \in \mathbb{R}^4$: per-component control quaternion
- 用与 opacity 相同的 Gaussian activation 作为 blending weights
- 分母归一化确保 $\|\mathbf{q}_i(t)\| = 1$（单位 quaternion）

**关键设计**: opacity 和 rotation 共享同一组 $\{w, \mu, \tau\}$，节省参数量；只在 control quaternion 上独立。这种"shared temporal kernel, separate spatial attribute"思路在 NeRF/Deformation 圈里也有类似做法（如 [[16] T-to-3DGS](https://github.com/luost26/to4dgs) 的 polynomial temporal basis）。

### 4.5 Explicit keyframe positions

- 在 keyframes $\{t_k\}$ 上学习 $\mathbf{x}_i^k$，中间用 linear interpolation
- Keyframe selection: 基于 motion mask 差异 + 每 5 帧至少 1 keyframe（follow 4DGS-SLAM）
- Linear interp 的优势: ① 显式可控 ② 与 optical flow propagation 兼容 ③ backward gradient 容易

---

## 5. Optical Flow-Guided 4D Mapping（Section 3.4）

这是 paper 中最 magic 的部分，把 optical flow 作为 scene flow prior 用。

### 5.1 Scene Flow Gaussian Propagation (Eq. 13-15)

给定 keyframe $k-1$ 的 dynamic Gaussians 集合 $\mathcal{G}^{t_{k-1}}$ 和 3D centers $\{\mathbf{x}_i^{k-1}\}$，要初始化 keyframe $k$ 的 centers。

**Step 1**: 投影到 keyframe $k-1$ 的 image
$$\bar{\mathbf{u}}_i^{k-1} = \mathbf{P}_{k-1}[\mathbf{x}_i^{k-1}], \quad \mathbf{u}_i^{k-1} = \Pi(\bar{\mathbf{u}}_i^{k-1})$$

- $\mathbf{P}_{k-1} = \mathbf{K}[\mathbf{R}_{k-1} | \mathbf{t}_{k-1}]$: keyframe $k-1$ 的 projection matrix
- $\Pi(\cdot)$: homogeneous normalization（除以最后一维）

**Step 2**: 用 optical flow propagate 到 keyframe $k$
$$\mathbf{u}_i^k = \mathbf{u}_i^{k-1} + \mathbf{F}^{t_{k-1}, t_k}(\mathbf{u}_i^{k-1})$$

- $\mathbf{F}^{t_{k-1}, t_k}$: RAFT 预测的 forward optical flow

**Step 3**: Unprojection 得到 keyframe $k$ 的 3D position

**Eq. (13)**:
$$\Delta \mathbf{x}_i^k = \mathbf{R}_k^\top \left(D_i^k \mathbf{K}^{-1} \bar{\mathbf{u}}_i^k - \mathbf{t}_k\right) - \mathbf{x}_i^{k-1}$$

- $\bar{\mathbf{u}}_i^k = [\mathbf{u}_i^k, 1]^\top$: homogeneous pixel
- $D_i^k$: 该 pixel 在 keyframe $k$ 的 depth
- $\mathbf{K}^{-1}\bar{\mathbf{u}}_i^k$: camera-space 的 normalized direction（在 z=1 plane）
- $D_i^k \mathbf{K}^{-1}\bar{\mathbf{u}}_i^k$: camera-space 3D point
- $\mathbf{R}_k^\top(\cdot - \mathbf{t}_k)$: world-space 3D point（反向 transform）
- 整个公式本质上是 $\text{unproject}_k(\mathbf{u}_i^k, D_i^k) - \mathbf{x}_i^{k-1}$，即"在新 keyframe 上 unproject 后的位移"

**Step 4**: KNN smoothing

**Eq. (14)**:
$$\Delta \hat{\mathbf{x}}_i^k = \sum_{j\in \mathcal{N}(i)} w_{ij}^{knn}\, \Delta \mathbf{x}_j^k$$

**Eq. (15)**:
$$w_{ij}^{knn} = \frac{\mathcal{N}\left(\|\mathbf{x}_j^{k-1} - \mathbf{x}_i^{k-1}\|_2; 0, \tau_{knn}^2\right)}{\sum_{l\in\mathcal{N}(i)} \mathcal{N}\left(\|\mathbf{x}_l^{k-1} - \mathbf{x}_i^{k-1}\|_2; 0, \tau_{knn}^2\right)}$$

- $\mathcal{N}(i)$: i-th Gaussian 的 nearest neighbors（半径 search）
- $\tau_{knn}$: smoothing bandwidth

**Intuition**: optical flow 有 noise（尤其 depth 不准时 unprojection 误差大），用 KNN 在 Gaussian 空间做 Gaussian-weighted averaging，强制 nearby Gaussians 有 similar motion（local rigidity prior）。这和 [[17] Dynamic 3D Gaussians](https://github.com/JonathonLuiten/Dynamic3D Gaussians) 的 as-rigid-as-possible regularization 是同一族思想，但这里做在 propagation stage（feedforward）而非 loss stage。

最终: $\mathbf{x}_i^k = \mathbf{x}_i^{k-1} + \Delta \hat{\mathbf{x}}_i^k$.

### 5.2 Adaptive Gaussian Insertion (Eq. 16)

**Problem**: 如果 keyframe $k$ 出现了 keyframe $k-1$ 没有的 dynamic region（新出现的物体，或 previously occluded 部分），propagation 无法处理。需要 insert 新 Gaussians。

**Solution**: 用 backward optical flow back-track 当前 dynamic mask 到前一帧，找"前帧不是 dynamic 但当前帧是 dynamic"的 pixels。

$$\mathcal{M}_{insert}^{t_k} = \left\{\mathbf{u}_p^k \in \mathcal{M}_{dy}^{t_k} \,\middle|\, \mathbf{u}_p^{k-1} \notin \mathcal{M}_{dy}^{t_{k-1}}\right\}$$

- $\mathbf{u}_p^{k-1} = \mathbf{u}_p^k + \mathbf{F}^{t_k, t_{k-1}}(\mathbf{u}_p^k)$: backward flow warp

然后以密度 $1/D_{init}$ 随机采样这些 pixels 并 unproject 初始化新 dynamic Gaussians。

**Intuition**: 这相当于"motion-aware"的 Gaussian densification，避免 4DGS-SLAM 那种需要 handcrafted dynamic start time 的尴尬。比如一个人 leave 然后 re-enter view，新的 frame 上对应 region 自然满足 $\mathbf{u}_p^{k-1} \notin \mathcal{M}_{dy}^{t_{k-1}}$（前帧没有这个 dynamic pixel），就自动 insert。

### 5.3 Mapping Loss (Eq. 17)

$$\mathcal{L}_{map} = \lambda_1 \mathcal{L}_c + \lambda_2 \mathcal{L}_d + \lambda_f \mathcal{L}_f + \lambda_m \mathcal{L}_m + \lambda_{iso} \mathcal{L}_{iso}$$

- $\mathcal{L}_c, \mathcal{L}_d$: standard color + depth L1 loss
- $\mathcal{L}_f$: flow loss（rendered flow vs RAFT flow），只在最后 25 iterations 用
- $\mathcal{L}_m$: binary mask loss，约束 rendered alpha map 与 motion mask 一致
- $\mathcal{L}_{iso}$: isotropic loss（继承自 4DGS-SLAM），防止 Gaussian 退化成极端 anisotropic

每个 mapping step 训 50 iterations（vs 4DGS-SLAM 的 200 iterations），主要节省来自：
- 不需要训练 deformation MLP
- Optical flow propagation 给了好的初始化，少 iterations 就收敛

---

## 6. 实验数据深度解析

### 6.1 Tracking Accuracy (Table 1, TUM RGB-D)

| Method | fr3/sit_st | fr3/walk_st | fr3/walk_xyz | fr3/walk_rpy | **Avg.** |
|---|---|---|---|---|---|
| MonoGS [[1]](https://github.com/muskie82/MonoGS) | 0.48 | 21.9 | 30.7 | 34.2 | 15.8 |
| SplaTAM [[2]](https://github.com/hermosayhlabs/splatam) | 0.52 | 83.2 | 134.2 | 142.3 | 62.2 |
| 4DGS-SLAM [[9]](https://arxiv.org/abs/2506.07492) | 0.58 | 0.61 | 2.7 | 3.0 | 2.1 |
| **Ours** | 0.70 | 0.48 | 2.5 | 3.6 | **1.9** |

**关键观察**:
- MonoGS / SplaTAM 在 walking sequences 上完全崩溃（130+ cm 误差），因为它们 filter 掉 dynamic 但 tracking drift 严重
- 4DGS-SLAM 已经很好（2.1 cm），Ours 在 dynamic-heavy sequences（walk_st, walk_xyz）上再降一点
- 在 sit_st（near-static）上 Ours 略差（0.70 vs 0.58），因为 dynamic 少，optical flow 的 noise 反而影响初始化
- 但 overall average 1.9 vs 2.1，**用更少 mapping iterations 取得更好 tracking**

### 6.2 Rendering Quality (Table 2)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| MonoGS | 17.74 | 0.608 | 0.382 |
| SplaTAM | 19.40 | 0.757 | 0.241 |
| SC-GS [[5]](https://github.com/yihua7/SC-GS) | 20.78 | 0.657 | 0.396 |
| 4DGS-SLAM | 22.55 | 0.788 | 0.229 |
| **Ours** | **26.55** | **0.831** | **0.177** |

**+4 dB PSNR 提升 over 4DGS-SLAM**！这是相当大的跃迁。来源主要是 hybrid representation + scene flow propagation 让 dynamic regions 重建质量大幅提升。

### 6.3 BONN 数据集 (Table 3, 4) - 更难

BONN 包含 balloon（category-agnostic dynamic object）、person_tracking、sync（fast camera motion）等场景。

| Method | PSNR (balloon2) | PSNR (sync) | Avg PSNR |
|---|---|---|---|
| MonoGS | 20.22 | 22.03 | 21.06 |
| 4DGS-SLAM | 22.91 | 24.37 | 23.81 |
| **Ours** | **28.36** | **29.85** | **29.71** |

**+6 dB PSNR 提升**，比 TUM 上还猛。原因：BONN 有 category-agnostic dynamic objects (balloon)，4DGS-SLAM 的 category-based segmentation 漏掉，而 Ours 的 residual-based M_ca 捕捉到。

### 6.4 Runtime (Table 5)

| Method | Dynamic Seg. | Tracking | Mapping | FPS |
|---|---|---|---|---|
| MonoGS | - | 476 ms | 557 ms | 1.93 |
| 4DGS-SLAM | 16 ms | 445 ms | **110562 ms** | 0.04 |
| Ours | 68 ms | 427 ms | **6285 ms** | 0.50 |

**核心 gain 在 mapping**: 6285 ms vs 110562 ms（**17.6x speedup**）。Dynamic Seg. 多花 52 ms 是 RAFT + YOLOv9 的代价，但 mapping 的省时碾压。

**Intuition**: 4DGS-SLAM 每个 mapping step 训 100 iterations deformation MLP + 100 iterations joint optimization；Ours 训 50 iterations，且 deformation MLP 被 optical flow propagation 替代，省时主要来自 MLP forward/backward 的开销消失。

### 6.5 Ablation Study (Table 6)

| Config | walk_xyz ATE / PSNR | balloon2 ATE / PSNR |
|---|---|---|
| w/o Motion Decomp. | 2.7 / 24.40 | 7.4 / 27.59 |
| w/o Flow Propagate | 2.6 / 24.04 | 3.9 / 27.93 |
| w/o Adaptive Insert | 3.4 / 24.47 | 3.7 / 27.91 |
| w/o GMM | 2.7 / 23.91 | 3.5 / 28.14 |
| w/o KNN smooth | 2.5 / 24.47 | 3.5 / 27.93 |
| **Ours** | **2.5 / 24.60** | **3.4 / 28.36** |

**关键 takeaways**:
- **Motion Decomp** 对 balloon2 影响最大（ATE 7.4 → 3.4 cm），因为 balloon 是 category-agnostic，没有 M_ca 就完全 miss
- **Flow Propagate** 主要影响 PSNR，对 tracking 影响小
- **Adaptive Insert** 对复杂 dynamic 场景关键（处理 re-entering objects）
- **GMM** 在 balloon2 上 PSNR 提升 0.22 dB（marginal），但 representation capability 上是必要的
- **KNN smooth** 是辅助 regularizer，影响小但 positive

---

## 7. 核心设计 intuition 总结

### 7.1 三层 optical flow 利用

| 层级 | 用途 | 机制 |
|---|---|---|
| Mask generation | 区分 static / dynamic | Residual after fitting J(x)ξ |
| Camera init | 提供 coarse pose | ξ* from IRLS on inlier pixels |
| Gaussian propagation | 初始化新 keyframe 上的 dynamic centers | F + depth → unproject |

**这是 paper 最 elegant 的设计**: optical flow 一个 prior signal 服务了 mask + camera + mapping 三个环节，避免了分别用三个 module 的开销。

### 7.2 Hybrid representation 的 trade-off

| Attribute | Representation | 为什么 |
|---|---|---|
| Position | Explicit per-keyframe + linear interp | 需要被 flow propagate、被 adaptive insert，必须 explicit |
| Opacity | GMM(K=3) | 时间 smooth，但要表达 "出现-消失"，需要 parametric 形式 |
| Rotation | GMM(K=3) | 与 opacity 共享 temporal kernel，compact |
| Scale, color | Static | 时间不变，不需要 temporal model |

如果 position 也用 GMM，会失去 optical flow propagation 的能力；如果 opacity 也用 explicit per-keyframe，会爆炸 storage 且不平滑。Hybrid 是 sweet spot。

### 7.3 与 SC-GS / 4DGS-SLAM 的根本区别

| 维度 | SC-GS | 4DGS-SLAM | Flow4DGS-SLAM |
|---|---|---|---|
| Deformation model | MLP on sparse control points | 同 SC-GS | Hybrid explicit + GMM |
| Dynamic segmentation | N/A (offline) | Category-based | Category-agnostic (residual) + Semantic |
| New object handling | N/A | Handcrafted start time | Adaptive insertion via backward flow |
| Training speed | Hours | 0.04 FPS | 0.50 FPS |

**Insuition**: SC-GS 风格的 deformation MLP 本质是"用一个 network 拟合整个 temporal deformation field"，但 SLAM 中 keyframes 是 sparse 的（temporal 上稀疏），MLP 容易 overfit 且训练慢。改成 explicit + flow propagation 是把"学习 temporal deformation"分解为"用 optical flow 做 propagation（feedforward，无需训练）+ 少量 gradient refinement"，本质是 inductive bias 的转换。

### 7.4 与 GFlow [[18] Wang et al. GFlow](https://github.com/sail-sg/GFlow) 的关系

Paper Section 3.4 提到 "take inspiration from GFlow"。GFlow 是 AAAI 2025 的工作，从 monocular video 恢复 4D world，核心也是用 optical flow 作为 scene flow prior。Flow4DGS-SLAM 把这个 idea 移植到 SLAM 中，但增加：
- Keyframe-based（vs GFlow 的 dense frame）
- KNN smoothing for local rigidity（vs GFlow 的全局 optimization）
- Adaptive insertion（GFlow 不需要因为它是 offline）

---

## 8. Potential limitations & future directions

### 8.1 可能的 limitations

1. **Optical flow quality依赖**: RAFT 在 fast motion、occlusion 边界处会失败，propagation 噪声大。KNN smoothing 缓解但不解决。
2. **Depth sensing 依赖**: Eq.(13) 的 unprojection 需要 $D_i^k$，depth 传感器的 noise（尤其 flying pixels、far range）直接 propagate 到 3D 误差。
3. **0.5 FPS 仍非 real-time**: 对 SLAM 来说还需要 ~10x speedup。Bottleneck 在 mapping（6285 ms）。可能改进：减少 K、用更 compact 的 GMM、async mapping。
4. **Linear interpolation between keyframes**: 复杂 motion（如 bouncing ball 的高阶 motion）linear interp 不够。可能要 spline / Bézier。
5. **Semantic mask 仍依赖 YOLOv9**: 完全 category-agnostic 还没做到，semantic prior 仍作为补充。

### 8.2 可能的 future extensions

- **Loop closure**: paper 没提，但 dynamic SLAM 中 loop closure 是大难题（dynamic map 无法直接 align）。
- **Multi-object tracking**: paper 把所有 dynamic Gaussians 当一个集合，没有 instance segmentation。如果加 instance label，可以更好地 impose per-object rigidity。
- **Physics priors**: GMM rotation 目前是 free learned，如果加 inertia / gravity constraint 可能更 physical-consistent。
- **Feed-forward 4DGS**: 结合 [[19] FreeSplat](https://github.com/NUS-HPC-AI-Lab/FreeSplat) 的 generalizable 思路，做 cross-scene 的 4DGS SLAM。
- **Diffusion-based motion prior**: 把 GMM 替换为 denoising diffusion model，可能表达更复杂的 temporal patterns。

---

## 9. 实现细节（推测，结合 4DGS-SLAM 与 paper 描述）

### 9.1 关键超参数（推测）

- $K=3$ (GMM components)
- Mapping: 50 iterations, window size 8
- Tracking: 100 iterations (TUM), 200 iterations (BONN)
- Color refinement after online training: 1500 iterations
- Keyframe selection: motion mask difference + 1 per 5 frames minimum
- Flow loss: only last 25 iterations
- $D_{init}$: Gaussian insertion density factor

### 9.2 训练 schedule 推测

```
For each new RGB-D frame I^t, D^t:
  1. Compute optical flow F^{t,t-1} via RAFT
  2. Compute semantic mask M_s via YOLOv9
  3. Camera-Induced Motion Decomposition:
     a. Fit ξ via IRLS on M_s=0 pixels
     b. Compute residuals, threshold → M_ca
     c. M_dy = M_s ∪ M_ca
     d. Refit ξ* on M_dy=0 pixels
     e. Initialize T_{cw}^t = T_{cw}^{t-1} exp_se(3)(ξ*)
  4. Tracking: gradient descent on L_track (100-200 iters)
  5. If new keyframe:
     a. Scene Flow Propagation: x_i^k = x_i^{k-1} + Δx̂_i^k
     b. Adaptive Insertion: unproject new dynamic pixels
     c. Mapping: train 50 iters on sliding window (8 keyframes) + 2 random previous
  6. Periodic color refinement (1500 iters) after online training
```

---

## 10. Web links for reference

**核心方法 related**:
- 3DGS 原文 & code: https://repo.acin.tuwien.ac.at/tmp/3dgs/ | https://github.com/graphdeco-inria/gaussian-splatting
- 4DGS-SLAM: https://github.com/yanyanli/4D-Gaussian-Splatting-SLAM (推测, paper 中 [21])
- SC-GS: https://github.com/yihua7/SC-GS
- GFlow (AAAI 2025): https://github.com/sail-sg/GFlow
- RAFT (optical flow): https://github.com/princeton-vl/RAFT
- YOLOv9: https://github.com/WongKinYiu/yolov9
- Deformable 3D Gaussians: https://github.com/ingra14t/Deformable-3D-Gaussians
- Spacetime Gaussians: https://github.com/Andy-Qi/STG
- 4D-GS (Yang et al.): https://github.com/ingra14t/4DGaussians

**SLAM baselines**:
- MonoGS (Gaussian Splatting SLAM, CVPR 2024): https://rmurai.io/projects/Gaussian-Splatting-SLAM/
- SplaTAM (CVPR 2024): https://github.com/hermosayhlabs/splatam
- GS-SLAM: https://github.com/erikwijmans/gs-slam
- DROID-SLAM: https://github.com/princetonvl/DROID-SLAM
- ORB-SLAM3: https://github.com/UZ-SLAMLab/ORB_SLAM3

**Datasets**:
- TUM RGB-D: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- BONN: https://www.ipb.uni-bonn.de/data/datasets/

**Math / Lie group reference**:
- A compact formula for SE(3) derivative (Gallego & Yezzi): https://arxiv.org/abs/1902.07220
- Visual servoing (Chaumette, Hutchinson, Corke): https://link.springer.com/referencework/10.1007/978-3-319-32552-1
- MAD outlier detection (Leys et al.): https://www.sciencedirect.com/science/article/pii/S0022103113000668

**作者主页**:
- Gim Hee Lee (NUS): https://www.comp.nus.edu.sg/~leegh/
- Yunsong Wang (project page): https://wangys16.github.io/Flow4DGS-SLAM

**Similar / concurrent works**:
- WildGS-SLAM (CVPR 2025): https://github.com/ethz-asl/wildgs-slam
- ADD-SLAM (2025): https://arxiv.org/abs/2505.19420
- DG-SLAM (NeurIPS 2025): https://github.com/feixue1205/DG-SLAM (推测)
- DGS-SLAM: https://arxiv.org/abs/2411.10722
- RoDyn-SLAM: https://arxiv.org/abs/2403.01779

---

## 11. 结语：对 SLAM + 3DGS 社区的启示

Flow4DGS-SLAM 的核心贡献不在于"又一种 dynamic SLAM"，而在于提出了一个 **template**：

> **"When explicit geometric priors (optical flow, depth, image Jacobian) are available, use them as feedforward signals to bootstrap learning, instead of learning them from scratch with implicit function approximators (MLPs)."**

这是 SLAM 圈一直在反思的事情：纯 learning-based 方法（NeRF-SLAM, MLP deformation）虽然 flexible 但 training 慢；传统 geometric 方法（optical flow + IRLS）快但 limited expressiveness。Hybrid 的 sweet spot 是：
- **Geometric prior 做初始化**（optical flow → camera pose, scene flow → Gaussian position）
- **Learning 做精细 refinement**（gradient descent on photometric loss + GMM temporal attributes）

这个 template 应该能推广到很多 scene：feature matching / ICP / photometric tracking 都可以替换 optical flow 的角色。未来 SLAM + 4DGS 系工作大概会沿这条路继续探索，把 optical flow 换成更 general 的 foundation model（如 [[20] SAM 2](https://github.com/facebookresearch/sam2) 提供的 mask + flow + feature），把 GMM 换成更 expressive 的 temporal basis（如 Fourier basis / Chebyshev polynomials）。

希望这个解析能 build your intuition, Andrej! 如果需要更深挖某个 module（比如 Eq.2 image Jacobian 的推导、GMM 的选择 motivation、KNN smoothing 的 τ 选择），我可以再展开。
