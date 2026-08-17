---
source_pdf: UNIDRIVE.pdf
paper_sha256: 12ca955465f24af9b01f0e63ac39479b11594d9a8ff43b62db640a49901922de
processed_at: '2026-08-12T19:27:03-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

设想我们在给自动驾驶汽车造 brain。现在的 vision-centric 模型（比如 BEVFusion, BEVFormer）有一个极其致命的弱点：它们把 camera 的 intrinsic matrix $K$ 和 extrinsic matrix $E$ 当成了死记硬背的常识。

你在 Tesla Model 3（摄像头偏低，FOV 是 120 度）上把模型训练好了，直接把它装到 Model X（摄像头偏高，FOV 是 100 度）上，模型瞬间变瞎。Table 1 的实验数据极其触目惊心：在 $6 \times 80°a$ 配置上训练的 BEVFusion-C，自身 mAP 是 66.7%。部署到 $6 \times 60°$ 配置时，mAP 直接暴跌到 1.8%。Camera configuration 的微小改变，对 neural network 来说就是毁灭性的 distribution shift。

UniDrive 的核心 idea 简单粗暴且极其优雅：既然 neural network 记不住多变的 camera 参数，干脆别让它记了。

我们在 network 前面加一个纯几何的 preprocessing 模块。不管你的车装的是 4 个广角还是 8 个窄角，不管装在车顶还是保险杠，这个模块统统把物理摄像头拍到的画面，扭曲、拼接、转换成一套“标准的虚拟视角图片”。

Neural network 永远只吃这套统一的 virtual images。它根本不知道物理世界里的 camera 长什么样。这就把一个困难的 generalization 问题，降维成了一个确定性的 geometry 问题。

### 1. Geometric Intuition: 如何造假图像

要做画面转换，必须解决单目相机的 depth ambiguity。画面里的一个 pixel，它可能代表远处的一个大卡车，也可能代表近处的一个小石头。不知道深度，就没法把 pixel 投影到 3D 空间去 warp。

UniDrive 用了一个非常符合驾驶场景物理直觉的 ground-aware assumption：

看画面的下半部分（近场），我们假设所有的东西都在地上。因为 camera 离地高度 $h_c$ 是已知的，用初中几何的相似三角形，就能算出 pixel 对应的 3D 坐标。
看画面的上半部分（远场），地面假设就不靠谱了（天上飞的东西、远处的楼房都不在地面上）。这时候我们假设这些点都在一个以车为中心、半径为 $D_0$ 的圆柱面上。

我们来看 paper 里的公式 1：
$$
(\hat{X}_c^{V_k}, \hat{Y}_c^{V_k}, \hat{Z}_c^{V_k}) = \left( \frac{f_y^{V_k}(u^{V_k} - c_x^{V_k})}{f_x^{V_k}(v^{V_k} - c_y^{V_k})} h_c, \; h_c, \; \frac{f_y^{V_k}}{v^{V_k} - c_y^{V_k}} h_c \right)
$$

这里的数学其实非常纯粹：
- $u^{V_k}, v^{V_k}$: virtual image 里的 pixel 坐标（列, 行）。
- $f_x^{V_k}, f_y^{V_k}$: virtual camera 的 focal length。
- $c_x^{V_k}, c_y^{V_k}$: virtual camera 的 principal point（图像中心）。
- $h_c$: camera 离地高度。

基于透视投影公式 $v = f_y \cdot \frac{Y_c}{Z_c} + c_y$，如果我们强制令 $Y_c = h_c$（假设这个点在地面上），我们就可以反解出深度 $Z_c = \frac{f_y h_c}{v - c_y}$。同理可以解出横向的 $X_c$。这就把一个不可解的 underdetermined inverse problem，变成了一个确定的 forward mapping。

拿着这些估算出来的 3D 坐标，根据物理相机的 $K$ 和 $E$，反向去原图里采样对应的 pixel，就完成了从“物理视角”到“虚拟视角”的 warp。多个物理相机拍到的同一块区域，再做一次 weighted blending：
$$
\mathbf{I}^{V_k} = \frac{1}{\mathbf{W}} \sum_{j=1}^{J} w_j \cdot \mathbf{I}^{V_k \leftarrow C_j}
$$
这就把多目图像完美地缝合成了统一的虚拟图像。

### 2. Optimization Intuition: 寻找最百搭的虚拟相机

这个 virtual camera 的 $K$ 和 $E$ 该怎么定？直接拍脑袋定在车顶正中央？

实验证明不行（Figure 5）。直觉配置往往会对某些物理配置产生偏好，导致 warp 的时候图像扭曲程度差异巨大。UniDrive 使用了 CMA-ES 算法去搜索最优的 virtual camera 参数。它的 objective function 是最小化 3D bounding box corner 在物理相机和虚拟相机之间的 angular projection error：

$$
\mathcal{E} = \sum_{n=1}^{N} \sum_{m=1}^{8} \mathcal{E}_{n,m}^{V_k} = \sum_{n=1}^{N} \sum_{m=1}^{8} D_{n,m}^{V_k} \cdot (\Delta\theta_{n,m}^{V_k} + \Delta\phi_{n,m}^{V_k})
$$

- $N$: 场景中 3D bounding box 的总数量。
- $m$: 每个 3D box 有 8 个 corner points。
- $\Delta\theta, \Delta\phi$: 真实物理投影与虚拟投影之间的 pitch 和 yaw 角度误差。
- $D_{n,m}^{V_k}$: 这个 corner 点到 camera 光心的距离。

为什么要用距离 $D$ 做权重？因为在远处，哪怕角度误差 $\Delta\theta$ 极小，实际的空间位移误差也会被距离放大。10 米处 0.01 度的误差是毫米级，100 米处 0.01 度的误差就是厘米级。

CMA-ES 不断采样不同的 virtual camera 参数（焦距、位置、朝向），评估它们对所有物理配置的投影误差总和，最终找到一个“无论你的实车配置怎么变，转换成我的虚拟配置时，信息损失最小”的完美中间态。

### 3. 数据层面的震撼与联想

看 Table 2 的数据，提升是碾压级的。还是在 $6 \times 80°a$ 配置上训练，部署到 $6 \times 60°$ 配置时：
- Baseline BEVFusion-C: mAP 1.8%
- UniDrive: mAP **59.6%**

网络根本没见过 $6 \times 60°$ 的画面，但因为几何转换，它以为自己还在看熟悉的 $6 \times 80°a$。

我极其欣赏这种 system design 的哲学。传统的 domain generalization 喜欢用 data augmentation，强行让 network 在训练时见识各种恶劣条件，试图让 network 自己学会 invariance。UniDrive 的哲学是 canonicalization，在 input stage 就用已知的数学物理公式把 variation 归一化掉。这跟 Spatial Transformer Networks 早期想把几何变换从 feature learning 中剥离出来的思想一脉相承，只不过 UniDrive 用的是极其硬核的 camera geometry。

这就引出了非常多可以 hallucination 的未来联想：

**A. Dynamic Object 的 Bug**：
目前的 ground assumption 假设近处所有点都在 $Y_c = h_c$ 的平面上。如果近处是一辆高高的大卡车，它的车头 pixel 在图像上方，被强行投影到地面平面上，会产生严重的 warp artifact。如果引入 monocular depth estimation（比如 DPT 或 Depth Anything），用 network 估计的 continuous depth 替代 piece-wise ground/cylinder assumption，是不是能进一步消除 dynamic object 的形变？

**B. Temporal Consistency**：
单帧的 warp 在视频流里可能会产生 jitter，因为每一帧的 blending 权重 $w_j$ 可能会因物理相机微小的 calibration 噪声而跳变。如果用 optical flow 约束 virtual image 的时序一致性，BEVFormer 里的 temporal attention 就能发挥更大威力。

**C. Multi-modal UniDrive**：
设想把 LiDAR point cloud 也 project 到这个 virtual camera space。车厂以后换车型，只换物理配置，不用重新训练 fusion 模型。LiDAR 和 camera 在 virtual space 里永远是对齐的。这简直是 fleet learning 的终极梦想。

**D. Absolute Scale 的注入**：
CMA-ES 找到的 virtual camera 配置，实际上包含了该数据集统计意义上最 robust 的 geometric prior。如果我们把 virtual camera 的 baseline 拉宽，是不是能人为制造出更强烈的 stereo effect，从而迫使 network 学到更绝对尺度的 depth 信息？

### Reference Web Links:
- UniDrive Project Page: https://wzzheng.net/UniDrive
- CARLA Simulator: http://carla.org/
- BEVFusion (MIT): https://github.com/mit-han-lab/bevfusion
- BEVFormer: https://github.com/fundamentalvision/BEVFormer
- Lift-Splat-Shoot: https://nv-tlabs.github.io/lift-splat-shoot/
- CMA-ES Algorithm: https://arxiv.org/abs/1604.00772
- Spatial Transformer Networks: https://arxiv.org/abs/1506.02025

---

# UniDrive: Universal Driving Perception Across Camera Configurations 深度解析

## 一、问题动机与直觉构建

Vision-centric autonomous driving 的核心痛点在于 **2D→3D lift transform 严重依赖 camera calibration**。当前 BEV-based 方法 (BEVDet, BEVFormer, PETR, BEVFusion 等) 都需要明确的 intrinsic matrix $\mathbf{K}$ 和 extrinsic matrix $\mathbf{E}$ 才能把 multi-view 2D feature 投到 3D/BEV 空间。这意味着模型不仅学到了 visual feature, 还隐式学到了 **固定的观测几何 (fixed observation perspective)**。

这里的关键 intuition: BEVFormer 这类方法把 image feature 通过 spatial cross-attention query 到 3D reference points, query points 的位置由 $\mathbf{K}, \mathmathbf{E}$ 决定。当 deployment 时 camera 参数变了, 同一个 3D point 在 image 上的位置完全不同, 但网络权重里编码了旧的 K, E 几何先验, 于是 attention query 找不到正确的 image feature。

Table 1 的数据触目惊心: BEVFusion-C 在 $6\times80°a$ 上训练, 自身 mAP = 66.7%, 但部署到 $6\times60°$ (只改了 FOV/intrinsic) 时 mAP 直接掉到 **1.8%**。这种 sensitivity 是 deployment 在不同 car models 上的根本障碍。

## 二、UniDrive 的核心 idea

UniDrive 的核心思路非常 elegant: **在 image 进入 network 之前, 先把所有可能 configuration 的图像 warp 到一个 unified virtual camera space**。网络只见过 virtual camera 配置的图像, 因此对物理 camera 参数完全 invariant。

整体 pipeline:
```
Original Images I^{C_j} (varying K, E)
        ↓ Virtual Camera Projection (ground-aware)
Unified Virtual Images I^{V_k} (fixed K*, E*)
        ↓ BEVFusion / BEVFormer / PETR 等
3D Detection / BEV Feature
```

关键 design choices:
1. **Virtual camera 是一组固定的、统一的虚拟相机** $\mathcal{V}=\{V_1, ..., V_K\}$, 不依赖物理 camera。
2. **Ground-aware projection** 解决 depth 不确定问题 (单目深度不可解)。
3. **CMA-ES 优化** virtual camera 参数, 最小化跨配置 projection error。

## 三、Virtual Camera Projection 详解

### 3.1 Ground-aware Depth Assumption

单目相机的根本问题是 depth ambiguity: 一个像素 $(u, v)$ 可以对应 ray 上的任意深度点。UniDrive 用 **piece-wise surface assumption** 来估计每个 virtual 像素的 3D 坐标:

**Step 1 - 假设地面 (近场)**: 把 virtual camera 中每个像素反投影, 假设其对应 3D 点在 ground plane $Y_c = h_c$ (相机离地高度)。

公式 (1):
$$
(\hat{X}_c^{V_k}, \hat{Y}_c^{V_k}, \hat{Z}_c^{V_k}) = \left( \frac{f_y^{V_k}(u^{V_k} - c_x^{V_k})}{f_x^{V_k}(v^{V_k} - c_y^{V_k})} h_c, \; h_c, \; \frac{f_y^{V_k}}{v^{V_k} - c_y^{V_k}} h_c \right)
$$

变量含义:
- $u^{V_k}, v^{V_k}$: virtual view 中像素坐标 (列, 行)
- $f_x^{V_k}, f_y^{V_k}$: virtual camera 在 x/y 方向的 focal length (像素单位)
- $c_x^{V_k}, c_y^{V_k}$: virtual camera 的 principal point (光轴与 image plane 的交点)
- $h_c$: camera 中心到 ground 的垂直高度
- $\hat{X}_c, \hat{Y}_c, \hat{Z}_c$: 假设地面时该像素对应 3D 点在 virtual camera 坐标系下的坐标

**几何推导**: 由 perspective projection $u = f_x \cdot X/Z + c_x$, $v = f_y \cdot Y/Z + c_y$, 设 $Y = h_c$, 联立解出 $X, Z$。这里 $Y_c = h_c$ 是因为相机坐标系下 Y 轴向下指地面, camera center 距离地面 $h_c$ 高度, 所以地面点的 Y 坐标就是 $h_c$。

**Step 2 - 判断远近场**: 计算 $\hat{D}_c^{V_k} = \|(\hat{X}_c, \hat{Y}_c, \hat{Z}_c)\|_2$, 与阈值 $D_0$ 比较。

- 若 $\hat{D}_c^{V_k} < D_0$: 接受地面假设。
- 若 $\hat{D}_c^{V_k} \geq D_0$: 远场地面假设不可靠 (远处的车辆、天空等不在地面上), 改用 **cylindrical surface assumption**。

**Cylindrical surface** (公式 2):
$$
(X_c^{V_k}, Y_c^{V_k}, Z_c^{V_k}) = \left( \frac{(u^{V_k} - c_x^{V_k}) D_0}{f_x^{V_k} d^{V_k}}, \; \frac{(v^{V_k} - c_y^{V_k}) D_0}{f_y^{V_k} d^{V_k}}, \; \frac{D_0}{d^{V_k}} \right)
$$
其中:
$$
d^{V_k} = \left\| \left( \frac{u^{V_k} - c_x^{V_k}}{f_x^{V_k}}, \; \frac{v^{V_k} - c_y^{V_k}}{f_y^{V_k}}, \; 1 \right) \right\|_2
$$

直觉: 把 ray direction $(\Delta u / f_x, \Delta v / f_y, 1)$ 归一化后乘以 $D_0$, 即沿 ray 方向取模长 $D_0$ 的点。这相当于把所有远场点投到一个 **以光心为圆心、半径 $D_0$ 的圆柱面** 上。

**为什么这个 assumption 合理**? 
- 近场 (路面、近车) 主要在 ground plane 上, ground assumption 几何准确。
- 远场 (远车、天空、建筑) 深度不可靠, 用 cylinder 提供一个 **一致的固定参考面**, 只要训练和测试用相同的 cylinder, network 学到的就是 cylinder 上的 feature, 而非真实 depth。这与 LSS 用 categorical depth distribution 的思路类似, 都是把 underdetermined inverse problem 替换为 consistent forward problem。

### 3.2 Point-wise Projection Pipeline

得到 virtual view 像素 $(u^{V_k}, v^{V_k})$ 的 3D 坐标后, 通过以下链路反查原图像对应像素:

1. Virtual camera frame → world frame:
$$\mathbf{p}_w = \mathbf{E}^{V_k} \cdot \mathbf{p}_c^{V_k}$$
   $\mathbf{E}^{V_k} \in \mathbb{R}^{4\times4}$ 是 virtual camera 的 extrinsic (RT 矩阵齐次形式)。

2. World frame → Original camera frame:
$$\mathbf{p}_c^{C_j} = (\mathbf{E}^{C_j})^{-1} \cdot \mathbf{p}_w$$

3. 3D point → Original image pixel:
$$(u^{C_j}, v^{C_j}, 1)^T = \mathbf{K}^{C_j} \cdot \mathbf{p}_c^{C_j}$$

整体映射: $\mathbf{P}_{V_k \leftarrow C_j}(\hat{D}_c^{V_k})$ 是一个 **depth-conditioned projection matrix**, 关键点是它依赖 virtual pixel 的 $\hat{D}_c^{V_k}$, 这把 depth assumption 编进了 sampling grid。

### 3.3 Image-level Warp 与 Blending

对整个 image 做 warping:
$$
\mathbf{I}^{V_k \leftarrow C_j} = \mathcal{T}(\mathbf{I}^{C_j}, \mathbf{P}_{V_k \leftarrow C_j}(\hat{D}_c^{V_k}))
$$
$\mathcal{T}$ 是可微 grid sampling (类似 Spatial Transformer Network 中的 bilinear sampler)。

由于多个原始相机都可能 cover 同一 virtual pixel, 需要 blending:
$$
\mathbf{I}^{V_k} = \frac{1}{\mathbf{W}} \sum_{j=1}^{J} w_j \cdot \mathbf{I}^{V_k \leftarrow C_j}, \quad \mathbf{W} = \sum_j w_j
$$
权重 $w_j$ 可以基于 angular distance 或 camera proximity。这一步本质上类似 multi-view stitching with soft blending, 解决 overlap 区域 ghosting 问题。

### 3.4 Algorithm 1 流程总结

对每个 virtual view $V_k$, 对每个 virtual pixel $(u^{V_k}, v^{V_k})$:
1. 用公式 1 计算 ground 假设下的 3D 点 + 距离 $\hat{D}$。
2. 比较阈值 $D_0$, 选择 ground 或 cylindrical assumption。
3. 通过 $\mathbf{E}^{V_k} \to (\mathbf{E}^{C_j})^{-1} \to \mathbf{K}^{C_j}$ 反查原图像素。
4. Bilinear sample 原图像。
5. 加权融合多个原相机的 warp 结果。

注意: 这整个 pipeline 是 **不需要训练的纯几何运算**, 因此可以作为完全 plug-and-play 模块插入任何 BEV 方法之前, 不改变下游网络的任何结构。

## 四、Virtual Projection Error 度量

为了优化 virtual camera 参数 $\mathbf{u} = \{V_k, \mathbf{K}^{V_k}, \mathbf{E}^{V_k}\}$, 需要可微/可评估的 error metric。UniDrive 用 **3D box corner 的 angular reprojection error** 作为度量。

### 4.1 角度计算

对 3D box $b_n$ 的 8 个 corners $(x_{n,m}, y_{n,m}, z_{n,m})$:

**通过 warp 链路投影** (公式 4):
$$
(\theta_{n,m}^{V_k}, \phi_{n,m}^{V_k}) = \left( \arctan \frac{v_{n,m}^{V_k} - c_y^{V_k}}{f_y^{V_k}}, \; \arctan \frac{u_{n,m}^{V_k} - c_x^{V_k}}{f_x^{V_k}} \right)
$$
- $\theta$: pitch angle (垂直方向)
- $\phi$: yaw angle (水平方向)
- 角度相对于 virtual camera 光心。

**直接投影到 virtual view** (公式 5, ground truth):
$$
(\theta_{n,m}^{V_k\prime}, \phi_{n,m}^{V_k\prime}) = \left( \arctan \frac{v_{n,m}^{V_k\prime} - c_y^{V_k}}{f_y^{V_k}}, \; \arctan \frac{u_{n,m}^{V_k\prime} - c_x^{V_k}}{f_x^{V_k}} \right)
$$

### 4.2 加权 error

$$
\Delta\theta_{n,m}^{V_k} = |\theta_{n,m}^{V_k} - \theta_{n,m}^{V_k\prime}|, \quad \Delta\phi_{n,m}^{V_k} = |\phi_{n,m}^{V_k} - \phi_{n,m}^{V_k\prime}|
$$

每个 corner 的 error 用其到光心的距离加权:
$$
\mathcal{E}_{n,m}^{V_k} = D_{n,m}^{V_k} \cdot (\Delta\theta_{n,m}^{V_k} + \Delta\phi_{n,m}^{V_k})
$$

**为什么用距离加权**? 物理直觉: 远处的小 angular error 对应大的 spatial displacement。一个 0.01 rad 的角度误差在 10m 处是 10cm, 在 50m 处是 50cm, 对检测 box corner 的位置精度影响完全不同。

### 4.3 总 error

$$
\mathcal{E} = \sum_{n=1}^{N} \sum_{m=1}^{8} \mathcal{E}_{n,m}^{V_k} \tag{6}
$$
对所有 3D box 的所有 corner 求和。

## 五、CMA-ES 优化 Virtual Camera Configuration

### 5.1 为什么用 CMA-ES?

Objective $\mathcal{E}_{total} = \sum_s \mathcal{E}^{(s)}(\mathbf{u})$ 是高度非凸、不可微 (有 thresholding、warping 等离散操作), gradient-based 方法不适用。CMA-ES (Covariance Matrix Adaptation Evolution Strategy) 是 black-box evolutionary optimization 的 state-of-the-art, 适合 10-100 维的 non-convex continuous optimization。

### 5.2 CMA-ES 数学框架

**采样**: 第 $t$ 代从多元正态分布采样 $N_t$ 个候选配置:
$$
\mathbf{u}_i^{(t)} \sim \mathcal{N}(\mathbf{m}^{(t)}, (\sigma^{(t)})^2 \mathbf{C}^{(t)})
$$
- $\mathbf{m}^{(t)}$: 均值向量 (当前最优估计)
- $\sigma^{(t)}$: 全局步长 (controls exploration scale)
- $\mathbf{C}^{(t)}$: 协方差矩阵 (controls search shape)

**Selection + Recombination** (公式 7):
$$
\mathbf{m}^{(t+1)} = \sum_{i=1}^{M_t} w_i \hat{\mathbf{u}}_i^{(t)}, \quad \mathcal{E}(\hat{\mathbf{u}}_1) \leq \mathcal{E}(\hat{\mathbf{u}}_2) \leq \cdots \leq \mathcal{E}(\hat{\mathbf{u}}_{M_t})
$$
选 error 最小的 $M_t$ 个, 加权平均得到下一代均值。注意 paper 这里写 $\geq$ 是 descending order, 实际 CMA-ES 标准是 ascending (best first)。

**Covariance Evolution Path** (公式 8):
$$
\mathbf{p}_C^{(t+1)} = (1-c_C) \cdot \mathbf{p}_C^{(t)} + \sqrt{1-(1-c_C)^2} \cdot \sqrt{\frac{1}{\sum_i w_i^2}} \cdot \frac{\mathbf{m}^{(t+1)} - \mathbf{m}^{(t)}}{\sigma^{(t)}}
$$
- $\mathbf{p}_C$: 累积搜索方向 (evolution path)
- $c_C$: covariance 学习率
- $\sqrt{1-(1-c_C)^2}$: 保持 $\mathbf{p}_C$ 方差的归一化系数
- $\sqrt{1/\sum w_i^2}$: selection 权重的归一化

**Covariance Update** (公式 9):
$$
\mathbf{C}^{(t+1)} = (1-c_C)\mathbf{C}^{(t)} + c_C \mathbf{p}_C^{(t+1)} (\mathbf{p}_C^{(t+1)})^T
$$
rank-1 update, 沿 evolution path 拉伸搜索分布。

**Step Size Evolution Path** (公式 10):
$$
\mathbf{p}_\sigma^{(t+1)} = (1-c_\sigma) \mathbf{p}_\sigma^{(t)} + \sqrt{1-(1-c_\sigma)^2} \cdot \sqrt{\frac{1}{\sum w_i^2}} \cdot \frac{\mathbf{m}^{(t+1)} - \mathbf{m}^{(t)}}{\sigma^{(t)}}
$$

**Step Size Update** (公式 11):
$$
\sigma^{(t+1)} = \sigma^{(t)} \exp\left( \frac{c_\sigma}{d_\sigma} \left( \frac{\|\mathbf{p}_\sigma^{(t+1)}\|}{\mathbb{E}\|\mathcal{N}(0, \mathbf{I})\|} - 1 \right) \right)
$$
- $c_\sigma$: step size 学习率
- $d_\sigma$: damping coefficient (控制 step size 调整速度)
- $\mathbb{E}\|\mathcal{N}(0, \mathbf{I})\|$: 标准正态分布范数的期望 (约 $\sqrt{n} \cdot (1 - 1/(4n) + ...)$, 用于归一化)
- 直觉: 当 evolution path 比期望长时, 增大步长; 反之缩小。

### 5.3 优化目标

$$
\mathcal{E}_{total} = \sum_{s=1}^{S} \mathcal{E}^{(s)}(\mathbf{u})
$$
$S$ 是共享 perception model 的所有 multi-camera system 数量。优化得到的 $\mathbf{u}^*$ 是一组 virtual camera 参数, 在所有 $S$ 个系统上平均 projection error 最小。

## 六、实验数据深度解读

### 6.1 数据集构建

- **CARLA simulator** 收集, Towns 1-6 地图, 6 类目标。
- **160,000 frames** multi-camera dataset, 8 种 configuration, 每个 config 500 scenes (20,000 frames)。
- 250 scenes 训练 / 250 scenes 验证。
- **关键**: 同一路线, 只改 camera configuration, 控制变量。

### 6.2 Camera Configuration 矩阵

8 种代表性 config:
| Config | 描述 |
|---|---|
| $4 \times 95°$ | 4 个 95° 大广角 |
| $5 \times 75°$ | 5 个 75° |
| $5 \times 70° + 110°$ | nuScenes 配置 (前后 intrinsic 不一致) |
| $6 \times 80°a$ | 6 个 80°, placement variant a |
| $6 \times 80°b$ | 6 个 80°, placement variant b |
| $6 \times 70°$ | 6 个 70° |
| $6 \times 60°$ | 6 个 60° (窄 FOV, 高分辨率) |
| $8 \times 50°$ | 8 个 50° (窄 FOV 多相机) |

### 6.3 Baseline BEVFusion-C 跨配置性能

读 Table 1 关键 cell:

**训练在 $6 \times 80°a$** (自测 mAP = 66.7%):
| Test Config | mAP | Gap |
|---|---|---|
| $6 \times 80°a$ (self) | 66.7% | - |
| $4 \times 95°$ | 3.8% | -62.9% |
| $5 \times 75°$ | 30.4% | -36.3% |
| $5 \times 70° + 110°$ | 9.2% | -57.5% |
| $6 \times 80°b$ (只改 placement) | 63.3% | -3.4% |
| $6 \times 70°$ | 16.4% | -50.3% |
| $6 \times 60°$ | 1.8% | -64.9% |
| $8 \times 50°$ | 0.4% | -66.3% |

**关键洞察**:
1. **Intrinsic 变化 (FOV) 是最大杀手**: FOV 从 80° 变 60° 或 50°, 性能直接掉到 < 2%, 几乎完全失效。这是因为 BEVFusion 的 view transformer 学到了 80° FOV 的 ray sampling pattern, FOV 变小后大部分 query points 落到 image 外部。
2. **Placement 变化影响小**: $6 \times 80°a \to 6 \times 80°b$ 只掉 3.4%, 因为 intrinsic 一致, 只是 extrinsic 微调, BEV attention 仍然能找到大致对应的 image region。
3. **Intrinsic 不一致 (nuScenes 类) 显著更难**: $5 \times 70° + 110°$ 比 $6 \times 80°$ 类配置明显难。

### 6.4 UniDrive 跨配置性能

读 Table 2 同样训练在 $6 \times 80°a$:
| Test Config | mAP | Gap vs self |
|---|---|---|
| $6 \times 80°a$ (self) | 69.4% | - |
| $4 \times 95°$ | 55.9% | -13.5% |
| $5 \times 75°$ | 65.2% | -4.2% |
| $5 \times 70° + 110°$ | 63.7% | -5.7% |
| $6 \times 80°b$ | 66.2% | -3.2% |
| $6 \times 70°$ | 68.9% | -0.5% |
| $6 \times 60°$ | 59.6% | -9.8% |
| $8 \times 50°$ | 61.2% | -8.2% |

**惊人提升**:
- $6 \times 60°$: 1.8% → **59.6%** (+57.8%)
- $8 \times 50°$: 0.4% → **61.2%** (+60.8%)
- $5 \times 70° + 110°$: 9.2% → **63.7%** (+54.5%)

最大性能下降仅 13.5% (跨到 4 个广角 95°), 而且这个 config 是相机数量都变了。在所有 6-camera 之间迁移, 下降都在 10% 以内, 这在 BEV-based 3D detection 是前所未有的 robustness。

### 6.5 Optimization 的作用 (Figure 5)

对比 UniDrive **w/o optimization** (直觉 virtual config: 所有相机居中放在车顶) vs **w/ optimization**:
- w/o optimization 虽比 baseline 大幅提升, 但 **对某些 config 有明显偏好** (性能分布不均, 有的 60+, 有的 30+)。
- w/ optimization 在 8 个 config 上 **均衡**, 全部在 55-70% 之间。

这印证了 CMA-ES 优化的必要性: 直觉配置可能在某些 direction 上有 blind spot, 优化后的 config 是 statistical 意义下对所有 training configs 都最优的 "中位数"。

### 6.6 Ablation Study (Figure 6)

**Intrinsic ablation**: 跨 FOV 变化时, UniDrive 最大掉 9.8%, BEVFusion-C 掉到 < 20% (最大掉 64.9%)。

**Camera height ablation** (1.6m 训练, 1.4/1.8/2.5m 测试):
- BEVFusion-C: > 10% drop
- UniDrive: 只 3.0% drop

直觉: virtual camera 高度固定为 $h_c$, warp 时通过 ground plane assumption 自动适应物理 camera 高度变化, 所以高度变化对网络几乎透明。

**Placement ablation**: $6 \times 80°a \leftrightarrow b$, UniDrive 只 4.6% drop (vs baseline 5.9%)。Placement 本来就影响小, UniDrive 略有改进。

### 6.7 Inconsistency in Intrinsics 分析

nuScenes 风格 $5 \times 70° + 110°$ 用了两种不同 FOV 的相机, BEVFusion-C 在这上面自身性能 63.9%, 比 $6 \times 80°a$ 自身 66.7% 略低; UniDrive 把 $5 \times 70° + 110°$ 自身性能提升到 **68.8%** (+4.9%)。

直觉: UniDrive 把不同 FOV 的物理相机全部 warp 到统一的 virtual FOV, 下游网络只看一致的 virtual image, 完全消除了 intrinsic inconsistency 的负担。

## 七、Intuition Building: 为什么 UniDrive 工作?

### 7.1 把 geometric invariance 从 network 移到 preprocessing

传统 BEV 方法让 network 隐式学习 $\mathbf{K}, \mathbf{E}$ 的几何先验, 这导致 network 的 weights 与具体 config 绑定。UniDrive 通过 deterministic geometric warping, 把这部分 invariance 实现成 **无参数的预处理**, network 只需学习 visual feature → 3D semantic 的映射。

类比: 这就像 data augmentation 的极端版本 — 不是 augmented, 而是 **canonicalized**, 把所有 input 标准化到一个固定 reference frame。

### 7.2 Ground-aware 假设的合理性

Driving scene 中, 摄像头朝外水平拍摄, 大部分 pixel content 集中在 ground 附近 (路、近车) 和远场 (天空、远景)。Ground assumption 在近场准确, cylindrical assumption 在远场提供 consistent reference。两者的切换通过 $D_0$ 控制, 形成类似 piece-wise 的几何模型。

这与 LSS (Lift-Splat-Shoot) 的 categorical depth 思路一致: 都是承认 monocular depth 是 underdetermined, 选择一个 **consistent 的假设** 而非 **accurate 的估计**, 让 network 在 consistent representation 上学习。

### 7.3 Virtual Camera 作为 Geometric Bottleneck

整个 pipeline 可视化为:
```
Any physical camera config → [geometric warp] → Fixed virtual config → [network]
```
virtual config 起到 **information bottleneck** 的作用: 不同 config 的图像都被压缩到统一的 perspective, 网络永远看到相同视角的 "virtual reality"。

这个 idea 与 Neural Radiance Fields (NeRF) 中 canonical coordinate、与 multi-view stereo 中的 reference frame 选择有思想上的关联。

### 7.4 CMA-ES 优化的意义

不优化时, virtual config 的选择是 ad-hoc (例如放在车顶中心), 可能导致某些 physical config 的 warp 角度极端 (大 distortion、信息丢失)。CMA-ES 找到的 config 是 **在所有 training configs 的分布上 expected projection error 最小的中位数 config**, 类似 median 在统计上的 robust 性质。

## 八、Limitations 与 Potential Extensions

Paper 自述 limitations:
1. Configuration 空间有限 (8 种), 真实世界 config 更多样。
2. 完全在 CARLA 仿真上验证, 真实场景未测。

我的额外思考:
- **Real-world depth ambiguity 更严重**: CARLA 的 synthetic image 在远场比 real image 更 clean。Real image 中 ground plane 经常被遮挡 (车辆、行人), 此时 ground assumption 失效, 可能需要 learned depth estimate 作为替代。
- **Dynamic objects**: ground assumption 对行驶中的车辆 (尤其是大型卡车) 不准确, 它们的 pixel 对应 3D 点 y > h_c, 但被强行解释为地面点, 可能造成 warp artifact。可以考虑 instance-aware depth assumption。
- **Temporal consistency**: 单帧 warp 没有利用 temporal cues, BEVFormer/PETRv2 的 temporal attention 可能与 UniDrive 配合时遇到 frame-to-frame warp 不一致问题。
- **Multi-modal extension**: LiDAR-camera fusion 中, 可以把 LiDAR point cloud 也 project 到 virtual camera, 实现跨 sensor config 的 multi-modal fusion。

## 九、Related Work 联想

### 9.1 与 LSS 系列 (BEVDet, BEVFusion) 的对比

LSS 用 categorical depth distribution per pixel, 然后 splat 到 BEV。UniDrive 把 depth 假设显式做在 virtual camera 这边, 而非 pixel 那边。LSS 的 depth 是 network 隐式学的 (有 BEV depth supervision 时显式), 仍然与具体 config 绑定。UniDrive 把 depth assumption 完全脱离 network, 实现 config-invariant。

### 9.2 与 BEVFormer 的 attention 机制对比

BEVFormer 用 deformable attention 在 image feature 上 sample 3D reference points 周围的 feature, query 点位置由 $\mathbf{K}, \mathbf{E}$ 决定。如果 K, E 变了, query 点位置变化, network 必须重新学 attention pattern。UniDrive 把 image warp 到 virtual view 后, BEVFormer 在 virtual config 上 query, query pattern 始终一致。

### 9.3 与 CamConv (Facil et al. 2019) 的关联

CamConv 把 camera 参数编码成 multi-scale convolution feature, 注入网络使其 camera-aware。UniDrive 是相反思路: camera-invariant, 通过 preprocessing 消除 camera 信息。

### 9.4 与 Domain Adaptation 的区别

传统 DA 解决 environmental domain gap (rain, night, sensor noise), UniDrive 解决 **geometric domain gap** (camera placement & intrinsics), 这是一个 relatively new 的方向。

## 十、Reference Web Links

- **UniDrive 项目主页**: https://wzzheng.net/UniDrive
- **CARLA Simulator**: https://carla.org/
- **nuScenes Dataset**: https://www.nuscenes.org/
- **BEVFusion (Liu et al. 2023b)**: https://github.com/mit-han-lab/bevfusion
- **BEVFormer**: https://github.com/fundamentalvision/BEVFormer
- **PETR / PETRv2**: https://github.com/megvii-research/PETR
- **Lift-Splat-Shoot (Philion & Fidler 2020)**: https://nv-tlabs.github.io/lift-splat-shoot/
- **CMA-ES Tutorial (Hansen 2016)**: https://arxiv.org/abs/1604.00772
- **CMA-ES Python 实现**: https://github.com/CMA-ES/pycma
- **BEVDet**: https://github.com/HuangJunjie2017/BEVDet
- **FCOS3D**: https://github.com/open-mmlab/mmdetection3d
- **SurroundOcc**: https://github.com/weiyithu/SurroundOcc
- **OccWorld (Zheng et al. 2024a)**: https://wzzheng.net/OccWorld/
- **GaussianFormer**: https://github.com/huang-yh/GaussianFormer
- **Cross-domain Perception survey (Wang et al. 2023b)**: https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Towards_Domain_Generalization_for_Multi-View_3D_Object_Detection_in_Bird-Eye-View_CVPR_2023_paper.pdf

## 十一、Final Intuition 总结

UniDrive 的 elegance 在于: **把一个本来要求 network 学的 invariance (camera config), 用纯几何运算显式实现**, 让 network 卸下了几何包袱, 专注学习 visual semantics。这与 NeRF 的 volume rendering、与 point cloud 的 voxelization 一样, 都是 **用 deterministic geometric prior 替代 learned prior** 的经典思路。

核心 insight 可以浓缩为: 当一个 invariance 可以通过已知数学公式 (这里: projective geometry + ground assumption) 表达时, **把它从 network 中拆出来放到 preprocessing**, 比让 network 隐式学习更 sample-efficient、更 robust、更 interpretable。

UniDrive 没有提出新 network, 没有新 loss, 没有新 benchmark metric — 它提出的是 **新的 representation space (virtual camera space)**, 这正是 system-level thinking 在 perception model 中的体现。这种思路对未来 multi-vehicle fleet learning、cross-platform model sharing 有重要意义。

如果让我推测下一步研究方向:
1. **Learned virtual camera**: 用网络学习 virtual config 而非 CMA-ES, 可以联合 end-to-end 训练。
2. **Instance-aware depth**: 结合 monocular depth estimation 替代 ground/cylinder 假设, 提升 dynamic object 的 warp 准确性。
3. **Real-world validation**: nuScenes/Waymo/Argoverse 跨 dataset 跨 config 实验, 验证真实 noise 下的 robustness。
4. **Multi-modal UniDrive**: LiDAR + camera joint virtual representation, 实现 sensor-level config invariance。
