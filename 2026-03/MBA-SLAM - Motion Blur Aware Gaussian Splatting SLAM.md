我来详细讲解这篇关于运动模糊感知的Gaussian Splatting SLAM论文。让我先读取更多内容来全面了解这篇论文。



# MBA-SLAM: Motion Blur Aware Gaussian Splatting SLAM 论文详解

## 一、研究背景与动机

### 1.1 核心问题

现有的**Neural Radiance Fields (NeRF)** 和 **3D Gaussian Splatting (3DGS)** 基于的 **SLAM** 系统（如 iMAP、NICE-SLAM、CoSLAM、ESLAM、SplaTAM、Photo-SLAM、MonoGS 等）存在一个关键缺陷：**它们都假设输入是高质量的清晰RGB-D图像**。

然而在真实场景中，特别是在**低光环境**或**长曝光条件下**，**运动模糊** 是一个非常普遍的问题。这种运动模糊会导致两个严重问题：

#### 问题1：跟踪阶段的姿态估计不准确
```
传统VO/SLAM的假设: 像素强度一致性 = 光度误差最小化
运动模糊破坏了: 像素特征发生空间偏移 → 光度一致性失效
结果: 从模糊帧中恢复准确的相机姿态变得极其困难
```

#### 问题2：建图阶段的多视图几何不一致
```
模糊帧之间的特征不匹配 → 引入错误的3D几何信息
→ 地图重建质量下降
→ 反过来进一步影响跟踪过程
```

### 1.2 相关工作回顾

| 方向 | 代表方法 | 局限性 |
|------|---------|--------|
| **NeRF-based SLAM** | iMAP[9]、NICE-SLAM[10]、CoSLAM[11]、ESLAM[12]、Point-SLAM[26] | 依赖清晰输入，运动模糊导致性能崩溃 |
| **3DGS-based SLAM** | GS-SLAM[14]、SplaTAM[15]、Photo-SLAM[17]、MonoGS[16] | 同样对运动模糊敏感 |
| **NeRF去模糊** | Deblur-NeRF[28]、DP-NeRF[29]、BAD-NeRF[22] | 需要COLMAP预估计姿态，收敛效率低，难以集成到SLAM系统 |
| **3DGS去模糊** | Deblur-GS[30]、BAGS[31]、BAD-Gaussians[23] | 同样需要准确的初始姿态 |
| **经典运动模糊VO** | Lee et al.[44] | 假设平滑运动，线性插值，依赖初始预测 |

## 二、MBA-SLAM系统架构

### 2.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    MBA-SLAM Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                              │
│  输入: 模糊RGB-D图像流                                      │
│         ↓                                                    │
│  ┌────────────────────────────────────────────────┐           │
│  │   Tracker (运动模糊感知跟踪)                 │           │
│  │   • 从当前关键帧渲染虚拟清晰图像            │           │
│  │   • 估计曝光期间相机运动轨迹:                │           │
│  │     - T_start (曝光开始时的姿态)            │           │
│  │     - T_end (曝光结束时的姿态)              │           │
│  │   • 基于轨迹对参考帧进行"重模糊"          │           │
│  │   • 最小化: 真实模糊帧 vs 重模糊帧的光度误差│           │
│  └────────────────────────────────────────────────┘           │
│         ↓                                                    │
│  ┌────────────────────────────────────────────────┐           │
│  │   Mapper (去模糊建图)                       │           │
│  │   方式1: NeRF-based Mapper                   │           │
│  │     • 三平面特征编码 (Tri-planes)           │           │
│  │     • MLP解码器                             │           │
│  │     • 体素渲染                              │           │
│  │   方式2: 3DGS-based Mapper                  │           │
│  │     • 3D高斯原语 (Gaussian primitives)      │           │
│  │     • 可微光栅化                            │           │
│  │   • 联合优化: 场景表示 + 相机轨迹          │           │
│  └────────────────────────────────────────────────┘           │
│         ↓                                                    │
│  输出: 清晰的3D地图 + 准确的相机轨迹                         │
│                                                              │
└─────────────────────────────────────────────────────────────────┘
```

## 三、核心方法详解

### 3.1 运动模糊成像模型

#### 3.1.1 连续模型

运动模糊图像的物理形成过程可以用积分表示：

$$\mathbf{B}(\mathbf{x}) = \phi \int_{0}^{\tau} \mathbf{I}_{\mathrm{t}}(\mathbf{x}) \, \mathrm{dt}$$

其中：
- $\mathbf{B}(\mathbf{x}) \in \mathbb{R}^{W \times H \times 3}$：捕获的模糊图像
- $\mathbf{x} \in \mathbb{R}^{2}$：像素位置
- $W, H$：图像宽度和高度
- $\phi$：归一化因子
- $\tau$：相机曝光时间
- $\mathbf{I}_{\mathrm{t}}(\mathbf{x}) \in \mathbb{R}^{W \times H \times 3}$：曝光时间 $t$ 时刻的虚拟清晰图像

**物理意义**：模糊图像是曝光期间所有虚拟清晰图像的积分平均。

#### 3.1.2 离散近似

$$\mathbf{B}(\mathbf{x}) \approx \frac{1}{n} \sum_{i=0}^{n-1} \mathbf{I}_{\mathrm{i}}(\mathbf{x})$$

其中 $n$ 是离散采样数量。

### 3.2 相机运动轨迹建模

#### 3.2.1 SE(3)空间中的连续运动

传统方法通常只估计一个瞬时姿态，但这无法处理曝光期间的连续运动。MBA-SLAM采用**双端点参数化**：

$$\mathbf{T}_{t} = \mathbf{T}_{\mathrm{start}} \cdot \exp\left(\frac{t}{\tau} \cdot \log(\mathbf{T}_{\mathrm{start}}^{-1} \cdot \mathbf{T}_{\mathrm{end}})\right)$$

其中：
- $\mathbf{T}_{\mathrm{start}} \in \mathbf{SE}(3)$：曝光开始时的相机姿态
- $\mathbf{T}_{\mathrm{end}} \in \mathbf{SE}(3)$：曝光结束时的相机姿态
- $t \in [0, \tau]$：曝光时间内的任意时刻
- $\exp(\cdot)$ 和 $\log(\cdot)$：**李群指数映射**和**对数映射**（在李代数 $\mathfrak{se}(3)$ 上进行线性插值）

#### 3.2.2 四元数形式的分解

为了实现，方程可以分解为旋转和平移部分：

**旋转部分（四元数插值）**：
$$\bar{\mathbf{q}}_{t} = \bar{\mathbf{q}}_{\mathrm{start}} \otimes \exp\left(\frac{t}{\tau} \cdot \log((\bar{\mathbf{q}}_{\mathrm{start}})^{-1} \otimes \bar{\mathbf{q}}_{\mathrm{end}})\right)$$

其中：
- $\bar{\mathbf{q}} = [q_x, q_y, q_z, q_w]^T$：单位四元数
- $\otimes$：四元数乘法算子
- $\Delta \bar{\mathbf{q}} = \exp(\Delta \mathbf{r})$：小量更新，$\Delta \mathbf{r} = [\Delta r_x, \Delta r_y, \Delta r_z]^T$

**平移部分（线性插值）**：
$$\mathbf{t}_{t} = \mathbf{t}_{\mathrm{start}} + \frac{t}{\tau}(\mathbf{t}_{\mathrm{end}} - \mathbf{t}_{\mathrm{start}})$$

或者等价地：
$$\mathbf{t}_{t} = \frac{\tau-t}{\tau}\mathbf{t}_{\mathrm{start}} + \frac{t}{\tau}\mathbf{t}_{\mathrm{end}}$$

#### 3.2.3 雅可比矩阵推导

为了进行优化，需要推导雅可比矩阵。

**平移部分的雅可比**：
$$\frac{\partial \mathbf{t}_{t}}{\partial \mathbf{t}_{\mathrm{start}}} = \begin{bmatrix} \frac{\tau-t}{\tau} & 0 & 0 \\ 0 & \frac{\tau-t}{\tau} & 0 \\ 0 & 0 & \frac{\tau-t}{\tau} \end{bmatrix}, \quad \frac{\partial \mathbf{t}_{t}}{\partial \mathbf{t}_{\mathrm{end}}} = \begin{bmatrix} \frac{t}{\tau} & 0 & 0 \\ 0 & \frac{t}{\tau} & 0 \\ 0 & 0 & \frac{t}{\tau} \end{bmatrix}$$

**旋转部分的雅可比**涉及四元数运算，较为复杂。论文使用 **Mathematica Symbolic Toolbox** 进行符号推导。

### 3.3 运动模糊感知跟踪器

#### 3.3.1 传统直接法图像对齐

对于清晰图像：
$$\mathbf{T}^* = \operatorname*{argmin}_{\mathbf{T}} \sum_{i=0}^{m-1} \left\lVert \mathbf{I}_{\mathrm{ref}}(\mathbf{x}_i) - \mathbf{I}_{\mathrm{cur}}(\hat{\mathbf{x}}_i) \right\rVert_{2}^{2}$$

其中：
- $\mathbf{I}_{\mathrm{ref}}$：参考图像
- $\mathbf{I}_{\mathrm{cur}}$：当前图像
- $\mathbf{x}_i$：参考图像中的像素位置
- $\hat{\mathbf{x}}_i$：当前图像中的对应像素位置，通过投影变换 $\hat{\mathbf{x}}_i = \pi(\mathbf{T} \cdot \pi^{-1}(\mathbf{x}_i, d_i))$ 得到

#### 3.3.2 运动模糊情况下的挑战

直接法假设**光度一致性**，但运动模糊破坏了这个假设：
```
清晰帧1中像素(100,100)的强度 I₁(100,100) ≈ 150
↓ 相机运动 → 像素内容空间偏移
模糊帧1中对应位置的实际强度 I₂(100,100) ≈ 平均化(150, 148, 152, 149) ≈ 149.75
```

#### 3.3.3 重模糊策略

MBA-SLAM采用**重模糊**而非**去模糊**，原因：
1. 去模糊是**病态问题**，对严重模糊的图像效果差
2. 重模糊是**物理可建模**的，更稳定

**重模糊过程**：
$$\hat{\mathbf{B}}_{\mathrm{cur}}(\mathbf{x}) = \frac{1}{n} \sum_{i=0}^{n-1} \mathbf{I}_{\mathrm{ref}}(\mathbf{x}_{\frac{i\tau}{n-1}})$$

其中 $\mathbf{x}_{\frac{i\tau}{n-1}}$ 是通过将参考图像中的像素通过第 $i$ 个虚拟相机姿态变换得到。

#### 3.3.4 跟踪器优化目标

$$\{\mathbf{T}_{\mathrm{start}}^{*}, \mathbf{T}_{\mathrm{end}}^{*}\} = \operatorname*{argmin}_{\mathbf{T}_{\mathrm{start}}, \mathbf{T}_{\mathrm{end}}} \sum_{i=0}^{m-1} \left\lVert \mathbf{B}_{\mathrm{cur}}(\mathbf{x}_i) - \hat{\mathbf{B}}_{\mathrm{cur}}(\mathbf{x}_i) \right\rVert$$

#### 3.3.5 像素传递几何细节

参考图2和图3，论文详细推导了像素传递的几何关系：

给定虚拟相机姿态 $\mathbf{T}_{i}^{\mathrm{ref}}$（相对于参考关键帧），其平移向量 $[p_x, p_y, p_z]^T$，旋转用四元数 $\bar{\mathbf{q}} = [q_x, q_y, q_z, q_w]^T$ 表示。

**1. 计算平面距离**：
$$d' = d - p_z$$

其中 $d$ 是前向平行平面相对于参考关键帧的深度。

**2. 计算射线与平面法向的夹角余弦**：
$$\lambda = \cos(\theta) = (\mathbf{R}_{i}^{\mathrm{ref}} \cdot \pi^{-1}(\mathbf{x}))^T \cdot [0, 0, 1]^T$$

简化为：
$$\lambda = 2x(q_x q_z - q_w q_y) + 2y(q_x q_w + q_y q_z) + z(q_w^2 - q_x^2 - q_y^2 + q_z^2)$$

**3. 计算射线长度**：
$$|L| = \frac{d'}{\lambda} = \frac{d - p_z}{\lambda}$$

**4. 计算3D交点**：
$$\mathbf{p}_{3d} = |L| [x, y, z]^T = \frac{d - p_z}{\lambda} [x, y, z]^T$$

**5. 变换到参考坐标系并投影**：
$$\mathbf{p}_{3d}' = \mathbf{T}_{i}^{\mathrm{ref}} \cdot \mathbf{p}_{3d}, \quad \mathbf{x}_{\frac{i\tau}{n-1}} = \pi(\mathbf{p}_{3d}')$$

### 3.4 去模糊建图器

#### 3.4.1 两种场景表示

**方式1：NeRF-based Mapper (隐式表示)**

使用**三平面特征编码**：
```python
# 伪代码
def get_features(X_w):
    # X_w: 3D点在世界坐标系中的位置
    fa_xy = Fa_xy(X_w)  # XY平面特征
    fa_xz = Fa_xz(X_w)  # XZ平面特征
    fa_yz = Fa_yz(X_w)  # YZ平面特征
    fg_xy = Fg_xy(X_w)
    fg_xz = Fg_xz(X_w)
    fg_yz = Fg_yz(X_w)
    
    # 汇聚特征
    fa = fa_xy + fa_xz + fa_yz  # 外观特征
    fg = fg_xy + fg_xz + fg_yz  # 几何特征
    
    # 通过MLP解码器
    color = ha(fa)
    sdf = hg(fg)
    density = beta * sigmoid(-beta * sdf)
    
    return color, density
```

**体渲染公式**：
$$w_i = \exp\left(-\sum_{k=1}^{i-1} \sigma_k\right)(1 - \exp(-\sigma_i))$$
$$\mathbf{I}(\mathbf{x}) = \sum_{i=1}^{n} w_i \mathbf{c}_i, \quad \mathbf{D}(\mathbf{x}) = \sum_{i=1}^{n} w_i d_i$$

**NeRF版本的总损失**：
$$\mathcal{L} = \lambda_c \mathcal{L}_c + \lambda_d \mathcal{L}_d + \lambda_{fs} \mathcal{L}_{fs} + \lambda_{sdf} \mathcal{L}_{sdf}$$

其中：
- $\mathcal{L}_c$：颜色损失
- $\mathcal{L}_d$：深度损失
- $\mathcal{L}_{fs}$：自由空间损失
- $\mathcal{L}_{sdf}$：SDF损失

**方式2：3DGS-based Mapper (显式表示)**

每个高斯原语 $\mathbf{G}$ 的参数化：
- $\boldsymbol{\mu} \in \mathbb{R}^3$：均值位置
- $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$：3D协方差
- $o \in \mathbb{R}$：不透明度
- $\mathbf{c} \in \mathbb{R}^3$：颜色

**高斯分布定义**：
$$\mathbf{G}(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})}$$

**协方差分解**：
$$\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T, \quad \boldsymbol{\Sigma}' = \mathbf{J} \mathbf{R}_c \boldsymbol{\Sigma} \mathbf{R}_c^T \mathbf{J}^T$$

其中：
- $\mathbf{S} \in \mathbb{R}^3$：尺度
- $\mathbf{R} \in \mathbb{R}^{3 \times 3}$：旋转矩阵（用四元数存储）
- $\mathbf{J} \in \mathbb{R}^{2 \times 3}$：透视变换的仿射近似雅可比

**光栅化渲染**：
$$\mathbf{I}(\mathbf{x}) = \sum_{i}^{N} \mathbf{c}_i \alpha_i \mathbf{T}_i, \quad \mathbf{D}(\mathbf{x}) = \sum_{i}^{N} d_i \alpha_i \mathbf{T}_i$$

其中：
$$\mathbf{T}_i = \prod_{j}^{i-1}(1 - \alpha_j), \quad \alpha_i = o_i \cdot \exp\left(-\frac{1}{2}\Delta_i^T {\boldsymbol{\Sigma}'}^{-1} \Delta_i\right)$$

**3DGS版本的总损失**：
$$\mathcal{L} = \lambda_c \mathcal{L}_c + \lambda_d \mathcal{L}_d + \lambda_{ssim} \mathcal{L}_{ssim} + \lambda_{reg} \mathcal{L}_{reg}$$

其中尺度正则化损失：
$$\mathcal{L}_{reg} = \frac{1}{|G|} \sum_{\mathbf{g} \in G} \max\{\max(\mathbf{S}_g)/\min(\mathbf{S}_g), r\} - r$$

#### 3.4.2 损失函数

**颜色损失和深度损失**：
$$\mathcal{L}_c = \frac{1}{|R|} \sum_{\mathbf{x} \in R} \left\lVert \mathbf{B}(\mathbf{x}) - \mathbf{B}^{gt}(\mathbf{x}) \right\rVert$$
$$\mathcal{L}_d = \frac{1}{|R|} \sum_{\mathbf{x} \in R} \left\lVert \mathbf{D}(\mathbf{x}) - \mathbf{D}^{gt}(\mathbf{x}) \right\rVert$$

其中 $\mathbf{B}(\mathbf{x})$ 是通过运动模糊成像模型从场景表示 $\mathbf{M}_{\theta}$ 合成的模糊颜色。

## 四、实验结果分析

### 4.1 数据集

| 类型 | 数据集 | 描述 |
|------|--------|------|
| **合成模糊** | ArchViz | Unreal Engine渲染的快速来回晃动相机序列 |
| **真实模糊** | RealSense采集 | 低光条件下用RealSense RGB-D相机采集（带动作捕捉GT） |
| **标准清晰** | Replica | 合成室内场景 |
| **标准清晰** | ScanNet | 真实RGB-D数据集 |
| **标准清晰** | TUM RGB-D | 真实RGB-D数据集 |

### 4.2 评估指标

- **跟踪**：ATE RMSE (cm) - 绝对轨迹误差均方根
- **渲染**：PSNR ↑, SSIM ↑, LPIPS ↓
- **重建**：Depth L1 ↓, Precision ↑, Recall ↑, F1 ↑ (<1cm%)

### 4.3 运动模糊数据集结果

#### 4.3.1 跟踪性能 (表I - ArchViz)

| 方法 | ArchViz-1 | ArchViz-2 | ArchViz-3 | Avg. |
|------|-----------|-----------|-----------|------|
| iMAP | 255.75 | 186.87 | 756.55 | 399.72 |
| CoSLAM | 5.28 | 4.67 | 14.17 | **8.04** (NeRF类第二) |
| ESLAM | 20.12 | 12.61 | ✖ | - |
| Point-SLAM | 289.56 | 181.27 | 596.25 | 355.69 |
| **Ours-NeRF** | **0.98** | **1.13** | **2.96** | **1.69** ✓ |
| SplaTAM | 36.88 | ✗ | 763.93 | - |
| Photo-SLAM | 4.57 | 0.64 | 2.01 | 2.41 |
| MonoGS | 1.92 | 2.96 | 38.37 | 14.42 |
| **Ours-GS** | **0.75** | **0.36** | **1.41** | **0.84** ✓ |

**关键观察**：
1. MBA-SLAM在两个版本上都取得了**最佳跟踪性能**
2. Point-SLAM和SplaTAM在运动模糊下性能严重下降
3. 隐式方法（CoSLAM、ESLAM）比显式点云/Gaussian方法在模糊下表现更好

#### 4.3.2 渲染性能 (表II - ArchViz)

| 方法 | Arch1 PSNR | Arch1 SSIM | Arch1 LPIPS | Avg. PSNR | Avg. SSIM |
|------|------------|------------|------------|-----------|-----------|
| VoxFusion | 17.97 | 0.519 | 0.511 | 16.19 | 0.452 |
| CoSLAM | 22.97 | 0.841 | 0.361 | 22.13 | 0.756 |
| ESLAM | 21.07 | 0.766 | 0.446 | - | - |
| Point-SLAM | 15.90 | 0.493 | 0.549 | - | - |
| **Ours-NeRF** | 24.77 | 0.905 | 0.312 | 25.08 | 0.887 |
| SplaTAM | 17.69 | 0.538 | 0.542 | - | - |
| Photo-SLAM | 23.45 | 0.823 | 0.387 | 24.88 | 0.855 |
| MonoGS | 23.18 | 0.852 | 0.337 | 23.63 | 0.811 |
| **Ours-GS** | **28.45** | **0.956** | **0.148** | **28.82** | **0.950** |

**关键观察**：
1. Ours-GS在PSNR上领先Photo-SLAM约4dB，在LPIPS上改善约0.2
2. Ours-NeRF在PSNR上领先CoSLAM约3dB
3. SSIM指标显示MBA-SLAM生成的图像结构更准确

#### 4.3.3 真实数据集结果 (表III)

| 方法 | pub1 | pub2 | pub3 | Real seq1 | Real seq2 | Real seq3 | Avg. (Public) | Avg. (Real) |
|------|------|------|------|-----------|-----------|-----------|--------------|------------|
| NICE-SLAM | 5.27 | 3.64 | 2.96 | 16.38 | 15.61 | 9.84 | 3.96 | 13.94 |
| CoSLAM | 6.32 | 1.58 | 2.75 | 12.97 | 7.63 | 7.15 | 3.55 | 9.25 |
| ESLAM | 4.71 | 1.29 | 2.44 | 11.58 | 6.49 | 8.06 | 2.81 | 8.71 |
| Point-SLAM | 5.13 | 3.16 | 2.56 | 10.52 | 9.18 | **235.79** | 3.62 | 85.16 |
| **Ours-NeRF** | 3.53 | 1.42 | 2.08 | 9.75 | 7.32 | 5.54 | **2.34** ✓ | **7.54** ✓ |
| SplaTAM | 4.89 | 3.29 | 2.58 | 33.56 | 14.51 | 7.26 | 3.59 | 18.44 |
| Photo-SLAM | 3.89 | 3.47 | 2.36 | 8.64 | **96.30** | 2.49 | 3.24 | 35.81 |
| MonoGS | 5.52 | 3.07 | 2.28 | 8.92 | 6.13 | 9.94 | 3.62 | 8.33 |
| **Ours-GS** | 4.16 | 2.78 | 2.16 | 8.79 | 5.56 | 3.27 | 3.03 | **5.87** ✓ |

**关键观察**：
1. Point-SLAM在真实序列seq3上完全失败（ATE RMSE = 235.79 cm）
2. MBA-SLAM在真实模糊数据上保持稳定性能
3. Photo-SLAM在seq2上也出现严重跟踪失败

### 4.4 标准清晰数据集结果

#### 4.4.1 跟踪性能 (表IV - Replica)

| 方法 | Avg. ATE (cm) | 与SOTA对比 |
|------|--------------|-----------|
| RTG-SLAM | 0.18 | 最佳 |
| **Ours-GS** | **0.35** ✓ | 第二 |
| Photo-SLAM | 0.43 | 第三 |
| **Ours-NeRF** | **0.41** ✓ | 第四 |
| Point-SLAM | 0.54 | |
| CoSLAM | 1.00 | |
| ESLAM | 0.63 | |

**关键观察**：
1. 即使不建模模糊过程，MBA-SLAM在清晰数据上仍然表现出色
2. Ours-NeRF和Ours-GS分别达到22.57 FPS和1.97 FPS

#### 4.4.2 渲染性能 (表V - Replica)

| 方法 | Avg. PSNR | Avg. SSIM | Avg. LPIPS | FPS |
|------|-----------|-----------|------------|-----|
| Point-SLAM | 35.05 | 0.975 | 0.124 | 1.33 |
| MonoGS | 37.13 | 0.985 | 0.069 | 769.00 |
| Photo-SLAM | 35.96 | 0.974 | 0.087 | 1084.00 |
| **Ours-GS** | **36.98** | **0.990** | **0.043** | 497.63 ✓ |
| **Ours-NeRF** | **30.85** | **0.947** | **0.220** | 2.75 ✓ |

**关键观察**：
1. Ours-GS在PSNR上领先Photo-SLAM约1dB
2. LPIPS指标大幅改善（从0.087降到0.043），说明感知质量更好

### 4.5 效率分析 (表VIII)

| 方法 | Tracking (ms) | Mapping (ms) | System FPS | ATE (cm) | Memory |
|------|---------------|--------------|------------|-----------|--------|
| NICE-SLAM | 6640 | 17178 | 2.91 | 1.69 | 48.48 MB |
| CoSLAM | 6010 | 13180 | 16.64 | 0.70 | 6.36 MB |
| ESLAM | 5480 | 29805 | 13.42 | 0.71 | 27.12 MB |
| Point-SLAM | 17440 | 104430 | 0.42 | 0.56 | 12508.62 MB |
| **Ours-NeRF** | **34.48** ✓ | **29940** | **22.57** ✓ | 0.34 | 27.12 MB |
| GS-SLAM | 1190 | 12800 | 8.34 | 0.48 | 198.04 MB |
| SplaTAM | 36250 | 21198 | 0.47 | 0.34 | 253.34 MB |
| Photo-SLAM | 4118 | 19400 | 6.61 | 0.33 | 59.88 MB |
| MonoGS | 254 | 26560 | 1.63 | 0.42 | 193.71 MB |
| **Ours-GS** | **34.23** ✓ | **75630** | **1.97** ✓ | 0.25 | 242.08 MB |

**关键观察**：
1. **Ours-NeRF**的CUDA优化跟踪器比传统方法快约**100倍**（34.48ms vs 5000-7000ms）
2. **Ours-GS**虽然慢于GS-SLAM，但快于SplaTAM
3. Point-SLAM内存占用极高（约12GB），不适合大规模场景

### 4.6 消融实验

#### 4.6.1 跟踪器设计 (表X和表XI)

比较**Frame-to-Map (f2m)** 跟踪（如CoSLAM、ESLAM、MonoGS）与论文提出的**Frame-to-Frame (f2f)** 跟踪：

**在模糊数据集上**：
| 方法 | ArchViz-1 ATE | ArchViz-1 PSNR | ArchViz-1 SSIM |
|------|---------------|----------------|----------------|
| CoSLAM (f2m) | 5.28 | 22.97 | 0.841 |
| CoSLAM-blur (f2m) | 0.87 | 25.08 | 0.904 |
| **CoSLAM-blur-ours (f2f)** | **0.61** | **25.43** | **0.922** |

**关键结论**：
1. 引入模糊模型改善性能（CoSLAM → CoSLAM-blur）
2. 使用论文的f2f跟踪器进一步提升（CoSLAM-blur → CoSLAM-blur-ours）

#### 4.6.2 虚拟图像数量 (表XII)

测试插值虚拟图像数量 $n$ 的影响：

| n | ArchViz-1 ATE | Mapping time | PSNR | SSIM |
|---|---------------|--------------|------|------|
| 7 | 1.024 cm | 5.931 ms | 27.89 | 0.948 |
| 9 | 0.909 cm | 6.730 ms | 27.85 | 0.947 |
| 11 | 0.771 cm | 7.899 ms | 28.29 | 0.953 |
| **13** | **0.749 cm** | **8.793 ms** | **28.45** | **0.956** |
| 15 | 0.714 cm | 9.918 ms | 28.49 | 0.955 |
| 17 | 0.723 cm | 10.516 ms | 28.37 | 0.955 |

**关键结论**：
1. 性能在n=13时开始饱和
2. 映射时间随n线性增长
3. 权衡后选择n=13作为默认值

## 五、技术细节与实现

### 5.1 CUDA优化跟踪器

论文的关键创新之一是使用**纯CUDA实现的跟踪器**，使得：

```
传统NeRF-based SLAM跟踪: 5000-7000ms/帧
MBA-SLAM跟踪 (CUDA优化): 34ms/帧 → 约200倍加速
```

### 5.2 NeRF vs 3DGS选择

| 维度 | NeRF版本 | 3DGS版本 |
|------|---------|----------|
| **速度** | 22.57 FPS (Ours-NeRF) | 1.97 FPS (Ours-GS) |
| **渲染质量** | 较低 | 更高 |
| **内存占用** | 27.12 MB | 242.08 MB |
| **适用场景** | 实时性要求高 | 高保真重建 |

### 5.3 超参数设置

- **虚拟图像数量**: NeRF版本使用7个，3DGS版本使用13个
- **正则化系数**: $\lambda_c, \lambda_d$ 等
- **Scale正则化**: $r = 1.0$
- **Patch大小**: 9×9

## 六、总结与未来方向

### 6.1 主要贡献

1. **首次将运动模糊物理模型完整集成到NeRF/3DGS-SLAM系统中**
2. **提出双端点SE(3)轨迹参数化**，有效建模曝光期间的相机运动
3. **CUDA优化跟踪器**实现近200倍加速
4. **在多个数据集上取得SOTA性能**，包括合成模糊、真实模糊和标准清晰数据
5. **提供两种实现**：基于NeRF的快速版本和基于3DGS的高质量版本

### 6.2 局限性与未来工作

1. **当前假设线性轨迹插值**，未来可探索高阶样条插值
2. **3DGS版本速度仍较慢**（1.97 FPS），需进一步优化
3. **仅考虑单帧曝光内的运动**，未处理帧间运动的不连续性
4. **未探索动态场景**下的运动模糊问题

### 6.3 潜在应用场景

1. **低光AR/VR应用**：室内增强现实需要长时间曝光
2. **夜间自动驾驶**：低光环境下的SLAM
3. **手持设备拍摄**：手部抖动导致的运动模糊
4. **机器人导航**：动态环境中的鲁棒定位

## 七、参考链接

- **论文原文**: https://arxiv.org/html/2411.08279
- **代码仓库**: https://github.com/WU-CVGL/MBA-SLAM
- **相关工作**:
  - NeRF: https://arxiv.org/abs/2003.08934
  - 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
  - MBA-VO (ICCV 2021 Oral): https://arxiv.org/abs/2108.08569
  - BAD-NeRF (CVPR 2023): https://arxiv.org/abs/2304.06536
  - BAD-Gaussians (ECCV 2024): https://arxiv.org/abs/2403.11831

---

这篇论文为运动模糊环境下的SLAM提供了系统性的解决方案，其核心思想是**物理可建模**而非纯粹的**端到端学习**，这使得方法更加鲁棒和可解释。