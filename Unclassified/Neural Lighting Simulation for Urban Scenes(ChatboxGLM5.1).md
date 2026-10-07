## 🎯 核心问题与动机（第一性原理分析）

### 从第一性原理出发

自动驾驶中的 camera-based 感知系统本质上是一个 **从像素到3D语义的映射函数** $f_\theta: \mathcal{I} \rightarrow \mathcal{Y}$。这个映射函数在训练数据分布 $p_{\text{train}}(\mathcal{I})$ 上学习，但在部署时遇到的是 $p_{\text{deploy}}(\mathcal{I})$。当光照条件变化时：

$$p_{\text{deploy}}(\mathcal{I}_{\text{sunny}}) \neq p_{\text{train}}(\mathcal{I}_{\text{cloudy}})$$

这个 **distribution shift** 导致感知性能下降。根据第一性原理，解决此问题有两条路：

1. **Data Augmentation**: 扩大 $p_{\text{train}}$ 的覆盖范围
2. **Domain Adaptation**: 学习 domain-invariant 特征

LightSim 选择路线1——通过 simulation 生成大量不同光照条件下的训练数据。但关键约束是：**模拟数据必须足够真实**，否则引入新的 domain gap。

### 现有方法的根本缺陷

| 方法类别 | 代表工作 | 根本缺陷 |
|---------|---------|---------|
| Graphics-based simulators | CARLA, AirSim | 有限的手工 assets，domain gap 大 |
| Data-driven simulators | UniSim, SUDS, Neural Scene Graphs | **光照 baked into radiance field**，无法 relighting |
| Image-based relighting | Self-OSR, Color Transfer | 缺乏3D一致性，无物理准确的光照效果 |
| Inverse rendering | NeRF-OSR, FEGR | 不完美的 geometry/material/lighting decomposition → 伪影 |

**核心洞察**: 之前的方法要么无法 relighting（lighting baked），要么 relighting 不够真实（分解不准确）。LightSim 的关键创新是：**结合 physically-based rendering 的物理正确性与 learnable deferred rendering 的数据驱动灵活性**，取两者之长。

---

## 🏗️ 系统架构总览

LightSim 由两大模块组成：

```
┌─────────────────────────────────────────────────────┐
│                   Sensor Data                        │
│              (Multi-camera + LiDAR + GPS)            │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┴───────────────┐
        ▼                              ▼
┌───────────────────┐      ┌──────────────────────┐
│  Module 1:        │      │  Module 2:           │
│  Lighting-Aware   │      │  Neural Lighting     │
│  Digital Twins    │      │  Simulation          │
│  (Reconstruction) │      │  (Rendering)         │
└───────┬───────────┘      └──────────────────────┘
        │                              │
        │  ┌──────────────┐           │
        │  │ Geometry &    │           │
        │  │ Appearance    │           │
        │  │ (Mesh M,     │           │
        │  │  Diffuse k_d)│           │
        │  └──────┬───────┘           │
        │         │                    │
        │  ┌──────┴───────┐           │
        │  │ HDR Sky Dome  │           │
        │  │ (E_src)       │           │
        │  └──────┬───────┘           │
        │         │                    │
        ▼         ▼                    ▼
┌─────────────────────────────────────────────────────┐
│          Physically-Based Rendering Engine          │
│          (Blender) → I_render|E, I_buffer, S        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│          Neural Deferred Rendering (RelitNet)       │
│          I_tgt = RelitNet([I_src, I_buffer,         │
│                             S_src, S_tgt],           │
│                            [E_src, E_tgt])          │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
               Relit Image I_tgt
```

---

## 📐 Module 1: Building Lighting-Aware Digital Twins

### 3.1 Neural Scene Reconstruction

**目标**: 从移动平台采集的 multi-camera images 和 LiDAR point clouds 重建场景的 geometry 和 appearance。

#### 神经场表示

定义神经场 $\mathcal{F}: \mathbf{x} \rightarrow (s, \mathbf{k}_d)$，其中：

- $\mathbf{x} \in \mathbb{R}^3$: 3D 空间位置
- $s \in \mathbb{R}$: Signed Distance Function (SDF) 值
- $\mathbf{k}_d \in \mathbb{R}^3$: view-independent diffuse color（朗伯反射率）

**关键设计**: 使用 SDF 而非 occupancy/density，因为 SDF 提供更准确的几何约束，这对后续的 physically-based rendering 至关重要。

#### 组合式场景表示

场景被分解为：

$$\text{Scene} = \underbrace{B}_{\text{Static Background}} \cup \underbrace{\{A_i\}_{i=1}^{M}}_{\text{Dynamic Actors}}$$

每个部分用独立的 multi-resolution spatial feature grid [Müller et al. 2022] + MLP 头部。**这种组合式表示允许 actor 的 insertion、removal 和 manipulation**。

#### 从神经场到 Mesh

通过 Marching Cubes [Lorensen & Cline 1987] 从 SDF volume 提取 mesh，再通过 quadric mesh decimation [Garland & Heckbert 1997] 简化，得到带纹理的 mesh $\mathcal{M}$。

**材质设定**: 对所有 asset 使用 Blender Principled BSDF 的基础材质 [Burley 2012]，vertex color 作为 base color，即：

$$\mathbf{k}_d(\mathbf{x}_\text{vertex}) = f_{k_d}(\mathbf{x}_\text{vertex}; \theta)$$

其中 $f_{k_d}$ 是 appearance MLP head。

> **直觉理解**: 这里做了简化——只用 diffuse color（朗伯模型），没有 specular/roughness 等 PBR 参数。这是因为从单次 drive-by 数据中准确恢复完整的 PBR 材质是高度 ill-posed 的。**不完美的材质分解正是后续需要 neural deferred rendering 来补偿的原因之一**。

#### 训练损失

$$\mathcal{L}_\text{scene} = \mathcal{L}_\text{rgb} + \lambda_\text{lidar} \mathcal{L}_\text{lidar} + \lambda_\text{reg} \mathcal{L}_\text{reg}$$

其中：

$$\mathcal{L}_\text{rgb} = \frac{1}{|\mathcal{R}_\text{img}|} \sum_{r \in \mathcal{R}_\text{img}} \| C(r) - \hat{C}(r) \|^2$$

$$\mathcal{L}_\text{lidar} = \frac{1}{|\mathcal{R}_\text{lidar}|} \sum_{r \in \mathcal{R}_\text{lidar}} \| D(r) - \hat{D}(r) \|^2$$

- $C(r)$: ray $r$ 的观测 RGB 颜色
- $\hat{C}(r)$: 通过 volume rendering 预测的颜色
- $D(r)$ / $\hat{D}(r)$: LiDAR depth 的观测值/预测值
- $\mathcal{L}_\text{reg}$: 包含 Eikonal constraint ($\|\nabla_\mathbf{x} s(\mathbf{x})\| = 1$) 和 free-space loss

---

### 3.2 Neural Lighting Estimation

**目标**: 从传感器数据恢复 HDR sky dome $E$（环境光照），用于 physically-based rendering 和 relighting。

#### 挑战

1. **有限视场 (Limited FoV)**: SDV 上的相机不拍摄完整天空
2. **低动态范围 (LDR)**: 相机存储 8-bit LDR 图像，HDR 信息丢失
3. **太阳位置和强度**: 太阳在 HDR 中占极小角度但贡献绝大部分能量，是户外光照的主导因素

#### Pipeline（三步走）

**Step 1: Panorama Reconstruction**

给定 $K$ 个相机图像 $\mathcal{I} = \{I_i\}_{i=1}^K$ 和对应位姿 $\mathcal{P} = \{P_i\}_{i=1}^K$：

1. 从提取的 mesh $\mathcal{M}$ 渲染深度图: $D_i = \psi(\mathcal{M}, P_i)$
2. 对每个像素 $(u', v')$，用深度和投影矩阵估计3D世界坐标
3. 应用等距柱状投影 $\mathcal{E}$ 映射到全景图 $(u, v)$：

$$I_\text{pano} = \Theta(\mathcal{I}, \mathcal{D}, \mathcal{P}) = \mathcal{E}\left(\pi^{-1}(\mathcal{I}, \mathcal{D}, \mathcal{P})\right)$$

其中 $\Theta$ 是像素级变换，$\pi^{-1}$ 是反投影函数。

> **关键**: 利用 LiDAR-derived depth 将多相机图像拼接为 partial LDR panorama，天空区域 depth 设为无穷大。

**Step 2: Panorama Completion**

用 DeepFill-v2 [Yu et al. 2019] inpainting network 补全缺失的天空区域，从 partial panorama 得到完整的 360°×180° LDR panorama $L$。

训练数据: Holicity [Zhou et al. 2020]，6K panorama images。

**Step 3: HDR Sky Dome Estimation**

用 encoder-decoder 网络将 LDR panorama $L$ 提升为 HDR sky dome $E$：

$$E = \text{HDRdecoder}(z_\text{sky}, [f_\text{int}, f_\text{dir}]), \quad z_\text{sky}, f_\text{int}, f_\text{dir} = \text{LDRencoder}(L)$$

其中：
- $z_\text{sky} \in \mathbb{R}^{64}$: sky appearance latent（天空外观的潜在表示）
- $f_\text{int}$: peak sun intensity（太阳峰值强度）
- $f_\text{dir}$: sun direction（太阳方向）

**重要设计**:
- **显式编码 sun intensity 和 direction**: 这使得光照控制具有人可解释性（可手动调整太阳位置和强度）
- **GPS 融合**: 当 GPS 和时间可用时，用 GPS-derived sun direction 替代网络预测的 $f_\text{dir}$，更精确
- **双编码器架构**: 一个在有清晰太阳的 HDR 上训练，一个在所有 HDR 上训练，以更鲁棒地估计 $f_\text{int}$ 和 $z_\text{sky}$

#### 与 SOTA 光照估计的对比

| 方法 | 输入 | Angular Error |
|------|------|---------------|
| SOLDNet | Limited-FoV | 69.98° |
| NLFE | Limited-FoV | 78.29° |
| NLFE* (Panorama) | Panorama | 47.39° |
| **LightSim (no GPS)** | Panorama | **20.01°** |
| **LightSim (with GPS)** | Panorama+GPS | **3.78°** |

> **直觉理解**: 利用多相机数据得到 panorama 比单张有限视场图像信息量大得多，GPS 又提供了物理上的太阳位置约束，所以精度大幅提升。

---

## 🎨 Module 2: Neural Lighting Simulation of Dynamic Urban Scenes

### 核心思想：Hybrid Rendering

直接用 PBR 渲染器（如 Blender）可以得到物理正确的光照效果，但由于 geometry 和 material 分解不完美，结果会有模糊、边界伪影、不真实反射等问题。直接用 2D 方法（如 style transfer）则缺乏物理约束。

LightSim 的关键创新是 **Physically-based Rendering + Learnable Neural Deferred Rendering**：

```
Physical Correctness  ←———  PBR Renderer  ——————→  PBR Artifacts (blurry, mesh boundaries)
                                           ↕
         Neural Deferred Rendering compensates artifacts
                                           ↕
Data-driven Realism   ←———  Real Images    ——————→  No physics (no new shadows)
```

### 4.1 Physically-Based Rendering Pipeline

给定 **augmented reality representation** $\{\mathcal{M}', E_\text{src}, I'_\text{src}\}$（可能经过 actor insertion/removal/trajectory modification），在 PBR 渲染器（Blender）中：

1. **渲染缓冲区 (Render Buffers)** $I_\text{buffer} \in \mathbb{R}^{h \times w \times 8}$:
   - Position (3 channels)
   - Depth (1 channel) 
   - Normal (3 channels)
   - Ambient Occlusion (1 channel)

2. **源光照和目标光照下的渲染图像**: $I_\text{render}|_{E_\text{src}}$ 和 $I_\text{render}|_{E_\text{tgt}}$

3. **Shadow Ratio Map**: 
$$S = \frac{I_\text{render}}{\tilde{I}_\text{render}}$$
其中 $\tilde{I}_\text{render}$ 是 **不渲染 shadow visibility rays** 的结果。分别计算 $S_\text{src}$ 和 $S_\text{tgt}$。

> **直觉理解 Shadow Ratio**: $S$ 本质上编码了"每个像素有多少被阴影遮挡"的信息。源和目标的 shadow ratio 之比告诉我们阴影如何变化。这是对 shadow editing 进行显式控制的关键机制。

### 4.2 Neural Deferred Rendering

**输入**:
- Source image $I_\text{src}$（真实图像）
- Render buffers $I_\text{buffer}$（position, depth, normal, AO）
- Shadow ratio maps $\{S_\text{src}, S_\text{tgt}\}$
- Source & target HDR sky domes $\{E_\text{src}, E_\text{tgt}\}$

**输出**: 目标光照下的 relit image $\hat{I}_\text{tgt}$

$$I_\text{tgt} = \text{RelitNet}\left([I_\text{src}, I_\text{buffer}, S_\text{src}, S_\text{tgt}], [E_\text{src}, E_\text{tgt}]\right)$$

#### 网络架构（Fig. A11）

```
┌──────────────┐    ┌───────────────────┐
│ Image Encoder │    │  Lighting Encoder   │
│ (I_src,       │    │  (E_src, E_tgt)     │
│  I_buffer,    │    │                     │
│  S_src, S_tgt)│    │                     │
└──────┬───────┘    └──────────┬──────────┘
       │                       │
       ▼                       ▼
┌──────────────────────────────────────┐
│         Latent Fuser                  │
│  Image latent + Source lighting latent │
│  + Target lighting latent             │
│  → Concatenate → Conv → Linear        │
│  → Upsample                          │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│         Rendering Decoder             │
│         (U-Net decoder)               │
│         → Î_tgt ∈ ℝ^{H×W×3}          │
└──────────────────────────────────────┘
```

**Lighting Encoder** 处理 $E_\text{src}$ 和 $E_\text{tgt}$，将 HDR 环境图编码为潜在向量，然后通过 latent fuser 与图像特征融合。

### 4.3 训练数据策略

这是另一个关键创新。由于缺乏真实世界的 paired data（同一场景在不同光照下的图像），LightSim 设计了 **混合训练方案**：

#### 数据对类型

1. **Sim-Sim Pairs**: $I_\text{render}|_{E_\text{src}} \rightarrow I_\text{render}|_{E_\text{tgt}}$
   - 从 digital twins 用 PBR 渲染器生成同一场景在源/目标光照下的 paired images
   - 教网络学习 relighting 的物理规律

2. **Sim-Real Pairs**: $I_\text{render}|_{E_\text{src}} \rightarrow I_\text{real}$
   - 从 digital twins 渲染源光照图像 → 映射到真实图像
   - 教网络从 synthetic 到 real 的 domain transfer
   - **关键**: 使用估计的 $E_\text{src}$ 作为目标光照，因为 $I_\text{real}$ 就是在该光照下拍摄的

3. **Identity Pairs**: 当 source 和 target lighting 相同时，输出应等于输入
   - 鼓励 self-consistency

#### 训练损失

$$\mathcal{L}_\text{relight} = \frac{1}{N} \sum_{i=1}^{N} \left(\underbrace{\| I_\text{tgt}^i - \hat{I}_\text{tgt}^i \|^2}_{\mathcal{L}_\text{color}} + \lambda_\text{lpips} \underbrace{\sum_{j=1}^{M} \| V_j(I_\text{tgt}^i) - V_j(\hat{I}_\text{tgt}^i) \|^2}_{\mathcal{L}_\text{lpips}} + \lambda_\text{edge} \underbrace{\| \nabla I_\text{tgt}^i - \nabla \hat{I}_\text{tgt}^i \|^2}_{\mathcal{L}_\text{edge}} \right)$$

- $\mathcal{L}_\text{color}$: 像素级 L2 重建损失
- $\mathcal{L}_\text{lpips}$: 感知损失，$V_j$ 是预训练 VGG 网络第 $j$ 层特征
- $\mathcal{L}_\text{edge}$: 边缘保持损失，$\nabla I$ 用 Sobel-Feldman 算子近似图像梯度
- $\lambda_\text{lpips} = 1$, $\lambda_\text{edge} = 400$

> **边缘损失的重要性**: $\lambda_\text{edge} = 400$ 权重很大，确保 relit 图像保留源图像的细节结构（如车道线、文字等），同时改变光照效果。

---

## 🔬 实验结果详解

### 5.1 数据集

- **主实验**: PandaSet [Xiao et al. 2021] — 103 个 San Francisco 城市驾驶场景，6个相机（1920×1080），360° 64-beam LiDAR
- **泛化性**: nuScenes [Caesar et al. 2020] — 10 个城市场景

### 5.2 场景 Relighting 对比（Table 1）

| Method | FID↓ | KID (×10³)↓ |
|--------|------|-------------|
| Self-OSR | 124.8 | 107.1±4.3 |
| NeRF-OSR | 143.9 | 94.0±7.5 |
| Color Transfer | **85.4** | **29.5±4.3** |
| EPE | 93.0 | 56.0±5.0 |
| **Ours** | 87.1 | **30.4±4.0** |

> **重要洞察**: Color Transfer 的 FID 最低，但它只是调整全局颜色直方图，**不产生物理正确的方向性光照效果**（如新投射的阴影）。LightSim 的 FID 接近 Color Transfer，同时能产生真实的阴影、高光等效果。

### 5.3 下游感知提升（Table 2）

在 BEVFormer [Li et al. 2022] 上做 3D vehicle detection：

| Training Data | mAP (%) |
|---------------|---------|
| Real only | 32.1 |
| Real + Color aug. | 33.8 (+1.7) |
| Real + Self-OSR | 30.3 (−1.8) |
| Real + EPE | 32.5 (+0.4) |
| Real + Color Transfer | 35.1 (+3.0) |
| **Real + LightSim** | **36.6 (+4.5)** |

> LightSim 数据增强带来了 **+4.5 mAP** 的显著提升，而 Self-OSR 甚至损害了性能（因为视觉伪影导致 domain gap）。

### 5.4 Ablation Study（Table A4）

| # | sim-real | identity | I_buffer | {S_src, S_tgt} | L_edge | FID↓ | KID↓ |
|---|----------|----------|----------|-----------------|--------|------|------|
| 0 | ✓ | ✓ | ✓ | ✓ | ✓ | 55.4 | 27.6 |
| 1 | ✗ | ✓ | ✓ | ✓ | ✓ | 62.5 | 32.7 |
| 5 | ✓ | ✗ | ✓ | ✓ | ✓ | 67.1 | 40.9 |
| 6 | ✓ | ✓ | ✗ | ✗ | ✓ | 57.3 | 31.8 |
| 7 | ✓ | ✓ | ✓ | ✓ | ✗ | 49.8 | 23.1* |

> *注：没有 I_buffer 时 KID/FID 可能更低，但这不代表视觉质量更好——因为参考数据集不包含新光照下的真实图像，metrics 不能完全衡量物理正确性。

**关键结论**:
- **Sim-real pairs** 至关重要：移除后 FID 从 55.4→67.1
- **Identity pairs** 提供自洽性约束：移除后 FID 从 55.4→67.1  
- **Rendering buffers** 提供几何先验，帮助生成正确的光照效果
- **Shadow maps** 控制阴影变化
- **Edge loss** 保持细节但需要适当权重

---

## 🧩 关键设计决策的深层理解

### 为什么不直接用 PBR 渲染？

从 Fig. 9 可以看出，PBR 渲染的结果 $I_\text{render}|_{E}$ 存在以下问题：

1. **Mesh 伪影**: Marching Cubes 提取的 mesh 在树木、建筑等复杂几何处有明显的 mesh boundary
2. **模糊纹理**: Diffuse-only 重建丢失了高频细节
3. **不完美的材质**: 固定的 base material 无法表达复杂反射
4. **阴影三角形**: 不平滑的 mesh 导致随机阴影三角形

Neural deferred rendering 以源图像 $I_\text{src}$ 为基底，用 render buffers 提供光照信息，在保持源图像细节的同时修正光照效果。

### 为什么不直接用 Image-to-Image Translation？

因为纯 2D 方法缺乏 3D 一致性：
- 无法跨相机产生一致的光照效果
- 无法准确模拟新投射的阴影（需要 3D 几何信息）
- 时间一致性差

### Deferred Rendering 的直觉

传统图形学中，deferred rendering 是将渲染过程分为：
1. **Geometry pass**: 输出 position, normal, depth, AO 等 (G-buffer)
2. **Lighting pass**: 基于 G-buffer 计算每个像素的光照

LightSim 将此思想融入神经网络：
- **Geometry pass** → 由 PBR 渲染器计算 G-buffer ($I_\text{buffer}$)
- **Lighting pass** → 由 RelitNet 网络基于 G-buffer 和光照信息计算 relit image

这个设计的优雅之处在于：**物理渲染器提供了"正确的光照应该长什么样"的先验，而神经网络学习了从"不完美的渲染结果"到"真实图像"的映射**。

---

## ⚠️ 局限性

1. **Diffuse-only 重建**: 只恢复 diffuse color，没有 specular/roughness 等 PBR 参数，导致 intrinsic decomposition 不完美
2. **阴影 baked into albedo**: 神经场景重建时阴影被 bake 进 diffuse color，导致阳光条件下无法完美移除阴影（见 Fig. A26）
3. **无夜间局部光源**: 无法处理路灯、交通灯、车灯等点光源
4. **渲染效率**: 目前使用 Blender 的 ray-tracing，尚未优化实时性
5. **Geometry 质量依赖 LiDAR**: 稀疏 LiDAR（如 nuScenes 的 32-beam）会导致 mesh 质量下降，出现孔洞和随机阴影三角形

---

## 🔗 与相关工作的关系网络

| 类别 | 相关工作 | LightSim 的改进 |
|------|---------|----------------|
| Neural Simulation | UniSim [Yang et al. 2023], SUDS [Turki et al.