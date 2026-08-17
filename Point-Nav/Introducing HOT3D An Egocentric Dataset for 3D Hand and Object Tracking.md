---
source_pdf: Introducing HOT3D An Egocentric Dataset for 3D Hand and Object Tracking.pdf
paper_sha256: c2970d09572972380288ef0708fa3b650fa9c4e2db45af6e618274b28f73d4af
processed_at: '2026-08-05T10:28:08-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HOT3D 用人话讲

## 一句话版本

Meta 搞了个**超大第一人称视角数据集**，让人戴着眼镜或 VR 头显，用手抓各种东西，全程用专业动作捕捉系统记录手和物体的**精确 3D 位置**，目的是让 AI 学会看懂人怎么用手操作物体。

---

## 为什么要搞这个？

想象一下未来的 AR 眼镜场景：

- 你拿起一支笔，眼镜自动识别这是笔，还知道你想用它写字，于是把桌面变成虚拟键盘，笔就变成了输入工具
- 你组装宜家家具，眼镜看到你不知道下一步怎么装，就在你视野里叠一层提示
- 机器人看着你怎么做菜，然后学会自己做

这些场景都需要 AI 能**实时、精确地**看懂你的手在干嘛、手里抓的是什么、抓在哪个位置。但现在的方法**精度不够、速度不够**，根本撑不起这种应用。

原因之一就是**缺好的训练数据**。以前的数据集要么是第三人称视角（别人拍你），要么是单摄像头，要么是静态抓取（手放上去不动），和真实 AR/VR 场景差太远。

HOT3D 就是为了填这个坑。

---

## 两个采集设备，两种真实场景

### Project Aria — 未来的 AR 眼镜长什么样

![](https://www.projectaria.com)

Aria 是 Meta 的研究型 AR 眼镜原型，长这样：

- **重量 75 克**（比一副普通眼镜重一点，但比 GoPro 轻）
- 中间一颗 RGB 摄像头（1408×1408，fisheye 110°）
- 左右两颗 monochrome 摄像头（640×480，150° 超广角）
- 左右各一颗眼动追踪摄像头
- 两颗 IMU、气压计、磁力计

关键设计：这眼镜是给**机器看**的，不是给人看屏幕的。所以没有显示屏，全靠传感器堆料，让你戴一整天也不累。

**为什么不是单摄像头？** 因为手是高自由度 articulated object（关节物体），手指互相遮挡严重。单视角下你看不见食指被中指挡住的部分在哪。多视角就能像 stereo 一样**triangulate** 出被挡住的关节位置。

深度估计的精度有个简单公式：

$$\sigma_z \approx \frac{z^2}{f \cdot B} \cdot \sigma_d$$

- $z$：手到相机的距离（大概 0.3-0.7m）
- $f$：focal length
- $B$：baseline（摄像头间距）
- $\sigma_d$：disparity 估计误差
- $\sigma_z$：深度误差

眼镜框的 baseline 只有 10cm 左右，所以近距离手部深度误差相对大，但比单目（无穷大误差）还是好太多。

### Quest 3 — 量产 VR 头显

![](https://www.meta.com/quest/quest-3/)

Quest 3 是你能在商店买到的量产 VR 头显，卖了上百万台。HOT3D 用的内部开发版，前面两颗 monochrome 摄像头（1280×1024，30fps）记录主视野，侧面两颗用于 SLAM 但没放进数据集。

**为什么两个设备都要用？** Aria 代表未来形态（轻便眼镜），Quest 3 代表现在就能买到的产品。方法如果在两个设备上都 work，说明它**跨设备迁移能力**强，不是 overfit 到某一个 sensor setup。

---

## Ground Truth 怎么搞？贴 marker 给 OptiTrack

这是 HOT3D 最核心的技术决策。

### 传统方法的痛点

以前的 hand-object 数据集（如 HO-3D、H2O）用**marker-less** 方法：多视角 RGB-D 拍下来，跑一个 optimization 把 hand model 拟合到 depth + RGB 上。问题：

1. 慢（每帧要优化几秒到几分钟）
2. 遇到 occlusion 就 drift
3. depth sensor 精度有限（~1cm）
4. 算法本身可能错，那 GT 就是错的

### HOT3D 的方案：贴 3mm 反光小球

在专门的 motion-capture lab 里：

- 几十台红外 OptiTrack 相机围绕场地
- 每只手贴 **19 个 3mm 的 retro-reflective markers**
- 每个物体贴约 **10 个 markers**
- OptiTrack 红外光照到 marker 上反射回来，相机精准定位每个 marker 的 3D 位置
- 精度 sub-millimeter，刷新率 100+ Hz

![](https://optitrack.com)

### 从 marker 到 hand pose 的流程

hand pose 怎么从 19 个 marker 算出来？

1. **事先扫描**：每个 subject 的手用 custom 3D hand scanner 扫一遍，得到 personalized hand mesh（UmeTrack 格式）
2. **Marker registration**：把 19 个 marker 在 hand model 上的对应位置 semi-automatically 标好
3. **在线 fitting**：OptiTrack 实时给出 19 个 marker 的 3D 位置，通过 inverse kinematics 算出每个 joint 的角度

数学上是最小二乘拟合：

$$\min_{\boldsymbol{\theta}} \sum_{i=1}^{19} \left\| \mathbf{p}_i^{Opti} - \Pi_i(\boldsymbol{\theta}, \boldsymbol{\beta}) \right\|^2 + \lambda \cdot \mathcal{R}(\boldsymbol{\theta})$$

变量解释：
- $\boldsymbol{\theta}$：要解的 joint 角度向量（每根手指的弯曲、侧摆）
- $\boldsymbol{\beta}$：这个人的手形状参数（已知，从 scanner 来）
- $\mathbf{p}_i^{Opti}$：OptiTrack 测到的 marker $i$ 的 3D 位置
- $\Pi_i(\boldsymbol{\theta}, \boldsymbol{\beta})$：给定关节角度和手形，通过 forward kinematics 算出 marker $i$ **应该**在的 3D 位置
- $\mathcal{R}(\boldsymbol{\theta})$：正则项，约束关节角度不超出 anatomic limit
- $\lambda$：正则项权重

**关键 intuition**： personalized hand model 比通用 MANO model 准确度高，因为不同人手指长度差很多。如果用 mean hand shape，长手指的人 fitting 出来关节角度会 systematic bias。

### Object pose 同理但更简单

物体是 rigid body，6 DoF。用 Kabsch algorithm 闭式解：

1. 算 marker centroid（中心点）
2. 中心化
3. SVD 分解
4. 直接得到 rotation 和 translation

$$\mathbf{H} = \sum_{i=1}^{N} w_i (\mathbf{p}_i - \bar{\mathbf{p}})(\mathbf{m}_i - \bar{\mathbf{m}})^T$$
$$\mathbf{H} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$$
$$\mathbf{R} = \mathbf{V}\mathbf{U}^T, \quad \mathbf{t} = \bar{\mathbf{p}} - \mathbf{R}\bar{\mathbf{m}}$$

变量：
- $\mathbf{p}_i$：OptiTrack 测到的 marker $i$ 的位置
- $\mathbf{m}_i$：marker $i$ 在 object model 坐标系中的位置
- $\bar{\mathbf{p}}, \bar{\mathbf{m}}$：两组点的 centroid
- $w_i$：confidence weight
- $\mathbf{R}, \mathbf{t}$：从 model frame 到 scene frame 的 rotation 和 translation

**陷阱**：marker constellation 必须 geometry distinct，否则会有 pose ambiguity。比如一个完美对称的球贴 markers，旋转对称轴方向无法确定。这就是为什么 HOT3D 的 33 个物体都精心设计了 marker placement。

---

## 同时提供 UmeTrack 和 MANO 两种 hand 格式

这点很贴心。

### UmeTrack

Meta 自己的 hand model 格式，来自 SIGGRAPH Asia 2022 paper:
![](https://arxiv.org/abs/2206.00160)

- 高精度，直接从 19 markers 拟合到 personalized mesh
- 21 个 joint，每根手指 4 DoF（MCP 2 DoF + PIP 1 DoF + DIP 1 DoF）
- mesh 是从这个人的真实手 scan 来的

### MANO

社区标准，来自 Max Planck Institute:
![](https://mano.is.tue.mpg.de/)

- 参数化 model：$\boldsymbol{\beta} \in \mathbb{R}^{10}$（shape）+ $\boldsymbol{\theta} \in \mathbb{R}^{48}$（pose）
- 778 个 vertices, 1538 个 faces
- 与 SMPL family 兼容
- 几乎所有 hand pose estimation 方法都支持

**为什么两个都给？** UmeTrack 准确，是训练 label 的首选。MANO 通用，方便和其他数据集混着训练，也方便用现成方法 evaluate。**两个格式并存让用户自己选**，这是对社区的尊重。

---

## 33 个物体：PBR 材质是亮点

物体用 in-house 3D scanner 扫描，除了 high-res mesh，还带 **PBR (Physically Based Rendering) materials**：

| 贴图 | 干嘛用 |
|------|--------|
| Albedo | 漫反射颜色 |
| Metallic | 金属度（0-1） |
| Roughness | 表面粗糙度 |
| Normal | 微表面法向 |

**为什么这是大事？** 有了 PBR 就能用 Blender / Unreal / Mitsuba 渲染出 photorealistic 合成图像：

![](https://github.com/DLR-RM/BlenderProc)

合成数据 + GT pose 100% 准确 + 无限多 → 可以大规模训练 deep model，缩小 **reality gap**。

Cook-Torrance BRDF 的 specular 项长这样：

$$f_{spec}(\mathbf{l}, \mathbf{v}) = \frac{D(\mathbf{h}) \cdot F(\mathbf{v}, \mathbf{h}) \cdot G(\mathbf{l}, \mathbf{v}, \mathbf{h})}{4 (\mathbf{n} \cdot \mathbf{l})(\mathbf{n} \cdot \mathbf{v})}$$

- $\mathbf{l}, \mathbf{v}, \mathbf{h}$：light 方向、view 方向、half vector
- $\mathbf{n}$：从 normal map 采样的法向
- $D$：GGX normal distribution
- $F$：Schlick Fresnel
- $G$：Smith geometry shadowing

有了这套，光照变、背景变、相机视角变，物体渲染出来都真实。这对 domain randomization training 至关重要。

---

## 4 个场景：不只是抓起来放下

| 场景 | 做什么 | 例子 |
|------|--------|------|
| Inspection | 拿起来观察放下来 | 33 个物体全部 |
| Kitchen | 倒水、搅拌、开盖 | mug, bowl, spoon, waffle |
| Office | 写字、敲键盘、打电话 | keyboard, phone, pen |
| Living room | 看杂志、按遥控器 | magazine, remote |

每条 recording 大约 2 分钟，一共 425 条 recording。

**关键设计**：lab 里的 lighting、furniture、decoration **定期随机化**。这相当于 dataset-level domain randomization，强迫方法学到 invariants，而不是 overfit 到某一堵墙的颜色。

---

## 数据多样性

- 19 个 subject，不同 hand shape、不同国籍
- 33 个物体，不同 size、affordance、appearance
- 总移动距离 13km（有趣统计：白杯子是探险家，键盘和华夫饼几乎不动）
- 1.5M frames 中 1.16M 通过 visual inspection 全标注
- 4117 个 curated clip，每个 150 帧（5 秒）

**object orientation 的 prior**：不同物体有明显的方位偏好。bowl 总是开口朝上，birdhouse 总是正面朝人。这个统计可以用来做 Bayesian prior，在 occlusion 下推断 pose。

数学上：

$$p(\phi, \psi \mid o) \propto \exp\left(-\frac{1}{2}(\mathbf{z} - \boldsymbol{\mu}_o)^T \boldsymbol{\Sigma}_o^{-1}(\mathbf{z} - \boldsymbol{\mu}_o)\right)$$

- $\mathbf{z} = (\phi, \psi)$：观察到的 azimuth 和 elevation
- $\boldsymbol{\mu}_o, \boldsymbol{\Sigma}_o$：物体 $o$ 的均值和协方差
- 这是 Gaussian prior，可以加权到 pose estimator 的 loss 里

---

## Object Onboarding：最有实用价值的设计

HOT3D 提供两类 **onboarding sequence**，模拟 AR/VR 真实场景。

### Type 1: Static Onboarding

物体静止放在桌上，直立 + 倒立两个 sequence，所有帧都有 GT pose。

**用途**：给 NeRF / SfM 这类方法用，从多视角重建物体 3D 模型。

![](https://arxiv.org/abs/2003.08934) NeRF 公式：

$$\hat{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) dt$$

- $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$：从相机原点 $\mathbf{o}$ 沿方向 $\mathbf{d}$ 的 ray
- $\sigma(\mathbf{x})$：volume density at point $\mathbf{x}$
- $\mathbf{c}(\mathbf{x}, \mathbf{d})$：color
- $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(s) ds\right)$：transmittance

### Type 2: Dynamic Onboarding（更真实更难）

物体被手拿起翻转，**只第一帧给 GT pose**，后续帧没有。

**为什么这样设计？** 模拟真实场景：你拿到一个新物体，系统只能从第一帧学会它长什么样，之后要在你翻转、移动、遮挡的过程中持续 track。

SfM 在动态场景下完全 fail（点对应关系断了），所以这是逼方法**从单帧 reference online 学物体外观**，代表方向是 OnePose:

![](https://github.com/zxhuang7/OnePose)

---

## Aria 还送你 scene point cloud 和 eye gaze

### Scene Point Cloud (SLAM)

Aria 边走边用 photometric stereo triangulate 出场景 3D 点：

![](https://facebookresearch.github.io/projectaria_tools/)

点云 causally 增长，包括任何静止几秒的物体。**注意**：如果你拿着一个物体站着不动一会儿，那个物体也会被加进 scene point cloud，可能污染后续 SLAM。

### Eye Gaze

两条 outward ray 估计用户在看哪、聚焦多深：

$$\mathbf{r}_{L/R}(\lambda) = \mathbf{o}_{L/R} + \lambda \cdot \hat{\mathbf{d}}_{L/R}$$

- $\mathbf{o}_{L/R}$：左右眼 origin
- $\hat{\mathbf{d}}_{L/R}$：左右眼 gaze direction
- $\lambda$：depth

通过 binocular vergence 估计 focus depth：两 ray 相交点 $\lambda^*$ 即 focus。

**研究价值**：foveated sensing。用户看哪儿就在哪儿 high-res 推理，peripheral 区域 low-res 省算力。计算预算分配：

$$\text{compute}(\mathbf{x}) \propto \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}_{gaze}\|^2}{2\sigma^2}\right)$$

这能在 AR 眼镜有限算力下大幅提升效率。

---

## 和其他数据集比：赢在哪

最关键的对比：

| 数据集 | 视角 | GT 质量 | 场景真实度 | 动态抓取 |
|--------|------|---------|-----------|---------|
| HO-3D | exocentric | optimization-based | 静态 | × |
| H2O | egocentric | optimization-based | 静态为主 | × |
| DexYCB | exocentric | RGB-D optimization | pick-up 放下 | 弱 |
| HOI4D | egocentric | RGB-D optimization | 单视角 | 弱 |
| ARCTIC | mostly exo | marker-based | articulated obj | ✓ |
| ContactPose | exocentric | marker + thermal | static grasp | × |
| **HOT3D** | **egocentric** | **marker mocap** | **real headset** | **✓** |

HOT3D 集齐了四个属性：egocentric + marker-level GT + real headset + dynamic manipulation。这是第一个做到的。

---

## 怎么用这个数据集

论文里点了几条研究方向：

1. **Model-based hand tracking**：用 UmeTrack/MANO model，从 multi-view image 拟合 hand pose
2. **Model-free object tracking**：从 onboarding sequence 学物体，然后 track
3. **Joint hand-object tracking**：用 contact constraint 联合优化：
   $$\min \mathcal{L}_{visual} + \lambda_c \mathcal{L}_{contact} + \lambda_p \mathcal{L}_{penetration}$$
4. **3D object reconstruction**：用 onboarding 重建 mesh
5. **Foveated sensing**：gaze-guided 计算
6. **Cross-device transfer**：Aria 和 Quest 3 共训一个 model

ECCV 2024 的两个 challenge 在跑：

- BOP 2024: ![](https://bop.felk.cvut.cz/challenges/bop-challenge-2024/)
- Hand Tracking 2024: ![](https://handtracking.github.io/)

---

## 一图总结

```
HOT3D 设计逻辑
├── 精度来源：marker-based mocap + personalized hand model
├── 多样性来源：real headset × 19 subject × 33 object × 4 scenario × randomization
├── 实用性来源：multi-view sync + onboarding sequence + PBR materials
├── 生态来源：UmeTrack + MANO + BOP + Project Aria tools
└── 目标：让 AR/VR 真的能看懂手在干嘛
```

**核心 intuition**：marker mocap 提供 gold standard，real headset 提供真实部署 distribution，PBR + randomization 填补 sim-to-real gap，onboarding 模拟 deployment。这套组合让 HOT3D 不只是大，而是**贴近真实应用 cycle**。

想再深挖哪块（比如 UmeTrack forward kinematics 的具体 DoF 设计、BOP metric 的数学定义、或者 NeRF onboarding 的实际训练流程），告诉我就行。

---

# HOT3D: An Egocentric Dataset for 3D Hand and Object Tracking — 详细技术解析

## 1. 论文核心定位

HOT3D 是 Meta Reality Labs 发布的 **egocentric（第一人称视角）3D 手-物体 tracking** 数据集，发布于 ECCV 2024 周边挑战赛（BOP Challenge 2024 + Hand Tracking Challenge 2024）。其设计动机源于 AR/VR 应用对 hand-object interaction 的精度和速度要求远超现有 SOTA 方法所能提供的水平。论文网站：https://facebookresearch.github.io/hot3d

### 1.1 数据集规模数据表

| 维度 | 数值 | 备注 |
|------|------|------|
| Total recording time | 833 minutes | 跨 425 个 recordings |
| Multi-view frames | 1.5M | synchronized |
| Total images | 3.7M | 含 RGB + monochrome |
| Fully annotated frames | 1.16M | 通过 visual inspection |
| Subjects | 19 | 不同 hand shapes 和国籍 |
| Objects | 33 | rigid, household + office |
| Curated clips | 4117 | 每个 150 帧 (5s @ 30fps) |
| Object travel distance | 13 km | 累计移动轨迹 |
| Train split | 13 subjects, 1.0M frames | GT 公开 |
| Test split | 6 subjects, 0.5M frames | GT 仅 evaluation server 可访问 |

---

## 2. 采集设备：Project Aria 与 Quest 3

### 2.1 Project Aria（研究型 AR 眼镜）

Aria 是为 **machine perception** 而非 human consumption 设计的轻量设备，重量仅 75g（对比单台 GoPro >150g），允许 wearer 长时间佩戴进行动态活动。

**Recording profile 15 的 sensor 配置表：**

| Sensor | Type | Resolution | FPS | FOV / Rate |
|--------|------|-----------|-----|------------|
| RGB camera (center) | rolling-shutter | 1408×1408 | 30 | F-Theta fisheye, 110° |
| Mono camera L/R | global-shutter | 640×480 | 30 | F-Theta fisheye, 150° |
| Eye-tracking cameras L/R | monochrome | 320×240 | 10 | — |
| IMU (high-rate) | — | — | 1000 Hz | 6-axis |
| IMU (low-rate) | — | — | 800 Hz | 6-axis |
| Barometer | — | — | 50 Hz | pressure |
| Magnetometer | — | — | 10 Hz | 3-axis magnetic field |

注：GNSS、WiFi scanning、audio 在 HOT3D 中被 disable（隐私与场景无关）。

### 2.2 Meta Quest 3（量产 VR 头显）

使用的是 Quest 3 internal developer version。

| Sensor | Type | Resolution | FPS | Pixels Per Degree |
|--------|------|-----------|-----|--------------------|
| Front cameras L/R (用于 HOT3D) | global-shutter mono | 1280×1024 | 30 | 18 PPD |
| Side cameras L/R (未用于 HOT3D) | global-shutter mono | 1280×1024 | 30 | 18 PPD |

相机内外参通过 **ChArUco board** 标定；头显和标定板都贴上 optical markers，由 motion-capture 系统同时 tracking，从而计算出 **camera-to-headset** 刚体变换 $T_{c \leftarrow h}$。再由 headset 在 OptiTrack 坐标系中的位姿 $T_{h \leftarrow w}^{Opti}$ 得到每帧 camera-to-world：

$$T_{c \leftarrow w}^{(t)} = T_{c \leftarrow h} \cdot T_{h \leftarrow w}^{Opti,(t)}$$

其中：
- 下标 $c$ = camera, $h$ = headset, $w$ = world (OptiTrack frame)
- 上标 $(t)$ 表示 timestamp $t$

---

## 3. Machine Perception Services (MPS) on Aria

### 3.1 6 DoF Localization（VIO + SLAM）

Aria 内置 state-of-the-art **Visual-Inertial Odometry (VIO) + SLAM** 算法，提供：
- 每帧 millimeter-accurate 6 DoF poses
- 1 kHz 高频 inter-frame motion（IMU 推算）

设状态向量 $\mathbf{x}_t = [\mathbf{p}_t, \mathbf{q}_t, \mathbf{v}_t, \mathbf{b}_g, \mathbf{b}_a]$，其中：
- $\mathbf{p}_t \in \mathbb{R}^3$：position in metric, gravity-aligned frame
- $\mathbf{q}_t \in \mathbb{R}^4$：quaternion orientation
- $\mathbf{v}_t \in \mathbb{R}^3$：linear velocity
- $\mathbf{b}_g, \mathbf{b}_a \in \mathbb{R}^3$：gyroscope / accelerometer biases

通过 **tightly-coupled nonlinear optimization**（类似 OKVIS / VINS-Fusion）联合最小化视觉重投影误差和 IMU 预积分误差：

$$\mathcal{J} = \sum_t \rho_{\text{huber}}\left(\| \mathbf{e}_{\text{vision}}(t) \|^2_{\Sigma_v}\right) + \sum_t \rho_{\text{huber}}\left(\| \mathbf{e}_{\text{imu}}(t) \|^2_{\Sigma_i}\right)$$

其中 $\rho_{\text{huber}}$ 是 Huber 鲁棒核函数，$\Sigma_v, \Sigma_i$ 是对应信息矩阵。

### 3.2 7 DoF Alignment with OptiTrack

由于 MPS 输出的坐标系（gravity-aligned metric frame）与 OptiTrack 坐标系不同，需要做 **7 DoF alignment**（含 1 DoF scale $s$，1 DoF rotation $\mathbf{R}$，1 DoF translation $\mathbf{t}$ 共 7 参数）：

$$\mathbf{p}^{Opti} = s \cdot \mathbf{R} \cdot \mathbf{p}^{MPS} + \mathbf{t}$$

求解通过 Procrustes 分析或 Kabsch 算法在若干共同可见点（如 headset markers）上最小化：

$$\min_{s, \mathbf{R}, \mathbf{t}} \sum_i \| \mathbf{p}_i^{Opti} - (s \mathbf{R} \mathbf{p}_i^{MPS} + \mathbf{t}) \|^2$$

### 3.3 3D Scene Point Clouds

通过 **photometric stereo** 在两种 pair 上进行三角化：
1. Aria 移动过程中的 consecutive frames
2. Left/right SLAM camera 的 stereo pair

3D 点 $\mathbf{X}$ 通过三角化获得：

$$\mathbf{X} = \text{Triangulate}\left(\mathbf{P}_L, \mathbf{x}_L, \mathbf{P}_R, \mathbf{x}_R\right)$$

其中 $\mathbf{P}_L = \mathbf{K}_L [\mathbf{R}_L | \mathbf{t}_L]$ 是 camera projection matrix，$\mathbf{x}_L, \mathbf{x}_R$ 是对应 2D observations。Points causally added over time；任何静止数秒的物体也会被纳入点云（这是要注意的细节，hand-held 物体若停留过久会污染 scene point cloud）。

### 3.4 Eye Gaze Estimation

Gaze 表示为 **两条 outward-facing rays**，分别 anchored 在 wearer 左右眼近似位置：

$$\mathbf{r}_{L/R}(\lambda) = \mathbf{o}_{L/R} + \lambda \cdot \hat{\mathbf{d}}_{L/R}, \quad \lambda > 0$$

其中：
- $\mathbf{o}_{L/R} \in \mathbb{R}^3$：左右眼 origin（约在眼球中心）
- $\hat{\mathbf{d}}_{L/R} \in \mathbb{S}^2$：单位 gaze direction vector
- $\lambda$：沿 ray 的 depth 参数

不仅可以估计用户看的方向，还可以通过 binocular vergence 估计 **focus depth**：

$$\lambda^* = \arg\min_\lambda \| \mathbf{r}_L(\lambda) - \mathbf{r}_R(\lambda) \|^2$$

个性化校准：wearer 通过 companion app 在手机屏幕上 gaze pattern + 做 specific head movements，模型学习 wearer-specific 参数 $\theta_{gaze}$，从而降低 cross-subject 误差。

---

## 4. Ground-Truth Annotation Pipeline（基于 OptiTrack Marker）

### 4.1 Motion-Capture Lab Setup

- 数十台红外 OptiTrack exocentric 相机
- Light diffuser panels 用于 illumination variability
- 每只手贴 **19 个 3mm markers**，每个物体约 **10 个 markers**
- Markers 在 3D model 坐标系中 semi-automatically registered（与 custom 3D scanner 扫描的 mesh 对齐）

### 4.2 Object Pose Estimation

物体 pose 表示为从 model space 到 scene space 的 rigid transformation $T_{o \leftarrow w} = [\mathbf{R} | \mathbf{t}]$。给定 object 上 $N$ 个 markers 在 model 坐标系下的位置 $\{\mathbf{m}_i\}_{i=1}^{N}$ 和 OptiTrack 测量到的 scene 坐标 $\{\mathbf{p}_i\}_{i=1}^{N}$，求解 Kabsch 问题：

$$\min_{\mathbf{R} \in SO(3), \mathbf{t}} \sum_{i=1}^{N} w_i \| \mathbf{p}_i - (\mathbf{R} \mathbf{m}_i + \mathbf{t}) \|^2$$

其中 $w_i$ 是 confidence weight（基于 marker tracking quality）。闭式解：
1. 计算 centroid $\bar{\mathbf{m}}, \bar{\mathbf{p}}$
2. 中心化 $\mathbf{m}_i' = \mathbf{m}_i - \bar{\mathbf{m}}, \quad \mathbf{p}_i' = \mathbf{p}_i - \bar{\mathbf{p}}$
3. 计算 Hessian $\mathbf{H} = \sum_i w_i \mathbf{p}_i' \mathbf{m}_i'^T$
4. SVD: $\mathbf{H} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$
5. $\mathbf{R} = \mathbf{V} \mathbf{U}^T$（若 $\det(\mathbf{V}\mathbf{U}^T) < 0$ 则翻最后一个 column 符号）
6. $\mathbf{t} = \bar{\mathbf{p}} - \mathbf{R}\bar{\mathbf{m}}$

**关键约束**：每个 object 上 marker constellation 必须足够 distinct（geometry uniqueness），否则会出现 pose ambiguity。这解释了为何 paper 选择 rigid objects 而非 articulated ones（articulated markers 容易在不同 configs 下产生 confusion）。

### 4.3 Hand Pose Estimation via UmeTrack Fitting

Hand pose 通过 fitting wearer 的 **personalized UmeTrack hand model** 到 19 个 tracked markers 来计算（参考 [22] Online optical marker-based hand tracking with deep labels）。

UmeTrack hand model 参数化表示为：
$$\mathcal{H}(\boldsymbol{\theta}, \boldsymbol{\beta}) = \{ \mathbf{J}_k(\boldsymbol{\theta}, \boldsymbol{\beta}), \mathbf{V}_j(\boldsymbol{\theta}, \boldsymbol{\beta}) \}$$

其中：
- $\boldsymbol{\theta} \in \mathbb{R}^{D_\theta}$：关节角度向量（每根手指多个 DoF，包括 MCP/PIP/DIP 关节的 flexion/abduction）
- $\boldsymbol{\beta} \in \mathbb{R}^{D_\beta}$：personalized hand shape parameters（来自 3D hand scanner）
- $\mathbf{J}_k$：第 $k$ 个 joint 的 3D 位置
- $\mathbf{V}_j$：第 $j$ 个 vertex 的 3D 位置

Forward kinematics：
$$\mathbf{J}_k(\boldsymbol{\theta}, \boldsymbol{\beta}) = \left( \prod_{i \in \text{ancestors}(k)} T_i(\theta_i, \boldsymbol{\beta}) \right) \mathbf{J}_k^{rest}(\boldsymbol{\beta})$$

其中 $T_i(\theta_i, \boldsymbol{\beta})$ 是 joint $i$ 处的 local transformation（由 joint angle $\theta_i$ 和 shape $\boldsymbol{\beta}$ 决定）。

Fitting 通过最小化 marker 重投影/重定位误差：
$$\min_{\boldsymbol{\theta}} \sum_{i=1}^{19} \| \mathbf{p}_i^{Opti} - \Pi_i(\boldsymbol{\theta}) \|^2 + \lambda_{reg} \mathcal{R}(\boldsymbol{\theta})$$

其中 $\Pi_i(\boldsymbol{\theta})$ 是 marker $i$ 在 hand model 上对应位置的 forward kinematics 投影，$\mathcal{R}(\boldsymbol{\theta})$ 是 anatomic 约束正则项（joint limit, interpenetration penalty）。

### 4.4 MANO 格式同时提供

MANO（Model with ANthropometric Objects）是社区标准 hand model，源自 SMPL family。MANO 参数：

$$M(\boldsymbol{\beta}, \boldsymbol{\theta}) = W(T(\boldsymbol{\beta}, \boldsymbol{\theta}), \mathbf{J}(\boldsymbol{\beta}), \mathcal{W}, \mathbf{M})$$

变量含义：
- $\boldsymbol{\beta} \in \mathbb{R}^{10}$：shape PCA 系数（从 CAESAR scan dataset 学得）
- $\boldsymbol{\theta} \in \mathbb{R}^{48}$：pose 参数 = 3 (global rotation) + 15×3 (per-joint axis-angle) = 48
- $T(\boldsymbol{\beta}, \boldsymbol{\theta})$：blend-skinning 前的 template + pose + shape 偏移
- $\mathbf{J}(\boldsymbol{\beta})$：joint regressor（从 vertices 回归出 joint positions）
- $\mathcal{W}$：linear blend skinning weights
- $\mathbf{M}$：rest pose mesh (778 vertices, 1538 faces)

HOT3D 同时提供 UmeTrack 和 MANO 两种格式：
- **UmeTrack**：more accurate（直接由 19 markers fitting，personalized mesh from hand scanner）
- **MANO**：more standard（社区方法兼容，便于和 FreiHAND, InterHand2.6M 等数据集混合训练）

---

## 5. 3D Object Models（PBR Materials）

通过 in-house 3D scanner 获得 **high-resolution geometry + PBR (Physically Based Rendering) materials**，包含三类 map：

| Map | 作用 | 维度示例 |
|-----|------|---------|
| **Albedo/Base Color** | 漫反射颜色 | $2048 \times 2048 \times 3$ |
| **Metallic** | 金属度（0 = dielectric, 1 = metal） | $2048 \times 2048 \times 1$ |
| **Roughness** | 表面粗糙度（控制 specular lobe 宽度） | $2048 \times 2048 \times 1$ |
| **Normal map** | 微表面切线空间法向扰动 | $2048 \times 2048 \times 3$ |

PBR 渲染（Cook-Torrance BRDF）的 specular 项：

$$f_{spec}(\mathbf{l}, \mathbf{v}) = \frac{D(\mathbf{h}) F(\mathbf{v}, \mathbf{h}) G(\mathbf{l}, \mathbf{v}, \mathbf{h})}{4 (\mathbf{n} \cdot \mathbf{l}) (\mathbf{n} \cdot \mathbf{v})}$$

其中：
- $\mathbf{l}, \mathbf{v}, \mathbf{h} = \frac{\mathbf{l}+\mathbf{v}}{\|\mathbf{l}+\mathbf{v}\|}$：light, view, half-vector
- $\mathbf{n}$：从 normal map 采样的微表面法向
- $D$：Normal Distribution Function (GGX/Trowbridge-Reitz)
- $F$：Fresnel term (Schlick approximation)
- $G$：Geometry shadowing/masking (Smith)

这使得合成训练图像接近 photorealistic，缩小 **reality gap**（参考 BlenderProc：https://github.com/DLR-RM/BlenderProc）。

---

## 6. Scenarios 与数据多样性策略

### 6.1 Four Everyday Scenarios

| Scenario | Action 类型 | 典型 objects |
|----------|-------------|-------------|
| **Inspection** | pick up / observe / put down | all 33 objects |
| **Kitchen** | 倒水、搅拌、开盖 | mug, bowl, spoon, waffle, etc. |
| **Office** | 写字、敲键盘、打电话 | keyboard, phone, pen, tape |
| **Living room** | 阅读杂志、操作 remote | magazine, remote, pillow |

### 6.2 Randomization Strategy

每条 recording (~2 min) 期间定期 randomize：
1. **Lighting conditions**（光强、色温、方向）
2. **Furniture placement**
3. **Decorative elements**

这是 sim-to-real / domain randomization 的 dataset-level 实践，迫使 method 学到 invariant features 而非 overfit 到特定 background。这种 randomization 与 OpenAI ESDI/DR2 类思想一脉相承。

### 6.3 Object Travel Distance 统计（有趣发现）

总移动距离 13 km。但分布极度不均：
- **white mug**：true explorer（最长 travel distance）
- **keyboard, waffle**：mostly resting（用户坐在桌前，物体几乎不动）

这个数据可帮助：1) 衡量 class imbalance；2) prior bias 设计（推断时若检测到 mug，预测更可能 dynamic）。

---

## 7. Object Orientation Statistics（方位先验）

对每个物体，统计在数据集中观察到的 **azimuth**（绕重力轴，0°~360°）和 **elevation**（相对水平面，-90°~+90°）分布。

观察：
- **Bowl**：tends to be upright（elevation ≈ 0°, azimuth 均匀）
- **Birdhouse**：from front and upright（azimuth 集中在前方，elevation ≈ 0°）
- 这些分布可作为 pose estimator 的 prior，特别是在 occlusion 下提供约束。

数学上可以表示为：
$$p(\phi, \psi | \text{object}=o) \propto \exp\left(-\frac{1}{2}(\mathbf{z} - \boldsymbol{\mu}_o)^T \boldsymbol{\Sigma}_o^{-1} (\mathbf{z} - \boldsymbol{\mu}_o)\right)$$

其中 $\mathbf{z} = (\phi, \psi)$ 是观测方位，$\boldsymbol{\mu}_o, \boldsymbol{\Sigma}_o$ 是物体 $o$ 的均值/协方差，可在 inference 时作为 Bayesian prior。

---

## 8. Object Onboarding Sequences（关键设计）

为支持 **model-free tracking**（如 OnePose https://github.com/zxhuang7/OnePose）和 **3D object reconstruction**（如 NeRF https://arxiv.org/abs/2003.08934）：

### Type 1: Static Onboarding
- 物体 upright 和 upside-down 两个 sequence
- 全程静止，所有帧 GT pose 可得
- 适合 NeRF / SfM (Structure-from-Motion, COLMAP https://github.com/colmap/colmap) 方法
- NeRF 训练目标：
$$\mathcal{L} = \sum_{\mathbf{r}} \left\| \hat{\mathbf{C}}(\mathbf{r}) - \mathbf{C}^{gt}(\mathbf{r}) \right\|^2_2$$
其中 $\hat{\mathbf{C}}(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) dt$，$T(t) = \exp(-\int_{t_n}^{t} \sigma(s) ds)$

### Type 2: Dynamic Manipulation Onboarding
- 物体被手 manipulates，更 realistic 但 challenging
- 仅第一帧提供 GT pose（定义 canonical object space）
- 后续帧 GT pose 不提供，模拟真实场景：static setup 可用 SfM，dynamic setup 几乎不可能自动获取 GT

这模拟 AR/VR 真实场景：用户拿起一个新物体后，系统必须从单帧 reference 在 hand manipulation 过程中持续 track 6 DoF pose。

---

## 9. 评估与挑战赛设计

### 9.1 BOP Challenge 2024
https://bop.felk.cvut.cz/challenges/bop-challenge-2024/

关注 **model-free 和 model-based 2D/6D object detection**。BOP 标准 metrics：
- **AR (Average Recall)** = mean of VSD, MSSD, MSPD
- **BOP Score** = mean AR over $\tau \in \{0.05, 0.1, 0.15, \dots\}$

公式细节：
- **VSD (Visible Surface Discrepancy)**: $e_{VSD} = \bar{d}_{vis}(\hat{\mathbf{R}}, \hat{\mathbf{t}}; \mathbf{R}, \mathbf{t}, \delta, K)$
- **MSSD (Maximum Symmetry-Aware Surface Distance)**: $e_{MSSD} = \max_i \| \mathbf{x}_i^{S \cdot \hat{\mathbf{R}} \cdot \hat{\mathbf{t}}} - \mathbf{x}_i^{S \cdot \mathbf{R} \cdot \mathbf{t}} \|$
- **MSPD (Maximum Symmetry-Aware Projection Distance)**: 2D 投影对应点最大距离

### 9.2 Hand Tracking Challenge 2024
https://handtracking.github.io/

关注 **hand pose + shape estimation**。常用 metrics：
- **MPJPE (Mean Per-Joint Position Error)**: $\frac{1}{N_j} \sum_j \| \hat{\mathbf{J}}_j - \mathbf{J}_j^{gt} \|$
- **PA-MPJPE (Procrustes-Aligned)**: 先做相似变换对齐再算 MPJPE
- **APE (Absolute Position Error)**
- **EPE (Euclidean Position Error) per vertex**

---

## 10. 与现有 dataset 的对比表

| Dataset | Image # | Egocentric? | RGB-D? | Hands GT | Objects GT | Multi-view? | Devices |
|---------|---------|-------------|--------|---------|-----------|-------------|---------|
| NYU [57] | 72K | No | Yes | Yes | No | No | Real |
| FreiHAND [64] | 130K | No | No | Yes (shape) | No | No | Real |
| InterHand2.6M [38] | 1.8M | No | No | Yes | No | Yes (multi-cam) | Real |
| HO-3D [20] | 78K | No | Yes | Yes | Yes (6DoF) | Yes (exo) | RGB-D |
| H2O [33] | 572K | Yes | Yes | Yes | Yes | Yes (4 cams) | RGB-D |
| DexYCB [7] | 582K | No | Yes | Yes | Yes (6DoF) | Yes (8 exo) | RGB-D |
| HOI4D [34] | 2.4M | Yes | Yes | Yes | Yes (cat-level) | No (single view) | RGB-D |
| ContactPose [4] | 2.9M | No | Yes | Yes (thermal) | Yes (markers) | Yes | RGB-D |
| ARCTIC [13] | 2.1M | Partially (1 of 9 views) | No | Yes (markers) | Yes (articulated) | Yes (9 cams) | RGB |
| **HOT3D (Ours)** | **3.7M** | **Yes** | **No (RGB + mono)** | **Yes (UmeTrack + MANO)** | **Yes (6DoF)** | **Yes** | **Aria + Quest 3** |

### HOT3D 的 uniqueness：
1. **3.7M multi-view egocentric images** from real headsets（不是 mock-up）
2. **High-quality GT poses** for hands + objects + cameras（来自 marker-based mocap）
3. **Dynamic grasps**（不仅是 static pick-up）
4. **PBR materials** for objects（enabling photorealistic synthesis）

---

## 11. 关键技术直觉（Building Intuition）

### 11.1 为什么 egocentric + multi-view 重要？

Egocentric 视角下 hand 经常出现 **self-occlusion**（手指互相遮挡）和 **object occlusion**（被握物体遮挡部分手）。单视角方法无法可靠 track 5 个手指的全部 joints。Multi-view（如 Aria 三视角、Quest 3 双视角）在 stereo triangulation 意义下能恢复被遮挡部分。

最小可观测性：要让 3D joint $\mathbf{J}_k$ 可被 triangulated，需要至少两个视角同时看到对应 2D observation $\mathbf{x}_L, \mathbf{x}_R$，且 baseline 不能太短：

$$\sigma_{depth} \approx \frac{z^2}{fB} \sigma_{disp}$$

其中 $z$ = depth, $f$ = focal length, $B$ = baseline, $\sigma_{disp}$ = disparity 估计标准差。Aria RGB 与左右 mono 的 baseline 较小（眼镜框尺度，约 10cm），故近距离（<1m）的 hand tracking depth 精度受限于 baseline。

### 11.2 为什么需要个性化 hand model？

不同人手指长度、手掌比例差异巨大。若用 mean shape hand model，会引入 systematic bias。HOT3D 用 custom 3D hand scanner 获得每人 personalized mesh，避免 shape error 污染 pose annotation。这也是 UmeTrack 比 MANO 更 accurate 的原因——后者 shape PCA 仅 10 维，无法表达所有 anatomic variation。

### 11.3 为什么 marker-based mocap 优于 marker-less？

Marker-less annotation（如 HO-3D 用的 HOnnotate）依赖 RGB-D 多视角 optimization：
1. 依赖深度传感器精度
2. 需 expensive multi-view optimization（每帧几秒到几分钟）
3. 在手-物 occlusion 下容易 drift

Marker-based 系统（OptiTrack）：
1. sub-mm 精度
2. 实时（>100 Hz）
3. 不受 visual texture / illumination 影响
4. 唯一缺点：markers 会略微影响外观（但 3mm markers 影响很小）

### 11.4 为什么提供 PBR materials 而非简单 texture？

PBR materials 包含 metallic/roughness/normal maps，使得：
1. 在新场景、新光照下可以 photorealistic 合成训练数据
2. 支持 sim-to-real training（domain randomization on lighting while keeping geometry/pose GT）
3. Renderer-agnostic（支持 Blender, Mitsuba, Unreal 等）

参考 BlenderProc (https://github.com/DLR-RM/BlenderProc) 已成为 BOP benchmark 标准 synthesis pipeline。

### 11.5 为什么 randomize lighting / furniture / decorations？

让 trained model 对 environment 不可知因素 invariant，相当于 implicit domain randomization。在 sim-to-real 中，环境 randomization 比单纯 texture randomization 更鲁棒，因为 real-world 部署时 background 总是 unobserved。

### 11.6 为什么 dynamic onboarding 仅给第一帧 GT？

模拟 AR/VR 真实部署场景：
- 用户拿起陌生物体
- 系统从未见过的物体，只能从当前帧的 1-2 张图学习
- 后续 tracking 必须依赖 **model-free methods**（不依赖 CAD model）

第一帧 GT 用于定义 canonical object frame $\mathcal{F}_o$，后续 tracking 输出 $T_{o \leftarrow w}^{(t)}$ 是相对该 frame 的 6 DoF。这种 setting 极度接近 product deploy，是 dataset 的最大实用价值之一。

---

## 12. 可能的研究方向（论文暗示）

1. **Foveated hand tracking**：利用 eye gaze，在注视区域 high-res 推理，peripheral 区域 low-res。计算预算分配公式：

$$\text{compute}(\mathbf{x}) \propto \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}_{gaze}\|^2}{2\sigma^2}\right)$$

2. **Joint hand-object tracking**：利用 physics constraint（contact points必须吻合），formulate 为 constrained optimization：

$$\min_{\theta_t, T_t} \mathcal{L}_{visual} + \lambda_c \mathcal{L}_{contact} + \lambda_p \mathcal{L}_{penetration}$$

3. **SLAM-aware tracking**：用 Aria 提供的 scene point cloud 作为 hand-object pose 的环境约束，例如手不能穿透桌面。

4. **Cross-device transfer**：Aria 和 Quest 3 的 camera 参数差异大（FOV、resolution、mono vs RGB），训练一个 model 同时 work on both devices，需要 domain adaptation 技术。

---

## 参考链接汇总

- HOT3D 项目主页：https://facebookresearch.github.io/hot3d
- BOP Challenge 2024: https://bop.felk.cvut.cz/challenges/bop-challenge-2024/
- Hand Tracking Challenge 2024: https://handtracking.github.io/
- Project Aria: https://www.projectaria.com/
- Project Aria GitHub tools: https://github.com/facebookresearch/projectaria_tools
- Project Aria docs: https://facebookresearch.github.io/projectaria_tools/
- Meta Quest 3: https://www.meta.com/quest/quest-3/
- MANO: https://mano.is.tue.mpg.de/
- UmeTrack (SIGGRAPH Asia 2022): https://arxiv.org/abs/2206.00160
- BlenderProc: https://github.com/DLR-RM/BlenderProc
- NeRF: https://arxiv.org/abs/2003.08934
- OnePose: https://github.com/zxhuang7/OnePose
- COLMAP (SfM): https://github.com/colmap/colmap
- EgoBlur (anonymization): https://arxiv.org/abs/2311.13620
- OptiTrack: https://optitrack.com/

希望这份详细拆解能 build 出你对 egocentric hand-object tracking 数据集设计直觉的核心：**精度来源于 marker mocap + personalized models，多样性来源于真实 headset + scenario randomization，实用性来源于 model-free onboarding + multi-view synchronization**。如果你接下来想深挖某个子方向（比如 UmeTrack 的 forward kinematics 细节、BOP metric 数学定义、或 foveated sensing 的具体实现），告诉我我可以继续展开。
