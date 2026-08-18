---
source_pdf: Classification of Vision-Based Tactile Sensors.pdf
paper_sha256: 6e2c09fd6cdb3717ab00c6a98bce9b76e623c99d495ad708e81fab4af1286586
processed_at: '2026-08-18T03:33:31-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 VBTS 综述

## 一句话总结

机器人手指头里塞个 camera，让手指"看见"自己摸到了啥——这就是 Vision-Based Tactile Sensor (VBTS)。这篇 paper 把市面上五花八门的 VBTS 整理成一张分类表，让新人能快速搞懂这个领域在玩什么。

---

## VBTS 到底是个啥？

想象一下你闭着眼睛摸一个东西。你怎么知道它是什么？你的 fingertip 皮肤被压变形，皮肤里的 mechanoreceptor 把变形信号传给大脑，大脑解读成"哦这是个苹果"。

VBTS 干的就是同样的事儿，只不过把皮肤换成一个 gel pad，把 mechanoreceptor 换成一个朝内看的小 camera。

关键 insight：**camera 看的不是外面的物体，是 gel 内部的变形**。gel 就像机器人手指的"皮肤"，camera 就像埋在皮肤下面的"视网膜"。外面摸到啥，gel 内部就会产生对应的 pattern，camera 拍下来给 AI 解读。

这和普通 camera 的根本区别在于——普通 camera 向外看世界，VBTS 的 camera 向内看自己的"皮肤"。

---

## 为什么需要 VBTS？

传统触觉 sensor 用的是 pressure sensor array（压阻、压容那种），像个网格一样排列。问题是：

- **分辨率太低**：一个 1cm×1cm 的 chip 上最多塞几十个 sensing unit，间距 mm 级，摸个螺丝帽都感觉不出螺纹
- **改设计麻烦**：想换个 shape 或 size，整个电路板得重新 layout、重新流片
- **多模态难**：想同时测力和温度，得堆不同的 sensing material，制造工艺复杂

VBTS 的思路完全不同：**用一个 high-res camera 替代几百上千个 sensing unit**。camera 的 pixel 数量轻松上百万，分辨率直接碾压传统方案。而且 gel 是 silicone 灌出来的，换个模具就改 design，3D 打印甚至能一次成型。

代价是：camera 体积大、frame rate 慢、数据量大、处理算法复杂。但权衡下来，对于机器人 manipulation 这种需要 high spatial resolution 的场景，VBTS 是目前 best 的选择。

---

## 这篇 paper 的核心 contribution：四大分类

作者把 VBTS 按 **"contact 怎么变成 image"** 分成两大类四小类。

### 大类一：Marker-Based Transduction (MBT)

思路：在 gel 里埋一些 marker（小点点），摸东西时 marker 会跟着挪位置，camera 看 marker 挪了多少就知道力有多大、方向往哪。

**1. Simple Marker-Based (SMB)**
最直接的方案。gel 里埋一层或两层 marker dot array，通常黑底白点或红蓝双色。摸上去 marker 随 gel 变形而位移，用 optical flow 或 blob detection 追踪。

经典案例：
- **GelForce** (2005)：最早期的 design，红蓝双层 marker。外层 marker 响应 shear，内层 marker 响应 normal force。
- **Soft-Bubble** (TRI, 2019)：高密度随机 marker + dense optical flow，Toyota Research Institute 给 in-hand manipulation 用的。
- **Tac3D**：用 stereo camera 看 marker 3D position。
- **ChromaTouch**：红蓝双层 marker 重叠时颜色混合，hue 偏移编码 depth 信息，很巧妙地用 color 代替 stereo。

直觉：SMB 像在果冻里撒一把芝麻，摸果冻时你看芝麻挪动了多少。

**2. Morphological Marker-Based (MMB)**
在 SMB 基础上加"形态结构"——pin、whisker、fingerprint ridge 等生物启发结构，把 sensitivity 放大。

经典案例：
- **TacTip** (Bristol, 2009)：black soft skin 内表面粘一堆 pin，pin 末端有白色 marker。皮肤被按下时 pin 像杠杆一样把位移放大 5-10 倍，camera 看到的 marker 移动就远大于实际 skin 变形。这就像 Pacinian corpuscle 的层状结构机械放大振动。
- **TacTip-Fingerprint**：表面加仿人类指纹 ridge，专门检测 incipient slip。指纹 ridge 在 shear 力下产生 stick-slip 微振动，正好落在 RA2 receptor 敏感频段。
- **MultiTip**：marker 涂热致变色材料，颜色随温度变化，单个 marker 同时编码 position 和 temperature。
- **BioTacTip**：4 个 cover tips 包围每个 marker，非接触时 marker 隐藏，接触时 cover tip 下压、marker 显露。露多少正比于 contact force，直接把 force estimation 变成 area measurement，计算超简单。

直觉：MMB 像 SMB + 机械 pre-amplifier。部分计算在 hardware 层面做掉了，类似 retina 在 photoreceptor 后面就做了 lateral inhibition。

### 大类二：Intensity-Based Transduction (IBT)

思路：不用 marker，直接看 gel 内部光强分布的变化。摸东西时 gel 表面被压出凹坑，内部光照反射的 pattern 变了，camera 看 intensity map 就能 reconstruct 接触面的 3D shape 和 texture。

**3. Reflective Layer-Based (RLB)**

最成功的 VBTS 范式。gel 表面涂一层 reflective paint（像镜面但带漫反射），内部用 RGB LED 从三个不同方向打光。camera 拍下来的 RGB image 里，每个 pixel 的 (R,G,B) 值直接编码了该点的表面法向量。

经典案例：
- **GelSight** (MIT Adelson, 2009)：开山之作。Photometric stereo 算法从 RGB shading 反推 surface normal，再积分出 depth map。能 reconstruct 微米级 texture。
- **DIGIT** (Meta AI, 2020)：GelSight 的 miniaturized 版本，小到能装进 robot fingertip。
- **GelSlim** (MIT, 2018)：把 LED 换成 internal light guide，form factor 更薄。
- **GelTip** (Luo, 2020)：gel 做成圆柱形 3D surface，能从多个方向感知接触。
- **Insight** (Martius, 2022)：thumb-like 锥形 shape，omnidirectional sensing。
- **DenseTact** (2022)：hemispherical + fisheye lens。
- **9DTact** (2023)：用 grayscale 替代 RGB，简化 fabrication。

直觉：RLB 就是用 camera + 多光源 reconstruct 一个小区域的 3D 几何，跟 computer vision 里的 photometric stereo 一模一样，只不过观测对象从"外面的大物体"变成"gel 表面的凹坑"。

**4. Transparent Layer-Based (TLB)**

gel 是透明的，camera 既看到内部触觉变化，也透过 gel 看到外面物体的颜色和 texture。这是多模态融合方案，但同时引入一个麻烦——两种信息混在一张图里，得分开。

经典案例：
- **TIRgel** (2023)：利用 total internal reflection。非接触时 gel 内表面全反射，接触时全反射条件被破坏，光从接触点泄漏形成 bright spot。这是 frustrated TIR 原理，跟指纹识别器一样。通过调焦距在 tactile 和 visual 两种 mode 间切换。
- **ViTac** (2018)：透明 gel 接触时局部曲率改变，折射 pattern 变化来 highlight contact。

直觉：TLB 像 GelSight + 一块透明玻璃，你想看啥就调光。但 transparent 带来的代价是 ambient light 会进来捣乱。

---

## 组合类型：各取所长

很多新 design 把上面几种 mechanism 拼起来，兼得好处：

**SMB + RLB**：在 GelSight 的 reflective layer 里埋 marker。RLB 给你 high-res texture 和 3D shape，marker 给你 shear force 感知。
- **GelSlim 3.0**、**GelStereo**、**F-Touch**、**UVtac**
- UVtac 特别巧妙：用 UV fluorescent marker + UV LED 切换。UV on 时 marker 发荧光（tactile shear mode），UV off + white on 时看 reflective texture。hardware-level modality switching，避免软件分离的 noise。

**SMB + TLB**：transparent skin 里埋 marker。
- **FingerVision** (CMU, Atkeson)：黑点 marker on transparent surface，最早最经典的 design。
- **SpecTac**：类似 UVtac 思路，UV 控制 marker 可见性。
- **MagicTac**：用 refraction pattern 被动 highlight contact zone，不用 active light switching。

**MMB + TLB**：transparent skin + morphological structure。
- **ViTacTip**：TacTip 的 transparent 变体，用 GAN 做图像分离，从 mixed image 中提取纯 tactile marker image。
- **FingerVision with Whiskers**：whisker 既作 marker 又作 contact medium。

**RLB + TLB**：reflective + transparent 双模态。
- **STS** (Hogan, 2021)：表面涂 half-silvered coating（半透半反膜）。内部灯开时光被反射回来（opaque, tactile mode），内部灯关时 external 光透过（transparent, visual mode）。就像 LCOS projector 的光路设计。
- **StereoTac**：STS + stereo camera，能重建 3D shape。
- **VisTac**：< 200ms 切换 transparency。
- **PolyTouch** (Adelson, 2025)：不做切换，fisheye FoV 分三区，中间 reflective 两侧 transparent，spatial multiplexing，同时拿两种 modality。还用 diffusion policy 做 multimodal manipulation。

---

## 数据处理方法一览

### Analytic 方法（不用 AI，用数学公式）

- **Marker tracking**：blob detection (LoG/DoG)、optical flow (Lucas-Kanade) 追踪 marker 位移
- **Voronoi tessellation**：把 marker 当点，画 Voronoi cell，cell 面积变化反映 local strain
- **Gaussian kernel density**：marker 位移做 KDE，peak 位置是 contact center，积分是 force magnitude
- **SSIM**：当前 tactile image 和 reference 未变形 image 算结构相似度，$1 - \text{SSIM}$ 作为整体 deformation 量度，threshold 一下就能检测 contact
- **Photometric stereo**：GelSight 的核心算法，RGB shading 反推法向量再积分 depth
- **Physical model**：FEM 仿真 elastomer 变形，反演 force

Analytic 的好处：interpretability 强，换 sensor 不用重新训，参数能调。坏处：精度上限低，复杂场景搞不定。

### Data-Driven 方法（AI 黑盒）

- **Image-based CNN**：直接拿 tactile image 喂 CNN，输出 force / slip / pose / classification。最通用但最贵。
- **Marker displacement-based GNN**：marker 位置当 graph node，GNN 比 CNN 高效，但可能丢失 image 中的非 marker 信息。
- **ResNet for force distribution** (Insight)：输出 spatial force map 而非单点 force vector。
- **Transformer for temporal sequence**：捕捉 grasp 过程中多帧的 temporal pattern。
- **GAN for modality separation** (ViTacTip)：从 mixed image 提取纯 tactile image。
- **Diffusion policy** (PolyTouch)：学习 multimodal action distribution，适合 contact-rich manipulation。
- **Transferable Tactile Transformer (T³)** (Adelson, 2024)：sensor-specific encoder + shared trunk + task-specific decoder，目标是跨 sensor 通用 representation。

Data-driven 的好处：精度高，能学复杂 pattern。坏处：吃数据、吃 GPU、black box、换 sensor 就废。

直觉：现在趋势是 data-driven 主导但 analytic 仍有价值。未来 sweet spot 可能是 physics-informed neural network——用 analytic 当 inductive bias，data-driven 当 refinement。

---

## 现在面临的麻烦

### Hardware 端

1. **手工制造，每个 sensor 都不一样**：gel 灌模、reflective layer 喷漆、marker 点涂全是手工，batch variation 5-15%。AI 在一个 sensor 上训好，换一个同款 sensor 就掉点。这是大规模部署的杀手。
2. **不耐用**：silicone 会老化变硬，几个月 sensitivity 就衰减。
3. **体积还是太大**：camera module 撑得整个 fingertip 比人手指粗一倍，塞进 Allegro hand 这种多指手已经很挤了，再要加 tendon drive 机构几乎没空间。
4. **没法公平比较**：不同 sensor 测的指标定义不一样、实验 setup 不一样，谁好谁坏说不清。
5. **设计空间还没探索完**：作者提到 1984 年 Mott et al. 提出过用 CCD camera 看 deformable membrane 的 internal refraction pattern，这个 mechanism 后 40 年没人重做，可能还有其他被遗忘的 design。

### Software 端

1. **帧率慢**：30-90 FPS，比传统 pressure sensor 的 1 kHz 慢一个数量级。slip detection 这种需要快速响应的任务吃亏。Event camera (NeuroTac, EveTac) 能 $\mu$s 级响应，但 size 仍大。
2. **跨 sensor 泛化差**：不同 VBTS 的 image format 差太多，universal model 难训。T³ 是早期尝试。
3. **算力贵**：data-driven 方法吃 GPU，对 embodied AI 系统不友好。
4. **Sim-to-real gap 大**：physics engine 模不准 silicone 大变形、marker adhesion、deformed medium 里的 ray tracing。Tactile Gym 是早期尝试，fidelity 有限。

---

## 为什么这事儿重要？给 Karpathy 的视角

作为搞 deep learning 的人看这个领域，几个 intuition：

**1. VBTS 是 embodied AI 的 missing piece**

LLM 给了 reasoning，vision model 给了 perception，但机器人要灵巧操作（in-hand manipulation、tool use、contact-rich task），光看不够，得摸。OpenAI 当年搞 Dactyl 解 Rubik's cube，为啥要给 Shadow Hand 加触觉？因为 vision 看不到指尖被 cube 遮挡的部分，得靠 touch 反馈 grasp stability。

没有 high-res tactile feedback，机器人就是"戴厚手套干活"——能干粗活，干不了细活。VBTS 是目前最 promising 的 high-res 触觉方案。

**2. 触觉数据比视觉数据稀缺得多**

ImageNet 有上亿张图，触觉数据集撑死几万帧。为啥？采集贵，得有 robot + sensor + physical object，没法像 web image 那样 crowdsource。这造成 sim-to-real 在触觉领域更关键——你得在仿真里生成大量 synthetic tactile image 训 model，再迁移到真实 sensor。但 sim 的 fidelity 又不够，死循环。

这可能是下一个 decade 的关键 bottleneck。如果有人能搞出 high-fidelity tactile simulator（像 MuJoCo 之于 rigid body dynamics），impact 巨大。

**3. Multimodal fusion 的物理层级**

VBTS 内部就在做 multimodal fusion——TLB sensor 一张图里既有 tactile 又有 visual。这和 LLM 的 multimodal（image + text token concatenation）很不一样。VBTS 是 hardware-level fusion，信息在物理层面就混在一起，得用 GAN 之类的方法在 software 层面拆开。

这对 model architecture 有启发：也许未来的 multimodal model 不应该简单 concat 不同 modality 的 embedding，而应该考虑 modality 在物理感知层面的 entanglement。

**4. Morphological computation 的回归**

MMB（TacTip 系列）的核心 insight 是：与其在 software 里做 amplification/filtering，不如在 hardware 层面用 mechanical structure 做。pin 的杠杆放大、fingerprint 的 stick-slip 共振、cover tip 的 visibility gating，都是把 computation 卸载到 physical structure。

这和 neuromorphic computing、event camera 是一个哲学流派——**不要把 sensor 当成 passive signal collector，让 sensor 本身就 pre-process 信号**。对 embodied AI 来说，这能降低对 model 计算量的需求，更接近生物系统的 efficiency。

**5. 设计空间的 exploration 还在早期**

这篇 paper 的 Figure 1 classification 看似完整，但作者自己承认还有 unexplored region（1984 年的 internal refraction mechanism 没人重做）。这就像 2012 年 ImageNet 之前的 CNN——架构可能性远未被 explore 完。VBTS 领域可能还有 AlexNet moment 没到来。

---

## Web Links for Further Reading

**主要 sensor 项目主页：**
- GelSight (MIT): http://gelsight.mit.edu/
- DIGIT (Meta): https://ai.meta.com/blog/digit-a-tactile-sensor-for-robot-hands/
- TacTip (Bristol): https://www.bristolroboticslab.com/tactile-robotics
- Soft-Bubble (TRI): https://github.com/RobotLocomotion/Soft-Bubble
- Insight (MPI): https://sites.google.com/view/insight-sensor
- 9DTact: https://github.com/lin-colin/9DTact
- PolyTouch: https://arxiv.org/abs/2504.19341

**Simulator 和 dataset：**
- Tactile Gym 2.0: https://github.com/robotics-ncl/tactile_gym
- TACTO (Meta): https://github.com/facebookresearch/tacto
- TacSL dataset: https://sites.google.com/view/tacsl

**关键综述：**
- Lepora 2021 TacTip review: https://ieeexplore.ieee.org/document/9442962
- Zhang 2022 hardware review: https://ieeexplore.ieee.org/document/9887448
- Abad 2020 GelSight review: https://ieeexplore.ieee.org/document/9036448

**Event-based tactile：**
- NeuroTac: https://ieeexplore.ieee.org/document/9196756
- EveTac: https://arxiv.org/abs/2403.20196

**Transformer for tactile：**
- Transferable Tactile Transformer: https://arxiv.org/abs/2406.13640

**Real-world application：**
- OpenAI Dactyl (Rubik's cube with Shadow Hand + tactile): https://openai.com/research/solving-rubiks-cube
- AnyRotate (Bristol, sim-to-real in-hand rotation): https://proceedings.mlr.press/v226/yang24d.html
- General in-hand rotation (Stanford + Meta): https://proceedings.mlr.press/v229/qi23b.html

---

## 最后的 Intuition

VBTS 现在所处的阶段，大致相当于 computer vision 在 2005-2010 的位置：hardware 在快速迭代、dataset 在积累、model 还以 CNN 为主、sim-to-real 是核心瓶颈、大规模 commercial deployment 还没到来。

这篇 paper 的价值在于把 hardware design space 梳理清楚，让后续做算法的人知道"我在哪种 sensor 上做、这种 sensor 的物理 prior 是什么、model 该怎么设计才能 exploit 这些 prior"。

下一个 decade 如果有人能解决：① high-fidelity tactile simulator ② cross-sensor 通用 model ③ 真正 human-fingertip-sized 的 VBTS hardware，机器人灵巧操作就会迎来 ImageNet moment。

---

# Vision-Based Tactile Sensors 分类综述：深度技术解读

这篇 paper 来自 Bristol Robotics Laboratory 的 Nathan Lepora 课题组（一作 Haoran Li），发表于 2024-2025 年间。核心 contribution 是提出了一个基于 **transduction principle** 的 VBTS 统一分类框架。下面我从物理原理、硬件架构、信号处理数学模型三个层面来 build intuition。

---

## 1. Motivation 与分类框架的核心思想

### 1.1 为什么需要新的分类？

之前的分类（Shimonomura 2019, Shah 2021）把 VBTS 简单分为 marker-based / reflective-based / waveguide 三类，但这种粗粒度分类无法捕捉 recent designs 中 **多机制融合** 的趋势。这篇 paper 的关键 insight 是：**分类的依据应该是 contact → tactile image 的物理 transduction 路径**，而非单纯的硬件部件。

这类似于在 computer vision 里，我们区分 camera 的依据应该是 sensor 的 photodiode 工作原理（CMOS vs CCD vs event-based），而不仅仅是镜头设计。

### 1.2 二级分类树

```
VBTS
├── Marker-Based Transduction (MBT)
│   ├── Simple Marker-Based (SMB)      — 离散 marker 点阵列
│   └── Morphological Marker-Based (MMB) — marker + 生物形态结构
└── Intensity-Based Transduction (IBT)
    ├── Reflective Layer-Based (RLB)    — 不透明反射涂层
    └── Transparent Layer-Based (TLB)   — 透明 skin，视觉+触觉融合
```

再加上组合类型：SMB+RLB, SMB+TLB, MMB+TLB, RLB+TLB，总共覆盖了 Figure 1 中的 4+4=8 种设计空间。

---

## 2. VBTS 的四模块架构解析

理想化 VBTS 由四个 module 组成：

| Module | 功能 | 典型实现 |
|--------|------|----------|
| **Contact module** | 与环境直接交互，transduction 物理界面 | silicone elastomer, 3D-printed TPU |
| **Illumination module** | 提供内部光源 | White LED, RGB LED, UV LED |
| **Perception module** | 图像采集 + onboard processing | CCD/CMOS camera, stereo camera, event camera |
| **Base module** | 结构支撑 + interconnect | 3D-printed ABS/PLA |

关键直觉：**camera 看到的不是外部物体，而是 contact module 内部的 deformation pattern**。这和普通 camera 的根本区别在于——VBTS 的 camera 是"向内看"的，类似于 retina 看 photoreceptor layer 的变形，而非直接看外部世界。

---

## 3. Marker-Based Transduction (MBT) 深度解析

### 3.1 SMB (Simple Marker-Based) 的物理原理

SMB 的 transduction chain 是：

$$\text{Contact force } \mathbf{F} \xrightarrow{\text{elastic deformation}} \text{Strain field } \epsilon(\mathbf{r}) \xrightarrow{\text{marker adhesion}} \text{Marker displacement } \mathbf{d}(\mathbf{r}) \xrightarrow{\text{camera}} \text{Tactile image}$$

#### 3.1.1 GelForce 的双层 marker 力学模型

GelForce (Kamiyama et al. 2005) 是最早期的 SMB 设计，采用红蓝双层 marker。其力-位移关系在线性弹性近似下：

$$\mathbf{F} = \mathbf{K} \cdot \mathbf{d}$$

其中：
- $\mathbf{F} \in \mathbb{R}^3$ 是 contact force vector $(F_x, F_y, F_z)$，$F_z$ 为 normal force，$F_x, F_y$ 为 shear
- $\mathbf{d} \in \mathbb{R}^{2N}$ 是 $N$ 个 marker 的 displacement 向量（每个 marker 有 $(\Delta u, \Delta v)$ 两个分量）
- $\mathbf{K} \in \mathbb{R}^{3 \times 2N}$ 是通过标定获得的 compliance matrix 的伪逆

双层 marker 的物理直觉：
- **外层 marker**（靠近 surface）主要响应 shear force，因为剪切变形在表面最大
- **内层 marker**（埋在 elastomer 深处）相对位移反映 normal compression，因为深度方向应变梯度 $\partial \epsilon_{zz}/\partial z$ 编码了 indentation depth

这和人类皮肤的 **Merkel disc (SA1, 缓慢适应, 感知 pressure)** 与 **Meissner corpuscle (RA1, 快速适应, 感知 shear/flutter)** 的分层编码有 intriguing parallel。

#### 3.1.2 Soft-Bubble 的 dense marker + optical flow

Soft-Bubble (Alspach et al. 2019, Tedrake group) 用随机分布高密度 marker，通过 dense optical flow 追踪。Lucas-Kanade 光流方程：

$$I_x(\mathbf{r}) \cdot u(\mathbf{r}) + I_y(\mathbf{r}) \cdot v(\mathbf{r}) + I_t(\mathbf{r}) = 0$$

其中：
- $I_x, I_y$ 是 tactile image 在空间方向的梯度（对 pixel coordinate $(x,y)$ 的偏导）
- $I_t$ 是时间方向梯度（帧间差分）
- $(u, v)$ 是 marker 在 image plane 上的 displacement vector

在窗口 $W$ 内最小化：
$$E(u,v) = \sum_{(x,y) \in W} [I_x u + I_y v + I_t]^2$$

对 $(u,v)$ 求导得线性方程：
$$\begin{pmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{pmatrix} \begin{pmatrix} u \\ v \end{pmatrix} = -\begin{pmatrix} \sum I_x I_t \\ \sum I_y I_t \end{pmatrix}$$

这是 structure tensor 的特征值分解，和 SIFT/Harris corner detection 共享相同的数学结构。

#### 3.1.3 ChromaTouch 的色彩混合机制

ChromaTouch (Scharff et al. 2022, Wiertlewski group) 创新点在于利用 red/blue 双层 marker 的 **hue blending**：

当两层 marker 在 image 上重叠时，HSV 空间的 hue 值 $H$ 从纯红（$H \approx 0°$）或纯蓝（$H \approx 240°$）偏移到中间紫色。hue 偏移量 $\Delta H$ 与两层 marker 的相对 vertical displacement $\Delta z$ 相关：

$$\Delta H \propto \Delta z \cdot \tan(\alpha_{view})$$

其中 $\alpha_{view}$ 是 camera 视角。这种方法把 depth 信息编码到 color space，避免了 stereo camera 的需求，类似于将 depth-from-stereo 压缩成 single-view color encoding。

#### 3.1.4 Tac3D 的 stereo vision

Tac3D (Zhang et al. 2022) 用双相机做立体视觉重建 marker 3D position。Triangulation 公式：

$$Z = \frac{f \cdot b}{d}$$

其中 $f$ 是 focal length, $b$ 是 baseline（两相机间距），$d = |x_L - x_R|$ 是 disparity。这直接借鉴了 stereo depth estimation。

### 3.2 MMB (Morphological Marker-Based) 的生物形态增强

MMB 在 SMB 基础上引入 **morphological computation**——通过物理结构设计来 pre-process 信号，类似于 retina 在 photoreceptor 到 ganglion cell 之间已经做了 edge enhancement。

#### 3.2.1 TacTip 的 micro-leverage 放大

TacTip (Chorley et al. 2009, Lepora group) 的核心设计是：在黑色 soft skin 内表面附着 pin（针状结构），pin 末端有白色 marker。当 skin 表面受力变形 $\delta_{skin}$，pin 的 tip 位移 $\delta_{tip}$ 被杠杆放大：

$$\delta_{tip} = \delta_{skin} \cdot \frac{L_{pin}}{h_{pivot}}$$

其中 $L_{pin}$ 是 pin 总长度，$h_{pivot}$ 是 pin base 到 skin 内表面的距离。这个放大比可以达到 5-10x，类似于 Pacinian corpuscle 的 layered structure 对 vibration 的机械放大。

#### 3.2.2 TacTip-Fingerprint 的 shear sensing

TacTip-Fingerprint (James et al. 2020) 在 surface 加上仿人类指纹的 ridge 结构。Ridge 的作用是当 shear force 施加时，在 ridge 之间产生 **stick-slip 微振动**，这些振动被 pin marker 捕获。这直接 mimics 人类指纹在 incipient slip detection 中的作用——指纹 ridge 的机械共振频率约 200-400 Hz，对应 RA2 (Pacinian) 的敏感频段。

#### 3.2.3 BioTacTip 的 cover-tip 机制

BioTacTip (Li et al. 2024) 设计了一个非常精巧的机制：4 个 cover tips 围绕每个 marker。非接触状态下 marker 不可见；接触时 cover tip 被压下，白色 marker tip 显露。显露程度 $\eta$ 与 contact force 线性相关：

$$\eta = \frac{A_{visible}}{A_{total}} = k \cdot F_z + b$$

其中 $A_{visible}$ 是可见 marker 面积，$A_{total}$ 是总 marker 面积，$k, b$ 是标定常数。这种方法把 force estimation 退化成了 **binary blob area measurement**，计算量极低，适合 edge deployment。

#### 3.2.4 MultiTip 的 thermochromic 多模态

MultiTip (Soter et al. 2018) 在 marker 表面涂 thermochromic material（热致变色材料）。温度变化 $\Delta T$ 改变 marker 颜色：

$$\lambda_{peak}(T) = \lambda_0 + \alpha \cdot (T - T_0)$$

其中 $\lambda_{peak}$ 是反射光谱峰值波长，$\alpha$ 是热致变色系数。这样单个 marker 同时编码了机械位移（位置）和温度（颜色），是 hardware-level multimodal fusion 的优雅实现。

---

## 4. Intensity-Based Transduction (IBT) 深度解析

### 4.1 RLB (Reflective Layer-Based) 的光学原理

RLB 的核心是 **photometric stereo**——通过多个已知方向的光源照亮同一表面，从 shading 变化反推表面法向量。

#### 4.1.1 GelSight 的 photometric stereo 算法

GelSight (Johnson & Adelson 2009, MIT) 的表面是涂有 reflective paint 的 silicone。当 RGB LED 从三个不同方向照明时，每个 pixel 的 RGB 值编码了法向量。

 Lambertian reflectance model：

$$I_c(x,y) = \rho(x,y) \cdot \mathbf{n}(x,y) \cdot \mathbf{l}_c, \quad c \in \{R, G, B\}$$

其中：
- $I_c(x,y)$ 是 channel $c$ 的 pixel intensity
- $\rho(x,y)$ 是 surface albedo（反射率，对 RGB 三通道假设相同）
- $\mathbf{n}(x,y) = (n_x, n_y, n_z) \in \mathbb{S}^2$ 是单位法向量
- $\mathbf{l}_c = (l_{cx}, l_{cy}, l_{cz})$ 是 channel $c$ 对应 LED 的光照方向向量

写成矩阵形式：
$$\underbrace{\begin{pmatrix} I_R \\ I_G \\ I_B \end{pmatrix}}_{\mathbf{I}} = \rho \underbrace{\begin{pmatrix} \mathbf{l}_R^T \\ \mathbf{l}_G^T \\ \mathbf{l}_B^T \end{pmatrix}}_{\mathbf{L}} \mathbf{n}$$

求解：
$$\rho \mathbf{n} = \mathbf{L}^{-1} \mathbf{I}$$

然后归一化得 $\mathbf{n} = \frac{\mathbf{L}^{-1}\mathbf{I}}{\|\mathbf{L}^{-1}\mathbf{I}\|}$，$\rho = \|\mathbf{L}^{-1}\mathbf{I}\|$。

从法向量场重建 depth map $z(x,y)$，利用 surface gradient：
$$\frac{\partial z}{\partial x} = -\frac{n_x}{n_z}, \quad \frac{\partial z}{\partial y} = -\frac{n_y}{n_z}$$

通过 Poisson reconstruction 或 Frankot-Chellappa 算法积分：
$$z = \mathcal{F}^{-1}\left[\frac{j\omega_x \mathcal{F}[p] + j\omega_y \mathcal{F}[q]}{\omega_x^2 + \omega_y^2}\right]$$

其中 $p = -n_x/n_z$, $q = -n_y/n_z$，$\mathcal{F}$ 是 2D Fourier transform，$(\omega_x, \omega_y)$ 是频率域坐标。

#### 4.1.2 GelSight 的 specular reflection 增强

实际的 GelSight reflective coating 不是纯 Lambertian，而是有 specular component。Phong 模型更准确：

$$I_c = \rho_d (\mathbf{n} \cdot \mathbf{l}_c) + \rho_s (\mathbf{r} \cdot \mathbf{v})^{\alpha_{sh}}$$

其中 $\rho_d, \rho_s$ 是 diffuse 和 specular 系数，$\mathbf{r}$ 是反射方向，$\mathbf{v}$ 是视角方向，$\alpha_{sh}$ 是 shininess exponent。Specular component 增强了对 **fine texture**（微米级）的敏感性，因为 specular highlight 对表面 micro-roughness 极其敏感。

#### 4.1.3 9DTact 的 grayscale photometric stereo

9DTact (Lin et al. 2023) 用 grayscale（单通道）替代 RGB，通过 **white light intensity 变化** 重建 depth。这放弃了 texture detail，但简化了 fabrication。其 depth 估计依赖于 intensity-depth 标定曲线：

$$I(x,y) = f(z(x,y))$$

其中 $f(\cdot)$ 是通过标定获得的 monotonic 映射。这本质上是 **shape-from-shading** 的单光源退化形式，需要假设 surface slope 较小。

#### 4.1.4 DenseTact 和 Insight 的 3D surface design

DenseTact (Do & Kennedy 2022) 用 hemispherical elastomer + fisheye lens，Insight (Sun et al. 2022) 用 conical (thumb-like) shape + ring RGB LED。3D surface 的好处是 omnidirectional sensing，但 photometric stereo 需要修改——光照方向 $\mathbf{l}_c$ 不再是全局常量，而是 position-dependent：

$$\mathbf{l}_c(\mathbf{r}) = \mathbf{R}(\mathbf{r}) \cdot \mathbf{l}_c^{(0)}$$

其中 $\mathbf{R}(\mathbf{r})$ 是 surface point $\mathbf{r}$ 处的局部 rotation matrix，取决于该点在 3D surface 上的位置。这增加了 calibration 复杂度，但换取了 larger sensing area。

#### 4.1.5 GelSlim 的 compact illumination design

GelSlim (Donlon et al. 2018) 的创新在于 internal light guide——用 transparent waveguide + specular reflection arrangement 替代 external LED，实现 slim form factor。光在 waveguide 内通过 total internal reflection 传播，在特定位置耦合出来照明 reflective layer。

### 4.2 TLB (Transparent Layer-Based) 的视觉-触觉融合

TLB 是这四种类型中最具创新性的。它的 skin 是 transparent/translucent 的，允许 **external light 进入**，因此 camera 同时看到：
1. 内部 deformation（tactile modality）
2. 外部物体表面（visual modality）

#### 4.2.1 TIRgel 的 Total Internal Reflection 机制

TIRgel (Zhang et al. 2023) 利用 **total internal reflection (TIR)** 物理现象。当光从光密介质（refractive index $n_1$，如 silicone $n \approx 1.4$）入射到光疏介质（$n_2$，如空气 $n \approx 1.0$）时，若入射角 $\theta > \theta_c$，发生全反射：

$$\theta_c = \arcsin\left(\frac{n_2}{n_1}\right) = \arcsin\left(\frac{1.0}{1.4}\right) \approx 45.6°$$

接触时，物体（$n_{obj} \approx 1.5$ for glass, skin）替代空气，临界角变为：
$$\theta_c' = \arcsin\left(\frac{n_{obj}}{n_1}\right) \approx \arcsin(1.07) \quad \text{(无实数解)}$$

意味着 TIR 被破坏，光在接触点泄漏，形成 bright spot。这是 **frustrated TIR** 的应用，和 fingerprint scanner 的原理相同。

TIRgel 通过调整 camera 焦距在 tactile mode（focus 在内表面 TIR pattern）和 visual mode（focus 在外部物体）间切换。

#### 4.2.2 ViTac 的 refraction-based contact highlighting

ViTac (Luo et al. 2018) 不用 TIR，而是利用 **contact-induced refraction**。当 transparent gel 表面平整时，外部光线按预期路径进入 camera；接触导致 gel 表面变形，局部曲率改变，折射角变化，在 image 上形成 contrast pattern。这类似于通过水面波纹看水底物体时的 distortion。

---

## 5. Combined Mechanisms 的设计创新

### 5.1 SMB+RLB：GelStereo, F-Touch, GelSlim 3.0, UVtac

这类传感器在 reflective layer 内嵌入 marker，同时获得：
- RLB 的高空间分辨率 texture/shape（photometric stereo）
- SMB 的 shear force 感知（marker displacement）

#### 5.1.1 UVtac 的 UV-switchable modality

UVtac (Kim et al. 2022) 用 **UV fluorescent marker** + UV LED。当 UV LED 开启时，marker 发出 visible 荧光，形成 marker image（tactile shear mode）；当 UV LED 关闭、white LED 开启时，reflective layer 的 texture image 可见。这实现了 hardware-level modality separation，避免了 software 分离的 noise。

荧光强度 $I_{fluo}$ 与 UV 激发强度 $I_{UV}$ 的关系：

$$I_{fluo} = \Phi_{QY} \cdot \epsilon \cdot c \cdot I_{UV} \cdot l$$

其中 $\Phi_{QY}$ 是 quantum yield，$\epsilon$ 是 molar absorptivity，$c$ 是荧光物质浓度，$l$ 是光程。通过 UV LED 脉冲控制，可实现 kHz 级 modality switching。

#### 5.1.2 F-Touch 和 L³F-Touch

F-Touch (Li et al. 2020) 在 GelSight-like 结构中嵌入 3 个 black marker 预测 force vector。L³F-Touch (Li et al. 2023) 进一步用 **AR tag（AprilTag）** 埋在 sensor 底部 + mirror 系统，通过 AR tag 的 6-DoF pose 估计反推 contact force：

$$\mathbf{T}_{camera}^{AR} = \mathbf{T}_{camera}^{contact} \cdot \mathbf{T}_{contact}^{AR}$$

其中 $\mathbf{T}$ 是 4×4 homogeneous transformation matrix。AR tag pose 变化 $\Delta \mathbf{T}$ 通过 stiffness matrix 映射到 force：
$$\mathbf{F} = \mathbf{K}_{6 \times 6} \cdot \Delta\mathbf{x}_{6D}$$

### 5.2 SMB+TLB：FingerVision, SpecTac, MagicTac, ViTacTip

#### 5.2.1 SpecTac 的 UV-controlled modality

SpecTac (Wang et al. 2022) 类似 UVtac 思路，但用于 transparent skin。UV LED 控制 UV fluorescent marker 的可见性，实现 visual mode（UV off, see external object）和 tactile mode（UV on, see marker displacement）切换。

#### 5.2.2 MagicTac 的 refraction-based contact highlighting

MagicTac (Fan et al. 2024) 用 soft support material 同时作为 marker 和 filler。当 external light 进入 skin 时在接触区发生 refraction，contact area 的 refraction pattern 与 non-contact 区域不同，由此 highlight contact zone。这是一种 **passive modality separation**——不需要 active light switching。

### 5.3 MMB+TLB：FingerVision with Whiskers, ViTacTip

#### 5.3.1 ViTacTip 的 GAN-based modality separation

ViTacTip (Fan et al. 2024) 是 TacTip 的 transparent-skin 变体。核心 challenge 是 tactile marker 和 visual texture 混合在同一 image 中。他们用 **GAN (Generative Adversarial Network)** 做 image-to-image translation：

Generator $G: x_{mixed} \rightarrow \hat{x}_{tactile}$

Discriminator $D: x \rightarrow \{real, fake\}$

训练目标：
$$\mathcal{L}_{GAN} = \mathbb{E}_{x_{tactile} \sim p_{data}}[\log D(x_{tactile})] + \mathbb{E}_{x_{mixed} \sim p_{mixed}}[\log(1 - D(G(x_{mixed})))]$$

加上 cycle consistency loss（CycleGAN 风格）和 L1 reconstruction loss。最终 Generator 能从混合 image 中提取纯 tactile marker image，去除 ambient light 和 object texture 的干扰。

这和 style transfer / domain adaptation 的数学结构一致，但 applied 到 sensor modality separation。

### 5.4 RLB+TLB：STS, StereoTac, VisTac, PolyTouch

#### 5.4.1 STS (Seeing Through Skin) 的 half-silvered mirror

STS (Hogan et al. 2021) 在 surface 涂 **half-silvered coating（半透半反膜）**。反射率 $R$ 和透射率 $T$ 满足 $R + T \approx 1$（忽略吸收）。

- **Internal illumination on**: internal light 被 half-silvered coating 反射回 camera（$R \approx 0.5$），external light 被 coating 阻挡（$T \approx 0.5$ 但 internal light 更强），skin 表现为 opaque → tactile mode
- **Internal illumination off**: 无 internal reflection，external light 透过 coating（$T \approx 0.5$）进入 camera → visual mode

这和 **LCOS (Liquid Crystal on Silicon) projector** 的光路设计有相似原理。

#### 5.4.2 VisTac 的 electro-optic transparency switching

VisTac (Athar et al. 2023) 用 transparency 随光强变化的 material，实现 < 200ms 的 modality 切换。

#### 5.4.3 PolyTouch 的 FoV partitioning

PolyTouch (Zhao et al. 2025, Adelson group) 不做 modality switching，而是用 fisheye camera 的 FoV 分三区：
- 中央：non-transparent reflective layer（tactile）
- 两侧：transparent optical film（visual）

这是 **spatial multiplexing** 而非 temporal multiplexing，同时获取两种 modality。PolyTouch 还用 **diffusion-based tactile policy**：

$$p_\theta(\mathbf{a}_t | \mathbf{o}_t^{vis}, \mathbf{o}_t^{tac}) = \mathcal{N}(\mu_\theta(\mathbf{o}_t^{vis}, \mathbf{o}_t^{tac}), \sigma^2 \mathbf{I})$$

其中 $\mathbf{a}_t$ 是 robot action，$\mathbf{o}_t^{vis}, \mathbf{o}_t^{tac}$ 是 visual 和 tactile observation。Diffusion policy 的 denoising process 学习 multimodal action distribution，优于 deterministic policy。

---

## 6. 数据处理方法的数学细节

### 6.1 Image Pre-processing

| 技术 | 公式/方法 | 目的 |
|------|-----------|------|
| Cropping | ROI extraction | 去除非 sensing area，加速 |
| Down-sampling | $\mathbf{I}_{ds}[i,j] = \frac{1}{k^2}\sum_{m,n} \mathbf{I}[ki+m, kj+n]$ | 降低计算量 |
| Binarization | $\mathbf{I}_{bin}(x,y) = \begin{cases} 255 & \text{if } I(x,y) > \tau \\ 0 & \text{otherwise}\end{cases}$ | marker vs background 分离 |
| Distortion calibration | $\mathbf{x}_{undist} = \mathbf{x}_{dist} + k_1 r^2 \mathbf{x}_{dist} + k_2 r^4 \mathbf{x}_{dist} + \dots$ | 校正 lens distortion |

### 6.2 Marker-Based 的 Analytic 方法

#### 6.2.1 Blob detection (LoG/DoG)

Laplacian of Gaussian：
$$\nabla^2 G_\sigma(x,y) = \frac{1}{\pi\sigma^4}\left[1 - \frac{x^2+y^2}{2\sigma^2}\right] e^{-\frac{x^2+y^2}{2\sigma^2}}$$

在 scale space 中检测 marker 中心，$\sigma$ 对应 marker 半径。

#### 6.2.2 Voronoi tessellation

给定 $N$ 个 marker 位置 $\{\mathbf{p}_i\}_{i=1}^N$，Voronoi cell $V_i$ 定义为：

$$V_i = \{\mathbf{r} \in \mathbb{R}^2 \mid \|\mathbf{r} - \mathbf{p}_i\| \leq \|\mathbf{r} - \mathbf{p}_j\|, \forall j \neq i\}$$

Cell 面积 $A_i = \text{Area}(V_i)$ 的变化反映 local strain。Contact area 可通过 $\sum_i \mathbb{1}[A_i < A_{threshold}]$ 估计。

#### 6.2.3 Gaussian kernel density map

Marker displacement $\mathbf{d}_i$ 的 density map：
$$D(\mathbf{r}) = \sum_{i=1}^N \|\mathbf{d}_i\| \cdot \mathcal{N}(\mathbf{r}; \mathbf{p}_i, \sigma^2 \mathbf{I})$$

其中 $\mathcal{N}(\mathbf{r}; \mathbf{p}_i, \sigma^2 \mathbf{I}) = \frac{1}{2\pi\sigma^2} e^{-\frac{\|\mathbf{r}-\mathbf{p}_i\|^2}{2\sigma^2}}$。$D(\mathbf{r})$ 的 peak 位置对应 contact center，integral 对应 total force magnitude。

#### 6.2.4 SSIM (Structural Similarity Index)

$$\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

变量说明：
- $\mu_x, \mu_y$：图像 $x$（当前 tactile image）和 $y$（reference 未变形 image）在 local window 内的均值
- $\sigma_x^2, \sigma_y^2$：local variance
- $\sigma_{xy}$：local covariance
- $C_1 = (k_1 L)^2$, $C_2 = (k_2 L)^2$：stability constants，$L$ 是 pixel dynamic range（通常 255），$k_1 = 0.01, k_2 = 0.03$

Dissimilarity $1 - \text{SSIM}$ 作为整体 deformation 量度，可用于 contact detection（thresholding）或 force feedback control（maintain $1 - \text{SSIM} = \text{const}$）。

### 6.3 Marker-Based 的 Data-Driven 方法

#### 6.3.1 Image-based CNN

Standard CNN pipeline:
$$\mathbf{I}_{tactile} \xrightarrow{\text{Conv+ReLU}} \xrightarrow{\text{Pool}} \xrightarrow{\text{Conv+ReLU}} \xrightarrow{\text{Pool}} \xrightarrow{\text{FC}} \hat{\mathbf{y}}$$

输出 $\hat{\mathbf{y}}$ 可以是 force vector（regression, MSE loss）、contact class（classification, CE loss）、slip label（binary classification）等。

#### 6.3.2 Marker displacement-based GNN

将 marker 作为 graph node，marker 间邻接关系作为 edge。Graph convolution：
$$\mathbf{h}_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \frac{1}{c_{ij}} \mathbf{W}^{(l)} \mathbf{h}_j^{(l)}\right)$$

其中 $\mathbf{h}_i^{(l)}$ 是 node $i$ 在 layer $l$ 的 feature，$\mathcal{N}(i)$ 是邻居集合，$c_{ij}$ 是 normalization constant，$\mathbf{W}^{(l)}$ 是可学习权重。

GNN 相比 CNN 的优势：marker 数量固定（如 TacTip 的 127 pins），graph structure 紧凑，计算量从 $O(HWC)$ 降到 $O(N^2)$，$N \sim 100$。

### 6.4 RLB 的 Data-Driven 方法

#### 6.4.1 ResNet for 3D force distribution (Insight)

Insight (Sun et al. 2022) 用 ResNet 将 conical surface 的 tactile image 映射到 3D force distribution field：

$$\hat{\mathbf{F}}(\mathbf{r}) = f_{ResNet}(\mathbf{I}_{tactile}; \theta)$$

输出是 spatial force map（每个 surface point 的 $(F_x, F_y, F_z)$），而非单一 force vector。

#### 6.4.2 Transformer for temporal tactile sequence

Recent work (Han et al. 2024) 用 Transformer 处理 tactile image sequence：

Self-attention：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q = \mathbf{X}\mathbf{W}_Q, K = \mathbf{X}\mathbf{W}_K, V = \mathbf{X}\mathbf{W}_V$，$\mathbf{X} \in \mathbb{R}^{T \times d}$ 是 $T$ 帧 tactile image 的 patch embedding 序列，$d_k$ 是 key 维度。

Transformer 能捕捉 long-term temporal dependency（如 grasp 过程中多帧 force 变化模式），这对 sequential manipulation task 很重要。

#### 6.4.3 Transferable Tactile Transformer (T³)

Zhao et al. 2024 (Adelson group) 提出 sensor-agnostic transformer：

$$\mathbf{z} = E_{sensor-specific}(\mathbf{I}_{tactile}) \xrightarrow{\text{shared trunk transformer}} \hat{\mathbf{y}}_{task-specific}$$

不同 sensor 用不同 encoder $E$（适配各自 tactile image format），但 shared trunk 学习跨 sensor 的通用 tactile representation，task-specific decoder 输出。这类似于 multilingual BERT 的思路——shared language model + language-specific tokenizer。

---

## 7. Table I 关键对比解读

| 特性 | SMB | MMB | RLB | TLB |
|------|-----|-----|-----|-----|
| Spatial resolution | Low | Middle | **High** | **High** |
| Sensitivity | Middle | **High** | Low | **High** |
| Fabrication 难度 | Easy | Medium | Hard | Hard |
| 环境光抗干扰 | 强（黑 skin） | 强 | 强（opaque） | **弱** |
| 推荐任务 | Force, slip | 3D force, slip, pose | Texture, 3D shape | Proximity, texture+color |

直觉理解：
- **SMB** 像 low-res pressure sensor array，简单但粗糙
- **MMB** 像 SMB + mechanical pre-amplifier，sensitivity 高但 design 复杂
- **RLB** 像 high-res depth camera（through photometric stereo），texture 强但 shear 弱
- **TLB** 像既看触觉又看视觉的 sensor，信息丰富但 noise 也多

---

## 8. Challenges 与 Future Directions

### 8.1 Hardware Challenges

#### 8.1.1 Manufacturing variability

手工 fabrication 导致 sensor-to-sensor variation，表现为：
- Elastomer stiffness $E$ 的 batch variation $\Delta E / E \sim 5-15\%$
- Marker position offset $\Delta \mathbf{p} \sim 0.5-2$ mm
- Reflective layer thickness variation $\Delta t \sim 10-50 \mu m$

这些 variation 导致 data-driven model 的 generalization 差。Multi-material 3D printing (如 C-sight) 是解决方向之一。

#### 8.1.2 Durability

Silicone 的 aging：shore hardness 随时间增加（$E(t) = E_0(1 + \alpha t)$），导致 sensitivity 衰减。

#### 8.1.3 Size integration

VBTS fingertip 仍大于人类 fingertip（human fingertip $\sim 15 \times 15 \times 15$ mm）。Camera module 是主要瓶颈。Alternative：
- Photodetector array（fast but low res, ~mm scale）
- Fiber bundle + remote camera
- Lensless imaging (ThinTact, Xu et al. 2025)

#### 8.1.4 Processing frequency

Camera FPS 限制：典型 30-90 FPS，远低于 piezoresistive 的 1 kHz+。Event-based camera (NeuroTac, EveTac) 可达 $\mu$s 级响应，但 size 仍大。

Event camera 输出的是 $(x, y, t, p)$ 事件流（$p \in \{-1, +1\}$ polarity），而非 frame：

$$\Delta L(x,y,t) > C \Rightarrow \text{emit event } (x, y, t, +1)$$
$$\Delta L(x,y,t) < -C \Rightarrow \text{emit event } (x, y, t, -1)$$

其中 $L = \log I$ 是 log intensity，$C$ 是 contrast threshold。这种 asynchronous spiking 输出和 biological retina 直接对应。

### 8.2 Information Processing Challenges

#### 8.2.1 Generalization across sensors

不同 VBTS 的 tactile image format 差异巨大（marker pattern vs intensity map vs mixed），universal model 难以训练。T³ (Transferable Tactile Transformer) 是早期尝试。

#### 8.2.2 Sim-to-real gap

Physics engine 难以准确模拟：
- Soft elastomer 的大变形非线性（hyperelastic Mooney-Rivlin model）
- Marker 在 elastomer 内的 adhesion 和 sliding
- Optical ray tracing in deformed medium

Tactile gym (Lin et al. 2022) 用 simplified FEM + rendering，但 fidelity 有限。Path tracing + IMPM (Shen et al. 2024) 是更高 fidelity 的尝试。

---

## 9. 与相关领域的关联（Intuition Building）

### 9.1 与 Human Somatosensory System 的对应

| VBTS 类型 | 人类对应 | 机制相似性 |
|-----------|----------|------------|
| SMB (marker array) | Merkel disc (SA1) | 离散点阵, slow adaptation, spatial encoding |
| MMB (TacTip with pin) | Meissner (RA1) + Pacinian (RA2) | mechanical amplification via structure |
| RLB (GelSight) | Ruffini ending (SA2) + Merkel | high spatial resolution, texture/edge |
| TLB (transparent skin) | multimodal (mechano-visual) | 无直接 biological 对应, novel fusion |

### 9.2 与 Computer Vision 的对应

- **Photometric stereo (RLB)** ↔ shape-from-shading in CV
- **Marker tracking (SMB)** ↔ optical flow / feature tracking
- **GAN modality separation (TLB)** ↔ image-to-image translation / domain adaptation
- **Diffusion policy (PolyTouch)** ↔ diffusion model for action generation

### 9.3 与 Embodied AI 的关联

VBTS 是 embodied agent 感知物理世界的关键 sensor。在 manipulation task 中：

$$\pi_\theta(\mathbf{a}_t | \mathbf{o}_t^{vision}, \mathbf{o}_t^{tactile}, \mathbf{h}_t)$$

Tactile observation $\mathbf{o}_t^{tactile}$ 提供 vision 无法获取的信息：
- Contact force feedback（grasp stability）
- Texture/roughness（object identification）
- Slip detection（grip force adjustment）
- 3D shape in occluded region（in-hand manipulation）

这和人类 in-hand manipulation 中 tactile feedback 的作用一致——失去 tactile sensation（如 peripheral neuropathy）的患者 grasp 能力严重退化。

---

## 10. Web Links for Reference

**主要传感器项目：**
- GelSight (MIT Adelson Lab): http://gelsight.mit.edu/
- DIGIT (Meta AI): https://ai.facebook.com/blog/digit-a-tactile-sensor-for-robot-hands/
- TacTip (Bristol Robotics Lab, Lepora): https://www.bristolroboticslab.com/tactile-robotics
- Soft-Bubble (TRI, Tedrake): https://github.com/RobotLocomotion/Soft-Bubble
- Insight (Max Planck, Martius): https://sites.google.com/view/insight-sensor
- 9DTact: https://github.com/lin-colin/9DTact
- PolyTouch (Adelson): https://arxiv.org/abs/2504.19341

**关键综述：**
- Lepora 2021 (TacTip review): https://ieeexplore.ieee.org/document/9442962
- Zhang et al. 2022 (hardware review): https://ieeexplore.ieee.org/document/9887448
- Abad & Ranasinghe 2020 (GelSight review): https://ieeexplore.ieee.org/document/9036448

**Sim-to-real 框架：**
- Tactile Gym 2.0: https://github.com/robotics-ncl/tactile_gym
- TACTO (Meta): https://github.com/facebookresearch/tacto

**数据集与 benchmark：**
- Touch and Vision dataset (Calandra): https://sites.google.com/view/visionandtouch
- TacSL dataset: https://sites.google.com/view/tacsl

**Event-based tactile:**
- NeuroTac: https://ieeexplore.ieee.org/document/9196756
- EveTac: https://arxiv.org/abs/2403.20196

**Transferable Tactile Transformer (T³):**
- https://arxiv.org/abs/2406.13640

---

## 11. 总结性 Intuition

这篇 paper 的核心贡献在于提供了一个 **transduction-principle-based taxonomy**，使得 VBTS 的设计空间变得 navigable。从 intuition 角度：

1. **MBT vs IBT** 的本质区别是信号载体——离散几何特征（marker position）vs 连续光强分布（pixel intensity）。这类似于 event-based vs frame-based camera 的哲学分歧。

2. **SMB vs MMB** 的区别在于是否利用 morphology 做 pre-computation。MMB 的 morphology 把部分计算 load 转移到 hardware，类似于 retina 的 lateral inhibition——在 photoreceptor 层级就做了 edge enhancement，减轻后续 neural processing 负担。

3. **RLB vs TLB** 的区别在于是否允许 external light 参与。RLB 是 closed optical system（高 SNR, 单 modality），TLB 是 open optical system（多 modality, 但 noise 高）。这对应到 imaging 中的 controlled lighting vs natural lighting。

4. **Combined mechanisms** 代表了 VBTS 的 future——单一 transduction principle 的 sensor 正在被 multimodal hybrid 取代。这和 smartphone camera 的 evolution 类似：从 single RGB sensor 到 multi-lens system（wide + tele + depth + LiDAR）。

5. **Data processing 的趋势** 从 analytic（photometric stereo, optical flow）向 data-driven（CNN, GAN, Transformer, Diffusion）演进，但 analytic method 在 interpretability 和 generalization 上仍有价值。未来可能的 sweet spot 是 **physics-informed neural network**——用 analytic model 作为 inductive bias，data-driven method 作为 refinement。

VBTS 领域正处于 camera-based tactile sensing 的 rapid evolution 期，类似于 2000s camera 在 computer vision 中的爆发。这篇 taxonomy paper 将帮助新进入者快速 navigate 设计空间，也为 expert 指出了 unexplored region（如 1984 Mott et al. 的 internal refraction mechanism 至今未被 modern 重新探索）。
