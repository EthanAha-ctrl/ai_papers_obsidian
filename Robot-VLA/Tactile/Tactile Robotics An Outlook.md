---
source_pdf: Tactile Robotics An Outlook.pdf
paper_sha256: ebf09674d5d00021926a8dda6ea50c131262c656d69f876431780e6973233b2f
processed_at: '2026-08-12T12:31:46-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Tactile Robotics 人话版

好，我换个讲法，把这 paper 当故事讲。

---

## 这 paper 到底在干嘛

六个这个领域的大佬坐下来写了个 manifesto，说：咱们做了十几年触觉传感器，现在该把 "Tactile Robotics" 正式立成一个 sub-field 了，别再散落着做。

为什么现在？因为过去几年几件事同时到位了：sensor 便宜了（DIGIT \$350，以前 BioTac 要 \$5k-10k）、simulator 成熟了、deep learning 能处理 tactile image 了、sim-to-real learning 在 manipulation 上 work 了。所有 ingredient 都在，就差一个统一 narrative 把它们拼起来。

---

## 为什么机器人需要触觉，vision 不够吗

Vision 给你 *global overview* — 你看到桌上有杯子，知道它大概在哪、长什么样。但你要 *grab* 这个杯子，视觉就帮不上多少忙了。你手指接触杯子的瞬间，有几件事必须靠 touch 才知道：

- 我抓得 *紧不紧*？太松掉下来，太紧捏碎；
- 杯子 *滑没滑*？slip detection 是纯 touch 的问题；
- 杯子 *多硬*？纸杯 vs 玻璃杯用力方式完全不同；
- 杯子表面 *什么 texture*？干不干？需不需要调整 grip；
- 杯子把手 *edge 在哪*？finger 要 servo 到 edge 上去。

Vision 在接触前给你全局信息，touch 在接触后给你 detail feedback。这俩在时间上是 *complementary* 的，不是 redundant 的。人脑就是这么用的 — 接触前靠眼睛，接触后 brain 自动 weight 更多到 touch [212]。

---

## Sensor 这边：五种物理原理，各打各的仗

### Piezoresistive（压阻）

材料被压变形，电阻变。最直觉的原理。公式就是 $\Delta R / R = GF \cdot \varepsilon$，$GF$ 是 gauge factor，$\varepsilon$ 是应变。代表是 NeuTouch [45] 和 NUskin [46]，后者采样率能到 4 kHz，因为他们想做 event-driven spike 输出，模仿生物触觉。

问题：hysteresis 严重，高分子材料有 memory。

### Piezoelectric（压电）

应力产生电荷。公式 $D_i = d_{ij} \sigma_j$，$d_{ij}$ 是压电系数。PVDF 这种 polymer $d_{33} \approx 20-30$ pC/N，PZT 这种 ceramic $d_{33} \approx 200-600$ pC/N。

**关键问题**：压电只对 *动态* 力敏感。DC load 下 voltage 会 leak 掉。所以压电 sensor 测不了 static pressure，但测 slip 和 texture vibration 特别好 — 这刚好是 slip detection 想要的。

### Capacitive（电容）

平行板电容 $C = \varepsilon_0 \varepsilon_r A / d$，加压力 $d$ 变小 $C$ 变大。最广泛应用的 array sensor 原理。iCub hand [47]、Allegro hand [54]、PR2 gripper [55] 都用这个。

问题：crosstalk、wiring 多、hysteresis。

### Magnetic（磁）

把小磁体埋在 elastomer 里，Hall effect sensor 测磁场变化。公式 $V_H = R_H \cdot I B / t$。外力 → gel 形变 → 磁体位移 → 磁场变 → Hall 电压变。

代表是 **uSkin** [57, 113] 和 **ReSkin** [58](CMU/Google)。ReSkin 的卖点是 *replaceable* — 用坏了撕下来换一片，因为就是柔性 PCB + magnet + elastomer，便宜。

问题：外部磁场干扰。机器人身上电机那么多，这个得小心。

### Optical（视觉触觉）— 过去十年的赢家

这跟其他四个根本不是一个 paradigm。它把 tactile 问题直接 *vision 化* 了：用一个 camera 拍 gel 的内表面，contact 时 gel 变形，camera 看到形变。

两大子类：

**GelSight 系** — 用三色 LED (R/G/B) 从不同方向照 gel 内表面，gel 表面有 reflective coating。不同法向的 surface 会反射不同 LED 的光，所以颜色 → 法向。这是 photometric stereo [183]：

$$I_c(\mathbf{x}) = \rho(\mathbf{x}) \cdot \mathbf{n}(\mathbf{x}) \cdot \mathbf{l}_c$$

三个 channel 三个方程解 $(n_x, n_y, n_z)$，然后 Poisson reconstruction 得 depth map。本质上是把 CV 的 shape-from-shading 套到一个 transparent gel 上。

代表：GelSight [28]、GelSlim [59]、DIGIT [60]、GelTip [31]、GelSight Mini、Insight [33]、Minsight [34]。Meta AI 的 DIGIT 360 [61] 是最新版。

**TacTip 系** — gel 内表面放 marker 颜料点，contact 时 marker 位移，用 optical flow 追踪。代表是 Bristol 的 TacTip [29]。

为什么 optical 赢了？因为它直接给你一张 image，所有 CV 工具链 (CNN, SIFT, optical flow) 都能用。Spatial resolution 也是 pixel level（百微米级），比 array sensor 高一个数量级。

代价：30+ fps video 数据量大，latency 高。所以下一个十年可能向 event-driven optical sensor 转 [71]。

---

## 人皮的 hyperacuity 这个概念很重要

人皮的 mechanoreceptor 物理间距是 X，但 perceptual acuity 可以到 X/10。怎么做到的？因为相邻 receptor 的 receptive field 重叠，多个 receptor 信号在 brain 里加权融合，sub-cellular resolution 就出来了 [105]。

这个对 robotic skin 设计直接启发：你不一定要密密麻麻铺 sensor，receptive field overlap + central processing 可以给你超 resolution。Lepora et al. [106] 在 TacTip 上实现了 tactile superresolution。

---

## Sim-to-real 这边

Tactile sensor 模拟有三大类方法：

**Physics-based** — 用 FEM 算 gel 形变，公式 $\mathbf{K} \mathbf{u} = \mathbf{F}$，$\mathbf{K}$ 是刚度矩阵。精确但慢得要死。GPU 加速的代表是 Narang et al. [128] 用 NVIDIA FleX。GelSight 的光学部分用 Phong reflection model 渲染：

$$I = k_a I_a + k_d (\mathbf{L}\cdot\mathbf{N}) I_d + k_s (\mathbf{R}\cdot\mathbf{V})^\alpha I_s$$

**Data-driven** — GAN / CNN 学 depth map → tactile image 映射。Church et al. [137] 用 GAN 把 robot simulator 的 depth map 翻译成 TacTip marker pattern，绕过精确建模。这条路简单，sim-to-real manipulation 已经做出来了 [138, 139]。

**Model-based** — 参数化轻量模型 + 小数据标定。Si & Yuan 的 Taxim [141] 用 polynomial lookup table 拟合 GelSight shape-to-color mapping，速度比 FEM 快几个量级。

**核心 dilemma**：高保真 simulator 真的必要吗？答案是不一定。basic physics + 好 domain transfer 也能 zero-shot。真正没解决的是 **shear force / friction 模拟**，大多数 simulator 就用 Coulomb friction 凑合，slip 的时候全靠 domain randomization 兜底。

参考：TACTO https://github.com/facebookresearch/tacto、Taxim https://github.com/ARISE-Initiative/taxim、Tactile Gym https://github.com/ac-93/tactile_gym

---

## 数据集这边 — benchmark 很难做

Vision benchmark 都是 passive 的 — 拍一个 dataset，所有人拿来跑。Tactile 做不到，因为 tactile 是 *active* 的：必须 robot 在动、policy 在选择、sensor 在接触。你把 sensor + robot + policy 三者耦合，benchmark 怎么标准化？

Paper 给的几个 dataset：GelFabric [152]、ViTac [153]、VisGel [154]、Touch and Go [161]、ObjectFolder [158]、PoseIt [159]、PHAC-2 [160]。

Paper 建议的 hierarchical benchmarking 思路我觉得很对：

1. 先单独 benchmark **sensor**（resolution、latency、hysteresis、drift）；
2. 再 benchmark **representation learning**（在 fixed sensor + fixed dataset 上）；
3. 再 benchmark **policy**（在 fixed sensor + representation 上）；
4. 最后 benchmark **整体 system**。

评估指标也要改。SSIM、PSNR 这些 vision metric 在 tactile image 上语义不一致 — luminance 跟 contact force 间接相关，没有直接物理含义。建议从 psychophysics 借：**two-point discrimination** [164]、**grating orientation** [165]，这些是人类 tactile acuity 的 gold standard。

参考：Touch and Go https://github.com/CMURoboTouch/TouchAndGo、ObjectFolder https://objectfolder.org/

---

## 怎么从 tactile data 提取信息

这是 Section VIII，方法分几类：

**Force estimation** — array sensor 直接标定；optical 通过 marker motion 或 image learning 反推。Inverse FEM [168] 是从 marker displacement $\mathbf{u}$ 反解 $\mathbf{F}$，解 $\mathbf{F} = \mathbf{K}\mathbf{u}$ 这种 inverse problem，ill-posed 但 work。

**Slip detection** — 五类方法，最 popular 的是 vibration-based（高通滤波看 energy burst）和 skin stretch asymmetry（marker displacement field 不均匀）。GelSight 上后者最常用，因为 marker 给了稠密 displacement field。

**Pose / shape** — GelSight 用 photometric stereo 算 normal 再 Poisson 重建 depth。TacTip 用 image moments [16] 或 SIFT [17] 算 contact pose。3D shape reconstruction 走 DeepSDF [189]：一个 latent vector + MLP decoder 拟合 signed distance function $f_\theta(\mathbf{x}, \mathbf{z}) = s$。

**Hardness** — 测力-位移曲线斜率 $k = dF/d\delta$。$k$ 大 = 硬。GelSight 可以不控制 pressing trajectory，直接看 contact geometry 和 normal force 的 correlation [198]。最新还有 biomimetic 方法 [202] 用 contact area rate-of-change 模仿 SA-I afferent。

**Texture** — BioTac Bayesian exploration [203] 是经典，用所有 modalities 做 Bayesian 似然比。最近用 attention network [157] 和 **spiking neural network** [204, 205] — 这条线很 neuromorphic，跟 event-driven sensor 配对。

**Outlook**：deep model 部署在 edge 是问题 — 人皮是 local processing，模型必须小。NAS 可能是答案 [206]。Data hungry 是问题 — 需要 zero-shot learning [207]。**Foundation model for touch 还没出现**，这是个大空白。

---

## Multimodal / Cross-modal

Vision 和 touch 不能简单 feature concat。Paper Fig. 2 给两种融合范式：

**Feature-level fusion** — vision encoder → $f_v$，touch encoder → $f_t$，concat → MLP → prediction。Lee et al. [220] 是代表。

**Point-cloud fusion** — vision 给 depth → $P_v$，GelSight 给 local contact → $P_t$，在 shared 3D space 用 ICP 或 Levenberg-Marquardt 对齐，joint reasoning。Izatt et al. [224] 和 Bimbo et al. [223] 是代表。

LM 优化：
$$\mathbf{x}_{k+1} = \mathbf{x}_k - (\mathbf{J}^T\mathbf{J} + \lambda\mathbf{I})^{-1}\mathbf{J}^T\mathbf{r}(\mathbf{x}_k)$$

Cross-modal 生成很火：cGAN 把视觉生成 touch，touch 生成视觉 [156, 229]。VisGel [154] 处理 scale discrepancy — vision 拍整个 scene，touch 只看一小块。

最近 large multimodal foundation model 开始出现：
- **TVL** [231]: Touch-Vision-Language dataset
- **Octopi** [232]: large tactile-language model
- **Touch100k** [233]: 10万 touch-language-vision triplet
- **Binding Touch to Everything** [234]: contrastive learning 把 tactile embedding 跟 CLIP image embedding 对齐

**关键洞察**：temporal asymmetry 没人 exploit。Vision 在 contact *前* 给 global overview，touch 在 contact *后* 给 detail feedback。这俩在时间上是 *异步* complementary 的，但目前工作都当 static snapshot fusion，浪费了时间维度的信息。这是 manipulation 闭环控制的关键。

参考：TVL https://tvl-dataset.github.io/、Binding Touch https://github.com/CMU-Perceptual-Computing-Lab/BindingTouch

---

## Active Touch — 这章是 paper 灵魂

Paper 花三页讲 *概念史*，这非常 rare。三个里程碑：

**Gibson 1962 [240]**：区分 active vs passive touch。Active touch 是 perceiver *主动 initiate* tactile event，运动是 *intentional* search for stimulation。手摸东西时手指有 purpose 在调整，不是瞎动。这是 phenomenological 区分，但含义是：纯被动 sensor 是 *sensor*，active touch 是 *sensorimotor system*。

**Bajcsy 1988 [241]**：active perception = intentionally changing sensor state parameters based on sensing strategies。关键：*not* simple feedback control，是 reasoning + decision making + control 的复杂 loop。Classical control 不够。

**Lederman & Klatzky 1987 [244]**：**Exploratory Procedures (EPs)** taxonomy。Hand + brain 是 intelligent device，motor 扩展 sensory。每个 EP 针对一个 object property：
- Lateral motion → texture
- Pressure → compliance
- Static enclosure → volume
- Contour following → exact shape
- Unsupported holding → weight

**Robotic 实现**：早期 Maekawa 1992 [247] finger-shaped sensor + active touch for profile；2010s BioTac 上 Fishel & Loeb [88, 203] 做 Bayesian exploration，action 选择最大化 expected information gain：

$$a^* = \arg\max_a \mathbb{E}\left[H(P(c|\mathbf{o})) - \mathbb{E}_{\mathbf{o}'} H(P(c|\mathbf{o},\mathbf{o}',a))\right]$$

Lepora group 在 iCub + TacTip 上做 contour following [254-256]，用 Bayesian filter 估 edge orientation，后来换 deep learning [257]。最新工作 Tactofind [262] 在 dexterous hand 上纯触觉 object localization + identification + grasping，无视觉。

**Outlook 两个 open question**：
1. Tactile control + active touch 结合 — 目前 active touch 都是 trivial primitive (tap/press/slide)，缺乏人类那种 dexterous exploratory motion。
2. Active inference framework [265] — Friston free energy principle，把 action + perception + learning 统一在一个 variational objective $\mathcal{F}(\phi, a) = \mathbb{E}_{q_\phi(s)}[\ln q_\phi(s) - \ln p(s, o|a)]$ 下。这是把 active touch 推到 lifelong learning 的理论框架。

---

## 我个人的几点直觉

**1. Tactile 本质是 active + closed-loop sensing**。这跟 vision 根本不同。Vision benchmark 可以静态 dataset 化，tactile 必然带 policy + embodiment。所以 tactile foundation model 不能照搬 CLIP 那套，必须包含 action 维度。这是为什么 TVL [231]、Octopi [232] 这类工作现在才刚开始，而且大家都在摸索 form。

**2. Optical tactile sensor 是过去十年赢家**，因为它把 tactile vision 化了，能用 CV 工具链。代价是 data rate 和 latency。下一个十年可能向 *neuromorphic event-driven* 转移 [45, 71, 109] — 跟生物触觉一样输出 spike train，带宽低、latency 低、跟 spiking neural network 配对。

**3. Sim-to-real 的核心 bottleneck 是 contact mechanics**，不是 optics、不是 electronics。摩擦、slip、shear 精确建模至今没解决。Model-based + domain randomization 比 physics fidelity 更实用。Narang et al. [128] 用 GPU FEM 算得动，但 shear 那块还是要 randomization 兜底。

**4. Multimodal 不能停留在 feature concat**，temporal asymmetry 是 manipulation 闭环的关键信息，几乎没人 exploit。这是大空白。

**5. Active touch 是真正的 frontier**。目前所有 active touch 工作用的都是 trivial motor primitive。Gibson/Bajcsy/Lederman-Klatzky 1980s 的 EP framework 在 robotics 上还远未实现。要实现它需要 *tactile servo control + lifelong learning + active inference* 三者合一。

**6. 评估指标缺失**。没有像 ImageNet 之于 vision 的 tactile benchmark。需要 hierarchical benchmarking + psychophysics-inspired metrics。

**7. 分布式供电 + 分布式计算 是工程瓶颈**。Full-body skin 的 wiring + battery + compute 是 nightmare。Neuromorphic + self-powered (solar skin [81]) + self-organizing network [94] 是仅有的几条出路。

---

## 如果只能 follow 一条线

我会选 **vision-based tactile + sim-to-real + active inference** 的交叉。因为它是唯一能同时解决 sensor fidelity、data scaling、和 closed-loop dexterity 三个问题的方向：

- Vision-based tactile 给你 high-fidelity signal；
- Sim-to-real 给你 data scaling；
- Active inference 给你 closed-loop perception-action-learning 统一 framework。

Paper 给的就是这个 frontier 的 map。剩下的事，做就完了。

---

# Tactile Robotics: An Outlook — 深度解读

Andrej，这篇 paper 是一篇领域级 outlook review（不是普通 survey，是一篇定义性 manifesto），由 Shan Luo (ICL)、Nathan F. Lepora (Bristol)、Wenzhen Yuan (GelSight 系列作者之一)、Kaspar Althoefer、Gordon Cheng (Munich)、Ravinder Dahiya (Glasgow) 六位核心玩家合著。它继承并扩展了 Dahiya 2010 那篇经典综述 "Tactile Sensing—from Humans to Humanoids" (IEEE TRO) 的视野，目标是把 "Tactile Robotics" 从一个散落的研究方向正式立成一个 sub-field。下面我会按 paper 结构走，但每个环节都补上对应公式、架构图逻辑、以及我自己联想的相关工作。

---

## 1. 论文的核心论点

Paper 在 Section II 给了一个 explicit 的 field definition：

> **Tactile Robotics is a field of robotics that focuses on the development and integration of tactile-sensing technologies into robotic systems, with the goal of enhancing a robot's ability to perceive and interact with its surroundings by providing it with a sense of touch.**

这个定义本身就排除了把 tactile sensing 仅当 sensor engineering 的视角。它强调 *integration*（感知+执行闭环）+ *interaction*（不被动感知）。

### 三个关键 concept 切分（Section II）

- **Tactile sensing vs Force/Torque sensing**: force/torque 是 single-axis scalar，所谓 "point sensor"，tactile 是 *spatially distributed* readings。论文强调：tactile sensing 应包含一个空间分布的 contact 信息场，不仅仅是单一 axis 上的力。但 6-axis F/T 配合已知 end-effector surface geometry 也能反解 contact location（早期 Liu 等的工作 [14]），缺点是只对 single contact 有效，且 6-axis F/T 体积大、贵、脆。
  
- **Tactile sensing vs Haptics**: haptics 传统上更偏向 human feedback 那一侧；haptic sensing = tactile sensing（contact area）+ kinesthetic sensing（body proprioception）。这条切分很重要，因为后面 multimodal 和 active touch 都依赖于 proprio-tactile coupling。

- **Taxel 与 Tactile image**: taxel = tactile pixel，类比 pixel。一旦 sensor 输出是 array form，它就是一张 "tactile image"，可以直接套 CV 工具链（CNN、SIFT、image moments）。这是过去十年整个领域从手工 feature 转向 deep learning 的物理基础。

---

## 2. Materials (Section III) — 仿生皮肤的工程问题

### 2.1 人皮分层结构 (Reference [18], [19])

人皮有三层：
- **Epidermis**: 75–150 μm，外层保护，含 Merkel disc（SA-I，慢适应，感受 edge/texture）；
- **Dermis**: 1–4 mm，结构支撑，含 Meissner corpuscle (RA-I, 快适应, 低频振动 ~40 Hz)、Ruffini ending (SA-II, 拉伸)、Pacini corpuscle (RA-II/PC, 高频振动 ~250 Hz)；
- **Hypodermis**: 脂肪层，可变厚度。

Epidermis–dermis 交界有 **dermal papillae** 这种手指状突起，shallow mechanoreceptors 就分布在这里，这个结构本身就是一个机械放大器（deformation amplification）。这一点对 TacTip 这种 biomimetic optical sensor 设计直接启发：它把 marker 放在 papillae-like 突起上 [29]。

### 2.2 机器人皮肤材料的硬度区间

Paper 列了一个有用的硬度参考表：

| Sensor | Shore 硬度 |
|---|---|
| GelSight [28] | 5A–20A |
| TacTip 外层 3D-printed skin [29, 30] | 26–28A |
| TacTip 内部 gel | 可变 |
| uSkin [13] | 00–50 |

注意这里有意思的矛盾：**皮肤越软 = 对小力越敏感 (optical sensor)**，但 **皮肤越软 = load 分布更宽 = 局部分辨率下降 (capacitive/barometric array)**。这是一个 fundamental design tradeoff：

$$
\text{Spatial resolution} \propto \frac{1}{\text{skin compliance}} \quad \text{(array sensors)}
$$
$$
\text{Force sensitivity} \propto \text{skin compliance} \quad \text{(optical sensors)}
$$

所以皮肤硬度必须 per-application 优化，不存在 universal optimum。

### 2.3 关键材料

- **PDMS** (Sylgard 184): biocompatible, soft (Shore 00-40 ~ A-30 之间)，可光刻；缺点是 high strain 下易撕裂，永久变形；
- **PVDF** (polyvinylidene fluoride): piezoelectric polymer，柔性，对动态 deformation 敏感；
- **PZT** (lead zirconate titanate): piezoelectric ceramic，高 d33 系数，但脆；
- **Hydrogel**: ionic conductivity，可 stretchable，最近几年很火；
- **EIT/ERT conductive layer** [26, 27]: 用一个 thin resistive layer + boundary electrodes，通过 tomographic reconstruction 反推内部 conductivity 分布，优点是无内部 wiring，缺点是 ill-posed inverse problem、分辨率有限。

### 2.4 Section III.B 的 Outlook

未来材料愿景：non-homogeneous, non-linear, visco-elastic, anisotropic 的多层复合皮肤。这要求：
- Multi-material 3D printing (PolyJet, EHD printing)；
- Laser micromachining；
- Lithography。

Paper [37] 提到 Nassar 等人用 fully 3D-printed piezoelectric pressure sensor 做了 dynamic tactile sensing demo。这个方向我个人认为会跟 *embedded microfluidic channels*（用流体压力做 sensing+actuation，参考 Shepherd group @ Cornell 的工作）有大量融合空间。

参考链接：
- GelSight project: http://gelsight.csail.mit.edu/
- TacTip: https://bristolroboticslab.github.io/tactile-gym/
- Dahiya group flexible electronics: https://www.gla.ac.uk/schools/engineering/research/researchthemes/beng/researchgroups/fbe/

---

## 3. Transduction Methods (Section IV) — 物理换能原理详解

这是 paper 最核心技术章节之一，我重点拆解五种方法各自的物理公式。

### 3.1 Piezoresistive

机制：材料电阻率随应力变化。基本关系：

$$
\frac{\Delta R}{R} = \pi_L \sigma_L + \pi_T \sigma_T
$$

其中 $\pi_L, \pi_T$ 是 longitudinal 和 transverse piezoresistive coefficients (Pa$^{-1}$)，$\sigma_L, \sigma_T$ 是相应方向的 stress (Pa)。

对于 conductive polymer / carbon-loaded elastomer 这种 strain-gauge-like 材料，更常见的是：

$$
\frac{\Delta R}{R} = GF \cdot \varepsilon, \quad GF \approx 2-200
$$

其中 $GF$ 是 gauge factor，$\varepsilon$ 是 strain。

**代表**: NeuTouch [45] (39 taxels + graphene piezoresistive thin film)，NUskin [46] (4 kHz 采样率，这是非常高的，因为它们想做 event-driven spike 输出)。

**缺点**: hysteresis 严重（高分子链 relaxation），temperature drift。

### 3.2 Piezoelectric

机制：应力产生电荷。本构关系：

$$
D_i = d_{ij} \sigma_j + \varepsilon_{ij} E_j
$$

$d_{ij}$ 是 piezoelectric coefficient (pC/N)，常见 PVDF $d_{33} \approx 20-30$ pC/N，PZT $d_{33} \approx 200-600$ pC/N。

输出电压：

$$
V = \frac{g_{ij} \cdot F \cdot t}{A}
$$

其中 $g_{ij} = d_{ij}/\varepsilon$ 是 voltage coefficient (V·m/N)，$F$ 是力，$t$ 是厚度，$A$ 是电极面积。

**关键限制**: 只对 *dynamic* force 敏感（voltage decay under DC load，因为 charge leakage through finite impedance）。所以 piezoelectric 不能测 static pressure，但适合 slip / texture vibration。

### 3.3 Capacitive

平行板电容器：

$$
C = \frac{\varepsilon_0 \varepsilon_r A}{d}
$$

加压力后 $d \to d - \Delta d$，所以：

$$
\frac{\Delta C}{C} \approx \frac{\Delta d}{d} \quad (\Delta d \ll d)
$$

线性化小信号响应。对于大变形需要非线性模型：

$$
C = \frac{\varepsilon_0 \varepsilon_r A}{d_0} \cdot \frac{1}{1 - \Delta d/d_0}
$$

代表: Pressure Profile Systems (PPS), iCub hand [47], Allegro hand [54], PR2 gripper [55], RoboSkin [103, 104]。

**缺点**: electromagnetic crosstalk（相邻电容串扰）、hysteresis、需要前端 ASIC 走线多。

### 3.4 Magnetic (Hall effect)

Hall 电压公式：

$$
V_H = \frac{I B}{n q t} = R_H \cdot \frac{I B}{t}
$$

其中 $I$ 是偏置电流，$B$ 是垂直磁场强度，$n$ 是载流子浓度，$q$ 是单位电荷，$t$ 是板厚，$R_H = 1/(nq)$ 是 Hall coefficient。

Magnetic tactile sensor 把一个小磁体埋在 elastomer 里，外力 → elastomer 形变 → 磁体位移 → $B$ 改变 → Hall 电压改变。

代表: **uSkin** [57, 113] (三轴力，3-axis Hall + 3D-printed magnet grid)；**ReSkin** [58] (CMU/Google，可替换、低成本，磁阻 + magnetometer)。

**优点**: linear shear response、wide dynamic range；**缺点**: 外部磁场干扰（电机、电磁阀）。

参考链接:
- ReSkin: https://sites.google.com/view/reskin
- uSkin: https://www.researchgate.net/project/uSkin

### 3.5 Optical Tactile Sensors (vision-based)

这是过去十年最 dominant 的方向。两大类：

#### A. Reflectance-based (GelSight family)

原理：弹性透明 gel 表面涂 reflective coating，三色 LED (R, G, B) 从不同方向照，camera 看 gel 内表面颜色 → 颜色对应法向。

数学上是 **photometric stereo** [183]：

$$
I_c(\mathbf{x}) = \rho(\mathbf{x}) \cdot \mathbf{n}(\mathbf{x}) \cdot \mathbf{l}_c, \quad c \in \{R, G, B\}
$$

其中 $I_c$ 是 channel $c$ 的像素强度，$\rho$ 是 albedo (这里近似 constant 因为 reflective coating 均匀)，$\mathbf{n}(\mathbf{x}) = (n_x, n_y, n_z)$ 是 surface normal at $\mathbf{x}$，$\mathbf{l}_c$ 是 LED $c$ 的 lighting direction (3D unit vector)。

三个方程三个未知数 ($n_x, n_y, n_z$ with $|\mathbf{n}|=1$ 约束)，可解。然后对 $\mathbf{n}$ 做 Poisson reconstruction 得到 depth map $z(\mathbf{x})$：

$$
\nabla^2 z = \nabla \cdot \mathbf{n}
$$

代表: GelSight [28], GelSlim [59], GelTip [31], DIGIT [60], DIGIT 360 [61], TouchRoller [62], GelFinger [63], Insight [33], Minsight [34], F-TOUCH [32]。

#### B. Marker-based (TacTip family)

原理：gel 内表面放 marker (颜料点)，contact 后 marker 位移。通过 optical flow / marker tracking 得到 displacement field $\mathbf{u}(\mathbf{x})$，再反推 contact force / geometry。

代表: TacTip [29], GelForce [66, 67], ChromoTouch [68], FingerVision [69]。

#### C. 其他光学：depth camera [70], event camera [71], thermal camera [72], multi-camera (OmniTact [73]), fiber-optic bundle [77] 用于 MRI。

**关键 tradeoff**: optical sensor 空间分辨率高 (~pixel level, 100s of μm)，但 data 流量大 (30+ fps video)，实时处理负担大。

参考链接：
- DIGIT: https://digit.dahiya.rocks/
- GelSight Mini: https://gelsight.com/
- OmniTact: https://sites.google.com/berkeley.edu/omnitact
- TACTO simulator (Calandra): https://github.com/facebookresearch/tacto

### 3.6 Paper IV.B 的 outlook

- 没有一种 universal transduction method，需要按 task 挑 (spatial resolution vs temporal resolution vs sensitivity vs durability vs cost)；
- 成本问题: BioTac 当年 \$5k–10k，现在 DIGIT \$350，GelSight Mini \$500；
- Hardware-software co-design: CNN 处理 optical image，frequency decomposition 处理 vibration，neuromorphic computing 处理 spike [93]。

---

## 4. Tactile Sensor Networks (Section V) — 分布式通信

### 4.1 人皮传感网络 [95, 96]

- 皮肤面积 1.5–2 m²，myelinated fiber 传导速度 33–75 m/s；
- Fingertip 与 lip 的 mechanoreceptor 密度比身体其他部位高 4–5 倍；
- Receptive field overlap → **tactile hyperacuity** [105]：单 taxel 物理间距是 X，但 perception acuity 可以是 X/10，因为多 receptor 重叠加权 + 中央 processing。

### 4.2 Robotic 实现

- **早期**: matrix sensors + 大量 wiring；
- **Modular**: RI-MAN [100], ARMAR-III [101], TWENDY-ONE [102], RoboSkin (triangular module) [103, 104]；
- **Hierarchical + real-time middleware**: Youssefi et al. [108]；
- **Neuromorphic 边缘压缩**: Bartolozzi et al. [109]，event-driven 把冗余去掉再传，大幅降低带宽；
- **Self-organizing network protocol**: Cheng et al. [94]，自动构建 bidirectional communication tree，dynamic re-routing, load balancing。

### 4.3 关键挑战

- **异构网络**: fingertip 用 optical, arm 用 capacitive, palm 用 magnetic — 需要 unified interface；
- **分布式供电**: 人皮是分布式能量 + 分布式 sensing；目前 robotic skin 集中供电是个 bottleneck。Paper 提到 [81] 用 miniaturized solar cell 既 sensing 又发电 (self-powered sensor)，这是 multifunctional device 的范例。

---

## 5. Simulation for Tactile Sensing (Section VI) — sim-to-real 关键

这是 sim-to-real learning 的瓶颈所在。Paper 把 simulation 方法分三类，我详细展开。

### 5.1 Physics-based Methods

#### A. FEM-based 形变模拟

弹性体小变形 linear elasticity:

$$
\nabla \cdot \boldsymbol{\sigma} + \mathbf{f} = 0, \quad \boldsymbol{\sigma} = \mathbf{C} : \boldsymbol{\varepsilon}
$$

$\boldsymbol{\sigma}$ 是 Cauchy stress tensor, $\boldsymbol{\varepsilon} = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)$ 是 strain tensor, $\mathbf{C}$ 是 4th-order stiffness tensor (hyperelastic: Neo-Hookean, Mooney-Rivlin for silicone)。

FEM discretization:

$$
\mathbf{K} \mathbf{u} = \mathbf{F}_{ext}
$$

$\mathbf{K}$ 是 global stiffness matrix ($N \times N$, $N$ = 3 × nodes)，$\mathbf{u}$ 是 node displacement vector，$\mathbf{F}_{ext}$ 是 external force vector。求解用 Newton-Raphson or conjugate gradient。GPU 加速的代表是 Narang et al. [128] (NVIDIA FleX/Warp)。

**问题**: 计算量爆炸，real-time 难做。简化方法：
- Particle-based (position-based dynamics, MPM); Chen et al. Tacchi [129]；
- Simplified mechanical model [130]。

#### B. Optical simulation (Phong reflection model)

GelSight 渲染简化用 Phong [131]:

$$
I = k_a I_a + k_d (\mathbf{L} \cdot \mathbf{N}) I_d + k_s (\mathbf{R} \cdot \mathbf{V})^\alpha I_s
$$

- $k_a, k_d, k_s$：ambient, diffuse, specular reflection coefficients (材质属性)；
- $I_a, I_d, I_s$：ambient, diffuse, specular light intensities；
- $\mathbf{L}$: light direction (from surface to light)；
- $\mathbf{N}$: surface normal；
- $\mathbf{R}$: reflected light direction = $2(\mathbf{N}\cdot\mathbf{L})\mathbf{N} - \mathbf{L}$；
- $\mathbf{V}$: view direction；
- $\alpha$: shininess exponent.

代表: Gomes et al. [123, 124] (GelSight simulation), Si & Yuan Taxim [141] (lookup table-based polynomial mapping)。

更精确的 physically-based rendering (PBRT) [132] 在 Wang et al. TACTO [133] 中实现。

### 5.2 Data-driven Methods

用 GAN / CNN 学习 geometry-to-reading 映射：

#### A. cGAN for GelSight [134]

$$
\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

conditional version 把 depth map 当 condition。

#### B. Image-to-image translation (TacTip) [137]

Church et al. 用 GAN 把 robot simulator 的 depth map 翻译成真实 TacTip marker shear pattern，绕过 skin dynamics 的精确建模。

代表应用: surface contour tracing [138], bimanual manipulation [139]。

### 5.3 Model-based Methods

轻量级参数化模型 + 小数据标定。

例 1：Capacitive array 用 Gaussian blur convolve ground truth contact geometry [140]：

$$
I_{sim}(\mathbf{x}) = \sum_{\mathbf{x}'} G_\sigma(\mathbf{x} - \mathbf{x}') \cdot \delta_{contact}(\mathbf{x}')
$$

$G_\sigma$ 是高斯核，$\sigma$ 表征 taxel spatial spread。

例 2：Taxim [141] 用 polynomial lookup table 拟合 GelSight 的 shape-to-color mapping，给定任意 geometry 快速生成图像。

### 5.4 Sim-to-Real 转移

Paper 区分两条路：

1. **Fidelity-first**: 模拟越像真越好，直接迁移 [123, 124, 141, 142]；
2. **Feature-first**: 不模拟 raw signal，模拟 representative feature (contact geometry, low-dim embedding) [121, 122, 147]。代表: Qi et al. [121] 用 vision+touch 做 in-hand rotation (AnyRotate [122] 是其扩展)。

Domain gap 处理：
- **Transfer learning + generative modeling** [144, 145]；
- **Domain randomization** [146]；
- **Signal normalization** [130]。

### 5.5 Section VI.C 的核心 dilemma

**效率 vs 精度**: model-based 看似最优，但 complex curved soft medium 没好模型。**高保真 simulator 真的必要吗**？不一定 — basic physics + 好 domain transfer 也能 zero-shot [126, 145]。还有一个未解决问题: **shear force / friction 模拟**仍非常粗糙，多数 simulator 假设 Coulomb friction + fixed coefficient。

参考链接：
- TACTO: https://github.com/facebookresearch/tacto
- Taxim: https://github.com/ARISE-Initiative/taxim
- Tactile Gym: https://github.com/ac-93/tactile_gym
- Tacchi: https://github.com/TactileVis/Tacchi

---

## 6. Tactile Data Collection and Benchmarking (Section VII)

### 6.1 Paper Table I 的核心 dataset 汇总

| Dataset | Sensors | Modalities | Collection |
|---|---|---|---|
| GelFabric [152] | GelSight + Canon T2i + Kinect | tactile video, RGB, depth | Manual |
| ViTac [153] | GelSight + Canon T2i | tactile video + RGB | Teleoperated |
| VisGel [154] | GelSight + camera | tactile + visual video | Robotic |
| Gomes [124] | GelSight + simulation | real + simulated | Robotic |
| Touch and Go [161] | GelSight + camera | visual + tactile video | Teleoperated |
| ObjectFolder [158] | 3D scanner + mic + GelSight | visual + sound + tactile | Teleop+Robotic |
| SoftSlidingGym [162] | Camera + GelTip | camera + tactile | Robotic |
| PoseIt [159] | Camera + GelSight + F/T | RGB + tactile + F/T | Robotic |
| PHAC-2 [160] | BioTac | tactile + haptic adjectives | Robotic |

### 6.2 Benchmarking 困境

Paper VII.B 指出 computer vision benchmark 是 *passive* 的（一个固定 dataset 就行），tactile 是 *active* 的，必须包含 robot + policy + sensor coupling，无法 decouple。

建议 **hierarchical benchmarking**：
1. 先单独 benchmark sensor (resolution, latency, hysteresis, drift)；
2. 再 benchmark representation learning (在 fixed sensor + fixed dataset 上)；
3. 再 benchmark policy (在 fixed sensor + representation 上)；
4. 最后 benchmark 整体 system。

### 6.3 评估指标问题

Vision 常用 SSIM, PSNR, MAE，但这些 metric 在 tactile image 上语义不一定一致：luminance 跟 contact force 间接相关，contrast 跟 skin movement 间接相关，没有直接物理含义。

Paper 建议从 psychophysics 借：**two-point discrimination** [164, 165], **grating-orientation** experiments，这些是人类 tactile acuity 的 gold standard。

参考链接：
- YCB: https://www.ycbbenchmarks.com/
- ObjectFolder: https://objectfolder.org/
- Touch and Go: https://github.com/CMURoboTouch/TouchAndGo
- PHAC-2: https://hapticslab.org/phac-2/

---

## 7. Tactile Data Interpretation (Section VIII) — 任务侧方法详解

### 7.1 Force estimation [166, 167]

- Array sensors: 出厂直接标定 force-per-taxel；
- Optical: 通过 marker motion 或 image learning 反推 force。Inverse FEM 方法 [168]：从 marker displacement $\mathbf{u}$ 反解 $\mathbf{F}$，需要解 inverse 问题 $\mathbf{F} = \mathbf{K} \mathbf{u}$ (ill-posed)。
- Learning-based: 直接 CNN regression from image to (F_x, F_y, F_z)。

### 7.2 Slip detection

五大类方法：
1. **Friction coefficient**: $\mu = F_{shear}/F_{normal}$, compare with threshold [169–171]；
2. **Vibration burst**: accelerometer 或 piezoelectric signal high-pass + energy [172–175]；
3. **Shear strain sudden change**: sensor surface strain field 异常 [176]；
4. **Object movement on surface**: 直接 optical tracking [177]；
5. **Skin stretch asymmetry**: marker displacement field unevenness [178, 179] — 这是 GelSight 上最常用的，因为 marker 给了稠密 displacement field。

最近还有 end-to-end learning [180, 181]，以及 Ou et al. [182] 的 *markerless* sensor 上用 supervised regression 预测虚拟 marker motion。

### 7.3 Pose / shape / size

- **Photometric stereo** on GelSight → surface normal → depth [183]；
- **Image moments** [16, 184]: 对 tactile image $I(x,y)$ 计算 
$$
m_{pq} = \int \int x^p y^q I(x,y) \, dx \, dy
$$
$p+q$ 阶矩，低阶 (centroid $m_{10}/m_{00}$, $m_{01}/m_{00}$) 给 contact center，高阶 (covariance) 给 contact shape；
- **SIFT descriptor** on tactile image [17]；
- **CNN-based** pose regression [191]；
- **DeepSDF** [189] for 3D shape reconstruction — Comi et al. 用 TacTip + DeepSDF network (latent vector + MLP decoder of signed distance function):
$$
f_\theta(\mathbf{x}, \mathbf{z}) = s: \quad \text{SDF}(\mathbf{x}) \approx s
$$

### 7.4 Hardness / softness

直觉：按一个 rigid object, normal force 随 penetration depth 线性增长；按一个 soft object, force 增长慢。所以测力-位移曲线斜率：

$$
k_{apparent} = \frac{dF}{d\delta}
$$

$\delta$ 是 indentation depth。$k$ 大 = 硬，$k$ 小 = 软。

方法：
- Controlled pressing + force rate analysis [193–197]；
- Free pressing + geometry-force correlation [198]；
- CNN+RNN 端到端 [199–201]；
- Biomimetic compliance from contact area rate-of-change [202] — 这是 Pagnanelli 等最近的工作，模仿人皮 SA-I afferent 的 response。

### 7.5 Texture / roughness

- BioTac Bayesian exploration [203]: 用 all BioTac modalities (DC/AC pressure, DC/AC temperature, 19 electrodes impedance) 做 Bayesian 似然比测试；
- Attention network for fabric texture [157]；
- **Spiking neural network** [204] + **spiking graph neural network** [205] — 这条线很 neuromorphic，跟 NUSkin / NeuTouch 的 event-driven sensor 配对。

### 7.6 Section VIII.B 的三大挑战

1. **Deep model 部署在 edge**: 人皮是 local processing 的，模型必须小。Neural architecture search (NAS) 可能是解决方案 [206]；
2. **Data hungry**: 需 zero-shot learning [207] — 用 haptic attribute 做 attribute-based zero-shot；或者 visual-semantic 联合 [155]；
3. **Foundation model for touch**: 跨 sensor、跨 task 的 tactile foundation model 还没出现。Spiking neural network 因为生物触觉本身是 spike train [208]，可能是更 native 的架构。

参考链接：
- GelSight force estimation: https://github.com/gelsightcsail/gelsight
- DeepSDF: https://github.com/facebookresearch/DeepSDF
- Spiking tactile: https://github.com/sohweb/tactilesgnet

---

## 8. Multimodal and Cross-modal Learning (Section IX)

### 8.1 神经科学背景 [209–212]

- Visual imagery 增强 tactile orientation discrimination [210]；
- Brain 用 shared object model across modalities [211]；
- Texture 细节 vision 看不清时，brain 自动 weight 更多到 touch [212]。

这是 cross-modal 的生物学基础。

### 8.2 Paper Fig. 2 的两种融合范式

**A. Feature-level fusion (Fig. 2a)**

```
vision → encoder_v → f_v ∈ R^d
                          ↘
                            concat → MLP → prediction
                          ↗
touch → encoder_t → f_t ∈ R^d
```

代表 [220]: Lee et al. (Making Sense of Vision and Touch) 在 contact-rich task 上 concat visual + tactile feature；[221] Mosaic (Tatiya et al.) 多 sensory object property。

**B. Point-cloud fusion (Fig. 2b)**

```
vision → depth → point cloud P_v
                          ↘
                            align (ICP / Levenberg-Marquardt) → joint reasoning
                          ↗
touch → GelSight → point cloud P_t (local contact)
```

代表: Izatt et al. [224] (vision + GelSight tracking), Bimbo et al. [223] (touch + vision pose estimation via gradient descent / LM [222])。

Levenberg-Marquardt 是经典非线性 least squares：
$$
\mathbf{x}_{k+1} = \mathbf{x}_k - (\mathbf{J}^T \mathbf{J} + \lambda \mathbf{I})^{-1} \mathbf{J}^T \mathbf{r}(\mathbf{x}_k)
$$

$\mathbf{J}$ 是残差 $\mathbf{r}$ 对参数 $\mathbf{x}$ 的 Jacobian, $\lambda$ 是 damping parameter (Gauss-Newton $\lambda \to 0$, gradient descent $\lambda \to \infty$)。

### 8.3 Cross-modal learning

- Pair vision+touch for material classification [225]；
- Cross-modal subspace learning [227]；
- **Cross-modal generation**:
  - cGAN: 视觉生成 touch, touch 生成视觉 [156, 229]；
  - VisGel [154]: scale discrepancy 处理 (vision captures whole scene, touch observes small region)；
  - Image restyling based on touch signal [229]: 把"粗糙岩石"图像 restyle 成"砖"纹理。

### 8.4 Large multimodal foundation models

- **Multiply** [230]: 多感官 object-centric embodied LLM in 3D world；
- **TVL** [231]: Touch-Vision-Language dataset, Fu et al. (Berkeley/Meta)；
- **Octopi** [232]: Large tactile-language model for object property reasoning；
- **Touch100k** [233]: 10万 touch-language-vision triplet；
- **Binding Touch to Everything** [234]: contrastive learning 把 tactile embedding 跟 CLIP image embedding 对齐。

### 8.5 Section IX.C 的两个关键挑战

1. **超越 feature concatenation**: 不能把 vision 和 touch 当 equal，应该用 correlation analysis [153] 或 attention mechanism [157, 236] 动态 weight；state-space model [237] 和 RL framework [238] 也是 emerging 路径。

2. **Temporal dimension**: 大多数现有工作把 vision 和 touch 当 static snapshot fusion，但实际两者在时间上是 *异步* 的：
   - Vision: 接触前提供 global overview；
   - Touch: 接触后持续 detail feedback。
   
   这种 *temporal complementarity* 还没被充分利用，是 manipulation 闭环控制的关键。

参考链接：
- VisGel: https://github.com/YunzhuLi/VisGel
- Touch and Go: https://github.com/CMURoboTouch/TouchAndGo
- TVL dataset: https://tvl-dataset.github.io/
- Binding Touch to Everything: https://github.com/CMU-Perceptual-Computing-Lab/BindingTouch

---

## 9. Active Tactile Perception (Section X) — 这是我个人觉得最深刻的章节

Paper X.A 花了三页讲人类 active touch 的 *概念史*，这非常 rare，因为大多数 engineering paper 跳过这一段直接上 algorithm。这里我把它当核心讲。

### 9.1 Gibson 1962 [240]: Active vs Passive touch

Gibson 的洞察：
- **Passive touch**: 一个 unanticipated external force 触发感觉（别人碰你）；
- **Active touch**: perceiver 主动 *initiate* tactile event。

Active touch 的运动是 **intentional**: 不只是为了动，而是 *search for stimulation*，目的是获得 *facilitate perception* 的那种 stimulation。手摸东西时，手指是有 purpose 的 adjustment。

这是 phenomenological 区分，但有深刻 engineering 含义：纯被动 sensor 是 *sensor*，active touch 是 *sensorimotor system*。

### 9.2 Bajcsy 1988 [241]: Active perception

Bajcsy 的定义（系统科学视角）：

> Active perception = intentionally changing sensor's state parameters based on sensing strategies for data acquisition, which depend on current state of data source and the aim of the task.

关键点：
- 不是 simple feedback control；
- 而是包含 **reasoning + decision making + control** 的复杂 loop；
- Classical control theory 不够，需要 model-based perception。

### 9.3 Aloimonos 1990 [242]: Active vision

Active vision 是控制 sensor 的 *geometric parameters* (camera pose, focal length, etc.)，目的是 manipulate observation constraints to improve perceptual quality。后来 Bajcsy, Aloimonos, Tsotsos 2018 [243] 综述整理了这条线。

### 9.4 Lederman & Klatzky 1987 [244]: Haptic exploration, Exploratory Procedures (EPs)

Hand + brain 是一个 intelligent device，motor 扩展 sensory。他们提出 **Exploratory Procedures (EPs) taxonomy**：

| EP | Object property |
|---|---|
| Lateral motion | Texture |
| Pressure | Compliance / hardness |
| Static enclosure | Volume / global shape |
| Contour following | Exact shape |
| Unsupported holding | Weight |

每一个 EP 针对一个特定 object property，做 active control。比如 contour following = 沿 edge 滑动，edge 的 perceived shape 直接控制 finger motion path。

### 9.5 Robotic Active Touch 实现

#### A. 早期 (1990s–2000s)

- Roberts 1990 [245]: constraints & strategies for active touch exploration；
- Allen 1990 [246]: mapping EPs to shape representations；
- Maekawa 1992 [247]: finger-shaped tactile sensor + active touch for profile delineation；
- Shimojo & Ishikawa 1993 [248]: active touch + spatial filtering for roughness；
- Kaneko & Tanie 1994 [249]: self-posture changing method for contact point detection；
- Okamura, Turner, Cutkosky 1997 [250]: rolling + sliding haptic exploration；
- Cutkosky, Howe, Provancher 2008 [251]: Spring Handbook 综述。

#### B. 2010s Bayesian exploration

BioTac 上 Fishel & Loeb [88, 203] 实现 Bayesian texture discrimination：

后验：
$$
P(c | \mathbf{o}) = \frac{P(\mathbf{o} | c) P(c)}{\sum_{c'} P(\mathbf{o} | c') P(c')}
$$

$c$ 是 texture class，$\mathbf{o}$ 是 observation (multi-modal BioTac)。Action 选择最大化 expected information gain:
$$
a^* = \arg\max_a \mathbb{E}_{c \sim P(c|\mathbf{o})} \left[ H(P(c|\mathbf{o})) - \mathbb{E}_{\mathbf{o}' | a, c} H(P(c|\mathbf{o}, \mathbf{o}', a)) \right]
$$

Su et al. [195] 用类似框架控制 normal force 来 discriminate compliance。

#### C. Bayesian contour following

Lepora et al. [254–256] 在 iCub + TacTip 上做 contour following，用 Bayesian filter 估计 edge orientation，控制 sensor 沿 edge 滑动。后来 Church, Lloyd, Lepora [257] 换成 deep learning-based pose estimation，generalize 到 3D 复杂物体。Lloyd & Lepora [192] 进一步把 shear 加进去做 pushing + tracking。

#### D. Active object learning

- Kaboli & Cheng [258, 259]: multimodal skin + active object discrimination；
- TANDEM [260]: 联合学习 exploration + decision making，扩展 TANDEM3D [261]；
- Smith et al. [188]: active 3D shape reconstruction from vision + touch with priors；
- Tactofind [262]: dexterous hand 上纯触觉 object localization + identification + grasping，无视觉；
- SonicSense [263]: 用 acoustic vibration sensing 增强 in-hand object perception。

### 9.6 Section X.C 的 Outlook

两个 open questions：

1. **Tactile control + active touch 结合**: 目前 active touch 都是简单 tap / press / slide，缺乏人类那样的 dexterous exploratory motion。需要更好的 tactile servo control；
2. **从 experience 主动学习**: lifelong / online learning [264]；active inference framework [265] — Friston 的 free energy principle 在 robotics 上的应用，把 action + perception + learning 统一在一个 variational objective 下：
$$
\mathcal{F}(\phi, a) = \mathbb{E}_{q_\phi(s)} [\ln q_\phi(s) - \ln p(s, o | a)]
$$
最小化 free energy $\mathcal{F}$ 同时优化 belief $\phi$ 和 action $a$。

参考链接：
- Tactile Gym: https://github.com/ac-93/tactile_gym
- TANDEM: https://github.com/songsheng0326/tandem
- Active inference survey: https://arxiv.org/abs/2112.01871

---

## 10. Application & Discussion (Section XI)

应用横跨：
- **MIS** (Minimally Invasive Surgery) [11]: tactile feedback 给 surgeon 力觉；
- **Agriculture** [268, 269]: fruit firmness, crop monitoring；
- **Legged robot** [270]: tactile foot for terrain；
- **Haptic displays** [271]: Vis2Hap 把 vision 转成 haptic；
- **Pseudo-hologram + aero-haptic** [272]: Dahiya group 的 floating display；
- **Full-body humanoid skin** [273, 274]: ARMAR, iCub, human-robot collaboration。

未覆盖的话题：
- **Self-deformation / ego-vibration**: 机器人自身运动引起的 tactile signal，跟相机 motion blur / object motion 的区分问题类似，需要 *tactile ego-motion compensation*；
- **Sensor coverage**: 全身覆盖的密度 vs. 部分覆盖的覆盖策略，依赖于 rigid body 上的 soft skin 还是 fully soft robot。

---

## 11. 我个人的几点直觉总结

把 paper 读完，我觉得有几个 meta-level 的洞察：

**1. Tactile 是 active + closed-loop 的 sensing modality，不是 passive perception。**这跟 vision 本质不同。Vision benchmark 可以静态 dataset 化，tactile 必然带 policy + embodiment。这暗示着 tactile foundation model 不能简单照搬 CLIP 那套，必须包含 action。

**2. Optical tactile sensor (GelSight 家族) 是过去十年的赢家**，因为它把 tactile 问题 *vision 化* 了，能直接用 CV 工具链。但代价是 data rate 和 latency。下一个十年可能向 *neuromorphic event-driven* 转移 [45, 71, 109]。

**3. Sim-to-real 的核心问题是 contact mechanics**，不是 optics、不是 electronics。摩擦、slip、shear 的精确建模至今没解决。Model-based + domain randomization 可能比 physics fidelity 更实用。

**4. Multimodal 不能停留在 feature concat**，temporal asymmetry (vision 预 contact, touch 后 contact) 是 manipulation 闭环的关键信息，目前几乎没人 exploit。

**5. Active touch 是真正的 frontier**。目前所有 active touch 工作用的都是 trivial motor primitive (tap, press, slide)。Gibson/Bajcsy/Lederman-Klatzky 1980s 提出的 EP framework 在 robotics 上还远未实现。要实现它需要 *tactile servo control + lifelong learning + active inference* 三者合一。

**6. 评估指标缺失**。没有像 ImageNet 之于 vision 那样的 tactile benchmark。需要 hierarchical benchmarking + psychophysics-inspired metrics (two-point discrimination, grating orientation)。

**7. 分布式供电 + 分布式计算 是工程瓶颈**。Full-body skin 的 wiring + battery + compute 是 nightmare。Neuromorphic + self-powered (solar skin [81]) + self-organizing network [94] 是仅有的几条出路。

---

## 关键参考链接汇总

综述 & 项目主页：
- GelSight: http://gelsight.csail.mit.edu/
- DIGIT: https://digit.dahiya.rocks/
- TacTip: https://bristolroboticslab.github.io/tactile-gym/
- ReSkin: https://sites.google.com/view/reskin
- TACTO simulator: https://github.com/facebookresearch/tacto
- Taxim: https://github.com/ARISE-Initiative/taxim
- Tactile Gym: https://github.com/ac-93/tactile_gym
- TactileSGNet (spiking): https://github.com/sohweb/tactilesgnet
- YCB Benchmarks: https://www.ycbbenchmarks.com/
- ObjectFolder: https://objectfolder.org/
- Touch and Go: https://github.com/CMURoboTouch/TouchAndGo
- TVL dataset: https://tvl-dataset.github.io/
- Binding Touch to Everything: https://github.com/CMU-Perceptual-Computing-Lab/BindingTouch
- VisGel: https://github.com/YunzhuLi/VisGel
- DeepSDF: https://github.com/facebookresearch/DeepSDF
- Active inference survey: https://arxiv.org/abs/2112.01871

经典论文：
- Dahiya 2010 (前序综述): https://ieeexplore.ieee.org/document/5390586
- Gibson 1962 (active touch): https://psycnet.apa.org/record/1963-02213-001
- Bajcsy 1988 (active perception): https://ieeexplore.ieee.org/document/5968
- Lederman & Klatzky 1987 (EPs): https://www.sciencedirect.com/science/article/pii/0010028587900085
- GelSight (Yuan 2017): https://www.mdpi.com/1424-8220/17/12/2762
- DIGIT (Lambeta 2020): https://arxiv.org/abs/2005.14051

如果只能 follow 一条线深入研究，我会选 **vision-based tactile + sim-to-real + active inference** 的交叉，因为它是唯一能同时解决 sensor fidelity、data scaling、和 closed-loop dexterity 三个问题的方向。这篇文章给的就是这个 frontier 的 map。
