---
source_pdf: REPeat.pdf
paper_sha256: 2821418612b15795e4cedefe9ce03584ede48c2dd0c0f12e105041bb14e0e7a7
processed_at: '2026-08-11T22:44:05-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 REPeat 这篇 paper

## 一、这帮人到底在干什么

Cornell 有个 lab 专门做 assistive robotics，给那些手不能动的人造自动喂饭机器人。这个问题听着简单，实际上特别恶心——你想让机器人拿叉子叉一块香蕉，香蕉是软的、滑的、还会滚，叉子戳上去可能滑掉、可能只叉到一半、可能把香蕉挤烂。

这篇 paper 干的事情：**在真正下叉子之前，先让机器人在 simulation 里"试试看"各种准备工作**——把食物推到墙边、翻个面、切小一点——看看哪种准备工作能让后面真正叉食物的成功率最高，然后再在真实世界里执行。

就这么个事。

## 二、为什么这个问题难

Soft food 是个特别恶心的东西类别。你想想一个 plate 上可能同时有：

- Jell-O（一碰就碎的弹性体）
- Mashed potato（塑性流体，像泥巴）
- Rice（一堆小颗粒）
- Banana slice（弹性固体，还会滚）
- Spaghetti（长条形，会缠在一起）
- Mac & cheese（复合材料，面条 + 酱）

这些玩意儿的物理性质从 Newtonian fluid 到 granular solid 到 elastic solid 全跨度覆盖。你没法用一套 policy 搞定所有东西。

更恶心的是 **friction**。你想精确仿真"叉子戳进湿润的香蕉表面"这个动作，需要建模叉子金属和香蕉表皮之间的微观摩擦。这种 friction 在 MPM、FEM 里基本不可能精确 capture，因为摩擦系数受 moisture、temperature、contact area、surface roughness 一堆因素影响，sim 里的数字和真实世界能差一个数量级。

这就是 Sim2Real gap 的核心难题。

## 三、他们的核心 insight（这是整篇 paper 的灵魂）

他们发现了一个特别巧妙的 observation：

**"准备工作"和"真正叉食物"这两步，Sim2Real gap 是不对称的。**

具体来说：

- 准备工作（pushing、cutting、flipping）：Sim2Real gap **小**
  - 因为你只需要仿真出"食物大概被推到哪了"、"大概被切成几块"、"大概翻成什么姿态"
  - 不需要精确到毫米级，只要 macroscopic configuration 大致对就行
  - MPM 对这种 large deformation / fracture / granular flow 仿真得相当不错

- 真正叉食物（skewering、scooping、twirling）：Sim2Real gap **极大**
  - 因为叉子能不能叉住完全依赖精确的 friction 和 contact mechanics
  - Sim 里叉子可能叉住了，真实世界一抬手食物就滑掉了

所以他们想到一个绕过问题的办法：

**让 simulation 只负责"探索准备工作"这一步，不负责"预测叉食物成功与否"。叉食物的成功率用真实世界数据训练的 neural network（SPANet-soft）从图片里预测。**

这就形成了 Real2Sim2Real 的 loop：

1. 真实相机拍一张 plate 照片
2. SPANet-soft 看图预测直接叉的成功率
3. 如果成功率高（>70%）直接叉
4. 如果不高，进入 Real2Sim2Real loop：
   - **Real2Sim**: 把真实食物的 3D shape 重建出来扔进 simulation
   - **Sim**: 在 simulation 里试各种准备工作，看每个准备工作执行完食物变成什么样
   - **Sim2Real**: 把 simulation 结果 render 成 photorealistic 图片
   - **再 evaluate**: SPANet-soft 看这些 render 出来的图片，预测叉食物成功率
   - 选成功率提升最大的那个准备工作，在真实世界执行

这个设计的精妙之处：simulation 做它擅长的（探索不可逆动作——cutting 切了就没法 undo），real data 做它擅长的（从 visual appearance 预测成功率），各司其职，避免了直接仿真 friction 这个老大难问题。

## 四、具体技术拆解

### 4.1 SPANet-soft：看图预测叉食物成功率

这是个 data-driven 的 surrogate model。输入一张 plate 的 RGB 图片，输出三个数字：skewering、scooping、twirling 各自的成功率。

架构上：
- 用 Grounded-SAM 做食物检测和 segmentation（替代原来 SPANet 的 RetinaNet）
- 对每个 food item crop 出 288×288 的图
- 加两个 one-hot vector：environment（isolated vs. 靠墙/被包围）+ bite-size（bite-sized vs. not）
- 网络输出 3×1 vector，用 smooth L1 loss 训练

为什么需要 environment 和 bite-size 这两个额外 input？因为同一个香蕉片，周围空旷和靠墙时叉起来的难度完全不同；bite-size 的食物和非 bite-size 的策略也不同。

训练数据是真实机器人叉 10 种食物收集的 empirical success rate。

### 4.2 Real2Sim: 怎么把真实食物变成 3D mesh

传统方法（Poisson reconstruction、alpha shapes）太慢，而且 depth sensor 在湿润反光的食物表面上噪声极大。

他们的方法：
1. 用 DepthAnything 做单目深度估计（比 depth sensor 鲁棒得多）
2. 拿一个 template quadrilateral mesh（平面的网格）
3. 把 depth map 当 displacement map，让每个 vertex 沿自己的 normal 方向位移

公式长这样：

$$p_i' = p_i + D(u_i, v_i) \cdot \hat{n}_i$$

变量含义：
- $p_i$：template mesh 上第 i 个 vertex 的原始位置 $(x_i, y_i, z_i)$
- $D(u_i, v_i)$：depth map 在 image coordinate $(u_i, v_i)$ 处的值
- $\hat{n}_i$：vertex $p_i$ 处的 surface normal（单位向量）
- $p_i'$：deformation 后的新位置

**直觉**：就像你拿一张平的橡皮膜，每个点往下按不同的深度，按出一个食物的形状。比 Poisson reconstruction 快很多，因为不需要解全局优化问题。

### 4.3 Sim: MPM 仿真准备工作

MPM（Material Point Method）是一种 hybrid Lagrangian-Eulerian 方法。简单说：

- **Particles**（Lagrangian）：每个 particle 携带物质属性（mass、velocity、deformation gradient $F$）
- **Grid**（Eulerian）：用于计算 momentum exchange 和 collision

每个 time step：
1. P2G：particles 的 mass 和 velocity 映射到 grid nodes
2. Grid 上计算 forces、更新 velocity
3. G2P：grid velocity 插值回 particles
4. Particles 更新位置

为什么选 MPM 而不是 FEM？因为 soft food 需要同时处理：
- Large deformation（banana 被 push）
- Fracture（Jell-O 被 cut）
- Granular behavior（rice）
- Fluid-like flow（mashed potato）

FEM 处理这些需要不断 re-mesh，计算量爆炸。MPM 天生支持这些现象。

他们用了三种 constitutive model：
- **Elastic**：Jell-O、banana、avocado、tofu、red velvet
- **Plastic**：mashed potato、oatmeal、rice
- **Elastoplastic**：mac & cheese、spaghetti

每种食物的 Young's modulus 和 Lamé constants 是 offline calibrated 的（这是 limitation，真实食物批次间差异大）。

Fork 和 plate 用 Signed Distance Field 建模为 rigid body，friction 用 Coulomb model。

**两个工程改进**：

1. **Adaptive Particle Sampling**：不同食物用不同 particle density。rice 需要高 density（小颗粒），mashed potato 中等，Jell-O 可以低一些。否则小食物 undersampled 精度不够，大食物 oversampled 显存爆掉。

2. **Render-on-Demand**：不在每个 time step 都 render，只在准备工作执行完后 render 一次 depth。大幅降低计算量，让 long-horizon 任务（cutting、flipping）变得可行。

### 4.4 Sim2Real: ControlNet 把 sim 结果变逼真

MPM 仿真物理准确但视觉不真实，render 出来的图像 CG 感很强。如果直接喂给 SPANet-soft，domain gap 太大，预测不准。

解决方案：用 ControlNet 做 image-to-image translation。

- Input: simulation 的 depth image + 食物类别的 text prompt
- Output: photorealistic RGB image

为每种食物类别训练一个 category-level ControlNet。比如 red velvet cake 的 prompt 是："red velvet; red brick; red cake; burgundy cake; dark red cake; brown cake; maroon cake; dark purple cake"——这么多种 synonym 是因为 VLM 对这种颜色描述敏感。

**直觉**：simulation 保证物理 configuration 正确（食物位置、形状对），ControlNet 保证 visual appearance 真实（texture、color 对），合起来既 physical-valid 又 visual-realistic，SPANet-soft 就能在熟悉的 domain 上做预测。

## 五、实验怎么做的

### 5.1 食物选择

10 种食物代表 5 个 property 维度的 extremes：

| 维度 | Low | High |
|------|-----|------|
| Elasticity | tofu | Jell-O |
| Plasticity | banana | mashed potato |
| Viscosity | rice | oatmeal |
| Texture | avocado (smooth) | red velvet (rough) |
| Shape | spaghetti (long) | mac & cheese (composite) |

15 个 plate 分三个 difficulty：
- Simple: 2 块食物，占 plate 40%
- Medium: 3 块，60%
- Hard: 5 块，80%

### 5.2 Hardware

三种 robot 验证 hardware-agnostic：
- Franka Emika Panda 7-DoF + Azure Kinect（frame-mounted）
- Kinova Gen3 6-DoF + RealSense D435（wrist-mounted）
- Kinova Gen3 7-DoF + RealSense D435（wrist-mounted）

Utensil 有两个 extra DoF：Pitch（scoop 动作）+ Roll（twirl 动作）。配 ATI Nano25 F/T sensor 检测 pushing 何时终止（碰到墙或其他食物）。

### 5.3 Decision Protocol

```
if SPANet-soft 预测成功率 > 70%:
    直接叉
else:
    在 sim 里试所有准备工作
    选 SPANet-soft 预测提升最大的那个
    在真实世界执行准备工作
    如果准备工作失败，retry 一次
    然后执行叉食物
```

70% 这个 threshold 来自 Gordon et al. prior work。

### 5.4 Success Metric

- 叉起来后食物在叉子上停留 ≥ 3 秒（转移到嘴的时间）
- 食物是 bite-sized（在两个 prior work 的 threshold 之间）

## 六、结果如何

**平均提升 27%**，在 15 个 plate × 10 种食物上验证。

Chi-square test 显著提升（p < 0.05）的食物：Jell-O、mashed potato、rice、oatmeal、non-bite-sized banana trunk、red velvet、mac & cheese、tofu。

不显著的：bite-sized banana slice（已经很高了）、spaghetti（twirling 本来就 work）、avocado（本来就好叉）。

每种准备工作的作用机制：

**Pushing**:
- 把 granular food（rice、mashed potato）推到一起，density 增加，scooping 时不容易滑掉
- 推到墙边，墙提供 back support 防止食物被推开

**Flipping**:
- 把 banana slice 从侧面翻到平面，flat surface 朝上，skewering 时不会滚走
- 改变 contact geometry

**Cutting**:
- 把 Jell-O 这种 fragile 食物切成 bite-size，减少 breakage
- 小块更容易稳定叉住

## 七、典型失败案例

1. **Fragile food 还是碎了**：即使 cutting 后 Jell-O 仍然碎裂
2. **Granular food 溢出 plate**：pushing 力度没控制好，rice 飞出 plate
3. **Perception 混淆**：red velvet cake 上的白色 cream 和旁边的 white rice 颜色太像，Grounded-SAM 把它们识别成一块食物

## 八、Limitations

Paper 自己提到的：
1. **Time-varying food properties**：食物放久了 moisture、viscosity 会变，但 system 用 fixed parameters
2. **VLM perception**：Grounded-SAM 需要 carefully crafted prompts，open-set detection VLM 可能更好
3. **Simulation computation**：MPM 计算量大，fidelity vs. speed trade-off

我能想到的额外问题：
4. **One-shot exploration**：每个准备工作只试一次就 evaluate，没有 sequential reasoning（推一下再推一下）
5. **Per-category ControlNet**：10 种食物训练 10 个 ControlNet，unseen food 需要重训
6. **Material parameter sensitivity**：offline calibrated 的参数对真实食物批次差异不鲁棒
7. **Friction modeling**：虽然绕过了 bite acquisition 的 friction 仿真，但 pre-acquisition 中的 friction（pushing 时食物滑）仍依赖简单 Coulomb model

## 九、为什么这个范式 work

回到根本问题：为什么 Real2Sim2Real 适合这个 task？

因为这个 task 有一个特殊的 **decomposition property**：

- **Pre-acquisition**：exploration-friendly（sim 能 approximate），evaluation-hard（需要看真实叉的结果）
- **Bite acquisition**：evaluation-friendly（从图片能预测），exploration-hard（sim 仿真 friction 不准）

这种 decomposition 让 simulation 和 real data 各司其职。如果没有这个 decomposition，比如要直接 simulate bite acquisition 的 friction，系统会 fail；如果只用 real data without simulation，无法 explore 不可逆的 pre-acquisition（会浪费食物）。

**可推广的 insight**：当一个 task 可以 decompose 成"exploration-friendly but evaluation-hard"和"evaluation-friendly but exploration-hard"两部分时，Real2Sim2Real 是 natural fit。

其他可能的 application：
- **Surgical robotics**：tissue retraction（sim 可探索）+ suture success（从图像评估）
- **Cloth manipulation**：unfolding（sim 可探索）+ folding precision（从图像评估）
- **Cooking**：ingredient preparation（sim 可探索）+ doneness（从图像评估）

## 十、我的 take

这篇 paper 的贡献不是单个 module 的技术突破——MPM、ControlNet、DepthAnything 都是现有技术。它的贡献是**系统集成的 insight**：识别了 pre-acquisition 和 bite acquisition 之间不对称的 Sim2Real gap，然后用 simulation 和 real data 各自的优势组合起来。

27% 的 average improvement 在 15 plates × 10 food types 上验证了这个 insight 的有效性。这个数字在 assistive robotics 领域是有实际意义的——每多成功一次，care recipient 的 dignity 和 caregiver 的负担就改善一点。

技术亮点：
1. MLS-MPM + adaptive sampling + render-on-demand 让复杂 food simulation 在 single GPU 上可行
2. Template mesh deformation for real-time 3D reconstruction
3. ControlNet bridging Sim2Real visual gap
4. Robot-and-camera-agnostic 设计

Limitation 也明确：fixed material parameters、one-shot exploration、per-category ControlNet training。这些指明了 future work 方向。

总的来说，这是一篇 elegant 的 system paper，给 Real2Sim2Real paradigm 提供了一个 concrete instantiation，对 assistive robotics 和 robotics manipulation 都有启发意义。

---

## 参考链接

**Project & Code**:
- REPeat project: https://emprise.cs.cornell.edu/repeat
- FluidLab: https://fluidlab.github.io/
- PlasticineLab: https://plasticinelab.github.io/

**Foundation Models**:
- DepthAnything: https://depth-anything.github.io/
- Grounded-SAM: https://github.com/IDEA-Research/Grounded-Segment-Anything
- ControlNet: https://github.com/lllyasviel/ControlNet

**Simulation Methods**:
- MLS-MPM: https://yuanming.hu/2019/mls-mpm/
- DiSECt: https://arxiv.org/abs/2108.13024

**Related Feeding Papers**:
- FLAIR (RSS 2024): https://flair-project.github.io/
- SPANet: https://github.com/personalrobotics/foodacquisition

---

# REPeat: Real2Sim2Real Approach for Soft Food Pre-acquisition 技术深度解析

## 一、Core Motivation 与 Problem Framing

这篇paper来自Cornell的EMPRIZE Lab（Tapomayukh Bhattacharjee组），针对的是robot-assisted feeding中一个被严重underexplored的子问题：**soft diet food的bite acquisition**。

背景数据非常stark：全球16%人口（约1.3 billion）有significant disability，其中142 million有severe mobility limitation。Dysphagia（吞咽困难）在ALS晚期和Parkinson's患者中prevalence高达80%。Soft diet是这些患者的刚需，但soft food的rheological properties跨度极大——从Newtonian fluid（water）、Bingham plastic（mashed potato）、pseudoplastic（oatmeal）、granular solid（rice）到elastic solid（banana, Jell-O），甚至composite（mac & cheese）。这种spectral diversity让传统的manipulation policy很难generalize。

**Key Insight的层次结构**（这是我理解这篇paper最关键的intuition）：

1. **Sim2Real gap是不对称的**：Bite acquisition（skewering, scooping, twirling）的Sim2Real gap极大，因为fork-food interaction依赖精确的friction modeling（moist banana表面 vs. fork tines的微观摩擦），这在MPM/FEM中几乎不可能精确capture。但是pre-acquisition actions（pushing, cutting, flipping）的Sim2Real gap较小，因为它们关注的是food的**macroscopic configuration**（food是否靠墙、是否被cut成bite-size、是否flip到flat surface），而非精确的接触力学。

2. **Configuration sufficiency**：Pre-acquisition后的food configuration（位置、形状、size、与wall的距离）包含足够信息来预测bite acquisition成功率，无需精确仿真skewering过程本身。

3. **Real2Sim2Real的分工**：Simulation负责exploration（探索pre-acquisition action space），real-world data-driven model（SPANet-soft）负责evaluation（评估结果configuration的bite acquisition success rate）。这避免了直接在sim中预测bite acquisition的friction难题。

这个设计的精妙之处：它没有试图解决Sim2Real gap，而是**绕过**了它——用simulation做它擅长的（exploration of irreversible actions），用real data做它擅长的（success prediction from visual appearance）。

参考：
- Project page: https://emprise.cs.cornell.edu/repeat
- FLAIR (RSS 2024): https://flair-project.github.io/

---

## 二、System Architecture 总览

整个pipeline的flow如下：

```
RGB Image → SPANet-soft (success prediction)
              ↓
         success rate > 70%? 
         ├── Yes → direct bite acquisition
         └── No → Real2Sim2Real loop:
                   ├── Real2Sim: DepthAnything + template mesh deformation
                   ├── Sim: MLS-MPM simulation of pre-acquisition actions
                   ├── Sim2Real: ControlNet renders photorealistic image
                   └── SPANet-soft re-evaluates → pick best action
```

Action space设计：
- **Pre-acquisition**: pushing（4个directions: ±x, ±z）、cutting（downward + flick）、flipping（perpendicular to major axis + slight upward）
- **Bite acquisition**: skewering、scooping、twirling（这3个parameterization继承自FLAIR）

这里有个设计决策值得注意：pre-acquisition action是**one-shot rollout**（每个action执行一次然后evaluate），而不是long-horizon planning。这降低了search space复杂度，但也限制了系统能力——paper Discussion部分提到了这个limitation。

---

## 三、SPANet-soft: Data-Driven Success Prediction

### 3.1 架构改进

SPANet-soft是在SPANet（Feng et al., IROS 2019）基础上的改进，针对soft diet food的特性：

| Component | SPANet (original) | SPANet-soft |
|-----------|-------------------|-------------|
| Food detection | RetinaNet | Grounded-SAM |
| Input size | 224×224×3 | 288×288×3 |
| Environment encoding | - | One-hot (Isolated / Wall) |
| Bite-size encoding | - | One-hot (bite-sized / not) |
| Output | Success rate per action | 3×1 vector [skewering, scooping, twirling] |

### 3.2 Input Representation

输入有三个streams：
1. **Visual**: 288×288×3 RGB crop（根据bounding box）
2. **Environment vector** (2×1): Isolated vs. Wall——区分food是否靠plate edge或被其他food包围
3. **Bite-size vector** (2×1): 基于segmentation mask的average height和area估算volume

这些vectors与image feature vector concatenate后预测3×1 success rate。

### 3.3 Training

- **Loss**: Smooth L1 loss（比L2对outlier更鲁棒）
  
  $$\mathcal{L}_{smoothL1}(y, \hat{y}) = \begin{cases} 0.5(y - \hat{y})^2 & \text{if } |y - \hat{y}| < 1 \\ |y - \hat{y}| - 0.5 & \text{otherwise} \end{cases}$$

  其中 $y$ 是ground truth success rate（从real robot实验收集），$\hat{y}$ 是predicted success rate。

- **Dataset**: 10种food types的real-robot bite acquisition empirical success rates

**Intuition**: 这个module本质上是一个learned surrogate model，它不需要理解physics，只需要从visual appearance + context中correlate到historical success rates。这是Real2Sim2Real中"Real"的那一端。

参考：
- SPANet: https://github.com/personalrobotics/foodacquisition
- Grounded-SAM: https://github.com/IDEA-Research/Grounded-Segment-Anything
- Smooth L1 loss (Fast R-CNN): https://arxiv.org/abs/1504.08083

---

## 四、Real2Sim: Mesh Reconstruction

### 4.1 为什么不用传统方法

传统3D reconstruction方法（Poisson surface reconstruction、alpha shapes、ball pivoting）在food场景下有两个问题：
1. **Speed**: 太慢，无法real-time
2. **Noise sensitivity**: food表面的moisture和reflectiveness导致depth sensor噪声极大

### 4.2 DepthAnything + Template Mesh Deformation

解决方案是**monocular depth estimation**（DepthAnything）+ **template mesh deformation**。

**核心公式**：

给定template quadrilateral mesh $\mathcal{M}$，其vertices集合 $V = \{p_1, p_2, ..., p_n\}$，其中每个vertex $p_i = (x_i, y_i, z_i) \in \mathbb{R}^3$。

给定depth image $D: \mathbb{R}^2 \to \mathbb{R}$，其中 $D(u, v)$ 表示在image coordinate $(u, v)$ 处的displacement值。

假设mesh $\mathcal{M}$ 和 depth image $D$ 共享相同resolution，则每对 $(u_i, v_i)$ 直接映射到 $(x_i, y_i)$。

**Vertex update rule**:

$$p_i' = p_i + D(u_i, v_i) \cdot \hat{n}_i$$

变量解释：
- $p_i$：original template mesh的vertex位置
- $D(u_i, v_i)$：depth map在vertex对应image coordinate处的displacement值
- $\hat{n}_i$：vertex $p_i$处的surface normal vector（单位向量）
- $p_i'$：deformation后的新vertex位置

**Intuition**: 这个方法本质上是"displacement mapping"——template mesh是一个flat quad mesh，每个vertex沿着自己的normal方向位移，位移量来自monocular depth estimation。这比Poisson reconstruction快得多，因为不需要求解global optimization。

### 4.3 为什么这个方法work

- Template mesh提供一个topologically consistent的base
- DepthAnything的monocular depth estimation比RGB-D sensor在reflective/moist food表面更鲁棒（因为它是learned from large-scale data，能处理appearance cues）
- Normal-direction displacement避免了tangential distortion

参考：
- DepthAnything (CVPR 2024): https://depth-anything.github.io/
- DepthLab (UIST 2020): https://research.fb.com/publications/depthlab-real-time-3d-interaction-with-depth-maps-for-mobile-augmented-reality/

---

## 五、Sim: MLS-MPM Simulation

### 5.1 为什么选MPM

Food simulation有两个mainstream approaches：

| Method | 优势 | 劣势 |
|--------|------|------|
| FEM (mesh-based) | Physics accuracy高 | 需要continuous re-meshing，计算昂贵，不适合granular/fluid |
| MPM (mesh-free) | 支持large deformation、fracture、multi-physics coupling | 计算量大，但可parallelize |

Paper选择MLS-MPM（Moving Least Squares Material Point Method），因为soft diet food需要同时处理：
- **Deformation**（banana被push）
- **Fracture**（Jell-O被cut）
- **Multi-physics**（mashed potato的plastic flow + rice的granular behavior）

### 5.2 MLS-MPM原理简述

MPM的核心思想是hybrid Lagrangian-Eulerian：
1. **Particles (Lagrangian)**: 携带material properties（mass, velocity, deformation gradient $F$, volume $J$）
2. **Grid (Eulerian)**: 用于计算momentum exchange和collision

每个time step的流程（简化版）：

```
1. P2G (Particle to Grid): 将particle的mass和velocity映射到grid nodes
   m_i = Σ_p m_p * w_ip
   (mv)_i = Σ_p m_p * v_p * w_ip
   
2. Grid momentum update: 计算forces, 更新grid velocity
   v_i^{t+1} = v_i^t + Δt * f_i / m_i
   
3. G2P (Grid to Particle): 将grid velocity插值回particles
   v_p^{t+1} = Σ_i v_i^{t+1} * w_ip
   
4. Particle advection: 更新particle position
   x_p^{t+1} = x_p^t + Δt * v_p^{t+1}
```

其中 $w_{ip}$ 是weight function（MLS-MPM用moving least squares shape functions），$m_p$ 是particle mass，$v_p$ 是particle velocity，$f_i$ 是grid node上的force（包括elastic force、external force、collision force）。

### 5.3 Constitutive Models

Paper用了3种constitutive model来approximate不同food types：

| Food Type | Constitutive Model | 来源 |
|-----------|-------------------|------|
| Elastic solids (Jell-O, banana, avocado, tofu, red velvet) | Elastic model | MLS-MPM [Hu et al. 2018] |
| Plastic solids (mashed potato, oatmeal, rice) | Plastic model | MLS-MPM |
| Composite (mac & cheese, spaghetti) | Elastoplastic model | PlasticineLab [Huang et al. 2021] |

每种model由Young's modulus $E$ 和 Lamé constants ($\lambda, \mu$) 参数化。这些参数固定地calibrate到FluidLab的preset values，而非online identification——这是Discussion提到的limitation之一。

### 5.4 两个工程改进

**Adaptive Particle Sampling Module**:
- 传统MPM用uniform density，导致小食物用太多particle，GPU memory爆
- Adaptive: 每种food type分配特定density
- 效果：能在single GPU上simulate多种food interaction

**Render-on-Demand Module**:
- 传统simulator每帧都render，浪费计算
- 改进：只在pre-acquisition action完成后render depth
- 效果：long-horizon task（cutting, flipping）变得feasible

### 5.5 Rigid-Soft Interaction

Fork和plate用Signed Distance Field (SDF)建模为rigid body：

$$\text{SDF}(\mathbf{x}) = \min_{\mathbf{y} \in \partial \Omega} \|\mathbf{x} - \mathbf{y}\|$$

其中 $\partial \Omega$ 是rigid body的boundary surface。SDF值表示点 $\mathbf{x}$ 到rigid body表面的有符号距离。

**Friction model**: Coulomb friction
$$f_{friction} \leq \mu_c \cdot f_{normal}$$

其中 $\mu_c$ 是friction coefficient，$f_{normal}$ 是法向接触力。通过计算SDF的surface normals来获得contact direction。

参考：
- MLS-MPM: https://yuanming.hu/2019/mls-mpm/ (siggraph 2018, 但website有更直观的explanation)
- FluidLab: https://fluidlab.github.io/
- PlasticineLab: https://plasticinelab.github.io/
- MPM original (Sulsky et al. 1993): https://doi.org/10.1016/0045-7825(93)90145-4

---

## 六、Sim2Real: ControlNet for Photorealistic Rendering

### 6.1 问题

MPM simulation的physics accuracy高，但**visual realism差**——渲染出来的depth/RGB看起来像CG，与real food appearance差距大。如果直接feed SPANet-soft simulation rendering，会因为domain gap导致prediction不准。

### 6.2 解决方案：ControlNet

ControlNet是diffusion model的conditional generation framework，这里用它来bridging visual gap。

**Input**:
1. **Depth image** from simulation（提供3D geometry）
2. **Text prompt**（food category name + properties）

**Output**: Photorealistic RGB image of the simulated food configuration

**Training**:
- 为每种food category训练一个category-level ControlNet
- Dataset: real RGB images of food items on plate
- Prompts: 手工编写（例如red velvet cake的prompt: "red velvet; red brick; red cake; burgundy cake; dark red cake; brown cake; maroon cake; dark purple cake"）

### 6.3 为什么这个design work

**Intuition**: ControlNet decouples了geometry（from simulation depth）和appearance（from learned texture generation）。Simulation保证physical configuration正确，ControlNet保证visual appearance realistic，合起来既physical-valid又visual-realistic，让SPANet-soft能在熟悉的domain上做prediction。

这是一个巧妙的"domain adaptation"——与其fine-tune SPANet-soft去理解simulation rendering，不如render simulation成SPANet-soft熟悉的photorealistic image。

参考：
- ControlNet: https://github.com/lllyasviel/ControlNet
- ControlNet paper: https://arxiv.org/abs/2302.05543

---

## 七、Evaluation 设计

### 7.1 Food Selection

10种food代表5个property维度的extremes：
- Elasticity: Jell-O (high) vs. tofu (low)
- Plasticity: mashed potato (high) vs. banana (low)
- Viscosity: oatmeal (high) vs. rice (low)
- Texture: avocado (smooth) vs. red velvet (rough)
- Shape: spaghetti (long) vs. mac & cheese (composite)

15个plates分3个difficulty levels：
- **Simple**: 2 pieces, 40% plate coverage
- **Medium**: 3 pieces, 60% coverage
- **Hard**: 5 pieces, 80% coverage

这种设计能test系统在不同clutter level下的robustness。

### 7.2 Hardware Agnostic设计

3种embodiment验证robot-and-camera-agnostic：
1. Franka Emika Panda 7-DoF + Azure Kinect (frame-mounted)
2. Kinova Gen3 6-DoF + RealSense D435 (wrist-mounted)
3. Kinova Gen3 7-DoF + RealSense D435 (wrist-mounted)

Utensil有2个extra DoFs：Pitch (scoop) + Roll (twirl)，配合ATI Nano25 F/T sensor检测pushing终止。

### 7.3 Decision Protocol

```
if SPANet-soft predicts success > 70%:
    direct bite acquisition
else:
    try pre-acquisition action (highest predicted improvement)
    if pre-acquisition fails: retry once
    then bite acquisition
```

这个threshold来自Gordon et al. [13]的prior work。

### 7.4 Success Metric

- **Acquisition success**: food stays on fork ≥ 3 seconds（mouth transfer time）
- **Bite-size check**: 在[Gordon 13]和[FLAIR 11]的quantitative threshold之间

---

## 八、实验结果分析

### 8.1 整体提升

**平均提升27%** across all 15 plates。

Chi-square significance test ($p < 0.05$)显著提升的food types：
- Jell-O, mashed potato, rice, oatmeal, non-bite-sized banana trunk, red velvet cake, mac & cheese, tofu

不显著的（已很高或pre-acquisition无帮助）：
- Bite-sized banana slice, spaghetti, avocado

### 8.2 各action的mechanism

**Pushing**:
- Consolidates granular food（rice, mashed potato）→ 增加scooping success
- Moves food toward wall → 防止slip away
- 机制：增大local density + 提供back support

**Flipping**:
- 暴露flat surface for skewering
- 例：banana slice从side翻到flat → 防止skewering时rolling
- 机制：改变contact geometry

**Cutting**:
- 把fragile food（Jell-O）切成bite-size
- 减少breakage + 减少fall-off probability
- 机制：减少size → 减少gravity-induced stress

### 8.3 Failure Cases

3类typical failure：
1. **Fragile food breakage**: 即使cutting后仍碎裂
2. **Granular food spillage**: pushing过猛溢出plate
3. **Perception confusion**: 多种food混合，VLM识别错误（例：white rice + white cream on red velvet cake被识别为一个piece）

这些failure揭示了系统的bottleneck：perception（VLM局限）和physics simulation（noise accumulation）。

---

## 九、与相关工作的Positioning

### 9.1 vs. FLAIR (RSS 2024)

FLAIR用VLM做long-horizon acquisition planning，优点是modular + handles user preference。缺点是**VLM缺乏physics understanding**，会生成unrealistic action sequence。

REPeat的改进：用physics-informed MPM simulation替代VLM的"imagination"，action sequence经过physical validation。但REPeat的action space更小（FLAIR支持更多action types），且不做long-horizon planning。

### 9.2 vs. Data-Driven Dynamics Models

Learned forward predictive models（particle-based, pixel-based, keypoint-based, latent-based）的问题：
- Long-horizon error accumulation
- 需要大量real-world interaction data（破坏food, 浪费）

REPeat用physics-based simulation替代learned dynamics，避免data collection cost和error accumulation，但需要预先calibrate material parameters。

### 9.3 vs. Differentiable Simulation (DiSECt)

DiSECt用differentiable FEM做cutting simulation，physics accuracy高但computation expensive + 需要re-meshing + 不适合granular/fluid。

REPeat用MPM，trade-off是per-step accuracy换genericity（支持多种food type）。

参考：
- DiSECt: https://arxiv.org/abs/2108.13024
- FLAIR: https://flair-project.github.io/

---

## 十、Limitations & Future Work

Paper Discussion提到的3个limitation：

1. **Time-varying food properties**: Room temperature下food的moisture、viscosity随时间变化，但system用fixed parameters。Future: online parameter identification。

2. **Food perception**: Grounded-SAM需要carefully crafted prompts（例：red velvet cake的8种synonym）。Future: open-set detection VLMs。

3. **Food simulation**: MPM compute-heavy, fidelity vs. speed trade-off。Future: 更好的simulation backend。

我能想到的额外limitation：

4. **One-shot action exploration**: 只执行一次pre-acquisition就evaluate，没有sequential reasoning。如果有chain of pre-acquisition actions会更好，但search space爆炸。

5. **ControlNet的domain gap**: 每种food category单独训练ControlNet，generalization到unseen food需要re-training。

6. **Material parameter sensitivity**: MPM的Young's modulus和Lamé constants是offline calibrated的，real food的batch variation大。

7. **Friction modeling**: 虽然绕过了bite acquisition的friction仿真，但pre-acquisition中的friction（pushing时food slip）仍然依赖Coulomb model，可能在某些food上不准。

---

## 十一、Build Intuition: 为什么这个approach有效

回到最根本的问题：为什么Real2Sim2Real这个范式适合这个task？

我的理解是：**这个task有一个特殊的decomposition property**——pre-acquisition action的effect可以在simulation中approximately capture（不需要精确，只需大致configuration正确），而bite acquisition的成功率可以从visual appearance中learned（不需要physics simulation）。

这种decomposition让simulation和real data各司其职：
- Simulation: cheap exploration of irreversible actions（cutting不能undo）
- Real data: learn visual-success correlation（SPANet-soft）

如果没有这个decomposition，比如要直接simulate bite acquisition的friction，系统会fail。如果只用real data without simulation，无法explore irreversible pre-acquisition actions（会浪费food且不可逆）。

**Generalizable insight**: 当一个task可以decompose成"exploration-friendly but evaluation-hard"（pre-acquisition）和"evaluation-friendly but exploration-hard"（bite acquisition）两部分时，Real2Sim2Real是natural fit。

这种decomposition在其他robotics task中也可能存在：
- Surgical robotics: tissue retraction (explorable) + suture (evaluation from appearance)
- Cloth manipulation: unfolding (explorable) + folding precision (evaluation)
- Cooking: ingredient preparation (explorable) + doneness (evaluation from appearance)

---

## 十二、技术细节的额外补充

### 12.1 Adaptive Sampling的intuition

Uniform sampling的问题：如果所有food用相同particle density，要么小食物undersampled（精度不够），要么大食物oversampled（memory爆）。

Adaptive sampling的核心：
$$\rho_{food\_type} = f(\text{size}, \text{stiffness}, \text{deformation\_magnitude})$$

每种food type有自己的density，根据其物理特性调整。例如：
- Rice: 高density（小颗粒，需要精细granular behavior）
- Mashed potato: 中density（soft bulk，deformation主导）
- Jell-O: 低density（brittle，但fracture需要足够resolution）

### 12.2 MLS-MPM vs. Standard MPM

MLS-MPM的关键改进是用**moving least squares shape functions**替代standard B-spline weights。好处是：
- 更高order accuracy
- 更稳定的large deformation
- 更少numerical noise

公式上，weight function从cubic B-spline改为MLS shape function，但具体的数学细节需要参考Hu et al. 2018的SIGGRAPH paper。

### 12.3 ControlNet的training trick

Paper提到为每种food category训练一个category-level ControlNet。这意味着有10个独立的ControlNet models。训练数据是real RGB images + 对应的depth。

Text prompt的作用是提供texture and color prior，例如"red velvet cake"告诉model应该生成burgundy/dark red的texture。这比单纯用depth condition更能生成food-specific appearance。

可能的improvement：用single multi-category ControlNet + category embedding，减少model数量。但需要更大数据集。

---

## 十三、延伸思考

### 13.1 这个paradigm在其他领域的应用

REPeat的Real2Sim2Real + physics-informed exploration paradigm可能在以下领域有应用：

1. **Medical robotics**: 组织manipulation前的pre-positioning
   - Simulation探索retraction configuration
   - Real data评估grasping success

2. **Agile robotics**: 复杂terrain前的foot repositioning
   - Simulation探索terrain interaction
   - Real data评估stability

3. **Dexterous manipulation**: 多指re-grasp前的object re-orientation
   - Simulation探索pushing/poking sequence
   - Real data评估grasp success

### 13.2 与Foundation Models的结合

当前system的perception（Grounded-SAM）和rendering（ControlNet）用了foundation models，但planning部分是search-based。如果用VLM（如GPT-4V）做high-level planning + MPM做physics validation，可能形成更强的hybrid system。

挑战是VLM的output需要translate成MPM-compatible的action specification，且需要iterative refinement when simulation shows plan is infeasible。

### 13.3 与Differentiable Simulation的结合

如果MPM是differentiable的（如PlasticineLab的differentiable MPM），可以做gradient-based optimization of pre-acquisition action parameters，而不是one-shot rollout + evaluate。这会大幅减少exploration cost。

但differentiable MPM的memory cost更高，且gradient through fracture/deformation可能unstable。

参考：
- Differentiable MPM: https://plasticinelab.github.io/
- DiffSkill: https://arxiv.org/abs/2111.00558

---

## 十四、总结

REPeat是一个elegant的system paper，它的核心贡献不是单个module的技术突破，而是**系统集成的insight**：通过识别pre-acquisition和bite acquisition的asymmetric Sim2Real gap，巧妙地用simulation和real data各自的优势。27%的average improvement在15 plates × 10 food types的evaluation上验证了这个insight的有效性。

技术亮点包括：
1. MLS-MPM + adaptive sampling + render-on-demand，让复杂food simulation在single GPU上feasible
2. Template mesh deformation for fast 3D reconstruction
3. ControlNet bridging Sim2Real visual gap
4. Robot-and-camera-agnostic hardware design

Limitations也明显：fixed material parameters、one-shot exploration、per-category ControlNet training。这些指明了future work的方向：online parameter identification、sequential planning with simulation-in-the-loop、multi-category generative models。

这个工作对assistive robotics community有实际意义，对robotics manipulation的broader community提供了一个Real2Sim2Real paradigm的concrete instantiation。

---

## 参考链接汇总

**Project & Code**:
- REPeat project: https://emprise.cs.cornell.edu/repeat
- FluidLab: https://fluidlab.github.io/
- PlasticineLab: https://plasticinelab.github.io/
- RCareWorld: https://rcareworld.github.io/

**Foundation Models**:
- DepthAnything: https://depth-anything.github.io/
- Grounded-SAM: https://github.com/IDEA-Research/Grounded-Segment-Anything
- ControlNet: https://github.com/lllyasviel/ControlNet

**Simulation Methods**:
- MLS-MPM: https://yuanming.hu/2019/mls-mpm/
- DiSECt: https://arxiv.org/abs/2108.13024
- MPM original: https://doi.org/10.1016/0045-7825(93)90145-4

**Related Feeding Papers**:
- FLAIR (RSS 2024): https://flair-project.github.io/
- SPANet: https://github.com/personalrobotics/foodacquisition
- Gordon et al. CoRL 2023: https://arxiv.org/abs/2305.06622
