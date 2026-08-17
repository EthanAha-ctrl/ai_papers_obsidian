---
source_pdf: DexGarmentLab.pdf
paper_sha256: 447ce546d76be47536883a64aac1e9fc7ea1f214433bc6915d9acce2eab57f65
processed_at: '2026-08-03T20:08:01-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的 Karpathy，咱们抛开论文的八股文格式，直接在白板上聊聊这篇 DexGarmentLab 到底在玩什么把戏。

核心就一句话：**Garment manipulation 隯就鳯在状态空间是无穷维的，你不可能让机器背下所有衣服的形态。这篇 paper 的核心 trick 是利用衣服的“拓扑结构不变性”，把无穷维问题降维打击，拆成两个有界的问题去解。**

---

### 1. 环境的 Hack：怎么让灵巧手不“穿模”

做 deformable object simulation，最头疼的就是 PBD（Position-Based Dynamics）。PBD 本质是把衣服模拟成一堆珠子串起来的网。问题在于，珠子之间有缝，灵巧手的手指一过去，珠子就卡缝里了，或者直接穿过去，根本提不起来。

之前的 GarmentLab（这组人的前作，https://arxiv.org/abs/2411.01200）怎么解决的呢？简单粗暴，在夹爪尖端贴个隐形的红方块，直接用约束把衣服粒子焊在方块上。这对平行夹爪勉强能用。

但你想想灵巧手，Shadow Hand 有十根手指头，你贴十个方块？那就成了只要有一根手指蹭到衣服，整件衣服就被“焊”在那根手指上了，这完全违背物理直觉，衣服会像破布一样挂在手指上乱晃（paper 里叫 unnatural sagging）。

DexGarmentLab 的解法是放弃硬焊，改用“软约束”——四个物理参数调参：

**Adhesion（粘附力）**
$$\mathbf{f}_{\mathrm{adh}} = -k_{\mathrm{adh}} \cdot (\mathbf{x}_i - \mathbf{x}_j), \quad \mathrm{if} \ \|\mathbf{x}_i - \mathbf{x}_j\| < r_{\mathrm{adh}}$$
这里的 $\mathbf{x}_i$ 是衣服粒子的位置，$\mathbf{x}_j$ 是手指表面的位置。直觉上，这就是个**有距离限制的微型弹簧**。手指靠近衣服粒子时，产生吸引力；离远了（超过 $r_{\mathrm{adh}}$），力就消失。这样多根手指围拢时，每根手指贡献一点微弱的吸力，合力就足以把衣服捏起来，而且不会出现“一指挂衣服”的灵异事件。

**Friction（摩擦力）**
$$\mathbf{f}_{\mathrm{fric}} = -\mu \cdot \|\mathbf{f}_n\| \cdot \frac{\mathbf{v}_t}{\|\mathbf{v}_t\|}$$
$\mu$ 是摩擦系数，$\mathbf{f}_n$ 是法向力，$\mathbf{v}_t$ 是切向相对速度。这就是经典库仑摩擦，防止衣服从手指上滑出去。

**Particle-Adhesion-Scale & Particle-Friction-Scale**
这两个是衣服粒子之间的内耗参数。PBD 模拟折叠时，粒子自碰撞很容易导致“爆炸”或者“高频抖动”。这两个 scale 相当于给衣服内部加了点阻尼和粘性，让折叠后的状态能稳定保持住，不会一松手就散架。

说白了，这部分工作就是 **Isaac Sim 里的高级调参**，把物理仿真从“不可用”调到“勉强像那么回事”。这是一种工程上的妥协，PBD 用于大件衣服（容易变形），FEM 用于手套帽子（体积感强、弹性足）。

---

### 2. 数据飞轮：One-shot 广播机制

这是这篇 paper 最漂亮的工程直觉。

Garment manipulation 训练 policy 需要海量数据。遥操作太慢，RL 采样效率太低。作者发现了一个巨大的先验知识：**同类衣服的结构是一样的**。不管这件 T 恤是红是蓝，是长是短，它都有领口、左袖口、右袖口、下摆。这些语义点在同类衣服的不同实例之间存在拓扑对应。

于是他们搞了个 **Garment Affordance Model (GAM)**。
基于他们之前的 UniGarmentManip（https://j9877554.github.io/uni-garment-manip/），用 PointNet++ 提特征，用 InfoNCE loss 做 contrastive learning。

InfoNCE 的公式核心：
$$\mathcal{L} = -\log \frac{\exp(\mathrm{sim}(\mathbf{f}_{p_i}, \mathbf{f}_{p_j^+})/\tau)}{\sum_{k=1}^{K} \exp(\mathrm{sim}(\mathbf{f}_{p_i}, \mathbf{f}_{p_k})/\tau)}$$
- $\mathbf{f}_{p_i}$: 衣服 A 上某个点（比如领口）的特征。
- $\mathbf{f}_{p_j^+}$: 衣服 B 上对应点（也是领口）的特征。这是正样本。
- $\mathbf{f}_{p_k}$: 衣服 B 上其他乱七八糟的点（袖口、下摆等）。这些是负样本。
- $\tau$: 温度系数，控制相似度分布的尖锐程度。

直觉上，这就是强迫模型在 high-dimensional feature space 里学会“语义对齐”。衣服 A 的领口点和衣服 B 的领口点在 feature space 里要挨得近，跟其他点要拉远。

**数据采集的飞轮怎么转？**
1. 人类用 LeapMotion（https://www.ultraleap.com/product/leap-motion-controller/）给一件衣服做一次 demo，记录下“抓哪几个点”、“手摆什么姿势”、“怎么动”。
2. 换一件同类新衣服，把它扔在随机位置。GAM 提取这件新衣服的点云，跟 demo 衣服的点云做 feature 匹配（dot product 找最大相似度）。
3. 找到了新衣服上对应的“领口”、“袖口”点。
4. 机器人用 IK 控制双臂移动到这些点，双手用 PD 控制保持 demo 时的 hand pose。
5. 执行动作序列，同时根据新衣服的长短自适应调整抬起高度（比如挂衣服时，要根据衣服中心对齐衣架）。

这相当于**用结构对应性做印章，把一次 demo 盖章盖到几千件衣服上**。采集 100 条 demo 只要 1 分钟左右，成功率 90%+。这比遥操作快了不知多少个数量级。

---

### 3. HALO 算法：为什么要分两层？

直接上 Diffusion Policy (DP, https://diffusion-policy.cs.columbia.edu/) 或者 DP3 (https://arxiv.org/abs/2403.03954) 为什么不行？

因为它们是 end-to-end 的。你给它一堆轨迹，它学的是“在当前 observation 下，action 的分布”。一旦来了一件训练集里没见过的奇形怪状的衣服，或者衣服皱得不成样子，它的 latent space 就懵了，它不知道该去抓哪。结果就是手在空中乱挥，或者抓到衣服中间的一块布，根本没法完成折叠或悬挂。

HALO (Hierarchical gArment manipuLation pOlicy) 的解法是**解耦**：
- **第一层：GAM 定位**。先用 GAM 找到“该去抓哪”，输出 affordance score。这把“语义推理”从轨迹生成里剥离出来。模型只需要在点云上找对应点，不需要管手怎么动。
- **第二层：SADP 生成轨迹**。这是个改版的 Diffusion Policy，叫 Structure-Aware Diffusion Policy。

SADP 的关键在于 condition $\mathbf{s}$ 的构造。它不是简单塞个图像进去，而是把场景信息结构化：
1. 衣服点云 + GAM 输出的左右手 affordance feature → PointNet++ → $F_{\mathrm{garment}}$
2. 交互物体点云（如衣架）→ MLP → $F_{\mathrm{object}}$
3. 拼起来得到 $F_{\mathrm{scene}}$
4. 再加上环境点云 feature 和机器人关节状态 feature，组成完整的 condition $\mathbf{s}$。

Diffusion 的去噪过程：
$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{s}) \right) + \sigma_t \mathbf{z}$$
- $\mathbf{x}_t$: 加了噪声的 action（60维 DoF）。
- $\boldsymbol{\epsilon}_\theta$: 神经网络预测的噪声。
- $\mathbf{s}$: 就是上面构造的那个包罗万象的 condition。
- $\alpha_t, \bar{\alpha}_t$: 调度噪声的系数。

**Intuition 在哪？**
SADP 在生成轨迹时，它的 condition 里包含了 $F_{\mathrm{garment}}$，也就是衣服的形状和结构特征。这意味着，**模型在“比划”怎么动手的时候，它能“看到”衣服长什么样**。如果裤子很长，它生成的抬起高度就高一点；如果衣架偏了，它的轨迹就往衣架那边偏一点。这就解决了纯 IL 算法“死记硬背轨迹无法外推”的毛病。

Paper 里的 Fig. 7 说明了这一点：
- 拿掉 GAM，手根本找不到该抓哪，抓到了袖子中间。
- 拿掉 SADP 换成 DP3，手抓对了位置，但轨迹是死板板的，挂衣服时高度不够，挂不上去。

---

### 4. 实验数据的直觉

Table 2 的数据很硬。在 Hang Coat 任务上，DP 只有 52% 成功率，HALO 达到了 90%。Hang 类任务对位置和轨迹的自适应要求最高，正好打在了 HALO 的两级火箭设计上。

跟 VLA 比（Table 7）也很有意思。pi0 和 RDT 这种预训练大模型，比从零训的 DP/DP3 强，说明视觉先验有用。但它们还是打不过 HALO。为什么？因为 VLA 擅长 high-level 语义理解，但在 garment 这种需要极度精细的 geometric correspondence 和 precise grasp-and-place 的任务上，VLA 的通用表征还是差了点意思，不如 GAM 这种专门针对 category-level correspondence 训出来的模型精准。

Sim-to-real（Way 2）的实验（Table 4）特别值得关注。纯仿真数据训出来的 policy 到真实世界，成功率掉到 53%。但只要加 15 条真实数据混进去训，成功率就飙到 86.7%。这说明 DexGarmentLab 的物理仿真已经足够接近真实，gap 不大，稍微补一点真实数据就能把 sim-to-real 的 loop 闭上。用了 Azure Kinect 相机提高点云精度，这也是关键。

---

### 5. Limitation 的潜台词

Paper 最后很诚实地聊了局限。
PBD 的抖动问题没彻底解决，FEM 对大件衣服完全不适用。
任务全是单件衣服，没做多件衣服堆叠的场景（这个在他们后续的 GarmentPile, https://arxiv.org/abs/2503.09243 里搞了）。
机器人是固定双臂，没带移动底盘，进不了真实家庭。
GAM 对极度遮挡或者非标设计（比如单袖）的衣服会瞎。

这其实描绘了这个领域的现状：**我们在一个极度受限的沙盒里，通过利用 category-level 的结构先验，勉强把 garment manipulation 从“不可解”推到了“能跑通”的阶段。** 离真正的人形机器人给你叠衣服还有十万八千里，但这条路走对了。

总结一下直觉：
1. **环境**：靠 PBD + Adhesion/Friction 参数堆出来的工程奇迹，让灵巧手能“捏”住衣服。
2. **数据**：靠 Skeleton Merger + InfoNCE 学到的拓扑对应，把一次 demo 广播成千上万次。
3. **算法**：靠 GAM + Diffusion 的分层架构，解耦了“抓哪”和“怎么动”，让 policy 在新形状衣服上能自适应轨迹。

这基本就是当前 deformable manipulation 最靠谱的范式了。这组人一直在顺着这个脉络做，从 UniGarmentManip 到 GarmentLab 再到 DexGarmentLab 和 GarmentPile，一脉相承。

希望这个角度能帮你 build intuition。

---

# DexGarmentLab: 灵巧手衣物操作环境的深度解析

Karpathy好，这篇paper从PKU + Berkeley联合出来，核心想解决的是 **dexterous（特别是bimanual）garment manipulation** 这个老大难问题。我尽量把核心直觉给你讲清楚，并展开技术细节。

---

## 1. Motivation与问题的本质

Garment manipulation的难点可以归为三件事纠缠在一起：

1. **Data scarcity**：dexterous hand的action space高维（60 DoF: 双臂6+6，双Shadow Hand 24+24），garment状态又是连续无穷的，teleoperation和RL都低效。
2. **Environment realism**：deformable object与rigid/articulated object的interaction simulation本身就难，dexterous hand的多指接触更难建模。
3. **Generalization**：garment的category-level几何变化大，IL算法（DP、DP3）在新形状的garment上失败在"抓不到正确的点" + "轨迹不能根据garment形状调整"。

作者的核心insight：**同类garment具有结构一致性（structural correspondence）**。一件top和另一件top虽然颜色尺寸不同，但都有collar、左右sleeve、hem这些语义对应点。这个对称性既是**自动化数据采集**的关键，也是**policy泛化**的关键。

project page: https://wayrise.github.io/DexGarmentLab/

---

## 2. Environment：DexGarmentLab

### 2.1 物理仿真的关键改进

PBD（Position-Based Dynamics）模拟garment的本质是把mesh当作一堆**松散连接的粒子**。问题在于：粒子间存在gaps，rigid body的gripper很容易穿透garment而没有effective lifting。

**GarmentLab的前作方案**（https://arxiv.org/abs/2411.01200）：在gripper tip上attach一个红色block，block和garment粒子之间通过constraint强制吸附。这在parallel gripper上勉强能用，但搬到dexterous hand（10个finger tip都要block）会导致：
- 即便一根手指碰到garment就建立attachment，违反物理直觉
- garment在多block之间被不自然地拉伸sagging

**DexGarmentLab的方案**：彻底抛弃attachment block，改用四个核心物理参数：

#### Adhesion（粒子-刚体吸附）
$$\mathbf{f}_{\mathrm{adh}} = -k_{\mathrm{adh}} \cdot (\mathbf{x}_i - \mathbf{x}_j), \quad \mathrm{if} \ \|\mathbf{x}_i - \mathbf{x}_j\| < r_{\mathrm{adh}}$$

- $\mathbf{x}_i$: garment粒子的position
- $\mathbf{x}_j$: rigid surface proxy的position（比如finger mesh上对应的接触点）
- $k_{\mathrm{adh}}$: adhesion coefficient，控制吸附强度
- $r_{\mathrm{adh}}$: adhesion激活半径阈值，超过这个距离吸附力为零

直觉：这个公式本质是个**截断的spring-like attractive force**，让garment粒子在finger附近"粘"住，但不是硬约束。所以多指接触时，每根finger都贡献一个局部的adhesion，自然形成稳定抓取。

#### Friction（Coulomb摩擦）
$$\mathbf{f}_{\mathrm{fric}} = -\mu \cdot \|\mathbf{f}_n\| \cdot \frac{\mathbf{v}_t}{\|\mathbf{v}_t\|}$$

- $\mu$: 摩擦系数
- $\mathbf{f}_n$: 接触法向力（normal force）
- $\mathbf{v}_t$: 切向相对速度（tangential relative velocity）

在PBD里这个公式被实现为positional correction，在constraint projection step里抵抗sliding。

#### Particle-Adhesion-Scale 与 Particle-Friction-Scale
这两个参数作用于**garment粒子之间**的自相互作用：
- Particle-Adhesion-Scale：抑制粒子在self-collision时的爆炸性分离
- Particle-Friction-Scale：作为internal damping，保留folds和wrinkles

直觉：这两参数本质是稳定garment的internal dynamics，避免garment折叠时出现"chaotic jittering"和"自穿透后炸开"。Fig. 3的对比很直观——GarmentLab折叠后状态松弛散乱，DexGarmentLab能保持folded state稳定。

### 2.2 模拟方法的选择

| Garment类型 | 方法 | 原因 |
|------------|------|------|
| Tops, Dresses, Trousers（大件、高变形） | PBD | 粒子模型捕捉柔软性，支持folding |
| Gloves, Hats（小件、弹性） | FEM | 体积块模型，弹性回弹更真实 |

这里有个trade-off：PBD柔软但不稳定，FEM稳定但偏刚性。作者承认FEM对大件garment会"顽固回到原形"，所以只用于本就少变形的gloves和hats。

PBD原始paper: https://matthias-research.pages.teknikum.tu-clausthal.de/publications/PBD.pdf
FEM基础: https://onlinelibrary.wiley.com/doi/abs/10.1002/047134608X

### 2.3 Asset与Task设计

- 来自 **ClothesNet**（https://arxiv.org/abs/2308.09987）：2500+ garments，8 categories
- 15个tasks，分两类：
  - **Garment-Self-Interaction**：Fling, Fold（6个）
  - **Garment-Environment-Interaction**：Hang, Wear, Store（9个）

随机化范围有限制（保持task feasibility同时增加policy generalization），garment position和environment-interaction object position都在小矩形区域内randomize。

---

## 3. GAM（Garment Affordance Model）

GAM是整篇paper的"对应性引擎"，基于UniGarmentManip（https://arxiv.org/abs/2406.01507，project: https://j9877554.github.io/uni-garment-manip/）。

### 3.1 训练流程

核心想法：让PointNet++学到garment点云的dense visual correspondence，即garment A上的点 $p_i$ 和garment B上的对应点 $p_j$ 在feature space里应该靠近。

#### 数据构造
- **Skeleton Merger**（https://arxiv.org/abs/2104.09902）网络获取flat garment间的skeleton point correspondences（semantic keypoints对齐）
- 在simulation中用point tracing方法建立flat garment ↔ deformed garment的correspondence

#### InfoNCE Loss
$$\mathcal{L}_{\mathrm{InfoNCE}} = -\log \frac{\exp(\mathrm{sim}(\mathbf{f}_{p_i}, \mathbf{f}_{p_j^+})/\tau)}{\sum_{k=1}^{K} \exp(\mathrm{sim}(\mathbf{f}_{p_i}, \mathbf{f}_{p_k})/\tau)}$$

变量解释：
- $\mathbf{f}_{p_i}$: anchor point $p_i$ 的PointNet++ feature
- $\mathbf{f}_{p_j^+}$: positive sample（与 $p_i$ 在另一garment上对应点）的feature
- $\mathbf{f}_{p_k}$: 所有candidate（含positive和negatives）的feature
- $\tau$: temperature parameter，控制分布的sharpness
- $\mathrm{sim}(\cdot, \cdot)$: cosine similarity或dot product
- $K$: 候选总数

直觉：这是contrastive learning的经典loss。正样本对在feature space拉近，负样本对推远。在UniGarmentManip里每个batch是32 garment pairs，每pair采样20 positive和150 negative point pairs，所以一个batch是 $32 \times 32 \times 20$ 个对比样本。

InfoNCE原始paper (CPC): https://arxiv.org/abs/1807.03748

### 3.2 推理流程

给定：
- Demo garment点云 $O$
- Demo grasp points $(p_1, p_2, \ldots)$
- 新garment点云 $O'$

步骤：
1. 用GAM提取 $O$ 上 $p_1, p_2, \ldots$ 的feature $\mathbf{f}_{p_1}, \mathbf{f}_{p_2}, \ldots$
2. 提取 $O'$ 上所有点的feature
3. 计算 $\mathbf{f}_{p_i}$ 与 $O'$ 所有点feature的dot product similarity
4. 选similarity最高的点作为 $p_i'$

这就是cross-garment dense correspondence的迁移。配合pre-normalization到canonical space保证translation和scale invariance，rotation通过训练数据augmentation处理（视为deformation state）。

---

## 4. Automated Data Collection Pipeline

这是数据效率的核心。整个pipeline从一个**single expert demonstration**出发：

```
Single Demo → {demo grasp points, demo task sequences, demo hand grasp poses}
                ↓
对每个新garment实例（带随机位置/朝向/形状）:
    GAM匹配 → 得到对应grasp points
    ↓
    IK控制双臂执行demo task sequences
    PD controller控制双手按demo hand grasp poses
    ↓
    轨迹自适应: 折叠高度根据sleeve长度调整，挂衣位置根据hanger位置调整
    ↓
    记录joint_state, image, point cloud, affordance feature等
```

### 4.1 关键设计

**Hand grasp poses的来源**：LeapMotion Controller（https://www.ultraleap.com/product/leap-motion-controller/）teleoperation生成。两个640×240 near-infrared cameras，120Hz，提取27个hand features（palm normal, hand direction, wrist position, 24 finger joint positions）。这里只用24 finger joint positions控制Shadow Hand。

**Task-specific hand poses的复用**：每个task预定义一组task-specific hand poses（比如Hang Coat需要closed grasping pose for collar + open pose for release），同一task的不同garment实例复用这些poses。garment的deformable性质让它能适应hand pose。

**Trajectory自适应**：不是固定轨迹。例如fold任务中，lifting height根据sleeve长度和garment总长调整；hang任务中，lifting height和placement position根据garment center与hanger的alignment调整。

### 4.2 数据采集效率

Table 5非常亮眼：每个task 100条demo，单条采集时间30秒～1分42秒，成功率82%～99%。Real-world也类似（Fold Tops 50/55，成功率90.9%）。

对比teleoperation：Table 6显示在Hang Tops、Wear Bowlhat、Fold Tops三个任务上，autonomous collected data训练的HALO和teleoperation data训练的HALO性能相当（0.92 vs 0.88，0.72 vs 0.70，13/15 vs 13/15），但autonomous采集的人力成本低几个量级。

---

## 5. HALO: Hierarchical Policy

HALO是核心算法贡献，分为两阶段：

### 5.1 Stage 1: GAM → Affordance Points

GAM输出的是每个garment点的affordance score（normalized到[0,1]），shape (2048, 2)表示left和right hand的affordance。这相当于告诉policy"去哪里抓"。

### 5.2 Stage 2: SADP（Structure-Aware Diffusion Policy）

SADP基于Diffusion Policy（https://arxiv.org/abs/2303.04137）框架，核心改动在observation representation $\mathbf{s}$。

#### Observation构造
```
Operated garment point cloud (2048, 3)
    + Left/right affordance features (2048, 2)  
    → concat → PointNet++ → F_garment

Interaction-object point cloud (2048, 3) [可选]
    → MLP → F_object

F_scene = concat(F_garment, F_object)
    [Garment-Self-Interaction任务里 F_scene = F_garment]

O_environment (full env point cloud) → MLP → F_env
O_state (robot joint state) → MLP → F_state

s = F_scene + F_env + F_state  (denoising condition)
```

这里的关键intuition：
- $F_{\mathrm{garment}}$ 编码garment的current state（position, shape, structure）
- $F_{\mathrm{object}}$ 编码interaction object的position
- $s$ 同时作为diffusion的condition，让生成的trajectory能"看到"garment的形状结构

#### Diffusion过程回顾

DDPM前向：
$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

- $\mathbf{x}_0$: 原始action（60维DoF target）
- $\mathbf{x}_t$: 第 $t$ 步加噪后的action
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$: 累积噪声系数
- $\mathbf{I}$: 单位矩阵

反向去噪学习：
$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{s}) \right) + \sigma_t \mathbf{z}$$

- $\boldsymbol{\epsilon}_\theta$: 神经网络预测的噪声
- $\mathbf{s}$: 上面构造的condition（garment+scene+state feature）
- $\sigma_t$: 反向过程方差
- $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

训练loss简化为：
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{s}) \|^2 \right]$$

#### Hyperparameters
- Horizon = 8, observation_steps = 3, action_steps = 4
- Denoising steps = 10
- 3000 epochs，每25 epoch验证，每100 epoch存checkpoint
- AdamW, lr = 1e-4, cosine schedule + 500 warmup steps
- A800 GPU，batch=200，约75GB显存，16小时训练

Diffusion Policy原始paper: https://diffusion-policy.cs.columbia.edu/
DP3 (3D Diffusion Policy): https://arxiv.org/abs/2403.03954

### 5.3 为什么分层？

直觉上：把"找位置"和"生成轨迹"解耦是关键。

- 纯DP/DP3失败的原因：它们试图从一个fixed trajectory分布里采样，但新garment的shape变了，原来抓的position不再对应正确的semantic region。IL的轨迹本质是分布拟合，无法外推到training set外的garment形状。
- HALO的GAM先解决"在哪抓"这个semantic correspondence问题
- SADP再解决"怎么动"这个trajectory generation问题，并且condition里包含garment shape feature，所以能根据garment几何调整轨迹

Fig. 7的ablation可视化很说明问题：
- Without GAM：手抓不到正确的sleeve位置
- Without SADP：手抓对了位置，但lifting height不对，挂不上去

---

## 6. 实验分析

### 6.1 主实验（Table 2）

15个任务上HALO全面胜出。挑几个关键数字：

| Task | DP | DP3 | HALO | 提升 |
|------|-----|-----|------|------|
| Fling Dress | 0.59 | 0.51 | 0.82 | +39% |
| Fling Trousers | 0.54 | 0.58 | 0.83 | +43% |
| Fold Trousers | 0.47 | 0.54 | 0.77 | +43% |
| Hang Tops | 0.45 | 0.53 | 0.92 | +74% |
| Hang Coat | 0.52 | - | 0.90 | +73% |

Hang类任务提升最大，因为这些任务对grasp position + trajectory shape都敏感，正好是HALO两阶段设计的强项。

### 6.2 消融

- **w/o GAM**（用SADP直接做）：性能明显下降，证明"找对位置"是前提
- **w/o SADP**（用GAM + DP3）：性能也下降，证明"基于garment shape生成轨迹"也必要

两个组件都不可或缺，是complementary的。

### 6.3 与VLA对比（Table 7, Appendix G）

加了ACT、pi0、RDT、Eureka四个baseline：

| Method | Fling Dress | Fold Trousers | Hang Coat | Wear Bowlhat | Fold Tops (Real) |
|--------|-------------|---------------|-----------|--------------|------------------|
| ACT | 0.35 | 0.49 | 0.43 | 0.51 | 7/15 |
| pi0 | 0.69 | 0.52 | 0.72 | 0.59 | 10/15 |
| RDT | 0.60 | 0.58 | 0.62 | 0.48 | 9/15 |
| Eureka | - | - | 0.16 | 0.08 | 1/15 |
| HALO | 0.82 | 0.77 | 0.90 | 0.72 | 13/15 |

观察：
- VLA类（pi0, RDT）比from-scratch的DP/DP3/ACT强，预训练知识有用
- 但VLA对garment shape/state的精细感知和precise grasp-and-place还是不如HALO
- Eureka（RL+VLM）失败：reward设计难，Isaac Sim parallel支持不够，long-horizon task reward稀疏

pi0: https://www.physicalintelligence.company/blog/pi0
RDT: https://arxiv.org/abs/2410.07864
Eureka: https://arxiv.org/abs/2310.04676

### 6.4 Real-World（Way 1: Real采集 + Real训练）

Setup: RealMan RM75-6F arms + Psibot G0-R dexhands + RealSense D435 + SAM2（https://arxiv.org/abs/2408.00714）做segmentation

| Task | DP | DP3 | HALO |
|------|-----|-----|------|
| Fold Tops | 9/15 | 8/15 | 13/15 |
| Hang Tops | 10/15 | 8/15 | 13/15 |
| Wear Scarf | 6/15 | 7/15 | 11/15 |
| Wear Hat | 10/15 | 9/15 | 14/15 |

### 6.5 Sim-to-Real（Way 2: Sim训练 + Real部署）

Setup对齐：UR10e + ShadowHand + Azure Kinect（比RealSense精度高）

| Task | Only Sim Data | Sim + 15 Real Data |
|------|---------------|---------------------|
| Hang Trousers | 8/15 (53.3%) | 13/15 (86.7%) |
| Wear Hat | 9/15 (60.0%) | 13/15 (86.7%) |

直觉：少量real data补sim-to-real gap效果显著。这说明DexGarmentLab的sim足够接近real（gap不大），加15条real demo就能close the loop。

---

## 7. Limitation的诚实讨论

### 7.1 Simulation方法局限
- PBD：粒子松散，self-collision导致jittering，penetration artifact难消除
- FEM：大件garment顽固回到原形，只适合gloves/hats

### 7.2 Task局限
- 全是single-garment tasks，没有multi-garment pile场景（GarmentPile https://arxiv.org/abs/2503.09243是同组的后续工作）
- 没有mobile base，dual-arm固定，实际home robot需要mobile platform

### 7.3 Policy局限
- GAM对occluded或highly deformed region的预测不准，需要先unfold
- 对asymmetric design（单袖top）或装饰繁复costume表现差，因为点云被干扰

---

## 8. 给Karpathy的Intuition总结

这篇paper的核心intuition可以这样概括：

**用structural correspondence解耦一个超难问题，分成两个相对可控的子问题。**

Garment manipulation难是因为它把"semantic reasoning（领口在哪、袖子在哪）"和"continuous control（高维手怎么动）"耦合在一起。HALO把这两件事拆开：
- GAM解决semantic reasoning，用contrastive learning在category-level上学dense correspondence，跨garment实例迁移
- SADP解决continuous control，用diffusion生成轨迹，condition里带garment shape feature实现shape-aware

**自动化数据采集的insight也是correspondence**：同一task类别下，grasp points的semantic位置在category内可迁移，所以一个demo就能broadcast到上千个garment实例。

**物理仿真的insight**：放弃硬约束（attachment block），改用软约束（adhesion + friction + scale参数），让dexterous hand能像人手一样靠多指摩擦+轻吸附稳定抓取garment。

**Hierarchical设计的好处**：每一步的输入空间都比端到端小。GAM只关心点云对应，SADP只关心trajectory生成。相比DP/DP3端到端学"从图像到60维动作"，HALO把"在哪抓"和"怎么动"分开，各自generalize更容易。

从你的视角看，这工作挺像deformable manipulation领域的"category-level structural prior + diffusion policy"组合拳，和category-level rigid object manipulation（NOCS, https://arxiv.org/abs/1904.04696）的思路有精神上的传承。

延伸阅读链接：
- UniGarmentManip: https://j9877554.github.io/uni-garment-manip/
- GarmentLab: https://arxiv.org/abs/2411.01200
- GarmentPile: https://arxiv.org/abs/2503.09243
- ClothesNet: https://arxiv.org/abs/2308.09987
- UniDexGrasp (category-level dex grasp): https://arxiv.org/abs/2303.00931
- FlingBot: https://arxiv.org/abs/2105.03655
- SpeedFolding: https://arxiv.org/abs/2208.10552
- Cloth Funnels: https://arxiv.org/abs/2210.09347
- DexDeform: https://arxiv.org/abs/2304.03223
- Bunny-VisionPro teleoperation: https://arxiv.org/abs/2407.03162
