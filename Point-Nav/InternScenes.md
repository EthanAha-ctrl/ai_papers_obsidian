---
source_pdf: InternScenes.pdf
paper_sha256: a92638fea597047fa77c8239c0de5a5b744b26e93e97e609fe8ac64c0212c7bc
processed_at: '2026-08-05T10:13:25-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# InternScenes 人话版

## 这帮人到底在干啥

想象你在教一个 robot 在屋里走路、开门、拿东西。它得先在 simulator 里练几十万次——但练啥呢？得有个"屋子"让它练。问题来了：**Simulator 里的屋子太难找了**。

现在能用的"屋子"就三类，各有各的坑：

1. **Real-world scan**：拿激光扫真屋子，得到点云。优点是真实，缺点是 simulator 根本跑不动——点云不是 mesh，没物理属性，robot 没法跟它交互（不能开抽屉、不能碰椅子）。

2. **Designer 搭的**：专业设计师用 3D 软件搭的屋子。优点是 simulator 能跑，缺点是太干净——设计师根本不会在桌上摆 14 本书、3 个杯子、1 个花瓶，所以屋子里平均就 6-7 件家具，跟真实生活完全不像。而且 designer 数据 collision 严重——椅子穿到桌子里、柜子嵌进墙里，根本没法拿来训练 robot。

3. **Procedural generation**：程序自动生成。优点是量大、零 collision，缺点是生成 100 个屋子长得都差不多，多样性不够，而且生成一个屋子要跑很久。

**InternScenes 做的事**：把这三类来源全揉到一起，加上一堆工程处理，搞出一个 40,000 个屋子的大数据集，平均每个区域 41.5 个物体（之前最厉害的 3D-FRONT 才 6.9），其中 20% 是可以开的抽屉、可以转的龙头这种 articulated object。第一次让"在真实布局上大规模训练 robot"变得可能。

---

## 最核心的 insight

用一句话讲：**真实屋子的复杂度不在大件家具，而在小物体堆叠**。

一个真实客厅里，沙发 + 茶几 + 电视柜就 3 件大家具，但茶几上有遥控器、书、水杯、果盘、手机，沙发上有抱枕、毯子、玩偶，电视柜上有路由器、相框、绿植——这些小物体构成了场景的 **layout entropy**，也是 robot navigation 和 manipulation 真正的难点所在。

之前的数据集要么没有这些小物体（3D-FRONT），要么有但 simulator 跑不动（ScanNet），InternScenes 第一次把它们以可仿真的形式保留下来。

---

## 关键工程细节，用人话讲

### 1. Real2Sim：把扫的真屋子"翻译"成 simulator 能跑的

EmbodiedScan 有真实屋子的扫描数据，每个物体都标了 9DoF bounding box（位置 + 尺寸 + 朝向）。但问题是这些只是点云，不是 simulator 能用的 mesh。怎么办？

**思路**：拿一个已有的 3D asset 库（Objaverse，800 万个 CAD model），给每个 bbox 找一个最像的 asset 塞进去。

**找最像的怎么找**？算 bbox 尺寸的 cosine similarity：

$$\text{sim}(\mathbf{c}_i, \mathbf{t}) = \frac{\mathbf{c}_i \cdot \mathbf{t}}{\|\mathbf{c}_i\| \|\mathbf{t}\|}$$

- $\mathbf{c}_i$：候选 asset 的尺寸向量 $(w, h, d)$
- $\mathbf{t}$：原始 bbox 的尺寸向量

为什么不用 L2 距离？因为 Objaverse 资产尺度极乱——同一个杯子有人标 0.3 米有人标 30 厘米。cosine 只看比例，先匹配形状，再单独处理 scale，更鲁棒。

**朝向问题**：Objaverse 的椅子可能躺着、朝向不一。他们 render 几张不同角度的图，让 InternVL 判断哪张是"正面"，再把 asset 主轴对齐到 +x 轴，用原始标注的 Euler 角放置。本质是用 VLM 替代了传统的 PCA orientation alignment。

**Label 模糊问题**：EmbodiedScan 里有个 "object" 类，意思是"不知道是啥的小东西"。他们根据位置替换——在地上就换成 bin/bag/shoe，在桌上就换成 book/cup/vase。这本质是个 hand-crafted 的 scene grammar prior，用 LLM 辅助。

### 2. 处理 collision：两阶段物理优化

**阶段一：梯度下降优化大件家具**

针对沙发、桌子这些大件，设计三个 loss：

$$\mathcal{L} = \lambda_{\text{IoU}} \mathcal{L}_{\text{IoU}} + \lambda_{\text{ground}} \mathcal{L}_{\text{ground}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{IoU}}$：家具之间重叠就惩罚
- $\mathcal{L}_{\text{ground}}$：让家具贴地，解决 scan 噪声导致的浮空
- $\mathcal{L}_{\text{reg}}$：别离原始标注太远，保留设计意图

**阶段二：SAPIEN 物理仿真处理小物体**

大件搞定后，把小物体丢进物理引擎，开重力 + repulsive force，让它们自然 settle。这里有个关键 trick：**带腔体的家具（抽屉、柜子）要先做 segmentation 暴露内部**，再用 COACD 做凸分解，否则柜子就是一整块实心 mesh，物体永远掉在外面进不去抽屉。

为什么需要两阶段？小物体形状复杂，OBB 优化失效——一个 cup 的 OBB 跟真实 mesh 差很远，bbox-level IoU 根本不反映真实碰撞。

---

## Infinigen Indoors 的 Simulated Annealing

InternScenes-Gen 部分用 Infinigen 自动生成场景。核心是 simulated annealing：

$$p(s'|s) = \min\left[\exp\left(\frac{l(s) - l(s')}{\tau}\right), 1\right]$$

- $s$：当前场景状态
- $s'$：提议的新状态（加个物体、旋转一下等）
- $l(s)$：违反约束的总 loss
- $\tau$：温度，从 0.25 降到 0.001

直觉：温度高时什么都接受（探索），温度低时只接受更好的（利用）。避免陷入"沙发靠左墙、电视柜靠右墙、中间茶几穿模"这种 local optimum。

Infinigen 单家具就有 17 generators + 216 controllable parameters，高度随机化避免 reuse 同一批 model 引入 bias——这对训练生成模型至关重要。如果训练集里同一把椅子出现 5000 次，模型会 overfit 这个 mesh 而不是学"椅子"概念。

---

## 数据集核心数字

- **40K scenes**，48K regions，15 种类型
- **1.96M objects**，288 类，来自 800K 个 CAD model
- **平均 41.5 objects/region**（之前最高 21.1）
- **20% 是 articulated**（PartNet-Mobility URDF）
- **每个容器/支撑物平均与 5.57 个其他物体有空间关系**（containment / support）

这个 5.57 是关键信号——它意味着场景有 hierarchical relational structure（书在桌上、笔在书上）。3D-FRONT 里这数字接近 0，模型学不到这种层级关系。

---

## 实验一：Scene Layout Generation

三个 baseline：ATISS（autoregressive transformer）、DiffuScene（diffusion）、PhyScene（diffusion + physics）。

做了两版数据：
- **Full version**：保留所有物体
- **Simplified version**：只留 45 类大家具

**结果**（FID，越低越好）：

| | Full | Simplified |
|---|---|---|
| ATISS | 101.85 | 23.20 |
| DiffuScene | 96.56 | 22.88 |
| PhyScene | 88.02 | 23.78 |

**用人话说**：一旦把小物体加进来，所有方法性能直接崩 5 倍。这说明现有 scene generation 模型的 contextual modeling capacity 在 ~50 个 object token 时就饱和了——小物体把 token length 拉长 6x 后，attention 被稀释，模型学不到"小物体应该怎么摆"的分布。

PhyScene 在 Living region 上 FID 66.59 明显好于 DiffuScene 107.49，说明 physics guidance 对 cluttered scene 有效——collision loss + repulsive force 让小物体不重叠不浮空。

**这指向一个 architecture 机会**：要 scaling 到 100+ objects，需要 hierarchical diffusion（room → region → object）或 sparse attention。

---

## 实验二：PointGoal Navigation

用 Isaac Sim（物理 realistic），ClearPath Dingo 轮式 robot，20 个 Real2Sim + 10 个 Gen 场景，每场景 20 episodes，起点终点距离 3-10m。

**结果**（Success Rate）：

| | Real2Sim | Gen |
|---|---|---|
| DD-PPO（RL）| 23.6 | 45.0 |
| NavDP（diffusion IL）| 48.3 | 61.9 |
| NavDP-FT（加 118K 数据微调）| 51.0 | 63.6 |

**用人话解读**：

1. **DD-PPO 在 Real2Sim 上只有 23.6%**，Gen 上 45%——差距来自 Real2Sim 更 clutter（真实扫的，小物体多）。DD-PPO 训练于 Habitat-Sim 的相对空旷 MP3D 场景，迁移到 cluttered 环境出现严重 sim-to-sim gap。

2. **NavDP 比 DD-PPO 高 25 个点**。Diffusion policy 对多模态目标分布更鲁棒，这点和你之前讲 diffusion policy 的直觉一致。

3. **NavDP-FT 加了 118K 轨迹只提升 2.7%**。这是最有意思的发现——**单纯加数据已经不能解决 cluttered navigation 问题了**。Navigation 在 InternScenes 上 break 了 power law scaling。

论文识别了三个核心挑战：
- **Cluttered layout 需要 collision recovery**：现有 policy 一碰就卡，没有"后退-绕行"行为
- **Narrow pathway 需要 embodiment-aware traversability**：办公椅 5 个腿形成的窄通道，Dingo（40cm）能过但 Fetch（60cm）不能过，现有方法只用视觉没用 robot footprint
- **Small connected components 的 perception**：椅子腿在视觉里只有几帧可见，但是 100% 的 collision source——典型 long-tail perception 问题

---

## 对你 Karpathy 视角的联想

### 1. 这数据像什么
InternScenes 之于 Embodied AI，类似 ImageNet-1k 之于 2D vision，或者 JFT-300M 之于 ViT——第一次让"在真实分布上 scaling 训练"成为可能。之前的 MP3D/HM3D 太小太干净，相当于在 MNIST 上训 ResNet。

### 2. Scene Generation 暴露的 Scaling Wall
FID 100+ on full version 不是模型差，是**任务复杂度爆炸**。小物体分布是 multi-modal 的（同一张桌上的物体组合有无数种合理配置），当前 flat token sequence 建模（autoregressive / diffusion）在 ~50 token 时就饱和了。

这暗示**下一代 scene generation 需要 hierarchical representation**——类似 GPT 处理自然语言的 subword tokenization，scene 也需要 region → object cluster → object 的多级 tokenization。你之前讲 World Model as Token Sequence 的思路在这里直接适用，但要保留 hierarchical structure。

### 3. Navigation 的 Data Wall
NavDP-FT 加 118K 数据只涨 2.7%，这是个 scaling law 信号。你在 LLM 里看到的 power law scaling 在这个 navigation 任务上 break 了。可能的解释：
- 任务 difficulty scaling 比 data scaling 快（场景越复杂，每个 episode 提供的 effective gradient 越少）
- 需要 architectural prior（memory、collision recovery、embodiment awareness），不是更多数据
- 这就是 LLM-RL 里常见的 "data wall" 在 embodied AI 上的体现

### 4. World Model 方向
如果让我赌一个方向，就是在 InternScenes 上训一个 **3D world model**——把 hierarchical relation 编码进 latent，然后做 long-horizon task planning（"做早餐" = 去厨房 → 开冰箱 → 拿鸡蛋 → 开灶 → 放锅 → ...）。这跟你 World Model as Token Sequence 思路契合，InternScenes 第一次提供了具备足够 realism + scale 的训练数据。

### 5. Manual Annotation 的 Bottleneck
论文自己承认的 limitation：Synthetic subset 还需要人工标 region type。这指向一个明显的自动化机会——用 VLM 做自动 region segmentation + instance labeling。他们已经用 InternVL 做 instance captioning 了（准确率 85%+），下一步完全可以让 VLM 端到端替代人工 region 标注。

---

## 资源链接

**数据集与代码**：
- InternScenes：https://github.com/opendatalab/InternScenes （待开源）
- EmbodiedScan：https://github.com/Tai-Wang/EmbodiedScan
- Infinigen：https://infinigen.org/
- 3D-FRONT：https://3d-front.org/
- Objaverse：https://objaverse.allenai.org/
- PartNet-Mobility / SAPIEN：https://sapien.ucsd.edu/

**Baselines**：
- ATISS：https://github.com/nv-tlabs/ATISS
- DiffuScene：https://github.com/zlzeng/DeepLayoutSynthesis
- PhyScene：https://github.com/yang-yang-666/PhyScene
- DD-PPO：https://github.com/facebookresearch/habitat-lab
- NavDP：https://github.com/Chenjia-Bai/NavDP

**工具链**：
- COACD（凸分解）：https://github.com/Weixiao-Hong/CoACD
- Isaac Sim：https://developer.nvidia.com/isaac-sim
- InternVL：https://github.com/OpenGVLab/InternVL

**Concurrent / Related**：
- MetaScenes：https://arxiv.org/abs/2503.18450
- ACDC：https://arxiv.org/abs/2410.14066
- MIDI：https://arxiv.org/abs/2412.03558
- SceneScript：https://meta-ai.github.io/scenescript/

---

## 最后的 Intuition

这篇论文的 takeaway 可以压缩成一句：**Embodied AI 的瓶颈不在算法，而在数据环境的 realism-density trade-off**。

过去十年我们在 simulator 容易性（Habitat MP3D）和 realism（Real-world scan）之间二选一。InternScenes 通过 Real2Sim + 物理后处理第一次同时拿到 realism + density + simulatability。代价是大量 engineering labor（标注、asset curation、physics tuning），但回报是让 scene generation 和 navigation 都暴露出新的 scaling wall——这些 wall 才是下一代 architecture 的真正机会。

对你而言，如果在这上面做事，最有 impact 的方向我赌 **3D World Model on InternScenes**——把 hierarchical scene relation 学进 latent，做 long-horizon task planning。这条路跟你 World Model as Token Sequence 的哲学完全一致，而且数据终于 ready 了。

---

# InternScenes: 大规模可仿真室内场景数据集深度解析

Andrej，这篇 paper 来自 Shanghai AI Lab 的 Miao Pang 组（也是 NavDP、PartNet-Mobility 工作线的延伸），它解决的核心痛点很直接：**Embodied AI 训练需要大规模、可仿真、布局真实的 3D 场景，但现有数据集各执一端**——real scans 不能仿真（点云 + incomplete geometry）、designer scenes 太"干净"（缺小物体）且 collision 严重、procedural generation 多样性受限。InternScenes 把三类来源融合成一个 40K scenes、1.96M objects、平均 41.5 objects/region 的统一可仿真数据集。

---

## 1. 问题动机的直觉构建

从你做 Ego-Learning 和 VPT 的经验出发，embodied agent 的 generalization 本质上受限于训练环境的 **layout entropy**——agent 需要见过足够多样的"物体拓扑关系"才能在 test time 泛化。而 layout entropy 由两个量决定：
- **物体密度**（objects per region）：3D-FRONT 只有 6.9，Structured3D 是 21.1，InternScenes 做到 41.5
- **small items 占比**：chair/bed 这种"大基底"很容易生成，但桌上 14 个 book、3 个 cup、1 个 vase、抽屉里塞满 toy 的这种"高频小物体堆叠"才是真实场景的核心结构

这也是为什么 paper 在 Table 2 中专门做了 Full vs Simplified 对照——简化版只保留 45 类大家具，结果所有 baseline (ATISS / DiffuScene / PhyScene) 的 FID 从 ~20 暴跌到 ~100。这背后的 intuition 是：当前 scene generation 模型的 contextual modeling capacity 在 ~50 个 object token 时就饱和了，small objects 把 scene token length 拉长 6x 后，autoregressive / diffusion 的 attention 都会被稀释。

---

## 2. 数据集的三源融合架构

数据集分三个 subset：

| Subset | 来源 | #Regions | 优势 |
|--------|------|----------|------|
| InternScenes-Real2Sim | EmbodiedScan (ScanNet/MP3D/3RScan) | 9,833 | 真实布局 + 288 类 + 小物体标注 |
| InternScenes-Gen | Infinigen Indoors | 11,454 | 零碰撞 + procedural 多样性 |
| InternScenes-Synthetic | Designer-created | 27,094 | 大规模 + 空间覆盖广 |

这种"三源融合"的设计哲学让我想到 ViT 训练中 JFT-300M + ImageNet-21k 的混合策略——每种来源都有自己的 distribution bias，融合可以抵消单一 bias。Real2Sim 提供"realistic 但 messy"的分布，Gen 提供"clean 但 constrained"的分布，Synthetic 提供"wide 但 designer-style"的分布，三者组合下 agent 学到的是更鲁棒的世界先验。

---

## 3. Real2Sim Pipeline 的核心技术细节

这是最 engineering-heavy 的部分，值得深入拆解。

### 3.1 Asset Retrieval 的核心公式

给定 EmbodiedScan 中某个 9DoF bbox 的目标尺寸 $\mathbf{t} \in \mathbb{R}^3$，需要从 asset library 中检索最佳匹配的 CAD model。论文用 **cosine similarity on bbox dimensions**：

$$\text{sim}(\mathbf{c}_i, \mathbf{t}) = \frac{\sum_{j=1}^{3} c_{i,j} t_j}{\sqrt{\sum_{j=1}^{3} c_{i,j}^2} \sqrt{\sum_{j=1}^{3} t_j^2}}$$

变量含义：
- $\mathbf{c}_i \in \mathbb{R}^3$：第 $i$ 个候选 asset 的 bbox 尺寸向量
- $\mathbf{t} \in \mathbb{R}^3$：EmbodiedScan 标注的目标 bbox 尺寸
- $j \in \{1,2,3\}$：对应 x/y/z 三个轴方向
- 上式等价于 $\cos\theta = \frac{\mathbf{c}_i \cdot \mathbf{t}}{\|\mathbf{c}_i\| \|\mathbf{t}\|}$

为什么用 cosine 而不是 L2 距离？直觉是 Objaverse 资产尺度极度不统一，cosine 对**形状比例**敏感而对**绝对尺寸**不敏感——一个 (0.6, 0.4, 0.8) 的杯子和 (6, 4, 8) 的大柜子 cosine similarity 是 1.0，但通过后续的 scale normalization 可以对齐绝对尺寸。这种"先比例匹配、再 scale 对齐"的两步策略比直接 L2 更鲁棒。

### 3.2 Label Mapping + Fuzzy Replacement

EmbodiedScan 有 288 类，但 Objaverse 资产用 Cap3D 描述（自由文本），需要做 label mapping：

1. **GPT-4o**：Cap3D caption → 288 类之一
2. **InternVL**：对 mapping 结果 + rendered images 做 verify/filter
3. **Position-based fuzzy replacement**：对于 ambiguous category "object"，根据支撑面语义替换

这个 fuzzy replacement 表（Table 4）的设计很有意思——它本质上是**一个 hand-crafted scene grammar prior**：在 floor 上的 object 只能是 bin/bag/backpack/basket/shoe/ball，在 table 上的只能是 book/plant/lamp/bottle/socket/cup/vase/bowl/plate/fruit/teapot。这类似于 procedural generation 中的 context-free grammar，但用 LLM 替换了规则系统。

### 3.3 Canonical Pose Correction

Objaverse 资产的朝向不统一（比如椅子可能躺着、朝向不一）。论文的做法是：

1. 从 oblique top-down 角度渲染多视角
2. InternVL 判断哪张图最像"front-facing view"
3. 将该 asset 主方向对齐到 +x 轴
4. 用 EmbodiedScan 标注的 Euler angles 放置

这种"render-then-VLM-judge"的闭环让我想到 LLaVA-style 的视觉理解 pipeline，本质是用 VLM 替代了传统的 PCA-based orientation alignment。在 URDF / articulated object 上这步尤其关键，因为关节轴方向必须正确。

### 3.4 L-shaped Couch 的镜像处理

这是个工程细节但很有启发性：L-shaped couch 分 left-L / right-L / standard 三类，资产库缺 right-L 时用 left-L 的镜像变换补齐。这种 mirror augmentation 在 furniture 这种 anthropic design 上是合法的（人类不会区分左右 L 沙发的语义），但对 asymmetric 物体（如冰箱把手方向）就会破坏语义。

---

## 4. Physics-Aware Scene Composition

这是 paper 的另一个核心贡献，分两阶段：**OBB optimization（梯度下降）** + **SAPIEN simulation（物理引擎）**。

### 4.1 OBB Optimization 的 Loss Function

针对大件家具，用三个 loss 联合优化：

$$\mathcal{L} = \lambda_{\text{IoU}} \mathcal{L}_{\text{IoU}} + \lambda_{\text{ground}} \mathcal{L}_{\text{ground}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$$

**IoU Loss**（防止重叠）：
$$\mathcal{L}_{\text{IoU}} = \sum_{1 \le j < k \le N} \left[\text{IoU}(b_j^{(t)}, b_k^{(t)})\right]^2$$
- $N$：场景中大件家具数量
- $b_j^{(t)}$：第 $t$ 次迭代时第 $j$ 个家具的 OBB
- 只对**实际相交**的 OBB 对计算 AABB IoU（OBB 相交检测后转 AABB）
- 平方是为了对严重穿透施加更强惩罚

**Ground Loss**（贴地约束）：
$$\mathcal{L}_{\text{ground}} = \sum_{j=1}^{N} (h_j^{(t)} - h_{\text{ground}})^2$$
- $h_j^{(t)}$：第 $j$ 个家具在第 $t$ 次迭代时的底面高度
- $h_{\text{ground}}$：地面高度（通常为 0）
- 解决 scan noise 导致的浮空 / 入地

**Regularization**（防止漂移）：
$$\mathcal{L}_{\text{reg}} = \sum_{j=1}^{N} \|t_j^{(t)} - t_j^{(0)}\|_2^2$$
- $t_j^{(t)}$：第 $t$ 次迭代的中心位置
- $t_j^{(0)}$：初始标注位置
- 保留原始 scene 的设计意图

这个三段 loss 的直觉是：在 EmbodiedScan 这种 scan-derived layout 上，原始 annotation 是有噪声的（点云重建误差），但完全放弃原布局去做 collision-free 优化会丢失真实分布特征，所以需要 $\mathcal{L}_{\text{reg}}$ 锚定。

### 4.2 为什么 OBB 优化不够，还需要 SAPIEN？

OBB 优化只能处理大件家具的"bbox 级"碰撞，但对**小物体**失效，原因有三：

1. **小物体形状复杂**：cup、book、vase 的 OBB 与真实 mesh 差距大，bbox-level IoU 不能反映真实碰撞
2. **腔体内部物体**：抽屉里、柜子里的物体，OBB 是包含整个 furniture 的，根本检测不到内部碰撞
3. **floating 问题**：小物体在 scan 中常有 1-5cm 浮空，OBB 优化无能为力

所以论文用 SAPIEN 做物理仿真：
- **COACD（Convex Decomposition）**：把每个 mesh 分解成多个 convex primitives 作为 collision mesh
- 对带腔体的家具先做 segmentation 暴露内部，再分别 COACD，最后合并——这是个关键 trick，否则柜子就是一个实心 convex，物体永远掉在外面
- 启用 gravity + repulsive forces，让小物体自然 settle

这种"gradient-based + physics-based"两阶段策略让我想到刚性 body tracking 中的 optimization-then-ICP 范式——粗对齐用解析方法，精对齐用物理约束。

---

## 5. Infinigen Indoors 的 Simulated Annealing

InternScenes-Gen 部分依赖 Infinigen Indoors 的约束求解器，其核心是 simulated annealing：

$$p(s' | s) = \min\left[\exp\left(\frac{l(s) - l(s')}{\tau}\right), 1\right]$$

变量含义：
- $s$：当前 scene state
- $s'$：proposed state（随机 move 后，如 add/rotate/remove 一个 object）
- $l(s)$：state $s$ 在 constraint graph 上的 loss（违反约束的总和）
- $\tau$：当前温度，从 0.25 冷却到 0.001

直觉：温度高时 $\tau$ 大，$\exp(\Delta l / \tau)$ 接近 1，接受坏解概率高 → 探索；温度低时 $\tau$ 小，几乎只接受 $\Delta l < 0$ 的好解 → 利用。这避免了陷入"先放沙发靠左墙、再放电视柜靠右墙、最后中间茶几穿模"这种 local optimum。

Infinigen 用了 17 generators + 216 controllable parameters for furniture alone，这种高自由度随机化是为了避免 reusing static models 引入的 distribution bias。对训练 generative model 来说这点至关重要——如果训练集里同一把椅子出现 5000 次，模型会 overfit 这个特定 mesh 而不是学椅子概念。

---

## 6. Articulated Object 集成

约 20% 物体替换为 PartNet-Mobility 的 articulated URDF assets。Pipeline 伪代码：

```python
if obj_info.type == "urdf":
    IsaacSim.URDF_Importer.import(
        urdf_path=full_asset_path,
        prim_path=prim_path,
        create_articulation=True  # key for interactivity
    )
```

这个 `create_articulation=True` 是关键——它把 URDF 的 kinematic tree 转成 Isaac Sim 的 articulation，保留 joint limits、friction、drive type。这对 manipulation task 训练很重要：drawer 拉开、oven 门打开、faucet 旋转这些动作都需要真实的 articulation semantics。

参考 PartNet-Mobility 主页: https://sapien.ucsd.edu/

---

## 7. 数据集统计的关键信号

| 指标 | 数值 | 含义 |
|------|------|------|
| Total scenes | 39,870 | |
| Total regions | 48,381 | 15 types |
| Total objects | 1.96M | |
| Object categories | 288 | 同 EmbodiedScan |
| CAD models | 800K | 来自 Objaverse + PartNet-Mobility |
| Avg objects/region | 41.5 | 行业最高（3D-FRONT 6.9, Structured3D 21.1） |
| Avg density | 1.296 obj/m² | |
| Containment/support relations | 3.45 per object (5.57 if non-zero) | |

**Containment relation 是关键**：5.57 这个数字意味着每个"支撑/容器"物体平均与 5-6 个其他物体有空间关系。这种 dense relational structure 正是 scene graph prediction、task planning、manipulation 这些 task 的核心训练信号。3D-FRONT 这种 sanitized dataset 里这个数接近 0，模型学不到"书在桌上、笔在书上"这种 hierarchical support 关系。

---

## 8. Experiment 1: Scene Layout Generation

### 8.1 实验设置

- 3 个 region type：Resting / Living / Dining
- 两个版本：Full（全物体）vs Simplified（仅 45 类大家具）
- 3 baselines：ATISS（autoregressive transformer）、DiffuScene（diffusion）、PhyScene（diffusion + physics guidance）
- 4 metrics：FID、KID、SCA、CKL
- 所有 baseline 用 InternScenes 的 800K asset library 重新训练了一个 VAE 做点云压缩

### 8.2 结果分析（Table 2）

关键观察：

**Simplified Version（仅大物体）**：
- ATISS: FID 23.20 / 30.49 / 30.89
- DiffuScene: FID 22.88 / 23.54 / 28.70
- PhyScene: FID 23.78 / 24.75 / 26.76
- 三者接近，DiffuScene 略优

**Full Version（含小物体）**：
- ATISS: FID 101.85 / 104.48 / 133.20（**5x 退化**）
- DiffuScene: FID 96.56 / 107.49 / 122.95
- PhyScene: FID 88.02 / 66.59 / 130.39（**Living region 上明显领先**）

**直觉构建**：
1. FID 从 23 → 100+ 不是模型变差，而是任务变难——小物体分布是 multi-modal 的（同一种桌上的物体组合有无数种），生成模型需要建模的组合爆炸了
2. PhyScene 在 Living region 上 FID 66.59 远好于 DiffuScene 107.49，说明 **physics guidance 对 cluttered scene 有效**——PhyScene 的 collision loss + repulsive force 让小物体不会重叠/浮空
3. SCA 在 simplified 版本接近 50-70%（健康），full 版本 95-99%（**过拟合 training distribution**，说明生成分布 collapse 了）

### 8.3 这个 benchmark 的价值

它揭示了一个 scaling law 形态的问题：当前 scene generation 模型的 object token 数受限于 attention 的 $O(n^2)$ 复杂度。50 objects 已经接近极限，InternScenes 平均 41.5 objects 已经把模型推到崩溃边缘。如果要 scaling 到 100+ objects，需要新的架构——可能是 hierarchical diffusion（room-level → region-level → object-level）或者 sparse attention。

---

## 9. Experiment 2: PointGoal Navigation

### 9.1 实验设置

- **Simulator**: Isaac Sim（物理 realistic，区别于 Habitat-Sim 的 kinematic-only）
- **Robot**: ClearPath Dingo（差速驱动 wheeled robot）
- **Scenes**: 20 from Real2Sim + 10 from Gen
- **Metrics**: Success Rate、SPL（Shortest Path Length）、Distance
- **Episodes**: 每 scene 20 episodes，起点终点距离 3-10m，过滤 ESDF > 0.5m 可通行区域
- **Baselines**: 
  - DD-PPO（RL，离散动作空间）
  - NavDP（diffusion-based imitation learning）
  - NavDP-FT（用 InternScenes 资产生成 118,784 trajectories 做微调）

DD-PPO 的动作映射 trick：原方法是离散 4-action {forward, turn-left, turn-right, stop}，映射到连续速度：
$$\{(u=0.5, \omega=0.0), (u=0.0, \omega=1.0), (u=0.0, \omega=-1.0), (u=0.0, \omega=0.0)\}$$
其中 $u$ 是 linear speed (m/s)，$\omega$ 是 angular speed (rad/s)。

NavDP 的映射：选 trajectory 的第 4 个 waypoint，转成 $u, \omega$：
$$u = K_u \cdot \|\text{waypoint}_{4}\|_2, \quad \omega = K_w \cdot \text{yaw}(\text{waypoint}_4, \text{current\_pose})$$
$K_u, K_w$ 是控制增益系数。

### 9.2 结果分析（Table 3）

| Method | Real2Sim Success | Real2Sim SPL | Gen Success | Gen SPL |
|--------|------------------|--------------|-------------|---------|
| DD-PPO | 23.6 | 23.1 | 45.0 | 44.2 |
| NavDP | 48.3 | 45.3 | 61.9 | 61.8 |
| NavDP-FT | 51.0 | 49.4 | 63.6 | 61.7 |

**关键观察**：

1. **DD-PPO 在 Real2Sim 上只有 23.6%**，Gen 上 45%。差距来自 Real2Sim 的 clutter 程度更高（来自真实 scan，小物体多、布局混乱）。DD-PPO 训练于 Habitat-Sim 的相对空旷 MP3D/HM3D 场景，迁移到 cluttered environment 出现严重 sim-to-sim gap。

2. **NavDP 比 DD-PPO 高 ~25%**。NavDP 是 diffusion policy，critic function 学的是 trajectory quality，对 cluttered scene 的 robustness 更强。这也印证了你之前说过的"diffusion policy 在多模态目标分布下更鲁棒"。

3. **NavDP-FT 仅提升 ~2.7%**。这看似 disappointing，但其实是 key insight——单纯加数据 scaling 已经不能解决 cluttered navigation 问题，需要 architectural innovation（比如 explicit memory、collision recovery module）。

### 9.3 三个核心挑战（论文 Discussion）

作者识别了三个 sim-to-real gap 的关键源：

**Challenge 1: Cluttered Layout 需要 Collision Recovery**
现有 navigation policy 一旦碰撞就卡住，没有"后退-绕行"的 recovery behavior。这让我想到 you areRobot 的 primal-dual RL 思路——把 collision 当作 constraint violation，policy 需要学 dual update 来恢复。

**Challenge 2: Narrow Pathway 需要 Embodiment-aware Traversability**
论文举的例子很具体：办公椅的 5 个腿形成窄通道，Dingo（40cm 宽）能过但 Fetch（60cm 宽）不能过。现有方法只用 exteroceptive（视觉/激光）观测，没有把 robot footprint 显式编码进 traversability map。这指向一个新方向：**embodied ESDF**——把 robot shape 卷进 ESDF 计算。

**Challenge 3: Small Connected Components 的 Perception**
椅子腿、桌腿在视觉观测中只有几帧可见，但是 critical collision source。这是典型的 **long-tail perception** 问题——训练数据中 chair leg 出现的视角占比 <5%，但碰撞后果 100%。

这三个 challenge 实际上定义了一个新的 navigation benchmark paradigm——**physics-aware, embodiment-aware, long-tail-aware** navigation，远超 HabitatChallenge 的 PointNav 设定。

---

## 10. 系统性能数据（Table 6-8）

这组数据对实际部署很重要：

| Scene Type | Parallel=1 FPS | Parallel=40 FPS | Parallel=40 GPU Mem |
|------------|-----------------|------------------|---------------------|
| Real2Sim | 246.95 | 131.86 | 5.385 GB |
| Gen | 263.95 | 200.07 | 5.679 GB |
| Synthetic | 175.04 | 51.61 | 8.168 GB |

**直觉**：
- Real2Sim/Gen 场景 mesh 较简单（来自 procedural 或 Objaverse 替换），FPS 高
- Synthetic 场景是 designer 全套 mesh，CPU 占用 73% → 79%（接近瓶颈），FPS 在 Parallel=40 时只有 51——这暗示着大规模 RL 训练时 Synthetic subset 会成为 throughput 瓶颈
- GPU memory 都在 8GB 以下，单卡 4090 可以跑 40 并行环境

---

## 11. 与你工作方向的联想

### 11.1 Ego-Learning / VPT 视角
InternScenes 的 Real2Sim 部分本质上是为 sim-to-real 提供"可交互的世界模型"。你 VPT 中用 YouTube 视频学 Minecraft policy，但 Minecraft 没有 articulated object。InternScenes 把真实世界扫描 + 800K Objaverse asset + PartNet-Mobility URDF 融合后，第一次让"在真实布局上做大规模 RL" 成为可能。

### 11.2 LLM-World Model 视角
这篇 paper 揭示的 scene generation 失败模式（FID 100+ on full version）实际上暗示了一个 deep 问题：当前 autoregressive/diffusion scene generator 都在做 **flat token sequence modeling**，但 scene 的本质是 hierarchical relational graph。如果用 LLM-as-world-model 思路，scene 应该被表示成 nested scene graph，类似你说的 "World Model as Token Sequence" 但保留 hierarchical structure。

### 11.3 Scaling Laws 视角
NavDP-FT 只提升 2.7% 这点最值得深思。你之前研究 scaling laws 时发现 model capacity / data / compute 是 power law，但 navigation 在 InternScenes 上似乎 break 了 power law——数据从 0 → 118K trajectories，性能几乎不动。可能的解释：
- 任务 difficulty scaling 比 data scaling 快
- 任务需要 architectural prior（如 memory、collision recovery）而不是更多 data
- 这其实就是 LLM-RL 中常见的 "data wall" 在 embodied AI 上的体现

### 11.4 与 MetaScenes、ACDC、MIDI 的对比
- **MetaScenes** (Yu et al., 2025): concurrent work, 同样做 ScanNet Real2Sim，但单一数据源
- **ACDC** (Dai et al., 2024): 用 VLM 从单图重建场景，但 fails on cluttered small objects
- **MIDI** (Huang et al., 2024): single-image multi-instance diffusion，但 artifact 多、object entanglement 严重
- **InternScenes** 的优势是 multi-source + 物理仿真 + 小物体保留，但代价是 manual annotation 比例高（Section 6 提到的 limitation）

---

## 12. Open Questions 与未来方向

1. **Manual annotation 的 scaling**：Synthetic subset 还需要人工标注 region type，能否用 VLM 自动化？
2. **Asset quality**：Objaverse 资产 PBR 材质、几何精度参差不齐，影响 sim-to-real。需要做 quality scoring + filtering
3. **Long-horizon task**：当前只 benchmark 了 layout generation 和 point-goal nav，但 InternScenes 真正价值在 manipulation（开抽屉、做饭）和 long-horizon planning（"做早餐"）
4. **Dynamic scenes**：当前所有 scene 是 static 的，但真实世界是 dynamic（人移动、门开合）。需要把 scene dataset 升级成 4D（spacetime）
5. **Language grounding**：288 类已不错，但 missing spatial relation language（"the book on the left side of the lamp"）。把 InternScenes 升级成 grounded scene language 类似 3D-LLM 但更大规模
6. **Generative world model on InternScenes**：能否训一个 video diffusion world model 在 InternScenes 上学物理先验？这接近 Sora 路径但有 ground truth geometry

---

## 13. 关键参考链接

**数据集与代码**:
- Project page (待开源): 跟踪 https://github.com/opendatalab/InternScenes 或 Shanghai AI Lab 主页
- EmbodiedScan: https://github.com/Tai-Wang/EmbodiedScan  
- Infinigen: https://infinigen.org/  
- 3D-FRONT: https://3d-front.org/  
- Objaverse: https://objaverse.allenai.org/  
- PartNet-Mobility: https://sapien.ucsd.edu/  
- SAPIEN: https://sapien.ucsd.edu/  

**Baselines**:
- ATISS: https://github.com/nv-tlabs/ATISS  
- DiffuScene: https://github.com/zlzeng/DeepLayoutSynthesis  
- PhyScene: https://github.com/yang-yang-666/PhyScene  
- DD-PPO: https://github.com/facebookresearch/habitat-lab  
- NavDP: https://github.com/Chenjia-Bai/NavDP  

**工具链**:
- COACD (convex decomposition): https://github.com/Weixiao-Hong/CoACD  
- Isaac Sim: https://developer.nvidia.com/isaac-sim  
- InternVL: https://github.com/OpenGVLab/InternVL  

**Concurrent / Related**:
- MetaScenes: https://arxiv.org/abs/2503.18450  
- ACDC: https://arxiv.org/abs/2410.14066  
- MIDI: https://arxiv.org/abs/2412.03558  
- SceneScript: https://meta-ai.github.io/scenescript/  
- SceneVerse: https://github.com/Chenjia-Bai/SceneVerse  

---

## 14. 总结的 Intuition

这篇 paper 的核心 insight 我会这样总结：**Embodied AI 的 bottleneck 不在算法，而在数据环境的 realism-density trade-off**。过去十年我们在 simulator 容易性（Habitat MP3D）和 realism（Real-world scan）之间二选一，InternScenes 通过 Real2Sim + 物理后处理第一次同时拿到 realism + density + simulatability。代价是 engineering labor（manual annotation、asset curation、physics tuning），但回报是让 scene generation 和 navigation 都暴露出新的 scaling wall——这些 wall 才是下一代 architecture（hierarchical scene diffusion、embodiment-aware navigation、physics-LLM world model）的真正机遇。

如果你接下来要在这个数据集上做事，我猜最有 impact 的方向是 **在 InternScenes 上训一个 3D world model**，把 scene 的 hierarchical relation 学到 latent 中，然后做 long-horizon task planning——这与你 World Model as Token Sequence 的思路高度契合，而且 InternScenes 提供了第一次具备足够 realism + scale 的训练数据。
