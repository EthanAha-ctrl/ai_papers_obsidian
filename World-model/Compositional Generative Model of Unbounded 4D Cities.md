---
source_pdf: Compositional Generative Model of Unbounded 4D Cities.pdf
paper_sha256: 5308d5ab19a2693cf919366b638d03a25650035635fa9b8517bcc2797316a001
processed_at: '2026-08-03T16:46:54-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CityDreamer4D 用人话说

## 这paper到底在干嘛

想象你玩Minecraft或者SimCity，想生成一个无限的、随时间变化的城市。开车的时候车在动，建筑不动，路边的树也不动。这就是4D城市——3D空间加上时间维度。

CityDreamer4D就是想自动生成这种东西。难点在于人对城市太熟悉了，你看到一栋歪掉的楼、一辆形状奇怪的车，立马就觉得不对劲。自然场景（山、树、草地）容错率高很多，因为大家也没那么在意某棵树长什么样。

## 为什么以前的方法不行

之前的方案大概分几类：

**视频生成路线**：生成一段视频当3D场景用。问题是你换个角度看，前后帧对不上，墙会飘、路会扭。

**图像外推路线**：给你一张图，往四周延展。问题是没有真正的3D结构，只是2D图片拼贴，走不远。

**程序化生成路线**：用规则写代码生成城市（游戏引擎那种）。问题是多样性受限于你手头有多少3D asset，而且不像generative model那样能学习真实数据的分布。

**3D-aware GAN路线**（SceneDreamer、InfiniCity这些）：用neural rendering在真3D空间里生成。这个方向最靠谱，但有个致命问题——它们把所有建筑、所有车都当成同一个类别处理。你想啊，一栋纽约摩天楼和一栋小商铺，在它们眼里都是"building"这个label，生成的结果就是所有建筑看起来都差不多，或者结构乱七八糟。

4D生成更惨，之前的方法要么只能搞小物体级别的4D，要么保证不了时间一致性，要么scale上不去。

## CityDreamer4D的核心思路

作者的关键insight特别简单：**不同东西的统计特性不一样，应该用不同方法处理**。

具体来说，城市里的东西可以分两大类：

**Stuff（背景类）**：路、草地、天空、水面。这些东西纹理不规则，但同类之间长得差不多——全世界的沥青路看起来都差不多。

**Instance（实例类）**：建筑、车辆。每个实例都独一无二——每栋楼的facade不一样，每辆车的款式不一样。但它们内部有很强的结构规律——楼有重复的窗户排列，车有固定的前后左右。

所以CityDreamer4D把生成过程拆成几个独立的generator，每个专门处理一类东西，用最适合它的表征方式。

## 拆解流水线

整个pipeline大概长这样：

```
先决定城市长什么样（layout）
    ↓
背景生成器：路、树、天空
    ↓
建筑生成器：一栋一栋生成建筑
    ↓
交通生成器：决定车怎么跑
    ↓
车辆生成器：一辆一辆生成车
    ↓
合成器：把所有东西拼到一起
```

### 第一步：生成城市layout

Layout就是城市的骨架——哪里是路、哪里是楼、哪里是绿地，以及每个地方的高度。

表示方法很直观：一张2D的semantic map（标记每个像素是什么类别）加上一张height field（标记每个位置有多高），然后把2D往3D挤压，就得到3D的城市layout。

这里有个小但重要的改进——之前只有top-down height（从地面往上的高度），CityDreamer4D加了bottom-up height（从上往下的边界）。为什么？因为高架桥这种东西，桥面在空中，下面有空间。你只有一个高度值表示不了"中间是空的"这件事。

生成layout用的是VQVAE加MaskGIT的组合。VQVAE把semantic map和height field压缩成离散token，MaskGIT用autoregressive的方式一个一个预测token。要生成无限大的城市，就用滑动窗口，每次生成一块，和前一块重叠25%，这样边界能接上。

### 第二步：生成背景

背景包括路、植被、天空这些。

场景表征用BEV（鸟瞰图）——只存2D的height field和semantic map，不存dense 3D voxel。这样省内存，城市scale也能扛住。

关键设计在scene parameterization。这里用了**generative neural hash grid**。原理是这样的：

先有一个encoder把整个local scene压成一个极低维的向量 $\mathbf{f}_G$（只有2维！）。然后对每个3D位置 $\mathbf{p}$，用一个hash function把 $\mathbf{p}$ 和 $\mathbf{f}_G$ 一起映射到hash table的一个entry，取出对应的feature。

hash function用的是XOR加上大质数乘法，这是经典hashing技巧（其中一个质数2654435761是Knuth推荐的那个）。

为什么这么做？因为hash lookup是O(1)的，而且hash table的size固定（$2^{19}$个entry），不用存dense voxel grid。generative的关键在于hash function的input包含了scene-level feature $\mathbf{f}_G$，所以同一个位置在不同scene里会映射到不同feature，这就实现了跨scene的generalization。

背景stuff的纹理不规则，hash grid的随机性正好匹配——你不需要刻板的结构，你需要的是丰富的纹理细节。

### 第三步：生成建筑

这里就开始体现compositional design的威力了。

每个建筑单独生成。先从layout里把某个建筑抠出来，做成一个local scene。注意一个细节：所有建筑在semantic map里都是同一个label，所以要先做connected components detection把每个建筑分离出来。分离之后，还要把建筑的facade和roof标记成不同label——因为真实世界里楼顶和楼侧面的视觉差异很大。

Scene parameterization跟背景完全不一样。这里用**local encoder加SinCos positional encoding**。

具体来说，encoder把local scene编成pixel-level feature（每个BEV位置一个63维向量）。然后对3D位置 $(p_x, p_y, p_z)$，先取 $(p_x, p_y)$ 对应的feature，拼上 $p_z$，再做NeRF那种sin/cos的positional encoding：

$$\mathcal{O}(\mathbf{x}) = \{\sin(2^i \pi \mathbf{x}), \cos(2^i \pi \mathbf{x})\}_{i=0}^{9}$$

为什么建筑用SinCos而背景用hash grid？因为建筑facade有强烈的周期性——窗户一层一层重复，楼层有规律。SinCos天然适合表示周期函数，hash grid的随机性反而会把这种规律打乱。

渲染的时候加了个style code $\mathbf{z}$，让同样结构的楼可以有不同的外观。这个style code就是控制"这栋楼是玻璃幕墙还是红砖"这种事情。

训练的时候建筑生成器只用了GAN loss，没有reconstruction loss。原因是每栋楼都不一样，你没有一个固定的ground truth去reconstruct。GAN loss给的是distribution层面的监督——生成的楼要看起来像真楼的分布，但不需要和某栋特定的真楼一模一样。

### 第四步：生成交通场景

这部分是CityDreamer4D相比前作CityDreamer的新增内容。

先从城市layout推导出HD map（高精地图）。HD map比普通layout多了车道线、路口、交通灯这些信息。推导过程是一系列图象处理操作：

- 对road区域做Canny边缘检测得到road edges
- 做skeletonization提取road骨架，找路口，用Bezier曲线连车道
- 根据车道位置和属性生成road lines（白线、黄线等）
- 在路口放stop signs和traffic lights

有了HD map，用一个现成的模型（TrafficGen）来决定每一帧哪些车在哪里、有多大。这样就得到了time-varying的traffic scenario。

### 第五步：生成车辆

车辆生成器有一个很巧的设计——**canonical feature space**。

什么意思呢？每辆车先平移到原点，然后旋转到标准朝向。具体就是对3D点 $\mathbf{p}$ 做变换：

$$\mathbf{p}^C = \mathbf{R}(\mathbf{p} - \mathbf{center})$$

其中 $\mathbf{R}$ 是用yaw角 $\theta$ 和pitch角 $\gamma$ 构造的旋转矩阵。

为什么车辆要canonicalize而建筑不需要？因为车是紧凑形状且结构规律性强——所有车都有前、后、侧面，这种结构在不同车之间是共享的。Canonicalization让网络能跨车共享feature，学起来更容易。

建筑就相反，每栋都独特，你把它转到标准朝向也没意义，因为没有"标准建筑"这回事。

车辆生成器用global encoder（和背景一样），而不是建筑的local encoder。这也是因为车辆之间共享结构——global encoder能提取跨instance的共性。

训练车辆生成器用了reconstruction loss加GAN loss。为什么车辆能用reconstruction而建筑不行？因为车的appearance变化比建筑小得多，有相对稳定的reconstruction target。

### 第六步：合成

最后用一个简单公式把所有东西拼起来：

$$\mathbf{I}_C = \hat{\mathbf{I}}_G \mathbf{M}_G + \sum \hat{\mathbf{I}}_{B_i} \mathbf{M}_{B_i} + \sum \hat{\mathbf{I}}_{V_i} \mathbf{M}_{V_i}$$

就是背景图乘背景mask，加上每栋楼图乘楼mask，加上每辆车图乘车mask。

这里有个隐含问题——没有learned blending，纯mask硬拼。如果mask边界不准，会有可见的拼接痕迹。这是compositional方法常见的trade-off，灵活性和无缝整合不可兼得。

## 数据集这件事

这篇paper花了很多笔墨在数据集上，这也是它的一大贡献。

**OSM数据集**：从OpenStreetMap爬了80个城市的semantic map和height field，覆盖6000多平方公里。这个数据集提供layout的真实分布。

**GoogleEarth数据集**：用Google Earth Studio绕纽约飞了400圈，拿到24000张图。annotation是自动生成的——把OSM的3D annotation用Google Earth的相机参数投影到2D图上。聪明，但有问题：Google Earth在地面附近重建差，没有真正的street view；OSM数据本身有误差；高架桥的高度信息缺失。

**CityTopia数据集**：为了解决GoogleEarth的问题，作者用Houdini和Unreal Engine搭了11个虚拟城市，用了CitySample项目的5000个高质量3D asset。渲染时做了8倍空间超采样加32倍时间超采样来避免摩尔纹。白天黑夜都渲染，夜晚场景把阳光去掉让网络更容易学光照一致性。总共37500张图，有精确的2D和3D annotation。

这三个数据集互补：OSM给layout，GoogleEarth给真实视觉，CityTopia给高质量标注和street view。

## 实验结果说明了什么

定量比较里CityDreamer4D全面领先。几个关键数字：

- FID在GoogleEarth上96.83，最好的baseline PersistentNature是123.8，降了22%
- KID在CityTopia上0.049，DimensionX是0.070，降了30%
- VBench（综合视频质量）0.825，最高
- Depth Error和Camera Error都是最低，说明3D几何和多视角一致性最好

Ablation studies最有意思，验证了每个设计选择的必要性：

**去掉建筑生成器**，FID从88.48飙到195.1。这说明把建筑和背景混在一起生成根本不行。

**去掉instance labels**，FID从88.48到167.8。所有楼用一个label，它们会塌缩到同一个样子。

**建筑用hash grid替代SinCos**，FID从88.48到196.8。hash grid破坏了facade的周期性结构。

**建筑用global encoder替代local encoder**，FID从88.48到182.3。global encoder导致facade pattern重复——所有楼看起来像复制粘贴的。

**车辆去掉canonicalization**，FID从142.3到273.4。车的结构乱掉。

**车辆用local encoder替代global encoder**，FID从142.3到200.5。没有跨车共享feature，学不好。

这些ablation共同验证了核心论点：不同object需要不同的representation和parameterization。

## VLN实验很说明问题

这篇paper还做了个很有意思的实验——用生成的4D城市测试VLM（vision-language model）的导航能力。

给VLM看当前视角的图加上一句话指令（比如"去前面那个红色建筑"），VLM从12个离散动作里选一个执行，直到它说"stop"。

结果很有意思：人类成功率100%，GPT-4o最好也只有36%，InternVL3在开源模型里最好25.6%。这说明即使是最强的VLM，在复杂城市环境里做空间推理还是很菜。

这个实验暗示了CityDreamer4D的一个潜在价值——生成的4D城市可以当embodied AI的benchmark generator用。你需要各种城市环境训练机器人导航，不可能都去真实世界采，生成模型就很重要。

## 几个值得琢磨的点

**Sequential generation的代价**。每栋楼、每辆车都单独forward pass，城市越大越慢。这是compositional设计固有的代价。好处是你能单独编辑某个instance，坏处是inference慢。

**没有全局光照**。因为背景、建筑、车是分开生成的，它们之间没有光照交互。白天还好，夜晚场景就很明显——楼里的灯不会照亮旁边的路，车灯不会在地上投光。这是compositional方法的根本限制，要解决得做inverse rendering加全局光照，很难。

**Compositor是硬拼的**。没有learned blending，靠mask叠加。如果mask边界不完美，会有artifacts。更好的做法可能是训一个小网络专门学怎么blend，但这样又引入了新的训练难度——没有composited ground truth。

**行人是后加的**。paper最后展示了行人过马路的例子，但那是用MoMask生成动作再retarget到avatar上渲染的，并没有真正集成到framework里。要真正支持多agent动态，还有很长的路。

## 我的直觉总结

CityDreamer4D给我的最大启发是：**复杂生成问题的核心不是更大的模型端到端学，而是正确地拆解问题让每部分用对的工具**。

这跟传统工程思路很像——你不会用一把锤子搞定所有东西，钉子用锤子，螺丝用螺丝刀，胶水粘纸。CityDreamer4D就是把城市生成拆成几个子问题，每个子问题选了最匹配其统计特性的representation：

- 不规则纹理的背景 → hash grid的随机性
- 周期性结构的建筑 → SinCos的周期性
- 共享结构的车辆 → canonical space的对称性

这种"representation要match object statistics"的原则其实挺通用的，不只限于城市生成。

另一个启发是数据集作为一等公民。这篇paper三个数据集各司其职，解决了不同瓶颈。在foundation model时代，数据质量决定上限，模型架构决定下限。花精力搞数据往往比花精力调模型更值得。

最后的limitation其实也指明了方向——全局光照、learned compositor、真正的多agent动态、Gaussian Splatting替代NeRF加速——每一个都是可以follow up的课题。作者诚实地列出来，反而让人觉得整个approach是靠谱的，因为你知道它的边界在哪里。

如果让我预测下一步，大概率是Gaussian Splatting版本的CityDreamer4D——作者后续已经在CVPR 2025发了CityGaussian做3D城市，4D版本应该在路上。那样inference speed能大幅提升，compositor的blending也能做得更自然。

---

# CityDreamer4D: Compositional Generative Model of Unbounded 4D Cities — 深度技术解析

## 1. High-Level Motivation 与 核心Insight

这篇paper解决的核心问题非常ambitious：**生成无界的4D城市**，即time-varying、spatially unbounded的urban environments。作者认为city generation比3D scene generation难得多，主要因为三点：

1. **Buildings与vehicles的appearance variance极大**：不同于自然场景中同一semantic class的appearance相对一致（如所有树看起来差不多），buildings与vehicles即便在同一class内也有巨大appearance variation
2. **Human对urban environment极其敏感**：我们对city的geometry与texture非常familiar，任何distortion都会被立即察觉
3. **4D consistency极难维护**：dynamic objects的motion需要physically plausible，static scenes需要temporal stable

CityDreamer4D的核心insight是**compositional decomposition**：将4D city拆解为static scenes与dynamic objects，且每个object type用不同的neural field表征。这种disentanglement的指导思想来自observation：background stuff（road、vegetation、sky）有irregular textures但similar appearance，而instances（buildings、vehicles）有regular periodic patterns但diverse appearances。

参考链接：
- Project page: https://citydreamer4d.github.io/
- CVPR 2024 版本（CityDreamer）: https://arxiv.org/abs/2403.01548

---

## 2. 整体架构解析

### 2.1 Pipeline Overview

整条pipeline非常像rendering engine的neural version：

```
City Layout L ──> City Background Generator ──> Î_G, M_G
                                       │
                                       ├──> Building Instance Generator ──> {Î_B_i, M_B_i}
                                       │
                                       └──> HD Map ──> Traffic Scenario T_t
                                                              │
                                                              └──> Vehicle Instance Generator ──> {Î_V_i^t, M_V_i^t}
                                                                                                       │
                                                            Compositor <─────────────────────────────┘
                                                              │
                                                              v
                                                          Î_C^t (final 4D city frame)
```

关键设计选择：
- **Static scene** 与 **dynamic objects** 用两个不同的generator chain
- **Buildings** 与 **vehicles** 进一步用独立的instance generator
- **Background stuff** 用stuff-oriented neural field
- **Instances** 用instance-oriented neural field

### 2.2 Compositional设计背后的Inductive Bias

每个generator都针对其object class的特点设计了专属的scene parameterization：

| Generator | Object Type | Scene Parameterization | Inductive Bias |
|-----------|-------------|------------------------|----------------|
| City Background Generator | Background stuff (road, vegetation, sky) | Generative neural hash grid | Similar appearance, irregular textures |
| Building Instance Generator | Buildings | Local encoder + SinCos positional encoding | Diverse facades with regular periodic patterns |
| Vehicle Instance Generator | Vehicles | Canonical feature space + Global encoder + SinCos | Compact shapes, strong structural regularity |

这种设计避免了SceneDreamer与InfiniCity的核心缺陷：将所有instances赋予同一semantic label会导致它们collapse到同一appearance。

---

## 3. 方法详解

### 3.1 Unbounded Layout Generator (ULG)

#### 3.1.1 City Layout Representation

City layout表示为3D volume **L**，通过extruding semantic map **S_L**与height field **H_L** = {H_L^BU, H_L^TD}构建：

$$\mathbf{L}(i,j,k) = \begin{cases} \mathbf{S}_L(i,j) & \text{if } \mathbf{H}_L^{BU}(i,j) \leq k \leq \mathbf{H}_L^{TD}(i,j) \\ 0 & \text{otherwise} \end{cases}$$

变量解析：
- $(i,j,k)$: voxel的3D grid index
- $\mathbf{S}_L(i,j)$: 在位置$(i,j)$的semantic label（如road、building、vegetation、water）
- $\mathbf{H}_L^{BU}(i,j)$: bottom-up height，即从ground起算的下方边界
- $\mathbf{H}_L^{TD}(i,j)$: top-down height，即上方边界
- 0: 表示empty space

**关键insight**：相比CityDreamer只使用top-down height，CityDreamer4D引入bottom-up height的目的是表示**hollow structures**如highway（高架桥下方有空间）。这是一个看似小但功能强大的extension。

#### 3.1.2 Generation via VQVAE + MaskGIT

ULG使用VQVAE [van den Oord et al., NIPS 2017] 对semantic map与height field的patches进行tokenization，然后用MaskGIT [Chang et al., CVPR 2022]进行autoregressive generation。

VQVAE的codebook $\mathcal{C} = \{\bar{c}_k | c_k \in \mathbb{R}^{d_C}\}_{k=1}^{d_K}$，其中：
- $d_K = 512$: codebook size
- $d_C = 512$: 每个code的dimension

**Unbounded extrapolation strategy**：VQVAE产生fixed-size outputs，所以用sliding window with 25% overlap iteratively预测local layout tokens。这种overlap策略的关键在于保证window boundaries的continuity。

#### 3.1.3 Loss Function

VQVAE的训练目标包含三部分：

$$\ell_{VQ} = \lambda_R \|\hat{\mathbf{H}}_L^p - \mathbf{H}_L^p\| + \lambda_S \mathcal{S}(\hat{\mathbf{H}}_L^p, \mathbf{H}_L^p) + \lambda_E \mathcal{E}(\hat{\mathbf{S}}_L^p, \mathbf{S}_L^p)$$

变量解析：
- $\hat{\mathbf{H}}_L^p$, $\mathbf{H}_L^p$: generated与ground truth的height field patch
- $\hat{\mathbf{S}}_L^p$, $\mathbf{S}_L^p$: generated与ground truth的semantic map patch
- $\mathcal{S}$: Smoothness Loss [Meister et al., AAAI 2018]，增强building edges的sharpness
- $\mathcal{E}$: Cross-Entropy Loss
- $\lambda_R = 10$, $\lambda_S = 10$, $\lambda_E = 1$: loss weights

MaskGIT的autoregressive transformer用reweighted ELBO loss [Bond-Taylor et al., ECCV 2022]训练。

参考：
- MaskGIT: https://arxiv.org/abs/2203.05239
- VQVAE: https://arxiv.org/abs/1711.00937

---

### 3.2 Traffic Scenario Generator

#### 3.2.1 Representation

Traffic scenario $\mathcal{T} = \{\mathbf{T}_t\}_{t=1}^{n_T}$，其中$n_T$是frame数。每个$\mathbf{T}_t$的定义与city layout L的形式完全相同，只是$\mathbf{S}_{T_t}$与$\mathbf{H}_{T_t}$描述的是dynamic objects的位置与高度。

#### 3.2.2 HD Map Generation

这是CityDreamer4D相比CityDreamer的key extension之一。HD map包含五类entities（采用Waymo Motion dataset的definition）：

1. **Road Edges**: 对$\mathbf{S}_L$应用Canny edge detection [Canny, TPAMI 1986]，然后vectorize成graph（detect corners并按顺序连接）
2. **Road Lanes**: 对$\mathbf{S}_L$做skeletonization [Zhang & Suen, 1984]提取road structure，识别intersections，再用graph-based traversal转换成road centerline graphs。Lane数量与位置由road width决定，intersection处的lanes用Bezier curves连接
3. **Road Lines**: 根据road lanes的position与attribute生成（如solid single white、solid double yellow）
4. **Stop Signs**: 放置在intersections处
5. **Traffic Lights**: 同样在intersections处

然后使用off-the-shelf model [TrafficGen, Feng et al., ICRA 2023]根据HD map决定per-frame的dynamic object bounding boxes。

**Intuition**：这种pipeline将traffic generation解耦为（1）HD map construction（确定性graph operations）与（2）traffic behavior generation（probabilistic model），使得两部分可以独立改进。

参考：
- Waymo Motion dataset: https://arxiv.org/abs/2104.10433
- TrafficGen: https://arxiv.org/abs/2210.03596

---

### 3.3 City Background Generator

#### 3.3.1 BEV Scene Representation

借鉴SceneDreamer [Chen et al., TPAMI 2023]，使用bird's-eye-view (BEV) representation而非voxel corner parameterization（如GANCraft、InfiniCity）。BEV representation的优势是：
- Memory-efficient（只存2D height field + semantic map，而非dense 3D voxel）
- Expressiveness sufficient for city-scale scenes
- 与city layout的天然契合

Local window resolution：
- GoogleEarth: $N_G^H = 1536$, $N_G^W = 1536$, $N_G^D = 640$
- CityTopia: $N_G^H = 3072$, $N_G^W = 3072$, $N_G^D = 2560$

#### 3.3.2 Generative Neural Hash Grid

这是background stuff的核心scene parameterization。目标是将3D position $\mathbf{p}$映射到feature $\mathbf{f}_G^\mathbf{p}$，并使feature generalizable across scenes。

Step 1: 用global encoder $E_G$编码local scene到scene-level feature $\mathbf{f}_G \in \mathbb{R}^{d_G}$：

$$\mathbf{f}_G = E_G(\mathbf{H}_L^G, \mathbf{S}_L^G)$$

其中$d_G = 2$（极低维度！这是个值得注意的设计选择）。

Step 2: 用neural hash function $\mathcal{H}$将$\mathbf{p}$与$\mathbf{f}_G$映射到feature index：

$$\mathbf{f}_G^\mathbf{p} = \mathcal{H}(\mathbf{p}, \mathbf{f}_G) = \left(\bigoplus_{i=1}^{d_G} f_G^i \pi^i \bigoplus_{j=1}^{3} p^j \pi^j\right) \mod N_E$$

变量解析：
- $\oplus$: bitwise XOR operation
- $\pi^i$, $\pi^j$: distinct large prime numbers
  - $\pi^1 = 1$
  - $\pi^2 = 2654435761$（Knuth's multiplicative hash constant！）
  - $\pi^3 = 805459861$
  - $\pi^4 = 3674653429$
  - $\pi^5 = 2097192037$
- $N_E = 2^{19}$: hash grid size
- $\mod$: modulo operation ensuring index within hash grid bounds

Multi-resolution hash grid with $N_H^L = 16$ levels，每个level最多$N_E$个entries，每个feature vector有$N_G^C = 8$ channels。

**Intuition**：Hash grid的核心优势是O(1) lookup且不需要dense voxel storage。Generative的key在于hash function的input不仅包含position $\mathbf{p}$，还包含scene-level feature $\mathbf{f}_G$，使得hash mapping scene-dependent。这使得同一position在不同scene映射到不同feature，从而实现cross-scene generalization。

这种思路与InstantNGP [Müller et al., 2022]的multi-resolution hash grid类似，但关键区别在于hash function的input被augmented by scene-level feature。

参考：
- InstantNGP: https://nvlabs.github.io/instant-ngp/
- SceneDreamer: https://arxiv.org/abs/2302.01330

#### 3.3.3 Volumetric Rendering

标准volume rendering公式：

$$C(\mathbf{r}) = \int_0^\infty A(t) \mathbf{c}(\mathbf{f}_G^{\mathbf{r}(t)}, l(\mathbf{r}(t))) \boldsymbol{\sigma}(\mathbf{f}_G^{\mathbf{r}(t)}) dt$$

其中：
- $\mathbf{r}(t) = \mathbf{o} + t\mathbf{v}$: camera ray，origin $\mathbf{o}$，direction $\mathbf{v}$
- $A(t) = \exp\left(-\int_0^t \sigma(\mathbf{f}_G^{\mathbf{r}(s)}) ds\right)$: accumulated transmittance
- $\mathbf{c}$: color function
- $\boldsymbol{\sigma}$: volume density function
- $l(\mathbf{p})$: semantic label at position $\mathbf{p}$

#### 3.3.4 Loss Function

$$\ell_G = \lambda_G^{L1} \|\hat{\mathbf{I}}_G - \mathbf{I}_G\| + \lambda_G^P \mathcal{P}(\hat{\mathbf{I}}_G, \mathbf{I}_G) + \lambda_G^G \mathcal{G}(\hat{\mathbf{I}}_G, \mathbf{S}_G)$$

- $\lambda_G^{L1} = 10$, $\lambda_G^P = 10$, $\lambda_G^G = 0.5$
- $\mathcal{P}$: perceptual loss [Johnson et al., ECCV 2016]
- $\mathcal{G}$: GAN loss [Lim & Ye, 2017]
- $\mathbf{S}_G$: perspective-view semantic map from accumulating labels along rays
- $\ell_G$只应用于background stuff pixels

---

### 3.4 Building Instance Generator (BIG)

#### 3.4.1 Local Scene Extraction

对building instance $B_i$，提取以$(c_x^{B_i}, c_y^{B_i})$为中心的local window $\mathbf{L}^{B_i}$，dimensions $N_B^H \times N_B^W \times N_B^D$。

GoogleEarth: $N_B^H = N_B^W = 672$, $N_B^D = 640$
CityTopia: $N_B^H = N_B^W = 768$, $N_B^D = 2560$

**Building instantiation**: 由于所有buildings在$\mathbf{S}_L$中share同一semantic label，需要通过connected components detection分离出每个building。

**Facade/Roof分离**：真实世界building facades与roofs有显著不同的visual distribution。所以对每个building $B_i$，给facade与roof分配不同semantic labels，roof assign为top-most voxel layer。其他buildings在$\mathbf{L}^{B_i}$中被assign为0。

#### 3.4.2 Scene Parameterization

与City Background Generator的hash grid不同，BIG用**local encoder + SinCos positional encoding**。

Step 1: 用encoder $E_B$将local scene编码到pixel-level features $\mathbf{f}_{B_i}$，resolution $N_B^H \times N_B^W \times N_B^C$，$N_B^C = 63$：

$$\mathbf{f}_{B_i} = E_B(\mathbf{H}_L^{B_i}, \mathbf{S}_L^{B_i})$$

Step 2: 对3D position $\mathbf{p} = (p_x, p_y, p_z)$，feature $\mathbf{f}_{B_i}^\mathbf{p}$：

$$\mathbf{f}_{B_i}^\mathbf{p} = \mathcal{O}(\text{Concat}(\mathbf{f}_{B_i}(p_x, p_y), p_z))$$

- $\mathbf{f}_{B_i}(p_x, p_y) \in \mathbb{R}^{N_B^C}$: feature at BEV coordinates
- $\mathcal{O}$: NeRF-style positional encoding

$$\mathcal{O}(\mathbf{x}) = \{\sin(2^i \pi \mathbf{x}), \cos(2^i \pi \mathbf{x})\}_{i=0}^{N_P^L - 1}$$

- $N_P^L = 10$: positional encoding levels
- $\mathcal{O}$分别applied to each element of $\mathbf{x}$，values normalized to $[-1, 1]$

**Intuition**：为什么buildings用SinCos而background用hash grid？因为buildings的facades有strong regular periodic patterns（windows、floors），SinCos positional encoding天然适合representing periodic functions。Hash grid的randomness反而会破坏这种regularity。

#### 3.4.3 Volumetric Rendering with Style Code

$$C(\mathbf{r}) = \int_0^\infty A(t) \mathbf{c}(\mathbf{f}_{B_i}^{\mathbf{r}(t)}, \mathbf{z}, l(\mathbf{r}(t))) \boldsymbol{\sigma}(\mathbf{f}_{B_i}^{\mathbf{r}(t)}) dt$$

- $\mathbf{z}$: style code捕获building appearance variability
- $\mathbf{r}(t) = \mathbf{o} + t\mathbf{v} - [c_x^{B_i}, c_y^{B_i}, 0]^T$: 在building-centric coordinate system中的ray

#### 3.4.4 Loss Function

BIG只使用GAN loss：

$$\ell_B = \mathcal{G}(\hat{\mathbf{I}}_{B_i}, \mathbf{S}_{B_i})$$

**Intuition**：为什么BIG只用GAN loss而不用reconstruction loss？因为每个building的appearance是diverse的，没有固定的ground truth。GAN loss提供distribution-level supervision，让generator学习building appearance的distribution。

---

### 3.5 Vehicle Instance Generator (VIG) — 关键Novelty

#### 3.5.1 Canonical Feature Space

VIG的核心创新是canonical feature space。给3D position $\mathbf{p} = (p_x, p_y, p_z)$，canonicalized point $\mathbf{p}^C$：

$$\mathbf{p}^C = \mathbf{R} \left(\mathbf{p} - [c_x^{V_i}, c_y^{V_i}, c_z^{V_i}]^T\right)$$

变量解析：
- $(c_x^{V_i}, c_y^{V_i}, c_z^{V_i})$: vehicle $V_i$的中心坐标
- $\mathbf{R}$: rotation matrix，将3D point normalize到canonical feature space

Rotation matrix $\mathbf{R}$:

$$\mathbf{R} = \begin{bmatrix} \cos\theta & \sin\theta & 0 \\ -\sin\theta \cos\gamma & \cos\theta \cos\gamma & \sin\gamma \\ \sin\theta \sin\gamma & -\cos\theta \sin\gamma & \cos\gamma \end{bmatrix}$$

- $\theta \in (-180°, 180°]$: yaw angle，vehicle在XY-plane相对-y-axis的heading
- $\gamma \in (-90°, 90°)$: pitch angle，正值表示upward tilt，负值downward tilt

**Intuition**：为什么vehicles需要canonicalization而buildings不需要？因为vehicles是compact shapes且structural regularity强（front、rear、body有distinct appearances，但同一type的vehicles共享这些structural features）。Canonicalization让network能在不同vehicles之间share features，从而更好学习vehicle的common structure。Buildings则每个都unique，canonicalization反而无意义。

#### 3.5.2 Feature Extraction

$$\mathbf{f}_{V_i}^t = E_V(\mathbf{H}_{T_t}^{V_i}, \mathbf{S}_{T_t}^{V_i})$$

其中$\mathbf{f}_{V_i}^t \in \mathbb{R}^{d_V}$，$d_V = 2$。

然后对canonicalized point $\mathbf{p}^C$在time step $t$的feature：

$$\mathbf{f}_{V_i}^{(\mathbf{p}^C, t)} = \mathcal{O}(\text{Concat}(\mathbf{f}_{V_i}^t, \mathbf{p}^C))$$

注意：这里$\mathcal{O}$的input包含**time $t$ implicitly通过$\mathbf{f}_{V_i}^t$**，使得vehicle appearance随traffic scenario变化（不同time step可能不同vehicle instance）。

#### 3.5.3 Volumetric Rendering

VIG的rendering与BIG相同，使用style code $\mathbf{z}$。Camera ray $\mathbf{r}(t)$按Equation 13 normalize到canonical feature space。

#### 3.5.4 Loss Function

VIG用hybrid loss（与City Background Generator相同）：

$$\ell_V = \lambda_V^{L1} \|\hat{\mathbf{I}}_{V_i}^t - \mathbf{I}_{V_i}^t\| + \lambda_V^P \mathcal{P}(\hat{\mathbf{I}}_{V_i}^t, \mathbf{I}_{V_i}^t) + \lambda_V^G \mathcal{G}(\hat{\mathbf{I}}_{V_i}^t, \mathbf{S}_{V_i}^t)$$

- $\lambda_V^{L1} = 10$, $\lambda_V^P = 10$, $\lambda_V^G = 0.5$

**Intuition**：为什么VIG用reconstruction loss而BIG不用？因为vehicles的appearance variation远小于buildings，可以有更稳定的supervision。而buildings每个instance都独特，没有稳定的reconstruction target。

#### 3.5.5 Local Window Resolution

VIG的local window极小：$N_V^H = 32$, $N_V^W = 32$, $N_V^D = 32$。这反映了vehicle的compact shape characteristic，与buildings的large $N_B$形成对比。

---

### 3.6 Compositor

由于没有ground truth的composited image（只有单独的background、building、vehicle ground truth），Compositor用heuristic mask-based composition：

$$\mathbf{I}_C^t = \hat{\mathbf{I}}_G \mathbf{M}_G + \sum_{i=1}^{n_B} \hat{\mathbf{I}}_{B_i} \mathbf{M}_{B_i} + \sum_{i=1}^{n_V} \hat{\mathbf{I}}_{V_i}^t \mathbf{M}_{V_i}^t$$

**Intuition**：这种compositor设计implicit假设每个generator的mask准确。当masks有errors时，会有visible boundary artifacts。这是compositional方法的核心trade-off：灵活性vs. seamless integration。

---

## 4. 数据集深入解析

### 4.1 OSM Dataset

来源：OpenStreetMap [OpenStreetMap, https://openstreetmap.org]

Coverage: 80个全球城市，超过6000 km²

Rasterization细节：
- 坐标系：EPSG:3857（Web Mercator projection）
- Zoom level: 18
- Resolution: 约0.597 meters/pixel

Semantic map color scheme:
- Red: roads
- Yellow: buildings
- Green: urban greenery
- Cyan: construction areas
- Blue: water bodies

Height field assignment:
- Buildings: from OpenStreetMap data
- Roads: 4 meters
- Water bodies: 0
- Urban greenery: Perlin noise [Perlin, SIGGRAPH 1985]，range 8-16 meters

### 4.2 GoogleEarth Dataset

来源：Google Earth Studio [https://earth.google.com/studio]

Coverage: New York City，400个orbit trajectories，24,000 images @ 960×540

Orbit参数：
- Radius: 125-813 meters
- Altitude: 112-884 meters

**自动annotation pipeline**：
1. 对OSM semantic map做connected components detection → building instance map
2. 用OSM height values extrude pixels → 3D volumes
3. 用Google Earth Studio的camera parameters将3D volumes project到images → 2D annotations

这个pipeline的优势是can be applied to worldwide cities，但有以下limitations：
- 缺street-view images（Google Earth Studio在ground level的3D reconstruction差）
- OSM annotations有些imprecision
- 缺highway的height data（无法annotation高架结构）

### 4.3 CityTopia Dataset

为解决GoogleEarth的limitations而构建。

#### 4.3.1 Virtual City Generation Pipeline

1. **City prototype generation** in Houdini (https://www.sidefx.com/products/houdini)
2. 使用CitySample project [https://www.unrealengine.com/marketplace/en-US/product/city-sample]的~5000个3D assets
3. **Surface sampling**：每个3D point assign semantic + instance label
4. **Unreal Engine instantiation**：生成完整virtual city

#### 4.3.2 Coverage与Image Collection

- 11个virtual cities
- 3,000 images for cities with buildings
- 7,500 images for vehicle-only city
- 总共37,500 images

**Rendering细节**：
- 每张image spatially 8x supersampled
- Temporally 32x supersampled
- 目的：avoid Moire effects
- Daytime与nighttime scenes都render，nighttime时sunlight被remove以简化lighting consistency learning

#### 4.3.3 Annotations

3D annotations natively generated from virtual city pipeline，2D annotations通过projecting 3D annotations用camera poses生成。

**Key advantage over GoogleEarth**：perfect alignment between 2D/3D annotations与images，包括street-view与aerial-view，包含highway的height data。

---

## 5. Experiments深入分析

### 5.1 Evaluation Metrics详解

- **FID** (Frechet Inception Distance): 15,000 generated frames vs 15,000 real images
- **KID** (Kernel Inception Distance): 同上
- **VBench**: 150 videos × 100 frames @ 16 FPS，评估background consistency、motion smoothness、dynamic degree、aesthetic quality、imaging quality
- **Depth Error (DE)**: L2 distance between normalized depth maps（pseudo ground truth from pretrained depth model [Ranftl et al., TPAMI 2022]）
- **Camera Error (CE)**: scale-invariant normalized L2 distance between generated与COLMAP-reconstructed camera poses，on 600-frame orbital videos

### 5.2 Main Quantitative Results

Table 2的关键数字：

| Method | GoogleEarth FID | CityTopia FID | CityTopia KID | VBench (CityTopia) | DE | CE |
|--------|-----------------|----------------|---------------|--------------------|----|-----|
| SGAM | 277.6 | 330.1 | 0.284 | 0.690 | 0.571 | 233.5 |
| PersistentNature | 123.8 | 235.3 | 0.215 | 0.713 | 0.428 | 127.3 |
| SceneDreamer | 232.2 | 195.1 | 0.126 | 0.708 | 0.185 | 0.162 |
| DreamScene4D | - | 288.2 | 0.136 | 0.715 | 0.199 | 0.146 |
| DimensionX | 206.9 | 171.4 | 0.070 | 0.815 | - | - |
| **CityDreamer4D** | **96.83** | **88.48** | **0.049** | **0.825** | **0.150** | **0.063** |

**Analysis**：
- CityDreamer4D在FID上比best baseline (PersistentNature on GoogleEarth) 低22%，比DimensionX on CityTopia低48%
- KID上improvement更显著：CityTopia上比DimensionX低30%
- VBench上CityDreamer4D最好，体现4D generation quality
- DE与CE最低，证明3D geometry与multi-view consistency的优秀

### 5.3 Ablation Studies详解

#### 5.3.1 ULG Ablation (Table 3)

| Method | FID | KID |
|--------|-----|-----|
| IPSM [Chen et al., 2008] | 321.47 | 0.502 |
| InfinityGAN [Lin et al., ICLR 2022] | 183.14 | 0.288 |
| ULG (Ours) | **124.45** | **0.123** |

ULG比InfinityGAN的FID低32%，证明VQVAE + MaskGIT的autoregressive generation比InfinityGAN的progressive GAN更适合city layout generation。

#### 5.3.2 BIG Ablation (Table 4)

| BIG | Inst. | G | L | Hash | SinCos | FID | KID | DE | CE |
|-----|-------|---|---|------|--------|-----|-----|-----|-----|
| X | X | X | X | - | - | 195.1 | 0.126 | 0.185 | 0.162 |
| ✓ | X | - | - | - | - | 167.8 | 0.094 | 0.157 | 0.087 |
| ✓ | ✓ | X | ✓ | X | ✓ | 196.8 | 0.124 | 0.165 | 0.159 |
| ✓ | ✓ | X | ✓ | ✓ | X | 197.9 | 0.132 | 0.162 | 0.152 |
| ✓ | ✓ | ✓ | X | X | ✓ | 182.3 | 0.111 | 0.155 | 0.092 |
| ✓ | ✓ | X | ✓ | X | ✓ | **88.48** | **0.049** | **0.150** | **0.063** |

**关键发现**：
1. Removing BIG（first row）→ 严重质量下降（FID从88.48到195.1）
2. Removing instance labels（second row）→ FID从88.48到167.8，证明instance-level disentanglement的重要性
3. Hash grid + SinCos（third row）→ 196.8 FID，证明hash grid破坏building的periodic patterns
4. Global encoder + SinCos（fifth row）→ 182.3 FID，证明global encoder导致repetitive facade patterns
5. Local encoder + SinCos（last row，full BIG）→ 88.48 FID，最优配置

#### 5.3.3 VIG Ablation (Table 5)

在vehicle-only city from CityTopia上的结果：

| VIG | Can. | G | L | Hash | SinCos | FID | KID | DE | CE |
|-----|------|---|---|------|--------|-----|-----|-----|-----|
| X | X | X | X | - | - | 419.3 | 0.576 | 0.364 | 1.276 |
| ✓ | X | ✓ | X | X | ✓ | 273.4 | 0.530 | 0.289 | 0.966 |
| ✓ | ✓ | ✓ | X | ✓ | X | 229.2 | 0.428 | 0.259 | 0.989 |
| ✓ | ✓ | ✓ | X | X | ✓ | **142.3** | **0.276** | **0.202** | **0.824** |
| ✓ | ✓ | X | ✓ | ✓ | X | 273.4 | 0.521 | 0.265 | 0.997 |
| ✓ | ✓ | X | ✓ | X | ✓ | 200.5 | 0.403 | 0.332 | 1.117 |

**关键发现**：
1. Removing VIG（first row）→ 419.3 FID，vehicles作为background stuff处理完全失败
2. Removing canonicalization（second row）→ 273.4 FID，证明canonicalization对vehicle structural regularity的必要性
3. Global encoder + SinCos（fourth row，full VIG）→ 142.3 FID，最优
4. Local encoder + SinCos（last row）→ 200.5 FID，证明vehicles需要global encoder来share features across instances

**与BIG对比的insight**：BIG用local encoder，VIG用global encoder，这反映了buildings与vehicles的本质区别——每个building独特，所以local；vehicles share common structure，所以global。

### 5.4 VLN Evaluation (Table 6) — 重要Application

VLN protocol：
- 100 instruction-trajectory pairs
- 12 discrete actions: forward/diagonal 2/4/6m, turn 45°, stop
- Vision-language model作为agent

| Method | #Param (B) | PL | SR | SPL | RT |
|--------|-----------|-----|-----|-----|-----|
| Human (5 participants) | - | 20.73 | 100.0 | 85.87 | 0.00 |
| Gemini 2.5 Pro | - | 9.32 | 12.40 | 4.43 | 0.45 |
| GPT-4o | - | 8.97 | 36.00 | 17.32 | 0.11 |
| SAIL-VL 1.6 | 8.33 | 14.56 | 23.40 | 7.63 | 0.28 |
| Ovis2 | 8.94 | 13.96 | 17.00 | 5.01 | 0.35 |
| Qwen2.5-VL | 8.29 | 5.01 | 15.00 | 7.01 | 0.37 |
| Ola | 8.88 | 9.15 | 18.00 | 8.30 | 0.32 |
| InternVL3 | 7.94 | 9.02 | 25.60 | 12.66 | 0.23 |

**Key insight**：所有VLMs的SR都低于36%，远低于human的100%，说明spatial reasoning in 4D cities对VLMs仍是开放问题。GPT-4o在SR上最强，InternVL3在open-source中最强。这表明generated 4D cities可以作为VLM spatial reasoning的benchmark，类似GRUtopia [Wang et al., 2024]的setting。

参考：
- GRUtopia: https://arxiv.org/abs/2407.10943
- Embodied Web Agents: https://arxiv.org/abs/2506.15677

---

## 6. Critical Analysis 与 Potential Issues

### 6.1 Strengths

1. **Compositional design**的inductive bias正确：不同object class有essentially不同的statistics，强制用同一neural field处理是mis-specified
2. **BEV representation**在city scale上memory-efficient，且与OSM data的rasterization天然契合
3. **Canonical feature space**对vehicles是elegant design，将structural regularity建模为input的invariance
4. **Compositional datasets**：OSM提供layout，GoogleEarth/CityTopia提供visual supervision，互补而非冗余

### 6.2 Limitations

1. **Sequential generation**：buildings与vehicles individually generated，computational cost高。Inference时需要per-instance forward pass，parallelism受限
2. **No global illumination**：Decoupled generation意味着no inter-reflection、no light emission from one object affecting another。Night scenes明显受限（Fig. 15）
3. **Compositor的heuristic nature**：Mask-based composition缺乏learned blending，boundary artifacts可能visible
4. **Pedestrians的integration是post-hoc**：Fig. 14显示用MoMask [Guo et al., CVPR 2024]合成motion后retarget到avatar再render，并未真正integrate到CityDreamer4D的generation framework

### 6.3 Open Questions与Future Directions

1. **Gaussian Splatting替代NeRF**：作者后续工作 [Xie et al., CVPR 2025]已经探索了generative Gaussian splatting for unbounded 3D city generation，但4D版本仍是open problem
2. **Dynamic object interactions**：当前traffic generator只生成bounding boxes，没有object-level interaction modeling（如vehicles避让、pedestrians crossing）
3. **Physics-based simulation**：当前是pure visual generation，没有physics（如gravity、collision）
4. **Lightfield relighting**：Fig. 13的relighting是Lambertian + shadow mapping的simple approximation，no view-dependent effects、no transparency、no subsurface scattering

参考：
- CityGaussian (后续): https://arxiv.org/abs/2403.01548 (CVPR 2025)
- 3D scene generation survey: https://arxiv.org/abs/2505.05474

---

## 7. 与Related Work的Positioning

### 7.1 3D-aware GANs Evolution

CityDreamer4D的neural field design可以追溯：
- **GRAF** [Schwarz et al., NeurIPS 2020]: first NeRF-based GAN
- **pi-GAN** [Chan et al., CVPR 2021]: periodic positional encoding → CityDreamer4D BIG借鉴
- **EG3D** [Chan et al., CVPR 2022]: tri-plane representation，但CityDreamer4D用BEV
- **StyleSDF** [Or-El et al., CVPR 2022]: high-res 3D-consistent generation

### 7.2 3D Scene Generation

CityDreamer4D的直接predecessors：
- **GANCraft** [Hao et al., ICCV 2021]: voxel-based neural rendering with semantic conditioning
- **SceneDreamer** [Chen et al., TPAMI 2023]: BEV representation + generative hash grid，CityDreamer4D的City Background Generator直接继承
- **InfiniCity** [Lin et al., ICCV 2023]: BEV-based但single neural field for all instances，正是CityDreamer4D要overcome的limitation
- **BlockFusion** [Wu et al., ACM TOG 2024]: latent tri-plane extrapolation for expandable 3D scene generation

### 7.3 4D Scene Generation

CityDreamer4D与existing 4D methods的根本区别：
- **4D-fy** [Bahmani et al., CVPR 2024]: text-to-4D via score distillation，object-scale
- **Comp4D** [Xu et al., 2024]: LLM-guided compositional 4D，但仍是object-level
- **DreamScene4D** [Chu et al., NeurIPS 2024]: monocular video to 4D multi-object scene，但cannot generate from scratch
- **DimensionX** [Sun et al., ICCV 2025]: controllable video diffusion for 3D/4D，但multi-view consistency差

CityDreamer4D是first to handle **unbounded** scale的4D generation。

### 7.4 4D Occupancy Generation

最近的DynamicCity [Bian et al., ICLR 2025]与DOME [Gu et al., 2024]用4D LiDAR occupancy作为representation，但缺乏rendering quality与unbounded scale。

参考：
- 4D-fy: https://arxiv.org/abs/2311.12684
- DynamicCity: https://arxiv.org/abs/2501.17691
- DreamScene4D: https://arxiv.org/abs/2410.01553

---

## 8. Implementation细节

### 8.1 Training Schedule

| Module | Iterations | Batch Size | Optimizer | Learning Rate |
|--------|-----------|------------|-----------|---------------|
| VQVAE (ULG) | 1,250,000 | 16 | Adam (β=0.5, 0.9) | 7.2e-5 |
| AR Transformer (ULG) | 250,000 | 80 | Adam (β=0.9, 0.999) | 2e-4 |
| Stuff/Instance Generators | 298,500 | 8 | Adam (β=0, 0.999) | 1e-4 (gen), 1e-5 (disc) |

注意generators用β=(0, 0.999)而discriminators用β=(0, 0.999) + lower lr，这是StyleGAN2的训练设置。

### 8.2 Image Crop

Training时images randomly cropped to 192×192 resolution，inference时rendered at 960×540。这种resolution gap是常见practice，但需要注意可能的distribution shift。

### 8.3 Multi-resolution Hash Grid Configuration

- Levels: $N_H^L = 16$
- Entries per level: $N_E = 2^{19}$ (约500K)
- Channels per feature: $N_G^C = 8$
- Total parameters: $16 \times 2^{19} \times 8 \approx 67M$ per hash grid

参考：
- StyleGAN2 training: https://arxiv.org/abs/1912.04958

---

## 9. Connections to Broader Research Themes

### 9.1 Generative Models of Neural Fields

CityDreamer4D延续了一个重要trend：将neural field作为generative model的latent space。这条lineage包括：
- NeRF [Mildenhall et al., ECCV 2020] → GRAF → GIRAFFE → StyleNeRF → EG3D → SceneDreamer → CityDreamer → CityDreamer4D
- InstantNGP [Müller et al., 2022]的hash grid是key enabling technology

### 9.2 Compositional Generation

Compositional design在3D/4D generation中越来越重要：
- **GIRAFFE** [Niemeyer & Geiger, CVPR 2021]: compositional generative neural feature fields
- **BlockFusion**: expandable 3D scene via tri-plane extrapolation
- **CityDreamer4D**: stuff-oriented + instance-oriented neural fields

这种trend反映了一个deep insight：monolithic generative models难以capture complex scenes的hierarchical structure。

### 9.3 Sim2Real与Synthetic Data

CityTopia dataset代表了synthetic data for training的趋势：
- GTA-V [Richter et al., ECCV 2016]
- SYNTHIA [Ros et al., CVPR 2016]
- MatrixCity [Li et al., ICCV 2023]
- CityTopia: 更高fidelity、更多annotations、unbounded scale

这反映了Unreal Engine等game engines作为ML data sources的growing importance。

### 9.4 Embodied AI Benchmarks

VLN evaluation与GRUtopia的connection非常重要：
- GRUtopia [Wang et al., 2024]: large-scale city benchmark for robot learning
- Embodied Web Agents [Hong et al., 2025]: bridging physical-digital realms
- CityDreamer4D: can serve as procedural city generator for embodied AI benchmarks

### 9.5 Procedural Content Generation (PCG)

CityDreamer4D与traditional PCG的关系：
- CityCraft [Deng et al., 2024]: LLM-driven 3D city generation
- CityX [Zhang et al., 2024]: controllable PCG for unbounded 3D cities
- SceneX [Zhou et al., AAAI 2025]: LLM + procedural controllable large-scale scene generation

CityDreamer4D相比这些LLM-based PCG方法的优势是end-to-end learning，劣势是lack of explicit controllability。

---

## 10. Final Thoughts与Karpathy-style Intuition

读完这篇paper，我build的intuition是：

**Decoupling before Composing**：Complex generative problems的key不是end-to-end learning，而是正确地decompose problem使得每个sub-problem有合适的inductive bias。CityDreamer4D的decoupling有4个层次：
1. Static vs. Dynamic
2. Stuff vs. Instance
3. Building vs. Vehicle
4. Layout vs. Appearance

每层decoupling都基于object class的**statistical properties**：appearance diversity、structural regularity、motion characteristics。

**Representation should match statistics**：Hash grid适合irregular textures（background），SinCos positional encoding适合periodic patterns（building facades），canonical space适合shared structure（vehicles）。这种representation-object的matching是deep learning中under-appreciated的设计原则。

**Datasets作为论文的一等公民**：CityDreamer4D投入大量精力构建3个complementary datasets，每个解决specific limitations。这种data-centric thinking在foundation model era尤其重要——data quality决定model capability的upper bound。

**Limitations作为Future Work的roadmap**：sequential generation cost、global illumination、learned compositor、true multi-agent dynamics都是值得follow-up的方向。作者诚实承认这些limitations反而让人信任整个approach。

**Compositional generation是Embodied AI的key infrastructure**：CityDreamer4D生成的4D cities可以作为autonomous driving、robot navigation、VLM spatial reasoning的procedural benchmark generator，这是比单纯visual quality更重要的impact。

参考：
- CityDreamer4D arXiv: https://arxiv.org/abs/2403.01548 (注意这是CityDreamer)
- S-Lab (Ziwei Liu's group): https://liuziwei7.github.io/
- NeurIPS 2024 schedule: https://neurips.cc/

---

希望这个deep dive能帮你build CityDreamer4D的intuition。这篇paper的核心take-away是：**compositional inductive biases matching object statistics > monolithic end-to-end learning**，这个insight应该可以transfer到其他large-scale generative problems。如果你想进一步讨论具体component的implementation细节或与specific baseline的comparison，告诉我。
