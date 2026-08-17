---
source_pdf: ETHcavation.pdf
paper_sha256: 50df604c687029eccfec1cb54455609dc41a847e3db5f6e9ccc1ff3151622c1b
processed_at: '2026-08-04T05:14:50-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 ETHcavation

好，前面那个太学术了，我用大白话再过一遍，顺便补一些直觉。

---

## 一句话说清楚这篇 paper 干了啥

ETH 的实验室给一台 Menzi Muck M545 挖掘机装了一套"眼睛 + 脑子"，让它能在工地上自己认路、绕开人、躲障碍。眼睛是 camera + LiDAR，脑子是一套 modular pipeline：先 segment 图像，再把语义抬到 3D，再追踪动态物体，再用 RRT* 规划路径。他们顺手还发了 502 张手工标注的工地图像数据集，代码全开源。

就这么个事。但里面有一堆工程细节值得拆开看。

参考 excavator 本体: https://www.sciencedirect.com/science/article/pii/S0926580521001827

---

## 为什么工地这么难

autonomous driving 那套 perception stack（nuScenes [1], Waymo [22] 训出来的 BEV 网络）搬到工地上直接废掉，原因有三：

**第一，类别完全不对**。城市道路关心的是 car, pedestrian, traffic light, lane marking。工地关心的是 bucket, gripper, gravel-pile, container, self-arm（挖掘机自己的大臂）。这两个 label space 几乎没交集。你拿 COCO 训出来的 model 去认挖掘机铲斗，它压根没见过这东西。

**第二，数据稀缺**。标一张工地 panoptic mask 很贵，因为类别专业、场景杂乱、遮挡严重。这篇 paper 也只标了 502 张，对比 Cityscapes 的 5000 张、nuScenes 的 40k keyframes，量级差两个数量级。

**第三，terrain 本身就是决策依据**。城市道路的路面基本都 traversable，障碍物是 discrete object。工地不一样——同一片地面，mud 不能走（会陷），gravel 能走，slope 太陡不能走。terrain 类别直接决定 cost，这和 driving 完全是两套逻辑。

所以这篇 paper 的核心 contribution 就是：**在小数据 + 新 domain + 动态场景下，怎么搭一套能用的 perception-to-planning pipeline**。

---

## 整套 pipeline 长啥样

Figure 2 那张图拆成三块：

### Block 1: 2D Panoptic Segmentation
- 输入：camera image（4K Ximea，竖直装在挖掘机顶上）
- 模型：Mask2Former [4] + Swin-Tiny backbone
- 输出：每 pixel 一个 class label + instance ID

### Block 2: LiDAR Lifting + Tracking + Mapping
- 输入：2D mask + raw LiDAR scan + state estimate（来自 Graph-MSF [24]）
- 输出：分好 static/dynamic 的 labeled point cloud + 2D semantic grid map（双层）

### Block 3: Motion Planning
- 输入：occupancy map + cost map
- 模型：online RRT*（OMPL 实现）
- 输出：1Hz 重规划的 trajectory

这是一个 **explicit modular pipeline**，每个 block 输入输出明确，可替换、可调试。对比 BEVFusion [19] 那种 end-to-end fusion 网络，这种设计在 small data + safety-critical 场景下更靠谱——你能解释每一步为什么这么决策，坏了能定位到哪个 module。

---

## Panoptic Segmentation 这块的水很深

### 为什么选 Mask2Former 而不是 DETR

paper 做了对比（Table 2）：

| Model | PQ | 速度 |
|-------|-----|------|
| DETR (ResNet-50) | 0.41 | 12 img/s |
| M2F Swin-Tiny | 0.68 | 8 img/s |
| M2F Swin-Big | 0.63 | 3 img/s |
| M2F Swin-Large | 0.68 | 1 img/s |

DETR [2] 是 2020 年的架构，encoder 还在用 ResNet-50（CNN），只有 decoder 是 transformer。Mask2Former [4] 是 2021 年的，encoder 和 decoder 都是 transformer（Swin + masked attention），架构上更现代。

**直觉**：纯 transformer 架构在 small data + domain shift 下，pretrain 的 prior 更强、迁移性更好。CNN 的 inductive bias（locality, translation invariance）在城市图像上是优势，但在工地这种 first-person + 奇特视角下反而限制表达。

Mask2Former 的核心机制：query 用 masked attention 限制在 predicted region 内部，这样 background 噪声不会干扰 instance-level feature 抽取。对工地这种 background 杂乱（一堆 gravel, mud, debris）的场景特别有用。

参考: https://arxiv.org/abs/2112.01577

### 反直觉：Swin-Big 比 Swin-Tiny 还差

这个很有意思。Swin-Tiny 29M 参数 PQ 0.68，Swin-Big 88M 参数 PQ 反而掉到 0.63。Swin-Large 200M 又追回 0.68。

**直觉解释**：502 images + 1290 COCO subset 这种数据量，对 88M 参数的 model 来说不够喂，出现 overfitting。Swin-Large 反而 OK，可能是因为参数大到某个 threshold 后 pretrain prior 占主导，抗 overfit 能力反而强。这和 LLM 在 small data 上的 in-context learning 现象有点像——model 越大越依赖 prior 而非 fit training data。

所以工程上：**small data regime 下，model size 不是越大越好，需要和 data 量匹配**。Swin-Tiny 在 RTX 3080Ti 上 8 img/s 又是 sweet spot，deploy 优选。

### 训练 DETR 的三步法（Table 1）

DETR 的训练他们试了两种：
- Freeze backbone：PQ 0.35
- Fine-tune backbone：PQ 0.41

**直觉**：ImageNet 1k pretrain 的 ResNet-50 见过的图像和工地 first-person view 差太远。freeze 的话 feature 不够 domain-aligned，RQ（分类准确率）从 0.44 → 0.52 提升主要来自 backbone 学到了 construction-specific feature。

三步训练法：
1. 先 fine-tune box detection head（建立 object-level representation）
2. freeze backbone，单独训 segmentation head
3. 整体 end-to-end fine-tune 到 200 epochs

这是一种 curriculum——先稳住 detection 再加 segmentation，避免 joint training 早期梯度噪声搞坏 backbone pretrain。**小数据下 joint training 容易崩，分阶段 warm-up 是实用经验**。

### Catastrophic forgetting 的实战处理（Table 3）

这个 ablation 很关键：

| 训练集 | 验证集 | PQ |
|--------|--------|-----|
| Custom only | COCO val | 0.06 |
| Custom only | Custom val | 0.41 |
| Custom + COCO | COCO val | 0.26 |
| Custom + COCO | Custom val | 0.41 |

**只用 Custom 训练，COCO 上 PQ 掉到 0.06——基本全忘了**。混入 COCO subset 后，COCO 上 PQ 恢复到 0.26，Custom 上 PQ 保持 0.41 不掉。

最佳 ratio 实证是 **COCO : Custom ≈ 3 : 1**。

**直觉**：这就是 continual learning 里的 experience replay 思想。fine-tune 时如果不混入原 dataset 的样本，新 gradient 会覆盖掉 pretrain 学到的 general feature。混一点 COCO 进去相当于"边学新东西边复习旧东西"，两边都不掉。

这个 finding 对所有做 domain adaptation 的人都有用：**fine-tune 时记得留一部分原 dataset 在 batch 里**。

参考 replay-based continual learning: https://arxiv.org/abs/1909.08370

### Dataset size scaling（Table 4）

| Fraction | 20% | 40% | 60% | 80% | 100% |
|----------|-----|-----|-----|-----|------|
| PQ | 0.53 | 0.57 | 0.59 | 0.62 | 0.68 |
| RQ | 0.63 | 0.67 | 0.70 | 0.72 | 0.81 |
| SQ | 0.77 | 0.77 | 0.78 | 0.79 | 0.78 |

**关键观察**：
- PQ 和 RQ 还在**线性涨，没 plateau**
- SQ 几乎不动

**直觉**：SQ（segmentation quality，mask 形状的 IoU）主要靠 backbone 学到的 spatial feature，这部分从 ImageNet/COCO pretrain 迁移过来就够用了，所以加数据没用。RQ（recognition quality，分类准）依赖 head 学 class-specific 决策 boundary，domain shift 大，需要更多 in-domain data 学。

**结论**：数据加到 100% 还在涨，说明 502 张远没到饱和。如果还能标，继续标 PQ 还会涨。这是 scaling law 在 supervised segmentation 上的实证。

---

## LiDAR Lifting 和 Tracking 这块是工程核心

### 投影 + Clustering + Voting 的三连击

1. **投影**：每个 LiDAR point $p_i \in \mathbb{R}^3$ 通过外参 $T_{CL} \in SE(3)$ 和内参 $K$ 投到 image plane：
   $$[u_i, v_i, 1]^T = \pi(K \cdot T_{CL} \cdot p_i)$$
   其中 $\pi$ 是 perspective division（除以 z 维）。该 pixel 的 class label 直接赋给这个 point。

   视野外的 point：标 "unknown"（保守处理）

2. **Ground removal + DBSCAN [7]**：
   - 先去掉 ground plane（不然 ground 点会和上面的 object 连成一个大 cluster）
   - DBSCAN 在剩余 point cloud 上做 density-based clustering
   - 两个关键参数：$\epsilon$（neighborhood radius）和 minPts（最小 cluster size）
   - DBSCAN 的好处是不用预设 cluster 数量，对工地这种 unknown object 数量的场景天然合适
   - 参考: https://www.aaai.org/ojs/index.php/AAAI/article/view/5073

3. **Majority voting**：每个 cluster 内部统计 label 分布，取 majority。如果 majority 比例低于阈值 $\tau$，整个 cluster 标 "unknown"。
   - **直觉**：这是 soft → hard decision 的鲁棒化。单个 point 被 misclassify 不会污染整个 cluster，要"多数同意"才算数。

### Kalman Filter Tracking 的关键 trick

State vector：$\mathbf{x} = [x, y, z, v_x, v_y, v_z]^T$（位置 + 速度）

Constant velocity 假设，transition matrix：
$$F = \begin{bmatrix} I_3 & \Delta t \cdot I_3 \\ 0 & I_3 \end{bmatrix}$$

$\Delta t$ 是时间间隔，$I_3$ 是 3×3 单位阵。这个 matrix 说"位置 += 速度 × 时间，速度不变"。

**最关键的设计**：当 object 离开 camera FoV（camera 看不到了，但 LiDAR 还能扫到几何形状），**保留之前的 semantic class label**，只用 LiDAR geometry 维持 tracking。

**直觉**：camera 给你语义（"这是 person"），LiDAR 给你几何（"这个 cluster 在哪里、怎么动"）。两者解耦后，即使 camera 暂时看不到，LiDAR 还能追，class label 不会丢。这是 explicit pipeline 比 end-to-end 强的地方——你能刻意设计 sensor fallback 逻辑。

### Static / Dynamic 分类逻辑

- "things" classes（person, machinery, vehicle 等）：进 dynamic cloud
- "stuff" classes（terrain, gravel, wall, sky 等）：进 static cloud
- "unknown"：进 dynamic cloud（保守策略——不知道就当作可能动的处理）

这个 stuff/things 区分来自 panoptic segmentation [16] 的标准定义。stuff 是 amorphous region（天空、草地），things 是 countable object（人、车）。

---

## 双层 Grid Map 是这篇 paper 的小亮点

用 GridMap library [8]（ETH 自己的 ROS 包，https://github.com/ANYbotics/grid_map）：

- **Static layer**：累积 stuff 点，**一旦写入长期保留**，即使之后被遮挡也不删
- **Dynamic layer**：每 frame 重写，只保留当前 things + unknown 点

每 update cycle：static layer 不动，dynamic layer 覆盖在 static 上，merged 输出。

**为什么这个设计重要**：想象一个人走过 gravel 路面。如果只有单层 map，person cell 会覆盖 gravel cell。人走开后，这个 cell 变成"没观测到"——gravel 信息丢了。下次规划路径时这个区域变成 unknown，robot 不敢走。

双层设计下：person 在 dynamic layer 覆盖 gravel cell，但 static layer 的 gravel 一直在。人走开，dynamic layer 这 cell 清空，static layer 的 gravel 自动重新显现。**信息不丢，记忆不乱**。

这个 idea 简单但很关键。很多 SLAM 和 mapping 系统在这里栽过跟头。

---

## Cost Map 和 RRT* 的细节

### Cost 公式（Equation 1）

$$C_{\text{total}} = \lambda_1 C_{\text{length}} + \lambda_2 C_{\text{semantic}}$$

- $C_{\text{length}}$：path 总长度（米）
- $C_{\text{semantic}}$：path 上每个 segment 在 excavator footprint 覆盖区域内的平均 traversability cost 累加
- $\lambda_1 = 1, \lambda_2 = 0.1$

**为什么 $\lambda_2$ 这么小**：semantic cost 是 cell-level 累积，path 越长 cell 越多总 cost 越大。如果不 discount，semantic cost 会主导，short path 反而输给"绕远走好路"的 path。$\lambda_2 = 0.1$ 是说"我还是想要 short path，但 terrain 差到一定程度时愿意绕"。

每个 segment 的 semantic cost：
$$C_{\text{semantic}}(\text{seg}) = \frac{1}{|A|} \sum_{c \in A} \text{cost}(c)$$

$A$ 是 excavator footprint 在该 segment 处覆盖的 cell 集合，$|A|$ 是 cell 数。这是 footprint-averaged 而非 point-cost——因为挖掘机底盘大，单 cell cost 不能代表整体 traversability。

### RRT* 配置

- Library: OMPL（https://ompl.kavrakilab.org/）
- 频率：1Hz，单次 max 0.95s
- Validity check：excavator footprint 下所有 occupancy 值为 0
- Switch logic：新 trajectory cost < 当前 path remaining cost → 切换
- Safety：collision risk 在 3m 内 → cost = $\infty$ → robot 停下

**为什么用 RRT* 不用 A* 或 optimization-based**：
- A* 在 grid 上需要精细 resolution，工地 free space 高度 non-convex（窄通道、料堆间隙）容易卡
- CHOMP/TrajOpt 这种 optimization-based 容易陷 local minima
- RRT* 在 high-dim + non-convex space 的 exploration 能力强，且 1Hz + 0.95s budget 配 reactive replanning 够用

RRT* 原始 paper: https://arxiv.org/abs/1103.4402

---

## 实地测试的 intuition

### Avully 的 adversarial test（Figure 7）

人故意挡 path，系统 1Hz 重规划绕过。这本质是 **modular pipeline 的 reactive 能力**——dynamic layer 实时更新 person 位置，RRT* 看到新 obstacle 重算路径。

注意这不是 learning-based policy 的 generalization，是 explicit tracking + replanning 的工程鲁棒性。两种路线的区别：
- Learning policy：训练时见过类似场景，inference 时模仿
- Explicit pipeline：没见过也行，只要 tracker 追得上、planner 算得出

### Figure 6 的 confidence threshold 问题

Mask2Former 在 dirt / gravel / pavement 边界这种**连续 terrain spectrum** 上经常给不出 dominant class——softmax 输出分散在多个 plausible class 上，max confidence 都低于 threshold，整块区域变 "unknown"。

**这是 closed-vocabulary segmentation 的本质限制**。terrain 是连续的，hard label 是离散的，强行 discretize 就会在边界处崩。

paper Future Work 提到 open-vocabulary 方向。2024 年后 SAM2（https://arxiv.org/abs/2407.09512）和 Grounding DINO（https://arxiv.org/abs/2303.05499）可以做 zero-shot 或 prompt-based segmentation，可能直接绕过 fine-tune。但 real-time 性能得测——SAM2 在 RTX 3080Ti 上能不能跑到 8Hz 是个问号。

---

## 我觉得这篇 paper 最值得拿走的三件事

### 1. Small data domain adaptation 的实战配方

- 大 model pretrain（ImageNet 21k / COCO）
- 小 data fine-tune，**但必须混入原 dataset 的子集**（3:1 ratio 实证最佳）
- Model size 要和 data 量匹配，不是越大越好
- 三步训练法（detection head → segmentation head → joint）适合 DETR 类架构

这套配方对所有做 domain adaptation 的 CV/robotics 工程师都适用。

### 2. Static/Dynamic 双层 map 设计

简单但解决真问题——dynamic obstacle 走过留下"hole"的经典痛点。GridMap library 实现也简单，直接可用。

### 3. Explicit pipeline 在 safety-critical 场景的价值

Construction、mining、agriculture 这种场景，end-to-end model 出了你都不知道为什么。Explicit pipeline 每步可调试、可解释、可审计，在工业落地时这是刚需。当然代价是 error propagation 和 module 间 calibration 敏感。

---

## 扩展思考方向

### 1. Foundation model 时代怎么改进

用 SAM2 + CLIP 做 prompt-based segmentation（text prompt："bucket", "gravel pile"），完全跳过 fine-tune。优势：
- Catastrophic forgetting 自然消失（不 fine-tune）
- 新类别加 prompt 就行，不用重训
- Ambiguous terrain 可以输出 continuous embedding 而非 hard label

劣势：
- Real-time 性能存疑
- 对 construction-specific 细粒度类别（self-arm, gripper）的精度可能不够

### 2. Cost map 从手工 tune 到 learned

$\lambda_1, \lambda_2$ 和 traversability cost 表都是手调的。可以用 IRL（inverse reinforcement learning）从 expert demonstration 学 cost function。ETH 同组的 ViPlanner [28] 已经在 depth + semantic image 上做了 imperative learning 的 local navigation，可以借鉴。

参考: https://arxiv.org/abs/2402.19278

### 3. 3D map 扩展

现在 2D grid map 处理不了 overhead structure（桥下、棚下、栈道）。扩展到 3D occupancy grid（如 Voxblox, https://github.com/ethz-asl/voxblox）+ 3D RRT* 是自然延伸。paper Future Work 也提了。

### 4. Tracking 的 CV model 假设太强

Constant velocity 对工地 worker（突然停、突然变向）和大机械（启动停止）都不准。可以上 IMM（Interacting Multiple Model）Kalman，多个 motion model 并行跑再 mix。或者直接用简单 LSTM 学 motion pattern。

---

## 一句话总结

这篇 paper 在 small data + 新 domain + 动态场景下，搭了一套 explicit modular pipeline，把 2D panoptic segmentation 抬到 3D 语义 map，再喂给 RRT* 做 reactive planning。工程完整度高，ablation 扎实，dataset 和 code 全开源。技术亮点是 stuff/things 双层 map 设计和 catastrophic forgetting 的实战处理。局限性是 closed-vocabulary segmentation 在连续 terrain 上的 brittleness，以及手工 cost tuning。未来方向是 foundation model（SAM2/CLIP）替换 segmentation 模块、IRL 学 cost、3D map 扩展。

代码: https://github.com/ETHZ-RobotX/ETHcavation
数据: https://rsl.ethz.ch/robotics-datasets.html

Karpathy 你看这种 modular pipeline 的设计，在 robot learning 大一统趋势下还有多少生命力？我觉得在 safety-critical + small data 的长尾场景里至少还能活 5-10 年，直到 foundation model 的 real-time 和可解释性问题被解决。你怎么看？

---

# ETHcavation: 深度技术解读

## 一、背景与核心动机

这篇 paper 来自 ETH Zürich 的 Robotic Systems Lab（Marco Hutter 团队），合作者还包括 MPI Stuttgart 的 Julian Nubert。这个实验室之前做过 HEAP autonomous walking excavator [14]，ViPlanner [28] 等工作，是 legged robot 和 construction robot 领域的强组。这篇工作可以看作是他们在 construction autonomy 方向上对 perception 层的一次系统性整合。

核心 motivation 在于：construction site 是 unstructured + dynamic 的场景，传统的 autonomous driving perception stack（比如基于 nuScenes [1] / Waymo [22] 训练的 BEV 检测网络）无法直接迁移过去，因为：
- 类别完全不一样（construction 需要 bucket, gripper, gravel-pile, container 等域内类别）
- 数据稀缺，annotation 成本高
- dynamic actors 是 worker + 大型机械，运动模式与城市道路完全不同
- terrain 本身就是 traversability 的关键因素，而非简单 obstacle

paper 的关键 insight：用 large-scale pre-trained panoptic segmentation model + 小规模 fine-tuning，再通过 explicit LiDAR-based lifting + Kalman tracking + grid map 维护，把 2D pixel-level 语义提升到可被 motion planner 直接消费的 2D occupancy / cost map。这是一条"explicit pipeline"路线，与目前流行的 end-to-end BEV learning（如 BEVFusion [19]）形成对比。

参考文献：
- HEAP excavator: https://www.sciencedirect.com/science/article/pii/S0926580521001827
- Robotic Systems Lab: https://rsl.ethz.ch/
- ViPlanner: https://arxiv.org/abs/2402.19278

---

## 二、整体系统架构解析

Figure 2 展示了完整 pipeline，可以拆成三大模块：

### 1. Perception 模块（Section 3.1）
- 输入：camera image（4K Ximea xiX PCIe camera，竖直安装）
- 处理：Mask2Former（Swin-Tiny backbone）
- 输出：2D panoptic mask，每 pixel 同时带 semantic class 和 instance ID

### 2. Mapping & Tracking 模块（Section 3.2）
- 输入：2D panoptic mask + raw LiDAR scan (Ouster OS0) + state estimation from Graph-MSF
- 输出：3D 分为 static/dynamic 的 labeled point cloud + 2D semantic grid map (static layer + dynamic layer)

### 3. Motion Planning 模块（Section 3.3）
- 输入：occupancy map + cost map
- 处理：online RRT* (OMPL)
- 输出：trajectory，1Hz replanning

这是一个典型的 **modular robotics pipeline**，每个模块都做了简化但工程化极强的设计。和 end-to-end approach 相比，这种设计的优势在于：可解释、可调试、每模块可独立替换、数据需求小；劣势在于 error propagation、模块间 calibration 敏感。

---

## 三、Panoptic Segmentation 训练策略深度分析

### 3.1 模型选择与对比

paper 比较了两类 transformer-based 架构：

**DETR (Carion et al. 2020) [2]**
- Encoder: ResNet-50（25M 参数），仍然用 CNN 抽 feature
- Decoder: transformer，使用 attention 机制做 detection
- 原始论文: https://arxiv.org/abs/2005.12872

**Mask2Former (Cheng et al. 2021) [4]**
- Encoder: Swin Transformer（可变 size）
- Decoder: masked-attention transformer
- 原始论文: https://arxiv.org/abs/2112.01577
- 关键机制：query 用 masked attention 限制在 predicted region 内部，避免 background 干扰

### 3.2 关键决策：训练 regime 的 ablation

Table 1 是 DETR 的 ablation，对比 freeze backbone vs fine-tune backbone：
| Method | PQ | SQ | RQ |
|--------|-----|-----|-----|
| Freeze backbone | 0.35 | 0.61 | 0.44 |
| Fine-tune backbone | 0.41 | 0.62 | 0.52 |

**Intuition 构建**：ResNet-50 在 ImageNet 1k 上 pretrain，但 construction site 的 first-person view（excavator 上方 vertical mount 的 camera）与 ImageNet 的 object-centric distribution 严重不一致。当 freeze backbone 时，feature representation 不够 "construction-aligned"，PQ 提升主要来自 decoder 学习。Fine-tune backbone 后 RQ（recognition quality，即分类准确率）从 0.44 → 0.52，提升最明显，这说明 backbone 学到了更好的 domain-specific 特征。

这一点呼应了视觉界长期争论的 "frozen vs tuned backbone for downstream task"，例如 ViT 在 segmentation 上的相关研究（https://arxiv.org/abs/2010.11929）。

### 3.3 三步训练法

DETR 的三步 training：
1. 先 fine-tune box detection 部分
2. freeze backbone + 主体，单独训 segmentation head
3. 整体 end-to-end fine-tune 直到 200 epochs

这是一种 "warm-up + special branch + joint" 的 curriculum，对应 detection 已经是较难的任务，需要先建立稳定的 object-level representation 再加 segmentation。这对 small dataset 尤其重要，避免 joint training 早期梯度噪声破坏 backbone 的 pretrain 特征。

### 3.4 Model size scaling 分析（Table 2）

| Model | PQ | SQ | RQ | Images/s |
|-------|-----|-----|-----|----------|
| DETR | 0.41 | 0.62 | 0.52 | 12 |
| M2F Swin-Tiny (29M) | 0.68 | 0.79 | 0.81 | 8 |
| M2F Swin-Big (88M) | 0.63 | 0.77 | 0.73 | 3 |
| M2F Swin-Large (200M) | 0.68 | 0.79 | 0.80 | 1 |

**反直觉现象**：Swin-Big 反而比 Swin-Tiny 差（PQ 0.63 vs 0.68）。但 Swin-Large 又追回来了。

Intuition：在 502 images + 1290 COCO subset 这种小数据规模下，更大的 model 容量没有相应的 data 支撑，出现过 fitting 现象（Swin-Big 的 RQ 0.73 vs Swin-Tiny 0.81）。Swin-Large 反而表现持平，可能是因为参数量到达某个 critical regime 后 pretrain 的 prior 更强、抗 overfit 能力更好。这与 LLM 中的 "scaling law breakdown in small data regime" 现象类似。

**Throughput tradeoff**：deploy 在 RTX 3080Ti 上，Swin-Tiny 8 images/s 是 sweet spot；Swin-Large 1 image/s 对于 1Hz planner 来说勉强够用，但留给 LiDAR processing 和 planning 的预算就紧了。

### 3.5 Dataset 组合策略（Table 3）

| Train set | Val set | PQ | SQ | RQ |
|-----------|---------|-----|-----|-----|
| Custom only | COCO val | 0.06 | 0.18 | 0.09 |
| Custom only | Custom val | 0.41 | 0.54 | 0.51 |
| Custom + COCO | COCO val | 0.26 | 0.46 | 0.36 |
| Custom + COCO | Custom val | 0.41 | 0.53 | 0.52 |

**Catastrophic forgetting**：仅用 Custom 训练时，COCO val 上的 PQ 掉到 0.06，基本完全忘了 COCO 的知识。引入 3:1 的 COCO:Custom ratio 后，COCO val 上 PQ 恢复到 0.26，同时 Custom val 上基本不损失（PQ 0.41 → 0.41）。

这是一个非常实用的 finding：**在小数据域适配时，混入原 pretrain dataset 的子集能 mitigation catastrophic forgetting，且几乎不损失 in-domain 性能**。这与 continual learning 领域的经验回放（experience replay）思想一致。

参考 replay-based continual learning 综述: https://arxiv.org/abs/1909.08370

### 3.6 Dataset size scaling（Table 4）

| Fraction | 20% | 40% | 60% | 80% | 100% |
|----------|-----|-----|-----|-----|------|
| PQ | 0.53 | 0.57 | 0.59 | 0.62 | 0.68 |
| RQ | 0.63 | 0.67 | 0.70 | 0.72 | 0.81 |
| SQ | 0.77 | 0.77 | 0.78 | 0.79 | 0.78 |

**关键观察**：
- PQ 和 RQ 近似线性增长，**没有 plateau**
- SQ 几乎不变

Intuition：SQ（segmentation quality）衡量的是 mask 形状的 IoU，主要依赖 backbone 学到的 spatial feature，这部分来自 ImageNet/COCO pretrain，迁移性好。而 RQ（recognition quality）依赖 head 的 class-specific 决策，domain shift 大，需要更多数据学习 construction-specific 类别的判别 boundary。这就解释了为什么 SQ 不动而 RQ 大幅提升。

paper 据此建议：如果有更多 annotation budget，继续扩充数据集 PQ 还会涨。这与 scaling law 在 supervised segmentation 上的实证经验一致。

---

## 四、Dynamic Mapping & Tracking 算法详解

这一块是 paper 的工程核心，值得细看。

### 4.1 Pipeline 分解

1. **LiDAR-camera 投影**：每个 LiDAR point $p_i \in \mathbb{R}^3$ 通过外参矩阵 $T_{CL} \in SE(3)$ 投影到 camera image plane，得到 pixel $(u_i, v_i)$，取该 pixel 的 segmentation class label。
   - 公式形式：$u_i = \pi(K T_{CL} p_i)$，其中 $K$ 是 camera intrinsic，$\pi$ 是 perspective division。
   - 视野外或无 depth 的 point：标为 "unknown"

2. **Ground removal + DBSCAN clustering**：
   - 先 RANSAC 或类似方法去掉 ground plane（避免 ground 点和上方的 object 连成大 cluster）
   - DBSCAN 在剩余 labeled point cloud 上做 density-based clustering
   - DBSCAN 的关键参数：$\epsilon$（neighborhood radius）和 $\text{minPts}$（最小 cluster 大小）
   - 参考: https://www.aaai.org/ojs/index.php/AAAI/article/view/5073

3. **Majority voting per cluster**：
   - 对每个 cluster，统计所有 point 的 label 分布
   - 取 majority label，若 majority 比例低于阈值 $\tau$（论文未给具体值），则 cluster 标为 "unknown"
   - 这是 "soft decision → hard decision" 的鲁棒化操作，避免单个 misclassified point 污染整个 cluster

4. **Kalman Filter tracking**：
   - State vector（标准 CV model）：$\mathbf{x} = [x, y, z, v_x, v_y, v_z]^T$
   - Transition matrix（constant velocity 假设）：
     $$F = \begin{bmatrix} I_3 & \Delta t \cdot I_3 \\ 0 & I_3 \end{bmatrix}$$
   - Observation：cluster 的 bounding box center（在 map frame 下）
   - 关键 trick：**object 离开 camera FoV 后，仍保留之前的 semantic class**，仅靠 LiDAR geometry 维持 tracking。这是 dynamic mapping 在"感知间断"下保持一致性的核心机制。

5. **Static / Dynamic 分类**：
   - "things" classes（person, machinery 等）：进入 dynamic cloud
   - "stuff" classes（terrain, gravel, wall 等）：进入 static cloud
   - "unknown"：进入 dynamic cloud（保守策略——不知道就当作 dynamic 处理）

### 4.2 Semantic Grid Map 双层结构

使用 GridMap library [8]（ETH 自己开发的 ROS 包，参考 https://github.com/ANYbotics/grid_map）：

- **Static layer**：累积 "stuff" 点，一旦写入就长期保留（即使之后被遮挡）
- **Dynamic layer**：仅保留当前 frame 的 "things" + "unknown" 点

每 update cycle 时：
- Static layer 不被覆盖
- Dynamic layer 覆盖在 Static layer 上
- 合并后输出 merged semantic layer

**关键 intuition**：双层设计避免了 "person 走过 gravel 时把 gravel cell 覆盖成 person 然后清除掉" 的常见问题。当 person 离开，gravel 信息自动重新出现在该 cell（因为 static layer 一直在）。

### 4.3 Cost / Occupancy map 生成

- **Occupancy map**：基于 class metadata，"stuff" 中的 traversable class（road, gravel, grass）→ 0；non-traversable（person, fence）→ 1
- **Cost map**：traversability cost 手动定义（mud 高 cost，gravel 中 cost，road 低 cost）
- 可选：**a priori geometric map union operation**：把预先知道的 geofence、site boundary 和 occupancy map 做 union，限制 robot 工作范围

这种 explicit metadata-driven 设计非常工程化：cost 是手动 tune 的，但因为是 grid map layer，可解释性极强，调试方便。

---

## 五、Motion Planning 公式与策略

### 5.1 Total cost 公式

$$C_{\text{total}} = \lambda_1 C_{\text{length}} + \lambda_2 C_{\text{semantic}} \tag{1}$$

变量解释：
- $C_{\text{total}}$：trajectory 总 cost
- $C_{\text{length}}$：path 在 metric space 的总长度（单位：meters）
- $C_{\text{semantic}}$：path 上每个 segment 在 excavator footprint 区域内 cost map 值的累积和
- $\lambda_1 = 1$：长度 weight
- $\lambda_2 = 0.1$：semantic cost weight

**$\lambda$ 取值的 intuition**：$\lambda_2$ 小（0.1）是因为 semantic cost 是 cell-level 的累积，path 越长 cell 越多累积越大，如果不 discount 会主导。$\lambda_1$ 主导说明他们偏好 short path，但 semantic cost 在差别大时（mud vs road）能改变 path 选择。

每个 segment 的 semantic cost 计算方式：
$$C_{\text{semantic}}(\text{segment}) = \frac{1}{|A|} \sum_{c \in A} \text{cost}(c)$$
其中 $A$ 是 excavator footprint 在该 segment 处覆盖的 cell 集合，$|A|$ 是 cell 数量。这就是 footprint-averaged traversability cost。

### 5.2 RRT* 配置

- Library: OMPL (Open Motion Planning Library，https://ompl.kavrakilab.org/)
- Online RRT*（asymptotically optimal），运行 1Hz，单次 max 0.95s
- Validity check：excavator footprint 下所有 occupancy 值为 0
- Trajectory comparison：新 trajectory 的 cost 与当前 path remaining cost 比较，若更低则切换
- Safety：若 collision risk 在 3m radius 内，cost 设为 $\infty$，robot 停止

### 5.3 为什么用 RRT* 而非优化-based planner

Construction site 的 free space 是高度 non-convex 的（窄通道、堆料间间隙、机械臂附近），A* 在 grid 上需要精细 resolution，optimization-based（CHOMP, TrajOpt）容易卡 local minima。RRT* 在 high-dim + non-convex space 上的 exploration 能力强，且 1Hz 频率配合 0.95s budget 足够 reactive。

参考: https://arxiv.org/abs/1103.4402 (RRT* 原始论文 by Karaman & Frazzoli)

---

## 六、实验分析

### 6.1 Offline Segmentation Evaluation

已在 3.2, 3.5, 3.6 节分析。重点结论：
1. Mask2Former Swin-Tiny 性价比最高（PQ 0.68, 8 img/s）
2. 必须混入 COCO 子集（catastrophic forgetting 防御）
3. 数据量 100% 时仍线性增长，无 plateau → 数据集还可扩

### 6.2 Online Field Test

- 测试地点 1：controlled testing field（narrow, fence/pole/bucket 等 obstacle-dense）
- 测试地点 2：Rescue Troop Training Center, Avully, Switzerland（open space + road-following + dynamic obstacle）

Figure 7 是 "adversarial test"——人故意挡 path，系统重规划绕过。关键 insight 是：**online planner + dynamic map + tracking 协同让 robot 在动态干扰下保持 robust**。这本质上是 modular pipeline 的 reactive 能力展示，而非 learning-based policy 的 generalization。

### 6.3 Limitation: Confidence Threshold 问题（Figure 6）

Mask2Former 在 ambiguous terrain（dirt / gravel / pavement 边界）上经常**无法给出单一 dominant class**——logits 在多个 plausible class 上分散，max confidence 低于阈值导致整个区域被标 "unknown"。

这是 closed-vocabulary segmentation 的本质限制：terrain 是 spectrum 而非 discrete class。未来方向（paper Future Work 提到）：open-vocabulary segmentation（如 SAM, CLIP-Seg），让 model 输出 continuous semantic embedding 而非 hard label。

参考 SAM: https://arxiv.org/abs/2304.02643

---

## 七、与相关工作的位置关系

### 7.1 vs End-to-End BEV Learning（如 BEVFusion [19], LSS [26]）

| 维度 | ETHcavation | BEVFusion 等 |
|------|-------------|------------|
| Data 需求 | 小（500 images） | 大（nuScenes 级） |
| Domain | construction (小众) | urban driving (大众) |
| Explainability | 高（每层可视化） | 中（BEV feature 可视但语义弱） |
| Dynamic handling | 显式 Kalman tracking | 隐式（时序网络） |
| Adaptation cost | 重训小 head + 少量 data | 重训整个 fusion network |

### 7.2 vs TNS (Guan et al. 2022) [10]

TNS 也是 excavator traversability mapping 的工作。区别：
- TNS 用 LiDAR-only + 2D semantic image（不带 tracking）
- TNS 不显式处理 dynamic obstacle
- ETHcavation 增加了 tracking + dynamic/static layer 分离

### 7.3 vs ViPlanner (Roth et al. 2024) [28]

同一组的前序工作，使用 imperative learning 从 depth + semantic image 学 reactive local navigation policy，**不显式建 BEV map**。ETHcavation 是 explicit map-based 路线，两者互补——ViPlanner 在小障碍物 reactive 时快，ETHcavation 在 long-horizon planning 时稳。

参考: https://arxiv.org/abs/2402.19278

---

## 八、Critique 与延伸思考

### 8.1 优点
- **Engineering 完整度**：从 dataset 到 model 到 mapper 到 planner，全栈打通
- **Ablation 设计**：model size, training regime, dataset composition, dataset size 四个维度都做了 ablation，质量高
- **Open-source**：dataset + code 都公开（https://github.com/ETHZ-RobotX/ETHcavation, https://rsl.ethz.ch/robotics-datasets.html）
- **Catastrophic forgetting 处理经验**：3:1 COCO:Custom 这个 ratio 的实证对后续 domain adaptation 工作很实用

### 8.2 可改进点

1. **Confidence threshold 的 brittle behavior**：建议用 temperature scaling + calibrated probability，或者直接用 entropy-based rejection
2. **手工 cost tuning**：$\lambda_1, \lambda_2$ 和 traversability cost 表都是手调的，缺乏从 demonstration 学的方法。可以用 IRL（inverse reinforcement learning）从 expert demonstration 学 cost
3. **仅 2D grid map**：对于有 overhead structure（如桥下、棚下）的场景 2D 不够。paper 在 Future Work 中提到扩展到 full 3D
4. **LiDAR-camera calibration 依赖**：投影误差会直接污染 label assignment，paper 没讨论 calibration sensitivity
5. **Tracking 用 CV model**：对于突然变速的 worker 或机械不合适。可以考虑 IMM（Interacting Multiple Model）Kalman 或简单的 acceleration-augmented state
6. **Pretrain→fine-tune paradigm 已被 foundation model 超越**：在 2025-2026 的时间点，用 SAM2、Grounding-DINO + CLIP 这类 open-vocabulary model 做 zero-shot 或 prompt-based adaptation 可能更高效

### 8.3 与 SAM2 / Foundation Model 时代的关系

paper Future Work 提到 "open-vocabulary systems" 方向。2024 年后 SAM2（https://arxiv.org/abs/2407.09512）和 Grounding DINO（https://arxiv.org/abs/2303.05499）已经可以做 zero-shot segmentation，construction site 上是否能直接用？可能的方向：
- 用 SAM2 + CLIP 做 prompt-based label assignment（"bucket", "gravel pile" 等 text prompt）
- 优势：无需 fine-tune，catastrophic forgetting 自然消失
- 劣势：real-time 性能（SAM2 在 RTX 3080Ti 上能否达到 8Hz？需要测试）

### 8.4 Dataset 的价值

502 images 看起来很小，但对比其他 construction-related dataset：
- 典型 construction dataset（如 Guan et al. 的 TNS dataset）也仅百级千级
- Cityscapes 是 5000 images，但 domain 不同
- nuScenes 是 40k annotated keyframes，但需要 multimodal sensor suite

ETHcavation 的价值在于：
1. **Domain-specific label space**：34 类含 bucket, gripper, self-arm 等，是任何通用 dataset 都没有的
2. **First-person excavator view**：camera 姿态特殊，对 reid/tracking 研究有价值
3. **Multi-site 覆盖**：construction site + road + natural environment，避免单 site overfit

可在 https://rsl.ethz.ch/robotics-datasets.html 找到。

---

## 九、关键 References 与扩展阅读

- **Mask2Former 原始 paper**: https://arxiv.org/abs/2112.01577
- **DETR**: https://arxiv.org/abs/2005.12872
- **Panoptic Segmentation (Kirillov et al. 2019)**: https://arxiv.org/abs/1901.02446
- **Swin Transformer**: https://arxiv.org/abs/2103.14030
- **Graph-MSF (state estimation)**: https://arxiv.org/abs/2204.12230
- **GridMap library**: https://github.com/anybotics/grid_map
- **OMPL**: https://ompl.kavrakilab.org/
- **HEAP excavator**: https://www.sciencedirect.com/science/article/abs/pii/S0926580521001827
- **ViPlanner**: https://arxiv.org/abs/2402.19278
- **DBSCAN 原始 paper**: https://www.aaai.org/ojs/index.php/AAAI/article/view/5073
- **RRT* 原始 paper**: https://arxiv.org/abs/1103.4402
- **ETHZ-RobotX GitHub（代码 + 数据）**: https://github.com/ETHZ-RobotX/ETHcavation
- **nuScenes**: https://www.nuscenes.org/
- **SAM2 (Meta)**: https://arxiv.org/abs/2407.09512
- **Grounding DINO**: https://arxiv.org/abs/2303.05499
- **Construction robot review (Jung et al.)**: https://doi.org/10.1016/j.autcon.2023.104951

---

## 十、最终 Takeaway

这篇 paper 不是追求 SOTA metric 的工作，而是一个 **complete stack + open dataset + practical guidelines** 的综合性 contribution。对 robotics researcher 来说最有价值的几个 takeaway：

1. **小数据域适配的标准配方**：大 model pretrain + 小数据 fine-tune + 混入原 pretrain dataset subset（3:1 ratio 实证最佳）
2. **Static/Dynamic layer 分离的 grid map 设计**：解决 dynamic obstacle 走过留下 "hole" 的经典问题，简单但有效
3. **Explicit pipeline 的可解释性优势**：在 safety-critical 的 construction 场景，比 end-to-end 更值得信赖
4. **Confidence threshold 的 brittle behavior** 是 closed-vocabulary segmentation 的本质问题，未来应转向 open-vocabulary / continuous embedding 路线

Karpathy 你应该会特别喜欢这里的 modular pipeline 设计思路——每个 component 都有明确 input/output contract，很像 software engineering 的 layer 分离。在 robot learning 大一统的今天，这种"还能调试、还能替换、还能 trust"的 explicit 方法依然有其不可替代的位置。同时这也是一个适合做 student project 的 baseline——dataset 和 code 都开源，可以从 fine-tune 自己的 SAM2-based panoptic segmentor 开始替换，进而改进 mapper 或 planner 中的任一模块。
