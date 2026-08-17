---
source_pdf: BridgeVLA.pdf
paper_sha256: 9076d1b5e3c4f746d9f1280b2d87cf6c761dc20b0ce6504edd28c00319bd9bf1
processed_at: '2026-08-03T14:28:28-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# BridgeVLA 人话版

---

## 这篇paper到底在干嘛

想象你在教一个实习生做手术。这个实习生读过所有医学书（大模型VLM），理论知识爆表，但手生。你怎么让他快速上手？

有两个选择：
- **A方案**：给他看100台手术视频再说 → 等他学会了病人都凉了
- **B方案**：直接手把手教3次他就 会了 → 但他可能不懂你在说啥

BridgeVLA说：**我两个都要**。

---

## 核心矛盾

现在机器人学界有两个流派在打架：

**大模型派**（π₀、OpenVLA这些）：拿个看过全网图片和文字的大模型，让它学机器人。好处是聪明、懂人话、能举一反三。坏处是巨婴——你得给它喂几百条示范它才学会一个动作。paper里π₀用10条数据训练，真实机器人上成功率3.8%，基本全废。

**3D派**（RVT-2、PerAct这些）：直接看3D点云，利用空间几何结构。好处是学得飞快，10条数据就能90%成功率。坏处是脑子笨，不懂语义，换个没见过的物体就懵。

之前有人想"把3D塞进大模型"（SpatialVLA、3D-VLA），结果两头不讨好——既没沾到大模型的语义知识红利，又把大模型的预训练分布搞乱了。

BridgeVLA的做法是：**别硬塞，当翻译**。

---

## 三个翻译技巧

### 翻译一：把3D拍扁成2D

大模型这辈子只见过2D照片。你给它3D点云，它一脸懵。

BridgeVLA说：那我给你拍三张照片呗。从上往下拍一张，从前往后拍一张，从右往左拍一张。这三张就是3D点云的"三个影子"。

大模型一看：哦这不就是照片嘛，我熟。然后该怎么处理怎么处理。

**关键**：大模型根本不知道有3D存在，它以为自己在看普通照片。预训练学到的本事全能用上。

### 翻译二：动作变成"找亮点"

大模型以前学的是"看图说话"——输出一串文字token。

但机器人动作是个3D坐标，你怎么变成token？之前的人硬把它变成一串数字token，大模型完全不知道这串数字跟图片哪个位置对应，学起来特别费劲。

BridgeVLA说：别变成数字串了，变成一张"热力图"吧。

热力图就是一张跟输入图片一样大的图，大部分地方是黑的，目标位置是亮的（一个高斯亮斑）。大模型的任务就是：在图上点出"机器人下一步该去哪"。

这张热力图跟输入图片**像素对像素对齐**。大模型觉得：哦这不就是找东西嘛，我以前预训练干的就是找物体的活。

而且这个trick有个隐藏好处：一张224×224的热力图有5万个像素，每个像素都贡献监督信号。你要是直接回归3个坐标值，只有3个监督信号。**监督信号多了5万倍**，当然学得快。

### 翻译三：预训练也改成"找亮点"

原始大模型预训练是next-token prediction（预测下一个文字token）。但你要它输出热力图，它不会啊。

BridgeVLA就先给它做个"岗前培训"：用12万张图片，每张图片配一句描述（比如"找杯子"），让模型学会在图上画热力图圈出对应物体。

这个岗前培训跟后面的机器人任务**输出格式一模一样**——都是画热力图。所以模型觉得：预训练和微调是同一件事，无缝衔接。

---

## 实验结果有多炸

### 真实机器人（最能说明问题）

13个任务，每个任务只给10条示范轨迹：

| 方法 | 成功率 |
|---|---|
| π₀（大模型派大哥） | 3.8% 基本全废 |
| SpatialVLA（50条数据！） | 28.5% |
| ACT | 22.3% |
| RVT-2（3D派SOTA） | 90% |
| **BridgeVLA（10条数据）** | **96.9%** |
| **BridgeVLA（3条数据！）** | **95.4%** |

**3条轨迹就95%成功率**。这是个什么概念？你在机器人前面摆个杯子，手动带它走3遍"拿起来放那边"，它就会了。

π₀用同一个大模型backbone却3.8%，纯粹是架构设计的差距。

### 泛化能力

换了灯光、换了背景、加了干扰物、换了没见过的物体组合——BridgeVLA平均比RVT-2高32个百分点。

特别是在"新组合"上：训练时见过红方块和蓝盘子，也见过"放到里面"这个动作，但没见过"把红方块放到蓝盘子里"这个具体组合。BridgeVLA能泛化过去，RVT-2不行。

这说明大模型的语义理解确实起作用了——它懂"红方块"是啥、"盘子"是啥、"放进去"是啥，能自由组合。

---

## 三个消融实验（最精彩的部分）

### 消融一：不画热力图，直接回归坐标 → 88%跌到31%

把热力图头换成同样大小的Transformer decoder，直接输出3个坐标值，用MSE loss。参数量一样。

**成功率从88%暴跌到31%**。

为什么？热力图每个像素都给梯度信号（5万个监督点），直接回归只有3个。而且分类问题比回归问题好优化——这是检测领域早就知道的道理（anchor box classification比直接回归bbox好做）。

### 消融二：给大模型额外塞3D坐标 → 88%跌到56%

模仿SpatialVLA的做法，把每个像素的3D坐标编码后跟图片特征拼一起喂进去。

**加了更多信息，结果反而更差**。

这个反直觉的结果说明：对预训练大模型来说，**别乱改它的输入格式**。它这辈子只见过纯RGB图片，你塞一堆3D坐标进去，它预训练学到的特征全乱了。输入分布一致性比额外信息更重要。

### 消融三：不做热力图预训练 → 语义泛化崩了

跳过"找物体"的岗前培训，直接微调。结果在"新组合"和"新类别"上泛化能力大幅下降。

这说明预训练不只是warmup，它教会的"语言-图像区域"关联能力是泛化的根基。

---

## 一句话总结

BridgeVLA的哲学就四个字：**别折腾大模型**。

它这辈子吃2D图片、吐文字token，你就让它继续这么干。3D信息在进大模型之前就拍扁成2D，动作输出在大模型之后才从热力图解出来。大模型全程不知道3D存在，活得跟预训练时一样舒服。

结果就是：大模型的语义知识没丢，3D的空间结构prior也用上了，3条数据就能教会机器人干活。

**当翻译，别当改造犯**。

---

# BridgeVLA 深度技术解析

 Andrej，这篇 paper 我读了三遍，下面是我对它的拆解，重点放在 build intuition 和技术细节上。

---

## 1. 核心问题：3D VLA 的两难困境

先讲清楚这篇 paper 在打什么 target。当前 robot manipulation 有两条路：

**路线 A — 2D VLA** (π₀, OpenVLA, RT-2, SpatialVLA)
- 用大规模预训练 VLM (PaliGemma, PaLI, PaLM-E) 作 backbone
- 强 generalization、强 language understanding
- 痛点：sample efficiency 极差。π₀ 在 paper 里显示只用 10 条轨迹基本全跪（3.8% success rate），SpatialVLA 用 50 条才 28.5%。

**路线 B — 3D policy** (PerAct, Act3D, RVT, RVT-2)
- 直接吃 point cloud / voxel / multi-view 投影
- 利用了 3D 空间结构 prior
- sample efficiency 极高（RVT-2 在 10 条数据上能到 90%）
- 痛点：没有 VLM 的 web-scale 知识，language grounding 弱

**已有的 3D VLA 尝试** (3D-VLA, SpatialVLA, PointVLA, FP3) 都犯了两个错误：
1. 把 action 转成 token sequence，用 next-token prediction — 丢掉了 3D 空间结构 prior
2. 把 3D position encoding 注入 VLM — 破坏了 VLM 预训练时的 2D RGB input distribution

BridgeVLA 的核心 claim：**用 alignment 同时吃到 VLM 知识 + 3D 结构 prior 的红利**。

---

## 2. 三个 alignment（这是 paper 的灵魂）

### 2.1 Input Modality Alignment
VLM 预训练时吃 2D RGB。Fine-tune 时把 point cloud 用 orthographic projection 渲染成 3 个 2D 图像（top / front / right view），跟 RVT/RVT-2 的 trick 一样。VLM 看到的还是 2D RGB，distribution shift 最小化。

### 2.2 Input-Output Structural Alignment
输入是 2D spatial grid (image patches)，输出也是 2D spatial grid (heatmap)。这两个 grid **共享同一分辨率**。这是跟 3D-VLA / SpatialVLA 把 action tokenize 成 1D 序列最根本的区别。

为什么这件事关键？我下面 ablation 会展开。

### 2.3 Pre-train / Fine-tune Task Alignment
Pre-training 做 object grounding (predict 2D heatmap of target object)，fine-tune 做 keyframe keypoint localization (predict 2D heatmap of end-effector target position)。两个 task 的 output modality、loss landscape、learning dynamics 都高度相似。

---

## 3. 架构细节

### 3.1 Backbone: PaliGemma
- Vision encoder: SigLIP (sigmoid loss for image-text contrastive pretraining)
- Language backbone: Gemma transformer
- 关键细节: **image token 和 prefix text token 用 bidirectional attention；suffix text token 用 causal attention**

这个设计意味着 image patch token 之间可以 fully attend 互相 fusion，并且能 attend 到 question text。这对 spatial reasoning 很重要 — 如果用 causal attention on image tokens，空间信息聚合会被严重限制。

PaliGemma 输出的 image token 是 patch-ordered 的，可以 rearrange 回 2D spatial grid。

### 3.2 Heatmap Head — Convex Upsampling
这是从 RAFT (Teed & Deng, ECCV 2020) 借来的 module。

具体机制：低分辨率 feature grid $\mathbf{F} \in \mathbb{R}^{H/W \times W/W \times C}$（$W$ 是 patch size，比如 14）要通过学习的 weights 上采样到 $\mathbb{R}^{H \times W}$。

RAFT 原版公式（高维版本）：
$$\hat{\mathbf{H}}(\mathbf{x}) = \sum_{i \in \mathcal{N}(\mathbf{x})} w_i(\mathbf{x}) \cdot \mathbf{F}(i)$$

其中 $\mathcal{N}(\mathbf{x})$ 是低分辨率 grid 上 $\mathbf{x}$ 周围的 $k \times k$ 邻域（RAFT 用 $k=9$，即 81 个 neighbors），$w_i(\mathbf{x})$ 是 learned convex weights，满足 $\sum_i w_i = 1$。Convex 约束通过 softmax 实现。

相比 bilinear（4 个 neighbors 固定 weights），convex upsampling 学到 81 个 learned weights per output pixel，能恢复 fine spatial detail。Parameter count 309M（包括 backbone）。

### 3.3 Ground Truth Heatmap 构造
对每个物体 $i$，先建一个 truncated 2D Gaussian：

$$H_i^{\text{gt}}(\mathbf{x}) = \begin{cases} p_i(\mathbf{x}) & \text{if } p_i(\mathbf{x}) \geq p_{\min} \\ 0 & \text{otherwise} \end{cases}$$

$$p_i(\mathbf{x}) = \exp\left(-\frac{\|\mathbf{x} - \hat{\mathbf{x}}_i\|^2}{2\sigma^2}\right)$$

变量解释：
- $\mathbf{x} = (u, v)$: pixel 坐标
- $\hat{\mathbf{x}}_i$: 第 $i$ 个 object 的 bounding box 中心
- $\sigma$: 高斯标准差，控制 heatmap "尖度"
- $p_{\min}$: threshold，截掉低概率区域让 supervision 聚焦在 peak 附近

多物体时 average + normalize：
$$H^{\text{gt}}(\mathbf{x}) = \frac{H_{\text{avg}}(\mathbf{x})}{\sum_{\mathbf{x}' \in \Omega} H_{\text{avg}}(\mathbf{x}')}, \quad H_{\text{avg}}(\mathbf{x}) = \frac{1}{N}\sum_{i=1}^N H_i^{\text{gt}}(\mathbf{x})$$

注意 normalize 是 sum-to-1，这样 cross-entropy 直接适用（heatmap 当成 categorical distribution over pixels）。

### 3.4 Fine-tune 阶段：3D → 2D → Heatmap → 3D

Pipeline:
1. RGB-D 重建 point cloud $\mathbf{P} \in \mathbb{R}^{N \times 3}$（带颜色）
2. 三个 orthographic 投影：top / front / right → 3 张 RGB 图像
3. 3 张图 + instruction → PaliGemma → 3 个 view 的 image token grid
4. Convex upsampling → 3 个 heatmap $\mathbf{H}^{\text{top}}, \mathbf{H}^{\text{front}}, \mathbf{H}^{\text{right}}$
5. **Back-project**：把 workspace 里均匀采样的 3D grid points $\{\mathbf{p}_j\}$ 投影到每个 view，得到像素坐标，从 heatmap 取值
6. 3D point 的 score = 三个 view heatmap 值的 mean
7. 取 argmax $\mathbf{p}^* = \arg\max_j \text{score}(\mathbf{p}_j)$ 作为 next keyframe translation

**关键 intuition**: 这里 heatmap 不是直接监督 3D 位置，而是通过 2D supervision + 几何投影间接监督。三个 view 的 intersection 提供 3D localization 的 disambiguation。这种 "soft intersection" 在数学上跟 multi-view geometry 的 triangulation 类似，但用 learned heatmap 替代精确的几何对应。

### 3.5 Rotation / Gripper / Collision Head
- Rotation: Euler angles，每轴 discretize 成 72 bins（5° per bin），共 216-way classification
- Gripper: binary
- Collision flag: binary（指示 motion planner 是否要避障）

特征融合策略有意思：
- **Global**: 每个 view 的 image tokens 做 max-pooling → 3 tokens（一个 view 一个）
- **Local**: 每个 view 的 heatmap peak 位置提取 token → 3 tokens
- Concat 6 tokens → MLP → rotation / gripper / collision

这种 global + local 的设计很关键。Global 给 scene context，local 给 "在 target 附近的细节"。类似 DETR 的 object query + global context 的关系。

### 3.6 Coarse-to-Fine Refinement
第一遍在完整 point cloud 上预测 translation $\mathbf{p}^*$。然后在 $\mathbf{p}^*$ 周围 crop 一个 cuboid，重新渲染 zoom-in 的 orthographic 投影，第二遍 forward pass 得到 refined action。这是 RVT-2 的 trick，主要为了高精度 task（insert peg 这种需要 mm 级精度）。

### 3.7 Total Loss
$$L = L_{\text{trans}} + L_{\text{rot}} + L_{\text{gripper}} + L_{\text{collision}}$$

- $L_{\text{trans}}$: cross-entropy on heatmap（heatmap 看成 pixel-level categorical）
- $L_{\text{rot}}$: cross-entropy on 216 rotation bins
- $L_{\text{gripper}}, L_{\text{collision}}$: binary cross-entropy

注意 $L_{\text{trans}}$ 是 dense pixel-level loss，每个 pixel 都贡献 gradient — 这是 sample efficiency 的关键。

### 3.8 Data Augmentation
Random rigid-body transformation 同时施加到 point cloud 和 ground-truth action 上。这增强了 geometric invariance — 模型学到的是 "在 point cloud 里的相对几何关系"，而不是绝对坐标。

---

## 4. 实验数据深度分析

### 4.1 RLBench (18 tasks, 100 demos/task)

| Method | Avg SR | Avg Rank | Insert Peg | Sort Shape |
|---|---|---|---|---|
| RVT-2 (SOTA) | 81.4 | 2.75 | 40.0 | 35.0 |
| 3D Diffuser Actor | 81.3 | 2.67 | 65.6 | 44.0 |
| **BridgeVLA** | **88.2** | **2.03** | **88.0** | **60.8** |

Insert Peg 和 Sort Shape 是 high-precision task。BridgeVLA 在这两个 task 上 margin 巨大（peg +48pp, shape +25.8pp）。我推测原因是 heatmap + coarse-to-fine 的组合在精度 task 上有优势 — heatmap 给的是 spatial distribution 而不是 single point estimate，对 noise 更鲁棒；coarse-to-fine 在 zoom-in 后还能保持 heatmap 表达。

Place Cups 是 BridgeVLA 唯一相对弱的 task（58.4%）。Paper 解释：target keypoint 在所有 orthographic view 中都被 occluded。这暴露了 fixed view scheme 的局限 — 如果物体被遮挡，三个 view 都看不到目标位置，heatmap 学不到。Future work 提到 dynamic view selection。

### 4.2 COLOSSEUM (generalization benchmark, 14 perturbations)

| Method | Avg SR | All-Pert | MO-Color | RO-Size | Lighting | Distractor |
|---|---|---|---|---|---|---|
| RVT-2 | 56.7 | 15.6 | 53.0 | 53.4 | 58.0 | 60.8 |
| **BridgeVLA** | **64.0** | **18.7** | **60.5** | **69.7** | **75.7** | **51.8** |

14 个 perturbation 中 13 个 best。**唯独 Distractor 输给 RVT-2**（51.8 vs 60.8）。我猜测原因：distractor 是 visually similar 的干扰物体，VLM 的 grounding 能力反而会被 distractor 误导（VLM 倾向于把所有相似物体都 high score）。RVT-2 没有 language grounding，只看几何结构，反而不容易被语义干扰。

Lighting 大幅领先（75.7 vs 58.0）说明 VLM 的 visual pretraining 带来了强 lighting invariance。

### 4.3 GemBench (4 levels of generalization)

| Method | Avg | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| 3D-LOTUS++ | 48.0 | 68.7 | 64.5 | 41.5 | 17.4 |
| **BridgeVLA** | **50.0** | **91.1** | **65.0** | **43.8** | **0.0** |

L4 (long-horizon) BridgeVLA 是 0%，跟其他 baseline 一样烂（除了 3D-LOTUS++ 用了 LLM task planner 才到 17.4%）。这印证了 VLA 模型在 long-horizon planning 上的共同短板 — heatmap per-step prediction 不能处理 sub-task decomposition。

### 4.4 Real Robot (核心 highlight)

13 个 task，10 trajectories/task：

| Method | Avg |
|---|---|
| SpatialVLA (50 traj) | 28.5% |
| π₀ | 3.8% |
| ACT | 22.3% |
| RVT-2 | 90.0% |
| **BridgeVLA (10 traj)** | **96.9%** |
| **BridgeVLA (3 traj)** | **95.4%** |

**3 trajectories 就能 95.4% 是这篇 paper 最强的 claim**。π₀ 完全失败（3.8%）尤其有信息量 — π₀ 和 BridgeVLA 用同一个 PaliGemma backbone，差异完全在 architecture 设计上。这证明：**在 low-data regime，inductive bias 比 model capacity 重要**。

7 个 generalization setting：
- Visual: Distractor / Lighting / Background / Height
- Semantic: Combination (unseen object-skill pair) / Category (unseen object category)

BridgeVLA 在所有 7 个 setting 上都超过 RVT-2，平均 +32%。特别在 **Combination** 上 margin 巨大 — 这印证了 pre-training 给的 language-image grounding 让模型能 compose unseen 的 object-skill pair。

### 4.5 三个 Ablation（这是最有技术含量的部分）

#### Ablation 1: Heatmap vs. Direct Regression (88.2% → 31.4%)

去掉 convex upsampling head (309M params)，换成 Transformer decoder (303M params) 直接回归 3D translation，用 MSE loss。参数量基本一样。

**结果：88.2% → 31.4%，掉了 56.8 个百分点。**

Paper 给了三个原因：
1. **Dense vs. sparse supervision**: heatmap 是 pixel-level dense supervision，每个 pixel 都贡献 gradient；direct regression 只有 3 个 scalar 提供 supervision
2. **Spatial prior**: 把 3D points 投影到 heatmap 是 inductive bias，限制 hypothesis space
3. **Structural alignment**: 2D heatmap 跟 2D input image 共享 spatial structure，loss landscape 更 friendly

从 optimization 角度，我再加一个 intuition：**heatmap 是 classification over a discretized space, direct regression 是 continuous regression**。Classification 的 loss landscape 通常更 flat、更好优化，gradient direction 更明确。这跟 detection 领域用 anchor box classification 而不是直接回归 box coordinates 是一个道理。

Paper 还提到 ablated model 更难训：需要 batch size 192 + 仔细调 lr，原版 batch size 64 就稳。

#### Ablation 2: 加入 3D Position Encoding (88.2% → 56.2%)

加 3D conv module 编码 per-pixel 3D position，跟 2D feature 融合后喂给 backbone。模仿 SpatialVLA 的 Ego3D position encoding。

**结果：88.2% → 56.2%，掉了 32 个百分点。**

这个结果反直觉 — 加更多信息反而更差。Paper 的解释：3D position 改变了 image feature distribution，VLM 在预训练时没见过这种 feature，alignment 被破坏。

**这个 ablation 的深层含义**: 对预训练 backbone 来说，**input distribution 一致性 > 额外 modality 信息**。这跟 SimVLM、ViT 等工作的发现一致 — pretrain-finetune domain gap 是 performance killer。

#### Ablation 3: 去掉 2D Heatmap Pre-training

在 Combination 和 Category 两个 semantic generalization setting 上掉很多。这印证：**pre-training 教会模型把 language semantics 跟 image region 关联起来**，这种能力通过 heatmap prediction 被保留并迁移到 policy learning。

---

## 5. Build Intuition — 几个深层 insight

### 5.1 为什么 heatmap 是 "right" output representation

从 information theory 角度：
- 直接回归 3D 位置 $\mathbf{p} \in \mathbb{R}^3$：3 个 scalar 监督信号
- Heatmap $\mathbf{H} \in \mathbb{R}^{H \times W}$ over 3 views：$3HW$ 个 scalar 监督信号

对 RLBench 用 224×224 输入，单 view heatmap = 50176 pixels × 3 views ≈ 15 万 supervision signal per step。比直接回归多了 5 万倍。

从 inductive bias 角度：
- Heatmap 隐含 "nearby pixels should have similar probability" 的 spatial smoothness prior
- Heatmap 可以表达 multimodal uncertainty（虽然这里没显式用）
- Convex upsampling 学到的 upsampling 是 spatially local 的，跟 image 的 locality 对齐

### 5.2 为什么 3D-as-2D-projection 比 inject-3D-into-VLM 好

BridgeVLA 跟 SpatialVLA 的对比是 paper 的 implicit 主轴：
- SpatialVLA: 把 3D position encoding **注入** VLM 的 image token → 88.2% → 56.2% (ablation)
- BridgeVLA: 把 3D **投影**成 2D，VLM 不知道有 3D 存在 → 88.2%

差别在 "注入" vs "投影"：
- **注入** 改变 VLM 内部 representation
- **投影** 是 pre-VLM 的 preprocessing，VLM 看到的还是 RGB

这个发现对未来 3D VLA 设计有指导意义：**应该把 3D 处理放在 VLM 之外，保持 VLM input/output distribution 不变**。

### 5.3 PaliGemma 的 bidirectional image attention
PaliGemma 的 image token 用 bidirectional attention。这意味着每个 patch 能 attend 到所有其他 patch。这对 spatial reasoning 很关键 — 比如 "red block 相对于 green plate 的位置" 需要远距离 patch 之间通信。

如果用 causal attention（像 LLaVA 早期），spatial 信息流受限。这可能是为什么 PaliGemma 比 LLaVA 在 spatial task 上更适合做 VLA backbone。

### 5.4 跟 Diffusion Policy 的关系
Diffusion Policy 在 action space 做 diffusion，能 express multimodal action distribution。BridgeVLA 的 heatmap 也是 distribution（categorical over pixels），但只 support 单峰（Gaussian）。

Paper future work 提到 "incorporate more expressive action-decoding methods (e.g., diffusion)"。我推测未来版本会是：heatmap 提供 coarse localization prior，diffusion head 在这个 prior 上做 fine-grained action 生成。类似 DiffusionDet 的思路。

### 5.5 关于 COLOSSEUM 的 distractor 失败模式
BridgeVLA 在 Distractor 上输给 RVT-2（51.8 vs 60.8）。这是个有趣的 failure mode。VLM 的 grounding 能力让它对 "visually similar" 的 distractor 也产生高 heatmap 响应。RVT-2 没有 language grounding，纯几何匹配，反而不容易被语义 distractor 干扰。

这暗示：**VLM grounding 是双刃剑** — 在 semantic generalization 上是 asset，在 visual distractor 上是 liability。

---

## 6. 局限与 Future Work

1. **Fixed view**: Place Cups 失败说明 fixed orthographic view 在 occlusion 下不行。Dynamic view selection 或者更多 view 是出路。
2. **L4 Long-horizon**: 跟所有 VLA 一样，0%。需要 LLM task planner 层。
3. **Category generalization 弱**: Paper 分析是 pre-training image 是 third-person view，robot data 是 projection image，distribution mismatch。这个 gap 需要 more diverse pretraining data 解决。
4. **Action expressiveness**: 当前 heatmap 只能表 unimodal，不能处理 ambiguous task。
5. **Rotation 用 Euler + 72 bins**: 离散化太粗，对精细 rotation task 可能不够。Quaternion 连续表达 + diffusion 可能更好。

---

## 7. 我的整体评价

这篇 paper 的 contribution 主要是工程性的 — 把几个 known trick (orthographic projection from RVT, heatmap prediction from keypoint detection, convex upsampling from RAFT, coarse-to-fine from RVT-2, PaliGemma backbone) 组合到一个 coherent 的 framework 里。但组合的方式有清晰的 design principle (alignment 三重奏)，而且 ablation 非常有说服力。

最强的 claim 是 **3 trajectories → 95.4% real robot success**。如果 reproducible，这是 VLA 领域 sample efficiency 的 SOTA by a large margin。

最有教学意义的 ablation 是 "加 3D position encoding 反而更差"。这对未来 VLA 设计有指导价值：**对预训练 backbone，respect its input distribution**。

---

## References

- Paper: https://bridgevla.github.io/
- PaliGemma: https://arxiv.org/abs/2407.07726
- RVT: https://arxiv.org/abs/2306.13096
- RVT-2: https://arxiv.org/abs/2406.08475
- RAFT (convex upsampling): https://arxiv.org/abs/2003.12039
- RoboPoint (pretraining data): https://arxiv.org/abs/2406.10721
- RLBench: https://arxiv.org/abs/1909.12271
- COLOSSEUM: https://arxiv.org/abs/2402.08191
- GemBench: https://arxiv.org/abs/2410.01345
- SpatialVLA: https://arxiv.org/abs/2501.15830
- π₀: https://arxiv.org/abs/2410.24164
- PerAct: https://arxiv.org/abs/2209.05451
- Act3D: https://arxiv.org/abs/2304.01537
- 3D Diffuser Actor: https://arxiv.org/abs/2402.14824
