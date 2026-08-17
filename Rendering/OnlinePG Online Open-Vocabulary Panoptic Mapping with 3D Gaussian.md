---
source_pdf: OnlinePG Online Open-Vocabulary Panoptic Mapping with 3D Gaussian.pdf
paper_sha256: 21bd806e3895b4ccb8c4b99afdfb852212b634564c1e2cbe8711db1f13cbf8ef
processed_at: '2026-08-05T23:57:34-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OnlinePG 用人话说一遍

## 1. 这玩意儿在干啥

想象你是个机器人，戴着一副 RGB-D 眼睛在房间溜达。你想知道"这房间哪有沙发""那个红色的东西是啥""有几把椅子"。这事听起来简单，实际上坑巨多。

坑在哪？你想用 CLIP / LSeg 这种 VLM 来识别每帧画面里的东西。VLM 给你两个输出：一个 mask（告诉哪些像素属于哪个物体），一个 feature vector（512维的语言向量，代表"这是啥"）。

问题来了——VLM 每帧独立预测，它根本不知道第 1 帧的"椅子A"和第 10 帧的"椅子A"是同一把椅子。它可能这帧叫 mask ID 7，下帧叫 mask ID 12，再下帧又把椅子切成两半变成 ID 3 和 ID 5。更烦的是，遮挡会让一把椅子在 5 个 keyframe 里被切成 7 个 fragments。

**Offline 方法**（PanoGS [58]、OpenGaussian [50]、LangSplat [33]）解决这事很暴力：我把所有帧都先拿到，然后让 3D Gaussian Splatting 慢慢做 contrastive learning，让属于同一物体的 Gaussian primitives 在 feature space 里聚到一起。跑几小时 OK，效果不错。

**Online 方法**呢？你边走边看，前面还没看清后面就来新的了，没时间搞 gradient descent 那套慢功夫。之前的 online 方法（O2V-Mapping [42]、OnlineAnySeg [41]）要么干脆不做 instance（只做 semantic），要么做的 instance 质量很差，查"pillow"会把多个枕头混成一坨。

OnlinePG 的卖点：**在线、实时、还能做 open-vocabulary panoptic（既能分 stuff 类别又能分 thing 实例）**。

参考链接：
- PanoGS (CVPR 2025): https://arxiv.org/abs/2505.14646
- OnlineAnySeg (CVPR 2025): https://arxiv.org/abs/2411.14022
- O2V-Mapping (ECCV 2024): https://arxiv.org/abs/2408.03523

---

## 2. 核心思路：Local 先洗干净再交给 Global

作者的 insight 其实很朴素，跟传统 SLAM 的 keyframe local-BA + global pose graph optimization 是一个味儿：

> 别一帧一帧往 global map 灌噪声。先用一个 sliding window（12 个 keyframe）攒一攒，在这 12 帧里把碎片化的 segment 用 graph clustering 合并成完整的 instance，然后再注册到 global map 上。

为啥要这样？因为单帧 VLM mask 太脏了，但 12 帧合起来投票就靠谱多了。这个"局部先去噪再全局融合"的范式在 SLAM 里已经用烂了，OnlinePG 把它搬到 semantic scene understanding 上，配合 3DGS 的几何优势，效果就出来了。

### Sliding window 的节奏

参考论文 Fig.7：
- Window size = 12 keyframes
- 每 7 个 keyframe 触发一次 clustering + fusion
- Keyframe 采样：每 20 帧取 1 帧

所以每秒 30fps 的视频，每秒 1.5 个 keyframe，大概每 5 秒做一次 local clustering + global fusion。10-18 FPS 这个数字就是这么来的（Tab.4, Tab.5）。

---

## 3. 拆开看每个模块

### 3.1 两个并行的 3D 表征

这是这论文我最喜欢的设计点。他们没有试图把所有信息都塞到 3D Gaussian 里，而是搞了两套并行表征：

**表征 A：3D Gaussian Splatting（负责 geometry + appearance）**

$$\mathcal{G}_i = \{\mu_i, \Sigma_i, \sigma_i, c_i\}$$

- $\mu_i \in \mathbb{R}^3$：中心位置
- $\Sigma_i \in \mathbb{R}^{3\times3}$：协方差，分解成 rotation quaternion × scale vector
- $\sigma_i$：opacity
- $c_i$：RGB color（论文没提 SH，估计直接用 RGB 够了）

Loss 就是标准 3DGS 那套：
$$\mathcal{L} = \alpha \cdot \mathcal{L}_c + (1-\alpha) \cdot \mathcal{L}_d, \quad \alpha = 0.9$$

$\mathcal{L}_c$ 是 RGB L1 + D-SSIM，$\mathcal{L}_d$ 是 depth L1。每次新 keyframe 进来，随机抽 5 个历史帧做 20 iter，避免 catastrophic forgetting。

**表征 B：Sparse Voxel Grid（负责 semantics + instance）**

每个 occupied voxel 存 4 个字段：
$$\{\mathcal{F}, \mathcal{C}, \mathcal{T}, \mathcal{K}\}$$

| 字段 | 维度 | 是啥 |
|---|---|---|
| $\mathcal{F}$ | $\mathbb{R}^{512}$ | LSeg/CLIP language feature |
| $\mathcal{C}$ | $\mathbb{R}$ | feature confidence |
| $\mathcal{T}$ | int | instance ID（离散） |
| $\mathcal{K}$ | $\mathbb{R}$ | instance weight（累积投票数） |

Voxel size = 3cm。这个 3cm 是 ablation (Fig.5 右) 试出来的最优值：5cm 性能开始掉，10cm 直接崩。因为 voxel 太粗，instance 边界处的两个物体会被混到同一个 voxel 里，污染语义。

**为啥要两套表征？**

直觉上，3D Gaussian 是 anisotropic continuous 的，适合"软"的属性（color、geometry）通过 gradient descent 优化。但 instance label 是 discrete 的、需要 voting 的、不可导的——硬塞到 Gaussian 上就要搞 contrastive learning，慢且不稳定。

Voxel grid 反过来，做 majority voting 是 $O(1)$ 的 hash lookup，做 Bayesian update 是简单加权平均，工程上极快。3cm 在室内场景大概是 100-150 voxel per axis，用 sparse hash map 完全能 hold 住。

这是典型的 "right tool for the right job" 工程哲学。

### 3.2 Local 窗口内：把碎片合并成 instance

#### 初始化 3D segment

对 sliding window 里的第 $i$ 个 keyframe：
1. 跑 LSeg 拿到 2D feature map $f_i \in \mathbb{R}^{H \times W \times 512}$
2. 跑 EntitySeg 拿到 2D instance mask $m_i \in \mathbb{R}^{H \times W}$（每个像素一个 mask ID）
3. 用 depth unproject 到 3D，初始化 Gaussian primitives
4. 同 mask ID 的 primitives 归为同一个 3D segment $S_i$

$$\mathcal{S} = \{S_1, ..., S_n\}, \quad n = \sum_{i \in \mathcal{W}} |m_i|$$

12 个 keyframe 里可能有 200+ segments，但真实 instance 数大概 30-50。所以接下来要做 clustering。

#### Multi-cue graph clustering

构图：节点是 segment，边是 affinity。每条边带三个 cue：

**Cue 1: Geometry overlap $\mathcal{O}_{ij}$（公式 4）**

$$\mathcal{O}(S_i, S_j) = \frac{1}{2} \cdot \left(\frac{|S_i \cap S_j|}{\text{Cont.}(S_i, S_j)} + \frac{|S_i \cap S_j|}{\text{Cont.}(S_j, S_i)}\right)$$

变量解释：
- $|S_i \cap S_j|$：两个 segment 占据的 voxel 交集数（用 3cm voxel 离散化后做 hash 交集，避免遍历 Gaussian primitive）
- $\text{Cont.}(S_i, S_j)$：把 $S_j$ 投回 $S_i$ 的视角时，$S_j$ 中 visible voxel 被 $S_i$ 包含的比例（处理 occlusion）
- 整体是 symmetrized IoU + visibility 修正

直觉：两个 segment 在 3D 空间里占同样位置，大概率是同一个东西被切成两半。

**Cue 2: Semantic cosine similarity $\mathcal{X}_{ij}$**

先对每个 segment 做 mask-pooling 拿 segment-level feature：
$$z_i = \Phi(\{f(u,v) : m(u,v) = i\})$$
$\Phi(\cdot)$ 是 average pooling。

然后算 cosine similarity：
$$\mathcal{X}(S_i, S_j) = \frac{z_i \cdot z_j}{\|z_i\| \cdot \|z_j\|}$$

LSeg feature 是 L2-normalized 训练的，cosine sim 等价于内积。

直觉：两把椅子虽然在不同帧被切了不同 mask，但 LSeg feature 都接近"chair"的 embedding。

**Cue 3: View consensus $\mathcal{V}_{ij}$（公式 5，来自 MaskClustering [53]）**

$$\mathcal{V}(S_i, S_j) = \frac{N_{\text{supp}}(S_i, S_j)}{N_{\text{vis}}(S_i, S_j)}$$

- $N_{\text{vis}}$：两 segment 同时被 keyframe 看到的次数
- $N_{\text{supp}}$：在共视帧里两 segment 互相 support（同一区域都被预测到）的次数

直觉：如果两个 segment 在 12 个 keyframe 里都"同时出现同时消失"，它们一定绑在一起。这个 cue 对 thin / flat 物体（窗帘、画框、地毯边缘）特别有用，因为这些东西 geometry overlap $\mathcal{O}$ 很低，但 view consensus 很高。

#### Merge 判决（公式 6）

$$\Delta_{ij} = \left((\mathcal{O}_{ij} + \mathcal{X}_{ij}) > \lambda_1\right) \cup \left(\mathcal{V}_{ij} > \lambda_2\right)$$

- $\lambda_1 = 1.5$：要求 geometry + semantic 都很强（$\mathcal{O}$ 和 $\mathcal{X}$ 都在 $[0,1]$，相加要 > 1.5 等于两者都得接近 1）
- $\lambda_2 = 0.8$：view consensus 单独够强也行（OR 逻辑）
- 找 connected components，每个 component 就是一个 instance

Ablation (Fig.5 左) 显示这三个 cue 一个都不能少：
- 只用 $\mathcal{O}$：PRQ 平均 ~25
- $\mathcal{O} + \mathcal{X}$：~30
- $\mathcal{O} + \mathcal{X} + \mathcal{V}$：~38

加 $\mathcal{V}$ 只多花 40ms，涨 8 个 PRQ，性价比爆表。

#### Local voxel grid 填充

Clustering 完，每个 voxel $v$ 被某个 instance $\mathcal{T}_i$ 占据：
$$\mathcal{T}_l^t(v) = \text{ID}_i, \quad \mathcal{K}_l^t(v) = N_i$$

$N_i$ 是该 instance 在 clustering 中合并的 segment 数，相当于"这个 instance 被看到了几次"，后面做 voting 用。

Language feature 用 confidence 加权平均：
$$\mathcal{F}_l^t(v) = \frac{1}{\mathcal{C}_l^t(v)} \sum_{i \in \mathcal{W}} c_i(u,v) \cdot f_i(u,v)$$
$$\mathcal{C}_l^t(v) = \sum_{i \in \mathcal{W}} c_i(u,v)$$

$c_i$ 是 VLM 给的 confidence map（低 confidence 帧不污染 pool）。

### 3.3 Local → Global：双向二分图匹配

这是最巧妙的设计。

#### 为啥不能单向匹配？

Local map 有新探索区域，global map 有历史区域，containment ratio 天然不对称：

- Local 视角看 Global：$|\mathcal{T}_l \cap \mathcal{T}_g| / \text{Cont}(\mathcal{T}_l, \mathcal{T}_g)$ — local 是新看到的，可能 global 里啥都没有，ratio 接近 0
- Global 视角看 Local：$|\mathcal{T}_g \cap \mathcal{T}_l| / \text{Cont}(\mathcal{T}_g, \mathcal{T}_l)$ — global 已经看全了，local 是其子集，ratio 接近 1

单向 Hungarian 用任一方向都会误匹配。Local 里的新 instance 找不到对应，会被强行匹配到最近的 global instance。

#### Forward + Backward 双向验证

Forward matrix（公式 10）：
$$\mathcal{M}_{lg} = \underbrace{\frac{z_l \cdot z_g}{\|z_l\| \cdot \|z_g\|}}_{\text{semantic cosine sim}} + \underbrace{\frac{|\mathcal{T}_l \cap \mathcal{T}_g|}{\text{Cont.}(\mathcal{T}_l, \mathcal{T}_g)}}_{\text{geometry containment}}$$

Backward matrix：第二项换成 $|\mathcal{T}_g \cap \mathcal{T}_l| / \text{Cont.}(\mathcal{T}_g, \mathcal{T}_l)$。

然后两边都跑 Hungarian，取交集（公式 11）：
$$\mathcal{A} = \text{Hung.}(\mathcal{M}_{lg}) \cap \text{Hung.}(\mathcal{M}_{gl})^T$$

意思：必须 local 选 global 同时 global 也选 local，这对匹配才算数。类似 mutual nearest neighbor，但 Hungarian 保证 globally optimal。

Ablation (Tab.2) 数据说话：

| 设置 | PRQ(T) | PRQ(S) |
|---|---|---|
| #1 NN Match（贪心最近邻） | 24.67 | 22.98 |
| #2 Forward only | 35.83 | 38.40 |
| #3 Backward only | 33.71 | 42.72 |
| **#4 Bidirectional** | **37.97** | **41.81** |

双向 vs 单向 forward：PRQ(T) +2.14, PRQ(S) +3.41

Backward 对 PRQ(S) 提升尤其大，因为 stuff 类（wall, floor）容易在新帧里"重新出现"，backward 验证要求 global 也"看到" local，避免了 wall 被错切成两块。

#### Global map 更新

匹配上的 instance：feature 用加权平均（公式 12）：
$$\mathcal{F}_g^t(v) = \frac{\mathcal{C}_l^t(v) \cdot \mathcal{F}_l^t(v) + \mathcal{C}_g^{t-1}(v) \cdot \mathcal{F}_g^{t-1}(v)}{\mathcal{C}_g^t(v)}$$
$$\mathcal{C}_g^t(v) = \mathcal{C}_l^t(v) + \mathcal{C}_g^{t-1}(v)$$

这就是 incremental EMA，等价于 Bayesian update with Gaussian likelihood。

Instance label 是离散的，按投票规则更新（公式 13-15）：

| 情况 | $\mathcal{T}_g^t$ | $\mathcal{K}_g^t$ | 直觉 |
|---|---|---|---|
| Matched | 保持 global | $\mathcal{K}_g^{t-1} + \mathcal{K}_l^t$ | 同意，权重累加 |
| Unmatched, $\mathcal{K}_l^t \le \mathcal{K}_g^{t-1}$ | 保持 global | $\mathcal{K}_g^{t-1} - \mathcal{K}_l^t$ | 反对，但 global 占优 |
| Unmatched, $\mathcal{K}_l^t > \mathcal{K}_g^{t-1}$ | 用 local 替换 | $\mathcal{K}_l^t - \mathcal{K}_g^{t-1}$ | local 翻盘 |

这套路来自 SemanticFusion [25] 和 PanopticFusion [28] 的 Bayesian voting，但 OnlinePG 的投票单元是 local clustered instance（多帧聚合），不是单帧 mask，噪声低很多。

---

## 4. 实验结果说了啥

### 4.1 主表 (Tab.1) 的关键信息

ScanNetV2：

| 方法 | Online? | mIoU | mAcc | PRQ(T) | PRQ(S) |
|---|---|---|---|---|---|
| PanoGS (offline SOTA) | ✗ | 50.72 | 70.20 | 33.84 | 36.22 |
| OpenScene (offline) | ✗ | 47.63 | 69.74 | 43.53* | 40.43* |
| O2V-Mapping (online) | ✓ | 33.74 | 55.52 | - | - |
| OnlineAnySeg (online) | ✓ | 31.28 | 52.20 | 35.98 | 26.27 |
| **OnlinePG (Ours)** | ✓ | **48.48** | **66.01** | **37.97** | **41.81** |

几个值得玩味的点：

1. **PRQ(T) 37.97 居然超过了 offline PanoGS 的 33.84**。这说明 graph clustering + bidirectional matching 在 instance 一致性上比 contrastive learning 更靠谱，因为组合优化直接收敛到全局最优，contrastive learning 经常陷局部最优。

2. **PRQ(S) 41.81 大幅超过 online baseline OnlineAnySeg 的 26.27**（+15.54）。Stuff 类（wall、floor）靠 voxel grid 的 majority voting，比 instance-level feature fusion 稳得多。

3. **mIoU 48.48 vs OnlineAnySeg 31.28**，提升 +17.2。主要来自 voxel grid 细粒度保留 512 维 language feature，而 OnlineAnySeg 把 feature pool 到 instance 级别，丢失了细粒度。

4. 跟 offline SOTA PanoGS 还差 2.24 mIoU。原因：PanoGS 用 GT point cloud + 全程 multi-view supervision，OnlinePG 是 streaming 重建，几何精度天生吃亏。

### 4.2 FPS 和 runtime

从 Tab.4 / Tab.5 看：
- Replica 上 12-18 FPS
- ScanNetV2 上 13-19 FPS
- 比 O2V-Mapping (NeRF-based, 3 FPS) 快一个数量级
- 跟 OnlineAnySeg (10-26 FPS) 同档

但 **VLM 推理时间没算进去**！LSeg + EntitySeg 在 RTX 4090 上每帧大概 100-300ms，端到端实际 FPS 可能 3-5。这是这类方法共同的"作弊"之处。

Runtime breakdown (Fig.8, 论文 supplementary)：
- Keyframe preprocessing + segments init: 150-300ms
- 3DGS optimization (5 kf × 20 iter): 410ms
- Segment clustering: 350ms
- **Local-to-global fusion: 1400ms**（瓶颈）

Fusion 慢是因为 bidirectional Hungarian 是 $O(n^3)$，室内 scene 有 50-100 instances，$n^3 \sim 10^5-10^6$，加上 voxel grid 遍历，1.4s 合理。要进一步加速可以用 approximate Hungarian 或 Sinkhorn。

### 4.3 Ablation 的几个关键 takeaway

**Feature grid 分辨率 (Fig.5 右)**：3cm 是甜点，5cm 开始掉，10cm 崩。Instance 边界精度直接依赖 voxel 分辨率。

**Segment clustering 必要性 (Tab.3)**：

| 设置 | mIoU | PRQ(T) | PRQ(S) |
|---|---|---|---|
| 不做 clustering 直接融合 | 48.48 | 32.25 | 30.68 |
| 不用 feature grid F | 30.40 | 26.71 | 24.92 |
| **Full** | **48.48** | **37.97** | **41.81** |

不做 clustering 的话 PRQ(S) 从 41.81 掉到 30.68，因为噪声 segment 直接投票会把 wall 切成 100 块。不用 feature grid 的话 mIoU 从 48.48 掉到 30.40，因为 instance-level pooling 丢失细粒度。

---

## 5. 我的直觉判断

### 5.1 这工作为啥有意思

它把"3D open-vocab panoptic"这个看似 end-to-end 的任务，拆成了三个 stage 各自最优的子问题：

1. **Local denoising**：graph clustering 是组合优化，在小窗口内比 contrastive learning 收敛快得多
2. **Local-to-global alignment**：bidirectional Hungarian 是经典匹配理论，对不对称场景 robust
3. **Feature storage**：voxel grid 解耦 semantics 和 geometry，离散 voting 和连续 averaging 各得其所

这是工程美感。不是 end-to-end learning，是 hybrid system（neural geometry + discrete semantic）。在 online / streaming / 少数据场景下，hybrid 完胜 end-to-end。

### 5.2 它的局限

论文自承两点：
1. **Dynamic objects 不行**：3DGS + voxel grid 都假设 static scene，遇到人会糊。要解决得用 4D Gaussian 或 dynamic layer decomposition。
2. **依赖 depth + pose**：未来方向是 SLAM3R [20] / VGGT [46] / DUSt3R [47] 这种 feed-forward pose-free 方法。

我补充几点：

3. **VLM inference 是隐藏成本**：没算进 FPS，真实部署可能 3-5 FPS。未来要 active perception——只在不确定区域 query VLM。
4. **Voxel 3cm 在 outdoor 会爆显存**：sparse hash grid 是必须的，但 3cm 对街道场景 voxel 数指数膨胀。要做 outdoor 得 hierarchical voxel（粗 voxel 先 voting，细 voxel 再 refine）。
5. **Bidirectional Hungarian 在 instance > 200 时会卡**：1.4s 是室内 50-100 instances 的数据，机场、商场场景可能 5-10s。要 approximate Hungarian 或 Sinkhorn。

### 5.3 跟其他工作的关系

| 工作 | 类别 | 关系 |
|---|---|---|
| PanoGS [58] | Offline SOTA | 同作者组，OnlinePG 是其 online 版本 |
| MaskClustering [53] | Offline 3D instance seg | View consensus cue $\mathcal{V}$ 借鉴自此 |
| SemanticFusion [25] | Classical semantic SLAM | Voxel voting 规则 (公式 13-15) 沿用此思路 |
| LERF [12] / LangSplat [33] | Open-vocab NeRF/3DGS | 没有 instance 概念，只能做 semantic |
| OpenGaussian [50] | Point-level open-vocab | KNN retrieval 在 primitive 上，OnlinePG 改到 voxel 上 O(1) 查询 |
| O2V-Mapping [42] | Online open-vocab | NeRF-based，3 FPS，无 instance |
| OnlineAnySeg [41] | Online zero-shot 3D seg | mask merging 思路类似，但缺 language feature grid |

### 5.4 如果我要在自己的项目里用

我会这样走：

1. 先跑 PanoGS 代码（https://github.com/Ghosts-Group/PanoGS 如果开源），验证 offline 上限
2. 把 contrastive learning 模块替换成 OnlinePG 的 graph clustering + voxel voting
3. 先在 Replica 8 个 scene 上跑通，对比 Tab.1 数据
4. 再迁移到真实 RGB-D stream（Azure Kinect / RealSense），加 SLAM frontend（SplaTAM [23] 或 GS-SLAM）
5. VLM 推理用 TensorRT 加速 LSeg，EntitySeg 换成 SAM 2 [35] 试试
6. 如果要 dynamic，加一层 4D Gaussian + temporal voxel grid

---

## 6. 给 Karpathy 的一句话总结

> OnlinePG 干的事就是把 offline 3D open-vocab panoptic 的 contrastive learning 替换成了 sliding-window graph clustering + bidirectional Hungarian matching + voxel Bayesian voting 这套组合优化，把"学习问题"降维成"匹配+投票问题"，速度从小时级压到秒级，instance 一致性还反超了 offline baseline。

工程哲学上跟古典 SLAM 一脉相承，把 end-to-end 难题拆成几个可解的 sub-problem 各个击破。在当前 3DGS-based embodied perception 这条路线上是个很 clean 的 baseline，值得后续工作往上加 dynamic / pose-free / active VLM query。

---

## 参考链接汇总

- OnlinePG 原文（推测 arXiv 链接）：https://arxiv.org/abs/OnlinePG
- PanoGS: https://arxiv.org/abs/2505.14646
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- LSeg: https://arxiv.org/abs/2201.07046
- EntitySeg: https://arxiv.org/abs/2305.16400
- CLIP: https://arxiv.org/abs/2103.00020
- MaskClustering: https://arxiv.org/abs/2401.04052
- SemanticFusion: https://arxiv.org/abs/1604.01322
- PanopticFusion: https://arxiv.org/abs/2104.03381
- LERF: https://lerf.io/
- LangSplat: https://arxiv.org/abs/2312.16084
- OpenGaussian: https://arxiv.org/abs/2406.11879
- O2V-Mapping: https://arxiv.org/abs/2408.03523
- OnlineAnySeg: https://arxiv.org/abs/2411.14022
- SplaTAM: https://arxiv.org/abs/2312.13260
- SAM 2: https://arxiv.org/abs/2408.00714
- ScanNetV2: http://www.scan-net.org/
- Replica: https://github.com/facebookresearch/Replica-Dataset
- 浙大 Guofeng Zhang 主页: http://www.guofengzhang.me/

---

# OnlinePG 深度解析：Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting

## 1. 高层 Motivation 与问题定位

OnlinePG 处理的是 embodied intelligence 场景下的核心感知问题：在 RGB-D 视频流上同时做 **几何重建** + **open-vocabulary panoptic 理解**。从直觉上看，这是把 2D VLM 的语义先验蒸馏到 3D 表示中的工程，难点在于 VLM 的 2D 输出在多视角间天然 inconsistent（over-segmentation / under-segmentation / mask ID drift），offline 方法（如 PanoGS [58]）可以通过对比学习慢慢收敛，online 系统则必须在一个 sliding window 内快速完成"消歧 + 融合"。

核心 insight：**Local-to-global paradigm**。先在局部窗口内把多视角噪声 segment 合并成 consistent 3D instances，再增量注册到 global map。这种两阶段设计与传统 SLAM 的 keyframe-based local BA + global pose graph optimization 在哲学上同源。

参考链接：
- 原论文（如已发表在 CVPR/ICCV 等）：https://arxiv.org/abs/OnlinePG
- 3D Gaussian Splatting 原作：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- PanoGS (CVPR 2025)：https://openaccess.thecvf.com/content/CVPR2025/papers/Zhai_PanoGS
- LSeg：https://arxiv.org/abs/2201.07046
- EntitySeg：https://arxiv.org/abs/2305.16400

---

## 2. 整体架构与数据流

```
RGB-D stream → Keyframe sampling (every 20 frames)
   → per-keyframe: LSeg feature map f_i (H×W×512) + EntitySeg mask m_i (H×W)
   → per-keyframe: depth-guided Gaussian unprojection, group primitives by mask ID → 3D segment S_i
   → sliding window W (size=12, step=4~7)：
        (a) Multi-cue Clustering Graph: vertices=segments, edges={O_ij, X_ij, V_ij}
        (b) Connected components → local consistent instances I
        (c) Voxelize → local spatial attribute grids {F_l, C_l, T_l, K_l}
   → Local-to-Global fusion：
        (d) Bidirectional bipartite matching (Hungarian) on local vs global instances
        (e) Update global grids {F_g, C_g, T_g, K_g} via weighted averaging
   → Query phase：CLIP text encoding ↔ 3D feature field retrieval
```

**Sliding window 设计 (Fig.7)**：window size=12, step=4 表示每次有 8 个 keyframe 是 reuse 的，4 个 keyframe 是 new，目的是让 segment clustering 有足够的多视角共视来支撑 view consensus cue，同时避免重复计算。Clustering 和 fusion 每 7 个 keyframe 触发一次（可推断 step=7 用于融合，step=4 用于示意）。

---

## 3. Scene Representation：双轨表征

### 3.1 3D Gaussian Splatting 部分

每个 Gaussian primitive 参数化：
$$\mathcal{G}_i := \{\mu_i, \Sigma_i, \sigma_i, c_i\}$$

| 符号 | 维度 | 含义 |
|---|---|---|
| $\mu_i$ | $\mathbb{R}^3$ | Gaussian 中心位置 |
| $\Sigma_i$ | $\mathbb{R}^{3\times3}$ | 协方差矩阵，通常分解为 $\Sigma = R \cdot S \cdot S^T \cdot R^T$（旋转 $R$ 用 quaternion，缩放 $S$ 用 3-vector） |
| $\sigma_i$ | $\mathbb{R}$ | opacity，过 sigmoid 后用于 alpha compositing |
| $c_i$ | $\mathbb{R}^3$ | 球谐系数或直接 RGB |

渲染 loss：
$$\mathcal{L} = \alpha \cdot \mathcal{L}_c + (1 - \alpha) \cdot \mathcal{L}_d$$

$\mathcal{L}_c$ 是 RGB L1 + D-SSIM，$\mathcal{L}_d$ 是 depth L1，$\alpha = 0.9$ 偏向 appearance。每次新 keyframe 进入时，随机抽 5 个历史 keyframe 做 20 次 iteration，这种 "lazy global batch" 的策略让 3DGS 不会遗忘已收敛区域。

### 3.2 Grid-based Spatial Attributes（核心创新之一）

3DGS 的 anisotropic Gaussian 对 instance label 这种 **离散、不可导、需要多数投票** 的信息并不友好。OnlinePG 设计了一个并行表征——sparse voxel grid，每个 occupied voxel 存储：
$$\{\mathcal{F}, \mathcal{C}, \mathcal{T}, \mathcal{K}\}$$

| 字段 | 维度 | 含义 |
|---|---|---|
| $\mathcal{F}$ | $\mathbb{R}^{D_f=512}$ | LSeg/CLIP language feature |
| $\mathcal{C}$ | $\mathbb{R}$ | feature confidence (来自 VLM 的概率) |
| $\mathcal{T}$ | $\mathbb{R}$ (实际是 int) | panoptic instance ID |
| $\mathcal{K}$ | $\mathbb{R}$ | instance weight（累积观测次数，类似 SemanticFusion [25] 的 Bayesian 更新） |

Voxel size = 3cm，这是 ablation (Fig.5 右) 显示的关键参数——5cm 以上 PRQ 急剧下降，因为 instance 边界处会被 mixing 污染。3cm 在 ScanNet 室内场景（房间尺度 3-5m）大约是 100-150 voxel / dimension，sparse hash grid 才能在显存上 hold 住。

直觉上，**3DGS 负责 geometry + appearance，voxel grid 负责 semantics + instance**，两条数据通路解耦，让 open-vocab 的 512 维 feature 不必绑定到每个 Gaussian 上（节省显存，且便于 voting）。

---

## 4. Local Consistent Map Construction

### 4.1 3D Gaussian Segments Initialization

对第 $i$ 个 keyframe：
- $f_i \in \mathbb{R}^{H\times W\times D_f}$：LSeg feature map
- $m_i \in \mathbb{R}^{H\times W}$：EntitySeg instance mask（每个像素一个 mask ID）
- 用 depth $d_i$ 把每个像素 unproject 到 3D，初始化 Gaussian primitive
- 同一 mask ID 的 primitives 归为一个 segment

$$\mathcal{S} := \{S_1, \cdots, S_n \mid S_i := \{G_j\}_{j=1}^{|m_i|}\}, \quad n = \sum_{i \in \mathcal{W}} |m_i|$$

**关键观察**：2D segmentation 在多视角间没有 ID consistency，所以一个真实物体在窗口 12 个 keyframe 里可能被切成 $n$ 个不同的 segment。Clustering 的任务就是把这 $n$ 个 segment 合并回 $\le n$ 个 consistent instance。

### 4.2 Multi-cue Clustering Graph

构图 $G = (\{S_i\}, \{\mathcal{E}_{ij}\})$，每个 edge 是一个 affinity vector $(\mathcal{O}_{ij}, \mathcal{X}_{ij}, \mathcal{V}_{ij})$。

#### (1) Geometry cue $\mathcal{O}_{ij}$（公式 4）

$$\mathcal{O}(S_i, S_j) = \frac{1}{2} \cdot \left(\frac{|S_i \cap S_j|}{\text{Cont.}(S_i, S_j)} + \frac{|S_i \cap S_j|}{\text{Cont.}(S_j, S_i)}\right)$$

- $|S_i \cap S_j|$：两个 segment 占据的 voxel 交集数（用 3cm voxel 离散化后做 hash 交集，避免遍历所有 Gaussian primitive）
- $\text{Cont.}(S_i, S_j)$：把 $S_j$ 投回 $S_i$ 的视角时，$S_j$ 中 visible voxel 被 $S_i$ 包含的比例（处理 occlusion）
- 形式上是 **Jaccard-like 的对称 IoU**，但加入了 visibility 修正

#### (2) Semantic cue $\mathcal{X}_{ij}$

先对每个 segment 做 mask-pooling：
$$z_i = \Phi(\{f(u,v) : m(u,v) = i\})$$
$\Phi(\cdot)$ 是 average pooling，得到 segment-level language feature。

$$\mathcal{X}(S_i, S_j) = \frac{z_i \cdot z_j}{\|z_i\| \cdot \|z_j\|}$$

这是 cosine similarity，对 LSeg 这种已经 normalize 过的 feature 等价于内积。

#### (3) View consensus cue $\mathcal{V}_{ij}$（公式 5，来自 MaskClustering [53]）

$$\mathcal{V}(S_i, S_j) = \frac{N_{\text{supp}}(S_i, S_j)}{N_{\text{vis}}(S_i, S_j)}$$

- $N_{\text{vis}}$：两 segment 同时被看到的关键帧数
- $N_{\text{supp}}$：在这些共视帧里，两 segment 互相 support（即在同一帧的同一区域都被预测到）的次数
- 直觉：如果两个 segment 在很多视角下都"同时出现同时消失"，它们大概率是同一物体的不同 mask 切片

#### (4) Merge decision（公式 6）

$$\Delta_{ij} = \left((\mathcal{O}_{ij} + \mathcal{X}_{ij}) > \lambda_1\right) \cup \left(\mathcal{V}_{ij} > \lambda_2\right)$$

- $\lambda_1 = 1.5$（注意 $\mathcal{O}$ 和 $\mathcal{X}$ 都在 $[0,1]$，相加阈值 1.5 要求两者都很高，是 AND-like 逻辑）
- $\lambda_2 = 0.8$（view consensus 单独够强也可以合并，处理 thin/flat 物体）
- 两者取并集（OR），最终通过 connected components 输出 instance

### 4.3 Local Spatial Attribute Grids

Voxelize 后，对每个 voxel $v$：
$$\mathcal{T}_l^t(v) = \text{ID}_i, \quad \mathcal{K}_l^t(v) = N_i$$

$N_i$ 是该 instance 在 clustering 中参与的 segment 数，相当于一种 **观测频率 confidence**。

Language feature 通过 confidence 加权平均：
$$\mathcal{F}_l^t(v) = \frac{1}{\mathcal{C}_l^t(v)} \sum_{i \in \mathcal{W}} c_i(u,v) \cdot f_i(u,v)$$
$$\mathcal{C}_l^t(v) = \sum_{i \in \mathcal{W}} c_i(u,v)$$

这里 $c_i(u,v)$ 是 VLM 给的 confidence map（LSeg 的 softmax entropy 反转或类似），低 confidence 帧不会污染 feature pool。

---

## 5. Local-to-Global Map Fusion：Bidirectional Bipartite Matching

这是论文最精巧的设计。Local map 有新探索区域，global map 有历史区域，**containment ratio 天然不对称**：$|T_l \cap T_g| / \text{Cont.}(T_l, T_g) \neq |T_g \cap T_l| / \text{Cont.}(T_g, T_l)$。

### 5.1 Forward matrix $\mathcal{M}_{lg}$（公式 10）

$$\mathcal{M}_{lg} = \frac{z_l \cdot z_g}{\|z_l\| \cdot \|z_g\|} + \frac{|\mathcal{T}_l \cap \mathcal{T}_g|}{\text{Cont.}(\mathcal{T}_l, \mathcal{T}_g)}$$

- 第一项：semantic cosine similarity（instance-level，从 $\mathcal{F}_l$ 查询到的 segment-level feature）
- 第二项：local-to-global 的几何包含率

### 5.2 Backward matrix $\mathcal{M}_{gl}$

把第二项换成 $|\mathcal{T}_g \cap \mathcal{T}_l| / \text{Cont.}(\mathcal{T}_g, \mathcal{T}_l)$，从 global 视角看 local。

### 5.3 Hungarian 双向验证（公式 11）

$$\mathcal{A} = \text{Hung.}(\mathcal{M}_{lg}) \cap \text{Hung.}(\mathcal{M}_{gl})^T$$

**关键直觉**：单向 Hungarian 容易在新探索区域误匹配（local instance 找不到对应，会强行匹配到最近的 global instance）。双向要求互相选对方，类似 mutual nearest neighbor 的思想，但 Hungarian 保证了 matching 是 globally optimal 而非贪心。Ablation (Tab.2)：
- NN match: PRQ(T)=24.67, PRQ(S)=22.98
- Forward only: 35.83 / 38.40
- Backward only: 33.71 / 42.72
- **Bidirectional: 37.97 / 41.81**

Backward 对 PRQ(S) 提升尤其明显（+4.3 over forward），因为 stuff 区域（wall, floor）容易在新帧中"重新出现"，backward 验证要求 global 也"看到" local，避免了 wall 被错误切分。

### 5.4 Global update rules

Feature 是连续的，可以直接加权平均（公式 12）：
$$\mathcal{F}_g^t(v) = \frac{\mathcal{C}_l^t(v) \cdot \mathcal{F}_l^t(v) + \mathcal{C}_g^{t-1}(v) \cdot \mathcal{F}_g^{t-1}(v)}{\mathcal{C}_g^t(v)}$$
$$\mathcal{C}_g^t(v) = \mathcal{C}_l^t(v) + \mathcal{C}_g^{t-1}(v)$$

这是 **incremental EMA**，等价于 Bayesian update with Gaussian likelihood。

Instance label 是离散的，分三种情况（公式 13-15）：

| 情况 | $\mathcal{T}_g^t(v)$ | $\mathcal{K}_g^t(v)$ | 含义 |
|---|---|---|---|
| Matched | $\mathcal{T}_g^{t-1}(v)$ | $\mathcal{K}_g^{t-1}(v) + \mathcal{K}_l^t(v)$ | 累加 weight，label 不变 |
| Unmatched, $\mathcal{K}_l^t \le \mathcal{K}_g^{t-1}$ | $\mathcal{T}_g^{t-1}(v)$ | $\mathcal{K}_g^{t-1}(v) - \mathcal{K}_l^t(v)$ | 投反对票，但 global 仍占优 |
| Unmatched, $\mathcal{K}_l^t > \mathcal{K}_g^{t-1}$ | $\mathcal{T}_l^t(v)$ | $\mathcal{K}_l^t(v) - \mathcal{K}_g^{t-1}(v)$ | Local 翻盘，label 替换 |

这套规则源自 SemanticFusion [25] / PanopticFusion [28] 的 Bayesian voting 思想，但用 instance weight $\mathcal{K}$ 代替了概率，且 local map 在送入前已经 clustering 过，投票单元从单帧变成了多帧聚合的 instance，noise 大幅降低。

---

## 6. 实验数据分析

### 6.1 Main results (Tab.1)

**ScanNetV2**：
| Method | Online? | Pano? | mIoU | mAcc | PRQ(T) | PRQ(S) |
|---|---|---|---|---|---|---|
| PanoGS (offline SOTA) | ✗ | ✓ | 50.72 | 70.20 | 33.84 | 36.22 |
| OpenScene (offline) | ✗ | ✗ | 47.63 | 69.74 | 43.53* | 40.43* |
| O2V-Mapping | ✓ | ✗ | 33.74 | 55.52 | - | - |
| OnlineAnySeg | ✓ | ✓ | 31.28 | 52.20 | 35.98 | 26.27 |
| **OnlinePG (Ours)** | ✓ | ✓ | **48.48** | **66.01** | **37.97** | **41.81** |

**关键 insight**：
1. PRQ(T) 37.97 超过 offline PanoGS 的 33.84，证明 local-to-global + bidirectional matching 的 instance 一致性优于 offline 的 contrastive learning
2. PRQ(S) 41.81 远超 online baseline OnlineAnySeg 的 26.27，因为 stuff 区域靠 voxel grid 的 majority voting 而非 instance-level feature
3. mIoU 48.48 vs online SOTA 31.28，提升 +17.2，主要来自 feature grid 的细粒度语言特征保留

### 6.2 Per-scene FPS (Tab.5)

ScanNetV2 各场景 FPS 范围 13.33-18.83，复杂场景 0645 (大空间) 仅 14.08，简单场景 0140 18.48。O2V-Mapping 因为是 NeRF-based，FPS 只有 3.x；OnlineAnySeg 是 15-26。OnlinePG 在保持 panoptic 能力的同时 FPS 与 OnlineAnySeg 接近。

### 6.3 Runtime breakdown (Fig.8)

- Keyframe preprocessing + segments init: ~150-300ms
- 3DGS optimization (5 kf × 20 iter): 410ms
- Segment clustering: 350ms (每 7 kf 触发一次)
- Local-to-global fusion: 1400ms (瓶颈)

Fusion 慢是因为 bidirectional Hungarian 是 $O(n^3)$，$n$ 是 instance 数，室内场景 instance ~50-100，$n^3 \sim 10^5-10^6$，加上 voxel grid 遍历，1.4s 是合理代价。

### 6.4 Ablation: feature grid resolution (Fig.5 右)

3cm 是 PRQ 峰值，5cm 开始下降，10cm 急剧退化。这印证了 3cm voxel 是 instance 边界保持精度的下限，再大就出现跨 instance 混叠。

### 6.5 Ablation: 多 cue 的必要性 (Fig.5 左)

| Cue | 时间 | PRQ avg |
|---|---|---|
| O only | ~50ms | ~25 |
| O+X | ~70ms | ~30 |
| O+X+V | ~90ms | ~38 |

多 cue 只多花 40ms 换 8-18 PRQ 提升，性价比极高。$\mathcal{V}$ (view consensus) 处理 thin object (窗帘、画框) 尤为关键，这些 object 的 geometry overlap $\mathcal{O}$ 很低。

---

## 7. 与相关工作的直觉对比

### 7.1 vs PanoGS [58] (offline SOTA)
PanoGS 用 SAM mask + contrastive feature learning 训练 3DGS 的 instance embedding，需要全程 gradient descent，无法 online。OnlinePG 用 graph clustering 替代 contrastive learning，把"学习"问题降为"组合优化"问题，但代价是无法处理需要 long-range 上下文的复杂 case（如多视角下才看得清的物体）。

### 7.2 vs LERF [12] / LangSplat [33]
两者都是把 CLIP feature 蒸馏到 NeRF/3DGS 的 density field，没有 instance 概念，只能做 semantic segmentation，无法区分两个相同语义的不同 instance（如两把椅子）。OnlinePG 通过 voxel-level instance label 显式建模 instance identity。

### 7.3 vs MaskClustering [53]
MaskClustering 是 offline 的纯 mask-based 3D instance segmentation，OnlinePG 借用了其 view consensus cue 思想，但加入了 language feature $\mathcal{X}$ 和 voxel-level geometry $\mathcal{O}$，且在 sliding window 内 incremental 执行。

### 7.4 vs OpenGaussian [50]
OpenGaussian 是 point-level open-vocab，通过 KNN 在 Gaussian primitives 之间做 feature retrieval，但同样 offline。OnlinePG 把 feature 挂到 voxel grid 而非 primitive 上，retrieval 从 $O(\log N)$ KNN 变成 $O(1)$ voxel query，更适合 online。

---

## 8. 局限与未来方向

论文自承两点：
1. **Dynamic objects**：3DGS + voxel grid 都假设 static scene，遇到人会失效。未来可能需要 4D Gaussian 或 dynamic layer decomposition。
2. **Dependence on depth + pose**：依赖 RGB-D + 已知 pose，未来方向是 SLAM3R [20] / VGGT [46] / DUSt3R [47] 这类 feed-forward pose-free 方法，把 pose estimation 也纳入 end-to-end pipeline。

补充观察：
- **VLM inference 没算在 FPS 里**：LSeg + EntitySeg 在 RTX 4090 上每帧大概 100-300ms，实际端到端 FPS 可能是 3-5，离真正实时（30 FPS）还有距离。
- **Voxel size 3cm 在 outdoor / large-scale 场景会爆显存**：sparse hash grid + incremental allocation 是必须的，但 3cm 分辨率对街道级场景 voxel 数会指数膨胀。
- **Bidirectional Hungarian 在 instance 数很多时（>200）会卡**：可以用 approximate Hungarian 或 Sinkhorn 加速。

---

## 9. 代码 / 资源链接

- 项目页（如有）：浙大 CAD&CG Guofeng Zhang 组主页 http://www.guofengzhang.me/
- 3DGS 原始代码：https://github.com/graphdeco-inria/gaussian-splatting
- PanoGS (CVPR 2025)：https://github.com/Ghosts-Group/PanoGS （如有）
- OnlineAnySeg：https://github.com/YijieTang/OnlineAnySeg
- LSeg：https://github.com/isl-org/lang-seg
- EntitySeg：https://github.com/sysu-yanglab/Entity-Segmentation
- MaskClustering (view consensus 来源)：https://github.com/YixingLiao/MaskClustering
- ScanNetV2：http://www.scan-net.org/
- Replica：https://github.com/facebookresearch/Replica-Dataset

---

## 10. Intuition 总结

OnlinePG 的工程美感在于把一个看似 end-to-end 的"3D open-vocab panoptic"任务，拆解成了三个 stage 各自最优的子问题：

1. **Local denoising**：graph clustering 是组合优化，比 contrastive learning 在小窗口内收敛快
2. **Local-to-global alignment**：bidirectional Hungarian 是经典匹配理论，比 greedy NN 在不对称场景下 robust
3. **Feature storage**：voxel grid 解耦了 semantics 和 geometry，让离散 voting 和连续 averaging 各得其所

这种"分而治之 + 显式表征"的工程哲学，正是当前 3DGS-based embodied perception 系统的主线：**end-to-end learning 适合 static / offline / 大数据场景，而 hybrid system（geometry neural + semantic discrete）适合 online / streaming / 少数据场景**。OnlinePG 给这条路线提供了一个非常 clean 的 baseline。

进一步思考的方向：
- 把 sliding window 机制换成 learned keyframe selection（类似 SLAM 中的 information-theoretic keyframe selection）
- voxel grid 替换成 sparse octree + hierarchical feature（粗 voxel 先投票，细 voxel 再 refine）
- VLM inference 通过 LLM-guided 的 active perception 减少（只在不确定区域 query VLM）
- Dynamic extension 用 4D Gaussian + temporal voxel grid

如果要在自己的 robot 项目里复现这套系统，建议从 PanoGS 代码起步（已经验证 offline 性能上限），把 contrastive learning 模块替换成 OnlinePG 的 graph clustering + voxel voting，先在 Replica 上跑通，再迁移到真实 RGB-D stream。
