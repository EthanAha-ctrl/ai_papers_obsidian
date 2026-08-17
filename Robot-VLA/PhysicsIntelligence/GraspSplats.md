---
source_pdf: GraspSplats.pdf
paper_sha256: d62956de29c09841b8d6166768ae9a4cb83f2ad6b916891332f5dfe78427d119
processed_at: '2026-08-04T22:21:45-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 GraspSplats

Andrej，我把刚才那堆公式翻译成大白话。想象我们在白板前聊天，我会这么讲：

---

## 这篇 paper 到底在干嘛

你想让机器人听一句话就去抓东西，比如"抓那个马克杯的把手"。这件事听起来简单，对机器人来说其实巨难，因为它要解决三个问题：

1. **"马克杯"是哪个？** 桌上可能有好几个杯子
2. **"把手"在马克杯的哪里？** 不是整个杯子都能抓，得抓把手
3. **如果杯子被人挪了怎么办？** 机器人总不能说"请别动，我重新建模一下"

过去两年大家用 NeRF 来做这件事（把 CLIP feature 蒸进神经辐射场），但 NeRF 这东西就像一个黑盒子 MLP —— 你问它"把手在哪"，它得沿射线采样几百个点做体积渲染才能回答，慢得要命；你想把杯子挪一下，它完全不知道怎么改，因为几何信息都隐式编码在权重里。

GraspSplats 说：**换成 3D Gaussian Splatting 就全解决了**。

---

## 为什么 Gaussian 比 NeRF 适合 robotics

Gaussian Splatting 的本质是一堆**显式的小椭球**（explicit ellipsoids），每个椭球有：
- 中心坐标 $\mu$（在哪）
- 协方差 $\Sigma$（多大、多扁）
- 颜色 $c$（什么颜色）
- 透明度 $\alpha$（多透）
- **feature 向量 $f$**（GraspSplats 新加的，装 CLIP 语义）

它既能像 NeRF 一样**可渲染**（alpha-blending splatting 出来还是一张图），又能像点云一样**直接操作**（你要抓哪个椭球？要平移哪几个椭球？直接改坐标就行）。

NeRF 是"我把场景记住了一个 MLP 里，你问我再算"；3DGS 是"我把场景摊成几百万个小球摆桌上，你要拿就拿"。

对 robotics 来说，"摊桌上"这个特性太关键了 —— 抓取采样需要几何，物体移动需要 transform，这两件事 implicit representation 都做不动，explicit representation 天然就支持。

---

## 三个核心 trick

### Trick 1：用 depth 直接初始化，不用 Colmap

传统 3DGS 要先用 Colmap 从 RGB 图算稀疏点云初始化，这个过程要十几秒。GraspSplats 说：robotics 场景我们有 RGBD 相机啊！**直接把 depth 图反投影成点，每个点就是一个 Gaussian 的初始中心**。

结果：预处理从 11.6 秒降到 0.7 秒，训练迭代数从 10000 降到 3000。这是 robotics 的免费午餐，因为 depth 本来就白送。

### Trick 2：分两层提 feature，规避 CLIP 的"乱激活"问题

CLIP 有个毛病：你问"mug handle"，它会把所有 mug 和所有 handle 都激活，因为它本质上是个 bag-of-words 模型。LERF 的解法是多尺度 query 一千次，巨慢。

GraspSplats 的解法是**分层**：
- **Object 层**：先用 MobileSAMV2 找出所有物体的 mask（"这一坨是一个物体"），再用 MaskCLIP 提特征，做 masked average pooling 得到 object-level feature
- **Part 层**：在每个 object 的 bbox 内 crop 出小 patch 单独跑 MaskCLIP，得到 part-level feature（"handle"只在 mug 的 bbox 内激活）

这样既快（SAM 提供了 object prior，不用满图乱 query），又准（part feature 只在 object 内部算，不会被别的 object 的 handle 干扰）。

### Trick 3：直接在 Gaussian 上做 grasp sampling，不用 GraspNet 那套重活

LERF-TOGO 为了用 GraspNet（一个端到端抓取网络），要把 NeRF 渲染成 100 个视角的点云再喂进去，一次抓取采样要 10 秒。

GraspSplats 说：Gaussian 本身就带法向量（从协方差矩阵算出来），直接用经典的 GPG（geometric grasp proposer，2017 年的老方法）在 Gaussian 上采样就行。每个 Gaussian 的 surface normal 是解析的，邻域内做个二阶矩矩阵 $M(p) = \sum \hat{n}\hat{n}^T$ 就能建局部坐标系，然后 grid search 找夹爪能闭合的位置。

结果：0.5 秒搞定，比 GraspNet-100 快 20 倍，成功率还高 4.7 个点。因为 Gaussian 的 soft aggregation 比 point cloud 的硬 bin 更 robust。

---

## 动态场景是怎么做的

这是最 cool 的部分，也是 NeRF 完全做不到的。

场景：机器人扫了一遍桌子，建好了 Gaussian 表示。然后人走过来把杯子挪走了。机器人怎么办？

GraspSplats 的流程：
1. 用语言查询"mug"找到对应的 Gaussian 子集
2. 把这些 Gaussian 渲染成 2D mask 投到相机视角
3. 在 mask 内离散出一些 2D 点，扔给 **CoTracker**（一个点跟踪器）持续跟踪
4. 多视角的 2D 跟踪点用 depth 反投影成 3D 对应点
5. DBSCAN 去掉噪声对应
6. **Kabsch 算法**闭式求出 SE(3) 刚体变换 $R, t$
7. 直接把 $R, t$ 应用到那几个 Gaussian 上（改 center 和 rotation），不重训

整个过程实时完成，几十毫秒。NeRF 要做同样的事只能重训 10 分钟。

如果有遮挡区域（比如杯子原来压住的地方没扫到），可以可选地做 partial fine-tuning，只重训那块区域。

---

## 数据说话

| | 训练时间 | 抓取延迟 | 静态成功率 | 动态成功率 |
|---|---|---|---|---|
| LERF-TOGO (NeRF) | 10 min | 9.9s | 65.1% | 做不了 |
| F3RM (NeRF) | 3 min | 1.6s | 72.1% | 做不了 |
| TrackAnything (2D) | 0 | 3.1s | 41.9% | 45% |
| **GraspSplats** | **60s** | **1.3s** | **81.4%** | **74.2%** |

训练快 10 倍，抓取快 7 倍，成功率高 16 个点，还能做动态场景。这在 robotics 里是碾压级别的提升。

---

## 一句话总结

**3D Gaussian Splatting 是 robotics 的正确 representation，因为它第一次让"可渲染优化"和"可直接几何操作"共存于一个数据结构里。** NeRF 只能做前者，point cloud 只能做后者，3DGS 两个都能做。GraspSplats 就是把这个 insight 落地成了一个完整的 zero-shot manipulation pipeline。

---

## 顺手提一句的相关联想

- **World model 方向**：如果你要把这个推广到 video prediction / world model，explicit representation 比 implicit 好太多了 —— 你可以想象每个 Gaussian 是一个"object token"，physics 直接作用在 token 上（move, collide, deform），渲染就是 splatting 一下。这比用 latent token 表示 world 的 Dreamer 系列更 grounded。
- **Sim-to-real**：3DGS 天然兼容 differentiable simulation，因为每个 primitive 有显式坐标。你可以想象把 GraspSplats 的场景扔进 Isaac Gym，每个 Gaussian 是一个 sphere collider。
- **Scaling**：当前场景是 tabletop，如果扩到 whole-room 或 whole-building，Gaussian 数量会爆炸。可能需要 hierarchical Gaussian（类似 Nanite 的 LOD 思路），或者 sparse voxel + Gaussian hybrid。
- **Feature 的开放性**：当前用 CLIP，换成 SigLIP 或 DINOv2 应该直接能换，因为 Feature Splatting 的 framework 对 feature 维度不敏感。DINOv2 的 dense feature 可能比 CLIP 更适合 part-level（因为 CLIP 训练目标是 image-text 对齐，dense localization 本来就弱）。

---

如果你下一步想动手，我觉得最值得做的是**把 DINOv2 换进去做 part-level**，然后**加 articulation estimation**（handle + body 两个 rigid body + 一个 revolute joint）。这两个 extension 都不改变核心 architecture，但能把 capability 推一大步。

---

# GraspSplats: 3D Feature Splatting 用于高效零样本抓取的深度解析

Andrej 你好，这篇来自 UC San Diego 的工作我觉得很对你的胃口 —— 它本质上是在问："NeRF-based representation (F3RM, LERF-TOGO) 真的是 robotic manipulation 的正确选择吗？"。作者给出的答案是：3D Gaussian Splatting (3DGS) 的 **explicit + editable + renderable** 特性更像 robotics 应该拥抱的 representation。下面我会把整个 pipeline 拆开来讲清楚 intuition、公式含义和工程细节。

项目主页：https://graspsplats.github.io

---

## 1. Motivation：为什么 NeRF 在 robotics 上"卡住了"

过去一年（2023-2024）robotics 社区流行用 **language-embedded radiance fields** 做零样本抓取，代表作品：

- **LERF** (Kerr et al., ICCV 2023)：https://www.lerf.io — 把 CLIP feature 蒸馏进 NeRF
- **LERF-TOGO** (Rashid et al., CoRL 2023)：https://to-go.github.io — 加上 conditional CLIP + DINO regularization 做任务导向抓取
- **F3RM** (Shen et al., CoRL 2023)：https://f3rm.github.io — 蒸馏 feature field + few-shot 模仿学习

这些方法的问题可以归纳为三点：

| 问题 | 体现 |
|---|---|
| **训练慢** | LERF-TOGO 训练 ~10min，F3RM ~3min，且 scene 一动就要重训 |
| **Implicit 不可编辑** | NeRF 用 MLP 隐式表示 occupancy/density，物体被外力移动后无法直接 transform，必须重新优化 |
| **渲染采样慢** | NeRF 体渲染需要沿 ray 采样点，做 CLIP feature query 需要逐体素 volumetric rendering，毫秒级难做到 |

而 2D point-cloud 方法（TrackAnything + GraspNet）虽然快，但 **没有 rendering-based optimization**，无法多视角融合语义，part-level（如 "抓马克杯的把手"）几乎做不到。

GraspSplats 的核心 insight：**3D Gaussian 是 explicit primitive，既能像 NeRF 一样做 differentiable rendering 把 2D feature 蒸馏到 3D，又能像 point cloud 一样直接做几何操作（采样抓取、刚体变换）。**

---

## 2. 核心 rendering equation 解读

整个方法建立在 **alpha-blending splatting** 上（参考 Feature Splatting, Qiu et al. 2024, https://arxiv.org/abs/2404.01223）。GraspSplats 同时渲染 depth、color、feature：

$$
\{\hat{D}, \hat{F}, \hat{C}\} = \sum_{i \in N} \{d_i, f_i, c_i\} \cdot \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j) \quad (1)
$$

**变量含义**：

- $\hat{D}, \hat{F}, \hat{C}$: 渲染出的 depth map、feature map、color image
- $i \in N$: 一条 ray 上 Gaussian primitive 的索引，**按 $d_i$ 升序排列**（从近到远）
- $d_i$: 第 $i$ 个 Gaussian 的中心到 camera origin 的距离
- $f_i \in \mathbb{R}^{16}$: 该 Gaussian 的 latent feature vector（注意：原始 3DGS 不存 feature，这里扩展了）
- $c_i \in \mathbb{R}^3$: 该 Gaussian 的球谐系数对应的颜色
- $\alpha_i$: 该 Gaussian 在该像素处的 opacity，由 $\alpha_i = o_i \cdot \exp(-\frac{1}{2}\Sigma_i^{-1}(p-\mu_i)^2)$ 决定（3DGS 标准）
- $\prod_{j=1}^{i-1}(1-\alpha_j)$: **transmittance**，即"光在前 $i-1$ 个 Gaussian 穿过后的剩余能量"

**Intuition**：与 NeRF 的 $C = \sum T_i(1-\exp(-\sigma_i\delta_i))c_i$ 完全等价，但 3DGS 是"投影 + 排序"而非"采样 + 积分"，所以快一个量级，而且每个 primitive 都是有具体坐标和协方差的 ellipsoid，可以直接拿来算几何。

**Isotropic feature assumption**：作者沿用 Feature Splatting 的做法，假设 feature 维度的 Gaussian 是 isotropic（球对称），即 feature 的传播不依赖于 spatial 的 anisotropic covariance。这个假设合理 —— CLIP/DINO feature 的语义应该是 view-invariant 的。

---

## 3. 三大核心模块

### 3.1 高效构造 feature-enhanced 3D Gaussians

作者发现现有 feature-GS 方法（LangSplat, Feature 3DGS, Feature Splatting）的瓶颈在 **预处理**：
1. **Reference feature 计算太贵**（LERF 要做 multi-scale 1000+ queries）
2. **SfM-based sparse Colmap 初始化**导致 Gaussian 过稀，需要大量 densification 才能拟合

GraspSplats 用两个 trick 同时解决：

#### (a) Hierarchical Reference Feature via MobileSAMV2 + MaskCLIP

- **Object-level feature**：用 MobileSAMV2 (https://github.com/ChaoningZhang/MobileSAM) 给出 class-agnostic bbox $\mathbf{D}_{obj}$ 和 mask 集合 $\{M\}$，再用 MaskCLIP (https://arxiv.org/abs/2203.08374) 提取整图粗 CLIP feature $\mathbf{F}_C \in \mathbb{R}^{H' \times W' \times C}$，然后做 **Masked Average Pooling (MAP)**:

$$
w_i = \text{MAP}(\mathbf{M}, \mathbf{F}_C) = \frac{\sum_{i \in \mathbf{F}_C} \mathbf{M}(i) \cdot \frac{\mathbf{F}_C(i)}{||\mathbf{F}_C(i)||}}{\sum_{i \in \mathbf{F}_C} \mathbf{M}(i)} \quad (3)
$$

  - $i$: 像素坐标
  - $\mathbf{M}(i) \in \{0,1\}$: mask 二值化后的归属
  - $\mathbf{F}_C(i)/||\mathbf{F}_C(i)||$: 先做 L2 normalize，避免某个高 feature norm 主导
  - 输出 $w_i$ 直接赋给 mask 区域内所有像素作为 object-level reference

- **Part-level feature**：从 $\mathbf{D}_{obj}$ crop 出 patch，wrap 到 $(224, 224)$，送 MaskCLIP 拿 $(28, 28, 768)$ 的 feature map，再 interpolate 回原 bbox 尺寸。重复 patch 多实例取平均得 $\mathbf{F}_{part}$。

- **Decoder MLP**：渲染时引入一个 shallow MLP，输出两分支：
  $$\hat{\mathbf{F}}_{obj}, \hat{\mathbf{F}}_{part} := \text{MLP}(\hat{\mathbf{F}})$$
  分别用 cosine loss 监督。Joint loss = $\mathcal{L}_{obj} + \lambda \mathcal{L}_{part}$，$\lambda = 2.0$ —— 偏向 part-level，因为 part 才是抓取真正关心的。

**Intuition**：通过 SAM 提供 object prior，把 N（patch 数量）压到远小于 LERF 的 uniform query 数量；同时 part-level feature 用 crop+inference，专门激活 "handle" "rim" 这些 sub-region，规避 CLIP 的 "bag-of-words" 问题（"mug handle" 在 LERF 中会激活所有 handle 而不限于 mug 的 handle）。

#### (b) Geometry Regularization via Depth

传统 3DGS 用 Colmap sparse points 初始化，densification 慢。GraspSplats **直接用 depth 图反投影每个像素作为 Gaussian center**，同时用 depth 做 L2 supervision：

$$\mathcal{L}_{depth} = ||\hat{D} - D_{gt}||_2^2$$

Table 5 给出明确证据：

| Method | Process Time (s) | Train Iteration |
|---|---|---|
| Colmap-S (原始 3DGS) | 11.6 | 10,000 |
| Colmap-D | 623.0 | 3,000 |
| **GraspSplats (depth init)** | **0.7** | **3,000** |

Colmap-D 反而比 Colmap-S 更慢（因为 dense 重建贵），但 GraspSplats 的 depth init 比 Colmap-S **快 17 倍预处理**且收敛 iter 数减少 70%。这是 robotics 场景的天然优势 —— RGBD 相机就是用来吃 depth 的。

---

### 3.2 Part-level 查询与抓取采样

#### Open-vocabulary Object Query

每个 Gaussian $j$ 有 16 维 latent feature $\mathbf{F}_{latent,j}$，经 Decoder 得到 768 维 CLIP feature $\mathbf{F}_{CLIP,j}$。给定 language set $L = \{L_0^-, L_1^-, ..., L_n^-, L^+\}$（一个 positive query 加多个 negative query，默认 negative 含 "object", "things"）：

$$
\mathbf{F}_{text,i} = \text{CLIP.encode}(L_i)
$$

$$
\text{sim}(\mathbf{F}_{CLIP,j}, \mathbf{F}_{text,i}) = \frac{\mathbf{F}_{CLIP,j} \cdot \mathbf{F}_{text,i}}{||\mathbf{F}_{CLIP,j}|| ||\mathbf{F}_{text,i}||}
$$

$$
\mathbf{S}_j = \text{softmax}(\{\text{sim}(\mathbf{F}_{CLIP,j}, \mathbf{F}_{text,i})\}_{i=0}^n)
$$

最后选 $\mathbf{S}_j[n] > \tau = 0.6$ 的 Gaussian，再用 DBSCAN 聚类去 outlier。

**Conditional Part Query**：LERF-TOGO 要先 render image → 再 voxel 查询 → 再 mask。GraspSplats 因为是 explicit primitive，**直接在 segmented object 的 Gaussian 子集上再做一次上面的 query 流程即可**，毫秒级完成。

#### Grasp Sampling (扩展 GPG)

这部分是把 GPG (Ten Pas et al., https://github.com/graspnetPy/graspnetAPI) 搬到 Gaussian primitive 上做。定义工作空间 $\mathcal{R}_{obj}$ 为 segmented part 的 Gaussian 包围盒扩张（半径 = Gaussian 最长轴 + 夹爪碰撞半径）。在 $\mathcal{R}_{obj}$ 内采样 $N$ 个点 $p$，在邻域 $R_p$ 内聚合 Gaussian 法向：

$$
M(p) = \sum_{g \in R_p} \hat{n}(g)\hat{n}(g)^T \quad (2)
$$

- $\hat{n}(g)$: Gaussian $g$ 的单位法向量（由 covariance 矩阵的最小特征向量决定）
- $M(p) \in \mathbb{R}^{3\times3}$: **二阶矩矩阵**（structure tensor），SVD 分解后 3 个特征向量分别对应 surface normal、二次方向、最小方向

构造参考坐标系 $F(p) = [v_3(p), v_2(p), v_1(p)]$，在 2D grid $G = Y \times \Phi$（深度平移 $y$ + 绕 z 轴旋转 $\phi$）搜索：

$$
F(h_{y,\phi}) = F(p) T_{x^*, y, \phi}
$$

其中 $x^*$ 是沿 -x 方向夹爪触到点云的最小距离。若闭合区域内 Gaussian 数 $N_{obj} > N_{th}$（说明夹到了而非擦边），加进候选集 $H$，最后用 geometry-aware scoring (Dex-Net 风格, Mahler et al. 2017) 排序。

**Intuition**：NeRF 方法（如 LERF-TOGO）为了用 GraspNet 要渲染 100 个视角的 point cloud（10.3s），而 GraspSplats 在 explicit primitive 上直接做 GPG，**0.5s** 搞定。Table 3 显示：GraspNet-100 (76.7%, 10.3s) vs GraspSplats (81.4%, 0.5s)，**速度 20×、成功率 +4.7pt**。说明 explicit representation 对几何抓取几乎"零开销"。

---

### 3.3 动态场景：实时跟踪与 partial fine-tuning

这是 NeRF 完全做不到的环节，也是本文最大亮点。

**Pipeline**：
1. 语言查询 → 选 Gaussian → 渲染 mask 到相机
2. 把 mask 离散成 2D 点 → 送 **CoTracker** (Karaev et al., https://co-tracker.github.io) 做 long-term tracking
3. 多视角 2D 对应 → depth 反投影成 3D 对应点
4. **DBSCAN** 过滤 outlier 对应
5. **Kabsch algorithm** (https://en.wikipedia.org/wiki/Kabsch_algorithm) 求 SE(3) rigid transform $T \in \text{SE}(3)$

Kabsch 求解最小化 $\sum_i ||R p_i + t - q_i||^2$，闭式解：
$$
H = \sum_i p_i q_i^T, \quad H = U\Sigma V^T, \quad R = V \text{diag}(1,1,\det(UV^T)) U^T
$$

把 $T$ 直接施加到 segmented Gaussian 上（更新 center $\mu$ 和 rotation quaternion），不重训。多 camera 情况下把所有 3D 对应点拼成一个超定方程组解 Kabsch。

**Partial fine-tuning**：被遮挡区域（如抽屉下方）在原始重建时未观测到，移动后可选用前后两次的 mask 做 partial retrain，远快于完整重建。

---

## 4. 实验结果与数据解读

### 4.1 主要定量结果（Table 1）

| Method | Train Latency | Grasp Latency | Static Succ. | Dynamic Succ. |
|---|---|---|---|---|
| TrackAnything (2D+GraspNet) | — | 3.1s | 41.9% | 45% |
| ConceptGraphs | ~30s | 0.7s | 51.1% | †(不支持) |
| LERF-TOGO | ~10min | 9.9s | 65.1% | † |
| F3RM* | ~3min | 1.6s | 72.1% | † |
| **GraspSplats** | **60s** | **1.3s** | **81.4%** | **74.2%** |

- 训练 60s vs LERF-TOGO 600s → **10× 加速**
- 抓取延迟 1.3s vs LERF-TOGO 9.9s → **7.6× 加速**
- Static +16.3pt vs LERF-TOGO
- Dynamic 唯一能跑的方法（74.2%）

### 4.2 拆分延迟（Table 2）

| Method | Segment | Grasp Sampling |
|---|---|---|
| LERF-TOGO | 5.1s | 4.8s |
| F3RM | 1.0s | 6.9s |
| GraspSplats | 0.8s | 0.5s |

LERF-TOGO 的 4.8s 几乎全花在 100-view GraspNet 推理上；GraspSplats 的 0.5s 来自 GPG on explicit primitives。

### 4.3 分解成功率（Table 6）

| Method | Object-level | Part-level |
|---|---|---|
| LERF-TOGO | 81.5% | 63.0% |
| F3RM* | 85.2% | 77.8% |
| **GraspSplats** | **96.3%** | **85.2%** |

Part-level 的优势 (+7.4pt vs F3RM*) 主要归功于 MobileSAMV2 的 mask prior + part-level supervision。

### 4.4 Segmentation IoU（Table 4）

LERF 39.0 vs GraspSplats 50.7（+11.7pt），hierarchical feature 的作用直接量化。

---

## 5. 失败模式（Section 4.5）

作者坦诚列出：
- **静态**：相似物体混淆；执行层碰撞（lift 时夹爪碰物体）—— 需要更好 motion planning，超出本工作 scope
- **动态**：单色/对称物体 long-term tracking 容易丢；快速旋转 + occlusion 导致 CoTracker 失效
- **假设**：Kabsch 假设 rigid transform，deformable object（dough, clay）未探索

---

## 6. Intuition 总结（这是我读完真正 internalize 的几点）

1. **Explicit vs Implicit 不是 theology，而是 robotics workflow 的实际需求** —— 抓取要直接拿几何，编辑要直接 transform，这两点 implicit MLP 都做不动。

2. **3DGS 的最大价值不是 "novel view synthesis 比 NeRF 快"，而是 "每个 primitive 都是 first-class object"** —— 你可以 query 它、transform 它、splatting 出来 still 可渲染。这是 NeRF 的 MLP 隐式表示天然无法提供的。

3. **Depth supervision + depth init 是 robotics 场景的免费午餐** —— RGBD 相机本来就有 depth，Colmap 完全是浪费。这个 insight 应该推广到所有 robotics+3DGS 工作。

4. **Hierarchical feature = object prior + part supervision** —— 用 SAM 提供 object 的概念边界，再用 CLIP 在每个 object 内部找 part，规避 CLIP 的 "bag-of-words" 弱点。这种 "先用 SAM 切再让 CLIP 学 part" 的范式比 LERF 的 multi-scale query 更高效且更准。

5. **CoTracker + Kabsch 是 cheap 的 dynamic extension** —— 不需要学习，不需要重训，纯几何就能在 explicit representation 上做刚体跟踪。如果想做 articulation，加 joint axes estimation 是自然 extension。

6. **GPG 在 Gaussian 上比 GraspNet 在 point cloud 上更好** —— 因为 Gaussian 带 covariance（surface normal 可解析），且邻域 $R_p$ 内的 aggregation 是 soft weighted（opacity-weighted），比硬点云 bin 更 robust。

---

## 7. 我会想 follow up 的方向

- **Articulated object**：当前 Kabsch 假设单刚体，handle + body 应分 two rigid bodies + 显式 joint axis
- **Deformable tracking**：CoTracker 给 dense correspondence，直接 drive Gaussian deformation field（参考 4DGS, Wu et al. https://arxiv.org/abs/2310.08528）
- **Re-sampling keypoints during task**：作者自己在 limitation 里提，长任务中点跟踪 drift，应该在 task execution 中重新采样
- **Gripper occlusion 下的 placement**：lift 时夹爪挡视线，placement 时不知物体 pose —— 应该用 proprioception + 夹爪内部 forward kinematics 反推
- **Feature field 的 multi-resolution**：当前 isotropic feature 假设可能在多尺度语义（fine-grained "rim" vs coarse "mug"）下损失信息，可以做 anisotropic feature covariance

---

## References

- 项目主页：https://graspsplats.github.io
- Feature Splatting (Qiu et al.): https://arxiv.org/abs/2404.01223
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- LERF: https://www.lerf.io
- LERF-TOGO: https://to-go.github.io
- F3RM: https://f3rm.github.io
- MobileSAMV2: https://github.com/ChaoningZhang/MobileSAM
- MaskCLIP: https://arxiv.org/abs/2203.08374
- CoTracker: https://co-tracker.github.io
- GraspNet-1Billion: https://graspnet.net
- ConceptGraphs: https://concept-graphs.github.io
- TrackAnything: https://github.com/gaomingqi/Track-Anything
- 4D Gaussian Splatting: https://arxiv.org/abs/2310.08528
- Kabsch algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm
- GaussianGrasper (concurrent): https://arxiv.org/abs/2403.09637

---

这篇工作的核心 message 我觉得是 robotics community 的一个范式信号：**3DGS 已经成为 robotic manipulation 的默认 representation，原因不是它做 NVS 更快，而是它让 "rendering-based optimization" 与 "explicit geometric operations" 第一次能同时存在于一个数据结构里。** NeRF 在 robotics 上的 implicit 死结被绕过去了。这对 future 的 sim-to-real、world model、active perception 都有直接的含义。
