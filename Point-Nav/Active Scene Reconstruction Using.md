---
source_pdf: Active Scene Reconstruction Using.pdf
paper_sha256: 78f20c3602fbf23de514912e5ca00e53366a987dc744ca69a7402de73c77515c
processed_at: '2026-07-18T01:22:12-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# ActiveGS：基于 Gaussian Splatting 的 Active Scene Reconstruction 深度解析

## 一、核心 problem framing

这篇 paper 来自 University of Bonn 的 Cyrill Stachniss 团队（也包含 Marija Popović），第一作者 Liren Jin 之前还做过 Neu-NBV 和 STAIR，是 active NeRF reconstruction 方向的连续工作。ActiveGS 可以看作是他们把 active reconstruction 从 NeRF 迁移到 Gaussian Splatting 的代表作。

**核心 problem**：给一个 mobile robot（UAV / ground robot）配一个 posed RGB-D 相机，在有限 mission time budget T 内，online 主动规划 camera viewpoint，把未知 scene 重建出 high-fidelity 的 3D 表示。这跟 passive reconstruction（固定 trajectory）和 pure exploration（只要覆盖，不管质量）都不一样——它需要在 **exploration**（覆盖未探索空间）和 **exploitation**（精细化观察 under-reconstructed 区域）之间做 trade-off。

**为什么用 GS 而不用 NeRF**：
- NeRF 的 volumetric rendering 需要 dense ray sampling，online incremental mapping 时每帧训练太慢；
- 之前 NeRF-based active 工作（ActiveNeRF、NeuAR、NARUTO）为了能 online，不得不压缩 model capacity，导致 representation 质量下降；
- GS 的 explicit primitive + differentiable rasterization 让 map update 非常快，且 explicit structure 方便做 viewpoint selection。

**两个核心 challenge**：
1. 没有 ground truth，如何评估 novel viewpoint 处的 reconstruction quality？
2. Gaussian primitive 只表示 occupied space，无法区分 unknown vs free space，而 exploration 和 path planning 又依赖 free/unknown 的概念。

作者的 solution 是 hybrid map：GS 负责高保真重建，voxel map 负责空间 occupancy 和 path planning。同时给每个 Gaussian primitive 加一个 confidence score $k_i$，作为 viewpoint utility 的指标。

GitHub repo: https://github.com/dmar-bonn/active-gs

---

## 二、System architecture 解析

Fig 2 给出的 pipeline 是一个 mapping-planning 交替迭代的 loop：

```
RGB-D + pose → ┬─→ Voxel map V (OctoMap-like, occupancy probability)
               └─→ GS map G (Gaussian surfel, incrementally trained)
                          │
                          ▼
              ┌───────────┴───────────┐
              │ confidence rendering   │
              │ unknown voxel detection │
              └───────────┬───────────┘
                          ▼
              View planner (ROI + random sampling)
                          │
                          ▼
              A* path → next best viewpoint p*
                          │
                          ▼
              robot move → 新的 RGB-D → 回到 mapping
```

**Hybrid map 的合理性 intuition**：
- GS map 的优势在于 photorealistic rendering 和 fine geometry，但它本质是 "biased sampling" —— 只在有观测的地方才长 Gaussian primitive，所以它天然无法回答 "这个 voxel 是否被探索过" 这个问题；
- Voxel map（OctoMap）的优势是 free/unknown/occupied 三态建模，是 exploration 的标准工具，但它太稀疏，没法做 high-fidelity rendering；
- 两者互补：voxel map 提供 topology（哪有 frontier、哪是 free space 可以飞行），GS map 提供 quality signal（哪的 surface 重建得不够好）。

---

## 三、关键技术细节

### 3.1 Gaussian surfel 参数化

作者用的是 2D Gaussian surfel（来自 Dai et al. 的 SIGGRAPH 2024 工作），不是原始 3D GS。每个 primitive $\mathbf{g}_i$ 参数化为：

$$g_i = (\mathbf{x}_i, \mathbf{q}_i, \mathbf{s}_i, \mathbf{c}_i, o_i, k_i)$$

变量含义：
- $\mathbf{x}_i \in \mathbb{R}^3$：primitive 中心位置
- $\mathbf{q}_i \in \mathbb{R}^4$：rotation quaternion
- $\mathbf{s}_i = [s_i^x, s_i^y] \in \mathbb{R}^2_+$：两个轴的 scale
- $\mathbf{c}_i \in [0,1]^3$：RGB color
- $o_i \in [0,1]$：opacity
- $k_i \in \mathbb{R}_+$：**confidence score**，本文新增

为什么 2D surfel 而不是 3D Gaussian？3D GS 在 surface alignment 上有 aliasing 问题，因为它用 ellipsoid volume 去 approximate surface。2D surfel 把 3D volume collapse 成 oriented disk，normal 可以直接从 rotation matrix 第三列取出来：

$$\mathbf{n}_i = \mathbf{R}(\mathbf{q}_i)_{:,3}$$

Covariance matrix 公式：

$$\mathbf{\Sigma}_i = \mathbf{R}(\mathbf{q}_i) \text{diag}((s_i^x)^2, (s_i^y)^2, 0) \mathbf{R}(\mathbf{q}_i)^\top$$

第三维 eigenvalue 为 0，意味着 surfel 在 tangent 平面之外没有 thickness —— 这是 2D disk 的本质。

### 3.2 Differentiable rasterization（Eq 2-3）

$$\mathbf{O}(\mathbf{u}) = \sum_{i=1}^{n} w_i, \quad \mathbf{M}(\mathbf{u}) = \sum_{i=1}^{n} w_i m_i$$

- $\mathbf{u}$：pixel coordinate
- $w_i$：primitive $\mathbf{g}_i$ 对 pixel $\mathbf{u}$ 的 rendering contribution
- $\mathbf{M}$ 可以是 $\mathbf{I}$ (color), $\mathbf{D}$ (depth), $\mathbf{N}$ (normal), $\mathbf{K}$ (confidence) 中的任意一种
- $m_i$ 是对应的 modality feature，比如 $c_i$ for color, $d_i$ for depth, $\mathbf{n}_i$ for normal, $k_i$ for confidence

$$w_i = T_i \alpha_i, \quad T_i = \prod_{j<i}(1-\alpha_j), \quad \alpha_i = \mathcal{N}(\mathbf{u}; \mathbf{u}_i, \mathbf{\Sigma}_i') o_i$$

- $T_i$：accumulated transmittance，前面所有 primitive 没挡住的剩余比例
- $\alpha_i$：当前 primitive 在 pixel $\mathbf{u}$ 处的有效 opacity
- $\mathbf{u}_i$、$\mathbf{\Sigma}_i'$：把 3D Gaussian 投影到 image plane 后的 2D 中心和 covariance（EWA splatting 的标准操作，参考 Zwicker 2001）

**关键 insight**：因为 rendering 是 linear combination，所以 confidence map $\mathbf{K}$ 可以像 color map 一样直接 rasterize 出来，无需额外计算。这跟 FisherRF 的做法形成对比——后者要算每个 candidate viewpoint 对 GS 参数的 Fisher information，需要 backward gradient，昂贵得多。

### 3.3 Densification mask（Eq 4）

$$\mathbf{B}(\mathbf{u}) = (\mathbf{O}(\mathbf{u}) < 0.5) \lor (\text{avg}(|\mathbf{I}(\mathbf{u}) - \mathbf{I}^\star(\mathbf{u})|) > 0.5) \lor (\mathbf{D}(\mathbf{u}) - \mathbf{D}^\star(\mathbf{u}) > \lambda \mathbf{D}^\star(\mathbf{u}))$$

- $\mathbf{I}^\star, \mathbf{D}^\star$：当前观测的 GT RGB 和 depth
- $\mathbf{I}, \mathbf{D}, \mathbf{O}$：从当前 camera viewpoint render 出来的 map
- $\lambda = 0.05$：depth sensing noise tolerance

三个 OR 条件，分别对应三种需要 densify 的情况：
1. **Opacity 低**：当前 GS map 在这里根本没几何体；
2. **Color error 大**：有几何体但颜色错了；
3. **Depth 突变**：新观测的 depth 比渲染的 depth 更近，说明前面挡了一个之前没建出来的物体（注意这里只检测 closer 的情况，因为远处的隐藏部分本来就该被前面挡住）。

新 spawn 的 primitive：position 用 unprojected pixel 的 point cloud，color 用 pixel color，normal 用 bilateral-filtered depth 的 central differencing 估计；scale = 1cm，opacity = 0.5，confidence = 0（因为还没被多个 viewpoint 观测过）。

### 3.4 Training loss（Eq 5）

$$\mathcal{L} = w_c \mathcal{L}_c + w_d \mathcal{L}_d + w_n \mathcal{L}_n$$

- $\mathcal{L}_c = L_1(\mathbf{I}, \hat{\mathbf{I}})$：photometric L1
- $\mathcal{L}_d = L_1(\mathbf{D}, \hat{\mathbf{D}})$：depth L1
- $\mathcal{L}_n = D_{cos}(\mathbf{N}, \tilde{\mathbf{N}}) + TV(\mathbf{N})$：normal loss
  - $D_{cos}$：rendered normal map 和从 rendered depth map 反算的 normal map 的 cosine distance
  - $TV$：total variation，鼓励 neighboring pixel 之间 normal 平滑
- 权重 $w_c = 1.0, w_d = 0.8, w_n = 0.1$

**重要细节**：每 mapping step 训练 10 iterations；每 iteration 用 3 recent frames + 5 random frames（避免 catastrophic forgetting，因为 online setting 下没有 full dataset）。训练只更新 $(\mathbf{x}_i, \mathbf{q}_i, \mathbf{s}_i, \mathbf{c}_i, o_i)$，**$k_i$ 不参与 gradient training**，它是 rule-based 更新的（见 3.5）。

每 5 个 mapping step 做一次 visibility check，把所有 history view 都看不到的 primitive 删掉——这是 online map compaction 的关键，避免 unbounded growth。Visibility 阈值 $w_i > 0.3$。

### 3.5 Confidence modelling（Eq 6-8）—— 本文核心

这是最有意思的部分。核心 idea 是：**一个 Gaussian primitive 只有被多个不同视角、从合适角度观察，才能被 well-optimized**。如果只被一个近距离正面观察，可能 surface 方向都估错了。

定义：viewpoint $p_j$ 到 primitive center $\mathbf{x}_i$ 的向量为 $\mathbf{d}_{ij} = \mathbf{x}_{p_j} - \mathbf{x}_i = d_{ij} \mathbf{v}_{ij}$。

$$k_i = \gamma_i \exp(\beta_i)$$

$$\gamma_i = \sum_{j \in \mathcal{S}(\mathbf{g}_i)} \left(1 - \frac{d_{ij}}{d_{\text{far}}}\right) \mathbf{n}_i \cdot \mathbf{v}_{ij}$$

$$\beta_i = 1 - \|\boldsymbol{\mu}_i\|, \quad \boldsymbol{\mu}_i = \frac{1}{|\mathcal{S}(\mathbf{g}_i)|} \sum_{j \in \mathcal{S}(\mathbf{g}_i)} \mathbf{v}_{ij}$$

逐项拆解：

- $\mathcal{S}(\mathbf{g}_i)$：所有观察过 $\mathbf{g}_i$ 的 viewpoint index 集合
- $d_{ij}$：viewpoint 到 primitive 的距离
- $\mathbf{v}_{ij}$：normalized view direction（从 viewpoint 指向 primitive）
- $d_{\text{far}}$：max depth sensing range（5.0 m）
- $\mathbf{n}_i$：primitive normal

**$\gamma_i$ 的直觉**：对每个观察 viewpoint，我们希望
- 距离近（$(1 - d_{ij}/d_{\text{far}})$ 大）—— 近距离深度噪声小，几何准；
- view direction 和 normal 对齐（$\mathbf{n}_i \cdot \mathbf{v}_{ij}$ 大）—— 正对 surface 看比斜视更准，斜视的 surfel 容易在 tangent 方向上滑动；
把这些加权 cosine similarity 加起来。

**$\beta_i$ 的直觉**：衡量 view direction 的 dispersion。$\boldsymbol{\mu}_i$ 是所有 view direction 的平均向量，如果所有 viewpoint 都从同一方向看，$|\boldsymbol{\mu}_i| \approx 1$，$\beta_i \approx 0$；如果 view direction 散布各方向，向量相消，$|\boldsymbol{\mu}_i| \approx 0$，$\beta_i \approx 1$。$\exp(\beta_i)$ 是 multiplier，所以 multi-view 观察的 primitive confidence 会更高。

**这个 formulation 的特点**：
- 不需要 backward gradient（vs FisherRF）；
- 是 feed-forward 渲染的（confidence map 直接 rasterize）；
- 跟 view planning 自然耦合 —— 低的 $k_i$ 区域自动成为下一步要观察的目标。

**潜在问题**（build your intuition 时考虑）：这个 confidence 只衡量 "observation diversity"，并不能直接衡量 reconstruction error。比如 surface 方向估错了，但多次从不同角度观察同样的错误几何，confidence 也会变高。不过实验结果显示这个 proxy 已经足够 work。

### 3.6 Viewpoint utility（Eq 9-10）

**Combined utility**：

$$U_{\text{view}}(\mathbf{p}_i^c) = \phi U_{\mathcal{V}}(\mathbf{p}_i^c) + U_{\mathcal{G}}(\mathbf{p}_i^c)$$

- $U_{\mathcal{V}}(\mathbf{p}_i^c) = N_u(\mathbf{p}_i^c) / |\mathcal{V}|$：candidate viewpoint $\mathbf{p}_i^c$ 能看到的 unexplored voxel 数量除以总 voxel 数
- $U_{\mathcal{G}}(\mathbf{p}_i^c) = -\text{mean}(\mathbf{K}_i)$：从 candidate viewpoint 渲染的 confidence map 的均值取负 —— mean confidence 越低，utility 越高（说明这个 viewpoint 能看到一些 under-reconstructed 区域）
- $\phi = 1000$：exploration weight

**为什么 $\phi = 1000$ 这么大？** $U_{\mathcal{V}}$ 量级是 $N_u / |\mathcal{V}|$，最大也就 1，而 $U_{\mathcal{G}}$ 量级是 confidence 均值，也在 0-1 量级。但实际 $N_u / |\mathcal{V}|$ 通常远小于 1（scene 很大时 unexplored voxel 占比可能就 0.001 量级），需要大 weight 才能在 mission 早期驱动 exploration；当 frontier 几乎消耗完，$U_{\mathcal{V}} \to 0$，自然 transition 到 exploitation 主导。这是个 implicit phase transition。

**Final selection（Eq 10）**：

$$\mathbf{p}^\star = \arg\max_{\mathbf{p}_i^c} \left( \frac{U_{\text{view}}(\mathbf{p}_i^c)}{\sum_i U_{\text{view}}(\mathbf{p}_i^c)} - \delta \frac{U_{\text{path}}(\mathbf{p}_i^c)}{\sum_i U_{\text{path}}(\mathbf{p}_i^c)} \right)$$

- 两个 utility 都归一化到 [0,1]，避免量纲问题
- $U_{\text{path}}$：从 current viewpoint 到 candidate 的 A* 路径长度
- $\delta = 0.5$：travel cost weight

**Viewpoint sampling strategy**：
- $N_{\text{total}} = 100$ candidate viewpoints
- $N_{\text{random}} = 70$：current viewpoint 周围 0.5m 内随机采样
- $N_{\text{ROI}} = 30$：从最近的 ROI voxel 开始往外采样

ROI 来源有两类（Fig 3）：
1. **Frontier voxels**（来自 Yamauchi 1997 的 frontier-based exploration 思想）：未知和已知的边界 voxel；normal 取 neighbor free voxel 方向的平均
2. **Low-confidence Gaussian 所在 voxel**：normal 取这些 low-confidence primitive 的 average normal —— 因为正对 normal 看 surfel 才能优化得最好

ROI-based sampling 在一个 cone 里：以 voxel center + normal 为基准，限制最小/最大采样距离和最大 angular deviation，确保采样出来的 viewpoint 大致正对这个 ROI。

### 3.7 View planning 中 visibility check 的技巧

传统的 voxel visibility check 要做 ray-casting，昂贵。本文的 trick：直接渲染 GS map 的 depth map $\mathbf{D}$，然后把 voxel center 投影到 image plane，比较 voxel 的 depth 和渲染 depth：

- 如果 voxel depth < rendered depth，则 voxel 是 visible 的（Gaussian 在它后面，没挡住）
- 反之 voxel 被 Gaussian 挡住了

这里复用 GS 的 rasterization，避免额外 ray-casting。**注意**：当 GS map 在某 pixel 处没有几何（opacity 低）时，rendered depth 可能是错的，所以这个 visibility 其实是 approximate 的，但实际效果 ok。

---

## 四、实验设置与结果分析

### 4.1 实验配置

- **Simulator**: Habitat
- **Dataset**: Replica（8 个 indoor scenes）
- **Camera**: FOV $[60°, 60°]$，resolution $512 \times 512$，depth range $[0.1, 5.0]$ m，depth noise $\sigma = 0.01 d$
- **Mission**: max 300s，每 60s evaluate 一次
- **Metrics**:
  - **PSNR**（rendering quality）：从 1000 个 uniform 分布的 test viewpoint 渲染 RGB
  - **Completeness ratio**（mesh quality）：TSDF fusion on rendered depth → Marching Cubes → completeness ratio with 2cm threshold
- **Hardware**: i9-10940X + RTX A5000
- **Timing**: mapping 1s, planning 0.5s（online 可用）
- **Memory**: 4-5 GB GPU RAM（voxel map ~10%）

### 4.2 Baselines 对比

| Method | Map type | View planning 信号 |
|---|---|---|
| **Ours (full)** | 2D GS + voxel | confidence + frontier |
| Ours (w/o ROI) | 2D GS + voxel | confidence + frontier (random sampling only) |
| Ours† | 2D GS + voxel | viewpoint count 替代 confidence |
| FBE [36] | 2D GS + voxel | frontier only (no GS-aware) |
| FisherRF [12] | 2D GS (replaced) | Fisher information |
| NARUTO [5] | NeRF + uncertainty grid | neural uncertainty |

### 4.3 主要结果（Fig 4）

观察：
1. **Ours (full) 在所有 8 个 scene 的 PSNR 和 completeness ratio 上都最好**
2. **vs NARUTO**：差距巨大，尤其 RGB PSNR——motivate 了用 GS 替代 NeRF 做 active reconstruction。NeRF-based 方法为了保证 online 训练，必须 cap model capacity，限制 representation 表达力。
3. **vs FisherRF**：FisherRF 要对每个 candidate viewpoint 计算 Fisher information（GS 参数的 gradient），计算成本极高，导致 planning 慢，在有限 mission time 内 reconstruction 不完整。此外 Fisher information conditioned on candidate viewpoint，不能用来做 viewpoint sampling，只能 evaluation；而 ActiveGS 的 confidence 是 primitive-level 的，既能 sampling 又能 evaluation。
4. **vs FBE**：FBE 只看 exploration，不看 quality，导致 coverage 完成后 quality 仍然不好。
5. **Ablation: ROI sampling**：去掉 ROI（只 random sampling）后 mean 降低、std 增大 —— 证明 targeted sampling 的价值。
6. **Ablation: confidence formulation**：用 viewpoint count 替代 spatial distribution-based confidence（Ours†），效果变差 —— 证明 $\gamma_i, \beta_i$ 的设计是必要的，光数 viewpoint 数量不够。

### 4.4 Real-world experiment

- UAV + Intel RealSense D455
- Scene: $6 \times 6 \times 3$ m
- 不考虑 pitch（UAV 控制限制）
- Pose: OptiTrack motion capture（即 perfect localization，这是 paper 自己承认的 limitation）
- 通过 ROS 跟地面 PC 通信（off-board computing）

---

## 五、我的 critical thoughts 与延伸联想

### 5.1 Confidence formulation 的局限性

当前 $k_i$ 只反映 observation 多样性，**不直接反映 reconstruction error**。考虑这个情况：surfels 的位置都估错了，但因为多次多角度观测，confidence 还是高。这种情况在某些场景下可能会让 planner 误判。

可能的改进方向：
- 加入 photometric residual 信号：每个 primitive 记录它在各 viewpoint 下的 color consistency，颜色一致性差就降 confidence；
- 用 rendered depth 和实际 depth 的 L1 残差作为 viewpoint utility 的补充信号。

### 5.2 与其他 active GS 工作的关系

同期有几个 active GS 工作：
- **GS-Planner** (Jin et al., IROS 2024) 和 **HGS-Planner** (Xu et al., 2024)：把 unknown voxel 引入 GS rendering pipeline 来检测 unseen region，但缺少 quality-aware view planning
- **ActiveSplat** (Li et al., 2024)：用 Voronoi graph 从 GS 提取 traversable topological map，但只支持 2D planning，cluttered environment 下不行
- **FisherRF** (Jiang et al., ECCV 2024)：用 Fisher information 评估 viewpoint 信息量，理论 elegant 但计算太贵

ActiveGS 的差异化优势在于：
1. Confidence 是 per-primitive 的，可以同时用于 viewpoint **sampling** 和 **evaluation**（FisherRF 只能 evaluation）
2. Hybrid voxel + GS 比纯 GS 更适合 path planning

### 5.3 与 Neu-NBV / STAIR / NARUTO 的演化

Liren Jin 之前的工作脉络：
- **Neu-NBV** (IROS 2023)：object-centric 的 NeRF NBV，uncertainty from ensemble
- **STAIR** (IROS 2024)：semantic-targeted active implicit reconstruction
- **NARUTO** (CVPR 2024)：scene-level NeRF active reconstruction，hybrid neural representation + uncertainty grid

ActiveGS 是把 NARUTO 的核心 idea（uncertainty-driven exploration + exploitation）迁移到 GS，并把 uncertainty 从 learned grid 改成 per-primitive confidence。这个迁移的成功说明：**active reconstruction 的核心 abstraction 不在 representation 的形式，而在 "如何 cheaply 评估当前 map 的不足"**。

### 5.4 关于 hybrid map 的设计哲学

Hybrid map 不仅仅是工程拼凑，它反映了一个 deeper insight：**不同的 task 需要 different inductive biases**。

- 高保真重建需要 continuous, differentiable, photorealistic 的 representation → GS
- Path planning 需要 discrete, topological, free/unknown 区分的 representation → voxel map

这种 "用不同 representation 处理不同 sub-problem" 的 pattern 在 robotics 里很常见：例如
- VINS 用 sliding window + global map
- SuMa 用 surfel map + pose graph
- Kimera 用 mesh + semantic 3D scene graph

### 5.5 跟 SLAM community 的连接

这篇 paper 假设 perfect localization（OptiTrack），这跟现实 UAV 工作差距较大。paper 自己在 Section V 提到 future work 要把 localization uncertainty 加进 confidence modelling。这其实是个很自然的 extension：

$$k_i^{\text{new}} = k_i \cdot f(\Sigma_{\text{pose}})$$

其中 $\Sigma_{\text{pose}}$ 是 camera pose 的 covariance。pose uncertainty 大的 viewpoint 对应 primitive 的 confidence 增长应该被打折。这跟 CIVER (Cadena et al.) 里 "joint pose and map uncertainty" 的思想类似。

### 5.6 跟 Next Best View 经典工作的对比

经典 NBV problem (Bajcsy, Connolly, etc.) 多用 voxel-based information gain (entropy reduction)。本文 $U_\mathcal{V}$ 本质就是这种信息增益的简化版。新的部分是 $U_\mathcal{G}$：把 "map quality" 量化为 confidence，并驱动 viewpoint selection。这其实是 active perception 的两个传统轴：**coverage-driven** vs **quality-driven**，本文做了一个加权融合。

### 5.7 训练 schedule 的设计

10 iterations/mapping step, 3 recent + 5 random frames/batch，这个 schedule 值得注意：
- Recent frames 让 map 跟上最新观测
- Random frames 防 catastrophic forgetting
- 10 iterations 是 trade-off：太多次会让新 frame overfit，太少次又不能充分 optimize

这种 "small-batch incremental training" 跟 Gaussian Splatting SLAM (Matsuki et al., CVPR 2024) 和 Photo-SLAM (Huang et al., CVPR 2024) 的工作类似，都是 GS 在 online SLAM 场景的训练范式。

---

## 六、参考资源

**Paper 主线工作**：
- ActiveGS 项目页 / 代码: https://github.com/dmar-bonn/active-gs
- Liren Jin 主页 (Bonn): https://www.ipb.uni-bonn.de/people/liren-jin/
- Cyrill Stachniss group: https://www.ipb.uni-bonn.de/

**Gaussian Splatting 基础工作**：
- 3D GS (Kerbl et al., SIGGRAPH 2023): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- 2D GS (Huang et al., SIGGRAPH 2024): https://surfsplatting.github.io/
- Gaussian Surfel (Dai et al., SIGGRAPH 2024): https://david-huangx.github.io/GaussianSurfels/

**Active reconstruction 相关**：
- NARUTO (CVPR 2024): https://naruto-active.github.io/
- FisherRF (ECCV 2024): https://jiangren2000.github.io/FisherRF-website/
- ActiveNeRF (ECCV 2022): https://active-nerf.github.io/
- Neu-NBV (IROS 2023): https://irvl.dandelion.io/publications/
- GS-Planner (IROS 2024): https://github.com/SJTU-ViSYS/GS-Planner
- ActiveSplat (arxiv 2024): https://dmar-bonn.github.io/active-splat/
- HGS-Planner: https://arxiv.org/abs/2409.17624

**基础设施**：
- Habitat: https://aihabitat.org/
- Replica dataset: https://github.com/facebookresearch/Replica-Dataset
- OctoMap: https://octomap.github.io/
- ROS: https://www.ros.org/

**经典 reference**：
- Frontier-based exploration (Yamauchi 1997): https://ieeexplore.ieee.org/document/6355488
- A* (Hart, Nilsson, Raphael 1968): https://ieeexplore.ieee.org/document/4082128
- EWA Splatting (Zwicker et al. 2001): https://www.cs.umd.edu/~zwicker/publications/EWAVolSplatting_VIS01.pdf
- NeRF (Mildenhall et al. ECCV 2020): https://www.matthewtancik.com/nerf
- KinectFusion (Newcombe et al. ISMAR 2011): https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-dense-surface-mapping-and-tracking/
- Marching Cubes (Lorensen & Cline 1987): https://dl.acm.org/doi/10.1145/37402.37422

---

## 七、总结与对你的 build intuition

如果你想把这个方向的工作串起来理解：

1. **从 NeRF 到 GS 的迁移**：active reconstruction 的 algorithmic skeleton（map-update ↔ viewpoint-plan 循环、exploration ↔ exploitation 平衡）是不变的，变的是 representation + uncertainty signal。GS 比 NeRF 在 online setting 下快得多，且 explicit primitive 让 confidence 可以 per-primitive 建模。

2. **Hybrid representation 是 robotics 中处理 active perception 的常见 pattern**：用一种 representation 做高质量 rendering，用另一种做 topology 和 planning。ActiveGS 用 GS + voxel，类似工作里有 TSDF + surfel、NeRF + occupancy grid 等。

3. **Confidence modelling 的核心**：不需要 ground truth，只需要一个 cheap proxy 衡量 "这个 primitive 是否被 well-observed"。本文用 viewpoint spatial distribution，简单但有效。可以推广到其他 quality proxy：photometric residual、multi-view consistency、normal map 的 TV 等。

4. **Viewpoint planning 的两个 axis**：coverage（exploration, voxel-based）vs quality（exploitation, GS-confidence-based）。Weight 设计要考虑量纲：本文 $\phi=1000$ 是因为 voxel ratio 量级小；这是 phase-transition 控制——mission 早期 exploration 主导，后期 exploitation 主导。

5. **Online training 的关键 trick**：recent + random frame mixing、limited iterations per step、periodic visibility-based pruning。这些 trick 在 GS-based SLAM 工作里也常见，是 incremental GS 的标准操作。

希望这个 walkthrough 对你 build intuition 有帮助。如果你想深挖某个具体方面——比如 confidence formulation 的替代设计、view planning 的 alternative formulation、或者跟 SLAM 中 uncertainty propagation 的连接——我们可以继续聊。
