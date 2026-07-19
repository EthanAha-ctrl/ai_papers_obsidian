---
source_pdf: autoURDF.pdf
paper_sha256: 7539bc21ea76ddca16491bac3e8fbfa0b264fa8ca7b120532ed3520bb7b59d23
processed_at: '2026-07-18T12:18:33-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AutoURDF: 从 Point Cloud 到 URDF 的无监督机器人建模

这是 Hod Lipson 组（Columbia Creative Machines Lab）的工作，第一作者 Jiong Lin。核心 idea 非常 elegant：给一段 robot 运动的 point cloud 视频（10 帧），无监督地输出一个能在 PyBullet 里跑起来的 URDF 文件。整个 pipeline 没有任何 ground truth label，没有 motor command，没有 forward kinematics，纯视觉输入。

paper link: https://jl6017.github.io/AutoURDF/

---

## 1. 动机与定位

机器人建模（URDF/MJCF/USD）传统上是个体力活，需要 CAD 转换或者手写 XML。已有 automated 方法大致分三类：

1. **Robot self-modeling**（如 [Chen et al. 2022](https://www.science.org/doi/10.1126/scirobotics.abn1944), [Liu et al. 2024](https://diffbot.cs.columbia.edu/)）：用 motor command + implicit neural representation 学 morphology。问题是 implicit 表示和 physics simulator 不兼容，并且需要 motor 信号做 supervision。

2. **Articulated object modeling**（PartNet-Mobility 系，[Ditto](https://ditto3d.github.io/), [Real2Code](https://real2code.github.io/), [URDFormer](https://urdformer.github.io/)）：处理 laptop、drawer 这种简单结构。假设每个 moving part 都挂在一个 single parent 上，对 serial chain robot 不成立。

3. **Watch-It-Move / Reart** ([Noguchi et al. 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Noguchi_Watch_It_Move_Unsupervised_Discovery_of_3D_Joints_CVPR_2022_paper.pdf), [Liu et al. 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Building_Rearticulable_Models_for_Arbitrary_3D_Objects_From_4D_Point_CVPR_2023_paper.pdf))：能 reconstruct robots，但用 custom reanimation code 而非标准 URDF，且训练慢（Reart 在同样机器上 35 分钟 vs AutoURDF 1 分钟）。

AutoURDF 的位置：第一个从 unlabeled point cloud 直接构造 functional URDF，能处理到 18 DoF 的复杂 morphology（包括 OP3 humanoid 这种 multi-branch 链）。

---

## 2. Method 深入讲解

整个 pipeline 输入 $\mathcal{P} = \{P^t \in \mathbb{R}^{3 \times N}\}_{t \in [1,T]}$，输出三件套：
- **Segmented parts** $\mathcal{L} = \{L_i\}$：每个 link 的 point cloud
- **Body topology** $\mathcal{G} = (I, E)$：parent-child 关系图
- **Joint parameters** $\mathcal{I} = \{J_i\}$：每个 joint 的 6-DoF 参数

核心 trick 是用一组 point clusters 作为 "motion probe"，跟踪它们的 6-DoF trajectory 来反推结构。

### 2.1 Cluster Registration Pipeline

第一帧用 K-means++ 切成 $S$ 个 clusters（$S$ 是超参，要 over-segment，远大于真实 link 数）。每个 cluster 有一个 6-DoF coordinate $X_i^t = (x_i^t, q_i^t)$，其中 $x \in \mathbb{R}^3$ 是 Cartesian 中心，$q \in \mathbb{R}^4$ 是 quaternion（共 7D 表示，参考 [Zhou et al. 2019](https://arxiv.org/abs/1812.07035) 关于 rotation representation 连续性的工作）。

为什么用 quaternion 而非 Euler angle / rotation matrix？因为：
- Euler angle 有 gimbal lock，且不连续
- Rotation matrix 9D 冗余，NN 学起来不稳
- Quaternion 4D 紧凑，geodesic distance 计算直接（$d_{Geo}(q_1, q_2) = \arccos(|q_1 \cdot q_2|)$）

#### Step Model + Anchor Model 双重 registration

这是 paper 最有意思的设计。两个 model 都是基于 PointNet 风格的 lightweight MLP（共享权重，permutation invariant）。

- **Step Model** $\mathcal{F}_S(X^t, C^t; \theta_S) \to \hat{X}^{t+1}$：相邻帧 $t \to t+1$，单步误差小但 drift 会累积。
- **Anchor Model** $\mathcal{F}_A(\hat{X}^{t+1}, C^1, P^{t+1}; \theta_A) \to \bar{X}^{t+1}$：从第 1 帧直接 align 到 $t+1$，refine Step Model 的输出，消除 drift。

两者配合的 intuition：Step Model 提供好的初值（local 容易 align），Anchor Model 拉回 global consistency。从 Table 2 ablation 看，去掉 anchor model CD 从 9.64 升到 10.73，去掉 step model CD 从 9.64 升到 9.68（影响小一些，因为 anchor model 单独也能撑住，但 step 提供了重要的初始化）。

#### Loss function

$$\mathcal{L}_{\text{Chamfer}}^1(\hat{P}^t, P^t) = \sum_{x=1}^{N} \min_y \|\hat{P}_x^t - P_y^t\|_1 + \sum_{y=1}^{N} \min_x \|\hat{P}_x^t - P_y^t\|_1$$

变量解释：
- $\hat{P}^t = \bigcup_{i=1}^{S} {}^w\hat{C}_i^t$：所有 cluster 变换到 world frame 后拼起来的预测 whole-body point cloud
- $P^t$：ground truth whole-body point cloud
- 第一项：每个预测点找最近的 GT 点（coverage）
- 第二项：每个 GT 点找最近的预测点（completeness）
- 用 L1 而非 L2：对 outlier 更鲁棒，参考 [pytorch3d Chamfer](https://github.com/facebookresearch/pytorch3d)

注意 cluster 优化时在 local frame 里做，Chamfer 评估时在 world frame 里做。这个 frame 切换是必要的：local frame 下 cluster 形状稳定，便于网络学到 shape-feature；world frame 下才能和 GT 比对。

#### Algorithm 1 关键步骤

```
for t = 1, ..., T:
    1. Step Registration:
       - 输入 X^t, local cluster C^t
       - 输出 predicted X^{t+1}, transformed cluster
       - 拼成 P̂^{t+1}
       - 优化 θ_S 最小化 Chamfer(P̂^{t+1}, P^{t+1})
    
    2. Resample:
       - 用预测的 x̂^{t+1} 作为 cluster center，对 P^{t+1} 重新做 K-means
       - 这样 cluster 跟着点云走，不会因为 rigid transformation 后边界漂移丢失 points
       - 转回 local frame
    
    3. Anchor Alignment:
       - 用 X̂^{t+1} 作为初值，再加上第 1 帧的原始 cluster C^1
       - 输出 refined X̄^{t+1}
       - 加入 trajectory X
```

Resampling 是个细节但重要的设计：如果不 resample，初始 cluster 经过几帧 rigid transformation 后，cluster 占用的 spatial region 会和当前 point cloud 错位，导致 Chamfer 优化时 points 不在 cluster 内。

### 2.2 Motion Correlation Matrix 与 Part Segmentation

拿到每个 cluster 在所有时间步的 6-DoF trajectory $X_i^t$ 后，计算 cluster 两两之间的"距离"（这里叫 correlation 但实际是 distance）：

$$\rho(X_i, X_j) = \frac{\sum_{t=1}^{T} \mathcal{D}(X_i^t, X_j^t)}{\max_{i,j} \sum_{t=1}^{T} \mathcal{D}(X_i^t, X_j^t)}$$

- 分子：cluster $i, j$ 在整个 trajectory 上的累计距离
- 分母：所有 cluster pair 中最大的累计距离（用于归一化到 [0, 1]）
- $\rho$ 越小 → motion 越相关 → 越可能属于同一 link

其中单步距离：

$$\mathcal{D}(X_i^t, X_j^t) = \alpha \cdot d_{Euc}(x_i^t, x_j^t) + d_{Geo}(q_i^t, q_j^t)$$

- $d_{Euc}$：position 之间的 Euclidean 距离（单位 mm）
- $d_{Geo}$：quaternion 之间的 geodesic 距离（单位 rad）
- $\alpha$：scaling parameter，从 bounding box 自动计算，用于平衡 position 和 orientation 的量纲（因为 mm 和 rad 不能直接相加）

**Intuition**: 同一 rigid link 上的所有 clusters 应该经历完全相同的 6-DoF motion（rigid body 假设），所以它们的 $\rho$ 应该接近 0。不同 link 之间相对 motion 越大，$\rho$ 越大。这就是用 motion 来 "定义" parts 的核心思想。

#### Ablation 关于 distance function（Figure 11）

- **只用 position (w/o ori)**：TED 从 2.26 升到 7.25。问题在于，rotation 上相似的 links（比如同向旋转的平行 joint）会被错分到一起。
- **只用 orientation (w/o pos)**：TED 升到 5.92。问题是 orientation 噪声大，特别是 small rotation 时 quaternion 区分度不够。
- **完整版**：position 提供 spatial 局部性，orientation 提供 motion 区分度，两者互补。

确定 number of parts 用 [Silhouette Score](https://en.wikipedia.org/wiki/Silhouette_(clustering))，扫不同 $K$ 值取最高分。Figure 9 显示 8 个 robot 都正确预测了 DoF 数量。这个 trick 比直接设定固定 cluster 数要鲁棒，但代价是 S（initial cluster 数）要设得足够大，paper 里 S 是怎么定的没明说，应该是经验值。

### 2.3 Topology Inference via MST

这一步推断哪些 links 之间有 joint 连接。核心算法：

1. 在 clusters 上构造 fully connected graph，edge weight = 累计 position distance
2. 跑 [Kruskal's MST](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm)（paper 引用 Kruskal 1956 原始论文）
3. 找 MST 上跨 segment 的 edges（连接不同 moving part 的 edge）
4. 这些 edges 就是 joint 连接

**Intuition**: 
- MST 保证 tree structure（无 cycle），符合 URDF 假设（除了 closed-chain robots）
- MST 保证 minimal total edge weight，意味着连接 spatial 上 close 的 parts，这符合物理直觉（相邻 link 通常 connected）
- 之前已经做了 part segmentation，所以只需在 segment 之间找连接

最后还需要选 root（base link）。启发式：选 total pose variation 最小的 link 作为 base。这通常合理，因为 base 一般是 fixed 或者 motion 最小的（比如 torso / chassis）。但从 OP3 humanoid 的 TED=6.0 看，这个 heuristic 在复杂 morphology 上会失败。

URDF 要求 directed tree，所以从 root 开始 BFS/DFS 建立 parent-child 关系。

### 2.4 Joint Parameter Estimation

每个 link 有一个平均 6-DoF coordinate，转成 homogeneous transformation matrix $H \in SE(3)$。对每个 parent-child pair：

$$H_i^p = H_i \cdot H_p^{-1}$$

- $H_i$：child link 在 world frame 的 pose
- $H_p$：parent link 在 world frame 的 pose
- $H_i^p$：child 在 parent frame 中的 relative transformation
- $H_p^{-1}$：parent 到 world 的逆变换

这个公式是 SE(3) 上的标准 relative pose。base link 设为单位矩阵 $I$。

对于 1-DoF revolute joint，$H_i^p$ 可以分解为：
- Rotation matrix $R \in SO(3)$
- Static translation $t \in \mathbb{R}^3$

从 $R$ 提取 rotation axis（用 Euler's rotation theorem：任何 3D rotation 都是绕某个固定轴的旋转），从 $t$ 提取 joint center。

具体细节：旋转矩阵 $R$ 的 eigenvector 对应 eigenvalue 1 就是 rotation axis。joint center 是 rotation 的 fixed point，可以通过解 $(R - I) \cdot p = 0$ 得到。

这套 formulation 自然契合 URDF 格式：URDF 的 revolute joint 就是定义一个 axis 和一个 origin（位置 + orientation offset）。

### 2.5 Mesh Generation 和 URDF 输出

每个 link 的 sparse point cloud 跨 10 帧整合成 dense point cloud（在 link local frame 里），然后跑 [Marching Cubes](https://en.wikipedia.org/wiki/Marching_cubes)（Lorensen & Cline 1987）生成 watertight mesh。

为什么跨 10 帧整合？因为单帧 point cloud 稀疏（5000 points / robot），无法直接 reconstruct mesh。利用 rigid body assumption，可以把不同时间步的同一 link 的 points 变换到 local frame 后拼接，相当于多视角融合。

URDF XML 文件直接写 link/joint 标签，mesh 文件存为 .obj 或 .stl。

---

## 3. Experiments 细节

### 3.1 Dataset

8 个 robot，DoF 从 5 到 18：
- Single-branch: WX200 (5), Franka Panda (7), UR5e (6)
- Multi-branch: Bolt (6), Solo8 (8), PhantomX (multi-leg), Allegro Hand (16), OP3 Humanoid (18)

数据：每个 video 10 frames，down-sample 到 5000 points/frame。对 Allegro Hand 和 OP3 固定部分 joint（不然 DoF 太多 segmentation 不动）。对 Franka 和 UR5e 去掉 end-effector。

### 3.2 Metrics

| Metric | 含义 |
|--------|------|
| CD (Chamfer Distance) | Registration 精度，mm |
| TED (Tree Edit Distance) | Topology 精度，编辑距离 |
| $E_{JD}$ | Joint 轴的法向距离，mm |
| $E_{JA}$ | Joint 角度差，degree |
| $CD_r$ | Repose 后的 shape error |

### 3.3 Quantitative Results（Table 1）

- **CD**: AutoURDF 9.63 ± 2.67 vs Reart 16.45 ± 12.18，全面碾压。特别 OP3 上 8.30 vs 44.95，说明 Reart 在复杂 morphology 上彻底崩溃。
- **TED**: AutoURDF 2.26 ± 2.16 vs MBS 6.64 ± 4.07 vs Reart 5.68 ± 4.37。特别 Solo8 上 0.00（完美）。

### 3.4 Multi-Sequence 实验（Table 3）

从 10 帧（1 sequence）扩到 50 帧（5 sequences）：
- $CD_r$：22.13 → 18.42
- $E_{JD}$：5.24 → 3.46
- $E_{JA}$：6.50 → 3.13

WX200 上 $E_{JA}$ = 1.13°, $E_{JD}$ = 1.16mm，非常精确。说明 data scaling 对 AutoURDF 有效，没饱和。

### 3.5 Speed

50s registration + 12s URDF construction on RTX 3090。Reart 在同样机器上 2120s（35.5 分钟）。**~40x speedup**。这是 unsupervised method 的 inherent 优势：不需要训练 GAN/flow model，只优化一个轻量 MLP。

### 3.6 Real-world Experiment（Figure 10）

WX200 真实扫描数据，5 个 servo 都动。10 帧 noisy point cloud，成功 reconstruct URDF。这个实验非常有说服力——sim-to-real gap 在 point cloud 层面主要体现在 noise 和 missing data，AutoURDF 的 cluster-based registration 对 noise 有一定鲁棒性（因为 Chamfer distance + cluster aggregation 起到了 averaging 作用）。

---

## 4. Ablation 关键发现

### 4.1 Anchor Model 的作用

Table 2: w/o anchor m → CD 10.73 vs Ours 9.64。Anchor model 主要解决 drift 问题。在没有 anchor 的情况下，Step Model 累积误差会导致后期的 cluster 偏离真实位置，进而影响 part segmentation（因为 motion correlation matrix 会不准）。

### 4.2 Position vs Orientation

Table 2:
- w/o ori: TED 7.25（position 不够区分 rotation-only joint）
- w/o pos: TED 5.92（orientation 不够区分 spatial 上 far apart 但 rotation 相似的 links）
- Full: TED 2.26

这验证了 paper 的核心 thesis：6-DoF representation 是必要的，3-DoF position 或者 3-DoF orientation alone 都不够。

---

## 5. 关键 Intuition 总结

1. **Over-segment then merge**：先用 K-means 切成很多小 cluster（远多于 link 数），再用 motion 相关性合并。这避免了直接预测 link 数量的难题。

2. **Motion defines parts**：rigid body 的定义就是 "所有点经历相同 6-DoF motion"。AutoURDF 直接 operationalize 这个定义，用 motion correlation 来 segment parts。这是非常 clean 的物理 intuition。

3. **MST for tree structure**：robot kinematic tree 是个 tree，MST 是从 complete graph 中提取 tree 的最自然方法。Edge weight = spatial distance 也合理（相邻 link 通常 connected）。

4. **Step + Anchor 双注册**：local + global registration 的组合，类似 SLAM 中的 odometry + loop closure。

5. **Euler rotation theorem 约束**：把 SE(3) 的 6-DoF 压到 1-DoF revolute joint，用 fixed point + axis 参数化。这是 mathematical prior，让 URDF 兼容。

---

## 6. Limitations（paper 自述 + 我的 critique）

1. **无 dynamics**：URDF 缺 mass / moment of inertia。paper 说留给 future work，但这个其实挺关键——没有 dynamics 的 URDF 只能做 kinematics simulation，不能做 dynamics simulation。要从 point cloud 估 mass property 非常难，需要 material 识别。

2. **复杂 morphology 需要更多 frames**：OP3 humanoid 上 TED = 6.00，segmentation 不干净。从 Figure 9 看 OP3 的 Silhouette Score 曲线也比较 noisy。这说明 motion diversity 不够时，method 会失败。

3. **只支持 revolute joint 和 tree structure**：不支持 prismatic、spherical、closed-chain。这个限制挺大，因为并联机构（如 Delta robot）、spherical joint（如 hip）、prismatic joint（如 linear actuator）在 robotics 中很常见。

4. **数据采集需要 collision-free motion**：要预先 random sample motor angle 让 robot 动，且不能撞自己。这其实需要知道 forward kinematics，有点 chicken-and-egg。如果 robot 已经有 controller，那肯定已经有 URDF/MJCF 了。所以实际场景应该是：有 CAD 但没 URDF，或者手动示教让 robot 动起来。

5. **Cluster 数 S 的选择**：paper 没说 S 怎么定。从 Figure 3 看 S 应该远大于 DoF（比如 50+），但具体多少？太大计算贵，太小 segmentation 粗。这是个未公开的 hyperparameter。

6. **Quaternion 距离的 ambiguity**：$q$ 和 $-q$ 表示同一 rotation。geodesic distance 用 $|q_1 \cdot q_2|$ 取绝对值处理这个，但在 trajectory 优化中可能仍有问题。

7. **静默假设：所有 link 都是 rigid**。如果 robot 有柔性 link（比如 soft robot），method 会失败。但对传统 rigid robot 这个假设合理。

---

## 7. 相关工作和延伸阅读

- **PointNet / PointNet++**: [Qi et al. 2017](https://arxiv.org/abs/1612.00593), [Qi et al. 2017 (NeurIPS)](https://arxiv.org/abs/1706.02413) — AutoURDF 的 network backbone
- **Chamfer Distance**: [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/loss/chamfer.py) — 标准 point cloud alignment loss
- **MultiBodySync**: [Huang et al. 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_MultiBodySync_Multi-Body_Segmentation_and_Motion_Estimation_via_3D_Scan_CVPR_2021_paper.pdf) — 主要 baseline
- **Reart**: [Liu et al. 2023](https://reart.github.io/) — 另一个 baseline，用 4D point cloud 重建 rearticulable models
- **Watch-It-Move**: [Noguchi et al. 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Noguchi_Watch_It_Move_Unsupervised_Discovery_of_3D_Joints_CVPR_2022_paper.pdf) — 早期工作，发现 3D joints
- **Real2Code**: [Mandi et al. 2024](https://arxiv.org/abs/2406.08474) — 用 code generation 重建 articulated objects
- **URDFormer**: [Chen et al. 2024](https://urdformer.github.io/) — 从 image 生成 URDF
- **Robot self-modeling**: [Kwiatkowski & Lipson 2019](https://www.science.org/doi/10.1126/scirobotics.aau9354) — Hod Lipson 组早期 self-modeling 工作
- **Differentiable Robot Rendering**: [Liu et al. 2024](https://diffbot.cs.columbia.edu/) — 同组近期工作，differentiable rendering
- **Marching Cubes**: [Wikipedia](https://en.wikipedia.org/wiki/Marching_cubes) — point cloud 转 mesh 的经典算法
- **Silhouette Score**: [scikit-learn doc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) — cluster 数量选择
- **Tree Edit Distance**: [Zhang's algorithm](https://en.wikipedia.org/wiki/Tree_edit_distance) — topology 比较

---

## 8. 我的思考

这个工作在 methodology 上很 solid，几个亮点：

1. **Hierarchical decomposition** 把一个 hard problem 拆成三个相对独立的 sub-problem，每个都用经典算法（registration、MST、Euler theorem）。这种 modularity 让 debugging 容易，每个 stage 可以单独评估。

2. **Motion-as-identity** 的思想非常 general。rigid body 的本质就是 rigid motion，所以用 motion coherence 来定义 part 是最 principled 的方法。这个思想其实在 [Dynamic Shape Analysis](https://arxiv.org/abs/2104.06981) 等工作中也出现过。

3. **Unsupervised + explicit representation** 是 sweet spot。纯 implicit（NeRF-based）虽然 supervised 简单但不能直接 plug 进 simulator；纯 explicit（CAD）需要 manual。AutoURDF 卡在中间，自动产出 explicit URDF。

4. **10 frames 是惊人的数据效率**。对比 Real2Code、URDFormer 需要 supervised training data，AutoURDF 单 video 就能 reconstruct。代价是只能 reconstruct 单个 robot，没有 generalization across robots（没有 prior knowledge）。

5. **潜在改进方向**:
   - 把 Step/Anchor Model 换成 Transformer 或更现代的 point cloud backbone（如 [Point Transformer V3](https://arxiv.org/abs/2304.09714)），可能进一步提升 registration 精度
   - 加 diffusion / generative prior 来处理 observation noise（特别是真实扫描）
   - 引入可微分 physics simulation（如 [Brax](https://github.com/google/brax) 或 [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)），让 URDF 在 loop 里 refine
   - 多模态融合：point cloud + RGB + IMU（如果有）能进一步降低 ambiguity
   - 扩展到 closed-chain：可以用 graph theory 中的 cycle basis 来表示 closed loop

6. **和 LLM-based robot modeling 的对比**：[URDFormer](https://urdformer.github.io/) 用 LLM 从 image 直接 generate URDF XML，需要大量 training data。AutoURDF 走的是纯几何/物理路线，更 interpretable 但需要 robot 主动运动。两者可以互补：LLM 先生成粗略结构，AutoURDF 用真实 motion data refine。

7. **Real-world application 场景**：
   - 机器人换新：买到新 robot 但没 URDF，扫描 + 录一段运动 → 自动生成 URDF
   - Robot damage recovery：part 损坏替换后，重新 self-model
   - Custom robot / DIY robot：maker 自己拼的 robot 没 CAD
   - Legacy robot：老 robot 没 digital twin

8. **和 Karpathy 自己工作的联系**：你之前在 Stanford CS231n 讲过 PointNet，这里的 network backbone 也是 PointNet。可以把这个工作看作 PointNet 在 robot self-modeling 上的应用。整个 pipeline 没有太多 deep learning magic，更多是 classical geometry + lightweight NN，这反而是优点——interpretable 且 data-efficient。

总结一句：AutoURDF 是一个 engineering 完成度高、mathematical formulation 清晰、实验充分的工作。它不追求 fancy deep learning architecture，而是用经典算法 + 现代 point cloud NN 解决实际问题。从 Hod Lipson 组一贯的风格（self-aware robots、self-modeling）看，这是 robot self-modeling 系列的延续，从 implicit representation 转向 explicit + simulator-compatible representation 是个重要 step。

值得 follow 的方向：把这个 pipeline 扩展到 closed-chain robots 和 prismatic joints，加 dynamics estimation，以及探索 LLM-guided 的 prior 注入。
