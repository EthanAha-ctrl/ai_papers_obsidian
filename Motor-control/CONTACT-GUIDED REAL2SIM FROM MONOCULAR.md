---
source_pdf: CONTACT-GUIDED REAL2SIM FROM MONOCULAR.pdf
paper_sha256: 131cf919958bb8c8a7c954cdffef4e8367b9465f58e53928aaf52d8536e69259
processed_at: '2026-08-03T17:03:01-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CRISP 人话版

## 一句话总结

你拿手机拍了段视频，里面一个人在走、在坐、在爬楼梯。CRISP能把这视频变成simulation里能用的东西——人的动作 + 周围环境——让你能在simulation里训练humanoid robot模仿这些动作，而且训练几乎不崩。

## 这问题为啥难

之前VideoMimic（concurrent work，https://arxiv.org/abs/2505.03729 ）的做法是：把场景重建成一个dense mesh，几十万个三角形那种。听起来很合理，但实际跑起来灾难性的：

重建出来的mesh有各种artifact——有些地方重复建了一层（duplicate structure），有些地方该平的不平（bumpy），有些地方该有的surface没了。你把humanoid放进去训练RL，agent会：
- 卡在"ghost surface"里出不來
- 被地上不该有的小突起绊飞
- 因为接触力忽大忽小直接抖死

VideoMimic的RL失败率是**55.2%**。一半的时候训练直接崩。

## CRISP的核心idea

场景里的东西，绝大部分是**平的**。地板是平的，墙是平的，楼梯是平的，椅子面是平的，桌子面是平的，沙发面是平的。

那干嘛不用一堆平板来表示场景？

具体说，用大概50个planar primitive（可以理解成薄板子）来拼出整个场景。这有几个立竿见影的好处：

1. **平板是convex的**，Isaac Gym算collision detection飞快
2. **没有artifact**，平板就是平板，不会有ghost surface
3. **对噪声鲁棒**，depth估计抖一点也不会凭空生出怪东西

结果：RL失败率从55.2%降到6.9%，simulation速度快43%。

## 具体怎么做

### 第一步：视频进来，先恢复相机和场景

用MegaSAM（https://arxiv.org/abs/2412.04463 ）做visual SLAM，拿到相机轨迹和dense point cloud。但MegaSAM自己的depth estimator不够好，CRISP把它换成了MoGe（https://wangrc.site/MoGePage/ ），因为MoGe是scale-invariant的，能减少duplicate structure这种artifact。

同时用GVHMR（https://research.nvidia.com/labs/dir/liva-siggraphasia2024/ ）恢复SMPL人体，再用相机轨迹lift到world frame。

有个小trick：MegaSAM重建出来的point cloud不知道真实scale（单目视频的通病），但人的身高是已知的。所以把point cloud缩放一下，让里面"人的深度"跟SMPL mesh的深度对上，就恢复出metric scale了。

### 第二步：point cloud变成一堆平板

这是CRISP最核心的算法（Algorithm 1）。输入是per-frame point cloud，输出是50个左右的planar primitive。

流程很朴素，但work得很好：

1. **每帧segmentation**：先估每个点的normal方向，然后用K-means按normal方向聚类。normal方向相近的点大概率在同一个平面上。再在同一个normal group里做DBSCAN空间聚类——因为两个平行墙面normal方向一样但物理上不连续。

2. **跨帧merge**：用optical flow把frame $i$ 的segment warp到frame $j$，看跟frame $j$ 的segment重合多少。如果重合度高而且normal方向也接近，就判定是同一个物理平面的不同观测，merge起来。这一步很关键，因为同一个墙面在不同帧可能被分成不同的segment。

3. **平面拟合**：对每个merge好的group做RANSAC plane fitting拿到plane normal和center，再把inlier投影到平面上fit一个最小面积矩形，就得到了平板的size和orientation。默认厚度0.05m。

整个fitting算法只占总runtime的8.1%，非常lightweight。

### 第三步：用contact补全被遮挡的surface

问题来了：场景有些关键surface被人挡住了。

比如人坐在椅子上，椅子座面被人挡住，你根本看不见。那simulation里humanoid坐哪？会直接掉下去。

CRISP的解法是用InteractVLM（https://interactvlm.github.io/ ）预测SMPL mesh上哪些vertex在接触场景。如果你知道大腿在接触什么东西，那你就能推断大腿下面应该有个面。

但InteractVLM有个毛病：在"即将接触但还没接触"的frame会over-predict false positive（因为它没在这种hard negative上train过）。CRISP的解法是temporal filtering——在一个长度为 $L$ 的window里，只保留那些**人最静止**的frame的contact prediction：

$$
t^* = \underset{t \in \{i, i+L\}}{\arg\min} v_t
$$

其中 $v_t$ 是frame $t$ 的人体运动幅度。contact的时候人通常是静止的，所以最静止的frame最可能是真contact。然后在那个frame，用SMPL mesh的位置去"幻觉"出被遮挡的surface。

### 第四步：用RL验证

把重建的人和场景放进Isaac Gym，训练一个DeepMimic-style（https://xbpeng.github.io/projects/DeepMimic/index.html ）的motion tracking policy。

**这步本质上是拿RL当reconstruction质量的validator**。如果重建好，RL能收敛，humanoid能稳定模仿；如果重建差，RL会崩。这比用Chamfer Distance当metric更discriminative——VideoMimic的CD不是特别烂（0.337），但RL success只有44.8%。

reward是标准的那几项：position match、rotation match、velocity match、root height match、energy penalty。policy是transformer encoder，follow MaskedMimic（https://research.nvidia.com/labs/toronto-ai/maskedmimic/ ）的架构。

## 结果说话

| 指标 | VideoMimic | CRISP |
|------|-----------|-------|
| RL success rate | 44.8% | **93.1%** |
| simulation FPS | 16K | **23K (+43%)** |
| EMDB W-MPJPE100 | 505.31mm | **175.93mm** |
| Non-penetration | 0.906 | **0.947** |

有个有意思的细节：VideoMimic加RL后error反而变大（110.64→145.24mm），因为场景太烂humanoid在simulation里越跑越偏。CRISP加RL后error变小（78.16→70.60mm），因为场景好humanoid能稳定track。**RL是reconstruction质量的照妖镜**。

## 几何ablation的insight

Paper里有个ablation比较了TSDF mesh、NKSR mesh、planar primitive：

- Planar的**双向Chamfer Distance略差**于NKSR（0.187 vs 0.163），因为平板重建有些fine structure没建出来
- 但planar的**单向Chamfer（Recon→GT）最好**（0.174），说明重建出来的部分都在GT附近
- 关键是**RL success最高**（93.1% vs 79.3%）

这告诉我们一个principle：**missing tiny non-contact details is harmless, but extra noisy geometry destabilizes rollouts**。对于simulation task，reconstruction的"精度"比"完整度"重要。

## 限制

Paper很诚实地承认了三个限制：
1. 曲面物体（比如球、圆柱）用平板表示会faceted，建议用superquadric
2. 不能model流体和可变形物体（沙子、水、布料）
3. 只支持static scene，不支持dynamic object manipulation

## 我的take

这篇paper最深的insight不在算法，而在**对"reconstruction quality"的重新定义**。传统reconstruction paper拼的是Chamfer Distance、F1 score这些geometric metric。CRISP说：这些metric好不等于downstream simulation好。CD略差的planar primitive反而simulation最稳定，因为没有spurious geometry。

这suggests future real2sim工作应该report downstream task performance（RL success rate、locomotion stability）作为metric，而不只是geometric metric。**Task-driven perception always beats pixel-driven perception**。

另外一个intuition：**representational simplicity often beats representational richness for downstream tasks**。50个平板比几十万triangles的mesh更适合simulation，因为convex、clean、no artifact。这是个反直觉但很practical的发现。

项目主页有interactive demo：https://crisp-real2sim.github.io/CRISP-Real2Sim/

---

# CRISP: Contact-Guided Real2Sim from Monocular Video with Planar Scene Primitives

这篇paper来自CMU的Deva Ramanan组，核心要解决的问题是：**如何从一个casually captured的单目视频，恢复出simulation-ready的human motion和scene geometry，使得RL humanoid controller能在里面跑起来**。这看起来是个reconstruction问题，但实际上是个"real2sim"的engineering问题——重点不是重建得多漂亮，而是重建出来的东西能不能让physics simulation稳定收敛。

## 1. 核心Motivation: 为什么prior work会失败

VideoMimic (Allshire et al., 2025, arXiv:2505.03729) 是concurrent work，走的是dense mesh路线——TSDF fusion + Marching Cubes。问题在于：

- **几何artifacts**: duplicate structures, over-smoothed surfaces, bumpy terrain
- **simulation不stable**: humanoid会卡在"ghost surfaces"里，或者从protruding geometry上弹飞
- **collision detection慢**: dense mesh有几十万triangles，convex decomposition成本高

CRISP的key insight：**用少量(~50个)convex planar primitives来近似scene**。这看似是个限制（很多物体不是平面的），但实验证明对于human-scene interaction（sitting, lying, parkour, stairs）足够用，而且：
1. Convex primitives在Isaac Gym里collision detection极快
2. Planar assumption对噪声robust
3. 没有artifacts就不会有simulation失败

## 2. Pipeline整体架构

输入：单目RGB video $\mathcal{V} = \{I_i \in \mathbb{R}^{H \times W}\}_{i=1}^N$

输出：simulation-ready的human motion + scene primitives

四个stage：

### Stage 1: Camera, Human, Scene初始化
- **MegaSAM** (Li et al., 2024, arXiv:2412.04463): 恢复相机内参 $\mathcal{K} \in \mathbb{R}^{3\times3}$、per-frame camera pose $\mathcal{T}_i = [\mathcal{R}_i | t_i] \in SE(3)$、dense depth map $\mathcal{D}$
- 关键trick: 把MegaSAM默认的depth estimator换成**MoGe** (Wang et al., 2025c, CVPR 2025)，因为MoGe是scale-invariant的，减少duplicate structures
- **GVHMR** (Shen et al., 2024, SIGGRAPH Asia): 输入intrinsics，输出camera-space的SMPL mesh，再用 $\mathcal{T}_i$ lift到world frame
- **Metric scale recovery**: 因为MegaSAM的point cloud $\mathcal{P}$ 是unknown scale的，但human的scale是已知的，所以通过让scaled point cloud的human深度匹配SMPL mesh的深度来恢复metric scale，得到 $\tilde{\Phi}$

### Stage 2: Normal-based Planar Primitive Fitting (核心算法)

Algorithm 1是这篇paper的精华。输入是 $[T, N, 3]$ 的per-frame pointmaps，输出是 $M$ 个planar primitives的rotation $\mathbf{R}$, translation $\mathbf{t}$, size $\mathbf{S}$，以及point-to-plane assignment $\Pi \in \{0,1\}^{[NT, M]}$。

**Step 1: Per-frame segmentation**
- 从points估计normals: $P_t[N, 3] \xrightarrow{\text{finite-diff}} N_t[N, 3]$
- **K-means on normals**: $N_t[N, 3] \xrightarrow{\text{KMeans}} y_t[N]$，把点分成K个normal方向相近的groups
- **DBSCAN spatial clustering**: 对每个group $X_k = \{p \in P_t : y_t(p) = k\}$，再做空间DBSCAN，因为同样normal方向的点可能物理上不连续（比如两个平行墙面）

**Step 2: Cross-frame association**
- 用optical flow $\Phi_{i \to j}$ 把frame $i$ 的segment warp到frame $j$
- 计算overlap ratio $\rho_{ab}$ 和normal cosine $\gamma_{ab} = \langle \bar{n}_{i,a}, \bar{n}_{j,b} \rangle$
- 如果两个score都过阈值，就link $(i, a) \sim (j, b)$
- 关键trick: 把 $A + B - X$ 作为新的segment集合（union minus intersection），这样能merge跨frame的同一物理平面
- 最终aggregate出 $M$ 个global planar groups

**Step 3: Primitive fitting**
- 对每个group做**RANSAC plane fitting**: $X_m = \bigcup \{p \in P_t\} \to (n, c)$，得到plane normal $n$ 和center $c$
- **Min-area rectangle**: 把inliers投影到plane上，fit最小面积矩形，得到in-plane axes $(x, y)$，然后 $\mathbf{R}_m = [x, y, n]$（right-handed）
- **Size & Center**: $S_x, S_y$ 从in-plane coverage来，$S_z$ 从normal-direction spread来（默认0.05m厚度）
- Offset: $\Delta = \frac{1}{2} S_z n$, $\mathbf{t}_m = c + \Delta$ —— 这把plane center移到cuboid的face上，而不是几何中心

**Step 4 (optional): Contact-guided hallucination**
- 在predicted contact points上重复上述fit，augment新的planes
- $S_z \geq 0.05$m clamp防止退化成infinitely thin plane

### Stage 3: Contact-Guided Scene Completion

问题：scene的关键interaction surface经常被人occlude掉。比如坐椅子时椅子seat被人挡住，爬楼梯时stair tread被脚挡住。

**InteractVLM** (Dwivedi et al., 2025, CVPR 2025): 输入image，输出SMPL vertices上的binary contact mask $c_t(v) \in \{0, 1\}$。

但InteractVLM在"near-contact" frame会over-predict false positives（因为它没在这种hard negative上train过）。

**Temporal-kinematic filtering**:

$$
t^* = \underset{t \in \{i, i+L\}}{\arg\min} v_t
$$

其中 $v_t$ 是frame $t$ 的human motion magnitude，$L$ 是window length。意思是在一个长度为 $L$ 的window里，找人体最静止的那个frame，因为contact时人通常是静止的。同时要求这个window内所有frame的contact confidence都高于阈值 $\tau$。这就是non-maximum suppression的时序版本。

用这个最静止frame的SMPL mesh $\mathcal{M}_{t^*}$ 来推断occluded surface的位置——比如大腿水平时，下面应该有chair seat；脚踩在空中时，下面应该有stair tread或ground。

### Stage 4: Physics-Based Motion Tracking (RL验证)

这是DeepMimic (Peng et al., 2018, arXiv:1804.02717) 风格的motion tracking。

**Observation** (robot state):
$$
s_t = \big(\theta_t \ominus \theta_t^{\text{root}}, \ (p_t - p_t^{\text{root}}) \ominus \theta_t^{\text{root}}, \ v_t \ominus \theta_t^{\text{root}}\big)
$$

变量解释：
- $\theta_t$: joint orientations (quaternions)，所有joint
- $\theta_t^{\text{root}}$: root joint的orientation
- $\ominus$: quaternion subtraction，表示relative rotation
- $p_t$: joint positions
- $p_t^{\text{root}}$: root position
- $v_t$: linear and angular velocities

所有东西都减去root的orientation，这样agent是在自己的local frame里观察世界。

**Target conditioning**: policy还conditioned on未来 $K$ 个target poses $g_t = [\hat{f}_{t+1}, \hat{f}_{t+2}, \ldots, \hat{f}_{t+K}]$。每个joint的target:
$$
\hat{f}_t^j = \big(\hat{\theta}_t^j \ominus \theta_t^j, \ \hat{\theta}_t^j \ominus \theta_t^{\text{root}}, \ (\hat{p}_t^j - p_t^j) \ominus \theta_t^{\text{root}}, \ (\hat{p}_t^j - p_t^{\text{root}}) \ominus \theta_t^{\text{root}}\big)
$$

这四项分别是：joint rotation error、joint rotation relative to root、joint position error、joint position relative to root。

**Action**: PD controller的target joint angles，policy是multivariate Gaussian，$\sigma_\pi = 0.055$。

**Reward**:
$$
\begin{aligned}
r_t = &\ w_p e^{-\alpha_p \|\hat{p}_t - p_t\|} + w_r e^{-\alpha_r \|\hat{q}_t \ominus q_t\|} + w_v e^{-\alpha_v \|\hat{p}_t - \dot{p}_t\|} \\
&+ w_\omega e^{-\alpha_\omega \|\hat{q}_t - \dot{q}_t\|} + w_h e^{-\alpha_h \|\hat{h}_t - h_t\|} + w_e \sum_j \|\tau_j \dot{q}_j\|
\end{aligned}
$$

六项的含义：
- $w_p e^{-\alpha_p \|\hat{p}_t - p_t\|}$: position matching，$\hat{p}_t$ 是reference的joint positions
- $w_r e^{-\alpha_r \|\hat{q}_t \ominus q_t\|}$: rotation matching，$\hat{q}_t$ 是reference的joint rotations
- $w_v e^{-\alpha_v \|\hat{p}_t - \dot{p}_t\|}$: linear velocity matching，$\dot{p}_t$ 是current linear velocity
- $w_\omega e^{-\alpha_\omega \|\hat{q}_t - \dot{q}_t\|}$: angular velocity matching
- $w_h e^{-\alpha_h \|\hat{h}_t - h_t\|}$: root height matching
- $w_e \sum_j \|\tau_j \dot{q}_j\|$: energy penalty，$\tau_j$ 是joint torque，$\dot{q}_j$ 是joint angular velocity，这是power dissipation的代理

具体weight (Appendix H):
$$w_p = 2.5, w_r = 1.5, w_v = 0.5, w_\omega = 0.5, w_h = 1, w_e = 0.001$$
$$\alpha_p = 1.5, \alpha_r = 0.3, \alpha_v = 0.12, \alpha_\omega = 0.05, \alpha_h = 20$$

$\alpha$ 控制exponential decay的sharpness，$\alpha_h = 20$ 很大说明root height的tolerance很窄。

**Training details**:
- Policy: transformer encoder, latent dim 256, FFN 512, 2 layers, 2 heads
- Critic: MLP [1024, 512]
- PPO with GAE, $\gamma = 0.99$, $\tau = 0.95$, lr $= 2 \times 10^{-5}$
- 2048 parallel environments, batch 8192
- Isaac Gym, 120Hz simulation, 30Hz policy
- **Reference State Initialization (RSI)**: 10%概率从第一帧开始，90%均匀采样
- **Early Termination (ET)**: 任何joint偏离reference超过0.5m就terminate

## 3. 实验结果分析

### Table 1: 几何表示的ablation

| Method | RL | Success ↑ | FPS ↑ | PROX Success | CD_bi ↓ | CD_one ↓ | Non-Pene ↑ | EMDB Success | W-MPJPE100 ↓ |
|--------|----|-----------|-------|--------------|---------|----------|------------|--------------|--------------|
| VideoMimic | ✓ | 44.8% | 16K | 27.3% | 0.337 | 0.311 | 0.906 | 50.0% | 505.31 |
| Ours (TSDF) | ✓ | 75.9% | 15K | 72.7% | 0.178 | 0.222 | 0.925 | 77.8% | 197.77 |
| Ours (NKSR) | ✓ | 79.3% | 16K | 90.9% | 0.163 | 0.187 | 0.937 | 75.0% | 185.00 |
| **Ours (Planar)** | ✓ | **93.1%** | **23K** | 90.9% | 0.187 | 0.174 | **0.947** | **93.8%** | **175.93** |

几个intuition：

1. **TSDF已经比VideoMimic好很多**（44.8% → 75.9%），说明VideoMimic的dense mesh质量确实差，artifacts多
2. **NKSR比TSDF好**：NKSR (Huang et al., 2023, CVPR) 用neural kernel做surface reconstruction，sharper surfaces
3. **Planar primitives的CD_bi略差于NKSR**（0.187 vs 0.163），但CD_one (Recon→GT) 最好（0.174）。这说明planar reconstruction"少而精"——重建出来的部分都在GT附近，但有些fine-grained结构没建出来
4. **FPS 23K vs 16K = 43% faster**：因为convex primitives的collision detection远快于dense mesh
5. **Non-Penetration 0.947最好**：planar surface没有bumps，humanoid不会突然穿进去

为什么CD_bi差但simulation好？因为**missing tiny non-contact details is harmless, but extra noisy geometry destabilizes rollouts**。这是这篇paper最核心的insight之一。

### Table 2: EMDB上的HMR

| Method | RL | WA-MPJPE100 ↓ | W-MPJPE100 ↓ | RTE ↓ | Jitter ↓ | ACCEL ↓ |
|--------|----|---------------|--------------|-------|----------|---------|
| WHAM | × | 98.45 | 267.53 | 3.30 | 22.57 | 5.21 |
| TRAM | × | 83.61 | 249.50 | 1.93 | 24.00 | 4.82 |
| GVHMR | × | 74.80 | 200.71 | 1.90 | 15.50 | 4.39 |
| VideoMimic | × | 110.64 | 521.09 | 2.12 | 9.29 | 4.65 |
| VideoMimic | ✓ | 145.24 | 505.32 | 3.00 | 8.34 | 4.17 |
| Ours | × | 78.16 | 179.84 | 1.88 | 13.04 | 4.59 |
| **Ours** | **✓** | **70.60** | **175.93** | 1.90 | **8.14** | **4.10** |

注意：
- **VideoMimic的RL反而让error变大**（110.64 → 145.24）！因为它重建的scene太烂，humanoid在simulation里根本track不住reference，越跑越偏
- CRISP的RL让error变小（78.16 → 70.60），因为scene好，humanoid能稳定track
- **Jitter 8.14最低**：RL能平滑时序动态，这是physics simulator的天然regularizer
- VideoMimic的W-MPJPE100高达521mm，基本是失败的reconstruction

### Table 3: Contact的ablation

| Method | Contact? | Success ↑ | CD two-way ↓ | CD GT→Recon ↓ | CD Recon→GT ↓ | Non-Pene ↑ |
|--------|----------|-----------|--------------|---------------|---------------|------------|
| VideoMimic | × | 27.3% | 0.337 | 0.3625 | 0.3114 | 0.928 |
| Ours | × | 90.9% | 0.193 | 0.211 | 0.175 | 0.947 |
| Ours | ✓ | 90.9% | 0.187 | 0.199 | 0.173 | 0.947 |

注意PROX上Success没变（都是90.9%），因为PROX主要是sitting，seat缺失时humanoid会自动switch to squatting pose——所以仍然"成功"但motion不faithful。但CD降低了，说明contact确实补全了occluded surface。

## 4. Runtime分析 (Table 4)

300帧(10s) 1440×1920视频，RTX A6000：

| Module | Runtime (s) | Proportion |
|--------|-------------|------------|
| Prior preparation | 297.33 | 32.3% |
| Visual SLAM (MegaSAM) | 518.18 | 56.3% |
| HMR (GVHMR) | 30.51 | 3.3% |
| Planar fitting | 74.97 | 8.1% |
| **Total** | **920.99** | **100%** |

关键观察：**Planar fitting只占8.1%，是lightweight的**。bottleneck是Visual SLAM和prior preparation。如果用real-time RGB-D SLAM，整个pipeline可以real-time跑。

VideoMimic在同样硬件上需要1282.94s，CRISP快了约30%。

## 5. 我的Intuition Building

读这篇paper我有几个insight：

### 5.1 "Simulation-ready"是个严格的quality bar

Reconstruction paper通常report Chamfer Distance、F1 score这些geometric metric。但CRISP指出，**geometric metric好不等于simulation好**。CD略差的planar primitives反而simulation最稳定，因为：
- 没有spurious geometry → 没有ghost collision
- Convex → collision detection快且stable
- Missing fine details在non-contact区域无害

这suggests future reconstruction paper应该report downstream simulation success rate作为metric，而不只是Chamfer Distance。

### 5.2 Planar assumption是个strong prior，但够用

Algorithm 1看着朴素（K-means + DBSCAN + RANSAC），但效果很好。原因是**human-scale environment确实是planar-dominated的**：floor, walls, stairs, chair seats, sofa surfaces, tables... 90%的interaction surface都是平的。

Paper在limitations里承认curved surfaces会有问题，但实验证明这不影响locomotion success rate。这让我想到一个principle：**对于downstream task，representational simplicity often beats representational richness**。

### 5.3 Contact作为geometric completion的cue

InteractVLM预测SMPL vertex上的contact mask，然后用人体的姿态推断occluded surface——这个idea很clever。本质上是说：**人体姿态是scene geometry的隐式观察**。如果人坐着，那大腿下面一定有surface；如果脚悬空，那脚下面一定有surface。

但limitation也很明显：如果HMR在world frame里drift了，contact-augmented plane也会drift。Paper承认这是future work。

### 5.4 RL作为reconstruction的validator

最有趣的是，CRISP用RL success rate来evaluate reconstruction quality。这是个circular的设计：用reconstruction训练RL，用RL success判断reconstruction好不好。但实验证明这个metric比Chamfer Distance更discriminative——VideoMimic的CD不是特别烂(0.337)，但RL success只有44.8%。

这让我想到DeepMind的"task-driven perception"传统：**perception的quality应该由downstream task定义，而不是by pixel-level metric**。

### 5.5 跟MaskedMimic的关系

Paper提到observation/action/reward设计follow MaskedMimic (Tessler et al., 2024, ACM TOG)。MaskedMimic是个unified physics-based character controller，通过masked motion inpainting训练。CRISP借用了它的transformer policy架构和RSI/ET策略。

CRISP和MaskedMimic的关系：MaskedMimic是controller，CRISP是data pipeline。CRISP产生simulation-ready的human+scene，MaskedMimic-style的policy在里面训练。

### 5.6 Limitations的诚实

Paper承认三个limitation：
1. **Planar primitives对curved surfaces有限**：建议用superquadrics
2. **不能model fluid/deformable**：sand, water, cloth
3. **不能model dynamic objects**：只支持static scene，不支持loco-manipulation

这跟HDMI (Weng et al., 2025, arXiv:2509.16757) 和SkillBlender (Kuang et al., 2025, arXiv:2506.09366) 的whole-body loco-manipulation形成对比——那些方法支持dynamic interaction但需要更多annotation。

## 6. 跟相关工作的比较

- **JoSH (Liu et al., 2025)**: joint optimization of human-scene-contact，但没physics in loop
- **TRAM (Wang et al., 2024)**: global trajectory estimation，用DROID-SLAM，但没scene reconstruction
- **WHAM (Shin et al., 2024)**: 用contact label refine body pose，但contact只是foot-ground
- **PLACE (Zhang et al., 2020a)**: non-penetration metric，CRISP也用这个
- **PhysDiff (Yuan et al., 2023)**: physics-guided diffusion，不做scene reconstruction

CRISP的独特之处：**完整的real2sim pipeline，从video到RL训练的assets，中间用planar primitives和contact completion桥接**。

## 7. Code & Demo

项目主页：https://crisp-real2sim.github.io/CRISP-Real2Sim/

## 8. 参考资料

- VideoMimic: https://arxiv.org/abs/2505.03729
- MegaSAM: https://arxiv.org/abs/2412.04463
- GVHMR: https://research.nvidia.com/labs/dir/liva-siggraphasia2024/ (SIGGRAPH Asia 2024)
- MoGe: https://wangrc.site/MoGePage/ (CVPR 2025)
- InteractVLM: https://interactvlm.github.io/ (CVPR 2025)
- DeepMimic: https://xbpeng.github.io/projects/DeepMimic/index.html
- MaskedMimic: https://research.nvidia.com/labs/toronto-ai/maskedmimic/
- NKSR: https://huajieshao.com/NKSR/ (CVPR 2023)
- Isaac Gym: https://developer.nvidia.com/isaac-gym
- SMPL: https://smpl.is.tue.mpg.de/
- EMDB: https://ait.ethz.ch/projects/2023/EMDB/
- PROX: https://prox.is.tue.mpg.de/
- PLACE: https://prox.is.tue.mpg.de/ (3DV 2020)
- HIL: https://arxiv.org/abs/2505.12619
- HDMI: https://arxiv.org/abs/2509.16757
- SkillBlender: https://arxiv.org/abs/2506.09366

---

**总结**：CRISP是个engineering-heavy的paper，核心贡献不在算法novelty，而在于把多个existing components（MegaSAM, MoGe, GVHMR, InteractVLM, DeepMimic-style RL）组装成一个能work的real2sim pipeline，并发现"planar primitives + contact completion"是让simulation稳定的关键。最深刻的insight是：**reconstruction quality应该由downstream simulation task定义，而不是by Chamfer Distance**。这个insight对未来real2sim工作有指导意义。
