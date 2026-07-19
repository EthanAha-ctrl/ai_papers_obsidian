---
source_pdf: BimArt.pdf
paper_sha256: adab08e379918b3782885cb8088b16bffde641536a6d146b4b464117a9c7ef77
processed_at: '2026-07-18T18:44:04-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BimArt: 3D Bimanual Interaction with Articulated Objects 合成方法详解

## 1. Problem Setup 的精确形式化

任务的形式化定义如下。给定一个 two-part articulated object 的 trajectory $\xi = \{\xi_i\}_{i=1}^N$，每帧 $\xi_i = [\mathbf{g}_i | \mathbf{a}_i] \in \mathbb{R}^7$，其中：

- $\mathbf{g}_i \in \mathbb{R}^6$：object root 的 global translation (3D) + rotation (3D, axis-angle or 6D continuous representation)
- $\mathbf{a}_i \in \mathbb{R}$：articulation joint angle，单一旋转 DOF（如 laptop 的 hinge angle，scissors 的 spring angle）

目标输出是 bimanual MANO [52] parameters $\Theta = \{\Theta_i\}_{i=1}^N$，每帧 $\Theta_i \in \mathbb{R}^{61 \times 2}$：每只手 51 维 pose 参数（包含 root translation 3、root rotation 3、15 个关节 × 3 维 axis-angle = 45）+ 10 维 shape parameter $\beta$（共享给两帧但作者写作 61 应该是 pose only）。两边 hand 各 61。

这里的关键约束（constraints）：
1. **Contact feasibility**：手必须与 object 表面产生合理接触，不能悬空
2. **Penetration-free**：手 mesh 不能穿入 object mesh
3. **Articulation consistency**：当 $\mathbf{a}_i$ 变化时，手必须保持与 articulating part（如 laptop lid）接触来"驱动"它
4. **Temporal smoothness**：手运动不能 jitter
5. **Diversity**：同一 trajectory 应能采样出多种合理 grasp pattern

## 2. Prior Work 的关键 gap 分析

让我梳理 related work 的 design space，因为这对理解 BimArt 的 contribution 至关重要：

**Single-hand methods on rigid objects**：ManipNet [77]（https://arxiv.org/abs/2103.04519）、GOAL [60]、IMOS [16]、D-Grasp [9] —— 这些都不处理 articulation，且部分依赖 reference grasp。

**Articulated object methods 的具体限制**：

- **CAMS** [83]（CVPR 2023, https://arxiv.org/abs/2303.11124）：category-specific（per category train one model），unimanual，要求 initial grasp 作为输入。它的 "canonicalized manipulation spaces" 设计 stage-wise contact targets，在 dynamic bimanual 设定下 under-constrain MANO fitting。
- **ArtiGrasp** [79]（3DV 2024, https://arxiv.org/abs/2311.03385）：bimanual 但是分两阶段（先 grasp 再 articulate），不能 simultaneous，且要求 reference pose，依赖物理 simulator，object initial state 必须 supported by table。
- **Text2HOI** [5]（CVPR 2024）、**DiffH2O** [10]（SIGGRAPH Asia 2024）：text-conditioned，缺乏对 object trajectory 的 fine-grained control，对 artistic creation workflow 不友好。
- **OMOMO** [35]（SIGGRAPH 2023, https://arxiv.org/abs/2212.08358）：whole-body，无 finger articulation，stage-wise 设计在 hand-only 设定下 suboptimal。
- **GeneOH Diffusion** [39]（ICLR 2024）：denoising，要求 input motion 作为起点，不能 from scratch 生成。
- **GraspXL** [78]（ECCV 2024）：rigid objects only，大规模 grasp generation。

BimArt 的四个 relaxation 同时满足：
1. No reference grasp（不需要 initial 或 goal pose）
2. Unified model across categories（单 model 跨类别）
3. Simultaneous grasping + articulation + root motion
4. No coarse hand trajectory input

唯一 concurrent work 是 ManiDext [80]（https://arxiv.org/abs/2403.02113），但作者没有直接对比。

## 3. Pipeline 整体架构

BimArt 是一个 **three-stage cascade**：

```
Object trajectory ξ (N×7)
    │
    ├─→ Object Encoder E_o (MLP)  ← 输入: O (BPS), G (6D global), s_o (scale)
    │            │
    │            ▼
    │   Contact Generation Model (DDPM, transformer encoder, 50 steps)
    │            │
    │            ▼
    │   Bimanual Contact Maps C = [C^left, C^right]  (per-frame, per-hand)
    │
    ├─→ Object Encoder E_α (MLP)
    │            │
    │            ├──← Contact Encoder E_c (MLP) ← C
    │            │
    │            ▼
    │   Motion Generation Model M (DDPM, transformer encoder, 50 steps)
    │            │  + Classifier-Free Guidance (λ_f = 0.5, dropout p_f = 0.5)
    │            │  + Contact Map Guidance (λ_c adaptive)
    │            ▼
    │   Hand motion X = [H | D]  (positions + direction vectors to object)
    │
    ▼
Optimization-based MANO Fitting (100 iter)
    │  + l_proj (w_proj = 100)
    │  + l_pen  (w_pen = 10)
    │  + l_acc  (w_acc = 1000 or 10^4)
    ▼
Final 3D bimanual meshes (MANO Θ)
```

设计的核心 intuition 是 **decompose-and-conquer**：直接生成 hand motion 需要在 high-dimensional space 中同时满足 contact、articulation、physics constraints，难以学习。先学一个 low-dimensional、dense 的 contact prior（contact map 嵌在 object 表面），用它作为 information bottleneck 引导后续 motion 生成。这与 D-Grasp [9] 中 contact map 作为 optimization target 的思想类似，但 BimArt 把它放到 generative pipeline 里。

## 4. Object Representation: Normalized Part-Based BPS

这是 paper 的核心 technical contribution 之一。让我详细拆解。

### 4.1 原 BPS [51] 的回顾

Basis Point Sets (BPS) 来自 Prokudin et al. ICCV 2019（https://arxiv.org/abs/1904.08928）。给定固定 basis point set $\mathbf{B} \in \mathbb{R}^{K \times 3}$（通常从 unit sphere 均匀采样，K ≈ 几千），对任意 object point cloud $\mathbf{V}$，BPS feature 定义为：

$$\text{BPS}(\mathbf{V}) = [\text{argmin}_{\mathbf{v} \in \mathbf{V}} d(\mathbf{v}, \mathbf{b}) - \mathbf{b}, \text{ for } \mathbf{b} \in \mathbf{B}]$$

即每个 basis point $\mathbf{b}$ 找到 object 上最近的 vertex $\mathbf{v}^*$，记录差向量 $\mathbf{v}^* - \mathbf{b}$。得到固定维度 $K \times 3$ 的 feature，与 object vertex 数量无关，适合作为 MLP 输入。

### 4.2 BimArt 的 modification：part-aware + scale normalized

**问题 1**：articulated object 的两个 part 尺寸可能差异很大。例如 bottle + cap，cap 占 surface 5%。若用 standard BPS，cap 上 mapping 的 basis points 极少，几何细节丢失。

**问题 2**：不同 object scale 差异大，MLP 难以 generalize。

**BimArt 的方案**：

**Step 1: Canonicalization**。把 object transform 到 canonical frame，其中 articulation axis 与 $-z$ 轴对齐。设 canonical-to-world matrix 为 $\mathbf{M}$，则：
$$\mathbf{V}_i^o = \mathbf{M}^{-1} \mathbf{V}_i$$
$\mathbf{M}$ 由 $\mathbf{g}_i$ 决定。这样每帧 object 都在一致的 frame 下表达，disentangle 全局运动。

**Step 2: Scale normalization**（Eq. 1）：
$$s_o = \frac{1 - d_{\text{margin}}}{\max_{\mathbf{v} \in \mathbf{V}_{ao}} \|\mathbf{v}\|}$$

其中：
- $\mathbf{V}_{ao}$：object 在 articulation angle = $\pi/2$ 或 $\pi$（取决于 category，见 Appendix B）的"开放"状态的 vertex positions，选取一个能 maximize object extent 的 articulation 状态作为 reference
- $d_{\text{margin}} = 0.15$：margin，让 object 不触及 unit ball 边界
- 分子 $1 - d_{\text{margin}}$ = 0.85，即 normalized object 完全包含在半径 0.85 的 ball 内

直觉：这一步把所有 object 在 canonical frame 下 normalize 到大致相同的 size。**关键设计决策**：是 **whole-object scale normalization，不是 per-part scale normalization**。作者解释：因为 hand motion $\mathbf{H}$ 是用 original scale 编码的，若每个 part 各自归一化，模型难以 reasoning hand-object distance。这是一个非 trivial 的 design choice，反映了对 hand-object coupling 的考量。

**Step 3: Part-based BPS**（Eq. 2-3）：

对每个 part $p \in \{\text{top}, \text{bottom}\}$ 独立计算 BPS feature，用同一组 basis points $\mathbf{B}$：
$$\mathbf{O}_i^p = [\text{argmin}_{\mathbf{v} \in \mathbf{V}_i^p} d(\mathbf{v}/s_o, \mathbf{b}) - \mathbf{b}, \text{ for } \mathbf{b} \in \mathbf{B}]$$

- $\mathbf{V}_i^p$：frame $i$ 时 part $p$ 的 vertices（在 canonical frame 下）
- 除以 $s_o$：normalize 到 unit ball scale
- 找 nearest vertex 的过程在 normalized part 上做
- 输出：$\mathbf{O}_i^p \in \mathbb{R}^{K \times 3}$ per part

总 feature $\mathbf{O} = [\mathbf{O}_i^p, \text{for } i \in \{1,...,N\}, p \in \{\text{top}, \text{bottom}\}] \in \mathbb{R}^{N \times 2K \times 3}$

**关键 insight**：top part 和 bottom part **共享同一组 basis points $\mathbf{B}$**，意味着每个 part 获得相等的 feature 容量 $K \times 3$，无论其物理大小。这保证了 small articulating part（如 bottle cap）也能获得 dense geometric encoding。

### 4.3 为什么不只用 BPS，还需要 G？

BPS feature $\mathbf{O}$ 只编码 shape + articulation，**不包含 global trajectory 信息**。但 global movement 对 hand motion 至关重要：
- 重力方向：若 object 向下移动，hand 必须从下方支撑
- 整体平移：hand 必须 follow object

所以引入 $\mathbf{G} = [\mathbf{g}_i]_{i=1}^N$，每帧 6D：relative translation (to first frame, 3D) + global rotation (3D)。用 **relative** 而非 absolute 是为了避免 overfitting，让 model 对 absolute position invariant。

Table III (Supplementary) 量化了不同 BPS 策略对 contact map reconstruction 误差的影响：

| Strategy | Avg Error (cm) |
|---|---|
| U-BPS (unnormalized) | 0.554 |
| PA-BPS (normalized, part-agnostic) | 0.32 |
| P-BPS (normalized, part-based, ours) | 0.361 (whole), top part 0.258 |

虽然 P-BPS 整体平均略高于 PA-BPS（0.361 vs 0.32），但 **top part error 显著降低**（0.258 vs 0.342），这正是 articulated 部分，是 interaction 最 critical 的区域。bottom part error 略升（0.38 vs 0.341），因为 bottom 通常更大、原本 PA-BPS 已经够好。trade-off 在正确的方向上。

## 5. Hand Representation

这是另一个关键 design choice，不是直接用 MANO 参数 $\Theta$，也不是只用 skeleton joints，而是：

$$\mathbf{X} = [\mathbf{H} | \mathbf{D}]$$

- $\mathbf{H}_i \in \mathbb{R}^{J \times 3}$：从 MANO surface 上 sparse sample 的 $J$ 个 keypoints 的 3D 位置（不是 joints，是 surface vertices）
- $\mathbf{D}_i \in \mathbb{R}^{J \times 3}$：每个 keypoint 到 nearest object vertex 的方向向量（同时 encode 方向和距离）

为什么这样设计？三个原因：

1. **Dense 比 skeleton joints 更易回归**：MANO surface 有 778 vertices，skeleton 只有 15 joints。dense surface vertices 与 MANO 参数的 mapping 更直接，减少 inverse kinematics 的歧义。
2. **D 显式 encode hand-object geometric relationship**：让 model 直接 "看到" hand 离 object 多远、朝哪个方向。
3. **Canonical frame encoding**：所有都在 object canonical frame 下表达，$\mathbf{H}_i^o = \mathbf{M}^{-1} \mathbf{H}_i^w$，使 model 对 global object motion 有 invariance。

Ablation (Table 3) 对比 MANO-Rep（直接用 MANO 6D pose + joints）：
- Penetration: 22.2% vs 20.3%（NP-BPS ours 更好）
- Contact: 95.3% vs 98.5%
- Articulation: 82.0% vs 83.1%
- Multi-modality: 6.48 cm vs 6.91 cm（ours 更 diverse）

MANO-Rep 在 contact 和 articulation 上都差，因为 MANO pose parameters 与 object 几何没有直接对应关系，model 难以 reasoning contact。

## 6. Contact Generation Model

### 6.1 Contact Map 的定义

Eq. (4) 定义 contact map：
$$\mathbf{C}_i^\rho = [\text{argmin}_{\mathbf{h} \in \Xi_i^\rho} d(\mathbf{h}, \mathbf{v}) - \mathbf{v}, \text{for } \mathbf{v} \in \tilde{\mathbf{V}}_i], \rho \in \{\text{left, right}\}$$

变量解释：
- $\Xi_i^\rho$：hand $\rho$ 的 MANO surface vertices at frame $i$
- $\tilde{\mathbf{V}}_i = \mathbf{O}_i + \mathbf{B}$：BPS feature 通过加回 basis point 重建的 nearest object vertices（这是从 BPS feature 空间映射回 object 表面的"sparse anchor points"）
- $\rho$：left 或 right hand

所以 contact map 是**对每个 BPS-mapped object vertex**，记录**离它最近的 hand vertex 与它的 displacement vector**。这是 dense、signed、continuous 的 representation，**不指定 correspondence**（即不指定具体哪个 hand vertex 应该碰哪个 object vertex），给后续 generation 留下 diversity 空间。

注意：**左右手分开生成 contact map**（不是合并）。作者说这减少 ambiguity，因为合并的话 model 不知道哪个 region 该是哪只手。

### 6.2 Diffusion Model 架构

- **Backbone**：transformer encoder，8 层，latent dim 512
- **Diffusion**：DDPM [24]（https://arxiv.org/abs/2006.11239），50 步
- **Training objective**：直接 predict clean sample $\mathbf{C}_0$（不是 predict noise ε），这与某些 motion diffusion 工作一致（如 MDM [62] https://arxiv.org/abs/2209.14991）
- **Conditioning**：$\mathbf{O}$, $\mathbf{G}$, $s_o$ 通过 MLP encoder $\mathcal{E}_o$ 进入
- **EMA** 用于稳定训练
- **Adam**, lr = 1e-4, cosine schedule, 200 epochs

**输出维度**：每帧每手 $K \times 3$（与 BPS feature 同维），所以**与 object mesh resolution 无关**，这使 cross-category training 可行。这是 BimArt 能 unify 多个 category 的关键。

### 6.3 为什么用 contact map 作为 intermediate target？

Paper 强调三个优势 vs 直接 stage-wise contact points（如 CAMS）：
1. **Dense**：每帧几千个 contact value vs CAMS 的几个 stage-wise targets
2. **Continuous**：displacement vector 包含距离信息，不只是 binary contact label
3. **Diversity-friendly**：不指定 correspondence，允许同一 contact map 对应多种 grasp pattern

这与 ContactDB [2]、GenOH [39] 等用 thermal/depth contact map 的思想一脉相承，但首次用于 generative bimanual articulated setting。

## 7. Motion Generation Model

### 7.1 Conditioning 机制

- $\mathbf{O}$, $\mathbf{G}$, $s_o$ → MLP $\mathcal{E}_\alpha$ → $\mathbf{Z}_o \in \mathbb{R}^{N \times L_o}$
- $\mathbf{C}$ → MLP $\mathcal{E}_c$ → $\mathbf{Z}_c \in \mathbb{R}^{N \times L_c}$
- Concatenate：$\mathbf{Z} = [\mathbf{Z}_o | \mathbf{Z}_c]$
- $\mathcal{M}$（transformer encoder, 8 layers, 512 dim）作为 diffusion denoiser，同样 predict clean sample

### 7.2 Classifier-Free Guidance (CFG)

训练时以概率 $p_f = 0.5$ 把 $\mathbf{Z}_c$ 替换为 learnable null token $\emptyset$。这让 model 同时学 conditional 和 unconditional distribution。

采样时（Eq. 7）：
$$\tilde{\mathbf{X}}_{(t-1)} = (1 + \lambda_f) \tilde{\mathbf{X}}_{(t)} - \mathcal{M}(\hat{\mathbf{X}}^{(t)}, t, \mathbf{Z}_o, \emptyset)$$

这里 $\lambda_f = 0.5$。展开：
- $\tilde{\mathbf{X}}_{(t)} = \mathcal{M}(\hat{\mathbf{X}}^{(t)}, t, \mathbf{Z}_o, \mathbf{Z}_c)$（conditional prediction）
- 减去 unconditional prediction
- 乘以 $1 + \lambda_f$

这是 standard CFG formula。直觉：conditional signal 袊强时，model 在 contact conditioning 方向更"激进"。

### 7.3 Contact Map Guidance（关键创新）

CFG 在 sample level 提供 contact conditioning，但**不直接 enforce**生成 motion 的 derived contact map 与 generated contact map 一致。作者观察到 fine-grained artifacts（如手指卡在两个 part 之间），引入显式 guidance。

在每个 denoising timestep $t$，对 predicted clean hand $\hat{\mathbf{H}}_{(t)}^\rho$，重新计算 derived contact map（Eq. 5）：
$$\tilde{\mathbf{C}}_{(t)}^\rho = [\text{argmin}_{\mathbf{h} \in \hat{\mathbf{H}}_{(t)}^\rho} d(\mathbf{h}, \mathbf{v}) - \mathbf{v}, \text{for } \mathbf{v} \in \tilde{\mathbf{V}}]$$

用 **differentiable one-nearest-neighbor** 实现，保证梯度可传到 $\hat{\mathbf{H}}$。

Guidance update（Eq. 6）：
$$\tilde{\mathbf{X}}_{(t)}^\rho = \hat{\mathbf{X}}_{(t)}^\rho - \lambda_c \nabla_{\hat{\mathbf{X}}_{(t)}^\rho} \|\hat{\mathbf{C}}^\rho - \tilde{\mathbf{C}}_{(t)}^\rho\|$$

- $\hat{\mathbf{C}}^\rho$：contact model 输出（fixed target）
- $\tilde{\mathbf{C}}_{(t)}^\rho$：current motion 推导出的 contact map
- $\lambda_c$：adaptive，取 $\lambda_c = 1 / \|\nabla \hat{\mathbf{X}}_{(t)}\|$，即按梯度 norm 归一化，避免 guidance 过强破坏 sample

直觉：这是 **diffusion guidance** 的一种 instance，类似 classifier guidance [55]（https://arxiv.org/abs/1907.05600），但 classifier 是 differentiable contact map discrepancy。与 score distillation sampling [68] 也有点像。

### 7.4 Ablation 量化（Table 3 "Contact" 部分）

| Variant | Mul | Pen 1cm | Con | Art | CM (cm) |
|---|---|---|---|---|---|
| w/o C (no contact) | 4.49 | 9.45 | 96.13 | 79.59 | – |
| w C (with contact cond) | 6.98 | 20.27 | 98.48 | 83.09 | 1.15 |
| w C + CG (+guidance) | 6.96 | 16.50 | 97.35 | 84.23 | 1.13 |
| w C + CG + Opt (full) | 6.91 | 2.03 | 99.63 | 85.57 | 1.18 |

观察：
- Contact conditioning 显著提升 **multi-modality**（4.49 → 6.98 cm），因为 contact map 携带多种合理 grasp pattern 的 information
- Contact conditioning 反而**增加 penetration**（9.45 → 20.27%）：因为 model 更激进地 push hand toward object，导致 finger 卡在 part 之间
- Contact guidance 略降 penetration（20.27 → 16.50）并降 CM discrepancy
- Optimization 是 penetration 的真正杀手：20.27 → 2.03%，**降 10x**

## 8. Post-Processing Optimization

### 8.1 MANO Fitting（Eq. 8）

先 estimate $\Theta = [\theta | \beta]$ 从 sparse surface keypoints $\hat{\mathbf{H}}$：
$$l_{\text{MANO}} = \|\hat{\mathbf{H}} - f_{\text{MANO}}(\theta, \beta)\|$$

- $\theta \in \mathbb{R}^{N \times 51 \times 2}$：每只手每帧 51 维 pose（root 6 + 15 joints × 3 axis-angle = 45）
- $\beta \in \mathbb{R}^{10 \times 2}$：每只手 10 维 shape（共享 across frames）
- $f_{\text{MANO}}$：MANO forward pass，输出 surface vertices

这是 inverse problem：从 sparse surface keypoint 反求 MANO 参数。Sparse → dense 通过 MANO model 的 shape prior 隐式完成。

### 8.2 Refinement（Eq. 9-12）

总 loss：
$$l_{\text{reg}} = w_{\text{proj}} l_{\text{proj}} + w_{\text{pen}} l_{\text{pen}} + w_{\text{acc}} l_{\text{acc}}$$

**Projection loss**（Eq. 10）：
$$l_{\text{proj}} = \sum_{\mathbf{p} \in P} \min_{\mathbf{v} \in \mathbf{V}} \|\mathbf{p} - \mathbf{v}\|$$
- $P = f_{\text{MANO}}(\theta, \beta) + \hat{\mathbf{D}}$：把 MANO-fitted hand surface vertices 加上 predicted direction vectors，得到 **"应该接触 object 表面" 的投影点**
- 让 $P$ 落在 object surface 上

直觉：$\hat{\mathbf{D}}$ 是 model 对 "hand 朝 object 哪个方向、多远" 的 prediction。这个 loss 把这个 prediction 当作 contact target，让 MANO fit 满足这个 spatial relationship。这解决 floating artifact（手离 object 太远）。

**Penetration loss**（Eq. 11）：
$$l_{\text{pen}} = \sum_{\mathbf{h} \in \text{Int}(\hat{\Xi})} \min_{\mathbf{v} \in \mathbf{V}} \|\mathbf{h} - \mathbf{v}\|$$
- $\hat{\Xi}$：dense MANO surface vertices after fitting
- $\text{Int}(\hat{\Xi})$：在 object 内部的 hand vertices 集合
- 把这些 vertices push 到最近的 object surface

这是 Hasson et al. [21]（https://arxiv.org/abs/1905.04308）的经典 penetration loss。

**Acceleration loss**（Eq. 12）：
$$l_{\text{acc}} = \sum_{\mathbf{h}_i \in \hat{\Xi}} \|\mathbf{h}_i - 2 \mathbf{h}_{i-1} + \mathbf{h}_{i-2}\|$$

这是二阶差分（second-order finite difference），近似 acceleration。惩罚 jitter。

### 8.3 Hyperparameters 与 Sensitivity

- $w_{\text{proj}} = 100$
- $w_{\text{pen}} = 10$
- $w_{\text{acc}} = 1000$（ARCTIC）/ $10^4$（HOI4D，因 HOI4D motion 更 jittery）
- 100 iterations

Appendix Fig. I 显示 sensitivity 分析：扰动 ±25% 看 metric 变化。**Trade-off**：$w_{\text{acc}}$ ↑ → motion 更 smooth，但 penetration ↑；$w_{\text{pen}}$ ↑ → penetration ↓，但 jitter ↑。这是 multi-objective optimization 的经典 Pareto front。

## 9. Experiments 深度分析

### 9.1 Datasets

**ARCTIC** [14]（CVPR 2023, https://arxiv.org/abs/2303.17912）：
- 11 articulated objects（laptop, scissors, microwave, box, ketchup, mixer, waffle iron, capsule machine, notebook, phone, espresso machine）
- 257 train sequences / 44 test sequences
- Bimanual, fully annotated MANO + object mesh per frame

**HOI4D** [40]（CVPR 2022, https://arxiv.org/abs/2111.06037）：
- 4D egocentric dataset，包含 articulated objects
- BimArt 用 pliers 和 scissors 两个 category，遵循 CAMS [83] 的 protocol

### 9.2 Metrics 解释

- **Mul (cm)**：multi-modality，对同一 trajectory 采样 10 次，计算所有 hand vertex 的 pairwise average distance。**越大越 diverse**。
- **Accel**：hand vertex acceleration 平均值，**越小越 smooth**
- **Pen 1cm (%)**：penetration frame percentage，threshold 1cm（也提供 5mm threshold 见 Table II）
- **Con (%)**：contact frame percentage
- **Art (%)**：在 articulation 发生 frame 中，hand 与 articulating part 接触的 frame percentage
- **CM (cm)**：contact map L1 discrepancy，只用于 ablation（因为 baseline 没有显式 contact map）

### 9.3 Main Results（ARCTIC, Table 1）

| Method | Mul ↑ | Accel ↓ | Pen 1cm ↓ | Con ↑ | Art ↑ |
|---|---|---|---|---|---|
| GT | – | 0.178 | 1.04 | 95.14 | 94.56 |
| CAMS-B | 8.56 | 0.120 | 42.52 | 98.92 | 76.70 |
| MDM-B | 0.55 | 0.277 | 66.71 | 93.66 | 73.73 |
| OMOMO-B | 0.038 | 0.197 | 30.44 | 96.92 | 80.09 |
| **Ours** | **6.91** | 0.188 | **2.03** | **99.63** | **85.57** |

关键观察：
1. **Penetration 2.03%**，比 OMOMO-B（30.44%）降 15x，比 CAMS-B（42.52%）降 21x。这是 optimization 的功劳。
2. **Multi-modality 6.91**，远超 MDM-B（0.55）和 OMOMO-B（0.038）。CAMS-B 略高（8.56）但代价是高 penetration（42.52%），即"假 diversity"——可能就是随机 jitter。
3. **Contact 99.63%** 比 GT 还高（95.14%）——这意味着 BimArt 过度 contact（hand 总是贴着 object），可能因为 contact guidance 太激进。但 qualitative 显示这没问题，因为 GT 也有 approach 阶段，那段时间没有 contact。

### 9.4 HOI4D Results（Table 2）

在 cross-category unified 设定下，BimArt 与 CAMS-X 对比：

| Method | Pliers Pen | Pliers Con | Pliers Art | Scissors Pen | Scissors Con | Scissors Art |
|---|---|---|---|---|---|---|
| CAMS-X | 0.017 | 0.485 | 0.015 | 0.198 | 0.858 | 0.167 |
| Ours w/o opt | 0.044 | 0.966 | 0.597 | 0.591 | 1.000 | 0.853 |
| Ours w/ opt | 0.464 | 0.870 | 0.595 | 1.204 | 1.000 | 0.887 |

CAMS-X 几乎完全失败（Art 0.015, 0.167），证明 cross-category 设定对 CAMS 不可行。BimArt 在 unified 设定下仍能达到合理的 articulation consistency（0.595, 0.887）。

值得注意：在 category-specific 设定下，CAMS 仍很强（Pliers Art 1.000），但这是 per-category train 的结果，不 fair 比较。

### 9.5 Perceptual User Study

55 个 participants，40 对 animations，5 objects × 4 trajectories。Force-choice：which is more natural。p < 1e-3 for all baselines。BimArt 在每个 object 上都被 preferred。

这类 user study 在 motion synthesis 论文中越来越重要，因为 quantitative metrics（特别是 penetration、contact）不一定 capture "naturalness"。Karpathy 你在 Stanford CS231n 也强调过 eval metrics 与人类感知的 gap 问题。

### 9.6 Overfitting Check（Supplementary）

作者证明不是 overfitting：对每个 test sequence，在 training set 中找 5 nearest neighbors（基于 object motion），计算 hand vertex 平均距离 = 15.08 cm，object vertex 距离 = 4.40 cm。**hand motion 显著不同**，证明 model 在 generate，不是 retrieve。

## 10. Architecture 图解析（Figure 2）

让我详细读 Figure 2 的信息流：

```
Input: N × 7 (object trajectory)
            │
            ├─→ extract O (N × 2K × 3, BPS)
            ├─→ extract G (N × 6)
            └─→ compute s_o (1, scalar)
            
            [Contact Model]
            O, G, s_o ──→ E_o (MLP) ──→ conditioning
                                            │
            Noise ~ N(0,I) ──→ DDPM (transformer, 50 steps) ──→ C (N × 2 × K × 3)
            
            [Motion Model]  
            O, G, s_o ──→ E_α (MLP) ──→ Z_o (N × L_o)
                                            │
            C ──→ E_c (MLP) ──→ Z_c (N × L_c)  ← dropout to ∅ w.p. 0.5
                                            │
            Z = [Z_o | Z_c]
                                            │
            Noise ~ N(0,I) ──→ DDPM (transformer, 50 steps, + CFG + Contact Guidance) ──→ X (N × 2 × J × 6)
                                                                                                │
                                                                                                ▼
            Optimization (100 iter): l_MANO + l_proj + l_pen + l_acc ──→ Θ (MANO params)
```

"Spiral" 在图里表示 iterative denoising process。

## 11. Limitations 与 Future Work

作者承认：
1. **Limited categories**：只 train 在 ARCTIC (11 类) + HOI4D (2 类)，real-world 应用要求 open-vocabulary generalization。
2. **No physics simulation**：purely kinematic + optimization-based，没有 simulator 验证 dynamic feasibility（如 object 是否会掉）。这与 D-Grasp [9]、InterDiff [70]（https://arxiv.org/abs/2308.05357）的 physics-informed diffusion 思路不同。
3. **Slow sampling**：50 步 DDPM + 100 步 optimization，未用 DDIM [56]（https://arxiv.org/abs/2010.02502）或 Latent Diffusion [7] 加速。

作者 propose future direction：
- 用 multimodal LLM（如 3D-LLM [26] https://arxiv.org/abs/2307.12981）做 open-vocabulary generalization
- DDIM / LDM 加速

## 12. 我的联想与 broader context

### 12.1 与 diffusion guidance 文献的关系

BimArt 的 contact guidance 是 **classifier guidance** 的 instance。Ho & Salimans [23]（https://arxiv.org/abs/2207.12568）的 classifier-free guidance 是 implicit，但 BimArt 又叠加了显式 differentiable guidance。这种 hybrid 在 image generation 中少见（通常二选一），但在 motion generation 中可能更适用，因为 motion 有 hard physical constraints（contact、penetration）。

类似思想在 **score distillation sampling (SDS)** [68]、**D-Grasp** [9] 中也出现。D-Grasp 用 contact map 作为 optimization target 在 grasp synthesis 中，但不是 generative。BimArt 把它 generative 化了。

### 12.2 与 manipulation policy learning 的关系

虽然 BimArt 是 animation 工作，与 robotics 的 dexterous manipulation 有 deep connection：
- **Diffusion Policy** [8]（RSS 2023, https://arxiv.org/abs/2303.04137）：用 diffusion 学 visuomotor policy
- **DexGraspNet** [69]、**UniDexGrasp++** [66]：大规模 dexterous grasp 数据
- **CyberDemo** [67]：simulated human demonstration for dexterous manipulation

BimArt 的输出（多样化的合理 bimanual trajectory）可以作为 **demonstration 数据** for imitation learning。特别是 articulated object manipulation（如剪刀、瓶盖）在 robotics 中仍是 open problem。

### 12.3 与 LLM-grounded 3D generation 的关系

作者提到 3D-LLM [26] 作为 future direction。这与近期趋势一致：
- **GenZI** [37]（CVPR 2024）：zero-shot 3D human-scene interaction generation
- **HOI-Diff** [48]：text-driven 3D HOI

如果 BimArt 加 text conditioning（"open the laptop with left hand on the lid"），并接 LLM 的 common-sense（哪些 grasp pattern 合理），可以扩展到 open-vocabulary。这可能是 next paper。

### 12.4 与 motion diffusion backbone 的关系

BimArt 用 transformer encoder 作为 denoiser，与 MDM [62] 类似。但近期 motion generation 倾向 **latent diffusion**：
- **MLD** [7]（CVPR 2023）：latent diffusion for motion
- **MotionLCM**（CVPR 2024）：latent consistency model，real-time

BimArt 在 feature space（H + D）diffuse，不是在 raw MANO parameter space。这与 latent diffusion 思想类似，但 latent 是 hand-crafted feature，不是 learned VAE latent。若加 VAE，可能 sample quality 更高、速度更快。

### 12.5 与 recent bimanual methods 的对比

- **MACS** [54]（3DV 2024）：mass-conditioned 3D hand + object，rigid only，sphere training
- **InterHandGen** [32]（CVPR 2024）：two-hand interaction generation via cascaded reverse diffusion，但无 object
- **PhysHOI** [4]（3DV 2024）：physically plausible full-body HOI，但与 BimArt 不直接竞争

BimArt 是目前 articulated bimanual 物体上最完整的 generative solution。

### 12.6 与 generative grasp 文献的关系

- **GraspTTA** [29]（ICCV 2021）：hand-object contact consistency reasoning
- **ContactGen** [38]（CVPR 2023）：generative contact modeling for grasp
- **G-HOP** [73]（CVPR 2024）：generative hand-object prior

BimArt 与这些工作的关键区别：那些都是 **static grasp**，BimArt 是 **temporal motion**。把 static grasp prior 升级为 dynamic motion prior 是 BimArt 的核心。

## 13. 总结性直觉

BimArt 的核心 thesis 可以压缩为一句话：**"在 high-dimensional bimanual articulated interaction 空间中，dense continuous contact map embedded on object surface 是一个有效的 information bottleneck，能将 diversity-friendly generation 与 constraint-aware refinement 解耦。"**

具体地：
1. **Generation stage** 用 diffusion 的 stochasticity 提供 diversity，contact conditioning 把 diversity 限制在合理 grasp pattern 上
2. **Refinement stage** 用 optimization 的 determinism 满足 hard physical constraints（penetration、smoothness）
3. **Intermediate representation**（contact map）让两个 stage 解耦，各自专注自己擅长的事

这种 "generate then refine" 范式与 RRT + IK in robotics、diffusion + guidance in image generation 一脉相承，但首次完整地用在 bimanual articulated HOI 上。

## References

- Project page: https://vcai.mpiinf.mpg.de/projects/bimart/
- ARCTIC dataset: https://bimanual.github.io/
- HOI4D dataset: https://hoi4d.github.io/
- MANO model: https://mano.is.tuebingen.mpg.de/
- BPS original: https://arxiv.org/abs/1904.08928
- DDPM: https://arxiv.org/abs/2006.11239
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12568
- MDM: https://arxiv.org/abs/2209.14991
- CAMS: https://arxiv.org/abs/2303.11124
- ArtiGrasp: https://arxiv.org/abs/2311.03385
- OMOMO: https://arxiv.org/abs/2212.08358
- D-Grasp: https://arxiv.org/abs/2204.04278
- InterDiff: https://arxiv.org/abs/2308.05357
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Hasson penetration loss: https://arxiv.org/abs/1905.04308
- DDIM: https://arxiv.org/abs/2010.02502
- 3D-LLM: https://arxiv.org/abs/2307.12981
- ManiDext (concurrent): https://arxiv.org/abs/2403.02113
- GeneOH Diffusion: https://arxiv.org/abs/2312.17110
- Text2HOI: https://arxiv.org/abs/2312.06873
- GraspXL: https://arxiv.org/abs/2409.18131
- Vector Heat method: https://arxiv.org/abs/1905.11431 (used for contact map densification)

希望这些细节对你的 intuition building 有帮助。如果你想深入某个部分（比如 differentiable nearest neighbor 的实现细节、或 BPS sampling 的几何分析、或 optimization 中的 trade-off），可以再问。
