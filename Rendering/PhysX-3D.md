---
source_pdf: PhysX-3D.pdf
paper_sha256: a930068229e1c390e4e61f911cdc66a35ae1b5d0e868e752a37e85871b8da41d
processed_at: '2026-08-06T03:42:32-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PhysX-3D 用人话说

## 一句话说清楚

现在AI能从一张图生成3D模型了，但生成出来的东西**好看但没用**——没有重量、没有材质、关节不会动，机器人根本没法拿去训练。PhysX-3D干的事就是让生成的3D模型**自带物理属性**，直接能扔进simulator里用。

---

## 问题出在哪

你让现在的model生成一把椅子，它会给你geometry和texture，看起来像椅子。但你问它：

- 这把椅子多高？多重？——不知道
- 哪个部分是木头哪个是金属？——不知道
- 人会坐哪里、手会扶哪里？——不知道
- 靠背能不能往后调？旋转轴在哪？——不知道

对game dev来说这够了，对robotics来说这就是垃圾。机器人要学会开抽屉，你给它一个没有joint axis、没有friction的3D模型，它学个屁。

---

## 他们怎么解决的

### 数据层面：PhysXNet

现有dataset要么没有physics，要么只有object-level的physics。他们基于PartNet的26K模型，用GPT-4o + 人工校对，给每个part标了五个维度的信息：

- **有多大**（absolute scale）
- **什么材质**（material + density + Young's modulus）
- **人会碰哪里**（affordance rank 1-10）
- **怎么动**（joint type + axis + range + parent/child关系）
- **干嘛用的**（三段文字描述）

这五个维度不是随便选的，它们正好覆盖了经典力学算gravity、friction、contact force所需的输入。有了这些，simulator就能跑起来。

annotation pipeline的核心trick是：把每个part单独render成一张图喂给GPT-4o，target part涂红色，其他涂灰色。这样VLM不会被occlusion干扰。kinematic参数（joint axis位置、运动范围）因为GPT-4o算不准，所以人工用plane fitting + k-means candidate + 人工选择的方式确定。

### 规模层面：PhysXNet-XL

26K不够train生成模型，所以他们写了procedural generation规则，把抽屉、门这些模块化part插到不同object上，自动合成出6M个physically plausible的3D对象。

### 生成层面：PhysXGen

这是核心。他们没从scratch train，而是拿TRELLIS的pretrained model当backbone，加了一个并行的physical branch。

**关键设计**：geometry和physics不是独立预测的，而是通过residual connection互相影响。直觉上很合理——把手的position决定了affordance，joint的位置和geometry紧密相关，材质和geometry的appearance也有correlation。如果完全independent预测，就会丢失这些correlation。

具体做法是：
- 一个VAE encode geometry+appearance latent（继承TRELLIS）
- 一个VAE encode physical latent（新train的）
- 两个latent都是 $N \times 8$ 的结构，方便在同一个spatial grid上做interaction
- Diffusion阶段用dual-branch + skip connection，让geometry branch的特征flow到physical branch
- 都用Conditional Flow Matching（CFM）而不是DDPM，因为CFM的velocity field对continuous physical properties更友好

---

## 效果如何

比独立预测物理属性的baseline，absolute scale误差降了42%，affordance提升56%，kinematic coverage提升37%。而且geometry质量也小幅提升——说明physics priors对geometry有正则化作用，不是trade-off而是mutual benefit。

唯一退步的是function description，因为CLIP是encoder-only不可逆，26K数据量不够train好embedding到text的逆向mapping。GPT-4o做后处理的baseline在这项上反而更好。

---

## 为什么这件事重要

之前的3D generation追求的是"看起来对"，PhysX-3D追求的是"用起来对"。对embodied AI来说，这意味着一条新的sim2real路径：不是靠domain randomization硬transfer，而是直接生成physics-ready的training data。

类比一下：这就像从"生成图片"进化到"生成可执行的代码"。图片只能看，代码能跑。PhysX-3D生成的3D asset能直接import到Isaac Sim里跑simulation，这对robotics的data pipeline是一个质的飞跃。

---

## 还有什么没做好

- **Function description**：CLIP bottleneck没解决，需要T5这种decoder或者更好的VLM
- **Kinematics的hierarchical结构**：用regression预测parent-child关系会引入artifacts，但classification又因为part数量不固定难做
- **Long-tail scale**：1cm到1000cm的跨度，normalization策略都不理想
- **只做static physics**：没有dynamic friction、damping这些时变参数
- **single object**：没有object之间的physical interaction

---

## 我的take

这篇工作的价值不在model architecture多novel——dual-branch + residual connection是很直觉的设计。真正的贡献是**把physical properties提升为3D generation的一等公民**，并且用26K+6M的数据规模把这个idea落地了。

之前community一直在geometry quality上卷PSNR、Chamfer Distance，PhysX-3D指出了一个新方向：physical plausibility可能比geometry fidelity对下游应用更重要。这个方向的未来空间很大——加上dynamics、加上scene-level interaction、加上real-world scan grounding，每一项都是一篇paper。

相关链接：
- 项目主页：https://physx-3d.github.io/
- TRELLIS（base model）：https://arxiv.org/abs/2412.01506
- PartNet（dataset base）：https://arxiv.org/abs/1812.02758
- SAPIEN（simulator）：https://sapien.ucsd.edu/

---

# PhysX-3D: Physical-Grounded 3D Asset Generation 深度解析

## 1. 高层直觉

传统3D generation的pipeline输出的是"看得见但摸不着"的assets——几何好看、texture漂亮，但是没有任何physical priors。对于robotics和embodied AI来说，一个没有mass、没有friction coefficient、没有joint axis的3D model基本上没有sim2real transfer价值。PhysX-3D这篇工作的核心insight是：**physics不应该作为3D generation的后处理，而应该作为latent space的一等公民与geometry共同生成**。

这篇paper的核心可分解为三件事：
- **PhysXNet**: 26K human-in-the-loop annotated + 6M procedurally generated的物理3D数据集
- **PhysXGen**: 基于TRELLIS [[10]](https://arxiv.org/abs/2412.01506) 的dual-branch feed-forward image-to-3D generator
- **Annotation pipeline**: GPT-4o + human refinement的半自动标注框架

项目主页: [https://physx-3d.github.io/](https://physx-3d.github.io/)

---

## 2. 五维物理属性的设计哲学

PhysXNet把object properties组织成三阶段递进：

| Stage | 阶段 | 物理量 |
|-------|------|--------|
| a) Identification | 物体的基本存在 | absolute scale, material (name, Young's modulus E, Poisson's ratio ν, density ρ) |
| b) Function | 物体的潜在应用 | affordance priority (1-10 rank), function descriptions (basic, functional, kinematical) |
| c) Operation | 详细使用方式 | kinematic parameters |

### 2.1 为什么选这五个维度

这里的设计逻辑与经典mechanics的对应关系：

- **Absolute scale** → 决定gravitational force $F_g = mg = \rho V g$, 所以scale缺失就无法计算weight
- **Material** (含Young's modulus $E$ 和Poisson's ratio $\nu$) → 决定elastic deformation: $\sigma = E \epsilon$, 以及Hooke's law的 $F = -kx$ 中 $k = \frac{EA}{L}$
- **Affordance** → 引入Gibson affordance theory到3D part-level reasoning
- **Kinematics** → joint type决定DOF, 与SAPIEN [[5]](https://sapien.ucsd.edu/) / PartNet-Mobility兼容
- **Function descriptions** → semantic grounding, 让VLM能readable的属性

### 2.2 Kinematic type的5+1设计

```
A. No movement constraints (如 bottle 中的 water — 流体)
B. Prismatic joints (如 drawer — 平移)
C. Revolute joints (如 laptop — 绕轴旋转)
D. Hinge joint (如 shower hose — 绕点旋转)
E. Rigid joint (固接)
CB. Revolute + Prismatic (如 bottle cap — 螺旋)
```

这与URDF/MuJoCo/Isaac Sim中joint类型的primitive set对齐，使得生成的assets可以直接import到simulator里。值得注意的是type A处理fluid的方式很聪明——把"无约束"显式建模而不是忽略。

---

## 3. PhysXNet Dataset统计与设计

### 3.1 关键数据

- **26K objects** in PhysXNet (基于PartNet [[6]](https://arxiv.org/abs/1812.02758) 扩展)
- **6M objects** in PhysXNet-XL (procedurally generated)
- 平均**~5 parts per object**, 长尾分布 (Figure 3a)
- 24K train / 1K val / 1K test split
- Density标准化为 $g/cm^3$
- Length-Width-Height跨度3个数量级 (1-1000 cm, 主要集中在 <300 cm)

### 3.2 与已有dataset对比 (Table 1)

| Dataset | #Objs | Part | Phys Dim | Material | Affordance | Kinematic | Desc |
|---------|-------|------|---------|----------|------------|-----------|------|
| ShapeNet [[1]](https://shapenet.org/) | 51K | × | × | × | × | × | × |
| PartNet [[6]](https://arxiv.org/abs/1812.02758) | 26K | ✓ | × | × | × | × | × |
| PartNet-Mobility [[5]](https://sapien.ucsd.edu/) | 2.7K | ✓ | × | × | × | ✓ | × |
| GAPartNet [[8]](https://arxiv.org/abs/2211.00360) | 1.1K | ✓ | × | × | × | ✓ | × |
| ABO [[7]](https://arxiv.org/abs/2110.06149) | 7.9K | × | ✓ | Obj-level | × | × | Obj-level |
| Objaverse [[2]](https://objaverse.allenai.org/) | 818K | × | × | × | × | × | × |
| OmniObject3D [[4]](https://omniobject3d.github.io/) | 6K | × | × | × | × | × | × |
| **PhysXNet** | **26K** | ✓ | ✓ | **Part-level** | ✓ | ✓ | **Part-level** |
| **PhysXNet-XL** | **6M** | ✓ | ✓ | **Part-level** | ✓ | ✓ | **Part-level** |

关键insight：ABO虽然有material，但是object-level的，无法支撑robotic manipulation这类part-aware任务。

---

## 4. Human-in-the-Loop Annotation Pipeline

这是整个工作的工程核心。pipeline分两个phase：

### 4.1 Phase 1: Preliminary Data Acquisition

用GPT-4o处理rendered images，输出basic info（material, density, descriptions等）。这里有两个关键设计决策：

**Part-based vs Segmentation-based annotation**：
- Segmentation-based：multi-view projective rendering，但occlusion严重
- Part-based：每个component单独render一张图，准确但expensive

为了平衡，他们引入了preprocessing pipeline：
```
1. Normalize 3D coords to [-1, 1] (proportional scaling + translation)
2. Geometric simplification:
   - Merge if area ≤ 0.2 (绝对阈值)
   - OR merge if (face count ≤ 100 AND area ≤ 0.06) (双重小part阈值)
3. Part-based rendering with alpha compositing (target part红色, others灰色)
```

这个merge策略实际上是在做topology pruning — 把PartNet里过度细分的微小part合并掉。这与PartField [[34]](https://arxiv.org/abs/2504.11451) 的part segmentation思路形成对比。

### 4.2 Phase 2: Kinematic Parameter Determination

四个subtask：

**(2.a) Contact region计算**：
给定child mesh点云 $P_c$ 和parent mesh点云 $P_p$，计算Euclidean distance $d(P_c^i, P_p^j) = \|P_c^i - P_p^j\|_2$，然后用预设阈值 $\tau$ 过滤：
$$\text{Contact}(P_c, P_p) = \{(p_c, p_p) : \|p_c - p_p\|_2 < \tau\}$$

**(2.b) Plane fitting**：
对contact region点云做plane fitting (RANSAC风格)。对于prismatic joint (B), the运动方向应该在fitted plane的法向上。

**(2.c) Candidate generation**：
- 对type B (prismatic)：在fitted plane上uniformly sample axes作为候选方向
- 对type C (revolute)：在contact region上运行k-means聚类，每个cluster center作为rotation axis的候选位置

**(2.d) Kinematic parameters**：
人工选择最佳candidate，然后确定具体的rotation/movement range。

### 4.3 System Prompt设计 (Listing 1)

prompt的核心设计是**global-to-local** reasoning：先确定object-level (name, category, dimension)，再对每个part逐一分析。每个part的annotation包含：
- material + density
- affordance priority rank (1-10, 1是最可能被touch)
- neighbors及其movement type (A/B/C/D/E)
- parent/child relationship (if B/C/D)
- 4种description (basic, functional, movement, grasped)

GPT-4o的标注成本超过**$1k**，这是dataset规模限制在26K的原因之一。

---

## 5. PhysXNet-XL: Procedural Generation

为了突破26K的data scarcity，作者设计了procedural generation规则。

### 5.1 两类组合

- **Intra-category combination** (类别内组合): cabinet, bottle, faucet, chair, oven, shower, knife, table, laptop
- **Cross-category combination** (跨类别组合): drawer + door作为模块化component

### 5.2 Workflow (Figure 8)

1. 选取base object和target part (typically similar physical properties)
2. 识别connected regions
3. 适配new component的scale以匹配base structure的geometry
4. 最终合成新的physical 3D object

这其实是把programmatic scene composition的思想 [[13]](https://arxiv.org/abs/2508.15228) 应用到part level，最终得到6M objects。

---

## 6. PhysXGen架构深度解析

### 6.1 整体设计哲学

核心idea：**physics和geometry共享latent space但通过residual interaction**。

为什么不train from scratch？因为26K规模不够train SOTA generative models。所以adopt TRELLIS [[10]](https://arxiv.org/abs/2412.01506) 的pretrained geometry+appearance latent space，然后插入physical branch。

### 6.2 Physical 3D VAE Encoding

**输入特征组织**：

对于N个voxel (N是voxel grid分辨率):

| 特征 | 维度 | 说明 |
|------|------|------|
| $P_{dim}$ | $\mathbb{R}^{N \times 1}$ | physical scaling (从dimension转换) |
| $P_{aff}$ | $\mathbb{R}^{N \times 1}$ | affordance priority |
| $P_\rho$ | $\mathbb{R}^{N \times 1}$ | density |
| $P_{mov}$ | $\mathbb{R}^{N \times 11}$ | kinematic params |
| $P_{phy}$ | $\mathbb{R}^{N \times 14}$ | channel-wise concat of above |
| $P_{sem}$ | $\mathbb{R}^{N \times 768 \times 3}$ | CLIP [[32]](https://arxiv.org/abs/2103.00020) embeddings × 3 (basic, functional, kinematic) |
| $P_{aes}$ | $\mathbb{R}^{N \times 1024}$ | DINOv2 structural features |

**Kinematic params $P_{mov}$ 的11维分解**：
- child group index: $\mathbb{R}^{N \times 1}$
- parent group index: $\mathbb{R}^{N \times 1}$
- movement direction: $\mathbb{R}^{N \times 3}$ (3D vector)
- movement location: $\mathbb{R}^{N \times 3}$ (3D position)
- movement range: $\mathbb{R}^{N \times 2}$ (min, max)
- kinematic type: $\mathbb{R}^{N \times 1}$ (A/B/C/D/E/CB的one-hot或scalar)

$1+1+3+3+2+1 = 11$ ✓

### 6.3 VAE Encoding公式 (Eq. 1)

$$P_{plat} = \mathcal{E}_{phy}(P_{phy}, P_{sem}), \quad P_{slat} = \mathcal{E}_{aes}(P_{aes})$$

其中：
- $P_{plat} \in \mathbb{R}^{N \times 8}$: physical latent (8 channels)
- $P_{slat} \in \mathbb{R}^{N \times 8}$: structural latent (8 channels, 继承TRELLIS)
- $\mathcal{E}_{phy}$: physical VAE encoder
- $\mathcal{E}_{aes}$: aesthetic VAE encoder (TRELLIS pretrained)

注意：physical latent和structural latent都是8 channels，这个对称设计很关键——它让后续的dual-branch diffusion可以在同一spatial resolution上做interaction。

### 6.4 VAE Loss (Eq. 2)

$$\mathcal{L}_{vae} = \mathcal{L}_{aes}^{color} + \mathcal{L}_{aes}^{geometry} + \mathcal{L}_{phy} + \mathcal{L}_{sem} + \mathcal{L}_{kl} + \mathcal{L}_{reg}$$

各项详解：

- $\mathcal{L}_{aes}^{color}$: L2 loss + LPIPS loss (perceptual)
- $\mathcal{L}_{aes}^{geometry}$: mask loss + normal loss + depth loss
- $\mathcal{L}_{phy}$: normalized L2 loss on physical properties
- $\mathcal{L}_{sem}$: normalized L2 loss on CLIP embeddings
- $\mathcal{L}_{kl}$: KL divergence约束 $P_{plat}$ 的分布
- $\mathcal{L}_{reg}$: 减少textured mesh的不必要结构

### 6.5 Dual-Branch Architecture的关键设计

作者特别提到："a branch from $\mathcal{D}_{phy} \circ \mathcal{D}_{aes}$ via a residual connection"——即physical decoder和aesthetic decoder之间存在residual path，让structural features可以作为physical prediction的prior。

这个设计避免了independent decoder的弊端：如果physical properties完全独立预测，就会丢失"把手通常在抽屉前部"这种spatial correlation。

### 6.6 Physical Latent Generation (Diffusion)

采用**Conditional Flow Matching (CFM)** 作为optimization objective，与TRELLIS保持一致。

**Geometric branch loss (Eq. 3)**：
$$\mathcal{L}_{aes} = \mathbb{E}_{t, x_0, \epsilon} \|f(x, t) - (\epsilon - x_0)\|_2^2$$

变量解释：
- $t$: timestep
- $x_0$: clean sample from $P_{slat}$ (structural latent)
- $\epsilon$: noise
- $f(x, t)$: neural network预测的velocity field
- $(\epsilon - x_0)$: target velocity (CFM的flow target)

**Final loss**:
$$\mathcal{L}_{diff} = \mathcal{L}_{aes} + \mathcal{L}_{phy}$$

这里的 $\mathcal{L}_{phy}$ 是physical branch的CFM loss，结构对称。

### 6.7 Cross-Domain Feature Interaction

"learnable skip-connection layers"将structural branch的特征fuse到physical generation branch。这与U-Net的skip connection思想类似，但作用在cross-domain上。

Hyperparameters (Table 4):
| Model | Resolution | Channel | Latent | Blocks | Heads | MLP ratio | Window |
|-------|-----------|---------|-------|--------|-------|-----------|--------|
| Geometry decoder | 64 | 768 | 8 | 12 | 12 | 4 | 8 |
| Physics decoder | 64 | 2048 | 8 | 4 | 16 | 4 | 8 |
| Physical encoder | 64 | 768 | 8 | 4 | 12 | 4 | 8 |

关键观察：Physics decoder用了**4 blocks而非24 blocks** (TRELLIS默认)，channel数2048是geometry的~2.7倍。这是trade-off：physics properties相对geometry更"local"，不需要那么deep的network，但需要higher capacity per layer。

---

## 7. 实验结果深度分析

### 7.1 Main Results (Table 2)

| Method | PSNR↑ | CD↓ | F-Score↑ | Scale↓ | Material↑ | Afford↑ | COV↑ | MMD↓ | Desc↑ |
|--------|-------|-----|----------|--------|-----------|---------|------|------|-------|
| TRELLIS | 24.31 | 13.2 | 76.9 | — | — | — | — | — | — |
| TRELLIS+PhysPre | 24.31 | 13.2 | 76.9 | 13.21 | 8.63 | 7.23 | 0.24 | 0.12 | 6.55 |
| **PhysXGen** | **24.53** | **12.7** | **77.3** | **7.24** | **13.01** | **11.30** | **0.33** | **0.08** | **10.11** |

几个关键insight：

1. **Geometry不降反升**：PSNR从24.31→24.53, CD从13.2→12.7。这说明joint training不仅没有hurt geometry，反而因为physical priors的regularization效果让geometry也变好了。这是positive transfer。

2. **Absolute scale下降42%** (13.21→7.24): 这个改进巨大。PhysPre是independent predictor，完全靠image inference scale；而PhysXGen利用了geometry的structural prior。

3. **Affordance提升56%** (7.23→11.30): 这个改进也很大，因为affordance本质是part-level semantic+geometry的函数，dual-branch正好捕获这种relation。

4. **Kinematic COV/MMD**: COV (coverage) 从0.24→0.33, MMD从0.12→0.08。COV衡量生成diversity, MMD衡量mode collapse。两者同时改善说明PhysXGen在kinematic space上既diverse又accurate。

### 7.2 Ablation Studies (Table 3)

| Dep-VAE | Dep-Diff | PSNR | CD | FS | Scale | Mat | Afford | COV | MMD | Desc |
|---------|----------|------|-----|-----|-------|-----|--------|-----|-----|------|
| × | × | 24.31 | 13.2 | 76.9 | 13.21 | 8.63 | 7.23 | 0.24 | 0.12 | 6.55 |
| × | ✓ | 24.31 | 13.2 | 76.9 | 12.01 | 10.69 | 8.95 | 0.26 | 0.11 | 7.71 |
| ✓ | × | 24.32 | 12.9 | 77.0 | 10.57 | 9.86 | 9.32 | 0.28 | 0.11 | 7.54 |
| ✓ | ✓ | **24.53** | **12.7** | **77.3** | **7.24** | **13.01** | **11.30** | **0.33** | **0.08** | **10.11** |

观察：
- **Dep-Diff单独** (第2行): physics metrics有提升但geometry不变。说明diffusion阶段的cross-domain fusion有帮助，但VAE端independent限制了上限。
- **Dep-VAE单独** (第3行): geometry开始小幅提升 (CD 13.2→12.9)。VAE端的correlation影响更基础的representation quality。
- **Both** (第4行): 全面提升，特别是absolute scale从10.57→7.24，说明VAE和diffusion的correlation是协同的，不是简单叠加。

### 7.3 vs GPT-based Baseline (Table 5)

GPT-based baseline = Trellis + PartField [[34]](https://arxiv.org/abs/2504.11451) + GPT-4o (后处理assign physical properties)

| Method | PSNR | CD | FS | Scale | Mat | Afford | COV | MMD | Desc |
|--------|------|-----|-----|-------|-----|--------|-----|-----|------|
| GPT-baseline | 24.31 | 13.2 | 76.9 | 8.81 | 7.95 | 6.73 | 0.09 | 0.24 | **14.31** |
| PhysXGen | 24.53 | 12.7 | 77.3 | **7.24** | **13.01** | **11.30** | **0.33** | **0.08** | 10.11 |

Relative improvements:
- Absolute scale: **+24%**
- Material: **+64%**
- Affordance: **+72%**
- Kinematics (COV): **+267%** (0.09→0.33)
- Function description: **-29%** (唯一退步的)

Function description退步的原因：CLIP [[32]](https://arxiv.org/abs/2103.00020) embedding学习比GPT-4o的free-form generation更难。这是training data scale的硬约束——26K训练样本不够train好CLIP空间的逆向mapping。作者提到T5 [[35]](https://arxiv.org/abs/1910.10683) 理论上可以decode但computational overhead太大。

### 7.4 Evaluation Metrics详解

- **Absolute scale**: Euclidean distance (单位cm) — 越小越好
- **Material/Affordance**: PSNR on property maps — 越大越好 (因为把continuous property看成image)
- **Kinematics**: instantiation distance [[33]](https://arxiv.org/abs/2305.16315) (NAP论文提出) — 包含COV (coverage↑) 和MMD (Minimum Matching Distance↓)
- **Description**: PSNR on cosine similarity score maps
- **Geometry**: PSNR (30 random views from unit sphere), Chamfer Distance ($\times 10^{-3}$), F-score (threshold=0.05, $\times 10^{-2}$)

---

## 8. Limitations与Open Challenges

### 8.1 Absolute Scale的Long-tailed问题

Scale跨3个数量级 (1-1000 cm)，大部分<300 cm。Linear normalization对large objects不友好，log normalization又会压缩middle range。Figure 11显示PhysXGen对extremely large objects的robustness差。

### 8.2 Material & Affordance的Spatial Inconsistency

Figure 10可视化显示生成结果有"scattered artifacts"——neighboring voxels的material prediction不consistent。这是因为VAE的spatial coherence没有显式约束。

### 8.3 Kinematics的Hierarchical Issue

Regression-based prediction无法处理离散的part数量。Parent-child relationship用regression会引入artifacts。Classification-based loss又因为part数量不固定难以实施。

### 8.4 Function Description的CLIP Bottleneck

CLIP是encoder-only，不可逆，所以embedding→text的disentanglement受限。这是为什么Table 5里function description表现最差。

---

## 9. 与相关工作的Positioning

### 9.1 vs DreamFusion [[9]](https://dreamfusion3d.github.io/) 类SDS方法

PhysXGen是feed-forward而非optimization-based，效率高几个数量级，避免了Janus problem。

### 9.2 vs Articulate-Anything [[24]](https://arxiv.org/abs/2410.13882) / Articulate AnyMesh [[26]](https://arxiv.org/abs/2502.02590)

这些是VLM-based的方法，逐个part建模，没有joint training。PhysXGen是unified generative framework。

### 9.3 vs PhysGaussian [[30]](https://arxiv.org/abs/2403.15430) / Physically Compatible 3D [[31]](https://arxiv.org/abs/2411.18046)

这些工作focus on dynamics simulation，不是generation。PhysXGen是generation-first。

### 9.4 vs TRELLIS [[10]](https://arxiv.org/abs/2412.01506) / 3DTopia-XL [[15]](https://arxiv.org/abs/2409.12957) / LGM [[11]](https://arxiv.org/abs/2402.05006)

这些是geometry+appearance的feed-forward generators，没有physical properties。PhysXGen是TRELLIS的physical extension。

### 9.5 vs PhysX-Anything [[25]](https://arxiv.org/abs/2511.13648)

同作者的后续工作，single image → simulation-ready assets。PhysX-3D是foundation，PhysX-Anything是更in-the-wild的应用。

---

## 10. 对Embodied AI的Implications

### 10.1 Sim2Real Transfer

物理属性完整后，生成的3D assets可以直接import到Isaac Sim/MuJoCo/SAPIEN，进行manipulation training。这是sim2real的关键enabler。

### 10.2 Affordance Learning

Part-level affordance rank (1-10)可以作为manipulation policy的reward signal或action prior。

### 10.3 Long-tail Manipulation

6M procedurally generated objects覆盖了大量rare configurations，有助于policy generalization。

---

## 11. 我的Intuition构建

把这篇工作放在更大的图景里看：

**3D generation的演进路径**：
1. **Stage 1** (ShapeNet era): Geometry only — points, meshes
2. **Stage 2** (Objaverse era): Geometry + Appearance — textured meshes
3. **Stage 3** (TRELLIS era): Latent + Versatile — structured latents
4. **Stage 4** (PhysX-3D): Latent + Physical — physics-grounded generation
5. **Stage 5** (未来): Latent + Physical + Behavioral — interaction-aware generation

PhysX-3D的关键贡献是把physical properties变成latent space的first-class citizen。这个insight类比于LLM里把reasoning从implicit变成explicit chain-of-thought——把implicit physics从"靠geometry pattern guess"变成"explicit latent representation"。

**为什么dual-branch有效**：
- Geometry和physics高度correlated：把手位置决定affordance，joint axis位置决定kinematics
- 但它们又not identical：相同geometry可以对应不同material (木桌vs金属桌)
- Dual-branch + residual connection正好捕获"shared backbone + property-specific heads"结构

**为什么CFM比diffusion好**：
- CFM的velocity field是deterministic ODE，更适合continuous physical properties
- Diffusion的stochastic noise injection可能破坏physical consistency (e.g., density突然变化)
- TRELLIS已经证明CFM在structured latent上有效，PhysXGen继承这个选择

**未来的瓶颈**：
1. **Function description**: 需要generative text decoder，可能需要T5 [[35]](https://arxiv.org/abs/1910.10683) 或future VLMs
2. **Articulated dynamics**: 当前只生成static joint parameters，没生成dynamic friction/damping
3. **Multi-object scenes**: 当前single object，scene-level physical interaction未涉及
4. **Real-world grounding**: PhysXNet基于PartNet (synthetic)，real scan的physical property还有gap

---

## 12. 关键References

- **TRELLIS** (base architecture): https://arxiv.org/abs/2412.01506
- **PartNet** (dataset base): https://arxiv.org/abs/1812.02758
- **SAPIEN / PartNet-Mobility**: https://sapien.ucsd.edu/
- **Objaverse**: https://objaverse.allenai.org/
- **ShapeNet**: https://shapenet.org/
- **GAPartNet**: https://arxiv.org/abs/2211.00360
- **OmniObject3D**: https://omniobject3d.github.io/
- **ABO Dataset**: https://arxiv.org/abs/2110.06149
- **DreamFusion**: https://dreamfusion3d.github.io/
- **CLIP**: https://arxiv.org/abs/2103.00020
- **DINOv2**: https://arxiv.org/abs/2304.07193
- **NAP (Neural 3D Articulation Prior)**: https://arxiv.org/abs/2305.16315
- **PartField**: https://arxiv.org/abs/2504.11451
- **3DTopia-XL**: https://arxiv.org/abs/2409.12957
- **LGM**: https://arxiv.org/abs/2402.05006
- **InstantMesh**: https://arxiv.org/abs/2404.07191
- **PhysX-Anything** (同作者后续): https://arxiv.org/abs/2511.13648
- **PhysGaussian**: https://arxiv.org/abs/2403.15430
- **Articulate-Anything**: https://arxiv.org/abs/2410.13882
- **Articulate AnyMesh**: https://arxiv.org/abs/2502.02590
- **T5**: https://arxiv.org/abs/1910.10683
- **PhysX-3D Project Page**: https://physx-3d.github.io/

---

## 13. 总结

PhysX-3D做了一件很优雅的事：它把3D generation从"长得对"推进到"用得对"。技术上，dual-branch + VAE correlation + CFM diffusion是一个完整的pipeline；数据上，PhysXNet的5维physical annotation + PhysXNet-XL的6M procedural scaling为后续工作铺路。

最大的限制是**function description的CLIP bottleneck**和**kinematics的hierarchical prediction**。这两个问题如果解决，physical 3D generation就基本闭环了。

对embodied AI community来说，这个工作的价值在于：第一次让generative 3D assets具备sim2real ready的物理属性，把"看图生成3D"扩展到"看图生成可仿真3D"。这是一个data-centric的sim2real新路径，与传统domain randomization形成互补。
