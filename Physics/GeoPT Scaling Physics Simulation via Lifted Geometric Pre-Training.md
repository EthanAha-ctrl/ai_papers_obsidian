---
source_pdf: GeoPT Scaling Physics Simulation via Lifted Geometric Pre-Training.pdf
paper_sha256: 038e00cd012325eea1591be34c1cf5f65016bec6b5e5637bcba7ecc4feebc2be
processed_at: '2026-08-04T21:26:35-07:00'
target_folder: Physics
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GeoPT 人话版

## 一句话总结

**搞 physics simulation 的 neural network 想做 pre-training，但 pre-training 数据（只有 geometry）和 downstream task（需要 geometry + dynamics）根本不在一个 space 里。GeoPT 的 trick 是：给每个 geometry 随机撒一堆"虚拟粒子"让它们瞎跑，把跑出来的轨迹当 supervision，这样 model 就在和 downstream task 同一个 space 里学了。**

## 问题在哪

工业界现在用 neural network 替代 CFD solver 做空气动力学、碰撞这些 simulation，一个 forward pass 替代几十个小时的 numerical solve，爽得很。Ansys SimAI、Altair PhysicsAI 都在用。

但是要训这种 network，你需要大量 labeled data，而每个 label 就是一次完整 CFD simulation 的结果。DrivAerML 那个汽车空气动力学数据集，**生成一个样本要 61000 CPU 小时**。这跟 NLP/CV 不一样 — 你不能去 web crawl 一亿张 physics simulation 结果。

那怎么办？**Pre-training** 嘛。NLP 有 BERT，CV 有 MAE，都是先在大量无标注数据上学个 prior，再 fine-tune 到下游任务。

所以思路是：去 ShapeNet 这种公开 3D model repo 拿一堆 geometry（cars, planes, ships），在 geometry 上做 self-supervised pre-training，然后 fine-tune 到有 label 的 physics task。

**但是直接这么干会坏。**

作者做了一个很直观的实验。拿 Transolver（现在的 SOTA neural PDE solver）在 DrivAerML 上训，对比两种方式：

1. 正常用 physics label 训
2. 先用 geometry-only supervision 预训练（让 model 预测每个点到 surface 的 vector distance），再 fine-tune

然后可视化 model 学到的 spatial attention pattern。

用 physics label 训出来的 pattern：**车头车尾分得开，左右对称** — 这就是空气动力学该有的样子，前面高压驻点区，后面尾流。

用 geometry-only 预训练出来的 pattern：**车头车尾被分到同一个 state，左右还不对称** — 因为纯几何 supervision 不知道"前面"和"后面"在物理上有区别，模型就按形状相似性乱分组了。

结果就是：**geometry-only pre-training 比 from scratch 还差**。这叫 negative transfer。

## 为什么会坏

核心问题在于 **pre-training space 和 downstream task space 不对齐**。

NLP pre-training 和 downstream 都在 token sequence space，CV pre-training 和 downstream 都在 image pixel space。所以 MAE、SimCLR 这些方法 work — 你 pre-training 学的 representation 直接能 transfer。

但 physics simulation 不是。Downstream task 是 (geometry, dynamics) 联合决定的 — geometry 定义空间和边界，dynamics 定义系统怎么被驱动（来流速度、撞击方向、材料属性等等）。而 geometry-only pre-training 只能看到 geometry，dynamics 维度完全缺失。

你在一个低维空间学的 representation，transfer 到高维空间，肯定有 gap。这就像你只用黑白图片 pre-training 一个 model，然后让它去做彩色图片分类 — 颜色维度从来没见过，怎么 transfer？

## GeoPT 的核心 idea

既然问题是缺 dynamics 维度，那就**人造一个 dynamics 维度**。

具体怎么造？作者观察到：任何物理系统里，粒子都在动。这个动可以用一个 velocity field 描述：

$$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}(\mathbf{x}_t, t) \cdot \mathbb{1}_G(\mathbf{x}_t)$$

- $\mathbf{x}_t$：粒子在时刻 $t$ 的位置
- $\mathbf{v}(\mathbf{x}_t, t)$：velocity field，由物理系统决定
- $\mathbb{1}_G(\cdot)$：indicator function，0 在 geometry 内部/边界，1 在外部 — 让粒子撞到 surface 就停下

真实的 $\mathbf{v}$ 要跑 simulation 才知道，太贵了。**所以 GeoPT 直接 random 采样**：

$$\mathbf{v} \sim \mathrm{Unif}(\mathbb{B}^C)$$

每个 query point 给一个随机方向、随机大小的 velocity（在一个 ball 里均匀采样），然后让它沿直线飞几步。飞到 surface 就停下。

这个随机 velocity 当然不是真实物理。**但它产生的轨迹满足一个非常重要的性质 — 质量守恒**。

作者在 Appendix B 给了理论：这种 free-flight + sticking boundary 的 dynamics，在 continuum limit 下对应 **collisionless Boltzmann equation**（也叫 Liouville equation）：

$$\partial_t f(x, v, t) + v \cdot \nabla_x f(x, v, t) = 0$$

- $f(x, v, t)$：phase-space density（在位置 $x$、velocity $v$、时刻 $t$ 的粒子密度）
- $\partial_t$：对时间偏导
- $\nabla_x$：空间梯度
- $v \cdot \nabla_x$：沿 velocity 方向的 advection

这个方程是 **conservative** 的 — 总粒子数不变，只是从 interior 转移到 boundary。

**关键 insight：很多物理方程都是这个 transport structure 的 augmentation。** Navier-Stokes 在 mass conservation 上加了 momentum 和 constitutive relations，Boltzmann equation 在 streaming term 上加了 collision operator。所以如果你让 model 学会"在任意 velocity field 下保持守恒律"，你就给了一个 universal physics prior，fine-tune 到具体 PDE 时就有好起点。

## 具体怎么做

### Pre-training

对每个 geometry：
1. 采样一堆 query points（体积内 32768 个 + 表面 4096 个）
2. 给每个 point 采样一个 random velocity（在一个半径为 2 的 ball 内均匀采样）
3. 让 points 沿 velocity 飞 3 步（$\tau = 2$，所以是 $t = 0, 1, 2$）
4. 每一步记录每个 point 到 geometry surface 的 **vector distance**（不是 SDF，是带方向的 vector，信息量更大）
5. 让 model 预测这个 trajectory：给定初始位置 $\mathbf{x}$、velocity $\mathbf{v}$、geometry $G$，预测 $\mathbf{x}$ 在未来几步的 vector distance 序列

Loss（Eq. 6）：

$$\mathcal{L}_{\mathrm{lifted}}^{\mathrm{pre}} = \mathbb{E}_{\mathbf{x}, G, V}\left[\|\mathcal{F}_{\widehat{\theta}}(\mathbf{x}; G, V) - \mathbf{h}_G(\mathbf{x}_{0:\tau})\|_2^2\right]$$

- $\mathbf{x}$：query position
- $G$：geometry
- $V$：所有 query points 的 velocity 集合
- $\mathcal{F}_{\widehat{\theta}}$：neural simulator（Transolver）
- $\mathbf{h}_G(\mathbf{x}_{0:\tau})$：trajectory 上的 vector distance 序列，是 supervision target
- $\|\cdot\|_2^2$：squared L2 norm

**这个 loss 看起来和 native pre-training 长得很像，但本质区别是输入多了 $V$，输出多了时间维度。** Model 现在要在 (geometry, velocity) 联合空间里学，而不是纯 geometry 空间。

数据生成快得离谱：36864 个 point 在一个 geometry-dynamics sample 上 tracking 只要 0.2 秒（80 CPU cores），比 industrial CFD 快 **10⁷ 倍**。100 万样本 3 天就生成完了，总共 5TB。

### Fine-tuning

Pre-training 用的是 random velocity。Fine-tuning 时，把 random velocity 换成**任务特定的 velocity field** $V_S$，encode 真实 simulation setting：

- **汽车/飞机空气动力学**：velocity 方向 = 来流方向，大小 = 来流速度。AoA、sideslip 都自然 encode 进去了
- **船舶水动力学**：水和空气两相分别配 velocity，方向和大小反映 two-phase flow
- **汽车碰撞**：velocity 方向 = 撞击方向，大小从撞击点 spatially decay（反映力传播）

Fine-tuning loss（Eq. 7）：

$$\mathcal{L}^{\mathrm{fine}} = \mathbb{E}_{\mathbf{x}, G, V_S}\left[\|\mathcal{F}_{\theta}(\mathbf{x}; G, V_S) - \mathbf{u}(\mathbf{x})\|_2^2\right]$$

- $V_S$：任务特定 velocity field，encode simulation setting $S$
- $\mathbf{u}(\mathbf{x})$：solver 生成的 physics label（pressure, velocity, stress 等）
- 其他符号同上

**这个 $V_S$ 就相当于 prompt** — 不同 velocity configuration 激活 model 不同的 pre-trained capability。作者在 Figure 7c 可视化了：给 crosswind velocity，model 学到的 attention pattern 变倾斜的；给 high speed，pattern 变 concentrated；给 zero speed，退化到 static geometry pattern。这就是 physics simulator 的 prompt engineering。

## 实验结果

5 个 industrial-scale benchmark：

| Benchmark | 啥任务 | 几何变化 |
|-----------|--------|----------|
| DrivAerML | 汽车外空气动力学 | 大 |
| NASA-CRM | 飞机翼压力系数 | 小（只有 aileron 角度变） |
| AirCraft | 飞机六分量气动力 | 中 |
| DTCHull | 船舶阻力 + 波浪 | 大 |
| Car-Crash | 汽车碰撞最大应力 | 中（形变） |

### 主要结论

**1. 减少 20-60% data requirement**

GeoPT 在所有 5 个 task 上一致地减少 data 需求。最夸张的是 DTCHull 减少 60% — 这个 task 几何变化大（hull 曲率、长宽比都变），pre-training 见过多样 geometry 帮助大。NASA-CRM 改善有限 — 几何变化太小（只有 aileron 角度微调），pre-training 优势不明显。

**2. 2× convergence speedup**

Fine-tune 时 GeoPT 收敛快一倍。在 industrial setting 里这很值钱。

**3. Model scaling 终于 work**

Transolver from scratch 在 limited data 下会 overfitting，model 从 8 层加到 32 层性能不升反降。**GeoPT pre-training 起 regularization 作用，让大模型终于能吃满 capacity** — 32 层 > 16 层 > 8 层。这是 foundation model 该有的 scaling behavior。

**4. 超越 SOTA**

在 DrivAerML surface（450 train samples）：GeoPT MSE 0.003370，beat GAOT 的 0.051729 一个量级。NASA-CRM、AirCraft 也都 SOTA。

**5. 泛化到没见过的物理**

在 radiosity（光线传播，Cornell box + Stanford bunny）上 fine-tune，pre-training 从来没见过光传播物理，也没见过 Cornell box 几何，GeoPT 还是比 from scratch 好（MAE 9.0×10⁻² vs 9.7×10⁻²）。这说明 lifted pre-training 学的 prior 确实是 universal 的。

## 关键 ablation

**Geometry-only pre-training 退化**：直接预测 vector distance，不加 dynamics，pre-training 比不预训练还差。这验证了 space misalignment 是真问题。

**Hunyuan3D geometry conditioning 不够**：用 Hunyuan3D VAE encoder 提取 3072 个 geometry token 做 cross-attention conditioning，虽然 geometry 信息精确（能 reconstruct），但缺乏 dynamics awareness，帮助有限。**直接 pre-train backbone 比加 frozen auxiliary feature 有效得多。**

**ShapeNet-V1 比 V2 好**：V1 质量差（normal 不对、朝向不齐）但数量多（13463），V2 质量好但数量少（9515）。V1 在多数 benchmark 反而更好 — **几何多样性比质量重要**。这是 scaling law 的经典体现。

**$\tau$ 步数**：$\tau = 0$ 退化到 static，没用；$\tau = 1$ 已经显著提升；$\tau = 2$ 最佳平衡；$\tau = 3, 4$ 离散误差累积，不再提升。

**Vector distance > SDF**：vector distance 带方向信息，每点信息量更大。

## 我的直觉

### Lifting 这个 idea 本质上是什么

**就是把 pre-training 从低维空间 lift 到高维空间，让 pre-training 和 downstream 在同一个空间里。**

数学上，native pre-training 是学 $\mathcal{G} \to \mathcal{H}$ 的 map。Downstream 是学 $(\mathcal{G}, \mathcal{S}) \to \mathcal{U}$。$\mathcal{S}$ 维度缺失导致 transfer gap。

GeoPT 通过引入 synthetic $\mathcal{V}$，把 pre-training 变成 $(\mathcal{G}, \mathcal{V}) \to \mathcal{H}_{\mathrm{traj}}$。现在 pre-training 和 downstream 都在 $(\text{geometry}, \text{velocity})$ 联合空间里，representation 直接 transfer。

这个 idea 其实挺通用的。任何"pre-training data space 比 downstream task space 低维"的场景，都可以考虑 lifting。比如：
- 静态图 pre-training → 动态图 downstream → 加 synthetic temporal dimension
- 2D image pre-training → 3D vision downstream → 加 synthetic depth
- Single-modality pre-training → cross-modality downstream → 加 synthetic cross-modal cue

关键是要找到合适的 synthetic dimension，让 supervision 仍然 cheap 但又 capture 到 downstream 需要的 structure。

### 为什么 random velocity work

Random velocity 不是真实物理，但它满足两个 universal 性质：

1. **Characteristic-driven correlations**：不同起点出发的 trajectory 可能相交，这对应物理中不同 location 通过 flow/force 传播耦合。Random velocity 让 model 见到各种可能的 coupling pattern。

2. **Boundary interaction**：sticking boundary 让 trajectory 在 surface 停下，强制 model 学 surface 附近的 local geometry-feature relationship。几乎所有工业 simulation（pressure on surface, contact force, light-surface interaction）都是 boundary-dominated。

这两个性质是 continuum/kinetic physics 的 backbone，所以学到的 prior 能 transfer。

### Sticking boundary 是精妙设计

如果用 reflecting boundary，model 会学反射 pattern，对绝大多数工业 simulation 没用。如果用 periodic boundary，model 学不到 surface interaction。Sticking boundary 让 trajectory 在 surface 累积，正好 capture 了 surface-dominated physics 的本质。

### 和 MAE/SimCLR 的根本区别

MAE/SimCLR 也在 native space 学 — pre-training 和 downstream 都在 image space。它们不需要 lifting，因为 space 已经对齐。

GeoPT 面对的是 **space misalignment** 问题，这是 physics simulation 特有的。NLP/CV 的 pre-training 成功掩盖了这个问题，但一旦 downstream task 比预训练数据维度高，native pre-training 就会 collapse。

### $V_S$ 作为 prompt 的意义

Pre-training 时用 random velocity，让 model 见到所有可能的 dynamics。Fine-tuning 时用 task-specific $V_S$，相当于告诉 model "现在你要做的是这种 dynamics"。

这跟 language model 的 in-context learning 很像 — 给个 prompt 激活对应 capability。Figure 7c 可视化证实了：不同 $V_S$ 确实激活不同 attention pattern。

这是 physics simulator 走向 prompt-based interface 的第一步。想象未来你下载一个 GeoPT foundation model，然后用 natural language 或 simulation parameter 自动生成 $V_S$ prompt，就能做各种 physics task。

### 和 diffusion model 的结构相似性

Diffusion model 在 random noise 下 evolve particles（forward SDE），记录中间状态，然后 learn to reverse。GeoPT 在 random velocity 下 evolve particles（deterministic transport），记录 trajectory，然后 learn to predict。

两者都在 random perturbation 下制造 supervision，都记录中间状态，都学一个 dynamics-aware prior。差别是 diffusion 是 stochastic + generative，GeoPT 是 deterministic + discriminative。

能否融合？比如用 diffusion model 在 mesh 上做 generative physics — 给 geometry 和 boundary condition，生成 physics field。GeoPT 的 lifted trajectory supervision 可能是很好的 pre-training objective for such diffusion model。

### 对 neural simulator scaling 的意义

Neural simulator 一直受限于 data generation cost。GeoPT 证明了可以用 web-scale geometry + cheap synthetic dynamics 做有效 pre-training，downstream 只需要 20-60% 的 labeled data。

这可能是 neural simulator 走向 foundation model 的路径。类似 GPT-3 之于 NLP — 大规模 self-supervised pre-training + 少量 fine-tuning。只不过 physics 的 "self-supervised" 不能用 native data，必须 lifting。

### 为什么几何多样性 > 质量

ShapeNet-V1 vs V2 的结果很经典。V1 噪声多但多样，V2 干净但少。V1 赢。

这跟 Chinchilla scaling law 一致 — fixed model size 下 data 数量主导。在 industrial physics 这种 high-difficulty low-data regime，model 需要见识更多 shape variation 才能 generalize 到 OOD geometry。少量干净数据让 model overfit 到 specific shape pattern，反而 hurt。

这对未来 data strategy 有指导意义：与其精修少量 geometry，不如大量采集/生成 diverse geometry。3D generative model（Hunyuan3D, 3D generative models）可以批量产 geometry，作为 GeoPT pre-training data source。

### 三个 regimes 的 error 分析

Figure 7 worst case：GeoPT 主要 improve wake flow 预测。Wake 是 fluid 中最复杂的结构（分离、recirculation、turbulence），最难 capture。这暗示 lifted pre-training 给了 model 对复杂 flow pattern 的 inductive bias — synthetic trajectory 在 boundary 附近变化最剧烈，类似 wake 的 boundary layer dynamics。

这跟 LLM pre-training 的 "emergent ability" 有点像 — pre-training 不直接教 downstream task，但给的 prior 让 model 在 hard case 上表现更好。

## 局限和未来

1. **Simulation setting 参数化不完整**：crash 里 elastic + strength properties 都要 encode 到一个 velocity value，信息损失。可以借鉴 ControlNet (https://arxiv.org/abs/2302.05543) 加 zero-init extra channel。

2. **Regular grid simulation**：3D turbulence 没有复杂 boundary。可以在 pre-training 中保留一定比例 empty-boundary iterations，让所有点 free-flight 学 isotropic dynamics。

3. **更多物理 domain**：目前测了 fluid/solid mechanics + radiosity。能否 extend 到 electromagnetics, heat transfer, quantum chemistry？只要能找到合适的 "dynamics" 参数化和 trajectory supervision。

4. **Scaling up**：15M params 已经是 neural simulator 的大模型了。但相比 LLM 的 100B+ 还小很多。如果能用更多 geometry（Objaverse (https://objaverse.allenai.org/) 有 800K+ models）+ 更多 dynamics + 更大 model，scaling law 会怎样？

5. **Multi-physics coupling**：现在每个 task 单独 fine-tune。能否做一个真正 unified foundation model，一个 checkpoint 同时做 aero + hydro + crash + thermal？Unified interface $(G, V_S)$ 是个好起点，但 $V_S$ 设计可能需要更抽象。

## 代码

https://github.com/Physics-Scaling/GeoPT

## 相关链接

- Transolver: https://arxiv.org/abs/2405.13975
- DrivAerML: https://arxiv.org/abs/2408.11969
- Hunyuan3D: https://hunyuan.tencent.com/blog/
- ShapeNet: https://shapenet.org/
- Objaverse: https://objaverse.allenai.org/
- MAE: https://arxiv.org/abs/2111.06377
- SimCLR: https://arxiv.org/abs/2002.05709
- DINO: https://arxiv.org/abs/2104.14294
- ControlNet: https://arxiv.org/abs/2302.05543
- Chinchilla: https://arxiv.org/abs/2203.15556
- FNO: https://arxiv.org/abs/2010.08895
- DeepONet: https://arxiv.org/abs/1910.03193
- Unisolver: https://arxiv.org/abs/2502.07343
- GAOT: https://arxiv.org/abs/2505.18781
- FCPW: https://github.com/rohan-sawhney/fcpw
- OpenFOAM: https://www.openfoam.com/
- OpenRadioss: https://www.openradioss.org/
- Vector distance functions: https://link.springer.com/chapter/10.1007/3-540-45561-7_3
- Liouville equation: https://en.wikipedia.org/wiki/Liouville%27s_theorem_(Hamiltonian)
- Radiosity: https://en.wikipedia.org/wiki/Radiosity_(computer_graphics)
- Ansys SimAI: https://www.ansys.com/products/simai
- Altair PhysicsAI: https://www.altair.com/physicsai

---

# GeoPT: 通过 Lifted Geometric Pre-Training 扩展 Physics Simulation

## 1. 问题动机：Neural Simulator 的 Scaling Bottleneck

Physics simulation 的 neural surrogate 在 industrial design 中已经商业化落地，比如 Ansys SimAI (https://www.ansys.com/products/simai)、Altair PhysicsAI (https://www.altair.com/physicsai)。这类模型通过 single forward pass 替代昂贵的 numerical solver，对 iterative design workflow 极有价值。

但核心瓶颈在 label generation：
- DrivAerML (https://arxiv.org/abs/2408.11969) 一个 industrial-fidelity 样本需要 **6.1×10⁴ CPU-hours**
- 这种 cost 让 physics data 无法像 web-scale vision/language 数据那样任意扩张

Vision 有 MAE (https://arxiv.org/abs/2111.06377)、SimCLR (https://arxiv.org/abs/2002.05709)、DINO (https://arxiv.org/abs/2104.14294)，Language 有 BERT/GPT，但这些都建立在"pre-training 数据 space 和 downstream task space 对齐"的前提上。Physics simulation 不满足这个前提：downstream 需要 (geometry, dynamics) coupled representation，而 geometry-only pre-training 只能学到 reduced representation。这是 GeoPT 要解决的核心矛盾。

## 2. 关键洞察：Geometry-Physics Gap

作者在 Figure 3 做了一个很直观的可视化实验。他们用 Transolver (https://arxiv.org/abs/2405.13975) 在 DrivAerML 上对比两种 supervision：
- (i) Physics supervision
- (ii) Geometry-only supervision (predict vector distance)

可视化 Transolver 学到的 aggregation weights（代表 model 学到的 spatial correlations）：

**Physics supervision 产生的 pattern**: 前后不对称、左右对称 — 这正是 aerodynamic flow 的结构（车前方高压驻点区，后方 wake recirculation，左右镜像）

**Geometry-only supervision 产生的 pattern**: 前后归到同一 state，左右不对称 — 因为静态 shape cues 无法区分 upstream/downstream，模型退化到按形状几何相似性分组

这个观察非常重要，它直接说明了 native pre-training 为什么 negative transfer：模型学到的 inductive bias 与 physics 完全错位。在 Figure 1 的 quantitatively comparison 里，geometry-only pre-training 比 from scratch 还差。

## 3. Lifted Geometric Pre-Training：方法详解

### 3.1 Lifting 的数学结构

Native pre-training 目标（Eq. 2）：
$$\mathcal{L}_{\mathrm{native}}^{\mathrm{pre}} = \mathbb{E}_{\mathbf{x}, G}\left[\|\mathcal{F}_{\widehat{\theta}}(\mathbf{x}; G) - \mathbf{h}_G(\mathbf{x})\|_2^2\right]$$

变量解释：
- $\mathbf{x} \in \mathbb{R}^C$：query position（在 3D 中 $C=3$）
- $G \in \mathcal{G}$：geometry（一个 mesh / surface）
- $\widehat{\theta}$：pre-training 阶段的 model parameters
- $\mathcal{F}_{\widehat{\theta}}(\mathbf{x}; G)$：neural simulator 在 position $\mathbf{x}$、geometry $G$ 下的预测
- $\mathbf{h}_G(\mathbf{x}) \in \mathcal{H}$：从 geometry $G$ 推导的 self-supervision target（如 vector distance field）
- $\|\cdot\|_2^2$：squared L2 norm

这是 static 的 $\mathcal{G} \to \mathcal{H}$ 映射，没有 dynamics 维度。

### 3.2 Dynamics 的参数化

Eq. 3 给出真实物理系统的 dynamics 描述：
$$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_S(\mathbf{x}_t, t) \cdot \mathbb{1}_G(\mathbf{x}_t), \quad \mathbf{x}_0 = \mathbf{x}$$

变量解释：
- $\mathbf{x}_t$：particle 在时刻 $t$ 的位置
- $\mathbf{v}_S(\mathbf{x}_t, t): \mathbb{R}^C \times \mathbb{R} \to \mathbb{R}^C$：由 simulation setting $S$ 决定的 instantaneous velocity field
- $\mathbb{1}_G(\cdot)$：indicator function，**0 inside/on boundary $G$，1 otherwise** — 这让 trajectory 在撞到边界时停下（"sticking boundary"）
- $S$：boundary types, external forces, governing equation, initial states 等所有 simulation 设置

这个 formulation 抓住两个关键结构：
1. **Spatial coupling via $\mathbf{v}_S$**：不同起点出发的 trajectory 可能 intersect，对应物理中不同 location 通过 flow/force transmission 耦合
2. **Boundary interaction via $\mathbb{1}_G$**：trajectory 在 boundary 停下，反映 surface pressure、contact force、radiosity 都受 boundary 主导

跨物理 regime 通用：
- Fluid dynamics: $\mathbf{v}_S$ = flow velocity
- Solid mechanics: $\mathbf{v}_S$ = displacement
- Radiative transport: $\mathbf{v}_S$ = propagation direction

### 3.3 Synthetic Dynamics Lifting

关键步骤：真实 $\mathbf{v}_S$ 需要 expensive simulation，所以用 synthetic random velocity 替代（Eq. 4）：
$$\frac{d\mathbf{x}_t}{dt} = \mathbf{v} \cdot \mathbb{1}_G(\mathbf{x}_t), \quad \mathbf{x}_0 = \mathbf{x}, \quad \mathbf{v} \sim \mathrm{Unif}(\mathbb{B}^C)$$

变量解释：
- $\mathbf{v} \in \mathbb{R}^C$：每个 query point 独立采样的 per-particle velocity
- $\mathbb{B}^C = \{\mathbf{v} \in \mathbb{R}^C : \|\mathbf{v}\|_2 \leq v_{\max}\}$：半径为 $v_{\max}$ 的 bounded ball
- $\mathrm{Unif}$：球内均匀分布
- $V \in \mathcal{V}$：所有 query points 的 velocity 集合

每个 query point 的 velocity 是 constant（不像 Eq. 3 那样随时间变），所以 trajectory 是 free-flight 直线。

### 3.4 Lifted Supervision Target

新的 supervision target（Eq. 5）：
$$\mathbf{h}_G(\mathbf{x}_{0:\tau}) = \{\mathbf{h}_G(\mathbf{x}_t)\}_{t=0}^{\tau} \in \mathcal{H}_{\mathrm{traj}}$$

变量解释：
- $\mathbf{x}_{0:\tau}$：从 $\mathbf{x}_0$ 出发的 trajectory，离散化成 $\tau+1$ 个时间点
- $\mathbf{h}_G(\mathbf{x}_t)$：每个 trajectory point 处的 geometric feature（vector distance）
- $\mathcal{H}_{\mathrm{traj}}$：trajectory 空间，比 static $\mathcal{H}$ 维度高一个时间维度

### 3.5 Lifted Pre-Training Loss

最终目标（Eq. 6）：
$$\boxed{\mathcal{L}_{\mathrm{lifted}}^{\mathrm{pre}} = \mathbb{E}_{\mathbf{x}, G, V}\left[\|\mathcal{F}_{\widehat{\theta}}(\mathbf{x}; G, V) - \mathbf{h}_G(\mathbf{x}_{0:\tau})\|_2^2\right]}$$

三个 expectation 的 source（Figure 4）：
- $G$：从 ShapeNet category-balanced sampling 的 geometry
- $\mathbf{x}$：从 surrounding volume space $\Omega_G$ 和 boundary $G$ 上采样
- $\mathbf{v} \in V$：从 $\mathbb{B}^C$ 上均匀采样

注意 inset diagram 那个 lifting 关系：
- $\mathcal{G} \to (\mathcal{G}, \mathcal{V})$：垂直向上的 lifting 箭头
- $(\mathcal{G}, \mathcal{V}) \to \mathcal{H}_{\mathrm{traj}}$：lifted pre-training task
- $\mathcal{G} \to \mathcal{H}$：native pre-training
- $\mathcal{H}_{\mathrm{traj}} \to \mathcal{H}$：slicing（取 $t=0$ 退化到 static）

Native pre-training 是 lifted 的 degenerate case：当 dynamics 被移除，trajectory collapse 到 single point。

## 4. Theoretical Interpretation (Appendix B)

这部分我觉得是 paper 最有思想深度的环节。每个 particle 携带 $(\mathbf{x}, \mathbf{v})$，最自然的 continuum 描述在 phase space。定义 phase-space density $f(x, v, t)$，dynamics (Eq. 8: $\mathbf{x}(t) = \mathbf{x}_0 + t\mathbf{v}$) 满足 collisionless transport equation（Eq. 9）：
$$\partial_t f(x, v, t) + v \cdot \nabla_x f(x, v, t) = 0$$

变量解释：
- $f(x, v, t)$：在位置 $x$、velocity $v$、时刻 $t$ 的 phase-space density
- $\partial_t$：时间偏导
- $\nabla_x$：空间梯度算子
- $v \cdot \nabla_x$：沿 velocity 方向的 advection operator

这就是经典 **Liouville equation / collisionless Boltzmann equation** (https://en.wikipedia.org/wiki/Liouville%27s_theorem_(Hamiltonian))，characteristic curves 满足 $\dot{\mathbf{x}}(t) = \mathbf{v}, \dot{\mathbf{v}}(t) = 0$（Eq. 10），完全对应 free-flight trajectory。

Sticking boundary condition（Eq. 11）：
$$\partial_t f_G(x, v, t) = (v \cdot n(x))_+ f(x, v, t) dv, \quad x \in G$$

变量解释：
- $f_G$：在 boundary $G$ 上累积的 phase-space density
- $n(x)$：$x$ 处的 outward unit normal
- $(v \cdot n(x))_+ = \max(v \cdot n(x), 0)$：positive part，只有 outgoing flux 累积到 boundary

**Mass conservation**（Proposition B.1）：
$$\frac{d}{dt}\left(\int_\Omega \int_{\mathbb{R}^d} f(x, v, t) dv dx + \int_G \int_{\mathbb{R}^d} f_G(x, v, t) dv dS(x)\right) = 0$$

变量解释：
- 第一项：interior phase-space mass
- 第二项：boundary accumulated mass
- $dS(x)$：boundary 上的 surface measure

这表明 GeoPT pre-training 教模型学的是 **mass conservation under arbitrary velocity fields**，相当于学一个 universal conservation-law prior。

Remark B.3 给了一个很漂亮的 generalizability 论述：Boltzmann equation 在 streaming term 上加 collision operator，Navier-Stokes 在 mass conservation 上加 momentum 和 constitutive relations，许多物理模型都是这个 transport structure 的 augmentation。所以 GeoPT 学到的 inductive bias 是 characteristic-driven correlations + boundary interactions，跨 continuum/kinetic system 共享。

## 5. Architecture & Implementation

### Backbone: Transolver

Transolver (https://arxiv.org/abs/2405.13975) 是 geometry-general neural solver，把 mesh points 当 tokens，attention 当 global integral operator (https://arxiv.org/abs/2304.13255)。关键设计是用 **latent physical states**（32 个 state tokens）绕过 mesh 结构，避免 quadratic attention 复杂度。

Model sizes：
- Base: 8 layers, 3M params
- Large: 16 layers, 7M params  
- Huge: 32 layers, 15M params
- 都用 256 hidden channels, 32 state tokens

15M params 在 physics simulation domain 已经算大模型（neural simulator 通常比 vision/language 小很多）。

### Pre-Training Data 构造

Algorithm 1 给了完整流程：
1. Normalize geometry: 旋转对齐前向 -x，zero-center xy，底面在 xy plane，scale 到 unit length
2. Sample query positions: 从 bounding box $\Omega_G$ + surface $\partial G$ 采样，去掉 inside points，保留 $N$ tracking points
3. Sample synthetic velocity field: 每点 i.i.d. 从 $\mathbb{B}^C$ 采样
4. Compute feature trajectory: 对每个 $t \in \{0, \ldots, \tau\}$，用 FCPW (https://github.com/rohan-sawhney/fcpw) 加速计算 vector distance，并 update position $\mathbf{x}_{t+1} = \mathbf{x}_t + \mathbf{v} \cdot \mathbb{I}_G(\mathbf{x}_t)$

配置：
- 32,768 volume points + 4,096 surface points per geometry
- $v_{\max} = 2$
- $\tau = 2$（3 步）
- 100 random velocity fields per geometry
- Vector distance 作为 $h_G$（比 SDF 好，因为含方向信息，见 Figure 10(c)）

总数据规模：**1,346,300 samples, ~5TB**，生成只需 **3 days on 80 CPU cores**，比 industrial CFD 快 **10⁷×**。

### Fine-Tuning 配置

对每个 task 把 simulation setting $S$ encode 成 $V_S = \{\mathbf{v}_S\}$：

**Aerodynamics** (cars, aircrafts): direction = incoming flow direction，magnitude = freestream speed
- 涵盖 angle of attack, sideslip angle, freestream velocity

**Hydrodynamics** (DTCHull, https://arxiv.org/abs/2305.11084): water/air 两相分别配置，方向和 magnitude 反映 two-phase flow

**Crash simulation** (Car-Crash): direction = impact direction，magnitude = spatially decaying from collision point（reflecting force propagation）

这个统一接口（geometry $G$ + velocity field $V_S$）让 single pre-trained model 通过 reconfigure velocity 适配 diverse physics。

### Configuration Recipe (Appendix D)

- **Direction**: 必须对齐 incoming flow / impact direction（zero shift 时性能最好，Figure 12）
- **Norm**: low-speed tasks ($< 100$ m/s, cars/ships) 用 [0.1, 1.0]，high-speed (aircraft) 用 [1.0, 2.0]
- 大致合理即可，不必精调

## 6. Benchmarks & Main Results

5 个 industrial-scale benchmarks (Table 1)：

| Benchmark | Type | #Mesh | #Variables | #Train | #Test | Output |
|-----------|------|-------|-----------|--------|--------|--------|
| DrivAerML | Aero | ~160M | Geometry | 100 | 20 | Pressure & Velocity |
| NASA-CRM (https://arc.aiaa.org/doi/10.2514/6.2025-0770) | Aero | ~450K | Geo, Speed, AoA | 105 | 44 | Pressure Coef. |
| AirCraft (https://arxiv.org/abs/2506.01094) | Aero | ~330K | Geo, Speed, AoA, Sideslip | 100 | 50 | 6 Aero Components |
| DTCHull | Hydro | ~240K | Geo, Yaw | 100 | 20 | Time-avg Pressure & Velocity |
| Car-Crash | Crash | ~1M | Impact Angle | 100 | 30 | Max 2D Von Mises Stress |

### Key findings (Figure 5):

1. **Data reduction**: GeoPT 一致地 reduce 20-60% data requirements，达到 full-data training performance
2. **Geometry generalization**: 几何多样性大的任务收益大
   - DTCHull: 60% reduction（hull curvature, length-to-beam ratio 变化大）
   - NASA-CRM: moderate improvement（只有 aileron angle变化，几何变化局部）
3. **Surface-only tasks 也 work**: Car-Crash 虽然 pre-training 用 volume + surface，配置 decayed velocity on surface 就能 adapt

### Scalability (Figure 6):

- **Model size**: Transolver from scratch 在 limited-data 下受 overfitting 限制，scaling 受阻。GeoPT 通过 pre-training regularization 始终从增大 model size 受益
- **Data diversity**: 几何多样性比 dynamics 多样性更重要（Figure 6b）。Task-specific: 固定 incoming flow 的 DrivAerML 用 6% dynamics 就够；varied speed/AoA/sideslip 的 AirCraft 多采 dynamics 显著提升

### Quantitative comparison (Table 4):

DrivAerML surface (450 train samples): GeoPT MSE 0.003370 vs Transolver 0.004223 vs GAOT (https://arxiv.org/abs/2505.18781) 0.051729

NASA-CRM (105 samples): GeoPT 0.010722 vs Transolver 0.011246 vs GAOT 0.077170

AirCraft (140 samples): GeoPT 0.062 vs Transolver++ (https://arxiv.org/abs/2506.01094) 0.064 vs GINO (https://arxiv.org/abs/2309.00583) 0.133

## 7. Ablations 深度解析

### 7.1 Geometry Usage (Figure 8a)

三种 geometry 用法对比：
- Geometry-only pre-training (vector distance supervision) — **negative transfer**，比 from scratch 还差
- Geometry conditioning (Hunyuan3D (https://hunyuan.tencent.com/blog/) VAE encoder 提取 3072 geometry tokens + cross-attention) — 改善但有限
- GeoPT (dynamics-lifted) — 最好

Insight: Hunyuan3D 学的 representation 虽然能精确 reconstruct geometry（Figure 20），但缺乏 dynamics awareness，所以对 physics 帮助有限。这印证了 native space pre-training 的局限性。

Pre-training backbone > conditioning as frozen feature：因为 conditioning 不显式 warm up physics-learning process。

### 7.2 ShapeNet-V1 vs V2 (Figure 10a)

V1: 低质量（incorrect normals, non-aligned orientations）但高多样性（13,463 geometries）
V2: 高质量（manually corrected）但低多样性（9,515 geometries）

V1 在大多数 benchmark 反而更好 — **几何多样性比质量更重要**。这呼应 web-scale pre-training 的 scaling law 思路。未来 3D generation models (https://hunyuan.tencent.com/blog/) 可以作为 data source。

### 7.3 Step Number $\tau$ (Figure 10b)

- $\tau = 0$：退化到 static supervision，无收益
- $\tau = 1$：已经有显著提升 — dynamics representation 必要
- $\tau = 2$：最佳平衡
- $\tau = 3, 4$：accumulated discretization error，只在部分 benchmark 更好

### 7.4 Vector Distance vs SDF (Figure 10c)

Vector distance (https://link.springer.com/chapter/10.1007/3-540-45561-7_3) 比 SDF 好。原因：vector distance 不仅含 distance 还含 direction 信息，对每点信息量更大。

### 7.5 Dynamics-Dependent Correlations (Figure 7c, 22-24)

可视化 GeoPT 在不同 $V_S$ 下的 learned aggregation weights：
- Crosswind（60° shifted direction）：产生倾斜 correlation pattern
- High speed (2 normalized)：更 concentrated correlation
- Zero speed：退化到 static geometry correlation

这说明 $V_S$ 类似 **prompt**，激活不同 pre-trained capability。

## 8. Extension to Radiosity (Appendix A)

测试 GeoPT 对未见过物理 domain 的泛化。Cornell box + Stanford bunny radiosity simulation (https://en.wikipedia.org/wiki/Radiosity_(computer_graphics))，governing physics 完全不同于 fluid/solid mechanics，Cornell box 几何也没出现在 pre-training。

参数化：light propagation direction 作为 $V_S$，类比 aerodynamics 的 flow direction。
- Train: 160 samples, Test: 40 samples
- GeoPT MAE: 9.0×10⁻² vs from scratch 9.7×10⁻²

定性上 (Figure 9)，GeoPT 更准确捕捉 high-frequency shadow boundaries，特别是 light-geometry interaction 复杂区域。这表明 dynamics-lifted prior 是相当 general-purpose 的。

## 9. Limitations & Future Work

1. **Simulation settings 参数化局限**：crash 中 elastic + strength properties 都要 encode 到一个 velocity value，可能损失 distinguishability。可借鉴 ControlNet (https://arxiv.org/abs/2302.05543) 的 zero-init extra channel 思路扩展。

2. **Regular grid simulation**：3D turbulence in regular grids (https://turbulence.pha.jhu.edu/) 没有 complex geometry boundaries。可以在 pre-training 中保留一定比例的 empty-boundary iterations，让 tracking points 全 free-flight，学 isotropic dynamics。

## 10. 我的 Intuition & Broader Connections

### 10.1 为什么 Lifting Work

Lifting 本质是把 pre-training 从低维 manifold 推到高维 manifold，让 model 在 downstream task 真正生活的空间内学习。这跟 SDP relaxation (https://en.wikipedia.org/wiki/Semidefinite_programming) 的 lift-and-project 思想一脉相承 — 通过升维简化约束结构。

具体到 GeoPT，关键 trick 是用 synthetic random velocity 制造高维 supervision，绕过 expensive real simulation。Random velocity 不是真实 physics，但它产生的 trajectory 满足 conservation law（这是 transport equation 的本质），而 conservation law 是所有 continuum/kinetic physics 的 backbone。

### 10.2 与 Vision Pre-Training 的根本区别

MAE/DINO 在 native image space 学，downstream recognition 也在 image space — 完美对齐。GeoPT 面对的是**空间不对齐**问题：pre-training 在 $\mathcal{G}$，downstream 在 $\mathcal{G} \times \mathcal{S}$。Lifting 通过引入 synthetic $\mathcal{S}$ dimension 把 pre-training 推到 $\mathcal{G} \times \mathcal{V}$，与 downstream 对齐。

这个 insight 对其他 pre-training → downstream mismatch 问题（比如 video → 4D physics, static scene → dynamic scene, single-modality → cross-modality）应该有启发。

### 10.3 与 Physics-Informed Neural Networks (PINNs)

PINNs (https://www.sciencedirect.com/science/article/pii/S0021999118307125) 把 PDE residual 作为 loss，需要知道 governing equation。GeoPT 不需要知道具体 PDE，只学 universal conservation prior。这是 data-driven 和 physics-informed 的中间路线。

### 10.4 与 Diffusion / Score-Based Models

Lifted trajectory supervision 跟 diffusion 的 forward SDE 有结构相似性 — 都在 random velocity/perturbation 下 evolve particles，记录中间状态。但 GeoPT 是 deterministic transport + random initial velocity，diffusion 是 stochastic noise injection。能否融合？Diffusion model 在 mesh 上做 generative physics 可能是 future direction。

### 10.5 与 Operator Learning

Neural operators (FNO (https://arxiv.org/abs/2010.08895), DeepONet (https://arxiv.org/abs/1910.03193)) 学 function space 之间的 map。GeoPT 实际学的是 (geometry, velocity) → feature trajectory 的 operator，这个 operator 在 fine-tuning 时被 specialize 到 physics task。Universal operator 的一个新方向。

### 10.6 与 In-Context Learning

Unisolver (https://arxiv.org/abs/2502.07343) 用 PDE 信息做 in-context conditioning。GeoPT 用 $V_S$ 作为 prompt，类似 in-context mechanism — 不同 $V_S$ 激活不同 correlation pattern (Figure 7c)。这是 physics simulator 的 prompt engineering。

### 10.7 为什么几何多样性 > 质量重要性

ShapeNet-V1 vs V2 的结果呼应了 Chinchilla (https://arxiv.org/abs/2203.15556) 和 Scaling Laws for NNs 的核心 insight：data 数量/多样性的 power law 在 fixed model size 下主导 performance。在 industrial physics 这种 high-difficulty low-data regime 下，更多样的 geometry 让 model 见识更多 shape variations，对 OOD generalization 帮助更大。

### 10.8 三个 Error Regime 的诊断

Figure 7 的 worst case study 很有意思：GeoPT 主要 improve wake flow prediction。Wake 是 fluid 中最复杂、最难 capture 的结构（分离、recirculation、turbulence）。这暗示 lifted pre-training 给了 model 一个对复杂 flow pattern 的 inductive bias — 因为 synthetic trajectory 在 boundary 附近变化最剧烈，类似 wake 的 boundary layer dynamics。

### 10.9 Boundary Sticking 的深层意义

Sticking boundary 让 trajectory 在 surface 停下，这强制 model 学习 surface 附近的局部 geometry-feature relationship。这正是 aerodynamics (surface pressure)、crash (contact force)、radiosity (light-surface interaction) 的核心。这个设计选择非常 skillful — 如果用 reflecting boundary，model 会学到反射 pattern，对绝大多数工业 simulation 没用。

### 10.10 与 Generative 3D Models 的协同

Hunyuan3D (https://hunyuan.tencent.com/blog/) 等 3D generation model 可以批量生成 synthetic geometry，作为 GeoPT 的 pre-training data source。Paper 在 Section 5.2 也提到 "advanced 3D generation models can be a good data source"。这可能是 scaling GeoPT 的下一步：用 generative model 制造几何多样性，再用 GeoPT 学 physics prior。形成 generative-geometry → physics-prior 的 pipeline。

## 11. 总结

GeoPT 的核心贡献是把 self-supervised pre-training 推广到 **space misalignment** 场景，通过 synthetic dynamics lifting 把 geometry 升到 (geometry, velocity) coupled space。理论层面与 transport equation / Liouville equation 对接，证明 pre-training 实际是学 universal conservation-law prior。实验层面在 5 个 industrial-scale benchmarks 上一致 reduce 20-60% data requirement 和 2× convergence speedup，还 generalize 到未见过的 radiosity domain。

这个工作可能开启 physics simulator 的 foundation model era — 类似 BERT 之于 NLP，ImageNet 之于 vision，但解决了 vision/language 没遇到的 pre-training/downstream space mismatch 问题。Lifting 这个 idea 可能对其他 domain（chemistry, biology, material science）的 pre-training 也有启发，只要能找到合适的 synthetic dynamics 来"扩大" pre-training space 到与 downstream 对齐。

Code: https://github.com/Physics-Scaling/GeoPT

### 参考链接汇总

- Transolver: https://arxiv.org/abs/2405.13975
- DrivAerML: https://arxiv.org/abs/2408.11969
- NASA-CRM: https://arc.aiaa.org/doi/10.2514/6.2025-0770
- AirCraft / Transolver++: https://arxiv.org/abs/2506.01094
- GAOT: https://arxiv.org/abs/2505.18781
- Hunyuan3D: https://hunyuan.tencent.com/blog/
- ShapeNet: https://shapenet.org/
- FCPW: https://github.com/rohan-sawhney/fcpw
- OpenFOAM: https://www.openfoam.com/
- OpenRadioss: https://www.openradioss.org/
- FNO: https://arxiv.org/abs/2010.08895
- DeepONet: https://arxiv.org/abs/1910.03193
- MAE: https://arxiv.org/abs/2111.06377
- SimCLR: https://arxiv.org/abs/2002.05709
- DINO: https://arxiv.org/abs/2104.14294
- ControlNet: https://arxiv.org/abs/2302.05543
- Unisolver: https://arxiv.org/abs/2502.07343
- Neural Operators survey: https://arxiv.org/abs/2108.08431
- Vector distance functions: https://link.springer.com/chapter/10.1007/3-540-45561-7_3
- Radiosity (Cornell box): https://en.wikipedia.org/wiki/Radiosity_(computer_graphics)
- Liouville equation: https://en.wikipedia.org/wiki/Liouville%27s_theorem_(Hamiltonian)
- Chinchilla scaling laws: https://arxiv.org/abs/2203.15556
