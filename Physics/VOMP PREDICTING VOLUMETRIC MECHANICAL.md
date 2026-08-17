---
source_pdf: VOMP PREDICTING VOLUMETRIC MECHANICAL.pdf
paper_sha256: 330018a732563997a00b8662b2d922c521e1bdd044f4ca5b042b0bb54812f53d
processed_at: '2026-08-13T03:21:21-07:00'
target_folder: Physics
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VoMP: 用“人话”讲讲这篇 NVIDIA 的论文

这篇 paper 解决的核心问题其实非常直觉：**怎么让虚拟世界里的东西，表现得像真东西一样**。

## 1. 痛点：为什么现在做物理仿真那么费劲

想象你在 Omniverse 或者游戏引擎里造一个虚拟世界，有一张椅子。你想让这张椅子能被推倒、能被压扁、能被砸碎。这就需要 **physics simulation**。

但 simulation 需要你告诉它这张椅子是用什么做的。具体来说，需要在物体 volume 内每一点提供三个数字：
- **Young's modulus (E)**：多硬？表示抵抗形变的能力（应力与应变的比例）
- **Poisson's ratio (ν)**：受压时横向膨胀多少？（比如捏橡皮会变宽）
- **density (ρ)**：多重？

问题来了：你从 Objaverse 这种仓库下载的 3D 模型，或者用 Gaussian Splat 扫描的 3D 物体，**只有形状和外观，完全没有这些物理参数**。

以前的解决办法有几种，都很痛苦：
- **手动填**：美术或工程师去网上查材料表，复制粘贴预设值。主观、耗时、不 spatially-varying。
- **Per-object optimization** (NeRF2Physics, PUGS)：针对每个物体优化一个特征场，然后查表。慢，单个物体要算 1000 秒以上，而且只能预测表面，预测不了内部。
- **Video distillation** (PhysDreamer 等)：生成一段物体受力的视频，然后反传 fast simulator 去拟合参数。慢，而且拟合出的是 “simulator-specific” 参数，换个 simulator 就不对了。

**Fig. 2 是这篇论文最核心的 motivation**：同一个 sphere，用同样的参数 $(E=10^4, \nu=0.3, \rho=10^3)$，在 XPBD/MPM 和高精度 FEM 下表现完全不同。因为 fast simulator 不是 “consistent” 的，你必须为它改参数。而真实测量的参数是可以跨 simulator 移植的。

VoMP 想做的是：**输出真实世界里能测到的参数，直接喂给任何高精度 simulator**。

---

## 2. 核心 Idea：两步走的“材料字典 + 看图猜材料”

VoMP 的架构 (Fig. 3) 拆开看就是两个组件：

### 第一步：MatVAE —— 造一本“材料字典”

想象真实世界所有的材料，比如橡胶、木头、钢铁、黄金，每种材料都有 三个数字。VoMP 从 MatWeb、Wikipedia 等数据库收集了 10 万多个真实材料三元组。

然后训练一个 VAE (Variational Autoencoder)，把这些材料压成一个 **2D 的 latent space**。

为什么要搞这个 2D 地图？这是 build intuition 的关键：

如果你直接让神经网络去回归 三个数字，它可能预测出一种 **在真实世界不存在的材料**。比如，又像钢一样硬，又像羽毛一样轻。这种材料造不出来，喂给 simulator 会出现不真实的行为。

有了这个 2D latent space，神经网络只要在这个地图上点一下，然后通过 decoder 解码，出来的就一定是 **真实存在的材料**。

Fig. 7d 特别能说明问题：从 Aerographite (超轻) 到 Diamond (超硬) 之间，如果直接在 $\mathbb{R}^3$ 空间做 naive 插值，中间会出现 “Invalid (X)” 的点。但在 MatVAE 的 latent space 插值，中间会经过 Carbon Fiber、Carbon Nitride，都是真实存在的材料。

**这个 2D latent space 起到了“连续 tokenizer”的作用**，保证了输出的物理有效性。

#### MatVAE 的几个技术细节

**Preprocessing**: $E$ 和 $\rho$ 先做 $\log_{10}$ 变换再 normalize，因为它们跨 6 个数量级 (E 从 $10^5$ 到 $10^{11}$ Pa)。$\nu$ 直接 normalize。

**Loss 设计** (公式 2):
$$\mathcal{L}_{\text{MatVAE}} = \mathcal{L}_{\text{Recon}} + \gamma \cdot \text{MI}(z) + \beta \cdot \text{TC}(z) + \alpha \cdot \sum_{j=1}^d \max(\delta, \text{KL}_j(z))$$

这里的设计非常讲究，每一个项都是为了解决一个具体问题：

- **Reconstruction loss**: MSE，让 decode 出来的值接近输入
- **MI(z)**: 互信息，控制 latent 携带多少信息
- **TC(z)**: Total Correlation，惩罚 latent 各维之间的依赖。论文发现如果不加这个，**两个维度都在编码 density**，导致 $\nu$ 信息丢失
- **Free nats $\delta=0.1$**: 防止 posterior collapse。论文发现如果不加，一个维度会 collapse，信息全压在另一个维度

**Normalizing Flow**: 标准 VAE 假设 posterior 是 Gaussian，但 材料数据是 heavy-tailed 的。用 radial flow 增大变分家族，让 posterior 能匹配真实分布。

### 第二步：Geometry Transformer —— “看图猜材料”

给定一个 3D 物体，VoMP 怎么预测每个 voxel 的材料？

**Step A: Feature Aggregation** (公式 3)
对任何可 voxel化 + 可渲染的输入：
1. 把物体切成 $64^3$ 的 voxel grid
2. 从 150 个视角渲染图像
3. 用 DINOv2 提取每个视角的 visual feature (1024 维)
4. 把每个 voxel 中心投影到各视角，取对应 feature 的平均值

$$\mathbf{f}_i = \text{Average}\left(\left\{\mathcal{F}_j(\Pi_j(\mathbf{p}_i)) \mid j \in J\right\}\right)$$

**关键区别**: 与之前工作只做 surface 不同，VoMP **voxelizes interior**。这让模型能学到内部材料组成——比如花盆里是土、树皮下是木头、橙子把儿比橙子肉硬。

**Step B: Geometry Transformer**
- 基于 TRELLIS 架构，用预训练权重初始化
- 12 层 transformer，12 attention heads
- 3D shifted window attention (Swin-style)，8×8 local windows
- 输入: voxel 位置 + DINOv2 feature
- 输出: 每个 voxel 的 2D material latent
- 用 MatVAE decoder 解码成

**训练 loss** (公式 4):
$$\mathcal{L}_F = \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \left\| \mu_\theta(\mathbf{F}(\mathbf{X}_\mathcal{S})_i) - ((E_i, \nu_i, \rho_i)^N)^\top \right\|_2^2$$

**关键设计**: MatVAE 是 **frozen** 的，只训练 transformer。这样让 transformer 在一个已经学好“什么是 valid 材料”的空间里学习，不需要在 loss 里显式加物理约束。

**Stochastic voxel subsampling**: 对于大物体，每 epoch 开始随机采 $L_N = 32768$ 个 voxels。这既解决了显存问题，又起到了 data augmentation 的效果。

---

## 3. 跨 Representation 的统一处理

VoMP 能处理 mesh、Gaussian Splat、NeRF、SDF，关键在于它把所有输入统一成 **(voxel position, multi-view feature)** 的形式：

| Representation | Voxelization 方式 | Rendering 方式 |
|---|---|---|
| Mesh | Flood-fill (Algorithm 2-3) | Path-tracing |
| Gaussian Splat | 3-phase carving (新提出) | Gaussian splat renderer |
| NeRF | Standard | nerfstudio |
| SDF | Standard | Mesh from points |

Transformer 看到的都是 $(\mathbf{p}_i, \mathbf{f}_i)$ 序列，与输入 representation 完全无关。

**Gaussian Splat voxelizer** (§6.1) 是新提出的，三阶段：
1. 3D Gaussians 当 solid ellipsoids，取 99-percentile iso-surface
2. 从球面采几十个视角渲染 depth maps
3. 用 depth maps carve 外部空体，但保留 unseen interior

测试物体 31ms 完成 voxelization。

---

## 4. 训练数据：怎么搞到带物理参数的 3D 物体

这是最大的挑战——没有现成的 “带 volumetric mechanical properties 的 3D dataset”。

### MTD (Material Triplet Dataset)
10 万个真实材料三元组，从 MatWeb、Wikipedia、Cambridge 等收集。用来训练 MatVAE。

### GVM (Geometry with Volumetric Materials)
1624 个 part-segmented 3D mesh (NVIDIA SimReady, Residential 等)，8089 parts。

**Annotation pipeline** (Fig. 4): 对每个 part，给 Qwen2.5-VL-72B 传：
1. 完整物体 rendering
2. 该 part 的 PBR texture 贴到 sphere 上的 detail rendering
3. Material name (来自 USD asset)
4. MTD 中 3 个最近材料的 $(E, \nu, \rho)$ 范围

VLM 输出该 part 的 $(E, \nu, \rho)$，映射到该 part 内所有 voxels。最终得到 37M 标注 voxels。

VLM annotation 误差 (Table 9): 与人工标注小数据集对比，$\log E$ 误差 0.0295，接近人工质量。

**与 concurrent work Pixie 对比**: Pixie 用固定的 in-context physics examples (一个 material name → $(E, \nu, \rho)$ 表)，但部分值落在真实材料范围外。比如 Pixie 的 "tree/leaves": $\rho=200$ kg/m³, E=2e4 Pa——真实叶子没有对应 MTD 范围。VoMP 用 VLM + MTD 检索 + texture cues，显式约束到真实材料范围。

---

## 5. 效果：快、准、有效

### 速度 (Table 1)
- NeRF2Physics: 1454.55s
- PUGS: 1058.33s
- Pixie: 201.63s
- Phys4DGen: 51.65s
- **VoMP: 3.59s**

VoMP 比 PUGS 快 295x。而且其中 transformer + MatVAE 推理只要 8ms，瓶颈是 rendering 和 DINOv2 (2.97s)，这部分还能优化。

### 精度 (Table 2)
- E ALDE: VoMP 0.3794 vs NeRF2Physics 2.8000 (7.4x better)
- ν ADE: VoMP 0.0241 vs Phys4DGen 0.0407 (1.7x better)
- ρ ADE: VoMP 142.70 vs PUGS 3568.22 (25x better)

§D.4 用 486 次单元 cube FEM 仿真校准了误差阈值：**ALRE < 0.05 (E), ARE < 0.15 (其他)** 时仿真行为接近 ground truth。VoMP 全部低于阈值，baseline 全部超阈值。

### Material Validity (Fig. 6d)
- VoMP: log(E) 误差 0.29, ν 误差 **0.00**, ρ 误差 11.75
- Pixie: log(E) 误差 11.90, ν 误差 3.46, ρ 误差 46.58

VoMP 的 ν 误差是 0.00——因为 MatVAE 训练时就保证 decode 出的值在真实材料 manifold 上。

### 端到端仿真 (Fig. 5, 8)
- 椅子跌落
- 保龄球砸椅子
- 18 个狗玩具过弹珠台
- 推土机穿过 100 棵 ficus 树 (带风场)

所有仿真 **零调参**，直接用 VoMP 预测值。

---

## 6. 这篇论文为什么重要

### 技术层面
1. **First feed-forward model** for volumetric mechanical property prediction across representations
2. **First material latent space** (MatVAE) 保证物理有效性
3. **Interior voxelization** 让模型能预测内部材料，不只表面
4. **Simulator-agnostic** 输出真实测量量，可移植到任何 consistent simulator

### 应用层面
这打通了 **Real-2-Sim** workflow 的关键瓶颈。你可以：
1. 用手机扫描一个物体得到 Gaussian Splat
2. 用 VoMP 在 3.6 秒内得到 volumetric mechanical properties
3. 直接放进 FEM simulator 仿真

这对 digital twin、robotics (Sim-2-Real)、game/VFX 工作流都有直接价值。

### 更深层的 insight
这篇论文的哲学是：**把物理约束编码到 representation 里，而不是 loss 里**。MatVAE 学了一个“什么是 valid 材料”的 manifold，让 transformer 在这个 manifold 上学习，不需要在 loss 里加物理约束。这比直接回归 更优雅，也更有效。

类似的思想在 other domains 也见过：比如 image generation 用 VAE 把图像压到 latent space，让 diffusion model 在 latent 上生成。VoMP 把材料压到 2D latent，让 transformer 在 latent 上预测。**都是“learn the manifold, then operate on it”的思路**。

---

## 7. 局限与未来方向

- 固定 grid voxelization 限制分辨率，heterogeneous 区域会 oversmooth
- 假设 part-level isotropic (wood 实际是 anisotropic)
- 未预测 yield strength, shear modulus, thermal expansion
- 当 fast simulator 需要 simulator-specific scale 时需后处理适配

未来方向可能包括：
- Adaptive resolution voxelization (octree-based)
- Anisotropic material models
- 更多物理属性的预测
- 与 fast simulator 的自动适配层

---

## 8. 总结一句话

VoMP = **DINOv2 多视角特征 + TRELLIS Transformer + MatVAE 材料流形约束**，用 feed-forward 方式几秒钟预测任意 3D 表示物体的体素化机械属性，输出可直接用于高精度仿真。

核心 insight 是把物理有效性编码到 representation (MatVAE latent space) 而不是 loss 里，让模型在 valid manifold 上学习。

---

## References

- VoMP 项目页: https://research.nvidia.com/labs/sil/projects/vomp
- TRELLIS (backbone): https://arxiv.org/abs/2412.01506
- DINOv2 (visual features): https://arxiv.org/abs/2304.07193
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- NeRF2Physics (baseline): https://arxiv.org/abs/2404.04242
- PUGS (baseline): https://arxiv.org/abs/2502.12231
- Phys4DGen (baseline): https://arxiv.org/abs/2411.16800
- Pixie (concurrent): https://arxiv.org/abs/2508.17437
- Simplicits (splat simulator): https://doi.org/10.1145/3658184
- β-TCVAE (TC decomposition): https://proceedings.neurips.cc/paper_files/paper/2018/file/1ee3dfcd8a0645a25a35977997223d22-Paper.pdf
- Normalizing Flows: https://proceedings.mlr.press/v37/rezende15.html
- Qwen2.5-VL (annotation VLM): https://arxiv.org/abs/2502.13923

如果你想深挖某个细节，比如 radial flow 的 invertibility 推导、Swin attention 在 sparse voxel 上的实现、Simplicits subspace 的构造、或者 VLM prompt 的具体设计，我可以继续展开。

---

# VoMP: 体素化机械属性预测 — 深度技术解读

## 1. 核心问题与动机

物理仿真 (FEM, MPM, XPBD) 需要在物体 volume 内每一点提供 **Young's modulus (E)**, **Poisson's ratio (ν)**, **density (ρ)** 三个参数。问题在于:

- 现有 3D 仓库 几乎不带这些标注
- 现有方法 (NeRF2Physics, PUGS) 需要 per-object optimization, 慢且只覆盖 surface
- 一些方法 通过 video distillation 反传 fast simulator, 得到的是 **simulator-specific** 参数, 无法跨 simulator 移植 (见 Fig. 2)

**Fig. 2 的关键 insight**: 同一组 $(E=10^4 \text{ Pa}, \nu=0.3, \rho=10^3 \text{ kg/m}^3)$ 在 XPBD/MPM 与高精度 FEM 下产生完全不同的弹跳行为 — 因为 fast simulator 不是 **consistent** 的, 必须修改参数避免 inaccuracy。VoMP 的目标: 输出 **真实测量** 的物理参数, 可以直接喂给任意 **consistent simulator** (FEM, Neo-Hookean, St-Venant-Kirchhoff 等)。

VoMP 的四大特性:
1. Feed-forward, 不需 per-object optimization
2. 跨 representation (mesh, splat, NeRF, SDF)
3. 输出 **物理有效** 的参数 (经 MatVAE 保证)
4. **体素化**, 不仅 surface (能预测内部材料, 如花盆里的土)

---

## 2. MatVAE: 机械属性的连续 tokenizer

### 2.1 为什么需要 latent space

直接回归 $\mathbb{R}^3$ 的 $(E, \nu, \rho)$ 有几个问题:
- 三者单位跨 11 个数量级 (E 从 $10^5$ 到 $10^{11}$ Pa)
- 三者强相关, 任意插值可能产生 **物理不存在** 的材料 (Fig. 7d: Aerographite 与 Diamond 之间 naive 插值会落入 invalid 区)
- Posterior collapse 到单一属性

MatVAE 学一个 **2D latent space** $z \in \mathbb{R}^2$, 让所有 decode 出的 $(E, \nu, \rho)$ 都落在真实材料 manifold 内。

### 2.2 预处理

$E$ 与 $\rho$ 先做 $\log_{10}$ 变换, 再 min-max 到 $[0,1]$; $\nu$ 直接 min-max 到 $[0,1]$。论文实验 (Table 8) 显示: **去掉 log(E)** 后 E 的 ALDE 从 0.3765 飙到 0.9033; 用 **Z-score** 后 ν 的 ADE 从 0.0250 飙到 0.0814。原因是 $\log E$ 与 $\log \rho$ 是 heavy-tailed, Z-score 会把极端值 (如黄金的 E~$10^{11}$) 拉得太远, 破坏 VAE 的 conditioning。

### 2.3 Reconstruction loss (公式 1)

$$\mathcal{L}_{\text{Recon}} = \frac{1}{N} \sum_{i=1}^{N} \left\| \left((E_i, \nu_i, \rho_i)^N\right)^\top - \left((\hat{E}_i, \hat{\nu}_i, \hat{\rho}_i)^N\right)^\top \right\|_2^2$$

变量解读:
- $N$: batch 中材料样本数
- $(E_i, \nu_i, \rho_i)$: 第 $i$ 个材料的真实三元组
- $(\hat{E}_i, \hat{\nu}_i, \hat{\rho}_i)$: MatVAE decoder 重构值
- $(\cdot)^N$: per-property normalization (log + min-max)
- $(\cdot)^\top$: 转置, 让向量变成 column 向量做差

### 2.4 完整 MatVAE 目标 (公式 2)

$$\mathcal{L}_{\text{MatVAE}} = \underbrace{\mathcal{L}_{\text{Recon}}}_{\text{MSE}} + \overbrace{\underbrace{\gamma \cdot \text{MI}(z)}_{\text{Mutual Info}} + \underbrace{\beta \cdot \text{TC}(z)}_{\text{Total Correlation}}}^{\text{Latent Reg}} + \alpha \cdot \sum_{j=1}^{d} \underbrace{\max\left(\delta, \text{KL}\left(q_\phi(z_j) \| p(z_j)\right)\right)}_{\text{Dim-wise KL with free nats}}$$

变量解读:
- $\text{MI}(z) = \mathbb{E}_{p_{\text{data}}}[\text{KL}(q_\phi(z|m) \| p(z))]$: 互信息, 控制 latent 携带的信息量
- $\text{TC}(z) = \text{KL}\left(\bar{q}_\phi(z) \| \prod_j \bar{q}_\phi(z_j)\right)$: total correlation, 衡量 latent 各维之间的依赖
- $\bar{q}_\phi(z) = \mathbb{E}_{m \sim p_{\text{data}}}[q_\phi(z|m)]$: aggregated posterior
- $z_j$: latent 第 $j \in \{1,2\}$ 维
- $\delta = 0.1$: free nats, 每维至少 0.1 nats 信息
- $\gamma=1.0, \beta=2.0, \alpha=1.0$

设计动机 (从 ablation Table 7 读出):
- **w/o TC penalty**: $\nu$ 的 Wasserstein-2 从 0.0437 涨到 0.1052, 因为两个 latent 维度都在编码 density (TC 高), ν 信息丢失
- **w/o free nats**: $\nu$ 的 Wasserstein-2 涨到 0.2064, 因为一个维度 posterior collapse, 信息全压在另一个维度
- **w/o NF**: $\rho$ 的 Wasserstein-2 从 0.0172 涨到 0.0819, 因为 Gaussian posterior 无法捕捉 heavy-tailed 分布

### 2.5 Normalizing Flow (公式 13-15)

材料 triplet 即使 normalize 后仍非高斯 ($\log E$, $\log \rho$ heavy-tailed; $\nu \in (0, 0.5)$ boundary-concentrated), 标准 diagonal Gaussian posterior 会 mode-average。用 radial flow 增大变分家族:

$$\log q_\phi(z|m) = \underbrace{\log q_0(f_\psi^{-1}(z)|m)}_{\text{base density}} - \underbrace{\log|\det J_{f_\psi}(u)|}_{\text{log-Jacobian}}\bigg|_{u=f_\psi^{-1}(z)}$$

Radial flow 变换:
$$f_\psi(u) = u + \underbrace{\beta h(u)}_{\text{radial scale}} \cdot \overbrace{(u - z_0)}^{\text{displacement}}, \quad h(u) = \frac{1}{\alpha + \|u - z_0\|_2}$$

Jacobian log-det (closed form):
$$\log\det J_{f_\psi}(u) = \underbrace{(D-1)\log(1+\beta h(u))}_{\text{angular}} + \underbrace{\log(1+\beta h(u) - \beta h(u)^2 r(u))}_{\text{radial}}$$

变量:
- $D$: latent 维数 (此处 = 2)
- $z_0$: 可训练 $D$ 维向量, 是 flow 的 deformation 中心
- $r(u) = \|u - z_0\|_2$
- $\alpha, \beta$ 通过 softplus 参数化保证可逆: $\alpha = \text{softplus}(\tilde{\alpha})$, $\beta = -\alpha + \text{softplus}(\tilde{\beta})$

### 2.6 Free-nats 防止 posterior collapse (公式 16-18)

对每个材料 $m$, 把 KL 分解:
$$\text{KL}(q_\phi(z|m) \| p(z)) = \sum_j \text{KL}_j(m) + \text{TC}(q_\phi(z|m))$$

Gaussian 情形下 $\text{KL}_j(m) = \frac{1}{2}(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1)$, 梯度会驱使 $\mu_j \to 0, \sigma_j \to 1$ (collapse)。Free-nats 约束:
$$\sum_{j=1}^d \max\left(\delta, \mathbb{E}_{p_{\text{data}}}[\text{KL}_j(z)]\right)$$

低于 $\delta=0.1$ 时 subgradient 为 0, 强制每维保留至少 0.1 nats 信息。

### 2.7 MatVAE 效果 (Table 5 + Fig. 7)

分布距离测度 (W1, W2, KL) 在 MTD test set 上:
- $E$: $W_1 = 0.0405, W_2 = 0.0798, D_{KL} = 0.1379$
- $\nu$: $W_1 = 0.0317, W_2 = 0.0437, D_{KL} = 0.0342$
- $\rho$: $W_1 = 0.0132, W_2 = 0.0172, D_{KL} = 0.0260$

Fig. 7c: 沿 latent 2D 网格 decode 时, $E, \nu, \rho$ 平滑变化, 落在 MTD 真实材料范围内。Fig. 7d: Aerographite → Diamond 在 latent 空间插值, 中间点是 Carbon Fiber / Carbon Nitride (valid), 而 naive 在 $\mathbb{R}^3$ 插值会得到 invalid (X) 点。

---

## 3. Geometry Transformer: 从 voxel 到 material latent

### 3.1 Multi-view 特征聚合 (公式 3)

对任何可 voxel化 + 可渲染的输入, 在 $N^3$ 网格上取 active voxels $\{\mathbf{p}_i\}_{i=1}^L$, 对每个 voxel:

$$\mathbf{f}_i = \text{Average}\left(\mathcal{C}_i = \left\{\mathcal{F}_j\left(\Pi_j(\mathbf{p}_i)\right) \mid j \in J\right\}\right) \in \mathbb{R}^{1024}$$

变量:
- $\mathbf{p}_i \in \mathbb{R}^3$: 第 $i$ 个 voxel 的中心
- $L$: active voxel 数 (最大 $L_N = 32768$)
- $\Pi_j: \mathbb{R}^3 \to [-1,1]^2$: 第 $j$ 个 view 的相机投影
- $\mathcal{F}_j$: 第 $j$ 个 view 的 DINOv2 特征图 (bilinearly sampled, 1024 维)
- $J$: 渲染视图集 (训练时 150 views, 来自 Hammersley sequence 球面采样)

**关键区别**: 与 prior work (Wang et al. 2023, Dutt et al. 2024) 不同, **voxelization 包括 interior**, 不只是 surface。这让模型能学到物体内部材料组成 (e.g. 花盆里是土, 树皮下是木头)。

### 3.2 Voxelization for Gaussian Splats (§6.1)

新提出的 voxelizer 三阶段:
1. 把 3D Gaussians 当作 solid ellipsoids, 取 99-percentile iso-surface 做 3D grid voxelization
2. 从球面采几十个视点渲染 depth maps
3. 用 depth maps carve away 外部 empty space, 但保留 unseen interior voxels, 形成 solid approximation

测试物体可在 31 ms 完成 voxelization (Table 1)。

### 3.3 Geometry Transformer 架构 (§F.2)

- 基于 **TRELLIS** encoder/decoder, backbone 用 TRELLIS 预训练权重初始化
- 操作在 $64^3$ voxel grid, 输入 1024 维 DINOv2 特征
- 12 层 transformer, 12 attention heads, MLP ratio 4:1
- **3D shifted window attention** (Swin-style), 8×8 local windows
- Latent channels: 2 (与 MatVAE latent 对齐)

处理 variable-length voxels: 定义最大序列长度 $L_N = 32768$。
- $L \le L_N$: 用全部 voxels
- $L > L_N$: 每个 epoch 开始随机采 $L_N$ 个 voxels (dynamic resampling), 让模型看到不同部位, 增大 "effective" max voxels

### 3.4 训练 loss (公式 4)

$$\mathcal{L}_F = \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \left\| \mu_\theta\left(\mathbf{F}(\mathbf{X}_\mathcal{S})_i\right) - \left((E_i, \nu_i, \rho_i)^N\right)^\top \right\|_2^2$$

变量:
- $\mathcal{S}$: 当前 iteration 采样的 voxel 索引集
- $\mathbf{X}_\mathcal{S} = \{(\mathbf{p}_i, \mathbf{f}_i)\}_{i \in \mathcal{S}}$: voxel 位置 + 特征
- $\mathbf{F}(\mathbf{X}_\mathcal{S})_i$: transformer 输出的第 $i$ 个 voxel 的 latent
- $\mu_\theta(\cdot)$: **冻结** 的 MatVAE decoder
- $((E_i, \nu_i, \rho_i)^N)^\top$: 标准化后的 ground truth

**关键**: MatVAE 是 frozen, 让 transformer 在 latent space 学习, 同时保证 decode 后的物理有效性。回传材料到原 representation (splat means, FEM tets) 用 nearest neighbor (§G.1), 避免 higher-order interpolation 引入 valid 之外的中间材料。

---

## 4. 训练数据生成 pipeline

### 4.1 Material Triplet Dataset (MTD)

100,562 个真实材料三元组, 来源: MatWeb, Wikipedia, Engineering Toolbox, Cambridge Materials Data Book。每个材料有 $E, \nu, \rho$ 的 valid 范围, 按范围大小成比例采样。

### 4.2 Geometry with Volumetric Materials (GVM)

1624 个 part-segmented 3D mesh (NVIDIA SimReady, Residential, Vegetation, Commercial), 共 8089 parts, 37M voxels 标注。

**Annotation pipeline** (Fig. 4): 每个 part 传给 Qwen2.5-VL-72B:
1. 完整物体 rendering
2. 该 part 的 PBR texture 贴到 sphere 上的 detail rendering
3. Material name (来自 USD asset)
4. MTD 中 3 个最近材料的 $(E, \nu, \rho)$ 范围 (基于 material name 检索)

VLM 输出该 part 的 $(E, \nu, \rho)$, 映射到该 part 内所有 voxels。

VLM annotation 误差 (Table 9): 与人工标注的小数据集对比, $\log E$ 误差 0.0295, $\nu$ 误差 0.0426, $\rho$ 误差 0.1348 — 接近人工标注质量。

### 4.3 与 Pixie (concurrent work) 对比 (Table 6, Fig. 15)

Pixie 用 in-context physics examples (一个固定的 material name → $(E, \nu, \rho)$ 表), 但部分值落在真实材料范围外。例如:
- Pixie 的 "tree/leaves": $\rho=200$ kg/m³, E=2e4 Pa — 但真实叶子无对应 MTD 范围
- Pixie 的 "soil": $\rho=1200$ — 真实 sandy loam 是 1600-1800

VoMP 用 VLM + MTD 检索 + texture cues, 显式约束到真实材料范围。

---

## 5. 实验结果深度分析

### 5.1 速度 (Table 1)

| Method | Time (s) |
|---|---|
| NeRF2Physics | 1454.55 (±1118) |
| PUGS | 1058.33 (±6.94) |
| Pixie | 201.63 (±27.74) |
| Phys4DGen* | 51.65 (±4.07) |
| **VoMP** | **3.59 (±1.36)** |

VoMP 时间分解: rendering 2.11s, voxelization 0.03s, DINOv2 0.86s, transformer 0.0082s, MatVAE 0.00032s。**Geometry Transformer + MatVAE 总共只要 8ms**, 主要瓶颈是 rendering 与 DINOv2 (可优化)。

VoMP 比 PUGS 快 295x, 比 NeRF2Physics 快 405x, 比 Pixie 快 56x。

### 5.2 主要指标 (Table 2, 完整数据集)

| Method | E ALDE↓ | E ALRE↓ | ν ADE↓ | ν ARE↓ | ρ ADE↓ | ρ ARE↓ |
|---|---|---|---|---|---|---|
| NeRF2Physics | 2.8000 | 0.1346 | — | — | 1432.03 | 1.0365 |
| PUGS | 3.3942 | 0.1688 | — | — | 3568.22 | 3.2429 |
| Phys4DGen* | 4.8967 | 0.2227 | 0.0407 | 0.1467 | 1865.57 | 1.4394 |
| **VoMP** | **0.3794** | **0.0409** | **0.0241** | **0.0818** | **142.70** | **0.0921** |

VoMP 在所有指标所有属性上大幅领先:
- E ALDE 比 NeRF2Physics 好 7.4x, 比 PUGS 好 8.9x
- ν ADE 比 Phys4DGen 好 1.7x (其他 baseline 不输出 ν)
- ρ ADE 比 PUGS 好 25x

§D.4 通过 486 次单元 cube FEM 仿真校准: **ALRE < 0.05 (E), ARE < 0.15 (其他)** 时仿真行为接近 ground truth。VoMP 的 E ALRE = 0.0409, ν ARE = 0.0818, ρ ARE = 0.0921 — 全部低于阈值, baseline 全部超阈值。

### 5.3 Material Validity (Fig. 6d)

测量每个预测 voxel 到 MTD 最近真实材料范围的相对误差:

| Method | log(E)↓ | ν↓ | ρ↓ |
|---|---|---|---|
| NeRF2Physics | 1.62 | — | 19.75 |
| PUGS | 1.87 | — | 13.24 |
| Phys4DGen* | 1.77 | 0.85 | 39.49 |
| Pixie | 11.90 | 3.46 | 46.58 |
| **VoMP** | **0.29** | **0.00** | **11.75** |

VoMP 的 ν 误差 **0.00** — 因为 MatVAE 训练时就保证 decode 出的值在真实材料 manifold 上。

### 5.4 Mass Estimation on ABO-500 (Fig. 6c)

| Method | ALDE↓ | ADE↓ | ARE↓ | MnRE↑ |
|---|---|---|---|---|
| NeRF2Physics | 0.736 | 12.725 | 1.040 | 0.564 |
| PUGS | 0.661 | 9.461 | 0.767 | 0.576 |
| Phys4DGen* | 0.664 | 9.961 | 0.825 | 0.566 |
| **VoMP** | **0.631** | **8.433** | 0.887 | **0.576** |

Mass = mean(ρ) × 已知 volume。VoMP 在 ALDE/ADE/MnRE 上领先, ARE 略输 PUGS (因 PUGS 系统性低估, ARE 不对称地惩罚 overestimate)。

---

## 6. Ablation 深度解读 (Table 7, 8)

### 6.1 MatVAE 设计 (Table 7)

去掉 NF: $\rho$ 的 $W_2$ 从 0.0172 涨到 0.0819 (4.7x), 因为 Gaussian posterior 无法匹配 heavy-tailed 密度分布。

去掉 TC penalty: $\nu$ 的 $W_2$ 从 0.0437 涨到 0.1052 (2.4x), 因为两个 latent 维度都在编码 $\rho$。

去掉 free nats: $\nu$ 的 $W_2$ 涨到 0.2064 (4.7x), 因为一个维度 collapse, 信息全压在另一个维度。

### 6.2 Geometry Transformer 设计 (Table 8)

**Image features** (从 random weight 训练):
- DINOv2: E ALDE = 0.2888
- CLIP: E ALDE = 0.2695 (略好, 但初始化 TRELLIS 后 DINOv2 更好)
- RGB colors: E ALDE = 1.2176 (4.2x 差), 因为颜色与材料属性非线性相关

**w/o MatVAE** (直接回归 R³): E ALDE 从 0.3765 涨到 1.1284 (3x), ν ADE 从 0.0250 涨到 0.0480 — 因为没有 manifold 约束, 预测可能落在 invalid 区。

**Normalization**:
- w/o log(E): E ALDE 从 0.3765 涨到 0.9033, 因为 heavy-tail 让大 E 值主导 loss
- w/o log(ρ): ρ ADE 从 113.38 涨到 549.95
- Z-score: 全属性严重退化 (E ALDE 0.8838, ν ADE 0.0814, ρ ADE 5269)

**Loss**: L1 替换 L2 → 所有指标退化 2-3x, 说明 squared error penalty 对 material regression 更有效。

---

## 7. Simulation 验证 (§G)

### 7.1 FEM 细节 (公式 24-27)

Corotational Hookean 模型:
$$W_{CR}(\mathbf{S}) = \underbrace{\mu \varepsilon : \varepsilon}_{\text{shear}} + \underbrace{\frac{\lambda}{2}\text{tr}(\varepsilon)^2}_{\text{volumetric}}, \quad \tau(\mathbf{S}) = 2\mu\varepsilon + \lambda\text{tr}(\varepsilon)\mathbf{I}$$

变量: $\mathbf{S}$ 是 stretch tensor (来自 $\mathbf{F} = \mathbf{U}\text{diag}(\sigma)\mathbf{V}^\top$ 的 $\mathbf{S} = \mathbf{V}\text{diag}(\sigma)\mathbf{V}^\top$), $\varepsilon = \mathbf{S} - \mathbf{I}$, $\mu, \lambda$ 是 Lamé 参数。

仿真用 incremental potential (公式 25), Newton 迭代求解 (公式 27), IPC 处理碰撞。

### 7.2 Simplicits for splat 仿真 (公式 29-32)

每个 quadrature 点从最近 voxel 取材料:
$$\lambda_q = \frac{E(\mathbf{X}_q)\nu(\mathbf{X}_q)}{(1+\nu(\mathbf{X}_q))(1-2\nu(\mathbf{X}_q))}, \quad \mu_q = \frac{E(\mathbf{X}_q)}{2(1+\nu(\mathbf{X}_q))}$$

Splat 变形: 把 rest anisotropy $\mathbf{L} = \mathbf{R}_0\text{diag}(\mathbf{s}_0)$ 通过局部 deformation gradient $\mathbf{F}$ 映射:
$$\Sigma = (\mathbf{F}\mathbf{L})(\mathbf{F}\mathbf{L})^\top + \varepsilon\mathbf{I}$$

$\varepsilon > 0$ 保证极端压缩下 covariance 仍 positive-definite。

### 7.3 Interpretation 实验 (§D.4, Fig. 18-21)

486 次单元 cube 仿真, 测量 volume change 与 potential energy:
- $\rho_{\text{new}} = \rho_0(1+\Delta)$ (线性 scaling)
- $\nu_{\text{new}} = \nu_0(1+\Delta)$ (线性 scaling)
- $E_{\text{new}} = E_0 e^\Delta$ (指数 scaling, 因 E 跨 6 个数量级)

总势能:
$$E_{\text{total}} = \int_\Omega W\,dV + \underbrace{\int_\Omega \frac{\rho}{2\Delta t^2}|\mathbf{u}^{n+1}-\mathbf{u}^n|^2 dV}_{\text{inertia}} + \underbrace{\int_\Omega -\rho\mathbf{u}\cdot\mathbf{g}\,dV}_{\text{gravity}} + \underbrace{\int_\Omega -\mathbf{u}\cdot\mathbf{f}_{\text{ext}}\,dV}_{\text{external work}}$$

场景: 140N 机器人夹持 (Franka Emika), 120N 跌落冲击, 330N 拉伸试验机, 200N tendon 张力。结果给出 relative error vs simulation deviation 的 confidence bounds, 用作 §6.3 中 ALRE/ARE 阈值的物理依据。

---

## 8. 关键 insight 总结

1. **Latent manifold 约束**: MatVAE 把 $(E, \nu, \rho)$ 压到 2D, 但不是为压缩 — 而是让 transformer 在一个 **物理有效** 的空间里学习, 任何 decode 出的值都是 valid 材料。这避免了 direct regression 的 "invalid 中间值" 问题。

2. **Interior voxelization**: 之前 multi-view feature lifting 只做 surface, 但材料属性本质上是 volume property。VoMP 把 interior 也 voxel 化 (用 flood-fill 或 splat carving), 让模型看到内部。

3. **Frozen decoder 解耦**: MatVAE 与 Geometry Transformer 分开训练。MatVAE 学 "什么是 valid 材料", Transformer 学 "哪个 voxel 该是哪个材料"。两者通过 2D latent 接口对接, 让 validity 约束不需要在 transformer loss 里显式表达。

4. **跨 representation 的统一**: 任何可 voxel + 可 render 的输入都能用。Splat 需要 carving voxelizer, mesh 用 flood-fill, SDF/NeRF 用各自 renderer。Transformer 看到的都是 $(\mathbf{p}_i, \mathbf{f}_i)$ 序列, 与 representation 无关。

5. **Simulator-agnostic**: VoMP 输出 ASTM 测量量, 可直接喂给 FEM (corotated, Neo-Hookean), MPM (FCR), XPBD (compliance), Simplicits。不需要 per-simulator 调参 (Fig. 2 问题)。

6. **Stochastic voxel subsampling**: 训练时每 epoch 重新采 $L_N = 32768$ 个 voxels, 让大物体也能训练, 同时 data augmentation 效果。

---

## 9. 局限与未来方向

- 固定 grid voxelization 限制输出分辨率, 高度 heterogeneous 区域会 oversmooth
- 假设 part-level 各向同性 (wood 实际是 anisotropic)
- 未预测 yield strength, shear modulus, thermal expansion
- 当 fast simulator 需要 simulator-specific scale 时, 需要后处理适配

---

## References & 进一步阅读

- **VoMP 项目页**: https://research.nvidia.com/labs/sil/projects/vomp
- **TRELLIS** (VoMP backbone): Xiang et al. 2025, https://arxiv.org/abs/2412.01506
- **DINOv2** (visual features): Oquab et al. 2024, https://arxiv.org/abs/2304.07193
- **3D Gaussian Splatting** (input representation): Kerbl et al. 2023, https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **NeRF2Physics** (baseline): Zhai et al. 2024, https://arxiv.org/abs/2404.04242
- **PUGS** (baseline): Shuai et al. 2025, https://arxiv.org/abs/2502.12231
- **Phys4DGen** (baseline): Lin et al. 2025, https://arxiv.org/abs/2411.16800
- **Pixie** (concurrent): Le et al. 2025, https://arxiv.org/abs/2508.17437
- **Simplicits** (splat simulator): Modi et al. 2024, https://doi.org/10.1145/3658184
- **β-TCVAE** (TC decomposition): Chen et al. 2018, https://proceedings.neurips.cc/paper_files/paper/2018/file/1ee3dfcd8a0645a25a35977997223d22-Paper.pdf
- **Normalizing Flows**: Rezende & Mohamed 2015, https://proceedings.mlr.press/v37/rezende15.html
- **β-VAE** (free nats): Higgins et al. 2017
- **IPC** (collision): Li et al. 2020, https://doi.org/10.1145/3386569.3392425
- **Qwen2.5-VL** (annotation VLM): Bai et al. 2025, https://arxiv.org/abs/2502.13923
- **Material property perception** (cognitive sci background): Fleming 2014, https://www.sciencedirect.com/science/article/pii/S0042698913002782
- **Ashby indices** (material efficiency): Ashby & Cebon 1993
- **libuipc** (FEM solver): Huang et al. 2024, https://doi.org/10.1145/3643028
- **Kaolin** (3D ML library): https://github.com/NVIDIAGameWorks/kaolin
- **gsplat** (splat renderer): Ye et al. 2024, https://arxiv.org/abs/2409.06765

如果你对某个部分想深挖 (比如 radial flow 的 invertibility 推导, Swin attention 在 sparse voxel 上的实现, 或者 Simplicits subspace 的构造), 我可以再展开。
