---
source_pdf: FLASHWORLD.pdf
paper_sha256: 4f9239a7ae5b2ee0ccdbe83750bfa7d29fa8aa1a5dd45d1627ac765ec3a70023
processed_at: '2026-08-04T08:58:15-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲FlashWorld

## 一句话总结

**用distillation把"图像质量好但不连贯"和"3D连贯但图像糊"这两种方法的长处拼到一起，顺便把生成速度从几分钟压到9秒。**

---

## 问题是什么？

你要从一张图或者一句话生成一个完整的3D scene（可以360度看的那种）。现在的两条路都有要命的毛病：

### 路线A：先多视角，再重建
先用diffusion model一口气生成24张不同角度的图，然后拿这些图去做3D reconstruction。

**问题**：24张图是各画各的，没有3D空间的硬约束。你让模型画一把椅子的正面和背面，正面三条腿背面四条腿——因为生成时根本没人告诉它"这是同一个3D物体"。重建之后texture全是噪点，geometry歪歪扭扭。

### 路线B：边画边渲染
在denoising的每一步都把当前的3D Gaussian渲染出来监督，强制3D consistency。

**问题**：rendering gradient非常noisy。3DGS的rasterization是个复杂的非线性操作，gradient回传到latent space时信号被搅得一塌糊涂。结果就是——3D是consistent了，但图像糊得像水彩画，细节全丢。

### 还有个共性问题：慢
不管A还是B，都要跑几十步denoising，每步还要rendering/encoding，几分钟起步。应用层面基本不可用。

---

## FlashWorld的核心trick

作者的insight很漂亮：**A和B其实是互补的**。

- A图像质量好，但3D不连贯
- B 3D连贯，但图像糊

那就让A当teacher教B。具体怎么做？用DMD2（Distribution Matching Distillation）。

### 直觉解释DMD

想象teacher是米其林大厨，student是学徒。传统distillation是"学徒做菜，大厨尝一口说差多少盐"。DMD更狠——它让学徒做菜，大厨做菜，然后比较两道菜的"风味分布"差异，用这个差异来推学徒往哪个方向调。

数学上就是match两个diffusion model的score function $s(x,t) = \nabla \log p_t(x)$。teacher的score代表"真实数据该长什么样"，student的score代表"我生成的分布长什么样"，两者差值就是student要移动的方向。

这里teacher是MV-oriented mode（质量高），student是3D-oriented mode（consistency好）。student学到了teacher的visual quality，同时保住了自己的3D consistency。

---

## 怎么把两种mode塞进一个model？

### Dual-mode Pre-training

同一个DiT backbone，共享weights，只是output head不同：

- **MV mode**：DiT输出denoised latent $\hat{Z}_{MV}$，直接和GT latent比MSE
- **3D mode**：DiT输出auxiliary feature $\mathcal{F}$，喂给3DGS decoder得到Gaussian参数，渲染novel view后和GT image比MSE

两个loss交替训，一个backbone学会两个task。

### 为什么用Video Diffusion初始化而非Image Diffusion？

这是很关键的design choice。Image diffusion model（比如Stable Diffusion）只懂单张图，对多视角一致性没有任何先验。Video diffusion model（WAN2.2-5B）见过海量视频，**video的连续帧本质上就是多视角**——相机移动一帧就是换一个viewpoint。所以它天然懂"同一个scene不同视角应该consistent"。

附带好处：video VAE压缩率更高（spatial 16x downsampling），同样显存能塞更多views（24张）和更高分辨率（480P）。

---

## Few-step Generation怎么做到9秒？

### 核心思路：轨迹缩短 + 噪声注入

传统diffusion要50-1000步。FlashWorld只用4步：$\{t_1, t_2, t_3, t_4\} = \{1000, 900, 750, 500\}$。

每步做两件事：
1. **Denoise**：3D-oriented student预测当前step的clean Gaussian，渲染出来再encode回latent
2. **Forward diffusion**：给刚denoise的latent重新加噪到下一个timestep

这种"denoise一下再加噪"的trick来自Consistency Models和Latent Consistency Models。直觉是：纯few-step生成容易sample到bad local minimum，反复加噪相当于让模型"多试几次"找到更好的解。

### DMD2加速

DMD2在DMD基础上加了GAN loss。纯DMD的score matching擅长match整体分布，但对高频细节（texture、边缘）不够敏锐。GAN的discriminator是个learned的细节critic，专治糊图。

组合loss = DMD score matching + GAN adversarial + R1 regularization。R1是为了防止discriminator太强把generator训崩。

---

## Cross-Mode Consistency Loss解决什么bug？

作者发现：光用DMD+GAN训3D-oriented student，会出现floating artifacts——一些Gaussian飘在场景外面，不贴任何surface。

原因：3D mode的gradient路径是
```
latent → DiT → 3DGS decoder → rendering → VAE encoder → latent
```
中间经过3DGS rasterization这个高度非线性操作，gradient非常noisy，optimizer容易被推到奇怪的position。

**解决方案**：顺便低频率更新一个MV-oriented student（共享DiT backbone），它的预测不经过rendering，gradient干净。然后加一个loss强制：
$$\|3D\text{ mode的输出latent} - MV\text{ mode的输出latent}\|^2$$

MV mode相当于一个"稳定锚"，把3D mode拉回合理区域。weight $\lambda = 0.1$，很小，作为regularization而非主导信号。

---

## OOD Data Co-training怎么搞？

### 问题
多视角数据集就那么几个（MVImgNet, RealEstate10K, DL3DV10K），domain窄。真实用户输入可能是任意风格图片、任意文本、任意相机轨迹。model在这些OOD输入上容易崩。

### Trick
拿海量单视角图片+文本（来自proprietary video dataset和Echo4O），配上**随机camera trajectory**（从多视角数据集采样，或用预定义轨迹），和正常multi-view数据混合训。

关键细节：**OOD数据训的时候关掉GAN loss**。因为GAN discriminator只见过真实multi-view分布，对合成的"单图+随机轨迹"会over-reject，把generator带偏。

训练比例：multi-view : OOD = 2:1。

---

## 训练compute的实际情况

- 64块H20 GPU
- bf16精度
- FSDP（Fully Sharded Data Parallel）+ activation checkpointing
- Pre-training 20k steps，3天
- Post-training 10k steps，2天
- 总计大约7680 GPU-hours

这个compute level基本只有大厂能玩。academic lab要复现很困难。

---

## 实验结果的几个takeaway

### 1. 速度
9秒/scene，在单H20上。对比：
- CAT3D: minutes-hours（per-scene optimization）
- Director3D: 7分钟（SDS++ refinement）
- Wonderland: 10秒但quality差
- WonderJourney/LucidDreamer: 6分钟（iterative inpainting）

### 2. Quality
WorldScore平均分最高（68.72），同时速度最快。这是quality-speed的Pareto frontier。

### 3. Ablation的story
- 纯MV diffusion：quality还行但3D重建后texture noisy
- 纯3D diffusion：quality最差，糊
- 纯MV distillation：distillation反而放大inconsistency，更差
- 加CMC loss：quantitative差不多但qualitative去掉floating artifacts
- 加OOD data：generalization明显提升，T3Bench和WorldScore上不掉分

---

## 我觉得有问题的地方

### 1. Compute门槛太高
7680 GPU-hours的training cost，加上用了proprietary video dataset（非开源），这个工作基本无法被academic lab复现。这限制了对方法的验证和改进。

### 2. 3D Consistency得分不高
WorldScore的3D Consistency Ours=85.87，低于LucidDreamer的90.37。作者解释说因为没用depth supervision。但这恰恰说明——3D consistency好只是相对于MV-oriented方法的"重建后inconsistent"而言，跟真正用了geometric prior的方法比还有差距。

### 3. 24 Views的Inference Memory
24 views × 480P × 1024 channel aux feature，光feature map就很大。inference时的显存占用和batch=1的真实throughput没明确report。9秒是wall time，但需要多少显存没说。

### 4. 对fine geometry的局限
作者自己承认mirror reflection和articulated objects不行。这其实是3DGS表示本身的limitation——anisotropic Gaussian对thin structure和specular surface建模困难。需要更好的representation或者更强的geometric prior。

---

## 这篇paper的真正贡献

技术上，FlashWorld把几个已有idea组合得很巧妙：
1. Dual-mode training（来自Dual3D）
2. Video diffusion初始化（提升prior quality）
3. DMD2 distillation（加速+提质量）
4. Cross-mode consistency（稳定gradient）
5. OOD data co-training（泛化）

单独看每个组件都不新，但"用MV mode当teacher distill 3D mode student"这个组合是新的，而且实验证明work。从long-term看，这种"两个paradigm互相teach"的思路可能启发后续更多工作——比如用NeRF-oriented mode distill 3DGS-oriented mode，或者用text-to-video teacher distill 3D scene student。

但最让我impressed的还是工程层面：把5B参数video diffusion model、3DGS decoder、DMD2 GAN、FSDP全塞进一个训练pipeline，64 GPU跑5天，这工程量在academic paper里很少见。

---

# FlashWorld: 高质量3D Scene生成的秒级框架深度解析

## 1. 核心Intuition：在两个paradigm之间架桥

FlashWorld的根本insight是**3D scene generation领域长期存在两个paradigm各自的inherent limitation**：

- **MV-oriented paradigm** (CAT3D, Bolt3D, Wonderland)：先用diffusion model生成multi-view images，再通过3D reconstruction得到scene。问题是生成阶段没有explicit 3D constraint，导致multi-view inconsistency，reconstruction后texture noisy、geometry不准。
- **3D-oriented paradigm** (Dual3D, Director3D, DiffusionGS)：在denoising过程中结合differentiable rendering，直接生成3D representation (3DGS/NeRF)。问题是visual quality blurry，因为rendering gradient noisy且optimization landscape复杂。

FlashWorld的key idea：**用distillation把两种paradigm的优势merge**——MV-oriented mode作为teacher提供high visual quality的score gradient，3D-oriented mode作为student保证3D consistency。这是一种cross-mode knowledge transfer，同时通过few-step generation获得inference speed。

参考link：
- DMD原始paper: https://arxiv.org/abs/2311.18828
- DMD2: https://arxiv.org/abs/2405.14867
- 3D Gaussian Splatting: https://repo.samuelgarcia.dev/lib/3d-gaussian-splatting

## 2. Architecture Overview解析

整体framework分为三个stage：

### 2.1 Backbone: WAN2.2-5B-IT2V
作者选择从video diffusion model而非image diffusion model初始化，这是关键design choice：
- Video diffusion自带temporal consistency prior
- VAE compression rate更高 (spatial downsampling factor=16)，支持24 views @ 480P
- Convergence更快

参考：WAN video model https://arxiv.org/abs/2503.20314

### 2.2 DiT + 3D Attention Blocks + 3DGS Decoder

Architecture的核心pipeline：

```
Multi-view images X 
  → VAE Encoder → Z (latents)
  → Forward diffusion → Z_t = α_t·Z + σ_t·ε
  → DiT (with 3D attention) → Ẑ_MV + auxiliary feature F
  → 3DGS Decoder D_G(F) → {τ, q, s, α, c} (Gaussian参数)
  → pixel-aligned Gaussians: μ = o + τ·d
  → Rendering R(G, C_novel) → X_novel_pred
```

这里3DGS decoder是从原始VAE decoder D初始化的，但首尾conv layer重新初始化以适配额外features和Gaussian output channels。

## 3. 数学公式深度解析

### 3.1 Forward Diffusion Process

$$x_t = F(x, t) = \alpha_t x + \sigma_t \epsilon$$

- $x$: clean data
- $x_t$: noisy data at timestep $t$
- $\alpha_t, \sigma_t$: 控制t时刻signal-to-noise ratio的系数，随$t$增大$\alpha_t$减小$\sigma_t$增大
- $\epsilon \sim \mathcal{N}(0, I)$: standard Gaussian noise
- 论文使用flow matching schedule (Lipman et al., 2023)而非传统DDPM schedule

### 3.2 Score Function

$$s(x_t, t) = \nabla_{x_t} \log p_t(x_t) = -\frac{x_t - \alpha_t \mu(x_t, t)}{\sigma_t^2}$$

这是Tweedie's formula推出的posterior mean，$\mu(x_t, t)$是denoised estimate。Score表示当前data distribution的gradient field——DMD distillation的核心就是match teacher和student的score function。

### 3.3 DMD Loss Gradient

$$\nabla \mathcal{L}_{\mathrm{DMD}} = -\mathbb{E}_t \left( \int \left( s_{\mathrm{real}}(F(G_\theta(z), t), t) - s_{\mathrm{fake}}(F(G_\theta(z), t), t) \right) \frac{dG_\theta(z)}{d\theta} dz \right)$$

变量解释：
- $z \sim \mathcal{N}(0, I)$: 输入noise
- $G_\theta(z)$: few-step student generator
- $F(\cdot, t)$: forward diffusion操作
- $s_{\mathrm{real}}$: frozen teacher的score (这里用MV-oriented mode)
- $s_{\mathrm{fake}}$: 动态更新的fake score network (在student生成samples上训练)
- $\frac{dG_\theta(z)}{d\theta}$: generator关于参数的Jacobian

Intuition：这个gradient推动student生成的sample分布向teacher分布靠拢，相当于reverse KL divergence的stochastic approximation。

### 3.4 Dual-mode Pre-training Loss

**MV-oriented mode loss:**
$$\mathcal{L}_{\mathrm{MV}} = \mathbb{E}_{\mathcal{X}, t, \epsilon, y, C} \left[ \|\mathcal{Z} - \hat{\mathcal{Z}}_{\mathrm{MV}}\|^2 \right]$$

- $\mathcal{X}$: multi-view images batch
- $\mathcal{Z} = E(\mathcal{X})$: encoded latents
- $\hat{\mathcal{Z}}_{\mathrm{MV}}$: DiT预测的clean latents
- $y$: conditioning (text or image)
- $C$: camera parameters (Plücker coordinates raymaps)

**3D-oriented mode loss:**
$$\mathcal{L}_{\mathrm{3D}} = \mathbb{E}_{\mathcal{X}, t, \epsilon, y, \mathcal{C}} \left[ \|\mathcal{X}_{\mathrm{novel}} - R(\mathcal{G}, \mathcal{C}_{\mathrm{novel}})\|^2 \right]$$

- $\mathcal{G} = \{\mu, q, s, \alpha, c\}$: 3D Gaussians集合
  - $\mu$: 位置 (由$\mu = o + \tau d$得到，$o$是camera origin，$d$是ray direction，$\tau$是predicted depth)
  - $q$: rotation quaternion
  - $s$: scale
  - $\alpha$: opacity
  - $c$: spherical harmonics coefficients (color)
- $R$: 可微分rendering操作
- $\mathcal{X}_{\mathrm{novel}}, \mathcal{C}_{\mathrm{novel}}$: GT novel-view images和对应camera

注意：3D-oriented mode下，denoised latents通过$\hat{\mathcal{Z}}_{\mathrm{3D}} = E(R(\mathcal{G}, \mathcal{C}))$重新获得，确保latent与rendered image一致。

### 3.5 Cross-Mode Consistency Loss

$$\mathcal{L}_{\mathrm{CMC}} = \mathbb{E}_{z, t, \epsilon, y, \mathcal{C}, i} \left[ \| E(R(G_{\theta, 3D}(\mathcal{Z}_{t_i}, t_i, y, \mathcal{C}), \mathcal{C})) - G_{\theta, \mathrm{MV}}(\mathcal{Z}_{t_i}, t_i, y, \mathcal{C}) \|^2 \right]$$

- $i$: few-step schedule中的第$i$步
- $G_{\theta, 3D}$: 3D-oriented student
- $G_{\theta, \mathrm{MV}}$: MV-oriented student (低频率更新)
- weight $\lambda = 0.1$

这个loss的关键作用：MV-oriented mode的prediction不经过rendering，所以gradient更稳定，能regularize 3D-oriented mode避免floating artifacts。

### 3.6 Few-step Generation Schedule

Inference时，3D-oriented mode用N=4步交替denoising和noise injection：

```
Z_{t_1} = z ~ N(0, I), schedule = {1000, 900, 750, 500}
for i = 1 to N:
    Ẑ_{t_i} = E(R(G_{θ,3D}(Z_{t_i}, t_i, y, C), C))  # denoise + render + encode
    Z_{t_{i+1}} = α_{t_{i+1}} · Ẑ_{t_i} + σ_{t_{i+1}} · ε  # forward step
final Gaussians = G_{θ,3D}(Z_{t_N}, t_N, y, C)
```

这是从Consistency Models / LCM借鉴的technique：few-step trajectory + 噪声注入增强sample quality。

## 4. 训练Configuration细节

### 4.1 Pre-training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | WAN2.2-5B-IT2V |
| Input views | 24 keyframes |
| Spatial downsampling | 16x |
| Aux feature channels | 1024 |
| Learning rate (DiT + 3DGS decoder) | 2×10⁻⁶ |
| Weight decay | 1×10⁻⁶ |
| Adam β₁, β₂ | 0.9, 0.95 |
| Warmup | 1,000 steps |
| Total steps | 20,000 |
| Training time | ~3 days |
| Precision | bf16 |

### 4.2 Post-training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Few-step schedule | {1000, 900, 750, 500} |
| Fake score update per generator step | 4× |
| Generator LR | 1×10⁻⁶ |
| Discriminator LR | 5×10⁻⁷ |
| GAN loss weight | 5×10⁻³ |
| Total steps | 10,000 |
| Training time | ~2 days |
| Task ratio (MV : 3D-input : 3D-novel) | 1:3:1 |
| Multi-view : OOD data ratio | 2:1 |

### 4.3 Hardware

- 64× NVIDIA H20 GPUs
- Batch size 64
- FSDP (Fully Sharded Data Parallel) + activation checkpointing

## 5. Datasets

**Multi-view训练数据：**
1. **MVImgNet** (https://github.com/aim-uofa/MVImgNet): object-centric, 480×704
2. **RealEstate10K** (https://google.github.io/realestate10k/): indoor, 704×480, frame stride [5-12]
3. **DL3DV10K** (https://github.com/DL3DV-10K/DL3DV): general scenes, 704×480, frame stride [2-4]

**OOD数据：**
1. Proprietary video dataset (images + text) + RealEstate10K/WorldScore trajectories
2. **Echo4O** (https://arxiv.org/abs/2508.09987) + **WildRGBD** (https://arxiv.org/abs/2403.17984) trajectories

## 6. 实验结果深度分析

### 6.1 Text-to-3D Generation (Table 1)

| Method | T3Bench Q-Align IQA | T3Bench CLIP Score | WorldScore Q-Align IQA | WorldScore Score | Time |
|--------|---------------------|---------------------|------------------------|------------------|------|
| Director3D | 3.24 | 27.84 | 2.51 | 29.05 | 7 min |
| Prometheus | 2.34 | 24.85 | 2.45 | 28.07 | 15 sec |
| **Ours** | **4.12** | **27.68** | **3.96** | **29.13** | **9 sec** |

观察：
- FlashWorld在Q-Align IQA上几乎翻倍Prometheus，说明visual quality优势明显
- Time cost = 9 sec，仅比WonderWorld慢一点但远快于iterative方法
- CLIP Aesthetic上Ours不是最高，作者解释该metric偏向smooth outputs

### 6.2 WorldScore Benchmark (Table 2)

| Method | 3D Consist. | Photometric | Object Ctrl | Content Align | Style | Subjective | Avg | Time |
|--------|-------------|-------------|-------------|---------------|-------|------------|-----|------|
| WonderJourney | 80.60 | 79.03 | 34.81 | 38.37 | 67.52 | 61.49 | 60.30 | 6 min |
| LucidDreamer | 90.37 | 90.20 | 43.48 | 59.41 | 66.41 | 48.02 | 66.32 | 6 min |
| WonderWorld | 86.91 | 85.56 | 52.09 | 56.82 | 75.92 | 41.28 | 66.43 | 10 sec |
| **Ours** | 85.87 | 86.72 | 49.61 | 53.96 | **81.52** | 54.63 | **68.72** | 9 sec |

关键观察：
- 3D Consistency Ours略低，因为baselines用monocular depth estimation (Depth Anything等)与evaluation protocol对齐，而Ours只用RGB supervision
- Style Consistency最高 (81.52)，反映visual fidelity优势
- Average最高且速度最快，是efficiency-quality的最佳trade-off

### 6.3 Ablation Study (Table 3) 深度解读

| Variant | Description | T3Bench IQA | WorldScore IQA |
|---------|-------------|-------------|-----------------|
| A | w/ MV-Diff (MV-oriented diffusion) | 3.11 | 2.48 |
| B | w/ 3D-Diff (3D-oriented diffusion) | 2.61 | 2.74 |
| C | w/ MV-Dist (MV-oriented distillation) | 3.46 | 3.06 |
| D | w/o CMC (no cross-mode consistency) | 4.12 | 3.90 |
| E | w/o OOD | 3.98 | 3.66 |
| F | Full model | 4.12 | 3.76 |

深度分析：
- **A vs F**: MV-Diff IQA 3.11 vs Full 4.12 → MV inconsistency严重降低quality
- **B vs F**: 3D-Diff IQA 2.61 (最低) → 3D-oriented mode本身quality差，印证paper核心claim
- **C vs F**: MV-Dist IQA 3.46 → distillation加剧MV pipeline的inconsistency
- **D vs F**: w/o CMC在T3Bench上持平4.12，但qualitative显示floating artifacts
- **E vs F**: w/o OOD在WorldScore上掉分3.66 vs 3.76，证明OOD strategy提升generalization

### 6.4 速度优势分解

| Stage | Time |
|-------|------|
| Pre-training (20k steps, 64 GPU) | 3 days |
| Post-training (10k steps, 64 GPU) | 2 days |
| **Inference (single H20)** | **9 sec** |

对比：
- CAT3D: minutes-hours (MV diffusion + per-scene 3D optimization)
- Bolt3D: ~sec level但quality差
- Wonderland: 10 sec级但Gaussian artifacts
- FlashWorld: 9 sec，10-100× faster than previous methods

## 7. 与相关工作的技术对比

### 7.1 vs. Dual3D (Li et al., 2024a)
Dual3D同样采用dual-mode LDM，但从image diffusion model初始化，VAE compression rate低，views数量受限。FlashWorld从video diffusion初始化，natural temporal prior + 高compression rate。

Dual3D没有distillation加速，仍需multi-step denoising。FlashWorld通过cross-mode distillation降到4步。

参考：https://arxiv.org/abs/2405.09874

### 7.2 vs. Director3D (Li et al., 2024b)
Director3D生成pixel-aligned Gaussians后还需SDS++ refinement，时间7 min。FlashWorld通过distillation直接4-step generation，9 sec完成。

参考：https://arxiv.org/abs/2405.09874 (Director3D NeurIPS 2024)

### 7.3 vs. DiffusionGS (Cai et al., 2024)
DiffusionGS在每一步输出pixel-aligned 3DGS保证3D consistency，但visual quality受限于rendering gradient noise。FlashWorld通过CMC loss + teacher distillation缓解此问题。

参考：https://arxiv.org/abs/2411.14384

### 7.4 vs. Cycle3D (Tang et al., 2025)
Cycle3D统一generation-reconstruction，在multi-step denoising中集成3D reconstruction。思想类似3D-oriented mode，但没有distillation加速，仍需50-step inference。

参考：https://arxiv.org/abs/2412.02795

## 8. 关键Design Choice的Intuition

### 8.1 为什么用Video Diffusion初始化？

1. **Temporal prior**: video data中consecutive frames本质就是multi-view，model已经learn了view consistency
2. **VAE efficiency**: video VAE设计为高compression (16x spatial)，支持更多views (24) 和更高resolution (480P)
3. **Convergence**: 不需要从scratch学习view correspondences

### 8.2 为什么DMD2而非Consistency Distillation？

- Consistency Model要求trajectory上任意点都能直接映射到clean data，对3DGS这种discrete representation有困难
- DMD通过score matching + GAN组合，对distribution shape更灵活
- DMD2的GAN objective提供高频细节，对texture quality关键

### 8.3 为什么需要Cross-Mode Consistency？

3D-oriented mode的gradient经过rendering pipeline，包含：
1. Differentiable rasterization (3DGS)
2. VAE encoder back to latent
3. 这些操作引入noisy gradient，导致floating Gaussians

MV-oriented mode的预测不经过rendering，gradient更clean。CMC loss让3D mode的输出latent match MV mode的输出latent，相当于用MV mode作为anchor稳定3D mode的training。

### 8.4 为什么OOD data中省略GAN loss？

GAN discriminator是在multi-view真实数据分布上训练的，对single-view image + random trajectory这种合成conditioning可能over-reject。省略GAN loss防止distribution mismatch破坏generator。

## 9. Limitations & Future Directions

作者承认的limitation：
1. Scene diversity受限于dataset coverage
2. Fine-grained geometry, mirror reflection, articulated objects仍有困难

可能改进方向：
- 融合depth prior (Depth Anything V2: https://arxiv.org/abs/2406.09414)
- Autoregressive generation扩展scene规模 (类似AR-1-to-3: https://arxiv.org/abs/2503.12929)
- 4D extension (4D-GS: https://arxiv.org/abs/2310.08528)

## 10. Critique与Potential Issues

作为critical reader，我注意到几个可能的问题：

### 10.1 Training Cost高昂
64× H20训练5天 (3+2 days)，总GPU-hours约 64×24×5 = 7680 GPU-hours。这远高于普通academic lab承受能力，reproducibility困难。

### 10.2 Evaluation Protocol争议
- WorldScore metrics作者re-evaluate随机frame而非anchor frame，可能overestimate baselines的weakness
- 3D Consistency Ours低分，作者归因于无depth supervision，但这也提示geometric accuracy可能不如baselines

### 10.3 Generalization Claim验证不充分
OOD strategy主要在T3Bench和WorldScore上验证，但真实world capture的images (e.g. 手机拍摄)的evaluation缺失。

### 10.4 24 Views的Memory开销
24 views @ 480P latent space计算，即使bf16 + FSDP + activation checkpointing，64× H20仍可能bottleneck。inference时24 views的rendering开销未明确report。

### 10.5 与最新SOTA的head-to-head缺失
没与3DV-2025的SplatFlow、VideoRFSplat直接比较，这些方法可能使用不同evaluation protocol。

## 11. 总结

FlashWorld的核心贡献是把MV-oriented和3D-oriented两个paradigm通过cross-mode distillation统一，解决了长期存在的quality-consistency trade-off。技术栈包括：

1. **Video diffusion backbone** (WAN2.2) - 强prior + 高效VAE
2. **Dual-mode pre-training** - 共享backbone学两个task
3. **DMD2-based distillation** - few-step generation加速
4. **CMC loss** - 稳定3D-oriented gradient
5. **OOD co-training** - generalization

最终实现9 sec/scene inference + SOTA quality，是3D scene generation走向real-time application的重要milestone。

参考链接汇总：
- FlashWorld项目主页: https://imlixinyang.github.io/FlashWorld-Project-Page/
- DMD2: https://arxiv.org/abs/2405.14867
- WAN video model: https://arxiv.org/abs/2503.20314
- 3DGS: https://repo.samuelgarcia.dev/lib/3d-gaussian-splatting
- WorldScore: https://arxiv.org/abs/2504.00983
- DL3DV: https://github.com/DL3DV-10K/DL3DV
- MVImgNet: https://github.com/aim-uofa/MVImgNet
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- Consistency Models: https://arxiv.org/abs/2303.01469
- Diffusion Forcing: https://arxiv.org/abs/2404.03728
- Flow Matching: https://arxiv.org/abs/2210.02747
