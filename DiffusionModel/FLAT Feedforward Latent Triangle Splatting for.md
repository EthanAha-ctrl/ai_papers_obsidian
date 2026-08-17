---
source_pdf: FLAT Feedforward Latent Triangle Splatting for.pdf
paper_sha256: 4d7f901d9349f2d7a37db57fe0ad5c21c309711b667104930b75cda245807891
processed_at: '2026-08-04T09:15:57-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，咱换个讲法。这 paper 其实讲的事特别朴素，只是被 academic 包装得有点绕。

## 一句话版本

**之前大家用 video diffusion 生成 3D 场景，都输出 3D Gaussian blobs——好看但没有真正的 surface，没法直接塞进 game engine。这篇 paper 说：我能让 video latent 直接 decode 出 triangle，triangle 就是 game engine 原生格式，直接能用。难的是怎么训练稳。**

## 为什么以前大家都做 Gaussian

先说现状。2024 年以来这条线（Wonderland、Lyra、Generative Gaussian Splatting、Bolt3D）都长一个样：

```
单张图 → 冻住的 video diffusion model → latent → 小 decoder → 3DGS
```

为什么全是 3DGS？因为 3DGS 训练特别 forgiving。一个 Gaussian 就是各向异性的"椭球 fog"，即使你 orientation、scale 全预测错，它投影到 image 上还是有面积，pixel loss 能合理回流 gradient，模型慢慢能学会。

这个 forgiving 性质让 feedforward 训练变得容易。3DGS 适合 train，但有问题：

- 它本质是 **semi-transparent volume**，没明确 surface
- 没 surface 就没法直接当 mesh 用
- 要从 Gaussian 抽 mesh 得用 TSDF / marching cubes / GS2Mesh 这类后处理，hyperparameter 极其敏感
- 室内室外得分别调参，常常直接 fail

所以 paper 的 motivation 就是：**能不能直接输出 triangle？triangle 就是 graphics pipeline 里最 native 的 primitive，输出直接能塞 Unreal、Unity、browser、iPhone 渲染。**

## 为什么直接做 triangle 难：gradient 病理

这里有个我觉得很漂亮的小观察，是整篇 paper 的核心 motivation。

Gaussian 跟 triangle 的根本区别：

**Gaussian 是 volumetric**，3D 空间里有体积。预测错了它仍然在 image 上有 footprint，gradient 回流。

**Triangle 是 flat**，2D 流形嵌在 3D 里。一旦 orientation 偏了，投影面积就趋近 0。模型早期 random initialization，rotation 在 SO(3) 上一通乱飘，triangle 全都"侧着"投影，render 出来全黑，gradient 全 0，模型永远学不到正确朝向，直接 diverge。

更糟的是，直接预测 3 个 vertex 还容易 collapse 成共线（degenerate triangle），面积 = 0，更没意义。

这俩问题就是 FLAT 要解决的全部。paper 的两个 key ingredients 一一对应：

1. **ray-centered parameterization**：同时解决 degeneracy + orientation 不稳定
2. **product window function**：解决 boundary 外 gradient 冻死

## Ingredient 1：ray-centered parameterization

核心想法：**给模型一个 good initialization，让它只学 residual**。

每个 decoder token 负责 image 上一个 2×2 区域，这个区域有一条 anchor ray（origin $\mathbf{r}_o$，direction $\mathbf{r}_d$）。token 不直接预测 3 个 vertex，而是预测：

- depth $D$：triangle 中心在 $\mathbf{r}_o + D \cdot \mathbf{r}_d$
- 3 个 shape 参数：构造一个 2D lower-triangular matrix $\mathbf{L}$
- rotation：不是 full 3D rotation，是绕 ray-tangent frame 的 residual tilt + spin

**Shape 部分的 Cholesky trick**

从 canonical equilateral triangle（正三角形）出发，用一个 2×2 lower-triangular 矩阵 $\mathbf{L}$ 变形：

$$\mathbf{L} = \begin{bmatrix} L_{00} & 0 \\ L_{10} & L_{11} \end{bmatrix}$$

- $L_{00} > 0, L_{11} > 0$ 强制为正
- 这就**保证 triangle 面积严格 > 0，永远不退化**
- $L_{10}$ 自由，提供 shear

这里 intuition 很干净：**把"面积 > 0"这个 geometric invariant 直接 baked 进 parameterization**，optimizer 完全不用操心 degeneracy。比"预测 3 个 vertex 然后验证面积正负"优雅太多。

这个思路跟 CVXNet、3D Convex Splatting 一脉相承——把 constraint 内化进 representation。

**Rotation 部分的 residual trick**

不预测 quaternion 或 full SO(3)，转而预测：
- 2 个 residual tilt angle
- 1 个 in-plane spin angle

为什么这样稳？因为 ray 本身就给了一个"大致正确"的 local frame：triangle 中心在 ray 上，自然朝向就是 ray-tangent plane。从这个 canonical 朝向出发做 small perturb，gradient 很容易找到正确方向。

直接预测 world-space rotation 的问题：random init 时 orientation 乱飘，render support 全消失，gradient 不回流，模型永远卡在 random 状态。Ablation Table 3 第一行就验证了这点——global rotation prediction 直接 diverge 到 PSNR < 10。

**这个设计哲学你应该很熟**：跟 residual connection、warm-start、pre-activation ResNet、v-prediction in diffusion 同构——**先给一个容易优化的起点，让模型学 delta**。

## Ingredient 2：product window function

第二个 trick 解决 triangle boundary 上的 gradient 病理。

triangle 渲染要 differentiable，不能直接用 hard coverage（0/1 不可微）。要 smooth 化。原始 Triangle Splatting [25] 用 sigmoid + max reduction，但 gradient 主要只流到"最近那条边对应的 vertex"，三个 vertex 收到不均衡的 gradient。

FLAT 的新 window function：

对每条 edge 算 signed distance $L_{m,i}(p)$（in triangle 为负），归一化：

$$u_{m,i}(p) = -\frac{L_{m,i}(p)}{\rho_m}$$

$\rho_m$ 是 inradius，归一化让所有 triangle scale-invariant。$u > 0$ 在 triangle 内，$u = 1$ 在 incenter。

然后 shifted clipping：

$$r_{m,i}(p) = \text{clamp}(u_{m,i}(p) + \epsilon, 0, 1)$$

$\epsilon > 0$ 是关键——**把 support 从 triangle 内部延伸到 boundary 外 $\epsilon \cdot \rho_m$ 距离**。即使 boundary 外，gradient 仍然非零。早期 training 即使 orientation 错了，gradient 还能回流到 vertex。

最后 product：

$$I_m(p) = \left(\prod_{i=1}^3 r_{m,i}(p)\right)^{\sigma_m}$$

$\sigma_m$ 是 sharpness，让模型自己学。三条 edge 都进乘积，gradient 均衡流到 3 个 vertex。

**这个套路你应该秒懂**：softmax 之于 argmax，sigmoid 之于 step function，Gumbel-Softmax 之于 discrete sampling，straight-through estimator 之于 quantization——**所有"把 hard operator smooth 化以让 gradient flow"的 trick 都是同一个 family**。核心 tradeoff 都是 smooth 程度 vs gradient quality vs 最终表达精度。FLAT 让模型自己学 $\sigma_m$ 是 elegant 解法。

## Architecture：直接 reuse Wan VAE decoder

这点我觉得很关键，是 paper 里被低估的 design choice。

之前的 feedforward 方法（Wonderland、Lyra）都是从 scratch 训 small transformer / Mamba decoder。FLAT 不干这个，转而**直接拿 Wan-2.1 video VAE 的 RGB decoder backbone，把最后输出 head 换成 triangle 参数 head**。

理由：
- Wan VAE decoder 已经学会 latent → RGB 的 mapping
- 它已经 implicit 捕获 local appearance 和 spatial pattern
- 把它当 strong initialization 比从 scratch 训小模型强
- 跟 ControlNet zero-conv 同思路：不破坏 pretrained latent，新分支慢慢学

Ablation 也验证了：用 Lyra 那个 LongLRM (Mamba-based) decoder 替换，PSNR 掉 0.21，说明 Mamba 容量不够 decode 复杂 non-volumetric primitive。

架构（Table 7）大概是：
- 输入：16-channel video latent + 32-channel Plücker embedding
- Plücker Adapter：3×3×3 CausalConv + SiLU + ZeroConv 跟 latent 相加
- Conv_in → Mid Block（2 ResBlock + SDPA attention）→ 3 个 Up Block
- 前两个 Up Block 做 temporal + spatial up（4× temporal expansion）
- 第三个 Up Block 是 identity（不做 spatial up），输出 2× stride，对应每 2×2 区域一个 triangle
- 用 CausalConv3D 保证 temporal causality

成本（Table 6）：768p, 49 views, 73M params, 9.8 TFLOPs, 在 H100 上 < 300ms。Video diffusion 本身要几十秒生成，scene decoding 几乎是 marginal cost。

**这个 design choice 跟 LLM 里 pretrain-then-finetune、ControlNet adapter、LoRA 一类思路完全同构**：已学好的 feature 是 cheap strong prior，应该尽量 reuse，不要重造轮子。

## Camera conditioning：RPPC

paper 用 Plücker ray embedding 给 decoder 注入 camera 信息。原始 Plücker：

$$\mathbf{r}_{\text{pl}} = (\mathbf{o} \times \mathbf{d}, \mathbf{d})$$

$\mathbf{o} \times \mathbf{d}$ 是 moment vector，把 position 信息藏在 cross product 里。FLAT 用 RPPC（来自 DiffusionGS）替换：

$$\mathbf{r}_{\text{rppc}} = (\mathbf{o} - (\mathbf{o} \cdot \mathbf{d})\mathbf{d}, \mathbf{d})$$

改成"ray 上离 world origin 最近的点"。position 和 direction 显式分离，decoder 容易学。然后 ref 和 dir 分别用 frozen VAE encoder encode，concat 之后用 lightweight fusion 投影回 decoder channel，通过 zero-init block 注入。

## Loss：photometric + depth + normal

$$\mathcal{L} = \lambda_{\text{rgb}}\mathcal{L}_2 + \lambda_{\text{perc}}\mathcal{L}_{\text{LPIPS}} + \lambda_D\mathcal{L}_D + \lambda_N\mathcal{L}_N + \lambda_O\mathcal{L}_O$$

权重 $\lambda_{\text{rgb}}=1.0, \lambda_{\text{perc}}=0.5, \lambda_D=0.01, \lambda_N=0.01, \lambda_O=0.001$。

- $\mathcal{L}_2$ + LPIPS：标准 photometric
- $\mathcal{L}_D$：MiDaS scale-invariant disparity loss（depth 监督）
- $\mathcal{L}_N$：normal loss，伪 GT 来自 NormalCrafter
  $$\mathcal{L}_N = \frac{\sum_i M_i(1 - \hat{\mathbf{n}}_i \cdot \mathbf{N}_i)}{\sum_i M_i}$$
  $\hat{\mathbf{n}}$ 是 rendered normal，$\mathbf{N}$ 是 pseudo-GT normal，$M_i = 1$ if $\alpha_i > 0.5$
- $\mathcal{L}_O$：opacity 正则，删 $\alpha < 0.4$ 的 triangle

**Normal supervision 是 geometric accuracy 的关键**。后面实验会看到，3DGS 用 finite-difference 算的 normal 几乎随机；2DGS 显式 surface 但直接 normal supervision 会 diverge，只能用 self-consistency loss；triangle + direct normal supervision 才能学到精确几何。

**伪 GT supervision 这个套路你应该熟**：DINO、BYOL、AlphaGo self-play 都是同一个 family——用 strong model 生成的伪标签 supervise 下游 model，bypass 真实 GT 缺失。

## 实验核心数据

### Table 1：Novel View Synthesis + Geometry

| Method | Repr | RE10K PSNR | RE10K LPIPS | DL3DV PSNR | Normal L1↓ | Normal Cos↑ |
|---|---|---|---|---|---|---|
| Wonderland | 3DGS | 17.15 | 0.292 | 16.64 | – | – |
| Lyra | 3DGS | 21.79 | 0.219 | 20.09 | – | – |
| **FLAT-3DGS variant** | 3DGS | **22.39** | **0.203** | 20.71 | 0.686 | 0.116 |
| FLAT-2DGS variant | 2DGS | 22.03 | 0.219 | 20.44 | 0.388 | 0.587 |
| **FLAT-Triangle** | Triangles | 21.45 | 0.245 | 20.04 | **0.211** | **0.853** |

几个 takeaway：

**1. FLAT-3DGS variant 是 visual SOTA**。PSNR 22.39，超过所有 prior method。这说明 training pipeline 没问题，3DGS 容错高所以容易出高分。

**2. Geometry：Triangles >>> 2DGS >>> 3DGS**。normal cosine 0.853 vs 0.587 vs 0.116。3DGS 是 blob，finite-difference 算 normal 几乎随机（0.116 接近 chance）；2DGS 有 surface 但 self-consistency loss 不够强；triangle + direct normal supervision → 0.853。

**3. Triangle 视觉略输 3DGS**（PSNR -0.94）。原因：Gaussian 天然 smooth，pixel-fitting 容易；triangle 是 hard primitive，thin / reflective / semi-transparent 内容难表达。Figure 8 给的例子特别直观——triangle 生成 sharper detail 但 PSNR 反而低。**PSNR 偏好 smooth，这是 metric 本身的 bias**。

**4. Triangle 几何精度碾压但 PSNR 仅小幅落后**——这是 representation tradeoff 的核心数据点。

### Table 2：Mesh 转换（最有实用价值的表）

| Repr | Conversion | Vertices | RE10K PSNR | DL3DV PSNR |
|---|---|---|---|---|
| 2DGS | TSDF | 5M | 15.89 | 12.00 |
| 3DGS | GS2Mesh | 4M | 14.18 | 12.31 |
| **Triangles** | **Ours** | **0.5M** | **21.23** | **19.71** |

**Triangles 直接转 mesh 比 GS methods 高 7+ dB，vertices 还少 10×**。GS methods 转 mesh 在 sparse view + 小场景上经常直接 fail，要逐场景调超参。FLAT 只需简单后处理就能在 iPhone、Pixel、browser 上实时渲染（Figure 12）。

这是 paper 真正的 practical contribution——**game-engine-ready 不是一个 marketing 词，是实测的 7 dB 差距**。

### Table 3：Ablation

| Setting | RE10K PSNR |
|---|---|
| Global rotation prediction | **< 10 (diverge)** |
| 3 Offsets 替代 Cholesky | 20.09 |
| 原 Triangle Splatting window | 20.65 |
| LongLRM (Mamba) decoder | 21.24 |
| **Ours (full)** | **21.45** |

几个 dramatic 结论：

- **Global rotation 直接 diverge**——验证 ray-centered residual 是必须的
- 直接预测 3 vertex offsets 替代 Cholesky → PSNR 掉 1.36，positivity constraint 是关键
- 用原 TS window（max reduction）→ PSNR 掉 0.80，product window 确实改善 gradient
- Mamba decoder 容量不够，VAE reuse 是正确选择

**所有 ablation 都验证了 paper 的两个核心 ingredient 是必需的**——不是 nice-to-have，是没它就 train 不起来。

### Table 4：Optional Post-Optimization

| Method | RE10K PSNR |
|---|---|
| FLAT | 21.45 |
| FLAT + 250 steps optimization | **23.01** |

250 步就 +1.56 PSNR，说明 feedforward prediction 已经是 strong initialization。Optional 但 cheap。

## 跟你常强调的 intuition 同构的地方

我整理一下这 paper 跟你以前讲过的几个 theme 的对应关系：

**1. Representation determines gradient flow**

3DGS vs Triangle 的本质差异是"哪种 representation 让 feedforward gradient 回流得好"。3DGS 容错高、容易 train；Triangle 严格、要精心 parameterize。这跟 LSTM vs Vanilla RNN、v-prediction vs ε-prediction、pre-LN vs post-LN Transformer 一类问题完全同构——**representation / parameterization 选择决定 optimization landscape 是否 navigable**。

**2. Parameterization as implicit initialization**

ray-centered + residual rotation + Cholesky positivity 把"好的 initialization"硬编码进 parameterization。模型不是从 random rotation 学起，是从"ray-tangent canonical triangle"开始 perturb。这跟 residual connection、warm-start、pre-activation ResNet 同构——**先给模型容易优化的起点，让它学 delta**。

**3. Smoothed surrogate for hard operator**

product window + $\epsilon$ shift 是把 hard triangle coverage smooth 化。跟 softmax-argmax、sigmoid-step、Gumbel-Softmax、STE-quantization 一个 family。**核心 tradeoff：smooth 程度 vs gradient quality vs 最终表达精度**。$\sigma_m$ 让模型自己学 sharpness 是 elegant 解法。

**4. Reuse pre-trained decoder > from scratch**

Wan VAE decoder 已经 implicit 学到 latent → pixel 的 structure，把它当 strong initialization 比从 scratch 训 small transformer 强。这跟 LLM pretrain-then-finetune、ControlNet zero-conv、LoRA 同构——**已学好的 feature 是 cheap strong prior，尽量 reuse**。

**5. Pseudo-GT self-supervision**

NormalCrafter / Metric3D 提供伪 GT supervision。跟 DINO、BYOL、AlphaGo self-play 同构——**用 strong model 生成的伪标签 supervise 下游 model，bypass 真实 GT 缺失**。

**6. Pipeline flexibility via decoder-swap**

FLAT 把 scene decoder 设计成"插拔式 head"——任何 Wan-2.1 变种（text-to-video、image-to-video、interactive、world-consistent）都能附加这个 decoder。这跟 LoRA、adapter、modular design 同构——**freeze backbone + swappable head = scaling efficiency**。Appendix A、Figure 5、6 演示了 text-to-3D scene 直接 work。

## Limitations（Appendix C）

老实说了几个：
- Thin / elongated / reflective / semi-transparent 内容仍然是 hard case（Figure 10）
- 输出 mesh 不是 watertight，10.6% non-manifold edges，local connectivity 可能 incomplete
- 单图 + 单 generated trajectory → occluded regions 仍 ambiguous
- Scale 有限，不是 persistent large-scale world（跟 WorldStereo、Lyra 2.0 那种方向还有差距）
- Dataset 比 modern video gen system 小很多

## 我看完的整体判断

这 paper 不 revolution，但是 solid engineering + 一个挺 elegant 的 insight。Insight 是：**video diffusion latent 里已经隐含 multi-view geometry，问题在于"把它映射到哪种 explicit primitive 可训练"。Triangle 是 graphics-native 的，但 feedforward 预测难。FLAT 给出让 training stable 的最小 ingredient 集，证明这条路 viable。**

最有价值的实验是 Table 1——**第一次 systematic 比较了 3DGS / 2DGS / Triangle 在同一 feedforward pipeline 下的 tradeoff**。Geometry 上 triangle 碾压（0.853 vs 0.116），visual 上略输（-0.94 PSNR），mesh 转换上碾压（+7 dB）。这数据对未来这条线的工作有指导意义。

两个核心 ingredient（ray-centered parameterization + product window）都是 well-motivated、well-validated 的工程贡献。ray-centered Cholesky 的设计特别漂亮，把 geometric constraint 内化进 parameterization 这个思路值得推广。

往后延伸的方向（paper 没明说但隐含）：
- 跟 world-consistent long video gen 结合做 persistent large-scale world
- Self-distillation：用 FLAT 输出当 pseudo-GT 训下一代 model（Lyra-style）
- Adaptive primitive：让 model 自己决定哪用 triangle、哪用 Gaussian（hybrid）
- Diffusion 直接在 triangle space 做，不再走 latent → decode triangle 的两阶段

## 相关链接

**项目主页**：
- FLAT: https://flat-splat.github.io

**Feedforward latent scene generation 这条线**：
- Wonderland (CVPR 2025): https://wonderland-3d.github.io
- Lyra (ICLR 2026): https://research.nvidia.com/labs/toronto-ai/lyra/
- Bolt3D (ICCV 2025): https://bolt3d.github.io
- Generative Gaussian Splatting (ICCV 2025)
- ViewCrafter: https://arxiv.org/abs/2409.02048
- Long-LRM (ICCV 2025)

**Scene representation**：
- 3D Gaussian Splatting (SIGGRAPH 2023): https://repo.mark96.xyz/siggraph/2023/3dgs
- 2D Gaussian Splatting (SIGGRAPH 2024): https://arxiv.org/abs/2403.17888
- Triangle Splatting (3DV 2025): https://arxiv.org/abs/2412.11458
- 3D Convex Splatting (CVPR 2025): https://arxiv.org/abs/2503.20747
- MeshSplatting: https://arxiv.org/abs/2512.06818

**Video diffusion backbone**：
- Wan-2.1: https://github.com/Wan-Video/Wan2.1
- Uni3C: https://arxiv.org/abs/2510.07224

**伪 GT provider**：
- NormalCrafter: https://arxiv.org/abs/2410.07561
- Metric3D v2: https://arxiv.org/abs/2404.15506
- MapAnything: https://arxiv.org/abs/2509.13414
- RealCam-Vid: https://arxiv.org/abs/2504.08212

**Dataset**：
- RealEstate10K: https://google.github.io/realestate10k/
- DL3DV-10K (CVPR 2024): https://github.com/DL3DV-10K/DL3DV-10K-Dataset

**Mesh extraction**：
- GS2Mesh (ECCV 2024): https://arxiv.org/abs/2402.12293
- MILO (TOG 2025): https://arxiv.org/abs/2404.09633

**Camera parameterization**：
- DiffusionGS: https://arxiv.org/abs/2412.14058

**World-scale 方向**：
- WorldStereo: https://arxiv.org/abs/2603.02049
- Genie (ICML 2024): https://arxiv.org/abs/2407.04494
- CAT3D: https://arxiv.org/abs/2405.10314
- Lyra 2.0: https://arxiv.org/abs/2604.13036

---

# FLAT：从 Video Diffusion Latent 直接 Decode Triangle Splats

## 1. 问题定位 & 核心动机

这篇 paper 想回答一个非常具体的问题：**能否从 frozen video diffusion model 的 compressed latent 中，单次 forward pass 直接 decode 出 explicit 的 surface primitive（triangle splats），而不再生成 volumetric 3D Gaussians？**

此前所有 feedforward latent scene decoder（Wonderland、Lyra、Generative Gaussian Splatting）都输出 3DGS。3DGS 是 anisotropic Gaussian blobs——适合 differentiable rendering 训练，但本质上是 semi-transparent volume，没有 well-defined surface。在 game engine / 标准 graphics pipeline 里直接不可用，需要复杂的 mesh extraction（TSDF、marching cubes、GS2Mesh），且 hyperparameter 极其敏感，室内室外场景甚至要单独调参。

FLAT 的核心 insight：**video diffusion latent 里已经隐含了 multi-view geometric structure**，问题在于"把它映射到哪种 explicit primitive 容易训练"。Triangle 是 graphics 里最经典的 primitive，直接预测 triangle 就避免了 mesh extraction 这一步。但 feedforward 预测 flat primitive 比预测 Gaussian 难得多——这是 paper 要解决的核心 technical challenge。

项目主页：https://flat-splat.github.io

## 2. 为什么 Feedforward Triangle 难：梯度病理

这是整篇 paper 的 motivation 核心，值得深挖。

**3DGS 的"宽容性"**：Gaussian 是各向异性 blob。即使 orientation / scale 预测错了，它投影到 image 仍然是一个有面积的高斯斑点，对 pixel 仍有贡献，gradient 合理回流。这是 3DGS 适合 feedforward 训练的根本原因——**容错性高**。

**Triangle 的"脆弱性"**：triangle 是 flat primitive（2D 流形 in 3D）。如果 orientation 偏离正确朝向，它在 image 上的投影面积会很小甚至接近 0，对 rendered image 几乎无贡献，gradient 趋近 0。训练早期尤其严重：world-space rotation 乱飘 → render support 消失 → gradient 不回流 → 模型永远学不到正确 orientation → divergence。

另外，直接预测三个 vertex 容易 collapse 成共线（degenerate triangle），面积 = 0，无意义。

paper 的两个核心 ingredients 就是针对这两个问题：
1. **Ray-centered parameterization**（同时解决 degeneracy + world-space rotation 不稳定）
2. **Product window function with $\epsilon$-shift**（解决 boundary 外 gradient flow 冻死）

## 3. 方法详解

### 3.1 Triangle 表示与渲染

每个 triangle $\mathbf{T}_m$ 由：
- 三个 vertex $\mathbf{v}_{m,i} \in \mathbb{R}^3$ ($i=0,1,2$)
- color $\mathbf{c}_m$
- smoothness $\sigma_m$（控制 splat 软硬）
- opacity $o_m \in [0,1]$

投影公式（Eq. 1）：
$$\mathbf{q}_{m,i} = \mathbf{K}(\mathbf{R}_t \mathbf{v}_{m,i} + \mathbf{t}_t)$$

变量含义：
- $\mathbf{q}_{m,i} \in \mathbb{R}^2$：第 $m$ 个 triangle 的第 $i$ 个 vertex 在 image plane 上的 2D 投影
- $\mathbf{K}$：相机 intrinsics（3×3）
- $\mathbf{R}_t, \mathbf{t}_t \in \text{SE}(3)$：第 $t$ 帧的 camera extrinsics
- 三个 $\mathbf{q}_{m,i}$ 形成 projected triangle $T_m^{2D}$

之后用 alpha compositing（front-to-back depth order）accumulate 所有 overlapping triangle，跟 3DGS / 2DGS 标准流程一致。

### 3.2 Ray-Centered Triangle Parameterization（核心贡献 #1）

每个 decoder token 对应 image 上一个 2×2 区域，有一个 anchor ray：
- origin $\mathbf{r}_o \in \mathbb{R}^3$
- direction $\mathbf{r}_d \in S^2$

token 预测的参数：
- depth $D$ → triangle 中心 = $\mathbf{r}_o + D \cdot \mathbf{r}_d$
- 3 个 shape 参数 → 构成 $\mathbf{L}$ 矩阵
- rotation 参数 → residual tilt + spin
- color $\mathbf{c}_m$, opacity $o_m$, sharpness $\sigma_m$

**Shape: Cholesky-style lower-triangular transform（Eq. 2）**
$$\mathbf{L} = \begin{bmatrix} L_{00} & 0 \\ L_{10} & L_{11} \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

- 从 2D canonical equilateral triangle 开始
- $L_{00} > 0, L_{11} > 0$ 强制 positive → 保证 transformed triangle 严格 positive area，**永远不退化**
- $L_{10}$ 自由 → 提供 shear degree of freedom
- transform 之后把 centroid 平移到 anchor point

**Intuition（这是 paper 最优雅的设计）**：lower-triangular 矩阵的 positivity constraint 直接编码了"三角形面积严格 > 0"这个几何不变量。直接预测 3 个 vertex 然后验证面积正负很尴尬；用 $\mathbf{L}$ 把约束内化进 parameterization，optimizer 不需要操心 degeneracy。这个思路跟 CVXNet [14]、3D Convex Splatting [26] 一脉相承——把 geometric invariant baked into parameterization。

**Rotation: residual tilt + spin around ray-aligned frame**

FLAT 不直接预测 quaternion 或 full 3D rotation matrix，转而预测：
- 2 个 residual tilt angles（局部倾斜校正）
- 1 个 in-plane spin angle（绕 ray 轴自转）

为什么这样更稳定？ray 本身给了一个"大致正确"的 local frame：triangle 中心在 ray 上，自然朝向就是垂直于 ray。在 ray-tangent frame 下做 small residual corrections，初始值接近"canonical ray-perpendicular triangle"，gradient 很容易找到正确方向。

直接预测 world-space rotation 的问题：早期 random initialization 时，orientation 在 SO(3) 上乱飘，render support 全部消失，gradient 不回流。Ablation Table 3 第一行验证了这点——global rotation prediction 直接 diverge 到 PSNR < 10。

这是 **"good initialization via parameterization"** 的经典思路，跟 NeRF positional encoding、Diffusion v-parameterization、residual connection 一类问题同构。

### 3.3 Product Window Function（核心贡献 #2）

这是 paper 的第二个 key ingredient，专门解决 triangle 边界 gradient flow 病理。

**Setup**：对 projected triangle $T_m^{2D}$，第 $i$ 条 supporting edge line：
- outward normal $\mathbf{n}_{m,i} \in \mathbb{R}^2$（使 $L_{m,i}(p) < 0$ inside triangle）
- signed distance $L_{m,i}(p) = \mathbf{n}_{m,i}^\top p + d_{m,i}$
- incenter $\mathbf{s}_m$
- inradius $\rho_m = -\max_i L_{m,i}(\mathbf{s}_m)$（incenter 到 edge 的归一化尺度）

**Step 1: Normalized edge response（Eq. 3）**
$$u_{m,i}(p) = -\frac{L_{m,i}(p)}{\rho_m}$$

变量含义：
- $u_{m,i}(p) > 0$ inside triangle
- $u_{m,i}(p) = 1$ at incenter
- 除以 $\rho_m$ 让所有 triangle 的 edge 距离归一化，scale-invariant

**Step 2: Shifted clipping（Eq. 4）**
$$r_{m,i}(p) = \text{clamp}(u_{m,i}(p) + \epsilon, 0, 1)$$

- $\epsilon > 0$：把 support 从 strict triangle 内部延伸到 boundary 外 $\epsilon \cdot \rho_m$ 距离
- 关键作用：boundary 外仍有非零 derivative，早期 training 即使 orientation 错了，gradient 仍能回流

**Step 3: Product window（Eq. 5）**
$$I_m(p) = \left(\prod_{i=1}^3 r_{m,i}(p)\right)^{\sigma_m}$$

- $\sigma_m$：控制 splat 软硬程度（越大越接近 hard triangle）
- 三个 $r_{m,i}$ 都进入乘积 → 每个 pixel 收到来自 full triangle 的信号
- gradient 对三个 vertex 全部 multiplicative contribution → 均衡回流

**为什么比原始 Triangle Splatting [25] 的 max-reduction 好？**

原始 TS formulation 大致是 $I(p) \propto \text{sigmoid}(\min_i u_{m,i}(p)/\tau)$，gradient 主要流到最近 edge 的对应 vertex。问题：
- boundary 附近 pixel：gradient 主要 pull 一个 vertex
- 远离 boundary 的 inside pixel：gradient 信号弱
- 三个 vertex 收到不均衡 gradient → 容易 collapse

Product window 的优势：
- 三条 edge 都进入乘积 → gradient 流到全部 3 vertex
- $\epsilon$ shift → boundary 外仍有 gradient → early training 不冻死

这跟 softmax 之于 argmax、sigmoid 之于 step function、smooth L1 之于 L2 一类"smoothing to improve gradient flow"的常见思路完全同构。Karpathy 你在 CS231n 讲 differentiable rendering 时强调过这个：**hard rasterizer 不可微，必须 smooth 化；smooth 的方式决定了 gradient quality**。

### 3.4 Architecture：Reuse Wan-2.1 VAE Decoder

**关键 design choice**：FLAT 不从 scratch 训 small transformer / Mamba decoder（Wonderland、Lyra 那样），转而 reuse Wan-2.1 [55] 的 video VAE RGB decoder backbone，把最后输出 head 换成"输出 triangle 参数"。

理由（重要直觉）：
- Wan VAE decoder 已经学会 latent → RGB pixel 的 mapping
- 它已经 implicit 捕获 local appearance + spatial patterns
- 把 output head 换成 geometry prediction，相当于把 decoder pre-trained features transfer 到新任务
- 跟 ControlNet 的 zero-conv 同思路：不破坏 pretrained latent，新分支慢慢学

**架构细节（Table 7）**：
- 输入 video latent $x_v \in \mathbb{R}^{B \times 16 \times T' \times H' \times W'}$（16-channel VAE latent）
- Plücker embedding $x_p \in \mathbb{R}^{B \times 32 \times T' \times H' \times W'}$
- Plücker Adapter Fusion：3×3×3 CausalConv + SiLU + ZeroConv Addition
- Conv_in → Mid Block (2 ResBlock + SDPA attention) → 3 Up Blocks
- Temporal upsampling 在前两个 Up Block（4× temporal expansion total）
- 最后一个 Up Block 用 Identity（不做 spatial up），输出 2× stride → 每 2×2 区域一个 triangle
- 用 CausalConv3D（只对过去帧 padding）保持 temporal causality

**为什么移除最后一个 upsampling stage？**减少 primitive 数量，节省计算，且 2×2 区域对应一个 triangle 在分辨率上已经足够 fine。

**Computational cost（Table 6）**：768p, 49 views → 73M params, 9.8 TFLOPs, 2.7 FPS（< 300ms on H100）。Video diffusion 本身要几十秒生成，scene decoding 几乎是 marginal cost。

### 3.5 Camera Conditioning: RPPC Parameterization

**Plücker ray embedding（Eq. 6）**
$$\mathbf{r}_{\text{pl}} = (\mathbf{o} \times \mathbf{d}, \mathbf{d})$$

- $\mathbf{o} \in \mathbb{R}^3$：ray origin
- $\mathbf{d} \in S^2$：ray direction
- $\mathbf{o} \times \mathbf{d}$：moment vector（隐式编码 ray position）

**RPPC（Reference Point Closest to origin, Eq. 7）**
$$\mathbf{r}_{\text{rppc}} = (\mathbf{o} - (\mathbf{o} \cdot \mathbf{d})\mathbf{d}, \mathbf{d})$$

intuition：Plücker moment $\mathbf{o} \times \mathbf{d}$ 不直观（位置信息藏在 cross product 里）。换成"ray 上离 world origin 最近的点" $\mathbf{o} - (\mathbf{o} \cdot \mathbf{d})\mathbf{d}$ 后，position 和 direction 显式分离，decoder 容易学。来自 DiffusionGS [6]。

之后（Eq. 8, 9）：
- $\mathbf{r}^{\text{ref}}$ 和 $\mathbf{r}^{\text{dir}}$ 分别用 frozen VAE encoder $\mathcal{E}$ encode → $\mathbf{E}^{\text{ref}}, \mathbf{E}^{\text{dir}} \in \mathbb{R}^{T' \times C \times H' \times W'}$
- concat 后通过 lightweight fusion $\phi$ 投影回 decoder channel → $\mathbf{E}^{\text{cam}}$
- 通过 zero-initialized block 注入 decoder

### 3.6 Loss Function（Eq. 10）

$$\mathcal{L} = \lambda_{\text{rgb}}\mathcal{L}_2 + \lambda_{\text{perc}}\mathcal{L}_{\text{LPIPS}} + \lambda_D\mathcal{L}_D + \lambda_N\mathcal{L}_N + \lambda_O\mathcal{L}_O$$

权重：$\lambda_{\text{rgb}}=1.0, \lambda_{\text{perc}}=0.5, \lambda_D=0.01, \lambda_N=0.01, \lambda_O=0.001$

- $\mathcal{L}_2$ + $\mathcal{L}_{\text{LPIPS}}$：photometric（pixel-wise L2 + perceptual）
- $\mathcal{L}_D$：MiDaS scale-invariant disparity loss（depth）
- $\mathcal{L}_N$：normal loss，伪 GT 来自 NormalCrafter [4]
  $$\mathcal{L}_N = \frac{\sum_i M_i(1 - \hat{\mathbf{n}}_i \cdot \mathbf{N}_i)}{\sum_i M_i}$$
  - $\hat{\mathbf{n}}_i$：rendered normal
  - $\mathbf{N}_i$：pseudo-GT normal
  - $M_i = 1$ if $\alpha_i > 0.5$（只算有效 coverage 区域）
- $\mathcal{L}_O$：opacity 正则化（high-res stage），同时 remove $\alpha < 0.4$ 的 triangle

**Normal supervision 是 geometric accuracy 的关键**：3DGS 是 blob，finite-difference 算出的 normal 几乎随机（cosine 0.116）；2DGS 显式 surface 但直接 normal supervision 会 diverge（paper 试过），只能用 normal self-consistency（cosine 0.587）；Triangles 直接 normal supervision + flat primitive → cosine 0.853。

### 3.7 Opaque Mesh Conversion（Test-time）

Feedforward 输出 semi-opaque triangle soup → 完全 opaque mesh：

1. 全局设 $\sigma = 0.5$
2. 用相同 photometric objective 做 50 iterations aggressive opacity selection
3. 把 opacity 推向 binary，删除 low-support triangle
4. 生存下来的 triangle snap 到 near-opaque，boundary 附近 densify
5. Stitch：merge 互相最近的 boundary vertex + prune floaters
6. Brief repair：调整 vertex position + color 恢复 image fidelity

输出能在 iPhone 15 / Google Pixel / Browser 上实时渲染（Figure 12）。

## 4. 实验结果分析

### 4.1 Novel View Synthesis + Geometry Quality（Table 1）

| Method | Repr | RE10K PSNR | RE10K LPIPS | DL3DV PSNR | Geo L1↓ | Geo Cos↑ |
|---|---|---|---|---|---|---|
| Wonderland | 3DGS | 17.15 | 0.292 | 16.64 | – | – |
| Lyra | 3DGS | 21.79 | 0.219 | 20.09 | – | – |
| **FLAT-3DGS** | 3DGS | **22.39** | **0.203** | 20.71 | 0.686 | 0.116 |
| FLAT-2DGS | 2DGS | 22.03 | 0.219 | 20.44 | 0.388 | 0.587 |
| FLAT-Tri | Triangles | 21.45 | 0.245 | 20.04 | **0.211** | **0.853** |

**核心观察**：

1. **FLAT-3DGS variant 是 SOTA visual quality**——这证明 training pipeline 本身没问题。3DGS 容错高、容易预测，PSNR 上限高。
2. **Geometry：Triangles >> 2DGS >> 3DGS**（cosine 0.853 vs 0.587 vs 0.116）。3DGS normal 几乎随机；2DGS 有改善但 self-consistency loss 不够强；Triangles + direct normal supervision → 几何最精确。
3. **Visual quality 上 Triangle 略低于 3DGS**（PSNR -0.94）。原因：Gaussian blob 天然 smooth，pixel-fitting 容易；triangle 是 hard primitive，thin / reflective / semi-transparent 内容难表达。Figure 8 的例子显示 triangles 往往生成 sharper detail 但 PSNR 反而低——PSNR 偏好 smooth。
4. **Triangle 几何精确度大幅领先但 PSNR 仅小幅落后**——这是 representation tradeoff 的核心数据点。

### 4.2 Opaque Mesh Conversion（Table 2）

| Repr | Conversion | Vertices | RE10K PSNR | DL3DV PSNR |
|---|---|---|---|---|
| 2DGS | TSDF | 5M | 15.89 | 12.00 |
| 3DGS | GS2Mesh | 4M | 14.18 | 12.31 |
| **Triangles** | **Ours** | **0.5M** | **21.23** | **19.71** |

**Triangles 直接输出 mesh 比 GS methods 高 7+ dB，且 vertices 少 10×**。3DGS/2DGS 转 mesh 需要密集覆盖 + 调超参，FLAT 只需简单后处理。这是 paper 的核心实用价值——**真正 game-engine-ready**。

### 4.3 Ablation（Table 3）

| Architecture | Window | Repr | Rotation | RE10K PSNR |
|---|---|---|---|---|
| Ours | Ours | Ours | **Global** | **< 10 (diverge)** |
| Ours | Ours | 3 Offsets | Residual | 20.09 |
| Ours | Triangle Splatting orig | Ours | Residual | 20.65 |
| LongLRM (Mamba) | Ours | Ours | Residual | 21.24 |
| **Ours (full)** | **Ours** | **Ours** | **Residual** | **21.45** |

**关键 ablation 结论**：

1. **Global rotation prediction → diverge**（直接 PSNR < 10）——验证了 ray-centered residual 的必要性。这是最 dramatic 的 ablation。
2. 改成 "3 Offsets"（直接预测 3 vertex offsets）→ PSNR 降 1.36，training 不稳定。Cholesky-style positivity constraint 是关键。
3. 用原始 triangle splatting window（max reduction）→ PSNR 降 0.80。Product window + $\epsilon$ shift 确实改善 gradient flow。
4. LongLRM (Mamba-based) decoder → PSNR 降 0.21。Mamba 容量不够 decode 复杂 non-volumetric primitive，Wan VAE reuse 是正确选择。

### 4.4 Post Optimization（Table 4）

| Method | RE10K PSNR |
|---|---|
| FLAT | 21.45 |
| FLAT + Optimization (250 steps) | **23.01** |

短短 250 steps optimization 就 +1.56 PSNR，说明 feedforward prediction 已经是 strong initialization。Optional 但 cheap。

## 5. Limitations（Appendix C）

- Thin / elongated structures、reflections、semi-transparent regions 仍是 hard case（Figure 10）
- 输出 mesh 不是 watertight，local connectivity 可能 incomplete（10.6% non-manifold edges）
- 单图 + 单 generated trajectory → occluded regions 仍 ambiguous
- Scale 有限，不是 persistent large-scale world（跟 WorldStereo [73]、Lyra 2.0 [50] 对比）
- Dataset 比 modern video gen system 小很多

## 6. 跟 Karpathy 视角的同构

这篇 paper 的设计哲学跟你常强调的几个 intuition 高度共鸣：

### (1) Representation determines gradient flow

3DGS vs Triangles 的差异本质是"哪种 representation 让 feedforward gradient 信号回流得好"。3DGS 容错高 → 容易训练；Triangles 严格 → 需要精心 parameterize。这跟 RNN 中 LSTM vs Vanilla RNN、Diffusion 中 v-prediction vs ε-prediction、Transformer 中 pre-LN vs post-LN 一类问题完全同构——**representation / parameterization 选择决定了 optimization landscape 是否 navigable**。

### (2) Parameterization as implicit initialization

ray-centered + residual rotation + Cholesky positivity 把"好的 initialization"硬编码进 parameterization。模型不需要从 random rotation 学起，转而从"ray-tangent canonical triangle"开始 perturb。这跟 residual connection、warm-start、pre-activation ResNet 一类思路同构——**先给模型一个容易优化的起点，再让它学 delta**。

### (3) Smoothed surrogate for hard operator

Product window + $\epsilon$ shift 是把"hard triangle coverage"smooth 化以让 gradient flow。跟 softmax 之于 argmax、sigmoid 之于 step、Gumbel-Softmax 之于 discrete sampling、straight-through estimator 之于 quantization 一类思路同构。**核心 tradeoff：smooth 程度 vs gradient quality vs 最终表达精度**——$\sigma_m$ 让模型自己学 sharpness 是 elegant 的解法。

### (4) Decoder reuse > from scratch

Transfer learning 的经典 wisdom。Wan VAE decoder 已经 implicit 学到 latent → pixel 的 structure，把它当 strong initialization 比从 scratch 训 small transformer 强。这跟 LLM 中"pretrain → finetune"、ControlNet zero-conv 同构——**已学好的 feature 是 cheap strong prior，应该尽量 reuse**。

### (5) Self-conditioning on pseudo-GT

NormalCrafter / Metric3D 提供伪 GT supervision，相当于 self-distillation / self-training。这跟 DINO、Bootstrap Your Own Latent、AlphaGo self-play 一类思路同构——**用 strong model 生成的伪标签 supervise 下游 model，bypass 真实 GT 缺失问题**。

### (6) Pipeline flexibility via decoder-swap（Appendix A）

FLAT 的 decoder-swap 设计让任何 Wan-2.1 变种（text-to-video、image-to-video、interactive、world-consistent）都能直接附加 scene decoder。这跟 LoRA、adapter 一类 modular design 同构——**freeze backbone + swappable head = scaling efficiency**。

## 7. 参考 & 相关工作链接

**项目主页与论文**：
- FLAT 项目主页：https://flat-splat.github.io
- Wan-2.1：https://github.com/Wan-Video/Wan2.1
- Uni3C（FLAT 用的 video model）：https://arxiv.org/abs/2510.07224

**Feedforward latent scene generation**：
- Wonderland (CVPR 2025)：https://wonderland-3d.github.io
- Lyra (ICLR 2026)：https://research.nvidia.com/labs/toronto-ai/lyra/
- Lyra 2.0：https://arxiv.org/abs/2604.13036
- Generative Gaussian Splatting (ICCV 2025)
- Bolt3D (ICCV 2025)：https://bolt3d.github.io
- ViewCrafter：https://arxiv.org/abs/2409.02048

**Scene representation**：
- 3D Gaussian Splatting (SIGGRAPH 2023)：https://repo.mark96.xyz/siggraph/2023/3dgs
- 2D Gaussian Splatting (SIGGRAPH 2024)：https://arxiv.org/abs/2403.17888
- Triangle Splatting (3DV 2025)：https://arxiv.org/abs/2412.11458
- MeshSplatting：https://arxiv.org/abs/2512.06818
- 3D Convex Splatting (CVPR 2025)：https://arxiv.org/abs/2503.20747
- Radiance Foam：https://arxiv.org/abs/2412.01047

**Pseudo-GT providers**：
- NormalCrafter：https://arxiv.org/abs/2410.07561
- Metric3D v2：https://arxiv.org/abs/2404.15506
- MapAnything：https://arxiv.org/abs/2509.13414
- RealCam-Vid：https://arxiv.org/abs/2504.08212

**Datasets**：
- RealEstate10K：https://google.github.io/realestate10k/
- DL3DV-10K (CVPR 2024)：https://github.com/DL3DV-10K/DL3DV-10K-Dataset
- S3OD (ICLR 2026)

**Camera parameterization**：
- DiffusionGS：https://arxiv.org/abs/2412.14058
- Long-LRM (ICCV 2025)

**Mesh extraction from Gaussians**：
- GS2Mesh (ECCV 2024)：https://arxiv.org/abs/2402.12293
- MILO (TOG 2025)：https://arxiv.org/abs/2404.09633

**World-scale & interactive**：
- WorldStereo：https://arxiv.org/abs/2603.02049
- Genie (ICML 2024)：https://arxiv.org/abs/2407.04494
- CAT3D：https://arxiv.org/abs/2405.10314

## 8. 总体评价

这篇 paper 在 representation tradeoff 上给出了第一个 systematic 的 feedforward 测量：**Triangle 在几何精度上碾压 3DGS/2DGS（cosine 0.853 vs 0.116/0.587），在视觉质量上仅小幅落后（PSNR -0.94），且直接输出 game-engine-ready mesh**。两个核心 ingredient（ray-centered parameterization + product window）是 well-motivated、well-validated 的工程创新。

最重要的 take-away：**video diffusion latent 已经隐含 multi-view geometry**，问题在于"把它映射到哪种 explicit primitive 可训练"。FLAT 证明了 triangle splats 是 viable 的，并给出了让 training stable 的最小 ingredients 集。这个 insight 对下一代 generative 3D system 有指导意义——**geometry-aware representation + careful parameterization = direct game-engine output**。

Potential future direction（paper 没明说但隐含）：
- 把 triangle decoder 跟 world-consistent long video gen [50, 73] 结合，做 persistent large-scale world
- Self-distillation：用 FLAT 输出当 pseudo-GT 训下一代 model（Lyra-style）
- Adaptive primitive：让 model 自己决定哪用 triangle、哪用 Gaussian（hybrid representation）
- Diffusion 直接在 triangle space 做（不再 latent → triangle decode，直接 denoise triangle soup）
