---
source_pdf: Sharp Monocular View Synthesis.pdf
paper_sha256: b9c6f8c44c708f76d52d7dc5123271aea0607a617bf8753f397f4f4f0553e86d
processed_at: '2026-08-12T05:44:49-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SHARP 用人话讲讲

## 一句话概括

你给一张照片,SHARP 在 1 秒内吐出一个 3D 场景,你可以戴着 AR/VR 头显晃脑袋看它,画面清晰到能看清头发丝,渲染 100 FPS 没压力。

---

## 这事儿为什么难

咱们先把问题拆开看。

**Single image to 3D 这件事本身就有歧义**。一张 2D 照片,你不知道后面藏了什么、玻璃后面是啥、反光里反射的是远处的山还是近处的人。无限个 3D 场景能投影出同一张 2D 图。

历史上大家怎么解决?两条路:

**第一条路:Per-scene optimization**。NeRF、3D Gaussian Splatting 这些方法给你一堆图(几十张),花几分钟到几小时优化出一个 scene。质量炸裂,但慢,且需要多图。

**第二条路:Diffusion prior**。最近大火的 ViewCrafter、Gen3C、SVC 这些方法,把 diffusion model 训练成"看过单图就能脑补新视角"的模型。质量也很猛,甚至能合成原图里看不到的内容。问题是慢 —— Gen3C 一张图要 14 分钟。

Apple 这帮人就想:AR/VR 头显的 headbox 其实很小,你戴着头显晃脑袋,最多移动半米。这个场景下,我根本不需要 "walk around the object",也不需要 "hallucinate 远处看不见的东西"。我只需要 nearby views 极其 sharp、极其快、有 metric scale。

于是 SHARP 就盯死了这个 niche。

Reference: [SHARP GitHub](https://github.com/apple/ml-sharp)

---

## 核心思路:回归路线的逆袭

SHARP 的选择是 **feed-forward regression**。一次 forward pass,网络直接吐出 1.2M 个 3D Gaussians 的参数,完事儿。

这听起来好像很朴素 —— 不就是 CNN + decoder 预测一堆数字吗?很多人觉得 regression 路线打不过 diffusion。但 SHARP 用一堆工程细节把这个路线推到了 SOTA,在 nearby views 上把 Gen3C 这种 diffusion 巨兽按在地上摩擦,速度快 910 倍。

我读这篇 paper 的最大感受是:**工程细节 + 问题定义 + scale,三件事凑齐了,regression 路线在特定 niche 依然能赢**。

Reference: [3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

---

## 架构全景:其实就是四个 Lego 拼起来

我先把整体画出来:

```
照片 ──→ Depth Pro Encoder ──→ 特征图
                                  │
                          ┌───────┴───────┐
                          ▼               ▼
                    Depth Decoder   Gaussian Decoder
                    (输出 2 层 depth)  (输出所有 Gaussian 的 delta)
                          │               │
                          ▼               │
                    Depth Adjustment      │
                    (训练时用,inference 时丢弃)
                          │               │
                          └───┬───────────┘
                              ▼
                      Gaussian Initializer
                      (从 depth + image 拼出粗 Gaussians)
                              │
                              ▼
                      Gaussian Composer
                      (粗 Gaussians + delta = 最终 Gaussians)
                              │
                              ▼
                      Differentiable Renderer
                      (渲染出新视角图,算 loss)
```

四个模块:Depth Pro encoder (提取特征) + Depth decoder (出 depth) + Depth adjustment (训练辅助) + Gaussian decoder (出 refinement delta)。总参数 702M,可训练 340M。

为什么这么设计?我一个个拆。

Reference: [Depth Pro (Bochkovskii et al., ICLR 2025)](https://arxiv.org/abs/2410.19115)

---

## 模块一:Depth Pro Backbone —— 站在巨人的肩膀上

Depth Pro 是 Apple 自己之前发的一个 metric monocular depth estimator,质量极强。SHARP 直接拿它的 backbone 当 feature extractor。

这里有个关键决定:**部分 unfreeze**。

- Low-resolution image encoder:unfreeze(允许 fine-tune)
- Patch encoder:frozen(保住预训练特征)
- Normalization layers:frozen(稳定)
- Depth decoder:unfreeze

为什么不全部 freeze?因为 Depth Pro 训练时优化的目标是 "depth MAE",而 SHARP 关心的是 "view synthesis 质量"。这俩目标不完全对齐。比如透明玻璃的 depth 本来就 ill-defined,Depth Pro 会预测一个 "平均" depth,但 view synthesis 更希望网络用 context 线索推理出合理的 layer 分配。

Ablation (Table 13) 显示:unfreezing 把 ScanNet++ 的 DISTS 从 0.084 降到 0.064,定性看 (Figure 12) boundary artifacts 和 reflection 都好了。

**Intuition**: 这就是 "foundation model + task-specific fine-tuning" 的典型 pattern。你拿一个强大的 generalist backbone,然后让它适配你的下游目标。Pretrained features 是金矿,直接 frozen 浪费了,完全从头训又太慢,部分 unfreeze 是 sweet spot。

Reference: [DPT (Ranftl et al., ICCV 2021)](https://arxiv.org/abs/2103.13413)

---

## 模块二:两层 Depth —— 处理 Disocclusion

Depth decoder 输出 $\hat{\mathbf{D}} \in \mathbb{R}^{2 \times H \times W}$,两通道。第一层是 primary visible surface,第二层是 occluded content(原本被前景挡住的东西)。

为啥要两层?因为新视角下,原本被挡住的部分会露出来。如果只有一层 depth,这部分就是黑的或者 warp 出 garbage。两层 depth + 两层 Gaussians(每个 pixel 2 个 Gaussian),一个管前面,一个管后面,就能 fill in 这些 disocclusion。

这和 multiplane images (MPI) 的思路类似,但 SHARP 只用 2 层,因为 nearby view 的 disocclusion 范围小,2 层够了。多了参数爆炸,少了 disocclusion 处理不好。

Reference: [AdaMPI (Han et al., SIGGRAPH 2022)](https://arxiv.org/abs/2204.07192)

---

## 模块三:Depth Adjustment —— 我觉得最 elegant 的设计

这个模块是整篇 paper 我最喜欢的地方。

**问题**: Monocular depth 有 inherent ambiguity。网络遇到拿不准的地方,倾向于预测 "mean scale" —— 多个 plausible depth 的平均值。这个 mean scale depth 看着 "差不多",但拿去 warp 就出现 smearing、banding、糊一片。

**传统 C-VAE 思路**: 训一个 posterior $q(z|x, y_{GT})$,输入 ground truth,输出 latent $z$,加 KL divergence bottleneck,强制 $z$ 只编码 "消解 ambiguity 的最小信息"。

**SHARP 的简化神操作**:

1. 把 latent vector $z$ 换成 **per-pixel scale map** $\mathbf{S} \in \mathbb{R}^{H \times W}$
2. 训练时 input 是 predicted depth $\hat{\mathbf{D}}$ 和 GT depth $\mathbf{D}$,输出 scale map $\mathbf{S}$
3. Adjusted depth: $\bar{\mathbf{D}} = \mathbf{S}(\hat{\mathbf{D}}, \mathbf{D}) \odot \hat{\mathbf{D}}$
4. **Inference 时,这个模块直接换成 identity** —— 网络必须学会自己输出正确的 depth

等等,这啥意思?训练时有个 "作弊通道" —— 当 depth estimator 错了,scale map 能矫正它,让 view synthesis loss 能正确 backprop。但 inference 时这个信息拿不到,所以网络必须学会 "没有矫正也能 work"。

怎么强制?用 information bottleneck loss:

$$\mathcal{L}_{\text{scale}} = \mathbb{E}_{p \sim \Omega} [| \mathbf{S}(p) |]$$

$$\mathcal{L}_{\nabla \text{scale}} = \sum_{k=1}^{6} \mathbb{E}_{p \sim \Omega_{\downarrow k}} [| \nabla \mathbf{S}_{\downarrow k}(p) |]$$

第一个 loss 惩罚 scale map 幅度(让 $\mathbf{S}$ 尽量接近 1,也就是 identity)。第二个 loss 在 6 个 scale 上惩罚 TV(让 $\mathbf{S}$ 尽量平滑)。weight 是 $\lambda_{\text{scale}} = 0.1$, $\lambda_{\nabla \text{scale}} = 5.0$ —— 平滑性比幅度重要 50 倍。

**Intuition**: 这其实是 "训练有辅助,inference 要 self-contained" 的 pattern。你给网络训练时一个 cheating channel,但通过 bottleneck 强制它学到 "不靠 cheating 也能 work" 的能力。这是一种 "learned curriculum" —— 训练初期网络菜,cheating channel 帮它;训练后期网络强了,bottleneck 让 cheating channel 越来越 useless,网络必须 self-sufficient。

Ablation (Table 11) 显示: ScanNet++ 上 DISTS 从 0.077 → 0.064,Figure 10 定性看 sharper。

这个 pattern 我觉得可以推广到很多场景 —— 任何 "训练时有 privileged info,inference 时没有" 的情况都能用。比如 medical imaging 里训练时有 expert annotation,inference 时只有 raw image。比如 robotics 里训练时有 sim privileged state,inference 时只有 sensors。

Reference: [C-VAE (Sohn et al., NeurIPS 2015)](https://proceedings.neurips.cc/paper/2015/hash/8d55a249e6ba2455a8b07c401be6bb38-Abstract.html)

---

## 模块四:Gaussian Initializer —— 在 Normalized Space 里操作

给定 adjusted depth $\bar{\mathbf{D}}$ 和 input image $\mathbf{I}$,这个模块拼出 base Gaussians $\mathbf{G}_0$。

几个关键设计:

**1. Min-pooling 做 depth downsample**: 从 1536² 到 768²,depth 用 min-pool 而不是 average pool。因为 foreground object 应该取近的(min depth),average 会让 foreground "晕开" 到 background。

这个细节看着不起眼,但其实很关键。Average pooling 一个 foreground edge,depth 值会被拉远,unproject 后 Gaussian 位置就漂移了。

**2. Unproject 不用 intrinsics**:
$$\mu(i, j) = [i \cdot \bar{\mathbf{D}}'(i, j), j \cdot \bar{\mathbf{D}}'(i, j), \bar{\mathbf{D}}'(i, j)]^T$$

注意:这里 **deliberately 不用 intrinsics matrix**。Gaussians 是在 normalized space 里预测的。

然后渲染时,把 intrinsics 吸收进 projection matrix:
$$\mathbf{P} = \mathbf{K}_{\text{tgt}} \mathbf{E}_{\text{tgt}} \mathbf{E}_{\text{src}}^{-1} \mathbf{K}_{\text{src}}^{-1}$$

**Intuition**: 这相当于 "uncalibrated" 表示。对于 view synthesis,我们不关心 absolute metric coordinates,只关心 "从 source view 到 target view 的相对变换"。网络在 normalized space 学一次,适用于任何 FOV 的图像。这让 zero-shot 泛化到不同 dataset(不同相机)更容易。

**3. 其他 attributes 的 sensible defaults**:
- Scale: $s(i, j) = s_0 \cdot \bar{\mathbf{D}}'(i, j)$ (远的 Gaussian 大,近的小,和 perspective 一致)
- Color: 直接从 downsampled image 拷贝
- Rotation: 单位四元数 $[1, 0, 0, 0]^T$
- Opacity: 0.5(留空间给 decoder 调整)

这就是 "coarse-to-fine" 的思路 —— 初始化一个合理的粗略 3D scene,然后让 decoder 做 refinement。

---

## 模块五:Gaussian Decoder + Composer —— Residual Learning with Constraints

Decoder 输出 deltas $\Delta \mathbf{G}$,包含所有 14 个 attributes 的 refinement。Composer 把 base + delta 组合:

$$\mathbf{G}_{\text{attr}} = \gamma_{\text{attr}} \left( \gamma_{\text{attr}}^{-1}(\mathbf{G}_{0, \text{attr}}) + \eta_{\text{attr}} \Delta \mathbf{G}_{\text{attr}} \right)$$

这里 $\gamma_{\text{attr}}$ 是 attribute-specific activation,$\eta_{\text{attr}}$ 是 scale factor。

具体配置(Supplement A.1):

| Attribute | $\gamma$ | $\eta$ |
|-----------|---------|--------|
| Position ($x/z, y/z$) | identity | $10^{-3}$ |
| Position ($z^{-1}$) | softplus | $10^{-3}$ |
| Color | sigmoid | $10^{-1}$ |
| Rotation | identity | 1 |
| Scale | sigmoid | 1 |
| Alpha | sigmoid | 1 |

几个关键点:

**Position 在 NDC space 操作**:先 $[x, y, z] \to [x/z, y/z, 1/z]$,apply activation 和 delta,再 transform 回 world coords。网络预测的 position delta 是在 image plane space 中,不是 metric 3D space。这对 cross-dataset 泛化更友好 —— 不同 FOV 的 metric scale 完全不同,但 NDC space 是 normalized 的。

**Position delta 的 $\eta = 10^{-3}$ 很小**:base Gaussian 位置基本不动,delta 只是微调。Color 的 $\eta = 10^{-1}$ 大些,允许网络处理 view-dependent 颜色变化。

**Intuition**: 这本质上是 "residual learning with attribute-specific constraints"。每个 attribute 都有 valid range(opacity ∈ [0,1], scale > 0, quaternion 必须是 unit norm),通过精心选 activation 把 delta 加到 normalized space 再 inverse 回 original space,保证 output validity,同时让 learning 稳定。

这种 "在 normalized space 学 delta" 的思路我觉得很 general —— 任何有 manifold constraint 的参数预测都能用这个 pattern。比如预测 camera pose 的 delta 在 Lie algebra space,预测 lighting 的 delta 在 log space,等等。

---

## Loss 设计:八个 Loss Term 各司其职

最终 loss 是:

$$\mathcal{L} = \sum_{d \in \mathcal{D}} \lambda_d \mathcal{L}_d + \sum_{r \in \mathcal{R}} \lambda_r \mathcal{L}_r + \sum_{s \in \mathcal{S}} \lambda_s \mathcal{L}_s$$

其中 $\mathcal{D}$ 是 data loss(color, alpha, depth, perceptual),$\mathcal{R}$ 是 regularizer(tv, grad, delta, splat),$\mathcal{S}$ 是 depth adjustment 的 bottleneck loss。

我挑几个关键的说。

### L1 Color Loss —— 基础但必要

$$\mathcal{L}_{\text{color}} = \sum_{\text{view} \in \{\text{input, novel}\}} \mathbb{E}_{p \sim \Omega} \left[ |\hat{\mathbf{I}}_{\text{view}}(p) - \mathbf{I}_{\text{view}}(p)| \right]$$

input view 和 novel view 都算。input view 的 loss 保证 "重建准确",novel view 的 loss 保证 "warp 准确"。

### Perceptual Loss —— 最关键的一个

$$\mathcal{L}_{\text{percep}} = \sum_{l=1}^{4} \lambda_l^{\text{feat}} \| \phi_l(\hat{\mathbf{I}}_{\text{novel}}) - \phi_l(\mathbf{I}_{\text{novel}}) \|^2 + \lambda_l^{\text{Gram}} \| M_l(\hat{\mathbf{I}}_{\text{novel}}) - M_l(\mathbf{I}_{\text{novel}}) \|^2$$

- $\phi_l$: ResNet-50 第 $l$ 层 feature
- $M_l$: 第 $l$ 层的 Gram matrix
- $\lambda_l^{\text{feat}} = \frac{1}{D_l \cdot H_l \cdot W_l}$: per-element normalization
- $\lambda_l^{\text{Gram}} = \frac{10}{D_l^2}$: Gram weight 放大 10 倍

这个 loss 只在 novel view 上算。

Ablation (Table 8) 显示:加上 perceptual loss,ScanNet++ DISTS 从 0.162 直降到 0.063 —— 这是最大幅度的 single-loss 贡献。

**Gram matrix 的复活**:Gram matrix loss 原本是 style transfer 里的东西,匹配 texture 的 auto-correlation。SHARP 把它用在 view synthesis 上,发现能显著提升 sharpness。Table 10 显示加 Gram loss 把 DISTS 从 0.070 降到 0.064。

**Intuition**: Perceptual loss 鼓励 "看起来对",Gram loss 鼓励 "纹理结构对"。单独的 feature MSE 会让生成内容偏 blurry(因为 feature space 距离小不等于 pixel space sharp)。Gram matrix 匹配 auto-correlation,逼着网络生成正确 texture structure,从而 sharp。

### Alpha Loss —— 防止透明漏点

$$\mathcal{L}_{\text{alpha}} = \sum_{\text{view}} \mathbb{E}_{p \sim \Omega} [\mathcal{L}_{\text{BCE}}(\hat{\mathbf{A}}_{\text{view}}(p), 1)]$$

Target 是 1(fully opaque),惩罚 transparent pixels。这防止网络通过 "半透明" 偷懒 —— 半透明能在 input view 上骗过 color loss,但 novel view 上会露馅。

### Floater Suppressor —— 专治飘浮 Gaussians

$$\mathcal{L}_{\text{grad}} = \mathbb{E}_{i \sim \mathcal{T}} \left[ \mathbf{G}_{\text{alpha}}(i) \cdot \left( 1 - \exp\left( -\frac{1}{\sigma} \max\{0, |\nabla \bar{\mathbf{D}}^{-1}(\pi(\mathbf{G}_0(i)))| - \epsilon\} \right) \right) \right]$$

- $\pi(\mathbf{G}_0(i))$: 第 $i$ 个 base Gaussian 在 image plane 的投影
- $\nabla \bar{\mathbf{D}}^{-1}$: 在投影位置处的 inverse depth gradient
- $\sigma = \epsilon = 10^{-2}$

**Intuition**: 这个 loss 说,"如果一个 Gaussian 落在 depth discontinuity(边缘)附近,它的 opacity 应该低"。这抑制 "floaters" —— 那些飘在 foreground/background 之间的 stray Gaussians,在边缘附近造成 artifacts。

公式逻辑:$\max\{0, |\nabla| - \epsilon\}$ 是 hinge,只在 depth gradient 超过 $\epsilon$ 才开始惩罚。$1 - \exp(-|\nabla|/\sigma)$ 在 $|\nabla|$ 大时趋近 1(惩罚力度大),小时趋近 0(惩罚力度小)。

### Splat Size Regularizer —— 防止 Gaussians 过大或过小

$$\mathcal{L}_{\text{splat}} = \mathbb{E}_{i \sim \mathcal{I}} [\max\{\sigma(\mathbf{G}(i)) - \sigma_{\max}, 0\} + \max\{\sigma_{\min} - \sigma(\mathbf{G}(i)), 0\}]$$

- $\sigma(\cdot)$: projected Gaussian variance(screen space 上的 splat size)
- $\sigma_{\min} = 10^{-1}$, $\sigma_{\max} = 10^{2}$

防止 Gaussians 在 screen space 上过大(导致 blur)或过小(导致 aliasing / hole)。

**有意思的副作用**: Table 9 显示,regularizers 把 ScanNet++ rendering latency 从 22.2ms 降到 5.5ms。因为它们抑制了 degenerate large Gaussians,这些 Gaussians 在 splatting 时计算量很大。这是一个意外的 "side benefit" —— regularization 既改善 quality 又提升 speed。

Reference: [LPIPS (Zhang et al., CVPR 2018)](https://arxiv.org/abs/1801.03924), [DISTS (Ding et al., PAMI 2022)](https://arxiv.org/abs/2004.07728), [Perceptual Losses (Johnson et al., ECCV 2016)](https://arxiv.org/abs/1603.08155)

---

## 工程黑科技:Computation Graph Surgery

Supplement A.4 描述了一个我觉得可以单独发 paper 的工程技巧。

**问题**: Perceptual loss 用 ResNet-50 提 feature,在 1536² 全分辨率上,input view + novel view 两个,backprop 时 computation graph 巨大,A100 40GB 都 OOM。

**朴素的解决方案**:
- BF16:不稳定,3DGS 的 singular values 对精度敏感
- Gradient checkpointing:throughput 严重下降

**SHARP 的方案: Computation Graph Surgery**

1. 实现 "surgery operator" —— forward 时 cache gradients 和 inputs,backward 时 inject cached gradients
2. 在 perceptual loss node 处,**eagerly pre-compute gradients w.r.t. features**(显式 autograd call)
3. **Release 部分 computation graph(ResNet 那部分)**
4. Override node with surgery-operated one

**效果**: Computation graph 大小 agnostic to pixel count 和 view count。可以在 full FP32 下训练,perceptual loss 同时在 reconstruction 和 novel views 上,without compromising throughput。

**Intuition**: 这本质上是 "manual gradient checkpointing" 但更激进 —— 直接切断 ResNet 部分的 graph,只在需要 gradient 的地方手动注入。类似 functional programming 里的 "explicit state passing"。

这个 trick 我觉得在 NeRF/3DGS 训练中有广泛应用前景。任何 "loss function 用大网络但只 backprop 到中间 features" 的场景都能用。比如 GAN-based loss、CLIP-based loss、深度监督 loss,都能用这个 pattern。

---

## 训练策略:Two-Stage Curriculum

### Stage 1: Synthetic Training

- 100K steps,128 A100 GPUs
- In-house synthetic data:2K outdoor + 5K indoor 场景,procedurally augmented
- 每 scene 放 10 个 virtual cameras 围绕一个 object(距离 < 60cm,模拟 headbox)
- V-Ray physically-based rendering,~700K scenes × 11 views ≈ 8M images at 1536² or 2048²
- 加入 thin structures, transparent materials, reflective surfaces,HDR environment maps

这个数据规模和 diversity 是关键。700K scenes 是非常大的数,让我想起 Meta 的 Habitat 或者 NVIDIA 的 Omniverse 数据生成 pipeline。

### Stage 2: Self-Supervised Fine-Tuning (SSFT)

- 60K steps,32 A100 GPUs
- OpenScene + Shutterstock/Getty/Flickr 商业许可图,2.65M images
- **没有 view synthesis GT**

**SSFT 的核心 trick —— Swap input/novel views**:

1. 对每张真实单图,用 Stage 1 model 生成 3D Gaussian
2. 渲染一个 pseudo-novel view
3. **把 pseudo-novel view 当 input,真实图当 novel view**
4. 用这个 pair 算 loss

**Intuition**: 这是一个 self-distillation 风格的 trick。Stage 1 model 能从单图生成 3D,但 pseudo-novel view 可能有 artifacts。把它当 input,要求 model 从这个 "略微 degraded" 的 view 重建出原来的真实图 —— 迫使 model 学会 "fix" 那些 artifacts,并适应真实图像的 distribution(而不是 synthetic data 的 distribution)。

这比 AdaMPI 的 "warp-back" 更优雅 —— warp-back 直接用 single view warp 不能很好处理 disocclusion,而 SHARP 用 3D Gaussian 表示,可以 fill in occluded content。

Ablation (Table 12): SSFT 在 metric 上没显著改善(甚至略升,因为 noise),但 Figure 11 定性看明显 sharper。作者 hypothesize 是因为 synthetic data 缺少 view-dependent effects(reflections, specular highlights),SSFT 让 model 适应真实世界。

**Intuition**: 这种 "用自己生成的伪 GT 反过来训练自己" 的 pattern 我觉得很有意思。类似 GAN 的 discriminator 机制 —— 网络通过 "识别自己生成结果的瑕疵" 来提升。可以推广到很多 self-supervised 场景。

Reference: [AdaMPI (Han et al., SIGGRAPH 2022)](https://arxiv.org/abs/2204.07192), [OpenScene](https://github.com/OpenDriveLab/OpenScene)

---

## Evaluation:为什么用 LPIPS/DISTS 而不是 PSNR/SSIM

Supplement C.1 做了个实验让我印象深刻。对一张图做 1% translation,看 metric 反应:

| Comparison | DISTS | LPIPS | PSNR | SSIM |
|------------|-------|-------|------|------|
| Translated (0.1%) | 0.008 | 0.059 | 21.3 | 0.623 |
| Translated (1.0%) | 0.079 | 0.491 | 11.2 | 0.375 |
| Translated (5.0%) | 0.121 | 0.723 | 8.1 | 0.249 |
| Mean Image | 0.859 | 0.970 | 10.7 | 0.351 |

1% translation 对人眼几乎不可见,但 PSNR 从 21.3 跌到 11.2,接近 mean image 的 10.7。PSNR 把 "几乎没区别" 的图当成 "和 mean image 一样差"。

这对 view synthesis 是致命的 —— 即使 depth 估计只差一点点,几何 warp 就会有 sub-pixel 错位,PSNR/SSIM 直接崩盘。

**Intuition**: PSNR 本质上惩罚 "pixel-wise exact match",但 view synthesis 任务本质上不可能 pixel-wise perfect(depth 不可能完美)。我们关心的是 "perceptual quality",LPIPS/DISTS 作为 perceptual metrics,对小 translation 鲁棒得多。这个 metric 选择直接反映 task 的本质 —— 我们要的是 "看着像",不是 "逐像素对齐"。

### 主实验结果

Table 1:SHARP 在所有 6 个 dataset 上 DISTS 和 LPIPS 都最好。对比 Gen3C(最强 diffusion baseline):

| Dataset | SHARP DISTS | Gen3C DISTS | 提升 |
|---------|-------------|-------------|------|
| Middlebury | 0.097 | 0.164 | -41% |
| ScanNet++ | 0.071 | 0.090 | -21% |
| WildRGBD | 0.069 | 0.106 | -35% |
| ETH3D | 0.258 | 0.408 | -37% |

Speed 对比(Table 6):
- SHARP: 0.91s inference,0.01s render (100 FPS)
- Gen3C: 830s inference (~14 minutes)
- ViewCrafter: 120s inference (2 minutes)

**SHARP 比 Gen3C 快 910 倍,且 fidelity 更高**。这个结果很 striking。

### With Privileged Depth (Table 7)

给所有方法 GT depth 做 scale alignment,SHARP 依然最好。这排除了 "SHARP 只是赢在 depth estimation" 这个 hypothesis —— 即使给其他方法 GT depth,SHARP 仍然更强。说明 SHARP 的优势来自整个 pipeline。

### Motion Range Analysis (Figure 7)

把 camera baseline 分 bin 看 DISTS:
- < 0.5m: SHARP 最好
- 0.5m - 3m: SHARP 仍 top-2
- > 3m: Gen3C 开始超越(SHARP 设计范围外)

这印证了 SHARP 的 design philosophy —— 牺牲大 motion range 换取 nearby view 的极致 fidelity。

### Number of Gaussians Ablation (Table 14)

| # Gaussians | ScanNet++ DISTS |
|-------------|-----------------|
| 2×196×196 (~77K) | 0.110 |
| 2×392×392 (~307K) | 0.077 |
| 2×784×784 (~1.2M) | 0.064 |

Monotonic improvement with more Gaussians,说明这个 regression framework 还有继续 scale 的空间。未来可能 4M、8M Gaussians,quality 还能涨。

Reference: [ScanNet++ (Yeshwanth et al., ICCV 2023)](https://github.com/scannetpp/scannetpp), [Gen3C (Ren et al., CVPR 2025)](https://research.nvidia.com/labs/toronto-ai/gen3c/)

---

## 失败案例:Depth Model 的 Long Tail

Figure 8 展示了 3 类典型失败:

**1. Macro photo (strong depth-of-field)**: 蜜蜂的 depth 被错判为花后面,导致翅膀脱离、尾巴扭曲。DoF 误导 depth model 的经典 case。背景虚化被网络理解为 "远处",主体被理解为 "更远"。

**2. Starry night sky**: 丰富 texture 让网络把 sky 当成 curvy surface(因为 texture 太多,网络以为是近处物体表面),新视角下严重扭曲。

**3. 复杂水面 reflection**: 网络把水中倒影当成远处山脉(因为倒影看起来像远处的山),水面 broken。

**根因**: 都是 depth prediction 的 long-tail failure。即使 unfreezing backbone 也救不回来 —— base initialization 已经错了,Gaussian decoder 的 delta 太小($\eta = 10^{-3}$)改不动。

**Intuition**: Regression 路线的 Achilles' heel 就是 "garbage in, garbage out"。如果 depth backbone 在某些场景彻底失败,下游 refinement 无法挽救。Diffusion 方法在这些场景反而有优势 —— 它的 prior 能 "hallucinate" 出合理内容,即使 depth 错了。

这其实指向了一个 unified solution 的方向:**SHARP 处理 nearby views, diffusion 处理 faraway + 失败 case**。Paper 的 Conclusion 也提到这个方向。

---

## 我的 Takeaways

读完整篇,我有几个层面的思考:

### (1) Problem Formulation 决定方法选择

SHARP 的成功首先来自精确定义问题:"nearby views, < 1 sec, metric scale, headbox motion"。这个 scope 让 regression 路线变得可行 —— 不需要 hallucinate 远处内容,不需要 walk around。Diffusion 方法在 "什么都能做" 的通用性上强,但在特定 niche 上,regression 用 1/1000 的算力就能超越。

这让我想到特斯拉 FSD 的 design philosophy —— 不是做 "general AI driving",而是做 "specific routes 的 robust driving"。问题定义的精度决定了方法的上下限。

### (2) Pretrained Backbone 是金矿

Depth Pro 的 backbone 提供了强大的 visual features,SHARP 直接复用并部分 unfreeze。这印证了 "foundation model + task-specific head" 范式在 3D vision 的有效性。

我觉得未来 3D vision 会有类似 BERT/ViT 在 NLP/CV 的地位 —— 一个 general geometry/depth backbone,各种下游任务(head reconstruction, view synthesis, SLAM, ...)都在它上面加 task-specific head。Depth Pro 可能就是早期 instance。

### (3) Training-only Module 的 Pattern

Depth adjustment module 是 training-only 的,inference 时换成 identity。这种 pattern 让 model 训练时享受 privileged information 的便利,inference 时干净,通过 bottleneck 强制 self-sufficient。

这个 pattern 我觉得可以推广到很多场景:
- Medical imaging:训练时有 expert annotation,inference 时只有 raw image
- Robotics:训练时有 sim privileged state,inference 时只有 sensors
- RL:训练时有 demo,inference 时只有 environment

本质上这是一种 "learned curriculum with decaying privileged info"。

### (4) Loss Engineering 依然重要

在大家都关注 architecture 的当下,SHARP 通过精心设计 8 个 loss term,每个都有 specific purpose,达到了 SOTA。特别是 Gram matrix loss 的复活(原本 style transfer)用在 view synthesis 上提升 sharpness,这种 cross-task insight transfer 值得学习。

我之前在 Tesla 讲过 "AI is just constraint satisfaction"。Loss function 就是 constraints 的表达。好的 loss design 比好的 architecture 更难,因为它需要对 task 的深度理解。

### (5) Computation Graph Surgery 的启发

这个工程技巧我觉得有独立价值。任何 "loss function 用大网络但只 backprop 到中间 features" 的场景都能用。GAN loss、CLIP loss、深度监督 loss,都能用这个 pattern 避免 OOM。

### (6) "Amortizable 3D Representation" 的概念

SHARP 展示了 3DGS 的一个新用法 —— feed-forward regression 的输出。这意味着 3DGS 可以作为 "amortizable 3D representation":一旦 synthesize 完成,可以 real-time render from arbitrary nearby views,不需要重新 inference。

这和 diffusion-based 方法每次推理都要从头 denoise 是本质区别。对于 AR/VR 这种 "一次生成,反复使用" 的场景,amortizable representation 有巨大优势。

### (7) Synthetic Data + SSFT 的组合拳

Stage 1 用 700K synthetic scenes 学 fundamental 3D reconstruction,Stage 2 用 2.65M real images 做 self-supervised adaptation。这个 pattern 我觉得会成为 standard:
- Synthetic data 提供 perfect GT 和 diversity
- Real data 提供 distribution adaptation
- Self-supervised bridge 连接两者

类似 NVIDIA 的 Omniverse + real data,Meta 的 Habitat + real data,pattern 都类似。

### (8) 未来的 Unified Solution

Paper Conclusion 提到一个有趣方向:把 diffusion 的 "synthesize faraway content" 能力 distill 到 feed-forward model,实现 unified solution。

我觉得这个方向有戏。Diffusion distillation(类似 DMD, Consistency Models)已经能把 diffusion 压到 few-step 甚至 one-step。如果能 distill 出 "nearby views 用 regression, faraway views 用 diffusion prior" 的 unified model,就能兼顾 quality 和 speed。

Reference: [DMD: Distribution Matching Distillation (Yin et al., CVPR 2024)](https://arxiv.org/abs/2311.18828), [Consistency Models (Song et al., 2023)](https://arxiv.org/abs/2303.01469)

---

## 可能的扩展与联想

读完 paper,我脑子里冒出几个方向:

**1. Video Input Extension**: 当前 single image。如果有 video input,可以利用 temporal consistency constraint,depth estimation 会更准,quality 可能进一步改善。类似 bundle adjustment 但 amortized。

**2. Dynamic Scenes**: 当前 static scene assumption。如何 handle 动态内容(人走动、风吹树叶)?可能需要 4D Gaussians(加 time dimension)或者 per-frame regression + temporal smoothing。

**3. Uncertainty-aware Rendering**: Depth adjustment 训练时学到 depth uncertainty,但 inference 时丢弃。能否保留 uncertainty 用于 adaptive rendering —— uncertain regions 用更多 Gaussians,某些用 fewer?

**4. Multi-resolution Gaussians**: 当前固定 2 层。能否做 adaptive layer count per pixel?或者 multi-resolution Gaussians(类似 mip-splatting)处理 aliasing?

**5. Joint Depth + View Synthesis**: 当前 depth loss 和 view synthesis loss 是 weighted sum。能否做 multi-task learning with uncertainty weighting(Kendall et al.)?

**6. CLIP-guided View Synthesis**: 把 perceptual loss 换成 CLIP-based loss,可能获得更好的 semantic consistency。Computation graph surgery trick 正好能用上。

**7. Generative Refinement as Post-processing**: SHARP 出 regression 结果,再用 diffusion model 做 "细节增强" —— 类似 Stable Diffusion 的 img2img 但专门针对 view synthesis artifacts。这可能解决 SHARP 在 faraway views 和 depth failure cases 的问题。

**8. Cross-modal Extension**: 当前 RGB → 3D。能否做 RGBD → 3D,或者 RGB + text prompt → 3D?Text prompt 能 disambiguate depth failures(比如 "这是 macro photo,蜜蜂在前面")。

**9. ADAS Application**: SHARP 的 metric scale + fast inference + single image input 很适合 ADAS 场景。单帧摄像头图像 → 3D scene representation,用于 motion planning。类似 Tesla FSD 的 vision pipeline。

**10. Memory Browsing App**: Apple 的 motivation 是 "revisiting memories"。如果 iPhone 相册集成 SHARP,用户浏览照片时能 3D 化,晃手机看不同角度。这个 product feature 其实很 compelling —— 类似 Apple Vision Pro 的 Spatial Photos 但 from 任意普通照片。

---

## 最后

SHARP 这篇 paper 让我想到一个更大的 trend:**AI 正在从 "generate" 走向 "reconstruct"**。

Diffusion model 把 "generate from nothing" 推到了极致。但实际应用中,我们经常有 input signal(一张图、一段 video、一组 sensor data),需要的是 "从 signal 重建出 structured representation"。这类问题里,regression 路线在特定 niche 依然有优势 —— 快、稳定、amortizable。

SHARP 在 "single image to 3D Gaussian" 这个具体问题上,把 regression 路线推到了超越 diffusion 的 fidelity。它没有 fundamentally new 的 ML idea,但通过 problem formulation + engineering details + scale 的三重奏,达成了 SOTA。

这种 "工程主义胜利" 的 pattern 在当前 deep learning 时代非常常见。我觉得值得仔细 study —— 因为下一个 breakthrough 可能不是来自 fundamentally new idea,而是来自对现有 ideas 的极致 engineering。

Reference: [SHARP Project Page](https://apple.github.io/ml-sharp), [SHARP GitHub](https://github.com/apple/ml-sharp), [Depth Pro](https://arxiv.org/abs/2410.19115), [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

---

# SHARP: Single-image High-Accuracy Real-time Parallax 深度解析

## 1. 问题设定:为什么需要 SHARP

这篇 paper 来自 Apple,作者阵容包括 Vladlen Koltun, Lars Mescheder 等。核心想解决的问题非常具体:**从一张照片,在 1 秒内,生成一个 metric scale 的 3D Gaussian 表示,支持 nearby views 的实时高分辨率渲染**。

这个设定有几个关键约束值得拆解:

**(a) Single image input**: 不依赖 multi-view,不需要 per-scene optimization。NeRF 类方法虽然 photorealism 高,但需要多张图 + 长 optimization,不适合交互式浏览个人照片库。

**(b) Nearby views only**: 不追求 "walking around" 物体,只支持 AR/VR headset 的 headbox (natural posture shifts,约 0.5m 范围)。这是一个关键的 scope 选择 —— 放弃了大范围 motion,换取了 sharpness 和 speed。

**(c) Metric scale**: 输出有绝对尺度,可以和物理 headset 准确耦合。这点和很多 affine-invariant depth 方法不同。

**(d) < 1 second synthesis + 100 FPS rendering**: 这是和 diffusion-based 方法 (ViewCrafter, Gen3C, SVC) 的本质区别 —— 那些方法动辄几分钟一张图,而 SHARP 是 feed-forward 单次推理。

**我的 intuition**: 这篇工作本质上是把 "single image to 3D" 这个问题从 "生成式" 路线 (diffusion prior) 拉回到了 "回归式" 路线 (feed-forward regression),并通过一系列精心设计的工程选择,在 nearby view 这个 niche 上把 regression 做到了超越 diffusion 的 fidelity。Figure 1 那张图很关键 —— 横轴 log scale 的时间,纵轴 LPIPS,SHARP 在左下角 (快且好),其他方法要么慢要么差。

Reference: [Apple ML SHARP](https://github.com/apple/ml-sharp), [Depth Pro paper](https://arxiv.org/abs/2410.19115)

---

## 2. 整体架构:四大模块 + 两个 assembly 操作

输入是 single RGB image $\mathbf{I} \in \mathbb{R}^{3 \times H \times W}$,其中 $H = W = 1536$ (Depth Pro 的标准输入尺寸)。输出是 $K=14$ 维属性的 3D Gaussians $\mathbf{G} \in \mathbb{R}^{K \times N}$,其中 $N \approx 1.2M$ (具体是 $2 \times 768 \times 768 \approx 1.18M$)。

14 维属性拆解:
- **Position**: 3 (xyz)
- **Scale**: 3 (各向异性 scale)
- **Rotation**: 4 (quaternion)
- **Color**: 3 (RGB,不用 spherical harmonics 因为 SH coefficients 随 order 二次增长会爆 output size)
- **Opacity**: 1

**关键设计选择: 2-layer depth**。作者复制了 Depth Pro decoder 的最后一层 conv,输出 $\hat{\mathbf{D}} \in \mathbb{R}^{2 \times H \times W}$。第一层是 primary visible surfaces,第二层表示 occluded regions 和 view-dependent effects。这直接对应输出 Gaussians 的 "2" 这个维度 —— 每个 pixel 位置产生 2 个 Gaussians,一个在前层,一个在次层,处理 disocclusion 时的内容补全。

**架构流程 (Figure 3 解析)**:

```
Input Image (3×1536×1536)
      │
      ▼
┌─────────────────────┐
│ Depth Pro Encoder   │  ← φ_enc, 326M params (low-res ViT, unfrozen)
│ (ViT low-res + ViT  │    + 326M params (patch ViT, frozen)
│  patch)             │
└─────────────────────┘
      │ 4 feature maps (f_1, f_2, f_3, f_4)
      ├──────────────────────┐
      ▼                      ▼
┌─────────────┐      ┌─────────────────┐
│ Depth       │      │ Gaussian        │
│ Decoder     │      │ Decoder         │  ← 7.8M params, DPT-based + custom head
│ (DPT-based,│      │ (takes f_i, I,  │
│  ~20M)      │      │  D_hat)         │
└─────────────┘      └─────────────────┘
      │                      │
      ▼ D_hat (2×H×W)        │ ΔG (14×2×384×384)
      │                      │
      ▼                      │
┌─────────────────┐         │
│ Depth Adjustment│         │
│ (2M U-Net,      │         │
│  training only) │         │
└─────────────────┘         │
      │                     │
      ▼ D_bar (2×H×W)      │
      │                     │
      └──────┬──────────────┘
             ▼
   ┌───────────────────┐
   │ Gaussian          │  ← differentiable,初始化 base Gaussians G_0
   │ Initializer       │
   └───────────────────┘
             │
             ▼  G_0 (14×2×384×384)
   ┌───────────────────┐
   │ Gaussian          │  ← compose G_0 + ΔG via attribute-specific activation
   │ Composer          │
   └───────────────────┘
             │
             ▼  G (14×2×384×384)
   ┌───────────────────┐
   │ Gaussian          │  ← R(G, P) → Î (rendered image)
   │ Renderer          │
   └───────────────────┘
```

总参数 702M,可训练 340M。

**Intuition**:这个架构本质上是 "Depth Pro backbone + 两个 DPT-style head (depth + Gaussian refinement) + 一个 U-Net (depth adjustment, training only)"。复用 Depth Pro 的预训练特征提取是非常聪明的一步 —— 它已经学到了 metric depth 所需的丰富 visual features,直接继承就避免了从零学 depth 的痛苦。

---

## 3. 关键技术一:Unfreezing Depth Pro Backbone

这是一个容易被忽略但很重要的细节。Depth Pro 本身是个强大的 metric monodepth 模型,但直接 frozen 使用会有问题:

> "depth is ill-defined and using a frozen monodepth model can degrade view synthesis fidelity, particularly for transparent or reflective surfaces"

作者只冻结了 **patch encoder** 和 **normalization layers**,而 **unfreeze 了 low-resolution image encoder + depth decoder**。这样做的目的:让 depth prediction 可以通过 view synthesis loss 的 backprop 适配下游目标,而不是单纯优化 depth accuracy。

Table 13 的 ablation 显示这个选择在 ScanNet++ 上把 DISTS 从 0.084 降到 0.064,LPIPS 从 0.158 降到 0.147。Figure 12 定性显示 unfreezing 解决了 boundary artifacts 和 reflection 问题。

**Intuition**: Depth Pro 训练时优化的是 depth MAE,但 view synthesis 关心的是 "depth 在新视角下能否正确 warp 出 plausible 图像"。这两个目标不完全对齐 —— 比如 transparent surface 的 depth 本来就 ill-defined,纯 depth supervision 会强制预测一个 "平均" 的 depth,而 view synthesis 更希望网络用其他线索 (context, geometry prior) 推理出合理的 layer 分配。Unfreezing 允许网络重新平衡这两个目标。

---

## 4. 关键技术二:Depth Adjustment Module (C-VAE inspired)

这是这篇 paper 我觉得最 elegant 的设计之一。

**问题**: Monocular depth estimation 有 inherent ambiguity。网络倾向于预测 "mean scale" 的 depth —— 即在多个 plausible depth 配置的均值附近。这对 depth 任务可能 OK,但对 view synthesis 会产生 artifacts (warp 时出现 "smearing" 或 "banding")。

**传统 C-VAE 思路**: 训一个 posterior $q(z|x, y_{GT})$ 输入 ground truth depth 产 latent $z$,加 KL divergence bottleneck,强制 $z$ 只编码 "消解 ambiguity 所需的最小信息"。Inference 时从 prior $p(z|x)$ 采样。

**SHARP 的简化版**:
- 不用 latent vector $z$,而用 **scale map** $\mathbf{S} \in \mathbb{R}^{H \times W}$ (per-pixel scale factor)
- 不用 KL divergence,而用 **task-specific regularizer** (MAE + multiscale TV)
- 训练时 input 是 $\hat{\mathbf{D}}^{-1}$ (predicted inverse depth) 和 $\mathbf{D}^{-1}$ (GT inverse depth),output 是 scale map $\mathbf{S}$
- Adjusted depth: $\bar{\mathbf{D}} = \mathbf{S}(\hat{\mathbf{D}}, \mathbf{D}) \odot \hat{\mathbf{D}}$
- **Inference 时直接用 identity 替换 depth adjustment module**

这里有个关键 insight: 这个模块是 **training-only** 的。它的作用是给训练提供一个 "作弊通道" —— 当 depth estimator 在 ambiguous 区域预测错时,这个 scale map 可以矫正它,让 view synthesis loss 能正确 backprop。但 inference 时这个信息不可得,所以网络必须学会在没有这个矫正的情况下也能输出正确的 depth。

这相当于一种 **information bottleneck training**: 允许训练时用 privileged information,但通过 regularizer 强制这个信息被压缩到最小,使得网络主体学到 "不依赖 privileged 信息也能 work" 的能力。

**Loss 设计**:

$$\mathcal{L}_{\text{scale}} = \mathbb{E}_{p \sim \Omega} [| \mathbf{S}(p) |]$$

$$\mathcal{L}_{\nabla \text{scale}} = \sum_{k=1}^{6} \mathbb{E}_{p \sim \Omega_{\downarrow k}} [| \nabla \mathbf{S}_{\downarrow k}(p) |]$$

其中 $\mathbf{S}_{\downarrow k}$ 是 downsample $2^k$ 倍的 scale map,$\Omega_{\downarrow k}$ 是对应 downsampled image domain。这两个 loss 分别惩罚 scale map 的幅度和多尺度平滑性,weight 是 $\lambda_{\text{scale}} = 0.1$, $\lambda_{\nabla \text{scale}} = 5.0$ (TV weight 比 L1 大 50 倍,说明平滑性更重要)。

**Ablation (Table 11)**: ScanNet++ 上 DISTS 从 0.077 → 0.064 (17% improvement),Figure 10 显示 sharpened details。

**Intuition**: 这其实是 "learned test-time augmentation" 的反向用法。传统 test-time augmentation 是 inference 时做手脚,这里是 training 时做手脚,inference 时干净。它的本质是让 depth estimator 在训练时看到 "如果 depth 对了 view synthesis 会怎样" 的信号,从而学到对下游任务更友好的 depth representation。

---

## 5. 关键技术三:Gaussian Initializer (Normalized Space)

这个模块值得仔细看。给定 $\bar{\mathbf{D}}$ 和 $\mathbf{I}$,它初始化 base Gaussians $\mathbf{G}_0 \in \mathbb{R}^{14 \times 2 \times 384 \times 384}$。

**Step 1: Subsample**
- Input image: average pool × 2 → $\mathbf{I}' \in \mathbb{R}^{3 \times 768 \times 768}$
- Adjusted depth: min pool × 2 → $\bar{\mathbf{D}}' \in \mathbb{R}^{2 \times 768 \times 768}$

这里用 **min-pooling 而不是 average pooling** for depth,是因为 depth 的 foreground object 应该取近的 (min depth),average 会导致 foreground "晕开" 到 background。

**Step 2: Unproject**
$$\mu(i, j) = [i \cdot \bar{\mathbf{D}}'(i, j), j \cdot \bar{\mathbf{D}}'(i, j), \bar{\mathbf{D}}'(i, j)]^T$$

这里 **deliberately 不用 intrinsics matrix**。这是非常关键的设计 —— 让网络在 **normalized space** 中预测 Gaussian attributes,不需要适配不同 FOV。Projection 时通过 $\mathbf{P} = \mathbf{K}_{\text{tgt}} \mathbf{E}_{\text{tgt}} \mathbf{E}_{\text{src}}^{-1} \mathbf{K}_{\text{src}}^{-1}$ 把 normalized space 变换吸收进 projection matrix。

**Intuition**: 这相当于 "uncalibrated" 表示。对于 view synthesis 任务,我们不关心 absolute metric coordinates,只关心 "从 source view 到 target view 的相对变换"。把 intrinsics 吸收进 projection matrix 后,网络只需要在 normalized space 学一次,适用于任何 FOV 的图像。这个设计让方法 zero-shot 泛化到不同 dataset (不同相机) 更容易。

**Step 3: 其他 attributes 初始化**
- Scale: $s(i, j) = s_0 \cdot \bar{\mathbf{D}}'(i, j)$,scale proportional to depth (远的 Gaussian 大,近的小,和 perspective 一致)
- Color: 直接从 downsampled image 拷贝 $c(i, j) = \mathbf{I}'(i, j)$
- Rotation: 单位四元数 $[1, 0, 0, 0]^T$ (无旋转)
- Opacity: 固定 0.5 (留空间给 decoder 调整)

**Intuition**: 初始化用 "sensible defaults" —— 直接从 image color 和 depth 推出一个粗略的 3D scene,然后让 Gaussian decoder 做 refinement。这种 "coarse-to-fine" 策略避免了 decoder 从零学起。

---

## 6. 关键技术四:Gaussian Decoder 与 Attribute-Specific Composer

Decoder 输出 $\Delta \mathbf{G} \in \mathbb{R}^{14 \times 2 \times 384 \times 384}$,包含所有 attributes 的 deltas:
- $\Delta \mathbf{G}_{\text{pos}}$: 3 channels (xyz delta)
- $\Delta \mathbf{G}_{\text{scale}}$: 3 channels
- $\Delta \mathbf{G}_{\text{rot}}$: 4 channels (quaternion delta)
- $\Delta \mathbf{G}_{\text{color}}$: 3 channels
- $\Delta \mathbf{G}_{\text{alpha}}$: 1 channel

**Composer 公式 (Eq. 3.2)**:
$$\mathbf{G}_{\text{attr}} = \gamma_{\text{attr}} \left( \gamma_{\text{attr}}^{-1}(\mathbf{G}_{0, \text{attr}}) + \eta_{\text{attr}} \Delta \mathbf{G}_{\text{attr}} \right)$$

变量解释:
- $\gamma_{\text{attr}}$: attribute-specific activation function
- $\gamma_{\text{attr}}^{-1}$: 对应的 inverse activation (在 normalized space 中加 delta)
- $\eta_{\text{attr}}$: scale factor 控制 delta 的幅度
- $\mathbf{G}_{0, \text{attr}}$: 初始化的 base Gaussian attribute
- $\Delta \mathbf{G}_{\text{attr}}$: decoder 预测的 delta

**激活函数选择 (Supplement A.1)**:

| Attribute | $\gamma$ | $\eta$ |
|-----------|---------|--------|
| Position ($x/z, y/z$) | identity | $10^{-3}$ |
| Position ($z^{-1}$) | softplus | $10^{-3}$ |
| Color | sigmoid | $10^{-1}$ |
| Rotation | identity | 1 |
| Scale | sigmoid | 1 |
| Alpha | sigmoid | 1 |

**关键设计**: Position 在 NDC (Normalized Device Coordinates) space 操作 —— 先 $[x, y, z] \to [x/z, y/z, 1/z]$ 再 apply activation 和 delta,然后 transform 回 world coords。这样网络预测的 position delta 是在 image plane space 中,而不是 metric 3D space,这对 cross-dataset 泛化更友好 (因为不同 FOV 的 metric scale 完全不同)。

Position 的 $\eta = 10^{-3}$ 很小,意味着 base Gaussian 的位置基本不动,delta 是微调。Color 的 $\eta = 10^{-1}$ 大一些,允许网络在初始化的 image color 基础上做较大修改 (处理 view-dependent 颜色变化)。

**Intuition**: 这个 composer 的设计本质上是 "residual learning with attribute-specific constraints"。每个 attribute 都有自己的 valid range (e.g. opacity ∈ [0,1], scale > 0, quaternion 必须是 unit norm),通过精心选 activation 把 delta 加到 normalized space 再 inverse 回 original space,保证了 output 的 validity,同时让 learning 稳定。

---

## 7. Loss 设计:细致的工程权衡

训练 loss 分三大类:**Rendering losses**, **Depth losses**, **Regularizers**,加上 depth adjustment 的 bottleneck loss。最终组合:

$$\mathcal{L} = \sum_{d \in \mathcal{D}} \lambda_d \mathcal{L}_d + \sum_{r \in \mathcal{R}} \lambda_r \mathcal{L}_r + \sum_{s \in \mathcal{S}} \lambda_s \mathcal{L}_s$$

其中 $\mathcal{D} = \{\text{color}, \text{alpha}, \text{depth}, \text{percep}\}$, $\mathcal{R} = \{\text{tv}, \text{grad}, \text{delta}, \text{splat}\}$, $\mathcal{S} = \{\text{scale}, \nabla\text{scale}\}$

### 7.1 Rendering losses

**L1 color loss (Eq. 3.3)**:
$$\mathcal{L}_{\text{color}} = \sum_{\text{view} \in \{\text{input, novel}\}} \mathbb{E}_{p \sim \Omega} \left[ |\hat{\mathbf{I}}_{\text{view}}(p) - \mathbf{I}_{\text{view}}(p)| \right]$$

- $p$: pixel
- $\Omega$: all pixels
- 同时在 input view 和 novel view 上计算 (input view 的 loss 约束 "重建准确",novel view 的 loss 约束 "warp 准确")

**Perceptual loss (Eq. 3.4)** —— 这是最重要的 loss,只在 novel view 上:
$$\mathcal{L}_{\text{percep}} = \sum_{l=1}^{4} \lambda_l^{\text{feat}} \| \phi_l(\hat{\mathbf{I}}_{\text{novel}}) - \phi_l(\mathbf{I}_{\text{novel}}) \|^2 + \lambda_l^{\text{Gram}} \| M_l(\hat{\mathbf{I}}_{\text{novel}}) - M_l(\mathbf{I}_{\text{novel}}) \|^2$$

- $\phi_l$: ResNet-50 第 $l$ 层 feature map
- $M_l$: 第 $l$ 层的 Gram matrix ($M_l = \phi_l^T \phi_l$,auto-correlation of features)
- $\lambda_l^{\text{feat}} = \frac{1}{D_l \cdot H_l \cdot W_l}$: per-element normalization,$D_l, H_l, W_l$ 是 feature map shape
- $\lambda_l^{\text{Gram}} = \frac{10}{D_l^2}$: Gram matrix weight (相对放大了 10 倍)

**Alpha loss (Eq. 3.5)**: BCE loss 惩罚 rendered alpha,鼓励 opaque rendering,避免 spurious transparent pixels:
$$\mathcal{L}_{\text{alpha}} = \sum_{\text{view}} \mathbb{E}_{p \sim \Omega} [\mathcal{L}_{\text{BCE}}(\hat{\mathbf{A}}_{\text{view}}(p), 1)]$$

- $\hat{\mathbf{A}}_{\text{view}}$: rendered alpha image
- Target 是 1 (fully opaque),惩罚透明像素

### 7.2 Depth loss

$$\mathcal{L}_{\text{depth}} = \mathbb{E}_{p \sim \Omega} [| \bar{\mathbf{D}}_{(1)}^{-1}(p) - \mathbf{D}^{-1}(p) |]$$

- $\bar{\mathbf{D}}_{(1)}$: first predicted depth layer (visible surface)
- $\mathbf{D}$: ground truth depth
- 只在 input view,只在 first layer (第二层是 occluded content,没有 GT)
- 用 inverse depth 而不是 depth,因为 inverse depth 在 near camera 处分辨率更高,更符合视觉重要性

### 7.3 Regularizers (artifact 抑制)

**TV on 2nd depth layer (Eq. 3.7)**:
$$\mathcal{L}_{\text{tv}} = \mathbb{E}_{p \sim \Omega} [|\nabla_x \bar{\mathbf{D}}_{(2)}^{-1}(p)| + |\nabla_y \bar{\mathbf{D}}_{(2)}^{-1}(p)|]$$

促进第二层 depth 的平滑性 (occluded content 的 depth 不应该高频波动)。

**Floater suppressor (Eq. 3.8)** —— 这个公式比较 intricate:
$$\mathcal{L}_{\text{grad}} = \mathbb{E}_{i \sim \mathcal{T}} \left[ \mathbf{G}_{\text{alpha}}(i) \cdot \left( 1 - \exp\left( -\frac{1}{\sigma} \max\{0, |\nabla \bar{\mathbf{D}}^{-1}(\pi(\mathbf{G}_0(i)))| - \epsilon\} \right) \right) \right]$$

- $i$: Gaussian index
- $\mathcal{T}$: index set for Gaussians
- $\mathbf{G}_{\text{alpha}}(i)$: 第 $i$ 个 Gaussian 的 opacity
- $\pi(\mathbf{G}_0(i))$: 第 $i$ 个 base Gaussian 的 2D image plane projection
- $\nabla \bar{\mathbf{D}}^{-1}$: 在 projected 位置处的 inverse depth gradient
- $\sigma = \epsilon = 10^{-2}$

**Intuition**: 这个 loss 惩罚 "在高 depth gradient 区域有高 opacity 的 Gaussians"。也就是说,如果一个 Gaussian 落在 depth discontinuity (边缘) 附近,它的 opacity 应该低。这抑制了 "floaters" —— 那些飘在 foreground/background 之间的 stray Gaussians,它们在边缘附近造成 artifacts。

公式形式上,$\max\{0, |\nabla| - \epsilon\}$ 是一个 hinge —— 只有当 depth gradient 超过 $\epsilon$ 才开始惩罚。$\exp(-|\nabla|/\sigma)$ 在 $|\nabla|$ 大时趋近 0,所以 $1 - \exp$ 趋近 1,惩罚力度大;在 $|\nabla|$ 小时趋近 0,惩罚力度小。

**Delta magnitude constraint (Eq. 3.9)**:
$$\mathcal{L}_{\text{delta}} = \mathbb{E}_{i \sim \mathcal{T}} [\max\{|\Delta \mathbf{G}_x(i)| - \delta, 0\} + \max\{|\Delta \mathbf{G}_y(i)| - \delta, 0\}]$$

- $\delta = 400.0$
- 只约束 $x, y$ 方向 (不约束 $z$),防止 Gaussian 远离 base 位置
- 这是 hinge loss,只在 delta 超过 threshold 时才惩罚

**Splat size regularizer (Eq. 3.10)**:
$$\mathcal{L}_{\text{splat}} = \mathbb{E}_{i \sim \mathcal{I}} [\max\{\sigma(\mathbf{G}(i)) - \sigma_{\max}, 0\} + \max\{\sigma_{\min} - \sigma(\mathbf{G}(i)), 0\}]$$

- $\sigma(\cdot)$: projected Gaussian variance (splat size in screen space)
- $\sigma_{\min} = 10^{-1}$, $\sigma_{\max} = 10^{2}$
- 防止 Gaussians 在 screen space 上过大 (导致 blur) 或过小 (导致 aliasing / hole)

### 7.4 Ablation 关键发现

Table 8 显示各 loss 的贡献 (ScanNet++ DISTS):
- 只有 color + alpha: 0.229
- + depth: 0.162 (大幅改善)
- + perceptual: 0.063 (再大幅改善)
- + regularizers: 0.064 (metric 上几乎没变化,但 qualitative 改善)

**Perceptual loss 是最大赢家**。Table 10 单独 ablation Gram matrix 显示:加上 Gram matrix 把 ScanNet++ DISTS 从 0.070 降到 0.064,LPIPS 从 0.153 降到 0.147。

**有意思的发现**: Table 9 显示 regularizers 把 rendering latency 从 22.2ms 降到 5.5ms (ScanNet++),**因为它们防止了 degenerate large Gaussians** —— 这些 Gaussians 在 splatting 时计算量很大,被 regularizer 抑制后 rendering 速度大幅提升。这是一个 "side benefit" of regularization。

**Intuition**: Perceptual loss 是 view synthesis 的灵魂 —— 它鼓励 network 生成 "看起来对" 的图,即使 pixel-wise 不完全对齐 (因为 depth 不可能完美)。Gram matrix loss 进一步引入 "texture auto-correlation" 匹配,让生成的内容有正确的 texture structure,这对 sharpness 至关重要。Regularizers 不直接改善 metric,但它们抑制了 pathological Gaussians,让 rendering 更快更稳定。

---

## 8. Perceptual Loss 的工程实现:Computation Graph Surgery

Supplement A.4 描述了一个非常聪明的工程技巧,我觉得值得单独讲。

**问题**: Perceptual loss 用 ResNet-50 提 feature,在 1536×1536 全分辨率上计算,在 input view 和 novel view 同时计算,backprop 时 computation graph 巨大,A100 40GB 都 OOM。

**朴素的解决方案**:
- BF16 精度:会不稳定,因为 3DGS 的 singular values 对精度敏感
- Gradient checkpointing:training throughput 严重下降

**SHARP 的方案: Computation Graph Surgery**

具体做法:
1. 实现 "surgery operator" —— forward pass 时接受并 cache gradients along with inputs
2. Backward pass 时 inject cached gradients
3. 在 perceptual loss node 处,**eagerly pre-compute gradients w.r.t. features** via explicit autograd call
4. Release partial computation graph involving ResNet
5. Override the node with surgery-operated one

**效果**: 避免了 computation graph accumulation,graph 大小 agnostic to pixel count 和 view count。可以在 full FP32 下训练,perceptual loss 同时在 reconstruction 和 novel views 上,without compromising throughput。

**Intuition**: 这本质上是 "manual gradient checkpointing" 但更激进 —— 直接切断 ResNet 部分的 graph,只在需要 gradient 的地方手动注入。类似 functional programming 里的 "explicit state passing"。这是一个非常实用的 trick,可以推广到任何 "loss function 用大网络但只 backprop 到中间 features" 的场景。

---

## 9. 训练策略:Two-Stage Curriculum

### Stage 1: Synthetic Training

- 100K steps,128 A100 GPUs
- 用 in-house synthetic data:2K outdoor + 5K indoor 场景,procedurally augmented
- 每个 scene 放置 10 个 virtual cameras 围绕一个 object of interest (距离 < 60cm,模拟 multi-view ring)
- 用 V-Ray physically-based rendering,~700K unique scenes × 11 views ≈ 8M images at 1536² or 2048²
- 加入 thin structures, transparent materials, reflective surfaces
- HDR environment maps 提供全局光照

### Stage 2: Self-Supervised Fine-Tuning (SSFT)

- 60K steps,32 A100 GPUs
- 用 OpenScene + Shutterstock/Getty/Flickr 商业许可图,2.65M images total
- **关键: 无 view synthesis GT**

**SSFT 的核心 trick**: swap input/novel views

1. 对每张真实单视图,用 Stage 1 model 生成 3D Gaussian 表示
2. 渲染一个 pseudo-novel view
3. **把 pseudo-novel view 当 input view,真实 image 当 novel view**
4. 用这个 pair 计算 loss

**Intuition**: 这是一个 "self-distillation" 风格的 trick。Stage 1 model 已经能从单图生成 3D 表示,但生成的 pseudo-novel view 可能不完美 (有 artifacts)。把它当 input,要求 model 从这个 "略微 degraded" 的 view 重建出原来的真实图 —— 这迫使 model 学会 "fix" 那些 artifacts,并适应真实图像的 distribution (而不是 synthetic data 的 distribution)。

这比 AdaMPI 的 "warp-back" 策略更优雅,因为 warp-back 直接用 single view warp 不能很好处理 disocclusion,而 SHARP 用 3D Gaussian 表示,可以 fill in occluded content。

**Ablation (Table 12)**: SSFT 在 metric 上没有显著改善 (ScanNet++ DISTS 0.063 → 0.071,实际略升,因为 metric noise),但 Figure 11 qualitative 显示明显 sharper。作者 hypothesize 这是因为 synthetic data 缺少 view-dependent effects (reflections, specular highlights),SSFT 让 model 适应真实世界。

---

## 10. Evaluation:Metric 选择与 Baseline 对比

### 10.1 为什么用 LPIPS / DISTS 而非 PSNR / SSIM

Supplement C.1 做了一个非常 illuminating 的实验。对一张 reference image 做 1% translation,观察 metrics:

| Comparison | DISTS | LPIPS | PSNR | SSIM |
|------------|-------|-------|------|------|
| Translated (0.1%) | 0.008 | 0.059 | 21.3 | 0.623 |
| Translated (1.0%) | 0.079 | 0.491 | 11.2 | 0.375 |
| Translated (5.0%) | 0.121 | 0.723 | 8.1 | 0.249 |
| Mean Image | 0.859 | 0.970 | 10.7 | 0.351 |

**关键观察**: 1% translation 对人眼几乎不可见,但 PSNR 从 21.3 跌到 11.2,接近 mean image 的 10.7。也就是说 PSNR 把 "几乎没区别" 的图当成 "和 mean image 一样差"。

这对 view synthesis 是致命的 —— 即使 depth 估计只差一点点,几何 warp 就会有 sub-pixel 错位,PSNR/SSIM 直接崩盘。LPIPS/DISTS 作为 perceptual metrics,对小 translation 鲁棒得多 (DISTS 1% translation 只 0.079,mean image 0.859,差距大)。

**Intuition**: PSNR 本质上惩罚 "pixel-wise exact match",但 view synthesis 任务本质上不可能 pixel-wise perfect (因为 depth 不可能完美),所以我们关心的是 "perceptual quality"。这个 metric 选择直接反映了 task 的本质。

### 10.2 主实验:Zero-shot Cross-dataset Evaluation

Table 1 的结果非常 striking。SHARP 在所有 6 个 dataset 上 DISTS 和 LPIPS 都最好:

| Method | Middlebury DISTS | ScanNet++ DISTS | WildRGBD DISTS | ETH3D DISTS |
|--------|------------------|------------------|----------------|-------------|
| Flash3D | 0.359 | 0.374 | 0.159 | 0.535 |
| TMPI | 0.158 | 0.128 | 0.114 | 0.396 |
| LVSM | 0.274 | 0.145 | 0.095 | 0.555 |
| SVC | 0.208 | 0.201 | 0.157 | 0.420 |
| ViewCrafter | 0.373 | 0.176 | 0.148 | 0.454 |
| Gen3C | 0.164 | 0.090 | 0.106 | 0.408 |
| **SHARP** | **0.097** | **0.071** | **0.069** | **0.258** |

对比 Gen3C (最强的 diffusion-based baseline):
- Middlebury: 0.097 vs 0.164 (-41%)
- ScanNet++: 0.071 vs 0.090 (-21%)
- WildRGBD: 0.069 vs 0.106 (-35%)
- ETH3D: 0.258 vs 0.408 (-37%)

**Speed 对比 (Table 6)**:
- SHARP: 0.91s inference,0.01s render (100 FPS)
- Gen3C: 830s inference (~14 minutes!)
- ViewCrafter: 120s inference (2 minutes)
- SVC: 60s inference

SHARP 比 Gen3C 快 **910 倍**,且 fidelity 更高。这是一个令人震惊的结果。

### 10.3 With Privileged Depth (Table 7)

给所有方法提供 GT depth 做 scale alignment,SHARP 依然最好。这排除了 "SHARP 只是赢在 depth estimation" 这个 hypothesis —— 即使给其他方法 GT depth,SHARP 仍然更强。说明 SHARP 的优势来自整个 pipeline,不只是 depth。

### 10.4 Motion Range Analysis (Figure 7)

把 camera baseline 分 bin 看 DISTS:
- < 0.5m: SHARP 最好
- 0.5m - 3m: SHARP 仍 top-2
- > 3m: Gen3C 开始超越 (SHARP 的设计范围外)

这印证了 SHARP 的 design philosophy —— 牺牲大 motion range 换取 nearby view 的极致 fidelity。

### 10.5 Number of Gaussians Ablation (Table 14)

| # Gaussians | ScanNet++ DISTS | ScanNet++ LPIPS |
|-------------|-----------------|-----------------|
| 2×196×196 (~77K) | 0.110 | 0.199 |
| 2×392×392 (~307K) | 0.077 | 0.160 |
| 2×784×784 (~1.2M) | 0.064 | 0.147 |

Monotonic improvement with more Gaussians,说明这个 regression framework 还有继续 scale 的空间。

---

## 11. 失败案例与局限性

Figure 8 展示了 3 类典型失败:
1. **Macro photo (strong depth-of-field)**: 蜜蜂的 depth 被错判为花后面,导致翅膀脱离、尾巴扭曲。这是 DoF 误导 depth model 的经典 case。
2. **Starry night sky**: 丰富 texture 让网络把 sky 当成 curvy surface,新视角下严重扭曲。
3. **复杂水面 reflection**: 网络把水中倒影当成远处山脉,水面 broken。

**根因**: 这些都是 depth prediction 的 long-tail failure,即使 unfreezing backbone 也救不回来 (因为 base initialization 已经错了)。

**Motion range 限制**: 超过 headbox (大约 0.5m) 的 motion,quality 下降。作者明确说这是 design choice,不是 bug。

**Future work 方向**:
- 整合 diffusion model 处理 faraway views
- Distillation 减少 diffusion latency
- View-dependent 和 volumetric effects 的 principled treatment (e.g. NeRF-Casting)

---

## 12. 我的整体 Intuition 与 Takeaways

读完这篇 paper,我有几个层面的 takeaways:

### (1) "回归胜过生成" 的 niche

SHARP 证明了在 "nearby view synthesis from single image" 这个特定 niche 上,纯 regression 路线可以超越 diffusion 路线,且快 3 个数量级。这挑战了 "diffusion 总是 better" 的趋势性认知。关键在于问题定义的精度 —— "support headbox motion" vs "support walking around" 决定了方法选择。

### (2) Pretrained backbone 的复用价值

复用 Depth Pro 的 backbone 是非常聪明的选择。340M trainable params 中大部分都是 Depth Pro 的 low-res encoder,这个 encoder 已经学到了 metric depth 所需的丰富 visual features。End-to-end fine-tuning 让它适配下游 task。这印证了 "foundation model + task-specific fine-tuning" 范式在 3D vision 的有效性。

### (3) Training-only Module 的设计 pattern

Depth adjustment module 是 training-only 的,inference 时换成 identity。这种 pattern 让 model 在训练时享受 "privileged information" 的便利,inference 时干净,但通过 information bottleneck (TV + L1 regularizer) 强制网络主体学到 self-sufficient 的能力。这是 C-VAE 思想的简化应用,我觉得非常有启发性 —— 可以推广到很多 "训练有辅助,inference 要 self-contained" 的场景。

### (4) Loss Engineering 仍然重要

在大家都关注 architecture 的当下,这篇 paper 通过精心设计 8 个 loss term (4 data + 4 regularizer),每个都有 specific purpose,达到了 SOTA。特别是 Gram matrix loss 的复活 (原本用于 style transfer) 用在 view synthesis 上提升 sharpness,这种 cross-task insight transfer 值得学习。

### (5) Computation Graph Surgery

Supplement A.4 的这个工程技巧我觉得可以单独发一篇 paper。在 loss function 用大网络 (ResNet-50) 但只 backprop 到中间 features 时,手动 surgery computation graph 避免 OOM,同时保持 FP32 精度。这个 trick 在 NeRF/3DGS 训练中应该有广泛应用前景。

### (6) "Sharp" 这个词的精确含义

Paper 标题的 "Sharp" 不是随便选的。SHARP 既代表 "Single-image High-Accuracy Real-time Parallax",也强调 output image 的 sharpness —— 这是和 diffusion-based method 在 nearby views 上的关键区别。Diffusion 倾向于生成 "plausible but not pixel-accurate" 的内容,而 SHARP 通过 regression 保留了 input image 的 sharp details,在新视角下也保持这个 sharpness。Gram matrix loss 是实现这一点的关键技术。

### (7) 3D Gaussian Splatting 作为 "Amortizable 3D Representation"

SHARP 展示了 3DGS 的一个新用法 —— 不只是 per-scene optimization 的输出,而是 feed-forward regression 的输出。这意味着 3DGS 可以作为 "amortizable 3D representation":一旦 synthesize 完成,可以 real-time render from arbitrary nearby views,不需要重新 inference。这和 diffusion-based 方法每次推理都要从头 denoise 是本质区别。

### (8) 工程细节决定上限

这篇 paper 的成功很大程度上来自工程细节:
- 用 min-pooling 而非 average pooling 做 depth downsample
- Position delta 在 NDC space 操作
- Attribute-specific activation function 和 scale factor
- TV regularizer 的多尺度 (downsample 6 个 scale)
- Splat size 在 screen space 约束而非 world space

这些细节单独看都不大,但累加起来造就了 SOTA。这印证了 "research is 1% inspiration, 99% perspiration" 在 deep learning 时代依然成立。

---

## 13. 与我之前工作的关联 (Karpathy 视角)

如果我 (Karpathy) 看这篇 paper,会想到几个关联点:

### (a) "Software 2.0" 视角

SHARP 是 Software 2.0 的一个经典 case —— 用神经网络替代了传统的 SfM + MVS + texture synthesis pipeline。整个 pipeline 就是 forward pass + differentiable rendering,所有 "engineering" 都被 absorbed into 神经网络的 weights 和 loss function。这种 "amortize everything into a forward pass" 的 pattern 在我之前讲 nanoGPT, micrograd 时也强调过。

### (b) "Bitter Lesson" 的体现

Rich Sutton 的 Bitter Lesson 说 "general methods that leverage computation are ultimately the most effective"。SHARP 某种程度上验证了这一点 —— 700K synthetic scenes + 2.65M real images,128 GPU × 100K + 32 GPU × 60K 的训练规模,这是 "scale up computation" 的胜利。但同时也反 Bitter Lesson —— SHARP 用了大量 human insight (loss design, depth adjustment, normalized space),纯 scale up 不够。

### (c) Modular Design 与 End-to-End

SHARP 是 modular design (4 个 module) 但 end-to-end training。这和 Tesla FSD 的设计哲学类似 —— 模块化结构便于理解和 debug,但 end-to-end gradient flow 让每个 module 都为最终目标优化。这种 "modular but end-to-end" 是当前 3D vision 的 sweet spot。

### (d) Depth Pro 作为 "Foundation Model for Geometry"

Depth Pro 不仅是 depth estimator,在 SHARP 中被用作 general feature extractor。这暗示了一个 trend —— depth/metric-geometry foundation model 可能成为 3D vision 的 "backbone",类似 BERT/ViT 在 NLP/CV 中的地位。未来可能会有 "Depth Pro-style backbone + task-specific head" 成为 3D vision 的 standard recipe。

---

## 14. 可能的扩展与开放问题

读完 paper,我有几个 "next step" 的想法:

1. **Video input extension**: Paper 提到 future work 可能 unify single-view, multi-view, video input。如果有 video input,可以利用 temporal consistency constraint,可能进一步改善 quality。
2. **Diffusion distillation for faraway views**: 把 diffusion model 的 "synthesize faraway content" 能力 distill 到 feed-forward model,可能是 unified solution 的方向。
3. **Dynamic scenes**: 当前 SHARP 假设 static scene。如何 handle dynamic content (人、动物运动) 是个 open problem。
4. **Uncertainty quantification**: Depth adjustment module 训练时学到 depth uncertainty,但 inference 时丢弃。能否保留 uncertainty 用于 adaptive rendering (e.g. uncertain regions 用更多 Gaussians)?
5. **Beyond 2 layers**: 当前固定 2 layer Gaussians。能否做 adaptive layer count per pixel? 或者用 multi-resolution Gaussians (类似 mip-splatting)?
6. **Joint depth + view synthesis training**: 当前 depth loss 和 view synthesis loss 是 weighted sum。能否做 multi-task learning with uncertainty weighting (Kendall et al.)?

---

## References

- [SHARP GitHub](https://github.com/apple/ml-sharp)
- [Depth Pro: Sharp monocular metric depth in less than a second (Bochkovskii et al., ICLR 2025)](https://arxiv.org/abs/2410.19115)
- [3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [DPT: Vision Transformers for Dense Prediction (Ranftl et al., ICCV 2021)](https://arxiv.org/abs/2103.13413)
- [LPIPS: The Unreasonable Effectiveness of Deep Features (Zhang et al., CVPR 2018)](https://arxiv.org/abs/1801.03924)
- [DISTS: Image Quality Assessment (Ding et al., PAMI 2022)](https://arxiv.org/abs/2004.07728)
- [AdaMPI (Han et al., SIGGRAPH 2022)](https://arxiv.org/abs/2204.07192)
- [Flash3D (Szymanowicz et al., 3DV 2025)](https://arxiv.org/abs/2406.04343)
- [Splatter Image (Szymanowicz et al., CVPR 2024)](https://arxiv.org/abs/2312.06604)
- [Gen3C (Ren et al., CVPR 2025)](https://research.nvidia.com/labs/toronto-ai/gen3c/)
- [ViewCrafter (Yu et al., 2024)](https://arxiv.org/abs/2409.02048)
- [Stable Virtual Camera (Zhou et al., 2025)](https://arxiv.org/abs/2503.14489)
- [C-VAE: Sohn et al., NeurIPS 2015](https://proceedings.neurips.cc/paper/2015/hash/8d55a249e6ba2455a8b07c401be6bb38-Abstract.html)
- [NeRF-Casting (Verbin et al., SIGGRAPH Asia 2024)](https://arxiv.org/abs/2406.06101)
- [ScanNet++ (Yeshwanth et al., ICCV 2023)](https://github.com/scannetpp/scannetpp)
- [ViT: Dosovitskiy et al., ICLR 2021](https://arxiv.org/abs/2010.11929)

---

总结一句: SHARP 是一个典型的 "problem formulation + engineering细节 + scale" 三者结合的胜利。它没有引入 fundamentally new 的 ML idea,但通过精确定义问题 (nearby view, < 1 sec, metric scale),精心设计每个组件 (depth adjustment, normalized space, attribute-specific composer, multi-loss engineering),并 scale up training data,在特定 niche 上大幅超越 prior work。这种 "工程主义胜利" 的 pattern 在当前 deep learning 时代非常常见,值得仔细 study。
