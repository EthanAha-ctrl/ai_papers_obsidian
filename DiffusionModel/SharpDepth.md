---
source_pdf: SharpDepth.pdf
paper_sha256: fcf5928cfaceab887bc37abe569274b6d9bb8d672c96a0aeaf6d2b00352ae46d
processed_at: '2026-08-12T05:46:24-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SharpDepth

好，我把那堆公式和术语扔一边，给你讲讲这 paper 到底在干啥。

---

## 一句话版本

**找两个"各有所长但各有缺陷"的深度模型，让它们的分歧当"哪里需要修"的导航，然后用一个新模型把两边的优点都偷过来。**

---

## 背景这俩模型各自啥毛病

### UniDepth（discriminative 路线）

UniDepth 是个直接 regress depth 的网络，trained on LiDAR GT。它 metric accuracy 很准——告诉它一张图，它能告诉你 "这车离你 12.3 米"。

但是 LiDAR 的 GT 本身有个致命问题：**稀疏**。远处基本没点，物体边缘也没点。所以模型训练时看到的 GT 就是这样：

```
真实场景:  车 ████████████  栅栏 ||||||||||
LiDAR GT:  车 ·  ·    ·  ·       ·  · · ·  
```

模型为了 minimize L2 loss，学会了一个 safe 策略：**不确定的地方就平均一下**。于是车辆边缘的 depth 就糊了，栅栏的栅栏条直接被抹平成一堵墙。

这是 regression 的宿命——L2 的最优解是 conditional mean，在 bimodal 区域（前景 vs 背景）mean 就是个中间值。

### Lotus（generative 路线）

Lotus 是 Stable Diffusion fine-tune 出来的 depth 模型，trained on synthetic data（dense GT）。它的输出非常 sharp，栅栏条一根根清楚，fence 的网眼都能看见。

但是它有俩毛病：
1. **不知道绝对 scale**——它只能告诉你 "这车比那个墙近"，但不知道具体几米
2. **synthetic data 训练，real image 上有 domain gap**——有时候 texture 会误导它，比如墙上贴张海报它可能以为是窗户

---

## 关键 insight：分歧就是信号

作者的核心观察特别 simple 但特别有效：

**两个模型如果都 agree 的地方，大概率是 metric model（UniDepth）对的——因为 generative model 在 plain region 上不会乱来。**

**两个模型 disagree 的地方，大概率是 generative model（Lotus）对 boundary 更准——因为 discriminative model 在那里 mean-seeking，会糊。**

所以他们算了张 difference map：

```
把 Lotus 的输出 align 到 UniDepth 的 scale 上（缩放+平移）
然后逐 pixel 算绝对差
亮的地方 = 俩模型意见分歧 = 需要修
暗的地方 = 俩模型意见一致 = 别动
```

你看 Figure 4 那张图，路面都是暗的（俩模型都说路面在 10 米左右，agree），车轮边缘、栅栏条都是亮的（UniDepth 糊成一团，Lotus 一根根分清楚，disagree 巨大）。

这相当于 free 拿到了一张 "uncertainty map"，不需要任何 GT label。

---

## 怎么用这张 difference map

### 第一步：Noise-aware Gating

传统 diffusion 是给整张 depth latent 都加噪声然后去噪。SharpDepth 不这么干，它**只给分歧大的地方加噪声**：

```
z_d' = diff_map ⊙ 噪声  +  (1 - diff_map) ⊙ 原始depth_latent
```

直觉：这就好比说"亮的地方我啥也不信，你重新生成；暗的地方我已经有答案了，你照抄就行"。

这招借鉴自 image inpainting——给要修的地方涂白，让模型重新画；保留不要动的地方。

好处是 model 的 capacity 集中在真正需要修的地方，不会浪费在已经很准的路面、墙面上。

### 第二步：SDS Loss（偷 Lotus 的 sharpness）

接下来训练一个 sharpener $G_\theta$（init 自 Lotus 的权重）。让它输出一个 sharpened depth latent $\hat{z}$，然后用 Score Distillation Sampling 拉 $\hat{z}$ 往 Lotus 的 manifold 靠：

$$\nabla_\theta \mathcal{L}_{SDS} = \mathbb{E}_{t,\epsilon}[w^t (\hat{z} - f_G(\hat{z}^t; z_i, t))]$$

人话：**把 $G_\theta$ 的输出稍微加噪声，再丢给 frozen Lotus 让它 predict 回来，如果 predict 出来的跟原始输出对不上，就 push $G_\theta$ 调整，让它对得上。**

这样 $G_\theta$ 就被迫学习 Lotus 的 output distribution——也就是 Lotus 那种 sharp boundary 的 prior。

关键 trick：**不需要 backprop through Lotus**，所以 memory 和 compute 都很省。这是从 DreamFusion 那里来的 SDS 技巧，本用来做 3D 资产生成，这里 repurpose 成 depth distillation。

### 第三步：Reconstruction Loss（锁住 metric scale）

但光靠 SDS 有问题——$G_\theta$ 会 drift 到 Lotus 的 affine-invariant scale 上，丢掉 metric accuracy。所以加个 regularization：

$$\mathcal{L}_{recons} = \| e \odot (\hat{d} - d) \|$$

人话：**在 difference map 大的地方（uncertain region），允许 $G_\theta$ 偏离 UniDepth 的 $d$（毕竟 UniDepth 在那里不准，要修）；在 difference map 小的地方（confident region），强制 $G_\theta$ 等于 $d$（UniDepth 在那里准，别乱改）。**

这是 soft regularization，不是 hard 的 $\|\hat{d} - d\|=0$，因为那样会把 sharpening 全抹掉。

### 第四步：EMA self-teacher 的小 trick

随着训练进行，$G_\theta$ 自己越来越强，比原始 Lotus 还强。这时候再算 difference map，应该用谁当 Lotus 的替代？

作者让 $G_\theta$ 自己的 EMA 版本当 teacher。这就成了 self-distillation：difference map 会逐渐变小，uncertain region shrinking，model 越来越聚焦在真正难的地方。

Ablation 里 EMA 版本比 frozen Lotus 好（$\delta_1=0.973$ vs $0.967$，DBE completion $36.4$ vs $40.6$），证明这个动态 self-teaching 有效。

---

## 为什么不需要 GT

整个 pipeline 只需要：
1. RGB image
2. UniDepth 的 inference output $d$
3. Lotus 的 inference output $\tilde{d}$（或者训练后期用 EMA 自身的 output）

没有 GT depth 参与训练。所以训练集只需要 90k 张 RGB image（用 6 个真实 dataset 各 1%），比 UniDepth/Metric3D 的训练数据小 100-150 倍。

这点我觉得挺重要——它本质上是在做 **foundation model 的 amortized refinement**，类似 LLM RLHF 不需要新数据但能 align 模型行为。SharpDepth 不需要新 label 但能 align 两个模型的互补优势。

---

## 效果咋样

### Accuracy（Table 1）

KITTI / NYUv2 上 SharpDepth 和 UniDepth 几乎打平（$\delta_1$ 都 0.97 左右），说明 sharpening 没破坏 metric。

但是 ETH3D 上 UniDepth 本身就崩（$\delta_1=0.25$），SharpDepth 也跟着崩（0.23）。这告诉我们一个 limitation：**SharpDepth 的 metric ceiling 取决于 base metric model $f_D$**。如果 $f_D$ 在某 domain 上完全失败，SharpDepth 救不回来。

### Sharpness（Table 2, DBE metrics）

DBE 是测边界 sharpness 的，分 accuracy（false positive）和 completion（false negative）。

UniDepth 的 completion 巨差（Sintel 113.3，Spring 229.7），因为它漏掉大量 thin structure 的 edge——这就是 "blurry mean" 的代价。

SharpDepth 的 completion 在 Sintel 上是 36.2，比 UniDepth 好 3 倍，比 Lotus 略差（31.9）。accuracy 上比 Lotus 还略好（1.94 vs 2.03）。

这就是 paper 卖的 "balance"——metric 接近 UniDepth，sharpness 接近 Lotus。

### Visual（Figure 5, 6）

最直观的是 Figure 6 的 point cloud：
- 榴莲的刺：UniDepth 一坨糊的，SharpDepth 一根根清楚
- 键盘键帽：UniDepth 平的，SharpDepth 一个个有立体感
- KITTI 上的车：UniDepth 的车像被风化了，SharpDepth 棱角分明

---

## 几个值得琢磨的点

### 1. 这套思路的 generalization

Difference map + SDS + soft reconstruction 这套 framework 其实不限 depth。任何有"discriminative 准但糊 vs generative sharp 但偏"的 task 都能用：
- Surface normal estimation（discriminative 准但 boundary 糊，generative sharp 但 global 不一致）
- Optical flow（discriminative 准但 motion boundary 糊，generative sharp 但可能有 hallucination）
- Segmentation edge refinement
- Panoptic segmentation 的 boundary

可能是个 general recipe。

### 2. Limitation 盲点

如果两个 base model 在某个区域都错，且错的方式类似（比如 reflective surface 上都 hallucinate），difference map 不会亮，sharpening 不会发生。Paper 没讨论这个。

更 robust 的做法可能是用 ensemble disagreement 或者 evidential deep learning 来 estimate 真正的 epistemic uncertainty，而不只是 model disagreement。

### 3. 跟 Gaussian Splatting 的契合

Section 9.1 把 SharpDepth 喂给 MonoGS（Gaussian Splatting SLAM），PSNR 从 18.47 涨到 18.86。这点其实有想象空间——Gaussian Splatting 对 depth edge 非常敏感，sharp edge 需要很多 small Gaussians，blurry edge 会被迫用大 Gaussian 摊平。如果 SharpDepth 作为 GS 的 depth initialization，可能能端到端 train "depth-conditioned Gaussian Splatting"。

### 4. Thin structure 对 MVS 的价值

Classic MVS（patch-match based）在 thin structure 上一直崩——fence、pole、leaf 这些地方 patch-match 没法 convergence。如果 SharpDepth 当 MVS 的 depth prior，可能能 fix 这个长期痛点。这是个潜在的 follow-up 方向。

### 5. 为什么不用 GT 反而更好

这个观察反直觉但合理：synthetic GT dense 但有 domain gap，real LiDAR GT sparse 但真实。SharpDepth 用 real image + 两个 foundation model 的 disagreement 当 supervision，**绕开了 GT 的 fundamental trade-off**。这思路挺有启发性的——未来可能很多 task 都往这方向走，不依赖完美 GT 而是利用现成 foundation model 之间的互补。

---

## 简单收尾

如果让我跟一个完全不懂的人讲这 paper：

> 以前要 depth 又准又清楚，要么用 LiDAR 训的模型（准但糊），要么用 diffusion 模型（清楚但不知道几米）。SharpDepth 想了招：把两个模型的分歧当 GPS，分歧大的地方用 diffusion 模型修边界，分歧小的地方保留 LiDAR 模型的 metric accuracy。整个过程不需要新的 GT 数据。

就这么个事。Paper 的 elegance 在于它把一个看似需要 dense GT 的问题（refine boundary）用 model disagreement 的 amortized supervision 解决了，技术上用 SDS 做了 cheap distillation。这个 recipe 我觉得未来会被很多类似 task 借鉴。

---

# SharpDepth: 深度讲解

Andrej，这篇 paper 我非常想跟你聊聊，因为它本质是在做一个非常 elegant 的 distillation——把两个 "正交弱点" 的模型互补，用 diffusion distillation 的技巧把它们 glue 起来。这个 setup 我觉得非常漂亮，因为它绕开了一个根本的困境：synthetic data 的 dense GT 能给 sharp boundary，real-world sparse LiDAR GT 能给 metric accuracy，但二者你鱼与熊掌不可兼得。SharpDepth 用一个 ground-truth-free 的 fine-tuning pipeline 把两者合在一起。

让我从 intuition 开始 build。

---

## 1. 问题本质：两套 prior 的 trade-off

要 build intuition，我们先把 monocular depth estimation 的两大家族拆开看：

### Discriminative methods（UniDepth, Metric3D, ZoeDepth）
这些模型 trained on real-world data + sparse LiDAR GT，直接 regress metric depth $d \in \mathbb{R}^+$ (meters)。

**为什么 sharpness 差？** 因为 LiDAR 在远处稀疏、且物体边缘 (boundary) 处常常 missing 或点云稀疏。模型在 regression objective 下会趋向 "最小化 expected L2 error"，而 L2 的最优解在不确定区域是 conditional mean，即**平均化**。这就是为什么 thin structures（fences, poles, leaves）的 depth 总是糊的。

更技术一点：如果 GT 分布是 $p(d|x)$，L2 最优 regressor 输出 $\hat{d}^* = \mathbb{E}[d|x]$。在边界处，$p(d|x)$ 是 bimodal（前景 vs 背景），mean 就是中间值，产生 "halo" 或 "bleeding"。

### Generative methods（Marigold, Lotus, GeoWizard）
这些基于 Stable Diffusion fine-tune，trained on synthetic data + dense GT。它们输出 affine-invariant depth $\tilde{d}$，只能 recover 相对深度，没有 metric scale。

**为什么 sharpness 好？** 因为 latent diffusion 学的是 $p(d|x)$ 的 sample（mode-seeking 而非 mean-seeking），且继承了 Stable Diffusion 在 LAION 上学到的强 image prior，对 texture、boundary、thin structure 都很敏感。

**为什么 metric 不准？** 因为 synthetic data (HyperSim, VirtualKitti) 的 camera intrinsic、scene scale 都和 real world 有 domain gap。Marigold 的输出经过 min-max normalize，丢掉了绝对 scale。

### SharpDepth 的核心观察

作者的关键 insight 是：**如果两个模型在某个 region 上 prediction 一致，那大概率 discriminative model 在那里是对的（metric 准）**；**如果不一致，那大概率 generative model 在那里 boundary 更准**（因为 generative 对 texture/boundary 敏感，而 discriminative 在那里 mean-seeking）。

这就是 difference map $e$ 的来源。它充当一个 "uncertainty mask" 的 proxy，告诉 sharpening model：哪里是 confident region（保留 metric），哪里是 uncertain region（需要 generative prior 来 sharpen）。

这其实是非常经典的 amortized inference 思路：用一个 cheap 的 disagreement signal 替代 expensive 的 per-pixel ground truth labeling。

---

## 2. 方法架构总览

参考 Figure 3，pipeline 是：

```
Input image I
    │
    ├──► f_D (UniDepth) ─────► d  (metric depth)
    │
    └──► f_G (Lotus) ─────────► d̃ (affine-invariant depth)
                                      │
              align d̃ to range of d (scale & shift)
                                      │
                            difference map e = |d̃_aligned - d|
                                      │
                          Noise-aware Gating (Eq. 4)
                                      │
                          z_d' = ê ⊙ ε + (1-ê) ⊙ z_d
                                      │
                                      ▼
                          G_θ (sharpener, init from Lotus)
                                      │
                                      ▼
                                  ẑ (predicted latent)
                                      │
                ┌─────────────────────┴─────────────────────┐
                ▼                                           ▼
        SDS loss (Eq. 5)                        Reconstruction loss (Eq. 6)
        distill from f_G                         stay close to d
                │                                           │
                └─────────────────► L_total (Eq. 7) ◄────────┘
```

整个训练 **不需要 GT depth**，只需要 RGB image + 两个 pre-trained model 的 inference outputs。这是这篇 paper 的另一大卖点——training data 只用了 ~90k images，比 Metric3D / UniDepth 用的少了 100-150×。

---

## 3. Noise-aware Gating 深度解析

这是 paper 最关键的创新，让我把数学摊开讲。

### 3.1 Difference map 的构造

给定：
- $d = f_D(I)$: metric depth prediction
- $\tilde{d} = f_G(I)$: affine-invariant depth prediction

由于 $d$ 和 $\tilde{d}$ 在不同的 scale 上，先对齐：

$$\tilde{d}_{aligned} = \alpha \cdot \tilde{d} + \beta$$

其中 $\alpha, \beta$ 通过 least-squares fitting 到 $d$ 求得（或者用 min-max normalize）。然后：

$$e = |\tilde{d}_{aligned} - d|$$

$e$ 是一张和 $I$ 同分辨率的 difference map，亮的地方代表两个 model disagreement 大（uncertain），暗的地方代表 agreement（confident）。

看 Figure 4 你能立刻 get 这个 intuition：在 KITTI 的车辆 + 道路场景里，路面这种大面积、texture-rich 但 depth-smooth 的区域，UniDepth 和 Lotus 都 predict 得差不多（difference map 暗）；但车辆边缘、薄结构（如车轮辐条、栅栏），二者 differ 剧烈（difference map 亮）。

### 3.2 Latent noise blending

接下来要把 $e$ 注入到 latent space。注意 diffusion model 的 latent $z_d$ 是 VAE encoded 后的（分辨率是原图的 1/8，通道 4）。所以 $e$ 先 downsample 到 $\hat{e}$ 匹配 $z_d$ 的空间尺寸。

Eq. (4) 的核心 operation：

$$z_d' = \hat{e} \odot \epsilon + (1 - \hat{e}) \odot z_d$$

变量含义：
- $z_d = \mathcal{E}(d)$: VAE encoder 把 depth map $d$ 编码到 latent
- $\epsilon \sim \mathcal{N}(0, I)$: 标准高斯噪声，shape 与 $z_d$ 相同
- $\hat{e} \in [0, 1]^{H/8 \times W/8}$: downsampled difference map，归一化到 [0,1]
- $\odot$: element-wise (Hadamard) product

**直觉解释**：这个公式本质上是一个 **soft mask 的 inpainting**。在 difference 大的地方（$\hat{e} \to 1$），latent 几乎全是噪声，sharpening model 必须从 conditional image latent $z_i$ 和 metric prior 中"重新生成" depth；在 difference 小的地方（$\hat{e} \to 0$），latent 保留原始 $z_d$ 不动，相当于告诉 model "这里不要改"。

这就把 inpainting 的思想 (BrushNet [20], RePaint [27], TryOnDiffusion [56]) 引到了 depth refinement 上。这种 selective noising 比传统 diffusion forward process（给所有 pixel 都加噪声）更高效，因为它把 model 的 capacity 集中在 uncertain region。

### 3.3 EMA self-teacher 的 trick

Section 4.1 末尾有个非常 subtle 的设计：随着训练进行，$G_\theta$ 的输出已经比 original Lotus 更好，于是 $\tilde{d}$ 不再来自 fixed Lotus，而来自 $G_\theta$ 的 EMA 版本 $G_{\bar\theta}$。

这是个 **self-distillation** 的味道。difference map 会随着 training 逐渐变小（因为 model 越来越准），uncertain region 也 shrinking。这避免了 model overfit 到 fixed 的 Lotus-UniDepth disagreement，而是动态地 refine。

Ablation Table 3 的 setting G vs H 印证了这一点：
- (G) Frozen Lotus teacher: $\delta_1=0.967$, $\epsilon_{PDBE}^{compl}=40.6$
- (H) EMA update (ours): $\delta_1=0.973$, $\epsilon_{PDBE}^{compl}=36.4$

EMA 版本在两个 metric 上都更好。

---

## 4. SDS Loss 深度解析

### 4.1 原始 SDS

先 recall DreamFusion 的 SDS（Eq. 3）：

$$\nabla_\phi \mathcal{L}_{SDS}(\theta, x=g(\phi)) \triangleq \mathbb{E}_{t,\epsilon}\left[ w(t) (\epsilon_\theta(z_i^t, y, t) - \epsilon) \right]$$

变量：
- $\phi$: 3D scene parameters (NeRF / 3DGS)
- $x = g(\phi)$: differentiable rendering
- $z_i^t = \sqrt{\alpha_t} x + \sqrt{1-\alpha_t} \epsilon$: forward noised latent at timestep $t$
- $\epsilon_\theta$: frozen diffusion U-Net (score function approximator)
- $w(t)$: timestep-dependent weighting

直觉：SDS 让 $x$ "looks like" 一个 sample from diffusion model 的 distribution，但不需要 backprop through U-Net（去掉 Jacobian 项），所以 cheap。

### 4.2 Marigold 的 ε-prediction

Marigold fine-tune Stable Diffusion 学 depth，用 ε-prediction（Eq. 1）：

$$\mathcal{L}_\theta = \mathbb{E}_{t, \epsilon \sim \mathcal{N}(0,1)} \| \epsilon_\theta(z_d^t, t, z_i) - \epsilon \|_2^2$$

变量：
- $z_d = \mathcal{E}(d)$: depth map 的 VAE latent
- $z_d^t = \sqrt{\bar\alpha_t} z_d + \sqrt{1-\bar\alpha_t} \epsilon$: 在 timestep $t$ 加噪
- $z_i = \mathcal{E}(I)$: condition image latent
- $\epsilon_\theta$: U-Net 预测的 noise

注意 Marigold **去掉了 text condition**，把 $z_i$ 作为唯一 condition concat 到 $z_d^t$ 上。

### 4.3 Lotus 的 z0-prediction

Lotus（Eq. 2）改用 z0-prediction，直接预测 clean latent：

$$\mathcal{L}_\theta = \|z_d - f_\theta(z_d^t, z_i, t, s_d)\|_2^2 + \|z_i - f_\theta(z_i^t, t, s_i)\|_2^2$$

变量：
- $f_\theta$: z0-prediction network (output is predicted clean latent)
- $s_d, s_i$: task indicator (depth vs image), 这是 multi-task joint training 防止 catastrophic forgetting
- $t$: 单一 timestep（Lotus 把 1000 步压到 1 步）

### 4.4 SharpDepth 的修改版 SDS

Eq. (5) 是 paper 的核心创新：

$$\nabla_\theta \mathcal{L}_{SDS} \triangleq \mathbb{E}_{t, \epsilon}\left[ w^t \left( \hat{z} - f_G(\hat{z}^t; z_i, t) \right) \right]$$

变量：
- $\hat{z} = G_\theta(z_d', z_i)$: 我们的 sharpener 输出（clean latent prediction）
- $\hat{z}^t = \sqrt{\bar\alpha_t} \hat{z} + \sqrt{1-\bar\alpha_t} \epsilon$: forward noise 一下 $\hat{z}$
- $f_G$: frozen Lotus（z0-predictor）
- $w^t$: weighting

**直觉**：我们让 $G_\theta$ 输出的 $\hat{z}$，"经过 forward noise 一下再被 Lotus 预测回来"应该等于 $\hat{z}$ 自己。如果 $\hat{z}$ 不在 Lotus 学到的 distribution 上，那么 $f_G(\hat{z}^t)$ 会和 $\hat{z}$ 不一致，gradient 会 push $\hat{z}$ 朝着 Lotus 的 manifold 移动。

这是 SwiftBrush [30] 的 one-step distillation 思路 extended 到 depth domain。SwiftBrush 是把 multi-step Stable Diffusion distill 成 one-step text-to-image，SharpDepth 是把 multi-step Lotus distill 成 one-step sharpener，**但同时保留 metric depth 的 anchor**。

注意 Eq. (5) 是对 $\theta$ 求 gradient，但 **不需要 backprop through $f_G$**——这就是 SDS 的妙处。$f_G$ 只需要 forward pass，大大节省 memory 和 compute。

### 4.5 为什么 SDS 能带来 sharpness

这一点 paper 没明说，但我想强调一下：SDS 的本质是 score matching，让 $G_\theta$ 的输出在 Lotus 的 data manifold 上。Lotus 的 manifold 是从 synthetic dense GT 学来的，**所以 manifold 上天然有 sharp boundaries**。把 $G_\theta$ 的 output push 到这个 manifold 上，就继承了 sharpness。

而 UniDepth 的 output 是从 sparse LiDAR 学的 regression，它的 manifold 是 "blurry mean" 的 manifold。

---

## 5. Noise-aware Reconstruction Loss

Eq. (6)：

$$\mathcal{L}_{recons} = \| e \odot (\hat{d} - d) \|$$

变量：
- $e$: difference map（注意这里用的是原图分辨率的 $e$，不是 downsampled $\hat{e}$）
- $\hat{d}$: $G_\theta$ 输出的 depth（decode 自 $\hat{z}$）
- $d$: UniDepth 输出的 metric depth
- $\odot$: element-wise product

**直觉**：这个 loss 在 difference 大的地方（uncertain region）给大权重，difference 小的地方（confident region）给小权重。它的作用是 **拉住 $G_\theta$，不让它完全 drift 到 Lotus 的 affine-invariant scale 上**。

注意这个 loss 是 asymmetric 的：
- 在 $e$ 大的地方：$(\hat{d} - d)$ 可以大（允许 sharpen），但 $e$ 加权后 gradient 仍然大（推动收敛）
- 在 $e$ 小的地方：$(\hat{d} - d)$ 应该接近 0（保留 metric），$e$ 加权后 gradient 也接近 0（不 push）

这相当于一个 **soft regularization**：metric accuracy 是 anchor，但允许偏离——偏离程度由 difference map 决定。比硬性的 $\|\hat{d} - d\|$ 好得多，因为后者会强制 $\hat{d} = d$ 完全抹平 sharpening。

### 5.1 完整 loss

Eq. (7)：

$$\mathcal{L}_{total} = \lambda_{SDS} \mathcal{L}_{SDS} + \lambda_{recons} \mathcal{L}_{recons}$$

作者取 $\lambda_{SDS} = 1.0, \lambda_{recons} = 0.3$。

这个 0.3 是经验值，可以从 ablation Table 3 推断：
- (C) w/o SDS loss：$\delta_1=0.978$ (metric 最好)，但 $\epsilon_{PDBE}^{compl}=112.5$ (boundary 最差)
- (D) w/o recons loss：$\delta_1=0.843$ (metric 最差)，$\epsilon_{PDBE}^{compl}=34.1$ (boundary 最好)

可以看到两个 loss 完全对立：SDS 给 sharpness 但伤害 metric，recons 给 metric 但伤害 sharpness。$\lambda_{recons}=0.3$ 是 sweet spot，让 metric 接近 UniDepth ($\delta_1=0.973$)，boundary 接近 Lotus ($\epsilon_{PDBE}^{compl}=36.4$)。

---

## 6. 实验数据精读

### 6.1 主表 Table 1: Real-image depth accuracy

| Method | GT-aligned? | KITTI $\delta_1$↑ | KITTI A.Rel↓ | NYUv2 $\delta_1$↑ | NYUv2 A.Rel↓ | ETH3D $\delta_1$↑ | ETH3D A.Rel↓ |
|---|---|---|---|---|---|---|---|
| UniDepth | No | **0.98** | **0.05** | **0.98** | **0.05** | 0.25 | 0.46 |
| Metric3Dv2 | No | 0.98 | 0.05 | 0.97 | 0.07 | 0.82 | 0.14 |
| UniDepth-aligned Lotus | No | 0.84 | 0.13 | 0.94 | 0.09 | 0.20 | 0.49 |
| PatchRefiner | No | 0.79 | 0.16 | 0.01 | 2.48 | 0.05 | 1.78 |
| **SharpDepth (ours)** | No | **0.97** | **0.06** | **0.97** | **0.06** | 0.23 | 0.47 |

读法：
- KITTI/NYUv2: SharpDepth 和 UniDepth 几乎打平，证明 sharpening 没破坏 metric accuracy
- ETH3D: 这里 UniDepth 本身就崩了 ($\delta_1=0.25$)，SharpDepth 也跟着崩。这暴露一个 limitation：SharpDepth 的 metric accuracy ceiling 取决于 base metric model $f_D$。如果 $f_D$ 在某 domain 上很差，SharpDepth 救不回来。
- PatchRefiner 在 NYUv2 上 $\delta_1=0.01$ 灾难性失败，因为它 trained on outdoor synthetic data，indoor domain shift 太大
- "UniDepth-aligned Lotus" 这个 baseline 很重要：它说明 naive 的 "Lotus sharpness + UniDepth scale" 不够好（KITTI $\delta_1=0.84$ vs SharpDepth 0.97）

### 6.2 Table 2: Depth detail (DBE metrics)

DBE (Depth Boundary Error) 是 iBims [22] 提出的，专门测边界 sharpness。Paper 用 Pseudo DBE (PDBE) 来处理 synthetic data 没有 GT edge 的情况，用 Canny edge detection 提取 edge sets 再比较。

| Method | Sintel $\epsilon_{PDBE}^{acc}$↓ | Sintel $\epsilon_{PDBE}^{compl}$↓ | Spring $\epsilon_{PDBE}^{acc}$↓ | Spring $\epsilon_{PDBE}^{compl}$↓ | iBims $\epsilon_{DBE}^{acc}$↓ | iBims $\epsilon_{DBE}^{compl}$↓ |
|---|---|---|---|---|---|---|
| Lotus | 2.03 | 31.9 | 1.27 | 102.8 | 1.92 | 11.0 |
| UniDepth | 3.73 | 113.3 | 5.29 | 229.7 | 2.00 | 30.0 |
| Marigold | 1.90 | 52.5 | 1.85 | 150.3 | 1.85 | 13.4 |
| **SharpDepth** | **1.94** | **36.2** | **1.24** | **147.6** | **1.80** | **13.1** |

读法：
- accuracy ($\epsilon^{acc}$): 预测 edge 中有多少是真 edge（false positive 衡量）
- completion ($\epsilon^{compl}$): 真 edge 中有多少没被预测到（false negative 衡量）

UniDepth 的 completion 都很差（113.3 / 229.7 / 30.0），意味着它**漏掉大量 edge**——这正符合 "blurry mean" 的预期。

SharpDepth 在 accuracy 上和 Lotus 持平甚至略好（iBims 1.80 vs Lotus 1.92），在 completion 上稍逊于 Lotus（36.2 vs 31.9）但远好于 UniDepth。这就是 paper 卖的 "balance"。

### 6.3 Ablation Table 3 总结

我把关键 setting 拎出来：

| Setting | KITTI $\delta_1$↑ | KITTI A.Rel↓ | Sintel $\epsilon_{PDBE}^{acc}$↓ | Sintel $\epsilon_{PDBE}^{compl}$↓ |
|---|---|---|---|---|
| Ours (full) | 0.973 | 0.060 | 1.94 | 36.4 |
| A: Noise latent (no diff map) | 0.817 | 0.135 | 1.94 | 34.6 |
| B: Input $z_d$ (no noise) | 0.701 | 0.186 | 3.30 | 116.9 |
| C: w/o SDS | 0.978 | 0.051 | 3.70 | 112.5 |
| D: w/o recons | 0.843 | 0.128 | 1.94 | 34.1 |
| F: Marigold teacher | 0.973 | 0.058 | 2.40 | 84.7 |
| G: Frozen Lotus | 0.967 | 0.069 | 2.00 | 40.6 |
| H: EMA update | 0.973 | 0.060 | 1.94 | 36.4 |

关键 takeaways：
1. **Setting A vs Ours**: 加 noise 但用 difference map 引导 vs 纯 random noise。Metric 差异巨大 (0.817 vs 0.973)。证明 difference map 的引导是核心，不是单纯加 noise。
2. **Setting B vs Ours**: 不加 noise（直接把 $z_d$ 喂进去）几乎完全失败 (0.701)。这反直觉但合理：不加 noise 等于让 model 见到 "clean metric latent"，diffusion model 没见过这种 input distribution，崩了。
3. **Setting C vs D**: 完美对称——去掉 SDS 就只剩 metric（sharpness 崩），去掉 recons 就只剩 sharpness（metric 崩）。证明两者缺一不可。
4. **Setting F (Marigold teacher) vs Ours (Lotus teacher)**: Marigold 的 metric accuracy 略好 ($\delta_1=0.973$ vs $0.973$，A.Rel 0.058 vs 0.060)，但 completion 差很多 (84.7 vs 36.4)。作者选 Lotus 是因为 boundary quality 更重要（这是 sharpening 的 selling point）。
5. **Setting G vs H**: EMA teacher > Frozen teacher，证明 self-distillation 的必要性。

---

## 7. 一些 paper 没明说但我觉得重要的点

### 7.1 为什么不用 GT depth？

Paper 强调 "ground-truth-free"，但仔细想想，UniDepth 和 Lotus 本身都是 trained on large GT datasets 的。所以严格说不是 "label-free"，而是 "no additional label needed for the refinement stage"。这个 distinction 重要——SharpDepth 是 leveraging existing foundation models 的 prior，而不是从零学起。

类比：这就像 LLM 的 RLHF——你不需要为下游任务标注新数据，但 base model 已经在 huge corpus 上 pretrained 了。

### 7.2 Limitation: bounded by base models

从 ETH3D 的实验看（UniDepth $\delta_1=0.25$, SharpDepth 0.23），SharpDepth 不能 fix base metric model 在某 domain 上的根本失败。它的天花板是 $\max(f_D, f_G)$ 的 union strength。

如果 base metric model 在 reflective surface / transparent object 上崩，SharpDepth 也救不回来。作者在 Booster dataset 上也只达到 0.28 $\delta_1$（和 UniDepth 持平）。

### 7.3 跟其他 refinement 方法对比

PatchRefiner [25] 是同期工作，思路类似但用 synthetic data + residual learning。它在 KITTI 上 OK ($\delta_1=0.79$) 但在 indoor (NYUv2 $\delta_1=0.01$) 完全崩——synthetic domain gap 太大。SharpDepth 用 real data + GT-free pipeline，generalization 好得多。

BetterDepth [55] 是另一个相关工作，refine affine-invariant depth，但仍然依赖 synthetic data。

### 7.4 Inference speed

虽然 paper 没强调，但因为 $G_\theta$ 是 one-step prediction（继承 Lotus 的 single-step design），inference 应该比 Marigold（50 steps）快很多。这点对 SLAM / AR 应用很关键。Paper 在 Section 9.1 用 MonoGS (Gaussian Splatting SLAM) 做了 downstream evaluation，PSNR 从 18.472 提升到 18.857，证明 sharper depth 真的能改善 3D reconstruction。

### 7.5 与 SfM / MVS 的关系

Paper 提到 thin structure（fences, poles, durian spikes, keyboard keycaps）的重建改善。这让我想到 classic MVS 在 thin structure 上也一直有问题（patch-match 在 thin structure 上很难 convergence）。如果 SharpDepth 能作为 MVS 的 initial depth prior，可能能大幅改善 MVS 在 thin structure 上的 failure mode。这个方向我觉得有 follow-up 价值。

### 7.6 Camera-aware extension

UniDepth 是 camera-aware 的（pseudo-spherical representation）。SharpDepth 继承了 UniDepth 的 camera prediction。如果在 metric SLAM 场景下，camera intrinsic 的微小误差会 propagate 到 depth 上。Paper 没讨论 camera-aware sharpening 是否能进一步改善——可能是个 future direction。

### 7.7 和 Gaussian Splatting 的天然契合

Section 9.1 的 MonoGS 实验很有意思。Gaussian Splatting 对 depth edge 很敏感（因为它用 Gaussian 来 represent scene，sharp edge 需要很多 small Gaussians）。SharpDepth 的 sharper depth → better Gaussian initialization → better novel view synthesis。这条线如果展开，可能能做 "depth-conditioned Gaussian Splatting" 的端到端 pipeline。

---

## 8. Reference & Further Reading

如果你想深挖：

- **Paper arXiv**: https://arxiv.org/abs/2501.01756 (SharpDepth)
- **UniDepth (base metric model)**: https://arxiv.org/abs/2403.18913 - https://github.com/cvlab-unibe/UniDepth
- **Lotus (base diffusion model + init)**: https://arxiv.org/abs/2409.18124 - https://github.com/EnVision-Research/Lotus
- **Marigold (diffusion depth prior)**: https://arxiv.org/abs/2312.02145 - https://github.com/prs-eth/Marigold
- **SwiftBrush (one-step distillation inspiration)**: https://arxiv.org/abs/2403.16931
- **DreamFusion (SDS original)**: https://arxiv.org/abs/2209.14988
- **PatchRefiner (concurrent refinement work)**: https://arxiv.org/abs/2410.02427
- **BetterDepth (another concurrent refiner)**: https://arxiv.org/abs/2407.17952
- **Depth Anything (affine-invariant SOTA)**: https://arxiv.org/abs/2401.10891 - https://github.com/DepthAnything/Depth-Anything-V2
- **Metric3D v2 (metric SOTA)**: https://arxiv.org/abs/2404.15506
- **MonoGS (downstream SLAM)**: https://arxiv.org/abs/2311.18200 - https://github.com/muskie82/MonoGS
- **BrushNet (inpainting inspiration)**: https://arxiv.org/abs/2403.06976
- **iBims-1 (DBE metric)**: https://arxiv.org/abs/1806.04021

---

## 9. 总结：intuition 的最后一击

Andrej，如果让我用一句话总结这篇 paper 的 essence：

> **SharpDepth 把 "metric depth" 和 "sharp depth" 视为两个独立 manifold 上的 sample，用 difference map 找出 disagreement 区域作为 "uncertain mask"，然后在 uncertain 区域用 SDS 把 metric sample 拉向 sharp manifold，在 confident 区域用 reconstruction loss 锁住 metric scale。**

这个 formulation 优雅在于：
1. 不需要新 GT（amortized supervision via model disagreement）
2. SDS 提供 cheap distillation（不用 backprop through U-Net）
3. Noise-aware gating 让 model 集中 capacity 在需要 sharpen 的地方
4. EMA self-teacher 让 training 动态进化

它本质上是个 **teacher-student distillation with dual teachers**：Lotus 教 sharpness，UniDepth 教 metric accuracy，difference map 是 student 自己 decide 谁更可信的 router。这个思想可以推广到很多其他 vision task 的 refinement（normal estimation, optical flow, segmentation boundary）。

如果让我提一个 follow-up idea：把 difference map 换成 explicit uncertainty estimation（比如 ensemble disagreement 或 evidential deep learning），可能能更 robust 地处理 "两个 model 都错" 的情况。目前 SharpDepth 假设 confident region 至少有一个 model 是对的，如果两个都错且错的类似，difference map 不会标记，sharpening 不会发生。这是 paper 没讨论的盲点。
