---
source_pdf: Let Denoising Generative Models Denoise.pdf
paper_sha256: 8c265e4adfd488ccbce035a53ad9055edddd02f5c0393738f54da583ee832b8f
processed_at: '2026-08-05T14:29:14-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话总结

Kaiming He 团队发现：**diffusion model 的神经网络到底该预测什么，这件事比大家想象的更重要**。在 high-dimensional pixel space 上，让网络直接预测 clean image（x-prediction）能 work，而预测 noise（ε-prediction）或 velocity（v-prediction）会彻底崩掉。这个发现让他们能用一个 plain ViT 直接在 raw pixels 上做 diffusion，不需要 tokenizer、不需要 pre-training，就能打到 SOTA。

## 背景：diffusion model 的"历史漂移"

咱们回顾一下 diffusion model 的历史。最早 Sohl-Dickstein 2015 年提出来的 idea 确实是 denoising——给网络看一张 noisy image，让它预测 clean image。这跟 classical image denoising（BM3D 那一脉）的直觉一致。

但 2020 年 DDPM 那篇 paper 做了一个 pivotal 的发现：**让网络预测 noise 而不是 clean image，效果会好很多**。这就是 ε-prediction，从那以后几乎成了整个领域的 default。后来 flow matching / rectified flow 这条线又引入了 v-prediction，预测的是 velocity $v = x - \epsilon$，一个 data 和 noise 的混合。

大家一直觉得这三种 prediction target 在数学上是等价的，只是 loss weighting 不同。Salimans & Ho 2022 年那篇 progressive distillation paper 系统分析过，结论是在 CIFAR-10 上差别不大。

**但 Li 和 He 说：等价是数学上的等价，实际操作中网络 capacity 有限，这三种 target 对网络的要求完全不一样。**

## 核心 insight：manifold assumption

这里有个很 deep 的 intuition。机器学习里有个 classical hypothesis 叫 **manifold assumption**：自然数据（比如 ImageNet 里的 image）虽然住在 196608 维的 pixel space 里（256×256×3），但实际只分布在一个 low-dimensional manifold 上。这个 manifold 的 intrinsic dimension 可能就几千维甚至更低。

关键来了：

- **clean image x**：住在 low-dimensional manifold 上
- **noise ε**：纯粹的 Gaussian noise，均匀填满整个 high-dimensional space
- **velocity v = x - ε**：也是 off-manifold 的，因为混了 Gaussian noise

所以当你让网络预测 x 时，网络只需要"找到 manifold"并输出 manifold 上的点，多余的高维信息可以丢掉。但当你让网络预测 ε 或 v 时，网络必须**完整地 preserve 全部 high-dimensional noise 信息**，这对网络 capacity 的要求完全不同。

这就像：让你描述一张照片里的猫，你只需要说"橘猫、坐在窗台上、阳光打在背上"——这是 low-dimensional 描述。但让你描述这张照片叠加的 random noise，你必须逐 pixel 报出每个 noise value，这是 full high-dimensional 信息。

## Toy experiment：最直观的证据

Paper 里有个特别 beautiful 的 toy experiment 来验证这个 intuition。

设定：真实的 underlying data 只有 2 维（d=2），但通过一个 random orthogonal projection matrix 埋到 D 维空间里。网络不知道这个 projection matrix，只看到 D 维输入。网络是一个 5-layer ReLU MLP，hidden layer 只有 256 维。

然后变化 D ∈ {2, 8, 16, 512}：

- **D=2**：三种 prediction 都 work，因为数据本就 2 维
- **D=8**：三种都 work，但 ε/v-prediction 开始退化
- **D=16**：ε/v-prediction 开始 struggle
- **D=512**：ε/v-prediction **catastrophic failure**，但 x-prediction 依然 work！

这个结果特别 striking。256-dim hidden layer 处理 512-dim input，理论上信息是 lossy 的。但 x-prediction 能 work，因为真实数据就在 2-dim manifold 上，网络可以大胆丢弃 510 维的冗余信息。而 ε/v-prediction 必须在 512 维空间里 preserve 全部 noise，256-dim hidden layer 是 information bottleneck，扛不住。

**这就是整篇 paper 的 anchor insight。**

## 从 toy 到 real：JiT 架构

把这个 insight 搬到真实数据上。Li 和 He 提出的架构叫 **JiT（Just image Transformers）**，其实就是 plain ViT 直接 applied 到 pixel patches 上。

设定：
- ImageNet 256×256，patch size 16，每个 patch 是 16×16×3 = 768 维
- ImageNet 512×512，patch size 32，每个 patch 是 32×32×3 = 3072 维
- ImageNet 1024×1024，patch size 64，每个 patch 是 64×64×3 = 12288 维

注意这些 patch 维度远超 standard ViT 的 hidden size。ViT-B 是 768，ViT-L 是 1024，ViT-H 是 1280。如果 patch 维度是 12288，理论上你至少需要 hidden size ≥ 12288 才能完整 preserve 信息，但根本没人这么做。

**结果**：用 x-prediction，JiT-B（hidden size 768）处理 12288 维 patch 依然 work，FID 只有 4.82。而 ε-prediction 和 v-prediction 在 768 维 patch 上就 catastrophic fail 了，FID 飙到几百。

这个对比非常 dramatic。同样的网络、同样的 loss space、同样的 training recipe，只是 prediction target 不同，FID 能差 40 倍。

## 九种组合的 systematic ablation

Paper 里 Table 1 列出了 prediction space × loss space 的 9 种组合。数学上这 9 种都能互相转换，但实际表现天差地别。

JiT-B/16 on ImageNet 256×256（patch dim = 768 = hidden size）：

| | x-pred | ε-pred | v-pred |
|---|---|---|---|
| x-loss | 10.14 | 379.21 | 107.55 |
| ε-loss | 10.45 | 394.58 | 126.88 |
| v-loss | **8.62** | 372.38 | 96.53 |

x-prediction 全部 work（绿色），ε/v-prediction 全部 catastrophic fail（FID > 90）。

但在 ImageNet 64×64 上，patch 只有 48 维，远小于 hidden size 768，9 种组合差别就很小了（FID 都在 3-6 之间）。这就是为什么之前的 latent diffusion 没暴露这个问题——latent token 通常只有 4 或 8 维。

## 最 surprising 的发现：bottleneck 反而有益

这个发现真的违反直觉。作者把 patch embedding 从一层 linear 改成两层 linear，中间加一个 bottleneck dimension d'。

patch 本身是 768 维，bottleneck d' 从 768 一直减到 16。按理说 d' 越小信息越 lossy，效果应该越差。

**实际结果**：d' 在 32 到 512 之间效果反而比 d' = 768 更好，提升约 1.3 FID。即使 d' = 16（把 768 维压到 16 维再展开）依然能 work。

这完全印证了 manifold assumption。真实信息本来就在 low-dimensional manifold 上，强制 bottleneck 帮助网络聚焦在 manifold 上，反而减少了噪声干扰。这跟 classical autoencoder 的哲学完全一致——bottleneck autoencoder 就是为了学 low-dimensional representation。

**Under-capacity 不是 bug，是 feature。**

## 最终算法

非常简洁：

**Training**：
1. Sample t, sample noise ε
2. Compute noisy sample $z_t = t \cdot x + (1-t) \cdot \epsilon$
3. Network预测 clean image：`x_pred = net(z_t, t)`
4. 从 x_pred 推导 velocity：`v_pred = (x_pred - z_t) / (1-t)`
5. Compute v-loss against target $v = (x - z_t)/(1-t)$

**Sampling**：标准 ODE solver（50-step Heun），每步：
1. `x_pred = net(z, t)`
2. `v_pred = (x_pred - z) / (1-t)`
3. `z_next = z + (t_next - t) * v_pred`

注意 training 时 network 直接输出 x_pred，但 loss 在 v-space 计算。这对应 Table 1 的 (3)(a) 组合：x-prediction + v-loss。作者发现 v-loss 比 x-loss 略好（8.62 vs 10.14），因为 v-loss 在不同 t 上有更好的 implicit weighting。

## 结果有多强

ImageNet 256×256，FID-50K：

| Method | FID | Pre-training | Gflops |
|---|---|---|---|
| DiT-XL/2 | 2.27 | SD-VAE | 119 |
| REPA + SiT-XL/2 | 1.42 | SD-VAE + DINOv2 | 119 |
| LightningDiT-XL/2 | 1.26 | SD-VAE + DINOv2 | 119 |
| RAE + DiTDH-XL/2 | 1.13 | RAE + DINOv2 | 146 |
| ADM-G (pixel) | 4.59 | 无 | 1120 |
| PixelFlow XL/4 (pixel) | 1.98 | 无 | 2909 |
| PixNerd XL/16 (pixel) | 2.15 | DINOv2 | 134 |
| **JiT-G/16** | **1.82** | **无** | **383** |

JiT-G 以 1.82 FID 跻身最强 pixel-space diffusion，而且**完全 self-contained**——不需要 tokenizer、不需要 DINOv2、不需要 adversarial loss、不需要 perceptual loss。Gflops 远低于 PixelFlow（383 vs 2909）和 ADM-G（383 vs 1120）。

ImageNet 512×512 上 JiT-G/32 达到 1.78 FID，甚至超过了 256 分辨率，而且 compute 几乎不变（因为 sequence length 相同，只是 patch 维度从 768 变 3072）。

## 为什么以前没人发现这个

几个原因：

1. **DDPM 当初发现 ε-prediction 比 x-prediction 好，是在 U-Net + CIFAR-10 设定下**。那个 regime 里 patch 维度小，9 种组合差别不大，ε-prediction 可能因为 optimization 原因略胜。但这个结论被 over-generalize 了。

2. **Latent diffusion 兴起后，大家都在 low-dim latent 上做 diffusion**，latent token 通常 4 或 8 维，远小于网络 hidden size，所以 ε/v-prediction 的 capacity 问题被"隐藏"了。但这是 hide 而不是 solve——你需要一个 pre-trained autoencoder（通常还要 adversarial + perceptual loss），破坏了 self-contained 性质。

3. **Pixel-space diffusion 的工作（SiD2, PixelFlow, PixNerd）都用特殊架构**：hierarchical patches、NeRF head、representation alignment 等。这些设计实际上也在绕开 high-dimensional prediction 的问题。

4. **最新的工作（REPA, LightningDiT, RAE）都依赖 DINOv2 self-supervised pre-training**，这是另一个 "hidden" 的 capacity 补偿机制。

Li 和 He 的贡献是**回到 first principle**：直接在 raw pixels 上、用 plain ViT、用最 minimal 的 design，系统性地证明 prediction target 的选择才是核心问题。

## 更深层的 philosophical 启示

我觉得这篇 paper 最 deep 的启示是：**神经网络的 capacity 是有限的，我们应该让网络把 capacity 用在建模数据上，而不是建模噪声上**。

这听起来像废话，但 diffusion model 发展了十年，这个 intuition 一直被 mathematical equivalence 的论述掩盖了。大家觉得"反正数学上等价，随便选 prediction target"，忽略了 network 是 finite-capacity 的物理实体。

从 representation learning 的视角看，这其实回归到了 DAE（Denoising Autoencoder, Vincent et al. 2008）的初心：DAE 当年就是为了 manifold learning 而 predict clean data，只不过后来 score-based diffusion 把 target 换成了 score function（本质上预测 noise），这个"漂移"在 high-dimensional regime 下暴露了问题。

另一个启示是 **architecture generality 的价值**。JiT 能直接受益于 LLM 领域的 advances（SwiGLU, RMSNorm, RoPE, qk-norm），因为 architecture 和 task 解耦了。这对其他 domain（protein, molecule, weather）很有启发——这些领域 tokenizer 难以设计，如果能在 raw data 上直接做 diffusion + Transformer，方法论就能直接迁移。

## 我的思考

读完这篇 paper 我有几个联想：

**1. 这跟 Kaiming He 之前的工作线是一脉相承的。** MAR（NeurIPS 2024）探索 autoregressive image generation without VQ，FractalMAR 探索 fractal generative models，JiT 探索 diffusion on raw pixels。共同主题是"self-contained generative models on raw natural data"，尽量减少 domain-specific inductive bias。

**2. Pre-conditioner 那个分析很有意思。** EDM 的 pre-conditioner $x_\theta = c_{\text{skip}} \cdot z_t + c_{\text{out}} \cdot \text{net}_\theta$ 实际上让 network 输出的是 mixed quantity，不是 pure x。Table 10 显示 EDM-style pre-conditioner 也 catastrophic fail（FID 35.49 vs pure x-prediction 8.62）。这说明 pre-conditioner 在 low-dim 设定下好用（因为数学上更稳定），但在 high-dim 设定下反而是阻碍。

**3. In-context class conditioning 那个细节很有趣。** 32 个相同 class token（不同 positional embedding）从第 4/8/10 个 block 开始 prepend，而不是从输入层。这暗示 class information 在中间层注入比在输入层注入更有效，可能是因为早期 layer 需要先建立 local feature representation，class conditioning 在更 abstract 的 level 更有用。

**4. Cross-resolution generation 的结果值得深思。** JiT-G/32@512 downsample 到 256 是 1.84 FID，几乎等于专门训练的 JiT-G/16@256 的 1.82。但反过来 upsample 就差很多（2.45 vs 1.78）。这说明高分辨率模型学到了更丰富的信息，只是 FID 指标在 256 分辨率上饱和了。

**5. Bottleneck 的发现可能对 understanding diffusion 有更深的含义。** 如果 network 宽度可以远小于 patch 维度还能 work，那 diffusion model 的 "effective capacity" 到底需要多大？这可能跟最近关于 scaling laws for diffusion models 的工作有关。

## Reference Links

- **本 paper** (JiT): https://arxiv.org/abs/2509.28953 (估计 arXiv ID，需查证)
- **DDPM** (Ho et al. 2020): https://arxiv.org/abs/2006.11239
- **EDM** (Karras et al. 2022): https://arxiv.org/abs/2206.00364
- **Flow Matching** (Lipman et al. 2023): https://arxiv.org/abs/2210.02747
- **Rectified Flow** (Liu et al. 2023): https://arxiv.org/abs/2209.14530
- **v-prediction** (Salimans & Ho 2022): https://arxiv.org/abs/2202.00512
- **DiT** (Peebles & Xie 2023): https://arxiv.org/abs/2212.09748
- **SiT** (Ma et al. 2024): https://arxiv.org/abs/2401.08728
- **Latent Diffusion** (Rombach et al. 2022): https://arxiv.org/abs/2112.10752
- **ViT** (Dosovitskiy et al. 2021): https://arxiv.org/abs/2010.11929
- **DINOv2** (Oquab et al. 2023): https://arxiv.org/abs/2304.07193
- **REPA** (Yu et al. 2024): https://arxiv.org/abs/2410.06940
- **MAR** (Li et al. 2024): https://arxiv.org/abs/2406.11838
- **DAE** (Vincent et al. 2008): https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf
- **Manifold assumption textbook** (Chapelle et al. 2006): Semi-Supervised Learning, MIT Press
- **SiD2** (Hoogeboom et al. 2025): https://arxiv.org/abs/2412.13237
- **PixelFlow** (Chen et al. 2025): https://arxiv.org/abs/2504.07963
- **PixNerd** (Wang et al. 2025): https://arxiv.org/abs/2507.23268
- **Score-based SDE** (Song et al. 2021): https://arxiv.org/abs/2011.13456
- **ADM** (Dhariwal & Nichol 2021): https://arxiv.org/abs/2105.05233

---

总结一句人话：**让 diffusion model 的网络专心预测它该预测的东西——干净的图像，不要让它去预测噪声。因为图像信息是低维的、好学的，噪声信息是高维的、难学的。这么一改，plain ViT 就能直接在 raw pixels 上 work，不需要任何 tokenizer 或 pre-training，效果还特别好。** 这是对 diffusion model 十年发展的一次 elegant 的"回归初心"。

---

# Back to Basics: Let Denoising Generative Models Denoise 深度解析

## 1. Paper 核心洞察

Tianhong Li 和 Kaiming He 的这篇 paper 提出了一个看似简单但被忽视很久的问题:**今天的 diffusion models 实际上并不做"classical denoising"**。DDPM 以来的主流做法是让 network 预测 noise (ε-prediction) 或者 flow velocity (v-prediction),而直接预测 clean image (x-prediction) 反而被认为效果差、被遗弃。

Paper 的核心论点建立在 **Manifold Assumption** 之上:自然数据位于高维 pixel space 中的一个低维 manifold 上,而 noise ε 或 velocity v = x - ε 则 inherently 分布在整个高维空间中。这两者在 information-theoretic 意义上有本质区别:

- **预测 x**:网络只需保留 low-dimensional manifold 上的信息,过滤掉 noise,可以用 under-capacity 的网络实现
- **预测 ε 或 v**:网络需要 preserve 全部 high-dimensional noise information,对 capacity 要求极高

## 2. 数学形式化:三种 prediction space 与九种组合

### 2.1 Flow formulation 基础

给定 clean data $\mathbf{x} \sim p_{\text{data}}(\mathbf{x})$ 和 noise $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$,采用 linear schedule:

$$z_t = t\mathbf{x} + (1-t)\boldsymbol{\epsilon}$$

其中:
- $t \in [0,1]$ 是 time parameter,$t=0$ 对应 pure noise,$t=1$ 对应 clean data
- $z_t$ 是 noisy sample
- 系数 $a_t = t$, $b_t = 1-t$

Flow velocity 定义为 $z_t$ 对 $t$ 的导数:

$$\mathbf{v} = \frac{dz_t}{dt} = \mathbf{x} - \boldsymbol{\epsilon}$$

Training objective(v-loss):

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}, \boldsymbol{\epsilon}} \|\mathbf{v}_\theta(\mathbf{z}_t, t) - \mathbf{v}\|^2$$

Sampling 通过求解 ODE:

$$\frac{dz_t}{dt} = \mathbf{v}_\theta(\mathbf{z}_t, t)$$

从 $z_0 \sim p_{\text{noise}}$ 积分到 $t=1$。

### 2.2 九种组合的关键 Table 1

Paper 中最关键的 Table 1 列出了 prediction space × loss space 的 9 种组合。给定三个未知量 {x, ε, v} 和一个 network output,需要两个额外的 constraint 来确定所有未知量,这两个 constraint 就是 Eq.(1) 和 Eq.(2)。

以 x-prediction 为例,求解方程组:

$$\begin{cases} 
\mathbf{x}_\theta = \text{net}_\theta \\ 
z_t = t\mathbf{x}_\theta + (1-t)\boldsymbol{\epsilon}_\theta \\ 
\mathbf{v}_\theta = \mathbf{x}_\theta - \boldsymbol{\epsilon}_\theta 
\end{cases}$$

得到:
$$\boldsymbol{\epsilon}_\theta = \frac{z_t - t\mathbf{x}_\theta}{1-t}, \quad \mathbf{v}_\theta = \frac{\mathbf{x}_\theta - z_t}{1-t}$$

注意 $1-t$ 在分母,所以当 $t \to 1$(接近 clean data)时这个变换会 blow up,paper 中用 clip(默认 0.05)来防止 zero division。

**关键 insight**:虽然这 9 种组合在数学上都能互相转换,但当 network 的 capacity 有限时,它们的实际表现截然不同。Loss weighting alone 不足以解释这种差异。

## 3. Toy Experiment:验证 manifold 假设

这个 toy experiment 非常 beautiful 地验证了核心论点。设定:
- 真实数据是 d-dimensional(d=2)
- 通过一个固定的 column-orthogonal projection matrix $P \in \mathbb{R}^{D \times d}$ 嵌入到 D-dimensional 空间
- $P^\top P = I_{d \times d}$
- 观测数据 $\mathbf{x} = P\hat{\mathbf{x}} \in \mathbb{R}^D$
- 模型不知道 $P$,只能看到 D-dimensional 输入
- 用 5-layer ReLU MLP,256-dim hidden units

测试 $D \in \{2, 8, 16, 512\}$,d=2 固定。

**结果**(Fig. 2):
- D=2:三种 prediction 都能 work
- D=8:三种都能 work,但 ε/v-prediction 开始退化
- D=16:ε/v-prediction 开始 struggle
- D=512:ε/v-prediction **catastrophic failure**,但 x-prediction 仍然 work

**为什么 x-prediction 能在 under-complete 的网络(256-dim hidden)上 work?** 因为真实的 underlying data 只在 d=2 的 manifold 上,网络可以"丢弃"多余的 D-2 个维度的信息,只要它能找到这个 low-dimensional manifold。而 ε/v-prediction 需要在 D=512 维空间里 preserve 全部 noise 信息,256-dim 的 hidden layer 是 information bottleneck,无法做到。

这个 toy experiment 是整个 paper 的 intuition anchor。

## 4. JiT Architecture:Just image Transformers

### 4.1 架构设计

JiT 就是 plain ViT 直接应用于 pixel patches:
- 输入:$H \times W \times 3$ image
- 分成 non-overlapping patches,size $p \times p$
- 每个 patch 是 $p \times p \times 3$ dim 的 vector
- Linear embedding + positional embedding
- Stack of Transformer blocks(用 adaLN-Zero 做 conditioning)
- Linear predictor 输出每个 token 回到 $p \times p \times 3$ dim

**关键设定**:
- JiT/16 on 256×256:patch dim = 768 (16×16×3)
- JiT/32 on 512×512:patch dim = 3072 (32×32×3)
- JiT/64 on 1024×1024:patch dim = 12288 (64×64×3)

这些 patch 维度远超 standard ViT-B 的 hidden size(768),也超过 ViT-L(1024)和 ViT-H(1280)。在这种 high-dimensional patch 设定下,ε/v-prediction 会 catastrophic fail,但 x-prediction 能 work。

### 4.2 Algorithm 伪代码

**Training step**(Algorithm 1):
```python
t = sample_t()
e = randn_like(x)
z = t * x + (1-t) * e
v = (x - z) / (1-t)        # target velocity
x_pred = net(z, t)         # network predicts clean image directly
v_pred = (x_pred - z) / (1-t)  # derive velocity from x_pred
loss = l2_loss(v - v_pred)     # v-loss with x-prediction
```

**Sampling step**(Algorithm 2, Euler):
```python
x_pred = net(z, t)
v_pred = (x_pred - z) / (1-t)
z_next = z + (t_next - t) * v_pred
```

最终采用 x-prediction + v-loss 的组合,对应 Table 1 的 (3)(a)。

## 5. 核心实验结果

### 5.1 九种组合在 ImageNet 上的对比

**Table 2(a): ImageNet 256×256, JiT-B/16** (patch dim = 768 = hidden size)

| | x-pred | ε-pred | v-pred |
|---|---|---|---|
| x-loss | 10.14 | 379.21 | 107.55 |
| ε-loss | 10.45 | 394.58 | 126.88 |
| v-loss | **8.62** | 372.38 | 96.53 |

x-prediction 在所有三个 loss space 都 work(绿色),ε/v-prediction 全部 catastrophic fail(红色,FID > 90)。

**Table 2(b): ImageNet 64×64, JiT-B/4** (patch dim = 48 << hidden size 768)

| | x-pred | ε-pred | v-pred |
|---|---|---|---|
| x-loss | 5.76 | 6.20 | 6.12 |
| ε-loss | 3.56 | 4.02 | 3.76 |
| v-loss | 3.55 | 3.63 | 3.46 |

当 patch dim 远小于 hidden size 时,所有 9 种组合都 work reasonably well。这说明之前的 latent diffusion models 因为 input dimensionality 小(典型 4 或 8 dim per token),没有暴露这个问题。

### 5.2 Noise-level shift 不够

Table 3:通过调整 logit-normal distribution 的 μ 来 shift noise level:

| μ (t-shift) | x-pred | ε-pred | v-pred |
|---|---|---|---|
| 0.0 (lower noise) | 14.44 | 464.25 | 120.03 |
| -0.4 | 9.79 | 372.91 | 109.93 |
| -0.8 (higher noise) | 8.62 | 372.36 | 96.53 |
| -1.2 | 8.99 | 355.25 | 106.85 |

提高 noise level 对 x-prediction 有帮助(从 14.44 降到 8.62),但无法挽救 ε/v-prediction。说明问题不在 noise schedule,而在 network capacity 与 prediction target 的根本矛盾。

### 5.3 Bottleneck 反而有益!这是最 surprising 的发现

Fig. 4:把 patch embedding 改成 low-rank 两层 linear,中间有一个 bottleneck dimension d'。

JiT-B/16 patch dim = 768:
- d' = 768(no bottleneck):FID ~7.5
- d' = 32:FID 略好
- d' = 16:仍然能 work
- 最优 d' 在 32-512 之间,提升约 1.3 FID

**这完全违反直觉**:patch 是 768 维,你把它压缩到 16 维再展开,信息上 lossy,但效果反而更好。这恰恰印证了 manifold assumption——真实信息本来就在 low-dimensional manifold 上,强制 bottleneck 帮助网络找到这个 manifold。这是 classical manifold learning 中常见的观察(autoencoder 的 bottleneck design)。

### 5.4 "Just Advanced" Transformers

Table 4:加入 general-purpose LLM 改进:
- Baseline(SwiGLU, RMSNorm):7.48(6.32)
- + RoPE, qk-norm:6.69(5.44)
- + in-context class tokens(32 个):5.49(4.37) for JiT-B, 3.39(2.79) for JiT-L

括号内是 with CFG interval。in-context conditioning 是把 32 个相同 class token(不同 positional embedding)prepend 到第 4/8/10 个 block 开始,提升约 1.2 FID。

### 5.5 高分辨率生成

Table 5:相同 sequence length(16×16 = 256 tokens),不同分辨率:

| Resolution | Model | Patch dim | Params | Gflops | FID |
|---|---|---|---|---|---|
| 256×256 | JiT-B/16 | 768 | 131M | 25 | 4.37 |
| 512×512 | JiT-B/32 | 3072 | 133M | 26 | 4.64 |
| 1024×1024 | JiT-B/64 | 12288 | 141M | 30 | 4.82 |

**惊人结果**:patch dim 从 768 暴增到 12288(16×),FID 只是从 4.37 退化到 4.82,而 params 和 flops 几乎不变。这说明 x-prediction 让网络 design 完全 decouple 于 observed dimensionality。

对 512 和 1024,rescale noise scale 2× 和 4× 以维持 SNR。

### 5.6 Scalability

Table 6:ImageNet 256 和 512 上四种 model size:

| | 256 200-ep | 256 600-ep | | 512 200-ep | 512 600-ep |
|---|---|---|---|---|---|
| JiT-B | 4.37 | 3.66 | JiT-B/32 | 4.64 | 4.02 |
| JiT-L | 2.79 | 2.36 | JiT-L/32 | 3.06 | 2.53 |
| JiT-H | 2.29 | 1.86 | JiT-H/32 | 2.51 | 1.94 |
| JiT-G | 2.15 | 1.82 | JiT-G/32 | 2.11 | **1.78** |

JiT-G 在 512 分辨率上甚至比 256 还好(1.78 < 1.82)!作者解释:大模型在 ImageNet 上容易 overfit,而 512 分辨率任务更难,更不易 overfit。

## 6. 与其他方法的对比

### 6.1 ImageNet 256×256 (Table 7)

**Latent-space diffusion**(需要 pre-trained tokenizer):
- DiT-XL/2:2.27 FID(SD-VAE tokenizer)
- SiT-XL/2:2.06 FID
- REPA + SiT-XL/2:1.42(DINOv2 self-supervised pre-training)
- LightningDiT-XL/2:1.26(DINOv2)
- RAE + DiTDH-XL/2:1.13(DINOv2 + RAE tokenizer)

**Pixel-space diffusion**(无 pre-training):
- ADM-G:4.59(554M params,1120 Gflops,U-Net)
- SiD2, UViT/1:1.38(N/A params,653 Gflops)
- PixelFlow XL/4:1.98(677M,2909 Gflops,multi-scale)
- PixNerd XL/16:2.15(700M,需要 DINOv2)

**JiT**(完全 self-contained,无任何 pre-training):
- JiT-B/16:3.66(131M,25 Gflops)
- JiT-L/16:2.36(459M,88 Gflops)
- JiT-H/16:1.86(953M,182 Gflops)
- **JiT-G/16:1.82(2B params,383 Gflops)**

JiT-G 以 1.82 FID 跻身最强 pixel-space diffusion 之列,而且完全不依赖任何 tokenizer 或 self-supervised pre-training。Gflops 远低于 PixelFlow(383 vs 2909)和 ADM-G(383 vs 1120)。

### 6.2 ImageNet 512×512 (Table 8)

JiT 在 512 分辨率上表现更亮眼:
- JiT-G/32:1.78 FID(2B params,384 Gflops)
- 超过了所有 latent diffusion 方法,包括需要 DINOv2 的 DDT-XL/2(1.28,但需要 525 Gflops)
- 在 pixel-space diffusion 中,SiD2 UViT/2 是 1.48(653 Gflops)

## 7. Pre-conditioner 分析

EDM [Karras et al. 2022] 用 pre-conditioner 包装 network output:

$$\mathbf{x}_\theta(z_t, t) = c_{\text{skip}} \cdot z_t + c_{\text{out}} \cdot \text{net}_\theta(z_t, t)$$

除非 $c_{\text{skip}} \equiv 0$,否则 network 的 direct output 就不是 pure x。Paper 中 Table 10 测试了:
- Pure x-prediction:8.62 FID
- EDM-style pre-conditioner:35.49
- Linear pre-conditioner($c_{\text{skip}}=t, c_{\text{out}}=1-t$):46.25

Pre-conditioner 会 catastrophic fail,因为它只在 $t \to 0$ 时才逼近 x-prediction。这进一步证明"直接预测 x"是关键。

EDM 的 pre-conditioner 中:
- $c_{\text{skip}} = \frac{1}{t} \cdot \frac{\sigma_{\text{data}}^2}{\sigma_{\text{data}}^2 + \sigma_t^2}$
- $c_{\text{out}} = \frac{\sigma_{\text{data}} \sigma_t}{\sqrt{\sigma_{\text{data}}^2 + \sigma_t^2}}$
- $\sigma_t = \frac{1-t}{t}$ (paper 中的 VP schedule 转换)
- $\sigma_{\text{data}} = 0.5$ (data standard deviation)

## 8. 训练 Loss 曲线与 Denoised Images 对比

Fig. 7:同样的 v-loss,但分别用 x-prediction 和 v-prediction:
- x-prediction 的 training loss 比 v-prediction 低约 25%
- v-prediction 的 denoised images 有明显 artifacts
- 单步 denoising 的 error 在多步 ODE solver 中会 accumulate,导致最终 catastrophic failure

这个观察很重要:虽然 v-prediction 在 v-space 上是 "native" parameterization,但实际 training loss 更高,因为 network 在 high-dimensional space 上难以 predict v。

## 9. Cross-resolution Generation (Table 12)

- JiT-G/32@512 直接 downsample 到 256:FID@256 = 1.84(vs 专门训练的 JiT-G/16@256 = 1.82)
- JiT-G/16@256 upsample 到 512:FID@512 = 2.45(vs 专门训练的 JiT-G/32@512 = 1.78)

512 model downsample 到 256 几乎无损,但 256 model upsample 到 512 会丢失 high-frequency detail。这暗示高分辨率模型学到的东西更丰富。

## 10. Intuition 总结

这篇 paper 给我的几个核心 intuition:

**1. Prediction target 选择是 information bottleneck 问题,不是数学等价问题。** 当 network capacity 无限时,9 种组合等价;但 capacity 有限时,选择预测什么决定了 network 能否 succeed。预测 on-manifold 的 x,网络可以"丢弃"off-manifold 的 noise 信息;预测 off-manifold 的 ε/v,网络必须 preserve 整个 high-dimensional noise,对 capacity 要求极高。

**2. Under-capacity 可以成为 feature,不是 bug。** Bottleneck 反而有益的发现非常 striking。256-dim hidden layer 处理 12288-dim patch 能 work,是因为真实信息本就在 low-dimensional manifold 上。这与 classical autoencoder 的 manifold learning 视角一致。

**3. Tokenizer 是"隐藏"问题,不是"解决"问题。** Latent diffusion 通过先压缩到 low-dim latent 再做 diffusion,实际上是绕开了 high-dimensional prediction 的问题。但这需要 pre-trained autoencoder(通常还要 adversarial + perceptual loss),破坏了 self-contained 性质。JiT 直接在 pixel 上做,证明这个问题可以通过正确选择 prediction target 来根本解决。

**4. Architecture generality 的价值。** JiT 能直接受益于 LLM 领域的 advances(SwiGLU, RMSNorm, RoPE, qk-norm),因为 architecture 与 task 解耦。这对其他 domain(protein, molecule, weather)很有启发,这些领域 tokenizer 难以设计。

**5. 从 DDPM 的历史看,ε-prediction 当初被发现比 x-prediction 好,可能是因为当时的 U-Net 架构和 low-resolution 设定掩盖了这个问题。** 在 CIFAR-10 这种 low-dimensional 设定下(Table 2(b) 和 Salimans & Ho 的工作),9 种组合差别不大。问题只在 high-dimensional patch 设定下才暴露出来。

## 11. 实现细节要点

**Time sampling**:logit-normal distribution,logit(t) ~ N(μ, σ²),默认 μ=-0.8, σ=0.8。这意味着 t 更倾向于采样在较低值(更多 noise),对 high-resolution 有帮助。

**High resolution noise scaling**:512 用 2× noise scale,1024 用 4×,以维持 SNR。即 $\epsilon \sim \mathcal{N}(0, s^2 \mathbf{I})$,$s$ 与 image_size/256 成正比。

**EMA**:维护多个 decay(0.9996, 0.9998, 0.9999),inference 时选最优。这是合理的工程实践,因为不同 model size 的最优 EMA 不同。

**CFG**:scale sweep 范围 [1.0, 4.0],CFG interval [0.1, 1](只在后期高 t 时应用 CFG)。

**Dropout**:只在 JiT-H 和 JiT-G 上用,applied to middle half of Transformer blocks,both attention 和 MLP。G 模型还需 early stopping(约 320 epochs)。

**In-context class conditioning**:32 个相同 class token(不同 PE),从第 4/8/10 个 block 开始 prepend,而不从一开始。这个细节很有趣,说明 class conditioning 在中间层注入比在输入层注入更好。

## 12. 思考与延伸

**与 DINO/MAE 的联系**:Paper 提到 DAE(Denoising Autoencoder)本来就是为了 manifold learning 而预测 clean data,这与 JiT 的哲学一致。但 score-based diffusion 的 score function 本质上是预测 noise(up to scaling),这是历史发展中一个"漂移"。

**为什么 ε-prediction 在传统 U-Net 上 work?** U-Net 的 dense convolution + long skip connections + over-complete channels 实际上提供了 high-capacity pathway 来 preserve noise information。JiT 的 plain ViT 没有 skip connection,完全靠 Transformer blocks 处理,所以对 prediction target 选择更敏感。

**潜在延伸**:
- 其他 domain(protein design, molecule generation)通常数据本就 high-dimensional 且 manifold 结构强,JiT 范式可能直接适用
- Video generation 中每个 frame patch 也 high-dimensional,x-prediction 可能有类似优势
- Audio generation 也是 high-dimensional raw data 的 candidate

**Limitation**:Paper 没有探索 text-to-image 这种 richer conditioning 场景,只在 class-conditional ImageNet 上验证。能否 scale 到 text-to-image billion-scale 数据集仍未验证。

## Reference Links

- Paper: https://arxiv.org/abs/2509.28953 (估计的 arXiv ID, 实际需要查证)
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- SiT (Ma et al.): https://arxiv.org/abs/2401.08728
- EDM (Karras et al.): https://arxiv.org/abs/2206.00364
- DDPM (Ho et al.): https://arxiv.org/abs/2006.11239
- Flow Matching (Lipman et al.): https://arxiv.org/abs/2210.02747
- Rectified Flow (Liu et al.): https://arxiv.org/abs/2209.14530
- Stable Diffusion / Latent Diffusion (Rombach et al.): https://arxiv.org/abs/2112.10752
- ViT (Dosovitskiy et al.): https://arxiv.org/abs/2010.11929
- v-prediction (Salimans & Ho): https://arxiv.org/abs/2202.00512
- SiD2 (Hoogeboom et al.): https://arxiv.org/abs/2412.13237
- PixelFlow (Chen et al.): https://arxiv.org/abs/2504.07963
- PixNerd (Wang et al.): https://arxiv.org/abs/2507.23268
- REPA (Yu et al.): https://arxiv.org/abs/2410.06940
- MAR (Li et al.): https://arxiv.org/abs/2406.11838
- FractalMAR (Li et al.): https://arxiv.org/abs/2502.17437
- JetFormer (Tschannen et al.): https://arxiv.org/abs/2411.19322
- DINOv2 (Oquab et al.): https://arxiv.org/abs/2304.07193
- REPA's DINOv2 alignment work
- Manifold learning textbook (Chapelle et al.): Semi-Supervised Learning, MIT Press, 2006
- DAE (Vincent et al.): https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf
- Denoising Score Matching (Vincent): https://www.iro.umontreal.ca/~vincentp/Publications/denoising_score_matching.pdf
- Score-based SDE (Song et al.): https://arxiv.org/abs/2011.13456
- LightningDiT: https://arxiv.org/abs/2502.00563 (估计)
- DDT: https://arxiv.org/abs/2504.05741
- RAE: https://arxiv.org/abs/2510.11690

**Kaiming He 和 Tianhong Li 的相关工作线**:
- MAR (NeurIPS 2024): autoregressive image generation without VQ
- FractalMAR: fractal generative models
- 这篇 JiT 是他们探索"self-contained generative models on raw data"系列的延续

总的来说,这篇 paper 用极简的设计(plain ViT on pixels + x-prediction)和清晰的论证(manifold assumption + toy experiment + 系统性 ablation)揭示了一个被 diffusion 社区长期忽视的问题:prediction target 的选择在 capacity-limited regime 下是决定性的,而 latent tokenizer 实际上是在掩盖而非解决这个问题。JiT-G 以 1.82/1.78 FID 在 ImageNet 256/512 上达到 SOTA pixel-space diffusion,且无需任何 pre-training,标志着"Diffusion + Transformer on raw data"范式的重要进展。
