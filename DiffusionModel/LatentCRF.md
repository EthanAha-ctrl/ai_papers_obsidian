---
source_pdf: LatentCRF.pdf
paper_sha256: 442f0eb53b8dc3bb6d94a32c67cdd64ae5a513b1c8cd67dc81bd449336e27e5d
processed_at: '2026-08-05T12:20:16-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

好，我把刚才那一堆公式翻译成大白话。

---

## 一句话总结

LDM生成图片最后那十几步其实没啥活干，就是修修图、磨磨皮。你用865M参数的U-Net干这活，就像开F1送外卖。作者搞了个130M的小CRF模块来替班，干得一样好，省33%时间。

---

## 他们观察到啥

跑LDM 50步DDIM，把每一步latent的variance画出来：

- **前20步**：latent还是乱码，variance接近1。这一段要凭空把"text说有个山有个湖"塞进latent，需要大model的representation power。
- **后10步**：山和湖的位置都定死了，latent variance很低，剩下就是"把山弄得更像山一点"。这一段U-Net其实是overkill。

Appendix A.1那张variance曲线就是整篇paper的motivation——**variance低意味着latent已经躺在natural image manifold附近了，剩下的refinement用inductive bias强的小model就够**。

---

## CRF是啥（说人话）

CRF就是一个能量函数，长这样：

**能量 = 别跑太远 + 邻居要协调 + 局部patch要像natural image**

三项分别：

1. **别跑太远**：denoised output别离input太远，差不多就是 $\|y-x\|^2$，一个anchor项。

2. **邻居要协调**：空间上挨着的latent，在一个"语义空间"里应该距离小。这个语义空间是text-conditioned的——prompt说"a face"，那眼睛和鼻子位置的latent在这个空间里就该近；prompt说"a landscape"，那sky和mountain位置的latent就该近。

3. **局部patch像natural image**：借鉴Field of Experts那套老东西——natural image的patch有特定统计规律，用一组learnable filter去"扫"latent patch，响应大说明像natural image、energy低。

三项加起来就是能量。inference就是minimize这个能量，找个最优的y。

---

## inference咋跑

就是5次循环，每次：
- 把latent跟邻居做一次message passing（conv）
- 过一个text-conditioned的compatibility MLP
- 过一组FoE filter（conv → ReLU → mirrored conv）
- 加上原始input做residual
- BatchNorm一下

5步就收敛，多了没变化（Appendix B实测）。

为啥是5步？mean-field inference本来是优化算法，作者把它展开成5层"RNN"，end-to-end可训。这是2015年CRFasRNN的老套路，只不过搬到latent space、加了text conditioning、加了FoE。

---

## 训练两步走

**第一步：学"natural image latent长啥样"**

给一张真实图的latent $z$，加噪声得 $\tilde{z}$，让CRF denoise回 $z$。纯MSE。

但这不够，latent space对微扰不敏感（Figure A.4，0.05 noise加进去pixel space完全看不出变化），纯MSE会blurry。所以加个adversarial loss——3层conv的小discriminator在latent space做real/fake判断，逼CRF保留structure。

**第二步：接上LDM**

第一步训出来的是个"通用latent denoiser"，不知道LDM在step 40吐出来的latent长啥样。第二步就distill一下：拿LDM在step 40的输出当input，step 50的最终输出当target，MSE拉一下。只跑10k iter就够。

**为啥不直接做第二步**？因为光有distillation target太sparse，CRF会学歪。先学general natural latent prior，再specialize到LDM的trajectory，稳。

---

## 推理pipeline长这样

```
跑31步LDM (40-step schedule) → 跑1次CRF → 跑2步LDM (50-step schedule)
```

最后那2步LDM是"touch up"，ablation里去掉它FID从11.58掉到11.74，不算大但有用。

**为啥前31步还是要LDM**？因为前31步还在high-variance regime，小CRF扛不住，需要大U-Net的representation power。CRF只在low-variance regime接班。

---

## 结果一句话

同step对比（都33步）：
- 光跑33步LDM：FID 12.78
- 31步LDM + CRF + 2步LDM：FID 11.58（比50步LDM的12.75还好）

加速33%，quality没掉还略涨。

---

## 最有意思的发现：diversity不掉

这个是真值得讲的。ADD那种temporal distillation，就算给它50步也丢了32%的diversity（Table 3）。LatentCRF 33步只掉4%，LatentCRF-L 33步掉0%。

**为啥distillation会丢diversity**？因为distillation的本质是让student学teacher的"deterministic trajectory"——给它一个noise，告诉它"你应该走这条路"。loss是MSE，student会往conditional mean塌缩，不同noise出来同一张图。

**为啥LatentCRF不丢**？因为它根本没去学teacher的trajectory。它只是学了个"natural image manifold上的denoiser"。noise的diversity在LDM前31步就已经inject了，CRF只是把manifold附近的latent"投影"到manifold上——每个noise对应不同的manifold点，自然保diversity。

这个insight其实挺深的：**distillation是学路径，structured replacement是学manifold**。后者天生不丢multi-modality。

---

## 为啥这paper有意思

temporal distillation那条路已经卷到SDXL-Turbo 1-4步、InstaFlow 1步，靠adversarial loss硬抗quality崩、diversity崩。LatentCRF开了一个qualitatively不同的方向：**承认diffusion不同阶段需要不同model class，后期交给inductive bias强的小model**。

这思路在video、3D、long-context generation里应该都有空间——那些场景后期refine阶段的budget占比更大，替班潜力更高。

---

## 但也有弱点

- man-made structure偶尔break lines（Appendix F）
- 20-step激进版虽然55%加速但有更多artifact
- 只在SD-class scale上测，没上SDXL/MJ scale
- FoE filter到底学到了啥visual pattern，paper没可视化，有点可惜

---

一句话：**别让大model干小model能干的活，用对inductive bias就能省一个数量级的compute，而且不会丢diversity。**

---

# LatentCRF 深度技术讲解

Andrej，这篇paper从Google Research和Stony Brook出来的工作（一作Kanchana Ranasinghe），把classical probabilistic graphical model的inductive bias重新引入到diffusion inference的最后几步，核心idea其实非常elegant——既然LDM后期迭代的latent variance极低（接近natural image manifold），那就可以用一个把natural image prior baked-in的轻量CRF模块来替换U-Net的最后一串steps。我从能量函数推导、mean-field inference、训练objectives、到schedule设计全讲一遍。

---

## 1. 高层intuition：为什么CRF能替换U-Net？

paper最informative的empirical observation藏在Appendix A（Figure A.1, A.3）。作者跑了50-step DDIM reverse diffusion，逐iteration记录latent的per-sample variance，发现：

- **早期iter (t<20)**: variance接近1（pure noise），latent熵高，需要大capacity model + 强text conditioning来注入semantic structure
- **后期iter (t>40)**: variance快速衰减到很低，latent已经"长得像"natural image latents，剩下的工作主要是photorealism的refinement

这个观察其实和Sander Dieleman在diffusion distillation里讨论的"mode collapse vs fidelity tradeoff"是同一个现象的不同侧写：https://sander.ai/2024/01/30/paradox.html

所以作者的赌注是：后期iter不需要U-Net那种大representation capacity，可以换成一个strong inductive bias、低capacity的模块——continuous CRF。CRF的优势在于它显式地encode了natural image latent的spatial + higher-order结构，把这些prior做成trainable filter banks和compatibility transforms，比U-Net的free-form容量小一个数量级。

---

## 2. 能量函数的构建（核心公式Eq. 7）

整个LatentCRF就是定义一个Gibbs distribution:

$$P(\mathbf{Y}=\mathbf{y}) = \frac{1}{Z}\exp(-E(\mathbf{y}))$$

其中energy function为：

$$E(\mathbf{y}|\mathbf{x}) = \sum_i \|\mathbf{y}_i - \mathbf{x}_i\|^2 + \sum_{i,j} W^s_{ij}\|\mathbf{W}(\mathbf{c})\mathbf{y}_i - \mathbf{W}(\mathbf{c})\mathbf{y}_j\|^2 + \sum_i\sum_m\big[-\log\phi(\mathbf{J}_m\odot\mathbf{y})\big]_i$$

### 变量逐一解释：

- **y**: 待求解的denoised latent，shape为 $\mathbb{R}^{h'\times w'\times d}$ (例如 $64\times64\times8$ for SD class)
- **x**: observed noisy latent（CRF的input，对应LDM在某个intermediate step $s$ 的output $z_s$）
- **y_i**: 第 $i$ 个spatial location的 $d$-dim latent vector，共 $n=h'w'$ 个
- **$W^s_{ij}$**: scalar spatial similarity weight，衡量location $i$ 和 $j$ 之间的"邻接强度"（learnable，但通常spatially-decaying kernel）
- **$\mathbf{W}(\mathbf{c})$**: text-conditioned compatibility matrix，把latent投到一个"语义兼容"空间——同一prompt下应该co-occur的parts在这个空间里距离小
- **$\mathbf{c}$**: text embedding（CLIP/T5）
- **$\mathbf{J}_m$**: 第 $m$ 个learnable FoE filter，共 $M$ 个
- **$\phi(\cdot)$**: 满足non-negative且monotonically increasing的pdf（关键选择见下）
- **$\odot$**: 2D convolution
- **$[\cdot]_i$**: 取conv output在location $i$ 的值

### 三个能量项的intuition

**Unary term** $\|\mathbf{y}_i - \mathbf{x}_i\|^2$：
来自forward process的Gaussian noise假设 $P(\mathbf{y}|\mathbf{x})\propto\exp(-\|\mathbf{y}-\mathbf{x}\|^2)$。它本质上是一个"data attachment"项——denoised结果不能跑得离observation太远。和score-based diffusion里 $\epsilon$-prediction的quadratic loss是同源的。

**Pairwise term** $W^s_{ij}\|\mathbf{W}(\mathbf{c})\mathbf{y}_i - \mathbf{W}(\mathbf{c})\mathbf{y}_j\|^2$：
这是CRF/RNN work（https://arxiv.org/abs/1502.03240）里的"compatibility transform"的text-conditioned版。原始CRFasRNN用 $W^T W$ 作为learned compatibility，这里把它改成 $\mathbf{W}(\mathbf{c})^T\mathbf{W}(\mathbf{c})$，让"哪些latent应该互相兼容"是prompt-dependent的。例如prompt="a face"时眼睛和鼻子位置的latent在这个空间里应该近，而prompt="a landscape"时同样的spatial位置应该是sky和mountain。

**Higher-order term (FoE)** $-\sum_m\log\phi(\mathbf{J}_m\odot\mathbf{y})$：
Field of Experts来自Roth & Black 2005（https://www.cs.toronto.edu/~pspeaks/dnndoku/FoE_cvpr2005.pdf）和Hinton的Product of Experts（https://www.cs.toronto.edu/~hinton/absps/nips00poe.pdf）。每个filter $\mathbf{J}_m$ 编码一个patch-level "expert"——某种natural image latent patch的pattern。整张图的patch probability是各expert的乘积（product），log-prob就是log求和。

注意FoE建模的是**patch-level统计**，这比pairwise capture更高-order的依赖关系（一个 $h''\times w''$ 的patch就是一个clique）。但FoE filter只能capture**局部**cliques，所以long-range pairwise term还是有必要的——这是为什么Eq. 7里两者共存。

---

## 3. $\phi$ 的关键trick（Eq. 10/15）

直接对 $-\log\phi(\mathbf{J}_m\odot\mathbf{y})$ 求导做mean-field update会得到一个包含 $\frac{\partial}{\partial y}\log\phi(y) \triangleq \omega(y)$ 的项。作者故意把 $\phi$ 选成：

$$\phi(y) = \begin{cases} e^{y^2/2} & y > 0 \\ \varepsilon & \text{otherwise}\end{cases}$$

这样:
$$\omega(y) = \frac{\partial\log\phi(y)}{\partial y} = \begin{cases} y & y > 0 \\ 0 & \text{otherwise}\end{cases} = \text{ReLU}(y)$$

这是一个非常聪明的"reverse engineering"：让inference的update rule里直接出现ReLU，整个CRF inference就完全是standard conv + matmul + ReLU stack，可以直接用JAX/PyTorch搭出来。这个trick的精神和CRFasRNN里把mean-field iteration展开成RNN step是一脉相承的。

$\varepsilon > 0$ 是为了避免log(0)。注意 $\phi$ 是monotonically increasing + non-negative，满足probability density的最低要求。

---

## 4. Mean-field Inference 推导

CRF inference需要 minimize $E(\mathbf{y}|\mathbf{x})$。对 $\mathbf{y}_i$ 求偏导并令为0（Eq. 8）：

$$\frac{\partial E}{\partial \mathbf{y}_i} = 2(\mathbf{y}_i - \mathbf{x}_i) + 2\sum_j W^s_{ij}\mathbf{W}(\mathbf{c})^T\mathbf{W}(\mathbf{c})(\mathbf{y}_i - \mathbf{y}_j) - \sum_m\big[\mathbf{J}_m^-\odot\omega(\mathbf{J}_m\odot\mathbf{y})\big]_i = 0$$

其中 $\mathbf{J}_m^-$ 是 $\mathbf{J}_m$ 的mirrored版（左右+上下翻转）——这是convolution transposed gradient的标准要求。

求解得到Eq. 9：

$$\mathbf{y}_i^{(t+1)} := \mathbf{K}^{-1}\bigg(\mathbf{x}_i + \mathbf{W}(\mathbf{c})^T\mathbf{W}(\mathbf{c})\sum_j W^s_{ij}\mathbf{y}_j^{(t)} + \frac{1}{2}\sum_m\big[\mathbf{J}_m^-\odot\omega(\mathbf{J}_m\odot\mathbf{y}^{(t)})\big]_i\bigg)$$

其中 $\mathbf{K} = \mathbf{I} + \mathbf{W}(\mathbf{c})^T\mathbf{W}(\mathbf{c})\sum_j W^s_{ij}$。

**作者的engineering修改** (Eq. 11)：

1. **Parallel update** 代替 sequential coordinate descent（收敛证明不成立，但empirical上几步就收敛，参考Krahenbuhl & Koltun 2011 https://arxiv.org/abs/1210.5644）
2. **Compatibility transform**从线性 $\mathbf{W}(\mathbf{c})^T\mathbf{W}(\mathbf{c})$ 换成 lightweight nonlinear NN $\psi_{\theta_C}$ —— capture更复杂的compatibility
3. **Normalization**换成learnable $\psi_N$（BatchNorm），训练更稳定
4. 把 $\mathbf{K}^{-1}$ 直接用 $\psi_N$ 替代，避免矩阵求逆

最终inference algorithm（Algorithm 1）每步是：

```
y ← x                              # init
y ← y + text_embedding             # condition
repeat num_iterations=5 times:
    ŷ_i = Σ_j W^s_{ij} y_j         # Message Passing (spatial conv)
    ŷ ← ψ_{θ_C}(ŷ, c)              # Compatibility Transform (text-conditioned MLP)
    ŷ ← ŷ + ξ_HO(y)                # Higher-Order (FoE conv stack)
    ỹ ← ŷ + x                      # Unary Addition (residual to input)
    y ← ψ_N(ỹ)                     # Normalization (BatchNorm)
return y
```

其中 $\xi_{H O}(\mathbf{y}) = \frac{1}{2}\sum_m\big[\mathbf{J}_m^-\odot\text{ReLU}(\mathbf{J}_m\odot\mathbf{y})\big]$，本质上是"conv → ReLU → mirrored conv"的FoE gradient。

Appendix B（Figure A.2）的ablation说明5 iterations就完全收敛（第5步和第10步生成的image像素差几乎为零）。这个num_iterations作为hyperparameter在训练和推理时都固定为5。

---

## 5. 两阶段训练（核心创新）

### Stage 1: Natural image latent prior（Section 3.4）

目标：让CRF学会"什么是natural image latent"的分布。

**Latent denoising loss**:
$$\mathcal{L}_{\text{NT}} = \|z - \mathcal{M}(\tilde{z})\|_2^2$$

其中 $\tilde{z} = \sqrt{\alpha}\cdot z + \sqrt{1-\alpha}\cdot n$，$n\sim\mathcal{N}(0,1)$，$\alpha$ 是noise ratio hyperparam。这是把CRF当成一个generic denoiser来训。

**Latent adversarial loss** $\mathcal{L}_{\text{NT-ADV}}$：
动机是latent space对微扰不敏感（Figure A.4展示0.01/0.05 noise在pixel space几乎没变化），所以纯MSE loss会让输出blurry、有artifact。Adversarial loss能capture spatial structure。

Discriminator是3层conv（64/64/128 channels），spectral norm + BatchNorm，pointwise conv投影到1 channel，per-spatial-location做binary classification（motivated by StyleGAN-T https://arxiv.org/abs/2304.14573 和 ADD https://arxiv.org/abs/2311.17042）。

非saturating sigmoid cross-entropy:
$$\mathcal{L}_{\text{SCE}}(a,t) = \max(a,0) - a\cdot t + \log(1+\exp(-\text{abs}(a)))$$

总loss:
$$\mathcal{L}_{\text{NT-F}} = \mathcal{L}_{\text{NT}} + \mathcal{L}_{\text{SCE}}(\mathcal{M}(\tilde{z}), 1)$$
$$\mathcal{L}_{\text{disc}} = \mathcal{L}_{\text{SCE}}(z, 1) + \mathcal{L}_{\text{SCE}}(\mathcal{M}(\tilde{z}), 0)$$

训练300k iterations，AdamW，cosine decay LR from 1e-3，5000 warmup，per-device batch 16，128 TPUv5e。

### Stage 2: LDM-aware distillation（Section 3.5）

目标：让CRF能"接上"LDM在某intermediate step $s$ 的output，并产出接近LDM最终step $f$ 的latent。

$$\mathcal{L}_{\text{DT}} = \|z_f - \mathcal{M}(z_s)\|_2^2$$

实验中 $s=40$, $f=50$（on 50-step DDIM schedule）。从Stage 1 checkpoint init，10k iterations，LR 1e-5，per-device batch 4。

这步是关键——Stage 1的CRF是"通用natural latent denoiser"，Stage 2把它specialize到"接住LDM step 40的输出"。

---

## 6. Inference pipeline设计（非常关键）

默认设置:
- 50-step DDIM baseline
- LatentCRF: **31 LDM steps (40-step schedule) → 1 CRF → 2 LDM steps (50-step schedule) = 33 effective steps**

为什么这么设计？作者的分析（Section 4 + Appendix A.3）:
- 前期LDM iter处理high-variance noise → 大model必要
- 后期LDM iter只是refine photorealism → 轻量CRF能替代
- 最后2 LDM steps做"细节touch up"——ablation (Table 4)显示去掉这2 steps FID从11.58掉到11.74（不算大但明显）
- 关键ablation: 把CRF去掉直接用20 LDM steps，FID从11.58掉到19.72（崩了8.14）

Table 4 ablation非常informative：
| Pre-CRF | CRF | Post-CRF | FID |
|---|---|---|---|
| 31 | √ | 2 | 11.58 |
| 31 | √ | 0 | 11.74 (-0.16) |
| 31 | × | 2 | 19.72 (-8.14) |
| 31 | × | 0 | 21.85 (-10.27) |

CRF贡献了大约8 FID——这是核心delta。

---

## 7. 实验数据表精读

### Table 1（主结果）

| Method | Iter | FID↓ | CLIP↑ | Vendi↑ | Speed(ms)↓ |
|---|---|---|---|---|---|
| LDM | 50 | 12.75 | 0.309 | 2.75 | 782.1 |
| LDM | 33 | 12.78 | 0.308 | 2.57 | 501.8 (+35.9%) |
| **LatentCRF** | 33 | **11.58** | 0.309 | 2.64 | 523.4 (+33.2%) |
| LDM-L | 50 | 12.23 | 0.312 | 2.82 | 2335 |
| LDM-L | 33 | 12.35 | 0.311 | 2.80 | 1528 (+34.6%) |
| **LatentCRF-L** | 33 | **11.27** | 0.311 | 2.82 | 1549 (+33.7%) |

关键读法：
1. 同step数对比：LDM-33 vs LatentCRF-33，FID从12.78 → 11.58（**好于**LDM-50的12.75！），CRF补回了"少17 step"的quality loss
2. LatentCRF-L完全retain Vendi (2.82 → 2.82, +0%)，LatentCRF base只掉4%
3. Speed 523ms vs 782ms baseline = 33.2%加速

### Table 2（vs prior distillation）

| Model | FID↓ | CLIP↑ |
|---|---|---|
| SnapFusion | 14.00 | 0.300 |
| BK-SDM-Base | 17.23 | 0.287 |
| InstaFlow (1 step) | 20.00 | 0.283 |
| Clockwork | 12.33 | 0.296 |
| LCM (8 steps) | 11.84 | 0.288 |
| ADD (4 steps) | 22.58 | 0.312 |
| **LatentCRF** | **11.58** | 0.309 |

LatentCRF的FID是这一堆里最好的之一，且没有用model compression（SnapFusion/BK-SDM是U-Net瘦身）或temporal distillation到极端step数（ADD 4-step/InstaFlow 1-step）。

### Table 3（diversity retention）

| Method | Steps | Vendi ↑ |
|---|---|---|
| SDXL (teacher) | 50 | 3.01 |
| ADD | 1 | 1.55 (-48.5%) |
| ADD | 4 | 1.69 (-43.9%) |
| ADD | 33 | 2.02 (-32.9%) |
| ADD | 50 | 2.05 (-31.9%) |
| LDM (teacher) | 50 | 2.75 |
| **LatentCRF** | 33 | 2.64 (-4.00%) |
| LDM-L (teacher) | 50 | 2.82 |
| **LatentCRF-L** | 33 | 2.82 (+0.00%) |

这个表是paper最striking的发现：ADD哪怕用50 step也丢了32%的diversity——这是temporal distillation的inherent limitation（学生over-fit到teacher的conditional mean trajectory）。LatentCRF因为不explicitly模仿trajectory、只是替换了denoiser module，所以retain了noise→image映射的多模态性。

### Table 6 / A.2（更激进sparsity）

| Method | Iter | FID↓ | Speed |
|---|---|---|---|
| LDM | 50 | 12.75 | 782.1 |
| LDM | 33 | 12.78 | 501.8 (+35.9%) |
| LDM | 20 | 12.80 | 323.9 (+58.6%) |
| LatentCRF | 33 | 11.58 | 523.4 (+33.2%) |
| LatentCRF | 20 | 11.67 | 344.7 (+55.9%) |

20-step variant能推到55.9%加速，FID只略微退化（11.58 → 11.67），但qualitatively偶尔见artifact。

### CRF ablation (Table 5)

| Text | Higher Order | FID-B | FID-L |
|---|---|---|---|
| √ | √ | 11.58 | 11.27 |
| √ | × | 11.62 | 11.58 |
| × | × | 11.78 | 11.72 |

Higher-order term在large model上贡献明显（11.58 → 11.27），text conditioning贡献中等。仅用unary时输入直接等于输出，省略。

---

## 8. 为什么CRF能retain diversity（intuition）

distillation-based方法（ADD https://arxiv.org/abs/2311.17042, LCM https://arxiv.org/abs/2310.04378, DMD https://arxiv.org/abs/2310.04378）的核心loss是让student在fewer steps内重现teacher的某个deterministic sample path。即使teacher本身是stochastic的（不同noise → 不同image），student容易collapse到conditional mode——因为loss是L2/MSE-style，平均化掉多种可能。

LatentCRF不一样：
1. 它只替换部分iter，前面的LDM iterations还是完整跑（high-variance regime，diversity主要在这里被inject）
2. CRF inference本身是deterministic的energy minimization，但它的input $z_s$ 是LDM跑出来的——diversity在input阶段就被preserve了
3. CRF学的是"natural image manifold上的denoiser"，不是"teacher trajectory mimicry"——所以对同一prompt不同input noise，CRF做不同的"manifold projection"，retain多样性

这一点其实让人联想到vector quantized token-based model（如MarkovGen, 同一作者组 https://openaccess.thecvf.com/content/CVPR2024/papers/Jayasumana_MarkovGen_Structured_Prediction_for_Efficient_Text-to-Image_Generation_CVPR_2024_paper.pdf）——MRF在discrete token space里也保留了diversity。LatentCRF是这个idea的continuous latent space版本。

---

## 9. 相关联想与延伸

### 9.1 和CRFasRNN的血统

LatentCRF的直接祖宗是Zheng et al. 2015的CRFasRNN（https://arxiv.org/abs/1502.03240），那篇把DenseCRF的mean-field inference展开成RNN step，让它end-to-end trainable。LatentCRF的Algorithm 1就是CRFasRNN的latent-space adaptation：
- Message passing换成latent conv
- Compatibility transform加text conditioning
- 加了FoE higher-order term（CRFasRNN只有pairwise）

Krahenbuhl & Koltun 2011的DenseCRF（https://arxiv.org/abs/1210.5644）提供了parallel update empirical convergence的依据，paper里explicit引用。

### 9.2 和Field of Experts / Product of Experts的联系

FoE来自Roth & Black 2005（https://www.cs.toronto.edu/~pspeaks/dnndoku/FoE_cvpr2005.pdf），是PoE（Hinton 1999 https://www.cs.toronto.edu/~hinton/absps/nips00poe.pdf）在image patch prior上的实例化。原版FoE用在pixel-space denoising/super-resolution/inpainting，LatentCRF把它移植到latent space。

值得注意的细节：原版FoE里 $\phi$ 通常是Student-t或Gaussian pdf，作者这里换成 $\phi(y) = e^{y^2/2}$ if $y>0$ else $\varepsilon$——一个"半-Gaussian"形式，目的是让导数变成ReLU，方便neural net实现。这其实意味着filter response只在正值区域被reward，负值被truncated——这个asymmetry从probabilistic视角看不太natural，但实用性强。

### 9.3 和diffusion distillation landscape的对比

| 类别 | 代表方法 | 加速方式 | quality/diversity tradeoff |
|---|---|---|---|
| Efficient sampler | DDIM, DPM-Solver, UniPC | 更好的ODE/SDE solver，少步数 | quality mostly OK, diversity保留 |
| Temporal distillation | Progressive Distill, LCM, ADD, SDXL-Turbo | student学fewer-step trajectory | diversity大幅下降 |
| Compression distillation | SnapFusion, BK-SDM, MobileDiffusion | 瘦身U-Net | quality下降 |
| Block caching | DeepCache, Clockwork, Cache-me-if-you-can | 重用中间feature | quality OK, ~25-50%加速 |
| **Structured replacement** | **LatentCRF, MarkovGen** | **CRF/MRF替换后期iter** | **quality+diversity保留** |

LatentCRF其实开了一个新类别：**structured prior replacement**。它假设"manifold上的最后一段不需要free-form capacity"，用inductive bias换取efficiency。这个思路在video generation、3D generation里应该也可用——后期refine stage常常是budget大头。

### 9.4 Vendi Score作为diversity metric

paper另一个contribution是用Vendi score（Friedman & Dieng, https://arxiv.org/abs/2210.02410）formalize text-to-image的diversity评估。Vendi score本质是effective number of species的generalization：给一个similarity matrix K，计算 $K$ 的特征值 $\lambda_i$，Vendi = $\exp(-\sum_i \hat{\lambda}_i \log \hat{\lambda}_i)$，其中 $\hat{\lambda}_i = \lambda_i / \sum_j \lambda_j$。

这里作者对每个Parti prompt（共1632个），生成16张图（16个noise），用CLIP ViT-L/14@336px提取feature，算16×16 similarity matrix → Vendi score → 平均over 1632 prompts。

这个protocol可以变成text-to-image diversity的标准benchmark。

### 9.5 Limitations

作者承认（Section 5）：
- 偶尔在man-made structure上break lines（Appendix F failure cases）
- 更激进schedule（20 steps）虽然加速55.9%但有更多artifact
- 没有测试在SDXL/MJ级别scale上

潜在问题（我的猜想）：
- CRF的mean-field inference虽然5步converge，但收敛到local minimum——对diverse modes可能不够expressive
- FoE filter在latent space学到的pattern和pixel-space的FoE filter会非常不同，paper没visualize这些filter，好奇它们长什么样
- text conditioning只进pairwise term，没进higher-order——可能限制了complex text-semantic关系

---

## 10. 对你（Andrej）可能感兴趣的points

1. **Inductive bias vs capacity tradeoff**：LDM的U-Net是free-form capacity，对high-variance regime必要，对low-variance regimeoverkill。LatentCRF用CRF的strong inductive bias（spatial smoothness + patch statistics + text conditioning）来cover low-variance regime。这和你之前讲过的"micrograd → modern deep learning的abstraction spectrum"是同一种思考：什么地方加什么prior。

2. **Differentiable PGM as neural layer**：这是2015-2017的lineage（CRFasRNN, DeepCRF等），但在transformer时代被冷落。LatentCRF把它revive在latent diffusion这个新setting里——证明PGM的"explicit structure prior"在生成模型后期refine阶段仍然有价值。

3. **Diversity vs Fidelity trilemma**：你之前在tweet/discussion里提过diffusion的"diversity vs fidelity vs speed"trilemma（参考Sander Dieleman的paradox of diffusion distillation https://sander.ai/2024/01/30/paradox.html）。LatentCRF给出了一个escape route：不distill trajectory，而是replace module with structured prior——diversity自然保留因为noise→image mapping的multi-modality在input端就已经被encode。

4. **训练成本极低**：CRF只有130M参数（vs U-Net 865M/2B），第一阶段300k iter，第二阶段10k iter，全部128 TPUv5e（不算大）。相比SDXL-Turbo那种distillation要超大batch + 大量compute，LatentCRF是"小成本addon"路线。

5. **可能的扩展**：
   - 把LatentCRF思路用在video diffusion（后期frame refinement）
   - 用CRF替换super-resolution阶段（phase 2 of Imagen/MJ）
   - 把FoE filter换成vision transformer patch embedding，capture更复杂patch pattern
   - 探索Gibbs sampling而非mean-field，可能capture multi-modal better

---

## 11. 参考链接汇总

**主paper**:
- LatentCRF (本paper): arxiv链接未提供，但是基于内容应在2024年CVPR/ICML或类似venue
- MarkovGen (同组前作，discrete MRF): https://openaccess.thecvf.com/content/CVPR2024/papers/Jayasumana_MarkovGen_Structured_Prediction_for_Efficient_Text-to-Image_Generation_CVPR_2024_paper.pdf

**LDM & Diffusion基础**:
- LDM (Rombach et al. CVPR 2022): https://arxiv.org/abs/2112.10752
- DDPM (Ho et al.): https://arxiv.org/abs/2006.11239
- DDIM (Song et al.): https://arxiv.org/abs/2010.02502
- Score-based diffusion (Song & Ermon): https://arxiv.org/abs/1907.05600
- EDM (Karras et al.): https://arxiv.org/abs/2206.00364

**CRF as differentiable layer**:
- CRFasRNN (Zheng et al. ICCV 2015): https://arxiv.org/abs/1502.03240
- DenseCRF (Krahenbuhl & Koltun): https://arxiv.org/abs/1210.5644
- Continuous CRF for point cloud (Yang et al.): https://arxiv.org/abs/2102.05891
- Neural MRF for stereo (Guan et al. CVPR 2024): https://arxiv.org/abs/2403.07935

**Field of Experts & Product of Experts**:
- FoE (Roth & Black CVPR 2005): https://ieeexplore.ieee.org/document/1541019
- PoE (Hinton NIPS 1999): https://www.cs.toronto.edu/~hinton/absps/nips00poe.pdf

**Diffusion加速baseline**:
- LCM (Luo et al.): https://arxiv.org/abs/2310.04378
- ADD / SDXL-Turbo (Sauer et al.): https://arxiv.org/abs/2311.17042
- SDXL-Lightning: https://arxiv.org/abs/2402.13929
- InstaFlow (Liu et al.): https://arxiv.org/abs/2309.06380
- DMD / DMD-v2 (Yin et al.): https://arxiv.org/abs/2311.18828
- SnapFusion: https://arxiv.org/abs/2306.00980
- BK-SDM: https://arxiv.org/abs/2305.15798
- Clockwork Diffusion: https://arxiv.org/abs/2402.14477
- DeepCache: https://arxiv.org/abs/2312.00858
- Progressive Distillation (Salimans & Ho): https://arxiv.org/abs/2202.00509

**Diversity metric**:
- Vendi Score (Friedman & Dieng): https://arxiv.org/abs/2210.02410
- Diffusion illusions (Burgert et al., 使用Vendi的diffusion diversity work): https://arxiv.org/abs/2312.03817

**Text-to-Image baselines**:
- Parti Prompts (Yu et al.): https://arxiv.org/abs/2206.10489
- SDXL (Podell et al.): https://arxiv.org/abs/2307.01952
- Imagen (Saharia et al.): https://arxiv.org/abs/2205.11487
- Muse (Chang et al.): https://arxiv.org/abs/2211.00752

**Inductive bias & diffusion distillation paradox**:
- Sander Dieleman's blog on diffusion distillation paradox: https://sander.ai/2024/01/30/paradox.html

**Autoregressive token-based generation**:
- Parti (Yu et al.): https://arxiv.org/abs/2206.10489
- Make-A-Scene (Gafni et al.): https://arxiv.org/abs/2202.01952

---

整体看，LatentCRF最让我impressed的不是33%加速，而是它**结构化地避开了distillation的diversity trap**。temporal distillation那条路已经走到SDXL-Lightning/DMD这种极少数step、靠adversarial loss硬抗diversity collapse的状态；LatentCRF展示了一个qualitatively不同的方向——承认diffusion不同阶段需要不同的model class，后期就交给strong-inductive-bias的小模型。这条路线在更大型setting（video, 3D, long-context generation）下应该有更大的空间。
