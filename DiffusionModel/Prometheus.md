---
source_pdf: Prometheus.pdf
paper_sha256: c580ef946e9545cac3c4e6ca88ef85814c5cb310a3f413462878e699f8842cc6
processed_at: '2026-08-06T06:55:03-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Prometheus 用人话说

好，咱们换个角度，把这篇paper拆开讲，讲清楚每个design choice背后的"为什么"。

---

## 一、这篇paper想解决什么问题？

3D generation领域有个很尴尬的现实：**2D data多到爆炸，3D data少得可怜**。

具体数字感受一下：
- 2D image datasets：100M ~ 2B samples（LAION、SAM-1B等）
- 3D multi-view datasets：撑死100K级（Objaverse、MVImgNet等）

这个数量级差距，导致3D generative model的generalizability根本打不过2D model。你拿Objaverse训练的3D生成模型，换个domain就崩；Stable Diffusion随便画什么都像模像样。

那现有方案怎么"借"2D prior？

**方案A：Score Distillation Sampling (SDS)**  
代表：DreamFusion (https://arxiv.org/abs/2209.14985)、ProlificDreamer (https://arxiv.org/abs/2305.02563)、GaussianDreamer (https://arxiv.org/abs/2311.13084)  
思路：拿一个2D diffusion model当"老师"，让3D representation（NeRF/Gaussian）渲染出来的image在2D model眼里看起来"真实"。通过梯度反传优化3D。

问题：
- 每个scene都要单独optimize，10~30分钟起步
- 2D model没有3D意识，经常出Janus problem（多头怪）
- 2D prior的artifact会传到3D里

**方案B：Multi-view diffusion + 3D reconstruction**  
代表：MVDream (https://arxiv.org/abs/2308.16512) + LGM (https://arxiv.org/abs/2402.05054)  
思路：fine-tune 2D diffusion model让它一次生成多视角image，再用reconstruction model拼成3D。

问题：两步pipeline，误差累积。第一步multi-view image有inconsistency，第二步reconstruction直接崩。

**方案C：直接学3D generative model**  
代表：DiffRF (https://arxiv.org/abs/2210.14760)、3D-aware GAN  
问题：domain-specific，只能生成特定类别（人脸、车），generalization弱。

**Prometheus的insight**：把3D generation formulate成**latent diffusion内的multi-view, pixel-aligned 3D Gaussian生成**。这样一举三得：
- inherit SD的2D prior和computational efficiency
- 通过Gaussian建立2D latent → 3D的bridge
- Feed-forward，几秒出结果

---

## 二、整体架构直觉

```
Text + Camera Poses → [Stage 2: MV-LDM] → multi-view RGB-D latents
                                              ↓
                                    [Stage 1: GS-VAE decoder]
                                              ↓
                                    pixel-aligned 3D Gaussians
                                              ↓
                                    aggregate → scene-level 3D
```

为什么两阶段？我intuition上这样理解：

**Stage 1是让网络"学会什么是3D"**。给它multi-view image + depth，让它重建出3D Gaussian。这是个well-defined reconstruction task，有ground truth supervision。网络在这里学到"如何从2D latent恢复出3D geometry"。

**Stage 2是让网络"学会如何生成3D"**。在Stage 1已经well-behaved的latent space里训练diffusion model，只需要学"如何从noise生成符合text condition的multi-view latent"。不用同时学"什么是3D"和"如何生成3D"两个entangled的难题。

这种decoupling是常见的good practice。就像你要训练一个image生成模型，先train一个VAE把image压成latent，再train diffusion在latent space生成，比直接在pixel space train diffusion高效得多。

---

## 三、Stage 1: GS-VAE 怎么把2D latent变成3D Gaussian

### 3.1 RGB-D Latent Encoding

输入：N个multi-view image $\{I_i\}$ + 对应的monocular depth $\{D_i\}$  
Depth用DepthAnything-V2-S (https://arxiv.org/abs/2406.09414)在线估计，不用GT depth。

编码：
$$\mathcal{E}_\phi : (\mathcal{I}, \mathcal{D}) \mapsto \mathcal{Z} \in \mathbb{R}^{N \times h \times w \times c}$$

- $\mathcal{E}_\phi$: SD image encoder，frozen
- $h \times w$: downsampled latent resolution（SD标准是缩小8倍）
- $c$: channel数（SD是4）

**关键trick**：RGB和depth分别用同一个SD encoder编码，然后concatenate。

为什么能这样？因为Marigold (https://arxiv.org/abs/2312.02145)发现SD encoder对depth map有robust generalization。Depth map长得很像image（都是2D grid of values），SD encoder的features能很好地表达depth structure。这样就避免了专门train一个depth encoder。

**Intuition**：RGB latent负责appearance，depth latent负责geometry，concatenate之后decoder能同时access两边信息。这就是paper说的"disentangle appearance and geometry"。

### 3.2 Plücker Camera Embedding

怎么告诉网络"这些image是从哪个角度拍的"？

用Plücker coordinates：
$$\mathbf{r} = (\mathbf{d}, \mathbf{p} \times \mathbf{d}) \in \mathbb{R}^6$$

- $\mathbf{d} \in \mathbb{R}^3$: normalized ray direction（从camera指向pixel的方向）
- $\mathbf{p} \in \mathbf{R}^3$: camera origin in world coordinates
- $\mathbf{p} \times \mathbf{d}$: ray的moment，表示ray在空间中的位置

每个pixel一条ray，整张image就是 $H \times W \times 6$ 的ray map $\mathcal{R}$。

**为什么用Plücker不用6DoF pose (R, t)？**

Intuition：6DoF pose是camera级别的，一个camera一个pose。但每个pixel的ray其实都不同（perspective projection），Plücker是per-pixel representation。这样ray map可以直接和image latent做channel-wise concatenation，网络容易consume。

而且Plücker是continuous的（比rotation matrix smooth），网络学起来更容易。RayDiff (https://arxiv.org/abs/2402.11722)、CAT3D (https://arxiv.org/abs/2405.10391)、Director3D (https://arxiv.org/abs/2406.17601)都用这个。

然后cross-view transformer做信息融合：
$$\mathcal{C}_\phi : (\mathcal{Z}, \mathcal{R}) \mapsto \tilde{\mathcal{Z}} \in \mathbb{R}^{h \times w \times c}$$

把latent $\mathcal{Z}$ 和ray map $\mathcal{R}$ concat，过transformer，输出fused latent $\tilde{\mathcal{Z}}$。

### 3.3 Pixel-Aligned Gaussian Decoder

Decoder输入：原始latent $\mathcal{Z}$ + fused latent $\tilde{\mathcal{Z}}$ + ray map $\mathcal{R}$（concat在一起）  
输出：每个pixel对应一个3D Gaussian，参数12维。

$$\mathcal{D}_\phi : (\mathcal{Z}, \tilde{\mathcal{Z}}, \mathcal{R}) \mapsto \mathcal{F} \in \mathbb{R}^{N \times H \times W \times C_G}$$

每个Gaussian的12个channel：
| Channel | 含义 |
|---------|------|
| 1 | depth（关键！） |
| 4 | rotation quaternion |
| 3 | scale |
| 1 | opacity |
| 3 | 0阶spherical harmonics (RGB DC term) |

**为什么用depth不用3D center？**

Intuition：用depth (1 channel) 比用3D center (3 channels) 更efficient。每个pixel的Gaussian center = unproject(pixel_coord, depth, camera_pose)，通过camera geometry就能算出来。

这样设计的好处：
- Dimensionality reduction：12维 vs 14维
- 与monocular depth prior对齐（depth是网络学得最稳的geometry signal）
- Pixel-aligned inductive bias：每个pixel对应一个Gaussian，spatial structure直接由image grid决定

最后aggregation：把所有views的Gaussians transform到global coordinate system，concatenate成 $N_G = N \times H \times W$ 个Gaussian primitives，就是scene-level 3D representation。

**Architecture trick**：Decoder直接repurpose SD image decoder，只改第一层和最后一层conv的channel数。这样保留了SD decoder的pretrained features，能well handle latent input。

### 3.4 Loss Function

三部分：
$$\mathcal{L}(\phi) = \lambda_1 \mathcal{L}_{mse} + \lambda_2 \mathcal{L}_{vgg} + \lambda_3 \mathcal{L}_{depth}$$

- $\mathcal{L}_{mse}$: 渲染image vs GT image的MSE
- $\mathcal{L}_{vgg}$: 渲染image vs GT image的VGG perceptual loss (https://arxiv.org/abs/1603.08155)
- $\mathcal{L}_{depth}$: scale-invariant depth loss

Scale-invariant depth loss特别有意思：
$$\mathcal{L}_{depth} = \|(w\hat{D} + q) - \bar{D}\|_2$$

- $\hat{D}$: rendered depth
- $\bar{D}$: monocular depth estimate（pseudo ground truth）
- $w, q$: scale和shift，用least-squares对齐 $\hat{D}$ 和 $\bar{D}$

为什么需要scale-invariant？因为monocular depth $\bar{D}$ 是metric-ambiguous的，只知道相对depth structure，不知道绝对scale和shift。Rendered depth $\hat{D}$ 是metric的（因为camera pose已知）。这个loss只惩罚align之后的residual，让网络学relative depth structure。

来自MiDaS (https://arxiv.org/abs/1907.01341)的经典设计。

---

## 四、Stage 2: MV-LDM 在latent space训练生成

### 4.1 为什么用EDM不用DDPM

Prometheus用EDM (https://arxiv.org/abs/2206.00364)的continuous-time formulation，而不是DDPM (https://arxiv.org/abs/2006.11239)的discrete timesteps。

EDM的好处：
- Noise level $\sigma$ 是连续的，更flexible的noise schedule
- Preconditioning functions让network更容易学

Forward process（加噪声）：
$$\mathcal{Z}_t = \mathcal{Z}_0 + \sigma_t^2 \epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

- $\mathcal{Z}_0$: clean multi-view RGB-D latents
- $\sigma_t$: noise level
- $\epsilon$: standard Gaussian noise
- $\log \sigma_t \sim \mathcal{N}(P_{mean}, P_{std}^2)$: log-normal noise schedule

Reverse process（去噪）：
$$\hat{\mathcal{Z}}_0 = \mathcal{G}_\theta(\mathcal{Z}_t; \sigma_t, \mathbf{y}, \mathcal{R})$$

- $\mathcal{G}_\theta$: denoiser network
- $\mathbf{y}$: text prompt
- $\mathcal{R}$: camera ray maps

EDM preconditioning：
$$\mathcal{G}_\theta(\mathcal{Z}_t; \sigma_t, \mathbf{y}, \mathcal{R}) = c_{skip}(\sigma_t)\mathcal{Z}_t + c_{out}(\sigma_t) F_\theta(c_{in}(\sigma_t)\mathcal{Z}_t; c_{noise}(\sigma_t), \mathbf{y}, \mathcal{R})$$

这些preconditioning functions的作用：
- $c_{skip}(\sigma_t)$: skip connection权重，高噪声时input全是noise，skip应该weight小；低噪声时input接近clean，skip应该weight大
- $c_{in}(\sigma_t)$: input normalization，避免不同noise level下input magnitude差异过大
- $c_{out}(\sigma_t)$: 控制network output的scale
- $c_{noise}(\sigma_t)$: noise level作为conditioning signal传给network
- $F_\theta$: 实际的UNet，从SD 2.1初始化

这些preconditioning的intuition：让network $F_\theta$ 不需要处理各种scale的input/output，只需要学"给定normalized input和noise level，输出normalized prediction"。这样训练更stable。

Training loss：
$$\mathcal{L}(\theta) = \mathbb{E}_{\mathcal{Z}, \mathcal{R}, \mathbf{y}, \sigma_t}[\lambda(\sigma_t)\|\hat{\mathcal{Z}}_0 - \mathcal{Z}_0\|_2^2]$$

Weighting function $\lambda(\sigma) = (1 + \sigma^2)\sigma^{-2}$，balance不同noise level的gradient magnitude。

### 4.2 High Noise Level for Multi-view Consistency

一个非常insightful的design choice：

| 训练模式 | $P_{mean}$ | $P_{std}$ | Intuition |
|---------|-----------|-----------|-----------|
| Multi-view | 1.5 | 2.0 | 偏高噪声 |
| Single-view | -0.5 | 1.2 | 中等噪声 |

**为什么multi-view需要更高noise level？**

Intuition：高噪声 = low SNR = low frequency content。Multi-view consistency本质上是global structure alignment——不同views需要share相同的layout、object identity、scene composition。

如果noise level太低（SNR高），network主要在high frequency details上做denoising，每个view独立refine自己的细节，容易diverge。

如果noise level高，network主要在low frequency structure上做denoising，被forced去先在global structure上"达成共识"，views之间自然更consistent。

这跟Zero123++ (https://arxiv.org/abs/2310.15110)和Stable Video Diffusion (https://arxiv.org/abs/2311.15127)的观察一致：temporal/multi-view consistency需要network在low-frequency space有充分"negotiation"。

### 4.3 3D Cross-View Self-Attention

原SD UNet的self-attention是single image内部的spatial attention。Prometheus把它替换成3D cross-view self-attention。

具体：把multi-view的spatial tokens一起做attention，让不同views的tokens能interact。比如view 1的某个pixel能attend到view 2的对应pixel（或其他pixel）。

Text conditioning仍然用cross-attention（保留SD的text-to-image capability）。

**初始化**：从pretrained SD 2.1 UNet初始化。新加的cross-view attention parameters需要从头学，但借助pretrained features能快速converge。

### 4.4 Hybrid CFG: 解决naive CFG的multi-view inconsistency

这是paper最clever的设计之一。

**Naive CFG** (https://arxiv.org/abs/2207.12598)：
$$\mathcal{G}_\theta^w(\mathcal{Z}_t; \mathbf{y}, \mathcal{R}) = w \cdot \mathcal{G}_\theta(\mathcal{Z}_t; \mathbf{y}, \mathcal{R}) + (w-1) \cdot \mathcal{G}_\theta(\mathcal{Z}_t; \mathcal{R})$$

- $w$: guidance strength
- $\mathbf{y}$: text prompt
- $\mathcal{R}$: camera poses

**问题**：提高 $w$ 让network过度拟合text condition，每个view独立去match text description，导致views之间diverge。

**Hybrid CFG** (HarmonyView, https://arxiv.org/abs/2312.15980)：
$$\mathcal{G}_\theta^w(\mathcal{Z}_t; \mathbf{y}, \mathcal{R}) = \mathcal{G}_\theta(\mathcal{Z}_t; \mathbf{y}, \mathcal{R}) + w_1 \cdot [\mathcal{G}_\theta(\mathcal{Z}_t; \mathbf{y}, \mathcal{R}) - \mathcal{G}_\theta(\mathcal{Z}_t; \mathcal{R})] + w_2 \cdot [\mathcal{G}_\theta(\mathcal{Z}_t; \mathbf{y}, \mathcal{R}) - \mathcal{G}_\theta(\mathcal{Z}_t; \mathbf{y})]$$

- $w_1$: text guidance weight
- $w_2$: pose guidance weight
- $w = w_1 + w_2$

**Intuition**：text和pose的guidance分开做。
- Text guidance项：$\mathcal{G}_\theta(\mathcal{Z}_t; \mathbf{y}, \mathcal{R}) - \mathcal{G}_\theta(\mathcal{Z}_t; \mathcal{R})$ —— 有text vs 没text的difference
- Pose guidance项：$\mathcal{G}_\theta(\mathcal{Z}_t; \mathbf{y}, \mathcal{R}) - \mathcal{G}_\theta(\mathcal{Z}_t; \mathbf{y})$ —— 有pose vs 没pose的difference

这样pose guidance explicitly强迫views respect共同的camera geometry，text guidance保留fidelity gain。两者balance，避免naive CFG把views推向independent text-conditioned images。

**CFG-rescale** (https://arxiv.org/abs/2304.09048)：避免over-saturation问题。从ablation看影响巨大（BRISQUE从89.70降到58.88），说明3D generation里over-saturation问题很严重。

### 4.5 Sampling

公式(14):
$$\mathcal{Z}_{t-1} = \frac{\mathcal{Z}_t - \mathcal{G}_\theta(\mathcal{Z}_t; \sigma_t, \mathbf{y}, \mathcal{R})}{\sigma_t} (\sigma_{t-1} - \sigma_t)$$

这是EDM的Heun discretization的简化形式。从 $\mathcal{Z}_T \sim \mathcal{N}(\mathbf{0}, \sigma_T^2 \mathbf{I})$ 开始，iteratively去噪到 $\mathcal{Z}_0$。

---

## 五、训练数据：scale matters

| Dataset | Type | #Frames | #Scenes |
|---------|------|---------|---------|
| SAM-1B (https://arxiv.org/abs/2304.02643) | Single-view | 11M | - |
| MVImgNet (https://arxiv.org/abs/2303.06047) | Object | 6.8M | 230K |
| DL3DV-10K (https://arxiv.org/abs/2312.16236) | Indoor/Outdoor | 2.2M | 6K |
| Objaverse (https://objaverse.allenai.org/) | Object | 11.5M | 784K |
| ACID (https://arxiv.org/abs/2104.00980) | Indoor | 510K | 11K |
| RealEstate10K (https://arxiv.org/abs/1805.09817) | Indoor | 2.8M | 57K |
| KITTI (https://arxiv.org/abs/1703.02842) | Driving | 42K | 0.8K |
| KITTI-360 (https://arxiv.org/abs/2109.03837) | Driving | 69K | 1.2K |
| nuScenes (https://arxiv.org/abs/1903.11027) | Driving | 340K | 0.85K |
| Waymo (https://arxiv.org/abs/2104.01599) | Driving | 200K | 1K |

**总计约33M frames**，覆盖object/indoor/outdoor/driving四大domain。

**训练scale**：
- Stage 1: 8×A800, batch 32, 200K iterations, 4 days
- Stage 2: 32×A800, batch 8/GPU × 384 GPUs = 3072 images/batch, 350K iterations, 7 days

这个scale在academic 3D generation work里算很aggressive的。

**为什么混训single-view + multi-view？**  
Ablation显示w/o single-view数据，CLIP-Score从0.369降到0.342。Single-view数据量级远大于multi-view，提供main的generalization prior。这跟MVDream (https://arxiv.org/abs/2308.16512)的观察一致。

---

## 六、实验结果：geometry大幅提升，scene-level超越concurrent work

### 6.1 Stage 1 Reconstruction (Tartanair, https://arxiv.org/abs/2003.14338)

Tartanair有Easy/Medium/Hard三档，按view overlap和distance区分。

| Method | Easy δ1↑ | Hard δ1↑ |
|--------|---------|---------|
| pixelSplat (https://arxiv.org/abs/2312.12366) | 0.373 | 0.307 |
| MVSplat (https://arxiv.org/abs/2403.14627) | 0.283 | 0.272 |
| **Ours** | **0.536** | **0.505** |

δ1是depth accuracy（阈值1.25），越高越好。

**Key insight**：Prometheus在Hard mode下相对pixelSplat的δ1相对提升**64%**。这意味着RGB-D latent space + large-scale pretraining让model对low-overlap场景robust得多。

但PSNR上Ours不是全部最优（Easy mode MVSplat 19.38 < Ours 20.95 > pixelSplat 21.65，Hard mode Ours 19.49 > pixelSplat 19.35 > MVSplat 17.87）。这说明**geometry和appearance的trade-off**：Prometheus在geometry上更优，极端case下appearance略次。

### 6.2 Stage 2 Text-to-3D (T3Bench, https://arxiv.org/abs/2310.02977)

Single-object:
| Method | BRISQUE↓ | NIQE↓ | CLIP-Score↑ | Time |
|--------|---------|-------|-------------|------|
| GaussianDreamer | 107.8 | 18.79 | 0.386 | ~15min |
| MVDream+LGM | 74.64 | 14.96 | 0.379 | ~10s |
| Director3D | **49.91** | **13.56** | **0.397** | ~22s |
| **Ours** | 59.43 | 14.23 | 0.329 | **~8s** |

Scene-level（更challenging）:
| Method | BRISQUE↓ | NIQE↓ | CLIP-Score↑ |
|--------|---------|-------|-------------|
| Director3D | 50.88 | 14.97 | 0.357 |
| **Ours** | **49.63** | **14.01** | **0.370** |

**Observations**:
- Scene-level上Ours全面超越Director3D
- Single-object上Director3D的CLIP-Score更高（0.397 vs 0.329）
- 速度上Ours最快（8s），比SDS-based GaussianDreamer快100倍以上，比Director3D快3倍

Single-object上CLIP-Score较低的原因：paper supplementary的Fig 9, 10显示object-centric failure cases，包括multi-view inconsistency和text misalignment。

### 6.3 Ablation Studies

Stage 1 (Hard mode):
- w/o RGB-D: PSNR 18.38, δ1 0.324 → RGB-D带来+1.11 PSNR, +0.181 δ1
- w/o single-view: PSNR 18.63, δ1 0.475 → single-view data带来+0.86 PSNR, +0.030 δ1
- Full: PSNR 19.49, δ1 0.505

Stage 2:
- w/o single-view data: CLIP 0.342 (vs 0.369 full)
- w/o high noise level: CLIP 0.343 (vs 0.369 full)
- w/o hybrid sampling: BRISQUE 66.19 (vs 58.88 full)
- w/o CFG-rescale: BRISQUE 89.70 (vs 58.88 full) — **影响最大**

CFG-rescale对image quality影响巨大，BRISQUE从89.70降到58.88，说明over-saturation问题在3D generation里也很严重。

---

## 七、Limitations：latent 3D generation的根本挑战

### 7.1 Multi-view Inconsistency under Extreme Viewpoints (Fig 9)

尽管有hybrid CFG，在大rotation或extreme viewpoints下仍会出现inconsistency。

**根本原因**：latent space没有explicit 3D representation。Cross-view attention虽然能mitigate，但不能eliminate这个问题。Network在latent space做generation，没有epipolar geometry之类的hard 3D constraint。

### 7.2 Text Misalignment (Fig 10)

Single-view和multi-view联合训练disrupt了原SD的text embedding layer。

**原因**：multi-view data和single-view data的分布差异，让text embedding在两类数据上需要balance，导致object-centric prompt的text alignment下降。

### 7.3 这些limitations指向什么

Latent 3D generation的根本挑战：**如何在latent space impose 3D geometric constraints？**

可能的后续方向：
- **Epipolar attention**：在cross-view attention里加入epipolar constraint，explicitly encode geometric prior。参考Epipolar Transformers在video generation的应用。
- **Volumetric latent**：把latent space本身design成volumetric representation（如3D grid或triplane），而不是multi-view 2D latent。
- **Joint reconstruction-generation training**：Stage 1和Stage 2联合训练，让generation受益于reconstruction的explicit geometric supervision。
- **CAT3D-style aggressive multi-view augmentation** (https://arxiv.org/abs/2405.10391)：用更多synthetic multi-view data boost consistency。

---

## 八、Intuition Summary

整个paper的intuition浓缩成几条：

**1. RGB-D latent space decouples appearance和geometry**  
Depth作为geometry的"anchor"，让Gaussian decoder有更稳定的prior去work with，不需要完全从RGB latent里"猜"geometry。

**2. Pixel-aligned Gaussians = 2D-3D bridge**  
每个pixel对应一个Gaussian，意味着Gaussian的spatial分布直接由camera geometry + depth决定，无需在3D space做explicit sampling。这让SD image decoder可以直接repurpose成Gaussian decoder。

**3. High noise level for global structure consensus**  
Multi-view consistency本质上是global structure alignment问题，需要network在low-frequency space充分"negotiate"。

**4. Hybrid CFG preserves both fidelity和consistency**  
Text和pose的guidance分开做，避免naive CFG把views推向independent text-conditioned images。

**5. Two-stage training decouples representation learning和generation**  
Stage 1学"什么是3D"，Stage 2学"如何生成3D"，避免end-to-end训练时两个objective互相interfere。

**6. Scale matters for generalization**  
混训single-view (11M) + multi-view (22M)比纯multi-view训练效果好，2D prior确实是generalization的关键。

---

## 九、个人联想与后续方向

### 9.1 Epipolar Attention in Latent Space
在multi-view latent attention里加入epipolar constraint。对于view $i$ 的某个pixel，在view $j$ 里只attend到对应epipolar line上的pixels。这样explicitly encode geometric prior，可能解决multi-view inconsistency问题。

### 9.2 Triplane Latent Representation
借鉴EG3D (https://arxiv.org/abs/2112.07945)的triplane representation，把latent space设计成3个orthogonal planes的feature grid。这样latent本身就有3D structure，generation时自然spatially consistent。

### 9.3 Joint Reconstruction-Generation Training
Stage 1和Stage 2联合训练。每个iteration既做reconstruction（有GT supervision）又做generation（从noise采样），让两个task互相regularize。

### 9.4 Higher-Order Spherical Harmonics
当前只用0阶SH (3 channels)，color model太简单。升级到2阶或3阶SH（9或27 channels）可能提升rendering quality，尤其是view-dependent effects。

### 9.5 Dynamic Scene Generation (4D)
扩展到4D，用video diffusion prior。参考V3D (https://arxiv.org/abs/2403.06738)和DimensionX (https://arxiv.org/abs/2411.04928)。每个Gaussian加time-varying parameters（trajectory, deformation）。

### 9.6 Relightable 3D Generation
让Gaussians携带material properties（normal, albedo, roughness），支持relighting。参考GS-IR (https://arxiv.org/abs/2311.16473)和RelightableGaussian (https://arxiv.org/abs/2311.16043)。

### 9.7 Feed-forward 3D Editing
基于InstructPix2Pix (https://arxiv.org/abs/2211.09800)的思路扩展到3D editing。Input一个3D scene + edit instruction，output edited 3D scene。

### 9.8 Better Text Alignment
Preserve SD text embedding via LoRA或separate text branch，解决text misalignment limitation。

### 9.9 Physics-aware Generation
让生成的Gaussians携带physics properties（mass, friction, elasticity），支持physics simulation。

### 9.10 Large-scale 3D World Generation
从scene-level扩展到world-level，生成完整的可交互3D world。这需要spatial compositional design，参考3DitScene (https://arxiv.org/abs/2404.18385)。

---

## 十、最终评价

Prometheus是一个engineering polish非常high的工作。它的核心贡献不在单个技术点，而在于**证明可以在latent diffusion paradigm内做feed-forward scene-level 3D generation，同时leverage 2D prior at scale**。

它把多个known techniques（latent diffusion、3DGS、EDM、Plücker embedding、HarmonyView、Marigold insight）组合成一个coherent framework，在efficiency、generalizability、fidelity上达到了新的balance。

Limitations（multi-view inconsistency under extreme viewpoints, text misalignment）指向了latent 3D generation的fundamental challenge：在latent space缺乏explicit 3D geometric constraint。后续工作很可能会在epipolar attention、volumetric latent、joint reconstruction-generation training等方向突破。

从research perspective，Prometheus的意义在于它**验证了一条path的可行性**：把3D generation完全纳入2D latent diffusion的framework，用2D prior at scale驱动3D generation。这个path的极限在哪里，是一个值得后续工作探索的问题。

参考Project page: https://freemty.github.io/project-prometheus  
参考Stable Diffusion 2.1: https://huggingface.co/stabilityai/stable-diffusion-2-1  
参考EDM framework: https://github.com/NVlabs/edm  
参考HarmonyView: https://github.com/byeongjun-park/HarmonyView  
参考gsplat: https://github.com/nerfstudio-project/gsplat  
参考DepthAnything-V2: https://github.com/DepthAnything/Depth-Anything-V2  
参考Plücker coordinates: https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates  
参考T3Bench: https://github.com/THU-LYJLab/T3Bench  
参考LRM: https://arxiv.org/abs/2311.04400  
参考PixArt-α: https://arxiv.org/abs/2310.00426  
参考Marigold: https://marigoldmonodepth.github.io/  
参考MVDream: https://mv-dream.github.io/  
参考LGM: https://github.com/3DTopia/LGM  
参考CAT3D: https://arxiv.org/abs/2405.10391  
参考Director3D: https://arxiv.org/abs/2406.17601  
参考3DGS原paper: https://arxiv.org/abs/2308.14737

---

# Prometheus: 3D-Aware Latent Diffusion for Feed-Forward Text-to-3D Scene Generation 深度解析

## 一、核心问题与Motivation的直觉

这篇paper的作者团队（Zhejiang University + Ant Group + Tübingen）瞄准的是一个long-standing的tension in 3D generation：

**3D data稀缺 vs 2D data海量**

具体数字上，largest multi-view datasets约100K级别（如Objaverse的rendered views），而single-view datasets + pretrained models可以到100M~2B samples（如LAION、SAM-1B有11M frames）。这种数据scale的不对称导致了一个非常自然的思路：**如何把2D generative prior"迁移"到3D generation，同时保留feed-forward efficiency和generalizability？**

现有方案各有缺陷：
- **SDS类方法**（DreamFusion https://arxiv.org/abs/2209.14985 、ProlificDreamer https://arxiv.org/abs/2305.02563 、GaussianDreamer https://arxiv.org/abs/2311.13084）：per-scene optimization，慢，Janus problem严重
- **Multi-view diffusion + reconstruction**（MVDream https://arxiv.org/abs/2308.16512 + LGM https://arxiv.org/abs/2402.05054）：需要额外reconstruction step，误差累积
- **Direct 3D generative models**（3D-aware GAN、DiffRF）：domain-specific，generalization弱

Prometheus的**关键insight**：把3D scene generation formulate成**在latent diffusion paradigm内的multi-view, feed-forward, pixel-aligned 3D Gaussian generation**。这样既inherit了SD的2D prior和computational efficiency，又通过Gaussian的pixel-aligned design建立了2D latent → 3D representation的bridge。

参考Stable Diffusion: https://arxiv.org/abs/2112.10752  
参考3D Gaussian Splatting: https://arxiv.org/abs/2308.14737  
参考Director3D (concurrent work): https://arxiv.org/abs/2406.17601

---

## 二、整体架构：两阶段范式

Prometheus follows standard latent diffusion (Rombach et al.)的两阶段设计，但每一阶段都被重新design以服务3D generation：

```
Stage 1: GS-VAE ── 建立 2D ⟷ 3D 的bridge
  Input: multi-view RGB-D images + camera poses
  → SD Encoder (frozen) → multi-view transformer fusion → Gaussian decoder
  → pixel-aligned 3D Gaussians → aggregate → scene-level 3D Gaussians

Stage 2: MV-LDM ── 在latent space训练生成模型
  Input: noise + text prompt + camera poses
  → Stable Diffusion 2.1 UNet (modified self-attention → 3D cross-view attention)
  → multi-view RGB-D latents
  → 用Stage 1的GS-VAE decoder解码成3D scene
```

**为什么是两阶段而不是end-to-end？**  
我的理解是：Stage 1先让网络"学会"如何从2D latent representation恢复出3D geometry，这是一个well-defined reconstruction task（有multi-view supervision + depth supervision）。Stage 2则只需要在已经well-behaved的latent space里训练diffusion，避免同时学习"什么是3D"和"如何生成3D"两个entangled的难题。

---

## 三、Stage 1: GS-VAE 技术深度解析

### 3.1 RGB-D Latent Encoding

公式(1):
$$\mathcal{E}_\phi : (\mathcal{I}, \mathcal{D}) \mapsto \mathcal{Z} \in \mathbb{R}^{N \times h \times w \times c}$$

变量含义：
- $\mathcal{I} = \{I_i \in \mathbb{R}^{H \times W \times 3}\}$: N个multi-view RGB images，原始分辨率 $H \times W$
- $\mathcal{D} = \{D_i \in \mathbb{R}^{H \times W \times 1}\}$: N个对应的monocular depth maps（用DepthAnything-V2-S https://arxiv.org/abs/2406.09414 在线估计）
- $\mathcal{Z}$: latent representation，downsample到 $h \times w$，channel数 $c$（SD标准是4 channels）
- $N$: views数量，训练时 $N=4$（multi-view）或 $N=1$（single-view）

**关键trick**: 直接复用SD image encoder，分别encode RGB和depth，然后concatenate。这基于Marigold（https://arxiv.org/abs/2312.02145）的发现：**SD encoder对depth maps有robust generalization，不需要fine-tune就能编码depth信息**。这是一个非常elegant的reuse——既保留了SD的2D prior，又避免了为depth专门train一个encoder。

### 3.2 Multi-View Fusion with Plücker Camera Embedding

公式(2):
$$\mathcal{C}_\phi : (\mathcal{Z}, \mathcal{R}) \mapsto \tilde{\mathcal{Z}} \in \mathbb{R}^{h \times w \times c}$$

这里关键是camera representation的选择：**Plücker coordinates**

$$\mathbf{r} = (\mathbf{d}, \mathbf{p} \times \mathbf{d}) \in \mathbb{R}^6$$

变量：
- $\mathbf{d} \in \mathbb{R}^3$: normalized ray direction (单位向量)
- $\mathbf{p} \in \mathbb{R}^3$: camera origin in world coordinates
- $\mathbf{p} \times \mathbf{d}$: cross product，表示ray的moment
- 整个ray map: $\mathcal{R} = \{R_i \in \mathbb{R}^{H \times W \times 6}\}$，每个pixel一条ray

**为什么用Plücker不用6DoF pose (R, t)？**  
Plücker coordinates的优势：
1. **Per-pixel representation** — 可以直接和image latent做channel-wise concatenation
2. **Continuous & differentiable** — 比rotation matrix/quaternion更smooth
3. **Decouples origin和direction** — 网络容易学习
4. 在RayDiff（https://arxiv.org/abs/2402.11722）、CAT3D、Director3D等工作中已被验证有效

**Cross-view transformer的设计直觉**：每个view的latent独立encode后，需要cross-view信息交换。Transformer的global attention天然适合这个任务。cross-view transformer从pretrained RayDiff初始化，是一个warm-start策略。

### 3.3 Gaussian Decoder: Pixel-Aligned 3D Gaussians

公式(3):
$$\mathcal{D}_\phi : (\mathcal{Z}, \tilde{\mathcal{Z}}, \mathcal{R}) \mapsto \mathcal{F} \in \mathbb{R}^{N \times H \times W \times C_G}$$

公式(4):
$$\mathbb{M}(\mathcal{F}) \mapsto G \in \mathbb{R}^{N_G \times C_G}$$

Decoder的输入是三部分concatenate：原始latent $\mathcal{Z}$ + fused latent $\tilde{\mathcal{Z}}$ + ray maps $\mathcal{R}$。这种residual/skip设计类似UNet的skip connection，让decoder既能access原始per-view信息，又能利用cross-view融合后的信息。

**3D Gaussian的参数化**（$C_G = 12$）：
| Channel数 | 含义 | 说明 |
|-----------|------|------|
| 1 | depth | 替代3D center position，每个pixel的depth |
| 4 | rotation quaternion | 3D Gaussian的orientation |
| 3 | scale | 3轴scaling |
| 1 | opacity | 不透明度 |
| 3 | SH coefficients | 只用0阶SH (DC term)，简化color model |

**这里有个重要的设计决策**：用**depth** (1 channel) 而不是 **3D center** (3 channels)。这有几个好处：
1. 与monocular depth prior对齐
2. Pixel-aligned: 每个pixel的Gaussian中心 = unproject(pixel, depth, camera)
3. Dimensionality reduction + inductive bias

**$N_G = N \times H \times W$** 意味着每个pixel对应一个Gaussian primitive，aggregation operation $\mathbb{M}(\cdot)$ 就是把所有views的Gaussians transform到global coordinate system并concatenate。

参考3DGS原paper: https://arxiv.org/abs/2308.14737  
参考pixelSplat (首个feed-forward 3DGS): https://arxiv.org/abs/2312.12366  
参考MVSplat: https://arxiv.org/abs/2403.14627  
参考GS-LRM: https://arxiv.org/abs/2404.19102

### 3.4 Loss Function细节

公式(5): $\text{R}(\hat{G}, \mathbf{c}) \mapsto \{\hat{I}, \hat{D}\}$  
公式(6): $\mathcal{L}_{render} = \mathcal{L}_{mse}(\hat{I}, I) + \mathcal{L}_{vgg}(\hat{I}, I)$  
公式(7): $\mathcal{L}_{depth} = \|(w\hat{D} + q) - \bar{D}\|_2$  
公式(8): $\mathcal{L}(\phi) = \lambda_1 \mathcal{L}_{mse} + \lambda_2 \mathcal{L}_{vgg} + \lambda_3 \mathcal{L}_{depth}$

**Scale-invariant depth loss**（来自MiDaS https://arxiv.org/abs/1907.01341）的关键：
- $w, q$: scale和shift，用least-squares计算，对齐 $\hat{D}$ 和 $\bar{D}$
- 为什么需要？因为monocular depth $\bar{D}$ 是metric-ambiguous的（up to scale and shift），而rendered depth $\hat{D}$ 是metric的。这个loss让网络学习**相对depth structure**而非绝对metric depth
- $w\hat{D} + q$ 是align operation，loss只惩罚align之后的residual

**为什么用VGG perceptual loss？** MSE alone对high-frequency detail不敏感，VGG features提供更rich的perceptual supervision。注意这里没有用LPIPS，可能是考虑training efficiency。

---

## 四、Stage 2: MV-LDM 技术深度解析

### 4.1 EDM-style Diffusion Formulation

Prometheus采用Karras et al. (EDM, https://arxiv.org/abs/2206.00364)的continuous-time formulation，而不是DDPM的discrete timesteps。

公式(9): $\mathcal{Z}_t = \mathcal{Z}_0 + \sigma_t^2 \epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

变量：
- $\mathcal{Z}_0$: clean multi-view RGB-D latents
- $\sigma_t$: noise level at time $t$
- $\epsilon$: standard Gaussian noise
- $\log \sigma_t \sim \mathcal{N}(P_{mean}, P_{std}^2)$: log-normal noise schedule

公式(10): $\hat{\mathcal{Z}}_0 = \mathcal{G}_\theta(\mathcal{Z}_t; \sigma_t, \mathbf{y}, \mathcal{R})$  
公式(11): $\mathcal{L}(\theta) = \mathbb{E}_{\mathcal{Z}, \mathcal{R}, \mathbf{y}, \sigma_t}[\lambda(\sigma_t)\|\hat{\mathcal{Z}}_0 - \mathcal{Z}_0\|_2^2]$  
公式(12):  
$$\mathcal{G}_\theta(\mathcal{Z}_t; \sigma_t, \mathbf{y}, \mathcal{R}) = c_{skip}(\sigma_t)\mathcal{Z}_t + c_{out}(\sigma_t) F_\theta(c_{in}(\sigma_t)\mathcal{Z}_t; c_{noise}(\sigma_t), \mathbf{y}, \mathcal{R})$$

**EDM preconditioning的intuition**：
- $c_{skip}(\sigma_t)$: skip connection weight，高噪声时input几乎全是noise，skip到output应该weight小；低噪声时input接近clean，skip应该weight大
- $c_{out}(\sigma_t)$: 控制network输出的scale
- $c_{in}(\sigma_t)$: 对input做normalization，避免不同noise level下input magnitude差异过大
- $c_{noise}(\sigma_t)$: 把noise level作为conditioning signal传给network（类似time embedding）
- $F_\theta$: 实际的UNet，从SD 2.1初始化

Weighting function: $\lambda(\sigma) = (1 + \sigma^2)\sigma^{-2}$  
这个weighting的作用是在不同noise level之间balance gradient magnitude，避免高噪声level dominate训练（因为高噪声level的MSE天然更大）。

### 4.2 The Importance of High Noise Level

这是paper中一个非常interesting的design choice：

| 训练模式 | $P_{mean}$ | $P_{std}$ | 含义 |
|---------|-----------|-----------|------|
| Multi-view training | 1.5 | 2.0 | 偏高噪声 |
| Single-view training | -0.5 | 1.2 | 中等噪声 |

**为什么multi-view需要更高noise level？**  
直觉上，low SNR（高噪声）阶段决定global low-frequency structure。Multi-view consistency本质上是要求不同views共享相同的global structure（layout、object identity、scene composition），所以multi-view training需要更多采样在高噪声区域，让network先"达成共识"在global structure上，再去渲染details。

这与Zero123++（https://arxiv.org/abs/2310.15110）和Stable Video Diffusion（https://arxiv.org/abs/2311.15127）的观察一致：temporal/multi-view consistency需要network在low-frequency space有充分的"negotiation"。

### 4.3 Architecture Modification: 3D Cross-View Self-Attention

原SD UNet的self-attention被替换为**3D cross-view self-attention**。具体的modification：
- 原self-attention: single image内部spatial attention
- 3D cross-view attention: 把multi-view的spatial tokens一起做attention，让不同views的同一spatial location（或不同location）能interact

Text conditioning仍然通过cross-attention（保持SD的text-to-image capability）。

**初始化策略**：从pretrained SD 2.1 UNet初始化，保留text-to-image prior。新加的cross-view attention parameters需要从头学，但借助pretrained features能快速converge。

### 4.4 Hybrid Classifier-Free Guidance: 关键的Sampling Innovation

公式(16) naive CFG:
$$\mathcal{G}_\theta^w(\mathcal{Z}_t; \mathbf{y}, \mathcal{R}) = w \cdot \mathcal{G}_\theta(\mathcal{Z}_t; \mathbf{y}, \mathcal{R}) + (w-1) \cdot \mathcal{G}_\theta(\mathcal{Z}_t; \mathcal{R})$$

公式(17) hybrid CFG (HarmonyView, https://arxiv.org/abs/2312.15980):
$$\mathcal{G}_\theta^w(\mathcal{Z}_t; \mathbf{y}, \mathcal{R}) = \mathcal{G}_\theta(\mathcal{Z}_t; \mathbf{y}, \mathcal{R}) + w_1 \cdot [\mathcal{G}_\theta(\mathcal{Z}_t; \mathbf{y}, \mathcal{R}) - \mathcal{G}_\theta(\mathcal{Z}_t; \mathcal{R})] + w_2 \cdot [\mathcal{G}_\theta(\mathcal{Z}_t; \mathbf{y}, \mathcal{R}) - \mathcal{G}_\theta(\mathcal{Z}_t; \mathbf{y})]$$

变量：
- $\mathbf{y}$: text prompt
- $\mathcal{R}$: camera poses
- $w_1$: text guidance weight
- $w_2$: pose guidance weight
- $w = w_1 + w_2$: total guidance strength

**为什么naive CFG破坏multi-view consistency？**  
直觉是：当只对text做guidance时，提高 $w$ 会让network过度拟合text condition，每个view独立去match text description，导致views之间diverge。Hybrid CFG同时explicitly对pose做guidance，强迫views respect共同的camera geometry，同时保留text的fidelity gain。

**CFG-rescale**（https://arxiv.org/abs/2304.09048）：避免over-saturation问题。这个trick在text-to-image领域已经标准化，Prometheus在3D generation里也需要。

公式(14) sampling:
$$\mathcal{Z}_{t-1} = \frac{\mathcal{Z}_t - \mathcal{G}_\theta(\mathcal{Z}_t; \sigma_t, \mathbf{y}, \mathcal{R})}{\sigma_t} (\sigma_{t-1} - \sigma_t)$$

这是EDM的Heun discretization（2nd-order）的简化形式，注意paper里写的是first-order Euler step。

---

## 五、训练数据与Scale

Table 1的数据组合非常ambitious：

| Dataset | Type | #Frames | #Scenes |
|---------|------|---------|---------|
| SAM-1B | Single-view | 11M | - |
| MVImgNet | Object | 6.8M | 230K |
| DL3DV-10K | Indoor/Outdoor | 2.2M | 6K |
| Objaverse | Object | 11.5M | 784K |
| ACID | Indoor | 510K | 11K |
| RealEstate10K | Indoor | 2.8M | 57K |
| KITTI | Driving | 42K | 0.8K |
| KITTI-360 | Driving | 69K | 1.2K |
| nuScenes | Driving | 340K | 0.85K |
| Waymo | Driving | 200K | 1K |

**总计约33M frames**，覆盖object/indoor/outdoor/driving四大domain。

参考SAM-1B: https://arxiv.org/abs/2304.02643  
参考MVImgNet: https://github.com/GAP-LAB-CUHK-MVIG/MVImgNet  
参考DL3DV-10K: https://arxiv.org/abs/2312.16236  
参考Objaverse: https://objaverse.allenai.org/  
参考RealEstate10K: https://arxiv.org/abs/1805.09817  
参考TartanAir: https://theairlab.org/tartanair/

**训练配置**：
- Stage 1: 8×A800, batch 32, 200K iterations, 4 days
- Stage 2: 32×A800, batch 8/GPU, total batch 3072 images, 350K iterations, 7 days
- Single-view + multi-view混训，single-view时loss只施加在input views

---

## 六、实验结果深度分析

### 6.1 Stage 1 Reconstruction (Tartanair)

Tartanair有Easy/Medium/Hard三档，按view overlap和distance区分。

| Method | Easy δ1↑ | Hard δ1↑ |
|--------|---------|---------|
| pixelSplat | 0.373 | 0.307 |
| MVSplat | 0.283 | 0.272 |
| **Ours** | **0.536** | **0.505** |

**Key insight**: Prometheus在Hard mode下相对pixelSplat的δ1相对提升**64%**。这意味着RGB-D latent space + large-scale pretraining让model对low-overlap场景robust得多。

但在PSNR上Ours并非最优（Easy mode MVSplat 19.38 < Ours 20.95 > pixelSplat 21.65，Hard mode Ours 19.49 > pixelSplat 19.35 > MVSplat 17.87），这说明**geometry和appearance的trade-off**：Prometheus在geometry上更优，但极端case下appearance略次。

### 6.2 Stage 2 Generation (T3Bench + 80 scene prompts)

| Method | BRISQUE↓ | NIQE↓ | CLIP-Score↑ | Time |
|--------|---------|-------|-------------|------|
| GaussianDreamer | 107.8 | 18.79 | 0.386 | ~15min |
| MVDream+LGM | 74.64 | 14.96 | 0.379 | ~10s |
| Director3D | **49.91** | **13.56** | **0.397** | ~22s |
| **Ours** | 59.43 | 14.23 | 0.329 | **~8s** |

Scene-level（更challenging）：

| Method | BRISQUE↓ | NIQE↓ | CLIP-Score↑ |
|--------|---------|-------|-------------|
| Director3D | 50.88 | 14.97 | 0.357 |
| **Ours** | **49.63** | **14.01** | **0.370** |

**Critical observation**: 在scene-level任务上Ours全面超越Director3D，但在single-object上Director3D的CLIP-Score更高（0.397 vs 0.329）。Paper的supplementary指出这是由于object-centric failure cases（Fig 9, 10）。

**速度上Ours最快**（8s），相比SDS-based GaussianDreamer快100倍以上，相比Director3D也快3倍。

### 6.3 Ablation Studies

Stage 1 (Hard mode):
- w/o RGB-D: PSNR 18.38, δ1 0.324 → RGB-D带来+1.11 PSNR, +0.181 δ1
- w/o single-view: PSNR 18.63, δ1 0.475 → single-view data带来+0.86 PSNR, +0.030 δ1
- Full: PSNR 19.49, δ1 0.505

Stage 2:
- w/o single-view data: CLIP 0.342 (vs 0.369 full)
- w/o high noise level: CLIP 0.343 (vs 0.369 full)
- w/o hybrid sampling: BRISQUE 66.19 (vs 58.88 full)
- w/o CFG-rescale: BRISQUE 89.70 (vs 58.88 full) — **影响最大**

CFG-rescale对image quality影响巨大，BRISQUE从89.70降到58.88，说明over-saturation问题在3D generation里也很严重。

---

## 七、Limitations 与思考

### 7.1 Multi-view Inconsistency (Fig 9)
尽管有hybrid CFG，在大rotation或extreme viewpoints下仍会出现inconsistency。这是因为latent space没有explicit 3D representation，cross-view attention虽然能mitigate但不能eliminate这个问题。

### 7.2 Text Misalignment (Fig 10)
Single-view和multi-view联合训练disrupt了原SD的text embedding layer。Paper提到设计specialized architecture来preserve text alignment capability是future work。

**这两个limitation指向一个更深层的问题**：在latent space做multi-view generation本质上缺少3D geometric constraints。后续工作可能需要：
- Explicit 3D-aware attention（如epipolar attention）
- Volumetric representation in latent space
- 或像CAT3D（https://arxiv.org/abs/2405.10391）那样更aggressive的multi-view数据augmentation

---

## 八、与Concurrent Works的对比

### Director3D vs Prometheus
- Director3D: 在image space做multi-view diffusion，需要image-space supervision
- Prometheus: 在latent space做，computational overhead更低，scalability更好
- Director3D在object-centric更强，Prometheus在scene-level更强

### CAT3D vs Prometheus
- CAT3D: multi-view diffusion + 3D reconstruction pipeline (两步)
- Prometheus: end-to-end latent diffusion + Gaussian decoder (一步)

### V3D vs Prometheus
- V3D (https://arxiv.org/abs/2403.06738): 用video diffusion model做3D generation
- Prometheus: 用multi-view latent diffusion，更explicit的3D Gaussian output

### LN3Diff vs Prometheus
- LN3Diff (https://arxiv.org/abs/2403.12019): latent neural fields diffusion
- Prometheus: pixel-aligned Gaussians，更直接的2D-to-3D bridge

---

## 九、Intuition Summary

把整篇paper的intuition浓缩成几个关键insights：

1. **RGB-D latent space decouples appearance and geometry**：depth作为geometry的"anchor"，让Gaussian decoder有更稳定的prior去work with，而不是完全从RGB latent里"猜"geometry。

2. **Pixel-aligned Gaussians = 2D-3D bridge**：每个pixel对应一个Gaussian，意味着Gaussian的spatial分布直接由camera geometry + depth决定，无需在3D space做explicit sampling。这让SD image decoder可以直接repurpose成Gaussian decoder。

3. **High noise level for global structure consensus**：multi-view consistency本质上是global structure alignment问题，需要network在low-frequency space充分"negotiate"。

4. **Hybrid CFG preserves both fidelity and consistency**：text guidance和pose guidance分开做，避免naive CFG把views推向independent text-conditioned images。

5. **Two-stage training decouples representation learning and generation**：Stage 1学"什么是3D"，Stage 2学"如何生成3D"，避免end-to-end训练时两个objective互相interfere。

6. **Scale matters for generalization**：混训single-view (11M frames) + multi-view (22M frames)比纯multi-view训练效果好，说明2D prior确实是generalization的关键。

---

## 十、可能的研究方向联想

1. **Epipolar Attention in Latent Space**：在multi-view latent attention里加入epipolar constraint，explicitly encode geometric prior，可能解决multi-view inconsistency问题。参考Epipolar Transformers in video generation。

2. **Joint Reconstruction-Generation Training**：Stage 1和Stage 2联合训练，让generation受益于reconstruction的explicit geometric supervision。

3. **More expressive SH / Anisotropic Gaussians**：当前只用0阶SH (3 channels)，升级到高阶SH或spherical Gaussian可能提升rendering quality。

4. **Dynamic Scene Generation**：扩展到4D，用类似V3D的video diffusion prior。参考DimensionX (https://arxiv.org/abs/2411.04928)。

5. **Conditional Editing**：基于Gaussian的compositional editing，如3DitScene (https://arxiv.org/abs/2404.18385)。

6. **Better Text Alignment**：preserve SD text embedding via LoRA或separate text branch，解决text misalignment limitation。

7. **Feed-forward 3D Editing**：用instruct-pix2pix-style training (https://arxiv.org/abs/2211.09800)扩展到3D editing。

8. **Physics-aware Gaussians**：让Gaussians携带material properties，支持relighting（参考GS-IR, RelightableGaussian）。

---

## 十一、实现细节补充

- **gsplat** (https://github.com/nerfstudio-project/gsplat) 作为可微分Gaussian renderer，比原版3DGS implementation更efficient
- **DepthAnything-V2-S** 在线估计depth，而不是用GT depth，让model在inference时也能work with任意input
- **Camera ray maps**和noisy latents沿channel concat，是conditioning的简单但有效方式
- **Drop condition probability 10%** 训练时随机drop text或pose，enable classifier-free guidance
- **CFG-rescale factor** 没有explicit给出具体值，但根据[41]通常在0.5-0.7之间

---

## 十二、最终评价

Prometheus是一个engineering polish非常high的工作，把多个known techniques（latent diffusion、3DGS、EDM、Plücker embedding、HarmonyView、Marigold insight）组合成一个coherent framework，在efficiency、generalizability、fidelity上达到了新的balance。其核心贡献不在单个技术点，而在于**证明可以在latent diffusion paradigm内做feed-forward scene-level 3D generation while leveraging 2D prior at scale**。

其limitation（multi-view inconsistency under extreme viewpoints, text misalignment）指向了latent 3D generation的fundamental challenge：在latent space缺乏explicit 3D geometric constraint。后续工作很可能会在epipolar attention、volumetric latent、joint reconstruction-generation training等方向突破。

参考Project page: https://freemty.github.io/project-prometheus  
参考Stable Diffusion 2.1: https://huggingface.co/stabilityai/stable-diffusion-2-1  
参考EDM framework: https://github.com/NVlabs/edm  
参考HarmonyView: https://github.com/byeongjun-park/HarmonyView  
参考gsplat: https://github.com/nerfstudio-project/gsplat  
参考DepthAnything-V2: https://github.com/DepthAnything/Depth-Anything-V2  
参考Plücker coordinates: https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates  
参考T3Bench: https://arxiv.org/abs/2310.02977  
参考LRM: https://arxiv.org/abs/2311.04400  
参考PixArt-α: https://arxiv.org/abs/2310.00426  
参考Marigold: https://marigoldmonodepth.github.io/  
参考MVDream: https://mv-dream.github.io/  
参考LGM: https://github.com/3DTopia/LGM
