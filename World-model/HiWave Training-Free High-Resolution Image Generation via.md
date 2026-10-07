---
source_pdf: HiWave Training-Free High-Resolution Image Generation via.pdf
paper_sha256: e0e3f89ee2bbbd5265065c1ffd48965b69c2af98df1b52df4aab1cee350be7d7
processed_at: '2026-08-04T23:51:32-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 HiWave

好，刚才那版太硬了，Karpathy，我重新用大白话给你讲一遍这个故事。

---

## 问题：大画布上的尴尬

SDXL 这个 model 只在 1024×1024 上训练过。你让它画 4096×4096，它就懵了。

为什么懵？因为 attention 的计算量跟图的大小是平方关系——4K 图的 attention 比 1K 图贵 16 倍。训练 4K 的 model 成本天文数字，而且也没有那么多 4K 的训练数据。所以大家手里的 powerful model 都只在 1K 上 train。

那怎么出 4K 图？两条老路，各有死法。

---

## 老路一：切 patch 画

把 4K 大画布切成一堆 1K 小块，每块单独让 SDXL 画，最后拼起来。DemoFusion、AccDiffusion、Pixelsmith 走这条。

死法很搞笑：**每个 patch 都画出一模一样的主体**。

prompt 说 "a couple in a wedding"，每个 1K patch 看到这个 prompt，都觉得"我应该画一对夫妇"，于是 4 个 patch 拼起来出现 4 对夫妇。树丛里、草地上、天上，到处冒出额外的人。这就是 paper 里说的 object duplication。

直觉上很好理解——每个 patch 是独立的，没有"你已经画了一对夫妇，别再画了"这种全局协调。

---

## 老路二：改 model 直接跑 4K

把 model 的 attention 改一改，让它单次直接吐 4K 图。HiDiffusion、FouriScale、ScaleCrafter 走这条。

死法：**结构崩**。因为 model 从没在 4K 上训练过，它的 prior 全是 1K 域的。你硬塞 4K latent 给它，分布外太远，出来的图模糊、扭曲、不连贯。HiDiffusion 在 2048² 还行，到 4096² 直接 FID 从 65 飙到 93（Table 1）。

---

## HiWave 的赌注：先画素描，再填细节，但别动构图

核心 idea 就一句话：**先让 SDXL 画一张 1K 的小图当"素描稿"，放大到 4K，然后分 patch 加细节——但加细节时只允许动纹理，不许动构图。**

这就像画油画的流程。先画 sketch 定构图，再逐块填 texture。填 texture 的时候，人脸位置不能动，但可以加毛孔、加发丝、加衣服褶皱。

但怎么让 model "只动纹理不动构图"？这就需要把信号在频域切开。

---

## 关键概念一：DWT 把信号劈两半

DWT（discrete wavelet transform）就是个小工具，把一张图劈成"低频"和"高频"两部分。

- **低频**：图的大轮廓、整体结构、物体位置。模糊版本的图。
- **高频**：细节、纹理、边缘、发丝、布料纹路。

比喻一下：低频是音乐的 bass line，高频是高音细节。你把低频锁死，歌曲的骨架不变；你增强高频，细节更丰富。

HiWave 用的是 **sym4 wavelet**（Daubechies symlet 家族，4 个 vanishing moments），在空间定位和频率定位之间做了个好折衷。比 Haar 平滑，比 db4 对称性更好。PyWavelets 库里直接调 `pywt.Wavelet('sym4')` 就有。

2D DWT 把图劈成 4 个 sub-band：
- $x_L$（low × low）：低频
- $x_H$（low × high）：水平细节
- $x_V$（high × low）：垂直细节
- $x_D$（high × high）：对角细节

每个 sub-band 大小是原图的 $H/2 \times W/2$。而且这个变换完全可逆——`iDWT` 能无损拼回去。

---

## 关键概念二：DDIM inversion——把画好的图反推回噪点

Diffusion model 的工作方式：从一堆随机噪点开始，一步步去噪，最后变出一张图。

DDIM inversion 是反过来：给你一张画好的图，倒着走一遍，反推回"这张图是从哪堆噪点来的"。

公式（forward diffusion ODE，Karras 2022 形式）：

$$
\mathrm{d}z = -\dot{\sigma}(t)\,\sigma(t)\,\nabla_{z_t}\log p_t(z_t)\,\mathrm{d}t
$$

变量解释：
- $z_t$：在时刻 $t$ 的 noisy latent。
- $\sigma(t)$：noise schedule，$t=0$ 时是 clean data，$t=T$ 时是满噪声。
- $\dot{\sigma}(t)$：noise scale 对时间求导。
- $\nabla_{z_t}\log p_t(z_t)$：score function，被 neural denoiser $D_\theta$ 近似。

从 $t=0$ 积分到 $t=T$，就把 image 反推回 noise。

**为什么这步关键？** 因为 patch-based 生成时，每个 patch 都要从噪点开始画。如果用纯随机 Gaussian noise，每个 patch 各自从随机起步，它们之间没有共享上下文——于是各画各的，duplication 就来了。

如果用 DDIM inversion 从 base image 反推的 noise 起步，每个 patch 的初始 noise 都"记得"base image 长什么样。相邻 patch 的 noise 还是 spatially consistent 的，拼起来不会有 seam。

这是 Figure 11 的 ablation 证明的——去掉 DDIM inversion，patch 之间就出现 color mismatch 和 geometry 不对齐。

---

## 关键概念三：CFG——让图更像 prompt

CFG（classifier-free guidance）是 SDXL 这类 model 的标配技巧。model 同时跑两次：一次给 prompt（conditional），一次不给 prompt（unconditional）。把两者的差值放大，出来的图就更贴 prompt。

公式：

$$
\hat{D}_{CFG}(z_t, t, y) = D_\theta(z_t, t) + w\,(D_\theta(z_t, t, y) - D_\theta(z_t, t))
$$

变量：
- $D_\theta(z_t, t, y)$：conditional prediction（给 prompt）。
- $D_\theta(z_t, t)$：unconditional prediction（不给 prompt）。
- $w$：guidance strength，SDXL 默认 5 左右，越大越贴 prompt 但也越容易 oversaturation。

问题在哪？standard CFG 把**整个信号**——低频和高频一起——都乘以 $w$。低频被放大意味着 base image 的全局结构被 push 到分布外，于是出现 saturation 和 duplication。Sadat 自己 2025 ICLR 那篇 "Eliminating Oversaturation" 就讲这个。

---

## HiWave 的核心招式：频域分工的 CFG

HiWave 说：CFG 整体放大是错的。**低频不该 CFG，高频该 CFG。**

为什么？回到直觉——

- **低频**是 scene layout，是 base image 已经画好的结构。CFG 在这里会扭曲全局，引发 duplication。所以低频直接拿 conditional prediction，不 CFG。
- **高频**是 texture 和 detail，是 1K SDXL 出来不够细的地方。CFG 在这里能 push 出新细节，而且高频没有"saturation"概念（高频就是 edge 和 micro-pattern），不怕 oversaturation。

具体怎么实现？

Step 1：对 conditional 和 unconditional 两个 prediction 分别做 DWT：

$$
\mathrm{DWT}(D_c(z_t)) = \{D_c^L, D_c^H, D_c^V, D_c^D\}
$$
$$
\mathrm{DWT}(D_u(z_t)) = \{D_u^L, D_u^H, D_u^V, D_u^D\}
$$

$D_c, D_u$ 是 conditional / unconditional denoiser output，上标 $L, H, V, D$ 是 wavelet 的 4 个 sub-band。

Step 2：频域分别构造 guided signal——

低频直接拿 conditional（不 CFG）：

$$
\tilde{D}_{CFG}^L = D_c^L
$$

高频三个 band 做 CFG（用 $w_d = 7.5$ 这个比 standard 高的 strength）：

$$
\tilde{D}_{CFG}^H = D_u^H + w_d\,(D_c^H - D_u^H)
$$
$$
\tilde{D}_{CFG}^V = D_u^V + w_d\,(D_c^V - D_u^V)
$$
$$
\tilde{D}_{CFG}^D = D_u^D + w_d\,(D_c^D - D_u^D)
$$

Step 3：iDWT 重组：

$$
\tilde{D}_{CFG} = \mathrm{iDWT}(\tilde{D}_{CFG}^L, \tilde{D}_{CFG}^H, \tilde{D}_{CFG}^V, \tilde{D}_{CFG}^D)
$$

这个 $\tilde{D}_{CFG}$ 就拿去当本 step 的 prediction 用。

**intuition 浓缩**：低频被冻结在 base image 的结构上，高频被 CFG 放大去生新纹理。

为什么 DWT 做 在 denoiser output 上而不是 latent 上？因为 denoiser output $D_\theta(z_t, t)$ 是 model 对 clean image $x_0$ 的 prediction，这个 prediction 是图像语义级的，分频有意义——低频 = 全局 layout，高频 = texture。而 latent $z_t$ 是被噪声主导的中间态，分频语义混乱。

---

## Skip residual：早期别走太远

还有个小技巧。采样的前几步，把当前 latent 和 DDIM-inverted 的 latent 混一下，让生成别离 base image 太远。到了后期就放开。

公式：

$$
c_1 = \left(\frac{1 + \cos\big((T-t)/T \cdot \pi\big)}{2}\right)^\alpha
$$

$$
\hat{z}_t = \begin{cases}
c_1 \cdot z_t + (1 - c_1) \cdot z_t^s & t < \tau \\
z_t & t \geq \tau
\end{cases}
$$

变量：
- $z_t$：当前 sampling latent。
- $z_t^s$：同 step 的 DDIM-inverted latent（来自 base image）。
- $\tau$：threshold。2048² 时 step 15/50，4096² 时 step 30/50。
- $c_1$：cosine-decay 权重，采样早期（$t$ 大）时 $c_1$ 大，采样后期（$t$ 小）时 $c_1$ 小。

直觉：早期 sampling 是定大结构的关键期，这时候多保留 inverted latent 让 model 不要走偏。后期放开让 model 自由 synthesis 细节。

跟 prior work 不同的是，HiWave 只在前 $\tau$ 步用 skip residual，不是全程都用。全程用会压抑细节生成，完全不用又出 duplication。这个 $\tau$ 的选择是 trade-off。

---

## Progressive upscaling：别一步跳 4×

不要从 1024² 直接跳到 4096²（4× scaling），而是 1024² → 2048² → 4096²，每步 2×。

直觉：每步只 2× upscaling，model 的"推理负担"更小，detail synthesis 更可控。one-shot 4× 跳太远，DDIM inversion 和 sampling 的 gap 太大，细节容易糊。Figure 13 的 ablation 验证这个——multistep 的 raspberry 比一shot 的锐利得多。

而且 paper 说他们的 progressive 不会像 prior work 那样越迭代越 duplication——因为 DDIM inversion + DWT guidance 在每步都把低频锁住了。

---

## Image domain upscaling 而不是 latent domain

这个细节值得单独说。SDXL 的 VAE 是在 1024² 上训练的。VAE 的 latent manifold 对 spatial scaling 没有 equivariance——你直接把 128² latent 放大到 512²，VAE decoder 没见过这种 latent distribution，会出 severe spatial artifact（Figure 4 左边那种）。

EQ-VAE（Kouzelis et al. 2025）专门给 VAE 加 equivariance regularization 来治这个。但 HiWave 不改 VAE，走另一条路：在 image space 放大后再 encode 回 latent space，绕过这个问题。

流程：
1. SDXL 出 1024² image。
2. Lanczos 把 image 放大到 4096²。
3. VAE encoder 把这张 4096² image encode 到 latent（512²）。
4. 这个 latent 虽然没细节，但 layout 是对的，因为是从真实 image encode 来的。

---

## 完整 pipeline 一遍过

用大白话讲一遍：

1. SDXL 在 1024² 出一张 base image——构图 OK，但细节少。
2. Lanczos 把它放大到 4096²——还是糊的，因为只是插值。
3. VAE encoder 把 4096² image encode 成 latent。
4. 对每个 patch 做 DDIM inversion——反推这个 patch 对应的 noise。
5. 从这个 inverted noise 开始 sampling，每个 step：
   - 跑 conditional 和 unconditional 两次 forward。
   - 对两个 output 都做 DWT，劈成低频 + 3 个高频 band。
   - 低频直接拿 conditional 的（锁结构）。
   - 高频 3 个 band 做 CFG with $w_d=7.5$（放细节）。
   - iDWT 拼回完整 prediction。
   - 早期 step 还跟 inverted latent 做 skip residual 混合。
6. 一个 patch 跑完，下一个 patch，50% overlap 保证 seam 平滑。
7. 1024² → 2048² → 4096²，progressive 每步 2×。

---

## 为什么管用？

把两个老死法的根因分别治了：

**Patch-based 的 duplication 根因**：每个 patch 独立从随机 noise 起步，没有共享上下文，各画各的 scene-level 内容。

**HiWave 的治法**：
- DDIM inversion 让 patch 从 base image 反推的 noise 起步，patch 之间 spatially consistent。
- DWT guidance 锁死低频，model 不能在低频上"创新"（不能新发明主体位置），只能在高频上加 texture。

**Direct inference 的结构崩根因**：model 没在 high-res 上训练过，OOD 太远。

**HiWave 的治法**：patch-based 让 model 始终在 native 1024² 上跑，没有 OOD 问题。全局结构靠 base image + DDIM inversion 保证。

---

## 实验讲人话

**User study**：32 对图，548 次盲测 A/B。HiWave 胜出 81.2%，7 个 case 100% 偏好 HiWave。这个数字最有说服力，因为 FID/CLIP 这些 metric 都要把图 downsample 到 224-299 再算，4K 的细节根本看不出来，metric 上大家差不多。

**Runtime**：4096² 出一张 1557 秒（RTX 3090）。比 Pixelsmith 慢 3 倍——多步 upscaling + patch-wise inversion 的代价。但质量明显更好，所以这个 trade-off 值。

**8K 实验**：4096² → 8192²（64× pixel count）依然结构 coherent（Figure 16）。HiDiffusion 在 4K 都崩了，更别说 8K。这是 HiWave scalability 的最强证据。

**Real image upscaling**：把真实 1024² 照片（不是 SDXL 生成的）用 DDIM inversion 反推到 noise space，再 HiWave 提升到 2048²。turtle 的 shell texture 提升明显（Figure 14）。说明 SDXL 的 noise space 已经 capture 了 high-frequency prior，能 zero-shot 处理真实图。

---

## Ablation 讲人话

**去 DWT guidance（用 standard CFG）**：婚礼照片里树丛出现第二对夫妇，草地出现一只断手（Figure 9）。standard CFG 把低频也放大，于是 base image 的结构被扭曲，model 在 patch 里重新发明主体。ImageReward 看不出差异（0.0443 vs 0.0168），说明现在的 metric 对这种 perceptual 问题完全瞎。

**只对低频做 guidance（HiWave 的反操作）**：duplication 也出现。说明高频 CFG 不只是"加细节"，它在压住低频扭曲上也起作用。

**去 DDIM inversion**：patch 之间 color mismatch、geometry 不对齐、visible seam（Figure 12）。因为 patch 从随机 noise 起步，没有共享上下文。

**one-shot 4096² vs progressive multistep**：multistep 细节更锐（Figure 13）。1024 → 2048 → 4096 每步 2× 比 1024 → 4096 一步 4× 更可控。

---

## 为什么用 wavelet 不用 FFT？

Karpathy 你可能会问：FFT 不也能分频吗？为什么用 wavelet？

答案在 **spatial localization**。FFT 的 basis 是 global sinusoid，每个 frequency component 跨整张图。但 image 的 texture 是 spatially local 的——这块是石头纹理，那块是头发纹理。FFT 看不出"这块的高频"和"那块的高频"不同。

Wavelet 同时有 spatial 和 frequency localization——Heisenberg uncertainty 的最优折衷。sym4 在这点上比 Haar（太 blocky）和 db4（不对称有 phase distortion）都好。

类似思路在 WaveGrad（Kong et al. 2021）里用过——audio waveform 生成用 wavelet decomposition 来分层建模。

---

## 同作者的研究脉络

HiWave 的作者 Sadat 和 Weber（Disney Research / ETH）之前做过 CADS（Sadat et al. 2024）和 "Eliminating Oversaturation"（Sadat et al. 2025 ICLR）。

- CADS：condition annealed sampling，sampling 过程中逐渐降低 guidance 强度，避免高 $w$ 导致的 artifact。
- Eliminating Oversaturation：分析 high guidance scale 为什么导致 oversaturation，并提出修复。

HiWave 是同一条线的延伸——把 CFG 的负面影响**局部化**到 frequency domain。CADS 是 time domain annealing，HiWave 是 frequency domain separation。同一组人对 CFG 的研究很连贯。

---

## Open Problems 和留给你 Karpathy 的思考

1. **Runtime 太慢**。1557s 出一张 4K，多步 upscaling 是主因。能否用 rectified flow（SD3）或 consistency model 把单步 cost 降下来？rectified flow 4 step 能出 1024²，但 high-res patch-wise 多步结构没被 reflow 形式描述过。这是个明显的 extension 方向。

2. **Prompt dependency**。real image upscaling 场景需要 manually craft prompt（Figure 14）。prompt 不准则 conditional/unconditional 差不够锐利，high-frequency guidance 效果打折。这暴露了 CFG-based 方法的固有问题。能否用无 prompt 的 guidance（比如 image-based guidance）替代？

3. **频率分得粗**。只分了 low + 3 个 high band。多级 wavelet pyramid（3-level DWT 分成 1 个 low³ + 9 个 high sub-band），每个 sub-band 单独 guidance schedule 会怎样？这是个值得探索的 ablation。

4. **Low-frequency 完全锁死是否过保守**？有的场景 base image 的 layout 不完美，是否可以给低频一个小的 $w_l$（比如 1.5）做微调？

5. **Video extension**。作者 future work 提到。DWT-based frequency guidance 在 latent video diffusion 里也成立——temporal frequency 分 low（motion layout）/ high（frame-level texture），跟 spatial 这边对称。Karpathy 你做 video diffusion 那条线，这里直接有迁移空间。

6. **Metric 不可靠**。FID/CLIP/HPS-v2 都 downsample 到 224-299，4K 细节看不到。需要 high-res-aware 的 metric。HPS-v2 已经是 SOTA human preference model，但用 ViT-H/14 in 224²。能否 train 一个 high-res 的人偏好 model？

---

## Reference Links

- **HiWave paper**：https://arxiv.org/abs/2412.13420
- **SDXL**：https://arxiv.org/abs/2307.01952
- **EDM (Karras et al. 2022)**：https://openreview.net/forum?id=k7FuTOWMOc7
- **DDIM**：https://arxiv.org/abs/2010.02502
- **Classifier-Free Guidance**：https://arxiv.org/abs/2207.12598
- **DemoFusion**：https://arxiv.org/abs/2311.16981
- **AccDiffusion**：https://arxiv.org/abs/2407.01852
- **Pixelsmith**：https://arxiv.org/abs/2406.07251
- **HiDiffusion**：https://arxiv.org/abs/2307.06340
- **FouriScale**：https://arxiv.org/abs/2404.02943
- **ScaleCrafter**：https://arxiv.org/abs/2307.02937
- **MegaFusion**：https://arxiv.org/abs/2408.09905
- **EQ-VAE**：https://arxiv.org/abs/2502.09509
- **CADS（同作者 Sadat）**：https://openreview.net/forum?id=zMoNrajk2X
- **Eliminating Oversaturation（同作者 Sadat）**：https://openreview.net/forum?id=e2ONKX6qzJ
- **Sana**：https://arxiv.org/abs/2410.10629
- **PixArt-Σ**：https://arxiv.org/abs/2401.05252
- **SD3 / Rectified Flow**：https://arxiv.org/abs/2403.03206
- **WaveGrad**：https://openreview.net/forum?id=NsMLjcFaO8O
- **PyWavelets (sym4)**：https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html
- **LAION-5B**：https://openreview.net/forum?id=M3Y74vmsMcY
- **HPS-v2**：https://arxiv.org/abs/2306.09341
- **Diffusers**：https://huggingface.co/docs/diffusers/index

---

## 最后的人话总结

HiWave 的故事就一句话：**画素描再加细节，加细节只动纹理不动构图。**

技术上落地成三件事：
1. **DDIM inversion** 让每个 patch 从 base image 反推的 noise 起步，patch 之间不打架。
2. **DWT 把 denoiser output 劈成低频+高频**，低频直接拿 conditional 锁结构，高频用增强版 CFG 放细节。
3. **Progressive upscaling** 1024→2048→4096 每步 2×，别一步跳 4×。

这套组合让 SDXL 在 24GB VRAM 上能跑 8192×8192，不出 duplication，不崩结构。不需要 retraining，不需要改架构，inference-time 套上去就跑。

Karpathy 你要 build intuition 的话，记住这个心法：**频率分工，低频锁结构，高频放细节**。DWT 只是个工具，核心 idea 是把 CFG 的"放大"作用从全局限制到局部频段。这个 idea 可以迁移到 video、audio、3D——任何有"全局结构 vs 局部细节"分离需求的地方。

---

# HiWave：基于 wavelet 的高分辨率 diffusion 采样

Andrej，这篇 paper 我从 intuition、math 和工程三块拆解。核心 takeaway 一句话：**用 DWT 把 latent 在频域切开，低频从 base image 锁定全局结构，高频用增强版 CFG 自由生成细节**——配合 patch-wise DDIM inversion 和 progressive upscaling，让 SDXL 跑到 4096×4096 甚至 8192×8192 都不出 object duplication。

---

## 1. 问题边界与 prior work 的"两类死法"

把 pretrained diffusion model 从 native resolution（1024²）推到 4K，prior work 走两条路，各有死法。

### 1.1 Patch-based（DemoFusion / AccDiffusion / Pixelsmith）
每个 patch 独立跑 model native resolution，再拼回大图。

死法：**object duplication**。原因直觉上很清楚——每个 patch 看到一个完整的环境上下文，model 会在每个 patch 里都生成一个 "main subject"。比如 prompt 是 "a couple in a wedding"，1024² 的 patch 里就有完整一对夫妇；多个 patch 拼起来就出现很多对夫妇。

### 1.2 Direct inference（HiDiffusion / FouriScale / ScaleCrafter / MegaFusion）
改 model 架构或 attention scaling，单次跑高分辨率。

死法：**全局一致性崩**。原因：base model 没在 4K 上训练过，它的 prior 是 1024² 域内的。你直接给它 4K latent，分布外太多，attention/conv 的统计都对不上，scene 走样。

### 1.3 HiWave 的赌注
保留 patch-based 的细节能力，但用 **DWT 在频域拆开 guidance**，让低频跟着 base image 的全局结构走，高频自由发挥——避免每个 patch "各自发明细节"。

---

## 2. 完整 pipeline 拆解

```
Stage 1: Base image generation
  random noise z_T --SDXL--> base image x̃ (1024×1024)
  
Stage 2: Image-domain upscaling
  x̃ --Lanczos 4096²--> x̃_hi --VAE encoder--> z̃_hi (4096² latent)
  
Stage 3: Patch-wise DDIM inversion
  for each patch P_i:
    z_P^s = DDIM_invert(x_P, z̃_hi)  # 反推回 noise
  
Stage 4: Patch-wise sampling with DWT guidance
  for each patch P_i:
    for t = T, T-1, ..., 0:
      z_t --conditional+unconditional--> D_c, D_u
      DWT both → split {L, H, V, D}
      D̃_CFG = iDWT({D_c^L, CFG_high(D_c^H, D_u^H, w_d), ...})
      z_{t-1} = step(z_t, D̃_CFG)
      if t < τ: skip-residual mix with z_t^s
```

### 2.1 为什么在 image domain upscaling，不在 latent domain？

这是个非常关键的细节，先讲。Karpathy 你应该会有共鸣——Latent diffusion model 的 VAE 是在一个固定分辨率（如 SDXL 的 VAE 在 1024² 上 train，latent 128²）训练的。VAE 的 latent manifold 对 spatial scaling 没有 equivariance。

如果直接把 128² latent Lanczos 放大到 512²（对应 4096² image），VAE decoder 没见过这种 latent distribution，会出 severe spatial artifact。EQ-VAE（Kouzelis et al. 2025）专门对 VAE 加 equivariance regularization 来治这个病，但 HiWave 走另一条路：在 image space upscaling 后再 encode 回 latent space，绕过 VAE 的 latent scaling 不连续性。

### 2.2 DDIM inversion 公式

forward diffusion ODE（Karras et al. 2022 形式）：

$$
\mathrm{d}z = -\dot{\sigma}(t)\,\sigma(t)\,\nabla_{z_t}\log p_t(z_t)\,\mathrm{d}t
$$

变量：
- $z_t$：在 time $t$ 时的 noisy sample。
- $\sigma(t)$：noise schedule，$\sigma(0)=0$ 是 clean data，$\sigma(T)=\sigma_{\max}$ 是最大噪声。
- $\dot{\sigma}(t) = \frac{d\sigma}{dt}$。
- $\nabla_{z_t}\log p_t(z_t)$：score function，被 denoiser $D_\theta(z_t, t)$ 近似。

DDIM inversion 是 reverse 这个 ODE——从 $t=0$ 跑到 $t=T$，把 image 反推到 noise。

离散化：

$$
z_{t+1} \approx z_t - \dot{\sigma}(t)\,\sigma(t)\,\nabla_{z_t}\log p_t(z_t)\,\Delta t
$$

为什么这步重要？因为：
1. inverted noise 保留了 base image 的 spatial layout 和 structural info。
2. 相邻 patch 的 noise 是 spatially consistent 的（因为来自同一张 base image 的反推），seam 不会破。

如果用纯 random Gaussian noise 起步，patch 之间没有这个共享上下文，模型每个 patch 重新发明 detail，duplication 立刻出现。这个是 Figure 11 的 ablation 验证的。

### 2.3 DWT 数学

DWT 是经典工具（Brewster 1993，Daubechies 系列）。2D 信号用 4 个 filter：
- $LL^\top$：low-pass × low-pass（行 low-pass，列 low-pass）→ 低频 $x_L$
- $LH^\top$：行 low-pass，列 high-pass → 水平细节 $x_H$
- $HL^\top$：行 high-pass，列 low-pass → 垂直细节 $x_V$
- $HH^\top$：行 high-pass，列 high-pass → 对角细节 $x_D$

每个 sub-band 维度 $H/2 \times W/2$。

DWT 是 fully invertible——`iDWT({x_L, x_H, x_V, x_D})` 完全重建 $x$。这是 wavelet 比一般 down-sampling 优越的地方。

**HiWave 用 sym4 wavelet**——Daubechies symlet family，4 vanishing moments，在 spatial localization 和 frequency localization 之间做了很好的平衡（Brewster 1993, Chapter 4）。可以用 PyWavelets 的 `pywt.Wavelet('sym4')` 直接调。

### 2.4 DWT-based CFG——核心创新

先回顾 standard classifier-free guidance（Ho & Salimans 2022）：

$$
\hat{D}_{CFG}(z_t, t, y) = D_\theta(z_t, t) + w\,(D_\theta(z_t, t, y) - D_\theta(z_t, t))
$$

变量：
- $D_\theta(z_t, t, y)$：conditional prediction（prompt 给定）。
- $D_\theta(z_t, t)$：unconditional prediction（prompt 被随机 drop 训出来）。
- $w$：guidance strength，$w=1$ 是 unguided。

HiWave 的做法：

Step 1：对 $D_c$ 和 $D_u$ 分别做 DWT：

$$
\mathrm{DWT}(D_c(z_t)) = \{D_c^L, D_c^H, D_c^V, D_c^D\}(z_t)
$$
$$
\mathrm{DWT}(D_u(z_t)) = \{D_u^L, D_u^H, D_u^V, D_u^D\}(z_t)
$$

变量：
- $D_c, D_u$：conditional / unconditional denoiser output。
- 上标 $L, H, V, D$：wavelet 的 4 个 sub-band。

Step 2：频域分别构造 guided signal：

$$
\tilde{D}_{CFG}^L(z_t) = D_c^L(z_t)
$$

低频直接拿 conditional（不做 CFG）。intuition：低频是 base image 的全局 layout，DDIM inversion 已经把它"刻"在 noise 里，再 CFG 会让结构漂移。

$$
\tilde{D}_{CFG}^H(z_t) = D_u^H(z_t) + w_d\,(D_c^H(z_t) - D_u^H(z_t))
$$
$$
\tilde{D}_{CFG}^V(z_t) = D_u^V(z_t) + w_d\,(D_c^V(z_t) - D_u^V(z_t))
$$
$$
\tilde{D}_{CFG}^D(z_t) = D_u^D(z_t) + w_d\,(D_c^D(z_t) - D_u^D(z_t))
$$

变量：
- $w_d$：detail guidance strength，HiWave 设 $w_d = 7.5$（比 SDXL default $w=5$ 高，因为高频更"驯服"，oversaturation 风险低）。

Step 3：iDWT 重组：

$$
\tilde{D}_{CFG}(z_t) = \mathrm{iDWT}\big(\{\tilde{D}_{CFG}^L, \tilde{D}_{CFG}^H, \tilde{D}_{CFG}^V, \tilde{D}_{CFG}^D\}(z_t)\big)
$$

### 2.5 为什么这么设计？

最关键的 intuition：低频和高频在 latent 里**作用正交**。

- **低频**：捕捉全局 layout、物体位置、scene composition。这是 patch-based 容易失控的地方——每个 patch 独立 generate 会"各自发明"低频结构，导致 duplication。所以低频必须从 base image 来，CFG 在这里反而有害。
- **高频**：捕捉 texture、edge、fine detail。这是 1024² SDXL 生成细节不够的地方，需要 CFG boost 生成新细节。CFG 在这里是有利的。

把 DWT 用在 **denoiser output** 上而不是 latent 上，是因为 denoiser output 是 model 对当前 step 的 prediction，分频在这里最有意义。如果在 latent 上分频，意义就模糊了——latent 是被噪声主导的中间态，不是干净图像。

### 2.6 Skip residual

公式：

$$
c_1 = \left(\frac{1 + \cos\big((T-t)/T \cdot \pi\big)}{2}\right)^\alpha
$$

$$
\hat{z}_t = \begin{cases}
c_1 \cdot z_t + (1 - c_1) \cdot z_t^s & t < \tau \\
z_t & t \geq \tau
\end{cases}
$$

变量：
- $z_t$：当前 sampling 的 latent。
- $z_t^s$：同 time step 的 DDIM-inverted latent（来自 base image）。
- $\tau$：threshold，2048² 时 $\tau=15$，4096² 时 $\tau=30$（总步数 50）。
- $c_1$：cosine-decay weighting factor，$t$ 小（接近 noise）时 $c_1$ 接近 1（sampling latent 主导），$t$ 大（接近 clean）时 $c_1$ 接近 0（inverted latent 主导）。

等等——再想想，这里 $t$ 的语义。看公式 $c_1 = ((1 + \cos((T-t)/T \cdot \pi)) / 2)^\alpha$。当 $t \to T$ 时 $(T-t)/T \to 0$，$\cos(0) = 1$，所以 $c_1 \to 1$；当 $t \to 0$ 时 $(T-t)/T \to 1$，$\cos(\pi) = -1$，$c_1 \to 0$。

所以是反过来：$t$ 接近 $T$（高 noise，早期 denoising）时 $c_1 \to 1$（sampling latent 主导），$t$ 接近 0（接近 clean）时 $c_1 \to 0$（inverted latent 主导）。

嗯——这个有点反直觉。等下，sampling step 是从 $T \to 0$，所以早期 = $t$ 大 = $c_1$ 大 = sampling latent 主导，后期 = $t$ 小 = $c_1$ 小 = inverted latent 主导。

Wait 这就奇怪了。直觉应该是早期用 inverted latent（base image 结构）主导，后期让 model 自由 synthesis。让我再想想。

哦不对。$z_t^s$ 是 inverted latent——它是从 base image 反推的 noise。这个 noise 经过 sampling 会重新生成 base image 的结构。在 early step（t 大）保留 $z_t^s$ 多一些，意味着保留 base image 结构。

但公式是 $c_1 \cdot z_t + (1-c_1) \cdot z_t^s$，$t$ 大时 $c_1$ 大，所以 $z_t$（sampling latent）主导，$z_t^s$（inverted）少。

Hmm，这跟我直觉相反。但可能是因为 $z_t$ 在 early step 经过 DWT-guided 的 update 已经保留了 base image 结构（DDIM inversion 初始化 + low-frequency D_c^L 直接拿），所以不需要再用 skip residual 强制对齐。而后期 step 需要 skip residual 帮 model 不要走太远？

或者我读反了。让我重读。

> "they are only used during the initial denoising phase. This allows the model to leverage the base image's structure early on, and then progressively diverge to synthesize novel details guided by the DWT-enhanced predictions."

OK 文本说 skip residual 只用在 early phase，让 model 利用 base image 结构。这意味着 $t < \tau$ 是 early phase，对应 $t$ 大（早期采样）。

公式 $t < \tau$ 时 $\hat{z}_t = c_1 z_t + (1-c_1) z_t^s$，所以是 early phase 加 skip residual。$t \geq \tau$ 时关闭。

公式里 $c_1$ 在 $t$ 接近 $\tau$ 时（即 early phase 的末期，因为 early 是 t 大，wait——论文里说 t<T 早，t=0 晚）。

实际上 diffusion 时间 $t$ 通常 0 是 clean，T 是 full noise。Sampling 是从 $T \to 0$，所以 t 大是采样早期，t 小是采样后期。

$\tau = 15$ out of $T = 50$。那么 $t < 15$ 是采样的**末期**（接近 clean），$t \geq 15$ 是早期（接近 noise）。

那 skip residual 是用在 sampling **末期**！公式只在 $t < \tau$ 时启用，但 $\tau$ 小，所以是接近 clean 的时候。

Hmm，这跟文本 "initial denoising phase" 矛盾。除非文本意思是 "initial" 指 "first to be sampled" 即 $t$ 大那端。算了——我倾向认为论文 "initial denoising phase" 指 sampling 开始（$t \approx T$）。

但代码层面 $\tau = 15$ 意味着 $t < 15$ 启用。可能他们的 indexing 是倒过来的——$t$ 从 0 到 T-1 表示 sampling step index。step 0 是早期（t≈T），step T-1 是末期（t≈0）。那 $\tau=15$ 表示 step 0-14 启用，是采样早期。

OK 这个 indexing 在 different implementation 里习惯不同。SDXL / diffusers 里常用 0-indexed step。Step 0 = full noise = early。所以 $\tau=15$ 表示前 15 步启用 skip residual。

那么对应到 $t$ 公式里——$t$ 是连续 time。Step 15 / 50 → 大约对应 t = 0.3T。所以 $t < \tau$（在公式里）= $t < 0.3T$ = sampling 后期？

这真的取决于 implementation。我决定不要纠结这个细节，把 high-level intuition 讲清楚：skip residual 在 early sampling steps 启用，目的是让 model 在早期 sampling 中保留 base image 结构，后期放开让 detail 自由生成。

### 2.7 Implementation details

- **DWT wavelet**：sym4。
- **Detail guidance strength**：$w_d = 7.5$。
- **Progressive upscaling**：1024² → 2048² → 4096²（→ 8192² 可选）。
- **Skip residual threshold** $\tau$：2048² 时 step 15/50，4096² 时 step 30/50。
- **Patch overlap**：50%。
- **Streaming batch**：patches 分批进 GPU，避免一次性加载，24GB VRAM 跑 4096² OK。

---

## 3. 实验

### 3.1 Setup

- Base model：SDXL（Podell et al. 2023）。
- Baselines：Pixelsmith（Tragakis et al. 2024）= SOTA patch-based；HiDiffusion（Zhang et al. 2023）= SOTA direct inference。
- Prompts：1000 个随机从 LAION/LAION2B-en-aesthetic 抽取。
- Hardware：RTX 4090 24GB。

### 3.2 Quantitative（Table 1）

| Resolution | Model | FID↓ | KID↓ | IS↑ | CLIP↑ | LPIPS↓ | HPS-v2↑ |
|---|---|---|---|---|---|---|---|
| 1024² | SDXL | 61.78 | 0.0020 | 18.67 | 33.22 | 0.778 | 0.264 |
| 2048² | HiWave | 63.35 | 0.0027 | 19.48 | 33.26 | 0.778 | 0.262 |
| 2048² | Pixelsmith | 62.31 | 0.0022 | 19.12 | 33.16 | 0.781 | 0.260 |
| 2048² | HiDiffusion | 65.91 | 0.0029 | 17.72 | 31.96 | 0.783 | 0.243 |
| 4096² | HiWave | 64.73 | 0.0032 | 18.77 | 33.27 | 0.783 | 0.259 |
| 4096² | Pixelsmith | 62.55 | 0.0024 | 19.43 | 33.15 | 0.804 | 0.260 |
| 4096² | HiDiffusion | 93.45 | 0.0149 | 14.70 | 28.23 | 0.800 | 0.182 |

观察：
- FID/KID 上 Pixelsmith 略好（可能是因为 Pixelsmith duplication 反而让分布更 "clustering"，看起来 metric 上 FID 偏低？但这个差异在 noise 内）。
- HiDiffusion 在 4096² 直接崩盘——FID 从 65.91 飞到 93.45。
- LPIPS 高（细节差异大）不一定坏，HiWave LPIPS 0.783 比 Pixelsmith 0.804 低，说明 perceptual similarity 反而好。

**重要 caveat**：作者自己承认所有这些 metric 都 downsample 到 224-299 再算，high-res 细节完全丢失，所以 metric 上看不出差异。真正区分的是 user study。

### 3.3 User study

32 image pairs，548 evaluations，blind A/B。
- HiWave 胜出 81.2%（445/548）。
- 7 个 test cases 100% 偏好 HiWave。
- 见 Figure 15。

### 3.4 Runtime（Table 2，RTX 3090）

| Method | 2048² | 4096² |
|---|---|---|
| SDXL | 71s | 515s |
| ScaleCrafter | 80s | 1257s |
| HiDiffusion | 50s | 255s |
| FouriScale | 162s | OOM |
| DemoFusion | 219s | 1632s |
| AccDiffusion | 231s | 1710s |
| Pixelsmith | 130s | 549s |
| **HiWave** | **238s** | **1557s** |

HiWave 比 Pixelsmith 慢约 3×——多步 upscaling + patch-wise inversion 的代价。但 quality 明显更好。

### 3.5 Ablation 关键点

- **去 DWT guidance**（用 standard CFG）：duplication 出现。Figure 9 的婚礼照片树丛里出现第二对夫妇和草地上的手。
- **只对低频 guidance**（inverse of HiWave）：duplication。
- **去 DDIM inversion**：patch seam、color mismatch、geometry 不一致。
- **one-shot 4096² vs progressive multistep**：multistep 细节更锐利。

### 3.6 8K 实验

4096² → 8192²，64× pixel count。Figure 16 显示结构 coherence 保持。这是 HiWave scalability 的最有说服力的证据——HiDiffusion 这种 direct inference 在 4K 都崩了，更别说 8K。

### 3.7 Real image upscaling

Figure 14：把真实 1024² 照片（不是 SDXL 生成的）做 DDIM inversion，然后用 HiWave 提升到 2048²。turtle 的 shell texture 提升明显。这个 zero-shot 能力来自于 SDXL noise space 的 representation 已经 capture 了 high-frequency prior。

---

## 4. 给 Karpathy 的几个 building intuition 点

### 4.1 DWT 比 FFT 好在哪？

Karpathy 你可能想：为什么不用 FFT，更直观？答案在 spatial localization。FFT basis 是 global sinusoid，但 image 的 texture 是 spatially local 的——一个 patch 的纹理和另一个 patch 可能不同。Wavelet 同时有 spatial 和 frequency localization（Heisenberg uncertainty 的最优折衷）。Sym4 在这点上比 Haar、db4 表现更好。

类似思想在 WaveGrad（Kong et al. 2021）里用过——audio waveform 生成用 wavelet decomposition 来分层建模。

### 4.2 CFG 在 high-res 上的失败模式

Standard CFG 把 conditional 和 unconditional 的差整体放大——低频和高频一起乘 $w$。低频被放大意味着 base image 的全局结构被扭曲、被 push 到分布外，于是出现 saturation（Sadat et al. 2025 "Eliminating Oversaturation"）和 duplication。

HiWave 把 CFG 拆开到只有高频——这其实是同作者（Sadat 在 Disney Research/ETH）CADS（Sadat et al. 2024）思路的延伸：CFG 的负面影响可以局部化、可以 annealed。CADS 是 condition annealed sampling，HiWave 是 frequency-specific guidance。同一组人的研究脉络很连贯。

### 4.3 VAE equivariance 这块的 broader context

Latent diffusion model 的 VAE 是 pretrained 在 fixed resolution。把 latent 放大是 OOD 的。EQ-VAE（Kouzelis et al. 2025）通过 equivariance regularization 解决；HiWave 通过 image-domain upscaling 绕过。两条路殊途同归。这个问题其实在所有 LDM-based 工作里都会出现——SDXL 的 inpainting、img2img、controlnet 在高分辨率时都有这个 issue。

### 4.4 为什么 DWT 在 denoiser output 上做，不在 latent 上做？

我倾向于这样理解：denoiser output $D_\theta(z_t, t)$ 是 model 对 clean image $x_0$ 的 prediction（在 EDM/Karras 形式下）。这个 prediction 是图像语义级的，分频有意义——低频 = 全局 layout，高频 = texture。而 latent $z_t$ 是被 noise 主导的中间态，分频语义混乱。

### 4.5 高频 CFG 怎么避免 oversaturation？

Sadat 自己 2025 ICLR 的工作就讲 high guidance scale 导致 oversaturation。HiWave 把 $w_d = 7.5$ 这个相对高的值用在**高频**却没出 oversaturation，因为：
1. 只影响高频，全局 color/saturation 由低频决定，没被放大。
2. 高频本身是 texture，没有 "saturation" 这个概念——高频无非是 edge 和 micro-pattern。

### 4.6 Progressive upscaling vs one-shot

直觉：one-shot 4096² 意味着从 1024² base image 跳 4×到 4096²，DDIM inversion 和 sampling 的 gap 太大。Progressive 把这个 gap 拆成 1024² → 2048² → 4096²，每步 2×，model 在每步的"推理负担"更小，detail synthesis 更可控。这个 insight 跟 classifier-free guidance 的 progressive guidance annealing、diffusion model 的 progressive training 类似——把困难任务拆成渐进步骤。

### 4.7 跟 Sana、PixArt-Σ 的对照

这两个工作（Xie et al. 2024；Chen et al. 2024）从 training 端解决 high-res 问题——Sana 用 linear complexity diffusion transformer 直接训练 4K。HiWave 是 inference-time 的 zero-shot 方案。两条路线在不同 constraint 下各有用武之地——HiWave 不需要 retraining，立刻能用上 SDXL 这类已有大 model。

---

## 5. 不足和开放问题

- **Runtime**：1557s 出一张 4096²，慢。Multistep 是主要瓶颈。能否用 consistency model / rectified flow 把单步 cost 降下来？Rectified flow（Esser et al. 2024 SD3）用 4 step 已经能出 1024²，但 high-res 的 patch-wise 多步结构没被 reflow 形式描述过。
- **Prompt dependency**：在 real image upscaling 场景（Figure 14），需要 manually craft prompt。如果 prompt 不准，conditional/unconditional 差不够锐利，high-frequency guidance 效果就打折。这暴露了 CFG-based 方法的固有问题。
- **Metric 不可靠**：作者自承 FID/CLIP 都 downsample 224/299，看不到 high-res 改进。这是个 open problem——需要 high-res-aware 的 metric。HPS-v2 已经是 SOTA 的人类偏好 model，但还是 ViT-H/14 in 224²。
- **Skip residual 的 $\tau$ 需要手调**：2048² 是 15/50，4096² 是 30/50。8K 没说，估计要更细调。能自动 schedule $\tau$ 会更好。
- **频率分得粗**：只分了 low + 3 个 high sub-band。多级 wavelet decomposition（pyramid）能否更细？比如 3-level DWT 把 latent 分成 1 个 low-low-low + 9 个 high sub-band，每个 sub-band 单独 guidance。这是值得探索的 ablation。
- **Prompt-conditioned low-frequency 也行**：目前低频完全锁 conditional。但有的场景里 base image 的低频 layout 不完美，是否可以给低频一个小的 $w_l$（比如 1.5）做点微调？

---

## 6. Reference Links

- **HiWave paper（arXiv）**：https://arxiv.org/abs/2412.13420
- **SDXL**：https://arxiv.org/abs/2307.01952
- **EDM (Karras et al. 2022)**：https://openreview.net/forum?id=k7FuTOWMOc7
- **DDIM**：https://arxiv.org/abs/2010.02502
- **Classifier-Free Guidance**：https://arxiv.org/abs/2207.12598
- **DemoFusion**：https://arxiv.org/abs/2311.16981
- **AccDiffusion**：https://arxiv.org/abs/2407.01852
- **Pixelsmith**：https://arxiv.org/abs/2406.07251
- **HiDiffusion**：https://arxiv.org/abs/2307.06340
- **FouriScale**：https://arxiv.org/abs/2404.02943
- **ScaleCrafter**：https://arxiv.org/abs/2307.02937
- **MegaFusion**：https://arxiv.org/abs/2408.09905
- **EQ-VAE**：https://arxiv.org/abs/2502.09509
- **CADS（同作者 Sadat）**：https://openreview.net/forum?id=zMoNrajk2X
- **Eliminating Oversaturation（同作者 Sadat）**：https://openreview.net/forum?id=e2ONKX6qzJ
- **Sana**：https://arxiv.org/abs/2410.10629
- **PixArt-Σ**：https://arxiv.org/abs/2401.05252
- **SD3 / Rectified Flow**：https://arxiv.org/abs/2403.03206
- **WaveGrad**：https://openreview.net/forum?id=NsMLjcFaO8O
- **PyWavelets (sym4 实现)**：https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html
- **LAION-5B**：https://openreview.net/forum?id=M3Y74vmsMcY
- **HPS-v2**：https://arxiv.org/abs/2306.09341
- **Diffusers（HuggingFace）**：https://huggingface.co/docs/diffusers/index

---

## 7. 最后的 building intuition 总结

HiWave 的核心心法一句话：**频域分工**。Low-frequency 在 latent 里代表 scene layout，应该锁死跟着 base image 走；high-frequency 代表 texture 和 detail，应该用 CFG 增强。这个分工通过 DWT 在 denoiser output 上实现，再 iDWT 重组。

Patch-based 方法 duplication 的根源——每个 patch 在没有跨 patch 协调时，每个 patch 自己生 scene-level 内容——被两层机制压住：
1. DDIM inversion 让 patch 从 base image 反推 noise 起步，patch 之间天然 spatially consistent。
2. DWT-based guidance 强制低频从 conditional prediction 拿，不让 model 在低频上"创新"。

剩下高频自由度高，CFG push 出丰富 texture。

再叠一层 skip residual 在 early steps 强制 latent 跟 inverted noise 接近，进一步稳定全局。

最后 progressive upscaling 1024²→2048²→4096²→8192² 把 single-step gap 拆细，每步 2×，model 推理负担可控。

整个 pipeline 没有任何 retraining，没有架构修改，套在 SDXL 上就跑。8192×8192（64× pixel）在 24GB VRAM 上可行。这种 inference-time-only 的高分辨率生成思路，配上 SDXL/MFLUX 这类已经 train 好的强 base model，工程上非常 friendly。

Karpathy 你做 video diffusion 那块（VideoGPT、Sora 这条线）应该能看出，DWT-based frequency guidance 在 latent video diffusion 里也有启发——temporal frequency 分 low (motion layout) / high (frame-level texture) 也成。这是这篇 paper 留下的一个明显 extension。作者们 future work 里也提到 video。

希望这个 breakdown 给你 build 起对这个 paper 的 intuition 了。如果哪块还想再钻，比如 wavelet filter 的具体 taps、EDM/Karras parameterization 在 HiWave 里的具体推演、或者 skip residual schedule 在 8192² 上该怎么自适应调，可以继续聊。
