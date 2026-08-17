---
source_pdf: Open-Sora.pdf
paper_sha256: 7e23af50b745489d1bc2b194f884843af6b823d1b1912c25718d12de7d9d82c5
processed_at: '2026-08-06T00:09:58-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Open-Sora：用人话讲讲这事儿

Andrej，咱们坐下来聊，不搞学术那一套，就讲这 paper 到底干了啥，为啥这么干，以及哪些地方我觉得聪明、哪些地方我觉得 hacky。

---

## 一、先讲个 big picture

Sora 那会儿 release 一个 demo，整个 community 都炸了，大家都在猜它怎么做的。OpenAI 就给了一篇 blog post，技术细节基本没有，就丢几个 keyword：spatiotemporal patches、scaling、world simulator。然后全 world 都在 reverse-engineer。

Open-Sora 这帮人（HPC-AI Tech，Yang You 的 group，做 ColossalAI 那个）大概用 3 个月时间，迭代了三个 version（1.0 → 1.1 → 1.2），基本上把 Sora report 里提到的每个 hint 都 engineering 化了，而且全部 open source。

所以这篇 paper 的价值不在于"发明了什么新东西"，而在于"把一个 black box 变成了 reproducible recipe"。这对 community 来说极其 valuable，因为你终于有个 baseline 可以 iterate 了。

Reference: [Open-Sora GitHub repo](https://github.com/hpcaitech/Open-Sora), [Sora blog](https://openai.com/research/video-generation-models-as-world-simulators)

---

## 二、Video Generation 到底在干啥

咱们从最 basic 的 level 讲。你想 generate 一个 video，本质上是想 sample 一个 distribution $p(x)$，其中 $x$ 是一个 video tensor，shape 是 $T \times H \times W \times 3$（时间、高、宽、RGB channel）。

直接 sample 这个 distribution 完全 intractable，所以用 diffusion 的思路：先 define 一个 forward process 把 data 慢慢加 noise 变成 pure Gaussian noise，然后 learn 一个 reverse process 把 noise 慢慢 denoise 回 data。

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

这里 $x_0$ 是 clean video，$x_t$ 是加了 $t$ 步 noise 的 video，$\bar{\alpha}_t$ 是 noise schedule（控制 signal/noise ratio 随 $t$ 变化），$\epsilon$ 是 random Gaussian noise。

Model 学的是预测 $\epsilon$（或者预测 velocity $v = x_1 - x_0$，Open-Sora 用的是后者），然后 inference 时从 $x_T \sim \mathcal{N}(0,I)$ 出发，一步步 denoise 到 $x_0$。

这跟 image diffusion 完全一样，唯一的区别是 $x$ 多了一个 temporal dimension $T$。但这个 $T$ 带来了巨大的 engineering challenge，下面讲。

---

## 三、第一个大问题：Video 太大了

一个 16s、720p、30fps 的 video，tensor shape 大概是 $480 \times 1280 \times 720 \times 3$，pixel 数量是 image 的 480 倍。你直接把这个扔进 DiT 做 attention，memory 直接爆掉。

所以你需要一个 **compressor** 把 video 压到一个小很多的 latent space，在 latent space 上做 diffusion，最后再 decode 回 pixel space。这就是 VAE（Variational Autoencoder）的作用。

### 3.1 Open-Sora 1.0/1.1 的 hacky 做法

他们一开始用 Stability-AI 的 2D VAE（就是 Stable Diffusion 那个），spatial 压缩 8×8。但 temporal 怎么办？他们的做法非常 brute force：**每 3 帧抽 1 帧**。

比如你原本 30fps 的 video，抽完之后变成 10fps，然后生成完了再插帧回去。这导致生成的 video 动起来一卡一卡的，不流畅。用 paper 的话说就是"low temporal fluency due to a reduction in generated FPS"。

### 3.2 Open-Sora 1.2 的 smart 做法

Key insight 非常漂亮：**2D VAE 压缩完之后，temporal 相邻的 latent features 仍然高度 correlated**。

你想想，2D VAE 是对每一帧 independent 压缩的，但相邻帧的 spatial content 几乎一样（就动了一点点），所以压缩出来的 latent 也几乎一样。这就意味着 temporal dimension 上有巨大的 redundancy，可以再压一次。

所以他们 stack 了一个 3D VAE 在 2D VAE 上面：

```
Video (T×H×W×3)
    → 2D VAE (每帧独立压缩，8× spatial)  → (T × H/8 × W/8 × 4)
    → 3D VAE (temporal 4× 压缩)            → (T/4 × H/8 × W/8 × C)
```

总压缩比：spatial 64×，temporal 4×，总共 256×。

一个 16s 720p 30fps video，原始 pixel 数 ~$480 \times 1280 \times 720 \times 3 \approx 1.3 \times 10^9$，压完之后 latent 大小 ~$120 \times 160 \times 90 \times C$，如果 $C=16$ 那就是 ~$2.8 \times 10^7$，压缩了约 46×（token 数层面）。这才能 fit 进 DiT。

### 3.3 3D VAE 怎么训练的

这部分我觉得是整篇 paper 最 clever 的 engineering。训练一个 3D VAE from scratch 很贵，而且容易训崩。他们用了三阶段：

**Stage 1（0-380k steps）**：2D VAE frozen，3D VAE 学着重建 2D VAE 的 output。加一个 identity loss，强制 3D VAE 在初始时 behave like identity mapping。

直觉讲：2D VAE 已经有很好的 spatial representation 了，你不想破坏它。所以先让 3D VAE "啥也别干"，就当个 pass-through，然后慢慢 learn temporal compression。

$$\mathcal{L}_{stage1} = \|Dec_{3D}(Enc_{3D}(z_{2D})) - z_{2D}\|^2 + \lambda \|Enc_{3D}(z_{2D}) - z_{2D}\|^2$$

这里 $z_{2D}$ 是 2D VAE 的 latent，$Enc_{3D}$/$Dec_{3D}$ 是 3D VAE 的 encoder/decoder。第一项是 reconstruction loss，第二项是 identity loss（让 encoder output 直接 align 到 2D latent）。

**Stage 2（380k-640k steps）**：去掉 identity loss，让 3D VAE 自由学习 temporal modeling。

**Stage 3（640k-1.2M steps）**：不再重建 2D VAE features，直接 end-to-end 重建原始 video pixels。同时引入 mixed-length training（random video length up to 34 frames + zero-padding）。

为什么 Stage 3 要 mixed-length？因为之前固定 17-frame clip 训练，结果发现 non-standard length 的 video 会 blurry。mixed-length 解决了这个问题。

还有一个 detail：他们用了 **causal convolution**。causal conv 的意思是 temporal 方向上只看过去，不看未来（像 autoregressive language model）。这为未来做 frame-by-frame generation 或 video extension 埋了伏笔。

最终 VAE 性能：PSNR 30.59，SSIM 0.88，跟 Open-Sora-Plan 的 3D VAE 差不多，但 computational cost 更低。

Reference: [SDXL VAE](https://arxiv.org/abs/2307.01952), [Magvit-v2 VAE architecture](https://arxiv.org/abs/2310.05737)

---

## 四、第二个大问题：Attention 的 complexity

VAE 解决了"video 太大"的问题，但 latent 仍然有 $T/4 \times H/8 \times W/8$ 个 tokens。对 16s 720p 来说大概是 $120 \times 160 \times 90 \approx 1.7M$ tokens。

如果做 full self-attention，complexity 是 $O(N^2)$ where $N = 1.7M$，那就是 $2.9 \times 10^{12}$，完全不可能。

### 4.1 STDiT 的解法：spatial 和 temporal attention 解耦

核心 idea 很简单：**一个 frame 内部的 spatial relationship 跟 across frames 的 temporal relationship 是两回事，没必要 joint attend**。

所以他们设计了两种 attention：

**Spatial self-attention**：每个 frame 内部，所有 spatial tokens 互相 attend。Complexity: $O(T \cdot (HW)^2)$

**Temporal self-attention**：每个 spatial position，across all frames 的 tokens 互相 attend。Complexity: $O(HW \cdot T^2)$

总 complexity 从 $O(T^2 H^2 W^2)$ 降到 $O(T \cdot H^2 W^2 + HW \cdot T^2)$。

用具体数字算一下：$T=120, H=90, W=160$
- Full attention: $(120 \times 90 \times 160)^2 = 2.98 \times 10^{12}$
- STDiT: $120 \times 90^2 \times 160^2 + 90 \times 160 \times 120^2 = 2.49 \times 10^{10} + 2.07 \times 10^8 \approx 2.5 \times 10^{10}$

降了 ~120×。这才让 16s video generation 变得 tractable。

### 4.2 从 PixArt-α 出发，zero-init temporal attention

另一个 key decision：不从 scratch 训练，而是用 **PixArt-α**（一个 image diffusion transformer，580M params）作为 backbone。

新增的 temporal attention projection layer 初始化为 **zero**：

$$W_{temporal}^{init} = \mathbf{0}$$

这意味着训练刚开始时，temporal attention 的 output 全是 0，model 完全 behave like 原 PixArt（pure image generator）。然后 gradually learn temporal dynamics。

这个 trick 的 intuition：**image generation 是 video generation 的 degenerate case**（$T=1$ 的 video 就是 image）。所以你不想丢失 image generation 的能力，zero-init 保证 temporal module 是 "additive" 的，只会 add temporal understanding 而不会破坏 spatial understanding。

参数量从 580M → 1.1B（temporal attention 翻倍了）。

### 4.3 两个 stabilization trick

**RoPE for temporal attention**：用 Rotary Position Embedding 替代 sinusoidal positional encoding。

RoPE 的核心：对 position $m$ 的 query/key vector，在每个 2D 子空间 $(x_{2i}, x_{2i+1})$ 上做 rotation：

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x'_{2i+1} \end{pmatrix}$$

变量解释：
- $m$：frame index（position），$m=0,1,2,...,T-1$
- $i$：dimension pair index，$i=0,1,...,d/2-1$，$d$ 是 head dimension
- $\theta_i = 10000^{-2i/d}$：frequency，不同 dimension pair 对应不同 frequency
- $x_{2i}, x_{2i+1}$：query/key 的第 $2i$ 和 $2i+1$ 个 component

RoPE 的好处是 **relative position encoding**：$q_m^T k_n$ 只依赖 $m-n$（相对距离），不依赖绝对位置。这对 video 的 temporal structure 天然合适——"第 5 帧跟第 3 帧的关系"比"第 5 帧的绝对位置"重要。

**QK-Normalization**：所有 attention 的 Q 和 K 做 L2 normalize：

$$Q' = \frac{Q}{\|Q\|_2 + \epsilon}, \quad K' = \frac{K}{\|K\|_2 + \epsilon}, \quad \epsilon = 10^{-15}$$

直觉：bf16 training 下，attention logits $QK^T$ 容易 explode（因为 bf16 的 dynamic range 有限），导致 softmax saturate 或 gradient spike。QK-norm 把 Q/K 的 norm 压到 1，logits 的 scale 就被控制住了。

$\epsilon = 10^{-15}$ 这么小是为了避免 division by zero 但又几乎不改变 normalization 效果。

Reference: [RoPE paper](https://arxiv.org/abs/2104.09864), [SD3 QK-norm](https://arxiv.org/abs/2403.03206), [PixArt-α](https://arxiv.org/abs/2310.00426), [Latte STDiT](https://arxiv.org/abs/2401.03048)

---

## 五、第三个问题：怎么支持 image-to-video

Text-to-video 很 flexible，但很多时候你想用一张 image 作为起点，让它动起来（image-to-video），或者给一段 video 让它 continue（video-to-video）。

### 5.1 Mixed timestep masking

这个设计我觉得很 elegant。Diffusion model 的核心是每个 sample 有一个 timestep $t$，表示"这个 sample 有多 noisy"。$t=0$ 是 clean，$t=T$ 是 pure noise。

对于 image-to-video，你想让 conditioning image 是 clean 的（$t=0$），其他 frames 是 noisy 的（$t>0$）。但预训练 model 从没见过同一个 sample 里不同 frames 有不同 timestep，直接用会崩。

解决方法：训练时 50% 的 samples 做 random masking，unmasked frames 设 $t=0$，masked frames 保留 normal timestep。

Masking pattern 包括：first frame、first k frames、last frame、last k frames、first+last k frames、fully random。

直觉：这跟 UL2 在 language model 里用 different span corruption modes 来 unify NLU/NLG 是一回事。你通过 input format 的 variation 让一个 model 同时学会多种 task，而不是训多个 model。

10k steps 后 model 就学会了 conditioning，且 text-to-video 性能不受影响。30% masking ratio 太低则学不进去。

### 5.2 Score conditioning

Caption 后面 append 额外 control signal：

```
"[Original Caption] aesthetic score: 5.5, motion score: 10, camera motion: pan left"
```

Inference 时你可以手动调这些 score 来 control 生成。比如想要高美学质量就调高 aesthetic score，想要画面动得大就调高 motion score。

Camera motion 用 13k 高置信 clip 人工标注，包括 pan left/right、zoom in/out、tilt up/down 等。

---

## 六、训练策略：怎么用 35k H100 hours 训出来

这是整篇 paper 最 "engineering art" 的部分。Sora 估计烧了几百万 H100 hours，Open-Sora 只用了 35k，差了 ~100×。怎么做到的？

### 6.1 从 Image Model 优雅过渡

起点是 **PixArt-Σ 2K checkpoint**（一个高质量 image generation model）。然后 8 个 sequential stages：

1. **Multi-resolution image gen** (20k steps)：让 model 学会 144p 到 2K 多分辨率
2. **加 QK-norm** (18k steps)：stabilize training
3. **DDPM → Rectified Flow** (10k steps)：换 training objective
4. **Rectified Flow 优化** (33k steps)：加 logit-norm timestep sampling + resolution-aware sampling
5. **减小 AdamW epsilon** (8k steps)：$\epsilon = 10^{-15}$，配合 QK-norm
6. **换 VAE + 加 FPS conditioning** (25k steps)：用 Open-Sora 自己的 3D VAE
7. **加 temporal attention blocks** (3k steps)：zero-init，先在 image 上训
8. **Temporal blocks masking** (38k steps)：只在 video 上训 temporal attention

每个 stage 只改一个东西，避免 training destabilize。这种 "one change at a time" 的策略在 large model training 里非常重要。

### 6.2 Flow Matching 替代 DDPM

DDPM 是离散时间的 Markov chain，sampling 需要 100 步。Rectified Flow 是连续时间的 ODE，trajectory 是直线，sampling 只需 30 步。

Rectified Flow 的 training objective：

$$\mathcal{L}_{RF} = \mathbb{E}_{t \sim \mathcal{U}(0,1), x_0 \sim \mathcal{N}(0,I), x_1 \sim p_{data}} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$$

变量解释：
- $t \sim \mathcal{U}(0,1)$：continuous timestep，uniform sampled from [0,1]
- $x_0 \sim \mathcal{N}(0,I)$：pure Gaussian noise（起点）
- $x_1 \sim p_{data}$：real data（终点）
- $x_t = (1-t)x_0 + t x_1$：linear interpolation between noise and data
- $v_\theta(x_t, t)$：model 预测的 velocity field
- $(x_1 - x_0)$：ground truth velocity（从 noise 到 data 的直线方向）

直觉：DDPM 学的是 "每一步加多少 noise 去掉"，Rectified Flow 学的是 "从 noise 到 data 的直线方向"。直线意味着你可以直接大步走，不用一步步挪。

还用了 **logit-norm timestep sampling**：不是 uniform sample $t$，而是让 $t$ 的分布集中在 mid-range（model 学得最难的地方）。

### 6.3 Bucket 策略处理多分辨率

Sora report 强调"保持原始 resolution/aspect ratio 训练"。三种方案：

1. **NaViT**：patch packing + masking，flexible 但不能用 Flash Attention
2. **FiT**：padding 到 max size，简单但浪费 memory
3. **Bucket**：预定义固定大小桶，batch 内统一

Open-Sora 选 Bucket。每个 bucket 是 $(resolution, \#frames, aspect\_ratio)$ 三元组。还有 keep probability（高分辨率 video 按 probability 下采样到低 bucket 省 compute）和 batch size（per-bucket 调整平衡 GPU load）。

直觉：Bucket 是 "curriculum learning by resolution"——低分辨率 bucket 见多 data 学 motion prior，高分辨率 bucket 见少 data 学 visual fidelity。

### 6.4 三阶段 Data Curriculum

| Stage | Dataset | Data | Resolution | Steps |
|-------|---------|------|------------|-------|
| 1 | Webvid-10M | 40k hours, low-res + watermark | 240p-360p | 30k (2 epochs) |
| 2 | Panda-70M filtered | 20M clips, 41k hours | 360p-480p | 23k (0.5 epoch) |
| 3 | Curated high-quality | ~2M clips, 5k hours | 720p-1080p | 15k (2 epochs) |

直觉：先用大量低质量 data 学 motion prior（怎么动），再用少量高质量 data 学 visual fidelity（怎么好看）。跟 LLM 的 "pretrain on web → SFT on curated data" 完全 analog。

Stage 1 用 Webvid-10M，这 dataset 全是 stock footage with watermark，quality 差，但量大，适合 warmup 学 basic motion。

Stage 2 用 Panda-70M 的 high-aesthetic subset（aesthetic score > 4.5），medium quality，学更 complex 的 motion。

Stage 3 用 curated collection（MiraData + Vript + Pexels/Pixabay），high quality，学 high-resolution detail。25% masking ratio 用于 support image-to-video。

Reference: [Flow Matching](https://arxiv.org/abs/2210.02747), [Rectified Flow](https://arxiv.org/abs/2209.03003), [Webvid-10M](https://arxiv.org/abs/2103.10563), [Panda-70M](https://arxiv.org/abs/2402.19479)

---

## 七、Data Pipeline：脏活累活

Data pipeline 是 video generation 最 unsexy 但最 critical 的部分。Open-Sora 的 pipeline：

### 7.1 Source

30M video clips，80k hours。来源：
- Webvid-10M（10M，low-res + watermark）
- Panda-70M（用了 20M high-quality subset）
- HD-VG-130M（BLIP-2 caption，quality 差）
- MiraData（77k long video，games + city exploration）
- Vript（400k densely annotated）
- Inter4K（1K 4K clip）
- Pexels/Pixabay/Mixkit（free-licensed high quality）
- LAION subset + Unsplash-lite（~3M image，joint training 用）

### 7.2 Filtering

1. **PySceneCut**：scene detection，把长 video 切成 clip
2. **Aesthetic Score**：LAION scorer，3 frames 取平均，过滤丑的
3. **Optical Flow Score**：UniMatch model，过滤 motion 太小的（static video 没意义）
4. **OCR**：DBNet++ via MMOCR，过滤 text 太多的（news broadcast、广告）

### 7.3 Captioning

- **PLLaVA 13B**：开源 video captioning model，每 video 取 4 frames，spatial pooling 2×2
- **GPT-4V**：API，贵但 quality 好，部分 dataset 用
- **Camera motion detection**：optical flow 检测 camera motion，append 到 caption（"pan left"、"zoom in" 等）

PLLaVA 的 issue：检测不到 camera motion，所以单独用 optical flow 补。这反映了 video captioning model 的 limitation——对 camera-level 运动不敏感。

Reference: [PLLaVA](https://arxiv.org/abs/2404.16994), [UniMatch](https://arxiv.org/abs/2211.03003), [PySceneDetect](https://github.com/Breakthrough/PySceneDetect), [MMOCR](https://github.com/open-mmlab/mmocr)

---

## 八、结果怎么样

### 8.1 VBench Score

VBench 是一个 automated video generation benchmark，评 2s 240p video。

| Model | Total (%) | Quality (%) | Semantic (%) |
|-------|-----------|-------------|--------------|
| **Open-Sora 1.2** | **79.76** | **81.35** | **73.39** |
| Open-Sora 1.1 | 75.66 | 77.74 | 67.36 |
| Open-Sora 1.0 | 75.91 | 78.82 | 64.28 |
| OpenSoraPlan V1.3 | 77.23 | 80.14 | 65.62 |
| Show-1 | 78.93 | 80.42 | 72.98 |
| Latte | 77.29 | 79.72 | 67.58 |

1.2 相比 1.1 的提升来自：3D VAE（temporal compression）、Rectified Flow、multi-stage data curriculum。

Open-Sora 1.2 在 open-source model 里是 SOTA。

### 8.2 VAE Reconstruction

| Model | SSIM↑ | PSNR↑ |
|-------|-------|-------|
| Open-Sora 1.2 | 0.880 | 30.590 |
| OpenSoraPlan 1.1 | 0.882 | 29.890 |

PSNR 30.59 在 256× compression 下相当不错。SSIM 0.88 说明 structural similarity 保持得好。

### 8.3 Training Loss 曲线

Figure 8/9 显示 validation loss 在不同 length（2s/4s/8s/16s）和 resolution（144p-720p）下持续下降，VBench score 也持续上升。说明 training 没有 plateau 或 diverge。

Reference: [VBench](https://arxiv.org/abs/2403.14822), [Open-Sora-Plan](https://arxiv.org/abs/2412.00131)

---

## 九、我的 Thoughts 和 Speculation

### 9.1 这篇 paper 真正的贡献

不是任何 single technique，而是 **系统性 engineering**。每个 component（VAE、STDiT、conditioning、training strategy）单独看都不是全新 idea，但组合在一起形成一个 reproducible pipeline，这才是 value。

对 community 来说，这相当于给了大家一个 "reference implementation"。你可以在上面 iterate：换更好的 VAE、试 sparse attention、加 RLHF、scale up data……每个 direction 都有 baseline 可以比较。

### 9.2 跟 Sora 的 gap

| 维度 | Open-Sora 1.2 | Sora（推测） |
|------|---------------|-------------|
| Compute | 35k H100 hours | ~millions |
| Data | 80k hours | ~millions of hours |
| Length | 16s | 分钟级 |
| Resolution | 720p | 1080p+ |
| World simulation | 弱 | 强（physics、object permanence） |
| Motion control | camera motion only | complex interaction |

Gap 主要在 scale 和 data quality。Sora 一定用了海量 internal data（YouTube 级别？），而且可能加了 RLHF 或 preference optimization 来 align human aesthetic。

### 9.3 Sora 可能还有啥没公开的

基于 Open-Sora 的复现经验，我 speculate Sora 可能还有：

1. **更激进的 VAE compression**：可能 temporal 8× 甚至 16×，支持更长 video
2. **Joint image-video training with huge image ratio**：image data 可能占 50%+ training，帮住 visual quality
3. **DPO/RLHF on human preference**：alignment 到 aesthetic preference 和 physical plausibility
4. **Autoregressive frame extension**：causal VAE + autoregressive generation 支持无限长 video
5. **Sparse attention 或 sliding window**：处理 long video 的 temporal attention
6. **Compute-optimal scaling law analysis**：可能做了 Chinchilla-style 的 scaling law 研究
7. **Multi-modal conditioning**：除了 text，可能还有 audio、sketch、depth 等 conditioning

### 9.4 未来方向

1. **Unified tokenizer + LM paradigm**：Magvit-v2 的 discrete tokenizer 路线可能 eventually beat diffusion，因为 LM 的 scaling law 更成熟
2. **Long video via KV cache + causal generation**：causal VAE 已经 enable 了这条路
3. **3D RoPE**：spatial 也用 RoPE，支持 native 3D position encoding
4. **World model integration**：video generation + action conditioning → true simulator
5. **Test-time compute**：inference 时用 search/optimize 来 improve quality（类似 AlphaGo 的 MCTS 思路）
6. **Hierarchical generation**：先 generate low-res long video，再 super-res + interpolate，避免一次性生成的高 complexity

### 9.5 最让我 impressed 的 design choice

**3D VAE 的 staged training with identity loss**。这个 trick 太聪明了——你有一个 pretrained 2D VAE，想 extend 到 3D 但不想破坏 spatial representation。Identity loss 让 3D VAE 初始时 behave like identity，然后 gradually learn temporal compression。这跟 LoRA 的 zero-init 思路一脉相承，都是 "add new capability without destroying existing capability"。

**Mixed timestep masking for conditioning**。用一个 model 同时支持 T2V/I2V/V2V，通过 input format variation 而不是 multi-head/multi-model。这跟 UL2 的 philosophy 一致，非常 elegant。

**Zero-init temporal attention**。从 image model 平滑过渡到 video model，zero-init 保证 temporal module 是 additive 的。这个 trick 在 LLM 的 adapter/LoRA 里很常见，但用在 video generation 里同样 powerful。

### 9.6 我觉得 hacky 的地方

**Bucket strategy**。虽然 engineering 上 simple 且 efficient，但理论上 NaViT 的 packing + masking 更 elegant，只是 Flash Attention 不支持。如果未来 Flash Attention 支持 variable-length attention，NaViT 会是更好的选择。

**每 3 帧抽 1 帧（1.0/1.1）**。虽然 1.2 改了，但前两个 version 的这个做法真的很 brute force，说明 temporal compression 在 video generation 里是个 hard problem。

**VBench 评测**。只评 2s 240p video，无法 capture long video coherence 和 physical plausibility。Community 需要更好的 benchmark。

---

## 十、给 Karpathy 的 TL;DR

用一句话讲：**Open-Sora 把 Sora 的每个 hint engineering 化了，用 35k H100 hours 和 80k hours data，通过 3D VAE + STDiT + flow matching + staged training，复现了一个 16s 720p 的 open-source video generator**。

核心 insight：
- 3D VAE：leverage pretrained 2D VAE，stack 一个 temporal compressor，identity loss 做 warmup
- STDiT：spatial/temporal attention 解耦，complexity 降 120×
- Zero-init：从 image model 平滑过渡，temporal module 是 additive
- Flow matching：trajectory 拉直，sampling 从 100 步降到 30 步
- Staged adaptation：one change at a time，avoid training destabilize
- Data curriculum：low-quality warmup → high-quality fine-tune
- Mixed masking：一个 model 支持 T2V/I2V/V2V

这篇 paper 的 value 在于 **recipe 而非 invention**。它给了 community 一个 reproducible baseline，让大家可以在此基础上 iterate，而不需要 reverse-engineer OpenAI 的 black box。

主要 references：
- [Open-Sora GitHub](https://github.com/hpcaitech/Open-Sora)
- [Open-Sora project page](https://hpcaitech.github.io/Open-Sora/)
- [Sora blog](https://openai.com/research/video-generation-models-as-world-simulators)
- [PixArt-α](https://arxiv.org/abs/2310.00426)
- [PixArt-Σ](https://arxiv.org/abs/2403.04672)
- [Latte](https://arxiv.org/abs/2401.03048)
- [SD3](https://arxiv.org/abs/2403.03206)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Magvit-v2](https://arxiv.org/abs/2310.05737)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [UL2](https://arxiv.org/abs/2205.05131)
- [VBench](https://arxiv.org/abs/2403.14822)
- [NaViT](https://arxiv.org/abs/2307.06304)
- [Panda-70M](https://arxiv.org/abs/2402.19479)
- [PLLaVA](https://arxiv.org/abs/2404.16994)
- [UniMatch](https://arxiv.org/abs/2211.03003)

Andrej，如果你想 deep dive 任何 specific part（比如 3D VAE 的 causal conv 细节、flow matching 的 ODE solver 选择、或者 bucket strategy 的 load balancing math），尽管问。

---

# Open-Sora 深度技术讲解

Andrej，这篇 paper 是 HPC-AI Tech 团队对 OpenAI Sora 的一次系统性开源复现，核心价值在于把 Sora report 里那些"只言片语"的技术线索全部 engineering 化了。我会从 architecture、data、training 三个 axis 展开讲，同时 build up 你的 intuition。

---

## 1. 整体框架的 Intuition

Open-Sora 的核心 pipeline 可以概括为：

```
Video → 3D Autoencoder (spatial 8x8 + temporal 4x compression) → latent tokens
                                                                     ↓
Text → T5 encoder → text embeddings → cross-attention → STDiT → denoised latent → 3D Decoder → Video
```

关键 insight：**把 video generation 看成是一个 "learned world simulator" 在压缩时空 latent space 上的 denoising 过程**。Sora report 强调 "video patches" 概念，Open-Sora 用 3D VAE 把这个 concept 落地了。

Reference: [Sora technical report](https://openai.com/research/video-generation-models-as-world-simulators)

---

## 2. 3D Autoencoder：从 2D VAE 到时空压缩

### 2.1 为什么需要 3D VAE

Open-Sora 1.0/1.1 用的是 Stability-AI 的 2D VAE（84M params），spatial 压缩 8×8，temporal 维度靠"每 3 帧抽 1 帧"硬砍。这导致 generated FPS 下降，视频不流畅。

1.2 版本的 key insight：**2D VAE 压缩后，temporal 相邻的 features 仍然高度相关**。所以可以"搭便车"——在 2D VAE 的 latent 上再加一个 temporal compressor。

### 2.2 架构细节

架构是 stacked 的：

```
Input video x ∈ R^{T×H×W×3}
    ↓ 2D VAE (frozen, from SDXL, 84M)
z_spatial ∈ R^{T×(H/8)×(W/8)×4}
    ↓ 3D VAE (Magvit-v2 style, 300M)
z ∈ R^{(T/4)×(H/8)×(W/8)×C}
```

总 compression factor：spatial 64×，temporal 4×，总体 256× 压缩比。这 extremely aggressive，但 PSNR 仍然有 30.59，SSIM 0.88。

### 2.3 三阶段训练策略

这个 staged training 非常聪明，intuition 是"渐进式 unlock 难度"：

**Stage 1 (0–380k steps, 8 GPUs, 2D VAE frozen)**
- Objective: 3D VAE 重建 2D VAE 压缩后的 features
- Identity loss: 强制 3D VAE 的输出 align 到 2D VAE 的 features

Identity loss 的直觉：让 3D VAE 在初始阶段 behave like 一个 "identity + temporal compression" 的 operator，避免破坏 2D VAE 已经学好的 spatial semantics。形式上：

$$\mathcal{L}_{identity} = \| \text{Dec3D}(\text{Enc3D}(z_{2D})) - z_{2D} \|_2^2$$

其中 $z_{2D}$ 是 2D VAE 的 latent，Dec3D/Enc3D 是 3D VAE 的 decoder/encoder。

**Stage 2 (380k–640k steps)**
- 去掉 identity loss，让 3D VAE 自由学习 temporal modeling

**Stage 3 (640k–1.2M steps, 24 GPUs)**
- 直接重建原始 video pixels（end-to-end）
- 引入 mixed-length training：random video length up to 34 frames + zero-padding

这里有个很重要的 trick：**causal convolution**。Magvit-v2 VAE 里用 causal conv 是为了让 model 在 inference 时支持 autoregressive 式的 frame extension（类似 language model 的 KV cache 思想）。

Reference: [Magvit-v2 paper](https://arxiv.org/abs/2310.05737), [SDXL VAE](https://arxiv.org/abs/2307.01952)

---

## 3. STDiT：Spatial-Temporal Diffusion Transformer

### 3.1 核心思想：解耦 attention

Full 3D self-attention 的 complexity 是 $O((T \cdot H \cdot W)^2)$，对长视频完全不可行。STDiT 借鉴 Latte 的思路，把 attention 拆成两路：

**Spatial self-attention**（within-frame）：
$$\text{Attn}_{spatial}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

只在同一个 frame 的所有 spatial tokens 之间做 attention，complexity $O(T \cdot (HW)^2)$。

**Temporal self-attention**（across-frames, same spatial location）：
对每个 spatial position $(i,j)$，在所有 frames $t=1..T$ 的 tokens 之间做 attention，complexity $O(HW \cdot T^2)$。

总体 complexity 从 $O(T^2 H^2 W^2)$ 降到 $O(T \cdot H^2 W^2 + HW \cdot T^2)$，对长视频是 game-changer。

### 3.2 初始化策略：从 PixArt-α 迁移

Open-Sora 没有从 scratch 训练，而是用 **PixArt-α 的 580M checkpoint** 作为 backbone。新增的 temporal attention 的 projection layers 初始化为 zero：

$$W_{temporal\_proj}^{init} = \mathbf{0}$$

直觉：训练初始时 temporal attention 的 output 是 zero，model 完全 behave like 原 PixArt（image generation），然后 gradually learn temporal dynamics。这种 "zero-init new modules" 的 trick 在 LLM 的 LoRA、Vision Transformer 的 adapter 里都常见。

参数量从 580M → 1.1B，几乎翻倍。

### 3.3 RoPE for temporal attention

替换 sinusoidal positional encoding 为 Rotary Position Embedding (RoPE)。RoPE 的核心公式：

对于 query/key 的 2D 子空间 $(x_{2i}, x_{2i+1})$，position $m$ 处的 rotation：

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

其中 $\theta_i = 10000^{-2i/d}$ 是 frequency base，$m$ 是 frame index，$i$ 是 dimension index（上标对应 pair index）。

为什么 temporal 用 RoPE 更好？直觉：**video generation 本质是 sequence prediction**，RoPE 的 relative position encoding 性质天然适合 extrapolation 到更长的 frame count（虽然 Open-Sora 没有显式做 length extrapolation，但这个 inductive bias 是对的）。

Reference: [RoPE paper](https://arxiv.org/abs/2104.09864), [Latte paper](https://arxiv.org/abs/2401.03048)

### 3.4 QK-Normalization

借鉴 SD3，对所有 attention 加 QK-norm：

$$Q' = \frac{Q}{\|Q\|_2 + \epsilon}, \quad K' = \frac{K}{\|K\|_2 + \epsilon}$$

用极小的 $\epsilon = 10^{-15}$。这个 trick 看起来 trivial，但对 half-precision (bf16) training 的稳定性至关重要。直觉：**bf16 下 attention logits 容易 explode 到 inf/nan**，QK-norm 把 logits 的 scale 控制在合理范围，避免 softmax saturation 和 gradient spike。

Reference: [SD3 paper](https://arxiv.org/abs/2403.03206), [Adam instability theory](https://arxiv.org/abs/2304.09871)

---

## 4. Conditioning：Image/Video-to-Video 的 Masking 策略

### 4.1 Mixed timestep masking

这是 Open-Sora 最 elegant 的设计之一。对于 image-to-video / video-to-video 任务，conditioning frames 应该是 "已经 denoised" 的，而其他 frames 仍处于 noisy 状态。

Forward pass 时：
- Conditioning frames: assign $t = 0$（clean）
- Other frames: retain diffusion timestep $t$

但预训练 model 从没见过同一个 sample 里 mixed timesteps，直接用会崩。借鉴 UL2 的 mode-switching 思想，训练时随机 masking：

```
Masking patterns:
- first frame
- first k frames
- last frame
- last k frames
- first + last k frames
- fully random frames
```

50% 的 samples 加 masking，10k steps 后 model 就学会了 image/video conditioning，且不影响 text-to-video 性能。30% masking ratio 太低则 conditioning 学不进去。

直觉：**这本质上是一个 "multi-task learning via input format" 的 trick**，类似 UL2 在 language model 里用 different span corruption modes 来 unify NLU 和 NLG。

Reference: [UL2 paper](https://arxiv.org/abs/2205.05131), [PixArt](https://arxiv.org/abs/2310.00426)

### 4.2 Score conditioning

Caption 后面 append 额外的 control signals：

```
"[Original Caption] aesthetic score: 5.5, motion score: 10, camera motion: pan left"
```

这本质是 **Classifier-Free Guidance 的 conditioning 维度扩展**。inference 时用户可以手动调 aesthetic score、motion score、camera motion 来 control 生成。camera motion 用 13k 高置信 clips 人工标注。

---

## 5. Training Strategy：从 Image Diffusion 到 Video Diffusion

### 5.1 Flow Matching 替代 DDPM

Open-Sora 用 Rectified Flow（连续时间 ODE）替代 DDPM（离散时间 Markov chain）。Rectified Flow 的核心：

学习一个 vector field $v_\theta(x_t, t)$ 使得 ODE $\frac{dx_t}{dt} = v_\theta(x_t, t)$ 从 $x_0$（noise）到 $x_1$（data）的轨迹是直线。

Training objective：

$$\mathcal{L}_{RF} = \mathbb{E}_{t \sim \mathcal{U}(0,1), x_0 \sim \mathcal{N}(0,I), x_1 \sim p_{data}} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$$

其中 $x_t = (1-t)x_0 + t x_1$ 是 linear interpolation，$t$ 是 continuous timestep ∈ [0,1]，$v_\theta$ 是 model 预测的 velocity。

直觉：**Rectified Flow 把 diffusion 的弯曲轨迹 "拉直"**，所以 sampling 时只需要 fewer steps（Open-Sora 从 100 步降到 30 步）。这跟 SD3 的思路一致。

还用了 **logit-norm timestep sampling**：$t \sim \text{Beta}(\alpha, \beta)$ 经过 logit 变换，让 $t$ 的分布更集中在 mid-range（那里 model 学得最难）。

Reference: [Flow Matching paper](https://arxiv.org/abs/2210.02747), [Rectified Flow](https://arxiv.org/abs/2209.03003)

### 5.2 Multi-resolution Bucket 策略

Sora report 强调"保持原始 resolution/aspect ratio/length 训练"。Open-Sora 比较了三种方案：

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| NaViT | Patch packing + masking | 灵活 | 无法用 Flash Attention |
| FiT (padding) | Pad to max size | 简单 | 内存浪费 |
| **Bucket** | 固定大小桶，batch 内统一 | 高效 | 灵活性受限 |

Open-Sora 选了 Bucket。每个 bucket 是 $(resolution, \#frames, aspect\_ratio)$ 三元组。还有两个属性：
- **Keep probability**：高分辨率 video 按 probability 下采样到低分辨率 bucket，省 compute
- **Batch size**：per-bucket 调整，平衡 GPU load

直觉：**Bucket 是 "curriculum learning by resolution" 的一种实现**，低分辨率 bucket 见多 data，高分辨率 bucket 见少但 fine-grained data。

Reference: [NaViT](https://arxiv.org/abs/2307.06304), [SDXL bucketing](https://arxiv.org/abs/2307.01952)

### 5.3 8 阶段渐进式 Adaptation

从 PixArt-Σ 2K checkpoint 出发，8 个 sequential stages：

1. Multi-resolution image gen (20k steps)
2. QK-norm 加入 (18k steps)
3. DDPM → Rectified Flow (10k steps)
4. Rectified flow + logit-norm + resolution-aware timestep (33k steps)
5. AdamW epsilon 减小到 $10^{-15}$ (8k steps)
6. 新 VAE + FPS conditioning (25k steps)
7. Zero-init temporal attention blocks，先在 image 上训 (3k steps)
8. Temporal blocks masking 策略，只在 video 上训 (38k steps)

每个 stage 解决一个 specific problem，避免同时改太多东西导致 training 不稳定。总训练 68k steps，~35k H100 GPU hours，cost 相当低。

### 5.4 三阶段 Data Curriculum

| Stage | Dataset | Data volume | Resolution | Steps |
|-------|---------|-------------|------------|-------|
| 1 | Webvid-10M | 40k hours | 240p-360p | 30k (2 epochs) |
| 2 | Panda-70M filtered | 20M clips / 41k hours | 360p-480p | 23k (0.5 epoch) |
| 3 | Curated high-quality | ~2M clips / 5k hours | 720p-1080p | 15k (2 epochs) |

直觉：**从 low-quality large-scale data 学 motion prior，再到 high-quality small-scale data 学 visual fidelity**。这跟 LLM 的 "pretrain on web → SFT on curated data" 的 curriculum 完全 analog。

---

## 6. 实验结果分析

### 6.1 VBench 评测

Table 2 的关键数字：

| Model | Total (%) | Quality (%) | Semantic (%) |
|-------|-----------|-------------|--------------|
| Open-Sora 1.2 | **79.76** | **81.35** | **73.39** |
| Open-Sora 1.1 | 75.66 | 77.74 | 67.36 |
| OpenSoraPlan V1.3 | 77.23 | 80.14 | 65.62 |
| Show-1 | 78.93 | 80.42 | 72.98 |

1.2 版本相比 1.1 的 Total Score 提升 4.1%，主要来自：
- 3D VAE（temporal compression 4×）→ 更流畅
- Rectified Flow → 更高效的 trajectory
- Multi-stage data curriculum → 更高 quality

### 6.2 VAE 重建质量

Table 1：

| Model | SSIM↑ | PSNR↑ |
|-------|-------|-------|
| Open-Sora 1.2 | 0.880 | 30.590 |
| OpenSoraPlan 1.1 | 0.882 | 29.890 |

PSNR 30.59 对应大约 MSE ~0.087，在 256× compression 下这是相当好的 result。

Reference: [VBench](https://arxiv.org/abs/2403.14822), [Open-Sora-Plan](https://arxiv.org/abs/2412.00131)

---

## 7. Data Pipeline 细节

### 7.1 Data Sources

- **Webvid-10M**：10M pairs，low-res + watermark（适合 warmup）
- **Panda-70M**：用了 20M high-quality subset
- **HD-VG-130M**：BLIP-2 generated captions，quality 较差
- **MiraData**：77k long videos，games + city exploration
- **Vript**：400k densely annotated
- **Inter4K**：1K 4K clips
- **Pexels/Pixabay/Mixkit**：free-licensed 高质量
- **LAION** subset（aesthetic > 6.5）+ **Unsplash-lite**：~3M images

总计 30M video clips，80k hours。

### 7.2 Filtering Pipeline

1. **PySceneCut**：scene detection + cut into clips
2. **Aesthetic Score**（LAION scorer，3 frames 平均）
3. **Optical Flow Score**（UniMatch）：过滤 low-motion videos
4. **OCR**（DBNet++ via MMOCR）：过滤 text-heavy videos（news/ads）

### 7.3 Captioning

- **GPT-4V**：API，但 expensive
- **PLLaVA 13B**：开源，4 frames/video，spatial pooling 2×2
- **Camera motion detection**：optical flow → append "pan left" 等 tag

直觉：**Caption quality 直接决定 text-video alignment**。PLLaVA 虽然有 hallucination，但足够训练 T2V model。Camera motion 是 PLLaVA 的 blind spot，所以单独用 optical flow 检测补充。

Reference: [PLLaVA](https://arxiv.org/abs/2404.16994), [UniMatch](https://arxiv.org/abs/2211.03003), [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)

---

## 8. 相关联想与 Intuition Building

### 8.1 与 Sora 的差距

Open-Sora 虽然复现了 Sora 的 most techniques，但还有明显差距：

1. **Scale**：Sora 估计是几亿到十亿级 H100 hours，Open-Sora 只用 35k hours（~3000× 差距）
2. **Data scale**：Sora 暗示用了海量 internal data，Open-Sora 只有 80k hours
3. **World simulation**：Sora 展现了 physical understanding（fluid dynamics, reflection），Open-Sora 更偏 visual fidelity
4. **Length**：Sora 可能分钟级，Open-Sora 16s 上限

### 8.2 与其他架构的对比

| 架构 | 代表 | Attention | 优势 |
|------|------|-----------|------|
| UNet-based | SVD, AnimateDiff | 2D conv + temporal | 成熟，kernel 优化好 |
| Full DiT | DiT, PixArt | full 3D | 简单，scale 友好 |
| STDiT | Open-Sora, Latte | 解耦 spatial/temporal | 长视频友好 |
| MMDiT | SD3 | multimodal dual stream | text-image 对齐好 |

Open-Sora 选 STDiT 是 engineering trade-off：full attention 在长 video 上不可行，UNet 又不利于 scaling。

### 8.3 与 LLM 训练的类比

Open-Sora 的很多设计都和 LLM training 高度 analog：

- **Zero-init new modules** ≈ LoRA 的 zero-init
- **Bucket curriculum** ≈ LLM 的 length bucketing
- **Stage-wise adaptation** ≈ continual pretraining 的 phase 划分
- **Mixed masking for conditioning** ≈ UL2 的 multi-mode training
- **RoPE for temporal** ≈ LLM 的 position encoding
- **Flow matching** ≈ LLM 的 next-token prediction（更直接的 regression 目标）

### 8.4 推测 Sora 的未公开技术

基于 Open-Sora 的复现经验，可以推测 Sora 可能还有：

1. **更激进的 VAE compression**：可能 8× temporal 甚至更高
2. **Joint image-video training with larger ratio**：Sora 一定用了海量 image data
3. **DPO/RLHF on human preference**：alignment 到 aesthetic preference
4. **Autoregressive frame extension**：causal VAE 支持 infinite length generation
5. **Compute-optimal scaling**：可能用了 Chinchilla-style scaling law analysis

### 8.5 未来方向猜想

1. **Unified tokenizer**：Magvit-v2 提示 discrete tokenizer + LM 可能 beat diffusion
2. **Long video via KV cache**：causal VAE + autoregressive DiT
3. **3D RoPE**：spatial + temporal 都用 RoPE，支持 native 3D position
4. **World model**：video generation + action conditioning → real simulator
5. **Test-time scaling**：inference 时 search/optimize trajectory

Reference: [Magvit-v2 tokenizer](https://arxiv.org/abs/2310.05737), [DiT scaling](https://arxiv.org/abs/2212.09748), [Open-Sora GitHub](https://github.com/hpcaitech/Open-Sora)

---

## 9. 关键 Takeaways

1. **3D VAE 是 video generation 的关键 bottleneck**：spatial 8× + temporal 4× 的 compression 让 DiT training tractable，但需要 careful staged training + identity loss 来 leverage 2D VAE 的 knowledge
2. **STDiT 解耦 attention 是长视频的工程最优解**：full attention 不可行，UNet 不 scale，解耦是 sweet spot
3. **从 image model adaptation 比从 scratch 训练 efficient 100×**：zero-init temporal attention + staged adaptation 是关键
4. **Data curriculum > data scale**：low-quality warmup + high-quality fine-tune 比 uniform training 更 efficient
5. **Flow matching + QK-norm 是 stable training 的 recipe**：rectified flow 减少 sampling steps，QK-norm 防 bf16 spike
6. **Masking 策略 unify 多任务**：一个 model 同时支持 T2V/I2V/V2V，类似 UL2 的 multi-mode 思想

---

## 10. 局限与 Critique

1. **16s 上限**：离 Sora 的分钟级还远，需要更好的 long-range attention 机制（可能是 sparse attention 或 sliding window）
2. **80k hours data**：相比 Sora 推测的 millions of hours，data scale 差 1-2 个数量级
3. **VBench 评测局限**：VBench 主要评 2s 240p videos，无法 capture long video coherence 和 physical plausibility
4. **No human evaluation**：纯自动 metric 可能 mislead
5. **Motion control 仍弱**：虽然有 camera motion conditioning，但 complex motion（多物体交互、physics）控制有限

---

这篇 paper 的真正贡献不是 single technique，而是 **把 Sora 的 each hint 都 engineering 化并 open source**。对于 community 来说，这是一个 "reference implementation"，让其他研究者可以在此基础上 iterate，而不需要 reverse-engineer OpenAI 的黑盒。

如果想 deep dive 某个 specific aspect（比如 3D VAE 的 causal conv 细节，或者 flow matching 的 numerical ODE solver 选择），可以继续聊。

主要 references：
- [Open-Sora GitHub](https://github.com/hpcaitech/Open-Sora)
- [Open-Sora project page](https://hpcaitech.github.io/Open-Sora/)
- [Sora technical report](https://openai.com/research/video-generation-models-as-world-simulators)
- [PixArt-α](https://arxiv.org/abs/2310.00426)
- [PixArt-Σ](https://arxiv.org/abs/2403.04672)
- [Latte](https://arxiv.org/abs/2401.03048)
- [SD3](https://arxiv.org/abs/2403.03206)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Magvit-v2](https://arxiv.org/abs/2310.05737)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [UL2](https://arxiv.org/abs/2205.05131)
- [VBench](https://arxiv.org/abs/2403.14822)
- [NaViT](https://arxiv.org/abs/2307.06304)
- [Panda-70M](https://arxiv.org/abs/2402.19479)
- [PLLaVA](https://arxiv.org/abs/2404.16994)
