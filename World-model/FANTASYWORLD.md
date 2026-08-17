---
source_pdf: FANTASYWORLD.pdf
paper_sha256: 1b0fa60a6f797c31c227abd626f164a89ceafe63db0cbc2335f5dce10101728c
processed_at: '2026-08-04T06:41:05-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 FANTASYWORLD

好，我换一种方式，像在白板前跟你聊天那样讲。

---

## 一句话概括

**video model 其实已经"懂"3D 了，只是没说出来。FANTASYWORLD 就是给它接个"嘴巴"，让它把心里的 3D 想法 explicit 吐出来，同时用这个 3D 想法反向帮它自己生成更 consistent 的 video。**

就这么个事。

---

## 为什么要做这件事

想象你看一段 video——比如有人绕着桌子走一圈拍的视频。你的大脑不光看到 2D pixel，你自动"脑补"出桌子的 3D shape、墙的位置、空间关系。video diffusion model 其实也在做类似的事情，因为它训练在海量 video 上，video 本身是 3D world 的 projection，所以 model 内部 hidden features 里**已经编码了 3D 信息**。

问题是，这个 3D 信息是 implicit 的、散落 在 latent space 里，没人去 explicitly extract。

现有做法的 pain point：

- **ReconX / ViewCrafter 路线**：先 generate video，再跑 NeRF/3DGS 重建。等于 video model 自己懂 3D，但你又重新跑一遍 3D reconstruction，浪费。
- **Voyager / Uni3C 路线**：从第一帧 extract point cloud，inject 进去当 prior。但 camera 大角度运动时，第一帧的 point cloud 根本不在视野里了，prior 失效，整个 scene 崩掉（你看 Table 1 的 Large motion，Voyager 3D Consistency 直接掉到 13.82，简直惨烈）。
- **Geometry Forcing 路线**：用 frozen VGGT 当 teacher，finetune video model。但 finetune 会破坏 video model 在海量 video 上学的 imagination 能力，得不偿失。

FANTASYWORLD 的 insight 是：**别动 video model 本身（keep frozen），挂一个 lightweight 旁路 branch 专门负责 "翻译" hidden features 到 explicit 3D**，两边通过 cross-attention 互相 chat。

---

## 架构怎么实现的

### 整体类比

你可以把它想成一个**双人配合**：
- 一个人（Wan2.1 frozen backbone）负责"想象画面"，他脑子里有海量 video 知识，能 generate 出 photorealistic frames
- 另一个人（geometry branch）负责"画工程图"，他听第一个人讲，把脑子里 implicit 的 3D structure 用工程语言（depth map、point cloud、camera pose）画出来
- 两个人不断对话（bidirectional cross-attention）：工程图给想象画面提要求（"这里 geometry 要 consistent"），想象画面给工程图补细节（"这个 occluded region 后面应该是这样"）

最终单次 forward，两个人同时给出答案：photorealistic video + 3D representation。

### PCB（Preconditioning Blocks）—— 给 geometry branch 一个干净的起跑线

这是个特别聪明的小 trick。

diffusion model 一开始输入是纯 noise（或者高 noise latent）。如果你直接把这个 noisy latent 喂给 geometry branch，会发生什么？geometry branch 训练早期会被 noise gradient 主导，它学不到 geometry 信号，反而先要学会"denoise"——这浪费 capacity，而且 gradient variance 极大。

paper 里有个非常漂亮的观察（Figure 3）：**diffusion 的 denoising 不光在 time 维度发生，也在 network depth 维度发生**。即使你 fix timestep，deeper WanDiT layer 出来的 feature 在 spatial structure 上更清晰（他们用 PCA 可视化证明了）。

所以 FANTASYWORLD 复用 Wan2.1 的前 16 层（frozen），先把 noisy latent "partially denoise" 一下，再喂给 geometry branch。这就好比——你要让一个学生学 geometry，先给他一本半成品教材（已经把 noise 过滤掉大半），而不是给他一本全是乱码的天书让他自己 denoise。

**直觉上**：PCB 就是把 latents 推到一个 "signal 已经 emerge" 的区域，让 geometry branch 直接处理 "有 structure 的 features"。

### IRG Block 的双向 cross-attention —— 核心魔法

这是整个 paper 的灵魂。

公式：

$$A = \mathrm{softmax}\left(\frac{Q_v K_g^\top}{\sqrt{d_k}}\right)$$

- $Q_v$：video tokens 的 query（"我 video 想知道哪个 geometry 信息能帮我"）
- $K_g$：geometry tokens 的 key（"我 geometry 有这些 structure 信息"）
- $A$：alignment matrix，谁对谁有多重要

更新：

$$X_v^+ = X_v + \gamma_v A V_g$$
$$X_g^+ = X_g + \gamma_g A^\top V_v$$

注意第二行那个 $A^\top$ —— 这是 geometry 在 query video。这就实现了**双向**：
- video 从 geometry 拿 "3D consistency 信号"（geometry 告诉 video：你不能 hallucinate 出 geometry 上对不上的东西）
- geometry 从 video 拿 "imagination 信号"（video 告诉 geometry：occluded region 后面长这样，你 fill 进去）

$\gamma_v, \gamma_g$ 是 learnable gate，初始化为 0（zero-init），保证训练开始时 backbone 行为不变，慢慢学到该注入多少。

**这其实就是 latent-space 的 alternating minimization**——video 和 geometry 互相 pull，收敛到一个 consistent 的 fixed point。

参考 MM-BiCrossAttn 的思路：https://arxiv.org/abs/2403.17368

### Camera control 简化 —— 删繁就简

Wan2.1 原版用 AdaLN，同时预测 scale $\gamma_i$ 和 shift $\beta_i$。FANTASYWORLD 发现只要 shift 就够了：

$$f_i = f_{i-1} + \beta_i$$

直觉上：camera 信息是 "additive" 的（"再往前走一点"），不是 "multiplicative" 的（"把这个 feature 放大 10 倍"）。删掉 scale 让训练更稳定，而且只在 first 24 of 40 blocks 应用（深层已经接近 output，camera 控制效果减弱）。

### 3D DPT Head 的反向 reassembly —— 另一个反直觉的洞察

传统 DPT (Dense Prediction Transformer) 假设：浅层 feature = 高频细节（upsample 多），深层 feature = semantic（downsample）。

但 diffusion backbone **不遵循这个规律**！深层 features 反而 spatial 更清晰（因为更接近完全 denoise）。

所以 FANTASYWORLD 反过来：选用 blocks {8, 12, 18, 24} 的 features，**最深层 (24) upsample 最多**，浅层 downsample。这把 anchor signal 锚定在 "最 mature 的 features" 上。

这个 insight 我觉得对所有 "用 diffusion backbone 做 dense prediction" 的工作都有启发。如果你做 flow estimation、segmentation from diffusion features，也应该 invert 这个 reassembly 逻辑。

### Temporal upsampling

每个 feature stream 后接两个 temporal block，4x temporal upsampling，输出 $T = 4(t-1) + 1$ 帧。$t-1$ 是因为 video latent 是 inter-frame 表示，对应 WanVAE 的时间变换。causal 3D convolution 保证不 leak future 信息。

---

## Loss 怎么设计的

总 loss：

$$\mathcal{L}_{\mathrm{total}} = \underbrace{\mathbb{E}_{z_0, \epsilon, t, c} \left[ \| \epsilon_\theta(z_t, t, c) - \epsilon \|_2^2 \right]}_{\text{标准 diffusion loss}} + \lambda \underbrace{\mathcal{L}_{\mathrm{geo}}}_{\text{geometry supervision}}$$

$\mathcal{L}_{\mathrm{geo}}$ 三部分：

$$\mathcal{L}_{\mathrm{geo}} = \mathcal{L}_{\mathrm{depth}} + \mathcal{L}_{\mathrm{pmap}} + 3\mathcal{L}_{\mathrm{camera}}$$

camera loss 乘 3，因为 camera 参数最稀疏（每帧就 9D: rotation 3 + translation 3 + focal length 1... 其实是 7D，paper 写 9D 包含了 focal length 的某种 parameterization），gradient signal 弱，需要 boost。

**Depth loss** 来自 [Video Depth Anything](https://arxiv.org/abs/2501.07463)，包含 temporal gradient matching（强制 depth 在时间上 consistent）+ per-frame scale-sensitive loss（不做 affine normalization，因为这里要绝对深度 anchor 3D）。

**Point map loss** 来自 VGGT：

$$\mathcal{L}_{\mathrm{pmap}} = \sum_{i=1}^{N} \left\| \Sigma_i^P \odot (\hat{P}_i - P_i) \right\| + \left\| \Sigma_i^P \odot (\nabla \hat{P}_i - \nabla P_i) \right\| - \gamma \log \Sigma_i^P$$

- $P_i \in \mathbb{R}^{T \times H \times W \times 3}$: 预测的 3D point map（world coordinates）
- $\hat{P}_i$: ground truth
- $\Sigma_i^P$: predicted uncertainty map（per-pixel confidence）
- $\odot$: Hadamard product

$\Sigma$ 让 model 对 hard region（occlusion、textureless）自适应降权，$-\gamma \log \Sigma$ 防止 trivial solution（$\Sigma \to \infty$）。

参考 VGGT：https://arxiv.org/abs/2503.11651

---

## 训练分两阶段

**Stage 1: Latent Bridging**（20k steps, 64 H20 GPUs, 36 hours）

- Wan2.1 frozen
- 只训练 geometry branch
- 让它学会 "听懂" Wan2.1 的 latent space

类比：先让工程画家学会听懂想象画家的"方言"，再开始画图。

**Stage 2: Unified Co-Optimization**（10k steps, 112 H20 GPUs, 144 hours）

- Wan2.1 依然 frozen
- 训练 bidirectional cross-attention adapters + camera adapter
- 两个人开始正式对话

**为什么 backbone 始终 frozen？** 因为 Wan2.1 在海量 video 上训练获得的 imagination prior 极其 valuable（能 fill occluded region、能 generate plausible texture），finetune 会 catastrophic forgetting。Adapter 训练保留 prior，只学 alignment。

---

## 实验数据解读

### Table 1: WorldScore (Large camera motion 最能暴露问题)

| Method | 3D Consist. | Photo Consist. | Style Consist. |
|--------|-------------|----------------|----------------|
| WonderWorld | 63.70 ± 24.37 | **3.22** ± 8.47 | 35.95 ± 33.47 |
| Voyager | **13.82** ± 19.96 | **9.52** ± 17.17 | 61.34 ± 35.29 |
| Uni3C | 73.95 ± 17.55 | 46.78 ± 32.64 | 71.43 ± 29.38 |
| AETHER | 63.97 ± 17.39 | 33.11 ± 23.99 | 61.99 ± 32.24 |
| **FANTASYWORLD** | **74.83** ± 16.31 | **60.61** ± 21.39 | **82.02** ± 19.56 |

读这表的人话翻译：
- **Voyager 在大视角运动时直接崩盘**——first-frame point cloud out of view 了，剩下 video model 自己 hallucinate，3D Consistency 13.82 简直惨不忍睹
- **WonderWorld Photo Consistency 跌到 3.22**——意味着 generate 出来的 frame 跟 reference image 几乎不像，style 完全 drift 了
- **FANTASYWORLD standard deviation 最小**（±16.31 vs WonderWorld ±24.37）——意味着 geometry branch 提供了 "stabilizing signal"，让 model 在不同 scene 上更 robust

**Ablation 关键点**：去掉 geometry branch，3D Consist. 从 74.83 → 72.06。证明 geometry branch 不光是 "decoder"，它通过 cross-attention 反过来 regularize video generation。

### Table 2: 3DGS Reconstruction

| 设置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| w/o 3D branch + VGGT init | 26.89 | 0.84 | 0.17 |
| w/ 3D branch + VGGT init | **28.24** | **0.86** | **0.14** |
| w/ 3D branch + self init | 26.54 | 0.85 | 0.19 |

人话翻译：
- 加 geometry branch 让 PSNR +1.35dB（在 3DGS 重建里这是显著提升）
- 自己预测的 point cloud 比 VGGT 略差（26.54 vs 28.24），说明 geometry branch 输出精度还没超过 VGGT
- 但 26.54 仍然 competitive，证明 geometry branch 输出的 3D structure 是 meaningful 的——这个 latent 可以直接当 downstream representation 用

---

## 这个工作真正的"道"在哪

我的 read：

**1. 它证明了一个 hypothesis：video diffusion model 的 internal features 已经是 3D-aware 的。**

这就像 CLIP features 早就具备 detection 能力，需要 GroundingDINO 把这个能力 surface 出来。FANTASYWORLD 就是 video latent 的 "geometry surfacer"。

**2. frozen + bidirectional adapter 的范式非常 elegant。**

类似 LoRA 的 spirit——保留 pretraining 能力，加 lightweight 旁路学新能力。但这里的能力是 3D structure extraction，而且用 bidirectional cross-attention 让两个 branch 互相 refine，比 ControlNet 那种单向 injection 更对称。

**3. "diffusion 的 denoising 也在 depth 维度发生" 这个观察非常重要。**

这解释了为什么 PCB 必要、为什么 DPT head 要 invert reassembly。如果这个 observation generalize 到其他 dense prediction task，会有很多 follow-up。

**4. 它指向一种新的 world model 范式。**

不是 Sora 那种 pure generative world model（implicit simulator），也 pure 3D reconstruction（explicit 但没 imagination）。而是 **implicit generative + explicit geometry 的 dual representation**——video latent 负责 "想象"，geometry latent 负责 "结构"，两者互相 anchor。

这其实呼应了 Yann LeCun 的 JEPA 思路：pure generative pixel-level 信号不足，需要 abstract joint-embedding representation。FANTASYWORLD 的 geometry latent 在某种意义上就是 video 的 abstract geometric abstraction。

参考 JEPA：https://openreview.net/forum?id=BZ5a1r-kVsf

---

## 一些延伸联想

**1. 这个 geometry latent 可以直接 plug 进 embodied AI policy**

paper 里说 geometry branch 输出是 task-agnostic 3D feature。如果这个 feature 质量 OK，可以直接当 navigation/manipulation policy 的 observation，省掉 explicit reconstruction。参考 [GaussianWorld](https://arxiv.org/abs/2412.10373) 的 4D occupancy forecasting 思路。

**2. Long video 是 obvious next step**

paper 自己承认只能 fixed-length clip。结合 [Context-as-Memory](https://arxiv.org/abs/2506.03141) 的 memory retrieval，把 implicit 3D field 做 persistent state，应该能 extend 到 long-range。这个方向我猜很快会有 follow-up。

**3. 4D extension**

从 video → 4D（dynamic scene），geometry branch 需要加 temporal dimension 处理 dynamic object。参考 [Vidu4D](https://arxiv.org/abs/2405.16822) 的 dynamic Gaussian surfel 思路。

**4. Multi-agent coordination**

如果多个 robot 各自跑 FANTASYWORLD，能否通过共享 implicit 3D field 实现 consistent world view？这是 cooperative SLAM 的 generative 版本。

**5. Plücker coordinate 的优势**

Camera encoder 用 Plücker ray $(d, m)$ 表示，$d$ 是 ray direction，$m = p \times d$ 是 moment（编码 ray 到原点距离）。优势是直接对应 "viewing ray"，对 video generation 自然——每个 pixel 一条 ray，ray 集合定义 camera frustum。参考 [CamCo](https://arxiv.org/abs/2401.01707) 和 [Wan2.1](https://arxiv.org/abs/2503.20314)。

**6. 为什么选 block 16 作为 PCB/IRG boundary**

paper 没 ablate，但我推测：太浅，noise 太多；太深，PCB 计算贵且 features 偏向 RGB detail。16 layers 是 Wan2.1 总 40 layers 的 40%，diffusion model 中 "signal emerges around 30-50% depth" 的经验法则。

---

## 我的核心 take-away

**FANTASYWORLD 的本质 insight**：video diffusion model 的 hidden features 是 "implicit 3D world memory"，挂一个 lightweight trainable branch 把它显式 extract 出来，再让两边互相 refine，单 forward pass 出 reusable 3D representation。

这其实跟 LLM 里的 steering vector / task vector 是同一个 family——**foundation model 内部已经 encode 了很多能力，只是没暴露出来，我们用 cheap adapter 把它 surface 出来用**。

最 inspiring 的点还是那个 "diffusion denoising happens in depth too" 的观察。这个如果 generalize 到 flow estimation、segmentation、optical flow from diffusion features，应该会有一波 follow-up paper。

---

## 一些可能你会想深挖的点

如果你感兴趣，下面这些细节可以再展开：
1. **geometry branch 的 token 数量**：cross-attention cost 是 $O(N_v \cdot N_g)$，paper 没 explicitly 说 $N_g$ 多少。如果 geometry branch 用 reduced resolution，这个 trade-off 怎么算？
2. **Stage 1 vs Stage 2 的 ablation**：如果不做 Stage 1 直接 Stage 2 联合训练，会崩吗？我猜会，因为 geometry branch 没 "学会读 Wan latent" 之前，cross-attention 会注入 garbage 信号。
3. **为什么 camera loss 权重是 3**：paper 没 ablate，但 9D 参数 vs depth map (HxW) vs point map (HxWx3)，camera signal 确实最稀疏。
4. **zero-init gate $\gamma_v, \gamma_g$ 的训练 dynamics**：从 0 慢慢学到 non-zero，这跟 ResNet skip connection 的学习动力学类似。
5. **Open question**：geometry branch 输出的 implicit feature 直接 plug 进 downstream（不经 VGGT/3DGS 重建），效果如何？paper 只展示了经过 3DGS 重建后的 PSNR，没展示 "raw latent 直接用于 navigation" 的实验。这可能是真正的 killer application，但 paper 没做到。

参考链接汇总：
- Wan2.1: https://github.com/Wan-Video/Wan2.1
- VGGT: https://arxiv.org/abs/2503.11651
- DUSt3R: https://github.com/naver/dust3r
- MASt3R: https://github.com/naver/mast3r
- Fast3R: https://github.com/facebookresearch/fast3r
- Video Depth Anything: https://github.com/DepthAnything/Video-Depth-Anything
- Geometry Forcing: https://arxiv.org/abs/2507.07982
- Voyager: https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager
- AETHER: https://github.com/InternRobotics/Aether
- WonderWorld: https://github.com/KovenYu/WonderWorld
- Uni3C: https://github.com/alibaba-damo-academy/Uni3C
- WorldScore: https://worldscore.github.io/
- 3DGS: https://github.com/graphdeco-inria/gaussian-splatting
- Plücker coordinates: https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates
- Han et al. diffusion feature learning theory: https://arxiv.org/abs/2406.09385
- Context-as-Memory: https://arxiv.org/abs/2506.03141
- GaussianWorld: https://arxiv.org/abs/2412.10373
- Vidu4D: https://arxiv.org/abs/2405.16822
- AnySplat: https://arxiv.org/abs/2505.23716
- JEPA: https://openreview.net/forum?id=BZ5a1r-kVsf
- MM-BiCrossAttn: https://arxiv.org/abs/2403.17368

希望这个版本更"人话"一些，但保留了 build intuition 所需的所有技术细节。如果你想 drill down 某个具体 component，告诉我。

---

# FANTASYWORLD 深度讲解

Andrej，这篇 paper 的核心 insight 我觉得特别 resonant：**video diffusion model 的 hidden features 本身就已经 implicit 编码了 3D structure**（因为 video 是 3D world 的 2D projection），而 FANTASYWORLD 做的事情就是把这个 implicit 的 geometry 信号 "extract" 出来并 refine，同时反向用 geometry 信号 regularize video generation。这本质上是一种 **latent-space 的 mutual information maximization**。下面我从直觉到公式逐一拆解。

---

## 1. Big Picture：为什么需要 unified video + 3D

### 1.1 现有路线的痛点

| 路线 | 代表方法 | 问题 |
|------|---------|------|
| 2D-prior SDS | DreamFusion, Fantasia3D | per-scene optimization，慢 |
| Video diffusion + 后处理 3DGS/NeRF | ReconX, ViewCrafter, DimensionX | video latent 无法直接支持 3D reasoning，需要二次优化 |
| First-frame point cloud 先验 | Uni3C, Voyager | 大视角变化时先验 frame out of view，崩 |
| Frozen 3D model 监督 finetune VFM | Geometry Forcing | 破坏 VFM 的 generative creativity，训练成本高 |

FANTASYWORLD 的设计哲学是：**frozen VFM 保留 imagination，trainable geometry branch 提取 geometry**，两者通过 bidirectional cross-attention 互相 refine，单 forward pass 输出 reusable 3D features。

直觉上，这类似 LoRA 的 spirit——冻结主网络保留 pretraining 能力，加一个轻量 branch 注入新能力，但这里的能力是 **3D structure extraction**。

参考：
- Wan2.1 (阿里巴巴的视频 diffusion backbone): https://github.com/Wan-Video/Wan2.1
- VGGT (geometry branch 的灵感来源): https://vggtpipeline.github.io/ / https://arxiv.org/abs/2503.11651
- Geometry Forcing: https://arxiv.org/abs/2507.07982

---

## 2. 架构深度解析

### 2.1 整体 pipeline

```
[Image (CLIP)] ─┐
[Text (umT5)]  ─┼──→ PCBs (Wan2.1 first 16 layers, frozen) 
[Camera]       ─┘                │
                                 ↓ partially denoised latents
                                 │
                  ┌──────────────┴──────────────┐
                  ↓                              ↓
        Imagination Prior Branch        Geometry-Consistent Branch
        (Wan2.1 last 24 blocks)         (trainable, VGGT-style)
                  │                              │
                  └──── MM-BiCrossAttn ──────────┘
                       (双向 cross-attention
                        在每个 IRG block 之后)
                  │                              │
                  ↓                              ↓
        Video frames (81 frames)        3D features → DPT heads
                                         → depth, point map, camera pose
```

### 2.2 PCB (Preconditioning Blocks) —— 一个非常聪明的细节

paper 的 Sec 3.2 提到一个非常 key 的经验观察：**denoising 不仅在 timestep 维度进展，也在 network depth 维度进展**。Figure 3 的 PCA 可视化显示，即使固定 timestep，deeper WanDiT layers 产生的 features 在 spatial structure 上更清晰。

这个观察的深层原因可以从 [Han et al. 2025](https://arxiv.org/abs/2406.09385) 的 diffusion feature learning theory 理解：

- 早期 diffusion timestep，signal-to-noise ratio 低，模型主要学习 noise structure
- 晚期 timestep，SNR 高，模型开始学习真实 signal 的 structure
- 类似地，network 的 depth 也是一个"去噪维度"——浅层处理 high-frequency 噪声，深层整合 low-frequency 结构

所以直接把 noisy latents 喂给 geometry branch 会：
1. 训练早期被 high-noise gradient 主导
2. Geometry branch 浪费 capacity 在 "denoise" 而不是 "extract geometry"

**解决方案**：复用 Wan2.1 前 16 层 frozen，让 input 给 geometry branch 时已经 "partially denoised"，gradient variance 大幅下降。直觉上，PCB 就像一个 "warmup phase"，把 latents 推到 SNR 足够高的区域。

这让我想到 consistency models 里 "distill from a partially denoised sample" 的 trick——本质上都是用 teacher model 给 student model 提供 easier starting point。

### 2.3 IRG Block 的 bidirectional cross-attention

这是最核心的 design。公式：

$$A = \mathrm{softmax}\left(\frac{Q_v K_g^\top}{\sqrt{d_k}}\right)$$

变量含义：
- $Q_v \in \mathbb{R}^{N_v \times d_k}$: video tokens $X_v$ 投影得到的 query
- $K_g \in \mathbb{R}^{N_g \times d_k}$: geometry tokens $X_g$ 投影得到的 key
- $d_k$: key 的维度，除以 $\sqrt{d_k}$ 是标准 scaled dot-product attention，防止内积过大导致 softmax 饱和
- $A \in \mathbb{R}^{N_v \times N_g}$: alignment matrix，video tokens 对 geometry tokens 的注意力

更新公式：

$$X_v^+ = X_v + \gamma_v A V_g, \qquad X_g^+ = X_g + \gamma_g A^\top V_v$$

变量含义：
- $V_g \in \mathbb{R}^{N_g \times d_v}$: geometry tokens 投影得到的 value
- $\gamma_v, \gamma_g$: learnable scalar gates，初始化通常为 0（LayerNorm-style zero-init），保证训练开始时 backbone 行为不变
- $A V_g \in \mathbb{R}^{N_v \times d_v}$: geometry 信息注入 video，这是 **3D consistency 的来源**
- $A^\top V_v$: 注意这里是 $A$ 的转置！这意味着 geometry tokens 在查询 video tokens（"video 中哪个区域对应的 geometry 我需要 refine？"），把 video 的 imagination prior 注入 geometry

直觉上，这个 bidirectional 设计类似 [MM-BiCrossAttn](https://arxiv.org/abs/2403.17368) 和最近 [JavisDiT](https://arxiv.org/abs/2503.23377) 中的多模态 fusion。它本质上实现了一种 **alternating minimization in latent space**——video 和 geometry 互相 pull，最终收敛到 consistent fixed point。

### 2.4 Camera control 的简化

paper 的 Appendix A.1 提到，相对于 Wan2.1 的完整 AdaLN（预测 scale $\gamma_i$ 和 shift $\beta_i$），FANTASYWORLD 只预测 shift：

$$f_i = f_{i-1} + \beta_i$$

这个简化背后的直觉：AdaLN 的 scale 参数在 video diffusion 中容易导致训练不稳定（因为放大 latent 可能 blow up），而且 camera 信号本质上是 "additive pose information"，不需要 multiplicative modulation。同时只在 first 24 of 40 blocks 应用，是因为深层 blocks 已经接近 output，camera 控制效果减弱。

这种 additive injection 让我想到 ControlNet 的 zero-conv design 和 Flan-T5 中 conditioning 的方式。

### 2.5 3D DPT Head 的 inverted reassembly

这个细节非常 clever。Conventional DPT (Dense Prediction Transformer) 假设：
- 浅层 encoder features = high-frequency detail（应该 upsample 多）
- 深层 encoder features = semantic abstraction（应该 downsample）

但 **diffusion backbone 不遵循这个规律**！深层 diffusion blocks 的 features 反而 spatial 更清晰（因为更接近完全 denoise）。所以 FANTASYWORLD 反转 reassembly 逻辑：

- 选用 blocks {8, 12, 18, 24} 的 features（这些是 IRG block 的 indices）
- **deepest block (24) 的 features 被 upsample 最多**——这是 anchor signal
- shallow blocks 被 downsample，仅作 context 补充

这个 insight 对所有想用 diffusion backbone 做 dense prediction 的工作都很有启发。参考 [VGGT 的 DPT head](https://arxiv.org/abs/2503.11651) 和原始 [DPT](https://arxiv.org/abs/2101.00877)。

### 2.6 Temporal upsampling

每个 feature stream 后接两个 temporal blocks，4x temporal upsampling：

$$T = 4(t-1) + 1$$

变量含义：
- $t$: input 帧数（latent temporal resolution）
- $T$: 输出帧数（与 WanVAE decoder 对齐）
- $t-1$ 是因为 video latent 是 "inter-frame" 表示，$4(t-1)+1$ 对应 $t$ 个 latent frame 解码到 $4t - 3$ 个 RGB frame 的 WanVAE 时间变换

每个 temporal block 先 double temporal resolution，再 apply causal 3D convolution。"causal" 保证 autoregressive 解码时不 leak future 信息。

---

## 3. Loss Function 详解

### 3.1 Geometry loss

$$\mathcal{L}_{\mathrm{geo}} = \alpha \mathcal{L}_{\mathrm{depth}} + \beta \mathcal{L}_{\mathrm{pmap}} + \gamma \mathcal{L}_{\mathrm{camera}}$$

Appendix A.3 中具体写为：

$$\mathcal{L}_{\mathrm{geo}} = \mathcal{L}_{\mathrm{depth}} + \mathcal{L}_{\mathrm{pmap}} + 3\mathcal{L}_{\mathrm{camera}}$$

camera loss 加权 3，说明 camera pose 信号最稀疏（每帧只有 9D），需要相对放大 gradient。

#### Depth loss（来自 Video Depth Anything）

$$\mathcal{L}_{\mathrm{depth}} = \alpha \mathcal{L}_{\mathrm{TGM}} + \beta \mathcal{L}_{\mathrm{frame}}$$

- $\mathcal{L}_{\mathrm{TGM}}$ (Temporal Gradient Matching): $\|\nabla_t \hat{D} - \nabla_t D\|$，强制深度在时间维度上一致
- $\mathcal{L}_{\mathrm{frame}}$: per-frame scale-sensitive depth error，**不**做 scale/shift normalization（与 Depth Anything 的 affine-invariant loss 不同），因为这里需要绝对深度来 anchor 3D structure

参考 [Video Depth Anything](https://arxiv.org/abs/2501.07463) 和 [Depth Anything](https://arxiv.org/abs/2401.10891)。

#### Point map loss（来自 VGGT）

$$\mathcal{L}_{\mathrm{pmap}} = \sum_{i=1}^{N} \left\| \Sigma_i^P \odot (\hat{P}_i - P_i) \right\| + \left\| \Sigma_i^P \odot (\nabla \hat{P}_i - \nabla P_i) \right\| - \gamma \log \Sigma_i^P$$

变量：
- $P_i \in \mathbb{R}^{T \times H \times W \times 3}$: 预测的 3D point map（world coordinates）
- $\hat{P}_i$: ground truth point map
- $\nabla \hat{P}_i$: spatial gradient
- $\Sigma_i^P \in \mathbb{R}^{T \times H \times W}$: predicted **uncertainty/confidence** map（per-pixel）
- $\odot$: Hadamard product

这里 $\Sigma$ 起到的作用是 **heteroscedastic aleatoric uncertainty**（Kendall & Gal 2017），让模型对 hard region（occlusion, textureless）自适应降权。$-\gamma \log \Sigma$ 项防止 $\Sigma \to \infty$ 的 trivial solution。

#### Camera loss

$$\mathcal{L}_{\mathrm{camera}} = \sum_{i=1}^{N} \lVert \hat{\mathbf{g}}_i - \mathbf{g}_i \rVert_\epsilon$$

- $\mathbf{g}_i \in \mathbb{R}^9$: 预测的 camera 参数（3D rotation + 3D translation + focal length，共 9D）
- $\hat{\mathbf{g}}_i$: ground truth
- $\lVert \cdot \rVert_\epsilon$: Huber loss，对 outlier 鲁棒

### 3.2 Total training objective

$$\mathcal{L}_{\mathrm{total}} = \mathbb{E}_{z_0, \epsilon, t, c} \left[ \| \epsilon_\theta(z_t, t, c) - \epsilon \|_2^2 \right] + \lambda \mathcal{L}_{\mathrm{geo}}$$

变量：
- $z_0$: clean video latent
- $\epsilon \sim \mathcal{N}(0, I)$: injected Gaussian noise
- $t \sim \mathcal{U}(0, T)$: diffusion timestep
- $c$: conditioning (image + text + camera)
- $z_t = \sqrt{\bar\alpha_t} z_0 + \sqrt{1 - \bar\alpha_t} \epsilon$: noised latent at step $t$
- $\epsilon_\theta$: 网络预测的 noise
- $\lambda$: 平衡系数

这是标准 DDPM/flow matching 损失加上 geometry supervision。注意 $\mathcal{L}_{\mathrm{geo}}$ 是**直接监督 geometry branch 输出**，而不是通过 diffusion latent 间接监督——这意味着 gradient 直接 flow 到 geometry branch 的 DPT heads。

---

## 4. Two-stage Training 协议

### Stage 1: Latent Bridging（20k steps, batch 64, 36 hours on 64 H20 GPUs）

- Wan2.1 backbone frozen
- 只训练 geometry branch
- 从 block 16 取 hidden features 通过 adapter 喂给 geometry branch
- 让 geometry branch 学会 "读懂" Wan2.1 的 latent space

这一步类似 phase 1 of InstructPix2Pix——先建立 conditioning signal 的 alignment，再做 joint training。

### Stage 2: Unified Co-Optimization（10k steps, batch 112, 144 hours on 112 H20 GPUs）

- Wan2.1 backbone 依然 frozen
- 训练 bidirectional cross-attention adapters（24 个，对应 block 16-40）和 camera control adapter
- 此时 video 和 geometry 双向 flow information

**为什么 backbone 始终 frozen？** 因为 video foundation model 在海量 video 上训练后获得了强大的 imagination prior（这部分对 geometry 是 valuable 的，可以 fill in occluded regions），finetune 会 catastrophic forgetting 这个 prior。

**计算 budget 估算**：Stage 2 = 112 H20 × 144h ≈ 16,128 GPU-hours。这个 cost 对应只训练 ~24 个 lightweight adapters，说明 bidirectional cross-attention 的 parameter count 远小于 backbone（按 Wan2.1 14B 参数 × 24/40 layers 估计，cross-attention adapters 大约 100M-500M 参数级别）。

---

## 5. 实验数据深度分析

### 5.1 Table 1: WorldScore (Small vs Large camera motion)

让我重点看 Large motion 的数据，因为这最能暴露方法的 robustness：

| Method | 3D Consist. | Photo Consist. | Style Consist. |
|--------|-------------|----------------|----------------|
| WonderWorld | 63.70 ± 24.37 | **3.22** ± 8.47 | 35.95 ± 33.47 |
| AETHER | 63.97 ± 17.39 | 33.11 ± 23.99 | 61.99 ± 32.24 |
| Uni3C | 73.95 ± 17.55 | 46.78 ± 32.64 | 71.43 ± 29.38 |
| Voyager | **13.82** ± 19.96 | **9.52** ± 17.17 | 61.34 ± 35.29 |
| Ours w/o 3D | 72.06 ± 20.14 | 56.98 ± 23.60 | 81.59 ± 22.23 |
| **Ours w/ 3D** | **74.83** ± 16.31 | **60.61** ± 21.39 | **82.02** ± 19.56 |

**关键观察**：
1. **Voyager 在 Large motion 完全崩溃**（3D Consist. 13.82, Photo Consist. 9.52），因为它的 first-frame point cloud prior 在大视角变化时完全 out of view，剩下 video diffusion 自己 hallucinate
2. **WonderWorld 的 Photo Consist. 跌到 3.22**——style drift 严重，torn holes 出现
3. FANTASYWORLD 的 standard deviation 普遍最小（±16.31 vs WonderWorld ±24.37），说明 geometry branch 提供了 "stabilizing signal"
4. Ablation: 去掉 geometry branch，3D Consist. 从 74.83 → 72.06（-2.77），证明 geometry branch 确实注入了 3D awareness

### 5.2 Table 2: 3DGS Reconstruction on RealEstate10K

| Method | Post Rec. Init | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|----------------|--------|--------|---------|
| Ours w/o 3D | VGGT point cloud | 26.89 | 0.84 | 0.17 |
| Ours w/ 3D | VGGT point cloud | **28.24** | **0.86** | **0.14** |
| Ours w/ 3D | own feed-forward pcloud | 26.54 | 0.85 | 0.19 |

**重要 insight**：
- 加 geometry branch 让 PSNR +1.35dB（这是非常显著的提升，类似 diffusion vs GAN 的提升量级）
- 自身 point cloud init 比 VGGT init 略差（26.54 vs 28.24），说明 FANTASYWORLD 自己预测的 geometry 还没达到 VGGT 的精度
- 但 26.54 仍然是 competitive，证明 geometry branch 输出的 3D structure 是 meaningful 的，可以作为 downstream task 的 representation

### 5.3 Limitations 的诚实承认

paper 末尾承认：模型只能 fixed-length clip generation，没有 streaming/long-range capability。这个 limitation 实际上指明了 future work 方向——结合 [Context-as-Memory](https://arxiv.org/abs/2506.03141) 类似的 memory retrieval 机制，把 implicit 3D field 当成 persistent state。

---

## 6. 我对这篇 paper 的整体评价和 intuition building

### 6.1 核心贡献的真正价值

我觉得这篇 paper 最大的 contribution 是证明了一个 hypothesis：**video diffusion model 的 internal features 已经是 3D-aware 的**，只是没人去 explicitly extract。这类似 CLIP features 早就具备 detection 能力，需要 DINO/GroundingDINO 把这个能力 "surface" 出来。

这种 "frozen large model + trainable adapter" 的范式在 2024-2025 越来越主流，背后的原因是：训练 large foundation model 太贵，而且会 catastrophic forgetting。adapter-based 方法保留 pretraining 的知识，只在新 task 上做 alignment。FANTASYWORLD 把这个 idea 应用到 video diffusion + 3D 这个 cross-modal 任务上，并且用 bidirectional cross-attention 而非单向 ControlNet-style injection——这是 architecture 上的关键创新。

### 6.2 与 Voyager / AETHER / Geometry Forcing 的对比

| 方法 | Coupling 方式 | Backbone 是否训练 | 3D representation |
|------|--------------|-----------------|-------------------|
| Voyager | cache + geometry-injected frames | end-to-end | point cloud projection |
| AETHER | joint RGB-depth modeling | end-to-end | RGB-D |
| Geometry Forcing | VGGT feature supervision | finetune VFM | frozen VGGT features |
| **FANTASYWORLD** | bidirectional cross-attention | **frozen VFM** | implicit field via geometry branch |

FANTASYWORLD 的独特性是 **frozen + bidirectional**——既保留 generative prior，又让两个 branch 互相 refine。Geometry Forcing 虽然也用 frozen VGGT，但只单向 finetune VFM，会破坏 VFM 的 creativity。

### 6.3 与 Sora-style world model 的关系

最近大家都在 debate "video model 是不是 world model"。Andrej 你之前说过 Sora 是 "simulator of physical world"。FANTASYWORLD 的 perspective 是：**video model 的 simulator 能力是 implicit 的，需要 explicit geometry supervision才能让它变成 usable 3D world model**。

这其实呼应了 Yann LeCun 的 JEPA 思路：pure generative 信号不足，需要 joint-embedding 的 abstract representation。FANTASYWORLD 的 implicit 3D field 在某种意义上就是 video latent 的 abstract geometric representation。

### 6.4 可能的 future direction（我的联想）

1. **Streaming/long video**: 结合 memory retrieval (类似 [Context-as-Memory](https://arxiv.org/abs/2506.03141))，把 implicit 3D field 做 persistent
2. **Embodied AI**: 既然 implicit 3D feature 是 task-agnostic，可以直接 plug 进 navigation policy（参考 [GaussianWorld](https://arxiv.org/abs/2412.10373)）
3. **4D extension**: 从 video → 4D（dynamic scenes），需要在 geometry branch 加 temporal dimension（参考 [Vidu4D](https://arxiv.org/abs/2405.16822)）
4. **Robotics**: 直接用 implicit 3D feature 作为 RL policy 的 observation，避免 explicit reconstruction
5. **Multi-agent coordination**: 共享 implicit 3D field 让多 robot 协同时有 consistent world view

---

## 7. 一些值得深挖的细节

### 7.1 为什么选 block 16 作为 PCB/IRG boundary

paper 没有详细 ablate，但我推测：
- 太浅（<16）：geometry branch 收到太多 noise，gradient variance 大
- 太深（>16）：PCB 计算成本高，而且越往后 features 越偏向 RGB detail，对 geometry 提取帮助减少
- 16 layers 大约是 Wan2.1 总 layers (40) 的 40%，符合 diffusion model 中 "signal emerges around 30-50% depth" 的经验法则

### 7.2 Geometry branch 的 token 数量

如果 Wan2.1 输入是 81 frames × (336/16) × (592/16) = 81 × 21 × 37 = 62,877 tokens per modality，cross-attention 的 cost 是 $O(N_v \cdot N_g)$。即使 $N_g$ 远小于 $N_v$（geometry branch 可能做 spatial pooling），这个 cost 也不能忽略。Appendix 没说具体 token 数，但 mention "lightweight adapter"，推测 geometry branch 用了 reduced resolution。

### 7.3 Plücker coordinates 的作用

Camera pose encoder 用 Plücker ray 表示（参考 [Wan2.1](https://arxiv.org/abs/2503.20314) 和 [CamCo](https://arxiv.org/abs/2401.01707)）。Plücker ray $(d, m)$：
- $d \in \mathbb{S}^2$: ray direction
- $m = p \times d \in \mathbb{R}^3$: moment, 编码 ray 到原点的距离
- 满足约束 $d \cdot m = 0$

这种表示相对于 6D rotation representation (Zhou et al. 2019) 的优势是它直接编码 "viewing ray"，对 video generation 更自然——每个 pixel 对应一个 ray，ray 集合定义了 camera frustum。

### 7.4 DPT head 的 feature selection {8, 12, 18, 24}

这些 indices 是 IRG blocks 中的 indices（block 16 之后的相对 index）。从 Figure 3 PCA 看，block 24 之后 features 已经非常 mature。这些 indices 大致是 logarithmically spaced，类似 FPN 的多尺度设计。

---

## 8. References 我整理的关键链接

- **Wan2.1** (backbone): https://github.com/Wan-Video/Wan2.1
- **VGGT** (geometry branch 灵感): https://vggtpipeline.github.io/ 
- **DUSt3R** (VGGT 前作): https://github.com/naver/dust3r
- **MASt3R** (DUSt3R 改进): https://github.com/naver/mast3r  
- **Fast3R** (大规模重建): https://github.com/facebookresearch/fast3r
- **Video Depth Anything** (depth loss): https://github.com/DepthAnything/Video-Depth-Anything
- **Geometry Forcing** (对比方法): https://arxiv.org/abs/2507.07982
- **Voyager** (对比方法): https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager
- **AETHER** (对比方法): https://github.com/InternRobotics/Aether
- **WonderWorld** (对比方法): https://github.com/KovenYu/WonderWorld
- **Uni3C** (对比方法): https://github.com/alibaba-damo-academy/Uni3C
- **WorldScore** (评测 benchmark): https://worldscore.github.io/
- **3DGS** (reconstruction): https://github.com/graphdeco-inria/gaussian-splatting
- **Plücker coordinates**: https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates
- **Diffusion feature learning theory** (Han et al.): https://arxiv.org/abs/2406.09385
- **Context-as-Memory** (long video future work): https://arxiv.org/abs/2506.03141
- **GaussianWorld** (embodied 3D): https://arxiv.org/abs/2412.10373
- **Vidu4D** (4D extension): https://arxiv.org/abs/2405.16822
- **AnySplat** (generalizable 3D features): https://arxiv.org/abs/2505.23716

---

## 9. 总结直觉

如果用一句话 build intuition：**FANTASYWORLD 把 video diffusion model 的 hidden features 当成 "implicit 3D world memory"，通过一个 VGGT-style trainable branch 显式 extract 出来，再用 bidirectional cross-attention 让 geometry 和 video 互相 refine，单 forward pass 输出 reusable 3D representation**。

这个 paradigm 的优美之处在于：它不破坏 VFM 的 imagination 能力（frozen backbone），但用 cheap 的 adapter 把 latent geometry 提取出来。从 high-level 看，这其实是 "model editing" 范式在 video-3D task 上的应用——类似 LLM 的 task vectors 和 steering vectors，但是 spatial-temporal domain。

我觉得最 inspiring 的点是 **diffusion backbone 不遵循 CNN 的 feature hierarchy 直觉**——深层 features 反而 spatial 更清晰。这个 observation 如果 generalize 到其他 dense prediction task（flow, segmentation），应该会有更多应用。

希望这些细节对你的 intuition building 有帮助！如果想深入聊某个 component（比如 geometry branch 的 token 数如何 trade off，或者 3D DPT head 的 temporal upsample 设计），我可以再展开。
