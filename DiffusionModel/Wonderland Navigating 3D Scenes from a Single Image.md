---
source_pdf: Wonderland Navigating 3D Scenes from a Single Image.pdf
paper_sha256: b65d528f43d665cb78b5db5ec532fb5ec5b594745d80131b4d2d0febdcf6b2e3
processed_at: '2026-08-13T04:46:53-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Wonderland 用人话说

## 一句话说清

**拍一张照片，AI 自动脑补出整个 3D 场景，还能让你在里面走来走去。**

---

## 这事儿难在哪

想象你站在客厅拍了一张照片。照片只记录了你眼前那个角度的画面。但如果我想知道沙发背后长什么样、厨房在哪个方向、窗户外面是什么——单凭这一张照片，完全猜不出来。

人类凭经验能脑补（你见过很多客厅，知道沙发背后通常有墙、厨房通常在某个角落），但让 AI 做这件事很难，原因有三：

**第一，AI 没见过那么多场景。** 传统的 3D 重建方法需要你围着同一个房间拍几十张甚至上百张照片，AI 才能拼出 3D 模型。一张照片根本不够。

**第二，用图像生成模型"脑补"不靠谱。** 你可以让 Midjourney 之类的模型生成"沙发背后的样子"，但它每次生成的东西都不一样，拼不成一个连贯的 3D 空间。就像让一个人闭着眼睛画同一个房间的不同角度，每张画都对不上。

**第三，重建过程太慢。** 现有的高质量方法（比如 NeRF、3DGS）需要针对每个场景单独优化几十分钟甚至几个小时，没法"拍一张照片立刻出 3D"。

---

## Wonderland 怎么解决的

核心想法特别巧妙：**用视频生成模型当"3D 脑补引擎"。**

### 关键洞察

视频生成模型（比如能生成视频的 AI）在训练时看过海量视频。视频是什么？就是相机在场景里移动时拍下的连续画面。所以视频模型天然"理解"一件事：**当相机这么移动时，场景应该长这样。**

这种理解本质上是 3D 的——虽然模型输出的是 2D 画面序列，但它学到的是"3D 世界在不同视角下看起来什么样"。

Wonderland 做的就是：
1. **给视频模型一张照片当起点**
2. **告诉它"相机往左走、往前走、转一圈"**
3. **模型生成一段视频，相当于它在脑补"如果相机真的这么动，会看到什么"**
4. **把这段视频的内部表示直接转成 3D 模型**

### 为什么不直接用生成的视频画面拼 3D？

因为视频画面太大太占内存了。49 帧 480×720 的视频，如果按像素处理，要处理上百万个 token，AI 的注意力机制算不过来。

所以 Wonderland 做了个聪明的事：**不处理画面，处理画面的"压缩版"（latent）。** 视频模型内部有个压缩表示，空间压缩 8 倍、时间压缩 4 倍，总共压缩 256 倍。这个压缩版保留了场景的核心结构和外观信息，但数据量小得多，AI 能处理。

这就好比：与其给你一本 1000 页的书让你背，不如给你一份 4 页的核心摘要。

---

## 怎么控制相机

这是另一个难点。你告诉视频模型"相机往左走"，它可能理解个大概，但位置偏了、角度歪了。

Wonderland 的做法是：**给每个像素都标注一条"视线方向"。**

想象你的照片上每个像素都有一根线从相机穿出去。这根线在世界里是什么方向、经过哪里，用 6 个数字就能精确描述（这叫 Plücker 坐标）。

整段视频的每个像素都有这么一组数字，就形成了一张极其精细的"相机运动地图"。AI 拿到这张地图，就知道每一步相机该在哪、看哪个方向。

然后用了两个"小帮手"来让模型听话：

**帮手一（ControlNet 分支）：** 复制了模型前半部分的一套权重，专门负责把相机指令"硬塞"进生成过程。类似于给画师配一个监工，确保他按你的要求画。

**帮手二（LoRA 分支）：** 在主模型里插入轻量级的可训练模块，让模型适应"静态场景"这个任务（因为视频模型默认会生成动态内容，但 3D 重建需要静态场景）。

两个帮手配合，既能精确控制相机，又能保证画面质量。

---

## 怎么从视频变 3D

拿到视频的压缩表示后，用一个大 Transformer 模型，直接预测每个像素位置对应的一个 3D 高斯点（3D Gaussian Splatting 的基本单元）。

每个高斯点用 12 个数字描述：
- 颜色（RGB 3个）
- 大小（3个方向的尺度）
- 朝向（4个四元数）
- 透明度（1个）
- 深度（1个）

49 帧画面 × 每帧的像素数 = 大约 420 万个高斯点，拼在一起就是一个完整的 3D 场景。

整个过程是直接前馈的，不需要针对每个场景慢慢优化。

---

## 训练的小心思

模型不是一次训成的，分阶段来：

**先低分辨率后高分辨率：** 先用小图训练让模型学会 3D 结构的大框架，再换大图磨细节。就像画画先打草稿再上色。

**先干净数据后野生数据：** 先用标准数据集（有准确相机标注的）训练，再混入用自己生成的"野生"视频微调。因为真实使用时输入的是视频模型生成的 latent，和标准数据集有差异，需要适应这个差异。

**见过和没见过的视角混着监督：** 训练时既要让模型在"见过的画面"上表现好，也要在"没见过的角度"上表现好。如果只练见过的，模型会变成复读机，学不会真正的 3D。

---

## 效果怎么样

**速度：** 一张 A100 显卡，5 分钟出一个 3D 场景。对比一下：
- ZeroNVS 要 3 小时
- Cat3D 要 16 分钟（还用了 16 张卡）
- ViewCrafter 要 6 分钟多

**质量：** 在三个数据集上全面领先。尤其在"没见过的区域"（被遮挡的地方、相机转到背面），Wonderland 明显比其他方法清晰、连贯。

**泛化能力：** 在训练时没见过的场景类型上（Tanks-and-Temples），表现也比对手强，说明 video latent 这个表示确实学到了通用的 3D 知识。

---

## 最妙的地方

这篇 paper 最让我觉得妙的是这个 framing：

**视频生成模型本质上是一个"世界模拟器"。** 它知道世界长什么样、相机移动时画面怎么变化。我们只是用相机控制当"探针"，把这个世界模拟器脑子里隐含的 3D 知识"提取"出来，变成显式的 3D 模型。

这意味着未来视频模型越强，3D 重建就越强——不用重新设计 3D 算法，搭便车就行。

就像 Sora 那篇技术报告说的："视频生成模型是世界模拟器。" Wonderland 给这句话提供了一个具体的证据和落地方式。

---

## 一句话总结

**用视频生成模型当 3D 脑补引擎，一张照片进去，5 分钟出来一个能走进去的 3D 场景。**

---

# Wonderland: 从 Single Image 到 3D Scene 的 Feed-forward 重建

## 1. Problem Context 与 Motivation

这篇 paper 解决一个极具挑战性的 problem: 给定一张 single RGB image, 如何高效地重建一个 wide-scope、high-fidelity 的 3D scene。这个 problem 的难度在于 single view 包含的 geometric 和 contextual information 极其有限, 尤其在 occluded regions 完全没有信息。

现有的方法路线大致分三类, 每一类都有 fundamental limitations:

**第一类: NeRF / 3DGS per-scene optimization**。需要 dense multi-view images 作为 input, 还要漫长的 per-scene optimization。代表工作 [NeRF (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934) 和 [3DGS (Kerbl et al., 2023)](https://repo.aknavi.com/gaussian-splatting)。

**第二类: Image diffusion prior-based 方法**。利用 image diffusion model 的 generative prior, 通过 SDS 或 inpainting 生成 novel views, 再做 3D reconstruction。问题是 image diffusion model 缺乏 spatio-temporal 一致性, 在 occluded regions 容易出现 geometry distortion, 在 background 容易出现 blurriness。代表 [ZeroNVS](https://arxiv.org/abs/2310.17994)、[Cat3D](https://arxiv.org/abs/2406.18588) 等。

**第三类: Feed-forward large reconstruction model**。用 transformer 从 sparse posed images 直接 regress 3DGS 参数。问题是 token 数量爆炸, 高分辨率下内存不可承受, 通常只能做 object-level 或 narrow-scope 的 reconstruction。代表 [LRM](https://arxiv.org/abs/2311.04400)、[pixelSplat](https://arxiv.org/abs/2311.17759)、[GS-LRM](https://arxiv.org/abs/2404.19702)。

Wonderland 的核心 insight: **video diffusion model 的 latent space 天然具有"3D awareness"**。因为 video diffusion model 在海量 video 数据上训练, 学习到了 multi-view 之间的 spatial relationship, 这种 prior 是 image diffusion model 所不具备的。更进一步, 如果能 control video diffusion model 的 camera trajectory, 就能 generate 一个 3D-consistent 的 multi-view 捕捉, 然后 feed-forward 地 reconstruct 出 3DGS。

## 2. Method 整体架构解析

整个 pipeline 由两个核心 module 组成 (见 Figure 2):

1. **Camera-Guided Video Diffusion Model**: 输入 single image + camera trajectory, 输出 3D-aware video latent
2. **Latent Large Reconstruction Model (LaLRM)**: 输入 video latent + camera poses, 输出 3DGS scene

关键设计选择是**在 latent space 而非 pixel space 上做 reconstruction**。这是因为 video latent 提供了 $256\times$ 的 spatiotemporal compression (spatial $8\times$ × temporal $4\times$), 在相同 memory constraint 下能处理更大 scope 的 3D scene。同时 video latent 保留了 perceptual equivalent 的 representation, 因为 3DVAE 在训练时使用了 perceptual loss 和 patch-based adversarial objective。

## 3. Plücker Embedding 的数学细节

为了实现 precise pose control, paper 采用 pixel-level 的 Plücker embedding 而非 frame-level 的 camera parameter。这是一个关键设计, 因为 frame-level 的 (R, t, K) 信息粒度太粗, 无法 capture 同一帧内不同 pixel 的 ray direction 差异。

给定 frame $f$ 的 camera parameter:
- 旋转矩阵 $\mathbf{R}_f \in \mathbb{R}^{3\times 3}$
- 平移向量 $\mathbf{t}_f \in \mathbb{R}^3$
- 内参矩阵 $\mathbf{K}_f \in \mathbb{R}^{3\times 3}$

对于 pixel $(u_f, v_f)$, 首先 unproject 到 camera coordinate 的 ray direction:
$$d_{u_f, v_f} = \mathbf{R}_f \mathbf{K}_f^{-1} [u_f, v_f, 1]^T + \mathbf{t}_f$$

这里 $[u_f, v_f, 1]^T$ 是 pixel 的 homogeneous coordinate, $\mathbf{K}_f^{-1}$ 将 pixel coordinate 映射到 normalized camera coordinate, $\mathbf{R}_f$ 转到 world coordinate, 最后加上 $\mathbf{t}_f$ 得到 ray 上的一个点。

归一化 ray direction:
$$d'_{u_f, v_f} = d_{u_f, v_f} / \|d_{u_f, v_f}\|$$

Plücker coordinate:
$$\dot{p}_{u_f, v_f} = (\mathbf{t}_f \times d'_{u_f, v_f}, d'_{u_f, v_f}) \in \mathbb{R}^6$$

这里 $\times$ 是 cross product。Plücker coordinate 的 6 维分别是:
- 前 3 维: moment $\mathbf{m} = \mathbf{t}_f \times d'$, 编码 ray 经过原点附近的位置信息
- 后 3 维: direction $d'$, 编码 ray 的方向

对整个 video $x \in \mathbb{R}^{T \times H \times W \times 3}$, 每个 pixel 都计算 Plücker coordinate, 得到 $p \in \mathbb{R}^{T \times H \times W \times 6}$ 的 spatiotemporal camera pose 表示。

这个表示的优势: 同一个 camera 的不同 frame 之间, 同一帧内的不同 pixel 之间, 都能 preserve 局部 ray 的差异性, 这对 fine-grained camera control 至关重要。

参考 [Plücker coordinates (Wikipedia)](https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates)。

## 4. Dual-Branch Camera Guidance 详解

这是 paper 的一个 core technical contribution。难点在于: 要让 pretrained video diffusion model (这里是 CogVideoX-5B-I2V) 在生成 static scene 时精确跟随 camera trajectory, 同时保留 visual quality, 还要避免在小规模 static scene 数据集上 overfit。

Paper 设计了 dual-branch 机制, 借鉴 [ControlNet (Zhang et al., 2023)](https://arxiv.org/abs/2302.05543) 和 [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) 的思想。

### 4.1 输入预处理

Plücker embedding $p \in \mathbb{R}^{T \times H \times W \times 6}$ 通过两个独立的 lightweight camera encoder, 各自包含:
- 3D Convolution layers 做 spatiotemporal compression
- Unfolding operation 产生 token sequence
- Zero-linear layer $\mathcal{F}_{d_v, d_v}$ (初始化为 zero) 保证训练稳定

得到两组 camera tokens: $O_{\mathrm{ctrl}}, O_{\mathrm{lora}} \in \mathbb{R}^{N_v \times d_v}$, 长度与 visual tokens $o_v$ 相同。

### 4.2 ControlNet Branch

构建方式:
1. 从 base video model 的前 $N=21$ 个 transformer blocks 复制权重 (base model 共 42 个 block), 创建一个 trainable copy
2. $O_{\mathrm{ctrl}}$ 与 $o_v$ 做 element-wise addition
3. 送入 trainable block
4. 每个 trainable block $i$ 的输出经过 zero-linear layer 后, element-wise add 到对应 frozen block 的输出
5. 这保证了训练初期 ControlNet branch 的 contribution 为零, 不会破坏 pretrained 权重

Ablation study (Table A2) 显示:
- Without weight copy: FID 18.92, $R_{\mathrm{err}}$ 0.065
- block-1: $R_{\mathrm{err}}$ 0.114 (控制能力弱)
- block-10: $R_{\mathrm{err}}$ 0.075
- block-21 (default): $R_{\mathrm{err}}$ 0.058, FID 18.75
- block-30: $R_{\mathrm{err}}$ 0.056, FID 20.15 (质量下降)

21 是 quality 和 controllability 的 sweet spot。

### 4.3 LoRA Branch

构建方式:
1. $O_{\mathrm{lora}}$ 与 $o_v$ channel-wise concatenation, 得到 $\mathbb{R}^{N_v \times 2d_v}$
2. 通过 linear layer $\mathcal{F}_{2d_v, d_v}$, 权重 $W_\mathcal{F} \in \mathbb{R}^{2d_v \times d_v}$
3. $W_\mathcal{F}$ 初始化时, $o_v$ 对应部分为 identity matrix, camera 部分为 zero, 保证初始时输出等同于 $o_v$
4. 输出送入 frozen transformer blocks (main branch), 其中插入 trainable camera-LoRA module (rank = 256)

LoRA branch 的作用是:
- 让 main branch 适配 static scene 数据分布
- 增强 camera 控制能力
- 减少参数量, 避免 overfit

### 4.4 Dual Branch 协同效果

Table 1 的 ablation (RE10K):
- Lora-branch only: FID 19.02, $R_{\mathrm{err}}$ 0.102, $T_{\mathrm{err}}$ 0.157
- Ctrl-branch only: FID 18.75, $R_{\mathrm{err}}$ 0.058, $T_{\mathrm{err}}$ 0.104
- Dual-branch: FID 17.22, $R_{\mathrm{err}}$ 0.052, $T_{\mathrm{err}}$ 0.095

ControlNet branch 主要负责 precise pose control, LoRA branch 在此基础上进一步 refine quality 和 controllability。

## 5. LaLRM 架构详解

### 5.1 输入 tokenization

Video latent $z \in \mathbb{R}^{t \times h \times w \times c}$ (具体为 $13 \times 60 \times 90 \times c$):
- Patch size $p_l = 2$ (spatial)
- 产生 visual latent tokens $o_l \in \mathbb{R}^{N_l \times d_l}$, 其中 $N_l = t \cdot \frac{h}{p_l} \cdot \frac{w}{p_l}$

Plücker embedding $p \in \mathbb{R}^{T \times H \times W \times 6}$:
- Temporal patch size = $r_t = 4$
- Spatial patch size = $p_l \cdot r_s = 16$
- 产生 pose tokens $o_p \in \mathbb{R}^{N_l \times d_l}$ (长度与 $o_l$ 匹配)

两者 channel-wise concatenation, linear projection 降维, 送入 24 个 base transformer blocks (hidden dim = 1024)。

### 5.2 Gaussian Regression

输出 tokens 经过 latent decoding module:
- 3D DeConv layer, upsampling strides = (4, 16, 16) (low-res) 或 (4, 8, 8) (high-res)
- 输出 12-channel Gaussian feature map $G \in \mathbb{R}^{(T \times H \times W) \times 12}$

12 个 channel 对应 3DGS 的参数:
- 3: RGB color
- 3: scale (各 axis 的 standard deviation)
- 4: rotation quaternion (单位四元数)
- 1: opacity $\alpha$
- 1: ray distance (depth)

每个 source video frame 的每个 pixel 对应一个 3D Gaussian, 总共 $T \times H \times W = 49 \times 480 \times 720 = 16,934,400$ 个 Gaussian primitives (实际实现中 high-res 时是 $T \times \frac{H}{2} \times \frac{W}{2} \approx 4,233,600$)。

### 5.3 Loss Function

训练时从 predicted Gaussians 渲染 $V = 48$ 个 supervision views:
$$\mathcal{L}_{\mathrm{recon}} = \lambda_1 \mathcal{L}_{\mathrm{mse}} + \lambda_2 \mathcal{L}_{\mathrm{perc}}$$

- $\mathcal{L}_{\mathrm{mse}}$: pixel-wise mean squared error
- $\mathcal{L}_{\mathrm{perc}}$: VGG-19 based perceptual loss

其中 $V' = 24$ 个 seen views (来自 sampled video clip), $V - V' = 24$ 个 unseen views (clip 外的 frame)。这个 mix 至关重要: 如果只 supervise seen views, 模型会 overfit 到 decoded views, 无法 learn 真正的 3D geometry。

## 6. Progressive Training Strategy

这是另一个关键设计, 分两个 dimension 渐进:

### 6.1 Resolution progression
- Stage 1: low-res video clips $49 \times 240 \times 360$, 对应 latents $13 \times 30 \times 45$, 训练 200K iterations
- Stage 2: high-res $49 \times 480 \times 720$, 对应 latents $13 \times 60 \times 90$, fine-tune 100K iterations, 学习率从 $4 \times 10^{-4}$ 降到 $1 \times 10^{-5}$

### 6.2 Data source progression
- Stage 1: 仅使用 benchmark datasets (RE10K, ACID, DL3DV), 这些有 ground truth camera poses
- Stage 2: 混入 in-the-wild data, 用自己的 camera-guided video diffusion model 生成 20K videos, image prompts 来自 [Flux.1](https://github.com/black-forest-labs/flux)

In-the-wild data 的作用 (Table A3):
- LaLRM- (无 in-the-wild): RE10K PSNR 17.06, Tanks PSNR 15.85
- LaLRM (有 in-the-wild): RE10K PSNR 17.15, Tanks PSNR 15.90

尤其在 out-of-domain (Tanks) 提升更明显, 说明 in-the-wild data 显著增强了 generalization。

### 6.3 Video clip sampling
- RE10K: stride $s \in \{3, 4, 5\}$, 覆盖 150-250 frames 的 scene range
- ACID: $s \in \{1, 2\}$ (frame 数较少)
- DL3DV: $s = 1$ (keyframe 之间 view change 剧烈)

## 7. 实验结果深度分析

### 7.1 Camera-Guided Video Generation (Table 1)

RE10K 数据集:
| Method | FID↓ | FVD↓ | $R_{\mathrm{err}}$↓ | $T_{\mathrm{err}}$↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|---|---|---|---|---|---|---|---|
| MotionCtrl | 22.58 | 229.34 | 0.231 | 0.794 | 0.296 | 14.68 | 0.402 |
| VD3D | 21.40 | 187.55 | 0.053 | 0.126 | 0.227 | 17.26 | 0.514 |
| ViewCrafter | 20.89 | 203.71 | 0.054 | 0.152 | 0.212 | 18.91 | 0.501 |
| **Wonderland** | **16.16** | **153.48** | **0.046** | **0.093** | **0.206** | **19.71** | **0.557** |

Wonderland 在所有 metric 上都 best。特别注意 $R_{\mathrm{err}}$ 和 $T_{\mathrm{err}}$ 是用 COLMAP 从 generated video 中反算 camera pose 后与 ground truth 比较的, Wonderland 的 $R_{\mathrm{err}} = 0.046$ 远低于 MotionCtrl 的 0.231, 证明 Plücker embedding + dual-branch 实现了 precise pose control。

### 7.2 3D Scene Generation (Table 2)

RE10K:
| Method | LPIPS↓ | PSNR↑ | SSIM↑ |
|---|---|---|---|
| ZeroNVS | 0.448 | 13.01 | 0.378 |
| ViewCrafter | 0.341 | 16.84 | 0.514 |
| **Wonderland** | **0.292** | **17.15** | **0.550** |

DL3DV 和 Tanks-and-Temples 上同样领先。Tanks-and-Temples 是 out-of-domain 测试 (训练集不含), Wonderland 的 PSNR 15.90 显著高于 ViewCrafter 的 14.93, 说明 video latent space 的 generalization 能力。

### 7.3 Latent vs RGB Reconstruction (Table 3)

RE10K:
| Method | LPIPS↓ | PSNR↑ | SSIM↑ |
|---|---|---|---|
| RGB-14 | 0.137 | 21.39 | 0.751 |
| RGB-49 | 0.126 | 25.06 | 0.830 |
| Latent (encoder finetuned) | 0.129 | 26.14 | 0.841 |
| **LaLRM (encoder frozen)** | **0.122** | **27.10** | **0.864** |

这里有个微妙之处: 表格里的数字似乎比 Table 2 的 PSNR 17.15 高很多。原因是 Table 3 是 in-domain training distribution 内的 video clips (从 source video 直接 encode latent), 而 Table 2 是用 video diffusion model 生成的 latents (有 distribution shift)。

关键 insight: encoder frozen 比 encoder finetuned 好。如果 finetune encoder, 会破坏 3DVAE 在 web-scale data 上学到的 robust representation, 反而损害 generalization。

### 7.4 Latency 对比

| Method | Latency |
|---|---|
| ZeroNVS | ~3 hours |
| Cat3D | ~16 min (1 min × 16 A100) |
| ViewCrafter | >6 min (25-frame) |
| **Wonderland** | **~5 min** (single A100) |

Wonderland 比 Cat3D 快 3.2×, 比 ZeroNVS 快 36×。

## 8. Mip-NeRF Comparison 与 Out-of-Domain 评估

Figure 6 展示了在 [Mip-NeRF 360](https://arxiv.org/abs/2111.12077) 复杂 scene 上的对比:
- Near conditional view: Wonderland 与 Cat3D 质量相当
- Far from conditional view (~120° rotation): Cat3D 出现 severe background blurriness, Wonderland 保持清晰纹理和 consistency

这说明 Wonderland 的 video diffusion prior 在 wide-scope generation 上有显著优势。

## 9. Limitations 与未来方向

Paper 坦诚地指出:
1. **Inference bottleneck**: Video generation 占了 pipeline 大部分时间, 可以用 [xDiT](https://github.com/xdit-project/xDiT) 之类并行推理加速
2. **Static scene only**: 偶尔会出现 motion, 影响 reconstruction, 未来可扩展到 4D

我认为还有几个潜在 limitation paper 没充分讨论:
- **Camera trajectory distribution**: 训练数据的 trajectory 主要来自 RE10K (real estate walkthrough), 可能对 extreme camera motion (如 aerial 360°) generalization 不足
- **Geometric precision**: 3DGS 表示对 thin structure 和 transparent material 仍有限制
- **Identity preservation**: 生成 scene 的 fine-grained texture 与 input image 的 identity 保持程度未充分量化

## 10. 与相关工作的深层对比

### 10.1 vs ViewCrafter

ViewCrafter 也用 video diffusion model + 3DGS, 但关键差异:
- ViewCrafter 用 incomplete point cloud 作为 video model 的 conditional frame, 在 occluded region 有 black artifacts
- ViewCrafter 用 per-scene optimization, Wonderland 是 feed-forward
- ViewCrafter scope 有限, Wonderland 能做 wide-scope

### 10.2 vs Cat3D

Cat3D 用 image diffusion model 生成 multi-view, 再做 per-scene 3DGS optimization:
- Cat3D 在 near view 质量不错, 但 far view 出现 blurriness (image diffusion 缺乏 multi-view consistency)
- Wonderland 的 video diffusion prior 在 consistency 上有 structural advantage
- Cat3D 慢 3.2×

### 10.3 vs GS-LRM / pixelSplat

这些是 feed-forward large reconstruction model, 但从 sparse posed images 直接 regress 3DGS:
- Token 数量爆炸: 8×8 patchify 时 260K+ tokens, 无法 scale
- Wonderland 通过 video latent 的 256× compression 解决了这个 bottleneck

## 11. 我的 Intuition 与思考

### 11.1 为什么 video latent 是好的 bridge?

Video diffusion model 训练时学到的是 "world 的 temporal evolution", 这种 evolution 包含了 implicit 3D structure。当 camera 在 static scene 中移动时, video model 实际上是在 hallucinate "如果 camera 这么移动, 会看到什么", 这本质上是 3D understanding。

更妙的是, video latent space 是高度 compressed 的, 这 forced model 学到 high-level semantic 和 structural representation, 而非 pixel-level detail。这种 representation 对 reconstruction 反而更友好, 因为它过滤掉了 spurious detail, 保留了 essential structure。

### 11.2 Dual-branch 为什么 effective?

ControlNet branch 负责 "硬约束" - 精确的 camera pose 控制, 通过 deep feature integration 实现。LoRA branch 负责 "软适应" - 让 model 适配 static scene 分布, 同时提供 additional camera 信号。

两者互补: ControlNet 提供 structural guidance, LoRA 提供 distribution adaptation。单独 ControlNet 的 $R_{\mathrm{err}}$ 已经很好 (0.058), 但 visual quality (FID 18.75) 略差; 加上 LoRA 后 quality 提升到 17.22, 说明 LoRA 帮助 model 更好地生成 static scene 而非 dynamic content。

### 11.3 Progressive training 为什么必要?

Video latent space 和 3DGS space 之间有 large domain gap:
- Video latent 是 compressed perceptual representation
- 3DGS 是 explicit 3D primitive

直接在 high-res 上训练, model 容易 overfit 到 seen views (因为高 res 提供太多 spurious detail), 无法 learn 真正 3D geometry。

Low-res → high-res 的 progression 让 model 先 learn coarse 3D structure, 再 refine detail。

Benchmark → in-the-wild 的 progression 则是先在 clean data 上 learn mapping, 再 adapt to video diffusion model 生成的 noisy latent。

### 11.4 未来可能的扩展

我想到几个方向:
1. **Text-conditioned 3D scene**: 用 text prompt 控制 scene 内容, 用 camera trajectory 控制 view, 生成 3DGS
2. **Dynamic scene**: 用 4DGS 替代 3DGS, video latent 本身就包含 temporal info, 可以自然扩展
3. **Iterative refinement**: 第一轮生成的 3DGS 可以作为下一轮 video diffusion 的 conditional, 实现 coarse-to-fine
4. **Multi-image input**: 扩展到 sparse view input, 类似 MVS, 但用 video latent 作为 unified representation
5. **Real-time application**: 用 distillation 把 video diffusion model 压缩成 real-time 模型

### 11.5 与 Sora 等 world model 的关系

这篇 paper 暗示了一个更深的 connection: video generation model 本质上是 world simulator ([Sora technical report](https://openai.com/research/video-generation-models-as-world-simulators))。当我们能用 camera 控制 video generation 时, 实际上是在 "查询" world model 的 3D structure。

Wonderland 可以看作是: 从 video world model 中 distill 出 explicit 3D representation (3DGS) 的一种方法。这个 framing 很 powerful, 因为它意味着未来更强大的 world model 会直接带来更强的 3D reconstruction 能力, 而无需重新设计 3D reconstruction algorithm。

## 12. 实现细节补充

### 12.1 Hyperparameter summary

- Base video model: CogVideoX-5B-I2V, 49 frames @ 480×720
- 3DVAE: $r_t = 4$, $r_s = 8$, latent dim $13 \times 60 \times 90$
- ControlNet: first 21 of 42 transformer blocks, weight copied
- Camera-LoRA: rank 256
- LaLRM: 24 transformer blocks, hidden dim 1024
- Patch size: $p_l = 2$ (latent), 4 temporal × 16 spatial (Plücker)
- Training: Adam, $\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $1 \times 10^{-4}$
- Video diffusion: batch 24, 40K steps, lr $2 \times 10^{-5}$
- LaLRM: batch 24, 200K (low-res) + 100K (high-res), cosine annealing, peak lr $4 \times 10^{-4} \to 1 \times 10^{-5}$
- Optimization: FlashAttention V2, BF16 mixed precision

### 12.2 数据集细节

- [RealEstate10K (RE10K)](https://google.github.io/realestate10k/): ~80K videos, real estate walkthrough
- [ACID](https://infinite-nature.github.io/): 11K train + 20K test, natural landscapes
- [DL3DV](https://github.com/DL3DV-10K/DL3DV-10K-Dataset): DL3DV-10K train + DL3DV-140 test, indoor/outdoor
- [Tanks-and-Temples](https://www.tanksandtemples.org/): 14 scenes, 用 COLMAP 标注 pose
- In-the-wild: 20K videos, image prompts from Flux.1, poses from RE10K

### 12.3 评估细节

- Video generation: 前 14 frames 用于 similarity metrics (因为后续 frame 会 deviate from conditional view)
- Camera pose error: COLMAP 反算后, 相对 first frame 归一化, 前 16 frames 平均
- 3D scene: 同样前 14 frames 的 rendering 评估

## 13. 总结

Wonderland 的核心贡献是**将 video diffusion model 的 latent space 作为 image space 和 3D space 之间的 bridge**, 通过 dual-branch camera conditioning 实现精确控制, 通过 LaLRM 在 latent space 上高效 reconstruct 3DGS。

这个 framework 的 deep insight 是: video diffusion model 已经学到了 implicit 3D structure, 我们只需:
1. 用 camera conditioning "extract" 出 multi-view consistent 的 representation
2. 用 large reconstruction model "decode" 出 explicit 3D representation

整个 process 是 feed-forward 的, 无需 per-scene optimization, 且在 wide-scope 和 out-of-domain 上 generalize well。

这种 "generative prior + feed-forward reconstruction" 的范式很可能成为 future 3D scene generation 的主流路线, 尤其当 video diffusion model 持续 scale up 时。

Project page: https://snap-research.github.io/wonderland/

相关参考:
- [CogVideoX](https://arxiv.org/abs/2408.06072)
- [3D Gaussian Splatting](https://repo.aknavi.com/gaussian-splatting)
- [ControlNet](https://arxiv.org/abs/2302.05543)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [pixelSplat](https://arxiv.org/abs/2311.17759)
- [GS-LRM](https://arxiv.org/abs/2404.19702)
- [ViewCrafter](https://arxiv.org/abs/2409.02048)
- [ZeroNVS](https://arxiv.org/abs/2310.17994)
- [NeRF](https://arxiv.org/abs/2003.08934)
- [Plücker coordinates](https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates)
- [Sora as world simulator](https://openai.com/research/video-generation-models-as-world-simulators)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
