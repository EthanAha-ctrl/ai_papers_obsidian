---
source_pdf: CAT4D Create Anything in 4D with Multi-View Video Diffusion Models.pdf
paper_sha256: cc7a751e171dcf037ee00ac5cda7493a368e53ebd2b42bb5815ce0ea2707ba75
processed_at: '2026-08-03T15:07:20-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CAT4D 用人话说

Andrej, 好，让我把那些公式和表格扔一边，用最接地气的方式重新讲一遍。

## 一句话版本

**你拍一段手机视频，他们帮你变成一个能任意转动视角、任意回放时间的4D场景。**

就这么简单。你拿iPhone绕着一只猫转一圈拍下来，CAT4D吐出来一个东西，你能从猫的头顶看、从地板往上看、还能看猫上一秒在干嘛。

## 为什么这事难

想象你在拍一只奔跑的狗。你绕着狗走，狗也在跑。你手上的视频里，**每一帧的像素变化其实混了两个东西**：你自己相机的移动 + 狗的移动。

要从这段2D视频还原出3D + 时间的4D模型，你得知道：
- 这块像素变化是**因为相机动了**（那这是几何问题，换个视角会看到同样的东西）
- 还是**因为物体动了**（那这是动态问题，换个时间会看到不同的东西）

人能分，但算法很难分。而且更糟的是：**狗的背面你根本没拍到**，你得猜。

之前的人怎么解决？两个套路：

**套路A**（Shape-of-Motion, MoSca这种）：用一堆别的模型先估depth、估2D track、估segmentation mask，然后拿这些当supervision signal去约束4D重建。问题是：你永远reconstruct不出你从没看到过的区域，比如狗的背面。而且需要用户手动点击物体mask。

**套路B**（4DiM这种）：训练一个diffusion model直接生成novel view + novel time。但是模型搞不清楚"我要你只变camera别变时间"和"我要你只变时间别变camera"的区别，两个信号纠缠在一起。

CAT4D说：**我两个套路都不用**。我就训练一个model，它能学会"世界在没看到过的视角和时间下应该长什么样"，直接hallucinate出来，再用这些"假"的多视角视频去fit一个标准的4D Gaussian模型。

## 他们到底怎么搞的

### 核心思路

把一个monocular video "扩写"成multi-view video。

你原本有 $L$ 帧（1个相机视角 × $L$ 个时间点），CAT4D给你生成一个 $K \times L$ 的grid（$K$ 个相机视角 × $L$ 个时间点）。这就像把你的一段视频，变成 $K$ 台虚拟摄像机同时从不同角度拍的同步视频。

有了这个multi-view video，4D重建就是个solved problem了——4D-GS之类的标准方法就能fit。

所以真正的magic在**前面那个diffusion model**。

### 这个model凭什么能干这事

它继承自CAT3D，Google之前做的multi-view image diffusion model。CAT3D能做"给几张图，生成任意视角的图"——但只有静态场景。

CAT4D在它基础上加了一个东西：**time conditioning**。每个frame除了有camera参数，还有个timestamp。模型学到："这个frame是在哪个时间点、从哪个角度看到的"。

听起来简单，但有个大坑：**你训练数据里根本没有大规模的multi-view video dataset**。同步多视角摄像机拍动态场景，这种数据极少，全世界加起来都不够训练一个好model。

### 数据问题的trick

这是paper最聪明的地方。他们画了个2×2的表格：

| | 相机不动 | 相机动 |
|---|---|---|
| **场景不动** | 多视角静态图数据集（CO3D等） | 有 |
| **场景动** | 固定机位的video（YouTube之类） | **没有！** |

右下角那个格子（相机和场景都动）是空的，而这恰恰是测试时最常见的情况。

怎么办？**他们用已有的model生成假数据来填这个格子**：

- 拿CO3D的静态多视角图，用Lumiere video model给animate起来（相机保持不动，让物体动）→ 得到"多相机 + 动态场景"的伪数据
- 拿固定机位的video，用CAT3D给每帧生成novel view → 得到"动相机 + 动场景"的伪数据

这种bootstrap的做法，用已有的strong prior去填补真实数据的空白，非常pragmatic的工程智慧。

### Sampling strategy: 绕过model capacity限制

model一次只能生成N=8到13帧，但4D重建需要 $K \times L$ 可能几百帧。

他们搞了个**alternating sampling**：

1. 先multi-view sampling（固定时间，生成所有相机视角）——空间一致，时间不一致
2. 再temporal sampling（固定相机，生成所有时间点）——时间一致，空间不一致  
3. 来回迭代，用SDEdit加噪再denoise

直觉理解：**就像两个人在修补同一个grid，一个负责让每一列（同一时间不同视角）一致，另一个负责让每一行（同一视角不同时间）一致，来回几轮就收敛了**。

这个trick让一个只能生成13帧的model，能生成几百帧一致的multi-view video。不是最优雅的解法（作者也承认，未来直接训练更大的model更好），但当下work。

### 最后一步：标准4D重建

有了multi-view video，就用4D-GS（3D Gaussian + K-Planes deformation field）去fit，standard photometric loss。没什么花活。

唯一小trick：真实input image的loss权重保持1.0，生成image的loss权重从1.0慢慢退火到0.5。意思是：**后期更相信真实数据，生成的假数据只是用来"撑起"unobserved区域的脚手架**。

## 效果如何

- **解耦控制**：比4DiM好不少。让4DiM"只变相机不变时间"，物体还是会动；CAT4D真的能分开。
- **4D重建**：在DyCheck上，mPSNR 18.24，不如MoSca的19.54，但LPIPS更好（0.227 vs 0.244）。关键是CAT4D不需要任何external supervision，MoSca要用depth + 2D tracks + 用户手动点击mask。
- **4D generation**：能处理scene-scale的生成（多个动态物体在动态环境中），之前的工作基本只能搞single object。

## 我觉得哪里真的聪明

1. **数据augmentation的bootstrap**：用Lumiere + CAT3D互相补对方的数据盲区，填满camera×time空间的2×2格子。这种"用现有strong model生成训练数据"的思路在data-scarce的4D领域会越来越常见。

2. **Dual-scale CFG**：把time conditioning和image conditioning分开做classifier-free guidance，用不同的guidance scale独立强化。$s_T=4.5$ 控时间一致性，$s_I=3.0$ 控其他一致性。这种hierarchical CFG对entangled task很关键。

3. **Alternating sampling**：虽然是个workaround，但思路清晰——把一个2D grid的约束分解成行约束和列约束，交替满足。这种decomposition trick在很多地方都work。

## 我觉得哪里一般

1. **Alternating sampling本质是个band-aid**。作者自己承认，如果能直接训练一个large multi-view video model生成几百帧，就不需要这个trick了。现在video model scaling这么快，这个trick可能一年后就过时。

2. **Temporal extrapolation不行**：超出input frame时间范围生成困难。这其实限制了"4D creation"——你不能从一段2秒视频生成一个10秒的4D场景。

3. **Motion field物理不准确**：生成的场景视觉上plausible，但underlying的3D motion可能不是physical accurate的。对于robotics这种需要精确geometry的应用，还是不够。

## 总结

**CAT4D = CAT3D (multi-view image prior) + time conditioning + 数据augmentation bootstrap + alternating sampling + 4D-GS**

核心贡献在前面三步，把"无supervision的4D重建"这个问题用generative prior的角度重新attack了。不是第一个用diffusion prior做4D的，但是第一个做到scene-scale、disentangled control、且不需要external supervision的。

从intuition层面，这篇paper验证了一个重要假设：**只要你能train一个足够好的multi-view video generative model，4D重建就成了一个"solved"的downstream task**。这跟CAT3D对static 3D做的事情是同一个故事——generative prior吃掉geometric reconstruction。

未来方向很明显：更大的multi-view video model，直接生成更多帧，不需要alternating sampling这种trick。但这篇paper在当下给出了一个非常practical的解决方案。

希望这个版本更对味，Andrej。说白了就三个idea的精妙组合：**bootstrap数据、解耦conditioning、交替sampling**。

---

# CAT4D 深度解析

Andrej, 这篇CAT4D的核心idea其实非常优雅, 让我从intuition层面到技术细节全面拆解一下。

## 1. Core Intuition: 解耦Camera和Time的生成Prior

CAT4D要解决的根本问题是: 给定一个monocular video (相机在动, 场景也在动), 如何重建一个4D场景?

传统pipeline的痛点在于**under-constrained**: 单视角视频无法提供occluded regions的信息, 也没有multi-view几何约束。Prior work像Shape-of-Motion和MoSca靠**外部supervision signals** (depth maps, 2D tracks, segmentation masks) 来regularize, 但是这些方法无法recover从未观测到的区域。

CAT4D的key insight: **训练一个multi-view video diffusion model作为generative prior**, 把monocular video "hallucinate"成multi-view videos, 再用标准4D-GS重建。这等价于让diffusion model学到scene在unobserved viewpoints和timestamps下应该长什么样的distribution。

关键挑战在于**disentanglement**: 模型需要区分camera motion (视角变化) 和scene dynamics (物体运动)。CAT4D通过精心curate的training data和conditional architecture来实现这个解耦。

参考CAT3D原作: [CAT3D Project](https://cat3d.github.io/)

## 2. Architecture: 在CAT3D基础上注入Time Conditioning

### 2.1 Base架构

CAT4D build on CAT3D的multi-view latent diffusion model:
- Latent diffusion: 在VAE的latent space操作
- 3D self-attention: 连接所有image latents (M+N个frames), 这是关键设计, 让不同view和时间点的tokens互相通信
- U-Net backbone with residual blocks

### 2.2 Time Conditioning Injection

对每个timestamp $t \in T^{cond} \cup T^{tgt}$:
1. 用sinusoidal positional embedding编码 (来自Transformer [Vaswani et al. 2017])
2. 过一个2-layer MLP
3. Add到diffusion timestep embedding
4. Project后add到每个residual block

这种设计使得time信号渗透到整个U-Net, 而camera parameters仍通过Plücker embedding等方式注入 (继承自CAT3D)。

### 2.3 Conditional Dropout for CFG

训练时随机dropout:
- 7.5%概率drop掉 $c_T = (T^{cond}, T^{tgt})$ — 只保留image conditioning
- 7.5%概率同时drop掉 $c_T$ 和 $c_I = (P^{cond}, I^{cond})$ — 完全unconditional

这enable了**two-scale classifier-free guidance**。

## 3. Two-Scale CFG: 数学推导

标准CFG公式扩展为双guidance scale:

$$
\epsilon_\theta(z^{tgt}(i), P^{tgt}, \emptyset, \emptyset) \\
+ s_I \cdot [\epsilon_\theta(z^{tgt}(i), P^{tgt}, c_I, \emptyset) - \epsilon_\theta(z^{tgt}(i), P^{tgt}, \emptyset, \emptyset)] \\
+ s_T \cdot [\epsilon_\theta(z^{tgt}(i), P^{tgt}, c_I, c_T) - \epsilon_\theta(z^{tgt}(i), P^{tgt}, c_I, \emptyset)]
$$

变量解释:
- $z^{tgt}(i)$: 所有target images的latents在diffusion timestep $i$ (噪声水平)
- $P^{tgt}$: target cameras (始终保留, 因为camera control是核心)
- $c_I = (P^{cond}, I^{cond})$: image conditioning (conditional cameras + conditional images)
- $c_T = (T^{cond}, T^{tgt})$: time conditioning (conditional + target timestamps)
- $s_I = 3.0$: image guidance scale, 强化与input图像及其他非时间条件的一致性
- $s_T = 4.5$: time guidance scale, 强化时间对齐

**Intuition**: 第一项是unconditional baseline, 第二项push模型向"只看image条件"的方向, 第三项在image条件基础上**额外**push向"加上时间条件"。这种hierarchical CFG让模型可以独立强化camera consistency和temporal consistency, 这对4D这种camera-time entangled任务至关重要。

参考4DiM的dual-scale CFG设计: [4DiM arXiv](https://arxiv.org/abs/2407.07860)

## 4. 数据工程: 一个令人叹服的Mixture

这是CAT4D最巧妙的部分。没有大规模real multi-view video dataset存在, 所以作者组合了多种数据源, 每种覆盖camera×time space的一个quadrant:

| Input camera/scene motion | Target camera/scene motion | Data source |
|---|---|---|
| Static / Static | Static / Static | CO3D, MVImgNet, Re10K, MC4K |
| Static / Static | Static / Dynamic | Static-viewpoint videos |
| Dynamic / Static | Static / Dynamic | CO3D augmented with Lumiere |
| Dynamic / Dynamic | Dynamic / Static | Static videos augmented with CAT3D |
| - | Static / Static | Single image (1% prob) |
| Any | Any | Kubric, Objaverse (synthetic 4D) |

**两个Augmentation策略的妙处**:

1. **Lumiere augmentation**: 拿CO3D的static多视角图, 用Lumiere video model animate每个视角 (保持相机不动), 这样得到"input有多个相机视角+多个时间点"的伪4D数据, ~24k sequences。

2. **CAT3D augmentation**: 拿static-viewpoint video, 用CAT3D给每帧生成novel views, 这样得到"input有相机运动+时间运动"的伪4D数据, ~160k sequences。

这种self-augmentation bootstrapping利用了已有模型的strong prior来填补真实4D数据的空白, 非常精妙。

**Static-viewpoint filtering**: 通过检查video四角patches (10×10) 是否随时间变化来判断。具体计算连续帧的L2 distance, threshold 0.05。简单但有效, 避免相机motion污染time control signal。

参考Lumiere: [Lumiere arXiv](https://arxiv.org/abs/2401.12945)
参考CAT3D: [CAT3D arXiv](https://arxiv.org/abs/2406.08752) (实际是NeurIPS 2024)

## 5. Sampling Strategy: Alternating Multi-view + Temporal

这是另一个亮点。模型原生只能生成N=8-13帧, 但4D重建需要 $K \times L$ 网格 (K cameras × L timestamps), 通常 $KL \gg N$。

### 5.1 问题分解

记 $G_{K,L}$ 为 $K \times L$ 的image grid。两种基础sampling:

**Multi-view sampling** (固定时间t, 生成所有K个相机视角):
- 对第j-th sliding window (size N), condition on N个input frames在对应target cameras $\{I_c | c \in \{k_{i \mod K}\}_{i=j}^{j+N}\}$ + 1个input frame在target time $I_t$
- 多个window的结果取pixel-wise median

**Temporal sampling** (固定相机k, 生成所有L个时间点):
- 类似multi-view, 但是swap camera和time的角色
- Condition on N个input frames在对应target timestamps + 1个input frame在target camera

### 5.2 Alternating with SDEdit

核心思想: **multi-view sampling保证空间一致性但时间不一致, temporal sampling保证时间一致性但空间不一致, 用SDEdit在两者间迭代refine**。

3 iterations:
1. Multi-view sampling from random noise (25 DDIM steps)
2. Temporal sampling, 用SDEdit从iteration 1的结果开始, noise level 16/25
3. Multi-view sampling, SDEdit从iteration 2开始, noise level 8/25

**Intuition**: 每次迭代加较少noise (相对前次), 然后denoise时用另一种conditioning引导, 逐步让grid在两个dimension都一致。这种coarse-to-fine的noise schedule类似SDEdit的image editing思路。

参考SDEdit: [SDEdit arXiv](https://arxiv.org/abs/2108.01073)

### 5.3 Stationary Video处理

如果input video几乎没有相机运动 (e.g. text-to-video生成的视频), 需要额外生成novel viewpoints:
1. 先在 $t=0$ 生成K个novel views
2. 把这些frames加入input set
3. 运行alternating sampling

### 5.4 Dense View Sampling

为了进一步增加coverage, 对每个timestamp, condition on已生成的K个views, 用nearest-anchoring生成 $K'=128$ 更多views。

## 6. 4D Reconstruction: Deformable 3D Gaussians

最终用4D-GS [Wu et al. 2024] 重建:
- Canonical space 3D Gaussians + K-Planes deformation field
- Loss: L1 (0.8) + DSSIM (0.2) + LPIPS (0.4)
- 两阶段优化:
  - Stage 1: 只优化canonical Gaussians, 用t=0的images, 2000 iterations
  - Stage 2: 联合优化Gaussians和deformation field, 18000 iterations
- Annealing trick: 生成图像的loss multiplier从1.0 linearly anneal到0.5, 而real input images保持1.0 — 让模型在后期更信任真实input, 减少生成图像artifacts的影响
- Total ~25 min on A100

参考4D-GS: [4D-GS arXiv](https://arxiv.org/abs/2402.07388)

## 7. 实验结果深度解读

### 7.1 Disentangled Control (NSFF dataset)

| Setting | Method | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| Fixed Viewpoint, Varying Time | 4DiM | 19.77 | 0.540 | 0.195 |
| | CAT4D | **21.97** | **0.683** | **0.121** |
| Varying Viewpoint, Fixed Time | 4DiM | 18.81 | 0.428 | 0.219 |
| | CAT4D | **21.68** | **0.588** | **0.105** |
| Varying Viewpoint, Varying Time | 4DiM | 17.28 | 0.378 | 0.256 |
| | CAT4D | **19.73** | **0.533** | **0.155** |

**Intuition**: 4DiM的key failure mode是conflate camera和scene motion — 即使指示只变camera, 动态物体也会动。CAT4D通过精心设计的训练数据 (特别是两个augmentation填补quadrant) 和dual-scale CFG实现了真正解耦。

### 7.2 Sparse-View Bullet-Time 3D Reconstruction

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| CAT3D-1cond | 15.33 | 0.379 | 0.527 |
| CAT3D-3cond | 20.19 | 0.568 | 0.258 |
| CAT4D | **20.79** | **0.576** | **0.160** |

**Intuition**: CAT3D-3cond因为input有时间不一致 (scene in motion), 导致blurry renderings。CAT3D-1cond虽然不需要处理dynamic inconsistency, 但是无法用其他帧的information确定global scene scale。CAT4D的multi-view video prior同时解决了这两个问题。

### 7.3 4D Reconstruction (DyCheck)

| Method | mPSNR | mSSIM | mLPIPS |
|---|---|---|---|
| 4D-GS | 16.54 | 0.594 | 0.347 |
| Shape-of-Motion | 16.72 | 0.630 | 0.450 |
| MoSca† | 19.54 | 0.738 | 0.244 |
| CAT4D† | 18.24 | 0.666 | 0.227 |

†: half resolution

**Intuition**: MoSca和Shape-of-Motion依赖大量external supervision (depth, tracks, segmentation, user clicks), 而CAT4D仅用straightforward photometric loss + generative prior。虽然mPSNR略低于MoSca, 但在LPIPS上更好, 说明perceptual quality更高。CAT4D的generality更强, 不需要per-scene tuning或user interaction。

参考Shape-of-Motion: [SoM arXiv](https://arxiv.org/abs/2407.13764)
参考MoSca: [MoSca arXiv](https://arxiv.org/abs/2405.17421)

### 7.4 Sampling Strategy Ablation

| Strategy | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Independent multi-view | 20.27 | 0.525 | 0.136 |
| Independent temporal | 21.63 | 0.615 | 0.130 |
| Multi-view only | 22.34 | 0.609 | 0.217 |
| Temporal only | 23.36 | 0.681 | 0.145 |
| **Alternating** | 22.15 | 0.633 | **0.108** |

**Intuition**: 注意LPIPS上alternating最好, 虽然PSNR不是最高。Temporal only虽然PSNR最高但LPIPS差, 因为multi-view不一致导致perceptual artifacts。Alternating在两个dimension都一致, perceptual质量最佳。

### 7.5 Training Data Ablation

| Training Data | Fixed View/Varying Time PSNR | Varying View/Fixed Time PSNR | Varying View/Time PSNR |
|---|---|---|---|
| Synthetic only | 22.19 | 21.41 | 19.50 |
| No augmentation | 20.84 | 22.03 | 19.41 |
| All datasets | **22.49** | 21.86 | **19.74** |

**关键发现**: 
- Synthetic only: 已经能给surprisingly good的camera-time control, 但generated scene motion往往不自然, generalization差
- No augmentation: 主要failure mode是"fixed viewpoint, varying time"时camera仍然动, 因为static-view filtering不完美
- All datasets: 全面最好, augmentation对disentanglement至关重要

## 8. Limitations & Future Directions

作者诚实指出:
1. **Temporal extrapolation**: 超出input frame时间范围生成困难
2. **Occlusion disentanglement**: 动态物体被occluded时无法完全解耦
3. **Physical accuracy**: 生成的4D场景viewable但motion field可能不physical accurate
4. **Scale**: 直接训练更大multi-view video model能直接生成更长sequence是未来方向

## 9. 与Concurrent Work的关系

- **DimensionX**: 用多个LoRAs训练video model, 每个针对特定camera motion类型。限制于predefined camera trajectories。
- **4DiM**: cascaded diffusion model, 但camera和time control conflate。
- **4D-fy, TC4D, Consistent4D等**: 基于SDS的4D generation, 主要聚焦single object, 计算昂贵。

CAT4D的优势: single unified model, 任意input数量, scene-scale (多动态物体), 无需external supervision。

参考DimensionX: [DimensionX arXiv](https://arxiv.org/abs/2411.04928)

## 10. 我对这篇paper的看法

从intuition角度, CAT4D最让我印象深刻的有三点:

1. **数据工程的智慧**: 通过两个self-augmentation策略填补真实4D数据空白, 这是engineering pragmatism的典范。Lumiere + CAT3D的组合让模型能学到camera-time space的所有quadrant。

2. **Alternating sampling的简洁性**: 用SDEdit在multi-view和temporal sampling之间迭代, 简单但有效。这种"用已有model的强大prior + 聪明sampling策略绕过model capacity限制"的思路在很多recent work中都有体现。

3. **Disentanglement的explicit design**: 不是依赖model自己学到解耦, 而是通过data curation + conditional dropout + dual-scale CFG三个层面explicitly enforce disentanglement。这种explicit inductive bias在数据稀缺的4D setting下尤其重要。

从research方向上, 这篇paper指向了一个interesting future: 随着video model scaling, 直接训练large multi-view video model可能让alternating sampling这种trick变得不必要, 但目前这种"strong prior + smart sampling"是practical的最佳path。

更多细节建议看project page: [CAT4D Project](https://cat-4d.github.io/) 和supplementary video。

希望这个解析对你build intuition有帮助, Andrej! 这篇paper的精髓在于把data engineering和sampling strategy做到了极致, 来弥补model capacity和数据规模的限制。
