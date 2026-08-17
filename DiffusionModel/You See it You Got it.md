---
source_pdf: You See it You Got it.pdf
paper_sha256: 8874df3265d53d9b34ef1e78d3982bcf469e0290ae65368faf1834ca291c5e28
processed_at: '2026-08-13T06:33:35-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 See3D

## 一句话总结

哥们，这篇paper干的事就是：**互联网上有海量的video，每个video其实就是一组多视角照片，但是没有camera pose标注。能不能让model纯靠"看"这些video，学会3D生成？答案是能，秘诀在于构造一个特殊的"visual condition"来替代pose。**

---

## 1. 为什么这件事难

先说背景。现在做3D生成，大家都在用什么数据？

- **Objaverse**: 80万个合成3D物体
- **RealEstate10K**: 8万个有pose的video clip  
- **MVImgNet**: 22万组多视角照片
- **DL3DV**: 1万个scene

这些数据量级都在**百万以下**。而你看2D image generation，LAION-5B是50亿张图，差了三四个数量级。所以3D生成的scaling law一直跑不起来。

问题是，**去哪搞海量3D数据？**

人工建模太贵，COLMAP跑SfM对web video不靠谱（很多video根本没parallax，或者dynamic太多，SfM直接发散）。Google那帮人搞CAT3D（https://arxiv.org/abs/2405.10314）用了大量pose-annotated数据，但学术界搞不起。

**但你想啊，Internet video本身就是多视角观测**——你拿着手机拍一段视频，就是在对同一个3D scene做连续的多视角采样。25M个video，44年的时长，这就是天然的3D数据矿。唯一的问题是：**没有pose**。

那能不能在pose-free的情况下训练multi-view diffusion model？

---

## 2. 传统MVD为什么需要pose

先回顾下现有的multi-view diffusion model怎么控制camera。典型方法：

- **MVDream**（https://arxiv.org/abs/2308.16512）: 输入camera extrinsics（一个4×4矩阵）
- **Zero-1-to-3**（https://arxiv.org/abs/2303.11308）: 输入relative camera pose
- **CamCo**（https://arxiv.org/abs/2406.02509）: 输入Plücker rays（每条像素一条ray）
- **ViewCrafter**（https://arxiv.org/abs/2409.02048）: 输入warped image（基于已知pose把reference image投影到目标视角得到的"半成品图"）

前三种都是**3D-inductive condition**——直接把camera参数塞进model。第四种是**2D-inductive**——用一张被warp过的图暗示camera运动方向。

See3D想做的事是：**能不能不用pose，纯靠visual signal就让model学会camera control？**

---

## 3. See3D的核心idea：visual condition

### 3.1 直觉

你想想warp image长什么样：它有hole（被occlude的区域）、有stretched pixel（depth不准导致的拉伸）、有artifact。本质上就是一张**corrupted image**。

那video frame呢？如果我对video frame做random masking + 加noise，它也是一张corrupted image。

**如果这两种corruption的distribution够接近，model在video frame上学到的能力就能transfer到warp image上。**

这就是整个paper的intuition：**构造一种task-agnostic的corruption方式，让training-time的video和inference-time的warp image能对齐。**

### 3.2 具体怎么做

训练时，从WebVi3D采一个video clip $X_0 = \{x_0^i\}_{i=1}^N$，N个frame。随机选S个作为reference view（保持clean），剩L个作为target view。

对target view做三件事：

**第一，random masking**

用irregular mask把target view的一部分遮掉。reference view不mask。这让model不能完全依赖像素级hint，必须学习3D结构。

**第二，time-dependent noise**（Eq. 2）

$$C_t = \sqrt{\bar{\alpha}_{t'}} (1-M) X_0 + \sqrt{1-\bar{\alpha}_{t'}} \epsilon$$

这里：
- $M$是mask，1表示遮掉
- $(1-M)X_0$是mask后的clean image
- $t' = f(t) = \beta \cdot t$，其中$\beta = 0.2$
- $\bar{\alpha}_{t'}$是DDIM的累积variance schedule
- $\epsilon$是标准Gaussian noise

**关键点**：用$t' = 0.2t$而不是$t$，意味着$C_t$的noise level比standard diffusion的$X_t$低很多。$C_t$保留了更多clean信息，作为visual hint更清晰。

但这里有个问题：如果$C_t$比$X_t$信息多，model可能直接copy $C_t$的内容输出，不学习真正的multi-view generation。这就是**signal leakage**（参考https://arxiv.org/abs/2406.15735）。

**第三，time-dependent mixture**（Eq. 3）——解决leakage

$$V_t = [W_t \cdot C_t + (1-W_t) \cdot X_t; M]$$

$W_t$是一个随t变化的weight：
- 大timestep（t=1000，全noise）: $W_t \approx 1$，用$C_t$（有hint）
- 小timestep（t→0，接近clean）: $W_t \to 0$，用$X_t$（model自己的denoising结果）

具体的piecewise function（Appendix C.3）：

$$W_t = \begin{cases} 
1 - (1-v_{\text{decay\_end}}) \cdot \frac{t_{\text{peak}} - t}{t_{\text{peak}} - t_{\text{decay\_end}}}, & t \geq 300 \\
v_{\text{decay\_end}} \cdot e^{-b(t_{\text{decay\_end}} - t)}, & t < 300
\end{cases}$$

其中$t_{\text{peak}}=1000$, $t_{\text{decay\_end}}=300$, $v_{\text{decay\_end}}=0.8$, $b=0.075$。

**用人话讲**：denoising的前期，model靠visual hint（$C_t$）知道要生成什么；denoising的后期，model靠自己的latent（$X_t$）做精细refine，避免直接copy hint。这就像画画：先有草图，再自己细化。

### 3.3 为什么这个设计能generalize到warp image

训练时model见到的是：**clean reference + masked noisy target**。

Inference时用的是：**clean reference + warped image with hole mask**。

warp image的hole mask就相当于random mask，warp image的artifact就相当于noise。所以distribution上是对齐的。

Ablation（Table 3）证明这点：
- MV-Posed（用真warped image训）: PSNR 26.21
- MV-UnPoseT（用visual condition训）: PSNR 25.56
- MV-UnPoseM（只mask不加noise mixture）: PSNR 16.14

差距只有0.65 dB！但换来的是训练数据从受限3D dataset扩展到unlimited web video。

---

## 4. 数据怎么curate

25M个raw video不能直接用。很多video是dynamic scene（人在动、树在摇），或者camera几乎不动（tripod拍的）。这些对学3D priors有害。

### 四步pipeline

**Step 1: 降采样**

480p，每2帧取1帧。只用于curation，不用于训练。

**Step 2: 语义级dynamic识别**

用Mask R-CNN（https://arxiv.org/abs/1703.06870）检测human/animal/sports equipment。超过半数frame有这些就丢弃。

**Step 3: 光流级dynamic过滤**

用RAFT（https://arxiv.org/abs/2003.12049）算optical flow，然后算Sampson Distance——每个pixel到epipolar line的距离。距离大的就是dynamic pixel。

但只看mask比例不够，因为dynamic object常在画面中心。所以定义dynamic score（Eq. 8）：

$$\Theta_i = \frac{\sum \mathcal{M}_s}{H \times W}, \quad \Theta_c = \frac{\sum_{\text{center}} \mathcal{M}_s}{(H/2)(W/2)}$$

中心区域是图的中间50%。然后根据$\Theta_i$和$\Theta_c$的组合给分（2/1.5/1/0.5），中心dynamic给更高weight。最终score $S \geq 0.25N$就丢弃。

**Step 4: 视角变化过滤**

用SuperPoint（https://arxiv.org/abs/1712.07129）提100个keypoint，用CoTracker（https://arxiv.org/abs/2307.07635）跨frame追踪。

**直觉**：如果camera静止或只panning，keypoint轨迹是直线或小弧；如果camera有大parallax运动，轨迹是大圆。所以用RANSAC对每个keypoint轨迹拟合circle，circle radius作为viewpoint变化proxy。小radius（$r \leq 20$）数量 > 40 且平均radius < 5就丢弃。

**结果**：25.48M raw video → 2.30M curated video → 15.99M clip，36.27K小时。

Human evaluation：filtered set中88.6%是真正的3D-aware video，raw set只有11.6%。pipeline有效。

---

## 5. Model架构几个关键设计

### 5.1 移除time embedding

这是**极其关键的反作弊设计**。

Stable Video Diffusion（https://arxiv.org/abs/2311.15127）有time embedding，让model知道frame之间的temporal order。如果保留，model会从video的时序结构"作弊"——根据前后帧推断motion，而不是真正学习multi-view geometry。

See3D把time embedding完全去掉，配合frame shuffle（随机打乱顺序），强制model把$X_0$视为**unordered multi-view set**。这与inference时的warp image场景对齐——warp image之间没有temporal order。

### 5.2 3D self-attention

在2D self-attention基础上inflate一个view axis。同一spatial position的不同view互相attend。这是MVDream的标准做法。

### 5.3 初始化

从MVDream weights初始化（本身已在Objaverse等多视角数据上预训练），然后用WebVi3D fine-tune所有参数。

### 5.4 Progressive training

| 阶段 | Resolution | Seq Len | Batch | Iters |
|------|-----------|---------|-------|-------|
| 1 | 512 | 5 (1 ref + 4 target) | 560 | 120K |
| 2 | 512 | 16 (1-3 ref + 13-15 target) | 228 | 200K |
| 3 | 1024 | super-res | 114 | - |

114×A100-40GB，25天。用FlashAttention（https://arxiv.org/abs/2205.14135）+ DeepSpeed ZeRO-2 + bf16。

### 5.5 混入少量3D data

训练时混了0.5M的真正3D annotated data（来自Objaverse/CO3D/RealEstate10K/MVImgNet/DL3DV），与16M WebVi3D混合。

Table 6的ablation：
- 纯video（MV-UnPoseT）: PSNR 25.56
- +10% 3D data: 25.95
- +20% 3D data: 26.19
- +60% 3D data: 26.14
- 100% 3D data（MV-Posed）: 26.21

**一点点3D data就能把performance从25.56推到26.19，接近纯3D的26.21**。这说明video data提供了scale和diversity，3D data提供了precision，两者互补。

---

## 6. Inference时怎么用：warping-based generation

训练完model，怎么用来生成3D？给定sparse input view，要生成dense multi-view供3DGS重建。

### 6.1 整体流程

```
Input views → estimate depth → align depth using keypoint matching 
→ warp to target viewpoint → feed warped image as visual condition to See3D 
→ get novel view → iterate → 3DGS reconstruction
```

### 6.2 Pixel-wise depth scale alignment（Eq. 4）

用MoGe（https://arxiv.org/abs/2411.05078）估的monocular depth是affine-invariant的（scale和shift未知）。需要align到metric scale。

用SuperPoint+LightGlue在anchor views之间match 1024个keypoint，对每个keypoint独立优化scale $\alpha^k$和shift $\beta^k$：

$$\alpha^{k*}, \beta^{k*} = \arg\min_{\alpha, \beta} \|\hat{d}_n^{k*} K_i T_i T_n^{-1} K_n^{-1} m_n^t - m_i^t\|_2^2$$

变量：
- $\hat{d}_n^{k*} = \alpha^k \odot \hat{d}_n^k + \beta^k$: per-keypoint recovered depth
- $K_i, K_n$: target和source的camera intrinsic
- $T_i, T_n$: camera extrinsic
- $m_n^t, m_i^t$: matched keypoint在source和target image的2D坐标

**直觉**：对每个keypoint，调depth的scale和shift，使得把source image的该keypoint warp到target viewpoint后，正好落在target image中对应keypoint位置。这是一个per-keypoint的1D regression。

为什么要per-keypoint？因为monocular depth在不同区域可能有不同的scale error，全局single scale假设不够。

### 6.3 Global metric depth recovery（Eq. 5-6）

1024个keypoint的depth已知，怎么spread到全图？用Locally Weighted Linear Regression（LWLR）。

对每个target pixel $(u,v)$，用Gaussian kernel对附近guided points加权：

$$w_i = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{\text{dist}_i^2}{2b^2}\right)$$

然后做加权线性回归：

$$\hat{\beta}_{u,v} = (X^T W_{u,v} X + \lambda I)^{-1} X^T W_{u,v} \hat{d}_n^*$$

- $b$: Gaussian bandwidth
- $\lambda$: L2 regularization
- $X$: $\hat{D}_n$的homogeneous表示
- $\hat{d}_n^*$: 1024个guided depth

**直觉**：近处的guided point权重大，远处的权重小，局部拟合出scale map和shift map，apply到全图depth。

### 6.4 Iterative generation

每次用最新生成的view作为anchor，warp到下m个target viewpoint，喂给See3D生成。Iterate直到覆盖整条camera trajectory。

### 6.5 3DGS重建

所有generated view喂给3D Gaussian Splatting（https://arxiv.org/abs/2308.14737）。Loss = photometric + SSIM + LPIPS（https://arxiv.org/abs/1801.03924）。

加LPIPS是因为generated multi-view之间有subtle high-frequency不一致，LPIPS在semantic level强制一致。

同时做joint pose-Gaussian optimization（借鉴InstantSplat, https://arxiv.org/abs/2403.20327），让camera pose也learnable，弥补generated view与预设pose的mismatch。

---

## 7. 实验结果有多炸

### 7.1 Single view to 3D（Table 2 top）

在三个benchmark上zero-shot NVS：

| Method | T&T | RE10K | CO3D |
|--------|-----|-------|------|
| LucidDreamer | 13.11 | 15.24 | 13.90 |
| ZeroNVS | 13.38 | 15.37 | 14.23 |
| MotionCtrl | 14.31 | 16.30 | 16.16 |
| ViewCrafter* | 19.13 | 20.49 | 19.07 |
| **See3D** | **23.76** | **25.36** | **24.28** |

在RealEstate10K上**+4.87 dB**。PSNR对viewpoint shift非常敏感，这表明See3D的camera control精度极高——visual condition完全替代pose没掉精度。

### 7.2 Sparse views to 3D（Table 2 bottom, 3 views）

| Method | LLFF | DTU | MipNeRF360 |
|--------|------|-----|-----------|
| MuRF | 21.34 | 21.31 | - |
| ReconFusion | 21.34 | 20.74 | 15.50 |
| CAT3D | 21.58 | 22.02 | 16.62 |
| **See3D** | **23.23** | **28.04** | **17.35** |

**DTU上28.04 vs CAT3D 22.02，+6 dB**。DTU是indoor多物体场景，对geometry精度敏感。这说明See3D学到的3D priors极robust。

### 7.3 Data scaling

10%/20%/40%/80%/100% data → PSNR 19.32/21.04/22.57/24.08/25.01。**清晰的log-linear scaling law**。

加unfiltered data → 19.55（比10% filtered还差）。**Data quality > Data quantity**。

---

## 8. 几个关键的intuition总结

### 8.1 Visual condition是"通用corruption"

Video frame的random mask + noise，和warp image的hole + artifact，在distribution上对齐。Training时让model见过这种generic corruption，inference时各种task-specific corrupted image都能fit。这是**distribution-level unification**。

### 8.2 Time-dependent mixture是leakage解药

不是简单加noise或简单mixture，而是让$W_t$随t变化。早期靠hint，后期靠自己。这个"分工"很elegant。参考signal leakage的分析（https://arxiv.org/abs/2406.15735）。

### 8.3 移除time embedding + frame shuffle是反作弊

强制model把multi-view视为unordered set，避免从video temporal structure cheat。这才能generalize到没有temporal order的warp image。

### 8.4 20% 3D data + 80% video ≈ 100% 3D data

这是对整个3D generation社区的重要信号。Video data提供scale和diversity，少量3D data提供precision。两者互补。

### 8.5 Pixel-wise depth alignment解monocular depth的scale ambiguity

不用全局single scale假设，per-keypoint独立优化，decouple不同区域的depth correlation。这比传统的global alignment精确得多。

---

## 9. 我觉得这篇paper的意义

**对3D generation社区**：这paper证明了可以用web-scale video训练3D generation model，绕过pose annotation瓶颈。CAT3D那种依赖大量pose-annotated data的路线不再是唯一选择。

**对diffusion model**：visual condition的思路可能可以推广。任何task-agnostic corruption只要distribution对齐，就能用同一套训练范式。这可能inspire更多"unified condition"的工作。

**对scaling law**：3D generation终于也能享受scaling law了。25M video只是开始，Internet video还在指数增长。

**潜在extension**：
- 4D generation: 用Internet video学dynamic scene priors，不需要pose annotation
- 与Emu3（https://arxiv.org/abs/2409.18869）结合：用next-token prediction统一visual condition
- 更大backbone: 现在是SVD UNet，换成DiT可能更强

**Limitations**：
- Inference慢（几分钟）
- 只处理static scene
- Model scalability未充分探索

---

## Reference汇总

- See3D project: https://vision.baai.ac.cn/see3d
- MVDream: https://arxiv.org/abs/2308.16512
- ViewCrafter: https://arxiv.org/abs/2409.02048
- CAT3D: https://arxiv.org/abs/2405.10314
- ReconFusion: https://arxiv.org/abs/2403.17712
- Stable Video Diffusion: https://arxiv.org/abs/2311.15127
- Mask R-CNN: https://arxiv.org/abs/1703.06870
- RAFT: https://arxiv.org/abs/2003.12049
- SuperPoint: https://arxiv.org/abs/1712.07129
- LightGlue: https://arxiv.org/abs/2306.13643
- CoTracker: https://arxiv.org/abs/2307.07635
- MoGe: https://arxiv.org/abs/2411.05078
- 3DGS: https://arxiv.org/abs/2308.14737
- LPIPS: https://arxiv.org/abs/1801.03924
- InstantSplat: https://arxiv.org/abs/2403.20327
- Signal leakage analysis: https://arxiv.org/abs/2406.15735
- FlashAttention: https://arxiv.org/abs/2205.14135
- DeepSpeed: https://arxiv.org/abs/1910.02054
- Classifier-free guidance: https://arxiv.org/abs/2207.12598
- Emu3: https://arxiv.org/abs/2409.18869
- DDIM: https://arxiv.org/abs/2010.02502

---

# See3D: "You See it, You Got it" 深度讲解

## 1. Motivation 与核心 insight

这篇 paper 由 BAAI 出品，背后是 Xinlong Wang 团队（也是 Emu3 的作者）。核心 motivation 来源于一个朴素的观察：

3D generation 模型被 **3D 数据规模**严重束缚。当前所有主流 3D dataset 量级都极小：
- Objaverse: 0.8M objects
- DL3DV: 0.01M scenes
- RealEstate10K: 0.08M clips
- MVImgNet: 0.22M multi-view sequences

而与此同时，Internet videos 是一个 **几乎免费且指数增长** 的 multi-view image 数据源——任何手持相机或 drone 拍摄的 video clip，本质上就是一组同一 3D scene 的 multi-view observation。如果能让 model **纯靠 seeing 大量 video** 学到 3D priors，就可以突破 3D 数据成本的天花板。

但是 web-scale videos **没有 camera pose 标注**——对 25M+ videos 跑 COLMAP/SfM 是不可行的（一些视频甚至没有足够的 parallax 让 SfM 收敛）。所以核心问题就变成：**如何在 pose-free 的条件下，依然让 model 学到 precise camera control**？

这正是 "You See it, You Got it" 这个标题的点睛之处——通过 purely visual 的 condition，让 model implicitly 学到 camera control。

Reference: 论文项目页 https://vision.baai.ac.cn/see3d

## 2. WebVi3D 数据 curation pipeline

数据来源：4 个网站
- Pexels (https://www.pexels.com) - open stock footage
- Artgrid (https://artlist.io/stock-footage) - royalty-free
- Airvuz (https://www.airvuz.com) - drone shots
- Skypixel (https://www.skypixel.com) - DJI 用户视频

总计 **25.48M source videos → 44.98 years**。最终 curated 为 **2.30M videos, 15.99M clips, 36.27K hours**。

一个 "3D-aware video" 需要满足两个条件：
1. **Temporal static**: scene 内容不随时间变化（dynamic content 会破坏 cross-view geometry）
2. **Sufficient viewpoint variation**: 相机要有足够的 ego-motion，否则 model 只学到 "邻接 view" 而非完整 3D understanding

### 4 步 pipeline 详解

**Step 1: Temporal-Spatial Downsampling**
- 480p resolution
- temporal downsampling rate = 2 (每 2 frame 取 1 frame)
- 仅用于 curation，不用于训练

**Step 2: Semantic-Based Dynamic Recognition**
用 Mask R-CNN (https://arxiv.org/abs/1703.06870) 检测 human/animal/sports equipment 等 dynamic object classes。如果一个 video 中超过半数的 frames 包含这些 objects，则丢弃。

**Step 3: Flow-Based Dynamic Filtering** — 这是最精细的一步
用 RAFT (https://arxiv.org/abs/2003.12049) 计算 optical flow，然后基于 **Sampson Distance** 检测 dynamic region：每个 pixel 到对应 epipolar line 的距离，如果超过 threshold 就被 mark 为 dynamic motion mask $\mathcal{M}_s$。

但仅靠 mask 比例不够 robust——因为很多 video 的 dynamic object 在 frame 中心，可能占面积不大但视觉重要。所以引入 dynamic score $S_i$：

$$
\Theta_i = \frac{\sum_{u,v} \mathcal{M}_s(u,v)}{H \times W}, \quad
\Theta_c = \frac{\sum_{u,v \in \text{center}} \mathcal{M}_s(u,v)}{(H/2) \times (W/2)}
$$

其中 $\Theta_i$ 是全图 mask 比例，$\Theta_c$ 是中心区域 mask 比例。中心区域定义为从 $0.25H, 0.25W$ 开始的中央 rectangle。

然后 score $S_i$ 按 (Eq. 8) 量化为 $\{2, 1.5, 1, 0.5\}$，对中心 dynamic 区域给更高 weight。整个 sequence 的 score $S = \sum_i S_i$，如果 $S \geq 0.25N$ 就 discard。

**Step 4: Tracking-Based Small Viewpoint Filtering**
用 SuperPoint (https://arxiv.org/abs/1712.07129) 提 100 个 keypoints，用 CoTracker (https://arxiv.org/abs/2307.07635) 跨 frames 追踪 trajectory。对每个 keypoint 的 visible 轨迹，用 **RANSAC-based circle fitting** 拟合一个 circle——

**直觉**：如果相机几乎静止或仅做 panning，keypoints 在 image plane 上的轨迹近似一条直线或一个小圆弧；如果相机做了大 parallax 运动（平移），轨迹会近似一个大圆。所以用 circle radius 作为 viewpoint 变化程度的 proxy。如果小半径 circle ($r \leq 20$) 的数量 > 40 且平均 radius < 5，则视为 small viewpoint，丢弃。

**User study validation**: 在 filtered set 中随机抽 10K clips，88.6% 被人工标注为 3D-aware；未 filtered set 只有 11.6%。这是 **77% 的 absolute improvement**，证明 pipeline 有效。

## 3. See3D model 核心：Visual-Condition

### 3.1 为什么传统 pose-conditional MVD 不能 scale

经典 MVD 方法如 MVDream (https://arxiv.org/abs/2308.16512), Zero-1-to-3 (https://arxiv.org/abs/2303.11308) 都用 camera extrinsics 或 Plücker rays (https://arxiv.org/abs/2406.02509) 作为 3D-inductive condition。这些方法必须配对 pose annotations，无法 scale 到 web data。

ViewCrafter (https://arxiv.org/abs/2409.02048) 用 **warped image**（基于已知 pose 渲染 point cloud 得到的"假视图"）作为 condition——这是一种像素空间 hint。但它依然需要 pose 来计算 warping，所以本质上还是 3D-inductive。

See3D 的核心 insight：**如果只把 warped image 视为一种"distorted, 不完整的 visual hint"**，那它的本质就是 "masked + noise 的 video frame"。既然 model 能从 corrupted video frames 学到 multi-view consistency，那它应该也能从 corrupted warped images 学到——只要 corruption 的 distribution 足够相似。

所以 See3D 的设计目标就是构造一个 **task-agnostic visual corruption**，让 model 既能在 video data 上训练，又能 generalize 到 warped-image 这种 OOD input。

### 3.2 训练目标 (Eq. 1)

$$
\mathcal{L} = \mathbb{E}_{X_0, Y_0, \epsilon, t}\left[\|\epsilon_\theta(X_t, Y_0, V_t, t) - \epsilon\|_2^2\right]
$$

变量含义：
- $X_0 = \{x_0^i\}_{i=1}^N$: 一个 video clip 的 N 个 frames，$N = S + L$
- $Y_0 = \{y_0^i\}_{i=1}^S$: 从 $X_0$ 中随机选的 S 个 reference views，保持 clean
- $G = \{g^i\}_{i=1}^L$: 剩余 L 个 frames 作为 targets
- $X_t$: 标准的 noisy latent at diffusion timestep $t$
- $\epsilon \sim \mathcal{N}(0, I)$: noise sample
- $\epsilon_\theta$: noise predictor (UNet / transformer)
- $V_t$: visual-condition，**不包含 pose**

Loss 只在 target images 上算，reference views 通过 $Y_0$ 直接 inject 到 model（通过 channel concat 或者 attention）。

### 3.3 Time-dependent noise (Eq. 2)

$$
C_t = \sqrt{\bar{\alpha}_{t'}} (1-M) X_0 + \sqrt{1 - \bar{\alpha}_{t'}} \epsilon
$$

变量：
- $M$: binary mask，1 表示被 mask 掉（target views 上的 random irregular masks），0 表示保留
- $(1-M) X_0$: 对 target views 做 mask 后的 clean image，reference views 完全保留
- $t' = f(t) = \beta \cdot t$, 其中 $\beta = 0.2$
- $\bar{\alpha}_{t'} = \prod_{s=1}^{t'} \alpha_s$，DDIM (https://arxiv.org/abs/2010.02502) 的 cumulative variance schedule

**直觉**：standard diffusion $X_t = \sqrt{\bar{\alpha}_t} X_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ 用的是 timestep $t$，但我们这里用 $t' = 0.2t$——也就是说 $C_t$ 的 noise level **显著低于** $X_t$ 的 noise level。这样 $C_t$ 含有更多的 $X_0$ information。

**为什么这么做**？因为 reference + mask image 是用来提供 "visual hint" 的，需要保留足够信号。如果完全 noise 到 $X_t$ 同等水平，hint 就消失了。

但是这里有一个被 [127] (https://arxiv.org/abs/2406.15735) 指出过的 **signal leakage** 问题：如果 $C_t$ 比 $X_t$ 含有更多 clean information，model 可能会"偷看" $C_t$ 直接 output 接近 ground truth 的图，而非真正学习 multi-view generation。

### 3.4 Time-dependent mixture (Eq. 3) — 解决 leakage 的关键

为了解决 leakage，作者引入 **mixture**:

$$
V_t = [W_t \cdot C_t + (1 - W_t) \cdot X_t;\ M]
$$

- $W_t \in [0, 1]$: weighting factor，随 timestep $t$ 变化
- $[;\ ]$: channel-wise concatenation
- $M = \{m^{0:S} \cup m^{S+1:N}\}$: mask tensor，reference 区域全 0 (clean)，target 区域为 random irregular masks

$W_t$ 是 piecewise function (Appendix C.3):

$$
W_t = \begin{cases}
1 - (1 - v_{\text{decay\_end}}) \cdot \frac{t_{\text{peak}} - t}{t_{\text{peak}} - t_{\text{decay\_end}}}, & t \geq t_{\text{decay\_end}} \\
v_{\text{decay\_end}} \cdot e^{-b \cdot (t_{\text{decay\_end}} - t)}, & t < t_{\text{decay\_end}}
\end{cases}
$$

其中 $t_{\text{peak}} = 1000$, $t_{\text{decay\_end}} = 300$, $v_{\text{decay\_end}} = 0.8$, $b = 0.075$。

**直觉分解**：
- **大 timestep (t = 1000, 接近全 noise)**: $W_t \approx 1$，$V_t \approx C_t$。此时 $X_t$ 几乎全是 noise，model 主要靠 $C_t$ 这个含 visual hint 的 corrupted signal 来 initialize generation。注意此时 $C_t$ 也是大量 noise（因为 $\bar{\alpha}_{t'}$ 对应 $t'=200$ 仍有不少 noise），leakage 风险不大。
- **小 timestep (t → 0)**: $W_t \to 0$，$V_t \to X_t$。此时 $X_t$ 越来越接近 $X_0$，而 $C_t$ (用 $t'=0.2t$ 算) 也接近 clean image——如果继续用 $C_t$，leakage 严重。所以让 model 转向依赖 $X_t$（自己的 denoised latent），避免"偷看"。

这个 piecewise 设计（在 $t=300$ 处从 linear decay 切换到 exponential decay）让 $W_t$ 在小 timestep 处快速降到 0，更激进地切断 leakage 通道。

**这就是 "time-dependent" 二字的精髓**：noise level 和 mixture weight 都随 $t$ 变化，让 model 在不同 denoising 阶段依赖不同的信号源。

### 3.5 β 的 trade-off

$\beta$ 控制 $C_t$ 中 noise 的程度：
- $\beta \to 1$: $t' = t$，$C_t$ 跟 $X_t$ 一样 noisy，controllability 弱（hint 模糊）
- $\beta \to 0$: $t' \to 0$，$C_t \approx X_0$，leakage 严重，且 training-time visual cue 与 inference-time warped image 之间 domain gap 巨大
- $\beta = 0.2$: sweet spot，足够 hint 但不至于 leak

### 3.6 Inference 时如何用

在 inference 阶段（warping-based generation），输入是 warped images 而非 video frames。warped image 包含很多 artifacts: self-occlusion holes、stretched pixels、depth errors。但因为 training 时 model 已经见过 "$C_t$ + heavily masked" 的类似 distribution，加上 time-dependent mixture 的 robustness，model 能容忍这种 domain gap。

Ablation (Table 3) 实证：
- **MV-Posed** (用 pose-warped image 训练): 26.21 PSNR
- **MV-UnPoseT** (proposed visual-condition): 25.56 PSNR — 几乎接近！
- **MV-UnPoseM** (random mask, 无 noise/mixture): 16.14 PSNR — 灾难

差距只有 0.65 dB，但换来的是 **训练数据从受限的 3D dataset 扩展到 unlimited web video**。这个 trade-off 极其值得。

## 4. Model Architecture 细节

- **Backbone**: Stable Video Diffusion (https://arxiv.org/abs/2311.15127) 的 2D UNet
- **Initialization**: MVDream (https://arxiv.org/abs/2308.16512) weights (本身已经在 multi-view 数据上预训练过)
- **3D self-attention**: 把原本 2D self-attention 的 spatial axes 扩展到额外的 view axis——同一 spatial position 的不同 views 之间 attend
- **Time embedding 移除** ⚠️ 关键设计：作者**完全去掉 time embedding**，目的是阻止 model 从 video 的 temporal order 中推断 motion——他们要 model **完全依赖 visual-condition**。配合 **frame shuffling**（random 抽 reference frames），让 $X_0$ 视为 unordered set
- **Reference view 注入**: $v_t^{0:S}$ 直接 assign 为 $Y_0$，把 clean reference 信息直接 inject 进 model；同时 CLIP text embeddings 也 cross-attend (来自 reference images 的 per-token features)
- **Zero-Initialize** (https://arxiv.org/abs/2301.03588): 新加的 conv kernels 和 biases 用于处理 visual condition
- **Noise schedule**: 从 scaled-linear 切到 linear，对 multi-view consistency 重要
- **Classifier-free guidance** (https://arxiv.org/abs/2207.12598): drop rate 0.1，随机 drop visual condition

### Training schedule (progressive)

| 阶段 | Resolution | Seq Len | Ref/Target | Batch | Iters |
|------|-----------|---------|-----------|-------|-------|
| 1 | 512×512 | 5 | 1/4 | 560 | 120K |
| 2 | 512×512 | 16 | 1 or 3 / 15 or 13 | 228 | 200K |
| 3 (super-res) | 1024×1024 | - | - | 114 | - |

114× A100-40GB, ~25 days, LR 1e-5, FlashAttention (https://arxiv.org/abs/2205.14135) + DeepSpeed ZeRO-2 (https://arxiv.org/abs/1910.02054), bf16。

**重要小 trick**: 训练时混入了少量 (0.5M) 真正 3D annotated data（来自 Objaverse/CO3D/RealEstate10K/MVImgNet/DL3DV 渲染的多视角），与 16M WebVi3D 混合。Table 6 显示混 20% 3D data 就接近 full-3D 水平。**这是关键 insight: 一点点高质量 3D data + 海量 unposed video = 接近全 3D 标注的效果**。

## 5. 3D Generation Framework: warping + depth alignment

给定 sparse input views，要生成大量 novel views 供 3DGS (https://arxiv.org/abs/2308.14737) 重建。流程：

1. 估计 source view 的 monocular depth (用 MoGe, https://arxiv.org/abs/2411.05078) 得到 affine-invariant depth $\hat{D}_n$
2. 用 SuperPoint+LightGlue 在多 anchor views 之间匹配 1024 keypoints
3. **Pixel-wise depth scale alignment** (Eq. 4): 对每个 keypoint $k$，优化 per-pixel scale $\alpha^k$ 和 shift $\beta^k$，使 warp 后的 keypoint 位置与 target image 中对应 keypoint 一致

$$
\alpha^{k*}, \beta^{k*} = \arg\min_{\alpha, \beta} \|\hat{d}_n^{k*} K_i T_i T_n^{-1} K_n^{-1} m_n^t - m_i^t\|_2^2
$$

变量：
- $\hat{d}_n^{k*} = \alpha^k \odot \hat{d}_n^k + \beta^k$: per-keypoint recovered depth
- $K_i, K_n$: 相机 intrinsics (target / source)
- $T_i, T_n$: 相机 extrinsics
- $m_n^t, m_i^t$: source/target image 中第 $t$ 个 matched keypoint 的 2D 坐标
- $\Pi_{n\to i}(\hat{d}_n) = \hat{d}_n K_i T_i T_n^{-1} K_n^{-1}$: 标准 warping projection

直觉：每个 keypoint 独立解一个 1D affine regression (depth 上)，让 warp 后该 keypoint 落在 target view 对应位置。**Per-keypoint 解耦**避免了全局 single scale 假设的不足（因为 monocular depth 在不同区域可能有不同 scale error）。

4. **Global metric depth recovery** (Eq. 5-6): 用 **LWLR** (Locally Weighted Linear Regression) 把 sparse 1024 个 guided points spread 到 dense depth map。每个 target pixel 用 Gaussian kernel weight 取附近 guided points 做加权回归：

$$
w_i = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{\text{dist}_i^2}{2b^2}\right)
$$

$$
\hat{\beta}_{u,v} = (X^T W_{u,v} X + \lambda I)^{-1} X^T W_{u,v} \hat{d}_n^*
$$

- $b$: Gaussian bandwidth
- $\lambda$: $L_2$ regularization (Ridge)
- $X$: $\hat{D}_n$ 的 homogeneous 表示
- $\hat{d}_n^*$: sparse guided depth (1024 个)
- 输出 $D_n$: 全图 metric depth

5. 用 $D_n$ warp source image 得到 $\hat{I}_j = \Pi_{n\to j}(D_n)$，每个 warp image有 hole mask $M_j$
6. 喂给 See3D: $I_j = \text{See3D}(\hat{I}_j, M_j, \{I_0, I_k\})$ — 用 warp image 作 visual-condition (无需 random mask 因为 warp mask 已经天然存在)
7. Iteratively expand: brown cameras (已生成) → gray cameras (target)
8. 最终所有 generated views 喂给 3DGS 做 reconstruction，并加入 LPIPS loss 来缓解 inter-frame 不一致；同时 joint pose-Gaussian optimization 让 camera pose 也 learnable

## 6. Experiments 分析

### 6.1 Single View to 3D (Table 2 top)

在 Tanks-and-Temples (https://arxiv.org/abs/1703.10593), RealEstate10K, CO3D (https://arxiv.org/abs/2109.11182) 上的 zero-shot NVS:

| Method | T&T PSNR | RE10K PSNR | CO3D PSNR |
|--------|----------|-----------|-----------|
| LucidDreamer (https://arxiv.org/abs/2311.13384) | 13.11 | 15.24 | 13.90 |
| ZeroNVS (https://arxiv.org/abs/2310.17994) | 13.38 | 15.37 | 14.23 |
| MotionCtrl (https://arxiv.org/abs/2310.05595) | 14.31 | 16.30 | 16.16 |
| ViewCrafter (re-implemented) | 19.13 | 20.49 | 19.07 |
| **See3D** | **23.76** | **25.36** | **24.28** |

在 RealEstate10K 上 **+4.87 dB PSNR** over re-implemented ViewCrafter。考虑到 PSNR 对 viewpoint shift 高度敏感（作者特意提到），这表明 See3D 的 camera control 极其精确——visual-condition 完全替代 pose 没有牺牲精度。

### 6.2 Sparse Views to 3D (Table 2 bottom, 3 views)

| Method | LLFF PSNR | DTU PSNR | MipNeRF360 PSNR |
|--------|----------|---------|-----------------|
| MuRF (https://arxiv.org/abs/2405.13148) | 21.34 | 21.31 | - |
| BGGS | 21.44 | 20.71 | - |
| ReconFusion (https://arxiv.org/abs/2403.17712) | 21.34 | 20.74 | 15.50 |
| CAT3D (https://arxiv.org/abs/2405.10314) | 21.58 | 22.02 | 16.62 |
| **See3D** | **23.23** | **28.04** | **17.35** |

**DTU 上 28.04 vs CAT3D 22.02，整整 +6 dB**——这是惊人的提升。DTU 是 indoor 多物体场景，对 geometry 精度敏感。这说明 See3D 学到的 3D priors 极其 robust，尤其 geometry 一致性。

### 6.3 Data scaling ablation

10%, 20%, 40%, 80%, 100% data → PSNR 19.32, 21.04, 22.57, 24.08, 25.01 on RealEstate10K。**清晰的 log-linear scaling law**。

加入 unfiltered 数据训练 → PSNR 19.55（比 10% filtered 还差）。**Data quality > Data quantity**——curated pipeline 是必要的。

## 7. 关键 insights & 与已有工作的关系

### 7.1 与 ViewCrafter 的本质差异
ViewCrafter (https://arxiv.org/abs/2409.02048) 是 closest baseline，用 warped image 作 condition，需要 pose 来 warp。它的训练数据 RealEstate10K 等 pose-annotated。See3D 用 visual-condition 替代 warped image，training data 是 unposed video。Inference 时虽然仍用 warp image 作 input，但 model 已经在 "generalized corrupted video frames" 上训练过，对 warp image 的 domain gap 鲁棒。

### 7.2 Time embedding 移除的深意
这是一个 anti-cheating 设计：Stable Video Diffusion 的 time embedding 让 model 知道 frame 之间的 temporal order——这会诱导 model 用"前后帧 temporal motion"做 hint，而非 visual content 本身。移除 time embedding + shuffling frames 强制 model 把 $X_0$ 视为 unordered multi-view set，与 inference 时的 "无 temporal order 的 multi-view 集合"对齐。

### 7.3 Signal leakage 文献
[127] (https://arxiv.org/abs/2406.15735) 在 image-to-video diffusion 中观察到：conditional image 在大 timestep 上被 over-relied on，导致 model 直接 copy conditional image 的内容。See3D 的 time-dependent noise + mixture 正是 explicitly address 这个问题。

### 7.4 与 CAT3D 的对比
CAT3D (https://arxiv.org/abs/2405.10314) Google 的工作，也是 multi-view diffusion for 3D reconstruction，但**依赖 pose-annotated 3D data**。See3D 的卖点就是用 video data 实现 CAT3D 的能力。在 DTU 上 See3D 大幅超过 CAT3D，说明 video data 的多样性弥补了 pose 标注缺失。

### 7.5 与 ReconFusion 对比
ReconFusion (https://arxiv.org/abs/2403.17712) 也是 diffusion prior for sparse 3D recon，但用 pose-annotated data。See3D 在 3 个 benchmark 上全面超越。

### 7.6 Plücker rays vs Visual-condition
很多 MVD 方法用 Plücker rays (e.g., https://arxiv.org/abs/2406.02509) 表示 camera rays，这是 3D-inductive。Visual-condition 的本质是用 "pixel-space 2D hint" 替代 "geometry-space 3D hint"——一种降维打击。

### 7.7 MoGe 的作用
MoGe (https://arxiv.org/abs/2411.05078) 是 BAAI 自家的 monocular geometry estimator，输出 affine-invariant depth。配合 pixel-wise alignment + LWLR 把 affine-invariant 转 metric。

### 7.8 SuperPoint + LightGlue
SuperPoint (https://arxiv.org/abs/1712.07129) 提 keypoints，LightGlue (https://arxiv.org/abs/2306.13643) 做 matching——这是当前 SOTA 的 local feature pipeline，比 LoFTR 等更准更快。

### 7.9 LPIPS loss 在 3DGS 阶段的作用
3DGS 重建时 standard loss = photometric + SSIM。作者加 LPIPS (https://arxiv.org/abs/1801.03924) 是因为 generated multi-view 之间有 subtle high-frequency 不一致，LPIPS 在 semantic level 强制一致，比 pixel loss 更鲁棒。

### 7.10 Joint pose-Gaussian optimization
借鉴 InstantSplat (https://arxiv.org/abs/2403.20327)，让 camera pose 也 learnable，弥补 generated views 与预设 pose 之间的 mismatch。

## 8. Limitations (作者承认)

1. **Inference慢**: 几分钟一个 sample，不能 real-time
2. **仅静态 scene**: 不 modeling object motion，无法做 4D
3. **Model scalability 未充分探索**: 现在 backbone 还是 SVD UNet，可能 transformer 更强

## 9. 我对这篇 paper 的 take

**直觉上的核心创新**：

1. **Visual-condition 是一个"通用 corruption"**：它不绑定具体 task。Video frames 用 random mask + noise；warped image 用自己的 hole mask + noise。Training 时让 model 见过这种 generic corruption distribution，inference 时各种 task-specific corrupted image 都能 fit。**这是一个 distribution-level unification 的思路**。

2. **Time-dependent mixture 是 leakage 解药**：很多 image-to-X diffusion 工作都遇到 leakage，常见做法是 conditioning dropout 或 noisy input。See3D 的精细之处在于让 weight $W_t$ 随 $t$ 变化，**让 model 在不同 denoising 阶段依赖不同 source**——早期靠 hint，后期靠自己。这个"分工"思路很 elegant。

3. **数据 scaling 是真正的 scaling law**：3D generation 一直被卡在 0.1M~1M data 规模，无法享受 scaling law。See3D 通过 video data 把 3D 生成模型推到 16M+ scale。Table 6 的 ablation 显示 **20% 3D data + 80% video ≈ 100% 3D data**，这对整个 3D generation 社区是个重要信号。

4. **Frame shuffle + no time embedding** 是个聪明设计：它强迫 model 把 multi-view 视为 set，避免从 video 的 temporal 结构 cheat。这是为什么 See3D 能 generalize 到 warped image（warped image 之间没有 temporal order）。

**潜在联想**：
- 这个 "generalized visual corruption" 思路可能可以推广到 4D generation：用 Internet videos 训练 time-conditioned multi-view generation，然后 inference 时把 dynamic scene 视为一种特殊 corrupted multi-view
- 与 Emu3 (https://arxiv.org/abs/2409.18869)（同一作者 Xinlong Wang）的 next-token prediction 思路互补：Emu3 用 token 统一所有 modality，See3D 用 visual corruption 统一 3D-inductive 和 2D-inductive conditions
- SDXL (https://arxiv.org/abs/2307.01952) + multi-view cross-attention 的设计可能是 next step，让 model 在更高 resolution 上 work
- 这个 paradigm 可能可以应用到 dynamic scene reconstruction: 用大量 in-the-wild video 学 4D priors，no pose annotation

**Reference 链接汇总**:
- Paper project page: https://vision.baai.ac.cn/see3d
- Stable Video
