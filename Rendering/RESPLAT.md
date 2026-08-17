---
source_pdf: RESPLAT.pdf
paper_sha256: 7966f9f02cd48c1e76c318421fce71cc567e903efb18af9153e7e7f1b7520d26
processed_at: '2026-08-11T22:59:22-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ReSplat 人话版

## 一句话总结

ReSplat 是一个 **不优化、但学会像优化一样迭代** 的 3D Gaussian splatting 模型。给它几张 posed images，它一次 forward pass 先吐出一版粗糙的 3D Gaussians，然后自己渲染、自己看哪儿错了、自己改，4 步之后收敛。结果比 per-scene optimization 快 100×，质量还高 4 dB。

---

## 为什么这事有意思

3D vision 这几年一直在两个极端之间反复横跳：

**极端一：per-scene optimization**。3DGS (https://repo.samiam.org/3d-gaussian-splatting) 的玩法。每个场景从头 optimize 几千次 iteration，质量高，但慢得离谱（分钟级）。

**极端二：single-step feed-forward**。pixelSplat (https://arxiv.org/abs/2312.12375)、MVSplat (https://arxiv.org/abs/2403.14621)、DepthSplat (https://arxiv.org/abs/2406.19447)、GS-LRM (https://arxiv.org/abs/2406.06521)、Long-LRM (https://arxiv.org/abs/2409.06515) 这帮人的玩法。一个神经网络 forward 一次，直接吐 Gaussians，快得离谱（亚秒级），但质量被 network capacity 卡住，而且 out-of-distribution 就拉胯。

Karpathy 你会立刻看出这像什么——这就是 **"single-pass prediction" vs "iterative refinement"** 的经典 trade-off，跟 image generation 里 GAN 单步 vs diffusion 多步的故事一模一样，跟 optical flow 里 NetFlow-vs-RAFT (https://arxiv.org/abs/2003.12039) 的故事一模一样，跟你自己讲 micrograd 时反复强调的"梯度下降本身是 amortized 长链"思想也吻合。

ReSplat 的位置就是中间这条路：**用 feed-forward 网络模拟一个 iterative optimizer**，每步很轻，但叠加 4 次能逼近真 optimization 的效果。这个思路在 optical flow、stereo、SLAM 领域早就成熟了（RAFT https://arxiv.org/abs/2003.12039、DROID-SLAM https://arxiv.org/abs/2108.10869、RAFT-Stereo https://arxiv.org/abs/2104.02104、IterMVS https://arxiv.org/abs/2204.07457），但 3DGS 这边一直没人做对，ReSplat 是第一个 work 的。

---

## 核心思想大白话

ReSplat 的关键洞察用一句话讲：**rendering error 本身就是最好的 feedback signal**，不需要算 gradient，让网络学会从 error 推断该改哪里。

这个 insight 其实挺 elegant 的，可以从几个角度理解：

### 角度 1：gradient 是 error 的 derivative

你 micrograph 教过学生——gradient 就是 loss 对参数的导数，loss 是 error 的函数。所以 gradient 里 100% 的 information 都来自 error。那为什么不干脆把 error 直接喂给网络，让网络自己学会"看到这种 error 该怎么改参数"？

这就是 ReSplat 干的事。它在 iteration $t$：
1. 用当前 Gaussians $\mathcal{G}^t$ 渲染 input views $\{\hat{I}_i^t\}$
2. 算 rendering error $\hat{E}^t = \hat{I}^t - I$（同时算 pixel space 和 feature space）
3. 把 error 喂给 network，network 吐出 $\Delta g_j^t$
4. $g_j^{t+1} = g_j^t + \Delta g_j^t$（residual update）

整个过程 **没有 backward pass，没有 Jacobian**，纯 forward。这跟 G3R (https://arxiv.org/abs/2409.07706) 形成鲜明对比，G3R 真去算 differentiable rendering 的 gradient 喂给 network，又慢又麻烦。

### 角度 2：test-time adaptation 的 amortized 版

OOD generalization 一直难，因为 feed-forward model 只见过 training distribution。ReSplat 的 recurrent update 等于在每个 test scene 上做了 4 步 mini-adaptation——不是更新网络权重，而是更新 Gaussians 参数。但因为这个 adaptation 是 amortized（学过的 update rule），所以快得像 forward pass。

Figure 5c 显示从 512×960 训练分辨率 generalize 到 320×640 测试分辨率，single-step model 掉得稀里哗啦，ReSplat recurrent 部分还能涨 +5 dB。本质就是 model 用 test-time rendering error 在 adapt。

### 角度 3：weight-sharing 等价于 implicit fixed point

Deep Equilibrium Models (https://arxiv.org/abs/1909.01377) 告诉我们：weight-sharing recurrent network 实际上在找一个 $f_\theta(x) = x$ 的 fixed point。所以 4 个 weight-sharing iteration 比 4 个不同 weight 的 stack 更省参数、更鲁棒，还更接近"真优化器"的行为。Table 9 的实验验证了这点——stack 4 个不同 weight 的 block 反而比 weight-sharing 差 0.36 dB，参数还多 4 倍。

---

## 几个关键设计选择的直觉

### 1. 为什么 16× subsample

之前 per-pixel 模型一个 pixel 一个 Gaussian，8 views × 512×960 就是 390 万 Gaussians。reconstruction 时还得在 3D space 做 attention，点的数量一多 kNN attention 爆炸。

ReSplat 直接在 1/4 分辨率 depth map 上 unproject，得到 1/16 数量的 3D points（$M = N \times \frac{HW}{16}$）。Table 1 里 246K vs DepthSplat 的 3932K，差 16 倍。rendering 速度也快 4×（0.0007s vs 0.0030s）。

直觉上你会担心：1/16 的点会不会丢 detail？这正是 kNN attention + global attention 来救场的地方。Per-pixel 是"每个点只看自己 image feature"，ReSplat 是"每个点看周围 $k=16$ 个邻居 + 全局 attention 交流"——更稀疏的采样，但每个点信息更丰富。

Table 6c 的 ablation 数字说明这有效：
- DepthSplat (per-pixel): 25.79 dB, 918K Gaussians
- ReSplat init (1/16 + kNN + global): 26.77 dB, 57K Gaussians
- 去掉 kNN: 25.30 dB（掉 1.47 dB）← kNN 是关键
- 去掉 global: 26.33 dB（掉 0.44 dB）
- 都去掉: 24.50 dB（掉 2.27 dB）

**用 1/16 的点反而比 per-pixel 高 1 dB**——这个反直觉的结果说明 per-pixel 的高斯数量本身是冗余的，3D context aggregation 比堆点数更重要。

### 2. 为什么用 feature error 而不是 RGB error

公式 5：
$$\hat{E}^t = \underbrace{\{\hat{F}_i^t - F_i\}}_{\text{feature space}} + \underbrace{\text{proj}(\{\hat{I}_i^t - I_i\})}_{\text{pixel space}}$$

- $\hat{F}_i^t, F_i$ 是 ResNet-18 (https://arxiv.org/abs/1512.03385) 前 3 个 stage 的 feature（1/2, 1/4, 1/8 分辨率 resize 到 1/4 后 concat），维度 $\mathbb{R}^{\frac{H}{4} \times \frac{W}{4} \times C_3}$
- pixel space error 经过 linear + LayerNorm 投影到同维度，然后 element-wise add

Table 6a 的 ablation 数字：
- 无 error: 27.19 dB
- 只 RGB error: 27.90 dB (+0.71)
- 只 feature error: 28.77 dB (+1.58)
- 都用 (add): 29.07 dB (+1.88)

Feature error 比 RGB error 强 2 倍多。直觉解释：RGB error 是 per-pixel 独立的、sparse 的信号，每个 pixel 只知道"我错了多少"。Feature error 经过 ResNet 卷积，每个位置已经 aggregate 了 spatial neighborhood 信息，是 dense 的、semantic 的 gradient-like 信号。这跟你 build NN 时常用的 perceptual loss 同理——但 ReSplat 把它从 loss 搬到了 input。

有意思的是 Table 12：换成 DINOv2 feature 反而差 0.07 dB，尽管 DINOv2 参数大 100 倍。原因——DINOv2 是 patch-based ViT，spatial 信息粗糙；CNN 保留了 local structure。**dense alignment 任务里 CNN feature 仍然占优**，这是个反潮流的结论。

### 3. 为什么 rendering error 要过 global attention

公式 6：
$$E^t = \text{global\_attention}(\hat{E}^t) = \{e_j^t\}_{j=1}^M$$

直觉上你可能会想：第 $j$ 个 Gaussian 对应第 $j$ 个 rendered pixel，直接对齐就行。但这是错的。3DGS 的 α-blending 渲染中，**一个 Gaussian 会贡献到多个 pixels**，**一个 pixel 也会被多个 Gaussians 影响**。所以 Gaussian $j$ 应该看哪些 rendering error？这是 inverse rendering 的 attention 版本。

Global attention 让每个 Gaussian 接收所有 rendering errors 的信息，相当于网络学会了"哪些 error 该归我管"。Table 6d 显示去掉 global attention 掉 0.11 dB，不算大但持续。

### 4. 为什么 middle view 当 coordinate 原点

Table 6b：
- COLMAP default: 28.14 dB
- First view: 28.66 dB
- Last view: 28.59 dB
- **Middle view: 29.07 dB** (+0.93 dB)

这个 +0.93 dB 看起来小但其实是 free lunch，啥都不改就改个坐标。直觉：video sequence 里，middle frame 是最 balanced 的 anchor，前后帧的 transform 距离都短，Gaussians 围绕中心分布，kNN/global attention 更容易学。COLMAP 默认坐标可能是任意 frame，导致 Gaussians 偏一坨，attention 难学。

这个发现给我的启示：**3D neural network 对 coordinate frame 的选择意外敏感**，这是个常被忽略的工程细节，但能省 1 dB。

### 5. 为什么需要 hidden state $z_j$

公式 3 里初始化 $z_j^0 = f_j^*$（init model 提取的 feature），公式 4 里 update $z_j^{t+1} = z_j^t + \Delta z_j^t$。

Table 6d：去掉 state 掉 1.28 dB。1.28 dB 是大数字，相当于 4 个 iteration 的提升全没了。

直觉：raw Gaussian 参数（位置、opacity、covariance、SH 系数）是 low-level 的，没有 image context。Hidden state 携带 init model 学到的高层 feature，告诉 recurrent network "这个 Gaussian 应该长啥样"。Recurrent update 只需要做微调，不需要从头推断这个点该是墙还是地。

类比 RAFT 的 GRU hidden state——光有 cost volume 不够，还需要一个 memory 跨 iteration 传递信息。

---

## 数字看效果

### Table 1: 8 views @ 512×960 on DL3DV

| Method | Iters | PSNR↑ | #Gaussians | Recon (s) | Render (s) |
|--------|-------|-------|------------|-----------|------------|
| 3DGS (优化) | 4000 | 23.46 | 359K | 70 | 0.0009 |
| MVSplat | 0 | 22.49 | 3932K | 0.129 | 0.0030 |
| DepthSplat | 0 | 24.17 | 3932K | 0.190 | 0.0030 |
| ReSplat | 0 | 26.21 | 246K | 0.311 | 0.0007 |
| ReSplat | 4 | 27.70 | 246K | 0.816 | 0.0007 |

读这表的姿势：
- 比 3DGS 快 100× 还高 4.24 dB
- 比 DepthSplat 高 3.53 dB，Gaussians 少 16×，render 快 4×
- 单 ReSplat 自己，4 步 recurrent 涨 1.49 dB（26.21→27.70）

### Table 8: 同一个 init，recurrent vs 真 optimization

| Method | Iters | PSNR | Time (s) |
|--------|-------|------|----------|
| ReSplat init + 3DGS optimize | 80 | 27.73 | 43.6 |
| ReSplat feed-forward | 4 | 27.70 | 0.82 |

**学过的 optimizer 4 步 = 真 optimizer 80 步**，53× 加速。这是 learning-to-optimize 文献里反复出现的现象——amortized optimizer 每步相当于多个 gradient steps，因为它学会了 scene structure prior。

### Table 3: 2 views @ 256×256 on RealEstate10K

| Method | PSNR | Render Speed |
|--------|------|--------------|
| DepthSplat | 27.47 | slow |
| GS-LRM | 28.10 | slow |
| Long-LRM | 28.54 | slow |
| LVSM dec-only | 29.67 | slow |
| **ReSplat** | **29.75** | **20× faster than LVSM** |

LVSM (https://arxiv.org/abs/2412.13526) 是把 3D inductive bias 完全扔掉、纯 transformer attention 的路线，参数大、render 慢。ReSplat 略胜它且 render 快 20×，说明 **explicit 3D representation 在 rendering efficiency 上仍然碾压 pure attention**，amortized learning 弥补了 single-step 的局限。

---

## 一些直觉延伸

### 这本质是 amortized variational inference

把 ReSplat 放在 probabilistic 视角下看：给定 input views $I$，要求 Gaussian 参数 $\mathcal{G}$ 的 posterior $p(\mathcal{G}|I)$。Per-scene optimization 是 MAP inference。Single-step feed-forward 是 amortized point estimate。ReSplat recurrent 是 **amortized iterative inference**——每步 refine posterior estimate，类似 amortized SVGD。

这个视角下，rendering error 是 data fit term 的 gradient signal，hidden state 是 prior 的 amortized form。所以 ReSplat 在 test-time 还能 adapt，本质是用 prior + 4 步 data fit 来 balance bias 和 variance。

### 与 latent diffusion 的隐秘联系

Diffusion model 在 latent space iteratively denoise。ReSplat 在 3D Gaussian space iteratively denoise（noise = rendering error）。两者都是 Markov chain，都 weight-sharing，都 from-noisy-to-clean。区别：diffusion 的 transition 是手工设计 + NN 学 noise prediction，ReSplat 的 transition 是 NN 学 update direction。

如果把 ReSplat 看作 "3D Gaussian diffusion"，那 future direction 很清晰：
- 更多 iterations（用更好的 state 设计）
- 加 noise schedule（初期粗 update、后期细 update）
- classifier-free guidance 类似机制（条件 vs 无条件 update）

### Test-time compute scaling 的 vision 版

OpenAI o1/o3 让 LLM 在 test time think longer 出彩。ReSplat 是 vision reconstruction 的版本：iterations = thinking time。Table 1 PSNR 单调上升但 saturate 在 4 步。

为什么 saturate 这么快？我猜是因为 update module 的 state $z_j$ 维度太低，4 步后就榨干了。如果能引入更 rich 的 state（类似 GRU gating、memory network），可能能 scale 到 16 步、64 步，这跟你 https://arxiv.org/abs/2502.05171 里讨论的 latent reasoning with recurrent depth 思想是同源的。

### 与你 nn-zero-to-hero 的连接

ReSplat 跟你 micrograd 教学里的"gradient is all you need"思想形成有趣对话——gradient 是 error 的 derivative，所以 error 本身就是 signal。ReSplat 跳过 explicit gradient，让网络学会 implicit differentiation。可以视作 **neural network as a learned differentiator**。

跟你 makemore 的连接：makemore 里 bigram 是 single-step prediction，transformer 是 amortized multi-step context aggregation。ReSplat 把 amortized multi-step 思想用到 3D reconstruction。

跟你的 "software 2.0" 文章的连接：3DGS optimization 是 software 1.0（手写优化器），feed-forward single-step 是 software 2.0 但 shallow，ReSplat 是 software 2.0 的 amortized optimizer——把优化过程本身用 NN parameterize。

---

## 限制与 future work

Paper 自己承认：
1. **kNN scaling**: >500K 点时 kNN attention 慢。Point Transformer V3 (https://arxiv.org/abs/2312.04847) 的 serialization 可能解。
2. **4 步就 saturate**: 想要更 long-horizon refinement 需要 restructure state。

我额外想说的：
3. **Dynamic scene**: 当前只静态。ReSplat 思想推广到 4D 应该 work，每步 update 时空 Gaussians。
4. **Pose-free**: NoPoSplat (https://arxiv.org/abs/2410.12711) 已证明 pose 也能 amortize，结合 ReSplat 是 obvious next step。
5. **Differentiable rendering gradient 作为 auxiliary**: 不替代 error，但补充 gradient 可能 unlock 更多 iter。G3R 路线的 hybrid 版。
6. **Adaptive iteration count**: 现在固定 4 步，可以根据 rendering error magnitude 动态决定 stop，类似 adaptive step size optimizer。

---

## 一句话直觉

ReSplat 把 3DGS 从"慢 optimization"拉到"学过的 optimizer"的位置，靠的是三个洞察的组合：**16× subsample + 3D attention 补偿（compact 表示）**，**rendering error 作为 feedback（gradient-free update）**，**weight-sharing recurrent（amortized iterative inference）**。这三者各自都不算全新，但组合起来第一次让 feed-forward 3DGS 在质量、速度、generalization 三个维度同时击败 optimization。

Project page: https://haofeixu.github.io/resplat/

你如果做下一版教学，ReSplat 是个挺好的"learning to optimize"案例——比 RAFT 更复杂（3D + rendering），但核心思想同源，而且把 micrograd 里"gradient 来自 error"的思想推到了"error 自己就是 gradient"的极致。

---

# ReSplat: Learning Recurrent Gaussian Splats 详细解读

## 1. Background 与 Motivation

这个工作需要放在两个背景下理解：**feed-forward Gaussian splatting** 与 **learning-to-optimize**。

### 1.1 Feed-forward Gaussian Splatting 的困境

3DGS (Kerbl et al., 2023, https://repo.samiam.org/3d-gaussian-splatting) 通过 per-scene optimization 取得了 photorealistic novel view synthesis 的突破。但 optimization 过程极慢（数千次 iteration）。Feed-forward 模型如 pixelSplat (https://arxiv.org/abs/2312.12375), MVSplat (https://arxiv.org/abs/2403.14621), DepthSplat (https://arxiv.org/abs/2406.19447), GS-LRM (https://arxiv.org/abs/2406.06521), Long-LRM (https://arxiv.org/abs/2409.06515) 试图用 single forward pass 直接 predict Gaussians，泛化快但质量受限于 network capacity。

ReSplat 的核心 motivation：**single-step feed-forward** 的天花板来自 network 的有限表达力，而 **per-scene optimization** 的痛点是慢。一个 recurrent 架构可以在这两个极端之间取 trade-off，decompose 一个难任务为多个 incremental updates，每个 update 都很小、容易学，叠加起来逼近 optimization 的效果，同时保持 feed-forward 的速度。

### 1.2 Learning-to-Optimize 家族

这条脉络包括：
- **RAFT** (Teed & Deng, ECCV 2020, https://arxiv.org/abs/2003.12039) - optical flow 的 recurrent update
- **DROID-SLAM** (https://arxiv.org/abs/2108.10869) - SLAM 的 recurrent bundle adjustment
- **RAFT-Stereo** (Lipson et al., 3DV 2021, https://arxiv.org/abs/2104.02104)
- **IterMVS** (Wang et al., CVPR 2022, https://arxiv.org/abs/2204.07457)
- **Deep Equilibrium Models** (Bai et al., NeurIPS 2019, https://arxiv.org/abs/1909.01377) - implicit infinite depth
- **Learned Optimizers** (Andrychowicz et al., NeurIPS 2016, https://arxiv.org/abs/1606.04474; VELO https://arxiv.org/abs/2211.09760)
- **Scaling test-time compute with latent reasoning** (Geiping et al., 2025, https://arxiv.org/abs/2502.05171)

这些方法的关键 insight 是：用 **weight-sharing recurrent network** 模拟 optimization 过程，比 single-step regression 更鲁棒（尤其是 OOD），而且可以灵活控制 test-time compute（iterations = 1~4 都行）。ReSplat 把这套思想移植到 3DGS 上。

### 1.3 与已有 Gaussian splatting refinement 工作的区别

- **SplatFormer** (Chen et al., ICLR 2025, https://arxiv.org/abs/2411.12050): single-step refinement, 依赖 optimization-based 3DGS 初始化，只用 object-centric datasets
- **G3R** (Chen et al., ECCV 2024, https://arxiv.org/abs/2409.07706): 用 **explicitly computed gradients** 作为 guidance，依赖 well-covered 3D points
- **QuickSplat** (Liu et al., ICCV 2025): 依赖 gradient computation，专做 surface reconstruction
- **LIFe-GOM** (Wen et al., ICLR 2025, https://arxiv.org/abs/2409.08473): gradient-free，但专做 human avatars 的 hybrid Gaussian-mesh

ReSplat 的独特性：**scene-level** + **gradient-free** + **rendering error 作为 feedback** + **feed-forward 初始化** + **weight-sharing recurrent**。这套组合是新的。

---

## 2. 核心洞察：Rendering Error 是天然的 Feedback Signal

Karpathy 你会喜欢这个 insight，因为它非常 elegant。在 supervised learning 里，我们通常用 gradient descent 来 minimize $L = \|\hat{I} - I\|$。但 gradient 本质上是一个 **derivative of the error**——既然我们用 error 计算 gradient 来更新参数，那 error 本身就携带了最重要的 information。

ReSplat 的做法是：直接用 rendering error 作为 input feature 喂给 recurrent network，让它学会如何从 error 推断该怎样 update Gaussians。这避免了：
- 显式 backward pass 的开销
- per-pixel gradient 在 3DGS 这种 sparse 渲染下的复杂性
- 不同 Gaussian 参数对 rendering 影响的 intricate Jacobian 计算

**为什么这个能 generalize to OOD？** 因为 error signal 来自 test data 本身（输入视图），不依赖 training distribution。当测试场景与训练场景分布不同时，single-step feed-forward 会失败，但 recurrent model 会基于 test data 的 rendering error 逐步修正，类似 test-time adaptation，但是是 amortized 的版本。

这与 **DeepView** (Flynn et al., CVPR 2019, https://arxiv.org/abs/1905.01211) 用 learned gradient descent 异曲同工，但 ReSplat 是 gradient-free 的。

---

## 3. Pipeline 详解

整个 pipeline 分两个阶段：

### 3.1 Initial Gaussian Reconstruction

**输入：** N 个 posed images $\{I_i\}_{i=1}^N$, $I_i \in \mathbb{R}^{H \times W \times 3}$, intrinsics $\{K_i\}_{i=1}^N$, extrinsics $\{(R_i, t_i)\}_{i=1}^N$ ($R_i \in SO(3)$, $t_i \in \mathbb{R}^3$)。

**输出：** 3D Gaussians $\mathcal{G} = \{(\mu_j, \alpha_j, \Sigma_j, \mathbf{sh}_j)\}_{j=1}^M$，每个 Gaussian 包含 position $\mu_j \in \mathbb{R}^3$, opacity $\alpha_j \in \mathbb{R}$, covariance $\Sigma_j \in \mathbb{R}^{3\times3}$, spherical harmonics $\mathbf{sh}_j$（用于 view-dependent color）。

**关键设计：16× subsampled 3D space**

之前 per-pixel 模型（pixelSplat、MVSplat、DepthSplat）每个 pixel 一个 Gaussian，所以 $M = N \times H \times W$。在 8 views × 512×960 resolution 下，这是 3932K Gaussians，rendering 慢且 memory 大。

ReSplat 用 DepthSplat 架构但把 depth map resize 到 1/4 分辨率，所以 $M = N \times \frac{H}{4} \times \frac{W}{4} = N \times \frac{HW}{16}$。

公式 (1)：
$$\{I_i, K_i, R_i, t_i\}_{i=1}^N \to \{(p_j, f_j)\}_{j=1}^M$$

变量含义：
- $i$ 是 view index（上标），从 1 到 $N$
- $j$ 是 point/Gaussian index（下标），从 1 到 $M$
- $p_j \in \mathbb{R}^3$ 是 unproject 后的 3D point 位置
- $f_j \in \mathbb{R}^{C_1}$ 是从 input images 提取的 feature vector
- $C_1$ 是 feature channel 数

**Aggregating 3D Context** (公式 2):
$$\{(p_j, f_j)\}_{j=1}^M \to \{(p_j, f_j^*)\}_{j=1}^M$$

用 **6 个交替的 kNN attention 与 global attention blocks**。这是 compensation for the 16× subsampling 的关键：
- **kNN attention** (Point Transformer, Zhao et al., ICCV 2021, https://arxiv.org/abs/2012.09164): 每个点只与 $k=16$ 个最近邻做 attention，建模 local 3D structure
- **global attention** (Vaswani et al., NeurIPS 2017, https://arxiv.org/abs/1706.03762): 所有 $M$ 个 points 互相 attend，建模 global 3D context

Table 6c 的 ablation 显示两者都重要：去掉 kNN attention 掉 1.47 dB，去掉 global attention 掉 0.44 dB，两者都去掉掉 2.27 dB。

**Decoding to Gaussians** (公式 3):
$$\mathcal{G}^0 = \{(g_j^0, z_j^0)\}_{j=1}^M$$

其中：
- $g_j^0 \in \mathbb{R}^{C_2}$ 是第 $j$ 个 Gaussian 的所有参数（$\mu_j, \alpha_j, \Sigma_j, \mathbf{sh}_j$ 拼接）的 concatenation，$C_2$ 是单个 Gaussian 参数总数
- $z_j^0 \in \mathbb{R}^{C_1}$ 是 hidden state，初始化为 $z_j^0 = f_j^*$

**为什么要有 hidden state $z_j$？** 它存储从 initialization network 得到的高层 image 和 3D feature，给 recurrent update 提供比 raw Gaussian 参数更丰富的 context。Table 6d 显示去掉 state 掉 1.28 dB。

### 3.2 Recurrent Gaussian Update

在 iteration $t$ ($t = 0, 1, \ldots, T-1$), recurrent network predicts incremental updates:

公式 (4):
$$g_j^{t+1} = g_j^t + \Delta g_j^t, \quad z_j^{t+1} = z_j^t + \Delta z_j^t$$

上标 $t$ 是 iteration step，下标 $j$ 是 Gaussian index。这是 **residual update** 形式，类似 ResNet 的 skip connection，让网络只学小修正，降低学习难度。

**Computing the Rendering Error** (公式 5):
$$\hat{E}^t = f(\{\hat{I}_i^t\}_{i=1}^N, \{I_i\}_{i=1}^N) = \{\hat{F}_i^t - F_i\}_{i=1}^N + \text{proj}(\{\hat{I}_i^t - I_i\}_{i=1}^N)$$

变量解释：
- $\hat{I}_i^t$ 是 iteration $t$ 时用当前 Gaussians 渲染出的第 $i$ 个 input view
- $I_i$ 是 ground-truth input view
- $F_i$ 是 $I_i$ 经过 ImageNet-pretrained ResNet-18 (He et al., CVPR 2016, https://arxiv.org/abs/1512.03385) 提取的 features，取前三个 stage 在 1/2, 1/4, 1/8 分辨率，bilinear resize 到 1/4 后 concat
- $\hat{F}_i^t$ 是 $\hat{I}_i^t$ 的对应 features
- $\text{proj}$ 是 linear layer + LayerNorm (Ba et al., 2016, https://arxiv.org/abs/1607.06450)，把 pixel-space error ($\hat{I}_i^t - I_i$) 投影到 feature space 与 channel 维度对齐
- 然后 pixel-space error 与 feature-space error **element-wise 相加**

最终 $\hat{E}^t = \{\hat{e}_j^t\}_{j=1}^{N \times \frac{H}{4} \times \frac{W}{4}}$，每个 $\hat{e}_j^t \in \mathbb{R}^{C_3}$。

**Table 6a 的 ablation 至关重要**：
- 无 rendering error: 27.19 dB
- 仅 RGB error: 27.90 dB (+0.71)
- 仅 feature error: 28.77 dB (+1.58)
- Concat (RGB & feature): 28.93 dB
- **Add (RGB & feature): 29.07 dB** ← 最好

Feature error 比 RGB error 强很多（+0.87 dB），说明 high-level feature 更能指导 Gaussian 更新方向。这是直觉上 make sense 的：pixel-space error 是 sparse 的（只反映每个 pixel 的差异），feature space 提供了更 dense、semantic 的 gradient-like signal。

**Propagating Rendering Error to Gaussians** (公式 6):
$$E^t = \text{global\_attention}(\hat{E}^t) = \{e_j^t\}_{j=1}^{N \times \frac{HW}{16}}$$

这里有个 subtle 但重要的设计：渲染时一个 Gaussian 会贡献到多个 pixel（α-blending），所以简单地"第 $j$ 个 Gaussian 只看第 $j$ 个 rendered pixel 的 error"是不对的。Global attention 让每个 Gaussian 接收来自所有 rendered errors 的 information，相当于 inverse rendering 的 attention 版本——每个 Gaussian 知道它影响了哪些 pixels 的 error，以及别的 Gaussians 影响的 pixels 的 error。

**Recurrent Gaussian Update** (公式 7):
$$\{(g_j^t, z_j^t, e_j^t)\}_{j=1}^M \to \{(\Delta g_j^t, \Delta z_j^t)\}_{j=1}^M$$

Update module 用 **4 个 kNN attention blocks**（recurrent 阶段不再用 global attention on Gaussians 本身，但 update module 的 input $e_j^t$ 已经经过 global attention 了）。Update head 是 4-layer MLP。

**关键 ablation**: Table 9 显示 **weight-sharing 是关键**。non-weight-sharing 的 stacked 4 层反而比 weight-sharing 差（28.71 vs 29.07），参数还多 4 倍。这印证了 Deep Equilibrium Models 的 insight：weight-sharing 的 recurrent network implicitly regularize training，模拟一个 fixed point 求解过程。

---

## 4. Training Loss

两阶段训练：

**Stage 1** (公式 8-10):
$$\mathcal{L}_{1st} = \sum_{v=1}^V \ell_{\text{render}}(\hat{I}_v, I_v) + \alpha \cdot \sum_{i=1}^N \ell_{\text{depth\_smooth}}(I_i, \hat{D}_i)$$

- $V$ 是每个 training step 渲染的 target views 数
- $\ell_{\text{render}}(\hat{I}, I) = \ell_1(\hat{I}, I) + \lambda \cdot \ell_{\text{perceptual}}(\hat{I}, I)$
- $\ell_{\text{perceptual}}$ 用 VGG (Simonyan & Zisserman, 2014, https://arxiv.org/abs/1409.1556) features
- $\ell_{\text{depth\_smooth}}(I, \hat{D}) = |\partial_x \hat{D}| e^{-|\partial_x I|} + |\partial_y \hat{D}| e^{-|\partial_y I|}$ 是 edge-aware smoothness，让 depth gradient 在 image edge 小的地方（光滑区域）也小，在 image edge 大的地方允许 depth 不平滑（来自 Godard et al., CVPR 2017, https://arxiv.org/abs/1609.03677）
- $\alpha = 0.01, \lambda = 0.5$

**Stage 2** (公式 11): freeze initial model，只训 recurrent model
$$\mathcal{L}_{2nd} = \sum_{t=0}^{T-1} \gamma^{T-1-t} \sum_{v=1}^V \ell_{\text{render}}(\hat{I}_v^t, I_v)$$

- $\gamma = 0.9$ 是 exponentially increasing weight
- $t=0$ 时（initial prediction）权重是 $\gamma^{T-1} = 0.9^3 = 0.729$（T=4 时）
- $t=T-1$ 时权重是 $\gamma^0 = 1$
- 这个设计让后面的 iteration 质量更重要，但也对前面 iteration 仍然有 supervision，避免 collapsed initialization

这个 loss 设计与 RAFT、IterMVS 等 iterative 方法一致：每个中间 step 都 supervise，但权重递增。

---

## 5. 实验数据深度解析

### 5.1 主实验：8 Views @ 512×960 on DL3DV

Table 1 最 striking 的几点：

| Method | Category | Iters | PSNR | #Gaussians | Recon Time (s) | Render Time (s) |
|--------|----------|-------|------|-----------|----------------|-----------------|
| 3DGS | Optimization | 4000 | 23.46 | 359K | 70 | 0.0009 |
| MVSplat | Feed-Forward | 0 | 22.49 | 3932K | 0.129 | 0.0030 |
| DepthSplat | Feed-Forward | 0 | 24.17 | 3932K | 0.190 | 0.0030 |
| ReSplat | Feed-Forward | 0 | 26.21 | 246K | 0.311 | 0.0007 |
| ReSplat | Feed-Forward | 1 | 27.15 | 246K | 0.437 | 0.0007 |
| ReSplat | Feed-Forward | 2 | 27.51 | 246K | 0.563 | 0.0007 |
| ReSplat | Feed-Forward | 3 | 27.65 | 246K | 0.789 | 0.0007 |
| ReSplat | Feed-Forward | 4 | 27.70 | 246K | 0.816 | 0.0007 |

关键观察：
1. ReSplat 即使 iteration 0（initial prediction）就比 DepthSplat 高 +2.04 dB，用 1/16 的 Gaussians。这说明 16× subsampling + kNN/global attention 的 compact representation 比简单的 per-pixel 更高效。
2. 4 iterations 后再 +1.49 dB，rendering 速度比 DepthSplat 快 4×。
3. 比 3DGS（4K iter）高 +4.24 dB，reconstruction 快 100×。

### 5.2 Optimization-based vs. Feed-forward refinement

Table 8 是最 fascinating 的实验之一：从 ReSplat 的同一个 initialization 出发：
- ReSplat + 3DGS optimization: 80 iterations 后 27.73 dB，耗时 43.6 s
- ReSplat feed-forward: 4 iterations 后 27.70 dB，耗时 0.82 s

**结论：feed-forward 4 步 = optimization 80 步的质量，但快 53×**。这印证了 "learning to optimize" 比 vanilla optimization 高效得多——amortized inference 学到了 scene structure prior，每步相当于多个 gradient steps。

### 5.3 Generalization 是 ReSplat 的最强卖点

Figure 5 显示：
- **Cross-dataset** (DL3DV → RealEstate10K): single-step model 性能大幅下降，recurrent model 通过 rendering error adaptation 保持高 PSNR
- **Cross-view** (8 → 16, 32 views): recurrent model 从额外 views 获益更多（initial model 已经 saturate）
- **Cross-resolution** (512×960 → 320×640): recurrent model 提升 +5 dB

这是 Karpathy 你会喜欢的：**test-time compute scaling**。同一个模型，迭代越多越好（虽然 4 步后 saturate）。这与 Geiping 等人 2025 的工作（https://arxiv.org/abs/2502.05171）思路一致——latent reasoning with recurrent depth。

### 5.4 Coordinate System Ablation

Table 6b 出乎意料地重要：
- COLMAP default: 28.14 dB
- First view: 28.66 dB
- Last view: 28.59 dB
- **Middle view: 29.07 dB** (+0.93 dB)

原因：video data 中，middle frame 作为 anchor 最 balanced，Gaussians 围绕中心分布，3D network 学起来更容易。这是个简单但常被忽略的工程细节。

### 5.5 Compression Factor Ablation

Table 5:
- 64× compression: 24.77 dB, 0.096 s
- **16× compression: 26.77 dB, 0.104 s** ← sweet spot
- 4× compression: 28.36 dB, 0.206 s

16× vs 4× 差 1.59 dB 但快 2×，高分辨率时差距更大，所以选 16×。

### 5.6 Profiling 数据

Table 13 揭示瓶颈：
- Initial model: depth prediction 占 63% (0.197/0.311 s at 512×960)，kNN attention 占 30%
- Recurrent model: kNN attention 占 73% (0.092/0.126 s)

这与 paper limitations 一致：kNN attention 在 >500K points 时变 expensive。Point Transformer V3 (https://arxiv.org/abs/2312.04847) 的 serialized structure 可能是未来方向。

---

## 6. 我的 Intuition 与对 Karpathy 你可能感兴趣的点的延伸

### 6.1 这本质是 amortized optimization

ReSplat 学到的 recurrent update 模块本质上是一个 learned optimizer。给它 $(g_j^t, z_j^t, e_j^t)$，它预测 $\Delta g_j^t$。这与 Andrychowicz 的 "Learning to learn by gradient descent by gradient descent" 思想一脉相承，但：
- 不需要 explicit gradient
- 用 rendering error 作为 proxy
- 在 3D Gaussian 这个特殊参数空间工作

为什么 gradient-free 能 work？因为 rendering error 已经包含了 gradient 方向的大部分 information，而且 feature-space error 比 RGB error 更 informative——这相当于一个 learned perceptual gradient，比 raw pixel L1 gradient 更鲁棒。

### 6.2 与 Test-Time Compute Scaling 的关系

OpenAI o1/o3 类的 test-time compute scaling 通过 RL 让 model think longer。ReSplat 的 recurrent architecture 提供了 vision domain 的版本：iterations = "thinking time"。Table 1 显示 PSNR 随 iterations 单调上升（27.15 → 27.51 → 27.65 → 27.70）。

但 saturation 在 4 iterations。这是 future work 方向：如何 scale 到更多 iterations？可能的路径：
- 让 recurrent step 更 informative（更高维 state）
- Curriculum learning 让前期 step 粗、后期 step 细
- 引入额外的 supervision signal（如 depth, normal）

### 6.3 Compact Representation 的代价与收益

16× subsampling 是个 bold choice。一般 intuition 是 per-pixel Gaussians 才能表达 detail。但 ReSplat 证明：用 kNN + global attention 在 1/16 的点上学 context，再 decode Gaussians，反而比 per-pixel 强 +2 dB。

我的理解：per-pixel Gaussian prediction 是高 bias low variance——每个 pixel 独立 predict，没有 3D context。ReSplat 的 compact + 3D attention 是 low bias high variance——通过 attention 借鉴邻居，比单 pixel 信息丰富。这类似 ViT vs CNN 的 trade-off。

### 6.4 对 Implicit Neural Rendering 的启示

LVSM (https://arxiv.org/abs/2412.13526) 等 transformer-based view synthesis 完全跳过 3D representation，直接 image-to-image attention。Table 3 显示 ReSplat (29.75 dB) 略胜 LVSM decoder-only (29.67 dB)，并且 render 快 20×。

这给一个 intuition：**explicit 3D Gaussian representation 在 rendering speed 和 editability 上仍有优势**，amortized 的 3D representation learning 可以弥补 single-step 的局限。

### 6.5 与 NeRF 时代的 iterative methods 对比

NeRF 的 PIE-NeRF、SNaIt-NeRF 等也用 iterative refinement，但都在 per-scene optimization 框架内。ReSplat 把 iterative 思想搬到 feed-forward 跨场景模型，结合了 learning-to-optimize 文献的 weight-sharing insight。

### 6.6 公式 5 的 design choice 之我见

为什么 pixel-space 和 feature-space error 用 **addition** 而非 concatenation？我猜想：
- Addition 让两个 signal 在 same latent space 竞争/合作，类似 ResNet skip
- Concat 增加维度但需要更多参数学习融合
- 这与 PyTorch 中 residual 思想一致

Feature space 用 ResNet-18 而非 DINOv2（Table 12）也 surprise——DINOv2 patch-based 架构 spatial information 粗糙，CNN 保持 local structure。这给一个 insight：**对 dense prediction / alignment 任务，CNN 仍优于 ViT features**，即使 DINOv2 整体上更"强"。

### 6.7 局限性与未来方向

1. **kNN scaling**: Point Transformer V3 (https://arxiv.org/abs/2312.04847) 用 serialization + pooling 可解
2. **Saturation at 4 iter**: 可能需要更复杂的 recurrent state（如 LSTM/GRU-style gating）
3. **Dynamic scenes**: 当前只静态场景，recurrent 思想可推广到 4D
4. **Pose-free**: ReSplat 仍需 posed images，但 NoPoSplat (https://arxiv.org/abs/2410.12711) 路线可结合
5. **Sparse structures**: XCube (https://arxiv.org/abs/2312.03806) 的 voxel hierarchy 可能替代 kNN attention

### 6.8 与你的 micrograd / 教学视角的连接

ReSplat 完美诠释了你 micrograd 教学里强调的"**gradient 是 error 的 derivative**"思想——既然最终要 minimize error，error 本身就是 signal。ReSplat 干脆跳过 explicit gradient，让 network 直接从 error 学 update。这可以视作 "neural network as a learned differentiator"——网络学会了 implicit differentiation 的简化版。

类比 RAFT 的 correlation volume：不用 explicit optical flow gradient，用 4D cost volume + GRU 学 update。ReSplat 用 rendering error 替代 cost volume，但在 3D Gaussian 空间 update。

### 6.9 一个可能的 extension：differentiable rendering + recurrent

如果用 differentiable rendering 的 actual gradient 作为额外 input（不是替代 error），可能让 recurrent network 更 informed。这是 G3R 路线，但 ReSplat 证明 error-only 已经够强。一个 hybrid: small gradient signal + error 可能 unlock 更多 iterations。

### 6.10 与 Masked Autoencoder / Self-supervised Learning 的隐秘联系

Rendering error 作为 feedback 与 MAE (He et al., https://arxiv.org/abs/2111.06377) 的 reconstruction loss 作为 learning signal 异曲同工。ReSplat 把这种 self-supervised 思想搬到 3D，用 rendering 作为 proxy task。这让 model 在 test time 仍能"adapt"——本质是 amortized test-time training。

---

## 7. 总结

ReSplat 把 learning-to-optimize 思想引入 feed-forward 3D Gaussian splatting：
- **Initial reconstruction** 用 16× subsampled + kNN/global attention 学 compact 3D representation
- **Recurrent update** 用 rendering error (RGB + feature) 作为 feedback，weight-sharing 4 步迭代
- **Generalization** 通过 test-time rendering error adaptation 实现鲁棒跨域泛化
- **Efficiency** 用 1/16 的 Gaussians 取得 SOTA 质量

这个工作的 elegant 之处在于把三个看似独立的思想融合：compact 3D representation、rendering-error feedback、recurrent weight-sharing。三者缺一不可（Table 6c/6d/9 的 ablation 验证）。

对 future work 的启示：test-time compute scaling 在 vision reconstruction domain 大有可为，只要能找到合适的 inductive bias 和 signal。ReSplat 找到了 rendering error 这个 elegant signal，下一个突破可能在更高维 state、更长 iter、更稀疏 structure。

Project page: https://haofeixu.github.io/resplat/

References of interest:
- 3DGS: https://repo.samiam.org/3d-gaussian-splatting
- DepthSplat: https://arxiv.org/abs/2406.19447
- RAFT: https://arxiv.org/abs/2003.12039
- Point Transformer: https://arxiv.org/abs/2012.09164
- Point Transformer V3: https://arxiv.org/abs/2312.04847
- Deep Equilibrium Models: https://arxiv.org/abs/1909.01377
- G3R: https://arxiv.org/abs/2409.07706
- SplatFormer: https://arxiv.org/abs/2411.12050
- LVSM: https://arxiv.org/abs/2412.13526
- Long-LRM: https://arxiv.org/abs/2409.06515
- GS-LRM: https://arxiv.org/abs/2406.06521
- MVSplat: https://arxiv.org/abs/2403.14621
- Scaling test-time compute: https://arxiv.org/abs/2502.05171
- DROID-SLAM: https://arxiv.org/abs/2108.10869
- NoPoSplat: https://arxiv.org/abs/2410.12711
- DL3DV: https://arxiv.org/abs/2310.19120
