---
source_pdf: SKYFALL-GS SYNTHESIZING IMMERSIVE 3D URBAN SCENES FROM SATELLITE IMAGERY.pdf
paper_sha256: d6b7807158244bfebdf790e173117a174de3c5bb043a342ee826f217c95b9333
processed_at: '2026-08-12T07:38:26-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Skyfall-GS

## 一句话版本

拿几张卫星俯瞰照片，凭空造出一个能飞进去逛的 3D 城市，连卫星根本拍不到的楼立面都给你想象出来。

---

## 1. 这事为什么难

你想从卫星照片重建 3D 城市，会遇到几个很根本的坑：

**坑一：卫星只能拍屋顶**

卫星在几百公里高的天上往下拍，它看到的就是一堆屋顶、马路、停车场。楼的外立面（窗户、门、招牌）它根本看不见。你拿这些照片去做 3D 重建，得到的模型从正上方看还行，一旦你把视角降到地面附近，楼就像被压扁的纸盒——朝向你的那一面是空的、模糊的、全是 artifact。

**坑二：卫星视角之间几乎没 parallax**

普通 3D 重建（NeRF、3DGS）能 work，靠的是多个视角之间的视差。你从不同角度拍同一栋楼，通过三角化就能算出 depth。但卫星拍照的角度变化极小（都从天上斜着看，elevation 通常 70-80°），basleline 也小。结果就是 depth 估计极其不准，3DGS 训练出来一堆"飘浮"的 Gaussian（floaters），像雾一样浮在楼旁边。

**坑三：多日期照片光照、季节全不一样**

DFC2019 数据集里同一块地的 9-21 张照片可能跨好几个月拍的。有的晴天有的阴天，有的夏天树叶绿的有的是冬天光秃秃。你直接拿这些照片训 3DGS，它根本无法收敛——同一个像素位置，今天该是绿色明天该是灰色，模型无所适从。

**坑四：现有 city generation 方法太死板**

CityDreamer、GaussianCity 这些方法需要你提供 pixel-aligned 的 semantic map（这是道路、这是楼、这是草地）和 height field（楼多高），然后它们用这些在 GoogleEarth 纽约数据上训练的生成模型去渲染。问题：

- 你得先有 semantic map 和 height field，这本身就不容易
- 它们只见过纽约，换到东京、迪拜就乱套
- 高度场本质上是把楼当成"拉伸的盒子"，楼再复杂也就一个高度值，桥、隧道、多层层级全完蛋

---

## 2. Skyfall-GS 的核心 idea

Skyfall-GS 干的事说起来很直觉：**既然 satellite 重建出来的楼立面那么糟糕，那就让 diffusion model 来"补画"它**。

但补画有几个细节问题：用什么补？怎么补？怎么保证补完之后 3D 还自洽？

### Key insight：noisy render = diffusion 中间态

这是 paper 我觉得最 elegant 的观察。

你用 satellite 训完 3DGS，从地面视角渲染一帧。这帧什么样？模糊、扭曲、噪点、楼像融化的蜡。这看着像什么？**像 diffusion model 去噪过程的中间步骤**。

Diffusion model 的工作原理就是从 noise 一步步去噪到清晰图。现在你给它一张"半噪点"的图（其实是 3DGS 的烂 render），它把剩下的去噪步骤走完，就得到一张"看起来像样"的图。

这张图虽然不是真的，但它是 FLUX.1（一个很强大的 T2I diffusion model）"理解"的城市该长什么样。它有合理的窗户排列、正确的光照、自然的纹理。你拿这张图当 pseudo ground-truth，回去继续训 3DGS。3DGS 就慢慢"学会"了楼立面该长什么样。

这就是 [SDEdit](https://sde-image-editing.github.io/) 的核心思想，只是 paper 在 3DGS + satellite 这个具体场景下把它用得很好。

### 为什么不用 SDS / DreamFusion 那套

DreamFusion 那种 Score Distillation Sampling 路线是：把 3D 渲染图喂给 diffusion，让 diffusion 给个 score gradient，用这个 gradient 反向传播优化 3D 参数。问题：over-smooth、训练慢、容易 saturation。

Skyfall-GS 走 [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/) / [IM-3D](https://im3d-url.github.io/) 风格的 **iterative dataset update**：直接拿 diffusion refined 出来的图当"真值"训练，比 SDS 直接、稳。

---

## 3. 两阶段 Pipeline

### Stage 1：先从 satellite 建个能用的初稿

这个阶段目标：用 satellite images 训出一个 3DGS，虽然 ground view 很烂，但至少"屋顶和马路"那部分几何是准的。

**Camera 处理**

卫星用的不是普通针孔相机，是 RPC（Rational Polynomial Camera）模型。3DGS 只认 perspective camera。所以先用 [SatelliteSfM](https://openaccess.thecvf.com/content_ICCVW_2019/papers/3DIMP/Leveraging_Vision_Reconstruction_Pipelines_for_Satellite_Imagery_ICCVW_2019_paper.pdf) 把 RPC 翻译成普通 perspective 参数 + sparse SfM 点云当 3DGS 初始化。

**Appearance Modeling：解决多日期光照**

借鉴 [WildGaussians](https://wild-gaussians.github.io/)，给每张训练图一个 32 维 embedding $e_j$（编码"这是哪天拍的、什么光照"），给每个 Gaussian 一个 24 维 embedding $g_i$（编码"这个点局部 appearance 怎么变"）。把这两个 embedding + Gaussian 的 0 阶 SH 颜色 $\bar{c}_i$ 喂进一个 2 层 MLP，输出仿射变换参数 $(\beta, \gamma)$：

$$\tilde{c}_i = \gamma \cdot \hat{c}_i + \beta$$

这相当于每个 Gaussian 在每张图上有自己的 brightness/contrast 调节。模型就能区分"这块地本来是绿的，今天是雪白的"vs"这块地一直是绿的，只是阴影让它显得暗"。

注意 paper 限制 SH 到 0+1 阶，避免 view-dependent effect 把 appearance variation 给吃了。

**Opacity Regularization：消灭 floaters**

公式：

$$\mathcal{L}_{\text{op}} = -\sum_i [\alpha_i \log \alpha_i + (1-\alpha_i)\log(1-\alpha_i)]$$

这其实就是 binary entropy 的负数。最小化它 = 让每个 Gaussian 的 opacity 要么接近 0 要么接近 1。那些"半透明"的 floater（satellite parallax 不够时最容易出现）被推到极端值，densification 时就自动 prune 掉了。

**Pseudo-camera Depth Supervision：补 parallax**

这步我觉得最聪明。Satellite parallax 不够，怎么办？自己造"假相机"。

在地面附近采样一批虚拟相机（24 个，每 10 iteration 采样一次），用 3DGS 渲染它们的 RGB 和 depth。然后用 [MoGe](https://arxiv.org/abs/2410.19115)（一个很强的 monocular depth estimator）从 RGB 估计 depth，用这个 depth 来监督 3DGS 自己渲的 depth。

但 MoGe 估计的 depth 是 scale-invariant 的（没有绝对尺寸信息），所以不能用 L2 loss，用 Pearson correlation：

$$\mathcal{L}_{\text{depth}} = \|\text{PCorr}(\hat{D}_{\text{GS}}, \hat{D}_{\text{est}})\|_1$$

Pearson correlation 只关心"哪里高哪里低"的相对结构，不强迫 3DGS 的绝对 scale 去匹配 MoGe。这样 3DGS 保留自己的 metric scale，但学到了 MoGe 提供的"形状先验"。

效果：屋顶变平、马路变平、斜坡变对。

**Stage 1 总 loss**：

$$\mathcal{L}_{\text{sat}} = \mathcal{L}_{\text{color}} + 10 \cdot \mathcal{L}_{\text{op}} + 0.5 \cdot \mathcal{L}_{\text{depth}}$$

跑 30,000 iteration，1 小时。出来的 3DGS 屋顶 OK，但楼立面还是烂。

### Stage 2：用 diffusion 补立面

这步是 paper 的核心贡献。

**Curriculum Learning：从天上掉到地上**

关键 observation（Figure 4）：satellite-trained 3DGS 在高 elevation 视角（接近 top-down）渲染 OK，elevation 越低越烂。因为高 elevation 接近训练 distribution，低 elevation 是 OOD。

如果一开始就用 diffusion refine 低 elevation render，diffusion 看到的是纯噪点，会乱 hallucinate，几何完全坏掉。

所以走 curriculum：5 个 episode，elevation 从 85° 慢慢降到 45°。每个 episode 先用当前 3DGS 渲染 curriculum views，用 diffusion refine，然后拿 refined views 训 3DGS 10,000 iteration。

第 1 个 episode 在高 elevation，render 质量还行，diffusion refine 出来的图也靠谱，3DGS 学到东西。第 2 个 episode elevation 稍微降低，但因为 3DGS 已经被上一个 episode 改善了，render 比之前好。如此累积。到最后一个 episode 到 45° 时，3DGS 已经学会了不少立面信息。

这就是 "Skyfall" 的名字来源——视角从天空逐渐"掉"到地面。

**Render Refinement：FlowEdit + FLUX.1**

用 [FlowEdit](https://arxiv.org/abs/2412.08629) 配合 [FLUX.1 dev](https://blackforestlabs.ai/)。FlowEdit 接受一对 prompt（source 描述原图、target 描述目标图），在两者之间做 flow-based translation，保留结构、改 attribute。

Prompt 设计：
- Source: "Satellite image of an urban area with modern and older buildings, roads, green spaces. Some areas appear distorted, with blurring and warping artifacts."
- Target: "Clear satellite image of an urban area with sharp buildings, smooth edges, natural lighting, and well-defined textures."

source 明说"有 distortion、blur、warping"，target 描述"sharp、smooth、natural lighting、well-defined textures"。FlowEdit 把 source 描述的 artifacts 替换成 target 描述的属性，楼的几何结构（buildings、roads）保留。

Noise level 在 $[n_{\min}=4, n_{\max}=10]$ 之间采样。低 noise 保留原始结构多但去 artifact 少，高 noise 改得多但可能 alter geometry。

**Multi-sample per view：解决 3D 不一致**

这是 subtle 但关键的一步。

每个 view 独立做 2D diffusion，不同 view 之间会不一致。3DGS 在不一致 supervision 上训练会 overfit 到 single view（[CoR-GS](https://cor-gs.github.io/) 说过的现象），novel view 又出 artifact。

Paper 的 trick：每个 view 做 $N_s = 2$ 次独立采样，得到 2 张略微不同的 refined 图。3DGS 在这 2 张上 minimize color loss，相当于隐式平均，自动找"两张都同意"的 3D 表示。

Intuition：单个 sample 是"理想 refined view"的 noisy estimator，2 个 sample 平均降 variance，3DGS 优化在它们中间找到 consensus。这跟 [ProlificDreamer/VSD](https://ml.cs.tsinghua.edu.cn/~prolificdreamer/) 的 motivation 殊途同归。

**IDU 训练细节**

- 每个 episode 10,000 iteration
- 训练采样：75% 用 refined views，25% 用原始 satellite views。**这 25% 是 anchor**——确保 3DGS 不忘 satellite ground truth 的 layout 和 semantic
- IDU 阶段关掉 opacity regularization，让 Gaussian 保留 variable opacity，对半透明物体（树叶、玻璃）友好
- 固定一个 appearance embedding，统一 refined views 的 appearance

总时间 ~6 小时。

---

## 4. 实验结果

### 主表

**DFC2019**（satellite reconstruction baselines）：

| Method | FID_CLIP ↓ | CMMD ↓ |
|---|---|---|
| Sat-NeRF | 88.36 | 4.868 |
| EOGS | 87.74 | 5.286 |
| Mip-Splatting | 87.19 | 5.405 |
| CoR-GS | 89.03 | 5.241 |
| **Skyfall-GS** | **27.35** | **2.086** |

FID_CLIP 从 ~88 直接掉到 27。这是数量级的差别。

**GoogleEarth**（city generation baselines）：

| Method | FID_CLIP ↓ | CMMD ↓ |
|---|---|---|
| CityDreamer | 36.52 | 4.152 |
| GaussianCity | 28.73 | 2.917 |
| CoR-GS | 27.32 | 3.752 |
| **Skyfall-GS** | **9.91** | **2.009** |

又是数倍提升。

**User Study**：89 个参与者，winrate 接近 90-97%。人眼直接对比也碾压。

### 渲染速度

11 FPS on T4，40 FPS on MacBook Air M2。CityDreamer 0.18 FPS on A100，GaussianCity 10.72 FPS on A100。Skyfall-GS 因为输出就是 vanilla 3DGS，real-time 渲染天然支持。

### Ablation

最关键的两个 ablation：

**Reconstruction stage**（Table 3）：
- 没 appearance modeling → 直接 Failed，multi-date 训不收敛
- 加 opacity reg → FID 从 41.9 → 39.95
- 加 depth sup → 再降到 38.01

**Synthesis stage**（Table 4）：
- 没 multi-sample 没 curriculum → FID 34.11
- 加 multi-sample 没 curriculum → 33.79（CMMD 反而升）
- 加 multi-sample + curriculum → **28.35**

curriculum 是关键。单独 multi-sample 帮助不大，但和 curriculum 配合效果大幅跃升。

---

## 5. 这工作牛在哪、弱在哪

### 牛的地方

1. **Insight 干净**：noisy 3DGS render 当 diffusion 中间态，这个 metaphor 优美
2. **Curriculum motivation 强**：从 Figure 4 的 observation 直接导出设计，不是凭空魔改
3. **Zero-shot**：FLUX.1 没见过 satellite，没见过 city，但能 work
4. **Output 是 vanilla 3DGS**：不需要特殊 viewer，可直接用任何 GS 工具链
5. **Ablation 完整**：每个设计都验证，包括 multi-sample 这种 subtle 点

### 弱的地方

1. **慢**：7 小时/AOI，scale 到 whole city 会爆炸
2. **Street-level 还是糊**：curriculum 最低 45°，再低没直接 supervise
3. **3D 一致性不完美**：multi-sample 缓解但没根治，$N_s=2$ 偏小
4. **Prompt 需手调**：source/target prompt 是手动设计的，换场景可能要重新调
5. **Identity preservation 弱**：对特定 building 的招牌、特殊窗户排列，diffusion 会 invent 不存在的内容

### 我能想到的延伸

1. **Video diffusion 替 per-view refinement**：[CAT4D](https://arxiv.org/abs/2411.18613) 风格，一次性 refine 多个 view，自然多视角一致
2. **Geometry-aware diffusion**：把 3DGS depth/normal 当 [ControlNet](https://arxiv.org/abs/2302.05543) condition，比纯 image-to-image 更准
3. **VLM 自动 prompt**：[LLaVA](https://llava-vl.github.io/) 看一眼 render 自动描述问题并 generate target prompt
4. **Hierarchical scale-up**：结合 [Hierarchical 3DGS](https://arxiv.org/abs/2406.19390) 或 [VastGaussian](https://arxiv.org/abs/2402.17427)，把 IDU 推到 whole city
5. **Robotics simulator**：Skyfall-GS 输出当 [Habitat](https://aihabitat.org/) 训练环境，从 satellite 直接造 robot playground

---

## 6. 给 Karpathy 的 take-away

如果你只记三件事：

1. **"Satellite-trained 3DGS 渲染的烂图 = diffusion 的中间去噪态"**——这是整个 paper 的核心 metaphor，把"satellite 缺立面信息"这个 ill-posed problem 转成"用 diffusion prior 补 plausible 立面"
2. **Curriculum learning from sky to ground**：satellite 3DGS 在高 elevation 渲染好、低 elevation 渲染烂，所以从高 elevation 开始 refine，逐步降低，让 3DGS 站在前一个 episode 的肩膀上学更难的视角
3. **Multi-sample 缓解 2D diffusion 的 3D 不一致**：每个 view 多采几个 sample，3DGS 在它们上面 minimize loss 隐式 ensemble，找 consensus 表示

这是把 [ReconFusion](https://arxiv.org/abs/2312.02981) / [CAT3D](https://arxiv.org/abs/2405.19415) 那套"diffusion prior 蒸馏进 3D 重建"范式在 satellite domain 的具体实现，而且用了 curriculum + multi-sample 这两个 trick 让它实际 work。

项目页面：https://skyfall-gs.jayinnn.dev/

---

# Skyfall-GS: 从 satellite imagery 合成可导航 3D 城市的深度解析

## 1. 核心 Problem 与 Key Insight

这篇 paper 解决的核心 problem：**仅用 multi-view satellite imagery 合成可自由飞行的、几何精确的、photorealistic 的 3D 城市场景**。

Satellite imagery 的优势在于 geographic coverage 广、automated 采集、high-resolution（WorldView-3 可达 31 cm/pixel）。但 satellite 视角有两大 fundamental limitations：
1. **Building facades 完全不可见**（satellite 从正上方拍摄，只能看到屋顶），导致 3D 重建的 ground-view 几何严重缺失
2. **Parallax 极有限**（不同 satellite view 之间 baseline 小，且基本是 top-down 视角），3DGS 训练后 floaters 严重，几何模糊

Karpathy 你应该会想到，这里的关键 trick 在于：**把 satellite-trained 3DGS 在低 elevation 视角下渲染的 noisy image，当作 diffusion model 的中间 denoising step**，再用 T2I diffusion prior 完成 denoising，得到 hallucinated 但 3D-consistent 的 ground-view appearance，作为 pseudo ground-truth 回灌到 3DGS 训练。这就是 paper 命名 "Skyfall" 的来源——camera 视角从天空逐渐 "fall" 到地面。

Project page: https://skyfall-gs.jayinnn.dev/

---

## 2. 两阶段 Pipeline 总览

```
Satellite Images (multi-view, multi-date)
       │
       ▼
[Stage 1: Reconstruction]
   3DGS + Appearance Modeling + Opacity Reg + Pseudo-camera Depth Supervision
       │
       ▼
Initial 3DGS G (有 floaters、facade 缺失、blurry)
       │
       ▼
[Stage 2: Synthesis]  Curriculum-based IDU
   for episode i = 1 to N_e:
     render curriculum views (elevation: high → low)
     FlowEdit refine (FLUX.1 [dev])
     N_s multi-samples per view
     train 3DGS on refined + original satellite mixture
       │
       ▼
Final Refined 3DGS G' (real-time navigable)
```

---

## 3. Stage 1: Satellite-to-3DGS Reconstruction 的技术细节

### 3.1 3DGS Preliminary 回顾

3DGS 把场景表示为 K 个 anisotropic Gaussians，每个 Gaussian 有：
- $\mu_i \in \mathbb{R}^3$：中心位置
- $\Sigma_i \in \mathbb{R}^{3\times3}$：covariance（用 scaling $S$ 和 rotation $R$ 参数化 $\Sigma = RSS^TR^T$）
- $\alpha_i \in [0,1]$：opacity
- $c_i$：view-dependent color（spherical harmonics）

投影到 image plane 时，使用 EWA splatting：

$$\Sigma_i'^{\bot} = J W \Sigma_i W^T J^T$$

变量含义：
- $W \in \mathbb{R}^{3\times3}$：viewing transformation（世界 → camera）的 rotation 部分
- $J$：projective Jacobian（相机 → 像素 plane 的一阶 Taylor 展开）
- 上标 $\bot$ 表示 image-space 2D covariance（去掉 depth 维度）

Pixel color 用 alpha compositing（front-to-back）：
$$C(\mathbf{u}) = \sum_{i \in \mathcal{N}} c_i \alpha_i' \prod_{j=1}^{i-1}(1-\alpha_j')$$

Color loss（公式 1）：
$$\mathcal{L}_{\text{color}} = \lambda_{\text{D-SSIM}} \text{DSSIM}(\hat{C}, C) + (1-\lambda_{\text{D-SSIM}}) \|\hat{C} - C\|_1$$

- $\lambda_{\text{D-SSIM}} = 0.2$（paper 设定）
- DSSIM = (1 - SSIM)/2

参考：3DGS 原文 https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

### 3.2 Camera Parameter Approximation

Satellite imagery 用 **Rational Polynomial Camera (RPC)** model，不是普通的 pinhole camera。RPC 是从 image coordinates 到 geographic coordinates 的 rational polynomial mapping，难以直接用 3DGS 的 perspective projection。

Skyfall-GS 用 [SatelliteSfM (Zhang et al., 2019)](https://openaccess.thecvf.com/content_ICCVW_2019/papers/3DIMP/Leveraging_Vision_Reconstruction_Pipelines_for_Satellite_Imagery_ICCVW_2019_paper.pdf) 把 RPC 近似为 perspective camera 的 extrinsic + intrinsic，并生成 sparse SfM points 作为 3DGS 的 initialization points。

**Intuition**：这相当于把 satellite-specific 的 camera model "翻译" 成 3DGS pipeline 能消化的标准 perspective model。Approximation 会有误差，但 DFC2019 这种 WorldView-3 数据视场窄，perspective 近似合理。

### 3.3 Appearance Modeling for Multi-date Imagery

DFC2019 的 satellite images 跨越多天甚至多月采集，illumination、season、transient objects（车、云、影子）变化巨大（参考 Figure 15）。直接用 single appearance 3DGS 训练会失败（ablation Table 3 第一行直接 Failed）。

Paper 借鉴 [WildGaussians (Kulhanek et al., 2024)](https://wild-gaussians.github.io/) 的设计：

- **Per-image embedding** $e_j \in \mathbb{R}^{32}$，$j = 1, \ldots, N$，每个 training image 一个
- **Per-Gaussian embedding** $g_i \in \mathbb{R}^{24}$，每个 Gaussian 一个
- **0-th order SH** $\bar{c}_i$（注意 paper 限制 SH 到 0 阶和 1 阶，避免 appearance change 被建模成 view-dependent effect）

Lightweight MLP $f$ 接受这三者，输出 affine color transform $(\beta, \gamma)$：

$$(\beta, \gamma) = f(e_j, g_i, \bar{c}_i)$$

Final transformed color：

$$\tilde{c}_i(\mathbf{r}) = \gamma \cdot \hat{c}_i(\mathbf{r}) + \beta$$

- $\hat{c}_i(\mathbf{r})$：原始 view-dependent SH color（$\mathbf{r}$ 是 view direction）
- $\gamma$：per-channel scale（类似于 contrast）
- $\beta$：per-channel shift（类似于 brightness）

**Intuition**：$\beta, \gamma$ 类似于 InstanceNorm 的 affine 参数，但被 $e_j$（image-level）和 $g_i$（Gaussian-level）共同 condition。$e_j$ 捕捉 global illumination（光照、季节），$g_i$ 捕捉局部 appearance variation（阴影、反射）。

MLP 架构：2 hidden layers, 128 neurons, ReLU。learning rate：$e_j$ = 0.001，$g_i$ = 0.005，$f$ = 0.0005。

### 3.4 Opacity Regularization：去掉 Floaters

公式 2：

$$\mathcal{L}_{\text{op}} = -\sum_i \left[\alpha_i \log \alpha_i + (1-\alpha_i)\log(1-\alpha_i)\right]$$

注意这里的数学：方括号内是负的 binary entropy，外面再乘 $-1$，所以 $\mathcal{L}_{\text{op}} = \sum_i H(\alpha_i)$（$H$ 是 binary entropy $-p\log p -(1-p)\log(1-p)$）。

最小化 $\mathcal{L}_{\text{op}}$ = 最小化 $\sum H(\alpha_i)$ = 让每个 $\alpha_i$ 趋向 0 或 1（binary 分布 entropy 最小）。

**Intuition**：Satellite imagery 的 parallax 太小，3DGS 容易用 low-opacity 的 "fog" Gaussians 模糊掉高 depth uncertainty 区域。Binary entropy penalty 强制每个 Gaussian 要么 "实在"（$\alpha \to 1$）要么 "消失"（$\alpha \to 0$），便于 densification 阶段 prune 掉那些 "fence-sitter" Gaussians。

权重 $\lambda_{\text{op}} = 10$，相当大，说明这个 term 很关键。

### 3.5 Pseudo-Camera Depth Supervision：解决 Limited Parallax

这是 paper 里我觉得最 clever 的 design 之一。

Satellite 视角本身 parallax 不够，但如果我们采样**离地面更近的 pseudo-cameras**，渲染它们的 RGB 和 depth，然后用 monocular depth estimator 估计 depth，反过来 supervise 3DGS，就把 monocular prior "蒸馏" 进 3DGS 了。

**Pipeline**：
1. Pseudo-camera sampling：
   - Look-at points $(x, y, z)$ with $x, y \sim \mathcal{N}(0, 128)$, $z = 0$（地面 plane）
   - 24 views per 10 iterations
   - Azimuth $\sim \text{Uniform}(0, 2\pi)$
   - Elevation 从 80° 线性降到 45°
   - Radius 从 300 线性降到 250
2. Render RGB $I_{\text{RGB}}$ (1024×1024) 和 depth $\hat{D}_{\text{GS}}$ from pseudo-cameras
3. [MoGe (Wang et al., 2024)](https://arxiv.org/abs/2410.19115) 估计 scale-invariant depth $\hat{D}_{\text{est}}$
4. Supervise via Pearson correlation（公式 3）：

$$\mathcal{L}_{\text{depth}} = \|\text{PCorr}(\hat{D}_{\text{GS}}, \hat{D}_{\text{est}})\|_1$$

$$\text{PCorr}(\hat{D}_{\text{GS}}, \hat{D}_{\text{est}}) = \frac{\text{Cov}(\hat{D}_{\text{GS}}, \hat{D}_{\text{est}})}{\sqrt{\text{Var}(\hat{D}_{\text{GS}})\text{Var}(\hat{D}_{\text{est}})}}$$

变量：
- $\text{Cov}$：协方差
- $\text{Var}$：方差
- PCorr $\in [-1, 1]$

**为什么用 Pearson correlation 而不是 L2**：因为 MoGe 输出的是 scale-invariant depth，绝对 scale 未知。L2 会强行让 3DGS 的绝对 depth 匹配 MoGe 的某个 scale，可能造成 metric scale distortion。Pearson correlation 是 scale-invariant 和 shift-invariant，只监督 depth 的 **relative structure**（哪里高、哪里低、slope 如何变化），保留 3DGS 自己学到的 metric scale。

权重 $\lambda_{\text{depth}} = 0.5$。

**Intuition**：这相当于 monocular depth estimator 充当 "shape prior"，告诉 3DGS "这里应该是平面屋顶"、"那里应该是斜坡"，而不用提供绝对 distance。这跟我们以前做 sparse-view NeRF 用 monocular depth prior 思路类似（[DNGaussian](https://arxiv.org/abs/2403.06912), [RegNeRF](https://arxiv.org/abs/2105.04085)）。

### 3.6 总体 Loss（公式 4）

$$\mathcal{L}_{\text{sat}}(G, C) = \mathcal{L}_{\text{color}} + \lambda_{\text{op}}\mathcal{L}_{\text{op}} + \lambda_{\text{depth}}\mathcal{L}_{\text{depth}}$$

- $G$：3DGS representation
- $C$：satellite ground-truth images
- $\lambda_{\text{op}} = 10$, $\lambda_{\text{depth}} = 0.5$

### 3.7 其他训练细节（Appendix A.1）

- 30,000 iterations，densification between iter 1,000-21,000
- Scaling learning rate 从标准 0.005 → 0.001（防止 satellite top-down view 下 Gaussian 沿 depth 方向 elongate）
- Densification gradient threshold 从 0.002 → 0.001（保证 close-up view 下 Gaussian 充足）
- Prune max covariance > 20 的 Gaussians（删 floater）
- 单卡 RTX A6000，~1 小时

---

## 4. Stage 2: Curriculum-based Iterative Dataset Update (IDU)

这是 paper 最核心的 contribution。

### 4.1 Motivation：为什么需要 Curriculum

Figure 4 揭示一个关键 observation：**satellite-trained 3DGS 在高 elevation 视角渲染质量好，低 elevation 视角严重退化**。

这其实很 intuitive：
- 高 elevation（接近 top-down）→ 接近训练 distribution → 渲染 OK
- 低 elevation（接近 ground view）→ OOD，facade 区域完全无 supervision → 渲染出 noise-like artifacts

如果直接用扩散模型 refine 低 elevation render，模型面对 "全是 noise" 的输入会乱 hallucinate，几何破碎。

**Solution**：从高 elevation 开始 refine（这时候 render 还有信号），让 3DGS 在高质量监督下 improve；然后逐渐降低 elevation，让 3DGS 适应越来越 challenging 的视角。这就像 curriculum learning，从 easy task → hard task。

### 4.2 Curriculum 参数化

- $N_p$ look-at points $\{P_i\}_{i=1}^{N_p}$：均匀放在场景里
- $N_v$ cameras per point，沿 orbital trajectory
- $N_e$ episodes（paper 用 5）
- 每个 episode $i$：elevation $E_i$ 和 radius $R_i$
  - DFC2019: elevation 80°→45°, radius 300→250, $N_p = 9$（3×3 grid）, $N_v = 6$, $N_s = 2$
  - GoogleEarth: elevation 85°→45°, radius fixed 600, $N_p = 16$, $N_v = 6$, $N_s = 2$
- 每个 episode 10,000 iterations，densification 到 9,000

### 4.3 Render Refinement via FlowEdit

Paper 用 [FlowEdit (Kulikov et al., 2024)](https://arxiv.org/abs/2412.08629) 配合 [FLUX.1 [dev]](https://blackforestlabs.ai/) 做 image editing。

**Why FlowEdit not SDEdit / SDS / InstructPix2Pix**：
- [SDEdit](https://sde-image-editing.github.io/) 加 noise 后会丢失结构，对 "噪点已经很重的 render" 会更糟
- [SDS / DreamFusion](https://dreamfusion3d.github.io/) 容易 over-smooth
- [InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix/) 在 NeRF 编辑里很流行（[Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/)），但需要专门训练，对 satellite-style domain 不一定 generalization 好
- FlowEdit 是 **inversion-free** 的 flow-based editing 方法，能基于 source/target prompt pair 做精确的 edit，保留结构、只改变指定 attribute

**Prompt 设计**：
- Source: "Satellite image of an urban area with modern and older buildings, roads, green spaces. Some areas appear distorted, with blurring and warping artifacts."
- Target: "Clear satellite image of an urban area with sharp buildings, smooth edges, natural lighting, and well-defined textures."

**Intuition**：source prompt 明确点出 degraded render 的问题（distortion, blurring, warping），target prompt 描述 desired properties。FlowEdit 在 source/target 之间做 flow-based translation，保留 geometry（source 提到的 buildings/roads），改变 appearance（distortion → sharp, blurring → well-defined textures）。

**Noise levels**：$n_{\min} = 4$, $n_{\max} = 10$
- Low noise（$n=4$）：保留更多原始结构，artifacts 去得少
- High noise（$n=10$）：变化大，可能 alter geometry
- Paper 在 $[n_{\min}, n_{\max}]$ 区间采样，balance 两者

### 4.4 Multiple Diffusion Samples per View

这是一个 subtle 但关键的设计。

**Problem**：对每个 view 独立做 2D diffusion refinement，不同 view 之间不一致。3DGS 在多 view 上 train，如果 supervision views 不一致，会 overfit 到 single view（[CoR-GS](https://cor-gs.github.io/) 指出过这个问题），导致 novel view 出现 artifacts。

**Why 不能简单 average**：理想情况下，optimal denoising trajectory 应该让所有 view 同步 3D appearance。但独立 2D denoising 每个 view 走自己的 trajectory，得到的 distribution 是 optimal trajectory distribution 的 super-set。从中采样一次得到 3D-consistent 结果的概率几乎为 0。

**Solution**：每个 view 做 $N_s = 2$ 次独立 sampling。这样每个 view 有 2 个 slightly different refinement 候选。3DGS 在它们上面 minimize $\mathcal{L}_{\text{color}}$，相当于隐式 average，找到 consensus representation。

**Intuition**：这有点像 ensemble learning。每个 sample 是一个 noisy estimator of "理想 refined view"，多次采样降低 variance，3DGS optimization 自动找到它们共同同意的 3D 表示。这与 [Variational Score Distillation (ProlificDreamer)](https://ml.cs.tsinghua.edu.cn/~prolificdreamer/) 的 motivation 有相通之处——单个 sample 路径太尖，多个 sample 平均更稳。

### 4.5 Iterative Dataset Update Loop（Algorithm 1）

```
Input: N_e episodes, N_v, N_s, N_p, look-at points {P_i},
       decreasing sequences {R_i}, {E_i}, FlowEdit params, initial 3DGS G
Output: refined 3DGS G'

G' ← G
for i = 1 to N_e:
    radius, elevation ← R_i, E_i
    cam_views ← OrbitViews({P_i}, radius, elevation, N_v)  # N_p × N_v views
    render_views ← Render(G', cam_views)
    refine_views ← FlowEditRefine(render_views, prompts, N_s)  # N_s samples per view
    G' ← Train(G', refine_views)
return G'
```

**IDU Loss**（公式 5）：

$$\mathcal{L}_{\text{IDU}}(G_{i-1}, \tilde{C}_i) = \mathcal{L}_{\text{color}} + \lambda_{\text{depth}}\mathcal{L}_{\text{depth}}$$

注意几点：
- 训练时 75% 用 refined views，25% 用 original satellite views。**这非常关键**——保证 3DGS 不忘 satellite input 的 ground truth，semantic 和 layout 始终 anchor 到 satellite data
- IDU 阶段关掉 opacity regularization——因为 curriculum 自然通过 multi-view consistency 抑制 floaters，且保留 variable opacity 对半透明结构（树叶、玻璃）有益
- IDU 阶段固定一个 single appearance embedding $e_j$，统一 refined views 的 appearance
- 总 IDU 时间：~6 小时 on A6000

---

## 5. 实验结果深度解读

### 5.1 Datasets

- **DFC2019** ([Le Saux et al., 2019](https://www.grss-ieee.org/community/technical-committees/2019-ieee-grss-data-fusion-contest/))：WorldView-3，Jacksonville, Florida，35 cm/pixel，2048×2048。4 个 AOI（JAX_004, 068, 214, 260），训练 images 数量 9-21 张（Table 5）。Ground truth 由 Google Earth Studio 渲染。
- **GoogleEarth** ([Xie et al., 2024](https://www.infinitivity.wustl.edu/citydreamer/))：NYC 场景。每个 AOI 60 张 80° elevation 渲染作为 satellite-like input。4 个 AOI（004, 010, 219, 336）。

### 5.2 Metrics

主要用 distribution-based：
- [FID_CLIP (Kynkäänniemi et al., 2023)](https://arxiv.org/abs/2201.12970)：用 CLIP 特征算 FID，比 InceptionV3-based FID 更适合现代生成模型
- [CMMD (Jayasumana et al., 2024)](https://arxiv.org/abs/2401.09603)：CLIP-based Maximum Mean Discrepancy

辅助用 pixel-level：PSNR, SSIM, LPIPS（在 GoogleEarth 上有意义，因为 GES 渲染的 ground truth 跟 input 来自同一个 3D representation，没有时变）。

**Important note**：pixel-level metric 在 generative task 上意义有限。Sat-NeRF 等 baseline 把 satellite 训练 view 当 ground truth 测 PSNR，自然高；但实际 ground-view 生成质量差。这也是 paper 主张用 distribution-based metric 的理由。

### 5.3 Quantitative Results（Table 1 & 2）

**DFC2019 (Table 1)**：
| Method | FID_CLIP ↓ | CMMD ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| Sat-NeRF | 88.36 | 4.868 | 10.05 | 0.269 | 0.864 |
| EOGS | 87.74 | 5.286 | 7.26 | 0.168 | 0.959 |
| Mip-Splatting | 87.19 | 5.405 | 11.89 | 0.318 | 0.819 |
| CoR-GS | 89.03 | 5.241 | 11.55 | 0.350 | 0.948 |
| **Ours** | **27.35** | **2.086** | **12.38** | 0.321 | **0.791** |

FID_CLIP 从 ~88 → 27，**3 倍以上提升**，巨大。CMMD 从 ~5 → 2.086，巨大。说明 perceptual 质量碾压。PSNR/SSIM/LPIPS 提升幅度不大，但本来这些 metric 对 generative task 意义就有限。

**GoogleEarth (Table 2)**：
| Method | FID_CLIP ↓ | CMMD ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| CityDreamer | 36.52 | 4.152 | 12.58 | 0.267 | 0.558 |
| GaussianCity | 28.73 | 2.917 | 13.41 | 0.291 | 0.541 |
| CoR-GS | 27.32 | 3.752 | 12.85 | 0.291 | 0.455 |
| **Ours** | **9.91** | **2.009** | **14.28** | 0.298 | **0.394** |

FID_CLIP 从 ~28-37 → 9.91，又是数倍提升。

### 5.4 User Study (Figure 7)

89 个 participants。
- DFC2019：vs Sat-NeRF winrate 97%/97%/97%，vs EOGS/CoR-GS 接近 100%
- GoogleEarth：vs CityDreamer 90%/90%/92%，vs GaussianCity 89%/87%/89%

人主观感知与 distribution metric 完全一致。

### 5.5 Ablation Studies

**Reconstruction Stage (Table 3)**：
| App. Modeling | Opacity Reg. | Depth Sup. | FID_CLIP ↓ | CMMD ↓ |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | Failed | Failed |
| ✓ | ✗ | ✗ | 41.90 | 2.45 |
| ✓ | ✓ | ✗ | 39.95 | 2.40 |
| ✓ | ✓ | ✓ | 38.01 | 2.31 |

- 没有Appearance modeling 直接 Failed——multi-date 不收敛
- Opacity reg 让 FID 降 2 个点
- Depth supervision 再降 ~2 个点

**Synthesis Stage (Table 4)**：
| Multi-sample | Curriculum | FID_CLIP ↓ | CMMD ↓ |
|---|---|---|---|
| ✗ | ✗ | 34.11 | 3.19 |
| ✓ | ✗ | 33.79 | 3.36 |
| ✓ | ✓ | **28.35** | **2.88** |

注意：单独加 multi-sample 反而让 CMMD 升高（3.19→3.36）！但配合 curriculum 后效果大幅提升。这说明 multi-sample 是 "细节增强器"，curriculum 是 "几何修复器"，两者协同。

整体看 ablation 数字（FID 从 Failed → 41 → 38 → 28），每个 component 都有意义，curriculum 是最后的杀手锏。

### 5.6 Per-AOI Results (Table 8, 9)

Per-AOI 一致地领先。例如 JAX_004 上 FID_CLIP 24.45 vs Sat-NeRF 79.97, EOGS 107.23, CoR-GS 91.01——EOGS 反而最差，可能因为 EOGS 是 photogrammetry 方法，OOD ground view 渲染严重。

### 5.7 Rendering Efficiency

- 11 FPS on NVIDIA T4（消费级）
- 40 FPS on MacBook Air M2
- 对比：CityDreamer 0.18 FPS on A100，GaussianCity 10.72 FPS on A100

Skyfall-GS 因为输出就是 vanilla 3DGS（没有复杂 data structure），real-time 渲染天然支持。

---

## 6. 与相关工作的 Positioning

### 6.1 vs City Generation Methods

[CityDreamer (Xie et al., 2024)](https://www.infinitivity.wustl.edu/citydreamer/) 和 [GaussianCity (Xie et al., 2025)](https://arxiv.org/abs/2506.06507) 用 BEV semantic map + height field 作为 input，在 GoogleEarth 数据集上训练生成模型。

**Limitations**：
- 需要 pixel-aligned semantic maps 和 height fields（不是任意 satellite imagery）
- 训练 domain 限制 NYC，OOD 场景 generalization 差
- Building geometry oversimplified（高度场 = 拉伸盒子）
- 无法生成 bridges/tunnels/multi-level structures

Skyfall-GS 用 diffusion prior 而非训练专用 generator，零样本泛化到任意 urban area。

### 6.2 vs Satellite-based 3D Reconstruction

[Sat-NeRF (Marí et al., 2022)](https://github.com/centreborelli/sat-nerf), [EOGS (Savant Aira et al., 2025)](https://arxiv.org/abs/2408.15762), [Mip-Splatting](https://github.com/autonomousvision/mip-splatting), [CoR-GS (Zhang et al., 2024)](https://cor-gs.github.io/) 都是 reconstruction 路线。

**Limitations**：fundamentally 受限于 satellite parallax，无法补全 facades（这部分在训练 images 里根本不可见，属于信息缺失而非优化问题）。

Skyfall-GS 的 solution：用 diffusion prior "想象" 出 facades，再回灌 3DGS。

### 6.3 vs SDS / Diffusion-Driven 3D Generation

[DreamFusion (Poole et al., 2022)](https://dreamfusion3d.github.io/), [Magic3D (Lin et al., 2023)](https://deepshading.org/magic3d/), [ProlificDreamer (Wang et al., 2023)](https://ml.cs.tsinghua.edu.cn/~prolificdreamer/), [GaussianDreamer (Yi et al., 2024)](https://github.com/buaacyw/GaussianDreamer) 都是 text-to-3D 路线，用 SDS/VSD 把 2D diffusion prior 蒸馏到 3D。

Skyfall-GS **不走 SDS 路线**，而是用 [IM-3D (Melas-Kyriazi et al., 2024)](https://im3d-url.github.io/) 和 [Instruct-NeRF2NeRF (Haque et al., 2023)](https://instruct-nerf2nerf.github.io/) 风格的 **iterative dataset update**：直接用 diffusion refined image 作为 pseudo ground truth 训练 3DGS，比 SDS 更直接、更稳。

**Why IDU > SDS**：
- SDS 用 score function 引导梯度，常常 over-smooth
- IDU 直接提供 image-level supervision，3DGS 优化目标明确
- IDU 可以 reuse 3DGS 的 standard loss（color + L1 + D-SSIM）
- IDU 自然支持 multi-view consistency（多个 view 同时 refine）

### 6.4 vs Street-view Synthesis

[Sat2Scene (Li et al., 2024)](https://sat2scene.github.io/), [SkyDiffusion (Ye et al., 2024)](https://arxiv.org/abs/2408.07959), [Sat2Vid (Li et al., 2021)](https://arxiv.org/abs/2012.06628) 都是 satellite-to-street view synthesis，但输出是 2D image 或 video，**不是 navigable 3D**。

Skyfall-GS 输出的是真正的 3D Gaussian 表示，支持 6-DOF 自由飞行。

---

## 7. 我的 Critique 和潜在改进方向

### 7.1 Strengths

1. **Concept clean**：把 noisy 3DGS render 当作 diffusion 中间 step，是很 elegant 的 idea。这把 "卫星图重建缺失 facades" 这个 fundamentally ill-posed problem 转化成 "用 diffusion prior hallucinate plausible facades"。
2. **Curriculum design motivation 强**：Figure 4 的 observation 是 paper 的 key insight，从中自然导出 curriculum 设计。
3. **Generalization 强**：zero-shot 用 FLUX.1，不需要 city-specific 训练。
4. **Output 是 vanilla 3DGS**：实时渲染、可移植到任何 GS viewer。
5. **Ablation 完整**：每个 component 都验证，包括 multi-sample 这种 subtle 的设计。

### 7.2 Weaknesses / 开放问题

1. **Compute cost**：6 小时 IDU + 1 小时 reconstruction = 7 小时 per AOI。Scaling 到 city-scale（不是 block-scale）会爆炸。
2. **Street-level perspective over-smooth**（paper 自己承认 limitation）：因为 IDU curriculum 最低只到 45° elevation，street-level（接近 0°）没直接 supervise。
3. **3D consistency 仍 imperfect**：multi-sample 缓解但没彻底解决。N_s = 2 偏小，increase 会有 diminishing return 但 sample 效率有 trade-off。
4. **Prompt engineering**：source/target prompt 需要手动设计。如果用 vision-language model 自动 generate prompts per view 会更鲁棒。
5. **No temporal / dynamic**：静态场景。Future work 提到 dynamic scenes。
6. **Identity preservation**：对于标志性 building 的特定 facade pattern（窗户排列、广告牌），diffusion hallucinate 可能 invent 不存在的内容。Paper 没显式 handle 这个问题。

### 7.3 Potential Extensions（我自己的联想）

1. **用 video diffusion 替代 per-view refinement**：[CAT4D (Wu et al., 2024)](https://arxiv.org/abs/2411.18613) 或 multi-view diffusion ([MVDream](https://arxiv.org/abs/2308.16512)) 能在 refinement 阶段就 impose multi-view consistency，省掉 multi-sample 的 hack。
2. **Self-supervised prompt generation**：用 [LLaVA](https://llava-vl.github.io/) 或类似 VLM 自动描述 degraded render 的问题，并 generate target prompt。
3. **Geometry-aware diffusion**：把 3DGS depth/normal 作为 condition 给 diffusion（类似 [ControlNet](https://arxiv.org/abs/2302.05543)），而不是 image-to-image refinement，能更精确 preserve geometry。
4. **Scale to city**：结合 hierarchical 3DGS ([Kerbl et al., 2024](https://arxiv.org/abs/2406.19390)) 或 [VastGaussian](https://arxiv.org/abs/2402.17427) 的分块策略，把 IDU 应用到 block-level。
5. **Dynamic extension**：[SpectroMotion](https://arxiv.org/abs/2501.05640) 风格处理 specular/dynamic 场景。
6. **Better depth prior**：用 [Depth Anything V2](https://arxiv.org/abs/2406.09414) 或 satellite-specific monocular depth estimator 替代 MoGe，可能在高 altitude view 更准。
7. **Inverse rendering**：从 refined 3DGS 提取 material/lighting，可后续 relighting（参考 [GS-IR](https://arxiv.org/abs/2311.16473)）。
8. **Robotics simulation**：Skyfall-GS 输出可作为 embodied AI 的 training environment（[Habitat](https://aihabitat.org/) 风格），satellite imagery → 3D playground for navigation policy training。

### 7.4 一些 Technical Curiosities

- **Opacity regularization 的设计选择**：paper 用 binary entropy，也可以用 [SpotLessSplats](https://arxiv.org/abs/2406.20055) 风格的 uncertainty-based pruning 或 [TransNeRF](https://arxiv.org/abs/2306.08738) 风格的 transient modeling。Binary entropy 简单但 effective。
- **Per-Gaussian embedding 24 维**：这个 dimension 选得偏小（WildGaussians 默认更大），可能因为 satellite view per-Gaussian appearance variation 较少。
- **Pearson vs Spearman**：Pearson 假设 linear relationship。如果用 rank correlation（Spearman）对 outlier 更鲁棒，但 gradient 不可微。Paper 用 Pearson 配合 L1 norm，可微且简单。
- **75/25 sampling ratio**：这个 mix 比例对最终结果很关键。如果 refined 占比太高，3DGS 漂离 satellite ground truth；太低，refined 信息传不进去。Paper 实验确定 75/25。
- **为什么 SH 限制到 0+1 阶**：避免 appearance modeling 被错误地解释为 view-dependent effect。0+1 阶 SH = essentially Lambertian + 简单 ambient term，足够 satellite 场景（high altitude, specular 少）。

---

## 8. 给 Karpathy 的 Intuition 总结

如果你只想 take away 三件事：

1. **Skyfall = "把 3DGS noisy render 当作 diffusion 中间 step" 的 metaphor**。Satellite-trained 3DGS 在 ground view 渲染出 noise-like artifacts，这些 artifacts 形态上像 DDIM/Flow matching 的 partial denoising 输出。Paper 用 FlowEdit 在 source prompt（描述 artifacts）和 target prompt（描述 clean image）之间走一段 flow，相当于完成剩下的 denoising，得到 pseudo ground-truth。

2. **Curriculum learning 的 motivation**：satellite-trained 3DGS 在 high elevation 渲染好（接近训练分布），low elevation 渲染坏（OOD）。如果一开始就 refine low elevation，diffusion 看到纯 noise 会乱 hallucinate。所以先 refine high elevation（信号多），让 3DGS 学到更好表示，再逐步降低 elevation，每一步都站在上一步的肩膀上。

3. **Multi-sample 缓解 3D inconsistency**：per-view 2D diffusion 独立采样会得到不一致 supervision，3DGS 在它们上面 minimize L_color 相当于 implicit ensemble，找到 consensus 3D 表示。这是 paper 一个 subtle 但 important 的 design。

这是 satellite-to-3D city generation 的一个 promising direction，把 generative prior 注入 reconstruction framework 的范式（参考 [ReconFusion](https://arxiv.org/abs/2312.02981), [CAT3D](https://arxiv.org/abs/2405.19415)）在 satellite domain 的具体实现。

---

## References

- [Skyfall-GS Project Page](https://skyfall-gs.jayinnn.dev/)
- [3D Gaussian Splatting (Kerbl et al., 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Mip-Splatting (Yu et al., 2024)](https://github.com/autonomousvision/mip-splatting)
- [WildGaussians (Kulhanek et al., 2024)](https://wild-gaussians.github.io/)
- [FlowEdit (Kulikov et al., 2024)](https://arxiv.org/abs/2412.08629)
- [FLUX.1 by Black Forest Labs](https://blackforestlabs.ai/)
- [MoGe: Monocular Geometry (Wang et al., 2024)](https://arxiv.org/abs/2410.19115)
- [SatelliteSfM (Zhang et al., 2019)](https://openaccess.thecvf.com/content_ICCVW_2019/papers/3DIMP/Leveraging_Vision_Reconstruction_Pipelines_for_Satellite_Imagery_ICCVW_2019_paper.pdf)
- [Sat-NeRF (Marí et al., 2022)](https://github.com/centreborelli/sat-nerf)
- [EOGS / Gaussian Splatting for Satellite (Savant Aira et al., 2025)](https://arxiv.org/abs/2408.15762)
- [CoR-GS (Zhang et al., 2024)](https://cor-gs.github.io/)
- [CityDreamer (Xie et al., 2024)](https://www.infinitivity.wustl.edu/citydreamer/)
- [GaussianCity (Xie et al., 2025)](https://arxiv.org/abs/2506.06507)
- [DFC2019 Dataset](https://www.grss-ieee.org/community/technical-committees/2019-ieee-grss-data-fusion-contest/)
- [DreamFusion (Poole et al., 2022)](https://dreamfusion3d.github.io/)
- [ProlificDreamer / VSD (Wang et al., 2023)](https://ml.cs.tsinghua.edu.cn/~prolificdreamer/)
- [SDEdit (Meng et al., 2022)](https://sde-image-editing.github.io/)
- [Instruct-NeRF2NeRF (Haque et al., 2023)](https://instruct-nerf2nerf.github.io/)
- [IM-3D (Melas-Kyriazi et al., 2024)](https://im3d-url.github.io/)
- [CAT3D (Gao et al., 2024)](https://cat3d.github.io/)
- [CAT4D (Wu et al., 2024)](https://arxiv.org/abs/2411.18613)
- [MVDream (Shi et al., 2023)](https://arxiv.org/abs/2308.16512)
- [Sat2Scene (Li et al., 2024)](https://sat2scene.github.io/)
- [SkyDiffusion (Ye et al., 2024)](https://arxiv.org/abs/2408.07959)
- [DNGaussian (Li et al., 2024)](https://arxiv.org/abs/2403.06912)
- [ReconFusion (Wu et al., 2023)](https://arxiv.org/abs/2312.02981)
- [VastGaussian (Lin et al., 2024)](https://arxiv.org/abs/2402.17427)
- [Hierarchical 3DGS (Kerbl et al., 2024)](https://arxiv.org/abs/2406.19390)
- [FID_CLIP (Kynkäänniemi et al., 2023)](https://arxiv.org/abs/2201.12970)
- [CMMD (Jayasumana et al., 2024)](https://arxiv.org/abs/2401.09603)
- [Google Earth Studio](https://earth.google.com/studio/)
