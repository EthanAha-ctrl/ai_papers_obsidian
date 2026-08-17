---
source_pdf: WorldSplat.pdf
paper_sha256: 21e1c00142cef6f290b89295523e64f7c4b60248f9d4504f4f182a1287fa3ed0
processed_at: '2026-08-13T05:59:29-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 WorldSplat

好，我换个讲法，把刚才那堆技术名词翻译成"你坐在副驾上听我讲"的版本。

---

## 一句话总结

**WorldSplat 干的事**：给它一张地图布局、几个 3D box、一段文字描述、一条 ego 车要开的轨迹，它直接吐出一坨 4D Gaussian，你想从哪个视角看、哪个时刻看，都能 render 出来，还自带时序一致性。

---

## 为什么这事难：Generation 和 Reconstruction 的老矛盾

先讲个故事。假设你想给自动驾驶训练做 synthetic data。

**路线 A：video generation**（MagicDrive, Vista, Panacea 这帮人）。你训练一个 video diffusion model，喂 BEV layout + box + caption，它给你生成 6 个 camera 的 driving video。效果惊艳，FVD 很低。但问题来了——你想让 ego car 往左偏 2 米，再看一遍这个场景，video model 直接懵逼。为什么？因为它在 2D pixel space 学的，没有"3D 世界"的概念。换视角 = 重新 sample 一次 latent = 完全不同的场景。同一个红车，原视角在左边，新视角跑到右边，identity 直接漂移。

**路线 B：scene reconstruction**（OmniRe, StreetGaussian, EmerNeRF 这帮人）。你拿真实录的 driving log，per-scene 优化一坨 4D Gaussian，得到 metric-consistent 的 3D 场景。换视角随便看，几何绝对对。但问题是你只能 reconstruct 录过的场景，没法 hallucinate "如果路口多一辆车会怎样"。

**以前的 bridge**（MagicDrive3D, DreamDrive）：先 generate video，再 reconstruct 4D。听起来对，但第二阶段永远被第一阶段拖累。video 自己的 perspective jitter、object drift 全传进 Gaussian 里。

**WorldSplat 的核心 idea**：别分两步了。让 diffusion 在 latent space 直接吐 4D structure，把 geometry 一致性变成 diffusion 自己的优化目标，从 root 上解决 drift。

---

## 核心直觉：output space 决定 model 学什么

这点我猜你会特别 appreciate，因为这跟你讲 micrograd / makemore 时强调的 "output space 定义 loss signal" 哲学一致。

传统 video diffusion 输出 RGB latent，模型只学 2D appearance statistics——它不知道"车是 3D 的"，只知道"这几个 pixel 长得像车"。

WorldSplat 在 latent 里塞了 **三个 channel**：
- `L_img`：RGB latent（OpenSora VAE encode 的）
- `L_depth`：metric depth latent（用 Metric3D v2 抽出来再 VAE encode）
- `L_seg`：dynamic object binary mask latent（SegFormer 抽出来）

这三个 latent 在 channel-wise 对齐。意思就是 latent 上每个 spatial position 都自带 RGB + depth + dynamic/static 标签。

**直觉**：等于给 diffusion 一个"几何先验通道"。它在 denoise 时必须同时 hallucinate appearance、geometry、semantic 三件事，三者在 latent space 是 spatially aligned 的。下游 decoder 读 latent 时，每个 pixel 都自带 metric depth 监督，几何直接长在 generation 里。

这跟 SD3 用 Rectified Flow、跟你讲 "output space 决定 model 学什么" 的思路一脉相承——你想让 model 学 4D，就别给它 2D output space。

---

## 三个模块串讲（人话版）

### Module 1：4D-aware Latent Diffusion

输入：BEV sketch + 3D boxes + ego trajectory + text caption

输出：multi-modal latent L = {RGB latent, depth latent, seg latent}

架构：OpenSora v1.2 改的 dual-branch DiT + ControlNet。Main stream 处理 video latent，ControlNet branch 注入 conditions。关键 trick 是 **cross-view attention 替换 self-attention**——把 6 个 camera 的所有 token 放进同一 attention sequence，强制 6 个视角在 latent space 互相 attend。这是 MagicDrive 的做法直接延续。

训练用 **Rectified Flow** 而不是 IDDPM。核心是 Eq. 1-3：

$$z(s) = (1-s)\epsilon + s \cdot x$$

- $s \in [0,1]$：mixing parameter，0 是纯噪声，1 是 clean latent
- $\epsilon$：Gaussian noise
- $x$：clean data
- 这条路径是 noise 到 data 的**直线**

训练目标（Eq. 2）：
$$\mathcal{L}(\psi) = \mathbb{E}\big[\|g_\psi(z(s), s, \mathcal{C}) - (x - \epsilon)\|^2\big]$$
- $g_\psi$：要训练的 network
- $(x - \epsilon)$：从 noise 指向 data 的 velocity vector
- 学 velocity field 而不是 noise prediction

推理（Eq. 3）：
$$z(s_{k-1}) = z(s_k) - \frac{1}{N} g_\psi(z(s_k), s_k, \mathcal{C})$$
- $s_k = k/N$，$N$ 步
- WorldSplat 只用 **8 步**（MagicDrive-V2 要 30+ 步）

**直觉**：Rectified Flow 路径是直线，Euler 积分走几步就行，IDDPM 路径是曲线要走很多步。WorldSplat 用这个主要是 inference latency——后面还有第二个 diffusion 要跑，diffusion 步数必须压缩。

---

### Module 2：Latent 4D Gaussian Decoder（最有 contribution 的部分）

输入：multi-modal latent L + Plücker ray map P

输出：per-frame pixel-aligned 3D Gaussians + dynamic mask

每个 pixel 对应一个 3D Gaussian $g = (\mu, r, s, \alpha, c)$：
- $\mu \in \mathbb{R}^3$：center
- $r \in \mathbb{R}^4$：quaternion rotation
- $s \in \mathbb{R}^3$：scale
- $\alpha$：opacity
- $c \in \mathbb{R}^3$：color

**最关键的 parameterization trick**：
$$\mu = R_o + d \odot R_d + \delta$$

- $R_o$：camera ray origin（per-pixel，从 Plücker map 来）
- $R_d$：camera ray direction（per-pixel，normalized）
- $d$：predicted per-pixel depth（scalar）
- $\delta$：learned residual offset

**为什么这么写重要**：如果直接预测 $\mu = (x, y, z)$，模型要从 pixel coordinate 学到 world coordinate 的非线性映射，难。用 camera ray 参数化，等于把"这个 pixel 在世界中的什么方向"先验地告诉 model，model 只需要学"沿这条 ray 走多远"（depth $d$）+ "微调一下"（$\delta$）。Gradient 沿 ray direction 自然 flow。

这个 trick 来自 pixelSplat（https://arxiv.org/abs/2312.12337）和 MVSplat（https://arxiv.org/abs/2403.14624）一脉。

**4D Aggregation**（Eq. 5）：
$$\mathcal{G}_{4D} = \Big\{ (G_t \odot M_t) \cup \bigcup_{i=1}^{T}(G_i \odot (1-M_i)) \Big\}_{t=1}^{T}$$

拆解：
- $M_t$：第 $t$ 帧 dynamic mask（1=dynamic, 0=static）
- $G_t \odot M_t$：第 $t$ 帧的 dynamic Gaussians
- $G_i \odot (1-M_i)$：第 $i$ 帧的 static Gaussians
- $\bigcup_{i=1}^{T}$：所有帧 static Gaussians 并集

**直觉**：static 场景（路、楼、天）在所有帧应该在 world frame 重合，union 起来增加 coverage。dynamic 物体每帧位置不同，必须用本帧的。每个时刻 $t$ 的 4D 表示 = 本帧 dynamic + 所有帧 static union。

这是 EmerNeRF 的 self-supervised decomposition 的简化版——不用 emergent learning，直接用 SegFormer 给的 mask 强行分。简单粗暴但有效。

**Loss**（Eq. 6）：
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{lpips}} + \lambda_2 \mathcal{L}_{\text{depth}} + \lambda_3 \mathcal{L}_{\text{seg}}$$

- $\mathcal{L}_{\text{recon}}$：photometric $L_1$（RGB 渲染 vs GT）
- $\mathcal{L}_{\text{lpips}}$：LPIPS perceptual loss
- $\mathcal{L}_{\text{depth}}$：metric depth $L_1$
- $\mathcal{L}_{\text{seg}}$：dynamic mask BCE

训练时随机抽 base timestep $t$ + 采 $T$ 个 target timesteps，把 4D Gaussians 投影到这些 target 渲染监督。**这个随机时间投影**让模型学会从几帧 latent 推出整个 4D scene。

---

### Module 3：Enhanced Diffusion Model（补丁）

为什么需要：Gaussian splatting 有两个老毛病：
1. **Unobserved region**：ego shift 后，原本被遮挡的区域没 Gaussian，渲染出来是空洞（Figure 3 上图天空被砍）
2. **Fast motion blur**：没 per-scene optimization，fast motion 区域 Gaussian 重叠不精确

解决方案：再来一个 diffusion model，input 是 4D Gaussian rendered video + 重新投影的 conditions，target 是 GT video latent。等于用 generative prior inpaint 一下 + sharpen 一下。

ReconDreamer（https://arxiv.org/abs/2411.19548）的思路是用 degraded rendering 训练。WorldSplat 改成 **mixed-conditioning**：训练时一部分 sample 用 degraded rendering，一部分用 high-quality，避免 condition-output alignment 太弱。

---

## Inference：trajectory perturbation

给定原 ego trajectory $\{\mathcal{T}_i\}$，沿 y-axis 加 offset $\Delta y \in \{+1m, -1m, +2m, -2m, +4m, -4m\}$，生成 6 条 perturbed trajectories。

**直觉**：driving 场景里最直观的 novel view 就是"如果车在隔壁车道开会看到什么"。±4m = 2 个车道外，相当激进的 extrapolation。

---

## 实验结果的人话解读

### Table 1：原视角 video generation

- **无 first frame**：WorldSplat FVD 74.13 vs MagicDrive-V2 94.84，降 22%。纯 generation 能力胜出。
- **有 first frame**：WorldSplat FVD 16.57，DriveDreamer-2 是 55.70，**3 倍提升**。说明 4D-aware diffusion 在有 anchor 时，长序列 extrapolate 非常稳。
- **Noisy latent**：60.84 FVD，比 UniScene 70.52 低，但 FID 6.51 略高于 UniScene 6.12。Frame quality 略差，video consistency 更好——4D 结构约束的 trade-off。

### Table 2：novel view synthesis（最关键）

±4m shift 下 WorldSplat FVD 64.07 vs DiST-4D 105.29，降 39%。vs OmniRe（per-scene SOTA）FID 13.38 vs 67.36，**5 倍**。Shift 越大优势越明显——4D aggregation 在大 baseline extrapolation 时 coverage 优势突出。

feed-forward 模型超越 per-scene optimization 是强证据。

### Table 3 ablation

- Version A（无任何 Gaussian，纯 condition reproject）：FVD 260
- Version B（加单帧 3D Gaussian）：FVD 75，**3.5 倍提升**。3D structure 是 game changer。
- Version C（3D → 4D aggregation）：FVD 50.7。Multi-frame aggregation 明显有用。
- Version D（去 Enhanced Diffusion）：FVD 107.6。Enhanced model 贡献 FVD 一半。
- Version E（完整）：FVD 47.4

### Table 5：inference speed

17-frame 6-view 424×800：
- MagicDrive-V2: 3.85 min, 26 GB
- WorldSplat: 2.50 min, 22 GB

两个 diffusion 居然比一个 diffusion 快——靠 Rectified Flow 8 步 vs 30 步。

---

## 我的几点 intuition 和 critique

### 1. 这个 design 为什么 work

output space 决定 model 学什么。传统 video diffusion 输出 RGB，model 只学 2D appearance statistics。WorldSplat 输出含 depth + seg 的 latent，model 被迫学 3D geometry + dynamic decomposition。把 3D prior 烧进 generative prior 里，比事后 reconstruct 高效得多。

类似思路：DiST-4D（https://arxiv.org/abs/2503.15208）、Gen3C（https://arxiv.org/abs/2410.15277）、SCube（https://arxiv.org/abs/2410.20030）。这一脉都在探索 "geometry-aware generation"。

### 2. paper 没说清楚的局限

**Dynamic object 跨帧 identity**：4D aggregation 用 per-frame dynamic Gaussian，但同一辆车在 t=1 和 t=5 的 Gaussian 没 explicit correspondence。车移动很快时 Gaussian 之间 disconnect，rendering 可能 "tear"。

**天空和远景**：Gaussian splatting 对 distant element 建模差，需要很多大 Gaussian。Paper 用 Enhanced Diffusion 补，但 root cause 没解决。

**Long video**：17 frames × 12Hz ≈ 1.4s。对 long-horizon simulation 不够。MagicDrive-V2 已经几十秒。

**Lighting/relighting**：Gaussian color 是 view-independent RGB，没 SH，无法 relighting。对 perception data aug 够，对 sensor simulation 不够。

**Ego perturbation 假设 static world**：现实里其他 dynamic car 会因 ego shift 反应。WorldSplat 假设 dynamic object 跟原 trajectory 走，simplification。

### 3. 这个方向让我联想到

- **Tesla world model**：你 Twitter 提过几次，Tesla 的 "occupancy + diffusion" hybrid。WorldSplat 把 explicit 4D 烧进 generative model，方向接近。
- **Wayve GAIA-2**：generalizable driving world model，思路类似但没用 explicit Gaussian。
- **Sora**（https://openai.com/sora）：spatiotemporal patches + DiT。WorldSplat 基于 OpenSora（Sora 开源复现），技术栈一脉相承。
- **L4GM**（https://arxiv.org/abs/2402.14574）：4D Gaussian from multi-view video，思路互补。
- **UniAD**（https://arxiv.org/abs/2212.10156）/ **VAD**（https://arxiv.org/abs/2403.14377）：end-to-end planning。下一步很可能看到 WorldSplat-style 4D generation 直接接进 planning loop，做 closed-loop driving simulation。这是这个方向最有想象力的应用。

### 4. 最有意思的 open question

两阶段 diffusion（generation + refinement）的 trade-off。Enhanced Diffusion 补 Gaussian splatting 固有缺陷，但带来额外 latency。Table 5 显示两段 diff 加起来 132s，占总 latency 88%。

如果能用更 robust 的 4D representation（比如 4D NeRF 或 adaptive Gaussian densification）替代 Enhanced Diffusion，会更 elegant。但工程难度大幅上升。这是我觉得这个方向最值得 follow 的问题。

---

## 关键 links

**核心**：
- WorldSplat 项目主页: https://wm-research.github.io/worldsplat/

**Driving world models**:
- GAIA-1: https://arxiv.org/abs/2309.17080
- Vista: https://arxiv.org/abs/2405.17398
- MagicDrive: https://arxiv.org/abs/2310.02601
- MagicDrive-V2: https://arxiv.org/abs/2411.13807
- DriveDreamer-2: https://arxiv.org/abs/2403.06845
- Panacea: https://arxiv.org/abs/2312.06709
- DiVE: https://arxiv.org/abs/2409.01595
- UniScene: https://arxiv.org/abs/2412.05435
- DiST-4D: https://arxiv.org/abs/2503.15208
- DreamDrive: https://arxiv.org/abs/2501.00601
- FreeVS: https://arxiv.org/abs/2410.18079
- ReconDreamer: https://arxiv.org/abs/2411.19548

**Scene reconstruction**:
- OmniRe: https://arxiv.org/abs/2408.16760
- Street Gaussians: https://arxiv.org/abs/2403.11127
- EmerNeRF: https://arxiv.org/abs/2311.02077
- HUGS: https://arxiv.org/abs/2311.14590
- DrivingGaussian: https://arxiv.org/abs/2406.02570

**Feed-forward generation**:
- pixelSplat: https://arxiv.org/abs/2312.12337
- MVSplat: https://arxiv.org/abs/2403.14624
- GS-LRM: https://arxiv.org/abs/2404.19494
- L4GM: https://arxiv.org/abs/2402.14574
- Gen3C: https://arxiv.org/abs/2410.15277
- SCube: https://arxiv.org/abs/2410.20030

**Foundational**:
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14722
- NeRF: https://arxiv.org/abs/2003.08934
- Rectified Flow: https://arxiv.org/abs/2209.14591
- SD3: https://arxiv.org/abs/2403.03206
- OpenSora: https://github.com/hpcaitech/Open-Sora
- ControlNet: https://arxiv.org/abs/2302.05543
- SegFormer: https://arxiv.org/abs/2105.15203
- Metric3D v2: https://arxiv.org/abs/2407.15287
- LPIPS: https://arxiv.org/abs/1801.03904
- nuScenes: https://arxiv.org/abs/1903.11027
- BEVFormer: https://arxiv.org/abs/2203.17270
- StreamPETR: https://arxiv.org/abs/2310.15157

---

## 最后说一句

这篇 paper 的 elegance 在于它把"generation 理解世界"和"reconstruction 重建世界"用一个 latent space 统一起来。你以前讲过 "you are what you predict"——WorldSplat 的意思是 "if you want model to understand 4D world, make it predict 4D world"。

这是把 explicit 3D structure 烧进 generative model 的一篇代表作，未来这个方向大概率会看到更多变种——occupancy-aware diffusion、neural field-aware diffusion、甚至 physics-aware diffusion。Closed-loop driving simulation 的 holy grail 可能就在这一脉里。

---

# WorldSplat 深度技术解析

Andrej, 这篇 paper 来自 Xiaomi EV 团队（联合南开大学、南京大学苏州校区），2025 年 9 月底放出来，核心 idea 是把 **generative video diffusion** 和 **explicit 4D Gaussian Splatting** 直接耦合在一个 feed-forward pipeline 里，跳过了以往 "video generation → reconstruction" 的两阶段串联。项目主页在 https://wm-research.github.io/worldsplat/ ，arXiv 应该在 2509 月底会挂出来（从 reference 看 arXiv ID 尚未确定）。

---

## 1. Problem Motivation: Generation 与 Reconstruction 的 dilemma

Driving world model 这个赛道目前分两派，paper 在 Section 1 把 dilemma 讲得很明确：

- **Generation 派**（GAIA-1 https://arxiv.org/abs/2309.17080, DriveDreamer https://arxiv.org/abs/2309.09777, Vista https://arxiv.org/abs/2405.17398, MagicDrive https://arxiv.org/abs/2310.02601, Panacea https://arxiv.org/abs/2312.01462, DriveDreamer-2 https://arxiv.org/abs/2403.06845, MagicDrive-V2 https://arxiv.org/abs/2411.13807）在 2D pixel domain 做 video diffusion，FVD/FID 很漂亮，但**完全没有几何 anchor**。换视角立刻漂移，stochastic latent 把 scene identity 搅散。
- **Reconstruction 派**（EmerNeRF https://arxiv.org/abs/2311.02077, Street Gaussians https://arxiv.org/abs/2403.11127, OmniRe https://arxiv.org/abs/2408.16760, HUGS https://arxiv.org/abs/2403.12750, DrivingGaussian https://arxiv.org/abs/2406.02570）在 per-scene optimization 下能拿到 metric-consistent 4D scene，但**无法 hallucinate 未观测的场景**。

早期 bridge 工作如 MagicDrive3D https://arxiv.org/abs/2405.14475、InfiniCube https://arxiv.org/abs/2412.03934、DreamDrive https://arxiv.org/abs/2501.00601 都是 "先 generate video，再 reconstruct 4D"。问题在于**第二阶段永远受限于第一阶段的质量**：video 自身 perspective jitter、object identity drift 会传播进 Gaussian 里，sparse view 区域会 collapse。

WorldSplat 的核心 hypothesis 是：**把 4D structure 作为 diffusion 的 output 之一，而不是事后 recover**。让 diffusion 本身在 latent space 就产出 pixel-aligned 4D Gaussians，于是 geometry 一致性变成 diffusion 优化目标的一部分，从 root 解决 drift。

---

## 2. 整体框架三模块串讲

Figure 2 的 pipeline 是三段式：

```
Conditions C = {BEV sketch s, 3D boxes B, ego trajectory T, captions D}
      ↓
[Module 1] 4D-aware Latent Diffusion Model  →  multi-modal latent L = {L_img, L_depth, L_seg}
      ↓
[Module 2] Latent 4D Gaussian Decoder  →  per-frame pixel-aligned 3D Gaussians {(G_t, M_t)}
      ↓
aggregate to unified 4D Gaussians G_4D (static union, dynamic per-frame)
      ↓
Gaussian Splatting render to novel trajectory τ'  →  R' (rendered novel-view video)
      ↓
[Module 3] Enhanced Diffusion Model  →  final high-fidelity novel-view videos
```

三个模块独立训练，inference 时串起来。这种 decoupled 设计好处是**每个模块可以单独 swap / finetune**，Gaussian decoder 不依赖 diffusion，reconstruction-only mode 下可以直接喂 clean latent（Section 3.5 末尾提到）。

---

## 3. Module 1: 4D-Aware Latent Diffusion Model

### 3.1 Multi-modal latent 设计

这是我最感兴趣的设计点。传统 latent diffusion 只有 RGB latent `L_img = E(T)`（VAE encode 后的 K-view × T frames）。WorldSplat 强行把 **depth latent 和 semantic mask latent 拼进同一个 latent tensor**：

```
L = concat{L_img, L_depth, L_seg}
```

具体细节：
- `L_img`: OpenSora VAE v1.2（https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2）encode 多视角视频
- `L_depth`: 用 Metric3D v2（https://arxiv.org/abs/2407.15287）抽 **metric depth**（不是 relative depth！这点很关键，metric depth 才能支持 ego trajectory perturbation 后的几何投影），归一化到 [-1,1]，replicate 到 3 通道再 VAE encode
- `L_seg`: SegFormer（https://arxiv.org/abs/2105.15203）输出 dynamic object binary mask，encode 成 latent

**Intuition**：把 depth 和 seg 塞进 latent 等于给 diffusion 一个"几何先验通道"，diffusion 在 denoise 时必须同时 hallucinate appearance、geometry、semantic layout 三件事。这三件事在 latent space 是 channel-wise aligned 的，所以 decoder 后面读 latent 时，每个 pixel 位置都自带 metric depth 监督。这就是 "4D-aware" 的真正含义——不是在 loss 上加 depth 监督，而是**让 diffusion 的 output space 本身就是 4D 的**。

### 3.2 Architecture: Dual-branch DiT + ControlNet

基于 OpenSora v1.2（https://arxiv.org/abs/2410.09881）改造，主结构是 Diffusion Transformer，双分支：

- **Main DiT stream**: 处理 spatiotemporal video latents L，shape 大致是 `V × T × C × H × W`（V=6 view, T=17 frames, C = 4+4+4 = 12 channels if 三个 latent 都是 4 通道）
- **ControlNet branch**: multi-block，注入 conditions C

关键 trick 是 **cross-view attention 替换 self-attention**：把 `B × V × T × H × W × C` reshape 成 `B × T × (V·H·W) × C`，让 V·H·W 成为 sequence length，同一时刻 6 个 camera 的所有 token 互相 attend。这是 MagicDrive 的做法（https://arxiv.org/abs/2310.02601）的直接延续，确保 6 个相机视角在 latent space 是 coupled 的。

Conditions 注入：
- BEV road sketch → VAE encode → sketch latent
- Text captions → T5 encoder（https://arxiv.org/abs/1910.10683）→ text embeddings
- 3D boxes → 3D conv 投影到 image plane embeddings
- Ego trajectory → 小 MLP
- 后三者通过 cross-attention 融合进 main stream

### 3.3 DataCrafter: Fine-grained Caption 生成

Section 3.2 里提到的 DataCrafter 是个挺有意思的子模块。paper 描述比较简略：把 K-view 视频切成 clips，用 VLM evaluator（实际是 Qwen2-VL，https://arxiv.org/abs/2409.12191）打分 + 生成 per-view captions，然后用 consistency module 融合。Captions 捕获 weather、time、layout 等 scene context，加上 object category、box、description 等 instance-level detail。

这块是工程层面的关键——driving 场景的 controllable generation 对 caption 质量非常敏感，单纯 "a driving scene on a highway" 撑不起 fine-grained 控制。这里能联想到 LLaVA-NeXT、ShareGPT4Video 这类 video captioning 工作的思路。

### 3.4 Rectified Flow 替代 IDDPM

这是从 OpenSora v1.2 起就有的趋势（实际上 Stable Diffusion 3 https://arxiv.org/abs/2403.03206 也是 Rectified Flow）。核心公式：

**Eq. 1**: Interpolated state
$$
z(s) = (1-s)\,\epsilon + s\,x
$$
- $s \in [0,1]$ 是连续 mixing parameter，$s=0$ 时纯噪声，$s=1$ 时是 clean latent
- $\epsilon \sim \mathcal{N}(0, I)$ 是 Gaussian noise
- $x \sim p_{\text{data}}$ 是 clean latent sample
- 这条路径是 noise 到 data 的**直线**（rectified 的来源），相比 DDPM 的曲线路径，ODE 数值积分步数可以大幅减少

**Eq. 2**: 训练目标
$$
\mathcal{L}(\psi) = \mathbb{E}_{x, \epsilon, s}\Big[\big\| g_\psi\big(z(s), s, \mathcal{C}\big) - (x - \epsilon) \big\|_2^2\Big]
$$
- $g_\psi$ 是要训练的 neural field（参数 $\psi$），输入是 noised latent $z(s)$、时间 $s$、conditions $\mathcal{C}$
- 学习目标是 velocity field $v = x - \epsilon$，即从 noise 指向 data 的常向量
- 用 $L_2$ 回归，比 IDDPM 的 $\epsilon$-prediction 更 stable

**Eq. 3**: 推理 backward Euler
$$
z(s_{k-1}) = z(s_k) - \frac{1}{N} \cdot g_\psi\big(z(s_k), s_k, \mathcal{C}\big)
$$
- $s_k = k/N$ for $k = N, \ldots, 1$，$N$ 是总步数
- WorldSplat 在 inference 只用 **8 步**（看 Table 5），而 MagicDrive-V2 要 30+ 步
- 因为 rectified flow 路径近乎直线，每步误差积累小，8 步就够了

**Intuition**：Rectified Flow 相比 IDDPM，本质是把"沿着 curved probability path 走很多小步"换成"沿着 straight line 走几步"。WorldSplat 选这个的原因是 4D Gaussian decoder 下游需要 latent 是 clean 的，diffusion step 越少 inference latency 越低（Table 5 显示 total 2.5 min for 17-frame 6-view 424×800，比 MagicDrive-V2 的 3.85 min 快 35%）。

---

## 4. Module 2: Latent 4D Gaussian Decoder

这是整篇 paper 最有 contribution 的部分，Section 3.3 是核心。

### 4.1 Pixel-Aligned 3D Gaussian 参数化

每个 pixel 对应一个 3D Gaussian $\mathbf{g} = (\boldsymbol{\mu}, \mathbf{r}, \mathbf{s}, \alpha, \mathbf{c})$：
- $\boldsymbol{\mu} \in \mathbb{R}^3$：Gaussian center 在 world coordinate
- $\mathbf{r} \in \mathbb{R}^4$：quaternion rotation（4 个数表示 3D 旋转，比 rotation matrix 紧凑）
- $\mathbf{s} \in \mathbb{R}^3$：scale，控制 Gaussian 沿三个轴的延展
- $\alpha \in \mathbb{R}^+$：opacity，alpha compositing 用
- $\mathbf{c} \in \mathbb{R}^3$：color（这里是 RGB，没有 view-dependent SH，简化了）

Decoder 最后一层输出：
- offset $\boldsymbol{\delta} \in \mathbb{R}^3$
- rotation $\mathbf{r}$
- scale $\mathbf{s}$
- opacity $\alpha$
- color $\mathbf{c}$
- depth $d$（scalar per pixel）
- logits $\mathbf{m}$ for static/dynamic classification

**关键公式**（paper Section 3.3）：
$$
\boldsymbol{\mu} = \mathbf{R}_o + d \odot \mathbf{R}_d + \boldsymbol{\delta}
$$
- $\mathbf{R}_o \in \mathbb{R}^3$：camera ray origin（per-pixel，来自 Plücker map）
- $\mathbf{R}_d \in \mathbb{R}^3$：camera ray direction（per-pixel，normalized）
- $d$：predicted per-pixel depth
- $\boldsymbol{\delta}$：learned offset，residual 修正
- $\odot$ 是 element-wise product

这个 parameterization 是从 pixelSplat（https://arxiv.org/abs/2312.12337）和 MVSplat（https://arxiv.org/abs/2403.14624）传承来的，**核心 trick 是把 Gaussian center 通过 camera ray 参数化**，而不是直接预测 $(x,y,z)$。好处是：
1. 训练时 gradient 自然沿 ray 方向流动
2. 不同 view 之间 Gaussian 在 world frame 是 aligned 的
3. depth 监督可以直接接 $d$，不需要再 decode $\mu$

### 4.2 Plücker Ray Map

paper 提到 Plücker（1865, https://www.jstor.org/stable/108930）的 ray map $\mathbf{P}$，编码 per-pixel 的 $\mathbf{R}_o, \mathbf{R}_d$。Plücker coordinate 在 3D 几何里用 6 个数 $(d_x, d_y, d_z, m_x, m_y, m_z)$ 表示一条线，其中 $\mathbf{d}$ 是方向、$\mathbf{m} = \mathbf{p} \times \mathbf{d}$ 是 moment（$\mathbf{p}$ 是线上任一点）。这里简化成只输入 origin + direction。

**Intuition**：把 camera intrinsics + extrinsics 显式 encode 进 decoder input，让 transformer 知道每个 pixel 对应的世界射线。这样 ego trajectory perturbation 后（±1m, ±2m, ±4m shift），新视角的 Plücker map 直接换算，decoder 在同一 latent 上读出新的 ray，Gaussian splatting 投影时就有 geometric grounding。

### 4.3 Decoder Architecture

Transformer-based，多个 cross-view attention block + temporal attention layer + hierarchy of upsampling block。paper 强调支持 **48+ simultaneous input views**，远超 pixelSplat 的 2-view、MVSplat 的 2-3 view、GS-LRM（https://arxiv.org/abs/2404.19494）的 4-8 view。

输入：
$$
(\mathbf{L}_{\text{img}}, \mathbf{L}_{\text{depth}}, \mathbf{L}_{\text{seg}}, \mathbf{P}) \mapsto \{(\mathbf{G}_t, \mathbf{M}_t) \in \mathbb{R}^{V \times H \times W \times (14, 1)}\}_{t=1}^{T}
$$
- 每个 pixel 输出 14 维 Gaussian（3 center + 4 rotation + 3 scale + 1 opacity + 3 color = 14）和 1 维 mask logit
- $V=6$ view, $H, W$ 是 latent 上采样后的 spatial size

### 4.4 4D Gaussians Aggregation

这是把 per-frame 3D Gaussian 变成 4D 的关键步骤。给定已知 ego trajectory $\tau$，所有 frame 的 3D Gaussian 通过 ego-coordinate transformation 转到统一坐标系。然后 Eq. 5：

$$
\mathcal{G}_{4D} = \Big\{ \big(\mathbf{G}_t \odot \mathbf{M}_t\big) \cup \bigcup_{i=1}^{T}\big(\mathbf{G}_i \odot (1-\mathbf{M}_i)\big) \Big\}_{t=1}^{T}
$$

仔细拆解：
- $\mathbf{M}_t$：第 $t$ 帧的 dynamic mask（1 表示 dynamic，0 表示 static）
- $\mathbf{G}_t \odot \mathbf{M}_t$：第 $t$ 帧的 **dynamic** Gaussians（只取 dynamic pixel 的 Gaussians）
- $\mathbf{G}_i \odot (1-\mathbf{M}_i)$：第 $i$ 帧的 **static** Gaussians（只取 static pixel）
- $\bigcup_{i=1}^{T}$：所有帧的 static Gaussians **并集**（这是 4D 的核心——static 部分是从所有帧观测累积的）
- 最外层 $\{\}_{t=1}^{T}$：每个时刻 $t$ 的 4D 表示 = (本帧 dynamic) ∪ (所有帧 static union)

**Intuition**：static 场景（路、建筑、天空）在所有帧都应该在 world 坐标系重合，所以 union 起来增加 coverage；dynamic 物体（车、人）每帧位置不同，必须用本帧的 Gaussian。这是 emergent 4D reconstruction 的简化版，比 EmerNeRF 的 self-supervised decomposition 简单粗暴但有效。

### 4.5 Loss 函数

Eq. 6 给出总训练目标：
$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{lpips}} + \lambda_2 \mathcal{L}_{\text{depth}} + \lambda_3 \mathcal{L}_{\text{seg}}
$$

- $\mathcal{L}_{\text{recon}}$：photometric $L_1$ loss，渲染 RGB vs GT RGB
- $\mathcal{L}_{\text{lpips}}$：LPIPS perceptual loss（https://arxiv.org/abs/1801.03904），用 VGG features 衡量感知距离
- $\mathcal{L}_{\text{depth}}$：metric depth $L_1$ loss，在 metric space 监督
- $\mathcal{L}_{\text{seg}}$：binary cross-entropy，static vs dynamic mask 监督

训练时随机抽 base timestep $t$，再采 $T$ 个 target timesteps $\{t_i\}$，把 4D Gaussians 投影到这些 target timesteps 渲染 RGB + depth 监督。这种**随机时间投影**让模型学会从单帧或几帧 latent 推出整个 4D scene。

---

## 5. Module 3: Enhanced Diffusion Model

### 5.1 为什么需要 refine

Section 3.4 提到 Gaussian splatting 的固有缺陷：
1. **Unobserved regions**：novel trajectory shift 后，原本被车或建筑遮挡的区域没 Gaussian，渲染出来是空洞（Figure 3 上图左侧天空区域被砍掉）
2. **Strong ego motion blur**：没有 per-scene optimization 时，fast motion 区域 Gaussian 重叠不精确，渲染模糊

ReconDreamer（https://arxiv.org/abs/2411.19548）的思路是用 degraded rendering 训练，让 diffusion 学会 restore。WorldSplat 改进成 **mixed-conditioning**：训练时一部分 sample 用 degraded rendering，一部分用 high-quality rendering，避免 condition-output alignment 过度弱化。

### 5.2 架构

跟 4D-aware Diffusion Model 同架构（双分支 DiT + ControlNet），只是 condition 换成：
$$
\mathcal{C}' = \{\mathcal{R}', \mathcal{S}', \mathcal{B}', \mathcal{T}', \mathcal{D}\}
$$
- $\mathcal{R}'$：4D Gaussian rendered novel-view video
- $\mathcal{S}', \mathcal{B}'$：sketch 和 box 根据 new trajectory $\mathcal{T}'$ 重新投影
- $\mathcal{D}$：原 captions

Regression target 是 $\mathcal{E}(\mathcal{T})$，即 GT video 的 VAE latent。在 latent space 优化，跟 latent diffusion 一致。

---

## 6. Inference Pipeline 和 Trajectory Perturbation

Section 3.5 给出完整 inference 流程，最关键的 **customized trajectory selection**：

给定原 ego trajectory $\{\mathcal{T}_i\}_{i=1}^{N}$，沿 y-axis 加 offset：
$$
\Delta y \in \{+1\text{m}, -1\text{m}, +2\text{m}, -2\text{m}, +4\text{m}, -4\text{m}\}
$$
生成 6 条 perturbed trajectories $\{\mathcal{T}_i + (0, \Delta y, 0)\}_{i=1}^{N}$。

这个思路直接来自 FreeVS（https://arxiv.org/abs/2410.18079），FreeVS 用 generative view synthesis 在 free trajectory 上做渲染。WorldSplat 用 4D Gaussian 替代了 FreeVS 的 latent rendering，几何一致性更强。

**Intuition**：driving 场景里 ego car 的 lateral shift 是最直观的 "novel view"——对应"如果车在隔壁车道开，会看到什么"。±4m 已经覆盖到 2 个车道外，是相当激进的 view extrapolation。

---

## 7. Experiments 深度分析

### 7.1 Dataset 和 Metrics

nuScenes（https://arxiv.org/abs/1903.11027），1000 scenes，2Hz 标注，upsample 到 12Hz（用插值）。700 train / 150 val。

Metrics：
- **FVD**（Fréchet Video Distance, https://arxiv.org/abs/1812.01717）：衡量 generated video distribution 和 real video distribution 的距离，基于 Inflated 3D ConvNet features
- **FID**（Fréchet Inception Distance, https://arxiv.org/abs/1706.08500）：单帧 image distribution 距离，基于 Inception-V3 features

### 7.2 Table 1 解读: Original View Video Generation

| Setting | Method | FVD_multi ↓ | FID_multi ↓ |
|---|---|---|---|
| w/o first cond | DriveDreamer-2 | 105.10 | 25.00 |
| w/o first cond | MagicDrive-V2 | 94.84 | 20.91 |
| w/o first cond | MagicDrive3D | 164.72 | 20.67 |
| w/o first cond | Panacea | 139.00 | 16.96 |
| **w/o first cond** | **Ours** | **74.13** | **8.78** |
| w first cond | CoGen | 68.43 | 10.15 |
| w first cond | DriveDreamer-2 | 55.70 | 11.20 |
| **w first cond** | **Ours** | **16.57** | **4.14** |
| w noisy latent | Vista* | 112.65 | 13.97 |
| w noisy latent | UniScene | 70.52 | 6.12 |
| **w noisy latent** | **Ours** | **60.84** | **6.51** |

关键观察：
1. **w/o first cond**：WorldSplat 74.13 FVD vs MagicDrive-V2 94.84，**降幅 21.8%**。这个 setting 最能体现纯 generation 能力，因为没有 first frame hint。
2. **w first cond**：16.57 FVD 是 SOTA，比 DriveDreamer-2 的 55.70 低 3 倍多。这说明 4D-aware diffusion 在有 anchor frame 的情况下，能稳定 extrapolate 长序列。
3. **w noisy latent**：60.84 FVD，比 UniScene 的 70.52 低，但 FID 6.51 略高于 UniScene 的 6.12。说明 frame-level quality 略差，但 video-level temporal consistency 更好。这个 trade-off 是 4D structure constraint 带来的——geometry 一致性强迫 video 不能太"花哨"。

### 7.3 Table 2 解读: Novel View Synthesis

最关键的对比，shift ±1m / ±2m / ±4m：

| Method | FID ±1m | FVD ±1m | FID ±2m | FVD ±2m | FID ±4m | FVD ±4m |
|---|---|---|---|---|---|---|
| PVG | 48.15 | 246.74 | 60.44 | 356.23 | 84.50 | 501.16 |
| EmerNeRF | 37.57 | 171.47 | 52.03 | 294.55 | 76.11 | 497.85 |
| StreetGaussian | 32.12 | 153.45 | 43.24 | 256.91 | 67.44 | 429.98 |
| OmniRe | 31.48 | 152.01 | 43.31 | 254.52 | 67.36 | 428.20 |
| FreeVS* | 51.26 | 431.99 | 62.04 | 497.37 | 77.14 | 556.14 |
| DiST-4D | 10.12 | 45.14 | 12.97 | 68.80 | 17.57 | 105.29 |
| **Ours** | **8.25** | **40.17** | **11.26** | **47.41** | **13.38** | **64.07** |

观察：
1. **±4m shift**：WorldSplat 64.07 FVD vs DiST-4D 105.29，**降幅 39.1%**。Shift 越大优势越明显，说明 4D Gaussian aggregation 在大 baseline extrapolation 时优势突出。
2. **vs OmniRe**（per-scene optimization SOTA）：WorldSplat 在 ±4m 上 FID 13.38 vs OmniRe 67.36，**5 倍提升**。这是 feed-forward 模型超越 per-scene optimization 的强证据。
3. FreeVS 表现差（FVD 431.99 at ±1m）是因为它用 2D video generation 做 view synthesis，没有 explicit 3D structure，shift 后直接 drift。

### 7.4 Table 3 Ablation 解读

±2m shift 下的 ablation：

| Version | C-Reproj | 3D Gs | 4D Gs | Mixed Aug | Enhanced | FVD ↓ | FID ↓ |
|---|---|---|---|---|---|---|---|
| A | ✓ | | | | ✓ | 260.07 | 41.40 |
| B | ✓ | ✓ | | | ✓ | 75.26 | 16.31 |
| C | ✓ | | ✓ | | ✓ | 50.73 | 11.60 |
| D | ✓ | | ✓ | ✓ | | 107.58 | 26.73 |
| E | ✓ | | ✓ | | ✓ | **47.41** | **11.26** |

拆解：
- **A → B**（加单帧 3D Gaussian）：FVD 260 → 75，**3.5 倍提升**。3D structure 是 game changer。
- **B → C**（3D → 4D Gaussian aggregation）：FVD 75 → 50.7。Multi-frame aggregation 在 ±2m shift 下明显有用，因为更多帧 = 更多 Gaussian coverage。
- **C → D**（去 Enhanced Diffusion）：FVD 50.7 → 107.6。Enhanced model 贡献 FVD 一半。
- **D → E**（用 Mixed Aug 替代 Enhanced）：FVD 107.6 → 47.4。说明 Enhanced 比 Mixed Aug 重要，但 Mixed Aug 在训练阶段也用得到（看 Version E 是两者都有）。等等，仔细看 Version D 和 E 表头是反的——D 是 "Mixed Aug ✓, Enhanced 空"，E 是 "Mixed Aug 空, Enhanced ✓"。所以 Enhanced > Mixed Aug。但 paper Section 3.4 说用 mixed-conditioning strategy，这里有点 inconsistent，可能是 paper 笔误。

### 7.5 Table 4 Downstream Evaluation

下游任务验证 generated data 的 domain gap：

**(a) BEVFormer 在 generated data 上的 zero-shot performance**：
- DiVE: 35.96 mIoU, 24.55 mAP
- **WorldSplat: 38.49 mIoU, 29.34 mAP**
- 提升 +2.53 mIoU, +4.79 mAP

**(b) StreamPETR 训练，加 generated data**：
- Real only: 34.5 mAP, 46.9 NDS
- Real + Panacea: 37.1 mAP (+2.6), 49.2 NDS (+2.3)
- **Real + WorldSplat: 38.5 mAP (+4.0), 50.1 NDS (+3.2)**

数据增广效果超过 Panacea，说明 generated video 的 realism 和 geometry consistency 都足够好，能 transfer 到 perception model。

### 7.6 Table 5 Inference Speed

17-frame 6-view 424×800 video：
- MagicDrive-V2: 215s diff + 15s VAE = 230s = 3.85 min, 26 GB GPU
- Cosmos-transfer1: 126s diff + 4s VAE = 130s = 2.18 min, 17 GB
- **WorldSplat: 66.6s diff-1 + 0.84s Gs dec + 66.4s diff-2 + 16s VAE = 150s = 2.50 min, 22 GB**

虽然有两个 diffusion，但 rectified flow 8 步 vs 30 步，total latency 反而比 MagicDrive-V2 快 35%。这是 WorldSplat 工程层面的一大胜利。

---

## 8. 训练 4-stage Pipeline

Appendix A.2 给的细节：
- **Stage 1**: OpenSora v1.2 checkpoint 起，256×256 fixed resolution，60k iter。Optimize ControlNet-Transformer + spatial attention + layout module。
- **Stage 2**: mixed resolution (144p / 240p / 360p) + varying frame length，40k iter，对齐 nuScenes 分布。
- **Stage 3**: 把 IDDPM 换成 Rectified Flow，20k iter at low resolution。
- **Stage 4**: 480p → full scale finetune，60k iter，rectified flow。

总训练：180k iter × 4 stage = 大约 180k+40k+20k+60k = 200k iter，32 张 H20 GPU。这个规模在 driving generation 领域算 standard scale，比 Vista 的 4096×2160 训练小，比 MagicDrive-V2 类似。

---

## 9. 个人 Intuition 和 Critique

### 9.1 为什么这个 design work

WorldSplat 的核心 insight 是 **diffusion 的 output space 决定了它"理解"什么**。传统 video diffusion 输出 RGB pixels，模型只学 2D appearance statistics；WorldSplat 输出 4D-aware latent（含 depth + seg），模型被迫学 3D geometry + dynamic object decomposition。这是把 3D prior 烧进 generative prior 的 efficient way。

类似思路：
- DiST-4D（https://arxiv.org/abs/2503.15208）：disentangled spatiotemporal diffusion with metric depth
- Gen3C（https://arxiv.org/abs/2412.02917）：3D-informed world-consistent video generation
- SCube（https://arxiv.org/abs/2410.20030）：instant large-scale scene reconstruction with VoxSplat

这一脉工作都在探索 "geometry-aware generation"。

### 9.2 局限性

paper 没明显讨论的：
1. **Dynamic object 跨帧 identity**：4D aggregation 用 per-frame dynamic Gaussian，但同一辆车在 t=1 和 t=5 的 Gaussian 没有 explicit correspondence。如果车移动很快，Gaussian 之间会 disconnect，rendering 时可能 "tear"。
2. **天空和远景**：Gaussian splatting 对 distant element（天空、远山）建模差，需要很多大 Gaussian 才能覆盖。Paper 用 Enhanced Diffusion 补，但 root cause 没解决。
3. **Long video**：17 frames × 12 Hz ≈ 1.4s。对 long-horizon driving simulation 不够。MagicDrive-V2 已经做到几十秒长视频。
4. **Lighting/relighting**：Gaussian color 是 view-independent RGB，没有 SH，无法做 relighting。这对 data augmentation for perception 没问题，对 sensor simulation 不够。
5. **Ego trajectory perturbation 假设 static world**：但现实里其他 dynamic car 的 trajectory 也会随 ego shift 变化（其他车会反应）。WorldSplat 假设 dynamic object 跟原 trajectory 走，这是个 simplification。

### 9.3 和其他 generative 4D 工作的关系

- **DreamDrive**（https://arxiv.org/abs/2501.00601）：也是 4D driving generation，但用 video-first + reconstruction 两阶段。WorldSplat 直接 feed-forward 4D Gaussian，跳过 reconstruction stage。
- **UniScene**（https://arxiv.org/abs/2412.05435）：occupancy-centric，输出 voxel 而不是 Gaussian。Voxel 对 perception task 友好但对 NVS 不如 Gaussian 灵活。
- **Cosmos Transfer1**（https://arxiv.org/abs/2503.14492）：NVIDIA 的 conditional world generation，用 multi-modal control。Table 5 显示 WorldSplat 速度更快（2.5 min vs 2.18 min）但 quality 未直接对比。

### 9.4 联想到的相关工作

可以想到的延伸：
- **4D Gaussian Splatting**（https://arxiv.org/abs/2310.08528, https://arxiv.org/abs/2403.11154）：原始 4D GS 用 polynomial deformation field。WorldSplat 不用 deformation，直接 per-frame Gaussian + static union。更简单但失去 temporal continuity within dynamic object。
- **3DGS-based SLAM**（SplaTAM, MonoGS）：real-time 4D reconstruction。WorldSplat 是 feed-forward 版本，可以看作"learning-based 替代 per-scene SLAM"。
- **Sora-style video models**（OpenAI Sora, https://openai.com/sora）：spatiotemporal patches + DiT。WorldSplat 基于 OpenSora（Sora 的开源复现），技术栈一脉相承。
- **Gen3C**（https://arxiv.org/abs/2410.15277）：3D-informed world-consistent video generation，precise camera control。和 WorldSplat 思路接近，但 Gen3C 是 general scene，WorldSplat 专注 driving。
- **L4GM**（https://arxiv.org/abs/2402.14574）：4D Gaussian生成 from multi-view video。和 WorldSplat 互补。
- **Vista**（https://arxiv.org/abs/2405.17398）：generalizable driving world model，FVD 在 noisy latent setting 112.65 vs WorldSplat 60.84，差一倍多。说明 WorldSplat 的 4D 结构在纯 generative setting 下也有优势。

---

## 10. 参考链接汇总

**Core paper**:
- WorldSplat 项目主页: https://wm-research.github.io/worldsplat/

**Driving World Models**:
- GAIA-1: https://arxiv.org/abs/2309.17080
- DriveDreamer: https://arxiv.org/abs/2309.09777
- DriveDreamer-2: https://arxiv.org/abs/2403.06845
- Vista: https://arxiv.org/abs/2405.17398
- MagicDrive: https://arxiv.org/abs/2310.02601
- MagicDrive-V2: https://arxiv.org/abs/2411.13807
- MagicDrive3D: https://arxiv.org/abs/2405.14475
- Panacea: https://arxiv.org/abs/2403.04805 (CVPR 2024)
- DiVE: https://arxiv.org/abs/2409.01595
- UniScene: https://arxiv.org/abs/2412.05435
- DiST-4D: https://arxiv.org/abs/2503.15208
- DreamDrive: https://arxiv.org/abs/2501.00601
- InfiniCube: https://arxiv.org/abs/2412.03934
- ReconDreamer: https://arxiv.org/abs/2411.19548
- FreeVS: https://arxiv.org/abs/2410.18079
- Cosmos Transfer1: https://arxiv.org/abs/2503.14492

**Urban Scene Reconstruction**:
- EmerNeRF: https://arxiv.org/abs/2311.02077
- Street Gaussians: https://arxiv.org/abs/2403.11127
- OmniRe: https://arxiv.org/abs/2408.16760
- HUGS: https://arxiv.org/abs/2311.14590
- DrivingGaussian: https://arxiv.org/abs/2406.02570
- PVG: https://arxiv.org/abs/2311.18561
- Desire-GS: https://arxiv.org/abs/2411.11921

**Feed-forward 3D/4D Generation**:
- pixelSplat: https://arxiv.org/abs/2312.12337
- MVSplat: https://arxiv.org/abs/2403.14624
- GS-LRM: https://arxiv.org/abs/2404.19494
- L4GM: https://arxiv.org/abs/2402.14574
- Gen3C: https://arxiv.org/abs/2410.15277
- SCube: https://arxiv.org/abs/2410.20030

**Foundational Models**:
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14722
- NeRF: https://arxiv.org/abs/2003.08934
- 4D Gaussian Splatting: https://arxiv.org/abs/2310.08528
- Rectified Flow: https://arxiv.org/abs/2209.14591 (以及 SD3: https://arxiv.org/abs/2403.03206)
- OpenSora: https://arxiv.org/abs/2410.09881, VAE: https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2
- ControlNet: https://arxiv.org/abs/2302.05543
- SegFormer: https://arxiv.org/abs/2105.15203
- T5: https://arxiv.org/abs/1910.10683
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- Metric3D v2: https://arxiv.org/abs/2407.15287
- LPIPS: https://arxiv.org/abs/1801.03904
- FVD: https://arxiv.org/abs/1812.01717
- FID: https://arxiv.org/abs/1706.08500
- nuScenes: https://arxiv.org/abs/1903.11027
- BEVFormer: https://arxiv.org/abs/2203.17270
- StreamPETR: https://arxiv.org/abs/2310.15157
- Plücker coordinates (original 1865): https://www.jstor.org/stable/108930

---

## 11. 最终 takeaways

如果你（Andrej）想 build intuition 的话，我会提炼三点：

1. **Latent space 设计决定 generation 能力边界**。把 depth 和 seg 烧进 latent，等于让 diffusion 在 4D space 学习，而不是 2D pixel space 事后 recover。这跟你在 "micrograd" / "makemore" 系列里讲的 "output space 决定 loss signal" 哲学一致。

2. **Pixel-aligned Gaussian via camera ray parameterization** 是 feed-forward 3D/4D reconstruction 的关键 trick。$\boldsymbol{\mu} = \mathbf{R}_o + d \odot \mathbf{R}_d + \boldsymbol{\delta}$ 这个公式比直接预测 $(x,y,z)$ 优秀得多，gradient signal 直接沿 ray flow。pixelSplat / MVSplat / GS-LRM 这一脉都用这个思路。

3. **两阶段 diffusion（generation + refinement）的 trade-off**。第二个 Enhanced Diffusion 是补 Gaussian splatting 固有缺陷（unobserved region + motion blur），但它也带来额外 latency。Table 5 显示 66.6s + 66.4s 两段 diff 加起来 132s，占总 latency 88%。如果能用更 robust 的 4D representation（比如 4D NeRF 或者 adaptive Gaussian densification）替代 Enhanced Diffusion，会更 elegant，但工程难度大幅上升。这是个 open question。

这篇 paper 的工作让我联想到 Tesla 的 world model 工作（你在 Twitter 提过几次）和 Wayve 的 GAIA-2。WorldSplat 把 explicit 4D structure 烧进 generative model，方向上和 Tesla 的 "occupancy + diffusion" hybrid 接近。下一步很可能看到把 WorldSplat-style 4D generation 直接接入 end-to-end planning loop（类似 UniAD https://arxiv.org/abs/2212.10156 或者 VAD https://arxiv.org/abs/2403.14377），做 closed-loop driving simulation。这是这个方向最有想象力的应用。
