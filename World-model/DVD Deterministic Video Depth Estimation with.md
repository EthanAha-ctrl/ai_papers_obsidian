---
source_pdf: DVD Deterministic Video Depth Estimation with.pdf
paper_sha256: cd01abed4af934f99b32faf5fc34c1c14b11b218746429d8076b44411c87e8e3
processed_at: '2026-08-18T07:13:33-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DVD 用人话怎么说

Andrej, 咱们抛开 academic 八股，用大白话聊一下这篇 paper 到底干了啥。

---

## 一句话版本

**把 video diffusion model 改造成一个 deterministic regressor——输入 RGB 视频直接 forward 一把出 depth，既绕过了 stochastic sampling 的 hallucination，又保留了 generative backbone 的世界先验，训练数据量只需要 SOTA 的 1/160。**

---

## 为什么要搞这个东西

当下 video depth estimation 两条路都走不通：

**Generative 派（DepthCrafter 这类）**：拿 video diffusion model（Wan、CogVideoX 这种）直接去 denoise 出 depth。听起来很美好——backbone 见过几千万小时视频，prior 超强。但 stochastic sampling 有个要命的毛病：每一帧采样都是随机的，时间一长 variance 累积，就出现 geometric hallucination——墙壁会呼吸、地面会起伏、物体边界会抖。长视频直接 scale drift 爆炸，没法用。

**Discriminative 派（VDA 这类）**：ViT 大力出奇迹，硬学 RGB→depth mapping。好处是 deterministic、快。坏处是需要海量标注数据（VDA 用了 60M frames），而且遇到 textureless 区域或者 motion blur，模型一脸懵——把模糊当边界，把均匀纹理当前景。semantic ambiguity 严重。

DVD 就想：**我能不能同时拿到两边的好处？** 用 generative backbone 的 prior 解决 ambiguity，用 deterministic regression 解决 hallucination。

---

## 它怎么做的——三个 trick

### Trick 1: 把 timestep 当成 "structural anchor"

Diffusion model 里有个 timestep $t \in [0,1]$，正常推理是从 $t=1$（纯噪声）逐步 denoise 到 $t=0$（清晰图）。

之前 Lotus（image 版的 deterministic adaptation）直接把 $t$ 固定在 1 或者干脆丢掉。DVD 发现这在 video 上不行——会过度平滑。原因是 **spectral bias**：

- 高 $t$（低 SNR）→ 网络学 low-frequency global structure（大块结构）
- 低 $t$（高 SNR）→ 网络学 high-frequency detail（边缘纹理）

DVD 的做法：**固定 $t = \tau_0 = 0.5$**，当成一个 "structural anchor" 送给网络。

具体怎么送进去？通过 sinusoidal embedding：

$$\mathbf{e}_{\sin}(t) = \big[\cos(\omega_1 t), \dots, \cos(\omega_{d/2} t), \sin(\omega_1 t), \dots, \sin(\omega_{d/2} t)\big]$$

- $t$：timestep scalar
- $\omega_i$：预定义的 angular frequencies
- $d$：embedding 维度
- 之后过一个 MLP $\mathbf{e}_\phi(\cdot)$ 注入 attention blocks

然后 deterministic mapping 长这样：

$$\hat{z}_d = \mathcal{F}_\theta\left(z_x; \mathbf{e}_\phi(\tau_0)\right)$$

- $z_x$：RGB latent（VAE encoder 出来的）
- $\tau_0 = 0.5$：固定的 structural anchor
- $\mathcal{F}_\theta$：frozen video DiT + LoRA

**为什么 $\tau=0.5$ 最好？** Table 7 的 ablation 特别有说服力。ScanNet 上：

| τ | AbsRel↓ |
|---|---|
| 0.0 (w/o) | 11.3 |
| 0.3 | 6.0 |
| **0.5** | **5.5** |
| 0.9 | 16.8 |
| 1.0 | 17.6 |
| learning τ | 16.3 |

极端低（0.0）或极端高（1.0）都崩。中间 0.5 是 sweet spot——既不过度偏向 global structure，也不过度偏向 local detail。

最炸裂的 ablation 是 "learning τ"——把 sinusoidal 换成同维度可学习 embedding，AbsRel 从 5.5 炸到 16.3！这说明 **$\tau$ 不是随便一个 condition，它和预训练时的 frequency pathway 是 entangled 的**。重新初始化就等于把几何先验这条路给断了。

这个 finding 让我想到 [Kim et al., 2024](https://arxiv.org/abs/2405.14126) 那篇"The disappearance of timestep embedding"——现代时间相关网络里 timestep embedding 扮演的角色比想象中微妙得多。

**Intuition 给你**：这就像 nanoGPT 里 positional encoding——你可以固定 sin/cos basis，但你绝对不能把它换成 random init 的 learnable embedding 重新训，预训练的 inductive bias 全绑死在这上面了。

---

### Trick 2: Latent Manifold Rectification (LMR)——治 mean collapse

这是 paper 最深刻的一个 insight。

**Mean collapse 是啥？** 用 L2 loss 做 regression，网络学的是 $\mathbb{E}[z_d | z_x]$——conditional expectation。当某个区域有 ambiguity（textureless、occlusion、motion blur），GT depth 在 latent manifold 上是 multi-modal 的，多个几何假设都讲得通。L2 会让网络输出所有 mode 的平均——高频细节被 wash out。

Video 上更惨：被压掉的高频 differential 在时间上累积，变成 boundary erosion + motion flickering。

**LMR 怎么治？** 加两个 parameter-free 的 supervision，在 latent space 强制 differential 对齐：

**Spatial Rectification（latent gradient）— Eq. 7：**

$$\mathcal{L}_{sp} = \frac{1}{F \cdot \Omega} \sum_{f=1}^{F} \sum_{\partial \in \{\nabla_h, \nabla_w\}} \|\partial \hat{z}_d^f - \partial z_d^f\|_1$$

- $F$：frame 数
- $\Omega$：spatial resolution
- $\nabla_h, \nabla_w$：行列方向的 finite difference
- $\hat{z}_d^f, z_d^f$：第 $f$ 帧预测和 GT 的 depth latent

强制预测的 latent gradient field 对齐 GT 的 gradient field。

**Temporal Rectification（latent flow）— Eq. 8：**

$$\mathcal{L}_{temp} = \frac{1}{(F-1) \cdot \Omega} \sum_{f=2}^{F} \|\nabla_t \hat{z}_d^f - \nabla_t z_d^f\|_1$$

其中 $\nabla_t z^f = z^f - z^{f-1}$（帧间差分）。

强制预测的 latent temporal flow 对齐 GT 的 motion dynamics。

**总 loss — Eq. 9：**

$$\mathcal{L}_{video} = \|\hat{z}_d - z_d\|_2 + \lambda_{sp} \mathcal{L}_{sp} + \lambda_{temp} \mathcal{L}_{temp}$$

$\lambda_{sp} = \lambda_{temp} = 0.5$（Table 5）。

**为什么 LMR 比传统 regularizer 都好？** Table 8 的对比：

| Regularizer | AbsRel↓ | B-F1↑ |
|---|---|---|
| L2 only | 8.5 | 0.210 |
| + RGB reconstruction | 10.5 | 0.174 |
| + Edge-aware smoothness | 7.5 | 0.193 |
| + Multi-scale gradient | 8.2 | 0.257 |
| **+ LMR** | **7.3** | **0.259** |

Edge-aware smoothness 提升全局 metric 但 destroy 边界（B-F1 掉到 0.193）。Multi-scale gradient 提升 B-F1 但全局 metric 改善有限。LMR **两个都同时拉满**。关键在 latent space 做——避开 pixel-space reconstruction 的 decoding artifact。

**Intuition 给你**：就像你训 next-token prediction 时，除了 cross-entropy loss 还加一个 "first-difference of logits" matching——强制模型在 token manifold 上的切向量对齐，保留 multi-modal 的微分结构而不是坍缩到 mean mode。这跟 [Papyan et al. 2020 Neural Collapse](https://www.pnas.org/doi/10.1073/pnas.2015500117) 和 [Song et al. 2020 Score SDE](https://arxiv.org/abs/2011.13456) 讲的是一个道理。

---

### Trick 3: Global Affine Coherence——治长视频 scale drift

长视频必须 sliding window inference。Generative model 在不同 window 独立 stochastic sampling，distortion 是 nonlinear 的，没法 align。

DVD 是 deterministic regressor，$\text{Var}[\hat{z}_d | z_x] = 0$。即便如此，windowed inference 还是有个 secondary bottleneck——**VAE decoder 的 context-dependent normalization 会让 depth value 波动**。

**关键 empirical finding**（Figure 6, Figure 12）：VAE decoding 引入的主要是 **global affine variation**（scale + shift），而不是 local spatial distortion。也就是说，两个 window 之间的 discrepancy 可以用线性变换 $s \cdot \mathbf{d} + t$ 近似。

这让我想到 Consistency Models 里那个 affine invariance 假设——其实是对 VAE 行为的一种 empirical characterization。

**Affine Alignment 公式：**

两个 window $\mathcal{W}_A, \mathcal{W}_B$，提取 overlap 区域的 flattened pixels $\mathbf{d}_A^{overlap}, \mathbf{d}_B^{overlap} \in \mathbb{R}^{\hat{N}}$。

**Least-squares objective — Eq. 10：**

$$\arg\min_{s,t} \|s \mathbf{d}_B^{overlap} + t \mathbf{1} - \mathbf{d}_A^{overlap}\|_2^2$$

**Closed-form solution — Eq. 11：**

$$s = \frac{\text{Cov}(\mathbf{d}_A^{overlap}, \mathbf{d}_B^{overlap})}{\text{Var}(\mathbf{d}_B^{overlap})}, \quad t = \mu_A - s \mu_B$$

- $s$：global scale
- $t$：global shift
- $\mu_A, \mu_B$：两 overlap 区域均值
- $\text{Cov}$：协方差
- $\text{Var}$：$\mathbf{d}_B^{overlap}$ 方差

广播到整个 window：$\hat{\mathcal{W}}_B = s \cdot \mathcal{W}_B + t$，overlap 区域 linear interpolation blend。

**为什么 generative model 没法做这个？** 它们不仅有 affine distortion，还有 nonlinear mode switching——每次 stochastic sample 跳到不同 mode，align 不回来。DVD 的 deterministic 保证把 distortion 限制在 VAE-induced 线性变换范围内，mathematically 可恢复。

Table 9 的 overlap ablation，**O=9 是 sweet spot**：

| Overlap O | AbsRel↓ | Rel. Time↓ |
|---|---|---|
| 3 | 7.9 | 1.00× |
| **9** | **7.3** | **1.17×** |
| 19 | 7.1 | 1.55× |

再大边际收益急剧递减，latency 成本暴涨。

---

## 训练 setup 小总结

- Backbone: **Wan2.1-1.3B** (frozen，只训 LoRA)
- LoRA: rank=512, target modules = $W_q, W_k, W_v, W_o, W_{ffn}$
- VAE: spatial 8× compression, temporal 4× compression (frozen)
- 数据：TartanAir + Virtual KITTI (video) + Hypersim + Virtual KITTI (image)
- Joint image-video training：batch video=16, image=128, image loss weight $\lambda_{image}=1.0$
- Optimizer: AdamW, lr=1e-4, constant schedule
- Hardware: 8× H100, **<36 小时收敛**
- Total training frames: **367K**（VDA 是 60M，是 DVD 的 163 倍）

Image-video joint training 的 rationale：纯 video 会 underfit 空间细节；sequential image→video 会 catastrophic forgetting；joint batch 让 image 当 high-frequency spatial anchor，video 当 temporal coherence supervisor。

---

## 实验结果有多炸

### 主结果 Table 1

| Method | Train Frames | KITTI AbsRel↓ | ScanNet AbsRel↓ | Bonn AbsRel↓ |
|---|---|---|---|---|
| DepthCrafter (generative) | ~30M | 9.9 | 7.1 | 5.9 |
| VDA (discriminative) | 60M | 7.2 | 5.8 | 4.7 |
| **DVD** | **367K** | **6.7** | **5.5** | **4.7** |

**用 1/160 的数据打赢了 SOTA**——这是 paper 最 sellable 的点。

### Long Video Table 2

| Method | Bonn AbsRel↓ | KITTI AbsRel↓ |
|---|---|---|
| DepthCrafter | 8.5 | 12.0 |
| VDA | 6.6 | 9.6 |
| **DVD** | **5.3** | **7.6** |

长视频上 DVD 的 margin 更大——affine alignment 真的发挥威力了。

### Boundary metrics Table 3

ScanNet B-F1：
- ChronoDepth: 0.204
- VDA: 0.210
- **DVD: 0.259**

KITTI B-Recall：
- VDA: 0.047
- **DVD: 0.217** ← **4.6 倍提升**

这是 LMR 直接见效的地方。

### Cross-backbone 验证 Table 6

在 CogVideoX-5B 上 deploy DVD，τ=0.5 依然最优——证明这套范式不是 Wan-specific 的 trick：

| τ | 0.1 | 0.3 | **0.5** | 0.7 | 0.9 |
|---|---|---|---|---|---|
| KITTI AbsRel↓ | 7.8 | 7.6 | **7.4** | 7.4 | 9.5 |

---

## 整体架构 Figure 2 文字版

```
Input RGB video x ∈ R^{F×3×H×W}
    │
    ▼ frozen VAE encoder E(·)
z_x ∈ R^{f×C×h×w}  (latent)
    │
    ▼
┌───────────────────────────────────────┐
│  Pre-trained Video DiT F_θ           │
│  (Wan2.1-1.3B, frozen)               │
│  + LoRA (r=512, on q/k/v/o/ffn)      │
│                                       │
│  Input: z_x + τ_0=0.5 (fixed anchor) │
│  Output: ẑ_d — SINGLE FORWARD PASS   │
│                                       │
│  Loss: L_video = L2 + 0.5·L_sp       │
│                  + 0.5·L_temp         │
└───────────────────────────────────────┘
    │
    ▼ frozen VAE decoder D(·)
d̂ ∈ R^{F×H×W}

Long video: sliding window (size=45, stride=9, overlap=9)
  Ŵ_B = s · W_B + t,  s,t via closed-form least-squares
```

---

## Limitations 也很诚实

1. **Extreme long video with scene transitions**：1100 帧 indoor desk → outdoor tunnel 这种，affine 假设失效，会有 global scale drift。但 VDA 同样甚至更严重——这是 field-level open challenge。
2. **Real-time deployment**：1.3B DiT 还是大，做到 ≥10 Hz on-device 还得靠 distillation。
3. **VAE 8× downsample**：限制 ultra-thin structure 恢复——指向 VAE-free tokenization 方向。

---

## 我自己的几个联想

### 1. 和 Consistency Models 的关系
DVD 的 single-pass regression 跟 [Consistency Models (Song et al., 2023)](https://arxiv.org/abs/2303.01469)、[LCM (Luo et al., 2023)](https://arxiv.org/abs/2310.04378) 思路相通但路径不同：
- Consistency Model 是 self-distillation，target 是自己 trajectory 上的点
- DVD 是直接监督到 GT depth latent（discriminative 范式）
- 某种意义上 DVD 更像 **flow matching with fixed anchor shortcut**——把整个 flow trajectory 用一个 anchor point 代替

### 2. Mean Collapse 和你熟悉的 phenomenon
这跟 neural collapse、distribution smoothing、token collapse 都是同构问题。训练 small dataset 上的 GPT，模型倾向于输出 generic token distribution——这就是一种 mean collapse。LMR 提供了一种思路：约束 differential，保留 manifold 上的局部切向量，强制保留 multi-modality。

### 3. 共享 VAE encoder 是 strong inductive bias
RGB 和 depth 共享 VAE encoder（Eq. 1）这设计很激进。Latent Diffusion 通常只 encode RGB。DVD 假设 depth 和 RGB 在 VAE latent 上能 align——这是 strong inductive bias，也是 Sintel（动画）上表现稍弱（44.5 vs VDA 39.7）的可能原因：动画的 VAE encoding 可能和真实 depth 有 distribution gap。

### 4. 从 Lotus 到 DVD 的演进路线
[Lotus (He et al., 2024)](https://arxiv.org/abs/2409.18124) 在 image 上做 deterministic adaptation，[Lotus-2](https://arxiv.org/abs/2512.01030) 改进 image 版本。DVD 把 paradigm 搬到 video，发现了三个 video-specific 问题：
1. Timestep 需要做 structural anchor（image 上不用）
2. Mean collapse 在 video 上被 temporal 累积放大
3. Long video 需要 affine alignment（image 上没这个问题）

这三个 finding 对后续做 video-to-video translation、video prediction 的 deterministic adaptation 都有借鉴价值。

### 5. 推广到 optical flow？
我特别好奇：这套范式能不能 extend 到 **optical flow / scene flow** 上？Video DiT 里编码的 motion prior 应该也能被 deterministic regression 解锁。这跟 [GeometryCrafter (Xu et al., 2025)](https://arxiv.org/abs/2504.07889) 的 point map 方向互补——一个押注 depth + affine alignment，一个押注 unbounded 3D point representation。

### 6. Wan2.1 作为 backbone 的选择
[Wan (Team et al., 2025)](https://arxiv.org/abs/2503.20314) 在开源 video DiT 里算 strong baseline，1.3B 这个 size 比较平衡——prior 够强，又能 8×H100 36h 训完。如果上 HunyuanVideo 或者更大 backbone，性能应该还能再上一台阶。这让我想到你以前讲过的 scaling law 直觉——deterministic adaptation 的 scaling 特性其实是个开放问题。

### 7. τ anchor 和预训练 inductive bias 的 entanglement
"learning τ" 这个 ablation（16.3 vs 5.5）我觉得是 paper 里最 underappreciated 的 finding。它说明：**预训练的 frequency basis 是 model 的 "DNA"**，你不能动它。这跟 [Variational Diffusion Models (Kingma et al., 2021)](https://arxiv.org/abs/2107.00608) 里 SNR schedule 和 frequency decomposition 的分析呼应——但更进一步，是关于 **conditioning pathway 的 entanglement**。

---

## 相关 references

- **DVD 项目主页**: https://dvd-project.github.io/
- **DVD Github**: https://github.com/EnVision-Research/DVD
- **Lotus (image 前作)**: https://arxiv.org/abs/2409.18124
- **Lotus-2**: https://arxiv.org/abs/2512.01030
- **Marigold**: https://arxiv.org/abs/2312.02112
- **DepthCrafter (CVPR 2025)**: https://arxiv.org/abs/2409.18120
- **ChronoDepth**: https://arxiv.org/abs/2403.07420
- **Video Depth Anything**: https://arxiv.org/abs/2501.10075
- **Wan2.1 (backbone)**: https://arxiv.org/abs/2503.20314
- **CogVideoX**: https://arxiv.org/abs/2408.06072
- **HunyuanVideo**: https://arxiv.org/abs/2412.03603
- **Rectified Flow**: https://arxiv.org/abs/2209.03003
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **Variational Diffusion (spectral analysis)**: https://arxiv.org/abs/2107.00608
- **Perception Prioritized Training**: https://arxiv.org/abs/2204.00227
- **Consistency Models**: https://arxiv.org/abs/2303.01469
- **Neural Collapse**: https://www.pnas.org/doi/10.1073/pnas.2015500117
- **Timestep embedding disappearance**: https://arxiv.org/abs/2405.14126
- **Hypersim**: https://github.com/apple/ml-hypersim
- **TartanAir**: https://theairlab.com/tartanair/
- **Virtual KITTI**: http://www.virtualkitchen.org/
- **GeometryCrafter**: https://arxiv.org/abs/2504.07889

---

## 最核心的 intuition

DVD 的 thesis 一句话：

> **Pre-trained video diffusion models 编码了 profound 的时空几何 prior，但这种 prior 被 stochastic sampling 的 aleatoric variance 掩盖了。一旦用 deterministic regression 把 prior 蒸馏出来，只需要 1% 的下游数据就能解锁。**

三大 design 分别治三个病：
- **τ anchor** → 治 spectral bias 导致的 frequency regime 错配
- **LMR** → 治 regression-to-mean 的高频 collapse
- **Affine coherence** → 治 long video 的 inter-window drift

这套范式我觉得是 **video foundation model 的 "deterministic unlock"** ——跟 image 域 Marigold→Lotus→DepthMaster 那条线对应，但 video 上挑战多一个数量级。后续如果有人把这套范式搬到 optical flow、normal estimation、point map 上，应该都能 work。

---

# DVD: Deterministic Video Depth Estimation with Generative Priors 深度解读

Andrej, 这篇 paper 我非常仔细地读了几遍，对你的工作来说应该会觉得相当有趣——它本质上是一个 **"distillation of generative video priors into a deterministic regressor"** 的故事，与你过去在 diffusion distillation、token regression 这些方向的直觉高度相关。下面我从 motivation、架构、公式、实验四个层面 build 一下 intuition，并且把我能联想到的相关工作都串起来。

---

## 1. 问题动机：为什么是 Video Depth，为什么是 Deterministic

### 1.1 现状的两条难走的路

Video depth estimation 当下被两条 paradigm 卡死：

**(I) Generative Diffusion** (DepthCrafter [Hu et al., CVPR 2025], ChronoDepth [Shao et al., CVPR 2025], RollingDepth [Ke et al., CVPR 2025a])
- 优点：站在 video diffusion foundation model (Wan, CogVideoX, HunyuanVideo) 的肩膀上，继承了海量时空先验，zero-shot 泛化强
- 致命问题：**stochastic geometric hallucination** 和 **scale drift**。每次 stochastic sampling 引入的 aleatoric variance 在时间维度上累积，长视频上直接爆炸

**(II) Discriminative ViT** (Video Depth Anything / VDA [Chen et al., 2025d], Depth Anything V2 [Yang et al., 2024c])
- 优点：deterministic, 高效, 单次 forward
- 致命问题：需要海量标注 (VDA 60M frames)，遇到 motion blur、textureless region 就 **semantic ambiguity**——把运动模糊当结构边界

DVD 的 insight 是：能否把 generative 的 video DiT prior **蒸馏**成一个 single-pass deterministic regressor？这正好是 Lotus (image 域) [He et al., 2024](https://arxiv.org/abs/2409.18124) 在图像上做的事情，DVD 把它扩展到 video。但 video 上有三个 unique challenge，对应 paper 的三个 core design。

---

## 2. 整体架构（Figure 2 解析）

```
┌──────────────────────────────────────────────────────────────────┐
│  Input: RGB video x ∈ R^{F×3×H×W}                                │
│      ↓ frozen VAE encoder E(·)                                   │
│  z_x ∈ R^{f×C×h×w}  (latent)                                     │
│      ↓                                                           │
│  ┌────────────────────────────────────────────────────┐          │
│  │  Pre-trained Video DiT F_θ (Wan2.1-1.3B)           │          │
│  │  + LoRA (r=512, on W_q/W_k/W_v/W_o/W_ffn)          │          │
│  │                                                    │          │
│  │  Input: z_x (RGB latent) + τ_0 (structural anchor)  │          │
│  │  Output: ẑ_d (depth latent) — SINGLE PASS          │          │
│  │  Loss: L_video = L2 + λ_sp·L_sp + λ_temp·L_temp     │          │
│  └────────────────────────────────────────────────────┘          │
│      ↓ frozen VAE decoder D(·)                                   │
│  d̂ ∈ R^{F×H×W}                                                   │
└──────────────────────────────────────────────────────────────────┘

Long video: sliding window with overlap O=9, affine align:
  Ŵ_B = s · W_B + t,  s,t via closed-form least-squares on overlap
```

关键点：
- VAE 是 frozen 的，同时编码 RGB 和 depth (Eq. 1)，共享 latent space
- DiT backbone 也是 frozen 的（保留 video generative prior），只训练 LoRA
- τ_0 是固定值（不采样！），相当于把 timestep "卡"住作为一个 structural code
- 推理是 single forward pass，没有 iterative ODE

---

## 3. 三大 Core Design 深入讲解

### 3.1 Timestep as Structural Anchor (§4.2)

#### Motivation
Lotus 这种 image deterministic adaptation 通常把 timestep 设成 t=1（terminal state）或者直接吸收掉。但 paper 在 Figure 3 上观察到，对 video backbone 这样做会 **geometric over-smoothing**。原因是 diffusion prior 的 **spectral bias**：

- 高 t (low SNR, early denoising) → 网络学的是 low-frequency global structure
- 低 t (high SNR, late denoising) → 网络学的是 high-frequency detail

这里引用的是 [Kingma et al., 2021 Variational Diffusion Models](https://arxiv.org/abs/2107.00608) 和 [Choi et al., 2022 Perception Prioritized Training](https://arxiv.org/abs/2204.00227) 的 spectrum 分析。

#### 公式拆解

**Sinusoidal embedding (Eq. 5):**

$$\mathbf{e}_{\sin}(t) = \big[\cos(\omega_1 t), \dots, \cos(\omega_{d/2} t), \sin(\omega_1 t), \dots, \sin(\omega_{d/2} t)\big]$$

变量说明：
- $t \in [0,1]$：timestep scalar
- $\omega_i$：predefined angular frequencies（通常 geometric series，类似 transformer position encoding）
- $d$：embedding dimension
- 输出 $\mathbf{e}_{\sin}(t) \in \mathbb{R}^d$，之后还会过一个 MLP projection $\mathbf{e}_\phi(\cdot)$ 注入到 attention blocks

**Deterministic mapping (Eq. 6):**

$$\hat{z}_d = \mathcal{F}_\theta\left(z_x; \mathbf{e}_\phi(\tau_0)\right)$$

- $z_x$：RGB latent
- $\tau_0$：固定的 conditioning state（实验最优是 0.5）
- $\mathcal{F}_\theta$：video DiT + LoRA

#### Fidelity-Stability Trade-off (Table 7, Figure 10)

在 ScanNet 和 KITTI 上扫 τ 的结果非常有意思：

| τ | ScanNet AbsRel↓ | ScanNet δ₁↑ | KITTI AbsRel↓ | KITTI δ₁↑ |
|---|---|---|---|---|
| w/o τ (=0.0) | 11.3 | 0.940 | 13.7 | 0.904 |
| τ=0.3 | 6.0 | 0.975 | 7.5 | 0.960 |
| **τ=0.5** | **5.5** | **0.974** | **6.7** | **0.967** |
| τ=0.7 | 6.5 | 0.970 | 8.4 | 0.941 |
| τ=0.9 | 16.8 | 0.941 | 23.0 | 0.630 |
| τ=1.0 | 17.6 | 0.769 | 22.7 | 0.619 |
| learning τ | 16.3 | 0.811 | 23.7 | 0.699 |

**最关键的一个 ablation 是 "learning τ"**：把 sinusoidal basis 换成同维度的可学习 embedding，性能崩盘（KITTI 13.7 → 23.7）。这证明 τ 不是普通的 condition，它是预训练时和 backbone 的 frequency pathway 紧密 entangled 的——一旦重新初始化，geometric prior 这条 pathway 就失效了。这点和 [Kim et al., 2024](https://arxiv.org/abs/2405.14126) "disappearance of timestep embedding" 的观察呼应。

**Intuition for Karpathy**：这就像你做 GPT 推理时把 positional encoding 当成固定的 condition token 来用，而不是从头学习一个新的。预训练的 inductive bias 是绑死在特定 frequency basis 上的。τ=0.5 在 embedding similarity 矩阵上是一个 "broad basin"——和很多其他 t 都 cosine 接近，给了一个低方差的 conditioning region。

---

### 3.2 Latent Manifold Rectification / LMR (§4.3)

#### Mean Collapse 的本质

这是我觉得这个 paper 最深刻的 observation。**Point-wise L2 regression 学到的是 conditional expectation $\mathbb{E}[z_d | z_x]$**。

直觉解释：当 $z_x$ 给定但有 ambiguity（比如 textureless 区域、motion blur），GT depth $z_d$ 在 latent manifold 上是 multi-modal 的（多个几何假设都解释得通）。L2 loss 会让网络预测所有 mode 的平均，结果就是高频细节被 wash out，这就是 mean collapse。

参考 [Song et al., 2020 Score-based SDE](https://arxiv.org/abs/2011.13456) 和 [Liu et al., 2022 Rectified Flow](https://arxiv.org/abs/2209.03003) 的分析——iterative sampling 是从一个 mode 跳到另一个 mode 的过程，deterministic regression 直接坍缩到 mean。

Video 上更严重：被抑制的高频 differential 在时间上累积，变成 **boundary erosion + motion flickering**。这跟你在 nanoGPT 里训 small dataset 容易 collapse 到 generic token distribution 是同构问题。

#### LMR 公式拆解

**Spatial Rectification (Latent Gradient) — Eq. 7:**

$$\mathcal{L}_{sp} = \frac{1}{F \cdot \Omega} \sum_{f=1}^{F} \sum_{\partial \in \{\nabla_h, \nabla_w\}} \|\partial \hat{z}_d^f - \partial z_d^f\|_1$$

变量：
- $F$：frame 数
- $\Omega$：spatial resolution（$h \times w$）
- $\nabla_h, \nabla_w$：finite difference operators（行/列方向梯度）
- $\hat{z}_d^f$：第 $f$ 帧预测的 depth latent
- $z_d^f$：第 $f$ 帧 GT depth latent

这个 loss 强制预测 latent 的 **空间 gradient field** 对齐 GT 的 gradient field。在 latent 空间做，不是 pixel space——这是关键。

**Temporal Rectification (Latent Flow) — Eq. 8:**

$$\mathcal{L}_{temp} = \frac{1}{(F-1) \cdot \Omega} \sum_{f=2}^{F} \|\nabla_t \hat{z}_d^f - \nabla_t z_d^f\|_1$$

其中 $\nabla_t z^f = z^f - z^{f-1}$（frame 间差分）。

这个 loss 在 latent 空间约束帧间 motion flow 的 differential。

**Overall Video Loss — Eq. 9:**

$$\mathcal{L}_{video} = \|\hat{z}_d - z_d\|_2 + \lambda_{sp} \mathcal{L}_{sp} + \lambda_{temp} \mathcal{L}_{temp}$$

$\lambda_{sp} = \lambda_{temp} = 0.5$ (Table 5)。

#### 为什么 LMR 比 Edge-aware Smoothness / Multi-scale Gradient 好 (Table 8)

| Regularizer | AbsRel↓ | δ₁↑ | B-F1↑ |
|---|---|---|---|
| L2 only | 8.5 | 0.966 | 0.210 |
| + RGB reconstruction | 10.5 | 0.951 | 0.174 |
| + Edge-aware smoothness | 7.5 | 0.978 | 0.193 |
| + Multi-scale gradient matching | 8.2 | 0.969 | 0.257 |
| **+ LMR (ours)** | **7.3** | **0.977** | **0.259** |

注意 edge-aware smoothness 提升全局 metric 但 destroy 边界（B-F1 0.210→0.193），而 multi-scale gradient 提升 B-F1 但全局 metric 改善有限。**LMR 同时改善两者**——因为 latent gradient 直接在 VAE latent 上约束，避开了 pixel-space reconstruction 的 decoding artifacts。

**Intuition for Karpathy**：这有点像你做 next-token prediction 时除了 cross-entropy 还加了一个 "first-difference of logits" matching loss——让模型在 token 空间的局部 manifold 上的切向量对齐，强制保留 multi-modal 的微分结构而不是坍缩到 mean mode。

---

### 3.3 Global Affine Coherence (§4.4)

#### 核心观察

长视频必须 sliding window inference。Generative model 的 stochastic sampling 在不同 window 独立采样，导致 **non-linear geometric distortion + flickering**，没法 align。

但 DVD 是 deterministic regressor：$\text{Var}[\hat{z}_d | z_x] = 0$。即使如此，windowed inference 还是有 secondary bottleneck——**VAE decoder 的 context-dependent normalization 导致 depth value 有 fluctuation**。

paper 关键的 empirical finding (Figure 6, Figure 12)：

> VAE decoding 主要引入的是 **global affine variation**（scale + shift），而不是 local spatial distortion。Inter-window discrepancy 可以被 $s \cdot \mathbf{d} + t$ 这种线性变换 well-approximate。

这个发现让我想起 Consistency Models 里的 affine invariance 假设。其实是对 VAE 行为的一种 empirical characterization。

#### Affine Alignment 公式

设两个连续 window 的 decoded depth tensors 为 $\mathcal{W}_A, \mathcal{W}_B$，提取 overlap 区域的 flattened pixels $\mathbf{d}_A^{overlap}, \mathbf{d}_B^{overlap} \in \mathbb{R}^{\hat{N}}$。

**Least-squares objective (Eq. 10):**

$$\arg\min_{s,t} \|s \mathbf{d}_B^{overlap} + t \mathbf{1} - \mathbf{d}_A^{overlap}\|_2^2$$

**Closed-form solution (Eq. 11):**

$$s = \frac{\text{Cov}(\mathbf{d}_A^{overlap}, \mathbf{d}_B^{overlap})}{\text{Var}(\mathbf{d}_B^{overlap})}, \quad t = \mu_A - s \mu_B$$

变量：
- $s$：global scale
- $t$：global shift
- $\mu_A, \mu_B$：两个 overlap 区域的均值
- $\text{Cov}$：两个 overlap 区域 depth 值的协方差
- $\text{Var}$：$\mathbf{d}_B^{overlap}$ 的方差

广播到整个 current window：$\hat{\mathcal{W}}_B = s \cdot \mathcal{W}_B + t$，overlap 区域用 linear interpolation blend。

#### Overlap size ablation (Table 9)

| Overlap O | AbsRel↓ | δ₁↑ | Rel. Time↓ |
|---|---|---|---|
| 3 | 7.9 | 0.937 | 1.00× |
| 6 | 7.7 | 0.941 | 1.04× |
| **9** | **7.3** | **0.945** | **1.17×** |
| 14 | 7.2 | 0.948 | 1.34× |
| 19 | 7.1 | 0.947 | 1.55× |

O=9 是 sweet spot——再大边际收益急剧递减，latency 成本显著。

#### 为什么 generative model 没法做这种 alignment？

因为它们的 distortion 不仅是 affine，还有 nonlinear mode switching。Stochastic sampling 每次可能跳到不同的 mode，align 不回来。DVD 的 deterministic 保证让 distortion 限定在 VAE-induced 的 linear 变换范围内。

---

## 4. Image-Video Joint Training (§4.5)

$$\mathcal{L}_{joint} = \mathcal{L}_{video} + \lambda_{image} \mathcal{L}_{image}$$

- Image (F=1, batch=128): 提供 high-frequency spatial anchor
- Video (batch=16): 强制 temporal coherence

**核心 insight**：纯 video 训练会 underfit 空间细节；先 image 后 video fine-tuning 会 catastrophic forgetting。

这个策略和 DepthCrafter 的"image-then-video"两阶段 sequential 训练完全不同。我个人的联想：这跟你在 micrograd / nanoGPT 里做的 "joint training on multiple contexts" 类似，避免 catastrophic forgetting 用 joint 而不是 sequential。

---

## 5. 实验数据深度解读

### 5.1 主结果 (Table 1)

**KITTI / ScanNet / Bonn / Sintel 四个 benchmark:**

| Method | Train Frames | KITTI AbsRel↓ | ScanNet AbsRel↓ | Bonn AbsRel↓ | Sintel AbsRel↓ |
|---|---|---|---|---|---|
| DepthCrafter (Diff-G) | ~30M | 9.9 | 7.1 | 5.9 | 37.1 |
| VDA (ViT-D) | 60M | 7.2 | 5.8 | 4.7 | 39.7 |
| **DVD (Diff-D)** | **367K** | **6.7** | **5.5** | **4.7** | 44.5 |

DVD 在 KITTI 和 ScanNet 上拿到 SOTA，**用的是 VDA 1/160 的训练数据**。这是 paper 最 sellable 的一点——163× less task-specific data。

Bonn 上 DVD=VDA (4.7)，但 Sintel 上 DVD 较弱（44.5 vs VDA 39.7）。Sintel 是动画 domain，可能 VAE encoder 对 stylized 内容编码偏弱。这是潜在的改进点。

### 5.2 Long Video (Table 2)

| Method | Paradigm | Bonn AbsRel↓ | KITTI AbsRel↓ |
|---|---|---|---|
| DepthCrafter | Diff-G | 8.5 | 12.0 |
| VDA | ViT-D | 6.6 | 9.6 |
| **DVD** | Diff-D | **5.3** | **7.6** |

DVD 在 long video 上 margin 比 short video 更大。这正是 affine alignment 策略发挥威力的地方。

### 5.3 Boundary metrics (Table 3)

ScanNet 上 B-F1：
- ChronoDepth: 0.204
- DepthCrafter: 0.173
- VDA: 0.210
- **DVD: 0.259**

KITTI 上 B-Recall：
- DepthCrafter: 0.082
- VDA: 0.047
- **DVD: 0.217** ← 4.6x improvement！

这是 LMR 直接见效的地方。

### 5.4 Efficiency (Figure 8)

- Training: 8×H100, <36h
- Inference: 接近 VDA（discriminative SOTA），完全跳过 iterative sampling
- Data scaling curve: 用 1% 数据已经超过 VDA

### 5.5 Cross-backbone Generalization (Table 6, Figure 13)

在 CogVideoX-5B 上 deploy，τ=0.5 依然是最优 anchor：
| τ | 0.1 | 0.3 | **0.5** | 0.7 | 0.9 |
|---|---|---|---|---|---|
| KITTI AbsRel↓ | 7.8 | 7.6 | **7.4** | 7.4 | 9.5 |

证明 deterministic adaptation 范式对 video DiT 架构有普遍性，不是 Wan-specific 的 trick。

### 5.6 LoRA Rank (Table 10)

| Rank | AbsRel↓ | δ₁↑ |
|---|---|---|
| 256 | 7.7 | 0.974 |
| 512 | 7.3 | 0.977 |
| 1024 | 7.3 | 0.979 |

Rank 512 是 sweet spot。说明把 video DiT 改成 dense depth regressor 需要比较大的 LoRA capacity——这不是 superficial style transfer，而是 fundamentally 学习一个 dense mapping。

---

## 6. Failure Case (§E, Figure 14)

1100 帧带剧烈 scene transition 的视频（indoor desk → outdoor tunnel）：

- **Global scale drift**: Frame #1 和 Frame #500 完全无视觉 overlap 时，DVD 和 VDA 都不可避免出现全局 scale drift
- **Local fidelity**: 但 DVD 在局部细节（手、笔记本屏幕）上始终 sharp，远胜 VDA 的 over-smoothing

paper 把这个 failure 归结为 open challenge for the field，合理。

---

## 7. 我的几点直觉和联想

### 7.1 和 Consistency Models 的关系
DVD 的 single-pass regression 让我直接想到 [Consistency Models (Song et al., 2023)](https://arxiv.org/abs/2303.01469) 和 [LCM (Luo et al., 2023)](https://arxiv.org/abs/2310.04378)。区别在于：
- Consistency Model 是 self-distillation，target 是网络自己的 trajectory
- DVD 是直接监督到 GT depth latent（discriminative 范式）

某种意义上，DVD 更接近 **flow matching with ODE-Euler shortcut**：把整个 flow trajectory 用一个 fixed anchor point 替代。

### 7.2 和 DINOv2 / VAE joint embedding 的关系
共享 VAE encoder 同时编码 RGB 和 depth (Eq. 1) 这个设计很关键。这跟 [Stable Diffusion VAE](https://arxiv.org/abs/2112.10752) 的设计相关，但 Latent Diffusion 通常只 encode RGB。DVD 假设 depth 和 RGB 在 VAE latent 上可以 align——这是 strong inductive bias，可能也是 Sintel 上表现稍差的原因（动画的 VAE encoding 可能和真实 depth 编码有 distribution gap）。

### 7.3 Mean Collapse 和 Neural Collapse
paper 引用 [Papyan et al., 2020](https://www.pnas.org/doi/10.1073/pnas.2015500117) 和 [Zhu et al., 2021](https://arxiv.org/abs/2105.13562) 的 Neural Collapse 概念。我觉得还可以关联 [Ma et al., 2025 ordinal regression neural collapse](https://arxiv.org/abs/2506.05801)——regression to mean 是一个更广义的 phenomenon。

### 7.4 和 Lotus / Lotus-2 的演进
[Lotus (He et al., 2024)](https://arxiv.org/abs/2409.18124) 在 image 上做了 single-step deterministic adaptation，[Lotus-2 (He et al., 2025)](https://arxiv.org/abs/2512.01030) 进一步改进。DVD 在 video 上做同样的事，但发现了三个 video-specific 的 issue：
1. Timestep 需要做 structural anchor (image 上不用)
2. Mean collapse 在 video 上被 temporal 累积放大
3. Long video 需要 affine alignment (image 上没这个问题)

这三个 finding 我觉得对后续做 video-to-video / video prediction 的 deterministic adaptation 都有借鉴意义。

### 7.5 Wan2.1 作为 backbone 的选择
[Wan (Team et al., 2025)](https://arxiv.org/abs/2503.20314) 是开源 video DiT 里的 strong baseline，1.3B 这个 size 选择得也比较平衡——既能借力 foundation prior，又能 8×H100 36h 训完。如果能上 HunyuanVideo 或者 Sora-class 的更大 backbone，性能应该还能再上一个台阶。

### 7.6 Limitations 的诚实
paper 诚实地指出三个 limitations：
1. Long video with massive scene transition → affine assumption 失效
2. Real-time deployment 困难（1.3B DiT 太重）
3. VAE 8× downsample 限制了 ultra-fine boundary 恢复

这些 limitation 都很合理，第 3 点尤其有意义——这指出了未来 VAE-free tokenization 的方向，和你之前在 token regression 上的 work 思路一致。

---

## 8. 总结性 intuition

DVD 这篇 paper 的核心 thesis 我觉得是这一句：

> **Pre-trained video diffusion models 编码了 profound 的时空几何 prior，但这种 prior 被 stochastic sampling 的 aleatoric variance 给掩盖了。一旦用 deterministic regression 把 prior 蒸馏出来，只需要 1% 的下游数据就能解锁。**

三大 design 分别对应三个 bottleneck：
- **τ as anchor** → spectral bias 下的 frequency regime selection
- **LMR** → regression-to-mean 的高频 collapse
- **Affine coherence** → long video 的 inter-window alignment

整篇 paper 的 style 我觉得非常像 [Marigold](https://arxiv.org/abs/2312.02112) 的 video 版延伸 + deterministic shortcut 的 binary swap。如果你的团队要复现，367K frames 训练、8×H100、36h，这个 setup 在 academic 可承受范围内。

---

## 9. 相关 references（web links）

- **DVD Project page**: https://dvd-project.github.io/
- **DVD Github**: https://github.com/EnVision-Research/DVD
- **Lotus (image predecessor)**: https://arxiv.org/abs/2409.18124
- **Lotus-2**: https://arxiv.org/abs/2512.01030
- **Marigold**: https://arxiv.org/abs/2312.02112
- **DepthCrafter**: https://arxiv.org/abs/2409.18120 (CVPR 2025)
- **ChronoDepth**: https://arxiv.org/abs/2403.07420
- **RollingDepth**: https://arxiv.org/abs/2504.07851
- **Video Depth Anything**: https://arxiv.org/abs/2501.10075
- **Wan2.1 (backbone)**: https://arxiv.org/abs/2503.20314
- **CogVideoX**: https://arxiv.org/abs/2408.06072
- **HunyuanVideo**: https://arxiv.org/abs/2412.03603
- **Rectified Flow**: https://arxiv.org/abs/2209.03003
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **Variational Diffusion Models (spectral analysis)**: https://arxiv.org/abs/2107.00608
- **Perception Prioritized Training**: https://arxiv.org/abs/2204.00227
- **Consistency Models**: https://arxiv.org/abs/2303.01469
- **Neural Collapse**: https://www.pnas.org/doi/10.1073/pnas.2015500117
- **Timestep embedding disappearance**: https://arxiv.org/abs/2405.14126
- **Hypersim dataset**: https://github.com/apple/ml-hypersim
- **TartanAir**: https://theairlab.com/tartanair/
- **Virtual KITTI**: http://www.virtualkitchen.org/

---

如果你想用这个 model 做 downstream experiments（比如 SLAM、robot manipulation、3D Gaussian Splatting），它的 deterministic output + affine-coherent long video 应该特别友好。我特别好奇的一点是：能不能把这个 deterministic adaptation 范式 extend 到 **flow estimation / optical flow** 上——video DiT 里编码的 motion prior 应该也能被 deterministic regression 解锁。这跟 [GeometryCrafter (Xu et al., 2025)](https://arxiv.org/abs/2504.07889) 的 point map 方向有点互补。
