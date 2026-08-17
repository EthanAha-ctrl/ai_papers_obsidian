---
source_pdf: Geometric Context Transformer for.pdf
paper_sha256: 720e05d56d29d616591c1a47a0c060e7ba7b0fadd2832a5478745aa45263c222
processed_at: '2026-08-04T21:15:41-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# LingBot-Map 人话版：用直觉理解 Streaming 3D Reconstruction

## 一句话总结

你拿手机拍视频，机器一边看一边实时建3D地图和自己的运动轨迹——LingBot-Map干的就是这事，而且比之前所有方法都好，甚至比离线处理的还强。

## 核心矛盾：记住多少 vs 算多快

想象你在陌生城市走路，要一边走一边脑子里建地图。你得记住走过的路才能不迷路（全局一致性），但脑子容量有限不能啥都记（算得快）。这是个根本矛盾。

之前的几条路都有问题：

**CUT3R**：用一个不断压缩更新的state，压太狠就"忘事"了，走远了自己在哪都不知道。

**StreamVGGT / Stream3R**：causal attention + cache，几乎啥历史都留着。走越久memory越大、越慢，10000帧就扛不住。

**VGGT-SLAM / MASt3R-SLAM**：3D foundation model + 经典SLAM后端。手工规则选keyframe，还得iterative optimization做bundle adjustment，实时性差。

## 关键洞察：从经典SLAM偷灵感

经典SLAM系统（ORB-SLAM3、DROID-SLAM这些）早就知道要维护**三类不同的空间记忆**：

1. **参考帧**：固定"我在哪"和"世界多大"——否则单目相机天生不知道scale，1米还是10米分不清
2. **局部窗口**：最近几帧的密集视觉信息，用来精准把新帧"注册"进来
3. **全局地图**：压缩的历史轨迹，用来纠正累积漂移

LingBot-Map的天才之处：把这三类手工设计的context**全部换成learnable attention**，叫Geometric Context Attention（GCA）。保留了structural inductive bias，但去掉了hand-crafted heuristics的局限。

## GCA 三级 Context 详解

### Anchor Context：固定坐标和scale

单目重建有个根本问题叫**scale ambiguity**——你看到的是一个远处的大东西还是近处的小东西，从图像本身分不清。离线方法（DUSt3R、VGGT）用全局点云normalize，但streaming没法这么做（你看不到未来）。

LingBot-Map的做法：前 $n$ 帧（$n$ 很小，比如3）当**anchor frames**，它们之间做full attention，加一个learnable anchor token $\mathbf{a}$ 标记"我是锚点"。所有后续帧都attend到这 $n$ 帧。

训练时用anchor帧的ground-truth点云算scale：

$$s = \frac{1}{|\bar{\mathcal{X}}^{\mathrm{anchor}}|} \sum_{\mathbf{x} \in \bar{\mathcal{X}}^{\mathrm{anchor}}} \|\mathbf{x}\|_2$$

- $s$：scale normalization factor（标量）
- $\bar{\mathcal{X}}^{\mathrm{anchor}}$：anchor帧的ground-truth 3D点云
- $\mathbf{x}$：单个3D点
- $\|\mathbf{x}\|_2$：点到原点的距离

然后所有depth和translation除以 $s$。直观说：**"这段视频的1米是多长"由anchor帧说了算**，之后所有帧都在这个固定scale下推理。

### Local Pose-Reference Window：密集局部信息

维护最近 $k$ 帧（比如64）的**完整image tokens**。新帧靠和这些近邻的dense visual overlap精准注册——因为你要判断"我现在相对上一帧在哪"，得看到上一帧长啥样。

### Trajectory Memory：压缩的全局历史

最elegant的设计。对于既不在anchor也不在window的"老帧"，**只保留6个context tokens**（1 camera + 1 anchor + 4 register），扔掉 $M$ 个image tokens（$M \approx 500$）。

加Video RoPE让模型知道这些token的时间顺序。没有这个temporal encoding，trajectory memory只是"存了点几何信息"，但模型不知道"这些帧谁先谁后、隔多远"。

### 复杂度对比——为什么这个设计work

T帧序列的总context tokens：

- **Causal attention**：$T \cdot (M + 6) = MT + 6T$，每帧加 $M+6$
- **GCA**：$(n+k) \cdot M + 6T$，每帧只加6

关键：两者都有 $6T$ 项，但causal多一个 $MT$ 项随帧数线性增长，GCA的 $(n+k)M$ 是**常数**。

具体数字（$n=3, k=16, T=10000$）：
- Causal：~$5 \times 10^6$ tokens
- GCA：~$7 \times 10^4$ tokens
- **约70×减少**

## 架构整体

```
Image → DINOv2 ViT (patch 14) → M image tokens
                                 + camera token c
                                 + 4 register tokens r_j  
                                 + anchor token a
                                      ↓
              24层交替的 Frame Attention (帧内) + GCA (跨帧)
                                      ↓
              Camera head (从camera token) → P̂_t
              Depth head (从image tokens)  → D̂_t
```

初始化从VGGT来，关键改动：把global attention换成GCA。由于Q/K/V projection的参数化一样，pretrained weights直接transfer。

## Loss Function：三个监督信号

$$\mathcal{L} = \lambda_{\mathrm{depth}} \mathcal{L}_{\mathrm{depth}} + \lambda_{\mathrm{abs-pose}} \mathcal{L}_{\mathrm{abs-pose}} + \lambda_{\mathrm{rel-pose}} \mathcal{L}_{\mathrm{rel-pose}}$$

**Depth Loss**（跟VGGT一致，带uncertainty）：

$$\mathcal{L}_{\mathrm{depth}} = \sum_{i=1}^{N} \|\Sigma_i^D \odot (\hat{D}_i - D_i)\| + \|\Sigma_i^D \odot (\nabla \hat{D}_i - \nabla D_i)\| - \alpha \log \Sigma_i^D$$

- $\Sigma_i^D$：per-pixel uncertainty map，让模型对没把握的地方降低loss权重
- $\nabla D$：depth空间梯度，监督边缘锐度
- $-\alpha \log \Sigma$：惩罚过大的uncertainty，防止模型"啥都说没把握"来偷懒

**Absolute Pose Loss**——关键改动：

$$\mathcal{L}_{\mathrm{abs-pose}} = \sum_{i=1}^{N} \|\hat{\mathbf{P}}_i - \mathbf{P}_i\|_{d_i^*}$$

用**camera-to-world**变换，不是world-to-camera。为什么？world-to-camera下rotation和translation**耦合**：rotation错一点，translation就跟着崩。长序列里这个耦合被error propagation放大，特别致命。camera-to-world解耦了两者。

**Relative Pose Loss**（创新点，受 $\pi^3$ 启发）：

$$\mathcal{L}_{\mathrm{rel-pose}} = \frac{1}{k(k-1)} \sum_{\substack{i \neq j \\ i,j \in \{1,...,k\}}} \left( \mathcal{L}_{\mathrm{rot}}(i,j) + \lambda_{\mathrm{trans}} \mathcal{L}_{\mathrm{trans}}(i,j) \right)$$

- $k$：window size
- $\mathcal{L}_{\mathrm{rot}}(i,j)$：帧 $i,j$ 之间的geodesic rotation error
- $\mathcal{L}_{\mathrm{trans}}(i,j)$：L1 translation error
- $\lambda_{\mathrm{trans}}$：translation权重

监督window内**所有帧对**的相对pose。因为window都是已观测帧，这个loss天生causal，不会"偷看未来"。直接约束frame-to-frame相对运动，防止小误差累积成长轨迹drift。

## 训练：两阶段课程学习

### Stage 1：Base Model（建几何先验）

- Global attention（不用GCA），2-24帧
- 29个数据集：BlendedMVS, HyperSim, MegaDepth, CO3D, TartanAir, ScanNet++等
- Nearby sampler：随机选reference frame，周围采其他帧，**不强制时间顺序**（适合混合模态数据）
- 160K iterations, AdamW, lr=$2 \times 10^{-4}$
- ~21,500 GPU hours

**Data augmentation很aggressive**：
- Color jitter (brightness/contrast/saturation ±0.5, hue ±0.1), prob 0.9
- Random grayscale, prob 0.05
- **Co-jittering**（prob 0.3）：同一scene所有帧用**相同**color transform。否则per-frame独立

Co-jittering的直觉：当帧间appearance相似时，逼模型靠**几何**cue而非**颜色**shortcut来做匹配。

### Stage 2：Streaming Model（迁移到长序列）

- 从Stage 1初始化，global attention→GCA
- **Progressive view curriculum**：24帧→320帧线性增长
  - 为什么不直接训long sequence？因为**早期帧的pose误差沿轨迹传播，destabilize loss landscape**，训练发散
  - 先短序列建立reliable local geometry，再scale up学long-range consistency
- Window size $k$ 随机16-64，让推理时不同window size都鲁棒
- **Ulysses context parallelism**：跨16 GPU分布views，用all-to-all通信算attention
- 160K iterations, lr=$5 \times 10^{-4}$
- ~15,360 GPU hours

**Foldback Video Sampler**（聪明的小trick）：

从random frame开始，random stride前进，到序列边界时**反向**取新stride（要求distinct from previous避免来回振荡）。产出naturally varying frame rates + no forward-time bias的subsequence。适合从长视频里采训练样本。

## 推理工程：让理论跑起来

### Paged KV-Cache

Naive causal attention的KV cache linearly增长，且window eviction需要频繁cache update（加新的、删旧的），contiguous layout下频繁memory reallocation开销大。

借鉴vLLM的PagedAttention：updates只影响新append的tokens。基于**FlashInfer**实现（native paged + sparse attention kernels）。

性能：518×378, 1000帧, window 64：
- FlashInfer: **~20 FPS**
- PyTorch baseline: ~10.5 FPS
- **约2×加速**

### Keyframe Selection

超过训练长度时，每帧算相对最近keyframe的光流（用predicted pose + depth），超阈值才当keyframe保留到KV cache，否则丢弃。

### 两种推理模式

**Direct Output Mode**（默认）：
- 全程causal，三级context持续积累
- 训练时最多320帧，实测稳定到**~3000帧**（10× training length）
- 超过3000帧逐渐degrade

**VO Mode**（超长序列）：
- 分overlapping windows
- 每个window：先处理initial subset建立local scale，剩余causal处理
- Window结束时reset state
- 用**Sim(3) alignment**在overlap region融合consecutive windows
- 能跑任意长（10000+帧），但每个boundary引入额外alignment误差

Trade-off：Direct更准（无inter-window alignment error），适合3000帧内；VO更长但有累积alignment drift。

## 实验结果：数字说话

### Oxford Spires（sparse, 320帧）——最有挑战的benchmark

| Method | Type | AUC@15↑ | ATE↓ |
|--------|------|---------|------|
| DA3 | offline | 49.84 | 12.87 |
| VIPE | optim | 45.35 | 10.52 |
| CUT3R | online | 5.98 | 18.16 |
| **LingBot-Map** | **online** | **61.64** | **6.42** |

震撼结果：**online streaming方法超越最强的offline（DA3）和optimization-based（VIPE）方法**。AUC@15比DA3高11.8 points，ATE降一半。

为什么offline方法在Oxford Spires上崩？因为它们在小viewpoint数据集上训练，遇到Oxford Spires的complex scene transitions（室外到黑暗楼梯间）和large viewpoint变化，learned priors无法transfer。

### Oxford Spires（dense vs sparse, 长序列压力测试）

| Method | ATE_sparse↓ | ATE_dense↓ | FPS↑ |
|--------|-------------|------------|------|
| CUT3R | 18.16 | - | 29.21 |
| TTT3R | 19.35 | 32.47 (+14.31) | 28.97 |
| Wint3R | 21.10 | 32.90 (+11.80) | 3.88 |
| Inf.-VGGT | 30.49 | 32.90 (+11.80) | 7.78 |
| Stream3R-w | 33.03 | 31.75 (+1.26) | 13.66 |
| **LingBot-Map** | **6.42** | **7.11 (+0.69)** | **20.29** |

12×长序列（320→3840帧），ATE只涨0.69！竞品涨10+。这证明GCA三级context有效preserve long-range consistency，**无需explicit optimization或loop closure**。

### 其他benchmark全面领先

| Dataset | Metric | LingBot-Map | 次优 | 提升 |
|---------|--------|-------------|------|------|
| ETH3D | F1↑ | 98.98 | 77.28 (Wint3R) | +21.70 |
| 7-Scenes | ATE↓ | 0.08 | 0.10 (Stream3R) | 1.25× |
| T&T | AUC@30↑ | 92.80 | 81.33 (Stream3R) | +11.47 |
| NRGBD | F1↑ | 64.26 | 56.96 (Wint3R) | +7.30 |

## Ablation：每个组件都重要

| Rel. Loss | A.Init | Co.Tok | V.RoPE | AUC@3↑ | ATE↓ | RPE-rot↓ |
|-----------|--------|--------|--------|--------|------|----------|
| ✓ | | | | 9.80 | 8.59 | 2.57 |
| ✓ | ✓ | | | 13.63 | 7.88 | 2.90 |
| | ✓ | ✓ | | 13.91 | 8.25 | 5.35 |
| ✓ | ✓ | ✓ | | 15.75 | 7.46 | 2.26 |
| ✓ | ✓ | ✓ | ✓ | **16.39** | **5.98** | **1.93** |

关键insight：

**Anchor Init**：AUC@3 +3.83。解决scale ambiguity，每帧注册到well-defined geometric reference。

**Context Tokens（trajectory memory）**：AUC@3 +2.12, ATE -0.42。lightweight memory减少long-range drift。

**Relative Pose Loss**：没它RPE-rot从2.26→5.35（2.4× worse）。**rotation estimation对local pairwise supervision特别敏感**——这很intuitive，相对旋转靠局部visual cue，没有显式pairwise监督模型容易在小误差上"摇摆"。

**Video RoPE**：ATE 7.46→5.98（−1.48），**单组件最大改善**！

为什么temporal encoding这么重要？没有它，trajectory memory tokens carry geometric information但缺乏"when"的concept。Video RoPE注入temporal ordering，让模型能reason about "how far apart two frames are in time, which direction camera has been moving"。

这让我想到LLM中position encoding的演变——从absolute到relative再到RoPE，每次都在改进模型对sequence structure的reasoning。3D reconstruction的trajectory本质也是sequence，temporal structure是intrinsic信息。

### Bounded Window vs Full Attention（反直觉）

| Window | ATE↓ | RPE-trans↓ | RPE-rot↓ | FPS↑ | Mem↓ |
|--------|------|------------|----------|------|------|
| 64 | **5.98** | **1.33** | 1.93 | **20.29** | **13.28GB** |
| Full | 6.60 | 1.50 | **1.71** | 11.87 | 36.06GB |

Bounded window**反而更准**（ATE 5.98 vs 6.60）！Intuition：retaining all historical image tokens introduces **noise from distant, less relevant frames** that confuse attention。GCA的哲学：evict image tokens但preserve compact context tokens for full trajectory，retains essential cues同时filter out redundancy。

这呼应信息bottleneck theory——**不是more information更好，是right information更好**。

## 直觉总结：为什么这个设计work

### 核心哲学

**streaming state应该selectively retain what matters most，不是retain how much**。selection要grounded in geometric priors且end-to-end learned。

### 三级context是human spatial cognition的computational analogue

- **Landmark-based navigation** → Anchor Context（你在哪个坐标系）
- **Local visual short-term memory** → Pose-Reference Window（最近看到了啥，用来定位）
- **Cognitive map for global consistency** → Trajectory Memory（整体走过了哪，别迷路）

人类spatial memory不是faithful recording of every moment，而是sparse, structured, efficient。LingBot-Map的three-level context结构在某种意义上是human spatial cognition的computational analogue。

### Streaming可以超越Offline

LingBot-Map在Oxford Spires上超越offline和optimization-based方法，challenge了一个常见假设：streaming是offline的approximation，性能上限更低。

当**structured inductive bias（来自classical SLAM的insight）+ end-to-end learning**结合时，streaming有自己独特优势——自然处理sequential information流，避免global attention在large viewpoint变化下的failure。Offline方法在小viewpoint数据集上训练，prior太narrow，遇到real-world复杂场景就崩。

### Camera-to-World解耦的妙处

用camera-to-world而不是world-to-camera，是个小改动但影响大。world-to-camera下rotation和translation耦合，rotation错一点translation就崩，长序列里error propagation放大。camera-to-world解耦了两者，让translation estimation不再被rotation error绑架。

### Engineering Excellence

理论漂亮不够，得让模型跑起来：
- Paged KV-cache避免频繁memory reallocation
- FlashInfer的paged/sparse attention kernels
- Ulysses context parallelism跨GPU训练长序列
- Keyframe selection基于optical flow
- Two inference modes处理不同length regimes

这些production-level细节让20 FPS实时推理成为可能。

## Limitations与Future

Paper自己承认的局限：
1. 没有explicit loop-closure detection（重访已观测区域时还能进一步减少drift）
2. Trajectory memory固定6 tokens/frame，超长序列可能丢fine-grained details
3. 没有test-time optimization（挑战场景下能进一步refine）

Future方向很exciting：
- Bundle-adjustment-like refinement + explicit loop-closure in attention
- 动态场景with moving objects
- 多模态inputs（LiDAR, IMU）
- 作为backbone for novel view synthesis, navigation, embodied AI

## Broader Implications

LingBot-Map让我看到3D foundation model的新paradigm：**不是简单把offline model放到streaming setting，而是从first principles重新设计context management**。

这跟LLM的context engineering异曲同工——LLM处理超长context也面临类似trade-offs（ring attention, sliding window, sparse attention, KV-cache compression）。LingBot-Map在3D vision领域探索了类似的design space，而且**structural prior从classical SLAM来**这一点很关键——不是盲目堆engineering trick，而是有principled的inductive bias。

最有意思的是，这个work证明了一个deep的点：**有时候结构化的先验知识（来自几十年SLAM研究的insight）+ end-to-end learning的结合，比纯data-driven的black-box更强大**。GCA的三级context结构不是凭空发明的，是站在classical SLAM巨人的肩膀上，用learnable attention重新实现了一遍。

## References

- [LingBot-Map Project](https://technology.robbyant.com/lingbot-map)
- [LingBot-Map GitHub](https://github.com/robbyant/lingbot-map)
- [VGGT - Visual Geometry Grounded Transformer](https://vggt.github.io/)
- [DUSt3R - Geometric 3D Vision Made Easy](https://dust3r.europe.naverlabs.com/)
- [DROID-SLAM](https://princeton-vl.github.io/DROID-SLAM/)
- [DINOv2](https://dinov2.metademolab.com/)
- [FlashInfer - LLM Inference Engine](https://flashinfer.ai/)
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180)
- [Ulysses Context Parallelism](https://arxiv.org/abs/2309.14509)
- [RoPE - Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [CUT3R - Continuous 3D Perception](https://cut3r.github.io/)
- [StreamVGGT](https://arxiv.org/abs/2504.16318)
- [Stream3R](https://arxiv.org/abs/2506.21436)
- [TTT3R - Test-Time Training for 3D](https://arxiv.org/abs/2506.22385)
- [Depth Anything 3](https://arxiv.org/abs/2511.10647)
- [VGGT-SLAM](https://arxiv.org/abs/2507.07257)
- [MASt3R-SLAM](https://huggingface.co/papers/2412.03941)
- [Oxford Spires Dataset](https://oxford-spires.lukasbieri.net/)
- [ETH3D Benchmark](https://www.eth3d.net/)
- [Tanks and Temples](https://www.tanksandtemples.org/)
- [TartanAir](https://theairlab.org/tartanair/)
- [Habitat-Sim](https://aihabitat.org/)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [π³ - Permutation-Equivariant Visual Geometry](https://arxiv.org/abs/2507.13347)

---

# LingBot-Map: Geometric Context Transformer for Streaming 3D Reconstruction 深度解析

## 1. 核心Problem与Motivation

这篇paper tackle的是streaming 3D reconstruction这个非常fundamental的问题。给定一个continuous video stream $\mathcal{T} = \{I_1, I_2, ...\}$, 模型需要online地estimate每个新frame $I_t$ 的 camera pose $\hat{P}_t$ 和 depth map $\hat{D}_t$, 并且只能用 current 和 previous frames $\{I_1, ..., I_t\}$, 无法访问future observations。

这里最关键的tension在于：**rich geometric context** for long-term consistency vs **compact streaming state** for efficient inference。这是一个非常经典的trade-off，在classical SLAM中早就被讨论过。

作者从classical SLAM系统中提取了一个非常重要的insight：robust real-time reconstruction需要维护三类distinct的spatial context：
- 一个**reference frame**用于coordinate和scale grounding
- 一个**local window** for dense local geometry estimation  
- 一个**global map** for drift correction

这个想法非常natural，让我想到 ORB-SLAM3 中的三个map（Atlas multi-map system）和DROID-SLAM中selective keyframe的设计。但LingBot-Map把这些hand-crafted的component替换成了end-to-end learned attention，这是一个很漂亮的unification。

## 2. Geometric Context Attention (GCA) - 核心方法

### 2.1 Anchor Context

Monocular reconstruction本质上 scale-ambiguous，所以必须先建立一个consistent的coordinate system和absolute scale。Offline方法如DUSt3R和VGGT通过global point cloud normalization解决，但这对causal streaming inference不适用。

LingBot-Map的方案：designate前 $n$ 帧（$n \ll N$）作为anchor frames，apply full attention among them，并且augment它们的image tokens with a learnable anchor token $\mathbf{a} \in \mathbb{R}^C$。

训练时用如下公式做scale normalization：

$$s = \frac{1}{|\bar{\mathcal{X}}^{\mathrm{anchor}}|} \sum_{\mathbf{x} \in \bar{\mathcal{X}}^{\mathrm{anchor}}} \|\mathbf{x}\|_2$$

变量含义：
- $s$: scale normalization factor（标量）
- $\bar{\mathcal{X}}^{\mathrm{anchor}}$: anchor frames的ground-truth point cloud
- $\mathbf{x}$: 单个3D point
- $\|\cdot\|_2$: L2 norm（point到coordinate origin的距离）

然后将所有ground-truth depths和camera translations除以 $s$。这个设计非常clean——anchor frames相当于定义了"这段视频的米是多少"。

### 2.2 Local Pose-Reference Window

维护一个sliding window of the $k$ most recent frames，保留它们的**full image tokens**。这提供了dense visual overlap，对于accurate frame registration至关重要。

这个设计让我想到StreamVGGT和CUT3R的不同选择：
- StreamVGGT保留near-complete history，导致memory linearly增长
- CUT3R用persistent recurrent state，aggressive compression导致forgetting
- LingBot-Map选择只保留最近的dense tokens，是个很合理的middle ground

### 2.3 Trajectory Memory

这是最elegant的设计。对于既不在anchor set也不在active sliding window的frames，只保留 **camera token + anchor token + 4 register tokens**（共6个context tokens per frame），丢弃memory-intensive的image tokens（M tokens per frame）。

并且加入video temporal positional encodings来impose temporal ordering。

这里的6 tokens = 1 camera + 1 anchor + 4 register，是个很compact的representation。M≈500时，每frame context从M+6减少到6，约80× reduction。

### 2.4 Complexity Analysis - 关键数学

对于T-frame sequence：
- n anchor frames: $n \cdot (M + 6)$ tokens
- k window frames: $k \cdot (M + 6)$ tokens  
- $(T - n - k)$ trajectory frames: $6$ tokens each

Total context = $(n + k) \cdot M + 6T$

对比causal attention: $T \cdot (M + 6) = MT + 6T$

两者都有 $6T$ term，但causal多了一个 $MT$ term，这个term随full token count增长。

具体数值（$n=3, k=16, T=10000$）：
- Causal: ~$5 \times 10^6$ tokens
- GCA: ~$7 \times 10^4$ tokens
- 约70× reduction

让我用代码示意这个结构：

```
Frame t arrives:
KV-cache contains:
  [Anchor_1..n (full)] + [Trajectory (6 tokens each)] + [Window_1..k (full)]
  
New frame t:
  - Computes attention to anchor (constant cost)
  - Computes attention to trajectory memory (grows 6 per frame)  
  - Computes attention to window (constant cost)
  - Old window frame evicted → compressed to 6 tokens → joins trajectory
```

### 2.5 Attention Mask Design

Figure 3展示了四种attention pattern的对比：
- (a) Full attention: 全部attend，无法streaming
- (b) Causal attention: 可以streaming，但memory linearly增长
- (c) Sliding window: bounded cost，但loss long-range context
- (d) GCA: anchor + trajectory + window，retains long-range consistency with bounded per-frame cost

这个设计有点让我想到Longformer的local + global attention混合，但更structured。

## 3. Architecture与Loss Function

### 3.1 Architecture

Pipeline（Figure 4）：
1. Input image $I_t$ 通过 DINOv2 ViT backbone (patch size 14) → M image tokens
2. Augment with: camera token $\mathbf{c} \in \mathbb{R}^C$, 4 register tokens $\mathbf{r}_j \in \mathbb{R}^C$ ($j=1,...,4$), anchor token $\mathbf{a} \in \mathbb{R}^C$
3. 24 alternating blocks of Frame Attention + GCA
   - Frame Attention: per-frame内部，per-frame feature refinement
   - GCA: 跨frame，按structured attention mask做cross-frame geometric reasoning
4. Camera head takes camera token → $\hat{P}_t$ (absolute camera pose)
5. Depth head takes image tokens → $\hat{D}_t$ (depth map)

初始化从VGGT架构来，关键改动是把global attention替换成GCA。由于query/key/value projections的parameterization相同，pretrained weights直接transfer。

### 3.2 Loss Function

$$\mathcal{L} = \lambda_{\mathrm{depth}} \mathcal{L}_{\mathrm{depth}} + \lambda_{\mathrm{abs-pose}} \mathcal{L}_{\mathrm{abs-pose}} + \lambda_{\mathrm{rel-pose}} \mathcal{L}_{\mathrm{rel-pose}}$$

变量含义：
- $\lambda_{\mathrm{depth}}, \lambda_{\mathrm{abs-pose}}, \lambda_{\mathrm{rel-pose}}$: 三个loss的权重系数

**Depth Loss** (跟VGGT一致)：

$$\mathcal{L}_{\mathrm{depth}} = \sum_{i=1}^{N} \left\| \Sigma_i^D \odot (\hat{D}_i - D_i) \right\| + \left\| \Sigma_i^D \odot (\nabla \hat{D}_i - \nabla D_i) \right\| - \alpha \log \Sigma_i^D$$

变量含义：
- $N$: number of frames
- $\Sigma_i^D$: 第 $i$ 帧的predicted uncertainty map（per-pixel）
- $\odot$: element-wise multiplication
- $\hat{D}_i, D_i$: predicted和ground-truth depth
- $\nabla D$: depth spatial gradient
- $\alpha$: uncertainty regularization coefficient

这个loss借鉴了 Kendall & Gal 的 aleatoric uncertainty estimation，让模型对high-uncertainty regions降低loss权重，同时惩罚过大的uncertainty（$-\alpha \log \Sigma$ term防divergence）。

**Absolute Pose Loss**:

$$\mathcal{L}_{\mathrm{abs-pose}} = \sum_{i=1}^{N} \left\| \hat{\mathbf{P}}_i - \mathbf{P}_i \right\|_{d_i^*}$$

关键改动：用 **camera-to-world** transformation而不是world-to-camera。在world-to-camera parameterization中，rotation和translation inherently coupled，让translation estimation对rotation error非常敏感，特别是long sequences中。

**Relative Pose Loss**（创新点）：

$$\mathcal{L}_{\mathrm{rel-pose}} = \frac{1}{k(k-1)} \sum_{\substack{i \neq j \\ i,j \in \{1, ..., k\}}} \left( \mathcal{L}_{\mathrm{rot}}(i,j) + \lambda_{\mathrm{trans}} \mathcal{L}_{\mathrm{trans}}(i,j) \right)$$

变量含义：
- $k$: sliding window size
- $i, j$: window内的frame index
- $\mathcal{L}_{\mathrm{rot}}(i,j)$: geodesic rotation error between frame $i$ 和 $j$
- $\mathcal{L}_{\mathrm{trans}}(i,j)$: $\ell_1$ translation error
- $\lambda_{\mathrm{trans}}$: translation loss的权重

这是inspired by $\pi^3$ (permutation-equivariant visual geometry learning)。因为window全是already-observed frames，这个loss inherently causal，鼓励local trajectory consistency。

## 4. Training Strategy - Two-Stage Curriculum

### 4.1 Stage 1: Base Model

- ViT backbone from DINOv2 (patch size 14)
- 24 alternating blocks of Frame Attention + cross-frame attention（标准global attention，没有GCA）
- Input views: 2-24 frames randomly sampled
- AdamW, lr = $2 \times 10^{-4}$, weight decay = 0.05
- Linear warmup (5%) + cosine annealing (95%) to $10^{-8}$
- 160K iterations
- FSDP + gradient checkpointing + bfloat16
- ~21,500 GPU hours

**Data augmentation** 很aggressive：
- Color jitter (brightness/contrast/saturation ±0.5, hue ±0.1) with prob 0.9
- Random grayscale with prob 0.05
- Random spatial rescale [0.8×, 1.2×], aspect ratio [0.33, 1.0]
- **Co-jittering**: 同一scene所有frame用同一color transform (prob 0.3)，否则per-frame独立

Co-jittering很有意思——它鼓励模型依赖geometric cues而非appearance shortcuts，因为frame间有相同photometric characteristics时，模型不能依赖color差异。

### 4.2 Stage 2: Streaming Model

- Initialize from Stage 1 weights
- Replace global attention with GCA
- 160K iterations, lr = $5 \times 10^{-4}$
- **Progressive view curriculum**: views 24 → 320 linearly（24是Stage 1 max，320是GPU memory budget under context parallelism）
- Window size k: randomly sampled 16-64 during training
- **Ulysses context parallelism** with parallelism dim = 16
- ~15,360 GPU hours
- Builds on TorchTitan + Magi Attention

**Foldback Video Sampler** 是个聪明的trick：从random frame开始，random stride前进，到boundary时**反向**并取新的stride（distinct from previous避免degenerate oscillation）。这产生具有naturally varying frame rates且no forward-time bias的subsequences。

### 4.3 Training Data

29 datasets，分两个stage：
- **Stage 1** (diverse short-sequence): BlendedMVS, HyperSim, MegaDepth, CO3D, Objaverse, TartanAir, ScanNet++, etc.，每scene采2-24帧，**nearby sampler**（无temporal order）
- **Stage 2** (long-trajectory video): 大幅增加TartanAir, TartanAirV2, MatrixCity, Waymo, KITTI-360, internal game datasets的权重，down-weight multi-view-only datasets

总训练数据规模惊人，包括14.4 TB的cross-scene traversal data (rendered via Habitat-Sim from Gibson, Matterport3D, HM3D)。

## 5. Inference System Design

### 5.1 Paged KV-Cache

借鉴vLLM的PagedAttention。Naive causal attention的KV cache linearly增长，且sliding-window和trajectory-eviction logic需要频繁cache update（append新entries, discard旧的），contiguous layout下频繁memory reallocation开销大。

Paged layout让updates只影响新append的tokens，不影响整个cached sequence。

实现基于 **FlashInfer**（native paged KV-cache + 优化的sparse/paged attention kernels）。

性能对比：518×378分辨率，1000帧序列，sliding window 64帧：
- FlashInfer-based: ~20 FPS
- PyTorch baseline (contiguous KV-cache): ~10.5 FPS
- 约2× speedup

### 5.2 Keyframe Selection

当input sequence超过training max length时，每 $m$ 帧选一个keyframe保留在KV cache中。对于每incoming frame：
1. 模型estimate depth map和camera pose
2. 用predicted pose和depth计算相对于most recent keyframe的optical flow
3. 如果flow magnitude > threshold → 新keyframe，append到KV cache
4. 否则 → discard

### 5.3 Two Inference Modes

**Direct Output Mode** (default)：
- 全程causal processing，三级context (anchor + trajectory + window) 持续积累
- 每frame直接输出absolute pose和dense depth
- 训练时最多320 views，empirically稳定到 ~3,000 frames (10× training length)
- 超过3,000 frames后gradually degrade

**Visual Odometry (VO) Mode** (for 超长sequences)：
- Input partition成overlapping local windows
- 每个window：先处理initial subset建立local scale/coordinate，剩下causal处理
- 每个window结束时reset model state
- 用 **Sim(3) alignment** 在consecutive windows的overlap region上fuse
- 可以处理arbitrarily long sequences，但每个window boundary引入额外alignment error

Trade-off：Direct mode更accurate（避免inter-window alignment error），适合~3,000 frames内；VO mode更长sequence但累计alignment drift。

## 6. Experimental Results - 关键数据

### 6.1 Oxford Spires (Sparse, 320 frames)

Table 2关键数据：

| Method | Type | AUC@15↑ | AUC@30↑ | ATE↓ | RPE-trans↓ | RPE-rot↓ |
|--------|------|---------|---------|------|------------|----------|
| DA3 | offline | 49.84 | 56.68 | 12.87 | 3.22 | 16.17 |
| VIPE | optim | 45.35 | 51.88 | 10.52 | 0.43 | 5.98 |
| CUT3R | online | 5.98 | 14.95 | 18.16 | 1.17 | 7.18 |
| **LingBot-Map** | online | **61.64** | **75.16** | **6.42** | 1.01 | **3.70** |

非常震撼的结果——一个online streaming方法**超越**了最强的offline method (DA3)和optimization-based method (VIPE)！AUC@15比DA3高11.8 points, ATE降低一半。

这个结果很有intuition：offline方法在small viewpoint datasets上训练，遇到Oxford Spires的complex scene transitions和large viewpoint changes时learned priors无法transfer。

### 6.2 Oxford Spires (Dense vs Sparse)

Table 3是关键的长序列test：

| Method | ATE_sparse↓ | ATE_dense↓ | FPS↑ |
|--------|-------------|------------|------|
| CUT3R | 18.16 | - | 29.21 |
| TTT3R | 19.35 | 32.47 (+14.31) | 28.97 |
| Wint3R | 21.10 | 32.90 (+11.80) | 3.88 |
| Inf.-VGGT | 30.49 | 32.90 (+11.80) | 7.78 |
| Stream3R-w | 33.03 | 31.75 (+1.26) | 13.66 |
| **LingBot-Map** | **6.42** | **7.11 (+0.69)** | **20.29** |

12×更长sequence (320→3840 frames)，ATE只增加0.69！这非常impressive，证明GCA的三级context structure有效preserve了long-range geometric consistency，无需explicit optimization或loop closure。

### 6.3 其他benchmarks (Table 4)

| Method | ETH3D AUC30↑ | ETH3D ATE↓ | 7-Scenes AUC30↑ | 7-Scenes ATE↓ | T&T AUC30↑ | T&T ATE↓ |
|--------|--------------|------------|-----------------|---------------|-------------|----------|
| Stream3R | 64.76 | 1.67 | 73.70 | 0.10 | 81.33 | 0.76 |
| Wint3R | 58.71 | 0.86 | 63.02 | 0.12 | 57.85 | 0.88 |
| TTT3R | 56.12 | 1.22 | 71.23 | 0.10 | 71.30 | 0.66 |
| **LingBot-Map** | **86.20** | **0.22** | **78.59** | **0.08** | **92.80** | **0.20** |

Tanks & Temples上AUC@30=92.80 vs Stream3R 81.33 (+11.47)，ATE 0.20 vs 0.76 (3.8× lower)。
ETH3D上ATE 0.22 vs Wint3R 0.86 (4× lower)。

### 6.4 3D Reconstruction (Table 5)

| Method | ETH3D Acc↓ | ETH3D Comp↓ | ETH3D F1↑ | 7-Scenes F1↑ | NRGBD F1↑ |
|--------|-----------|-------------|-----------|--------------|-----------|
| Wint3R | 0.28 | 0.21 | 77.28 | 78.81 | 56.96 |
| Stream3R | 0.44 | 0.28 | 72.87 | 78.79 | 54.07 |
| **LingBot-Map** | **0.09** | **0.03** | **98.98** | **80.39** | **64.26** |

ETH3D F1=98.98 vs Wint3R 77.28 (+21.70 points!)——这是huge gap，reconstruction质量大幅提升。

### 6.5 Ablation Study (Table 6)

| Rel. Loss | A.Init | Co.Tok | V.RoPE | AUC@3↑ | AUC@30↑ | ATE↓ | RPE-rot↓ |
|-----------|--------|--------|--------|--------|---------|------|----------|
| ✓ | | | | 9.80 | 65.84 | 8.59 | 2.57 |
| ✓ | ✓ | | | 13.63 | 68.71 | 7.88 | 2.90 |
| | ✓ | ✓ | | 13.91 | 68.25 | 8.25 | 5.35 |
| ✓ | ✓ | ✓ | | 15.75 | 69.92 | 7.46 | 2.26 |
| ✓ | ✓ | ✓ | ✓ | **16.39** | **71.87** | **5.98** | **1.93** |

关键insights：
- **Anchor Init**: AUC@3 +3.83, ATE -0.71。解决scale ambiguity，让每frame注册到well-defined geometric reference
- **Context Tokens**: AUC@3 +2.12, ATE -0.42。Lightweight trajectory memory reduces long-range drift
- **Relative Pose Loss**: 没有它RPE-rot从2.26→5.35 (2.4× worse)。Rotation estimation对local pairwise supervision特别敏感
- **Video RoPE**: ATE 7.46→5.98 (−1.48)，**single largest ATE improvement**。Temporal ordering是让trajectory memory fully realize其potential的missing ingredient

### 6.6 Window Size Ablation (Table 7)

| Window Size | ATE↓ | RPE-trans↓ | RPE-rot↓ | FPS↑ | Mem(GB)↓ |
|-------------|------|------------|----------|------|----------|
| 64 | 5.98 | 1.33 | 1.93 | 20.29 | 13.28 |
| Full | 6.60 | 1.50 | 1.71 | 11.87 | 36.06 |

Counterintuitive结果——bounded window **更accurate** (ATE 5.98 vs 6.60)，因为retaining all historical image tokens introduces noise from distant, less relevant frames。1.7× speedup + 2.7× memory reduction。

## 7. 与相关Work的Connection

### 7.1 Feed-Forward 3D Foundation Models

- **VGGT** (https://vggt.github.io/): LingBot-Map的基础架构，24 alternating blocks of frame + cross attention。LingBot-Map把global attention换成GCA
- **DUSt3R/MASt3R** (https://dust3r.europe.naverlabs.com/): 范式转变的起点，但只支持two-view
- **Depth Anything 3 (DA3)** (https://arxiv.org/abs/2511.10647): 强offline baseline，LingBot-Map在Oxford Spires上大幅超越它
- **$\pi^3$**: Permutation-equivariant design，启发了relative pose loss
- **Fast3R** (https://arxiv.org/abs/2501.04661): 1000+ images一次forward pass，但是offline

### 7.2 Streaming Methods

- **CUT3R** (https://cut3r.github.io/): Persistent recurrent state，但aggressive compression导致forgetting
- **StreamVGGT** (https://arxiv.org/abs/2504.16318): Causal VGGT，但near-complete history retention
- **Stream3R** (https://arxiv.org/abs/2506.21436): Causal transformer with caching
- **TTT3R** (https://arxiv.org/abs/2506.22385): Test-time training策略
- **Wint3R** (https://openreview.net/forum?id=...): Window-based with camera token pool
- **Scal3R** (https://arxiv.org/abs/2507.19146): 大规模TTT
- **ZipMap** (https://arxiv.org/abs/2506.09238): Linear-time bidirectional reconstruction via TTT

LingBot-Map的关键区别：**purely feed-forward**，no test-time training或post-optimization。

### 7.3 Hybrid SLAM Methods

- **VGGT-SLAM** (https://arxiv.org/abs/2507.07257): VGGT + SLAM backend on SL(4) manifold
- **MASt3R-SLAM** (https://huggingface.co/papers/2412.03941): MASt3R + dense SLAM
- **DROID-SLAM** (https://princeton-vl.github.io/DROID-SLAM/): 经典的end-to-end learning-based SLAM

LingBot-Map借鉴了classical SLAM的三级context结构（reference frame + local window + global map），但用end-to-end learned attention替换了hand-crafted optimization。

### 7.4 System-Level Innovations

- **PagedAttention** (https://arxiv.org/abs/2309.06180): vLLM的核心，LingBot-Map在3D reconstruction场景应用
- **FlashInfer** (https://flashinfer.ai/): 高效attention engine，支持paged和sparse KV layouts
- **Ulysses Context Parallelism** (https://arxiv.org/abs/2309.14509): DeepSpeed的工作，用于训练时跨GPU分布views
- **RoPE** (https://arxiv.org/abs/2104.09864): Rotary Position Embedding，LingBot-Map用Video RoPE给trajectory memory加temporal ordering

### 7.5 Datasets

- **Oxford Spires** (https://arxiv.org/abs/2507.08960): 大规模outdoor/indoor混合，LiDAR-inertial SLAM ground truth
- **ETH3D** (https://www.eth3d.net/): 高分辨率indoor/outdoor with laser scanner depth
- **7-Scenes** (https://www.microsoft.com/en-us/research/project/rgb-d-camera-pose-estimation/): 经典room-scale RGB-D
- **Tanks and Temples** (https://www.tanksandtemples.org/): 大型outdoor multi-view
- **TartanAir** (https://theairlab.org/tartanair/): Synthetic aerial data
- **Habitat-Sim** (https://aihabitat.org/): 用于生成cross-scene traversal data

## 8. 个人Intuition与思考

### 8.1 为什么三级Context结构Work？

我从这篇paper得到的核心intuition是：**streaming state应该selectively retain what matters most，而不是retain how much**。这个selection应该grounded in geometric priors且end-to-end learned。

Classical SLAM系统早就知道三类context的功能分工：
- Reference frame：解决"在哪里"和"多大"
- Local window：解决"如何register新frame"
- Global map：解决"如何correct累积drift"

LingBot-Map的contribution是把这个structured prior变成learnable attention，避免了hand-crafted heuristics的局限性，同时保留了structural inductive bias。

### 8.2 Video RoPE的惊人效果

Ablation中Video RoPE带来single largest ATE improvement (−1.48)。这个结果让我重新思考trajectory memory的本质——不只是store geometric information，还要store **temporal structure**。

没有temporal encoding，trajectory memory tokens carry geometric information但缺乏"when"的concept。Video RoPE注入temporal ordering让模型能reason about sequential structure：how far apart two frames are in time，which direction camera has been moving。

这让我想到LLM中position encoding的演变——从absolute到relative再到RoPE，每次都在改进模型对sequence structure的reasoning能力。

### 8.3 Bounded Window > Full Attention的反直觉

Table 7的结果非常interesting——bounded window比full attention更accurate。Intuition：retaining all historical image tokens introduces noise from distant, less relevant frames that confuse attention computation。

GCA的design哲学：**evict image tokens但preserve compact context tokens for full trajectory**，retains essential geometric cues同时filter out redundant information。

这有点像信息bottleneck theory——不是more information更好，是**right information**更好。

### 8.4 Camera-to-World vs World-to-Camera

Paper提到一个technical detail：用camera-to-world而不是world-to-camera。在world-to-camera parameterization中，rotation和translation inherently coupled，让translation estimation对rotation error非常敏感，特别是long sequences。

这个观察很有深度。在classical SLAM中（比如Sophus on SE(3)/Sim(3) manifold），这种coupling通过Lie algebra的structure来handle。但end-to-end learning中，直接的loss formulation对这种coupling敏感。Camera-to-world decoupling是一个简单但effective的trick。

### 8.5 Training Recipe的智慧

Two-stage progressive training非常关键：
- Stage 1: 短序列，diverse data，build geometric priors
- Stage 2: 长序列，curriculum from 24→320 views，transfer到streaming setting

为什么直接训long sequence不行？因为**early-stage pose errors propagate along trajectory and destabilize loss landscape**。先用短序列建立reliable local geometry estimation，再scale up到long trajectories。

这让我想到curriculum learning的general principle——先学简单再学复杂，但这里"简单"="短序列"，"复杂"="长序列"，是因为long-horizon credit assignment困难。

### 8.6 Engineering Excellence

LingBot-Map的engineering非常solid：
- Paged KV-cache避免频繁memory reallocation
- FlashInfer的native paged/sparse attention kernels
- Context parallelism（Ulysses）训练时跨GPU分布views
- Keyframe selection基于optical flow magnitude
- Two inference modes (Direct vs VO) 处理不同length regimes

这些都是production-level的细节，让一个理论上漂亮的方法实际跑起来。

### 8.7 Limitations与Future Directions

Paper自己提到的limitations：
1. 没有explicit loop-closure detection
2. Trajectory memory compression成固定tokens per frame，可能在超长sequences丢失fine-grained details
3. 没有test-time optimization

Future directions很exciting：
- Bundle-adjustment-like refinement + explicit loop-closure in attention
- 动态场景with moving objects
- 多模态inputs (LiDAR, IMU)
- 作为backbone for novel view synthesis, navigation, embodied AI

### 8.8 Broader Implications

LingBot-Map让我看到3D foundation model的一个新paradigm：**不是简单把offline model放到streaming setting，而是从first principles重新设计context management**。

这跟LLM中的context engineering有异曲同工之妙——LLM在处理超长context时也面临类似trade-offs（ring attention, sliding window, sparse attention, KV-cache compression等）。LingBot-Map在3D vision领域探索了类似的design space。

最有意思的是，LingBot-Map在Oxford Spires上**超越**了offline和optimization-based方法。这challenge了一个常见假设：streaming methods是offline methods的approximation，性能上限更低。当structured inductive bias（来自classical SLAM的insight）+ end-to-end learning结合时，streaming可能不只是offline的approximation，而是有自己独特的优势——比如自然处理sequential information流，避免global attention在large viewpoint变化下的failure。

### 8.9 与人类Spatial Cognition的类比

Paper开头那段很poetic："we perceive the world through a continuous stream of visual input, yet our spatial memory is not a faithful recording of every moment: it is sparse, structured, and efficient."

人类spatial cognition selectively preserves essential cues：
- Landmark-based navigation（对应anchor context）
- Local visual short-term memory（对应pose-reference window）  
- Cognitive map for global consistency（对应trajectory memory）

LingBot-Map的three-level context结构在某种意义上是human spatial cognition的computational analogue，这是一个很deep的connection。

## 9. 总结

LingBot-Map是一个milestone级别的工作，几个核心贡献：

1. **Geometric Context Attention (GCA)**：unified attention framework with three complementary context types (anchor, window, trajectory memory)，per-frame context growth减少~80×

2. **End-to-end learning of structured SLAM priors**：把classical SLAM的三级context设计变成learnable attention，避免了hand-crafted heuristics

3. **State-of-the-art streaming performance**：在Oxford Spires等benchmark上甚至超越offline和optimization-based方法

4. **Real-time inference**：20 FPS at 518×378, 处理10,000+ frames

5. **Solid engineering**：Paged KV-cache, FlashInfer, context parallelism, two inference modes

6. **Comprehensive training recipe**：Two-stage curriculum, progressive view training, foldback video sampler, 29 datasets

最重要的是，LingBot-Map给我们一个清晰的设计哲学：**streaming state应该selectively retain what matters most，而不是retain how much**。这个selection应该grounded in geometric priors且end-to-end learned。这个principle我相信会influence future streaming 3D reconstruction的work。

## References

- [LingBot-Map Project](https://technology.robbyant.com/lingbot-map)
- [LingBot-Map GitHub](https://github.com/robbyant/lingbot-map)
- [VGGT](https://vggt.github.io/)
- [DUSt3R](https://dust3r.europe.naverlabs.com/)
- [DROID-SLAM](https://princeton-vl.github.io/DROID-SLAM/)
- [DINOv2](https://dinov2.metademolab.com/)
- [FlashInfer](https://flashinfer.ai/)
- [vLLM/PagedAttention](https://arxiv.org/abs/2309.06180)
- [Ulysses Context Parallelism](https://arxiv.org/abs/2309.14509)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [TartanAir](https://theairlab.org/tartanair/)
- [Oxford Spires](https://oxford-spires.lukasbieri.net/)
- [ETH3D](https://www.eth3d.net/)
- [Tanks and Temples](https://www.tanksandtemples.org/)
- [Habitat-Sim](https://aihabitat.org/)
- [CUT3R](https://cut3r.github.io/)
- [StreamVGGT](https://arxiv.org/abs/2504.16318)
- [Stream3R](https://arxiv.org/abs/2506.21436)
- [TTT3R](https://arxiv.org/abs/2506.22385)
- [Depth Anything 3](https://arxiv.org/abs/2511.10647)
- [Fast3R](https://arxiv.org/abs/2501.04661)
- [VGGT-SLAM](https://arxiv.org/abs/2507.07257)
- [MASt3R-SLAM](https://huggingface.co/papers/2412.03941)
- [TorchTitan](https://github.com/pytorch/torchtitan)
