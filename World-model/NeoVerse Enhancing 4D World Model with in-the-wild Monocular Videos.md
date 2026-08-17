---
source_pdf: NeoVerse Enhancing 4D World Model with in-the-wild Monocular Videos.pdf
paper_sha256: 652e3a7f8d9374cb33d5313d989c359db4e793f9bce6a2e270f5fd1953c042cc
processed_at: '2026-08-05T22:09:56-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# NeoVerse 人话版

## 一句话总结

之前做 4D world model 的人都被数据卡死了，要么用死贵死少的多相机同步视频，要么离线跑一遍深度估计/重建管线把训练数据预先准备好。NeoVerse 说"去他妈的，我直接用网上爬的单目视频训"，为了让这件事可行，他们搞了三个工程 trick：feed-forward 4DGS、bidirectional motion、online degradation simulation。

---

## 先说清楚之前为什么难

Karpathy 你想想 4D world model 这个 task 想干嘛：给一段视频，要能重建出 3D 场景、还能生成任意新视角的新视频。reconstruction 负责给出几何约束（哪是墙、哪是人、camera 怎么动），generation 负责把几何渲染可能不准、可能缺失的部分用 diffusion prior 补全。这就是所谓 reconstruction-generation hybrid。

问题是训练数据怎么来。两条旧路：

**路 A：多相机同步采集**。SynCamMaster、ReCamMaster、CamCloneMaster 这类（https://arxiv.org/abs/2503.07777）。搞几个相机架在同一个场景同时拍，天然就有多视角 GT。问题是你能拍到的场景有限，不能 scale 到 internet 上任何视频。domain 窄，泛化差。

**路 B：单目视频 + 离线预处理**。TrajectoryCrafter（https://arxiv.org/abs/2503.05859）、FreeSim（https://arxiv.org/abs/2503.07777）、GEN3C（https://arxiv.org/abs/2503.16918）这类。拿任意单目视频，先用 DepthCrafter（https://arxiv.org/abs/2409.02048）这种重量级 depth estimator 跑一遍深度，或者先用 3DGS 离线重建一遍场景，把"重建得到的几何 condition"和"原始 RGB"凑成训练对。问题：1M 视频跑离线预处理要花巨量 GPU 时间和存储；训练时想做 augmentation 没法做；调个超参要重跑预处理。

NeoVerse 的洞察：**预处理搬到 online**，每个 training iteration 现场重建、现场生成 condition、现场监督。这个改动看似简单，但需要 reconstruction 部分足够快、condition 足够真实，否则训练根本跑不动。

---

## 三个核心 trick 人话版

### Trick 1: Feed-forward 4DGS（把 VGGT 改造成动态 Gaussian 输出器）

VGGT（https://vgg-t.github.io）是 2025 年 CVPR 的 geometry foundation model，single forward pass 输出 depth、point map、camera pose，特别猛。但它是 3D 的、temporal-unaware 的。NeoVerse 在它上面加了一个 motion branch：

- 用 DINOv2（https://github.com/facebookresearch/dinov2）提 per-frame feature
- frame feature 过 VGGT 的 alternating attention
- 再过一层 bidirectional cross-attention：前向 $F_t \to F_{t+1}$，后向 $F_t \to F_{t-1}$
- 输出每个 pixel 一个 4D Gaussian，含 position、opacity、rotation、scale、SH、life span、还有 forward/backward velocity 和 angular velocity

公式 (1) 就是两个 cross-attention：

$$
F_t^{fwd} = \text{CrossAttn}(q=F_t;\ k,v=F_{t+1})
$$
$$
F_t^{bwd} = \text{CrossAttn}(q=F_t;\ k,v=F_{t-1})
$$

直觉：每个 Gaussian 不仅知道"我现在在哪"，还知道"我下一帧要去哪"和"我上一帧从哪来"。这跟 4DGT（https://arxiv.org/abs/2506.08015）不一样，4DGT 只能往前推，NeoVerse 两边都能推。

### Trick 2: Bidirectional Motion 用来做关键帧插值

训练时如果每帧都跑 feed-forward 重建，太慢。NeoVerse 只取稀疏关键帧（11-21 帧）做重建，然后用 bidirectional motion 把 Gaussian 插值到任意中间时间戳 $t_q$。

公式 (3) 位置插值：

$$
\mu_i(t_q) = \mu_i + v_i^{\pm} |t_q - t|
$$

$t$ 是最近关键帧时间戳，$t_q \ge t$ 用 forward velocity $v^+$，$t_q < t$ 用 backward velocity $v^-$。线性假设。

公式 (4) rotation 插值用 quaternion 复合：

$$
r_i(t_q) = r_i \cdot \phi(\omega_i^{\pm} |t_q - t|)
$$

$\phi$ 把 axis-angle 转 quaternion。

公式 (5) opacity 用一个很巧的衰减：

$$
\alpha_i(t_q) = \alpha_i \exp(-\gamma \cdot d(t_q, t)^{1/(1-\tau_i)})
$$

$\tau_i \in (0,1)$ 是 life span。$\tau_i \to 1$ 时指数趋于 0，opacity 不衰减（永久物体，比如墙）；$\tau_i$ 小时 opacity 快速衰减（瞬态物体，比如路过的人）。$d(t_q, t) = |t_q - t| / |T_{k+1} - T_k|$ 是归一化时间距离，处理关键帧间隔不均匀的情况。

直觉：life span 这个标量把"这个 Gaussian 是不是短暂存在"压进一个数，避免瞬态物体在时间外推时变 ghost。

这个 trick 让 reconstruction 时间从全帧 10s 降到 11 帧 2s，速度 5x，而且 Table 3 显示 image quality 几乎不降。

### Trick 3: Online Monocular Degradation Simulation（最聪明的一招）

单目视频没多视角 GT，怎么造"degraded rendering → high-quality RGB"训练对？NeoVerse 提了三种 degradation pattern，全部 first-principles 推导：

**Pattern A: Visibility-based Gaussian Culling（模拟 occlusion）**
- 用预测的 camera trajectory 做随机 transform 得到 novel trajectory
- 用 depth 判断哪些 Gaussian 在新视角下被遮挡
- cull 掉它们，把剩下的 render 回原始视角 → 出现空洞
- 这就是 novel view 渲染时 occlusion 看起来的样子

**Pattern B: Average Geometry Filter（模拟 flying pixels）**
- depth 估计在 depth discontinuity 边缘会输出平均深度（因为网络最小化 L2 loss）
- 直接模拟：在 transformed novel trajectory 上渲染 depth，apply average filter，再根据 filtered depth 调整 Gaussian 位置
- render 回原始视角 → 出现 flying pixel pattern

**Pattern C: 更大 kernel filter**
- 用更大 average filter kernel → 模拟更大空间范围的 depth error 引起的 distortion

paper 在 Supplementary F 里很坦诚地讨论了 linear motion assumption 的局限，说"非关键帧渲染不精确"本身就是一种 temporal degradation，让 generation model 学会从这种烂渲染重建非线性运动。这个 insight 真的很 Karpathy-style：don't fix it, embrace it as a feature。

condition 最终包含四模态：RGB 渲染、depth map、opacity mask（二值化表示空区域）、Plücker embedding（编码原始 camera motion，https://arxiv.org/abs/2504.14899）。通过 control branch 注入 Wan-T2V 14B（https://arxiv.org/abs/2503.20314），类似 ControlNet（https://github.com/lllyasviel/ControlNet）的思路。

**关键设计**：generation model 冻结，只训 control branch。这样可以直接挂载到 LoRA 蒸馏加速的 Wan 上（https://arxiv.org/abs/2106.09685），推理速度 18s vs TrajectoryCrafter 121s。

---

## 训练目标

Reconstruction loss（Eq. 6）：

$$
\mathcal{L}_{recon} = \mathcal{L}_{rgb} + 5\mathcal{L}_{camera} + \mathcal{L}_{depth} + \mathcal{L}_{motion} + 0.1\mathcal{L}_{regular}
$$

- $\mathcal{L}_{rgb}$：L2 + LPIPS（https://arxiv.org/abs/1801.03924）
- $\mathcal{L}_{camera}$：camera pose 监督
- $\mathcal{L}_{depth}$：predicted depth + rendered depth
- $\mathcal{L}_{motion}$：双向 velocity 监督，GT 来自 DynamicReplica、Kubric、PointOdyssey 这种带 3D flow 的合成数据
- $\mathcal{L}_{regular} = \sum_i |1 - A_i|$：防止网络走捷径把 Gaussian 变 transparent

Generation loss（Eq. 7）用 Rectified Flow（https://arxiv.org/abs/2403.03206）：

$$
\mathcal{L}_{gen} = \mathbb{E}\|f_\theta(x_t, t, c_{render}, c_{text}) - v_t\|^2
$$

$x_t$ 是 $x_1$（video latent）和 $x_0$（noise）的线性插值，$v_t = x_1 - x_0$ 是 ground-truth velocity。这是 Flow Matching 的标准形式。

---

## Inference 的 global motion tracking trick

这个细节很容易被忽略但很重要。一个 Gaussian 在某个时刻瞬时速度可能是 0（比如走路的人停下脚步），但整体应该归类为 dynamic。如果只看瞬时 velocity 做静态/动态分离，会错判。

NeoVerse 的做法（Eq. 8）：

$$
m_i = \max_t \mathbb{1}(d_{i,t} \le D_t[p_{i,t}]) \cdot m_{i,t}
$$

- 把每个 Gaussian 投影到所有帧
- 检查 visibility（投影 depth $\le$ 该帧该 pixel 的 depth）
- 取所有可见帧中 velocity magnitude 的 max
- 用阈值 $\eta$ 分到 static / dynamic set

直觉：max-over-time 替代 last-frame 或 mean。只要这个 Gaussian 在任何可见时刻动过，就归 dynamic。处理"走走停停"对象。

然后 static 跨所有帧聚合得到完整背景，dynamic 只在邻近几帧聚合避免 motion drift。

---

## 实验数据的直觉解读

**Table 1 静态重建**：VRNeRF 上 PSNR 20.73 vs AnySplat 18.02，高 2.71 dB。Scannet++ 上 25.34 vs 22.79，高 2.55 dB。VGGT + dynamic branch 比 AnySplat 专门为 unconstrained view 设计的还强。

**Table 2 动态重建**：ADT 上 32.56 vs 4DGT 30.09（4DGT 还需要 camera pose 输入，NeoVerse 是 pose-free 的），高 2.47 dB。

**Table 3 VBench**：最关键是效率。81 帧总耗时，NeoVerse 28s vs ReCamMaster 168s，快 6 倍。Recon 时间随关键帧数线性 scale：11→2s, 21→3s, 41→5s, full→10s。Image quality 也是 NeoVerse 最高（61.51）。

**Table 4 Ablation**：
- w/o Regularization: 10.86 → Reconstruction: 11.56（regularization 贡献 +0.7 dB）
- w/o Bidirectional: 11.27 → Reconstruction: 11.56（bidirectional 贡献 +0.3 dB，但 PSNR 体现不出 interpolation 优势）
- Reconstruction only 11.56 → w/ Generation 14.59（**+3 dB！** 这个差距巨大）

最后这个 +3 dB 是 paper 的灵魂证据：generation model 作为 universal geometry-to-RGB prior，能修复 reconstruction 的 artifact。这跟 Difix3D+（https://arxiv.org/abs/2503.03282）发现的"diffusion 能当 3D 重建的 prior teacher"是同一个现象。

---

## 我个人觉得最 elegant 的地方

**Degradation simulation 的方向反了**。传统 sim-to-real 是 synthetic 上训、real 上用。NeoVerse 反过来：训练 GT 是 real video，condition 是从 real video 合成的 degraded rendering。本质是让模型学"如何把 artifact-prone 几何渲染变成 high-fidelity RGB"，这个反演映射正好是 diffusion 擅长的。这个反向 sim2real 的思路很妙，可以推广到很多 unsupervised task。

**Life span $\tau$ 一个标量表达 occlusion-aware**。短寿命 Gaussian 对应瞬态物体，长寿命对应永久结构。opacity 衰减公式 (5) 用 $\tau$ 控制衰减速度，把"是否短暂存在"压成一个数。这种 minimal parameterization 很 Karpathy 美学。

**Bidirectional 是 slow motion 的 enabler**。单向 motion model 在 bullet time 任务里要循环外推、误差累积；双向直接两边内插、error bounded。这跟 video diffusion 里 bidirectional noise scheduling 思路一致。

---

## 局限

- 2D cartoon 失效（Figure S1）：VGGT-style 几何先验在缺乏真实 3D 结构的数据上崩
- 文字渲染仍然不 legible
- 1M clip 数据集相对 internet 总量还是小
- Linear motion assumption 在快速运动/大形变下可能不足，但 paper 通过"非关键帧渲染不精确作为 temporal degradation"绕过了
- Feed-forward 4DGS 对 T×H×W 个 pixel 都生成 Gaussian，长视频 memory 爆炸，sparse keyframe 只是缓解

---

## Reference Links

主 paper: https://neoverse-4d.github.io  
VGGT: https://vgg-t.github.io / https://arxiv.org/abs/2503.19551  
4DGT: https://arxiv.org/abs/2506.08015  
DUSt3R: https://arxiv.org/abs/2312.14132  
DINOv2: https://github.com/facebookresearch/dinov2  
3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/  
GSplat: https://github.com/nerfstudio-project/gsplat  
ViewCrafter: https://arxiv.org/abs/2409.02048  
TrajectoryCrafter: https://arxiv.org/abs/2503.05859  
ReCamMaster: https://arxiv.org/abs/2503.07777  
AnySplat: https://arxiv.org/abs/2505.23716  
NoPoSplat: https://arxiv.org/abs/2503.06705  
StreamSplat: https://arxiv.org/abs/2506.08862  
MoVieS: https://arxiv.org/abs/2507.10065  
Wan 2.1: https://arxiv.org/abs/2503.20314  
Rectified Flow: https://arxiv.org/abs/2403.03206  
LoRA: https://arxiv.org/abs/2106.09685  
ControlNet: https://github.com/lllyasviel/ControlNet  
VACE: https://arxiv.org/abs/2503.07598  
SAM 2: https://arxiv.org/abs/2408.00714  
TAPVid-3D: https://arxiv.org/abs/2511.16770  
SpatialTracker: https://arxiv.org/abs/2404.19078  
ST4RTrack: https://arxiv.org/abs/2507.02621  
Difix3D+: https://arxiv.org/abs/2503.03282  
LPIPS: https://arxiv.org/abs/1801.03924  
DepthCrafter: https://arxiv.org/abs/2409.02048  
GEN3C: https://arxiv.org/abs/2503.16918  
FreeSim: https://arxiv.org/abs/2503.07777  
Uni3C: https://arxiv.org/abs/2504.14899

---

## 最后一句

NeoVerse 没发明什么新数学，但工程上把三件事串得很漂亮：VGGT 做 geometry foundation、4D Gaussian 做 explicit representation、Wan-T2V 做 implicit video prior，用 degradation simulation 当胶水。让 in-the-wild monocular video 终于能直接 scale 进 4D world model training。这种"把多个 foundation model 串成可 scale pipeline"的工程能力，才是 2025 年这个阶段真正稀缺的东西。

---

# NeoVerse: 用 in-the-wild 单目视频构建可扩展的 4D World Model

## 一、Paper 的核心 motivation 与定位

这篇 paper 来自 CASIA 与 CreateAI，作者 Yuxue Yang、Lue Fan 等。核心要解决的问题是 **4D world model 的 scalability 瓶颈**。Karpathy 你一直在强调 scale 是 deep learning 的灵魂，这篇 paper 在这个问题上做出了很有意思的设计选择。

作者把 scalability 瓶颈拆成两层：

**Data scalability**：早期 reconstruction-guided video generation 方法被两类数据卡死：
- ViewCrafter [88]（https://arxiv.org/abs/2409.02048）这类只能用 static scene 的 multi-view 视频做训练数据，没法扩展到动态 4D。
- SynCamMaster [3]、ReCamMaster [2]（https://arxiv.org/abs/2503.07777）依赖同步多相机采集的 dynamic video，数据获取成本极高，domain 窄。

**Training scalability**：TrajectoryCrafter [87]（https://arxiv.org/abs/2503.05859）、FreeSim [18] 这些方法虽然能用更灵活的数据，但每次训练前都要跑一遍离线 depth estimation 或重建管线（DepthCrafter、3D 检测器等），storage 和 compute 都极其昂贵，而且 online augmentation 基本上不能做。

NeoVerse 的核心哲学：**把整条 pipeline 改造得能直接吃 in-the-wild monocular video**，这样就可以利用海量 internet video（paper 里他们爬了 1M clip）来 scale。

---

## 二、整体架构与三个核心设计

Figure 2 展示了 pipeline：

```
Monocular Video → Sparse Key Frames → Feed-forward 4DGS (VGGT-based) 
                                      ↓
                      Degraded Rendering from Novel Trajectory
                                      ↓ (condition)
                      Video Diffusion (Wan-T2V 14B + ControlNet-like branch)
                                      ↓
                      High-quality Novel-view Video
```

三个关键设计：

1. **Pose-free Feed-forward 4DGS**：基于 VGGT [66]（https://vgg-t.github.io/）做动态化，加 bidirectional motion modeling，输出 4D Gaussians。
2. **Bidirectional Motion Modeling**：区分前向/后向速度，用于非关键帧的 Gaussian 时间插值。
3. **Online Monocular Degradation Simulation**：用三个 first-principles 的技巧在单目视频上合成 degraded rendering，与原始 frame 配对，做训练监督。

---

## 三、Pose-Free Feed-Forward 4DGS：把 VGGT "Gaussianize"

### 3.1 VGGT 回顾

VGGT 是 CVPR 2025 的工作，https://arxiv.org/abs/2503.19551，它的核心是用一个 transformer 在 single forward pass 内同时输出 depth、point map、camera pose 等几何量。它用 DINOv2 [54]（https://github.com/facebookresearch/dinov2）提取 per-frame feature，然后送入 Alternating-Attention blocks 做 frame-wise aggregation，得到 frame features $\{F_t\}$。

VGGT 本身是 3D 的、temporal-unaware 的，所以 NeoVerse 在它上面加了动态分支。

### 3.2 Bidirectional Motion Encoding (Eq. 1)

给定 frame features $\{F_t\}_{t=1}^{T}$，做两次 cross-attention：

$$
\{F_t^{fwd}\}_{t=1}^{T-1} = \text{CrossAttn}(q=\{F_t\}_{t=1}^{T-1};\ k,v=\{F_t\}_{t=2}^{T})
$$

$$
\{F_t^{bwd}\}_{t=2}^{T} = \text{CrossAttn}(q=\{F_t\}_{t=2}^{T};\ k,v=\{F_t\}_{t=1}^{T-1})
$$

变量解释：
- $F_t^{fwd}$：从 timestamp $t$ 到 $t+1$ 的 forward motion feature
- $F_t^{bwd}$：从 timestamp $t$ 到 $t-1$ 的 backward motion feature
- $q$：query；$k, v$：key/value

直觉：区别 $t \to t+1$ 与 $t \to t-1$ 的瞬时 velocity，这听上去只是双向化，但实际意义是让 Gaussian 在时间轴上具有"内插"能力。4DGT [78]（https://arxiv.org/abs/2506.08015）是 uni-directional 的，意味着只能从前一帧推后一帧；而 NeoVerse 双向化后，任意中间时间戳 $t_q$ 都可以从前后两个方向的高斯插值过来，这在生成慢动作、bullet-time 这类任务上是直接刚需。

Decoder block 用 DUSt3R [71]（https://arxiv.org/abs/2312.14132）的结构：self-attention 做 intra-frame spatial modeling，cross-attention 做 inter-frame temporal modeling，最后两个 DPT [56] heads 分别预测前向和后向运动。

### 3.3 4D Gaussian 参数化 (Eq. 2)

每个 Gaussian i 参数：

$$
(\mu_i, \alpha_i, r_i, s_i, sh_i, \tau_i, v_i^+, v_i^-, \omega_i^+, \omega_i^-)
$$

- $\mu_i$：3D 位置（通过 backproject 像素 depth + camera 参数得到）
- $\alpha_i$：opacity
- $r_i$：rotation（quaternion）
- $s_i$：scale
- $sh_i$：spherical harmonics coefficients（颜色）
- $\tau_i$：life span，用 sigmoid 约束到 (0, 1)
- $v_i^+, v_i^-$：forward / backward linear velocity（3D vector）
- $\omega_i^+, \omega_i^-$：forward / backward angular velocity（axis-angle）

总共 $T \times H \times W$ 个 Gaussian，每个像素一个。

这里有个直觉值得讲：4D Gaussian 不需要复杂数学，就是 3D Gaussian 加上 time-aware 属性。每个 Gaussian 不仅"知道"自己现在在哪，还"知道"自己在 $t \to t+1$ 和 $t \to t-1$ 两个方向上的运动，所以可以在任意时间戳上 self-interpolate。

### 3.4 关键帧 Gaussian 时间插值 (Eq. 3, 4, 5)

这是 paper 里我最喜欢的一个工程细节。给定一个非关键帧查询时间戳 $t_q$，找到最近的关键帧时间戳 $t$，把 Gaussian $i$ 转移到 $t_q$：

**Position (Eq. 3)**：

$$
\mu_i(t_q) = \begin{cases} \mu_i + v_i^+ |t_q - t|, & t_q \ge t \\ \mu_i + v_i^- |t_q - t|, & t_q < t \end{cases}
$$

**Rotation (Eq. 4)**：

$$
r_i(t_q) = \begin{cases} r_i \cdot \phi(\omega_i^+ |t_q - t|), & t_q \ge t \\ r_i \cdot \phi(\omega_i^- |t_q - t|), & t_q < t \end{cases}
$$

其中 $\phi(\cdot)$ 把 axis-angle 转为 quaternion，与基础旋转 $r_i$ 复合。

**Opacity (Eq. 5)**：

$$
\alpha_i(t_q) = \alpha_i \exp\left(-\gamma \cdot d(t_q, t)^{\frac{1}{1-\tau_i}}\right)
$$

$$
d(t_q, t) = \frac{|t_q - t|}{|T_{k+1} - T_k|} \le 1
$$

变量解释：
- $\gamma$：控制衰减速度的超参
- $\tau_i \in (0,1)$：life span
- $[T_k, T_{k+1}]$：包含 $t_q$ 的关键帧区间
- $d(t_q, t)$：normalized temporal distance

直觉：当 $\tau_i \to 1$ 时，$1/(1-\tau_i) \to \infty$，对于 $d < 1$ 有 $d^{1/(1-\tau_i)} \to 0$，exp 趋近 1，所以 $\alpha_i(t_q) \approx \alpha_i$，Gaussian 持久稳定。当 $\tau_i$ 小（短寿命，比如瞬态物体、遮挡区域），opacity 快速衰减。这个设计很 elegant：把"这个 Gaussian 是否在长时间内有效"压缩成一个 life span 标量。

**First-order linear motion assumption**：在相邻关键帧之间，paper 假设运动近似线性。这是简化假设，但 paper 在 Supplementary F 中专门讨论：训练时用稀疏关键帧重建，渲染到全部帧，自然把"非关键帧渲染不准确"作为一种 temporal degradation 注入训练，让 generation model 学会从 degraded rendering 重建非线性运动；推理时如果需要精确渲染，可以让所有帧都做 keyframe。

---

## 四、Monocular Degradation Simulation：训练数据合成的关键

这是 paper 的核心 insight。在多视角数据（ViewCrafter）或静态场景（ViewCrafter）下，可以直接拿到 degraded rendering + GT pair。但单目视频没有 GT，所以必须构造。

paper 提出三种退化模式（对应 Figure 3 a/b/c）：

### 4.1 Visibility-based Gaussian Culling (occlusion simulation)

- 用 sparse key frame 预测的 camera trajectory 做随机 transform 得到 novel trajectory（约束新视角大致指向 scene center）。
- 用 depth 检查从新视角下被遮挡的 Gaussians。
- Cull 掉不可见 Gaussians，把剩下的 render 回原始视角 → 形成空洞，模拟 occlusion pattern。

### 4.2 Average Geometry Filter (flying-edge-pixel simulation)

第二个退化更微妙。Karpathy 你应该对 depth estimation 的"flying pixels"现象熟悉：在 depth discontinuity 边缘，网络为了最小化回归 loss，会输出平均 depth（这个现象在 Pixel-Perfect Depth [76] 中也确认过）。NeoVerse 直接模拟这种现象：

1. 在 transformed novel trajectory 上渲染 depth map。
2. 对 rendered depth map 应用 average filter。
3. 根据 filtered depth 调整每个 Gaussian 的中心位置。
4. 重新 render 回原始视角 → flying-pixel pattern 出现。

**关键直觉**：直接 first-principles 推导，不依赖跑一个大 depth estimator。Average filter 是数学上的低通，等价于"边缘像素的 depth 取邻域均值"，这恰好就是网络在 edge 处的行为。用更大的 kernel 模拟更广的 spatial distortion（对应 depth error 范围更大）。

### 4.3 三种退化合并

最终训练样本：原始单目 video 作为 GT，degraded rendering 作为 condition。

condition 包含四个模态：
1. RGB 渲染图
2. Depth map
3. Mask（opacity map 二值化，表示空区域）
4. Plücker embeddings（来自原始 trajectory，显式编码 3D camera motion [9]）

这些通过一个 control branch 注入 generation model，类似 ControlNet [90] / VACE [33] 的设计。

---

## 五、Reconstruction-Guided Generation 训练

### 5.1 Reconstruction Loss (Eq. 6)

$$
\mathcal{L}_{recon} = \mathcal{L}_{rgb} + \lambda_1 \mathcal{L}_{camera} + \lambda_2 \mathcal{L}_{depth} + \lambda_3 \mathcal{L}_{motion} + \lambda_4 \mathcal{L}_{regular}
$$

- $\mathcal{L}_{rgb}$：photometric loss，含 $L_2$ + LPIPS [91]
- $\mathcal{L}_{camera}$：camera pose 监督（VGGT 风格）
- $\mathcal{L}_{depth}$：predicted depth + rendered depth 双重监督
- $\mathcal{L}_{motion} = \sum_i \|\hat{v}_i^+ - v_i^+\| + \|\hat{v}_i^- - v_i^-\|$：双向 velocity 监督，GT 来自 DynamicReplica、Kubric、PointOdyssey 等带 3D flow 的数据集
- $\mathcal{L}_{regular} = \sum_i |1 - A_i|$，$A_i$ 是 rendered accumulated opacity，防止网络走捷径把 Gaussian 变 transparent（与预定义背景色匹配的区域）

权重：$\lambda_1 = 5.0, \lambda_2 = 1.0, \lambda_3 = 1.0, \lambda_4 = 0.1$。

### 5.2 Generation Loss (Eq. 7)

采用 Rectified Flow [16]（https://arxiv.org/abs/2403.03206）训练 Wan-T2V 14B [65]（https://arxiv.org/abs/2503.20314）：

$$
\mathcal{L}_{gen} = \mathbb{E}_{x_1, x_0, c_{render}, c_{text}, t} \|f_\theta(x_t, t, c_{render}, c_{text}) - v_t\|_2^2
$$

- $x_1$：video latent
- $x_0 \sim \mathcal{N}(0, I)$：噪声
- $x_t$：$x_1$ 与 $x_0$ 在时间 $t$ 的线性插值
- $v_t = x_1 - x_0$：ground-truth velocity
- $c_{text}$：umT5 [14] 提取的文本 embedding
- $c_{render}$：degraded rendering condition
- $f_\theta$：去噪网络

**只训 control branch，冻结 base model**：这样做的关键好处是 NeoVerse 可以直接挂载在 LoRA [24]（https://arxiv.org/abs/2106.09685）蒸馏的 Wan 上加速推理（speed-up distillation），不用重训。Table 3 显示 NeoVerse 总推理 20-28 秒（81 帧）vs TrajectoryCrafter 146 秒 vs ReCamMaster 168 秒，差距巨大。

### 5.3 训练细节

- 32×A800 GPU
- Stage 1 (Reconstruction): 150K iterations，cosine LR schedule，peak LR 1e-4，5K warmup
- Stage 2 (Generation): 50K iterations，constant LR 1e-5
- 输入分辨率 336×560，81 frames
- Reconstruction 训练每次采样 2-8 个关键帧 + N-1 个中间帧，loss 在所有 2N-1 帧上算
- Temporal reverse augmentation with p=0.5
- Generation 训练用 11-21 关键帧做 on-the-fly 重建
- Mask drop with p=0.2（全置 0 表示全图需要 inpainting，增强鲁棒性）

---

## 六、Inference：Global Motion Tracking 与 Aggregation

### 6.1 Global Motion Tracking (Eq. 8)

这里有一个很重要的设计。一个 Gaussian 在某些时刻瞬时速度可能为 0（比如行人停下），但整体应该归类为 dynamic。如果只用瞬时 velocity 做分离，会漏判。

paper 的做法：

$$
m_{i,t} = \max\{\|V_t^+[p_{i,t}]\|_2, \|V_t^-[p_{i,t}]\|_2\}
$$

$$
m_i = \max_{t=1,\ldots,T} \mathbb{1}(d_{i,t} \le D_t[p_{i,t}]) \cdot m_{i,t}
$$

- $\bar{P}_t$：world-to-camera pose
- $K_t$：intrinsics
- $\mu_i$：Gaussian 中心
- $p_{i,t}$：投影到帧 t 的像素坐标
- $d_{i,t}$：投影深度
- $D_t[p_{i,t}]$：帧 t 在 $p_{i,t}$ 处的采样 depth
- $V_t^+[p_{i,t}]$：在 $p_{i,t}$ 处采样的前向 velocity
- $\mathbb{1}(\cdot)$：可见性指示
- $m_i$：跨所有帧的可见性加权最大速度

用阈值 $\eta$ 把 Gaussians 分到 static set $S$ 和 dynamic set $D$。

**直觉**：用 max-over-time 替代 mean 或 last-frame，只要这个 Gaussian 在任何可见时刻动过，就归为 dynamic。这处理了"走走停停"的对象。

### 6.2 Temporal Aggregation

- Static part：跨所有帧聚合（形成更完整的背景）。
- Dynamic part：只在邻近几帧聚合，避免 motion drifting。

### 6.3 时序插值

中间时间戳的 Gaussian 插值用 Sec 3.2 同样的 Eq. 3-5。用于 slow motion、bullet time shots。

---

## 七、实验数据表深度解读

### Table 1：静态重建（VRNeRF & Scannet++）

| Method | VRNeRF PSNR/SSIM/LPIPS | Scannet++ PSNR/SSIM/LPIPS |
|---|---|---|
| NoPoSplat [83] | 11.27 / 0.408 / 0.620 | 8.69 / 0.312 / 0.614 |
| Flare [92] | 12.62 / 0.597 / 0.623 | 12.19 / 0.619 / 0.611 |
| AnySplat [32] | 18.02 / 0.705 / 0.366 | 22.79 / 0.773 / 0.217 |
| **Ours** | **20.73 / 0.766 / 0.352** | **25.34 / 0.834 / 0.195** |

VRNeRF 上比 AnySplat 高 2.71 dB，Scannet++ 上高 2.55 dB。这个差距很大，说明 VGGT backbone + dynamic branch 的设计在静态场景上也强于专门为 unconstrained view 设计的 AnySplat。

### Table 2：动态重建（ADT & DyCheck）

| Method | ADT PSNR/SSIM/LPIPS | DyCheck PSNR/SSIM/LPIPS |
|---|---|---|
| MonST3R [89] | 17.42 / 0.554 / 0.534 | 9.32 / 0.103 / 0.710 |
| 4DGT† [78] | 30.09 / 0.909 / 0.178 | 9.94 / 0.208 / 0.639 |
| **Ours** | **32.56 / 0.927 / 0.120** | **11.56 / 0.293 / 0.558** |

4DGT 标 † 表示需要 camera pose 作为输入，NeoVerse 是 pose-free 的。ADT 上高 2.47 dB，DyCheck 上高 1.62 dB。DyCheck 是单目动态重建最难的 benchmark 之一，提升明显。

### Table 3：VBench 新视角生成 + 推理效率

| Method | Frames | Recon Time | Gen Time | Total | Aesth. | Imag. Quality |
|---|---|---|---|---|---|---|
| TrajectoryCrafter | 49 | 25s | 121s | 146s | 44.63 | 54.59 |
| ReCamMaster | 81 | - | 168s | 168s | 44.29 | 58.87 |
| Ours (11 key) | 81 | 2s | 18s | 20s | 44.55 | 59.75 |
| Ours (21 key) | 81 | 3s | 18s | 21s | 44.59 | 60.01 |
| Ours (41 key) | 81 | 5s | 18s | 23s | 44.89 | 60.37 |
| Ours (full) | 81 | 10s | 18s | 28s | 44.78 | 61.51 |

关键观察：
- Recon 时间几乎线性 scale 关键帧数：11→2s，21→3s，41→5s，full→10s，证明 sparse-keyframe reconstruction 的效率优势
- Gen 时间恒定 18s，因为 control branch 速度快且支持 distillation
- Total 28s 比 ReCamMaster 168s 快 6 倍，比 TrajectoryCrafter 146s 快 5 倍
- Image quality 也最高

### Table 4：Ablation on DyCheck

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| w/o Regularization | 10.86 | 0.244 | 0.576 |
| w/o Bidirectional Motion | 11.27 | 0.285 | 0.570 |
| Reconstruction only | 11.56 | 0.293 | 0.558 |
| w/ Generation (full) | 14.59 | 0.323 | 0.501 |

直觉：
- Regularization（防止 transparent shortcut）带来 0.7 dB
- Bidirectional motion 带来 0.3 dB（看起来不大，但 paper 强调它对 interpolation 任务的必要性，PSNR 体现不出来）
- Generation 带来 +3 dB（关键发现：diffusion prior 修复了 reconstruction 的 artifact）

这个 ablation 印证了 paper 的核心论点：generation model 作为 "the missing ingredient" 能补足 reconstruction 的不完美，而 degradation simulation 训练让 generation 学会信任几何条件而非盲信，避免把 artifact 复制过来。

### Table S2：Video Editing (FiVE)

| Method | Struct Dist ↓ | CLIP ↑ | NIQE ↓ | Sec/Frame ↓ |
|---|---|---|---|---|
| AnyV2V | 0.071 | 24.89 | 5.04 | 6.11 |
| Wan-Edit | 0.013 | 26.39 | 6.54 | 3.07 |
| VACE | 0.015 | 26.92 | 4.37 | 4.30 |
| Ours | 0.018 | 26.66 | 5.13 | **0.49** |

结构距离略大，但 sec/frame 是其他方法的 6-12 倍快。

### Table S3：3D Tracking (TAPVid-3D DriveTrack)

| Method | APD (δ3D=0.1m) ↑ | EPE ↓ |
|---|---|---|
| SpatialTracker | 3.79 | 3.35 |
| St4RTrack | 2.47 | 5.64 |
| Ours | **7.31** | **3.10** |

APD (average percentage of durable tracks) 几乎是 SpatialTracker 的两倍。这表明 bidirectional motion + 4D Gaussian 本身就是一个很强的 3D tracker。

---

## 八、训练数据组成（Table S1）

paper 把数据分五组：

- Group ①：Dynamic + 3D flow（PointOdyssey 131 clip、DynamicReplica 483、Kubric 5.7K、Spring 37）→ 用于 velocity 监督
- Group ②：Dynamic + depth + pose（TartanAir 50、BEDLAM 798、Panoptic 81）
- Group ③：Dynamic + partial 3D（HOI4D 3.0K、CoP3D 2.8K）
- Group ④：Static（DL3DV 6.4K、Scannet++ 853、ARKitScenes 4.5K、HyperSim 457）
- Group ⑤：Monocular videos（SpatialVID 371K + 自爬 1M）

①-④ 训 reconstruction，⑤ 训 generation。这是关键设计：reconstruction 需要几何 GT，所以用合成/室内数据；generation 只需要 RGB，所以直接吃海量 in-the-wild 视频。这是 paper scalability 论点的实际体现。

---

## 九、下游应用与 intuition

- **3D Tracking**：用 4D Gaussian 的预测 3D flow 关联相邻帧最近 Gaussian，paper 在 TAPVid-3D 上 APD=7.31，比专门的 tracker 都好。
- **Video Editing**：结合 SAM 2 [57]（https://arxiv.org/abs/2408.00714）做 segmentation mask，再用 mask condition + text condition 做编辑。Figure 10 把白车改红车、镜面壶改透明。
- **Video Stabilization**：smooth 预测的 camera trajectory 后重新 render + generate。
- **Video Super-Resolution**：Gaussian representation 支持任意分辨率渲染，渲染大尺寸 + diffusion 增细节。
- **Background Extraction**：用 global motion tracking 分离 static Gaussians，自然得到背景。
- **Image to World**（Figure S2）：单图起步，iteratively 生成新视角 → 重建更大 Gaussian scene，相当于 3D outpainting。
- **Single-view to Multi-view**（Figure S3）：iteratively 应用 NeoVerse。

---

## 十、与相关工作的对比 intuition

- **vs ViewCrafter** [88]：ViewCrafter 用 static 场景做训练数据，能生成新视角，但 dynamic 场景做不了。NeoVerse 用 4D 重建处理 dynamic。
- **vs TrajectoryCrafter** [87]：TrajectoryCrafter 思路类似（reconstruction + diffusion），但离线用 DepthCrafter 处理训练数据，数据 scale 受限；其 generation 在大 camera motion 下出现 ghosting artifact。NeoVerse 在线处理 + degradation simulation 直接抑制这种 artifact。
- **vs ReCamMaster** [2]：纯生成方法，visual quality 好但 trajectory controllability 差。Figure 4 是 paper 最有说服力的对比。
- **vs 4DGT** [78]：4DGT 是 pose-required 的 uni-directional feed-forward 4DGS。NeoVerse 把它升级为 pose-free + bidirectional，且训练数据多了一两个数量级（4DGT 主要用 posed monocular dataset）。
- **vs AnySplat** [32]：AnySplat 处理 unconstrained long sequence 的 3D 重建，但仍是 3D。NeoVerse 直接做 4D，且 PSNR 在 Scannet++ 上高 2.55 dB。
- **vs MoVieS / StreamSplat** [42, 74]：paper 说这两者不开源且没详细 protocol，没列对比。
- **vs CUT3R** [69]：CUT3R 用 persistent state 做 continuous 3D perception，NeoVerse 的数据集组合参考了 CUT3R，但 backbone 选了 VGGT。

---

## 十一、我自己对这篇 paper 的几点直觉

### 11.1 Bidirectional 是"成本极低但语义丰富"的设计

单向 motion 模型在 slow motion 任务里要循环外推，误差累积；双向直接两边内插，error bounded。这跟 video diffusion 里 bidirectional noise scheduling 类似的思路。

### 11.2 Degradation Simulation 是 unsupervised domain adaptation 的精巧应用

通常 sim-to-real 是"在 synthetic 上训，real 上用"；这里反向了：训练 GT 是 real video，但 condition 是从 real video 合成的 degraded rendering。本质是 sim2real 的 inverse direction，让模型学到"如何把 artifact-prone 几何渲染变成 high-fidelity RGB"，这个反演映射恰好就是生成模型擅长的。

### 11.3 Reconstruction-Generation Hybrid 的二阶效应

Table 4 显示 +3 dB from generation。这暗示一个有趣现象：generation model 已经成为一个 universal geometry-to-RGB prior，能修正几何错位。这跟 Difix3D+ [73]（https://arxiv.org/abs/2503.03282）的发现一致：diffusion model 可以作为 3D 重建的 prior teacher。

### 11.4 Pose-Free 是 scalability 的关键

NoPoSplat / AnySplat 已经证明 pose-free 在静态场景可行，NeoVerse 把它推到 4D。pose-free 意味着可以直接吃 internet video，不需要 COLMAP 之类的离线 SfM。这点比 4DGT 更进一步。

### 11.5 Life Span $\tau$ 的角色

这个参数有点像 4D 中的 "occlusion-aware"机制。短寿命 Gaussian 对应瞬态物体（移动人物、被遮挡区域），插值时 opacity 衰减；长寿命对应永久结构（背景、建筑），插值时保持。这避免了瞬态物体在时间外推时"幽灵"残留。

### 11.6 可能的扩展方向

- **Text-to-4D generation**：当前 NeoVerse 还是 video-conditioned，text 主要作为弱 guidance。可以直接做 text-conditioned 4D scene 生成，类似 4D-DiT dream。
- **Embodied simulation**：精准 trajectory controllability 让 NeoVerse 适合 embodied agent 的 perception simulation。
- **更长时间尺度**：当前 bidirectional linear motion assumption 适合 short interval，长序列可能需要加入 acceleration 或 polynomial motion。
- **物理一致性**：4D Gaussian 没有物理约束（碰撞、重力），未来可能加入 physics-aware motion。
- **Larger backbone**：VGGT 本身不算特别大，scale 上去可能让 4D 重建更准。

---

## 十二、潜在 limitations

- **2D Cartoon 失效**：Figure S1 显示在缺乏真实 3D 几何的卡通素材上失败。这是 VGGT-style 几何先验的固有限制。
- **文字生成失效**：跟大部分 video diffusion 一样，文字渲染仍然不 legible。
- **1M clip 数据集**：相比 internet 上的 video 总量仍小，paper 自己说未来要 scale 更多。
- **Linear motion assumption**：在快速运动/非刚性大形变下可能不足，但 paper 在 Supplementary F 中论证了 "degradation 自然训练" 缓解了这个问题。
- **Memory 限制**：feed-forward 4DGS 对所有 T×H×W 个像素都生成 Gaussian，长视频会 memory 爆炸。当前用 sparse keyframe 缓解，但 fundamentally 没解决。

---

## Reference Links

主 paper：https://neoverse-4d.github.io  
arXiv (假设): 搜索 "NeoVerse 4D World Model" on arXiv  
相关核心工作：
- VGGT: https://vgg-t.github.io / https://arxiv.org/abs/2503.19551  
- 4DGT: https://arxiv.org/abs/2506.08015  
- DUSt3R: https://arxiv.org/abs/2312.14132  
- DINOv2: https://github.com/facebookresearch/dinov2  
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/  
- GSplat (renderer): https://github.com/nerfstudio-project/gsplat  
- ViewCrafter: https://arxiv.org/abs/2409.02048  
- TrajectoryCrafter: https://arxiv.org/abs/2503.05859  
- ReCamMaster: https://arxiv.org/abs/2503.07777  
- AnySplat: https://arxiv.org/abs/2505.23716  
- NoPoSplat: https://arxiv.org/abs/2503.06705  
- StreamSplat: https://arxiv.org/abs/2506.08862  
- MoVieS: https://arxiv.org/abs/2507.10065  
- Wan 2.1 (T2V 14B): https://arxiv.org/abs/2503.20314  
- Rectified Flow: https://arxiv.org/abs/2403.03206  
- LoRA: https://arxiv.org/abs/2106.09685  
- ControlNet: https://github.com/lllyasviel/ControlNet  
- VACE: https://arxiv.org/abs/2503.07598  
- SAM 2: https://arxiv.org/abs/2408.00714  
- TAPVid-3D: https://arxiv.org/abs/2511.16770  
- SpatialTracker: https://arxiv.org/abs/2404.19078  
- ST4RTrack: https://arxiv.org/abs/2507.02621  
- Difix3D+: https://arxiv.org/abs/2503.03282  
- TrajectoryCrafter (project): https://xdannyw.github.io/TrajectoryCrafter/  
- ViewCrafter (project): https://github.com/Drryaniewang/ViewCrafter

---

## 总结直觉

NeoVerse 的关键贡献是 engineering 上的"打通"：把 VGGT 这类几何基础模型与 4D Gaussian 重建 + video diffusion 三者串成一个能直接吃 in-the-wild video 的 pipeline，每个环节都做了 efficient + scalable 的工程优化。技术上的核心 insight 是用 first-principles 的 degradation simulation 反向构造训练对，从而把海量无标签单目视频变成可用的 supervision。这个思路比类似工作的"先离线处理再做训练"高效得多，也是它能 scale 到 1M clip 的根本原因。

理论上更深的 takeaway：4D world model 的本质可能就是 "geometry-conditioned video prior"。NeoVerse 把 geometry 部分用 4D Gaussian 显式化、把 video prior 用 diffusion 隐式化，然后用 degradation simulation 当桥梁。这种 explicit-implicit hybrid 是当前 4D world model 最实用的设计模式。
