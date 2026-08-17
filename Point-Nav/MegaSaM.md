---
source_pdf: MegaSaM.pdf
paper_sha256: 74db10a14a9dfc0e86848af2085d38bd8ba799cf536b1b26f88e569c5f55022c
processed_at: '2026-08-05T17:17:50-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MegaSaM 用人话讲

## 这 paper 到底在干嘛

你拿手机随手拍一段视频，里面有人在跑、有车在开、有树叶在晃 —— paper 想做的就是从这样一段视频里，自动算出来：

1. **相机自己怎么动的**（往前走？往左转？站着不动？）
2. **镜头焦距大概是多少**（手机拍的时候没记录 EXIF 怎么办）
3. **每一帧每个像素离相机多远**（dense depth map）

这三样东西一旦算出来，下游做 3D 重建、做 novel view synthesis、做 AR 特效就有了 foundation。

听起来简单，但这件事过去几十年都没人做得好。

## 为什么这事这么难

想象你在房间里拍一段视频，三种糟糕情况：

**糟糕情况一**：房间里孩子在跑、狗在追、窗帘在飘。传统 SfM 假设场景是 static 的，它会把所有运动都解释成"是相机在动"。结果它说相机走了 2 米，其实你站在原地没动，是狗跑过去了。

**糟糕情况二**：你站在原地拍 360 度 panorama，相机几乎不 translate，只 rotate。这种情况下三角测量（triangulation）完全退化，就像你闭上一只眼看世界，depth 全部 unobservable。COLMAP 直接躺平。

**糟糕情况三**：你不知道焦距。焦距错了，整个几何推导都崩了 —— 因为 focal length 决定了 pixel 怎么反投影到 3D ray，错了所有东西都错。

过去的方法要么要求你拍 large baseline 视频（"请走两步"），要么要求场景里没什么运动（"请别拍动的东西"），要么要求你知道焦距（"请用 DSLR 拍"）。但 casually captured videos 这三个条件都不满足。

## 之前别人怎么搞的

**传统派**（COLMAP、ORB-SLAM）：feature matching + bundle adjustment。假设 static + large baseline，遇到 dynamic + low parallax 直接崩。

**Mask 派**（Particle-SfM、LEAP-VO）：先用 long-term trajectory 算出哪些 pixel 是 dynamic 的，mask 掉它们，再跑传统 SfM。问题是 long-term trajectory 本身在 occlusion 严重时就不靠谱。

**Fine-tune 派**（CasualSAM、RoDynRF）：拿个 pre-trained mono-depth 网络，对每个 video 做 test-time fine-tuning。效果好，但一个视频要跑几分钟到十几分钟，慢得要命。

**Point cloud 派**（MonST3R、Dust3R）：用网络直接预测 pairwise 3D point cloud，然后 global alignment。最新的 idea，但在 long video 上 alignment 会 drift。

MegaSaM 走了一条不一样的路。

## MegaSaM 的核心 insight

**核心发现**：DROID-SLAM 这个 deep visual SLAM 框架，原本是给机器人做的，假设 static 场景 + 充分相机运动。但 paper 发现，只要做几个 careful 的修改，这个 framework 竟然可以 scale 到随手拍的 dynamic video 上。这个"surprising effectiveness"是整篇 paper 的灵魂。

为什么 DROID-SLAM 这个底子好？因为它有一个 **learned differentiable BA layer**。传统 BA 是手写公式，写死了假设。DROID-SLAM 让 network 学会预测 optical flow、预测 confidence、预测 damping factor，整个优化过程是 learned 的。这意味着我们可以通过 training data 教它处理 dynamic 场景，而不是改公式。

但直接拿 DROID-SLAM 跑 dynamic video 会崩 —— paper 的 Figure 2 第一列展示了这个 failure。所以需要做几个关键修改。

## 四个核心 trick 用大白话讲

### Trick 1：分两阶段训练（最重要的 trick）

**问题**：让 network 在 dynamic video 上 end-to-end 训练，结果 unstable。为什么？因为 BA 是个迭代优化，dynamic pixels 上 depth 不影响 reprojection error（动了 depth，flow 不变），所以 Hessian 奇异，gradient through BA layer 非常 noisy。这个 noise 会 corrupt flow prediction 的学习信号。

打个比方：你想同时学走路和跳探戈，肯定摔。先学走路，学扎实了，再学探戈。

**解法**：
- **Stage 1**：在 synthetic static data 上训练主网络 $F$，让它学会预测纯 ego-motion 的 flow 和 confidence。这时 BA Hessian 是 well-conditioned 的，学习信号 clean。学 4 天。
- **Stage 2**：冻结 $F$ 的所有参数，只训练一个额外的小网络 $F_m$，让它学会预测"哪些 pixel 是 moving object"。这个小网络看 multi-frame 信息（因为单帧看不出来物体动没动），输出一张 probability map。

**为什么这样 work**：Stage 1 把最难的部分（学 dense correspondence）在干净环境下学好。Stage 2 只学一个相对简单的 binary segmentation 任务，不需要 backprop through BA 的复杂 optimization，所以稳定。

Ablation 数据：不用 two-stage，RTE 从 0.008 飙到 0.136（17 倍变差）。这个 trick 是 critical 的。

### Trick 2：用 mono-depth 当初始化

**问题**：DROID-SLAM 原本把 depth 初始化为常数 1。在 large baseline 下没问题，几帧 BA 就收敛了。但在 low parallax video 上，从常数 1 出发永远收敛不到正确答案，因为几何上 depth 不可观，optimization 没有信号。

打个比方：你在大雾天开车，看不见路，只能靠 GPS。GPS 就是 mono-depth prior。

**解法**：用 DepthAnything（relative depth，准但没 scale）+ UniDepth（metric depth，有 scale 但 noisy）拼起来：
- DepthAnything 给 per-pixel 结构（"这个像素比那个像素远"）
- UniDepth 给 global scale（"整体大概几米"）
- 两者 median align 得到 metric-aligned initialization

这个 initialization 在两个地方用：训练时（用 GT depth 对齐 DepthAnything）和 inference 时（用 UniDepth 对齐 DepthAnything）。

Ablation：不用 mono-init，ATE 从 0.019 升到 0.038。

### Trick 3：Uncertainty-aware BA（最 elegant 的部分）

**问题**：什么时候该信几何、什么时候该信 prior？

举个例子：
- 拍一段前向走的视频，camera 有 parallax，depth well-constrained。这时再强行让 depth 跟 mono-depth prior 一致，反而把 mono-depth 的误差引进来了。
- 拍一段原地旋转的视频，depth 完全不可观。这时如果只跑 BA 不加 prior，depth 会 drift 到任意值，结果 garbage。

**解法**：直接量化"depth 不可观测的程度"。数学上是看 Hessian matrix 的 diagonal entry。Hessian 表示"参数 perturb 一下，reprojection error 变多少"。如果 perturb depth 完全不改变 error，Hessian $\approx 0$，说明 depth 不可观。

具体公式：
$$
w_d = \gamma_d \cdot \exp(-\beta_d \cdot \text{med}(\text{diag}(\mathbf{H}_{\mathbf{d}})))
$$

直觉：Hessian 大（geometry well-constrained）→ 指数项 → 0 → $w_d \to 0$ → 关闭 prior。
Hessian 小（unobservable）→ 指数项 → 1 → $w_d \to \gamma_d$ → 启用 prior。

类似地 focal length：如果 focal length 的 Hessian entry 太小（narrow FoV rotational video），说明 focal length 不可观，直接关掉 focal length optimization。

这个 trick 的 beauty 在于：**不是 hand-tuned "if-else" 规则，是从 BA 内部的数学结构自然 derive 出来的 observability measure**。

Ablation：不用 uncertainty-aware BA，ATE 从 0.019 升到 0.033。

### Trick 4：Movement probability map

**问题**：dynamic pixels 上的 flow 是 ego-motion + object motion 叠加。如果 confidence $\hat{\mathbf{w}}_{ij}$ 不能识别这些 pixels，BA 会被 corrupt。

**解法**：除了 stage 1 学到的 pairwise confidence $\hat{\mathbf{w}}_{ij}$，再加一个 per-frame 的 movement probability $\mathbf{m}_i$。在 BA 中最终 weight 是两者相乘：
$$
\tilde{\mathbf{w}}_{ij} = \hat{\mathbf{w}}_{ij} \cdot \mathbf{m}_i
$$

为什么这样设计？因为 $\hat{\mathbf{w}}_{ij}$ 学的是 pairwise photometric uncertainty（遮挡、光照变化），$\mathbf{m}_i$ 学的是 per-frame semantic object motion。这两个信号是 complementary 的，让 model 不需要在 confidence prediction 里隐式 encode dynamic 信息，简化学习任务。

$\mathbf{m}_i$ 的网络看 $I_i$ 和它所有 neighboring keyframes $\mathcal{N}(i)$，因为单帧无法判断 object motion —— 一只猫在画面里，单看一帧不知道它动没动，看相邻几帧就知道了。

Ablation：不用 $\mathbf{m}_i$，RTE 从 0.008 飙到 0.127（16 倍变差）。

## 后处理：consistent video depth

BA 输出的是 low-resolution disparity。要得到高分辨率 temporally consistent depth map，paper 做了一个额外的 first-order optimization。这里有两个关键改进：

1. **不 fine-tune mono-depth network**：CasualSAM 是 fine-tune DepthAnything，每 video 跑 1-6 分钟。MegaSaM 改成 optimize 一组 disparity + uncertainty variables，每 video 跑 1-2 秒。快 100 倍。

2. **Fix camera**：相机参数已经从 BA 得到了，不再 joint optimize。joint optimize 反而会让 camera accuracy 变差（ablation 数据：w/ ft-pose 配置 ATE=0.041 vs full=0.019）。

Loss 包含三部分：flow reprojection（带 aleatoric uncertainty 处理 dynamic pixels）、temporal consistency（让 depth 沿 optical flow 一致）、depth prior（用 surface normal + multi-scale gradient matching 把 depth 拉回 mono-depth 附近）。

## 效果到底多好

**Sintel 数据集**（uncalibrated）：
- MegaSaM：ATE=0.049, 1.0 秒/视频
- CasualSAM：ATE=0.067, 1.6 分钟/视频
- MonST3R：ATE=0.109, 10 秒/视频

**直觉**：accuracy 提升 2 倍多，速度提升 100 倍。

**DyCheck 数据集**（uncalibrated）：
- MegaSaM：ATE=0.041, 1.0 秒
- CasualSAM：ATE=0.209, 2.8 分钟
- MonST3R：ATE=0.690, 6.6 分钟

**直觉**：accuracy 提升 5-17 倍。

这些数字看着像"改进"，实际上是"质变"。从"勉强能用"到"production-ready"。

## 为什么这事 work —— 第一性原理思考

我从 paper 里抽出来的几个 deep insights：

### Insight 1：Learned optimization 可以通过 careful training scale 到 OOD 场景

DROID-SLAM 在 static data 上训练，MegaSaM 在 static data 上训练 stage 1 + dynamic data 上训练 stage 2。两个 stage 都用 synthetic data。但 inference 时跑 real-world dynamic video，generalization 很强。

这说明：**learned BA 的 generalization 瓶颈不在 optimization 本身，而在 supervision signal 的 quality**。two-stage training 让 supervision signal 在每个 stage 都 clean，所以整个系统能 generalize。

### Insight 2：Observability 是 geometric 问题，不是 learning 问题

Low parallax 不是"数据不够学不到"，是"几何上根本不可观"。再多的数据也救不了一个 static camera video 的 depth unobservable 问题。

MegaSaM 用 Hessian diagonal 直接 measure observability，在 unobservable 时 inject prior。这告诉我们可以从 optimization 内部的数学结构 derive 出 principled decision rule，比 hand-tuned heuristic 好得多。

### Insight 3：解耦学习任务

Two-stage training 的本质是 task decomposition：先学稳定任务（pure ego-motion flow），再学复杂任务（dynamic mask）。每个 stage 的 supervision signal 都 clean。

这个 insight 对任何涉及 differentiable optimization 的 pipeline 都有启发：当 objective landscape 复杂时，分阶段训练，让 stability-critical 的部分在简单 setting 下先学好。

### Insight 4：Foundation model 提供 prior，但不替代 geometry

DepthAnything 是 foundation model，能给很好的 relative depth prior。但它不能替代 geometric reasoning —— 它的 metric scale 不准，temporal consistency 差。

MegaSaM 用 mono-depth 当 initialization 和 regularization，但最终 depth 还是从 video 自身的 geometric constraint（multi-view consistency）优化出来的。prior 帮你 escape bad local minimum，geometry 帮你 refine 到 accurate solution。

这个 hybrid 思路在 SLAM / 3D vision 领域会越来越重要。

## 局限性

paper 自己讲了两个失败 case：

1. **Moving object 占满整个画面**：没有足够 static pixels 来 anchor camera，系统 fail。这种情况下 fundamentally unobservable，需要 IMU 或其他 sensor。

2. **Camera motion 和 object motion 共线**：例如 selfie video 跟着人脸走，camera 往前走、人也往前走。这种情况下 reprojection error 对 camera translation 和 object translation 同时 perturb 时不变，是 observability 的 degeneracy。

还有几个没在 paper 里讲但我觉得是局限：

3. **Variable focal length**：现在固定 shared focal length，zoom 视频处理不了
4. **Radial distortion**：手机广角镜头 distortion 明显，没 model
5. **Rolling shutter**：手机 video often 有 rolling shutter effect，没 model

## 对未来的启发

如果让我做 follow-up work，会想几个方向：

1. **加 IMU**：手机都有 IMU，IMU 给的 acceleration 信号能 break camera-object motion 共线 degeneracy
2. **Per-frame focal length + temporal smoothness**：处理 zoom 视频
3. **用 SAM features 提供 semantic motion prior**：SAM 能 segment 出 foreground objects，能帮 $\mathbf{m}_i$ network
4. **Joint with 4D Gaussian Splatting**：把 BA 输出作为 4DGS 的初始化，可能得到更 robust 的 dynamic scene reconstruction
5. **Self-supervised training on real videos**：现在 stage 2 用 synthetic dynamic data（Kubric），如果能用 real dynamic video self-supervised training，能 generalize更好

## 一句话总结

MegaSaM 告诉我们：deep visual SLAM 框架只要做对几个 careful modifications —— 分阶段训练解 stability、mono-depth init 解 low parallax initialization、uncertainty-aware BA 解 observability、motion map 解 dynamic masking —— 就能 scale 到 casually captured dynamic videos 上，accuracy 提升几倍，速度提升百倍。

这篇 paper 最大的贡献其实不在某个 specific trick，而在于它 systematically 分析了 deep SLAM 在 dynamic video 上的 failure modes，每一个 failure mode 都给了一个 principled solution。这种"系统性工程化"的研究风格比"找一个新 trick"更难得，对 real-world deployment 更有价值。

---

# MegaSaM 深度技术解析

## 1. 核心问题与 motivation

MegaSaM 要解决的是从 casually captured monocular dynamic videos 中同时估计 camera poses、focal length 和 dense video depth maps 的问题。这个问题非常 hard，因为这类视频有三个 critical difficulties：

1. **Scene dynamics**: 场景中存在移动物体，违反了 classic SfM/SLAM 的 static scene assumption，会导致 BA 优化被 corrupt
2. **Limited parallax**: 手持相机常常是 near-rotational motion（几乎原地旋转），导致 triangulation 退化，几何 unobservable
3. **Unknown focal length**: in-the-wild 视频没有 EXIF 信息，focal length 必须联合估计

传统方法（COLMAP、ORB-SLAM 等）依赖 feature matching + large baseline triangulation，在这三个条件下都会 degrade 甚至完全 fail。最近的方法要么 test-time fine-tuning mono-depth networks（CasualSAM, RoDynRF）非常 expensive，要么用 long-term trajectory segmentation（Particle-SfM, LEAP-VO）对 brittle 场景敏感。

MegaSaM 的 insight 是：**deep visual SLAM framework（DROID-SLAM）的 learned differentiable BA layer 实际上可以 scale 到 dynamic videos 上，只要做 careful modifications**。这个 surprising effectiveness 是论文的核心 finding。

参考链接：
- DROID-SLAM: https://proceedings.neurips.cc/2021/hash/e45e3f6f5e67725c1b69311093f49e66-Abstract.html
- CasualSAM: https://arxiv.org/abs/2210.00194
- DepthAnything-V2: https://arxiv.org/abs/2406.09414
- UniDepth: https://arxiv.org/abs/2403.18913
- MonST3R: https://arxiv.org/abs/2410.03825
- 项目主页: https://mega-sam.github.io

---

## 2. DROID-SLAM 基础 formulation

MegaSaM 建立在 DROID-SLAM 的 differentiable BA layer 之上，我先讲清楚这个 base framework。

### 2.1 状态变量

系统维护两类 state variables：
- Per-frame low-resolution disparity map: $\hat{\mathbf{d}}_i \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{8}}$（低分辨率以节省内存）
- Camera poses: $\hat{\mathbf{G}}_i \in SE(3)$（每个 frame 一个 6-DoF pose）
- MegaSaM 额外加入了 shared focal length $\hat{f}$（DROID-SLAM 假设 focal length 已知）

### 2.2 Frame graph

构建一个 frame graph $\mathcal{P}$，节点是 video frames，边是 overlapping FoV 的 frame pairs。这个 graph 是 dynamically maintained 的（前端 sliding window + 后端 global）。

### 2.3 Network prediction

对于每个 frame pair $(I_i, I_j)$，network 通过 ConvGRU 迭代预测：
$$
(\hat{\mathbf{u}}_{ij}^{k+1}, \hat{\mathbf{w}}_{ij}^{k+1}) = F(I_i, I_j, \hat{\mathbf{u}}_{ij}^k, \hat{\mathbf{w}}_{ij}^k)
$$

变量解释：
- $\hat{\mathbf{u}}_{ij}^k \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{8} \times 2}$：第 $k$ 次迭代预测的 2D correspondence field（即从 $I_i$ 到 $I_j$ 的 dense optical flow）
- $\hat{\mathbf{w}}_{ij}^k \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{8}}$：per-pixel confidence weight
- $k$：BA iteration index

### 2.4 Multi-view 刚性对应

从 camera parameters 和 disparity 推导出 "应该" 的 correspondence：
$$
\mathbf{u}_{ij} = \pi\left(\hat{\mathbf{G}}_{ij} \circ \pi^{-1}(\mathbf{p}_i, \hat{\mathbf{d}}_i, \mathcal{K}^{-1}), \mathcal{K}\right)
$$

变量解释：
- $\mathbf{p}_i$：pixel coordinate grid
- $\pi$：perspective projection operator
- $\pi^{-1}(\mathbf{p}_i, \hat{\mathbf{d}}_i, \mathcal{K}^{-1})$：将 pixel + disparity 反投影到 3D point：$\mathbf{X} = \hat{\mathbf{d}}_i \cdot \mathcal{K}^{-1} \mathbf{p}_i$
- $\hat{\mathbf{G}}_{ij} = \hat{\mathbf{G}}_j \circ \hat{\mathbf{G}}_i^{-1}$：relative pose
- $\mathcal{K} \in \mathbb{R}^{3 \times 3}$：intrinsic matrix，里面包含 $\hat{f}$

### 2.5 Differentiable BA via Levenberg-Marquardt

Cost function 是 weighted reprojection error：
$$
\mathcal{C}(\hat{\mathbf{G}}, \hat{\mathbf{d}}, \hat{f}) = \sum_{(i,j) \in \mathcal{P}} ||\hat{\mathbf{u}}_{ij} - \mathbf{u}_{ij}||_{\Sigma_{ij}}^2
$$

其中 $\Sigma_{ij} = \text{diag}(\hat{\mathbf{w}}_{ij})^{-1}$，即 confidence 越低的 pixel 在 BA 中 weight 越小。

LM 迭代公式：
$$
(\mathbf{J}^T \mathbf{W} \mathbf{J} + \lambda \text{diag}(\mathbf{J}^T \mathbf{W} \mathbf{J})) \Delta \boldsymbol{\xi} = \mathbf{J}^T \mathbf{W} \mathbf{r}
$$

变量解释：
- $\Delta \boldsymbol{\xi} = (\Delta \mathbf{G}, \Delta \mathbf{d}, \Delta f)^T$：参数更新量（待求）
- $\mathbf{J}$：reprojection residual $\mathbf{r} = \hat{\mathbf{u}}_{ij} - \mathbf{u}_{ij}$ 对参数的 Jacobian
- $\mathbf{W}$：confidence 组成的 diagonal matrix
- $\lambda$：damping factor，由 network 预测（这是 learned BA 的精髓）

### 2.6 Schur complement trick

将 Hessian 分块成 camera/disparity 两个 block：
$$
\begin{bmatrix} \mathbf{H}_{\mathbf{G},f} & \mathbf{E}_{\mathbf{G},f} \\ \mathbf{E}_{\mathbf{G},f}^T & \mathbf{H}_{\mathbf{d}} \end{bmatrix} \begin{bmatrix} \Delta \boldsymbol{\xi}_{\mathbf{G},f} \\ \Delta \mathbf{d} \end{bmatrix} = \begin{bmatrix} \tilde{r}_{\mathbf{G},f} \\ \tilde{r}_{\mathbf{d}} \end{bmatrix}
$$

由于 $\mathbf{H}_{\mathbf{d}}$ 是 diagonal（每个 reprojection term 只涉及一个 disparity 变量），可以用 Schur complement 高效求解：
$$
\Delta \boldsymbol{\xi}_{\mathbf{G},f} = \left[\mathbf{H}_{\mathbf{G},f} - \mathbf{E}_{\mathbf{G},f} \mathbf{H}_{\mathbf{d}}^{-1} \mathbf{E}_{\mathbf{G},f}^T\right]^{-1} \left(\tilde{r}_{\mathbf{G},f} - \mathbf{E}_{\mathbf{G},f} \mathbf{H}_{\mathbf{d}}^{-1} \tilde{r}_{\mathbf{d}}\right)
$$
$$
\Delta \mathbf{d} = \mathbf{H}_{\mathbf{d}}^{-1}(\tilde{r}_{\mathbf{d}} - \mathbf{E}_{\mathbf{G},f}^T \Delta \boldsymbol{\xi}_{\mathbf{G},f})
$$

这个 trick 让整个 BA 过程 differentiable，可以 backprop end-to-end。

### 2.7 为什么 DROID-SLAM 在 dynamic video 上 fail

直觉：当场景里有 moving objects 时，$\hat{\mathbf{u}}_{ij}$ 预测的 flow 包含了 ego-motion + object motion 的叠加，但 $\mathbf{u}_{ij}$ 是用纯 ego-motion 假设 rigid transformation 推导出来的，两者在 dynamic pixels 处 mismatch。如果 confidence $\hat{\mathbf{w}}_{ij}$ 不能识别这些 dynamic pixels，BA 就会被它们 corrupt。

---

## 3. MegaSaM 的核心创新

### 3.1 Two-stage training：分离 ego-motion 和 object motion 学习

#### 关键问题：为什么直接在 dynamic videos 上 end-to-end 训练不稳定？

直觉是 differentiable BA 的 Hessian matrix 在 dynamic pixels 上奇异，会导致 gradient 不稳定。如果直接训练 $F$ 同时学 flow 和 dynamic mask，BA 的 optimization 过程会扰动 flow prediction 的学习信号。

#### 解法：分两阶段

**Stage 1 - Ego-motion pretraining**: 在 synthetic static scenes 上训练 $F$（TartanAir 163 scenes + 5K static Kubric videos），让它先学会预测纯 ego-motion induced flows 和 confidence：
$$
\mathcal{L}_{\text{static}} = \mathcal{L}_{\text{cam}} + w_{\text{flow}} \mathcal{L}_{\text{flow}}
$$
其中 $\mathcal{L}_{\text{cam}}$ 是 camera pose L2 loss，$\mathcal{L}_{\text{flow}}$ 是 ego-motion induced flow L2 loss。$w_{\text{flow}} = 0.02$。

**Stage 2 - Dynamic fine-tuning**: 冻结 $F$ 的参数，只训练额外的 motion module $F_m$。$F_m$ 的输入是 $I_i$ 和它的 neighboring keyframes $\mathcal{N}(i)$，输出 object movement probability map：
$$
\mathbf{m}_i \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{8}} = F_m\left(\{I_i\} \cup \mathcal{N}(i)\right)
$$

Loss：
$$
\mathcal{L}_{\text{dynamic}} = \mathcal{L}_{\text{cam}} + w_{\text{motion}} \mathcal{L}_{\text{CE}}
$$

$w_{\text{motion}} = 0.1$，$\mathcal{L}_{\text{CE}}$ 是 movement map 的 cross-entropy loss。

#### 关键 trick：组合 confidence

在 BA 中，最终 weight 是 pairwise confidence 乘以 movement map：
$$
\tilde{\mathbf{w}}_{ij} = \hat{\mathbf{w}}_{ij} \cdot \mathbf{m}_i
$$

直觉：$\hat{\mathbf{w}}_{ij}$ 学到的是 per-pair 的 photometric uncertainty（光照变化、遮挡等），而 $\mathbf{m}_i$ 是 per-frame 的 object-level semantic prior。这两个信号是 complementary 的，让 model 不需要在 confidence prediction 里隐式 encode dynamic object 信息，简化了学习任务。

#### Architecture 细节

$F_m$ 的设计（从 Figure 10 中读出）：
1. 接收 ConvGRU 的 hidden state features（来自 $F$，已冻结）
2. Spatial average pooling 提供 global spatial context
3. Temporal average pooling 融合 $I_i$ 与所有 $\mathcal{N}(i)$ 的信息（让 model 看到多帧 motion pattern）
4. Decoder 输出 $\frac{H}{8} \times \frac{W}{8}$ 的 probability map

这种 design 利用了：单帧无法判断 object motion，必须看 temporal context。例如：一个 stationary person 和一个 moving person 在单帧上看起来完全一样，只有 cross-frame 才能区分。

### 3.2 Monocular depth initialization

DROID-SLAM 原本把 disparity 初始化为常数 1，这在 dynamic + low parallax 视频上完全 insufficient。MegaSaM 用 pre-trained mono-depth 提供 initialization：

#### 训练阶段
用 DepthAnything 预测 relative disparity $\bar{D}_i^{\text{rel}}$，然后用 ground-truth depth 算 global scale/shift 对齐：
$$
\hat{\mathbf{d}}_i^{\text{init}} = \bar{\alpha} \bar{D}_i^{\text{rel}} + \hat{\beta}
$$

#### Inference 阶段：metric alignment trick

这里有个很巧的 trick。问题：DepthAnything 给的是 affine-invariant relative depth，没有 metric scale；UniDepth 给的是 metric depth 但 noise 更大。怎么 combine？

解法：用 UniDepth 的 median 把 DepthAnything 对齐到 metric：
$$
\hat{\alpha}_i = \frac{D_i^{\text{abs}} - \text{median}_i(D_i^{\text{abs}})}{D_i^{\text{rel}} - \text{median}(D_i^{\text{rel}})}
$$
$$
\hat{\beta} = \text{median}(D_i^{\text{abs}} - \hat{\alpha} D_i^{\text{rel}})
$$

直觉：UniDepth 在 scale 上是 robust 的（虽然 per-pixel noisy），所以用它的 median 提供 global scale 信号；DepthAnything 在 relative structure 上更 accurate，所以用它提供 per-pixel structure。两者 median alignment 各取所长。

### 3.3 Uncertainty-aware global BA（最 elegant 的部分）

#### 问题陈述

Question：什么时候应该 enable mono-depth regularization $w_d$？两种极端：
- 视频有 large parallax：geometric 本身 well-constrained，加 mono-depth 反而引入 noise
- 视频是 pure rotation：reprojection BA 退化（depth 不可观），不加 regularization 会得到 degenerate solution

#### 数学 formulation

用 Laplace approximation 估计参数的 epistemic uncertainty：
$$
p(\boldsymbol{\theta} | \mathcal{T}) \approx \mathcal{N}(\boldsymbol{\theta}^*; \boldsymbol{\mu}, \Sigma_{\theta})
$$
其中 $\Sigma_{\theta} = -\mathbf{H}(\theta^*)^{-1}$，$\theta^*$ 是 MAP estimate。

由于完整 inverse Hessian 在 large video 上 prohibitive，用 diagonal approximation：
$$
\Sigma_{\theta} \approx \text{diag}(-\mathbf{H}(\theta^*))^{-1}
$$

直觉：Hessian $\mathbf{H}_{\mathbf{d}}$ 的 diagonal entries 表示每个 disparity variable 被 reprojection error "约束" 的强度。如果 camera 是 static 的，perturbing disparity 不会改变 reprojection error，所以 Jacobian $\mathbf{J}_d \approx 0$，Hessian diagonal $\approx 0$，uncertainty $\to \infty$。这就是 disparity "unobservable" 的数学表达。

#### Adaptive regularization

基于这个，MegaSaM 设计了 adaptive scheme：
$$
w_d = \gamma_d \cdot \exp\left(-\beta_d \cdot \text{med}(\text{diag}(\mathbf{H}_{\mathbf{d}}))\right)
$$

参数：$\gamma_d = 10^{-4}$, $\beta_d = 0.05$。

直觉：当 median disparity Hessian 大（geometric well-constrained）时，指数项 $\to 0$，$w_d \to 0$，关闭 regularization；当 median disparity Hessian 小（low parallax, unobservable）时，$w_d \to \gamma_d$，启用 regularization 把 disparity 拉回 mono-depth prior。

类似地，focal length：
- 计算 $H_f$（focal length 对应的 Hessian entry）
- 如果 $H_f < \tau_f$（$\tau_f = 50$），说明 focal length 不可观（例如 narrow FoV rotational video），关闭 focal length optimization

#### 这个设计的物理直觉

考虑几个 case：
1. **Forward-moving camera（large parallax）**： disparity 与 flow 强相关，Hessian 大，无需 prior
2. **Side-moving camera**：triangulation 良好，Hessian 大，无需 prior
3. **Pure rotation（panning）**：epipole 在 image plane 上，parallax = 0，disparity 完全 unobservable，Hessian $\to 0$，必须用 prior
4. **Static camera**：所有 disparity unobservable，必须用 prior

Figure 4 visualizes 这个：rotation-dominant video 的 disparity uncertainty map 整体偏高，而 forward-moving video 只在 epipole 附近 uncertainty 高（这是结构性的 degeneracy）。

### 3.4 Frontend & backend tracking pipeline

**Frontend（sliding window）**：
1. 累积 $N_{\text{init}} = 8$ 个 keyframes，进行 camera-only BA（fix disparity）作为初始化
2. 增量添加/删除 keyframes，sliding window BA
3. Cost function：
$$
\mathcal{C} = \sum_{(i,j) \in \mathcal{P}} ||\hat{\mathbf{u}}_{ij} - \mathbf{u}_{ij}||_{\Sigma_{ij}}^2 + w_d \sum_i ||\hat{\mathbf{d}}_i - D_i^{\text{align}}||^2
$$
前端固定 $w_d = 0.05$（小但非零，作为 weak stabilization）

**Backend（global BA）**：
1. Global BA over all keyframes（用上面 adaptive $w_d$）
2. Pose graph optimization for non-keyframes
3. Global BA over all frames

---

## 4. Consistent video depth estimation

BA 输出的是 low-resolution disparity。为了得到 high-quality video depth，MegaSaM 跟随 CasualSAM 的 spirit 但做了重要修改。

### 4.1 关键改进

1. **不 fine-tune mono-depth network**：CasualSAM 要 per-video fine-tuning DepthAnything，非常慢。MegaSaM 改成 optimize 一组 disparity + uncertainty variables
2. **Fix camera**：相机参数已经从 BA 得到，不再 joint optimize
3. **Better depth prior**：用 surface normal + multi-scale gradient matching 替代 CasualSAM 的简单 prior

### 4.2 Loss 函数

$$
\mathcal{C}_{\text{cvd}} = w_{\text{flow}} \mathcal{C}_{\text{flow}} + w_{\text{temp}} \mathcal{C}_{\text{temp}} + w_{\text{prior}} \mathcal{C}_{\text{prior}}
$$

权重：$w_{\text{flow}} = w_{\text{prior}} = 1.0$, $w_{\text{temp}} = 0.2$

#### Flow reprojection loss（带 aleatoric uncertainty）

把 object motion 建模为 heteroscedastic aleatoric uncertainty（Kendall & Gal 的 framework）：
$$
\mathcal{C}_{\text{flow}}^{ij} = \hat{M}_i ||\mathbf{u}_{ij} - \mathbf{p}_i, \text{flow}_{ij}(\mathbf{p}_i)||_1 + \log\left(\frac{1}{\hat{M}_i}\right)
$$

变量：
- $\hat{M}_i$：per-pixel aleatoric uncertainty（要学的）
- $\mathbf{u}_{ij}$：estimated camera + disparity induced flow
- $\text{flow}_{ij}$：RAFT 预测的 dense optical flow
- $\log(1/\hat{M}_i)$：regularizer 防止 uncertainty collapse 到 infinity

直觉：dynamic pixels 的 $\text{flow}_{ij}$ 包含 object motion，与 $\mathbf{u}_{ij}$（pure ego-motion）mismatch，optimization 会自动调高这些 pixels 的 $\hat{M}_i$（降低 weight），同时 $\log$ 项防止 trivially 调高所有 uncertainty。

#### Temporal consistency loss

$$
\mathcal{C}_{\text{temp}}^{ij} = \hat{M}_i \delta(\mathbf{P}_z^{i \to j}, \hat{D}_j(\mathbf{p} + \text{flow}_{ij}(\mathbf{p}))) + \log\left(\frac{1}{\hat{M}_i}\right)
$$
$$
\delta(a, b) = ||\max(a/b, b/a)||_1
$$
$$
\mathbf{P}_z^{i \to j} = \left(D_i(\mathbf{p}) \mathbf{R}_{i \to j} \mathcal{K}^{-1} \mathbf{p} + \mathbf{t}_{i \to j}\right)_{[z]}
$$

直觉：把 pixel $\mathbf{p}$ 在 $I_i$ 的 depth 投影到 $I_j$ 应该和 $I_j$ 在对应位置（$\mathbf{p} + \text{flow}$）的 depth 一致。用 ratio loss $\max(a/b, b/a)$ 比 L2 robust（对 depth scale 不敏感）。

#### Depth prior loss（三部分）

$$
\mathcal{C}_{\text{prior}} = \mathcal{C}_{\text{si}} + w_{\text{grad}} \mathcal{C}_{\text{grad}} + w_{\text{normal}} \mathcal{C}_{\text{normal}}
$$

**Scale-invariant loss** (Eigen's formulation)：
$$
\mathcal{C}_{\text{si}} = \frac{1}{n} \sum_{\mathbf{p}} (R(\mathbf{p}))^2 - \frac{1}{n^2}\left(\sum_{\mathbf{p}} R(\mathbf{p})\right)^2
$$
$$
R_i = \log(\hat{D}_i) - \log(D_i^{\text{align}})
$$

第二项是减去全局 mean，让 loss 对 affine transform invariant。$w_{\text{grad}} = 1$, $w_{\text{normal}} = 4$。

**Multi-scale gradient matching**：
$$
\mathcal{C}_{\text{grad}} = \frac{1}{n} \sum_s w_{\nabla}^s(\mathbf{p}) \sum_{\mathbf{p}} \left(|\nabla_x R^s(\mathbf{p})| + |\nabla_y R^s(\mathbf{p})|\right)
$$
$$
w_{\nabla}^s(\mathbf{p}) = 1 - \exp\left(-\beta_{\nabla} (\nabla_x R^s(\mathbf{p}) + \nabla_y R^s(\mathbf{p}))\right)
$$

$\beta_{\nabla} = 5$。$w_{\nabla}$ 是一个软 mask：只有当 gradient 偏差大时才惩罚，允许 depth 估计在 detail 上偏离 prior（这是为了 recover detail beyond mono-depth 的能力）。

**Surface normal consistency**：
$$
\mathcal{C}_{\text{normal}} = \sum_{\mathbf{p}} 1 - \hat{\mathbf{N}}(\mathbf{p}) \cdot \mathbf{N}^{\text{align}}(\mathbf{p})
$$

直觉：normal 是 depth 的 derivative，constrain normal 比 constrain depth 更 local，能让优化在保持 shape structure 的同时调整 scale。

### 4.3 Optimization 流程

1. 初始化 disparity 从 metric-aligned mono-depth
2. 初始化 uncertainty map 从 camera tracking 阶段的 movement probability map
3. Warm-up（100 steps）：fix disparity，只优化 uncertainty + per-frame scale/shift
4. Joint optimization（400 steps）：disparity 和 uncertainty 一起优化

Frame pairs 用固定 intervals: $j \in (i+1, i+2, i+4, i+8, i+15)$，覆盖 short-term 和 long-term temporal context。

---

## 5. 实验数据解读

### 5.1 Camera estimation（Tables 1, 2, 3）

以 Sintel（Table 1）uncalibrated setting 为例：
- MegaSaM: ATE=0.049, RTE=0.018, RRE=0.31, Time=1.0s
- CasualSAM: ATE=0.067, RTE=0.019, RRE=0.47, Time=1.6m
- MonST3R: ATE=0.109, RTE=0.051, RRE=1.32, Time=10s
- ACE-Zero: ATE=0.065, RTE=0.028, RRE=1.92, Time=1.3s

MegaSaM 在 accuracy 上 2-10× 优于 baselines，速度上还快 100×。MonST3R 虽然用了更新的 Dust3R-style global 3D point cloud representation，但在 dynamic videos 上明显 worse，且慢 10×。

### 5.2 Depth estimation（Table 4）

DyCheck dataset：
- MegaSaM: abs-rel=0.22, log-rmse=0.29, $\delta_{1.25}$=84.7（论文数据中数值不完整，但论文声称最佳）
- DA-v2 raw: abs-rel=0.20（单帧反而更低，但没有 temporal consistency）
- DepthCrafter: abs-rel=0.22, log-rmse=0.29, $\delta_{1.25}$=83.7
- CasualSAM: abs-rel=0.31

直觉：raw mono-depth 在单帧 metric 上可能更好，但 temporal consistency 差（看 Figure 6 的 x-t slice 就知道有 flickering）。MegaSaM 在保持 metric accuracy 的同时获得 strong temporal consistency。

### 5.3 Ablation study（Table 5）

Sintel 数据上：
| Config | ATE | RTE | RRE | Abs-Rel | $\delta_{1.25}$ |
|--------|-----|-----|-----|---------|------------------|
| Droid-SLAM | 0.030 | 0.022 | 0.50 | - | - |
| w/o mono-init | 0.038 | 0.026 | 0.49 | - | - |
| w/o $\mathbf{m}_i$ | 0.032 | 0.127 | 0.14 | - | - |
| w/o 2-stage train | 0.035 | 0.136 | 0.17 | - | - |
| w/o u-BA | 0.033 | 0.013 | 0.11 | - | - |
| w/ ft-pose | 0.041 | 0.018 | 0.33 | 0.23 | 71.2 |
| w/o new $\mathcal{C}_{\text{prior}}$ | - | - | - | 0.36 | 72.5 |
| Full | 0.019 | 0.008 | 0.04 | 0.21 | 73.1 |

关键 insights：
1. **w/o 2-stage train**：RTE 从 0.008 飙到 0.136（17× worse），证明 two-stage training 是 critical 的。直接在 dynamic data 上 end-to-end 训练确实 unstable
2. **w/o $\mathbf{m}_i$**：RTE 从 0.008 飙到 0.127，证明 motion probability map 是必要的。光靠 confidence 无法 subsume object motion
3. **w/o u-BA**：ATE 从 0.019 升到 0.033（74% worse），证明 adaptive regularization 比总是 enable regularization 好
4. **w/ ft-pose**（joint optimize camera + depth）：camera accuracy 反而 worse，证明分两阶段（先 camera 后 depth）更 robust

---

## 6. Architecture 细节（Figure 10, 11）

### 6.1 Feature encoder（Figure 11）

两个 encoder 都输出 $\frac{1}{8}$ 分辨率的 feature maps：
- **Feature encoder**：用于构建 4D correlation volume
- **Context encoder**：作为 ConvGRU 的 initial hidden state

### 6.2 ConvGRU update block

迭代过程：
1. 从 correlation volume lookup 当前 flow 估计的 correlation features
2. Concatenate with context features + current flow + confidence
3. ConvGRU update hidden state
4. Output new flow + confidence（gray blocks，stage 1 trained）
5. Output motion probability map（blue blocks，stage 2 trained on top of frozen gray blocks）

### 6.3 Motion module $F_m$ 架构

输入：
- ConvGRU 的 hidden state features（来自 frozen $F$）
- Spatial average pool 提供 global context
- Temporal average pool over $\{I_i\} \cup \mathcal{N}(i)$

这个 temporal pooling 很关键：单看一帧无法判断 object motion，必须看 cross-frame 信息。例如：图像中有一只猫，单帧无法知道它是静止的还是移动的；看相邻几帧的 cat appearance 变化就能判断。

---

## 7. Limitations

论文诚实地讨论了失败 case（Figure 8）：
1. **Moving object dominates**：如果 moving object 占满整个 image，没有足够 static pixels 来约束 camera，系统 fail
2. **Colinear motion**：camera motion 和 object motion 方向相同时（例如 selfie video 跟着人脸走），无法 disambiguate。数学上是 observability 的 degeneracy：reprojection error 对 camera translation 和 object translation 同时 perturb 时不变

---

## 8. Intuition 总结

让我把核心 intuitions 提炼一下：

### 8.1 为什么 deep SLAM framework 能 extend 到 dynamic videos？

关键 insight 是：differentiable BA 的 learned confidence mechanism 提供了一个 natural place 来 inject dynamic object prior。如果 confidence 能正确 downweight dynamic pixels，BA 自然能 ignore 它们。MegaSaM 通过额外的 motion probability map 把这个 downweighting 显式化，并用 two-stage training 让 confidence learning 不受 dynamic 干扰。

### 8.2 为什么 uncertainty-aware BA 重要？

Low parallax 是 geometric observability 的问题，不是 learning capacity 的问题。再多的数据也救不了一个 static camera video 的 unobservable depth。MegaSaM 通过 Hessian diagonal 直接 measure observability，在 unobservable 时 inject prior。这是一个 principled way 来 decide "when to trust geometry vs when to trust prior"。

### 8.3 为什么不直接 fine-tune mono-depth network？

Test-time fine-tuning 有两个问题：
1. 慢（每 video 要 train）
2. Overfitting risk：fine-tune 后 mono-depth 在 dynamic pixels 上不再 reliable

MegaSaM 的方案：把 depth 当作 free variable 优化，但用 mono-depth 作为 prior loss 约束。这样既能 escape mono-depth 的 inconsistency，又不会 drift too far。

### 8.4 Why two-stage training works

这是最重要的 intuition。直接在 dynamic data 上 end-to-end 训练 differentiable BA 时，Hessian matrix 在 dynamic pixels 上奇异（因为 depth 不影响 reprojection error），gradient through BA layer 非常 noisy。这个 noise 会 corrupt flow prediction 的学习信号。

Two-stage trick 把 "learn flow correspondence" 和 "learn dynamic mask" 解耦：
- Stage 1：在 static data 上，BA Hessian 是 well-conditioned 的，flow 学习信号 clean
- Stage 2：fix flow predictor，只学 mask。即使 Hessian 奇异，mask prediction 不依赖 BA 的 gradient，只依赖 cross-entropy loss，所以稳定

这个 insight 对 future work 很有启发：任何涉及 differentiable optimization 的 pipeline，当 objective landscape 复杂时，都可以考虑分阶段训练，让 stability-critical 的部分先在简单 setting 下学好。

---

## 9. 与 Concurrent Work 的对比

### 9.1 vs MonST3R

MonST3R 用 Dust3R-style global 3D point cloud + 额外 alignment optimization。MegaSaM 比 MonST3R：
- ATE: 2× better
- 速度: 10× faster
- Memory: 更少（point cloud 表达比 dense depth 更冗余）

直觉：Dust3R 是 pair-wise 架构，扩展到 long video 需要 pairwise alignment，globally inconsistent。MegaSaM 用 BA 直接做 global optimization，inherently consistent。

### 9.2 vs CasualSAM

CasualSAM 是 MegaSaM 的 spiritual predecessor，但用 test-time fine-tuning。MegaSaM 改进：
- 不用 fine-tune（快 100×）
- Fix camera（更 stable）
- Better depth prior（normal + multi-scale gradient）

### 9.3 vs Particle-SfM / LEAP-VO

这些方法用 long-term trajectory segmentation mask dynamic pixels，然后在 conventional SfM 上跑。MegaSaM 优势：
- 不依赖 long-term trajectory prediction（ brittle in occlusion-heavy scenes）
- 直接在 learned BA 中集成 mask，更 end-to-end

---

## 10. 可能的 Future Directions

基于论文 limitations 和我自己的 speculation：

1. **Variable focal length**: 现在固定 shared focal length，无法处理 zoom 视频。可以用 per-frame focal length + temporal smoothness
2. **Radial distortion**: Mobile phone wide-angle lens distortion 很常见，可以在 intrinsic model 里加 distortion parameters
3. **Rolling shutter**: Mobile phone 视频 often 有 rolling shutter effect，可以在 BA 里 model per-row timestamp
4. **Better priors from foundation models**: 用 DINO/SAM features 做 semantic prior 可能 improve motion segmentation
5. **Joint optimization with NeRF/Gaussian Splatting**: 把 BA 输出作为 dynamic radiance field 的初始化，可能得到更 robust 的 dynamic scene reconstruction

---

## 11. Reference Links

- MegaSaM project page: https://mega-sam.github.io
- DROID-SLAM paper: https://proceedings.neurips.cc/2021/hash/e45e3f6f5e67725c1b69311093f49e66-Abstract.html
- DROID-SLAM code: https://github.com/princeton-vl/DROID-SLAM
- CasualSAM: https://arxiv.org/abs/2210.00194
- DepthAnything-V2: https://arxiv.org/abs/2406.09414
- DepthAnything project: https://depth-anything.com
- UniDepth: https://arxiv.org/abs/2403.18913
- UniDepth code: https://github.com/cvg/LightGlue
- MonST3R: https://arxiv.org/abs/2410.03825
- MonST3R code: https://github.com/Junyi42/monst3r
- Dust3R: https://arxiv.org/abs/2312.14132
- RAFT (optical flow): https://arxiv.org/abs/2003.12039
- Particle-SfM: https://arxiv.org/abs/2204.05663
- LEAP-VO: https://arxiv.org/abs/2404.00640
- ACE-Zero: https://arxiv.org/abs/2404.14351
- RoDynRF: https://arxiv.org/abs/2304.02503
- Kubric dataset generator: https://github.com/google-research/kubric
- TartanAir: https://theairlab.org/tartanair
- MPI Sintel: http://sintel.is.tue.mpg.de
- DyCheck: https://dynamic-video-stereo.github.io
- DynIBaR: https://arxiv.org/abs/2211.11082
- Kendall & Gal uncertainty paper: https://papers.nips.cc/paper/2017/hash/2650d6089a4d27af6c4eebf783f0cf8a-Abstract.html
- Schur complement / Bundle adjustment synthesis (Triggs et al.): https://link.springer.com/chapter/10.1007/3-540-44480-7_21
- Laplace approximation for NNs (Ritter et al.): https://openreview.net/forum?id=SkdvdjoAQf

---

这篇 paper 真正 elegant 的地方在于：它没有 invent 全新的 framework，而是仔细分析 deep SLAM 在 dynamic videos 上的 failure modes，然后用针对性的 design choices 解决每一个 issue。Two-stage training 解 stability 问题，mono-depth init 解 low parallax initialization 问题，uncertainty-aware BA 解 observability 问题，motion map 解 dynamic pixel masking 问题。每个 design choice 都有清晰的 mathematical motivation 和 ablation 验证。这种"系统性工程化"的研究风格比"找一个新 trick"更难得，对 real-world deployment 更有价值。
