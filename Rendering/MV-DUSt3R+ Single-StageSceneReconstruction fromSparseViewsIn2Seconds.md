---
source_pdf: MV-DUSt3R+ Single-StageSceneReconstruction fromSparseViewsIn2Seconds.pdf
paper_sha256: 72643e5a9884b76f221d0084380b6db0b1fcb46db7851c73ad90449026eaba62
processed_at: '2026-08-05T21:48:56-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 MV-DUSt3R+

## TL;DR

DUSt3R 每次只看两张图，把一堆两两配对的结果硬拼起来，经常拼错。MV-DUSt3R 让一个 network 一次性看完所有图，像人脑一样综合判断，又快又准。MV-DUSt3R+ 更进一步，把"以哪张图做参考"这件事也并行做了多份，让远处的图能"借光"，长距离信息传递更顺。

paper: https://mv-dust3rp.github.io/

---

## 1. 这事为什么难：一个类比

想象你看 24 张照片拍的是同一套大房子里不同房间。你脑子里能轻松拼出整套房子的 3D 结构——因为你能同时回忆所有照片，对照"这个椅子在 A 房间角落出现，那个椅子在 B 房间角落出现，它们其实是同一把椅子"。

经典 SfM (Structure-from-Motion) 做这件事的方式很笨拙：先两两比对照片找特征点（SIFT/ORB），再 triangulation 算 3D 点，再 bundle adjustment 全局微调。每一步都积累误差，pipeline 一长，雪球越滚越大。

DUSt3R (Wang et al., CVPR 2024, https://dust3r.europe.naverlabs.com/) 聪明了一步：直接让 network 学会"两张图 → 每个像素的 3D 坐标"。不需要标定、不需要 pose、不需要 SIFT。但是它有个硬伤——**一次只能看两张图**。

24 张图怎么处理？两两配对，$\binom{24}{2} = 276$ 次推理，然后希望有一个 global optimization (GO) 把 276 个零碎结果拼成一个整体。问题是：

- 276 次里如果有几次配对就配错了（比如把 A 房间的窗户匹配到 B 房间的窗户），GO 只会旋转平移这些错误结果，**修不好**
- 276 次 inference 在 24-view 时耗时 27 秒，慢

Figure 2 里那个例子特别生动：场景是"三把椅子围着一张桌子 + 另一张桌子旁一把椅子"，DUSt3R+GO 重建出来变成"四把椅子全围着一张桌子"——pairwise 配对错乱了，GO 把错误固化了。

---

## 2. MV-DUSt3R 的核心想法：一次看完全部

**直觉**: 既然人脑能同时看所有照片，network 也应该能。把 DUSt3R 的两图 decoder 换成 N 图 decoder，让每张图的 token 在 transformer 里互相 cross-attention。

### 架构流程（Figure 3）

```
N 张 RGB 图
    ↓
ViT encoder (共享权重) → N 组 tokens, 每组分辨率是原图 1/16
    ↓
12 层 DecBlock: 每张图的 token 与其他所有图的 token 做 cross-attention
    ↓
Prediction head: 每张图 → 一个 H×W×3 的 pointmap + H×W 的 confidence
```

具体公式（论文 eq 1）：

$$F_d^v = \text{DecBlock}_d^{v}(F_{d-1}^v, \mathcal{F}_{d-1}^{-v})$$

解释每个符号：
- $F_d^v$: 第 $d$ 层 decoder 后，第 $v$ 张图的 token
- $F_{d-1}^v$: 第 $d-1$ 层时这张图自己的 token（**primary**）
- $\mathcal{F}_{d-1}^{-v} = \{F_{d-1}^1, \ldots, F_{d-1}^{v-1}, F_{d-1}^{v+1}, \ldots, F_{d-1}^N\}$: 其他所有 N-1 张图的 token（**secondary**）
- 上标 $v$ 表示 view index，下标 $d$ 表示 decoder 层数
- $\text{DecBlock}^{\text{ref}}$ vs $\text{DecBlock}^{\text{src}}$: reference view 和 source views 用不同权重的同款 block

**这里的关键洞察**: 当 N=2 时，MV-DUSt3R 数学上等价于 DUSt3R。N>2 时，每张图的 token 能"看到"所有其他图，这是 multi-view consistency 的天然 inductive bias。重复结构（同款椅子）可以通过其他 view 的上下文消歧——A 房间的椅子和 C 房间的椅子如果场景里只有一把，network 就能从 C 房间的视图里发现"这个椅子从 C 角度看长这样"，从而正确识别 A 房间那把。

### 训练 loss（eq 3a, 3b）

$$\mathcal{L}_{\text{conf}} = \sum_v \sum_{p \in P^v} \left[ C_p^{v,r} \cdot \ell_{\text{regr}}(v,p) - \beta \log C_p^{v,r} \right]$$

$$\ell_{\text{regr}}(v,p) = \left\| \frac{1}{z} X_p^{v,r} - \frac{1}{\bar{z}} \bar{X}_p^{v,r} \right\|$$

人话翻译：
- $X_p^{v,r}$: 网络预测的，第 $v$ 张图像素 $p$ 在 reference view $r$ 坐标系下的 3D 坐标
- $\bar{X}_p^{v,r}$: groundtruth 的对应 3D 坐标
- $C_p^{v,r}$: 网络对这个像素的"自信度"，0~1
- $z, \bar{z}$: 预测和 groundtruth 整体 scale 的归一化因子（所有 valid 3D 点到原点平均距离）
- $\beta = 0.2$: 防止 confidence 全部塌成 0 的正则

逻辑：对 confident 的像素，loss 主要看预测准不准；对 uncertain 的像素，network 可以输出小 confidence 减小 loss，但 $-\beta \log C$ 这一项惩罚 confidence 太小，逼 network 在能 confident 的地方尽量 confident。这种 adaptive weighting 让 network 自己学哪些像素靠谱。

### 速度对比（Table 2）

24-view HM3D 上：
- DUSt3R: 27.21 秒
- MV-DUSt3R: 0.35 秒（78× 加速）
- MV-DUSt3R+: 1.97 秒（14× 加速）

为什么这么快？因为 MV-DUSt3R 是 single forward pass，不用 276 次 pairwise inference。MV-DUSt3R+ 慢一点是因为有 M=4 个 path 并行跑，但仍然比 DUSt3R 快一个数量级。

---

## 3. MV-DUSt3R 的弱点：单 reference view 不够用

Figure 4 那张图特别说明问题：同一个 16-view 大场景，选不同的 reference view，重建质量在不同区域波动很大。

**直觉**: reference view 是"坐标系原点"，所有其他 view 的 pointmap 都要在它坐标系下表达。如果某个 source view 离 reference view 视角差很大（比如 reference 在客厅，source 在楼上卧室），它们之间 stereo cues 弱，network 容易猜错。

对大房子来说，**找不到一个 reference view 能与所有其他 view 都"看得到差不多的东西"**。总有些 view 离 reference 太远。

---

## 4. MV-DUSt3R+ 的核心想法：并行多参考系

**直觉**: 既然一个 reference view 不够，那就用 M=4 个 reference view 并行跑，让信息在它们之间流动。远处的 view 可以"借"近的 reference view 路径的高质量信息。

### Multi-Path 架构（Figure 5）

```
N 张图
    ↓ ViT encoder
    ↓
对每个 reference view r^m (m=1..M):
    12 层 DecBlock → 得到 G_d^{v,m}（第 v 张图在第 m 个 path 下的中间表示）
    ↓
    CrossRefViewBlock: G_d^{v,m} 与同一张图在其他 path 下的 G_d^{v,-m} 做 cross-attention
    ↓
最终每张图在每个 path 下都有一组融合后的 tokens
```

公式（eq 4-6）：

$$G_d^{v,m} = \text{DecBlock}_d(F_{d-1}^{v,m}, \mathcal{F}_{d-1}^{-v,m})$$

$$F_d^{v,m} = \text{CrossRefViewBlock}(G_d^{v,m}, \mathcal{G}_d^{v,-m})$$

- $G_d^{v,m}$: view $v$ 在 path $m$（reference 是 $r^m$）下的中间 token
- $\mathcal{G}_d^{v,-m}$: 同一个 view $v$ 在其他 M-1 个 path 下的 token 集合
- CrossRefViewBlock 架构跟 DecBlock 一样（cross-attn + self-attn + MLP），但权重独立

**关键设计**: CrossRefViewBlock 的 cross-attention/self-attention/MLP 最后一层 **zero-init**。这是 LayerScale 风格的 trick——训练初期这个 block 接近 identity，不破坏从 DUSt3R 初始化的预训练知识；随着训练进行，cross-path fusion 逐渐"开闸"。

### 为什么这个设计 work

考虑 view $v$ 离 reference $r^1$ 远，离 reference $r^2$ 近：
- Path 1 下，$G_d^{v,1}$ 质量差（视角差大，stereo cues 弱）
- Path 2 下，$G_d^{v,2}$ 质量好
- CrossRefViewBlock 让 path 1 的 token 从 path 2 "抄"高质量几何信息

这相当于 network 内置了一个 **implicit reference view selection**——不显式选最佳 reference，而是让所有 reference 的信息都流通。

### 推理时的取巧

训练时 M=4 个 path 都算 loss。推理时只用 path 1 的 head 输出最终结果。因为 path 1 已经通过 12 层 CrossRefViewBlock 吸收了其他 path 的信息，path 1 的预测相当于"融合所有 reference view 后的综合预测"。

这个设计避免了显式 ensemble M 个 path 的输出（那样会 4× 慢），而是让信息在 decoder 内部就融合好。

---

## 5. Gaussian Splatting Head：顺便做 NVS

### 直觉

既然每个像素都有 3D 坐标 $X^{v,m}$ 了，把它当 Gaussian center 就行。剩下 scale、rotation、opacity 让 network 预测。Color 直接用像素 RGB，spherical harmonics 用 degree 0（constant color）。

### 公式（eq 7, 8, S1）

每个像素预测：
- $S^{v,m} \in \mathbb{R}^{H \times W \times 3}$: Gaussian 的 3D scale
- $q^{v,m} \in \mathbb{R}^{H \times W \times 4}$: rotation quaternion
- $\alpha^{v,m} \in \mathbb{R}^{H \times W}$: opacity

渲染 target view $t^k$ 时，要把 Gaussian 从 reference view $r^m$ 坐标系变换到 $t^k$ 坐标系：

$$\hat{X}^{v,m \to k} = P^k (P^m)^{-1} \left( \frac{\bar{z}}{z} X^{v,m} \right)$$

- $P^m, P^k$: reference 和 target view 的 groundtruth pose
- $\bar{z}/z$: scale 校正（预测与 groundtruth 之间有 scale ambiguity）
- $\hat{X}^{v,m \to k}$: 变换后的 Gaussian center

**这里有个 caveat**: NVS 任务假设 poses 已知（用 groundtruth）。MVS/MVPE 任务才是 pose-free。所以严格说 MV-DUSt3R+ 不是"全 pose-free NVS"。这是一个细节但很重要——它没完全解决 NoPoSplat (https://arxiv.org/abs/2410.24207) 那种 pure pose-free NVS。

### 渲染 loss（eq S3, S4）

$$\mathcal{L}_{\text{render}} = \frac{1}{|T||M|} \sum_{k,m} \left( \|\hat{I}^{k,m} - I^k\|_2^2 + \gamma \text{LPIPS}(\hat{I}^{k,m}, I^k) \right)$$

$$\mathcal{L}_{\text{all}} = \mathcal{L}_{\text{conf}} + \delta \mathcal{L}_{\text{render}}$$

- $|T| = N + N'$: input views + novel views 总数
- $\gamma = 1$: LPIPS 权重
- $\delta = 1$: rendering loss 整体权重
- LPIPS: perceptual similarity (Zhang et al., CVPR 2018, https://richzhang.github.io/PerceptualSimilarity/)

### 为什么 joint training 不损害 MVS

Table 6 的 ablation: 加 Gaussian head 后 MVS 性能几乎不变。这意味着：
- Rendering loss 没有把 representation 容量从 pointmap head 抢走
- 反而可能因为 rendering 提供"额外监督信号"（要求 geometry 必须 render 出对的样子），对 pointmap 有微弱帮助

---

## 6. 实验结果怎么读

### MVS Reconstruction (Table 2)

24-view HM3D 上：
- DUSt3R CD = 32.4（重建烂）
- MV-DUSt3R CD = 10.0（好 3.2×）
- MV-DUSt3R+ CD = 3.9（再好 2.6×）

**直觉解释**: view 越多，DUSt3R 的 276 次 pairwise 配对里出错概率越高，GO 越拼不上。MV-DUSt3R 一次性看全，view 多反而信息多。MV-DUSt3R+ 的 multi-path 在大场景下尤其有用——多 reference view 覆盖不同区域。

### Pose Estimation (Table 3)

24-view HM3D 的 mAE：
- DUSt3R: 30.9%
- MV-DUSt3R+: 11.1%（提升 2.8×）

**直觉**: DUSt3R 的 pose 是从 pairwise pointmap 用 RANSAC+PnP 算出来的，pairwise 错误会污染 pose。MV-DUSt3R+ 的 pointmap 已经在同一坐标系下，pose 估计直接且准。

### NVS (Table 4)

4-view ScanNet 的 PSNR：
- DUSt3R baseline: 17.0 dB
- MV-DUSt3R: 21.9 dB
- MV-DUSt3R+: 22.2 dB

**直觉**: DUSt3R baseline 用启发式 Gaussian 参数（固定 scale=0.001, identity rotation, opacity=1），render 出来糊。MV-DUSt3R+ 学了 Gaussian 参数，且 pointmap 更准，所以清晰得多。5 dB PSNR 提升在 NVS 里是很大的差距。

### Oracle Analysis

论文做了 `MV-DUSt3R_oracle`（用 groundtruth 选最佳 reference view）和 `MV-DUSt3R+_oracle`（用 groundtruth 选最佳 path）。

观察：MV-DUSt3R 与其 oracle 差距大，MV-DUSt3R+ 与其 oracle 差距小。**这直接证明了 multi-path 设计有效**——MV-DUSt3R+ 接近"知道答案再选 reference"的天花板。

### 100-view Generalization (Figures S3-S6)

用 8-view 训练，直接喂 100-view 推理，居然 work。ScanNet 上 19.1 秒重建 100-view 场景。说明 architecture 的 inductive bias 不绑死训练时的 view 数量。这非常 impressive，也侧面说明 transformer 的"处理任意长度序列"特性在 3D 重建里也成立。

---

## 7. 我觉得有意思和存疑的地方

### 有意思

1. **CrossRefViewBlock 的 zero-init**: 这是 ConvNeXt v2、LayerScale 一脉相承的 trick。新加的模块初始时不破坏预训练知识，逐步学习。这种"渐进开闸"的设计在迁移学习里很实用。
2. **N=2 时退化为 DUSt3R**: 这个数学等价性让预训练权重迁移变得自然。不是"重新设计一个架构"，而是"扩展一个架构"。
3. **Inference 只用 path 1**: 训练时 4 个 path 都参与，推理只用 1 个。这是一种 information bottleneck——把多 path 信息压回单 path 输出。比 ensemble 高效。
4. **Overlap-based trajectory generation**: 公式 S7 用 3D 点云 nearest neighbor 距离算 overlap，控制 view 之间的视角差。这种数据筛选对评估 fairness 很关键。

### 存疑

1. **224×224 分辨率**: 现在的 3D 重建都在往 512、1024 走，224 的分辨率对真实场景细节（纹理、小物体）肯定不够。作者承认这是 future work，但训练 cost 是真实瓶颈。
2. **NVS 仍需 poses**: 公式 S1 用 groundtruth poses 做 Gaussian 变换。所以 NVS 严格说不是 pose-free。这是 NoPoSplat (https://arxiv.org/abs/2410.24207) 真正想解决的问题——直接预测 target view 坐标系下的 Gaussians。
3. **训练 cost**: 180 小时 × 64 H100。这相当于 ~11500 GPU-hours。对小团队来说复现门槛高。预训练 DUSt3R 权重的依赖也很重。
4. **Spann3R 对比可能不公平**: Spann3R (https://arxiv.org/abs/2408.16061) 原本为 sequential video 设计，sparse-view setting 对它本来就不友好。论文这个对比有点"打稻草人"味道。
5. **没有跟 MASt3R 对比**: MASt3R (https://arxiv.org/abs/2406.09756) 是 DUSt3R 的升级版，加入了 feature matching head。这里只跟 DUSt3R 比，没跟 MASt3R 比，可能是因为 MASt3R 仍是 pairwise 架构。
6. **Trajectory generation 的 threshold**: $t_c = 0.0015$ 这个 3D 距离阈值是场景 scale 相关的。不同数据集 scale 不同，这个阈值是否需要调整？论文没讨论。

---

## 8. 跟其他工作的关系

- **DUSt3R**: 这篇的直接前作和 baseline。https://dust3r.europe.naverlabs.com/
- **MASt3R**: DUSt3R + feature matching，仍是 pairwise。https://arxiv.org/abs/2406.09756
- **Spann3R**: DUSt3R + spatial memory，sequential online 重建。https://arxiv.org/abs/2408.16061
- **MASt3R-SfM**: MASt3R + gradient descent global optimization。https://arxiv.org/abs/2409.19152
- **NoPoSplat**: 真正的 pose-free NVS，直接预测 target view 坐标系下的 Gaussians。https://arxiv.org/abs/2410.24207
- **Splatt3R**: pose-free 2-view Gaussian splatting。https://arxiv.org/abs/2408.13912
- **pixelSplat / MVSPlat**: 2-view 或 sparse-view Gaussian splatting，但需要 poses。https://pixelsplat.github.io/, https://arxiv.org/abs/2406.06521
- **GS-LRM / GRM**: Large Reconstruction Model for Gaussian Splatting，需要 poses。https://arxiv.org/abs/2404.19498, https://arxiv.org/abs/2403.14621
- **3D Gaussian Splatting**: 基础渲染方法。https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **ViT**: encoder backbone。https://arxiv.org/abs/2010.11929

---

## 9. 一句话总结

DUSt3R 像"看完一对图打分，最后投票"; MV-DUSt3R 像"一次看完全班成绩单直接排名"; MV-DUSt3R+ 像"从四个不同维度排名然后交叉验证"。速度更快，准确度更高，reference view 的选择问题用 multi-path 架构优雅解决，NVS 顺手做出来还比 DUSt3R baseline 好 5 dB。代价是 224 分辨率和训练 cost。这是 sparse-view 3D 重建领域一个相当干净的工程进步。

---

# MV-DUSt3R+ 论文深度解析

## 1. 核心动机与问题定义

这篇paper 由 Meta Reality Labs 与 UIUC 团队提出，核心目标是用一个 **single-stage feed-forward network** 从 sparse pose-free RGB views 重建场景，速度比 DUSt3R 快 **48-78×**。

### DUSt3R 的根本缺陷

DUSt3R (Wang et al., 2024) 采用 pairwise 策略：每次处理一对 views，预测 pixel-aligned pointmap。处理 N 个 views 时需要 $\binom{N}{2}$ 次 pairwise inference，然后用一个 expensive 的 second-stage global optimization (GO) 来 align 局部重建。问题在于：

1. **Pairwise stereo ambiguity**: 两视图的 stereo cues 本身存在歧义，尤其当场景含有重复结构（窗户、椅子、门）时
2. **GO 无法修复 pairwise 错误**: GO 只能 rotate pairwise predictions，但无法 rectify 错误的 pairwise matches
3. **Combinatorial 复杂度**: $O(N^2)$ 次 pairwise inference 在 24-view 时耗时 27 秒

Figure 2 给出了一个典型 failure case：8-view 场景中三把椅子围绕一张桌子，外加另一张桌子旁的椅子。DUSt3R+GO 的重建结果把所有椅子都错误地围绕一张桌子——pairwise 错误被 GO 放大了。

### 形式化目标

给定 N 个 input views $\{I^v\}_{v=1}^N$，其中 $I^v \in \mathbb{R}^{H \times W \times 3}$，选择一个 reference view $r \in \{1, \ldots, N\}$，目标是预测 per-view 3D pointmaps $\{X^{v,r}\}_{v=1}^N$，其中 $X^{v,r} \in \mathbb{R}^{H \times W \times 3}$ 表示 view $I^v$ 中每个像素对应的 3D 点在 reference view $r$ 相机坐标系下的坐标。

---

## 2. MV-DUSt3R 架构详解

### 2.1 整体架构

参考 Figure 3 的 pipeline：

```
Input: N images {I^v}
    ↓
[ViT Encoder (shared weights)]  →  {F_0^v}  (分辨率降低16×)
    ↓
[DecBlock_ref + DecBlock_src × D layers]  (跨view token fusion)
    ↓
[Head_pcd^ref + Head_pcd^src]  →  {X^{v,r}, C^{v,r}}
```

### 2.2 Encoder

ViT encoder (shared weights) 作用于每个 input view：
$$F_0^v = \text{Enc}(I^v)$$

输出 token 分辨率是输入的 1/16，然后 flatten 成 token sequence。具体配置 (Table S1)：
- `Conv2d(in=3, out=1024, kernel=16, stride=16)` 做 patch embedding
- 16 个 `EncBlock(embed_dim=1024, n_head=16)`

### 2.3 Decoder Blocks — 关键创新

这是 MV-DUSt3R 与 DUSt3R 的核心差异。DUSt3R 的 decoder 只处理 2 个 view 的 tokens；MV-DUSt3R 的 decoder 处理 N 个 view 的 tokens。

两类 decoder block（架构相同，权重不同）：
- `DecBlock_d^ref`: 专门更新 reference view tokens $F^r$
- `DecBlock_d^src`: 更新所有 source view tokens $\{F^v\}_{v \neq r}$

每个 block 的输入：
- **Primary tokens**: 当前 view 自己的 tokens
- **Secondary tokens**: 所有其他 view 的 tokens $\mathcal{F}_d^{-v}$

block 内部流程：
1. Self-attention 仅作用于 primary tokens
2. Cross-attention 将 primary tokens 与 secondary tokens 融合
3. MLP 作用于 primary tokens
4. LayerNorm 在 attention 和 MLP 之前

形式化（公式 1）：

$$
F_d^v = \begin{cases} 
\text{DecBlock}_d^{\text{ref}}(F_{d-1}^v, \mathcal{F}_{d-1}^{-v}) & \text{if } v = r, \\
\text{DecBlock}_d^{\text{src}}(F_{d-1}^v, \mathcal{F}_{d-1}^{-v}) & \text{otherwise.}
\end{cases}
$$

其中 secondary tokens $\mathcal{F}_d^{-v} = \{F_d^1, \ldots, F_d^{v-1}, F_d^{v+1}, \ldots, F_d^N\}$。

**Intuition**: 这里关键的洞察是，当 N>2 时，每个 view 都能"看到"所有其他 view 的 tokens，从而能利用 multi-view 一致性，而 DUSt3R 只能看到一个 reference view。比如处理 view $v$ 时，cross-attention 把 view $v$ 的 tokens 与其他 N-1 个 view 的 tokens 融合，这种 global 上下文能有效消除 pairwise ambiguity。

### 2.4 Prediction Heads

两个 head（架构相同，权重不同）：
- `Head_pcd^ref`: 处理 reference view
- `Head_pcd^src`: 处理 source views

每个 head 包含：
1. Linear projection layer
2. PixelShuffle layer (upscale factor 16) 恢复原分辨率
3. (新加的) ConvNet skip connection from input image

公式 2：
$$
X^{v,r}, C^{v,r} = \begin{cases} 
\text{Head}_{\text{pcd}}^{\text{ref}}(F_D^v) & \text{if } v = r, \\
\text{Head}_{\text{pcd}}^{\text{src}}(F_D^v) & \text{otherwise.}
\end{cases}
$$

其中 $X^{v,r} \in \mathbb{R}^{H \times W \times 3}$ 是 3D pointmap，$C^{v,r} \in \mathbb{R}^{H \times W}$ 是 confidence map。

Skip connection 细节（公式 S5, S6）：
$$X_c^v = \text{PixelShuffle}(\text{Linear}(F_D^v))$$
$$X^v = \text{ConvNet}(X_c^v, I^v) + X_c^v$$

其中 ConvNet 包含 4 个 stride-1 卷积层（kernel size 分别为 3, 5, 5, 3）。这个 skip 把原始 RGB 信息注入到 coarse pointmap 上，用于恢复 fine-grained details。这是 MV-DUSt3R 相比 DUSt3R 的另一个改进点。

### 2.5 训练损失

Confidence-aware pointmap regression loss（公式 3a, 3b）：

$$\mathcal{L}_{\text{conf}} = \sum_{v \in \{1, \ldots, N\}} \sum_{p \in P^v} C_p^{v,r} \ell_{\text{regr}}(v, p) - \beta \log C_p^{v,r}$$

$$\ell_{\text{regr}}(v, p) = \left\| \frac{1}{z} X_p^{v,r} - \frac{1}{\bar{z}} \bar{X}_p^{v,r} \right\|$$

变量解释：
- $P^v$: view $v$ 中 valid pixels 集合（有 groundtruth 3D point 的像素）
- $C_p^{v,r}$: 像素 $p$ 在 view $v$ 的 confidence 预测
- $\beta$: 正则化权重，控制 confidence 的熵（论文中 $\beta=0.2$）
- $X_p^{v,r}$: 像素 $p$ 的预测 3D point
- $\bar{X}_p^{v,r}$: 像素 $p$ 的 groundtruth 3D point
- $z = \text{norm}(\mathcal{X}^{\{v\}, r})$: 预测 pointmap 的 scale normalization factor（所有 valid 3D 点到原点的平均距离）
- $\bar{z} = \text{norm}(\bar{\mathcal{X}}^{\{v\}, r})$: groundtruth 的 scale normalization factor

**Intuition on the loss design**: 
- 第一项 $C_p^{v,r} \ell_{\text{regr}}$ 是 confidence-weighted regression，让网络对 confident pixels 更关注
- 第二项 $-\beta \log C_p^{v,r}$ 是 log-barrier，防止 confidence 全部 collapse 到 0
- Scale normalization $\frac{1}{z}, \frac{1}{\bar{z}}$ 处理 prediction 与 groundtruth 之间的 scale ambiguity
- 整个 loss 鼓励网络：对 difficult pixels 输出低 confidence，对 easy pixels 输出高 confidence 并准确预测

### 2.6 与 DUSt3R 的特殊关系

当 $N=2$ 时，MV-DUSt3R 退化为 DUSt3R。但因为参数量几乎相同（只多了 skip connection 和 ConvNet），可以从 DUSt3R 预训练权重初始化。这一点对训练 efficiency 至关重要——180 小时 × 64 H100 GPU 是已经基于 DUSt3R 初始化的训练 cost。

---

## 3. MV-DUSt3R+ — Multi-Path 架构

### 3.1 动机：单 reference view 的局限

Figure 4 揭示了 MV-DUSt3R 的局限：reference view 选择严重影响重建质量。当 source view 与 reference view 视角变化大时，stereo cues 不足，重建质量差。对 large scenes (multi-room)，找不到一个 reference view 能与所有 source view 都有 moderate viewpoint change。

### 3.2 Multi-Path 架构

设 $R = \{r^m\}_{m=1}^M$ 为从 input views 中随机选择的 M 个 reference views。MV-DUSt3R+ 在 multi-path 架构中部署相同的 decoder blocks（公式 4, 5）：

$$
G_d^{v,m} = \begin{cases} 
\text{DecBlock}_d^{\text{ref}}(F_{d-1}^{v,m}, \mathcal{F}_{d-1}^{-v,m}) & \text{if } v = r^m, \\
\text{DecBlock}_d^{\text{src}}(F_{d-1}^{v,m}, \mathcal{F}_{d-1}^{-v,m}) & \text{otherwise.}
\end{cases}
$$

这里 $F_{d-1}^{v,m}$ 是 view $v$ 在第 $m$ 个 path（即 reference view $r^m$）下的 token 表示。

### 3.3 Cross-Reference-View Block — 核心创新

在每个 decoder block 之后，添加一个 `CrossRefViewBlock`（公式 6）：

$$F_d^{v,m} = \text{CrossRefViewBlock}_d(G_d^{v,m}, \mathcal{G}_d^{v,-m})$$

其中 $\mathcal{G}_d^{v,-m} = \{G_d^{v,1}, \ldots, G_d^{v,m-1}, G_d^{v,m+1}, \ldots, G_d^{v,M}\}$ 是同一 view $v$ 在其他 path 下的中间表示。

**架构细节**: CrossRefViewBlock 与 DecBlock 共享架构（cross-attention + self-attention + MLP），但参数随机初始化（除了 cross-attention, self-attention 和 MLP 的最后一层 zero-initialized，这是 LayerScale 风格的初始化，让 block 初始时接近 identity）。

**Intuition**: 这个 block 是 long-range information propagation 的核心机制。考虑 view $v$ 在 path $m$ 下与 reference view $r^m$ 视角差很大，但在 path $m'$ 下与 reference view $r^{m'}$ 视角接近。通过 CrossRefViewBlock，path $m$ 的 tokens 可以从 path $m'$ 借用高质量的几何信息。这相当于让网络"考虑"多个 reference view 选择，并融合它们的优势。

### 3.4 训练与推理

**训练**: 随机选择 M=4 个 views 作为 reference views，对每个 reference view 都计算 pointmap regression loss 并平均。

**推理**: 均匀选择 M 个 views 作为 reference views，第一个 view 总是选中。使用 M-path model，但只用第 1 个 path 的 head 输出最终预测。

**Intuition on inference strategy**: 第一个 path 的 head 已经通过 CrossRefViewBlock 从其他 path 吸收了信息，所以第 1 个 path 的输出相当于"所有 reference views 信息融合后的预测"。这是一种 implicit 的 reference view selection——而不是显式 ensemble M 个 path 的输出。

---

## 4. Gaussian Splatting Head — 支持 NVS

### 4.1 预测的 Gaussian 参数

每个像素预测的 Gaussian 参数（公式 7, 8）：
- Scaling factor $S^{v,m} \in \mathbb{R}^{H \times W \times 3}$ (3D scale)
- Rotation quaternion $q^{v,m} \in \mathbb{R}^{H \times W \times 4}$
- Opacity $\alpha^{v,m} \in \mathbb{R}^{H \times W}$

其他参数：
- Center: 用预测的 pointmap $X^{v,m}$
- Color: 用 pixel color $I^v$
- Spherical harmonics degree: 0（即只用 constant color）

### 4.2 Gaussian 变换与渲染

对 target view $t^k$，需要把 Gaussian 从 reference view $r^m$ 坐标系变换到 $t^k$ 坐标系（公式 S1）：

$$\hat{X}^{v,m \to k} = P^k (P^m)^{-1} \left( \frac{\bar{z}}{z} X^{v,m} \right)$$

变量解释：
- $P^m$: reference view $r^m$ 的 groundtruth camera pose
- $P^k$: target view $t^k$ 的 groundtruth camera pose
- $\bar{z}/z$: scale 校正因子（解决 prediction 与 groundtruth 的 scale ambiguity）
- $\hat{X}^{v,m \to k}$: 变换后的 Gaussian center

注意：训练时使用 groundtruth poses 做这个变换，**inference 时** 也需要 poses 来做 view transformation。论文这里的 setup 是 NVS 任务下已知 poses 用于 view transformation，但 reconstruction 任务下不需要 poses。这是一个值得注意的 subtle point。

渲染（公式 S2）：
$$\hat{I}^{k,m} = \text{Rendering}(Q^{1,m \to k}, \ldots, Q^{N,m \to k})$$

其中 $Q^{v,m \to k} = \{I^v, \hat{X}^{v,m \to k}, \hat{S}^{v,m}, q^{v,m \to k}, \alpha^{v,m}\}$ 是变换后的 Gaussian 参数集。

### 4.3 渲染损失

公式 S3：
$$\mathcal{L}_{\text{render}} = \frac{1}{|T||M|} \sum_{k,m} \left( \|\hat{I}^{k,m} - I^k\|_2^2 + \gamma \text{LPIPS}(\hat{I}^{k,m}, I^k) \right)$$

其中 $|T| = N + N'$（input views + novel views），$\gamma=1$ 控制 LPIPS 权重。

总损失（公式 S4）：
$$\mathcal{L}_{\text{all}} = \mathcal{L}_{\text{conf}} + \delta \mathcal{L}_{\text{render}}$$

其中 $\delta=1$。

**Intuition on joint training**: Pointmap regression loss 监督几何精度，rendering loss 监督外观与几何的一致性。两者联合训练时，Gaussian heads 反过来也帮助 pointmap head——因为渲染需要准确的 geometry 才能产生 sharp images。Table 6 显示加入 Gaussian head 后 MVS 性能基本持平（略有变化但可忽略），说明 rendering loss 不损害重建质量，反而赋予了 NVS 能力。

---

## 5. 数据与 Trajectory Generation

### 5.1 评估数据集（Table 1）

| Dataset | Setting | Scene Type |
|---------|---------|-----------|
| HM3D | Supervised | multi-room (large) |
| ScanNet | Supervised | single-room (small) |
| MP3D | Zero-shot | multi-room & outdoor (largest) |

### 5.2 Trajectory Generation 算法

这是论文的一个 underrated 但关键的细节。生成 trajectories 的核心是 overlap ratio 控制（公式 S7a, S7b）：

$$O(X^i, X^j) = \frac{1}{2} (\text{Cov}(X^i, X^j) + \text{Cov}(X^j, X^i))$$

$$\text{Cov}(X^i, X^j) = \frac{1}{|A|} \sum_{p \in X^i} [\text{NearestDis}(p, X^j) < t_c]$$

变量解释：
- $O(X^i, X^j)$: view $i$ 与 view $j$ 的 overlap ratio（symmetric）
- $\text{Cov}(X^i, X^j)$: view $i$ 中有多少比例的 3D 点离 view $j$ 的最近 3D 点距离小于阈值 $t_c = 0.0015$
- $|A|$: view $i$ 中的点数

接受 criterion：新 candidate view $X^i$ 的最大 overlap ratio $\max_j \{O(X^i, X^j)\}$ 必须在 $[t_{\min}, t_{\max}]$ 之间。

**两套 thresholds**:
- $(t_{\min}, t_{\max}) \in \{(30\%, 70\%), (30\%, 100\%)\}$ for ScanNet/ScanNet++
- $(30\%, 70\%)$ only for HM3D/Gibson（避免 overfitting）

**Intuition on overlap control**: 
- $t_{\min}$ 保证 view 之间有足够 overlap，使得 stereo cues 可用
- $t_{\max}$ 保证 view 之间视角变化足够大，避免退化的 trivial cases
- 这个 trajectory generation 是 sparse-view evaluation 的关键，决定 difficulty

训练数据规模：3.2M trajectories from ScanNet/ScanNet++，7.8M trajectories from HM3D/Gibson。

---

## 6. 实验结果深度分析

### 6.1 MVS Reconstruction (Table 2)

关键 metric：
- **CD (Chamfer Distance)**: 越低越好，括号内为 median
- **ND (Normalized Distance)**: $\ell_{\text{regr}}$ with zero-centering，scale & translation invariant
- **DAc (DistanceAccu@0.2)**: normalized distance ≤ 0.2 的像素比例

4-view results on HM3D:
| Method | ND↓ | DAc↑ | CD↓ | Time |
|--------|-----|------|-----|------|
| Spann3R | 37.1 | 0.0 | 225(184) | 0.36s |
| DUSt3R | 1.9 | 75.1 | 5.6(2.3) | 2.42s |
| MV-DUSt3R | 1.1 | 92.2 | 2.0(1.1) | 0.05s |
| MV-DUSt3R+ | **1.0** | **95.2** | **1.5(0.9)** | 0.29s |

24-view results on HM3D:
| Method | ND↓ | DAc↑ | CD↓ | Time |
|--------|-----|------|-----|------|
| DUSt3R | 6.8 | 7.3 | 32.4(5.2) | 27.21s |
| MV-DUSt3R | 3.4 | 36.7 | 10.0(3.5) | 0.35s |
| MV-DUSt3R+ | **2.1** | **64.5** | **3.9(2.0)** | 1.97s |

**关键观察**:
1. **Speed**: MV-DUSt3R 在 24-view 上 78× faster (0.35s vs 27.21s)。MV-DUSt3R+ 在 24-view 上 14× faster
2. **View 数量越多，MV-DUSt3R 优势越明显**: 4-view 时 DUSt3R 还能 GO 对齐，但 24-view 时 pairwise 错误累积让 GO 失效
3. **MV-DUSt3R+ 在 large scenes + many views 优势显著**: 24-view HM3D CD 从 10.0 降到 3.9，提升 2.6×
4. **Spann3R 灾难性失败**: 因为 spatial memory 大小有限，sparse-view 下 drift 严重

### 6.2 Oracle Analysis

`MV-DUSt3R_oracle`: 用 groundtruth 选最佳 reference view
`MV-DUSt3R+_oracle`: 用 groundtruth 选最佳 path

观察：
- MV-DUSt3R 与其 oracle 之间 gap 较大 → 单 reference view 选择敏感
- MV-DUSt3R+ 与其 oracle 之间 gap 很小 → multi-path 架构有效缓解 reference view 选择问题

### 6.3 Multi-View Pose Estimation (Table 3)

Metric:
- RRE@15 = 1 - RRA@15 (Relative Rotation Accuracy)
- RTE@15 = 1 - RTA@15 (Relative Translation Accuracy)
- mAE@30 = 1 - mAA@30 (mean Average Accuracy)

Camera intrinsics 估计用 Weiszfeld algorithm (公式 S8)：
$$f_1^* = \arg\min_{f_1} \sum_{(i,j)} C_{i,j}^{1,1} \left\| (i', j') - f_1 \frac{(\bar{X}_{i,j,0}^{1,1}, \bar{X}_{i,j,1}^{1,1})}{\bar{X}_{i,j,2}^{1,1}} \right\|_2^2$$

假设：principal point 在图像中心，pixels 是 square。

Pose 估计用 RANSAC + PnP 基于 2D pixels 与 3D pointmap 的对应关系。

4-view HM3D results:
| Method | RRE↓ | RTE↓ | mAE↓ |
|--------|------|------|------|
| DUSt3R | 2.4 | 3.1 | 12.5 |
| MV-DUSt3R | 1.5 | 1.5 | 5.5 |
| MV-DUSt3R+ | **1.2** | **1.1** | **4.9** |

24-view HM3D:
| Method | RRE↓ | RTE↓ | mAE↓ |
|--------|------|------|------|
| DUSt3R | 8.8 | 18.1 | 30.9 |
| MV-DUSt3R+ | **1.4** | **2.4** | **11.1** |

mAE 提升 2.8× (12.5→4.9) 和 2.0× (30.9→15.8)，这是大幅改进。原因：MV-DUSt3R 直接预测同一坐标系下的 pointmaps，poses 可以从 pointmap 对应关系直接估计，避免了 pairwise pointmaps 之间 GO 对齐的误差。

### 6.4 Novel View Synthesis (Table 4)

Metric: PSNR, SSIM, LPIPS

4-view ScanNet:
| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| DUSt3R baseline | 17.0 | 6.0 | 3.0 |
| MV-DUSt3R | 21.9 | 7.1 | 1.6 |
| MV-DUSt3R+ | **22.2** | **7.1** | **1.5** |

PSNR 提升 5.2 dB——这是 substantial improvement。原因：DUSt3R baseline 用启发式 Gaussian 参数（constant scale 0.001, identity rotation, opacity 1.0, SH degree 0），而 MV-DUSt3R+ 通过 joint training 学习 Gaussian 参数。

### 6.5 Ablation: Training Recipe (Table 5)

HM3D 24-view:
| Recipe | ND↓ | DAc↑ | CD↓ |
|--------|-----|------|-----|
| 1-stage, 4 views | 17.7 | 0.0 | 81.4 |
| 1-stage, 8 views | 2.1 | 64.5 | 3.9 |
| 2-stage, mixed | **1.7** | **81.4** | **2.6** |

**Intuition**: 
1. 1-stage 4-view 训练完全无法泛化到 24-view（CD 81.4 vs 2.6）
2. 1-stage 8-view 训练 decent，但在 more views 时退化
3. 2-stage training（先 8-view，再 mixed 4-12 views fine-tune）最好，因为它让 model 见过不同 view 数量

### 6.6 100-view Generalization (Figures S3-S6)

MV-DUSt3R+ 用 8-view 训练，能泛化到 100-view input！ScanNet 100-view 推理时间 19.1 秒。这是一个 impressive 的 generalization 结果，说明 architecture 设计的 inductive bias 是 N-agnostic 的。

---

## 7. 关键 Architecture 对比 (Table S1)

| Module | DUSt3R | MV-DUSt3R+ |
|--------|--------|------------|
| Enc | Conv2d + 16×EncBlock(1024, 16) | 同 |
| Dec | Linear + 12×DecBlock(768, 12) | Linear + 12×DecBlock(768, 12) + **12×CrossRefViewBlock(768, 12)** |
| Head_pcd | Linear + PixelShuffle | Linear + PixelShuffle + **ConvNet skip** |
| Head_3DGS | - | Linear + PixelShuffle |

新增参数：CrossRefViewBlock × 12 + ConvNet skip。论文称这些只是 marginal 参数增加，能从 DUSt3R 权重初始化大部分。

---

## 8. 关键 Insights 总结

### 8.1 为什么 single-stage 优于 pairwise + GO

DUSt3R 的 GO 阶段只是 rigid alignment，无法 fix pairwise 错误。当场景有 repeated structures（同款椅子、窗户），pairwise matching 容易混淆，GO 把这些错误"锁定"到 global coordinate system。MV-DUSt3R 的 multi-view decoder 让网络同时看到所有 views，在 token-level 就能 disambiguate。

### 8.2 为什么 multi-reference (MV-DUSt3R+) 进一步提升

Single reference view 在 large scenes 中无法与所有 source views 都有 small viewpoint change。Multi-path 架构相当于让网络"考虑"多个 reference view 候选，通过 CrossRefViewBlock 在 path 之间 exchange 信息。这类似于 implicit ensemble，但参数共享所以高效。

### 8.3 初始化的重要性

DUSt3R 预训练权重让 MV-DUSt3R+ 不需要从零训练 3D 几何先验。Zero-init 的 CrossRefViewBlock 也让 multi-path 在训练初期接近 single-path MV-DUSt3R，逐步学习 cross-path fusion。

### 8.4 Gaussian Head 不损害 MVS 性能

Table 6 显示加入 GS head 后 MVS 性能基本不变。这说明 joint training 不让 GS head "偷走" representation 容量，反而可能因为 rendering loss 提供额外 supervision 信号而略微帮助。这是一个 nice property。

### 8.5 Resolution 限制

论文固定在 224×224 输入分辨率。作者承认 512 分辨率 DUSt3R 更精细，但训练 cost 太高，留作 future work。这是当前方法的一个 limitation。

---

## 9. 局限与 Future Directions

1. **Resolution**: 224×224 限制细节重建，未来需要 scaling 到 512
2. **Diffusion priors**: 论文未利用 generative priors（InFusion, ZeroNVS 等），对 unseen regions 处理有限
3. **NVS 需要 poses**: 渲染时需要 groundtruth poses 做 Gaussian transformation，未完全 pose-free
4. **训练 cost**: 180 小时 × 64 H100，规模化挑战

---

## References

- 论文: https://mv-dust3rp.github.io/
- DUSt3R: Wang et al., CVPR 2024, https://dust3r.europe.naverlabs.com/
- MASt3R: Leroy et al., 2024, https://arxiv.org/abs/2406.09756
- Spann3R: Wang & Agapito, 2024, https://arxiv.org/abs/2408.16061
- 3D Gaussian Splatting: Kerbl et al., ACM TOG 2023, https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- HM3D: Ramakrishnan et al., 2021, https://arxiv.org/abs/2109.08238
- ScanNet: Dai et al., CVPR 2017, http://scannet.cs.princeton.edu/
- MP3D: Chang et al., 2017, https://arxiv.org/abs/1709.06158
- ViT: Dosovitskiy et al., 2020, https://arxiv.org/abs/2010.11929
- PoseDiffusion: Wang et al., ICCV 2023, https://posediffusion.github.io/
- Weiszfeld algorithm: Plastria, 2011
- LPIPS: Zhang et al., CVPR 2018, https://richzhang.github.io/PerceptualSimilarity/
- RANSAC + PnP: Fischler & Bolles 1981; Lepetit et al. IJCV 2009
