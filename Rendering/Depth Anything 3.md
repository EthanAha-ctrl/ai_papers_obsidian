---
source_pdf: Depth Anything 3.pdf
paper_sha256: afa74735e20b667b91183f604224abbcd61d4f927c23a3c635015d0c9d83d083
processed_at: '2026-08-03T19:55:31-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Depth Anything 3 人话版

好，咱们抛开公式，用大白话重新讲一遍这个工作的本质。

---

## 一句话讲 DA3 在干什么

给它任意数量的图片（单张也行、几十张也行），它直接吐出每张图的深度图和相机位姿，把 3D 空间恢复出来。整个过程一次 forward pass 搞定，不需要 COLMAP 那种迭代式 pipeline，也不需要针对每个任务单独训一个 specialized model。

---

## 它面对的根本问题

过去 3D vision 这个领域有个怪现象：**任务定义被人为切割了**。

Monocular depth、SfM、MVS、SLAM、NVS——本质上都是"从图片恢复 3D 结构"，只是输入图片数量不同、有没有已知 pose 不同。但社区给每个任务单独发明了 architecture、loss、benchmark，训出来的 model 互相不能复用。

后来 DUSt3R、VGGT 这些工作开始想做 unified model，但它们的做法是**堆架构**：多个 transformer 串联、多个 head 并联、多个 loss 联合优化，从 scratch 训。结果就是虽然能统一多个任务，但**完全浪费了已有的 pretrained features**（DINOv2 这种）。

DA3 的作者反过来问：**如果回到最根本的目标——从任意视觉输入恢复 3D 空间——最少需要什么？**

两个问题：
1. 预测目标能不能最小化？非要多任务吗？
2. 单个 plain transformer 够不够？非得定制架构吗？

答案都是 yes——最小化就够了，plain transformer 够了。

---

## 第一个 insight：Depth-Ray 表示

这是整个 paper 的灵魂，讲清楚它就懂了一大半。

### 传统相机投影在干嘛

一个像素 p 想变成 3D 点 P，标准公式是：

```
P = R × ( D × K⁻¹ × p ) + t
```

这里 R 是 3×3 旋转矩阵，K 是相机内参，D 是这个像素的深度，t 是相机中心。

问题出在 R 上。R 必须满足正交约束（R^T R = I），神经网络直接预测 9 个数很难保证这个约束。传统 workaround 是用 quaternion 或者 6D representation，但本质上还是在预测一个**全局的 per-image 参数**。

### DA3 的 trick

作者观察到一件事：你其实不需要直接预测 R 和 K，你需要的是**每个像素对应的那条射线**。

一条射线由两个东西定义：
- **起点 t**：射线从哪发出（就是相机中心）
- **方向 d**：射线往哪走（d = R × K⁻¹ × p，一个 3D 向量）

如果你对每个像素都预测 (t, d)，再加一个 depth 标量 D，那么 3D 点就是：

```
P = t + D × d
```

就这么简单，element-wise 操作，几何完全一致。

关键 design choice：**d 不归一化**。保留它的 magnitude，这样它编码了 K 和 R 的 scale 信息。

### 为什么这个表示这么好用

直觉上有三个原因：

**第一，把全局参数变成了 dense 预测。** 以前预测一个相机的 9-DoF 参数，是 9 个数决定一整张图。现在每个像素都预测一个 ray，transformer 做 dense prediction 本来就擅长，监督信号也密集，学起来容易得多。

**第二，depth 和 ray 解耦。** Depth 是个标量，物理意义清晰，监督简单（L1 loss 就行）。Ray 编码了位姿和内参，让网络自己学怎么用。两者乘一下就是 3D 点，完全可微。

**第三，从 ray map 反推 (K, R, t) 是 closed-form。** 推理时如果你真的需要相机参数（比如做 NVS），可以用 DLT 算法 + RQ 分解从 dense ray map 恢复出来。这是个标准的 least-squares 问题，不需要学习。

Ablation 里这个优势非常明显：depth + ray 比 depth + point map + cam 在 pose accuracy 上提升近 100%。而且加一个 auxiliary cam head 反而没帮助——说明 depth + ray 已经是 **minimal sufficient** 的目标。

---

## 第二个 insight：Plain Transformer 够了

### 之前的做法

VGGT 用两个不同的 transformer 堆叠，总共 1.19B 参数。但其中大约 2/3 的 blocks 是**没有预训练的**——它从 scratch 训这些 layers。

这等于把 DINOv2 学到的强大 visual features 给丢了。

### DA3 的做法

直接用 vanilla DINOv2 ViT，一个 backbone，L 个 blocks，**不做任何架构修改**。

那 cross-view reasoning 怎么做？**token 重排**。

把 L 个 layers 分两组：
- 前 L_s = 2L/3 层：每张图内部做 self-attention（within-view）
- 后 L_g = L/3 层：交替 cross-view 和 within-view

cross-view attention 就是把 token tensor 从 `(N_views, H×W, C)` 重排成 `(H×W, N_views, C)`，让不同图的同一位置 token 互相 attend。within-view 再排回去。纯 tensor 操作，零架构改动。

### 为什么 partial 比 full alternation 好

Ablation 里有个有意思的发现：如果所有层都做 cross-view attention（full alternation），性能反而掉。只有部分层做 cross-view（2:1 比例）才最好。

直觉是这样：DINOv2 的 pretrained features 是在单图上学的，它强大的 within-view 表示能力是**基础**。如果你从第一层就开始 cross-view，会破坏这个基础。让前 2/3 充分提取单图特征，后 1/3 再做 cross-view reasoning，这样既继承了 pretrained features，又获得了多视角一致性。

这跟 LLM 里用 pretrained backbone 做 downstream task 的哲学一样：**别动 pretrained 的部分，只在最后加 task-specific head**。

### 单图输入的优雅退化

这个设计还有个 bonus：输入单张图时，cross-view attention 自然退化成 within-view，模型自动变成一个 monocular depth estimator，没有额外开销。一个 model 同时处理单图和多图，不用切分支。

---

## 第三个 insight：Teacher-Student 解决数据困境

### 问题

Real-world depth 数据有个致命问题：**质量太差**。

LiDAR 深度图稀疏、有空洞；COLMAP 重建不完整、有噪声。Figure 4 展示了各种数据集的惨状——有些深度图甚至和 RGB 图对不齐。

但 real-world 数据又是必须的，因为你要训 pose estimation，synthetic 数据的 pose 分布太窄。

### 解法

典型的 teacher-student：

1. **在 synthetic data 上训一个强大的 monocular teacher**。Synthetic data 有完美的 GT depth，覆盖 indoor/outdoor/object 各种场景。
2. **Teacher 给 real-world data 生成 dense pseudo-depth**。这些 pseudo-depth 质量高、细节丰富。
3. **用 RANSAC scale-shift alignment 把 pseudo-depth 对齐到 real-world 的 noisy GT**。对齐公式就是解一个 least-squares：minimize ||s × D̃ + t - D||²，用 RANSAC 防止 outlier 污染。

这样 student model 同时拿到了：
- Real-world 数据的多样性（pose 分布、场景类型）
- Synthetic data 的细节质量（teacher 传递过来）

Ablation 里去掉 teacher supervision，HiRoom 上 F1 从 47.0 掉到 16.0。Qualitative 对比也很直观：有 teacher 的 depth map 细节明显丰富很多。

### Teacher 本身也有改进

Teacher 基于 Depth Anything 2，但有几个关键升级：

1. **预测 depth 而非 disparity**。Disparity 在近处太敏感、远处太平坦，depth 更适合做 multi-view geometry。
2. **预测 exponential depth**。近相机区域 depth 变化小，指数化以后近处被放大，判别力更强。
3. **加了 distance-weighted surface normal loss**。用邻近点算 normal，按距离加权，让局部几何更精确。
4. **数据规模大幅扩张**。从 DA2 的几个 synthetic dataset 扩到 20 个，覆盖更广。

---

## 一个顺带的发现：Geometry 是 NVS 的真瓶颈

这部分我觉得最有意思，因为它揭示了一个 field-level 的 lesson。

### 实验

作者在 feed-forward novel view synthesis（FF-NVS）上做了系统对比。把不同的 geometry backbone（Fast3R, MV-DUSt3R, VGGT, DA3）接上同一个简单的 GS-DPT head，训 3D Gaussian Splatting，跟专门为 NVS 设计的架构（pixelSplat, MVSplat, DepthSplat）比。

结果（DL3DV benchmark PSNR）：
- pixelSplat: 16.55
- MVSplat: 18.13
- DepthSplat: 19.24
- Fast3R backbone: 19.30
- MV-DUSt3R: 20.01
- VGGT: 20.96
- **DA3: 21.33**

### 这个结果说明了什么

**Geometry model 越强，NVS 越好**。而且用简单 backbone + DPT head 的组合，全面超越了那些精心设计的 NVS-specific 架构（epipolar attention、cost volume、cascaded modules 之类的）。

这跟 LLM 领域的 lesson 完全一致：**用大 pretrained backbone + 简单 task head，通常 beats 从头设计的 task-specific 架构**。因为 pretraining 带来的 generalization 和 scalability 是 task-specific engineering 给不了的。

所以 future NVS 的方向应该是：搞更好的 geometry foundation model，而不是发明更巧的 NVS 架构。

---

## 和 LLM Foundation Model 的共鸣

Andrej，你应该会 appreciate 这个 paper 的整体哲学。

它在 3D vision 领域做的事，本质上跟 LLM 领域这几年做的事一样：

1. **找到正确的 minimal target representation**。就像 GPT 用 next-token prediction 统一了所有 NLP 任务，DA3 用 depth + ray 统一了所有 3D vision 任务。不需要多任务 learning，一个目标足够。
2. **Leverage pretrained features，别重新发明轮子**。DINOv2 已经学会了 visual priors，直接用就行。VGGT 重新堆 transformer 等于浪费了这个 free lunch。
3. **Scale and simplicity > architectural sophistication**。Plain transformer + token rearranging 比 bespoke multi-stage architecture 更好训、更好 scale、更好继承 pretraining。
4. **Foundation model 思维渗透到下游任务**。NVS 不需要 task-specific 架构，只需要一个 strong geometry backbone 加简单 head。

唯一不同的是，3D vision 的 representation choice 比 NLP 难找得多。NLP 天然有 token 这个离散单元，3D vision 的 "atomic target" 是什么——point map？depth + pose？还是 depth + ray？——需要摸索。DA3 的贡献就是找到了 depth + ray 这个 minimal sufficient 的答案。

---

## 一些值得细想的点

1. **为什么不归一化 d？** 归一化 d 会丢失 scale 信息，导致 depth 和 ray 的 magnitude 不匹配。保留 magnitude 让 K 的 scale 被编码进 d，整个系统自洽。

2. **从 ray map 反推 (K, R) 的 homography trick 很巧妙。** 定义一个 identity camera，它的 ray 就是 pixel 本身。target camera 的 ray 是 K R × pixel。所以 K R 就是一个把 identity rays 映射到 target rays 的 homography。用 DLT 解这个 homography，再 RQ 分解出 K 和 R。整个过程 closed-form，不需要学习。

3. **Pose conditioning 的 scaling behavior 有意思。** Ablation 显示有已知 pose 时，model scale 的收益变小。直觉上 pose estimation 是更难的任务，需要更大 model 才能充分受益；depth estimation 在中等 model 就接近饱和。如果你 future 想重点攻 pose，需要 scale up。

4. **Dual-DPT head 的 shared reassembly 是关键。** 两个 task share 前面的 feature processing，只在最后 fusion 阶段分开。这鼓励 features 对齐，避免 redundant representation。去掉 shared design，HiRoom Auc3 从 39.2 掉到 5.59，差距恐怖。

5. **Partial cross-view attention 保护了 pretrained features。** 前 2/3 within-view，后 1/3 cross-view。Full cross-view 会破坏 DINOv2 学到的单图表示基础。

---

## Web Links

- DA3 Project Page: https://depth-anything-3.github.io
- DINOv2: https://arxiv.org/abs/2304.07193
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- VGGT: https://arxiv.org/abs/2503.11651
- DUSt3R: https://arxiv.org/abs/2312.14132
- Pi3: https://arxiv.org/abs/2507.13347
- Fast3R: https://arxiv.org/abs/2501.13928
- MapAnything: https://arxiv.org/abs/2509.13414
- MoGe (global-local loss 来源): https://arxiv.org/abs/2503.21717
- Metric3D v2: https://arxiv.org/abs/2404.15506
- UniDepth v2: https://arxiv.org/abs/2502.20110
- DepthPro: https://arxiv.org/abs/2410.02073
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14737
- pixelSplat: https://arxiv.org/abs/2312.12381
- MVSplat: https://arxiv.org/abs/2403.14677
- DepthSplat: https://arxiv.org/abs/2412.00062
- DL3DV-10K: https://arxiv.org/abs/2406.03263
- Hartley & Zisserman, Multiple View Geometry (DLT, RQ decomposition): http://cvrs.whu.edu.cn/courses/MVG/A_SolutionManual.pdf
- Umeyama 1991 (相似变换对齐): https://web.stanford.edu/class/cs273/refs/umeyama.pdf

---

总结一句人话：DA3 告诉我们，3D vision 不需要那么多花活，找到一个好的 minimal representation（depth + ray），用一个 plain pretrained transformer，加个 teacher-student 把数据质量搞上去，就能统一所有 3D 任务还拿到 SOTA。这跟 LLM 里 "simple objective + big pretrained model + scale" 的配方是同一个配方。

---

# Depth Anything 3: Recovering the Visual Space from Any Views

很高兴为你详细讲解这篇 paper。我会从 motivation 出发，逐步拆解每个技术组件，重点 build intuition about **为什么这些 minimal design choices 能 work**。

---

## 1. Core Motivation: 回归 "Spatial Intelligence" 的根本目标

Andrej，你应该很 appreciate 这个 paper 的哲学起点。它 step back from 传统 3D vision 任务的定义（monocular depth, SfM, MVS, SLAM），回归到一个更 fundamental 的目标：**从任意视觉输入（单图、多视图、视频）恢复 3D visual space**。

传统 paradigm 的问题：
- 每个任务用 specialized model（MVSNet for MVS, SuperPoint+SfM pipeline, mono depth networks 等）
- 即使有 unified models（VGGT, DUSt3R），也依赖 **complex bespoke architectures** 和 **multi-task joint optimization from scratch**，无法 leverage 大规模 pretrained models

DA3 提出两个核心问题：
1. 是否存在 **minimal set of prediction targets**，而不需要 multi-task learning？
2. **单个 plain transformer** 是否够用，而不需要 architectural specialization？

答案是 yes to both，这是 paper 的灵魂。

Project page: https://depth-anything-3.github.io

---

## 2. Depth-Ray Representation: 核心创新

这是 paper 最关键的 insight，理解它就能理解整个 paper。

### 2.1 传统相机投影模型

给定 N_v 张图，每张图有 depth D_i ∈ R^(H×W)，extrinsics [R_i | t_i]，intrinsics K_i。像素 p = (u, v, 1)^T 投影到 3D 点 P：

$$
\mathbf{P} = \mathbf{R}_i \big( \mathbf{D}_i(u,v) \mathbf{K}_i^{-1} \mathbf{p} \big) + \mathbf{t}_i
$$

变量解释：
- **R_i** ∈ SO(3): 3×3 rotation matrix（相机朝向）
- **t_i** ∈ R^3: translation（相机位置）
- **D_i(u,v)**: 像素 (u,v) 处的深度值（标量）
- **K_i**: 3×3 upper-triangular intrinsics matrix
- **p** = (u, v, 1)^T: pixel homogeneous coordinates
- **K_i^{-1} p**: 把 pixel 反投影到 camera ray direction（在相机坐标系）

直接预测 R_i 有困难：**orthogonality constraint** (R^T R = I) 难以满足。传统方法用 6D rotation representation (Zhou et al.) 或 quaternion，但仍是 global per-image 参数。

### 2.2 DA3 的 Depth-Ray Formulation

DA3 用 **per-pixel ray map** 隐式表示 camera pose。对每个像素 p，定义 ray：

$$
\mathbf{r} = (\mathbf{t}, \mathbf{d}) \in \mathbb{R}^6
$$

- **t** ∈ R^3: ray origin（= camera center）
- **d** ∈ R^3: ray direction = R K^{-1} p，即 pixel 反投影到 camera frame 后旋转到 world frame

Dense ray map M ∈ R^(H×W×6) 存储所有像素的 (t, d)。

**关键设计**：**不归一化 d**！保留 magnitude 以保留 projection scale。这样 3D 点就直接是：

$$
\mathbf{P} = \mathbf{t} + \mathbf{D}(u,v) \cdot \mathbf{d}
$$

Intuition：把 "距离" (depth, scalar) 和 "方向" (ray, 3D vector with scale) **解耦**。Depth 是标量，物理意义清晰、容易监督；ray 隐式编码了 camera pose 和 intrinsics。二者 element-wise 相乘加 t 就得 3D 点，**完全可微且几何一致**。

### 2.3 从 Ray Map 恢复 Camera Parameters

推理时需要从 dense ray map 恢复 (K, R, t)，paper 给了一个巧妙的 closed-form 方法。

**Step 1: 恢复 translation**
相机中心 t_c 通过平均所有 per-pixel ray origin：

$$
\mathbf{t}_c = \frac{1}{H \times W} \sum_{h=1}^{H} \sum_{w=1}^{W} \mathbf{M}(h, w, :3)
$$

变量解释：
- M(h, w, :3): ray map 前 3 个 channel，即 per-pixel ray origin
- 因为所有 ray 都从同一个相机中心发出，平均后噪声被 smoothing 掉

**Step 2: 恢复 (K, R) 通过 Homography**

定义一个 "identity camera" K_I = I。对这个 canonical camera，pixel p 的 ray direction 就是 d_I = K_I^{-1} p = p。

从 canonical ray d_I 到 target camera 下的 ray direction：

$$
\mathbf{d}_{cam} = \mathbf{K} \mathbf{R} \mathbf{d}_I
$$

所以 **H = K R** 是一个 homography，把 canonical rays 映射到 target rays。

优化问题（用 cross product 衡量共线性）：

$$
\mathbf{H}^* = \arg\min_{||\mathbf{H}||=1} \sum_{h=1}^{H} \sum_{w=1}^{W} ||\mathbf{H} \mathbf{p}_{h,w} \times \mathbf{M}(h, w, 3:)||
$$

变量解释：
- p_{h,w} = (h, w, 1)^T: pixel coordinates
- M(h, w, 3:): ray map 后 3 个 channel = target ray direction
- ||H p × M||: cross product 衡量两向量共线性（平行时为 0）
- ||H|| = 1: 避免平凡解 H = 0

这是 **standard least-squares 问题**，用 **DLT (Direct Linear Transform)** 算法求解（见 Hartley & Zisserman, Multiple View Geometry）。

**Step 3: RQ decomposition**

因为 K 是 upper-triangular，R 是 orthonormal，可以 uniquely 分解 H* = K R：

$$
\mathbf{H}^* = \mathbf{K} \mathbf{R} \xrightarrow{\text{RQ decomp}} (\mathbf{K}, \mathbf{R})
$$

这个推导很 elegant：通过预测 dense per-pixel ray，绕过了直接预测 rotation matrix 的 orthogonality 问题，同时 dense supervision 让 transformer 学起来更容易。

**Computational concern**：从 ray map 反推 camera parameters 在 inference 时有 cost。Paper 加了一个 lightweight camera head D_C（一个 transformer，每个 view 一个 token），直接预测 (f, q, t)，开销 ~0.1% of backbone。

---

## 3. Architecture: Minimal but Effective

三个组件：single transformer backbone + optional camera encoder + Dual-DPT head。

### 3.1 Single Plain Transformer Backbone

用 vanilla **DINOv2** ViT，L 个 blocks，**无架构修改**。

Cross-view reasoning 通过 **input-adaptive token rearrangement** 实现：
- 前 L_s 层：within-view self-attention（每张图独立）
- 后 L_g 层：交替 cross-view 和 within-view attention
- L_s : L_g = 2 : 1，L = L_s + L_g

实现上就是 tensor reordering：在 cross-view 层把 token 重排成 (H×W, N_v, C) 让不同 view 的同位置 token attend，在 within-view 层重排回 (N_v, H×W, C)。

**为什么 partial alternation 比 full alternation 好**（Table 7 ablation）：
- Full Alt. (所有层都交替): HiRoom Auc3 = 24.7
- Partial Alt. (2:1 比例): HiRoom Auc3 = 39.2

Intuition：前 2/3 层让 backbone 充分提取单图 features（继承 DINOv2 强大的 within-view 表示能力），后 1/3 才开始 cross-view reasoning。如果在所有层都 cross-view，会破坏单图 feature extraction 的基础。

**Input-adaptive**：单图输入时自然退化为 monocular depth，无额外 cost。

### 3.2 Camera Condition Injection

每个 view 前面 prepend 一个 camera token c_i：
- 有 pose 时：c_i = E_c(f_i, q_i, t_i)，MLP 编码 (FOV f_i ∈ R^2, rotation quaternion q_i ∈ R^4, translation t_i ∈ R^3)
- 无 pose 时：用 shared learnable token c_l

这些 camera tokens 与 patch tokens 拼接，参与所有 attention operations。这让模型可以无缝处理 posed 和 unposed 输入。

### 3.3 Dual-DPT Head

预测 dense depth 和 ray map，结构如下：
- **Shared reassembly modules**: 处理 backbone features
- **Two distinct fusion layers**: 一个给 depth branch，一个给 ray branch
- **Two separate output layers**: 分别输出 depth 和 ray

Intuition（为什么 share reassembly）：
- Depth 和 ray 本质相关（depth 是 ray 方向的标量投影）
- Share reassembly 鼓励 features 对齐，避免 redundant intermediate representations
- 只在 fusion 阶段分开，让两个 task 有 strong interaction

Ablation（Table 7, item d）：
- w/o Dual DPT (用两个独立 DPT head): HiRoom Auc3 = 5.59
- w/ Dual DPT: HiRoom Auc3 = 39.2

差异巨大，证明 shared representation 极其重要。

---

## 4. Training: Teacher-Student Paradigm

### 4.1 为什么需要 Teacher-Student

Real-world depth 数据质量差（Fig. 4 显示 LiDAR 稀疏、COLMAP 不完整、有噪声）。直接用这些 noisy GT 监督会限制 model 学到 fine geometry。

策略：
1. 在 synthetic data 上训练强大的 monocular teacher
2. Teacher 生成 dense high-quality pseudo-depth
3. 用 RANSAC scale-shift alignment 把 pseudo-depth 对齐到 noisy metric GT

### 4.2 Scale-Shift Alignment

给定 teacher 的 relative depth D̃ 和 sparse noisy depth D with validity mask m_p：

$$
(\hat{s}, \hat{t}) = \arg\min_{s > 0, t} \sum_{p \in \Omega} m_p \big( s \tilde{\mathbf{D}}_p + t - \mathbf{D}_p \big)^2
$$

$$
\mathbf{D}^{T \to M} = \hat{s} \tilde{\mathbf{D}} + \hat{t}
$$

变量解释：
- s > 0: scale factor（约束为正保持 depth 单调性）
- t: shift
- m_p: pixel p 是否有效
- Ω: 有效像素域
- Inlier threshold = mean absolute deviation from residual median（RANSAC 用）

对齐后 D^(T→M) 提供 scale-consistent 且 pose-depth coherent 的监督。

### 4.3 Training Objective

总 loss：

$$
\mathcal{L} = \mathcal{L}_D(\hat{\mathbf{D}}, \mathbf{D}) + \mathcal{L}_M(\hat{\mathbf{R}}, \mathbf{M}) + \mathcal{L}_P(\hat{\mathbf{D}} \odot \mathbf{d} + \mathbf{t}, \mathbf{P}) + \beta \mathcal{L}_C(\hat{\mathbf{c}}, \mathbf{v}) + \alpha \mathcal{L}_{grad}(\hat{\mathbf{D}}, \mathbf{D})
$$

各项含义：
- **L_D**: depth loss（带 confidence）
- **L_M**: ray map loss
- **L_P**: point cloud consistency loss（Đ ⊙ d + t = 重建 3D 点，⊙ 是 element-wise 乘）
- **L_C**: camera pose loss（β=1）
- **L_grad**: depth gradient loss（α=1）

Depth loss 细节（confidence-weighted）：

$$
\mathcal{L}_D(\hat{\mathbf{D}}, \mathbf{D}; D_c) = \frac{1}{Z_0} \sum_{p \in \Omega} m_p \big( D_{c,p} |\hat{\mathbf{D}}_p - \mathbf{D}_p| - \lambda_c \log D_{c,p} \big)
$$

变量解释：
- D_{c,p}: 像素 p 处 depth 的 confidence（模型预测）
- Z_0: 归一化常数
- 第一项: confidence-weighted L1（高 confidence 处 loss 权重大）
- 第二项: -log(D_c) 是 entropy-like regularizer，防止 confidence 全部趋向 0（类似 Kendall & Gal 的 aleatoric uncertainty）

Gradient loss：

$$
\mathcal{L}_{grad}(\hat{\mathbf{D}}, \mathbf{D}) = ||\nabla_x \hat{\mathbf{D}} - \nabla_x \mathbf{D}||_1 + ||\nabla_y \hat{\mathbf{D}} - \nabla_y \mathbf{D}||_1
$$

- ∇_x, ∇_y: 水平/垂直 finite difference operators
- 保留 sharp edges，确保 planar regions 平滑

**Scale normalization**：所有 GT 信号用 common scale factor 归一化：
scale = mean ℓ2 norm of valid reprojected point maps P

这确保 depth 和 ray map 之间 magnitude 一致，stabilize training。

### 4.4 Teacher Model 设计

基于 DA2 扩展，几个关键改进：

**Data scaling**：大量扩展 synthetic corpus，包括 Hypersim, TartanAir, IRS, vKITTI2, BlendedMVS, SPRING, MVSSynth, UnrealStereo4K, GTA-SfM, TauAgent, KenBurns, MatrixCity, EDEN, ReplicaGSO, UrbanSyn, PointOdyssey, Structured3D, Objaverse, Trellis, OmniObject。

**Depth representation**：
- DA2 预测 scale-shift-invariant **disparity**
- DA3 teacher 预测 scale-shift-invariant **depth**，更适合下游 metric depth 和 multi-view geometry
- 预测 **exponential depth** 而非 linear depth（增强近相机区域判别力，因为近处 depth 变化小）

**Distance-weighted surface normal loss**：
对每个中心像素采样 4 个邻居，计算 unnormalized normals n_i，权重：

$$
w_i = \sum_{j=0}^{4} ||\mathbf{n}_j|| - ||\mathbf{n}_i||
$$

变量解释：
- n_j: 5 个点（中心 + 4 邻居）的 unnormalized normal
- ||n_i|| 小的（靠近中心平面）权重高，downweight 远离中心的贡献

加权平均 normal：

$$
\mathbf{n}_m = \sum_{i=0}^{4} w_i \frac{\mathbf{n}_i}{||\mathbf{n}_i||}
$$

Normal loss：

$$
\mathcal{L}_N = \mathcal{E}(\hat{\mathbf{n}}_m, \mathbf{n}_m) + \sum_{i=0}^{4} \mathcal{E}(\hat{\mathbf{n}}_i, \mathbf{n}_i)
$$

- E: angular error between normals
- 第一项监督加权平均 normal，第二项监督 individual normals

总 teacher loss：

$$
\mathcal{L}_T = 0.5 \mathcal{L}_{grad} + \mathcal{L}_{gl} + \mathcal{L}_N + \mathcal{L}_{sky} + \mathcal{L}_{obj}
$$

- L_gl: global-local loss (ROE alignment, from MoGe)
- L_sky, L_obj: sky/object mask losses（处理 GT undefined regions）

---

## 5. Feed-Forward 3D Gaussian Splatting

DA3 作为 backbone + GS-DPT head 输出 pixel-aligned 3D Gaussians。

### 5.1 GS-DPT Head

输出每个像素的 3D Gaussian 参数：
- **σ_i**: opacity
- **q_i** ∈ H (quaternion): rotation
- **s_i** ∈ R^3: scale
- **c_i** ∈ R^3: RGB color

3D Gaussian 的 global position P_i 由预测 depth unproject 到 world coordinates 得到。

### 5.2 Pose-Adaptive 版本

两种模式统一：
- **有 pose 时**：scale + unproject 到 world space（用 [Umeyama 2002] 的相似变换）
- **无 pose 时**：直接用 DA3 预测的 pose unproject

设计 choices：
1. 3DGS 参数在 local camera space 预测（pose 不可知）
2. 额外预测 **depth offset** 减少 geometry-rendering trade-off
3. 用 **spherical harmonics** 替代 per-Gaussian color，建模 view-dependent surface

训练策略：
- Freeze DA3 backbone，只 tune GS-DPT head（避免 unstable training）
- Varying image resolutions + varying context views count（high-res + few views, low-res + many views）

Training objective:
- Photometric loss: L_MSE + L_LPIPS on rendered novel views
- Scale-shift-invariant depth loss L_D on observed views

---

## 6. Visual Geometry Benchmark

新 benchmark 覆盖 pose estimation, any-view geometry, visual rendering。

### 6.1 Datasets

5 个 geometry datasets：
- **HiRoom** (29 scenes, synthetic): Blender-rendered indoor, F1 threshold d=0.05m, voxel size 0.007m
- **ETH3D** (11 scenes, LiDAR): indoor + outdoor, d=0.25m, voxel 0.039m
- **DTU** (22 scenes, LiDAR): 49 views/object, CD metric (mm), background removed by RMBG 2.0
- **7Scenes** (7 scenes, LiDAR): low-res with motion blur, d=0.05m, voxel 0.007m
- **ScanNet++** (20 scenes, LiDAR): high-res indoor, d=0.05m, voxel 0.02m

NVS benchmark:
- DL3DV (140 scenes), Tanks and Temples (6), MegaDepth (19)

### 6.2 Metrics

**Pose metrics**: AUC based on RRA (Relative Rotation Accuracy) and RTA (Relative Translation Accuracy)。报告 Auc3 和 Auc30 (threshold in degrees)。

**Reconstruction metrics**:
- accuracy = dist(R → G): reconstructed 点到 GT 的距离
- completeness = dist(G → R): GT 点到 reconstruction 的距离
- CD (Chamfer Distance) = (accuracy + completeness) / 2
- Precision = (1/|R|) Σ [dist(R_i → G) < d]
- Recall = (1/|G|) Σ [dist(G_i → R) < d]
- F1 = 2 × precision × recall / (precision + recall)

**NVS metrics**: PSNR, SSIM, LPIPS

### 6.3 Reconstruction Pipeline

1. 用 feed-forward model 生成 consistent pose + depth
2. 用 evo + RANSAC 把预测 pose 对齐到 GT pose
3. 用最佳 transformation 通过 TSDF fusion 融合 aligned 点云
4. 比较 aligned reconstruction 与 GT point cloud

---

## 7. Experimental Results

### 7.1 Pose Accuracy (Table 2)

| Method | Params | HiRoom Auc3 | ETH3D Auc3 | DTU Auc3 | 7Scenes Auc3 | ScanNet++ Auc3 |
|--------|--------|-------------|------------|----------|--------------|----------------|
| DUSt3R | 0.57B | 17.6 | 4.30 | 4.00 | 6.90 | 8.10 |
| Fast3R | 0.65B | 25.9 | 8.10 | 9.50 | 19.0 | 17.9 |
| MapAnything | 0.56B | 17.9 | 19.2 | 6.50 | 12.6 | 20.2 |
| Pi3 | 0.96B | 67.0 | 35.2 | 62.5 | 25.5 | 50.7 |
| VGGT | 1.19B | 49.1 | 26.3 | 79.2 | 23.9 | 62.6 |
| **DA3-Giant** | 1.10B | **80.3** | **48.4** | **94.1** | **28.5** | **85.0** |
| DA3-Large | 0.36B | 58.7 | 32.2 | 70.2 | 29.2 | 60.2 |
| DA3-Base | 0.11B | 19.0 | 15.1 | 60.1 | 20.1 | 25.1 |

DA3-Giant 在 Auc3 上比 VGGT 平均提升 **35.7%**，ScanNet++ 上有 33% relative gain。

### 7.2 Reconstruction Accuracy (Table 3)

DA3-Giant 在所有 5 个 pose-free settings 上 SOTA，平均比 VGGT 提升 **25.1%**，比 Pi3 提升 **21.5%**。

值得注意的是 **DA3-Large (0.36B)** 比 VGGT (1.19B) 小 3×，但在 5/10 settings 上超越 VGGT，特别在 ETH3D 上表现强。

### 7.3 Monocular Depth (Table 4)

δ1 metric：
| Method | KITTI | NYU | SINTEL | ETH3D | DIODE |
|--------|-------|-----|--------|-------|-------|
| DA2 | 94.6 | 97.9 | 77.2 | 86.5 | 95.2 |
| VGGT | 91.7 | 97.9 | 67.9 | 97.5 | 95.3 |
| **DA3** | **95.3** | 97.4 | 75.5 | **98.6** | **95.4** |
| Teacher | 97.2 | 97.9 | 81.4 | 99.8 | 96.6 |

ETH3D 上从 86.5 (DA2) 提升到 98.6，巨大提升。

### 7.4 Feed-Forward NVS (Table 5)

DL3DV PSNR:
| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| pixelSplat | 16.55 | 0.456 | 0.480 |
| MVSplat | 18.13 | 0.559 | 0.393 |
| DepthSplat | 19.24 | 0.620 | 0.322 |
| Fast3R | 19.30 | 0.604 | 0.320 |
| MV-DUSt3R | 20.01 | 0.645 | 0.294 |
| VGGT | 20.96 | 0.697 | 0.253 |
| **DA3** | **21.33** | **0.711** | **0.241** |

关键发现：geometry-model-based frameworks **consistently outperform** specialized feed-forward models (pixelSplat, MVSplat, DepthSplat)。NVS 性能与 geometry estimation 能力正相关，DA3 是最强 backbone。

### 7.5 Sufficiency of Depth-Ray (Table 6)

ViT-L, 10 views, 120k steps:

| Method | HiRoom Auc3 | HiRoom F1 | ETH3D Auc3 | ETH3D F1 |
|--------|-------------|-----------|------------|----------|
| depth + pcd + cam | 9.1 | 12.8 | 19.0 | 60.4 |
| depth + cam | 10.8 | 16.5 | 9.9 | 48.0 |
| **depth + ray** | **48.7** | **60.3** | 25.5 | **65.4** |
| depth + ray + cam | 37.2 | 45.4 | 22.3 | 59.4 |

**depth + ray 几乎在 Auc3 上比 depth + cam 提升 100%**，证明 depth-ray 是 minimal sufficient target。加 auxiliary cam head 无进一步提升（depth + ray 已 sufficient）。

### 7.6 Architecture Ablation (Table 7)

| Method | HiRoom Auc3 | HiRoom F1 |
|--------|-------------|-----------|
| a. Proposed Arch | **39.2** | **47.0** |
| b. VGGT Style | 3.72 | 14.5 |
| c. Full Alt. | 24.7 | 29.3 |
| d. w/o Dual DPT | 5.59 | 11.5 |
| e. w/o Teacher | 11.2 | 16.0 |
| f. w/o Pose Cond.* | - | 65.8 |
| g. w/ Pose Cond.* | - | 73.8 |

VGGT Style（两个不同 transformer 堆叠）严重退化。Intuition：2/3 的 blocks 没预训练，无法 leverage DINOv2 的强大 features。

### 7.7 Efficiency (Table 8)

| Model | Max #Images | Backbone Params | Speed (FPS) |
|-------|-------------|-----------------|-------------|
| VGGT | 400-500 | 0.91B | 34.1 |
| DA3-Giant | 900-1000 | 1.130B | 37.6 |
| DA3-Large | 1500-1600 | 0.300B | 78.37 |
| DA3-Base | 2100-2200 | 0.086B | 126.5 |
| DA3-Small | 4000-4100 | 0.022B | 160.5 |

DA3-Giant 比 VGGT 处理 image 数量翻倍，速度更快。

### 7.8 Metric Depth (Table 11)

DA3-metric 在 ETH3D 上 δ1=0.917, AbsRel=0.104，大幅超越 second-best UniDepthv2 (δ1=0.863)。在 SUN-RGBD 上 AbsRel=0.105 是 SOTA。

---

## 8. Key Intuitions 总结

让我把核心 intuitions 再凝练一下：

### 8.1 为什么 Depth-Ray 比 Point Map 更好？

Point map 直接预测 (X, Y, Z) per pixel，但缺乏 "距离" 和 "方向" 的解耦。Depth + ray 让：
- Depth 是标量，物理清晰、监督容易
- Ray 隐式编码 pose + intrinsics，dense 预测让 transformer 学起来自然
- 二者 element-wise 组合就给 3D 点，几何一致且可微

### 8.2 为什么 Single Plain Transformer 够用？

- DINOv2 已经学到强大 visual features，直接继承
- Cross-view reasoning 只需 token rearranging，不需架构修改
- 关键：partial alternation (2:1) 让前 2/3 充分提取单图 features，后 1/3 做 cross-view
- 这让 model 能 inherit backbone 的 scaling properties

### 8.3 为什么 Teacher-Student 重要？

- Real-world depth 噪声大、稀疏、不完整
- Synthetic data GT 完美但分布窄
- Teacher 在 synthetic 上学 fine geometry
- Student 用 teacher pseudo-labels + scale-shift alignment，既有 real-world diversity 又有 fine detail

### 8.4 为什么 Exponential Depth 比 Linear Depth 好？

近相机区域 depth 变化小，linear depth 难以区分。Exponential depth 在近处放大变化（类似 disparity 的性质），远处压缩。这增强了近距离区域的判别力。

### 8.5 为什么 NVS 性能与 Geometry 正相关？

Geometry foundation model 提供一致的 3D structure prior，让 3DGS 的位置预测有强 inductive bias。Simple backbone + DPT head 就能超越 complex task-specific architectures (epipolar transformers, cost volumes)，因为大规模 pretraining 的 generalization 和 scalability 优势。

### 8.6 Pose Conditioning 的 Scaling Behavior

Table 7 (f vs g) 显示有 pose conditioning 时 model size 收益小。Intuition：**pose estimation 是更难的任务，需要更大 model 才能充分受益**。Depth estimation 在中等 model size 就接近饱和。这指导 future work：如果重点在 pose，需要 scale up model。

---

## 9. Web Links & References

- **DA3 Project Page**: https://depth-anything-3.github.io
- **DINOv2**: https://arxiv.org/abs/2304.07193
- **Depth Anything V2**: https://arxiv.org/abs/2406.09414
- **VGGT**: https://arxiv.org/abs/2503.11651
- **DUSt3R**: https://arxiv.org/abs/2312.14132
- **Pi3**: https://arxiv.org/abs/2507.13347
- **MASt3R**: https://arxiv.org/abs/2406.09414
- **Fast3R**: https://arxiv.org/abs/2501.13928
- **MapAnything**: https://arxiv.org/abs/2509.13414
- **MoGe**: https://arxiv.org/abs/2503.21717
- **Metric3D v2**: https://arxiv.org/abs/2404.15506
- **UniDepth v2**: https://arxiv.org/abs/2502.20110
- **DepthPro**: https://arxiv.org/abs/2410.02073
- **3D Gaussian Splatting**: https://arxiv.org/abs/2308.14737
- **pixelSplat**: https://arxiv.org/abs/2312.12381
- **MVSplat**: https://arxiv.org/abs/2403.14677
- **DepthSplat**: https://arxiv.org/abs/2412.00062
- **DL3DV-10K**: https://arxiv.org/abs/2406.03263
- **Hartley & Zisserman, Multiple View Geometry** (DLT, RQ decomposition): http://cvrs.whu.edu.cn/courses/MVG/A_SolutionManual.pdf
- **Umeyama 1991** (相似变换对齐): https://web.stanford.edu/class/cs273/refs/umeyama.pdf

---

## 10. 我的思考：对 Future Foundation Models 的启示

Andrej，从你的 micrograd / nanoGPT 哲学看，DA3 的核心 message 应该很 resonate：**minimal modeling + leverage pretrained features > bespoke architecture engineering**。

几个 takeaway 对 future foundation models 有启发：

1. **Representation choice > Architecture engineering**：找到正确的 minimal target representation（depth-ray 而非 point map）比设计复杂架构更重要。

2. **Pretrained features 是 free lunch**：DINOv2 已经学到大量 visual priors，不要浪费它们。Single plain transformer + token rearranging 就能 inherit 这些 features。

3. **Teacher-Student 解耦 data quality 和 data diversity**：当 real-world data 质量差，用 synthetic-trained teacher 提供 fine-grained supervision，scale-shift alignment 保留 geometric accuracy。

4. **Dense per-pixel prediction > Global parameters**：Camera pose 不用 global 9-DoF 参数，而用 per-pixel ray map。Dense supervision 更容易学，且 inference 时可 closed-form 恢复 global parameters。

5. **Geometry 是 NVS 的瓶颈**：Simple backbone + DPT head > complex task-specific NVS architectures。Future NVS work 应该 focus on 更好的 geometry backbone，而不是 NVS-specific 架构创新。

6. **Partial > Full cross-view attention**：让 backbone 在早期层保持 within-view，后期才 cross-view。这保护了 pretrained features 的 integrity。

希望这个讲解能 build 你的 intuition。如果对某个具体 component 想深入（比如 DLT 算法的具体实现，或者 RQ decomposition 的 numerical stability，或者 teacher model 的 global-local loss 细节），可以继续讨论。
