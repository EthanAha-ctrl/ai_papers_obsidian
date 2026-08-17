---
source_pdf: HunyuanWorld-Mirror Technical Report.pdf
paper_sha256: fdaa4e7bdfb96c32f0fd4d849ba4fd233e592e0435255788ea953ec6b00d5929
processed_at: '2026-08-05T08:28:14-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好嘞，我用白板聊天的 style 给你过一遍这篇 paper，尽量把 intuition 讲出来，而不是复述一遍 abstract。

---

## 这篇 paper 到底在干嘛

一句话：**之前那些 feed-forward 3D 重建模型（DUSt3R、VGGT、π³）都只吃 image，只吐一种东西。这篇说，我能吃 image + 任何 prior（pose / intrinsics / depth），同时吐 5 种东西（point cloud、depth、camera、normal、3DGS）。**

听起来好像就是 "加了 input、加了 output"，但里面的 design choice 有很多值得拆解的地方。

---

## Motivation 为什么这么设计

你想想 VGGT 这种 model 在干嘛——给它几张图，它猜一个 3D scene。但现实里你常常已经知道一些东西了：

- AR / robot 场景，相机的 intrinsics 是已知的（出厂就标定好了）
- SLAM 跑过的视频，pose 是知道的
- RGB-D 相机或 LIDAR，depth 是有 measurement 的

VGGT 这种 model 把这些 information 全 ignore，非要从 pixel 重新猜。这就好比人家给你 GPS 你不用，非要靠看星星认路。能走，但傻。

**所以核心问题就是：怎么把这些 prior 塞进 feed-forward pipeline，让 model "用得上"，而不是被当成 noise ignore 掉。**

同时 output 端，VGGT 能给 point map、depth、camera，但给不了 normal 和 novel view。Normal 对 surface reconstruction 很重要（Poisson 重建直接吃 normal），NVS 是 3DGS 的核心应用。这两块缺失让 VGGT 这种 model 在实际 3D pipeline 里不能直接 plug-in。

WorldMirror 就把这两件事一起补了。

---

## 架构上最有意思的几个 choice

### 1. 三种 prior 的 embedding 方式不一样，这是关键

**Camera pose**：6 自由度的东西，信息量极小。Paper 把 rotation 转成 quaternion（4 维），translation 归一化后 3 维，拼起来 7 维，过个 MLP 变成 **1 个 token**，concat 到 image token sequence 前面。

**Intrinsics**：focal length + principal point，4 个数，也是压成 **1 个 token**。

**Depth**：$H \times W$ 的 dense map，信息量大，压不成一个 token。Paper 用 conv 把它变成 $H_p \times W_p$ 个 token（与 image patch 对齐），然后 **直接加到 image token 上**。

这里有个 intuition 我觉得挺巧妙：

> **Compact global 信息用 token concatenation（attention 来处理），dense spatial 信息用 feature addition（直接 enrich 每个 patch）。**

为什么？因为 pose / intrinsics 是整个 view 共享的 global parameter，它与每个 pixel 都有关系，让 attention 自己去学哪个 pixel 需要这个信息。Depth 不一样，depth 就是 spatial 的，每个 patch 该知道自己的 depth，直接 add 上去就行，attention 不需要 cross 来 cross 去。

Table 7 的 ablation 印证了这个 intuition：

| 方式 | Params | Pose AUC@5 | Avg |
|------|--------|-----------|-----|
| Dense Plücker（pose） | 9.02M | 72.74 | 60.44 |
| Single Token（pose） | 1.06M | 74.55 | 61.06 |
| Dense Raymap（intrinsics） | 6.65M | 60.57 | 66.58 |
| Single Token（intrinsics） | 1.06M | 66.52 | 68.96 |

参数少了 9 倍，性能还更好。说明 compact global 信号强行 dense 化反而引入 noise，model 得去 "denoise" 这些 redundant embedding。

### 2. Depth 怎么加到 image token 上

公式 (1)：

$$T_i^{prompt} = [T_i^{cam}, T_i^{intr}, T_i^{img} + T_i^{depth}]$$

这里 $+$ 是 element-wise addition，$T_i^{img}, T_i^{depth} \in \mathbb{R}^{(H_p \times W_p) \times D}$。

Intuition：image token 编码的是 appearance，depth token 编码的是 geometry，两者 spatial aligned，直接 add 相当于在每个 patch 的 feature 里注入了 depth channel。ViT 后续的 self-attention 自然会 fuse 这两个 signal。

为什么不 concat？Concat 会让 sequence length 翻倍，attention 是 $O(L^2)$，计算开销大。Add 是 free 的。

### 3. Dynamic Prior Injection（训练时 50% 概率随机关掉某个 prior）

这其实是整个 paper 最 "实用" 的设计。训练时每个 prior 以 0.5 概率被 zero out，让 model 学会在任何 prior subset 下都能 work。

Intuition 上这就像 dropout，但 drop 的是整个 modality。好处：

1. Inference 时 prior 缺失不会崩
2. Model 学到的是 "prior 是辅助，image 是主信号" 的关系，不会 over-rely prior
3. 一个 model 就能 cover 所有 prior 组合，不需要部署多个 model

这个思路其实跟 diffusion model 里的 classifier-free guidance 很像——训练时随机 condition / uncondition，inference 时灵活组合。

### 4. Normal 监督用 hybrid supervision

Normal 的 GT 数据稀缺，paper 的做法是：有 normal label 的 dataset 直接用；没有的，从 GT depth 用 plane fitting 算 pseudo normal。

具体来说，对每个 pixel，取它 local neighborhood 的 3D 点（从 depth back-project 出来），fit 一个 plane，plane 的 normal 就是这个 pixel 的 pseudo normal。

这个 trick 让 model 能利用大量 depth dataset（比如 MegaDepth、Hypersim）间接学 normal，而不用局限于少数有 normal annotation 的 dataset（比如 iBims-1）。

Loss 用 Angle Loss：

$$\mathcal{L}_{normal} = \sum_{i=1}^N \alpha_l \cdot (1 - |\hat{N}_i \cdot N_i|)$$

- $\hat{N}_i$：predicted normal（unit vector）
- $N_i$：GT normal（unit vector）
- $\cdot$：dot product，等于 $\cos\theta$
- $|\cdot|$：取绝对值，处理 normal 的 ±1 flip ambiguity（normal 方向有二义性，朝里朝外都算对）
- $\alpha_l$：per-pixel weight

$1 - |\cos\theta|$ 是个很标准的 normal loss，惩罚的是角度差，对 direction 敏感对 magnitude 不敏感（因为都 unit vector）。

### 5. 3DGS 的设计：GS head 单独预测 position

这个 ablation（Table 8）值得讲一下。Paper 有个 design choice：3DGS 的 position（Gaussian center）不用 depth head 的 output，而是 GS head 自己再预测一个 Gaussian depth $\hat{D}_g$。

为什么？因为 depth head 优化的是 "几何 accuracy"，GS position 优化的是 "rendering quality"，这两个 objective 不完全一致。

比如一个 textureless 的墙面，depth head 应该给一个平面 depth，但 GS 为了 rendering 可能需要在这个 depth 附近撒一些略微 offset 的 Gaussian 来 capture view-dependent appearance。如果强行用 depth head 的 output 当 GS position，就会限制 GS 的 flexibility。

Ablation 数据：

| 配置 | RealEstate10K PSNR | DL3DV PSNR | VR-NeRF PSNR |
|------|-------------------|-----------|-------------|
| w/o GS DPT（用 depth head 的 depth） | 20.28 | 20.55 | 25.08 |
| Ours（GS head 独立预测） | 20.29 | 20.91 | 25.75 |

VR-NeRF 32 views 的差距最明显，0.67 dB。说明 view 多的时候 GS position 的独立性更重要。

### 6. Consistency Loss 治 "floating Gaussian"

3DGS 的常见 artifact 是 Gaussian 漂在空中（floating artifact），原因是 multi-view rendering 的 ambiguity 和 GT depth noise。

Paper 加了一个 gradient consistency loss：

$$\mathcal{L}_{consis} = \sum_{i=1}^N \|\nabla\hat{D}_i[\hat{M}_i] - \nabla\tilde{D}_i[\hat{M}_i]\|$$

- $\hat{D}_i$：depth head 预测的 depth（当作 pseudo GT）
- $\tilde{D}_i$：3DGS 渲染出来的 depth
- $\nabla$：spatial gradient
- $\hat{M}_i$：confidence mask，取 confidence top 30%

Intuition：不强求 GS rendering depth 与 depth head 的 depth 数值一致（那会过度约束），但强求 gradient 一致。Gradient 一致意味着 GS 的 surface 形状与 depth head 一致，但允许整体 offset。这样 floating Gaussian（gradient 不连续）会被惩罚。

这个 loss 权重 $\lambda_{consis}=0.1$，不大，说明只是个 regularizer 而不是主要 supervision。

---

## 训练策略的 intuition

### Curriculum Learning 的三个维度

**Task sequencing**：
1. 先训 multi-modal prior prompting + VGGT 的原有 task（point/depth/camera）
2. 加 normal task
3. Freeze backbone，只训 3DGS head

为什么这么排？因为 3DGS 依赖前面所有东西的 quality。如果一开始就 joint train 3DGS，backbone 还没学好 geometry，GS 就会在垃圾 geometry 上学 appearance，学到一堆 wrong pattern。先 freeze backbone 再训 GS head，GS 学的是 "在已经不错的 geometry 上怎么 refine appearance"。

**Data scheduling**：
1. 前期 real + synthetic 混着喂，model 学 generalization
2. 后期只喂高质量 synthetic，fine-tune 细节

Real data 有 noise（COLMAP 重建的 pose / depth 有 error），synthetic data 干净但 distribution narrow。先混着喂让 model 见世面，再洗干净喂让 model 学精度。这跟小孩学习一个道理——先广泛接触，再精读经典。

**Progressive resolution**：
低分辨率 warm-up，高分辨率 fine-tune。低分辨率收敛快（sequence length 短，attention 计算少），高分辨率捕捉 detail。

---

## 实验结果里最值得注意的几个点

### 1. Prior 加进来不仅帮对应 task，还帮其他 task（Figure 6）

这个 synergy 是 paper 最 strong 的 claim。比如给 depth prior，不仅 depth accuracy 提升，point map、camera、focal length accuracy 全部提升。

Intuition：这些 geometric quantity 是 coupled 的。Depth 约束了几何，几何约束了 camera 位置，camera 位置约束了 focal length。Model 学的是统一的 3D representation，而不是 isolated 的 task head。给一个 prior 等于给 representation 加了一个 anchor，整个 representation 都 sharpen 了。

### 2. Normal 超过 StableNormal（Table 4）

ScanNet mean angular error：WorldMirror 13.8° vs StableNormal 16.0°。

这有点 surprising，因为 StableNormal 是专门做 normal 的，用 diffusion prior。WorldMirror 是 multi-task model，normal 只是其中一个 head。

为什么能超？我猜测是 multi-view context 提供了 single-image 方法看不到的信息。Single-image 估计 normal 在 ambiguous 区域（比如 textureless 墙面）很 rely prior，但 multi-view 能从不同角度看到同一个 surface，直接 resolve ambiguity。

### 3. NVS 超过 AnySplat 3 dB（Table 5）

RealEstate10K 2 views：WorldMirror 20.62 dB vs AnySplat 17.62 dB。3 dB 是 roughly 2 倍的 MSE 改善，很大。

为什么？AnySplat 也是 feed-forward 3DGS，也是 pose-free。差异可能在：
1. WorldMirror 的 geometry representation 更 robust（multi-task joint training 让 geometry 更准）
2. Depth prior 注入让 GS position 初始化更好
3. Consistency loss 让 GS 分布更干净

### 4. Post-optimization 加速（Table 6）

用 WorldMirror 的 predicted point cloud 作为 3DGS optimization 的 initialization，1000 iterations 就能达到 27.79 dB，比 AnySplat 3000 iterations 的 26.03 dB 还高。

这个结果挺 impactful。说明 feed-forward prediction + short optimization 是比纯 optimization 快很多的路线，而且 quality 还更好。实际部署里这个 trade-off 很划算。

---

## Limitation 与未来方向

Paper 自己承认两个 limitation：

1. **Dynamic scene 与 autonomous driving 表现次优**。Sintel 上略低于 best method，KITTI 上 depth 不如 π³。原因是 training data 里这两类 under-represented。这个 limitation 是 data 问题，不是 architecture 问题，扩 data 能解决。

2. **Resolution 限制 300-700 pixels，无法处理 thousands of views**。这是 memory 问题，ViT 的 attention 是 $O(N^2)$，view 数多了显存爆。

我觉得还有几个没提但很关键的 limitation：

3. **Prior 的 quality 假设**。Paper 假设 prior 是 relatively clean 的。如果给的 depth prior 很 noisy（比如 monocular depth estimation 的 output），model 会不会被带偏？这个 paper 没讨论 robustness to noisy prior。

4. **Prior 之间的 consistency**。如果给的 pose 与 depth 不 consistent（pose 说相机在这里，depth back-project 出来的点与 pose 不 match），model 怎么处理？这种 conflict 在实际场景里很常见。

5. **4D / dynamic 重建**。Paper 处理的是 static scene，video 输入也是当 multi-view 用。真正 dynamic scene（有运动物体）的 4D 重建还需要 extension。

未来方向我想到几个：

- **Long sequence**：chunked attention 或 state-space model（Mamba 这种）来处理 1000+ views
- **Noisy prior robustness**：训练时给 prior 加 noise，让 model 学会 discount unreliable prior
- **4D extension**：把 Gaussian 升级成 4D Gaussian（deformable），处理 dynamic scene
- **Generative prior integration**：用 video diffusion model 生成 multi-view video，WorldMirror 重建，两者闭环
- **IMU / sensor fusion**：除了 image prior，把 IMU、GPS 这种 temporal signal 也 fuse 进来

---

## 我读完的整体感受

这篇 paper 工程味很重，没太多 theory 创新，但 design choice 都很 solid，每个 decision 都有 ablation 撑着。它本质上是在 VGGT 这个 backbone 上做 "input/output 扩展"，但扩展的方式很 thoughtful，不是简单堆 task。

最值得借鉴的两个 idea：

1. **Compact vs dense prior 的 differential embedding**——这个 intuition 可以推广到其他 multi-modal fusion 场景。Global 信号用 token，spatial 信号用 addition，这可能是 generalizable 的 design pattern。

2. **Dynamic prior injection**——训练时随机 drop modality，让 model 学会 any-subset inference。这个思路在 multi-modal model 里应该会越来越流行，类似 classifier-free guidance 之于 diffusion。

从 big picture 看，这篇 paper 是 3D vision foundation model 向 "universal input + universal output" 演进的一步，类似 LLM 在 NLP 里走过的路。下一步应该是跟 generative world model 结合——WorldMirror 这种 model 提供几何 structure prior，generative model 提供 appearance / dynamics，两者闭环就是完整的 3D world generation + reconstruction system。

---

**参考资源**

- VGGT 论文与代码：https://arxiv.org/abs/2503.11651 | https://github.com/facebookresearch/vggt
- π³: https://arxiv.org/abs/2507.13347
- DUSt3R: https://arxiv.org/abs/2312.14132 | https://github.com/naver/dust3r
- MASt3R: https://arxiv.org/abs/2406.09656
- Fast3R: https://arxiv.org/abs/2411.15381
- Pow3R: https://arxiv.org/abs/2409.07543
- AnySplat: https://arxiv.org/abs/2505.23716
- FLARE: https://arxiv.org/abs/2502.15757
- Dens3R: https://arxiv.org/abs/2507.16290
- StableNormal: https://arxiv.org/abs/2409.16991
- GeoWizard: https://arxiv.org/abs/2403.12013
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14737
- gsplat: https://arxiv.org/abs/2409.11342 | https://github.com/nerfstudio-project/gsplat
- DPT: https://arxiv.org/abs/2103.13413
- UniDepth: https://arxiv.org/abs/2403.18913
- NoPoSplat: https://arxiv.org/abs/2410.24207
- MVS-Splat: https://arxiv.org/abs/2405.12218
- DepthSplat: https://arxiv.org/abs/2406.07597
- CUT3R: https://arxiv.org/abs/2412.04657
- MonST3R: https://arxiv.org/abs/2410.03825
- Splatt3R: https://arxiv.org/abs/2408.13912
- Depth Anything: https://arxiv.org/abs/2401.10891
- Classifier-Free Guidance（diffusion 里的类似思路）: https://arxiv.org/abs/2207.12598
- Plücker coordinates（对比 dense pose embedding）: https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates
- HunyuanWorld 1.0: https://hunyuan.tencent.com/modelSquare/home/playground?modelId=2025

如果你想在某个点上深挖（比如 3DGS 的 differentiable rasterization 细节、VGGT backbone 的 alternated attention 机制、或者 consistency loss 为什么用 gradient 而不是 value），告诉我，我可以再展开。

---

# HunyuanWorld-Mirror 技术报告深度解析

## 一、整体定位与核心 motivation

这份 report 来自 Tencent Hunyuan 团队，是 HunyuanWorld 1.1 的核心组件。HunyuanWorld 1.0 只能从 text 或 single-view image 生成 3D world，无法处理 video 或 multi-view input。WorldMirror 的出现填补了这个 gap，让 video-to-3D 与 multi-view-to-3D 成为可能。

从 high-level 看，这篇 paper 要解决的是 feed-forward 3D reconstruction 领域两个长期被忽视的 limitation：

1. **Input space 受限**：现有方法（DUSt3R、MASt3R、Fast3R、VGGT、π³）几乎只接受 raw images 作为输入，无法利用真实场景中常常可获取的 prior modalities，比如 calibrated intrinsics、camera pose、来自 LIDAR/RGB-D 的 depth measurement。这些 prior 在特定场景下价值极高：intrinsics 解决 scale ambiguity，camera pose 保证 multi-view consistency，depth 在 textureless / reflective region 提供关键 constraint。

2. **Output space 割裂**：现有方法大多 task-specific，depth estimation、point map regression、pose prediction、point tracking 各做各的。VGGT 尝试 unify，但 surface normal estimation 与 novel view synthesis 仍然缺失。

WorldMirror 的核心 thesis 是：**一个统一架构，既能接受任意 subset 的 prior modality，又能输出全谱系 geometric representation**（point clouds、multi-view depth、camera parameters、surface normals、3D Gaussians）。

- Paper link: https://arxiv.org/abs/2510.10726
- Project: https://3d-models.hunyuan.tencent.com/world/
- Code: https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror
- HuggingFace: https://huggingface.co/tencent/HunyuanWorld-Mirror

---

## 二、架构解析：从 token 到 multi-task output

整个 pipeline 是 fully transformer-based 的，结构上沿用 VGGT 的 backbone，但 input 与 output 两端都做了扩展。Figure 2 给出了 overview。

### 2.1 Multi-Modal Prior Prompting

这是 paper 最核心的设计。三种 prior modality 各自的 embedding 策略不同，根据信息密度区分对待。

**Camera Pose 编码**

给定 $N$ 个 input image 的 camera pose $\{[R_i | t_i]\}_{i=1}^N$，其中：
- $R_i \in \mathbb{R}^{3\times 3}$ 是 rotation matrix
- $t_i \in \mathbb{R}^3$ 是 translation vector

第一步是**scene scale normalization**：

$$t_i^{norm} = (t_i - c) / \alpha$$

变量含义：
- $c$：所有 camera 的中心点（centroid）
- $\alpha$：每个 camera 到 $c$ 的最大距离

这一步确保数值 range 与 scene scale 无关，模型看到的是 unit cube 内的几何关系。这对训练 stability 极重要，因为不同 dataset 的 absolute scale 差异巨大（Hypersim 室内 vs TartanAir 大场景）。

接着把 rotation matrix 转为 quaternion $q_i \in \mathbb{R}^4$（4 维，去除冗余自由度），与 normalized translation $t_i^{norm} \in \mathbb{R}^3$ 拼成 7 维 vector，通过 two-layer MLP 投影到 single token $T_i^{cam} \in \mathbb{R}^{1\times D}$，$D$ 是 image token 的 channel dimension。

**为什么用 quaternion 而不是 rotation matrix？** Quaternion 是 continuous、compact 的 representation，避免 rotation matrix 的 6 参数冗余（SO(3) 只有 3 个自由度），同时不会像 Euler angles 那样出现 gimbal lock。从 optimization 角度，quaternion 也更适合神经网络回归。

**Calibrated Intrinsics 编码**

Intrinsic matrix $K_i \in \mathbb{R}^{3\times 3}$ 提取 focal length 与 principal point $(f_x, f_y, c_x, c_y)$，分别除以 image width $W$ 与 height $H$ 做归一化。归一化的 motivation 是让模型对 resolution 不敏感——训练时 dynamic resolution（100k-250k pixels），如果用 raw pixel value 会导致同一相机的 focal 在不同 resolution 下数值不同。

投影方式与 camera pose 相同：two-layer MLP 得到 $T_i^{intr} \in \mathbb{R}^{1\times D}$。

**Depth Map 编码**

Depth 是 dense spatial signal，信息密度远高于 pose 与 intrinsics，因此 embedding 策略完全不同。

给定 depth map $D_i \in \mathbb{R}^{H\times W}$，先归一化到 $[0, 1]$，再用 kernel size 与 patch size 对齐的 conv layer 生成 dense tokens：

$$T_i^{depth} \in \mathbb{R}^{(H_p \times W_p) \times D}$$

其中 $H_p = H / \text{patch\_size}$，$W_p = W / \text{patch\_size}$。这些 token 与 image token spatial alignment 完全对应，所以采用**直接相加**而非 concatenation：

$$T_i^{img} + T_i^{depth}$$

这是一个关键设计选择。Addition 的好处：
1. 不增加 sequence length，计算开销不变
2. Spatial alignment 天然保留，depth 直接 enrich 每个 patch 的 geometric 含义
3. 与 image feature 的 cross-attention 在 ViT 内部自然完成 fusion

**Versatile Prior Prompting 总公式**

$$T_i^{prompt} = [T_i^{cam}, T_i^{intr}, T_i^{img} + T_i^{depth}]$$

最终 $T_i^{prompt} \in \mathbb{R}^{(1 + 1 + H_p \times W_p) \times D}$。可以看到，pose 与 intrinsics 作为 prefix token（类似 CLS token）concat 在前面，depth 则 add 到 image token 上。这种**混合 embedding** 策略基于一个 insight：compact global 信息适合 token-level conditioning（用 attention 处理），dense spatial 信息适合 feature-level conditioning（用 addition 融合）。

Table 7 的 ablation 印证了这一点。对 camera pose，对比 dense Plücker ray embedding（9.02M params）vs single token（1.06M params），single token 在 Pose AUC@5 上从 72.74 提升到 74.55，且参数少近 9 倍。对 intrinsics，dense raymap 6.65M params vs single token 1.06M params，single token 平均分 68.96 vs dense 66.58。**Compact global representation 在这里更优**，这与 Plücker coordinates 的设计哲学（per-pixel encoding相机光线方向）相反，但合理：pose 与 intrinsics 本身是 global 的、跨整个 view 共享的参数，强行 dense 化反而引入 noise。

### 2.2 Universal Geometric Prediction

Output 端统一了 5 个 task：point map、camera、depth、surface normal、3DGS。

**Point Map、Camera、Depth**

沿用 VGGT 设计，从 backbone 输出 token $T_i^{out} \in \mathbb{R}^{L\times D}$ 出发：

$$\hat{P}_i = \text{DPT}_p(\hat{T}_i^{img}), \quad \hat{D}_i = \text{DPT}_d(\hat{T}_i^{img}), \quad \hat{E}_i = \text{Transformer}(\hat{T}_i^{cam})$$

DPT（Dense Prediction Transformer）是 Ranftl et al. 提出的 dense prediction head，通过 multi-scale feature aggregation 重构 spatial resolution。Point map $\hat{P}_i \in \mathbb{R}^{H\times W\times 3}$ 是每个 pixel 的 3D 坐标，depth $\hat{D}_i \in \mathbb{R}^{H\times W}$ 是 scalar depth，camera $\hat{E}_i$ 从 camera token 经 transformer 层回归。

**Surface Normal**

$$\hat{N}_i = \text{DPT}_n(\hat{T}_i^{img}) / \|\text{DPT}_n(\hat{T}_i^{img})\|_2$$

L2 normalization 保证输出是 unit vector。这里有个细节值得注意：normal 是 per-pixel 的 unit vector，但 ground truth normal 数据稀缺。Paper 提出一种 **hybrid supervision**：对有 annotation 的 dataset 直接监督；对没有 normal label 的 dataset（如 depth dataset），用 plane fitting 从 ground truth depth 推 pseudo normal。具体做法是对每个 pixel 的局部邻域拟合平面，平面的法向量即为 pseudo normal。这种做法让 model 能利用大量 depth dataset 间接学 normal。

Normal loss 是 Angle Loss：

$$\mathcal{L}_{normal} = \sum_{i=1}^N \alpha_l \cdot (1 - |\hat{N}_i \cdot N_i|)$$

其中 $\cdot$ 是 dot product，$|\cdot|$ 取绝对值处理 normal 的二义性（normal 方向有 ±1 的 flip ambiguity）。$\alpha_l$ 是 per-pixel weight，可能用于 down-weight 边缘或低 confidence 区域。$1 - |\hat{N}_i \cdot N_i|$ 等价于 $1 - \cos\theta$，其中 $\theta$ 是预测 normal 与 GT normal 的夹角。

**3D Gaussians for Novel View Synthesis**

这是 paper 在 VGGT 之外扩展的最重要 task。设计上分两部分：

1. **Gaussian center**：DPT head $\text{DPT}_g$ 回归 per-pixel Gaussian depth $\hat{D}_g$ 与 feature map $F_g$。$\hat{D}_g$ 通过 GT camera pose $[R|t]$ 与 intrinsic $K$ back-project 得到 Gaussian center $\mu_g$。

2. **Gaussian attributes**：剩下的属性（opacity $\sigma_g$、orientation $r_g$（quaternion）、scale $s_g$、residual spherical-harmonic color coefficients $\Delta c_g$、fusion weight $w_g$）通过 conv network 从 $F_g$ 与 appearance feature 联合预测：

$$\hat{G} = \text{Conv}(F_g, I), \quad \hat{D}_g, F_g = \text{DPT}_g(\hat{T}^{img})$$

这里有个**重要的 design choice**：Gaussian 的位置不直接用 depth head 的输出，而是 GS head 单独预测一个 Gaussian depth $\hat{D}_g$。Table 8 的 ablation 验证了这点——"w/o GS DPT"（用 depth head 的 depth 替代 GS head 的 depth）在 VR-NeRF 32 views 上 PSNR 从 25.75 降到 25.08，说明 Gaussian 位置需要 task-specific prediction，因为它优化的是 rendering quality 而非几何 accuracy。

为了减少 overlapping region 的 Gaussian redundancy，paper 借鉴 AnySplat 的 voxelization + pruning 策略，对 per-pixel Gaussian 做 cluster 与 prune。

**Dual supervision**：训练时 input image 分为 context set 与 target set。3DGS 只从 context view 构建，但通过 differentiable rasterizer 同时 render 到 context view 与 target view，loss 共同监督。这种设计让 model 既能 fit input observation，又能 generalize到 novel view。

**Loss 函数总览**

$$\mathcal{L} = \lambda_{points}\mathcal{L}_{points} + \lambda_{depth}\mathcal{L}_{depth} + \lambda_{cam}\mathcal{L}_{cam} + \lambda_{normal}\mathcal{L}_{normal} + \lambda_{3dgs}\mathcal{L}_{3dgs}$$

权重设置：$\lambda_{points}=1.0, \lambda_{depth}=1.0, \lambda_{cam}=5.0, \lambda_{normal}=1.0, \lambda_{3dgs}=1.0$。Camera loss 权重最高（5.0），原因是 camera 参数误差会 propagate 到所有下游 task（point map back-projection、Gaussian position），需要更强约束。

**Point loss**（带 uncertainty）：

$$\mathcal{L}_{point} = \sum_{i=1}^N \|\Sigma_i^P \odot (\hat{P}_i - P_i)\| + \|\Sigma_i^P \odot (\nabla\hat{P}_i - \nabla P_i)\| - \alpha\log\Sigma_i^P$$

变量：
- $\Sigma_i^P$：per-pixel point uncertainty（model 预测的 confidence）
- $\odot$：channel-broadcast element-wise product
- $\nabla$：spatial gradient operator
- $\alpha$：regularization 系数

三项含义：
1. 第一项：confidence-weighted L1 reconstruction error
2. 第二项：confidence-weighted gradient error，强化 local smoothness / detail preservation
3. 第三项：$-\alpha\log\Sigma_i^P$ 是 entropy regularization，防止 model 把所有 $\Sigma$ 推到 0 来 trivially minimize 前两项（如果 $\Sigma \to 0$，前两项 vanish 但 $-\log\Sigma \to +\infty$）

**Camera loss**（Huber loss）：

$$\mathcal{L}_{cam} = \sum_{i=1}^N \|E_i - \hat{E}_i\|_\epsilon$$

Huber loss $\|\cdot\|_\epsilon$ 在小误差区域是 L2，大误差区域是 L1，对 outlier 更鲁棒。

**3DGS loss**：

$$\mathcal{L}_{3dgs} = \mathcal{L}_{rgb} + \lambda_{gsdepth}\mathcal{L}_{gsdepth} + \lambda_{consis}\mathcal{L}_{consis}$$

其中 $\mathcal{L}_{rgb}$ 是 rendering loss：

$$\mathcal{L}_{rgb} = \sum_{i=1}^N \|I_i[M_i] - \hat{I}_i[M_i]\| + \lambda_{lpips}\text{LPIPS}(I_i[M_i], \hat{I}_i[M_i])$$

- $M_i$：visibility mask，标记当前 view 哪些 pixel 在 context view 中可见
- LPIPS：perceptual loss，权重 $\lambda_{lpips}=0.05$

$\mathcal{L}_{gsdepth}$ 监督 Gaussian 渲染深度与 GT depth 一致，公式同 point loss。

$\mathcal{L}_{consis}$ 是 gradient consistency loss：

$$\mathcal{L}_{consis} = \sum_{i=1}^N \|\nabla\hat{D}_i[\hat{M}_i] - \nabla\tilde{D}_i[\hat{M}_i]\|$$

- $\hat{D}_i$：depth head 预测的 pseudo depth
- $\tilde{D}_i$：3DGS 渲染得到的 depth
- $\hat{M}_i$：depth confidence mask，取 confidence map top 30% quantile

这个 loss 的 motivation 是解决 **floating artifact 问题**：multi-view rendering ambiguity 与 GT depth noise 会导致 Gaussian 漂浮在空中。通过强制 GS rendering depth 的 gradient 与 depth head 的 gradient 一致，可以隐式约束 Gaussian 在空间中的分布更连贯。

---

## 三、训练策略：Dynamic Prior Injection + Curriculum Learning

### 3.1 Dynamic Prior Injection Scheme

训练时每个 prior modality 以 0.5 概率随机 toggle，disabled 时对应 token 设为 zero。这个看似简单的策略有几个深层好处：

1. **Robustness**：强制 model 学会在 missing information 下做 inference，类似 dropout
2. **Graceful degradation**：inference 时 prior 缺失不会导致 catastrophic failure
3. **Single unified model**：一个模型就能处理任意 prior 组合，不需要为每种组合单独训练

这是典型的**multi-conditioning training** 思路，与 classifier-free guidance 在 diffusion model 中的随机 conditional/unconditional 切换有相通之处。

### 3.2 Curriculum Learning Strategy

三个维度的 curriculum：

**Task sequencing**

1. 阶段一：从 pretrained VGGT weights 初始化，joint training multi-modal prior prompting module 与其他参数，建立 prior-aware prediction 的基础能力
2. 阶段二：加入 normal prediction task，joint training
3. 阶段三：freeze 所有参数，只训练 3DGS head

这种 sequencing 的 intuition：先建立 multi-view geometry 的核心能力（point/depth/camera），再扩展到 normal（依赖 geometry 的 local 属性），最后训练 3DGS（依赖 geometry + appearance 的综合表达）。3DGS head 单独训练避免影响 backbone 已学好的几何 representation。

**Data scheduling**

- 初期：real + synthetic 混合数据，提升 generalization，防止 overfitting
- 后期：只用高质量 synthetic 数据 fine-tune，mitigate 真实数据 annotation noise

**Progressive resolution**

低分辨率 warm-up（快速 stable convergence）→ 高分辨率（捕捉 fine detail）。这与 ViT 训练的常用策略一致。

### 3.3 Implementation Details

- 两阶段训练：100 epochs（with normal head）+ 50 epochs（fine-tune with Gaussian head）
- Dynamic resolution：total pixel count 100k-250k，aspect ratio 0.5-2.0
- 32 H20 GPUs，24 images per GPU
- Parameter-specific learning rate：
  - Patch embedding：2e-5（pretrained，小 lr）
  - Alternated attention + pretrained heads：1e-4
  - 新增参数：2e-4
- CosineAnnealing scheduler
- 15 个 training dataset：DL3DV、BlenderMVS、TartanAir、ASE、Unreal4K、Habitat、MapFree、MVS-Synth、ArkitScenes、ScanNet++、MegaDepth、Hypersim、Matterport3D、Co3dv2、WildRGBD

这个 dataset 组合覆盖 indoor/outdoor、real/synthetic、static/dynamic，是 generalization 的基础。

---

## 四、实验结果解读

### 4.1 Point Map Reconstruction（Table 1）

在 7-Scenes、NRGBD、DTU 三个 dataset 上评估 Acc 与 Comp。

基线对比：
- VGGT：7-Scenes mean Acc 0.046，DTU mean Acc 1.338
- π³：7-Scenes mean Acc 0.048，DTU mean Acc 1.198
- WorldMirror（no prior）：7-Scenes mean Acc 0.043，DTU mean Acc 1.017

无 prior 就已经超过 VGGT 与 π³。7-Scenes 提升 10.4%（相对 VGGT），DTU 提升 17.8%（相对 π³）。

加入 prior 后的增益：
- 7-Scenes：全 prior 配置 mean Acc 从 0.043 → 0.018，相对 no-prior baseline 提升 58.1%
- NRGBD：从 0.041 → 0.016，提升 53.1%
- DTU：从 1.017 → 0.735，提升 27.7%

可以看到 prior 的增益在不同 dataset 上有差异。7-Scenes 与 NRGBD 是 indoor scene，结构复杂但 scale 有限，depth prior 帮助大；DTU 是 object-level，camera pose 已经能 capture 大部分几何关系，增益相对小。

### 4.2 Camera Pose Estimation（Table 2）

三个 zero-shot dataset：RealEstate10K（static, mixed）、Sintel（outdoor, dynamic）、TUM-dynamics（indoor, dynamic）。

RealEstate10K：AUC@30 达 86.28，超过 π³ 的 85.90 与 VGGT 的 77.62。TUM-dynamics：ATE 0.010，RPE rot 0.297，是所有方法中最优。

Sintel 略低于最优，paper 解释为 training data 中 outdoor dynamic scene 不足。这暴露了一个 limitation：dynamic scene 的 generalization 仍依赖 data distribution。

### 4.3 Depth Estimation（Table 3）

NYUv2、Sintel、KITTI 上评估 monocular 与 video depth。

- NYUv2 monocular：Abs Rel 0.052，δ<1.25 达 0.957，与 π³（0.054）相当
- KITTI video：Abs Rel 0.063，略逊于 π³ 的 0.038，原因同样是 training data 中 urban driving scene 不足
- Sintel video：Abs Rel 0.289，δ<1.25 0.668，优于多数方法

值得注意的是，WorldMirror 并未针对 monocular metric depth 专门优化，仍能达到与专用方法可比的性能，说明 multi-task joint training 让 model 学到了更通用的 depth representation。

### 4.4 Surface Normal Estimation（Table 4）

ScanNet、NYUv2、iBims-1 三个 dataset，对比 OASIS、Omnidata v1/v2、DSine、GeoWizard、StableNormal。

ScanNet：mean angular error 13.8°（vs StableNormal 16.0°、DSine 16.2°），22.5° threshold 内 pixel 比例 82.5%（vs StableNormal 81.5%）。

NYUv2：mean 15.1°，22.5° threshold 80.1%，全面超过 StableNormal 与 DSine。

这是 paper 的一个亮点。Surface normal 通常是 single-image task，专门方法（StableNormal 用 diffusion prior）很强。WorldMirror 通过 multi-view context + hybrid supervision 超越它们，说明 multi-view geometry 信息能显著帮助 normal estimation，特别是在 textureless 或 ambiguous region。

### 4.5 Novel View Synthesis（Table 5, 6）

Feed-forward setting（Table 5）：
- RealEstate10K 2 views：PSNR 20.62（vs AnySplat 17.62），提升 3 dB
- DL3DV 8 views：PSNR 20.92（vs AnySplat 18.31）
- RealEstate10K 32 views：PSNR 25.14（vs AnySplat 19.96）
- DL3DV 64 views：PSNR 21.25（vs AnySplat 18.40）

加入 intrinsics 进一步提升 2 views PSNR 到 22.03，加入 intrinsics + camera pose 提升到 22.30。这印证了 intrinsics 解决 scale ambiguity 对 NVS 帮助最大。

Post-optimization setting（Table 6）：
- WorldMirror feed-forward：RealEstate10K 32 views PSNR 25.14，<2s
- 用 predicted point cloud 初始化 + 1000 iterations 优化：PSNR 27.79，23s
- AnySplat + 3000 iterations：PSNR 26.03，56s

WorldMirror 的 predicted point cloud 作为 initialization，1000 iterations 就能达到 27.79 PSNR，比 AnySplat 3000 iterations 的 26.03 更高且更快。这说明 feed-forward prediction 提供了优质的几何先验，极大加速 3DGS optimization。

### 4.6 Prior Guidance Benchmark（Section 3.3, Figure 5, 6）

一个非常有意思的发现：**加入任何一种 prior 不仅提升对应 task，还提升其他 task**。

- Camera pose 帮助 capture global geometry，间接提升 point map 与 depth
- Intrinsics 解决 scale ambiguity，提升所有依赖 metric scale 的 task
- Depth 提供 pixel-level constraint，特别在 geometrically complex region 帮助 point map 与 normal

Figure 6 的 bar chart 展示了这种 synergy：depth prior 加进来，focal accuracy、pose accuracy、point accuracy 全部提升。这暗示 model 学到的是**统一的 3D scene representation**，而非 task-specific 的 isolated mapping。

---

## 五、Applications 与 Visualization

**Surface Reconstruction**：用 predicted normal 替代从 point cloud 估计的 geometric normal，通过 Poisson surface reconstruction 得到更 clean、sharp detail 的 surface。这是因为 normal 直接预测避免了 point cloud 的 noise 放大。

**In-the-wild generalization**：Figure 8、10 展示了 AI-generated video 与 real multi-view image 的重建结果。Model 对不同 style 的 AI video（cartoon、realistic、stylized）都能生成 plausible 3DGS，说明 generalization 能力很强。

---

## 六、Limitations

1. **Dynamic scene 与 autonomous driving** 表现次优，原因是 training data 中这两类 under-represented
2. **Resolution 限制**：300-700 pixels，无法处理 thousands of input views（消费级 GPU memory 限制）

这两个 limitation 都指向未来方向：data expansion + computational optimization（如 chunked processing、memory-efficient attention）。

---

## 七、与相关工作 positioning

- **DUSt3R / MASt3R**：point map prediction 的 pioneer，但限于 image pair
- **Fast3R**：扩展到 1000+ images，但仍 image-only input
- **VGGT**：multi-task 里程碑，但缺 normal 与 NVS
- **π³**：去除 reference-view bias，permutation-equivariant
- **Dens3R**：dense prediction backbone for joint geometry
- **Pow3R**：DUSt3R + extra modality input，但仍限于 sparse-view "3R" paradigm
- **AnySplat**：feed-forward 3DGS from uncalibrated images
- **FLARE**：cascaded pose-geometry-appearance pipeline for sparse NVS

WorldMirror 的独特性：**首次在 dense multi-view reconstruction 框架内系统性探索 multi-modal prior injection**，同时 unify 了最广范围的 geometric task（point/depth/camera/normal/NVS）。

---

## 八、Intuition 总结

这篇 paper 的核心 insight 可以归纳为三点：

1. **Prior 是 "可插拔" 的 context signal**。通过把 camera pose、intrinsics、depth 各自 embedding 为合适形式的 token，model 能在 inference 时根据 available prior 自适应调整。这本质上是把 3D 重建从 "blind inference" 推向 "informed inference"。

2. **Multi-task joint training 让 representation 互相 regularize**。Normal 预测依赖 local geometry，depth 是 scale-aware 的 geometry，point map 是 multi-view consistent 的 geometry，3DGS 是 geometry + appearance 的综合。Joint training 让这些 supervision signal 互相 sharpen model 的 internal 3D representation。

3. **Curriculum 与 dynamic injection 是 multi-task 训练的关键 stabilizer**。Multi-task training 容易 gradient conflict 与 task imbalance。Curriculum 通过 sequencing 控制复杂度递增，dynamic injection 通过 prior dropout 避免 over-reliance，二者共同让 training 收敛到 robust 解。

从 engineering 角度，这个工作展示了一个 trend：3D vision foundation model 正在向 "universal input + universal output" 演进，类似 LLM 在 NLP 中的发展轨迹。WorldMirror 是这个方向上的一个重要 milestone。

---

**参考链接**

- VGGT: https://arxiv.org/abs/2503.11651
- π³: https://arxiv.org/abs/2507.13347
- DUSt3R: https://arxiv.org/abs/2312.14132
- MASt3R: https://arxiv.org/abs/2406.09656
- Fast3R: https://arxiv.org/abs/2411.15381
- Pow3R: https://arxiv.org/abs/2409.07543
- AnySplat: https://arxiv.org/abs/2505.23716
- FLARE: https://arxiv.org/abs/2502.15757
- Dens3R: https://arxiv.org/abs/2507.16290
- StableNormal: https://arxiv.org/abs/2409.16991
- GeoWizard: https://arxiv.org/abs/2403.12013
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14737
- gsplat: https://arxiv.org/abs/2409.11342
- DPT: https://arxiv.org/abs/2103.13413
- UniDepth: https://arxiv.org/abs/2403.18913
- CUT3R: https://arxiv.org/abs/2412.04657
- MonST3R: https://arxiv.org/abs/2410.03825
- Splatt3R: https://arxiv.org/abs/2408.13912
- NoPoSplat: https://arxiv.org/abs/2410.24207
- MVS-Splat: https://arxiv.org/abs/2405.12218
- DepthSplat: https://arxiv.org/abs/2406.07597
- Depth Anything: https://arxiv.org/abs/2401.10891
- DUSt3R GitHub: https://github.com/naver/dust3r
- VGGT GitHub: https://github.com/facebookresearch/vggt
- 3D Gaussian Splatting 官方: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- HunyuanWorld 1.0: https://hunyuan.tencent.com/modelSquare/home/playground?modelId=2025

后续如果继续 build 在这个方向上，关键问题会围绕：long sequence 处理（chunked attention 或 state-space model）、dynamic scene（4D representation）、metric scale recovery（IMU / GPS fusion）、以及与 video diffusion model 的 deeper integration（让 generative model 直接以 WorldMirror 输出为 3D structure prior）。这是 3D foundation model 与 generative world model convergence 的重要 axis。
