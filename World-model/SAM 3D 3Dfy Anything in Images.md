---
source_pdf: SAM 3D 3Dfy Anything in Images.pdf
paper_sha256: e0b8cf4668efa243227d28ebcf3e02adcce3a84d3d20eb42951e66cc084b82c4
processed_at: '2026-08-12T02:51:33-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SAM 3D 人话版

## 一句话说清楚这帮人干了啥

Meta 的人把 LLM 训练那套流程 — pretraining → mid-training → SFT → DPO → distillation — 原封不动搬到 3D 重建上, 做出一个从单张照片生成完整 3D 物体 (shape + texture + 6D pose) 的 model, 吊打所有同行。

核心难点跟 LLM 当年一样: **没数据**。

---

## 为什么 3D 这事儿一直卡着

文字有互联网, 图像有 SA-1B 那种 11 亿张标注, 视频有 YouTube。3D 呢? 你让一个标注员画个 mesh, 人家不会 — 这得 3D artist 上, 一个 mesh 5 分钟到 5 小时, 中位数 4751 vertices。你算算, 一万个 mesh 就得烧掉多少 artist hour。这就是 paper 里说的 "3D data barrier"。

更扎心的是: 普通人连 *生成* 一个候选 mesh 都做不到, 但让他从 8 个候选里挑一个最好的, 他又很在行。这是整篇 paper 最关键的观察, 也是数据飞轮能转起来的物理基础。

---

## 他们的骚操作: 把"生成"降维成"选择"

你让 GPT-4 写代码很难评估好坏, 但你让一个 senior engineer 从 5 个候选实现里挑一个最好的, 他两分钟搞定。SAM 3D 玩的就是这个:

1. Model (或者一堆 model 的 ensemble) 生成 8 个候选 mesh
2. 标注员 pairwise 比较, 选出 best-of-8
3. 选中的当 positive training sample, 没选中的当 DPO 的 negative
4. 用这些数据 SFT + DPO 训练 model
5. 新 model 更强 → 生成更好的候选 → 标注更容易 → 数据更多 → model 更强...

这就是他们说的 "data flywheel"。滚几轮之后, 自家 model 产生的候选占 80%, 其他 baseline 沦为陪跑。

---

## 架构: 两段式, 粗到细

跟 Trellis 的思路一脉相承:

**第一段 (Geometry Model, 1.2B 参数)**: 输入 image + mask, 输出 $64^3$ voxel 的 coarse shape + rotation + translation + scale。用 flow matching 训练, 就是个 rectified flow 的 transformer。

**第二段 (Texture & Refinement Model, 600M 参数)**: 拿第一段的 voxel 当 skeleton, 往上贴 texture + 细化几何, 输出 SLAT latents, 通过 VAE 解码成 mesh 或 3D Gaussian splats。

关键架构 trick 是 **Mixture-of-Transformers (MoT)**: shape 和 layout (R, t, s) 各有独立的 transformer stream, 但在 attention 层共享信息。好处是 shape-only 的数据集可以只训 shape stream, layout 的 finetune 不会破坏 shape 能力。这就跟 LLM 里给不同 modality 各配 expert 一个道理, 只是这里按 modality 而非 token position 切分。

另一个小创新是 **Depth-VAE**: 原版 Trellis 把 image feature back-project 到所有 voxel 包括被遮挡的, 导致 blur。SAM 3D 用 depth buffer 只 project 到可见 voxel, PSNR 从 30.65 提到 31.60。

---

## 训练流程: 完整复刻 LLM playbook

这是 paper 最值得复用的方法论。

### Step 1: Pretraining (Iso-3DO)

2.7M 个 synthetic mesh, 每个渲染 24 个视角, 64.8M 张图, 2.5T tokens。学的是 shape 和 texture 的 vocabulary, 跟 LLM pretrain 学语法一个意思。

过滤策略: rule-based, 滤掉退化几何 (体积太小、太平、有 spatial outliers)。aesthetic score 不能 capture geometric quality, 这跟 LLM 里"漂亮 ≠ 正确"同理。

### Step 1.5: Mid-Training (RP-3DO)

这步学的是: 怎么在有遮挡、有 clutter 的真实图像里找 object、估计 pose。完全靠 render-paste 合成数据, 61M samples。

三个难度档:

**Flying Occlusions (FO)**: 两个 synthetic object 互挡, 粘到真实图上。学 occlusion robustness。借鉴 Flying Chairs 那个 optical flow 的老套路。

**Object Swap - Random (OS-R)**: 用 depth estimator 估原 object 位置, inpaint 掉, 塞个 random synthetic mesh 进去。学 scale 和 translation 估计。关键是只保留有物理支撑 (放桌上) 或 partial occlusion 的样本, 这给 T-junction 和 depth ordering 的视觉 cue, 否则单目里 size 跟 depth 是 coupled 的, 学不出东西。

**Object Swap - Annotated (OS-A)**: 用 MITL 标注的精确 pose 和 best-match mesh 做 in-place 替换。这给 texture model 提供 pixel-aligned 的训练对, 让 texture 学到 fine-grained 的 image→mesh 对应。

### Step 2: Post-Training (MITL + Artists)

**MITL 数据引擎**: 就是前面说的 best-of-N + human ranking。每轮 generate → rank → SFT → DPO。quality threshold $\alpha_k$ 像退火一样逐轮提升, 早期宽松收数据, 后期严格保质量。最终 500K samples 入选。

**Art-3DO**: 最难的 case 所有 model 都失败, 路由给专业 3D artist 手工建模。这步看起来奢侈, 实际上 ablation 显示它掉点最多 — 去掉它 F1 从 0.2344 掉到 0.2027。MITL 能扩张 model 已有的"good islands", 但 seed 新 islands 得靠 expert。这跟 LLM 里"数学/代码 expert data > 通用 instruction data"的现象完全一致。

**Best-of-N + Reward Model**: N=8 不够时, 把 N 拉到 50, 用 learned reward model (VLM 或 DPO implicit reward) 做 tournament 筛到几个再交给人。这让 hard cases 的 recovery rate 从 0% 飙到 86.8%, food 类别 9× 提升。DPO implicit reward 跟 VLM 表现差不多, 说明训出来的 policy 本身就是 reward model, 不用单独训。

### Step 2.5: Distillation

最后用 shortcut model (Frans et al. 2024) 把 25 步推理压到 4 步, 10× 提速, 质量几乎不掉。初始化 trick: step size embedder 最后一层 linear 全置零, 因为这是新参数其他都加载 trained checkpoint。

---

## 为什么这套 recipe work

paper 反复强调一个观点: **geometric priors 不需要写进 loss**, scale 上来之后 flow matching 自己能学到。

Section C.2 原话: "flow matching objective is sufficient for SAM 3D to learn the task of 3D reconstruction, without explicitly enforcing geometric constraints through loss objectives."

这跟 LLM 上的观察一样: 小 model 小数据时加 inductive bias 有用, 大 model 大数据时 inductive bias 反而限上限, 让 model 自己从数据里学更 general 的 prior。SAM 3D 把 symmetry, closure, physical plausibility 这些都靠 DPO preference 信号隐式学到, loss 里一个 explicit geometric term 都没有。

rotation representation 的 ablation 也印证这点: 6D rotation (Zhou et al. 2019) 比 quaternion 好, 加 normalization 更好。flow matching 对 representation 的 smoothness 敏感, quaternion 在 SO(3) 上的 discontinuity 让 velocity field 学起来事倍功半。

---

## 结果

SA-3DAO (新 benchmark, 1000 个 artist 手工 mesh): Chamfer distance 几乎是 Trellis 的 1/2, F1 提升 60%。Human preference 对 Trellis 88% win rate, 对 Hunyuan3D-2.1 也是 87%。Layout 上 ADD-S@0.1 从 baseline 的 2% 飙到 77% — 这是数量级跳跃。

Table 4 的 multi-stage 累积 ablation 非常干净: 每加一个 stage 单调改进, 没有任何 stage 是多余的。Pretraining 0.1349 F1 → 加 mid-training 0.1705 → 加 SFT 0.2027 → 加 DPO 0.2156 → 加 Art-3DO SFT 0.2331 → 加 Art-3DO DPO 0.2344。这套 recipe 在 3D 上跟 LLM 上一样, 缺一不可。

---

## 可迁移到其他领域的 insight

1. **Generation → Verification 降维**: 当 human 无法 generate 但能 verify, best-of-N + ranking 是数据飞轮的引擎。对任何 "human 是 oracle 但不是 generator" 的 modality 都适用 — 3D mesh, 长文写作, 复杂代码, 数学证明。

2. **Multi-stage pretraining 适用于 perception/generation**: LLM 上的 mid-training (Code Llama 加代码能力) 经验可以 transfer 到 3D perception。RP-3DO 的三个变体就是 curriculum: random 在前, annotated 在后, 难度递增。

3. **Policy 本身就是 reward model**: DPO implicit reward 跟 VLM 表现相当, 不用单独训 reward。这跟 RLHF 里 reward model 与 policy 共享 backbone 的思路一致, 也跟 "DPO 让 LLM 自带 reward signal" 的观察一致。

4. **Expert data 在长尾上不可替代**: MITL 能扩张 good islands, 但 seed new islands 必须 expert。这是 "数据质量 > 数据数量" 在长尾上的极致体现, 也解释了为什么 LLM 上一小撮高质量数学/代码数据的效果远超大量普通 instruction data。

5. **Inductive bias 随 scale 衰减**: 小数据时 geometric prior loss 有用, 大数据时反而限上限。Flow matching 一个 explicit constraint 都不加, 全靠数据 + DPO, 这条路走到 SOTA。

6. **Representation smoothness 决定 generative model 难度**: 6D rotation > quaternion, 因为 flow matching 对 discontinuity 敏感。任何用 flow matching / diffusion 训 generative model 的任务, 都得先想想 target representation 在参数空间是不是 smooth 的。

---

## 我会怎么预测下一步

- **Resolution**: $64^3$ voxel 是硬上限, 手指、人脸细节会被吃掉。XCube 的 sparse voxel hierarchy 或 parts-based generation 是自然的下一步。
- **Multi-object joint reasoning**: 现在一个一个 object 预测, 不考虑 contact/stability/互穿。Scene-level diffusion + inter-object attention 应该会出来。
- **Pose-conditioned texture**: Texture model 不吃 pose 导致对称物体纹理可能"旋转错"。把 (R, t, s) feed 进去就解决了。
- **Video SAM 3D**: SAM 2 给了 mask track, 加时序 consistency, multi-frame 重建一个 object 应该 quality 提升明显。
- **Articulated objects**: 人、动物这种 articulated 结构现在靠数据长尾硬撑, skeleton-aware latent (类 SMPL 但通用) 应该有空间。
- **Robotics closed loop**: Paper 自己说 robotics 是 downstream, FoundationPose 整合已经显示路径。闭环 manipulation 里的 in-the-wild 3D perception 是显而易见的应用。

---

## 一句话总结

这篇 paper 的 contribution 不在 architecture (基本是 Trellis + MoT 微调), 而在 **把 LLM 训练 playbook 工程化到 3D**。核心 insight 是把 "human cannot generate but can verify" 这个性质榨干, 用 best-of-N 数据飞轮 + expert seeding 打破数据壁垒。这套方法论对任何 "标注昂贵但验证便宜" 的 modality 都适用, 远不止 3D。

Demo 可以去玩玩: https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d

---

# SAM 3D: 3Dfy Anything in Images — 深度解析

这篇 paper 来自 Meta Superintelligence Labs (Meta FAIR 的延续), 作者阵容里 Piotr Dollár, Georgia Gkioxari, Jitendra Malik 都是你熟悉的老朋友。核心 idea 用一句话总结: **把 LLM 的多阶段训练 recipe (pretraining → mid-training → post-training with preference alignment) 完整搬到 3D 生成上, 用一个 human- & model-in-the-loop data engine 打破 3D 数据稀缺的瓶颈, 最终从单张 natural image 重建出 object 的 shape + texture + 6D layout, 处理 occlusion 和 clutter 如家常便饭**。

项目主页: https://ai.meta.com/sam3d  
代码: https://github.com/facebookresearch/sam-3d-objects  
在线 demo: https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d

---

## 1. 为什么这件事 hard: 3D 的 "数据壁垒"

文本/图像/视频领域有 internet-scale 的监督信号 (web crawl + CLIP-style contrastive), 但 3D 完全没有这种 free lunch。原因有二:

(a) **3D ground truth 极贵**: 一个 experienced artist 做一个 mesh 需要 5 分钟到 5 小时 (Section D.1), median 4751 vertices。这跟 SAM 当年用 1B masks 的速度差了几个量级。

(b) **Generalist annotator 完全无法直接产生 mesh**: 这跟 segmentation 不一样, 普通人画不出 mesh, 但普通人 *能从 N 个候选里挑出最好的*。这是整个 paper 最深刻的 insight: **把 "生成" 任务降维成 "验证" 任务**, 让数据飞轮得以转动。这跟 RLHF 里 human as reward model 的思想异曲同工, 但更彻底——这里 human 既做 verifier 又做 selection oracle。

数据飞轮最终产出:
- 3.14M trainable shapes
- 1.23M layout samples  
- 100K trainable textures
- 7M+ pairwise preferences
- 总共 ~1M annotated images

---

## 2. Problem Formulation: 把摄影看成可逆映射

摄影是一个 3D → 2D 的 lossy projection。设 object 有 shape $S$, texture $T$, 在相机坐标系下的 rotation $R$、translation $t$、scale $s$。Photo 把 $(S, T, R, t, s)$ 映射到 image $I$ 中的像素集合 (由 mask $M$ 指定)。我们要 invert 这个 mapping:

$$p(S, T, R, t, s \mid I, M)$$

由于 inverse 是 one-to-many, 直接建模成条件分布而非 point estimate。模型 $q_\theta(S, T, R, t, s \mid I, M)$ 学着去逼近 $p$。

变量含义:
- $S$: shape (后面用 voxel latent $O \in \mathbb{R}^{64^3}$ 表示)
- $T$: texture (SLAT latents)
- $R \in \mathbb{R}^6$: 6D rotation representation (Zhou et al. 2019, https://arxiv.org/abs/1812.07035)
- $t \in \mathbb{R}^3$: translation in normalized camera coordinates
- $s \in \mathbb{R}^3$: scale per-axis
- $I$: 输入 image, $M$: 输入 mask (指定重建哪个 object)

为什么 6D 而不是 quaternion? Table 10 ablation 说明 6D 给 generative flow matching 一个 smoother optimization landscape, ICP-Rot error 从 17.96° 降到 15.54°, 加上 normalization 后再降到 14.59°。Quaternion 在 SO(3) 上的 discontinuity 让 flow matching 的 velocity field 学起来更难。

---

## 3. Architecture: 两阶段 + Mixture-of-Transformers

整体设计承接 Trellis (Xiang et al. 2025, https://arxiv.org/abs/2412.01506) 的 latent flow matching 思路, 但有三个关键升级:

### 3.1 Input Encoding
用 **DINOv2** (Oquab et al. 2023, https://arxiv.org/abs/2304.07193) 做 encoder, 提取 4 组 conditioning tokens:

1. Cropped image $I_{crop}$ (由 $M$ 裁出) + cropped binary mask
2. Full image $I$ + full image binary mask

Cropped view 给 object 高分辨率细节, full image 给全局 context 和 recognition cues。心理学上这对应 "pictorial cues" — 影调、纹理、context、familiar object recognition。Karpathy 你会喜欢这里: Koenderink 1992 那篇 perception 论文被引用, 真正把 "recognition enables reconstruction" 这个 idea (Roberts 1963) 在大规模数据上落地了。

可选: coarse scene pointmap $P$ (来自 LiDAR / iPhone depth / Depth Anything (Yang et al. 2024, https://arxiv.org/abs/2401.10891) / MoGe (Wang et al. 2025, https://arxiv.org/abs/2503.21744))。这让 SAM 3D 可以跟 sensor pipeline 互补。ablation (Section E.5) 显示 pointmap 几乎不影响 shape 质量 (48% / 48% 平局), 但对 layout 至关重要。

### 3.2 Geometry Model: Mixture-of-Transformers

1.2B 参数 flow transformer, 采用 **Mixture-of-Transformers (MoT)** (Liang et al. 2025, https://openreview.net/forum?id=Nu6N69i8SB; Deng et al. 2025, https://arxiv.org/abs/2505.14683)。两个并行的 transformer stream:

- Stream A: shape tokens (4096 个, 来自 $16^3 \times 8$ latent of $64^3$ voxels)
- Stream B: layout tokens (1 个, 对应 $(R, t, s) \in \mathbb{R}^{12}$)

两个 stream 在 multi-modal self-attention 层共享信息, 但各自有独立的 FFN / projection。Attention mask 设计让某些层允许 cross-modal attention, 某些层只允许 intra-modal attention (Figure 2 right)。好处是:

1. 当某个 dataset 只有 shape label (没 layout), 可以单独训 shape 而不损害 layout 能力;
2. 当 frozen shape 只 finetune layout 时, shape stream 不动, 训练 cost 极低;
3. Cross-modal attention 让 rotation $R$ 锚到正确的 shape 上 (self-consistency)。

**Intuition**: MoT 本质上是给不同 modality 各自的 "专家 FFN", 但在 attention 层做 information fusion。这跟 MoE 在 LLM 里的角色很像, 只是这里的 "experts" 是按 *modality* 而非 *token position* 划分的。

### 3.3 Texture & Refinement Model
600M 参数 sparse latent flow transformer (DiT-style, Peebles & Xie 2023, https://arxiv.org/abs/2212.09748), 输入是 Geometry model 预测的 active voxels (sparse), 输出是 refined SLAT (Structured Latent, Xiang et al. 2025) features。两个 VAE decoder 共享同一个 encoder:

- $\mathcal{D}_m$: 解出 mesh
- $\mathcal{D}_g$: 解出 3D Gaussian splats (3DGS)

### 3.4 Depth-VAE: 重要的小创新

原版 Trellis VAE 把 image features back-project 到 *所有* voxel, 包括被遮挡的 voxel, 导致 reconstruction blur。SAM 3D 引入 **Depth-VAE**:

对 feature map $\mathbf{F} \in \mathbb{R}^{B \times C \times H \times W}$, 坐标 $\mathbf{U} \in [-1, 1]^{B \times N \times 2}$, 算法:

1. **Bilinear sample**: 对每个 $\mathbf{u}_i$ 提取 feature $\mathbf{f}_i$ via GridSample
2. **Depth buffer**: 对每个 pixel 取最小预测深度
   $$\mathbf{D}_{surf}(x, y) = \min_{i: (x_i, y_i) = (x, y)} \hat{d}_i$$
3. **Visibility mask**:
   $$\mathbf{M}_i = \mathbb{I}[\mathbf{d}_{ref,i} > \hat{d}_i - \tau]$$
   其中 $\tau$ 是 tolerance。$\hat{d}_i$ 是该 voxel 的预测深度, $\mathbf{d}_{ref,i}$ 是该 2D 位置上 surface 的最小深度。如果 voxel 深度比 surface 浅 (即在 surface 前面或贴近), 则 visible。
4. **Weighted aggregation**:
   $$\mathbf{F}_{depth} = \sum_b \tilde{\mathbf{M}}_b \odot \mathbf{f}_b$$

其中 $\tilde{\mathbf{M}}$ 是 batch 维度归一化的 mask。Table 11 显示 Depth-VAE 把 PSNR 从 30.65 提到 30.87, 加上数据 scaling 到 31.60, LPIPS 从 0.0478 降到 0.041。

---

## 4. Training Recipe: LLM-style 多阶段

这是 paper 的核心 methodological contribution, 几乎 1:1 复刻 LLM 训练 playbook:

### Stage 1: Pretraining — Iso-3DO

**数据**: 2.7M meshes (来自 Objaverse-XL (Deitke et al. 2023, https://arxiv.org/abs/2307.05663) + 授权数据), 每个从 24 视角渲染, ~64.8M images, 总共 **2.5T tokens**。

**目的**: 学 shape & texture vocabulary, 建立 "prior over plausible 3D objects"。这一步是 supervised pretraining — 给定渲染图像 + ground truth shape/texture, 直接 flow matching 训练。

**过滤策略** (Section B.1): rule-based, 滤掉退化几何 (volume 太小 / normal variance 太小 / spatial outliers)。这跟 Xiang et al. 2025 的 aesthetics filter 不同, 因为 aesthetic score 不能 capture geometric quality。

**Modality weights**: $S = 1.0, R = 0.1$。Shape 是主角, rotation 给一点 weight 预热。

### Stage 1.5: Mid-Training — RP-3DO (Render-Paste)

这是 paper 的另一个关键 idea: 用 "render-paste" 把 synthetic mesh 嵌进 natural image, 桥接 synthetic→real gap。共 61M samples, 2.87M unique meshes, ~2.7T tokens。

三个子集, 难度递增:

**(a) Flying Occlusions (FO)** — 55.1M samples

Inspired by Flying Chairs (Dosovitskiy et al. 2015, https://arxiv.org/abs/1504.06852) 和 FlyingThings3D (Mayer et al. 2016, https://arxiv.org/abs/1612.02001)。把两个 synthetic object 合成到一张 natural image 上: occluder + occludee。

可见 mask:
$$M_{vis} = M_{obj} \odot (1 - M_{occluder})$$

约束:
- $0.1 \leq |M_{vis}|/|M_{obj}| \leq 0.9$ (occlusion 不能太重也不能太轻)
- $|M_{vis}|/|I| \geq 0.2\%$ (object 在图里不能太小)

1/3 样本里把 selected mesh 当 occluder (mask 完整), 避免模型永远预测被遮挡 object。

**Intuition**: 这训练了 occlusion robustness, 同时强制模型用 full image context 而不只是 object crop。

**(b) Object Swap - Random (OS-R)** — 5.95M samples

比 FO 更 realistic: 用 pointmap 估计原 object 的 3D centroid 和 bbox, inpaint 掉原 object, 再塞一个 random synthetic mesh 进去 (随机旋转, 用 bbox 拟合 scale)。

**关键 trick**: 只保留 *有 physical support 或 partial occlusion* 的样本, 这给模型 depth ordering 和 T-junction cues:
- Physical support: object 底部 10% 边界处 background 比 object 更近 (说明 object 在表面上)
- Partial occlusion: 至少 10% 周长被前景遮挡

**Intuition**: 单目图像里 size 与 depth 是 coupled 的, 只有当 object 与环境有 spatial 关系 (放桌上、被前面物体挡) 时, scale 估计才 well-posed。Random 粘贴没有这种约束, 模型学到的 scale 会偏。

**(c) Object Swap - Annotated (OS-A)** — 0.4M samples

OS-R 的精修版: 用 MITL-3DO 标注的精确 $(R, t, s)$ 和 best-match mesh。这给 Texture & Refinement model 提供 pixel-aligned 训练对, 让 texture 学到 fine-grained 的 image-mesh 对应关系。

### Stage 2: Post-Training — MITL Data Engine

这是 paper 的灵魂。Cold-start 问题: 第一轮没有好 model 生成候选, 怎么办? 用一个 **ensemble** (retrieval + text-to-3D + 各种 image-to-3D baseline + 自家 checkpoint) 一起生成 N=8 候选, 让 human annotator 选最好的。随着训练进行, 自家 model 占比逐渐升至 ~80%。

#### Algorithm 1 (SAM 3D Basic Alignment)

核心循环:
```
for k = 1 to K:
    # Collection
    for (I, M) ~ p(I, M):
        π̃_k = Amplify(π_{k-1})  # ensemble + best-of-N
        {d_i}_{i=1}^N ~ π̃_k(I, M)  # 生成 N 个候选
        d*, r = HumanRank({d_i})  # 人选最好的 + 评分
        R = {d_i : i ≠ argmax}  # 拒掉的做 negative
        C_k ∪= {(d*, r, R)}
    # Update
    C = {(d+, R) : r ≥ α_k}  # quality threshold
    D = {(d+, d-) : d- ∈ R}  # preference pairs
    π_SFT = argmin L_CFM(π; d+)
    π_k = argmin L_DPO(π, π_SFT; d+, d-)
```

关键设计:
- **Quality threshold curriculum $\alpha_k$**: 随训练提升, 像 cross-entropy method (de Boer et al. 2005, https://link.springer.com/article/10.1007/s10479-005-5724-z)。早期 $r \geq \alpha_1$ 宽松, 后期 $r \geq \alpha_K$ 严格, 最终 500K samples 入选。
- **Amplification factor**: 当前 model 与 expert policy 之间的 gap, 决定每一轮上界。
- **Stepwise efficiency**: 新 model 多接近上一轮的 expert。这跟 Expert Iteration (Anthony et al. 2017, https://arxiv.org/abs/1705.08439) 同构。

#### Stage 2.5: 3D Artists (Art-3DO)

最难的 case (white region in Figure 14), 所有 model 都失败时, 路由给专业 3D artist 手工建模。这相当于 "seed new islands" 在 data distribution 上 — 不靠 artist, 这些区域永远靠 MITL 自己长不出来。

#### Stage 3: Pose Alignment to 2.5D

给 annotator 一个 depth estimator 估出的 2.5D point cloud, 让他们用键盘/鼠标把 3D mesh 旋转平移缩放到对齐 point cloud。提供 mesh IoU indicator 实时反馈。这一步是 layout ground truth 的来源。平均 150s/sample。

#### A.7: Best-of-N with Reward Models — 推进 tail 的关键 trick

普通 best-of-N 中 N 不能太大, 否则 human 选项太多, choice overload (Diehl & Poynor 2010) 让 ranking 变随机。SAM 3D 的解法: 先用 **learned reward model** 做 tournament ranking, 把 N=50 候选筛到几个再交给人。

两个 reward model 都试过:
- VLM (CLIP-style): 68.9% 与 human 二选一 agreement
- DPO implicit reward: ~65% agreement
- 两个 human 之间: <75% agreement

DPO-as-reward 贡献了 ~80% 的 recovery data。把原本 0% 成功率的 hard cases 提到 86.8%, food 类别从 4% 提到 36% (9×)。Table 12 显示 finetune 在这些 recovered data 上, Chamfer 和 F1 在 tail holdout / Epic Kitchens / SA-3DAO 上全部改善。

---

## 5. Training Objectives — 公式逐项拆解

### 5.1 Conditional Rectified Flow Matching (Pretraining & SFT)

$$\mathcal{L}_{CFM} = \sum_{m \in \mathcal{M}} \lambda_m \cdot \mathbb{E}\left[\| \mathbf{v}^m - \mathbf{v}_\theta^m(\mathbf{x}_\tau^m, c, \tau) \|^2\right]$$

变量:
- $m \in \mathcal{M} = \{S, R, t, s\}$: modality index
- $\lambda_m$: per-modality weight (e.g. $S=1.0, R=0.1, t=1.0, s=0.1$)
- $c = (I, M)$: conditioning (image, mask, optional $P$)
- $\mathbf{x}_1^m$: ground-truth clean state (annotation)
- $\mathbf{x}_0^m \sim \mathcal{N}(0, \mathbf{I})$: initial Gaussian noise
- $\mathbf{x}_\tau^m = \tau \mathbf{x}_1^m + (1 - \tau) \mathbf{x}_0^m$: linear interpolation path, $\tau \in [0, 1]$
- $\mathbf{v}^m = \dot{\mathbf{x}}_\tau^m = \mathbf{x}_1^m - \mathbf{x}_0^m$: target velocity (constant, because path is linear)
- $\mathbf{v}_\theta^m(\mathbf{x}_\tau^m, c, \tau)$: 网络预测的 velocity field

**Intuition**: rectified flow 是 straight-path diffusion, 跟 DDPM 的 curved trajectory 相比, sampling 可以用更少步数 (Liu et al. 2022, https://arxiv.org/abs/2209.03003)。Linear interpolation 的 target velocity 是常数, 训练目标很 clean。

为什么 multi-modal 一起训用 shared backbone 而不是 N 个独立 model? 因为 $(S, R, t, s)$ 之间有强 dependency: rotation 只有 anchored 到具体 shape 才有意义, scale 与 shape 的物理 extent 耦合, translation 依赖 scale。MoT 的 cross-modal attention 让这些 dependency 在 forward pass 自然涌现。

### 5.2 DPO (Preference Alignment)

承接 Diffusion-DPO (Wallace et al. 2024, https://arxiv.org/put/DPO_diffusion), 改写成 flow matching:

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(-\beta T w(\tau) \cdot \Delta\right)\right]$$

$$\Delta = \underbrace{\|\mathbf{v}^w - \mathbf{v}_\theta(\mathbf{x}_\tau^w, c, \tau)\|_2^2 - \|\mathbf{v}^w - \mathbf{v}_{ref}(\mathbf{x}_\tau^w, c, \tau)\|_2^2}_{\text{winner term}} - \underbrace{\left(\|\mathbf{v}^l - \mathbf{v}_\theta(\mathbf{x}_\tau^l, c, \tau)\|_2^2 - \|\mathbf{v}^l - \mathbf{v}_{ref}(\mathbf{x}_\tau^l, c, \tau)\|_2^2\right)}_{\text{loser term}}$$

变量:
- $(\mathbf{x}_0^w, \mathbf{x}_0^l)$: human-preferred 和 less-preferred 样本
- $\mathbf{v}^w, \mathbf{v}^l$: 对应的 target velocity
- $\mathbf{v}_\theta$: learnable policy
- $\mathbf{v}_{ref}$: frozen reference (post-SFT checkpoint)
- $\beta$: temperature (DPO 的 KL penalty strength)
- $T$: 总时间
- $w(\tau)$: time-dependent weighting
- $\Delta > 0$: winner 在 learnable policy 下比在 ref 下 *更差* (loss 更大), loser 反之 — 即 model 当前偏向 loser, gradient 拉回 winner

**Intuition**: 这本质上是在 score (loss) 空间做 Bradley-Terry。winner 的 loss 应该 *小于* ref 的 loss, loser 的 loss 应该 *大于* ref 的 loss。$\Delta$ 衡量的是 (winner gap) - (loser gap), 我们要 $\Delta < 0$ (winner gap 比 loser gap 小, 即 model 更贴近 winner)。

Implementation detail: 只用 SAM 3D 自己生成的 negatives 做 DPO, 把 retrieval-based 和 multi-view diffusion 的 negatives 排除, 因为它们 out-of-distribution, 信号噪声比太低。

### 5.3 Shortcut Model Distillation

承接 Frans et al. 2024 (https://arxiv.org/abs/2410.12557):

$$\mathcal{L}_S(\theta) = \mathbb{E}_{\mathbf{x}_0 \sim \mathcal{N}(0, \mathbf{I}), \mathbf{x}_1 \sim p(\mathbf{x})}\left[\underbrace{\|\mathbf{v} - \mathbf{v}_\theta(\mathbf{x}_\tau, c, \tau, d=0)\|^2}_{\text{flow matching term}} + \underbrace{\|\mathbf{v}_{consistency} - \mathbf{v}_\theta(\mathbf{x}_\tau, c, \tau, 2d)\|^2}_{\text{shortcut term}}\right]$$

变量:
- $d$: step size, $d=0$ → 标准 flow matching; $d > 0$ → shortcut mode (大步跳)
- $\mathbf{v}$: empirical instantaneous velocity
- $\mathbf{v}_{consistency}$: 由 Algorithm 2 构造, 用两次 step $d$ 拼成一次 step $2d$ 的 target velocity
- $p(\tau, d)$: 联合采样分布

Algorithm 2 还引入 CFG (classifier-free guidance) 把 $w_{CFG}$ 蒸进 shortcut mode: Stage 1 用 $w=2$, Stage 2 用 $w=1$。最后 4K iterations, 75% flow matching + 25% shortcut。

**结果**: NFE 从 25 降到 4, 推理快 10×; 1-step 提速 38× (Figure 18)。

初始化 trick: step size embedder 的最后 linear 层 weights 和 bias 全初始化为 0, 因为这是新参数, 其他都从 trained checkpoint 加载。

---

## 6. SA-3DAO Benchmark

新提出的 benchmark: 1000 个 3D artist 手工建的 mesh, paired with natural images。覆盖 churches, ski lifts, escalators, animals, household items, tribal masks 等长尾 object。每图 object 数大致 power-law 分布 (Figure 16)。Median 4751 vertices, 每个 mesh 5min–5h。

Metric 定义:

### Shape
- **F1@0.01**: 在 0.01 阈值下点云 correspondence 的 precision/recall harmonic mean
- **Voxel-IoU (vIoU)**: $64^3$ voxelization 后 IoU, 敏感于 volume/silhouette/topology 大错
- **Chamfer Distance (CD)**: 双向最近邻距离, 量化局部几何偏差
- **Earth Mover's Distance (EMD)**: 最优传输 cost, 比 CD 更严, 全局结构对齐

### Perceptual (no GT 时, ISO3D)
- **ULIP** (Xue et al. 2023, https://arxiv.org/abs/2303.05457)
- **Uni3D** (Zhou et al. 2023, https://arxiv.org/abs/2310.06773)
- 都基于 point cloud → CLIP-style embedding → 与 image embedding 相似度

### Layout
- **3D IoU**: 3D axis-aligned bounding box IoU
- **ICP-Rot**: ICP 对齐后的残余旋转角 (度)
- **ADD-S**: 对称化 average point distance, normalized by diameter:
  $$\text{ADD-S} = \frac{\text{ADD}(\mathcal{M}, \mathcal{M}_{gt}) + \text{ADD}(\mathcal{M}_{gt}, \mathcal{M})}{2d}$$
  其中 $d = \max_{x} \min_{y \in \mathcal{M}_{gt}} \|x - y\|$ 是 ground truth diameter
- **ADD-S@0.1**: ADD-S < 10% diameter 的 binary 指标

---

## 7. 实验数据

### 7.1 Shape (Table 2)

SA-3DAO 上 SAM 3D 对 SOTA 大幅领先:

| Model | F1@0.01 ↑ | vIoU ↑ | Chamfer ↓ | EMD ↓ |
|---|---|---|---|---|
| Trellis | 0.1475 | 0.1392 | 0.0902 | 0.2131 |
| Hunyuan3D-2.1 | 0.1399 | 0.1266 | 0.1126 | 0.2432 |
| Direct3D-S2 | 0.1513 | 0.1465 | 0.0962 | 0.2160 |
| Hi3DGen | 0.1629 | 0.1531 | 0.0937 | 0.2134 |
| **SAM 3D** | **0.2344** | **0.2311** | **0.0400** | **0.1211** |

Chamfer 几乎是 Trellis 的 1/2, F1 提升约 60%。

ISO3D (perceptual, 无 GT): SAM 3D ULIP 0.1488 vs TripoSG 0.1529 (TripoSG mesh resolution 更高), Uni3D 0.3707 (略胜)。

### 7.2 Human Preference (Figure 8)

- Object-level: SAM 3D 对 Trellis 88% win rate, 对 HY3D-2.0 77.5%
- Scene-level: 对 Trellis 92%, 对 MIDI 86%
- Texture (Figure 9, 给所有方法 SAM 3D shape): 对 Trellis 87%, 对 HY3D-2.1 87%, 对 Unitex 84.7% (Table 8)

### 7.3 Layout (Table 3)

| Model | 3D IoU ↑ | ADD-S@0.1 ↑ |
|---|---|---|
| HY3D-2.0 + FoundationPose | 0.2937 | 0.5396 |
| HY3D-2.0 + FoundationPose (ADT) | 0.3864 | 0.5992 |
| MIDI (ADT) | 0.0336 | 0.0175 |
| **SAM 3D** | 0.4254 | 0.7232 |
| **SAM 3D (ADT)** | 0.4970 | 0.7673 |

ADD-S@0.1 从 baseline 的 2% 提到 77% — 这是数量级跃迁。Section E.3 显示 test-time render-and-compare optimization 还能再加几个点 (0.4837→0.5258 3D IoU, 0.7545→0.7617 ADD-S@0.1)。

### 7.4 Multi-stage 训练的累积收益 (Table 4)

| Stage | F1@0.01 | vIoU | Chamfer | EMD |
|---|---|---|---|---|
| Pretraining (Iso-3DO) | 0.1349 | 0.1202 | 0.1036 | 0.2396 |
| + Mid-training (RP-3DO) | 0.1705 | 0.1683 | 0.0760 | 0.1821 |
| + SFT (MITL-3DO) | 0.2027 | 0.2025 | 0.0578 | 0.1510 |
| + DPO (MITL-3DO) | 0.2156 | 0.2156 | 0.0498 | 0.1367 |
| + SFT (Art-3DO) | 0.2331 | 0.2337 | 0.0445 | 0.1257 |
| + DPO (Art-3DO) | 0.2344 | 0.2311 | 0.0400 | 0.1211 |

每加一阶段都单调改进, 验证了 multi-stage 训练 recipe 的必要性。

### 7.5 Knockout ablation (Table 7)

去掉 MITL-3DO: F1 0.2344→0.2211  
去掉 Art-3DO: 0.2344→0.2027 (掉最多, artist data 极珍贵)  
去掉 DPO: 0.2344→0.2156

Art-3DO 影响最大, 说明 hard cases 的 expert annotation 是 quality ceiling 的关键决定因素。

---

## 8. 跟其他工作的关系

### 跟 SAM 系列
SAM (Kirillov et al. 2023, https://arxiv.org/abs/2304.02643) 解决 2D segmentation, SAM 2 (Ravi et al. 2025, https://arxiv.org/abs/2408.00714) 加上 video, 这篇 SAM 3D 把 "anything" 延伸到 3rd dimension。三者共用 promptable philosophy, mask 是 SAM 3D 的输入 prompt。

### 跟 Trellis
Trellis 是 single-object 的, Iso-3DO 训练, 推理只看 cropped object。SAM 3D 加上: full image conditioning, layout prediction, MoT 架构, multi-stage post-training, MITL data engine。可以理解为 Trellis + scene grounding + LLM-style alignment。

### 跟 FoundationPose / Megapose
这些是 model-based pose estimation, 需要 CAD model 输入。SAM 3D 是 model-free, joint 生成 shape + pose。Section E.3 显示 SAM 3D 还能当 FoundationPose 的 proposal generator, 通过 render-and-compare 进一步 refine。

### 跟 MIDI (Huang et al. 2025, https://arxiv.org/abs/2502.14492)
MIDI 也是 joint shape + pose 的 diffusion, 但限于 indoor / tabletop。SAM 3D 用 mask prompt 灵活选 object, 在 SA-3DAO / LVIS / ADT 都覆盖到。

### 跟 RAFT (Dong et al. 2023, https://arxiv.org/abs/2304.06698) / RFT / Expert Iteration
SAM 3D 的 alignment algorithm 最接近 RAFT — 都用 reward-ranked finetuning。区别:
1. SAM 3D 用 model ensemble 做专家策略 amplification (RAFT 用 reward model 直接 rank)
2. SAM 3D 加 preference (DPO) supervision, RAFT 是纯 imitation
3. SAM 3D 引入 human-in-the-loop 闭环, RAFT 是纯 self-training

### 跟 LLM 训练 playbook
paper Section 5 把自己定位为把 LLM recipe (pretraining → mid-training → SFT → DPO → distillation) 搬到 3D。类比:
- Iso-3DO ~ pretraining on web text
- RP-3DO ~ mid-training on domain-specific corpora (code, math)
- MITL-3DO ~ SFT on instruction data
- Art-3DO ~ high-quality expert data
- DPO ~ RLHF
- Shortcut distillation ~ quantization / speculative decoding

这个 analogy 在 Section 5 Related Work 里被 explicit 论证, 引用 Grattafiori et al. 2024 (Llama 3, https://arxiv.org/abs/2407.21783), OLMo 2, Rozière et al. 2024 (Code Llama, https://arxiv.org/abs/2308.12950), Lambert 2025 (RLHF book, https://rlhfbook.com)。

---

## 9. Limitations (Section F)

1. **Resolution cap**: $O \in \mathbb{R}^{64^3}$ voxels, 最多 32 splats/voxel。对 thin structures (手指、人脸细节、栏杆) 不够。Whole-body reconstruction 时 hands/face 占的 voxel budget 太少。
2. **No multi-object reasoning**: 一个一个 object 预测, 没考虑 contact / stability / 互穿 / 共面。Object 间可能 float 或 interpenetrate。
3. **Texture 跟 pose 解耦**: 对称物体可能预测出"旋转过"的 texture, 即使 shape 对了。这是因为 Texture & Refinement model 不吃 pose 作为输入。
4. **不处理 dynamic / non-rigid**: dataset 里大部分是 rigid, 对 articulated object (人、动物) 表现取决于训练数据的长尾覆盖。

---

## 10. 我 (Karpathy) 视角的 takeaway

这篇 paper 真正的 contribution 不是 architecture (基本是 Trellis + MoT 微调), 而是 *data engine 的工程化和理论化*。几个值得复用到其他领域的 idea:

1. **从 generation 降维到 verification**: 当 human 无法 generate 但能 verify 时, best-of-N + ranking 是数据飞轮的引擎。这跟 LLM RLHF 几乎一模一样, 但在 3D 这种人类 *完全无法* 直接 label 的 modality 上更彻底。

2. **Multi-stage pretraining 在 perception / generation 上的有效性**: LLM 上的 mid-training (Code Llama) 经验现在 transfer 到 perception。RP-3DO 的三个变体 (FO / OS-R / OS-A) 像一个 curriculum — random 在前, annotated 在后, 难度递增。

3. **Reward model 做 best-of-N pre-filter**: 当 N 需要 scale 但 human 无法消化时, learned reward 做 tournament 是必要的。Paper 显示 DPO implicit reward 跟 VLM 表现相当, 这意味着 *训出来的 policy 本身就是 reward model*, 不需要单独训 reward。这跟 RLHF 里 reward model 与 policy 共享 backbone 的思路一致。

4. **Expert data 在长尾上的不可替代性**: Table 7 显示去掉 Art-3DO 掉最多 (0.2344→0.2027)。这对应 LLM 里 "code/math expert data > general instruction data" 的现象。MITL 能扩张 islands 但不能 seed new islands, artist 才能。

5. **Flow matching + DPO 的简洁性**: 几何约束 (symmetry, closure) 不写进 loss, 而是靠 DPO 从 preference 信号里学。这是 paper 一个 counterintuitive 但 validated 的选择 — Section C.2 明确说: "flow matching objective is sufficient for SAM 3D to learn the task of 3D reconstruction, without explicitly enforcing geometric constraints". Geometric priors 在 scale 上会被 implicit 学到。

6. **6D rotation + normalization 比 quaternion 好**: 这是 Zhou et al. 2019 在 generative setting 下的再次确认。Flow matching 对 representation smoothness 敏感, 任何 discontinuity 都会让 velocity field 学起来吃力。

如果让我预测下一步:
- Resolution: sparse voxel hierarchy (XCube, Ren et al. 2024, https://arxiv.org/abs/2401.04594) 或 parts-based generation
- Multi-object joint reasoning: scene-level diffusion with inter-object attention
- Pose-conditioned texture: 把 $(R, t, s)$ feed 进 Texture model 解决对称物体 texture 旋转问题
- Video SAM 3D: 用 SAM 2 的 mask track + 时序 consistency 约束, multi-frame 重建一个 object
- Articulated objects: skeleton-aware latent (SMPL 之类 + 通用 mesh)
- 跟 robot learning 结合: SAM 3D 已经讨论了 robotics application, FoundationPose 整合显示 path, 下一步是 closed-loop manipulation 用的 in-the-wild 3D perception

Demo 真的可以玩玩, 几秒出 mesh + texture: https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d

---

## 参考链接汇总

- SAM 3D 项目: https://ai.meta.com/sam3d  
- SAM 3D 代码: https://github.com/facebookresearch/sam-3d-objects  
- SAM 3D Demo: https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d  
- Trellis (Xiang et al. 2025): https://arxiv.org/abs/2412.01506  
- DINOv2: https://arxiv.org/abs/2304.07193  
- 6D rotation (Zhou et al. 2019): https://arxiv.org/abs/1812.07035  
- DPO (Rafailov et al. 2023): https://arxiv.org/abs/2305.18290  
- Flow matching (Liu et al. 2022): https://arxiv.org/abs/2209.03003  
- Diffusion-DPO (Wallace et al. 2024): https://arxiv.org/abs/2310.03708  
- SAM (Kirillov et al. 2023): https://arxiv.org/abs/2304.02643  
- SAM 2 (Ravi et al. 2025): https://arxiv.org/abs/2408.00714  
- Objaverse-XL: https://arxiv.org/abs/2307.05663  
- Shortcut models (Frans et al. 2024): https://arxiv.org/abs/2410.12557  
- Megapose (Labbé et al. 2022): https://arxiv.org/abs/2212.06870  
- FoundationPose (Wen et al. 2024): https://arxiv.org/abs/2312.08379  
- MIDI (Huang et al. 2025): https://arxiv.org/abs/2502.14492  
- Hunyuan3D 2.1: https://arxiv.org/abs/2506.15442  
- Direct3D-S2: https://arxiv.org/abs/2505.17412  
- TripoSG: https://arxiv.org/abs/2502.06608  
- Hi3DGen: https://arxiv.org/abs/2503.22236  
- MoT (Liang et al. 2025): https://openreview.net/forum?id=Nu6N69i8SB  
- EmerCoMo / Deng et al. 2025 (multi-modal pretrain): https://arxiv.org/abs/2505.14683  
- ProcTHOR: https://arxiv.org/abs/2206.06994  
- Flying Chairs: https://arxiv.org/abs/1504.06852  
- FlyingThings3D: https://arxiv.org/abs/1612.02001  
- Depth Anything: https://arxiv.org/abs/2401.10891  
- MoGe: https://arxiv.org/abs/2503.21744  
- ULIP: https://arxiv.org/abs/2303.05457  
- Uni3D: https://arxiv.org/abs/2310.06773  
- XCube (Ren et al. 2024): https://arxiv.org/abs/2401.04594  
- Llama 3: https://arxiv.org/abs/2407.21783  
- Code Llama: https://arxiv.org/abs/2308.12950  
- RLHF book (Lambert 2025): https://rlhfbook.com  
- Expert Iteration (Anthony et al. 2017): https://arxiv.org/abs/1705.08439  
- RAFT (Dong et al. 2023): https://arxiv.org/abs/2304.06698  
- Cross-entropy method: https://link.springer.com/article/10.1007/s10479-005-5724-z  
- DiT (Peebles & Xie 2023): https://arxiv.org/abs/2212.09748  
- LVIS: https://arxiv.org/abs/1908.03195  
- 3D Arena: https://arxiv.org/abs/2506.18787  
- Ego4D: https://arxiv.org/abs/2110.07058  
- Ego-Exo4D: https://arxiv.org/abs/2401.10889  
- Aria Digital Twin: https://arxiv.org/abs/2305.17875  
- Unitex (Liang et al. 2025): https://arxiv.org/abs/2505.23253
