---
source_pdf: GEN3C.pdf
paper_sha256: 5f2462a60f2bed4f097ea79680de022eec4e5ea2f2a30660af9c2a2e80a067d7
processed_at: '2026-08-04T13:17:54-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GEN3C 用人话讲

好，我重新来一遍，抛开公式，讲讲这帮人到底干了啥。

---

## 一句话总结

别让 neural network 猜 3D，你直接把 3D 塞给它，它只管画图就行。

---

## 问题是啥

现在 video generation model（Sora、SVD 这些）已经很猛了，生成画面漂亮。但你想控制 camera，比如"往左走三步再看回来"，基本上会翻车：

- CameraCtrl 那帮人的做法是把 camera pose 变成一组数字（Plücker embedding），喂给 NN，让它"学会"相机和画面的关系。问题是这个 mapping 太难学了——同样是往左走，厨房和马路看到的东西完全不一样，NN 根本 generalize 不过去
- 更要命的是"回头看"的问题：camera 转过去再转回来，之前画面里那把椅子应该还在那里，但 NN 已经忘了它画过啥，于是椅子变了形状、消失了、或者冒出来一个新东西

本质原因：**NN 的 memory 全在 latent 里，latent 不是 3D 的**。你让它维护一个 3D-consistent 的世界，它没这个能力。

---

## GEN3C 怎么解决的

非常简单粗暴：**在 NN 外面挂一个点云**。

流程是这样的：

1. 你给一张图（或几张图、或一段视频）
2. 用 depth estimator（Depth Anything V2）估个 depth，把每个 pixel 往 3D 空间里一推，变成一个 colored point cloud
3. 你说要什么 camera trajectory，就把这个点云按新相机角度"投影"回 2D，得到一张 render。这个 render 肯定不完美——有洞（disocclusion）、有 depth 误差、有 misalignment
4. 把这个 render 喂给 video diffusion model，让它把洞补上、把错误修掉、生成漂亮视频

**核心 insight**：NN 不需要从 camera pose 推断 scene structure，因为 render 已经把 structure 告诉它了。NN 只需要做它擅长的事——hallucinate missing content 和 enhance visual quality。

---

## 几个 clever 的设计细节

### 1. Mask 怎么塞进去

Render 出来的图里有"洞"（点云没覆盖到的地方）。你得告诉 NN 哪里有洞需要补。

Naive 做法：把 mask 当成额外一个 channel concat 上去。但这会引入新的 model parameters，而这些 parameters 在 pre-training data 里没有对应的 representation，generalization 会差。

GEN3C 的做法：直接把 mask 乘到 latent 上。洞的地方 latent 直接变零。Model architecture 完全不变，所有 pre-trained weights 照常工作。NN 看到一片零，自然知道"这里我得自己画"。

这个设计很小巧但很关键——它意味着你不需要 retrain 整个 model，只需要 fine-tune，而且 fine-tune 的量很小。

### 2. 多视角怎么融合

如果你有好几个 input view（比如 driving 场景的前/左/右三个 camera），每个 view 都 unproject 出一个点云。怎么办？

Prior work（ReconX、ViewCrafter）的做法：先用 DUSt3R 把点云在 3D 里 align 成一个，再 render。问题是 DUSt3R 对 thin structures（电线杆、栏杆）不 robust，align 错了就出 ghosting artifact。

GEN3C 的做法：**不 fuse**。每个 view 单独 render、单独过 model 第一层，然后在 feature level 做 max-pool。

为什么 max-pool？因为它是一个 OR 操作——任何一个 view 在某个位置提供了 feature，就保留。这天然处理了不同 view 覆盖不同区域的情况。而且 max-pool 是 permutation-invariant 的，view 顺序不影响结果，view 数量也不受限。

### 3. 长视频怎么办

Video diffusion model 一次只能生成 14 帧。要生成几百帧的长视频，就得 autoregressive 一段一段来。

但问题是：第二段怎么知道第一段画了啥？

GEN3C 的做法：每生成一段，把最后一帧的 depth 估出来，scale align 到现有 cache 上，unproject 成点云，append 到 cache。下一段 render 的时候就能看到之前生成的内容。

这是一个"生成→观测→记忆→再生成"的循环。Cache 就是一个 external memory，而且是 3D 的，所以 camera 怎么转都能看到一致的内容。

---

## 为什么这个思路 work

几个关键原因：

**Depth estimator 够好了**。Depth Anything V2 在各种 domain 上都很 robust，不需要 perfect，approximate 就够了。论文做了实验，加 30% noise 到 depth 上，PSNR 还能撑住 18+。NN 有很强的 correct geometric error 的能力。

**Video diffusion prior 够强了**。SVD 在互联网视频上训练过，见过大量场景。你给它一个 rough render，它能 generate 出 plausible 的细节。后面换成 Cosmos（更强的 model），效果立刻提升，证明这个 framework 是 model-agnostic 的。

**不做 explicit fusion 是对的**。让 NN 在 feature level 处理 multi-view inconsistency，比在 3D space 做 alignment robust 得多。NN 天然能处理 lighting difference、depth misalignment 这些东西。

---

## 效果怎么样

几个 highlight：

- **Single image → video**：比 CameraCtrl 高 3-4 dB PSNR，OOD generalization 好很多（因为不依赖 camera pose → video 的隐式 mapping）
- **Two-view NVS extrapolation**：比 MVSplat 高 6 dB，因为 generative prior 能 hallucinate unseen regions
- **Driving simulation**：偏离原始轨迹 4 米，FID 只有 35，而 3DGS 已经 81 了。Reconstruction 方法在 sparse observation 下根本撑不住
- **3D editing**：因为 cache 是 explicit point cloud，你可以直接删掉一辆车、改一辆车的轨迹，再让 model re-render。这个是 latent-based method 做不到的

---

## 我的直觉

这篇 paper 的核心 message 其实挺深的：

**Generative model 和 graphics pipeline 不应该是对立的，应该 hybrid**。

Graphics 有精确的 geometry 和 control，但缺 photorealism 和 generative power。Generative model 有 photorealism，但缺 control 和 consistency。GEN3C 的做法是让 graphics 提供 skeleton（3D cache + camera control），让 generative model 提供 muscle（fill content、enhance quality）。

这个 pattern 其实在很多地方都适用：你有一个 approximate 的 structured representation，用 generative model 去 refine 和 complete 它。比如 code generation 里，你可以先给一个 rough skeleton，让 LLM fill in。一样的 idea。

而且这个 framework 的美妙之处在于它是 **future-proof** 的——video diffusion model 越强，GEN3C 就越强。你不需要重新设计架构，换一个更好的 base model fine-tune 一下就行。Cosmos 那个实验已经证明了这一点。

---

## 局限

- 动态场景需要 pre-generated video 提供物体运动，不能从 text prompt 直接生成运动
- Point cloud 没有 view-dependent appearance modeling，反光、透明物体可能有问题
- Autoregressive 长期生成可能有 error accumulation，论文没展示几百帧的 stability
- 没有 uncertainty modeling，cache 里哪些区域 depth 不确定，NN 不知道

但这些都是 future work 的方向，不影响这个 framework 本身的 value。

---

参考：
- Project page: https://research.nvidia.com/labs/toronto-ai/GEN3C/
- SVD: https://arxiv.org/abs/2311.15127
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- Cosmos: https://arxiv.org/abs/2501.03575

---

# GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control 深度解析

Andrej，这篇来自 NVIDIA Toronto 的 GEN3C 是一个我个人觉得非常有意思的工作，因为它把 graphics pipeline 里的 "explicit 3D geometry" 思想重新注入到 video diffusion model 里，绕开了一个很多人都在用但效果不好的范式（把 camera pose 当成 NN 的 conditioning input）。下面我尽量 build 你的 intuition。

---

## 1. 核心问题与 Motivation 的 Intuition

先讲清楚 prior work 在做什么、为什么不行：

**Prior paradigm（CameraCtrl / MotionCtrl / Camera Pose Conditioning）**：
把 Plücker embedding 或 camera parameters 直接 concat 到 video diffusion model 的 conditioning 上，fine-tune 让 NN "learn" 相机→视频的映射。问题在于：
- NN 必须从 camera pose **推断** scene structure，这是一个高度 underdetermined 的 mapping
- 不同的 scene layout 对同一个 camera trajectory 会产生完全不同的视频
- 当 camera 来回 sweep（look back and forth）时，NN 会 "忘记" 之前生成的内容，因为没有 explicit memory of history
- Generalization 差：训练数据里没见过的 camera motion 会崩

**GEN3C 的核心 insight**：
不要让 NN 做 "camera → video" 的隐式推理。用一个 **explicit 3D cache**（点云）作为中间表示：
1. NN 不需要记住 "之前生成了什么" → 这些信息都在 cache 里
2. NN 不需要从 camera 推断 image structure → cache 渲染出来的 2D image 已经把 structure 给出来了
3. NN 只需要做两件事：(a) **inpaint** disocclusion / unobserved regions，(b) **advance scene state**（动态场景）

这是一个非常 "graphics-friendly" 的思路：把 generation 和 geometry 解耦。NN 的 generative power 集中在 "hallucinate missing content" 上，而不是 "reason about 3D structure"。

---

## 2. 方法架构详解

### 2.1 Spatiotemporal 3D Cache 构建

**输入**：可以是 single image、sparse multi-view images、或 dynamic video(s)。

**Cache 结构**：一个 $L \times V$ 的点云数组，其中 $L$ 是 temporal length，$V$ 是 camera view 数量。每个元素 $\mathbf{P}^{t,v}$ 是从一张 RGB image 通过 depth estimation + unprojection 得到的 colored point cloud。

**关键设计选择**：
- 用 **depth estimation (DAV2, Depth Anything V2)** 而不是 structure-from-motion 来构建 cache。这是因为 DAV2 已经在多种 domain（indoor/outdoor/driving）上 robust
- 不做 explicit multi-view point cloud fusion（这是和 ReconX、ViewCrafter 的关键区别），而是 **per-view maintain separate cache**，让 NN 来处理 misalignment

**为什么不做 explicit fusion**？这是论文 Section 4.3 Discussion 的核心论点：
- Explicit fusion 依赖 DUSt3R 之类的 alignment，对 thin structures 不 robust
- Fused cache 会丢失 view-dependent lighting information
- Misalignment 会在 fused point cloud 里产生 ghosting artifacts

### 2.2 Rendering 3D Cache

渲染函数 $\mathcal{R}$ 把点云 $\mathbf{P}^{t,v}$ 投影到新相机 $\mathbf{C}^t$ 下：

$$
(I^{t,v}, M^{t,v}) := \mathcal{R}(\mathbf{P}^{t,v}, \mathbf{C}^t)
$$

- $I^{t,v} \in \mathbb{R}^{3 \times H \times W}$：rendered RGB image
- $M^{t,v} \in \mathbb{R}^{1 \times H \times W}$：coverage mask，标记哪些 pixel 在 cache 里没有覆盖（disocclusion / unobserved regions）

对于相机轨迹序列 $\mathbf{C} = (\mathbf{C}^1, \ldots, \mathbf{C}^L)$，对每个 view $v$ 渲染得到视频 $\mathbf{I}^v \in \mathbb{R}^{L \times 3 \times H \times W}$ 和 mask 视频 $\mathbf{M}^v \in \mathbb{R}^{L \times 1 \times H \times W}$。

**直觉**：这个 rendering 就像 graphics pipeline 里的 rasterization，但用 point cloud 而不是 mesh。Point cloud rendering 的好处是 trivially parallel，不需要 mesh reconstruction。

### 2.3 Fusion and Injection（最关键的设计）

这部分是论文最 interesting 的技术贡献。让我详细讲：

#### 2.3.1 基础：Video Diffusion Model 背景

GEN3C 基于 Stable Video Diffusion (SVD)，是一个 latent diffusion model。给定 RGB video $\mathbf{x} \in \mathbb{R}^{L \times 3 \times H \times W}$，VAE encoder $\mathcal{E}$ 压缩到 latent space：

$$
\mathbf{z} = \mathcal{E}(\mathbf{x}) \in \mathbb{R}^{L \times 4 \times h \times w}
$$

其中 SVD 只做 spatial compression（不做 temporal compression），所以 $h = H/8, w = W/8$，channel 数 $C = 4$。

Denoising 的训练 objective（Eqn. 1）：

$$
\mathbb{E}_{\mathbf{x}_0 \sim p_{\text{data}}, \tau \sim p_\tau, \epsilon \sim \mathcal{N}(0, I)} \left[ \| \mathbf{f}_\theta(\mathbf{x}_\tau; \mathbf{c}, \tau) - \mathbf{y} \|_2^2 \right]
$$

变量解释：
- $\mathbf{x}_0$：clean data sample
- $\tau$：diffusion timestep，从 $p_\tau$ 采样
- $\alpha_\tau, \sigma_\tau$：noise schedule 参数，控制 signal/noise ratio
- $\epsilon$：标准 Gaussian noise
- $\mathbf{x}_\tau = \alpha_\tau \mathbf{x}_0 + \sigma_\tau \epsilon$：noisy version
- $\mathbf{f}_\theta$：denoising network
- $\mathbf{c}$：condition（在 GEN3C 里就是 rendered cache）
- $\mathbf{y}$：target，可以是 $\epsilon$（predict noise）、$\alpha_\tau \epsilon - \sigma_\tau \mathbf{x}_0$（velocity prediction）、或 $\mathbf{x}_0$（predict clean data）。SVD 用的是 $\mathbf{y} = \mathbf{z}_0 = \mathcal{E}(\mathbf{x})$

#### 2.3.2 Mask Injection 的设计哲学

这是论文里我觉得最 clever 的设计点之一。

**Naive approach**：把 mask concatenate 到 latent 上作为额外 channel。
- 问题：这会引入新的 trainable parameters（conv weights 需要适配新的 input channel 数）
- 这些新参数在 large-scale pre-training data 里没有对应的 representation
- **Generalization 会差**：driving scene 里 novel trajectory 的 mask pattern 和训练数据不一样时，模型不会处理

**GEN3C 的做法**：直接用 element-wise multiplication 把 mask 应用到 latent：

$$
\mathbf{z}^{v,\prime} = \text{In-Layer}(\text{Concat}(\mathbf{z}^v \odot \mathbf{M}^{v,\prime}, \mathbf{z}_\tau))
$$

- $\mathbf{z}^v = \mathcal{E}(\mathbf{I}^v)$：VAE encoded latent of rendered video for view $v$
- $\mathbf{M}^{v,\prime} \in \mathbb{R}^{L \times 1 \times h \times w}$：mask downsampled 到 latent resolution（用 min-pooling with size $\frac{H}{h} \times \frac{W}{w}$，即 $8 \times 8$）。min-pooling 而不是 avg-pooling 是因为：只要 $8 \times 8$ patch 里有任何 pixel 没覆盖，整个 patch 都应该被 mask 掉（conservative）
- $\odot$：element-wise multiplication，未覆盖区域 latent 直接置零
- $\text{Concat}$：和 noisy latent $\mathbf{z}_\tau$ 在 channel 维拼接（4 + 4 = 8 channels）
- $\text{In-Layer}$：diffusion model 的第一层 conv

**关键 intuition**：mask 直接作用在 latent 上，**不改变 model architecture**。这意味着所有 pre-trained weights 都还在发挥作用，model 看到的 input distribution 只是在 "未覆盖区域" 上变成了 zero latent。这相当于告诉 model："这些地方你需要 hallucinate"。

#### 2.3.3 Multi-View Fusion：Permutation-Invariant Max-Pooling

当 $V > 1$ 时，需要 fuse 多个 view 的信息。设计原则：**permutation-invariant**（view 顺序不应影响结果）。

$$
\mathbf{z}^\prime = \text{Max-Pool}\{\mathbf{z}^{1,\prime}, \ldots, \mathbf{z}^{V,\prime}\}
$$

- 每个 view 单独过 In-Layer，得到 feature map $\mathbf{z}^{v,\prime}$
- 在 view 维度做 max-pooling，得到 fused feature map $\mathbf{z}^\prime$
- 这个 $\mathbf{z}^\prime$ 再进入 diffusion model 的后续层

**为什么 max-pooling 而不是 average？**
- Max-pooling 是一个 "OR" 操作：只要任何一个 view 在某个 spatial position 提供了 feature，就保留它
- Average-pooling 会把不同 view 的 feature 混在一起，可能模糊化
- Max-pooling 天然处理 "不同 view 覆盖不同区域" 的情况

**对比其他 fusion 策略**（论文 Fig. 4 和 Table 6 的 ablation）：

| Strategy | Pros | Cons |
|----------|------|------|
| Explicit 3D Fusion (ReconX/ViewCrafter) | Simple | Relies on depth alignment, artifacts on misalignment, loses view-dependent lighting |
| Concat all latents | Works empirically | Bounded $V$, imposes order, not permutation-invariant |
| **Max-Pool (GEN3C)** | Permutation-invariant, unbounded $V$, lets NN handle aggregation | Slightly indirect |

Ablation 结果（Table 6, RE10K）：
- Explicit Fusion: PSNR 21.81 / 19.87 (interp / extrap)
- GEN3C: PSNR 24.08 / 21.56

差距 2-3 dB，主要来自 misalignment 场景。

### 2.4 Training

**Paired data curation 的 trick**：
- 真实多视角动态视频数据稀缺
- 用 **static real-world video**（RE10K, DL3DV, WOD）训练 spatial consistency
- 用 **synthetic multi-view dynamic video**（Kubric4D）训练 temporal consistency

具体做法：
- RE10K / DL3DV：从视频 clip 里随机选 $V \in [1,4]$ 个等间距 frame 作为 cache input，GT video 是包含其中一个 frame 的连续 $L$ 帧。这相当于让 model 学习 "从 sparse observation 预测中间或延伸的 frame"
- WOD：用 3 个 camera（front/left/right）的同步帧作为 cache（$V=3$），GT 只用 front camera 的序列。这逼迫 model 学习 resolve cross-camera inconsistency（depth scale, ISP, exposure）
- Kubric4D：合成数据，每个 scene 渲染 $V \in [1,4]$ 个 camera trajectory 作为 cache，再渲染一个不同 trajectory 作为 GT

**Progressive training**：
1. Stage 1: RE10K + DL3DV @ 320×576, 100K iter
2. Stage 2: All 4 datasets @ 576×1024, 100K iter
3. Stage 3: Finetune temporal layers @ 320×576, sequence length 14→56, 10K iter

**Condition dropout**：15% dropout on rendered cache + CLIP embedding，这是为了 classifier-free guidance。

### 2.5 Autoregressive Inference with 3D Cache Update

长视频生成的关键。分 chunk（长度 $L$，overlap 1 frame），autoregressive 生成。

**3D Cache 更新**：对每个生成的 chunk 的最后一帧，用 DAV2 预测 depth，然后 align 到现有 cache 的 scale：

$$
s, t = \arg\min_{s,t} \| (s \cdot \mathbf{d} + t - \mathbf{d}^{\text{tgt}}) \cdot M \|_2^2
$$

- $\mathbf{d}$：DAV2 预测的 depth（scale 未知）
- $\mathbf{d}^{\text{tgt}}$：从现有 3D cache 渲染到该 camera viewpoint 的 depth
- $M$：coverage mask（只对 cache 覆盖的 pixel 计算 reprojection error）
- $s, t$：global scaling and translation，用于 align $\mathbf{d}$ 到 cache 的 metric scale

**优化后**：

$$
\mathbf{d}^\prime = s \cdot \mathbf{d} + t
$$

将 $\mathbf{d}^\prime$ unproject 成 point cloud，append 到 3D cache。下一个 chunk 用更新后的 cache 渲染。

**Intuition**：这是一个 "生成→观测→记忆" 的循环。每个 chunk 生成后，把新视角的观测加入 cache，下一个 chunk 就能看到之前生成的内容。这解决了 "look back and forth" 时 content inconsistency 的问题。

---

## 3. 实验结果深度分析

### 3.1 Single View → Video (Table 1)

| Method | T&T (OOD) PSNR | RE10K (ID) PSNR | TSED |
|--------|---------------|-----------------|------|
| MotionCtrl | 13.46 | 13.60 | 0.1363 |
| CameraCtrl | 15.88 | 18.40 | 0.8033 |
| GenWarp | 16.04 | 15.50 | 0.0330 |
| NVS-Solver | 16.95 | 16.90 | 0.7286 |
| **GEN3C** | **18.66** | **19.88** | **0.9143** |

**关键观察**：
- CameraCtrl 在 ID (RE10K) 上 PSNR 18.40 还行，但在 OOD (T&T) 上掉到 15.88（-2.5 dB）。这是 Plücker embedding 方法的通病：camera parameter → video 的映射学不到 generalizable structure reasoning
- GEN3C 只掉 1.2 dB（18.66 vs 19.88），因为 3D cache 是显式 geometry，不依赖 NN 推断 structure
- TSED（3D consistency metric）GEN3C 0.9143 远超其他方法，证明 cache 机制确实 enforce consistency

### 3.2 Two-View NVS (Table 2)

GEN3C 在 extrapolation 上优势尤其明显：
- RE10K extrapolation: GEN3C 21.56 vs MVSplat 15.51（+6 dB！）
- 这说明 pre-trained video prior 在 "extreme viewpoint" 下远超 feed-forward reconstruction 方法

### 3.3 Driving NVS (Table 3, FID)

| Method | y±0.0m | y±1.0m | y±2.0m | y±4.0m |
|--------|--------|--------|--------|--------|
| Nerfacto | 48.34 | 67.77 | 80.41 | 112.40 |
| 3DGS | 34.81 | 53.85 | 61.78 | 81.26 |
| **GEN3C** | **7.93** | **18.19** | **25.11** | **35.33** |

**Intuition**：reconstruction 方法在偏离原始 trajectory 时迅速恶化（FID 翻倍），因为 sparse observation 无法 cover novel viewpoint。GEN3C 靠 generative prior inpaint missing regions，FID 增长平缓。

### 3.4 Robustness to Noisy Depth (Table 5)

| Noise Ratio | PSNR (interp/extrap) |
|-------------|----------------------|
| 0% | 24.08 / 21.56 |
| 3% | 22.39 / 21.00 |
| 10% | 20.85 / 19.64 |
| 30% | 18.52 / 17.91 |

**关键 insight**：即使 30% noise（相对 depth range 的 30%），PSNR 还能维持在 18+。这说明 video diffusion model 有很强的 "denoising" 能力——它能 correct cache 里的几何误差。这验证了论文的核心 thesis：NN 不需要精确 geometry，只需要 approximate geometry 来 anchor generation。

### 3.5 Extending to Cosmos (Section 5.7)

把 base model 从 SVD 换成 Cosmos（NVIDIA 的更 advanced video diffusion model），同样的 fine-tuning protocol，质量显著提升。这证明 GEN3C 的 framework 是 **base-model-agnostic** 的：只要 video diffusion model 更强，GEN3C 就能 generate 更好的结果。这是一个很有价值的 property，因为 video generation model 还在快速进化。

---

## 4. 与相关工作的 positioning

### vs. ReconFusion / CAT3D
- 需要per-scene optimization，慢
- GEN3C 是 feed-forward，30s 生成 14-frame video on single A100

### vs. ReconX / ViewCrafter
- 用 DUSt3R 做 explicit multi-view alignment，对 thin structures fragile
- GEN3C 让 NN 处理 misalignment，更 robust

### vs. MultiDiff
- 只支持 single view
- GEN3C 支持 arbitrary number of views（permutation-invariant fusion）

### vs. StreamingT2V
- 用 latent feature map 作为 history，camera control 难
- GEN3C 用 explicit 3D cache，camera control 精确

### vs. CVD (Collaborative Video Diffusion)
- 同步 frames 之间 consistent，但 content 离开 view 后就 lost
- GEN3C 的 3D cache 是 persistent memory

### vs. Streetscapes
- 需要精确 height map
- GEN3C 用 depth estimation，更 general

---

## 5. Limitations 与未来方向

论文自己提到的：dynamic content 依赖 pre-generated video 提供物体运动。生成这样的 video 本身是个 challenge。Promising extension：incorporate text conditioning 来 prompt motion。

我自己的思考：
- **3D cache 是 point cloud**，没有 view-dependent appearance modeling（不像 NeRF 的 radiance field）。这对 specular / reflective surface 可能有问题
- **Depth estimation 的 metric scale** 需要和 camera pose 一致。论文用 DROID-SLAM 的 scale，但 SLAM 本身可能 drift
- **Autoregressive 的 error accumulation**：虽然 cache update 用 reprojection error alignment，但长期生成可能 still drift。论文没讨论很长的 video（几百帧）的 stability
- **没有 explicit uncertainty modeling**：cache 里的 confidence 没有传递给 NN。如果某个 region 的 depth 不确定，NN 应该知道。这可能是未来的改进方向

---

## 6. 我的 take-away

GEN3C 的核心贡献是一个 **architectural philosophy**：在 generative model 里注入 explicit 3D structure，让 NN 专注于 generation 而不是 reasoning。这个 idea 不新（Mallya et al. 2020 的 vid2vid 就用过类似 point cloud conditioning），但 GEN3C 把它做对了：

1. **不做 explicit fusion**，让 NN 处理 multi-view inconsistency
2. **Mask 通过 element-wise multiplication 注入**，不引入新参数
3. **Permutation-invariant max-pooling fusion**，支持任意 view 数量
4. **Autoregressive cache update**，支持长视频
5. **Base-model-agnostic**，能 leverage 不断进化的 video diffusion model

这个 framework 的 generality 很强：single image、sparse views、driving、dynamic video 都能 cover。而且随着 video generation model 进步（SVD → Cosmos → future models），GEN3C 的效果会自然提升。这是一个 "搭便车" 的 design，很聪明。

参考链接：
- 论文网页：https://research.nvidia.com/labs/toronto-ai/GEN3C/
- Stable Video Diffusion: https://arxiv.org/abs/2311.15127
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- DROID-SLAM: https://proceedings.neurips.cc/paper/2021/hash/d902c3e186c9e64c0c4b8159a3b8276c-Abstract.html
- Cosmos: https://arxiv.org/abs/2501.03575
- CameraCtrl: https://arxiv.org/abs/2404.02101
- ViewCrafter: https://arxiv.org/abs/2409.02048
- ReconX: https://arxiv.org/abs/2408.16767
- DUSt3R: https://arxiv.org/abs/2312.14132
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Nerfstudio: https://docs.nerf.studio/
- Kubric: https://github.com/google-research/kubric
- RE10K (Stereo Magnification): https://people.eecs.berkeley.edu/~tinghuiz/projects/mpi/
- DL3DV: https://github.com/DL3DV-10K/Dataset
- Waymo Open Dataset: https://waymo.com/open/
- CAT3D: https://arxiv.org/abs/2405.10314
- ReconFusion: https://arxiv.org/abs/2402.10863
- GCD (Generative Camera Dolly): https://gdll.github.io/
- MultiDiff: https://github.com/norman-mu/MultiDiff
- StreamingT2V: https://arxiv.org/abs/2403.14773
- CVD: https://arxiv.org/abs/2405.17414
