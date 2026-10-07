---
source_pdf: OmniRoam- World Wandering via Long-Horizon Panoramic Video.pdf
paper_sha256: faae9df70f56a5855f8c306ce753b84dd70acd7546d1d59c6ede7afe352ccbfc
processed_at: '2026-08-05T23:28:23-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 OmniRoam

## 一句话讲清楚

你给一张 360° 全景图，再画一条 camera 走的路线，它给你生成一段让你在场景里"漫游"的长视频。难点是怎么让这段长视频不"漂"——走一圈回来还能看到原来的样子。

## 问题的根源在哪

普通 video generation model 都是 perspective view，也就是人眼的 narrow FoV。这有个先天缺陷：每帧只看到 scene 的一个角落。你想让 model 生成一段 600 帧的漫游视频，它每帧只看到局部，又得保持整段视频的 global layout 一致，怎么办？

现有方法基本靠 **autoregressive**——生成第 1 段，喂给第 2 段当 condition，再生成第 3 段……问题就在这：每段都有点小误差，误差累积起来，到第 600 帧整个 scene 已经面目全非。论文里 autoregressive baseline 的 loop consistency 只有 0.89（小于 1，意思是结尾比起始帧还不像起始帧），drift 严重到崩盘。

## Panoramic 为什么天然就赢

这里有个非常 elegant 的 insight。ERP panorama 每一帧就是 360° 全场景，**每一帧本身就是 global memory**。你不需要"累积"全局信息，每帧都自带。

用 LLM 的类比讲更直观：
- Perspective video 像 context window 只有 50 token 的 LLM，要靠 RAG 不断 retrieval 才能 maintain long context
- Panoramic video 像 context window 有 100k token 的 LLM，所有 context 天然在视野里

所以 OmniRoam 的核心 claim：在 long-horizon scene generation 这个 task 上，**representation 选择比 architecture 更重要**。同样的 Wan2.1 backbone，换成 panoramic representation，loop consistency 直接从 1.42 跳到 1.96。这个 gap 是 representation 带来的，不是 model 大小或训练 trick 带来的。

## 两阶段：Preview → Refine

为什么不直接生成 720p × 641 frames 的高质量全景视频？因为 attention complexity 是 O((f·h·w)²)，长视频高分辨率单 pass 在算力上做不到。

OmniRoam 用 **global-to-local** 策略：

### Preview Stage
- 输入：一张全景图 + camera trajectory
- 输出：480×960，81 frames，但 playback speed 加速 8 倍
- 直觉：相当于"快进模式"浏览整个 scene 的 layout，确认 trajectory 走得对、scene layout 合理
- 用户可以多生成几次（diffusion 的 stochasticity），挑一个最喜欢的 layout

### Refine Stage
- 输入：preview 视频 + target scale (一般 1.0 = 正常速度)
- 输出：720×1440，641 frames
- 直觉：把快进的 preview "慢放"到正常速度，同时把分辨率提升

关键 trick：refine stage **不再需要 trajectory condition**，只需要一个 scalar (scale)。因为 preview 已经 encode 了 trajectory 信息，refine 只做"时间插值 + 空间超分"。这极大简化了 refine model 的学习目标。

类比：preview 像画素描，确定构图和大致 layout；refine 像上色加细节。两阶段分离让每个 stage 的学习任务都变简单。

## Flow + Scale 分解的妙处

这是 paper 最有 creativity 的设计。传统方法 (Matrix-3D) 把 6-DoF camera pose 全塞给 model。但 panoramic 有特殊性：

**ERP 中 rotation = cyclic pixel shift**，不需要 3D 几何意义下的 rotation learning。所以 OmniRoam 直接把 rotation 去掉，camera 假设始终 face forward，只做 translation。

进一步，translation 又被分解为：
- **Flow** $\mathcal{D}$：每帧一个 3D 单位向量，表示运动方向
- **Scale** $s$：一个 global scalar，表示每帧走多远

为什么这样分解？

1. **Intuitive control**：用户想"快进"就调大 scale，想"慢走"就调小 scale，trajectory 方向不变。preview 阶段 scale=8 快速浏览，refine 阶段 scale=1 正常速度。

2. **Refine stage 只需调 scale**：preview 和 refine 用同样的 trajectory，只是 scale 不同。所以 refine model 不需要重新 encode trajectory，只需一个 scalar embedding。

3. **Log-space scale embedding**：scale 范围 [1.1, 8.0]，跨度大。用 $\log(s)$ 后变成 [0.1, 2.0]，linear projector 就能 handle。

4. **Zero-init camera encoder**：flow encoder 用 zero initialization，训练初期不影响 pretrained Wan 的 visual prior，逐渐 learn 到使用 trajectory signal。这是 ControlNet 的经典 trick。

## Loop Consistency Metric 的妙处

现有 metric (FVD, FID) 衡量 distribution distance，但无法 capture "long-term global consistency" 这个概念。OmniRoam 提了个非常 elegant 的 metric：

让 model 沿一条 **closed loop trajectory** 生成视频，然后看：
- **$S_1$ (Loop Closure Score)**：开头 5 帧和结尾 5 帧的 CLIP similarity。理想情况下应该高（回到了起点，看到一样的东西）
- **$S_2$ (Intermediate Score)**：开头 5 帧和中间所有帧的平均 CLIP similarity。理想情况下应该低（中间真的探索了不同地方，不是停在原地）

$$C_{\text{loop}} = \frac{1 - S_2}{1 - S_1}$$

妙处在于分式结构同时 reward 两个 property：
- 分子 $1 - S_2$ 大：中间探索充分，scene 真的变了
- 分母 $1 - S_1$ 小：结尾回到起点，scene 又变回来了

避免了 trivial solution：如果 model 生成静态视频（不动），$S_1 = S_2 = 1$，metric 变成 $\frac{0}{0}$，undefined。所以必须真的动了且真的回来了才能拿高分。

这个 metric 概念上类似 **dynamical system 的 Poincaré recurrence**——well-behaved 的 generative world model 应该具有 returnability。这个 idea 可以推广到更广的 world model evaluation。

## 实验告诉我们什么

Table 2 是最有 informative 的 ablation：

| Setting | Loop Consistency |
|---|---|
| Perspective + Refine (641 frames) | 1.42 |
| Panoramic + Autoregressive (641 frames) | 0.89 |
| Panoramic + Two-stage (Ours, 641 frames) | 1.96 |

两个独立 dimension 的 ablation：

**Representation 维度**：同样 two-stage design，panoramic 1.96 vs perspective 1.42。Panoramic representation 本身贡献了 ~38% 的 loop consistency 提升。

**Strategy 维度**：同样 panoramic representation，two-stage 1.96 vs autoregressive 0.89。Two-stage design 贡献了 ~120% 的提升。Autoregressive 的 0.89 (<1) 说明它彻底崩了——结尾比起始帧还不像起始帧。

两个 dimension 都重要，但 strategy (两阶段 vs autoregressive) 的 gap 更大。这说明 long-horizon generation 中，**error accumulation 是比 representation 更 fundamental 的问题**，但 panoramic representation 提供了解决这个问题的 foundation（因为 preview 才能 serve as global anchor）。

## CLIP Similarity Curve 的故事

Figure 5 把 loop consistency 的 temporal dynamics 画出来特别直观：

- **Ours**：similarity 从 1.0 开始下降（camera 远离起点），到 loop 中点最低，然后回升，最终接近 1.0。漂亮的 U 型曲线。
- **Autoregressive**：similarity 单调下降，永远不回来。Drift 不可逆。
- **Perspective**：有 slight recovery 但 final frames 与起点 object geometry 差异大，curve 回升但回不到接近 1.0。

这条 curve 直接 visualize 了 "global memory" 的概念。Panoramic + two-stage 让 model 在 long-horizon 下仍然 maintain "回到原点" 的能力，autoregressive 在 long-horizon 下 fundamental 地 lose 这个能力。

## Extensions 的 intuition

### Real-time Preview via Self-Forcing

Self-Forcing 的核心 idea：训练 student model 时，让它看到 **自己的 rollout 错误**，而不是 teacher 的 perfect context。这样 student 在 inference 时遇到自己的错误输出不会崩。

类比：teacher 示范 perfect solution，student 只是模仿，inference 时遇到自己产生的 intermediate state 就 OOD。Self-Forcing 让 student 在训练时就面对自己的错误 distribution，bridge train-test gap。

效果：81 frames 从 5 分钟 → 7 秒，100× 加速。质量略降但 preserve 整体 structure。

### 3DGS Reconstruction

生成的 641 frames panoramic video 抽 100 帧，每帧 crop 5 个 perspective views (120° FoV)，喂给 3DGS reconstruction。

Intuition：如果生成的 panoramic video 真的 multi-view consistent，从不同 perspective crop 出来的 view 应该能重建出 coherent 3D scene。这是对生成 video 的 multi-view consistency 的 ultimate test。

这个 extension 把 video generation → 3D scene generation 的 pipeline 串起来。未来如果 panoramic video generation 质量足够高，可以直接替代 photogrammetry/NeRF/3DGS 的 data capture 步骤。

## 大局观：这篇 paper 的位置

OmniRoam 在一个更宽的 research trend 中：

1. **Generative model 从 2D → 3D → scene-level**：image → video → panoramic video → 3D scene。每一步都是 representation 升级，带来新的 capability。

2. **Long-horizon consistency 是核心瓶颈**：无论 LLM 还是 video model，long context 下 maintain consistency 都是 fundamental problem。Autoregressive 累积误差是 universal issue。

3. **Representation 作为 implicit memory**：panoramic 是 spatial dimension 的 implicit memory，类似 transformer attention 是 sequence dimension 的 implicit memory。选择合适的 representation 可以把 explicit accumulation 变成 implicit access。

4. **Two-stage / hierarchical generation 是 universal pattern**：从 DALL-E 的 prior + decoder，到 coarse-to-fine image generation，到 OmniRoam 的 preview + refine。Hierarchical decomposition 让每一步的学习任务变简单，是 scaling long-horizon generation 的 robust pattern。

5. **Evaluation metric 驱动 research**：loop consistency 这个 metric 让 "long-term global consistency" 从模糊概念变成 measurable quantity。这种 metric innovation 往往比 method innovation 影响更深远。

## 最核心的 takeaway

如果只记一句话：**在 long-horizon scene generation 中，panoramic representation 提供了 implicit global memory，配合 two-stage preview-refine strategy 避免 autoregressive error accumulation，是当前最 robust 的方案。**

如果记三句话：
1. Panoramic 每帧自带 global context，不需要累积就 maintain consistency
2. Two-stage preview-refine 用 preview 作为 global anchor，refine 只做局部插值，避免 drift
3. Loop consistency metric 把 "long-term consistency" 变成可量化指标，揭示 autoregressive 的 fundamental 局限

这就是 OmniRoam 的核心故事。

---

# OmniRoam 深度解析

## 一、核心 Motivation 与 Insight

OmniRoam 要解决的核心 problem 是 **long-horizon scene wandering**——给定一张 panoramic image 和一条 camera trajectory，生成一段长的、世界一致的视频。现有的 perspective video generation methods 有一个根本性的 structural deficiency：每帧只能观察到 scene 的一个 narrow FoV，要在长时间生成中维持 global consistency 就必须靠 temporal accumulation，而累积必然引入 error drift。这就是为什么 Sora、Wan、CogVideoX 这类 model 在长视频生成时会出现 object 重新出现、texture 漂移、geometry 崩塌等问题。

OmniRoam 的核心 insight：**panoramic representation 本身就是一个 implicit global memory**。每一帧的 ERP (equirectangular projection) 全景图天然包含 360° scene context，无需累积就能 maintain global spatial layout。把生成任务放在 panoramic space，consistency 问题就从 "how to accumulate" 变成 "how to preserve"——后者在 generative model 中天然 easier，因为每帧都看到 global context。

这其实呼应了 NeRF/3DGS vs. perspective image 的差异：3D representation 是 explicit global memory，panoramic 是 implicit global memory，perspective 是 local memory requiring accumulation。

GitHub repo: https://github.com/yuhengliu02/OmniRoam

---

## 二、整体 Architecture：Two-Stage Preview–Refine

### 2.1 为什么需要 Two-Stage？

直接生成一段 641 frames、720×1440 的 panoramic video 在 computation 上几乎不可行——latent diffusion model 的 attention complexity 是 O((f·h·w)²) 量级。论文采用 **global-to-local** 策略：

- **Preview stage**：生成 480×960、81 frames、accelerated speed (scale s ∈ [1.1, 8.0]) 的粗略 panoramic 视频，快速 traverse 整个 scene
- **Refine stage**：把 preview 视频做 temporal expansion (factor = s/s') + spatial upscaling 到 720×1440、641 frames、normal speed (s' = 1.0)

### 2.2 Preview Stage 细节

#### Frame-Dimension Video Conditioning

借鉴 ReCamMaster (https://arxiv.org/abs/2503.11647)，把 source video 和 target video 在 VAE latent space 中沿 **frame dimension** concat，确保 strict visual continuity。

公式 (1):
$$z_{in} = \mathscr{E}_v(V_{in}), \quad z_v = \mathscr{E}_v(V_v), \quad z_{in}, z_v \in \mathbb{R}^{f' \times c \times h \times w}$$

- $V_{in} \in \mathbb{R}^{f \times 3 \times H \times W}$: input video/image
- $V_v \in \mathbb{R}^{\hat{f} \times 3 \times H \times W}$: target video
- $\mathscr{E}_v$: 3D VAE encoder (from Wan, https://arxiv.org/abs/2503.20314)
- $f'$: latent temporal length (因 VAE temporal downsampling factor，比如 4×，所以 81 frames → ~21 latent frames)
- $c$: latent channel (Wan 是 16)
- $h, w$: latent spatial dims

这种 frame-wise concat 的好处：source frames 保持 clean (no noise injection)，作为 strong condition 保证 visual continuity；target frames 接受 noise injection 进行 diffusion。

#### Decomposed Trajectory Conditioning — 这是论文最有意思的设计

传统做法 (Matrix-3D, https://arxiv.org/abs/2508.08086) 直接把 6-DoF camera pose 作为 condition，但 panoramic 有特殊性：ERP 中 camera rotation 对应 **cyclic pixel shift**，不是 perspective 中的 occlusion/parallax。所以论文做了两个 simplification：

**两个 key assumptions**:
- (a) **uniform velocity**: 沿 trajectory 匀速运动
- (b) **fixed orientation**: camera orientation 保持不变 (no roll/pitch/yaw)

在这两个 assumptions 下，trajectory 被分解为两个 orthogonal components:

**Scale** $s \in \mathbb{R}^+$: global scalar，每 timestep 的位移 magnitude。用 log-space 表示来处理 wide range 速度：

公式 (2):
$$z_s = \phi(\log(s))$$

- $s$: scalar scale
- $\phi \in \mathbb{R} \to \mathbb{R}^{c_s}$: learnable linear projector，输出 channel size $c_s$
- $z_s$: scale embedding，被全局注入到所有 transformer blocks 中，uniformly modulate 所有 temporal tokens (类似 time embedding)

**Flow** $\mathcal{D} = \{\mathbf{d}_k\}_{k=1}^f$: 一组 3D 单位向量，每帧一个，表示 normalized direction of camera displacement。

公式 (3):
$$z_d = \mathcal{E}_c(\mathcal{D})$$

- $\mathbf{d}_k$: 第 k 帧的 3D direction vector, $\|\mathbf{d}_k\| = 1$
- $\mathcal{E}_c$: zero-initialized camera encoder (一个 FC layer，插到每个 transformer block)
- $z_d \in \mathbb{R}^{f' \times c_d}$: per-frame flow embedding

为什么 zero-init？这是 controlnet 的经典 trick (https://arxiv.org/abs/2105.05230)，初始化时 control signal 不影响 pretrained model 的 output，训练时逐渐 learn 到使用 control signal，避免 catastrophic forgetting of pretrained visual prior。

#### Training Objective

基于 Wan 的 **Rectified Flow** framework (而不是 vanilla DDPM):

公式 (4):
$$\mathcal{L}_{\text{preview}} = \mathbb{E}_{t, z_0, \epsilon} \| v_\Theta([z_{in}, z_t]_{\text{frame}}, t, \mathbf{c}) - (z_v - \epsilon) \|_2^2$$

- $z_t = t \cdot z_v + (1-t) \cdot \epsilon$: forward process interpolation
- $t \in [0, 1]$: flow time
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: Gaussian noise
- $v_\Theta$: neural network 预测的 velocity field
- $[\cdot]_{\text{frame}}$: frame-wise concatenation (沿 temporal dim)
- $\mathbf{c} = \{z_d, \tilde{z}_s\}$: conditioning set，$\tilde{z}_s$ 是 scale embedding 经过 temporal duplication 到 f' frames

Rectified flow vs. DDPM: rectified flow 用 linear interpolation path，让 flow trajectory 尽量 "straight"，使得 ODE 求解时 fewer steps 就能 high quality (Liu et al., https://arxiv.org/abs/2209.03003)。Wan 用的就是这个 framework。

### 2.3 Refine Stage 细节

Preview stage 输出是 accelerated (scale s 大)，比如 s=8.0 意味着每帧移动距离是 normal 的 8 倍。Refine stage 要做的是：把这段 compressed 视频 temporal expand 到 s/s' 倍长度，同时 spatial upscale 到 720p。

#### Scale Alignment + Visibility Mask

公式 (5):
$$m_j^{(i)} = \begin{cases} 1 & \text{if } j_0^{(i)} \leq j < j_0^{(i)} + w \\ 0 & \text{otherwise} \end{cases}$$

- $i \in \{1, \ldots, n\}$: segment index
- $n = \lceil s/s' \rceil$: 需要的 segment 数量
- $w = \lceil (f-1)/n \rceil$: window size
- $j_0^{(i)} = (i-1) \cdot w$: 第 i 个 segment 的起始 frame index (inference 时)
- training 时 $j_0 \sim \mathcal{U}[0, f-w]$ 随机采样，增加 generalization

公式 (6):
$$\tilde{z}_p^{(i)} = z_p \odot \mathbf{m}'^{(i)}$$

- $z_p = \mathscr{E}_v(V_p) \in \mathbb{R}^{f' \times c \times h \times w}$: preview video 的 latent
- $\mathbf{m}'^{(i)} \in \mathbb{R}^{f'}$: mask 经过 average pooling downsample 到 latent temporal resolution
- $\odot$: element-wise multiplication，mask broadcast 到 channel 和 spatial dims
- $\tilde{z}_p^{(i)}$: masked latent conditioning，只保留对应 segment 的 preview frames

这里有个 subtle 之处：refine stage **不再需要 trajectory condition**，只调节 scale 这一个 scalar。因为 preview 已经 encode 了 trajectory 信息，refine 只需 "slow down" playback。

公式 (7) refine loss:
$$\mathcal{L}_{\text{refine}} = \mathbb{E}_{t, j_0, z_v^{(i)}, \epsilon} \| v_\Phi([\tilde{z}_p^{(i)}, z_t']_{\text{frame}}, t) - (z_v' - \epsilon) \|_2^2$$

- $z_t' = t \cdot z_v' + (1-t) \cdot \epsilon$
- $z_v'$: target video segment at scale $s'$
- $v_\Phi$: refine model 预测的 velocity field
- 注意这里 condition set 没有 $\mathbf{c}$，因为 refine 不需要 trajectory control

#### Visibility Mask 的 Intuition

这其实是一种 **sparse anchor conditioning**：preview 视频被均匀切成 n 段，每段用对应位置的 preview frame 作为 "anchor"，在 anchor 之间插值生成 high-res content。这种设计避免了 autoregressive 的 error accumulation，每个 segment 都直接 condition 到 preview global context。

---

## 三、Data Pipeline

### 3.1 Canonical Panoramic Coordinate System

这是支撑整个 framework 的基础。传统 6-DoF (XYZ translation + roll/pitch/yaw rotation) 在 panoramic 中冗余且有害：ERP 中 rotation 就是 cyclic shift，没必要 learn。所以论文设计 **rotation-invariant coordinate system**：

- gravity-align footage (消除 roll)
- eliminate camera self-rotation (yaw, pitch 在 post-processing 中 cyclic shift 掉)
- 只保留 translation $(x, y, z)$ relative to ERP center $(\phi=0, \theta=0)$

这样 trajectory space 从 6-DoF 降到 3-DoF，极大简化了 learning。

### 3.2 Hybrid Dataset

**Real-world data**: 
- 2,000 handheld panoramic videos
- 5M frames
- 场景：hotels, schools, outdoor landscapes
- 用 COLMAP (https://colmap.github.io/) 估计 trajectory
- 过滤 abnormal scale 数据，确保所有 video 大致同一 scale

**Synthetic data**:
- 1,000 3DGS scenes from InteriorGS (https://huggingface.co/datasets/spatialverse/InteriorGS)
- 自动 trajectory generation pipeline
- 定义 valid cruising area：camera vertical range 1.3m-1.5m
- candidate waypoints 覆盖 50% free space
- constant-speed trajectory，保证 scale 一致

Real data 提供 visual realism + scale，synthetic data 提供 precise geometry + diverse trajectories。两者互补。

---

## 四、Loop Consistency Metric

这是论文另一个重要 contribution。FVD、FID 这种传统 metric 无法 capture long-term global consistency。论文设计了一个 elegant metric：

公式 (8):
$$C_{\text{loop}} = \frac{1 - S_2}{1 - S_1}$$

公式 (9):
$$S_1 = \frac{1}{P^2} \sum_{q=1}^{P} \sum_{p=1}^{P} \text{Sim}(I_p, I_{f-q+1})$$

公式 (10):
$$S_2 = \frac{1}{P(f-2P)} \sum_{p=1}^{P} \sum_{q=P+1}^{f-P} \text{Sim}(I_p, I_q)$$

- $V = \{I_i\}_{i=1}^f$: generated sequence following a loop trajectory
- $\text{Sim}(\cdot, \cdot)$: CLIP embedding cosine similarity
- $S_1$: **Loop Closure Score**，衡量 first P frames 和 last P frames 的相似度
- $S_2$: **Intermediate Score**，衡量 first P frames 和中间 frames 的平均相似度
- $P = 5$: buffer frames (robust to minor temporal misalignment)

Intuition: 
- 理想情况：开头和结尾应该 high similarity ($S_1 \to 1$)，开头和中间应该 low similarity ($S_2 \to 0$)
- $C_{\text{loop}} = (1 - S_2)/(1 - S_1)$: 分子接近 1 (中间探索充分不同)，分母接近 0 (开头结尾相似)，所以 $C_{\text{loop}}$ 越大越好
- 这 metric 既 reward "回到起点"，又 reward "中间真的探索了不同地方"，避免 trivial solution (整个视频都不变)

---

## 五、Experiments

### 5.1 Evaluation Protocols

- 三方面: visual quality, trajectory controllability, loop consistency
- 两个 resolution: 480p (preview), 720p (refine)
- 七个 trajectory: forward, backward, left, right, s-curve, loop, GT
- 每 trajectory 每 method 生成 24 videos
- **None of the test trajectories 在训练中出现**——测试 generalization

### 5.2 Quantitative Results (Table 1)

| Method | Res | Frames | FAED↓ | SSIM↑ | LPIPS↓ | PSNR'25↑ | PSNR'55↑ | PSNR'75↑ | Loop↑ |
|---|---|---|---|---|---|---|---|---|---|
| Matrix-3D | 480p | 81 | 8.64 | 0.63 | 0.37 | 19.05 | 17.35 | 17.04 | 1.38 |
| Imagine360 | 480p | 81 | 23.35 | 0.33 | 0.61 | - | - | - | - |
| **Ours** | 480p | 81 | **5.27** | **0.70** | **0.18** | **23.06** | **20.69** | **19.87** | **2.34** |
| Matrix-3D | 720p | 81 | 9.76 | 0.66 | 0.36 | 19.06 | 17.63 | 17.12 | 1.41 |
| **Ours** | 720p | 641 | **5.07** | 0.66 | 0.33 | 19.75 | 18.59 | 18.24 | **1.96** |

观察:
- FAED (Fréchet Auto-Encoder Distance): 用 auto-encoder feature space 替代 InceptionV3，避免 ERP distortion artifacts 干扰
- PSNR 在三个 temporal windows 都明显领先 Matrix-3D，说明 trajectory following 准确
- Loop consistency 480p 是 2.34 vs Matrix-3D 的 1.38，**接近 2× 提升**
- 720p 长视频 (641 frames) 仍能 maintain 1.96 loop consistency，远好于 autoregressive baseline

### 5.3 Design Analysis (Table 2) — 这是最 informative 的实验

| Method | Frames | FAED↓ | SSIM↑ | LPIPS↓ | PSNR'25↑ | PSNR'55↑ | PSNR'615↑ | PSNR'635↑ | Loop↑ |
|---|---|---|---|---|---|---|---|---|---|
| Perspective (Preview) | 81 | 16.90 | 0.62 | 0.44 | 17.72 | 16.06 | - | - | 1.70 |
| Perspective (Refine) | 641 | 15.48 | 0.63 | 0.57 | 16.60 | 14.94 | 13.76 | 14.07 | 1.42 |
| Autoregressive | 641 | 16.04 | 0.33 | 0.64 | 15.44 | 14.84 | 10.14 | 10.11 | 0.89 |
| **Ours** | 641 | **7.70** | 0.58 | 0.44 | **19.75** | **18.59** | **15.55** | **15.63** | **1.96** |

关键 insight:

**Representation choice (panoramic vs perspective)**: 用同样的 model design，只是把 panoramic 换成 perspective (从 panorama crop 出来作为训练数据)，FAED 从 5.27 飙到 16.90 (preview) / 15.48 (refine)。Loop consistency 从 1.96 降到 1.42。这 quantitative 地证明了 panoramic representation 的 advantage。

**Generation strategy (global-to-local vs autoregressive)**: Autoregressive baseline 的 loop consistency 只有 0.89 (< 1!)，意味着生成的视频末尾**比中间帧还更不像起始帧**——发生了 catastrophic drift。PSNR'615 和 PSNR'635 都是 10.11/10.14，比 PSNR'25 的 15.44 还低，说明 error accumulation 严重。

**Long-horizon 性能衰减**: Ours 在 PSNR'25 是 19.75，到 PSNR'615 是 15.55，PSNR'635 是 15.63。decay 是 ~4 dB，相对温和。Perspective 衰减是 17.72 → 13.76，autoregressive 衰减是 15.44 → 10.14。Ours 的 long-horizon stability 是其他方法不可比的。

### 5.4 CLIP Similarity over Loop Trajectories (Figure 5)

这是 visualizing loop consistency 的 temporal dynamics:
- **Ours**: similarity 从 1.0 下降 (camera 远离起点)，到 loop 中点最低，然后逐渐回升，loop 关闭时 similarity 接近起点
- **Autoregressive**: monotonic decline，similarity 持续下降，drift 不可逆
- **Perspective**: 有 slight recovery 但 structural degradation 严重，final frames 与起点 object geometry/scene details 差异明显

---

## 六、Extensions

### 6.1 Real-time Preview via Self-Forcing

借鉴 Self-Forcing (Huang et al. 2025, https://arxiv.org/abs/2506.08009)，把 full preview model distill 成一个 lightweight autoregressive previewer:

$$\min \mathbb{E}_{\hat{\mathbf{x}} \sim p_{\hat{\theta}}, \mathbf{x} \sim p_\theta} [D(\hat{\mathbf{x}}_{1:T}, \mathbf{x}_{1:T})]$$

- $\hat{\theta}$: student (real-time previewer)
- $\theta$: teacher (full preview model)
- $D(\cdot)$: distribution matching loss (可以是 discriminative loss 或 DMD score)

效果: 81-frame panoramic video 从 ~5min 缩短到 7 秒，比 Matrix-3D (~11min) 快 100×。Self-Forcing 的核心 trick 是在训练时让 student 看到 **自己的 rollout 错误**，而不是 teacher 的 perfect context，从而 bridge train-test gap。

### 6.2 3DGS Reconstruction

- 从 641-frame generated video 中 uniformly 抽 100 frames
- 每 panoramic frame crop 5 个 perspective views (FoV 120°, 512×512)
- 用这些 perspective views 作为 3DGS (https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 的 input images
- 重建出 coherent 3D scene

这相当于把 panoramic video generation 转化为 3D scene generation 的 pipeline，类似 WorldPrompter (Zhang et al. 2025b, SIGGRAPH Asia 2025) 的思路。

---

## 七、Intuitive 总结

OmniRoam 的设计哲学可以归纳为几点：

1. **Representation matters more than architecture**: 同样的 Wan2.1-1.3B backbone，panoramic vs perspective 的 loop consistency 差 1.96 vs 1.42。这告诉我们在 generative model 时代，input/output representation 的选择仍然是 first-order design decision。

2. **Decompose control signals for simplicity**: 把 6-DoF trajectory 分解为 flow (3-DoF direction) + scale (1 scalar)，让 user control 变得 intuitive，且 refine stage 只需调一个 scalar。

3. **Global-to-local beats autoregressive for long-horizon**: 在 long video generation 中，autoregressive 的 error accumulation 是 fundamental problem。Two-stage preview-refine 用 "preview 作为 global memory anchor" 来 avoid drift，loop consistency 0.89 vs 1.96 的差距说明这 design 的 value。

4. **Panoramic = implicit global memory**: 每帧 ERP 都包含 360° scene，这意味着即使 short-clip generation 也 implicitly 携带 global context。这跟 LLM 中的 in-context learning 有类似 intuition——context 中的全部信息都 visible to model。

5. **Evaluation metric drives research direction**: Loop consistency 这个 metric 让 "long-term consistency" 从模糊概念变成 measurable quantity。未来 long-horizon video generation 工作都会受益于这种 metric。

---

## 八、潜在 Limitations 与 Future Directions

1. **Two assumptions (uniform velocity, fixed orientation)** 限制了表达力。比如 staircase climbing、抬头看 sky 这种 motion 无法表达。但这是 trade-off for simpler control。

2. **3DGS-based synthetic data** 可能 bias toward indoor scenes (InteriorGS dataset)，outdoor large-scale 可能 underrepresented。Real data 部分 2000 段可能不够 cover extreme scenes。

3. **Loop consistency metric** 假设 loop trajectory 是 closed 的，对 non-loop exploration 没有直接 measure。可以扩展到 "return-to-anchor" metric，让 anchor 不是 first frame 而是任意 reference frame。

4. **3DGS reconstruction** 假设 panoramic video 本身 multi-view consistent。但生成的 video 在 extreme angles 可能仍有 inconsistency，影响 3DGS 质量。可以引入 test-time optimization 或额外 consistency loss。

5. **Real-time preview quality** 略低于 full preview (Figure 7 vs Figure 4)，distillation loss 可能需要更多 iteration 或更好的 distillation strategy。

6. **Scale range s ∈ [1.1, 8.0]** 是离散的范围，s > 8 的 extreme fast traversal 可能 OOD。可以引入 continuous scale embedding 或 multi-scale training。

---

## 九、相关工作的 wider context

- **Wan2.1** (https://arxiv.org/abs/2503.20314): 阿里通义万相的视频生成 model，OmniRoam 的 base model。1.3B 参数，rectified flow framework。
- **Matrix-3D** (https://arxiv.org/abs/2508.08086): 同期工作，从 panoramic image 生成 3D explorable world，但用 single-stage generation，长视频能力弱。
- **Imagine360** (https://arxiv.org/abs/2412.03552): 从 perspective video 转 panoramic，本质是 video-to-panorama，没有 trajectory control。
- **ReCamMaster** (https://arxiv.org/abs/2503.11647): frame-dimension conditioning 的源头，OmniRoam 直接借鉴。
- **Self-Forcing** (https://arxiv.org/abs/2506.08009): 解决 autoregressive video diffusion train-test gap 的工作，OmniRoam 用于 real-time previewer distillation。
- **WorldPrompter** (Zhang et al. 2025b, SIGGRAPH Asia 2025): 类似 idea，用 panoramic 生成 + 3DGS reconstruction。
- **EvoWorld** (https://arxiv.org/abs/2510.01183): evolving panoramic world generation with explicit 3D memory，与 OmniRoam 思路相近但 memory mechanism 不同。

OmniRoam 在这条线上代表了 "panoramic + 两阶段 + decomposed control" 的具体 instantiation，关键 contribution 是把 long-horizon panoramic video generation 这件事 **make it work** 并 quantify 优势 via loop consistency metric。

---

## 十、个人思考 (build your intuition further)

从 Karpathy 的视角看，这篇 paper 有几个有意思的 connection:

1. **类似 LLM 的 "context window" vs "memory"**: Perspective video generation 像 LLM with limited context window，需要 RAG-like accumulation (autoregressive) 来 maintain long context。Panoramic video 像 LLM with large context window，每帧本身就看到 all context。这是 representation-as-memory 的另一个 instance。

2. **Two-stage = coarse-to-fine in planning**: 第一阶段做 "high-level planning" (preview 整个 trajectory 的 global layout)，第二阶段做 "low-level execution" (refine 每段细节)。这跟 RL 中的 hierarchical planning、LLM 中的 chain-of-thought vs answer、image generation 中的 coarse latent → high-res 都有类似 intuition。

3. **Decomposed control = disentangled representation**: Flow + scale 的分解类似 VAE 的 latent disentanglement，让 user 控制 granularity 提升。可以想象进一步 decompose: flow = direction + curvature，scale = base_speed + acceleration，但 trade-off 是 control 变复杂。

4. **Rectified flow + zero-init control**: 这种 "保留 pretrained prior + zero-init control signal" 的 pattern 在 controlnet、T2I-Adapter、IP-Adapter 中反复出现，是 generative model fine-tuning 的 robust recipe。

5. **Loop consistency = returnability of generative dynamics**: 这其实是 dynamical system 的 "recurrence" property 在 generative model 上的 measure。一个 well-behaved generative world model 应该具有 Poincaré-like return property——给定 closed trajectory，应该 return to原 state。这 metric 概念上可以扩展到更广的 world model evaluation。

总之，OmniRoam 是 panoramic video generation + long-horizon consistency + practical two-stage pipeline 的 careful integration。最有价值的设计是 decomposed trajectory conditioning 和 loop consistency metric，这两个 contribution 都可能被 future work 广泛 adopt。

参考链接汇总:
- OmniRoam GitHub: https://github.com/yuhengliu02/OmniRoam  
- Wan: https://arxiv.org/abs/2503.20314
- ReCamMaster: https://arxiv.org/abs/2503.11647
- Matrix-3D: https://arxiv.org/abs/2508.08086
- Imagine360: https://arxiv.org/abs/2412.03552
- Self-Forcing: https://arxiv.org/abs/2506.08009
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- COLMAP: https://colmap.github.io/
- InteriorGS: https://huggingface.co/datasets/spatialverse/InteriorGS
- ControlNet: https://arxiv.org/abs/2105.05230
- Rectified Flow: https://arxiv.org/abs/2209.03003
- EvoWorld: https://arxiv.org/abs/2510.01183
