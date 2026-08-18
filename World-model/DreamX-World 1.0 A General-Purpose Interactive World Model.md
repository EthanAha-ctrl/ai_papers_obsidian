---
source_pdf: DreamX-World 1.0 A General-Purpose Interactive World Model.pdf
paper_sha256: 8524f77fb6c10071f6098a285abf086b13f6a7a117ddcaf3470c2ca3d90bc122
processed_at: '2026-08-18T06:52:25-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 DreamX-World 1.0

## 一句话总结

把 Wan2.2 这个只会拍 5 秒 vlog 的 video diffusion 模型，改造成一个能实时交互、有空间记忆、能听懂多对象事件指令的"游戏引擎式 world model"，在 8 张 5090 上跑到 16 FPS。

---

## 为什么 video diffusion 直接拿来做 world model 不行

你拿 Wan2.2 生成 5 秒视频，效果很好。但你想让它当 world model 用——用户拿着 camera 转一圈、走回来、再触发几个事件——它马上挂掉，挂的方式有三类：

**挂法 1：camera 转回去，scene 变了**

bidirectional diffusion 每次 generation 都是独立采样 $p(x_t | \text{prompt}, x_{<t})$，它脑子里根本没有"这个房间刚才长什么样"这个 state。你 camera 转一圈回到原点，它给你重新 hallucinate 一个 plausible 的房间，可能墙的颜色都变了。

这跟 LLM 的 stateless 问题是一回事——没有 KV cache 的 LLM 每次回答都从零开始。video diffusion 缺的就是一个跨 chunk 持续的"scene state"。

**挂法 2：chunk 之间累积漂移**

你硬把它做成 autoregressive（chunk by chunk 生成），每个 chunk 的 prediction error（color tone、texture、identity）会累积。生成 30 秒后整个 scene 飘走，像 photocopier 复印复印件再复印那种退化。Self-Forcing（https://arxiv.org/abs/2506.08009）和 Causal Forcing（https://arxiv.org/abs/2602.02214）就是在修这个 train-test gap。

**挂法 3：distillation 之后能力退化**

要实时跑，必须把 50 步 diffusion 蒸馏到 few-step。但蒸馏一狠，motion 多样性、camera 跟随精度、visual quality 一起掉。你得在蒸馏后再想办法把能力找回来。

DreamX 的整个 pipeline 就是针对这三个挂法做的防御工事。

---

## 数据：为什么 UE5 是核心

world model 训练数据要 per-frame 的 6-DoF camera pose + action vector + object state change。real-world video 拿不到这种标注——你用 MegaSaM（https://arxiv.org/abs/2412.04463）估出来的 camera pose 是稀疏 keyframe 上的，插值到 dense 后精度有限。

UE5 直接给你 ground-truth：每帧的 position + Euler angle + WASD/IJKL action + character world position。这是合成数据在 world model 训练上的真正价值——不是"real lookalike"，是"annotation ground truth"。

工程上有个聪明设计：**trajectory 采集和渲染解耦**。先 lightweight 用 NavMesh 在 scene 里 explore，过滤掉 stuck / too short / no motion 的 trajectory，存下来；然后 offline 用 Movie Render Queue 分布式重渲染 valid trajectory。如果边采边渲染，GPU 会浪费在无效 clip 上。

real-world data（SpatialVID https://arxiv.org/abs/2509.09676, RealEstate10K, Sekai https://arxiv.org/abs/2506.15675, DL3DV）和 game data（Sekai-Game, OmniWorld-Game）一起塞进来，统一到同一个 camera coordinate system。

---

## E-PRoPE：给 attention 装"相机感"

原版 PRoPE（https://arxiv.org/abs/2406.06423，Li et al. NeurIPS 2025）的想法是：standard RoPE 只编码 token 在 spatiotemporal grid 上的位置，但同样一个 pixel 位置在 camera A 和 camera B 下指向完全不同的世界点。要让 model 真正理解 camera geometry，得把 projective camera relationship 显式塞进 attention 的 $Q, K$。

做法：每个 token 配一个 $d \times d$ 的 block diagonal 矩阵

$$D_s^{PRoPE} = \begin{bmatrix} D_s^{Proj} & 0 \\ 0 & D_s^{RoPE} \end{bmatrix}$$

- $D_s^{Proj}$：$d/2 \times d/2$ 的 submatrix，编码 token $s$ 对应的 world-to-image projection（依赖相机内外参 $K_s, R_s, t_s$ 和 token 的 pixel 位置 $(u_s, v_s)$，通过 $\pi_A^{-1} \circ \pi_B$ 这种 relative projective 关系表达）。
- $D_s^{RoPE}$：$d/2 \times d/2$ 的 submatrix，复制标准 RoPE 的 spatiotemporal encoding。

直接乘到 $Q, K$ 上。intuition：让 attention 在计算 query-key 相似度时，"知道"这两个 token 来自相机的什么相对几何关系，而不是只看 grid 距离。

**问题**：在 DiT 每个 attention layer 都加这个分支，几乎翻倍计算量。5 秒 720p 的视频经 Wan2.2 5B 的 VAE 出 $S=18480$ tokens，attention $O(S^2)$ 非常贵。

**E-PRoPE 的简化**：

1. **空间下采样**：把进 PRoPE 分支的 tokens 从 18480 下采样到 $N=4096$（4.5× 下采样），再投影到更低维度 $d' < d$。insight 是：PRoPE 主要捕捉 view-dependent high-level semantics（"这块区域大致对应 camera B 哪一块"），不需要 pixel-level granularity——DiT 主 attention 本身已经提供 spatiotemporal inductive bias，PRoPE 只是个 "additive geometry prior"。

2. **去掉 $D_s^{RoPE}$**：这部分冗余，DiT 主 attention 已经有 RoPE。只保留 $D_s^{Proj}$。

公式简化为：

$$D_s^{E\text{-}PRoPE} = D_s^{Proj} \in \mathbb{R}^{d' \times d'}$$

训练时 freeze 整个 DiT backbone，只训 PRoPE 参数，loss 用标准 rectified flow denoising。这避免 PRoPE 训练扰动预训练 visual prior。PRoPE attention 的输出上采样回原分辨率，残差加到主 attention 输出上。

**实验结果**（Table 1，5s 720p on 8× H20）：

| Method | Camera ↑ | Latency (s) ↓ |
|---|---|---|
| PRoPE | 73.89 | 80 |
| E-PRoPE | 73.75 | 59 |

Camera control 几乎不掉（73.89 → 73.75），latency 降 26%。image quality 甚至略升（66.15 → 66.75），因为更少参数变动 = 更稳定 visual prior。这说明 PRoPE 学到的 projective bias 高度 redundant。

还有个有趣观察：downstream model 训练时不加 PRoPE，inference 时 plug-and-play 加上 pre-trained PRoPE 也能 work。说明 PRoPE 编码的几何 prior 是 robust 的、不依赖特定 weight 耦合。

---

## Memory-Conditioned Scene Persistence：让 model 记得自己看过什么

修挂法 1。核心：生成当前 chunk 时，除了 recent history frames，再 retrieve 一组 memory frames（早期生成的、与当前 camera view 有 overlap 的）一起作为 condition。

### 训练时的 packing

latent video $z_{1:T}^0$，camera signals $\pi_{1:T}$。采样三组：

- $z_\mathcal{M}$：memory latents（从早期 history 检索，是 model 之前预测的 clean latent）
- $z_\mathcal{H}$：recent history latents（紧邻 target window 之前 denoised 的）
- $z_\mathcal{C}^\tau$：target latents + noise level $\tau$

pack 成一条 sequence：

$$z_{\text{pack}} = [z_\mathcal{M} \mid z_\mathcal{H} \mid z_\mathcal{C}^\tau] \tag{1}$$

$[\cdot | \cdot]$ 是 token 维度 concat。DiT self-attention 一次处理这三段，rectified flow loss 只算在 $z_\mathcal{C}$ 上。

**为什么 concat 进 self-attention 而不是 cross-attention**：paper 提试过 cross-attention 和 VACE-style conditioning（https://arxiv.org/abs/2503.06448），都更差。intuition 是：cross-attention 把 memory 当 external signal，model 可以选择性忽略；self-attention concat 让 memory frame 直接参与 spatiotemporal token 间相互作用，更接近 video diffusion 原生 frame-to-frame consistency 机制。

### 几何检索

memory frame 不是按 temporal distance 选（"最近 10 帧"早就离开这个房间了），而是按 camera pose overlap 选——选与当前 target view 在 3D 空间里有 view overlap 的 history frame。这才是有价值的 memory。

检索后每个 memory frame 加它**原 temporal location 对应的 RoPE**，否则会被 attention 当成"紧邻 target 的 recent frame"。对超大 time gap，借鉴 NTK-aware RoPE scaling、YaRN（https://arxiv.org/abs/2309.00071）、randomized positional encodings（https://arxiv.org/abs/2305.16843）。intuition：标准 RoPE 在距离超出训练 range 后 attention 衰减到 0，长程 memory 就"看不见"了，需要外推。

### Exposure bias 与 residual recycling

经典 train-test gap：训练时 conditioning frame 来自 ground-truth，inference 时来自 model 自己生成的、含 error 的 frame。

借鉴 Stable Video Infinity（https://arxiv.org/abs/2510.09212）的 **error injection**：训练时对 conditioning tokens（$z_\mathcal{M}, z_\mathcal{H}$）加扰动，但保持 target $z_\mathcal{C}$ clean。model 学到：memory 可靠时用 memory，memory 含 error 时 fall back to learned prior。

"residual recycling" 的具体机制 paper 写得简略，本质应该是：conditioning path 上的 perturbation 通过 residual connection 传到 attention 输出，但 supervised target 不动，让 model 在 corrupted conditioning 和 clean target 之间建立 robust mapping。

---

## Event Instruction Tuning：多对象事件

这是 DreamX 相对其他开源 world model（HY-WorldPlay 1.5 https://arxiv.org/abs/2512.14614, LingBot-World https://arxiv.org/abs/2601.20540, Yume-1.5 https://arxiv.org/abs/2512.22096, Matrix-Game 3.0 https://arxiv.org/abs/2604.08995）的差异化卖点：支持 **multi-entity composition + inter-object interaction**。

### 数据格式

每个 event instruction 两部分：

- **global description**：scene context + 整体 temporal evolution；
- **per-entity event records**：每个 entity 一条 record，含 entity reference / event predicate / spatial anchor / temporal interval。

举例：交通场景下，"行人过马路 + 车辆减速 + 信号灯变红"是三条 entity record，interaction（车避让行人）在 global caption 里描述。model 学到 atomic event grounding 和 compositional reasoning。

### 训练

事件语义 **只通过 text-conditioning interface** 注入，不改 architecture，把 structured event instruction 渲染成 natural-language prompt 喂给原 text encoder。聪明选择：避免新 condition branch，复用 Wan2.2 的 text-video alignment prior。

training mixture 混合 event-instruction 样本和 non-event clip，保留 general world generation 能力。strict gradient clipping + conservative update 防止 catastrophic forgetting。

Table 2 对比：

| Model | Multi-Entity Composition | Inter-Object Interaction |
|---|---|---|
| LingBot-World | △ | ✗ |
| HY-WorldPlay 1.5 | △ | ✗ |
| Matrix-Game 3.0 | ✗ | ✗ |
| Yume-1.5 | ✗ | ✗ |
| DreamX-World 1.0 | ✓ | ✓ |

---

## Autoregressive Long Video Distillation

把 bidirectional Wan2.2 转成 few-step AR generator 的核心是 **Causal Forcing + DMD + long rollout** 三件套。

### Causal Forcing（https://arxiv.org/abs/2602.02214）

bidirectional teacher 训练时所有 frame 同步 denoise，student inference 时只能看过去 frame。这个 structural mismatch 让蒸馏效果差。Causal Forcing 的解法是从 autoregressive teacher 蒸馏，让 student 在 training-time 就暴露于"自己的过去输出"。

### DMD（https://arxiv.org/abs/2311.18828）

Distribution Matching Distillation：student few-step 采样，teacher 多步采样，loss 让两者 output distribution 在 feature space 对齐。DreamX 这里做 **DMD-forcing**：long video 上 sample local temporal window，camera-controlled AR student rollout 与 bidirectional E-PRoPE teacher 在 window 内做 distribution match。

### Long rollout training + Infinity-RoPE

借鉴 LongLive（https://arxiv.org/abs/2509.22622）和 Infinity-RoPE（https://arxiv.org/abs/2511.20649）：long sequence 上 long rollout 训练，让 model 见到自己的 generated history，学到 chunk 间 style/color drift 模式并修正。Infinity-RoPE 扩展 AR context length，避免长视频 identity drift、background mutation。

### I2V DMD

为保留 I2V 质量，每个 DMD window 第一 latent frame 用 VAE decode 出来作为 image condition 喂给 bidirectional E-PRoPE teacher，teacher 在 local window 上监督 camera-controlled AR student。student 既支持 T2V 也支持 I2V，I2V anchor 到 reference image。

最终稳定生成 up to 1 minute video，跨 chunk 保持 camera controllability + temporal coherence。

---

## RL Post-training：恢复 distillation 丢的能力

DMD 之后 model 已经是 few-step AR，但 visual quality 和 camera 跟随都退化了。RL 阶段做 recovery。

### 为什么 RL 而不是继续 supervised

perceptual quality 和 camera 跟随是 **non-differentiable objective**，supervised loss 表达不准。RL 直接优化 reward model 评分。

### 流程

参考 Astrolabe（https://arxiv.org/abs/2603.17051）和 WorldCompass（https://arxiv.org/abs/2602.09022）：

1. 对每个 (text, image, camera) condition，current model 生成多个 long-horizon rollout candidate；
2. 从 rollout 中 sample 短 clip，送 reward model；
3. reward 通过 **DiffusionNFT**（https://arxiv.org/abs/2509.16117）做 forward-process RL 更新，绕过 reverse-process likelihood 估计，对 few-step distilled model 友好；
4. KL regularization 保持 updated model 接近 DMD-distilled model。

两个 reward：
- **Camera-control reward**：水平 translation + rotation 精度；
- **Visual-quality reward**：生成 clip 的感知质量。

### 稳定性

few-step model 对 reward update 极其敏感——一步大 update 就 collapse。**Gradual update strategy**：小步长逐步更新。这是 RL post-training diffusion model 的通用经验（参考 DDPO https://arxiv.org/abs/2305.13301, Flow-GRPO https://arxiv.org/abs/2505.05470, "Taming preference mode collapse" https://arxiv.org/abs/2506.xxxxx）。

**长 rollout + 短 clip** 设计：long rollout 提供 autoregressive context（让 reward 看到长程影响），backprop 只走短 clip（控 GPU memory）。把 rollout horizon 和 optimization window 解耦。

---

## 推理加速：16 FPS on 8× RTX 5090

### DiT denoising 优化

- **INT8 SageAttention**（https://arxiv.org/abs/2410.02371）：attention 量化到 INT8，plug-and-play。
- **FP8 FFN with AngelSlim**（https://arxiv.org/abs/2602.21233）：FFN 量化到 FP8。
- **Sequence parallelism**：长 spatiotemporal token 序列跨 GPU shard，只同步必要 attention 和 norm 统计量。
- **Fused Triton kernels**：elementwise ops、layout transform、small reduction 融合成单 kernel，降 intermediate allocation 和 kernel launch overhead。
- **TeaCache**（Liu et al. CVPR 2025 "Timestep embedding tells"）：相邻 diffusion step residual 变化小的 timestep 区域，skip 部分 Transformer block forward。diffusion 后期 step 接近收敛，residual 变化小，可 reuse。

### VAE decoding 优化

- **Matrix-Game 3.0 VAE**（https://arxiv.org/abs/2604.08995）：75% pruning ratio，single-chunk decoding 降到 ~0.25s。
- **torch.compile**：第一次 iteration 后 compile，进一步降 latency。
- **ParaVAE**（https://github.com/RiseAI-Sys/ParaVAE）：latent video 沿 height 切分到不同 GPU，每卡 decode 一个 local patch，gather 成最终 video。降 peak per-GPU memory。

### Asynchronous pipeline parallelism

chunk k 的 VAE decoding 与 chunk k+1 的 control reception + KV-cache update + DiT denoising 异步重叠。VAE latency 被藏在 diffusion computation 后面，几乎不可见。

### AR streaming inference

chunk-wise：每个 chunk 从 noise 起，用 distilled few-step sampler 在 text prompt + chunk-relative camera trajectory + rolling KV cache 下 denoise，输出 token 写回 cache。camera control 是 **chunk-relative**：第一 chunk 相对第一帧，后续相对上一 chunk 最后一帧。避免 long sequence 上 conditioning signal 衰减。

I2V 唯一区别：第一 chunk 第一帧替换为 input image，anchor 整个 video。

---

## 评估：revisit consistency 是关键创新

### Basic evaluation（5s）

camera control error 公式：

$$e_{\text{camera}} = \sqrt{e_\theta \cdot e_t} \tag{2}$$

- $e_\theta$：scale-invariant rotation error；
- $e_t$：scale-invariant translation error；
- 跨所有 frame 平均，normalize 到 [0, 100]。

几何意义：旋转和位移误差的几何平均——任一差都拉低分数，对"旋转好但位移差"或"位移好但旋转差"不偏袒。

Visual quality 用 Omni-WorldBench（https://arxiv.org/abs/2603.22212）：imaging quality / temporal flicker / motion smoothness / dynamic degree / transition detection。

Artifact detection 用 Gemini-3.1-Pro VLM 做 binary pass/fail（duplicated limbs、object vanishing、geometric pass-through），2 FPS 采样，每 case 两次取平均。

Table 3：

| Model | Params | Camera ↑ | Quality ↑ | Artifact ↑ | Overall ↑ |
|---|---|---|---|---|---|
| HY-WorldPlay 1.5 | 8B | 65.12 | 68.23 | 71.66 | 80.79 |
| LingBot-World | 14B | 71.73 | 67.76 | 58.33 | 80.45 |
| DreamX-World-1.0-5B | 5B | **73.75** | 66.75 | **73.75** | **84.76** |

5B 在 camera control 和 overall 上超过 8B / 14B 对手。

### Long-horizon（30s）

Table 4：DreamX overall 70.41 vs HY-WorldPlay 68.85 vs LingBot 67.43。但所有 model 在 30s 上 artifact 分都跌到 12-17，说明 long-horizon world model 仍是 open problem。

### Revisit consistency —— paper 最重要的 evaluation创新

现有 benchmark（WorldScore https://arxiv.org/abs/2504.00983, Omni-WorldBench）只测 short-term，不测"model 记不记得自己之前生成过什么"。DreamX 引入 revisit-based 协议。

**三种轨迹模板**（Fig. 11）：

- (a) Out-and-back：横向 translate（D×3）再 reverse（A×3），回到原位 + 同 orientation，测 appearance stability；
- (b) Translation-rotation：W·S·L·R·R·L·L，回到原位但 yaw 不同，测 viewpoint 变化下 place identity；
- (c) Closed-loop：矩形路径回到 exact starting pose，测 loop closure 下 global layout consistency。

**Revisit pair 检测**：

$$|\theta_i - \theta_j| \leq \tau_\theta, \quad \|t_i - t_j\|_2 \leq \tau_t \tag{3}$$

- $\theta$：yaw；$\mathbf{t} = (t_x, t_y, t_z)$：position；
- $\tau_\theta = 2°$，$\tau_t = 0.1$；
- 最小 temporal gap $|j - i| \geq \lfloor 0.2T \rfloor$，强制 long-horizon memory；
- 多候选选 weighted pose distance $|\theta_i - \theta_j| + 10\|t_i - t_j\|_2$ 最小的。

**6 个 metrics 跨抽象层级**：

- ∆PSNR / ∆SSIM：pixel-level fidelity；
- ∆LPIPS（https://arxiv.org/abs/1801.03924）：perceptual；
- ∆DINO-Sim（DINOv2 https://arxiv.org/abs/2304.07193）：semantic identity；
- ∆VPR-Sim（MutualVPR https://arxiv.org/abs/2612.xxxxx，NeurIPS 2026）：place recognition，训练时对 viewpoint 变化 robust；
- ∆SP-Match（SuperPoint https://arxiv.org/abs/1712.07629 + LightGlue https://arxiv.org/abs/2306.13643，最多 1024 keypoints，$r_{\text{match}} = N_{\text{match}} / \min(N_i, N_j)$）：geometric structure；
- CLIP-Video（https://arxiv.org/abs/2103.00020）：temporal smoothness，相邻 frame 平均 CLIP 相似度。

**Gain-based scoring**：$S_{\text{revisit}} - S_{\text{baseline}}$（similarity 类）/ $S_{\text{baseline}} - S_{\text{revisit}}$（LPIPS 类）。baseline 是同 temporal gap 但 non-revisit 的 pair。这个设计关键——避免 slow camera movement 造成的虚假高分（如果 camera 几乎不动，revisit 和 baseline 看起来都很相似，但那不是 memory）。

Table 5 结果（10s video）：

| Model | ∆PSNR | ∆SSIM | ∆LPIPS | ∆DINO-Sim | ∆VPR-Sim | ∆SP-Match | CLIP-V |
|---|---|---|---|---|---|---|---|
| LingBot-World | 0.61 | 0.019 | 0.039 | 0.090 | 0.100 | 0.088 | 0.987 |
| HY-WorldPlay 1.5 | 3.19 | 0.079 | 0.202 | 0.200 | 0.110 | 0.251 | 0.992 |
| DreamX-World-1.0-5B | **3.92** | **0.098** | **0.232** | **0.246** | **0.142** | 0.216 | 0.991 |

DreamX 在 pixel / perceptual / semantic / place recognition 四层最强，HY-WorldPlay 在 SP-Match（geometric structure）和 CLIP-V（temporal smoothness）略胜。分层评估的好处是能定位"哪一层 memory 失败"——pixel-level 好但 semantic 差，说明学到的是局部 texture copy 而非 scene-level identity。

### Human preference

blind side-by-side，4 个维度。DreamX vs HY-WorldPlay win/tie/lose = 57.5/14.4/28.1，vs LingBot = 61.9/10.6/27.5。visual quality 和 artifact 上明显胜，camera control tie rate 高（说明 perceived controllability 接近）。

---

## 几个值得注意的 design choice

### 为什么从 Wan2.2-TI2V 起步

Wan2.2（https://arxiv.org/abs/2503.20314）是当前开源 SOTA 级别 TI2V video diffusion，5B / 14B 双 size，原生支持 text + image 双 condition。从它初始化意味着继承一个很强的 visual prior，避免从零训练 world model 的成本。这也解释了为什么 5B 能打 14B——base model 强 + 训练 pipeline 精细。

### 为什么 E-PRoPE 只 freeze backbone 训 PRoPE 参数

全量 fine-tune 会让 PRoPE 加的 attention 分支扰动 DiT 主 attention 的 spatiotemporal prior，visual quality 退化。freeze + 只训 PRoPE 让 camera geometry 作为 "additive bias" 加到 attention 输出上，最小化对原 model 干扰。这也解释了为什么 E-PRoPE 在 image quality 上甚至略高于 PRoPE（66.75 vs 66.15）——更少参数变动 = 更稳定 visual prior。

### 为什么 RL 用 DiffusionNFT 而不是 DDPO / Flow-GRPO

DDPO（https://arxiv.org/abs/2305.13301）把 denoising 当 sequential decision process，需要在 reverse process 上 backprop，对 solver 依赖强。Flow-GRPO（https://arxiv.org/abs/2505.05470）是 flow-matching 上的 online policy opt，要求 reverse sampling 可微。DiffusionNFT（https://arxiv.org/abs/2509.16117）直接在 forward process 上做 negative-aware fine-tuning，绕过 reverse likelihood 估计，对 distilled few-step model 友好——few-step model 本来就没什么 reverse process 可微。

### Long-horizon 仍掉得厉害的根因

30s 上所有 model artifact 分都跌到 12-17，说明 AR world model 在长 horizon 上仍会累积错误。DreamX 的 mitigation 是 memory conditioning + long rollout training + RL alignment，只是减缓不是消除。Limitations 里也承认"generated worlds may drift drastically in object appearance or layout after extended interaction"。这是下一步 research 的核心战场。

### Future work 的两个方向

- **Character-centric world model**：persistent character identity + 多角色长程 interaction。当前 model 的弱项——character face / clothing 在 AR chunk 间漂移严重。
- **Native audio-visual world model**：联合生成同步 speech + ambient sound + action-dependent audio，把 audio 作为 interactive signal。world model 场景下 audio 还是空白。

---

## 我对这篇 paper 的整体判断

**Full-stack engineering 的胜利**：从 UE5 数据 → E-PRoPE → Memory conditioning → Event instruction → DMD forcing → RL → INT8/FP8 量化 + VAE pruning + async pipeline，每一环都做了 specific 设计。这种 paper 的价值在于"把所有 piece 拼起来跑通 16 FPS"，单个 piece 都不是惊天突破，组合的 system-level 表现远超单独优化任一组件能拿到的效果。

**E-PRoPE 是最 elegant 的算法贡献**：通过"PRoPE 主要捕捉 high-level view-dependent semantics"这一 insight，把 attention 计算从 18480 token 压到 4096，几乎无 quality loss。这种"先理解 module 学到什么再决定怎么 simplify"的思路值得借鉴。

**Revisit consistency benchmark 是 evaluation 层的重要贡献**：现有 world model benchmark 都在测 short-term quality，没人测"model 记不记得自己生成过什么"。这套分层 metric（pixel → perceptual → semantic → place → geometric）+ gain-based scoring 应该会成为后续 world model 评估的标准组件。

**Open problem 清单**：long-horizon drift（30s artifact 12-17）、caption/camera/event conflict、character identity persistence、native audio。这些是下一代 world model 的 research agenda。

参考链接汇总：
- Wan2.2: https://arxiv.org/abs/2503.20314
- PRoPE (Cameras as Relative Positional Encoding): https://arxiv.org/abs/2406.06423
- Self-Forcing: https://arxiv.org/abs/2506.08009
- Causal Forcing: https://arxiv.org/abs/2602.02214
- DMD: https://arxiv.org/abs/2311.18828
- Stable Video Infinity: https://arxiv.org/abs/2510.09212
- LongLive: https://arxiv.org/abs/2509.22622
- Infinity-RoPE: https://arxiv.org/abs/2511.20649
- YaRN: https://arxiv.org/abs/2309.00071
- Randomized positional encodings: https://arxiv.org/abs/2305.16843
- RoPE (RoFormer): https://arxiv.org/abs/2104.09864
- TeaCache (Timestep embedding tells): https://arxiv.org/abs/2510.xxxxx
- SageAttention: https://arxiv.org/abs/2410.02371
- AngelSlim: https://arxiv.org/abs/2602.21233
- ParaVAE: https://github.com/RiseAI-Sys/ParaVAE
- Matrix-Game 3.0: https://arxiv.org/abs/2604.08995
- HY-WorldPlay 1.5: https://arxiv.org/abs/2512.14614
- LingBot-World: https://arxiv.org/abs/2601.20540
- Yume-1.5: https://arxiv.org/abs/2512.22096
- WorldScore: https://arxiv.org/abs/2504.00983
- Omni-WorldBench: https://arxiv.org/abs/2603.22212
- WBench: https://arxiv.org/abs/2605.25874
- MegaSaM: https://arxiv.org/abs/2412.04463
- SpatialVID: https://arxiv.org/abs/2509.09676
- RealEstate10K: https://arxiv.org/abs/1802.05581
- Sekai: https://arxiv.org/abs/2506.15675
- DL3DV: https://arxiv.org/abs/2310.03543
- VACE: https://arxiv.org/abs/2503.06448
- DDPO: https://arxiv.org/abs/2305.13301
- Flow-GRPO: https://arxiv.org/abs/2505.05470
- DiffusionNFT: https://arxiv.org/abs/2509.16117
- Astrolabe: https://arxiv.org/abs/2603.17051
- WorldCompass: https://arxiv.org/abs/2602.09022
- DINOv2: https://arxiv.org/abs/2304.07193
- LightGlue: https://arxiv.org/abs/2306.13643
- SuperPoint: https://arxiv.org/abs/1712.07629
- LPIPS: https://arxiv.org/abs/1801.03924
- CLIP: https://arxiv.org/abs/2103.00020
- CameraCtrl: https://arxiv.org/abs/2402.02606
- MotionCtrl: https://arxiv.org/abs/2312.03641
- AC3D: https://arxiv.org/abs/2411.18673
- GameNGen: https://arxiv.org/abs/2408.14837
- Genie: https://arxiv.org/abs/2401.15441
- GameGen-X: https://arxiv.org/abs/2410.11961
- MutualVPR: https://arxiv.org/abs/2612.xxxxx
- Project page: https://dreamx-world.github.io
- GitHub: https://github.com/AMAP-ML/DreamX-World

如果你想深挖某一环，比如 E-PRoPE 的 projective matrix 具体怎么构造、或 DMD-forcing 的 window sampling 策略，告诉我，我再展开。

---

# DreamX-World 1.0 深度解读

Andrej，这篇 paper 是个相当完整的 full-stack 工程，把 video diffusion 从 offline short-clip 推到了 real-time interactive world model。核心思路是：把 bidirectional 的 Wan2.2-TI2V（参考 https://arxiv.org/abs/2503.20314）通过一系列渐进式训练改造成 few-step autoregressive generator，同时解决 camera control、long-horizon memory、composable events 三个交互式场景下独有的难题。下面按技术栈分层讲。

---

## 1. 核心问题：为什么 video diffusion 不能直接当 world model 用

如果只看 offline video generation（Wan2.2, HunyuanVideo, CogVideoX 等），它们能在 5s 内生成高质量 clip。但当作为 interactive world model 时，会暴露三类失败：

**Failure mode A — Camera revisits 后 scene drift**：bidirectional model 没有真正的 scene state，每次生成都是条件分布 $p(x_t | x_{<t}, \text{prompt})$ 的独立采样。当 camera 转一圈回到原位，模型没有"记忆"之前生成的 scene layout，于是会采到一个 different plausible scene。这本质是 latent state 在 autoregressive chunk 间没被显式保存。

**Failure mode B — Autoregressive chunk drift**：一旦做 chunk-wise AR，每个 chunk 的 prediction error（color tone、texture、identity）会累积，scene 整体慢慢漂走。这就是 Self-Forcing（https://arxiv.org/abs/2506.08009）和 Causal Forcing（https://arxiv.org/abs/2602.02214）想解决的 train-test gap。

**Failure mode C — Distillation 退化**：要把 bidirectional 50+ step 的 diffusion 压到 few-step autoregressive，必须做 DMD（https://arxiv.org/abs/2311.18828）这类 distillation，但 aggressive distillation 会丢 motion 多样性、camera 跟随精度和 visual quality。

DreamX 的整个 pipeline 就是逐一对这三类失败做防御：E-PRoPE → Memory conditioning → DMD forcing + long rollout → RL alignment → 系统 level 加速。

---

## 2. 数据引擎：为什么 UE5 是核心

interactive world model 的训练数据要求"per-frame 6-DoF camera pose + discrete action + 物体 state change annotation"，这些在 real-world video 上几乎拿不到。real-world 数据（SpatialVID https://arxiv.org/abs/2509.09676, RealEstate10K https://arxiv.org/abs/1802.05581, Sekai https://arxiv.org/abs/2506.15675, DL3DV https://arxiv.org/abs/2310.03543）的 camera pose 是用 MegaSaM（https://arxiv.org/abs/2412.04463）sparse 估计 keyframe 再 SLERP 插值出来的，精度有上限。UE5 反过来提供 ground-truth pose + Euler angles + WASD/IJKL action vector + character world position。

UE5 pipeline 的关键设计是 **decoupled trajectory collection 和 offline rendering**：
- online exploration engine 用 NavMesh 做 collision-aware goal sampling，reject invalid trajectory（stuck detection、min duration、min path length）；
- valid trajectory 存下来后用 Movie Render Queue 分布式重渲染。

这是个很重要的工程取舍：如果 online 边采样边渲染，会浪费 GPU 在无效 clip 上。先 lightweight explore → 离线 batch 渲染，把 GPU 利用率拉满。

**Geometric filtering** 那一段也很关键：real-world 的 sparse pose 通过 SLERP（rotation）+ linear（translation）插值到 dense，然后做"translation spike / rapid rotation / vertical jitter / inconsistent intrinsics"剔除。这些异常值如果不清，PRoPE 在训练时会学到错误的 projective mapping，把 noise 当 geometry 学。

---

## 3. E-PRoPE：camera control 的工程化简化

这是这篇 paper 最有意思的算法贡献。先回顾原版 PRoPE（Li et al. 2025a, https://arxiv.org/abs/2406.06423，Cameras as Relative Positional Encoding, NeurIPS 2025）。

### 3.1 原版 PRoPE 的形式

给定 token 序列 $X = \{x_s\}_{s=1}^S$，每个 token 对应一个 per-token 投影矩阵 $D_s^{PRoPE} \in \mathbb{R}^{d \times d}$，结构为：

$$D_s^{PRoPE} = \begin{bmatrix} D_s^{Proj} & 0 \\ 0 & D_s^{RoPE} \end{bmatrix}$$

- $D_s^{Proj} \in \mathbb{R}^{d/2 \times d/2}$：编码 token $s$ 对应的 world-to-image projection geometry。具体来说，给定相机内外参 $(K_s, R_s, t_s)$ 和 token 在 latent grid 上的像素坐标 $(u_s, v_s)$，通过 projective 变换得到与 reference camera 的相对几何关系。
- $D_s^{RoPE} \in \mathbb{R}^{d/2 \times d/2}$：复制标准 RoPE（https://arxiv.org/abs/2104.09864）的 spatiotemporal 位置编码。
- 整个矩阵通过矩阵乘法作用到 attention 的 $Q, K$ 上。

intuition：camera frustum 之间的几何关系不是单纯的 spatiotemporal 距离能表达的——同一个 pixel 位置在 camera A 和 camera B 下指向完全不同的世界点。PRoPE 把 projective geometry 显式塞进 attention，让 model 学到"如果 query token 在 camera A 的某个像素，key token 在 camera B 的某个像素，它们的几何关系由 $\pi_A^{-1} \circ \pi_B$ 决定"。

### 3.2 E-PRoPE 的简化

原版 PRoPE 的痛点是：要在 DiT 每个 attention layer 上加额外的 attention 分支，几乎翻倍计算量。在长视频（5s 720p → Wan2.2 5B VAE 出 $S=18480$ tokens）上极其贵。

E-PRoPE 的两个简化：

**简化 1 — 空间下采样**：把输入 PRoPE 分支的 tokens 从 18480 下采样到 $N=4096$（>4.5× spatial 下采样），同时投影到更低的 $d' < d$ 维度。intuition 来自一个观察：PRoPE 主要捕捉 view-dependent high-level semantics（"这块区域大致对应 camera B 哪一块"），不需要 fine-grained 像素级 attention——DiT backbone 本身的 attention 已经提供 spatiotemporal inductive bias。下采样后做 PRoPE attention，再上采样回原分辨率，加到原 attention 输出上。

**简化 2 — 去掉 $D_s^{RoPE}$**：原 PRoPE 的 RoPE 分支是冗余的，因为 DiT 主 attention 已经有 RoPE。E-PRoPE 只保留 $D_s^{Proj}$：

$$D_s^{E-PRoPE} = D_s^{Proj} \in \mathbb{R}^{d' \times d'}$$

训练时 freeze DiT backbone，只训 PRoPE 参数，loss 是标准 rectified flow denoising loss。这避免了 PRoPE 训练扰动预训练 visual prior。

### 3.3 实验数据

Table 1 对比 PRoPE vs E-PRoPE（5s 720p，8× H20）：

| Method | Camera ↑ | Latency (s) ↓ |
|---|---|---|
| PRoPE | 73.89 | 80 |
| E-PRoPE | 73.75 | 59 |

Camera control 几乎不丢（73.89 → 73.75），latency 降 26%。这是个相当好的 trade-off，说明 PRoPE 学到的 projective bias 是高度 redundant 的，下采样后语义仍完整。

还有一个 plug-and-play 的有趣发现：训练时不加 PRoPE 的 downstream model 在 inference 时加上 pre-trained PRoPE 也能 work，说明 PRoPE 编码的几何 prior 是 robust 的、不依赖特定 weight 耦合。

---

## 4. Memory-Conditioned Scene Persistence

这一段解决 Failure mode A。核心 idea：在 AR 生成当前 chunk 时，除了用 recent history frames，再 retrieve 一组 memory frames（早期生成的、与当前 camera view 有 overlap 的 frames）一起作为 condition。

### 4.1 训练时的 packing

latent video sequence $z_{1:T}^0$，camera signals $\pi_{1:T}$。训练时采样三组：

- $z_\mathcal{M}$：memory latents（从早期 history 检索，预测的 clean latent）
- $z_\mathcal{H}$：recent history latents（紧邻 target window 之前的 denoised latents）
- $z_\mathcal{C}^\tau$：target latents + diffusion noise level $\tau$

pack 成一条 sequence：

$$z_{\text{pack}} = [z_\mathcal{M} \mid z_\mathcal{H} \mid z_\mathcal{C}^\tau] \tag{1}$$

$[\cdot | \cdot]$ 是 token 维度 concat。DiT 的 self-attention 一次性处理这三段，rectified flow loss 只在 $z_\mathcal{C}$ 上算。

**为什么 concat 而不是 cross-attention**：paper 提到试过 cross-attention 和 VACE-style conditioning（https://arxiv.org/abs/2503.06448, Jiang et al. ICCV 2025），都更差。intuition：cross-attention 把 memory 当 external signal，模型可以选择性忽略；self-attention concat 让 memory frame 直接参与 spatiotemporal token 间相互作用，更接近 video diffusion 原生的"frame-to-frame consistency"机制。

### 4.2 几何检索 + 长程位置编码

memory frame 检索不是按 temporal distance，而是按 **camera pose overlap**：选与当前 target view 在 3D 空间中有 view overlap 的 history frame。这很关键——如果你走过一个房间又回来，"最近 10 帧"已经不在那个房间了，但 100 帧前某帧的 camera pose 与现在高度 overlap，那帧才是有价值的 memory。

检索后每个 memory frame 还要加它**原 temporal location 对应的 RoPE**，否则会被 attention 当成"紧邻 target 的 recent frame"。对超大 time gap，借鉴 NTK-aware RoPE scaling、YaRN（https://arxiv.org/abs/2309.00071）、randomized positional encodings（https://arxiv.org/abs/2305.16843）做长程位置处理。intuition：标准 RoPE 在距离超出训练 range 后 attention 衰减到 0，长程 memory 就"看不见"了，需要外推。

### 4.3 Exposure bias 与 residual recycling

训练时 conditioning frame 来自 ground-truth data，inference 时来自 model 自己生成的、含 prediction error 的 frame。这是经典的 exposure bias。DreamX 借鉴 Stable Video Infinity（https://arxiv.org/abs/2510.09212）的 **error injection**：训练时对 conditioning tokens（$z_\mathcal{M}, z_\mathcal{H}$）加扰动，但保持 target $z_\mathcal{C}$ clean。模型学到：memory 帮助时用 memory，memory 含 error 时 fall back to learned prior。

"residual recycling" 的具体含义 paper 写得简略，我推断是指：conditioning path 上的 perturbation 通过 residual connection 传到 attention 输出，但 supervised target 不动，相当于训练 model 在 "corrupted conditioning" 与 "clean target" 之间建立 robust mapping。

---

## 5. Event Instruction Tuning：composable events

这是相对其他 open-source world model（HY-WorldPlay 1.5 https://arxiv.org/abs/2512.14614, LingBot-World https://arxiv.org/abs/2601.20540, Yume-1.5 https://arxiv.org/abs/2512.22096, Matrix-Game 3.0 https://arxiv.org/abs/2604.08995）的差异化卖点：支持 **multi-entity composition + inter-object interaction**。

### 5.1 数据格式

每个 event instruction 由两部分组成：
- **global description**：scene 上下文 + 整体 temporal evolution；
- **per-entity event records**：每个 entity 一个 record，包含 entity reference / event predicate / spatial anchor / temporal interval。

举例：交通场景下，"行人过马路 + 车辆减速 + 信号灯变红"是三个 entity record，但它们之间有 interaction（车避让行人），interaction 在 global caption 里显式描述。这让 model 学到 atomic event grounding 和 compositional reasoning。

### 5.2 训练

事件语义 **只通过 text-conditioning interface** 注入——不改 architecture，把 structured event instruction 渲染成 natural-language prompt 喂给原 text encoder。这是个聪明的工程选择：避免引入新的 condition branch，复用 Wan2.2 的 text-video alignment prior。

training mixture 混合 event-instruction 样本和 non-event clip，保留 general world generation 能力。strict gradient clipping + conservative update 防止 catastrophic forgetting。

Table 2 对比：

| Model | Multi-Entity Composition | Inter-Object Interaction |
|---|---|---|
| LingBot-World | △ | ✗ |
| HY-WorldPlay 1.5 | △ | ✗ |
| Matrix-Game 3.0 | ✗ | ✗ |
| Yume-1.5 | ✗ | ✗ |
| DreamX-World 1.0 | ✓ | ✓ |

△ = qualitative / partial support。

---

## 6. Autoregressive Long Video Distillation

把 bidirectional Wan2.2 转成 few-step AR generator 的核心是 **Causal Forcing + DMD + long rollout** 三件套。

### 6.1 Causal Forcing（https://arxiv.org/abs/2602.02214）

bidirectional teacher 和 causal student 之间有结构 mismatch：teacher 训练时所有 frame 同步 denoise，student inference 时只能看到过去 frame。Causal Forcing 的解法是从 autoregressive teacher 蒸馏，让 student 在 training-time 就暴露于"自己的过去输出"。

### 6.2 DMD（https://arxiv.org/abs/2311.18828）

Distribution Matching Distillation：student few-step 采样，teacher 多步采样，loss 让两者的 output distribution 在 feature space 对齐。DreamX 这里做 **DMD-forcing**：在 long video 上 sample local temporal window，camera-controlled AR student rollout 与 bidirectional E-PRoPE teacher 在 window 内做 distribution match。

### 6.3 Long rollout training + Infinity-RoPE

借鉴 LongLive（https://arxiv.org/abs/2509.22622）和 Infinity-RoPE（https://arxiv.org/abs/2511.20649）：在 long sequence 上 long rollout 训练，让 model 见到自己的 generated history，学到 chunk 间的 style/color drift 模式并修正。Infinity-RoPE 扩展 AR context length，避免长视频 identity drift、background mutation。

### 6.4 I2V DMD

为保留 I2V（image-to-video）质量，每个 DMD window 的第一 latent frame 用 VAE decode 出来作为 image condition 喂给 bidirectional E-PRoPE teacher，teacher 在 local window 上监督 camera-controlled AR student。这样 student 既能做 T2V 也能做 I2V，且 I2V anchor 到 reference image。

最终能稳定生成 up to 1 minute video，跨 chunk 保持 camera controllability + temporal coherence。

---

## 7. RL Post-training：恢复 distillation 损失的能力

DMD 之后 model 已经是 few-step AR，但 visual quality 和 camera 跟随都退化了。RL 阶段做 recovery。

### 7.1 为什么 RL 而不是继续 supervised

视觉质量和 camera 跟随是 **non-differentiable objective**（perceptual quality、trajectory adherence），supervised loss 表达不准。RL 可以直接优化 reward model 评分。

### 7.2 流程

参考 Astrolabe（https://arxiv.org/abs/2603.17051）和 WorldCompass（https://arxiv.org/abs/2602.09022）：

1. 对每个 (text, image, camera) condition，current model 生成多个 long-horizon rollout candidate；
2. 从 rollout 中 sample 短 clip，送 reward model；
3. reward 通过 **DiffusionNFT**（https://arxiv.org/abs/2509.16117）做 forward-process RL 更新，避免 reverse-process likelihood 估计；
4. KL regularization 保持 updated model 接近 DMD-distilled model。

两个 reward：
- **Camera-control reward**：水平 translation + rotation 精度；
- **Visual-quality reward**：生成 clip 的感知质量。

### 7.3 工程稳定性

few-step model 对 reward update 极其敏感——一步大 update 就 collapse。DreamX 用 **gradual update strategy**：小步长逐步更新，避免 early collapse。这是 RL post-training diffusion model 的通用经验（参考 DDPO https://arxiv.org/abs/2305.13301, Flow-GRPO https://arxiv.org/abs/2505.05470, "Taming preference mode collapse" https://arxiv.org/abs/2506.xxxxx）。

**长 rollout + 短 clip** 的设计很巧妙：long rollout 提供 autoregressive context（让 reward 看到长程影响），但 backprop 只走短 clip（控 GPU memory）。这把 rollout horizon 和 optimization window 解耦。

---

## 8. 推理加速：16 FPS on 8× RTX 5090

这部分是系统工程的精华，每一项都是单独可拆的贡献。

### 8.1 DiT denoising 优化

- **INT8 SageAttention**（https://arxiv.org/abs/2410.02371, Zhang et al. ICLR 2025）：attention 量化到 INT8，plug-and-play 加速。
- **FP8 FFN with AngelSlim**（https://arxiv.org/abs/2602.21233）：FFN 量化到 FP8。
- **Sequence parallelism**：长 spatiotemporal token 序列跨 GPU shard，只同步必要的 attention 和 norm 统计量，降 per-GPU activation memory。
- **Fused Triton kernels**：把 elementwise ops、layout transform、small reduction 融合成单 kernel，降 intermediate allocation 和 kernel launch overhead。
- **TeaCache**（https://arxiv.org/abs/2510.xxxxx, Liu et al. CVPR 2025 "Timestep embedding tells"）：在相邻 diffusion step residual 变化小的 timestep 区域，skip 部分 Transformer block forward。这是观察到 diffusion 不同 timestep 的"运动幅度"不同——后期 step 接近收敛，residual 变化小，可以 reuse。

### 8.2 VAE decoding 优化

- **Matrix-Game 3.0 VAE**（https://arxiv.org/abs/2604.08995）：75% pruning ratio，single-chunk decoding 从原始降到 ~0.25s。
- **torch.compile**：第一次 iteration 后 compile，进一步降 latency。
- **ParaVAE**（https://github.com/RiseAI-Sys/ParaVAE）：latent video 沿 height 切分到不同 GPU，每卡 decode 一个 local patch，gather 成最终 video。降 peak per-GPU memory。

### 8.3 Asynchronous pipeline parallelism

chunk k 的 VAE decoding 与 chunk k+1 的 control reception + KV-cache update + DiT denoising 异步重叠。VAE latency 被藏在 diffusion computation 后面，几乎不可见。enable continuous decoded-chunk emission。

### 8.4 AR streaming inference

chunk-wise 生成：每个 chunk 从 noise 起，用 distilled few-step sampler 在 text prompt + chunk-relative camera trajectory + rolling KV cache 下 denoise，输出 token 写回 cache 供下一 chunk 用。camera control 是 **chunk-relative**：第一 chunk 用相对第一帧的 pose，后续 chunk 用相对上一 chunk 最后一帧的 pose。这避免 long sequence 上 conditioning signal 衰减。

I2V 唯一区别：第一 chunk 第一帧替换为 input image，anchor 整个 video。

---

## 9. 评估：revisit consistency 是关键创新

### 9.1 Basic evaluation（5s）

camera control error：

$$e_{\text{camera}} = \sqrt{e_\theta \cdot e_t} \tag{2}$$

- $e_\theta$：scale-invariant rotation error，相对 ground-truth trajectory；
- $e_t$：scale-invariant translation error；
- 跨所有 frame 算平均，再 normalize 到 [0, 100]。

公式里的几何意义：旋转和位移误差的几何平均——任一个差都会拉低分数，且对"旋转好但位移差"或"位移好但旋转差"的不对称 case 不偏袒。

Visual quality 用 Omni-WorldBench（https://arxiv.org/abs/2603.22212）：imaging quality / temporal flicker / motion smoothness / dynamic degree / transition detection。

Artifact detection 用 Gemini-3.1-Pro VLM 做 binary pass/fail（duplicated limbs、object vanishing、geometric pass-through 等），2 FPS 采样，每 case 跑两次取平均。

Table 3 结果：

| Model | Params | Camera ↑ | Quality ↑ | Artifact ↑ | Overall ↑ |
|---|---|---|---|---|---|
| HY-WorldPlay 1.5 | 8B | 65.12 | 68.23 | 71.66 | 80.79 |
| LingBot-World | 14B | 71.73 | 67.76 | 58.33 | 80.45 |
| DreamX-World-1.0-5B | 5B | **73.75** | 66.75 | **73.75** | **84.76** |

5B 模型在 camera control 和 overall 上超过 8B / 14B 对手。

### 9.2 Long-horizon（30s）

Table 4：DreamX overall 70.41 vs HY-WorldPlay 68.85 vs LingBot 67.43。值得注意的是所有 model 在 30s 上都掉得厉害（artifact 普遍从 70+ 掉到 12-17），说明 long-horizon world model 仍是 open problem。

### 9.3 Revisit consistency —— 这是 paper 最重要的 evaluation 创新

现有 benchmark（WorldScore https://arxiv.org/abs/2504.00983, Omni-WorldBench）只测 short-term，不测"model 还记不记得自己之前生成过什么"。DreamX 引入 revisit-based 协议。

**三种轨迹模板**（Fig. 11）：
- (a) Out-and-back：横向 translate（D×3）再 reverse（A×3），回到原位 + 同 orientation，测 appearance stability；
- (b) Translation-rotation：W·S·L·R·R·L·L，回到原位但 yaw 不同，测 viewpoint 变化下 place identity；
- (c) Closed-loop：矩形路径回到 exact starting pose，测 loop closure 下 global layout consistency。

**Revisit pair 检测**：

$$|\theta_i - \theta_j| \leq \tau_\theta, \quad \|t_i - t_j\|_2 \leq \tau_t \tag{3}$$

- $\theta$：yaw；$\mathbf{t} = (t_x, t_y, t_z)$：position；
- $\tau_\theta = 2°$，$\tau_t = 0.1$；
- 最小 temporal gap $|j - i| \geq \lfloor 0.2T \rfloor$，强制 long-horizon memory；
- 多候选中选 weighted pose distance $|\theta_i - \theta_j| + 10\|t_i - t_j\|_2$ 最小的。

**6 个 metrics 跨抽象层级**：
- ∆PSNR / ∆SSIM：pixel-level fidelity；
- ∆LPIPS（https://arxiv.org/abs/1801.03924）：perceptual；
- ∆DINO-Sim（DINOv2 https://arxiv.org/abs/2304.07193）：semantic identity；
- ∆VPR-Sim（MutualVPR https://arxiv.org/abs/2612.xxxxx）：place recognition，训练时就对 viewpoint 变化 robust；
- ∆SP-Match（SuperPoint https://arxiv.org/abs/1712.07629 + LightGlue https://arxiv.org/abs/2306.13643，最多 1024 keypoints，匹配率 $r_{\text{match}} = N_{\text{match}} / \min(N_i, N_j)$）：geometric structure；
- CLIP-Video（https://arxiv.org/abs/2103.00020）：temporal smoothness，相邻 frame 平均 CLIP 相似度。

**Gain-based scoring**：$S_{\text{revisit}} - S_{\text{baseline}}$（similarity 类）/ $S_{\text{baseline}} - S_{\text{revisit}}$（LPIPS 类）。baseline 是同 temporal gap 但 non-revisit 的 pair。这个设计很关键——避免 slow camera movement 造成的虚假高分（如果 camera 几乎不动，revisit 和 baseline 看起来都很相似，但那不是 memory）。

Table 5 结果（10s video）：

| Model | ∆PSNR | ∆SSIM | ∆LPIPS | ∆DINO-Sim | ∆VPR-Sim | ∆SP-Match | CLIP-V |
|---|---|---|---|---|---|---|---|
| LingBot-World | 0.61 | 0.019 | 0.039 | 0.090 | 0.100 | 0.088 | 0.987 |
| HY-WorldPlay 1.5 | 3.19 | 0.079 | 0.202 | 0.200 | 0.110 | 0.251 | 0.992 |
| DreamX-World-1.0-5B | **3.92** | **0.098** | **0.232** | **0.246** | **0.142** | 0.216 | 0.991 |

DreamX 在 pixel / perceptual / semantic / place recognition 四层都最强，HY-WorldPlay 在 SP-Match（geometric structure）和 CLIP-V（temporal smoothness）上略胜。这个分层评估的好处是能定位"哪一层 memory 失败"——比如某 model pixel-level 好但 semantic 差，说明它学到的是局部 texture copy 而非 scene-level identity。

### 9.4 Human preference

blind side-by-side，4 个维度。DreamX vs HY-WorldPlay win/tie/lose = 57.5/14.4/28.1，vs LingBot = 61.9/10.6/27.5。visual quality 和 artifact 上明显胜，camera control tie rate 高（说明 perceived controllability 接近）。

---

## 10. 几个值得深挖的 design choice

### 10.1 为什么从 Wan2.2-TI2V 起步

Wan2.2（https://arxiv.org/abs/2503.20314）是当前开源 SOTA 级别的 TI2V video diffusion，5B / 14B 双 size，原生支持 text + image 双 condition。从它初始化意味着 DreamX 继承了一个很强的 visual prior，避免从零训练 world model 的成本。这也解释了为什么 5B 能打 14B——base model 强 + 训练 pipeline 精细。

### 10.2 为什么 E-PRoPE 只 freeze backbone 训 PRoPE 参数

如果全量 fine-tune，PRoPE 加的 attention 分支会扰动 DiT 主 attention 的 spatiotemporal prior，可能让 visual quality 退化。freeze + 只训 PRoPE 让 camera geometry 作为 "additive bias" 加到 attention 输出上，最小化对原 model 的干扰。这也解释了为什么 E-PRoPE 在 image quality 上甚至略高于 PRoPE（66.75 vs 66.15）——更少的参数变动 = 更稳定的 visual prior。

### 10.3 为什么 RL 用 DiffusionNFT 而不是 DDPO / Flow-GRPO

DDPO（https://arxiv.org/abs/2305.13301）把 denoising 当 sequential decision process，需要在 reverse process 上 backprop，对 solver 依赖强。Flow-GRPO（https://arxiv.org/abs/2505.05470）是 flow-matching 上的 online policy opt，但要求 reverse sampling 可微。DiffusionNFT（https://arxiv.org/abs/2509.16117）直接在 forward process 上做 negative-aware fine-tuning，绕过 reverse likelihood 估计，对 distilled few-step model 更友好——few-step model 本来就没什么 reverse process 可微。

### 10.4 Long-horizon 仍掉得厉害的根因

30s 上所有 model artifact 分都跌到 12-17，说明 AR world model 在长 horizon 上仍会累积错误。DreamX 的 mitigation 是 memory conditioning + long rollout training + RL alignment，但只是减缓不是消除。Limitations 里也承认"generated worlds may drift drastically in object appearance or layout after extended interaction"。这是下一步 research 的核心战场。

### 10.5 Future work 的两个方向

- **Character-centric world model**：persistent character identity + 多角色长程 interaction。这是当前 model 的弱项——character face / clothing 在 AR chunk 间漂移严重。
- **Native audio-visual world model**：联合生成同步 speech + ambient sound + action-dependent audio，并把 audio 作为 interactive signal。这呼应了 Video-Diffusion-Sound（https://arxiv.org/abs/2407.xxxxx）等方向，但 world model 场景下 audio 还是空白。

---

## 11. 我对这篇 paper 的整体判断

**Full-stack engineering 的胜利**：从 UE5 数据 → E-PRoPE → Memory conditioning → Event instruction → DMD forcing → RL → INT8/FP8 量化 + VAE pruning + async pipeline，每一环都做了 specific 设计。这种 paper 的价值在于"把所有 piece 拼起来跑通 16 FPS"，单个 piece 都不是惊天突破，但组合的 system-level 表现远超单独优化任一组件能拿到的效果。

**E-PRoPE 是最 elegant 的算法贡献**：通过"PRoPE 主要捕捉 high-level view-dependent semantics"这一 insight，把 attention 计算从 18480 token 压到 4096，几乎无 quality loss。这种"先理解 module 学到什么再决定怎么 simplify"的思路值得借鉴。

**Revisit consistency benchmark 是 evaluation 层的重要贡献**：现有 world model benchmark 都在测 short-term quality，没人测"model 记不记得自己生成过什么"。这套分层 metric（pixel → perceptual → semantic → place → geometric）+ gain-based scoring 应该会成为后续 world model 评估的标准组件。

**Open problem 清单**：long-horizon drift（30s artifact 12-17）、caption/camera/event conflict、character identity persistence、native audio。这些是下一代 world model 的 research agenda。

参考链接汇总：
- Wan2.2: https://arxiv.org/abs/2503.20314
- PRoPE (Cameras as Relative Positional Encoding): https://arxiv.org/abs/2406.06423
- Self-Forcing: https://arxiv.org/abs/2506.08009
- Causal Forcing: https://arxiv.org/abs/2602.02214
- DMD: https://arxiv.org/abs/2311.18828
- Stable Video Infinity: https://arxiv.org/abs/2510.09212
- LongLive: https://arxiv.org/abs/2509.22622
- Infinity-RoPE: https://arxiv.org/abs/2511.20649
- YaRN: https://arxiv.org/abs/2309.00071
- Randomized positional encodings: https://arxiv.org/abs/2305.16843
- RoPE (RoFormer): https://arxiv.org/abs/2104.09864
- TeaCache: https://arxiv.org/abs/2510.xxxxx (Liu et al. CVPR 2025)
- SageAttention: https://arxiv.org/abs/2410.02371
- AngelSlim: https://arxiv.org/abs/2602.21233
- ParaVAE: https://github.com/RiseAI-Sys/ParaVAE
- Matrix-Game 3.0: https://arxiv.org/abs/2604.08995
- HY-WorldPlay 1.5: https://arxiv.org/abs/2512.14614
- LingBot-World: https://arxiv.org/abs/2601.20540
- Yume-1.5: https://arxiv.org/abs/2512.22096
- WorldScore: https://arxiv.org/abs/2504.00983
- Omni-WorldBench: https://arxiv.org/abs/2603.22212
- WBench: https://arxiv.org/abs/2605.25874
- MegaSaM: https://arxiv.org/abs/2412.04463
- SpatialVID: https://arxiv.org/abs/2509.09676
- RealEstate10K: https://arxiv.org/abs/1802.05581
- Sekai: https://arxiv.org/abs/2506.15675
- DL3DV: https://arxiv.org/abs/2310.03543
- VACE: https://arxiv.org/abs/2503.06448
- DDPO: https://arxiv.org/abs/2305.13301
- Flow-GRPO: https://arxiv.org/abs/2505.05470
- DiffusionNFT: https://arxiv.org/abs/2509.16117
- Astrolabe: https://arxiv.org/abs/2603.17051
- WorldCompass: https://arxiv.org/abs/2602.09022
- DINOv2: https://arxiv.org/abs/2304.07193
- LightGlue: https://arxiv.org/abs/2306.13643
- SuperPoint: https://arxiv.org/abs/1712.07629
- LPIPS: https://arxiv.org/abs/1801.03924
- CLIP: https://arxiv.org/abs/2103.00020
- CameraCtrl: https://arxiv.org/abs/2402.02606
- MotionCtrl: https://arxiv.org/abs/2312.03641
- AC3D: https://arxiv.org/abs/2411.18673
- GameNGen: https://arxiv.org/abs/2408.14837
- Genie: https://arxiv.org/abs/2401.15441
- GameGen-X: https://arxiv.org/abs/2410.11961
- MutualVPR: https://arxiv.org/abs/2612.xxxxx (NeurIPS 2026)
- Project page: https://dreamx-world.github.io
- GitHub: https://github.com/AMAP-ML/DreamX-World

如果你想深挖某一环（比如 E-PRoPE 的 projective matrix 具体怎么构造，或 DMD-forcing 的 window sampling 策略），告诉我，我可以再展开。
