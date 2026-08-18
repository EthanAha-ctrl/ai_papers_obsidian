---
source_pdf: AlayaWorld Interactive Long-Horizon World.pdf
paper_sha256: ecfe162ed101f47d18e3be467ea93cb8b49077e62d390539fd5e6ecfea90ec64
processed_at: '2026-08-18T00:31:12-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# AlayaWorld 用人话讲

想象你要做一个 AI 版的 Minecraft——玩家随便给 camera 指令和 prompt，model 现场生成画面，可以玩很久很久不崩。这就是 AlayaWorld 想做的事。

---

## 1. 为什么这事难？四个 capability 互相打架

你想做 interactive world model，表面上就四件事：

- **Interaction**：玩家给 camera trajectory 和 text prompt，model 得听话
- **Consistency**：你走过去再走回来，场景得长得一样
- **Stability**：生成一小时画面不能越来越糊、越来越偏色
- **Efficiency**：得快，否则不能 interactive

难点在于这四件事互相牵制。你让 interaction 更自由（玩家随时切 prompt），consistency 就难保持。你 roll-out 更长，stability 就崩。你做更激进的 acceleration（少采样步），visual quality 就掉。

大部分 paper 只解决其中一两个，AlayaWorld 想一把全搞定。

---

## 2. 核心 insight：用 memory 替代无限 attention

你想让 model 生成无限长 video，最 naive 的做法是把所有过去帧塞进 attention context。但 context 增长 → compute 线性增长 → 玩不下去。

LLM 用 KV cache 缓解，但 video 的 token 量比 text 大太多，这条路走不通。

AlayaWorld 的思路：**给 model 四种不同时间尺度的 memory，每种干不同的事，compute 恒定**。

这就像人脑——你不需要把这辈子看过的每一帧都装在 working memory 里，你有 sensory memory（刚看到的）、working memory（最近几秒）、episodic memory（去过的地方）、semantic anchor（这地方长啥样）。

---

## 3. 四个 conditioning stream（这是最优雅的部分）

Model 每生成一个 chunk（K=4 latent frames，大概 1 秒），前面拼四个 conditioning stream：

$$
S_i = [s; h_i; g_i; n_i; z_i^\tau]
$$

逐个讲：

### 3.1 Sink $s$ —— "这地方长啥样"

一个 clean latent frame，pin 在 RoPE position 0，跨所有 chunk 不变。作用是告诉 model"我们在哪个 scene"。

**妙招**：训练时 $s$ 从离 target ≥8 latent frames 的 remote frame 采。这样 model 没法直接从 sink extrapolate 下一个 chunk，逼它依赖 camera control signal。如果 sink 太近，model 会偷懒只 copy sink，camera control 就废了。

### 3.2 Temporal memory $h_i$ —— "最近发生了啥"

$$
h_i = H_\phi(w_i), \quad w_i = z_{i-6:i}
$$

- $w_i$: 过去 6 个 latent frame 的 sliding window
- $H_\phi$: history-compression module，把这 6 帧压成 lightweight embedding

干啥用？保持 frame-to-frame 的 local dynamics 连续。人话：别让 chunk boundary 跳变。

### 3.3 Spatial memory $g_i$ —— "我去过那，长这样"

这是 long-horizon consistency 的关键。维护一个 explicit cache：

$$
B = \{(I_j, D_j, \pi_j)\}
$$

- $I_j$: 之前生成的 frame $j$ 的 pixels
- $D_j$: 用 Depth-Anything-3 估计的 per-pixel depth
- $\pi_j$: frame $j$ 的 camera pose

每次生成新 chunk 前，做这件事：

**Step 1**: 从 cache 里 greedy 选最多 10 帧，按"能覆盖 target view 多少 pixel"排序。深度 unproject 到 3D world point，再 project 到 target camera，z-buffer 判可见性。

**Step 2**: Forward splatting warp：

$$
u' = \pi_i(\pi_j^{-1}(u, D_j(u)))
$$

- $u$: source frame 的 pixel
- $\pi_j^{-1}(u, D_j(u))$: unproject 到 3D world point（用 source camera + depth）
- $\pi_i(\cdot)$: project 到 target camera，得 $u'$
- 整体：**2D pixel → 3D point → 2D pixel in new view**

**Step 3**: VAE-encode warped image 到 $g_i$，放 target 的 RoPE 位置。Coverage mask $M_i$ 作为 attention key bias——uncovered region（从没看过的区域）被 ignore，model 只补这些地方的细节，geometric warp 部分直接信。

**Step 4**: 新 chunk 生成完，decode + 估 depth + 存进 $B$。

**为什么这招管用？** Forward splatting 是 classical image-based rendering，几何上是精确的，不需要 model 学。Model 只负责"补 hole 和细节 synthesis"，把可证明正确的几何变换和需要 learned 的生成部分解耦。Revisit 一个地方时，cache 里直接有那地方的图，warp 过来就行，不靠 latent memory"回忆"——回忆容易 hallucinate，warp 不会。

参考 GEN3C 思路：https://research.nvidia.com/labs/toronto-ai/GEN3C/

### 3.4 Nearby frame $n_i$ —— "上一帧长啥样"

$w_i$ 的最后一帧，patch-embedded，放 target 紧前面。这就是 image-to-video conditioning，保 frame-to-frame 全分辨率连续。

| Stream | 时间尺度 | 功能 | 更新频率 |
|--------|---------|------|---------|
| $s$ | 全局 | scene identity anchor | 永不更新 |
| $h_i$ | 6 frames | local dynamics | 每 chunk recompute |
| $g_i$ | 全历史 | revisit consistency | 每 chunk retrieve+render |
| $n_i$ | 1 frame | full-res continuity | 每 chunk slide |

四个加起来 compute 恒定（sink 1 帧 + temporal 6 帧 + spatial 10 帧 warp + nearby 1 帧），不管生成到第 1000 个 chunk 还是第 1 个 chunk，cost 一样。Horizon 原则上 unbounded。

---

## 4. Anti-drift：让 model 见过自己犯错的样子

Long autoregressive 最大的敌人是 **error accumulation**。第 1 个 chunk 的小 error 进第 2 个 chunk 的 context，第 2 个 chunk 在 corrupted context 上又生成带 error 的 chunk，雪球越滚越大。实际表现：brightness 慢慢漂、color temp 偏、细节越来越糊、几何慢慢 drift。

Naive 训练用 GT history（teacher-forcing），inference 时 model 第一次见自己的 imperfect output，OOD，崩。

AlayaWorld 用两招：

### 4.1 Helios drift simulation：手工加 artifact

Latent space 加三种模拟 drift 的 corruption：

$$
z \mapsto (1-\sigma)z + \sigma\epsilon, \quad \sigma \sim \mathcal{U}(0, \rho)
$$

Additive noise → 对应 high-frequency artifact 累积。

$$
z \mapsto \text{up}(\text{down}(z; r)), \quad r \sim \mathcal{U}(0.9, 1)
$$

Down/up-sample blur → 对应细节丢失。

$$
z \mapsto (z - \bar{z})\alpha + \bar{z}, \quad \alpha \sim \mathcal{U}(0.3, 1.7)
$$

Saturation shift → 对应 brightness/color drift。

这三种恰好对应 Table 3 里 brightness consistency / color temp constraint / sharpness retention 三个 metric。训练时 model 见过这些 artifact，inference 时遇到自己生成的 imperfect context 就能 recover。

参考 Helios: https://arxiv.org/abs/2603.04379

### 4.2 Error Bank：存 model 自己的真实 error

更精妙。Helios 是 hand-crafted corruption，不一定 match model 真实 drift 模式。Error Bank 直接采 model 自己的 residual：

$$
\delta = \hat{z}_0 - z_0, \quad \hat{z}_0 = z^\tau - \tau v_\theta
$$

- $z_0$: GT clean latent
- $\hat{z}_0$: model 一步预测的 clean latent（从 noised $z^\tau$ 减去 $\tau$ 倍 velocity $v_\theta$）
- $\delta$: model 实际犯的错

按 chunk length 和 noise level 分桶存进 buffer。训练时 replay：

$$
z \gets z + \gamma\delta
$$

把 model 真实犯过的错加到 context 和 target，让 model 学会 recover。

**直觉**：这比 Helios 强，因为 Error Bank 是 model 自己的真实 error distribution。从 RL 角度看，类似 decision transformer 的 data augmentation——把"假设 model 犯了这个错"的 trajectory 喂回去训练。从 GAN 角度，类似让 discriminator 看到 generator 真实输出，但这里用 supervised 替代 adversarial，更稳定。

Schedule：Error bank warm-up 之前只用 Helios，warm 起来后 Error Bank 优先、Helios 概率降，两者 mutually exclusive per step。Curriculum learning 味道。

参考 Stable Video Infinity: https://arxiv.org/abs/2510.09212

---

## 5. Distillation：30 步压到 4 步

Teacher model 每个 chunk 要 ~30 步采样，太慢不能 interactive。Distill 成 4 步 student。

三个 loss 组合：

### 5.1 DMD (Distribution-Matching Distillation)

$$
\nabla_\theta D_{\text{KL}}(p_{\theta,\tau} \| p_{\text{data},\tau}) = -\mathbb{E}\left[(s_{\text{real}} - s_{\text{fake}})\frac{\partial \hat{z}_i}{\partial \theta}\right]
$$

- $s_{\text{real}}$: teacher score（critic LoRA off）
- $s_{\text{fake}}$: critic score（critic LoRA on）
- **同一个 backbone 通过 LoRA on/off 提供两个 score**，省一个网络
- Critic update 更频繁（two-timescale update rule），让它 stay ahead of student

GAN-style 训练，用 single network 通过 LoRA swap 当两个网络。Critic 跑得快，student 追 critic，critic 追 teacher。

参考 DMD: https://arxiv.org/abs/2405.14881

### 5.2 Self-Forcing++

Student 不在 teacher-forced clip 上 distill（GT context），而是自己 roll-out 多 chunk trajectory，沿 self-generated path 对比 teacher。

**关键**：close train/inference gap。Teacher-forced 训练时 context 是 GT，inference 时 context 是 student 自己生成的 imperfect output，distribution shift 导致 chunk boundary 出 seam。Self-forcing++ 让 student 训练时就见过自己的 drift，inference 时 context OOD 问题缓解。

参考 Self-Forcing++: https://arxiv.org/abs/2510.02283

### 5.3 Consistency Distillation

$$
\mathcal{L}_{\text{cm}} = \mathbb{E}[d(G_\theta(z_i^\tau, \tau | c_i), G_{\theta^-}(z_i^{\tau'}, \tau' | c_i))], \quad \tau' < \tau
$$

- $G_\theta$: student generator
- $G_{\theta^-}$: EMA copy of student
- $d$: Huber distance
- $\tau' < \tau$: 相邻 lower noise level
- 50-level noise grid

相邻 noise level 的 prediction 一致，stabilize few-step solution，抑制 chunk boundary 的 brightness flicker。Consistency model 标准用法。

参考 Consistency Models: https://arxiv.org/abs/2303.01469

总 objective：$\mathcal{L}_{\text{DMD}} + 0.5\mathcal{L}_{\text{cm}}$。Student 是 LoRA on frozen backbone，temporal 和 spatial memory 在这阶段 frozen。

---

## 6. 训练三阶段总结

| Stage | 干啥 | 关键设计 |
|-------|------|---------|
| 1. Bidirectional pre-training | Full-param fine-tune LTX-2.3 backbone (~13B)，注入 domain 知识 | Bidirectional，无 control 和 memory，540p/720p 混合，variable-length up to 20s |
| 2a. History pre-training | Frozen backbone + LoRA 训 $H_\phi$ | Noise masking $\sigma_i \sim \mathcal{U}(0.2, 1)$ 模拟 imperfect history |
| 2b. Full-stack fine-tuning | 全参数训 backbone + history + camera control + next forcing | Camera via AdaLN，next forcing 用 shifted $\tilde\tau = \frac{10\tau}{1+9\tau}$ |
| 3. Post-training acceleration | Distill 30 步 → 4 步 | DMD + Self-Forcing++ + Consistency，LoRA on frozen backbone |

---

## 7. 数据 mix 怎么来的

222k clips，real 和 synthetic 混：

| Source | Type | #Clips | 作用 |
|--------|------|--------|------|
| Sekai-Real | real FPV walking | 21,561 | Photorealism anchor |
| SpatialVid | real indoor | 23,210 | 密集 indoor camera motion |
| RealEstate10K | real indoor | 17,429 | 房产 walkthrough |
| DL3DV | real walkthrough | 7,905 | 长 contiguous multi-view |
| MUGEN† | real FPV | 21,436 | 内部 curated，准确 annotation |
| GameVerse† | synthetic game | 124,116 | **大头**，66s each，controllable camera + action |
| GenEvent† | synthetic event | 6,490 | spell casting, combat 等 action-triggered events |

GameVerse 占大头（124k / 222k ≈ 56%），提供 game-style controllable camera + action。GenEvent 只 6.5k 但提供"魔法事件"的 training signal。Real data anchor photorealism。

Curation pipeline 用 single-decode 设计——每个 clip 一次 NVDEC decode + 一次 RAFT optical flow forward，cache frame-level + flow-derived features，后续所有 rule-based gate 读 cache，millisecond cost。六类 gate：technical / photometric / shot-boundary / motion / text+interface / person control。每新 gate 启用前 profile ~200-clip slice 调 threshold，因为 score distribution 跨 source 差异巨大。

Caption 用 Kimi-K2.6（默认）/ Gemini / Gemma，frames sampled at 1-2 fps 带 [mm:ss] timestamp 鼓励 temporal segmentation。两级 schema：video-level（weather, time of day, location type, camera perspective, camera motion, video style，词汇表 59→26 values）+ segment-level（subject_motion, environment_motion, static_scene, camera_description 分离）。

**关键**：subject motion 和 camera motion 分离标注，让 model 能 disentangle "walking subject + dolly-in camera" vs "orbit camera + turning agent"。

---

## 8. 实验结果的直觉解读

Table 3 里 AlayaWorld 的强项和弱项都很 informative：

**强项**：
- **Brightness consistency 0.9492**（second best HY-World 0.8051，差 14 个点）
- **Color temp 0.9379**（second 0.7819，差 15 个点）
- **Sharpness retention 0.8361**（second 0.6634，差 17 个点）

这三项恰好对应 Helios 的三种 corruption（saturation shift → brightness/color，blur → sharpness）。说明 anti-drift training 极其有效——专门训的 artifact 对应的 metric 大幅领先。

- **Memory symmetry 0.8871**（second 0.8481）
- **Trajectory alignment 0.7018**（second 0.6776）

Spatial memory + camera control via AdaLN 工作。

**弱项**：
- **Image quality 0.6620**（HunyuanVideo 0.7128 最高）

Trade-off 显然：anti-drift training 让 model 学会 recover from corrupted context，slightly 损失 single-frame quality——model 追求"在 corrupted context 下仍 robust"而非"single-frame 完美"。这是合理 trade-off，因为 interactive world model 关心 long-horizon stability 而非 single-frame peak quality。

注意所有结果都是 distilled 4-step student 在 480p 跑的，对比的 baseline 多是各自 default sampling budget。AlayaWorld 用 1/7 的 step 还能在 stability/memory 上大幅领先，说明 design 本身优势明显。

---

## 9. 这套设计的哲学

我觉得 AlayaWorld 最 elegant 的地方在于：**每个 design choice 同时 benefit 多个 objective**。

- Bounded visual context → 既支持 unbounded horizon（efficiency），又通过 spatial memory cache 支持 revisit consistency
- Anti-drift training → 既支持 long-horizon stability，又让 self-forcing++ distillation 更 robust
- 4-step distillation → 既支持 efficiency，又通过 consistency loss 抑制 chunk boundary flicker（stability）
- Sink frame 的 remote distance → 既逼 model 用 camera control（interaction），又防 model 过度依赖单一 anchor（stability）

这种"一石多鸟"的设计是好的系统设计的标志。对比 naive 做法：用无限 attention context 简单但贵且 horizon bounded，纯 latent memory 简单但 revisit 一致性差，纯 teacher-forcing 简单但 inference 崩。

AlayaWorld 用 explicit geometry（depth + pose + forward splatting）+ learned synthesis 分工，把可证明正确的几何部分 externalize，让 model 只学难学的部分。这种 decomposition 思路和 NeRF 把几何和外观分离、Gaussian Splatting 把 representation explicit 化一脉相承。

---

## 10. Limitation 和 future work

Paper 自己承认：model 主要通过 visual observations + estimated geometry + visual memory represent world。Object state（门后面有谁）、physical causality（推一个球会滚多远）、long-term task structure（"先去厨房拿刀再回客厅"）的理解限于 visible consequences。

这指向几个 future direction：
1. **Symbolic state representation**：在 visual memory 之外加 object-state cache，track "门是不是开着"等 hidden state
2. **Action-conditioned physics**：让 model 显式学 physical dynamics，而非只 visual synthesize
3. **Neuro-symbolic hybrid**：visual generation + symbolic world state 联合

参考 GameNGen (https://arxiv.org/abs/2508.14803) 在固定 game 上做 RL agent + diffusion model 的路子，是限定场景版的 AlayaWorld。Sora 这类 bidirectional DiT 是"一次生成整段"范式，AlayaWorld 是"无限滚动"范式，从固定长度到 unbounded horizon 是 fundamental shift。

---

## 11. 一句话总结

**AlayaWorld = video diffusion transformer 的 autoregressive 化 + 分层 bounded memory（sink/temporal/spatial/nearby）+ 显式 geometry cache（depth+pose+warp）替代无限 attention + anti-drift training（Helios+Error Bank）让 model 见过自己的错 + discrete autoregressive distillation（DMD+Self-Forcing+++Consistency）压到 4 步可 interactive**。

把它想成 LLM 的 autoregressive generation 嫁接到 video，但用 geometry-aware retrieval memory 替代 KV cache，让 compute 恒定、horizon 无限、revisit 一致、drift robust、4 步可玩。

Project: https://alaya-lab.github.io/AlayaWorld/
Code: https://github.com/AlayaLab/AlayaWorld
Arena: https://warena.ai/

---

想让我深挖哪块？比如 spatial memory retrieve 算法的 occlusion tolerance $\delta=0.1$ 怎么定的、Error Bank 按 chunk length 分桶的具体逻辑、或 Self-Forcing++ 在 chunk-level rollout 的具体 schedule——告诉我。

---

# AlayaWorld 深度技术讲解

这篇 paper 提出了一个 interactive long-horizon video world model，核心贡献是把 video diffusion transformer 改造成 autoregressive chunk-by-chunk 生成的"世界模拟器"，同时保持 bounded compute cost 实现 unbounded horizon。我会从架构哲学出发逐层 build intuition。

---

## 1. 范式定位：从 LLM autoregression 到 video world model

可以把 AlayaWorld 类比为 video 版的 LLM autoregressive generation。LLM 生成 token 序列 $x_{1:N}$，每步 $p(x_i | x_{<i})$；AlayaWorld 生成 latent chunk 序列 $z_{1:N}$，每个 chunk 是 K=4 latent frames 的 block。区别在于：

- LLM 的 context window 增长时 compute 线性增长（除非用 KV cache）
- AlayaWorld 用 **bounded visual context**，让每步 compute 恒定，horizon 原则上 unbounded

这是关键设计哲学：**用 memory + retrieval 替代无限增长的 attention context**。类似 Ring Attention 或 RAG 的思路，但嫁接到 diffusion transformer 上。

paper 提出四个 tightly coupled capability：**interaction, consistency, stability, efficiency**。这四个不是 independent axis 而是互相 trade-off：
- 更强 interaction（更频繁 prompt 切换）→ consistency 更难保持
- 更长 roll-out → residual error 累积
- 更激进 acceleration（fewer steps）→ visual stability 受损

AlayaWorld 用 unified autoregressive framework 同时解决这四点。

---

## 2. Formulation 详解

### 2.1 Causal factorization (Eq. 1)

$$
p_\theta(z_{1:N} | \pi_{1:N}, y_{1:N}) = \prod_{i=1}^{N} p_\theta(z_i | z_{<i}, \pi_{\le i}, y_i)
$$

变量：
- $z_i$: 第 $i$ 个 latent chunk，K=4 latent frames
- $\pi_i$: chunk $i$ 的 target camera trajectory，是 per-latent-frame 的 absolute camera pose 序列
- $y_i$: chunk-level text prompt，可以在 chunk boundary 切换驱动 prompt-driven action（combat, spell-casting 等）
- $\pi_{\le i}$: 用 ≤ 而不是 <，意味着当前 chunk 的 camera 控制可见
- $z_{<i}$: 过去所有 latent chunks，作为 in-context prefix

关键：**camera trajectory 和 visual past 走两条不同路径**：
- camera trajectory 通过 AdaLN 注入（per-frame relative pose increment）
- visual past 通过 in-context token prefix 注入

这个分离很重要——camera control 是 low-dimensional 的 6-DoF signal，适合用 AdaLN（类似 timestep embedding 的处理方式）；visual past 是 high-dimensional 的 latent，适合做 attention prefix。

### 2.2 Token sequence (Eq. 2)

$$
S_i = [\underbrace{s}_{\text{sink}}; \underbrace{h_i}_{\text{temporal memory}}; \underbrace{g_i}_{\text{spatial memory}}; \underbrace{n_i}_{\text{nearby/I2V}}; \underbrace{z_i^\tau}_{\text{target}}]
$$

变量：
- $s$: sink frame，单帧 clean latent ($\sigma=0$)，patch-embedded，pin 在 RoPE temporal position 0，跨所有 chunk 固定
- $h_i = H_\phi(w_i)$: temporal memory，$w_i = z_{i-L:i}$ 是 sliding window（L=6 latent frames），通过 history-compression module $H_\phi$ 压缩成 lightweight embedding
- $g_i$: spatial memory，geometry-aligned 过去视角 render 到当前 view 的结果
- $n_i$: nearby frame，$w_i$ 的最后一帧，patch-embedded，放在 target 前，承担 image-to-video conditioning
- $z_i^\tau$: noised target chunk，$\tau$ 是 flow-matching timestep，$z_i^\tau = (1-\tau)z_i + \tau\epsilon$

处理方式：**full (non-causal) self-attention**，整个 prefix 一起 attention，最后 slice off prefix 只 denoise target segment。这意味着 conditioning streams 和 target 之间是 bidirectional attention（不是 causal mask 的），这有点反直觉——因为我们说 "autoregressive"，但 within-chunk 是 bidirectional，across-chunk 才是 causal。

### 2.3 四个 conditioning stream 的角色分工

这是我最喜欢的设计点。四个 stream 各自承担不同时间尺度的功能：

| Stream | 时间尺度 | 功能 | 是否更新 |
|--------|---------|------|---------|
| $s$ (sink) | 全局 | 身份/外观 anchor | 固定 |
| $h_i$ (temporal) | 短期（6 frames） | local dynamics, frame-to-frame continuity | 每 chunk recompute |
| $g_i$ (spatial) | 长期 | revisit 一致性，concrete visual evidence | 每 chunk retrieve+render |
| $n_i$ (nearby) | 极短期 | full-res frame-to-frame continuity | 每 chunk slide |

这种分层 design 让我想到人类 memory 的 multiple time scale：working memory (h_i), episodic memory (g_i), semantic anchor (s)。

**Sink frame 的巧妙之处**：训练时 $s$ 是从 remote frame（距离 target ≥8 latent frames）采样，这 prevent 模型直接从 sink extrapolate 下一个 chunk，逼迫它依赖 camera-control signal。这是 data augmentation 的妙用——通过 corruption 让 conditioning signal 真正 informative。

---

## 3. Spatial Memory 详解（GEN3C 风格）

这部分是 AlayaWorld 实现 long-horizon consistency 的核心。引用 GEN3C 的 explicit cache 思路：

### 3.1 Cache 结构

$$
B = \{(I_j, D_j, \pi_j)\}
$$

- $I_j$: 之前生成的 frame $j$ 的 pixels
- $D_j$: frame $j$ 的 monocular depth，用 Depth-Anything-3 估计
- $\pi_j$: frame $j$ 的 camera pose
- 按 global frame index 索引

### 3.2 Rendering pipeline

**Step 1: Retrieve up to 10 frames by greedy maximum-coverage**

每个候选 frame 的 depth unproject 到 world points，project 到 target camera $\pi_i$。用 z-buffer with occlusion tolerance $\delta=0.1$ 标记覆盖哪些 target pixel。Greedy 选 maximum newly-covered pixels 的 frame，直到选满 10 帧。

直觉：这是 view synthesis 中的 standard coverage-based selection，类似 NeRF 或 image-based rendering 中的 keyframe selection。

**Step 2: Forward splatting warp (Eq. 3)**

$$
u' = \pi_i(\pi_j^{-1}(u, D_j(u)))
$$

变量：
- $u$: frame $j$ 的 pixel coordinate
- $D_j(u)$: pixel $u$ 的 depth
- $\pi_j^{-1}(u, D_j(u))$: unproject pixel $u$ 到 3D world point $(X, Y, Z)$
- $\pi_i(\cdot)$: 把 3D world point project 到 target camera $\pi_i$ 的 image plane，得到 $u'$

操作链：**2D pixel → 3D world point → 2D pixel in target view**。这是 classical image-based rendering 的 forward warping。

Occlusion resolution：per-pixel 用 nearest depth，多个 source 融合成一个 warped image $\tilde{I}_i$ 加 binary coverage mask $M_i$。

**Step 3: Inject**

$\tilde{I}_i$ VAE-encoded 到 $g_i$，放在 target 的 RoPE coordinates（不是 position 0，是 target position）。Coverage mask $M_i$ 作为 self-attention key bias，让 uncovered 区域被 ignored 而不是 trusted——这是关键，防止模型 hallucinate unseen region。

**Step 4: Update**

chunk $i$ 生成完毕 → decode 到 pixels → 估计 depth → append $(I_i, D_i, \pi_i)$ 到 $B$。

### 3.3 为什么这比纯 attention memory 好？

如果用 attention memory 存所有过去 frames：
1. Compute cost 随 horizon 线性增长
2. 重新 visit 一个地方时，模型需要从 latent memory 中"想起"几何结构，这很容易 drift
3. Geometry alignment 需要显式 3D reasoning

Spatial memory 用 explicit geometry（depth + pose）做 view synthesis，相当于把 3D structure externalize 到一个 explicit cache，让模型只负责"补全"和"细节合成"，而不是"重建几何"。这种 decomposition 非常重要——它把可证明正确的几何变换（forward splatting 是精确的）和需要 learned 的部分（细节 synthesis）分离。

参考 GEN3C: https://arxiv.org/abs/2506.08948 (大致，具体 reference 没在 paper 里但我从 paper 的描述推断是 GEN3C paper)

---

## 4. 训练三阶段

### Stage 1: Bidirectional Pre-Training

- Base: LTX-2.3 的 22B multimodal model，移除 audio module → ~13B backbone（paper 标题说 15B，可能是后续 fine-tune 增加了 adapter 参数）
- Full-parameter fine-tune
- 24 fps, 540p/720p 混合
- Variable-length clips up to 20s（LTX 训练时的 temporal range）
- Mixed objectives: image-, video-, text-conditioned
- Data: weighted mixture，主导是 balanced scene/camera-pose corpus，加 AAA-gameplay recordings + real FPV walkthroughs + magic-event clips
- Adaptive sigma-shift schedule：flow-matching timestep shift 随 clip length scale
- 短 low-$\sigma$ refinement pass 收尾 sharpen 细节

**Intuition**：这一阶段是给 base video model 注入 domain knowledge（real captures + synthetic game + generated events），让它有 photorealistic appearance 和 controllable motion 的 prior。所有 control 和 memory mechanism 还没引入，保持 bidirectional。

### Stage 2: Autoregressive Training（两个 phase）

**Phase 1: History Pre-Training**

- Backbone frozen
- LoRA adapter 训 history-compression module $H_\phi$
- 训练目标 Eq. (4):

$$
\mathcal{L}_{2a} = \mathbb{E}\|v_\theta(z_\Omega^\tau, \tau | H_\phi(\tilde{z})) - (\epsilon - z_\Omega)\|_2^2
$$
$$
z_\Omega^\tau = (1-\tau)z_\Omega + \tau\epsilon, \quad \tau \sim \mathcal{U}(0,1)
$$

变量：
- $z_\Omega$: target window，长度 varied
- $\tilde{z}_i = (1-\sigma_i)z_i + \sigma_i\epsilon_i$: masked history frame，$\sigma_i \sim \mathcal{U}(0.2, 1)$
- $v_\theta$: rectified-flow velocity field，预测 $\epsilon - z_\Omega$（flow matching 的 velocity）
- $\tau$: flow-matching timestep

**关键设计**：history 用 noise masking（每个 frame 用自己的 noise level $\sigma_i$ 加噪），模拟 inference 时 history 是 model 自己生成的（带 noise 的）而非 GT。这是 self-forcing 的思想雏形。Target window 长度 varied，让 module serve short and long horizons。

**Phase 2: Full-Stack Fine-Tuning**

Starting from history-pretrained weights，full-parameter supervised fine-tune，训练 backbone + history-compression + camera-control + next forcing head 四个 module。

**Camera Control (Eq. 5)**:

$$
c_{\text{cam}} = \text{MLP}\left(\bigoplus_{k=1}^{6} \text{PE}(\Delta\pi_k)\right), \quad e \gets e + c_{\text{cam}}
$$

变量：
- $\Delta\pi_k$: per-frame relative pose increment 的第 $k$ 个分量（6 个：translation xyz + rotation rpy 或 quaternion）
- $\text{PE}$: Fourier embedding（类似 NeRF 的 positional encoding）
- $\bigoplus$: concatenation
- $e$: timestep embedding
- AdaLN scale/shift 从 $e$ 产生

**Intuition**：camera control 用 AdaLN（不是 cross-attention）是因为 camera pose 是 low-dim 6-DoF signal，类似 timestep。Fourier embedding 让 MLP 能 fit 高频 camera motion。

Per-axis scale 从 real data motion statistics 校准——这很关键，因为 walking video 和 game footage 的 motion scale 差异巨大。

**Next Forcing (Eq. 6)**:

$$
\mathcal{L}_{\text{nf}} = \|f_\psi(F, z_0^{+,\tilde\tau}, \tilde\tau) - (\epsilon - z_0^+)\|_2^2
$$
$$
\mathcal{L} = \mathcal{L}_{\text{flow}} + 0.5\mathcal{L}_{\text{nf}}
$$
$$
\tilde\tau = \frac{10\tau}{1+9\tau}
$$

变量：
- $z_0^+$: next chunk（chunk $i+1$）
- $F$: hidden states from several layers hooked from backbone
- $f_\psi$: small head decoding $F$ + noised next chunk 到 velocity
- $\tilde\tau$: shifted higher noise level，$\tilde\tau \geq \tau$ when $\tau \in [0,1]$
- $\mathcal{L}_{\text{flow}}$: 主 chunk 的 flow-matching objective

**Intuition**：next forcing 让 backbone 在生成 chunk $i$ 的同时，hidden states 也 encode chunk $i+1$ 的预测。这强化了 frame-to-frame causal continuity。Shifted higher noise level $\tilde\tau$ 是 trick：让 next chunk 在更高 noise level 监督，避免 next forcing 和 main flow 争夺相同 noise level 的监督信号。

参考 Next Forcing paper: https://arxiv.org/abs/2606.11187 (这个 arXiv ID 是 paper 里的，2026 年的 paper)

---

## 5. Anti-Drift Training（关键创新）

Long autoregressive roll-out 最大的问题是 **error accumulation**。每一步的 residual error 进下一步的 context，drift 越来越严重——brightness shift, color drift, blur accumulation, geometric drift。

AlayaWorld 用两个机制训练模型 tolerate corrupted past：

### 5.1 Helios Drift Simulation

在 latent space 加三种 artifact，模拟 roll-out drift 到的样子：

1. **Additive noise**: $z \mapsto (1-\sigma)z + \sigma\epsilon$, $\sigma \sim \mathcal{U}(0, \rho)$
2. **Down/up-sampling blur**: $z \mapsto \text{up}(\text{down}(z; r))$, $r \sim \mathcal{U}(0.9, 1)$
3. **Saturation shift**: $z \mapsto (z - \bar{z})\alpha + \bar{z}$, $\alpha \sim \mathcal{U}(0.3, 1.7)$

Noise 和 blur step 后可选接 saturation step。

**Intuition**：这三种 artifact 对应实际 drift 的 failure modes——noise 对应 high-frequency artifact 累积，blur 对应 detail loss，saturation shift 对应 brightness/color drift。让模型在训练时见过这些 artifact，inference 时遇到自己生成的 imperfect context 就能 recover。

### 5.2 Error Bank

更精妙的设计。模型存自己的 reconstruction residuals：

$$
\delta = \hat{z}_0 - z_0, \quad \hat{z}_0 = z^\tau - \tau v_\theta
$$

变量：
- $\hat{z}_0$: model 的预测 clean latent（从 noised $z^\tau$ 通过 velocity $v_\theta$ 一步推断）
- $z_0$: GT clean latent
- $\delta$: 残差，model 实际犯的错误

Buffer 按 chunk length 和 noise level bucketed。训练时 replay 这些 residual：

$$
z \gets z + \gamma\delta
$$

加到 context 和 target latent。让模型 learn to recover from 自己实际产生的 failure modes。

**Intuition**：这比 Helios 强，因为 Helios 是 hand-designed artifact 分布，Error Bank 是 model 自己的真实 error distribution。从 RL 的角度，这类似 decision transformer 的 data augmentation——把"如果 model 犯了这个错"的 trajectory 喂回去训练。从 GAN 的角度，这类似 feedback loop 让 discriminator 看到 generator 真实输出。

参考 Stable Video Infinity (Error Bank 概念): https://arxiv.org/abs/2510.09212

### 5.3 Scheduling

Error bank warm-up：bank 没填之前只用 Helios（fixed probability）；warm up 完成后 Error Bank 优先，Helios 概率降低，两者 mutually exclusive per step。

这种 scheduling 类似 curriculum learning——先用 synthetic artifact 热身，再用真实 error 精调。

---

## 6. Post-Training Acceleration: Discrete Autoregressive Distillation

teacher model（~30 steps per chunk）太慢不能 interactive。Distill 成 4-step student。

### 6.1 三个 loss 组合

**Distribution-Matching Distillation (DMD)**

让 student 分布 $p_{\theta,\tau}$ 匹配 data 分布 $p_{\text{data},\tau}$，通过 score-difference gradient (Eq. 7)：

$$
\nabla_\theta D_{\text{KL}}(p_{\theta,\tau} \| p_{\text{data},\tau}) = -\mathbb{E}\left[(s_{\text{real}}(\hat{z}_i^\tau, \tau | c_i) - s_{\text{fake}}(\hat{z}_i^\tau, \tau | c_i))\frac{\partial \hat{z}_i}{\partial \theta}\right]
$$

变量：
- $\hat{z}_i$: student 自己 self-rollout 出的 chunk
- $s_{\text{real}}$: teacher score（critic LoRA off）
- $s_{\text{fake}}$: critic score（critic LoRA on）
- 同一个 score backbone，通过 LoRA swapping 实现两个 score

**关键 trick**：用同一个 backbone 通过 LoRA on/off 提供 real/fake score，省一个网络。Critic update 更频繁（two-timescale update rule），让它 stay ahead of student。这是 GAN-style 训练但用 single network。

参考 DMD: https://arxiv.org/abs/2405.14881 (Improved DMD)

**Self-Forcing++**

Student 不在 teacher-forced clip 上 distill，而是 roll-out 自己的多 chunk trajectory，沿 self-generated path 对比 teacher。

**Intuition**：这 close 了 train/inference gap。如果只在 teacher-forced clip 上 distill，student 没见过自己的 error distribution，inference 时 student 自己 rollout 出的 context 会 OOD。Self-forcing++ 让 student 在训练时就见过自己的 drift，类似 self-forcing 在 long video generation 的成功。

参考 Self-Forcing++: https://arxiv.org/abs/2510.02283

**Consistency Distillation (Eq. 8)**

$$
\mathcal{L}_{\text{cm}} = \mathbb{E}[d(G_\theta(z_i^\tau, \tau | c_i), G_{\theta^-}(z_i^{\tau'}, \tau' | c_i))], \quad \tau' < \tau
$$

变量：
- $G_\theta$: student generator
- $G_{\theta^-}$: EMA copy of student
- $d$: Huber distance
- $\tau' < \tau$: 相邻 lower noise level
- 50-level noise grid

**Intuition**：consistency loss 让相邻 noise level 的 prediction 一致，stabilize few-step solution，suppress chunk boundary 的 brightness/appearance flicker。这是 consistency model 的标准用法。

参考 Consistency Models: https://arxiv.org/abs/2303.01469

### 6.2 总 objective

$$
\mathcal{L}_{\text{DMD}} + 0.5\mathcal{L}_{\text{cm}}
$$

Student 是 LoRA on frozen backbone，temporal 和 spatial memory 在这一阶段 frozen。

最终：4 sampling steps per chunk，相同 24 fps output，full camera control + temporal + spatial memory，inference cost 是 teacher 的一小部分。

---

## 7. 数据 Pipeline

### 7.1 数据 mixture (Table 1)

| Source | Type | #Clips |
|--------|------|--------|
| Sekai-Real (walking) | real, FPV | 21,561 |
| SpatialVid | real, indoor | 23,210 |
| RealEstate10K | real, indoor | 17,429 |
| DL3DV | real, walkthrough | 7,905 |
| MUGEN† | real, FPV | 21,436 |
| GameVerse† | synthetic, game | 124,116 |
| GenEvent† | synthetic, event | 6,490 |
| **Total** | | **222,147** |

**Intuition**：GameVerse 是大头（124k clips, 66s each），提供 game-style controllable camera + action。GenEvent 只 6,490 clips 但提供 action-triggered events（spell casting, combat）的 training signal。Real data 提供 photorealism anchor。

### 7.2 Curation Pipeline

三阶段：ingest → run → select。

**Shared feature cache**: 单 decode 设计，每个 clip 一次 NVDEC decode + 一次 RAFT forward。Cache frame-level summaries（luminance, frame diff, border）和 flow-derived features（flow-per-second, temporal variance, directional variance）。所有 rule-based gate 读 cache，millisecond cost。

**六类 gate**：

1. Technical validation: decode integrity, ≥720p, ≥3s, 24-65fps, H.264/HEVC/AV1
2. Photometric validation: extreme exposure, black border ratio ≤0.10
3. Shot-boundary filtering: classical cut/dissolve + OmniShotCut，确保 single continuous shot
4. Motion analysis: 去静态 clip，camera-motion bucket (static/pan/gameplay/mixed)，pose-free camera shake detection
5. Text & interface suppression: EasyOCR-based text detection + pixel-stability UI mask, overlay ratio ≤0.04
6. Person control: YOLO11 检测，bound 前景 human count + screen occupancy

Pose stability gate: trajectory jitter, peak acceleration, median reconstruction residual, long-horizon drift。

Perceptual quality: COVER, VBench, SigLIP2/CLIP/V-JEPA2 embeddings → optional global rank-cut + near-duplicate dedup。

**Calibration**: 每个新 gate 启用前，profile ~200-clip slice 调 threshold，因为 score distribution 跨 capture condition 差异大。这让 filtering 严格但保留 source-specific motion statistics。

### 7.3 Hierarchical Caption Annotation

两级 schema：

**Video-level context**: weather, time of day, location type, camera perspective, camera motion, video style。词汇表 59 → 26 values，可作为 conditioning token + data balancing key。

**Segment-level tracks** (Table 2):

| Field | Type | Role |
|------|------|------|
| subject_motion | free text | primary-agent motion |
| environment_motion | free text | entity/lighting/weather dynamics |
| static_scene | free text | time-invariant scene attributes |
| camera_description | free text | viewpoint/framing/motion/stability |
| full_prompt | free text | fused caption for text encoder |
| short_prompt | free text | compact caption for dropout/augmentation |
| camera_path | enum (16) | discrete camera-trajectory control target |

**关键设计**：subject motion 和 camera motion 分离。这让 annotation 能区分"walking subject + dolly-in camera"vs"orbit camera + turning agent"vs"static subject + panning viewpoint"。如果用一个 entangled sentence，model 无法 disentangle。

VLM backend: Kimi-K2.6 默认，Gemini/Gemma 备选。Frames sampled at 1-2 fps with explicit [mm:ss] timestamp，鼓励 temporal segmentation 而非 collapsed description。

---

## 8. 实验结果 (Table 3)

### 8.1 iWorld-Bench

| Metric | Cosmos | HunyuanVideo-1.5 | WAN 2.2 | YUME 1.5 | Matrix-Game 2.0 | HY-World 1.5 | **AlayaWorld** |
|--------|--------|------|------|------|------|------|------|
| **Generation Quality** | | | | | | | |
| Image Quality | 0.6778 | 0.7128 | 0.5545 | 0.6232 | 0.4851 | 0.6675 | 0.6620 |
| Brightness Consistency | 0.6952 | 0.7027 | 0.3886 | 0.3810 | 0.2963 | 0.8051 | **0.9492** |
| Color Temp. Constraint | 0.7170 | 0.7477 | 0.3411 | 0.4165 | 0.2937 | 0.7819 | **0.9379** |
| Sharpness Retention | 0.4363 | 0.5545 | 0.3428 | 0.4023 | 0.4149 | 0.6634 | **0.8361** |
| **Trajectory Following** | | | | | | | |
| Motion Smoothness | 0.9907 | 0.9908 | 0.9557 | 0.9765 | 0.9848 | 0.9921 | **0.9924** |
| Trajectory Accuracy | 0.4955 | 0.6844 | 0.6514 | 0.7113 | 0.7008 | 0.7472 | **0.7985** |
| **Memory Ability** | | | | | | | |
| Memory Symmetry | 0.3738 | 0.6336 | 0.4480 | 0.5276 | 0.3311 | 0.8481 | **0.8871** |
| Trajectory Alignment | 0.6419 | 0.6449 | 0.5703 | 0.5988 | 0.6362 | 0.6776 | **0.7018** |

**关键观察**：

1. **Brightness consistency (0.9492)** 和 **Color temp (0.9379)** 远超 second best (HY-World 0.8051/0.7819)。这验证 anti-drift training 极其有效——saturation shift artifact 训练直接对应这两项 metric。

2. **Sharpness retention (0.8361)** 远超 second best (HY-World 0.6634)。这验证 Helios 的 blur artifact 训练有效。

3. **Memory symmetry (0.8871)** 远超 second best (HY-World 0.8481)。这验证 spatial memory + revisit consistency 设计有效。

4. **Image quality (0.6620)** 不是最好（HunyuanVideo 0.7128）。Trade-off：anti-drift 训练可能 slightly 损失 single-frame 质量，因为 model 学会 recover from corrupted context 而非追求 single-frame 完美。

5. **Trajectory accuracy (0.7985)** best，camera control via AdaLN 工作。

### 8.2 WorldMark Arena

Public Elo ratings at https://warena.ai/，across Visual Quality, Control Alignment, World Consistency。

---

## 9. 关联与思考

### 9.1 与 GameNGen 的关系

GameNGen (https://arxiv.org/abs/2508.14803) 训练 RL agent 玩 DOOM 并用 diffusion model 生成 next frame，但限于固定 game + finite context。AlayaWorld 是 general world model 不限 game，并且用 bounded context 实现 unbounded horizon。

### 9.2 与 Sora 类 model 的关系

Sora 用 bidirectional DiT 一次性生成整段 video，长度有限。AlayaWorld 是 autoregressive chunk-by-chunk，原则 unbounded。这是 fundamental paradigm shift——从"一次生成"到"无限滚动"。

### 9.3 与 LeCun JEPA 的关系

Paper 用 V-JEPA2 embeddings 做 near-duplicate dedup。JEPA 哲学是 predict in latent space，不直接生成 pixels。AlayaWorld 直接生成 pixels，但 spatial memory 的 latent cache 部分类似 JEPA 的 latent world model 思想——把 world state encode 在 latent + geometry cache 中，而非 raw pixels。

### 9.4 Memory hierarchy 与 cognitive science

四层 conditioning stream 让我联想到 Atkinson-Shiffrin model：
- Sensory memory: $n_i$ (nearby frame)
- Short-term/working memory: $h_i$ (temporal, 6 frames)
- Long-term/episodic memory: $g_i$ (spatial memory cache)
- Semantic anchor: $s$ (sink frame)

这种分层让 model 不需要无限 attention context，类似人脑不需要全部记住 past experience。

### 9.5 与 LLM KV cache 的类比

LLM 推理时 KV cache 增长，compute 线性增长。AlayaWorld 的 spatial memory cache 也增长，但 compute 不增长——因为 retrieve 只选 10 frames render 到 target view，固定 compute。这类似 sparse attention / retrieval-augmented LLM 的思路。

### 9.6 Limitation

Paper 自己承认：model 主要通过 visual observations + estimated geometry + visual memory represent world。Object state, physical causality, long-term task structure 理解限于 visible consequences。

这意味着 model 不能 simulate hidden physics（比如一扇门后面有谁），不能 track object state change across long horizon unless visible。这是 future work——可能需要 symbolic state representation 或 neuro-symbolic hybrid。

### 9.7 Why 15B 而 base 是 13B?

Paper 说 LTX-2.3 是 22B multimodal model，移除 audio module 后 ~13B video DiT backbone。但标题说 15B。可能：
- Fine-tune 增加 camera-control module + history-compression + next forcing head + spatial memory encoder，这些 adapter 加起来 ~2B
- 或者 paper 的数字略 approximate

### 9.8 关于 video VAE 的 causal chunk

Paper 说 "causal video VAE encodes a clip into a latent sequence partitioned into chunks $\{z_1, z_2, \ldots\}$"。Causal VAE 保证 chunk boundary 不会有 leak——chunk $i$ 的 latent 只依赖 pixels up to chunk $i$。这关键，否则 Eq.(1) 的 causal factorization 在 latent space 破坏。

参考 LTX-Video: https://arxiv.org/abs/2501.00103

### 9.9 Flow matching vs DDPM

Paper 用 rectified flow / flow matching formulation：$z^\tau = (1-\tau)z + \tau\epsilon$, velocity target $v = \epsilon - z$。这比 DDPM 简单且 sample path 更直，few-step distillation 更友好。$\tau \sim \mathcal{U}(0,1)$ 是 uniform timestep，不是 DDPM 的 importance sampling。

### 9.10 关于 sink frame distance

训练时 sink 是 ≥8 latent frames 远的 remote frame。8 latent frames × 4 frames/chunk × (fps factor) ≈ 数秒。这距离够远让 model 不能直接 extrapolate，但够近还在同 scene。如果太远（跨 scene）就失效。

### 9.11 关于 coverage mask bias

$M_i$ 作为 self-attention key bias——uncovered region 被 ignored 而非 trusted。这防止 model 用 warped image 的 hole 区域当 conditioning。实现细节：可能是 additive bias 加到 attention logits，uncovered 的 key 加大 negative bias 让 attention weight →0。

### 9.12 Error bank vs GAN discriminator

Error bank 和 GAN discriminator 都用 model 自己的输出训练，但：
- GAN discriminator：判断 real vs fake，gradient 通过 backprop
- Error bank：直接把 model 的 residual δ 加到 training input，supervised learning

Error bank 更类似 decision transformer 的 data augmentation，而非 adversarial training。简单且稳定。

---

## 10. 总结

AlayaWorld 的核心贡献是把 video diffusion transformer 改造成 **interactive long-horizon world model**，通过：

1. **Bounded visual context** (sink + temporal + spatial + nearby) 替代无限 attention context，让 compute constant, horizon unbounded
2. **Spatial memory with explicit geometry** (depth + pose + forward splatting) 实现 revisit consistency
3. **Anti-drift training** (Helios + Error Bank) 让 model tolerate corrupted past
4. **Discrete autoregressive distillation** (DMD + Self-forcing++ + Consistency) 把 30 steps 压到 4 steps

这套 design 把"autoregressive generation + bounded memory + geometry-aware retrieval + drift-aware training + few-step distillation"组合成 unified framework，每个 design choice 同时 benefit 多个 objective（bounded context 支持 interaction 和 memory；anti-drift 支持 stability；distillation 支持 efficiency）。

Project page: https://alaya-lab.github.io/AlayaWorld/
Code: https://github.com/AlayaLab/AlayaWorld
Video: https://www.youtube.com/watch?v=n0jIEg7taTI
WorldMark Arena: https://warena.ai/

---

如果你想让我深入某个具体部分（比如 spatial memory 的 retrieve 算法、Helios 三种 corruption 的 latent space 效果、DMD 的 LoRA swapping 细节、或 chunk boundary 的 flicker 抑制），告诉我。
