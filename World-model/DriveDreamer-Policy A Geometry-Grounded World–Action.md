---
source_pdf: DriveDreamer-Policy A Geometry-Grounded World–Action.pdf
paper_sha256: 3ff851e73e9a647a8cb8ffe764b6a58d8c9a2d845e54497e0f371a492cd63fd4
processed_at: '2026-08-18T06:58:00-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好嘞 Karpathy，咱们抛开那些学术黑话，用最直白的人话把这篇 paper 捋一遍。

目前自动驾驶模型主要有两条路。一条叫 VLA，直接把摄像头画面和指令丢给 LLM，吐出方向盘动作，好处是快，但是缺乏对未来的预判，遇到盲区容易翻车。另一条叫 World Model，专门预测未来的视频画面，能做仿真数据，但是通常没法直接用来开车。大家想把这两者缝合成 World-Action Model (WAM)。但是现有的 WAM 大多只盯着 2D 画面，缺了 3D 几何感。

DriveDreamer-Policy 的核心 insight 非常直白：把 Depth 拉进来做骨架。开车本质上是个 4D 物理过程，如果你只给 AI 看 RGB 视频，它很难搞懂前面那辆车离我有多远、哪里是障碍物。Depth map 正好就是纯几何的“距离图”。

他们的架构像个工厂流水线， LLM 作为大脑，吃进去 multi-view 图像、language 指令和 current action。大脑里挂着一排固定数量的 query tokens（类似任务分配卡）。因为有固定数量，所以系统可以随时插拔模块，这就是 paper 里说的 controllable latency。

大脑把这些卡分发给三个 diffusion expert 小弟：
1. Depth Generator：画当前场景的 3D 深度图。
2. Video Generator：画未来的 RGB 视频。
3. Action Generator：算未来的行车轨迹。

最精髓的设计在于 causal attention mask（单向信息流）。大脑规定：Video 小弟可以偷看 Depth 小弟的草稿，所以画出来的未来视频自带 3D 边界感，不会画面糊在一起。Action 小弟可以同时看 Depth 和 Video 的草稿，所以规划路线时既懂当前的 free space，又懂未来会不会撞车。这种 Depth -> Video -> Action 的单向流动，正好符合从 3D 到 2D 再到 1D（轨迹点）的物理降维过程。

为什么选 Depth 而不是 BEV occupancy？因为 Depth 是单通道，维度低，不需要学复杂的 VAE，而且现在有 Depth Anything 3 这种 foundation model 可以直接打伪标签，白嫖 supervision。

三个小弟都使用 Flow Matching 训练。因为 path 是直线，velocity 是常数，训练比传统 DDPM 稳定很多，推理时跑几步 Euler 就行。

工程上最大的好处是 modularity。在车上跑的时候，如果赶时间，只跑 LLM + Action Generator 就行。如果要做 simulation 或者需要前瞻预判，把三个 Generator 全开，就能生成带深度的未来视频和轨迹。

结果非常 solid：Navsim v2 上直接比 VLA SOTA 高了 2.6 分。生成的未来视频质量也大幅度提升，FVD 降了 30 多分。这证明了一个直觉：让模型先算清楚几何，再去想象未来画面，最后做决策，这条路比直接从 2D 像素里瞎猜要靠谱得多。

对你平时推的 autoregressive world model 路线来说，这其实是一个变体。这里 LLM 不去预测 next token，而是输出一堆 query embedding，然后让 diffusion 模型在原生的连续空间里做生成。好处是 depth 这种东西用 token 表达太浪费，用像素空间 diffusion 生成保真度更高。这就是这篇 paper 最核心的 trade-off。

参考链接：
- Project: https://drivedreamer-policy.github.io/
- Depth Anything V3: https://arxiv.org/abs/2511.10647
- Flow Matching: https://arxiv.org/abs/2210.02747

---

# DriveDreamer-Policy 深度技术解析

Andrej, 这篇 paper 非常对你的胃口 — 它把 world model 和 VLA policy 用一种 DETR-style 的 query interface 缝合在一起, 并且把 monocular depth 作为 geometric scaffold 显式插入到 information flow 中。我会从 architecture、math、training dynamics 三个层面 build your intuition, 顺便把相关的 lineage 串起来。

---

## 1. 核心立论 — 为什么需要 "Geometry-Grounded" WAM?

作者的核心 claim 是: 现有的 World-Action Models (WAM) 大多把 "world" 实现为 image/video prediction 或者 latent rollout, **缺了 geometric grounding**。而 autonomous driving 本质上是一个 4D physical process (3D geometry evolves over time)。所以一个 actionable world model 应当:

- synthesize appearance (2D)
- preserve geometric structure (3D)
- 用几何约束 occlusion reasoning、distance estimation、physically consistent motion

这里的关键 insight 是: **depth 是一个 compact、geometry-tied、可以直接 query 的 modality**。RGB video 是高维的、生成 expensive;BEV occupancy 又需要复杂的 sensor calibration 与 annotation;depth 单图 256×144 就够用, 而且 foundation model (Depth Anything V2/V3) 现在可以直接 pseudo-label 大规模无标注数据,免去了从零训 depth estimator 的成本。

直觉上, depth 像是 scene 的 "骨架",video 是 "皮肤",action 是 "意图投影到 4D 轨迹"。三者按 causal 顺序级联,就形成了 paper 中的 **depth → video → action** single-pass information flow。

参考链接:
- Project page: https://drivedreamer-policy.github.io/
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- Depth Anything 3: https://arxiv.org/abs/2511.10647
- UniDepth (metric depth): https://arxiv.org/abs/2403.18913

---

## 2. 整体架构图解析

```
┌──────────────────────────────────────────────────────────────┐
│ Inputs:                                                       │
│   - Language instruction  → text tokens (via LLM tokenizer)    │
│   - Multi-view RGB (×3)   → visual patch tokens (vision enc.) │
│   - Current action       → action tokens (2-layer MLP+LN)    │
│                                                              │
│ Appended learnable queries (固定 slots):                     │
│   [Depth Queries: 64] → [Video Queries: 64] → [Action Q: 8]  │
│                                                              │
│              ↓ All tokens → LLM (Qwen3-VL-2B)                │
│                                                              │
│   Causal mask across query groups:                          │
│   depth_q ──attends──→ video_q ──attends──→ action_q         │
│   (no reverse attention)                                     │
└──────────────────────────────────────────────────────────────┘
                  ↓            ↓             ↓
            World Depth    World Video   World Action
            Embeddings     Embeddings    Embeddings
                  ↓            ↓             ↓
              ┌──────┐    ┌────────┐    ┌─────────┐
              │Depth │    │Video   │    │Action   │
              │ Gen  │    │Gen     │    │Gen      │
              │(pix) │    │(latent)│    │(diff.)  │
              └──────┘    └────────┘    └─────────┘
                  ↓            ↓             ↓
              Depth map   Future video   Trajectory
              (256×144)   (9 frames)     (x,y,cosθ,sinθ)
```

这里设计上的精髓:

1. **Fixed-size query bottleneck** — LLM 永远看见相同数量、相同位置的 query slots。这避免了 sequence length 变化造成的 KV-cache 抖动,允许 depth/video/action heads 随时 "插拔",实现 modularity & controllable latency。这与 DETR object queries (https://arxiv.org/abs/2005.11972) 在 philosophy 上完全一致, 只是 object queries 用于 detection, 这里的 queries 用于 **multi-modal generative readout**。

2. **Causal cross-group mask** — 让 information flow 沿着 3D → 2D → 1D 的几何抽象层次自然下降。video queries 可以读 depth (occlusion, boundary), action queries 可以读 depth + video (free-space + future dynamics)。这是一种 "geometry-conditioned imagination-conditioned planning" 的因果链。注意这是 **single-pass**, 没有 EM-style iterative refinement, 也没有 cross-branch synchronization, 训练和推理都极其简单。

3. **LLM 的角色**: Qwen3-VL-2B 在这里不生成 action token (与 AutoVLA 不同), 也不生成 image token (与 DriveVLA-W0 不同)。它只输出 **world embeddings 和 action embeddings**, 作为下游 generative experts 的 cross-attention key/value。LLM 是 perception + reasoning engine, 生成的是 **latent conditions**, 不是 direct outputs。

Qwen3-VL 参考: https://arxiv.org/abs/2511.21631

---

## 3. 数学细节 — Flow Matching 公式拆解

paper 用的是 conditional flow matching (Lipman et al. 2023), 而非 DDPM 的 Markov forward/reverse。所有三个 generative experts 共享同一个 FM 训练原则。

### 公式 (1): Linear interpolation path

$$
x_t = (1 - t) \cdot x_0 + t \cdot x_1, \quad t \sim \mathcal{U}(0, 1)
$$

变量解释:
- $x_0 \sim p_{\text{data}}$: 一个真实数据样本 (例如一张 clean depth map, 或一段 clean video latent, 或一条 expert trajectory)
- $x_1 \sim p_{\text{noise}}$: 一个噪声样本 (Gaussian)
- $t \in [0, 1]$: 连续时间变量,均匀采样
- $x_t$: 在 data 和 noise 之间的线性插值

直觉: 当 $t = 0$ 时 $x_t = x_0$ (clean data), 当 $t = 1$ 时 $x_t = x_1$ (pure noise)。这条 path 是 **直线路径**, 比 DDPM 的 Markov chain 简单很多, 而且 ODE 数值积分时 step size 可以自由选择 (用 Euler、Heun、DPM-Solver 都行)。

### 公式 (2): Velocity regression loss

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_0, x_1, t} \left[ \left\| v_\theta(x_t, t \mid c) - (x_1 - x_0) \right\|_2^2 \right]
$$

变量解释:
- $v_\theta(\cdot)$: 一个时间依赖的 velocity field,由 neural network 参数化 (参数 $\theta$)
- $c$: conditioning information (在 DriveDreamer-Policy 里就是 LLM 输出的 world embeddings 通过 cross-attention 注入)
- $\dot{x}_t = x_1 - x_0$: 目标 velocity,即沿直线 path 的导数 (常数)
- $v_\theta$ 试图回归到这个常数 velocity

直觉: 这就是一个 **regression-to-constant-velocity** 问题。网络只需要学会 "在当前 $x_t$、当前 $t$、给定 condition $c$ 下, 朝哪个方向走"。因为 path 是直的, velocity 是常数, 训练目标比 DDPM 的 $\epsilon$-prediction (噪声预测) 更 stable。

### Inference

从 $t = 1$ (纯噪声) 开始, 反向积分 ODE:
$$
\frac{dx_t}{dt} = v_\theta(x_t, t \mid c), \quad t: 1 \to 0
$$

得到的 $x_0$ 就是采样结果。用 5 步或 10 步 Euler 就能拿到不错的结果 (flow matching 在小 NFE 下比 DDPM 强很多)。

Flow Matching 原始 paper: https://arxiv.org/abs/2210.02747

---

## 4. Depth Generator 细节

Depth generator 是一个 **pixel-space diffusion transformer**, 从 PPD (Pixel-Perfect Diffusion, Xu et al. 2025, https://arxiv.org/abs/2510.07316) 初始化。

训练流程:
1. 取 ground-truth depth (用 Depth Anything 3 pseudo-label)
2. 用 log transform + per-map percentile 归一化到 $[-0.5, 0.5]$
3. 采样 flow time $t$, 把 depth 加噪: $x_t = (1-t) \cdot d_0 + t \cdot \epsilon$
4. Denoiser 输入: **concatenation of $\{x_t, \text{RGB}\}$**
5. Denoiser 通过 cross-attention 注入 LLM 的 depth-query embeddings 作为 global semantic context
6. 回归 velocity $v_\theta = \epsilon - d_0$

为什么 pixel-space?
- Depth 维度低 (单通道, 256×144 = 36864 像素)
- 边界 fidelity 重要 (sharp depth discontinuities at object edges)
- 不需要学一个 VAE codec (latent-space 的 codec 在 boundary 处会 blur)

为什么 cross-attention on LLM depth embeddings?
- 解决 monocular depth 的 inherent ambiguity (纹理缺失区域、镜面反射、远距离)
- LLM 看到的是 multi-view + language + action context, 能 "全局约束" 局部 depth
- 这也是为什么 Table 3(b) 显示 DriveDreamer-Policy 的 AbsRel (8.1) 比 PPD zero-shot (18.5) 和 PPD fine-tuned (9.3) 都更低

---

## 5. Video Generator 细节

Video generator 从 **Wan-2.1-T2V-1.3B** (https://arxiv.org/abs/2503.20314) 初始化, 改造为 image-to-video 任务。

架构组成:
1. **VAE encoder**: 把当前 RGB 压成 latent representation (空间维度降低 ~8×)
2. **Noisy video latents**: 对 9 帧 target horizon 初始化噪声
3. **Diffusion transformer backbone**: DiT-style (https://arxiv.org/abs/2212.09748)
4. **Cross-attention conditioning**: 用 LLM 的 video-query embeddings 而非 text embeddings
5. **CLIP visual condition**: 从当前 frame 提取 CLIP feature,与 video embeddings concat 后注入

设计要点:
- 用 **world video embeddings** 替代 standard text-to-video 中的 text embedding — 这把 multi-view perception + language intent + action context + upstream depth cues 一起注入
- CLIP 当前帧特征保证 **appearance/identity/camera content** 一致性, 避免生成的 future frame "漂移"
- 9 帧 @ 2Hz 对应 4 秒 lookahead horizon,与 action generator 的轨迹 horizon 对齐

训练分辨率: 144 × 256 (降低 compute/memory)

---

## 6. Action Generator 细节

Action generator 是 **standalone diffusion transformer**, 输入 noise trajectory, 输出 feasible future action sequence。

### Trajectory parameterization

$$
\text{state} = (x, y, \cos\theta, \sin\theta)
$$

为什么这个表示?
- 避免 angular wrap-around: $\theta = 0$ 和 $\theta = 2\pi$ 是同一个 heading, 但数值上差很大, 用 cos/sin 分量把角度嵌入到单位圆上, 网络不需要学 periodicity
- Smooth turn dynamics: cos/sin 对 $\theta$ 的导数连续
- 来自 Zhou et al. SmartRefine (CVPR 2024, https://arxiv.org/abs/2402.17179) 的设计

### Cross-attention conditioning

action queries 的 LLM embeddings 聚合了:
- Instruction semantics
- Multi-view observations
- **Upstream depth cues** (free space, distance-to-collision)
- **Upstream video cues** (predicted future dynamics)

关键设计: action generator **不需要等 depth/video 生成完才能跑**。因为 LLM 已经把这些 cues "压缩" 进 action embeddings 里。所以推理时可以:
- **Planning-only mode**: 只跑 LLM + action generator (低 latency)
- **Imagination-enabled mode**: 同时跑 depth/video + action
- **Full generation mode**: 用于 offline simulation & data synthesis

这就是 paper 反复强调的 "modularity and controllable latency"。

---

## 7. Training Objective — Joint Multi-Task Loss

$$
\mathcal{L} = \lambda_d \mathcal{L}_d + \lambda_v \mathcal{L}_v + \lambda_a \mathcal{L}_a
$$

变量:
- $\mathcal{L}_d$: depth prediction loss (flow matching velocity regression)
- $\mathcal{L}_v$: video prediction loss (flow matching on latents)
- $\mathcal{L}_a$: trajectory prediction loss (flow matching on (x,y,cosθ,sinθ))
- $\lambda_d = 0.1$: depth loss weight (因为 depth dimension 小, 防止 over-weighting)
- $\lambda_v = \lambda_a = 1.0$: video 和 action 等权

**Single-stage joint training**: 不需要 pretrain-then-finetune 两阶段。100k steps, batch size 32, 8× NVIDIA H20 GPUs, AdamW (lr=1e-5)。

为什么 $\lambda_d = 0.1$? Depth map 的 MSE 在数值上会比 video latent 的 FM loss 大 (depth 是 pixel-space, video 是 latent-space 经 VAE 压缩)。降权是 balance gradient magnitude, 防止 depth loss dominate。

AdamW 参考: https://arxiv.org/abs/1711.05101

---

## 8. 实验结果深度解读

### Table 1 — Navsim v1 (PDMS)

| Method | Category | NC↑ | DAC↑ | TTC↑ | C↑ | EP↑ | PDMS↑ |
|---|---|---|---|---|---|---|---|
| Human | - | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 |
| TransFuser | Vision E2E | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| UniAD | Vision E2E | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| DiffusionDrive | Vision E2E | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| AutoVLA | VLA | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| DriveVLA-W0 | VLA | 98.7 | 96.2 | 95.5 | 100 | 82.2 | 88.4 |
| WoTE | World-Model | 98.5 | 96.8 | 94.4 | 99.9 | 81.9 | 88.3 |
| PWM | World-Model | 98.6 | 95.9 | 95.4 | 100 | 81.8 | 88.1 |
| Epona | World-Model | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| **DriveDreamer-Policy** | World-Model | 98.4 | **97.1** | 95.1 | 100 | **83.5** | **89.2** |

观察:
- DAC (Drivable Area Compliance) 97.1 是 SOTA, 说明 depth scaffold 帮助 vehicle 严格保持在 drivable area 内
- EP (Ego Progress) 83.5 是 SOTA, 说明 model 不会因为太保守而停滞
- 比 AutoVLA (89.1) 高 0.1 PDMS — 在 world-model 类别里 +1.1 over PWM (88.1)
- 相比 VLA 类 SOTA (AutoVLA), 优势在于 DAC (97.1 vs 95.6) 和 EP (83.5 vs 81.9) — 即 "更敢走但更守规矩"

Navsim benchmark: https://arxiv.org/abs/2406.15349
DiffusionDrive: https://arxiv.org/abs/2411.15139
AutoVLA: https://arxiv.org/abs/2506.13757
DriveVLA-W0: https://arxiv.org/abs/2510.12796
PWM: https://arxiv.org/abs/2510.19654

### Table 2 — Navsim v2 (EPDMS)

DriveDreamer-Policy 在 v2 上达到 EPDMS 88.7, 比 DriveVLA-W0 (86.1) 高 **+2.6**。v2 比 v1 多了:
- DDC (Directional Drivable Compliance)
- TLC (Traffic Light Compliance)
- LK (Lane Keeping)
- HC (Human Comfort)

观察: DriveDreamer-Policy 在 LK (97.6) 和 DDC (99.5) 上表现强, 表明 depth + video 的 joint world modeling 帮助 lane-level geometry reasoning。EPDMS 比 PDMS 更全面, v2 上的 +2.6 比 v1 上的 +0.1 大很多, 暗示 v2 的 sub-metrics 更能体现 world model 的价值。

Navsim v2 (Pseudo-simulation): https://arxiv.org/abs/2506.04218

### Table 3 — World Generation

**Video (a)**:
| Method | LPIPS↓ | PSNR↑ | FVD↓ |
|---|---|---|---|
| PWM | 0.23 | 21.57 | 85.95 |
| DriveDreamer-Policy | **0.20** | 21.05 | **53.59** |

FVD 降低 32.36 (相对 PWM)。LPIPS 更低 (perceptual similarity 更高)。PSNR 略低 (因为 PSNR 偏好模糊, generative model 会保留更多 high-frequency uncertainty)。

FVD 参考: https://arxiv.org/abs/1812.01717
LPIPS 参考: https://arxiv.org/abs/1801.03954

**Depth (b)**:
| Method | AbsRel↓ | δ1↑ | δ2↑ | δ3↑ |
|---|---|---|---|---|
| PPD (zero-shot) | 18.5 | 80.4 | 94.0 | 97.2 |
| PPD (fine-tuned) | 9.3 | 91.4 | 98.3 | 99.5 |
| DriveDreamer-Policy | **8.1** | **92.8** | **98.6** | 99.5 |

DriveDreamer-Policy 比 PPD fine-tuned 还低 1.2 AbsRel — 这归功于 LLM 的 cross-attention conditioning。$\delta_1, \delta_2$ 都是最高的 ($\delta_k$ = percentage of pixels where $\max(d_{pred}/d_{gt}, d_{gt}/d_{pred}) < 1.25^k$)。

---

## 9. Ablation Studies 深度分析

### Table 4 — World Learning 对 Planning 的贡献

| Strategy | Depth | Video | PDMS↑ |
|---|---|---|---|
| Without World Learning | ✗ | ✗ | 88.0 |
| + Depth | ✓ | ✗ | 88.5 (+0.5) |
| + Video | ✗ | ✓ | 88.9 (+0.9) |
| + Both | ✓ | ✓ | 89.2 (+1.2) |

直觉: depth 单独贡献 +0.5, video 单独贡献 +0.9, 合起来 +1.2。**不是简单相加**, 说明 depth 和 video 提供的信号有部分重叠 (都是 "future world" 的不同视角), 但各自又有 unique 的信息 (depth → geometry, video → dynamics & appearance)。Joint training 让 LLM 学到的 world embedding 更 generalizable。

### Table 5 — Depth Learning 对 Video Generation 的贡献

| Strategy | LPIPS↓ | PSNR↑ | FVD↓ |
|---|---|---|---|
| Without Depth | 0.22 | 19.89 | 65.82 |
| With Depth | **0.20** | **21.05** | **53.59** |

FVD 从 65.82 降到 53.59 (-12.23, -18.6% relative)。这验证了 paper 的核心 hypothesis: **depth 是 video generation 的 3D scaffold**, 通过 causal attention mask, video queries "看到" depth queries 的 hidden state, 自然获得 occlusion boundary 和 free-space 几何先验。

### Table 6 — Number of Queries 的影响

| Depth | Video | Action | PDMS↑ | FVD↓ | AbsRel↓ |
|---|---|---|---|---|---|
| 32 | 32 | 48 | 88.9 | 57.97 | 9.7 |
| 64 | 64 | 8 | **89.2** | **53.59** | **8.1** |

更多 query tokens = 更高容量 slot 存 context。注意 action query 数量从 48 降到 8 反而更好, 暗示 action signal 本身是 low-dimensional (一条 trajectory 就几个 waypoints), 太多 action query 会分散注意力并引入噪声。这与 DETR 中 "object query 数量需要匹配 object 数量" 的发现一脉相承。

---

## 10. 直觉总结 — 为什么这个设计 work?

我会把它总结成 5 个 "design choices that compound":

### (1) LLM 作为 perception,generators 作为 imagination

LLM 擅长 semantic reasoning + contextual aggregation, generators 擅长 multimodal sampling + uncertainty modeling。把它们用 query bottleneck 解耦, 既享受 LLM 的 stable semantics, 又享受 diffusion 的 multi-modal imagination。

### (2) Depth 作为 "几何骨架"

Depth 的信息密度恰到好处:
- 比 RGB video 低维 (单通道, 单图)
- 比 BEV occupancy 高保真 (preserves perspective)
- 直接关联 occlusion, free-space, distance-to-collision
- 可由 foundation model pseudo-label, 免训练 supervision

把 depth 显式插入 information flow, 让 video 和 action 都 "站在 3D 几何的肩膀上"。

### (3) Causal cross-group mask 实现 single-pass 3D→2D→1D 流

这种 monotonically decreasing abstraction hierarchy 让 information flow 自然, 不需要 iterative refinement (像 EM 那种)。**Single-pass = 训练简单 + 推理低延迟**。

### (4) Fixed-size query bottleneck = modularity + controllable latency

任何 generator 都可以独立 on/off。Planning-only 跑 LLM + action gen 就行, 想要 imagination 就开 depth/video gen, 想做 simulation 就全开。这是工程友好性。

### (5) Flow matching 而非 DDPM

Linear path + constant velocity target + ODE integration → 训练 stable, 推理 step 少。所有 3 个 generator 共享同一个训练 principle, 代码可以高度复用。

---

## 11. 与相关 lineage 的对比

| Method | Venue | World 实现 | 与本作区别 |
|---|---|---|---|
| **Epona** | ICCV'25 | Autoregressive diffusion, decoupled causal latents | 没有显式 depth |
| **ReSim** | 2025 | Diffusion transformer world sim + Video2Reward | 没有显式 depth |
| **DriveVLA-W0** | ICLR'26 | Future-image world modeling + MoE action expert | Image token 而非 latent video |
| **PWM** | NeurIPS'25 | Unified autoregressive transformer | Action-free forecasting |
| **DriveLaW** | 2025 | Video-gen latents → diffusion planner | 模块分离, 没有 depth |
| **OmniNWM** | 2025 | Panoramic RGB+semantics+depth+occupancy, Plücker | 多模态但有冗余, 没有 unified query interface |
| **UniPGP** | 2025 | Pretrained VLM + video gen via hybrid experts | 与 DriveDreamer-Policy 最像, 但缺 depth |
| **DriveDreamer-Policy** | - | **Depth + Latent Video + Action via fixed query** | **首次三模态 unified** |

Epona: https://arxiv.org/abs/2601.05083 (推测)
ReSim: https://arxiv.org/abs/2506.09981
DriveVLA-W0: https://arxiv.org/abs/2510.12796
PWM: https://arxiv.org/abs/2510.19654
DriveLaW: https://arxiv.org/abs/2512.23421
UniPGP: https://arxiv.org/abs/2512.09864
OmniNWM: https://arxiv.org/abs/2510.18313

---

## 12. 我 (作为读者) 的几个观察与潜在质疑

### 观察 1: Depth supervision 的 "cheapness" 是关键

paper 用 Depth Anything 3 做 pseudo-label, 这把 depth supervision 从需要 LiDAR/标注降到了 zero-cost。这一点是 paper 能跑通的 infrastructure 撑腰。如果 depth foundation model 退化 (domain shift、metric 不准), 整个 pipeline 受影响。

### 观察 2: Causal mask 没有 reverse attention — 是否 lossy?

depth queries 只看 input tokens, 看不到 video/action queries 的 hidden state。这意味着 depth 生成时无法 "知道" 未来将发生什么动作 (它依赖 LLM 输入的 current action context)。这是一个设计选择: **depth 是 current scene 的几何, 不应该 condition on future**, 让 future prediction 反过来 condition on depth 是更自然的因果方向。

### 观察 3: 9 帧 video @ 2Hz = 4.5s horizon

action generator 的 trajectory horizon 没明说, 但很可能也是 4 秒级别。这种 alignment 让 "imagination" 和 "planning" 在时间尺度上 comparable。如果 video horizon 是 10 秒, action 是 2 秒, 那 video 的长尾未来对 action 没有直接帮助。

### 观察 4: Single-stage training 的风险

joint loss $\mathcal{L} = 0.1 \mathcal{L}_d + \mathcal{L}_v + \mathcal{L}_a$ 在 single-stage 下训 100k steps。三个 generator 的 capacity (PPD depth, Wan-1.3B video, 单独 action DiT) 不平衡。LLM backbone (Qwen3-VL-2B) 同时是 perception 和 condition provider。是否所有 component 都收敛到 optimal, 还是某个 generator underutilized? Paper 没给 per-component convergence curve。这是一个潜在的 follow-up 方向。

### 观察 5: 与 DiffusionDrive (CVPR'25) 的对比

DiffusionDrive PDMS 88.1, 是 vision-based E2E 的 SOTA, 不带 world model。DriveDreamer-Policy 89.2 只高 1.1。这暗示 world model 在 Navsim v1 上的 marginal benefit 不大。但 v2 上 DriveDreamer-Policy (88.7) 比 DiffusionDrive (84.5) 高 **4.2** — v2 的更复杂 sub-metrics (lane keeping, traffic light, direction compliance) 真正受益于 world modeling。这是 paper 一个被低估的论点。

### 观察 6: Action query 只用 8 个 tokens

trajectory 一般 4-8 waypoints 就够, 8 个 action query 与之对应。这意味着 LLM 把整个 trajectory 压成 8 个 latent vectors, 给 action DiT 做 cross-attention key。这种 "extreme compression + diffusion decoder" 模式让我联想到 VQ-VAE 的 discrete prior + decoder, 只是这里是 continuous latent + diffusion decoder。

### 观察 7: 与 Cosmos / GAIA / VISTA 等 large-scale driving world model 的关系

NVIDIA Cosmos (https://arxiv.org/abs/2501.03575)、VISTA (NeurIPS 2024)、GAIA-2 (https://arxiv.org/abs/2503.20523) 是 large-scale pure world model, 没有动作 output。DriveDreamer-Policy 在他们之上加了 **policy head**, 形成了 WAM。这种 "world model + policy head" 是当前 driving WAM 的主流路线 (与 robotics 中 π0, RDT 类似)。

---

## 13. 一些可能的延伸方向

1. **Metric depth**: 现在 depth 是 log-normalized relative depth (来自 DA3), 如果改成 metric depth (UniDepthV2, https://arxiv.org/abs/2403.18913), 可以直接计算 distance-to-collision, 给 action generator 更强的物理约束。

2. **Multi-view video**: 现在 video 是 single-view (front), 如果扩展到 6-view 同步生成 (像 DriveDreamer / OmniNWM 那样), 可以做 BEV-level consistency。

3. **Longer horizon**: 9 帧 @ 2Hz = 4.5s, 对 highway merging 不够。如果用 autoregressive rollout (像 Epona), 可以到 10s+。

4. **Reinforcement fine-tuning**: 现在 action generator 是纯 imitation learning (flow matching)。可以像 ReCogDrive (https://arxiv.org/abs/2506.08052) 那样加 RL fine-tuning, 用 world model 做 roll-out reward。

5. **3D occupancy head**: 在 depth + video 之外加 3D occupancy head, 让几何 scaffold 更显式。OmniNWM 已经做了这个, 但没用 unified query interface。

6. **Cross-attention vs full self-attention across generators**: 现在 depth/video/action 之间是 LLM 内部的 causal mask, generators 之间没有 cross-attention。如果让 video generator 在每个 denoising step 都 attend 到 depth generator 的输出 (iterative refinement), 会不会更好? 但这会破坏 single-pass 优势。

7. **Action generator 是否真的需要 video context**: ablation Table 4 显示 +Video alone (+0.9) > +Depth alone (+0.5), 但 +Both (+1.2) < +0.5 + 0.9 = +1.4, 说明 video 和 depth 有 overlap。是否 action 只看 depth 就够? 这需要更细的 ablation (e.g., block action queries 看 video queries 的 attention)。

---

## 14. 与你 (Karpathy) 平时关心的 topic 的连接

你一直在推 **autoregressive world model + action** 的范式, 这篇 paper 是这条路线的一个 instance, 但用了 **query-based interface** 而非纯 autoregressive token stream。两者本质区别:

- **Autoregressive token paradigm** (像 DriveVLA-W0, AutoVLA, Cosmos): 把 depth, video, action 都 tokenize, 用 next-token prediction 训练。优点是 unified, 缺点是 modality 之间 resolution mismatch (depth 是 dense 2D, action 是 sparse 1D)。
- **Query-based modular paradigm** (DriveDreamer-Policy): LLM 输出 low-dimensional query embeddings, 下游 experts 用 diffusion 在各自 native space 生成。优点是 each modality 在自己最优的 representation space 工作, 缺点是 query bottleneck 可能丢失细节。

另一个连接点: **System 1 / System 2 thinking**。LLM 在这里是 System 2 (slow reasoning, contextual aggregation), diffusion generators 是 System 1 (fast pattern sampling, multi-modal imagination)。这与你之前在 "Software 2.0" / "Software 3.0" 上的 framing 很契合。

最后一个连接: 这篇 paper 的 "modularity + controllable latency" 是一个非常 production-friendly 的设计。Planning 在车里必须 < 100ms, imagination 可以异步跑。这种 "policy 必须实时, world model 可以慢" 的 split 在 robotics 里也通用。

---

希望这个 build up 给你一个完整的 mental model。这篇 paper 的核心 contribution 在我看来是把 **depth 这个 modality** 重新放回 WAM 的中心位置, 用一个非常 simple 的 causal cross-group attention 实现了 3D→2D→1D 的自然信息流。剩下的工程 (flow matching, query bottleneck, foundation model initialization) 都是成熟的 building blocks, 但组合起来的 ablation 验证了 depth 的 unique 价值, 这一点很 solid。

如果让我赌未来 12 个月的方向: **metric depth + multi-view video + 3D occupancy head + RL fine-tuning via world model rollout** 会是 DriveDreamer 系列的下一站。你呢?
