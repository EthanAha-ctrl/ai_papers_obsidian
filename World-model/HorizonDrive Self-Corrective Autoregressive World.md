---
source_pdf: HorizonDrive Self-Corrective Autoregressive World.pdf
paper_sha256: 01988fe0b0da9d5bfc6e783bdc5702fa728ea874f3d730f118ab6927998a2b23
processed_at: '2026-08-19T11:28:35-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 HorizonDrive

## 一句话说清楚

想象你在写一本很长的小说。你每写一页，就基于前面写过的内容继续往下写。问题是：你是人，你会犯错——某页写歪了，后面所有页都会被带偏。

**这篇论文做的事**：训练一个"老师"模型，让它专门学会"从自己写歪的内容里把故事拉回正轨"，然后把这个纠错能力教给一个"学生"模型，学生就能一直写下去不崩。

---

## 问题到底有多难

先理解为什么 driving video generation 这个任务特别恶心。

Driving world model 要做的事情：给它一段历史的行车视频帧，加上"接下来要怎么开"的控制信号（方向盘转多少、地图长什么样、周围有哪些车），它要生成接下来的画面。

$$\hat{\mathbf{z}}_{T+1:T+K} \sim p_\theta(\mathbf{z}_{T+1:T+K} \mid \mathbf{z}_{1:T}, \mathbf{c}_{T+1:T+K})$$

这里：
- $\mathbf{z}_{1:T}$：前面 $T$ 帧的 latent representation（历史画面）
- $\mathbf{c}_{T+1:T+K}$：接下来 $K$ 帧的控制信号
- $\hat{\mathbf{z}}_{T+1:T+K}$：生成的 $K$ 帧

**核心矛盾**：训练的时候，$\mathbf{z}_{1:T}$ 是 ground truth 干净数据。但真正用的时候，$\mathbf{z}_{1:T}$ 是模型自己上一轮生成的 $\hat{\mathbf{z}}$——有误差的。

这就好比：你练钢琴时永远弹的是 perfect tempo 的曲谱，但上台表演时前面弹错了一个音，后面就全乱了。这就叫 **exposure bias**。

在 driving 里这个问题尤其严重，因为：
1. Ego-motion 快——每帧车都在往前走，画面变化大
2. Scene 变化快——路边物体、其他车、交通标志都在动
3. Geometric consistency 要求高——车道线歪了、路面弯曲了、车身变形了，人眼一眼就看出来

所以 SOTA driving world models（Vista、MagicDrive-V2 这些）rollout 几十秒就崩了，远远不到 closed-loop simulation 需要的分钟级。

---

## 之前别人怎么解决，为什么不够好

### 方案 A：Frame sinks（锚帧）

像 StoryDiffusion、In-context LoRA 这些方法，在长序列里插入一些 "anchor frames" 当锚点，防止漂移。

**问题**：driving 场景下 ego-motion 太快了，anchor frame 很快就不 relevant 了。你在十字路口插一个锚帧，车开出去 10 秒后那个锚帧毫无意义。

### 方案 B：Self-Forcing（在 student 端解决）

Self-Forcing（Huang et al., 2025, https://arxiv.org/abs/2506.08009）的思路：训练 student 时就让 student 基于自己的 outputs 做 conditioning，提前暴露给 rollout error。

**问题**：student 的 supervision 来自 teacher，但 teacher 的 single-pass generation window 是固定的（比如 40 帧）。DiT 的 attention cost 是 $O(L^2)$，想扩大 window 内存就爆了。

所以 supervision horizon 被卡在 teacher 一次性能生成的长度。student 学得再好，也学不到超过这个 horizon 的 long-horizon behavior。

### 方案 C：Student-side degradation training

类似 Self-Forcing 的家族，包括 Causal-Forcing、LongLive 这些。都是想办法在 student 端注入 robustness。

**同样的问题**：supervision horizon bounded by teacher single-pass capacity。

---

## HorizonDrive 的核心 insight

**Key question**: 能不能让 teacher 本身通过 AR rollout 扩展到任意长度，在 bounded memory 下提供 unbounded supervision？

**Key difficulty**: standard teacher 在自己 predictions 下会 drift，supervision 会被污染。

**Key insight**: 先让 teacher 变得 rollout-capable（在自己 AR predictions 下保持稳定），就能突破 single-pass horizon barrier。

这个 insight 的精妙之处在于 **问题重构**。传统思路把 AR drift 视为 deployment-time problem——部署时模型会 drift，要想办法在部署时 fix。HorizonDrive 把它转化为 training-time problem——在训练时就让模型学会 handle drift，把 drift handling 内化到 teacher 的 weights 里。

用类比说：不是给司机一个 GPS 防止迷路，而是让司机本身变成一个永远不会迷路的人。

---

## 三个阶段，一步步看

### Stage 1: 训练一个基础 driving world model ($G_0$)

用 Wan 2.1 1.3B（https://arxiv.org/abs/2503.20314）作为 backbone，full bidirectional attention DiT。

**怎么变成 video continuation model**：把每个 training clip 分成两部分：
- **Condition window**（前 $T=11$ 帧）：保持 clean，noise level $t=0$
- **Generation chunk**（后 $K$ 帧）：加噪，用 flow-matching loss 监督

Flow-matching 的 loss：

$$\mathcal{L}_{CFM} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} \| v_\Theta(z_{(t)}, t) - (z_{(0)} - \epsilon) \|_2^2 \tag{3}$$

变量解释：
- $\epsilon$：从标准正态分布采样的 noise
- $z_{(t)}$：中间 latent state，由 $z_{(t)} = \sigma_t z_{(0)} + (1-\sigma_t)\epsilon$ 构造
- $\sigma_t$：预定义的 noise schedule
- $v_\Theta$：模型预测的 vector field（flow matching 的 velocity）
- $z_{(0)}$：clean data
- 目标是让 $v_\Theta$ 学会从 noise $\epsilon$ 流向 data $z_{(0)}$ 的 velocity

**Control 怎么注入**（disentangled control，参考 https://arxiv.org/abs/2603.12864）：

1. **Spatial structure**（HD map + 3D bounding boxes）：
   - 渲染成 layout tokens $h_{b_f} \in \mathbb{R}^{f \times s \times d}$
   - 通过 zero-initialized projector $f_{\text{zero}}$ 加到 DiT features：
     $$h_{(t)} \leftarrow h_{(t)} + f_{\text{zero}}(h_{b_f})$$
   - $f$=frames, $s$=spatial tokens, $d$=feature dim
   - Zero-init 意味着训练初期这部分贡献为 0，逐渐学到怎么用 spatial layout

2. **Ego action** $\mathbf{a} = (\Delta x, \Delta y, \Delta\text{yaw}) \in \mathbb{R}^{F \times 3}$：
   - $F$=frames, $(\Delta x, \Delta y, \Delta\text{yaw})$=相对平移和偏航角变化
   - Sinusoidal embedding $\phi(\mathbf{a})$
   - 通过 AdaLN-style gating 注入：$f_{\text{zero}}(\phi(\mathbf{a})) \in \mathbb{R}^{f \times 6 \times d}$
   - 6 个 channels 分成两组：pre-norm shift/scale + post-layer residual gate，分别对应 self-attention 和 feed-forward sublayers

**为什么这样 disentangle？** Spatial layout 决定 scene composition（什么东西在哪），应该 additive 到 features 里。Ego action 决定 temporal dynamics（画面怎么动），应该通过 modulation 影响整个 layer 的行为。两种 control 机制完全不同，分开注入避免互相干扰。

这个 stage 产出 $G_0$，但 $G_0$ 只在 clean GT history 下训练，直接 AR rollout 会崩。

---

### Stage 2: Scheduled Rollout Recovery (SRR) —— 最核心的创新

目标：把 $G_0$ 变成 rollout-capable teacher $G_{\text{roll}}$。

#### Step 1: 用 $G_0$ 做 AR rollout 生成 "corrupted trajectory"

从 GT history $\mathbf{z}_{1:T}$ 出发，用 $G_0$ 做 $N$ 步 AR rollout：

$$\hat{\mathbf{z}}_{s_n+1:s_n+K} = G_0(\hat{\mathbf{z}}_{s_n-T+1:s_n}, \mathbf{c}_{s_n+1:s_n+K}), \quad s_n = T + (n-1)K \tag{6}$$

变量：
- $s_n$：第 $n$ 步 rollout 的起始 index，$s_n = T + (n-1)K$
- $N$：rollout 步数
- $\hat{\mathbf{z}}_{s_n-T+1:s_n}$：第 $n$ 步的 context，来自之前步骤的 predictions
- 产出 trajectory $\hat{\mathbf{z}}_{T+1:T+NK}$，prediction errors 跨 AR steps 累积

**这就是模型自己造成的 "垃圾"**：一个有累积误差的 trajectory。

#### Step 2: 用 corrupted trajectory 作为 training condition

Sample 一个 generation boundary $s$，把 condition history 换成 rollout predictions，但 supervision target 保持 GT：

$$\tilde{\mathbf{z}}_{s-T+1:s} = \hat{\mathbf{z}}_{s-T+1:s}, \quad \mathbf{z}^{\star}_{s+1:s+K} = \mathbf{z}_{s+1:s+K} \tag{7}$$

**直觉**：给模型看一个"已经被自己搞砸了的历史"，让它学会"从这个烂摊子里把未来恢复成 GT"。模型训练时就经历 deployment-time 会遇到的 distribution shift，获得 recovery capability。

#### Step 3: Local pred-to-GT transition schedule

问题：直接从 prediction 跳到 GT 会有 temporal discontinuity。模型会困惑——是要延续 prediction 的 state，还是要"纠正"到 GT？

Solution：在 boundary $s$ 两侧各 $w$ 帧做 blending：

$$\bar{\mathbf{z}}_i = \begin{cases} 
\tilde{\mathbf{z}}_i, & s-T+1 \leq i \leq s-w \\
\alpha_i \tilde{\mathbf{z}}_i + (1-\alpha_i) \mathbf{z}_i^{\star}, & s-w+1 \leq i \leq s+w \\
\mathbf{z}_i^{\star}, & s+w+1 \leq i \leq s+K 
\end{cases} \tag{8}$$

变量：
- $w$：blending window radius
- $\alpha_i$：从 1 线性衰减到 0 的 mixing coefficient
- 左边 $\tilde{\mathbf{z}}_i$ 是 rollout prediction
- 右边 $\mathbf{z}_i^{\star}$ 是 GT

**直觉**：在 boundary 附近构造一个 continuous temporal bridge，让模型平滑过渡，而不是 hard jump。

**$w$ 的 curriculum**：
- 训练初期 $w=0$：sharp boundary，强迫模型直接面对大 deviation（hard task）
- 训练后期 $w \to 8$：扩大 transition region，task 变成 fine-grained correction（easy task）

**为什么先 hard 后 easy？** 这是反直觉的。传统 curriculum learning 是先易后难。这里反过来——先学 hard recovery（建立大幅纠错能力），再 refine 到 small correction（精细调整）。

类比：先学摔跤后怎么爬起来（rough recovery），再学怎么走直线不摔跤（fine stability）。前者更基础，后者建立在前者之上。

#### Step 4: Global boundary-decay sampling schedule

这是另一个关键 schedule，控制 $s$ 在 rollout trajectory 上 sample 的位置。

**Empirical observation**（论文 Figure 3）：

论文做了 error heatmap 分析，把 rollout trajectory 分成几个 interval（10-30, 30-50, 50-70, 70-90），发现：

1. **Late rollout position**（大 $s$，比如 70-90）：error heatmap 更强，semantic drift 更严重
2. **Early rollout position**（小 $s$，比如 10-30）：cross-case cosine similarity 更高，errors 更 generic

这说明什么？早期 rollout 的 error 是"通用"的（所有 case 都会遇到类似的），后期 rollout 的 error 是"特定"的（每个 case drift 的方式都不一样）。

**Curriculum 设计**：训练从 large $s$ 开始，逐渐 decay 到 small $s$。

- 先学 severe case-specific semantic drift（建立 robustness ceiling）
- 再学 mild generic degradation（refine 到通用 case）

Schedule（Table 4）：
- $N(k)$: 10 → 4 (steps 0-8000)，rollout depth 逐渐减小，意味着 sample 的 $s$ range 从远端向近端收缩
- $w$: 0 → 8 (steps 0-8000)，blending radius 逐渐增大

**两个 schedule 的协同**：
- Global schedule 决定 sample 哪个 rollout position（决定 error severity 和 specificity）
- Local schedule 决定在该 position 如何 transition from prediction to GT（决定 recovery granularity）

训练初期：global large $s$ + local $w=0$ → 处理 severe late-rollout drift with sharp boundary（hardest task）
训练后期：global small $s$ + local $w=8$ → 处理 mild early-rollout drift with smooth transition（easiest task）

#### SRR 与传统 anti-drifting 的本质区别

| 维度 | 传统方法 | SRR |
|------|---------|-----|
| 干预对象 | Deployed generator | Supervisor (teacher) |
| 干预时机 | Deployment-time | Training-time |
| 干预方式 | Regularize generator output | Stabilize supervisor's rollout |
| Supervision horizon | Bounded by teacher single-pass | Unbounded (teacher 自己 rollout) |

**这个 inversion 是论文最 deep 的贡献。** 传统思路是"让 deployed model 更 robust"，HorizonDrive 是"让 supervisor 更 reliable"。前者治标，后者治本——因为 distillation 的本质是 supervision transfer，supervisor 不可靠，student 再怎么训练也是 garbage in garbage out。

---

### Stage 3: Teacher Rollout DMD (TRD) —— 蒸馏到 real-time student

SRR 产出的 $G_{\text{roll}}$ 虽然稳定，但 multi-step diffusion sampling 慢得要死，做不了 real-time interactive simulation。需要 distill 成 faster student。

#### Asymmetric Teacher-Student 设计

| | Teacher $G_{\text{roll}}^{\mathcal{T}}$ | Student $G_{\text{roll}}^{S}$ |
|---|---|---|
| Init | SRR weights | SRR weights |
| Chunk size | $K^{\mathcal{T}} = 40$ | $K^{S} = 10$ |
| Denoising steps | multi-step | 4 |
| Trainable | ❌ frozen | ✅ params $\phi$ |
| Context $T$ | 11 | 11 |

**为什么 asymmetric？**

如果对称设计：
- 都用长 chunk → student inference 慢
- 都用短 chunk → teacher supervision horizon 受限

Asymmetric 打破 trade-off：
- Teacher 长 chunk + multi-step → capture longer temporal dynamics，提供 high quality target
- Student 短 chunk + few-step → 满足 real-time

**Memory 分析**：

Single-pass $N \times 40$ frames generation：attention cost $O((N \cdot 40)^2)$，N 大了直接爆炸。

AR rollout with fixed $(T, K^{\mathcal{T}})$：每步 $O((T + K^{\mathcal{T}})^2) = O(51^2)$，bounded，与 $N$ 无关。

Teacher frozen 不需要 gradient through，省 memory。Student 需要 gradient through 4 denoising steps，但因为 chunk 短，memory 也可控。

#### Student 也做 AR rollout during training

$$\hat{\mathbf{z}}^S_{s_n+1:s_n+K} = G_{\text{roll},\phi}^{S}(\hat{\mathbf{z}}^S_{s_n-T+1:s_n}, \mathbf{c}_{s_n+1:s_n+K^S}), \quad n=1,\ldots,N \tag{9}$$

Student 自己也做 $N=20$ 步 AR rollout。每当累积 student rollout 覆盖 teacher chunk length $K^{\mathcal{T}}=40$ frames（即每 4 个 student chunks of 10 frames），就 apply DMD gradient 并 backprop。

**关键设计**：每 $D=5$ 个 student chunks update 一次 DMD gradient（Table 5），避免每个 chunk 都做 expensive distribution matching。

#### TRD Gradient 公式

$$\nabla_\phi \mathcal{L}_{\text{TRD}} = \mathbb{E}_\tau \left[ -\left( \underbrace{s_{\text{cond}}^{\text{real}}(z_{(\tau)}) - s_{\text{cond}}^{\text{fake}}(z_{(\tau)})}_{\text{Distribution Matching}} + \underbrace{\mathbf{1}_{\{\tau \leq \tau_{\text{th}}\}}(\alpha-1)\left(s_{\text{cond}}^{\text{real}}(z_{(\tau)}) - s_{\text{uncnd}}^{\text{real}}(z_{(\tau)})\right)}_{\text{Noise-truncated CFG}} \right) \frac{\partial G_{\text{roll},\phi}^S}{\partial \phi} \right] \tag{10}$$

变量逐个解释：
- $\phi$：student parameters
- $\tau$：renoised noise level（从 generated sample 加噪到 level $\tau$）
- $s_{\text{cond}}^{\text{real}}$：real-data conditional discriminator score（衡量真实数据分布的 score）
- $s_{\text{cond}}^{\text{fake}}$：fake-data conditional discriminator score（衡量生成数据分布的 score）
- $s_{\text{uncnd}}^{\text{real}}$：real-data unconditional score
- $\alpha$：CFG scale，default 6
- $\tau_{\text{th}}$：noise threshold for CFG application
- $\mathbf{1}_{\{\tau \leq \tau_{\text{th}}\}}$：indicator function，$\tau \leq \tau_{\text{th}}$ 时为 1，否则为 0
- $\partial G_{\text{roll},\phi}^S / \partial \phi$：generator Jacobian（student 对参数的梯度）

**第一部分 Distribution Matching**（参考 DMD, Yin et al., 2024b, https://arxiv.org/abs/2312.02639）：

$s_{\text{cond}}^{\text{real}} - s_{\text{cond}}^{\text{fake}}$ 衡量 real data distribution 和 fake (student generated) data distribution 的 score difference。Gradient 方向是让 student output distribution 匹配 real data distribution。

在 DMD 里，"real" 其实是 teacher 的 output（teacher 作为 reference distribution），"fake" 是 student 的 output。Student 通过 matching teacher distribution 学到 teacher 的 generation behavior。

**第二部分 Noise-Truncated CFG**（参考 Decoupled DMD, Liu et al., 2025a, https://arxiv.org/abs/2511.22677）：

标准 DMD 用 CFG 强化 teacher gradient guidance。CFG 本质是 condition 和 unconditional score 的 difference，放大 condition 信号。

但论文发现 full CFG 在 video rollout 时会 oversaturation——每个 chunk 都受 CFG 影响，cumulative over AR steps 放大 color/contrast saturation。

**Noise-truncated 的 intuition**：
- High noise levels（$\tau$ 大）：latents 接近 noise，CFG 影响整体 structure
- Low noise levels（$\tau$ 小）：latents 接近 data，CFG 影响 fine details

在 high noise 用 CFG 会 propagate 到后续 chunks，导致 oversaturation 累积。Restricted 到 low noise，CFG 只影响当前 chunk 的 fine details，不会 propagate。

**$\tau_{\text{th}}$ 的 schedule**（ablation Table 3 验证）：

| Schedule | $\tau_{\text{th}}$ 设置 | FVD↓ | 说明 |
|----------|----------------------|------|------|
| None | 0 | 110.81 | 不用 CFG |
| Full | 1000 throughout | 184.06 | 全程 CFG，严重 oversaturation |
| Early | decays 1000→0 from step 0 | 111.99 | Early decay 没效果 |
| **Delayed** | 1000 for 100 steps warmup, then decay to 0 by step 400 | **92.99** | 最佳 |

**Delayed schedule 的 intuition**：warmup 阶段用 full-range CFG 建立 strong conditional controllability（让模型学会响应 condition），然后 progressively restrict CFG 到 low noise levels，maintain visual quality。

类比：先让模型在所有 noise level 都学会"听话"（响应 condition），然后只在"细节修正"阶段保留 CFG 的影响，避免 CFG 在"结构构建"阶段过度放大导致 oversaturation。

---

## 实验结果有多猛

### 对比 long-horizon streaming methods（Table 1 bottom）

所有 baseline 都在同一 backbone（Wan 2.1 1.3B）+ 同一 driving control modules + 同一数据上 re-train，唯一变量是 long-horizon training framework。

| Method | FID↓ | FVD↓ | ARE↓ | DTW↓ |
|--------|------|------|------|------|
| Self-Forcing | 41.53 | 161.00 | 3.47 | 6.22 |
| Self-Forcing++ | 28.84 | 147.57 | 3.78 | 3.61 |
| LongLive | 29.05 | 161.41 | 3.28 | 3.65 |
| **HorizonDrive** | **13.82** | **92.99** | **2.60** | **3.27** |

- FID 降低 52%（28.84 → 13.82）
- FVD 降低 37%（147.57 → 92.99）
- ARE 降低 21%（3.28 → 2.60）
- DTW 降低 9%（3.61 → 3.27）

**Baseline 为什么差？** 它们只靠 long-horizon fine-tuning mitigate drift。但 long tuning alone 不保证 rollout-capable teacher——teacher 不能可靠 rollout 就产生 degraded trajectories，distilling from poor supervision yields poor students。SRR 先 strengthen teacher，raise the ceiling of what student can learn。

### 对比 driving-specific methods（Table 2）

**Short-video（8-25 frames）**：

| Method | Frames | FID↓ | FVD↓ |
|--------|--------|------|------|
| Vista | 25 | 6.90 | 89.40 |
| DreamForge | 16 | 14.61 | 103.61 |
| **HorizonDrive (N=1)** | 21 | 12.54 | **84.53** |

HorizonDrive FVD 最好，FID 有竞争力，同时支持 full driving control suite（T+M+B+A）。

**Long-video（241 frames）**：

| Method | Frames | FID↓ | FVD↓ |
|--------|--------|------|------|
| MagicDrive-V2 (single-pass) | 241 | 20.91 | 94.84 |
| **HorizonDrive (N=20)** | 211 | **13.82** | **92.99** |

HorizonDrive 用 sequential rollout + few-step denoising 能 match 甚至超越 MagicDrive-V2 的 many-step single-pass generation。这说明 AR rollout with bounded memory 是 viable 的 long-video generation paradigm。

### Error Accumulation 分析（Figure 5）

论文报告了每个 cumulative chunk 的 FID。**HorizonDrive 在所有 19 chunks 上保持 stable image quality，无 error accumulation。** Self-Forcing++ 的 FID 单调退化。

这是最直观的 evidence——SRR 训练的模型真的不会在 long rollout 下崩。

### Minute-level generation（Figure 13）

在 self-collected dataset 上，10 FPS rollout，$K^S=10, T=11$，约 1 分钟连续 rollout，模型保持 coherent road geometry、lane structure、traffic-agent behavior。

Sliding-window rollout 的关键性质：每步 compute bounded by $T + K^S$，与 total horizon 无关。

### Closed-loop simulation（Figure 14）

Planner 和 world model step-by-step 交互：
1. Planner 消费 latest generated frame，输出 ego trajectory
2. Ego trajectory + HD map + bounding boxes 重新编码成 next-step action condition
3. 喂给 HorizonDrive，loop 重复
4. **No GT ego trajectory used during simulation**——所有 conditioning signal 都是 self-generated

即使 world model rollout error + planner prediction error 都 compound，HorizonDrive 仍保持 coherent scene structure 和 stable agent behavior。这对 closed-loop policy evaluation 是 practical viable 的。

### Inference efficiency

| Resolution | Per-chunk latency | Effective FPS |
|------------|-------------------|--------------|
| 256×512 | 1.8s | ~5.6 FPS |
| 384×768 | 5.8s | ~1.7 FPS |

单张 NVIDIA 5090 GPU。Bounded per-step memory and compute，可以 indefinitely rollout。

---

## Ablation 的关键发现

### Init: SRR 到底有没有用（Table 3）

| Init (Stu/Tea) | FID↓ | FVD↓ | ARE↓ | DTW↓ |
|----------------|------|------|------|------|
| Base/Base | 19.24 | 141.88 | 2.76 | 3.30 |
| SRR/Base | 20.34 | 128.77 | 3.15 | 3.39 |
| Base/SRR | 14.44 | 107.54 | 2.75 | 3.80 |
| **SRR/SRR** | **13.82** | **92.99** | **2.60** | **3.27** |

**最关键的 finding**：Base/SRR（仅 teacher 用 SRR）的 single-factor gain 最大（FID 19.24→14.44）。

这证实了 **long-horizon supervision 由 teacher 的 rollout reliability 主导**。Teacher 可靠 rollout 提供高质量 trajectory targets，即使 student 从 Base 起步也能学好。

反过来，SRR/Base（仅 student 用 SRR）几乎没用，甚至 FID 略变差。**Rollout-aware student alone 无法补偿 teacher 的 drift**——distilling from degraded trajectories 必然得到 degraded student。

这个 ablation 是对 core insight 最强的验证：问题在 supervisor 端，fix 也要在 supervisor 端。

### Rollout steps N

| N | FID↓ | FVD↓ | ARE↓ | DTW↓ |
|---|------|------|------|------|
| 1 | 21.15 | 135.35 | 3.39 | 5.42 |
| 4 | 18.19 | 139.28 | 3.03 | 3.66 |
| 20 | 13.82 | 92.99 | 2.60 | 3.27 |

N=1 时 student 只在 single-chunk generation 上监督，从未经历自己的 rollout errors。ARE 3.39° 和 DTW 5.42 最差。

N 增大后 ARE 和 DTW 单调下降。FVD 在 N 足够大时急剧下降（139.28 at N=4 → 92.99 at N=20）。

**Intuition**：student 必须在训练时就经历 multi-step rollout error accumulation，才能学会 correct 它们。这跟人类学开车一样——只练单次操作和练连续操作完全是两码事。

---

## 我的整体直觉

### 1. "治本 vs 治标"的范式转变

传统 anti-drifting 方法都是治标——让 deployed model 更 robust。Self-Forcing 让 student 训练时就暴露 rollout error，Causal-Forcing 用 causal attention，frame sinks 用 anchor frames。

HorizonDrive 是治本——fix the supervisor。因为 distillation 是 supervision transfer，supervisor 不可靠，student 再怎么训练都是 garbage in garbage out。

**这个 inversion 可能启发 future work 在其他 long-horizon generation 任务中的应用**：
- Robotics 里的 model-based RL：world model 在自己 predictions 下 rollout 会 drift，SRR 思路可以让 world model 自己 rollout-capable
- Embodied AI 里的 long-horizon planning：planner 在自己 plan 上 execution 会 accumulate error
- Video generation 里的 long-form content creation：movie-level narrative consistency

### 2. Schedule 设计的精妙

两个 schedule（local blending + global boundary decay）协同工作：

- Local schedule 处理 single-step recovery 的 granularity
- Global schedule 处理 trajectory 上的 error distribution

两者协同：global 决定 sample 哪个 rollout position，local 决定在该 position 如何 transition。

**Curriculum 设计的哲学**：先 hard 后 easy，先建立 robustness ceiling 再 refine。这跟传统 curriculum learning 相反，但在 recovery 任务里更合理——先学会从大错中爬起来，再学会精细调整。

### 3. Asymmetric design 的工程智慧

Teacher 长 chunk + multi-step，student 短 chunk + few-step。这个 asymmetric 打破了 quality vs speed 的 trade-off。

关键 insight：用 AR rollout 的 bounded memory 性质，teacher 也能在固定 memory 下无限 rollout。所以 teacher 的 supervision horizon 不受 single-pass 限制，可以无限 extend。

Student 通过 chunk-wise distribution matching 学到 teacher 的 long-horizon behavior，同时保持 fast inference。

### 4. Noise-truncated CFG 的 subtle 之处

CFG 在 image generation 里是标配，但在 video rollout 里会 oversaturation。这个现象之前没被充分重视。

HorizonDrive 的 ablation 清楚显示：full CFG 几乎让 FVD 翻倍（110.81 → 184.06）。Noise-truncated CFG 只在 low noise level 应用 CFG，避免 high-noise CFG propagate 到后续 chunks。

Delayed schedule 先用 full-range CFG warmup，建立 conditional controllability，然后 decay 到 low noise only，maintain visual quality。这个 schedule 跟 SRR 的 schedule 哲学一致——先建立 core capability，再 refine。

### 5. 对 closed-loop simulation 的 implication

Closed-loop driving simulation 需要 world model 和 planner 交互。传统方法因为 drift 无法做长时 closed-loop。HorizonDrive 在 Figure 14 展示了 closed-loop simulation——planner 消费 generated frame 输出 ego trajectory，ego trajectory 喂回 world model，所有 signal 都是 self-generated。

这对 autonomous driving 的 policy evaluation 有实际价值。参考类似思路：
- CARLA simulator：传统 rule-based simulation
- Cosmos（NVIDIA, https://arxiv.org/abs/2506.09042）：world foundation model for physical AI
- Genie（DeepMind）：interactive environment generation

HorizonDrive 把 generative world model 推到了可以实际做 closed-loop evaluation 的程度。

### 6. Limitation 和未来方向

论文承认 SRR 是 offline 的——rollout trajectory cache 每 R=2000 optimizer steps 刷新一次。这有问题：
- Training 中的 rollout 不能反映 student 最新能力
- Cache refresh 有 latency

**Future direction**: Online rollout-recovery training，world model continuously improves AR robustness from its own interaction trajectories。

这类似于 DAgger（Ross et al., 2011, https://arxiv.org/abs/1011.0686）在 imitation learning 里的思想——agent 在自己 trajectories 上收集错误并修正。Online SRR 会更符合 deployment 分布，可能比 offline SRR 更 effective。

参考类似思想的 recent work：
- Test-Time Training for video（Dalal et al., 2025, https://arxiv.org/abs/2507.27471）：deployment 时持续 adapt
- Self-resampling（Guo et al., 2025, https://arxiv.org/abs/2512.15702）：AR diffusion 的 end-to-end training

---

## 一句话总结

HorizonDrive 把 AR drift 问题从 "deployed model 的问题" 重构为 "supervisor 的问题"，通过 SRR 让 teacher 自己 rollout-capable，再通过 TRD 把 long-horizon behavior distill 到 real-time student。这个 inversion 让 bounded memory 下的 unbounded horizon supervision 成为可能，在 nuScenes 上把 FID 和 FVD 砍掉 52% 和 37%，并 demo 了 minute-scale rollout 和 closed-loop simulation。

Project page: https://zcliangyue.github.io/HorizonDrive

---

# HorizonDrive: 自校正 Autoregressive World Model 深度解析

## 1. 论文核心问题与 Motivation

这篇论文处理的是 **autonomous driving closed-loop simulation** 的根本瓶颈：如何让 driving world model 在分钟级别的 AR rollout 下保持稳定？

### 1.1 问题的本质

AR rollout 过程可以形式化为：

$$\hat{\mathbf{z}}_{T+1:T+K} \sim p_{\theta}(\mathbf{z}_{T+1:T+K} \mid \mathbf{z}_{1:T}, \mathbf{c}_{T+1:T+K}) \tag{5}$$

其中：
- $\mathbf{z}_{1:T}$: context window 中的 latent history（前 T 帧 latents）
- $\mathbf{c}_{T+1:T+K}$: 未来 K 帧的 driving controls（包含 ego action, HD map, bounding boxes）
- $\hat{\mathbf{z}}_{T+1:T+K}$: 生成的下一 chunk

关键矛盾在于 **exposure bias**：训练时 context $\mathbf{z}_{1:T}$ 来自 clean GT，但推理时 context 来自 model 自己生成的 $\hat{\mathbf{z}}$。每个 chunk 上累积的小误差递归复合，导致 visual artifacts、geometric inconsistency、semantic drift。

### 1.2 现有方法的瓶颈

Self-Forcing (Huang et al., 2025) 试图在 student 端解决——让 student 训练时就基于自己之前的 outputs 进行 conditioning。但它的 supervision horizon 被 teacher 的 single-pass generation window 卡死。如果直接扩大 teacher 的 single-pass window，DiT 的 attention cost 是 $O(L^2)$ 的——很快超出 memory。

### 1.3 核心洞察：Rollout-Capable Teacher

论文的关键 question：**能否让 teacher 本身通过 AR rollout 扩展到任意 horizon，在 bounded memory 下提供 unbounded supervision？**

难点在于：standard teacher 在自己 predictions 下会 drift，污染它提供的 supervision。

**核心 insight**：先让 teacher 变得 rollout-capable（在自己的 AR predictions 下保持稳定），就能 break the single-pass horizon barrier。

这是一个 **problem reframing**：原本把 AR drift 视为 deployment-time problem，HorizonDrive 把它转化为 training-time problem——把 drift handling 内化到 teacher 的训练中。

---

## 2. 方法架构详解

HorizonDrive 是一个三阶段 framework：

```
Stage 1: Conditional Driving World Model (G_0)
   ↓
Stage 2: Scheduled Rollout Recovery (SRR) → G_roll (rollout-capable teacher)
   ↓
Stage 3: Teacher Rollout DMD (TRD) → G_roll^S (real-time student)
```

### 2.1 Stage 1: Conditional Driving World Model

Backbone 是 Wan 2.1 1.3B T2V (Wan et al., 2025)，full bidirectional attention。架构层面有几个关键设计：

**Video Continuation via Differential Noise Levels**：把每个 training clip 分成 T-frame condition window + K-frame generation chunk，condition latents $\mathbf{z}_{1:T}$ 保持 noise level $t=0$，generation chunk $\mathbf{z}_{T+1:T+K}$ 加噪并用 flow-matching loss 监督（公式3）。

**Disentangled Control**（重要设计）：
- **Spatial structure** (HD map, 3D boxes) → 渲染成 layout tokens $h_{b_f} \in \mathbb{R}^{f \times s \times d}$，通过 zero-initialized projector 加到 DiT features：

$$h_{(t)} \leftarrow h_{(t)} + f_{\text{zero}}(h_{b_f})$$

- **Ego action** $\mathbf{a} = (\Delta x, \Delta y, \Delta\text{yaw})$ → sinusoidal embedding $\phi(\mathbf{a})$ → AdaLN-style gating，产生 6 个 channels，分成 pre-norm shift/scale 和 post-layer residual gate。

**Intuition**：spatial layout 影响 scene composition（应该 additive），ego action 影响 temporal dynamics（应该 modulate via AdaLN）。这种 disentanglement 让不同 condition 不互相干扰。

参考：Disentangled control 思路来自 Ren et al., 2025 (Cosmos-Drive-Dreams, https://arxiv.org/abs/2506.09042) 和 Zhan et al., 2026 (https://arxiv.org/abs/2603.12864)。

### 2.2 Stage 2: Scheduled Rollout Recovery (SRR)

这是论文最核心的创新。让我详细解析：

#### 2.2.1 Rollout Trajectory 作为 Degraded Training Condition

先用 $G_0$ 做 N-step AR rollout 生成 degraded trajectory：

$$\hat{\mathbf{z}}_{s_n+1:s_n+K} = G_0(\hat{\mathbf{z}}_{s_n-T+1:s_n}, \mathbf{c}_{s_n+1:s_n+K}), \quad s_n = T + (n-1)K \tag{6}$$

变量解释：
- $s_n$: 第 n 步 AR rollout 的起始 index
- $N$: rollout 步数
- $\hat{\mathbf{z}}_{s_n-T+1:s_n}$: 第 n 步的 context window，来自之前步骤的 predictions
- 累积产生 trajectory $\hat{\mathbf{z}}_{T+1:T+NK}$

然后 sample 一个 generation boundary $s$，对 condition history 做 swap，supervision target 保持 GT：

$$\tilde{\mathbf{z}}_{s-T+1:s} = \hat{\mathbf{z}}_{s-T+1:s}, \quad \mathbf{z}^{\star}_{s+1:s+K} = \mathbf{z}_{s+1:s+K} \tag{7}$$

**关键 insight**：teacher 在训练时就被暴露在自己 rollout 的 corrupted context 下，但 supervision target 始终是 clean GT。这让 teacher 学到 "如何从 prediction-induced errors 中 recover" 的能力。

#### 2.2.2 Local Pred-to-GT Transition Schedule

公式 (8) 是 blending：

$$\bar{\mathbf{z}}_i = \begin{cases} 
\tilde{\mathbf{z}}_i, & s-T+1 \leq i \leq s-w \\
\alpha_i \tilde{\mathbf{z}}_i + (1-\alpha_i) \mathbf{z}_i^{\star}, & s-w+1 \leq i \leq s+w \\
\mathbf{z}_i^{\star}, & s+w+1 \leq i \leq s+K
\end{cases} \tag{8}$$

变量解释：
- $w$: blending window radius（boundary s 两侧各 w 帧）
- $\alpha_i$: 从 1 线性衰减到 0 的 mixing coefficient
- $\tilde{\mathbf{z}}_i$: rollout prediction（左边历史）
- $\mathbf{z}_i^{\star}$: GT（右边 supervision target）

**Intuition 构建**：如果直接从 prediction jump 到 GT，会产生 temporal discontinuity。模型困惑：是要延续 prediction state？还是要"纠正"到 GT？Blending 在 boundary 附近构造 continuous temporal bridge，让模型平滑过渡。

**$w$ 的 curriculum 设计**：
- 训练初期 $w=0$：sharp boundary，强迫模型直接 recover from large deviation（hard task）
- 训练后期 $w \to 8$：扩大 transition region，task 变成 finer-grained correction（easy task）

**反直觉但合理**：先学 hard recovery（大幅纠正能力），再 refine 到 small correction。这与人类学习修错的直觉一致——先学会从大错中拉回，再学会精细调整。

#### 2.2.3 Global Boundary-Decay Sampling Schedule

这是另一个关键 schedule，控制 boundary $s$ 在 rollout trajectory 上的位置。

**Empirical observation**（论文 Figure 3b/c）：
- Late rollout position（大 $s$，比如 70-90）：error heatmap 更强，semantic drift 更严重
- Early rollout position（小 $s$，比如 10-30）：cross-case cosine similarity 更高，errors 更 generic

**Curriculum**：训练从 large $s$ 开始（处理 severe case-specific semantic drift），逐渐 decay 到 small $s$（处理 generic cross-case degradation）。

**为什么这个顺序？**

我个人的 intuition 是：severe drift 是 harder recovery task。先建立这个能力，模型获得 robustness ceiling。然后 refine 到 small errors 是 easier task，可以建立在 robustness 之上。

如果反过来：先学 small generic errors，模型建立的是 fine correction capability，但当遇到 severe case-specific drift 时，它没有 emergency recovery 能力。这就是 curriculum learning 中"先难后易"的逆向应用——这里"难"指 severe deviation，"易"指 fine correction。

Schedule 详见 Table 4:
- $N(k)$: 10 → 4 (steps 0-8000)，rollout depth 逐渐减小
- $w$: 0 → 8 (steps 0-8000)，blending radius 逐渐增大

#### 2.2.4 SRR 与现有 anti-drifting 方法的对比

| 方法 | 核心思想 | 瓶颈 |
|------|---------|------|
| Frame sinks (Huang et al., 2024a) | 用 keyframes 作为 anchor 防止 drift | 在 driving 中 fast ego-motion 下 transfer 差 |
| Self-Forcing (Huang et al., 2025) | Student-side: 在自己 outputs 上训练 | Supervision horizon 受 teacher single-pass 限制 |
| SRR (本文) | Teacher-side: 让 teacher 自己 rollout-capable | Break single-pass barrier，提供 unbounded supervision |

关键差异：SRR **stabilizes the supervisor**，而传统方法 **regularize the deployed generator**。这个 inversion 是论文最 deep 的贡献。

### 2.3 Stage 3: Teacher Rollout DMD (TRD)

TRD 把 rollout-capable teacher 的 long-horizon 行为 distill 到 real-time student。

#### 2.3.1 Asymmetric Architecture

| | Teacher $G_{\text{roll}}^{\mathcal{T}}$ | Student $G_{\text{roll}}^{S}$ |
|---|---|---|
| Init | SRR weights | SRR weights |
| Chunk size | $K^{\mathcal{T}} = 40$ | $K^{S} = 10$ |
| Denoising steps | multi-step | 4 |
| Trainable | ❌ frozen | ✅ params $\phi$ |
| Context T | 11 | 11 |

**为什么 asymmetric？**

Memory 分析：
- Single-pass $N \times 40$ frames generation：attention $O((N \cdot 40)^2)$ → 爆炸
- AR rollout with fixed $(T, K^{\mathcal{T}})$：每步 $O((T + K^{\mathcal{T}})^2) = O(51^2)$ → bounded

Teacher 长 chunk + multi-step 能 capture longer temporal dynamics，提供 high quality target。Student 短 chunk + few-step 满足 real-time 要求。Teacher frozen 节省 memory（不需要 gradient through teacher）。

#### 2.3.2 Student AR Rollout During Training

公式 (9)：

$$\hat{\mathbf{z}}^S_{s_n+1:s_n+K} = G_{\text{roll},\phi}^{S}(\hat{\mathbf{z}}^S_{s_n-T+1:s_n}, \mathbf{c}_{s_n+1:s_n+K^S}), \quad n=1,\ldots,N \tag{9}$$

Student 自己也做 $N=20$ 步 AR rollout。每当累积 student rollout 覆盖 teacher chunk length $K^{\mathcal{T}}=40$ frames（即每 4 个 student chunks of 10 frames），就 apply DMD gradient 并 backprop。

**关键设计**：每 D 个 student chunks update 一次 DMD gradient（Table 5: $D=5$）。这避免了每个 chunk 都做 expensive distribution matching。

#### 2.3.3 TRD Gradient 公式详解

公式 (10)：

$$\nabla_\phi \mathcal{L}_{\text{TRD}} = \mathbb{E}_\tau \left[ -\left( \underbrace{s_{\text{cond}}^{\text{real}}(z_{(\tau)}) - s_{\text{cond}}^{\text{fake}}(z_{(\tau)})}_{\text{Distribution Matching}} + \underbrace{\mathbf{1}_{\{\tau \leq \tau_{\text{th}}\}}(\alpha-1)\left(s_{\text{cond}}^{\text{real}}(z_{(\tau)}) - s_{\text{uncnd}}^{\text{real}}(z_{(\tau)})\right)}_{\text{Noise-truncated CFG}} \right) \frac{\partial G_{\text{roll},\phi}^{S}}{\partial \phi} \right] \tag{10}$$

变量解释：
- $\phi$: student parameters
- $\tau$: renoised noise level
- $s_{\text{cond}}^{\text{real}}$: real-data discriminator (conditional) score
- $s_{\text{cond}}^{\text{fake}}$: fake-data discriminator (conditional) score
- $s_{\text{uncnd}}^{\text{real}}$: real-data unconditional score
- $\alpha$: CFG scale (default 6)
- $\tau_{\text{th}}$: noise threshold for CFG application
- $\mathbf{1}_{\{\tau \leq \tau_{\text{th}}\}}$: indicator function，只在 low noise level 启用 CFG
- $\partial G_{\text{roll},\phi}^S / \partial \phi$: generator Jacobian

**Distribution Matching 部分**（参考 DMD, Yin et al., 2024b, https://arxiv.org/abs/2312.02639）：让 student 的 output distribution 匹配 teacher 的 output distribution。

**Noise-Truncated CFG 部分**（参考 Decoupled DMD, Liu et al., 2025a, https://arxiv.org/abs/2511.22677）：

标准 DMD 用 CFG 强化 teacher gradient guidance，但在 video rollout 时会 oversaturation。这个改进把 CFG 限制在 low noise levels（$\tau \leq \tau_{\text{th}}$），避免 high-noise CFG 引起的 oversaturation。

**$\tau_{\text{th}}$ 的 schedule**（ablation 验证）：
- **None** ($\tau_{\text{th}}=0$): 不用 CFG，FVD 110.81
- **Full** ($\tau_{\text{th}}=1000$ throughout): 全程 CFG，严重 oversaturation，FVD 184.06（最差！）
- **Early** ($\tau_{\text{th}}$ decays 1000→0 from step 0): FVD 111.99
- **Delayed** ($\tau_{\text{th}}=1000$ for 100 steps warmup, then decay to 0 by step 400): FVD 92.99（最佳！）

**Delayed schedule 的 intuition**：warmup 阶段用 full-range CFG 建立 strong conditional controllability，然后 progressively restrict CFG 到 low noise levels，maintain visual quality。

---

## 3. 实验结果深度分析

### 3.1 主实验对比（Table 1）

**Group (i): Long-horizon interactive world model frameworks**（无 driving control）

| Method | FID↓ | FVD↓ | Img.↑ |
|--------|------|------|-------|
| Matrix-Game3 | 35.69 | 338.22 | 60.44 |
| Helios | 30.53 | 218.23 | 58.82 |
| Causal-Forcing | 49.07 | 373.29 | 59.00 |
| HY-WorldPlay | 33.51 | 580.72 | 58.60 |
| LingBot-World | 37.67 | 325.55 | 55.55 |

这些方法没有 driving-specific control，visual fidelity 严重受限。

**Group (ii): Streaming video generation methods**（re-trained on same base + driving control）

| Method | FID↓ | FVD↓ | ARE↓ | DTW↓ |
|--------|------|------|------|------|
| Self-Forcing | 41.53 | 161.00 | 3.47 | 6.22 |
| Self-Forcing++ | 28.84 | 147.57 | 3.78 | 3.61 |
| LongLive | 29.05 | 161.41 | 3.28 | 3.65 |
| **HorizonDrive** | **13.82** | **92.99** | **2.60** | **3.27** |

**HorizonDrive 相对最强 streaming baselines 的提升**：
- FID 降低 52%（28.84 → 13.82）
- FVD 降低 37%（147.57 → 92.99）
- ARE 降低 21%（3.28 → 2.60）
- DTW 降低 9%（3.61 → 3.27）

### 3.2 Driving-specific Methods 对比（Table 2）

**Short-video (8-25 frames)**：

| Method | Frames | FID↓ | FVD↓ |
|--------|--------|------|------|
| Vista | 25 | 6.90 | 89.40 |
| DreamForge | 16 | 14.61 | 103.61 |
| **HorizonDrive (N=1)** | 21 | 12.54 | **84.53** |

**Long-video (241 frames)**：

| Method | Frames | FID↓ | FVD↓ |
|--------|--------|------|------|
| MagicDrive-V2 (single-pass) | 241 | 20.91 | 94.84 |
| **HorizonDrive (N=20)** | 211 | **13.82** | **92.99** |

**关键观察**：HorizonDrive 用 sequential rollout + few-step denoising 竟然能与 MagicDrive-V2 的 many-step single-pass generation 持平甚至超越。这表明 AR rollout with bounded memory 是 viable 的 long-video generation paradigm。

### 3.3 Ablation Studies (Table 3) - 设计直觉验证

**Initialization ablation**：

| Init (Stu/Tea) | FID↓ | FVD↓ | ARE↓ | DTW↓ |
|----------------|------|------|------|------|
| Base/Base | 19.24 | 141.88 | 2.76 | 3.30 |
| SRR/Base | 20.34 | 128.77 | 3.15 | 3.39 |
| Base/SRR | 14.44 | 107.54 | 2.75 | 3.80 |
| **SRR/SRR** | **13.82** | **92.99** | **2.60** | **3.27** |

**关键 finding**：Base/SRR（仅 teacher 用 SRR 初始化）的 single-factor gain 最大（FID 19.24→14.44）。这证实了 **long-horizon supervision 由 teacher 的 rollout reliability 主导**。Teacher 可靠 rollout 提供高质量 trajectory targets，即使 student 从 Base 起步也能学好。

而 SRR/Base（仅 student 用 SRR）几乎没用，甚至 FID 略变差。**Rollout-aware student alone 无法补偿 teacher 的 drift**——distilling from degraded trajectories 必然得到 degraded student。

**Rollout steps N ablation**：

| N | FID↓ | FVD↓ | ARE↓ | DTW↓ |
|---|------|------|------|------|
| 1 | 21.15 | 135.35 | 3.39 | 5.42 |
| 4 | 18.19 | 139.28 | 3.03 | 3.66 |
| 20 | 13.82 | 92.99 | 2.60 | 3.27 |

**关键 finding**：N=1 时 student 只在 single-chunk generation 上被监督，从未经历自己的 rollout errors。ARE 3.39° 和 DTW 5.42 是 worst。N 增大后 ARE 和 DTW 单调下降，FVD 在 N 足够大时急剧下降（139.28 at N=4 → 92.99 at N=20）。

**Intuition**：longer autoregressive training chains 是 deployment-time stability 的必要条件。Student 必须在训练时就经历 multi-step rollout error accumulation，才能学会 correct 它们。

### 3.4 Self-Collected Dataset (Table 6)

在 internal e2e driving dataset 上（higher ego speeds, more diverse scenarios）：

| Method | FID↓ | FVD↓ | ARE↓ | DTW↓ |
|--------|------|------|------|------|
| LongLive | 28.39 | 374.94 | 4.05 | 8.11 |
| **HorizonDrive** | **12.01** | **117.27** | **3.67** | **5.29** |

证实 SRR + TRD 的设计 transfer 到 substantially different driving distribution。

### 3.5 Error Accumulation Analysis (Figure 5)

论文报告了每个 cumulative chunk 的 FID。**HorizonDrive 在所有 19 chunks 上保持 stable image quality，无 error accumulation**。Self-Forcing++ 的 FID 单调退化，证实了 long-horizon AR generation 的 distribution shift 问题。

### 3.6 Minute-Level AR Generation (Figure 13)

在 self-collected dataset 上，10 FPS rollout，$K^S=10, T=11$，约 1 分钟的连续 rollout，模型保持 coherent road geometry、lane structure、traffic-agent behavior。

**Sliding-window rollout 关键性质**：每步 compute bounded by $T + K^S$，与 total horizon 无关。这是 bounded memory 的核心。

### 3.7 Inference Efficiency

| Resolution | Per-chunk latency | Effective FPS |
|------------|-------------------|--------------|
| 256×512 | 1.8s | ~5.6 FPS |
| 384×768 | 5.8s | ~1.7 FPS |

硬件：单张 NVIDIA 5090 GPU。

---

## 4. 评估指标技术细节

### 4.1 Visual Quality

- **FID** (Fréchet Inception Distance): 基于 Inception-V3 features 的 distribution distance
- **FVD** (Fréchet Video Distance): 基于 I3D video features 的 distribution distance
- **VBench** (Huang et al., 2024b, https://arxiv.org/abs/2310.21250): comprehensive video quality suite

### 4.2 Spatio-Temporal Consistency

这两个 metric 只适用于接受 driving control 的方法。

**ARE (Average Rotation Error)**：用 VGGT (Wang et al., 2025, https://arxiv.org/abs/2503.20251) 从 generated 和 GT videos 恢复 per-frame camera poses，然后计算 predicted 和 GT rotation matrices 之间的 mean geodesic distance。

Geodesic distance 公式：$d(R_1, R_2) = \arccos\left(\frac{\text{tr}(R_1^T R_2) - 1}{2}\right)$

反映 heading accuracy。

**DTW (Dynamic Time Warping)** (Keogh & Pazzani, 2000, https://dl.acm.org/doi/10.1145/347003.335376)：
- 通过 non-rigid time warping 对齐 predicted 和 GT ego-motion trajectories
- 计算 optimal alignment 下的 cumulative Euclidean distance
- 捕获 path-shape fidelity，即使有时间 misalignment 也能比较

这比单纯 trajectory error更鲁棒，因为 AR rollout 可能有 temporal drift 但 path shape 正确。

---

## 5. 个人 Intuition 构建

### 5.1 为什么 SRR 在 teacher 端做 correct 比 student 端做更有效？

我的理解：distillation 本质是 supervision signal 的 transfer。如果 supervision 本身（teacher 的 rollout）有 drift，再好的 student training recipe 也无法补救。"Garbage in, garbage out"。

Self-Forcing 的根本限制：teacher 在 single-pass window 内是 clean 的，但 student 在 rollout 时经历的 context 是 student 自己生成的（不是 teacher 的）。即使 student 学习 match teacher 的 single-pass output，它在 multi-step rollout 下的 context 分布与训练时不同。Supervision horizon 还是 bounded by teacher's single-pass window。

HorizonDrive 的 inversion：让 teacher 自己 rollout-capable。Teacher 在自己 predictions 下也能 stable generation，于是 teacher 的 supervision 可以 extend 到任意 horizon。Student 现在能学到 true long-horizon distribution。

### 5.2 Local + Global Schedule 的协同

Local schedule 处理的是 **single-step recovery 的 granularity**：
- 大 deviation 时 sharp boundary（hard recovery）
- 小 deviation 时 smooth transition（fine correction）

Global schedule 处理的是 **trajectory 上的 error distribution**：
- Late rollout: severe, case-specific semantic drift
- Early rollout: mild, generic degradation

两者协同：global 决定 sample 哪个 rollout position（决定 error severity），local 决定在该 position 如何 transition from prediction to GT（决定 recovery granularity）。

**为什么先 global large s + local w=0？**：先强迫模型处理 severe late-rollout drift with sharp boundary——这是 hardest task。建立 robustness ceiling。然后 global decay to small s + local w↑——处理 mild generic error with smooth transition，这是 refinement task。

### 5.3 Asymmetric Teacher-Student 的 Memory 智慧

如果对称设计（teacher 和 student 同样 chunk size），要么：
1. Teacher 短 chunk：supervision horizon 还是受限
2. Student 长 chunk：inference latency 高，破坏 real-time

Asymmetric 打破这个 trade-off：teacher 长 chunk 多 step（提供 quality supervision），student 短 chunk few step（提供 fast inference）。

**Memory 分析**：
- Teacher: 每 AR step compute $O((T+K^\mathcal{T})^2) = O(51^2)$，frozen 不需要 gradient
- Student: 每 AR step compute $O((T+K^S)^2) = O(21^2)$，需要 gradient through 4 denoising steps
- 总 training memory: bounded，与 N 无关

### 5.4 CFG Augmentation 在 Distillation 中的微妙角色

CFG (Classifier-Free Guidance) 在 standard DMD 中用于 strengthen teacher gradient guidance。但论文发现 full CFG 在 video rollout 时 oversaturation。

**为什么 oversaturation？**

CFG 本质是 $\hat{\epsilon} = (1+\omega)\epsilon_\theta(z, c) - \omega \epsilon_\theta(z, \emptyset)$，用 condition 和 unconditional 的 difference 放大 condition 信号。在 video generation 中，每个 chunk 都受 CFG 影响，cumulative over AR steps 会放大 color/contrast saturation。

**Noise-truncated CFG 的 intuition**：
- High noise levels ($\tau$ 大): latents 接近 noise，CFG 影响整体 structure
- Low noise levels ($\tau$ 小): latents 接近 data，CFG 影响 fine details

在 high noise 用 CFG 会 propagate 到后续 chunks，导致 oversaturation 累积。Restricted 到 low noise，CFG 只影响当前 chunk 的 fine details，不会 propagate。

**Delayed schedule 的 warmup 逻辑**：先用 full-range CFG 建立 strong conditional controllability（让模型学会响应 condition），然后 decay 到 low noise only，maintain visual quality。

---

## 6. Limitations 和 Future Work

论文承认 SRR 是 offline 的——rollout trajectory cache 每隔 R=2000 optimizer steps 刷新一次。这有一些问题：
1. Training 中的 rollout 不能反映 student 最新的 capability
2. Cache refresh 有 latency

**Future direction**: Online rollout-recovery training，world model continuously improves AR robustness from its own interaction trajectories。这类似于 DAgger (Dataset Aggregation) 在 imitation learning 中的思想——agent 在自己 trajectories 上收集错误并修正。

参考类似思想：
- DAgger: Ross et al., 2011 (https://arxiv.org/abs/1011.0686)
- Test-Time Training for video: Dalal et al., 2025 (https://arxiv.org/abs/2507.27471)

---

## 7. 与相关工作的 broader context

### 7.1 Driving World Models 谱系

- **Gaia-1/Gaia-2** (Hu et al., 2023; Russell et al., 2025, https://arxiv.org/abs/2503.20523): early generative world models for autonomous driving
- **Vista** (Gao et al., 2024b, https://arxiv.org/abs/2405.17398): generalizable driving world model with versatile controllability
- **MagicDrive-V2** (Gao et al., 2025a): high-resolution long video generation with adaptive control
- **Cosmos-Drive-Dreams** (Ren et al., 2025, https://arxiv.org/abs/2506.09042): scalable synthetic driving data with world foundation models

### 7.2 Long Video Generation 方法谱系

- **Training-free**: FreeNoise (Qiu et al., 2023), FIFO-Diffusion (Kim et al., 2024)
- **Rollout-aware training**: Self-Forcing (Huang et al., 2025, https://arxiv.org/abs/2506.08009), Causal-Forcing (Zhu et al., 2026), LongLive (Yang et al., 2025, https://arxiv.org/abs/2509.22622)
- **Frame sinks**: StoryDiffusion (Zhou et al., 2024), In-context LoRA (Huang et al., 2024a)
- **Test-time training**: One-Minute Video TTT (Dalal et al., 2025)

### 7.3 Diffusion Distillation 谱系

- **DMD** (Yin et al., 2024b, https://arxiv.org/abs/2312.02639): distribution matching distillation
- **DMD2** (Yin et al., 2024a, https://arxiv.org/abs/2405.14867): improved DMD
- **Decoupled DMD** (Liu et al., 2025a, https://arxiv.org/abs/2511.22677): CFG augmentation + distribution matching
- **Consistency Models** (Song et al., 2023, https://arxiv.org/abs/2303.01969): consistency-style distillation
- **Mean Flows** (Geng et al., 2025, https://arxiv.org/abs/2505.13447): one-step generative modeling

---

## 8. 总结：HorizonDrive 的核心贡献

1. **Identification of missing prerequisite**: rollout-capable teaching 是 scalable long-horizon distillation 的前提条件。
2. **SRR**: 把 standard driving world model 转化为 stable AR teacher，通过 local pred-to-GT transition + global boundary-decay sampling 两个协同 schedule。
3. **TRD**: 在 bounded memory 下把 teacher 的 long-horizon AR behavior distill 到 short-chunk, few-step student，通过 asymmetric chunk sizes + noise-truncated CFG。

**核心 inversion**：传统 anti-drifting 方法 regularize the deployed generator；HorizonDrive stabilizes the supervisor。这个思想 inversion 是论文最 deep 的贡献，可能启发 future work 在其他 long-horizon generation 任务（如 robotics, embodied AI）中的应用。

**实测数据**：在 nuScenes 上 FID 降低 52%、FVD 降低 37%，ARE 和 DTW 分别降低 21% 和 9%。Minute-scale rollout 在 self-collected dataset 上 demo。Closed-loop simulation with self-generated ego trajectories 也 viable。

Project page: https://zcliangyue.github.io/HorizonDrive
