---
source_pdf: FM-VLA Force-based Memory for Vision-Language.pdf
paper_sha256: 705e82070aa5fe4f189d766855734afaa208a93245ab19cb1570d8b902a02691
processed_at: '2026-08-18T22:32:18-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 FM-VLA

好, 我尽量用人话把这篇 paper 的核心 idea 说清楚。

## 一句话版本

**让 robot 记住"我到现在为止摸了啥、碰了啥、按了多少下", 靠的是 force sensor 的时间序列, 不是回头看录像。**

## 为什么这件事重要

现在 VLA model 基本上都是"金鱼脑"——看一眼当前画面, 直接输出 action, 7秒后就忘了刚才干了啥。这对大多数 pick-and-place 任务 OK, 但下面这些 task 直接跪:

- 你让 robot "按按钮 3 次然后停"。按钮按下去就弹回来, 视觉上**每一帧长得一模一样**。你给它看 100 帧过去录像, 它也数不清到底按了几下, 因为视觉上根本没区别。
- 你让 robot "擦碗擦 3 圈"。擦完一圈碗还是那个碗, sponge 回到起点, 视觉上 zero information gain。
- 你让 robot "掀杯子找 hidden block"。掀了一个杯子没找到, 放回去, 场景**视觉上完全回到初始状态**。它怎么知道哪个杯子翻过了?

这些 task 的共同点: **关键 information 不在画面里, 在 physical interaction 里**。

人类做这些 task 靠什么? 靠手指头的触觉。按按钮有 click 的 impulse, 擦碗有 friction 的周期 pattern, 翻杯子有 lift 的 force profile。你的手指就是一个 event counter, 这些信号在 force/torque sensor 里全都有, 而且比 visual frame 便宜 100 倍——force 就 6 个数字 per timestep, RGB frame 是几百万个 pixel。

## 核心做法

整个 pipeline 说白了就三步:

### Step 1: 训一个 force "压缩器" (Force-VAE)

Robot wrist 上的 6-axis force sensor 以 100Hz 吐数据, 一个 episode 可能几十秒, 就是几千个 6D 数据点。直接塞给 VLA model 太多, model 学不会用。

所以先单独训一个 VAE, 干一件事: **吃进去一长串 force 时间序列, 压成 8 个 token, 再从这 8 个 token 重建出原来的序列**。

这就像让一个 student 先做 "读厚书→写摘要→从摘要复述厚书" 的训练。训练完之后, 这个 student 写的 8-token 摘要就 capture 了序列里所有重要的 macro structure——按了几次 button、每次多大的 impulse、什么时间发生的——因为如果没 capture, 它就重建不出来。

这个 VAE 是 **task-agnostic** 的, 所有 task 的 force data 混一起训, 不需要 label, 不需要告诉它 "这是按按钮" 还是 "擦碗"。它自己从 reconstruction 里学到 force 的 general structure。

### Step 2: 把 force 摘要塞进 VLA

拿到 8 个 force memory token 之后, 直接 append 到 π0.5 action expert 的 suffix。就跟给 model 加了一个 "post-it note" 一样——"嘿, 到目前为止你的 force history 长这样"。

这个 post-it note 跟 visual frame 走完全不同的通路: visual frame 进 VLM (PaliGemma), force memory token 直接进 action expert 的 suffix。这样 force memory 不跟 vision token 抢 attention bandwidth, 也不破坏 base model 的 RoPE positional encoding。

### Step 3: 加一个 short-term state memory 补刀

光有 force memory 有个 bug: force 是 episodic 的——按按钮那一下有 force, 按完抬起手的中间几秒没 force, policy 不知道手现在在哪、该怎么移动到下一个 press 位置。Force-only ablation 在 Buttons task 上 0% success, 就是这个原因。

解决: 再加一个 very lightweight 的 linear projection, 把最近 0.9 秒的 joint position state 压成 1 个 token, 跟 force memory 一起 append。这个 state token 告诉 policy "手现在在哪、最近在往哪动"。

Force 管 "宏观历史" (按了几次), state 管 "微观当下" (手在哪), 两者互补缺一不可。

## 几个非常聪明的小 trick

### Trick 1: Random noise pre-padding

训练时在 force history 前面塞一段随机长度的噪声。为什么? 因为如果不塞, model 会发现 "序列越长 = episode 越靠后 = 应该停了" 这个 shortcut。它就不用真的去理解 force content, 直接看 sequence length 就行。

这就像考试时老师故意把试卷页数搞乱, 防止学生靠 "做到第 5 页就该交卷了" 这种 cue 偷懒。

### Trick 2: Free-bits KL

VAE 训练有个经典坑叫 posterior collapse——latent 退化成标准 normal, 啥信息都不 encode。Free-bits 的做法: 给每个 latent dimension 一个 "最低信息 budget" (0.5 nats), 在 budget 以下时关掉 KL gradient, 让 reconstruction loss 自由地把信息塞进去。等 latent 真的 encode 了够多信息再恢复 KL 约束。

### Trick 3: Suffix injection 保持 RoPE

把 memory token 放在 noisy-action tokens **之后** 而不是之前, 保证 noisy-action tokens 的 RoPE position 跟 base model pretraining 时完全一致。这是对 base model 最大的尊重——你是在它已有的 "舒适区" 外面加东西, 而不是 shift 它内部的 positional structure。

## 结果有多好

| Method | 平均成功率 | 备注 |
|---|---|---|
| π0.5 (memoryless baseline) | 27.8% | 按钮和擦碗几乎全挂 |
| TA-VLA (short force window) | 22.2% | short window ≠ memory |
| π-MEM (visual memory baseline) | 53.7% | 视觉看不出来按了几次 |
| **FM-VLA (ours)** | **83.3%** | force memory 真的能 count |

而且 inference latency 只比 base model 多 3.3ms, 而 visual memory baseline 多 39ms (K=5) 到 129ms (K=16)。Force 是 6D 信号, visual frame 是几百万 pixel, 计算量天差地别。

## 为什么 VAE 比 GRU 和 Q-Former 都好

这个 ablation 我觉得特别有启发性。

- **GRU 在长 100Hz 序列上 vanishing gradient**, 早期的 contact event 全忘, Wipe task 只有 5.6%。RNN 处理长序列的老毛病。
- **Q-Former end-to-end 训练, overfit 到 instantaneous peaks**, 没学到 holistic pattern, Buttons 只有 16.7%。因为它没有 reconstruction objective, loss signal 只来自下游 task, 监督信号太弱, model 走捷径只看 peak。
- **VAE 因为有 reconstruction pretraining**, latent 被强制 encode 整个序列的 macro structure, 不是局部 peak。这跟 BERT 的 MLM、MAE 的 image reconstruction 是一个套路——self-supervised pretraining 给一个 "honest" objective, 防止 downstream short-cut。

更深一层的 intuition: **当你想把 long sequence 压成 fixed-size latent 给 downstream 用, 先 self-supervised pretrain encoder 是关键**。这跟 LLM 的 pretrain→fine-tune 范式完全一致, FM-VLA 把它 apply 到 force signal 上。

## 人话总结

Paper 的 conceptual contribution 其实就一句话: **memory modality 应该 match task 的 information bottleneck**。

Visual memory 假设 "看到的过去 = 知道的过去", 但 contact-rich 任务里这个假设 fail。按钮按 3 次和按 1 次在视觉上 identical, 但在 force 上截然不同。这时候 memory 必须 ground 在 physical interaction 里, 不是在 visual appearance 里。

技术上它做了三件事:
1. 用 VAE self-supervised pretrain 把 long force history 压成 8 个 informative token
2. 用 suffix injection 把这些 token 加进 VLA, 不破坏 base model
3. 用 short-term state memory 补 force 的 episodic 盲区

这是个非常 elegant 的 work, 不堆参数不堆 data, 而是 **找到对的 modality + 对的 representation + 对的 injection point**。我觉得它会启发一批把 tactile, audio, EMG 等 non-visual signal 当 long-horizon memory 的工作。

---

**主要 reference**:
- [FM-VLA Project Page](https://qft-333.github.io/FM-VLA-Page/)
- [π0.5 base model](https://arxiv.org/abs/2504.16054)
- [Perceiver IO (VAE backbone)](https://arxiv.org/abs/2107.14795)
- [Free-bits improved VAE](https://arxiv.org/abs/1606.04934)
- [TA-VLA (short force baseline)](https://arxiv.org/abs/2509.07962)
- [MEM (visual memory baseline)](https://arxiv.org/abs/2603.03596)
- [MemoryVLA](https://arxiv.org/abs/2508.19236)
- [ForceVLA](https://arxiv.org/abs/2505.22159)

---

# FM-VLA: Force-based Memory for Vision-Language-Action Models

这是一篇非常 solid 的工作。来自 Tsinghua + Microsoft Research + Fudan + USTC 的合作，第一作者群很庞大（共同一作有 7 位）。核心 insight 我觉得非常 elegant：**用 force/torque wrench signal 作为 long-horizon memory，而不是 visual frame**。这个 idea 在 contact-rich manipulation 里非常合理，因为人类做这类任务时其实也是靠触觉+本体感觉来计数和追踪进度，而不是靠视觉。下面我尽量把这篇 paper 拆解到直觉层面，并尽可能多联想一些相关工作。

## 1. 问题动机：为什么需要 force memory？

现有的 VLA model（RT-2 [1], OpenVLA [2], π0 [3], π0.5 [4], RDT-1B [14], CogACT [16], GR00T N1 [17], SpatialVLA [18], GR-3 [19] 等）绝大多数都是 **Markovian policy** $\pi(a_t \mid o_t, l)$，即只看当前 observation $o_t$ 和 language instruction $l$，直接预测下一个 action chunk $a_t$。这个 assumption 在 single-step, visually grounded 的任务上没问题，但在三种场景下会塌掉：

1. **隐状态 / 部分可观**：例如一个 block 藏在两个倒扣的杯子下面，看完一个杯子放回去后场景视觉上回到初始状态，policy 必须 "记得" 哪个杯子已经看过。
2. **重复动作的计数**：按按钮 N 次、擦碗 N 圈，每次的视觉位移几乎不可见（按钮 mechanical click displacement 通常 < 1mm），视觉 frame 之间没有可识别的差异。
3. **进度追踪**：multi-step 任务里需要知道当前在 sequence 的哪一步。

之前的 memory-augmented VLA 走两条路：
- **Language-based memory**：把过去总结成 text，类似 scratchpad [25]
- **Vision-based memory**：MemoryVLA [5] 维护 explicit memory bank，MEM [6] 用 multi-scale（short video + long text），还有 ReMem-VLA [20], EchoVLA [21], VPWEM [22], VQ-Memory [23], TempoFit [24], Notes-to-self [25], dual-memory [26], object-centric memory [27] 等等

这些方法的根本缺陷：**当任务相关 state change 在视觉上 invisible 或 ambiguous 时，再多的 visual frame 也救不了**。这就像让一个闭着眼睛的人去数自己按了多少次按钮——他靠的是指尖的触觉 impulse，而不是任何视觉 evidence。

FM-VLA 的核心洞察：**wrench signal (force + torque) 天然 capture contact events**。按钮的每次 click 会产生 sharp impulse，擦碗每次来回会产生 periodic force pattern，这些都是 visual 上 hidden 但 force 上 crystal clear 的信号。更重要的是，force 是 6D low-dim signal at 100Hz，比 RGB frame 便宜得多。

## 2. 与现有 force-augmented VLA 的区别

这里需要 clarify 一下，因为最近 force/tactile modality 在 VLA 里非常火。相关工作的谱系：

- **ForceVLA [7]**: 把 6-axis wrench 作为 first-class modality，用 MoE 跟 vision-language tokens 融合。但只用 instantaneous force。
- **TA-VLA [8]**: 系统探索 torque-aware VLA design space，提出 action decoder-side torque adapter，window 是 short-term。
- **ForceFlow [28]**: contact-driven flow matching
- **TacVLA [29]**: contact-aware tactile fusion
- **HapticVLA [30]**: 推理时不直接用 tactile sensing
- **AT-VLA [31]**: adaptive tactile injection
- **TaF-VLA [32]**: tactile-force alignment
- **VTLA [33]**: vision-tactile-language with preference learning
- **DreamTacVLA [34]**: world-model-style force prediction

这些工作的共同特点：**force 只作为 short-window instantaneous conditioning**，回答 "现在有没有接触？现在 force 多大？" 这种 local 问题。FM-VLA 是第一个把 wrench stream 当成 **long-horizon memory** 用的，回答 "我到目前为止按了多少次按钮？" 这种全局历史问题。

这个区分很重要：instantaneous force = 状态估计，memory force = 累积事件 tracking。

## 3. 方法论详解

### 3.1 Problem Formulation

目标 policy 形式：
$$\pi(a_t \mid o_t, l, h_t)$$

其中：
- $a_t$: action chunk at time $t$（chunk 长度 $H=30$，action 维度 $d_a=32$，对应 bimanual 7-DoF arms + 2 1-DoF grippers + ... 具体见 [4]）
- $o_t$: current observation（3 个 RGB stream：head + 2 wrist cameras）
- $l$: language instruction
- $h_t$: **temporal history**，是 FM-VLA 新增的

Vanilla VLA 是 $h_t = \emptyset$，memoryless。

FM-VLA 把 $h_t$ 拆成两个互补的 proprioceptive stream：

**Long-horizon wrench history** $\{f_\tau\}_{\tau=1}^{t}$，其中 $f_\tau \in \mathbb{R}^{d_f}$，$d_f=6$（3-axis force + 3-axis torque），来自 wrist-mounted 6-axis F/T sensor at 100Hz。这个 stream 是 **unbounded length**——会随着 episode 进行越来越长，capture 整个 episode 的 contact event 历史。

**Short-window joint-state history** $\{s_\tau\}_{\tau=t-W+1}^{t}$，其中 $s_\tau \in \mathbb{R}^{d_s}$，$d_s=16$ for 7-DoF bimanual + 2 1-D grippers。这个 stream 是 **bounded length $W$**，只看最近的 motion context。

最终 temporal history representation：

$$
h_t = \left[\underbrace{\text{Enc}_\phi(\{f_\tau\}_{\tau=1}^{t})}_{Z_f \in \mathbb{R}^{K \times d_h}} \parallel \underbrace{\text{Proj}_\psi(\{s_\tau\}_{\tau=t-W+1}^{t})}_{z_s \in \mathbb{R}^{d_h}}\right] \tag{1}
$$

变量解释：
- $\text{Enc}_\phi$: pretrained VAE encoder，参数 $\phi$
- $\text{Proj}_\psi$: lightweight linear projection，参数 $\psi$，end-to-end 学习
- $Z_f$: wrench memory，$K=8$ tokens，每个 dim $d_h$
- $z_s$: 单个 state memory token
- $\parallel$: sequence-wise concatenation
- $K=8$, $W=10$ (stride 3, 约 0.9s)

直觉上：$Z_f$ 告诉 policy "到目前为止发生了哪些 contact events"，$z_s$ 告诉 policy "手现在在哪、最近怎么动"。两者缺一不可。

### 3.2 Architecture Overview

Base model 是 π0.5 [4]，由 PaliGemma [35] (with SigLIP [36] vision encoder) + flow matching [37] action expert 组成。VLM 处理 current image + language，通过 cross-attention condition action expert 生成 action chunk。

FM-VLA 在 action expert 的 suffix 位置注入 memory tokens。完整的 action expert sequence layout：

$$
\underbrace{[a_k^{(1)}, \ldots, a_k^{(H)}]}_{\text{noisy-action tokens}} \parallel \underbrace{[Z_f^{(1)}, \ldots, Z_f^{(K)}]}_{\text{wrench memory}} \parallel \underbrace{[z_s]}_{\text{state window}} \tag{3}
$$

放在 noisy actions **之后** 是一个很 clever 的设计选择：保持 noisy-action tokens 的 RoPE positions 跟 base π0.5 pretraining 时一致，从而最小化对原模型的破坏。这有点像 prefix-tuning 的反向版本——suffix tuning，类似 LoRA 的 philosophy，但作用在 sequence level。

### 3.3 Wrench History Processing

这一小节有两个非常实用的 trick：

**First-order EMA smoothing**:
$$\tilde{f}_\tau = \alpha f_\tau + (1-\alpha) \tilde{f}_{\tau-1}$$

$\alpha=0.3$，causal（只用过去信息），去除高频 noise 同时保留 contact onset 和 peak。这比简单的 moving average 更适合 real-time policy，因为 moving average 会引入 lag。

**Randomized noise pre-padding**：训练时在每个 history 前面 prepend 随机长度的高斯噪声（uniformly sampled up to 10s，σ=0.05 after quantile normalization）。推理时关闭。

这个 trick 的目的是什么？我读的时候想了挺久。原来是因为 **wrench history 的长度会泄露 episode 进度**！比如按了 3 次按钮的 history length 一定比按了 1 次的长。如果模型发现 "sequence 越长 → 应该停了"，它就会走 shortcut 不去真正理解 signal content。Random pre-padding 把这个 length cue 抹掉了，强迫模型 attend to signal 的 temporal structure 而不是 length。这个 trick 应该可以推广到很多 long-horizon sequence modeling 场景。

### 3.4 Force-VAE Encoder

这是 paper 的核心模块。用 Perceiver-IO [10] 作为 encoder/decoder backbone，因为 Perceiver 天生适合把 unbounded length input 压缩成 fixed set of latent tokens（cross-attention 从 input sequence 提取信息到 learnable latent queries）。

Encoder 流程：
1. Wrench history $F = [f_1, \ldots, f_T] \in \mathbb{R}^{T \times 6}$ 输入
2. 每个 timestep 的 $f_t$ 先 quantile-normalize（基于全 dataset 统计），这一步很重要因为不同 task 的 force magnitude 差异巨大（button click impulse 远大于 wiping steady force）
3. 通过 input MLP 投影到 384-dim token
4. 加 Fourier positional encoding（32 bands, $f_{\max}=1500$），告诉 Perceiver 时间顺序——Perceiver 本身是 permutation invariant 的，必须靠 PE 注入 temporal info
5. Encoder 是 2 个 cross-attention blocks (input → K=8 latent queries) + 10 个 self-attention blocks (over latents)
6. 每个 latent token 通过一个 linear head 输出 posterior parameters:

$$
(\mu_k, \log \sigma_k^2) = \text{Head}_{\text{VAE}}(\text{Enc}_\phi(F)_k), \quad z_k = \mu_k + \sigma_k \odot \epsilon_k, \quad \epsilon_k \sim \mathcal{N}(0, I) \tag{2}
$$

变量解释：
- $\mu_k \in \mathbb{R}^{d_z}$: posterior mean for $k$-th latent token
- $\sigma_k \in \mathbb{R}^{d_z}$: posterior std for $k$-th latent token
- $\log \sigma_k^2$: 用 log space 保证 std 正定
- $\epsilon_k$: reparameterization noise (standard normal)
- $\odot$: element-wise (Hadamard) product
- $d_z = 96$: per-latent dimension
- $K = 8$: latent token count

Decoder 是镜像结构，2 个 cross-attention blocks，让 time-step-specific Fourier-encoded queries attend to latent tokens $Z$ 输出 reconstructed sequence $\hat{F} \in \mathbb{R}^{T \times 6}$。

为什么要用 VAE 而不是单纯的 autoencoder？我猜主要是为了：
- **Regularized latent space**：KL regularization 让 latent distribution 接近 prior，避免 latent space 过度拟合训练数据
- **Smooth interpolation**：在 latent space 里不同 contact event count 的表示会连续过渡
- **Posterior collapse 防御**：通过 free-bits trick（见下）让 latent 真的 encode 信息

### 3.5 Short State History

这个分支比 force 分支简单很多——一个 zero-initialized linear layer，把 last 10 frames 的 joint state (stride 3, ~0.9s, $10 \times 16 = 160$-dim) flatten 后投影成单个 $d_h$-dim token。

为什么不用第二个 VAE？我的理解是：
- Joint state 信号已经被 π0.5 在大规模预训练时大量见过，结构 well-known
- 不需要 representation learning，linear projection 足够
- Short window 已经 capture 了所需信息

### 3.6 Two-Stage Training

**Stage 1: Force-VAE pretraining**。Masked-ELBO objective:

$$
\mathcal{L}_{\text{VAE}} = \frac{1}{\sum_\tau m_\tau \cdot d_f} \sum_{\tau=1}^{T} m_\tau \|f_\tau - \hat{f}_\tau\|^2 + \beta \cdot \frac{1}{K d_z} \sum_{k,j} \max(D_{\text{KL}}^{(k,j)}, \lambda) \tag{4}
$$

变量解释：
- $m_\tau \in \{0,1\}$: padding mask，1 表示有效帧
- $f_\tau$: ground truth wrench at step $\tau$
- $\hat{f}_\tau$: reconstructed wrench
- $\|\cdot\|^2$: L2 重建误差
- $d_f = 6$: wrench 维度
- $T$: 序列长度
- $D_{\text{KL}}^{(k,j)}$: per-dimension KL divergence of $\mathcal{N}(\mu_{k,j}, \sigma_{k,j}^2)$ vs 标准 normal prior
- $\beta = 1 \times 10^{-3}$: KL weight
- $\lambda = 0.5$ nats: free-bits floor
- $K = 8, d_z = 96$

这个 free-bits KL trick 来自 [VAE improvement literature](https://arxiv.org/abs/1606.04934)：标准 KL 容易导致 posterior collapse——latent 退化成 prior，完全不 encode 信息。Free-bits 的做法是：每个 latent dimension 至少保留 $\lambda$ nats 的 information，如果某 dim 已经 collapse（$D_{\text{KL}} < \lambda$），就**关掉它的 KL gradient**，让 reconstruction loss 继续推它 encode 信息，直到它达到 $\lambda$ nats 之后再恢复 KL 梯度。这是一个非常实用的工程 trick。

**Stage 2: VLA finetuning**。Freeze VAE encoder，只用 posterior mean $\mu_f$（inference mode，不采样 noise）。Flow matching objective 跟 π0.5 一致（rectified flow [38]）：

$$
\mathcal{L} = \mathbb{E}_{a_0, \epsilon, k} \left[\left\| v_\theta(a_k, k, c_t, Z_f, z_s) - (\epsilon - a_0) \right\|^2 \right] \tag{5}
$$

变量解释：
- $a_0$: clean action chunk 从 dataset 采样（30-step, 32-dim）
- $\epsilon \sim \mathcal{N}(0, I)$: Gaussian noise
- $k \in [0,1]$: noise level
- $a_k = (1-k) a_0 + k \epsilon$: interpolated sample（rectified flow 的直线插值）
- $v_\theta$: velocity prediction network（policy 本身）
- $c_t$: VLM conditioning (vision-language features via cross-attention)
- $Z_f, z_s$: force 和 state memory tokens
- $(\epsilon - a_0)$: constant velocity target of the straight-line path from $a_0$ to $\epsilon$

Rectified flow 比 DDPM 优越的地方是它走直线，所以 inference 时可以少步数。FM-VLA eval 时只用 10 步 flow matching，速度很快。

## 4. 实验详解

### 4.1 实验设置

Robot platform: **AgiBot G1** [39] bimanual humanoid，两个 7-DoF arms + 两个 1-DoF grippers，每个 wrist 装一个 6-axis F/T sensor at 100Hz。这是国产 humanoid 里比较 well-instrumented 的一个，特别适合 contact-rich 实验。

三个 task 设计得非常 thoughtful，分别测试不同的 memory 类型：

**Task 1: Find a Block Under Two Cups** (200 demos)
- 两个 visually identical 倒扣杯子，block 藏在其中一个下面
- 必须 front-to-back 顺序检查：先 lift 前杯，找到 block 就用 left hand pick，没找到就放回去再 lift 后杯
- 完成后必须把当前持有的杯子放回
- **关键约束**：不能 re-inspect 已 lift 过的杯子
- Memory 类型：**spatial + partially-observed state**

**Task 2: Push Buttons** (350 demos)
- 蓝色按钮，instruction 指定 $N \in \{1, 2, 3\}$
- 用 right gripper 闭合后 press N 次，press 必须深到 trigger audible click
- 完成后 open gripper 信号
- **关键约束**：button 位移极小，视觉上看不出
- Memory 类型：**contact event counting**

**Task 3: Wipe Dishes** (200 demos)
- 拿 sponge wipe bowl 内部 N 圈（back-and-forth = 1 round）
- $N \in \{1, 2, 3\}$
- 每圈必须保持接触，partial pass 不算
- Memory 类型：**periodic contact pattern counting**

### 4.2 Main Results

Table 1 的核心数据：

| Method | Task 1 (Cups) | Task 2 (Buttons) | Task 3 (Wipe) | Average |
|---|---|---|---|---|
| π0.5 (no history) | 72.2 | 11.1 | 0.0 | 27.8 |
| TA-VLA [8] | 50.0 | 11.1 | 5.6 | 22.2 |
| π-MEM [6] | 77.8 | 33.3 | 50.0 | 53.7 |
| FM-VLA (VAE, ours) | **100.0** | **72.2** | **77.8** | **83.3** |

观察：
- **Task 1 上 memoryless π0.5 已经有 72.2%**：因为 task 1 主要是 spatial reasoning，视觉上确实有信息（哪个杯子在前面），π0.5 的 VLM 能处理一部分。但仍然失败 5/18，说明 memory 确实有帮助。
- **TA-VLA 在 task 2 和 3 上几乎全失败**：short force window (10 frames, ~0.9s) capture 不到 long-horizon counting。这印证了 paper 的核心论点——**instantaneous force ≠ force memory**。
- **π-MEM 在 Buttons 上只有 33.3%**：visual memory 完全 fail 在 visually-ambiguous contact task 上，这是 FM-VLA 最大的 win 点。
- **FM-VLA 在 Buttons 上 72.2%**：还有提升空间，说明 VAE 压缩仍有 information loss。

### 4.3 Ablation: 为什么是 Force + State？

| Ablation | Cups | Buttons | Wipe | Avg |
|---|---|---|---|---|
| Force only | 55.6 | 0.0 | 22.2 | 25.9 |
| State only | 100.0 | 11.1 | 11.1 | 40.7 |
| FM-VLA (full) | 100.0 | 72.2 | 77.8 | 83.3 |

这个 ablation 非常有启发性：
- **Force only 在 Buttons 上 0%**！这看似反直觉——明明 button press 会产生 force impulse，为什么只用 force 反而学不会？
  
  Paper 给的解释：**没有 short-term state context，policy 在 contact 前的 motion 是 erratic 的**。意思是 force memory 告诉模型 "已经按了 2 次"，但模型不知道手现在在 button 上方哪里、应该怎么动到 button 上方准备按第 3 次。Force 是 episodic 的（按下去那一瞬间有信号），但按钮之间手要抬起、移动、再按下，这中间没有 force，policy 就迷失了。

- **State only 在 Cups 上 100%**：因为 Cups task 主要是 spatial reasoning，state history 包含 cup 已经 lift 过的信息（arm 位置不同），足以解决。
- **State only 在 Buttons/Wipe 上失败**：state 没法 count，因为每次按完按钮 arm 回到同一位置，state 看起来 identical。

这印证了 intuition：**Force = long-horizon episodic memory, State = short-horizon spatial context**，两者 truly complementary。

### 4.4 Ablation: 为什么 VAE？

| Architecture | Cups | Buttons | Wipe | Avg |
|---|---|---|---|---|
| GRU | 55.6 | 38.9 | 5.6 | 33.3 |
| Q-Former | 100.0 | 16.7 | 55.6 | 57.4 |
| VAE (ours) | 100.0 | 72.2 | 77.8 | 83.3 |

- **GRU 在 Wipe 上只有 5.6%**：vanishing gradient on long 100Hz sequence，早期 contact event 丢失。这是 RNN 在长序列上的经典问题。
- **Q-Former 在 Buttons 上 16.7%**：end-to-end 训练，overfit 到 instantaneous peaks 而不是 holistic temporal structure。Q-Former [41] 来自 BLIP-2，本身设计是 cross-attention 从 image 提取信息到 learnable queries，但没有 reconstruction objective 时容易学 short-cut。
- **VAE 显著超越两者**：pretrained reconstruction objective 强制 latent encode macroscopic structure（force magnitudes, onset timings, contact counts），不是局部 peak。

这个对比让我想到一个更一般的原则：**当你要把 long sequence 压成 fixed-size latent 给 downstream 用时，pretrained self-supervised reconstruction objective 是关键**。这跟 BERT 的 MLM、MAE 的 image reconstruction、wav2vec 的 speech reconstruction 都是一个套路。

### 4.5 Token Count Ablation

K ∈ {4, 8, 16, 32}：
- K=4: information bottleneck, 不够
- K=8: peak
- K=16, 32: 意外下降

Paper 的解释：π0.5 在 pretraining 时 action expert 见过至多 50 tokens，加 32 个 force tokens 超过这个 limit，破坏 coherent action generation。这是 distribution shift 现象。

这个发现很有意思——它揭示了 VLA fine-tuning 的一个 implicit constraint：**不能大幅改变 action expert 的 token budget**。这对未来设计 memory-augmented VLA 是个 useful heuristic。

### 4.6 Inference Efficiency

| Method | Latency (ms) | Δ vs base |
|---|---|---|
| π0.5 (base) | 60.7 | - |
| π-MEM (K=5) | 99.8 | +39.1 |
| π-MEM (K=16) | 190.0 | +129.3 |
| FM-VLA | 64.0 | **+3.3** |

FM-VLA 的 overhead 几乎可以忽略。这是 force memory 相对 visual memory 的另一个重大优势——visual memory 要 encode 多帧 RGB，cost 高得多。FM-VLA 的 force 是 6D low-dim signal，VAE encoder 是 small Perceiver-IO，几乎没 cost。

## 5. Intuition Building

让我把几个关键 insight 总结一下：

### 5.1 为什么 force 比 vision 适合 contact-rich memory？

回到人类认知的类比。考虑 task 2 (Press button N times)。如果你让一个**人**做这个任务但**蒙住眼睛**，他能完成吗？完全能——他靠的是：
- 指尖的 tactile impulse（每次 click 都有一个清晰的 force spike）
- 手指的 proprioception（手在哪、按到没按到）
- 工作记忆里的 counter（按了 1, 2, 3...）

这三者对应到 FM-VLA：
- Force memory $Z_f$ ≈ tactile impulse accumulation
- State memory $z_s$ ≈ proprioception
- VAE latent ≈ compressed counter representation

而如果你**让一个人睁着眼睛但指尖麻木**，他能完成吗？很难——按钮视觉上几乎不动，他没法 count。这就是 visual memory 失败的本质原因。Visual memory 假设 "看到的过去 = 知道的过去"，但在 contact-rich 任务里这个假设 fail。

### 5.2 VAE 作为 self-supervised temporal summarizer

更深一层的 intuition 是关于 representation learning。考虑一下：你有一个 unbounded 100Hz 6D time series，你想把它压成 8 个 token 给 downstream 用。怎么做？

**Naive 方案**：end-to-end learn 一个 encoder with downstream loss（这是 TA-VLA 的思路，但 TA-VLA 只用 short window）。问题是 downstream loss (flow matching velocity) 是非常 task-specific 的 signal，对 encoder 的 supervision 很弱——尤其当 task 涉及 long-horizon counting 时，单个 flow matching step 看不到全局 counting 信息。

**FM-VLA 方案**：先 self-supervised pretrain encoder with reconstruction loss。这强制 latent encode**整个时间序列的 macroscopic 结构**——onset timing, magnitude pattern, contact count。Reconstruction 是一个 "honest" objective，不会 short-cut。

这跟 LLM 的 pretraining→fine-tuning paradigm 完全一致：先用 general self-supervised objective 学 general representation，再 task-specific fine-tune。FM-VLA 把这个范式 apply 到 force signal 上。

### 5.3 Length cue shortcut 与 random pre-padding

这是一个很精彩的观察。如果你不 pre-pad，模型会学到 "sequence length ≈ episode progress"，从而 short-cut。比如按按钮 3 次的 history length 一定 > 按按钮 1 次的，模型只要检测 length 就够了，不需要真正理解 force content。

这听起来好像也行？但问题是：
1. Length cue 在 inference 时也成立，所以模型可能"误打误撞"工作
2. 但模型对 force content 完全没理解，一旦 task 稍微变化就崩
3. 这种 short-cut 阻止了 latent 真正 encode 任务相关 structure

Random pre-padding 类似 image augmentation 中的 random crop——打破 spurious correlation，强制模型学 true signal。

### 5.4 Free-bits KL 与 posterior collapse

VAE 训练的经典痛点是 **posterior collapse**：encoder 输出的 posterior 退化成 prior $\mathcal{N}(0, I)$，意味着 latent 不 carry 任何 task information。这是因为 KL term 太强时会 dominate loss，把所有 latent 推向 prior。

Free-bits trick 解决方法：每个 dimension 至少保留 $\lambda$ nats 的 information budget，未达 budget 时关掉该 dim 的 KL gradient。这相当于**给 latent 一个 "minimum information guarantee"**，强制它至少 encode 一些东西。

这是个非常实用的 trick，我之前在 [Improved VAE](https://arxiv.org/abs/1606.04934) 这篇里看到过，但没想到在 force signal compression 这么有效。

### 5.5 Suffix injection 保持 RoPE positions

这个设计细节反映了作者对 base model 的尊重。π0.5 在大规模 pretraining 时，noisy-action tokens 在固定 RoPE positions 上训练过，如果改变它们的位置会引入 distribution shift。

把 memory tokens 放 suffix 位置（noisy actions 之后）保证 noisy actions 的 RoPE 不变，相当于 memory tokens 是 "extra context" 而不是 "shifted context"。这跟 prefix tuning 的反向版本类似。

## 6. 局限性和我的联想

Paper 提到的 limitations:
- Fixed 8-token bottleneck for very long horizons (hundreds of contact events)
- VAE 只在自己 dataset 上训，没 scale up

我能想到的更多方向：

### 6.1 Hierarchical Memory

8 个 token 对 long horizon 是 hard limit。可以考虑：
- **Two-tier VAE**：short-term VAE (recent 10s) + long-term VAE (episode-level summary)
- 类似人类 working memory vs episodic memory 的分工
- 类似 [22] VPWEM 的工作记忆+情景记忆双系统

### 6.2 Adaptive Token Budget

不同 task 复杂度不同——按按钮 3 次需要少 token，但擦碗 10 圈需要多。可以让 model 学一个 dynamic token count，或者用一个 large budget VAE 然后 mask 掉不重要的 tokens（类似 [24] TempoFit 的思路）。

### 6.3 Cross-embodiment Force Pretraining

VAE 现在只在 AgiBot G1 的 wrench signal 上训。如果能在 Open-X-Embodiment 等大型 robot dataset 上 pretrain（如果数据有 force recording），可以学到更通用的 force representation。这是一个潜在的 scaling law 方向。

### 6.4 Force World Model

TA-VLA 用了一个 auxiliary future force prediction head，FM-VLA 没用。可以想象一个**force world model**——用 VAE latent 预测 future force trajectory，类似 DreamerV3 在 visual/raycast 上的做法。这会让 latent encode 不仅 past 而且未来 predictable structure，可能更 task-relevant。这方面的尝试见 [34] DreamTacVLA。

### 6.5 Tactile Beyond Force

Force/torque 是 6D macroscopic 信号。更细的 tactile（如 digit-level pressure distribution）可以提供更丰富 information。TacVLA [29], VTLA [33] 已经在探索。Future work 可以把 tactile image 也用类似 VAE 压成 memory tokens。

### 6.6 与 Brain 的类比

更深的联想：人类 cerebellum 实际上就在做类似的事情——整合 muscle spindle (proprioception)、Golgi tendon organ (force)、cutaneous receptor (tactile) 的信号，建立 internal model 用于 motor coordination。FM-VLA 的 force VAE + state projection + cross-attention to action expert，某种意义上是 cerebellar model 的简化版。当然 paper 没这么说，但这个 analogy 对 intuition 有帮助。

### 6.7 Pre-padding trick 的推广

Random pre-padding 防止 length shortcut 这个 idea 我觉得很有 generalization potential。任何用 Transformer 处理 variable-length sequence 的地方都可能遇到类似 shortcut——比如 video understanding, audio processing, time series forecasting。一个 future direction 可能是把这个 idea formalize 成一个 general augmentation technique。

### 6.8 Free-bits 在 VLA 其他地方的应用

Free-bits KL 不只对 force VAE 有用。任何用 VAE 做 memory compression 的 VLA 都可能受益。比如 [23] VQ-Memory 用 VQ-VAE 压缩 visual memory，可能也有类似的 collapse 问题，free-bits 思路可能帮助。

### 6.9 6-axis vs full 6D wrench

FM-VLA 用 3-axis force + 3-axis torque。这是工业标准的 6-axis F/T sensor。但其实在某些任务里，单一 axis 就够（比如 button press 主要是 vertical force）。一个 future direction 是学一个 task-adaptive wrench projection，把 6D 投到 task-relevant subspace。

### 6.10 VAE 的 latent space 可解释性

Paper 没有做 latent interpretability analysis。一个有趣的实验是：可视化不同 latent token 编码了什么——比如 token 1 编码 contact count, token 2 编码 magnitude, token 3 编码 timing。这种 analysis 会让 model 更 interpretable。

### 6.11 与 Diffusion Policy 的关系

π0.5 用 flow matching（rectified flow），是 diffusion policy 的变种。FM-VLA 的 memory conditioning 也可以推广到其他 diffusion policy 框架，比如 RDT-1B [14], CogACT [16]。这些 model 的 action expert 都是 transformer-based，suffix injection 应该都能 work。

### 6.12 Memory retrieval vs compression

MemoryVLA [5] 走 retrieval 路线（explicit memory bank + retrieval），FM-VLA 走 compression 路线（VAE 压成 fixed tokens）。两条路线的本质区别：retrieval 适合 sparse, salient events (一张图胜千言)；compression 适合 dense, continuous signals (force signal 每个 timestep 都有信息)。这个分工可能是个 useful design principle。

## 7. 一些 Critical 的观察

我也想提几个我读 paper 时产生的一些 questions：

1. **Buttons 上 72.2% 还有提升空间**：16.7% 的 trial 失败说明 VAE 8-token bottleneck 可能丢了一些 counting 信息。或者可能是 force sensor noise 在某些 trial 干扰了。Paper 没给 failure case analysis。

2. **为什么右 wrist only？**：所有 task 都是 right hand 主导（right hand 按按钮、擦碗、lift cup）。如果用 bimanual force（左+右 wrist），可能捕捉到更丰富的 coordination 信息。这可能是未来 bimanual task 的方向。

3. **Pre-padding max 10s 是否足够？**：对于非常长的 task（比如擦碗 10+ 圈），10s pre-padding 可能不够 mask length cue。Paper 没测非常 long-horizon 的 case。

4. **VAE 训练用所有 task joint**：会不会有 task interference？比如 button impulse 和 wiping periodic force 是 very different patterns。Paper 用 inverse-frequency sampling 缓解，但没显式 measure cross-task transfer。

5. **No comparison to discrete token approaches**：比如把 force signal discretize 成 token（类似 VQ-VAE），然后当 language token 处理。这是一个 alternative 方案，paper 没探索。

## 8. 总结与影响

FM-VLA 是一个我认为会被广泛 cite 的工作，因为它指出了一个 important direction：**memory modality 应该 match task 的 information bottleneck**。Visual memory 不是 universal solution——当任务关键 state 在 force/tactile 上时，必须用对应模态的 memory。

技术 contribution 上：
- **Two-stage paradigm** (self-supervised VAE pretrain + downstream fine-tune) 把 NLP 的范式带到 force signal
- **Suffix injection** 保持 base model behavior
- **Random pre-padding** 防止 length shortcut
- **Complementary short state history** 解决 pre-contact motion 问题

工程 contribution 上：
- 6D force memory 比 K-frame visual memory 计算便宜两个数量级
- 8-token bottleneck 是非常 efficient 的 design point

Conceptual contribution 上：
- 明确区分 **instantaneous force conditioning** vs **force memory**
- 提出 contact-rich manipulation 的 memory 应该 ground in physical interaction，不是视觉

我觉得这篇工作会催生一系列 follow-up：
- **Hierarchical force memory** for very long horizons
- **Multi-modal force + tactile + visual** joint memory
- **Force world models** that predict future contact events
- **Cross-embodiment force pretraining** at scale
- **Interpretable force latents** that explicitly code count/magnitude/timing

这是一篇值得反复读的 paper，尤其对做 manipulation, memory, multimodal learning 的人。

---

**Reference Links**:

- [π0.5 (base model)](https://arxiv.org/abs/2504.16054)
- [π0 (flow matching VLA)](https://arxiv.org/abs/2410.24164)
- [MemoryVLA](https://arxiv.org/abs/2508.19236)
- [MEM](https://arxiv.org/abs/2603.03596)
- [ForceVLA](https://arxiv.org/abs/2505.22159)
- [TA-VLA](https://arxiv.org/abs/2509.07962)
- [Perceiver IO](https://arxiv.org/abs/2107.14795)
- [Original VAE (Kingma & Welling)](https://arxiv.org/abs/1312.6114)
- [Free-bits improved VAE](https://arxiv.org/abs/1606.04934)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [PaliGemma](https://arxiv.org/abs/2407.07726)
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [RT-2](https://arxiv.org/abs/2307.15818)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [RDT-1B](https://arxiv.org/abs/2410.07864)
- [CogACT](https://arxiv.org/abs/2411.19650)
- [GR00T N1](https://arxiv.org/abs/2503.14734)
- [SpatialVLA](https://arxiv.org/abs/2501.15830)
- [GR-3](https://arxiv.org/abs/2507.15493)
- [BLIP-2 / Q-Former](https://arxiv.org/abs/2301.12597)
- [GRU (Cho et al.)](https://arxiv.org/abs/1412.3555)
- [AgiBot G1](https://www.agibot.com/products/G1)
- [FM-VLA Project Page](https://qft-333.github.io/FM-VLA-Page/)
