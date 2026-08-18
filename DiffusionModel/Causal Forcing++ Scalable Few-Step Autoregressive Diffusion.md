---
source_pdf: Causal Forcing++ Scalable Few-Step Autoregressive Diffusion.pdf
paper_sha256: 0a6242efff345d703d20d1be23be075706aff78ca624f9239b222d3e790c0703
processed_at: '2026-08-18T03:00:04-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Causal Forcing++

Andrej, 咱们把那些复杂的数学符号先放一边, 用最直白的 intuition 来拆解这篇 paper. 这篇论文的核心目标非常简单: **让 video generation 像 ChatGPT 一样做到 real-time streaming 和 interactive**.

你要 interactivity, 就必须 **Autoregressive (AR)** 生成 (一帧一帧或者一小段一小段往外吐), 因为你需要根据用户当前的输入决定下一秒生成什么. 但传统的 AR diffusion model 每生成一块需要跑几十步 diffusion step, 延迟太高. 所以我们需要 **Distillation** (把多步模型压缩成 1-2 步的模型).

这篇 paper 就是把现有的 AR diffusion distillation pipeline 往极限推了一步: **frame-wise autoregression (一次只生成一帧) + 1~2 sampling steps**.

---

## 1. 为什么现有方法在这个极限下会崩?

现有的 distillation pipeline 通常分三步: (1) 训练一个多步的 AR teacher; (2) 把它初始化成一个 few-step 的 AR student; (3) 用 asymmetric DMD (Distribution Matching Distillation) 机制做最后的 polish.

这个 Stage 2 (initialization) 是绝对的核心. 如果 Stage 2 给的起点太烂, Stage 3 的 DMD 根本救不回来. 但在这个极端的 1-frame 1-step 设定下, 以前的三种初始化方法全挂了:

1. **CausVid / Self Forcing 的坑 (用 Bidirectional Teacher 初始化)**: 
   Bidirectional model 生成视频时是可以看到未来的 (基于全帧的 attention). 你拿这种依赖未来信息的模型生成的轨迹去教一个只能看历史的 AR student, 等于让蒙眼的人模仿睁眼的人走路. 同样的 noisy frame $x_t^i$, 在不同未来上下文下对应不同 clean frame. Student 学到最后, 只会输出所有可能画面的平均, 变成一团模糊的 conditional expectation $\mathbb{E}[x_0^i | x_t^i, x_{gt}^{<i}, t]$. 在生成 1 帧走 1 步的苛刻条件下, 这种模糊会导致 error 快速累积, 直接 catastrophic breakdown.

2. **LiveAvatar / WorldPlay 的坑 (直接用多步 AR 初始化)**: 
   它们跳过了 Stage 2, 直接拿多步 AR model 当 few-step student 丢给 DMD. 结构上虽然对齐了, 但它没经过 few-step 的训练, 本身单步生成误差极大. 当你减小 chunk size (增加 AR 调用次数) 同时减少 sampling steps (增加每次调用的误差), 两个负面效应在 self-rollout 时疯狂叠加. Exposure bias 导致画面越往后越烂.

3. **Causal Forcing 的坑 (用多步 AR Teacher 跑 ODE 初始化)**: 
   Causal Forcing 修正了上面的问题: 它先用 AR teacher 模型跑一遍, 生成完整的 PF-ODE 轨迹 (比如 48 步), 然后让 student 去拟合这个轨迹. 这在理论上完美, 但实操成本极其昂贵. 你得为每一个 training video 在 offline 生成并存储一整条轨迹. 只要 teacher 换了, 或者数据分布变了, 全部得重跑. 80K 视频要烧 11,600 A800 GPU 小时, 占用 1900 GiB 存储. 根本没法 scale.

所以我们急需一种新的 initialization, 必须 **同时满足** AR, few-step, 和 scalable.

---

## 2. Causal Forcing++ 的核心 Insight

作者发现了一个绝妙的等价性: **Causal ODE distillation 和 Consistency Distillation (CD) 其实学的是同一个东西!**

它们学的都是 AR teacher 的 **Consistency Function** $f_\phi$. 这个函数的物理意义就是: 给定一个 noisy frame $x_t^i$ 和过去的历史 $x_{gt}^{<i}$, 预测出最终去噪后的 clean frame $x_0^i$.

既然目标一样, 那 CD 就可以直接拿来替换 ODE distillation. 我们看看公式的直觉:

$$
\theta^* = \arg\min_\theta \mathbb{E}_{x_{gt}, \epsilon, t, i} \left[ w(t) d\left( G_\theta(x_t^i, x_{gt}^{<i}, t), G_{\theta^-}(\hat{x}_{t-\Delta t}^i, x_{gt}^{<i}, t-\Delta t) \right) \right]
$$

变量拆解:
- $G_\theta$: Student 模型.
- $x_t^i$: 第 $i$ 帧在 timestep $t$ 的加噪状态.
- $x_{gt}^{<i}$: 过去真实的历史帧.
- $t$: Diffusion timestep, $t=1$ 纯噪声, $t=0$ 干净画面.
- $\hat{x}_{t-\Delta t}^i$: 用 AR teacher 走一步 ODE, 从 $t$ 到 $t-\Delta t$ 得到的稍微去噪一点的画面.
- $\theta^-$: $\theta$ 的 Exponential Moving Average (EMA), 且 stop-gradient (只作为目标, 不回传梯度).
- $w(t)$: 针对不同 $t$ 的权重.
- $d(\cdot, \cdot)$: 距离函数.

**这个公式的直觉非常自洽**: 
ODE distillation 是让 student 从 $t$ 直接一步蹦到 $0$ (一个 huge gap, 极难优化, 且必须离线生成 $t=0$ 的目标). 
而 Causal CD 是在真实视频上 online 加噪到 $t$, 然后让 AR teacher 走一小步到 $t-\Delta t$. Student 只要保证自己走一步的结果, 和 teacher 走一步的结果一致就行了. 

这就把一个 long-range 的 regression 问题, 变成了 local consistency 的问题. 优化难度骤降, 而且由于是 local step, 根本不需要 offline 存储轨迹, 直接 online 算就行! 结果就是: 训练时间从 11,600 降到 2,900 GPU 小时, 存储从 1900 GiB 降到 0.

---

## 3. 为什么不用 Score Distillation (DMD) 做初始化? (最精妙的 Insight)

既然 CD 可以, 那 DMD (Distribution Matching Distillation) 行不行? 在 bidirectional distillation 中, DMD 通常比 CD 效果更好. 作者试了一下 Causal DMD, 发现了一个反直觉的现象: **Causal DMD 前几帧极其清晰, 但越往后越烂, exposure bias 严重放大.**

原因在于 KL 散度的方向. 看这个 DMD 的目标公式:

$$
\nabla_\theta \mathbb{E}_t [D_{KL}(p_{\theta,t}(\tilde{x}_t^i | x_{gt}^{<i}) \| p_{data,t}(\tilde{x}_t^i | x_{gt}^{<i}))] = -\mathbb{E}_{x_{gt}^{<i}, \tilde{x}^i, t, \tilde{x}_t^i} \left[ (s_{real} - s_{fake}) \frac{\partial \tilde{x}^i}{\partial \theta} \right]
$$

变量拆解:
- $p_{\theta,t}$: Student 生成的帧加噪到 $t$ 后的分布.
- $p_{data,t}$: 真实数据加噪到 $t$ 后的分布.
- $s_{real}$: 评估真实数据 score 的 frozen 模型.
- $s_{fake}$: 评估 student score 的 online 训练模型.

DMD 优化的是 **Reverse KL $D_{KL}(p_\theta \| p_{data})$**, 这会引发 **Mode-seeking** 行为. 模型会把所有概率 mass 塞到最 sharp 的那个 mode 上, 前几帧看起来极度逼真.
CD 隐式优化的是 **Forward KL**, 这会引发 **Mode-covering** 行为. 模型会把概率 mass 摊开, 覆盖所有可能, 牺牲一点 sharpness 换取 robustness.

**在 Autoregressive 序列生成中, 这是个致命差异.**
AR 生成会有 history drift (自己生成的前缀偏离 ground-truth). 在 mode-seeking 的 DMD 下, 分布是个极窄的尖峰, 一旦 history 发生偏移, 这个尖峰直接被推移到 poor-quality region, 后面全崩. 
而在 mode-covering 的 CD 下, 分布是个宽包络, 即使 history 偏移了, 依然有很大一部分 mass 落在 good-quality region, 能够自我纠正.

这就像是在走钢丝 vs 走平地. DMD 让你在完美天气下走钢丝 (前几帧好), CD 让你刮风下雨走平地 (一直稳). 在 AR 这种充满 compounding error 的 sequential 任务里, **Robustness 碾压 Sharpness**. 这个 insight 对未来的 AR LLM (比如 video token 预测的 RLHF 阶段) 也有极强的指导意义.

---

## 4. 实验数据直观读

从 Table 1 看, Causal Forcing++ 的 2-step frame-wise 设定, 完全超越了之前 4-step chunk-wise 的 SOTA (Causal Forcing):

| Model | Latency ↓ | Total ↑ | Quality ↑ | Vision ↑ |
| :--- | :--- | :--- | :--- | :--- |
| Causal Forcing (4-step) | 0.60s | 84.04 | 84.59 | 6.326 |
| CF++ (2-step) | 0.27s | 84.14 | 84.89 | 6.661 |

在首帧延迟砍半 (0.60s -> 0.27s) 的情况下, 各项指标全面超越. 这意味着我们用更少的步数、更细粒度的自回归, 依然取得了更好的质量.

Table 2 的 Ablation 最有意思, 直接用数据锤死了前面的分析:
- 在 1-step 设定下, Self Forcing init 和 AR diffusion init 的 Dynamic Degree 全是 0 (画面完全静止, 崩了), 只有 Causal CD 和 Causal ODE 能撑住.
- Causal CD 在所有步数设定下, 质量均 ≥ Causal ODE, 且完全不需要 1900 GiB 的 storage 和 11600 GPU 小时.

---

## 5. 总结与应用

作者把这套东西套在了 Genie3-style 的 action-conditioned world model 上. 用 camera-pose 作为 action signal, 通过 PRoPE 注入, 蒸馏出了一个可以实时交互的 AR world model. 这基本上指明了下一代 interactive game engine 的做法.

**一句话人话总结**: 
Causal Forcing++ 发现把 ODE distillation 换成 Consistency Distillation 来做初始化, 数学目标完全等价, 但免去了离线存轨迹的巨额开销. 同时它揭示了在 AR rollout 场景下, Mode-covering (CD) 比 Mode-seeking (DMD) 更能扛住 exposure bias. 最终用 2 步生成 1 帧的极致设定, 达到了降本增效破纪录的结果.

### Web Links for Reference:
- **Paper (arXiv)**: [Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation](https://arxiv.org/abs/2602.02214)
- **Project Page & Code (GitHub)**: [thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing)
- **World Model Implementation**: [shengshu-ai/minWM](https://github.com/shengshu-ai/minWM)
- **Prequel Paper**: [Causal Forcing (arXiv:2602.02214)](https://arxiv.org/abs/2602.02214)
- **Base Video Model**: [Wan2.1 (arXiv:2503.20314)](https://arxiv.org/abs/2503.20314)
- **Original CD Theory**: [Consistency Models (arXiv:2303.01969)](https://arxiv.org/abs/2303.01969)
- **DMD Theory**: [One-Step Diffusion with DMD (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-Step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html)

---

# Causal Forcing++ 深度解读

Andrej, 这篇paper非常对你的胃口 - 它本质上是在做 **AR diffusion distillation 的 initialization 阶段的 scaling law fix**。让我从 intuition 到技术细节完整拆解。

---

## 1. 核心问题与动机

**Real-time interactive video generation** 需要 three properties: low-latency, streaming rollout, user-controllable。现有 AR diffusion distillation 方法 (CausVid, Self Forcing, Causal Forcing) 都在 **chunk-wise (3 latent frames) 4-step** 设定下工作, 但 chunk granularity 太粗, sampling latency 仍然 non-negligible。

作者 push 到更 aggressive 的 regime: **frame-wise autoregression with 1-2 sampling steps**。这里的关键 bottleneck 是 asymmetric DMD 之前的 **few-step AR student initialization**。DMD 本身对 initialization 极其敏感 - 它是 refiner 而非 complete trainer。

### 三个 existing strategies 的失败模式

让我用 Karpathy 风格的 mental model 解释:

**(i) Bidirectional teacher ODE init (CausVid, Self Forcing)**: 
- Bidirectional teacher 的 PF-ODE trajectory 依赖 future frames
- AR student 拿不到 future context
- 违反 **frame-level injectivity**: 同一个 noisy frame $x_t^i$ 在不同 future contexts 下对应不同的 clean frames $x_0^i$
- 最优解 collapse 到 conditional expectation $\mathbb{E}[x_0^i | x_t^i, x_{gt}^{<i}, t]$, 即 blurred target
- 在 frame-wise 1-step 下 catastrophic breakdown (Fig. 2 col 1)

**(ii) Multi-step AR diffusion init (LiveAvatar, WorldPlay)**:
- 看似合理 (都是 AR), 但缺少 few-step capability
- 减小 chunk size → 增加 AR calls 数量; 减少 sampling steps → 增加每个 call 的 approximation error
- 两个 effect 在 self-rollout 时 compound, exposure bias 严重 amplify
- 在 1-step setting 下 Dynamic Degree = 0, Instruction Following = -14 (Table 2)

**(iii) Causal ODE init (Causal Forcing)**:
- 理论 correct (AR teacher → AR student, target aligned)
- 但需要 offline precompute full PF-ODE trajectories (48 steps/sample) 并 storage
- 80K videos → 11,600 A800-GPU hours + 1,900 GiB storage
- 任何 teacher / data / chunk-size 变化都要 regenerate
- **Scaling bottleneck**

所以需要的 initialization 必须 simultaneously **AR + few-step + scalable**。

---

## 2. Causal Forcing++ 的核心 insight

**Key observation**: causal ODE distillation 和 causal consistency distillation (CD) 学习的 **是同一个 object** - AR teacher 的 flow map (consistency function)。区别只在 supervision 获取方式。

### 数学等价性

**Causal ODE distillation** (Eq. 1):
$$\theta^* = \arg\min_\theta \mathbb{E}_{x_{gt}^{<i}, t, i, x_t^i} \left[ \| G_\theta(x_t^i, x_{gt}^{<i}, t) - x_0^i \|^2 \right]$$

变量解释:
- $\theta$: student model parameters
- $G_\theta$: student generator (映射 noisy state → clean prediction)
- $x_t^i$: 第 $i$ 帧在 diffusion timestep $t$ 的 noisy state
- $x_{gt}^{<i}$: ground-truth prefix (前 $i$ 帧的 clean latent)
- $t \in [0,1]$: diffusion timestep, $t=1$ 纯噪声, $t=0$ clean
- $i$: frame index
- $x_0^i$: AR teacher PF-ODE trajectory 上 timestep 0 的 sample (clean target)

最小化器是 AR-conditional flow map (Eq. 2):
$$f_\phi: (x_t^i, x_{gt}^{<i}, t) \mapsto x_0^i$$

- $f_\phi$: AR teacher 的 consistency function
- $\phi$: AR teacher parameters
- 任意 $t$ 下的 noisy state 都映射到 teacher PF-ODE 在 $t=0$ 的 endpoint

**Causal CD** (Eq. 3):
$$\theta^* = \arg\min_\theta \mathbb{E}_{x_{gt}, \epsilon, t, i} \left[ w(t) d\left( G_\theta(x_t^i, x_{gt}^{<i}, t), G_{\theta^-}(\hat{x}_{t-\Delta t}^i, x_{gt}^{<i}, t-\Delta t) \right) \right]$$

变量解释:
- $x_t^i$: 从 ground-truth $x_{gt}^i$ 通过 forward diffusion 得到 ($x_t = (1-t)x_0 + t\epsilon$)
- $\hat{x}_{t-\Delta t}^i$: 从 $x_t^i$ 通过 **single ODE step** 用 AR teacher (conditioned on $x_{gt}^{<i}$) 得到
- $\theta^-$: $\theta$ 的 EMA with stop-gradient (target network)
- $w(t)$: timestep-dependent weight
- $d(\cdot, \cdot)$: distance under pre-defined norm (paper 中用 square norm)
- $\Delta t$: adjacent timesteps 的 difference (48 discretized timesteps, 所以 $\Delta t \approx 1/48$)

**Flow-matching parameterization** (Eq. 4):
$$G_\theta(x_t^i, x_{gt}^{<i}, t) = x_t^i - t \cdot v_\theta(x_t^i, x_{gt}^{<i}, t)$$

- $v_\theta$: neural network 预测 velocity
- 来自 flow matching: $\alpha(t) = 1-t$, $\sigma(t) = t$, $v_t := dx_t/dt = \epsilon - x_0$
- 所以 $x_0 = x_t - t \cdot v_t$ (一步去噪的 Tweedie's formula 形式)

**Error bound** (Eq. 5):
$$\sup \| f_\phi(x_t^i, x_{gt}^{<i}, t) - G_{\theta^*}(x_t^i, x_{gt}^{<i}, t) \|_2 = \mathcal{O}((\Delta t)^p)$$

- $\Delta t$: 相邻 timesteps 的最大 difference
- $p$: ODE solver 的 order of accuracy (Euler solver $p=1$)
- 这个 bound 说明 causal CD 学到的 optimal $G_{\theta^*}$ 与 target $f_\phi$ 的距离只由 numerical error 决定, 当 $\Delta t$ 足够小时 negligible

### Intuition: 为什么这个等价成立

想象 AR teacher 是一个 ODE solver, 给定起点 $x_t^i$ 解到 $t=0$ 得到 $x_0^i$。Consistency function $f_\phi$ 就是 "any point on the trajectory → endpoint" 的映射。

- **Causal ODE distillation**: 直接 regresses 这个 endpoint (large jump $t \to 0$)
- **Causal CD**: 在 trajectory 上 enforce self-consistency (local step $t \to t-\Delta t$)

两者在 optimal 下都 recover 同一个 $f_\phi$。但 CD 用 local supervision, 不需要 precompute 整条 trajectory。

---

## 3. 两个 practical advantages

### Efficiency

| Method | Stage 2 Time | Storage |
|---|---|---|
| Causal ODE (80K videos) | 11,600 A800-hrs | 1,900 GiB |
| Causal CD (80K videos) | 2,900 A800-hrs | 0 GiB |
| **Speedup** | **4×** | **∞** |

Causal ODE 需要 teacher 生成 full multi-step PF-ODE trajectory per sample (48 steps), paired trajectories offline storage, 任何 config change 都要 regenerate。Causal CD 只需 per-iteration 一次 online teacher ODE step, on real videos。

### Quality (更微妙的优势)

Causal ODE: regress $x_t^i \to x_0^i$, 整个 $t \to 0$ 区间, **large gap one-shot**
Causal CD: pair $x_t^i \to \hat{x}_{t-\Delta t}^i$, **local consistency**, gap 只有 $\Delta t$

这点和 InstaFlow [25] 的 rectified flow 观察一致 - **local pairing 比 long-range regression 更易优化**。Per-step optimization gap 从 $\mathcal{O}(t)$ 降到 $\mathcal{O}(\Delta t)$。

---

## 4. Causal DMD 的失败分析 (Section 3.4.1, 最有意思的部分)

作者还尝试了 causal score distillation (DMD), 发现虽然 early frames sharper, 但 AR rollout 时 rapid drift。这个分析很 Karpathy - 解释了 mode-seeking vs mode-covering 在 sequential generation 下的不同行为。

### Causal DMD 公式 (Eq. 7)

$$\nabla_\theta \mathbb{E}_t [D_{KL}(p_{\theta,t}(\tilde{x}_t^i | x_{gt}^{<i}) \| p_{data,t}(\tilde{x}_t^i | x_{gt}^{<i}))] = -\mathbb{E}_{x_{gt}^{<i}, \tilde{x}^i, t, \tilde{x}_t^i} \left[ (s_{real}(\tilde{x}_t^i, x_{gt}^{<i}, t) - s_{fake}(\tilde{x}_t^i, x_{gt}^{<i}, t)) \frac{\partial \tilde{x}^i}{\partial \theta} \right]$$

变量:
- $\tilde{x}^i$: student 生成的第 $i$ 帧
- $\tilde{x}_t^i$: $\tilde{x}^i$ 通过 forward diffusion perturbed 到 timestep $t$
- $s_{real}$: 估计 data score 的 frozen model
- $s_{fake}$: 估计 student score 的 online-trained model
- $p_{\theta,t}$: student 在 timestep $t$ 的 marginal distribution

### KL 方向决定 mode behavior

- **DMD**: reverse KL $D_{KL}(p_\theta \| p_{data})$ → **mode-seeking** (concentrate mass on high-density modes)
- **CD**: forward KL $D_{KL}(p_{data} \| p_\theta)$ → **mode-covering** (spread mass to cover all modes)

### AR rollout 下的 sensitivity 分析 (Fig. 5b 的 intuition)

想象 conditional distribution $p(\tilde{x}^i | x_{gt}^{<i})$:
- Mode-seeking DMD: sharp peak, probability mass concentrated
- Mode-covering CD: broad distribution, mass dispersed

**History drift** (self-generated prefix 偏离 ground-truth) 会 shift conditional distribution 朝 poor-quality region。

- DMD: peak 集中, shift 一旦发生, 大部分 mass 跟着 mode 移到 poor-quality → **rapid quality collapse**
- CD: mass 分散, 即使 shift, 仍有 substantial mass 留在 good-quality region → **gradual decay**

这就是 **exposure bias 在 mode-seeking 分布下被 amplify** 的机制。Early frames DMD sharper (concentration 减少了 low-quality 概率), 但越往后累积 error 越多, sharp distribution 无法 "hedge"。

这个 insight 对所有 sequential generation 都有启发 - **distribution sharpness vs robustness to compounding error** 的 trade-off。

---

## 5. 完整 pipeline

### Three stages (继承自 Causal Forcing)

1. **Stage 1**: Teacher forcing AR diffusion training (20K steps)
   - Bidirectional Wan2.1-1.3B → AR diffusion model (causal mask)
   
2. **Stage 2**: Few-step initialization (5K steps)
   - **Causal Forcing++ 的核心替换**: causal CD 替代 causal ODE
   - Online teacher ODE step on real videos
   - 48 discretized timesteps, Euler solver
   
3. **Stage 3**: Asymmetric DMD with self-rollout (1K steps)
   - Student: AR few-step (causal CD initialized)
   - Teacher & critic: bidirectional (Wan2.1-14B as real score model)
   - Self-rollout training (Student Forcing 的创新)
   - 4-step: $t \in \{1, 0.9375, 0.8333, 0.625\}$
   - 2-step: $t \in \{1, 0.8333\}$
   - 1-step: $t = 1$
   - ASD trick [50]: first latent frame 保持 4-step, 后续 20 latent frames 用 2-step 或 1-step

### World model 扩展 (Section 3.3)

Genie3-style camera-pose conditioning:
1. WorldPlay 构建 camera-pose-annotated dataset
2. Wan2.1-1.3B fine-tune 成 bidirectional camera-pose-conditioned diffusion model (用 PRoPE [61] inject pose)
3. Causal Forcing++ distill 成 interactive AR world model

---

## 6. 实验数据深度解析

### Table 1: Main results

| Model | Throughput ↑ | Latency ↓ | Total ↑ | Quality ↑ | Semantic ↑ | Dynamic ↑ | Vision ↑ | Instruct ↑ |
|---|---|---|---|---|---|---|---|---|
| CausVid | 10.4 | 0.60 | 83.98 | - | 70.72 | 62 | 5.741 | 12 |
| Self Forcing | 10.4 | 0.60 | 83.74 | 84.48 | - | 80.77 | 5.820 | 48 |
| Causal Forcing | 10.4 | 0.60 | 84.04 | 84.59 | - | 81.84 | 6.326 | 56 |
| **CF++ (1-step)** | 20.7 | 0.27 | 83.35 | 84.50 | - | 78.75 | 5.412 | 38 |
| **CF++ (2-step)** | 14.1 | 0.27 | **84.14** | **84.89** | - | 81.13 | **6.661** | 51 |
| **CF++ (4-step)** | 8.69 | 0.27 | 84.10 | 84.94 | - | 80.75 | 6.798 | 47 |

关键观察:
- **CF++ (2-step) 在 frame-wise 设定下超越 Causal Forcing (4-step chunk-wise)**
- Total +0.1, Quality +0.3, VisionReward +0.335
- Latency 从 0.60s → 0.27s (**50% reduction**)
- CF++ (4-step) 在 Dynamic Degree 上达到 71, 超越所有 SOTA
- 1-step 设定下 Throughput 20.7 FPS, 接近 real-time

### Table 2: Ablation (最 informative)

**1-step asymmetric DMD**:
| Init | Total | Quality | Dynamic | Vision | Time | Storage |
|---|---|---|---|---|---|---|
| Self Forcing ODE | 78.87 | 79.85 | 0 | 1.992 | 5000 | 1500 |
| AR diffusion | 80.54 | 80.97 | 0 | 1.101 | 0 | 0 |
| Causal ODE | 83.06 | - | 46 | 5.464 | 11600 | 1900 |
| Causal DMD | 82.34 | 83.50 | 62 | 4.868 | 2900 | 0 |
| **Causal CD** | **83.35** | **84.50** | **66** | 5.412 | 2900 | 0 |

**2-step**:
| Init | Total | Quality | Vision | Time | Storage |
|---|---|---|---|---|---|
| Causal ODE | 83.77 | 84.42 | 6.224 | 11600 | 1900 |
| **Causal CD** | **84.14** | **84.89** | **6.661** | 2900 | 0 |

**4-step**:
| Init | Total | Quality | Vision | Time | Storage |
|---|---|---|---|---|---|
| Causal ODE | 83.78 | 84.90 | 6.435 | 11600 | 1900 |
| **Causal CD** | **84.10** | **84.94** | **6.798** | 2900 | 0 |

Pattern 非常 clear:
1. Causal CD 在所有 step settings 下 Total/Quality/VisionReward 都 ≥ Causal ODE
2. Time cost 4× reduction, Storage 0
3. Causal DMD 虽然比 AR diffusion init 好, 但 VisionReward 比 CD 低约 0.5
4. Self Forcing ODE 在 frame-wise 下完全 collapse (Dynamic Degree = 0, VisionReward < 2)
5. AR diffusion init 在 1-step 下 Instruct = -14 (近 catastrophic)

---

## 7. 与相关工作的 connection

### Consistency Distillation 谱系
- Song et al. [21] 原始 consistency models
- Phased CM [22], ICT [23] 改进
- Zheng et al. [24] score-regularized continuous-time CM (这篇是作者组的, 也是 DMD vs CD 比较的 source)
- **Causal Forcing++ 是 CD 在 AR conditional setting 的 lift**

### AR Diffusion Distillation 谱系
- CausVid [18]: 第一个 bidirectional → AR distillation, ODE init 有 flaw
- Self Forcing [19]: 修 DMD stage 的 train-test gap (self-rollout)
- Causal Forcing [20]: 修 ODE init 的 architectural mismatch (AR teacher)
- **Causal Forcing++**: 修 Causal Forcing 的 scalability bottleneck (causal CD)
- APT2 [63]: GAN-based, 也用 teacher-forcing CD 但无理论 motivation, 不开源

### 和 InstaFlow [25] 的 connection
InstaFlow 观察 rectified flow 的 local pairing 比 long-range regression 易优化。Causal Forcing++ 在 AR setting 复现了这个观察。

### 和 Genie3 [9] 的 connection
World model 部分直接 follow Genie3 的 camera-pose conditioning paradigm。minWM 是他们的开源实现。

---

## 8. 我的 intuition & 思考

### 为什么 causal CD 在 AR 下比 causal DMD 好 (反直觉)

在 bidirectional distillation 中, DMD 通常优于 CD [24, 58, 62]。但 AR setting 下相反。Key 在于:

**Sequential generation amplifies distribution sharpness sensitivity**

Bidirectional: 一次性生成, distribution sharpness = quality
AR: 累积误差, distribution sharpness = sensitivity to history drift

这类似于 **GAN 在 long-horizon RL 中的不稳定性** - sharp mode 在 compounding error 下 fragile。Mode-covering 提供 "robustness hedge"。

这个 insight 对 **autoregressive LLM 的 RLHF** 也有启发 - mode-seeking KL (reverse KL) 在 long generation 下可能比 mode-covering (forward KL) 更不稳定。

### Causal Forcing++ 作为 "infrastructure" 工作

这篇 paper 的贡献主要是 **scaling fix**:
- 把 Causal Forcing 的 Stage 2 从 O(storage × steps) 降到 O(1)
- 把 offline pipeline 变成 online pipeline
- 解锁了 1-step / 2-step / frame-wise 等 aggressive regime 的实验探索

之前 Causal ODE 的 11,600 A800-hrs 让 ablation 极其昂贵。Causal CD 的 2,900 A800-hrs 让 systematic exploration 变得 feasible。这就是为什么作者能做 Table 2 这样完整的 5×3 = 15 个 cell 的 ablation。

### 关于 first latent frame 4-step trick

ASD [50] 的 trick: first latent frame 4-step, 后续 20 latent frames 1-2 step。这暗示 **first frame 的 quality 对整个 rollout 重要** (类似 LLM 的 prompt sensitivity)。这是个值得深挖的方向 - 是否有 adaptive step allocation 的最优策略?

### Limitations 我读出的

1. World model 部分只在 chunk-wise 4-step 验证, frame-wise 2-step 留作 future work
2. 只在 Wan2.1-1.3B 上验证, 14B 只作为 DMD real score model
3. 80K dataset 规模有限, 更大规模 scaling 未验证
4. 1-step 设定下 VisionReward (5.412) 仍低于 4-step (6.798), real-time 和 quality 的 trade-off 未完全 close

---

## References

- Paper arXiv: [Causal Forcing++](https://arxiv.org/abs/2602.02214) (注: 实际 arxiv ID 以官方为准)
- Project page: [https://github.com/thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) & [https://github.com/shengshu-ai/minWM](https://github.com/shengshu-ai/minWM)
- Causal Forcing (前作): [arXiv:2602.02214](https://arxiv.org/abs/2602.02214)
- Self Forcing: [arXiv:2506.08009](https://arxiv.org/abs/2506.08009)
- CausVid: [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html)
- Consistency Models: [arXiv:2303.01969](https://arxiv.org/abs/2303.01969)
- DMD: [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-Step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html)
- InstaFlow: [arXiv:2307.16027](https://arxiv.org/abs/2307.16027)
- Wan2.1: [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
- Genie 3: [DeepMind blog](https://deepmind.google/models/genie/)
- Zheng et al. score-regularized CM: [arXiv:2510.08431](https://arxiv.org/abs/2510.08431)
- VBench: [arXiv:2311.17982](https://arxiv.org/abs/2311.17982)
- VisionReward: [arXiv:2412.21029](https://arxiv.org/abs/2412.21029)
- ASD: [arXiv:2511.01419](https://arxiv.org/abs/2511.01419)
- WorldPlay: [arXiv:2512.14614](https://arxiv.org/abs/2512.14614)
- PRoPE: [NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/...)

---

## 一句话总结

Causal Forcing++ 把 consistency distillation 从 bidirectional lift 到 AR conditional setting, 利用 "CD 和 ODE distillation 共享 consistency function target" 的等价性, 同时解决 Causal Forcing 的 scalability bottleneck 和 DMD 在 sequential generation 下的 exposure bias 问题, 让 frame-wise 2-step real-time interactive video generation 第一次达到 SOTA quality。
