---
source_pdf: Neural Computers.pdf
paper_sha256: 3c8fe9bf7030ab183d9c5aa79cd04030ad419c29d160f5fb8b1950889e155c27
processed_at: '2026-08-05T22:18:51-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Neural Computers 人话版

Andrej，我用更口语的方式讲讲这篇 paper 到底在说什么，以及为什么我觉得它有意思。

## 一句话总结

把整个计算机塞进一个 neural network 的 latent state 里，让它自己学会"运行"。

## The Big Picture

先想想传统计算机怎么工作。你有 CPU 算东西，RAM 存东西，显示器给你看东西，这三块是 **physically separate** 的，靠 OS 把它们粘起来。这套 von Neumann architecture 我们用了 80 年，work 得很好。

那 paper 的 radical idea 是什么呢？**把这些功能全部 fold 进一个 neural network 的 hidden state**。不是在 OS 上面加一层 neural agent，是用 network 本身当 computer。

你给它一个初始屏幕 + 用户输入，它 rollout 下一帧屏幕。就这么简单。但关键是那个 hidden state $h_t$ —— 它 **同时是** memory、是 program counter、是 register file、是 I/O buffer，全部混在一起。

## 公式人话版

$$h_t = F_\theta(h_{t-1}, x_t, u_t), \quad x_{t+1} \sim G_\theta(h_t)$$

翻译成人话：
- **当前内存** = 某个函数(**上一刻内存**, **当前屏幕画面**, **当前用户操作**)
- **下一帧画面** = 从**当前内存**采样出来

就这么一个 loop。$F_\theta$ 是 state update（相当于 CPU + OS kernel 的活），$G_\theta$ 是 decoder（相当于 GPU rendering），$h_t$ 是 everything else。

这跟 world model (Ha & Schmidhuber, https://arxiv.org/abs/1803.10122) 长得很像，区别在于：world model 是预测环境，NC 的 latent state **本身就是 runtime**。不是 "model the computer"，是 "be the computer"。

## 他们实际做了什么

很务实，没 claim 太多。拿了 Wan2.1（Meta 的 video generation model, https://arxiv.org/abs/2503.20314）当 base，做了两个 prototype：

### Prototype 1: NC_CLIGen (终端)

让它学会生成 terminal 画面。数据来源：
- **CLIGen (General)**: 823k 个 asciinema 录屏，约 1100 小时真实人类用 terminal 的 trace
- **CLIGen (Clean)**: 用 vhs 工具脚本化跑出来的 ~128k 个干净 trace

训练就是 text + first frame → predict future frames。架构是 Wan2.1 + CLIP + T5 + DiT，标准 video gen 套路。

### Prototype 2: NC_GUIWorld (桌面)

让它学会生成 desktop 画面 + 响应 mouse/keyboard。数据：
- Random Slow: 1000 小时随机乱点
- Random Fast: 400 小时快速乱点
- **Claude CUA supervised: 110 小时**（这个是 key）

架构上他们搞了四种 action injection 方式：external、contextual、residual、internal。简单说就是 action 信号从 "只在 input 加一下" 到 "塞进 transformer block 内部" 四种深度。

## 实验结果里最有意思的几个

### 1. VAE 对 terminal 文字够用

13px 字体：PSNR 40.77 dB，SSIM 0.989。挺好的。但 6px 极密文字会糊。这说明 generic image VAE 对 structured screenshot 不是完全不行，只要字体别太变态。

### 2. 训练 25k 步就饱和

这个特别有意思。PSNR/SSIM 在 25k 步基本 plateau，继续练到 460k 步甚至 **slightly degrade**。

我的解读：大部分 learnable pattern 早期就学完了，剩下的 error 都是 hard cases（rendering glitches、rapid screen changes），单纯堆训练步数解决不了，需要更好的 supervision signal。

这跟 "double descent" (Belkin et al., https://doi.org/10.1073/pnas.1821543116) 有点像但又不一样 —— double descent 是 over-parameterization 导致的，这里是 objective 本身不够 informative。

### 3. Caption 越详细越好

| Prompt 风格 | PSNR | 字数 |
|---|---|---|
| Semantic (高层概述) | 21.90 | 55 |
| Regular (关键命令) | 23.63 | 52 |
| **Detailed (逐字转录)** | **26.89** | 76 |

差了 5 dB，在 PSNR 里这是 huge difference。

为什么？因为 terminal 画面主要就是文字位置，detailed caption 直接告诉你 "哪个 token 在哪里"，相当于 explicit scaffolding。Natural image 靠 style，terminal 靠 layout。

### 4. 字符级准确率 54%

用 Tesseract OCR 跑生成的 terminal 帧，跟 ground truth 比字符级 edit distance。60k 步训练后达到 0.54 char accuracy，0.31 exact line accuracy。

从 0.03 涨到 0.54，这已经是 "usable" 的边缘了。但 0.54 还远不够 production-grade。从 "plausible" 到 "correct" 还有很大 gap。

### 5. 算术测试：最 sobering 的结果

让他们做 terminal 里的简单算术（python REPL 里跑 `10+15` 这种）：

| Model | Accuracy |
|---|---|
| Wan2.1 | 0% |
| NC_CLIGen | 4% |
| Veo 3.1 | 2% |
| **Sora 2** | **71%** |

Sora 2 的 71% 是 outlier。可能的解释（Table 6）：
- 更大的 base model
- 额外的 RL training
- System-level recaptioning（内部 LLM 先算好答案再 condition generation）

这个结果呼应了 Gary Marcus (https://arxiv.org/abs/1801.00631) 和 Lake & Baroni (https://arxiv.org/abs/1711.00350) 对 deep learning symbolic reasoning 的批评。video model 长于 rendering，短于 reasoning。

### 6. Reprompting: 4% → 83%

这个实验特别 clever。他们发现如果你在 prompt 里给更强的 hint（甚至直接把答案塞进去），准确率从 4% 飙到 83%。

**这说明什么？** 大部分 apparent "reasoning" gain 来自 conditioning，不是 native computation。model 学会了 "faithful rendering of conditioned symbolic content"，没学会 "actually do arithmetic"。

这跟 chain-of-thought (Wei et al., https://arxiv.org/abs/2201.11903) 的争议很像 —— CoT 到底是 "reasoning" 还是 "structured lookup"？这里 reprompting 更像是后者。

Paper 很诚实地承认了这一点，没 overclaim。

### 7. 数据质量 >> 数据量

| Data | Hours | FVD | SSIM |
|---|---|---|---|
| Random Fast | 400 | 48.17 | 0.695 |
| Random Slow | 1000 | 20.37 | 0.830 |
| **Claude CUA** | **110** | **14.72** | **0.885** |

110 小时 goal-directed 数据 beat 1400 小时 random exploration。这是 **13x 数据效率**。

为什么？Goal-directed trajectories 提供清晰的 action semantics 和可预测的 state transitions。Random exploration 虽然覆盖广，但 noisy、bursty、semantic 信号弱。

这跟 LIMA (https://arxiv.org/abs/2305.11206) "1000 examples suffice" 的哲学完全一致。Quality > Quantity for alignment。

### 8. 光标控制：visual supervision 是关键

| Loss | Cursor accuracy |
|---|---|
| 只用 (x, y) 坐标 | 8.7% |
| (x, y) + Fourier features | 13.5% |
| **(x, y) + SVG mask/reference** | **98.7%** |

从 8.7% 到 98.7%，差距巨大。**Intuition**: 光标必须作为 visual object 学习，不能只靠 abstract coordinates。他们 render 一个 SVG 箭头在 $(x_t, y_t)$ 位置，把箭头作为 explicit conditioning signal。

这让我想到 embodied AI 里 visual grounding 的重要性 —— agent 需要 grounded 在 pixel space，symbolic state 不够。

### 9. Action injection 越深越好

| Mode | SSIM_+15 | FVD_+15 |
|---|---|---|
| External (input level) | 0.746 | 33.4 |
| Contextual (token level) | 0.813 | 24.8 |
| Residual (block external) | 0.857 | 18.8 |
| **Internal (block internal)** | **0.863** | **14.5** |

很 monotonic 的 trend：action 信号越深入 transformer computation，integration 越好。Internal mode（在 transformer block 里加一个 action cross-attention sublayer）最优。

这符合直觉。浅层 injection 让 action 信息 "implicit in latents"，深层 injection 让 action 信息 "explicit in attention"。

### 10. Action encoding 影响不大

Raw-action (event stream, 182-d multi-hot) vs Meta-action (API-like, typed schema) 在同样 injection mode 下差距很小：SSIM 0.847 vs 0.863。

**结论**: encoding granularity 不是 dominant factor，alignment quality 和 injection strategy 才是。

## CNC: 终极目标

他们定义了 Completely Neural Computer (CNC) 的四个条件：

1. **Turing complete** —— 理论上能算任何东西
2. **Universally programmable** —— 能 "安装" 新能力并 reuse
3. **Behavior-consistent** —— 不主动改就行为不变
4. **Machine-native semantics** —— 有自己的机器语义，不是模仿传统计算机

这个 roadmap 很 honest。paper 明确说 current prototypes 离 CNC 很远，只是 "outline of a distinct machine form has begun to emerge at prototype scale"。

### 最难的两个问题

**Symbolic stability**: 算术都做不好，更别说复杂 reasoning。可能需要 neuro-symbolic integration (Calanzone et al., https://arxiv.org/abs/2502.15655) 或 fundamentally new architecture。

**Capability reuse**: 你 "install" 了一个 routine（比如 incident response），下次遇到类似情况能不能 reliable 调用？这牵涉到 continual learning (Parisi et al., https://arxiv.org/abs/1802.07569) 和 catastrophic forgetting (Kirkpatrick et al., https://arxiv.org/abs/1612.00796)。

paper 提了个 hypothesis：LSTM-style gating (Hochreiter & Schmidhuber, https://www.bioinf.jku.at/publications/older/2604.pdf) 可能能实现 "run/update separation" —— 普通输入 execute installed capability，explicit update 才改变 behavior。

## 与其他 system objects 的关系

Table 13 很 illuminating：

| System | Organized around | Source of truth |
|---|---|---|
| Conventional computer | Explicit programs | Programs + machine state |
| AI agent | Tasks | External environments |
| World model | Environment dynamics | Solution |
| **Neural computer** | **Runtime** | **Installed capabilities + runtime state** |

NC 的独特之处：它 **organized around runtime**，source of truth 在 **learned system 内部**，不是 external environment。

Agent 是 "mediate existing computers"，NC 是 "be the computer"。

## 我的 Intuition 和联想

### 1. Video models 已经是 latent computers 的雏形

Sora 2 在算术上 71% 这个数字真的 make you think。结合 Wiedemer et al. (https://arxiv.org/abs/2509.20328) "video models are zero-shot learners and reasoners"，可能 video models 内部确实学到了某种 general computation 能力，只是还不稳定。

我的猜测：scale + RL + better conditioning 三者结合，1-2 年内 video-based NCs 会有显著突破。但 symbolic reliability 想达到 production-grade，可能需要 architectural breakthrough。

### 2. Tensor-uniform representation 是 real insight

Paper 提到传统计算机有 heterogeneous abstractions（scalars, pointers, files, sockets, processes），每个都有自己的 layout 和 failure modes，OS 用 disjoint subsystems 协调。

NC 把所有东西统一成 tensors。这让 operator fusion、memory planning、graph rewriting 可以 system-wide 应用（就像 MLIR/XLA 在 ML compiler 里做的，但 scale 到整个 computer）。

这可能是 NC 最 practical 的 intermediate benefit —— 即使不追求 full CNC，tensor-uniform 的 system stack 已经能简化很多 engineering。

### 3. Programming language as learned semantics

NC 的 "language" 是从数据里学出来的，没有 human-designed syntax/semantics boundary。Natural language 就是 programming language。

这让我想到 LLMs as programmable computers (Reynolds & McDonell, https://arxiv.org/abs/2102.07350) —— prompts 就是 programs。NC 把这个 idea extend 到 full computer stack。

### 4. Schmidhuber's ONE vision

ONE (One Big Net for Everything, https://arxiv.org/abs/1802.08864) 1992 年就设想了 single neural substrate incrementally absorbs skills。CNC 是 ONE 的 systems-level realization。

这个 lineage 很 important —— NC 不是凭空冒出来的，是 Schmidhuber 一脉 30 年工作的 systems-level 落地。

### 5. Closed-loop interaction 是 next frontier

Appendix A.2 描述了一个 online interaction pipeline（虽然没用进 final paper）。Closed-loop interaction 可以 induce natural curriculum、surface rare failure modes、支持 targeted data collection。

这跟 AlphaGo 的 self-play (https://www.nature.com/articles/nature24270)、recursive self-improvement (https://arxiv.org/abs/2507.02595) 是一个套路。我 bet 下一版 paper 会重点搞这个。

### 6. Energy efficiency concern

Video generation 极其 compute-intensive。一个 diffusion model rollout 16 帧要好几秒 GPU time。传统计算机执行 `ls` 命令是微秒级。

这个 gap 巨大。即使 NC 在 capability 上追上传统计算机，energy efficiency 上可能永远追不上。除非有 fundamentally faster inference（比如 neuromorphic chips, Mead & Ismail, https://link.springer.com/book/10.1007/978-1-4613-1783-9）。

### 7. Verifiability 问题

传统计算机的程序可以 formal verification。NC 的行为是 stochastic、continuous、high-dimensional 的，怎么 verify？

这可能需要 new type of "neural verification" —— 证明 latent state 的某些 invariants 在某种扰动下保持。Connect 到 representation learning 的 robustness 研究。

## 现实评估

**什么 work 了**：
- Basic rendering fidelity（54% char accuracy）
- I/O alignment（cursor 98.7% with SVG supervision）
- Short-horizon control（action injection）
- Data efficiency（110h > 1400h）

**什么没 work**：
- Symbolic computation（算术 4%）
- Long-horizon reasoning
- Capability reuse
- Runtime governance

**我的判断**：
- 1-2 年：video-based NCs 在 rendering 和 short-horizon control 上会显著 improve
- 3-5 年：会出现 hybrid neural-symbolic architectures
- 5-10 年：specific domains（CLI tools, simple GUIs）可能有 usable CNC
- 长期：full general-purpose CNC 需要 architectural breakthrough，单纯 scaling video models 不够

这个方向最 exciting 的地方是：它 fundamentally rethinks "what is a computer"。不是 faster CPU、不是 more RAM，是把整个 abstraction 都 dissolve 进 learned substrate。即使最终没实现 full CNC，沿途的 insights（tensor-uniform stack、learned programming semantics、visual cursor supervision）都很有价值。

---

**Further reading**:
- World Models: https://arxiv.org/abs/1803.10122
- Neural Turing Machines: https://arxiv.org/abs/1410.5401
- NeuralOS: https://arxiv.org/abs/2507.08800
- Wan2.1: https://arxiv.org/abs/2503.20314
- Schmidhuber's ONE: https://arxiv.org/abs/1802.08864
- Video models as reasoners: https://arxiv.org/abs/2509.20328
- LIMA (less is more): https://arxiv.org/abs/2305.11206
- ControlNet: https://arxiv.org/abs/2302.05543
- Attention is Turing Complete: https://arxiv.org/abs/2101.09773
- Lake & Baroni systematicity: https://arxiv.org/abs/1711.00350
- Gary Marcus critique: https://arxiv.org/abs/1801.00631

---

# Neural Computers: 深度解析

Andrej, 这篇 paper 确实是一个非常有野心的 vision piece, 让我详细 unpack 一下技术细节和我的理解。

## 1. 核心概念与 Motivation

这篇 paper 提出的核心问题是: **能否用一个 neural network 内化传统计算机的 computation, memory, I/O 到一个 learned runtime state 中?** 

传统计算机 (von Neumann architecture) 将 computation, memory, I/O 分离, 由 OS 协调。NCs 的 radical 之处在于: 把这些功能全部 fold 进一个 single neural system 的 latent state。

这个想法与 Schmidhuber 早期的 world models 工作 (Ha and Schmidhuber, 2018, https://arxiv.org/abs/1803.10122) 一脉相承, 但推向了 extreme: environment 本身就是一个 computer, 而 model 就是 runtime。

## 2. 形式化定义与公式解析

核心抽象是 Eq. (2.1):

$$h_t = F_\theta(h_{t-1}, x_t, u_t), \quad x_{t+1} \sim G_\theta(h_t)$$

变量含义:
- $h_t \in \mathbb{R}^d$: **latent runtime state** at time step $t$, 承载 persistent runtime memory。这是 NC 的 "computer state"
- $h_{t-1}$: 上一时刻的 runtime state, 提供 recurrence
- $x_t$: **observation** at time $t$, 在 paper 中是 screen frame (RGB pixels)
- $u_t$: **conditioning input** at time $t$, 可以是 user action (mouse/keyboard) 或 text prompt
- $F_\theta: \mathbb{R}^d \times \mathbb{R}^{d_x} \times \mathbb{R}^{d_u} \to \mathbb{R}^d$: **state-update function**, 参数为 $\theta$。整合 new observations 和 inputs 到 runtime state
- $G_\theta$: **decoder**, parameterizes 一个 distribution $p(x_{t+1} | h_t)$ over next frames
- $x_{t+1}$: 从 $G_\theta(h_t)$ 采样得到的下一帧

关键 insight: $h_t$ 不只是 world model 的 hidden state 用于 prediction, 它 **本身就是计算机的运行时状态**。传统 OS queue, device driver, UI toolkit 的功能都 shift 到 latent-state dynamics 中。

## 3. 两个 Prototype 实现细节

### 3.1 NC_CLIGen: Command Line Interface

**架构** (Figure 3):
- Base model: **Wan2.1** (Wan et al., 2025, https://arxiv.org/abs/2503.20314), SOTA video generation model
- VAE: encodes terminal frames 到 latent space
- CLIP image encoder (Radford et al., 2021, https://arxiv.org/abs/2103.00020): extracts visual features from first frame
- T5 text encoder (Raffel et al., 2020, https://arxiv.org/abs/1910.10683): embeds caption
- DiT (Diffusion Transformer) stack: processes concatenated conditioning features + diffusion noise
- Decoupled cross-attention: injects joint caption + first-frame context

**数据 pipeline**:
- **CLIGen (General)**: 823,989 video streams (~1,100 hours) from public asciinema .cast archives, 15 FPS
  - Three caption styles via Llama 3.1 70B (https://arxiv.org/abs/2407.21783):
    - Semantic (avg 55 words): high-level summary
    - Regular (avg 52 words): key commands + outputs
    - Detailed (avg 76 words): exhaustive transcription including exact text, colors, formatting
- **CLIGen (Clean)**: ~250k vhs scripts, after filtering 51.21% retained → 78k regular + 50k Python math traces
  - Deterministic Dockerized environments
  - Standardized font, palette, resolution

**Training**: AdamW, lr $5 \times 10^{-5}$, weight decay $10^{-2}$, bfloat16, gradient clipping 1.0
- CLIGen (General): ~15,000 H100 GPU hours
- CLIGen (Clean): ~7,000 H100 GPU hours

### 3.2 NC_GUIWorld: Graphical User Interface

**数据 pipeline**:
- **GUIWorld Random Slow** (~1,000 hours): longer pauses, deliberate cursor movements, avg cursor speed 1.51 px/frame
- **GUIWorld Random Fast** (~400 hours): denser cursor motion, avg cursor speed 195.15 px/frame
- **Claude CUA supervised** (~110 hours): goal-directed trajectories, avg cursor speed 3.79 px/frame
- Environment: Ubuntu 22.04, XFCE4, 1024×768, 15 FPS
- Apps: Firefox ESR, GIMP, VLC, VS Code, Calculator, Terminal, file manager, Mahjongg

**四种 action injection modes** (Figure 7):

**1. External conditioning**:
$$z'_{1:T} = z_{1:T} + \Delta z_{1:T}(u_{1:T})$$
- $z_{1:T}$: VAE latents for video sequence
- $u_{1:T}$: temporally aligned action features
- $\Delta z_{1:T}(u_{1:T})$: residual update from temporal self-attention + action cross-attention
- Action 信息只在 input level modulate latents, transformer 不直接 attend action tokens

**2. Contextual conditioning**:
- Action tokens $A \in \mathbb{R}^{L_a \times D}$ 与 visual tokens $V \in \mathbb{R}^{L_v \times D}$ concatenate 成 $[V; A]$
- Structured temporal mask (Figure 12):
  - V2V block: frame $V_i$ 只 attend $\pm w$ window 内的 frames
  - V2A block: frame $V_i$ 只 attend $j \in [\max(0, i-\ell), \min(i, A-1)]$ 的 actions, $\ell$ 是 action lag
  - A2V block: action $A_i$ 只 attend $t \geq i + \ell$ 的 frames (boundary clipping)
  - A2A block: strict diagonal, 每个 action 只 attend 自己

**3. Residual conditioning** (类似 ControlNet, Zhang et al., 2023, https://arxiv.org/abs/2302.05543):
$$\tilde{h}^{(l)} = h^{(l)} + \Delta h^{(l)}(a, \text{mouse})$$
- $h^{(l)}$: hidden states at layer $l$
- $\Delta h^{(l)}(a, \text{mouse})$: auxiliary action module 输出的 residual update
- 不修改 transformer block 内部 computation

**4. Internal conditioning** (类似 Matrix-Game-2, He et al., 2025, https://arxiv.org/abs/2508.13009):
$$h' = \text{FFN}\Big(h + \text{CA}_{\text{text}}\big(\text{SA}(h), c\big) + \text{CA}_{\text{action}}(h, a)\Big)$$
- SA: self-attention
- CA_text: text/reference cross-attention
- CA_action: dedicated action cross-attention sub-layer
- Action features 作为 keys/values 直接注入 transformer block 内部

## 4. 关键实验数据深度分析

### Experiment 1: VAE Reconstruction Quality

| Font size | PSNR | SSIM |
|-----------|------|------|
| 13 px | 40.77 dB | 0.989 |
| 6 px | 高但 localized blurring | 高 |

**Intuition**: Wan2.1 VAE 对 regular terminal content adequate, 但 extreme dense text (6px) 会有 localized blurring。Background regions dominate global PSNR/SSIM, 所以这些 metrics 可能 mask 局部问题。这呼应了 NeuralOS paper (Rivard et al., 2025, https://arxiv.org/abs/2507.08800) 对 generic natural-image VAEs 在 structured screenshots 上表现差的批评。

### Experiment 2: Training Plateau

- PSNR/SSIM 在 ~25k steps plateau
- Extended training (up to 460k steps) 可以略微 degrade performance
- **Intuition**: 大部分 learnable structured patterns 早期就学到了, further gains 需要 higher-quality, better-paced supervision。这让我联想到 "double descent" 现象 (Belkin et al., 2019, https://doi.org/10.1073/pnas.1821543116) - 过度训练可能反而有害。

### Experiment 3: Caption Specificity

| Prompt style | PSNR | SSIM | Avg. words |
|--------------|------|------|------------|
| Semantic | 21.90 | 0.813 | 55 |
| Regular | 23.63 | 0.843 | 52 |
| Detailed | 26.89 | 0.867 | 76 |

**Intuition**: Detailed captions 提供 ~5 dB PSNR gain。与 natural images 不同, terminal frames 主要由 text placement 决定。Detailed captions act as scaffolding, explicitly specifying which tokens appear where。这与 recent work on prompt engineering for precise generation 一致。

### Experiment 4: Character-Level Accuracy

| Steps (k) | Char. acc. | Exact line |
|-----------|------------|------------|
| 0 | 0.03 | 0.01 |
| 20 | 0.33 | 0.30 |
| 40 | 0.52 | 0.26 |
| 60 | 0.54 | 0.31 |

**Protocol** (Appendix B.2):
- Tesseract OCR on 5 uniformly sampled frame pairs
- Character accuracy: $\text{char\_acc} = \max\Big(0, 1 - \frac{d(s, t)}{\max(|s|, 1)}\Big)$
  - $s, t$: concatenated ground-truth and generated texts
  - $d(s, t)$: Levenshtein distance
  - $\max(|s|, 1)$: 防止除零
- Exact-line accuracy: $\text{exact\_line\_acc} = \frac{1}{N_g} \sum_{i=1}^{N_g} \mathbf{1}[i \leq N_p \land p_i = g_i]$
  - $N_g, N_p$: ground-truth 和 generated 的行数
  - $g_i, p_i$: 第 $i$ 行的 normalized text

**Intuition**: Character accuracy 0.54 说明 model 已经学到 text structure, font rendering, spatial relationships, 但离 perfect 还有距离。这是从 "plausible" 到 "usable" 的关键一步。

### Experiment 5: Arithmetic Probe

| Model | Accuracy |
|-------|----------|
| Wan2.1 | 0% |
| NC_CLIGen | 4% |
| Veo3.1 | 2% |
| Sora2 | 71% |

**Intuition**: 这是 paper 最 sobering 的结果之一。Sora2 的 71% 是 outlier, 可能 reflect system-level advantages 或 additional RL training (Table 6 列出三个 hypotheses)。这呼应了 Gary Marcus 对 deep learning symbolic reasoning 的批评 (https://arxiv.org/abs/1801.00631), 以及 Lake and Baroni 2018 的 systematicity challenge (https://arxiv.org/abs/1711.00350)。

### Experiment 6: Reprompting

- NC_CLIGen arithmetic: **4% → 83%** with reprompting
- **Key insight**: 大部分 "reasoning" gain 来自 better specification and instruction-following, 而非 native computation
- Paper 明确说: "we do not treat it as direct evidence of native arithmetic inside the NC backbone"

这让我想到 chain-of-thought prompting (Wei et al., 2022, https://arxiv.org/abs/2201.11903) - 很多 apparent reasoning 其实是 conditioning-assisted。

### Experiment 7: Data Quality > Scale

| Split | Hours | FVD_all | SSIM_all | LPIPS_all |
|-------|-------|---------|----------|-----------|
| Untrained baseline | - | 149.61 | 0.496 | 0.605 |
| Random Fast | 400 | 48.17 | 0.695 | 0.483 |
| Random Slow | 1000 | 20.37 | 0.830 | 0.237 |
| Claude CUA | 110 | 14.72 | 0.885 | 0.144 |

**Intuition**: 110 hours 的 goal-directed data 打败 1400 hours 的 random exploration。这与 LIMA paper (Zhou et al., 2023, https://arxiv.org/abs/2305.11206) 的 "less is more for alignment" 哲学一致。Goal-directed trajectories 提供更清晰的 action semantics 和可预测的 state transitions。

### Experiment 8: Cursor Control

| Loss variant | Cursor accuracy |
|--------------|-----------------|
| Position (x, y) only | 8.7% |
| Position (x, y) + Fourier | 13.5% |
| Position (x, y) + SVG mask/ref | **98.7%** |

**SVG cursor pipeline**:
- 从 logs 到 normalized trajectories: 
$$x_t = \frac{s x_{\text{screen}, t} + p_x}{w_{\text{dst}} - 1}, \quad y_t = \frac{s y_{\text{screen}, t} + p_y}{h_{\text{dst}} - 1}$$
  - $s$: uniform scale
  - $(p_x, p_y)$: padding offsets
  - $(w_{\text{dst}}, h_{\text{dst}})$: target resolution
- SVG arrow template positioned at $(x_t, y_t)$, alpha-blended over neutral background
- Produces: cursor foreground $f_t \in [-1, 1]^{3 \times H \times W}$ + soft mask $m_t \in [0, 1]^{1 \times H \times W}$
- Reference stream: $t=0$ full desktop + all-ones mask; $t>0$ cursor foreground + cursor mask only
- Diffusion input: $\text{concat}(z_{1:T}, \tau_{1:T}, z_{1:T}^{\text{ref}})$

**Intuition**: Cursor 必须作为 visual object 学习, 而非 abstract coordinates。这让我想到 embodied AI 中 visual grounding 的重要性 - agent 需要 grounded 在 pixel-level representation 中, 而非 symbolic state。

### Experiment 9: Action Injection Schemes

| Mode | SSIM_+15 | LPIPS_+15 | FVD_+15 |
|------|----------|-----------|---------|
| Untrained | 0.326 | 0.649 | 184.3 |
| External | 0.746 | 0.251 | 33.4 |
| Contextual | 0.813 | 0.190 | 24.8 |
| Residual | 0.857 | 0.138 | 18.8 |
| **Internal** | **0.863** | 0.141 | **14.5** |

**Intuition**: Deeper injection consistently 更好。External (input-level) → Contextual (token-level) → Residual (block-external) → Internal (block-internal)。这符合直觉: action 信息越深入 transformer 的 computation, integration 越紧密。

### Experiment 10: Action Encoding

| Encoding | SSIM_+15 | LPIPS_+15 | FVD_+15 |
|----------|----------|-----------|---------|
| Raw-action (event-stream) | 0.847 | 0.144 | 16.6 |
| Meta-action (API-like) | 0.863 | 0.141 | 14.5 |

**Meta-action schema** (Table 16):
- Type 0: None
- Type 1: Mouse Click/Drag (button, click_count, drag_flag)
- Type 2: Mouse Scroll (direction, amount)
- Type 3: Keyboard Type (text → shared text encoder)
- Type 4: Shortcut (shortcut_id)

**Intuition**: Encoding granularity 不是 dominant factor。这让我想到 recent work on action representations in RL - 从 primitive actions 到 hierarchical actions 的转变通常带来 moderate gains, 但 architecture 和 data quality 才是 game-changer。

## 5. CNC 定义与 Roadmap

### 四个 CNC Requirements (Table 14):

**1. Turing complete**
- 定义: 对于任何 Turing machine, 存在 initial memory state 使 NC 能 exact emulate
- 注意: 虽然 RNN, NTM (Graves et al., 2014, https://arxiv.org/abs/1410.5401), DNC (Graves et al., 2016, https://www.nature.com/articles/nature20101) 在 asymptotic sense 是 Turing complete (Siegelmann and Sontag, 1992, https://dl.acm.org/doi/10.1145/130385.130432), 但 finite precision 实例不能 Turing complete
- 需要: unbounded effective memory (progressive parameter growth (Fritzke, 1994, https://papers.nips.cc/paper/1994) 或 context growth (Vaswani et al., 2017, https://arxiv.org/abs/1706.03762))

**2. Universally programmable**
- 定义: 对于每个 Turing machine, 存在 input sequence 使 NC 实现新的 memory state representing that machine
- 路径: compositional neural programs (Pierrot et al., 2019, https://papers.nips.cc/paper/2019)

**3. Behavior-consistent**
- Run/Update contract:
  - **Run**: invoke installed capability without silently changing persistent behavior
  - **Update**: behavior-changing modification 通过 explicit programming interface
  - Required boundaries: state (what persists), update (what counts as reprogramming), evidence (what can be replayed, compared, rolled back)
- Hypothesis: gating mechanisms like LSTM (Hochreiter and Schmidhuber, 1997, https://www.bioinf.jku.at/publications/older/2604.pdf) 可以实现 conditional invariance

**4. Machine-native semantics**
- 不只是 imitate conventional computers with neural components
- 要 develop 自己的 machine semantics 和 programming interfaces
- 支持: probabilistic inference over high-dimensional latent states (Kingma and Welling, 2013, https://arxiv.org/abs/1312.6114), dense memory retrieval, end-to-end differentiable pipelines

### 系统对象对比 (Table 13)

| System object | Organized around | Source of truth | Primary role |
|---------------|------------------|-----------------|--------------|
| Conventional computer | Explicit programs | Explicit programs + machine state | Reliably execute explicit programs |
| AI agent | Tasks | External environments, tools, workflow state | Accomplish tasks through existing software stack |
| World model | Environment dynamics | Solution | Predict/roll out environment evolution |
| **Neural computer** | **Runtime** | **Installed capabilities + runtime state inside learned system** | **Sustain execution, accumulate capability, govern updates within one learned machine** |

## 6. 与相关工作的关系

### World Models 谱系
- Schmidhuber 1990 (https://arxiv.org/abs/1803.10122 附录): 最早提出 fully recurrent neural world models for dynamic RL and planning
- Ha and Schmidhuber 2018 (https://arxiv.org/abs/1803.10122): "World Models" paper, neural networks internalize environment dynamics
- Dreamer (Hafner et al., 2019, https://arxiv.org/abs/1912.01603): latent imagination for control
- Genie 3 (Bruce et al., 2024, https://arxiv.org/abs/2401.08901): action-controllable generative environments

### Neural Memory Architectures
- Fast weight programmers (Schmidhuber, 1992, https://link.springer.com/article/10.1007/BF00367903): 早期 differentiable memory
- Self-referential weight matrices (Schmidhuber, 1993, https://link.springer.com/chapter/10.1007/978-1-4471-2083-2_87)
- Neural Turing Machines (Graves et al., 2014): neural controller + external memory
- Differentiable Neural Computers (Graves et al., 2016): 改进 NTM
- Neural Programmer-Interpreters (Reed and De Freitas, 2015, https://arxiv.org/abs/1511.06279): neural networks execute structured procedures

### Video Generation Models
- Veo 3.1 (Google, 2025, https://deepmind.google/models/veo/)
- Sora 2 (OpenAI, 2025, https://openai.com/sora/)
- Movie Gen (Polyak et al., 2024, https://arxiv.org/abs/2410.13720)
- Wan2.1 (Wan et al., 2025): 本 paper 的 base model

### Computer Use Agents
- Anthropic Claude CUA (https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)
- OpenAI CUA (https://openai.com/index/computer-using-agent/)
- MetaGPT (Hong et al., 2023, https://arxiv.org/abs/2308.00352)
- GPTSwarm (Zhuge et al., 2024, https://arxiv.org/abs/2404.09642)

### Neural OS 相关
- NeuralOS (Rivard et al., 2025, https://arxiv.org/abs/2507.08800): neural simulation of GUIs
- Neural Virtual Machine (Katz et al., 2019, https://www.sciencedirect.com/science/article/pii/S0893608019300706)
- NeuroLISP (Davis et al., 2022, https://arxiv.org/abs/2105.06166): symbolic programming with attractor neural networks

## 7. 我的 Intuition 和联想

### 7.1 Video Models as Latent Computers

Sora2 在 arithmetic probe 上达到 71% 是一个非常 striking 的数据点。这暗示 video models 可能在内部学到了某些 symbolic reasoning 能力, 与 Wiedemer et al., 2025 (https://arxiv.org/abs/2509.20328) 的 "video models are zero-shot learners and reasoners" 一致。

可能的解释:
1. **Scale effect**: Sora2 可能比 Wan2.1 大很多, 更多 parameters → 更强的 in-context computation
2. **RL training**: OpenAI 可能对 Sora2 做了 RL training (reward shaping for math)
3. **System-level reprompting**: 内部可能有 LLM 做 computation 然后 condition generation

这让我想到 **Turing completeness of Transformers** (Pérez et al., 2021, https://arxiv.org/abs/2101.09773) - 理论上 attention is Turing complete, 但实际中 finite precision 和 finite context 限制了表达能力。

### 7.2 Latent State as Operating System

NC 的核心 insight 是把 OS 的功能 (scheduling, memory management, I/O coordination) 全部 shift 到 latent state dynamics。这让我想到几个方向:

**与 Differentiable Programming 的关系**:
- JAX (https://github.com/google/jax), PyTorch 已经实现了 end-to-end differentiable pipelines
- NC 把这个 idea 推到 extreme: 整个 computer 都是 differentiable 的
- "differentiable configuration of the computer itself" (Innes et al., 2019, https://arxiv.org/abs/1907.07587)

**与 Neural Architecture Search 的关系**:
- NC 提到 "neural network generation via NC interaction"
- 这与 NAS (Elsken et al., 2019, https://arxiv.org/abs/1908.00712) 有 overlap, 但 NC 更 radical: interaction logs 就是 programs

### 7.3 Tensor-Uniform Representation

Paper 提到 "tensors and tensor-to-tensor transformations act as primary computational primitives"。这是一个非常重要的 insight:

传统计算机的 heterogeneous abstractions:
- Scalars, pointers, linked structures, files, sockets, processes
- 每个有自己的 memory layout, invariants, APIs, failure modes
- OS 通过 disjoint subsystems 协调 (virtual memory, filesystems, networking, scheduling, drivers)

NC 的 tensor-uniform pipeline:
- 所有数据类型都用 tensors 表示
- 共享 intermediate representation
- Operator fusion, memory planning, computational-graph rewriting 可以 system-wide 应用 (Vasilache et al., 2018, https://arxiv.org/abs/1802.04730)
- Multimodal computation (vision, language, audio, control, planning) 都在一个 representational space

这让我想到 **MLIR** (https://mlir.llvm.org/) 和 **XLA** (https://www.tensorflow.org/xla) - 它们已经在朝这个方向走, 但 NC 把整个 computer stack 都 tensor 化了。

### 7.4 Programming Language Semantics

NC 的 "learned programming-language semantics" 是一个非常深刻的 insight:
- LLMs 可以 viewed as programmable computers, prompts 是 programs (Reynolds and McDonell, 2021, https://arxiv.org/abs/2102.07350)
- Natural language 作为 programming language, 不受 human-specified syntax/semantics boundary 约束
- 训练数据 (I/O traces) 远比 high-quality human-written code 丰富

这呼应了 **program synthesis by demonstration** (Cypher and Halbert, 1993) 的传统, 但用 neural networks 实现 end-to-end learning。

### 7.5 Symbolic vs Neural 的 Tension

Paper 坦诚承认 symbolic computation 是 bottleneck。reprompting 从 4% → 83% 的结果特别 illuminating:

- 大部分 apparent "reasoning" 是 conditioning-assisted, 不是 native computation
- 这支持了 Gary Marcus (https://arxiv.org/abs/1801.00631) 和 Lake and Baroni (https://arxiv.org/abs/1711.00350) 的批评
- Paper 提出的 "machine-native neural architectures" hypothesis 很有意思: 可能需要 explicitly designed discrete operations, compositional structures, verifiable computation

这让我想到 **neuro-symbolic integration** 的研究方向:
- Logically consistent language models (Calanzone et al., 2025, https://arxiv.org/abs/2502.15655)
- Differentiable theorem provers
- Neural theorem provers (Irving et al., 2016)

### 7.6 ONE (Schmidhuber, 2018)

Paper 提到 **ONE (One Big Net for Everything)** (Schmidhuber, 2018, https://arxiv.org/abs/1802.08864): single neural substrate incrementally absorbs and reuses diverse learned skills。

CNC 可以 viewed as ONE 的 systems-level realization:
- Specialized world-model-like components 不再是 separate external systems
- 而是 installable capabilities within one persistent neural runtime
- 这与 continual learning (Parisi et al., 2019, https://arxiv.org/abs/1802.07569) 和 avoiding catastrophic forgetting (Kirkpatrick et al., 2017, https://arxiv.org/abs/1612.00796) 紧密相关

### 7.7 Data Efficiency Insight

110 hours supervised > 1400 hours random exploration。这个结果非常重要:

- 与 **LIMA** (Zhou et al., 2023, https://arxiv.org/abs/2305.11206) "1,000 examples for alignment" 哲学一致
- 与 **InstructGPT** (Ouyang et al., 2022, https://arxiv.org/abs/2203.02155) 的 RLHF 数据效率一致
- 说明: **alignment quality > data scale** for action-response learning

### 7.8 Closed-Loop Interaction 的潜力

Appendix A.2 描述了一个 agentic online interaction pipeline, 虽然没有用在 final pipeline 中, 但潜力巨大:

- Closed-loop interaction 可以 induce natural curriculum
- Surface rare 和 safety-critical failure modes
- 支持 targeted data collection (specific tools, error recovery, long-horizon tasks)

这让我想到:
- **Self-play** (Silver et al., 2017, https://www.nature.com/articles/nature24270) 在 AlphaGo 中的作用
- **Recursive self-improvement** (Zhuge et al., 2026, https://arxiv.org/abs/2507.02595)
- **Agent-as-a-judge** (Zhuge et al., 2024, https://arxiv.org/abs/2410.10934) 用于 evaluation

### 7.9 Computer-Human Interface 的未来

Paper 的 Figure 9 描述了一个 fascinating 的 systems-level shift:

1. **Conventional computer era**: human 直接使用 computer
2. **Agent era**: agent mediates existing computers, world model 作为 parallel predictive layer
3. **NC era**: unified learned runtime 作为 machine 本身

这让我想到 **Douglas Engelbart's Mother of All Demos** (1968) - 从 command-line 到 GUI 到 agent-mediated 到 neural computer, interface 的 abstraction level 不断提高。

### 7.10 潜在的 Criticisms

虽然 vision 很 compelling, 但几个 concerns:

1. **Symbolic stability**: 当前 NC instantiations 在 symbolic reasoning 上还很弱
2. **Capability reuse**: 如何 install 一个 routine 并 reliably reuse 它?
3. **Runtime governance**: 如何 ensure behavior consistency without explicit reprogramming?
4. **Energy efficiency**: Video generation 非常 compute-intensive, vs 传统计算机的 deterministic execution
5. **Verifiability**: 传统计算机的程序可以 formal verification, NC 的行为如何 verify?
6. **Privacy/security**: 如果整个 computer state 在 latent space, 如何 enforce access control?

## 8. 实验数据表汇总

### CLI Experiments

| Experiment | Key result |
|------------|------------|
| VAE reconstruction | 13px: PSNR 40.77, SSIM 0.989 |
| Training plateau | PSNR/SSIM plateau ~25k steps |
| Caption specificity | Detailed: +5 dB PSNR vs Semantic |
| Character accuracy | 60k steps: 0.54 char acc, 0.31 exact line |
| Arithmetic probe | Wan2.1: 0%, NC: 4%, Sora2: 71% |
| Reprompting | 4% → 83% with reprompting |

### GUI Experiments

| Experiment | Key result |
|------------|------------|
| Data quality | 110h CUA > 1400h random (FVD 14.72 vs 48.17) |
| Cursor control | SVG mask: 98.7% vs position-only: 8.7% |
| Injection schemes | Internal: SSIM 0.863, FVD 14.5 (best) |
| Action encoding | Meta-action slightly > raw-action |

## 9. 总结与 Outlook

这篇 paper 是一个 **vision paper with initial prototypes**。它不 claim 已经实现了 CNC, 而是 demonstrate:
- NCs 可以 learn elementary runtime primitives (I/O alignment, short-horizon control)
- Stable reuse, symbolic reliability, runtime governance 仍然 unresolved
- Video models 作为 pragmatic prototype substrate, 但 long-term 需要 fundamentally new architecture

**我的 prediction**: 
1. **Short-term** (1-2 years): Video-based NCs 会显著 improve, 特别是随着 Sora/Veo/Wan 等 video models 的 scaling
2. **Medium-term** (3-5 years): 会看到 hybrid neural-symbolic architectures, 结合 video generation 和 explicit symbolic reasoning modules
3. **Long-term** (5-10 years): 可能出现 first practical CNCs, 在 specific domains (如 CLI tools, simple GUIs) 达到 usable 级别
4. **Fundamental challenge**: Symbolic stability 和 capability reuse 需要 architectural breakthroughs, 单纯 scaling video models 不够

这个方向让我非常 excited, 因为它 fundamentally rethinks what a "computer" is。与 quantum computing 的 paradigm shift 不同, NC 的 paradigm shift 是把 hardware/software boundary dissolve 到一个 learned substrate 中。

---

**References for further reading**:
- World Models: https://arxiv.org/abs/1803.10122
- Neural Turing Machines: https://arxiv.org/abs/1410.5401
- NeuralOS: https://arxiv.org/abs/2507.08800
- Wan2.1: https://arxiv.org/abs/2503.20314
- Schmidhuber's ONE: https://arxiv.org/abs/1802.08864
- Genie 3: https://arxiv.org/abs/2401.08901
- Attention is Turing Complete: https://arxiv.org/abs/2101.09773
- Video models as reasoners: https://arxiv.org/abs/2509.20328
- MetaGPT: https://arxiv.org/abs/2308.00352
- GPTSwarm: https://arxiv.org/abs/2404.09642
- ControlNet: https://arxiv.org/abs/2302.05543
- LIMA: https://arxiv.org/abs/2305.11206
- Neural Programmer-Interpreters: https://arxiv.org/abs/1511.06279
