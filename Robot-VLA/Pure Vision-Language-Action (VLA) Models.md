---
source_pdf: Pure Vision-Language-Action (VLA) Models.pdf
paper_sha256: 60096bf95584ec8b520919978d9ee12b69306d3bb8b7b3114d485a04e290453b
processed_at: '2026-08-06T07:04:48-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Vision-Language-Action (VLA) Models: “人话”版拆解

Andrej，咱们换个频道，抛开那些学术包装，直接聊点实在的。这篇 paper 本质上是在给机器人圈画一张地图，告诉大家：我们现在想让机器人听懂人话、看懂环境、还能自己动手干活，到底走到哪一步了，各路神仙都在用什么招数。

## 1. 核心矛盾：书呆子 vs 泥瓦匠

VLA 想解决的核心问题，其实就是把 VLM 这个“书呆子”变成“泥瓦匠”。

VLM 现在很厉害，能看图说话，能写文章，懂语义，有常识。但是，你让它输出一段连续的 7-DoF 机械臂关节控制指令，它就傻眼了。语言是离散的、符号化的，而物理世界的动作是连续的、高频的、容错率极低的。VLA 就是想把这个 gap 桥接起来。

这篇 paper 把大家尝试桥接 gap 的方法分成了四大派系。

## 2. 四大派系的技术直觉

### 派系一：Autoregressive (AR) — 把动作当单词造句

**人话**：这派人的思路最简单粗暴。既然 GPT 造句那么牛，那我就把机器人的动作（比如关节角度 1.5 度）切分成一个个小 bin，当成词汇表里的新单词。然后让 Transformer 玩 next-token prediction，根据看到画面和指令，“吐”出下一个动作单词。

**代表人物**：RT-1, RT-2, OpenVLA, Gato。

**技术细节**：
公式特别直观：
$$ \mathcal{L}_{AR} = -\sum \log P_\theta(a_t | o_{1:t}, l) $$
就是在给定视觉观测 $o$ 和语言指令 $l$ 的情况下，最大化下一个动作 token $a_t$ 的概率。
RT-2 最聪明的一点在于，它把 action tokens 直接塞进了 VLM 的 vocabulary 里。这意味着 action tokens 享受了整个互联网 web-scale 的预训练红利，模型把“抓起可乐”这个动作指令和“抓起可乐”这个文本概念在同一个 embedding space 里对齐了。
[RT-2 Link](https://arxiv.org/abs/2307.15818)

**痛点**：
1.  **离散化损失**：你把连续的 0.001 毫米精度的动作强行分到 256 个 bin 里，控制精度天生就受限，干精细活儿手抖。
2.  **Error accumulation**：像写文章一样写动作，前面错一个 token，后面全错。机器人撞墙了还得继续往下写。
3.  **Latency**：高频控制需要 50Hz，你一个个 token 蹦出来，机器人早摔了。为了解决这个问题，有人搞了 FAST tokenizer，把一串动作压缩成一个“词”，像 BPE 一样，减少 token 数量。([FAST Link](https://arxiv.org/abs/2501.09747))

### 派系二：Diffusion — 把动作当图片去噪

**人话**：这派人觉得 AR 太硬核了，物理动作本质是连续的，强行离散化不对。他们借用了画图 AI 的思路：把一段完美的机械臂轨迹想象成一张高清图，然后一步步往里加噪声变成纯噪声。训练的时候，让模型学怎么把噪声一步步还原回高清轨迹。推理的时候，随便给一堆随机噪声，模型就能“去噪”出一条合理的动作轨迹。

**代表人物**：Diffusion Policy, $\pi_0$, RDT-1B。

**技术细节**：
Diffusion 的 loss 公式看着唬人，其实就是让模型预测加进去的噪声 $\epsilon$ 是什么：
$$ \mathcal{L}_{simple} = \mathbb{E}[\| \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} a_0 + \sqrt{1 - \bar\alpha_t}\epsilon, c) \|^2] $$
这里 $a_0$ 是真实动作轨迹，$c$ 是视觉语言条件，$\bar\alpha_t$ 是噪声调度表。模型 $\epsilon_\theta$ 学会了在当前噪声状态下，这团乱码里原本的信号长什么样。
最近爆火的 $\pi_0$ 用了 flow matching，比 DDPM 更简洁，就是把噪声和真实数据之间画条直线，让模型学怎么沿着直线走。
[$\pi_0$ Link](https://arxiv.org/abs/2410.24164)

**爽点**：
1.  **Multi-modality**：同一场景下，杯子可以放左边也可以放右边。AR 模型可能会把两者的概率平均掉，导致动作卡在中间。Diffusion 天然支持多峰分布，随便采一个都是合理的。
2.  **平滑**：轨迹天然平滑，因为是在连续空间优化。

**痛点**：
慢。太慢了。你要去噪 10 步甚至 50 步才能出一个动作 chunk，这在实时控制里是致命的。所以现在一堆人在搞 consistency distillation（一步去噪）或者 TinyVLA 来加速。([TinyVLA Link](https://arxiv.org/abs/2409.12514))

### 派系三：Reinforcement Learning (RL) — 让机器人知道疼

**人话**：前面两派都是 Behavior Cloning (BC) 的变种，本质是“师傅做，徒弟看”。但徒弟看着看着就会 overfit，遇到没见过的场景就傻了。这派人主张，得让机器人自己去试，试对了给糖，试错了挨打。这就是 RL fine-tune。

**代表人物**：SafeVLA, ConRFT, SimpleVLA-RL。

**技术细节**：
RL 的核心在于 reward function $r(s,a)$。现在最流行的是拿 VLM 来当 reward model。比如你问 VLM：“这个动作符合‘把杯子放到桌上’的指令吗？”VLM 说符合就给 1，不符合给 0。
$$ \mathcal{L}_{RL} = -\mathbb{E}_{a \sim \pi_\theta}[r(o,a)] + \beta \cdot \text{KL}[\pi_\theta \| \pi_{SFT}] $$
这里有个关键设计：加了 KL 散度约束，防止 RL 把 SFT 模型好不容易学到的常识给练偏了。
SimpleVLA-RL 极端到只用一条 trajectory 和 0/1 二值 reward 就能 kick-start 训练，这说明 reward signal 的质量比数量重要得多。
([SimpleVLA-RL Link](https://github.com/PRIME-RL/SimpleVLA-RL))

**痛点**：
在 7B+ 的 VLA 模型上跑 RL，方差大得离谱，训练不稳定，reward 难定义，sim-to-real gap 在 RL 里被放大得更严重。

### 派系四：Hybrid & Specialized — 混搭与魔改

**人话**：前三派都有硬伤，于是有人开始混搭。比如上层用 LLM 做 reasoning（System 2，慢思考），下层用 Diffusion 做高精动作生成（System 1，快反应）。或者针对自动驾驶、人形机器人这种特殊 embodiment 专门魔改架构。

**代表人物**：HybridVLA, Fast-in-Slow, 3D-VLA。

**HybridVLA 的直觉**：
$$ \text{LLM generates reasoning } r \rightarrow \text{Diffusion generates action } a $$
先想后做，符合人类直觉。
([HybridVLA Link](https://arxiv.org/abs/2503.10631))

---

## 3. 数据：穷人的诅咒

这篇 paper 里让我最有共鸣的一点是关于数据的焦虑。

VLA 和 LLM 最大的差距就在数据。
- LLM 有整个互联网的文本，PB 级别。
- VLA 呢？最大的 Open X-Embodiment (OXE) 也就 1M+ episodes，这还是 21 家机构联合搞出来的。([OXE Link](https://arxiv.org/abs/2310.08895))

更惨的是，物理世界的数据是 long-tail 的。互联网上“抓杯子”的视频很多，但“在特定角度避开障碍物抓特定杯子”的精确动力学数据几乎没有。

所以现在的出路就两条：
1.  **疯狂搞仿真**：Isaac Gym, Genesis，在虚拟世界里跑几亿次，寄希望于 domain randomization 能迁移到现实。([Isaac Gym Link](https://arxiv.org/abs/2108.10470))
2.  **榨干互联网视频**：像 GR-1 那样，先在人类操作视频上做 video prediction pretraining，学一个 world dynamics 的 prior，再去 fine-tune 机器人动作。这其实就是在偷数据。([GR-1 Link](https://arxiv.org/abs/2401.07552))

## 4. 几个硬骨头

除了 paper 里列的那些 challenge，我特别想强调几个 field 里大家不愿意多提的痛点：

1.  **Inference 频率的死亡线**：控制机器人需要 50-100Hz。你现在拿个 7B 的 VLA 模型，AR 逐个 token decode，或者 Diffusion 跑 10 步去噪，算上 image preprocessing，能跑出 5Hz 就谢天谢地了。EdgeVLA 拼命压缩，DeeR-VLA 搞 early exit，VLA-Cache 复用 KV，都是在跟 latency 拼命。([EdgeVLA Link](https://arxiv.org/abs/2507.14049))
2.  **Pseudo-interaction 的幻觉**：现在的 VLA 其实没有真正的 closed-loop feedback。它生成 action 就像 LLM 生成文本一样，一旦生成完就不管了。即使环境变了，它也不会根据新的 sensor feedback 修正。这导致一旦执行偏离预期，模型没有 re-plan 的能力。这就是个巨大的 system 1，没有 system 2。
3.  **Action Tokenizer 的 bottleneck**：FAST 是个好开头，但远远不够。什么时候我们能找到一种 action representation，既像自然语言那样 compressible、hierarchical，又能无缝 decode 成 continuous control signals？这是整个 field 等待的 breakthrough。

## 5. 我的赌注

看完这篇 paper，如果要我对未来 12-18 个月下注，我会赌在这两个方向：

**赌注一：World Model 和 VLA 的彻底融合**
现在的 VLA 还是 reactive 的。未来一定是 predictive 的。模型不光要输出 action，还要在 latent space 里 predict 下一秒的 visual state。如果预测的和真实的发生偏离，立刻 re-plan。
像 EnerVerse 和 DreamGen 这种用 video generation 做 world model 的尝试，方向对了。([EnerVerse Link](https://arxiv.org/abs/2501.01895))
公式上可能会演化成 joint optimization：
$$ \mathcal{L} = \mathcal{L}_{action} + \lambda \mathcal{L}_{world\_model} $$

**赌注二：Latent Action Space 的统治**
我们现在都在用 raw action space (关节角度、end-effector pose)。这太底层了，就像用像素值来写小说。人类是通过“意图”和“原语”来控制的。未来一定会出现一种 pre-trained 的 latent action space，把高层意图和底层控制解耦。像 LAPO (Latent Action Pretraining) 这种从视频里学 latent action 的工作，是通往这条路的钥匙。([LAPO Link](https://openreview.net/forum?id=VYOe2eBQeh))

---

总结一句，这篇 paper 做了个很有价值的梳理：VLA 不是魔法，它就是把 LLM 的 reasoning 能力强行嫁接到 physical control 上。现在嫁接的技术还很粗糙，接口在漏水（latency），排异反应很严重（sim-to-real），营养跟不上（data scarcity）。但方向是没跑偏的，下一个 GPT 时刻大概率会出在 robotics foundation model 上。

---

# Vision-Language-Action (VLA) Models: A Comprehensive Survey — 深度解析

Andrej，这篇paper把过去几年VLA领域的300多篇work做了系统taxonomy，本质上是把robotics control从policy-based paradigm重新framing为generative modeling问题。下面我从intuition、technical details、critical commentary几个层面给你拆解。

---

## 1. 这篇paper的核心positioning

VLA模型的本质是把VLM从**passive sequence generators**转变成**active agents**。传统的robotic stack是 *perception → planning → control* 的模块化pipeline；VLA做的是把这三个stage压进**single sequence modeling framework**：

$$
\underbrace{o_{1:t}, s_{1:t}, l}_{\text{vision/state/language tokens}} \xrightarrow{\text{Transformer}} \underbrace{a_{t:t+H}}_{\text{action tokens}}
$$

其中 $o_{1:t}$ 是visual observations (image patches)，$s_{1:t}$ 是robot proprioceptive state (joint positions, gripper state等)，$l$ 是language instruction，$a_{t:t+H}$ 是chunk of future actions with horizon $H$。

paper的taxonomy轴是**action generation strategy**，分四类：
1. **Autoregression-based** (3.1)
2. **Diffusion-based** (3.2，含flow matching, VAE)  
3. **Reinforcement-based fine-tune** (3.3)
4. **Hybrid & specialized** (3.4)

这个taxonomy的intuition是：每种generation paradigm对应不同的action distribution assumption和inductive bias，决定了model能capture什么类型的policy。

---

## 2. Autoregression-Based VLA (Section 3.1)

### 2.1 核心mechanism

把action $a_t \in \mathbb{R}^d$ discretize成tokens，然后用standard next-token prediction：

$$
\mathcal{L}_{\text{AR}} = -\sum_{t=1}^{T} \sum_{k=1}^{K} \log P_\theta(a_t^{(k)} | o_{1:t}, s_{1:t}, l, a_{<t}^{(<k)})
$$

其中 $a_t^{(k)}$ 是第 $t$ 个action step的第 $k$ 维被quantize到 $K$ 个bin里的token。这是RT-1/RT-2/OpenVLA的core formula。

### 2.2 代表性methods的技术insight

**Gato (2022)** [32]: 证明single Transformer能跨heterogeneous modalities统一token化。Action tokenization直接用bin discretization。Link: https://arxiv.org/abs/2205.06175

**RT-1 (2022)** [30]: 130k demonstrations训练，用**FiLM-based multimodal fusion**：
$$
h = \gamma(z_l) \odot f_v(o) + \beta(z_l)
$$
其中 $z_l$ 是language embedding，$f_v$ 是visual encoder，$\gamma, \beta$ 是learned affine。这让instruction作为gain/bias注入visual features。Link: https://arxiv.org/abs/2212.06817

**RT-2 (2023)** [34]: 关键insight是把action tokens直接嫁接到VLM vocabulary上 — action就是"另一种language"，享受web-scale VLM pretraining的semantic priors。Link: https://arxiv.org/abs/2307.15818

**PaLM-E (2023)** [33]: 把continuous robot observations作为token prefix注入PaLM。设计上是：
$$
h = \text{PaLM}([\text{ViT}(o), \text{embed}(s), \text{tokens}(l), \text{tokens}(a)])
$$

Link: https://arxiv.org/abs/2303.03378

**OpenVLA (2024)** [21]: 7B model，970k trajectories，在OXE上训练。架构是Prismatic VLM backbone + action tokenization head。Link: https://arxiv.org/abs/2406.09246

**FAST (2025)** [22]: 解决action sequence tokenization效率问题。核心是把固定bin discretization换成**variable-length tokenization**，类似BPE for actions：
$$
\text{FAST}: (a_t)_{t=1}^{H} \xrightarrow{\text{compressor}} (\tau_1, \tau_2, ..., \tau_M), \quad M \ll H
$$
通过在action chunks上learn一个lookup table，把7-DoF轨迹从~60 tokens压到~10 tokens。Link: https://arxiv.org/abs/2501.09747

### 2.3 Reasoning增强方向 (3.1.2)

**Inner Monologue [47]**: pre-action planning + post-action reflection:
$$
a_t = f_\theta(o_t, l, \underbrace{r_{t-1}}_{\text{previous reflection}})
$$

**ECoT (2024)** [51]: Embodied Chain-of-Thought，在action前先generate reasoning trace:
$$
P_\theta(r_t, a_t | \text{context}) = P_\theta(r_t | \text{context}) \cdot P_\theta(a_t | r_t, \text{context})
$$

Link: https://arxiv.org/abs/2407.08693

**CoT-VLA (2025)** [59]: 视觉chain-of-thought，预测intermediate visual goals before action。Link: https://arxiv.org/abs/2503.22020

### 2.4 效率优化方向 (3.1.4)

这是paper里很rich的一块，因为有deployment的现实需求：

**DeeR-VLA [83]**: Multi-exit architecture，根据confidence提前停止解码：
$$
\text{exit if } H(A_t) < \tau, \quad H(\cdot) = -\sum_a P(a)\log P(a)
$$

**VLA-Cache [85]**: Reuse KV states across timesteps（因为visual context在robot task里temporally smooth）:
$$
\text{cache}: (K_t, V_t) \approx (K_{t-1}, V_{t-1}) \text{ if } \|o_t - o_{t-1}\| < \delta
$$

**BitVLA [89]**: 1-bit quantization，权重 $W \in \{-1, +1\}^{d \times d}$，内存降到30%。

**MoLe-VLA [87]**: MoE layer skipping，每个token动态路由：
$$
y = \sum_{e \in \text{top-}k} g_e \cdot \text{Layer}_e(x), \quad g_e = \text{softmax}(\text{router}(x))_e
$$
40% computation reduction。

**PD-VLA [88]**: Parallel fixed-point decoding — 把autoregressive decoding换成parallel Jacobi iteration on a fixed point:
$$
a^{(k+1)} = f_\theta(o, l, a^{(k)})
$$
直到convergence。

### 2.5 这类的intuition

AR-VLA的**优点**：能inherit LLM scaling laws、reasoning能力、in-context learning。
**缺点**：
- **Error accumulation**: 每个token错一个，后续context都被污染
- **Latency**: autoregressive decoding + high-frequency control (typical 10-50Hz)有天然矛盾
- **Discretization loss**: 把continuous action chunk强行bin化，可能丢掉fine-grained control信息

---

## 3. Diffusion-Based VLA (Section 3.2)

### 3.1 Core mechanism

Diffusion Policy的core formula (DDPM形式):
$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, a_0, \epsilon \sim \mathcal{N}(0, I)} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} a_0 + \sqrt{1 - \bar\alpha_t}\epsilon, t, c) \|^2 \right]
$$

其中：
- $a_0 \in \mathbb{R}^{H \times d}$ 是ground truth action chunk
- $t \in \{1, ..., T\}$ 是diffusion timestep
- $\bar\alpha_t = \prod_{i=1}^t \alpha_i$ 是cumulative noise schedule
- $c$ 是conditioning $(o, s, l)$
- $\epsilon_\theta$ 是learned denoising network

Action sampling从 $a_T \sim \mathcal{N}(0, I)$ 开始，iterative denoise到 $a_0$。

**π0 [2]** 用的是flow matching variant:
$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t \sim U(0,1), a_0, a_1} \left[ \| v_\theta(a_t, t, c) - (a_1 - a_0) \|^2 \right]
$$
其中 $a_t = (1-t)a_0 + t a_1$，$a_1$ 是noise，$a_0$ 是data。Link: https://arxiv.org/abs/2410.24164

### 3.2 为什么diffusion适合robotic action？

**Multi-modality**: 同一observation下可能有多种valid action — deterministic regression会average掉。Diffusion的probabilistic nature直接建模这个distribution：
$$
p_\theta(a | o) = \int p(a_T) \prod_{t} p_\theta(a_{t-1} | a_t, o) \, da_T
$$

**Smoothness**: trajectory是temporal smooth的，diffusion的iterative refinement天然fit这个prior。

**Geometry-aware**: SE(3)-DiffusionFields [130] 把diffusion扩展到manifold上：
$$
\mathcal{L}_{\text{SE(3)}} = \mathbb{E}[\| \nabla \log p(a_T^{\text{SE(3)}} | c) \|^2]
$$

### 3.3 代表性methods

**Diffusion Policy (2024)** [7]: Chi et al.的经典工作。Link: https://arxiv.org/abs/2303.04137

**3D Diffuser Actor [133]**: 把3D scene embedding注入condition，对trajectory生成做conditional diffusion。Link: https://arxiv.org/abs/2402.10885

**RDT-1B (2025)** [135]: 1B parameter diffusion Transformer for bimanual manipulation。关键设计：temporal attention + cross-attention对condition。Link: https://arxiv.org/abs/2410.04874

**Dita (2025)** [146]: Scale diffusion Transformer to generalist VLA，连续action denoise。Link: https://arxiv.org/abs/2503.19757

**Diffusion-VLA [1]**: Self-Generated Reasoning + Diffusion Policy。Reasonging先generate symbolic intermediates $r$，然后：
$$
a \sim p_\theta(a | o, l, r), \quad r \sim p_\phi(r | o, l)
$$
Link: https://arxiv.org/abs/2412.03293

**CogACT [142]**: Semantic scene graphs作为中间表示。Link: https://arxiv.org/abs/2411.19650

### 3.4 Cognitive-inspired architectures

**MinD (2025)** [155]: Dual-system — low-frequency video prediction (System 2, slow reasoning) + high-frequency diffusion policy (System 1, fast reactive):
$$
\underbrace{\hat{o}_{t+1:t+H_{\text{slow}}}}_{\text{video prediction}} \rightarrow \underbrace{a_{t:t+H_{\text{fast}}}}_{\text{diffusion action}}
$$
Link: https://arxiv.org/abs/2506.18897

**TriVLA [157]**: Triple-system (36Hz) — vision-language reasoning / dynamics perception / policy learning三个decoupled模块。Link: https://arxiv.org/abs/2507.01424

### 3.5 Deployment效率

**TinyVLA [149]**: LoRA fine-tuning，只5% trainable params。Link: https://arxiv.org/abs/2409.12514

**SmolVLA [150]**: Consumer hardware deployment + async inference。Link: https://arxiv.org/abs/2506.01844

**CEED-VLA [235]**: Consistency distillation + early-exit，4× speedup:
$$
a_{\text{1-step}} = f_\theta^{\text{consistency}}(a_T, c) \approx a_{\text{multi-step}}^{\text{diffusion}}
$$
Link: https://arxiv.org/abs/2506.13725

### 3.6 这类的trade-off

**优点**: multi-modal action distribution, smooth trajectories, no discretization loss, SE(3)-aware可以做到。
**缺点**: 
- **Computational cost**: 每个inference要跑10-100次denoise step
- **Latency**: 实时控制难达到
- **Temporal coherence under distribution shift**: 长horizon任务里diffusion assumption会break down

---

## 4. Reinforcement-Based Fine-Tune (Section 3.3)

### 4.1 为什么需要RL fine-tune?

纯SFT (behavior cloning)的VLA有两个问题：
1. **Distribution shift**: 部署时model state会偏离training distribution
2. **No notion of optimality**: BC只learn mimic，不learn what's good

RL fine-tune用reward $r(s, a)$ 来refine policy：
$$
\mathcal{L}_{\text{RL}} = -\mathbb{E}_{a \sim \pi_\theta} [r(o, s, a)] + \beta \cdot \text{KL}[\pi_\theta \| \pi_{\text{SFT}}]
$$

### 4.2 Reward sources

**Vision-language reward proxies**:
- **VIP [174]**: Value-Implicit Pre-training，self-supervised goal-conditioned value function。Link: https://arxiv.org/abs/2210.00030
- **LIV [175]**: Joint vision-language reward from action-free videos with text。Link: https://arxiv.org/abs/2306.00958

Reward computation:
$$
r(o, o_g) = -\| \phi(o) - \phi(o_g) \|^2
$$
其中 $\phi$ 是pretrained visual encoder。

**VLM-based reward**: LLM生成reward function code or reward proxy:
$$
r(o, s, a) = \text{VLM}(\text{"is this action good for task } l \text{?"}, o, a)
$$

### 4.3 代表性methods

**SafeVLA [182]**: Constrained policy optimization。Link: https://arxiv.org/abs/2503.03480
$$
\max_\pi \mathbb{E}_{\pi}\left[\sum_t \gamma^t r_t\right] \quad \text{s.t.} \quad \mathbb{E}_{\pi}\left[\sum_t \gamma^t c_t\right] \leq d
$$
其中 $c_t$ 是safety cost，$d$ 是constraint threshold。Lagrangian转化为：
$$
\mathcal{L} = \mathbb{E}[r] - \lambda \cdot (\mathbb{E}[c] - d)
$$

**iRe-VLA [183]**: Combine SFT stability with RL exploration。Link: https://arxiv.org/abs/2501.16664

**SimpleVLA-RL [187]**: 仅用**single trajectory + binary reward** (0/1)训练。Insight是只要reward signal对，一条trajectory的signal就足够kick-start learning。
$$
r = \mathbb{1}[\text{task completed}]
$$

**ConRFT [185]**: Offline BC + Q-learning + online consistency。Hybrid策略：
$$
\mathcal{L}_{\text{ConRFT}} = \underbrace{\mathcal{L}_{\text{BC}}}_{\text{offline}} + \underbrace{\mathcal{L}_{\text{Q}}}_{\text{offline value}} + \underbrace{\mathcal{L}_{\text{consistency}}}_{\text{online}}
$$
Link: https://arxiv.org/abs/2502.05450

**AutoVLA [189]**: Chain-of-Thought Reasoning + Group Relative Policy Optimization (GRPO):
$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\frac{1}{|G|} \sum_i \min(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i)\right] - \beta \text{KL}
$$
其中 $\rho_i = \frac{\pi_\theta(a_i | c)}{\pi_{\text{old}}(a_i | c)}$，$G$ 是group size，$\hat{A}_i$ 是group-relative advantage。Link: https://arxiv.org/abs/2506.13757

### 4.4 RL-VLA的tension

**核心难题**: VLA的policy空间是high-dimensional multimodal distribution，传统RL的policy gradient估计variance很大。paper里没有fully discuss的一个深层问题是：**如何为LLM-scale model设计stable RL training?** GRPO/PPO的clip机制在7B+ VLA上是否还能work？这是个open question。

---

## 5. Hybrid & Specialized (Section 3.4)

### 5.1 Hybrid architectures

**HybridVLA [201]**: 7B unified framework同时做diffusion trajectory + autoregressive reasoning:
$$
\text{HybridVLA}: \underbrace{\text{AR-LLM generates } r}_{\text{reasoning}} \rightarrow \underbrace{\text{Diffusion head generates } a}_{\text{action}}
$$
Link: https://arxiv.org/abs/2503.10631

**Fast-in-Slow [243]**: Kahneman双系统理论operationalize。Fast module嵌在slow VLM里:
$$
a_{\text{fast}} = f_{\text{low-latency}}(o, s), \quad r_{\text{slow}} = f_{\text{VLM}}(o, l)
$$

### 5.2 3D-aware spatial reasoning

**CLIPort [206]**: "What" + "Where" dual pathway:
$$
\text{pick}_t, \text{place}_t = f_{\text{where}}(\text{CLIP}_v(o)) \odot f_{\text{what}}(\text{CLIP}_l(l))
$$
Link: https://arxiv.org/abs/2109.12098

**VoxPoser [245]**: LLM生成composable 3D value maps:
$$
V_{\text{target}}, V_{\text{avoid}} = \text{LLM}(l) \rightarrow \text{voxel grid} \in \mathbb{R}^{X \times Y \times Z}
$$
Link: https://arxiv.org/abs/2307.05973

**3D-VLA [207]**: Generative 3D world model + diffusion action。Link: https://arxiv.org/abs/2403.09631

**ReKep [208]**: Relational keypoint graphs:
$$
\mathcal{G} = (\mathcal{V}_{\text{keypoints}}, \mathcal{E}_{\text{constraints}}(t))
$$
其中edges是time-varying relational constraints。Link: https://arxiv.org/abs/2409.01652

### 5.3 Domain specialization

- **CoVLA [217]**: 50k paired language-trajectory videos for AD。Link: https://arxiv.org/abs/2408.10845
- **LeVERB [188]**: Humanoid whole-body control，hierarchical VLA + RL dynamics。Link: https://arxiv.org/abs/2506.13751
- **Helix [218]**: Figure AI的humanoid unified policy
- **AutoRT [219]**: Multi-robot fleet orchestration，PaLM-E/RT-2作为orchestrator
- **ShowUI [247]**: GUI manipulation as VLA (点击/dragging/form filling)
- **CubeRobot [221]**: Rubik's Cube用VisionCoT dual-loop + memory stream

### 5.4 Foundation models

**DROID [226]**: 150k+ trajectories，1000+ objects。Link: https://arxiv.org/abs/2403.12945
**OXE [8]**: 22 datasets，527 skills，160k tasks，1M+ episodes。Link: https://arxiv.org/abs/2310.08895
**RoboBrain [231]**: Unified perception-reasoning-planning foundation model。Link: https://arxiv.org/abs/2502.21257

---

## 6. Datasets & Benchmarks Landscape (Section 4)

paper给了一个很好的表格(Table 5)。关键观察：

### 6.1 Real-world dataset scaling curve

| Dataset | Year | Episodes | Tasks | Modality |
|---------|------|----------|-------|----------|
| MIME [259] | 2018 | 8,300 | 20 | RGBD |
| RoboNet [261] | 2019 | 162,000 | - | RGB |
| MT-Opt [262] | 2021 | 800,000 | 12 | RGB |
| BridgeData [24] | 2021 | 60,100 | 24 | RGBD |
| RT-1 [264] | 2022 | 13,000 | 700 | RGB |
| RH20T [268] | 2024 | 110,000 | 147 | RGBD |
| DROID [269] | 2024 | 76,000 | - | RGBD |
| OXE [8] | 2025 | >1,000,000 | 160,266 | RGBD |

数据从单lab到multi-institution collaboration，规模6年增长100倍。但相对互联网文本数据，仍是**5-6 orders of magnitude smaller**。

### 6.2 Simulation benchmarks

- **Meta-World [273]**: 50 tasks
- **RLBench [275]**: 100 tasks
- **VIMA-Bench [276]**: 4-level generalization evaluation
- **CALVIN [277]**: Long-horizon language-conditioned
- **LIBERO [278]**: Knowledge transfer benchmark
- **RoboCasa [281]**: >100k episodes, 100 tasks
- **Mobile ALOHA [280]**: 825 episodes bimanual mobile

### 6.3 Evaluation metrics

- **Success Rate**: % tasks completed
- **Language Following Rate**: instruction adherence
- **L2 trajectory error** (autonomous driving)
- **Interactive Navigation Score**: $\text{INS} = w_1 \cdot \text{PathEff} + w_2 \cdot \text{EffortEff}$

---

## 7. Simulators (Section 5)

| Simulator | Type | Use case |
|-----------|------|----------|
| THOR [25] | Photo-realistic indoor | Navigation, VQA |
| Habitat [26] | Real scanned buildings | Embodied AI |
| MuJoCo [27] | Physics engine | RL training |
| Isaac Gym [28] | GPU-accelerated | Large-scale RL |
| CARLA [29] | Urban driving | Autonomous driving |
| iGibson [304,305] | Real-home replicas | Household tasks |
| Genesis [309] | Universal physics | General robotics |
| ManiSkill3 [311] | GPU rendering | Manipulation |
| Agibot World [310] | Large-scale manipulation | New platform |

关键trend是GPU-based simulation + photorealistic rendering。Isaac Gym在RT-2/X等大规模RL训练里是基础设施。

---

## 8. Challenges & Future (Section 7)

paper总结了5个核心challenge：

### 8.1 Data scarcity
真实世界数据采集成本极高。即使OXE 1M+ episodes也只相当于web text的微秒级数据。**关键问题**：如何用video pretraining（YouTube等infinite resource）来fill the gap？GR-1/GR-2 [68,69]、EnerVerse [163]、DreamGen [162] 是这个方向的尝试。

### 8.2 Architectural heterogeneity
没有unified standard。Vision encoder可能是ViT/DINOv2/SigLIP，language backbone可能是PaLM/LLaMA/Qwen，action head可能是discrete tokens/continuous vectors/diffusion。这让cross-model comparison几乎impossible。

### 8.3 Real-time inference
VLA模型动辄7B+，autoregressive decoding latency + high-freq control需求是天然矛盾。EdgeVLA [232] 报告6× speedup，DeeR-VLA [233] 用early exit。但实际部署仍需要50-100Hz control frequency，目前SOTA VLA大多在5-15Hz。

### 8.4 Pseudo-interaction
Model基于statistical co-occurrence生成action，没有真正的causal reasoning。Karpathy你之前在Tesla讲的"system 1 vs system 2"问题，在VLA里依然存在 — 大多数VLA只是超强的system 1。

### 8.5 Benchmark limitations
现有benchmark主要是tabletop manipulation，open-world deployment评估缺失。

### 8.6 Opportunities

paper提的4个opportunity：
1. **World modeling**: VLA作为proto-world model
2. **Causal reasoning**: 真正的interactive intelligence
3. **Virtual-real integration**: Trillion-scale synthetic data
4. **Societal embedding**: Trustworthy ecosystem

---

## 9. 我的critical commentary

### 9.1 paper的盲区

paper写得comprehensive，但有几个aspect我觉得没有讲透：

**(1) Action representation debate**: Discrete tokens (AR-VLA) vs Continuous (diffusion) vs Hybrid — 这个是**整个VLA field最核心的unresolved question**。paper把两个category并列，但其实背后是LLM scaling law能否extend到continuous control的深层问题。RT-2证明了discrete action tokens能inherit web knowledge，但dexterous manipulation的精度离散化损失明显。这个tension需要更深入的discussion。

**(2) World model integration**: 3D-VLA [207]、WorldVLA [31]、EnerVerse [163] 都开始把world model和policy learning统一。这个方向可能是VLA的next paradigm，但paper只是在diffusion category里lightly提到。在我看来，**world model + policy的joint training**是比AR/Diffusion更fundamental的设计choice。

**(3) Pre-training vs Fine-tuning data ratio**: 像π0、GR00T N1这些foundation model，pretraining用了多少互联网video vs fine-tuning用多少robot demos？这个ratio是scaling VLA的关键，paper没有quantitative analysis。

**(4) Embodiment gap**: OpenVLA、π0.5等claim cross-embodiment generalization，但实际sim-to-real和robot-to-robot transfer的真实成功率，paper里只有success rate数字，没有failure mode分析。

### 9.2 Field的real bottleneck

我观察到的真正瓶颈，paper提了但没强调：

1. **Hardware-in-the-loop data**: 互联网video只能学semantic priors，不能学dynamics。Robot需要physical interaction data，这个数据增长是sublinear的（每个new robot embodiment都要重新collect）。

2. **Reward specification**: VLA的RL fine-tune依然需要reward，但robotic task的reward engineering比LLM的RLHF难得多 — 因为物理world没有明确的"good response"信号。

3. **Long-horizon credit assignment**: Hierarchical VLA (Hi Robot [56], LoHoVLA [91])是方向，但实际long-horizon tasks（>100 steps）的success rate依然很低。

### 9.3 几个我觉得promising但paper没充分cover的方向

1. **Latent action pretraining (LAPO [116])**: 从video里learn latent action space，再用少量robot data align。这是data efficiency的key。
2. **Test-time scaling for VLA**: LLM的test-time compute (CoT, self-consistency) 在VLA里还没systematically explored。
3. **Embodied RLHF**: 把Constitutional AI的思想转到robotics — "what is a safe action under uncertainty"。

---

## 10. 总结intuition

VLA的本质tension是：**LLM-scale的semantic understanding vs physical world的fine-grained control**。这两个axis目前是用四种generation paradigm分别address：

- **AR** maximizes semantic understanding inheritance (但discretization loss)
- **Diffusion** maximizes action distribution expressiveness (但latency cost)
- **RL** maximizes task-specific optimality (但sample efficiency差)
- **Hybrid** 试探各种组合

未来12-18个月，我predict几个关键milestone：

1. **World model + VLA unified architecture** — 把action prediction和world prediction放进同一个token stream
2. **Action tokenizer的breakthrough** — 类似BPE对NLP的影响，FAST是first step但不够
3. **Cross-embodiment foundation model** — 单一model控制所有形态机器人
4. **Real-time inference的hardware-aware co-design** — 不只是model compression，是model+hardware的joint optimization

paper的contribution是把这个fragmented field的taxonomy理清楚了，但真正推动field前进的会是foundation model scaling laws在physical world能否replicate — 这是open question。

---

## Reference Links

| Topic | Paper | Link |
|-------|-------|------|
| AR-VLA | RT-1 | https://arxiv.org/abs/2212.06817 |
| AR-VLA | RT-2 | https://arxiv.org/abs/2307.15818 |
| AR-VLA | PaLM-E | https://arxiv.org/abs/2303.03378 |
| AR-VLA | OpenVLA | https://arxiv.org/abs/2406.09246 |
| AR-VLA | FAST | https://arxiv.org/abs/2501.09747 |
| AR-VLA | Gato | https://arxiv.org/abs/2205.06175 |
| AR-VLA | CoT-VLA | https://arxiv.org/abs/2503.22020 |
| Diffusion | Diffusion Policy | https://arxiv.org/abs/2303.04137 |
| Diffusion | π0 | https://arxiv.org/abs/2410.24164 |
| Diffusion | RDT-1B | https://arxiv.org/abs/2410.04874 |
| Diffusion | Diffusion-VLA | https://arxiv.org/abs/2412.03293 |
| Diffusion | Dita | https://arxiv.org/abs/2503.19757 |
| RL | VIP | https://arxiv.org/abs/2210.00030 |
| RL | LIV | https://arxiv.org/abs/2306.00958 |
| RL | SafeVLA | https://arxiv.org/abs/2503.03480 |
| RL | ConRFT | https://arxiv.org/abs/2502.05450 |
| RL | AutoVLA | https://arxiv.org/abs/2506.13757 |
| Hybrid | HybridVLA | https://arxiv.org/abs/2503.10631 |
| Spatial | VoxPoser | https://arxiv.org/abs/2307.05973 |
| Spatial | 3D-VLA | https://arxiv.org/abs/2403.09631 |
| Spatial | ReKep | https://arxiv.org/abs/2409.01652 |
| Data | OXE | https://arxiv.org/abs/2310.08895 |
| Data | DROID | https://arxiv.org/abs/2403.12945 |
| Data | BridgeData | https://arxiv.org/abs/2109.07891 |
| Sim | Isaac Gym | https://arxiv.org/abs/2108.10470 |
| Sim | CARLA | https://arxiv.org/abs/1711.03938 |
| Foundation | RoboBrain | https://arxiv.org/abs/2502.21257 |
| Foundation | GR00T N1 | https://arxiv.org/abs/2503.14734 |
| Foundation | π0.5 | https://arxiv.org/abs/2504.16054 |
| World Model | WorldVLA | https://arxiv.org/abs/2506.21539 |
| World Model | EnerVerse | https://arxiv.org/abs/2501.01895 |
| World Model | DreamGen | https://arxiv.org/abs/2505.12705 |

---

希望这个detailed breakdown对你的intuition building有帮助。如果你想drill deeper到任何具体方向（比如π0的flow matching细节、OpenVLA的tokenization、或者RL fine-tuning的stability问题），可以继续聊。这个field的下一步突破，我赌在**world model + VLA的unified foundation model** + **action tokenizer breakthrough**这两件事上。
