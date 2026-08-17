---
source_pdf: World Action Models The Next Frontier in Embodied AI.pdf
paper_sha256: 4b729cae694621b327156c6cd75778764664a6a3a8cebce7637162d89d544b6b
processed_at: '2026-08-13T04:58:41-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

Karpathy 我尽量讲得像我们在白板前聊天，不堆公式，只讲 intuition。

---

## 这篇 paper 到底在说什么

先说一个最根本的问题：**现在的 robot policy 不够聪明，因为它只学了一个 "看到什么就做什么" 的 mapping**。

你给它一张桌子的照片，说"把杯子拿起来"，它学会的是 "杯子在左 → 手往左伸" 这种统计相关性。它脑子里没有 "如果我手往左伸，杯子会怎么样" 这个预测过程。这就像一个人只会条件反射，不会 "想一下后果再动手"。

World Model 能想后果——给它当前状态和 action，它能预测 "下一秒世界会变成什么样"。但它自己不会动手，它只是个 simulator。

WAM 的核心 idea 就是：**把这两件事揉到一起，让 model 同时学会 "我该做什么" 和 "做完之后世界变成什么样"**。

直觉上，这两个问题是互相约束的。如果你能准确想象未来，那么从现在到未来的那个 action 就被物理规律框死了；反过来，如果你知道 action 是对的，那未来必然满足某些物理一致性。这种 mutual supervision 是 WAM 比纯 VLA 强的根本原因。

---

## 两个大的架构派系

paper 把所有方法分成两大类，这个划分其实特别直觉。

### Cascaded：先想清楚再动手

这派做法很简单：**先用 world model 生成一段 "未来会怎么演" 的 video，然后再从这段 video 反推出每一步该做什么 action**。

就像你打台球之前先在脑子里模拟一下球的轨迹，然后决定怎么打。world model 负责 "想象"，inverse dynamics model 负责 "从想象里提取 action"。

这个做法的好处是分工明确——world model 不用管 robot 长什么样，inverse dynamics 不用管长程物理预测。坏处也很明显：**两段式 error compounding**。你想的未来错了，action 一定跟着错。

这派内部又分两条路：

一条是 **生成真的 RGB video** 当中间载体。UniPi 是开创者，后来一堆改进——有的用 VLM 做 sub-goal 分解，有的用 keyframe interpolation，有的干脆不学 inverse dynamics，直接从 video 里算 optical flow 再用几何方法推出 SE(3) 变换（AVDC 这条路完全不需要 action annotation）。

另一条是 **不 decode 回 pixel，用 latent feature 当中间载体**。VPP 是代表，直接在 latent space 做 single-step prediction，能跑到 real-time frequency。问题是 generation quality 下降，所以 S-VAM 用 self-distillation 来 bridge。

### Joint：边想边做

这派是现在最火的。**world modeling 和 action generation 在同一个 model 里同时训练，不拆成两段**。

这里又分两个路线：

**Autoregressive 路线**：把 future state 和 action 都变成 token，causal left-to-right 预测。就像 GPT 一样，一个 token 一个 token 往外吐。这派早期的做法是 visual 和 action 用不同的 head 解码（GR-1），后来演化到全部 quantize 到同一个 vocabulary（CoT-VLA、WorldVLA），最近又开始往 latent space 转移，不做 explicit pixel prediction（VLA-JEPA）。

**Diffusion 路线**：用 multi-step denoising 同时 generate future state 和 action。这是当前主流。好处是不用 sequential decoding，能并行出 action，适合 closed-loop control。

Diffusion 路线内部又分 **Single DiT**（world 和 action 在同一个 trunk 里 joint denoise）和 **Multi DiT**（world 和 action 是两个独立 branch，通过 cross-attention / hidden-state / shared representation 交互）。

Single DiT 最有意思的设计是 UWM：world 和 action 各自独立控制 noise level，于是同一个 model 在 test time 可以切换成 policy / world model / inverse dynamics 四种模式。

Multi DiT 最 pragmatic 的设计是 Fast-WAM：训练时保留 video branch 提供 supervision，inference 时直接移除——只用 Video DiT 单次 forward 的 latent feature 去 condition action denoising。**inference 时根本不生成 video，但训练时享受了 world modeling 的好处**。

---

## 几个关键 mental model

帮你 build intuition 的几个比喻：

**Cascaded 像先写剧本再拍戏**。world model 写剧本（imagined future），inverse dynamics 拍戏（extract action）。剧本错了，戏一定崩。

**Joint Autoregressive 像边写边拍的连续剧**。一边想象下一幕，一边决定动作，但容易前后不一致，error 会累积。

**Joint Diffusion 像同时打磨剧本和动作的多稿创作**。world 和 action 一起 denoise，互相 refine，最后同时收敛到一个 coherent 的 result。

**Implicit Future Prediction（FLARE/FRAPPE 这派）像给 policy 装了一个 "第六感"**。它不 explicit 生成未来画面，但内部 representation 被强制 align 到 teacher encoder 的 future embedding。policy 看不到未来，但它的 internal state 已经 "预感" 到了未来。

---

## 数据这块的核心 insight

WAM 相对 VLA 的一个根本优势是 **数据消化能力**。

VLA 严格需要 $(o, a)$ pair——有 observation 必须有对应 action。这把数据量卡死了，因为 teleoperation 很贵。

WAM 因为有 world modeling objective，可以吃 action-free 的 video——只要 $(o, o')$ 就能学 "世界怎么演化"。这意味着 internet 上所有人类操作 video 都是 WAM 的训练数据。

paper 把数据分四类，从贵到便宜：
1. **Robot teleoperation**：最精确，最贵，跨 embodiment
2. **UMI-style human demo**：人在 wild 拿个 3D-printed gripper 拍，有 action constraint 又有 diversity
3. **Simulation**：physics engine 本质是个 perfect world model，提供 privileged supervision
4. **Internet egocentric video**：无限量，但没 action label

**WAM 的 magic 在于它能同时消化这四类数据**。用 robot teleoperation 学精确 action grounding，用 internet video 学 general world dynamics。这种 "mix 严格 paired data 和 unpaired video" 的能力是 VLA 做不到的。

---

## 评估的核心 insight

paper 对 evaluation 的梳理我觉得很有价值。它把 world modeling 评估分三层：

**第一层：看起来像不像**。PSNR、SSIM、LPIPS、FVD——这些是传统 video generation metric。

**第二层：物理上对不对**。物体有没有凭空消失、有没有违反重力、碰撞因果对不对。VideoPhy、Physics-IQ 这些 benchmark 在做这个。

**第三层：能不能 extract 出可执行 action**。这是 WAM 独有的维度。Wow, wo, val! 这个工作提了个 IDM Turing Test：用 inverse dynamics model 从生成的 video 推 action，然后真在 real robot 上执行看 success rate。结果发现很多 visual 上很 convincing 的 model，在这个 test 下 success rate 几乎为零。

**这说明 visual plausibility 不等于 action plausibility**。一个生成的视频可以看起来很漂亮、物理也对，但里面根本没有 "可提取的 action signal"。这就像一段动画看起来像人在跳舞，但你没法从动画里反推出每个关节的 torque。

paper 最后呼吁 community 建 joint evaluation benchmark——不只 separate 评 world modeling 和 action policy，而是评 "imagined future 和 executed action 之间的 causal consistency"。这是很重要的方向。

---

## 整个 field 的趋势

如果我画一张 evolution 图，大概是这样：

**从 VLA 到 WAM**：从 reactive mapping 到 predictive reasoning

**架构上从 Cascaded 到 Joint**：从两段式拆解到统一 co-optimization

**Joint 内部从 Autoregressive 到 Diffusion**：从 sequential token 到 parallel denoising

**Diffusion 内部从 Single Stream 到 Multi-Stream**：从单 trunk 到 multi-expert + cross-attention / hidden-state / shared representation coupling

**Future prediction 从 explicit pixel 到 implicit latent**：从生成 RGB 到 align 到 teacher embedding，甚至 test-time 完全移除 video branch

整个趋势的核心张力是 **predictive fidelity vs computational efficiency**。world modeling 越精确，action 越有物理 grounding，但 inference 越慢。DreamZero 把 WAM 推到 7Hz 已经很拼，但 non-generative VLA 是 50Hz。这个 gap 的根本原因是 diffusion 的 multi-step denoising。

paper 提了一个很有意思的 open question：**downstream control 真的需要 full diffusion synthesis 那么高的 fidelity 吗**？如果 performance gain 在远低于 full quality 时就 plateau 了，那正确的方向是 task-adaptive predictive fidelity，不是无脑提高 generation quality。

---

## 一句话总结

**WAM 把 robot policy 从 "条件反射" 升级成 "预判式行动"——它让 model 在 action 之前先想象 action 的物理后果，然后让这个想象反过来约束 action 本身**。

这个 paradigm shift 的意义在于：robot policy 不再只是学统计相关性，而是开始有 "物理直觉" 了。这跟 LeCun 一直在推的 JEPA 哲学是相通的——重要的不是 reconstruct pixel，而是在 abstract representation space 里预测 state transition。

但这离真正的 physical intelligence 还有距离。Long-horizon planning、multimodal physical state、latency、safety——这些都是 open problem。paper 在 Section 7 把这些列得很清楚，算是给 community 画了一张 roadmap。

---

# World Action Models (WAMs)：Embodied AI 的下一个范式跃迁

Karpathy 你好，这篇 paper 我仔细读完了，整体感觉是一个非常 systematize 的 survey，把 2024-2026 这两年间突然爆发的 "world model + action generation 耦合" 这条线梳理得相当清楚。下面我从 intuition 的角度一层层拆开讲。

---

## 1. 为什么需要 WAM 这个新概念

先把 motivation 讲清楚。当前的 VLA 模型比如 RT-2、OpenVLA、π₀，本质上学的是一个 **reactive observation→action mapping**：

$$\mathcal{L}_{\mathrm{VLA}} = \mathbb{E}_{(o, l, a) \sim \mathcal{D}} \left[ -\log p(a \mid o, l) \right]$$

这里 $o$ 是当前 observation（RGB、proprioception 等），$l$ 是 language instruction，$a$ 是 action token。这个 objective 的根本问题在于：**模型内部没有 explicit 的 "如果我执行 $a$，世界会变成什么样" 的预测**。它学的是统计相关性，不是因果 dynamics。

而 classical World Model 是另一个极端：

$$\mathcal{L}_{\mathrm{WM}} = \mathbb{E}_{(o, a, o') \sim \mathcal{D}} \left[ -\log p(o' \mid o, a) \right]$$

其中 $o'$ 是 next observation。WM 能预测 "world 在 action 干预下如何演化"，但它不直接生成 action——它是个 simulator，不是 agent。

WAM 的核心 insight 就是 **把这两件事统一在一个 joint distribution 里**：

$$\boxed{\mathcal{L}_{\mathrm{WAM}} = \mathbb{E}_{(o, l, o', a) \sim \mathcal{D}} \left[ -\log p(o', a \mid o, l) \right]}$$

直觉上的解释：agent 要同时回答两个问题——"我该做什么" 和 "做完之后世界会变成什么样"。这两个问题是相互约束的：如果你能准确想象未来 $o'$，那么从 $o$ 到 $o'$ 的 action $a$ 就被物理约束所限定；反过来，如果你知道 $a$ 是对的，那么 $o'$ 必然满足某些物理一致性。这种 mutual supervision 是 WAM 相对于纯 VLA 的根本优势。

paper 还专门 disambiguate 了几个容易混淆的概念：
- **VAM (Video Action Model)**：特指用 video prediction 作为中间载体的方法，是 WAM 的一个子集
- **Video Policy**：只是用了 video generation backbone 提取 spatiotemporal feature，但未必有 explicit 的 future state prediction 作为 supervision
- **AWM (Action World Model)**：早期文献用的术语，paper 改成 WAM 是为了强调 "Action" 和 "World" 是 co-equal 的，不是把 world model 当成 augmented simulator

homepage: https://openmoss.github.io/Awesome-WAM  
github: https://github.com/OpenMOSS/Awesome-WAM

---

## 2. 架构 Taxonomy 的核心二分：Cascaded vs Joint

这是整篇 paper 最核心的架构划分。我用一个 mental model 来帮你 build intuition。

### 2.1 Cascaded WAM：先想清楚再动手

形式化的 factorization：

$$p(o', a \mid o, l) = p(a \mid o', o, l) \cdot p(o' \mid o, l)$$

直觉：先用 world model "想象" 未来 $o'$，然后再从 $(o, o')$ 反推 action $a$。这相当于把 policy 拆成 "visual planner" + "inverse dynamics" 两段。

这种范式的好处是 **inductive bias 清晰**——world model 只管物理预测，不用学 robot kinematics；action decoder 只管 motor control，不用学长程 scene evolution。坏处是 **two-stage error compounding**：如果 $o'$ 预测错了，action 一定跟着错。

Cascaded 内部又分两条路：

#### (a) Explicit Planning via Pixel-Space

中间载体是 RGB video。代表作 UniPi (Du et al., NeurIPS 2023, https://arxiv.org/abs/2302.00111) 是奠基性的：text-conditioned spatiotemporal U-Net diffusion 生成 task execution video，然后用一个 lightweight CNN+MLP inverse dynamics model (IDM) 从相邻 frame pair 回归 action。

后续工作有几个关键改进方向：
- **Semantic intervention**：VLP (https://arxiv.org/abs/2310.10625) 用 PaLM-E 12B 做 hierarchical sub-action decomposition + tree search value scoring，缓解 single-pass long-horizon 的 semantic drift
- **Non-autoregressive keyframe**：RoboEnvision (https://arxiv.org/abs/2506.22007) 用 VLM 分解 subtask terminal state keyframe，再 interpolate
- **Deictic grounding**：This&That (https://arxiv.org/abs/2407.05530) 用 "this/that" + gesture coordinate 解决多同类物体的语言歧义
- **Geometric extraction**：AVDC (https://arxiv.org/abs/2310.08576) 完全抛弃 learned IDM，从生成 video 中算 dense optical flow，然后 analytic 地推导 SE(3) transformation——**训练时完全不需要 action annotation**，这是个很 elegant 的设计

Geometric extraction 这条线还有 3DFlowAction (https://arxiv.org/abs/2506.06199)、NovaFlow (https://arxiv.org/abs/2510.08568)、Dream2Flow (https://arxiv.org/abs/2512.24766) 等，趋势是从 2D flow 升到 3D flow，从需要训练到 zero-shot。

#### (b) Implicit Planning via Latent Representation

中间载体是 latent feature sequence，不 decode 回 pixel。代表是 VPP (https://arxiv.org/abs/2412.14803)：

- 用 pretrained VAE encode observation frame
- diffusion model 做 **single-step** latent sequence prediction（不是 multi-step denoising，这是关键加速点）
- lightweight policy network 直接从 latent 出 action

这样能达到 **real-time control frequency**，这是 pixel-space 方法做不到的。代价是 generation quality 下降，所以 S-VAM (https://arxiv.org/abs/2603.16195) 用 self-distillation 来 bridge：训练时 frozen multi-step SVD 做 teacher，inference 时用 one-step student。

LAPA (https://openreview.net/forum?id=VYOe2eBQeh) 走的是另一条路：用 VQ-VAE 学 latent action token，从 unlabeled video 自监督学 "state → latent action" prior，下游只需少量 real action annotation 来 align。villa-X (https://arxiv.org/abs/2507.23682) 在此基础上加了 proprioceptive Forward Dynamics Model 来 ground latent action 到物理 dynamics。

### 2.2 Joint WAM：边想边做

直接建模 joint distribution $p(o', a \mid o, l)$，world modeling 和 action generation 在同一个 model 里 co-optimize。这是当前最活跃的方向。

Joint WAM 内部按 generation mechanism 分两大支：

#### (a) Autoregressive Generation

把 future state 和 action 都 serialize 成 token，causal left-to-right decoding。这里又有三个子范式：

**Explicit Decoupled Representation**：modality 保持 heterogeneous，用 special token 路由到不同 head。GR-1 (https://openreview.net/forum?id=NxoFmGgWC9) 是开创性的：transformer 先在 video reconstruction 上 pretrain，然后 fine-tune 同时输出 future visual patch + continuous action，dual-branch head。GR-2 (https://arxiv.org/abs/2410.06158) 把 visual 也 discretize 成 VQGAN token，用 CVAE parameterize action chunk。

**Unified Discrete Representation**：所有 modality 全部 quantize 到一个共享 vocabulary，用同一个 next-token prediction head。挑战是 **action token 的 compounding error**——autoregressive 采样一长串 ungrounded action token 很容易飘。解决方案：
- CoT-VLA (https://arxiv.org/abs/2506.21539 之类，实际是 CVPR 2025 paper)：causal attention 先 autoregress 出 visual chain-of-thought，再 switch 到 full attention 同步预测 action token
- WorldVLA (https://arxiv.org/abs/2506.21539)：modality-specific causal masking，禁止当前 action token attend 到同 chunk 内已生成的 action，强制 ground 在 historical context
- F1 (https://arxiv.org/abs/2509.06951)：Mixture-of-Transformer，Generation expert 预测 VQ token，Action expert 做 foresight-guided inverse dynamics

**Predictive Latent Representation**：抛弃 explicit token，在 abstract continuous latent space 上 autoregress。VLA-JEPA (https://arxiv.org/abs/2602.10098) 是代表：extract continuous latent action token，conditional guide autoregressive world model 预测 frozen target network encode 的 future representation。因为 future frame 只作为 isolated supervision target，架构上 leakage-free。Action 通过 embodied action token condition flow-matching head 出来。这个思路跟 LeCun 的 JEPA 哲学一脉相承 (https://arxiv.org/abs/2301.08243)。

#### (b) Diffusion-based Generation

这是当前最主流的 Joint WAM 路线。核心是用 multi-step denoising 同时 generate future state 和 action，避开 autoregressive 的 sequential bottleneck。

按 backbone coupling 再分三类：

**Unified Stream (Single DiT)**：world 和 action 在同一个 DiT trunk 里 joint denoise。又分 explicit 和 implicit future prediction：

- *Explicit*：future observation 是 direct prediction target。PAD (https://arxiv.org/abs/2411.18179) 把 RGB、pose、depth encode 成 unified latent sequence，joint 预测 future frame + action chunk。VideoVLA (https://arxiv.org/abs/2512.06963) repurpose CogVideoX-5B (https://arxiv.org/abs/2408.06072) 的 video diffusion backbone，joint denoise future video latent + 7-DoF action chunk。
- UWM (https://arxiv.org/abs/2504.02792) 的设计很巧妙：world 和 action **各自独立控制 noise level**，于是同一个 model 可以在 test time switch 到 policy / forward dynamics / inverse dynamics / pure video generation 四种模式——这相当于一个 universal foundation model。
- DreamZero (https://arxiv.org/abs/2602.15922) 是 Physical Intelligence 的 π₀.7 工作：基于 Wan2.1-I2V-14B (https://arxiv.org/abs/2503.20314)，加 lightweight state/action encoder + action decoder，joint denoise video latent + action latent。用 KV-cache-based observation replacement 在每个 chunk 执行后把 imagined future frame 换成 ground-truth observation，避免 visual drift。再加 asynchronous execution + DiT caching + quantization + CUDA-graph，把 inference 推到 7Hz——但仍落后于 non-generative VLA 的 50Hz。
- Cosmos Policy (https://arxiv.org/abs/2601.16163) 把 proprioception、action chunk、future state、value function 都 encode 成 latent frame interleaved 进 denoising sequence，**一个 checkpoint 同时是 policy + world model + value function**。

- *Implicit*：future state 不直接 generate，而是通过 auxiliary future token 的 internal representation align 到 teacher encoder 的 embedding。FLARE (https://arxiv.org/abs/2505.15659) 在 action token 序列后 append learnable future token，propagate 过 DiT，在 internal layer 用 MLP project 后监督 match frozen teacher encoder 的 visual embedding。FRAPPE (https://arxiv.org/abs/2602.17259) 把这做成 post-training recipe，在 frozen RDT-1B (https://arxiv.org/abs/2410.07864) 上加 Mixture-of-Prefix-and-LoRA，多个 alignment expert 各对齐一种 teacher representation。

**Multi-Stream (Multiple DiTs)**：world 和 action 在 structurally independent 的 branch，通过显式 coupling 交互。再分三种 coupling 接口：

- *Cross-Attention Coupled*：CoVAR (https://arxiv.org/abs/2512.16023) 用 Bridge Attention 模块 concat video+action feature 做 joint attention 再 split。LDA-1B (https://arxiv.org/abs/2602.12215) 用 shared MM-DiT attention layer + modality-specific projection，world 在 DINO latent space 而非 video VAE latent。DUST (https://arxiv.org/abs/2510.27607) 让 world 和 action 用 **independent noise timestep**，asynchronous schedule sampling。
- *Mixture-of-Transformers*：LingBot-VA (https://arxiv.org/abs/2601.21998) interleaved video latent + action token 进 autoregressive sequence，KV-cache 跨 chunk 累积。Motus (https://arxiv.org/abs/2512.13030) 加第三个 semantic understanding expert，Tri-modal Joint Attention，UniDifuser-style scheduler 给每个 modality 独立 noise level。
- *Hidden-State Coupling*：world branch 产生 intermediate representation 传给 action branch。DiT4DiT (https://arxiv.org/abs/2603.10448) 用 hook operator 在选定 feature-extraction timestep intercept hidden activation，传给 Action DiT。Fast-WAM (https://arxiv.org/abs/2603.16666) 训练时保留 video branch，inference 时完全移除——只用 Video DiT 单次 forward 的 latent world feature condition action denoising，**inference cost 消除**。这是个很 pragmatic 的设计：训练时享受 world modeling supervision，inference 时不需要实际 generate video。
- *Shared Representation*：UVA (https://arxiv.org/abs/2503.00200) 把 historical observation + action chunk + masked future token encode 进 shared Transformer，再由两个 lightweight diffusion head 分别 decode video 和 action。Masked training scheme 让同一个 backbone 能切换成 policy / video / forward / inverse dynamics / planner。

---

## 3. 数据生态：四源混合

这是 paper 里我觉得很有 insight 的一节。WAM 相对于 VLA 的一个根本优势是 **数据消化能力**：VLA 严格需要 $(o_t, a_t)$ pair，而 WAM 因为有 world modeling objective，可以吃 action-free 的 $(o_t, o_{t+1})$ sequence。

paper 把数据分四类：

### 3.1 Robot-Centric Teleoperation
最贵但最 precise。从 QT-Opt (https://arxiv.org/abs/1806.10293)、BridgeData (https://arxiv.org/abs/2109.13396) 演化到 Open-X Embodiment (https://arxiv.org/abs/2310.08864) 的 1M+ trajectory 跨 22 robot，再到 DROID (https://arxiv.org/abs/2403.12945)、AgiBot World (https://arxiv.org/abs/2503.06669)。最新的 UnifoLM-WBT 用 LeRobot framework 收了 1.89M trajectory 的 humanoid data。

### 3.2 UMI-style Portable Human Demo
UMI (https://arxiv.org/abs/2402.10329) 是个里程碑——handheld 3D-printed gripper + wearable camera，让非专家用户在 wild environment 收数据。演化到 RealOmin (1M traj, 3000+ household, https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData)、RDT2 (10,000 hours, https://arxiv.org/abs/2602.03310)。这给 WAM 提供了 "wild diversity + centimeter-level action constraint" 的独特组合。

### 3.3 Simulation
physics engine 本质是个 exact world model。MimicGen (https://arxiv.org/abs/2310.17596)、DexMimicGen (https://arxiv.org/abs/2410.24185)、RoboCasa (https://arxiv.org/abs/2406.02523)、RoboTwin 2.0 (https://arxiv.org/abs/2506.18088)、SynGrasp-1B (10M grasping traj) 等。仿真提供 privileged supervision——perfect depth、exact 6D pose、precise collision boundary——这些在 real-world 是 partial observable 的。

### 3.4 Human & Egocentric Video
这是 scaling 的 ultimate 路径。Ego4D (https://arxiv.org/abs/2110.07058, 3670h)、Ego-Exo4D (https://arxiv.org/abs/2311.18259, 1286h)、DreamDojo-HV (https://arxiv.org/abs/2602.06949, 43827h)、EgoScale (https://arxiv.org/abs/2602.16710, 20854h)、HumanNet (1M hours)。关键演化是从 **passive world dynamics**（无 action 的 video-text pair）到 **active action-conditioned dynamics**（用 3D hand pose 作 universal end-effector，桥接 human video 和 robot policy）。

paper 在 Section 7 提了一个很有意思的 open challenge：**数据 mixture 的 information-theoretic principle**。human video 至少贡献三个层次的知识：(1) low-level physical priors（object permanence、gravity）；(2) mid-level causal dynamics（特定 interaction → 物理结果）；(3) high-level task logic（task-relevant temporal dependency）。如何 disentangle 这三层、如何 design curriculum 从 internet prior 过渡到 precise robot demo——这些都没有 principled 答案。

---

## 4. Evaluation：三个维度

paper 把 world modeling 的 evaluation 分三层，这个划分我觉得很到位：

### 4.1 Visual Fidelity
最基础。包括：
- **PSNR**：$\mathrm{PSNR}(x, y) = 10 \log\left(\frac{\mathrm{MAX}^2}{\mathrm{MSE}(x, y)}\right)$，$x, y$ 是 reference 和 generated frame，MAX 是 pixel 最大值（255 for 8-bit）
- **SSIM**：$\mathrm{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$，$\mu$ 是 mean、$\sigma$ 是 variance、$\sigma_{xy}$ 是 covariance、$C_1, C_2$ 是稳定常数
- **LPIPS**：$\mathrm{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\hat{f}_l(x)_{hw} - \hat{f}_l(y)_{hw}) \|_2^2$，$\hat{f}_l(\cdot)_{hw}$ 是 layer $l$ 在 spatial location $(h,w)$ 的 channel-normalized feature，$w_l$ 是 learned channel-wise weight，$H_l, W_l$ 是 layer $l$ 的 spatial dimension
- **FVD**：$\mathrm{FVD} = \|\mu_r - \mu_g\|_2^2 + \mathrm{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$，$\mu_r, \Sigma_r$ 是 real video 在 pretrained video feature space 的 mean 和 covariance，$\mu_g, \Sigma_g$ 是 generated 的——这是 Fréchet distance，跟 FID 同源

### 4.2 Physical Commonsense
这是 visual fidelity 之上的 layer。代表 benchmark：
- **VideoPhy** (https://arxiv.org/abs/2406.03520)：solid-solid / solid-fluid / fluid-fluid interaction，binary human annotation
- **PhyGenBench** (https://arxiv.org/abs/2410.05363)：PhyGenEval 用 VLM+LLM 自动评 key physical phenomena detection、physics order verification、naturalness
- **VBench-2.0** (https://arxiv.org/abs/2503.21755)：mechanical / thermal / material state change，用 clip-level abnormal-entity detector 查 merging / splitting / appearing / disappearing
- **WorldModelBench** (https://arxiv.org/abs/2502.20694)：五个 binary physical-law check——Newton 第一定律、质量守恒、固体力学、流体力学、impenetrability、gravitation
- **Physics-IQ** (https://arxiv.org/abs/2506.21552 之类)：用 real video 测 future evolution 预测，Spatial IoU / Spatiotemporal IoU / Weighted Spatial IoU / MSE
- **WorldScore** (https://arxiv.org/abs/2504.00983)：motion accuracy / magnitude / smoothness，用 optical flow
- **EWMBench** (https://arxiv.org/abs/2505.09694)：用 EEF trajectory 的 HSD、nDTW、DYN 测 spatial deviation、spatiotemporal alignment、motion dynamics

### 4.3 Action Plausibility
这是 WAM 独有的 evaluation 维度，也是最有意思的。核心 question：**生成的 video 里有没有足够的 action-relevant 信息**？

- **WorldSimBench** (https://arxiv.org/abs/2410.18072)：Implicit Manipulative Evaluation，测 situation-aware video 能不能 translate 成正确 control signal
- **Wow, wo, val!** (https://arxiv.org/abs/2601.04137)：**IDM Turing Test**——用 IDM 从 generated video 推 action，然后 real-world 执行看 success rate。结果发现很多 visually convincing 的 model 在这个 test 下 success rate 几乎为零。这是个很强的 finding：**visual plausibility ≠ action plausibility**

paper 在 Section 7 的 open challenge 里指出：当前 evaluation 严重 decoupled——world modeling 用 pixel metric，action 用 task success——但缺少 **joint metric** 量化 imagined future 和 generated action 的 causal consistency。提出了 Counterfactual Consistency 和 Foresight-Conditioned Success 这种新 metric 方向。

---

## 5. 一些直觉性思考

讲完 paper 的 formal 内容，我加几点自己的 intuition。

### 5.1 Pixel prediction 真的必要吗

paper 在 Section 7 抛了一个很尖锐的问题：explicit pixel-space prediction 是不是 WAM coupling 的必要组件？最近的证据（Fast-WAM、GigaWorld-Policy）显示，**移除 test-time 的 future prediction head 不一定 degrade control performance**——world modeling 的主要价值可能来自 training-time 的 auxiliary gradient，而不是 inference-time 的 explicit generation。

这指向一个 computational efficient 的 paradigm：在 jointly learned latent space 预测 abstract future state representation，跳过 pixel bottleneck。JEPA (https://arxiv.org/abs/2301.08243) 就是这个哲学。但 open question 是：latent-only representation 能不能 maintain generative approach 的 grounding quality？

### 5.2 Latency tax 是个真问题

DreamZero 把 WAM inference 推到 7Hz 已经很拼了（asynchronous execution + DiT caching + quantization + CUDA-graph），但 non-generative VLA 是 50Hz。这个 gap 的根本原因是 **diffusion 的 multi-step denoising**。

paper 提了一个理论问题：**downstream control 到底需要多少 predictive fidelity**？如果 performance gain 在远低于 full diffusion synthesis quality 时就 plateau 了，那目标应该是 **task-adaptive predictive fidelity**——根据 task 的 error tolerance 动态调整 prediction depth 和 resolution。这是个很有意思的研究方向。

### 5.3 Multimodal physical state

当前 WAM 几乎全在 RGB modality 里预测 future。但 contact-rich manipulation 最关键的信息——tactile distribution、contact force、acoustic signature、material compliance——**在 pixel space 几乎不可见**。这是个 systematic blind spot。

OmniVTA (https://arxiv.org/abs/2603.19201)、VT-WM (https://arxiv.org/abs/2602.06001) 开始探索 visuo-tactile world model，AdaWorldPolicy (https://arxiv.org/abs/2602.20057) 加 force predictor 作为第三个 expert。但这还远远不够。未来可能需要 **modality-adaptive prediction**——有 rich sensor stream 时做 multimodal prediction，没有时 gracefully degrade 到 visual-only。

### 5.4 Long-horizon 的 hierarchy 问题

WAM 当前主要在 short-horizon single-context task 上 evaluate。long-horizon 有三个 compounding challenge：world model prediction 的 distributional drift、action error 在没有 corrective replanning 时的 compounding、continuous generative output 表示 full long-horizon trajectory 的 computational intractability。

paper 设想了三条 path：
- **Modular hierarchy**：WAM 做 low-level physical executor，VLM 做 high-level planner decompose mission 成 subgoal
- **Intrinsic hierarchical WAM**：multi-resolution future prediction，coarse-grained 做 strategic planning，fine-grained 做 reactive control
- **Temporal context scaling**：architectural innovation 扩展 WAM 的 "memory"，避免 standard attention 的 quadratic overhead

Act2Goal (https://arxiv.org/abs/2512.23541) 的 Multi-Scale Temporal Hashing 是个有意思的尝试——proximal frame/action 保持 dense，distal frame/action 用 logarithmically increasing spacing sparse 化。

### 5.5 Safety 的 duality

WAM 的 predictive capacity 让它 potentially 比 reactive VLA 更 capable，但 failure mode 也更 consequential——一个 confidently imagine 了错误 physical future 的 model 可能 commit 到一长串 action sequence，real-world consequence 难以 interrupt。

但同样的 predictive capacity 也提供了 safety enforcement 的机会：world prediction 可以在 action 执行前 check physical constraint 或 conservative uncertainty estimate。这指向 **prediction-integrated safety**——把 uncertainty estimate over imagined future 当作 first-class input 给 safety monitor。

---

## 6. 总结性 intuition

如果让我用一句话概括这篇 paper 的核心 message，我会说：**WAM 是 VLA 的 conceptual successor，它把 "predict next action" 升级成 "predict next state AND next action jointly"，从而把 reactive mapping 变成 causal reasoning**。

从 architecture evolution 的角度看，整个 field 正在从 Cascaded（先想后做）向 Joint（边想边做）迁移，而 Joint 内部又从 Autoregressive（sequential token）向 Diffusion-based（parallel denoising）迁移。Diffusion-based 内部再从 Unified Stream（单 trunk）向 Multi-Stream（多 expert + cross-attention / hidden-state / shared representation coupling）演化。

这个演化轨迹跟 LLM 的 "unified token sequence → mixture of experts → modality-specific routing" 有结构上的相似性，但 WAM 的特殊性在于 **action 的物理 grounding**——action 不是 symbol，是 motor command，它的 correctness 由 real-world physics 裁决，不由 next-token likelihood 决定。这也是为什么 action plausibility 这个 evaluation 维度如此关键。

paper 最后呼吁 community 建立 joint evaluation benchmark——不是 separate leaderboard for world modeling 和 action policy，而是一个 framework 量化 imagined future 和 executed action 之间的 causal consistency。我觉得这是这篇 survey 最重要的 contribution 之一：它不只是 catalog 方法，而是 **重新定义了 embodied AI 的 evaluation standard**。

---

## Key References

- Survey homepage: https://openmoss.github.io/Awesome-WAM
- GitHub: https://github.com/OpenMOSS/Awesome-WAM
- UniPi: https://arxiv.org/abs/2302.00111
- VPP: https://arxiv.org/abs/2412.14803
- LAPA: https://openreview.net/forum?id=VYOe2eBQeh
- GR-1: https://openreview.net/forum?id=NxoFmGgWC9
- GR-2: https://arxiv.org/abs/2410.06158
- CoT-VLA: https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models_CVPR_2025_paper.html
- VLA-JEPA: https://arxiv.org/abs/2602.10098
- PAD: https://arxiv.org/abs/2411.18179
- UWM: https://arxiv.org/abs/2504.02792
- DreamZero / π₀.7: https://arxiv.org/abs/2602.15922
- Cosmos Policy: https://arxiv.org/abs/2601.16163
- FLARE: https://arxiv.org/abs/2505.15659
- FRAPPE: https://arxiv.org/abs/2602.17259
- UVA: https://arxiv.org/abs/2503.00200
- Motus: https://arxiv.org/abs/2512.13030
- Fast-WAM: https://arxiv.org/abs/2603.16666
- AVDC: https://arxiv.org/abs/2310.08576
- UMI: https://arxiv.org/abs/2402.10329
- Open-X Embodiment: https://arxiv.org/abs/2310.08864
- DROID: https://arxiv.org/abs/2403.12945
- AgiBot World: https://arxiv.org/abs/2503.06669
- Ego4D: https://arxiv.org/abs/2110.07058
- Ego-Exo4D: https://arxiv.org/abs/2311.18259
- DreamDojo: https://arxiv.org/abs/2602.06949
- Wan2.1: https://arxiv.org/abs/2503.20314
- CogVideoX: https://arxiv.org/abs/2408.06072
- JEPA (LeCun): https://arxiv.org/abs/2301.08243
- VideoPhy: https://arxiv.org/abs/2406.03520
- WorldScore: https://arxiv.org/abs/2504.00983
- Wow, wo, val!: https://arxiv.org/abs/2601.04137
- RDT-1B: https://arxiv.org/abs/2410.07864
- π₀: https://arxiv.org/abs/2410.24164
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246

如果你想 deep dive 某个具体方向（比如 Joint WAM 的 Multi-Stream coupling 设计空间，或者 action plausibility evaluation 的 IDM Turing Test 细节），告诉我，我可以再展开。
