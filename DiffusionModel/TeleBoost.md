---
source_pdf: TeleBoost.pdf
paper_sha256: 0346fa71c1675232941c4d04b086a36c62e57fed9bc3c11c022b6dda17c42a9b
processed_at: '2026-08-12T12:57:38-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# TeleBoost 人话版

Andrej, 我用最直白的方式给你讲讲这篇paper在搞什么。

---

## 一句话概括

这篇paper就是在说：**pretrained video model 不能直接用，post-training 才是决定成败的关键，而 video post-training 的 bottleneck 是 feedback signal 质量问题，不是 generator 能力问题。**

---

## 为什么 video post-training 难

你想象一下 LLM 的 RLHF，token 采样便宜得很，随便 rollout 几万次都行。Video generation 呢？一次 rollout 要跑几十到几百步 diffusion，每步都在 high-res latent space 做 denoising，成本比 LLM 高好几个数量级。

而且 video 的 error 会 compound。frame-3 手坏了，frame-4、5 全跟着崩。frame-level metric 根本看不出来这种 temporal propagation。

T2V 还是 many-to-many mapping，一个 prompt 对应无数 valid output，reward model 在这种 ambiguous region 给的 score noise 很大，直接 optimize 就 overfit noise。

更恶心的是，generator 变强之后 reward model 会 saturate —— 所有 sample 分都接近满分，intra-group variance 趋零，gradient 消失，training 停滞。

这四个问题合起来就是：**你被 feedback signal 的 quality、reliability、granularity 卡住了，而不是被 generator 的 expressiveness 卡住。** TeleBoost 整个 framework 就是围绕这个 insight 设计的。

---

## 三阶段 pipeline

### Stage 1: SFT —— 不为了好看，为了稳定

SFT 的作用不是优化 perceptual quality，是 **shape behavior space**，把 catastrophic failure mode 干掉，给下游 RL 一个 stable starting point。

具体做三件事：

**Instruction & control SFT**：curate instruction-centric data，explicit 描述 temporal structure、camera behavior、compositional constraint。建立 predictable、compositional 的 baseline policy。

**Spatial-Structure-Aware SFT**：大 camera motion 下 3D structure 容易 collapse，background drift、object deform、depth ordering 错乱。这些 error 下游 reward model 基本捕捉不到。解决方法：从 real + sim + generated data 提取 structural assessment signal，做 **loss reweighting** —— distortion 严重的 sample 加大 gradient weight，stable 的降低 weight。类似 focal loss 思路，把 learning capacity 集中到 hard case 上。

**Physics-Aware SFT**：fluid (水、烟、雾) 的 motion 容易违背物理 —— 水往高处流、烟往下沉。用 joint real + simulation training，加 auxiliary motion prediction branch 估计 optical flow，features 通过 zero-init module (类似 ControlNet trick) fused 到 RGB decoder。Zero-init 保证训练初始时 auxiliary branch 不 perturb pretrained backbone。

Fig. 12 很直观：baseline model 在 lateral camera motion 下 background 结构 progressively collapse，reward-weighted SFT model 保持 coherent layout 和 depth ordering。

### Stage 2: GRPO Stack —— 核心战场

这部分是 paper 的 heart。以 GRPO 为 backbone，叠加四个 modular refinement。

**GRPO backbone**：给一个 prompt，sample 一组 video，每个得 scalar reward，group 内 normalize 出 advantage：
$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G) + \epsilon}$$

只看 relative ranking，不看 absolute scale，省掉 value critic。Video domain 训一个稳定 critic 几乎不可能，这步设计很关键。

然后加四个 refinement：

**ViPO (Where to learn)**：标准 GRPO 给整个 video 一个 scalar advantage，但 defect 通常 localized —— frame-30 手坏了，frame-1 背景完美，scalar gradient 会 uniform 抑制整个 video。ViPO 用 frozen visual backbone (DINOv2 / VideoMAE) 提取 spatiotemporal feature map，construct Advantage Map $M \in \mathbb{R}^{T \times H \times W}$，把 scalar advantage "back-project" 到 pixel/latent level。Policy gradient 在每个 spatiotemporal position 按 local advantage weight 加权。相当于给 gradient 加 spatial-temporal attention mask。

不用 dense supervision 的原因是 dense reward 难构造且容易 reward hacking。ViPO 用 frozen backbone 的 saliency 作为 free structural prior，不需要额外标注。实验效果显著：Wan2.1 baseline MQ 0.5896 → DanceGRPO 0.8639 → ViPO 1.1515。

**BPGO (What to trust)**：T2V/I2V many-to-many mapping，reward model 在 ambiguous region high variance。BPGO 引入 Bayesian Prior (通常来自 SFT model 的 historical reward distribution) 作为 trust anchor。

两个机制：

*RAS (Reliability-Adaptive Scaling)*：对每个 prompt group，算其 reward distribution 与 prior 的 deviation。Deviation 大 + variance 高 → ambiguous，down-scale optimization weight。Deviation 大 + variance 低 → clear signal，up-weight。

*CRT (Contrastive Reward Transformation)*：计算 advantage 时以 prior score 作 baseline 而非 group mean，stretch high-confidence positive sample 的 advantage，compress ambiguous sample。

实验：I2V task 上 GRPO 实际 collapse 了 (Qwen3-VL-Embedding 0.4513)，BPGO 稳定在 0.6890。说明 ambiguous setting 下 trust allocation 是 critical 的。

**Self-Paced GRPO (When to learn)**：解决 reward saturation。当 generator 变强，static reward model saturate，intra-group variance → 0，advantage → noise，training stagnate。

Self-Paced 把 reward 视为 evolving curriculum，co-evolve with generator。三个 phase：
- Phase 1：Visual Fidelity (basic quality + dynamic fluency)
- Phase 2：Temporal Coherence (basic metric 饱和后 shift 到 temporal consistency)
- Phase 3：Semantic Alignment (structural foundation 稳定后 focus fine-grained text alignment)

Phase transition 由实时监控 reward statistics 决定。这是 Bengio 2009 Curriculum Learning 在 RL setting 的应用，但 curriculum 不是 data 顺序，而是 reward function 本身。

Table 9 ablation 很说明问题：Joint training (固定 multi-reward) Total Score 79.80，Self-Paced 80.22，说明 progressive scheduling 本身带来提升，不只是 reward model 变强。

**Joint Reward (How to balance)**：多个 reward (visual quality, motion, text alignment, safety) gradient 方向可能 conflict。标准方法是 gradient-level 找 conflict-aware update direction (PCGrad / CAGrad 风格)，但大 video model 上 prohibitive。

TeleBoost 的 trick 是把问题重写到 advantage level：
$$\min_{\{c_i\}} \left\| \sum_{i=1}^N c_i A_i \right\|^2 \quad \text{s.t.} \sum_i c_i = 1, c_i \geq 0$$

在 advantage space 解最小范数问题，避免 backprop 所有 reward gradient，极大降 memory 和 compute。

### Stage 3: DPO —— "Last Mile" alignment

GRPO 优化 measurable objectives，但有些 holistic quality (cinematic pacing, aesthetic composition, nuanced motion realism) 很难 encode 成 scalar reward。DPO 用 curated preference pair capture 这些 signal。

Loss 就是标准 DPO 形式，关键在 **preference data construction** 三策略：

**Policy-on-Policy Hard Negatives**：从当前 $\pi_{\text{GRPO}}$ 生成 pair，用 VLM critic ensemble + heuristic filter 自动 rank，选 semantic overlap 高但 quality distinct 的 pair。Model 学会 avoid 自己 most probable error。

**Synthetic Temporal Negatives**：高质 real/successful video 作 $y_w$，procedurally 生成 $y_l$：
- Temporal Reversal：反转 frame order (penalize 水往高处流)
- Frame Shuffling：随机 local frame order (enforce causality)
- Stalling/Freezing：重复 frame (penalize static generation under motion prompt)

强制 model 学 strictly causal & temporally coherent density。

**Holistic Human-Aligned Ranking**：人工标注 cinematic quality、style，focus "soft" criteria (lighting consistency, narrative logicality)，fine-tune reward scale $\beta$ 防 metric over-optimization 损害 naturalness。

Training protocol 关键细节：reference policy 用 Stage II final checkpoint (不是 SFT model)，dynamic $\beta$ schedule (start high 保稳定 → anneal slightly 允许 frontier exploration)，mixed-resolution (preference gradient 在 480p 2s 上算，structural failure 在这里最 apparent)。

---

## Infrastructure 细节

这部分容易被 academia 忽略，但 production setting 下决定 scalability。

**GRPO 两个 inefficiency**：VLM reward model 部署在独立 GPU pool，与 rollout/actor GPU 分离，stage transition 期间 GPU idle；Joint reward 下 multiple worker 串行，每个 lightweight 无法 saturate GPU compute。

**Dual optimization**：

*Temporal Multiplexing + Node Consolidation*：用 Ray orchestrate monolithic cluster，single GPU node pool 动态 transition across training stage。Rollout 完成后同一硬件立即 trigger reward evaluation，消除 disaggregated pool 间 data movement overhead。

*NVIDIA MPS for Joint Reward*：MPS 允许 logical GPU partitioning，多个 reward worker 同 GPU concurrent run。形式化是 bin-packing + quota allocation 问题，用 lightweight greedy search (guided by per-worker scaling profile) 解。

**Memory-Efficient DPO via Decoupled Gradient Backpropagation**：

标准 DPO 实现 launch 单个 GraphTask，chosen 和 rejected branch 共享 parameter，Autograd 必须保留第一个 branch 的 intermediate gradient 直到第二个 branch backward 完成。Memory complexity $O(L_w + L_l)$。

DPO gradient 可以 decompose：
$$\mathcal{I}(\theta) = -\eta \cdot \nabla_\theta \log \pi_\theta(y_w | x) + \eta \cdot \nabla_\theta \log \pi_\theta(y_l | x)$$

其中 $\eta = \sigma(-\beta \hat{A}) \cdot \beta$ 是 scalar，可以 pre-compute。拆成两个 independent backward pass，每个完成后立即 release intermediate tensor。Memory complexity 降到 $O(\max(L_w, L_l))$，FLOPs 不变。

Intuition：这是把 single autograd graph 拆成两个 sequential graph，牺牲一点 launch overhead 换取大幅 memory 节省。类似 gradient checkpointing 思想，但更精细 —— 不重算 forward，而是拆 backward。

---

## 实验数据解读

**Human Eval (Table 1)** vs Wan2.2-14B I2V，GSB protocol：
- Motion Quality: WinRate 70.72%, Margin 24.90% (最大提升)
- Text Alignment: WinRate 77.39%, Margin 24.13%
- Overall: WinRate 71.18%, Margin 32.71%
- Visual Quality: WinRate 58.71%, Margin 4.35% (baseline 已经强，提升 modest)

Motion 和 Alignment 提升最大，因为这是 GRPO + ViPO + BPGO 直接 optimize 的 target。Visual Quality modest 提升说明没 sacrifice appearance 换 motion —— multi-objective balancing 的效果。

**VBench (Table 6)** ViPO：
- Dynamic Degree 52.77 → 63.89
- Multiple Objects 69.96 → 74.70
- Spatial Relationship 72.94 → 81.44

**Optical Flow (Table 2)** Physics-aware SFT：
- Real-world fluid EPE 0.538
- Simulation fluid EPE 1.541
- Real-world 上比 simulation 好，说明学到 generalizable motion prior，没 overfit synthetic dynamics。

---

## 我的 Intuition

读完这篇 paper 我 build 出来的 intuition：

**Video post-training 是 signal engineering 问题，不是 algorithm engineering 问题。**

整个 TeleBoost 没提任何全新 algorithm，是把已知 technique (GRPO, DPO, curriculum learning, Bayesian prior, gradient surgery) 在 video domain specific constraint 下重新组装。每个 component 解决一个 specific failure mode：

- SFT：shape behavior space，建立 stable reference
- ViPO：where to learn (spatiotemporal credit assignment)
- BPGO：what to trust (reliability under ambiguity)
- Self-Paced GRPO：when to learn (competence-aware curriculum)
- Joint Reward：how to balance (multi-objective trade-off)
- DPO：what to prefer (holistic human judgment)

三个 dimension (Where / What / When) 对应三个 RL refinement，这种 decomposition 很 clean。跟 LLM RLHF 发展路径惊人相似 —— naive PPO → RLHF → DPO → RLAIF → 各种 variance reduction trick。Video domain 因为 rollout cost 和 temporal structure，每步都更难，但 conceptual framework transferable。

---

## 几个值得深挖的 Open Question

1. **ViPO 的 advantage map 怎么 validated？** 用 frozen backbone 的 saliency 作为 unsupervised structural prior，如果 backbone 对某类 artifact 不 sensitive (e.g., 微妙 color shift)，ViPO 会 miss。能否 learn task-specific advantage map extractor？

2. **BPGO 的 prior selection 对 final performance 多敏感？** SFT model 本身有 bias 会 propagate。能否用 ensemble of priors 做 robustification？

3. **Self-Paced GRPO 的 phase transition 检测 robust 吗？** Reward model 突然 degrade (distribution shift) 会不会 misfire transition？

4. **DPO synthetic temporal negative 会 over-constrain 吗？** Frame shuffling penalize 所有 non-causal motion，但 reverse motion、time-lapse 这种 creative effect 可能 valid。能否 design prompt-conditional synthetic negative？

5. **Three-stage pipeline 真的 optimal 吗？** 能否 unified SFT + RL + DPO into single joint optimization？Joint optimization 通过更好 regularization 能否达到 same effect？

6. **整个 pipeline 的 compute cost 和 scaling behavior？** Paper 没给 total training cost / FLOPs / wall-clock time。Video post-training 的 scaling law 还是 open problem。

7. **Feedback 能否更 diagnostic？** Paper conclusion 提到 "feedback should become more diagnostic rather than merely scalar"。如果 reward model 不只给 scalar，还给 failure category + uncertainty + suggested fix，能否 build closed-loop self-debugging 系统？

这些问题跟 LLM domain 当前 frontier (process reward model, test-time compute, self-correction) 高度 parallel。Video domain 因 temporal structure 和 high rollout cost，每个都更难，但 conceptual framework 相通。

---

## 相关工作链接

核心 paper：
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [DanceGRPO](https://arxiv.org/abs/2505.07818)
- [DPO](https://arxiv.org/abs/2305.18290)
- [ViPO](https://arxiv.org/abs/2511.18719)
- [BPGO](https://arxiv.org/abs/2511.18919)
- [Self-Paced GRPO](https://arxiv.org/abs/2511.19356)
- [Discriminator-Free DPO](https://arxiv.org/abs/2504.08542)

Backbone & Benchmark：
- [Wan video model](https://arxiv.org/abs/2503.20314)
- [VBench](https://arxiv.org/abs/2311.17982)
- [VideoAlign](https://arxiv.org/abs/2501.13918)

Foundation：
- [DDPM](https://arxiv.org/abs/2006.11239)
- [DiT](https://arxiv.org/abs/2212.09748)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [ControlNet](https://arxiv.org/abs/2302.05543)

Infrastructure：
- [Ray](https://www.ray.io/)
- [NVIDIA MPS](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)

LLM RLHF 对照：
- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Curriculum Learning (Bengio 2009)](https://dl.acm.org/doi/10.1145/1553374.1553380)
- [Gradient Surgery](https://papers.nips.cc/paper/2020/hash/3fe78a8a02f4f6d5b46c14b9d4b8c866-Abstract.html)

---

# TeleBoost 深度解析:Video Post-Training 的系统化框架

Andrej, 这篇 paper 本质上是在回答一个工程化的问题:**如何把一个 pretrained video diffusion model 转化成 production-ready 的 generator**。作者把整个 post-training 过程拆成三个 staged optimization,每一步解决一类特定的 failure mode。下面我从 first principles 出发,逐层拆解。

---

## 1. 为什么 Video Post-Training 与 LLM/Image 本质上不同

作者在 §1.1 提出了四个核心困难,这四个困难决定了整个 framework 的设计:

### 1.1 High Rollout Cost
单次 video rollout 需要几十到几百个 diffusion steps,且每个 step 都在高分辨率 latent space 上做 denoising。对比 LLM 的 token sampling,video rollout 的 cost 要高出几个数量级。这意味着 naive PPO-style trial-and-error 完全不可行 —— 你无法 afford 大量 exploratory rollouts。

Intuition: 这就像你有一个极其昂贵的 simulator,每次执行都要几百美元,所以你必须 batch-sample (group rollout) 然后从相对比较中榨取信息,而不是依赖绝对 value estimation。

### 1.2 Temporal Compounding Errors
Video failures 不是 i.i.d. 的 frame-level errors。一个 frame-3 的 hand deformation 会 propagate 到 frame-4, 5...,导致后面的 motion 全部 collapse。Frame-level metrics (像 FID per frame) 完全捕捉不到这种 compounding effect。

这是 video domain 独有的 "temporal credit assignment" 问题 —— 你必须知道一个 error 是在何时 (which timestep) 发生的,以及它会 propagate 到哪里。

### 1.3 Many-to-Many Ambiguous Supervision
T2V 是 many-to-many mapping: 一个 prompt "a cat running" 可以对应无数 valid renderings (不同的 cat、不同的背景、不同的 camera angle)。一个 reward model 给出的 score 在这种 ambiguous region 会有 high variance,直接优化会 overfit evaluator noise。

### 1.4 Evaluator Brittleness
CLIP-based scorer、VLM-based alignment model 都对 decoding choices (fps、resolution、sampling schedule) 极其敏感。当 generator 变强,reward model 会 saturate (所有 sample 都接近 ceiling score),intra-group variance 趋零,gradient signal 消失。

**这四个困难合起来告诉我们**: video post-training 的 bottleneck 不是 generator 的 expressiveness,而是 feedback signal 的 quality、reliability 和 granularity。整个 TeleBoost 就是围绕这个 insight 设计的。

Reference: [DanceGRPO (Xue et al., 2025)](https://arxiv.org/abs/2505.07818) 首次系统化地把 GRPO 应用到 visual generation,TeleBoost 在此基础上做了大量 extension。

---

## 2. Stage I: Supervised Fine-Tuning (SFT) —— Policy Shaping 而非 Quality Optimization

这一步的核心 insight 是: **SFT 的作用是 shape behavior space,建立稳定的 reference policy,让 downstream RL 有一个 stable 的 starting point**。很多 RLHF 失败的根本原因是 SFT policy 不稳定,导致 RL 阶段的 rollout variance 爆炸。

SFT 分三个互补的 supervision:

### 2.1 Instruction & Control-Oriented SFT
标准 text-video supervision 不够,作者 curate 了 instruction-centric data,explicitly 描述:
- temporal structure & event ordering ("先 A 后 B")
- camera behavior (rotation, translation, dolly)
- compositional & editing-oriented instructions

这些 sample 虽然占 minority,但它们 establish 了一个 predictable、compositional 的 baseline policy。

### 2.2 Spatial-Structure-Aware SFT (3D Consistency)
**问题**: 大 camera motion 下,3D structure 会 collapse —— background drift、object deform、depth ordering 错乱。这些 error 在下游 RL 中很难被 reward model 捕捉 (因为 evaluators 通常 focus on appearance 而非 geometry)。

**方法**: Loss reweighting。从 real videos + simulator data + model-generated sequences 中提取一个 structural assessment signal $s_{\text{struct}}$。对于 distortion 严重的 sample,加大 gradient weight;对于 stable sample,降低 weight。

Intuition: 这就像 focal loss 的思想 —— 把 learning capacity 集中到 hard、structurally-failing 的 sample 上,而不是 uniform 分配。

数学上,可以理解为 modified diffusion loss:
$$\mathcal{L}_{\text{SFT-struct}} = \mathbb{E}_{x, \epsilon, t} \left[ w(x, t) \cdot \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$
其中 $w(x, t)$ 是 sample-level & timestep-level 的 reweighting,由 structural signal 决定。$x$ 是 video sample,$\epsilon$ 是 noise,$\epsilon_\theta$ 是 model prediction,$x_t$ 是 noised latent at timestep $t$。

Fig. 12 的可视化很说明问题: baseline model 在 lateral camera motion 下,background 结构 progressively collapse;reward-weighted SFT model 保持 coherent global layout 和 depth ordering。

### 2.3 Physics-Aware SFT (Fluid Dynamics)
**问题**: Fluids (水、烟、雾) 的 motion 很容易违背物理 —— 水往高处流、烟往下沉。这些 error 也很难被 downstream evaluator 惩罚。

**方法**: Joint real-and-simulated training + auxiliary motion branch。

具体架构:
1. 标准 RGB decoding pathway (frozen backbone)
2. Auxiliary motion prediction branch: 估计 inter-frame optical flow
3. Ground-truth optical flow 用 RAFT ([Teed & Deng, 2020](https://arxiv.org/abs/2003.12059)) 从 training data 提取
4. Motion branch 的 intermediate features 通过 zero-initialized module (类似 LoRA / ControlNet 的 zero-conv) fused 到 RGB decoder

为什么 zero-init? 保证训练初始时 auxiliary branch 不 perturb pretrained backbone,让 gradient 从 zero 开始平滑注入 —— 这是 [ControlNet (Zhang et al., 2023)](https://arxiv.org/abs/2302.05543) 的经典 trick。

Table 2 的实验: optical flow EPE 在 real-world fluid 上 0.538,在 simulation fluid 上 1.541,说明 model 学到了 generalizable motion pattern,不只是 overfit synthetic dynamics。

### 2.4 SFT 作为 RL Foundation 的核心思想
作者反复强调: SFT 不是为了优化 perceptual quality,而是为了:
1. Eliminate catastrophic failure modes (geometry collapse, physics violation)
2. Build a stable, interpretable reference policy $\pi_{\text{ref}}$

这样 downstream GRPO 可以 focus on relative quality 和 preference,而不是花 capacity 修正 fundamental structural deficiency。

Intuition: 这跟 LLM RLHF 的教训一致 —— 如果 SFT model 还在 hallucinate basic facts,RL 阶段会陷入 correcting basic errors 而非 learning nuanced preference。

---

## 3. Stage II: GRPO Optimization Stack —— 核心

这是整个 framework 的 heart。作者把它构建成一个 modular stack,以 GRPO 为 backbone,上面叠加四个 refinement:

### 3.1 GRPO Backbone: Critic-Free Group-Relative Advantage

GRPO ([Shao et al., 2024, DeepSeekMath](https://arxiv.org/abs/2402.03300)) 的核心是消除 value critic。在 video domain 这尤其重要,因为训一个能在 high-dimensional latent space 上稳定估计 value 的 critic 几乎不可能。

Protocol:
1. 给 prompt $c$,policy $\pi_\theta$ 采样一组 video $\{v_1, \ldots, v_G\}$
2. 每个 video 得 scalar reward $r_i$
3. Group-relative advantage:
$$A_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\}) + \epsilon}$$
变量: $r_i$ 是第 $i$ 个 video 的 raw reward,$\text{mean}$ 和 $\text{std}$ 是 group 内统计量,$\epsilon$ 是 numerical stability 的小常数 (通常 $10^{-8}$)。

为什么这个 formulation 关键?
- **Scale invariance**: reward model 的 absolute scale 无关,只看 relative ranking。这解决了不同 reward model scale 不一致的问题。
- **Variance reduction**: 通过 group 内 normalization,reward 的 systemic bias 被 subtract 掉。
- **No critic**: 省掉一个高维 value network,极大降低 training instability。

Policy update 加上 KL constraint (typically per-token KL) 防止偏离 $\pi_{\text{ref}}$:
$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[ A_i \cdot \log \pi_\theta(v_i | c) \right] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

### 3.2 ViPO: Spatiotemporal Credit Assignment —— "Where to Learn"

**问题**: 标准 GRPO 给一个 video 一个 scalar advantage $A_i$。但 video 的 defect 通常 localized —— frame-30 的 hand collapse vs frame-1 的 perfect background。Scalar gradient 会 uniform 抑制整个 video,包括 good parts。

**方法**: ViPO ([Ni et al., 2025](https://arxiv.org/abs/2511.18719)) 引入 Perceptual Structuring Module (PSM),把 scalar advantage "back-project" 到 spatiotemporal map。

PSM 用 frozen pretrained visual backbone (DINOv2 或 VideoMAE) 提取 feature maps $F \in \mathbb{R}^{T \times H \times W \times d}$。通过 feature saliency 或 alignment analysis,construct Advantage Map:
$$M_i \in \mathbb{R}^{T \times H \times W}$$
其中 $T$ 是 temporal dimension (frame 数),$H, W$ 是 spatial dimensions,$M_i^{(t,h,w)}$ 表示 frame $t$ 位置 $(h,w)$ 的 local advantage weight。

Modified loss:
$$\mathcal{L}_{\text{ViPO}} = \mathbb{E}\left[ \sum_{t,h,w} M_i^{(t,h,w)} \cdot A_i \cdot \log \pi_\theta(v_i) \right]$$
变量解读:
- $M_i^{(t,h,w)}$: spatiotemporal position $(t,h,w)$ 的 perceptual saliency weight (从 frozen visual backbone 提取)
- $A_i$: scalar group-relative advantage (from GRPO)
- $\log \pi_\theta(v_i)$: log-likelihood of video $v_i$ under policy $\pi_\theta$

Intuition: 这相当于给 policy gradient 加了一个 spatial-temporal attention mask。Model 被引导 focus 在 visually critical / failure-prone 的区域,而不是 uniform 更新。

为什么不用 dense supervision (e.g., pixel-level reward)? 因为 dense reward 极难构造,且容易引入 reward hacking。ViPO 的 trick 是: 用 frozen visual backbone 的 saliency 作为 free 的 structural prior,不需要额外标注。

Table 7 实验: Wan2.1 baseline MQ 0.5896 → DanceGRPO 0.8639 → ViPO 1.1515。VBench Dynamic Degree 从 52.77 提升到 63.89,Multiple Objects 从 69.96 提升到 74.70。这种 localized credit assignment 的效果非常显著。

### 3.3 BPGO: Bayesian Prior-Guided Optimization —— "What to Trust"

**问题**: T2V/I2V 是 many-to-many mapping,reward model 在 ambiguous region 给出 high variance score。如果 uniform trust 所有 group 的 feedback,会 overfit noise。

**方法**: BPGO ([Liu et al., 2025b](https://arxiv.org/abs/2511.18919)) 引入 Bayesian Prior 作为 "trust anchor"。Prior 通常来自 SFT model 的 historical reward distribution。

两个机制:

#### 3.3.1 Inter-group Trust Allocation (RAS - Reliability-Adaptive Scaling)
对每个 prompt group,计算其 reward distribution 与 prior 的 deviation。如果 deviation 大且 variance 高 → 标记为 "ambiguous",down-scale optimization weight。如果 deviation 小且 variance 低 → up-weight。

形式化: 给 group $g$ 的 trust weight
$$\alpha_g = f\left( \frac{|\mu_g - \mu_{\text{prior}}|}{\sigma_g + \epsilon} \right)$$
其中 $\mu_g$ 是 group $g$ 的 mean reward,$\mu_{\text{prior}}$ 是 prior mean,$\sigma_g$ 是 group variance,$f$ 是 monotonically increasing function (具体形式见原文)。

Intuition: deviation 大 + variance 高 → 这个 group 的 reward 不可靠,可能是 prompt 太 ambiguous 或 evaluator 在该区域 noisy。Deviation 大 + variance 低 → 这个 group 有 clear preference signal,要充分利用。

#### 3.3.2 Intra-group Prior-Anchored Renormalization (CRT - Contrastive Reward Transformation)
计算 advantage 时,以 prior score 作为 baseline 而不是 group mean:
$$A_i^{\text{CRT}} = \frac{r_i - r_{\text{prior}}}{\sigma_{\text{group}} + \epsilon}$$
其中 $r_{\text{prior}}$ 是 prior 在该 prompt 下的 reference score。

对 high-confidence positive sample (significantly beating prior): stretch advantage。对 ambiguous sample (close to prior): compress。这是 adaptive scaling 的 contrastive 版本。

Table 3 实验: T2V task 上,BPGO VideoAlign-TA 1.1193 vs GRPO 0.8984,提升 24.6%。I2V task 上,BPGO Qwen3-VL-Embedding 0.6890 vs GRPO 0.4513 —— GRPO 实际上 collapse 了,说明在 ambiguous I2V setting 下,trust allocation 是 critical 的。

Table 4 的 VBench 详细 breakdown: BPGO (RAS+CRT) 在 9 个 metric 中 6 个 best,包括 Object Multiple、Human Action、Color 等关键 dimension。

### 3.4 Self-Paced GRPO: Competence-Aware Curriculum —— "When to Learn"

**问题**: Reward Saturation。早期 reward model (e.g., 简单 quality scorer) 给出有效 gradient。但当 generator 变强,score 全部聚到 ceiling,intra-group variance → 0,advantage → noise,training stagnate。

**方法**: Self-Paced GRPO ([Li et al., 2025b](https://arxiv.org/abs/2511.19356)) 把 reward 视为 evolving curriculum,co-evolve with generator。

三个 phase:
- **Phase 1 (Visual Fidelity)**: Focus on basic visual quality & dynamic fluency。Reward 优先惩罚 structural collapse。
- **Phase 2 (Temporal Coherence)**: 当 basic metric 饱和 (variance → 0),shift weight 到 temporal consistency & complex motion logic。
- **Phase 3 (Semantic Alignment)**: 最后阶段 focus on fine-grained text alignment & aesthetic details。

Phase transition 由实时监控的 reward statistics 决定: 当当前 phase 的 reward distribution sparsity 高、intra-group discriminability 低时,trigger transition。

Intuition: 这是 [Bengio et al., 2009 Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380) 在 RL setting 的应用,但 curriculum 不是 data 顺序,而是 reward function 本身。

Table 8 实验: Wan2.1-14B + Self-Paced GRPO (Qwen2.5VL-72B reward) Total Score 82.09 vs Wan14B baseline 81.46。在 Spatial Relationship 上 79.06 vs 74.97,Human Action 80.00 vs 77.00。

Table 9 ablation: Joint training (固定 multi-reward) Total Score 79.80 vs Self-Paced 80.22,说明 progressive scheduling 本身带来提升,不仅仅是 reward model 变强。

### 3.5 Joint Reward: Multi-Objective Balancing at Advantage Level

**问题**: 多个 reward (visual quality, motion, text alignment, safety) 之间 gradient 方向可能 conflict。简单 sum 会导致 imbalanced progress。

**标准方法 (gradient-level)**: PCGrad / CAGrad 风格,找 conflict-aware update direction:
$$\min_{\{c_i\}_{i=1}^N} \left\| \sum_{i=1}^N c_i \nabla_\theta \mathcal{L}_i(\theta) \right\|^2 \quad \text{s.t.} \sum_i c_i = 1, c_i \geq 0$$
变量: $c_i$ 是第 $i$ 个 reward 的 weight,$\nabla_\theta \mathcal{L}_i(\theta)$ 是第 $i$ 个 reward 对参数 $\theta$ 的 gradient,$N$ 是 reward 数量。

问题: 显式计算所有 reward 的 gradient 在大 video model 上 prohibitive。

**TeleBoost 的方法**: 重写为 advantage level:
$$\min_{\{c_i\}_{i=1}^N} \left\| \sum_{i=1}^N c_i A_i \right\|^2 \quad \text{s.t.} \sum_i c_i = 1, c_i \geq 0$$
其中 $A_i$ 是第 $i$ 个 reward model 诱导的 advantage。

最终 loss:
$$\mathcal{L} = \sum_i c_i \mathcal{L}_i$$

Intuition: 在 advantage space 解最小范数问题,避免 backprop 所有 reward gradient,极大降低 memory 和 compute overhead。这是 multi-objective optimization ([Yu et al., 2020 Gradient Surgery](https://papers.nips.cc/paper/2020/hash/3fe78a8a02f4f6d5b46c14b9d4b8c866-Abstract.html)) 在 video post-training 上的 scalable 改造。

---

## 4. Stage III: Direct Preference Optimization (DPO) —— "Last Mile" Alignment

GRPO 优化 measurable objectives (motion, alignment),但有些 holistic quality (cinematic pacing, aesthetic composition, nuanced motion realism) 很难 encode 成 scalar reward。DPO ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) 用 curated preference pair 来 capture 这些 signal。

### 4.1 Diffusion DPO Loss
给定 triplet $\mathcal{D} = \{x, y_w, y_l\}$ (prompt $x$, preferred video $y_w$, dispreferred video $y_l$):
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]$$

变量解读:
- $\pi_\theta$: current policy ( trainable)
- $\pi_{\text{ref}}$: frozen reference policy (Stage II checkpoint)
- $\beta$: KL strength (控制 deviation from reference)
- $\sigma$: sigmoid function
- $\log \pi_\theta(y_w | x) / \pi_{\text{ref}}(y_w | x)$: log-likelihood ratio for preferred sample
- 类似 term for $y_l$

关键设计: 与 standard SFT (maximize $\log \pi_\theta(y_w | x)$) 不同,DPO 同时 suppress $\log \pi_\theta(y_l | x)$。这强制 model "unlearn" specific failure mode (zombie motion, static frame, warped geometry),而 SFT alone 无法 easily penalize 这些。

### 4.2 Preference Data Construction —— 三种策略

DPO 的效果由 data quality 决定,而非 loss formulation。作者用三种策略:

#### Strategy 1: Policy-on-Policy Hard Negatives
从当前 $\pi_{\text{GRPO}}$ 生成 pair。用 VLM-based critic ensemble (Video-LLaVA, Gemini-Vision) + heuristic filter (optical flow magnitude, aesthetic scorer) 自动 rank。选 semantic overlap 高但 visual quality distinct 的 pair 作为 high-information training signal。

这是 "self-correction" loop: model 学会 avoid 自己 most probable 的 error。

#### Strategy 2: Synthetic Temporal Negatives (Discriminator-Free)
解决 high-quality pairwise video data 稀缺问题。取 high-quality real/successful video 作为 $y_w$,procedurally generate $y_l$:
- **Temporal Reversal**: 反转 frame order (惩罚 water flowing up, smoke condensing)
- **Frame Shuffling**: 随机 local frame order (enforce causality & smoothness)
- **Stalling/Freezing**: 重复 frame (惩罚 static generation under motion prompt)

这强制 model 学 strictly causal & temporally coherent probability density。参考 [Cheng et al., 2025 Discriminator-Free DPO](https://arxiv.org/abs/2504.08542)。

#### Strategy 3: Holistic Human-Aligned Ranking
最高 tier: 人工标注 cinematic quality、style。Focus on "soft" criteria (lighting consistency, narrative logicality, emotional tone)。用这些 pair fine-tune reward scale $\beta$,防止 metric over-optimization 损害 visual naturalness。

### 4.3 Training Protocol 细节
- **Reference policy**: 用 Stage II final checkpoint (不是 SFT model) 作为 $\pi_{\text{ref}}$。这样 DPO refine RL-optimized frontier,而非 revert to mean SFT behavior。
- **Dynamic $\beta$ schedule**: Start high (preserve stability) → anneal slightly (allow preference frontier exploration)。Low $\beta$ 导致 visual degeneration,high $\beta$ 阻止 artifact correction。
- **Mixed-resolution**: Preference gradient 在 low resolution / short duration (480p, 2s) 上计算 (structural failure 在这里最 apparent),visual encoder feature cached/frozen 降 memory。

---

## 5. Reward Modeling: Heterogeneous Signal Integration

作者强调 reward modeling 是 signal integration 挑战,而非单一 scoring function training。三个 representational modality:

1. **Semantic Alignment (VLM-Driven)**: Video-Language Model 作为 semantic judge。比 CLIP score 更 fine-grained —— verify object count, spatial relation, action sequence。Challenge: 从 large VLM 提取 calibrated probability 作为 dense gradient。

2. **Temporal Dynamics & Physics**: 专门 motion estimator。Penalize non-causal transition, static frame repetition, unnatural warping。Anti-"slide-show" regularizer。

3. **Perceptual & Safety**: Low-latency critic for artifact detection (blur, noise, compression) + safety compliance。作为 "gating" signal,impose strict boundary on feasible policy space,防止 adversarial noise pattern 去 hack VLM score。

Reference: [RewardBench (Lambert et al., 2025)](https://aclanthology.org/2025.findings-naacl.112/) 是 reward model 评估的标准 benchmark,[Gao et al., 2023 Scaling Laws for Reward Overoptimization](https://arxiv.org/abs/2210.10760) 解释了为什么 reward model 会 over-optimize。

---

## 6. Infrastructure —— 工程细节决定 scalability

这部分是 production-level 的关键,容易被 academia 忽略。

### 6.1 GRPO 的两个 inefficiency
1. **Disaggregated reward GPU**: VLM reward model 部署在独立 GPU pool,与 rollout/actor GPU 分离。Strict sequential dependency (rollout → reward → actor) 导致 stage transition 期间 GPU idle。
2. **Serial reward worker**: Joint reward 场景下,multiple reward worker 串行执行,每个 worker 是 lightweight,无法 saturate GPU compute。

### 6.2 Dual Optimization Strategy

#### 6.2.1 Temporal Multiplexing + Node Consolidation
用 [Ray](https://www.ray.io/) orchestrate monolithic physical cluster。Single GPU node pool 动态 transition across training stage:
- Rollout phase 完成后,同一硬件立即 trigger VLM inference + joint reward evaluation
- 消除 disaggregated pool 之间的 data movement 和 coordination overhead

#### 6.2.2 NVIDIA MPS (Multi-Process Service) for Joint Reward
MPS 允许 logical GPU partitioning,多个 reward worker 在同一 GPU 上 concurrent run。

形式化: reward worker set $W$,partition $P = \{p_1, \ldots, p_G\}$。同 group worker concurrent,group 间 sequential。Worker $w$ 在 quota $q_w$ 下执行时间 $C_w(q_w)$。Group $g$ 执行时间:
$$T_g = \max_{w \in p_g} C_w(q_w), \quad g = 1, \ldots, G$$
变量: $p_g$ 是第 $g$ 个 group 的 worker 集合,$C_w(q_w)$ 是 worker $w$ 在 quota $q_w$ 下的 execution time,$\max$ 因为 group 内 concurrent 执行受 slowest worker 限制。

Total reward makespan:
$$T_{\text{reward}} = \sum_{g=1}^G T_g = \sum_{g=1}^G \max_{w \in p_g} C_w(q_w)$$

Quota allocation constraint:
$$\sum_{w \in p_g} q_w \leq 1, \quad \forall g = 1, \ldots, G$$
变量: $q_w \in (0, 1]$ 是 worker $w$ 的 compute quota (通过 MPS set default active thread percentage 设置),约束保证同 GPU 上 concurrent worker 总 compute 不超过 GPU capacity。

用 lightweight greedy search (guided by per-worker scaling profile) 联合决定 grouping 和 quota split。

修改 RayWorkerGroup 允许 overlapping GPU assignment,移除 default GPU exclusivity。参考 [NVIDIA MPS Documentation](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)。

### 6.3 Memory-Efficient DPO via Decoupled Gradient Backpropagation

#### 6.3.1 Root Cause: Shared Parameter Dependency
标准 DPO 实现 launch 单个 GraphTask,chosen 和 rejected branch 共享 parameter $W \in \theta$。Autograd 必须保留第一个 branch 的 intermediate gradient tensor 直到第二个 branch backward 完成。Memory complexity:
$$O(L_w + L_l)$$
其中 $L_w$ 是 chosen branch 的 intermediate tensor 数量,$L_l$ 是 rejected branch 的。

#### 6.3.2 Decoupled Gradient Backpropagation (DGB)
DPO gradient (Eq. 11):
$$\mathcal{I}(\theta) = \nabla_\theta \mathcal{L}_{\text{DPO}} = -\sigma(-\beta \hat{A}) \cdot \beta \cdot \left( \nabla_\theta \log \pi_\theta(y_w | x) - \nabla_\theta \log \pi_\theta(y_l | x) \right)$$
变量:
- $\hat{A} = \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}$ 是 preference margin
- $\sigma(-\beta \hat{A}) \cdot \beta = \eta$ 是 scalar weighting coefficient

关键 insight: $\eta$ 是 scalar,可以 non-gradient context 预计算。Decompose:
$$\mathcal{I}(\theta) = -\eta \cdot \nabla_\theta \log \pi_\theta(y_w | x) + \eta \cdot \nabla_\theta \log \pi_\theta(y_l | x)$$

拆成两个 independent backward pass:
$$\mathcal{I}_w(\theta) = -\eta \cdot \nabla_\theta \log \pi_\theta(y_w | x)$$
$$\mathcal{I}_l(\theta) = \eta \cdot \nabla_\theta \log \pi_\theta(y_l | x)$$

执行 $\mathcal{I}_w.\text{backward}()$ 后立即 release intermediate tensor,再执行 $\mathcal{I}_l.\text{backward}()$。Memory complexity:
$$O(\max(L_w, L_l))$$
两 advantage:
- **Peak Memory Reduction**: $O(L_w + L_l) \to O(\max(L_w, L_l))$,防 OOM 不需额外 GPU
- **Zero Computational Overhead**: FLOPs 与 standard DPO 完全一致

Intuition: 这是把 single autograd graph 拆成两个 sequential graph,牺牲一点 launch overhead 换取大幅 memory 节省。PyTorch 的 autograd 默认保留整个 graph 的 intermediate 直到 backward 完成,DGB 用 manual decomposition 绕开这个限制。类似 [Gradient Checkpointing](https://github.com/cybertronai/gradient-checkpointing) 的思想,但更精细 —— 不重算 forward,而是拆 backward。

---

## 7. Experiments 关键数据解读

### 7.1 Human Evaluation (Table 1)
vs Wan2.2-14B I2V,用 GSB (Good-Same-Bad) protocol:
- Motion Quality: WinRate 70.72%, Margin 24.90% (最大提升)
- Text Alignment: WinRate 77.39%, Margin 24.13%
- Overall: WinRate 71.18%, Margin 32.71%
- Visual Quality: WinRate 58.71%, Margin 4.35% (相对 modest,因为 baseline 已经 strong)
- Preservation: WinRate 63.28%, Margin 8.15%

GSB metric 定义:
- $\text{WinRate} = G / (G + B)$ (ignore tie)
- $\text{Preference} = (G + 0.5S) / (G + S + B)$ (half credit to Same)
- $\text{Margin} = (G - B) / (G + S + B)$ (signed preference gap)

为什么 Motion Quality 和 Text Alignment 提升最大? 这两个 dimension 正是 GRPO + ViPO + BPGO 直接 optimize 的 target。Visual Quality modest 提升 说明 framework 没有 sacrifice appearance 换 motion —— 这是 multi-objective balancing 的效果。

### 7.2 VBench Detailed Metrics (Table 4, 6, 8)
- ViPO: Dynamic Degree 52.77 → 63.89,Multiple Objects 69.96 → 74.70,Spatial Relationship 72.94 → 81.44
- BPGO: 6/9 metric best,包括 Object Multiple 0.6899,Human Action 0.6500
- Self-Paced GRPO: 14B model Total Score 82.09 vs baseline 81.46

### 7.3 Optical Flow Evaluation (Table 2)
Physics-aware SFT:
- Real-world fluid: EPE 0.538, >1px 14.7%, >3px 0.0%, F1-all 4.680
- Simulation fluid: EPE 1.541, >1px 21.8%, >3px 10.0%, F1-all 10.040

EPE (End-Point Error) = average $\| \mathbf{u}_{\text{pred}} - \mathbf{u}_{\text{gt}} \|_2$,其中 $\mathbf{u}$ 是 optical flow vector。$>1$px 和 $>3$px 是 error threshold 的 pixel 比例。F1-all 是 overall flow inconsistency 的综合 metric。

Real-world 上比 simulation 好,说明 model 没有简单 overfit synthetic dynamics,而是学到 generalizable motion prior。

---

## 8. 整体 Design Philosophy 的 Intuition

把整个 TeleBoost 放一起看,作者其实在表达一个深层 thesis:

**Video post-training 的 bottleneck 是 feedback signal 的 operational quality,不是 generator 的 expressiveness。**

具体来说:
1. **SFT** 是 shape behavior space,eliminate structural failure,建立 stable reference
2. **GRPO + ViPO** 解决 "where to learn" —— spatiotemporal credit assignment
3. **GRPO + BPGO** 解决 "what to trust" —— reliability-aware optimization under ambiguity
4. **Self-Paced GRPO** 解决 "when to learn" —— competence-aware curriculum
5. **Joint Reward** 解决 "how to balance" —— multi-objective trade-off at advantage level
6. **DPO** 解决 "what to prefer" —— holistic human judgment capture

三个 dimension (Where / What / When) 对应三个 RL refinement,这种 decomposition 非常 clean。每个 component 解决一个 specific failure mode,而不是堆 trick。

这跟 LLM RLHF 的发展路径惊人相似 —— 也是从 naive PPO 到 RLHF → DPO → RLAIF → Constitutional AI → 各种 variance reduction trick。Video domain 因为 rollout cost 和 temporal structure,每个 step 都更难,但 conceptual framework 是 transferable 的。

---

## 9. 相关工作和延伸阅读

### Core References
- [DeepSeekMath / GRPO (Shao et al., 2024)](https://arxiv.org/abs/2402.03300) —— GRPO 原始 paper
- [DanceGRPO (Xue et al., 2025)](https://arxiv.org/abs/2505.07818) —— GRPO 应用到 visual generation
- [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) —— Direct Preference Optimization 原始 paper
- [ViPO (Ni et al., 2025)](https://arxiv.org/abs/2511.18719) —— Visual Preference Policy Optimization
- [BPGO (Liu et al., 2025b)](https://arxiv.org/abs/2511.18919) —— Bayesian Prior-Guided Optimization
- [Self-Paced GRPO (Li et al., 2025b)](https://arxiv.org/abs/2511.19356) —— Self-paced curriculum for video RL
- [Discriminator-Free DPO (Cheng et al., 2025)](https://arxiv.org/abs/2504.08542) —— Synthetic temporal negative
- [DenseDPO (Wu et al., 2025)](https://arxiv.org/abs/2506.03517) —— Fine-grained temporal DPO
- [VideoDPO (Liu et al., 2025c)](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_VideoDPO_Omni-Preference_Alignment_for_Video_Diffusion_Generation_CVPR_2025_paper.pdf) —— Omni-preference alignment

### Backbone & Dataset
- [Wan (Wang et al., 2025)](https://arxiv.org/abs/2503.20314) —— Wan video generation model (TeleBoost 的 backbone)
- [CogVideoX (Yang et al., 2024)](https://arxiv.org/abs/2408.06072) —— 另一个开源 video DiT
- [VBench (Huang et al., 2024)](https://arxiv.org/abs/2311.17982) —— Video generation benchmark
- [VideoAlign (Liu et al., 2025a)](https://arxiv.org/abs/2501.13918) —— Video generation with human feedback
- [VideoScore (He et al., 2024b)](https://aclanthology.org/2024.emnlp-main.130/) —— Fine-grained human feedback simulation

### Foundation Papers
- [DDPM (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) —— Denoising Diffusion Probabilistic Models
- [DiT (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748) —— Scalable Diffusion Models with Transformers
- [Video LDM (Blattmann et al., 2023)](https://arxiv.org/abs/2304.00850) —— Align Your Latents
- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193) —— ViPO 的 perceptual structuring backbone
- [VideoMAE (Tong et al., 2022)](https://arxiv.org/abs/2203.12602) —— 另一个 ViPO backbone 选项
- [ControlNet (Zhang et al., 2023)](https://arxiv.org/abs/2302.05543) —— Zero-init trick 来源
- [RAFT (Teed & Deng, 2020)](https://arxiv.org/abs/2003.12059) —— Optical flow extraction

### LLM RLHF 对照
- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) —— SFT + RLHF pipeline 原型
- [Constitutional AI (Bai et al., 2022)](https://arxiv.org/abs/2212.08073) —— AI feedback for harmlessness
- [Reward Model Overoptimization (Gao et al., 2023)](https://arxiv.org/abs/2210.10760) —— Scaling laws
- [Curriculum Learning (Bengio et al., 2009)](https://dl.acm.org/doi/10.1145/1553374.1553380) —— Self-Paced GRPO 的思想源头
- [Gradient Surgery (Yu et al., 2020)](https://papers.nips.cc/paper/2020/hash/3fe78a8a02f4f6d5b46c14b9d4b8c866-Abstract.html) —— Multi-task gradient conflict

### Infrastructure
- [Ray (Moritz et al., 2018)](https://www.usenix.org/conference/osdi18/presentation/moritz) —— Distributed framework
- [NVIDIA MPS](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf) —— Multi-Process Service
- [LoRA (Hu et al., 2022)](https://arxiv.org/abs/2106.09685) —— Low-rank adaptation

---

## 10. 我的 Intuition 和几个 Open Question

读完这篇 paper,我 build 出来的 intuition 是:

**Video post-training 是一个 signal engineering 问题,不是 algorithm engineering 问题。** 整个 TeleBoost 没有提出任何全新 algorithm,而是把已知的 technique (GRPO, DPO, curriculum learning, Bayesian prior, gradient surgery) 在 video domain 的 specific constraint 下重新组装,每个 component 解决一个 specific failure mode。这种 systematic approach 比 "提出一个新 algorithm 然后 claim SOTA" 要扎实得多。

几个我想深入探讨的 open question:

1. **ViPO 的 advantage map $M_i$ 如何 validated?** Paper 说用 frozen visual backbone 的 saliency,但 saliency 本身是 unsupervised 的 structural prior。如果 backbone 对某类 artifact 不 sensitive (e.g., 微妙的 color shift),ViPO 会 miss 这个 failure mode。能否 learn 一个 task-specific advantage map extractor?

2. **BPGO 的 prior selection 对 final performance 多 sensitive?** Paper 用 SFT model 的 historical reward 作为 prior。如果 SFT model 本身有 bias (e.g., 系统性偏好某种风格),prior 会 propagate 这个 bias。能否用 ensemble of priors 做 robustification?

3. **Self-Paced GRPO 的 phase transition 检测是否 robust?** 用 reward distribution sparsity 和 intra-group discriminability 作为 trigger。如果 reward model 在某 phase 突然 degrade (e.g., distribution shift in training data),transition 会不会 misfire?

4. **DPO 的 synthetic temporal negative 是否 over-constrain?** Frame shuffling 作为 $y_l$ 会 penalize 所有 non-causal motion,但有些 creative effect (e.g., reverse motion, time-lapse) 可能 valid。能否 design prompt-conditional synthetic negative?

5. **整个 pipeline 的 compute cost 和 scaling behavior?** Paper 给了 infrastructure 优化,但没给 total training cost / FLOPs / wall-clock time。Video post-training 的 scaling law 还是一个 open problem —— 是否跟 LLM 一样有 power law?还是因为 temporal dimension 导致 critical threshold?

6. **Three-stage pipeline 是否真的 optimal?** 能否 unified SFT + RL + DPO into single joint optimization?Paper 强调 staged design 是 because early stage shape behavior space 给 later stage stable foundation,但 joint optimization 是否能通过更好的 regularization 达到 same effect?

7. **Feedback 信号能否更 diagnostic?** Paper 在 conclusion 提到 "feedback should become more diagnostic rather than merely scalar"。这是非常 promising 的方向 —— 如果 reward model 不只给 scalar,还给 failure category + uncertainty + suggested fix,能否 build 一个 closed-loop "self-debugging" 系统?

这些 question 跟 LLM domain 当前的 frontier (process reward model, test-time compute, self-correction) 高度 parallel。Video domain 因为 temporal structure 和 high rollout cost,每个 question 都更难,但 conceptual framework 是相通的。

Reference 一下 Karpathy 你自己关于 neural network intuition 的 [blog post](https://karpathy.github.io/),和 [nanoGPT](https://github.com/karpathy/nanoGPT) 的 minimalism 思想 —— TeleBoost 这种 systematic framework 的价值在于 reproducibility 和 extensibility,而不是单个 algorithm 的 novelty。如果想 build 一个 production video generation system,这个 paper 提供了一个非常 actionable 的 blueprint。
