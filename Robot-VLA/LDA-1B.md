---
source_pdf: LDA-1B.pdf
paper_sha256: ff6b61c4ae1cc4c3c96aa868d10081bc61346e96f2c5f6d32db9f23ecd469ed9
processed_at: '2026-08-05T12:29:25-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# LDA-1B 人话版讲解

## 一句话说清楚这 paper 在干嘛

现在做 robot foundation model 的人基本都在玩同一个游戏：**收集一堆 expert teleoperation 数据，然后 behavior cloning**。π0、π0.5、GR00T、RDT、InternVLA 全是这条路。问题是 robot 数据太贵了，你很难 scale 到 internet-level，所以 everyone 卡在 10k hours 左右。

LDA 这群人说：**别只拿 expert data 做 BC 了，你手上有大量 noisy robot data、actionless human video，这些数据里其实藏着 "物理世界怎么运作" 的 knowledge，只不过 BC paradigm 用不上**。我们就设计一个 framework，让不同质量的数据各司其职——好数据学 policy，烂数据学 dynamics，没 action 的 video 学 visual forecasting——最后凑出 30k hours 训出一个 1B model，效果比 π0.5 还猛。

核心 thesis 就这么一句话：**BC paradigm 把 heterogeneous data 当 homogeneous 用，浪费了；我们应该 quality-aware 地用**。

## 为什么 BC paradigm 有问题

你想想 BC 在干嘛：给定 observation，predict expert action。这意味着：

1. 你只能用 expert data，low-quality trajectories 直接扔掉（因为 imitate suboptimal action 会教坏模型）
2. 模型只学 "看到 X 就做 Y"，没学 "做 Y 之后世界会变成什么样"
3. Long-horizon 任务一执行就 compounding error，因为 policy 不懂 consequence

证据就在他们的 real-world experiment 里：**Throw Rubbish**（捡纸团→放簸箕→倒垃圾桶）这个 long-horizon 任务，π0.5 和 GR00T 都 **0% success**，LDA 35%。为什么？因为 LDA 有 forward dynamics head，它在 latent space 里 "想象" 了 action 之后会发生什么，能 recover from intermediate failure。BC policy 没这个能力。

还有一个特别 striking 的实验（Table IV）：他们故意在 finetune 数据里掺 30-37% low-quality trajectories：

| Task | Method | High Only | High + Low | Δ |
|------|--------|-----------|------------|---|
| Place pen into box | π0.5 | 60 | 40 | **-20** |
| Place pen into box | LDA | 70 | 80 | **+10** |

π0.5 一加 noise 就崩 -20%，LDA 反而 **+10%**。这直接证明：**low-quality data 对 BC 是毒药，对 LDA 是营养**。这意味着 deployment 时数据采集成本可以大幅降低——不用精心 filter expert demo，随便录的 teleoperation data 直接喂进去就行。

## UWM 框架：四个 objectives 一起学

LDA 建立在 UWM (Unified World Model) 这个框架上，核心 idea 是一个模型同时学四个东西：

1. **Policy**: $p(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_t)$ — 看现在，predict 接下来 k 步 action
2. **Forward Dynamics**: $p(\mathbf{o}_{t+1:t+k} \mid \mathbf{o}_t, \mathbf{a}_{t+1:t+k})$ — 看现在 + 做什么 action，predict 接下来世界变成啥样
3. **Inverse Dynamics**: $p(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_{t:t+k})$ — 看一段 observation 序列，反推中间做了什么 action
4. **Visual Forecasting**: $p(\mathbf{o}_{t+1:t+k} \mid \mathbf{o}_t)$ — 只看现在，predict 未来（无 action condition）

为什么这四个一起学有用？因为它们 share 同一个 "世界怎么运作" 的 knowledge。Policy 告诉你 "做什么"，forward dynamics 告诉你 "做了之后会怎样"，inverse dynamics 让你从 passive video 里反推 action（这样 actionless video 也能 contribute），visual forecasting 让模型学 action-free 的 scene evolution（给 actionless video 用）。

实现上就是一个 flow-matching diffusion model，同时 denoise action chunk 和 future observation。Loss 就是 standard flow-matching：

$$
\ell_{\text{action}}^\theta = \mathbb{E} \left\| v_a^\theta - (\boldsymbol{\epsilon}_a - \mathbf{a}_{t+1:t+k}) \right\|_2^2
$$

$$
\ell_{\text{obs}}^\theta = \mathbb{E} \left\| v_o^\theta - (\boldsymbol{\epsilon}_o - \mathbf{o}_{t+1:t+k}) \right\|_2^2
$$

$$
\ell^\theta = \ell_{\text{action}}^\theta + \ell_{\text{obs}}^\theta
$$

变量解释：
- $v_a^\theta, v_o^\theta$：模型 predict 的 flow-matching velocity field（可以理解为 "noise 到 clean data 的方向"）
- $\boldsymbol{\epsilon}_a, \boldsymbol{\epsilon}_o \sim \mathcal{N}(\mathbf{0}, I)$：加的 Gaussian noise
- $\mathbf{a}_{t+1:t+k}$：ground-truth action chunk，长度 k=16
- $\mathbf{o}_{t+1:t+k}$：ground-truth future observation chunk
- 训练时 $\tau_a, \tau_o \sim \mathcal{U}(0, T_\tau)$ 独立采样 flow timestep

关键 trick：**根据 task 选择性 activate loss**。Policy training 只 activate $\ell_{\text{action}}$，visual forecasting 只 activate $\ell_{\text{obs}}$，high-quality data 两个都 activate。这样同一个 model 能吃 heterogeneous data。

工程上怎么实现？引入 4 个 learnable task embedding（对应四个 objective）加到 diffusion timestep embedding 上，再加 2 个 learnable register token（一个 action placeholder，一个 visual placeholder）。比如做 policy 时 visual 那个位置塞 register token，做 visual forecasting 时 action 位置塞 register token。这样 network topology 不变，只是 token content 变。

## 最关键的一个 ablation：VAE vs DINO

这是整个 paper 最重要的实验结果，看 Table II：

| Model | Vis. Rep. | Success Rate |
|-------|-----------|-------------|
| UWM-1B (VAE) | VAE | 19.3% |
| UWM + MM-DiT (VAE) | VAE | 20.0% |
| LDA (DINO, no MM-DiT) | DINO | 48.9% |
| **LDA-1B (DINO + MM-DiT)** | **DINO** | **55.4%** |

注意 UWM + MM-DiT 只有 20.0%，换成 DINO 直接跳到 55.4%，**+35.4%**。这个 single design choice 比加 MM-DiT、scale model size 都重要得多。

为什么差这么多？UWM 在 **pixel space / VAE latent space** 预测 future observation。VAE latent 把 appearance、geometry、dynamics 全 entangle 在 low-level feature granularity 里。你训练 forward dynamics head 的时候，模型大部分 gradient 都花在 reconstruct illumination 变化、texture 细节、background clutter 这些跟 action-induced state transition 无关的东西上。本质上 optimization landscape 被低频 appearance noise 主导，真正学不到 "物体怎么动" 的 dynamics。

DINO 不一样。DINOv3 是 self-supervised pretrain 出来的，features 已经自然 factorize 出 object-level semantics + spatial structure，suppress 了 low-level visual variations。你在 DINO latent space 做 forward dynamics prediction，loss 直接 optimize 在 "object 在哪、怎么动" 这个 level，gradient signal 干净得多。

Fig. 16 可视化了这点：左边 RGB，中间 ground-truth DINO feature，右边 LDA predict 的 DINO feature。Predict 的 feature 在 object permanence、contact continuity、motion consistency 上都对了，而且 **invariant to visual distractors 不在 control loop 里的**。

Fig. 11 的 attention map 实验更直观：他们对比 active action（"Push Right"）vs No-Op 的 attention difference $\Delta A = |A_1 - A_2|$。Push Right 时 attention 集中在 mug 的 leading edge + 预期运动方向；Push Close 时集中在 drawer 接触面。Background clutter 被 suppress。这证明模型 attention 是 **action-conditioned** 的，不是 passive encode 静态 appearance。

## 为什么能 scale 到 30k hours

Fig. 10 的 scaling analysis 是这 paper 的核心 evidence。他们在 held-out Agibot World subset 上测 action prediction L1 error，比 success rate 更 stable 可复现。

四条曲线对比：
- **Policy Only (grey)**：加 data 一开始 reduce error，但加 low-quality data 后开始 degrade（BC overfit suboptimal action）
- **Policy + Visual Forecasting (green)**：好一点但没 fully exploit
- **Policy + Forward/Inverse Dynamics (brown)**：同上
- **Full co-training (blue, ours)**：consistent monotonic improvement，**即使 action-labeled data 全用完了，再加 10k actionless video 还在降 error**

这说明 universal data ingestion 真的 work：actionless video 通过 visual forecasting objective 提供 supervisory signal，low-quality trajectory 通过 dynamics objective 提供 signal。换 BC paradigm 这些 data 全是 noise。

对比 UWM 的 scaling curve：UWM 很快 saturate，加 data 和 model capacity 都 diminishing/negative return。原因就是前面说的 VAE latent entanglement——latent space 没法 support compositional + causal reasoning，再 scale 也没用。LDA 在 DINO latent 里 scale 0.1B → 0.5B → 1B 单调降 error。

这给你一个重要 intuition：**scaling law 不仅仅看 data 量和 model size，还要看 representation space 是否 "scalable"**。Pixel/VAE latent 有 scaling ceiling，DINO latent 没有（至少在 1B 范围内没有）。

## Architecture 细节：MM-DiT

架构图看 Fig. 2。核心是 Multi-Modal Diffusion Transformer，jointly process action tokens 和 visual tokens。

Conditioning signals：
- Current observation + language instruction → Qwen3-VL-4B-Instruct 编码成 conditioning tokens
- Diffusion timestep → sinusoidal embedding
- Task specification → learned task embedding
- History (过去 2 timesteps 的 DINO obs + action)

所有 conditioning 通过 **AdaLN** 注入每个 transformer block：
$$
\text{AdaLN}(x, c) = \gamma(c) \cdot \text{LayerNorm}(x) + \beta(c)
$$
$\gamma, \beta$ 是 learned function of conditioning $c$。这跟 DiT (Peebles & Xie) 原设计一致。

Action 和 future visual feature 各自加 Gaussian noise，各自过 modality-specific linear projection 成 tokens，concatenate 后进 MM-DiT blocks。每个 block：
1. Multi-modal self-attention over concatenated action + visual tokens（cross-modal interaction）
2. Modality-specific QKV projection + FFN（preserve inductive bias）
3. Shared attention across modalities
4. Language tokens via cross-attention（high-level semantic guidance）
5. Modality-specific output heads predict denoised action + visual

这个设计借鉴 SD3 / FLUX 的 MM-DiT。Ablation 显示去掉 MM-DiT 换普通 DiT 掉 6.5%，缩到 0.5B 掉 4.7%。

Hidden size 1536，16 layers，32 heads，image (224,224,3) → DINO latent (14,14,384)。Action chunk 长度 16。Pretrain batch 32×48=1536，finetune 12×8=96。LR 1e-4，AdamW，cosine schedule。

## 一个容易被忽略的细节：asynchronous sampling

Visual 3Hz，Action 10Hz。这看着像工程细节，其实有物理意义。

连续两帧视觉变化很小（尤其 10Hz 时），redundant computation 很浪费。降到 3Hz visual 保留 semantic-level dynamics，10Hz action 保留 fine-grained motor control。这跟人类 perception-motor asymmetry 类似——视觉处理慢，motor control 快。

工程上让两个 stream 在 transformer 里通过 positional encoding 对齐，模型自己学 multi-scale temporal dynamics。

## EI-30K Dataset 怎么搭的

30k+ hours，四类：

**Real-world Robot (8.03k hours)**
- Open X-Embodiment: 3000h
- Agibot World: 3276h
- RoboMIND: 305h
- Humanoid Everyday: 30h
- RoboCOIN: 500h（noisy 但 environment diverse）
- Galaxea: 500h

**Simulated Robot (8.6k hours)**
- InternData-A1: 7433h（dense 自动生成）
- Behavior-1k: 1200h（long-horizon household）
- LET: 1000h

**Ego Human w/ Action (7.2k hours)**
- Ego4D: 3670h
- Ego-Exo4d: 1286h
- EgoDex: 830h（fine-grained 3D hand）
- SSV2: 240h
- Epic-Kitchens: 100h
- HOT3D: 16h
- HoloAssist: 166h
- OAKINK2: 6.5h
- TACO: 3.2h
- HOI4D: 7.6h

**Ego Human Actionless (10k+ hours)**
- Egocentric-10k: 10000h（最大 source）
- RH20T-human: 100h
- Taste-Rob: 130h
- Egome: 80h
- ARCTIC: 2.3h

工程上全 convert 成 LeRobot 2.1 format（HuggingFace 的 robot data standard）。Action representation 统一成 hand-centric：robot 是 6-DoF EEF pose + gripper width / dexterous joints，human 是 6-DoF wrist pose + full MANO params。所有 coordinate frame 手动 align 到 canonical EEF frame，camera extrinsics 保留用来 decouple hand motion from egocentric head motion（ego video 头会动，得 reprojection）。

Quality label 按 action accuracy + annotation completeness 打，但 **low-quality 不扔**，留着 quality-aware training 用。

## 实验结果最有意思的几个点

**Simulation (RoboCasa-GR1, 24 tasks)**：

LDA-1B 55.4% vs GR00T-N1.6 (3B) 47.6% vs GR00T-EI10k (1B, same data) 51.3%。注意 GR00T-EI10k 是他们 reproduce 的 strong baseline，用同样的 EI-30k high-quality subset pretrain，所以 gain 不是来自 data，是来自 method。

最大 gains 都在 "Close" 类任务（PnP X To Y Close）：需要 place object 后 retract arm 不碰倒东西。GR00T 经常 retract 时撞倒 object（Fig. 12），LDA 通过 forward dynamics 能 anticipate post-action consequence 避免。

具体数字：
- PnP Can To Drawer Close: GR00T 13% → LDA 71% (+58%)
- PnP Wine To Cabinet Close: 16.5% → 57% (+40.5%)
- PnP Milk To Microwave Close: 14% → 52% (+38%)

**Real-world (Galbot G1 + Unitree G1)**：

Galbot G1 不在 pretraining data 里，所以这是 few-shot adaptation to new embodiment。结果：

| Category | LDA | GR00T | π0.5 |
|----------|-----|-------|------|
| Pick & Place (Handover) | 90% | 50% | 70% |
| Contact-rich (Flip Box) | 60% | 20% | - |
| Fine (Pouring) | 80% | - | 60% |
| Long-horizon (Throw Rubbish) | 35% | **0%** | **0%** |

Long-horizon 那个 0% vs 35% 最 striking——baselines 完全做不到。

Dexterous manipulation（Fig. 7）：
- Pull Nail (low DoF): LDA 80% vs π0.5 ~0% vs GR00T 40%
- Flip Bread (high DoF, 22-DoF Sharpa hand): LDA 90% vs baselines 10%

High-DoF 上 advantage 巨大，作者归因于 large-scale human data（MANO params）提供的 latent prior。

**Generalization (Table III, Pick & Place)**：
- Unseen objects: LDA 60% vs π0.5 26.7%
- Unseen background: 60% vs 20%
- OOD position: 40% vs 6.7%

LDA 在 visual/spatial perturbation 下 robust，因为 DINO latent 已经 factorize 掉 appearance，dynamics learning focus 在 affordance 上。

## 你的 intuition 应该怎么 build

我觉得这 paper 最 important 的 take-away 有几条：

1. **Data quality heterogeneity 是 feature 不是 bug**。BC paradigm 把所有 data 当 imitation target，所以 low-quality data 是毒。但如果你 decompose 成多个 objectives，low-quality data 能贡献 dynamics knowledge（"世界怎么变"），不需要贡献 policy knowledge（"该做什么"）。这 unlock 了大量 previously wasted data。

2. **Latent space 选择决定 scaling ceiling**。Pixel/VAE latent entangle appearance + dynamics，dynamics signal 被 appearance noise 淹没。DINO latent 已经 pretrain 出 object-level semantics，dynamics learning 在这个 space 里 clean 得多。这跟 LeCun 一直 push 的 "reason in latent space, not pixel space" 思路一致。

3. **Forward dynamics 是 long-horizon 的 enabler**。BC policy 没后果预期，长序列 compounding error 必然崩。加 forward dynamics head 让模型 "想象" action consequence，能 recover from intermediate failure。Throw Rubbish 0% vs 35% 就是 evidence。

4. **Actionless video 有用**。通过 visual forecasting + inverse dynamics，actionless human video 也能 contribute supervisory signal。Fig. 10 显示加 10k actionless video 后 error 继续降。这开了一条路：internet-scale video data 可以 ingest 进 robot foundation model。

5. **Cross-embodiment learning 通过 hand-centric action space 实现**。统一到 wrist-centered coordinate，gripper width / MANO keypoints 都在 wrist frame，让 human 和 robot data 能 joint learn。Dexterous task 的 +48% gain 主要来自 human MANO data 的 prior。

6. **Scaling law 不只看 data 量和 model size**。Representation space 的 "scalability" 同样重要。UWM 在 VAE latent 里 scale 不动，LDA 在 DINO latent 里 scale 顺畅。这跟你之前讲 LLM 时强调的 "data quality > data quantity" 思路一脉相承——只是这里 quality 维度变成 "representation 是否支持 compositional + causal reasoning"。

7. **同一个 model 多 objective 通过 task embedding + register token 实现**。这个工程 trick 让一个 diffusion model 灵活切换 input-output structure，inference 时想用 policy 就用 policy embedding，想用 forward dynamics 就换 embedding。简洁有效。

## 跟你 (Karpathy) 工作的 connection

你一直强调 world model 是 AI 最重要 capability，Tesla AI Day 展示的 "vector space perception" 思路，以及 "data has different roles, don't treat it all the same"。LDA 这 paper 基本上在 robotics domain 印证这几条：

- World model: forward + inverse dynamics 在 latent space 实现 model-based reasoning
- Vector space: DINO latent 替代 pixel space，reasoning 在 structured latent 里
- Data roles: quality-aware role assignment，expert 学 policy，noisy 学 dynamics，actionless 学 visual forecasting

你 nanoGPT 系列 work 强调 diffusion + transformer 的 scaling properties，LDA 的 MM-DiT + flow-matching 是 consistent with 这条路。

唯一你可能会 push 的方向：他们 freeze 了 DINO，没 jointly learn visual representation 和 dynamics。Limitations 里也承认了。你大概会 argue 应该 end-to-end train visual encoder + dynamics together，让 representation 更 task-specific。这是他们 future work 列的第一条。

## 关键 references

- UWM (他们 build on): https://arxiv.org/abs/2503.00200
- DINOv3 (visual encoder): https://arxiv.org/abs/2508.10104
- DINO-WM (类似 latent world model 思路): https://arxiv.org/abs/2411.04983
- π0.5 (主要 baseline): https://arxiv.org/abs/2504.16054
- GR00T N1 (NVIDIA humanoid model): https://arxiv.org/abs/2503.14734
- LeRobot (data format): https://github.com/huggingface/lerobot
- MM-DiT (SD3): https://arxiv.org/abs/2403.07714
- Flow Matching (Lipman et al.): https://arxiv.org/abs/2210.02747
- MANO (hand model): https://psbasel.github.io/mano/
- Motus (concurrent work): https://arxiv.org/abs/2512.13030
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- EI-30K 数据组合细节看 paper Appendix D 和 Table IX

项目主页：https://pku-epic.github.io/LDA

---

# LDA-1B 深度技术讲解

## I. 核心定位与设计哲学

LDA-1B (Latent Dynamics Action Model, 1.6B parameters) 是一个 robot foundation model，由 Peking University、Galbot、CASIA、BAAI、Tsinghua、NVIDIA 等联合提出。核心 thesis 是：**在 structured DINO latent space 中联合学习 policy、forward dynamics、inverse dynamics 和 visual forecasting，通过 universal embodied data ingestion 将 heterogeneous data 按质量分层利用，从而突破 behavior cloning-centric paradigm 的 scaling ceiling**。

项目主页：https://pku-epic.github.io/LDA

这里关键 insight 是 current robot foundation models (π0, π0.5, RDT, InternVLA, GR00T) 都 largely 依赖 behavior cloning，只 utilizes expert demonstrations，丢弃了大量 heterogeneous data 中 embeddable 的 transferable dynamics knowledge。UWM (Unified World Model) framework 虽然 theoretically 可以 leverage diverse data，但现有 instantiations 在 pixel-space 预测，entangle 了 appearance modeling 和 dynamics learning，导致 illumination、texture、background clutter、camera viewpoint 等 low-level variations 主导 training objective，无法 scale 到 foundation-level。

## II. Preliminary: Unified World Model (UWM) Formulation

给定 current observation $o_t$ (RGB image)，UWM jointly models 四个 conditional distributions:

1. **Policy**: $p(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_t)$ — 给定当前观测预测 action chunk
2. **Forward Dynamics**: $p(\mathbf{o}_{t+1:t+k} \mid \mathbf{o}_t, \mathbf{a}_{t+1:t+k})$ — 给定当前观测和 action chunk 预测未来观测
3. **Inverse Dynamics**: $p(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_{t:t+k})$ — 给定观测序列反推 action
4. **Visual Planning / Forecasting**: $p(\mathbf{o}_{t+1:t+k} \mid \mathbf{o}_t)$ — 无条件 action 的未来观测预测

UWM 用 joint diffusion model 同时 denoise action 和 future observation:

$$
(\boldsymbol{\epsilon}_a^\theta, \boldsymbol{\epsilon}_o^\theta) = s_\theta\big(\tilde{o}, \tilde{a}_{t_a}, \tilde{o}_{t_o}', t_a, t_{o'}\big)
$$

其中:
- $s_\theta$ 是 denoising network with parameters $\theta$
- $\tilde{a}_{t_a}$ 是 corrupted action chunk at diffusion timestep $t_a$
- $\tilde{o}_{t_o}'$ 是 corrupted future observation at diffusion timestep $t_o'$
- $t_a, t_{o'}$ 是 independently sampled diffusion timesteps
- $\boldsymbol{\epsilon}_a^\theta, \boldsymbol{\epsilon}_o^\theta$ 是 predicted noise for action 和 observation respectively

LDA 在此基础上引入 language conditioning $\ell$ via VLM (Qwen3-VL)，实现 instruction-guided prediction。

参考 UWM paper: https://arxiv.org/abs/2503.00200

## III. Universal Data Ingestion via Multi-task Co-training

这是 LDA 的核心 methodological contribution。数据分三类 role:

| Data Type | Role | Supervision |
|-----------|------|-------------|
| High-quality robot/human demos | Policy + Forward Dyn + Inverse Dyn + Visual Forecasting | All objectives |
| Lower-quality trajectories | Forward Dyn + Inverse Dyn + Visual Forecasting | Dynamics only (no policy) |
| Actionless human videos | Visual Forecasting only | Visual Planning only |

这种 role-aware usage prevents overfitting 到 expert-only behaviors，并 enable scalable learning of transferable dynamics。

### Task Embedding 与 Register Token 机制

为实现 single diffusion model 支持不同 input-output structure，引入：
- **4 个 learnable task embeddings**: 对应 policy、forward dynamics、inverse dynamics、visual forecasting，added 到 diffusion timestep embedding $f_t$ 来 condition denoising process
- **2 个 learnable register tokens**: 一个 action register，一个 visual register，作为 absent modality 的 placeholder

例如在 policy training 中，模型收到 noisy action tokens + visual register token（代表未观测的未来状态）；visual forecasting 中则反之。

### Flow-Matching Training Objective

模型预测 denoising vector field $\mathbf{v}_a^\theta$ (action) 和 $\mathbf{v}_o^\theta$ (observation)，loss 为:

$$
\ell_{\text{action}}^\theta = \mathbb{E}_{(o_{t:t+k}, a_{t+1:t+k}, \ell) \sim \mathcal{D}} \left\| v_a^\theta - (\boldsymbol{\epsilon}_a - \mathbf{a}_{t+1:t+k}) \right\|_2^2
$$

$$
\ell_{\text{obs}}^\theta = \mathbb{E}_{(o_{t:t+k}, a_{t+1:t+k}, \ell) \sim \mathcal{D}} \left\| v_o^\theta - (\boldsymbol{\epsilon}_o - \mathbf{o}_{t+1:t+k}) \right\|_2^2
$$

$$
\ell^\theta = \ell_{\text{action}}^\theta + \ell_{\text{obs}}^\theta
$$

变量解释：
- $v_a^\theta, v_o^\theta$: 模型预测的 vector field（flow-matching velocity）
- $\boldsymbol{\epsilon}_a, \boldsymbol{\epsilon}_o \sim \mathcal{N}(\mathbf{0}, I)$: Gaussian noise
- $\tau_a, \tau_o \sim \mathcal{U}(0, T_\tau)$: uniformly sampled flow timesteps
- $\mathbf{a}_{t+1:t+k}$: ground-truth action chunk (长度 k)
- $\mathbf{o}_{t+1:t+k}$: ground-truth future observation chunk
- $\mathcal{D}$: data distribution
- $\ell$: language instruction

Training 时根据 task specification 选择性 activate action/visual loss，让 heterogeneous data 在 appropriate supervision 下 contribute。Inference 时通过指定 task embedding 和对应 inputs，同一模型可灵活 invoke 不同 objectives。

Flow matching (Lipman et al., 2023) 是 DDPM 的 continuous-time 变体，比 discrete diffusion timesteps 更 stable，与 π0、GR00T 等 SOTA VLA model 一致。

## IV. Representation of Predictive Targets

### Visual: DINO Latent (而非 VAE Pixel Space)

这是 LDA 与 UWM 的关键 architectural 区别，也是 ablation 中最大的 single performance jump (20.0% → 55.4%, +35.4%)。

LDA 采用 pretrained DINOv3-ViT-s encoder 提取的 latent features，而非 VAE-based pixel-space representations。

理由：
- DINO latents encode high-level semantic + spatial structure，suppress background noise 和 low-level visual variations
- VAE latents entangle appearance、geometry、dynamics 在 low-level feature granularity，导致 dynamics learning 被 redundant appearance modeling 主导
- DINO 通过 large-scale visual pretraining (self-supervised) 获得 object-level semantic invariance，让 dynamics learning focus on interaction-relevant transitions

参考 DINOv3: https://arxiv.org/abs/2508.10104
参考 DINO-WM (类似思想): https://arxiv.org/abs/2411.04983

### Action: Hand-Centric Unified Action Space

为统一 heterogeneous embodiments 的 action representation，定义 hand-centric action space:
- **Delta wrist pose**: 6-DoF end-effector motion
- **Finger configuration**: 
  - Parallel-jaw gripper: 1-DoF gripper width
  - Dexterous hand: keypoints in wrist coordinate frame

这个设计 enable consistent action modeling across different embodiments (gripper vs. multi-finger hand vs. human hand with MANO params)。

### Temporal Streams: Asynchronous Sampling

两个 synchronized temporal streams with 不同 sampling rates:
- Visual: 3 Hz
- Action: 10 Hz

这种 mixed-frequency 设计 reduces redundant computation (连续帧 highly correlated)，同时 preserve fine-grained action dynamics。这种 asynchronous 设计与 human perception 类似（视觉处理慢于 motor control），也让 model 学到 action-induced state transitions 的多尺度 dynamics。

## V. Architecture: MM-DiT (Multi-Modal Diffusion Transformer)

参考 Fig. 2 架构图。

### Backbone

MM-DiT jointly denoise action chunks 和 predict future visual features 在 unified diffusion framework 内。共享 Transformer backbone，处理 heterogeneous tokens。

### Conditioning Signals

| Condition | Encoding |
|-----------|----------|
| Current observation + language | Pretrained VLM (Qwen3-VL-4B-Instruct) → conditioning tokens |
| Diffusion timestep | Sinusoidal embedding |
| Task specification | Learned task embedding |
| History (2 timesteps) | Past DINO-encoded obs + actions |

所有 conditioning signals 通过 **AdaLN (Adaptive Layer Normalization)** 注入每个 Transformer block。

AdaLN 公式：给定 conditioning $c$，layer 输出为
$$
\text{AdaLN}(x, c) = \gamma(c) \cdot \text{LayerNorm}(x) + \beta(c)
$$
其中 $\gamma, \beta$ 是 learned scale/shift functions of conditioning。

### Token Processing

- Actions: fixed-length chunks + Gaussian noise corruption → modality-specific linear projection → action tokens
- Future visual features (DINO futures): parallel noising → modality-specific linear projection → visual tokens
- Concatenated 后进入 MM-DiT blocks

### MM-DiT Block 结构

每个 block:
1. **Multi-modal self-attention** over concatenated action + visual tokens (enabling cross-modal interaction)
2. **Modality-specific QKV projections + FFNs** (preserve inductive biases)
3. **Shared attention across modalities**
4. **Language tokens via cross-attention** (high-level semantic guidance)
5. **Modality-specific output heads** predict denoised action sequences + future visual features

这个设计借鉴了 Stable Diffusion 3 / PixArt-Σ / FLUX 的 MM-DiT (Peebles & Xie, DiT; Esser et al., SD3)。

## VI. EI-30K Dataset: Embodied Interaction Dataset

EI-30K 是 30k+ hours 的 heterogeneous embodied interaction trajectories:

| Category | Hours | Examples |
|----------|-------|----------|
| Real-world Robot | 8.03k | Open X-Embodiment (3k), Agibot World (3276), RoboMIND (305), Humanoid Everyday (30), RoboCOIN (500), Galaxea (500) |
| Simulated Robot | 8.6k | LET (1k), InternData-A1 (7433), Behavior-1k (1200) |
| Ego Human (w/ Action) | 7.2k | Ego4D (3670), Epic-Kitchens (100), Ego-Exo4d (1286), SSV2 (240), EgoDex (830), HOT3D (16), HoloAssist (166), OAKINK2 (6.5), TACO (3.2), HOI4D (7.6) |
| Ego Human (Actionless) | 10k+ | Egocentric-10k (10k), RH20T-human (100), Egome (80), Taste-Rob (130), ARCTIC (2.3) |

### Data Unification

- 所有数据 → LeRobot 2.1 format (HuggingFace LeRobot: https://github.com/huggingface/lerobot)
- 统一 observation、action、language 表示

### Aligned Action Representation

- **Robot**: 6-DoF end-effector pose + gripper width / dexterous hand joints
- **Human**: 6-DoF wrist pose + full MANO (Mano: https://psbasel.github.io/mano/) hand params
- Camera extrinsics retained to decouple hand motion from egocentric head motion
- 所有 coordinate frames 手动 aligned 以 ensure geometric consistency

### Quality Annotation

每个 trajectory assigned quality label based on action accuracy + annotation completeness。**Unlike aggressive filtering, low-quality trajectories 被保留**，让 downstream models 通过 quality-aware training exploit full spectrum。

## VII. Pre-training 与 Post-training

### Pre-training

- 48× NVIDIA H800 GPUs
- 400k iterations
- 4,608 GPU hours total
- VLM (Qwen3-VL) 和 DINOv3 encoder **frozen** throughout pre-training
- Only MM-DiT + action encoder/decoder updated

### Data-Efficient Finetuning

- 同 pretraining 数据 regime
- 利用 mixed-quality teleoperation data（无需 expert-level demonstrations）
- 在 finetuning stage **unfreeze VLM** enable end-to-end adaptation

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| VLM | Qwen3-VL-4B-Instruct |
| Observation Encoder | DINOv3-ViT-s |
| Hidden Size | 1536 |
| Layers | 16 |
| Attention Heads | 32 |
| Image Shape | (224, 224, 3) |
| Latent Image Shape | (14, 14, 384) |
| Action Chunk | 16 |
| Batch Size | 32 × 48 (pretrain) / 12 × 8 (finetune) |
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| Weight Decay | 1e-5 |
| Betas | [0.9, 0.95] |
| Epsilon | 1e-8 |
| LR Schedule | Cosine with min lr |

## VIII. Simulation Experiments: RoboCasa-GR1

### Benchmark

RoboCasa-GR1: 24 tabletop rearrangement + articulated-object manipulation tasks with GR-1 humanoid robot + Fourier dexterous hands，egocentric RGB observations from head-mounted camera。Finetune 1,000 trajectories/task，51 trials/task。

RoboCasa paper: https://arxiv.org/abs/2406.02523 (referenced as [39])

### Main Results (Table II)

| Model | Vis. Rep. | MM-DiT | VLM | Success Rate ↑ |
|-------|-----------|--------|-----|----------------|
| GR00T-N1.6 | - | - | Cosmos | 47.6 |
| StarVLA | - | - | Qwen3vl | 47.8 |
| GR00T-EI30k | - | = | Qwen3vl | 51.3 |
| UWM-0.1B | VAE | × | - | 14.2 |
| UWM-1B | VAE | × | Qwen3vl | 19.3 |
| UWM(MM-DiT) | VAE | √ | Qwen3vl | 20.0 |
| LDA(DiT) | DINO | × | Qwen3vl | 48.9 |
| LDA-0.5B | DINO | √ | Qwen3vl | 50.7 |
| **LDA-1B** | **DINO** | **√** | **Qwen3vl** | **55.4** |

### Ablation Insight

1. **UWM (VAE-based)** 即使 scale 到 1B params + MM-DiT，依然只有 20.0% → architectural constraints fundamentally limit performance
2. **VAE → DINO**: 20.0% → 55.4% (+35.4%) → 最大的 single jump，证明 semantically structured latent spaces 对 scaling 至关重要
3. **MM-DiT**: 移除导致 -6.5%
4. **Model size**: 1B → 0.5B 导致 -4.7%

### Per-Task Analysis (Table VI, 24 tasks)

最大 gains:
- PnP Can To Drawer Close: 13% (GR00T) → 71% (LDA), +58%
- PnP Bottle To Cabinet Close: 51.5% → 76%, +24.5%
- PnP Milk To Microwave Close: 14% → 52%, +38%
- PnP Wine To Cabinet Close: 16.5% → 57%, +40.5%

这些 "Close" 类任务需要 anticipation of post-action consequences（如 retract arm 时避免碰倒 object），正是 dynamics modeling 的价值所在。

Fig. 12 qualitative 显示 GR00T 经常在 place object 后 retract arm 时 collide with object 导致 tip over，LDA 通过 anticipation 避免。

## IX. Real-World Experiments

### Platforms

- **Galbot G1**: 2× 7-DoF arms，可装 2-finger parallel gripper 或 22-DoF SharpaWave dexterous hand
- **Unitree G1**: 10-DoF BrainCo hand + Zed Mini camera

所有配置只 receive egocentric RGB from head-mounted camera。

### Task Categories

| Category | Tasks |
|----------|-------|
| Pick & Place | Pick Vegetable, Handover |
| Contact-rich | Flip Box, Hammer (pnp2) |
| Fine Manipulation | Water Flower (pouring), Wipe Board |
| Long-horizon | Sweep Table, Throw Rubbish |
| Dexterous | Pick Bottle, Open MacBook, Pull Nail, Pick Bread, Flip Bread |

### Gripper Results (Fig. 6)

LDA vs. baselines across 4 categories:
- Pick & Place: LDA 显著领先（如 Handover 90% vs π0.5 70% vs GR00T 50%）
- Contact-rich: Flip Box LDA 60% vs GR00T 20%
- Fine Manipulation: Pouring 80% vs π0.5 60%
- **Long-horizon**: Throw Rubbish LDA 35%，baselines **0%** (完全失败)

### Dexterous Results (Fig. 7)

| Task | DoF | LDA | GR00T | π0.5 |
|------|-----|-----|-------|------|
| Pick Bottle | Low | 90% | 75% | 20% |
| Open MacBook | Low | ~high | ~high | ~high |
| Pull Nail | Low | 80% | 40% | ~0% |
| Pick Bread | High | 70% | 20% | 10% |
| Flip Bread | High | 90% | 10% | 10% |

Pull Nail 需要 precise force direction + stable contact，Flip Bread 需要 coordinated finger motion + continuous contact reasoning。LDA 在 high-DoF dexterous 上优势尤为显著，证明 large-scale human data 提供 strong latent priors for dexterous control。

### Generalization (Table III, Pick & Place)

| Setting | π0.5 | GR00T | LDA |
|---------|------|-------|-----|
| Object | 26.7 | 40.0 | 60.0 |
| Background | 20.0 | 40.0 | 60.0 |
| OOD Pos. | 6.7 | 20.0 | 40.0 |

LDA 在 unseen objects、backgrounds、OOD positions 都维持 60% success，证明 latent dynamics pretraining 让模型 focus on task-critical affordances 而非 visual distractors。

### Data-Efficient Mixed-Quality Finetuning (Table IV)

| Task | Method | High Only | High + Low | Δ |
|------|--------|-----------|------------|---|
| Place pen into box | π0.5 | 60 | 40 | -20 |
| Place pen into box | LDA | 70 | 80 | **+10** |
| Bimanually remove lid | π0.5 | 50 | 40 | -10 |
| Bimanually remove lid | LDA | 50 | 60 | **+10** |

这是 universal data ingestion 的关键 evidence：低质量 trajectories 对 baseline 有害，对 LDA 反而 helpful，+10% improvement。这意味着 data collection 成本可大幅降低。

## X. Scaling Analysis (Fig. 10)

在 held-out Agibot World subset 上评估 action prediction L1 error。

### Training Configurations

| Config | Components |
|--------|------------|
| (i) Policy Only | 仅 policy objective |
| (ii) Policy + Visual Forecasting | policy + visual planning |
| (iii) Policy + Forward/Inverse Dyn | policy + dynamics (无 visual forecasting) |
| (iv) Full co-training (Ours) | all 4 objectives |

### Key Findings

1. **Policy Only (grey line)**: 增加 dataset size 时 unstable；incorporating low-quality data 导致 degradation（因为 BC overfits to suboptimal actions）

2. **Partial co-training (green, brown)**: Improve robustness 但 fail to fully exploit data

3. **Full co-training (blue)**: Consistent improvement as heterogeneous data 引入。**Notably, 即使所有 action-labeled trajectories exhausted，添加 10k actionless videos 仍 continue reduce prediction error**

这证明 LDA 可从 low-quality data 和 non-action data 通过 latent dynamics + visual forecasting extract useful supervisory signals，而非 treat as noise。

### UWM vs. LDA Scaling Divergence

UWM (VAE-based) quickly saturates as data scale 和 model capacity 增加，additional supervision yields diminishing/negative returns。原因：VAE-derived latent representation entangles appearance + geometry + dynamics at low-level granularity，restricts 模型 factorize action-induced state transitions。

LDA (DINO-based) 在 semantically structured latent space 操作，dynamics learning scales smoothly with model capacity、richer objectives、more diverse datasets。0.1B → 0.5B → 1B 单调 reduce error。

## XI. Dynamics Learning Analysis

### Latent Forward Dynamics Visualization (Fig. 9, 16)

通过 PCA projection of DINO features 可视化。模型 produce coherent future-state predictions，respect physical constraints:
- Object permanence
- Contact continuity
- Motion consistency under applied action

Predicted dynamics focus on task-relevant objects，invariant to visual distractors 不影响 control loop。这表明 LDA 学到了 dynamics-aware latent world model，捕捉 actions causally propagate through scene，而非 mere visual appearance extrapolation。

### Action-Conditioned Attention (Fig. 11)

为 interpret action-induced state transitions，可视化 attention maps conditioned on action primitives:

1. 提取 middle transformer blocks 的 attention maps（high-level semantic + geometric info）
2. Active action (e.g. "Push Right"): compute attention $A_1$
3. No-Op (static) baseline: compute $A_2$
4. Difference: $\Delta A = |A_1 - A_2|$，isolates action-induced attention changes

$$
\Delta A = |A_1 - A_2|
$$

Results:
- **Push Right**: attention difference highlights mug 的 leading edge + anticipated motion direction
- **Push Close**: attention concentrates on drawer surface (contact + force application region)
- Background clutter 和 visually salient 但 non-interactive regions 被 suppressed

这证明 DiT dynamically re-weights visual tokens based on physics implied by action，而非 passively encode static appearance。

## XII. 与 Concurrent Work 对比

### Motus (concurrent UWM instantiation)
Motus: https://arxiv.org/abs/2512.13030
- Adopt UWM paradigm + integrate VLM + video generation priors
- 但仍 pixel-space operation，未 explicit consider data quality/scale/heterogeneity

### DyWA, FLARE, WorldVLA series
- Co-train next-state prediction + policy
- Demonstrate generalization improvement
- 但 pixel-space + 未 consider data quality roles

### Being-H0, UniVLA (Hybrid approaches)
- Being-H0: https://arxiv.org/abs/2507.15597 (human video VLA pretraining)
- UniVLA: https://arxiv.org/abs/2505.06111 (latent action model + BC)
- 依赖 action alignment 或 auxiliary latent action models，limiting effective data scale to ~6k hours

### LDA-1B 突破点
- Break 6k hour ceiling → 30k hours
- Unified world model formulation + DINO latent + quality-aware data roles

## XIII. Limitations 与 Future Work

### Limitations
1. **Reliance on fixed DINO visual features**: 无法 jointly learn visual representation + latent dynamics，可能 constrain generalization 到 novel visual perspectives
2. **Predominantly egocentric camera viewpoints**: multi-view 和 third-person views 未 extensively 验证
3. **Manual data role assignment**: 当前 quality labels based on action accuracy + annotation completeness，未 automated

### Future Directions
1. Jointly learn visual representation + latent dynamics
2. Extend to richer sensory modalities (tactile, audio, proprioceptive)
3. Automatically optimize data roles via learned curriculum
4. Foster community adoption of scalable heterogeneous data-driven paradigm

## XIV. Intuition Building: 为什么这套设计 work？

### Core Intuition 1: Decoupling Appearance from Dynamics

UWM 在 pixel space 失败的核心原因是 **appearance modeling 和 dynamics modeling 的优化 landscape 不对齐**。Pixel reconstruction loss 主要 driven by low-level variations (illumination, texture)，这些 variations 与 action-induced state transitions 弱相关。DINO features 通过 self-supervised pretraining 已学会 factorize appearance from semantic structure，让 dynamics objective 直接 optimize 在 object-level + spatial structure 空间，signaling更 clean。

### Core Intuition 2: Heterogeneous Data 的 Complementary Roles

BC paradigm 强迫所有 data 扮演同一 role (imitation target)，但 data 本身有 quality heterogeneity：
- High-quality demos: 同时提供 "what to do" (policy) 和 "what happens" (dynamics)
- Low-quality trajectories: 提供 "what happens" 但 "what to do" 可能 suboptimal
- Actionless videos: 仅提供 "what happens in natural manipulation"

LDA 通过 multi-task co-training 让每类 data contribute 到 appropriate objective，avoid over-constraining BC on noisy actions。

### Core Intuition 3: Forward Dynamics 作为 Generalization Backbone

BC policy 在 long-horizon 失败 (Throw Rubbish 0%) 是因为 compounding errors：policy 无法 anticipate post-action consequences。Forward dynamics modeling force policy 学会 "action → state transition" 的 causal mapping，让模型 recover from intermediate deviations。这也是 GR00T 在 place object 后 retract arm 时 tip over，而 LDA 能 anticipate 并避免。

### Core Intuition 4: Mixed-Frequency Streams

视觉 3Hz + action 10Hz 对应 physical reality：视觉变化慢于 motor control。这种 asynchronous 设计让 model 学到 action-induced state transitions 的 multi-scale dynamics，reduces redundant computation from highly correlated consecutive frames。

## XV. Engineering 与 Implementation 细节

### Pre-training Cluster
- 48× H800 GPU
- 400k iterations
- 4,608 GPU hours
- 推算单卡 throughput ≈ 4608/400000/48 ≈ 0.24 sec/iter/8-GPU node (合理范围 for 1B MM-DiT + frozen VLM/DINO)

### VLM Frozen Strategy
Pre-training 时 freeze VLM + DINO，only train MM-DiT + action encoder/decoder。这 preserve pretrained foundation models 的 cross-modal understanding 和 fine-grained visual feature extraction capability，避免 catastrophic forgetting。

Finetuning 时 unfreeze VLM enable end-to-end adaptation，further improve performance。

### Coordinate Frame Alignment
Manually align coordinate frames across diverse robot + human embodiments (Fig. 3)。这是 engineering-heavy 但 critical 的步骤，让 heterogeneous data 在 shared geometric space 中 joint learn。Camera extrinsics retained to decouple hand motion from egocentric head motion (egocentric video 中 head 会 move，需 reprojection 到 fixed world frame)。

### Data Cleaning Pipeline
1. **Dataset Standardization**: LeRobot 2.1 format，10 Hz resample
2. **Coordinate Alignment + Cleaning**: EEF alignment, camera motion decoupling, MANO keypoints standardization, hand visibility validation
3. **Language Normalization**: VLM-based 统一 annotation
4. **Quality-Aware Labels**: action accuracy + annotation completeness

### Inference Flexibility
同一模型通过 task embedding + register tokens 切换不同 modes:
- Policy mode: noisy action tokens + visual register → predict action
- Forward dynamics: noisy action + noisy visual → predict future visual
- Visual forecasting: action register + noisy visual → predict future visual (无 action supervision)
- Inverse dynamics: noisy action + clean visual sequence → predict action

## XVI. 关键 References 深度链接

### Foundation Components
- **UWM**: https://arxiv.org/abs/2503.00200 (Unified Video Action Model)
- **DINOv3**: https://arxiv.org/abs/2508.10104 (visual encoder)
- **Qwen3-VL**: https://arxiv.org/abs/2505.09388 (language-vision encoder)
- **DINO-WM**: https://arxiv.org/abs/2411.04983 (DINO latent world model for planning)
- **DiT (Peebles & Xie)**: https://arxiv.org/abs/2212.09748 (Diffusion Transformer)
- **DDPM (Ho et al.)**: https://arxiv.org/abs/2006.11239 (Denoising Diffusion Probabilistic Models)
- **LeRobot**: https://github.com/huggingface/lerobot (data format standard)

### Robot Foundation Models
- **π0**: https://arxiv.org/abs/2410.24164 (Physical Intelligence VLA flow model)
- **π0.5**: https://arxiv.org/abs/2504.16054 (open-world generalization VLA)
- **GR00T N1**: NVIDIA humanoid foundation model (referenced as [40])
- **RDT-1B**: https://arxiv.org/abs/2410.07864 (bimanual diffusion foundation)
- **InternVLA-M1**: https://arxiv.org/abs/2510.13778
- **UniVLA**: https://arxiv.org/abs/2505.06111 (latent action VLA)
- **Being-H0**: https://arxiv.org/abs/2507.15597 (human video VLA pretraining)
- **GRASPVLA**: https://arxiv.org/abs/2505.03233 (synthetic data grasping)
- **OpenVLA**: https://arxiv.org/abs/2406.09246 (open-source VLA)
- **Octo**: https://arxiv.org/abs/2405.12213 (generalist policy)

### Datasets in EI-30K
- **Open X-Embodiment**: https://arxiv.org/abs/2310.08864
- **Agibot World**: https://arxiv.org/abs/2508.02744 (referenced as [6])
- **RoboCasa**: https://arxiv.org/abs/2406.02523
- **Ego4D**: https://arxiv.org/abs/2110.07058 (referenced as [18])
- **Ego-Exo4D**: https://arxiv.org/abs/2403.13320 (referenced as [19])
- **Epic-Kitchens**: https://arxiv.org/abs/1804.02748
- **EgoDex**: https://arxiv.org/abs/2505.11709 (referenced as [21])
- **HOT3D**: https://arxiv.org/abs/2506.04145 (referenced as [2])
- **ARCTIC**: https://arxiv.org/abs/2205.08304 (bimanual hand-object)
- **HOI4D**: https://arxiv.org/abs/2207.09253 (egocentric 4D HOI)
- **TACO**: https://arxiv.org/abs/2402.01147 (tool-action-object)
- **MANO**: https://psbasel.github.io/mano/ (hand model)
- **DROID**: https://arxiv.org/abs/2403.09096 (referenced as [26])
- **RH20T**: https://arxiv.org/abs/2307.00595 (referenced as [16])

### Concurrent/Related World Models
- **Motus**: https://arxiv.org/abs/2512.13030 (concurrent latent action world model)
- **DyWA**: https://arxiv.org/abs/2503.16806 (dynamics-adaptive world action)
- **WorldVLA**: https://arxiv.org/abs/2506.21539 (autoregressive action world model)
- **FLARE**: https://arxiv.org/abs/2505.15659 (implicit world modeling)
- **R3M**: https://arxiv.org/abs/2207.05349 (universal visual representation for robotics, referenced as [38])
- **VIP**: https://arxiv.org/abs/2210.00030 (value-implicit pretraining, referenced as [37])

### Architectural Background
- **AdaLN**: https://arxiv.org/abs/1907.01490 (Meta-Networks) and Peebles & Xie DiT
- **MM-DiT** (Stable Diffusion 3): https://arxiv.org/abs/2403.07714
- **PixArt-α/Σ**: https://arxiv.org/abs/2305.11274

## XVII. 与 Andrej Karpathy 工作 的 Connection

### World Models 思想
LDA 的设计 philosophically aligned with Karpathy 关于 "world models" 的多次 talk (e.g., "World Models" at NeurIPS, AI's Most Important Capability)。LDA 通过 explicit forward/inverse dynamics modeling 在 latent space 实现 model-based reasoning，与 Karpathy 提倡的 "model that understands physics and consequences" 直接呼应。

### Data Scaling
Karpathy 在多个 talk (e.g., "The State of Open Source AI") 强调 data quality 的 heterogeneous utilization 是 key bottleneck。LDA 的 universal data ingestion 直接 address 这个问题——quality-aware role assignment，让 low-quality data 不再 garbage-in-garbage-out。

### Latent Representation
DINO (Caron et al., 2021) 是 Yann LeCun lab 的 work，与 Karpathy 在 Tesla AI Day 展示的 "vector space" perception 思想相似——both argue for structured latent space over pixel-space reasoning。LDA 验证了这一 thesis 在 robotics foundation model 上的有效性。

### Diffusion + Transformer
Karpathy 多次称赞 diffusion models + transformer backbone 的 scaling properties (e.g., nanoGPT 系列 work)。LDA 的 MM-DiT 选用 flow-matching + DiT 是 consistent with 这一 philosophy。

### Curriculum / Data Roles
Karpathy 在 talk 中提过 "data has different roles, you shouldn't treat it all the same"。LDA 的 task embedding + register token mechanism engineering 上实现这一 idea。

## XVIII. 总结：LDA 的真正 contribution

1. **Methodological**: Quality-aware role assignment for heterogeneous embodied data via UWM framework
2. **Architectural**: DINO latent + MM-DiT 让 dynamics learning decoupled from appearance modeling，enable stable 1B-scale training
3. **Data Infrastructure**: EI-30K 是目前 largest open heterogeneous embodied interaction dataset with unified format + aligned action representation
4. **Empirical**: +21% contact-rich, +48% dexterous, +23% long-horizon over π0.5；+10% via leveraging 30% low-quality trajectories
5. **Scaling Laws**: Jointly scale data + objectives + model capacity → monotonic improvement；policy-only saturates fast

这项 work 在 robot foundation model 路线上标志着从 BC-centric paradigm 到 dynamics-centric paradigm 的 transition，与 LLM 中 pretraining paradigm 的 evolution ( supervised → self-supervised next-token prediction) 类似——前者 limited by labeled data scale，后者 unlock internet-scale data utilization。

希望这个 detailed walkthrough 帮助 build intuition 关于 LDA-1B 的设计 motivation、architecture choices、experimental evidence 和 future implications。如果想 deep dive 某个 component (e.g., MM-DiT block 实现、DINO feature extraction pipeline、EI-30K preprocessing details、scaling law curves fitting)，可以进一步 discuss。
