---
source_pdf: From Video to Control A Survey of.pdf
paper_sha256: b5374df3834120792124c264aff2a03f0a675476ae37092ca97ed30d2e023221
processed_at: '2026-08-04T11:06:22-07:00'
target_folder: Robot-VLA/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 survey

## 0. 一句话总结

这篇 paper 在讲一件事：**互联网上有海量 video，但 robot 需要 action。怎么把 video 里的 "世界怎么动" 的知识，变成 robot 能执行的控制信号**。作者把这个 gap 拆成三种解法，每种都有自己的 trade-off。

---

## 1. 为什么这事难——data asymmetry

Robot learning 现在的 mainstream path 很 clear：collecting 大量 (observation, action, reward) 的 tuples，训一个 policy。RT-1 训了 130k real episodes，OXE 凑了 22 个 robot 才累积到 1M trajectories。每个 trajectory 都需要人 teleoperate，巨贵。

但 web 上的 video 呢？Ego4D 3000 小时，EPIC-Kitchens 100 小时，HowTo100M 有 100M 个 clips。这些 video 捕捉了物体如何运动、手怎么抓东西、contact 怎么发生——本质上就是 world dynamics 的完整记录。问题就一个：**没有 action label**。

更微妙的是，video 和 robot data 的 distribution 也 mismatch：
- **Embodiment mismatch**：video 里是人手，robot 是 gripper
- **Viewpoint mismatch**：video 是 ego-centric 或 third-person，robot 是 wrist cam + scene cam
- **Modality mismatch**：video 可能只有 RGB，robot 还有 proprioception、force
- **Physical constraint mismatch**：robot 有 joint limit、payload limit、workspace limit，video 里人随便怎么动都行

所以核心矛盾：**最 abundant 的 supervision (action-free video) 和最 useful 的 supervision (robot action) 之间有 gap**。这个 survey 就是在 survey 各种 bridge 这个 gap 的 method。

---

## 2. Taxonomy 的核心 idea——不要按 model class 分类

传统 survey 都按 architecture 分类：transformer vs diffusion vs RNN。这篇 paper 拒绝这种分类法，理由很简单：同一个 transformer 可以做成 GR-1 (direct action prediction) 也可以做成 LAPA (latent action pretraining)。同一个 diffusion model 可以是 PAD (joint video-action denoising) 也可以是 UniPi (video plan + inverse dynamics)。**Architecture 不是 design space 的主要 axis**。

那什么是？作者提出两条 axis：

**Axis 1: Interface explicitness**——video-derived 的 knowledge 在 control loop 里有多 visible？
- 完全 implicit：藏在 shared feature 里（Direct family）
- 中间：是个 learned variable 但 opaque（Latent action）
- 完全 explicit：是个 human 能 inspect 的 plan / trajectory / pose（Explicit interface）

**Axis 2: Distance from robot action**——预测的 target 离 motor command 有多远？
- 近：直接 predict action（Direct）
- 远：predict subgoal / trajectory / pose，还要 controller 翻译（Explicit）

这两条 axis 把方法分成三个 cluster：

| Family | Interface 在哪 | 例子 |
|---|---|---|
| Direct video-action | 隐式在 shared backbone | GR-1, GR-2, PAD, UWM, UVA, VidMan, VPP |
| Latent action | 学出来的 bottleneck variable | CLASP, FICC, LAPO, Genie, LAPA, UniVLA |
| Explicit visual | human-interpretable target | UniPi, SuSIE, AVDC, VRB, MimicPlay, GeneralFlow |

**Key intuition**：这三个 family 其实对应 classical robotics 里的三个 control pattern——end-to-end visuomotor、learned MPC、hierarchical planner-controller。Learned method 只是给 classical pattern 加上了 video pretraining 的 scalability，同时失去了 classical 的 verifiability。

---

## 3. Family I: Direct Video-Action Policies

### 3.1 核心 idea

最直觉的解法：**video prediction 当 auxiliary loss，shape 一个好 representation，然后 robot demos 把这个 representation ground 到 action**。Deployment 时 bypass video generation，直接 decode action。

形式化讲，training 同时 optimize：
- $\mathcal{L}_{video}(\phi) = \mathbb{E}_{(o_{t:t+H}) \sim \mathcal{D}_{video}} [-\log p_\theta(o_{t+H} | o_{t:t})]$ — video prediction loss
- $\mathcal{L}_{action}(\phi) = \mathbb{E}_{(o_t, a_t) \sim \mathcal{D}_{robot}} [-\log p_\theta(a_t | o_{t:t-K})]$ — action prediction loss

变量含义：
- $o_t$：observation at time $t$
- $a_t$：robot action
- $H$：video prediction horizon
- $K$：observation history length
- $\mathcal{D}_{video}$：action-free video dataset
- $\mathcal{D}_{robot}$：action-labeled robot dataset

### 3.2 三个 sub-pattern

**(a) Joint generators**：一个 backbone 同时生成 video 和 action

GR-1 (https://arxiv.org/abs/2312.06517) 是 GPT-style transformer，把 future visual tokens 和 action tokens 放在同一个 sequence 里 autoregressive predict。两阶段：先 Ego4D 上 pretrain video prediction，再 CALVIN 上 fine-tune joint。CALVIN ABC→D SR@1 = 85.4%。

GR-2 (https://arxiv.org/abs/2410.06158) 把 video pretraining data 扩到 HowTo100M + Ego4D + Something-Something V2 + EPIC-KITCHENS + Kinetics-700 + robot video。100+ real tasks avg success 74.7%。改用 action chunk 而不是 single step，smoothness 好。

PAD (https://arxiv.org/abs/2411.18179) 把 autoregressive 换成 diffusion。Joint denoise image 和 action。对 action-free video 用 mask trick：只 denoise image branch。Receding-horizon execution（只 execute 第一个 action）。MetaWorld 50 tasks avg 72.5%。

UWM (https://arxiv.org/abs/2506.08812) 关键 trick：**给 video branch 和 action branch 不同 diffusion timestep**，同一个 model 可以 query 成 policy / forward dynamics / inverse dynamics。这把 unified world model 做成了 multi-query interface。

UVA (https://arxiv.org/abs/2503.00200) 进一步 factorize：shared encoder + 两个轻量 diffusion head（video 一个，action 一个）。Deployment bypass video head，高频 control。

**(b) Two-stage (frozen video predictor + adapter)**：

VidMan (https://arxiv.org/abs/2411.09153) 用 Open-Sora video diffusion，frozen 后插 layerwise adapter 学 inverse dynamics。CALVIN SR@1 91.5%。

VPP (https://arxiv.org/abs/2412.14803) 直接用 Stable Video Diffusion，extract first-pass representation（avoid multi-step denoising），condition action diffusion policy。CALVIN SR@1 96.5%，Avg.Len 4.33。

**(c) Boundary: latent-state world models**

APV (https://arxiv.org/abs/2203.13880) 和 ContextWM (https://arxiv.org/abs/2305.18499) 用 action-free video pretrain RSSM，然后 DreamerV2-style model-based RL ground action。

RSSM 的 latent state $s_t = (h_t, z_t)$，其中：
- $h_t = f_\theta(s_{t-1}, a_{t-1}, o_t)$：deterministic recurrent state
- $z_t \sim q_\theta(z_t | h_t, o_t)$：stochastic latent，从 posterior 采样
- $\hat{z}_t \sim p_\theta(\hat{z}_t | h_t)$：prior，rollout 时用

### 3.3 Direct family 的 fundamental limitation

§4.4 是这篇 survey 最 sharp 的分析。Direct family 没有 inspectable intermediate，意味着：

1. **No pre-execution verification**：执行前没法 check reachability、collision、contact consistency
2. **No natural point for constraint injection**：collision checking、workspace filtering 没地方插
3. **Failure localization 困难**：degradation 来源无法 isolate——是 representation 弱？rollout 组织有问题？embodiment mismatch？action decoding 错？没 separable intermediate

而且 execution mode（stepwise / chunked / receding / feature-conditioned / latent-rollout）是 deployment-level choice，和 training architecture 正交。Diffusion 可以 chunked 也可以 receding，autoregressive 可以 stepwise 也可以 chunked。

Intuition：**Direct family 用 simplicity 和 scalability 换 inspectability**。当它 work 时很 elegant，当它 fail 时你只能 black-box debug。

---

## 4. Family II: Latent Action Interfaces

### 4.1 核心 idea

**Video transition 本身就 constrain 了产生它的 action**。即使 action 不可观测，从 $o_t$ 到 $o_{t+1}$ 的变化暗示了某个 cause。学一个 bottleneck 变量 $z_t$ 来 capture 这个 cause。

核心公式：
$$z_t \sim q_\phi(z_t | o_t, o_{t+H}), \quad \hat{o}_{t+H} \sim p_\theta(\cdot | o_t, z_t)$$

变量含义：
- $q_\phi(z_t | o_t, o_{t+H})$：encoder，本质是 inverse dynamics（从 transition 推 cause）
- $p_\theta(\hat{o}_{t+H} | o_t, z_t)$：decoder，本质是 forward dynamics（从 cause 推 effect）
- $z_t$：latent action
- $H$：prediction horizon

Bottleneck 强制 $z_t$ 只 retain transition-relevant info，suppress static content。Training 就是 reconstruct $\hat{o}_{t+H}$ 接近 $o_{t+H}$。

这个 factorization 的精妙之处：$z_t$ 是 inverse dynamics 的 output 也是 forward dynamics 的 input，一个变量两个角色。学完后 $z_t$ 是 compact action-like interface，可以用少量 robot data ground 到 real action。

### 4.2 三种 bottleneck

**(a) Continuous IB (CLASP)**：

CLASP (https://arxiv.org/abs/1806.09655) 用 β-VAE KL upweight 强制 minimality。还有 composer network 把多个 latent action 组合成 trajectory-level code。Control 用 image-goal MPC，在 latent space 里 CEM search。

**(b) Discrete VQ (FICC, LAPO, Genie)**：

VQ-VAE (https://arxiv.org/abs/1711.00937) 把 continuous latent 离散化成 codebook $\mathcal{C} = \{c_1, ..., c_K\}$：
$$z_t = \arg\min_{c_k \in \mathcal{C}} \|z_t^{enc} - c_k\|$$

梯度用 straight-through estimator。好处：finite action token set，grounding 简单（co-occurrence dictionary），compatibility with LM backbone。

FICC (https://openreview.net/forum?id=Sy-o2N0hF4f) 在 learned feature space 用 VQ，加 cycle consistency。Atari-50k 上 work。

LAPO (https://arxiv.org/abs/2312.10812) 直接训 latent policy predict discrete code，Procgen 4M frames 恢复 expert。

Genie (https://arxiv.org/abs/2402.15391) 在 pixel space VQ，在 RT-1 video（action removed）上训。Figure 7 显示 latent code 自发对应 end-effector 的稳定方向——很 surprising 的 emergent property。CoinRun 200 expert samples match oracle BC。

**(c) Latent action for VLA (LAPA, UniVLA)**：

LAPA (https://arxiv.org/abs/2410.11758) 关键 idea：**latent action 只当 pretraining target，不当 deployment interface**。先用 latent code pretrain VLA backbone，然后 replace 成 real-action head fine-tune。LIBERO avg SR 65.7%。

UniVLA (https://arxiv.org/abs/2506.06269) 解决一个 fundamental problem：transition-based code 会 entangle controllable 和 task-irrelevant dynamics。两阶段 decoupling：
- Stage 1：language condition encoder/decoder，让 $z_t$ encode **task-irrelevant** residual（language 已经 explain task-relevant）
- Stage 2：freeze Stage 1，新 init task-centric codebook（no language），让它 capture task-relevant change

LIBERO avg SR 95.2%。

### 4.3 Latent action 的三个 failure mode（§5.6 很精彩）

**(1) Identifiability 问题**：$z_t$ 不保证对应 controllable 物理量

Video transition 里同时有：robot action、camera egomotion、other actor、lighting、gravity-induced dynamics。Bottleneck 鼓励 compress 但不 enforce disentanglement。

比如 CLASP、Genie 这种 pixel-space method，camera viewpoint 变化和 arm motion 可能 encode 到同一个 $z_t$，grounding 到 action 后 distribution shift 一来就崩。UniVLA 的 task-centric/task-irrelevant factorization 是 step toward 解决，但只是 approximate factorization。

**(2) Forward model 的 physical consistency**：

$p_\theta(\hat{o}_{t+H} | o_t, z_t)$ 只 optimize reconstruction loss，不 enforce physics。可能预测出 visually plausible 但 dynamically impossible 的 transition：
- 物体穿透
- Grasp without contact
- Instantaneous acceleration

Multi-step rollout 时这些 error compounding。Classical physics engine 用 non-penetration constraint、friction cone 保证 consistency，learned model 得 implicitly 从 data 学这些 regularity，常常学不全。

**(3) Grounding brittleness**：

$z_t \to a_t$ 的 mapping 通常很轻量（decoder、co-occurrence dictionary、head replacement），容量小，没法 correct latent space 和 action space 的 misalignment。

Co-occurrence grounding（FICC、Genie）是 discrete assignment，多个 real action 产生类似 short-horizon transition 时就 break。Head-replacement（LAPA）有 representation gap：pretraining shape backbone 给 latent-code，swap 成 real action head 后可能需要额外 adaptation。

**Intuition**：latent action family 的 fundamental tension 是 **abstraction vs executability**。Compress transition 成 compact code 容易，但保证 code 对应 robot 能 execute 的东西难。

---

## 5. Family III: Explicit Visual Interfaces

### 5.1 核心 idea

**不要 compress 到 latent 或 implicit feature，直接 predict 一个 human 能 inspect 的 target**——subgoal image、video plan、point trajectory、pose sequence。让 downstream controller 直接 track。

好处：
- **Transparency**：执行前可 visualize 检查
- **Cross-embodiment transfer**：interface 在 visual/geometric space，不在 action space
- **Modularity**：predictor 和 controller 可独立 improve

代价：
- Interface prediction error
- Perception/transfer pipeline compounding error
- Generative model 的 physics hallucination

### 5.2 Sub-cluster

**(a) Dense video plan + direct grounding**：

UniPi (https://arxiv.org/abs/2306.17672) hierarchical video diffusion：先 sparse keyframe 再 temporal super-resolution，inverse dynamics 翻译 frame-pair 成 action。CALVIN SR@1 56%。

Gen2Act (https://arxiv.org/abs/2409.16283) 直接训 video-conditioned policy，不用 inverse dynamics module。

**(b) Video plan + interface transfer to pose/trajectory**：

AVDC (https://arxiv.org/abs/2310.08576) 流程很复杂：
1. Diffusion 生成 imagined execution video
2. Optical flow between frames
3. First frame depth lift 2D flow to 3D
4. PnP (https://link.springer.com/article/10.1007/s11263-008-0152-6) + RANSAC (https://dl.acm.org/doi/10.1145/358669.358692) fit rigid SE(3)
5. Robot grasp + apply SE(3) trajectory

Meta-World 10 tasks avg SR 43.1%。

RIGVid (https://arxiv.org/abs/2507.00990) 加 VLM-based rollout selection（filter for instruction consistency）+ closed-loop 6D pose tracking。

Dreamitate (https://openreview.net/forum?id=InT87E5sr4) stereo video + MegaPose (https://arxiv.org/abs/2212.06870) for tool 6D pose。

GVF-TAPE (https://arxiv.org/abs/2509.00361) 直接 extract **end-effector pose** 而不是 object pose，avoid rigid-object assumption。LIBERO Sp/Ob/Go: 95.5/86.7/66.8%。

Dream2Flow (https://arxiv.org/abs/2512.24766) 生成 video → segmentation + point tracking + Video-Depth-Anything (https://arxiv.org/abs/2501.12375) → 3D object flow → trajectory optimization。支持 deformable 和 granular flow。

**(c) Subgoal image + goal-conditioned policy**：

SuSIE (https://arxiv.org/abs/2310.10639) 用 InstructPix2Pix (https://arxiv.org/abs/2211.09800) 作 planner，predict subgoal image，low-level policy 执行短 horizon 到达。CALVIN SR@1 87%。

CLOVER (https://arxiv.org/abs/2409.09016) RGB-D subgoal sequence + embedding-distance error signal → inverse dynamics policy。SR@1 96%, Avg.Len 3.53。

V2A (https://arxiv.org/abs/2411.07223) goal-reaching policy 从 self-collected rollout + hindsight relabel 学，不需要 demo。

### 5.3 Trajectory-based interfaces

**(a) Affordance-based contact interface**：

VRB (https://arxiv.org/abs/2304.08488) 从 egocentric human video 学 contact point + post-contact 2D trajectory。

SWIM (https://arxiv.org/abs/2308.10901) 把 affordance 作 latent world model 的 control input，CEM search。Boundary case。

**(b) 2D pixel trajectory**：

ATM (https://arxiv.org/abs/2401.00025) 用 CoTracker (https://arxiv.org/abs/2307.07614) 生成 pseudo-label，训 track transformer predict any-point 2D trajectory。LIBERO Sp/Ob/Go/Lo: 68.5/68.0/77.8/39.3%。

Tra-MoE (https://arxiv.org/abs/2307.07614) 加 Mixture-of-Experts (https://arxiv.org/abs/1701.06538) 处理 multi-domain。

Im2Flow2Act (https://openreview.net/forum?id=cNI0ZkK1yC) object-centric 2D flow，sim-trained zero-shot transfer to real。

Track2Act (https://arxiv.org/abs/2405.01527) 2D track → depth back-project → PnP fit rigid SE(3) → execute + residual policy。

**(c) 3D/6D structured trajectory**：

GeneralFlow (https://arxiv.org/abs/2401.11439) 从 HOI4D RGB-D 学 3D object-point trajectory，SVD alignment (https://ieeexplore.ieee.org/document/4767965) produce SE(3) update。Zero-shot human→robot, 18 tasks avg SR 81%。

SKIL-H (https://arxiv.org/abs/2501.14400) semantic 3D keypoint trajectory，通过 foundation feature clustering + descriptor matching 发现 keypoint。

MimicPlay (https://arxiv.org/abs/2302.12422) 3D human hand trajectory from multi-view "human play"，plan code condition 低 level policy。

ZeroMimic (https://arxiv.org/abs/2503.23877) 6D wrist pose from EPIC-Kitchens + SfM。Post-grasp policy。Real robot avg SR 71.9% (Franka, 9 skills)。

### 5.4 Explicit interface 的四个 failure mode（§6.3）

**(1) Tracking error problem**：

Controller 要 track 一个 target（subgoal image、6D pose、point track），但 target 可能：
- 在 kinematic singularity 附近，小 target 变化要求大 joint motion
- 超出 workspace envelope
- Self-collision
- 对非 rigid interface（dense flow、point track），visual target → robot motion 是 underdetermined mapping

GeneralFlow 的 SVD alignment 可能 produce 超出 joint limit 的 displacement，ZeroMimic 的 6D chunk 可能 place wrist 在 singularity boundary。**关键问题**：feasibility check（reachability、collision-freeness、joint-limit compliance）几乎没 method 集成。

**(2) Open-loop vs closed-loop**：

UniPi/Dreamitate 是 open-loop（feedforward trajectory execution），robustness 完全靠 prediction accuracy。SuSIE/CLOVER/Im2Flow2Act/GeneralFlow/Track2Act 是 closed-loop（look-and-move visual servoing 变体）。Closed-loop mitigates compounding error 但需要 interface 能 high-frequency refresh。

**(3) Interface-transfer pipeline fragility**：

很多 method 需要 multi-step transfer：video → segmentation → tracking → depth → correspondence → pose fitting。每一步都是 learned module，每一步都有 error。Classical cascade estimation 也有这问题，但 classical 能 cross-check intermediate vs known dynamics，video pipeline 通常没法 verify。Error compounding：small segmentation error → depth lifting error → pose error → controller faithfully track 一个 wrong target。

**(4) Hallucinated physics in generated plans**：

Generative video model 最大化 visual likelihood 不保证 physics：
- Object sliding before contact
- Object penetration
- Disappearing parts
- Instantaneous acceleration
- Gravity-defying motion

Classical physics-based planning 用 constraint satisfaction 保证 consistency，learned generative model 没有。RIGVid 的 VLM-based rollout selection 部分缓解，但 systematic physical-consistency checking 是 open problem。

**Intuition**：explicit interface 用 inspectability 和 modular transfer 换 grounding pipeline 的 fragility。Classical visual servoing 早就有类似 issue，但 learned pipeline 还没建立 robust verification 机制。

---

## 6. Cross-Family Synthesis（§8）

### 6.1 三条 design axis

**Axis 1: Interface explicitness**——从 implicit (Direct) 到 opaque variable (Latent) 到 human-interpretable (Explicit)。

**Axis 2: Training factorization**——Direct 通常 joint/interleaved on mixed data；Latent 和 Explicit 是 two-stage（先 action-free 学 interface，再 ground 到 robot）。Recurring tension：**predictive ≠ realizable**。Video 给 scalable prior over scene change，必须 careful ground。

**Axis 3: Temporal abstraction & planning horizon**——Direct absorb temporal structure into policy（implicit）；Latent 最适合 abstraction（discrete code 作 compact unit for search）；Explicit 最 transparent 但当前 system 通常 short-horizon replan。

### 6.2 Deployment-level 的差异

| Family | Execution loop | Physical feasibility | Failure detection | Embodiment mismatch |
|---|---|---|---|---|
| Direct | Stepwise/chunked/receding, no intermediate | Implicit via action distribution | Opaque, hard to localize | Action head 紧 tie embodiment |
| Latent | MPC search or latent policy | Forward model 可能 visually predictive 但 dynamic invalid | Indirect rollout discrepancy | Grounding 对 distribution shift 敏感 |
| Explicit | Open-loop plan 或 closed-loop tracking | Predicted target 可能 kinematically unreachable | Discrepancy 可 measure, trigger replan | Interface-domain gap（human hand vs robot） |

### 6.3 Open challenges 四个 cluster（§8.3 是这篇 paper 最 future-looking 的部分）

**Cluster 1: Execution-aware and physically grounded learning**

Video prediction objective 实际 constrain 什么不明。可能 gain 来自：
- Better visual representation
- Regularization
- Stronger generative prior
- Action-video alignment

任意组合。Latent/explicit 都有 predictability vs controllability tension。

**Future direction**：
- Augment predictor with feasibility signal（constraint violation、uncertainty estimate）
- Incorporate lightweight physics prior into generation/decoding
- Couple temporal-abstraction learning to execution feedback（让 discovered primitive 保持 controllable）
- Multi-resolution interface（high-level visual context + short-horizon verifiable target）

**Cluster 2: Robust grounding and cross-embodiment transfer**

两方面：
1. **Separate controllable 和 exogenous dynamics**——action-free video 混了 robot-caused、camera motion、other agent、environment-driven dynamics。Possible handle：multi-view constraint、ego-motion compensation、counterfactual objective、multi-agent factorization
2. **Efficient embodiment adaptation**——current approach 从 learned inverse dynamics 到 retargeting 到 geometry-based controller 都有，但 retraining per robot 仍是 norm。Direction：lightweight adapter、shared action representation、retargeting with explicit embodiment constraint

**Cluster 3: Multimodal sensing and contact-rich manipulation**

当前 method 几乎都 vision-only + rigid/quasi-rigid task。Contact-rich assembly、deformable manipulation、force modulation 需要：
- Non-rigid state interface（dense 3D flow、keypoint field）
- Tactile / proprioceptive feedback alongside visual prediction
- Joint video-tactile representation learning

Open problem：video 缺 force，force data 缺 diverse video，怎么学 joint representation？RH20T (https://arxiv.org/abs/2307.00595) 等 dataset 是起点。

**Cluster 4: Evaluation, verification, safe deployment**

Method 间 benchmark、pretraining corpora、robot-data scale、modality、protocol 都不同，cross-method comparison 几乎不可能（Tables 4/7/10 都标注 "Not directly comparable"）。

**Need**：
- Control for robot-data budget + modality + task difficulty
- Metrics beyond success rate（robustness to perturbation、recovery behavior、uncertainty calibration）
- Standardized real-robot benchmark for long-horizon 和 contact-rich
- Verification hooks：lightweight module screen predicted interface for feasibility、estimate confidence、trigger replanning or safe fallback

---

## 7. 我的几个 meta-observation

### 7.1 "Video pretraining transfers how" 的 empirical evidence 还很弱

Tables 4/7/10 显示大部分 method 的 quantitative evidence 是 within-method ablation（with vs without pretraining），cross-method leaderboard 几乎不存在。这个 field 还在 "证明 video 有用" 阶段，没到 "证明哪种 video 用法最好" 阶段。

这和 LLM pretraining 早期很像——scaling law 还没建立。Bitter lesson 是 scale 会赢，但 video→robot 的 scale law 我们还不知道。是 pretraining data scale 重要还是 robot data scale 重要？是 video diversity 重要还是 video quality 重要？这些都没答案。

### 7.2 Latent action identifiability 是这个 field 的 alignment problem

学出来的 $z_t$ 不保证 controllable。这是因果推断里的 ICA-style identifiability 问题（https://arxiv.org/abs/2207.09141），但 video transition 里 multiple cause 同时变化，bottleneck 不够 enforce disentanglement。

UniVLA 的 task-centric/task-irrelevant 分离是 patch，但只是 approximate。真正解决可能需要：
- Multi-view geometry constraint
- Counterfactual objective（"如果 action 不变，scene 怎么变"）
- Causal intervention

### 7.3 Explicit interface 的 transfer pipeline 是新的 "sim-to-real"

AVDC、Dream2Flow 这类 method 把 generative video → segmentation → tracking → depth → SE(3) 串成 pipeline。每一步都是 learned module，每一步都有 error。

这本质上是把 classical perception-control pipeline 的 geometric engine 换成 learned module，但失去了 classical 的 verifiability。Classical visual servoing 有 stability theory、ISS 等保证，learned pipeline 还没建立类似的 theory。

### 7.4 Execution mode orthogonal to training architecture

Direct family 里同 architecture 可以 stepwise / chunked / receding / feature-conditioned / latent-rollout。这个 decoupling 意味着 deployment-level control theory 还需要单独研究，不能假设 end-to-end learned policy 自动有好的 control property。

Diffusion policy (https://arxiv.org/abs/2303.04137) 本身就是 receding-horizon，但 paper 没把它和 PAD/UWM/UVA 这类 joint video-action diffusion explicit connect。PAD 的 receding-horizon execution 实际上是 diffusion policy + video prediction 的自然组合。

### 7.5 Closed-loop vs open-loop 的回归

Classical robotics 几十年建立 closed-loop visual servoing 的稳定性 theory（https://link.springer.com/article/10.1023/A:1010046902645），learned method 又回到了 open-loop video plan（UniPi、Dreamitate）。这个 regression 是因为 generative model latency 高没法 closed-loop。

UVA、VPP、ATM、GeneralFlow 这类 bypass video generation 或 lightweight predictor 是回到 closed-loop 的路径。但 closed-loop 的 stability 分析在 learned setting 下还没人做。

### 7.6 World model + RL 的 scaling 还没在 manipulation 上证明

DreamerV3 (https://arxiv.org/abs/2301.04104) 在 Minecraft diamond 上证明 world model + RL 能 scale，但 APV/ContextWM 这类 action-free pretrain → action-conditional RL 的范式还没在 manipulation 上 scale 到这种程度。

可能原因：
- Manipulation 的 visual diversity 比 Minecraft 高得多
- Contact dynamics 比 Minecraft physics 复杂
- Robot data 比 Minecraft game playthrough 贵得多

### 7.7 V-JEPA 路线缺失

V-JEPA 2 (https://arxiv.org/abs/2506.07947) 是 LeCun 路线的 video representation learning，完全 non-generative，joint-embedding predictive architecture。Paper inclusion criteria (i) 要求 temporal prediction 但没明确排除 JEPA。JEPA 学的 representation 是否适合 manipulation grounding 是 open question。

JEPA 的优势是不生成 pixel，representation 更 abstract，可能更适合 control。但还没有 manipulation paper 用 V-JEPA 做 pretraining。

---

## 8. 最终的 intuition

这篇 survey 真正的 contribution 不是 catalog method，而是提出一个 **"robotics integration layer" thesis**：

> 当前 video-based manipulation 的 bottleneck 不在 representation learning（video prediction model 越来越强），而在**如何把 video-derived prediction 接入 closed-loop control 同时保持 physical feasibility、verifiability、cross-embodiment transfer**。

三个 family 在这条 axis 上的 trade-off：

- **Direct** 牺牲了 inspectability 换 simplicity，但 deployment 失败无法 localize
- **Latent** 引入 structured intermediate 但 identifiability 不保证，grounding 在 distribution shift 下 brittle
- **Explicit** 提供 inspectable target 但 transfer pipeline compounding error + physics hallucination 是新 bottleneck

Paper 最后指出四个 open direction（execution-aware learning、robust grounding、multimodal sensing、evaluation infrastructure）实际上对应 robotics integration layer 的四个 sub-problem。

**核心 takeaway**：video 是 world dynamics 的 scalable observation，但 observation ≠ control signal。中间必有一个 interface design 决定 dynamics knowledge 如何 enter control loop，而这个 design 的 trade-off 是 classical robotics 已经研究过的问题（visuomotor vs MPC vs planner-controller hierarchy），只是 learned method 加入了 scalability 和 flexibility，同时失去了 classical 的 verifiability。

下一个十年的工作是把 verifiability 加回来——不是退回 classical，而是在 learned setting 下建立新的 verification theory。这需要：
- Execution-aware learning（让 predictor 知道什么是 infeasible）
- Robust grounding（让 interface 能 transfer 到新 embodiment）
- Multimodal sensing（让 interface 包含 contact 信息）
- Evaluation infrastructure（让 method 能 fair compare）

这些都不是 sexy 的 ML problem，但是 robotics deployment 的真问题。这篇 survey 把这些问题 frame 出来，是这个 field 走向成熟的重要一步。

---

一些相关 reference 汇总：

- **Survey 本身**: 此次提供的 paper
- **Direct family**: GR-1 (https://arxiv.org/abs/2312.06517), GR-2 (https://arxiv.org/abs/2410.06158), PAD (https://arxiv.org/abs/2411.18179), UWM (https://arxiv.org/abs/2506.08812), UVA (https://arxiv.org/abs/2503.00200), VidMan (https://arxiv.org/abs/2411.09153), VPP (https://arxiv.org/abs/2412.14803), APV (https://arxiv.org/abs/2203.13880), ContextWM (https://arxiv.org/abs/2305.18499)
- **Latent family**: CLASP (https://arxiv.org/abs/1806.09655), FICC (https://openreview.net/forum?id=Sy-o2N0hF4f), LAPO (https://arxiv.org/abs/2312.10812), Genie (https://arxiv.org/abs/2402.15391), LAPA (https://arxiv.org/abs/2410.11758), UniVLA (https://arxiv.org/abs/2506.06269), VQ-VAE (https://arxiv.org/abs/1711.00937)
- **Explicit family**: UniPi (https://arxiv.org/abs/2306.17672), Gen2Act (https://arxiv.org/abs/2409.16283), AVDC (https://arxiv.org/abs/2310.08576), RIGVid (https://arxiv.org/abs/2507.00990), Dreamitate (https://openreview.net/forum?id=InT87E5sr4), GVF-TAPE (https://arxiv.org/abs/2509.00361), Dream2Flow (https://arxiv.org/abs/2512.24766), SuSIE (https://arxiv.org/abs/2310.10639), CLOVER (https://arxiv.org/abs/2409.09016), V2A (https://arxiv.org/abs/2411.07223), VRB (https://arxiv.org/abs/2304.08488), SWIM (https://arxiv.org/abs/2308.10901), ATM (https://arxiv.org/abs/2401.00025), Im2Flow2Act (https://openreview.net/forum?id=cNI0ZkK1yC), Track2Act (https://arxiv.org/abs/2405.01527), GeneralFlow (https://arxiv.org/abs/2401.11439), SKIL-H (https://arxiv.org/abs/2501.14400), MimicPlay (https://arxiv.org/abs/2302.12422), ZeroMimic (https://arxiv.org/abs/2503.23877)
- **Datasets**: Ego4D (https://ego4d-data.org/), EPIC-Kitchens (https://epic-kitchens.github.io/2024), HOI4D (https://hoi4d.github.io/), OXE (https://robotics-transformer-x.github.io/), DROID (https://droid-dataset.github.io/), CALVIN (https://calvinrobot.github.io/), LIBERO (https://lifelong-robot-learning.github.io/libero/)
- **Classical foundations**: DreamerV3 (https://arxiv.org/abs/2301.04104), Diffusion Policy (https://arxiv.org/abs/2303.04137), EPnP (https://link.springer.com/article/10.1007/s11263-008-0152-6), RANSAC (https://dl.acm.org/doi/10.1145/358669.358692), V-JEPA 2 (https://arxiv.org/abs/2506.07947)

---

# From Video to Control: 深度技术解析

好的 Andrej，这篇 paper 本质上是一个 **interface-centric taxonomy** 的 survey，核心论点是：video 是 world dynamics 的 scalable observation，但如何把这种 passive observation 转换成 robot control loop 里的 actionable signal，存在一个 design spectrum。我下面从 motivation、taxonomy 构造、每个 family 的技术细节、控制环路分析、以及 open problems 几个角度展开，尽量把 intuition 和数学/架构细节都讲透。

---

## 1. Motivation：为什么 video 是 robot learning 的 next frontier

### 1.1 Data asymmetry 的核心矛盾

Robot manipulation 当前的 mainstream approach（RT-1/RT-2/π0/Open X-Embodiment）依赖 action-labeled trajectories，collecting cost 极高。OXE 花了 22 个 robot embodiment 才累积到 1M+ trajectories。但 web 上的 video（Ego4D 3000h、EPIC-Kitchens 100h、HowTo100M 100M clips）是 action-free 的，capture 了 contact、affordance、object dynamics 的全部 temporal structure，只是没有 robot action supervision。

这个 asymmetry 的本质：**最 abundant 的 supervision 是 action-free video，最 directly useful 的 supervision 是 robot actions，二者很少 align**。这就是为什么 survey 提出需要 learning 一个 "interface" 来 bridge 这个 gap。

### 1.2 为什么不能直接用 video pretrain 一个 encoder 就完事

survey 在 Section 2.3 里明确排除了几类 boundary case：MOKA、FlowBot3D、ReKep（只用 static affordance/keypoint）；ManipulateBySeeing、GENIMA（只用 pretrained encoder 作 feature）；VIP、LIV（只学 reward 不学 predictive interface）；UniSim、PointWorld（action-conditioned simulator）。

排除它们的理由是一个关键 insight：**temporal prediction 是 supervision signal 的核心**，单纯用 image-level contrastive 或 reconstruction objective 没有学习到 "how scenes evolve over time under interaction" 这个 dynamics-relevant structure。所以 inclusion criteria 明确要求：(i) 用 temporal video supervision；(ii) interface 从 non-action video 学到；(iii) grounding 到 manipulation。

这个 framing 实际上对应一个更深的假设：**world dynamics knowledge 只有通过 forecasting objective 才能被 forced into representation**。这一点和 Hafner 的 world model 路线、和 LeCun 的 JEPA 路线在哲学上是一致的——predictive objective 是 representation learning 的 inductive bias。

Reference: Ego4D (https://ego4d-data.org/), OXE (https://robotics-transformer-x.github.io/), JEPA (https://openreview.net/forum?id=BvP9zLesn4M)

---

## 2. Taxonomy 的构造逻辑：两条 design axis

### 2.1 为什么不是 by model class

传统 survey 通常 by architecture（transformer vs diffusion vs SSM）or by supervision type。这篇 paper 拒绝这种 organization，理由很尖锐：methods 的核心 trade-off 不在 architecture，而在 **interface 在 control stack 里的位置** 和 **interface 的 explicitness**。同一个 transformer 可以做成 direct video-action（GR-1）也可以做成 latent-action（LAPA），同一个 diffusion model 可以做成 joint generator（PAD）也可以做成 interface predictor（UniPi）。

所以 paper 提出两条 axis（Figure 3）：

**x-axis: Interface explicitness** — 从 "linkage 隐藏在 shared feature 里" 到 "输出一个 human-interpretable target"。

**y-axis: Distance from robot actions** — 从 "直接 decode action" 到 "输出 subgoal/trajectory/pose 等需要 controller 翻译的抽象 target"。

### 2.2 三大 cluster 的涌现

虽然 design space 是连续的，但方法们 cluster 成三个 family：

| Family | Interface 位置 | Explicitness | 典型代表 |
|---|---|---|---|
| Direct video-action | implicit in shared backbone | 低 | GR-1/2, PAD, UWM, UVA, VidMan, VPP |
| Latent-action | learned bottleneck variable z | 中 | CLASP, FICC, LAPO, Genie, LAPA, UniVLA |
| Explicit visual | subgoal/trajectory/pose | 高 | UniPi, SuSIE, AVDC, VRB, MimicPlay, GeneralFlow |

这个 taxonomy 的一个关键好处：它**直接对应 classical robotics 的 control pattern**。Direct 对应 end-to-end visuomotor；Latent 对应 learned MPC over compressed dynamics；Explicit 对应 hierarchical planner-controller decomposition。这意味着 learned methods 继承了 classical pattern 的 trade-off，但加上了 video pretraining 的 scalability 和 flexibility。

---

## 3. Family I: Direct Video-Action Policies

### 3.1 核心假设与 formalization

这个 family 的 central assumption：**预测未来 frame 会 force representation 编码 dynamics-relevant structure**（object motion、contact transitions、interaction outcomes），然后这些 representation 可以通过 robot demos 被 ground 到 executable action。

形式化（paper §4）：
- Observation $o_t \in \mathcal{O}$
- Action $a_t \in \mathcal{A}$
- Language instruction $l$（optional）

Training 同时 optimize 两个 objective：
1. 大规模 action-free video 上的 temporal prediction loss
2. 小规模 robot dataset 上的 action label loss

部署时通常 **bypass video generation**（Table 2 里 "Video at Inference = Optional"），说明 video prediction 是 training-time representation shaping mechanism，不是 deployment-time plan。

### 3.2 三个 sub-pattern

**(a) Joint video-action generators**：一个 backbone 同时学 video prediction 和 action prediction。

GR-1 (Wu et al. 2023, https://arxiv.org/abs/2312.06517) 是 GPT-style autoregressive transformer，把 future visual tokens 和 action tokens 放在同一个 token stream 里。两阶段 training：先在 Ego4D 上 pretrain video prediction，再在 CALVIN 上 fine-tune joint prediction。CALVIN ABC→D 上的 SR@1 是 85.4%（Table 4）。

GR-2 (Cheang et al. 2024, https://arxiv.org/abs/2410.06158) scaling up：video pretraining 用 HowTo100M + Ego4D + Something-Something V2 + EPIC-KITCHENS + Kinetics-700 + robot video，并在 100+ real-world tasks上 evaluation，avg success 74.7%。把 GR-1 的 stepwise decoding 改成 action chunk，improves temporal consistency。

PAD (Guo et al. 2024, https://arxiv.org/abs/2411.18179) 把 autoregressive 换成 diffusion transformer，joint denoising future images 和 actions。对 action-free video 用 masked co-training：video-only sample 上 action branch 被 mask，只 denoise image。Receding-horizon execution（只 execute first predicted action）。MetaWorld 50 tasks avg SR 72.5%。

UWM (Zhu et al. 2025, https://arxiv.org/abs/2506.08812) 关键 insight：**给 video branch 和 action branch 不同的 diffusion timestep**，同一个 model 可以 query 成 policy（只 denoise action）、forward dynamics（denoise both）、inverse dynamics（condition on 两个 time step 的 observation denoise action）。这本质上把 unified world model 从单一 forward prediction 扩展成一个 multi-query interface。

UVA (Li et al. 2025, https://arxiv.org/abs/2503.00200) 进一步 factorize：shared latent encoder + 两个 lightweight diffusion head（一个 video、一个 action）。部署时 bypass video head，只跑 action head，inference 高频。这是 dual-process analogy 的实现：slow video predictor 学 dynamics，fast action head 用 predictive features。

**(b) Two-stage (frozen video predictor + action adapter)**：

VidMan (Wen et al. 2024b, https://arxiv.org/abs/2411.09153) 用 Open-Sora-style video diffusion transformer 在 OXE 上 pretrain，frozen 后插入 layerwise self-attention adapter 学 inverse dynamics。CALVIN ABC→D SR@1 91.5%（比 GR-1 高 6pt）。

VPP (Hu et al. 2024, https://arxiv.org/abs/2412.14803) 直接用 Stable Video Diffusion 作 backbone，extract predictive representation（first forward pass，avoid multi-step denoising），condition 一个 action diffusion policy。CALVIN SR@1 高达 96.5%，Avg.Len 4.33。MetaWorld 50 tasks avg 68.2%。

**(c) Boundary: Latent-state world models**：

APV (Seo et al. 2022, https://arxiv.org/abs/2203.13880) 和 ContextWM (Wu et al. 2023b, https://arxiv.org/abs/2305.18499) 用 action-free video pretrain RSSM (Recurrent State-Space Model)，然后 ground action 通过 DreamerV2-style model-based RL。

RSSM 的核心方程（来自 Hafner et al. 2019, https://arxiv.org/abs/1910.01375）：
$$h_t = f_\theta(s_{t-1}, a_{t-1}, o_t)$$
$$s_t = (h_t, \hat{z}_t) \text{ with } \hat{z}_t \sim q_\theta(z_t | h_t, o_t)$$

变量含义：
- $h_t$：deterministic recurrent state（GRU-like）
- $z_t$：stochastic latent state，从 posterior $q_\theta(z_t | h_t, o_t)$ 采样
- $\hat{z}_t$：prior sampled from $p_\theta(\hat{z}_t | h_t)$（用于 rollout 时 imagination）
- $s_t = (h_t, \hat{z}_t)$：full latent state

APV 的关键：先用 action-free video 训 reconstruction + latent prediction，把 RSSM 的 state encoder / transition model 初始化好，再在 robot interaction 上加 action-conditioned dynamics + reward/value head。ContextWM 进一步引入 context variable $c$ 从 random frame 提取，通过 multi-scale cross-attention condition decoder，把 time-invariant appearance（背景、纹理）absorb 到 $c$，让 $z_t$ 专注 dynamics。

### 3.3 Control integration 分析（§4.4 是这篇 survey 最精彩的部分之一）

Direct family 的核心 deployment-level 问题：**没有 inspectable intermediate 意味着什么**？

paper 给出三个具体后果：

1. **No pre-execution verification**：没办法在 robot 动之前检查 reachability、dynamic feasibility、contact consistency。
2. **No natural point for constraint injection**：collision checking、workspace filtering、constraint projection 这些 classical safeguard 都没地方插。
3. **Failure localization 困难**：degradation 可能来自 weak temporal representation、inference-time rollout 组织、embodiment mismatch、action decoding，没有 separable intermediate 来 isolate。比如 frozen video feature 漏掉 contact geometry 时，policy 会 silent wrong，不会产生 visibly flawed intermediate。

execution mode 是 deployment-level design choice，paper 在 Table 2 里整理了：stepwise（GR-1）、chunked（GR-2, UWM, UVA）、receding（PAD）、feature-conditioned（VidMan, VPP）、latent rollouts（APV, ContextWM）。**这个 axis 和 training architecture 正交**：diffusion 可以 chunked 也可以 receding，autoregressive 可以 stepwise 也可以 chunked。

Intuition：direct family 用 simplicity 换 inspection。代价是当 policy 失败时，你只能 black-box debug，没法 inject 任何 prior knowledge。

---

## 4. Family II: Latent-Action Interfaces

### 4.1 核心动机与 building blocks

Latent-action family 的 motivation：**video 的 transition 本身 constrain 了产生它的 action**，即使 action label 不可观测。所以可以学一个 bottleneck 变量 $z_t$ 来 capture "what caused the change"，再用少量 robot data 把 $z_t$ ground 到 executable action。

通用公式（§5.1）：
$$z_t \sim q_\phi(z_t | o_t, o_{t+H}), \quad \hat{o}_{t+H} \sim p_\theta(\cdot | o_t, z_t)$$

变量含义：
- $q_\phi(z_t | o_t, o_{t+H})$：encoder，本质上是 inverse dynamics——从 observed transition 推 latent cause
- $p_\theta(\hat{o}_{t+H} | o_t, z_t)$：decoder，本质上是 forward dynamics——给 $z_t$ 预测 future
- $H$：prediction horizon，通常 $H=1$（next frame）
- $z_t$：latent action

这个 factorization 的精妙之处：**$z_t$ 同时是 inverse dynamics 的 output 和 forward dynamics 的 input**。学习目标是 reconstruct $\hat{o}_{t+H}$ 接近 $o_{t+H}$，bottleneck 强制 $z_t$ 只 retain transition-relevant info，suppress static content。

### 4.2 三种 bottleneck 实现

**(a) Continuous information bottleneck（CLASP）**：

CLASP (Rybkin et al. 2019, https://arxiv.org/abs/1806.09655) 用 β-VAE 风格 KL upweight 强制 minimality。还引入 composer network 把连续 latent action 组合成 trajectory-level code，约束 "decode from composed code = decode step-by-step"，bias $z_t$ toward reusable primitives。

控制时 CLASP 用 image-goal MPC：在 latent-action space 里 CEM search，找 latent action sequence 使 predicted future 最 match goal image，然后 ground 到 real action execute，receding-horizon。

**(b) Discrete VQ codebook（FICC, LAPO, Genie）**：

FICC (Ye et al. 2023, https://openreview.net/forum?id=Sy-o2N0hF4f) 在 learned feature space 用 VQ bottleneck。VQ-VAE (van den Oord et al. 2018, https://arxiv.org/abs/1711.00937) 的核心：
$$z_t = \text{quantize}(z_t^{enc}, \mathcal{C}) = \arg\min_{c_k \in \mathcal{C}} \|z_t^{enc} - c_k\|$$

其中 $\mathcal{C} = \{c_1, ..., c_K\}$ 是 codebook。梯度通过 straight-through estimator 传到 encoder。FICC 还加 cycle consistency（feature space cosine similarity）+ difference-reconstruction term，bias code toward transition-relevant change。Grounding 通过 co-occurrence adapter：每个 real action 对应一个 latent embedding，从 interaction 数据里统计 co-occurrence。Atari-50k 上 median HNS 0.360。

LAPO (Schmidt & Jiang 2024, https://arxiv.org/abs/2312.10812) 用 VQ 学 latent action vocabulary，然后训 latent policy by imitation（直接 predict discrete code），最后再 train latent→action decoder 或 PPO fine-tune。Procgen 上 4M frames 恢复到 expert 性能（PPO baseline 只有 44%）。

Genie (Bruce et al. 2024, https://arxiv.org/abs/2402.15391) 把 VQ 用在 pixel space。在 RT-1 video（action removed）上训，发现 latent code 自发对应 end-effector 的稳定方向（Figure 7）。Grounding 用 co-occurrence dictionary。CoinRun BC 上 200 expert samples 就 match oracle BC。RT-1 video world model metrics：FVD 136.4，ΔtPSNR 2.07（Table 7）。

**(c) Latent actions for VLA（LAPA, UniVLA）**：

LAPA (Ye et al. 2025, https://arxiv.org/abs/2410.11758) 关键 insight：**latent action 不是 deployment interface，只是 pretraining target**。先用 video 学 latent action code，然后用这个 code 作 supervision pretrain VLA backbone（predict latent code from obs+language），最后 **replace latent head with real-action head** 并 fine-tune。部署时 latent action 完全不存在。LIBERO avg SR 65.7%（在 UniVLA 的 reproduce setting）。

UniVLA (Bu et al. 2025, https://arxiv.org/abs/2506.06269) 解决 latent-action 的一个 fundamental problem：**transition-based code 会 entangle controllable change 和 task-irrelevant dynamics（camera motion、other agent、clutter）**。解决方案是两阶段 decoupling：

- Stage 1：language-conditioned encoder + decoder，让 $z_t$ encode **task-irrelevant** residual（因为 language 已经 explain 了 task-relevant part）
- Stage 2：freeze Stage 1，新 init 一个 task-centric codebook，**no language conditioning**，让新 codebook capture task-relevant change that replaces language

最终 task-centric code 作 action token，VLA autoregressive predict 这些 token，部署时 lightweight decoder ground 到 real action。LIBERO avg SR 95.2%（full pretrain），88.7%（human-video only）。

### 4.3 Latent-action 的三个 cross-cutting failure mode（§5.6）

paper 提出三个关键 failure mode，对 build intuition 非常有用：

**1. Identifiability 问题**：$z_t$ 不保证对应物理量。Video transition 同时 encode：robot action、camera egomotion、other actor、lighting、scene dynamics（gravity 等）。IB / VQ bottleneck 鼓励 compress 但不保证 retain 的是 controllable info。pixel-space method（CLASP、Genie）可能把 camera-correlated viewpoint effect 吸进 $z_t$，grounding 在 distribution shift 下 brittle。UniVLA 的 task-centric/task-irrelevant factorization 是 step toward 解决，但只是 approximate。

**2. Physical consistency of latent forward model**：$p_\theta(\hat{o}_{t+H} | o_t, z_t)$ 训练时只 optimize reconstruction loss，不 enforce physics。可能预测出 visually plausible 但 dynamically impossible transition：object 穿透、grasp without contact、instantaneous acceleration。multi-step rollout 时 compounding。

**3. Grounding brittleness**：latent → action 的 mapping（decoder、dictionary、head replacement）容量小，没法 correct latent space 和 action space 的 misalignment。Co-occurrence grounding（FICC、Genie）是 discrete assignment，多个 real action 产生类似 short-horizon transition 时会 break。Head-replacement（LAPA）有 representation gap：pretraining shape backbone 给 latent-code prediction，swap 成 real-action head 后可能需要额外 adaptation。

Intuition：latent-action family 的 fundamental tension 是 **abstraction（compress transition 成 compact code）vs executability（code 对应 robot 能 execute 的东西）**。这个 tension 在所有 method 里都以不同形式出现。

---

## 5. Family III: Explicit Visual Interfaces

### 5.1 核心思想

explicit interface family 的 insight：**不 compress 到 latent 或 implicit feature，而是 predict 一个 human-interpretable target**（subgoal image、video plan、point trajectory、pose sequence），让 downstream controller 直接 track。好处：transparency（执行前可 inspect）、cross-embodiment transfer（interface 在 visual/geometric space）、modularity（predictor 和 controller 可独立 improve）。

代价：interface prediction 本身的 error、perception/transfer pipeline 的 compounding error、物理 hallucination（generative video model 最大化 likelihood 不保证 physics）。

### 5.2 三个 sub-cluster

**(a) Dense video plans + direct grounding**：

UniPi (Du et al. 2023, https://arxiv.org/abs/2306.17672) hierarchical video diffusion：先 sparse keyframe 再 temporal super-resolution，inverse dynamics 接 frame-pair 翻译成 action。CALVIN ABC→D SR@1 56%（Table 10）。

Gen2Act (Bharadhwaj et al. 2024, https://arxiv.org/abs/2409.16283) 不用 inverse dynamics，直接训 video-conditioned policy，policy 同时 condition on 生成 video + observation history。加 point-track motion auxiliary loss。

**(b) Video plan + interface transfer to pose/trajectory**：

AVDC (Ko et al. 2023, https://arxiv.org/abs/2310.08576) 关键流程：
1. Diffusion model 生成 "imagined execution" video
2. Dense optical flow between successive predicted frames
3. 用 first frame 的 depth 把 2D flow lift 到 3D
4. PnP-style (Lepetit et al. 2009, EPnP, https://link.springer.com/article/10.1007/s11263-008-0152-6) + RANSAC (Fischler & Bolles 1981, https://dl.acm.org/doi/10.1145/358669.358692) fit rigid SE(3) transform
5. Robot grasp object + apply SE(3) trajectory

Meta-World 10 tasks avg SR 43.1%（sim only）。

RIGVid (Patel et al. 2025, https://arxiv.org/abs/2507.00990) 加 proposal selection（VLM filter for instruction consistency）+ closed-loop 6D pose tracking。

Dreamitate (Liang et al. 2024, https://openreview.net/forum?id=InT87E5sr4) stereo video + MegaPose (Labbé et al. 2022, https://arxiv.org/abs/2212.06870) for tool 6D pose。

GVF-TAPE (Zhang et al. 2025, https://arxiv.org/abs/2509.00361) 从 predicted frame 提取 **end-effector pose** 而不是 object pose，avoid rigid-object assumption。LIBERO Sp/Ob/Go: 95.5/86.7/66.8%。

Dream2Flow (Dharmarajan et al. 2025, https://arxiv.org/abs/2512.24766) 生成 video → segmentation + point tracking + monocular video depth (Video-Depth-Anything, Chen et al. 2025, https://arxiv.org/abs/2501.12375) → 3D object flow → trajectory optimization / RL。支持 deformable 和 granular flow。

**(c) Subgoal image + goal-conditioned policy**：

SuSIE (Black et al. 2023, https://arxiv.org/abs/2310.10639) 用 InstructPix2Pix (Brooks et al. 2023, https://arxiv.org/abs/2211.09800) 作 high-level planner，predict subgoal image，low-level goal-conditioned policy 执行短 horizon 到达。Iterate 长 horizon。CALVIN SR@1 87%。

CLOVER (Bu et al. 2024, https://arxiv.org/abs/2409.09016) RGB-D subgoal sequence + embedding-distance error signal → inverse dynamics policy。SR@1 96%，Avg.Len 3.53。

V2A (Luo & Du 2025, https://arxiv.org/abs/2411.07223) goal-reaching policy 从 self-collected rollout + hindsight relabel 学，不需要 demo。

### 5.3 Trajectory-based interfaces

**(a) Affordance-based contact interface**：

VRB (Bahl et al. 2023, https://arxiv.org/abs/2304.08488) 从 egocentric human video 学 contact point + post-contact 2D trajectory。8 tasks, 2 robots。

SWIM (Mendonca et al. 2023, https://arxiv.org/abs/2308.10901) 把 affordance 作 latent world model 的 control input，CEM search。Boundary case（既是 explicit interface 又是 latent planning）。

**(b) 2D pixel trajectory**：

ATM (Wen et al. 2024a, https://arxiv.org/abs/2401.00025) 用 CoTracker (Karaev et al. 2024, https://arxiv.org/abs/2307.07614) 生成 pseudo-label，训 track transformer predict any-point 2D trajectory。Policy condition on 这些 track。LIBERO Sp/Ob/Go/Lo: 68.5/68.0/77.8/39.3%。

Tra-MoE (Yang et al. 2025) 加 Mixture-of-Experts (Shazeer et al. 2017, https://arxiv.org/abs/1701.06538) 处理 multi-domain video，sparsely-gated experts。

Im2Flow2Act (Xu et al. 2024, https://openreview.net/forum?id=cNI0ZkK1yC) object-centric 2D flow，exclude background。Sim-trained policy zero-shot transfer to real。

Track2Act (Bharadhwaj et al. 2024, https://arxiv.org/abs/2405.01527) 2D track → depth back-project → PnP fit rigid SE(3) transform → execute + residual policy。25 tasks, 5 locations。

**(c) 3D/6D structured trajectory**：

GeneralFlow (Yuan et al. 2024, https://arxiv.org/abs/2401.11439) 从 HOI4D RGB-D 学 3D object-point trajectory。Grounding：online track gripper-附近点，SVD-based alignment (Arun et al. 1987, https://ieeexplore.ieee.org/document/4767965) produce SE(3) update。Zero-shot human→robot。Real robot 18 tasks, 6 scenes, avg SR 81%。

SKIL-H (Wang et al. 2025, https://arxiv.org/abs/2501.14400) semantic 3D keypoint trajectory。Keypoint 通过 foundation feature clustering 发现 + descriptor matching 定位。Cross-embodiment study。

MimicPlay (Wang et al. 2023, https://arxiv.org/abs/2302.12422) 3D human hand trajectory from multi-view "human play"。Plan code condition 低 level policy。4 long-horizon tasks。

ZeroMimic (Shi et al. 2025, https://arxiv.org/abs/2503.23877) 6D wrist pose from egocentric video（EPIC-Kitchens）+ SfM camera param。Post-grasp policy。Real robot avg SR 71.9% (Franka, 9 skills), 65.0% (WidowX, 4 skills)。

### 5.4 Control integration 的四个 failure mode（§6.3）

1. **Tracking error problem**：kinematic singularity、unreachable target、self-collision、underdetermined visual→action mapping。GeneralFlow 的 SVD alignment 可能 produce 超出 joint limit 的 displacement；ZeroMimic 的 6D chunk 可能 place wrist 在 singularity boundary。
2. **Open-loop vs closed-loop**：UniPi/Dreamitate 是 open-loop（feedforward trajectory execution），SuSIE/CLOVER/Im2Flow2Act/GeneralFlow/Track2Act 是 closed-loop（look-and-move visual servoing 变体）。
3. **Interface-transfer pipeline fragility**：AVDC/RIGVid/Dreamitate/GVF-TAPE/Dream2Flow/Track2Act 都需要 segmentation、tracking、depth estimation、correspondence matching、rigid fitting 等步骤，每一步 error compounding。Classical cascade estimation 也有这问题，但 classical 能 cross-check intermediate，video pipeline 通常没法 verify。
4. **Hallucinated physics in generated plans**：generative video model 最大化 visual likelihood 不保证 contact mechanics、geometry（penetration）、dynamics（instantaneous acceleration）。dense video plan method 尤其 acute。RIGVid 的 VLM-based rollout selection 部分缓解，但 systematic physical-consistency checking 仍 open。

Intuition：explicit interface 的 trade-off 是 **inspectability 和 modular transfer 换 grounding pipeline 的 fragility**。Classical visual servoing 早就有类似 issue，但 learned pipeline 还没建立 robust verification 机制。

---

## 6. Cross-Family Synthesis（§8）

### 6.1 三条 cross-family design axis

**Axis 1: Interface location & explicitness**。Direct implicit in shared feature；Latent 是 structured 但 opaque；Explicit 是 human-interpretable。在同 family 内部也连续：dense video plan → subgoal image → trajectory → pose（从 rich context 到 precise target）。

**Axis 2: Training factorization**。Direct 通常 joint / interleaved on mixed data；Latent 和 Explicit 是两阶段（先 action-free 学 interface，再 ground 到 robot）。Recurring tension：**predictive ≠ realizable under robot kinematic/contact/embodiment constraint**。Video 给的是 scalable prior over scene change，必须 careful ground。

**Axis 3: Temporal abstraction & planning horizon**。Direct absorb temporal structure into policy，planning implicit；Latent 最适合 abstraction（discrete/continuous latent transition 作 compact unit）；Explicit 最 transparent，但当前 system 通常 short-horizon replan，long-horizon 仍 open。

### 6.2 Control-loop closure 的差异（§8.2）

| Family | Execution loop | Physical feasibility | Failure detection | Embodiment mismatch |
|---|---|---|---|---|
| Direct | Stepwise/chunked/receding, no intermediate | Implicit via action distribution | Opaque, hard to localize | Action head 紧 tie embodiment |
| Latent | MPC search or latent policy | Forward model 可能 visually predictive 但 dynamic invalid | Indirect rollout discrepancy | Grounding 对 distribution shift 敏感 |
| Explicit | Open-loop plan 或 closed-loop tracking | Predicted target 可能 kinematically unreachable | Discrepancy 可 measure, trigger replan | Interface-domain gap（human hand kinematic vs robot） |

### 6.3 Open challenges 四个 cluster（§8.3）

**1. Execution-aware and physically grounded learning**：video prediction objective 究竟 constrain 什么不明。可能 gain 来自 visual representation、regularization、generative prior、action-video alignment 任意组合。Latent/explicit 都有 predictability vs controllability tension。Future：augment predictor with feasibility signal（constraint violation、uncertainty）、incorporate physics prior、couple temporal-abstraction learning to execution feedback、multi-resolution interface（high-level visual context + short-horizon verifiable target）。

**2. Robust grounding and cross-embodiment transfer**：两方面。第一，分离 controllable 和 exogenous dynamics——multi-view constraint、ego-motion compensation、counterfactual objective、multi-agent factorization。第二，efficient embodiment adaptation——lightweight adapter、shared action representation、retargeting with explicit embodiment constraint。

**3. Multimodal sensing and contact-rich manipulation**：当前 method 几乎都 vision-only + rigid/quasi-rigid task。Contact-rich assembly、deformable manipulation、force modulation 需要 non-rigid state（dense 3D flow、keypoint field）+ tactile/proprioceptive feedback。RH20T (https://arxiv.org/abs/2307.00595) 等 force dataset 是起点。Joint video-tactile representation learning 是 open problem（video 缺 force，force data 缺 diverse video）。

**4. Evaluation, verification, safe deployment**：method 间 benchmark、pretraining corpora、robot-data scale、modality、protocol 都不同，cross-method comparison 几乎不可能（Tables 4/7/10 都标注 "Not directly comparable"）。需要：control for robot-data budget + modality + task difficulty；metrics beyond success rate（robustness to perturbation、recovery、uncertainty calibration）；standardized real-robot benchmark for long-horizon 和 contact-rich。Verification hooks：lightweight module screen predicted interface for feasibility、estimate confidence、trigger replanning or safe fallback。

---

## 7. 从 Karpathy 视角的几个 meta-observation

**A. "Video pretraining transfers how" 的 empirical evidence 还很弱**。Tables 4/7/10 显示大部分 method 的 quantitative evidence 是 within-method ablation（with vs without pretraining），cross-method leaderboard 几乎不存在。这个 field 还在 "证明 video 有用" 的阶段，没到 "证明哪种 video 用法最好" 的阶段。这一点和 LLM pretraining 早期很像——scaling law 还没建立。

**B. Latent-action identifiability 是这个 field 的 "alignment problem"**。学出来的 $z_t$ 不保证 controllable。这是因果推断里的 ICA-style identifiability 问题（https://arxiv.org/abs/2207.09141），但 video transition 里 multiple cause 同时变化，bottleneck 不够 enforce disentanglement。UniVLA 的 task-centric/task-irrelevant 分离是 patch，但只是 approximate。

**C. Explicit interface 的 transfer pipeline 是新的 "sim-to-real"**。AVDC、Dream2Flow 这类方法把 generative video → segmentation → tracking → depth → SE(3) 串成 pipeline。每一步都是 learned module，每一步都有 error。这本质上是把 classical perception-control pipeline 的 geometric engine 换成 learned module，但失去了 classical 的 verifiability。这个 gap 没人系统研究。

**D. Execution mode orthogonal to training architecture**。Direct family 里同 architecture 可以 stepwise / chunked / receding / feature-conditioned / latent-rollout。这个 decoupling 意味着 deployment-level control theory 还需要单独研究，不能假设 end-to-end learned policy 自动有好的 control property。

**E. Closed-loop vs open-loop 的回归**。Classical robotics 几十年建立 closed-loop visual servoing 的稳定性 theory（https://link.springer.com/article/10.1023/A:1010046902645），learned method 又回到了 open-loop video plan（UniPi、Dreamitate）。这个 regression 是因为 generative model latency 高没法 closed-loop。UVA、VPP、ATM、GeneralFlow 这类 bypass video generation 或 lightweight predictor 是回到 closed-loop 的路径。

---

## 8. 一些我没在 paper 里看到但 related 的方向

1. **Diffusion policy 的 receding-horizon 重新理解**。Chi et al. 2023 (https://arxiv.org/abs/2303.04137) 的 diffusion policy 本身就是 receding-horizon，但 paper 没把它和 PAD/UWM/UVA 这类 joint video-action diffusion explicit connect。PAD 的 receding-horizon execution 实际上是 diffusion policy + video prediction 的自然组合。

2. **V-JEPA / V-JEPA 2 (Bardes et al. 2025, https://arxiv.org/abs/2506.07947)** 是 LeCun 路线的 video representation learning，完全 non-generative，joint-embedding predictive architecture。Paper 排除了 "non-generative predictive objective" 类的方法（inclusion criteria (i) 要求 temporal prediction 但没明确排除 JEPA）。但 JEPA 学的 representation 是否适合 manipulation grounding 是 open question。

3. **World model + RL 的 scaling**。DreamerV3 (Hafner et al. 2023, https://arxiv.org/abs/2301.04104) 在 Minecraft diamond 上证明 world model + RL 能 scale，但 APV/ContextWM 这类 action-free pretrain → action-conditional RL 的范式还没在 manipulation 上 scale 到这种程度。

4. **VLM as world model for planning**。 paper 排除纯 VLA（RT-2 等），但最近 Socratic Models (Zeng et al. 2022)、Code as Policies (Liang et al. 2022)、SayCan (Ahn et al. 2022) 这类用 LLM 做 high-level planner 的 method 和 explicit interface 的 hierarchical decomposition 在 spirit 上一致，可以视作 explicit interface 的 language-code 变体。

5. **Cross-embodiment 的 retargeting theory**。DexMimicGen (Wang et al. 2024, https://arxiv.org/abs/2407.07788)、Any-point Trajectory (https://arxiv.org/abs/2401.00025)、HumanPlus (https://arxiv.org/abs/2406.10454) 等 humanoid retargeting 工作其实在做 explicit interface family 的 grounding 那一步，但 paper 没系统连接。

---

## 9. 总结的 intuition

这篇 survey 真正的 contribution 不是 catalog method，而是提出一个 **"robotics integration layer" thesis**：当前 video-based manipulation 的 bottleneck 不在 representation learning（video prediction model 越来越强），而在 **如何把 video-derived prediction 接入 closed-loop control 同时保持 physical feasibility、verifiability、cross-embodiment transfer**。

三个 family 在这条 axis 上的 trade-off：

- **Direct** 牺牲了 inspectability 换 simplicity，但 deployment 失败无法 localize；
- **Latent** 引入 structured intermediate 但 identifiability 不保证，grounding 在 distribution shift 下 brittle；
- **Explicit** 提供 inspectable target 但 transfer pipeline compounding error + physics hallucination 是新 bottleneck。

paper 最后指出四个 open direction（execution-aware learning、robust grounding、multimodal sensing、evaluation infrastructure）实际上对应 robotics integration layer 的四个 sub-problem。我觉得这个 framing 比 "learning from video" 这个 high-level category useful 得多——它强迫 method paper 不仅 report success rate，还要说明 failure mode 在哪、能不能 verify、transfer 到新 embodiment 怎么办。

对 build intuition 的 takeaway：**video 是 world dynamics 的 scalable observation，但 observation ≠ control signal**。中间必有一个 interface design 决定 dynamics knowledge 如何 enter control loop，而这个 design 的 trade-off 是 classical robotics 已经研究过的问题（visuomotor vs MPC vs planner-controller hierarchy），只是 learned method 加入了 scalability 和 flexibility，同时失去了 classical 的 verifiability。下一个十年的工作是把 verifiability 加回来。

---

一些相关 reference 链接汇总：

- **Survey 本身**: 此次提供的 paper
- **Direct family**: GR-1 (https://arxiv.org/abs/2312.06517), GR-2 (https://arxiv.org/abs/2410.06158), PAD (https://arxiv.org/abs/2411.18179), UWM (https://arxiv.org/abs/2506.08812), UVA (https://arxiv.org/abs/2503.00200), VidMan (https://arxiv.org/abs/2411.09153), VPP (https://arxiv.org/abs/2412.14803), APV (https://arxiv.org/abs/2203.13880), ContextWM (https://arxiv.org/abs/2305.18499)
- **Latent family**: CLASP (https://arxiv.org/abs/1806.09655), FICC (https://openreview.net/forum?id=Sy-o2N0hF4f), LAPO (https://arxiv.org/abs/2312.10812), Genie (https://arxiv.org/abs/2402.15391), LAPA (https://arxiv.org/abs/2410.11758), UniVLA (https://arxiv.org/abs/2506.06269), VQ-VAE (https://arxiv.org/abs/1711.00937)
- **Explicit family**: UniPi (https://arxiv.org/abs/2306.17672), Gen2Act (https://arxiv.org/abs/2409.16283), AVDC (https://arxiv.org/abs/2310.08576), RIGVid (https://arxiv.org/abs/2507.00990), Dreamitate (https://openreview.net/forum?id=InT87E5sr4), GVF-TAPE (https://arxiv.org/abs/2509.00361), Dream2Flow (https://arxiv.org/abs/2512.24766), SuSIE (https://arxiv.org/abs/2310.10639), CLOVER (https://arxiv.org/abs/2409.09016), V2A (https://arxiv.org/abs/2411.07223), VRB (https://arxiv.org/abs/2304.08488), SWIM (https://arxiv.org/abs/2308.10901), ATM (https://arxiv.org/abs/2401.00025), Im2Flow2Act (https://openreview.net/forum?id=cNI0ZkK1yC), Track2Act (https://arxiv.org/abs/2405.01527), GeneralFlow (https://arxiv.org/abs/2401.11439), SKIL-H (https://arxiv.org/abs/2501.14400), MimicPlay (https://arxiv.org/abs/2302.12422), ZeroMimic (https://arxiv.org/abs/2503.23877)
- **Datasets**: Ego4D (https://ego4d-data.org/), EPIC-Kitchens (https://epic-kitchens.github.io/2024), HOI4D (https://hoi4d.github.io/), OXE (https://robotics-transformer-x.github.io/), DROID (https://droid-dataset.github.io/), CALVIN (https://calvinrobot.github.io/), LIBERO (https://lifelong-robot-learning.github.io/libero/)
- **Classical foundations**: DreamerV3 (https://arxiv.org/abs/2301.04104), Diffusion Policy (https://arxiv.org/abs/2303.04137), EPnP (https://link.springer.com/article/10.1007/s11263-008-0152-6), RANSAC (https://dl.acm.org/doi/10.1145/358669.358692), V-JEPA 2 (https://arxiv.org/abs/2506.07947)
