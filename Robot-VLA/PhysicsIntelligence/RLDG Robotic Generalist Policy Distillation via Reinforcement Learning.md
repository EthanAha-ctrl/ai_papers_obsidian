---
source_pdf: RLDG Robotic Generalist Policy Distillation via Reinforcement Learning.pdf
paper_sha256: 040579da5b9f77e9ecebf9e38f2be94de26b1b57b74ab8fdd11fd2b7d6147311
processed_at: '2026-08-12T00:01:16-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 RLDG

## 一句话版本

**与其让 Foundation model 艰难地从人类 shaky 的 teleoperation demo 里猜最优 action，不如先让 RL policy 在这个 task 上"刷到满分"，再把它的"完美答案"抄给 Foundation model。**

---

## 为什么这件事 obvious 但没人好好做

现在 robot foundation model 这条路线（OpenVLA、Octo、RT-2、π0 这些），pipeline 基本都是：

1. Pretrain 一个 huge model on internet data（学 visual features、language understanding）
2. Fine-tune on robot demos（学怎么 output action）
3. 部署

第 2 步的 data，几乎全世界都在用 human teleoperation demos。SpaceMouse 摇杆，研究生坐在 robot 前面，record 几百条 trajectory。这个 workflow 有几个根本性的 pain point，特别是 precise manipulation task 上：

**Pain point 1：Human 本身就做不好这些 task**

比如 USB connector insertion，sub-millimeter precision，contact-rich。你让一个研究生 teleop 100 次，他可能有 30 次都卡住了，好不容易成功的 70 次里，action 也是 "晃晃悠悠地试探"。这个 "晃晃悠悠" 的 action distribution 进到 7B VLA model 里，model 要学习一个 multi-modal distribution，但这个 multi-modal 里大部分 mode 其实是 noise / suboptimal。

**Pain point 2：Human action distribution centered around "safe middle"**

你看 paper 的 Figure 8，human 在 critical state 附近，action 分布是在 action space 中心附近一团，slightly biased 朝正确方向。意思是 human 在这个 position 时"不太确定该往哪走，先小幅动一下试试"。RL policy 则 commit 到 correct corner——它确切知道该往哪个方向走，而且走多远。

**Pain point 3：收集 human demo 很贵**

一个 graduate student 一天能 record 多少条成功 demo？在 precise manipulation task 上可能 50 条都难。而 RLDG 让 RL policy 自动 rollout，一晚上能 collect 几百条。

---

## RLDG 怎么做的

非常 simple，甚至 simple 到让你怀疑 "这也能发 paper？"。但 simple 往往是 best idea 的特征。

### Stage 1：Train specialist RL policy

对每个 task，用 HIL-SERL（Levine group 自己的 real-world RL framework）训练一个 RL policy。这个 RL policy：
- Input：128×128 wrist image + robot proprioception
- Output：6D end-effector delta pose + gripper command
- Reward：用 human teleop 的 positive/negative samples 训练一个 binary success classifier
- Training time：1-3 小时，达到 100% success rate

为什么 RL 能在 1-3 小时收敛到 100%？因为 HIL-SERL 用了 human intervention（训练时人可以 step in prevent catastrophic state）+ RLPD（prior demo data seeding）+ sample-efficient RL algorithm。这是 Levine group 过去几年积累的 real-world RL engineering 的集大成。Reference: https://arxiv.org/abs/2410.21845

### Stage 2：用 RL policy rollout 生成 data

RL policy 收敛后，让它自己跑，collect trajectories。因为 RL policy 是 100% success rate 的，所以 collect 的几乎都是 successful trajectory。直接作为 fine-tuning dataset。

### Stage 3：Fine-tune generalist policy

拿这个 RL-generated dataset，用 standard supervised learning（behavioral cloning objective）fine-tune OpenVLA 或 Octo。

$$\mathcal{L}(\theta) = -\mathbb{E}_{(s_t, a_t) \sim D} [\log \pi_\theta(a_t | s_t)]$$

这里 $s_t$ 是 state（image + proprioception），$a_t$ 是 action label（来自 RL policy 而非 human），$\pi_\theta(a_t|s_t)$ 是 generalist policy 输出 action $a_t$ 的 probability，$\theta$ 是 network parameters，$D$ 是 RL rollout 构成的 dataset。Loss 就是 negative log-likelihood，minimize 它就是让 policy 更 likely 输出 RL policy 的 action。

就这三步。没有 fancy 的 distillation loss，没有 GAN-style adversarial training，没有 feature matching。就是 "用 RL policy 的 output 作为 ground truth label 去 train foundation model"。

---

## 为什么这么 simple 的 idea 效果这么好

### 直觉解释 1：RL data 是 "purified" 的 human data

Human demo 里包含了 human 的 hesitation、mistake、exploration。RL policy 通过 reward maximization 把这些 noise 都洗掉了，只保留 "state → optimal action" 的 clean mapping。Foundation model 学这种 clean mapping 当然比学 noisy mapping 容易。

### 直觉解释 2：RL policy 相当于给 Foundation model 提供了 "标准答案"

想象你教一个 student 解数学题。方法 A：给 student 看一堆人解题的过程，有些人解对了，有些人解错了，有些人绕了弯路。方法 B：先让一个 expert 把每道题都解出来，然后把 expert 的解法给 student 学。方法 B 当然更 sample efficient。

### 直觉解释 3：Generalist pre-training 提供了 "semantic scaffolding"，RL data 提供 "precise execution"

OpenVLA 在 970k Open X-Embodiment demos 上 pre-trained，它已经 understand "USB connector 长什么样"、"insertion 大概是个什么动作"、"gripper 应该怎么 grasp"。它缺的是 "在精确到 0.5mm 的 alignment 上，action 应该 commit 到哪个具体方向"。RL data 恰好提供了这个 precise execution knowledge。两者 complementary。

---

## 实验数据到底有多 convincing

### Headline number

在 precise manipulation task（connector insertion、FMB insertion）上，RLDG 比 human demo fine-tuning：
- Success rate 平均高 30-50%
- Data efficiency 高 6-10 倍
- 在 unseen scenario 上 gap 更大（2× 以上）

### 最 striking 的实验：Scaling analysis（Figure 5）

这个实验我觉得是整篇 paper 最 valuable 的数据。在 VGA connector（seen）和 Type-C connector（unseen）上：

| Data Source | 达到 100% success 需要的 episodes | 900 episodes 时 success rate |
|-------------|-----------------------------------|---------------------------|
| RL-generated | 45 episodes | 100% |
| Human demos | 300 episodes（6.7× more） | plateau at 90% |

在 unseen Type-C 上，这个 gap 更极端：RL data 用 150 episodes 就能达到 100%，而 human data 用 900 episodes（20× more）还卡在 90%。

这个 result 说明了什么？Human demo 的 information content 在这个 task 上有 ceiling——无论你 collect 多少 human demo，foundation model 都学不到 100% success，因为 human demo 本身就 suboptimal。而 RL data 的 information content 更高，少量 data 就能让 model 学到 optimal behavior。

### 另一个 striking 的实验：Generalization vs RL policy（Figure 4）

RL policy 在 training scenario 上 20/20，但在 unseen scenario 上崩到 1/20。这是 RL 的经典问题——overfit to training distribution。

但 RLDG fine-tuned 的 OpenVLA，在 unseen connector 上能 73/80。为什么？

因为 foundation model 的 internet-scale pre-training 提供了 visual 和 semantic 的 generalization prior。RL policy 只 see 过 USB connector，它的 visual encoder 是 from-scratch 的 small CNN，没有 "connector 这个 concept 长什么样" 的 prior。Foundation model 在 millions of images 上 pre-trained，它 know connector 的 visual variability，RL data 只是 teach 它 precise action strategy。

**RLDG = RL 的 precision + Foundation model 的 generalization。** 这是 paper 的核心 thesis，数据 support 了这个 thesis。

---

## 最有 insight 的 ablation：Why RL data is better

Section 5 的 ablation 实验很 clever。他们构造了三种 dataset：

1. **Human**：纯 human demo
2. **RL**：纯 RL rollout
3. **Human + RL actions**：human demo 的 states（视觉、proprioception），但 action label 用 RL policy 重新标注

第 3 种是 key experimental design。它 isolate 了两个 factor：
- State distribution：human 和 RL 看到的 states 是否不同
- Action quality：human 和 RL 输出的 action 是否不同

结果：**"Human + RL actions" 比 "Human" 提升 50%+，但仍略低于 "RL"**。这说明 action quality 是主因，state distribution 是次因。

换句话说：**RL data 之所以好，主要是因为 RL policy 的 action 更 optimal，而不是因为 RL policy 探索了不同的 states。**

这个 insight 对 future work 有指导意义。如果你想让 RLDG 更好，应该 focus on 让 RL policy 的 action 更 optimal（比如 better reward shaping、longer training、better exploration），而不是 focus on 让 RL policy 探索更多 diverse states。

---

## 关于 Architecture 的一个观察

OpenVLA 和 Octo 在这个 benchmark 上表现差异挺大的。OpenVLA 在 precise task 上 success rate 明显更高，Octo 整体偏低。

我怀疑这跟 action representation 有关。OpenVLA 用 256 bins discretization + autoregressive prediction，本质是分类问题，在 precise action 上可能更容易学 sharp distribution。Octo 用 diffusion head，设计初衷是 model multi-modal human action distribution，但 RL data 本身是 unimodal 的（RL policy 近似 deterministic），用 diffusion head 来 model unimodal data 可能 overkill，反而 harder to optimize。

这个 paper 没有深入讨论 architecture 和 data source 的 interaction。如果是我做 follow-up，会试试：
- Octo + Gaussian MSE head（而非 diffusion head）+ RL data，看是否能 match OpenVLA
- OpenVLA + continuous action head + RL data，看是否能进一步提升 precision

这个 architecture-data co-design 的 question 其实挺 open 的。

---

## 对 Foundation Model 领域的 broader implication

### Implication 1：Data quality > Data quantity

这是 RLDG 最核心的 message。在 LLM 领域，这个 insight 已经被反复验证（"data quality is all you need"，Phi series、LIMA paper 都是证据）。现在这个 insight 被 extend 到 robot foundation model。

未来 robot foundation model 的 competitive advantage，可能不在于谁有更多 human demos（Open X-Embodiment 已经 970k 了），而在于谁能 generate 更 high-quality 的 fine-tuning data。RL 是一个 automated quality improvement mechanism。

### Implication 2：RL 的角色从 "deployment" 变成 "data generation"

传统上 RL 被视为 deployment-time 的 algorithm（policy 直接 deploy）。但 RLDG 把 RL 的角色降级为 "data generator"——RL policy 不直接 deploy，它只负责生成 data 给 foundation model 学。

这个 shift 有几个好处：
- RL policy 可以是 small、task-specific、non-generalizable 的，反正它不需要 deploy
- RL training 可以在 narrow scope 上快速收敛（1-3 小时），avoid long-horizon credit assignment
- Foundation model 负责 generalization，RL 负责 specialization，分工清晰

### Implication 3：Self-improving loop 的雏形

RLDG 的 pipeline 可以想象成 self-improving loop 的一个 cut：
1. Foundation model deployed
2. Collect real-world interaction data
3. 用 reward function identify successful trajectories
4. 这些 trajectories 反过来 fine-tune foundation model（RLDG-style）
5. 循环

Paper 没有做 close the loop 的实验，但 infrastructure 和 methodology 已经有了。如果结合 automated reward generation（比如 VLM-based reward），这个 loop 可以 fully autonomous。

### Implication 4：RL fine-tuning foundation model 仍然 open

Paper 明确提到，直接用 RL fine-tune foundation model 是 "largely an open problem"。为什么？

- 7B parameter 的 VLA model，RL gradient propagation 很不稳定
- Online RL 需要大量 real-world interaction，expensive and risky
- Catastrophic forgetting of pre-trained capabilities 是 real risk
- Value function learning on 7B model 在 diverse data 上很难 scale

所以 RLDG 这种 indirect approach（RL 先 narrow 训练，再 distill）是 pragmatic 的 workaround。但如果未来有人 solve 了 stable RL fine-tuning of large VLA model，那 RLDG 可能会被 obsolete。不过在那之前，RLDG 是一个非常 practical 的 solution。

---

## 最终 takeaway

如果让我给 Karpathy 你 summarize 这篇 paper 的 contribution，我会说：

**RLDG 提供了一个 simple、practical、effective 的方法，把 RL 的 precision 和 foundation model 的 generalization 结合起来。核心 insight 是 RL policy 可以作为 automated high-quality data generator，替代 expensive 且 suboptimal 的 human teleoperation。这个 idea 虽然 retrospectively obvious，但实验数据 convincing 地证明了 data quality 的提升是 structural 的，不是 trick。**

对 robot foundation model 领域，这个 work 暗示了一条 path：未来的 competitive 优势可能在于 automated data generation pipeline（RL、simulation、video generation 等），而非单纯堆 human demos。这跟 LLM 领域从 "more data" 转向 "better data" 的趋势一致。

References:
- RLDG project page: https://generalist-distillation.github.io/
- HIL-SERL: https://arxiv.org/abs/2410.21845
- OpenVLA: https://arxiv.org/abs/2406.09246
- Octo: https://arxiv.org/abs/2405.12213
- FMB benchmark: https://arxiv.org/abs/2401.08553
- RLPD: https://proceedings.mlr.press/v202/ball23a.html
- LIMA (data quality in LLM): https://arxiv.org/abs/2305.11206
- Phi series: https://arxiv.org/abs/2306.11644

---

# RLDG: 通过 Reinforcement Learning 蒸馏 Robotic Generalist Policy

## 1. Paper 的核心 Motivation 与 Positioning

这篇来自 UC Berkeley Levine group 的工作，触及了一个当前 robot foundation models 领域的关键 tension：**generalist policy 的 fine-tuning data quality 决定了其 task-specific performance 的上限**。目前主流的 OpenVLA、Octo、RT-2、π0 这类 VLA model 都依赖 human teleoperation demos 做 fine-tuning，但 human demos 在 contact-rich precise manipulation（比如 USB connector insertion 这种 sub-millimeter precision 任务）上存在固有缺陷：

- Human action distribution 倾向于 "safe but suboptimal"，centered around 中间区域，缺乏 exploratory 的 commit
- 执行 style 不一致，导致 policy 要 modeling multi-modal 但其实多模态里大部分都是 noise
- Contact-rich 场景下 human 本身就难以 demonstate 出最优的 compliance 和 force strategies

RLDG 的 insight：与其直接用 RL 去 fine-tune foundation model（这在 7B param 的 VLA 上面临 optimization instability、catastrophic forgetting、compute cost 等问题），不如让 specialist RL policy 先在 narrow task 上收敛到 100% success rate，然后用它的 rollouts 作为 "purified" demonstration data 去 supervise fine-tune generalist。这本质上是一种 **asynchronous distillation**：teacher（RL policy）和 student（generalist）不必同时存在，通过 data 作为 medium 传递 knowledge。

Project page: https://generalist-distillation.github.io/

## 2. Method 细节拆解

### 2.1 RL Training Stage (HIL-SERL)

每个 task 被 formulate 为 MDP $(S, A, T, R, \rho_0, \gamma)$：
- $s_t$：128×128 wrist RGB image + end-effector pose + velocity + wrench（force/torque measurement）
- $a_t$：6D end-effector delta pose in wrist frame + 1 binary gripper action
- $\rho_0$：initial robot configuration distribution
- $R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$：reward function，用 binary success classifier 实现（先 teleop 采集 positive/negative samples 训练 classifier）

Policy objective 是标准 discounted return maximization：

$$J(\pi) = \mathbb{E}_{\substack{s_0 \sim \rho_0 \\ a_t \sim \pi(a_t|s_t)}} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right]$$

**变量解释**：
- $J(\pi)$：policy $\pi$ 的 expected cumulative discounted return，这是要 maximize 的 objective
- $\mathbb{E}$：expectation over trajectories
- $s_0 \sim \rho_0$：initial state 从 initial state distribution $\rho_0$ 中采样
- $a_t \sim \pi(a_t|s_t)$：action 从 stochastic policy $\pi$ 在 state $s_t$ 下的分布中采样
- $T$：episode 的 horizon length
- $\gamma \in [0,1)$：discount factor，越靠后的 reward 贡献越小，这会 induce policy 倾向于更快完成任务（这也是 cycle time 改善的来源）
- $R(s_t, a_t)$：reward function，在 state $s_t$ 执行 $a_t$ 后获得的即时 reward

实现用的是 HIL-SERL（Human-in-the-Loop Sample Efficient Reinforcement Learning），reference: https://arxiv.org/abs/2410.21845。这个 framework 的核心是结合 RLPD（Reinforcement Learning with Prior Data，https://proceedings.mlr.press/v202/ball23a.html）允许 offline demonstrations 和 online interaction 混合训练，加上 human intervention 机制防止 catastrophic states。训练 1-3 小时就能达到 100% success rate，这是 real-world RL 的 sample efficiency 突破。

### 2.2 Experience Collection Stage

Converged RL policy 用于 rollout 生成 fine-tuning dataset。关键设计：
- **Multi-policy balancing**：Connector Insertion 任务中，USB/Ethernet/VGA 各训练独立的 RL policy，rollout 时每个 connector 收集 equal number of episodes，避免 dataset imbalance
- **Long-horizon decomposition**：FMB Assembly 任务中，RL 只训练 insertion 这个 "bottleneck" 阶段，grasping 和 transport 阶段用 human demos。这种 decomposition 降低了 RL 训练 complexity（避免 long-horizon credit assignment 问题），同时保留了 human demos 在 non-critical 阶段的 diversity
- **Failed trajectory filtering**：只保留 successful rollouts 进入 fine-tuning set

### 2.3 Generalist Fine-tuning Stage

给定 pre-trained policy $\pi_0$（在 Open X-Embodiment 上预训练），用 supervised learning objective fine-tune：

$$\mathcal{L}(\theta) = -\mathbb{E}_{(s_t, a_t) \sim D} [\log \pi_\theta(a_t | s_t)]$$

**变量解释**：
- $\mathcal{L}(\theta)$：negative log-likelihood loss，要 minimize
- $\theta$：policy network 的 parameters
- $\mathbb{E}_{(s_t, a_t) \sim D}$：expectation over dataset $D$ 中的 $(s_t, a_t)$ pairs
- $\pi_\theta(a_t|s_t)$：parameterized policy 在 state $s_t$ 下输出 action $a_t$ 的 probability（或 density）
- $D$：fine-tuning dataset，由 RL rollouts 构成

这就是标准的 maximum likelihood imitation learning / behavioral cloning objective，但 data source 从 human demos 换成了 RL rollouts。

## 3. 两个 Generalist Architecture 解析

### 3.1 OpenVLA

OpenVLA（https://arxiv.org/abs/2406.09246）是一个 7B parameter 的 VLA model，backbone 是 Llama 2（https://arxiv.org/abs/2307.09288）：

```
[Image 224×224] → Vision Encoder (DINOv2 + SigLIP fusion) → Visual Tokens
                                                                    ↓
[Language Instruction] → Tokenizer → Text Tokens → Llama 2 Backbone → Action Tokens
                                                                    ↓
                                                    7D action (256 bins/dim, autoregressive)
```

- **Action representation**：7-dimensional action（这里用 6D delta pose + 1 gripper），每个 dimension 离散化成 256 bins，用 autoregressive next-token prediction + cross-entropy loss 训练
- **Fine-tuning 方式**：LoRA（Low-Rank Adaptation，https://arxiv.org/abs/2106.09685）rank=32，应用于每个 linear layer。LoRA 的原理是 freeze 原 weight $W_0$，添加 low-rank update $\Delta W = BA$，其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$。这样 trainable parameters 大幅减少（约 1%），但 expressivity 仍然足够 task adaptation
- **Inference frequency**：4Hz（相比 RL policy 的 10Hz，这是 cycle time 差距的主要来源之一）
- **Training config**：batch size 2，gradient accumulation 3（effective batch 6），Nvidia RTX 4090，3-5 小时收敛

### 3.2 Octo

Octo（https://arxiv.org/abs/2405.12213）是另一个 open-source generalist，架构不同：

```
[Wrist Image 128×128] → Primary Image Tokenizer → Observation Tokens
                                                              ↓
[Language Instruction] → Tokenizer → Goal Tokens → Transformer Backbone → Readout Embedding e
                                                              ↓
                                              Diffusion Head (DDPM) → Continuous Action
```

- **Action representation**：continuous action，用 diffusion head（DDPM，https://arxiv.org/abs/2006.11239）建模 multi-modal action distribution。Diffusion policy 的优势在于能 naturally 表达 multi-modal action distribution，这对于 human demos 这种 inherently multi-modal 的 data 特别有用（reference: Diffusion Policy https://arxiv.org/abs/2303.04137）
- **Diffusion head 机制**：给定 readout embedding $e$ 作为 condition，denoising process 从 $\mathbf{a}_T \sim \mathcal{N}(0, I)$ 出发，通过 $T$ 步去噪得到 action $\mathbf{a}_0$。每一步 $\mathbf{a}_{t-1} = \frac{1}{\sqrt{\alpha_t}}(\mathbf{a}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(\mathbf{a}_t, t, e)) + \sigma_t \mathbf{z}$，其中 $\epsilon_\theta$ 是 noise prediction network，$\alpha_t, \bar{\alpha}_t$ 是 noise schedule，$\mathbf{z} \sim \mathcal{N}(0, I)$
- **Fine-tuning 方式**：full fine-tuning（不是 LoRA），batch size 64，RTX 4090，3-5 小时
- **Modality 适配**：移除 secondary image tokenizer，mask out image goal，只保留 wrist camera 作为 visual input

## 4. Experiment Setup 详细配置

### Hardware
- **Robot arm**：Franka Emika Panda
- **Gripper**：parallel jaw gripper
- **Low-level control**：1kHz impedance controller（确保 action execution 的 fidelity）
- **Wrist camera**：Intel RealSense D405，提供 128×128 RGB image 作为 policy observation
- **Teleoperation device**：3Dconnexion SpaceMouse（用于 human demos 和 RL intervention）

### Tasks Breakdown

| Task | Precision Requirement | Training Objects | Generalization Test Objects | Randomization |
|------|----------------------|-----------------|---------------------------|---------------|
| Connector Insertion | sub-millimeter | USB, Ethernet, VGA | Type-C, HDMI, DisplayPort, 3-pin XLR | 10cm × 10cm plane, 5cm above port |
| Pick and Place | centimeter | green pepper | yellow corn + beige wood background | 18cm × 18cm object placement |
| FMB Insertion | ±1.5mm tolerance | single FMB object | - | 35cm × 35cm + ±15° rotation |
| FMB Assembly | ±1.5mm tolerance + grasping | single FMB object | - | 3cm × 7cm grasp area + 5cm × 5cm insertion area |

## 5. Results 核心数据表

### 5.1 Success Rate Comparison（Figure 4 数据）

| Task | Method | OpenVLA Success | Octo Success | RL Policy Success (train/unseen) |
|------|--------|-----------------|--------------|----------------------------------|
| FMB Insertion | Human Demo | baseline | baseline | 20/20 → 1/20 |
| FMB Insertion | RLDG | +33% | +10% | - |
| Connector Insertion | Human Demo | lower | lower | - |
| Connector Insertion | RLDG | +23% | +37% | - |
| Pick and Place | Human Demo | 16/20 | 1/20 | 20/20 → 1/20 |
| Pick and Place | RLDG | 19/20 | 4/20 | - |
| FMB Assembly | Human Demo | 12/20 | - | N/A (RL only on insertion) |
| FMB Assembly | RLDG (hybrid) | 20/20 | - | - |
| Connector Insertion (unseen) | Human Demo | low | low | - |
| Connector Insertion (unseen) | RLDG | 2× higher | - | - |
| Pick and Place (unseen) | Octo Human | 0/20 | - | - |
| Pick and Place (unseen) | Octo RLDG | 4/20 | - | - |

### 5.2 Scaling Analysis（Figure 5 核心数据）

| Data Source | VGA (seen) Episodes for 100% | Type-C (unseen) Success at 150 episodes | Type-C (unseen) Success at 900 episodes |
|-------------|------------------------------|------------------------------------------|------------------------------------------|
| RL-generated | 45 episodes | 100% | 100% |
| Human demos | 300 episodes | <100% | plateau at 90% |

**这个 scaling result 是 paper 最 striking 的结论之一**：RL data 达到 100% success 只需 45 episodes，而 human data 需要 300 episodes（6.7× more），且在 unseen Type-C 上 RL data 仍能达到 100%，而 human data 即使 900 episodes（20× more）也只能 plateau 在 90%。这说明 RL data 的优势随着 data size 增长不会消失，而是 structural advantage。

### 5.3 Multi-connector Generalization（Figure 4 数据）

在 4 个 unseen connectors（Type-C, HDMI, DisplayPort, 3-pin XLR）上，每个评估 20 次（共 80 trials）：

| Method | Success / 80 |
|--------|--------------|
| Best single-connector RL policy | 49/80 |
| OpenVLA + RLDG (trained on USB/Ethernet/VGA) | 73/80 |
| Octo + RLDG | 50/80 |

这证明了 generalist policy 的 multi-task pre-training 提供的 semantic prior 能 help generalization，而 RL data 提供的 optimal action distribution 能 improve performance，两者结合超过纯 RL policy 的泛化能力。

## 6. 为什么 RL Data 更好？——核心 Analysis（Section 5）

这是 paper 最有 insight 的部分。作者设计了 ablation 实验 disentangle 两个 hypothesis：

### 6.1 Action Quality vs State Distribution Ablation

在 FMB Insertion task 上构造三种 dataset：
1. **Human**：纯 human demo trajectories
2. **RL**：纯 RL policy rollouts
3. **Human + RL actions**：human trajectories 的 states（视觉观察 + proprioception），但 action labels 用 RL policy 重新标注（relabel actions）

这个实验设计很精巧：state distribution 保持 human 的（更 diverse 但可能包含 "stuck" states），只替换 action labels。如果性能提升主要来自 state distribution，这个 mixed version 应该接近 human baseline；如果主要来自 action quality，应该接近 RL baseline。

**结果（Figure 7）**：在 25/50/75 trajectories 上，"Human + RL actions" 比 "Human" 提升 50%+，但仍略低于 "RL"。这说明 **action quality 是主要因素，state distribution 是次要因素**。

### 6.2 Action Distribution 可视化（Figure 8）

作者在 FMB insertion 的某个 critical state 附近（end-effector position 在某个 insertion 点附近，x/y within 4mm, z within 10mm），可视化 dataset 中所有 transitions 的 action distribution 的前两维（对应 x, y 方向的 delta movement）：

- **Human actions**：clustered around action space 的 center，slight bias towards correct direction（bottom-left）
- **RL actions**：更 concentrated 在 correct corner（bottom-left），即 RL policy 学会了在 critical state 下 commit 到正确的方向

这个 visualization 直观解释了为什么 RL data 更 sample efficient：generalist policy 在 critical region 不需要再 figure out 应该往哪个方向 move，RL data 直接告诉它 optimal action，而 human data 中 action 分布 dilute 了信号。

### 6.3 Qualitative Failure Modes

| Task | Human Demo Policy 失败模式 | RLDG Policy 改善 |
|------|---------------------------|------------------|
| Connector/FMB Insertion | "Stuck" state：contact board 但 alignment 失败，maintain contact pressure without exploratory movement；approach trajectory 过早 descent 导致 connector 卡在 socket lip | RL data 消除 stuck state，improved approach trajectory |
| Pick and Place | Premature gripper closure during grasping | Improved grasp reliability |
| Pick and Place (RL-specific failure) | - | 有时 drop object too early，bouncing out of bowl（RL 的 speed optimization 副作用）|
| FMB Assembly | Insertion 阶段 alignment 问题 | Grasping/transport 阶段 performance 相似（因为用相同 human data），insertion 阶段 RL data 明显更好 |
| Octo (general) | Grasping errors due to lack of depth perception | - |

**RL-specific failure mode 的讨论很有意思**：Pick and Place 中 RL policy 为了 maximize discounted return，倾向于在 object 一旦 clear bowl edge 就立即 release（faster completion = higher return），但 distilled policy 缺乏 precise timing，导致 early drop。这说明 RL 的 objective function（temporal discounting）虽然 induce speed，但可能不 induce robustness to distillation error。这是一个 honest limitation disclosure。

## 7. Related Work Context 与 Broader Implications

### 7.1 Policy Distillation 谱系

RLDG 在 policy distillation 传统中的位置：

| 方法 | Teacher → Student 关系 | RLDG 的 innovation |
|------|----------------------|-------------------|
| GPS (Levine & Abbeel 2014) | RL expert → neural net policy | 直接 distillation，无 foundation model |
| Policy Distillation (Rusu et al. 2015) | Multiple DQN experts → single net | 从 scratch 训练 student |
| Actor-Mimic (Parisotto et al. 2015) | Multi-task experts → student | 类似 distillation |
| Distral (Teh et al. 2017) | Bi-directional constraints | 介于 distillation 和 multi-task 之间 |
| Progressive Neural Networks (Rusu et al. 2016) | Columnar transfer | Avoid catastrophic forgetting |
| **RLDG** | **RL experts → pre-trained foundation model** | **Leverage large-scale pretraining；通过 data 而非 direct gradient transfer** |

RLDG 的关键区别在于：student 是已经在 internet-scale data 上 pre-trained 的 foundation model，所以 distillation 不需要从 scratch 学习 low-level visual features 和 semantic understanding，只需要 adapt action distribution。这解释了为什么 RLDG 能在 3-5 小时内完成 fine-tuning 而 from-scratch distillation 需要更长。

### 7.2 Foundation Models for Robotics Landscape

当前 robot foundation models 的几个 axis：

| Model | Size | Action Param | Pre-training Data | Inference Speed |
|-------|------|--------------|-------------------|-----------------|
| RT-1 (Brohan et al. 2023b) | - | discretized | 130k demos | ~3Hz |
| RT-2 (Brohan et al. 2023a) | 55B (PaLI-X) | tokenized | web-scale VLM + robot | slow |
| OpenVLA (Kim et al. 2024) | 7B | 256 bins/dim autoregressive | 970k Open X-Embodiment | 4Hz |
| Octo (Team et al. 2024) | ~93M | diffusion head | 800k Open X-Embodiment | 10Hz |
| π0 (Black et al. 2024) | 3B (PaliGemma + flow matching) | flow matching | large-scale | faster |
| GR-2 (Cheang et al. 2024) | - | generative video-action | web-scale video | - |
| RDT-1B (Liu et al. 2024) | 1B | diffusion | Open X-Embodiment | - |
| TinyVLA (Wen et al. 2024) | smaller | - | data-efficient | faster |

RLDG 的 insight 对这个整个 landscape 都适用：所有这些 model 的 fine-tuning 都可以用 RL-generated data 替代 human demos。特别是对于 π0 这种用 flow matching（continuous action space）的 model，理论上更容易 distill RL policy 的 continuous optimal action distribution。

### 7.3 RL for Manipulation 谱系

Paper 提到的 RL 在 manipulation 上的成功案例：
- Precision insertion (Luo et al. 2021, 2019, 2018; Zhao et al. 2022; Schoettler et al. 2020)
- Multi-stage assembly (Gupta et al. 2021)
- Dexterous in-hand manipulation (Hu et al. 2024b; Rajeswaran et al. 2017, 2018)
- Residual RL (Johannink et al. 2019)
- Imitation bootstrapped RL (Hu et al. 2024a)

这些 RL 方法都面临 generalization 不足的问题（一个 task 训一个 policy），而 RLDG 通过 distillation 让 generalist policy 继承这些 RL policy 的 precision 同时获得 generalist 的 generalization。

## 8. Limitations 与 Future Directions

Paper 自己 disclose 的 limitations：
1. **Reward function 需求**：RLDG 假设能 access reward function（通过 success classifier）。对于 reward 难以 specify 的 task（比如 "make a beautiful arrangement"），需要 VLM-based reward generation
2. **RL objective 的 speed bias**：temporal discounting 导致 policy 优化 speed 而非 robustness，这在 Pick and Place 的 early drop 问题中体现。Future direction 可能需要 multi-objective RL（同时 optimize success, speed, robustness）
3. **Long-horizon decomposition 的人工介入**：哪些阶段用 RL，哪些用 human，目前需要 human judgment

我自己的一些 additional thoughts on limitations/future directions：

4. **RL policy 本身的 failure transfer**：如果 RL policy 在某些 corner cases 有 systematic bias（比如对某种 connector orientation 总是 approach 错误），distilled generalist 会继承这个 bias。需要 active learning 识别 RL policy 的 failure modes 并 targeted 补充 data
5. **Diffusion policy 的 unimodal issue**：Octo 用 diffusion head 本意是 capture multi-modal human action distribution，但 RL data 本身是 unimodal（RL policy 是 deterministic 或近 deterministic）。这意味着用 diffusion head 来 model RL data 可能 overkill，用 simple Gaussian 或 MSE loss 可能更 efficient。这个 paper 没有探讨 architecture 和 data source 的 co-design
6. **Online vs Offline distillation**：RLDG 是 offline（先训练好 RL policy，再 rollout，再 fine-tune generalist）。Online distillation（generalist policy 在 RL policy 指导下 online explore）可能更 sample efficient，但 implementation 更复杂
7. **Reward hacking transfer**：如果 RL policy 学会了某种 reward hacking behavior（比如利用 simulator bug 或 real-world physics quirk），这个 behavior 会被 distilled 进 generalist。需要 reward shaping 和 adversarial training mitigate

## 9. 对 Karpathy 你的一些 Direct Thoughts

考虑到你对 "software 2.0" 和 dataset quality 的长期关注，这篇 paper 的 message 其实非常 Karpathy-ian：**data quality 比 data quantity 更重要，而 RL 是一种 automated data quality improvement mechanism**。

这与你在 Tesla 时期的 experience 类似：用 simulation 和 RL-style optimization generate "better than human" driving data，然后 distill into production model。RLDG 本质上把这个 idea 应用到 robot manipulation foundation model 上。

更深层的问题：这个 framework 是否可以 extend 到 **self-improving loop**？即：
1. Generalist policy deployed，collect real-world interaction data
2. 用 reward function 识别 successful trajectories（或用 preference learning 识别 high-quality ones）
3. 这些 trajectories 反过来 fine-tune generalist policy（RLDG 风格）
4. 或者更激进：直接用 RL fine-tune generalist policy（避开 RLDG 的 two-stage pipeline）

Paper 提到 direct RL fine-tune foundation model 是 "largely an open problem"，但近期一些工作（比如 RLHF for VLA，PPO-style fine-tuning of LLM backbone）正在推进这个方向。如果这个 open problem 被解决，RLDG 这种 indirect distillation 可能只是一个 transitional technique。

但从 practical engineering 角度，RLDG 的 two-stage decoupling 有显著优势：RL training 可以在 narrow task 上用 small policy 快速 iterate（1-3 小时收敛），而 generalist fine-tuning 只需要 forward inference of RL policy 来 generate data，避免 RL gradient 的 instability 直接影响 7B parameter model。这种 **separation of concerns** 在 production system 中可能比 end-to-end RL fine-tuning 更 robust。

References:
- RLDG project: https://generalist-distillation.github.io/
- HIL-SERL: https://arxiv.org/abs/2410.21845
- OpenVLA: https://arxiv.org/abs/2406.09246
- Octo: https://arxiv.org/abs/2405.12213
- FMB benchmark: https://arxiv.org/abs/2401.08553
- RLPD: https://proceedings.mlr.press/v202/ball23a.html
- Llama 2: https://arxiv.org/abs/2307.09288
- LoRA: https://arxiv.org/abs/2106.09685
- DDPM: https://arxiv.org/abs/2006.11239
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://arxiv.org/abs/2307.15818
- π0: https://arxiv.org/abs/2410.24164
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- GR-2: https://arxiv.org/abs/2410.06158
- RDT-1B: https://arxiv.org/abs/2410.07864
- TinyVLA: https://arxiv.org/abs/2409.12514
- Policy Distillation (Rusu 2015): https://arxiv.org/abs/1511.06295
- Actor-Mimic (Parisotto 2015): https://arxiv.org/abs/1511.06342
- Distral (Teh 2017): https://arxiv.org/abs/1707.04175
- GPS (Levine & Abbeel 2014): https://proceedings.neurips.cc/paper_files/paper/2014/file/6766aa2750c19aad2fa1b32f36ed4aee-Paper.pdf
- Progressive Neural Networks (Rusu 2016): https://arxiv.org/abs/1606.04671
- Residual RL (Johannink 2019): https://arxiv.org/abs/1812.06298
