---
source_pdf: ConRFT A Reinforced Fine-tuning Method for.pdf
paper_sha256: 4356a2f2fed39e9286d22bd0238b918f4e2fa20fe183a0d94c6a37bab0b1667a
processed_at: '2026-08-18T03:50:05-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 ConRFT

## 一句话说清楚

这篇 paper 做的事情很简单：**pre-trained VLA model (Octo) 直接拿来用不行，SFT 几十条 demo 也不太行，那就先 offline 热身一下 Q-function，再放到真实机器人上边探索边有人盯着干预，用 RL 把 policy 推到 96% success rate。**

整个 story 可以拆成三个 layer 来理解。

---

## Layer 1: 为什么 SFT 不够好

假设你拿了个预训练好的 Octo model，想让它学会"把香蕉放盘子上"。你找了个人 teleop 20 条 demo，跑 NLL loss fine-tune。

问题在哪：

**Human demo 本身就是 noisy 的。** 同一个人今天 teleop 香蕉放盘子走轨迹 A，明天心情不好走轨迹 B，两次 action 在某些关键帧能差不少。SFT 把这些 action 都当 ground truth 去拟合，相当于让 policy 学一个平均化的、含噪声的 behavior manifold。

**Contact-rich task 尤其惨。** Insert Wheel 这种任务，pin 和 slot 的 tolerances 可能就几毫米，human teleop 时稍微手抖一下，demo trajectory 在关键 contact phase 的 action 就偏了。policy 学到这种偏的 action，部署时直接插不进去。

**State coverage 太窄。** 20 条 demo 覆盖的 state distribution 跟真实部署遇到的 state distribution 完全不重合。robot 初始位置稍微 randomize 一下 3cm，policy 就碰到大量 OOD state，直接崩。

这就是为什么 Table III 里 SFT 用 150 条 demo 才 58.3% avg success — **data quantity 救不了 data quality 的问题**。

---

## Layer 2: 为什么直接上 RL 也不行

那扔掉 SFT，直接在真实机器人上跑 RL 行不行？

**Sample efficiency 灾难。** Real-world 每个 episode 要几秒到几十秒物理时间，robot arm 还得 reset，reward 又 sparse (binary classifier 判任务完成与否)。RL from scratch 在这种条件上要收敛，少说几小时，多则几天。

**Safety 灾难。** RL early stage exploration 是随机性的，robot 会撞东西、用过大力、把 bread 压扁、把 wheel 硬塞弄坏 slot。物理世界不像 simulator，撞坏了得修，cost 很高。

**HIL-SERL (Luo et al. 2024, https://arxiv.org/abs/2410.21845) 就是 RL from scratch + human intervention 的 baseline**，Table II 显示它用同样时间只达到 31.9% avg success，远不如 ConRFT 的 96.3%。

所以 naive RL 在 real-world 不 work。

---

## Layer 3: ConRFT 怎么把两边拼起来

ConRFT 的核心 insight 是：**SFT 提供 safe initialization，RL 提供 exploration + optimization，Human 提供 safety net。三者缺一不可，且要用 unified objective 让 stage transition 顺畅。**

### Stage I (Cal-ConRFT, offline): 热身

用 20-30 条 demo，在 offline 条件下同时做两件事：

**(a) 训 Q-function (Cal-QL)。** 用 conservative Q-learning 让 critic 对 OOD action 给低值，对 in-distribution action 学真实 value。但这里有个关键 ablation — Cal-QL alone 在所有 task 上 success rate = 0%，因为 20 条 demo 的 state coverage 实在太窄，Q-function 学不出有意义的 landscape。

**(b) 训 consistency policy (BC-augmented)。** 加一个 BC loss，把 demo action 当作 consistency model 的 denoising target。这个 loss 既训练了 consistency network 的 denoising 能力 (生成 multi-modal action distribution)，又直接让 policy 对齐 demo action。

两个 loss 加起来就是 Eq. 3：
$$
\mathcal{L}_\pi^{offline}(\psi) = \beta \mathcal{L}_\pi^{BC} + \eta \mathcal{L}_\pi^Q
$$

Offline 阶段 weight 是 $\beta=1.0, \eta=0.1$，即 **以 BC 为主，Q 为辅**。Q loss 在这里不直接提升 offline success rate (Table I 显示 Cal-ConRFT = SFT = 39.4%)，但它 **预训练了一个合理的 Q-function landscape**，为 online 阶段 RL exploration 做 anchor。

这个 design 很关键，Figure 4 的 ablation 证明：从 SFT 起步做 online fine-tuning，intervention rate 飙高 (policy forgetting 严重)；从 Cal-ConRFT 起步，intervention rate 低且稳定。

### Stage II (HIL-ConRFT, online): 真实世界探索

Robot 跑 policy → 存 transition 到 replay buffer $\mathcal{R}$ → 同时 human 通过 SpaceMouse 监督 → 出问题 human 接管 → human 的 action 存到 demo buffer $\mathcal{D}$ (不是 $\mathcal{R}$!)。

这个 **"human intervention 存到 $\mathcal{D}$ 而非 $\mathcal{R}$"** 是一个细节但重要的 design choice。$\mathcal{D}$ 在 online 阶段依然被 symmetric sampling (每个 batch 一半 $\mathcal{D}$ 一半 $\mathcal{R}$)，所以 human correction 会以 50% 频率被反复采样，持续 anchor policy 不 drift 太远。

Online 阶段的 Q loss (Eq. 4) **去掉了 conservative regularization**：
$$
\mathcal{L}_Q^{online}(\theta) = \mathbb{E}_{(s,a,s') \sim (\mathcal{D} \cup \mathcal{R})}[(Q_\theta(s,a) - B^\pi \overline{Q}(s,a))^2]
$$

因为 online 数据分布随 policy 一起 evolve，不存在 offline 那种 OOD 问题。

Policy loss (Eq. 5) 结构跟 offline 一样，但 weight 调成 $\beta=0.5, \eta=1.0$，即 **以 Q (RL optimization) 为主，BC 为辅**。BC 保留是因为 RL exploration 在高维 action space 不稳定，BC anchor 防止 policy 跑去危险区域。

---

## 几个关键 intuition 串联起来

**1. Offline Q loss 不是为了 offline 性能，是为了 online adaptation 速度。**

这是 paper 最 counterintuitive 的地方。Table I 显示 Cal-ConRFT 和 SFT 的 offline success rate 都是 39.4%，看起来 offline 加 Q loss 没用。但 Figure 4 证明 online 阶段从 Cal-ConRFT 起步明显比从 SFT 起步稳。原因：Q-function 已经有一个 reasonable landscape，online RL 一开始就有 value guidance，不会盲目探索破坏 SFT 学到的 behavior。

类比 LLM 的场景：SFT-only model 直接跑 PPO，early stage 会有 severe policy collapse (InstructGPT paper 里也提到过类似现象)。如果先做一些 value 预热 (像 RLHF 里的 reward model 预训) 再上 PPO 会稳很多。Cal-ConRFT 在 robot 上的作用类似。

**2. Unified consistency-based objective 让 stage transition 顺畅。**

Offline loss 和 online loss 结构一样，只是 weight 不同。这意味着 stage 切换时 policy 不用 re-warm-up，consistency network 已经训好 denoising 能力，online 一开始就可以 generate reasonable action。

如果 offline 用 diffusion policy + NLL，online 切换到 Gaussian policy + Q loss，stage 切换时 action head 整个变了，policy 相当于从头学，sample efficiency 会差很多。

**3. Consistency policy 一个网络身兼三职。**

Consistency policy $\pi_\psi(a|s) = f_\psi(a^k, k | E_\phi(s))$ 同时承担：
- (a) Multi-modal action distribution 建模 (比 Gaussian 强，比 Diffusion 快)
- (b) BC loss 的载体 (consistency distillation objective 自然 align 到 demo action)
- (c) Inference 加速 (single-step denoising vs diffusion 的 multi-step)

这三者统一在一个 network + 一个 loss 里，比分别用不同 module 做 BC + RL + generation 简洁很多。

**4. Human intervention 的双重作用。**

不只是 safety net 防 robot 撞坏东西，还能 **escape local optima**。当 policy 卡在某个 unreachable state (比如 robot 把 wheel 顶死在 slot 边缘)，human 接管把 robot 推回 safe region，相当于给 policy 一个 reset 到 recoverable manifold 的 demonstration。这种 demonstration 的价值远高于随机 exploration。

---

## 跟 LLM RLHF 的类比

ConRFT 跟 LLM RLHF 的结构其实很像：

| Stage | LLM (RLHF) | VLA (ConRFT) |
|---|---|---|
| Pre-train | Next-token prediction on web data | VLA pre-training on Open X-Embodiment |
| SFT | Instruction tuning on human demonstrations | SFT on 20-30 human teleop demos |
| Reward model | Train RM on human preference pairs | Train binary classifier on task success |
| RL fine-tuning | PPO on policy rollouts | Cal-QL + consistency policy on robot rollouts |
| Safety | Constitutional AI / RLHF 本身的 preference | Human-in-the-loop intervention |

关键区别：
- LLM 的 rollout 是 synthetic 的 (生成 token 几乎免费)，VLA 的 rollout 是 physical 的 (每秒 10Hz 真实动作，撞坏要修)；
- LLM reward 来自 learned reward model，VLA reward 来自 binary classifier (更 sparse)；
- LLM action space 离散 (token)，VLA action space 连续 (6-7 dim delta pose)；
- LLM exploration 是 free 的，VLA exploration 需要 human 监督保证 safety。

所以 ConRFT 相对于 LLM RLHF 多出来的 engineering 复杂度，几乎都是在解决 "physical world 的 cost + safety" 这个问题。

---

## 还有什么没解决

Paper 自己承认的 limitations：

**1. Reward classifier 太脆。** Binary classifier 判断任务成没成，online exploration 时的 state distribution 跟 classifier training distribution 有 shift，policy 容易 reward hacking — 找个 false positive 状态骗 reward。比如 robot 把 end-effector 摆在某个特定位置，classifier 误判任务完成。Sparse reward 也让 policy 学得慢。

更好的方向：dense reward shaping，或者用 VLM 本身做 reward model (类似 VLM-RM, https://arxiv.org/abs/2312.16245)，不过 real-world 部署 dense reward 工程量大。

**2. Frozen encoder 限制 generalization。** 当前只 fine-tune action head (consistency network 那个 MLP)，visual encoder 和 transformer backbone 都 frozen。好处是 real-time 性能好 (不 forward 整个 backbone gradient)，坏处是 policy 无法 refine perception，遇到 OOD visual input 还是会崩。

Paper 提到 LoRA-style partial unfreeze 可能是改进方向 (Hu et al. LoRA, ICLR 2022: https://arxiv.org/abs/2106.09685)。这个方向如果 work，相当于把 LLM 的 PEFT 范式也搬到 VLA 上。

**3. 多任务泛化。** 当前是 task-specific fine-tuning，每个任务训一个 binary classifier + 一套 policy。没法 zero-shot 到新任务。如果要做 multi-task，需要 dense reward + multi-task Q-function (类似 Multitask CAL-QL 之类的延伸)。

---

## 我觉得有意思的几个点

**(a) "Offline Q loss 不提升 offline 性能但加速 online adaptation" 这个 finding 很关键。**

这跟 LLM 里 "SFT + preference pretrain 再 RLHF 比 SFT-only 直接 RLHF 更稳" 的直觉是一致的。value landscape 的预热对 RL 的 stability 极其重要，即使它不直接反映在 supervised metric 上。这条 insight 对 LLM RL fine-tuning 也有启发。

**(b) Consistency policy + Q loss unified objective 这个 formulation 很 elegant。**

把 generative model training (consistency distillation) 和 RL actor-critic 统一在一个 loss 里，是一种 "generative RL" 的思路。LLM 领域也有类似 line — 比如 GRPO (DeepSeekMath: https://arxiv.org/abs/2402.03300) 把 PPO 简化成 group relative advantage，或者 DPO (https://arxiv.org/abs/2305.18290) 把 RLHF 简化成 classification。Consistency + Q loss 这种 formulation 在 continuous action space 可能是个有价值的 direction。

**(c) Symmetric sampling + human-in-D 是一个很 simple 但 effective 的 trick。**

把 human correction 显式存到 demo buffer 并保持 50% sampling ratio，相当于让 human knowledge 持续 anchor policy 不被 RL exploration 冲垮。这个 trick 可以直接搬到 LLM RL fine-tuning — 比如把 human-edited response 存到单独 buffer 并保持一定 sampling ratio，可能缓解 RL fine-tuning 的 policy collapse。

---

## TL;DR

ConRFT 是一个把 LLM RL fine-tuning 的思路 (pre-train → SFT → RL) 搬到 VLA 上的 work，重点解决了 real-world deployment 的 safety + sample efficiency 问题。核心 tricks 是：offline 用 Cal-QL + BC 预热 Q-function，online 用 consistency policy + symmetric sampling + human-in-the-loop 做 safe exploration。8 个任务 96.3% avg success，45-90 分钟训完。

核心 insight 是：**RL fine-tuning 的 stability 来自好的 initialization (pre-trained VLA + offline Q 预热) + 安全 exploration (human intervention) + 统一 objective (consistency policy + BC + Q) 三者的协同，单一 component 都不够。**

Project page: https://cccedric.github.io/conrft/

---

# ConRFT: 一篇关于 VLA Model Reinforced Fine-tuning 的 paper 解读

## 一、整体故事与 motivation

Andrej, 这篇 paper 的核心 contribution 是把 LLM 领域的 "reinforced fine-tuning" 范式 (类似 RLHF, ReFT, RL fine-tuning 那一套) 移植到 VLA (Vision-Language-Action) model 上, 用于 real-world robotic manipulation。区别于 LLM 的 RLHF 主要对齐人类 preference, ConRFT 直接用 task-specific reward + Q-learning 来 optimize policy。

**关键痛点**: 现有 VLA model 的 SFT pipeline 有两个问题:
1. Human demonstrations 本身 sub-optimal 且 inconsistent — 比如同一个 task 不同人 teleop 时的 trajectory 可能差很多, 特别在 contact-rich 任务中 (Insert Wheel, Hang Chinese Knot 这种需要精细控制的场景)。
2. 状态覆盖太窄 — 20-30 条 demo 根本覆盖不了 real-world 的 state distribution, 碰到 OOD state policy 直接崩。

**为什么直接照搬 RL 不行**: LLM 的 RL fine-tuning 是在 simulator / 自生成环境里跑大量 rollout, 而 VLA 要直接和真实物理世界交互, safety + cost + sample efficiency 三重约束让 naive RL 不可行。

ConRFT 的解决方案是 two-stage pipeline + consistency-based unified objective + Human-in-the-loop 干预, 在 45-90 分钟 real-world 训练内把 8 个 manipulation 任务的 avg success rate 推到 96.3%。

项目主页: https://cccedric.github.io/conrft/

---

## 二、Two-stage 架构概览

```
Pre-trained VLA (Octo-small)
        │
        ▼
┌───────────────────────────┐
│ Stage I: Cal-ConRFT       │  ← offline, 用 20-30 demos
│  - Cal-QL critic           │
│  + BC-augmented            │
│  + Consistency policy head │
└───────────────────────────┘
        │
        ▼  (provides stable init policy + Q-function)
┌───────────────────────────┐
│ Stage II: HIL-ConRFT      │  ← online, real-world
│  - Human intervention      │
│  + Symmetric sampling      │
│  + Same loss structure     │
└───────────────────────────┘
        │
        ▼
Final policy (96.3% avg success)
```

设计上有点类似 offline-to-online RL (Nakamoto et al. Cal-QL, NeurIPS 2023: https://arxiv.org/abs/2310.10529), 但有两个 critical twist:
- 离线阶段不是只有 Cal-QL, 而是加了 BC loss 作为 auxiliary supervision;
- 用 consistency policy (Prasad et al. RSS 2024: https://arxiv.org/abs/2405.07503) 替代 diffusion policy 做 action head, 推理快。

---

## 三、Stage I: Cal-ConRFT 公式详解

### 3.1 Cal-QL critic loss (Eq. 1)

$$
\mathcal{L}_Q^{offline}(\theta) = \alpha \Big( \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi(\cdot|s)}[\max(Q_\theta(s,a), V^\mu(s))] - \mathbb{E}_{s,a \sim \mathcal{D}}[Q_\theta(s,a)] \Big) + \frac{1}{2}\mathbb{E}_{(s,a,s')\sim \mathcal{D}}\big[ (Q_\theta(s,a) - B^\pi \overline{Q}_{\overline{\theta}}(s,a))^2 \big]
$$

**变量解释**:
- $\theta$: critic (Q-function) 参数, 与下面 policy 参数 $\psi$ 不同。
- $\overline{\theta}$: target network 的参数 (delayed update, 用 Polyak average, 这是 TD-learning 标准技巧, 防 moving target 导致训练不稳)。
- $\mathcal{D}$: demo buffer, 存着 pre-collected demonstrations。
- $Q_\theta(s,a)$: 当前 critic 估计的 state-action value。
- $V^\mu(s)$: 参考策略 $\mu$ (通常是 behavior policy 估计) 的 value, 起到 baseline 作用。
- $B^\pi \overline{Q}(s,a) = r(s,a) + \gamma \mathbb{E}_{a' \sim \pi(\cdot|s')}[\overline{Q}(s',a')]$: Bellman backup operator, 就是 TD target。
- $\alpha$: conservative penalty 强度, 控制对 OOD actions 的惩罚有多狠。

**intuition**: 第一项是 conservative regularization 的核心。$\max(Q_\theta, V^\mu)$ 这一项说: 对任意当前 policy $\pi$ 采样的 action $a$, 它的 Q-value 不应该超过 behavior policy 的 value $V^\mu(s)$ 太多。对 in-distribution 的 $a$ (从 $\mathcal{D}$ 里采的), 减掉 $Q_\theta(s,a)$ 形成对 OOD 的惩罚。
- 在 $\mathcal{D}$ 内的 actions: penalize + compensate 净效果接近 0, 不影响学习;
- OOD actions: 只有 penalize 没有 compensate, 压低 Q 值, 阻止 policy 跑到没见过的 action space 区域。

这就是 Cal-QL 相比朴素 Q-learning 解决 offline RL "extrapolation error" (Kumar et al. Bear-Claw, NeurIPS 2019: https://arxiv.org/abs/1910.00900) 的核心机制。

**为什么 Cal-QL 单独不够**: 当只有 20-30 条 demos 时, state coverage 太窄, Q-function 几乎学不到有意义的值, 论文里直接说 Cal-QL alone 在所有任务上 success rate = 0%。这就是为什么需要 BC loss 辅助。

### 3.2 Consistency Policy (Eq. 2)

$$
\pi_\psi(a|s) = f_\psi(a^k, k | E_\phi(s))
$$

**变量解释**:
- $\psi$: consistency policy (action head) 参数;
- $\phi$: VLA model (encoder + transformer) 参数;
- $f_\psi$: consistency network, 学习 "denoise" 函数;
- $a^k \sim \mathcal{N}(0, kI)$: 加了 noise step $k$ 的 action, $k$ 从 $[\epsilon, K] = [0.002, 80]$ 中采;
- $E_\phi(s)$: VLA encoder 输出的 state embedding (frozen, 只 fine-tune $\psi$);
- $k$: diffusion step index, 表示 noise 强度。

**intuition**: Consistency policy 本质是 diffusion model 的 "蒸馏" 版本 (Song et al. Consistency Models, ICML 2023: https://arxiv.org/abs/2303.01400)。原始 diffusion policy 要迭代 K 步 (e.g. K=80) 才能从 pure noise 走到 action, consistency policy 学一个 single-step mapping, 在任意 noise level $k$ 都能直接跳到 action。

为什么选 consistency 不用 diffusion (Chi et al. Diffusion Policy, RSS 2023: https://arxiv.org/abs/2303.04137):
- Diffusion policy 推理慢 (要迭代多步), real-time control 10Hz 时压力很大;
- Consistency policy single-step inference, 但训练时利用 multi-step consistency objective 保证质量。

### 3.3 Combined BC + Q loss (Eq. 3)

$$
\mathcal{L}_\pi^{offline}(\psi) = \beta \mathcal{L}_\pi^{BC} + \eta \mathcal{L}_\pi^Q
$$

其中:
$$
\mathcal{L}_\pi^{BC} = \mathbb{E}_{(s,a)\sim \mathcal{D}, m \sim \mathcal{U}[1,M-1]}\big[ d(f_\psi(a + k_m z, k_m | E(s)), a) \big], \quad z \sim \mathcal{N}(0, I)
$$
$$
\mathcal{L}_\pi^Q = -\mathbb{E}_{s\sim \mathcal{D}, a\sim \pi_\psi}[Q(s,a)]
$$

**变量解释**:
- $\beta, \eta$: loss 权重 (论文里 offline 用 $(\alpha, \beta, \eta) = (0.01, 1.0, 0.1)$);
- $M = 40$: sub-interval 数量;
- $m \sim \mathcal{U}[1, M-1]$: 均匀采样一个 sub-interval index;
- $k_m$: 第 $m$ 个 sub-interval 的边界 (用 $\rho=7$ 的 schedule 算出来);
- $d(x,y) = \|x-y\|_2$: Euclidean distance。

**intuition for BC loss**: 这里很巧妙。BC loss 不是简单的 NLL, 而是把 demo action $a$ 当作 "noise-free target", 加 $k_m z$ 噪声到它身上 (相当于前向 diffusion), 然后让 consistency network 从这个 noised version 预测回 clean action $a$。这等价于 consistency distillation 的 training objective, 同时又承担了 BC 的模仿学习作用。所以一个 loss 既训练了 consistency model 的 denoising 能力, 又让 policy 对齐 demo action。

**intuition for Q loss**: 就是标准的 actor-critic 里 actor 的 objective — 最大化 critic 估计的 Q-value。让 policy 主动往高 Q-value 的 action 方向走。

**为什么 unified objective 重要**: 论文反复强调 offline 和 online 阶段用同一个 loss structure, 这让 online stage 不用 warm-up, policy 可以立刻利用 offline 学到的 Q-function 和 consistency model 表征, 这是 sample efficiency 高的关键。

---

## 四、Stage II: HIL-ConRFT 公式详解

### 4.1 Online Q loss (Eq. 4)

$$
\mathcal{L}_Q^{online}(\theta) = \mathbb{E}_{(s,a,s') \sim (\mathcal{D} \cup \mathcal{R})}\big[ (Q_\theta(s,a) - \mathcal{B}^\pi \overline{Q}(s,a))^2 \big]
$$

**变量解释**:
- $\mathcal{R}$: replay buffer, 存 online 探索得到的 transitions;
- $\mathcal{D} \cup \mathcal{R}$: 联合数据集;
- $\mathcal{B}^\pi \overline{Q}(s,a)$: 同上 Bellman backup。

**intuition**: 注意这里**去掉了 conservative regularization**! 原因是 online 阶段 policy 自己采样, 数据分布随 policy 一起 evolve, 不再是 "offline fixed distribution", 没有 OOD 问题 (Ball et al. Efficient Online RL with Offline Data, ICML 2023: https://arxiv.org/abs/2305.20016)。

### 4.2 Online policy loss (Eq. 5)

$$
\mathcal{L}_\pi^{online}(\psi) = \beta \mathcal{L}_\pi^{BC} + \eta \mathcal{L}_\pi^Q
$$

形式跟 offline 完全一样, 但有两个 critical 调整:
1. $\beta$ 减小, $\eta$ 增大 (offline: $\beta=1.0, \eta=0.1$ → online: $\beta=0.5, \eta=1.0$);
2. 数据从 $\mathcal{D} \cup \mathcal{R}$ 采, 且用 **symmetric sampling** — 每个 batch 一半从 $\mathcal{D}$, 一半从 $\mathcal{R}$, 保证 demo 信号不被 online 噪声淹没。

**为什么 online 阶段还保留 BC loss**: 论文给两个理由, 我觉得第一个更关键:
- **防 policy drift 太远**: RL exploration 在高维 action space 不稳定, BC anchor 防止 policy 跑到危险区域;
- **保持对齐 demo**: contact-rich task 里 sudden policy change 可能导致 collision / excessive force。

### 4.3 Human-in-the-loop 机制

```
Interaction Thread (online stage):
  if no human intervention:
      a_t ~ π_ψ(·|s_t)        ← policy 自己跑
      store (s_t, a_t, r_t, s_{t+1}) in R
  else:
      a_t = a_intv            ← human 用 SpaceMouse 接管
      store (s_t, a_intv, r_t, s_{t+1}) in D  ← 关键: 加到 D 不是 R
```

**intuition**: 人类干预的 transition **加到 demo buffer $\mathcal{D}$**, 这样它会一直以 50% 概率被采样 (symmetric sampling), 持续 anchor policy。这与 HG-DAgger (Kelly et al. ICRA 2019: https://arxiv.org/abs/1810.02190) 思路类似但更精细, 因为 HG-DAgger 把所有数据混一起, ConRFT 显式区分 "human expert correction" 和 "policy exploration"。

人类干预的两个作用:
1. **Safety**: 防止 robot 撞东西、用过大 force、破坏环境;
2. **Escape local optima**: 当 policy 卡在 unreachable 状态时, human 把 robot 推回 safe / recoverable region (类似 HIL-SERL, Luo et al. 2024: https://arxiv.org/abs/2410.21845)。

---

## 五、实验结果分析

### 5.1 主表 (Table I) 关键数据

| Method | Avg Success Rate | Avg Episode Length | Training Time |
|---|---|---|---|
| SFT (offline baseline) | 39.4% | 59.9 | — |
| Cal-ConRFT (offline) | 39.4% | 57.5 | — |
| HG-DAgger | 65% (+65%) | 56.3 (1.1x shorter) | 48.8 mins |
| PA-RL | 71.3% (+81%) | 51.1 (1.2x) | 48.8 mins |
| **HIL-ConRFT** | **96.3% (+144%)** | **30.7 (1.9x)** | **48.8 mins** |

**重要细节**:
- Cal-ConRFT 和 SFT 的 offline success rate **几乎相同** (39.4%), 这看起来很奇怪 — offline 加 Q loss 没收益? 但论文 ablation 解释了: Q loss 的好处体现在 **online 阶段的 adaptation 速度**, 不是 offline 性能。
- HG-DAgger 在 Hang Chinese Knot 上甚至 -10% (从 55% 掉到 50%), 因为 human correction 本身 inconsistent, 反而引入 noise。
- PA-RL (Policy Agnostic RL, Mark et al. 2024: https://arxiv.org/abs/2412.06685) 在 Insert Wheel 上 -14%, 因为 policy-agnostic Q-function 在 contact-rich task 上 generalize 不了。

### 5.2 为什么从 Cal-ConRFT 起步而非 SFT (Figure 4 ablation)

论文 Figure 4 对比从 SFT 和 Cal-ConRFT 起步的 online 学习曲线, 发现:
- 都从相近 success rate 出发;
- 但从 SFT 起步的 **intervention rate 显著更高** → SFT 起步的 policy 在 online 早期 severe policy forgetting, 因为 Q-function 没初始化好, 一开始 RL exploration 会破坏 SFT 学到的 behavior;
- 从 Cal-ConRFT 起步的 policy 利用 offline 学到的 Q-function 做 anchor, 平滑过渡。

**intuition**: 这就是为什么 offline 阶段要花心思训 Q — 它本身不直接提升 offline 性能, 但提供 "value landscape" 让 online RL 不至于一开始就崩。

### 5.3 增加 demos 不能 fix SFT (Table III)

| Method | Demos | Insert Wheel | Avg |
|---|---|---|---|
| Diffusion Policy | 150 | 35% | 41.7% |
| SFT (Octo) | 150 | 40% | 58.3% |
| RLDG | 150 (RL 收集) | 50% | 83.3% |
| Cal-ConRFT | 20 (human) | 35% | 36.7% |
| **HIL-ConRFT** | **20 + 80-120 rollout** | **80%** | **93.3%** |

**intuition**: 把 SFT 的 demos 从 20 加到 150 (7.5x), success rate 只 39→58, 远低于 HIL-ConRFT 用 20 demos + online 的 93%。说明 problem 不在 data quantity 而在 data quality (sub-optimal + inconsistent)。RLDG 用 RL policy 采集 demos, 因为 "optimal" 所以性能就高 (83.3%), 但这相当于 chicken-and-egg — 你得先有好的 RL policy 才能采集好 demos。

HIL-ConRFT 直接打破这个循环: 用少量人 demos 做初始化 + online RL 自己探索 + human intervention 保证 safety。

### 5.4 VLA backbone 泛化 (Table IV)

ConRFT 还在 RoboVLM (Li et al. 2024: https://arxiv.org/abs/2412.14058) 上用 Kosmos-2 (1.6B) 和 PaliGemma (3B) 两个 backbone 测试, frozen encoder + fine-tune action head, 都从 ~50% 提到 100%。这说明 method 不绑定特定 VLA architecture。

---

## 六、Limitations (论文自己列的)

### 6.1 Reward engineering sensitivity
用 binary classifier 判断任务是否完成作为 reward。问题:
- Classifier training data 与 online exploration 的 state-action distribution 有 shift, 容易 **reward hacking** (policy 找到 false positive 状态骗 reward);
- Sparse reward → policy 学得慢;
- Task-specific, 无法 generalize。

### 6.2 Frozen encoder + transformer
当前实现只 fine-tune action head, 视觉 encoder 和 transformer backbone 都 frozen。这样 real-time 性能好但限制了 policy 对 perception 模块的 refine。论文提了 LoRA (Hu et al. ICLR 2022: https://arxiv.org/abs/2106.09685) 可能的改进方向。

---

## 七、Intuition summary

把核心想法浓缩成几条:

1. **RL fine-tuning 的两个障碍是 safety 和 sample efficiency**。Safety 用 Human-in-the-loop 解决, sample efficiency 用 (a) pre-trained VLA 做 init, (b) offline Cal-QL 预训 Q-function, (c) consistency policy 加速 inference, (d) BC anchor 防 policy drift。

2. **Offline 阶段 Q loss 的作用是 "热身" Q-function, 不是直接提升 offline 性能**。Cal-ConRFT 和 SFT 的 offline success rate 持平, 但 online 阶段 Q 已经 ready, policy 不用从头学 value landscape。

3. **Unified consistency-based objective 是 pipeline 顺畅的关键**。Offline 和 online 用同一个 loss, 只是 weight 不同, policy 在 stage 切换时不用 re-warm-up。

4. **Data quality > data quantity for SFT**。150 demos 远比 20 demos 多, 但 SFT 仍然打不过 20 demos + online RL, 因为 human demos 本身 sub-optimal 且 noisy。

5. **Consistency policy 选得好**。比 diffusion policy 快, 比 Gaussian policy 表达能力强 (能建模 multi-modal action distribution), 训练时利用 consistency distillation objective 同时担任 BC 的角色。

---

## 八、相关参考链接

**核心方法 reference**:
- Cal-QL (Nakamoto et al. NeurIPS 2023): https://arxiv.org/abs/2310.10529
- CPQL / Consistency Policy + Q-learning (Chen et al. AAMAS 2024): https://arxiv.org/abs/2407.06100
- Consistency Policy for visuomotor (Prasad et al. RSS 2024): https://arxiv.org/abs/2405.07503
- Consistency Models original (Song et al. ICML 2023): https://arxiv.org/abs/2303.01400
- Diffusion Policy (Chi et al. RSS 2023): https://arxiv.org/abs/2303.04137

**VLA model reference**:
- Octo (Ghosh et al. RSS 2024): https://arxiv.org/abs/2405.12213
- Open X-Embodiment / RT-X (O'Neill et al. ICRA 2024): https://arxiv.org/abs/2310.08864
- RT-2 (Brohan et al. CoRL 2023): https://arxiv.org/abs/2307.15818
- π0 (Black et al. 2024): https://arxiv.org/abs/2410.24164
- RoboVLM (Li et al. 2024): https://arxiv.org/abs/2412.14058

**Baselines reference**:
- HG-DAgger (Kelly et al. ICRA 2019): https://arxiv.org/abs/1810.02190
- PA-RL (Mark et al. 2024): https://arxiv.org/abs/2412.06685
- HIL-SERL (Luo et al. 2024): https://arxiv.org/abs/2410.21845
- RLDG (Xu et al. 2024): https://arxiv.org/abs/2412.09858

**LLM RL fine-tuning reference (作为对照)**:
- Deep RL from Human Preferences / RLHF (Christiano et al. NeurIPS 2017): https://arxiv.org/abs/1706.03741
- InstructGPT (Ouyang et al. NeurIPS 2022): https://arxiv.org/abs/2203.02155
- ReFT reasoning (Trung et al. ACL 2024): https://arxiv.org/abs/2401.08967
- PPO (Schulman et al. 2017): https://arxiv.org/abs/1707.06347
- DigiRL in-the-wild device control (Bai et al. NeurIPS 2024): https://arxiv.org/abs/2406.07351

---

## 九、一点个人观察

这 paper 跟你之前在 Eureka Labs / NanoGPT 思路上其实有不少共鸣点 — 把 LLM 里的 RL fine-tuning recipe 拿过来, 用在另一个 modal (robotic action) 上。但有意思的地方在于 LLM 的 RL fine-tuning 是离散 token space + 大量 synthetic rollout, VLA 是连续 action space + 物理交互, 所以这里加了不少 "physical world-specific" 设计 — Human-in-the-loop 安全网、frozen encoder 换 real-time、consistency policy 替代 diffusion 加速 inference。

更值得 follow 的方向是 paper 在 limitations 里点出的 dense reward 和 partial unfrozen encoder (LoRA-style), 这两个落地后估计能再 push success rate 和 generalization 到下一个台阶。

Consistency policy 在这里很巧妙地同时扮演了 (1) multi-modal action distribution 建模器, (2) BC loss 的载体 (通过 consistency distillation objective), (3) inference 加速器。这其实暗示了一个更 general 的 idea: 把 generative model 的训练 objective 和 RL 的 actor objective 统一起来, 而不是分开训 actor 和 critic。这条 line of work 在 diffusion + RL (e.g., Wang et al. Diffusion Q-learning, ICLR 2023: https://arxiv.org/abs/2304.12876) 上已经有探索, consistency 版本是更 efficient 的 branch。

整篇 paper 的 ablation 设计 (从 Cal-QL alone 失败 → 加 BC → 再到 unified online pipeline) 很清晰, 把 "为什么需要每一步" 讲得比较到位, 是 real-world robot RL 里 rare 的 well-motivated engineering paper。
