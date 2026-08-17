---
source_pdf: Visual Verification Enables Inference-time Steering and Autonomous Policy
  Improvement.pdf
paper_sha256: 1174fb20f0f7fe9982b8adf30da6c78e062d46c6e23ee3ae68b1b6ca0e17c4c3
processed_at: '2026-08-13T02:25:03-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 VERITAS

## 1. 这篇 paper 在干嘛？

讲一个简单的 idea：**让 robot 自己跟自己玩，玩成功的就记下来，拿去练自己**。

听起来像 self-play？有点像，但 setup 不一样。Robotics 的痛点不是算法不行，是 **data 太贵**。你 Karpathy 训 GPT 可以爬互联网拿几万亿 token，robotics 想 collect 100 条 demonstration 得请人 teleop 一个下午。Physical Intelligence 训 $\pi_0$ 用了 thousands of hours 数据，这成本你想想。

那能不能让 robot 自己产生数据？问题来了：robot 自己 rollout 的大部分都是 failure，你把 failure 喂给自己 train，只会越练越烂。所以需要一个 "**裁判**" 来判断哪些 rollout 是好的。

VERITAS 的裁判是 VLM (Gemini 2.5)。但 VLM 不会直接说 "这个 action 好/不好"，它只会看图说话。所以 trick 是：让 VLM 在 task 开始时 **画一条理想轨迹** (在 image 上标 5-10 个 pixel waypoint)，然后 robot 每次决策时 sample 5 个候选 action，看哪个 action 的执行轨迹跟 VLM 画的那条更接近，就执行哪个。

就这么简单。

---

## 2. Generator-Verifier 的 mental picture

想象你在考试，有 5 个答案可以选，你不会做题但你会 "对答案"。你 sample 5 个候选解，然后用某种方式 verify 哪个最像正确答案，挑那个。LLM 里这叫 Best-of-N sampling，OpenAI 的 "Let's verify step by step" [17] 就是这个思路。

VERITAS 把这个搬到 robotics：

- **Generator**: 预训练好的 $\pi_0$ policy。它输出的是 action distribution $p(a|o,l)$，不是单个 action。你从里面 sample N 个候选 action chunk。
- **Verifier**: VLM 画一条 visual trace，然后 geometric check 每个 candidate 跟 trace 有多接近。接近的得分高。
- **Select**: 选得分最高的那个执行。

关键 trick 是：**verifier 是 gradient-free 的**。它不训练任何神经网络，就 VLM call 一次 + 算个 Euclidean distance。这跟你 nanoGPT 的精神一致 —— 能用简单几何解决的事，别上 learned value function。

为什么这能 work？因为 $\pi_0$ 训了 thousands of hours，它的 action distribution 里其实已经 cover 大量 valid strategy，只是 greedy decoding 总取 mode，把其他 valid modes 丢了。Sample N 个 + verify 相当于 **重新激活 distribution 里被埋没的好 candidates**。

公式 (2) 是 generator：
$$\pmb{a}_{t:t+H}^{(i)} \sim \pi_\theta(\cdot \mid o_t, l), \quad i \in \{1, \ldots, N\}$$

- $\pmb{a}_{t:t+H}^{(i)}$: 第 $i$ 个候选 action chunk，从 $t$ 开始延伸 $H$ 步
- $\pi_\theta$: frozen 的预训练 policy
- $N$: 采样数量，paper 用 5，sweep 显示 8 之后 saturate

公式 (4) 是 selection：
$$i^\star = \arg\max_{i} \nu_i, \quad \pmb{a}_{t:t+H}^\star = \pmb{a}_{t:t+H}^{(i^\star)}$$

- $i^\star$: 最高分 candidate 的 index
- $\nu_i = V(o_t, \pmb{a}_{t:t+H}^{(i)}, l)$: verifier 给第 $i$ 个 candidate 的分数

---

## 3. Visual Verifier 的 clever 之处

Verifier 怎么打分？这里 paper 的设计很 smart。

**Naive 方案**: 每步都 call VLM，给它看当前 frame + candidate action，问 "这个 action 好不好"。问题是 VLM 推理慢，每步 call 会卡死 control loop (15Hz)。

**VERITAS 方案**: 把 VLM 推理 **前置到 episode 开始**。VLM 只 call 一次，生成一条 **absolute pixel-space waypoint trace**：

$$\mathcal{W} = \{w_1, w_2, \ldots, w_K\}, \quad K \in [5, 10]$$

每个 $w_k = (u_k, v_k, \text{tol\_px}, \text{min\_hold}, \text{skippable})$：
- $(u_k, v_k)$: pixel 坐标
- $\text{tol\_px}$: 容差半径
- $\text{min\_hold}$: 在容差内至少停留几帧
- $\text{skippable}$: 可不可以跳过 (soft constraint)

之后 inference 时，verification 就退化成 **pure geometric check**：把 candidate action chunk 的 end-effector position 投影到 image plane，算它跟当前 waypoint 的 Euclidean distance。distance 小就 score 高。

Verification score (Algorithm 2):
$$s_t = \alpha \cdot p_t + (1-\alpha) \cdot \exp(-d_t / \tau)$$

- $s_t \in [0,1]$: normalized 分数
- $p_t = i / K$: 进度分 (已通过 $i$ 个 waypoint / 共 $K$ 个)
- $d_t = \|e_t - g_i\|$: end-effector pixel 位置 $e_t$ 到当前 waypoint goal $g_i$ 的距离
- $\alpha$: 进度 vs 接近度的权重
- $\tau$: temperature，控制距离衰减的 soft 度

这个设计的 elegance：
1. **VLM 只 call 一次**，不卡 control loop
2. **Geometric check < 1ms** per candidate，N=5 完全 parallelizable
3. **Zero-shot**: VLM 用 internet-scale prior 直接 reason 新任务，不用 train task-specific verifier
4. **Soft constraint**: waypoint 可 skippable，tolerance 是 soft 的，允许多种 valid 执行风格

跟你讲，这个设计是 paper 最漂亮的地方。它把 VLM 的 embodied reasoning 能力 "外化" 成一条 explicit 的 visual trace，然后用 trivial geometric 距离做 verification。VLM 不用懂 robotics，不用 output action，只要会 "看图想象轨迹"。这是 VLM 的 sweet spot。

---

## 4. Data Flywheel —— 真正的 magic

Inference-time steering 只是 paper 的 phase 1。真正 magic 在 phase 2：**把成功的 verified rollout 收集起来，fine-tune policy 自己**。

这为什么重要？因为 inference-time steering 有 cost：每次决策要 sample N 次 + verify。这 cost 是 recurring 的。如果你能把 verification 的 reasoning "bake" 进 policy weights，下次 deployment 就不需要 verify 了，policy 自己就直接输出好 action。

具体流程：

1. Deploy steered policy (with verifier)
2. Log 所有 successful trajectory: $\mathcal{D}_{\text{auto}} = \{(o_t, \pmb{a}_{t:t+H}^\star, l)\}_{\text{success}}$
3. Fine-tune base policy on $\mathcal{D}_{\text{auto}}$ with BC loss:
   $$\theta' \gets \arg\min_\theta \mathbb{E}_{\mathcal{D}_{\text{auto}}}[-\log \pi_\theta(\pmb{a}_{t:t+H}^\star \mid o_t, l)]$$
   
   - $\theta'$: 更新后的 policy 参数
   - $\pmb{a}_{t:t+H}^\star$: verifier 选出的 best action
   - Loss 就是标准 cross-entropy / NLL

这个流程形成了 flywheel：

```
Policy (generator) 
  → Sample N candidates 
  → Verifier 选 best 
  → Execute best 
  → 如果成功，log 进 D_auto
  → Offline BC fine-tune 
  → 更好的 Policy (generator)
  → 下一轮 deployment 需要的 N 更小
  → ... 循环
```

每次循环，policy 都更 robust，需要 sample 的 candidate 数 N 更少，inference cost 更低。VLM 的 reasoning 被 distill 进 policy weights。

---

## 5. 为什么这解决了 imitation learning 的老问题？

Imitation learning 有个经典痛点叫 **covariate shift** [29]。Expert demo 的 state distribution 跟 robot deployment 时的 state distribution 不一样，robot 一旦 drift 出 expert 覆盖的区域，就不知道怎么办。DAGGER [29] 的方案是让 expert 在 robot-visited states 上 re-label，但这需要 expensive human queries。

VERITAS 怎么 solve？三点：

1. **On-policy by construction**: $\mathcal{D}_{\text{auto}}$ 里的 trajectories 是 robot 自己 policy 生成的，state distribution **天然** match policy's exploration region。没有 covariate shift。

2. **Kinematically feasible**: 都是 robot 真实执行过的 action，不存在 expert demo 里可能的 "robot 做不到的 motion"。

3. **Success-filtered by verifier**: 只保留成功的、high-quality 的。Verifier 充当 "expert filter"。

这相当于 **用 VLM 替代了 human labeler**，closed the loop without any human supervision。这是 paper 的 strongest claim。

DAGGER 论文：https://arxiv.org/abs/1011.0686

---

## 6. 实验结果说得通吗？

### Simulation (Figure 4)

用 $\pi_0$-Bridge 在 4 个 SIMPLER task 上测：
- Base policy average: ~50%
- VERITAS steering: ~62% (+12.6%)
- Hardest task (Stack Blocks): 31% → 59% (+28%)

所有 verifier designs (包括最简单的 heuristic FSM) 都 beat base policy。这说明 generator-verifier paradigm 的 benefit **不依赖 specific verifier**。

### Real-world (Figure 6)

$\pi_0$-DROID + VERITAS: +35% success rate over base
$\pi_{0.5}$-DROID + VERITAS: 也显著提升

Baseline PIVOT [49] (用 action primitives 不用 policy prior) 完全失败 (0%)。这 highlight：**strong action prior from pre-trained policy 是必需的**，光靠 VLM visual optimization 不够。

### Data efficiency (Figure 7) —— 最 striking 的结果

比较 autonomous verified data vs human expert demos 做 fine-tune:
- 20 demos: autonomous ≈ human
- 50 demos: autonomous **>** human on some tasks (Carrot on plate: 0.70 vs 0.65)
- 100 demos: convergence

**20-100 条 autonomous trajectory 就能 match human expert data**。这是 robotics 的 "synthetic data match real data" 时刻。

---

## 7. 为什么 20-100 条 autonomous data 能 match human data？

这个结果初看 surprising，细想 make sense：

1. **On-policy**: 没有 distribution shift，每条 trajectory 都在 policy 的 "exploration region" 里，information density 高
2. **Success-filtered**: 只有成功的，没有 failure noise
3. **Diverse**: $\pi_0$ 是 multi-modal policy，sample 出的 trajectory cover 不同 strategy，比单个 expert teleop 多样
4. **Kinematically feasible**: 都是 robot 真的执行过的

Human demo 有个隐藏 cost：human expert 用的是 human motor control，可能产生 robot 难复现的 motion (kinematic mismatch)。Autonomous data 没这个问题。

换句话说，autonomous data 的 **per-trajectory information content** 比 human demo 高。所以 20 条 autonomous ≈ 50 条 human 是合理的。

---

## 8. Intuition: 为什么这能 work？

我给你几个 mental model：

### Mental model 1: Pre-trained VLA 是 "over-confident mode-seeker"

$\pi_0$ 训了 thousands of hours data，它的 action distribution $p(a|o,l)$ 其实已经 cover 大量 valid strategy。问题是 greedy decoding 总取 mode，丢弃了其他 valid modes。VERITAS 通过 sampling + verification **重新激活 distribution 里被埋没的好 candidates**。

N=5 就够，因为 distribution 里 top-5 candidates 已经 cover 大部分 valid strategy。N=8 saturate 说明 distribution 集中度还可以。

### Mental model 2: Verification 比 Generation 容易

LLM 里我们都知道 verifying a solution 比 generating one 容易。Process Reward Models [17] 证明了这点。VERITAS 把这个 asymmetry exploit 到 robotics：

- Generator (难): 需要 robotics data 训练，$\pi_0$ 用了 thousands of hours
- Verifier (相对易): VLM 用 internet image + text 训练，已经 internalize "approach object", "grasp", "place" 等 spatial reasoning。不需要 robotics-specific training

VLM 不需要 generate valid robot action (难)，只需要 generate visual trace (相对易) + 做 geometric check (trivial)。

### Mental model 3: VLM 是 "free internet supervision"

Gemini 2.5 在海量 internet image 上训练，已经懂 "怎么把杯子放到盘子上" 这种 spatial reasoning。这个 reasoning 是 **internet-scale supervision 的副产物**，对 robotics 是 "免费" 的。

VERITAS 把这个 free reasoning 外化成 visual trace，作为 verification signal。这相当于 **从 VLM 的 internet prior 里 distill supervision 给 robotics policy**，bypass 了 robotics data collection bottleneck。

### Mental model 4: Inference compute 比 human time 便宜得多

这是 paper 的核心 economic argument：human demo 一个下午 100 条，inference sampling + verification 几小时可以产生几百条 autonomous data。trade inference compute for human time 是划算的。

这个 economic argument 跟 LLM test-time scaling [13] 一样：train-time compute 贵，inference compute 便宜，所以 inference-time 多算一会儿 (sampling + verification) 比 train 更多 data 划算。

Snell et al. "Scaling test-time compute": https://arxiv.org/abs/2408.03314

---

## 9. 跟 LLM test-time scaling 的 connection

你 Karpathy 在 OpenAI 肯定熟悉这套：

| LLM test-time scaling | VERITAS |
|------------------------|---------|
| Generate N solution traces | Sample N action chunks |
| Verifier / Reward model scores | Visual verifier scores |
| Best-of-N selection | Best-of-N selection |
| Process reward model (trained) | VLM + geometric check (gradient-free) |
| Inference compute for accuracy | Inference compute for success rate |
| --- | + Data flywheel (distill into policy) |

最大区别：LLM 里 verifier 是单独 train 的 reward model，VERITAS 里 verifier 是 **pre-trained VLM + geometric check**，完全 gradient-free，zero-shot。

VERITAS 多了一步 LLM 里没有的：**把成功的 trajectory 收集起来 fine-tune policy**。LLM 里你 verify 完就完了，不会回头 train base model。Robotics 里因为 data 贵，所以要把 inference-time compute 转化成 permanent policy improvement。这就是 flywheel 的意义。

---

## 10. 跟你之前 work 的几个 connection

### nanoGPT 精神

VERITAS verifier 极简：VLM call 一次 + geometric distance check。没有 trained value function，没有 RL，没有 complex reward shaping。这跟你 nanoGPT [https://github.com/karpathy/nanoGPT] 的精神一致 —— 用最简洁 implementation 揭示最核心 mechanism。

### State of GPT 里的 system 1 / system 2

你在 "State of GPT" [https://www.youtube.com/watch?v=zxXyOlqaqb0] 讲过 system 1 vs system 2 thinking。Inference-time compute 是 system 2 的 enabler。VERITAS 把这个 idea 具体化到 robotics: inference-time sampling + verification 就是 robot 的 system 2 thinking。Policy 自己是 system 1 (fast, automatic)，verifier-augmented sampling 是 system 2 (slow, deliberative)。

Flywheel 的意义是：**把 system 2 的 reasoning distill 回 system 1**。下次同样的情况，policy 直接 (system 1) 输出好 action，不用再 verify (system 2)。这跟你讲的 "internalize system 2 into system 1" 完全对应。

### Software 2.0 的有趣张力

你写过 "Software 2.0" [https://karpathy.medium.com/software-2-0-a64152b37c35] —— 神经网络取代 explicit code。VERITAS 是个有趣 reverse case：它把 neural network (VLM) 当 "code" 用 (生成 explicit waypoint trace)，然后用 explicit geometric check 做 verification。这是 **Software 1.5** —— hybrid of neural reasoning 和 explicit algorithmic verification。

### "A few useful things" 里的 simple baselines

你最近 blog 强调 simple baselines 力量。VERITAS ablation 显示连最简单的 Heuristic verifier (FSM-based) 都能 beat base policy。这说明 generator-verifier paradigm 的 benefit **不依赖 fancy verifier**，simple geometric check 就有 gain。这对 simple-baseline-first 方法论是支持。

---

## 11. 几个我会 push back 的点

### N=5 真的够吗？

Simulation sweep 到 N=8 saturate。但这个 saturation 可能在更难 task 上 break。如果 task 难到 policy prior 很差，top-5 可能都是 bad，需要 N=20, 50, 100。Long-horizon task 或者 OOD task 可能 require 更大 N。

### Static trace 对 dynamic scene 会 break

VLM 在 episode 开始时生成 static trace。如果中间 object 被人 move 了，或者 stochastic dynamics 让 object 滑动，trace 就 stale。VLM-Constraints variant (用 object tracking 动态 resolve waypoint) 可以 address，但需要持续 tracking，可能 fail under occlusion。

### Verifier 会 confidently wrong

VLM verifier 依赖 embodied reasoning prior。如果 task 涉及 VLM 从未见过的 physical interaction (某种 special tool use, deformable object)，VLM 可能生成错误 trace，verifier confidently wrong。这是 VLM-as-verifier 的 fundamental limitation。

### On-policy data 没有 exploration

VERITAS 只保留 successful trajectories。Policy 在它能 succeed 的 region 里 self-improve，但 **不能 expand 到它 currently fails 的 region**。这是 pure exploitation 没有 exploration。RL fine-tuning (用 verifier as reward) 可以 address，但 paper 没做。

---

## 12. 一句话总结

**用 VLM 当 robot 的 "裁判"，让 robot 自己试 5 次挑最好的执行，成功的记下来拿去练自己，20-100 条自主 trajectory 就能 match human expert data**。

这个 paradigm 的 beauty 在于它把 inference compute 转化成 permanent policy improvement，解决了 robotics 的 data scaling bottleneck。VLM 提供 "free" supervision，robotics policy 提供 strong action prior，两者协作形成了 self-improving flywheel。

这是 robotics foundation models 时代的 "RLHF moment" —— 用 VLM reasoning 替代 human supervision，用 inference compute 替代 data collection labor。

---

# VERITAS: 从 Inference-time Verification 到 Robot Self-Improvement 的 Data Flywheel

## 1. 核心 intuition：把 LLM 的 test-time compute scaling 搬到 robotics

这篇 paper 的 mental model 非常漂亮。Karpathy 你应该立刻能联想到 OpenAI 的 "Let's verify step by step" [17] 和 Snell 等人的 "Scaling test-time compute" [13] —— 在 LLM 里，我们通过 generate multiple solution traces + verifier 选择最好的那个 (Best-of-N) 来 trade inference compute for accuracy。VERITAS 把这个 paradigm 搬到 robot manipulation，但有一个 critical 的 twist：**verification 不只是 "momentary boost"，而是 data engine**，形成 inference-time steering → 成功 trajectory logging → offline fine-tuning → policy improvement 的 flywheel。

这个 twist 解决了 robotics 里一个 fundamental 的 pain point：**data scaling bottleneck**。Text/image data 可以爬互联网，robotics data 必须由人类 demonstrator 一个一个 teleop 收集，scaling linearly with human expert time [1, 2, 3, 4, 5, 6, 7]。VERITAS 的 insight 是：既然 inference compute 比 human demonstration 便宜得多，不如让 robot 自己 "try N 次 + verify + 记下来成功的"，然后 distill 这个 verification reasoning 回 policy。

项目主页：https://veritas-improvement.github.io

参考相关工作：
- Test-time scaling in LLMs: https://arxiv.org/abs/2408.03314 (Snell et al.)
- Process reward models: https://arxiv.org/abs/2103.02355 (Cobbe et al.)
- Let's verify step by step: https://arxiv.org/abs/2305.20050 (Lightman et al.)

---

## 2. Problem Formulation 的 POMDP 设定

形式化为 POMDP $(S, O, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$：
- $S$: state space (hidden, 部分可观测)
- $O$: observation space (图像 + proprioception)
- $\mathcal{A}$: action space (end-effector poses / deltas)
- $\mathcal{T}$: transition function (physics)
- $\mathcal{R}$: reward (这里其实是 sparse 的 success signal)
- $\gamma$: discount factor

在每个 timestep $t$，robot 收到 observation $o_t \in O$ 和 language instruction $l \in \mathcal{L}$。目标是学一个 policy $\pi_\theta$ 最大化 expected success rate。

**Key difference from 标准 imitation learning**: 标准假设 fixed static dataset，VERITAS 要 policy 在 deployment 后能 self-improve from own experience。

### Action chunking 的重要性

公式 (1):
$$\pmb{a}_{t:t+H} \doteq (a_t, a_{t+1}, \ldots, a_{t+H-1}) \in \mathcal{A}^H$$

这里：
- $\pmb{a}_{t:t+H}$: 从时刻 $t$ 开始 horizon 为 $H$ 的 action chunk (黑体表示 sequence)
- $a_t, a_{t+1}, \ldots$: 单步 actions
- $H$: chunk length (在 real-world 用 $H=8$，simulation 用 $H=4$)

为什么 chunking 关键？两个原因：
1. **Amortize verification cost**: verifier 一次评估 $H$ 步而不是每步评估
2. **Temporal context**: verifier 能看到 behavior 的 consequence，而不是 micro-instantaneous movement

这个设计直接借鉴了 ACT [42] 和 Diffusion Policy [43] 的 chunking 思想。参考：https://diffusion-policy.cs.columbia.edu/

### Generative policy 作为 prior

这里他们用的是 $\pi_0$ [5]，flow-matching action head。和 deterministic regression policy 不同，generative policy 学的是 conditional probability distribution:

$$p(\pmb{a}_{t:t+H} \mid o_t, l)$$

这个 multi-modal 分布是 VERITAS 能 work 的前提：policy 不只是输出 mean action，而是建模 "抓物体的不同角度"、"多步任务的不同顺序" 等多种 affordance [44]。Inference-time 通过 stochastic sampling 从这个 prior 里 sample 出 $N$ 个 candidates。

$\pi_0$ 论文：https://physicalintelligence.company/blog/pi0
Open-π-zero 实现 (他们用这个)：https://github.com/AllenzRen/open-pi-zero

---

## 3. Generator-Verifier 框架的数学结构

### Generator (公式 2)

$$\pmb{a}_{t:t+H}^{(i)} \sim \pi_\theta(\cdot \mid o_t, l), \quad i \in \{1, \ldots, N\}$$

- $\pmb{a}_{t:t+H}^{(i)}$: 第 $i$ 个 candidate action chunk
- $N$: sample 数量 (paper 里用 $N=5$，sweep 到 $N=8$ 之后 saturation，见 Figure 9)
- $\pi_\theta$: 预训练好的 policy，参数 $\theta$ 在 inference-time steering 阶段 **frozen**

### Verifier (公式 3)

$$V(o_t, \pmb{a}_{t:t+H}, l) \in \mathbb{R}$$

- $V$: 一个 plug-and-play, gradient-free function (不是 neural net 的梯度更新对象)
- 输入：当前 observation $o_t$，candidate action chunk $\pmb{a}_{t:t+H}$，instruction $l$
- 输出：标量 score $\nu_i$，表示这个 action chunk 的 utility/alignment

**Crucially**，$V$ **不更新** $\pi_\theta$。它只是 inference-time 的 filter，可以替换为 VLM、geometric constraints、learned value model 等等。这种 decoupling 让 framework 对 verifier 设计 agnostic。

### Best-of-N Selection (公式 4)

$$i^\star = \arg\max_{i \in \{1, \ldots, N\}} \nu_i, \quad \pmb{a}_{t:t+H}^\star = \pmb{a}_{t:t+H}^{(i^\star)}$$

- $i^\star$: 最高 score 的 candidate index
- $\pmb{a}_{t:t+H}^\star$: 被选中执行的 action chunk
- $\nu_i = V(o_t, \pmb{a}_{t:t+H}^{(i)}, l)$: 第 $i$ 个 candidate 的 score

这就是 LLM Best-of-N sampling 的 robotics 版本。Karpathy 你在 OpenAI 时应该熟悉这个。但 robotics 版本有一个关键不同：**LLM 里 verifier 是单独 train 的 reward model**，**这里 verifier 是 pre-trained VLM + geometric scoring**，完全 gradient-free，不需要训练任何 verifier model。

---

## 4. Visual Verifier Design —— 最精彩的部分

这是 paper 里最 clever 的设计。他们不用 learned value function (像 V-GPS [38] 那样需要 offline RL 训练)，而是用 **两阶段 VLM-based visual trace + geometric consistency check**。

### Stage 1: Trace Generation (episode 开始时一次性)

VLM (用 Gemini 2.5 [45]) 接收 initial observation $o_0$ 和 instruction $l$，被 prompt 去生成 **absolute pixel-space waypoint trace**:

$$\mathcal{W} = \{w_1, w_2, \ldots, w_K\}$$

每个 waypoint $w_k = (u_k, v_k, \text{tol\_px}, \text{min\_hold}, \text{skippable}, \text{weight})$：
- $(u_k, v_k)$: 在 image plane 上的 absolute pixel coordinate
- $\text{tol\_px}$: tolerance radius (pixels)
- $\text{min\_hold}$: 必须 dwell 在 tolerance 内的最小 frame 数
- $\text{skippable}$: 是否可以跳过 (soft constraint)
- $\text{weight}$: 在 composite score 中的权重

Algorithm 2 里 clamp $|\mathcal{W}| \in [5, 10]$，强制 start/end 不可跳过。

VLM prompt (附录里给的):
```
Generate 5-10 absolute pixel waypoints (u,v) for the robot end-effector to
complete the task. Return JSON matching schema:
waypoints: [{"uv": [u,v], "tol_px": float, "min_hold": int,
"skippable": bool, "weight": float}],
confidence: float, notes: optional.
Instruction: {instruction}.
pixel coords are absolute.
```

### Stage 2: Geometric Scoring (每个 verification step)

Inference-time，对每个 candidate action chunk $\pmb{a}_{t:t+H}^{(i)}$，他们做：
1. 把 action chunk 对应的 end-effector positions 投影到 image plane (用 camera intrinsics)
2. 计算投影轨迹和 VLM trace 之间的 negative Euclidean distance

公式化：
- $e_t \in \mathbb{R}^2$: end-effector 在 image plane 的位置
- $g_i$: 当前 active waypoint $w_i$ 解析出来的 pixel goal
- $d_t = \|e_t - g_i\|$: end-effector 到当前 waypoint 的距离

Verification score (Algorithm 2 里):
$$s_t = \alpha p_t + (1-\alpha) \exp(-d_t / \tau)$$

- $s_t \in [0, 1]$: normalized verification score
- $p_t = i / K$: progress through waypoint sequence ($i$ 是当前 waypoint index, $K$ 是总数)
- $\alpha$: weighting coefficient (progress vs proximity)
- $\tau$: temperature / length scale
- $\exp(-d_t / \tau)$: Gaussian-like proximity score

如果 $i \geq |\mathcal{W}|$ (所有 waypoints 完成)，$s_t = 1.0$。

### 为什么这个设计 brilliant？

1. **VLM 只调用一次** (episode 开始时)，之后 verification 是 fast geometric check (< 1ms per candidate)。这意味着 verifier 不在 control loop 里造成 latency bottleneck。

2. **Zero-shot generalization**: VLM 的 embodied reasoning 能力直接 transfer 到新任务，不需要训练 task-specific verifier。这和 V-GPS 形成鲜明对比 —— V-GPS 需要 offline RL 训练 value function，每个 task 都要训。

3. **Multi-modal tolerance**: waypoint 可标记 skippable，tolerance 是 soft 的，允许 "alternative but valid execution styles"。这让 verifier 不会 over-constrain policy。

4. **Temporal consistency**: trace 在整个 episode 里 static，避免 per-frame reasoning 的 inconsistency。

---

## 5. Autonomous Self-Improvement 的 Data Flywheel

### Dataset 构造 (公式 5)

$$\mathcal{D}_{\text{auto}} = \bigcup_k \{(o_t, \pmb{a}_{t:t+H}^\star, l)\}_{t \in \mathcal{I}_{\text{success}}}$$

- $\mathcal{D}_{\text{auto}}$: 自主收集的 dataset
- $k$: index 不同 episodes
- $(o_t, \pmb{a}_{t:t+H}^\star, l)$: (observation, verified best action, instruction) tuples
- $\mathcal{I}_{\text{success}}$: 成功 episode 内的 timesteps (失败 episode 整个被过滤掉)

### Behavior Cloning Fine-tuning (公式 6)

$$\theta' \gets \arg\min_\theta \mathbb{E}_{\mathcal{D}_{\text{auto}}}[-\log \pi_\theta(\pmb{a}_{t:t+H}^\star \mid o_t, l)]$$

- $\theta'$: 更新后的 policy parameters
- $\theta$: 原始 policy parameters (initialization)
- $\mathbb{E}_{\mathcal{D}_{\text{auto}}}$: 在 autonomous dataset 上的期望
- $-\log \pi_\theta(\pmb{a}^\star \mid o, l)$: negative log-likelihood (cross-entropy / BC loss)

Algorithm 1 的 Phase 2:
```
Initialize θ' ← θ
for e = 1 to E do
    Sample batch B ~ D_auto
    Update θ' via Behavior Cloning:
    θ' ← θ' - η ∇θ' Σ_{(o,a,l)∈B} L_BC(π_θ'(·|o,l), a)
end
```

- $E$: epochs (real-world 用 20,000 steps)
- $\eta$: learning rate ($5 \times 10^{-5}$)
- $L_BC$: BC loss

### Mitigating Distribution Shift 的关键 insight

这是 paper 里 **最深刻** 的部分之一。Imitation learning 的经典问题是 covariate shift [29]: policy 在 deployment 时 drift 出 expert 的 state distribution，performance 崩溃。DAGGER [29] 的解决方案是让 expert 在 robot-visited states 上 re-label，但这需要 expensive human queries。

VERITAS 的解决方案非常 elegant：
1. **On-policy by construction**: $\mathcal{D}_{\text{auto}}$ 里的 trajectories 是 robot 自己 policy 生成的 (经过 verifier filter)。state distribution **inherently** matches policy's exploration prior。
2. **Kinematically feasible**: 因为是 robot 真正执行过的 action，不存在 expert demo 里可能的 kinematic 不匹配 (比如 expert 用了 robot 不能做的 motion)。
3. **Verifier 充当 "expert filter"**: 只保留成功的、high-quality 的 trajectories，丢弃 failures 和 suboptimal behaviors。

这本质上是 **replaced human labeler with automated verifier**，closed the loop without shared autonomy [26, 27, 28] 或 human-in-the-loop [23, 24, 25]。

DAGGER 论文：https://arxiv.org/abs/1011.0686

---

## 6. 实验结果的细节拆解

### 6.1 Inference-Time Steering 的 quantitative results

Table (Figure 4 数据) —— Simulation (SIMPLER [21], BridgeData [46], $\pi_0$-Bridge policy):

| Task | Base Policy | V-GPS | Heuristic | VLM+Constraints | VERITAS |
|------|-------------|-------|-----------|-----------------|---------|
| Close Box | ~62% | ~65% | ~68% | ~70% | **~73%** |
| Stack Blocks | ~31% | ~38% | ~45% | ~50% | **~59%** |
| ... | ... | ... | ... | ... | ... |

Overall: VERITAS verifier 在 simulation 上平均提升 **+12.6%** success rate，在 real world 上 **+35%**。

关键观察：
1. **所有 verifier designs 都 beat base policy**，连最简单的 heuristic verifier 都有提升。这说明 generator-verifier paradigm 的 benefit **不依赖于特定 verifier 架构**。
2. **VERITAS (visual trace) > V-GPS (learned value function)** across all tasks。这暗示 explicit verification of action consequences 比 learned value estimates 更可靠。
3. **Hardest task (Stack Blocks) gain 最大** (31.3% → 59.2%)。解释：policy 在 difficult task 上更 uncertain，sample 出的 action distribution 更 diverse，给 verifier 更多 opportunity 去挑选 better candidates。Simple task 上 policy 已经 confident，samples 不 diverse，提升空间小。

### 6.2 Inference-Time Steering 的 Real-World Results

(Figure 6) 用 $\pi_0$-DROID 和 $\pi_{0.5}$-DROID 两个 policy，每个 policy 2 个 task，每个 task 50 trials:

- $\pi_0$-DROID + VERITAS 显著超过 $\pi_0$-DROID base
- $\pi_{0.5}$-DROID + VERITAS 也显著超过 base
- 都超过 V-GPS-DROID baseline

值得注意的是 PIVOT [49] baseline —— 用 naïve action primitives + iterative visual optimization —— 完全失败 (0% success)。这 highlight 了 **strong action prior (from pre-trained policy) 的必要性**。VLM-only 没有 policy prior 的话，even with visual optimization，也无法完成 manipulation。

PIVOT 论文：https://arxiv.org/abs/2402.07872

### 6.3 Offline Policy Improvement (Figure 5)

在 simulation 上用 656 verified demonstrations fine-tune $\pi_0$-Bridge:
- Base policy average: ~50%
- Inference-time steering: ~62%
- Fine-tuned on verified data: ~60% (略低于 steering，但 permanently baked in)

这表明 **fine-tuning 成功 distill 了 verification-time reasoning 进 policy weights**。

Real world (Figure 7) 更有意思，他们做了 data efficiency comparison:
- 20 demos: autonomous data matches human data
- 30 demos: autonomous data matches or slightly better
- 50 demos: autonomous data **outperforms** human data on some tasks (Carrot on plate: 0.70 vs 0.65; Pick up mouse: 0.65 vs 0.60)
- 100 demos: convergence, autonomous data remains competitive

这是 paper 的 strongest claim: **20-100 条 autonomous verified trajectories 可以 match human expert demos 的 data efficiency**。这相当于 robotics 的 "synthetic data match real data" 时刻。

### 6.4 Hyperparameter Sweeps

**Action execution horizon (Figure 8)**: 
- 在 $\pi_0$-DROID 上 sweep 1-10 steps
- Open-loop execution 不显著 degrade verification performance
- 选 $H=8$ 用于 real-world

**Number of samples N (Figure 9)**:
- Performance 随 $N$ 增加而 improve
- $N=8$ 之后 saturate
- Cost linearly scales with $N$
- 选 $N=5$ 作为 trade-off

---

## 7. Algorithmic Flow 完整解析

### Algorithm 1 (Phase 1: Inference-time Steering) 详解

```
Input: π_θ, V, l, H, N
Output: Executed trajectory, D_auto, π_θ'

Initialize D_auto ← ∅, observe o_0
for t = 0, H, 2H, ... until termination:
    // Step 1: Sample N candidates in parallel
    {a^(i)_{t:t+H}}_{i=1}^N where a^(i) ~ π_θ(·|o_t, l)
    
    // Step 2: Verify each candidate
    ν_i ← V(o_t, a^(i)_{t:t+H}, l)
    
    // Step 3: Best-of-N selection
    i* ← argmax_i ν_i
    a*_{t:t+H} ← a^(i*)_{t:t+H}
    
    // Step 4: Execute and log
    Execute a*_{t:t+H}, observe o_{t+H}
    Log (o_t, a*_{t:t+H}, l) into D_auto
end
```

**Important details**:
- t 以 $H$ 步进 (因为 chunking)，每 $H$ 步做一次 verification
- 这就是为什么 verification cost 被 amortize —— 不是 per-step verification
- $N=5$ candidates 并行 sample + 并行 verify (geometric scoring 极快)

### Algorithm 1 (Phase 2: Offline Self-Improvement) 详解

```
if Offline update is enabled:
    Initialize θ' ← θ  (start from base policy)
    for e = 1 to E:
        Sample batch B ~ D_auto
        Update θ' via BC:
        θ' ← θ' - η ∇θ' Σ_{(o,a,l)∈B} L_BC(π_θ'(·|o,l), a)
    return π_θ'
end
```

- E = 20,000 steps (real-world)
- η = 5e-5
- Batch size: 1024 (π0-Bridge), 256 (π0-DROID)
- 这就是标准 BC fine-tuning，loss 是 negative log-likelihood

---

## 8. Verifier 设计的 Ablation (附录 A.1)

Paper ablate 了 3 种 verifier design:

### 8.1 VERITAS (absolute pixel trace)
- VLM 生成 absolute pixel-space waypoints (5-10 个)
- Verification: geometric distance check
- **优点**: VLM 只调用一次，verification 极快 (< 1ms)
- **缺点**: trace static，不能适应 dynamic scene (objects 移动)

### 8.2 VLM-Constraints (relative reference trace)
- VLM 生成 reference-based waypoints: `role:source`, `role:target`, `midpoint(refA,refB)`
- 动态 resolve via object tracking (SAM 2 [51] / Grounded SAM [52])
- **优点**: 适应 dynamic scene，objects 移动时 waypoints 跟着移动
- **缺点**: 需要持续 object tracking，每帧都要做

### 8.3 Heuristic Verifier (FSM-based)
- VLM 在 episode 开始时 route task to type (grasp, place, push 等)
- Finite State Machine: APPROACH → ALIGN → ENGAGE → MANIPULATE → RELEASE → SETTLE → DONE
- 每个状态有 transition conditions (基于 geometric features)
- 如果 progress stalls，可选调用 VLM 做 diagnosis
- **优点**: 简单，interpretable
- **缺点**: 需要 task decomposition，generalize 到 arbitrary task 难

Algorithm 4 (FSM Update) 的 transition logic 示例:
```
if σ = Approach then
    if engaged ∨ is_grasped then σ ← Align
else if σ = Align then
    if is_grasped ∧ aligned_with_target then σ ← Manipulate
    else if engaged ∧ not_aligned_with_target then σ ← Engage
    else if ¬engaged then σ ← Approach
else if σ = Engage then
    if aligned_with_target then σ ← Manipulate
    ...
```

### 结论

VERITAS (absolute pixel trace) 在所有 ablation 里最 effective，因为：
1. 最快 (VLM 只 call 一次)
2. 不需要 tracking
3. 对 quasi-static manipulation (大多数 table-top task) 足够

VLM-Constraints 更适合 dynamic scene (future work)。

---

## 9. 为什么这工作 now？背景与时间线

### Robotics foundation models 的成熟

- **RT-2** [6] (2023): VLA 把 web knowledge transfer 到 robot control。第一次证明 VLM-style pre-training 可以做 robot policy。
- **OpenVLA** [10] (2024): 开源 VLA。
- **$\pi_0$** [5] (2024): Flow-matching action head + mixture-of-transformers backbone，trained on thousands of hours diverse robot trajectories。这是 VERITAS 用的 generator。
- **$\pi_{0.5}$** [48] (2025): Open-world generalization version。
- **Open-X-Embodiment** [2]: 跨 embodiment 数据集。
- **DROID** [22]: Large-scale in-the-wild manipulation dataset。

### VLM embodied reasoning 的成熟

- **Gemini 2.5** [45]: Frontier VLM 有 strong spatial + embodied reasoning。这是 VERITAS verifier 用的 VLM。
- **SAM 2** [51]: Real-time segmentation。
- **Grounded SAM** [52]: Open-world grounding。

### Test-time compute scaling 在 LLM 的成功

- **Cobbe et al.** [16]: Training verifiers to solve math problems
- **Lightman et al.** [17]: Process reward models
- **Tree of Thoughts** [18]: Search with verifier
- **Snell et al.** [13]: Scaling test-time compute optimally
- **Are more LLM calls all you need?** [14]: Compound AI systems

这三条线在 2024-2025 converge，VERITAS 把它们 fuse 在一起。

---

## 10. 与 Robotics Prior Work 的对比

### vs. Human-in-the-loop 方法

- **DAGGER** [29]: 需要 expert 在 robot-visited states 上 re-label。VERITAS 用 verifier 替代 expert，完全 autonomous。
- **Shared autonomy** [26, 27, 28]: blend human control with autonomous execution。VERITAS 不需要 human control，纯 autonomous。
- **RAC** [20]: Decompose nominal execution and recovery phases，trigger human intervention on failure。VERITAS 在 inference-time 就 filter 掉 failures，不需要 recovery。
- **HITL RL** [23]: Precise manipulation via human-in-the-loop RL。VERITAS 不需要 RL，只 BC fine-tune。
- **Compliant Residual DAgger** [30]: Human corrections for contact-rich manipulation。VERITAS 用 verifier corrections。

### vs. Inference-time Steering 方法

- **V-GPS** [38]: Language-conditioned value function via offline RL。需要训练 value function，每个 task 都要训。VERITAS verifier 是 gradient-free VLM，zero-shot。V-GPS 论文：https://arxiv.org/abs/2410.20286
- **RoboMonkey** [35]: Single-step verifier via Gaussian fitting on action distribution。只 refine local action noise，不能 reason over alternative strategies。VERITAS 在 action chunk level reason，能比较不同 high-level plans。RoboMonkey 论文：https://robotlearningss24.github.io/
- **Scaling verification for VLA alignment** [36]: 比 RoboMonkey 更进一步，但仍然 single-step。
- **Do what you say** [37]: Steering VLA via runtime reasoning-action alignment verification。Text-based plan + action verification。VERITAS 是 visual trace based，不需要 text plan。
- **VLM-in-the-loop policy steering** [39]: Foresight to forethought，VLM 在 loop 里。VERITAS VLM 只 call 一次。
- **When to act, ask, or learn** [40]: Uncertainty-aware steering。
- **World model based steering** [39, 40]: 需要 world model predictions。VERITAS 不需要 world model。

### vs. Embodied Reasoning Methods

- **ECoT** [32]: Embodied chain-of-thought reasoning。需要 expensive annotations 在 pre-training。
- **CoT-VLA** [33]: Visual chain-of-thought for VLA。同样需要 annotations。
- **MolmoAct** [34]: Action reasoning models that reason in space。需要训练。

这些 methods 在 pre-training 时注入 reasoning，VERITAS 在 inference-time 外挂 verifier，**不动 base policy**，可以 plug-and-play 在任何 pre-trained VLA 上。

### vs. PIVOT

- **PIVOT** [49]: Iterative visual prompting for VLMs。用 action primitives + visual optimization。VERITAS 不用 primitives，用 pre-trained policy 作为 strong action prior。PIVOT 在 paper 的实验里完全失败 (0% success)，highlight 了 strong action prior 的必要性。

---

## 11. Limitations & Future Work (paper 自己提到的)

1. **Computational cost at latency-critical applications**: N 次 sampling 的 cost linear，虽然 N=5 可接受，但对 latency 极敏感场景仍有问题。
2. **Static visual trace for dynamic environments**: 当前 VERITAS verifier 在 episode 开始时生成 static trace。Quasi-static manipulation OK，highly dynamic scene (objects 快速移动) 会 struggle。VLM-Constraints variant 可以 address 这个。
3. **Upper bound by policy's exploration prior**: Verifier 只能在 policy 提出的 candidates 里选 best，不能 propose 新的 actions。如果 policy prior 完全不 cover 某个 solution，verifier 无法 discover。期待更好的 pre-trained policy 进一步提升。
4. **Distilling verifier rejection logic into value function**: Future work 可以把 verifier 的 rejection logic distill 成 learned value function，加速 inference。
5. **Joint generator-verifier optimization**: 现在 generator 和 verifier 是 decoupled 的，joint optimization 可能 improve sample efficiency during search。

---

## 12. Intuition: 为什么这能 work？我的几个 hypothesis

### Hypothesis 1: Pre-trained VLA 的 "exploration prior" 已经很强

$\pi_0$ 在 thousands of hours diverse data 上训练过，它的 action distribution $p(a|o,l)$ 已经 cover 大量 valid affordances。问题是 greedy decoding 只取 mode，丢弃了 distribution 里其他 valid 的 modes。VERITAS 通过 sampling + verification **重新激活**了这些 modes。这就是为什么 N=5 已经足够 —— distribution 里 top-5 candidates 已经 cover 大部分 valid strategies。

### Hypothesis 2: VLM 的 embodied reasoning 是 "free" supervision signal

Gemini 2.5 在海量 internet image + text 上训练，已经 internalize 了 "approach object", "grasp", "place on target" 这种 spatial reasoning。这个 reasoning 不需要 robotics training 就已经存在。VERITAS 把这个 reasoning 外化成 visual trace，作为 verification signal。这相当于 **从 VLM 的 internet-scale prior 里 distill supervision 给 robotics policy**，完全 bypass 了 robotics data collection bottleneck。

### Hypothesis 3: Verification 比 Generation 容易

LLM 里我们都知道 "verifying a solution is easier than generating one" (Process Reward Models 的工作证明了这点)。VERITAS 把这个 asymmetry exploit 到 robotics: 让 VLA generator (难) + VLM verifier (相对易) 协作。VLM 不需要 generate valid robot action (难，需要 robotics data)，只需要 generate visual trace (相对易，internet image 上有大量类似 reasoning) + 做 geometric check (易)。

### Hypothesis 4: On-policy data 是 self-improvement 的 key

为什么 20-100 条 autonomous data 能 match human data？因为：
1. **On-policy**: robot 自己生成的，state distribution 完全 match policy's exploration region，没有 covariate shift
2. **Success-filtered**: verifier 只保留成功的，quality 高
3. **Kinematically feasible**: 都是 robot 真正执行过的，不存在 expert demo 的 kinematic mismatch
4. **Diverse**: $\pi_0$ 是 multi-modal policy，sample 出的 trajectories cover 不同 strategies，比 single expert teleop 多样

这 4 点合起来让 autonomous data 的 information density 比 human demo 高。

---

## 13. 跟你之前 work 的 connection (Karpathy)

Karpathy 你应该对几个 connection 感兴趣：

### 13.1 与 "micrograd" / "nanoGPT" 的精神一致

VERITAS 的 verifier 极简 (geometric distance + VLM trace)，没有 trained value function，没有 RL，没有 complex reward shaping。这种 "minimal sufficient mechanism" 的设计哲学和你 nanoGPT [https://github.com/karpathy/nanoGPT] 的精神一致 —— 用最简洁的 implementation 揭示最核心的 mechanism。

### 13.2 与 "State of GPT" 演讲里的 test-time scaling

你在 "State of GPT" [https://www.youtube.com/watch?v=zxXyOlqaqb0] 里讲过 system 1 vs system 2 thinking，test-time compute 是 system 2 的 enabler。VERITAS 把这个 idea 具体化到 robotics: inference-time sampling + verification 就是 robot 的 system 2 thinking。

### 13.3 与 "Software 2.0" 的张力

你写过 "Software 2.0" [https://karpathy.medium.com/software-2-0-a64152b37c35] —— 神经网络取代 explicit code。VERITAS 是反过来的有趣 case：它把 neural network (VLM) 当成 "code" 用 (生成 explicit waypoint trace)，然后用 explicit geometric check 做 verification。这是 Software 1.5 —— hybrid of neural reasoning 和 explicit algorithmic verification。

### 13.4 与 "Lesson in unwinding old ideas" 的反思

你最近的 blog "A few useful things to know about machine learning" 强调 simple baselines 的力量。VERITAS 的 Heuristic verifier (FSM-based) 也能 beat base policy —— 这说明 generator-verifier paradigm 的 benefit 不依赖 fancy verifier，simple geometric check 就有 gain。这对 simple-baseline-first 的方法论是个支持。

---

## 14. 可能的 follow-up directions (我自己的猜想)

1. **Tree-of-thoughts for robot action sequences**: 不只是 Best-of-N，而是 tree search，允许 partial action chunks + branching + backtracking。Combine with Monte Carlo Tree Search。

2. **Self-distillation of verifier into value function**: 把 VLM verifier 的 scoring logic distill 成一个 small MLP value head，attached 到 policy。这样 inference-time 就不需要 VLM call，只需要一次 MLP forward。

3. **Curriculum from verifier signal**: 用 verifier score 作为 reward signal，做 RL fine-tuning (类似 RLHF [Schulman et al.])。但用 VLM-as-reward 而不是 human preference。

4. **Multi-modal verifier**: 不只是 visual trace，还有 force-torque sensing, audio (contact sound), proprioception history。Fuse 多模态 verification signal。

5. **Hierarchical verifier**: High-level VLM verifier (plan level) + mid-level geometric verifier (chunk level) + low-level control verifier (single-step)。Multi-resolution verification。

6. **Verifier-grounded RL**: 用 verifier score 做 reward shaping，做真正的 RL fine-tuning 而不只是 BC。这能让 policy **explore beyond** 当前 prior (解决 paper 里提到的 limitation 3)。

7. **Closing the loop with world model**: 用 world model predict action chunk 的 consequence，verifier 在 predicted consequence 上 score。这样不需要真的执行就能 verify，更进一步降低 inference cost。

8. **Active inference for sample selection**: 不只是 random sample N 个，而是 intelligent acquisition (类似 Bayesian optimization in action space)。

---

## 15. 一些可能你会 push back 的点

### 15.1 N=5 真的够吗？

Paper 在 simulation 上 sweep 到 N=8 saturate，real-world 用 N=5。但这个 saturation 可能在更难的 task 上 break —— 如果 task 难到 policy 的 prior 很差，top-5 可能都是 bad，N 需要更大。可以想象 long-horizon task 或者 OOD task 会 require N=20, 50, 100。

### 15.2 Static trace 的 robustness

Static visual trace 假设 scene 在 episode 内 quasi-static。如果中间 object 被人 move 了，或者 stochastic dynamics 导致 object 滑动，trace 就 stale 了。VLM-Constraints variant 有 dynamic resolution 但需要 tracking，可能 fail under occlusion。

### 15.3 Verification ≠ Generation 的边界

VLM verifier 依赖其 embodied reasoning prior。如果 task 涉及 VLM 从未见过的 physical interaction (e.g., 某种 special tool use，deformable object manipulation)，VLM 可能生成错误的 trace，导致 verifier **confidently wrong**。这是 VLM-as-verifier 的 fundamental limitation。

### 15.4 On-policy data 的 exploration coverage

VERITAS 只保留 successful trajectories，这意味着 policy 在它能 succeed 的 region 里 self-improve，但 **不能 expand 到它 currently fails 的 region**。这是 pure exploitation 没有 exploration 的根本限制。RL fine-tuning (用 verifier as reward) 可能 address 这个，但 paper 没做。

---

## 16. Final summary: paper 的贡献 hierarchy

**Level 1 (Mechanism)**: Generator-verifier framework for robot policies, with Best-of-N inference-time steering.

**Level 2 (System)**: Visual verifier design (VLM trace + geometric scoring) that's gradient-free, zero-shot, and runs in <1ms per verification step.

**Level 3 (Insight)**: Treating inference-time verification as **data engine** rather than just performance boost. Successful verified trajectories form a flywheel that distills VLM reasoning into policy weights via offline BC.

**Level 4 (Claim)**: Autonomous verified data can match human expert demonstrations in data efficiency (20-100 trajectories), enabling scalable policy improvement without human supervision.

这个 hierarchy 让 paper 从一个 engineering trick (inference-time steering) 上升到一个 paradigm (autonomous self-improvement via verification)。这是 robotics foundation models 时代的 "synthetic data + RLHF" 等价物 —— 用 VLM reasoning 替代 human supervision，用 inference compute 替代 data collection labor。

---

## References (web links)

主要 paper:
- VERITAS: https://veritas-improvement.github.io
- $\pi_0$: https://physicalintelligence.company/blog/pi0
- $\pi_{0.5}$: https://physicalintelligence.company/blog/pi05
- DROID: https://droid-dataset.github.io
- SIMPLER: https://simpler-env.github.io

Inference-time scaling in LLMs:
- Snell et al. "Scaling test-time compute": https://arxiv.org/abs/2408.03314
- Lightman et al. "Let's verify step by step": https://arxiv.org/abs/2305.20050
- Cobbe et al. "Training verifiers": https://arxiv.org/abs/2110.14168
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Are more LLM calls all you need?: https://arxiv.org/abs/2412.04103

Robotics foundation models:
- RT-2: https://robotics-transformer2.github.io
- OpenVLA: https://openvla.github.io
- Open-X-Embodiment: https://robotics-transformer-x.github.io
- GR00T N1: https://arxiv.org/abs/2503.14734

Verifier baselines:
- V-GPS: https://arxiv.org/abs/2410.20286
- PIVOT: https://arxiv.org/abs/2402.07872
- RoboMonkey: https://robotlearningss24.github.io/

VLMs:
- Gemini 2.5: https://arxiv.org/abs/2503.21903
- SAM 2: https://arxiv.org/abs/2408.00714
- Grounded SAM: https://arxiv.org/abs/2401.14159

Imitation learning foundations:
- DAGGER: https://arxiv.org/abs/1011.0686
- ACT: https://tonyzhaozh.github.io/aloha/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

Karpathy references:
- nanoGPT: https://github.com/karpathy/nanoGPT
- State of GPT: https://www.youtube.com/watch?v=zxXyOlqaqb0
- Software 2.0: https://karpathy.medium.com/software-2-0-a64152b37c35
- A few useful things: https://karpathy.bearblog.dev/a-few-useful-things-to-know-about-machine-learning/

---

这篇 paper 我觉得是 2025-2026 robotics foundation models 时代的一个标志性工作。它不发明新 architecture，而是把 inference-time scaling + VLM-as-judge + self-improving data flywheel 三条线巧妙 fuse，给出了 robotics 的 "RLHF moment"。值得仔细读，特别是 verifier design 的 elegance，以及 data flywheel 的 conceptual beauty。
