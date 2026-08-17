---
source_pdf: Why AI systems don’t learn and what to do about it.pdf
paper_sha256: 265f1a631054aca96d12f5dc66d10d035655fb1e8abd2e6318b8403eb951cb92
processed_at: '2026-08-13T04:26:10-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

## 一句话总结

现在的 AI 就像一个 **被填鸭式教育的大学生**——考前突击刷题能考高分，但毕业后再也不学习了，遇到新情况就傻眼。paper 说，真正聪明的系统应该像小孩一样，**自己会学、自己会探索、自己知道什么时候该看、什么时候该做**。

---

## 问题出在哪？

你 Karpathy 自己早就说过：LLM 撞上 data wall 了，text 数据快用完了，模型部署后就不长了。这篇 paper 把这个问题讲得更透：

**现在的 AI 训练流程本质上是一个手工流水线**。一帮 PhD 花 6 个月 curate data，设计 pretraining recipe，调 RLHF reward，最后产出一个 frozen model。这个 model 部署以后，它**不会从自己的经验里学任何东西**。用户用得不爽？对不起，等下一个版本吧，又是半年后。

这和生物完全相反。小孩从出生第一天就在学，而且**自己决定学什么**：盯着人脸看、往嘴里塞东西、摔了再爬起来。没人给小孩做 "pretraining + SFT + RLHF" 的 pipeline。

---

## 三个 System 是什么？

### System A：看中学

就是 self-supervised learning。BERT 预测 next token，SimCLR 做 contrastive，JEPA 预测 latent representation——这些都是 System A。

数学上很简单：

$$\theta^* = \arg\min_\theta \mathbb{E}_{x \sim \mathcal{D}} \mathcal{L}(f_\theta(x_{\text{in}}), x_{\text{tar}})$$

- $\theta$：网络参数
- $x_{\text{in}}$：输入（比如 masked image）
- $x_{\text{tar}}$：目标（被 mask 掉的部分）
- $\mathcal{D}$：数据分布
- $\mathcal{L}$：loss function
- $f_\theta$：你的 neural network

**优点**：能吃海量数据，能学到 hierarchical 抽象 representation。

**缺点**：
1. $\mathcal{D}$ 和 $\mathcal{G}$（task generator）都要人手设计
2. 学到的东西和 action 脱节，grounding 困难
3. 分不清 correlation 和 causation——光看不做，你永远不知道 "公鸡打鸣" 和 "太阳升起" 谁因谁果

参考 [Schölkopf 2019 on causality](https://arxiv.org/abs/1911.10500)

### System B：做中学

就是 reinforcement learning。Agent 在 world 里交互，通过试错优化 reward：

$$J(\pi) = \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]$$

- $s_t$：world state
- $a_t$：agent 的 action
- $r$：reward function
- $\gamma$：discount factor（远期 reward 打折）
- $\pi$：policy，给定 state 输出 action
- $T$：时间 horizon

**优点**：grounded in control，能从 sparse reward 学，能通过 search 发现新解法（AlphaGo）。

**缺点**：
1. sample efficiency 极差——学个简单任务要百万次交互 ([Dulac-Arnold 2021](https://link.springer.com/article/10.1007/s10994-021-05961-4))
2. action space 一大就炸——robotics 有 200-300 个 DOF
3. reward function 在真实世界很难定义 ([Amodei 2016 Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565))

### System M：总指挥

这是 paper 的核心创新。你可以把它理解成大脑里的 **prefrontal cortex**，或者 SDN 里的 **control plane**。

它不直接处理 high-bandwidth 数据流（pixels、motor commands），它处理的是 **low-bandwidth telemetry**：

$$\pi(a^m | s^m)$$

- $s^m$：meta-state，三类信号：
  - **Epistemic**：内部状态（prediction error、uncertainty、novelty）
  - **Species-specific**：进化硬编码的 detector（人脸、直视眼神、looming 威胁）
  - **Somatic**：身体信号（sleep、pain、hunger、stress）
  
- $a^m$：meta-action，就是 **打开/关闭数据通路**：
  - 把 System A 的输出接到 System B 的输入
  - 把 memory replay 接到 System A 做 offline learning
  - 断开 sensory input（睡觉时）
  - 给 System B 一个 intrinsic reward（好奇）

**最反直觉的 insight**：System M 的核心 routing policy 是 **hardwired 的，不需要学**。它是 evolutionary fixed transition table。

为什么？看生物证据：
- 新生儿看人脸偏好 = 硬编码的 "dark T pattern on white" detector ([Johnson et al. 1991](https://doi.org/10.1016/0010-0277(91)90045-6))
- 睡觉时 sensory gating = 硬编码
- Pain 打断所有 plan = 硬编码
- Stress 低时 explore，高时 exploit = 硬编码

这些 routing 决策不需要 lifetime 学，它们是 **evolutionary prior**。真正通过 learning 学的是 System A 和 B 的 weights。这把 meta-learning 问题从 "学一个 meta-policy" 简化到 "evolutionarily search 一个 meta-policy"——搜索空间大大缩小了。

---

## A 帮 B：让 RL 不那么蠢

RL 的核心痛点是 search space 太大。System A 通过三个方式帮忙：

### 1. 压缩 state 和 action space

[CURL (Laskin 2020)](https://arxiv.org/abs/2004.04136) 用 contrastive learning 把 pixels 压成 compact representation，喂给 RL agent，性能和 hand-coded state 一样。

[DIAYN (Eysenbach 2019)](https://arxiv.org/abs/1806.08642) 学 latent skills。[Action chunking (Li 2025)](https://arxiv.org/abs/2503.00744) 把连续 actions 分组成 macro-actions。

[Radosavovic 2024 humanoid locomotion](https://arxiv.org/abs/2410.03654) 先 generative 预训 sensorimotor trajectories，再 RL，迁移到真实复杂地形。

### 2. 提供 predictive world model

这是把 model-free RL 升级成 model-based RL 的关键。

经典：
- [PlaNet (Hafner 2019)](https://arxiv.org/abs/1811.04551)
- [Dreamer (Hafner 2020)](https://arxiv.org/abs/1912.01603)
- [MuZero (Schrittwieser 2020)](https://www.nature.com/articles/s41586-020-03051-4)

最新：[V-JEPA 2 (Assran 2025)](https://arxiv.org/abs/2506.09985) ——在 latent space 预测，不生成 pixels，modeling physics 更好，能快速 transfer 到 robotics。这和 LeCun 一直推的 JEPA philosophy 一致：**别在 pixel space 生成，在 latent space 预测**。

### 3. 提供 intrinsic reward

[Pathak 2017 ICM](https://arxiv.org/abs/1705.05363) 用 prediction error 当 curiosity reward。不过 robotics 上应用还有限。

---

## B 帮 A：让 SSL 不那么瞎

这方向讨论少，但很重要。System A 依赖 passive data，容易被垃圾数据毒化 ([Lavechin 2023 BabySLM](https://arxiv.org/abs/2306.09186))。

Gibson 的话：**"We see in order to move and we move in order to see"**。

两种方式：

### 1. Active SSL

System B 通过 eye/head movement 选择 System A 觉得 "interesting" 的部分。"Interesting" 由 System A 自己定义——uncertainty 高的、prediction error 大的、learning progress 快的。

[Kidd 2012 Goldilocks effect](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0036399)——婴儿 attend 中等复杂度的 visual sequence，太简单无聊，太难也无聊。

### 2. Goal-directed SSL

System B 优化自己的 task reward，顺带给 System A 提供 grounded data。[Pong 2020 Skew-Fit](https://arxiv.org/abs/1912.02888) 是个好例子。

---

## Imitation Learning：A 和 B 的深度 integration

Figure 3 展示了三种 mode：

- **Self Play**：B 产生 trajectories → A 学 world model + 提供 intrinsic reward
- **Social Observation**：B directs attention 到 peers → A 从 complex trajectories 推 latent actions  
- **Retargeted Imitation**：A 映射 exocentric actions/states 到 egocentric → B 做 goal-directed behavior

这里有个 **retargeting problem**：demonstrator 和 learner 身体不同，image space ≠ motor space。

当前 robotics 用 teleoperation 绕开（[Mobile ALOHA (Fu 2024)](https://arxiv.org/abs/2401.02117)），但限制了 scale。[VPT (Baker 2022)](https://arxiv.org/abs/2210.02717) 和 [LAWM (Tharwat 2025)](https://arxiv.org/abs/2509.18428) 学 latent actions，减少 teleoperation 依赖，但还做不到 zero teleoperation。

---

## 高级模式：Communication 和 Imagination

只有大脑子物种才有的能力。

### Learning from Communication

从 basic attention（看到别人玩玩具也感兴趣）→ behavioral copying → guided learning（parent 带 child）→ higher-level social learning（verbal instruction）。

关键是 **epistemic vigilance** ([Sperber 2010](https://onlinelibrary.wiley.com/doi/10.1111/j.1468-0017.2010.01394.x))——评估信息源可靠性。小孩会判断 "这个大人靠不靠谱"再决定学不学。当前 AI 完全没有这能力，RLHF 训练时人手 curate reliable sources。

### Learning from Imagination

三种层次：

1. **Memory replay at rest**：rodents 在决策点后，place cells 按正/逆序重激活，速度更快，关联 RL value update ([Foster & Wilson 2006](https://www.nature.com/articles/nature04587))

2. **Memory replay during sleep**：non-REM sleep，recent experience compressed replay，形成 long-term schemas ([Wilson & McNaughton 1994](https://www.science.org/doi/10.1126/science.8036517))

3. **Long-horizon planning**：无 trial-and-error 的 problem solving，counterfactual reasoning

System M 实现 imagination 的方式：把 A/B 切到 inference mode，input 从 memory 而非 sensors 来，output 去 internal simulation。在成功的 imagined trajectories 上 trigger learning。

这就是 [Dreamer (Hafner 2020)](https://arxiv.org/abs/1912.01603) 在做的事——在 learned world model 里 rollout，从 imagined trajectories 学 policy。

---

## 最难的部分：怎么 bootstrap？

如果 A 依赖 B 产生的数据，B 依赖 A 提供的 representation，M 依赖 A/B 的 uncertainty signal——**三方互相依赖，怎么开始？**

### 生物的答案：Evo/Devo 分工

生物不 start from random initialization ([Zador 2019](https://www.nature.com/articles/s41467-019-11786-6))。Animal inherit species-typical nervous system，通过 developmental time 展开。这 inherited structure 提供 inductive biases。

机制：synaptic growth/pruning、critical periods、spontaneous neural activity、progressive DOF increase（婴儿出生时 myopic，muscle 高度 synergistic，相当于 built-in curriculum）。

### AI 的 Evo/Devo 形式化

设 meta-parameter $\phi$ = genetic code。

**Inner loop (developmental scale)**：A 和 B 通过和 environment 交互更新，M fixed：

$$A_{i+1}, B_{i+1} = \text{Update}(M, A_i, B_i, \text{Env})$$

**Outer loop (evolutionary scale)**：$\phi$ 通过 fitness function $\mathcal{L}$ 优化：

$$\phi_{t+1} = \arg\min_{\phi_t} \mathcal{L}(A_0:A_K, B_0:B_K)$$

约束：
$$A_0, B_0, M = \text{Init}(\phi_t)$$

这是 **bilevel optimization** ([Sinha 2018 survey](https://ieeexplore.ieee.org/document/7930004))。

### Bilevel Optimization 的挑战

1. **Outer level 数据稀疏**：一次 life cycle = 一个 data point。优化 $\phi$ 要跑百万级 simulated life cycles，每个又包含百万级 datapoints。需要 memory/compute efficiency 极大突破。

2. **Scalability**：bilevel optimization 在 large architecture 上有 severe issues ([Lorraine 2020](https://arxiv.org/abs/1911.02590); [Metz 2021 "Gradients are not all you need"](https://arxiv.org/abs/2111.05803))

3. **Dynamic system optimization**：优化目标是 dynamic system (learner + environment)，不是 static loss。

**Proposed solution**：Evolutionary Curriculum——逐渐增加 environment diversity 和 unpredictability，让三个 component co-evolve。这和 [Leike 2017 AI Safety Gridworlds](https://arxiv.org/abs/1711.09883) 呼应。

---

## 澄清：System 1/2 vs System A/B

Appendix A 的好澄清：Kahneman 的 System 1/2 是 **modes of inference** (fast vs slow)，paper 的 System A/B 是 **modes of learning** (observation vs action)。它们是正交的：

| | System A (observation) | System B (action) |
|---|---|---|
| **System 1 (fast)** | predictive coding, statistical learning | policy learning |
| **System 2 (slow)** | counterfactual reasoning, causal learning | planning |

A 可以用 System 1 (一次 backprop) 也可以用 System 2 (先 imagination steps)；B 可以用 System 1 (learned reflexive policy) 也可以用 System 2 (tree search / chain of thought)。

---

## 我的 Intuition 和 Takeaway

### 1. 这篇 paper 真正的价值

它**重新定义了 problem space**。把当前 AI 的 limitation 重新 frame 成三个明确 roadblock：
- Paradigm fragmentation (SSL/RL siloed)
- Learning externalization (MLOps 由人做)
- Bootstrapping difficulty (chicken-egg-rooster)

这 frame 非常 clean。你之前讲 "software 2.0" 时隐隐触及第 2 点——现在的 training pipeline 本质是 software 1.5，一半人写代码，一半 gradient descent。

### 2. System M 是最 sharp 的 insight

System M **不需要学，它是 hardwired evolutionarily**。这非常反直觉。我们一直以为 meta-learning 应该是 learned 的。但生物证据很 clear：

- 婴儿 face preference = 硬编码 "dark T pattern on white" detector
- 睡觉 sensory gating = 硬编码  
- Pain 打断 plan = 硬编码
- Stress 低 explore 高 exploit = 硬编码

这些 routing decisions 不需要 lifetime 学。真正学的是 A 和 B 的 weights。这把 meta-learning 问题从 "学 meta-policy" 简化到 "evolutionarily search meta-policy"——搜索空间大大缩小。

### 3. 和 LeCun 之前工作的关系

这 paper 是 [LeCun 2022 "A Path Towards Autonomous Machine Intelligence"](https://openreview.net/forum?id=BZ5a1r-kVsf) 的延续：
- 那篇 propose H-World model + actor + critic + configurator
- 这篇把 configurator 明确成 System M，强调 hardwired 性质
- 加入了 Evo/Devo 框架解决 bootstrap 问题
- 加入了 cognitive science 大量证据

### 4. 我对 bilevel optimization 的担忧

paper 第 4.4 节坦承 bilevel optimization 在 large scale 有 severe issues。[Metz 2021](https://arxiv.org/abs/2111.05803) 已经指出 meta-gradient 在大网络上有 pathological behavior。

要让这工作，可能需要：
- 大幅简化 architecture（回到 small agents）
- 大量 simulator innovation（既要 realistic 又要 fast）
- 可能需要 evolutionary algorithm (gradient-free) 而非 gradient-based meta-learning
- 可能需要 neuroevolution 方向的突破 ([Real 2019](https://arxiv.org/abs/1802.01548))

### 5. 实际 path forward

paper 最后 section 6 提到 test-time training、adaptive retrieval、mixture of world models——但这些是 minor variations，不是真正的 autonomous learning。

真正 breakthrough 我觉得会来自：
1. **V-JEPA 2 路线**：在 latent space 学 world model，不生成 pixels。这是 System A 的正确形式。
2. **Robotics 上 latent action pretraining**（LAWM 路线）：用 unlabeled video 学 action representation，减少 teleoperation 依赖。
3. **Bilevel optimization 的 hardware/simulator breakthrough**：如果能 cheaply simulate embodied life cycles，Evo/Devo 就 feasible。
4. **Meta-control 的简化实现**：从 SDN-like routing 开始，用 simple epistemic signals（prediction error）作为 trigger。

### 6. 为什么这很重要

paper 的核心 message：**autonomous learning 是 robust AI 的 necessary condition**。在 dynamic、non-stationary、heavy-tailed 的 real world，没有 autonomous learning 的 system 注定 stale。

当前 LLM 的 scaling law 让人兴奋，但 scaling text data 有 ceiling。真正的 next frontier 是 **让 agent 自己在 world 里持续学习**。这和 [Silver & Sutton 2025 "Era of Experience"](https://deepmind.google/discover/blog/welcome-to-the-era-of-experience/) 的论点完全一致。

### 7. 几个 immediate research question

1. System M 的最小可行实现？需要哪些 hardwired meta-signals？
2. Bilevel optimization 在多大的 agent 上可行？需要怎样的 simulator？
3. 如何设计 evolutionary curriculum？
4. 如何 evaluate autonomous learning system？Unit test（单独 component）vs integration test（end-to-end）？
5. 如何在 System M 中实现 epistemic vigilance（评估信息源可靠性）？
6. Imagination mode 的具体实现——如何决定 replay 什么、imagine 什么？

paper 说 "decades away"。但这方向是对的。这篇 paper 像 1986 年的 backpropagation paper——现在看起来 obvious，但需要十几年才被 realize。

### 8. 推荐你关注的 work

- [V-JEPA 2 (Assran 2025)](https://arxiv.org/abs/2506.09985) ——System A + B 初步 integration
- [DreamerV3 (Hafner 2023)](https://arxiv.org/abs/2301.04104) ——imagination-based learning
- [LAWM (Tharwat 2025)](https://arxiv.org/abs/2509.18428) ——latent action pretraining
- [Lake et al. 2017](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/building-machines-that-learn-and-think-like-people/959DB1D1A0A4C1A2C8C2E5F4F2A2C5A5) ——cognitive science 视角
- [LeCun 2022 position paper](https://openreview.net/forum?id=BZ5a1r-kVsf) ——架构 blueprint
- [Zador 2019](https://www.nature.com/articles/s41467-019-11786-6) ——为什么不能从 random init 开始

如果你想 build intuition，我推荐先吃透 V-JEPA 2 + DreamerV3，这两个 system 已经某种程度上实现了 System A + B 的初步 integration，离 full A-B-M 架构还有距离，但是最好的 starting point。

---

# 深度解读：Why AI systems don't learn and what to do about it

这篇paper是 Emmanuel Dupoux, Yann LeCun, Jitendra Malik 在2026年3月发表的, 核心论点非常sharp: **当前的AI systems本质上不会学习, 学习被外包给人类专家了**. 这和你（Karpathy）以及 Ilya Sutskever 之前提到的 "data wall", "Era of Experience" 的反思 ([Silver & Sutton 2025](https://deepmind.google/discover/blog/welcome-to-the-era-of-experience/)) 直接呼应. 

---

## 1. 核心诊断: 为什么说现在的AI"不学习"?

paper的第一张Figure 1对比了两个场景:
- **Standard ML (左)**: 一群 research engineers + data scientists 像流水线工人一样收集、清洗、curate data, 每个组件用 specifically engineered loss/reward 训练. 部署后模型 frozen, 不再学习.
- **Autonomous ML (右)**: agent 直接和 world 交互, 数据由 agent 自己通过不同 learning modes 生成, 有一个 meta-controller 在 real world 中持续学习.

关键insight: 现在的 LLM pipeline (pretrain → SFT → RLHF) 本质上是 **rigidly fixed by human engineers**, 而儿童在出生第一天就开始 **fluidly switch** between observation 和 action. paper 把这种 externalization of learning 叫做 "learning is outsourced to human experts instead of being an intrinsic capability".

还有 domain mismatch 问题. 训练数据从 internet lift 出来, 部署到 real world 必然遇到 heavy-tailed + non-stationary 的分布偏移 ([Geirhos et al. 2020](https://www.nature.com/articles/s42256-020-00257-z)). 现在的 pretrain + fine-tune 只是 mitigation, 不是 fix, 因为 system 从一开始就没被设计成 "可以在 raw data 上被 fine-tune".

---

## 2. 三个System的数学定义

### 2.1 System A (Learning from Observation)

这是 passively accumulating sensory input 的学习, 包括 SSL, world modeling, language modeling.

**一般形式**:

给定 data distribution $x \sim \mathcal{D}$, 定义一个 task generator $\mathcal{G}$:

$$(x_{\text{in}}, x_{\text{tar}}) = \mathcal{G}(x) \quad (1)$$

这里:
- $x_{\text{in}}$: input 部分 (例如 masked image patches, 或历史 token sequence)
- $x_{\text{tar}}$: target 部分 (例如被 mask 的 patches, 或 next token)
- $\mathcal{G}$: 比如 BERT 的 masking, GPT 的 next-token split, JEPA 的 latent prediction

学习 representation $z = f_\theta(x)$, 最小化:

$$\theta^\star = \arg\min_\theta \mathbb{E}_{x \sim \mathcal{D}} \mathcal{L}(f_\theta(x_{\text{in}}), x_{\text{tar}}) \quad (2)$$

- $\theta$: network 参数
- $\mathcal{L}$: 可以是 contrastive loss (SimCLR/MoCo), predictive loss (JEPA), masked prediction (BERT/HuBERT)
- $\mathcal{D}$: data distribution, 在 standard ML 中需要人手 curate

paper 的 Table 1 列了从 speech (CPC, HuBERT), language (GPT, BERT), images (SimCLR, DINO, I-JEPA), video (V-JEPA), vision+language (CLIP, Flamingo) 的各种 System A 实例.

**Strengths**: scale 好, 能学到 hierarchical 抽象 representation, 支持 transfer.

**Limitations** (这是关键):
1. 需要 $\mathcal{D}$ 和 $\mathcal{G}$, 都需要 human expertise 设计
2. 没有 built-in mechanism 决定 "什么数据有用, 该去 acquire 什么" (没 active learning)
3. representations 和 action 脱节, grounding 困难
4. 只基于 observation, 难以区分 correlation 和 causation ([Schölkopf 2019](https://arxiv.org/abs/1911.10500))

### 2.2 System B (Learning from Action)

这是通过 interaction 优化 goal 的学习, 即 RL.

**MDP 形式**:

$$\text{Maximize } J(\pi) = \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right] \quad (3)$$

变量解释:
- $s_t$: world state at time $t$
- $a_t$: action taken by agent
- $r(s_t, a_t)$: reward function
- $\gamma \in [0, 1)$: discount factor, 让远期 reward 折减
- $T$: time horizon
- $\pi(\cdot | s_t)$: agent's policy, 给定 state 输出 action 分布
- $M(s_{t+1} | s_t, a_t)$: world dynamics, transition function

paper 的 Table 2 对比了 5 个 System B 范式:
- **Control Theory**: $M$ fixed, $\pi$ 解析推导. 例: 脊髓反射, 眼跳
- **Adaptive Control**: $M$ 在线估计, $\pi$ 解析推导. 例: motor adaptation ([Shadmehr & Mussa-Ivaldi 1994](https://www.jneurosci.org/content/14/5/3208))
- **Model-free RL**: 没有 $M$, $\pi$ 是 NN. 例: habitual actions
- **Model-based RL**: $M$ 和 $\pi$ 都是 NN, 可以 unroll. 例: goal-directed behavior
- **Planning**: simulator 在 inference 时 search. 例: detour planning, mental simulation

**Strengths**: grounded in control, 能从 sparse/delayed feedback 学, 能通过 search 发现 novel solutions (如 AlphaGo).

**Limitations**:
1. 极度 sample-inefficient ([Dulac-Arnold et al. 2021](https://link.springer.com/article/10.1007/s10994-021-05961-4))
2. 高维 open-ended action space 困难
3. 依赖 well-specified reward function, naturalistic setting 几乎拿不到 ([Amodei et al. 2016](https://arxiv.org/abs/1606.06565))

---

## 3. A 和 B 的相互帮助 (这是 paper 最有意思的部分)

### 3.1 System A 帮 System B

这个方向的核心 motivation: 在 real world, action space 维度约 200-300 (robotics), state space 几乎无限大. System A 通过三个 mechanism 压缩 search space:

**(1) Abstract representations of states and actions**

CURL ([Laskin et al. 2020](https://arxiv.org/abs/2004.04136)) 用 contrastive vision pretraining 把 pixels 压成 compact representation, 喂给 RL agent, 达到 hand-coded state 相当的性能. 

对 action space, DIAYN ([Eysenbach et al. 2019](https://arxiv.org/abs/1806.08642)) 学习 latent skills, action chunking ([Li et al. 2025](https://arxiv.org/abs/2503.00744)) 把连续 actions 分组. Radosavovic et al. 2024 的 [humanoid locomotion](https://arxiv.org/abs/2410.03654) 先用 generative modeling 预训 sensorimotor trajectories, 再做 RL, 转移到 real challenging terrain.

**(2) Predictive World Models**

这是把 model-free RL 升级成 model-based RL 的关键. 代表作:
- [PlaNet (Hafner et al. 2019)](https://arxiv.org/abs/1811.04551)
- [Dreamer (Hafner et al. 2020)](https://arxiv.org/abs/1912.01603)
- [MuZero (Schrittwieser et al. 2020)](https://www.nature.com/articles/s41586-020-03051-4)
- [V-JEPA 2 (Assran et al. 2025)](https://arxiv.org/abs/2506.09985) - 在 latent space 预测, 能 modeling physics, 快速 transfer 到 robotics

关键 insight: V-JEPA 2 不在 pixel space 生成, 而在 latent space 预测, 这避免了 generative model 在 high-dimensional pixel space 的 sample efficiency 问题. 这和 LeCun 一直 push 的 JEPA philosophy 一致.

**(3) Intrinsic reward signals**

System A 提供 prediction error, uncertainty, novelty 作为 intrinsic reward, 解决 exploration/exploitation dilemma. [Pathak et al. 2017](https://arxiviv.org/abs/1705.05363) 的 ICM 是经典, 但在 robotics 上应用还有限.

### 3.2 System B 帮 System A

这方向被讨论得少, 但很重要: System A 依赖 passive/static data, 容易被 noisy/irrelevant data 毒化 ([Lavechin et al. 2023](https://arxiv.org/abs/2306.09186)).

Gibson 的 [active perception](https://en.wikipedia.org/wiki/James_J._Gibson): "We see in order to move and we move in order to see" - 这是 System B 帮 A 的经典陈述.

**两种方式**:

**(a) Active self-supervised learning**: System B 通过 eye/head movement 选择 System A 觉得 "interesting" 的 sensory portion. "Interesting" 由 System A 自己定义 (uncertainty, prediction error, learning progress). 这像 [Kidd et al. 2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0036399) 的 Goldilocks effect - 婴儿 attend 中等复杂度的 visual sequence.

**(b) Goal-directed SSL**: System B 优化自己的 task reward, 顺带给 System A 提供 grounded data. 例: [Pathak et al. 2017](https://arxiv.org/abs/1705.05363) 和 [Pong et al. 2020](https://arxiv.org/abs/1912.02888).

### 3.3 Imitation Learning 作为 deep integration 的例子

Figure 3 展示了 imitation learning 的三种 mode:

- **(a) Self Play**: System B 产生 trajectories → System A 学 world model + 提供 intrinsic reward
- **(b) Social Observation**: System B directs attention 到 peers → System A 从 complex trajectories infers latent actions
- **(c) Retargeted Imitation**: System A 映射 exocentric actions/states 到 egocentric → System B 做 goal-directed behavior

这里有一个 "retargeting problem" - demonstrator 和 learner 身体不同, image space 和 motor space 不一致. 当前 robotics 用 teleoperation 绕开 ([Fu et al. 2024 Mobile ALOHA](https://arxiv.org/abs/2401.02117)), 但这限制 scale. [Baker et al. 2022 VPT](https://arxiv.org/abs/2210.02717) 和 [Tharwat et al. 2025 LAWM](https://arxiv.org/abs/2509.18428) 学 latent actions, 但还做不到 zero teleoperation.

---

## 4. System M: Meta-control (核心创新)

这是 paper 的核心 contribution. Figure 4 画了一个像 SDN (Software-Defined Networking) 的架构:

- **Data Plane**: System A, System B, episodic memory 之间用 high-bandwidth streams 传 sensory input, motor command, latent representation
- **Control Plane**: System M 处理 low-bandwidth telemetry (prediction error, uncertainty, somatic signals), 输出 meta-actions 来 dynamically open/close data pathways

### 4.1 数学形式

System M 是一个 meta-policy:

$$\pi(a^m | s^m)$$

- $s^m$: meta-state, 三种类型:
  1. **Epistemic signals**: 监控内部 cognitive components (confidence, prediction error, learning gain, novelty)
  2. **Species-specific signals**: 进化 hardwired 的 detectors (direct gaze, dominance display, looming stimuli, heights)
  3. **Somatic signals**: 来自 physical body (energy level, pain, arousal)
  
- $a^m$: meta-action, 包括:
  1. 连接/断开 subsystem 的 input/output streams
  2. 开/关 subsystem 的不同 mode (learning, inference, optimization)
  3. 给 subsystem 提供 targets 或 internal rewards
  4. 访问 episodic memory 做 replay 或 batch learning

**关键 claim**: System M 的 core routing policy 是 **hardwired** - 即 evolutionary fixed transition table, 决定 when to explore, when to plan, when to act. 这和 System B 的 policy (lifetime 学出来的) 不同.

### 4.2 生物类比 (Table C.1)

paper 在 Appendix C 列了详尽的 meta-states 生物例子, 非常有启发:

| Meta-state 类型 | 例子 | Effect |
|---|---|---|
| Species-specific | Faces (dark T pattern on white) | Input Selection (Johnson et al. 1991) |
| Species-specific | Direct gaze | Learning Efficacy boost (Farroni et al. 2002) |
| Species-specific | Looming stimuli | Mode Control → freezing (Fanselow 1994) |
| Epistemic | Prediction error | Mode Control → exploration (Dayan & Balleine 2002) |
| Epistemic | Intermediate complexity | Input Selection (Kidd et al. 2012 Goldilocks) |
| Somatic | Sleep | 全 mode 切换: sensory disconnect + memory replay |
| Somatic | Pain | 中断所有 goal-directed plan, 切到 reactive |
| Somatic | Stress (low) | world modeling simulation, exploration |
| Somatic | Stress (high) | reactive policies, exploitation |

非常有意思: 大部分 meta-signals 计算很 simple. 比如婴儿 face preference 其实就是 "dark T pattern on white background". 这暗示 meta-controller 不需要复杂, 可以从简单 priors emerge.

### 4.3 System M 的关键高级模式

Appendix B 讨论了两个只有大脑物种才有的 mode:

**(1) Learning from Communication**: 从 basic attention (看到别人玩玩具也感兴趣) → behavioral copying → guided learning (parent 带 child) → higher-level social learning (verbal instruction). 关键是 **epistemic vigilance** ([Sperber et al. 2010](https://onlinelibrary.wiley.com/doi/10.1111/j.1468-0017.2010.01394.x)) - 评估信息源的可靠性. 当前 AI 完全没有这能力, RLHF 训练时人手 curate reliable sources.

**(2) Learning from Imagination**:
- **Memory replay at rest**: rodents 在决策点或 reward 后, place cells 按正/逆序重激活, 速度更快. 和 RL value function update 关联 ([Foster & Wilson 2006](https://www.nature.com/articles/nature04587))
- **Memory replay during sleep**: 非 REM sleep, recent experience compressed replay, 长 episode 包括 novel combinations. 和 memory consolidation, schema formation 关联 ([Wilson & McNaughton 1994](https://www.science.org/doi/10.1126/science.8036517))
- **Long-horizon planning**: 无 trial-and-error 的 problem solving, counterfactual reasoning

System M 实现 imagination mode 的方式: 把 System A/B 切到 inference mode, input 从 memory 而非 sensors 来, output (actions) 去 internal simulation. 在成功的 imagined trajectories 上 trigger learning.

---

## 5. Evo/Devo 双层优化 (解决 chicken-and-egg 问题)

这是 paper 第 4 节, 也是最难的部分. 

### 5.1 问题陈述

如果:
- System A 依赖 action-generated data 来 acquire grounded representations
- System B 依赖 perceptual structure 来 guide efficient action
- System M 依赖 well-calibrated uncertainty/error signals 来 orchestrate

那怎么初始化? 这是 chicken-egg-rooster 问题.

### 5.2 生物的答案: Evo/Devo 分工

生物不 start from random initialization ([Zador 2019](https://www.nature.com/articles/s41467-019-11786-6)). Animal inherit species-typical nervous system, 通过 developmental time 展开. 这 inherited structure 提供 inductive biases, 决定:
- 什么能学
- 多快学
- 通过什么 modality 学

机制包括: synaptic growth/pruning, temporally regulated plasticity, critical periods, spontaneous neural activity, progressive DOF increase.

### 5.3 AI 的 Evo/Devo 形式化 (Figure 5)

设 meta-parameter $\phi$ = genetic code. At "birth", $\phi$ specify architecture $(A_0, B_0, M_0)$ 的初始参数.

**Inner loop (developmental scale)**: A 和 B 通过和 environment 交互更新, M fixed:

$$A_{i+1}, B_{i+1} = \text{Update}(M, A_i, B_i, \text{Env})$$

**Outer loop (evolutionary scale)**: $\phi$ 通过 fitness function $\mathcal{L}$ 优化:

$$\phi_{t+1} = \arg\min_{\phi_t} \mathcal{L}(A_0:A_K, B_0:B_K)$$

约束:
$$A_0, B_0, M = \text{Init}(\phi_t)$$

其中:
- $\text{Init}$: 初始化过程
- $\text{Update}$: inner loop update rule
- $\text{Env}$: interactive environment
- $K$: lifetime 的步数

这是 **bilevel optimization** ([Sinha et al. 2018](https://ieeexplore.ieee.org/document/7930004)).

### 5.4 双层优化的挑战

1. **Outer level 数据稀疏**: 一次 life cycle = 一个 data point. 优化 $\phi$ 需要跑百万级 simulated life cycles, 每个又包含百万级 datapoints. 这要求 memory/compute efficiency 极大的突破.

2. **Scalability**: bilevel optimization 在 large architecture 上有 severe scalability issues ([Lorraine et al. 2020](https://arxiv.org/abs/1911.02590); [Metz et al. 2021](https://arxiv.org/abs/2111.05803))

3. **Dynamic system optimization**: 优化目标是 dynamic system (learner + environment), 不是 static loss.

**Proposed solution**: Evolutionary Curriculum - 逐渐增加 environment diversity 和 unpredictability, 让三个 component co-evolve. 这和 [Leike et al. 2017 AI Safety Gridworlds](https://arxiv.org/abs/1711.09883) 的思路呼应.

---

## 6. 我的理解与 intuition

### 6.1 这篇 paper 真正的 contribution

paper 不是 propose 一个具体算法, 而是 **重新定义了 problem space**. 它把当前 AI 的 limitation 重新 frame 成三个明确 roadblock:
1. Paradigm fragmentation (SSL/RL siloed)
2. Learning externalization (MLOps 由人做)
3. Bootstrapping difficulty (chicken-egg-rooster)

这 frame 非常 clean. 我之前讲过 "software 2.0" 时也隐隐触及第 2 点 - 现在的 training pipeline 本质是 software 1.5, 一半人写代码, 一半 gradient descent.

### 6.2 System M 的 deep insight

最 sharp 的 point: System M **不需要学**, 它是 **hardwired evolutionarily**. 这非常反直觉, 我们一直以为 meta-learning 应该是 learned. 但生物证据很 clear:
- 婴儿 face preference 是 hardwired
- 睡眠时 sensory gating 是 hardwired  
- pain interrupting plans 是 hardwired

这些 routing decisions 不需要 lifetime 学, 它们是 evolutionary prior. 真正学的是 A 和 B 的 weights. 这把 meta-learning 问题从 "学 meta-policy" 简化到 "evolutionarily search meta-policy" - 仍然是 search, 但 search space 大大缩小.

### 6.3 和 LeCun 之前工作的关系

这 paper 是 [LeCun 2022 "A Path Towards Autonomous Machine Intelligence"](https://openreview.net/forum?id=BZ5a1r-kVsf) 的延续:
- 那篇 propose 了 H-World model + actor + critic + configurator
- 这篇把 configurator 明确成 System M, 强调 hardwired 性质
- 加入了 Evo/Devo 框架解决 bootstrap 问题
- 加入了 cognitive science 大量证据 (婴儿, 动物)

### 6.4 我对 bilevel optimization 的担忧

paper 第 4.4 节坦承 bilevel optimization 在 large scale 有 severe issues. [Metz et al. 2021](https://arxiv.org/abs/2111.05803) "Gradients are not all you need" 已经指出 meta-gradient 在大网络上有 pathological behavior. 

我的 concern: 要让这工作, 可能需要:
- 大幅简化 architecture (回到 small agents)
- 大量 simulator innovation (既要 realistic 又要 fast)
- 可能需要 evolutionary algorithm (gradient-free) 而非 gradient-based meta-learning

### 6.5 实际 path forward 的 intuition

paper 最后 section 6 提到一些近期进展: test-time training, adaptive retrieval, mixture of world models - 但这些都是 minor variations over rigid system, 不是真正的 autonomous learning.

真正 breakthrough 我觉得会来自几个方向:
1. **V-JEPA 2 路线**: 在 latent space 学 world model, 而不是 pixel generation. 这是 System A 的正确形式.
2. **Robotics 上的 latent action pretraining** (LAWM 路线): 用 unlabeled video 学 action representation, 减少 teleoperation 依赖.
3. **Bilevel optimization 的 hardware/simulator breakthrough**: 如果我们能 cheaply simulate embodied life cycles, Evo/Devo 就变得 feasible.
4. **Meta-control 的简化实现**: 从 SDN-like 路由开始, 用 simple epistemic signals (prediction error) 作为 trigger.

### 6.6 关于 System 1/2 的澄清

Appendix A 有个很好的澄清: Kahneman 的 System 1/2 是 modes of inference (fast vs slow), paper 的 System A/B 是 modes of learning (observation vs action). 这是正交的:

| | System A (observation) | System B (action) |
|---|---|---|
| **System 1 (fast inference)** | predictive coding, statistical learning | policy learning |
| **System 2 (slow inference)** | counterfactual reasoning, causal learning | learning through planning |

很多 confusion 来自这两个 taxonomy 看起来类似. A/B 可以用 System 1 (一次 backprop) 也可以用 System 2 (先 imagination steps); B 可以用 System 1 (learned reflexive policy) 也可以用 System 2 (tree search / chain of thought).

---

## 7. 总结与开放问题

paper 的核心 message: **autonomous learning 不是 nice-to-have, 是 robust AI 的 necessary condition**. 在 dynamic, non-stationary, heavy-tailed 的 real world, 没有 autonomous learning 的 system 注定 stale.

但路径长且难. 几个 immediate research question:
1. System M 的最小可行实现? 需要哪些 hardwired meta-signals?
2. Bilevel optimization 在多大的 agent 上可行? 需要怎样的 simulator?
3. 如何设计 evolutionary curriculum?
4. 如何 evaluate autonomous learning system? Unit test (单独 component) vs integration test (end-to-end)?

paper 提到 "decades away" - 但这是正确的方向. 这篇 paper 像 1986 年的 backpropagation paper - 现在看起来 obvious, 但需要十几年才被 realize.

建议你关注: 
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [LAWM (Tharwat et al. 2025)](https://arxiv.org/abs/2509.18428)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [LeCun 2022 position paper](https://openreview.net/forum?id=BZ5a1r-kVsf)
- [Lake et al. 2017 "Building machines that learn and think like people"](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/building-machines-that-learn-and-think-like-people/959DB1D1A0A4C1A2C8C2E5F4F2A2C5A5)

如果你想 build intuition, 我推荐先研读 V-JEPA 2 + DreamerV3, 这两个 system 已经某种程度上实现了 System A + System B 的初步 integration, 离 full A-B-M 架构还有距离, 但是最好的 starting point.
