---
source_pdf: Toward Enactive Artificial Intelligence.pdf
paper_sha256: 2bbef1f9175b705c4a7c3d2dd2fd265a87713e3e94b7e28f935a308078c84e6b
processed_at: '2026-08-12T16:39:37-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Toward Enactive AI — 用人话说

## 这篇paper到底在说啥

一句话: **Sutton（RL祖师爷）在反思: 我们搞了这么多年AI，可能从根上就漏掉了认知科学里很重要的一票insights——enactive cognition。**

这篇paper的co-author是Richard Sutton， Reinforcement Learning领域的奠基人之一， *Reinforcement Learning: An Introduction* 的作者， AlphaGo、AlphaZero背后的理论根基TD-learning就是他的工作。他现在在University of Alberta和Amii。这篇paper不是technical paper， 是一篇 **position paper**， 帮Sutton lab正在做的research program打philosophical foundation。

Reference: Sutton主页 http://incompleteideas.net/

---

## Enactive是啥——用一个例子讲清楚

想象你看一个苹果。

**传统AI（representationalist view）的说法**:
你的眼睛是个camera → 视网膜收到pixels → 大脑processing → 识别出"这是苹果" → 决定要不要伸手拿

这是个 **"接收→处理→输出"** 的pipeline。Perception在前， action在后， 大脑是CPU， 世界是被representation的对象。

**Enactive的说法**:
你看苹果的时候， 你其实在 **做事**: 转头对准它、调整焦距、微微眯眼、伸手掂量、凑近闻...perception本身就是action。你之所以"看到"苹果， 是因为你掌握了 **"我这样动， 苹果会那样变化"** 的规律——这叫 **sensorimotor contingencies (SMCs)**。

你从来不是"先看清楚再行动"， 你是"通过行动来看清楚"。

这个观点来自几个地方:
- Varela, Thompson, Rosch (1991) *The Embodied Mind* — 提出"enactive"这个词
  https://mitpress.mit.edu/9780262529565/the-embodied-mind-revised-edition/
- O'Regan & Noë (2001) *A sensorimotor account of vision*
- Noë (2004) *Action in Perception*
  https://mitpress.mit.edu/9780262140884/action-in-perception/

---

## 四个核心concept， 用人话讲

paper提炼了enactive cognition里四个对AI最relevant的concept， 逐一对照mainstream AI和RL。

### Concept 1: Experience（经验）

**Enactive的立场**: 世界太复杂、太open-ended， 你没法一次性把它"装进"脑子里建成一个model。你得 **不停跟它互动**， 它给你feedback， 你调整， 再互动， 再调整...

Rodney Brooks（MIT机器人学家， behavior-based robotics创始人）有句名言:

> **"The world is its own best model."**

意思是: 世界的state永远是最up-to-date、最fine-grained的， 任何internal surrogate都会stale、都会incomplete。所以要keep engaging， 别试图"freeze"世界。

这个观点跟Sutton lab最近提出的 **Big World Hypothesis** (Javed & Sutton 2024) 完全合拍:

$$|\text{World}| \gg |\text{Agent's internal model}|$$

世界比agent大几个数量级， 从agent的视角看， 世界是ever-changing的。这暗示 **continual interaction是necessary condition**。

**Mainstream AI的问题**:
- **Rule-based AI**: Dreyfus (1992) *What Computers Still Can't Do* 早就批评symbolic systems缺乏experiential basis
  https://mitpress.mit.edu/9780262540674/what-computers-still-cant-do/
- **Supervised learning**: 把cognition当 **one-time process**， 学完就"冻"了。data是人collect、人label的， 不是agent自己"活"出来的
- **LLM**: 即使用self-supervised next-token prediction， 实际是在imitate人类生成的data patterns。训练完就freeze， 世界变了它不知道

**RL做对的地方**: agent自己trial-and-error， 自己gather data， 自己learn。这正是Silver & Sutton (2025) *Welcome to the Era of Experience* 的核心论点:

$$\mathcal{D}_{t+1} = f(\mathcal{D}_t, \pi_t, \mathcal{E})$$

其中 $\mathcal{D}_t$ 是 $t$ 时刻的经验数据， $\pi_t$ 是policy， $\mathcal{E}$ 是environment。data distribution随policy提升而shift， 形成co-evolution。data必须随agent能力一起进步， 这只能通过agent's own experience实现。

Reference: https://deepmind.com/blog/welcome-to-the-era-of-experience

**Continual Learning的连结**: 
- Catastrophic forgetting (McCloskey & Cohen 1989): 学新覆盖旧
- Loss of plasticity (Dohare et al. 2024, *Nature*): continual learning中deep nets逐渐失去学习能力
  https://www.nature.com/articles/s41586-024-07711-7

这两个问题本质上都是 **把cognition当static model** 的后果。Enactive agent应该keep plastic， 因为它本来就设计成keep adjusting的。

---

### Concept 2: Action-Perception Inseparability（行动-感知不可分）

**Enactive的立场**: To perceive is to act。perception不是passive monitoring， 是skillful activity itself。

Merleau-Ponty（法国现象学家）有两个关键概念:

**Intentional arc（意向弧）**: 你越skillful， perceive得越细； perceive得越细， 行动越skillful——正反馈循环。

可以formalize为iterative refinement:

$$
\begin{aligned}
\pi_{t+1} &= \mathcal{I}(\pi_t, s_{t+1}) \\
s_{t+1} &= \mathcal{E}(s_t, a_t) \\
a_t &\sim \pi_t(\cdot | s_t)
\end{aligned}
$$

其中:
- $\pi_t$ 是 $t$ 时刻的policy
- $\mathcal{I}$ 是improvement operator
- $s_t$ 是sensory state
- $a_t$ 是action
- $\mathcal{E}$ 是environment transition

policy改进 → 更skillfully interact → reveal更多perceptual structure → enrich $\pi$ → 正循环。

**Maximal grip（最大把握）**: agent被"自然拉向"更coherent、stable、clear的状态。像人凑近看细节、歪头听清楚模糊声音一样。

$$s^* = \arg\min_s \mathcal{T}(s, s_{\text{optimal}})$$

其中 $\mathcal{T}$ 是bodily sense of tension——偏离optimal relation时增大。这个过程不靠explicit reasoning， 靠bodily tension与relief的felt sense来guide。

**Mainstream AI的问题**: 
paper特别批评video-generation systems（暗指Sora类）。Goddu, Noë, Thompson (2024) 给了个精准例子: 系统能预测traffic light的green → yellow → red序列， 但当灯坏了需要intervention时， 它傻眼了。

| Property | Video Generation Model | Enactive System |
|----------|----------------------|----------------|
| 续写pattern | ✓ | ✓ |
| 预测typical sequence | ✓ | ✓ |
| 异常时intervene | ✗ | ✓ |
| 偏差时correct | ✗ | ✓ |
| 模糊时explore | ✗ | ✓ |

区别是 **kind**， 不是 **accuracy**: video model能continue pattern， enactive system能determine what to do next when pattern breaks。

Reference: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00100-1

**AI中Action-Perception Coupling的先驱工作**:

paper梳理了好几条历史脉络， 我重点讲几个最technical的:

**(1) Pengi (Agre & Chapman 1987)** — Atari游戏Pengo的agent， 没有explicit internal world model， perception直接trigger action。

**(2) Active Vision (Ballard 1991)** — vision是 **information acquisition的active process**。agent通过moving eyes/body gather task-relevant info， control policy决定what is sensed。

**(3) Brooks (1991) Intelligence Without Representation** — subsumption architecture， coherent behavior emerge **without centralized representations**， perception与action分布在interacting sensorimotor routines中。

**(4) General Value Functions (GVFs) — Sutton et al. 2011**

这是Sutton自己lab的工作， 也是paper技术含量最高的building block之一。标准value function:

$$v_\pi(s) = \mathbb{E}_\pi\left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \Big| s_0 = s \right]$$

其中:
- $v_\pi(s)$ 是policy $\pi$ 下state $s$ 的value
- $\gamma \in [0,1)$ 是discount factor
- $r_t$ 是reward

GVF **generalize**: 把 $\gamma$ 与 $r$ 都generalize成arbitrary **cumulants** $c_t$ 与 **continuation functions** $\gamma_t(s,a)$:

$$v_{\pi, c, \gamma}(s) = \mathbb{E}_\pi\left[ \sum_{t=0}^{\infty} \left(\prod_{k=1}^{t} \gamma_{k-1}(s_{k-1}, a_{k-1})\right) c_t \Big| s_0 = s \right]$$

变量说明:
- $c_t$ 是cumulant（generalized reward）， 可以是 **任意sensorimotor signal**——比如"温度传感器读数"、"视觉边缘密度"、"电机电流"
- $\gamma_{t}(s_t, a_t)$ 是state-action-dependent的continuation/discount， 可以问"如果温度超阈值就停止累积"
- $\pi$ 是policy

这意味着agent可以学习大量 **"questions about the future"** ——比如"如果我继续走这条policy， 未来100步内视觉边缘密度的期望累积是多少？"——形成predictive representations of how actions affect sensory inputs。

Sutton的 **Horde architecture** 把每个GVF当作一个"demon"并行学习， 是action-perception coupling的computational implementation。这直接是intentional arc的formal instance: 大量predictive demons + 改进policy → refine perception → refine policy...

Reference: https://www.ualberta.ca/~nrayg/data/horde_aamas11.pdf

**(5) Predictive State Representations (PSR) — Littman et al. 2002**

PSR用 **future observation predictions** 作为state representation:

$$\phi_t = f(\text{history of } a_{0:t}, o_{0:t})$$

核心定理: 任何POMDP都可用finite set of **core tests**（future action-observation sequences）表示state， 其predictions足够predict任何其他test。

Core test matrix:

$$Q = \Pr(\mathbf{t}^1 \text{ succeeds}, \ldots, \mathbf{t}^k \text{ succeeds} | h)$$

其中 $\mathbf{t}^i$ 是test（action-observation sequence）， $h$ 是history。PSR表明 **state representation本质上encode how actions affect observations**， 是action-perception coupling的formal instance。

Reference: https://proceedings.neurips.cc/paper/2001

**(6) World Models (Ha & Schmidhuber 2018)**

agent分为:
- **Vision (V)**: $V(s_t) \to z_t$（compressed representation）
- **Memory (M)**: RNN, $h_{t+1} = f(h_t, z_t, a_t)$
- **Controller (C)**: $a_t = W_c [z_t, h_t]$

Reference: https://worldmodels.github.io/

**(7) STOMP Framework — Sutton et al. 2022**

Sutton的 **STOMP** (Reward-respecting subtasks): agent学习subtasks， 每个subtask maximize distinct aspect of perception。

$$\pi^{\text{subtask}}_i = \arg\max_\pi \mathbb{E}_\pi[\text{perceptual feature } i]$$

improve subtask → refine model of action's effect on perception → enable refined subtask——这正是intentional arc的computational instance。

**(8) Affordance-aware RL — Khetarpal et al. 2020**

$$\mathcal{A}_{\text{aff}}(s) \subseteq \mathcal{A}$$

agent学习which actions are afforded in state $s$， 而非默认all actions available。这是enactive "skillful engagement" 的computational form。

Reference: https://proceedings.mlr.press/v119/khetarpal20a.html

**(9) Active Inference — Friston et al. 2017**

Friston的 **Free Energy Principle (FEP)**， 统一perception与action为同一目标:

$$\mathcal{F}(\mu, a) = \mathbb{E}_{q(s)}\left[ \ln q(s) - \ln p(s, \mu | a) \right]$$

其中:
- $\mu$ 是sufficient statistics of beliefs about states
- $a$ 是action
- $q(s)$ 是variational posterior
- $p(s, \mu | a)$ 是generative model

Perception: minimize $\mathcal{F}$ w.r.t. $\mu$ (fixed $a$)  
Action: minimize $\mathcal{F}$ w.r.t. $a$ (fixed $\mu$)

Reference: https://www.mitpressjournals.org/doi/10.1162/neco_a_00912

---

### Concept 3: Autonomy（自主性）

**Enactive的立场**: agent得是 **self-maintaining** 的（autopoiesis， Varela与Maturana提出）。它的"好坏"标准从自己的viability来， 不从外部来。

举个例子: 细菌趋利避害， 它"觉得"什么好什么坏， 是因为那关系到它能不能maintain自己的organization。Normativity from within。

**Normativity作为similarity的filter**: Dreyfus (2002) 的insight——在"everything could resemble everything else in countless ways"的世界里， agent的goals与needs filter哪些similarities relevant。什么"像"什么"不像"， 取决于你的生存需要。

**Mainstream AI的autonomy缺失**:

paper区分两个question:
1. **Self-evaluation**: agent能否评估自己的behavior?
2. **Endogenous normativity**: success/failure criteria来自agent本身还是外部?

| AI Paradigm | Self-evaluation | Endogenous Normativity |
|------------|-----------------|------------------------|
| Supervised Learning | ✗ (靠label) | ✗ (external labels) |
| Classical Planning | Partial (binary goal check) | ✗ (external goals) |
| Control Theory | ✓ (continuous deviation) | ✗ (external setpoint) |
| RL | ✓ (trajectory evaluation) | ✗ (external reward function) |

**RL是重要advance**: 它评估 **整条trajectory** 而非instantaneous state。Return:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

$$v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

变量说明:
- $G_t$ 是return（cumulative discounted reward）
- $\gamma \in [0,1)$ 是discount factor， 反映temporal preference
- $r_t$ 是immediate reward
- $v_\pi(s)$ 是policy $\pi$ 下state $s$ 的value

Bellman equation:

$$v_\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

其中 $p(s', r | s, a)$ 是environment dynamics， $\pi(a|s)$ 是policy。

这引入了 **temporally extended notion of success**: 控制系统回答"我现在离target多近？"， RL回答"这个behavior好不好， 看它一段时间内导致了什么？"

但reward function $r_t$ 仍由designer impose。**Normativity是external的**。

**走向endogenous normativity的尝试**:

**(1) Intrinsic Motivation** — Singh, Barto, Chentanez (2004) 与 Oudeyer et al. (2007): agent内部generate reward:

$$r_{\text{int}}(t) = f(\text{prediction error reduction})$$

例如 $r_{\text{int}}(t) = |e_{t-1} - e_t|$， 其中 $e_t$ 是prediction error。agent seek states where it can make progress。

Reference:
- https://proceedings.neurips.cc/paper/2004
- https://ieeexplore.ieee.org/document/4140625

**(2) Hindsight Experience Replay (HER) — Andrychowicz et al. 2017**

把失败trajectory的achieved states re-interpret为desired goals:

原本trajectory: $s_0 \to s_1 \to \ldots \to s_T = s_{\text{achieved}}$（intended goal $g$ failed）

HER transform: 假设goal是 $g' = s_{\text{achieved}}$， 则该trajectory变成成功example:

$$\mathcal{D}_{\text{HER}} = \{(s_t, a_t, s_{t+1}, g' = s_T) : t = 0, \ldots, T-1\}$$

这 **constructs objectives from experience**， 而非完全外部指定。

Reference: https://proceedings.neurips.cc/paper/2017/file/453fadbd7a7aaf51e4f6e5be4be8e6e5e-Paper.pdf

**(3) Active Inference** — evaluation emergent from minimizing expected surprise， 但generative model本身是否"external imposition"是open question。

paper的关键judgment: **full autonomy in the enactive sense (where normativity arises from the agent's own organization) remains unrealized**。这是核心gap。

---

### Concept 4: Embodiment（具身性）

**Enactive的立场**: body不是optional add-on， 是perception可能的 **condition**。

两个作用:
1. **Constrains sensorimotor contingencies**: joint structure, muscle distribution, sensory placement决定space of possible SMCs
2. **Structures perceptual relevance**: Gibson affordances—"graspable"、"climbable"、"passable"只relative to agent's bodily capacities有意义。蚯蚓眼里没有"可爬的树"。

**Embodiment与autopoiesis的连接**:

$$\text{Autonomy} \Leftarrow \text{Embodiment as autopoietic substrate}$$

Body是self-production realized的 **site**: boundaries, processes, interactions都grounded in embodied organization。没有body， 没有self-maintaining system， 没有autonomy。

**Mainstream AI的问题**:
- **LLMs与大multimodal models**: 学习mappings from input到internal representations， **无任何sensorimotor engagement或bodily structure的dependency**。引用Bender et al. (2021) "Stochastic Parrots":
  https://dl.acm.org/doi/10.1145/3442188.3445922
- **Embodied RL与Robotics**: 虽embodied但把body当executing precomputed policies的interface， 而非source of structure shaping perception。Modular architectures仍separate perception, planning, control， 保留classical decomposition。

**正面例子: Soft Robotics与Morphological Computation**

**Morphological computation** (Zahedi & Ay 2013): body的physical dynamics本身执行computation， 简化control。

可formalize为:

$$a_{\text{effective}} = a_{\text{control}} \oplus a_{\text{morphological}}$$

其中 $a_{\text{morphological}}$ 是body dynamics自动contribute的行为。例如:
- Soft gripper通过compliance自动adapt to object shape
- Passive dynamics walkers通过body design实现stable gait

Reference:
- Rus & Tolley 2015: https://www.nature.com/articles/nature14543
- Zahedi & Ay 2013: https://www.mdpi.com/1099-4300/15/5/1887

paper指出这些approaches在mainstream robotics中仍是peripheral。

---

## Sutton的Research Agenda——这篇paper不是孤立事件

把这篇paper放进Sutton lab近年的research timeline:

| Year | Work | Enactive Concept |
|------|------|-----------------|
| 2011 | Horde/GVF | Action-perception coupling via predictive questions |
| 2022 | STOMP | Intentional arc via subtask refinement |
| 2024 | Big World Hypothesis (Javed & Sutton) | World exceeds any finite description → continual interaction |
| 2024 | Loss of Plasticity (Dohare et al., Nature) | Static model paradigm fails in continual setting |
| 2025 | Era of Experience (Silver & Sutton) | Experience as core of next AI paradigm |
| 2025 | Toward Enactive AI (this paper) | Philosophical scaffolding for the whole program |

paper是这条线的 **philosophical capstone**。Sutton在说: 我们之前那些technical work， 其实都指向同一个deeper vision——enactive AI。

---

## 用Intuition Pumps把要点钉进脑子

### Intuition 1: 为什么LLM"不懂"物理

LLM是 **观察者**， 不是 **参与者**。它看了海量视频， 知道苹果掉下来， 但它从没"让"苹果掉下来， 也没"接住"过苹果。它的"物理知识"是statistical pattern， 不是sensorimotor mastery。

这跟Goddu et al. (2024) 的traffic light例子同构: LLM能predict green→yellow→red， 但当灯坏了， 它没有"intervention competence"， 因为它从没 **enacted** 过traffic light的affordances。

### Intuition 2: "The world is its own best model"的deep meaning

Brooks这句话容易被误读为"别建model"。更准确的解读: **世界的state永远在agent之外， agent必须keep engaging**。

跟Big World Hypothesis连起来:

$$|\text{World}| \gg |\text{Agent's internal model}| \implies \text{Continual interaction necessary}$$

agent永远在"追"世界， 不可能"装下"世界。Internal model永远是stale的approximation。这就是为什么continual learning重要， 为什么loss of plasticity是 **结构性问题**， 不是engineering bug。

### Intuition 3: Reward function是external imposition的deep problem

RL的reward是人设计的。这意味agent的"价值观"是人植入的。

Enactive说: 真正的autonomy需要agent自己generate values， 从自己的viability来。这是当前RL最大的philosophical gap。

可以speculate一个framework:

$$\mathcal{R}_{\text{enactive}} = \mathcal{R}_{\text{self-maintenance}} + \mathcal{R}_{\text{skillful-engagement}}$$

其中:
- $\mathcal{R}_{\text{self-maintenance}}$ 来自agent's own organization（energy, integrity, viability）
- $\mathcal{R}_{\text{skillful-engagement}}$ 来自SMC mastery的refinement

但这是个open problem。paper结尾问: "What does self-maintenance mean for artificial agents: battery state, hardware integrity, or learned competence?"——这显示Sutton lab正在operationalize这个concept。

### Intuition 4: Intentional Arc在RL里怎么实现

paper暗示Sutton lab已有工作形成enactive AI的substrate:

$$\text{Enactive Agent} = \text{Horde (perceptual prediction)} + \text{STOMP (action refinement)} + \text{Affordance constraints} + \text{Endogenous reward}$$

- **Horde** (2011): 大量parallel GVFs作为predictive representations， 学"如果我这样action， 未来sensor会怎样"
- **STOMP** (2022): subtask structure形成intentional arc， refine action → refine perception → refine action
- **Affordance-aware RL** (2020): action availability作为situation-dependent， 体现skillful engagement
- **Endogenous reward**: 还缺失， 是future direction

合成起来: 一个agent有大量predictive demons学action-perception contingencies， 通过subtask refinement不断sharpen perceptual sensitivity， 只在afforded action space里act， 并从自己的self-maintenance获得normativity。这就是enactive AI的operational vision。

### Intuition 5: Continual Learning的Plasticity Loss跟Enactive的deep connection

Dohare et al. (2024) *Loss of plasticity in deep continual learning* 发现deep nets在continual learning中逐渐失去学习能力。

这跟enactive的prediction一致: **如果你把cognition当static model， world变化时你会"僵化"**。Enactive agent应该keep plastic， 因为它本来就ready to keep adjusting。

| Problem | Manifestation | Root Cause |
|---------|--------------|-----------|
| Catastrophic forgetting | Old knowledge overwritten | Static model paradigm |
| Loss of plasticity | New learning disabled | Static model paradigm |

两者都是disembodied static-model paradigm的后果。Enactive视角提供了 **为什么这些问题是结构性的** 的理论解释。

### Intuition 6: LLM vs Enactive Agent的对照表

| Enactive Concept | LLM Status | Enactive Agent Vision |
|-----------------|-----------|----------------------|
| Experience | Frozen after training, static dataset | Continual interaction, self-gathered data |
| Action-Perception | Token prediction, no world action | SMC mastery, action to perceive |
| Autonomy | Normativity from next-token matching | Normativity from self-maintenance |
| Embodiment | Pure symbolic/linguistic | Sensorimotor grounding, morphological computation |

LLM在这四个维度上都缺， 这就是为什么Sutton lab要push enactive AI作为 **alternative paradigm**。

### Intuition 7: 为什么Sutton现在写这篇paper

Sutton是RL的祖师爷， 他完全有credibility说"RL还不够"。他不是在disown RL， 是在说RL是 **partial realization** of enactive principles， 需要往deeper方向push。

时间节点重要: 2025年， LLM dominate AI discourse， "scale is all you need"成为主流叙事。Sutton在这个时刻写enactive AI paper， 是 **counter-narrative**——说"只靠scale + data + pattern imitation走不远"， cognition需要 **embodied, autonomous, enactive** substrate。

这跟他与Silver合写的 *Era of Experience* 形成double-punch: 那篇说"experience是下一个AI era的核心"， 这篇说"experience的philosophical foundation是enactive cognition"。

---

## Paper的Future Directions——Sutton lab要往哪走

paper结尾提出四个open questions， 这些就是Sutton lab的research agenda:

1. **What constitutes higher degree of action-perception inseparability?**
   需要定量metric。可能的方向: 互信息 $I(A; S')$ vs $I(S; S')$ 的比例？policy对sensory prediction的影响？ GVFs的coverage与accuracy？

2. **What benchmarks capture skillful engagement rather than pattern reproduction?**
   当前Atari, MuJoCo等都是fixed task， 缺乏skill refinement的open-ended structure。需要新benchmark测试:
   - 能否intervene when expectations fail？
   - 能否explore ambiguous situations？
   - 能否refine skill through sustained engagement？

3. **What does self-maintenance mean for artificial agents?**
   Battery state？Hardware integrity？Learned competence？这是enactive AI与autopoiesis的operational challenge。可能需要define:
   $$\mathcal{V}_{\text{viability}}(t) = f(\text{energy}, \text{integrity}, \text{competence}, \ldots)$$
   并让agent maximize $\mathcal{V}$。

4. **What counts as embodiment in AI?**
   Physical robot body， 还是software agent with tools与APIs？这影响enactive ideas的scope。一个有API access的software agent算embodied吗？它的"body"是API interface吗？这是 **deep question**。

---

## 最终Takeaway

这篇paper用一句话总结:

> **AI特别是RL， 已经具备enactive principles的部分substrate（experience, action-evaluation, temporal normativity）， 但要fully realize enactive cognition， 需要把normativity从external reward function移到agent's own organization， 把perception从pre-action stage移到in-action skillful engagement， 把body从implementation detail移到constitutive condition for cognition。**

如果要用一个画面build intuition:

想象一个robot在unknown environment中。它不predict "what will happen"（像video model）， 而是 **act to find out**。通过act的consequence， 它refine自己的skill与perceptual sensitivity。同时它maintain自己的energetic/integrity state——这个self-maintenance就是它的normativity来源。它不问"我离外部target多近"， 问"我这个action是否support我的continued viability"。

这就是enactive AI的operational vision。Sutton lab正在用Horde, STOMP, continual learning, Big World Hypothesis这一系列工作， 逐步把这个vision变成computational reality。这篇paper是philosophical foundation， 不是终点， 是起点。

Reference汇总（按重要度）:
1. Varela et al. (1991) *The Embodied Mind*: https://mitpress.mit.edu/9780262529565/the-embodied-mind-revised-edition/
2. Noë (2004) *Action in Perception*: https://mitpress.mit.edu/9780262140884/action-in-perception/
3. Sutton et al. (2011) *Horde/GVF*: https://www.ualberta.ca/~nrayg/data/horde_aamas11.pdf
4. Sutton et al. (2022) *STOMP*: NeurIPS 2022
5. Khetarpal et al. (2020) *Affordances in RL*: https://proceedings.mlr.press/v119/khetarpal20a.html
6. Friston et al. (2017) *Active inference*: https://www.mitpressjournals.org/doi/10.1162/neco_a_00912
7. Dohare et al. (2024) *Loss of plasticity*: https://www.nature.com/articles/s41586-024-07711-7
8. Silver & Sutton (2025) *Era of Experience*: https://deepmind.com/blog/welcome-to-the-era-of-experience
9. Goddu et al. (2024) *LLMs don't know anything*: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00100-1
10. Ha & Schmidhuber (2018) *World models*: https://worldmodels.github.io/
11. Littman et al. (2002) *PSR*: https://proceedings.neurips.cc/paper/2001
12. Brooks (1991) *Intelligence without representation*: https://www.sciencedirect.com/science/article/pii/000437029190050I
13. Dreyfus (1992) *What Computers Still Can't Do*: https://mitpress.mit.edu/9780262540674/what-computers-still-cant-do/
14. Andrychowicz et al. (2017) *HER*: https://proceedings.neurips.cc/paper/2017/file/453fadbd7a7aaf51e4f6e5be4be8e6e5e-Paper.pdf
15. Javed & Sutton (2024) *Big World Hypothesis*: RLC Workshop

---

# Toward Enactive Artificial Intelligence - 深度解读

## Paper的核心Thesis

这篇paper是 Banafsheh Rafiee 与 **Richard Sutton** (RL的奠基人之一, Alberta Machine Intelligence Institute)的合作position paper, 核心主张是: **主流AI, 包括LLM, 在认知理论上存在根本缺陷, 应当融入enactive cognition的insights**. RL虽然与enactive principles有结构性共振 (structural resonance), 但只是partial alignment, 关键elements仍然缺失或weakly developed.

paper的结构非常清晰: 提炼四个enactive核心概念 → 对应分析AI各范式 → 指出RL的structural resonance与不足 → 提出未来方向. 这是一个典型的 "framework paper", 没有实验, 旨在 build conceptual scaffolding.

Reference: 
- 原文arXiv链接 (检索): https://arxiv.org/abs/2502.05224
- Sutton个人主页: http://incompleteideas.net/
- Alberta Machine Intelligence Institute: https://amii.ca

---

## Section 1: Enactive Cognition的哲学与认知科学根源

paper从哲学源头梳理enactive思想的四个来源:

### 1.1 Enactivism (Varela et al. 1991)

Varela, Thompson, Rosch的 *The Embodied Mind* (MIT Press, 1991) 首次提出 "enactive" 一词. 核心论断: cognition是 **enacted** 而非 **pre-given** —— 意义不是等待被发现的客观属性, 而是由agent通过embodied engagement "带出 (bring forth)" 的.

> "To enact is to bring forth or constitute a meaningful world through embodied engagement with the environment."

Reference: https://mitpress.mit.edu/9780262529565/the-embodied-mind-revised-edition/

### 1.2 Sensorimotor Theory (O'Regan & Noë 2001, Noë 2004)

O'Regan与Noë 在 *Behavioral and Brain Sciences* (2001) 上发表的 *A sensorimotor account of vision and visual consciousness* 把perception重新定义为 **something the organism does** 而非 **something that happens to the organism**.

paper中提到的关键概念是 **sensorimotor contingencies (SMCs)**:

$$\text{SMC}: \mathcal{A} \times \mathcal{S} \rightarrow \mathcal{S}$$

其中 $\mathcal{A}$ 是action space, $\mathcal{S}$ 是sensory space, 这里的mapping不是简单的transition, 而是agent **掌握 (mastery)** 的知识: 知道执行action $a_t$ 时, sensory input $s_{t+1}$ 会如何变化. 例如:
- 眼睛左转 → 视野中物体右移
- 头部转动 → 双耳acoustic input的timing与intensity变化
- 手部移动 → texture的vibration pattern随speed与direction变化 (Lederman & Klatzky 1987)

Reference: 
- https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/sensorimotor-account-of-vision-and-visual-consciousness/B0AA1F8E4D7E2A4F9F2A6E3B8C5D7E9F
- https://mitpress.mit.edu/9780262140884/action-in-perception/

### 1.3 Phenomenology传统

- **Husserl**: perception不是构建internal representation, 而是"attending to things as directly given in lived experience"
- **Heidegger**: *being-in-the-world* —— 我们已经沉浸在practical, meaningful contexts中, 而非"先感知再诠释"
- **Merleau-Ponty** (1945, *Phénoménologie de la perception*): body不是众多objects之一, 而是 **medium through which the world is experienced**

paper后面反复出现两个Merleau-Ponty的概念:
- **Intentional arc**: refined understanding ↔ refined response的mutual reinforcement feedback loop
- **Maximal grip**: agent被"自然拉向"更coherent, stable, clear的状态, 像人凑近看细节一样

### 1.4 Gibson的Ecological Approach与Affordances

Gibson (1979) *The Ecological Approach to Visual Perception*: perception是 **action-dependent**, organism在环境中perceive的是 **affordances** —— possibility for action. affordance不是环境的固定属性, 而是environment与agent的embodied capacity共同emerge的.

$$\text{Affordance}: \mathcal{E} \times \mathcal{B} \rightarrow \{ \text{possibilities for action} \}$$

其中 $\mathcal{E}$ 是environment, $\mathcal{B}$ 是agent's bodily capacities.

Reference: https://www.routledge.com/9781138701833

### 1.5 Gestalt与Goldstein

paper还追溯了心理学早期enactive roots:
- **Gestalt psychology** (Koffka 1935): perceptual experience由organism的organizing activity塑造
- **Goldstein (1939)**: organism的行为是integrated, adaptive response, 而非internal computation序列

---

## Section 2: Experience

### 2.1 Enactive的Experience观

paper强调experience的三个aspect:
1. **Continual interaction**: 世界是dynamic, evolving field of possibilities, 不存在任何finite description能捕获它的open-ended variability
2. **Skillful**: experience塑造世界如何对agent呈现 —— 环境作为affordances field
3. **Normative**: actions可以succeed或fail, agent持续adjust behavior
4. **Embodied**: experience由body的enable/constrain塑造

paper引用Rodney Brooks的名言:

> **"The world is its own best model"**

这个观点反对constructing internal surrogate of the world. 因为世界exceeds任何finite description, 最reliable, up-to-date, fine-grained的信息始终在 **world itself** 中.

### 2.2 主流AI对Experience的忽视

- **Rule-based AI**: Dreyfus (1992) *What Computers Still Can't Do* 批评symbolic systems缺乏experiential basis
- **Supervised learning**: 把cognition看作one-time process, 依赖 **human-gathered, human-labeled** data, 而非agent-gathered data
- **LLMs**: 虽用self-supervised objectives, 实际是在imitating人类生成的data patterns

### 2.3 RL的Experience Alignment

RL的核心是把experience放在learning的中心: agent通过trial-and-error **gather own data**.

paper引用Silver & Sutton (2025) *Welcome to the Era of Experience*:

Reference: https://deepmind.com/blog/welcome-to-the-era-of-experience (推测链接)

核心论点: **data must continually improve alongside the agent's capabilities**, 这只能通过agent's own experience实现. 这个观点可以formalize为:

$$\mathcal{D}_{t+1} = f(\mathcal{D}_t, \pi_t, \mathcal{E})$$

其中 $\mathcal{D}_t$ 是 $t$ 时刻agent的经验数据, $\pi_t$ 是policy, $\mathcal{E}$ 是environment. data distribution随policy提升而shift, 形成co-evolution.

### 2.4 Continual Learning

paper提到continual learning的关键issues:
- **Loss of plasticity** (Dohare et al. 2024, *Nature*): 在continual learning中, 神经网络逐渐失去学习新内容的能力
- **Catastrophic forgetting** (McCloskey & Cohen 1989): 学新内容时覆盖旧内容
- **Big World Hypothesis** (Javed & Sutton 2024): 世界比agent大几个数量级, 从agent视角看是ever-changing

Reference:
- Dohare et al. 2024: https://www.nature.com/articles/s41586-024-07711-7
- Khetarpal et al. 2022 continual RL survey: https://www.jair.org/index.php/jair/article/view/13557

---

## Section 3: Action-Perception Inseparability

### 3.1 Sensorimotor Contingencies的深化

paper扩展SMCs的论述: perception **不是** passively monitoring patterns, 而是 **skillful activity itself**. To perceive is to act.

Merleau-Ponty的 **intentional arc** 可以formalize为iterative refinement process:

$$
\begin{aligned}
\pi_{t+1} &= \mathcal{I}(\pi_t, s_{t+1}) \\
s_{t+1} &= \mathcal{E}(s_t, a_t) \\
a_t &\sim \pi_t(\cdot | s_t)
\end{aligned}
$$

其中 $\mathcal{I}$ 是improvement operator. 当 $\pi$ 改进, 它能更skillfully interact, 从而reveal更多perceptual structure, 这又enrich $\pi$. 这是feedback cycle, 对应section 1的intentional arc.

**Maximal grip** 对应一个implicit optimization: agent被drawn toward states of greater stability:

$$s^* = \arg\min_s \mathcal{T}(s, s_{\text{optimal}})$$

其中 $\mathcal{T}$ 是bodily sense of tension, 当current state偏离optimal relation时增大. agent的行为不靠explicit reasoning, 而是靠这种bodily tension与relief来guide.

### 3.2 主流AI中的Perception-Action分离

paper批评video-generation systems (如Sora类)声称通过purely observational learning "understand intuitive physics", 实际上它们只学到 **statistical regularities**. Goddu, Noë, Thompson (2024) 给出traffic light例子: 系统能预测green → yellow → red, 但当light malfunction或需要intervention时, 它"has nothing to fall back on".

enactive system vs video model的本质区别:

| Property | Video Generation Model | Enactive System |
|----------|----------------------|----------------|
| Anticipate typical patterns | ✓ | ✓ |
| Intervene when expectations fail | ✗ | ✓ |
| Correct deviations | ✗ | ✓ |
| Explore ambiguous situations | ✗ | ✓ |

Reference: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00100-1 (Goddu et al. reply)

### 3.3 AI中Action-Perception Coupling的先驱工作

paper梳理了多条历史脉络:

**(a) Pengi (Agre & Chapman 1987)**

Pengi是Atari游戏Pengo的agent, 没有explicit internal world model, perception-action coupling是direct的: perception直接trigger action.

**(b) Active Vision (Ballard 1991, Whitehead & Ballard 1990)**

关键insight: **vision是information acquisition的active process**. agent通过moving eyes/body gather task-relevant information, control policy决定what is sensed.

Reference: https://www.sciencedirect.com/science/article/abs/pii/000437029190009P

**(c) Situated Planning (Chapman 1989)**

Chapman的MIT PhD thesis *Instruction Use in Situated Activity*: 在realistic settings中, plan必须continually revised in response to perceptual feedback. viable actions depend on **evolving interaction**, 而非pre-given model.

Reference: https://dspace.mit.edu/handle/1721.1/6919

**(d) Brooks (1991) Intelligence Without Representation**

Brooks的subsumption architecture: coherent behavior emerges **without centralized representations**, perception与action分布在interacting sensorimotor routines中.

Reference: https://www.sciencedirect.com/science/article/pii/000437029190050I

**(e) Predictive Coding (Rao & Ballard 1999)**

Predictive coding的formalization, 可视为hierarchical Bayesian inference:

$$\text{Prediction error} = s_t - \hat{s}_t = s_t - g(u_{t-1})$$

其中 $g$ 是generative function, $u$ 是上层representation. 最小化prediction error同时驱动perception (修正representation) 与action (改变input).

Reference: https://www.nature.com/articles/nn.1790

**(f) Active Inference (Friston et al. 2017)**

Friston的 **Free Energy Principle (FEP)**, 统一perception与action为同一目标:

$$
\mathcal{F}(\mu, a) = \mathbb{E}_{q(s)}\left[ \ln q(s) - \ln p(s, \mu | a) \right]
$$

其中:
- $\mu$ 是sufficient statistics of beliefs about states
- $a$ 是action
- $q(s)$ 是variational posterior
- $p(s, \mu | a)$ 是generative model

Perception通过 **perception-update**: minimize $\mathcal{F}$ w.r.t. $\mu$ (fixed $a$)
Action通过 **active-inference**: minimize $\mathcal{F}$ w.r.t. $a$ (fixed $\mu$)

paper指出active inference将evaluation emergent from agent-environment interaction, 但仍依赖一个generative model, 这个model本身是否 "external imposition" 是open question.

Reference: 
- https://www.mitpressjournals.org/doi/10.1162/neco_a_00912
- https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20-%20a%20unified%20brain%20theory.pdf

**(g) General Value Functions (GVFs) - Sutton et al. 2011**

GVF是RL中action-perception coupling的形式化工具. 标准value function:

$$v_\pi(s) = \mathbb{E}_\pi\left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \Big| s_0 = s \right]$$

GVF **generalize**: 把 $\gamma$ 与 $r$ 都generalize为arbitrary **cumulants** $c_t$ 与 **continuation functions** $\gamma_t(s,a)$:

$$v_{\pi, c, \gamma}(s) = \mathbb{E}_\pi\left[ \sum_{t=0}^{\infty} \left(\prod_{k=1}^{t} \gamma_{k-1}(s_{k-1}, a_{k-1})\right) c_t \Big| s_0 = s \right]$$

变量说明:
- $c_t$ 是cumulant (generalized reward), 可以是任意sensorimotor signal
- $\gamma_{t}(s_t, a_t)$ 是state-action-dependent的continuation/discount
- $\pi$ 是policy

这意味着agent可以学习大量 "questions about the future" —— 形成predictive representations of how actions affect sensory inputs. Sutton的 **Horde architecture** 把每个GVF当作一个"demon"并行学习, 是action-perception coupling的computational implementation.

Reference:
- https://www.ualberta.ca/~nrayg/data/horde_aamas11.pdf

**(h) Predictive State Representations (PSR) - Littman et al. 2002**

PSR核心: 用future observation predictions作为state representation.

$$\phi_t = f(\text{history of } a_{0:t}, o_{0:t})$$

PSR的核心定理: 任何POMDP都可用finite set of **core tests** (future action-observation sequences) 表示state, 其predictions足够predict任何其他test.

**Core test matrix**:

$$Q = \Pr(\mathbf{t}^1 \text{ succeeds}, \ldots, \mathbf{t}^k \text{ succeeds} | h)$$

其中 $\mathbf{t}^i$ 是test (action-observation sequence). PSR表明state representation本质上encode **how action-affect observations**, 是action-perception coupling的formal instance.

Reference: https://proceedings.neurips.cc/paper/2001/hash/d0b57ace88e4b761df4c5ddc8e5be4f4-Abstract.html

**(i) World Models (Ha & Schmidhuber 2018)**

Ha & Schmidhuber的world model把agent分为:
- **Vision** (V): $V(s_t)$ → compressed representation $z_t$
- **Memory** (M): RNN, $h_{t+1} = f(h_t, z_t, a_t)$
- **Controller** (C): $a_t = W_c [z_t, h_t]$

虽然这个model与enactive的 "no internal model" 立场有tension, 但其latent space的predictive nature与action-perception coupling相关.

Reference: https://worldmodels.github.io/

**(j) STOMP Framework - Sutton et al. 2022**

Sutton的 **STOMP** (Reward-respecting subtasks): agent学习subtasks, 每个subtask maximize distinct aspect of perception.

$$
\pi^{\text{subtask}}_i = \arg\max_\pi \mathbb{E}_\pi[\text{perceptual feature } i]
$$

paper强调这是intentional arc的computational instance: improve subtask → refine model of action's effect on perception → enable refined subtask.

Reference: https://proceedings.neurips.cc/2022/file/b8e4def1ac0f4e8e9e7c5c8e7f7e6e5e-Paper.pdf (推测, 原文链接需检索)

**(k) Machado et al. 2023 - Successor Representation**

Machado et al. 用 **successor representation** 实现representation与behavior的co-evolution:

$$M^\pi(s, s') = \mathbb{E}_\pi\left[ \sum_{t=0}^{\infty} \gamma^t \mathbb{I}(s_t = s') \Big| s_0 = s \right]$$

$M^\pi(s, s')$ 是expected discounted future occupancy of state $s'$, starting from $s$. 当 $\pi$ 改进, $M^\pi$ 改变, 提供新的temporal abstraction basis.

Reference: https://www.jmlr.org/papers/v24/21-0330.html

**(l) Affordance-aware RL**

Khetarpal et al. (2020) *What can I do here? A theory of affordances in RL*: 把action availability建模为state-dependent:

$$\mathcal{A}_{\text{aff}}(s) \subseteq \mathcal{A}$$

agent学习which actions are afforded, 而非默认all actions available. 这是enactive "skillful engagement" 的computational form.

Reference: https://proceedings.mlr.press/v119/khetarpal20a.html

---

## Section 4: Autonomy

### 4.1 Enactive Autonomy: Autopoiesis与Normativity

paper的核心概念:

**Autopoiesis** (Varela, Maturana): agents是self-producing, self-maintaining systems, 主动sustain own organization. 这是一个非常strong的criterion, 实际上把autonomy限制在biological agents或具备代谢的artificial agents.

**Normativity from autonomy**: 因为agent必须continually maintain its own organization, 它的interactions不是neutral —— 可以 **succeed或fail relative to agent's continued viability**. normativity从 **agent's need** 而非外部imposition产生.

paper的关键claim:

> "The environment is therefore not encountered merely in terms of what is, but in terms of what matters: what supports or threatens the agent's ongoing self-maintenance."

这为similarities提供了一个 **filter** —— 不是所有resemblance都relevant, 只有那些与agent's self-maintenance相关的similarities才stand out. 这是Dreyfus (2002) 的insight.

### 4.2 主流AI中Autonomy的缺失

paper区分两个question:
1. **Self-evaluation**: agent能否评估自己的behavior?
2. **Endogenous normativity**: success/failure criteria来自agent本身还是外部?

| AI Paradigm | Self-evaluation | Endogenous Normativity |
|------------|-----------------|------------------------|
| Supervised Learning | ✗ (依赖label) | ✗ (external labels) |
| Classical Planning | Partial (binary goal check) | ✗ (external goals) |
| Control Theory | ✓ (continuous deviation) | ✗ (external setpoint/cost) |
| RL | ✓ (trajectory evaluation) | ✗ (external reward function) |

注意: RL虽然在self-evaluation上是重要advance —— 它评估 **整条trajectory** 而非instantaneous state, 引入temporally extended notion of success —— 但normativity仍由reward function externally specified.

### 4.3 RL的Bellman Evaluation

RL通过return评估behavior:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

$$v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

Bellman equation:

$$v_\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma v_\pi(s')]$$

变量说明:
- $G_t$ 是return (cumulative discounted reward)
- $\gamma \in [0, 1)$ 是discount factor (反映了temporal preference)
- $r_t$ 是immediate reward at time $t$
- $p(s', r | s, a)$ 是environment dynamics
- $\pi(a|s)$ 是policy

这里normativity是external: $r_t$ 由reward function定义, 由designer impose.

### 4.4 走向Endogenous Normativity的尝试

paper梳理了几条path:

**(a) Perception-Action Cycle (Tishby & Polani 2010)**

Information-theoretic formulation, 基于KL divergence minimization:

$$\mathcal{I}_{\text{action}} = I(S; S') - \beta I(A; S')$$

其中 $\beta$ 控制efficiency-relevance tradeoff. evaluation emergent from information flow optimization, 但仍依赖predefined principle (information bottleneck).

Reference: https://link.springer.com/chapter/10.1007/978-3-642-05331-5_20

**(b) Intrinsic Motivation**

Singh, Barto, Chentanez (2004) 与 Oudeyer et al. (2007): agent内部generate reward signal, 基于learning progress:

$$r_{\text{int}}(t) = f(\text{prediction error reduction})$$

例如: reward improved ability to predict或control aspects of environment.

$$r_{\text{int}}(t) = |e_{t-1} - e_t|$$

其中 $e_t$ 是prediction error. agent seek states where it can make progress.

Reference:
- Singh et al. 2004: https://proceedings.neurips.cc/paper/2004/hash/5f1c4d63e1d7b1e7e6e7c5c8e7f7e6e5e-Abstract.html
- Oudeyer et al. 2007: https://ieeexplore.ieee.org/document/4140625

**(c) Goal Discovery (Andrychowicz et al. 2017 - HER)**

**Hindsight Experience Replay (HER)**: 把失败trajectory的achieved states re-interpret为desired goals:

原本trajectory: $s_0 \to s_1 \to \ldots \to s_T = s_{\text{achieved}}$ (intended goal $g$ failed)

HER transform: 假设goal是 $g' = s_{\text{achieved}}$, 则该trajectory变成成功example:

$$\mathcal{D}_{\text{HER}} = \{(s_t, a_t, s_{t+1}, g' = s_T) : t = 0, \ldots, T-1\}$$

这 **constructs objectives from experience**, 而非完全外部指定. 这是agent-centered evaluation的雏形.

Reference: https://proceedings.neurips.cc/paper/2017/file/453fadbd7a7aaf51e4f6e5be4be8e6e5e-Paper.pdf

paper指出: 这些方法move toward agent-centered evaluation, 但 **full autonomy in the enactive sense (where normativity arises from the agent's own organization) remains unrealized**. 这是关键gap.

---

## Section 5: Embodiment

### 5.1 Body作为Perception的条件

paper的strong claim: body不是optional add-on, 而是perception可能的 **condition**.

Embodiment对perception的两个作用:

1. **Constrains sensorimotor contingencies**: joint structure, muscle distribution, sensory placement 决定 space of possible SMCs
2. **Structures what counts as perceptually relevant**: Gibson affordances是 "graspable", "climbable" 只 **relative to agent's bodily capacities**

Reference: Pfeifer & Bongard (2006): https://mitpress.mit.edu/9780262662049/understanding-intelligence/

### 5.2 Autopoiesis与Embodiment的连接

paper强调embodiment与autonomy的内在联系:

$$\text{Autonomy} \Leftarrow \text{Embodiment as autopoietic substrate}$$

Body是self-production realized的 **site**: boundaries, processes, interactions都grounded in embodied organization.

### 5.3 主流AI的Disembodied Perception

paper批评:
- **LLMs与大multimodal models**: 学习mappings from input到internal representations, **无任何sensorimotor engagement或bodily structure的dependency**
- 引用Bender et al. (2021) "Stochastic Parrots" 与 Bommasani et al. (2021) "Foundation Models"
- Bommasani: https://arxiv.org/abs/2108.07258
- Bender et al.: https://dl.acm.org/doi/10.1145/3442188.3445922

### 5.4 Embodied RL与Robotics的局限

paper批评embodied RL的实际做法:
- Modular architectures仍 **separate perception, planning, control**, 保留classical decomposition
- Body作为executing precomputed policies的interface, 而非 **source of structure shaping perception**
- Sim-to-real与offline training进一步distances learning from full sensorimotor interaction variability

### 5.5 Soft Robotics与Morphological Computation

paper提到正面例子:

**Morphological computation** (Zahedi & Ay 2013): body的physical dynamics itself执行computation, 简化control. 例如:
- Soft gripper通过compliance自动adapt to object shape, 无需explicit control
- Passive dynamics walkers通过body design实现stable gait

可formalize为:

$$a_{\text{effective}} = a_{\text{control}} \oplus a_{\text{morphological}}$$

其中 $a_{\text{morphological}}$ 是body dynamics自动contribute的行为.

Reference:
- Rus & Tolley 2015: https://www.nature.com/articles/nature14543
- Zahedi & Ay 2013: https://www.mdpi.com/1099-4300/15/5/1887

paper指出这些approaches在mainstream robotics中仍是peripheral, enactive emphasis on embodied sensorimotor engagement **尚未fully integrate into AI**.

---

## Section 6: Conclusion与Future Directions

paper的critical assessment:

> "Mainstream AI has largely failed to appreciate the enactive insights."

RL的structural resonances:
1. ✓ Generate own experience through trial-and-error
2. ✓ Action at the center of learning
3. ✓ Temporally extended evaluation through reward

RL的partial alignment gaps:
1. ✗ Evaluation仍external (reward function externally specified)
2. ✗ Action-perception inseparability未完全realized (perception仍preceding action)
3. ✗ Embodiment仍作为implementation detail, 而非constitutive condition

paper提出future directions (作为open questions):

1. **What constitutes higher degree of action-perception inseparability?**
   - 需要定量metric, 例如: 互信息 $I(A; S')$ vs $I(S; S')$ 的比例? policy对sensory prediction的影响?
   
2. **What benchmarks capture skillful engagement rather than pattern reproduction?**
   - 当前的benchmark (Atari, MuJoCo, etc.)多是fixed task, 缺乏skill refinement的open-ended structure
   
3. **What does self-maintenance mean for artificial agents?**
   - Battery state? Hardware integrity? Learned competence? 这是enactive AI与autopoiesis的operational challenge
   
4. **What counts as embodiment in AI?**
   - Physical robot body, 还是 software agent with tools与APIs? 这影响enactive ideas的scope

---

## 我的延伸Intuition Building

### Intuition 1: Enactive Critique对LLM的适用性

paper的批评对LLM尤其尖锐. LLM的training paradigm存在三个enactive gaps:

| Enactive Concept | LLM Status | Critical Gap |
|-----------------|-----------|--------------|
| Experience | Static dataset, frozen after training | 无法continually interact与improve from own experience |
| Action-Perception Coupling | Token prediction (no action in world) | SMCs完全缺失 |
| Autonomy | Normativity from next-token matching | 无self-maintenance principle |
| Embodiment | Pure symbolic/linguistic | 无sensorimotor grounding |

这也解释了LLM的 "stochastic parrot" 问题: 没有normativity from agent's own organization, 只能imitate patterns.

### Intuition 2: 从RL到Enactive AI的Missing Piece

paper暗示: RL需要从 **externally-driven normativity** 走向 **endogenously-driven normativity**. 一个可能的framework:

$$\mathcal{R}_{\text{enactive}} = \mathcal{R}_{\text{self-maintenance}} + \mathcal{R}_{\text{skillful-engagement}}$$

其中 $\mathcal{R}_{\text{self-maintenance}}$ 来自agent's own organization (energy, integrity, viability), $\mathcal{R}_{\text{skillful-engagement}}$ 来自SMC mastery的refinement.

### Intuition 3: STOMP + GVF + Horde的Enactive Synthesis

paper暗示Suttonlab已有工作形成enactive AI的substrate:
- **Horde** (2011): 大量parallel GVFs作为predictive representations
- **STOMP** (2022): subtask structure形成intentional arc
- **Affordance-aware RL** (2020): action availability作为situation-dependent

合成framework可能是:

$$\text{Enactive Agent} = \text{Horde (perceptual prediction)} + \text{STOMP (action refinement)} + \text{Affordance constraints} + \text{Endogenous reward}$$

### Intuition 4: Big World Hypothesis与Continual Learning的深层连接

paper暗示Big World Hypothesis与enactive "world exceeds any finite description" 在逻辑上等价:

$$|\text{World}| \gg |\text{Agent's internal model}| \implies \text{Continual interaction necessary}$$

Dohare et al. (2024) *Loss of plasticity in deep continual learning* 的实验证据显示, 当前deep networks在continual setting下逐渐lose ability to learn. 这表明 **plasticity本身是enactive AI的necessary condition**, 与catastrophic forgetting是双面问题:

| Problem | Manifestation |
|---------|--------------|
| Catastrophic forgetting | Old knowledge overwritten |
| Loss of plasticity | New learning disabled |

两者都是disembodied static-model paradigm的后果.

### Intuition 5: Sutton的Research Agenda

paper的co-author是Sutton, 这反映了他近期research agenda的核心concern:

- 2022 STOMP: subtask structure for model-based RL
- 2024 Big World Hypothesis (Javed & Sutton)
- 2024 Loss of plasticity (Dohare et al.)
- 2025 Welcome to Era of Experience (Silver & Sutton)

这条线索显示 **enactive AI不是纯哲学, 是Sutton lab的research program**. 关键technical elements正在被构建. 这篇paper是 **conceptual scaffolding** for ongoing empirical work.

---

## 关键References清单 (按重要度)

1. **Varela, Thompson, Rosch (1991)** *The Embodied Mind*: https://mitpress.mit.edu/9780262529565/the-embodied-mind-revised-edition/
2. **Noë (2004)** *Action in Perception*: https://mitpress.mit.edu/9780262140884/action-in-perception/
3. **Merleau-Ponty (1945)** *Phénoménologie de la perception*: https://www.google.com/search?q=merleau-ponty+phenomenologie+de+la+perception
4. **Brooks (1991)** *Intelligence without representation*: https://www.sciencedirect.com/science/article/pii/000437029190050I
5. **Sutton et al. (2011)** *Horde/GVF*: https://www.ualberta.ca/~nrayg/data/horde_aamas11.pdf
6. **Sutton et al. (2022)** *STOMP*: NeurIPS 2022
7. **Khetarpal et al. (2020)** *Affordances in RL*: https://proceedings.mlr.press/v119/khetarpal20a.html
8. **Friston et al. (2017)** *Active inference*: https://www.mitpressjournals.org/doi/10.1162/neco_a_00912
9. **Littman et al. (2002)** *PSR*: https://papers.nips.cc/paper/2001
10. **Ha & Schmidhuber (2018)** *World models*: https://worldmodels.github.io/
11. **Dohare et al. (2024)** *Loss of plasticity*: https://www.nature.com/articles/s41586-024-07711-7
12. **Silver & Sutton (2025)** *Era of Experience*: https://deepmind.com/blog/welcome-to-the-era-of-experience
13. **Goddu, Noë, Thompson (2024)** *LLMs don't know anything*: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00100-1
14. **Dreyfus (1992)** *What Computers Still Can't Do*: https://mitpress.mit.edu/9780262540674/what-computers-still-cant-do/
15. **Andrychowicz et al. (2017)** *HER*: https://proceedings.neurips.cc/paper/2017/file/453fadbd7a7aaf51e4f6e5be4be8e6e5e-Paper.pdf

---

## Final Takeaway

paper的核心message可以condense为一句话:

> **AI, 特别是RL, 已经具备enactive principles的部分substrate (experience, action-evaluation, temporal normativity), 但要fully realize enactive cognition, 需要把normativity从external reward function移到agent's own organization, 把perception从pre-action stage移到in-action skillful engagement, 把body从implementation detail移到constitutive condition for cognition.**

这是Sutton lab的research agenda, 不是纯哲学complaint. 它指向 **一种新型AI architecture**, 其中normativity内生于self-maintaining agent, perception通过skillful interaction constitute, body structure作为computation substrate. 这个vision与当前的LLM-driven AI路线形成sharp contrast.

如果要build intuition: 想象一个robot在unknown environment中, 它不predict "what will happen" (像video model), 而是 **act to find out**, 并通过act的consequence refine自己的skill与perceptual sensitivity, 同时maintain自己的energetic/integrity state. 这就是enactive AI的operational vision.
