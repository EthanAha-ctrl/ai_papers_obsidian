---
source_pdf: Dont Yell at Your Robot.pdf
paper_sha256: 6ad3b9c18c0099d0a8dbe86948d899d057bfd233f0441c60143374e5181c0de8
processed_at: '2026-08-03T23:10:41-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：Don't Yell at Your Robot

## 一句话版本

**与其冲着机器人喊"不对不对往右往右！"，不如直接伸手推它一把 — 机器人能听懂你推的意思，还会把这个"教训"记下来下次不再犯**。

---

## Why this paper exists — 背景的痛点

你玩过那种用 ChatGPT 控制机器人的 demo 吗？比如 SayCan ([say-can.github.io](https://say-can.github.io/))，你跟机器人说"我渴了帮我拿可乐"，它就屁颠屁颠去拿。听起来很酷，但你真用起来会发现一个尴尬事实：**LLM 经常搞错事，但你想纠正它的时候，你得继续打字或继续说话 — 你说"不是这个可乐，是冰箱里那个不是桌子上的那个"，LLM 可能更糊涂**。

Stanford 之前有一篇叫 "Yell at Your Robot" ([yell-at-your-robot.github.io](https://yell-at-your-robot.github.io/))，意思是你真的就大声喊"no, to the right!" 来纠正机器人，然后 LLM 把这个 verbal correction 喂回去做 fine-tuning。UPenn 这帮人 (Nadia Figueroa 组) 就反着写了一篇叫 "Don't Yell at Your Robot"，title 本身是个 joke：**你吼啥呀，直接推它一把不就完了**。

这个 motivation 很直觉 — 你跟一个学徒工说"不是这样做"，跟你直接手把手把他的手按到对的位置，哪个更高效？显然是后者。语言 correction 有 ambiguity、有 latency、有 misunderstood 的风险；物理 correction 是 instantaneous、unambiguous、robust。**人类的本能就是物理交互，从婴儿学抓东西就是手把手教的**。

---

## 系统长啥样 — 一张图讲完

```
你 (人) ──推一把──┐
                  ↓
              [Kuka 机械臂]
                  ↓ (感受到力的方向)
        ┌──────────────────┐
        │ Particle Filter  │  ← 这层是"猜你想去哪"
        │ (一堆候选目标)    │
        └──────────────────┘
                  ↑↓
        ┌──────────────────┐
        │   LLM (GPT-4o)   │  ← 这层是"想下一步该干啥"
        │ "去拿 cooking pot" │
        └──────────────────┘
                  ↓
            [Semantic Action]
            "Pick; cooking pot"
                  ↓ (查表)
            [DS Action: 去pot的pose]
                  ↓
            机械臂开始动
```

人话讲：**LLM 当大脑，决定"干啥"；particle filter 当小脑，决定"具体往哪去"；机械臂当肌肉，负责执行**。你推一把 = 给小脑一个信号"我说错了，应该是那边的目标"，小脑猜出来后告诉大脑"刚才你想让我去 stove 不是 cutting board"，大脑记下来下次别犯同样错。

---

## 三个核心 trick，每个都很聪明

### Trick 1: 用 Dynamical System 当"action 的语言"

机械臂的运动本质是连续的 force/torque，LLM 输出的是离散的 token。中间怎么桥接？

UPenn 用了一个老朋友 — **Dynamical System (DS)**。简单说，DS 就是"目标点 + 收敛方式"。

$$
\dot{\mathbf{x}}^{\mathrm{d}} = \mathbf{A}\, d(\mathbf{x}, \mathbf{x}^*)
$$

逐个翻译：
- $\mathbf{x}$ — 机械臂末端现在在哪 (position $\mathbf{p}$ + orientation $\mathbf{q}$)
- $\mathbf{x}^*$ — 想让去的目标 pose
- $\mathbf{A}$ — 一个 6×6 对角负定矩阵，控制"以多快速度收敛"
- $d(\cdot, \cdot)$ — 当前 pose 跟目标 pose 之间的差
- $\dot{\mathbf{x}}^{\mathrm{d}}$ — 输出的 reference velocity

**人话**：DS 就是说"我要去 $\mathbf{x}^*$，去的速度跟距离成正比"。等价于一个 spring-damper 把 end-effector 拉向 attractor。好处：stable by construction (Lyapunov 一阶导数负定直接证完)，参数极简 (12 个 scalar 就描述一个完整 6-DoF motion policy)。

LLM 不输出 DS 参数，输出的是 semantic action "Pick; cooking pot"，然后通过一个 dictionary 查到 cooking pot 的 pose 当 $\mathbf{x}^*$。**这就是 symbol grounding**：LLM 的符号 ("cooking pot") 通过 perception module 的物理 anchor (pot 的 pose) 接地。

### Trick 2: 置信度可变阻抗控制 — 让机械臂"听人话"

这部分是最妙的。机械臂正常工作时，要 firm 地 track DS 给的轨迹。但人一推它，它得立刻变软让人推得动。怎么知道"现在人在推我"？

公式 (2)：

$$
c(t) = -\int_{-T}^{0} \|\dot{\mathbf{x}} - \dot{\mathbf{x}}^{\mathrm{d}}\|_2 \, ds
$$

翻译：**过去 $T$ 时间里，机械臂实际速度跟期望速度差了多少，累加起来，取负号再 clip 到 [0,1]**。

- 没人推 → 实际 ≈ 期望 → $c(t) \to 1$ → 高 confidence
- 有人在推 → 实际 ≠ 期望 → $c(t) \to 0$ → 低 confidence

然后用 $c(t)$ 调 controller gain：

$$
\mathbf{u}_{\mathrm{DS}} = -c(t)\, \mathbf{D}\, [\dot{\mathbf{x}} - \dot{\mathbf{x}}^{\mathrm{d}}]
$$

- $c \to 1$：满 gain，robot 硬邦邦地执行 LLM 给的动作
- $c \to 0$：gain 归零，robot 变成一团泥，你推哪它跟哪

**这就实现了 "compliance by confidence" — 人一推，robot 自动让路**。物理上这是 passive interaction control，参考 Kronander & Billard ([RA-L 2016](https://ieeexplore.ieee.org/document/7155952))。passive 是关键 — 系统不主动输出能量，只耗散能量，所以稳定。

### Trick 3: Particle Filter 把"推"翻译成"语义"

这是最 hack-the-intuition 的部分。你推 robot 的时候，robot 怎么知道你想推它去哪？

答案：**保持一群"假说粒子"，每个粒子代表一个候选目标 (一个 DS action)**。robot 实际速度是 observation，每个粒子根据"如果我的 attractor 真在这，那现在 end-effector 应该往哪个方向走"算 likelihood，跟 observation 比对。

粒子 $i$ 的权重：

$$
w_i \propto \exp\left(-\|\dot{\mathbf{x}} - \hat{\dot{\mathbf{x}}}_i\|_2^2\right)
$$

翻译：**"如果我的目标是 stove，那么现在我应该往 stove 方向走；但实际我往 sink 方向走了，说明我 (粒子) 错了，weight 降低"**。

聪明的点在 resampling：

$$
r(t) = 1 - c(t)
$$

- High confidence ($c \to 1$)：$r \to 0$，粒子维持 LLM 给的 DS action
- Low confidence ($c \to 0$)：$r \to 1$，粒子被 **uniformly 撒到所有"看到的物体对应的合法 DS action"上**

最后效果：你推 robot 朝 stove 方向，撒在 stove 那边的粒子 weight 越来越高，最后 dominate；estimate $\hat{\mathbf{x}}^*$ 收敛到 stove pose；然后 dictionary 反查，得出 semantic action "Move; on stove"。这个 semantic action 被塞进 LLM 的 context window，作为 correction 记录。

**整个流程相当于：物理推 → 概率猜测 → 符号翻译 → LLM 理解**。LLM 完全不知道你"推"了，它只看到"human corrected robot to: on the stove"，跟它本来理解的语言 correction 是一个形式。

---

## LLM 这一头做了啥

LLM 用 GPT-4o。System prompt 给它 ChefBot 角色 + kitchen 环境 + 三类物体 hierarchy：

- Category A: 可被 pickup 的容器
- Category B: 放置地点
- Category C: 食物，必须放在 A 上才能 manipulate

每次 LLM 被调用时，user prompt 包含：
1. 人和 robot 现在手里拿着啥
2. 人在朝哪个物体走 (用 particle filter 算的)
3. robot 上一步干啥
4. 上一步成功没
5. **人上一步有没有物理纠正 robot，纠正到哪**
6. 当前可用的 semantic action 列表

LLM 输出格式强制：`# Pick ; cooking pot &`，开头 `#`、用 `;` 分隔、结尾 `&`。这跟 Code as Policies ([code-as-policies.github.io](https://code-as-policies.github.io/)) 一脉相承 — 结构化输出便于 parsing。

---

## Experiment — 简陋但够说明问题

Setup 是 hybrid real + sim：
- **Real**: Kuka iiwa-14 机械臂 + Optitrack motion capture 追人手
- **Sim**: 物体位置虚拟，gazebo 渲染
- **Force**: 通过 joystick 传入 (没真用 load cell)

也就是说，"人推 robot"实际上是"人推 joystick 模拟推 robot"。这个 hack 是因为真正的 6-DoF 物理交互 + robust grasping 还做不出来 — 作者自己承认这是 future work。

实验任务：cooking beans。流程大致是：robot 帮人搬 pot → LLM 错说放 cutting board → 人推 robot 朝 stove → robot 改去 stove → LLM 记下 → 下次类似场景 LLM 直接说放 stove。

Table I 的数据 (20 trials per condition)：

| Correction 距离当前步数 | 0 | 5 | 10 | 15 |
|---|---|---|---|---|
| LLM 记住 correction 的成功率 | 100% | 85% | 85% | 80% |

**解读**：
- 100% 是 trivial 的 (刚纠正完当然记得)
- 5 步之后掉到 85% — GPT-4o 的 in-context recall 开始衰减
- 10 步和 5 步一样 85% — 出现 plateau，这跟"lost in the middle"现象 ([arxiv 2307.03172](https://arxiv.org/abs/2307.03172)) 吻合，context 中间位置信息 retention 不严格衰减
- 15 步还有 80% — 说明 cooking 这个 domain GPT-4o 有 prior knowledge 兜底 (它本来就"知道" beans 应该放 stove 而不是 cutting board)

80% 这个 floor 很关键 — 它暗示 correction 实际只贡献了 +20% 的提升，剩下 80% 是 LLM 自带的 world knowledge。换到 LLM 不熟悉的 domain（比如 surgical robotics、精密装配），baseline 会更低，correction 的相对贡献会更大，但也可能更难翻译成 semantic action。

---

## Hyperparameter 里的 intuition

Table II 几个有意思的数字：

```
Damping Gain Cartesian D_p: Low=1,   High=85
Damping Gain Rotation D_o:  Low=1,   High=13
Dynamics Cartesian A_p:     Low=-0.6, High=-0.4
Dynamics Rotation A_o:      Low=-0.9, High=-0.6
Ascent Rate Cartesian d_p:  0.41
Ascent Rate Rotation d_o:   0.49
```

直觉解读：

- $\mathbf{D}_p / \mathbf{D}_o \approx 6.5$：推 position 比推 rotation 难多了，所以 position damping 要 6.5 倍才能稳定 track
- $\mathbf{A}_p \in [-0.6, -0.4]$：position 的"弹簧"较软，因为 position 误差可能很大 (物体在 1 米外)
- $\mathbf{A}_o \in [-0.9, -0.6]$：rotation 的"弹簧"硬，因为 rotation 误差小 (一般几度)，可以激进点
- $d_p = 0.41, d_o = 0.49$：confidence 从 0 爬到 1 的时间常数 $\tau = 1/d$ 大约 2 秒 — 这就是人松手后 robot"缓过神"重新自信的时长，跟人的体感一致

Low 列是低 confidence (被推中) 的参数，High 列是高 confidence (自由执行) 的参数。整个 variable impedance 的 essence 就是：**confidence 低 → 软 → 让人推；confidence 高 → 硬 → 精确执行**。

---

## 这篇 paper 真正"破"在哪

我觉得最 elegant 的点不是任何单个公式，而是这个 **modality bridging 的架构**：

```
物理世界 (force, motion) 
    ↓ particle filter
符号世界
    ↓ LLM context
推理世界
```

**LLM 永远活在符号世界，物理 correction 永远活在物理世界，particle filter 是翻译官**。

为什么这很重要？因为 LLM 的 grounding 问题本质上是个 modality gap 问题 — LLM 训练在 text 上，但 robot 要在 force/pose 空间运作。RT-2 ([robotics-transformer2.github.io](https://robotics-transformer2.github.io/)) 走的是 end-to-end 路线：把 action token 化，让 VLM 直接吐 action token，端到端训练。这条路线 expressive 但数据 hungry (需要百万级 demonstration)。

这篇 paper 走相反路线：**保持 LLM 在符号层，让符号层跟物理层通过 DS + dictionary + particle filter 桥接**。好处是 data efficient (20 trials 就能 measure in-context learning)，坏处是 dictionary 限制了 expressive power — 你想让 robot "把 pot 推到 sink 旁边再倾斜 30 度"，没在 dictionary 里就做不到。

这两种 paradigm 在 robotics 里会一直并存。Modular 路线更可能先 deploy (因为不需要海量数据)，end-to-end 更可能成为 long-term solution (因为表达能力强)。这种张力跟 classic AI 里 GOFAI vs connectionism 的张力是同一个 thing。

---

## 跟我（Karpathy）熟悉的 LLM training 思路对照

我自己看这 paper 时不由自主联想到 RLHF：

| | RLHF (LLM training) | This paper (robot correction) |
|---|---|---|
| Human feedback modality | Thumb up/down 或 ranking | 物理推 robot |
| Feedback 转换 | Reward model | Particle filter → semantic action |
| Feedback 用途 | Fine-tune policy | In-context learning |
| Policy | LLM weights | LLM (GPT-4o) frozen |

两种 case 都是把 human feedback 转成 LLM 能 digest 的形式。RLHF 转成 scalar reward，这里转成 symbolic correction。**核心 insight 一致：human feedback 是 ultimate grounding signal，关键是找到把 feedback "翻译" 成 LLM 能吃的格式的 bridge**。

更进一步，这跟 robot learning 里的 IRL (inverse reinforcement learning) 一脉相承 — Losey et al. ([IJRR 2022](https://journals.sagepub.com/doi/10.1177/02783649211045658))、Bobu et al. ([HRI 2021](https://dl.acm.org/doi/10.1145/3434073.3444647)) 都用 physical correction 学 human objective。但 IRL 需要一个 learning phase 才能 deploy，这篇 paper 用 particle filter 做 real-time estimation + LLM 做 reasoning，**绕开了 learning phase**。

这其实是 robotics 越来越明显的趋势：**用 LLM 当 zero-shot policy，用 probabilistic estimator 做 real-time adaptation，用 human feedback 做 in-context refinement**。三者分工：LLM 提供 prior，estimator 提供 posterior，human feedback 提供 likelihood update。

---

## 我（Karpathy）的几个批评

读这篇 paper 时我心里有几个 question mark：

**1. Perception bottleneck**: particle filter 把粒子撒在"perceived objects"上。如果人想推 robot 去 perception 没看到的物体 (比如远处的桌角)，整个 system 失败。这篇 paper 把这个 limitation 藏起来了，因为 cooking scenario 所有相关物体都在 perception list 里。换成 open-world scenario 立刻崩。

**2. Dictionary 的 expressive ceiling**: semantic action 是 pre-defined 的 6 种。复杂 compound action ("把 pot 推到 sink 旁边再 tilt 30 度") 表达不出来。Code as Policies ([arxiv 2209.07753](https://arxiv.org/abs/2209.07753)) 用 code generation 解决了这个，这里没有。这是 modular architecture 的 inherent trade-off。

**3. Time scale mismatch**: LLM query 1-3 秒，particle filter 20 Hz，controller 200 Hz。当 LLM 在思考时 robot 在干啥？paper 没明说。如果 robot idle，体验有卡顿；如果 robot 继续执行 last-DS，可能 LLM 出来时 robot 已经到了不该去的地方。这是 hierarchical control 经典问题，paper 没正面回答。

**4. Hybrid sim 的 cheating**: joystick 模拟物理推力，没有真实 load cell 数据。真实 6-DoF 物理交互的 force profile 比 joystick 复杂得多 (有 friction、有 inertia、有 contact patch)。当真正接 load cell 时，particle filter 的 observation model 可能需要重新设计。

**5. Cooking domain 的 cherry-picking**: cooking 是 LLM 最熟悉的 domain 之一，GPT-4o 在 cooking recipe 上训练充分。80% baseline 是 LLM prior 兜底，correction 只贡献 +20%。换到 LLM 不熟悉的 domain，correction 的相对贡献会大，但 translation 难度也会大 (更难把 force 翻译成 LLM 能懂的 semantic action)。

**6. In-context learning 的 robustness**: Table I 只测了"同一 semantic state 下 LLM 能否 recall 之前的 correction"。没测 cross-domain generalization — 如果人纠正过"pot 去 stove"，LLM 遇到"pan 去 stove"会自动 generalize 吗？还是每次都要从头纠正？这是衡量 in-context learning 真正有效性的关键实验，paper 没做。

---

## 跟我（Karpathy）自己工作 (Eureka Labs, nanoGPT 等) 的 connection

我自己最近在折腾 education 用 LLM ([eurekalabs.com](https://www.eurekalabs.com/))，看这篇 paper 时想到一个类比：

**教育场景里，学生犯错时老师有两种 correction 方式**：
1. 语言：("你这步算错了，重新看第二步")
2. 操作：老师拿过笔在学生纸上直接改

LLM tutor 现在只能做 (1)。这篇 paper 启示是：如果学习场景是 embodied 的 (比如学生学写书法、学弹钢琴)，**物理 correction 通过 sensor 转成符号，喂回 LLM tutor** — 这就是 embodied AI tutor 的雏形。

更具体的 connection：Karpathy 在 nanoGPT 里强调过 in-context learning 的本质是"implicit gradient descent on activations"([arxiv 2211.07663](https://arxiv.org/abs/2211.07663))。这篇 paper 的 in-context correction 也可以从这个视角看：**每次物理 correction = 一次 implicit policy update**。20 步内 retention 衰减到 80%，对应于 attention 机制对 context 信号的 decay — 这是 attention decay 的物理 embodiment 测量。

---

## Big picture — 这篇 paper 在哪个 narrative 里

我把它放在三个 narrative 里：

### Narrative 1: LLM-grounded robotics 的 modality 演化
- SayCan (2022): 纯 language → affordance score
- Inner Monologue (2022): language + visual feedback
- Code as Policies (2022): language → code → actions
- RT-2 (2023): vision-language-action end-to-end tokenization
- Yell at Your Robot (2024): verbal correction → VLA fine-tuning
- **Don't Yell (2024): physical correction → LLM in-context**

时间线很清楚：modality 越来越 rich，从 text 到 speech 到 physical force。这篇是 force modality 的 first attempt。

### Narrative 2: Probabilistic estimation vs deep learning 的拉锯
- Pre-2012: probabilistic estimation 主导 (particle filter, Kalman filter)
- 2012-2020: deep learning 主导 (CNN, RNN, transformer)
- 2020+: hybrid 回潮 — 深度模型提供 prior，probabilistic model 提供 posterior

这篇 paper 是 hybrid 范式的典型 — LLM 提供 prior (semantic action)，particle filter 做 posterior estimation (DS parameters)。跟我熟悉的 neural rendering (NeRF) 思路同源 — NN 提供 prior，volume rendering 提供 differentiable forward model。

### Narrative 3: Human-AI collaboration 的 interface 设计
- GUI interface (WIMP 模型) — 鼠标键盘
- Language interface (ChatGPT) — text
- Multimodal interface (GPT-4o) — text + image + audio
- Embodied interface (this paper) — physical interaction

Embodied interface 是 frontier — 人类之间最高带宽的协作就是物理合作 (一起搬桌子、一起做手术)，AI 系统到这个 level 才算真正 embodied。

---

## 给你的 takeaway（Karpathy 视角）

如果让我给 Andrej Karpathy 一句话总结这 paper，我会说：

**"Physical correction is to robotics what thumb-up is to ChatGPT — a low-bandwidth but high-signal feedback channel that, with the right probabilistic bridge, can be made digestible by the language model. This paper shows the bridge; the next decade is about widening it."**

更具体地：

1. **Particle filter + DS 是 LLM-friendly 的 grounding mechanism**，跟 RT-2 的 tokenization 是两个极端，各自有 trade-off
2. **Confidence-based variable impedance 是 "compliance on demand" 的优雅实现**，把"听话"做成了一个数学性质
3. **Modality bridging 比 end-to-end 在 short term 更 deployable**，但 long term 还得看 end-to-end 能否 scale
4. **In-context learning 在 robotics 里是真有用的**，但 decay 曲线 (Table I) 说明不能无限依赖，long-horizon task 还得 fine-tuning 或 RAG

我个人觉得这 paper 的真正 contribution 不是任何一个公式，而是 **architecture pattern**：LLM 在顶、probabilistic estimator 在中、compliant controller 在底。这个 pattern 后面会被无数次 instantiate。比如把 LLM 换成 VLA、把 particle filter 换成 diffusion model、把 variable impedance 换成 MPC — 同样的 pattern，不同的 instantiation。

---

## 参考链接

- **Paper 作者主页**: [https://chuye-zhang.github.io/](https://chuye-zhang.github.io/) | [https://www.seas.upenn.edu/~nadiafig/](https://www.seas.upenn.edu/~nadiafig/)
- **GRASP Lab**: [https://www.grasp.upenn.edu/](https://www.grasp.upenn.edu/)
- **Shao et al. RSS 2024 (DS co-manipulation)**: [https://arxiv.org/abs/2406.13356](https://arxiv.org/abs/2406.13356)
- **Yell at Your Robot (对标工作)**: [https://yell-at-your-robot.github.io/](https://yell-at-your-robot.github.io/)
- **SayCan**: [https://say-can.github.io/](https://say-can.github.io/)
- **Inner Monologue**: [https://innermonologue.github.io/](https://innermonologue.github.io/)
- **Code as Policies**: [https://code-as-policies.github.io/](https://code-as-policies.github.io/)
- **RT-2**: [https://robotics-transformer2.github.io/](https://robotics-transformer2.github.io/)
- **TidyBot**: [https://tidybot.cs.princeton.edu/](https://tidybot.cs.princeton.edu/)
- **TransIC (sim-to-real via correction)**: [https://transic-2024.github.io/](https://transic-2024.github.io/)
- **Octo generalist policy**: [https://octo-models.github.io/](https://octo-models.github.io/)
- **Kronander & Billard passive DS**: [https://ieeexplore.ieee.org/document/7155952](https://ieeexplore.ieee.org/document/7155952)
- **Khoramshahi & Billard human guidance detection**: [https://link.springer.com/article/10.1007/s10514-020-09929-y](https://link.springer.com/article/10.1007/s10514-020-09929-y)
- **Figueroa DS learning (CoRL 2018)**: [https://proceedings.mlr.press/v87/figueroa18a.html](https://proceedings.mlr.press/v87/figueroa18a.html)
- **Losey et al. physical interaction IJRR 2022**: [https://journals.sagepub.com/doi/10.1177/02783649211045658](https://journals.sagepub.com/doi/10.1177/02783649211045658)
- **Bobu et al. feature expansive reward HRI 2021**: [https://dl.acm.org/doi/10.1145/3434073.3444647](https://dl.acm.org/doi/10.1145/3434073.3444647)
- **Lost in the middle (in-context recall)**: [https://arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)
- **Von Oswald et al. in-context as gradient descent**: [https://arxiv.org/abs/2211.07663](https://arxiv.org/abs/2211.07663)
- **GPT-4o technical report**: [https://arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774)
- **rlabbe Kalman/Bayesian filters (PF 教程)**: [https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
- **iiwa_ros ROS stack**: [https://github.com/epfl-lasa/iiwa_ros](https://github.com/epfl-lasa/iiwa_ros)
- **Karpathy Eureka Labs**: [https://www.eurekalabs.com/](https://www.eurekalabs.com/)
- **Physical Intelligence π0**: [https://www.physicalintelligence.company/](https://www.physicalintelligence.company/)
- **Covariant RFM**: [https://covariant.ai/](https://covariant.ai/)

---

# Don't Yell at Your Robot — 深度技术解读

## 一、Paper 的核心 thesis

这篇 paper 来自 UPenn GRASP Lab (Nadia Figueroa 组)，核心 thesis 非常清晰：**LLM-powered robot 不应该只用 language 这条 single modality 接口来交互；物理 correction (push the end-effector) 是 instantaneous、unambiguous、robust 的 feedback channel，应该被 absorb 进 LLM 的 context window**。

这个 thesis 直接对标了一组 Stanford 的工作 — Shi et al. 的 "Yell at Your Robot" ([yell-at-your-robot.github.io](https://yell-at-your-robot.github.io/)) 和 Cui et al. 的 "No, to the right" ([arxiv 2307.01858](https://arxiv.org/abs/2307.01858))。但 Penn 这篇走的是物理路径，通过一个 special mount 让 robot 能 pickup / dropoff / tilt 6-DoF 物体，所以"correcting the robot by pushing it"在物理上是 well-defined 的。

paper title 是一句反讽："Don't Yell" — 因为物理推一下比" yell 'no, to the right'!"更高效。

---

## 二、Method 三层结构 — Intuition 拆解

整个 pipeline 可以拆成三层 stack，从下往上越来越抽象：

```
┌────────────────────────────────────────────────┐
│  Layer 3: LLM (GPT-4o) — semantic action space │
└────────────────────────────────────────────────┘
                  ↑   ↓ (semantic ↔ DS dictionary)
┌────────────────────────────────────────────────┐
│  Layer 2: Particle Filter over DS params (xA)  │
└────────────────────────────────────────────────┘
                  ↑   ↓ (estimate, control update)
┌────────────────────────────────────────────────┐
│  Layer 1: Variable Impedance Controller (Kuka) │
└────────────────────────────────────────────────┘
```

直觉：**LLM 是离散、symbolic 的高层；robot controller 是连续、force-based 的底层；中间的 particle filter + DS parameterization 是 bridge**。这一点非常关键 — LLM 不能直接输出 wrench 或 joint torque，因为 LLM 没有物理 grounding；但 LLM 可以输出 "pick up the cooking pot" 这种 semantic action，然后 dictionary 查到对应的 DS attractor $\mathbf{x}^*$ 和 dynamics $\mathbf{A}$，再用 DS 生成 reference velocity。

---

## 三、Dynamical System Action — 公式逐符号解析

核心方程 (1)：

$$
\dot{\mathbf{x}}^{\mathrm{d}} = \mathbf{f}(\mathbf{x}; \mathbf{A}, \mathbf{x}^*) = \mathbf{A}\, d(\mathbf{x}, \mathbf{x}^*)
$$

逐项拆解：

- $\mathbf{x} = [\mathbf{p}, \mathbf{q}]$ — end-effector 的 full pose state
  - $\mathbf{p} \in \mathbb{R}^3$ — Cartesian 位置 (e.g. xyz in world frame)
  - $\mathbf{q} \in \mathbb{S}^3$ — unit quaternion，所以是 3-sphere 上的元素，自带 normalization 防漂移
- $\dot{\mathbf{x}}^{\mathrm{d}} = [\dot{\mathbf{p}}, \boldsymbol{\omega}] \in \mathbb{R}^6$ — desired reference velocity
  - $\dot{\mathbf{p}} \in \mathbb{R}^3$ — linear velocity
  - $\boldsymbol{\omega} \in \mathbb{R}^3$ — angular velocity（注意 quaternion 的导数不是直接 $\dot{\mathbf{q}}$，而是 body-frame angular velocity）
- $\mathbf{x}^* = [\mathbf{p}^*, \mathbf{q}^*]$ — attractor pose，是 DS 的目标点
- $\mathbf{A} \in \mathbb{R}^{6 \times 6}$ — diagonal negative definite dynamics matrix
  - 对角负定意味着 $\mathbf{A}$ 的所有特征值都是负的
  - 直觉：相当于 spring-damper 系统中的 $-k - b$，让 $\mathbf{x}$ 以指数速率收敛到 $\mathbf{x}^*$
  - 对角意味着 position 和 orientation 各 3 个 DoF 解耦
- $d(\mathbf{x}, \mathbf{x}^*)$ — abstract difference function，来自 Shao et al. 的 [Constraint-aware intent estimation](https://arxiv.org/abs/2406.13356) (RSS 2024)
  - 对 position 是普通欧氏差 $\mathbf{p} - \mathbf{p}^*$
  - 对 quaternion 是 geodesic 距离 (shortest rotation on $\mathbb{S}^3$)，保证 quaternion 不会通过"long way around"绕路

**为什么用 DS 而不是直接的 trajectory 或 MPC**？

- DS 是 closed-form、autonomous、stable by construction (因为 $\mathbf{A}$ 负定，Khalil 非线性系统理论里 $\mathbf{x}^*$ 是 global asymptotically stable equilibrium)
- DS 可以 online 被 edit，例如 Li & Figueroa 的 elastic DS ([arxiv 2310.03821](https://arxiv.org/abs/2310.03821)) 允许 obstacle avoidance
- DS 的参数 $(\mathbf{x}^*, \mathbf{A})$ 是 low-dimensional (6+6=12 个 scalar)，正好 fit particle filter 的状态空间

直觉上，DS action 是一个"goal + stiffness profile"的紧凑 encoding：你想让 end-effector 去 $\mathbf{x}^*$，并且以 $\mathbf{A}$ 决定的收敛速率去那里。这跟 RT-2 等 end-to-end VLA policy ([rt-x.github.io](https://rt-x.github.io/)) 完全不一样 — 那种是黑盒 policy 输出 raw action，没有显式的 attractor structure。

---

## 四、Confidence-based Variable Impedance — 物理直觉

公式 (2) confidence measure：

$$
c(t) = -\int_{-T}^{0} \|\dot{\mathbf{x}} - \dot{\mathbf{x}}^{\mathrm{d}}\|_2 \, ds
$$

注意这里前面有负号 — 因为积分里面 $\|\cdot\|$ 是 non-negative，积分出来也是 non-negative，加负号后再 clip 到 $[0, 1]$。**直觉**：$c(t)$ 测量过去 $T$ 时间窗口内，end-effector 实际速度与 DS 期望速度之间的累积偏差。

- 没人推 robot → $\dot{\mathbf{x}} \approx \dot{\mathbf{x}}^{\mathrm{d}}$ → integral 小 → $c(t) \to 1$ → 高 confidence
- 人推 robot → $\dot{\mathbf{x}} \neq \dot{\mathbf{x}}^{\mathrm{d}}$ → integral 大 → $c(t) \to 0$ → 低 confidence

公式 (3) variable damping-only impedance：

$$
\mathbf{u}_{\mathrm{DS}} = -c(t)\, \mathbf{D}\, [\dot{\mathbf{x}} - \dot{\mathbf{x}}^{\mathrm{d}}]
$$

- $\mathbf{u}_{\mathrm{DS}} \in \mathbb{R}^6$ — Cartesian wrench (force + torque)
- $\mathbf{D} \in \mathbb{R}^{6 \times 6}$ — constant diagonal negative definite damping gain（看 Table II，$\mathbf{D}_p = 85$ high，$\mathbf{D}_o = 13$ high，可见 position damping 远大于 orientation damping — 因为推位置比推角度更费劲，需要更高 gain 才能稳定 track）

关节力矩恢复：
$$
\boldsymbol{\tau}_{\boldsymbol{\theta}} = \mathbf{J}(\boldsymbol{\theta})^{\top} \mathbf{u}_{\mathrm{DS}}
$$

$\mathbf{J}(\boldsymbol{\theta}) \in \mathbb{R}^{6 \times 7}$ 是 Kuka iiwa-14 (7-DoF arm) 的 Jacobian。$\mathbf{J}^{\top}$ 是 transpose，这是经典的 statically consistent force-to-torque mapping，参考 Khatib 1987。

**这个 controller 的精髓**：当 $c(t) \to 0$，$\mathbf{u}_{\mathrm{DS}} \to 0$，robot becomes *passive* — 也就是说人推 robot 时，robot 几乎不抵抗，pure damping mode。这就是 "Don't Yell" 的物理实现：robot 听话、给人让路。

passivity 证明来自 Kronander & Billard 的 [Passive interaction control with DS](https://ieeexplore.ieee.org/document/7155952) (RA-L 2016) 和 Khoramshahi & Billard 的 [Detection and reaction to human guidance](https://link.springer.com/article/10.1007/s10514-020-09929-y) (Autonomous Robots 2020)。直觉：因为 controller 里只有 damping term（没有 stiffness term 直接拉回 $\mathbf{x}^*$），系统是 purely dissipative 的，所以从能量角度 passive。

---

## 五、Particle Filter — 把物理意图"读"出来

这是 paper 最 elegant 的部分。状态空间是 DS 参数 $(\mathbf{x}^*, \mathbf{A})$。

**关键设计**：resampling rate $r(t) = 1 - c(t)$。

- High confidence ($c \to 1$)：$r \to 0$，粒子维持原状，robot 执行 LLM 给的 DS action
- Low confidence ($c \to 0$)：$r \to 1$，大部分粒子被 *uniformly resampled over valid DS actions on perceived objects* — 也就是说，粒子被撒到所有"看到的物体对应的合法 DS action"上

观察模型：粒子 $i$ 的权重

$$
w_i \propto \exp\left(-\|\dot{\mathbf{x}} - \hat{\dot{\mathbf{x}}}_i\|_2^2\right)
$$

- $\dot{\mathbf{x}}$ — 实际 end-effector velocity
- $\hat{\dot{\mathbf{x}}}_i$ — 粒子 $i$ 通过公式 (1) 预测的 velocity，用粒子自己的 $(\mathbf{x}_i^*, \mathbf{A}_i)$ 算

这是 Gaussian observation noise 假设下的标准 importance weighting。**直觉**：哪个粒子的 DS attractor $\mathbf{x}_i^*$ 跟人推的方向一致，哪个粒子的 $\hat{\dot{\mathbf{x}}}_i$ 就跟 $\dot{\mathbf{x}}$ 接近，权重就高。

人推 robot 朝 stove 方向，那么 attractor 在 stove 附近的粒子权重就会 dominate；当 $c$ 回升、$r$ 下降后，粒子收敛到 stove 附近，estimate $\hat{\mathbf{x}}^*$ 就是 stove 的 pose。

最终 estimate 用加权平均：
$$
\hat{\mathbf{x}}^* = \sum_i w_i \mathbf{x}_i^*, \quad \hat{\mathbf{A}} = \sum_i w_i \mathbf{A}_i
$$

(对 $\mathbf{x}^*$ 是位置平均 + quaternion slerp 加权；对 $\mathbf{A}$ 因为是对角矩阵所以是各对角元素加权平均。)

---

## 六、LLM 的 integration — System Prompt 设计

paper 的 Appendix A 给了完整的 system prompt，这里有几个值得注意的设计：

1. **ChefBot 角色** + kitchen environment — 场景 domain-specific，让 LLM 不用 general reasoning 也能给出合理 action
2. **三类物体 hierarchy**：
   - Category A (有 mount，可被 pickup)：cooking pot, gallon of water, cutting board
   - Category B (放置地点，是 environment 而非 object)：on the stove, in the sink, on the counter
   - Category C (无 mount，必须放在 A 上才能 manipulate)：lettuce, chicken breast, beans
   
   这个 hierarchy 是 *affordance* 的 explicit encoding — 直接对应 Gibson 的 affordance 概念和 SayCan ([say-can.github.io](https://say-can.github.io/)) 里的 affordance scoring。

3. **Action syntax**：`# Action ; object &` — 强制结构化输出，例如 `# Pick ; cooking pot &`。这跟 Code as Policies ([code-as-policies.github.io](https://code-as-policies.github.io/)) 的代码生成思路一致，但更简洁。

4. **In-context learning through interaction history**：
   ```
   In the previous step, the human corrected the robot's action by pushing it to: 'on the stove'.
   The final action executed by the robot was: Move 'on the stove'.
   ```
   每次 correction 都被翻译成 semantic correction 后 append 到 interaction history，下一次 prompt 时一起送给 LLM。

**关键的 bidirectional dictionary**：

$$
\text{Semantic action} \xleftrightarrow{\text{interface manager}} \text{DS action } (\mathbf{x}^*, \mathbf{A})
$$

- Semantic → DS：直接查表
- DS → Semantic：$\arg\min_i \|\mathbf{x}_i^* - \hat{\mathbf{x}}^*\|_2$，找最近的 DS action，再反向查 semantic

这个 dictionary 是整个 system 的 grounding — LLM 不直接跟物理世界打交道，而是通过 dictionary 间接 binding。**直觉**：dictionary 就是 Borchert / Harnad 的 "symbol grounding problem" 在 robotics 里的实现 — symbols (semantic actions) 通过 perceptual anchors (object poses from perception module) 接地。

---

## 七、Experiment — Hybrid Real + Sim Setup

实验是 hybrid digital twin：
- **Real**：Kuka iiwa-14 manipulator + Optitrack motion capture 追踪人手
- **Sim**：object 位置虚拟，gazebo 渲染
- **Force**：通过 joystick 传入 (real human force 没有直接 load cell 测量)

这个 setup 略 hacky — 作者承认 robust grasping hardware 是 future work。但 hybrid 让他们可以测 LLM + DS + physical correction 这条 pipeline，不需要先解决 grasping。

**Table I 的 in-context learning 测试**：

| Correction n steps ago | 0 | 5 | 10 | 15 |
|---|---|---|---|---|
| Success Rate | 100% | 85% | 85% | 80% |

20 trials per $n$。这条曲线很有意思 — 5 步之后就开始掉到 85%，但 10 步和 5 步一样，15 步只再掉 5%。**直觉解读**：LLM (GPT-4o) 的 in-context recall 不是单调衰减的，存在"plateau" — 这跟 recent work on ICL "lost in the middle" 现象 ([arxiv 2307.03172](https://arxiv.org/abs/2307.03172)) 印证：context 中靠后或中间位置的信息 retention 不 strictly 衰减。

80% 的 floor 也暗示：GPT-4o 的 base knowledge (cooking beans 要放 stove) 本身就能给 80% 的成功率，correction 只是再 +20%。

---

## 八、Table II 超参 — 数值直觉

```
Damping Gain Cartesian D_p: Low=1, High=85
Damping Gain Rotation D_o:  Low=1, High=13
Dynamics Cartesian A_p:     Low=-0.6, High=-0.4
Dynamics Rotation A_o:      Low=-0.9, High=-0.6
```

- $\mathbf{D}_p / \mathbf{D}_o \approx 6.5$：position 比 rotation 难推，需要 6.5x 的 damping 才能稳定 track
- $\mathbf{A}_p \in [-0.6, -0.4]$：position 的 dynamics 较"软"，收敛慢
- $\mathbf{A}_o \in [-0.9, -0.6]$：rotation 的 dynamics 较"硬"，收敛快 — 因为 rotation 误差通常更小，可以 afford 更激进的 gain

低 confidence 时切换到 Low column（更软、更 compliant），高 confidence 切换到 High column（更 firm、更 accurate tracking）。这就是 variable impedance 的 essence — impedance 跟着 confidence 走。

`Ascent Rate Cartesian d_p = 0.41`，`Ascent Rate Rotation d_o = 0.49` — 这是 confidence 从 0 爬升到 1 的速率参数，近似 $\dot{c} = d(1-c)$ 的一阶 ODE，time constant $\tau \approx 1/d$，所以 position 大约 2.4 秒爬回 confidence，rotation 2.0 秒。这与人在物理 correction 后松手、robot 重新稳定下来的 typical 时长吻合。

---

## 九、跟相关工作的横向对比

| Paper | Interface | Correction Modality | Real-time? | Updates LLM? |
|---|---|---|---|---|
| SayCan ([arxiv 2204.01691](https://arxiv.org/abs/2204.01691)) | Language | None (just prompt) | No | No |
| Inner Monologue ([arxiv 2207.05608](https://arxiv.org/abs/2207.05608)) | Language + visual feedback | Text | Yes | In-context |
| Code as Policies ([arxiv 2209.07753](https://arxiv.org/abs/2209.07753)) | Language → Code | None | Yes | No |
| TidyBot ([arxiv 2305.05658](https://arxiv.org/abs/2305.05658)) | Language + LLM summarization | Text (post-hoc) | No | Fine-tunes summary |
| Yell at Your Robot ([arxiv 2403.12910](https://arxiv.org/abs/2403.12910)) | Verbal corrections | Speech | Yes | Fine-tunes VLA |
| TransIC ([arxiv 2406.10930](https://arxiv.org/abs/2406.10930)) | Physical corrections (sim) | Force | Yes | Fine-tunes sim-to-real policy |
| **This paper** | Physical corrections | **Force on end-effector** | **Yes** | **In-context via history** |

**关键差异**：这篇是第一个把物理 correction 通过 particle filter 转成 symbolic feedback 喂回 LLM 的工作。TransIC 也用物理 correction，但它的目标是用 correction 数据 fine-tune 一个 sim-to-real policy；这里则是把 correction 翻译成 semantic action (e.g. "Move on stove") 塞进 LLM context window。

这种设计的好处：**LLM 永远在 reasoning 层，物理 correction 在 perception 层**。Layer separation 干净，LLM 不需要学习 force 视角的 reasoning，只需要理解 "human wanted me to go to stove instead of cutting board" 这种符号化的因果。

---

## 十、Limitations & 直觉思考

paper 自己点出了几个 future work：
1. **Robust grasping** — 当前用 hybrid sim 规避了真 grasping
2. **Alternative LLM integration** — fine-tuning vs RAG vs pure in-context 都没比较

我自己的直觉补充：

1. **Particle filter 的 prior 问题**：粒子被 uniformly resampled 到"所有 perceived objects"上，但如果人对的 object 不在 perception list 里，整个 system 会 stuck。比如人想推 robot 去"那边的桌角"，但 perception 没看到桌角 → particle filter 无法 converge 到正确 attractor。这是 perception bottleneck。

2. **Single correction modality**：现在只能"纠正方向"，不能"纠正力度"、"纠正速度"、"纠正 intermediate pose"。比如人想 robot 倾斜 30 度而不是 20 度，但 tilt action 是 fixed 20 度，物理推也不会改变 semantic action。要支持这种 correction 需要 continuous action space 的 LLM grounding，超出当前 dictionary 设计。

3. **Time scale mismatch**：LLM query 大约 1-3 秒（GPT-4o API call），particle filter 是 20 Hz (50 ms)，controller 是 200 Hz (5 ms)。三层 stack 的 latency gradient 很大。当 LLM 在思考时，robot 是 idle 还是 last-DS 继续？paper 没说清。如果 idle，体验会有"卡顿感"。

4. **Quaternion attractor 的 stability**：$\mathbf{q}^* \in \mathbb{S}^3$ 有 double cover 性质，$\mathbf{q}$ 和 $-\mathbf{q}$ 是同一个 rotation。Particle filter 的 weighted average 在 $\mathbb{S}^3$ 上需要 enforce hemisphere constraint，否则会出现"flip"导致 attractor 跳到 antipodal point。Figueroa 之前的工作 [Physically-consistent Bayesian non-parametric mixture](https://proceedings.mlr.press/v87/figueroa18a.html) (CoRL 2018) 处理过这个问题，这里继承了下来。

5. **LLM 推理的 grounding gap**：semantic action dictionary 是 pre-defined 的，LLM 不能"invent"新 action。比如人想"把 pot 推到 sink 旁边再 tilt 一下"，这种 compound action 没在 dictionary 里，LLM 只能拆成两步执行。这跟 Code as Policies 的 expressive code generation 形成对比 — 后者可以动态组合 primitives。

6. **Cooking scenario 的 choice bias**：选 cooking 是因为 LLM 对 cooking 有大量 prior knowledge (GPT-4o 在 cooking recipe 上训练充分)，所以 80% baseline 已经很高。换到不那么 LLM-friendly 的 domain（比如精密装配、surgical assistance），baseline 会大幅下降，correction 的相对贡献可能更大但也可能更难翻译成 semantic action。

---

## 十一、Intuition 上的 takeaway

如果让我提炼这篇 paper 留给后人的关键 idea，是这一句：

**"Physical interaction is a high-bandwidth, low-latency, low-ambiguity feedback channel that, via a probabilistic bridge (particle filter + DS parameterization), can be translated into symbolic feedback digestible by LLMs."**

这跟 RLHF 在 LLM 训练里的哲学一脉相承 — 人类偏好通过 reward model 转成 scalar 信号，再 fine-tune policy。这里类似：人类物理 correction 通过 particle filter 转成 semantic action，再 in-context fine-tune LLM。**Human feedback 是 ultimate 的 grounding signal，不管是 RLHF 的 thumb-up/thumb-down，还是 robotics 里的物理推一下**。

更深一层的 intuition： robotics 跟 NLP 的根本差异在于 *grounding modality*。NLP 里 grounding 是文字（instruction → text response）；robotics 里 grounding 是物理（force → motion）。这篇 paper 把物理 grounding 通过 particle filter 转换成符号 grounding，让 LLM 这个文字引擎能 work in robotics — 这是 modality bridging 的工作。

类似 philosophy 在 covariant 的 RFM (Robotics Foundation Model)、physical intelligence 的 π0 ([physicalintelligence.company](https://www.physicalintelligence.company/))、google deepmind 的 RT-2 ([robotics-transformer2.github.io](https://robotics-transformer2.github.io/)) 里都能看到不同 flavor：RT-2 是直接把 action token 化塞进 VLM；这篇是间接 — LLM 只产 semantic action，action 执行由 DS + particle filter 接管。

**两种 paradigm 的 trade-off**：
- RT-2 (end-to-end)：expressive 但数据 hungry，需要百万 demonstration
- This paper (modular)：data efficient (20 trials 就能 measure in-context learning) 但 dictionary 限制 expressive power

 Modular 方式更可能先 deploy，end-to-end 更可能成为 long-term solution。这种 modular ↔ end-to-end 的张力在 robotics 里会持续很长时间。

---

## 参考链接

- **Paper PDF (作者主页)**: [https://chuye-zhang.github.io/](https://chuye-zhang.github.io/) 和 [https://www.seas.upenn.edu/~nadiafig/](https://www.seas.upenn.edu/~nadiafig/)
- **GRASP Lab**: [https://www.grasp.upenn.edu/](https://www.grasp.upenn.edu/)
- **Shao et al. RSS 2024 (DS co-manipulation base)**: [https://arxiv.org/abs/2406.13356](https://arxiv.org/abs/2406.13356)
- **Figueroa DS 学习经典**: [https://proceedings.mlr.press/v87/figueroa18a.html](https://proceedings.mlr.press/v87/figueroa18a.html)
- **Kronander & Billard passive DS control**: [https://ieeexplore.ieee.org/document/7155952](https://ieeexplore.ieee.org/document/7155952)
- **Yell at Your Robot (对标)**: [https://yell-at-your-robot.github.io/](https://yell-at-your-robot.github.io/)
- **SayCan**: [https://say-can.github.io/](https://say-can.github.io/)
- **Inner Monologue**: [https://innermonologue.github.io/](https://innermonologue.github.io/)
- **Code as Policies**: [https://code-as-policies.github.io/](https://code-as-policies.github.io/)
- **TidyBot**: [https://tidybot.cs.princeton.edu/](https://tidybot.cs.princeton.edu/)
- **TransIC (sim-to-real via correction)**: [https://transic-2024.github.io/](https://transic-2024.github.io/)
- **Octo generalist policy (引文 [1])**: [https://octo-models.github.io/](https://octo-models.github.io/)
- **GPT-4o technical report**: [https://arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774)
- **Particle filter 教科书资源 (rlabbe)**: [https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
- **iiwa_ros ROS stack**: [https://github.com/epfl-lasa/iiwa_ros](https://github.com/epfl-lasa/iiwa_ros)
