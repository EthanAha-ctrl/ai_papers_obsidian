---
source_pdf: Learning Physical Principles from Interaction Self-Evolving Planning via
  Test-Time Memory.pdf
paper_sha256: 1ad30868b2db902dafe0b5e7fd87fc49dff0eb3a62b2f91f5ed1a8dfe54998bd
processed_at: '2026-08-05T13:35:13-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PhysMem 人话版

## 这篇 paper 在解决什么问题?

想象你雇了一个很聪明的助手，他读过很多书，知道"摩擦力""重心""弹性"这些概念。但当你让他帮你把一个球推过障碍物到目标位置时，他搞砸了——因为他虽然知道摩擦力是什么，但不知道**这个球**在**这张桌子**上到底会滚多远。

这就是现在 VLM（vision-language model）当 robot 大脑的尴尬处境。这些模型在互联网上看了一堆图文，能侃侃而谈物理概念，但真正上手干活时，它们缺一样东西：**对这个具体世界的手感**。

你怎么获得手感?只能自己试。就像 Aristotle 说的那句话——"我们先通过做来学会做事"。骑自行车看再多书也没用，得摔几次。

## 作者的核心 idea

让 robot 在干活的过程中自己"做科学实验"。

人类科学家怎么发现规律?观察到现象 → 提出假设 → 设计实验验证 → 确认了就变成定律。PhysMem 让 VLM planner 走一遍同样的流程：

1. 干活，记录发生了什么（experience）
2. 发现"诶这个结果跟我预期的不一样"（surprise）
3. 把类似的经验攒到一起，总结出一个假设（hypothesis）
4. 特意去试几次，看看这个假设对不对（verification）
5. 对了就升级成规律（principle），错了就扔掉

这个 loop 一直在转，robot 越干越聪明。

## 为什么不能直接"照搬上次经验"?

这是 paper 里最 sharp 的发现。他们做了个对比实验：

- 直接检索最相似的过去经验来用：**23% 成功率**
- 先抽象成规律再用：**76% 成功率**

差了三倍多。为什么?

因为现实世界从来不重复。你上次推球用了中等速度成功了，这次桌子稍微换了一块、球磨损了一点，你照搬"用中等速度"就翻车了。**把经验当教条**，这是大忌。

Popper 早就说过这个——没有经过 falsification 的知识是教条的。你得 constantly 检查你的规律还成不成立。PhysMem 的设计核心就是 **verification before application**：假设得通过实验验证才能用，而且用了之后发现不对还能撤回。

## 三个记忆层，模拟人脑

physMem 把 memory 分三层，跟人脑的机制很像：

**第一层：Episodic memory（情景记忆）**
就是 raw 经历的流水账。每次操作都记下来：看到了什么、做了什么、成功还是失败。容量有限（3000 条），满了就按优先级删旧的。

**第二层：Working memory（工作记忆）**
放正在测试中的假设。比如"我觉得球离障碍物近的时候应该用低速"，这个假设还没被充分验证，先放这儿。每个假设带一个 confidence score，随着证据积累上下浮动。

**第三层：Long-term memory（长期记忆）**
放已经验证过的规律。这些是"毕业"了的 principles，可以直接指导决策。但它们也不是永久的——score 会按 Ebbinghaus 遗忘曲线慢慢衰减，很久没用到的就自然忘掉。

从第一层到第三层的过程叫 **consolidation**（巩固），每 50 个 episode 跑一次。

## 怎么判断"这个结果让我惊讶"?

这是个很聪明的设计。每次干完活，系统会算一个 **resonance score** $\rho$：

简单说就是——我手上有一套正在用的 principles，这次的结果跟它们预测的一致吗?如果完全一致（$\rho = 1$），那这次经历只是 reinforcement，不触发新学习。如果不一致（$\rho < 1$），说明有 surprise，这个 experience 会被标记为"值得深入分析"。

这跟人脑的机制很像——你做一件习以为常的事不会学到什么，但一旦结果出乎意料，你就会停下来想"怎么回事?"

paper 里展示的数据很漂亮：三个 task 的 resonance score 都从 0.2 左右（早期瞎搞）单调上升到 0.9 左右（后期有谱了），大约在第 5-6 个 episode 跨过 0.7 这个"理性"阈值。

## 假设怎么变成规律?

**第一步：聚类**
把 symbolic state 相似的 experiences 攒到一起。比如所有"在障碍物附近用高速推球"的经历归一类。

**第二步：生成假设**
用另一个 LLM（Qwen3-VL）分析这个 cluster，总结出 1-3 个假设。假设有三种类型：
- **AVOID**："别在 Y 的时候做 X"（来自失败的总结）
- **PREFER**："在 Y 的时候做 X"（来自成功的总结）
- **SEQUENCE**："先做 X 再做 Y"（时序约束）

**第三步：Action-level attribution**
这是个很 subtle 的设计。判断假设好不好，不看整个 episode 成不成功，看**假设涉及的那个具体 action** 成不成功。

为什么?一个 episode 有好多步，整个 episode 成功不代表每步都对，整个失败也不代表每步都错。Action-level attribution 把具体 action 的效果 isolate 出来，避开 confounding factors。这个思路跟 causal inference 里估计 treatment effect 是一回事。

**第四步：Promotion 或 Refutation**
- confidence ≥ 0.8 且至少 3 次支持证据且准确率 ≥ 85% → 升级成 principle
- confidence ≤ 0.3 且至少 2 次矛盾证据 → 扔掉

升级的时候还会做 **memory folding**——把支撑这个 principle 的原始 experiences 删掉，因为知识已经被"压缩"进 principle 了。这保持 context 不会无限膨胀。

## 三个实验任务

**任务一：Parts Organization（拼图 packing）**
把 6 个不规则形状的 3D 打印零件放到 3×10 的网格里，目标是占用最少格子。坑点在于——有些零件在特定旋转角度下可以 3D 重叠（一个的凸起卡进另一个的凹陷），但这个关系从外观看不出来，只能试。

作者搞了一套很精致的命名系统和 2×2 template coordinate：
```
+---+---+
| a | b |
+---+---+
| c | d |
+---+---+
```
不同形状在不同旋转下占据不同的 cells，而且即使占据相同 cells，internal geometry 也可能冲突或互补。比如两个 q-shape 的 [a,c] 区域重叠会撞车，但 white-q 的 [c] 和 red-q 的 [d] 可以互补共享 cells。

**任务二：Ball Navigation（推球过障碍）**
把 soccer ball 推过障碍场到目标，6 步内完成。有三个障碍：蓝色拱门（必须穿过中间的洞）、红色方块（绕过去）、紫色方块（千万别让球落到它上面，否则 robot 够不着）。

推球动作包括起始方向、结束方向、坐标、速度（low/medium/high）。球的滚动距离取决于速度、摩擦力、弹性——这些参数从外观看不出来，只能通过 trial-and-error 校准。

学到的典型 principle：
- "穿过拱门后必须用低速，否则球会滚到紫色方块上面卡住"
- "离拱门 100 像素以内必须用低速"
- "从危险区域（y>550）千万别直接推向目标，robot arm 会撞限位"

**任务三：Balanced Stacking（平衡叠石）**
把 5 个形状、材质、重量分布都不同的石头叠成稳定塔。命名系统是 {Surface}_{Shape}_{Size}_{Orientation}，比如 `3M_HEX_L_C` = 3M 表面 + 六边形 + 大号 + 紧凑型。

学到的典型 principle：
- "第一层必须选最大、摩擦系数最高、接触面积最大的石头"
- "V 朝向（竖立）的石头绝对不能当底座，接触面积太小"
- "TREE 形状的石头必须放最后，表面不规则没法支撑上面任何东西"
- "方形截面不能叠在水平横杆上——100% 倒塌"

## 最关键的实验结论

**结论一：principle abstraction 碾压 raw retrieval**
23% vs 76%，差三倍。直接照搬经验是 fragile 的，抽象成规律才 robust。

**结论二：test-time learning 放大 base capability，不能替代**
四个 VLM 测试：
- Gemini-3-Flash：53% → 76%（+23%）
- GPT-5.1：43% → 57%（+14%）
- Qwen3-VL：38% → 50%（+12%）
- Gemini-ER-1.5：29% → 34%（+5%）

越强的模型获益越大。这说明 memory 是 amplifier 不是 compensator。你得先有足够的 reasoning capability 才能 generate 和 verify 有意义的假设。

**结论三：prior knowledge 和 test-time adaptation 互补**
当 physics 相似（比如换一组石头但 friction 规律类似），prior knowledge 直接就管用。当 dynamics 变了（比如换了网球代替 soccer ball），prior 没用，必须 test-time adaptation 重新学。两者结合总是最好的。

**结论四：forgetting 是必要的**
不遗忘的话，简单任务能涨 2%，但复杂任务反而掉 3%，而且 token 消耗暴涨 3-5 倍。noise 累积会干扰决策。Ebbinghaus 遗忘曲线让 outdated principles 自然淡出，这是个 favorable trade-off。

**结论五：难任务主要靠学"别做什么"**
Hard task 上 68% 的 learned principles 是 AVOID 类型。即使 VLM 没法完全 solve 任务，至少能学会避免重复犯错。

## 我的 intuition 总结

这篇 paper 的核心 insight 可以用一句话概括：**科学方法本身就是一个 learning algorithm**。

观察 → 假设 → 实验 → 验证 → 理论，这个 loop 不只是人类科学家的专属。把它 instantiate 在 VLM planner 上，robot 就能在 test time 通过自己的 interaction 学会 physical principles。

几个关键的 design choice 构成了完整的直觉：

1. **Surprise 驱动学习**：只在不一致的时候才触发 hypothesis generation，把资源集中在 novel situations
2. **Action-level attribution**：把具体 action 的效果 isolate 出来，避开 episode-level 的 confounding
3. **Verification before promotion**：假设必须通过 targeted interaction 验证才能升级成 principle，避免 dogmatism
4. **Three-tier memory with folding**：raw experience → hypothesis → principle 的渐进式 consolidation，配合遗忘机制保持 tractable
5. **Text-based principles**：human-readable、可 inspect、可 edit、可 transfer，比 implicit parameter update 透明得多

最大的 limitation 是 text-based principles 表达不了 continuous dynamics。未来如果能用 visual chain-of-thought 或 latent world model 来 represent principles，可能突破这个瓶颈。想象一下——principles 不只是文字规则，而是 latent space 里的 dynamics prediction，可以 imagine outcome 然后 verify。这可能是下一步。

参考链接：
- Popper falsification: https://plato.stanford.edu/entries/popper/
- Ebbinghaus forgetting curve: https://en.wikipedia.org/wiki/Forgetting_curve
- Episodic control: https://arxiv.org/abs/1606.04460
- Options framework: https://www.sciencedirect.com/science/article/pii/S0004370299000369
- CoT-VLA visual reasoning: https://arxiv.org/abs/2504.00268
- MemoryVLA: https://arxiv.org/abs/2508.19236

---

# PhysMem: Test-Time Physical Principle Learning via Scientific Memory Loop

## 1. Motivation - 为什么VLM planners在物理任务上失败?

VLM (vision-language model) 能够描述 friction、balance、momentum 这些 physical concepts,这个能力来自 web-scale pretraining。当把 VLM 当作 robot planner 部署时,会出现一个 deep gap:**declarative knowledge** 与 **physical grounding** 之间的鸿沟。一个 VLM 知道 friction 是什么,却无法预测一个 specific 的 ball 在一个 specific 的 surface 上会滚多远;知道 stability 概念,却无法判断哪一块 irregular stone 适合做 foundation。

这种 gap 在 planning 中会 compound:一个关于 contact 或 dynamics 的 misjudgment 会 invalidate 整个 action sequence。论文用 Aristotle 的话开头——"For the things we have to learn before we can do them, we learn by doing them"——直接点明了 core thesis:**physical understanding 必须通过 interaction 获得**,不能仅靠 pretraining。

论文聚焦三个 task,每一个都精心设计让 correct strategy 无法从 vision 单独 infer:

1. **Parts Organization**: 学习 irregular shapes 之间的 spatial relationships,允许 3D overlap,但这个 overlap 关系只有 placement attempt 之后才显现
2. **Ball Navigation**: soccer ball 在 obstacle course 中的 contact dynamics 不可从 appearance 预测;friction 和 elasticity 跨 workspace 变化
3. **Balanced Stacking**: stone 的 mass distribution 和 surface friction 不可见,只有 contact 之后才 reveal

## 2. 核心洞察 - 为什么直接 retrieval 失败,而 principled abstraction 成功?

这是 paper 最 sharp 的实验结论之一。在 controlled brick insertion benchmark 上:

| Method | Success Rate |
|--------|--------------|
| Direct experience retrieval | 23% |
| Principled abstraction (PhysMem) | 76% |

直接 episodic replay 失败的 root cause:**embodied situations never repeat exactly**。一个小的 friction 变化或 object shape 变化就能把 useful heuristic 变成 repeated error。当 planner 把 past experience 当作 fixed rule 应用,就陷入了 "dogmatism" problem——这正是 Popper 在 *The Logic of Scientific Discovery* (1959) 中批判的:未经 falsification 的知识是教条的。

Principles 通过 abstraction over specific instances 来 generalize,但 abstraction 本身也有 limits——irrelevant principles 会 active hurt performance。所以 PhysMem 的核心 design choice 是 **verification before application**:hypotheses 必须通过 targeted interaction 验证才能 promote 为 principles,且 principles 可以被新证据 refuted。

参考链接:
- Popper's falsification: https://plato.stanford.edu/entries/popper/
- Episodic control (Blundell et al. 2016): https://arxiv.org/abs/1606.04460
- Neural episodic control (Pritzel et al. 2017): https://arxiv.org/abs/1703.01948

## 3. Problem Formulation

论文采用 Sutton 的 **options framework** (Sutton, Precup, Singh 1999),把 physical manipulation 形式化为 sequential decision problem。

**Option 定义**: $\omega = \langle \mathcal{I}, \pi, \beta \rangle$
- $\mathcal{I} \subseteq \mathcal{S}$: initiation set,在哪些 state 下这个 option 可用
- $\pi: \mathcal{S} \times \mathcal{A} \to [0,1]$: intra-option policy
- $\beta: \mathcal{S} \to [0,1]$: termination condition

**High-level VLM policy**:
$$\omega_t = \pi_\theta^H(o_t, \tau, \mathcal{P}_t) \tag{1}$$

变量含义:
- $\omega_t$: 第 $t$ 步选择的 option (temporally-extended action)
- $\pi_\theta^H$: high-level VLM-based policy,参数为 $\theta$ (注意:这个 $\theta$ 在整个 test-time learning 过程中 **不变**)
- $o_t \in \mathcal{O}$: 当前 observation (visual)
- $\tau$: task description (language)
- $\mathcal{P}_t \subseteq \mathcal{P}$: 在时间 $t$ 从 memory 中 retrieved 的 active principles 子集

Low-level policy $\pi^L$ 执行 option 直到 termination。

**Learning objective**:
$$\mathbb{E}\left[\sum_{t=0}^{T} r_t \mid \pi_\theta^H(\cdot, \cdot, \mathcal{P}^*)\right] > \mathbb{E}\left[\sum_{t=0}^{T} r_t \mid \pi_\theta^H(\cdot, \cdot, \emptyset)\right] \tag{2}$$

变量含义:
- $r_t$: 第 $t$ 步的 reward
- $T$: episode horizon
- $\mathcal{P}^*$: 学到的 optimal principle set
- 右侧:不带任何 principles 的 baseline
- 关键约束:$\mathcal{P}^*$ 必须 **在 test time 学到**,不修改 VLM parameters $\theta$

这个 formulation 的妙处在于把 "learning" 重新定义为 "principle set 的演化",而非 parameter update。这避开了 on-device fine-tuning 的所有问题 (catastrophic forgetting, compute cost, gradient instability),同时保持 knowledge 的 interpretability。

参考链接:
- Options framework paper: https://www.sciencedirect.com/science/article/pii/S0004370299000369

## 4. System Architecture - Scientific Memory Loop

整个 PhysMem 由三个组件构成:

1. **VLM-based Planner**: 接收 observations + retrieved principles,生成 high-level decisions
2. **Three-tier Memory System**: episodic → working → long-term
3. **Executor**: low-level policy (motion planner / VLA / 其他 controller)

核心创新是 **scientific memory loop**,灵感来自科学方法 (observation → hypothesis → experiment → verification → theory)。这四个 phase 是:

### Phase 1: Experience Collection with Resonance Checking

每个 experience 存储为:
$$e = (o, \omega, r, c, \mathbf{s})$$
- $o$: observation
- $\omega$: selected option
- $r \in \{0, 1\}$: outcome
- $c$: context (task description, subtask)
- $\mathbf{s}$: symbolic state (discrete features: action type, object properties)

**Resonance score** 是关键机制:
$$\rho(e, \mathcal{P}_{\text{active}}) = \frac{|\{p \in \mathcal{P}_{\text{active}} : \text{consistent}(e, p)\}|}{|\mathcal{P}_{\text{active}}|} \tag{3}$$

变量含义:
- $\rho$: resonance score ∈ [0, 1]
- $e$: 新 experience
- $\mathcal{P}_{\text{active}}$: decision-making 时 active 的 principles 子集
- $\text{consistent}(e, p)$: 谓词,检查 experience outcome 是否与 principle $p$ 的 prediction 一致
- 分子: 与 experience 一致的 principles 数量
- 分母: active principles 总数

**关键逻辑**:
- 当 $\rho < 1$: experience 是 "surprising",触发 consolidation (hypothesis generation)
- 当 $\rho = 1$: experience reinforces existing principles,不触发新 hypothesis

这是 **surprise-driven filtering**,把 learning resource 集中在 novel situations 上。这个思想非常类似 reinforcement learning 中的 prediction error 驱动 learning,也呼应了 neuroscience 中 dopamine neurons 对 surprise 的响应 (Schultz 1997 的 reward prediction error theory)。

### Phase 2: Hypothesis Generation

Experiences 周期性地 (每 50 episodes) 按 symbolic similarity 聚类。对每个足够大的 cluster $\mathcal{C}_k$ ($|\mathcal{C}_k| \geq n_{\min} = 2$):

$$\mathcal{H}_k = f_\phi(\mathcal{C}_k, \mathcal{P}, \mathcal{H}_{\text{existing}}) \tag{4}$$

变量含义:
- $\mathcal{H}_k$: 第 $k$ 个 cluster 生成的 hypotheses 集合
- $f_\phi$: reflection model (real-world 用 Qwen3-VL,simulation 用 LLM),参数 $\phi$
- $\mathcal{C}_k$: 第 $k$ 个 experience cluster
- $\mathcal{P}$: 当前 principles (避免 duplication)
- $\mathcal{H}_{\text{existing}}$: 当前 hypotheses (避免 duplication)

每个 hypothesis $h \in \mathcal{H}_k$ 有 typed form:
- **AVOID**: "Don't do X when Y" (来自 failures)
- **PREFER**: "Do X when Y" (来自 successes)
- **SEQUENCE**: "Do X before Y" (temporal constraints)

这个 typed form 是 paper 的一个重要 design choice,让 hypothesis 直接 actionable,且让 VLM 容易在 prompt 中理解。

### Phase 3: Action-Level Attribution

这是 paper 最 subtle 的设计之一。Hypotheses 不用 episode-level success 来 judge,而用 **action-level outcome**。对于关于 action type $a^*$ 的 hypothesis $h$:

$$\text{conf}(h) \leftarrow \text{conf}(h) + \alpha \cdot \frac{|\{e \in \mathcal{E}_h : a_e = a^*, r_e = 1\}|}{|\{e \in \mathcal{E}_h : a_e = a^*\}|} \tag{5}$$

变量含义:
- $\text{conf}(h)$: hypothesis $h$ 的 confidence
- $\alpha$: learning rate
- $\mathcal{E}_h$: 与 hypothesis $h$ 相关的 experiences 集合
- $a_e$: experience $e$ 中执行的 action
- $a^*$: hypothesis $h$ 所关于的 specific action type
- $r_e = 1$: experience $e$ 成功
- 分子: 在 $\mathcal{E}_h$ 中执行了 $a^*$ 且成功的 experiences 数
- 分母: 在 $\mathcal{E}_h$ 中执行了 $a^*$ 的所有 experiences 数

**为什么 action-level 而非 episode-level?** 一个 episode 包含多个 actions,episode 失败不代表每个 action 都错;episode 成功也不代表每个 action 都对。Action-level attribution 把 specific action 的效果 isolate 出来,避开 confounding factors。这是一个非常类似 causal inference 中 treatment effect estimation 的思想。

具体的 confidence update rule (Eq.7):
$$\text{conf}(h) \leftarrow \begin{cases} \min(1.0, \text{conf}(h) + 0.1 \cdot r_a) & \text{if } r_a \geq 0.7 \\ \max(0.0, \text{conf}(h) - 0.1 \cdot (1 - r_a)) & \text{if } r_a \leq 0.3 \\ \text{conf}(h) \pm 0.02 & \text{otherwise} \end{cases} \tag{7}$$

变量含义:
- $r_a$: action-level success rate for actions matching the hypothesis
- 当 $r_a \geq 0.7$: 强支持,confidence 上调 (上限 1.0)
- 当 $r_a \leq 0.3$: 强反驳,confidence 下调 (下限 0.0)
- 中间地带: 微调 ±0.02

### Phase 4: Verification and Principle Promotion

**Promotion criteria**:
- $\text{conf}(h) \geq \tau_p = 0.8$
- $|\mathcal{E}_{\text{support}}| \geq 3$
- Accuracy $\geq 85\%$

**Refutation criteria**:
- $\text{conf}(h) \leq \tau_r = 0.3$
- $|\mathcal{E}_{\text{contradict}}| \geq 2$

**Memory folding** (Eq.6):
$$\mathcal{P} \leftarrow \mathcal{P} \cup \{h\}, \quad \mathcal{E} \leftarrow \mathcal{E} \setminus \mathcal{E}_{\text{folded}} \tag{6}$$

当 hypothesis 被 promote 为 principle,source experiences 被 "folded" 进 principle,从 episodic memory 中移除。这是 **compression** 机制,保持 context tractable over extended deployment。这个思想类似数据库的 materialized view 或者 knowledge compilation in classical AI。

**Principle decay** (Ebbinghaus forgetting curve):
$$\text{score}_{t+1} = \text{score}_t \cdot \gamma, \quad \gamma = 0.995 \tag{8}$$

变量含义:
- $\text{score}_t$: principle 在 episode $t$ 的 importance score
- $\gamma$: decay factor
- 大约 138 episodes 后 retention 降到 50%

这个 Ebbinghaus forgetting curve 来自 experimental psychology (Ebbinghaus 1885 的 memory experiments),让 outdated principles 自然 fade out。这避免了 principle set 无限膨胀的问题。

参考链接:
- Ebbinghaus forgetting curve: https://en.wikipedia.org/wiki/Forgetting_curve
- Algorithm distillation (Laskin et al. 2023): https://arxiv.org/abs/2210.14215

## 5. Memory Architecture - Three-Tier System

PhysMem 的 memory 组织灵感明显来自 human memory 的 Atkinson-Shiffrin model (1968) 和后续的 working memory theories (Baddeley 1992):

| Tier | 存储内容 | 特征 |
|------|---------|------|
| **Episodic memory** | Raw experiences with symbolic state | Capacity $N_{\max} = 3000$,高效 filtering |
| **Working memory** | Unverified hypotheses under test | 带 confidence scores,supporting/contradicting evidence |
| **Long-term memory** | Verified principles | Importance scores decay over time ($\gamma = 0.995$) |

**Retrieval mechanism**: symbolic filtering (matching action type 和 object properties) → semantic ranking → top-k principles + active hypotheses 注入 VLM prompt。

**Garbage collection** (当 episodic memory 达到 capacity):
1. 移除 folded experiences older than TTL = 100 episodes
2. 按 priority 移除: folded > old failures > old successes

这个 priority 顺序很有意思:folded experiences 已经被 principle 吸收,优先删;old failures 价值低 (因为已经被转化成 AVOID principles);old successes 保留最久。

参考链接:
- Atkinson-Shiffrin memory model: https://en.wikipedia.org/wiki/Atkinson%E2%80%93Shiffrin_memory_model
- MemoNav working memory model: https://arxiv.org/abs/2508.19236

## 6. Memory Injection into VLM Prompts

Top-k principles 和 active hypotheses 以结构化格式注入 VLM prompt:

```
## Learned Principles (Apply these!)
1. [92%] [SEQUENCE] Always place the largest stone as the base.
2. [88%] [AVOID] High-speed pushes near obstacles cause rebounds.
3. [82%] [PREFER] Place L-shaped parts in corners facing inward.

## Hypotheses (Consider but verify)
1. [TESTING] Rough stones grip smooth stones better than vice versa.
2. [TESTING] Medium speed is optimal for most ball navigation.
```

Confidence display tiers:
- HIGH (≥ 85%): strong evidence, should be followed
- MEDIUM (60-84%): moderate evidence, consider carefully
- LOW (< 60%): weak evidence, use with caution

这个 injection 方式让 VLM 在 reasoning 时可以 **看到** knowledge 的 confidence,这比 implicit embedding 更 interpretable。也呼应了 chain-of-thought 的思想——让 model 显式地 reason about 知识的可靠性。

## 7. Experimental Setup - 三个 Real-World Tasks

### Task 1: Parts Organization

- **Setup**: 5 个 3D-printed parts (red-L, white-q, red-q, white-U, black-U, black-I) 放入 3×10 grid (30 cells)
- **Objective**: 最小化 total cells occupied
- **Challenge**: parts 允许 3D overlap 当 aligned correctly,但这个 spatial relationship 只通过 placement attempts 显现

**Part naming convention**: {color}-{shape}
- L-shape (3 cells): occupies [a, c, d]
- q-shape (3 cells): occupies [a, c, d] - 不同 internal geometry
- I-shape (2 cells): occupies [a, c]
- U-shape (4 cells): occupies [a, b, c, d]

**Template-based 2×2 coordinate system**:
```
+---+---+
| a | b |  <- top row
+---+---+
| c | d |  <- bottom row
+---+---+
```

**Rotation transformations** (counterclockwise):
- 90°: a→c, b→a, c→d, d→b
- 180°: a→d, b→c, c→b, d→a
- 270°: a→b, b→d, c→a, d→c

**Internal biases** (关键 - 决定 overlap 兼容性):
- black-U / white-U: physical mass biased toward [b, d];[a, c] 是 U opening 的 empty space
- white-q: left edge aligns with left boundary of [a, c]
- red-q: right edge aligns with right boundary of [a, c]
- black-I: top/right edges align with top-right of [a, c]

**Scoring** (Eq.9):
$$\text{Score} = 5 - 5 \cdot \text{clip}\left(\frac{s - s_{\min}}{s_{\max} - s_{\min}}, 0, 1\right) \tag{9}$$

变量含义:
- $s$: actual grid occupancy
- $s_{\min}$: theoretical minimum occupancy (optimal packing)
- $s_{\max}$: maximum possible occupancy
- 5 分 = optimal packing,0 分 = worst-case packing

### Task 2: Ball Navigation

- **Setup**: soccer ball 通过 obstacle course 到达 target,6 steps 内
- **Obstacles**:
  - OBS1 (Blue archway): y∈[200,550], x∈[100,300],bridge hole at y~400-480
  - OBS2 (Red): solid blocker
  - OBS3 (Purple): DANGER zone,如果 ball 落到 top (y~600-700) 就 stuck
- **Action format**: `push(start_dir, end_dir, start_y, start_x, end_y, end_x, speed)`
- **Speed tiers**: low (150mm/s), medium (300mm/s), high (450mm/s)
- **Coordinate system**: normalized 0-1000, (0,0) top-left, (1000,1000) bottom-right

**Scoring**:
- +1: ball moves
- +3: passes through OBS1 archway
- +5: reaches target
- +2×(6-i): early completion bonus at step i
- -2: collision or invalid plan

### Task 3: Balanced Stacking

- **Setup**: 5 个 balance stones 堆成 stable tower
- **Stone naming**: {Surface}_{Shape}_{Size}_{Orientation}
  - Surface: 3M > BLK > WHT > WOOD > PAINT (friction high to low)
  - Shape: SQR, HEX, DMND, PENT, OVAL, TREE
  - Size: L, M, S
  - Orientation: H (horizontal), V (vertical), C (compact)

**Critical incompatibility**: SQR on DMND_H 是 FORBIDDEN - 100% collapse rate

**Scoring**:
- +i: placing stone on layer i
- -2j: j stones fall during placement

## 8. Key Experimental Results

### 8.1 Resonance Score Evolution

Resonance score $\rho$ 是 paper 提出的 reasoning quality metric,比 raw success rate 更 informative。一个 planner 可能通过 conservative strategies 达到 high success,但 resonance 要求 internal model 真的 reflect underlying physics。

**关键观察**:
- 所有 3 个 task 都展现一致 pattern:resonance 从 $\rho \approx 0.2$ (early episodes) 上升到 $\rho = 0.9$ (by episode 10)
- 大约 episode 5-6 跨过 $\rho = 0.7$ threshold
- Rational regime ($\rho > 0.7$) 的 episodes achieve 2.3× higher scores than episodes 1-3
- 单调上升的 $\rho$ 表明 PhysMem 积累 verified physical understanding,而非 overfit 到 narrow solutions

### 8.2 Test-Time Evolution

不同 experience utilization levels (0%, 25%, 50%, 100%) 的对比:

| Task | No Memory (0%) | Full Memory (100%) |
|------|----------------|---------------------|
| Parts Organization | -1 | 9.7 |
| Ball Navigation | 0.7 | 14.7 |
| Balanced Stacking | ~0 | 12.3 |

**Task complexity 决定 experience 需求**:
- Ball Navigation (复杂 dynamics) benefits from full experience (50% vs 100%: 11.0 vs 14.7)
- Balanced Stacking 显示 diminishing returns (50% nearly matches 100%)

### 8.3 Memory Transfer to OOD Scenarios

| Condition | Parts Org. Score | Parts Succ. | Ball Nav. Score | Ball Succ. | Stack Score | Stack Succ. |
|------------|------------------|-------------|-----------------|------------|-------------|-------------|
| No Prior, No Adapt | -0.6 | 0/10 | 1.6 | 1/10 | 6.7 | 4/10 |
| No Prior, Adapt | 3.3 | 1/10 | 5.5 | 2/10 | 8.3 | 7/10 |
| Prior, No Adapt | 6.9 | 3/10 | 2.9 | 1/10 | 9.2 | 8/10 |
| Prior, Adapt (Full) | 8.3 | 4/10 | 7.1 | 4/10 | 12.3 | 9/10 |

**关键洞察**:
- **Balanced Stacking**: prior knowledge alone achieves 80% success (stability principles transfer well)
- **Parts Organization**: prior knowledge improves -0.6→6.9
- **Ball Navigation (new ball types)**: prior knowledge matches zero-shot (因为 dynamics 不同),adding adaptation 提升 success 从 10% 到 40%

这印证了 paper 的 thesis: **prior knowledge 和 test-time adaptation 是 complementary 的**。当 physics 相似时 prior 足够;当 dynamics 改变时,adaptation 才是 essential。

### 8.4 Scaling Across VLMs

| VLM | Baseline | +PhysMem | Δ |
|-----|----------|----------|---|
| Gemini-3-Flash | 53% | 76% | +23% |
| GPT-5.1 | 43% | 57% | +14% |
| Qwen3-VL-235B | 38% | 50% | +12% |
| Gemini-ER-1.5 | 29% | 34% | +5% |

**关键发现**: test-time learning benefits **scale with VLM capability**。Gemini-3-Flash 提升 +23%,而 Gemini-ER-1.5 只 +5%。

**Interpretation**: memory **amplifies** existing capabilities,而非 compensates for fundamental limitations。一个 VLM 必须有足够 understanding 才能 generate 和 verify meaningful hypotheses。这和 scaling laws 的精神一致——base capability 是 multiplier,test-time learning 是 amplifier。

参考链接:
- VLM4VLA review: https://arxiv.org/abs/2601.03309
- Towards generalist VLA: https://arxiv.org/abs/2412.14058

### 8.5 Principle Scaling

500 episodes on Gemini-3-Flash,测量 performance 随 principle count (1→128) 变化:

| Difficulty | Pattern | Saturation |
|------------|---------|------------|
| Easy | 83%→89% (saturate quickly) | ~16 principles |
| Medium | 55%→67% (rapid learning 2-8 principles) → stabilize 76% | ~64 principles |
| Hard | +11% improvement | >128 principles |

**关键发现**: Hard tasks 上 68% of learned principles 是 AVOID constraints (vs. 41% on medium)。这表明 **learning what not to do** 对难任务特别有效,即使 VLM 无法 fully solve task。

## 9. Ablation Studies - 完整分析

### 9.1 Memory Architecture Ablations

| Config | Easy | Medium | Hard | Tokens |
|--------|------|--------|------|--------|
| PhysMem (Full) | 89% | 76% | 39% | 1.0× |
| w/o Episodic Memory | 54% (-35) | 37% (-39) | 14% (-25) | 0.3× |
| w/o Working Memory | 84% (-5) | 69% (-7) | 28% (-11) | 0.85× |
| w/o Long-term Memory | 81% (-8) | 64% (-12) | 26% (-13) | 1.15× |

**分析**:
- **Episodic memory** 是 foundation (去掉了 -35 到 -39),没有 raw experience 就没有 learning source
- **Working memory** 让 hypothesis exploration 成为可能,importance scales with difficulty (-5 → -11)
- **Long-term memory** 让 verified hypotheses 持久化,没有它就反复 re-learn (-8 → -13)

### 9.2 Mechanism Ablations

| Config | Easy | Medium | Hard | Tokens |
|--------|------|--------|------|--------|
| PhysMem (Full) | 89% | 76% | 39% | 1.0× |
| w/o Resonance Filtering | 81% (-8) | 58% (-18) | 21% (-18) | 1.3× |
| w/o Verification | 85% (-4) | 64% (-12) | 27% (-12) | 0.85× |
| w/o Forgetting | 91% (+2) | 78% (+2) | 36% (-3) | 3.4× |
| w/o Folding | 88% (-1) | 74% (-2) | 37% (-2) | 2.1× |

**关键洞察**:

1. **Resonance filtering**: importance scales with difficulty (-8 → -18)。没有它,planner 会 retrieve 学自不同 context 的 principles,产生 conflicting guidance。Token overhead 也增加 (1.3× on medium, 1.45× on hard),因为处理更多 irrelevant principles。

2. **Verification**: 保证 hypothesis quality 在 promotion 之前。Easy tasks 影响小 (-4),因为 simple hypotheses rarely incorrect;hard tasks 影响大 (-12),因为 complex dependency hypotheses 最 error-prone。

3. **Forgetting**: 准确率-效率 trade-off。Simple tasks marginally gain (+2%) 但 hard tasks degrade (-3%)。**Token overhead 巨大** (1.8× easy, 3.4× medium, 4.8× hard)。这表明 noise accumulation 在 complex tasks 上 actively hurts decision quality,forgetting 提供了 favorable balance。

4. **Folding**: 主要影响 efficiency,accuracy 影响小 (-1 到 -2)。它 compresses redundant experiences 同时 preserve essential patterns。

### 9.3 Simplified Baselines

| Config | Easy | Medium | Hard |
|--------|------|--------|------|
| Direct Retrieval | 48% (-41) | 23% (-53) | 8% (-31) |
| Only Episodic | 52% (-37) | 28% (-48) | 10% (-29) |
| Only Principles | 67% (-22) | 51% (-25) | 21% (-18) |

**Direct Retrieval** 失败最严重 (-41 到 -53),证实了 raw episodic replay 的 fragility。**Only Principles** 表现中等,但无法 adapt 到 novel situations (-18 到 -25 gap from full system)。

## 10. Discussion - 关键 Design Insights

### 10.1 Why Abstraction Beats Retrieval?

Raw episodic replay 失败因为 situations never repeat exactly。这呼应了 episodic memory research (Blundell et al. 2016, Pritzel et al. 2017) 的发现。Principles 通过 abstraction over specific instances 来 generalize。

但 abstraction 也有 limits。Transfer experiments 表明 irrelevant principles 会 active hurt performance。PhysMem 通过 verification 避免 "dogmatism":**检查 principles 是否还 hold**,这个 ability to update beliefs 区分了 learning 和 memorization。

### 10.2 Observation Space Limitations

当前 experiments 用 visual observations + outcome feedback。Physical world 提供更丰富 signals:
- **Tactile**: VLMs 含有 latent physical knowledge,tactile grounding 可以 activate (Huang et al. 2025)
- **Audio**: 揭示 material properties inaccessible to vision (Liu & Chen 2024, SonicSense)
- **Active perception**: targeted exploration strategies (Sripada et al. 2024)

PhysMem 的 architecture 自然 extends 到这些 inputs:principles 可以 emerge from tactile failure patterns 或 acoustic contact signatures。

参考链接:
- Tactile-VLA: https://arxiv.org/abs/2507.09160
- SonicSense: https://arxiv.org/abs/2409.13706
- AP-VLM: https://arxiv.org/abs/2409.17641

### 10.3 Reasoning Representation Limits

Text-based principles 在 discrete rules 上 excel,但 struggle with continuous dynamics (trajectories, force profiles)。Language 无法完全 capture physical interaction 的连续本质。

**Future directions**:
- **Visual chain-of-thought** (Zhao et al. 2025, CoT-VLA): 17% improvements over text-based approaches 通过 predicting future frames
- **Continuous visual tokens** (Qin et al. 2025): reasoning through continuous visual tokens
- **Latent world models** (Bi et al. 2025, Motus): bypass language entirely for physics prediction
- **Genie 3** (Ball et al. 2025): world models that imagine outcomes,naturally complement scientific loop

参考链接:
- CoT-VLA: https://arxiv.org/abs/2504.00268
- Chain-of-visual-thought: https://arxiv.org/abs/2511.19418
- Motus: https://arxiv.org/abs/2512.13030
- Genie 3: https://arxiv.org/abs/2507.01861

## 11. Limitations

1. **High-level planning only**: 当前 work 聚焦 high-level planning,integrating learned principles into VLA execution 仍然 open。当前 VLAs struggle 当 task descriptions 包含 novel physical constraints。

2. **Environment reset**: objects 落下或 parts 断裂时需要 human intervention。Autonomous recovery 和 repair 才能实现 truly lifelong learning (Liu et al. 2021)。

3. **Text-based principles 的表达力**: 无法完全 capture continuous physics,需要 visual/latent representations。

参考链接:
- Lifelong learning for mobile robot navigation: https://arxiv.org/abs/2103.06412
- π0 VLA flow model: https://arxiv.org/abs/2410.24164

## 12. Connection to Broader Research Landscape

### 12.1 与 Memory-Augmented RL 的关系

PhysMem 与 model-free episodic control (Blundell 2016) 和 neural episodic control (Pritzel 2017) 在精神上相似,但有关键区别:
- 不存储 raw state-action pairs,而存储 abstracted principles
- 引入 verification step 避免教条应用
- 三层 memory hierarchy 而非 flat episodic buffer

### 12.2 与 Test-Time Training 的关系

Test-time training methods (Sun et al. 2020) 通过 self-supervision 在 unlabeled test data 上 update models。PhysMem 的区别:
- Implicit policy adjustment → **explicit principle learning**
- 不可 inspect 的 parameter updates → **human-readable hypotheses**
- 无法 transfer 的 learned weights → **principles 可 inspect、edit、transfer**

### 12.3 与 Reflection-Based Methods 的关系

Reflexion (Shinn et al. 2023) 和 Self-Refine (Madaan et al. 2023) 用 LLMs 从 failures 中学习。PhysMem 的区别:
- Reflection-based 只从 failures 学习 → PhysMem 从 **successes 和 failures** 都学习
- Blind retrieval 应用 experience → PhysMem **verifies** before application
- 没有显式 memory consolidation → PhysMem 有 three-tier hierarchy 和 folding

### 12.4 与 World Models 的关系

World models (DreamerV3, Genie, Cosmos) 学习想象 future scenarios,enabling planning without real-world interaction。PhysMem 的 verification loop 可以视为 **explicit world model testing**:principles 是 model 的 predictions,targeted interactions 是 experiments。

未来可能 **integrate latent world models** (Motus, Genie 3) 来提供 richer hypothesis testing。World model 可以 imagine outcome,PhysMem 验证 imagination 是否与 reality 一致。

参考链接:
- DreamerV3: https://arxiv.org/abs/2301.04104
- Cosmos world foundation model: https://arxiv.org/abs/2501.03575
- MemoryVLA: https://arxiv.org/abs/2508.19236
- Memer scaling memory: https://arxiv.org/abs/2510.20328

## 13. 总结 - Build Intuition

PhysMem 的核心 contribution 可以浓缩为几个 intuition:

1. **Physical understanding requires interaction**: VLM 的 declarative knowledge 无法 substitute for grounded physical experience。这是 embodiment 的核心 thesis。

2. **Abstraction generalizes, raw retrieval fails**: 因为 embodied situations never repeat exactly,raw episodic replay 在 small state 变化下崩溃。Principles 通过 abstraction 来 generalize。

3. **Verification avoids dogmatism**: Knowledge 不验证就变成 dogma。Scientific method 的核心是 falsification——principles 必须能被新证据 refuted。这是 Popper 哲学的工程化。

4. **Surprise drives learning**: Resonance score $\rho < 1$ 是 surprise signal,把 learning resource 集中在 novel situations 上。这呼应了 neuroscience 的 prediction error theory。

5. **Memory hierarchy enables consolidation**: Three-tier system (episodic → working → long-term) 让 raw experiences 逐渐 refine 为 verified principles,每个阶段有不同的 confidence threshold。

6. **Forgetting balances accuracy and efficiency**: Ebbinghaus decay 让 outdated principles 自然 fade。没有 forgetting,noise accumulation 在 complex tasks 上 hurts performance,且 token overhead 巨大 (3-5×)。

7. **Test-time learning amplifies base capability**: 不能 compensate for fundamental limitations。一个 VLM 必须有足够 understanding 才能 generate 和 verify meaningful hypotheses。

8. **Action-level attribution isolates causes**: Episode-level success/failure 包含太多 confounding factors。Action-level attribution 把 specific action 的效果 isolate 出来,类似 causal inference。

9. **Principles are interpretable and transferable**: Text-based principles 可以 human-inspect、edit、transfer to new settings。这比 implicit parameter updates 更 transparent。

10. **Knowledge compaction via memory folding**: 当 hypothesis 被 promote,source experiences 被 folded 进 principle。这保持 context tractable over extended deployment,类似 knowledge compilation。

这篇 paper 的 deep insight 是: **scientific method 本身就是一个 learning algorithm**。Observation → hypothesis → experiment → verification → theory 这个 loop 不只是人类科学家的方法,它可以 instantiate 在 VLM planner 上,让 robot 在 test time 通过自己的 interaction 学习 physical principles。这个 framing 把 epistemology (Popper)、cognitive science (memory models)、robotics (manipulation) 和 ML (test-time learning) 优雅地 unify 在一个 framework 中。

未来最有 promise 的方向是 **multi-modal principle representations**——让 principles 不只是 text,而是 visual predictions、tactile patterns、latent dynamics。这会突破 text-based principles 在 continuous physics 上的表达力限制,让 robot 能够学习更 fine-grained 的 physical intuition。World models + scientific memory loop 的结合可能是一个 generative direction。
