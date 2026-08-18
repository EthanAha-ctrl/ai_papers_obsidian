---
source_pdf: Anticipation-VLA Solving Long-Horizon Embodied Tasks via.pdf
paper_sha256: db96c4990a404f0f4e85dbd5970cdf631bf0cff7ff0a36fe506513dc7e2840c1
processed_at: '2026-08-18T00:58:47-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

Andrej，我换个讲法，假设我们坐咖啡馆聊这篇 paper。

## 故事从哪开始

假设你让机器人去 "泡杯咖啡"。这事对人来说很简单——站起来、走到厨房、拿杯子、放咖啡粉、倒水、端回来。但你直接给机器人这个指令，它大概率会搞砸。Why？因为这个任务太长了，十几步下来，每一步都有小误差，误差越积越多，最后机器人可能拿着空杯子在原地转圈。

之前大家想了个办法：**先把任务拆成几步**。比如先用大模型说 "第一步拿杯子，第二步放咖啡粉..."，然后让机器人一步步执行。但这有几个问题：

1. **拆得太粗或太细都不好**。"拿杯子" 这种指令可能包含 "走到柜子前、打开柜子、伸手、抓住、收回" 很多小动作，机器人执行 "拿杯子" 时如果卡住了，没人告诉它该怎么办
2. **拆完就不变了**。任务执行到一半发现实际情况跟预期不一样，原来拆好的步骤就不合适了，但系统没能力调整
3. **拆几层也是拍脑袋**。有的任务简单拆两层够了，有的复杂任务需要拆四五层，但系统不知道该拆几层

Anticipation-VLA 想解决的就是：**怎么让机器人自己根据情况动态决定要不要继续拆任务，拆到什么粒度**。

## 核心思路

想象你在爬一座山。山顶是你的终极目标。直接盯着山顶走很容易迷路。正常人的做法是：先看哪个方向大概对，往那个方向走到一个 landmark，到了 landmark 再看下一个 landmark 在哪，一步步走。如果走到某处发现路断了，你会回头看是不是上一个 landmark 选错了，可能要退回去重新选路。

Anticipation-VLA 就是模仿这个过程：

**Goal Stack（目标栈）**：想象一个栈结构，栈底是终极目标 "泡咖啡"，栈顶是当前最细粒度的子目标 "伸手抓住杯子把手"。机器人执行时总是盯住栈顶走。走到了就把这个子目标弹出栈，露出下面一个稍粗的目标。走不到就再往下拆一层，生成更细的子目标压入栈顶。

关键问题来了：**机器人怎么知道自己 "走到了" 还是 "卡住了"？**

这里用了一个 Value Model。它的工作不是精确告诉你 "还有 3.7 米到目标"，而是做一个三分类判断："有进展 / 没进展 / 到了"。就像你问路人 "我离山顶还远吗"，他说 "快了" 或 "还早呢" 或 "你这不是已经到了吗"——这种粗粒度判断其实够用，而且比精确数值容易得多。

每过 K 步（比如 10 步），Value Model 看一眼当前画面和上次画面，判断三种情况之一：

- **到了**：弹出栈顶，开始执行下面一个目标
- **没进展**：栈还有空间就再拆细一层（生成新子目标压栈）；栈满了说明这条路走不通，干脆清空栈，机器人 reset 回初始位置，重新来一遍
- **有进展但没到**：什么都不变，继续按当前子目标走

这就是 Algorithm 1 的全部内容。

## 子目标长什么样

子目标不是单纯的文字，也不是单纯的图片，而是 **文字 + 图片** 的一对。

为什么两个都要？因为文字提供语义（"抓起红色杯子"），图片提供视觉参照（目标状态长这样）。只有文字不够，机器人不知道 "抓起来" 之后杯子该举到多高；只有图片不够，机器人不知道图片里哪个物体是要动的。

但直接生成图片很容易 hallucinate——大模型可能给你画一个跟当前场景完全对不上的图。他们用了个 trick：

**两阶段生成**：先用 UMM 生成文字子目标（"抓起红色杯子"），再根据文字 + 当前画面生成图片子目标。文字就像一个 bottleneck，强制图片必须语义一致。

生成完之后还有一个 self-check：用 inverse dynamics 从 "当前画面 → 生成的子目标画面" 反推 "这中间执行了什么动作"，如果反推出来的动作跟原来的文字子目标对不上，就说明生成的图片有问题，丢掉重新生成。这相当于让模型自己验证自己。

## 跟之前方法比，好在哪

π0.5 已经有 subtask 预测了，但它的 subtask 是**固定粒度**的——每次预测未来固定步数的状态。Anticipation-VLA 的区别是粒度自适应：

- 任务顺利时，一个大子目标就能走完好几步，省计算
- 任务卡住时，自动往下拆，拆到机器人能执行的粒度
- 实在拆到底还卡住，直接退回起点重新规划

这跟 MPC 的 receding horizon 有点像，但加了 hierarchical 结构。也跟 MCTS 的回溯有点像，但用 value classification 替代了精确的 value estimation。

## 为什么 Value Model 用分类不用回归

这是我觉得最聪明的设计之一。传统 RL 里 value function 要预测精确回报，需要 dense reward 训练。但机器人任务的现实是：你只有 "成功 / 失败" 这种 sparse signal，没法训出精确的 value。

但 Anticipation-VLA 里 value model 不需要预测绝对值，只需要判断 **"进度状态"**：没进展 / 有进展 / 到了。这就把一个困难的回归问题转成了简单的三分类问题，只要有 "前一个画面、当前画面、目标" 三张图就能判断。

人类其实也是这么规划的。你开车去陌生地方不会时刻精确计算 "还剩 3.7 公里"，你只需要看路牌判断 "快了" "还远" "到了"。粗粒度判断对规划来说够用，而且容易训练得多。

## Random Masking 这个小 trick

训练 low-level VLA 的时候，他们随机 mask 掉子目标图片的 tokens。意思是告诉模型："你看到的子目标图片可能是噪声的、不完整的，你要学会容忍"。

Why important? 因为 inference 时子目标图片是 anticipation model 生成的，不可能完美。如果 VLA 训练时只见过 ground-truth 子目标图片，inference 时遇到生成的 noisy 图片就会 brittle。提前在训练时注入噪声，相当于 dropout 但作用在 goal representation 上，让模型学到 robust behavior。

## 实验告诉我们什么

最有说服力的几个数据点：

1. **Libero-Long 任务**上 Anticipation-VLA 63.2，π0.5 是 54.6。Long horizon 任务收益最大，符合预期
2. **π0.5+VLM 反而比纯 π0.5 略差**。说明单纯加个 VLM 做 planning 没用，反而引入噪声。Adaptive granularity 才是关键
3. **Real-world Unseen 场景 +107%**。 unseen 比 seen 提升更大，说明 anticipation mechanism 提供了更强的泛化能力
4. **Ablation 里 w/o recursive 在后期 stage 急剧崩盘**。这直接证明 fixed-granularity planning 在 long horizon 上注定失败
5. **VLABench Hammer Nail 任务**所有 baseline 几乎 0% 成功率，Anticipation-VLA 翻倍。这种需要 "锤钉子 → 放锤子 → 拿画 → 挂画" 的长链条任务正好是 adaptive subgoal 的 sweet spot

## 我的直觉理解

把这套系统想成一个**带反馈的 hierarchical decomposition**：

- 传统 hierarchical planning 是开环的：一次拆完，执行到底，错了也不知道
- Anticipation-VLA 是闭环的：执行中持续监测，根据进展决定要不要再拆、要不要回退

这跟人类解决长任务的认知机制很像。你做一道复杂数学题不会一次性想好所有步骤，而是边做边看："这步走通了，下一步该怎么走；这步卡住了，要不要换个思路"。

Anticipation Model 本质上是把这个 "边做边想" 的过程形式化了。Value Model 扮演 "我进展如何" 的感知器，Stack 扮演 "当前思路" 的工作记忆，Anticipation Model 扮演 "下一步往哪走" 的规划器。

## 局限和未来

我看完觉得几个没解决好的地方：

1. **Reset 假设太强**。栈满了就 reset robot 回初始位置，real-world 中 reset 成本可能很大（比如打翻的东西收不回来）
2. **子目标标注依赖**。训练 anticipation model 还需要人工标注的 hierarchical subgoals，未来需要 zero-shot 版本
3. **Stack depth 固定上限 d**。不同任务需要的最大深度不同，固定值不灵活
4. **子目标图片生成慢**。paper 自己承认 inference 偶尔会 pause，因为 image generation 计算贵

但总体来说这篇 paper 把 RL 中的 value-based planning 思想以 classification 形式融入 VLA，我觉得比直接做 RL fine-tuning 更 practical。RL fine-tuning VLA 一直很难（reward sparse、样本效率低、训练不稳定），而 Anticipation-VLA 用监督学习训练 value classifier 替代了 RL 中的 value regression，用 hierarchical decomposition 替代了端到端 long-horizon policy。这种 "soft RL" 思路可能才是 VLA scaling 的正确方向。

参考链接：
- Anticipation-VLA 原文: https://arxiv.org/abs/2509.05545
- Yu 2025 RL with Anticipation: https://arxiv.org/abs/2509.05545
- π0.5: https://arxiv.org/abs/2504.16054
- Bagel UMM: https://arxiv.org/abs/2506.06674
- Uni-Plan self-discriminative: https://arxiv.org/abs/2505.01515
- LIBERO benchmark: https://arxiv.org/abs/2306.03310
- VLABench: https://arxiv.org/abs/2412.01554

---

# Anticipation-VLA: 深度解析

Andrej, 这篇 paper 我读完后觉得有几个非常巧妙的 design choice 值得仔细拆解。我从高层 intuition 讲到低层公式细节，最后回到为什么这套设计能 work。

## 1. 这篇 paper 到底在解决什么问题

VLA 模型在 short-horizon 任务上已经做得不错（π0、π0.5、OpenVLA 等），但 long-horizon 任务上 fundamental 的问题是 **compounding error**。这其实是 imitation learning 的老问题了，Ross et al. 2010 的 DAgger paper 就讲过——policy 每一步都有 ε 的误差，T 步之后误差累积变成 O(εT)，而 long-horizon 任务中 T 很大。

之前大家尝试过用 subgoal decomposition 来缓解：
- **VLM-based planning** (SayCan, π0.5+VLM)：用 VLM 分解成 text subtask，但 granularity 是 fixed 的
- **Subgoal image prediction** (GR-MG, VLA-OS)：直接预测 future image 作为 subgoal
- **Implicit world modeling** (DreamVLA, WorldVLA)：单步预测未来帧

这些方法的核心问题在于 **fixed granularity**：要么 subgoal 太细（引入不必要复杂性），要么太粗（无法有效指导 policy）。Anthropic-style 的 hierarchical decomposition 没有自适应机制。

Anticipation-VLA 的核心 insight 是：**subgoal 的 granularity 应该跟着 execution 进展动态调整**，并且 **subgoal 本身可以被递归分解**。这其实是从 Yu 2025 的 "Reinforcement Learning with Anticipation" 工作继承来的理论框架。

参考链接：
- 原始 paper (arXiv): https://arxiv.org/abs/2509.05545 (Yu 2025, RL with Anticipation)
- π0.5 paper: https://arxiv.org/abs/2504.16054
- DAgger: https://arxiv.org/abs/1011.0686

## 2. GMDP 形式化：为什么 goal 需要这样定义

他们用 **Goal-Conditioned MDP (GMDP)** 来形式化问题，tuple 是 $(S, \mathcal{A}, \mathcal{G}, P, r, T)$。关键创新在 goal space 的定义：

$$
\mathcal{G} = (\mathcal{L} \cup \{\emptyset_{\mathcal{L}}\}) \times (S \cup \{\emptyset_S\}) \setminus \{(\emptyset_{\mathcal{L}}, \emptyset_S)\} \tag{1}
$$

变量解释：
- $\mathcal{L}$: instruction 集合（所有可能的自然语言指令）
- $S$: state 集合（视觉观察，比如 camera image）
- $\emptyset_{\mathcal{L}}$: null instruction（空指令占位符）
- $\emptyset_S$: null state observation（空图像占位符）
- $\times$: Cartesian product
- $\setminus$: 集合差，排除掉完全空的 goal

这个定义的妙处在于，goal 可以是 **混合模态** 的：
- 纯语言 goal: $(\emptyset_S, \text{"Make a coffee"})$
- 纯图像 goal: $(\text{"Image of a cup of coffee"}, \emptyset_{\mathcal{L}})$
- 多模态 goal: $(\text{"arrange apples"}, \text{goal image})$

而 exclusion $\setminus \{(\emptyset_{\mathcal{L}}, \emptyset_S)\}$ 保证了 goal 永远 informative。

Reward function $r: S \times A \times \mathcal{G} \to \mathbb{R}$ 定义为 "the improvement to the goal from state s after taking action a"。这个定义很关键——它不是 sparse 的 0/1 reward，而是 **goal-reaching progress**。

直觉上，$V^*(s, g)$ 可以理解为 "from $s$ to $g$ 的最短路径代价的负值"。

## 3. Anticipation Model 的理论核心

这是 paper 的核心，来自 Yu 2025。Anticipation model $G: S \times \mathcal{G} \to \mathcal{G}$ 是一个映射：输入当前 state $s$ 和当前 goal $g$，输出一个 subgoal $g'$。

### 3.1 Optimal Value Function

首先回顾 Bellman Optimality Equation：

$$
V^*(s, g) = \max_a \left[ r(s, a, g) + \mathbb{E}_{s' \sim P(\cdot|s,a)}[V^*(s', g)] \right] \tag{2}
$$

变量解释：
- $V^*(s, g)$: 在 state $s$ 追求 goal $g$ 时的最大期望累积 reward
- $a$: 当前 action
- $r(s, a, g)$: 在 state $s$ 执行 action $a$ 朝向 goal $g$ 的即时 reward
- $P(\cdot|s,a)$: 转移概率，给定 $s, a$ 后 $s'$ 的分布
- $\max_a$: 在所有 action 中取最优

### 3.2 Optimal Decomposition（关键公式）

$$
V^*(s_0, g) = V^*(s_0, g') + V^*(s_{g'}, g) \tag{3}
$$

变量解释：
- $s_0$: 初始 state
- $g$: 最终 goal
- $g'$: 中间 subgoal（由 anticipation model 输出）
- $s_{g'}$: 达到 subgoal $g'$ 时的 state

这个公式说的是：**从 $s_0$ 到 $g$ 的最优值，必须可以完美分解为从 $s_0$ 到 waypoint $g'$ 的最优值，加上从 $g'$ 到 $g$ 的最优值**。

这相当于在说：subgoal $g'$ 必须恰好在从 $s_0$ 到 $g$ 的 **最优路径上**。如果 $g'$ 偏离了最优路径，那么 $V^*(s_0, g') + V^*(s_{g'}, g) > V^*(s_0, g)$（因为绕路了）。

这就是 anticipation model 的训练目标：找一个 $g'$ 让等式（3）成立。这个约束比一般的 subgoal generation 严格得多——它要求 subgoal 是 **value-optimal** 的。

### 3.3 Recursive 性质

Anticipation model 的 recursive 性质来自：$g'$ 可以作为新的 "current goal" 再次输入到 $G$，得到更精细的 $g''$。这个过程可以一直递归下去，直到 subgoal 变成 atomic executable。

这跟 hierarchical RL 里的 option framework 有相似之处，但关键区别是：**option 是预先定义的，而 anticipation 是动态生成的**；**option 的 termination 是固定的，而 anticipation 通过 value model 动态检测**。

## 4. Anticipation-VLA 的整体架构

三个核心组件：

1. **Anticipation Model $G$**：high-level planner，递归生成 subgoal
2. **Optimal Value Function $V^*$**：估计 value，监测 progress
3. **Goal-Conditioned VLA $\pi$**：low-level controller，根据当前观察和当前 subgoal 产生 action

架构图大致是这样：

```
Final Goal g
    │
    ▼
┌─────────────┐
│ Goal Stack  │ ◄── 维护 subgoal 栈
└─────────────┘
    │ peek (top of stack)
    ▼
┌─────────────┐    每 K 步检查    ┌─────────────┐
│ VLA Policy │ ◄─────────────── │ Value Model │
│     π      │                   │     V*      │
└─────────────┘                   └─────────────┘
    │ action                              │ progress label
    ▼                                     ▼
┌─────────────┐                   ┌─────────────┐
│  Environment│ ◄──────────────── │ Anticipation│
│            │    state s         │  Model G   │
└─────────────┘                   └─────────────┘
```

### 4.1 Dynamic Subgoal Management（Algorithm 1）

这是 paper 最具创新性的部分。系统维护一个 goal stack $\mathcal{G}_{\text{stack}}$，每 K 步（planning check interval）做一次评估，分三种条件：

**Condition 1: Goal Achievement**
$$|V^*(s, g) - V^*(s_g, g)| < \delta$$
当前 value 接近 goal state 的 value → pop subgoal from stack

**Condition 2: Insufficient Progress + stack 未满**
$$|V^*(s, g) - V^*(s_{\text{prev}}, g)| < \delta \text{ 且 } |\mathcal{G}_{\text{stack}}| < d$$
没有明显进展 → anticipation model 生成 refined subgoal，push to stack

**Condition 3: Insufficient Progress + stack 已满**
$$|V^*(s, g) - V^*(s_{\text{prev}}, g)| < \delta \text{ 且 } |\mathcal{G}_{\text{stack}}| = d$$
陷入 local stagnation → 清空 stack，push final goal，reset robot pose（**backtrack**）

变量：
- $s$: 当前 state
- $s_{\text{prev}}$: 上一次 check 时的 state
- $s_g$: goal state
- $g$: 当前 active subgoal（stack top）
- $\delta$: progress threshold
- $d$: stack 最大深度

这个 backtracking 机制非常巧妙。它不是简单的 retry，而是承认当前 hierarchical decomposition 失败了，需要从初始状态重新规划。这避免了 policy 在 local minimum 中无限循环。

完整 Algorithm 1 (附录 D.1) 我解读一下关键步骤：

```
Line 4: push final goal 到 stack
Line 6-7: peek stack top 作为当前 subgoal g
Line 7: VLA 根据 执行 action
Line 11: 每 K 步做一次 planning check
Line 12-13: 如果 goal 达到 → pop
Line 15-17: 如果没进展 + stack 没满 → push 新 subgoal
Line 19-22: 如果没进展 + stack 满 → 清空 + backtrack
```

Stack 结构的妙处在于：**top 元素比 bottom 元素更细粒度**，pop 出来意味着 coarse-grained subgoal 变成新的 active goal，自然回到上层规划。

## 5. 实际实现：UMM-based Anticipation + Value

### 5.1 为什么用 UMM (Unified Multimodal Model)

他们用 **Bagel** (Deng et al. 2025) 作为 backbone，一个 UMM 能同时理解和生成 text/image。好处是 anticipation model 和 value model 可以在同一个 model 内实现，cross-modal knowledge 可以共享。

参考 Bagel paper: https://arxiv.org/abs/2506.06674

### 5.2 两阶段 Subgoal Generation（关键 design）

直接从 $(s, g)$ 预测 $g'$ 会 hallucinate。他们分两阶段：

**Stage 1: Language Policy $l_\theta$**
$$l_\theta: S \times \mathcal{G} \to \mathcal{L}$$
输入 $(s, g)$，输出 subgoal 的 textual instruction $\ell_{g'}$

**Stage 2: Forward Dynamics $P_\theta$**
$$P_\theta: S \times \mathcal{L} \to S$$
输入 $(s, \ell_{g'})$，预测对应的 subgoal image $s_{g'}$

完整 anticipation mapping:
$$G_\theta = (P_\theta \circ l_\theta, l_\theta)$$

这个两阶段 design 的妙处是 **semantic bottleneck**：text subgoal 作为语义约束，强制 image generation 与 task 语义一致。这比直接 image-to-image generation 受 hallucination 影响小得多。

### 5.3 Self-Discriminative Regularization

为了进一步减少 hallucination，他们引入 inverse dynamics verification：

$$P_\theta^{-1}: S \times S \to \mathcal{L}$$

给定 $(s, s_{g'})$，反推可能的 instruction $l'_{\text{inv}}$。如果 $l'_{\text{inv}}$ 和 $l_{g'}$ 语义等价，保留 subgoal；否则丢弃重新生成。

这相当于 **cycle consistency** 的思想：forward 生成 + inverse 验证。这种 self-verification 在 Uni-Plan (Sun et al. 2025) 中首次提出。

参考 Uni-Plan: https://arxiv.org/abs/2505.01515

### 5.4 Value Model 转为 Classification（重要 trick）

标准 value model 是回归问题，需要 dense reward 做 TD learning。但 real-world 只有 sparse trajectory-level success signal，TD learning 不稳定。

他们的关键 insight 是：**在 Anticipation-VLA 中，value model 只需要判断 Goal Achievement 或 Progress Stagnation，不需要绝对 value 值**。所以转成三分类问题：

$$V_\theta: S \times S \times \mathcal{G} \to \{0, 1, 2\}$$

输入：$(s_{\text{prev}}, s, g)$，输出类别：
- 0: No Progress (stagnant)
- 1: Progress (但未达到)
- 2: Achieved

这是一个很重要的 engineering insight——当你只需要 ordinal judgment 而不是 cardinal value 时，classification 比 regression 容易得多。

数据集构造的 label 函数（附录 C.2）：

$$
\text{label}(f_1, f_2, g) = \begin{cases}
\text{Insufficient Progress} & \text{if } f_2.\text{frame} \in [0, f_1.\text{frame} + \beta) \\
\text{Sufficient Progress} & \text{if } f_2.\text{frame} \in [f_1.\text{frame} + \beta, g.\text{frame} - \gamma) \\
\text{Goal Achievement} & \text{if } f_2.\text{frame} \in [g.\text{frame} - \gamma, \infty)
\end{cases}
$$

变量：
- $f_1$: 采样的前一帧
- $f_2$: 采样的后一帧
- $g$: subgoal
- $\beta$: progress 判定阈值（多少帧之后才算 progress）
- $\gamma$: achievement 判定阈值（离 goal 多少帧之内算 achieved）

### 5.5 训练 Loss 详解

四个 loss 同时优化在 UMM 内：

**Policy Loss（公式 4）**
$$\mathcal{L}_{\text{policy}}(\theta) = \mathbb{E}_{\mathcal{D}_{\text{anti}}}[-\log l_\theta(\ell_{g_{h+1}} | s, g_h)]$$
- $s$: 当前观察
- $g_h$: 当前 level $h$ 的 goal
- $\ell_{g_{h+1}}$: 下一层 $h+1$ 的 subgoal 的文本指令
- 标准 cross-entropy loss，训练 policy model 预测更精细的 text subgoal

**Forward Dynamics Loss（公式 5）**
$$\mathcal{L}_{\text{dyna}}(\theta) = \mathbb{E}_{t, \mathcal{D}_{\text{anti}}}\left[\|v_\theta(s, s_{g_{h+1}}^t, \ell_{g_{h+1}}, t) - v\|^2_2\right]$$
- $v_\theta$: 预测的 velocity field（flow matching 中的）
- $v$: 实际采样的 velocity
- $s$: 当前观察
- $s_{g_{h+1}}^t$: 在 flow time $t$ 处的 noisy latent
- $\ell_{g_{h+1}}$: subgoal 文本
- $t$: flow matching 的 diffusion time step
- 这是 flow matching (Lipman et al. 2023) 的标准 MSE loss

Flow matching 在这里用于 image generation，比 diffusion 更高效。

参考 Flow Matching: https://arxiv.org/abs/2210.02747

**Inverse Dynamics Loss（公式 6）**
$$\mathcal{L}_{\text{inverse}}(\theta) = \mathbb{E}_{\mathcal{D}_{\text{anti}}}[-\log P_\theta^{-1}(\ell_{g_{h+1}} | s, s_{g_h})]$$
- 输入两个 state $s, s_{g_h}$
- 输出连接它们的 instruction $\ell_{g_{h+1}}$
- 用于 self-discriminative regularization

**Value Loss（公式 7）**
$$\mathcal{L}_{\text{value}}(\theta) = \mathbb{E}_{\mathcal{D}_{\text{value}}}[-\log V_\theta(y | s_1, s_2, g_h)]$$
- $s_1$: 前一观察
- $s_2$: 当前观察
- $g_h$: 当前 goal
- $y \in \{\text{progress}, \text{achieve}, \text{no progress}\}$: 分类 label

**Total Loss（公式 8）**
$$\mathcal{L}(\theta) = \lambda_1 \mathcal{L}_{\text{policy}} + \lambda_2 \mathcal{L}_{\text{dyna}} + \lambda_3 \mathcal{L}_{\text{inverse}} + \lambda_4 \mathcal{L}_{\text{value}}$$
- $\lambda_1, \ldots, \lambda_4$: 各 loss 的权重
- 根据附录 Table 4，MSE weight=1.0, Cross entropy weight=0.01，意味着 image generation 是主要信号，text/value 是辅助

### 5.6 Goal-Conditioned VLA（基于 π0.5）

Low-level policy 用 **π0.5**，是 flow matching-based VLA。他们对 π0.5 做了改造：在输入序列中加入 **current subgoal image $s_g^t$** 和 **subgoal instruction $\ell_g^t$**。

输入序列结构：
```
[s_1^t, ..., s_n^t (多视角观察), s_g^t (subgoal image), q (robot config), ℓ_g^t (subgoal instruction)]
```

分布分解（公式 9）：
$$\pi_\theta(\mathbf{a}^{t:t+h} | \mathbf{s}_o^t, g) = \pi_\theta(\mathbf{a}^{t:t+h} | \mathbf{s}_o^t, g_t) \cdot G_\theta(g_t | g)$$
- $\mathbf{a}^{t:t+h}$: action chunk（未来 $h$ 步的 action）
- $\mathbf{s}_o^t$: 多视角观察
- $g$: 最终 goal
- $g_t$: 时刻 $t$ 的当前 subgoal
- 这是 hierarchical decomposition 的概率形式

**Random masking trick**：训练时随机 mask 掉 goal image 的 token（概率 $p$），增强 robustness。这其实是个 augmentation trick——因为 anticipation model 在 inference 时可能产生 noisy subgoal，提前在 training 时暴露给 model 这种 noise，能避免 brittle 依赖。

参考 π0.5: https://arxiv.org/abs/2504.16054

### 5.7 Causal Mask Configuration（附录 D.2 Figure 12）

UMM 内部不同任务用不同的 causal mask，这个细节很有意思：

- **Dynamics Model**: Prompt(causal) + ViT(full) + VAE cond(full) + ViT(full) + action(causal) + VAE gen(noise)
- **Inverse Dynamics**: Prompt(causal) + ViT(full) + ViT(full) + answer(causal)
- **Policy Model**: Prompt(causal) + ViT(full) + text goal(causal) + answer(causal)
- **Value Model**: Prompt(causal) + ViT(full) + ViT(full) + text goal(causal) + answer(causal)

注意：ViT blocks 内部都是 bidirectional（full attention），但不同 block 之间用 causal。这是一个 mixed attention pattern 的设计——视觉特征内部需要 bidirectional 来获得 holistic representation，但 text 部分保持 causal 以匹配 autoregressive generation。

## 6. 实验结果深度分析

### 6.1 LIBERO Results (Table 1)

| Model | Spatial | Object | Goal | Long | Avg |
|-------|---------|--------|------|------|-----|
| π0 | 70.2 | 80.0 | 70.6 | 37.6 | 64.6 |
| UniVLA | 26.0 | 40.0 | 18.0 | 1.8 | 21.5 |
| DreamVLA | 38.0 | 34.0 | 16.6 | 20.6 | 27.3 |
| π0.5 | 78.2 | 88.6 | 85.8 | 54.6 | 76.8 |
| π0.5+VLM | 82.0 | 88.0 | 80.8 | 53.2 | 76.0 |
| **Anticipation-VLA** | **81.8** | **91.6** | **86.6** | **63.2** | **80.8** |

关键观察：
1. **Libero-Long 上的提升最大**（63.2 vs 54.6，+8.6）：这正符合 anticipation model 的设计目标——长 horizon 任务收益最大
2. **π0.5+VLM 反而比 π0.5 略低**（76.0 vs 76.8）：说明单纯加 VLM 做 planning 没有 adaptive granularity，反而引入噪声
3. **UniVLA 和 DreamVLA 表现差**：one-trajectory SFT setting 下，这两种需要大量数据预训练的方法没发挥出优势

### 6.2 VLABench Results (Table 2)

| Model | Process Reward | Success Rate |
|-------|---------------|--------------|
| π0 | 39.6 | 1.0 |
| π0.5 | 42.7 | 2.1 |
| π0.5+VLM | 47.9 | 2.1 |
| **Anticipation-VLA** | **56.3** | **4.2** |

VLABench 的 Hammer Nail & Hang Picture 任务极其困难（需要 pick hammer → drive nail → place hammer back → pick picture → hang picture），baseline 几乎全军覆没。Anticipation-VLA 把 success rate 翻倍，并且 **process reward 大幅提升**，说明中间步骤更连贯。

### 6.3 Real-World Experiments (Figure 4)

两个任务：
- **Rearrange Objects**: 多物体 rearrangement，goal 通过 image 指定
- **Spell Words**: 字母块拼词，goal 通过 language 指定

Seen/Unseen 配置：
- Seen: +60% improvement over π0.5
- Unseen: **+107% improvement** over π0.5

特别值得注意的是 **Unseen Spell Words**：Anticipation-VLA 是唯一达到 non-zero success rate 的 model。这印证了 anticipation mechanism 对 OOD generalization 的关键作用。

### 6.4 Ablation Study (Figure 5, Table 7)

三个 ablation variant：
- **w/o subgoal image**: 去掉 visual subgoal
- **w/o subgoal text**: 去掉 textual subgoal
- **w/o recursive**: 用 fixed-level generation 替换 adaptive recursive

Stage-wise degradation 分析（Table 7）揭示关键 pattern：
- Anticipation-VLA (Seen Rearrange Objects): Stage 1=0.95, Stage 5=0.43
- w/o recursive: Stage 1=0.95, Stage 5=0.14（急剧下降）

**w/o recursive 在后期 stage 崩溃** 是最重要的 finding——这直接验证了 fixed-granularity planning 在 long-horizon 上不可避免地 fail。

### 6.5 Anticipation Quality (Table 3)

| Benchmark | Text Pred Acc | PSNR | MAE | SSIM | FID |
|-----------|--------------|------|-----|------|-----|
| Libero | 84.4 | 20.4 | 9.4 | 0.85 | 31.0 |
| VLABench | 88.8 | 15.5 | 19.0 | 0.76 | 55.1 |
| Rearrange Objects | 88.1 | 28.0 | 6.1 | 0.93 | 45.1 |
| Spell Words | 98.9 | 26.4 | 6.9 | 0.92 | 34.7 |

观察：
1. **Text subgoal prediction 准确率非常高**（84-99%），说明 UMM 对 task semantics 理解透彻
2. **Real-world image subgoal 质量好于 simulation**（PSNR 28 vs 20）——paper 解释是 UMM 没在 simulation 数据上预训练，并且 VLABench 包含未见过的复杂 picture 生成
3. **Spell Words text prediction 几乎完美**（98.9%），因为 sub-task 是离散的 letter selection

## 7. 关键 Insights 和 我的理解

### 7.1 为什么 stack + value monitoring 比 fixed planning 好

Fixed planning 的根本问题是 **granularity 不匹配 execution dynamics**：
- 任务简单的部分不需要细粒度 subgoal，但 fixed planning 仍生成 → 浪费计算
- 任务困难的部分需要细粒度 subgoal，但 fixed planning 没生成 → policy 在 local minimum 徘徊

Stack-based adaptive planning 的妙处在于：**让 execution 反馈决定 decomposition depth**。当 VLA 能顺畅执行时，stack 保持不变；当 VLA 卡住时，stack 自动加深。

这跟 MPC 的 receding horizon 思想类似，但加了 hierarchical 结构。

### 7.2 Backtracking 的设计哲学

Condition 3（stack 满 + 没进展 → 清空 stack + reset）这个设计很大胆。它承认：**当 hierarchical decomposition 达到极限仍无法推进时，问题不在 subgoal 上，而在初始 state 或 path 选择上**。

这跟 MCTS 的 backpropagation + selection 类似——失败的 path 应该被放弃，从根重新探索。

### 7.3 Two-stage generation 是 anti-hallucination 关键

直接 image-to-image generation 容易 hallucinate 因为缺乏 semantic constraint。两阶段（text → image）的 bottleneck 强制 generation 必须通过 semantic layer。

这与 Chain-of-Thought 在 LLM 中的作用类似：**显式中间表示比直接映射更稳健**。

### 7.4 Value Model 转 Classification 的深层意义

这不只是工程 trick，而是反映了一个 deep insight：**在 hierarchical planning 中，相对判断比绝对估计更重要**。

人类规划也是这样——你不需要精确知道距离 goal 还有多少米，只需要知道 "我在前进吗？"、"到了吗？"。把 value model 从回归改成分类，跟人类规划的认知机制更接近。

### 7.5 Random Goal Masking 的鲁棒性

训练时 mask goal image tokens 是个简单但关键的 trick。它告诉 VLA：**subgoal 是 noisy 的，不要 brittle 依赖**。

这跟 dropout 的思想一样，但作用在 goal representation 上。Anticipation model 在 inference 时输出的 subgoal 不可能完美，VLA 必须容忍这种 noise。

## 8. Limitations 和 Future Directions

Paper 自己承认的 limitations：
1. 仍需 annotated subgoal demonstrations 做 finetuning
2. Visual subgoal generation 计算昂贵，导致 inference 偶有 pause

我认为还有几个潜在 issues：
1. **Backtracking 假设 robot 可以 reset**——很多 real-world 任务中 reset 代价巨大
2. **Value model 的 progress label 是手工设定的阈值** $\beta, \gamma$，对 task-sensitive
3. **Stack depth $d$ 是固定上限**，更复杂任务可能需要更深 decomposition
4. **Anticipation model 用单一 UMM 实现，cross-task 优化可能 conflict**

## 9. 与相关工作的对比

### 9.1 vs. π0.5

π0.5 已经有 subtask prediction pretraining，但它的 subtask 是 fixed granularity 的（每个 subtask 对应固定 horizon）。Anticipation-VLA 把 subtask generation 改成 adaptive + recursive，并且加了 value monitoring 的 closed-loop。

### 9.2 vs. DreamVLA / UniVLA

DreamVLA 和 UniVLA 都用 implicit world modeling 预测未来帧，但都是单步预测，granularity 太细。Anticipation-VLA 的 multi-level hierarchy 解决了这个问题。

### 9.3 vs. SayCan / VLM-Planning

SayCan 用 LLM 做 task decomposition 输出 text subtask。Anticipation-VLA 的优势在于：
- 同时输出 text + image subgoal（multimodal）
- Adaptive granularity（SayCan 是一次性 decompose）
- Closed-loop monitoring（SayCan 是 open-loop）

### 9.4 vs. Hierarchical RL

HRL 中的 option framework 也用 hierarchical structure，但：
- Options 是预定义的，Anticipation 是动态生成
- Options 的 termination function 需要单独训练，Anticipation 用 value monitoring
- Antipation 有 explicit 的 value-optimal decomposition 理论保证（公式 3）

参考 Option Framework: https://arxiv.org/abs/1606.00503

## 10. Summary

这篇 paper 的核心贡献是把 **Yu 2025 的 anticipation theory** 落地到 **VLA framework** 中，通过三个关键设计实现：

1. **Recursive + Adaptive subgoal generation**（公式 3 的 optimal decomposition）
2. **Stack-based dynamic subgoal management**（Algorithm 1 的三条件控制）
3. **UMM-based two-stage generation with self-verification**（anti-hallucination）

实验上，在 long-horizon 任务（Libero-Long, VLABench Hammer, Real-world unseen tasks）取得显著提升，验证了 adaptive granularity 的必要性。

我认为这个工作最有启发性的地方是：**把 RL 中的 value-based planning 思想以 classification 形式融入 VLA framework**。这种 "soft" RL 思想可能比直接做 RL fine-tuning 更 practical。

后续可能的发展方向：
- Zero-shot anticipation（去掉 subgoal annotation 依赖）
- 更高效的 image generation（用 VAE latent 替代 pixel-space）
- 学习 backtracking 策略而不是简单 reset
- 多 robot / 多 agent 场景下的 anticipation

参考论文和相关链接：
- Paper: https://arxiv.org/abs/2509.05545 (Yu 2025, RL with Anticipation)
- Bagel: https://arxiv.org/abs/2506.06674
- π0.5: https://arxiv.org/abs/2504.16054
- Uni-Plan: https://arxiv.org/abs/2505.01515
- LIBERO: https://arxiv.org/abs/2306.03310
- Flow Matching: https://arxiv.org/abs/2210.02747
- SayCan: https://arxiv.org/abs/2204.01691
- Option Framework: https://arxiv.org/abs/1606.00503
- DreamVLA: https://arxiv.org/abs/2502.01871
