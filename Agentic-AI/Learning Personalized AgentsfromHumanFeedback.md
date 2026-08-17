---
source_pdf: Learning Personalized AgentsfromHumanFeedback.pdf
paper_sha256: b8ec9f66cf6ddd0a36ec3d7f592b49817ec00d2c9bab3f238b9fad307603ba67
processed_at: '2026-08-05T13:32:22-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PAHF 用人话说

## 这篇 paper 到底在解决什么问题？

想象你家里有个 robot assistant。第一天它问你 "想喝什么"，你说 "Coke"。第二天你说 "bring me my favorite drink"，它给你 Coke，对。第三天你累了想喝茶，它还是给你 Coke，错。第四天你彻底改口味了喜欢 Sprite，它还是给你 Coke，又错。

现在市面上的 personalized agent 基本都这样。问题出在哪？

**Static memory 是个陷阱**。一旦它把 "用户喜欢 Coke" 写进 memory，就死死记着。你改主意了它不知道，它也不会主动问——因为它觉得自己已经知道答案了，干嘛还问？这就是 paper 里说的 "confidently wrong"。

**Cold-start 也是个坑**。新用户来了，memory 空的，传统方法要么瞎猜要么靠 pre-collected data。但现实中新用户没 history。

PAHF 的核心 insight 就一句话：**Interaction 本身就是学习信号，得用两个 feedback channel 配合 explicit memory**。

---

## 三个步骤，人话版

### Step 1: Pre-action（动手前先问）

Agent 收到指令 "bring me my favorite drink"，先查 memory。Memory 空的？那就主动问："你喜欢啥饮料？" 用户答了，先写进 memory，再动手。

这步解决的是 **"我知道自己不知道"** 的情况。新用户、新任务、memory 没相关 note——agent 感知到 ambiguity，主动消歧。

公式 (1) 就是这步的形式化：

$$\hat{M}_t' = \mathcal{F}_{\text{update}}^{\text{pre}}(\hat{M}_t, I_t, O_t, m_t, q_t, f_t^{\text{pre}})$$

变量解释：
- $\hat{M}_t$：round $t$ 开始时 agent 的 memory 状态
- $I_t$：user 这轮的 instruction（"bring me my favorite drink"）
- $O_t$：observation（看到 counter 上有 Coke、Sprite、water）
- $m_t$：从 memory 里 retrieve 出来的相关 notes（可能是空的）
- $q_t$：agent 生成的 clarification question（"你喜欢 Coke 还是 Sprite？"）
- $f_t^{\text{pre}}$：user 的回答（"Sprite"）
- $\hat{M}_t'$：更新后的 memory，这轮 action 会用它，未来 rounds 也用它

### Step 2: Action（动手）

公式 (2)：

$$a_t = \pi_{\text{act}}(I_t, O_t, m_t, q_t, f_t^{\text{pre}})$$

Agent 综合所有信息——原始 instruction、observation、retrieved memory、刚才的 clarification 对话——生成 executable action。如果 Step 1 没问问题，$q_t$ 和 $f_t^{\text{pre}}$ 就是 null，agent 直接靠 memory + observation 决策。

### Step 3: Post-action（做错了就改）

这步是 paper 的核心创新。Agent 做完动作，用户看了结果，如果错了会说 "actually I now prefer Sprite"。Agent 收到这个 feedback，先过个 **salience detector**（LLM-as-judge）判断这反馈有没有信息量——"Thank you" 这种就过滤掉，"Actually I like Sprite now" 这种就留下来。然后 update memory。

公式 (3)：

$$\hat{M}_{t+1} = \mathcal{F}_{\text{update}}^{\text{post}}(\hat{M}_t', I_t, m_t, q_t, f_t^{\text{pre}}, a_t, f_t^{\text{post}})$$

新变量：
- $a_t$：刚才 agent 做的 action（给用户拿了 Coke）
- $f_t^{\text{post}}$：user 事后的 corrective feedback（"Actually I like Sprite now"）
- $\hat{M}_{t+1}$：这轮结束后的 memory，下一轮开始用它

如果 $f_t^{\text{post}}$ 没信息量，$\hat{M}_{t+1} = \hat{M}_t'$，memory 原样保留。

**为什么这步关键？** 因为它解决了 pre-action 解决不了的问题。当 memory 被旧的 preference 占据时，agent 不觉得 ambiguous——它觉得自己知道答案，不会主动问。只有做错了被 user 纠正，才能 trigger memory rewrite。这是唯一能 escape "confidently wrong" 状态的机制。

---

## 理论为什么这么漂亮

### Proposition 1：没有 post-action feedback，在 preference drift 下必死

设定很简单：假设没有 ambiguous round（$\gamma = 0$），user preference 在 $T$ 轮里至多变 $K$ 次。

**不用 post-action feedback 的 policy 一定 $\Omega(T)$ 错误**。

证明的核心招数：令 switch time $\tau$ 在 $\{1, ..., T\}$ 上 uniform 分布。两个 action $\{0, 1\}$，switch 前 optimal 是 0，switch 后是 1。

关键 Lemma 2：如果 agent 不用 post-action feedback，那它的 action sequence $a_{1:T}$ 的分布 **完全独立于 $\tau$**——因为它没法感知 switch 何时发生，$(I_t, O_t)$ 对所有 $\tau$ 都一样，它内部状态演化只依赖 $(I_t, O_t)$，所以输出 action sequence 也独立于 $\tau$。

然后对任何 deterministic $a_{1:T}$，算 expected loss over $\tau$：

$$\mathbb{E}_\tau[\ell] = \frac{1}{T}\sum_{t=1}^T \big(\mathbf{1}[a_t=1]\cdot(T-t) + \mathbf{1}[a_t=0]\cdot t\big)$$

解释一下每一项：
- $\mathbf{1}[a_t=1]\cdot(T-t)$：如果 agent 在 round $t$ 选了 action 1，那 switch time $\tau > t$ 时这个 action 是错的（switch 前应该选 0），错的 round 数是 $T - t$ 个可能的 $\tau$ 值
- $\mathbf{1}[a_t=0]\cdot t$：如果 agent 在 round $t$ 选了 action 0，那 switch time $\tau \leq t$ 时这个 action 是错的，错的 round 数是 $t$ 个可能的 $\tau$ 值

对每个 $t$，agent 要么选 0 要么选 1，所以 contribution 至少是 $\min(t, T-t)$。Sum 起来：

$$\frac{1}{T}\sum_{t=1}^T \min(t, T-t) \approx \frac{1}{T}\cdot\frac{T^2}{4} = \frac{T}{4} = \Omega(T)$$

人话：在 switch time 未知的情形下，无论 agent 怎么选，总有大约 $T/4$ 个 round 是错的。Linear regret。

**反过来**，如果 agent 在每次 switch 后第一次错时用 post-action feedback 修正 memory，那每次 switch 至多错 1 次，total $\leq K$ 次。

### Proposition 2：没有 pre-action feedback，在 ambiguity 下必死

设定：$M_t^*$ stationary，$\gamma$ 比例的 round 是 ambiguous（Bayes-optimal error $\geq \varepsilon_0 > 0$）。

不用 pre-action query，每个 ambiguous round 至少 $\varepsilon_0$ 概率错，summed over $\gamma T$ rounds → $\Omega(\gamma T)$。

用 $k$ 个 balanced m-ary query（每个问题有 $m$ 个选项），由 Lemma 1，每个 query 把 wrong-action posterior mass 缩 $1/m$ 倍。$k$ 个 query 后错判概率 $\leq m^{-k}$。Summed → $O(\gamma T \cdot m^{-k})$。

### Theorem 1：两个 channel 一起用，regret 与 horizon 几乎无关

Dynamic regret（相对于知道 $M_t^*$ 的 oracle）：

$$\mathbb{E}[\mathcal{R}_T] = O(K + \gamma T m^{-k})$$

把 rounds 分两类加起来：
- Unambiguous rounds：Proposition 1 给 $O(K)$
- Ambiguous rounds：Proposition 2 给 $O(\gamma T m^{-k})$

取 $k = \lceil \log_m T \rceil$，则 $\gamma T \cdot m^{-k} = \gamma T / T = \gamma$，所以：

$$\mathbb{E}[\mathcal{R}_T] = O(K + \gamma)$$

**这个结果人话是什么意思？** Agent 长期运行 $T$ 轮，累积错误数只跟 "preference 变了几次"（$K$）和 "环境有多 ambiguous"（$\gamma$）有关，跟运行了多久 $T$ **几乎无关**。这是 sublinear regret，非常强。

对比经典 non-stationary bandit：没 detection 机制就 $\Omega(T)$ linear regret，有 sliding window 或 restart 策略才能 sublinear。PAHF 的 post-action update 本质就是个 "error-triggered restart"，比 time-based window 更 efficient——只在真错时才 update，不浪费。

---

## 实验设计为什么 convincing

### Embodied Manipulation

40 个 persona，每个有 conditional preference logic。比如 Alex：
- Default: black coffee
- Drowsy: herbal tea（相信 rest 治疲劳）
- Dehydrated: ice-cold water

关键：preference 是 **latent context 的函数**，不是简单的 key-value。Agent 不能光记 "Alex 喜欢咖啡"，得记 "Alex 在什么状态下喜欢什么"。

**Evolved version**：1-to-1 inversion。Alex 原来 drowsy 喝 herbal tea（rest），evolved 改喝 energy drinks（stimulation）。Storage 从 "high shelves for hygiene" → "low shelves for accessibility"。这是 **hard distribution shift**——原来对的全变错，逼 agent unlearn + relearn。

### Online Shopping

20 persona，10 个 product category，每个 category 3 个 feature dimension。Persona 用 **三层 preference**：Preferred / Acceptable / Disliked。

严格 conjunctive policy：一个 product acceptable iff **所有** feature 都符合。Adversarial "near-miss" distractor：把 highly preferred attributes + 一个 disqualifying "poison pill" 组合。比如用户喜欢 OLED + webOS，有个 option 是 OLED + Roku TV——第一个 feature 对，第二个错，agent 得仔细查每个 feature 不能光看第一个匹配就选。

Drift 是 **stochastic reshuffle**：Preferred 和 Disliked 可能互换。比 embodied 的 logical inversion 更 soft。

Phase 3 scenarios 跟 Phase 1 **structural identical**（同样的 products、同样的 instructions），但 ground truth 不同。这 cleanly isolate "agent 对 drift 敏不敏感"——如果 agent 依赖 memory，Phase 3 会照搬 Phase 1 的答案，全错。

### 四阶段协议

| Phase | 干啥 | Memory 起点 | Persona | Feedback |
|---|---|---|---|---|
| 1 | 训练 | 空 | original | 有 |
| 2 | 测试 | Phase 1 学的 | original | 无 |
| 3 | 训练（drift） | Phase 2 留下的 | evolved | 有 |
| 4 | 测试 | Phase 3 改的 | evolved | 无 |

Phase 2 和 Phase 4 没 feedback，pure 测 memory 里的 preference 对不对。

---

## 实验结果人话版

主结果表（GPT-4o agent + GPT-4o human simulator + SQLite）：

| Method | Embodied P2 | Embodied P4 | Shopping P2 | Shopping P4 |
|---|---|---|---|---|
| No memory | 32.3 | 44.8 | 27.8 | 27.0 |
| Pre-action only | 54.1 | **35.7** ↓ | 34.4 | 56.0 |
| Post-action only | 67.9 | 68.3 | 38.9 | 66.9 |
| PAHF | **70.5** | **68.8** | **41.3** | **70.3** |

几个关键 takeaway：

**Pre-action only 在 Phase 4 比 No memory 还差**（embodied 35.7% vs 44.8%）。这听起来反直觉，其实完全符合 Proposition 1。Memory 里全是 stale notes，agent 觉得自己知道答案不再 ask，一直蒙错。No memory 没历史包袱，反而随机性能蒙对一些 evolved scenario。

**Post-action only 的 Phase 4 跟 PAHF 接近**（embodied 68.3 vs 68.8）。但 Phase 1 学习曲线很慢——必须先错一次才学到。PAHF 的 pre-action 避免 initial cost。

**PAHF 全场最优**，所有 4 phase × 2 domain 都第一。

学习曲线（Figures 3, 4）的故事：
- Phase 1 iteration 1：pre-action agents（Pre-action only + PAHF）直接 high success rate，因为动手前就问了。Post-action only 从低开始，靠 trial-and-error 慢慢爬。
- Phase 3：Pre-action only flat，feedback frequency 接近 0——不再 ask 了，confidently wrong。Post-action only + PAHF 陡峭爬升，post-action signal 触发 memory rewrite。

---

## Ablation 几个关键发现

**Simulator 质量**（Table 2）：GPT-4.1 agent × GPT-4o human 下 PAHF embodied Phase 2 达 82.3%，Phase 4 78.8%。更强 backbone 放大 PAHF 优势。说明 PAHF 不是 GPT-4o 特定的 hack。

**Memory backend**（Table 3）：SQLite vs FAISS 结果几乎一致（PAHF embodied P2: 70.5 vs 68.0；shopping P4: 70.3 vs 70.8）。效果来自 feedback mechanism 本身，跟 memory 实现无关。

**Llama-4-Scout 当 human simulator**（Figures 11, 12）：PAHF 还是赢，说明对 human simulator 质量不敏感。

---

## Qualitative trace 最直观

Appendix E.1 的 Avery 例子，完整四阶段 trace：

**PAHF**：
- Phase 1: "favorite drink?" → memory 空 → Ask → "Herbal tea" → 写 memory → Action A 对
- Phase 2: "favorite drink?" → memory 有 "Avery prefers herbal tea" → 直接 Action A，对
- Phase 3 (drift): "favorite drink?" → memory 还说 herbal tea → Action A → User: "I changed preference, now I prefer coffee" → post-action update
- Phase 4: "favorite drink?" → memory 现在 "Avery's favorite is coffee" → Action B，对

**Pre-action only** 同样 Avery：
- Phase 3: memory 还说 herbal tea → Thought "context specifies herbal tea" → Action A → **没有 post-action update 机制** → memory 一直错
- Phase 4: 仍 Action A，**完全没意识到 drift**

这个对比 vivid 展示了 Proposition 1 的实际表现：没 post-action 通道，drift 后永远 stuck 在 stale belief。

---

## 跟相关工作的区别，人话版

**vs RLHF/DPO**：RLHF 学一个 global reward model，所有用户共享。DPO 用 preference data 做 SFT，也是 global。都没法 per-user 个性化，更别说 non-stationary。Personalized RLHF（如 Li et al. 2024b）用 multi-objective 或 personalized fine-tune，但需要 pre-collected data，one-off fine-tune，drift 了就废。

**vs Mem0 / A-Mem / MemGPT**：这些 memory-augmented agent 强调 long-term memory、summarization、retrieval，但都假设 pre-populated user profile，不处理 online learning from scratch + drift。PAHF 的 memory design 故意简单，为了 isolate feedback channels 效果。这些 advanced memory 可作为 drop-in replacement。

**vs 作者自己前作 RLHS**（Liang et al., 2025a, https://arxiv.org/abs/2501.08617 ）：指出 RLHF 会 induce untruthful behavior，hindsight simulation 是 mitigation。这个 hindsight perspective 直接 motivate 了 PAHF 的 post-action channel——user 在 action 后给反馈，agent 用它 correct memory，本质是 hindsight signal。

**vs TidyBot**（Wu et al., 2023, https://tidybot.cs.princeton.edu ）：TidyBot 做 personalized robot assistance，但 implicit modeling + offline。PAHF 是 explicit memory + online。

**vs PREFDISCO**（Li et al., 2025, https://arxiv.org/abs/2510.00177 ）：benchmark interactive preference discovery，但 limit 在 static persona + short-horizon dialogues。PAHF 显式 handle non-stationarity + long-horizon。

---

## 我觉得这篇 paper 的 contribution 在哪

1. **识别了两个 asymmetric failure mode**：known unknown（partial observability）vs unknown unknown（non-stationarity）。前者 pre-action 能解，后者必须 post-action。这个 insight 本身就值 paper。

2. **Theorem 1 的 sublinear regret**：$O(K + \gamma)$ 当 $k = \log_m T$ 时。这不是 trivial 的——non-stationary online learning 里能 sublinear 就很不错了，更何况这里还是 partial observable + non-stationary 同时存在。

3. **Benchmark 设计精妙**：Phase 3 与 Phase 1 structural identical 但 ground truth 不同，cleanly isolate drift sensitivity。Embodied 用 logical inversion（hard shift），shopping 用 stochastic reshuffle（soft shift），两种 drift mode 都覆盖。

4. **Explicit memory + LLM-as-operator**：model frozen，所有 personalization 通过 memory read/write。这让 deployment 实用：一个 base model + N 个 user memory stores，而非 N 个 fine-tuned models。跟 LoRA-based personalization（Tan et al., 2024, https://arxiv.org/abs/2402.04401 ）比，scalable 得多。

---

## 跟其他领域的 connection

**Bayesian Theory of Mind**（Baker et al., 2011, https://proceedings.mlr.press/v9/baker11.html ）：把 user 视为 latent state，agent 维护 belief。PAHF 用 explicit memory 替代 implicit Bayesian update，更 scalable——Bayesian update 要 maintain posterior over all hypotheses，memory 只存当前 best guess。

**Active Learning**（Settles, 2009, https://www.morganclaypool.com/doi/10.2200/S00429ED1V01Y201207AII006 ）：pre-action clarification 本质是 active query selection。Theorem 1 的 $k = \log_m T$ 对应 active learning 的 exponential error reduction。

**Non-stationary Bandits**（Garivier & Moulines, 2008, https://arxiv.org/abs/0809.2053 ）：Proposition 1 的 $\Omega(T)$ lower bound 对应 classic non-stationary bandit 无 detection 时的 linear regret。Post-action update 类似滑动窗口 / restart 策略，但是 error-triggered 而非 time-based，更 efficient。

**Memory Networks**（Graves et al., 2014, https://arxiv.org/abs/1410.5401 ；Sukhbaatar et al., 2015, https://arxiv.org/abs/1503.08895 ）：可微 memory 的 write/read 本质就是 PAHF 的 retrieve/update op，但 PAHF 用 LLM 实现，是 "neural-symbolic memory"。

**Dialog Systems with State Tracking**：传统 task-oriented dialog 有 belief tracking（Mrksic et al., 2017, https://arxiv.org/abs/1606.04533 ），但都假设 fixed ontology + stationary user goal。PAHF 处理 open-vocabulary natural language notes + drifting preference。

**Continual Learning**（Kirkpatrick et al., 2017, https://arxiv.org/abs/1612.00796 ）：EWC 等 weight-level continual learning 防 catastrophic forgetting，但参数空间固定。PAHF 的 continual 在 memory space，append/rewrite notes，天然 avoid forgetting 旧 preference（只是 mark 为 outdated）。

---

## Limitations 的诚实评价

作者自己承认几个点：

1. **Memory 太简单**：dense retrieval + natural language notes，没用 hierarchical / structured memory。但这恰恰是设计 choice——isolate feedback channels 效果。未来换 Mem0 这类 advanced memory 应该能 boost。

2. **Noisy feedback**：当前 salience detector 只过滤 "Thank you" 这种，不处理 user confused 给错反馈的情况。现实中 user 会 inconsistent，需要 conflict detection + re-clarification。

3. **Shopping benchmark 仍 hard**：PAHF Phase 2 只 41.3%。Tasks 故意 adversarial + sparse feedback。未来更强 reasoning model 应该能提升。

4. **单 clarification/task**：模拟 user friction，但 shopping 中常需多轮 sequential disambiguation。Multi-turn clarification + learned turn allocation 是 obvious next step。

---

## 一句话总结

PAHF 把 personalized agent 重新形式化为 online learning with non-stationary latent state，用 explicit memory + dual feedback channels（pre-action ask + post-action correct）实现 continual personalization。理论上证明两个 channel 互补且 necessary，实验上 four-phase protocol 显示 PAHF 全场最优，sublinear regret $O(K + \gamma)$。

对人话来说就是：**别光记着用户以前喜欢啥，得在用户改主意时能被纠正，新用户来了能主动问**。听起来简单，但 prior work 真的没做到。

---

# PAHF: Personalized Agents from Human Feedback 深度解析

## 1. Paper 核心问题与动机

这篇 paper 来自 Meta Superintelligence Labs、Princeton、Duke，第一作者 Kaiqu Liang（Princeton，曾在 Meta 实习期间完成此工作），通讯作者还包括 Shuyan Zhou（Duke）和 Saghar Hosseini（Meta）。Project page: https://personalized-ai.github.io ；Code: https://github.com/facebookresearch/PAHF 。

核心痛点非常清晰：modern LLM agents 在面对 **individual user 的 idiosyncratic, evolving preferences** 时几乎全面失败。prior approaches 有两条主流路线，但都有根本缺陷：

1. **Implicit preference modeling from logs**：在历史交互数据上训练 model 去 infer user preference（如 Personalized RLHF, PAD, Leqi et al.）。缺陷：无法 handle cold-start（新用户没 history），无法在 deployment 时 online 纠错。
2. **Pre-populated memory with static profiles**：把 user profile 塞进 memory（Mem0, A-Mem 等）。缺陷：profile 是静态的，当 user preference drift 时 memory 变成 "confidently wrong" 的毒药。

PAHF 把问题重新形式化为 **online learning with non-stationary latent state**，关键 insight 是：interaction 本身就是学习信号，需要 **dual feedback channels**（pre-action + post-action）配合 explicit per-user memory。

---

## 2. Continual Personalization 的形式化

### 2.1 问题设定

每个 round $t \in [T]$：
- $M_t^*$：user 的 **latent preference state**（隐变量，agent 不可直接观测）
- $I_t$：user 发出的 instruction
- $O_t$：agent 的 observation（scene / product catalogue），其分布 conditional on $M_t^*$
- $\hat{M}_t$：agent 维护的 **explicit preference memory**（persistent estimate）
- $a_t$：agent 选择的 action
- $a_t^*$：在 true preference $M_t^*$ 下的 optimal action

Agent 目标是学习 personalized policy $\pi(a_t \mid I_t, O_t, \hat{M}_t)$，最小化 cumulative personalization error：
$$\sum_{t=1}^{T} L_t, \quad L_t = \mathbf{1}[a_t \neq a_t^*]$$

即 0-1 loss 的累积和。

### 2.2 两类核心错误

**Partial Observability（known uncertainty）**：
$M_t^*$ hidden，$\hat{M}_t$ 可能 incomplete（如新用户 $\hat{M}_t = \emptyset$）。Agent 知道自己不知道 → 这正是 pre-action clarification 的用武之地。

**Non-Stationarity（preference drift）**：
$M_t^*$ 可以 evolve 成 $M_{t+1}^*$。这时 $\hat{M}_t$ 可能 confidently wrong（agent 相信 $\hat{M}_t = \{\text{likes Coke}\}$，但 $M_t^* = \{\text{likes Sprite}\}$）。Agent 不知道自己错了 → pre-action query 触发不了，必须靠 post-action corrective feedback。

### 2.3 Ambiguous round 的定义

一个 round 称为 ambiguous，如果在 agent 当前 information state（posterior conditioned on transcript up to $t$）下，Bayes-optimal error probability $\geq \varepsilon_0 > 0$（某固定常数）。令 $\gamma \in [0,1]$ 表示 ambiguous round 的比例。

---

## 3. PAHF 三步循环

### 3.1 Pre-Action Interaction（主动消歧）

Agent 收到 $(I_t, O_t)$ 后，先 query memory：$m_t = \text{Retrieve}(\hat{M}_t, I_t, O_t)$。

如果检测到 ambiguity（instruction ambiguous 且 memory 中没相关 preference），agent 主动生成 clarification query $q_t$，收到 pre-action feedback $f_t^{\text{pre}}$，**在 act 之前**就更新 memory：

$$\hat{M}_t' = \mathcal{F}_{\text{update}}^{\text{pre}}(\hat{M}_t, I_t, O_t, m_t, q_t, f_t^{\text{pre}}) \tag{1}$$

这里 $\hat{M}_t'$ 是 "mid-round updated memory"，既用于本 round 的 action，也持久化到未来 rounds。

### 3.2 Action Execution

Action policy $\pi_{\text{act}}$ 综合所有信息生成 executable action：
$$a_t = \pi_{\text{act}}(I_t, O_t, m_t, q_t, f_t^{\text{pre}}) \tag{2}$$

如果不需要 clarification，则 $q_t = \text{null}$, $f_t^{\text{pre}} = \text{null}$, $\hat{M}_t' = \hat{M}_t$。

### 3.3 Post-Action Feedback Integration（核心创新）

执行 $a_t$ 后，环境转换到 $O_{t+1}$。如果 $a_t$ 非 optimal（即 $a_t \neq a_t^*$），user 提供 post-action feedback $f_t^{\text{post}}$。

关键设计：先用一个 LLM-as-judge **salience detector** 判断 $f_t^{\text{post}}$ 是否含有 salient personalized information（过滤掉 "Thank you" 这类非信息反馈）。如果 salient，执行 update：

$$\hat{M}_{t+1} = \mathcal{F}_{\text{update}}^{\text{post}}(\hat{M}_t', I_t, m_t, q_t, f_t^{\text{pre}}, a_t, f_t^{\text{post}}) \tag{3}$$

否则 $\hat{M}_{t+1} = \hat{M}_t'$。

Post-action channel 的独特威力：能纠正 **confidently wrong** 的 miscalibration 状态——这种状态下 agent 不感知 ambiguity，不会主动 ask，pre-action channel 完全失效。

---

## 4. 理论分析（这是 paper 最有意思的部分）

### 4.1 Information-theoretic assumptions

**A1（Balanced m-ary pre-queries）**：在 ambiguous round 上，agent 能选一个 m-ary question，使得对每个可能 answer，posterior mass on wrong-action hypotheses 缩减到原来的 $1/m$。Answers noise-free。

**A2（Corrective post-signal on errors）**：在 unambiguous round 上若 $a_t \neq a_t^*$，post-action feedback 足以 identify $a_t^*$（等价于 update 到一个能在下次 unambiguous rounds 都诱导 $a_t^*$ 直到下次 switch 的 memory）。

### 4.2 Lemma 1（m-ary question 的 error shrinkage）

证明很简洁：令 $q_0$ 为任何 query 前 wrong-action hypotheses 的 posterior mass。每问一个 balanced m-ary question，$q_{i+1} \leq q_i / m$（A1 保证）。问 $k$ 个问题后 $q_k \leq q_0 m^{-k} \leq m^{-k}$（因 $q_0 \leq 1$）。Bayes-optimal 误差 ≤ wrong-hypothesis posterior mass，所以 misclassification 概率 $\leq m^{-k}$。

### 4.3 Proposition 1：post-action feedback 在 preference drift 下 necessary

**设定**：$\gamma = 0$（无 ambiguous round），preferences 至多 switch $K \geq 1$ 次。

**Lower bound（无 post-action 反馈 → $\Omega(T)$ 错误）**：

只需证 $K=1$ 情形。令 switch time $\tau$ uniformly 取自 $\{1, ..., T\}$。考虑两个 action $\{0, 1\}$，$(I_t, O_t)$ 对所有 $t, \tau$ 都 identical。Optimal action：
$$a_t^*(\tau) = \begin{cases} 0, & t < \tau \\ 1, & t \geq \tau \end{cases}$$

由 Lemma 2（无 post-action update 时 agent 内部状态 + action sequence 分布独立于 $\tau$），对任何 deterministic $a_{1:T}$：
$$\mathbb{E}_\tau[\ell(\tau; a_{1:T})] = \frac{1}{T}\sum_{t=1}^T \big( \mathbf{1}[a_t=1](T-t) + \mathbf{1}[a_t=0] \cdot t \big) \geq \frac{1}{T}\sum_t \min\{t, T-t\} = \frac{\lfloor T^2/4 \rfloor}{T} = \Omega(T)$$

直觉：在 switch time 未知的情况下，任何固定 action sequence 都会被卡在中间一段 "half wrong" 区域，期望错误 $\approx T/4$。Randomized policy 通过对 policy randomness 取期望也只能更差。所以 no post-action feedback 的 policy 必然 $\Omega(T)$ 错误。

**Upper bound（有 post-action 反馈 → $O(K)$ 错误）**：

每次 switch 后，agent 可能错一次；那一次错时 A2 保证 feedback identify $a_t^*$，agent update memory，后续 unambiguous round 全部正确。所以每个 switch 至多 1 个 mistake，total $\leq K$。

### 4.4 Proposition 2：pre-action feedback 在 partial observability 下 necessary

**设定**：$M_t^*$ stationary，$\gamma > 0$ 比例的 round 是 ambiguous。

**Lower bound（无 pre-action query → $\Omega(\gamma T)$ 错误）**：
每个 ambiguous round Bayes-optimal 误差 $\geq \varepsilon_0$，summed over $\gamma T$ rounds → $\Omega(\gamma T)$。

**Upper bound（$k$ 个 m-ary query → $O(\gamma T \cdot m^{-k})$ 错误）**：
由 Lemma 1，每个 ambiguous round 误差 $\leq m^{-k}$，summed over $\gamma T$ rounds → $O(\gamma T m^{-k})$。

### 4.5 Theorem 1：两通道互补性（main theorem）

Oracle $\pi^*$ 知道 $M_t^*$，每 round 0 loss。Dynamic regret：
$$\mathcal{R}_T = \sum_{t=1}^T \big(L_t - \mathbf{1}[a_t^{\pi^*} \neq a_t^*]\big) = \sum_t L_t$$

PAHF policy 同时 (i) 在 ambiguous round 上问 $k$ 个 m-ary question，(ii) 每次 switch 后第一个 error 立即 post-action update。则：
$$\mathbb{E}[\mathcal{R}_T] = O(K + \gamma T m^{-k})$$

**证明思路**：把 rounds 分两类。
- Unambiguous rounds：Proposition 1 → $O(K)$ mistakes
- Ambiguous rounds：Proposition 2 + Lemma 1 → $O(\gamma T m^{-k})$ mistakes
- Sum：$O(K + \gamma T m^{-k})$

**推论**：取 $k = \lceil \log_m T \rceil$，则 $\mathbb{E}[\mathcal{R}_T] = O(K + \gamma)$，即 **sublinear in $T$**——agent 只在每次 switch 时付 1 次 mistake，再付 $\gamma T \cdot m^{-k} = \gamma T / T = \gamma$ 次 ambiguous 误判。这是很漂亮的结果：dynamic regret 与 horizon $T$ 几乎无关，只跟环境 non-stationarity 程度 $K$ 和 ambiguity 程度 $\gamma$ 有关。

**Remark 1（context-dependent preferences）**：当 preferences depend on context $C_t$（如 time, location, state），$C_t$ 改变时用 context-agnostic "global" note 会导致 confidently wrong（pre-action 不触发）→ 必须靠 post-action signal 修正。这对应 paper 后面 Phase 3 的 "context overgeneralization" failure mode。

---

## 5. 实现细节

### 5.1 Agent backbone

用 GPT-4o（ablation 中也试了 GPT-4.1），基于 ReAct 框架（Yao et al., 2022, https://arxiv.org/abs/2210.03629 ）interleave reasoning + acting。

### 5.2 Memory backend（deliberately simple）

两个 backend，identical retrieval semantics：
- **SQLite note store**：on-disk table，similarity on demand，简单可复现
- **FAISS vector index**：in-memory nearest neighbor，快速 retrieval，可选 save/load

每个 memory entry = 短自然语言 note + embedding（用 DRAGON+ encoder, Lin et al., 2023, https://arxiv.org/abs/2302.07452 ）。

API 提供：add note, top-k retrieval, near-duplicate detection, in-place update, enumerate by id。**Per-user 严格隔离**。

### 5.3 Memory interaction pipeline

**Reading**：标准 RAG pipeline，从 $(I_t, O_t)$ 算 query embedding，跑 kNN search 取 top-k。然后 lightweight information-extraction 步骤把 retrieved notes distill 成 task-relevant cues，再 insert 进 action-selection context。

**Writing**（detect-summarize-integrate pipeline）：
1. Salience detector（LLM-as-judge）：判断 feedback 是否含 personalized info
2. Summarize：LLM 提取核心 personalized note
3. Integrate：retrieve 最 relevant 现有 note，若 similarity $> \tau$ 则 merge（LLM 生成 updated note 替换 old），否则 add new note

### 5.4 四个 baseline 配置

(i) **No Memory**：无 persistent store
(ii) **Pre-action Only**：允许 clarification，无 post-action update（针对 $\gamma > 0, K=0$）
(iii) **Post-action Only**：无 clarification，只从 corrective feedback 学（针对 $\gamma=0, K \geq 1$）
(iv) **PAHF**：两通道都开

---

## 6. Benchmark 设计

### 6.1 Embodied Manipulation Domain

- 40 users，每人 30 scenarios/phase
- 模拟室内 mobile manipulation（home/office）
- 两类任务：select right item, place item at right location
- **9 个 preference category**：Drinks, Snacks, Storage, Location, Temperature, Environmental Approach, Health Considerations, Social Context, Time of Day
- Persona 有 **conditional logic**：preference 是 latent context 的函数。如 "Alex" 一般偏好 black coffee，但 "Drowsy" 时偏好 herbal tea（相信休息治疲劳），"Dehydrated" 时偏好 ice-cold water
- **Evolved version**：1-to-1 inversion of belief system。如原 "Alex" drowsy 时喝 herbal tea（rest），evolved 改为 energy drinks（stimulation）；storage 从 "high shelves for hygiene" → "low shelves for accessibility"
- Human simulator：另一个 LLM with persona prompt
- 4 phases × 1200 scenarios/phase = 4800 总 interactions，2400 train + 2400 eval

### 6.2 Online Shopping Domain

- 20 users，每人 45 scenarios/phase
- 10 product category：TVs, Laptops, Smartphones, Refrigerators, Washing Machines, Microwave Ovens, Air Conditioners, Dishwashers, Cameras, Headphones
- 每个 category 由 **3 个 feature dimension** 定义（如 TV: Smart TV OS, Panel Tech, Base Type）
- Persona 是 **tiered preference system**：Preferred / Acceptable / Disliked 三层
- 严格 conjunctive acceptance policy：一个 candidate acceptable iff ALL features meet criteria
- **Adversarial "near-miss" distractors**：组合 highly preferred attributes + 单个 disqualifying "poison pill"。强迫 agent 做细粒度 reasoning
- Agent 必须选 A/B/C/D（D = abstain）
- **Hybrid human simulator**：clarification 用 LLM，purchase verdict 用 deterministic rule-based evaluator（确保 ground-truth 严格）
- **Stochastic drift**：Phase 3/4 对每个 feature reshuffle，Preferred ↔ Disliked 可能互换
- Phase 3 scenarios 与 Phase 1 **structurally identical**（same products & instructions），但 ground truth 不同——counterfactual design，cleanly isolate sensitivity to drift
- 4 phases × 900 scenarios/phase = 3600 总，1800 train + 1800 eval

### 6.3 四阶段评估协议

| Phase | 用途 | Memory 初始 | Persona | Feedback |
|---|---|---|---|---|
| 1: Initial Learning | 训练 | empty $\hat{M} = \emptyset$ | original | full |
| 2: Initial Personalization | 测试 | from Phase 1 | original | none |
| 3: Adaptation to Drift | 训练 | from Phase 2 | evolved | full |
| 4: Adapted Personalization | 测试 | from Phase 3 | evolved | none |

### 6.4 Metrics

- **Success Rate (SR)**：$SR = \frac{1}{N}\sum_{i=1}^N \mathbf{1}[\text{correct}_i]$
- **Feedback Frequency (FF)**：$FF = \frac{1}{N}\sum_{i=1}^N \mathbf{1}[\text{pre}_i + \text{post}_i > 0]$
- **Average Cumulative Personalization Error (ACPE)**：$ACPE_t = \frac{1}{t}\sum_{s=1}^t PE_s$，其中 $PE_t \in [0,1]$ 是 round $t$ 的错误率。低 ACPE 早期就意味着快速 warm-start

---

## 7. 实验结果深度解读

### 7.1 主结果表（Table 1，GPT-4o agent + GPT-4o simulator + SQLite memory）

| Method | Embodied Phase 2 | Embodied Phase 4 | Shopping Phase 2 | Shopping Phase 4 |
|---|---|---|---|---|
| No memory | 32.3 ± 0.4 | 44.8 ± 0.5 | 27.8 ± 0.2 | 27.0 ± 0.4 |
| Pre-action only | 54.1 ± 1.1 | 35.7 ± 1.0 | 34.4 ± 0.5 | 56.0 ± 0.7 |
| Post-action only | 67.9 ± 1.5 | 68.3 ± 1.2 | 38.9 ± 0.5 | 66.9 ± 0.8 |
| **PAHF** | **70.5 ± 1.7** | **68.8 ± 1.3** | **41.3 ± 0.8** | **70.3 ± 1.1** |

几个关键观察：

**Pre-action only 在 Phase 4 反而比 No memory 还差**（embodied: 35.7% vs 44.8%）。这非常 counterintuitive 但完全符合 Proposition 1 的预测——一旦 memory 被 "confidently wrong" 的 notes 占据，agent 不再感知 ambiguity，停止 ask clarification，stale belief 一直 poison 决策。No memory 反而随机性更高，Phase 4 在 evolved persona 下偶然能蒙对一些。

**Post-action only 的 Phase 4 ≈ PAHF**（embodied: 68.3 vs 68.8）。但 Post-action only 的 Phase 1 学习曲线很慢——必须靠 trial-and-error，每个新 preference 都要先错一次才学到。这正是 PAHF 的价值：pre-action 避免 initial cost。

**PAHF 在所有 4 个 phase × 2 domain 都最优**，且 ACPE 最低（Figures 3, 4 bottom-right）。

### 7.2 学习曲线的关键 insight（Figures 3, 4）

**Phase 1（top row）**：
- Pre-action + PAHF 在 **iteration 1 就达到 high success rate**（pre-action warm start）
- Post-action only 从低开始，逐渐爬升
- ACPE 早期：pre-action agents 显著低于 post-action only——证明 pre-action channel 防止 initial personalization error

**Phase 3（bottom row）**：
- Pre-action only 几乎 flat，FF 接近 0（不再 ask）——典型 confidently wrong 失效
- Post-action only + PAHF 陡峭爬升——post-action corrective signal 触发 memory rewrite
- ACPE 中：pre-action only 持续高，PAHF/post-action only 快速下降

### 7.3 Ablation：simulator 质量影响（Table 2）

测试 GPT-4.1 agent × GPT-4o human、GPT-4o agent × GPT-4.1 human、GPT-4o agent × Llama-4-Scout human（Figures 11, 12）。

GPT-4.1 agent × GPT-4o human 下 PAHF embodied Phase 2 达 **82.3%**，Phase 4 **78.8%**——更强 agent backbone 进一步放大 PAHF 优势。这表明 PAHF 不是依赖某个特定 model 的 hack。

### 7.4 Ablation：memory backend（Table 3）

SQLite vs FAISS，结果几乎一致（PAHF embodied Phase 2: 70.5 vs 68.0；shopping Phase 4: 70.3 vs 70.8）。证明效果来自 feedback mechanism 本身，而非 memory 实现细节。

### 7.5 Qualitative example（Appendix E.1，极度 informative）

**PAHF 完整四阶段 trace**：

Phase 1：Avery 问 "favorite drink"，memory empty → Ask → "Herbal tea" → 写 memory → Action A 正确。

Phase 2：Avery 同样问，memory 有 "Avery prefers herbal tea as favorite morning drink" → 直接 Action A，正确。

Phase 3（drift）：Avery 同样问，memory 仍说 herbal tea → Action A → Human: "I changed my previous preference for herbal tea, but now I prefer coffee" → post-action update memory。

Phase 4：Avery 问，memory 现在是 "Avery's favorite is now coffee" → Action B 正确。

**Pre-action only 的失败 trace**（同样 Avery）：
Phase 3：memory 仍说 herbal tea，agent thought "context specifies herbal tea, I should pick up the herbal tea" → Action A → 没有 post-action update 机制 → memory 一直保留错误偏好。
Phase 4：仍 Action A，**完全没意识到 drift**。

这个对比非常 vivid 地展示了 Proposition 1 的实际表现：没有 post-action 通道，agent 在 drift 后永远 stuck 在 stale belief。

---

## 8. 与相关工作的 positioning

### 8.1 vs RLHF / DPO / KTO 系列

经典 RLHF（Christiano et al., 2017, https://arxiv.org/abs/1706.03741 ）、DPO（Rafailov et al., 2024, https://arxiv.org/abs/2305.18290 ）、KTO（Ethayarajh et al., 2024, https://arxiv.org/abs/2402.01306 ）都是 **aggregate preference** alignment——学一个 global reward model。无法处理 per-user idiosyncrasy，更无法处理 non-stationarity。Personalized RLHF（Li et al., 2024b, https://arxiv.org/abs/2402.05133 ；Tan et al., 2024, https://arxiv.org/abs/2402.04401 ；Chen et al., 2024, https://arxiv.org/abs/2410.04070 ）试图 multi-objective 或 personalized fine-tune，但仍依赖 pre-collected data + one-off fine-tuning。

PAHF 的关键不同点：**online, no pre-existing data, handles non-stationarity**。

### 8.2 vs Memory-augmented agents

MemGPT（Packer et al., 2023, https://arxiv.org/abs/2310.08560 ）、Mem0（Chhikara et al., 2025, https://arxiv.org/abs/2504.19413 ）、A-Mem（Xu et al., 2025b, https://arxiv.org/abs/2502.12110 ）、MemoryBank（Zhong et al., 2024）都强调 long-term memory、summarization、retrieval，但都假设 pre-populated user profile，不处理 online learning from scratch + drift。PAHF 的 memory design 反而 **deliberately simple**——是为了 isolate feedback channels 的效果。这些 advanced memory 可以作为 drop-in replacement。

### 8.3 vs Hindsight RLHF

作者自己的前作 RLHS（Liang et al., 2025a, https://arxiv.org/abs/2501.08617 ）和 "Machine Bullshit"（Liang et al., 2025b, https://arxiv.org/abs/2507.07484 ）指出 RLHF 训练会 induce untruthful behavior，并提出 hindsight simulation 作为 mitigation。这个 hindsight perspective 直接 motivate 了 PAHF 的 post-action feedback channel：human 在 action 后给反馈，agent 用它来 correct memory——本质是 hindsight signal。

### 8.4 vs Personalized embodied agents

TidyBot（Wu et al., 2023, https://tidybot.cs.princeton.edu ）、Personalized Instance-Based Navigation（Barsellotti et al., 2024）、Personalized Planning（Xu et al., 2025a, https://arxiv.org/abs/2502.00858 ）大多 implicit modeling + offline。Bayesian Teaching（Qiu et al., 2025, https://arxiv.org/abs/2503.17523 ）让 LLM approximate probabilistic reasoning，但仍 offline。

### 8.5 vs PREFDISCO

PREFDISCO（Li et al., 2025, https://arxiv.org/abs/2510.00177 ）benchmark interactive preference discovery，但 limit 在 static persona + short-horizon dialogues。PAHF 显式 handle non-stationarity + long-horizon sequential decisions。

---

## 9. Limitations 与未来方向（Appendix B）

1. **Memory architecture**：故意用简单 dense retrieval，未来可换 hierarchical / structured memory。
2. **Noisy feedback**：当前 salience detector 只做 basic filtering，不处理 inconsistent / mistaken feedback（用户 confused 给错反馈）。未来需要 conflict detection + clarification query + diverse noise modeling。
3. **Reasoning capability**：online shopping 仍 challenging（PAHF Phase 2 只 41.3%）。Tasks 故意 hard + sparse feedback，stress-test fine-grained reasoning。
4. **Limited disambiguation**：每 task 至多 1 个 clarification question，反映 user friction。但 shopping 中常需 sequential disambiguation。未来 explore multi-turn clarification + learned turn-allocation policies。

---

## 10. 我的 intuition 总结

读完这篇 paper，几个关键 takeaway：

**1. Two failure modes 本质上 asymmetric**：
- Partial observability 是 "known unknown"——agent 知道自己不知道，所以主动 ask 就行
- Non-stationarity 是 "unknown unknown"——agent 不知道自己不知道，pre-action channel 永远不会触发，**必须**有 reactive post-action signal

这个 asymmetry 是 Proposition 1 vs Proposition 2 背后的深层 reason。任何只做 pre-action clarification 的 system（如很多 RAG-based personalized assistant）在用户偏好改变后都会 silently fail。

**2. Explicit memory 的关键价值**：
不是 implicit latent representation，而是 **explicit natural language notes** + 检索 + 重写。这让 update 变得 localized、interpretable、fast——不需要 fine-tune，单个 LLM call 就能 rewrite memory。这种 design choice 实际上让 Theorem 1 的 $O(K)$ bound 成为可能：每次 switch 后只需 update 一个 note，不需要重训整个 model。

**3. Complementarity 是数学必然，不是工程拼凑**：
Theorem 1 证明，dynamic regret $O(K + \gamma)$ 当 $k = \log_m T$ 时——sublinear in $T$。这意味着 PAHF 在长期部署中，**regret 与 horizon 几乎无关**，只与 non-stationarity 程度 $K$ 和 ambiguity 程度 $\gamma$ 有关。这是非常强的理论 guarantee，远超经典 online learning 在 non-stationary 环境下的 bound。

**4. Benchmark 设计的精妙**：
- Phase 3 用 **structural identical** scenarios but different ground truth——cleanly isolate drift sensitivity，confounders 全部 control
- Embodied 用 logical inversion（hard shift），shopping 用 stochastic reshuffle（soft shift）——两种 drift mode 都测了
- "Near-miss poison pill" 设计让 shopping 不能靠 partial match 蒙混——必须 fine-grained verify every attribute

**5. 真实部署考量**：
限制 1 clarification/task 是为了模拟 user friction。但 Theorem 1 告诉我们 $k = \log_m T$ 才最优——实践中需要在 user burden 和 performance 间 trade-off。Multi-turn clarification + learned turn allocation 是 obvious next step。

**6. 与 RLHF 的本质区别**：
RLHF 学一个 **global** reward model，是 model-centric。PAHF 是 **memory-centric**——model frozen，所有 personalization 通过 memory read/write 完成。这种 decoupling 让 personalization 可增量、可 per-user、可 revoke——更像一个 "user profile database" 而非 "personalized model weights"。这也让 deployment 更实用：一个 base model + N 个 user memory stores，而非 N 个 fine-tuned models。

潜在 connection 我还想指出：
- 这个 framework 和 **Bayesian Theory of Mind**（Baker et al., 2011, https://proceedings.mlr.press/v9/baker11.html ）有深层联系——都把 user 视为 latent state，agent 维护 belief。但 PAHF 用 explicit memory 替代 implicit Bayesian update，更 scalable。
- 与 **Active Learning** 的 connection：pre-action clarification 本质是 active query selection。Theorem 1 的 $k = \log_m T$ 对应 active learning 中 well-known 的 exponential error reduction（Settles, 2009, https://www.morganclaypool.com/doi/10.2200/S00429ED1V01Y201207AII006 ）。
- 与 **Bandits with non-stationary rewards** 的 connection：Proposition 1 的 lower bound $\Omega(T)$ 对应 classic non-stationary bandit 中无 detection 时的 linear regret。Post-action update 类似滑动窗口 / restart 箖略。可参看 Garivier & Moulines 2008, https://arxiv.org/abs/0809.2053 。
- 与 **Memory Networks**（Graves et al., 2014, https://arxiv.org/abs/1410.5401 ；Sukhbaatar et al., 2015, https://arxiv.org/abs/1503.08895 ）的 connection：可微 memory 的 write/read 操作本质就是 PAHF 的 retrieve/update operation，但 PAHF 用 LLM 来 implement 这些 op，所以是 "neural-symbolic memory"。

希望这些细节对你 build intuition 有帮助，Andrej。如果想深挖某个 proof、某个 benchmark 的具体 scenario、或者某个 baseline 的失败模式 trace，再告诉我。
