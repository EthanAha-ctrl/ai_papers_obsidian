---
source_pdf: OpenClaw-RL_ Train Any Agent Simply by Talking.pdf
paper_sha256: aea0b6c731880bdc4d497d392a0e298d9f74516d35a2baa76b714f5daabb0d70
processed_at: '2026-08-06T00:19:10-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 OpenClaw-RL

## 一句话版本

你跟 agent 说话的时候，agent 每次回复完，你都会给个反应——重问、纠正、点个赞、骂一句。这个反应本身就在告诉 agent "你刚才那句话好不好" 以及 "应该怎么改"。**所有现成的 RLHF 系统 都把这个信号扔了，攒成 dataset 慢慢训。这篇 paper 说：别攒了，实时塞回去训。**

就这么个 idea。

---

## 为什么这个 idea obvious in hindsight

你 deploy 一个 agent，无论它是 ChatGPT、Claude Code、还是 terminal agent、GUI agent，每一次 action $a_t$ 之后都会有个 $s_{t+1}$——用户下一句话、tool 输出、test 结果、exit code。

这个 $s_{t+1}$ 是**免费的、每一步都有的、自动出现的**。但是现有系统怎么用它的？

- ChatGPT 这类对话产品：$s_{t+1}$ 就是下一轮对话的 prompt，喂回去继续生成，**当 context 用完就丢**
- SWE-agent / ReAct 这类：$s_{t+1}$ 是 tool output，塞回 context，继续走
- RLHF 训练：要专门请人标 preference pair，要几万条，攒 dataset，offline 训

这中间有个巨大的浪费——$s_{t+1}$ 本身就隐含了对 $a_t$ 的评价。用户 requery 同一个问题 = 隐含"你没答好"；test pass = "你做对了"；error trace = "你做错了"。这信号 **天生就有**，根本不需要请人标。

而且 $s_{t+1}$ 还经常带**方向信息**：用户说 "你应该先检查文件再改"，这不仅说你错了，还告诉你应该怎么改。这个方向信息比 scalar reward 丰富多了——但传统 RL 全部把它压成一个数 $\{+1, -1\}$ 扔掉。

OpenClaw-RL 的全部贡献就是：**把 $s_{t+1}$ 当成 live training signal，跨所有 agent 类型统一用**。

---

## 两个核心 trick

### Trick 1: Binary RL — 把 $s_{t+1}$ 变 scalar reward

最简单的版本。拿个 LLM 当 judge（叫 PRM），输入 $(a_t, s_{t+1})$，输出 $r \in \{+1, -1, 0\}$。

- $a_t$ 是 agent 刚才的回复
- $s_{t+1}$ 是用户接下来怎么说、tool 返回什么
- judge 推理："用户这句话像满意还是不满意？" 给个分

跑 $m$ 个 vote 取 majority，得到 $r_{\text{final}}$，直接当 advantage $A_t = r_{\text{final}}$，套 PPO clip 训练。

公式就一个：

$$\mathcal{L}_{\text{pg}} = -\mathbb{E}_t\left[\min\left(\rho_t A_t, \; \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon_{\text{high}}) \cdot A_t\right)\right]$$

变量讲一下：
- $\rho_t = \pi_\theta(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$：当前 policy 和采样 policy 在这个 action 上的概率比，超过 1 说明你比之前更爱这个 action
- $A_t$：advantage，这里就是 scalar reward $\{+1, -1, 0\}$
- $\varepsilon = 0.2$, $\varepsilon_{\text{high}} = 0.28$：PPO clip 的不对称边界，positive advantage 时 ratio 允许更大一点（DAPO https://arxiv.org/abs/2503.14476 的 trick），negative 时紧一点

**这里有个工程细节值得注意**：paper 明确说 "real-time conversational setting, no group structure available for standardization as in GRPO"。意思是——GRPO 之所以不需要 critic，是因为它对同一 prompt 采 8 个 response 形成一组，组内 standardize。但 online 部署每个 prompt 只来一次，没组可分。所以这里 advantage 就是 raw reward，没有 baseline 减。这会导致 gradient 非常 noisy，Binary RL 单独训效果差（实验里 0.17 → 0.25 → 0.23，基本没涨甚至倒退）。

这就引出 Trick 2。

### Trick 2: OPD (On-Policy Distillation) — 把 $s_{t+1}$ 变 token-level directional signal

核心观察：scalar reward 把 $s_{t+1}$ 压成一个数，浪费了 99% 的信息。"你应该先检查文件再改" 这句话比一个 $+1$ 或 $-1$ 丰富太多了——它告诉你 **哪些 token** 应该不同。

但怎么把这种文本信息变成 gradient？你没法直接从 "先检查文件" 这种自然语言拿到 token-level 的 training signal。

OPD 的 trick 非常 elegant，分四步：

**Step 1**: 让 judge 从 $s_{t+1}$ 提取一个 concise hint。不直接用 $s_{t+1}$ 因为它 noisy（用户可能既纠正又问新问题），judge 蒸馏成 1-3 句 actionable instruction。

**Step 2**: 严格 filter——只保留长度 > 10 chars 的 hint，选最长的。没好 hint 直接 drop 这个 sample。这一步让 OPD 的 sample 数量比 Binary RL 少很多，但每个 sample 的 signal quality 高得多。

**Step 3**: 把 hint 拼到原 prompt 后面，构造 $s_{\text{enhanced}} = s_t \oplus \text{hint}$。这就模拟了 "如果用户一开始就把 correction 说清楚" 的场景。

**Step 4**: 关键一步——**用同一个 model（teacher = student 自己）**，在 $s_{\text{enhanced}}$ 下，把 $a_t$ 作为 forced input 喂进去，拿到每个 token 的 log-prob。然后算 per-token advantage：

$$A_t[k] = \log \pi_{\text{teacher}}(a_t[k] \mid s_{\text{enhanced}}) - \log \pi_\theta(a_t[k] \mid s_t)$$

- $a_t[k]$ 是 response 里第 $k$ 个 token
- $\pi_{\text{teacher}}$ 和 $\pi_\theta$ 是 **同一个 model**，区别只在 context（一个看 hint 一个没看）
- $A_t[k] > 0$：看到 hint 后 model 给这个 token 更高概率 → student 应该 increase 它
- $A_t[k] < 0$：看到 hint 后 model 觉得这个 token 不太对 → student 应该 decrease 它

这个 advantage 是 **per-token directional** 的——同一个 response 里有的 token 被推 up，有的被推 down，非常 surgical。而 scalar reward 是把整个 response 推一个方向，粗糙得多。

为什么 forced input 很关键？因为如果让 teacher 自由生成，它可能生成完全不同的 response，那 log-prob 差异就混杂了 "生成 stochasticity" 和 "context 改变" 两个因素。forced input 让 teacher 必须重新评估 student 已经写的那些 token，差异纯粹来自 context（hint），signaling 干净。

**这个 trick 的本质**：teacher 不需要比 student 强，因为 teacher 就是 student。它只是在更好的 context 下激活了 student 本来 latent 就有的知识。这跟 in-context learning 的解释是同一个思路——pretraining 时 model 已经见过类似 pattern，hint 只是激活钥匙。

跟传统 distillation 比：
- 经典 distillation：strong teacher → weak student，注入新知识
- OPD：self-as-teacher，激活 latent 知识，capability ceiling 受限于 student 自己
- 跟 RLHF 比：RLHF 把 preference 压成 scalar
- 跟 DPO 比：DPO 需要 paired preference data
- OPD 不需要 external teacher，不需要 paired data，只需要 live $s_{t+1}$

### 两者结合

$$A_t = w_{\text{binary}} \cdot r_{\text{final}} + w_{\text{opd}} \cdot \left(\log \pi_{\text{teacher}}(a_t \mid s_{\text{enhanced}}) - \log \pi_\theta(a_t \mid s_t)\right)$$

默认 $w_{\text{binary}} = w_{\text{opd}} = 1$。两个 advantage 进同一个 PPO loss。

互补关系：
- Binary RL：所有 scored turn 都要，broad coverage，coarse signal
- OPD：只有能提取 hint 的 turn 才要，narrow coverage，fine signal
- 合起来：broad gradient floor + surgical directional correction

---

## 工程上的关键设计：四个异步 loop

要真正实现 "deploy 即训练"，架构必须做到 serving 和 training 互不阻塞。OpenClaw-RL 基于 slime（https://github.com/THUDM/slime），但 slime 只解耦 rollout 和 training，OpenClaw-RL 扩到四个完全独立的异步 loop：

| Loop | 实现 | 干啥 |
|---|---|---|
| Serving | SGLang | 处理 live user request，顺便收集 log-prob |
| Environment | HTTP API | 用户设备（personal）或云 sandbox（general） |
| Reward Judging | SGLang | PRM 异步打分 |
| Training | Megatron | 异步 apply gradient |

四个 loop 互不等待。Model 在 serve 下一个 request，PRM 在 judge 上一个 response，trainer 在 update weight——三个动作并行进行。

对比之前的 RL infra：
- OpenRLHF (https://arxiv.org/abs/2405.11143)、AReal (https://arxiv.org/abs/2505.24298)、veRL (https://arxiv.org/abs/2505.24298 hybridflow) 都是 engine 解耦但数据 batch 收集
- OpenClaw-RL 是真正 continuous learning from live deployment，zero interruption to serving

对 personal agent：env 就是用户手机/电脑，通过 confidential API 连 RL server，每条 request 分类成 main-line turn（可训练）或 side turn（不训练）。

对 general agent：env 是云上 sandbox，128 个 terminal env / 64 个 GUI env / 64 个 SWE env 并行跑。

日志系统是 fire-and-forget，背景线程写 JSONL，对 serving path 零 latency。每次 weight update 边界清空 log——保证 log 始终对应单一 policy 版本，方便后续数据分析。

---

## 实验告诉我们什么

### Personal Agent 模拟实验

用 LLM 模拟用户场景：
- **Student setting**：模拟学生用 OpenClaw 做 GSM8K 作业，但不想被发现用了 AI。什么是 "AI-like phrasing" 完全取决于该学生偏好。
- **Teacher setting**：模拟老师用 OpenClaw 批改作业，希望 comment 具体 friendly。

Policy 是 Qwen3-4B，每 16 个 sample 训一次。

核心结果（Table 3）：

| Method | 8 steps | 16 steps |
|---|---|---|
| Binary RL | 0.25 | 0.23 |
| OPD | 0.25 | 0.72 |
| Combined | **0.76** | **0.81** |

base 是 0.17。三个观察：

1. **Binary RL 单独基本没用**。online 没组可 standardize，scalar advantage 太 noisy，16 步反而倒退。
2. **OPD 启动慢**。8 步只到 0.25，因为严格 filter 让大量 sample 被 drop，要积累够 hint-bearing turn 才见效。
3. **Combined 一开始就猛**。Binary RL 提供 broad gradient floor 让 OPD 早期就有信号可用，OPD 的 directional signal 反过来 stabilize Binary 的 noisy scalar。这种 coarse-to-fine 的互补 pattern 在 RL 里很经典。

Figure 2 显示 student 36 次交互、teacher 24 次交互就显著见效——这是极低的 sample efficiency 阈值。传统 RLHF 要几万 preference pair。

定性证据（Appendix B）也很直观：
- Student setting 优化前：用 `**bold**`、`**Answer:**`、step-by-step 结构化——典型 AI 味
- 优化后：自然口语段落，去结构化
- Teacher setting 优化前："Correct. Well done!" 一句
- 优化后：具体指出学生哪步做得好、加 emoji、个性化鼓励

这是 personal preference 被 RL 捕获的直接证据——policy 变成 *这个特定用户喜欢的好*，不是 generic 的好。

### General Agent 实验

跨 terminal / GUI / SWE / tool-call 四个设定，不同 model size（4B 到 32B）、不同 modality（text 和 vision）。Process reward vs outcome-only 对比：

| Setting | Integrated | Outcome only |
|---|---|---|
| Tool-call | **0.30** | 0.17 |
| GUI | **0.33** | 0.31 |

Tool-call 提升 76% relative，GUI 只提升 6%。直觉解释：
- Tool-call 是 medium horizon，每步 next-state（return value / error trace）PRM 好判，dense reward 价值高
- GUI 是 long horizon，PRM 判 visual diff 本身就 noisy，很多步骤是 no-op 或探索，PRM 难给准

Trade-off：要 host PRM 模型，额外 GPU 资源。

---

## Reward 集成的工程细节

General agent 里 outcome + process reward 直接加：

$$R_t = o + \sum_{i=1}^{m} \frac{r_i}{m}$$

- $o$ 是最终 outcome reward（terminal 时给）
- $r_i$ 是第 $i$ 个 PRM vote
- $m$ 是 vote 数

但 advantage 怎么算？GRPO 在同 prompt 下 group 多 response standardize。但 real-world agent 每个 rollout 的 state 都不一样，没法 group similar states（Feng et al. 2025b https://arxiv.org/abs/2505.10978 试图做但难）。

OpenClaw-RL 的简单近似：**group actions with same step index**。所有 trajectory 的第 $k$ step 形成一个 group，组内 standardize。这实证 work，因为 long-horizon agent 在相同 step index 上的 state 在 task-progress 上 roughly comparable。

这是个 pragmatic 工程妥协，比 ArCHer（https://arxiv.org/abs/2402.19444）的 hierarchical credit assignment 轻量得多。

---

## PRM Judge Prompt 的设计哲学

Appendix C 的 prompt 值得细读，因为它们是 reward function 的实际定义。

**Personal Binary Judge**：三分类 $\{+1, -1, 0\}$，不是 binary。多一个 "neutral" 缓冲，避免 ambiguous case 强行 ±1。这个设计很关键——online live 数据 noisy，强制二分会引入大量 label noise。

**OPD Hint Extraction**：强制 hint "concrete and actionable, 1-3 sentences"。这过滤掉 "be more careful" 这种无用 generic 建议，只保留 specific instruction 比如 "check file before editing"。Hint quality 直接决定 OPD 信号质量。

**GUI Judge**：特别判 "hallucinates tools/objects/facts" 为 -1。这是 GUI agent 的常见 failure mode——agent 描述不存在的 UI 元素。用 next observation（执行 action 后的屏幕状态）判 action 是否真的生效，避免 agent "假装" 做了某事。

**SWE Judge**：判 "going in circles (repeating previously failed command)" 和 "wasting steps (reading files already fully examined)" 为 -1。这是 SWE agent 的经典浪费模式，PRM 能在 process level 抓住这种低效行为，而 outcome-only reward 完全看不到。

---

## Hyperparameter 里的魔鬼细节

Appendix D 有几个细节值得注意：

**Adam β2 = 0.98**（标准是 0.999）。RL 中 reward signal 非平稳（policy 在更新），gradient distribution 漂移大。β2 小一点让 second moment estimate 跟得上变化，避免 step size 被陈旧 gradient magnitude 拖累。这是 RL-specific 的调参。

**Entropy coef = 0.0**。RLVR 中 entropy bonus 通常 hurts——不像 game RL 需要鼓励探索，LLM 的 exploration 已经通过 sampling temperature 实现，额外 entropy bonus 会 push policy 趋向 uniform，破坏预训练 distribution。

**PRM votes m = 3 (GUI) / 1 (其他)**。GUI 难判所以多 vote 取 majority 提一致性，其他设定 judge 本身就 confident，单 vote 够了。

**Max new tokens 4096 (RL) / 8192 (OPD)**。OPD 需要 teacher 在 forced input 下重新跑 $a_t$ 取 log-prob，序列长，max token 翻倍。

---

## 我的 intuition 和 open questions

### Intuition 1: Next-State Signal 是 RLHF 一直被忽略的"零成本标注"

传统 RLHF 需要 human annotator 标 preference pair——贵、慢、scale 不起来。但 every agent deployment 每天 generate 数百万 next-state signal：每次 user 重问、每次 user 改 phrasing、每次 user 显式 correction 都是免费 reward。OpenClaw-RL 第一个把这些信号 *实时* 接入 RL training loop，而不是攒成 dataset 再 offline 训。这本质上是把 RLHF 从 batch supervised learning 退化回它本应有的 online learning 形态。

### Intuition 2: Self-as-Teacher 是 Distillation 的根本突破

OPD 中 teacher 和 student 是同一个 model，区别只在 context。Teacher 的 capability ceiling 等于 student——distillation 不是注入新知识，而是把 student latent 知识在合适 context 下激活出来，再蒸馏回原 context。这跟 in-context learning 的解释是同构的——pretraining 时 model 已经见过类似 pattern，hint 只是激活钥匙。Self-distillation 在 capability 上限受限于 student，但 directional 信息是 student 自己也信服的，gradient 方向更可信。

### Intuition 3: 不对称 PPO Clip 的直觉

ε_low = 0.2, ε_high = 0.28。positive advantage 时 ratio 允许更大一点。直觉是：positive sample（model 不太爱生成但 reward 高）是稀缺好东西，让它多 push 一点；negative sample 多且 noisy，clip 紧一些防 over-penalize。这与 DAPO (https://arxiv.org/abs/2503.14476) 思路相通。

### Open Question 1: OPD 没扩展到 General Agent

Paper 把 OPD 限制在 personal agent。General agent 只用 Binary RL + outcome。为什么？我猜测：
- SWE 的 error trace 已经结构化，Binary PRM judge 已能捕获大部分信号
- GUI 的 next-state 是 image diff，提取 textual hint 需要 visual-language judge，更复杂
- Long-horizon 下 hint-augmented context 重新跑 forced input 在 30 step × 8192 token 上 forward cost 不低

但理论上 SWE 的 detailed error trace 极适合 OPD——"you should have used `git diff` instead of `cat`" 是典型 directive hint。这是明显的 future work 方向。

### Open Question 2: Simulated LLM 评估的信赖度

Personal agent 实验用 LLM 模拟 student/teacher 评分。Simulated LLM 的 preference 与真实 user preference 一致性是多少？如果实际部署，需要 real user study validate。Appendix C.3 的 evaluative prompt 把 score 限定在 $\{0, 0.25, 0.5, 0.75, 1\}$ 五档，这种离散 scale 限制评估粒度。

### Open Question 3: Hint Selection 的 Bias

OPD 选 longest valid hint，隐含假设 "longer = more informative"。但 long hint 可能 verbose 而非 informative。是否应该用其他 quality signal，比如让另一个 judge 给 hint 打分？

### Open Question 4: 长期稳定性

Table 3 显示 Binary RL 16 步从 0.25 降到 0.23——instability 信号。Combined 8 步就到 0.76 说明组合策略 early-stable，但 paper 没展示更长训练（比如 64 步）的曲线。长期是否会出现 reward hacking 或 distribution collapse？

### Open Question 5: OPD 的 Hint Distribution Shift

Hint 由 judge 提取，judge 本身是另一个 LLM。Judge 的 hint 分布会不会有 systematic bias？比如总是偏好某种 correction style？这会让 student policy 慢慢 drift 到 judge 偏好的 style，而不是真正 user 偏好的 style。这是 self-referential training 的经典风险（参考 Self-Rewarding Language Models https://arxiv.org/abs/2401.10020 的类似讨论）。

---

## 与 Concurrent Work 的对比

Paper 提到 Buening et al. 2026 (https://thomasklbg.github.io/files/Aligning_Language_Models_from_User_Interactions.pdf) 也从 user interaction 学，但做法是把 next-state 直接塞 prompt 让 model 自己 figure out，correction 是 implicit 的。

OpenClaw-RL 的 OPD 多了一步——让 judge 把 next-state 蒸馏成 explicit textual hint，再塞 enhanced context。Explicit hint 让 teacher distribution 的 shift 更明确、更 directional，而不是 model 自由解释 next-state 产生的 ambiguous shift。这是工程上更重但信号更干净的设计。

---

## 最终判断

这篇 paper 抓住了一个所有人都见过但没人认真对待的观察。$s_{t+1}$ 一直在那里，免费、密集、跨所有 agent 类型通用。从 RLHF 到 DPO 到 GRPO 到 DAPO，整个 LLM alignment 领域都在 batch data paradigm 里打转——怎么标数据更好、怎么用 preference 更高效、怎么 group 标准化。OpenClaw-RL 说：别标了，生产环境每天都在产生千万级 next-state signal，直接接进 training loop。

两个 trick 也各司其职：
- Binary RL 是 floor，广覆盖但粗糙
- OPD 是 surgical knife，精准但稀疏
- 合起来既有广度又有分辨率

四 loop 异步 infra 让 "deploy 即训练" 真正可行，没有 batch 收集阶段，serving 完全不被阻塞。

这是我看到的第一个认真把 LLM 从 "train then deploy" 的 batch paradigm 推向 "deploy is training" 的 online learning paradigm 的工作。如果这个方向成立，下一个十年的 alignment 不靠更大的 preference dataset，靠的是把 every deployed token 当 training data。

参考链接汇总：
- OpenClaw-RL: https://github.com/Gen-Verse/OpenClaw-RL
- slime: https://github.com/THUDM/slime
- PPO: https://arxiv.org/abs/1707.06347
- GRPO: https://arxiv.org/abs/2402.03300
- DAPO: https://arxiv.org/abs/2503.14476
- DPO: https://arxiv.org/abs/2305.18290
- On-policy distillation (Agarwal): https://arxiv.org/abs/2310.12948
- RL via self-distillation (Hübotter): https://arxiv.org/abs/2601.20802
- Buening et al. 2026: https://thomasklbg.github.io/files/Aligning_Language_Models_from_User_Interactions.pdf
- Self-Rewarding: https://arxiv.org/abs/2401.10020
- ArCHer: https://arxiv.org/abs/2402.19444
- RLAnything: https://arxiv.org/abs/2602.02488
- OpenRLHF: https://arxiv.org/abs/2405.11143
- AReal: https://arxiv.org/abs/2505.24298
- Feng et al. 2025b (group-in-group): https://arxiv.org/abs/2505.10978

---

# OpenClaw-RL 深度解析：从 Next-State Signal 中榨取 Live Learning 信号

这篇 paper 在我看来抓住了一个被长期忽视但又极其显然的观察——每一次 agent action 之后都会自然产生一个 next-state signal $s_{t+1}$（user reply / tool output / terminal state / GUI diff / test verdict），所有现有 agentic RL 系统都把它当成"下一轮 prompt 的一部分"就丢掉了，从没有人把它当作 live、online、跨 interaction type 的统一训练信号来用。OpenClaw-RL 的全部贡献都建立在 reframe 这一点上。

项目主页：https://github.com/Gen-Verse/OpenClaw-RL
基础设施 slime：https://github.com/THUDM/slime

---

## 1. Core Insight：把 $s_{t+1}$ 重新当作 Reward 与 Teacher

作者区分了 next-state signal 中隐含的两种 wasted information：

### Waste 1 — Evaluative Signals
$s_{t+1}$ 隐式对 $a_t$ 进行评分。例如：
- 用户 re-query（重问）→ 隐含 dissatisfaction
- terminal exit code = 0 → success
- SWE test pass → success
- error trace → failure
- GUI state diff 显示 progress → good

这种信号对每个 turn 都是免费的 process reward，不依赖 ground-truth annotation。但传统 PRM 几乎只在 math reasoning with verifiable ground truth 中研究（Math-Shepherd, Lightman et al., OpenAI PRM800K），从未被作为 online live deployment 中的 dense per-step reward 来源。

### Waste 2 — Directive Signals
$s_{t+1}$ 不仅告诉你 $a_t$ 好不好，还告诉你它本应如何不同。比如用户说 "you should have checked the file before editing it"，这给出了 token-level 的修正方向。SWE 的 detailed error trace 同样暗示具体修正方向。这是 scalar reward 永远无法编码的信息维度。

这个二分法非常关键，因为 RLHF / DPO / GRPO 全部把 preference 压缩为 scalar，而 distillation 又依赖 pre-curated teacher 数据。OpenClaw-RL 同时解决两个问题。

---

## 2. Problem Setting：把异构 Interaction Stream 统一进 MDP

作者把每一个 interaction stream（personal conversation、terminal、GUI、SWE、tool-call）都形式化为同一个 MDP $(S, \mathcal{A}, \mathcal{T}, R)$：

- **State** $s_t \in S$：到第 $t$ 轮为止的完整对话/环境上下文
- **Action** $a_t \in \mathcal{A}$：agent 生成的 token 序列
- **Transition** $\mathcal{T}(s_{t+1} \mid s_t, a_t)$：在环境中是 deterministic 的，$s_{t+1}$ 就是 user reply / execution result / tool output
- **Reward** $r(a_t, s_{t+1})$：由 PRM judge 从 next-state 中推断

注意这里的标准 RLVR 设定中 outcome $o$ 只在 trajectory 结尾给出，而 process reward $r(a_t, s_{t+1})$ 在每个 step 都有——这个 dense reward 才是 long-horizon agent 训练真正需要的（参考 RLAnything: https://arxiv.org/abs/2602.02488）。

---

## 3. Infrastructure：四组件全解耦 Asynchronous Pipeline

### 3.1 架构核心

OpenClaw-RL 基于 slime（https://github.com/THUDM/slime）构建，但 slime 只有 rollout + training 两个解耦 loop。OpenClaw-RL 进一步把它拆为 **四个完全独立的异步 loop**：

| 组件 | 实现 | 职责 |
|---|---|---|
| Policy Serving | SGLang | 处理 live request，同步收集 log-prob |
| Environment Server | HTTP/API | 个人设备（personal）或云上 sandbox（general） |
| Reward Judging | SGLang/API | PRM judge 异步打分 |
| Policy Training | Megatron | 异步应用 gradient update |

四个 loop 之间没有 blocking dependency，意思是：
- Model 在 serve 下一个 user request 的同时
- PRM 在 judge 上一个 response 的同时
- Trainer 在 apply gradient update 的同时

这是连续训练 live interaction 的核心前提。任何 stream 都不需要 pause 来等其他组件。这对比 OpenRLHF（https://arxiv.org/abs/2405.11143）、AReal（https://arxiv.org/abs/2505.24298）、veRL（https://arxiv.org/abs/2505.24298 hybridflow）都假设 batch data collection——它们本质上是 offline-style，只是 engine 解耦，数据仍是预收集的。

### 3.2 Session-Aware Environment Server for Personal Agents

Personal agent 的 env 是 user 的 device，通过 confidential API 连到 RL server。每个 API request 被分类：

- **Main-line turn**：agent 的主响应 + tool 执行结果 → 可训练样本
- **Side turn**：auxiliary query、memory organization、env transition → 转发但不训练

新 main-line request 的 message 包含对上一轮的 reaction（user reply 或 execution result），这就成为上一轮的 $s_{t+1}$，用作 reward 计算的输入。

### 3.3 Non-Blocking Logging

所有 interaction 与 reward 评估实时写 JSONL：full message history、prompt/response、tool calls、next-state content、PRM per-vote 分数、选中的 hint、accept/reject 决策。日志 fire-and-forget 写到 background thread，对 serving/PRM path 零 latency 影响。每次 weight update 边界处清空 log，确保日志始终对应单一 policy 版本——这一点对 reproducibility 与数据分析很重要。

---

## 4. Binary RL：从 $s_{t+1}$ 提取 Scalar Process Reward

### 4.1 PRM Judge 构造

给定 response $a_t$ 和 next state $s_{t+1}$，judge 模型评估 $a_t$ 质量：

$$\mathrm{PRM}(a_t, s_{t+1}) \to r \in \{+1, -1, 0\}$$

- tool-call results 通常结论清晰
- user reply 可能包含 satisfaction/dissatisfaction 信号
- 没有明确信号时 model 基于场景 estimate（鼓励用户 explicit feedback）
- general agent 的 judge 推理 environment feedback 是否表明向 task goal 推进

跑 $m$ 个独立 query，取 majority vote：

$$r_{\mathrm{final}} = \mathrm{MajorityVote}(r_1, \ldots, r_m)$$

$m$ 在 GUI 设定为 3，其他设定为 1（见 Appendix D Hyperparameters）。

### 4.2 RL Training Objective：Asymmetric PPO Clipped Surrogate

直接令 advantage $A_t = r_{\mathrm{final}}$，采用 Schulman 2017 PPO clipped surrogate，但用 asymmetric bounds（这是 DAPO https://arxiv.org/abs/2503.14476 的关键 trick）：

$$\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\mathrm{old}}(a_t \mid s_t)}$$

$$\mathcal{L}_{\mathrm{pg}} = -\mathbb{E}_t\left[\min\left(\rho_t A_t, \; \mathrm{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon_{\mathrm{high}}) \cdot A_t\right)\right]$$

$$\mathcal{L} = \mathcal{L}_{\mathrm{pg}} + \beta_{\mathrm{KL}} \cdot \mathcal{L}_{\mathrm{KL}}$$

变量解析：
- $\rho_t$：importance sampling ratio，新策略 $\pi_\theta$ 与采样策略 $\pi_{\mathrm{old}}$ 在 action $a_t$ 上的概率比
- $\varepsilon = 0.2$：lower clip bound，防止 positive advantage 时 over-update
- $\varepsilon_{\mathrm{high}} = 0.28$：upper clip bound，比下界稍大；asymmetric 设计的直觉来自 DAPO——negative advantage 的样本（坏样本）往往 model 已经 assign 低概率，$\rho_t$ 自然小，过度 clip 会浪费 signal；positive advantage 则容易 over-trust 单一样本
- $\beta_{\mathrm{KL}} = 0.02$：KL penalty 系数，防止 policy 漂离 reference 太远
- $\mathcal{L}_{\mathrm{KL}}$：当前 policy 与 reference policy 之间的 KL 散度

关键设定差异：personal agent 设定中 $\beta_{\mathrm{KL}} = 0$（因为 online live setting，不需要 KL anchor），general agent 用 $\beta_{\mathrm{KL}} = 0.01$。

**这里有个很重要的观察**：paper 明确说 "this is a real-time conversational setting, so there is no group structure available for standardization as in GRPO"。GRPO 之所以能 eliminate critic，是因为它对同一 prompt 采样多个 response 形成 group，group 内 standardize 当作 advantage。但 online live deployment 每个 prompt 只来一次，没有 group。所以这里只能直接用 $A_t = r_{\mathrm{final}} \in \{+1, -1, 0\}$，没有 baseline subtraction。这是 online vs batch RL 一个结构性 tradeoff。

参考 GRPO: https://arxiv.org/abs/2402.03300
参考 PPO: https://arxiv.org/abs/1707.06347
参考 DAPO asymmetric clip: https://arxiv.org/abs/2503.14476

---

## 5. Hindsight-Guided On-Policy Distillation (OPD)：从 $s_{t+1}$ 提取 Token-Level Directional Advantage

这是这篇 paper 最有意思的部分。

### 5.1 为什么 Scalar Reward 不够

Binary RL 把 $s_{t+1}$ 的全部信息压成一个 scalar $r \in \{+1, -1, 0\}$。但用户写 "you should have checked the file before editing it" 传达的信息远远超过 good/bad——它告诉你 *哪些 token* 应该不同以及如何不同。这个 directive 信息在 scalar reward 中完全丢失。

OPD 的核心 insight：如果我们在原始 prompt 上 augment 一个从 $s_{t+1}$ 提取的 textual hint，同一个 model 在这个 enhanced context 下会产生 *不同的 token 分布*——这个分布"知道" response 本应该是什么。两个分布的 per-token gap 给出 directional advantage。

这与三个传统范式都不同：
- RLHF（https://arxiv.org/abs/1706.03741）：scalar preference
- DPO（https://arxiv.org/abs/2305.18290）：需要 paired preference
- Standard distillation：需要更强的 separate teacher model

OPD 是 **self-distillation via context enrichment**，teacher 就是 student 自己，只不过看到了 hindsight hint。

### 5.2 Token-Level OPD 四步走

**Step 1 — Hindsight Hint Extraction**

$$\mathrm{Judge}(a_t, s_{t+1}) \to \{\mathrm{score} \in \{+1, -1\}, \; \mathrm{hint} \in \mathcal{T}^*\}$$

如果 $\mathrm{score} = +1$，judge 在 `[HINT_START]...[HINT_END]` 中产生一个简短 hint。跑 $m$ 个并行 judge call。

**关键设计选择**：不直接用 $s_{t+1}$ 当 hint。原始 next-state 信号常有 noise、verbose、无关信息（用户 reply 可能既包含 correction 又包含一个不相关的 new question）。Judge model 把 $s_{t+1}$ 蒸馏成一个 concise、actionable 的 instruction，通常 1–3 句，聚焦 "response 本应如何不同"。

**Step 2 — Hint Selection & Quality Filtering**

在 positive vote 中挑 length > 10 chars 的 hint，选最长的（信息最丰富）。如果没有任何 valid hint，**直接 drop 整个 sample**。

这是有意为之的 tradeoff：
- **Binary RL**：接受所有 scored turn，broad coverage + coarse signal
- **OPD**：只接受有清晰 directional signal 的 turn，narrow coverage + high-resolution signal

两者完全 complementary。

**Step 3 — Enhanced Teacher Construction**

把 hint append 到 last user message：

$$s_{\mathrm{enhanced}} = s_t \oplus \text{[user's hint / instruction]}\backslash n\{\mathrm{hint}\}$$

这就模拟了 "如果用户一开始就给 correction" 的场景。

**Step 4 — Token-Level Advantage**

在 $s_{\mathrm{enhanced}}$ 下用 forced input $a_t$ query policy model，计算每个 response token 的 log-prob：

$$A_t = \log \pi_{\mathrm{teacher}}(a_t \mid s_{\mathrm{enhanced}}) - \log \pi_\theta(a_t \mid s_t)$$

- $A_t > 0$：teacher（知道 hint）给这个 token 更高概率 → student 应该 increase
- $A_t < 0$：teacher 认为这个 token 在 hint 下不太合适 → student 应该 decrease

**对比 scalar advantage**：scalar 把整个 response 推一个方向；OPD 在同一个 response 内有的 token 被 reinforce，有的被 suppress——这是真正 per-token 的 directional guidance。

Training 用同一个 clipped surrogate（公式 1），但 advantage 现在携带更丰富信息。

### 5.3 这与 Hindsight Relabeling 的关系

OPD 实际上 unifies 三个 thread：
1. **Hindsight relabeling**：用 retrospective information 重新标注过去经验（HER, STaR https://arxiv.org/abs/2203.14465, HIR https://arxiv.org/abs/2302.02998, Self-Rewarding https://arxiv.org/abs/2401.10020）
2. **Context enrichment**：augment prompt with structured info（Buffer of Thoughts https://arxiv.org/abs/2406.04309, SuperCorrect https://arxiv.org/abs/2411.19805）
3. **On-policy distillation**：从 self-generation 学（Agarwal et al. 2024 https://arxiv.org/abs/2310.12948, Hübotter et al. 2026 https://arxiv.org/abs/2601.20802, Shenfeld et al. 2026 https://arxiv.org/abs/2601.19897）

但前两者都依赖 fixed dataset，OPD 在 online live setting 完成。Agarwal 2024 的 on-policy distillation 也需要 pre-collected feedback-response pairs；OPD 是从 live $s_{t+1}$ 提取。

---

## 6. Combine Binary RL + OPD：Weighted Advantage

两者的 advantage 都进入同一个 PPO loss，只是 advantage 计算方式不同。直接加权组合：

$$A_t = w_{\mathrm{binary}} r_{\mathrm{final}} + w_{\mathrm{opd}} \left(\log \pi_{\mathrm{teacher}}(a_t \mid s_{\mathrm{enhanced}}) - \log \pi_\theta(a_t \mid s_t)\right)$$

默认 $w_{\mathrm{binary}} = w_{\mathrm{opd}} = 1$。

**互补性分析表（Table 2 详解）**：

| Dimension | Binary RL | OPD | Combined |
|---|---|---|---|
| Signal type | Evaluative (good/bad) | Directional | Evaluative + directional |
| Advantage | Sequence-level scalar | Token-level directional | Mixed |
| Density | All scored turns | Hint-accepted only | All scored turns |
| Feedback type | User / environment | Explicit corrections | Implicit + explicit |
| Signal richness | 1 scalar per sample | 1 value per token | 1 value per token |

直觉：Binary RL 是 broad blanket gradient，覆盖所有 turn 但粗糙；OPD 是精准 surgical knife，只在能 recover directive signal 的 turn 上提供 per-token supervision。两者梯度方向不会 conflict——前者推 sequence-level好坏，后者在 token level 微调，组合后既有广度又有分辨率。

---

## 7. General Agentic RL：Step-wise Reward Integration

### 7.1 Why Process Rewards Vital for Long-Horizon

Long-horizon agent 任务中，outcome-only reward 只在 terminal step 提供梯度，前面绝大多数 turn 完全 unsupervised。PRM 给每个 turn 基于 next-state signal 打分，提供 dense credit assignment。RLAnything 已经给出 strong empirical evidence：step-wise PRM + outcome 在 GUI agent、text-game agent、coding task 上都 outperform outcome-only。

### 7.2 Outcome + Process Reward 的组合

直接相加：

$$R_t = o + \sum_{i=1}^{m} \frac{r_i}{m}$$

作为 step $t$ 的 reward。其中：
- $o$：最终 outcome reward（terminal 时给出）
- $r_i$：第 $i$ 个 PRM vote 给的 process reward
- $m$：vote 数

**Advantage 计算的关键 trick**：GRPO 在 prompt 层 group 同 state 下的 multiple action 来 standardize。但 real-world agent env（如 terminal）state 难以聚类——每个 rollout 的 state 都不同。Feng et al. 2025b（https://arxiv.org/abs/2505.10978）建议 group similar states。OpenClaw-RL 采取更简单的近似：**group actions with same step index**。即所有 trajectory 的第 $k$ step 形成一个 group，group 内 standardize。这在实证上 work，因为 long-horizon agent 在相同 step index 上的 state 在 task-progress 上 roughly comparable。

这种 group-by-step-index 的方式让我想起 ArCHer（https://arxiv.org/abs/2402.19444 hierarchical RL for agent）和 LOOP（https://arxiv.org/abs/2502.01600）的 multi-turn credit assignment 思路，但更工程化、更轻量。

---

## 8. Experiments 深度分析

### 8.1 Personal Agent 模拟实验

两个 setting 用 LLM 模拟 user：

**Student Setting**：LLM 扮演学生用 OpenClaw 做 GSM8K 作业，但不想被发现用了 AI。这本质上是一个 **personal preference alignment** 任务——什么是 "AI-like phrasing" 完全取决于该 student 的写作风格。Qwen3-4B 作为 policy，lr = 1e-5，KL = 0，每 16 个 sample trigger 一次训练。

**Teacher Setting**：LLM 扮演老师用 OpenClaw 批改作业，希望 comment 具体 friendly。同样 Qwen3-4B。

### 8.2 关键结果 1：组合方法的协同效应（Table 3）

Base score = 0.17（policy 起始分）：

| Method | Updated 8 steps | Updated 16 steps |
|---|---|---|
| Binary RL | 0.25 | 0.23 |
| OPD | 0.25 | 0.72 |
| Combined | **0.76** | **0.81** |

观察：
1. **Binary RL 几乎没用**——8 步到 16 步反而从 0.25 降到 0.23。在 online live personal setting，没有 group standardize，scalar reward 单独驱动 gradient 非常 noisy。
2. **OPD 延迟启动**——8 步只到 0.25，16 步跳到 0.72。原因 paper 给出："sparsity of training samples"。OPD 的 strict filtering 让大量 sample 被 drop，需要积累足够 hint-bearing turn 才能看到效果。
3. **Combined 一开始就猛**——8 步直接到 0.76。Binary RL 的 broad coverage 给 OPD 提供了早期梯度 floor，OPD 的 directional signal 又让 Binary 的 noisy scalar 不至于发散。这是 **互补** 的实证体现。

这个 pattern 在 RL 中很经典：coarse signal 提供 warm-up，fine signal 提供 late-stage refinement。

Figure 2 显示 student setting 只需 36 次 problem-solving interaction，teacher setting 只需 24 次批改 interaction 就显著见效。这是一个 **极低 sample efficiency 阈值**——传统 alignment pipeline 需要几万到几十万 preference pair。

### 8.3 Personalization 质性证据（Appendix B）

**Student Setting Example 1 Before**：用 `**bold**`、`**Answer:**`、结构化 step-by-step——明显 AI-like。
**After**：写成自然口语段落 "The answer is 100%. Here's the breakdown..." 完全去结构化。

**Teacher Setting Example 2 Before**：就一句 "Correct answer: 189 hours." 
**After**：具体表扬学生 "You correctly converted 3 weeks to 21 days - that's a key step many students miss!" 加 emoji、加个性化鼓励。

这是 personal preference 被 RL 捕获的直接证据——policy 不仅变"好"，而是变成 *这个特定用户喜欢的好*。

### 8.4 General Agent 跨设定结果（Figure 4 + Table 4）

| Setting | Model | Env Parallel | Dataset |
|---|---|---|---|
| Terminal | Qwen3-8B | 128 | SETA RL |
| GUI | Qwen3VL-8B-Thinking | 64 | OSWorld-Verified |
| SWE | Qwen3-32B | 64 | SWE-Bench-Verified |
| Tool-call | Qwen3-4B-SFT | 32 | DAPO RL data |

**Process vs Outcome Reward 对比（Table 4）**：

| Setting | Integrated (outcome + process) | Outcome only |
|---|---|---|
| Tool-call (250 steps) | **0.30** | 0.17 |
| GUI (120 steps) | **0.33** | 0.31 |

Tool-call 的提升幅度 76% relative 提升，远大于 GUI 的 6%。直觉解释：
- Tool-call 是 medium horizon（Table 1），每步的 next-state signal 是 return value / error trace，PRM 容易判，dense reward 价值高。
- GUI 是 long horizon，PRM 在视觉状态 diff 上判 progress 本身就 noisy，且 GUI 任务很多步骤是 no-op 或探索，PRM 难给准确分。

**Trade-off**：hosting PRM 需要额外 GPU 资源。这是 dense reward 的代价。

---

## 9. Pseudocode 解读（Appendix A）

### Algorithm 1: Binary RL Pipeline

```
1: a_t, logp_old ← SGLang(M_t)        # serve & 收集 log-prob
2: Buffer{prompt_ids, response_ids, logp_old} for (T, t)
4: s_{t+1} ← first message of M_{t+1}  # 下一轮的 user reply 或 env feedback
5: {r_i}_{i=1}^m ← PRM(a_t, s_{t+1})   # m 个并行 vote, async
6: r ← MajorityVote({r_i})
7: A_t ← r (broadcast to all tokens)   # 每个 token 都用同一个 scalar
8: Apply "at-least-one guarantee" if zero effective samples in T
9: Submit Sample to trainer queue
```

Step 8 "at-least-one guarantee" 是个工程 trick——如果整个 trajectory 没有任何 effective sample（比如所有 vote 都是 0），强制至少保留一个 sample 以维持训练节奏。

### Algorithm 2: OPD Pipeline

```
1: {(score_i, hint_i)} ← Judge(a_t, s_{t+1})
2: valid ← {h_i : score = +1 ∧ |h| > 10}
3: if valid = ∅: drop sample, return
7: hint ← argmax_{h ∈ valid} |h|       # 选最长的
8: s_enhanced ← M_t ⊕ "[user's hint]\n{hint}"
9: logp_teacher ← Teacher(a_t | s_enhanced)  # forced input, 取 log-prob
10: A_t[k] ← log π_teacher[k] - logp_old[k]   # per-token advantage
11: Submit Sample(teacher_log_probs=A_t) to trainer queue
```

Step 9 的 forced input 是关键：不让 teacher 自由生成，而是把 student 已经生成的 $a_t$ 喂回去，让 teacher 在 hint-augmented context 下给每个 token 打分。这样得到的 log-prob 差异纯粹来自 context 改变（hint 加入），不来自生成 stochasticity，signaling 干净。

---

## 10. PRM Judge Prompt 设计（Appendix C）

值得仔细看这几个 prompt，因为它们是 reward function 的实际定义。

### Personal Binary RL Judge
- 输入：assistant output + subsequent user reply
- 输出：`\boxed{1}` (good), `\boxed{-1}` (bad), `\boxed{0}` (neutral)
- 三分类，比 binary 多一个 "neutral" 缓冲，避免 ambiguous case 强行 ±1

### OPD Hint Extraction Prompt
- Decide whether next state reveals useful hindsight
- `\boxed{1}` + `[HINT_START]...[HINT_END]` 写 hint
- `\boxed{-1}` no hint
- **constraint**: "concrete and actionable (1-3 sentences)"

这个 "concrete and actionable" 约束非常重要——它强制 hint 不是 generic 建议而是 specific instruction。比如 "be more careful" 这种无用 hint 会被排除。

### Terminal Agent Judge
判 +1 if ALL true:
- assistant message is correct/helpful step advancing task
- tool usage appropriate
- tool results consistent with progress

判 -1 例子：
- tool-call format broken (invalid JSON / parse error)
- tool usage clearly wrong or irrelevant
- tool results show failure

### GUI Agent Judge
特别之处：用 next observation AFTER executing action 判断 action 是否生效。
判 +1 if ALL:
- step clearly relevant to objective
- action executable & coherent given next observation
- next observation shows concrete progress (not just no-op)

判 -1 例子包括 "hallucinates tools/objects/facts"——这在 GUI agent 中是常见 failure mode，agent 可能描述不存在的 UI 元素。

### SWE Agent Judge
判 +1 if ALL:
- command executed without unexpected errors (returncode=0 或 expected non-zero 如 grep no match)
- step clearly relevant to diagnosing/fixing issue
- output provides useful info OR edit makes logically correct change

判 -1 例子包括 "going in circles (repeating previously failed command)" 和 "wasting steps (reading files already fully examined)"。这是 SWE agent 的经典浪费模式。

---

## 11. Hyperparameter Table 解读（Appendix D）

| Parameter | Value | Note |
|---|---|---|
| Learning rate | 1e-6 | constant decay |
| Weight decay | 0.1 | |
| Adam β1, β2 | 0.9, 0.98 | β2 比常见的 0.999 小，可能因为 RL 中 gradient noisy，更小 β2 让 second moment 估计更 responsive |
| KL coef β_KL | 0.01 | k3 / low-var KL |
| Clip ε / ε_high | 0.2 / 0.28 | asymmetric PPO |
| Entropy coef | 0.0 | disabled——RLVR 中 entropy bonus 通常 hurts |
| Rollout batch | 8/16/32 | GUI/SWE > terminal > tool-call 反过来，因为 horizon 长的 env 单 rollout 慢 |
| Sample per task | 8 | 每个 task 8 个 rollout 用于 group statistics |
| Max response | 8192 tokens | |
| Max context | 16384 tokens | |
| Max interactive steps | 30/20/10 | GUI > SWE > terminal，因为 GUI 单步信息少，需要更多步 |
| Temperature | 1.0 | 探索时高 temp |
| PRM votes m | 3 (GUI) / 1 (其他) | GUI 难判所以多 vote |
| PRM temperature | 0.6 | judge 用较低 temp 提一致性 |
| Max new tokens | 4096 (RL) / 8192 (OPD) | OPD 需要 teacher 重新跑 $a_t$ 取 log-prob，序列长 |
| Min hint length | 10 chars | OPD quality filter |

β2 = 0.98 这个细节很值得注意。Adam 标准 β2 = 0.999 对 supervised learning 友好，因为 gradient distribution 稳定。RL 中 reward signal 是非平稳的（policy 在更新），gradient 方差大，β2 小一点让 second moment estimate 跟得上变化，避免 step size 被陈旧的 gradient magnitude 估计拖累。

---

## 12. 与 Concurrent Work Buening et al. 2026 的对比

Paper 在 introduction 提到 Buening et al. 2026 (https://thomasklbg.github.io/files/Aligning_Language_Models_from_User_Interactions.pdf) "improves online policy by directly prompting with next-state information, but the corrective hints remain implicit"。

这里有个细微但重要的区别：Buening 的做法是把 next-state 直接塞进 prompt 让 model 自己 figure out，correction 是 *implicit* 在 context 里。OpenClaw-RL 的 OPD 多了一步——让 judge 把 next-state 蒸馏成 explicit textual hint，再塞进 enhanced context。Explicit hint 让 teacher distribution 的 shift 更明确、更 directional，而不是 model 自由解释 next-state 可能产生的 ambiguous shift。

---

## 13. 我的 Intuition Building 与 Open Questions

读完整篇 paper，我建立几条 intuition：

### Intuition 1：Next-State Signal 是 RLHF 一直被忽略的"零成本标注"

传统 RLHF 需要 human annotator 标 preference pair——昂贵、慢、scale 不起来。但 every agent deployment 每天 generate 数百万 next-state signal：每次 user 重问、每次 user 改 phrasing、每次 user 显式 correction 都是免费 reward。OpenClaw-RL 第一个把这些信号 *实时* 接入 RL training loop，而不是攒成 dataset 再 offline 训练。这本质上是把 RLHF 从 batch supervised learning 退化回它本应有的 online learning 形态。

### Intuition 2：OPD 的 "self-as-teacher" 是 Distillation 的根本突破

Distillation 经典定义是 strong teacher → weak student。但 OPD 中 teacher 和 student 是同一个 model，区别只在 context（hint 或没 hint）。这意味着 *teacher 的能力 ceiling 等于 student*——distillation 不是注入新知识，而是把 student latent 知识在合适 context 下激活出来，再蒸馏回原 context。

这让我想起 In-context Learning 的解释：model 在 pretraining 时已经见过类似 pattern，in-context example 只是激活它。OPD 把这个 logic 用于训练——hint 是激活钥匙，log-prob 差异揭示哪些 token 是激活后"本应"高概率的。这种 self-distillation 在 capability 上限受限于 student，但 directional 信息是 student 自己也信服的。

### Intuition 3：Asymmetric PPO Clip 是 RLVR 的 small but important upgrade

ε_low = 0.2, ε_high = 0.28 这个 asymmetry 来自 DAPO，但 paper 里没有详细 intuition 解释。我的理解：positive advantage 样本（model 不太愿意生成但 reward 高）一旦被信赖，policy 想大幅提升 prob，但单一 positive sample 可能 noise，所以 ε_high 稍宽（0.28）允许更大 update——但这是 positive 而不是 negative。仔细想想——其实 paper 里写的是 "asymmetric bounds"，但直觉可能是 opposite：negative sample 多，positive sample 稀缺，positive 应该更保守？

Actually 读 DAPO 原文（https://arxiv.org/abs/2503.14476），DAPO 的 asymmetric 是 *去掉 lower bound*（即 negative advantage 的 clip）——叫 "Dynamic sAmpling"，避免 model collapse 到低概率 mode。OpenClaw-RL 这里 ε_high = 0.28 比 ε_low = 0.2 大，意味着 positive advantage（$\rho_t > 1$）允许更大 ratio——直觉是：positive sample 是好东西，让它多 push 一点；negative sample 是 noise source，clip 紧一些防 over-penalize。这与 DAPO 思路是相通的。

### Open Question 1: OPD 在 Long-Horizon General Agent 上的应用

Paper 把 OPD 限制在 personal agent track。General agent（terminal/GUI/SWE/tool-call）只用 Binary RL + outcome。为什么不把 OPD 扩展到 general agent？我猜测：
- SWE 的 error trace 已经结构化，Binary PRM judge 已能捕获大部分信号
- GUI 的 next-state 是 image diff，提取 textual hint 需要 visual-language judge
- Long-horizon 下 hint-augmented context 重新跑 forced input 在 30 step × 8192 token 上 forward cost 不低

但理论上 SWE 的 detailed error trace 极适合 OPD——"you should have used `git diff` instead of `cat`" 是典型 directive hint。这是 future work 方向。

### Open Question 2: Personal Agent Setting 的 Evaluation 信赖 Simulated LLM

Personal agent 实验用 LLM 模拟 student/teacher 评分。这是 reasonable 的 setup（real user study 难大规模），但 simulated LLM 的 preference 与真实 user 的 preference 一致性是多少？Appendix C.3 的 evaluative prompt 把 score 限定在 $\{0, 0.25, 0.5, 0.75, 1\}$ 五档——这种离散 scale 限制评估粒度。如果实际部署，需要 real user study validate。

### Open Question 3: OPD 的 Hint Distribution Bias

OPD 选 longest valid hint。这隐含假设 "longer hint = more informative"。但 long hint 可能 verbose 而非 informative——比如重复说 "you should be more careful and pay attention to detail and check things thoroughly"。是否应该用其他 quality signal（比如让另一个 judge 给 hint 打分）？

### Open Question 4: Sample Efficiency 与 Stability 的 Trade-off

Table 3 显示 Binary RL 在 16 步反而从 0.25 降到 0.23——这是 instability 信号。Online live setting 没有 group standardize，scalar reward 训练不稳定。OPD 提供稳定 directional signal 但需要积累（8 步看不到效果）。Combined 8 步就到 0.76 说明组合策略 early-stable，但 paper 没有展示更长训练（比如 64 步）的曲线——长期是否会出现 reward hacking 或 distribution collapse？

---

## 14. 与其他系统的关系图

```
              RLHF (scalar preference)
                 |
       ┌─────────┴─────────┐
       |                   |
      DPO                 PPO
   (paired pref)      (online RL)
       |                   |
      GRPO             PPO-clip
   (no critic,      (OpenClaw-RL 用
    group base)     asymmetric clip)
       |                   |
   DeepSeek-R1          OpenClaw-RL
   DAPO                  |
                    ┌────┴────┐
                    |         |
                 Binary RL   OPD
                 (scalar)  (token-level
                            self-distill
                            via hint)
```

PRM 系：
- Math-Shepherd (MC estimation, offline)
- OpenAI PRM800K (human labeled)
- PRIME (implicit from outcome)
- ReasonFlux-PRM (trajectory-aware, long-CoT)
- **OpenClaw-RL PRM** (live online from next-state, cross heterogeneous env)

Agent infra：
- OpenRLHF (engine decouple, batch data)
- AReal (asynchronous RL system)
- veRL/HybridFlow (decouple engines)
- slime (OpenClaw-RL 基础, 加上 PRM + serving loop)
- **OpenClaw-RL** (four-loop decouple, live online)

Hindsight / Distillation：
- HER (goal relabeling, classical RL)
- STaR (rationalize failure with answer hint)
- HIR (feedback → relabeled instruction)
- Self-Rewarding (LLM as own judge)
- Agarwal 2024 (on-policy distill from mistakes)
- Hübotter 2026 (RL via self-distillation)
- Shenfeld 2026 (self-distill enables continual learning)
- Buening 2026 (aligning from user interactions)
- **OpenClaw-RL OPD** (live hint extraction → forced input → token-level directional advantage)

---

## 15. 最终 Takeaway

OpenClaw-RL 的核心贡献可以浓缩为三句话：

1. **Reframe**：next-state signal $s_{t+1}$ 是 universal、free、live 的训练信号，跨所有 agent interaction type 通用。从对话到 terminal 到 GUI 到 SWE 到 tool-call，所有 next-state 都是同一个抽象——既是 evaluative 也是 directive。

2. **Recover**：两种方法分别恢复两类 wasted signal——Binary RL 通过 PRM 把 evaluative 信号变 scalar reward；OPD 通过 judge 提取 textual hint + enhanced context + forced input 把 directive 信号变 per-token directional advantage。两者 weighted combine 取得最佳。

3. **Reuse infrastructure**：slime 的异步 engine 解耦扩展到四 loop 全异步（serving/env/judging/training），让 live deployment 直接驱动 RL training，无 batch、无 offline、无 coordination overhead。

这是把 RL 从 "训练时收集数据→训练" 的 batch pipeline 转回 "部署即训练" 的 online learning paradigm 的关键一步。下一个十年的 LLM alignment 我猜测会沿这条路走——不是更大的 preference dataset，而是把 every deployed token 当 training data，让 model 在生产环境中持续自我改进。

参考链接汇总：
- OpenClaw-RL repo: https://github.com/Gen-Verse/OpenClaw-RL
- slime: https://github.com/THUDM/slime
- PPO: https://arxiv.org/abs/1707.06347
- GRPO / DeepSeekMath: https://arxiv.org/abs/2402.03300
- DAPO: https://arxiv.org/abs/2503.14476
- DPO: https://arxiv.org/abs/2305.18290
- Math-Shepherd: https://arxiv.org/abs/2310.08668 (link in paper)
- On-policy distillation (Agarwal): https://arxiv.org/abs/2310.12948
- RL via self-distillation (Hübotter): https://arxiv.org/abs/2601.20802
- Aligning from user interactions (Buening): https://thomasklbg.github.io/files/Aligning_Language_Models_from_User_Interactions.pdf
- Buffer of Thoughts: https://arxiv.org/abs/2406.04309
- SuperCorrect: https://arxiv.org/abs/2411.19805
- STaR: https://arxiv.org/abs/2203.14465
- HIR (Zhang 2023): https://arxiv.org/abs/2302.02998
- Self-Rewarding: https://arxiv.org/abs/2401.10020
- ArCHer: https://arxiv.org/abs/2402.19444
- LOOP: https://arxiv.org/abs/2502.01600
- RLAnything: https://arxiv.org/abs/2602.02488
- Group-in-group policy opt (Feng): https://arxiv.org/abs/2505.10978
- ReasonFlux-PRM: https://arxiv.org/abs/2506.18896
- OpenRLHF: https://arxiv.org/abs/2405.11143
- AReal: https://arxiv.org/abs/2505.24298
- ReAct: https://openreview.net/forum?id=WE_vluYUL-X
- Toolformer: https://arxiv.org/abs/2302.04761
- SWE-Bench: https://arxiv.org/abs/2310.06770
- OSWorld: https://arxiv.org/abs/2404.07972
- OpenClaw: https://github.com/openclaw/openclaw
