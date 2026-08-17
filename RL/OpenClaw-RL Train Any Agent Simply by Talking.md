---
source_pdf: OpenClaw-RL Train Any Agent Simply by Talking.pdf
paper_sha256: aea0b6c731880bdc4d497d392a0e298d9f74516d35a2baa76b714f5daabb0d70
processed_at: '2026-08-06T00:14:15-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 OpenClaw-RL

## 一句话版本

每个 agent 每次做完一件事，都会收到一个"回应" — 用户回了一句话、terminal 打印了一段输出、GUI 界面变了、test 跑出结果。这些回应里藏着一堆学习信号，但现有系统全都扔掉了。这篇 paper 说：把这些信号捡回来，就能让 agent 在正常使用过程中自动变强。

---

## 1. 核心观察：免费的信号被浪费

你跟 ChatGPT 聊天，它回答完，你说了句 "这不对，你应该先检查文件再编辑"。这句话发生了什么？

在现有系统眼里：这只是下一轮对话的 context，喂给模型生成下一轮回复就完事了。

但你仔细想 — 这句话其实对**上一轮回复**做了一个隐式评估 + 明确指导：
- 评估部分：你表达了不满（"这不对"）
- 指导部分：你告诉它应该怎么做（"先检查文件"）

这两类信息白白流走了。paper 把它们叫 **evaluative signal** 和 **directive signal**。

关键 insight 在于：这种 $(a_t, s_{t+1})$ 配对结构是 universal 的。不管你跑的是 chatbot、terminal agent、GUI agent、SWE agent 还是 tool-call agent，每次 agent 做 action $a_t$ 后都会收到 next-state $s_{t+1}$。表面上这五个场景风马牛不相及，底层结构完全一致。所以一个 policy 可以同时从所有这些 stream 里学，在同一个 training loop 里。

这就好比：不管你是在跟朋友聊天、在 shell 里敲命令、在 GUI 上点按钮、在写代码、在调 API，只要"你做了个动作 → 世界给了个反馈"这个结构在，就能学。

GitHub: https://github.com/Gen-Verse/OpenClaw-RL

---

## 2. 现有系统的两个浪费

### Waste 1：评估信号丢了

用户重新提问 = 不满意。Test 通过 = 成功。Error trace = 失败。这些都是天然的 process reward，不需要额外标注。

但现有 PRM（Process Reward Model）研究几乎全在 math reasoning 上做，依赖可验证的 ground truth。Math-Shepherd 用 Monte Carlo 估计 step-level reward；PRIME 学 implicit process reward。https://aclanthology.org/2024.acl-long.510/ https://arxiv.org/abs/2502.01456

到了 agentic 场景，大家要么忽略这个信号，要么只在 offline、pre-collected 的数据上用，要么只用 terminal outcome reward。长程任务里 outcome-only reward 只在最后一步有梯度，中间几十步全 unsupervised。

### Waste 2：方向信号丢了

用户说 "you should have checked the file first" — 这句话告诉你**哪些 token 应该不一样**，应该怎么改。一个 scalar reward $r = -1$ 根本承载不了这种 directional 信息。

现有方法怎么处理这个问题？
- RLVR 类（DeepSeek-R1、DAPO）用 scalar reward，directional info 全丢。https://arxiv.org/abs/2501.12948 https://arxiv.org/abs/2503.14476
- Distillation 类（on-policy distillation、Hübotter et al.）需要 pre-curated feedback-response pair，live 场景下没有。https://arxiv.org/abs/2601.20802
- Hindsight relabeling（HIR、STaR）在 fixed dataset 上做。https://arxiv.org/abs/2302.05206 https://openreview.net/forum?id=_3ELRdg2sgI

没人把这些 directional 信号从 live interaction 里实时抽出来用。

---

## 3. Binary RL：最朴素的回收

第一招很直接。既然 next-state 隐式评分了上一个 action，那就让 PRM judge 读 $(a_t, s_{t+1})$ 给个分：

$$\text{PRM}(a_t, s_{t+1}) \to r \in \{+1, -1, 0\}$$

跑 $m$ 次独立 vote，majority vote 出最终 $r_{\text{final}}$。然后直接拿 $r_{\text{final}}$ 当 advantage 做 PPO：

$$\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$$

$$\mathcal{L}_{\text{pg}} = -\mathbb{E}_t\left[\min\left(\rho_t A_t, \ \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon_{\text{high}}) \cdot A_t\right)\right]$$

变量解释：
- $\rho_t$：importance sampling ratio，新策略 vs 旧策略对 action $a_t$ 的概率比
- $\varepsilon = 0.2$：lower clip（保守压制 bad action）
- $\varepsilon_{\text{high}} = 0.28$：upper clip（更激进放大 good action，DAPO 的 asymmetric clip 思想）
- $A_t = r_{\text{final}}$：直接用 scalar reward 当 advantage

注意这里**没法用 GRPO 的 group-relative standardization**，因为 personal agent 场景下每次对话就一个样本，没有 group 结构。这是 personal agent 跟传统 batch RL 的一个本质区别。GRPO: https://arxiv.org/abs/2402.03300

Binary RL 简单粗暴，覆盖所有 scored turn，但信号粗糙 — 整个 response 一个 scalar，所有 token 同方向推。

---

## 4. OPD：精妙的方向信号回收

第二招是这篇 paper 最 elegant 的部分。

### 核心直觉

用户说 "you should have checked the file first"。这句话的信息量远超 $r=-1$。它告诉你 response 里哪个部分错了、应该怎么改。但 scalar reward 装不下。

OPD 的 trick：**让同一个 model 在"如果提前知道 hint"的 context 下，重新评估自己原来的 response**。teacher 和 student 是同一个 $\pi_\theta$，差异只在于 context 里有没有 hindsight hint。

### 四步流程

**Step 1 — 抽 hint**：

$$\text{Judge}(a_t, s_{t+1}) \to \{\text{score} \in \{+1, -1\}, \ \text{hint} \in \mathcal{T}^*\}$$

Judge 读 $a_t$ 和 $s_{t+1}$，如果发现 next-state 里有可抽取的 directive 信息，就生成 concise hint（1-3 句话）。**关键设计**：不直接用 $s_{t+1}$ 当 hint。原始 next-state 经常 noisy、verbose，用户回复里可能同时有 correction 和无关的新问题。Judge 把它蒸馏成 actionable instruction。

**Step 2 — 严格过滤**：在所有 score=+1 且 hint >10 字符的候选里选最长的。没有 valid hint 就直接丢弃样本。OPD 故意用 sample quantity 换 signal quality。

**Step 3 — 构造 enhanced context**：

$$s_{\text{enhanced}} = s_t \oplus \text{"[user's hint]\n{hint}"}$$

把 hint 追加到 last user message 后面。这就是"如果用户一开始就告诉你应该怎么做"的 context。

**Step 4 — Token-level advantage**：

policy model 在 $s_{\text{enhanced}}$ 下被查询，原 response $a_t$ 作为 forced input，算每个 token 的 log-prob。然后：

$$A_t[k] = \log \pi_{\text{teacher}}(a_t[k] \mid s_{\text{enhanced}}) - \log \pi_\theta(a_t[k] \mid s_t)$$

- $A_t[k]$：第 $k$ 个 token 的 directional advantage
- $\log \pi_{\text{teacher}}(a_t[k] \mid s_{\text{enhanced}})$：teacher（同 model，hint context 下）给 token $k$ 的 log-prob
- $\log \pi_\theta(a_t[k] \mid s_t)$：student（同 model，原始 context 下）给 token $k$ 的 log-prob

语义：
- $A_t[k] > 0$：teacher 在知道 hint 的情况下给该 token 更高概率 → student 应该 upweight
- $A_t[k] < 0$：teacher 觉得该 token 在 hint context 下不合适 → student 应该 downweight

这就做到了**同一 response 内，某些 token reinforce，某些 suppress**。scalar reward 永远做不到这种 per-token 分化。

### 为什么这个设计聪明

对比一下现有方法：
- RLHF 用 scalar preference，directional info 全丢。https://arxiv.org/abs/1706.03741
- DPO 需要 paired preference，live 场景难构造。https://arxiv.org/abs/2305.18290
- Standard distillation 需要外部 stronger teacher。OPD 用 self-as-teacher via context enhancement
- On-policy distillation（Agarwal）需要 pre-collected feedback-response pair。OPD 从 live next-state 实时抽。https://arxiv.org/abs/2310.12948

OPD 本质是一个 **context-conditioned self-distillation**：teacher 和 student 是同一个 model，差异只在 context 里有或没有 hindsight hint。这绕开了传统 distillation 的所有痛点。

---

## 5. 为什么要两者结合

Binary RL 和 OPD 互补：

| | Binary RL | OPD | Combined |
|---|---|---|---|
| 信号类型 | 评估（good/bad） | 方向 | 评估 + 方向 |
| Advantage 粒度 | sequence-level scalar | token-level directional | 混合 |
| 样本覆盖 | 所有 scored turn | 只有 valid hint 的 turn | 所有 scored turn |
| 信号丰富度 | 1 scalar/sample | 1 value/token | 1 value/token |

Binary RL 覆盖广但粗糙，OPD 高分辨率但样本稀疏。两者共享同一个 PPO loss，只有 advantage 计算不同，可以直接加权加：

$$A_t = w_{\text{binary}} \cdot r_{\text{final}} + w_{\text{opd}} \cdot \left(\log \pi_{\text{teacher}}(a_t \mid s_{\text{enhanced}}) - \log \pi_\theta(a_t \mid s_t)\right)$$

默认 $w_{\text{binary}} = w_{\text{opd}} = 1$。

直觉：Binary RL 给所有 turn 一个 baseline 梯度，OPD 在那些有明确 correction 的 turn 上叠加 fine-grained directional 修正。broad coverage + high resolution。

---

## 6. 架构：异步解耦才撑得住 live training

这是工程上最关键的设计。四个组件完全 decouple 成独立 async loop：

```
Policy Serving (SGLang) ⇄ Environment (HTTP/API) ⇄ Reward Judging (SGLang/API) ⇄ Policy Training (Megatron)
```

**为什么必须异步？**

想象 personal agent 场景：你在用 OpenClaw 写作业，每问一个问题，模型要 serve 你的请求，PRM 要 judge 上一个 response，trainer 要 apply gradient。如果同步，任何一个慢都会 block 其他。用户体验直接崩。

General agent 场景：长程 rollout 有 long-tail 问题，某些 task 跑半小时，快的 task 5 分钟。同步架构下快 task 必须等慢 task。异步让各组件各跑各的 loop，零协调开销。实验里他们用 128 个 parallel env for terminal、64 for GUI/SWE、32 for tool-call，这个规模只有异步架构下才可能。

这个设计建立在 slime 框架上。https://github.com/THUDM/slime

对比现有 RL infrastructure — OpenRLHF、AReal、veRL 都 decouple rollout 和 training，但都假设 batch data collection。OpenClaw-RL 是第一个真正做到 continuous training from live deployment 的。https://arxiv.org/abs/2405.11143 https://arxiv.org/abs/2505.24298 https://arxiv.org/abs/2409.19256

### Session-Aware Environment Server

Personal agent 的 environment 就是用户设备，通过 confidential API 连 RL server。每个 API request 分两类：
- **Main-line turn**：agent 的 primary response + tool execution results → 形成训练样本
- **Side turn**：auxiliary queries、memory organization → forward 但不产生 training data

每条新 main-line request 包含对前一轮的反应（user reply 或 env execution result），这自然成为前一轮的 $s_{t+1}$，用于 reward 计算。

---

## 7. General Agent：Step-wise Process Reward

对 general agent，长程任务 outcome-only reward 中间几十步全 unsupervised。按 RLAnything (Wang et al., 2026) 的做法，把 outcome 和 process reward 简单相加：https://arxiv.org/abs/2602.02488

$$R_t = o + \sum_{i=1}^{m} r_i / m$$

$r_i$ 由 $\text{PRM}(a_t, s_{t+1})$ 独立分配，$o$ 是 terminal outcome。

这里有个 subtle 问题：有了 step-wise reward 后，advantage standardization 不像 GRPO 那么直接。Feng et al. (2025b) 按相似 state 分组 standardize。但真实 terminal agent 场景下 state 很难 cluster，OpenClaw-RL 直接按相同 step index 分组，empirical 上有效。https://arxiv.org/abs/2505.10978

---

## 8. 实验告诉我们什么

### Personal Agent Track

两个 simulation：学生用 OpenClaw 做 GSM8K 作业（不想被发现用 AI）、老师用 OpenClaw 批改作业（希望 comment 具体友好）。policy 是 Qwen3-4B。https://arxiv.org/abs/2505.09388

Table 3 核心结果（base score 0.17）：

| Method | 8 steps | 16 steps |
|--------|---------|----------|
| Binary RL | 0.25 | 0.23 |
| OPD | 0.25 | 0.72 |
| Combined | 0.76 | 0.81 |

几个观察：
- Combined 效果最强，0.17 → 0.76 只需 8 步训练、36 次交互
- OPD 单独比 Binary RL 强（0.72 vs 0.23），但效果显现慢 — 8 步时只有 0.25，16 步才到 0.72。因为 hint 过滤严格，训练样本稀疏
- Binary RL 单独几乎没用（0.25 → 0.23），scalar reward 信号太粗糙

具体例子（Appendix B）：
- 学生 setting：优化前用 "**bold**"、step-by-step 结构化格式（明显 AI-like），优化后变成自然口语化直接给答案
- 老师 setting：优化前只说 "Correct. Well done!"，优化后给具体 step-by-step 表扬 + emoji

### General Agent Track

模型：Qwen3-8B (terminal)、Qwen3VL-8B-Thinking (GUI)、Qwen3-32B (SWE)、Qwen3-4B-SFT (tool-call)。

数据：SETA RL (terminal)、OSWorld-Verified (GUI)、SWE-Bench-Verified (SWE)、DAPO data (tool-call)。https://github.com/camel-ai/seta https://arxiv.org/abs/2404.07939 https://arxiv.org/abs/2310.06770

Scale：128 parallel env (terminal)、64 (GUI/SWE)、32 (tool-call)。

Table 4 — Process reward vs outcome-only：

| Setting | Integrated | Outcome only |
|---------|-----------|--------------|
| Tool-call | 0.30 | 0.17 |
| GUI | 0.33 | 0.31 |

Tool-call 上 process reward 提升巨大（+76%），GUI 上提升较小（可能因为 visual state diff 本身信息密度高，信号已经 dense）。Trade-off：hosting PRM 需要额外资源。

---

## 9. 我的直觉判断

### 统一视角的力量

这篇 paper 最有价值的贡献是把"训练 chatbot"、"训练 terminal agent"、"训练 GUI agent"、"训练 SWE agent"、"训练 tool-call agent"从五个独立工程问题压缩成一个统一问题。$(a_t, s_{t+1})$ 这个 pair 是所有 agentic RL 的最小公分母。从系统角度，一套 infrastructure 服务所有 deployment scenario；从学习理论角度，一个 policy 同时从所有 stream 学。

### OPD 的精妙

OPD 的核心 trick — **让 model 在 hint-enhanced context 下当自己的 teacher** — 在本质上是一个 context-conditioned self-distillation。teacher 和 student 是同一 model，差异只在 context。token-level log-prob gap $A_t[k] = \log\pi_{\text{teacher}}[k] - \log\pi_\theta[k]$ 提供的方向信息比 scalar reward 丰富一个数量级。

这个 trick 让我联想到：
- STaR 用 answer hints rationalize 失败的 reasoning。https://openreview.net/forum?id=_3ELRdg2sgI
- HIR 把 feedback 转成 relabeled instructions。https://arxiv.org/abs/2302.05206
- Buffer of Thoughts 检索 thought template 增强 prompt context。https://arxiv.org/abs/2406.04292
- SuperCorrect 从 teacher 抽 hierarchical template 做 cross-model DPO。https://arxiv.org/abs/2412.08893

OPD 把这些 thread 统一到 online setting：从 live next-state 抽 hint（hindsight relabeling）+ 同 model 当 teacher（self-distillation via context enrichment）+ token-level log-prob gap 当 directional advantage。没有 pre-collected data，没有 external teacher，没有 paired preference。

### 异步架构的工程价值

四组件完全 decouple 才能真正实现 continuous training from live deployment。这是把 RL 从 offline/batch 模式推向 online/continuous 模式的关键工程一步。传统 RL infrastructure 假设"收集一批数据 → 训练一批 → 再收集一批"，这种模式服务不了 live user，也撑不住 personal agent 的稀疏 stream。OpenClaw-RL 让 serving、PRM judging、training 各跑各的 loop，零协调开销。

### 几个开放问题

1. **PRM judge 的质量上限**：如果 judge 是同一个 $\pi_\theta$（self-rewarding），会不会 reward hacking？Self-Rewarding (Yuan et al., 2024) 暴露过类似问题。https://arxiv.org/abs/2401.10020

2. **OPD 在 implicit feedback 下的表现**：paper 的 student/teacher simulation 都比较 explicit（用户会明确说哪里不对）。真实用户经常只是 re-query 而不说哪里不对，这种 implicit 场景下 hint 抽取会失败，sample 被丢弃，OPD 训练数据极度稀疏。

3. **Token-level advantage 的 variance**：$\log\pi_{\text{teacher}}[k] - \log\pi_\theta[k]$ 在 token 级别可能很高方差，尤其 hint 只在 response 一小部分 token 上相关时。paper 没讨论这个 variance 问题。

4. **Asymmetric PPO clip 的贡献**：$\varepsilon_{\text{low}}=0.2$, $\varepsilon_{\text{high}}=0.28$ 是 DAPO 思想，但 paper 没给 ablation 显示这个 asymmetry 在 agentic setting 下的具体贡献。

---

## 10. 相关参考

- OpenClaw-RL: https://github.com/Gen-Verse/OpenClaw-RL
- slime: https://github.com/THUDM/slime
- OpenClaw: https://github.com/openclaw/openclaw
- PPO: https://arxiv.org/abs/1707.06347
- RLHF: https://arxiv.org/abs/1706.03741
- DPO: https://arxiv.org/abs/2305.18290
- GRPO: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DAPO: https://arxiv.org/abs/2503.14476
- RLAnything: https://arxiv.org/abs/2602.02488
- Math-Shepherd: https://aclanthology.org/2024.acl-long.510/
- PRIME: https://arxiv.org/abs/2502.01456
- ReasonFlux-PRM: https://arxiv.org/abs/2506.18896
- ReasonFlux: https://arxiv.org/abs/2502.06772
- On-policy distillation (Agarwal): https://arxiv.org/abs/2310.12948
- Self-distillation (Shenfeld): https://arxiv.org/abs/2601.19897
- STaR: https://openreview.net/forum?id=_3ELRdg2sgI
- HIR: https://arxiv.org/abs/2302.05206
- Self-Rewarding: https://arxiv.org/abs/2401.10020
- Buffer of Thoughts: https://arxiv.org/abs/2406.04292
- SuperCorrect: https://arxiv.org/abs/2412.08893
- SWE-Bench: https://arxiv.org/abs/2310.06770
- OSWorld: https://arxiv.org/abs/2404.07939
- ReAct: https://arxiv.org/abs/2210.03629
- Toolformer: https://arxiv.org/abs/2302.04761
- SWE-agent: https://arxiv.org/abs/2405.15793
- DigiRL: https://arxiv.org/abs/2406.11896
- ArCHer: https://arxiv.org/abs/2402.19309
- OpenRLHF: https://arxiv.org/abs/2405.11143
- AReal: https://arxiv.org/abs/2505.24298
- veRL (HybridFlow): https://arxiv.org/abs/2409.19256
- ReTool: https://arxiv.org/abs/2504.11536
- GSM8K: https://arxiv.org/abs/2110.14168
- Qwen3: https://arxiv.org/abs/2505.09388
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- SETA: https://github.com/camel-ai/seta
- Aligning LMs from User Interactions (Buening): https://thomasklbg.github.io/files/Aligning_Language_Models_from_User_Interactions.pdf
- Hindsight relabeling (Hübotter): https://arxiv.org/abs/2601.20802

---

# OpenClaw-RL 深度技术解析

## 1. 核心洞察：Next-State Signal 的双重信息

这篇 paper 建立在一个非常简单的观察之上：每次 agent 执行 action $a_t$ 之后都会收到一个 next-state signal $s_{t+1}$，这个信号在现有系统里只被当作下一步的 context，但实际上它编码了两类被浪费掉的信息：

**Waste 1 - Evaluative signals（评估性）**：next-state 隐式地给上一个 action 打分。用户重新提问 = 不满意；测试通过 = 成功；error trace = 失败。这是一个天然的 process reward，不需要额外的标注 pipeline。

**Waste 2 - Directive signals（指导性）**：next-state 经常携带"上一个 action 应该怎么做"的方向信息。用户说 "you should have checked the file first" 不仅告诉你 response 错了，还告诉你哪些 token 应该不一样。详细的 SWE error trace 也暗示了具体的修正方向。

这个观察的统一性在于：personal conversation、terminal execution、GUI interaction、SWE task、tool-call trace 在表层上完全不同，但是它们都共享一个相同的结构 — 都有 $(a_t, s_{t+1})$ pair。这意味着同一个 policy 可以同时从所有这些 stream 中学习，在同一个 training loop 里。

GitHub repo: https://github.com/Gen-Verse/OpenClaw-RL
相关背景 — slime 框架：https://github.com/THUDM/slime

---

## 2. Problem Setting：MDP 形式化

每个 interaction stream 都被形式化为一个 MDP $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R})$：

| 符号 | 含义 |
|------|------|
| $s_t \in \mathcal{S}$ | 完整 conversational 或 environmental context up to turn $t$ |
| $a_t \in \mathcal{A}$ | agent 的 response，由 $\pi_\theta$ 生成的 token sequence |
| $\mathcal{T}(s_{t+1} \| s_t, a_t)$ | transition，deterministic given environment |
| $r(a_t, s_{t+1})$ | reward，从 next-state signal 通过 PRM judge 推断 |

关键区别：在标准 RLVR 里，reward 只在 trajectory 终点用 outcome $o$；而 process reward $r(a_t, s_{t+1})$ 依赖 next state $s_{t+1}$，包含 dense 的 per-step 信号，并且 — 当 $s_{t+1}$ 携带 explicit directive 信息时 — 可以通过 on-policy distillation 转换成 token-level 的 directional gradient。

---

## 3. 架构：四组件异步解耦

OpenClaw-RL 构建在 slime 之上，四个组件完全 decouple 成独立的 async loop，没有任何 blocking dependency：

```
Policy Serving (SGLang) ⇄ Environment (HTTP/API) ⇄ Reward Judging (SGLang/API) ⇄ Policy Training (Megatron)
```

**为什么必须异步？**

对于 personal agent：environment 就是用户的个人设备，interaction stream 稀疏、session-based、个性化。模型要 serve live user request，同时 PRM 要 judge 上一个 response，trainer 要 apply gradient。如果同步，任何一个组件慢都会阻塞其他组件。

对于 general agent：长程 rollout 有 long-tail 问题（某些 task 跑很久）。异步让快 task 不必等慢 task，这是 large-scale environment parallelization 的前提。

实验里他们用 **128 个 parallel environments for terminal, 64 for GUI/SWE, 32 for tool-call**，这个规模只有在异步架构下才可行。

### Session-Aware Environment Server

每个 API request 被分成两类：
- **Main-line turn**：agent 的 primary response 和 tool execution results → 形成可训练样本
- **Side turn**：auxiliary queries、memory organization、environment transitions → forward 但不产生 training data

当前只训练 main-line turns。每条新 main-line request 包含对前一轮的反应（user reply 或 env execution result），这自然成为前一轮的 $s_{t+1}$，用于 reward 计算。

---

## 4. Binary RL：把 Evaluative Signal 转成 Scalar Process Reward

### 4.1 PRM Judge 构造

给定 response $a_t$ 和 next state $s_{t+1}$，judge model 评估 $a_t$ 的质量：

$$\text{PRM}(a_t, s_{t+1}) \to r \in \{+1, -1, 0\}$$

运行 $m$ 次独立查询取 majority vote：

$$r_{\text{final}} = \text{MajorityVote}(r_1, \ldots, r_m)$$

PRM 的 prompt（Appendix C.1）让 judge 先 step-by-step think，最后在 $\boxed{}$ 里给出分数。对 personal agent 的判定基于 user reply 中的满意/不满意信号；对 general agent 则判断 environment feedback 是否表明 task progress。GUI 用了 $m=3$，其他设置用 $m=1$。

### 4.2 PPO-style Training Objective

直接用 $A_t = r_{\text{final}}$ 作为 advantage，训练目标是 asymmetric clipped PPO surrogate：

$$\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$$

$$\mathcal{L}_{\text{pg}} = -\mathbb{E}_t\left[\min\left(\rho_t A_t, \ \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon_{\text{high}}) \cdot A_t\right)\right]$$

$$\mathcal{L} = \mathcal{L}_{\text{pg}} + \beta_{\text{KL}} \cdot \mathcal{L}_{\text{KL}} \tag{1}$$

变量含义：
- $\rho_t$：importance sampling ratio，新策略 vs 旧策略在 action $a_t$ 上的概率比
- $\varepsilon = 0.2$：lower clip bound（保守下降）
- $\varepsilon_{\text{high}} = 0.28$：upper clip bound（更激进上升，这是 DAPO 的 asymmetric clip 思想，鼓励 good action 更大幅度更新）
- $\beta_{\text{KL}} = 0.02$：KL penalty coefficient
- $A_t = r_{\text{final}}$：直接用 scalar reward 作为 advantage

注意这是 real-time conversational setting，**没有 group structure 可用于 GRPO 的 standardization**，所以只能用 plain PPO。这正是 personal agent 场景的特殊之处 — 每次对话只有一个样本，没法 group-relative。GRPO 论文：https://arxiv.org/abs/2402.03300

---

## 5. Hindsight-Guided On-Policy Distillation (OPD)：把 Directive Signal 转成 Token-Level 监督

这是 paper 里最 elegant 的部分。Binary RL 把整个 $s_{t+1}$ 压缩成一个 scalar $r \in \{+1, -1, 0\}$，但用户写 "you should have checked the file before editing it" 时，信息量远超一个标量 — 它告诉你哪些 token 应该不一样、应该怎么不一样。OPD 通过四步恢复这个信息：

### Step 1: Hindsight Hint Extraction

$$\text{Judge}(a_t, s_{t+1}) \to \{\text{score} \in \{+1, -1\}, \ \text{hint} \in \mathcal{T}^*\}$$

若 score = +1，judge 在 `[HINT_START]...[HINT_END]` 内生成 concise hint。运行 $m$ 次并行 judge calls。

**关键设计**：**不直接把 $s_{t+1}$ 当 hint**。原始 next-state 经常 noisy、verbose、含无关信息（用户回复里同时有 correction 和无关的新问题）。Judge 把 $s_{t+1}$ 蒸馏成一个 concise、actionable 的 instruction，通常 1–3 句话，聚焦"response 应该怎么做不一样"。

### Step 2: Hint Selection & Quality Filtering

在所有 score=+1 且 hint 长度 >10 字符的候选里，选最长的（最 informative 的）。如果没有 valid hint，**直接丢弃样本**。这是 OPD 的 deliberate design — 用 sample quantity 换 signal quality。只有那些 next-state 真正携带明确、可抽取 correction direction 的 turn 才进入训练。

这种严格过滤恰好与 Binary RL 互补：Binary RL 接受所有 scored turn（broad coverage，coarse signal），OPD 只接受高质量 hint turn（targeted，high-resolution）。Table 2 的对比很清楚：

| Dimension | Binary RL | OPD | Combined |
|-----------|-----------|-----|----------|
| Signal type | Evaluative (good/bad) | Directional | Evaluative + directional |
| Advantage | Sequence-level scalar | Token-level directional | Mixed sequence and token-level |
| Density | All scored turns | Hint-accepted turns only | All scored turns |
| Feedback type | User / environment | Explicit corrections | Both implicit and explicit |
| Signal richness | 1 scalar per sample | 1 value per token | 1 value per token |

### Step 3: Enhanced Teacher Context Construction

Hint 被追加到 last user message 后面，格式为 `[user's hint / instruction]\n{hint}`，构造 enhanced prompt：

$$s_{\text{enhanced}} = s_t \oplus \text{hint}$$

这就是"如果用户一开始就告诉 agent 应该怎么做"的 context。注意 teacher 和 student 是**同一个 model**，区别只在于 context 里有没有 hindsight hint。这是一个 self-distillation 变体，但 context-conditioned。

### Step 4: Token-Level Advantage

policy model 在 $s_{\text{enhanced}}$ 下被查询，原 response $a_t$ 作为 forced input，计算每个 response token 的 log-probability。然后：

$$A_t[k] = \log \pi_{\text{teacher}}(a_t[k] \mid s_{\text{enhanced}}) - \log \pi_\theta(a_t[k] \mid s_t)$$

变量解释：
- $A_t[k]$：第 $k$ 个 token 的 directional advantage
- $\log \pi_{\text{teacher}}(a_t[k] \mid s_{\text{enhanced}})$：teacher 在 hint-enhanced context 下给该 token 的 log-prob
- $\log \pi_\theta(a_t[k] \mid s_t)$：student（同一 model）在原始 context 下给该 token 的 log-prob

语义：
- $A_t[k] > 0$：teacher（知道 hint）给该 token 更高概率 → student 应该 upweight 这个 token
- $A_t[k] < 0$：teacher 认为该 token 在 hint context 下不合适 → student 应该 downweight

**关键 difference**：scalar advantage 把整个 response 的所有 token 推向同一方向；OPD 给出 per-token directional guidance，在同一 response 内，某些 token 被 reinforce，另一些被 suppress。

与现有方法的对比：
- **RLHF** (Christiano et al., 2017)：scalar preference signal → 失去 directional info。https://arxiv.org/abs/1706.03741
- **DPO** (Rafailov et al., 2023)：需要 paired preferences → 在线场景难构造。https://arxiv.org/abs/2305.18290
- **Standard distillation**：需要 separate、stronger teacher model → OPD 用 self-as-teacher via context enhancement
- **On-policy distillation** (Agarwal et al., 2024; Hübotter et al., 2026)：依赖 pre-collected feedback-response pairs → OPD 直接从 live next-state 抽取。https://arxiv.org/abs/2310.12948

训练依然用 Equation (1) 的 clipped surrogate，只是 advantage 现在携带 per-token rich info。

---

## 6. Combined Method

由于 Binary RL 和 OPD 共享同一个 PPO loss，只有 advantage computation 不同，可以直接相加 advantage：

$$A_t = w_{\text{binary}} \cdot r_{\text{final}} + w_{\text{opd}} \cdot \left(\log \pi_{\text{teacher}}(a_t \mid s_{\text{enhanced}}) - \log \pi_\theta(a_t \mid s_t)\right)$$

默认 $w_{\text{binary}} = w_{\text{opd}} = 1$。这个组合的设计动机：
- Binary RL 提供所有 turn 上的 broad gradient coverage
- OPD 在那些有 directive signal 的子集上提供 high-resolution、per-token correction
- 两者优势互补，缺点相互 offset

---

## 7. General Agent：Step-wise Process Reward

### 7.1 为什么 Process Reward 在长程任务里 vital

在长程 agentic 任务里，outcome-only reward 只在 terminal step 提供梯度信号，绝大多数 turn 处于 unsupervised 状态。PRM 给每个 turn 基于 next-state signal 分配 reward，提供 dense credit assignment。RLAnything (Wang et al., 2026) 提供了 strong empirical evidence — 在 GUI、text-game、coding 任务上，step-wise PRM + outcome consistently 胜过 outcome-only。https://arxiv.org/abs/2602.02488

### 7.2 Outcome + Process Reward 整合

按 RLAnything 做法，简单相加：

$$R_t = o + \sum_{i=1}^{m} r_i / m$$

其中 $r_i$ 由 $\text{PRM}(a_t, s_{t+1})$ 独立分配，$o$ 是 terminal outcome reward。

**这里有个 subtle 的技术问题**：有了 step-wise reward 后，standardize advantage 不像 GRPO 那么直接。Feng et al. (2025b) 在 group-in-group 里按相似 state 分组后 standardize。但 terminal agent 这种真实场景下 state 很难 cluster，所以 OpenClaw-RL 直接按相同 step index 分组 standardize，empirical 上有效。

---

## 8. 实验：两个互补 Track

### 8.1 Personal Agent Track

两个 simulation setting：

**Student setting**：LLM 模拟一个学生用 OpenClaw 在个人电脑上做 GSM8K 作业，但不想被发现用 AI。是否"AI-like"完全依赖学生的个人偏好和写作风格。policy model 是 Qwen3-4B，lr $1 \times 10^{-5}$，KL coef 0，每 16 个训练样本触发一次 training。

**Teacher setting**：teacher 用 OpenClaw 批改 AI 写的作业，希望 comment 具体、友好。

Table 3 的核心结果（base score 0.17）：

| Method | Updated 8 steps | Updated 16 steps |
|--------|-----------------|------------------|
| Binary RL | 0.25 | 0.23 |
| OPD | 0.25 | 0.72 |
| Combined | 0.76 | 0.81 |

**Takeaway [Q1]**：Combined method 效果最强。OPD 单独比 Binary RL 强，但因为训练样本稀疏，效果显现较慢（8 步时只有 0.25，16 步才到 0.72）。Binary RL 单独只提供 marginal 改善。

**Takeaway [Q2]**：Combined 方法下，只需 36 个 problem-solving interactions（student setting）或 24 个 grading interactions（teacher setting）就能看到显著提升。Figure 2 显示 base 0.17 → updated 0.76（student）/ 0.90（teacher），仅 8 步训练。

具体例子（Appendix B）展示优化效果：
- Student setting：优化前用 "**bold**"、step-by-step 结构化格式（明显 AI-like），优化后变成自然口语化的解释，直接给出答案
- Teacher setting：优化前只说 "Correct. Well done!"，优化后给出具体的 step-by-step 表扬 + emoji

### 8.2 General Agent Track

模型设置：
- Terminal: Qwen3-8B + SETA RL data (Shen et al., 2026), 128 envs
- GUI: Qwen3VL-8B-Thinking + OSWorld-Verified (Xie et al., 2024), 64 envs。https://arxiv.org/abs/2404.07939
- SWE: Qwen3-32B + SWE-Bench-Verified (Jimenez et al., 2023), 64 envs。https://arxiv.org/abs/2310.06770
- Tool-call: Qwen3-4B-SFT + DAPO data (Yu et al., 2025a), 32 envs，evaluated on AIME 2024

Hyperparameters（Table 5）：
- lr $10^{-6}$, KL coef 0.01, clip ε/ε_high = 0.2/0.28
- Max response 8192 tokens, max context 16384 tokens
- Max interactive steps: 30 (GUI), 20 (SWE), 10 (terminal)
- Rollout batch: 8 (GUI/SWE), 16 (terminal), 32 (tool-call)
- 8 samples per task

**Takeaway [Q3]**：Framework 在 diverse 真实世界场景下 work（Figure 4），large-scale env parallelization 提升 scalability。

**Takeaway [Q4]**：Integrating outcome + process reward 胜过 outcome-only（Table 4）：

| Setting | Integrated | Outcome only |
|---------|-----------|--------------|
| Tool-call | 0.30 | 0.17 |
| GUI | 0.33 | 0.31 |

Tool-call 上 process reward 的提升尤其显著（0.17 → 0.30，+76%）。GUI 上也有提升但相对小（因为 GUI 的 next-state signal — visual state diff — 信息密度本就高）。Trade-off：hosting PRM 需要额外资源。

---

## 9. 与相关工作对比的直觉理解

### 9.1 vs. RLHF / DPO / GRPO 家族

RLHF 用 scalar preference 训 reward model 然后 PPO；DPO 绕过 reward model 用 closed-form preference optimization；GRPO 用 group-relative advantage 去掉 critic。这些都在 **batch-offline mode** 下工作，数据收集和训练分离。OpenClaw-RL 是 **continuous online**，从 live interaction 训练，没有 pre-collection phase。DeepSeek-R1: https://arxiv.org/abs/2501.12948；DAPO: https://arxiv.org/abs/2503.14476

### 9.2 vs. Agentic RL（ReAct、Toolformer、FireAct、SWE-agent、DigiRL、ArCHer）

每个现有 agentic RL 工作都 target 单一环境用 dedicated pipeline。OpenClaw-RL 统一了 personal conversation、terminal、GUI、SWE、tool-call 五种 setting，在同一个 training loop 里。ReAct: https://arxiv.org/abs/2210.03629；Toolformer: https://arxiv.org/abs/2302.04761；SWE-agent: https://arxiv.org/abs/2405.15793

### 9.3 vs. Process Reward Models

Math-Shepherd (Wang et al., 2024) 用 Monte Carlo 估计自动 step-wise supervision；GenPRM 用 generative CoT verification；ReasonFlux-PRM 扩展到 trajectory-aware 长链评估；PRIME 学 implicit process reward。这些都在 math reasoning 上做，依赖 verifiable ground truth。OpenClaw-RL 把 PRM judging 扩展到 online、heterogeneous long-horizon agentic setting，process reward 从 live next-state 推断，不依赖 pre-collected ground truth。Math-Shepherd: https://aclanthology.org/2024.acl-long.510/；ReasonFlux-PRM: https://arxiv.org/abs/2506.18896

### 9.4 vs. On-policy Distillation & Hindsight Methods

Buffer of Thoughts (Yang et al., 2024b) 检索 high-level thought templates 增强 prompt；SuperCorrect (Yang et al., 2025c) 从 teacher 提取 hierarchical templates 做 cross-model DPO；STaR (Zelikman et al., 2022) 用 answer hints rationalize 失败；HIR (Zhang et al., 2023) 把 feedback 转成 relabeled instructions；Self-Rewarding (Yuan et al., 2024) 用 LLM 当自己的 judge。这些都在 fixed dataset 上工作。OPD 把这些 thread 统一到 online setting：
- 从 live next-state 抽 textual hint（hindsight relabeling）
- 同一 model 在 hint-enhanced context 下当自己的 teacher（self-distillation via context enrichment）
- token-level log-prob gap 提供 directional advantage（无需外部 teacher、无需 paired preference、无需 pre-collected data）

On-policy distillation (Agarwal): https://arxiv.org/abs/2310.12948；STaR: https://openreview.net/forum?id=_3ELRdg2sgI；Buffer of Thoughts: https://arxiv.org/abs/2406.04292；SuperCorrect: https://arxiv.org/abs/2412.08893；Self-Rewarding: https://arxiv.org/abs/2401.10020

### 9.5 vs. RL Infrastructure

OpenRLHF、AReal、veRL、slime 都 decouple rollout 和 training engine。OpenClaw-RL 在 slime 基础上做到 **四个完全独立的 async loop**（serving、rollout、PRM judging、training），允许从 live multi-stream interaction 连续训练且 zero interruption to serving。这个 capability 是 prior RL infrastructure 缺失的，它们都假设 batch data collection 而非 live deployment。OpenRLHF: https://arxiv.org/abs/2405.11143；AReal: https://arxiv.org/abs/2505.24298；veRL (HybridFlow): https://arxiv.org/abs/2409.19256

---

## 10. 算法伪代码

### Algorithm 1: Binary RL Pipeline

```
Require: Session/trajectory T, turn t, messages M_t
1: a_t, logp_old ← SGLang(M_t)           // serve and collect log-probs
2: Buffer {prompt_ids, response_ids, logp_old} for (T, t)
3: // On next turn: extract s_{t+1} and fire PRM
4: s_{t+1} ← first message of M_{t+1}    // user reply or env feedback
5: {r_i}_{i=1}^m ← PRM(a_t, s_{t+1})     // m parallel votes, async
6: r ← MajorityVote({r_i})
7: A_t ← r broadcast
8: Apply at-least-one guarantee if zero effective samples in T
9: Submit Sample to trainer queue
```

注意 Step 8 的 "at-least-one guarantee"：如果 trajectory 里 zero effective samples（所有 vote 都是 0 或被过滤），仍要至少保留一个样本避免 trajectory 浪费。

### Algorithm 2: OPD Pipeline

```
Require: Turn data (a_t, M_t, logp_old), next state s_{t+1}
1: {(score_i, hint_i)}_{i=1}^m ← Judge(a_t, s_{t+1})
2: valid ← {h_i : score=+1 ∧ |h_i| > 10}
3: if valid = ∅ then
4:    Drop sample; return
5: end if
6: hint ← arg max_{h ∈ valid} |h|
7: s_enhanced ← M_t ⊕ "[user's hint]\n{hint}"
8: logp_teacher ← Teacher(a_t | s_enhanced)
9: A_t[k] ← logp_teacher[k] - logp_old[k]
10: Submit Sample(teacher_log_probs=A_t) to trainer queue
```

关键步骤 6 选最长 hint，步骤 9 是 token-level advantage 的逐 token 计算。

---

## 11. 个人直觉总结：这篇 paper 的真正价值

### 11.1 统一视角的力量

OpenClaw-RL 最有价值的地方在于识别出 next-state signal 是一个 universal、stream-agnostic 的学习源。这把"训练对话 agent"、"训练 terminal agent"、"训练 GUI agent"、"训练 SWE agent"、"训练 tool-call agent"从五个独立问题压缩成一个统一问题。从系统设计角度，这意味着同一套 infrastructure 服务所有 deployment scenario；从学习理论角度，这意味着 $(a_t, s_{t+1})$ 这个 pair 是所有 agentic RL 的最小公分母。

### 11.2 OPD 的精妙之处

OPD 的核心 trick 在于**让 model 在 hint-enhanced context 下当自己的 teacher**。这避开了几个传统 distillation 的痛点：

1. 不需要 external stronger teacher model（self-as-teacher via context）
2. 不需要 pre-collected feedback-response pair（从 live next-state 抽 hint）
3. 不需要 paired preference（token-level log-prob gap 直接给 directional gradient）
4. 不需要 verifiable ground truth（hint 来自 textual correction）

这个设计在本质上是一个 **context-conditioned self-distillation**：teacher 和 student 是同一 model，差异只在 context 里有或没有 hindsight hint。token-level log-prob gap $A_t[k] = \log\pi_{\text{teacher}}[k] - \log\pi_\theta[k]$ 提供的方向信息比 scalar reward 丰富一个数量级 — 同一 response 内某些 token 应增、某些应减，scalar reward 永远做不到这种 per-token 分化。

### 11.3 异步架构的工程价值

四组件完全 decouple 才能真正实现 continuous training from live deployment。现有 RL infrastructure 假设 batch data collection — 收集一批数据、训练一批、再收集一批。这种模式不能服务 live user，也不能在 personal agent 的稀疏 stream 下 work。OpenClaw-RL 让 serving、PRM judging、training 各跑各的 loop，零协调开销，这是把 RL 从 offline/batch 模式推向 online/continuous 模式的关键工程一步。

### 11.4 实验的两个层次发现

- **Personal agent**：Combined (Binary + OPD) 显著优于单独任何一个。OPD 因为 hint 过滤严格、样本稀疏，效果显现慢但最终上限高；Binary RL 覆盖广但信号粗糙。两者叠加 = broad coverage + high resolution。0.17 → 0.76 仅需 8 步训练、36 次交互。
- **General agent**：Process reward 在长程任务（特别是 tool-call）上提升巨大（0.17 → 0.30），印证 RLAnything 的结论。GUI 上提升较小可能因为 visual state diff 本身信息密度高、信号已经 dense。

### 11.5 局限性与开放问题

paper 没深入讨论的几个点：
1. PRM judge 本身的质量上限 — 如果 judge model 是同一个 $\pi_\theta$（self-rewarding），可能存在 reward hacking 或 feedback loop 问题。Self-Rewarding (Yuan et al., 2024) 暴露过类似问题。
2. OPD 的 hint extraction 依赖 judge model 能从 noisy next-state 中稳定抽取 actionable instruction — 在用户回复高度 implicit 的情况下（比如只是 re-query 而不说哪里不对），hint 抽取会失败，sample 被丢弃，OPD 这部分训练数据会极度稀疏。paper 的 student/teacher simulation 都是比较 explicit 的 correction，真实用户可能更 implicit。
3. Token-level advantage 的 variance — $\log\pi_{\text{teacher}}[k] - \log\pi_\theta[k]$ 在 token 级别可能很高方差，尤其是 hint 只在 response 的一小部分 token 上相关时。paper 没讨论这个 variance 问题。
4. Asymmetric PPO clip（ε_low=0.2, ε_high=0.28）是 DAPO 思想，但 paper 没给 ablation 显示这个 asymmetry 在 agentic setting 下的具体贡献。

---

## 12. 相关参考链接

- OpenClaw-RL GitHub: https://github.com/Gen-Verse/OpenClaw-RL
- slime (OpenClaw-RL 的 base infrastructure): https://github.com/THUDM/slime
- OpenClaw (personal AI assistant): https://github.com/openclaw/openclaw
- PPO (Schulman et al., 2017): https://arxiv.org/abs/1707.06347
- RLHF (Christiano et al., 2017): https://arxiv.org/abs/1706.03741
- DPO (Rafailov et al., 2023): https://arxiv.org/abs/2305.18290
- GRPO / DeepSeekMath (Shao et al., 2024): https://arxiv.org/abs/2402.03300
- DeepSeek-R1 (Guo et al., 2025): https://arxiv.org/abs/2501.12948
- DAPO (Yu et al., 2025a): https://arxiv.org/abs/2503.14476
- RLAnything (Wang et al., 2026): https://arxiv.org/abs/2602.02488
- Math-Shepherd (Wang et al., 2024): https://aclanthology.org/2024.acl-long.510/
- ReasonFlux-PRM (Zou et al., 2025): https://arxiv.org/abs/2506.18896
- ReasonFlux (Yang et al., 2025b): https://arxiv.org/abs/2502.06772
- On-policy distillation (Agarwal et al., 2024): https://arxiv.org/abs/2310.12948
- Self-distillation enables continual learning (Shenfeld et al., 2026): https://arxiv.org/abs/2601.19897
- STaR (Zelikman et al., 2022): https://openreview.net/forum?id=_3ELRdg2sgI
- HIR (Zhang et al., 2023): https://arxiv.org/abs/2302.05206
- Self-Rewarding LMs (Yuan et al., 2024): https://arxiv.org/abs/2401.10020
- Buffer of Thoughts (Yang et al., 2024b): https://arxiv.org/abs/2406.04292
- SuperCorrect (Yang et al., 2025c): https://arxiv.org/abs/2412.08893
- SWE-Bench (Jimenez et al., 2023): https://arxiv.org/abs/2310.06770
- OSWorld (Xie et al., 2024): https://arxiv.org/abs/2404.07939
- ReAct (Yao et al., 2023): https://arxiv.org/abs/2210.03629
- Toolformer (Schick et al., 2023): https://arxiv.org/abs/2302.04761
- SWE-agent (Yang et al., 2024a): https://arxiv.org/abs/2405.15793
- DigiRL (Bai et al., 2024): https://arxiv.org/abs/2406.11896
- ArCHer (Zhou et al., 2024): https://arxiv.org/abs/2402.19309
- OpenRLHF (Hu et al., 2024): https://arxiv.org/abs/2405.11143
- AReal (Fu et al., 2025): https://arxiv.org/abs/2505.24298
- HybridFlow / veRL (Sheng et al., 2025): https://arxiv.org/abs/2409.19256
- ReTool (Feng et al., 2025a): https://arxiv.org/abs/2504.11536
- PRIME (Cui et al., 2025a): https://arxiv.org/abs/2502.01456
- GSM8K (Cobbe et al., 2021): https://arxiv.org/abs/2110.14168
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- Qwen3-VL (Bai et al., 2025): https://arxiv.org/abs/2511.21631
- SETA (Shen et al., 2026): https://github.com/camel-ai/seta
- Aligning LMs from User Interactions (Buening et al., 2026): https://thomasklbg.github.io/files/Aligning_Language_Models_from_User_Interactions.pdf
- Hindsight relabeling (Hübotter et al., 2026): https://arxiv.org/abs/2601.20802

---

这篇 paper 的核心 takeaway：在所有 agentic RL 场景里，next-state signal 都免费提供 evaluative + directive 两类信息，过去都被浪费。Binary RL 通过 PRM 把 evaluative 部分转成 dense scalar process reward，OPD 通过 hindsight hint extraction + context-conditioned self-distillation 把 directive 部分转成 token-level directional advantage。两者互补叠加 + 异步四组件架构 = 一个 system 同时个性化 personal agent 并提升 general agent 的长程任务能力，全部从 agent 已经在进行的 interaction 中学习。
