---
source_pdf: SkillOrchestra_ Learning to Route Agents via Skill Transfer.pdf
paper_sha256: 871529a6c139b07aec92119c01552ab0cee212e76a8431623cb7624f74a60079
processed_at: '2026-08-12T07:19:02-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 SkillOrchestra

Andrej，我用最朴素的话重新讲一遍。

## 一句话说清楚

你有一堆 LLM，有的贵有的便宜，有的擅长 math 有的擅长 coding。现在来了一个复杂任务，要多步解决。**谁来决定每一步用哪个 model？** 这个 paper 就是解决这个问题的。

## 之前的两种做法都不太好

**做法一：传统 router**。就像你去医院，前台护士看你一眼就决定你挂哪个科。问题是，复杂任务不是一步能搞定的——你可能先要 search 一下资料，再写段 code 算一算，最后综合一下给答案。每一步需要的"能力"不一样，一个 query 级别的 decision 不够用。

**做法二：RL 训练一个 router**。就是拿一个 LLM，用 reinforcement learning 训它，让它学会 multi-turn routing。听起来很美好，end-to-end 嘛。但实际训练出来会发现一个尴尬的现象——**router 学着学着就偷懒了**。它发现"反正最后用最强的那个 model 最容易答对"，于是 98% 的时候都 call LLaMA-70B，其他 model 全闲置。paper 里管这叫 **routing collapse**。

这就好比你雇了一个调度员，本意是让他根据不同情况派不同的人去干活，结果他发现"反正老张最靠谱，什么事都叫老张"，最后变成一个人干所有活，其他人全白养了。

## SkillOrchestra 的思路

核心 idea 特别朴素：**别让 router 用"直觉"做决策，给它一本"能力手册"让它查表**。

这本手册叫 **Skill Handbook**，里面记三样东西：

### 1. 有哪些"技能"（Skill）

技能是介于"做什么"和"谁来做"之间的中间层。比如在 code mode 下，技能可以是 "symbolic_logic"（符号逻辑推理）、"numerical_approximation"（数值近似）、"data_processing"（数据处理）等等。

为什么要搞这一层？因为如果你直接问"这道题该用 GPT-5 还是 Qwen"，你是在 model identity 空间里做决策，这个空间太小、太 discrete。但如果你先问"这道题需要什么能力"，再问"哪个 model 在这个能力上最强"，决策就变得 structured 了。

### 2. 每个 model 在每个技能上的成功率（Agent Profile）

用 Beta 分布来建模。简单说就是记录：Qwen2.5-Coder 在 "symbolic_logic" 上试了 20 次，成功 17 次，那成功率大概是 85%。这个数字会不断更新——每次执行完都 record 成功还是失败，Beta 参数 $\alpha$ 和 $\beta$ 就更新。

好处是还能告诉你 **不确定性**：如果只试了 2 次，成功 2 次，Beta posterior 很宽，router 知道"这个数字不太靠谱"，会谨慎一点。

### 3. 什么时候该用什么 mode 的经验

比如"search 完之后经常需要 code 来验证数值"这类 meta-level 的 routing 经验。

## 运行时怎么用这本手册

来了一个 query，router 做两步决策：

**第一步：现在该 search、code 还是直接 answer？** 查手册里的 mode-level 经验。

**第二步：在选定的 mode 下，用哪个 model？** Router 先分析当前需要哪些 skill 以及各占多少权重，然后对每个 candidate model 算一个分数：

$$\text{score} = \text{加权成功率} - \lambda \times \text{成本}$$

就是这个 model 在这些 skill 上的预期表现，减去它的 cost。选分数最高的。

就这么简单。没有 gradient descent，没有 policy network，就是一个 **weighted competence minus cost 的 linear utility model**。

## 手册怎么建出来

不用大量 data，流程是：

**Phase 1: 发现技能**。拿少量样本（每个 dataset 不到 50 个），跑不同的 trajectory。找到一个"成功"的 trajectory 和一个"失败"的 trajectory，对比它们差在哪。差的那部分就是一个 missing skill。用 LLM 把这个 gap 抽象成一句 skill description。

同时统计每个 model 在每个 skill 上的成功/失败次数，更新 Beta 参数。

**Phase 2: 精炼**。技能太多太碎不行，太少太粗也不行。定期检查：
- 某个 skill 上 model 表现 variance 很大 → 说明这个 skill 其实混了好几种能力，该拆开
- 两个 skill 上 model 表现几乎一样 → 说明这俩 skill 对 routing 来说没区别，该合并

用 LLM 来 review 这些 split/merge proposal，生成优化后的 skill definitions。

## 还有一个精妙的设计：手册要按 router 能力裁剪

这点很关键。细粒度的 skill（比如区分 "symbolic_logic" 和 "numerical_approximation"）对强 router 好用，因为它能准确判断当前需要哪个。但弱 router 可能判断错——把需要 symbolic_logic 的题误判成 numerical_approximation，route 错 model，反而更差。

所以手册要根据 router 的能力做裁剪。做法是：拿不同 granularity 的手册版本，在 validation set 上跑，看哪个版本在这个 router 上表现最好。本质是个 **Pareto-optimal selection**——找 accuracy 和 cost 的最佳平衡点。

## 结果怎么样

数字说话：

**Model routing**（7 个 QA benchmark + 2 个 math benchmark）：
- 比 RL 训练的 Router-R1 高 5.8 个点（47.4 vs 41.6 EM）
- 训练 data 用量少 700 倍
- 推理 cost 降低约 2 倍
- Router-R1 有 98% 的调用都给了 LLaMA-70B（routing collapse），SkillOrchestra 的调用分布是 Mixtral 44% / Qwen-7B 26% / LLaMA-70B 15% / Qwen-3B 11%，均衡得多

**Agent orchestration**（FRAMES benchmark，有 search/code/answer 三种 mode）：
- 比 ToolOrchestra（RL 训练）高 8 个点 accuracy，cost 降 21%
- 比 Claude Opus 4.5 当 orchestrator 高 6.4 个点，cost 降 90%（$758 → $73）

**Transferability**：手册在 Qwen2.5-3B 上学的，直接拿去给 Qwen-7B、Llama-8B、Mistral-7B、Mixtral-8x22B 用，全都大幅提升。而且 **越强的 model 收益越大**——Qwen-7B 提了 24 个点，Qwen-3B 只提 15 个点。这说明手册里的知识是 model-agnostic 的，强 model 能更好地利用它。

## 为什么这个方法 work

我觉得有三个根本原因：

**第一，显式建模避开了 reward sparsity**。RL 的 reward 是 trajectory-level 的——只有最后答对没有才有 signal。中间每一步选谁，reward 无法区分，所以 policy 容易塌缩到"用最强的就完事"。SkillOrchestra 把每一步的 decision 都 grounding 到 explicit 的 capability estimate 上，不存在 "credit assignment 不知道该 credit 谁" 的问题。

**第二，Bayesian profile 提供了 calibrated uncertainty**。当你知道"Qwen-Coder 在 symbolic_logic 上 85% 成功，在 numerical 上只有 30%"，你的 routing decision 是 informed 的。RL policy 是个黑箱，你不知道它为什么选这个 model，它可能只是因为 gradient 恰好加强了那条 path。

**第三，knowledge 和 reasoning 解耦**。手册是独立的知识库，router 只是查表推理。换 router 不用手冊重学，换 model pool 只需要更新 profile。这种 modularity 在实际部署中价值很大——model pool 经常变，RL 训的 router 每次都要重训。

## 最让我兴奋的点

这篇 paper 暗示了一个更大的趋势：**对于 LLM 时代的复杂决策系统，structured knowledge base + symbolic reasoning 可能比 end-to-end RL 更 scalable**。

这跟纯 deep learning 的哲学不太一样。更像是一种 "Software 1.5"——hybrid approach：
- Skill 是 auto-discovered 的（用 LLM 抽象 capability gap）
- Profile 是 auto-estimated 的（Bayesian update from execution data）
- 但 routing logic 是 symbolic 的（linear utility over explicit profile）

纯 RL（Software 2.0）在这种 multi-step、sparse reward、large action space 的场景下 sample inefficient 且不稳定。纯 hand-written rules（Software 1.0）不 adaptive。SkillOrchestra 找到了一个 sweet spot：**用 learned knowledge 替代 hand-crafted rules，用 symbolic reasoning 替代 end-to-end policy learning**。

如果这个方向对，未来的 agent system 架构可能更像：**LLM as reasoning engine + auto-constructed skill library + Bayesian capability database**，而不是 "one giant RL-trained policy"。

---

打个比方总结：RL 方法是培养一个"全能调度员"，从零开始靠经验学习怎么派活，学成之后可能偷懒只派一个人。SkillOrchestra 是给调度员一本详细的"员工能力档案"——张三擅长什么、李四擅长什么、每种活该派谁——让他照着手册做决策。手册是自动生成的，还能随着新员工入职自动更新。

---

# SkillOrchestra 深度解析

Andrej，这篇 paper 抓住了一个 RL-based orchestration 的本质痛点，然后从一个完全不同的角度切入——把 orchestration 从 **policy optimization** 重构成 **skill knowledge base construction**。让我一层一层剥开。

## 1. 核心问题的本质：为什么 RL-based orchestrator 会崩

先 build intuition。RL-based 方法（Router-R1、ToolOrchestra）训练一个 LLM 作为 router，用 GRPO 优化 trajectory-level reward。直觉上这很优雅——end-to-end、可微分、能学到复杂的 sequential decision。但实验里观察到 **routing collapse**：Router-R1 在 9 个 benchmark 上 98.02% 的调用都给了 LLaMA-3.1-70B，其他 5 个 model 几乎被遗忘（Qwen2.5-7B 0.35%, Mistral-7B 0.92%, Mixtral-8x22B 0.04%, Qwen2.5-3B 0.00%）。

这个现象的 root cause 在于 RL 的 reward landscape。目标函数：

$$J(\tau) = \mathbb{E}_{\tau \sim \pi}\left[R(\tau) - \lambda \sum_{t=0}^{T} C(A_t, z_t)\right]$$

- $R(\tau) \in [0,1]$: trajectory-level task success reward（最终答对没有）
- $C(A_t, z_t)$: step $t$ 上 agent $A_t$ 产生 trace $z_t$ 的 cost（token usage 或 latency）
- $\lambda$: tradeoff 系数，控制你多在乎 cost
- $T$: trajectory 长度（multi-turn 的总步数）

问题是 $R(\tau)$ 是 sparse 的——只有 trajectory 结束才有 signal。中间每一步选哪个 model，reward 无法区分。GRPO 这种 trajectory-level credit assignment 在长 horizon 下会退化：policy 发现"反正最后用最强的 LLaMA-70B 最容易答对"，于是梯度反复加强这条 path，其他 path 的 gradient signal 被淹没。这就是 routing collapse 的数学本质——**reward sparsity + capability-agnostic exploration → policy 收敛到 trivial solution**。

Reference: [GRPO paper (DeepSeekMath)](https://arxiv.org/abs/2402.03300), [Router-R1](https://arxiv.org/abs/2502.12025), [ToolOrchestra](https://arxiv.org/abs/2511.21689)

## 2. SkillOrchestra 的核心 insight：解耦 capability 和 identity

SkillOrchestra 不学 policy 参数 $\theta$，而是学一个 **Skill Handbook** $\mathcal{H}$。优化目标从：

$$\theta^* = \arg\max_\theta \mathbb{E}_{\tau \sim \pi_\theta}[J(\tau)]$$

变成：

$$\mathcal{H}^* = \arg\max_\mathcal{H} \mathbb{E}_{\tau \sim \pi(\cdot|\mathcal{H})}[J(\tau)]$$

这里 $\pi(\cdot|\mathcal{H})$ 表示 policy 是被 handbook 引导的，而非由可训练参数定义的。直觉上：**不训练 router 的"直觉"，而是给它一本"参考手册"让它查表推理**。

这种重构带来三个结构性优势：

1. **State-conditioned granularity**: 不同 state 需要不同 capability，handbook 让 router 在 capability space 推理，而不是在 model identity space 推理。
2. **Routing stability**: 因为决策基于 explicit competence estimate（Beta posterior），不会因为 reward sparsity 而 collapse。
3. **Transferability**: handbook 是 model-agnostic 的知识，可以跨 orchestrator backbone 复用。

## 3. 形式化：Agent, Mode, Skill 三层抽象

### 3.1 基础设定

系统组件：
- **Orchestrator**: 中央 controller
- **Model Pool** $\mathcal{M} = \{m_1, \ldots, m_{K_M}\}$: candidate LLMs
- **Tool Pool** $\mathcal{T} = \{t_1, \ldots, t_{K_T}\}$: web search, code exec, retrieval 等
- **Agent**: $A = (m, T_A)$, 其中 $m \in \mathcal{M}$, $T_A \subseteq \mathcal{T}$
- **Operational Mode** $\psi \in \Psi = \{\text{search}, \text{code}, \ldots\}$: 决定"做什么"

每个 mode $\psi$ 限制可用 tool set $T_\psi \subseteq \mathcal{T}$，诱导出 valid agent set：

$$\mathcal{A}_\psi = \{(m, T_A) \mid m \in \mathcal{M}, T_A \subseteq T_\psi\}$$

### 3.2 Policy factorization

关键设计——把 routing 拆成两层决策：

$$\pi(a_t | s_t) = \pi_{\text{mode}}(\psi_t | s_t) \cdot \pi_{\text{route}}(A_t | s_t, \psi_t)$$

- $\pi_{\text{mode}}$: 先决定"这一步该 search 还是 code 还是 answer"
- $\pi_{\text{route}}$: 在选定 mode 下，决定用哪个 agent

这种 factorization 的 intuition：**"做什么"和"用谁做"是两个正交的决策维度**。传统 model routing 把它们混在一起（单步、单 mode、选 model），RL-based 方法虽然 multi-turn 但也把它们耦合在 end-to-end policy 里。SkillOrchestra 显式分开，让每层决策有独立的 knowledge base 支撑。

传统 model routing 是这个 formulation 的 special case：$T=0$, $\Psi = \{\text{answer}\}$, no tools。退化成 $\max_{\pi_{\text{route}}} \mathbb{E}_{A \sim \pi_{\text{route}}(\cdot|q)}[R(A,q) - \lambda C(A,z)]$。

## 4. Skill 和 Agent Profile 的形式定义

### 4.1 Skill

$$\sigma \triangleq \langle \mathcal{D}, \mathcal{Z} \rangle$$

- $\mathcal{D}$: natural-language description，描述这个 capability 是什么（e.g., "symbolic logic coding for constraint verification"）
- $\mathcal{Z}$: contextual indicators——keywords, structural patterns, exemplar queries——用来判断当前 state 是否需要这个 skill

Skill 是 **mode 和 agent 之间的中间层**。直觉：你不说"用 GPT-5 解这道题"，而说"这道题需要 symbolic_logic skill，而 Qwen2.5-Coder-32B 在 symbolic_logic 上的 success rate 是 85%"。这把 capability requirement 从 agent identity 上解耦。

### 4.2 Agent Profile

$$\mathcal{P}_{A,\psi} = \left(\{\phi_{A,\sigma}\}_{\sigma \in \Sigma_\psi}, \hat{C}_A(\psi), \mathcal{R}_{A,\psi}, \Gamma_A\right)$$

- $\phi_{A,\sigma}$: agent $A$ 在 skill $\sigma$ 上的 success probability 估计
- $\hat{C}_A(\psi)$: mode-conditioned cost estimate（latency / token usage）
- $\mathcal{R}_{A,\psi}$: routing signals——usage constraints, systematic failures（e.g., "这个 model 在 code mode 下经常生成 syntactically invalid Python"）
- $\Gamma_A$: high-level strengths/weaknesses summary

$\phi_{A,\sigma}$ 用 Beta distribution建模：

$$\phi_{A,\sigma} \sim \text{Beta}(\alpha_{A,\sigma}, \beta_{A,\sigma})$$

- $\alpha_{A,\sigma}$: agent $A$ 在 skill $\sigma$ 上的成功 count
- $\beta_{A,\sigma}$: 失败 count
- Posterior mean: $\alpha/(\alpha+\beta)$，作为 point estimate

用 Beta 而不是 point estimate 的好处：**uncertainty quantification**。新 agent 或 rare skill 上 $\alpha+\beta$ 小，posterior 宽，router 会更谨慎（可能 fallback 到更可靠的选择）。这是个 Bayesian bandit 的思路。

## 5. Skill Handbook 的图结构

Handbook 是个 graph $\mathcal{G_H} = (\mathcal{V}, \mathcal{E})$:

$$\mathcal{V} = \mathcal{V}_\Psi \cup \mathcal{V}_\Sigma \cup \mathcal{V}_\mathcal{P}$$

- $\mathcal{V}_\Psi$: mode-level routing metadata $\mathcal{R}_\psi$（什么时候该 search，什么时候该 code）
- $\mathcal{V}_\Sigma$: skill registry
- $\mathcal{V}_\mathcal{P}$: agent profiles
- $\mathcal{E}$: mode-skill 关联，induced mapping $M: \Psi \to 2^\Sigma$, $\Sigma_\psi := M(\psi)$

这个 graph structure 的作用：**runtime 不需要搜整个 skill space，只在 mode-consistent 的子集 $\Sigma_\psi$ 里搜**。这是计算效率的关键。

## 6. Runtime: Skill-Grounded Routing

### 6.1 Handbook Selection（per-orchestrator）

这是 paper 里一个很精妙的设计。先 intuition：fine-grained skills（e.g., 把 "data_processing" 拆成 "symbolic_logic" + "numerical_approximation"）对 strong orchestrator 好，因为它能准确判断当前需要哪个 subskill。但 weak orchestrator 可能 misidentify——把需要 symbolic_logic 的 query 误判成 numerical_approximation，route 到一个 numeric computation specialist 上，反而更差。

所以 handbook 需要根据 orchestrator capacity 做 granularity 调整。形式化：

$$\mathcal{H}_{\text{base}}^{(\mathcal{O})} = \arg\max_{\mathcal{H} \subseteq \mathcal{H}^*} \mathbb{E}_{q \sim \mathcal{D}_{\text{val}}}\left[R(\tau_\mathcal{H}(q)) - \lambda \sum_{t=0}^{|\tau_\mathcal{H}(q)|} C(\psi_t, A_t)\right]$$

- $\mathcal{H}^*$: 全量 learned handbook
- $\mathcal{H}_{\text{base}}^{(\mathcal{O})}$: 给 orchestrator $\mathcal{O}$ 的 induced subgraph
- 在 validation set 上直接评估 trajectory-level objective

这是个 **Pareto-optimal selection**——遍历 handbook 的不同 granularity 配置，选在 performance-cost Pareto frontier 上的那个。

可选的 query-time augmentation：

$$\mathcal{H}_q = \mathcal{H}_{\text{base}}^{(\mathcal{O})} \cup \bigcup_{\sigma \in \mathcal{N}_k(q)} \left(\{\sigma\} \cup \{\mathcal{P}_{A,\psi} \mid A \in \mathcal{A}_\psi\}\right)$$

- $\mathcal{N}_k(q)$: query $q$ 在 embedding space 里的 $k$ nearest skills
- 用 semantic similarity 补充 base handbook 可能遗漏的 skills

### 6.2 Competence-Aware Agent Routing

核心 routing 公式：

$$A_t^* = \arg\max_{A \in \mathcal{A}_{\psi_t}} \left[\sum_{\sigma \in \Sigma_t} w_{t,\sigma} \frac{\alpha_{A,\sigma}}{\alpha_{A,\sigma} + \beta_{A,\sigma}} - \lambda_c \cdot \hat{C}_A(\psi_t)\right]$$

逐项解析：
- $\Sigma_t \subseteq \Sigma_{\psi_t}$: 当前 state $s_t$ 下 active 的 skill set
- $w_{t,\sigma}$: skill $\sigma$ 在 state $s_t$ 的 weight（orchestrator 判断这个 skill 有多重要，sum 约等于 1）
- $\frac{\alpha_{A,\sigma}}{\alpha_{A,\sigma} + \beta_{A,\sigma}}$: Beta posterior mean，agent $A$ 在 skill $\sigma$ 上的 expected success probability
- $\sum_{\sigma} w_{t,\sigma} \cdot \text{posterior mean}$: 加权 competence estimate，即 agent $A$ 在当前 skill mix 下的预期成功率
- $\lambda_c \cdot \hat{C}_A(\psi_t)$: cost penalty，$\lambda_c$ 控制 cost 权重
- 整个 expression: **competence - cost utility**，选 utility 最大的 agent

Intuition：这本质上是个 **linear utility model over capability profile**。每个 agent 有一个 capability vector（在各个 skill 上的 success rate），当前 state 有一个 skill demand vector（weights），点积就是 expected competence，再减去 cost。比 RL policy 的黑箱决策 interpretable 得多。

Algorithm 1 的完整流程：
1. Mode selection: $\psi_t \sim \pi_{\text{mode}}(\cdot | s_t; \mathcal{R}_\psi)$
2. Retrieve active skills $\Sigma_t$ from $\mathcal{H}_q$
3. 对每个 candidate agent $A$，compute $\hat{P}(A) = \sum_\sigma w_{t,\sigma} \cdot \alpha/(\alpha+\beta)$
4. Compute utility $U(A) = \hat{P}(A) - \lambda_c \cdot \hat{C}_A(\psi_t)$
5. $A_t = \arg\max_A U(A)$
6. Execute, get $(z_t, o_t)$, update state $s_{t+1}$

## 7. Skill Handbook Learning

### 7.1 Phase 1: Skill Discovery + Profile Construction

训练数据 $\mathcal{D}_{\text{train}} = \{(q_i, \mathcal{B}_i)\}_{i=1}^N$，其中 $\mathcal{B}_i = \{\tau_i^{(1)}, \tau_i^{(2)}, \ldots\}$ 是在固定 query 下 vary agent choice 得到的不同 trajectory。

**Contrastive skill discovery**: 对每个 query 和 mode $\psi$，找一个成功 trajectory $\tau_+^\psi$ 和失败 trajectory $\tau_-^\psi$，compute：

$$\mathcal{D}_{\text{diff}}(\tau_+^\psi \| \tau_-^\psi)$$

这个 difference isolate 了"missing capability"——成功 trajectory 做对了但失败 trajectory 没做对的那部分。然后 LLM-based discoverer 把这个 capability gap 抽象成 skill definition $\sigma_{\text{new}}$，加入 registry。

**Beta profile update**:

$$\alpha_{A,\sigma}^{(t+1)} \leftarrow \alpha_{A,\sigma}^{(t)} + \sum_{\tau \in \mathcal{B}_i} \mathbb{I}[A \text{ succeeds on } \sigma \text{ in } \tau]$$

$$\beta_{A,\sigma}^{(t+1)} \leftarrow \beta_{A,\sigma}^{(t)} + \sum_{\tau \in \mathcal{B}_i} \mathbb{I}[A \text{ fails on } \sigma \text{ in } \tau]$$

- $\mathbb{I}[\cdot]$: indicator function
- 每次 agent $A$ 在 skill $\sigma$ 上成功就 $\alpha$ 加 1，失败就 $\beta$ 加 1
- 这是 Beta-Bernoulli conjugate update，standard Bayesian inference

**Mode-level routing signal distillation**: 从 trajectory 里提取 recurring patterns（e.g., "search 之后经常需要 code 来做数值验证"），存成 $\mathcal{R}_\psi$。

### 7.2 Phase 2: Handbook Refinement

防止 skill registry 过度碎片化或冗余。

**Splitting criterion**: skill $\sigma$ 的 agent performance variance 高 → 说明 $\sigma$ 其实混了多个不同 capability，该拆。Intuition：如果 agent A 在 $\sigma$ 上时好时坏，且不同 agent 的成功 pattern 不一致，那 $\sigma$ 的 definition 太粗了。

**Merging criterion**: 两个 skill $(\sigma_i, \sigma_j)$ 的 agent performance profile 统计上 indistinguishable → 说明它们对 routing 来说是 redundant，该合并。Intuition：如果所有 agent 在 $\sigma_i$ 和 $\sigma_j$ 上的 success rate 几乎一样，那区分它们对 routing 决策没价值。

LLM-based reflector（e.g., GPT-5）review 这些 split/merge proposal，生成 revised skill definitions。Approved 的 refinement 同步更新 $(\alpha_{A,\sigma}, \beta_{A,\sigma})$。

## 8. 实验数据深度分析

### 8.1 Model Routing Results

Table 1 的核心数字（EM accuracy）：

| Method | NQ | TriviaQA | PopQA | HotpotQA | 2wiki | Musique | Bamboogle | Avg |
|--------|-----|----------|-------|----------|-------|---------|-----------|-----|
| Router-R1 (RL) | 38.8 | 70.6 | 38.4 | 35.2 | 43.4 | 13.8 | 51.2 | 41.6 |
| SkillOrchestra | 54.2 | 71.6 | 42.6 | 39.0 | 48.0 | 18.2 | 58.4 | 47.4 |
| SkillOrchestra+ | 54.8 | 80.2 | 48.8 | 44.2 | 49.6 | 20.6 | 63.2 | 51.6 |

- SkillOrchestra 用 Qwen2.5-3B 作为 orchestrator（和 Router-R1 一样）
- SkillOrchestra+ 允许在同一 agent pool 内切换 orchestrator model，但用同一个 learned handbook
- Multi-hop QA（Musique, Bamboogle）增益最大，因为这些 task 最需要 state-conditioned routing
- Training cost: SkillOrchestra 只用 $k<50$ samples/dataset，Router-R1 用 14k samples，cost reduction 700×

### 8.2 Routing Collapse 对比

Figure 6 (left) 的数据：

| Model | Router-R1 调用占比 | SkillOrchestra 调用占比 |
|-------|-------------------|------------------------|
| LLaMA-3.1-70B | 98.02% | 15.38% |
| Mixtral-8x22B | 0.04% | 44.53% |
| Qwen2.5-7B | 0.35% | 25.99% |
| Qwen2.5-3B | 0.00% | 11.50% |
| Mistral-7B | 0.92% | ~3% |

SkillOrchestra 的 distribution 反映 **capability-aware specialization**：Mixtral-8x22B（strong + 相对 cheap）承担最多，Qwen2.5-7B 处理中等难度，LLaMA-70B 只在真正需要时用，Qwen2.5-3B（orchestrator 自身）也能直接回答简单 query。

Cost 对比：Router-R1 51.8¢ → SkillOrchestra 38.4¢ → SkillOrchestra+ 41.6¢。注意 **per-token price ≠ total cost**：便宜的 model 如果生成很长的 reasoning chain，total cost 反而更高。SkillOrchestra 显式优化这个 tradeoff。

### 8.3 Transferability

Figure 6 (right) 的关键数据——handbook 在 Qwen2.5-3B 上学，直接 transfer：

| Orchestrator | Before | After | Gain |
|--------------|--------|-------|------|
| Qwen2.5-3B | 40.7% | 56.1% | +15.4 |
| Qwen2.5-7B | 35.7% | 60.0% | +24.3 |
| Llama3.1-8B | 35.5% | 58.0% | +22.5 |
| Mistral-7B | 36.5% | 59.8% | +23.3 |
| Mixtral-8x22B | 46.5% | 61.3% | +14.8 |

注意一个反直觉的点：**stronger backbone 收益更大**（绝对值上 Qwen-7B 增 24.3 vs Qwen-3B 增 15.4）。这说明 handbook 提供的 structured skill guidance 和 backbone 的 reasoning capability 是 complementary 的——stronger model 能更好地利用 handbook 里的 fine-grained skill info 做 routing 决策。

这验证了 Section 4.3 的 handbook selection 设计：handbook 本身是 model-agnostic 的，但 **optimal granularity depends on orchestrator capacity**。

### 8.4 Agent Orchestration (FRAMES)

更复杂的 setting，3 个 mode（search/code/answer），50 turn horizon：

| Method | Accuracy | Cost |
|--------|----------|------|
| ToolOrchestra (RL) | 76.3% | $92.7 |
| GPT-5 orchestrator | 74.6% | $120.4 |
| Claude Opus 4.5 | 77.9% | $758.1 |
| Gemini 3 Pro | 78.9% | $1729.3 |
| SkillOrchestra | 84.3% | $72.7 |

SkillOrchestra 比 ToolOrchestra：accuracy +8.0, cost -21.6%。比 GPT-5 orchestrator：accuracy +9.7, cost -39.6%。比 Claude Opus 4.5：accuracy +6.4, cost **-90.4%**。

这里的 insight：用更强的 model 当 orchestrator 能提升 raw performance，但 cost 爆炸（Claude Opus 4.5 要 $758）。SkillOrchestra 通过 explicit skill modeling 让 cheap orchestrator（Qwen3-8B）+ specialized agent pool 达到更好的 performance-cost tradeoff。

### 8.5 Ablation Study

Table 2 的 component contribution：

| Setting | Acc | Cost |
|---------|-----|------|
| No HB | 71.0% | $122.9 |
| No Ref + Sel | 79.0% | $5.5 |
| No Selection | 79.3% | $3.4 |
| No FG Skills | 80.4% | $15.1 |
| Full System | 85.0% | $9.3 |

关键 insight：
- **No HB → Full**: accuracy +14, cost -92.5%。Handbook 是核心。
- **No Ref + Sel → No Selection**: accuracy 基本不变（79.0 → 79.3），但 cost 从 $5.5 到 $3.4。Refinement 主要帮 cost efficiency。
- **No FG Skills → Full**: accuracy +4.6, cost -38.4%。Fine-grained skills 帮 accuracy（更精准的 capability matching）也帮 cost（避免 over-competent model 的浪费）。
- **No Selection → Full**: accuracy +5.7, cost +173%。Selection 让 accuracy 大涨但 cost 上升——因为选了更 fine-grained 的 handbook，router 能更精准地调用 strong model，但调用次数/质量提升带来 cost。

注意 **No Selection 的 cost 最低（$3.4）但 accuracy 也低**。这说明 coarse handbook 会 conservative routing，少用贵 model，但错过很多需要 strong model 的 case。Full system 找到了 optimal tradeoff。

### 8.6 Tool-Level Allocation Analysis (Appendix C)

对比 SkillOrchestra vs ToolOrchestra 在各 mode 的 model 选择：

**Search mode**:
- ToolOrchestra: 99.7% GPT-5-mini（routing collapse）
- SkillOrchestra: 100% Qwen3-32B（cheapest，但 capability 足够）

**Answer mode**:
- ToolOrchestra: 97.9% GPT-5（routing collapse）
- SkillOrchestra: 58.4% GPT-5 + 10.0% GPT-5-mini + 31.6% Qwen3-32B / math-expert models

SkillOrchestra 的 cost reduction 来自 **smarter cross-mode allocation**，不是单纯少调用。Search mode 用最 cheap 的就够，answer mode 才需要 strong model，且 answer mode 内部也按 difficulty 分流。

## 9. Demo Trace 分析

### 9.1 AMC Example (Figure 7): Multi-turn correction

Query: "How many complex numbers satisfy $z^5 = \bar{z}$?"

Turn 1: Skill analysis → `algebra.logs_and_exponents` 70%, `algebra.rational_linear_manipulation` 30%。Route to Mixtral-8x22B（在 logs_and_exponents 上 50% success，且是 expensive tier 里最强的）。Mixtral 给出正确 form 但 miscount（说"infinitely many"）。

Turn 2: Router 检测到 count 不对，re-route to LLaMA-3.1-70B（在 `rational_linear_manipulation` 上 0% 但 overall 30%，且是 Pell-type expert）。LLaMA 正确 restrict $\theta \in [0, 2\pi)$，count 出 7。

Turn 3: Final answer `7`。

**Insight**: Router 能基于 intermediate result 判断"这个 model 答错了"，然后 route 到另一个 model 做 correction。这种 **multi-turn verification + re-routing** 是 single-shot router 做不到的。

### 9.2 AMC Example (Figure 8): Autonomous answer

Router 分析后认为可以自己解（Qwen2.5-3B orchestrator），不发任何 `<search>` call，直接给 answer。

**Insight**: Skill-based routing 的 external call 是 optional 的。如果 orchestrator 判断自己的 capability 足够，可以省掉 model call，进一步降 cost。RL-based router 通常没有这种"自我评估"能力。

### 9.3 PopQA Example (Figure 9): Cross-model verification

Query: "Who was the composer of It's Going to Take Some Time?"

Turn 1: Route to Qwen2.5-7B（Attribute/Location Retrieval 56% success）。返回 noisy + incorrect content（"Frank Wildhorn"）。
Turn 2: Route to LLaMA-3.1-70B（Multi-hop Bridge 67%）。返回 "Toni Stern and Carole King" 但 express uncertainty。
Turn 3: Route to Mixtral-8x22B 做 cross-verification。确认 "Carole King"。
Turn 4: Final answer `Carole King`。

**Insight**: Router 对 model response 的 confidence 做 assessment，不确定时 cross-verify。这比 single-model answer 可靠得多。

## 10. 更深层的技术联想

### 10.1 和 Bayesian Bandit 的关系

Competence-aware routing 公式本质上是 **contextual bandit with Beta priors**：
- 每个 (agent, skill) pair 是一个 arm
- Beta posterior 是 arm 的 reward distribution
- Context 是当前 state 的 skill demand $\{w_{t,\sigma}\}$
- Policy 是 posterior mean - cost 的 greedy selection

区别于标准 bandit：这里不是纯 exploration-exploitation tradeoff，而是 **capability-conditioned selection**。Exploration 已经在 handbook learning phase 做完了（contrastive trajectory collection），runtime 是 pure exploitation。

Reference: [Thompson Sampling](https://en.wikipedia.org/wiki/Thompson_sampling), [Contextual Bandits survey](https://arxiv.org/abs/1803.09821)

### 10.2 和 Mixture of Experts 的对比

MoE 在 neural network 层面做 routing（gate network 选 expert），SkillOrchestra 在 system 层面做 routing（orchestrator 选 agent）。相似点：
- 都有 explicit specialization
- 都有 routing decision per input

不同点：
- MoE 的 expert 是 trained from scratch 的 sub-network，capability 是 emergent 的
- SkillOrchestra 的 agent 是 fixed pre-trained model，capability 是 discovered + profiled 的
- MoE routing 是 differentiable 的，SkillOrchestra routing 是 symbolic reasoning over handbook

### 10.3 和 Retrieval-Augmented Generation 的类比

可以把 Skill Handbook 看成一种 **structured knowledge base for routing decisions**：
- RAG: query → retrieve relevant docs → LLM generates answer
- SkillOrchestra: state → retrieve relevant skills + agent profiles → orchestrator routes

这种 framing解释了为什么 SkillOrchestra 数据效率高：它把 orchestration knowledge 显式存成可检索的结构，而不是压缩进 model 参数。

### 10.4 和 Constitutional AI / Self-Correction 的联系

Demo trace 里的 multi-turn correction（Figure 7）本质上是 **self-correction via re-routing**。如果 router 自己答错，它不知道；但如果它 route 的 model 答错，router 可以基于 skill analysis + response quality 判断"可能错了"，然后 re-route。这比 self-correction（同一个 model 自己纠错）更 powerful，因为不同 model 的 error pattern 不相关。

Reference: [Constitutional AI](https://arxiv.org/abs/2212.08073), [Self-Refine](https://arxiv.org/abs/2303.17651)

### 10.5 Potential limitation 我觉得值得思考

1. **Skill discovery 的 coverage**: contrastive trajectory 只能发现"被 test 过的 capability gap"。如果某个 skill 从没在 $\mathcal{D}_{\text{train}}$ 里被触发，handbook 里就没有它。这限制了 generalization 到 OOD query。
2. **Beta prior 的 cold start**: 新 agent 加入 pool 时 $\alpha = \beta = 0$，posterior undefined。需要 exploration phase 收集 data。Paper 没详谈 online adaptation。
3. **Skill hierarchy 的 optimality**: split/merge 是 LLM-based reflector 决定的，没有 formal optimality guarantee。可能 reflect 出 suboptimal structure。
4. **Cost model 的 accuracy**: $\hat{C}_A(\psi)$ 是 static estimate。实际 cost 随 query length, tool response size 变化。Dynamic cost estimation 可能更准但更复杂。
5. **Multi-agent coordination**: 当前 framework 是 sequential routing，没有 parallel agent execution 或 agent-to-agent communication。Complex task 可能需要 parallel exploration。

### 10.6 和 Meta's ToolFormer / Gorilla 的对比

ToolFormer 学什么时候 call tool，Gorilla 学 call 哪个 API。SkillOrchestra 在更高 abstraction level 做 decision：在 skill space 里 reasoning，skill 再 map 到 (model, tool) pair。这种 abstraction layering 更 scalable——新 model 加入时只需要 update profile，不需要 retrain router。

Reference: [ToolFormer](https://arxiv.org/abs/2302.04761), [Gorilla](https://arxiv.org/abs/2305.15334)

### 10.7 和 Andrej 你自己提的 "Software 2.0" 的关系

你之前提过 Software 2.0 是用 learned model 替代 hand-written rules。SkillOrchestra 有点像 **Software 1.5**——hybrid approach：
- Skill discovery 是 learned（LLM-based discoverer）
- Skill profile 是 learned（Beta update from data）
- 但 routing logic 是 symbolic（linear utility over explicit profile）

这可能是 practical 的 sweet spot：纯 RL（Software 2.0）sample inefficient 且 collapse-prone，纯 hand-written rules（Software 1.0）不 adaptive。SkillOrchestra 用 learned knowledge base + symbolic reasoning engine 兼顾两者。

Reference: [Software 2.0 by Karpathy](https://karpathy.medium.com/software-2-0-a64152b37c35)

## 11. 总结：这篇 paper 的真正贡献

表面上是又一个 routing framework，但深层贡献是 **提供了一个 non-RL path to complex orchestration**。它 show 了：

1. Explicit skill abstraction 能避免 RL 的 routing collapse
2. Bayesian profile + symbolic routing 能比 end-to-end RL 更 sample efficient（700× reduction）
3. Knowledge base 和 reasoning engine 解耦后，knowledge 可以跨 orchestrator transfer

这暗示了一个更广的 thesis：**对于 multi-step decision making in LLM era，structured knowledge base + symbolic reasoning 可能比 end-to-end RL 更 scalable**。这和 AlphaGo 时代 "RL 解决一切" 的 narrative 不太一样，更像 expert system 的 renaissance——但这次 knowledge base 是 auto-constructed 的，knowledge 是 learned 的。

如果这个方向对，未来的 agent system 可能会更像 "LLM as reasoning engine + auto-constructed skill libraries + Bayesian profile database"，而不是 "one giant RL-trained policy network"。

Code: [github.com/jiayuww/SkillOrchestra](https://github.com/jiayuww/SkillOrchestra)

---

如果你想 dive deeper，几个方向值得探索：
- Skill discovery 的 active learning（怎么选 contrastive pair 最 informative）
- Online profile adaptation（runtime 拿到新 success/failure signal 怎么 update）
- Skill hierarchy 的 automatic ontology construction
- Multi-agent parallel execution within skill framework
- 和 Tree-of-Thought / Graph-of-Thought 的结合（在 skill space 做 search 而不只是 select）
