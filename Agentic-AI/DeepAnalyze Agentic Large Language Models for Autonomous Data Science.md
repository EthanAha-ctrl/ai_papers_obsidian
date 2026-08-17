---
source_pdf: DeepAnalyze Agentic Large Language Models for Autonomous Data Science.pdf
paper_sha256: 282a12564d395e1f827697737be5a1483381f5f992b94879e42a985340a1b434
processed_at: '2026-08-03T18:36:06-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好嘞 Karpathy, 换个频道, 我用大白话重新讲一遍, 就当我们俩在咖啡馆聊这个工作。

---

## 这帮人到底想干嘛

你看啊, data scientist 日常工作其实很套路: 拿到一坨 raw data (可能 CSV、JSON、SQL 数据库), 先清洗, 再分析, 画图, 建个模型, 最后写个报告给老板看。这整个过程, 行话叫 "data science pipeline"。

那问题来了: 能不能让 AI 把这整套活儿自己干完? 从你扔给它一堆数据文件开始, 到它吐出一份像样的分析师级别报告结束, 中间不需要你介入?

这就是这篇 paper 想解决的事儿, 而且他们用了一个 **8B 参数的小模型** 干成了, 还干得比 GPT-4o 套各种 agent 框架还要好。

---

## 之前别人怎么做的, 为啥不行

之前有两条路:

**路线 A**: 训个专门的小模型, 比如就专门做 TableQA (给张表格回答问题), 或者专门生成 Pandas 代码。问题是你训出来的模型只会干一件事, 你让它把整套 pipeline 串起来跑, 它就傻了, 因为它没有 "我先做 A 再做 B 再做 C" 这种编排能力。

**路线 B**: 用很强的 LLM (GPT-4o 啥的) + 一个手工设计的 workflow 框架 (比如 AutoGen、ReAct), 让 LLM 按框架一步步走。这种确实能跑通不少任务, 但有两个毛病: 一是 workflow 是人手写的, 一旦遇到框架设计时没预料到的情况就崩; 二是你每跑一次都要调 GPT-4o API, 又贵又慢。

打个比方: 路线 A 像训了个只会炒一道菜的厨师, 路线 B 像给一个聪明厨师一本详细菜谱让他照着做, 但菜谱没写的菜他就不会做了。

这帮人想说: 能不能训一个厨师, 不需要菜谱, 自己看食材决定怎么处理?

---

## 他们怎么做的: 把 "agent 行为" 直接塞进 model weights 里

核心 idea 其实挺优雅的: 之前路线 B 的 workflow 是外挂在 LLM 外面的, 这帮人把它 **内化到模型权重里**。

具体来说, 他们给 LLM 的词表加了 5 个特殊 token:

- `<Analyze>`: 模型自言自语, 思考下一步该干嘛
- `<Understand>`: 模型表示 "我要去理解下这个数据文件"
- `<Code>`: 模型要写代码了
- `<Execute>`: 执行刚才的代码, 接收执行结果
- `<Answer>`: 出最终答案, 整个过程结束

模型自己决定什么时候用哪个 token, 没有外部框架指挥它。就像一个 data scientist 脑子里自己会想 "嗯我先看看这表长啥样" → "那我写段代码统计一下" → "嗯报错了, 我看看哪错了" → "好的我再改改" → "出报告"。

这个设计为啥比 ReAct 那种 thought-action 套路好? 因为它把 **"理解结构化数据"** 单独拎出来了。你想啊, data scientist 跟数据打交道, "理解数据本身" 跟 "推理下一步" 是两种不同的心智活动, 揉在一起模型容易糊, 拆开反而清晰。Ablation 实验也验证了: 把 `<Understand>` 去掉, DABStep 那个 benchmark 直接掉 7 个点。

---

## 训练是这个 paper 的精华

光设计 architecture 没用, 关键是怎么训。这里有个大坑: data science 任务太复杂, 你让 base model (R1-8B) 直接做完整的 data science pipeline, 它十有八九做不出来, 你给它 reward 全是 0 或负的, RL 训练根本启动不了。这叫 **reward sparsity** 问题。

他们的解法特别像教小孩: **先学单技能, 再学组合**。

### Stage 1: 单技能 SFT

先让模型练基本功, 类似 data scientist 入门先学 SQL、Pandas、统计这些单项。

训练数据 470K 条, 序列长 8K, 涵盖:
- 推理 (用 R1 蒸馏出来的 long CoT 数据)
- 结构化数据理解 (TableQA 类数据)
- 代码生成 (data science 相关代码)

这一步之后, 模型在 DS-1000 (代码生成 benchmark) 上从 30.4 跳到 54.8, 单技能算打牢了。

### Stage 2: 组合技能 agentic RL

基本功扎实后, 才开始训它怎么把这些技能串起来用。

分两小步:
1. **Cold start**: 先用 20K 条合成的多轮交互轨迹做 SFT, 让模型学会 "在推理之间插入 `<Code>` 和 `<Execute>`" 的格式
2. **GRPO 强化学习**: 15K 条数据, 真实环境里跑 RL, 让模型在 trial-and-error 中学会 orchestration

序列长度也从 8K 拉到 32K, 因为多轮交互轨迹本来就长。

**这里有个关键的 ablation**: 如果跳过 Stage 1 直接搞 Stage 2 (相当于不学单技能直接学组合), DABStep 只有 30.66; 如果把 Stage 1 和 Stage 2 数据混一起一次性训, 36.89; **老老实实分两阶段走 curriculum, 38.88**。

这就是 curriculum 的价值: 数据量一样, 顺序不同, 效果差出 2 个点。就跟教小孩一样, 先加减法再乘除法, 比一锅烩教效果好, 哪怕讲的内容一样。

---

## GRPO 这个优化器, 简单说

RL 训练一般要用一个 critic 网络估 "这个 state 值多少分", 然后让 actor 往更高分方向走。但 critic 网络不好训, 在 data science 这种 reward 极稀疏的场景里, critic 几乎学不准。

GRPO 的 trick 是: **不要 critic 了**, 直接对同一个 question 采样 G 个 output, 算这 G 个 reward 的均值方差, 用 (r_i - mean) / std 当 advantage。高于组平均的 output 就鼓励, 低于的就抑制。

公式长这样:

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i, \, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right) A_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})\right)\right]
$$

变量解释:
- $\theta$: 当前正在优化的 policy 参数
- $\theta_{\text{old}}$: 采样时的旧 policy 参数
- $G$: 每个 question 采样的 output 数量 (group size)
- $o_i$: 第 $i$ 个 output
- $\pi_\theta(o_i|q)$: 当前 policy 给 $o_i$ 的概率
- $\pi_{\text{ref}}$: 参考 policy (一般是 RL 开始时的 SFT model), 用来 KL 约束防止 policy 跑偏太远
- $A_i$: group-relative advantage, 就是 $\frac{r_i - \text{mean}(r)}{\text{std}(r)}$
- $\varepsilon$: PPO 那个 clip 范围, 一般 0.1-0.2
- $\beta$: KL penalty 系数

直觉就是: 同一道题让模型答 G 遍, 答得比同组平均好的就强化, 答得差的就弱化, 不需要单独训 critic。简单粗暴但有效, DeepSeek 那边已经在数学推理上证明可行了。

---

## Reward 设计也有讲究

他们把任务分三类, 每类 reward 不一样:

**所有任务先过格式关**: 如果 5 个 special token 用得不对 (比如漏了 `<Execute>` 或者格式乱), 直接 $R = -1$, 啥也别说了先学会规矩。

**有标准答案的任务** (TableQA、data-centric tasks):

$$
R = \frac{1}{2}(\mathbb{1}_{\text{acc}}(o) + S_{\text{interaction}}(o))
$$

一半分给最终答案对不对, 一半分给交互过程质量。这个 $S_{\text{interaction}}$ 是过程监督, 防止模型瞎蒙答案。

**开放式研究任务** (没有标准答案的):

$$
R = \frac{1}{3}\left(S_{\text{report}}(o) + \min\left(\frac{|T|}{N^T}, 1\right) + \frac{1}{|T|}\sum_{T_i \in o}\mathbb{1}_{\text{success}}(T_i)\right)
$$

三个 term 分别是:
- $S_{\text{report}}(o)$: 用 LLM-as-judge 给最终报告打分, 5 个维度 (usefulness, richness, soundness, interpretability, readability)
- $\min(|T|/N^T, 1)$: 交互轮数, $|T|$ 是实际轮数, $N^T = 10$ 是上限, 鼓励多交互但封顶防刷分
- $\frac{1}{|T|}\sum\mathbb{1}_{\text{success}}(T_i)$: 交互轮次的平均成功率, 鼓励高效交互别瞎试

这个设计的妙处在于三个 term 各有各的意图, 组合起来既防偷懒也防刷分。

---

## 数据合成是隐藏的工程亮点

500K 训练数据哪来的? 这是这个工作最耗时耗力的地方。

**单技能数据 (Stage 1 用)**: 现有数据集只有 instruction-response 对, 没有 reasoning trace。他们就:
1. 用 SOTA LLM 当 teacher, 蒸馏出 reasoning trajectory, 跟 ground truth 比对验证
2. 把 reasoning 拆成 `<Analyze>` 和 `<Understand>` 两块
3. 用关键词引导 refinement, 比如插入 "Let's take a closer look at the table" 这类话, 强迫 reasoning 更关注结构化数据

这个关键词 refinement 看起来不起眼, 但 WikiTQ 上从 78.80 又涨到 80.25, 1.45 个点, 稳定收益。

**多轮交互数据 (Stage 2 用)**: 这是 multi-agent 系统:
- **Questioner agent**: 看 environment 里的数据源, 按采样到的 task type 出题, 同时生成 checklist (这题怎么算做对)
- **Solver agent**: 用 5 个 action 在 environment 里交互, 把题做了
- **Inspector agent**: 验证 solver 的 trajectory, 既看交互过程, 也看 environment 变化 (比如要求生成 result.csv, 那就 check 文件是否真生成了)

**为什么 Inspector 要双验证?** 因为只看 final answer 对不对的话, trajectory 中间步骤可能很 noisy, 模型学了反而学坏。双验证强制 trajectory 在过程层面也合理, 这是 20K + 15K 这么小规模 RL 数据就能 work 的关键。

---

## 实验结果最打动我的几点

1. **DABStep Hard level 32.80, 第二名只有 28.04**: 复杂任务上 deepanalyze 把所有 workflow agent 都甩开, 这说明 agentic 训练学到的不是某个特定 workflow, 是 "根据当前状态决定下一步" 的元能力, 泛化更好。

2. **o1-mini 在 DataSciBench 只有 29.77**: 这个数字特别有启发。o1-mini reasoning 能力强得不行, 但 data science 不只需要 reasoning, 还需要 instruction following + strategic planning + 多步执行。纯 reasoning 模型在 agentic 任务上不一定强。这印证了 paper 的核心 thesis: **agentic capability 是独立维度, 需要专门训**。

3. **DSBench Success Rate 90.63% 全场最高**: 8B 模型在 "能不能跑通" 这个维度超过 GPT-4+AutoGen 的 87.84%。说明 RL 训练让模型学会了 "怎么 recover 错误、怎么改代码再试", 这种 adaptive optimization 是 workflow agent 没有的。

4. **Data Modeling 那个细分项只有 33.33, GPT-4o 也才 57.67**: 说明 data modeling (选算法、调参、评估) 是整个 data science pipeline 里最难的, 8B 模型的 prior knowledge 不够。这是 future work 的方向。

---

## 我看完的整体感觉

这个工作其实回答了一个挺深的问题: **agentic 能力到底是从 scale 来的, 还是从训练范式来的?**

之前大家隐约觉得, 要做 agent 就得上最强 LLM, 因为 agent 需要 reasoning、需要 tool use、需要长 context。GPT-4o + AutoGen 这套范式就是基于这个假设。

但这篇 paper 用 8B 小模型证明: **agentic capability 可以通过 RL 在特定 domain 内训出来**, 不需要等 model scale 撑到天上去。前提是你要:
- 设计合适的 action space (5 个 special token)
- 用 curriculum 解决 reward sparsity
- 用高质量合成数据解决 trajectory scarcity
- 用分层 reward 引导不同层级的能力

这跟你之前讲 "Software 2.0" 的思路挺像的: behavior 内化到 weights 里, 而不是外挂在 prompt 和 framework 上。只不过 DeepAnalyze 把这个 idea 用到了 agentic data science 这个垂直领域。

模型、代码、500K 数据全开源 (ruc-datalab/DeepAnalyze), 这是社区可以接着玩的好底座。

参考链接:
- DeepAnalyze 主页: https://ruc-deepanalyze.github.io/
- GRPO 原文 (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Search-R1: https://arxiv.org/abs/2503.09516
- DeepResearcher: https://arxiv.org/abs/2504.03160
- DSBench: https://openreview.net/forum?id=DSsSPr0RZJ
- DABStep: https://arxiv.org/abs/2506.23719
- DataSciBench: https://arxiv.org/abs/2502.13897

要不要我再展开哪一块? 比如 multi-agent trajectory synthesis 的具体 prompt 怎么设计的, 或者 GRPO 跟 PPO 的细节对比, 或者 ablation 数据的更细解读?

---

# DeepAnalyze: Agentic LLM for Autonomous Data Science 深度解读

你好 Karpathy, 这篇 paper 是最近做 agentic data science 比较扎实的工作, 我从 architecture、training paradigm、reward modeling、trajectory synthesis 到实验细节层层拆解, 顺便串联 Search-R1、DeepResearcher、R1、GRPO 这些相关脉络, 帮你 build intuition。

---

## 1. Paper 的核心 thesis

传统 data science agent 的方法路径有两条, 都走不通:
- **Domain-specific LLMs** (TableGPT, StructGPT, CodeLlama): 只擅长单一任务, 没有 orchestration 能力
- **Workflow-based agents** (AutoGen, ReAct, DS-Agent, Data Interpreter): 依赖手工设计的 workflow + 强 proprietary LLM, 比如用 GPT-4 套 ReAct 框架, 但 workflow 本身是 brittle 的, 一旦 task 偏离预定义流程就崩溃

DeepAnalyze 的核心主张是: **把 workflow 内化到 model weights 里**, 通过在 real-world environments 中 agentic training, 让 8B 模型学到 autonomous orchestration + adaptive optimization 两个能力。结果是在 12 个 benchmark 上 8B 模型超越大部分 proprietary LLM workflow agents, 这是 agentic training paradigm 在 data science 领域的首次成功移植 (之前主要在 search 和 coding 上 work)。

参考:
- DeepResearcher: https://arxiv.org/abs/2504.03160
- Search-R1: https://arxiv.org/abs/2503.09516
- Search-o1: https://arxiv.org/abs/2501.05366
- AutoGen: https://openreview.net/forum?id=BAakY1hNKS
- ReAct: https://openreview.net/forum?id=WE_vluYUL-X
- DS-Agent (case-based reasoning): https://openreview.net/forum?id=LfJgeBNCFI
- Data Interpreter: https://arxiv.org/abs/2402.18679

---

## 2. Architecture: 把 Agent Protocol 编码进 vocabulary

DeepAnalyze 扩展了 base LLM 的 vocabulary, 新增 5 个 special tokens, 对应 5 种 action:

| Action Token | 功能 | 类比 |
|---|---|---|
| `<Analyze>...</Analyze>` | planning / reasoning / reflection / self-verification | ReAct 的 Thought |
| `<Understand>...</Understand>` | 理解 structured data (databases / tables / docs) | 这是新增的, 关键创新 |
| `<Code>...</Code>` | 生成 Python (data science 用) | ReAct 的 Action |
| `<Execute>...</Execute>` | 执行 code, 接收 environment feedback | Observation, 但 model 自己生成 boundary |
| `<Answer>...</Answer>` | 最终答案 | terminate signal |

**关键点 1: `<Understand>` 是独立的 action**。这是与一般 ReAct agent 的本质区别。ReAct 把 thought 和 action 混在一起, DeepAnalyze 把 "理解 structured data" 显式分离成独立的能力维度。Ablation (Table 6) 显示去掉 `<Understand>` 后 DABStep 从 38.88 掉到 31.78, 跌 7 个点, 表明 structured data understanding 是 reasoning 之外的一个独立维度。

**关键点 2: 输入的双模态**。模型既接收 text form 的 structured data (markdown table, 适合小表), 又接收 external file names (适合大表), 后者由模型主动 inspect。这模仿人类 data scientist 不会去 "背诵" 整张大表, 而是按需 explore 的行为。

**关键点 3: Algorithm 1 推理循环**

```
while ⟨Answer⟩ not in A:
    y ← M(Q, A)               # 模型自己决定下一步
    A ← A + y
    if ⟨Code⟩ in y:
        code ← extract_code(y)
        feedback ← Env.execute(code)
        A ← A + ⟨Execute⟩ + feedback + ⟨/Execute⟩
```

注意这里 **没有任何人工 workflow 控制器**, 模型自己决定什么时候 generate code、什么时候切换到 `<Analyze>`、什么时候 terminate。这就是 paper 强调的 "fully autonomous orchestration"。

参考 Algorithm 1 在 paper Section 3.1, 推理引擎用 vLLM: https://arxiv.org/abs/2309.06180

---

## 3. Curriculum-based Agentic Training: 两阶段提升

这是 paper 的核心 training paradigm, 直接对应两个挑战:

| 挑战 | 解决方案 |
|---|---|
| **Reward sparsity** (基础 LLM 做不了复杂 data science, 拿不到正 reward) | Stage 1 Single-ability SFT 先打基础 |
| **Trajectory scarcity** (真实 data science long-chain trajectory 极少) | Data-grounded trajectory synthesis |

### Stage 1: Single-ability Fine-tuning
Base model: **DeepSeek-R1-0528-Qwen3-8B** (这是个关键选择, R1 已经有 reasoning 能力)。
数据: ~470K samples, 序列长度 8K。
覆盖三种 single ability:
- Reasoning (对应 `<Analyze>`): 用 R1-distill 的 long CoT 数据
- Structured data understanding (对应 `<Understand>`): TableQA、Structured KG 等数据集的 reasoning trajectory
- Code generation (对应 `<Code>`): data science code 生成数据

这一步是模仿 "beginner → data science practitioner" 学习单一技能的过程。

### Stage 2: Multi-ability Agentic Training
两步:
- **Cold start**: SFT 在 20K synthesized interaction trajectories 上, 让模型学会 orchestrate 这 5 个 action 的交互格式
- **RL with GRPO**: 15K samples, 在 real environment 中通过 RL 进一步强化

序列长度从 8K 提升到 32K, 因为 multi-turn interaction trajectory 更长。

**为什么不能直接做 RL-Zero 或 One-stage Training?** Table 7 ablation 给出答案:
- Only Single-ability Fine-tuning: DABStep 只有 15.34 (没有 multi-turn orchestration 能力)
- Only Multi-ability Agentic Training (跳过 Stage 1): DABStep 30.66 (单能力没打牢, RL 探索效率低)
- One-stage Training (单能力 + 多能力数据混一起做 cold-start): DABStep 36.89
- **Curriculum (先单后多): 38.88**

这印证了: 在 multiple ability 复合的高难度任务上, **从 simple 到 complex 的 schedule 比直接混合训练更有效**, 即使数据量相同。

参考 DeepSeek-R1 关于 cold-start 的讨论: https://arxiv.org/abs/2501.12948

---

## 4. GRPO 目标函数详解 (Equation 1)

paper 直接用 Group Relative Policy Optimization, 来自 DeepSeekMath。我把公式逐项拆解:

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim D, \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min\left( r_i(\theta) A_i, \, \text{clip}(r_i(\theta), 1-\varepsilon, 1+\varepsilon) A_i \right) - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}}) \right) \right]
$$

其中 $r_i(\theta) = \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}$ 是 importance ratio。

**变量逐个解释**:
- $\theta$: 当前 policy 网络参数 (正在优化的)
- $\theta_{\text{old}}$: 旧 policy 参数 (sampling 时用的, 用于 importance sampling)
- $D$: 训练数据集 (这里指 DataScience-Instruct-500K 中 RL 部分的 15K samples)
- $q$: 从 $D$ 中采样的 question / instruction
- $G$: 每个 question 采样的 group size (一次采样 G 个 outputs)
- $\{o_1, \dots, o_G\}$: 从 $\pi_{\theta_{\text{old}}}$ 采样的 G 个 candidate outputs
- $\pi_\theta(o_i|q)$: 当前 policy 给定 $q$ 生成 $o_i$ 的概率
- $\pi_{\theta_{\text{old}}}(o_i|q)$: 旧 policy 给定 $q$ 生成 $o_i$ 的概率
- $A_i$: **group-relative advantage**, 关键创新点, 后面单独讲
- $\pi_{\text{ref}}$: reference model (通常是 RL 开始时的 SFT model, 用于 KL anchor)
- $\varepsilon$: PPO clip 范围, 通常 0.1-0.2, 防止 importance ratio 偏离太远
- $\beta$: KL penalty 系数, 控制 policy 不要 drift 太远
- $D_{KL}$: KL divergence, $\sum_x \pi_{\text{ref}}(x) \log \frac{\pi_{\text{ref}}(x)}{\pi_\theta(x)}$

**$A_i$ 的计算 (group-relative)**:
传统 PPO 用 value network 估 baseline, GRPO 去掉 value network, 直接用 group 内 reward 的归一化:

$$
A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}
$$

这个设计的 intuition 是: 同一个 question 采样 G 个 outputs, 然后让 reward 高于组均值的 output 概率上升, 低于的下降, 不需要训练 value network。对 data science 这种 task-level reward 极稀疏的场景特别合适, 因为 critic 很难学好。

参考 GRPO 原文: https://arxiv.org/abs/2402.03300

---

## 5. Hybrid Reward Modeling: 三类 task 三种 reward

paper 把 data science 任务分成三类, 设计不同的 reward:

### 5.1 Format check (所有任务)
如果输出格式不正确 (5 个 action 不完整或格式错), **直接 $R = -1$**。这是个 hard gate, 强制模型先学会 interaction protocol。

### 5.2 Data QA + data-centric tasks (有 reference answer, Equation 2)

$$
R = \frac{1}{2} \left( \mathbb{1}_{\text{acc}}(o) + S_{\text{interaction}}(o) \right)
$$

- $R$: total reward
- $\mathbb{1}_{\text{acc}}(o) \in \{0, 1\}$: indicator, 最终答案是否正确
- $S_{\text{interaction}}(o) \in [0, 1]$: interaction trajectory 质量分, 评估 code 是否合理、是否有多余步骤等

这是 rule-based reward, 一半权重给答案正确性, 一半给过程质量。$S_{\text{interaction}}$ 的具体计算 paper 没展开, 但显然是个 process supervision signal。

### 5.3 Open-ended data research (无 reference, Equation 3)

$$
R = \frac{1}{3} \left( S_{\text{report}}(o) + \min\left(\frac{|T|}{N^T}, 1\right) + \frac{1}{|T|} \sum_{T_i \in o} \mathbb{1}_{\text{success}}(T_i) \right)
$$

- $S_{\text{report}}(o)$: LLM-as-judge 给最终 report 打的分, 5 个维度 (usefulness, richness, soundness, interpretability, readability)
- $|T|$: output $o$ 中与 environment 的交互轮数
- $N^T = 10$: 超参数, 交互轮数的归一化上限
- $T_i$: 第 $i$ 轮交互
- $\mathbb{1}_{\text{success}}(T_i)$: indicator, 第 $i$ 轮交互是否成功 (code 执行成功、数据获取成功等)

**这个 reward 设计有三层意图**:
1. $S_{\text{report}}$ 引导报告质量
2. $\min(|T|/N^T, 1)$ 鼓励充分 interaction (但 cap 在 1, 防止刷轮数)
3. $\frac{1}{|T|} \sum \mathbb{1}_{\text{success}}$ 鼓励高效交互, 不要瞎试错

第三个 term 是 process reward 的关键, 它 average 了所有 turn 的成功率, 等于惩罚 "无效 interaction"。

---

## 6. Data-grounded Trajectory Synthesis: 两类合成

500K 训练数据的来源是个隐藏的工程亮点。paper 把数据合成分成两个独立框架:

### 6.1 Reasoning Trajectory Synthesis (用于 Stage 1)

针对已有 instruction-response pair 但缺 reasoning trace 的数据集 (TableQA、StructLM 等):

1. **Distillation**: 用 SOTA closed-source LLM 作 teacher, 生成 reasoning trajectory, 与 ground truth 比对验证
2. **Reformulation**: 把 reasoning 重写成 `<Analyze>` (reasoning process) + `<Understand>` (structured data understanding) 两个互补组件
3. **Keyword-guided refinement**: 这是创新点。从 reasoning vocabulary 中采样关键词 (如 "What happens at the boundaries?", "Let's review the prior reasoning", "Let's take a closer look at the table") 插入 reasoning trajectory, 引导模型更关注 structured data

Table 8 ablation: WikiTQ 从 75.54 → 78.80 (+distill) → 80.25 (+refinement), refinement 单独贡献 1.45 个点。

**这个 keyword-guided refinement 的 intuition**: 类似 R1 / O1 类推理模型中 "wait", "but", "let me reconsider" 这类 reflection trigger token 的作用, 但这里显式用于强化 structured data focus。参考 "Reasoning models know when they're right": https://arxiv.org/abs/2504.05419

### 6.2 Interaction Trajectory Synthesis (用于 Stage 2)

这是 multi-agent framework, 三个 role:

| Role | 职责 |
|---|---|
| **Questioner** | 观察 environment 中的 data source (来自 Spider / BIRD), 按采样到的 task type 生成 data science problem + checklist (interaction-level constraints + environment-level constraints) |
| **Solver** | 用 5 个 action 在 environment 中交互, 完成任务, 生成 trajectory |
| **Inspector** | 验证 trajectory: 检查 interaction 过程是否符合 checklist, 同时验证 environment 变化 (新生成的文件名对不对、内容对不对) |

**为什么基于 NL2SQL datasets (Spider, BIRD)?** 因为它们有大量真实 structured data source, 可以复用 environment, 但任务类型从 NL2SQL 扩展到完整 data science pipeline。

**为什么 Inspector 双重验证 (interaction + environment)?** 传统 trajectory validation 只看 final answer 对不对, 这里加上 environment state check (比如要求生成 `result.csv`, 那要 check 文件是否真生成、内容是否符合), 大幅提升 trajectory 质量。这是数据合成质量的关键。

参考:
- Spider: https://aclanthology.org/D18-1425/
- BIRD: https://arxiv.org/abs/2305.03111

---

## 7. 实验数据精读

### 7.1 DataSciBench (Table 1, end-to-end pipeline)

| Model | Success Rate | Score |
|---|---|---|
| GPT-4o | 66.31 | 64.51 |
| **DeepAnalyze-8B** | **59.91** | **61.11** |
| GPT-4o-mini | 50.63 | 54.18 |
| Claude-3.5-Sonnet | 47.48 | 52.29 |
| GPT-4-Turbo | 51.93 | 54.65 |
| o1-mini | 29.77 | 38.78 |
| Qwen2.5-Coder-7B | 45.18 | 47.67 |
| Llama-3.1-8B-Instruct | 24.73 | 29.69 |

注意 **o1-mini 只有 29.77**, 这是个有意思的现象: o1-mini reasoning 强但 instruction following + strategic planning 弱, data science task 不只需要 reasoning, 更需要 orchestration。这印证了 paper 的 thesis——reasoning ≠ agentic capability。

Fine-grained metrics (F1-F5) 显示 DeepAnalyze 在 Data Preparation (71.68)、Plot Validity (67.86)、Data Exploration (58.62)、Data Visualization (69.09) 都领先开源 LLM agents, 唯独 Data Modeling (33.33) 较弱, 但 GPT-4o 也只有 57.67, 说明 modeling 是难点。

参考 DataSciBench: https://arxiv.org/abs/2502.13897

### 7.2 DSBench (Table 2, data modeling)

DeepAnalyze-8B **Success 90.63%**, Performance 39.41, **Success Rate 全场最高** (GPT-4-based AutoGen 是 87.84%)。但 Performance metric 略低于 GPT-4 (45.52)。这说明 DeepAnalyze 在 "能不能跑通" 维度极强, 在 "做得多精" 维度还有提升空间, 这正是 8B 参数的 capacity 局限。

参考 DSBench: https://openreview.net/forum?id=DSsSPr0RZJ

### 7.3 DABStep (Table 3, multi-step reasoning)

| Model | Easy | Hard | Overall |
|---|---|---|---|
| **DeepAnalyze-8B** | **70.83** | **32.80** | **38.88** |
| I2I-Agent (Claude-3.5-Sonnet) | 80.56 | 28.04 | 36.44 |
| Open Data Scientist (Deepseek-v3) | 84.72 | 16.40 | 27.33 |
| o4-mini (reasoning prompt) | 76.39 | 14.55 | 24.44 |
| o3-mini | 72.22 | 13.76 | 23.11 |
| GPT-4.1 | 80.56 | 12.43 | 23.33 |
| Claude-3.5-Sonnet | 77.78 | 9.26 | 20.22 |

**关键 insight**: DeepAnalyze 在 Easy level 不是最高 (I2I-Agent 80.56 更高), 但在 Hard level 32.80 显著领先所有 baselines (第二名只有 28.04)。这表明:
- Workflow agent 在简单任务上靠 LLM 本身能力吃红利
- 复杂任务上, **autonomous orchestration + adaptive optimization** 击败手工 workflow
- Hard level 跌幅小 = 模型对 long-chain reasoning 鲁棒性更强

参考 DABStep: https://arxiv.org/abs/2506.23719

### 7.4 TableQA (Table 4, 7 个子集)

DeepAnalyze-8B 平均 **64.47**, 超过 Reasoning-Table (SFT+RL) 的 62.62、DeepSeek-R1-0528 的 60.22、GPT-4o 的 58.96、Claude 的 58.79。单看 MultiHiertt (48.29) 和 HiTab (78.16) 提升特别明显, 这两个是 hierarchical table 的难点。

### 7.5 DS-1000 (Table 5, code generation)

| Model | Pandas | NumPy | Matplotlib | Sklearn | SciPy | TF | PyTorch | Overall |
|---|---|---|---|---|---|---|---|---|
| **DeepAnalyze-8B** | 50.2 | 74.5 | 67.7 | 56.5 | 54.7 | 68.9 | 70.6 | **61.7** |
| GPT-4-Turbo | 42.3 | 61.8 | 71.6 | 50.4 | 50.0 | 53.3 | 50.0 | 53.9 |
| DeepSeek-R1-0528-Qwen3-8B (base) | 17.5 | 37.3 | 52.9 | 27.8 | 21.7 | 31.1 | 29.4 | 30.4 |
| DeepAnalyze-8B (single-ability only) | 43.6 | 69.1 | 54.8 | 53.0 | 50.9 | 64.4 | 58.8 | 54.8 |

Base model 在 DS-1000 上只有 30.4, Stage 1 single-ability fine-tuning 直接拉到 54.8 (+24.4), Stage 2 agentic training 进一步到 61.7 (+6.9)。这显示了 curriculum 的两阶段收益分布: **Stage 1 主要建立 single ability, Stage 2 通过 environment interaction 强化 composite ability**。

---

## 8. Ablation 深读

### 8.1 `<Understand>` action 的价值 (Table 6)

去掉 `<Understand>`:
- WikiTQ: 83.24 → 80.78 (-2.46)
- MultiHiertt: 48.29 → 45.43 (-2.86)
- DS-1000: 61.70 → 61.20 (-0.50)
- DABStep: 38.88 → 31.78 (-7.10)

DABStep 跌得最多, 说明 multi-step data analysis 任务最依赖 explicit 的 structured data understanding。WikiTQ/MultiHiertt 是 pure TableQA, 理解需求直接, 跌幅小。DS-1000 是 code generation, structured understanding 影响较小。

### 8.2 Curriculum 训练 (Table 7)

四种设置在 DABStep 上的对比:
- Curriculum (本文): 38.88
- One-stage (混合训练): 36.89
- Only Multi-ability Agentic: 30.66
- Only Single-ability: 15.34

最有趣的对比是 **Curriculum vs One-stage**: 数据量相同, 只是训练顺序不同, Curriculum 高 1.99 个点。这说明 schedule 本身 (先易后难) 提供了额外的 inductive bias, 类似人学习时先掌握加法再学乘法, 比混在一起学更高效。

### 8.3 Reasoning Trajectory Synthesis (Table 8)

WikiTQ 三个版本:
- Original (Reasoning-Table 数据): 75.54
- + Distillation: 78.80 (+3.26)
- + Keyword Refinement: 80.25 (+1.45)

distillation 是大头收益, keyword refinement 是 marginal 但稳定的提升。

---

## 9. 与其他 agentic training 工作的关系

### 9.1 与 Search-R1 / DeepResearcher 的对比

Search-R1、DeepResearcher 把 agentic RL 应用到 search domain: 模型在 loop 中调用搜索引擎、读 results、再 reasoning。DeepAnalyze 的不同点:
- **Environment 类型**: search domain 是 read-only (search engine 返回结果), data science 是 read-write (执行 code, 改变 environment state, 生成文件)
- **Action space**: search 通常只有一个 action (search), data science 有 5 个 action (analyze/understand/code/execute/answer)
- **Reward**: search 用 answer accuracy, data science 需要 hybrid reward (format + accuracy + report quality + interaction quality)
- **Trajectory 长度**: data science trajectory 通常更长 (multi-stage pipeline), 因此 reward sparsity 更严重, 必须 curriculum

### 9.2 与 DeepSeek-R1 的关系

DeepAnalyze 用 R1-0528-Qwen3-8B 作 base, 继承了 R1 的 long CoT reasoning 能力。但 R1 是单轮 reasoning, 没有 environment interaction。DeepAnalyze 的 Stage 2 agentic training 把 R1 的 reasoning 能力扩展到 multi-turn agentic setting, 关键是用 cold-start SFT 让模型学会 "在 reasoning trace 之间插入 `<Code>` + `<Execute>`" 的格式, 然后 RL 强化。

### 9.3 与 RL-Zero vs Cold-start SFT 的讨论

DeepSeek-R1 paper 讨论了 R1-Zero (直接 RL) vs R1 (cold-start + RL) 的取舍。R1-Zero 在 reasoning 上 work, 但 format 不稳定, 最终采用 cold-start + RL。DeepAnalyze 把这个讨论延伸到 agentic data science:
- RL-Zero 在 data science 完全失败 (reward 太稀疏, base 模型直接交白卷)
- One-stage cold-start (混入单能力数据) 比 pure multi-ability 好, 但不如 curriculum
- Curriculum (SFT 单能力 → cold-start multi-ability → RL) 最优

---

## 10. Build Your Intuition: 我看到的几个关键 insight

### Insight 1: Agentic capability 是 emergent 的, 不是 prompting 出来的
传统 wisdom 是 "用 strong LLM + complex prompt + tool use" 就能做 agent。DeepAnalyze 显示 8B 模型通过 RL 训练后, 即使 single-turn reasoning 弱于 GPT-4o, 在 multi-turn agentic task 上反而更强。这暗示 **agentic capability 是一个独立的能力维度**, 需要专门训练, 不能靠 LLM scale 完全解决。

### Insight 2: Workflow 内化 vs 外部框架的 trade-off
Workflow-based agent (AutoGen + GPT-4) 在 known task pattern 上很 robust, 但泛化到 novel pattern 就脆。DeepAnalyze 把 workflow 编码进 model weights, 通过 RL 在 diverse tasks 上训练, 学到的不是某个 workflow, 而是 "如何根据当前 state 决定下一步 action" 的元能力。代价是训练成本高、不易 debugging。

### Insight 3: Curriculum 在 RL 中的作用被低估
传统 RL 习惯 end-to-end 训练, 但在 high-dimensional action space + sparse reward 场景下, curriculum 几乎是 must-have。DeepAnalyze 的两阶段对应 "先掌握 building blocks 再组装" 的人类学习模式。这个 idea 在 OpenAI 的 O1 训练、Anthropic 的 constitutional AI 里也有隐含身影, 但 paper 把它显式化应用到 agentic domain。

### Insight 4: Trajectory quality > trajectory quantity
Inspector 同时验证 interaction details + environment state change 这一点很关键。传统 SFT 数据往往只看 final answer 对不对, 导致 trajectory 中间步骤可能是 noisy 的。Inspector 强制 trajectory 在 process level 也合理, 这是为什么 20K + 15K (相对小规模) RL 数据就能 work 的关键。

### Insight 5: Reward shaping 的层次
paper 的 reward 设计是一个三层 pyramid:
1. **Format reward** (hard gate, R=-1 if format wrong): 学会 protocol
2. **Task reward** (accuracy + interaction quality): 学会 solve task
3. **Report reward** (5 个 quality 维度): 学会产出 analyst-grade artifact

这个分层对应 Bloom taxonomy: 先 remember/understand, 再 apply/analyze, 最后 evaluate/create。这是 curriculum 训练的另一种体现, 在 reward 层面也分层。

---

## 11. Limitations 和潜在改进

我从 paper 字里行间读出几个局限:

1. **Data Modeling 弱项**: Table 1 中 F5 (Data Modeling) 只有 33.33, 远低于其他 sub-task。这暗示 8B model 对 ML algorithm selection、hyperparameter tuning 这种需要深厚 prior knowledge 的任务还是吃力。改进方向: 引入 ML-specific reward、更多 modeling trajectory。

2. **Single 8B 模型, 没有 mixture of experts 或 larger variant**: paper 只 release 8B, 没尝试 32B/70B 的 scaling 曲线。如果能训 32B, DABStep Hard level 有望突破 50。

3. **Reward hacking 风险**: LLM-as-judge 给 $S_{\text{report}}$ 评分, 模型可能学会 "看起来像好报告" 而非 "实质上是好报告" 的 hack 方式。Paper 没讨论这个, 但实际部署会是个问题。

4. **Environment 是 sandbox Python**: 真实 data scientist 还要连接 SQL DB、调用 API、与 BI tool 集成, 当前 environment 主要是 file system + Python execution。扩展到 SQL execution、云 API call 是 future work。

5. **Trajectory synthesis 依赖 Spider/BIRD**: 这些数据集本身偏简单 SQL, 难以合成极复杂的 open-ended research trajectory。可能需要更多 domain-specific data source (Kaggle notebooks、Jupyter notebooks from real analysts)。

6. **没有显式 multi-modal**: 当前只处理 structured data (CSV/JSON/tables) 和 text, 没涉及 chart understanding、image data analysis 等 multimodal data science。

---

## 12. 我个人的几个直觉观察

**关于 architecture 的简洁性**: 5 个 special token 这个设计极简但有效。每个 token 对应一个 "mental action" (planning / data inspection / code generation / feedback intake / output), 像把人类 data scientist 的 cognitive cycle 显式化。这种显式分解比 ReAct 的 thought-action 二分更细粒度, 在 data science 这种多步任务上更适配。

**关于 R1 作 base 的选择**: 不是随机选的。R1 已经有 long CoT 能力, 在 single-ability SFT 阶段不用从零学 reasoning, 只需要学 structured data understanding + code generation。这就是为什么 470K samples 就能从 DS-1000 30.4 跳到 54.8。如果用 Qwen2.5-7B-Instruct (DS-1000 30.4 也是它的水平) 作 base, 可能需要更多数据。

**关于 inference cost**: DeepAnalyze 推理时是 multi-turn 的, 一条 query 可能要 5-20 turn interaction。这比 single-turn LLM call 慢且贵, 但相对 GPT-4o + AutoGen workflow 反而便宜 (因为不需要调用 expensive API)。这是 8B 开源模型的成本优势。

**关于 open-ended research 的真正价值**: DABStep-Research 是 paper 新建的 benchmark, 显示 DeepAnalyze 在 open-ended data research 上明显超越 GPT-4o agent (Figure 8)。这是 paper 最有 "AI scientist" 影子的一节, 类似 Sakana AI 的 "AI Scientist", 但聚焦 data-centric research。

参考 AI Scientist: https://arxiv.org/abs/2408.06292

---

## 13. 总结: 这篇 paper 的位置

DeepAnalyze 在我看来是 **三个 trend 的交汇点**:

1. **Agentic RL training** (来自 R1、Search-R1、DeepResearcher 的范式): 把 RL 从 single-turn reasoning 扩展到 multi-turn interaction
2. **Domain-specific agentic model** (vs general agent): 不试图做通用 agent, 聚焦 data science 这一垂直领域, 因此能定义清晰的 5 个 action + 3 类 reward
3. **Trajectory synthesis as data engineering** (来自 SFT 数据合成的成熟范式): 用 multi-agent system + dual validation 自动构造高质量 RL 数据

对 Karpathy 你之前讲过的 "Software 2.0" 和 "Software 3.0" 框架, DeepAnalyze 是个有趣的中间状态: 它仍然是 Software 2.0 (weights encode behavior), 但 behavior 本身是 agentic 的 (类似 Software 3.0 的 prompt-driven agent, 但内化到 weights 里)。这种 "agent-as-model" 范式可能是接下来几年 LLM 应用的主流——不靠 framework 编排, 而靠 weights 内化 orchestration。

**最值得借鉴的工程经验**:
- Curriculum training 在 sparse reward RL 中几乎是 must-have
- Reward shaping 要分层 (format → task → quality)
- Trajectory synthesis 要 dual validation (process + outcome)
- Special token action space 比自然语言 prompt action space 更易训
- 8B 模型在垂直 domain 上完全有可能超 GPT-4o, 前提是 RL 训练充分

paper 的代码、模型、500K 训练数据全部开源 (ruc-datalab/DeepAnalyze, DataScience-Instruct-500K), 这对社区复现和扩展极有价值, 是个值得 fork 研究的项目。

参考 DeepAnalyze 主页: https://ruc-deepanalyze.github.io/

如果你打算基于这个做 next experiment, 我建议的方向:
1. 训 32B variant, 验证 agentic capability 是否随 scale 进一步提升
2. 加入 SQL execution environment, 扩展到 enterprise data stack
3. 用 self-play 替代 multi-agent synthesis, 让 DeepAnalyze 自己生成 trajectory 给自己 RL
4. 在 multimodal data science (chart understanding + image-based analysis) 上扩展, 这是 8B 文本模型够不到的地方

希望这个讲解帮到你的 intuition building。如果想深入任何一个 section (比如 GRPO 数学细节、trajectory synthesis 的 multi-agent prompt 设计、或 abation 的更细分析), 我可以再展开。
