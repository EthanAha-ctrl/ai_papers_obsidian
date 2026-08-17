---
source_pdf: On-Policy Context Distillation for Language Models.pdf
paper_sha256: 0164b1b23319253a082a4d4c51a1ac1314f1cb85e86350c3c63234a3c31561ee
processed_at: '2026-08-05T23:43:50-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我用最朴素的话讲一遍。

## 这篇 paper 在干嘛

你用 ChatGPT 的时候，前面贴一长段 system prompt，或者贴几道例题（few-shot），模型表现就变好了。但每次 inference 都要重新贴，很贵，也很蠢——同样的 context 反复读。

**能不能把这些 context 直接"焊"进模型权重里？** 这就是 context distillation 想干的事。

## 老办法的问题

之前有人试过，做法很直白：teacher 模型带着 context 生成一堆答案，student 模型去 mimic 这些答案。但有两个坑：

**坑一**：student 训练时看的是 teacher 写的句子，但 inference 时要自己从零开始写。这就像你抄学霸作业能抄对，让你自己考试就懵了——训练和测试的 distribution 对不上。

**坑二**：loss function 选的是 forward KL，逼 student 把概率 mass 摊到 teacher 的所有可能输出上。结果 student 学了个"什么都要可能"，分布变宽，开始 hallucination。

## 这篇 paper 的 fix

两个改动，简单粗暴：

**改动一**：student 自己写答案，不抄 teacher 的。写完之后让 teacher 打分。这样 student 学的是"在我自己写的过程中该怎么走"，exposure bias 就没了。

**改动二**：loss 换成 reverse KL。这东西的特性是——student 只在自己已经高概率的地方被惩罚或奖励。teacher 觉得 OK 的就加强，teacher 觉得不行就抑制。Long tail 的地方 student 根本不去采样，所以不会浪费时间在没用的地方摊概率。

一句话总结：**student 自己 generate，teacher 在旁边 context 加持下打分，student 只往 teacher 觉得靠谱的方向靠。**

## 两个应用

**应用一：经验积累**

模型做完一道题，自己反思总结出几条"经验"（格式就是 `- EXPERIENCE ITEM: 不要走右边那个洞`）。这些经验积累成 context，下次做题 prepend 上去能涨点。然后用 OPCD 把这些经验内化进权重，以后不用贴了。

这里有个很有意思的结果：如果你直接把原始解题过程（raw trace）贴给模型当 context，**性能反而下降**。因为原始过程充满了 noise——错误尝试、绕弯路、无关步骤。模型得自己 abstract 出"经验"才行，raw data 是 garbage。

**应用二：system prompt 内化**

医学问答的 system prompt 很长，每次 inference 都贴太贵。用 OPCD 蒸馏进权重，inference 时就不用贴了，性能还能涨。

## 最 striking 的实验结果

**跨尺寸蒸馏**：8B teacher 积累的经验，直接塞进 1.7B 的 context 里——**1.7B 反而变差了**。因为 8B 的 reasoning 方式 1.7B 学不来，硬跟反而搞砸。

但用 OPCD 让 1.7B 在自己 trajectory 上学，只取它能力范围内能用的部分——稳定提升。

这个结果我直觉上特别 meaningful：**经验是 model-specific 的，不能直接 transfer，但通过 on-policy alignment 可以间接 transfer。**

**抗遗忘**：用 safety prompt 蒸馏后，老方法在 medical 任务上掉 4 个点，OPCD 几乎不掉。因为 student 只在自己采样到的 region 上更新，medical 那个 manifold 它本来就不去碰，自然不受影响。

**frozen teacher vs self-distillation**：frozen teacher 完胜 self-distillation（Sokoban 上 53.9 vs 18.8）。原因直觉上很清楚——self-distillation 时 teacher 一直在变，student 追一个 moving target，类似 self-play 的不稳定问题。Frozen teacher 就是稳定的 supervised signal。

## 一句话版本

让 student 自己写答案，teacher 带着 context 在旁边打分，用 reverse KL 让 student 只往 teacher 靠近的方向走——既内化了 context，又不会忘掉原本能力，也不会 hallucination。

最反直觉的 finding 是：**经验必须经过 model 自己 abstract，raw experience 是毒药；跨 model 直接 transfer 经验也是毒药，必须经过 on-policy alignment。**

---

# On-Policy Context Distillation (OPCD) — 深度解析

Andrej，这篇 Microsoft Research 的工作把两个想法拼在一起：context distillation 和 on-policy distillation。核心 motivation 是把 "in-context knowledge" 这种 transient 的东西固化进 model weights。我会一层一层拆开来讲，尽量 build 你的 intuition。

## 1. 问题背景：为什么需要这个工作

LLM 的 in-context learning 让你能通过 prepend prompt 来 steer 行为，但这个 knowledge 是 **session-bound 的**：一旦 context reset，所有 insight 都丢失了，下次还要从头 "re-learn"。

Context distillation 早期工作 (Askell et al. 2021, Snell et al. 2022) 想把 context 压进 weights：
- https://arxiv.org/abs/2112.00861 (Askell et al., "A General Language Assistant as a Laboratory for Alignment")
- https://arxiv.org/abs/2209.15189 (Snell et al., "Learning by Distilling Context")

但传统做法有两个 fundamental problems：
1. **Off-policy training** → exposure bias：student 在 teacher 生成的轨迹上训练，但 inference 时要自己 autoregressive 采样，train/test distribution mismatch
2. **Forward KL minimization** → mode-covering：student 试图覆盖 teacher 的所有 mode，包括低概率长尾，导致 distribution 过宽 / hallucination

OPCD 的 fix 就是：用 on-policy sampling + reverse KL。

## 2. 方法：公式细节剖析

### 2.1 主 loss (Equation 1)

$$
\mathcal{L}(\theta) = \mathbb{E}_{(x,c) \sim \mathcal{D},\, y \sim \pi_\theta(\cdot | x)} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} D_{\mathrm{KL}}\left( \pi_\theta(\cdot | x, y_{<t}) \,\|\, \pi_{\mathrm{teacher}}(\cdot | c, x, y_{<t}) \right) \right]
$$

变量解释：
- $\theta$: student model 的可训练参数
- $x$: 输入 prompt (例如一道数学题)
- $c$: 要内化的 context knowledge (经验 / system prompt)
- $y$: **从 student 自己采样** 的完整 response，注意 $y \sim \pi_\theta(\cdot|x)$ —— 这是 on-policy 的关键
- $|y|$: 序列长度，作为归一化
- $t$: token 位置索引，从 $1$ 到 $|y|$
- $y_{<t}$: position $t$ 之前所有已生成 tokens (prefix)
- $\pi_\theta(\cdot|x, y_{<t})$: student 在 prefix 下的下一个 token 分布 (无 context c)
- $\pi_{\mathrm{teacher}}(\cdot|c, x, y_{<t})$: teacher 在同样 prefix 下的分布 (但有 context c)
- $\mathcal{D}$: 训练数据分布 (x, c) pairs
- $\|\|$: KL divergence 的方向 — 注意这里是 **reverse KL**: $D_{\mathrm{KL}}(P \| Q) = \mathbb{E}_P[\log(P/Q)]$，其中 $P$ 是 student，$Q$ 是 teacher

注意 $y$ 是从 student 采样的，所以 prefix $y_{<t}$ 来自 student 的 distribution。这是和 off-policy context distillation 的根本差别。

### 2.2 Token-level reverse KL 展开 (Equation 2)

$$
\begin{aligned}
& D_{\mathrm{KL}}\bigl(\pi_\theta(\cdot | x, y_{<t}) \,\|\, \pi_{\mathrm{teacher}}(\cdot | x, y_{<t})\bigr) \\
&= \mathbb{E}_{y'_t \sim \pi_\theta(\cdot|x,y_{<t})} \left[ \log \frac{\pi_\theta(y'_t | x, y_{<t})}{\pi_{\mathrm{teacher}}(y'_t | c, x, y_{<t})} \right] \\
&= \sum_{y'_t \in \mathcal{V}} \pi_\theta(y'_t | x, y_{<t}) \bigl( \log \pi_\theta(y'_t | x, y_{<t}) - \log \pi_{\mathrm{teacher}}(y'_t | c, x, y_{<t}) \bigr)
\end{aligned}
$$

变量解释：
- $y'_t$: vocabulary 中的任意 candidate token (不是序列里实际的 token)
- $\mathcal{V}$: 完整 vocabulary
- 实现中用 $\mathcal{V}_{\mathrm{top-k}}$ 近似，取 student 预测概率最高的 top-256 tokens (Appendix A.3)

### 2.3 Reverse KL vs Forward KL 的 intuition

这点是最关键的。给定两个分布 $P$ (student) 和 $Q$ (teacher)：

**Forward KL**: $D_{\mathrm{KL}}(Q \| P) = \mathbb{E}_Q[\log(Q/P)]$
- 期望在 $Q$ (teacher) 上取
- 只要 teacher 有概率的地方 $P$ 必须覆盖，否则 $P \to 0$ 时 $\log(Q/P) \to \infty$
- 行为：**mode-covering** — student 必须把概率 mass 摊开覆盖所有 teacher modes
- 当 student capacity 不够时，会变成 mode-averaging，distribution 变得 uninformative

**Reverse KL**: $D_{\mathrm{KL}}(P \| Q) = \mathbb{E}_P[\log(P/Q)]$
- 期望在 $P$ (student) 上取
- 只要 student 自己不采样的地方，loss 不在乎 $Q$ 是多少 ($P=0$ 时乘进去就是 0)
- 但 student 高概率而 teacher 低概率的地方会被严重惩罚
- 行为：**mode-seeking** — student 集中到 teacher 的某个高概率 mode

paper 里用一个很简洁的 intuition 描述 (Section 3)：
- 如果 student 采样到 teacher 觉得 high-prob 的 token → 鼓励 student 进一步提高这个 token 的概率
- 如果 student 采样到 teacher 觉得 low-prob 的 token → 抑制 student 在这个 token 上的概率
- Long tail 直接被忽略，因为 student 不会采样到那里

这正合 context distillation 的需求：你想把 context 里有用的 behavior 内化，不想 student 因为 capacity 不够而 spread probability mass 到 teacher 都不关注的 region。

### 2.4 Algorithm 1 流程

每个 batch：
1. Sample $(x, c) \sim \mathcal{D}$
2. **Student 不看 context c，只看 x，生成 $y \sim \pi_\theta(\cdot|x)$**
3. 对每个 position $t \in [1, |y|]$：
   - Student 计算 $\pi_\theta(y'_t | x, y_{<t})$ for $y'_t \in \mathcal{V}_{\text{top-256}}$
   - Teacher 计算 $\pi_{\mathrm{teacher}}(y'_t | c, x, y_{<t})$ (input 是 $[c; x; y_{<t}]$ 拼接)
   - Compute reverse KL
4. 平均 over positions
5. Update $\theta$ via gradient descent

注意 teacher 的 input 是 $[c; x; y_{<t}]$ — teacher 看到了 student 自己生成的 prefix $y_{<t}$，而**不只是看 teacher 自己生成的 prefix**。这是 on-policy 的本质。

### 2.5 Teacher 配置

paper 区分两种：

**Teacher-Student Distillation** ($\pi_{\mathrm{teacher}} \neq \pi_\theta$):
- Teacher 可以是更大的 model，或者同 init 但 frozen 的 model
- Section 4.6 显示 frozen teacher 更稳定 (Table 5)
- 这是 paper 的 default configuration

**Self-Distillation** ($\pi_{\mathrm{teacher}} = \pi_\theta$):
- 同一个 model，teacher 看 $[c;x]$，student 看 $[x]$
- 持续更新的 teacher 引入高 variance → 训练不稳定甚至 diverge

我直觉上和作者的解释一致：self-distillation 时 teacher 一直在动，loss landscape 是非平稳的，RL-style 训练的不稳定就被放大了。frozen teacher 提供 stationary target，更接近标准 supervised learning 的稳定性。

## 3. 两个应用场景

### 3.1 Experiential Knowledge Distillation

这是 paper 里最 interesting 的部分。流程分三步：

**Stage 1: Knowledge Extraction**
- 给 model 一道题，让它生成 solution trace
- 然后再 prompt 它：conditioning on (problem, trace)，让 model 总结出 "experience item" (格式：`– EXPERIENCE ITEM: ...`)
- 注意：不需要 ground-truth label！这非常关键 — 全程 self-supervised

**Stage 2: Knowledge Accumulation**
- 从不同题目提取出的 experience items 拼接成 context $c$
- 直接 prepend 到新题目前 → 性能提升 (Figure 7 展示 validation accuracy 随积累单调上升)

**Stage 3: Knowledge Consolidation via OPCD**
- 把 $c$ 通过 OPCD 内化进 weights
- Inference 时不需要再 prepend，节省 context length 和 latency

**Datasets**:
- DAPO-Math-17K: ~14K 可验证数学题 (https://arxiv.org/abs/2503.14476)
- Frozen Lake (TextArena): 3×3 grid navigation，避洞到目标
- Sokoban (TextArena): 6×6 grid 推箱子 (https://arxiv.org/abs/2504.11442)

### 3.2 System Prompt Distillation

System prompts 长且每次 inference 都要 prepend，浪费 compute。把 system prompt 内化进 weights 就可以零成本 inference。

- Medical: MedMCQA (https://arxiv.org/abs/2203.14367)
- Safety: Tweet Eval + Hatecheck + Ethos
- 用 MetaSPO (https://arxiv.org/abs/2505.09666) 优化过的 system prompts

## 4. 实验数据深度解析

### 4.1 Table 1: Test-time experiential knowledge (随机选 300 context 中的)

| Model | Task | Method | Accuracy | IF-Eval (OOD) |
|---|---|---|---|---|
| Qwen3-8B | Math | Base | 75.0 | 81.3 |
| | | In-Context | 77.6±1.1 | – |
| | | Context Distill | 78.5±0.5 | 81.2±0.2 |
| | | **OPCD** | **79.7±0.5** | **81.7±0.4** |
| Qwen3-1.7B | Frozen Lake | Base | 6.3 | 67.3 |
| | | In-Context | 20.2±2.2 | – |
| | | Context Distill | 22.9±4.0 | 65.1±0.5 |
| | | **OPCD** | **26.5±6.4** | **67.1±0.5** |

观察：
- OPCD 在 in-distribution accuracy 上比 Context Distillation 高 1-4 个点
- OOD (IF-Eval) 几乎不掉，OPCD 甚至比 Base 还高一点 (81.7 vs 81.3)
- Off-policy Context Distill 在 OOD 上轻微退化 (81.2 vs 81.3)
- Frozen Lake 这种弱模型场景 (Base 只有 6.3) OPCD 提升 ratio 最大 (+3.6 vs +2.7)

### 4.2 Table 2: Filtered experiential knowledge (用 best-scoring context)

| Model | Task | Base | In-Context | Context Distill | OPCD |
|---|---|---|---|---|---|
| Qwen3-8B | Math | 75.0 | 79.0 | 79.5 | **80.9** |
| Qwen3-1.7B | Frozen Lake | 6.3 | 31.4 | 35.2 | **38.3** |
| Qwen3-4B-Ins | Sokoban | 9.4 | 48.4 | 51.6 | **53.9** |

Filtered > Test-time 是合理的，因为 high-quality context 提供了更强 signal。

注意 Sokoban：In-Context 从 9.4 → 48.4，OPCD 进一步到 53.9。这意味着 model 在游戏策略上确实学到了 transferable 的东西。

### 4.3 Table 3 & 4: System prompt distillation

Medical:
| Model | Base | In-Context | Context Distill | OPCD |
|---|---|---|---|---|
| Llama-3.1-8B-Ins | 68.4 | 72.2 | 75.2 | **76.7** |
| Llama-3.2-3B-Ins | 59.4 | 66.4 | 71.0 | **76.3** |
| Qwen2.5-7B-Ins | 46.4 | 52.6 | 58.5 | **62.3** |

Safety:
| Model | Base | In-Context | Context Distill | OPCD |
|---|---|---|---|---|
| Llama-3.1-8B-Ins | 70.7 | 75.3 | 77.2 | **79.6** |
| Llama-3.2-3B-Ins | 30.7 | 69.5 | 83.3 | 83.1 |
| Qwen2.5-7B-Ins | 69.1 | 72.7 | 77.0 | **78.1** |

Qwen2.5-7B 在 medical 上 Base 只有 46.4，这个数据点很奇怪，可能 base model 对 medical QA 能力弱，OPCD 提升到 62.3 (+16 pts)。

### 4.4 Cross-size distillation (Figure 2, Section 4.4)

这个结果我直觉上非常 interesting：

- Teacher: Qwen3-8B frozen
- Student: Qwen3-1.7B / 4B / 8B
- Direct injection ("In-Context" curve): 把 8B 提取的 experience 直接塞进 smaller model 的 context → **性能反而下降**
- OPCD distill: smaller model 稳定提升

为什么直接 injection 会掉？因为 experience 是 8B 自己生成的，可能依赖 8B 的 reasoning pattern、vocabulary、self-consistency。Small model 用同样的 context 时可能 attempt 跟随那种 reasoning 但 capability 不够，反而搞砸了。

OPCD 的 on-policy alignment 解决了这个问题：student 在自己的 trajectory 上学，context 只是作为 teacher 的 hint，student 只内化自己能力范围内能表达的部分。

### 4.5 Forgetting mitigation (Figure 3, Section 4.5)

实验设置：用 safety system prompt distill Qwen2.5-3B-Instruct，teacher 是 frozen Qwen2.5-7B-Instruct。然后 evaluate:
- In-distribution: safety test
- OOD: medical test

结果：
- Off-policy context distillation: safety ↑，medical **↓ ~4 points**
- OPCD: safety ↑ 更多，medical **maintained**

Intuition：off-policy forward KL 强制 student 在 teacher 的所有 distribution region 上都匹配，包括 OOD region (这里是 medical)，导致 capacity 被 reallocate 到 safety，原本 medical 能力被 overwrite。OPCD 是 reverse KL + on-policy：student 只在自己 sample 的 region 上学，所以 OOD region (student 自己不太采样 medical tokens) 几乎不受影响。

这个和 "RL's razor" (Shenfeld et al., https://arxiv.org/abs/2509.04259) 以及 "Retaining by doing" (https://arxiv.org/abs/2510.18874) 的观察一致：on-policy 数据天然 mitigate forgetting，因为 gradient 只在 model 当前访问的 manifold 上 push。

### 4.6 Teacher-Student vs Self-Distillation (Table 5)

| Task | Self | Teacher-Student |
|---|---|---|
| Sokoban | 18.8 | 53.9 |
| Medical | 50.0 | 56.8 |

差距巨大。Sokoban 上 Self 简直没学到东西。

我直觉上的解释：on-policy RL 训练里，non-stationary target 是大问题。如果 teacher 和 student 是同一个 model 同步更新，那么每次 update 后 teacher 分布也在变，student 在追一个 dynamic target。再加上 reverse KL 的 mode-seeking 特性，student 容易 collapse 到某个 mode 后 teacher 已经移动了，loss signal 变得 noisy 甚至 misleading。

Frozen teacher 提供 stationary target，是稳定的 distillation 的关键。这跟 AlphaGo / AlphaZero 的策略网络 distillation 思路一致 — frozen teacher 比自我博弈稳定。

### 4.7 Importance of Experiential Knowledge (Table 6)

| Experience Type | Accuracy |
|---|---|
| w/o Experience | 75.1 |
| Raw Trace (just prepend past solutions) | **70.5** (掉 4.6!) |
| Knowledge (extracted experience) | 77.4 |
| + OPCD | **79.7** |

非常 striking 的数据：直接 prepend raw solution trace 反而让 model 变差。因为 raw trace 充满了 noise (探索、错误、不相关的中间步骤)，model 容易被 distractor 误导。

而 "Knowledge" (从 trace 中提取出的抽象 insight) 才是 transferable 的 — 这呼应了 STaR (https://arxiv.org/abs/2203.14465) 和 self-refine 的思路：model 需要从经验里 abstract 出规律，而不是仅仅 mimic 行为。

这个对比给了一个 deep insight：**context distillation 的有效性 heavily depends on context 的 quality**。Garbage in, garbage out — 即便是 OPCD 也救不了 raw trace 的 noise。

## 5. 我的 intuition 构建

把所有 piece 拼起来：

**On-policy + reverse KL 的组合不是偶然，而是互相 reinforce 的设计**：
- On-policy 保证 student 在自己会访问的 manifold 上学习 → mitigation of forgetting
- Reverse KL 让 student 只往 teacher 的 high-prob mode 靠 → 避免 mode-covering 的 hallucination
- 两者共同作用：student 学到的是 "在我自己的 trajectory 上，teacher 会怎么做"，而不是 "在 teacher 的 trajectory 上，我应该怎么做"

这跟 RL 里 on-policy algorithms (PPO, TRPO) vs off-policy (DQN, SAC) 的 trade-off 是相似的：
- Off-policy sample efficiency 高，但 behavior policy ≠ target policy → distribution shift
- On-policy 必须 sample 自己的数据，但 train/test 一致

**Experiential knowledge 是 RL 的 "experience replay" 在 LLM 上的类比**：
- 传统 RL 把 trajectory 存进 buffer
- 这里把 abstract insight 提取成 context
- 但 abstract insight 比 raw trajectory 更 transferable (Table 6 的 raw trace vs knowledge 对比)

**Cross-size 不 transfer 是 because experience 是 model-specific**：
- 8B 的 reasoning 可能依赖 8B 的 specific internal representation
- 直接给 1.7B 用，1.7B 没法 "act out" 那个 reasoning → 反而干扰
- OPCD 让 1.7B 在自己 trajectory 上学，只取能用的部分 → 干净的 transfer

**Frozen teacher vs moving teacher 的本质**：
- Frozen teacher = supervised learning with consistent labels
- Moving teacher = non-stationary target，类似 self-play 的 instability
- 经典 RL 经验：当 teacher 弱或者 moving 时，self-distillation 容易 diverge

## 6. 与相关工作的脉络

- **Context distillation**: Askell et al. 2021 (alignment via context)，Snell et al. 2022 (off-policy forward KL) — OPCD 的 baseline
- **On-policy distillation**: MiniLLM (Gu et al., https://arxiv.org/abs/2306.08543), Agarwal et al. (GKD, https://arxiv.org/abs/2306.13649), Thinking Machines blog (https://thinkingmachines.ai/blog/on-policy-distillation) — 提供了 reverse KL + on-policy 的 paradigm
- **Self-distillation 系列**: STaR (Zelikman et al.), RL via self-distillation (Hübotter et al.), Privileged information distillation (Penaloza et al.) — paper 区分自己：teacher 可以是 frozen 或不同 model，更灵活
- **Black-box 扩展**: Ye et al. (https://arxiv.org/abs/2503.15701) 把 on-policy distill 扩展到 black-box 设置
- **MetaSPO** (Choi et al.): 优化 system prompt 的来源

## 7. 我看到的一些 limitation / open question

1. **Top-256 truncation**：实现中 reverse KL 只在 student top-256 tokens 上算。这暗含 assumption：teacher 的 high-prob mass 也在 student 的 top region。如果 teacher 和 student distribution 差异大，这个 truncation 可能漏掉 important divergence signal。
2. **训练步数少 (50 steps)**：可能是 hyperparameter tuning 有限。更大规模训练是否还能稳定？
3. **Experiential knowledge 提取依赖 teacher quality**：如果 teacher 提取的 insight 本身就有 bias，OPCD 会 faithful 内化这个 bias (Table 6 raw trace 的对比)
4. **没 ablate context length 影响**：8K vs 16K context 提取出来的 knowledge 质量怎么 trade off？
5. **Self-distillation 不稳定的具体机制**：paper 只说 "high variance"，但 variance 的 source (gradient noise? mode collapse? distribution shift?) 没仔细 disentangle
6. **Exposure bias 实际度量缺失**：paper 主打 fix exposure bias，但没有直接 measure train/inference distribution gap 的 metric，只能从 final accuracy infer

## 8. 实操 takeaways (如果我想 implement)

1. Teacher 必须 froze，不要 attempt self-distillation
2. Reverse KL 用 top-k truncation (256 在 paper 里 work)
3. 50 steps，batch 128，lr 1e-6 ~ 5e-6
4. 每 2 steps checkpoint，选 best test accuracy (训练 50 步可能不稳定)
5. Experiential knowledge 提取 prompt 要明确格式化 (`– EXPERIENCE ITEM:`)
6. 不要直接 prepend raw solution trace — 必须 abstract 成 insight

## References

- Context distillation baseline: https://arxiv.org/abs/2112.00861, https://arxiv.org/abs/2209.15189
- On-policy distillation (GKD/MiniLLM): https://arxiv.org/abs/2306.08543, https://arxiv.org/abs/2306.13649
- Thinking Machines blog on on-policy distillation: https://thinkingmachines.ai/blog/on-policy-distillation
- Black-box on-policy distillation (Ye et al.): https://arxiv.org/abs/2503.15701
- RL's razor (forgetting): https://arxiv.org/abs/2509.04259
- Retaining by doing: https://arxiv.org/abs/2510.18874
- DAPO-Math: https://arxiv.org/abs/2503.14476
- Qwen3: https://arxiv.org/abs/2505.09388
- Qwen2.5: https://arxiv.org/abs/2412.15115
- TextArena: https://arxiv.org/abs/2504.11442
- MetaSPO: https://arxiv.org/abs/2505.09666
- IFEval: https://arxiv.org/abs/2311.07911
- STaR (self-taught reasoner): https://arxiv.org/abs/2203.14465
- Llama 3 herd: https://arxiv.org/abs/2407.21783
