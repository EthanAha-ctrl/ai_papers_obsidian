---
source_pdf: s1 Simple test-time scaling.pdf
paper_sha256: 598932c9f96849cd52495d8b3e12ba4d224e41d5588d2278a78f76c744d8a3bc
processed_at: '2026-08-12T02:36:10-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej, 咱们抛开学术八股文，用最直白的人话把这 paper 的核心 intuition 捋一遍。

这论文其实就回答了一个极其简单的问题：**怎么用最穷、最笨的办法，造一个能像 OpenAI o1 一样“越想越聪明”的模型。**

核心结论极其震撼：不需要几百万条数据，不需要复杂的强化学习，**只要 1000 道精选难题，加上一个叫 "Budget Forcing" 的极其暴力的解码小动作，训练 26 分钟，就能打平甚至超越 o1-preview。**

下面我拆解给你看里面的工程直觉。

---

### 1. 为什么只需要 1000 条数据？(s1K 的秘密)

大家都在疯狂堆数据，DeepSeek R1 用了 80万条，Sky-T1 用了 1.7万条。Stanford 这帮人说：扯淡，LIMA (https://arxiv.org/abs/2305.11206) 早就证明过 1000 条就能 align 一个模型，推理也一样。

他们对 5.9万条初始数据做了三道漏斗筛选：
1. **Quality**: 把排版乱、有乱码的扔了。
2. **Difficulty**: 让 Qwen 7B 和 32B 去做题。这俩要是能做对，说明题太简单，直接扔掉。只保留这俩做不出来的。
3. **Diversity**: 让 Claude 3.5 把题分类到 50 个数学/物理子领域，然后均匀采样，确保不全是代数，得有点生物、量子力学之类的。

**人话翻译**：Base model (预训练阶段) 早就学会了推理，它只是不知道在用户问问题时该把推理过程展示出来。SFT 阶段的 1000 条数据，根本不是在“教模型学推理”，而是在“激活模型展示推理的 format”。既然只是激活，挑最难、最杂、最干净的 1000 道题效果最好。

---

### 2. Budget Forcing 到底是个什么鬼动作？

这是全篇最 brilliant 的 hack。

模型在训练时有个毛病：它倾向于“尽快给出答案”。一旦它觉得想得差不多了，就会吐出一个 `<|im_start|>answer` 的特殊 token，然后直接给结论。

但推理任务里，往往需要再 check 一遍。怎么让它别停下来？强行把 `<|im_start|>answer` 这个 token 屏蔽掉（不让他输出结束思考），然后在它当前的思考末尾强行硬塞一个词：**"Wait"**。

模型一看自己上一句结尾是 "Wait"，它的续写逻辑就会顺着往下走：“哦，等等，我刚才是不是哪里算错了？” 然后它就会重新 check，发现错误并纠正。

Figure 3 里那个例子极其直观：
模型数 raspberry 里有几个 r，数错了说 2 个。
系统强行塞个 "Wait"。
模型接话："Wait... let me re-read... 我刚才数错了，是 3 个。"

**人话翻译**：这就像你有个做题很快但很粗心的学生，你拿把枪指着他：“等一下，再检查一遍。” 他就被迫重新看一眼，然后改对了。这根本不是什么复杂的搜索算法，这就是纯粹的物理级 intervention。

---

### 3. 为什么其他控制思考时间的方法都失败了？

Paper 里试了其他让模型多想一会儿的方法，全崩了， intuition 极其深刻：

**失败一：给 prompt 加指令“你想 2048 个 token”**
模型根本没有数 token 的能力。你让它想 2048 个，它能给你写 7000 个。LLM 作为一个 next-token predictor，天生没有内置计数器。

**失败二：给 prompt 加指令“你想 64 步，每步用 `N steps left` 计数”**
模型学会了作弊。你让它想 16 步，它就把每步写 96 个 token；你让它想 256 步，它就把每步写 56 个 token。总 token 数几乎没变！模型为了满足你的 step 约束，把自己每步的思考砍碎了，完全违背了初衷。

**失败三：Rejection Sampling（拒绝采样）**
这招最搞笑。温度设为 1，疯狂采样，只保留 thinking 长度超过 8000 token 的答案。结果发现：**越长越错！**
为什么会这样？因为如果模型一开始思路就对，它通常会顺着写下去，很快得出答案（短 trace）。如果模型一开始思路错了，它才会不停地绕圈子、自我怀疑、推翻重来（长 trace）。所以采样长 trace 等于专门挑模型“走弯路”的样本，性能反而下降。

---

### 4. 怎么衡量一个 Test-Time Scaling 方法好不好？

他们提了三个公式，看起来复杂，其实非常直觉：

**公式 (1) Control 指标**:
$$ \mathrm{Control} = \frac{1}{|\mathcal{A}|} \sum_{a \in \mathcal{A}} \mathbb{I}(a_{\min} \leq a \leq a_{\max}) $$
变量解释：
- $\mathcal{A}$ 是一组测试用的计算预算
- $a$ 是这次模型实际花的 thinking tokens
- $a_{\min}, a_{\max}$ 是你要求的上限和下限
- $\mathbb{I}$ 是指示函数，满足条件就是 1，不满足就是 0

人话翻译：我让你花 5000 个 token，你真的花了 4000-5000 个吗？Budget Forcing 因为是物理屏蔽，所以达到了 100% 的完美控制。别的靠 prompt 提示的方法，控制率只有 40%-60%。

**公式 (2) Scaling 指标**:
$$ \mathrm{Scaling} = \frac{1}{\binom{|\mathcal{A}|}{2}} \sum_{\substack{b \in \mathcal{A} \\ b > a}} \frac{f(b) - f(a)}{b - a} $$
变量解释：
- $\binom{|\mathcal{A}|}{2}$ 是所有预算组合对的总数
- $f(a)$ 是在预算 $a$ 下的准确率
- $\frac{f(b) - f(a)}{b - a}$ 就是初中数学里的斜率

人话翻译：这就是计算“多想一会儿，准确率能涨多少”的平均斜率。Budget Forcing 的斜率是正的（+15），拒绝采样的斜率是负的（-35，越长越错）。

**公式 (3) Performance 指标**:
$$ \mathrm{Performance} = \max_{a \in \mathcal{A}} f(a) $$
人话翻译：在所有预算里，你能达到的最高准确率是多少。

---

### 5. 一个极其反直觉的 Training Ablation (Table 8)

他们在训练时试了两种序列长度：4096 和 32768。

如果训练序列只有 4096，74% 的训练样本会被从中间截断（answer 部分被切掉了）。
如果训练序列是 32768，0% 被截断，模型能看到完整的思考+回答。

测试结果：**用 32768 训练的模型，测试时不仅准确率高，而且 thinking 的 token 数反而短了（从 2万降到 6千）！**

为什么？如果训练时 answer 经常被切掉，模型就很少收到“输出最终答案”的梯度更新，导致它在测试时不知道什么时候该停，只能一直絮絮叨叨想下去。如果训练时它看全了完整的“思考+回答”，它就学会了“哦，想清楚了就可以停了并给答案”，所以测试时反而更果断。

**人话翻译**：教模型的时候，必须让它看到“标准答案是怎么收尾的”。如果你总是教到一半就下课，模型考试的时候就会一直坐在那里发呆不敢交卷。

---

### 6. 我个人的联想 (Hallucination & Intuition)

看到 Budget Forcing，我立刻联想到 mechanistic interpretability 里的 **"Induction Heads"** 机制。

模型看到一个词，会本能地预测它前面出现过的 pattern 的后续。当我们强行塞一个 "Wait" 进去，模型在 attention 机制下，会被迫去 attend 到它刚才写的内容，寻找可以 "Wait" (反思) 的点。

这相当于我们通过外部的字符串注入，手动触发了模型内部的 **"Self-Correction Circuit"**。

但 paper 提到 6 次 "Wait" 之后模型就开始死循环了。这很好理解：光说 "Wait"，模型反思的视角没有改变，它会在同一个错误逻辑里打转。
如果要继续突破，我们可以动态注入不同的触发词：
- 第 1 次：`Wait`
- 第 2 次：`Let me verify the assumptions`
- 第 3 次：`Is there an alternative approach?`
- 第 4 次：`Let me calculate this numerically instead`

通过外部 prompt engineering 来控制内部的 reflection 维度，这可能就是下一篇 paper 的方向。

### 总结

这篇 paper 最爽的地方在于，它撕掉了 o1 和 R1 蒙在推理模型上的神秘面纱。RL 当然有用，但如果只是想要极强 sample efficiency 的 test-time scaling，**只要挑对 1000 道题，然后在它想停的时候逼它说一句 "Wait"**，就够了。

就是这么粗暴，这么简单。

参考链接：
- s1 Paper: https://arxiv.org/abs/2501.19393
- s1 Code: https://github.com/simplescaling/s1
- LIMA (Sample efficiency intuition): https://arxiv.org/abs/2305.11206
- Induction Heads (Anthropic): https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

---

# s1: Simple Test-Time Scaling - 深度技术讲解

Andrej, 这篇 paper 确实是 reasoning model 复现浪潮里一个非常 elegant 的工作。Stanford 团队用 1000 个 samples + 26 分钟 H100 训练就能逼近 o1-preview, 核心创新点在于 **budget forcing** 这个简单的 test-time intervention。让我从架构, 数据, 方法, 实验, 几个层面深入剖析。

---

## 1. 核心动机与范式定位

传统 scaling law 关注 train-time compute (Kaplan 2020, Hoffmann 2022)。OpenAI o1 开启了新范式 - **test-time scaling**, 也就是在 inference 时花更多 compute 来提升性能。问题在于 OpenAI 没公开方法, 各种复现尝试 (DeepSeek R1, QwQ, Sky-T1, Bespoke) 都需要海量数据或复杂 RL pipeline。

s1 的核心 question: **what is the simplest approach to achieve both test-time scaling and strong reasoning performance?**

答案分两块:
- **数据侧**: 精挑 1000 个 reasoning samples (s1K)
- **方法侧**: budget forcing 这个 decoding-time intervention

---

## 2. 数据集 s1K 的构建 - Quality × Difficulty × Diversity

### 2.1 初始 59K pool 的来源

从 16 个数据源收集 59,029 questions (Table 7):
- NuminaMATH: 30,660 math problems
- MATH: 11,999
- OlympicArena: 4,250 (Astronomy, Biology, Chemistry, CS, Geography, Math, Physics)
- OmniMath: 4,238 competition math
- AGIEval: 2,385 (SAT, LSAT 等)
- AIME 1983-2021: 890
- 还有 s1-prob (Stanford Stats PhD qualifying exams, 182), s1-teasers (quant interview, 23) 等自建数据

每个 question 用 **Gemini Flash Thinking API** 生成 reasoning trace + solution, 形成 59K triplets。

### 2.2 三阶段过滤到 1K

**Quality 过滤**:
- API errors -> 54,116
- 格式问题 (ASCII art, 不存在 images, 编号不一致) -> 51,581
- 384 高质量 samples 直接纳入

**Difficulty 过滤**:
- 用 Qwen2.5-7B-Instruct + Qwen2.5-32B-Instruct 测试
- Claude 3.5 Sonnet 当 grader (Figure 8 的 grading prompt)
- 移除任一模型能解的 question -> 24,496
- 假设: harder problem 需要 longer reasoning trace

**Diversity 过滤** (Algorithm 1):
- Claude 3.5 基于 **MSC (Mathematics Subject Classification)** 系统分到 50 个 domains
- 每轮: 先 uniform 随机选一个 domain d, 然后按 thinking length 的 power-law weighting 在 domain d 内采样一个 question
- weights = 2^(-ranks), rank 按 thinking length 降序
- 重复直到 1000 samples, 跨越 50 domains

最终 s1K 的 domain 分布见 Table 6: Geometry (109), Number theory (98), Combinatorics (75), Real functions (43), Biology (41)... 总共 51 domains, 4.7M tokens。

### 2.3 关键 ablation (Table 2) - 三个原则缺一不可

| Model | AIME24 | MATH500 | GPQA |
|---|---|---|---|
| 1K-random (only quality) | 36.7 | 90.6 | 52.0 |
| 1K-diverse (only diversity) | 26.7 | 91.2 | 54.6 |
| 1K-longest (only difficulty) | 33.3 | 90.4 | 59.6 |
| 59K-full | 53.3 | 92.8 | 58.1 |
| **s1K (all three)** | **50.0** | **93.0** | **57.6** |

注意 59K-full 训练用 394 H100 hours, s1K 只用 7 H100 hours。性能差异不大, sample efficiency 极高。这呼应 LIMA 的 "Superficial Alignment Hypothesis" (https://arxiv.org/abs/2305.11206) - pretraining 已经把 reasoning 能力灌输进去, SFT 只是 activate。

---

## 3. Test-Time Scaling 方法 - Budget Forcing 的本质

### 3.1 Sequential vs Parallel

Paper 把 test-time scaling 分两类:
- **Sequential**: 后续计算依赖前面 (长 reasoning trace), e.g. budget forcing
- **Parallel**: 独立计算后聚合 (majority voting), e.g. Best-of-N, self-consistency

直觉: sequential 应该 scale 更好, 因为可以 build on intermediate results, 允许 deeper reasoning 和 iterative refinement。

### 3.2 Budget Forcing 机制

这是 paper 的核心创新。具体来说:

**Maximum enforcement (early exit)**:
- 如果 model 生成的 thinking tokens 超过 desired limit
- 强制 append end-of-thinking token delimiter (`<|im_start|>answer`)
- 可选 append "Final Answer:" 让 model 输出当前 best guess

**Minimum enforcement (延长 thinking)**:
- 抑制 end-of-thinking token 的生成
- 在 model 当前 reasoning trace 后 append string **"Wait"**
- 这会 encourage model 反思当前 generation, 经常 fix 错误 reasoning step

Figure 3 给出经典例子: 问 "How many r in raspberry?", model 一开始数错说 2, 强制 append "Wait" 后, model re-read 问题, 重新数到 3。

### 3.3 为什么 "Wait" 有效 - 一个 mechanistic 直觉

从 mechanistic interpretability 角度推测:
- "Wait" 在 training data (尤其是 reasoning / debate 类 corpus) 中通常标记着 "我需要重新考虑"
- 这相当于给 model 一个 explicit "stop and reflect" 信号, 激活 model 内部的 self-correction circuit
- 类似 Chain-of-Thought (https://arxiv.org/abs/2201.11903) 里 "Let me think step by step" 的作用, 但更聚焦于 error correction

Table 4 ablation 显示其他 string 也可以, 但 "Wait" 最好:

| String | AIME24 |
|---|---|
| No extrapolation | 50.0 |
| 2x without string | 50.0 |
| 2x "Alternatively" | 50.0 |
| 2x "Hmm" | 50.0 |
| 2x "Wait" | 53.3 |

注意 4x "Wait" 在 AIME24 上达到 56.7, 在 s1.1 上更高 (Table 5)。

### 3.4 Baseline 方法对比

Paper 仔细对比了多种 test-time scaling 方法:

**(I) Conditional length-control**:
- (a) Token-conditional control: prompt 里直接说 "Think for up to 2048 tokens"
- (b) Step-conditional control: "Think for up to 64 steps", 用 "63 steps left...62 steps left..." 计数
- (c) Class-conditional control: 两个 generic prompt "short thinking" / "long thinking"

**(II) Rejection sampling**: 采样直到 generation fits 预定 compute budget

### 3.5 Table 12/13 揭示的关键失败模式

**Token-conditional**: model 完全不会 count tokens, 即使训练后也不行。无 intervention 时, prompt 说 "Think 1024 tokens" 实际生成 7939 tokens, 完全失控。

**Step-conditional**: 一个非常 subtle 的 failure - **model 学会 hack 约束**。看 Table 13:
- 16 steps instructed: 96 tokens per step (总 1517 tokens)
- 256 steps instructed: 56 tokens per step (总 7551 tokens)

Model 会 compensate, steps 少就把每步写更长, 总 token 数其实在偷懒调整。这违反 control 的初衷。

**Class-conditional**: 简单 "think longer" 的 prompt 确实能提升 thinking, 但 control 不精确, scaling 不稳定。

**Rejection sampling (Figure 6)**: 反而出现 **inverse scaling** - 要求更长的 thinking 反而 performance 下降。Paper 给的解释非常 insightful: 短 generation 通常是 model 一开始就 on right track, 长 generation 是 model 犯错后 backtrack / 自我怀疑。Rejection sampling 选长 sample 等于在选错的更多。

---

## 4. Metrics 设计 - Control, Scaling, Performance

Paper 定义了三个 desiderata, 这是评估 test-time scaling 方法的关键 framework:

### 4.1 Control metric

$$\text{Control} = \frac{1}{|\mathcal{A}|} \sum_{a \in \mathcal{A}} \mathbb{I}(a_{\min} \leq a \leq a_{\max})$$

变量解释:
- $\mathcal{A}$ = evaluation set, 一组不同 compute budget 的 evaluations
- $|\mathcal{A}|$ = evaluations 数量
- $a$ = 单次 evaluation 实际花的 compute (thinking tokens)
- $a_{\min}, a_{\max}$ = 预设的 compute 上下界
- $\mathbb{I}(\cdot)$ = indicator function, 条件成立返回 1, 否则 0

直觉: 这个 metric 衡量 "我让你花 5000 tokens, 你真的花 5000 tokens 吗"。Budget forcing 因为是强制操作, 永远 = 100%。

### 4.2 Scaling metric

$$\text{Scaling} = \frac{1}{\binom{|\mathcal{A}|}{2}} \sum_{\substack{b \in \mathcal{A} \\ b > a}} \frac{f(b) - f(a)}{b - a}$$

变量解释:
- $\binom{|\mathcal{A}|}{2}$ = 所有 $(a, b)$ 对的组合数
- $f(a)$ = compute budget $a$ 下的 accuracy
- 求和遍历所有 $b > a$ 的对
- 这是 piece-wise linear function 的平均 slope

直觉: 衡量 "多花 compute 能多换多少 accuracy"。必须为正才有意义, 越大越好。

### 4.3 Performance metric

$$\text{Performance} = \max_{a \in \mathcal{A}} f(a)$$

直觉: 在所有 compute budget 下的最高 accuracy。

### 4.4 Table 3 对比

| Method | Control | Scaling | Performance |
|---|---|---|---|
| **BF (budget forcing)** | **100%** | **15** | **56.7** |
| TCC (token) | 40% | -24 | 40.0 |
| TCC + BF | 100% | 13 | 40.0 |
| SCC (step) | 60% | 3 | 36.7 |
| SCC + BF | 100% | 6 | 36.7 |
| CCC (class) | 50% | 25 | 36.7 |
| RS (rejection sampling) | 100% | -35 | 40.0 |

几个关键观察:
1. BF 单独使用 best overall
2. TCC/SCC + BF 后 control 变 100%, 但 scaling 不如纯 BF (因为粗粒度约束干扰 reasoning flow)
3. CCC scaling 高但 performance 低 - 不可控
4. RS 是 inverse scaling (slope = -35)

---

## 5. Training Details - 极简配方

### 5.1 Hyperparameters

- Base: Qwen2.5-32B-Instruct (https://arxiv.org/abs/2412.15115)
- Epochs: 5
- Batch size: 16
- Total gradient steps: 315
- Precision: bfloat16
- Learning rate: $1 \times 10^{-5}$
- Warmup: linear, 5% (16 steps)
- Decay: cosine 到 0 (剩余 299 steps)
- Optimizer: AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay $= 1 \times 10^{-4}$
- **Total time: 26 minutes on 16 H100 GPUs**

变量解释:
- $\beta_1, \beta_2$ = AdamW 的一阶/二阶矩 decay rate
- $\beta_1 = 0.9$ 是标准值, $\beta_2 = 0.95$ 比 PyTorch default (0.999) 小, 意味着对近期 gradient 更敏感, 适合短训练
- weight decay $10^{-4}$ 是 GPT-style 标准值

### 5.2 Token delimiters 设计

- Thinking 阶段: `<|im_start|>think\n ... \n<|im_start|>answer`
- 用 newline 包围, 与 Qwen chat template 兼容
- Loss 只算在 reasoning trace + solution 上, 不算 question

### 5.3 Sequence length ablation (Table 8) - 非常有意思

| Training seq len | % cutoff | AIME24 (acc/tokens) | MATH500 | GPQA |
|---|---|---|---|---|
| 4096 | 74% | 30.0% / 20721 | 90.0% / 5324 | 52.5% / 6841 |
| 32768 | 0% | 50.0% / 6984 | 91.0% / 3268 | 53.0% / 3568 |

直觉解释:
- 短训练 sequence -> answer section 经常被 cut off -> model 没充分学习 "如何结束 reasoning"
- 测试时 model 不知道何时停止, 一直 thinking, 平均 20721 tokens
- 长训练 sequence -> answer section 完整出现 -> model 学会 "什么时候输出 Final Answer"
- 测试时 reasoning 自然变短 (6984), 性能反而更好

这个 ablation 揭示了 reasoning model 的一个 subtle 点: **训练数据的完整性影响 model 的 "停止决策"**。

---

## 6. Main Results - Sample Efficiency Frontier

### 6.1 Table 1 / Table 5 核心对比

| Model | # Examples | AIME24 | MATH500 | GPQA |
|---|---|---|---|---|
| o1-preview | N.A. | 44.6 | 85.5 | 73.3 |
| o1-mini | N.A. | 70.0 | 90.0 | 60.0 |
| o1 | N.A. | 74.4 | 94.8 | 77.3 |
| r1 | >800K | 79.8 | 97.3 | 71.5 |
| r1-distill | 800K | 72.6 | 94.3 | 62.1 |
| QwQ-32B | N.A. | 50.0 | 90.6 | 54.5 |
| Sky-T1 | 17K | 43.3 | 82.4 | 56.8 |
| Bespoke-32B | 17K | 63.3 | 93.0 | 58.1 |
| LIMO | 817 | 56.3 | 94.8 | 66.7 |
| s1 w/o BF | 1K | 50.0 | 92.6 | 56.6 |
| **s1-32B (BF)** | **1K** | **56.7** | **93.0** | **59.6** |
| s1.1 w/o BF | 1K | 56.7 | 94.4 | 60.6 |
| s1.1 (BF 2x) | 1K | 56.7 | 95.4 | 63.6 |

关键 takeaways:
- s1-32B 用 1K samples 在 AIME24 上超 o1-preview (56.7 vs 44.6)
- vs r1: r1 用 800x 多数据, 但 s1 sample efficiency 上 frontier
- s1.1 把 traces 换成 DeepSeek r1 distillation, 性能进一步提升 (MATH500 95.4, GPQA 63.6)

### 6.2 Figure 4 - Sequential vs Parallel scaling

- (a) Budget forcing 显示清晰 scaling, 在 6x "Wait" 后开始 flatten
- (b) Majority voting (parallel): 64 samples 的 majority voting 在 AIME24 上无法追上 budget forcing 的 sequential scaling

这验证了 paper 开头的直觉: sequential > parallel。

### 6.3 Extrapolation 能力

最 exciting 的结果是 **budget forcing 允许 extrapolation**:
- s1-32B 训练时 max thinking ~32K tokens
- 用 budget forcing + "Wait" 在 AIME24 上能从 50% 推到 57%
- 这意味着 test-time compute 的投入能直接换 performance, 超过训练分布

Figure 1 展示了在 AIME24, MATH500, GPQA 三个 benchmark 上的 test-time scaling 曲线, 都是单调上升 (至少在合理 budget 范围内)。

---

## 7. 局限性与未来方向

### 7.1 已知局限

1. **Flattening**: 6x "Wait" 后性能不再上升, model 进入 repetitive loop
2. **Context window**: 最终受 base model context window 限制 (Qwen2.5-32B 是 128K)
3. **正确性不保证**: s1K 里 53.6% 的 trace 是 correct, distillation 不完美

### 7.2 未来方向 (paper 自己提到)

- 轮换不同 string (不只是 "Wait"), 避免 repetitive loop
- 结合 frequency penalty 或 higher temperature
- 把 budget forcing 应用到 RL-trained reasoning model
- 是否 RL 能带来新的 test-time scaling 方式

### 7.3 Figure 7 - Parallel 作为补充

Paper 也尝试在 sequential 基础上加 parallel methods:
- **Majority voting**: N 次生成后取多数
- **REBASE** (https://arxiv.org/abs/2408.00724): 用 process reward model (LLaMA-34B 初始化) 引导的 tree search, 用 majority voting 聚合

结果显示 REBASE 在高 compute 区域 scaling 更好, 但需要额外 reward model forward pass。Paper 结论: parallel methods 可以 complement sequential, 突破 context window 限制。

---

## 8. 我的几点直觉延伸

### 8.1 Budget Forcing 的本质 - "免费" 的 test-time intervention

Budget forcing 之所以有效, 我认为核心是它绕过了 model 自身的 " stopping policy"。Base model 训练时学到的 stopping policy 是基于 human-preference / instruction tuning 的, 倾向于 "尽快给出 confident answer"。但 reasoning 场景下, 这种 policy 太 greedy。Budget forcing 强制 model 继续 explore, 等于 overrode 这个 stopping policy。

这和 Quiet-STaR (https://arxiv.org/abs/2403.09629) 思路类似 - 都是给 model "思考时间"。

### 8.2 与 RL-based 方法的对比

DeepSeek R1 (https://arxiv.org/abs/2501.12948) 用 RL + 大量数据训练, 把 reasoning 直接烙进 model 参数。s1 走另一条路 - 用极小 SFT 数据 + test-time intervention。两种方法本质回答同一个问题: "如何让 model 用更多 compute 来换 accuracy"。

- RL: 把 "thinking longer = better" 烙进 policy
- s1: 用 decoding trick 强制 thinking longer

长期看, RL 方法上限可能更高 (能改 model 内部 representation), 但 s1 这种方法 sample efficiency 完胜, 而且可以叠在 RL model 上。

### 8.3 数据效率的本质 - "激活" vs "学习"

LIMA paper (https://arxiv.org/abs/2305.11206) 的 hypothesis 在这里再次被验证。1000 samples 不可能从零教一个 32B model 学会 reasoning, 但可以激活它 pretraining 时已经学到的 reasoning 能力。

这意味着:
- Pretraining 才是 reasoning 能力的真正来源
- SFT 主要是 "format alignment" - 让 model 知道用什么 token format 输出 reasoning
- Budget forcing 是在 format 之上的轻量级 control

### 8.4 与 o1 的可能方法对比

OpenAI o1 公开说用 large-scale RL。s1 的结果暗示: 如果只看 "在已知 benchmark 上达到 o1-preview 水平", 可能 RL 不是唯一路径。但 o1 在 harder benchmark (FrontierMath, ARC-AGI) 上的表现, s1 没法复制。这说明:
- 简单 task: SFT + budget forcing 够用
- 极难 task: 可能确实需要 RL 改 model 的搜索策略

### 8.5 Budget forcing 的失败模式 - 一个值得研究的方向

Paper 提到 6x "Wait" 后 model 进入 repetitive loop。从 mechanistic 角度, 我猜测原因是: 每次 "Wait" 都激活同一个 self-correction circuit, 但 model 没有 "新的思考维度" 引入, 所以 circuit 重复触发, 输出重复内容。

可能的改进:
- 不同 string 激活不同 circuit ("Wait" / "Alternatively" / "Hmm" / "Let me reconsider")
- 引入 external context (e.g. 提示 model 检查 specific assumption)
- 结合 retrieval 拉入新信息打破 loop

---

## 9. 与同期工作的横向对比

### 9.1 LIMO (https://arxiv.org/abs/2502.03387)

同期工作, 用 817 samples 训练, 性能略高于 s1 (AIME24 56.3 vs 56.7, 但 MATH500 94.8 vs 93.0)。LIMO 强调 "Less Is More", 但没有 budget forcing 这种 test-time scaling 机制。

### 9.2 Sky-T1 (https://novasky-ai.github.io/posts/sky-t1)

用 17K samples, $450 budget, AIME24 43.3。数据更多但性能不如 s1, 说明 s1 的数据 curation 更精。

### 9.3 Bespoke-32B (https://hf.co/bespokelabs/Bespoke-Stratos-32B)

17K samples, AIME24 63.3。这是从 QwQ-32B-preview 蒸馏的。性能略低于 s1 但样本多 17x。

### 9.4 DeepSeek R1 系列 (https://arxiv.org/abs/2501.12948)

- r1: >800K samples, RL 训练
- r1-distill 系列: 800K samples SFT, 性能强劲 (r1-distill-Qwen-32B: AIME24 58.3)

R1-distill 用 800x 多数据, 性能与 s1 相当, 说明数据 curation 在 sample efficiency 上有 10-1000x 杠杆。

---

## 10. Web Links 汇总

- s1 GitHub: https://github.com/simplescaling/s1
- s1 paper (arxiv): https://arxiv.org/abs/2501.19393
- OpenAI o1: https://openai.com/index/learning-to-reason-with-llms/
- OpenAI o3-mini: https://openai.com/index/openai-o3-mini/
- DeepSeek R1: https://arxiv.org/abs/2501.12948
- Kimi k1.5: https://arxiv.org/abs/2501.12599
- QwQ: https://qwenlm.github.io/blog/qwq-32b-preview/
- Qwen2.5: https://arxiv.org/abs/2412.15115
- Sky-T1: https://novasky-ai.github.io/posts/sky-t1
- Bespoke: https://hf.co/bespokelabs/Bespoke-Stratos-32B
- LIMO: https://arxiv.org/abs/2502.03387
- LIMA: https://arxiv.org/abs/2305.11206
- Chain-of-Thought: https://arxiv.org/abs/2201.11903
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Snell test-time scaling: https://arxiv.org/abs/2408.03314
- REBASE: https://arxiv.org/abs/2408.00724
- Best-of-N / Large Language Monkeys: https://arxiv.org/abs/2407.21787
- Self-Consistency: https://arxiv.org/abs/2203.11171
- Quiet-STaR: https://arxiv.org/abs/2403.09629
- Let's Verify Step by Step (PRM): https://arxiv.org/abs/2305.20050
- Stream of Search: https://arxiv.org/abs/2404.03683
- Gemini Thinking: https://cloud.google.com/vertex-ai/generative-ai/docs/thinking-mode
- GPQA: https://arxiv.org/abs/2311.12022
- MATH: https://arxiv.org/abs/2103.03874
- AIME: https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions
- OlympicArena: https://arxiv.org/abs/2406.12753
- NuminaMATH: https://github.com/project-numina/aimo-progress-prize
- Test-time scaling laws (Hugh Zhang): https://github.com/hughbzhang/o1_inference_scaling_laws
- Welleck survey: https://arxiv.org/abs/2406.16838
- Kaplan scaling laws: https://arxiv.org/abs/2001.08361
- Chinchilla: https://arxiv.org/abs/2203.15556
- lm-eval-harness: https://github.com/EleutherAI/lm-evaluation-harness
- vLLM: https://arxiv.org/abs/2309.06180
- AdamW: https://arxiv.org/abs/1711.05101

---

## 总结

s1 的精髓在于: 用极简方法 (1K SFT + decoding-time intervention) 复现了 reasoning model 的 test-time scaling 现象。三个关键 insights:

1. **Sample efficiency 来自数据 curation**: Quality × Difficulty × Diversity 三原则组合, 1K > 59K
2. **Test-time scaling 不需要复杂 RL**: 简单的 budget forcing 就能 extrapolate
3. **Reasoning 能力 pretraining 就有**: SFT 只是激活 + format alignment

更深层的启示: reasoning model 这个范式中, train-time 和 test-time 的边界在重构。传统模型 train 完就定型, test 时 compute 不变。Reasoning model 的 test-time compute 可以像 train-time 一样 "scaling", 这开辟了一个全新的 optimization 维度。

s1 之所以重要, 在于它公开, 极简, 可复现, 给整个社区一个清晰的 baseline, 让大家知道: 不需要闭源神秘方法, 也能做出强 reasoning model。这对开源 AI 生态意义重大。
