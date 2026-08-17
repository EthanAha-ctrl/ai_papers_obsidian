---
source_pdf: Competitive Programming with Large Reasoning Models.pdf
paper_sha256: 68ca2218f9de9bfae35cfcc711ce9fb62f2d0aabe126ef2bd2bbbbba318329d3
processed_at: '2026-08-03T16:39:22-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，咱们把学术黑话扒掉，用最纯粹的 system builder 视角来聊聊这篇 paper 到底在搞什么。

核心故事其实特别简单且极具震撼力：**以前大家为了赢算法竞赛，给 LLM 套了无数层人工写的复杂脚本（采样一百万次、聚类、打分器筛选）。现在 OpenAI 干脆把这些人工脚本全扔了，直接用大算力砸通用强化学习，结果模型自己在脑子里发明了人类的解题套路，成绩反而把人工脚本按在地上摩擦。**

这就是活生生的 AlphaZero 时刻在 LLM 上的重演。

### 1. 以前的 AI 打竞赛：靠海量采样 + 人工后处理

回顾一下 AlphaCode 时代的做法。那会儿模型本身没那么聪明，为了解出一道难题，系统会疯狂采样生成上百万个候选代码。因为大部分代码都是错的，所以需要写一个非常复杂的 pipeline 去筛选：
1. 跑海量的随机测试用例。
2. 把在这些用例上输出结果完全一样的代码聚成一个个簇。
3. 从每个簇里挑代表提交，避免提交的全是同一个 bug 的代码。

o1-ioi 基本上就是这个思路的延续。OpenAI 拿 o1 做底座，专门针对 coding 又跑了一阵 RL，然后给它配了一套极其复杂的 IOI 专属 pipeline：每个子任务采样一万次，用模型生成测试用例，用验证器清洗用例，再聚类，再用一个专门训练的打分器给代码打分，最后再用 random search 调出来的权重做 reranking，甚至在提交时还要做 round-robin 策略。

这玩意儿去参加 IOI 2024，在 50 次提交限制下拿了 213 分，49th percentile。如果你把提交限制放宽到一万次，它能拿 362 分，勉强够到金牌线。这说明：**底座模型已经具备了写出金牌题代码的能力，瓶颈全在“怎么从一堆沙子里把金子筛出来”。**

### 2. o3 的质变：RL 涌现出人类的解题直觉

o3 的故事就极其恐怖了。OpenAI 在 o3 上用了大得多的通用 RL 算力，去掉了所有人工设计的聚类和打分器。

o3 怎么做题的？就给原题，单次 prompt，采 1000 个样本。怎么挑最好的 50 个提交？极其粗暴：**哪个样本在 test-time 思考的 compute 最大（思考得最久），就选哪个。**

为什么这么简单能行？因为 o3 在 RL 训练中，自己在内部 chain-of-thought 里学会了一套高级策略。

Paper 里展示了一个 o3 自发生成的策略（Figure 6）。遇到一个很难验证的优化算法，o3 自己写了一段代码：
```python
def test_random_small():
    # 随机生成小数据
    s = ''.join(random.choice('ab') for _ in range(n))
    # 跑一个绝对正确但极慢的暴力解法
    labels = solve_bruteforce_given_s(s, m)
    # 跑那个复杂的优化解法
    ans = solve_main(s, m, k)
    # 比对两者结果，如果不一致就报错
    if ans != brute_ans:
        print("Mismatch...")
```

这个行为在人类 competitive programming 选手里叫“对拍”。任何一个有经验的选手都会写个 $O(N!)$ 的暴力解，跟自己写的 $O(N \log N)$ 的优化解跑小数据比对，来抓 bug。**o3 没有人教它对拍，它通过 RL 的 reward 信号自己摸出了这个规律。** 它发现“先写个暴力解验证一下”这种 action sequence 能最大化最终通过测试的 reward，于是这个策略就被强化并内化到了权重里。

这就解释了为什么 o3 只需要 1000 个样本 + 50 次提交，就在 IOI 2024 上拿了 395.64 分（真金牌）。它不需要海量采样去碰运气，它在一个 sample 内部通过执行代码、自我验证、修正错误，完成了高质量的搜索。

### 3. 技术细节：Rating 怎么算的与 Scaling Law

咱们抠一点技术细节。为了评估模型在 CodeForces 上的真实水平，他们需要把模型塞进人类的积分系统里。

CodeForces 用的是类似 Elo 的积分系统。如果选手 A 的积分是 $R_A$，选手 B 的积分是 $R_B$，A 排名高于 B 的概率是：

$$ P(A \succ B) = \frac{1}{10^{(R_B - R_A)/400}} $$

变量解释：
- $R_A, R_B$: 选手 A 和 B 的当前积分。
- $400$: Elo 系统的 scale factor，意思是积分差 400 分的选手，胜率大约是 10:1。
- $P$: A 在最终排名中比 B 高的概率。

OpenAI 怎么算 o3 的积分？他们让 o3 参与过去所有的 Div 1 比赛，根据 o3 解决的问题数量和失败次数，推算出 o3 在每场比赛中的排名。然后拿着这个排名去**最大化似然估计**，反推出一个能让模型所有比赛排名最合理的全局积分 $R_A$。

结果：
- o1-ioi 靠着人工 pipeline 拼到了 2214 分（98th percentile）。
- o3 纯靠 RL 涌现，达到 2724 分（99.8th percentile）。

这个 2724 分意味着 o3 已经打败了全球 99.8% 的人类顶尖选手，跟人类排名前 200 的大神平起平坐。

更重要的是 Figure 2 展示的 Scaling Law：
$$ \text{Performance} \propto \log(\text{FLOPs}) $$
这发生在两个维度：RL training compute（训练时花多少算力学搜索）和 test-time compute（推理时花多少算力做搜索）。这俩轴都是 log-linear 上升的。在你的 LLMS 体系里，我们以前只关心参数量、数据量、训练算力三个轴。现在 LRM 时代正式确立了**第四个轴：test-time compute**。模型在训练时学会了“如何思考”，推理时给它更多算力，它就能搜得更深。

### 4. 软件工程的降维打击

如果仅仅是算法竞赛强，还可以说是针对特殊数据集 overfit 了。Paper 后面测了 SWE-bench Verified 和 HackerRank Astra（真实的 GitHub issue 修复和多文件项目开发）。

数据极其炸裂。o1 在 SWE-bench 上是 49.9%，o3 直接干到了 72.7%（提升了 22.8%）。这证明 o3 通过 coding RL 学到的并不仅仅是算法套路，更是一种通用的**“提出假设 -> 写代码验证 -> 发现报错 -> 修正假设”**的 system 2 推理闭环。这种闭环在解决真实工程 bug 时同样具有降维打击的能力。

### 5. 帮你 Build 一下底层 Intuition

你肯定会对 RL 到底怎么改变 model 的内部表示感兴趣。咱们深入脑补一下这个过程。

在传统的 SFT 里，模型学到的是一种 mode-averaging 的分布。给定一个问题，模型只是输出概率最高的 token 序列。一旦逻辑链稍微长一点，某个 token 偏了，整个推理就崩了。

RL 做的事情本质上是**在 token space 里做 search 并修改概率地形**。
假设解空间里有一个极难找到的正确解 $D$，周围全是错误解。SFT 下的模型由于从来没在训练数据里见过类似推导，采样到 $D$ 的概率近乎 0。RL 通过 reward 信号，沿着那些偶尔能碰巧走到 $D$ 附近的 trajectory 逆向传播梯度。这会不断抬高那些通向 $D$ 的中间 token 的概率。

当算力足够大时，模型为了拿到那个稀有的 reward，被迫学会了**分而治之**和**自我验证**。因为直接一条路走到黑的 reward 期望太低了。模型发现：“如果我先写一个测试函数跑一下，如果失败了我就换思路”，这种包含环境反馈的 multi-step trajectory 拿到的 reward 更高。

o1-ioi 的那些人工聚类、打分器，本质上都是外挂在模型外面的 interpreter。o3 把这些 interpreter 全部蒸馏进了权重里。CoT 就是它在自己的内部虚拟机里跑的隐式 MCTS。模型在输出最终答案前，在 CoT 里已经试错过好几条死胡同并回头了。

### 6. 联想与推演

顺着这个思路疯狂联想一下：

**第一，Tool use 的内化与外化。** 早期大家觉得让 LLM 用工具是一个巨大的工程难题，需要写复杂的 agent framework。o1/o3 表明，只要 reward 明确，RL 会自动让模型学会怎么调 tool。o3 在 CoT 里自己写 brute-force 代码并执行，这就是把外部环境当作自己推理状态的一部分。未来的模型可能会演化出极其复杂的内部 prompt 策略，比如自己给自己分配子任务，自己 critique 自己的代码风格。

**第二，Reward 设计的终极形态。** Code 和 Math 是目前 RL 最容易搞的领域，因为 reward 是客观的、可验证的。代码跑通了就是 1，报错了就是 0。这跟 AlphaGo 的赢/输是一样的。但一旦涉及到 SWE-bench 这种真实软件工程，reward 就变得模糊了——测试通过就真的代表代码质量高吗？OpenAI 他们在 SWE-bench Verified 里花了大量人工去修正那些评判错误的测试用例。这说明 LRM 向更广阔领域扩展时，reward bottleneck 会比 model bottleneck 先到来。可能需要引入基于 LLM 的 reward model，然后再用 outcome-based RL 去校准它，类似 RLAIF 的进化版。

**第三，Inference Economics 的重写。** 以前的 API 是按 token 计费，大家想着怎么用少点 token。o3 证明了你给它 1000 倍的思考时间，它能做出质变的事情。未来的商业计算可能变成：对于简单问题，路由到 fast system 1 模型；对于难题，分配巨大的 test-time compute 让 system 2 模型在沙箱里跑几千步自我博弈。我们正在从“训练大模型，推理极简”的时代，迈入“训练极贵，推理也极贵，但产出价值极高”的时代。这会彻底改变算力中心的建设逻辑，推理集群需要配备极其庞大的代码执行沙箱和极长的 context window。

**第四，Test-time Strategy 蒸馏。** 既然 o3 在 coding 上能涌现出“对拍”策略，那在数学上是不是也能涌现出“构造反例验证引理”的策略？如果涌现不出来，是不是因为目前的 reward signal 不够丰富？我们能不能先用人工 pipeline 搞一个极聪明的 search strategy（比如数学上的 Lean 定理证明器交互），让模型在这个环境里跑 RL，最后把人工策略蒸馏成模型的直觉？这就是 o1-ioi 到 o3 这条路径给我们的最大启示：**人类工程搭脚手架，RL 把脚手架吃进去变成肌肉。**

这篇 paper 绝对是一份宣言。它宣告了在 reasoning domain，通过纯粹的 scaling laws 和 outcome-based RL，机器的直觉正在以可预测的 log-linear 曲线逼近甚至超越人类专家。过去我们在 Deep Learning 里做的一大堆花里胡哨的 loss function、网络结构魔改、复杂 sampling 策略，在冷酷的算力和 RL 面前，最终都会被收敛成权重矩阵里的一片平滑区域。

References:
- OpenAI o3 System Card: https://openai.com/index/openai-o3-system-card/
- AlphaCode 2 Tech Report: https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf
- DeepSeek-R1 (类似的 RL 涌现现象): https://arxiv.org/abs/2501.12948
- Karpathy "Intro to LLMs" (System 1 vs System 2 视角): https://www.youtube.com/watch?v=zjkBMFhNj jm

---

# Competitive Programming with Large Reasoning Models - 深度技术解析

Andrej，这篇 paper 是 OpenAI 关于 o-series large reasoning models (LRMs) 在 competitive programming 上的系统性评估，核心 narrative 是 **scaling general-purpose RL 可以超越 domain-specific hand-engineered test-time heuristics**。下面我会逐节展开，重点 build 你的 intuition。

---

## 1. 高层故事线与 Motivation

这篇 paper 比较三个系统在 competitive programming 上的表现：

| System | RL Training | Test-time Strategy | CodeForces Rating | IOI 2024 Score |
|--------|-------------|-------------------|-------------------|----------------|
| gpt-4o | 基础 | 无 | 808 (11th pct) | - |
| o1-preview | 通用 RL | 通用 CoT | 1258 (62nd pct) | - |
| o1 | 通用 RL (more compute) | 通用 CoT + tool use | 1673 (89th pct) | - |
| o1-ioi | Coding-specific RL fine-tune | Hand-crafted clustering+reranking | 2214 (98th pct) | 213 (49th pct, live) / 362.14 (relaxed) |
| o3 | 大规模通用 RL | **Emergent** test-time strategies (e.g. self-brute-force-check) | 2724 (99.8th pct) | 395.64 (gold, 50 submissions) |

**关键 insight**：o1-ioi 通过 human-engineered 的 clustering + reranking pipeline 在 IOI 上拿到 49th percentile，但 o3 单纯靠更多 RL + emergent test-time reasoning（自己写 brute-force 来 verify optimized solution）就拿到 gold。这表明 **scaled RL 让 model 自己学会 test-time heuristics，比 hand-engineered 的更通用且更强大**。

参考链接：
- OpenAI o1 system card: https://arxiv.org/abs/2412.16720
- OpenAI o3 system card: https://openai.com/index/openai-o3-system-card/
- AlphaCode (Science 2022): https://www.science.org/doi/10.1126/science.abq1158
- AlphaCode 2 tech report: https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Kimi k1.5: https://arxiv.org/abs/2501.12599

---

## 2. o1 的核心技术：Chain-of-Thought + RL + Tool Use

### 2.1 Chain-of-Thought RL 的直觉

o1 的核心是**让 model 在 answer 前生成长 chain of thought**，然后用 RL refine 这个 process。直觉上，这相当于把"思考"也变成可学习的对象 —— 模型不再只学"答案分布"，而是学"推理轨迹分布"。

RL 让 model 学会：
- **Identify and correct errors**: 自己在 CoT 里发现错误并 backtrack
- **Break down complex tasks**: 自动把问题分成 sub-problems
- **Explore alternate paths**: 一种方法失败后换路径

这对应 Sutton 在 RL 里讲的 "stream of observations, actions, rewards" 的本质 —— CoT 的每一步 token 都是一个 action，每个 action 改变后续的 context state。

### 2.2 Tool Use 的关键性

o1 被训练成会**在沙箱中执行 code** 来 verify 自己的 outputs。这个细节很重要：因为它让 model 可以在一个 sample 内**iteratively refine** —— 比如写一段 C++，编译运行，看到 segfault，回去改指针，再运行。这是一种局部的 model-based RL search，在 single rollout 内完成。

参考：Toolformer (Schick et al., 2023): https://arxiv.org/abs/2302.04761

### 2.3 CodeForces 评估方法

关键细节：
- 用 **Division 1 contests** from 2024 和 Dec 2023
- 所有 test contests 都在 pretraining 和 RL data cutoff 之后
- 用 OpenAI embedding API 做 **contamination check**
- 用 **完整 test suite**（不是只有 pre-tests）
- 遵守时间/内存限制
- **pass@10**：10 次 independent submissions 中任一通过即算 solved（human 在 Division 1 的 pre-tests 通常 strong，所以这接近 human 的 affordance）

**Rating 计算公式**（Elo-like）：

$$
P(A \text{ beats } B) = \frac{1}{10^{(R_B - R_A)/400}}
$$

变量解释：
- $R_A, R_B$：competitor A 和 B 的 ratings
- $P$：A 在 final standings 中排名高于 B 的概率
- 400：标准 Elo scale factor，表示 rating 差 400 时胜率约 10:1
- 公式来自 CodeForces creator 的三篇 blog: https://codeforces.com/blog/entry/102, https://codeforces.com/blog/entry/20762, https://codeforces.com/blog/entry/77890

**估计 model rating 的方法**：先在所有 test contests 中算 model 的 rank（按总 score），然后**最大化 observed rankings 与 human ratings 的 likelihood**，对参赛人数多的 contest 取平均避免 overweight。

**Thinking time 的处理**：human 提交越快得分越高，但 model 可以并行思考，所以 OpenAI 选择**用 human 中位数**估计 model 的 score（即 solved same problem with same number of failed attempts 的 human 参赛者的 score 中位数）。如果用真实并行时间会得到更高 rating。

---

## 3. o1-ioi 的 Test-time Strategy 详解

这是 paper 里**最有信息量**的部分 —— 一个完全 hand-engineered 的 pipeline，可以让你理解 RL 之前人们怎么做 test-time compute。

### 3.1 Coding RL Fine-tuning

从 o1 checkpoint resume RL，但 focused on：
1. **Challenging programming problems**（提升 C++ generation 和 runtime checks）
2. **IOI submission format**（让 model 直接输出 IOI 格式）

直觉上这是 **domain-specific RL fine-tuning**，比 o1 多了一阶段的高难度 coding RL。

### 3.2 Test-time Strategy 的四步 pipeline

**Step 1: Problem formulation（subtask 分解）**

IOI 每题分为多个 subtasks，scoring 按 subtask 分别计分，取每次 attempt 的最大值。所以 o1-ioi 把每题**拆成独立的 subtask 文档**（其他 subtask 信息去除），分别求解。这相当于把一个 hard problem 切成多个 easier problems。

**Step 2: 大规模 sampling + clustering**

对每个 subtask：
1. 从 o1-ioi 采样 **10,000 candidate solutions**
2. Prompt model 写 C++ random test input generator
3. 生成 256 random test inputs
4. Prompt model 写 validator（检查 test input 是否满足 subtask constraints）
5. 只保留通过 ≥75% validators 的 test inputs
6. 用 256 个 test inputs 的 outputs 聚类：**outputs 完全一致的 programs 进入同一 cluster**

直觉：clustering 是为了 **diversity**。如果 10K samples 中 9K 都是同一个 buggy 解，但 1K 是各种不同解，naive majority voting 会被 buggy 解主导。Clustering 让你从每个"解族"中挑一个代表，分散 risk。

**Step 3: Reranking**

每个 solution 的 score 由三部分组成：
- **Learned scoring function** 的输出（一个训练好的 ranker）
- **Model-generated test inputs** 上的 error
- **Public test cases** 上的 fail

每个 cluster 的 score：

$$
\text{Score}(\text{cluster}) = \frac{1}{|C|}\sum_{s \in C} \text{score}(s) - \lambda \cdot \text{penalty per submission from cluster}
$$

变量解释：
- $|C|$：cluster $C$ 中 samples 数
- $\text{score}(s)$：sample $s$ 的综合得分
- $\lambda$：每次从 cluster 提交的 penalty（鼓励不要从一个 cluster 提交太多次）
- Penalty weights 用 **random search on past IOI problems** 直接模拟 submission process 来 tune

**Step 4: Submission（round-robin + filtering）**

- Up to 50 submissions（human 限制）
- Round-robin over subtasks，**从最难的开始**
- 每个 subtask 选 top-ranked cluster 的 top-ranked solution
- Subtask 一旦 solved（满分）停止采样
- 当某个 subtask 是另一个 solved subtask 的 strict superset 时，**filter 掉与 solved subtask outputs 不匹配的 candidates**（因为如果在简单 subtask 上都 fail，难 subtask 几乎肯定也 fail）

### 3.3 实验数据表

| Model/Strategy | CodeForces Rating | Percentile |
|----------------|-------------------|------------|
| gpt-4o | 808 | 11% |
| o1-preview | 1258 | 62% |
| o1 | 1673 | 89% |
| o1-ioi (no test-time) | 1807 | 93% |
| o1-ioi + public test filter | 2092 | 96% |
| o1-ioi + full test-time strategy | 2214 | 98% |
| o3 | 2724 | 99.8% |

每一步带来的 rating 提升：
- o1 → o1-ioi (extra coding RL)：+134 rating
- o1-ioi → + public test filter：+285 rating
- + public filter → + full strategy：+122 rating
- o1-ioi full → o3：+510 rating（**这是质变的一步**）

### 3.4 IOI 2024 Live Competition

- 实战：213 points，49th percentile
- Random 50 submissions 平均只有 156 points
- → **test-time selection strategy 贡献了约 60 points**
- Relaxed 10K submissions（无 selection）：362.14 points，刚好过 gold threshold（≈360）

**关键 insight**：这说明 o1-ioi 的 base model 已经有解出 gold-level 题的能力，瓶颈在**selection**。这是 AlphaCode 范式的延续 —— 大规模 sample + filter。

参考 AlphaCode 的 sample 数：up to **1 million candidates per problem**，最后选 top 10。AlphaCode2 用 Gemini Pro + RL + clustering，CodeForces 85th percentile。

---

## 4. o3：Emergent Test-time Strategy 的质变

o3 是 paper 的"高潮"。核心 claim：**complex test-time reasoning strategies 自然从 end-to-end RL 中涌现**，不需要 human 定义。

### 4.1 o3 的 emergent strategy：brute-force 自检

Figure 6 展示了一个 o3 自己发现的 strategy：对 verification non-trivial 的问题，o3 会写一个**简单的 brute-force solution**（牺牲 efficiency 换 correctness），然后**用 brute-force 的 output cross-check 优化算法的 output**。

代码示例（o3 自己生成的）：

```python
def test_random_small():
    import random
    random.seed(1)
    for n in range(1, 5):
        for m in range(1, n+1):
            s = ''.join(random.choice('ab') for _ in range(n))
            labels = solve_bruteforce_given_s(s, m)
            for k in [1, len(labels)//2+1, len(labels)]:
                k = max(1, min(k, len(labels)))
                ans = solve_main(s, m, k)  # optimized solution
                brute_ans = labels[k-1]    # brute-force solution
                if ans != brute_ans:
                    print("Mismatch...")
                    return False
    print("random small tests passed")
    return True
test_random_small()
```

**直觉**：这就是 competitive programmer 的标准操作 —— 写个 $O(n!)$ 的暴力解来 verify $O(n \log n)$ 的优化解。但 o3 是**自己学会**这个的，没有 human prompt。这非常像 RL agent 在 environment 中 discover 的 "exploration strategy"。

**与 o1-ioi hand-crafted pipeline 的对应**：
- o1-ioi 用 **model-generated test inputs + clustering** 来 filter 错误解
- o3 用 **brute-force cross-check** 来 filter 错误解
- 两种方法目的相同，但 o3 的方法更轻量、更 generalizable

### 4.2 IOI 2024 上的 o3

o3 在 IOI 2024 上的关键差异：

| 维度 | o1-ioi | o3 |
|------|--------|----|
| RL training | Coding-specific fine-tune | 大规模通用 RL |
| Sampling approach | 10K per subtask（per-subtask prompts） | 1K per problem（single prompt with full statement） |
| Selection strategy | Clustering + reranking + round-robin | Top 50 by test-time compute |
| Subtask handling | Manual decomposition | 自然 emerge（单 prompt 处理所有 subtasks） |
| Score (50 submissions) | 213 | 395.64 |
| Score (10K submissions) | 362.14 | - |

**重要细节**：o3 用 **top 50 by test-time compute** 作为 selection —— 也就是说，**让 model "思考最久"的 50 个 solutions** 是最好的。这直接对应了 Figure 2 的发现：test-time compute 与 performance 正相关。这非常像你在 LLMS 讲过的 "universal scaling law for test-time compute"。

直觉：**test-time compute ≈ search depth**。Model 思考越久，相当于在 solution space 里搜索越深，越可能找到正确解。RL 训练让 model 学会了**有效利用思考时间**，而 hand-crafted strategy 则是 external 的 search 算法。

### 4.3 o3 与 o1-ioi 的本质对比

o1-ioi 是 **"sample-then-filter"** paradigm：
- 大量采样
- 外部 oracle（clustering + learned scorer）选最好

o3 是 **"reason-then-verify"** paradigm：
- 适度采样
- Model 自己用 brute-force + self-test 验证

这两种 paradigm 的根本区别在于 **knowledge 在哪里**：
- o1-ioi：knowledge 在 **pipeline 设计者**的脑中
- o3：knowledge 在 **model weights** 中（RL 训练出来的）

这对你来说应该很 intuitive —— 这正是 deep learning 一直以来的 trend：把 human heuristics 蒸馏成 learned representations。

---

## 5. Software Engineering 评估

### 5.1 HackerRank Astra

- 65 个项目级 coding challenges
- 涵盖 React.js, Django, Node.js
- **Multi-file, long-context** scenarios
- **没有 public test cases**（防止 hand-crafted tactics）
- 评估 pass@1 和 average score

| Model | pass@1 | Average Score |
|-------|--------|---------------|
| GPT-4o | ~53.6% | ~64.5% |
| o1-preview | 63.6% (+9.98%) | 70.5% (+6.03) |
| o1 | 63.92% (+3.03% vs o1-preview) | 75.80% |

直觉：reasoning 能力从 competitive programming 迁移到 industry-style software dev。这打破了"算法 ≠ 软件工程"的传统看法。

### 5.2 SWE-bench Verified

OpenAI preparedness team 的人类验证子集（500 tasks），修复了 SWE-bench 的 grading 错误、under-specified statements、过严 unit tests 等问题。

评估 protocol：
- 用 Agentless scaffold（因为 o1-preview 未训练用 code execution / file editing tools）
- 5 次 attempts 生成 candidate patch
- 5 次失败算 incorrect attempt
- 3 trials 平均
- 不惩罚 system failures（container hangs 等），retry

| Model | SWE-bench Verified |
|-------|---------------------|
| gpt-4o | ~33.2% |
| o1-preview | 41.3% (+8.1%) |
| o1 | 49.9% (+8.6%) |
| o3 | 72.7% (+22.8% vs o1) |

**o3 的 22.8% 提升是最显著的跃迁**，说明 RL scaling 对真实软件工程任务的 ROI 甚至比 competitive programming 更大。

参考：
- SWE-bench: https://arxiv.org/abs/2310.06770
- SWE-bench Verified: https://openai.com/index/introducing-swe-bench-verified/
- Agentless: https://arxiv.org/abs/2407.01489

---

## 6. CodeForces Per-problem 详细分析

Table 1 给出了 o3 在 12 个 contests 上的 per-problem 表现，可以提取出几个模式：

**Rating threshold 与 solve rate 的对应**：
- Rating ≤ 1400（easy）：pass@1 ≈ 100%（1140+/1162）
- Rating 1700-2300（medium）：pass@1 大多 50-90%，pass@10 通常 100%
- Rating 2500-2900（hard）：pass@1 急剧下降到 0-50%，pass@10 不稳定
- Rating ≥ 3000（extreme）：基本 unsolved

**典型 hard problem 示例**：
- Contest 2002 F1（rating 2600）：pass@1 = 12/1162 (0.10)，但 pass@10 通过，**8 次 failed submissions** 才解出
- Contest 1975 D（rating 1700）：pass@1 = 9/1162，pass@10 = 0.08，3 次 failed 后 solved —— 说明 o3 在这类 problem 上需要多次 retry

**直觉**：o3 已经接近 top-tier human competitor 在 medium-hard problems 上的表现，但在 3000+ rating 的 extreme problems 上仍有显著 gap。这与 Figure 10 的 global leaderboard 对比一致 —— top 200 humans solve rate >85%，o3 还达不到。

---

## 7. IOI 2024 提交代码深度解析

Paper Appendix C 给出了 o1-ioi 在 IOI 2024 上各题的实际提交代码。这些代码透露出 o1-ioi 学到的算法模式：

### 7.1 Nile（100/100，单次提交满分）

核心数据结构：**Union-Find with augmented metadata**。

```cpp
struct UF {
    vector<int> parent, sizee, minIndex;
    vector<array<long long, 2>> singledPat; // min d[i] for i mod 2 = 0/1 in block
    vector<long long> singledBr;  // singled bridging
    vector<long long> singled;    // singled in block
};
```

**算法直觉**：
- Nodes 按 weight $W$ 排序
- Edges 是相邻 sorted nodes 之间的 diff $W[i+1] - W[i]$
- **Bridging edges** 是跳过一个 node 的连接 $W[i+1] - W[i-1]$
- 当 query $D$ 增大时，逐步 union 相邻 nodes，并 add bridging
- 每个 block 维护 min index mod 2（奇偶性影响是否能 singled）

变量：
- `singledPat[0]` / `singledPat[1]`：block 中 $i \mod 2 = 0$ / $1$ 的最小 $d[i]$，其中 $d[i] = A[i] - B[i]$
- `singledBr`：bridging edge 带来的额外 singled cost
- `sumSingledGlobal`：所有 block 的 singled cost 总和，每次 union 时更新

这是典型的**offline query processing** —— 按 $D$ 排序 queries，用两个 pointer 在 edges 和 bridging 上滑动。

### 7.2 Message（79.64/100）

这是一个**通信协议设计题**，需要在一个 31 列的 channel 上传递 message，其中某些列被 sabotaged（不可靠）。

**Strategy**：
1. **First 4 packets**：在 16 个 safe columns 上放 distinct 4-bit sequences（编码 column index）
2. 从返回的 4 个 packets 中找**唯一的 sequence**，定位一个 known safe column
3. **Next 31 packets**：用 known safe column 传 sabotage subset 的 31 bits，其他 15 safe columns 传 message bits
4. **Next 11 packets**：用 known safe column 传 message length（11 bits 足以表示长度）
5. **Remaining packets**：用全部 16 safe columns 传剩余 message bits

直觉：这是一个 **bootstrapping protocol** —— 先用小规模 redundancy 建立 trusted channel，再用 trusted channel 传 metadata，最后用 metadata 解码 message。和实际网络协议设计很像。

### 7.3 Tree（30/100，两个 submissions）

第一份 submission 17 points：用**前缀和 + 二分**，处理 $b[i] \leq R/L$ 的 case。
第二份 submission 13 points：用**piecewise linear function (PWL) 合并**，每个 node 维护一个 cost function $f(s)$，combine children 的 PWL，再用 `parentFormula` 加入 parent 的 weight。

`combineChildren` 用 priority queue 按 slope 增量合并多个 PWL，这是**convex hull trick** 的扩展。

### 7.4 Hieroglyphs（44/100）

这是 **universal common subsequence (UCS)** 问题。

第一份 submission 34 points：用 **greedy + segment tree** 检查 conflict。Key idea：对每个 letter $x$ with $cVal[x] = 1$，检查是否存在另一个 letter $y$ 的 interval $[eA_y, lA_y] \times [eB_y, lB_y]$ 与 $x$ 的 interval 相交 —— 如果相交则 UCS 不存在。

第二份 submission 10 points：处理 subtask 3（只有 0/1 两种 letter）。用 `cZ[z]` 和 `cO[w]` 表示"放置 z 个 0 后还需要多少 1"和"放置 w 个 1 后还需要多少 0"，greedy 构造。

### 7.5 Mosaic（42/100）

这是 cellular automaton 题，$A[i][j] = (1 - A[i-1][j]) \cdot (1 - A[i][j-1])$。

第一份 submission 22 points：直接 $O(N^2)$ 填表 + 2D prefix sum。对 $N \leq 2000$ 可行。

第二份 submission 20 points：观察到 interior 区域是 **periodic pattern**（周期 2），用 `countEvenInRange` 和 `countOddInRange` 直接 $O(1)$ 算 interior，对 $N \leq 2 \cdot 10^5$ 可行。但只 cover 部分 subtasks。

### 7.6 Sphinx（71.5/100）

这是 **graph coloring with queries** 题，可以用 `perform_experiment` 探测 connected components。

第一份 submission 50 points：**binary search on connected components**，递归地用 experiment 决定 vertex v 应该 merge 到哪些之前的 components。复杂度 $O(N \log N)$ queries。

第二份 submission 43 points：**independent set heuristic + group testing**，先找 independent set（同一 color 内部不连通），再对每个 color 用 binary search 定位 vertices。

直觉：o1-ioi 学到的 graph algorithms 相当 sophisticated，对应 IOI 选手的高水平策略。

---

## 8. 关键 Takeaways 与 Open Questions

### 8.1 Scaling Laws 的两个维度

Paper 反复强调的两个 axes：

$$
\text{Performance} = f(\text{RL training compute}, \text{test-time compute})
$$

Figure 2 显示两个 axes 都是 log-linear。这和你之前在 "Scaling Laws for Neural Language Models" 以及后续 work 里的传统三轴（data, params, compute）形成对比 —— **LRM 时代，test-time compute 成为第四个 axis**。

### 8.2 Hand-crafted vs Emergent 的 trade-off

| 维度 | Hand-crafted (o1-ioi) | Emergent (o3) |
|------|----------------------|---------------|
| Engineering effort | 高（需 IOI 专家设计 pipeline） | 低（只需更多 RL compute） |
| Generality | 仅限 IOI-like tasks | 跨 task 迁移（CodeForces + SWE-bench + Astra） |
| Interpretability | 高（pipeline 透明） | 低（CoT 不易 audit） |
| Upper bound | 受 designer 想象力限制 | 受 RL compute 限制 |
| Sample efficiency | 低（需 10K samples） | 高（1K samples） |

### 8.3 与 AlphaCode 范式的对比

AlphaCode / AlphaCode2 的范式：
- Large-scale sampling（up to 1M）
- Clustering by outputs
- Hand-crafted selection

o3 的范式：
- Moderate sampling（1K）
- Self-verification via brute-force
- Emergent test-time reasoning

**AlphaCode 是 sample-then-filter，o3 是 reason-then-verify**。后者的 sample efficiency 显著更高，因为 model 学到了**主动验证**而非被动 filter。

### 8.4 Open Questions

Paper 没明确说但 implied 的几个：

1. **CoT 的内部结构**：o3 在 CoT 里"思考"的是什么？是否对应 human 算法设计过程？
2. **Sample efficiency 上限**：o3 用 1K samples + 50 submissions 拿 gold，能否 100 samples + 1 submission？
3. **Cross-domain transfer**：o3 在 competitive programming 上的 RL 是否迁移到数学、定理证明？
4. **Failure modes**：o3 在 rating 3000+ problems 上的 failure 是什么模式？是 missing algorithm、insufficient search depth，还是 reasoning error？
5. **RL curriculum**：o3 的 RL training 用了什么 curriculum？是否从 CodeForces Div 2 逐步到 Div 1？

---

## 9. 个人 Intuition 与 Broader Context

这篇 paper 在我看来是 RL scaling 在 reasoning domain 的**第一次大规模实证**。几个深层直觉：

### 9.1 RL 在 LLM 上的本质

o1 / o3 的 RL 本质上是让 model 学会**搜索自己的 reasoning space**。传统 MCTS 在 explicit game tree 上搜索，LRM 则在 token sequence space 上搜索 —— 每个 token 是一个 action，context 是 state。RL reward 是最终答案的正确性，所以 model 必须学会**long-horizon planning**。

这也解释了为什么 test-time compute 和 RL training compute 都重要：RL training 学会"如何 search"，test-time compute 提供"实际搜索的预算"。

### 9.2 与 AlphaGo 的类比

AlphaGo 的 evolution：
- Supervised learning on human expert moves
- RL self-play
- MCTS at test time

AlphaZero 进一步：pure RL，MCTS 完全 learned。

o1 → o3 的 evolution 完全 parallel：
- o1：基础 RL + 通用 CoT
- o1-ioi：hand-crafted test-time strategy（类似 AlphaGo 的 hand-tuned MCTS）
- o3：emergent test-time strategy（类似 AlphaZero 的 learned search）

如果这个类比成立，**下一代模型可能完全摒弃 hand-crafted scaffolding，包括 tool use 协议、verification 步骤等**。

### 9.3 Test-time Compute 的经济学

Paper Figure 2 的 log-linear 关系暗示：

$$
\text{Performance} \propto \log(\text{FLOPs at test time})
$$

这意味着每翻倍 test-time compute，性能提升固定 delta。这与 training-time scaling law 形式相同。**两个 scaling laws 叠加** —— training-time 学会"如何用 test-time compute"，test-time compute 实际执行。

直觉：LRM 的 cost model 正在从"训练贵，推理便宜"转向"训练贵，推理也贵"。这可能改变 deployment economics —— 高价值任务（IOI、复杂 code refactor、科研）愿意花 $10-100 per query 的 inference compute。

### 9.4 与 Karpathy 自己工作的联系

你之前讲过 "software 2.0" —— 神经网络取代 hand-written code。o1-ioi → o3 的 evolution 是 **software 2.0 的 second-order version**：不仅 solution 是 learned，连"如何找 solution 的 strategy"也是 learned。这进一步压缩了 hand-engineering 的空间。

参考你自己之前关于 LLM scaling 的 talk：
- "State of GPT" (MS Build 2023): https://www.youtube.com/watch?v=bZQun8X4B2g
- "Intro to LLMs" (YouTube): https://www.youtube.com/watch?v=zjkBMFhNj jm

### 9.5 与 DeepSeek-R1 / Kimi k1.5 的对比

DeepSeek-R1 和 Kimi k1.5 也用 RL 学 CoT，但他们的 focus 是数学。这篇 paper 暗示 RL 在 coding 上同样有效，且 competitive programming 是**比数学更 objective 的 benchmark**（code 可执行，数学证明验证更难）。

DeepSeek-R1 的训练分两阶段：
1. Cold-start with long CoT SFT
2. RL with rule-based rewards（math/coding 可验证）

OpenAI 的 o1/o3 训练细节没公开，但 paper 暗示是 **end-to-end RL without explicit cold-start**。如果属实，这是显著差异。

参考：
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Kimi k1.5: https://arxiv.org/abs/2501.12599

### 9.6 局限与未来方向

Paper 没讨论但我觉得重要的：

1. **Long CoT 的可解释性**：o3 在 256K+ tokens 的 CoT 里到底在做什么？是否有清晰的 algorithm discovery 过程？
2. **Out-of-distribution generalization**：o3 在 CodeForces 上 99.8th percentile，但在不同风格的 competitive programming（如 AtCoder、LeetCode hard、ICPC）上表现如何？
3. **Sample diversity**：o3 的 1K samples 是独立同分布，还是有 implicit diversity？（如果是 i.i.d.，clustering 应该没用，但 RL 训练的 model 是否真的产生 diverse samples？）
4. **Verification bottleneck**：o3 用 brute-force 自检，但有些 problem 没有 obvious brute-force（如 interactive problems, optimization problems）。这类 problem 上 o3 的表现如何？这可能是 o3 在 3000+ rating problems 上的主要 failure mode。

---

## 10. 总结

这篇 paper 的核心贡献：

1. **实证了 RL scaling 在 competitive programming 上的 power**：o3 达到 CodeForces 2724（99.8th percentile），IOI 2024 gold
2. **证明了 emergent test-time strategy 优于 hand-crafted**：o3 不用 clustering/reranking，用自己学的 brute-force 自检，反而更好
3. **展示了 reasoning 能力跨 domain 迁移**：competitive programming 上的 RL 也提升 SWE-bench、Astra 等真实 software engineering tasks
4. **确立了 test-time compute 作为第四个 scaling axis**：与 training compute 一样 log-linear

对你 build intuition 来说，最关键的点是：

**LRM 的本质是 model 学会"如何 search" —— 在 token space 上做 RL agent，每个 sample 是一个 episode，CoT 是 action sequence，答案正确性是 reward。** Scaling 这个 agent 的 training compute 让它学会更聪明的 search heuristic；scaling test-time compute 让它实际搜索更深。Hand-crafted test-time strategies 在 LRM 时代正在被 learned strategies 取代，正如 hand-crafted features 在 CNN 时代被 learned features 取代一样。

未来方向（paper 暗示但没明说）：
- 完全废弃 scaffolding 的 agentic system
- Multi-turn RL with environment feedback（不只是 code execution，而是真实 IDE / browser）
- Cross-modal reasoning（code + math + vision）
- Long-horizon tasks（multi-day software projects）

参考相关 work：
- OpenAI o1 blog: https://openai.com/index/learning-to-reason-with-llms/
- OpenAI o3 system card: https://openai.com/index/openai-o3-system-card/
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Kimi k1.5: https://arxiv.org/abs/2501.12599
- AlphaCode (Science): https://www.science.org/doi/10.1126/science.abq1158
- AlphaCode 2: https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf
- Toolformer: https://arxiv.org/abs/2302.04761
- Chain-of-Thought (Wei et al.): https://arxiv.org/abs/2201.11903
- SWE-bench: https://arxiv.org/abs/2310.06770
- Codex: https://arxiv.org/abs/2107.03374
- Program Synthesis with LLMs (Austin et al.): https://arxiv.org/abs/2108.07732
- Codeforces rating system: https://codeforces.com/blog/entry/102

希望这个深度解析对你 build intuition 有帮助。如果有具体 sub-section 想进一步 drill down（比如 o1-ioi 的 clustering math、o3 的 CoT 内部结构、或具体 IOI 题目的 algorithm），可以继续展开。
