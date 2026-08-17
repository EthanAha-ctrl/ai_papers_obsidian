---
source_pdf: GDPO.pdf
paper_sha256: 77a9412a04c3ea358d1305e7815ed3af82afb783261b25966f93f9ff8f922193
processed_at: '2026-08-04T12:54:12-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说GDPO

## 一句话总结

**GRPO在多个reward同时优化的时候，会把不同情况压成同一个分数，导致模型分不清谁好谁坏。GDPO的解法是：每个reward先各自打分，最后再合起来。**

---

## 一个生活化的比喻

想象你在训练一个学生参加考试，考试有两部分：**数学**和**语文**，每科0-100分。

你让这个学生做两套试卷（两个rollout），你想通过比较这两套试卷的表现来告诉学生"下次多往哪个方向努力"。

### GRPO的做法（有问题）

GRPO先把两科分数加起来：
- 试卷A：数学80 + 语文80 = 160
- 试卷B：数学90 + 语文0 = 90

然后比较总分：A比B高70分，所以告诉学生"试卷A的方式更好"。

看起来合理？但考虑这个场景：
- 试卷A：数学80 + 语文80 = 160
- 试卷B：数学90 + 语文70 = 160

总分一样！GRPO会说"两个差不多"。但实际上试卷B数学更好、语文稍差，这个信息**被总分压掉了**。

更极端的情况：
- 试卷A：数学0 + 语文100 = 100
- 试卷B：数学100 + 语文0 = 100
- 试卷C：数学50 + 语文50 = 100

GRPO看到三个都是100分，觉得都一样。但A是偏科语文、B是偏科数学、C是均衡——这三种情况应该给学生不同的反馈。

### GDPO的做法（更好）

GDPO说：**别急着加总，先每科单独比**。

对于数学：
- A=0, B=100, C=50 → B最好，C其次，A最差

对于语文：
- A=100, B=0, C=50 → A最好，C其次，B最差

然后再合起来告诉学生：
- A：数学很差，语文很好 → 多补数学
- B：数学很好，语文很差 → 多补语文  
- C：都还行，保持

**每个学生都得到了精准的反馈**，这就是GDPO的核心。

---

## 为什么GRPO会出问题？根本原因

GRPO的流程是：**先求和 → 再归一化**。

求和这一步是罪魁祸首。求和是一个"多对一"的操作——不同的reward组合可以加出相同的总和。一旦加总了，你没法从总分公司还原出原来的分布。

打个比方：你去超市买了苹果3个、香蕉2个，总共5个水果。别人只知道"5个水果"，不知道你买了什么。信息在求和的那一刻就丢了。

归一化（减均值除标准差）只是一个线性变换，它不能把丢掉的信息找回来。

---

## GDPO怎么解决的？

GDPO的流程是：**先每个reward各自归一化 → 再求和 → 最后再归一化一次**。

还是用考试的例子：

### Step 1: 每科单独打分

数学分数归一化：把每个学生的数学分减去全班数学平均分，除以全班数学分的标准差。语文同理。

这样每个学生得到两个分数：数学的相对表现、语文的相对表现。**两个维度的信息都保留了**。

### Step 2: 把两个分数加起来

数学相对分 + 语文相对分 = 综合相对分。

这时候虽然也加了，但加的是**已经分别归一化过的分数**，每个维度都在相同的scale上，不会出现"数学分天然比语文分大100"这种问题。

### Step 3: 再做一次batch-wise归一化

因为如果你有10个reward，加起来之后的数值可能比较大，影响训练稳定性。所以最后再归一化一次，把数值拉回稳定范围。

---

## 为什么光调reward weight不够用？

论文里有个很有意思的发现：当两个reward**难度差很多**的时候，你给容易的reward很小的权重，模型还是先去刷那个容易的reward。

### 真实例子

数学推理任务里有两个reward：
- **Length reward**（容易）：回答不超过4000 token就得1分，特别容易满足
- **Correctness reward**（难）：答对数学题得1分，很难

你心想：那我把length reward的权重调低，比如从1.0调到0.5，模型应该更关注correctness了吧？

**实验结果：几乎没用**。length-exceeding ratio（超长回答的比例）几乎没变化。只有把权重降到0.25，才看到一点点效果。

### 为什么会这样？

因为length reward太容易了，模型稍微一努力就能拿到满分。权重0.5意味着"拿到了就是0.5分"，但cost几乎为零，模型还是觉得"白拿白不拿"。

这就像打工：时薪100块的活和时薪50块的活，你不会因为50块的降到了25块就不做了——反正顺手就做了，还能多拿点钱。

### GDPO的解决方案：Conditioned Reward

既然调权重没用，那就改规则：**你必须先答对数学题，才有资格拿length reward**。

公式就是：
- 如果回答正确且不超长 → length reward = 1
- 如果回答错误，不管超不超长 → length reward = 0

这相当于在reward之间建立了**依赖关系**：correctness是length的前提。

效果立竿见影：模型不再一开始就疯狂刷length reward了，而是先去解决correctness，因为不解决correctness，length reward根本拿不到。

这就像老板说："这个月的奖金，只有当你KPI达标了才发，KPI没达标一分钱没有。"员工自然会把KPI放在第一位。

---

## 实验结果有多 convincing？

### Tool Calling（工具调用）

用Qwen2.5-1.5B做工具调用训练，在BFCL-v3上测试：

| 方法 | 准确率 | 格式正确率 |
|------|--------|------------|
| Baseline | 17.88% | 4.74% |
| GRPO | 30.18% | 76.33% |
| **GDPO** | **32.81%** | **80.66%** |

GDPO比GRPO高了2.7%准确率，4.3%格式正确率。

### Math Reasoning（数学推理）

用DeepSeek-R1-7B在AIME上测试：

| 方法 | 准确率 | 超长回答比例 |
|------|--------|-------------| 
| Baseline | 55.4% | 85.6% |
| GRPO | 50.2% | 2.1% |
| **GDPO** | **53.1%** | **0.2%** |

注意GDPO不仅准确率更高，超长回答的比例也更低（0.2% vs 2.1%）。**两个目标都更好**。

### Training Stability（训练稳定性）

这个可能是最impressive的结果。在数学推理任务中，GRPO训练到400步左右，correctness reward开始**下降**——训练崩了。而GDPO一路稳定上升，没有崩。

这个现象论文里叫"training collapse"。GRPO在多reward下长时间训练容易崩，GDPO不会。

### Coding Reasoning（代码推理）

三个reward同时优化（pass rate + length + bug ratio），GDPO在bug ratio上有明显优势：

| Benchmark | 方法 | Pass | Bug Ratio |
|-----------|------|------|-----------|
| Codeforces | GRPO | 69.5% | 2.5% |
| Codeforces | **GDPO** | 69.4% | **1.8%** |
| Taco | GRPO | 44.4% | 30.0% |
| Taco | **GDPO** | 45.1% | **28.0%** |

三个reward的情况下GDPO依然work，说明它的scalability没问题。

---

## 这个工作为什么重要？

### 1. 指出了一个被忽视的问题

现在大家都在做多reward RL（accuracy + length + format + safety...），默认都用GRPO。但这篇paper发现GRPO在多reward下有**结构性缺陷**——不是调参能解决的。

### 2. 解决方案简单且principled

GDPO不需要改架构、不需要加模型、不需要额外数据。就是把normalization的顺序换一下：先per-reward归一化，再求和，再batch归一化。**几行代码的事**。

### 3. 揭示了reward设计的深层问题

Weight adjustment在很多情况下不够用，需要改reward的结构（conditioned reward）。这个insight对做RL alignment的人很有价值。

### 4. 实验非常充分

三个不同任务（tool calling、math、coding），不同model size（1.5B到7B），不同reward数量（2个到3个），GDPO都稳定优于GRPO。这个generalizability很impressive。

---

## 给practitioner的建议

如果你正在做多reward RL，而且用的是GRPO，建议：

1. **换成GDPO**：代码改动很小，论文有HF-TRL、verl、Nemo-RL的实现
2. **检查reward难度差异**：如果某个reward特别容易刷满，考虑用conditioned reward而不是调权重
3. **长训练要注意**：GRPO在多reward下长时间训练可能崩，GDPO更稳定
4. **Batch-wise normalization别省**：论文附录实验证明去掉它偶尔会convergence failure

---

## 相关链接

**论文和代码**：
- GDPO paper (NVIDIA): 搜索 "GDPO Group reward-Decoupled Normalization Policy Optimization"
- verl框架: https://arxiv.org/abs/2409.19256
- HF-TRL: https://github.com/huggingface/trl
- Nemo-RL: NVIDIA的RL训练框架

**对比方法**：
- GRPO原始paper (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DAPO: https://arxiv.org/abs/2503.14476
- Dr.GRPO: https://arxiv.org/abs/2503.20783
- DLER (conditioned reward的灵感来源): https://arxiv.org/abs/2510.15110

**评测数据集**：
- BFCL-v3 (tool calling): https://gorilla.cs.berkeley.edu/leaderboard.html
- DeepScaleR (math): https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2
- AIME: https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions

**基础方法**：
- PPO: https://arxiv.org/abs/1707.06347
- DeepSeek-R1: https://arxiv.org/abs/2501.12948

---

## 最后的intuition

用一个更抽象的比喻收尾：

GRPO像是用一个**总分**来评价学生——不管你是偏科还是均衡，只要总分一样就一视同仁。

GDPO像是**先看每科排名，再综合排名**——能区分出"偏科天才"和"均衡好学生"的差别。

在多reward RL里，这种区分能力至关重要，因为不同的reward组合代表不同的行为模式，模型需要知道"哪种行为模式更好"，而不是只看"总分更高"。

这就是GDPO的核心价值：**让训练信号更精细，让模型学到真正有区分度的行为**。

---

# GDPO: 多奖励RL优化的深度解析

## 核心问题: GRPO的Reward Collapse

这篇paper揭示了一个被广泛忽视的问题: 当GRPO直接应用于multi-reward setting时, 会发生**reward signal collapse**。让我用最简单的例子build your intuition。

### 问题可视化

考虑两个binary reward $r_1, r_2 \in \{0, 1\}$, 两个rollout的场景。总reward $r_{sum} \in \{0, 1, 2\}$, 所有可能的组合(忽略顺序)有6种:

| Reward Combination | $r_{sum}$ values | GRPO Normalized Advantage |
|---|---|---|
| (0, 0) | (0, 0) | (0, 0) |
| (0, 1) | (0, 1) | (-0.7071, 0.7071) |
| (0, 2) | (0, 2) | **(-0.7071, 0.7071)** |
| (1, 1) | (1, 1) | (0, 0) |
| (1, 2) | (1, 2) | **(-0.7071, 0.7071)** |
| (2, 2) | (2, 2) | (0, 0) |

**关键观察**: 6种不同的reward combination只产生了2种distinct advantage group! 这就是collapse。

Intuition上, (0, 2)应该比(0, 1)有更强的learning signal—因为(0, 2)意味着同时满足了两个reward, 而(0, 1)只满足了一个。但GRPO把它们压缩成了相同的advantage值。

### 为什么会collapse? 数学直觉

GRPO的核心公式:

$$A_{sum}^{(i,j)} = \frac{r_{sum}^{(i,j)} - \text{mean}\{r_{sum}^{(i,1)}, \ldots, r_{sum}^{(i,G)}\}}{\text{std}\{r_{sum}^{(i,1)}, \ldots, r_{sum}^{(i,G)}\}}$$

这里 $r_{sum}^{(i,j)} = r_1^{(i,j)} + \cdots + r_n^{(i,j)}$, 上标 $(i,j)$ 表示第 $i$ 个question的第 $j$ 个rollout, 下标 $sum$ 表示对所有reward求和。

问题的本质: **先sum再normalize**会丢失reward维度信息。Sum操作把高维信息压扁成一维标量, normalize只在这个标量空间做线性变换, 无法恢复被压掉的信息。

### Dr.GRPO / DeepSeek-v3.2的修改不够

Dr.GRPO移除了std normalization:

$$\bar{A}_{sum}^{(i,j)} = r_{sum}^{(i,j)} - \text{mean}\{r_{sum}^{(i,1)}, \ldots, r_{sum}^{(i,G)}\}$$

表面上看这解决了binary case的问题:
- (0, 1) → (-0.5, 0.5)
- (0, 2) → (-1.0, 1.0)

但当rollout数量 $G$ 或reward数量 $n$ 增大时, distinct advantage group数量增长非常缓慢(paper Figure 3的实验证实)。

---

## GDPO方法: Decoupled Normalization

### 核心idea

**不要sum再normalize, 而是normalize再sum**。这保留了每个reward维度的相对信息。

### 三步公式

**Step 1: Per-reward group-wise normalization**

对每个reward $k \in \{1, \ldots, n\}$ 独立归一化:

$$A_k^{(i,j)} = \frac{r_k^{(i,j)} - \text{mean}\{r_k^{(i,1)}, \ldots, r_k^{(i,G)}\}}{\text{std}\{r_k^{(i,1)}, \ldots, r_k^{(i,G)}\}}$$

变量含义:
- $A_k^{(i,j)}$: 第 $i$ 个question的第 $j$ 个rollout在第 $k$ 个reward上的normalized advantage
- $r_k^{(i,j)}$: 第 $i$ 个question的第 $j$ 个rollout的第 $k$ 个reward原始值
- $\text{mean}\{r_k^{(i,1)}, \ldots, r_k^{(i,G)}\}$: 第 $i$ 个question的 $G$ 个rollout在第 $k$ 个reward上的均值
- $\text{std}\{\cdot\}$: 对应的standard deviation

**Step 2: Sum normalized advantages**

$$A_{sum}^{(i,j)} = A_1^{(i,j)} + \cdots + A_n^{(i,j)}$$

如果有reward weight:

$$A_{sum}^{(i,j)} = w_1 A_1^{(i,j)} + \cdots + w_n A_n^{(i,j)}$$

**Step 3: Batch-wise advantage normalization**

$$\hat{A}_{sum}^{(i,j)} = \frac{A_{sum}^{(i,j)} - \text{mean}\{A_{sum}^{(i',j')} | i' \in D_{Batch}, j' = 1, \ldots, G\}}{\text{std}\{A_{sum}^{(i',j')} | i' \in D_{Batch}, j' = 1, \ldots, G\} + \epsilon}$$

这里 $D_{Batch}$ 是当前batch的所有question, $\epsilon$ 是数值稳定性的小常数。

### 为什么要batch-wise normalization?

Intuition: 随着reward数量 $n$ 增加, $A_{sum}$ 的数值scale会线性增长(因为是 $n$ 个unit-variance的量相加)。这会导致gradient scale不稳定。Batch-wise normalization把advantage重新scale到稳定的数值范围。

Paper Appendix A的实验证实: 去掉batch-wise normalization偶尔会导致convergence failure。

### 验证GDPO的有效性

回到binary two-reward example:

| Reward Combination | $A_1$ | $A_2$ | $A_{sum} = A_1 + A_2$ |
|---|---|---|---|
| (0, 1) | (-0.7071, 0.7071) | (0, 0) | (-0.7071, 0.7071) |
| (0, 2) | (-0.7071, 0.7071) | (-0.7071, 0.7071) | (-1.4142, 1.4142) |
| (1, 2) | (0, 0) | (-0.7071, 0.7071) | (-0.7071, 0.7071) |

现在有3个distinct advantage group, 而且(0, 2)确实比(0, 1)有更强的learning signal(1.4142 > 0.7071)。

### 为什么(1, 2)和(0, 1)得到相同的advantage?

这其实合理! (1, 2)意味着rollout1满足reward1, rollout2满足reward2; (0, 1)意味着rollout2满足reward1(或reward2), 两者在"一个rollout比另一个好"的程度上是一样的, 只是方向不同。GDPO通过per-reward normalization捕捉到了这个对称性。

---

## 优化目标公式

完整的GDPO目标函数(省略KL term):

$$\mathcal{J}_{GDPO}(\theta) = \mathbb{E}_{(q_i, o_j) \sim D, \{o_j\}_{j=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{j=1}^G \frac{1}{|o_j|} \sum_{t=1}^{|o_j|} \min\left(s_{i,t}(\theta) \hat{A}_{sum}^{(i,j)}, \text{clip}(s_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{sum}^{(i,j)}\right) \right]$$

其中:
- $s_{i,t}(\theta) = \frac{\pi_\theta(o_j^t | q, o_j^{<t})}{\pi_{\theta_{old}}(o_j^t | q, o_j^{<t})}$ 是importance sampling ratio
- $o_j^t$ 是第 $j$ 个rollout的第 $t$ 个token
- $o_j^{<t}$ 是前 $t-1$ 个token的prefix
- $\epsilon$ 是PPO的clipping threshold
- $|o_j|$ 是rollout $j$ 的长度
- $\hat{A}_{sum}^{(i,j)}$ 是经过GDPO三步计算得到的最终advantage

---

## Priority Variation: 当reward难度差异大时

### 问题: Reward Weighting的局限性

标准做法: $r_{sum} = w_1 r_1 + \cdots + w_n r_n$, 通过调整 $w_k$ 来控制priority。

但paper发现: 当reward难度差异显著时, 调整weight效果有限。模型倾向于先优化容易的reward, 即使给容易的reward很小的weight。

Paper Section 4.2.1的实验: 在math reasoning中, length reward比correctness reward容易得多。把length reward的weight从1.0降到0.5, 对length-exceeding ratio几乎没有影响。只有降到0.25才看到明显效果。

### 解决方案: Conditioned Reward

给定两个reward $r_k$(容易)和 $r_l$(难), 把 $r_k$ conditioned on $r_l$:

$$r_k = \begin{cases} r_k, & \text{if } r_l \geq t \\ 0, & \text{otherwise} \end{cases}$$

其中 $t$ 是预设的threshold。

**具体例子(Math reasoning)**:

原始length reward:
$$\mathcal{R}_{length} = \begin{cases} 1, & \text{if response length} \leq l \\ 0, & \text{otherwise} \end{cases}$$

Conditioned length reward:
$$\tilde{\mathcal{R}}_{length} = \begin{cases} 1, & \text{if response length} \leq l \text{ and } \mathcal{R}_{correct} = 1 \\ 0, & \text{otherwise} \end{cases}$$

Intuition: 模型必须先满足correctness, 才能拿到length reward。这强制模型先解决难的问题。

### 实验数据(Paper Table 4)

DeepSeek-R1-7B on AIME:
- GRPO + $\mathcal{R}_{length}$: Acc 50.2%, Exceed 2.1%
- GDPO + $\mathcal{R}_{length}$: Acc 53.1%, Exceed 0.2%
- GRPO + $\tilde{\mathcal{R}}_{length}$: Acc 53.3%, Exceed 29.2%
- GDPO + $\tilde{\mathcal{R}}_{length}$: **Acc 57.7%, Exceed 12.3%**

关键观察: conditioned reward让Exceed ratio上升(因为放松了length约束), 但只有GDPO能把这种放松转化成accuracy提升。GRPO虽然放松了约束, accuracy提升有限。

---

## 实验详解

### Task 1: Tool Calling

**Setup**:
- Models: Qwen2.5-Instruct-1.5B, Qwen2.5-Instruct-3B
- Training: ToolRL setup, 2k ToolACE + 1k Hammar + 1k xLAM samples
- 4 rollouts per question, batch size 512, max response 1024 tokens, 100 steps
- 两个reward:
  - Format reward $\mathcal{R}_{format} \in \{0, 1\}$: 检查输出结构
  - Correctness reward $\mathcal{R}_{correct} \in [-3, 3]$: 基于tool name matching, parameter name matching, parameter content matching

**Correctness Reward详细公式**(Paper Appendix C):

Tool Name Matching:
$$r_{name} = \frac{|N_G \cap N_P|}{|N_G \cup N_P|} \in [0, 1]$$

其中 $N_G$ 是ground-truth tool name集合, $N_P$ 是predicted tool name集合。

Parameter Name Matching:
$$r_{param} = \sum_{G_j \in G} \frac{|\text{keys}(G_j) \cap \text{keys}(P_j)|}{|\text{keys}(G_j) \cup \text{keys}(P_j)|} \in [0, |G|]$$

其中 $\text{keys}(G_j)$ 是第 $j$ 个ground-truth call的parameter name集合。

Parameter Content Matching:
$$r_{value} = \sum_{G_j \in G} \sum_{k \in \text{keys}(G_j)} \mathbf{1}[P_G[k] = P_P[k]]$$

Total Match Score:
$$r_{match} = r_{name} + r_{param} + r_{value} \in [0, S_{max}]$$

其中 $S_{max} = 1 + |G| + \sum_{G_j \in G} |\text{keys}(G_j)|$。

最终correctness reward通过optimal matching最大化:
$$\mathcal{R}_{correct} = 6 \cdot \frac{R_{max}}{S_{max}} - 3 \in [-3, 3]$$

**BFCL-v3 Results**(Paper Table 1):

| Model | Avg Acc ↑ | Correct Format ↑ |
|---|---|---|
| Qwen2.5-1.5B baseline | 17.88% | 4.74% |
| GRPO (1.5B) | 30.18% | 76.33% |
| **GDPO (1.5B)** | **32.81%** | **80.66%** |
| Qwen2.5-3B baseline | 31.90% | 58.37% |
| GRPO (3B) | 39.20% | 81.64% |
| **GDPO (3B)** | **40.87%** | **82.23%** |

GDPO在1.5B上: +2.7% accuracy, +4.3% format compliance。

### Task 2: Math Reasoning

**Setup**:
- Models: DeepSeek-R1-1.5B, DeepSeek-R1-7B, Qwen3-4B-Instruct
- Training: DeepScaleR-Preview dataset, 40k competition-level math problems, 500 steps
- 16 rollouts, batch size 512, max response 8000 tokens
- 两个reward:
  - Length reward $\mathcal{R}_{length} \in \{0, 1\}$: response length ≤ 4000 tokens
  - Correctness reward $\mathcal{R}_{correct} \in \{0, 1\}$: final answer matches ground truth

**Key Results**(Paper Table 3):

DeepSeek-R1-1.5B on AIME:
- Baseline: 29.8% acc, 91.5% exceed
- GRPO: 23.1% acc, 10.8% exceed
- **GDPO: 29.4% acc, 6.5% exceed**

DeepSeek-R1-7B on AIME:
- Baseline: 55.4% acc, 85.6% exceed
- GRPO: 50.2% acc, 2.1% exceed
- **GDPO: 53.1% acc, 0.2% exceed**

Qwen3-4B on AIME:
- Baseline: 63.7% acc, 71.3% exceed
- GRPO: 54.6% acc, 2.5% exceed
- **GDPO: 56.9% acc, 0.1% exceed**

**Training dynamics观察**(Paper Figure 5):
1. 初期: 两个方法都快速maximize length reward(容易的)
2. 这导致correctness reward初期下降
3. GDPO恢复correctness更快, 最终更高
4. GRPO在~400 steps后correctness开始下降(training collapse)
5. GDPO在整个训练过程中持续改进correctness

### Task 3: Coding Reasoning

**Setup**:
- Model: DeepSeek-R1-7B
- Training: Eurus-2-RL dataset, 24k coding problems, 400 steps
- 三个reward:
  - Pass rate $\mathcal{R}_{pass} \in [0, 1]$: $\frac{\text{number of passed test cases}}{\text{total test cases}}$
  - Conditioned length $\tilde{\mathcal{R}}_{length} \in \{0, 1\}$: length ≤ $l$ and $\mathcal{R}_{pass} = 1$
  - Bug reward $\mathcal{R}_{bug} \in \{0, 1\}$: no runtime/compilation error

**Three-reward Results**(Paper Table 5):

| Benchmark | Method | Pass ↑ | Exceed ↓ | Bug ↓ |
|---|---|---|---|---|
| Apps | GRPO_3-obj | 68.1% | 11.2% | 20.3% |
| Apps | **GDPO_3-obj** | 67.8% | 8.5% | **18.8%** |
| Codeforces | GRPO_3-obj | 69.5% | 16.9% | 2.5% |
| Codeforces | **GDPO_3-obj** | 69.4% | 13.6% | **1.8%** |
| Taco | GRPO_3-obj | 44.4% | 14.7% | 30.0% |
| Taco | **GDPO_3-obj** | 45.1% | 10.6% | **28.0%** |

GDPO在三个reward setting下仍然优于GRPO, 特别是在bug ratio上有明显改善。

---

## Intuition Summary

### 1. 为什么decoupled normalization保留更多信息?

Geometric intuition:
- GRPO: 在标量sum空间做归一化, 类似于把高维点投影到一维直线上
- GDPO: 在每个reward维度独立归一化, 保留了高维空间的几何结构

Information theory视角:
- Sum操作 $r_{sum} = \sum_k r_k$ 是一个many-to-one映射, 不可逆
- 先normalize再sum保留了每个维度的相对ordering信息

### 2. 为什么batch-wise normalization重要?

Variance analysis:
- 每个per-reward normalized advantage $A_k$ 近似unit variance
- $A_{sum} = \sum_k A_k$ 的variance约为 $n$(假设reward独立)
- 不做batch-wise normalization, advantage scale随reward数量线性增长
- 这导致gradient scale不稳定, 影响训练

### 3. 为什么conditioned reward比weight adjustment更有效?

Game theory视角:
- Weight adjustment: 改变了reward的"价格", 但model仍然可以"购买"容易的reward
- Conditioned reward: 改变了reward的"依赖关系", 制造了prerequisite结构
- 这类似于把independent objectives变成sequential objectives

### 4. GRPO何时会fail?

从实验观察:
- 当reward难度差异大时(如length vs correctness)
- 当reward数量增加时(collapse更严重)
- 长时间训练时(training collapse after ~400 steps in math)

---

## 相关工作和web links

### GRPO相关
- DeepSeek-R1 (GRPO原始应用): https://arxiv.org/abs/2501.12948
- DeepSeekMath (GRPO原始提出): https://arxiv.org/abs/2402.03300
- DAPO (GRPO variant): https://arxiv.org/abs/2503.14476
- Dr.GRPO (移除std normalization): https://arxiv.org/abs/2503.20783
- GSPO (sequence-level): https://arxiv.org/abs/2507.18071
- DLER (length penalty): https://arxiv.org/abs/2510.15110

### Multi-reward RL
- Safe RLHF: https://arxiv.org/abs/2310.12773
- ALARM (hierarchical rewards): https://aclanthology.org/2024.acl-long.444/
- DeepSeek-v3.2: https://arxiv.org/abs/2512.02556
- L1 (length control): https://arxiv.org/abs/2503.04697
- O1-Pruner: https://arxiv.org/abs/2501.12570

### Tool calling
- ToolRL: https://arxiv.org/abs/2504.13958
- ToolACE: https://arxiv.org/abs/2409.00920
- BFCL-v3: https://gorilla.cs.berkeley.edu/leaderboard.html

### Math reasoning
- DeepScaleR: https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2
- MATH dataset: https://arxiv.org/abs/2103.03874
- OlympiadBench: https://arxiv.org/abs/2402.14008

### Implementation
- verl framework: https://arxiv.org/abs/2409.19256
- PRIME (coding): https://arxiv.org/abs/2502.01456

---

## 关键Takeaway

1. **GRPO在multi-reward下有fundamental limitation**: reward collapse导致信息丢失, 这是structural问题, 不能通过简单调参解决。

2. **GDPO的核心**: decouple normalization per reward, 然后sum, 再batch-wise normalize。简单但effective。

3. **Conditioned reward > Weight adjustment**: 当reward难度差异大时, 改变reward结构比改变weight更有效。

4. **GDPO的通用性**: 在2-reward和3-reward setting都有效, 在不同task(tool calling, math, coding)都有效, 在不同model size(1.5B, 3B, 4B, 7B)都有效。

5. **训练稳定性**: GDPO避免了GRPO的training collapse问题, 在长训练中持续改进。

这篇paper的核心贡献在于指出了GRPO在multi-reward setting下的一个被忽视的fundamental问题, 并提出了一个简单principled的解决方案。这对于正在做multi-reward RL的practitioner非常有价值。
