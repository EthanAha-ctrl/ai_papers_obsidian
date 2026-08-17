---
source_pdf: rStar-Math.pdf
paper_sha256: a7c2a0e5b09c118c7ba4092d85ba1a045185e5b19f86915bcfbf9806344844f8
processed_at: '2026-08-12T02:23:39-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 rStar-Math

Andrej，好，我换种风格，就像咱俩在白板前聊天那样讲。

---

## 一句话概括

7B 的小模型，通过自己跟自己下棋（MCTS）+ 自己给自己打分（PPM），迭代 4 轮，最后在 math reasoning 上干翻了 OpenAI o1-preview。没有从 GPT-4 蒸馏，完全 self-evolved。

---

## 问题是什么

你想让 LLM 解 math 题。传统做法是 System 1——一次性 generate 完整 solution，像你写 micrograd 那样一气呵成。问题是中间步骤可能错了但最后答案碰巧对，或者中间步骤对了但最后算错了。这种 "answer-level supervision" 没法教模型学好 reasoning。

o1 的思路是 System 2——test-time 多搜索、多思考。但 o1 是黑盒，你不知道它怎么训练的。rStar-Math 想用 open source 的方式复现这个能力，而且只用小模型。

核心难点：你要训练两个东西——
1. **Policy model**：生成 reasoning step 的
2. **Reward model**：判断每步好坏的

两个都需要 high-quality training data。但 high-quality math data 稀缺，GPT-4 distill 有 ceiling，rejection sampling 不保证 intermediate step 正确。

---

## rStar-Math 怎么做的

### Trick 1: Code-augmented CoT

最朴素的想法：让 MCTS 的每个 node 生成 natural language CoT。但 NL 的问题是你没法 verify 它对不对。模型可能写了一堆看似合理的推理，最后答案对了，但中间全是 hallucination。

rStar-Math 的 trick：**每步同时生成 NL CoT + Python code**，NL 嵌在 code comment 里。只有 Python 能跑通才保留这个 node。

这就像你教学生解题，不光让他讲思路，还让他把计算写代码里跑一遍。代码能跑通 = 计算这一步是对的。这是一个 hard filter，比 NL 的 soft verification 强太多了。

具体流程：
```
Step i 的 state = x ⊕ s_1 ⊕ s_2 ⊕ ... ⊕ s_{i-1}
→ policy model 生成 n 个候选 s_{i,0}, ..., s_{i,n-1}
→ 每个候选拼上前面的 code，执行 Python
→ 只有 execute 成功的保留
→ PPM 给保留的打分
→ UCT 选最好的一个
```

### Trick 2: PPM（Process Preference Model）

这是最 clever 的部分。

传统 PRM 的训练方式：给每个 step 标一个 score，然后 model 学 predict 这个 score。问题是——**你怎么标 step 的 score？**

Human annotation 太贵。MCTS 跑出来的 Q-value 太 noisy。比如三个都对的 step，你很难说哪个 0.8 哪个 0.9，这种细粒度 ranking 连 expert human 都做不好。

rStar-Math 的 insight：**Q-value 做不了 absolute scoring，但做得了 relative ordering**。也就是说，Q-value 能告诉你"这步比那步好"，但告诉你"这步值 0.73 分"是扯淡。

所以 PPM 不学 absolute score，学 **pairwise preference**。对每个 step：
- 选 Q-value 最高的 2 个作 positive（必须导向 correct answer）
- 选 Q-value 最低的 2 个作 negative（必须导向 incorrect answer）
- 用 Bradley-Terry loss 训练

Loss 公式：

$$\mathcal{L}_{\text{ppm}}(\theta) = -\frac{1}{4} \mathbb{E}\left[\log \sigma(r_\theta(x, y_i^{\text{pos}}) - r_\theta(x, y_i^{\text{neg}}))\right]$$

变量解释：
- $\theta$：PPM 的参数
- $r_\theta(x, y_i)$：PPM 对 "问题 $x$ + trajectory $y_i$" 的 scalar 输出，范围 $[-1, 1]$（tanh 压缩）
- $y_i^{\text{pos}}$：第 1 到第 $i$ 步的 positive trajectory
- $y_i^{\text{neg}}$：第 1 到第 $i$ 步的 negative trajectory（和 pos 共享前 $i-1$ 步，只第 $i$ 步不同）
- $\sigma$：sigmoid
- $\frac{1}{4}$：2 个 positive × 2 个 negative = 4 对，取平均

这个 loss 就是在说："positive trajectory 的 score 应该比 negative 高"。不要求高多少，只要求高。这避开了 absolute score 的 noise 问题。

这就像你不给学生打绝对分，只让他说"这个解法比那个好"——相对判断比绝对判断容易得多，也更 robust。

### Trick 3: 4-Round Self-Evolution

整个 pipeline 是个 bootstrap loop：

**Round 1**：用 DeepSeek-Coder-V2-Instruct (236B) 当 policy model，跑 MCTS 生成 data。用 terminal-guided Q-value（answer 对就 +1，错就 -1，反向传播给中间 step）。8 rollouts（因为 236B 太贵）。训练出 SLM-r1 和 PPM-r1。PPM-r1 不太可靠因为 Q-value noise。

**Round 2**：换成 7B SLM-r1 当 policy model。16 rollouts（小模型便宜，可以多跑）。Q-value 精度大幅提升。训练出 PPM-r2，这是第一个 reliable 的 reward model。AIME 从 10% 跳到 43.3%——这是质变点。

**Round 3**：用 PPM-r2 在 MCTS 里直接给每个 step 打 initial Q-value（之前是 0 初始化，现在 PPM 给 non-zero 初始值）。这让 MCTS 搜索效率大增。训练出 SLM-r3 和 PPM-r3。

**Round 4**：对 hard problem 加大 rollout 数（16→64→128），换不同 random seed 多次 expand tree。Olympiad-level 覆盖从 62% 提到 80%。

最后 90.25% 的 747k 问题被解决。剩下 10% 没解决的，作者随机查了 20 个，19 个是 GPT-4 合成时 ground-truth label 就错了。所以不是模型不行，是数据 dirty。

---

## UCT 公式详解

MCTS 选 node 用的是经典 UCT：

$$\text{UCT}(s) = Q(s) + c \sqrt{\frac{\ln N_{\text{parent}}(s)}{N(s)}}$$

变量：
- $s$：当前考察的 node
- $Q(s) = \frac{q(s)}{N(s)}$：node $s$ 的平均 Q-value（exploitation 项）
  - $q(s)$：累积 reward（来自 back-propagation）
  - $N(s)$：node $s$ 被访问的次数
- $N_{\text{parent}}(s)$：$s$ 的父节点被访问的次数
- $c$：exploration 常数，paper 里设 2

第一项 $Q(s)$ 是 exploitation——选 reward 高的。
第二项是 exploration——选访问少的。$\ln$ 让 exploration 不会太激进，$c=2$ 偏大说明 rStar-Math 在 SLM 能力弱时需要更多 exploration 来发现 correct path。

---

## Q-value 的两种 annotation 模式

### Terminal-guided（Round 1-2）

$$q(s_i)^k = q(s_i)^{k-1} + q(s_d)^k$$

变量：
- $q(s_i)^k$：第 $k$ 次 rollout 后 step $s_i$ 的 q-value
- $q(s_i)^{k-1}$：上次 rollout 累积的 q-value
- $q(s_d)^k$：本次 rollout 终端 step $s_d$ 的 reward
  - $q(s_d) = +1$ 如果最终答案对
  - $q(s_d) = -1$ 如果最终答案错
- 初始 $q(s_i)^0 = 0$

这就像 AlphaGo——下完棋回头看每一步贡献。多次 rollout 后，好的 step q-value 高，坏的 low。但需要 extensive rollouts 才收敛。

### PPM-augmented（Round 3-4）

$$q(s_i)^0 = \text{PPM}(x \oplus s_1 \oplus \ldots \oplus s_i)$$

PPM 直接给 initial q-value，不靠多次 rollout 累积。后面还是用 Eq. 2 的 back-propagation 更新。好处是搜索效率高，坏处是依赖 PPM 质量。

---

## 架构细节

PPM 架构很简单：
- 从 fine-tuned policy model 初始化（共享 backbone，继承 math reasoning 能力）
- 替换 LM head 为 scalar-value head
- scalar-value head = Linear layer + tanh
- tanh 把输出压到 $[-1, 1]$

这和 AlphaGo 的 value network 思路一样——policy network 和 value network 可以共享 representation。

---

## 实验结果的重点

### Table 1 核心数字

| Task | rStar-Math 7B | o1-preview | o1-mini |
|------|--------------|-----------|---------|
| MATH | **90.0** | 85.5 | 90.0 |
| AIME 2024 | **53.3** | 44.6 | 56.7 |
| Olympiad Bench | **65.6** | - | 65.3 |

7B 模型在 MATH 上超 o1-preview 4.5 个点，AIME 上超 8.7 个点。AIME 53.3% 意味着解了 8/15 题，相当于美国 top 20% 高中生水平。没解的 8 题里很多是 geometry（需要 visual，rStar-Math 不支持）。

### Self-evolution 效果（Table 6）

| Round | MATH | AIME | 
|-------|------|------|
| Base 7B | 58.8 | 0.0 |
| Round 1 (SFT only) | 75.2 | 10.0 |
| **Round 2 (reliable PPM)** | **86.6** | **43.3** |
| Round 3 | 87.0 | 46.7 |
| Round 4 | 89.4 | 50.0 |

Round 2 是 magic moment——reliable PPM 介入，AIME 从 10 跳到 43。这说明 System 2 reasoning 的 bottleneck 在 reward model，不在 policy model。

### PPM vs ORM vs PQM（Table 8）

| RM 类型 | MATH | AIME |
|---------|------|------|
| ORM (end-level reward, Best-of-N) | 82.6 | 26.7 |
| PQM (Q-value regression, MCTS) | 88.2 | 46.7 |
| **PPM (preference, MCTS)** | **89.4** | **50.0** |

ORM 只在 trajectory 末尾给 reward，signal 稀疏。
PQM 直接用 noisy Q-value 做 regression target，上限受限。
PPM 用 Q-value 只做排序，robust，效果最好。

### Figure 5 的核心 insight

不同 size 的 policy model（1.5B/3.8B/7B）Pass@1 差异大，但配上同一个 PPM 做 System 2 后，最终 accuracy 收敛。**Reward model 决定上限，policy model size 不是关键**。

---

## 两个 surprising finding

### 1. Self-reflection 涌现

没有训练 self-correction data，没有 prompt 模型反思。但在 Figure 4 的例子中，模型前 3 步走错路，第 4 步突然意识到不对，backtrack 换方法，最后解对了。

这说明 **MCTS + PPM 的 search 过程本身就蕴含反思**。PPM 给低分 step 告诉模型"这步不行"，模型学会识别自己的烂 step 并换路径。这可能就是 o1 self-reflection 的本质——不是 explicit training 出来的，是 System 2 search 的 emergent behavior。

### 2. PPM 识别定理应用

PPM 对关键定理应用步骤（Fermat Little Theorem, Vieta's formulas, AM-GM, Pythagorean, Shoelace）给 high score。比如 Vieta + AM-GM 那个例子，相关 step 的 PPM score 达到 0.9989 和 0.9999。

PPM 不只学到了"这步对不对"，还学到了"这步是不是 key insight"。这是 semantic understanding，不只是 syntactic check。

---

## 为什么这个 work 成立——intuition

1. **Code execution 是硬验证**：NL 的 correctness 难判，Python 能跑通是客观信号。这就像你让学生用代码验证答案，比让他口头说"我觉得对"靠谱得多。

2. **MCTS 分解难度**：SLM 一次解 Olympiad 题难，但生成单步容易。MCTS 把 search space 分解，每步都是 SLM 能 handle 的简单任务。

3. **Preference > Regression**：Q-value noisy 当 absolute score，但 reliable 当 relative ordering。Bradley-Terry loss 只需要排序对，不需要绝对值精确。这避开 noisy label 痛点。这就像你给学生判卷，"A 比 B 好" 比 "A 得 87 分" 容易得多。

4. **Self-evolution 正反馈**：每轮 data quality 提升 → model 变强 → 生成更好 data。Round 1 的外部强模型 kick-start，之后 SLM 自驱动。

5. **Reward model 是 bottleneck**：Figure 5 证明 policy model size 不重要，PPM 决定上限。这和 o1 的发现一致——reasoning 的关键是 process supervision，不是 policy scale。

---

## 局限

1. **不支持 geometry**：需要 visual understanding
2. **Theorem proving 有限**：Appendix A.2 有 Fermat 例子但还不成熟
3. **GPT-4 合成数据有 noise**：19/20 未解决问题是 label 错了
4. **Inference 贵**：AIME 一个 trajectory 15693 tokens，64 trajectories = 百万 token 量级

---

## 和其他工作的关系

- **vs AlphaGo**：MCTS + value network 的思路完全一致。AlphaGo 的 policy/value network 从 human data bootstrap，然后 self-play 进化。rStar-Math 从 DeepSeek-Coder bootstrap，然后 self-evolve。
- **vs o1**：o1 是单模型 inference，rStar-Math 是 MCTS + policy + reward 多组件。o1 的 self-reflection 是 trained，rStar-Math 是 emergent。
- **vs PRM800k (Let's Verify Step by Step)**：PRM800k 用 human annotation，rStar-Math 用 MCTS Q-value 自动生成 preference pair，可 scale。
- **vs Math-Shepherd**：Math-Shepherd 直接用 Q-value 做 regression，rStar-Math 用 Q-value 做 preference，更 robust。

---

## 代码

https://github.com/microsoft/rStar

---

## 我的 take

这篇 paper 最让我觉得 elegant 的设计是 **PPM 用 preference 代替 regression**。这是一个对 reward signal 本质的深刻理解——承认你的 label noisy，但不让 noise 阻止你提取有用信号。Q-value 的 noise 在 absolute scale 上是致命的，但在 relative ordering 上是 tolerable 的。这种 "understand the nature of your signal" 的思考方式，和你做 micrograd 时强调 "理解 gradient 怎么 flow" 的哲学是一致的。

另外 self-reflection 的涌现很有意思。如果 System 2 search 本身就能涌现反思，那 o1 的 secret sauce 可能不是什么 special training trick，而是 "MCTS + good reward model" 这个架构本身的性质。这对开源社区复现 o1 是个重要 signal。

参考链接：
- Paper: https://arxiv.org/abs/2501.04519
- GitHub: https://github.com/microsoft/rStar
- rStar 前作: https://arxiv.org/abs/2408.06195
- Let's Verify Step by Step: https://arxiv.org/abs/2305.20050
- Math-Shepherd: https://arxiv.org/abs/2406.06592
- AlphaGo: https://arxiv.org/abs/1712.01815
- OpenAI o1: https://openai.com/o1/
- InstructGPT (Bradley-Terry loss): https://arxiv.org/abs/2203.02155
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Scaling test-time compute: https://arxiv.org/abs/2408.03314

---

# rStar-Math 论文深度解读

Andrej，这篇 rStar-Math 是一篇非常扎实的 work，来自 Microsoft Research Asia。核心 contribution 是展示 small language models (SLMs) 通过 self-evolved "deep thinking" 可以在 math reasoning 上 rival 甚至 surpass OpenAI o1，而且**无需从更强模型蒸馏**。让我一层一层拆解给你看。

---

## 1. High-Level Problem & Motivation

传统 LLM 解 math 题是 System 1 thinking——一次性 inference 生成完整 solution，fast but error-prone。rStar-Math 想做的是 System 2：通过 test-time compute scaling（MCTS + reward model）让模型在 inference 时进行 slow、deep 的多步搜索。

要做这件事需要两个 model：
- **Policy model**：生成 promising 的 reasoning steps
- **Reward model**：准确评估每一步的质量

但训练这两个 model 的 bottleneck 在于 high-quality data。现有方法要么 distill from GPT-4（受限于 teacher 能力），要么用 rejection sampling（不保证 intermediate steps 正确）。rStar-Math 的核心洞察是：**SLM 自己 + MCTS 就能 bootstrap 出 high-quality data**，通过 4 轮 self-evolution 迭代提升。

---

## 2. 三大核心创新

### 2.1 Code-augmented CoT Synthesis

这是数据合成的核心 trick。之前的 MCTS 方法主要生成 natural language CoT，但 LLM 经常 hallucinate——intermediate step 错了，但 final answer 碰巧对了（参考 Lanham et al. 2023 的 faithfulness 研究）。这些 flawed steps 极难检测。

rStar-Math 的解法：**每一步同时生成 NL CoT + 对应的 Python 代码**，NL CoT 嵌入为 Python comment。只有 Python 代码能成功 execute 的 generation 才被保留为 valid node。这是一个硬性 filter——代码能跑通是一个客观的 verification signal。

### 2.2 Process Preference Model (PPM)

Process Reward Model (PRM) 最大的难点是 step-level annotation。现有方法要么用 human annotation（贵、不可 scale），要么用 MCTS-generated Q-values 直接作为 reward labels（noisy、imprecise）。

rStar-Math 的关键 insight：**Q-values 虽然不够精确到能 score 每个 step，但足以区分 positive vs negative steps**。所以 PPM 不直接 predict absolute score，而是通过 pairwise preference learning 训练。

### 2.3 Self-evolution Recipe

4 轮迭代，每轮 policy SLM 和 PPM 都变得更强，生成更高质量的 training data。这是一个 bootstrap loop——从 DeepSeek-Coder-V2-Instruct (236B) 开始，之后用自己的 7B SLM 接管。

---

## 3. Methodology 深度解析

### 3.1 MCTS 搜索架构

给定问题 $x$ 和 policy model $M$，MCTS 增量构建搜索树：
- Root node = 问题 $x$
- Child node = 中间 step $s$（由 $M$ 生成）
- 一条 root-to-leaf path 形成 trajectory：$\mathbf{t} = x \oplus s_1 \oplus s_2 \oplus \ldots \oplus s_d$
- 每个 step $s_i$ 被赋予一个 Q-value $Q(s_i)$
- 从搜索树 $\tau$ 提取 trajectories 集合 $\mathbb{T} = \{\mathbf{t}^1, \mathbf{t}^2, \ldots, \mathbf{t}^n\}$

**Selection 用的 UCT 公式**：

$$\text{UCT}(s) = Q(s) + c \sqrt{\frac{\ln N_{\text{parent}}(s)}{N(s)}}$$

其中：
- $Q(s) = \frac{q(s)}{N(s)}$：节点 $s$ 的平均 Q-value
- $N(s)$：节点 $s$ 的访问次数（visit count）
- $N_{\text{parent}}(s)$：$s$ 的父节点的访问次数
- $q(s)$：由 PPM 预测的 reward（后续 back-propagation 更新）
- $c$：exploration 常数，paper 中设为 2，偏重 exploration

这个公式第一项是 exploitation（选高 Q-value 的节点），第二项是 exploration（选访问次数少的节点，用 $\ln$ 软化避免过度偏好 unexplored）。常数 $c=2$ 偏大，说明 rStar-Math 在 SLM 能力较弱时，更需要 exploration 来发现 correct paths。

### 3.2 Code-augmented CoT Generation 流程

在每一步 $i$：
1. 收集当前 trajectory $x \oplus s_1 \oplus s_2 \oplus \ldots \oplus s_{i-1}$ 作为 state
2. Prompt policy model 生成 $n$ 个候选 $s_{i,0}, \ldots, s_{i,n-1}$（paper 中 $n=8$，round 4 增至 16）
3. 每个候选 $s_{i,j}$ 与前面所有步骤的 code 拼接，执行 Python
4. 只有成功 execute 的候选保留为 valid node
5. PPM 对 valid node 打分，赋 $q(s_i)$
6. 用 UCT 选最佳节点

这是一个非常 tight 的 verify loop——每一步都被 Python execution 验证，避免了 "correct answer by chance" 的污染。

### 3.3 Q-value Annotation 的两种模式

这是 rStar-Math 的精髓之一。Q-value 的可靠性直接决定 MCTS 的搜索质量。

#### Terminal-guided annotation（Round 1-2）

前两轮 PPM 还不可靠，用终局信号反向传播：

$$q(s_i)^k = q(s_i)^{k-1} + q(s_d)^k$$

其中：
- $q(s_i)^k$：第 $k$ 次 rollout 后 step $s_i$ 的 q-value
- $q(s_i)^{k-1}$：上一次 rollout 的累积 q-value
- $q(s_d)^k$：本次 rollout 终端节点的 reward
- 初始 $q(s_i)^0 = 0$

终端节点：
- $q(s_d) = 1$ 如果 final answer 正确
- $q(s_d) = -1$ 如果 final answer 错误

这就像 AlphaGo 的思路——通过多次对弈回顾每一步的贡献。频繁导向正确答案的 step，q-value 会累积变高；反之变低。但需要 extensive rollouts 才能让 q-value 收敛到可靠值。

#### PPM-augmented annotation（Round 3 起）

当 PPM 足够可靠后，PPM 直接提供 non-zero 初始 q-value：

$$q(s_i)^0 = \text{PPM}(x \oplus s_1 \oplus s_2 \oplus \ldots \oplus s_{i-1} \oplus s_i)$$

这个初始 q-value 仍然通过 Eq. 2 的 back-propagation 更新。好处是：
1. 不需要多次 rollout 才能得到 meaningful q-value
2. PPM 直接引导 policy model 生成更高质量的 step
3. terminal node 仍然用 ground truth label 赋分（更准确）

### 3.4 Process Preference Model (PPM) 训练

这是 paper 的最关键技术贡献。核心 insight：**Q-values 精度不够做 absolute scoring，但足以做 pairwise preference**。

#### Preference pair 构造

对每个 step：
- 选 Q-value 最高的 2 个候选作为 positive steps
- 选 Q-value 最低的 2 个候选作为 negative steps
- **Critical constraint**：positive steps 必须导向 correct final answer，negative steps 必须导向 incorrect answer
- 对 intermediate steps：positive 和 negative pair 共享相同的前序 steps（控制变量）
- 对 final answer step：relax 限制，选 2 个 correct trajectories（最高平均 Q-value）作 positive，2 个 incorrect trajectories（最低平均 Q-value）作 negative

#### Bradley-Terry pairwise ranking loss

$$\mathcal{L}_{\text{ppm}}(\theta) = -\frac{1}{2 \times 2} \mathbb{E}_{(x, y_i^{\text{pos}}, y_i^{\text{neg}} \in \mathbb{D})} \left[ \log \left( \sigma(r_\theta(x, y_i^{\text{pos}}) - r_\theta(x, y_i^{\text{neg}})) \right) \right]$$

其中：
- $r_\theta(x, y_i)$：PPM 的 scalar 输出（由 linear layer + tanh 组成，范围 $[-1, 1]$）
- $x$：问题
- $y_i$：从第 1 步到第 $i$ 步的 trajectory
- $y_i^{\text{pos}} = s_1 \oplus \ldots \oplus s_{i-1} \oplus s_i^{\text{pos}}$：positive trajectory
- $y_i^{\text{neg}} = s_1 \oplus \ldots \oplus s_{i-1} \oplus s_i^{\text{neg}}$：negative trajectory
- $\sigma$：sigmoid 函数
- $\frac{1}{2 \times 2}$：归一化因子（2 positive × 2 negative）

这个 loss 是经典 RLHF 的 pairwise ranking loss（Ouyang et al. 2022 InstructGPT）。关键区别在于：**不是用 absolute Q-value 做 regression target，而是用 Q-value 排序构造 preference pair**。这样 PPM 只需要学好"相对好坏"，不需要学"绝对分值"，鲁棒性大幅提升。

### 3.5 Architecture 细节

PPM 的架构：
- 初始化自 fine-tuned policy model
- 替换 next-token prediction head 为 scalar-value head
- Scalar-value head = linear layer + tanh function
- tanh 约束输出在 $[-1, 1]$ 范围内

这个设计让 PPM 和 policy model 共享 backbone，节省参数同时继承 policy model 的 math reasoning 能力。

---

## 4. Self-Evolution 的 4 轮详细拆解

### Round 1: Bootstrapping

- **Policy model**: DeepSeek-Coder-V2-Instruct (236B)——一个外部强模型作为起点
- **Rollouts**: 8 次（因为 236B 太贵）
- **Q-value**: Terminal-guided annotation
- **SFT data**: 选 top-2 trajectories（最高平均 Q-value 且 correct）
- **结果**: 训练出 policy SLM-r1，同时训练 PPM-r1（但 limited rollouts 导致 Q-value 不可靠，PPM-r1 效果有限）
- **MATH 准确率**: 58.8% → 69.6%（policy SLM-r1）
- **覆盖**: 60.17% 的 747k 问题被解决

### Round 2: Reliable PPM

- **Policy model**: 7B SLM-r1（换上自己的模型）
- **Rollouts**: 16 次（SLM 小，可以多 rollout）
- **Q-value**: 仍然 Terminal-guided，但 16 次 rollout 让 Q-value 精度大幅提升
- **结果**: 
  - PPM-r2 变得 reliable（Table 4: MATH 75.2 → 84.1）
  - policy SLM-r2 继续提升（Table 3: MATH 69.6 → 73.6）
- **覆盖**: 66.60%
- **关键**: 这是第一次有 reliable 的 reward model，为 Round 3 的 PPM-augmented MCTS 铺路

### Round 3: PPM-augmented MCTS

- **Policy model**: SLM-r2
- **Reward model**: PPM-r2（第一次在 MCTS 中使用 PPM 引导）
- **Q-value**: PPM-augmented annotation（Eq. 3）
- **效果**: 数据质量显著提升，覆盖更多 Olympiad-level 问题
  - MATH-level: 67.40% → 88.69%
  - Olympiad-level: 56.04% → 62.16%
- **结果**:
  - policy SLM-r3: MATH 75.8%
  - PPM-r3: MATH 85.2%

### Round 4: Solving Challenging Problems

- **Policy model**: SLM-r3 + PPM-r3
- **针对 hard problems**: 对 16 rollouts 没解决的问题，额外做 64 rollouts，必要时增至 128
- **Multiple MCTS expansions**: 不同 random seeds 多次扩展
- **结果**:
  - Olympiad-level 覆盖率: 62.16% → 80.58%
  - 总覆盖: 90.25%
  - policy SLM-r4: MATH 78.4%, AIME 26.7%
  - PPM-r4: MATH 87.0%, AIME 43.3%

### 何时停止？

剩余 ~10% 未解决问题，作者随机检查 20 个，发现 **19 个是 GPT-4 合成时 ground-truth label 错了**。所以未解决的不是模型能力问题，而是数据质量问题。这是一个有趣的发现——数据合成 pipeline 本身就是 noise source。

---

## 5. 实验结果深度分析

### Table 1: 主要 Benchmark 结果

| Task | rStar-Math (Qwen-7B) | o1-preview | o1-mini | QWQ-32B | GPT-4o | DeepSeek-V3 |
|------|---------------------|-----------|---------|---------|--------|-------------|
| MATH | **90.0** | 85.5 | 90.0 | 90.6 | 76.6 | 90.2 |
| AIME 2024 | **53.3** | 44.6 | 56.7 | 50.0 | 9.3 | 39.2 |
| Olympiad Bench | **65.6** | - | 65.3 | 61.2 | 43.3 | 55.4 |

关键观察：
1. 7B SLM + 7B PPM 在 MATH 上**超过 o1-preview +4.5%**，**匹配 o1-mini**
2. AIME 2024 上解决 8/15 问题，相当于 top 20% 美国高中生水平
3. 未解决的 8 个问题中有 geometry 题（需要 visual understanding，rStar-Math 目前不支持）

### Table 5: 完整对比（关键行）

**Qwen2.5-Math-7B 路径**：
- Base: MATH 58.8%, AIME 0.0%
- + Instruct: MATH 82.6%, AIME 6.0%
- + 72B ORM (Best-of-N): MATH 88.4%, AIME 26.7%
- **rStar-Math (7B SLM + 7B PPM, 8 trajectories)**: MATH 89.4%, AIME 50.0%
- **rStar-Math⁶⁴ (64 trajectories)**: MATH 90.0%, AIME 53.3%

注意 rStar-Math 用 7B PPM **超过了** Qwen 官方用 72B ORM 的 Best-of-N。这证明 step-level reward 比 outcome-level reward 更 powerful，即使 reward model 小 10 倍。

### Table 6: Self-evolution 效果

| Round | MATH | AIME | AMC | Olympiad Bench | College Math | GSM8K |
|-------|------|------|-----|---------------|-------------|-------|
| GPT-4o | 76.6 | 9.3 | 47.5 | 43.3 | 48.5 | 92.9 |
| Base 7B | 58.8 | 0.0 | 22.5 | 21.8 | 41.6 | 91.6 |
| Round 1 | 75.2 | 10.0 | 57.5 | 35.7 | 45.4 | 90.9 |
| **Round 2** | **86.6** | **43.3** | **75.0** | **59.4** | **55.6** | 94.0 |
| Round 3 | 87.0 | 46.7 | 80.0 | 61.6 | 56.5 | 94.2 |
| Round 4 | 89.4 | 50.0 | 87.5 | 65.3 | 59.0 | 95.0 |

**Round 2 是质变点**——AIME 从 10.0 跳到 43.3。这正是 reliable PPM-r2 介入的时刻。说明 System 2 reasoning 的 bottleneck 在 reward model，policy model 够用之后，reward model 决定上限。

### Table 7: Step-by-step Verified Trajectory vs Baselines

| Dataset | MATH | AIME | AMC | Olympiad Bench |
|---------|------|------|-----|---------------|
| MetaMath (GPT-4 distill) | 55.2 | 3.3 | 32.5 | 19.1 |
| NuminaMath-CoT | 69.6 | 10.0 | 50.0 | 37.2 |
| Random sample (SLM-r3) | 72.4 | 10.0 | 45.0 | 41.0 |
| Rejection sampling (SLM-r3) | 73.4 | 13.3 | 47.5 | 44.7 |
| **Step-by-step verified (ours)** | **78.4** | **26.7** | 47.5 | **47.1** |

关键发现：
1. PPM-augmented MCTS + code verification > 所有 baseline
2. 即使是 SLM-r3 的 **random sample** 也 comparable to GPT-4 distilled NuminaMath——说明 self-evolution 后的 SLM 已经很强
3. AIME 上提升最明显（10.0 → 26.7），说明 verification 对 hard problem 更关键

### Table 8: PPM vs ORM vs PQM

| RM | Inference | MATH | AIME | Olympiad Bench |
|----|----------|------|------|---------------|
| ORM | Best-of-N | 82.6 | 26.7 | 55.1 |
| PQM (Q-value as label, MSE loss) | MCTS | 88.2 | 46.7 | 62.9 |
| **PPM (preference learning)** | MCTS | **89.4** | **50.0** | **65.3** |

这个 ablation 非常 clean：
- ORM 只能 end-of-trajectory 评分 → sparse signal
- PQM 用 Q-value 做 regression target → noisy label 限制上限
- PPM 用 Q-value 做 preference ranking → robust，达到 o1-mini 可比水平

### Figure 5: Reward Model 决定上限

这个图非常有 insight：不同大小的 policy model（1.5B/3.8B/7B）Pass@1 差异很大，但配上同一个 PPM 做 System 2 reasoning 后，最终 accuracy 收敛。说明 **PPM 是 reasoning boundary 的主要决定者**。

---

## 6. 两个有趣的 Findings

### 6.1 Intrinsic Self-Reflection 的涌现

这是最 surprising 的发现。rStar-Math 没有专门训练 self-reflection，但在 Figure 4 的例子中，模型在前 3 步走错路（用 SymPy 形式化方程但会导向错误答案），第 4 步突然 backtrack，换用更简单的方法重新解，最终正确。

这暗示 **System 2 deep thinking（MCTS + PPM）本身就能 foster self-reflection**，不需要额外的 self-correction training data。这可能是因为 PPM 给低分 step 的 signal 模拟了"反思"的过程——policy model 学会了识别自己低质量 step 并换路径。

### 6.2 PPM 识别定理应用步骤

在解 challenging math problem 时，PPM 对关键定理应用步骤（Fermat's Little Theorem, Vieta's formulas, AM-GM inequality, Pythagorean theorem, Shoelace Theorem）给出高 reward score。Paper Appendix A.2 有具体例子，比如 Vieta's formulas + AM-GM 的例子中，相关步骤的 PPM Score 高达 0.9989 和 0.9999。

这说明 PPM 不仅学到了"这步对不对"，还学到了"这步是不是关键 insight"——这是非常 semantic 的理解。

---

## 7. 训练细节

### 数据收集
- 747k math word problems，主要来自 NuminaMath（只用 competition-level）和 MetaMath
- 用 GPT-4 合成新问题（基于 7.5k MATH train + 3.6k AMC-AIME seed）
- 过滤：GPT-4 生成 10 个 solution，至少 3 个一致才保留

### SFT 训练
- 每轮从 base model 重新 fine-tune（不是 incremental）
- 2 epochs，sequence length 4096，batch size 128
- AdamW optimizer，linear LR scheduler
- Qwen models: initial LR 7e-6
- Phi3-mini: cosine scheduler, initial LR 5e-6

### PPM 训练
- 1 epoch，batch size 512，initial LR 7e-6
- 过滤掉全对或全错的问题（没有 preference signal）
- 只保留 mixed outcome 的问题

### Inference 设置
- 每个 step 生成 32 个 candidate node
- PPM 对每个 node 打分
- MCTS 只做 4 rollouts per step（因为 PPM 已经提供 dense signal）
- 选 PPM score 最高的 trajectory 作为 final answer

### 计算成本
- Round 1: 10 nodes × 8×80GB H100，2 周
- Round 2-4: 15 nodes × 4×40GB A100，每轮 3 天
- Round 4 增至 64 rollouts：1 周
- Inference token cost: MATH 5453 tokens/trajectory, AIME 15693 tokens/trajectory

---

## 8. 为什么这个 work 成立？我的直觉解读

1. **Code execution 是硬 verification**：自然语言 CoT 的 correctness 难判，但 Python 代码能跑通是客观信号。这过滤掉了大量"看似对实则错"的 step。

2. **MCTS 把难题分解成简单题**：SLM 一次性解 Olympiad 题很难，但生成单步容易。MCTS 把 search space 分解，让 SLM 在每步上发挥。

3. **Q-value 做 preference 而非 regression**：这是最关键的 insight。Q-value 是 noisy 的 absolute score，但 reliable 的 relative ordering。Bradley-Terry loss 只需要 ordering 正确，不需要 absolute value 精确。这避开了 noisy label 的痛点。

4. **Self-evolution 是 bootstrap**：Round 1 用外部强模型 kick-start，之后 SLM 自驱动。每轮都让 data quality 提升，data quality 提升又让 model 变强，形成正反馈。

5. **Reward model 是 bottleneck**：Figure 5 显示 policy model 大小不重要，PPM 决定上限。这和 OpenAI o1 的发现一致——reasoning 的关键是 process supervision。

---

## 9. 局限与未来方向

1. **不支持 geometry**：未解决的 AIME 题多为 geometry，需要 visual understanding
2. **Theorem proving 还不完善**：虽然 Appendix A.2 展示了 Fermat Little Theorem 的证明，但还不是大规模 theorem proving
3. **GPT-4 合成数据有 noise**：Round 4 后发现 19/20 未解决问题是 GPT-4 label 错了
4. **Inference cost 高**：AIME 一个 trajectory 15693 tokens，64 trajectories 意味着百万级 token

---

## 10. 与相关工作的对比

### vs GPT-4 distillation (MetaMath, NuminaMath)
- rStar-Math 不受 teacher model 能力上限限制
- Table 7 显示 SLM self-generation 已经 comparable to GPT-4 distill

### vs Best-of-N + ORM
- rStar-Math 用 step-level reward（PPM）vs end-level reward（ORM）
- Table 8: PPM 在所有 benchmark 上超 ORM
- 即使 ORM 是 72B，7B PPM 仍然更好

### vs Q-value based PRM (Math-Shepherd, AlphaMath)
- 这些方法直接用 Q-value 做 MSE regression
- Table 8 PQM 行显示这有上限（MATH 88.2 vs PPM 89.4，Olympiad 62.9 vs 65.3）
- rStar-Math 的 preference learning 更 robust

### vs OpenAI o1
- rStar-Math 用 7B 模型达到 o1-preview 水平，部分 benchmark 超 o1-mini
- 但 o1 是单模型 inference，rStar-Math 是 MCTS + policy + reward 的 multi-component 系统
- o1 的 self-reflection 是 trained-in 的，rStar-Math 是 emergent 的

---

## 11. 代码与数据

Paper 承诺开源：https://github.com/microsoft/rStar

---

## 12. 我的整体评价

这篇 paper 的工程完成度很高，几个 design choice 都很 solid：
1. Code-augmented CoT 解决 intermediate step verification
2. PPM 用 preference 代替 regression 解决 noisy Q-value
3. 4-round self-evolution 是合理的 bootstrap loop
4. 每轮的 ablation 都有数据支撑

最让我 impressed 的是 **PPM 的设计哲学**——承认 Q-value 不精确，但利用其相对排序信息。这比硬要 model 学 absolute score 聪明得多。

self-reflection 的 emergent behavior 也很 intriguing，暗示 System 2 search 本身就蕴含了反思能力，不需要 explicit training。

如果让我挑一个 most important takeaway，那就是：**reward model 的 design 比 policy model 的 scale 更重要**。Figure 5 是这篇 paper 最有说服力的图。

参考链接：
- Paper: https://arxiv.org/abs/2501.04519
- GitHub: https://github.com/microsoft/rStar
- 相关 rStar (前作): https://arxiv.org/abs/2408.06195
- OpenAI o1: https://openai.com/o1/
- Let's Verify Step by Step (PRM800k): https://arxiv.org/abs/2305.20050
- Math-Shepherd: https://arxiv.org/abs/2406.06592
- AlphaMath Almost Zero: https://arxiv.org/abs/2405.03553

Andrej，这篇 paper 的核心 insight 其实和你的 micrograd/zero-to-hero 教学哲学很像——**理解 gradient signal 的本质比堆参数更重要**。PPM 的 preference learning 就是对 reward signal 本质的重新理解：不要追求绝对精度，只要相对排序对，learning 就能 work。这种 "first principles" 思考方式在 System 2 reasoning 的 design 中特别重要。
