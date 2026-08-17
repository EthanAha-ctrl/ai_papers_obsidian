---
source_pdf: Scaling of Search and Learning A Roadmap to Reproduce o1.pdf
paper_sha256: bd8d5e01cf20c8b1a7438131753e2789e72e2cb625c7b2c9ee341d73a2338d33
processed_at: '2026-08-12T03:36:38-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

---

## 一句话总结

OpenAI o1 的核心秘密就是两件事：**搜索** 和 **学习**。这两件事能随着算力增长持续变强，而其他花里胡哨的技巧都有天花板。

---

## 为什么这么说

Sutton 2019 年写过一篇很短但很狠的文章叫 "Bitter Lesson"。他回顾了 AI 几十年历史，发现一个规律：**凡是靠人类知识堆出来的方法，最后都被靠算力堆的通用方法打败了**。

国际象棋：人类花了几十年手工编码大师策略，最后被 AlphaGo 的纯搜索+学习碾压。
机器翻译：人类花了几十年搞语法规则，最后被纯数据驱动的神经网络干掉。

o1 就是这个 lesson 的最新验证。它没发明新架构，没用 neuro-symbolic 的花活，就是把 search 和 learning scale 到极致。

---

## 四个组件，逐个说

### 1. Policy Initialization —— 给模型装上"人会怎么想"的能力

一个从零开始用 RL 训练的 LLM 是基本没法用的，因为 action space 太大了——vocabulary 几万个 token，随便组合就是天文数字。所以得先给模型一个还不错的起点。

怎么给？三步走：

**Pre-training**：读一堆互联网文本，学会语言本身、世界知识、基础推理。这是打底子。

**Instruction fine-tuning**：让模型从"接龙"变成"听话"。你问它问题，它知道该回答而不是继续接龙。

**注入 human-like reasoning behaviors**：这是关键。o1 的 CoT 不是简单的"让我一步一步想"，它有六种行为模式：

- 先把问题分析一遍（"所以用户想要一个 bash 脚本..."）
- 把大问题拆成小步骤（"第一步：捕获输入，第二步：去空格..."）
- 执行步骤（"让我先写骨架..."）
- 遇到死路换方案（"选项一：... 选项二：..."）
- 自己检查（"让我验证一下这个映射..."）
- 发现错误自己改（"等等，正确的公式应该是..."）

这六种行为本质上是**探索 solution space 的工具箱**。没有这些工具，模型只能在 solution space 里瞎走；有了这些工具，它能系统性地搜索。

怎么注入？两种路子：要么用 SFT 训练（在带这些行为的轨迹数据上做监督学习），要么用 prompt 触发（模型其实已经隐含这些能力，prompt 能激活）。

**核心 tradeoff**：你想让模型采样效率高，就得让 distribution 尖锐（集中到几个好策略上）；但你想让模型探索到更好的策略，就得保持多样性。AlphaGo 从人类棋谱学，AlphaGo Zero 从零开始 self-play，后者反而更强——因为人类数据限制了探索。

---

### 2. Reward Design —— 告诉模型什么好什么坏

模型做了一堆推理步骤，谁来打分？

**最简单的方案：Outcome Reward**。数学题做对了给 1 分，做错了给 0 分。问题：中间步骤可能全错但碰巧蒙对答案，或者中间步骤大部分对但最后一步算错——模型学不到中间过程的质量。

**更好的方案：Process Reward**。每一步都打分。问题是标注成本极高——Lightman 的 PRM 数据集需要人类标注员逐步判断每一步对不对。

**三种 reward 来源**：

**来自真实环境**：代码能跑通就是好，跑不通就是坏。编译器、单元测试、数学验证器，这些都是免费的、准确的 reward。但测试时通常没有这些环境。

**模拟环境**：训练一个 verifier 模型来预测 reward。问题：policy 变了之后，verifier 训练时的数据分布跟现在不匹配了，这就叫 distribution shift。这是 o1 复现的核心难题。

**AI 当裁判**：用 GPT-4 这种强模型来打分。好处是不依赖 policy，不存在 distribution shift。但贵，而且 GPT-4 自己也有错。

**Reward Shaping 的关键公式**：

$$F(s_t, a_t) = r(s_t, a_t) + \gamma \phi(s_{t+1}) - \phi(s_t)$$

人话翻译：你可以在原始 reward 上加一个"势能差"项 $\phi(s_{t+1}) - \phi(s_t)$，只要这个势能函数只依赖状态，加完之后最优策略不变。这个定理的意义在于：你可以把稀疏的 outcome reward 通过 shaping 变成 dense 的 process reward，而且不会改变模型最终应该学到的东西。

有意思的发现：DPO 其实隐含了这个 shaping——它自带一个 baseline 项，所以不会有 REINFORCE 那种高方差问题。

**未来的方向：World Model**。不仅预测 reward，还预测下一个状态。这样模型可以在脑子里"模拟"未来，而不需要真的跟环境交互。MuZero 就是这么干的——在脑子里模拟围棋的未来几步。

---

### 3. Search —— 让模型多想几个方案再选

这是 paper 的核心。Search 分训练时和测试时两个阶段。

#### 训练时：生成高质量训练数据

训练时可以并行采样很多方案，可以用真实环境验证（跑代码、查数学答案），所以适合用 **tree search + external guidance**。

为什么不用简单采样？因为简单采样是按模型当前概率分布采的，采到的大多是"还行"的方案。Tree search 能找到比模型当前水平更好的方案——这些更好的方案就是高质量的训练数据。

最典型的例子是 Best-of-N：采 N 个方案，用 verifier 选最好的。Brown et al. 2024 发现小模型 + 大量 BoN sampling 在 MATH 上能接近 100% pass@1——说明模型"知道"正确答案，只是单次采样采不到。

更高级的是 MCTS：四步循环——Selection（用 PUCT 公式选最有潜力的 child）、Expansion（展开新 node）、Evaluation（用 value network 或 rollout 估 leaf 的 value）、Backpropagation（沿路径更新 Q 值）。MCTS 的好处是平衡探索和利用，坏处是计算量大、并行性差。

#### 测试时：让模型在推理时多想

测试时不能访问真实环境，只能靠模型自己。所以适合 **sequential revisions + internal guidance**。

Sequential revisions 就是：生成一个答案，自己反思，改一改，再反思，再改。o1 blog 里的 CoT 风格就是这个——"让我想想... 不对，应该是... 等等，让我换个思路..."。

**为什么测试时不用 tree search？** 两个原因：

第一，长推理的 tree search 开销太大。如果 CoT 有 1000 步，tree search 的分支组合是指数爆炸的。

第二，proxy reward model 在测试时大规模搜索会导致 inverse scaling。Gao et al. 2023 发现：reward model 训练在旧 policy 数据上，新 policy 跑出来的轨迹跟训练数据分布不匹配，reward model 给的分越来越不靠谱，搜索越多反而越差。

所以 o1 测试时只能靠模型自己内化的能力——self-evaluation、self-correction 这些在 policy initialization 阶段就注入的行为。

#### Inverse Scaling 是最大的坑

这是复现 o1 的核心 obstacle。你训练了一个 reward model，policy 改进了，但 reward model 没跟着更新，于是 reward model 在新 policy 上的判断越来越不准。搜索规模越大，这个 mismatch 越严重，反而可能降低性能。

Gao et al. 的发现：proxy reward 跟 true reward 的 gap 随 KL 距离呈二次增长。这意味着你不能无限 scale test-time search。

OpenAI 怎么解决的？paper 只能推测——要么用很大的、泛化能力很强的 reward model，要么迭代更新 reward model，要么测试时主要靠 internal guidance 而不是 external reward model。

---

### 4. Learning —— 用搜索产生的数据改进模型

搜索产生了一堆 state-action pairs，有些好有些坏。怎么用这些数据？

**两条路**：

**Policy Gradient（PPO、DPO 等）**：用好和坏的样本都要。好的样本增强，坏的样本抑制。数据利用率高，但训练复杂。

**Behavior Cloning（SFT）**：只用最好的样本，直接监督学习。简单高效，但浪费了负样本的信息。

**公式直觉**：

REINFORCE 的梯度是 $\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$。人话：如果某个 action 的总回报 $G_t$ 是正的，就增加这个 action 的概率；如果是负的，就降低。问题是 $G_t$ 的方差极大——有时候你随机走运拿了大 reward，有时候随机倒霉拿了小 reward，梯度估计抖得厉害。

PPO 加了 clipping 限制每次更新幅度，加了 value function 做 baseline 来降方差。但需要 4 个模型同时跑，显存爆炸。

DPO 更聪明——直接把 reward optimization 转成 policy optimization，不需要显式的 reward model 和 value model。只要偏好数据（哪个答案比哪个好），就能训。而且 DPO 隐含了 reward shaping，自带 baseline，方差低。

**paper 推测 o1 的学习路线**：

先用 behavior cloning 做 warm-up（简单高效，快速起步），等 BC 的收益停滞了，再切换到 PPO 或 DPO（利用负样本，继续提升）。这跟 Llama 2、Llama 3 的后训练 pipeline 一致。

**Search + Learning 的闭环**：

这是整个 paper 最核心的点。搜索找到更好的方案 → 学习把这些方案内化 → 模型变强 → 搜索能找到更好的方案 → 继续学习... 这是一个正反馈循环。AlphaGo Zero 就是这么干的，最后超越了所有人类棋手。

---

## 三个最重要的 Intuition

### 1. Train-time 和 Test-time 的 Asymmetry

训练时：资源充足，可以并行，可以访问真实环境，用 tree search + external reward。
测试时：资源受限，单线程，没有真实环境，用 sequential revisions + internal capability。

这个 asymmetry 的根源是 inverse scaling。proxy reward model 在测试时大规模搜索会失效，所以测试时只能靠模型自己内化的能力。这意味着 policy initialization 阶段注入的那些 reasoning behaviors 至关重要——它们是模型在测试时唯一能依赖的东西。

### 2. Search-Learn 闭环是 superhuman 的来源

人类数据有天花板——人类的推理能力有限，蒸馏出来的模型最多跟人类一样强。但 search + learning 的闭环可以超越人类：搜索找到人类没想过的方案，学习内化这些方案，下一轮搜索找到更牛的方案。AlphaGo 的"第 37 手"就是这么来的——人类几千年没人这么下过，但 self-play 发现了这个妙手。

o1 的 RL 路线本质上就是把这个范式从围棋搬到语言推理。

### 3. RL for LLM 的 Scaling Law 还是空白

Pre-training 有 Chinchilla law，告诉你算力怎么分配最优。但 RL for LLM 的 scaling law 还没人建立——model size、search budget、learning iterations 之间怎么分配算力最优？这是复现 o1 的理论 obstacle，也是下一个研究热点。

---

## 最让我兴奋的联想

World Model + Search + Learning 的三角闭环。

o1 是 OpenAI 五级 AGI 路线的第二级（Reasoner）。下一级是 Agent——能在真实环境里行动。但真实环境不可逆——你点了"购买"按钮，钱就花了，不能撤销。怎么在不可逆环境里搜索？

答案：World Model。在脑子里模拟未来，在模拟里搜索最优策略，然后在真实环境里执行。

MuZero 已经证明了这条路——在脑子里模拟围棋的未来几步，搜索最优走法。如果把这个能力跟 o1 的 RL pipeline 结合，可能就是从 Reasoner 走向 Agent 的关键技术。

更深一层：如果 thought 不用 text token 表示，而用 continuous latent vector 呢？text 是很低效的 thought representation——你需要生成几百个 token 来表达一个推理步骤。如果直接在 latent space 里做 search，效率可能高几个数量级。这可能就是下一代 o1 的方向。

---

# Scaling of Search and Learning: A Roadmap to Reproduce o1 深度解析

这篇 paper 由 Fudan University 与 Shanghai AI Laboratory 联合发表，从 reinforcement learning (RL) 的视角系统性地拆解了 OpenAI o1 的可能技术路线图。核心论点引用了 Sutton 的 "Bitter Lesson"——**search 和 learning 是能够持续 scale 的两大 general purpose methods**。下面我逐层拆解。

参考链接：
- Paper: https://arxiv.org/abs/2412.06592
- OpenAI o1 blog: https://openai.com/index/learning-to-reason-with-llms/
- Sutton's Bitter Lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html

---

## 1. 整体框架：四支柱架构

Roadmap 由四个相互依赖的 components 构成：

```
Policy Initialization  →  Reward Design  →  Search  →  Learning
      (准备)              (准备)          (生成)      (改进)
                                                    ↓
                                                 回到 Policy
                                                 (迭代)
```

**关键 intuition**：o1 的本质是 "scaling both training and inference computation"。Pre-training 与 SFT 给模型一个能产生 human-like reasoning 行为的初始 policy；reward design 提供 dense signal；search 在 train/test 两个阶段生成高质量 trajectories；learning 用这些 trajectories 改进 policy。这构成一个 **search-learn 迭代闭环**，类似于 AlphaGo Zero 的 self-play 范式。

Reference: AlphaGo Zero (Silver et al., 2017) — https://www.nature.com/articles/nature24270

---

## 2. Policy Initialization（Section 3）

### 2.1 三层 action granularity

LLM 的 policy π(a|s) 可以在三个粒度上定义：
- **Token-level**：动作空间是 vocabulary（约 30k-100k），最 fine-grained
- **Step-level**：动作是一个 reasoning step（自然以 newline 分隔），中间粒度
- **Solution-level**：整个 solution 作为一个 action，最 coarse

**intuition**：粒度越细，搜索树越深、越窄；粒度越粗，搜索树越浅、越宽。MCTS 在不同粒度下的复杂度差异显著——token-level tree depth 极大但宽度受 vocab 限制，step-level 是平衡点。

### 2.2 初始化三阶段

1. **Pre-training**：建立 language understanding、world knowledge、basic reasoning
2. **Instruction Fine-Tuning**：从 next-token prediction 转向 task-oriented behavior
3. **Human-like Reasoning Behaviors 注入**：通过 SFT 或 prompt engineering

### 2.3 六种 human-like reasoning behaviors（Table 1）

paper 从 o1 blog 中归纳出六类行为：

| Behavior | 例子 | 触发场景 |
|---------|------|---------|
| Problem Analysis | "So the user is requesting a bash script..." | 任务开始 |
| Task Decomposition | "Implementation Steps: 1. Capture input..." | 复杂任务 |
| Task Completion | "Let me try coding the bash script step by step" | 执行 |
| Alternative Proposal | "Option 1: ... Option 2: ..." | 遇到障碍 |
| Self-Evaluation | "Let's check... Let's test..." | 完成后 |
| Self-Correction | "Wait, the correct formula is..." | 发现错误 |

**intuition**：这六种行为本质上是对 solution space 的 **systematic exploration 工具**。Self-evaluation 对应 DG-gap (Generator-Discriminator Gap)——评估比生成容易。Self-correction 打破了 autoregressive 模型无法回退的根本限制。Alternative proposal 是 divergence，self-evaluation 是 convergence。

DG-gap reference: Leike, 2022 — https://substack.com/home/post/p-51216719

### 2.4 Policy Initialization 的核心挑战

**Sampling efficiency vs diversity tradeoff**：
- 从 human demo 学习 → sharp distribution → 高效 sampling
- 过度收敛 → 限制探索 → 错失 superhuman strategies

paper 用 AlphaGo vs AlphaGo Zero 作类比：AlphaGo 从人类棋谱初始化，AlphaGo Zero 完全从零开始 self-play，后者反而更强，因为 human data 限制了探索。

### 2.5 推测：o1 的 policy initialization 可能方案

- **Long-text generation**：训练支持超长 CoT 输出，参考 LongWriter (https://arxiv.org/abs/2408.07055)、Self-Lengthen (https://arxiv.org/abs/2410.23933)
- **逻辑编排能力**：通过代码与结构化逻辑数据强化
- **Self-reflection**：通过 SFT 注入，难以通过 PEFT 学习

---

## 3. Reward Design（Section 4）

### 3.1 Outcome Reward vs Process Reward（Figure 5）

- **Outcome Reward (ORM)**：只对最终结果给分。Sparse、容易学到错误中间步骤。
- **Process Reward (PRM)**：对每一步给分。Dense、可监督中间过程，但需要昂贵标注。

Lightman et al. 2024 的 "Let's Verify Step by Step" 是 PRM 里程碑工作：https://openreview.net/forum?id=v8L0pN6EOi

### 3.2 三类 reward 来源

#### 3.2.1 From Environment
- **Realistic environment**：compiler feedback (StepCoder: https://arxiv.org/abs/2402.01391)、unit tests、math verification
- **Simulating environment**：训练 verifier 模型预测 reward，存在 distribution shift 问题
- **AI Judgment**：用 GPT-4 当 judge (LLM-as-a-Judge: https://arxiv.org/abs/2411.15594)，不依赖 policy，避免 reward over-optimization

#### 3.2.2 Reward Modeling from Data
- **Preference data → RLHF**：基于 Bradley-Terry model
  $$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$
  Bradley-Terry reference: https://www.jstor.org/stable/2334029
- **Expert data → Inverse RL (IRL)**：从 expert trajectories 反推 reward function，参考 Garg et al. IQ-Learn (https://proceedings.neurips.cc/paper/2021/hash/210f760a89db30aa72ca258a3483cc7f-Abstract.html)。IRL 在 LLM 上尚未大规模验证，可能是 o1 的潜在技术。

#### 3.2.3 Reward Shaping（公式 1）

Potential-based reward shaping 的核心定理：

$$F(s_t, a_t) = r(s_t, a_t) + \gamma \phi(s_{t+1}) - \phi(s_t)$$

**变量解释**：
- $F(s_t, a_t)$：shaped 后的 reward function
- $r(s_t, a_t)$：原始 reward function
- $\gamma \in [0,1]$：discount factor，控制未来 reward 的权重
- $\phi(s)$：potential function，只依赖 state
- $s_t$：时刻 t 的 state，$s_{t+1}$：执行 action $a_t$ 后的下一 state

**intuition**：这个 shaping 保证了 optimal policy 不变。$\phi(s_{t+1}) - \phi(s_t)$ 本质上是 "potential 的差分"，类似于物理学中的势能差，只在状态转换时释放 reward，不改变最优策略。

重要推论：**DPO 隐含了 potential-based reward shaping**（Rafailov et al., 2024 — https://arxiv.org/abs/2404.12358）。

### 3.3 Reward Design 核心挑战

1. **Distribution shift**：policy 更新后，旧 reward model 失效。Gao et al. 2023 的 scaling laws for reward over-optimization 显示 proxy reward 与 true reward 的 gap 随 KL 距离呈二次增长：https://proceedings.mlr.press/v202/gao23h.html

2. **Fine-grained reward 设计**：language 的 step 定义模糊，token-level action space 指数爆炸。

3. **Data selection for complex tasks**：Wen et al. 2024 警告对于 code/math 任务，使用 preference feedback 反而可能降低性能：https://arxiv.org/abs/2409.12822

### 3.4 Generalization: World Model

paper 提到 future direction：构建 general reward signal 需要 **World Model**。World model 不仅预测 reward，还预测 next state（或 next state 的 representation）。参考：
- Ha & Schmidhuber, World Models (https://arxiv.org/abs/1803.10122)
- MuZero (https://www.nature.com/articles/s41586-020-03051-4)
- Sora as world simulator (https://openai.com/research/video-generation-models-as-world-simulators)

---

## 4. Search（Section 5）

这是 paper 最核心的部分。Search 分两个阶段：**train-time search** 和 **test-time search**。

### 4.1 Search Guidance 两大类（Figure 6）

#### Internal Guidance（不依赖外部环境）
- **Model Uncertainty**：Self-consistency (https://openreview.net/forum?id=1PL1NIMMrw) 用 majority voting 选最低不确定度的答案
- **Semantic Entropy**：Kuhn et al. 用 NLI 模型做语义聚类（https://openreview.net/forum?id=VD-AYtP0dve）
- **Self-evaluation**：LLM-as-a-Judge，基于 DG-gap 假设

#### External Guidance（依赖环境/reward model）
- **Environmental Feedback**：compiler、unit test、math verifier
- **Heuristic Rules**：A* search 的启发函数

#### Value Function（公式 2）

$$v_\pi(s) \doteq \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s\right]$$

**变量解释**：
- $v_\pi(s)$：在 policy $\pi$ 下，state $s$ 的 value（期望累积 reward）
- $\mathbb{E}_\pi$：在 policy $\pi$ 下的期望
- $G_t$：从时刻 t 开始的 return（累积 discounted reward）
- $\gamma \in [0,1]$：discount factor
- $R_{t+k+1}$：时刻 $t+k+1$ 的 reward
- $\mathcal{S}$：所有可能的 state 集合

**intuition**：Value function = "从这个 state 出发，按 policy $\pi$ 走，未来能拿到多少 reward"。它是 internal+external guidance 的结合体——需要 reward signal（external）但由模型估计（internal）。

### 4.2 Tree Search 策略

#### 4.2.1 Best-of-N (BoN)
最简单的 tree search（depth-1）。生成 N 个 candidate，用 verifier 选最好。

**Scaling law**（Brown et al. 2024 — https://arxiv.org/abs/2407.21787）：
- pass@k 精度随 k 增长呈 power law 改善
- 小模型 + 大量 sampling 可在 MATH 上接近 100% pass@1

**Variants**：
- Speculative Rejection (https://arxiv.org/abs/2410.20290)：early discard 低分 partial sequence
- BOND (https://arxiv.org/abs/2407.14622)：用 Jeffreys divergence 蒸馏 BoN 分布
- vBoN (https://arxiv.org/abs/2407.06057)：variational BoN with PPO

#### 4.2.2 Beam Search
经典 token-level tree search。LLM 时代的改进：
- **TreeBoN** (https://arxiv.org/abs/2410.16033)：用 DPO 的 token-level reward 替代 probability
- **OVM** (https://aclanthology.org/2024.findings-naacl.55)：从 outcome supervision 训练 value model
- **Self-evaluation guided beam search** (https://papers.nips.cc/paper_files/paper/2023/hash/81fde95c4dc79188a69ce5b24d63010b-Abstract-Conference.html)：用 policy model 自评估替代 token probability

#### 4.2.3 Monte Carlo Tree Search (MCTS)

MCTS 是 AlphaGo 的核心算法。每次 simulation 包含四个阶段：

**Selection**：用 PUCT (Predictor + UCB applied to Trees) 公式选择 child：
$$a^* = \arg\max_a \left[ Q(s, a) + c \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]$$

其中：
- $Q(s, a)$：state-action pair 的 value 估计
- $P(s, a)$：prior probability（来自 policy network）
- $N(s)$：state s 的访问次数
- $N(s, a)$：state s 下 action a 的访问次数
- $c$：exploration constant

**intuition**：PUCT 平衡 exploitation（高 Q 值）和 exploration（低访问次数 + 高 prior）。

**Expansion**：在 leaf node 展开 child nodes，用 policy 输出作为 prior。
**Evaluation**：用 rollout policy 或 value network 评估 leaf value。
**Backpropagation**：沿 path 更新 Q 值和访问次数。

**LLM-MCTS 的三种粒度**：
- **Token-level** (https://arxiv.org/abs/2309.15028)：深度大、效率低
- **Step-level** (RAP: https://aclanthology.org/2023.emnlp-main.507)：自然单位，平衡选择
- **Solution-level** (MCTSr: https://arxiv.org/abs/2406.07394)：把整个 solution 当 node，self-refine 当 action

#### 4.2.4 其他 Tree Search
- **Tree of Thoughts (ToT)** (https://arxiv.org/abs/2305.10601)：DFS/BFS
- **A*-inspired best-first search** (https://arxiv.org/abs/2407.01476)：多模态 LLM 评估 node

### 4.3 Sequential Revisions

与 tree search 相反，基于上一次答案迭代改进。

- **SELF-REFINE** (https://papers.nips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)：同 LLM 给 feedback + refine
- **Reflexion** (https://papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html)：基于外部环境反馈
- **Self-Debug** (https://openreview.net/forum?id=KuPixIqPiq)：代码 execution feedback

**争议**：Huang et al. 2024a (https://openreview.net/forum?id=IkmD3fKBPQ) 认为 LLM 无法在没有 external feedback 时 self-correct。但 DG-gap 假设支持 self-correction 可行。Chen et al. 2024d 的实证研究：**只有当 discriminator accuracy ≥ 90% 时 sequential revisions 才优于 BoN** (https://aclanthology.org/2024.acl-long.738)。

### 4.4 Search 在 o1 中的角色推测

**Train-time search**：
- 倾向 **tree search + external guidance**
- 可并行采样大量 candidate
- 可访问真实环境（执行代码、验证数学）
- 类似 AlphaGo Zero 的 MCTS + behavior cloning 闭环

**Test-time search**：
- 倾向 **sequential revisions + internal guidance**
- o1 blog 的 reasoning 风格更像 sequential revisions
- Tree search 在长推理过程中开销过大
- Test-time 无法依赖真实环境
- Proxy reward model 会导致 over-optimization（Gao et al. 2023）

### 4.5 Inverse Scaling 问题

关键 warning：Gao et al. 2023 和 Stroebl et al. 2024 (https://arxiv.org/abs/2411.17501) 发现 **scaling best-of-n search 可能降低性能**——因为 reward model 训练在旧 policy 分布上，无法泛化到新 policy。这是 o1 复现的核心 obstacle。

---

## 5. Learning（Section 6）

### 5.1 数据集区分（Figure 8）

- $D_{\text{search}}$：search 产生的所有 state-action pairs（含负样本）
- $D_{\text{expert}} \subset D_{\text{search}}$：最高 reward 的 state-action pairs

**Policy gradient 用 $D_{\text{search}}$，behavior cloning 用 $D_{\text{expert}}$**。

### 5.2 Policy Gradient 方法

#### 5.2.1 REINFORCE（公式 4）

$$\nabla_\theta J(\theta) = \frac{1}{|D_{\text{search}}|} \sum_{(s_t, a_t) \in D_{\text{search}}} \left[ G_t \nabla_\theta \log \pi_\theta(a_t | s_t) \right]$$

**变量解释**：
- $\theta$：policy network 参数
- $J(\theta)$：期望累积 reward
- $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$：discounted return
- $\pi_\theta(a_t | s_t)$：在 state $s_t$ 选择 action $a_t$ 的概率
- $D_{\text{search}}$：search 产生的 dataset

**intuition**：high return $G_t$ 的 action → 增加 $\log \pi_\theta(a_t|s_t)$ 的梯度。本质是 "trial and error 强化"。

**缺点**：$G_t$ 的 variance 极高。

#### 5.2.2 Actor-Critic

用 advantage function 替代 $G_t$：
$$A(s_t, a_t) = R_{t+1} + \gamma V_{\pi_\theta}(s_{t+1}) - V(s_t)$$

$V(s)$ 是 value function，由独立 critic network 学习。Advantage = "实际 return - 预期 return"，variance 显著降低。

#### 5.2.3 PPO (https://arxiv.org/abs/1707.06347)

PPO 用 clipping 限制 policy 更新幅度：

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 是 probability ratio。

**PPO 在 LLM 中的实现复杂度**：需要 4 个模型——policy $\pi_\theta$、reference policy $\pi_{\text{ref}}$、reward model、value model。

**改进**：
- **GRPO** (DeepSeekMath, https://arxiv.org/abs/2402.03300)：用 Monte Carlo 估计替代 value model
- **ReMax** (https://openreview.net/forum?id=Stn8hXkpe6)：用 greedy decoding return 作为 baseline

PPO 实现细节参考：37 implementation details (https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

#### 5.2.4 DPO（公式 5）

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{search}}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

**变量解释**：
- $x$：question/prompt
- $y_w$：preferred (winning) response
- $y_l$：dispreferred (losing) response
- $\pi_\theta$：current policy
- $\pi_{\text{ref}}$：reference policy（通常是 SFT model）
- $\beta$：temperature parameter，控制偏离 reference 的强度
- $\sigma$：sigmoid function

**intuition**：DPO 把 reward optimization 转化为 policy optimization。Rafailov et al. 2024 证明 DPO 是 inverse Q-learning，DPO logits 就是 Q function。

**DPO 的 reward shaping**（公式 6）：

$$f(r, \pi_{\text{ref}}, \beta)(x, y) = r(x, y) - \beta \log \sum_y \pi_{\text{ref}}(y|x) \exp\left(-\frac{r(x, y)}{\beta}\right)$$

第二项是 baseline，解释了 DPO 为什么没有 REINFORCE 的高 variance 问题。

### 5.3 Behavior Cloning（公式 7）

$$\min_\theta - \frac{1}{|D_{\text{expert}}|} \sum_{(s, a) \in D_{\text{expert}}} \left[ \log \pi_\theta(a | s) \right]$$

即 cross-entropy loss，只在最高 reward 的 trajectories 上训练。

**Expert Iteration** = Search + Behavior Cloning 的迭代。经典案例：
- **STaR** (https://papers.nips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html)：reject sampling + SFT
- **V-STaR** (https://arxiv.org/abs/2402.06457)：训练 verifier 用 DPO
- **AlphaGo Zero**：MCTS + behavior cloning

### 5.4 四种方法对比（Table 3）

| Method | Variance | Memory | Reward Model | Value Model | Reference Policy | Replay Buffer | Negative Solutions |
|--------|----------|--------|--------------|-------------|------------------|---------------|---------------------|
| REINFORCE | High | Low | ✓ | ✗ | ✗ | ✗ | ✓ |
| PPO | Low | High | ✓ | ✓ | ✓ | ✓ | ✓ |
| DPO | Low | Mid | ✗ | ✗ | ✓ | ✓ | ✓ |
| Behavior Cloning | Low | Low | ✗ | ✗ | ✗ | ✓ | ✗ |

### 5.5 o1 Learning 的推测

paper 推测 o1 采用 **warm-up + main phase** 的混合策略：

1. **Warm-up**：Behavior Cloning（高效、低 memory）
2. **Main phase**：PPO 或 DPO（利用负样本，更好 data utilization）

这与 Llama 2 (https://arxiv.org/abs/2307.09288)、Llama 3 (https://arxiv.org/abs/2407.21783) 的后训练 pipeline 一致。

---

## 6. Open-source o1 Projects 对比（Table 4）

| Project | Initialization | Reward | Train Search | Learning | Test Search |
|---------|----------------|--------|--------------|----------|-------------|
| g1 | Prompt | - | - | - | Sampling |
| Thinking Claude | Prompt | - | - | - | Sampling |
| Open-o1 | SFT | - | - | - | Sampling |
| o1-journey (P1) | SFT | PRM | Beam Search | BC | Sampling |
| o1-journey (P2) | SFT | - | - | - | Sampling (distill) |
| Open-Reasoner | - | PRM | Sampling | PPO | MCTS |
| Slow Thinking (P1) | SFT | ORM | Sampling | DPO | MCTS |
| Marco-o1 | SFT | ORM | MCTS | BC | MCTS |
| o1-coder | SFT | PRM | MCTS | PPO/DPO | MCTS |

**intuition**：开源项目覆盖了 roadmap 的不同 components，但没有任何一个完整复现四支柱。o1-journey Part 2 通过蒸馏 o1-mini 让 Qwen-72B 在 AIME 上超越 o1-preview——但这是 teacher model ceiling 限制下的 distillation，不是真正的 RL 路线。

DeepSeek-R1 (https://arxiv.org/abs/2412.19437)、QwQ (https://qwenlm.github.io/blog/qwq-32b-preview/) 是工业界更接近的尝试。

---

## 7. Future Directions

1. **General domain adaptation**：需要 general reward model，可能用 IRL from expert data
2. **Multi-modal o1**：Interleaved-modal CoT (https://arxiv.org/abs/2411.19488)，用 continuous representation 替代 text token
3. **World Model 集成**：Stage 3 (Agent) 需要 world model 支持 search 与 planning
4. **Training efficiency**：MCTS-DPO 在 A800 上训练 MATH 需要一周 (https://arxiv.org/abs/2405.00451)
5. **Question generator**：自动生成挑战性问题（curriculum learning），参考 WizardLM (https://openreview.net/forum?id=CfXh93NDgH)

---

## 8. 我的 Intuition 与联想

### 8.1 这篇 paper 的核心洞察

**Search 和 Learning 的迭代闭环**是 o1 的灵魂。这与 AlphaGo Zero 的 self-play 范式高度一致——search 产生更优 trajectories，learning 内化这些 trajectories，policy 改进后 search 又能找到更好的 trajectories。这是一个正反馈循环，理论上可以无限 scale。

### 8.2 Train-time vs Test-time Asymmetry

paper 揭示了一个关键 asymmetry：
- **Train-time**：tree search + external guidance（可并行、可访问真实环境）
- **Test-time**：sequential revisions + internal guidance（单线程、无法访问真实环境）

这个 asymmetry 的根本原因是 **inverse scaling**——proxy reward model 在 test-time 大规模 search 时会失效。所以 o1 在 test-time 必须依赖 **internalized** 的 reasoning 能力，而非 external reward signal。

### 8.3 与 Sutton's Bitter Lesson 的呼应

Sutton 说 "general methods that scale with computation" 才是 AI 的未来。Search 和 learning 正是这两个方法。o1 验证了这一论断——它没有发明新的架构，只是把 search 和 learning scale 到极致。

### 8.4 三个可能的 Extension 方向

1. **World model 作为 search 的 substrate**：当真实环境不可逆时（如 robotics、web agent），world model 提供可回滚的 simulation。MuZero 已经证明了这条路线。
2. **Continuous CoT**：text token 是低效的 thought representation。用 continuous latent 作为 thought space 可能是下一代 o1 的关键。
3. **Self-play for question generation**：AlphaGo Zero 的 self-play 在固定规则下，o1 需要 open-ended 的 question generation 来持续挑战自己。

### 8.5 与 Neuro-Symbolic 的潜在联系

paper 没有讨论，但 o1 的 reasoning behaviors（decomposition、self-evaluation、self-correction）本质上是符号推理。这可能暗示 LLM 在长 CoT 中 **emergent 地学习到了符号推理能力**，而非依赖显式的符号系统。

### 8.6 RL Scaling Law 的缺失

paper 指出一个重要 gap：**RL for LLM 的 scaling law 尚未建立**。Pre-training 有 Chinchilla law，但 RL 的 model size、data size、search budget 之间的最优分配仍是 open problem。这是 reproduce o1 的核心理论 obstacle。

OpenAI 的 RL scaling law 研究 (https://arxiv.org/abs/2301.13442) 只覆盖传统 RL 任务，未涉及 LLM。

---

## 9. 总结

这篇 paper 的贡献在于提供了一个 **compositional framework** 来理解 o1：四个 components 相互独立又紧密耦合。任何试图 reproduce o1 的工作都是这个 framework 的某个 component 或 variant。

**核心 takeaway**：
1. o1 = Policy Init (long CoT behavior) + Reward Design (PRM + shaping) + Search (train: tree/external, test: sequential/internal) + Learning (BC warmup → PPO/DPO main)
2. Search-Learn 闭环是 superhuman performance 的来源
3. Inverse scaling 是 test-time search 的核心 obstacle
4. RL for LLM 的 scaling law 仍是 open problem

**最让我兴奋的联想**：World Model + Search + Learning 的三角闭环，可能是从 o1 (Reasoner) 走向 AGI Stage 3 (Agent) 的关键技术路径。Ha & Schmidhuber 2018 的 World Models 论文 + MuZero 的 learned dynamics model + o1 的 RL pipeline，这三者的融合可能就是下一代 AI 系统。

参考核心文献：
- AlphaGo Zero: https://www.nature.com/articles/nature24270
- MuZero: https://www.nature.com/articles/s41586-020-03051-4
- World Models: https://arxiv.org/abs/1803.10122
- DPO: https://arxiv.org/abs/2305.18290
- GRPO/DeepSeekMath: https://arxiv.org/abs/2402.03300
- Let's Verify Step by Step: https://openreview.net/forum?id=v8L0pN6EOi
- Scaling LLM Test-Time Compute: https://arxiv.org/abs/2408.03314
- Reward Over-Optimization: https://proceedings.mlr.press/v202/gao23h.html

希望这些技术细节、公式拆解、和 architecture 分析能帮您 build intuition about this roadmap。这个领域正在 fast-moving，DeepSeek-R1 的开源已经验证了 paper 的部分推测，但完整的 search-learn 闭环仍是 open frontier。
