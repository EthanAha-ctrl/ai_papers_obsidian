---
source_pdf: SELF-DISTILLATION ENABLES CONTINUAL LEARNING.pdf
paper_sha256: c27949d1b7888b128bf40f6b031d4e84a82b8ba40d6e03d671491f8e414a516b
processed_at: '2026-08-12T04:37:54-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SDFT

Paper: http://idanshenfeld.com/SDFT

---

## 一句话说清楚

**让模型自己当老师教自己**——给它看一道例题，它当场学会怎么做，然后把这个"当场学会的能力"通过 gradient 蒸馏回 weights 里。

---

## 问题在哪？

想象你在训练一个很聪明的学生（foundation model）。他已经会很多东西了——会写代码、会做数学、会聊天。现在你想教他一个新技能：**用 API 调用工具**。

你有 3000 道带答案的练习题（expert demonstrations）。

**方法一：SFT (Supervised Fine-Tuning)**

最直觉的做法：把题目和答案一对一对喂给他，让他学会模仿答案。

问题出在哪？答案的"风格"跟学生本来说话的风格不一样。比如答案是简短的 `tool_call(weather, location="NYC")`，但学生平时喜欢先讲一堆 reasoning 再下结论。SFT 就会强迫学生学这个简短风格，结果他**以前会的东西也开始忘**——catastrophic forgetting。

更糟的是 [DAgger](https://arxiv.org/abs/1011.0686) 早就指出的：学生在 inference 时候的状态分布跟训练时答案的分布不一样。学生一旦在某一步稍微偏离答案的路径，后面就一路错下去——compounding errors。

**方法二：On-policy RL**

理论上最优雅。让学生自己试，用 reward function 告诉他对不对。这种方式 forget 最少（[Shenfeld et al. 2025](https://arxiv.org/abs/2509.04259) 证明过），generalize 最好。

问题：**reward function 哪来的？** 现实世界大多数时候没有 reward function，只有一堆 demonstration。

**方法三：IRL (Inverse RL)**

从 demonstration 反推 reward。听起来好，做起来难——reward function space 太大，需要强先验。max-entropy IRL 假设 expert 是 Boltzmann policy，adversarial IRL 假设有 classifier 能区分 expert vs learner，RLHF 假设有 pairwise preference。没这些先验，IRL 就是 ill-posed。

---

## SDFT 的核心 insight

大模型有个神奇能力：**in-context learning**。给它看一个例子，它不用改参数，当场就能学会类似的任务。

Paper 的核心观察：**这个"当场学会"的行为 shift，本身就是一次 trust-region policy improvement**。

换句话说：模型看到 demonstration $c$ 后，它的 behavior 从 $\pi(\cdot|x)$ shift 到 $\pi(\cdot|x,c)$，这个 shift **就等价于做了一步 on-policy RL 的 update**。

那我们能不能把这个 implicit update **显式化**，蒸馏回 weights？

---

## 怎么做？

一个模型，两个角色：

- **Teacher** = 模型 + 看 demonstration：$\pi(\cdot | x, c)$
- **Student** = 模型，不看 demonstration：$\pi_\theta(\cdot | x)$

训练循环：

1. Student 自己采样一个 response $y \sim \pi_\theta(\cdot | x)$（**on-policy**！）
2. 让 teacher 给这个 $y$ 打分（log-prob）
3. 训练 student 让它在自己采样的 $y$ 上更接近 teacher 的分布

损失函数：

$$
\mathcal{L}(\theta) = D_{KL}\big(\pi_\theta(\cdot|x) \,\|\, \pi(\cdot|x,c)\big)
$$

**变量解释**：
- $\theta$ = student 参数（要优化的）
- $x$ = 输入 prompt
- $c$ = expert demonstration
- $\pi_\theta(\cdot|x)$ = student 给 response 的概率分布
- $\pi(\cdot|x,c)$ = teacher 给 response 的概率分布

**为什么是 reverse KL？**

KL divergence 有两个方向：
- **Forward KL** $D_{KL}(\pi_{teacher} \| \pi_{student})$：student 试图覆盖 teacher 所有可能的 mode。但 on-policy 时 student 没采样到的地方根本没 gradient——所以 forward KL 在 on-policy 下不好用。
- **Reverse KL** $D_{KL}(\pi_{student} \| \pi_{teacher})$：student 在自己已经采样到的地方让 teacher 也满意。这是 **mode-seeking**，自然匹配 on-policy 训练——student 只在自己 explore 过的地方做局部修正。

---

## 关键的 IRL 视角（这是 paper 最漂亮的部分）

从 [TRPO](https://arxiv.org/abs/1502.05477) 的 trust-region 目标出发：

$$
\pi_{k+1} = \max_\pi \mathbb{E}_{y \sim \pi}[r(y,x)] - \beta D_{KL}\big(\pi(\cdot|x) \,\|\, \pi_k(\cdot|x)\big)
$$

**变量**：
- $k$ = 训练 step
- $r(y,x)$ = reward function
- $\beta$ = KL penalty 系数
- $\pi_k$ = 当前策略

这个目标的 closed-form 解（[Korbak et al. 2022](https://aclanthology.org/2022.findings-emnlp.77/) 推导过）：

$$
\pi^*_{k+1}(y|x) \propto \pi_k(y|x) \exp\left(\frac{1}{\beta} r(y,x)\right)
$$

反解 reward：

$$
r(y,x) = \beta \big[\log \pi^*_{k+1}(y|x) - \log \pi_k(y|x)\big] + C
$$

**关键假设（In-Context Assumption）**：

$$
\pi^*_{k+1}(y|x) \approx \pi(y|x, c)
$$

也就是说：**模型 + in-context demo $\approx$ 一步 trust-region 后的最优 policy**。

这个假设的物理含义：大模型 in-context learning 足够强，看一个 expert demo 就能让 behavior shift 到接近 optimal 的方向。

代入得到 **implicit reward**：

$$
r(y,x,c) = \log \pi(y|x,c) - \log \pi_k(y|x)
$$

（$\beta$ 和 $C$ 扔掉，因为 reward 的线性变换不改 optimal policy，这是 [Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf) 里的基本事实）

**token-level reward**：

$$
r_t(y_t | y_{<t}, x, c) = \log \frac{\pi(y_t | y_{<t}, x, c)}{\pi_k(y_t | y_{<t}, x)}
$$

**变量**：
- $t$ = token 位置
- $y_t$ = 第 $t$ 个 token
- $y_{<t}$ = 第 $t$ 个 token 之前的所有 token
- $\pi(y_t | y_{<t}, x, c)$ = teacher 给这个 token 的概率
- $\pi_k(y_t | y_{<t}, x)$ = 当前 student 给这个 token 的概率

**Intuition**：reward 就是 "teacher 比 student 更喜欢这个 token 的程度"。如果 teacher 喜欢（log-ratio > 0），就 enhance 这个 token；如果不喜欢（log-ratio < 0），就 penalize。

把 policy gradient 写出来：

$$
\nabla_\theta J = \mathbb{E}_{y \sim \pi_k}\left[\log \frac{\pi(y|x,c)}{\pi_k(y|x)} \nabla_\theta \log \pi_k(y|x)\right]
$$

这跟 reverse KL 的 gradient **完全等价**。

**所以 SDFT 可以这样理解**：on-policy RL with implicit reward = log teacher - log student。模型自己就是 reward model，不需要显式 infer reward。

---

## 为什么这个能 work？两个 empirical 验证

**条件 1：Teacher 要表现好（接近 optimal）**

在 ToolAlpaca 任务上：
- Base Qwen2.5-7B：42% accuracy
- Teacher (with demo)：**100% accuracy**

人工检查 50 个 teacher reasoning trace，发现 CoT 全部 valid 且 grounded，不是简单 copy demo。

**条件 2：Teacher 要离 base policy 近（trust-region 要求）**

测量 KL divergence to base model：
- SFT model：1.26 nats
- Teacher：**0.68 nats**（大约一半）

**这就是为什么 SDFT 比 SFT 好**：teacher 既表现好，又离 base 近——正是 trust-region formulation 要求的"在 KL 球面内找最优"。

SFT 直接 jump 到 expert distribution，违反了 trust-region 约束，所以 forget 一堆东西。

---

## Token-level gradient estimator 怎么选？

这个细节比看上去重要。Sequence-level KL 是：

$$
KL = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{\pi(y|x,c)}\right]
$$

难点：$\pi_\theta$ 既出现在采样分布里，又出现在 log 里。[Tang & Munos 2025](https://arxiv.org/abs/2506.09477) 分析了几种 estimator：

**Estimator 1: Token-level (partial)**

$$
\hat{g}_{token} = \sum_t \log \frac{\pi_\theta(y_t | y_{<t}, x)}{\pi(y_t | y_{<t}, x, c)} \nabla_\theta \log \pi_\theta(y_t | y_{<t}, x)
$$

只考虑 sampled token，忽略 early token 对 future token 分布的影响。Biased，variance 高。

**Estimator 2: Full analytic per-token（论文选用）**

$$
\hat{g}_{analytic} = \sum_t \sum_{v \in \mathcal{V}} \log \frac{\pi_\theta(v | y_{<t}, x)}{\pi(v | y_{<t}, x, c)} \nabla_\theta \log \pi_\theta(v | y_{<t}, x)
$$

**变量**：$v$ 遍历整个 vocabulary $\mathcal{V}$。

对每个时间步，对 vocabulary 里所有 token 都算一遍 log-ratio × gradient。虽然理论上有 bias（没考虑 token 之间的影响），但 variance 低，复用 forward pass 计算过的 logits，工程上 friendly。**实验中效果最好**。

**Estimator 3: Rao-Blackwellized**（[Amini et al. 2025](https://arxiv.org/abs/2410.20952)）

理论无偏、方差最低，但计算昂贵。实验中没有显著优势，不用。

**Intuition**：在 LLM training 里，theoretical unbiasedness 不一定 win。Analytic per-token 虽然 biased，但每个 token 都得到精确的 KL contribution（不只 sampled token），优化路径更平滑。

---

## Teacher 参数怎么设？EMA 是关键

这个问题容易被忽略但极其关键。Teacher 的 weights 有 3 种选择：

**选项 A: Frozen base model**

稳定，但 teacher 不能反映 student 的进步。最终性能差。

**选项 B: Current student**

跟踪 student 进步，但**训练 diverge**。原因：on-policy 是个反馈循环，student 微小 stochastic 波动被 teacher 立即放大，正反馈爆炸。

**选项 C: EMA of student（论文选择）**

$$
\phi \leftarrow \alpha \theta + (1 - \alpha) \phi
$$

**变量**：
- $\phi$ = teacher weights
- $\theta$ = student weights
- $\alpha$ = EMA rate（实验中 $\alpha \in \{0.01, 0.02, 0.05\}$）

**Intuition**：EMA 是 control theory 里的 low-pass filter。滤掉 high-frequency 噪声，保留 low-frequency 信号。Teacher 既不会 frozen（能 track 进展），也不会被 noise 主导（避免 divergence）。

$\alpha = 0.01$ 意味着 teacher 每次只吸收 1% 的 student 更新——足够 track 趋势，但不被瞬时噪声带跑。

这个 trick 在 self-distillation 里反复出现：[DQN 的 target network](https://arxiv.org/abs/1312.5602)、[MOCO 的 momentum encoder](https://arxiv.org/abs/1911.05722)、[BYOL](https://arxiv.org/abs/2006.07733)、[DINO](https://arxiv.org/abs/2104.14294)。

---

## 实验结果

### Knowledge Acquisition（注入 2025 Wikipedia 自然灾害文章）

| Method | Strict Acc | Lenient Acc | OOD Acc |
|--------|-----------|-------------|---------|
| Base | 0 | 0 | 0 |
| Oracle RAG | 91 | 100 | 100 |
| CPT | 9 | 37 | 7 |
| SFT | 80 | 95 | 80 |
| **SDFT** | **89** | **100** | **98** |

**关键观察**：

1. **CPT 几乎没用**（9% strict）——直接在 raw text 上 next-token prediction 不能注入知识。
2. **SDFT 在 OOD 上接近完美（98%）**，SFT 只有 80%。说明 SDFT 真正把信息 integrate 进 model 的 knowledge base，SFT 只是在 narrow 形式上 memorize 答案。
3. SDFT 几乎追上 oracle RAG（91 vs 89），但 RAG 需要外部 retrieval system，SDFT 内化到 weights 里。

**Intuition**：OOD question 比如 "2025 哪些国家需要人道主义援助？"——SFT 没见过这个具体形式，因为它学的是 narrow 的"问-答"mapping。SDFT 通过 teacher-conditioned-on-text-plus-answer 把 knowledge 真正蒸馏进 reasoning 路径，能 generalize 到新问题形式。

### Skill Learning（三个任务汇总）

Science Q&A / Tool Use / Medical，SDFT 在三个任务上都做到：**新任务准确率 > SFT，旧能力保持 ≈ base model**。

SFT 在所有任务上都造成 5-12 个百分点的 prior capability 下降。SDFT 几乎零 forgetting。

**这是双 win**：理论上 on-policy 应该有 trade-off，但实际上没有。原因：teacher 离 base policy 近（KL = 0.68 vs SFT 的 1.26），trust-region 约束天然满足，update 是小幅度的局部修正，不会 destroy 旧能力。

### Multi-Task Continual Learning（Figure 3）

模型顺序学三个 skill。SDFT 每学一个新 skill，旧 skill 性能基本保持。SFT 则剧烈 oscillation——一切到新任务，旧任务立刻崩。

**这是 paper 最有力的证据**：SDFT 实现了 true continual learning。

### Scaling（Figure 5 left）

Science Q&A 上：
- 3B model：SDFT < SFT（ICL 太弱，teacher 信号差）
- 7B model：+4 points
- 14B model：+7 points

**关键 insight**：SDFT 完全依赖 in-context learning 能力。模型越大，ICL 越强，SDFT 越有效。这暗示未来更大模型上 SDFT 越来越有优势。

### Reasoning Model 保护（Table 2）

用 Olmo-3-7B-Think 训 medical，demonstration 只有 final answer，没有 CoT：

| Method | Accuracy | Avg # tokens |
|--------|----------|-------------|
| Base | 31.2 | 4612 |
| + SFT | 23.5 ↓ | 3273 ↓ (collapse) |
| + SDFT | **43.7** ↑ | 4180 |

**Intuition**：SFT 直接 match 短答案，惩罚长 CoT，导致 reasoning collapse。SDFT 的 teacher 是 same model + demo，虽然 demo 短，但 ICL 让 teacher 自己 reconstruct 长 CoT，student 学的是这个长 CoT 分布，reasoning depth 被保留。

**Practical takeaway**：只要 demo 缺 CoT，就别用 SFT，用 SDFT。

### On-policy 是必须的（Figure 6）

关键 ablation：如果有这么好的 teacher，直接 offline distillation 行不行？

对比：
- SFT from teacher samples：offline imitate teacher
- Offline distillation：固定 dataset 上做 KL loss
- **On-policy SDFT**

结果：**两种 offline 方案都比 on-policy SDFT 差**。

**Intuition**：再次印证 [DAgger](https://arxiv.org/abs/1011.0686) 的核心 insight——off-policy 在 inference 时有 state distribution mismatch。Teacher 采样分布 ≠ Student inference 分布，即使 teacher 再好，offline 学到的也会 drift。On-policy 通过让 student 在自己分布上被 correct，根本消除这个 mismatch。

---

## 完整算法（Algorithm 1 简化版）

```
Input: demonstration dataset D = {(x_i, c_i)}
Input: model π, EMA rate α

1. Initialize teacher weights φ = θ
2. For each training step:
   a. Sample minibatch from D
   b. For each (x, c):
      - Student rollout: y ~ π_θ(·|x)  # ON-POLICY!
      - Compute student log-probs: log π_θ(y_t | y_<t, x)
      - Compute teacher log-probs: log π_φ(y_t | y_<t, x, c)
   c. Compute gradient using analytic per-token estimator
   d. Update student: θ ← θ - η·g
   e. EMA update teacher: φ ← α·θ + (1-α)·φ
```

**Compute cost**：约 2.5× SFT 的 FLOPs，4× wall-clock time（因为要 on-policy generation）。但比 Re-invoke 这种多阶段方法总时间反而更短。

---

## 失败模式 & 局限

1. **Learned artifacts**：Student 会继承 teacher 的语言 pattern（"Based on the text...", "Following the example..."），即使 student 没看到 context。Hack：mask 前几个 token 的 loss。Principled solution 仍是 open problem。

2. **小模型不能用**：< 7B 模型 ICL 太弱，teacher 信号差。

3. **大行为变化困难**：把 non-reasoning model 变成 explicit CoT model 很难——SDFT 适合 refinement & knowledge injection，不适合 fundamental pattern shift。

4. **仍有一些 forgetting**：虽然大幅减少，但不是零。

---

## 整体直觉

### 三层 intuition

**Layer 1: Distribution geometry**

SFT 是远距离 jump 到 expert distribution，违反 trust-region 约束。SDFT 的 teacher 是 **same model + context shift**，所以 teacher 在 KL 球面内但向 task-optimal 方向 tilt。Student 跟随 teacher，每个 update 都是**小幅度的 on-policy 修正**，自然满足 trust-region。

**Layer 2: On-policy vs Off-policy**

[DAgger](https://arxiv.org/abs/1011.0686) 早就指出 off-policy 在 inference 时会 drift。On-policy 通过让 model 在自己 distribution 上被 correct，根本消除 train/test distribution mismatch。

SDFT 把这个推到极致：teacher 不是 external expert，是 **self + demo**，所以 teacher 和 student 共享 model 的 internal reasoning style，零 external distribution shift。

**Layer 3: IRL interpretation**

经典 IRL 要 infer 一个 explicit reward function，但 reward function space 巨大，需要强先验。

SDFT 的 insight：**model 的 in-context learning 已经隐式定义了一个 reward function**——$r = \log \pi(\cdot|c) - \log \pi_k$。这个 reward 不需要任何先验，因为它直接来自 model 自己的 behavior shift。

这其实是 [Korbak et al. 2022](https://aclanthology.org/2022.findings-emnlp.77/) 的 bayesian inference 视角——KL-regularized RL 等价于 bayesian inference，posterior $\propto$ prior $\times$ likelihood。SDFT 的 prior 是 $\pi_k$，"likelihood" 是 $\pi(\cdot|c) / \pi_k$，即"看到 demo 后 model behavior 的 shift ratio"。In-context learning 就是这个 likelihood 的 implicit form。

### 一句话总结

**SDFT = 用 in-context learning 当免费 reward model 做 on-policy distillation**。

它绕过了 IRL 的 reward inference 难题，绕过了 SFT 的 off-policy forgetting 问题，绕过了 RL 的 explicit reward 需求，把"持续从 demonstration 学习"变成一个 stable、scalable 的训练算法。在 7B+ 模型上效果显著，随 scale 增长。这是 continual learning from demonstrations 的一个真正可工程化路径。

---

**References**:
- Paper: http://idanshenfeld.com/SDFT
- [DAgger (Ross et al. 2011)](https://arxiv.org/abs/1011.0686)
- [TRPO (Schulman et al. 2015)](https://arxiv.org/abs/1502.05477)
- [DPO (Rafailov et al. 2023)](https://arxiv.org/abs/2305.18290)
- [On-policy distillation (Agarwal et al. 2024)](https://arxiv.org/abs/2406.12851)
- [RL's Razor (Shenfeld et al. 2025)](https://arxiv.org/abs/2509.04259)
- [Retaining by doing (Chen et al. 2025)](https://arxiv.org/abs/2510.18874)
- [On-policy distillation (Lu & Thinking Machines Lab 2025)](https://thinkingmachines.ai/blog/on-policy-distillation)
- [KL gradient pitfalls (Tang & Munos 2025)](https://arxiv.org/abs/2506.09477)
- [KL estimator (Amini et al. 2025)](https://arxiv.org/abs/2410.20952)
- [Constitutional AI (Bai et al. 2022)](https://arxiv.org/abs/2212.08073)
- [Context Distillation (Snell et al. 2022)](https://arxiv.org/abs/2209.15189)
- [Cartridges (Eyuboglu et al. 2025)](https://arxiv.org/abs/2506.06266)
- [Kujanpää et al. 2025](https://arxiv.org/abs/2502.08195)
- [Mecklenburg et al. 2024](https://arxiv.org/abs/2404.00213)
- [IRL (Ng et al. 2000)](https://arxiv.org/abs/cs/9905114)
- [KL as Bayesian inference (Korbak et al. 2022)](https://aclanthology.org/2022.findings-emnlp.77/)
- [DQN target network (Mnih et al. 2013)](https://arxiv.org/abs/1312.5602)
- [MOCO (He et al. 2020)](https://arxiv.org/abs/1911.05722)
- [BYOL (Grill et al. 2020)](https://arxiv.org/abs/2006.07733)
- [DINO (Caron et al. 2021)](https://arxiv.org/abs/2104.14294)
- [Sutton & Barto RL book](http://incompleteideas.net/book/RLbook2020.pdf)

---

# Self-Distillation Enables Continual Learning 深度解读

Paper link: http://idanshenfeld.com/SDFT
Authors: Idan Shenfeld, Mehul Damani, Jonas Hubotter, Pulkit Agrawal (MIT / Improbable AI Lab / ETH Zurich)

---

## 1. 核心问题：为什么 Continual Learning 这么难？

Foundation models 在deploy后就变成static了。要让它们持续学习新skill / 新knowledge，会撞上一个老大难问题：**catastrophic forgetting**。

现有的两条路都有问题：

- **SFT (Supervised Fine-Tuning)**：直接用expert demonstrations训练，简单可扩展，但本质上是 **off-policy** 的——训练数据来自expert的轨迹分布，而inference时模型在自己的分布上跑。Ross et al. 2011 ([DAgger paper](https://arxiv.org/abs/1011.0686))早就证明这会导致 **compounding errors**：模型稍微偏离demonstration覆盖的state，error就指数级累积。
- **On-policy RL**：理论上最优雅，[Shenfeld et al. 2025](https://arxiv.org/abs/2509.04259) 和 [Chen et al. 2025](https://arxiv.org/abs/2510.18874) 都证明 on-policy 更不容易遗忘。但 RL 需要 **explicit reward function**，现实中经常没有。
- **IRL (Inverse RL)**：从demonstrations反推reward，[Ng et al. 2000](https://arxiv.org/abs/cs/9905114)。理论优雅但实际不scale，需要强先验（max-entropy IRL假设Boltzmann policy，adversarial IRL假设有classifier能区分expert/learner，RLHF假设有preference pairs）。

SDFT 的核心 insight：**能不能不显式推断reward，只用模型自己的 in-context learning 能力，就把 on-policy 的好处拿到手？**

---

## 2. 方法：Self-Distillation Fine-Tuning (SDFT)

### 2.1 核心idea

用一个模型扮演两个角色：

- **Teacher** = 模型 + expert demonstration conditioning: $\pi(\cdot | x, c)$
- **Student** = 基础模型: $\pi_\theta(\cdot | x)$

Prompt template（防止teacher直接抄demo）：
```
<Question>
This is an example for a response to the question:
<Demonstration>
Now answer with a response of your own, including the thinking process:
```

Training时：student **on-policy** 采样 $y \sim \pi_\theta(\cdot | x)$，然后用 teacher $\pi(\cdot | x, c)$ 当supervision，最小化 **reverse KL divergence**。

### 2.2 损失函数

$$
\mathcal{L}(\theta) = D_{KL}\big(\pi_\theta(\cdot | x) \,\|\, \pi(\cdot | x, c)\big) = \mathbb{E}_{y \sim \pi_\theta(y|x)}\left[\log \frac{\pi_\theta(y|x)}{\pi(y|x,c)}\right]
$$

**变量解释**：
- $\theta$: student 模型参数（要优化的对象）
- $x$: 输入 prompt
- $c$: expert demonstration
- $y$: 模型生成的 response（sequence）
- $\pi_\theta(y|x)$: student 给整个 sequence $y$ 的概率
- $\pi(y|x,c)$: teacher 给同一个 $y$ 的概率

**为什么是 reverse KL 而不是 forward KL？**

- **Reverse KL** $D_{KL}(P \| Q)$ 是 **mode-seeking**：student 会集中在 teacher 的高概率区域。因为 expectation 是在 student 自己分布上取的，student 必须在自己已经sample的地方让teacher也高兴，否则penalty大。
- **Forward KL** $D_{KL}(Q \| P)$ 是 **mode-covering**：student 试图覆盖 teacher 的所有mode，但这在 on-policy 下不合适——student 没sample到的mode根本不会出现在gradient里。

在on-policy setting下，reverse KL 自然匹配：student只在自己已经explore过的地方做局部改进，不会乱跳。

### 2.3 Token-level gradient estimator

利用autoregressive分解，Equation 1展开成token-level：

$$
\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_{y \sim \pi_\theta}\left[\sum_t \sum_{y_t \in \mathcal{V}} \log \frac{\pi_\theta(y_t | y_{<t}, x)}{\pi(y_t | y_{<t}, x, c)} \nabla_\theta \log \pi_\theta(y_t | y_{<t}, x)\right]
$$

**变量解释**：
- $\mathcal{V}$: token vocabulary
- $t$: 时间步（token位置）
- $y_{<t}$: 第 $t$ 个token之前已生成的所有token
- $y_t$: 第 $t$ 个token
- $\pi_\theta(y_t | y_{<t}, x)$: student 给第 $t$ 个token的条件概率
- $\pi(y_t | y_{<t}, x, c)$: teacher 给同一token的条件概率

**Intuition**：gradient方向 = "对每个student采样到的token，根据 teacher vs student 的log-ratio来reward或penalize"。如果 teacher 比 student 更喜欢这个token（log-ratio > 0），就增强它；反之就压低。本质上是 **token-level 的 advantage estimation**，而 advantage 就是 log-prob ratio。

---

## 3. IRL 视角：为什么这玩意儿其实是隐式 Inverse RL

这是paper最有意思的理论部分。从 [TRPO](https://arxiv.org/abs/1502.05477) 的 trust-region 目标出发：

$$
\pi_{k+1} = \max_\pi \mathbb{E}_{y \sim \pi}[r(y,x)] - \beta D_{KL}\big(\pi(\cdot|x) \,\|\, \pi_k(\cdot|x)\big)
$$

**变量**：
- $k$: 训练step
- $\beta$: KL penalty 系数
- $r(y,x)$: reward function
- $\pi_k$: 当前策略

[Korbak et al. 2022](https://aclanthology.org/2022.findings-emnlp.77/) 和 [DPO Rafailov et al. 2023](https://arxiv.org/abs/2305.18290) 都指出这个目标的closed-form解是 tilted distribution：

$$
\pi^*_{k+1}(y|x) \propto \pi_k(y|x) \exp\left(\frac{1}{\beta} r(y,x)\right)
$$

反解 reward：

$$
r(y,x) = \beta \big[\log \pi^*_{k+1}(y|x) - \log \pi_k(y|x)\big] + C
$$

**关键假设（In-Context Assumption, Equation 4）**：

$$
\pi^*_{k+1}(y|x) \approx \pi(y|x, c)
$$

即 **model conditioned on demonstration 近似optimal next policy**。这个假设的物理含义：大模型的in-context learning足够强，看一个expert demo就能把behavior shift到接近optimal的方向。

代入得到 **intrinsic reward**：

$$
r(y,x,c) = \log \pi(y|x,c) - \log \pi_k(y|x)
$$

（$\beta$ 和 $C$ 丢掉，因为reward的线性变换不改变optimal policy，见 [Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf)）

**Token-level decomposition**：

$$
r_t(y_t | y_{<t}, x, c) = \log \frac{\pi(y_t | y_{<t}, x, c)}{\pi_k(y_t | y_{<t}, x)}
$$

满足 $\sum_t r_t = r(y,x,c)$（autoregressive分解的自然性质）。

**Policy gradient**：

$$
\nabla_\theta J(\pi_k) = \mathbb{E}_{y \sim \pi_k}\left[\log \frac{\pi(y|x,c)}{\pi_k(y|x)} \nabla_\theta \log \pi_k(y|x)\right]
$$

这跟 Equation 2 的 reverse KL gradient **完全等价**。所以 SDFT 可以视为：**on-policy RL with implicit reward = log teacher - log student**。

这个interpretation很重要的原因是：传统IRL要显式infer reward function，需要强结构假设；SDFT 用in-context learning **隐式** 实现了这件事——model自己就是reward model。

---

## 4. ICL Assumption 的两个验证条件

Paper明确指出，Equation 4 成立需要两个条件：

### Condition 1: Optimality
$$
\mathbb{E}_{y \sim \pi(\cdot|x,c)}[r(y,x)] \approx \mathbb{E}_{y \sim \pi^*_{k+1}}[r(y,x)]
$$
Teacher 采样的 trajectory 应该接近 optimal reward。

### Condition 2: Minimal Deviation
$$
D_{KL}\big(\pi(\cdot|x,c) \,\|\, \pi_k(\cdot|x)\big) \approx D_{KL}\big(\pi^*_{k+1}(\cdot|x) \,\|\, \pi_k(\cdot|x)\big)
$$
Teacher 在所有能达到 optimal reward 的策略中，应该是离 $\pi_k$ 最近的（trust-region要求）。

**为什么第二个条件重要？** 如果 teacher 只是抄 demo（verbatim copy），它会大幅度偏离 base model，丢掉 on-policy 的好处。真正有价值的 teacher 是 **保留 base model 的 reasoning style，但行为向 task-optimal 方向shift**。

### 实验验证（ToolAlpaca, Qwen-2.5-7B-Instruct）

| Metric | 结果 |
|--------|------|
| Base model accuracy | 42% |
| Teacher (conditioned on demo) accuracy | **100%** |
| 人工检查50个teacher推理trace | CoT全部valid且语义grounded，非复制 |
| **KL to base** SFT model | 1.26 nats |
| **KL to base** Teacher | 0.68 nats（约一半） |

第二个数字特别关键：teacher 的 KL divergence 是 SFT 的 **一半**，这正是 trust-region formulation 要求的——找到的"optimal policy"要在 KL 球面内尽量靠近 $\pi_k$。

---

## 5. KL Gradient Estimator 的选择（Appendix A.1）

这部分很技术性但实际很关键。Reverse KL 在 sequence level 是：

$$
KL(\pi_\theta \| \pi) = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{\pi(y|x,c)}\right]
$$

由于 $\pi_\theta$ 既出现在采样分布里，又出现在log里，gradient估计tricky。[Tang & Munos 2025](https://arxiv.org/abs/2506.09477) 详细分析了不同estimator的bias/variance。

### Estimator 1: Token-level (partial)
$$
\hat{g}_{token} = \sum_t \log \frac{\pi_\theta(y_t | y_{<t}, x)}{\pi(y_t | y_{<t}, x, c)} \nabla_\theta \log \pi_\theta(y_t | y_{<t}, x)
$$
**Bias**: 忽略了 early token 对 future token 分布的影响，biased w.r.t. true sequence KL。
**Pros**: 单。
**Cons**: 实验中variance高、KL control弱。

### Estimator 2: Full analytic per-token（论文最终选用）
$$
\hat{g}_{analytic} = \sum_t \sum_{v \in \mathcal{V}} \log \frac{\pi_\theta(v | y_{<t}, x)}{\pi(v | y_{<t}, x, c)} \nabla_\theta \log \pi_\theta(v | y_{<t}, x)
$$
**变量**：$v$ 遍历整个 vocabulary。
**Bias**: 仍是 sequence-level biased（没考虑 $y_t$ 怎么影响 $y_{>t}$）。
**Pros**: 比 sample-based token estimator 方差更低，且 forward pass 已经计算了这些量，计算上 friendly。
**实验结果**: 最稳定、下游性能最好。论文最终选这个。

### Estimator 3: Rao-Blackwellized ([Amini et al. 2025](https://arxiv.org/abs/2410.20952))
$$
\hat{g}_{rb} = \sum_t \left[\sum_{v \in \mathcal{V}} \log \frac{\pi_\theta(v|y_{<t},x)}{\pi(v|y_{<t},x,c)} \nabla_\theta \log \pi_\theta(v|y_{<t},x) + k_\theta(y_{<t}) \sum_{i=1}^{t-1} \nabla_\theta \log \pi_\theta(y_i|y_{<i},x)\right]
$$

其中 $k_\theta(y_{<t}) = KL\big(\pi_\theta(\cdot|y_{<t},x) \,\|\, \pi(\cdot|y_{<t},x,c)\big)$ 是 stepwise KL。

**性质**: 无偏、方差最低。
**Cons**: 计算昂贵。
**实验**: 没有显著优势，所以不用。

**Intuition**: 在实际 LLM training 中，theoretical unbiasedness 不一定带来 win。Analytic per-token estimator 虽然有 bias，但它复用 forward pass 的 logits，每个token都得到了精确的 KL contribution（不只是 sampled token），所以优化路径更平滑。

---

## 6. Teacher 的 Parameterization 选择（Appendix A.3）

这是论文中一个**容易被忽视但极其关键**的设计。Teacher 的 weights 有3种选择：

### 选项 A: Frozen base model
- **Pros**: 稳定。
- **Cons**: Teacher 不能反映 student 的进步，最终性能差。
- 类似于把 in-context learning 当作完全 static 的 oracle。

### 选项 B: Current student (online teacher)
- **Pros**: 跟踪 student 进步。
- **Cons**: **训练 diverge**。原因是 on-policy feedback loop：student 的微小 stochastic 波动被 teacher 立即放大，造成正反馈爆炸。

### 选项 C: EMA of student（论文选择）
$$
\phi \leftarrow \alpha \theta + (1-\alpha) \phi
$$
- $\phi$: teacher weights
- $\theta$: student weights
- $\alpha$: EMA rate（实验中 $\alpha \in \{0.01, 0.02, 0.05\}$）

**Intuition**: EMA 是 control theory 里的经典 trick——给反馈loop加一个 low-pass filter，滤掉 high-frequency 噪声但保留 low-frequency 信号。Teacher 既不会 frozen（能跟踪进展），也不会被 noise 主导（避免 divergence）。

类似 [Target network in DQN](https://arxiv.org/abs/1312.5602) 和 [MOCO's momentum encoder](https://arxiv.org/abs/1911.05722) 的思路。

---

## 7. 实验结果详解

### 7.1 Knowledge Acquisition (Table 1)

任务：注入 2025 年的 Wikipedia 自然灾害文章（约 200K tokens），用 GPT-5 生成 5× 大小的 QA 对。

| Method | Strict Acc | Lenient Acc | OOD Acc |
|--------|-----------|-------------|---------|
| Base | 0 | 0 | 0 |
| Oracle RAG | 91 | 100 | 100 |
| CPT (Continual Pre-Training) | 9 | 37 | 7 |
| SFT | 80 | 95 | 80 |
| **SDFT** | **89** | **100** | **98** |

**关键观察**：
1. **CPT 几乎没用**（9% strict）——直接在 raw text 上做 next-token prediction 不能注入知识，跟 [Mecklenburg et al. 2024](https://arxiv.org/abs/2404.00213) 一致。
2. **SDFT 在 OOD 上接近完美（98%）**，而 SFT 只有 80%。这是 paper 最 striking 的结果——说明 SDFT 真正把信息 **integrate 进 model 的 knowledge base**，而 SFT 只是在 narrow 形式上 memorize 答案。
3. SDFT 几乎追上 oracle RAG（91 vs 89），但 RAG 需要外部 retrieval system，SDFT 是内化到 weights 里。

**Intuition**: OOD question 比如 "2025 哪些国家需要国际人道主义援助？"，SFT 没见过这个具体形式，因为它学的是"问-答"的 narrow mapping；SDFT 通过 teacher-conditioned-on-text-plus-answer 把 knowledge 真正蒸馏到 student 的 reasoning 路径里，所以能 generalize 到新问题形式。

### 7.2 Skill Learning (Table 5)

Science Q&A（SciKnowEval Chemistry L-3）:

| Method | New Task | Avg Prior |
|--------|---------|-----------|
| Base Qwen2.5-7B | 32.1 | 65.5 |
| SFT | 66.2 | **53.4** ↓ |
| SFT + re-invoke | 66.0 | 60.2 |
| DFT | 54.8 | 60.2 |
| **SDFT** | **70.2** | **64.5** |

Tool Use (ToolAlpaca):

| Method | New Task | Avg Prior |
|--------|---------|-----------|
| Base | 42.9 | 65.5 |
| SFT | 63.2 | **56.0** ↓ |
| DFT | 63.1 | 63.7 |
| **SDFT** | **70.6** | **65.4** |

Medical (HuatuoGPT-o1):

| Method | New Task | Avg Prior |
|--------|---------|-----------|
| Base | 30.1 | 65.5 |
| SFT | 35.5 | 60.2 |
| DFT | 36.2 | 64.0 |
| **SDFT** | **40.2** | **65.4** |

**关键观察**：
1. SFT 在所有任务上都造成显著 prior capability 下降（5-12 个百分点）。
2. **SDFT 在新任务上比 SFT 更好（70.6 vs 63.2 on Tool Use），同时几乎完全保留 prior capabilities**。这是个双赢——理论上 on-policy 应该 trade off，但实际上没有。
3. **Re-invoke 方法**（[Lu & Thinking Machines Lab 2025](https://thinkingmachines.ai/blog/on-policy-distillation)）只能部分恢复 prior，不能完全恢复，且要 sequential 训练阶段。

### 7.3 Multi-Task Continual Learning (Figure 3)

模型顺序训练三个 skill。SDFT 是唯一能让模型累积学习的——每学一个新 skill，旧 skill 性能基本保持。SFT 则表现出剧烈 oscillation——一旦切到新任务，旧任务性能立刻崩。

这印证了 paper 核心claim：**SDFT enable true continual learning**。

### 7.4 Scaling (Figure 5 left)

在 Science Q&A 上：

| Model Size | SDFT vs SFT |
|-----------|-------------|
| 3B | SDFT < SFT |
| 7B | +4 points |
| 14B | +7 points |

**关键 insight**: SDFT 完全依赖 in-context learning 能力。3B 模型 ICL 太弱，conditioned teacher 不够好；越大模型越受益。这暗示 **未来更大模型上 SDFT 会越来越有优势**。

### 7.5 Reasoning Model 保护 (Table 2)

用 Olmo-3-7B-Think 训练 medical task，但 demonstration 只有 final answer，没有 CoT。

| Method | Accuracy | Avg # tokens |
|--------|----------|-------------|
| Base Olmo-3-7B-Think | 31.2 | 4612 |
| + SFT | 23.5 ↓ | 3273 ↓ (collapse) |
| + SDFT | **43.7** ↑ | 4180 |

**Intuition**: SFT 直接 match 短答案，惩罚长 CoT，导致 reasoning collapse。SDFT 的 teacher 是 same model + demo，所以 teacher 自己也会生成长 CoT（虽然 demo 短，但 ICL 让它reconstruct reasoning），student 学的是这个长 CoT 的分布，所以 reasoning depth 被保留。

这是个很重要的 practical finding：**只要 demo 缺 CoT，就别用 SFT**。

### 7.6 On-policy 是必须的 (Figure 6)

Paper 做了关键 ablation：如果有这么好的 teacher，直接 offline distillation 行不行？

对比：
- (1) SFT from teacher：offline imitate teacher samples
- (2) Offline distillation from teacher：固定 dataset 上做 KL loss
- (3) **On-policy SDFT**

结果：**两种 offline 方案都比 on-policy SDFT 差**。

**Intuition**: 这再次印证 [Ross et al. 2011](https://arxiv.org/abs/1011.0686) 的核心发现——off-policy 在 inference 时有 state distribution mismatch。Teacher 采样分布 ≠ Student inference 分布，即使 teacher 再好，offline 学到的也会在测试时drift。On-policy 通过让 student 在自己分布上被 correct，从根上解决这个 mismatch。

### 7.7 Demonstration Context 的组成 (Figure 7, Appendix A.2)

在 Knowledge Acquisition 上 ablation teacher 的 context：

| Teacher context | Strict Acc |
|----------------|------------|
| Only article text | 75 |
| Only answer | 中等 |
| Text + Answer (full) | **89** |

这表明：**单纯的 text-only distillation 是弱信号**（跟 [Cartridges](https://arxiv.org/abs/2506.06266) 和 [Kujanpää et al. 2025](https://arxiv.org/abs/2502.08195) 的发现一致）。Answer 提供了 task-relevant 的 guidance，让 teacher 知道"这段 text 里什么重要"。

---

## 8. Algorithm 1 完整流程

```
Input: demonstration dataset D = {(x_i, c_i)}
Input: autoregressive model π
Input: teacher EMA rate α

1. Initialize teacher weights φ = θ
2. For each training step:
   a. Sample minibatch B from D
   b. For each (x_i, c_i) in B (parallel):
      - Build student context s_i = Ctx_S(x_i)
      - Sample y_i ~ P_sample(·|s_i)  # ON-POLICY rollout
      - Build teacher context t_i = Ctx_T(x_i, c_i)
      - Compute student log-probs ℓ^S = log π_θ(y_{i,t} | y_{i,<t}, s_i)
      - Compute teacher log-probs ℓ^T = log π_φ(y_{i,t} | y_{i,<t}, t_i)
   c. Compute gradient g using analytic per-token estimator
   d. Update student: θ ← θ - η·g
   e. Update teacher EMA: φ ← α·θ + (1-α)·φ
```

**Compute cost**: 约 2.5× SFT 的 FLOPs，4× wall-clock time（因为要 on-policy generation）。但对比 Re-invoke 这种多阶段方法，总时间反而更短。

---

## 9. Limitations & 失败模式

1. **Learned artifacts**：Student 会继承 teacher 的语言 pattern（"Based on the text...", "Following the example..."），即使 student 没看到 context。Hack: mask 前几个 token 的 loss。Principled solution 仍是 open problem。

2. **小模型不能用**：< 7B 的模型 ICL 太弱，teacher 信号差。

3. **大行为变化困难**：把 non-reasoning model 变成 explicit CoT model 很难——SDFT 适合 refinement & knowledge injection，不适合 fundamental pattern shift。

4. **仍有一些 forgetting**：虽然大幅减少，但不是零。需要更多 complementary techniques。

---

## 10. 我的整体直觉构建

### 10.1 为什么 SDFT work？三层直觉

**Layer 1: Distribution geometry**

SFT 把 student 推到 expert distribution，这是远距离 jump。Trust-region 视角下，这违反了 KL 约束。

SDFT 的 teacher 是 **same model + context shift**，所以 teacher 在 KL 球面内，但向 task-optimal 方向tilt。Student 跟随 teacher，每个 update 都是 **小幅度的 on-policy 修正**，自然满足 trust-region。

**Layer 2: On-policy vs Off-policy**

[DAgger](https://arxiv.org/abs/1011.0686) 的核心 insight: off-policy 在 inference 时会drift。On-policy 通过让 model 在自己 distribution 上被 correct，根本消除 train/test distribution mismatch。

SDFT 把这个 insight 推到极致：teacher 不是外部 expert，是 **自己 + demo**，所以 teacher 和 student 共享 model's internal reasoning style，没有任何 external distribution shift。

**Layer 3: IRL interpretation**

经典 IRL 要 infer 一个 explicit reward function，但 reward function space 巨大，需要强先验。

SDFT 的 insight 是：**model 的 in-context learning 已经隐式定义了一个 reward function**——$r = \log \pi(\cdot|c) - \log \pi_k$。这个 reward 不需要任何先验，因为它直接来自 model 自己的 behavior shift。

这其实是个 bayesian inference 视角——[Korbak et al. 2022](https://aclanthology.org/2022.findings-emnlp.77/) 证明 KL-regularized RL 等价于 bayesian inference，posterior $\propto$ prior $\times$ likelihood。SDFT 的 prior 是 $\pi_k$，"likelihood" 是 $\pi(\cdot|c) / \pi_k$，即 "看到 demo 后 model behavior 的 shift ratio"。In-context learning 就是这个 likelihood 的 implicit form。

### 10.2 为什么 EMA teacher 是 critical？

这个问题让我想到 [Deep RL 中的 target network](https://arxiv.org/abs/1312.5602)。On-policy training 是个 **反馈循环**：

```
student 更新 → teacher 跟着变 → supervision signal 变 → student 更新...
```

如果 teacher 跟得太紧（current student as teacher），任何 noise 都被放大，导致 divergence。如果 teacher 不动（frozen base），无法跟踪进展。

EMA 是 **低通滤波器**：滤掉 high-frequency 的 stochastic noise，保留 low-frequency 的真实进展信号。$\alpha = 0.01$ 意味着 teacher 每次只吸收 1% 的 student 更新——足够 track 趋势，但不会被瞬时噪声带跑。

这个 trick 在 self-distillation / self-supervised learning 里反复出现（[BYOL](https://arxiv.org/abs/2006.07733), [DINO](https://arxiv.org/abs/2104.14294)），不是新东西，但 SDFT 把它放在 on-policy distillation 的语境下重新理解，很有意思。

### 10.3 这个方法的根本性意义

我觉得这篇 paper 最 deep 的贡献是把三件事联系起来：

1. **In-context learning is implicit policy optimization**：给 model 看一个 demo，它的 behavior shift 等价于做了一步 trust-region policy improvement。
2. **On-policy distillation from a context-conditioned self**：把这个 implicit improvement 显式化，通过 gradient 蒸馏回 weights。
3. **Continual learning without reward function**：因为 (1) 和 (2)，我们能在没有 explicit reward 的情况下做 on-policy continual learning。

这给了一个 unifying view：**fine-tuning 本质上是把 in-context learning 能力 internalize 到 weights 里**。SFT 是 bad 版本（off-policy jump），SDFT 是 good 版本（on-policy KL-constrained update）。

### 10.4 跟其他工作的联系

- **[Cartridges (Eyuboglu et al. 2025)](https://arxiv.org/abs/2506.06266)**：也是 self-study 思路，但 offline、text-only。SDFT 在 Knowledge Acquisition 上印证了 text-only context 是弱信号，需要 + answer + on-policy。
- **[On-policy distillation (Agarwal et al. 2024)](https://arxiv.org/abs/2406.12851)**：on-policy distillation from self-generated mistakes，但需要 reward。SDFT 把 reward 替换成 demonstration-conditioned teacher。
- **[RL's Razor (Shenfeld et al. 2025)](https://arxiv.org/abs/2509.04259)**：on-policy RL forget less。SDFT 是这个 insight 在 demonstration-based setting 的扩展。
- **[DPO (Rafailov et al. 2023)](https://arxiv.org/abs/2305.18290)**：implicit reward from preference。SDFT 是 implicit reward from demonstration + ICL。
- **[Constitutional AI (Bai et al. 2022)](https://arxiv.org/abs/2212.08073)** 和 **[Context Distillation (Snell et al. 2022)](https://arxiv.org/abs/2209.15189)**：context as teacher，但 offline + global context。SDFT 是 instance-wise + on-policy。

---

## 11. 未来方向（个人推测）

1. **SDFT + RL 混合**：先用 SDFT 把 demonstration signal 蒸馏到 weights，再用 RL refinement。SDFT 给 RL 一个更好的 starting point（Figure 5 显示 pass@k 全面提升）。
2. **Non-expert demonstrations**：现在要求 expert demo，但真实世界经常是 noisy / suboptimal demo。能不能用 confidence-weighted SDFT？
3. **Multi-modal extension**：vision-language model 上 in-context learning 也强，SDFT 应该可以直接迁移。
4. **Active learning**：既然 on-policy 是关键，可以主动选择 model 不 confident 的 demo，更 efficient。
5. **More aggressive behavioral change**：论文承认 SDFT 不擅长 fundamental pattern shift。可能需要更复杂的 objective（比如 reverse KL + forward KL 混合）或更强的 prompting。

---

## 总结一句话

**SDFT = "用 in-context learning 当免费 reward model 做 on-policy distillation"**。它绕过了 IRL 的 reward inference 难题，绕过了 SFT 的 off-policy forgetting 问题，绕过了 RL 的 explicit reward 需求，把"持续从 demonstration 学习"这件事变成了一个 stable, scalable 的训练算法。在 7B+ 模型上效果显著，且随 scale 增长。这是 continual learning from demonstrations 的一个真正可工程化路径。

---

**References**:
- Paper: http://idanshenfeld.com/SDFT
- [DAgger (Ross et al. 2011)](https://arxiv.org/abs/1011.0686)
- [TRPO (Schulman et al. 2015)](https://arxiv.org/abs/1502.05477)
- [DPO (Rafailov et al. 2023)](https://arxiv.org/abs/2305.18290)
- [On-policy distillation (Agarwal et al. 2024)](https://arxiv.org/abs/2406.12851)
- [RL's Razor (Shenfeld et al. 2025)](https://arxiv.org/abs/2509.04259)
- [Retaining by doing (Chen et al. 2025)](https://arxiv.org/abs/2510.18874)
- [On-policy distillation (Lu & Thinking Machines Lab 2025)](https://thinkingmachines.ai/blog/on-policy-distillation)
- [KL gradient pitfalls (Tang & Munos 2025)](https://arxiv.org/abs/2506.09477)
- [KL estimator (Amini et al. 2025)](https://arxiv.org/abs/2410.20952)
- [Constitutional AI (Bai et al. 2022)](https://arxiv.org/abs/2212.08073)
- [Context Distillation (Snell et al. 2022)](https://arxiv.org/abs/2209.15189)
- [Cartridges (Eyuboglu et al. 2025)](https://arxiv.org/abs/2506.06266)
- [Kujanpää et al. 2025](https://arxiv.org/abs/2502.08195)
- [Mecklenburg et al. 2024](https://arxiv.org/abs/2404.00213)
- [IRL (Ng et al. 2000)](https://arxiv.org/abs/cs/9905114)
- [RLHF (Ouyang et al. 2022)](https://arxiv.org/abs/2203.02155)
- [KL as Bayesian inference (Korbak et al. 2022)](https://aclanthology.org/2022.findings-emnlp.77/)
- [Knowledge injection (Kujanpää et al. 2025)](https://arxiv.org/abs/2502.08195)
- [ToolAlpaca (Tang et al. 2023)](https://arxiv.org/abs/2306.05301)
- [HuatuoGPT-o1 (Chen et al. 2024)](https://arxiv.org/abs/2412.18925)
- [SciKnowEval (Feng et al. 2024)](https://arxiv.org/abs/2406.09098)
- [Olmo 3 (2025)](https://arxiv.org/abs/2512.13961)
- [Qwen 2.5 (Hui et al. 2024)](https://arxiv.org/abs/2409.12186)
