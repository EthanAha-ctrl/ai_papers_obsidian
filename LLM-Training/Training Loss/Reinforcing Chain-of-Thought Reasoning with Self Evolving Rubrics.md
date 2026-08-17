---
source_pdf: Reinforcing Chain-of-Thought Reasoning with Self Evolving Rubrics.pdf
paper_sha256: 1a0e40ec65e431a5c926891f76801acaf8b7ced07dda98ec540894b9aa1c1bf6
processed_at: '2026-08-11T22:26:59-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 RLCER

## 一句话概括

让 model 自己学会"怎么判断一条推理过程是好是坏"，然后拿这个判断标准去 reward 自己的 reasoning process，而且这个标准还会随着训练自动升级。

---

## 背景故事

现在的 RLVR 训练（比如 DeepSeek-R1、DAPO 这些）有一个很尴尬的情况：你问 model 一道数学题，它输出一长串 CoT 再给个 final answer。你只 check 答案对不对，对就 +1，错就 -1。

**问题在哪**？同一个正确答案，背后可能有 1000 种不同的 reasoning 路径。有的路径是扎实的推导，有的路径是瞎猫撞上死耗子。RL 的 gradient 对这 1000 条路径一视同仁，全部给同样的 reward。

结果就是 model 慢慢学会走捷径——那些恰好撞对答案但推理很烂的 strategy。这就是 paper 里说的 "shortcut drift" 和 "underconstrained"。

**传统解法**是训一个 PRM（Process Reward Model），让人标注每个 reasoning step 好不好。Lightman 的 PRM800k 就是这么干的。但这有两个大麻烦：
1. 标注成本极高，要数学专家一个 step 一个 step 看
2. PRM 训好后是 static 的，但 model 的 CoT 分布在 RL 训练中一直变，PRM 很快就 out of sync 了

---

## RLCER 的 idea

核心 insight 很简单：**model 自己最知道自己写的 reasoning 哪里好哪里差**。

那为什么不让 model 自己来当这个"裁判"？具体做法是让同一个 $\pi_\theta$ 穿两顶帽子：

**帽子一：Reasoner**（普通解题角色）
- 输入 question Q
- 输出 CoT + final answer

**帽子二：Rubricator**（评分标准生成器）
- 输入 question Q + 某条 CoT
- 输出一组 rubric，每个 rubric 是一句话 + 一个分数
- 比如生成这样的东西：
  - ("avoid tangential explorations after interval validation", +0.5)
  - ("uses systematic permutation formula instead of manual listing", +0.8)
  - ("concludes prematurely without double-checking boundary cases", -0.3)

然后用一个 frozen 的 verifier model（就是个判官）去 check：这条 CoT 满不满足每个 rubric？满足就 1，不满足就 0。

把所有 valid rubric 的 satisfaction 结果加权求和，再 normalize 一下，就是 CoT reward。

---

## 最关键的 trick：怎么判断一个 rubric 是好 rubric

rubricator 可能会生成一堆 garbage rubric，比如 "the response contains the letter e"——这种当然没用。

paper 的判断标准很 clever。对同一个 question，采样 N=8 条 rollout，对每个 rubric k，你能拿到两个 binary 向量：

- $\mathbf{v}_k = [1, 0, 1, 1, 0, ...]$ —— 8 条 rollout 各自满不满足 rubric k
- $\mathbf{z} = [1, 0, 0, 1, 1, ...]$ —— 8 条 rollout 各自答得对不对

**一个 rubric 是 valid 的，当且仅当 $\mathrm{corr}(\mathbf{v}_k, \mathbf{z}) > 0.2$ 且 $\mathrm{std}(\mathbf{v}_k) > 0$**

翻译成人话：
- **correlation > 0.2**：满足这个 rubric 的 rollout 更容易答对。说明这个 rubric 确实 capture 了好推理的某个特征
- **std > 0**：不是所有 rollout 都满足（也不是都不满足）。如果 8 条全满足，那这个 rubric 太 trivial，没区分度

这个 gate 就把 "rubric 是否 informative" 变成了一个纯统计问题，完全不需要 human 判断。

---

## Self-evolving 怎么实现的

这才是 paper 的精华。

rubricator 也有自己的 reward：$r_{evolving}^{Rub} = \frac{K_{valid}}{K}$，就是它生成的 rubric 里有多少比例是 valid 的。这个 reward 通过 PPO 反传回 $\theta$，让 rubricator 学会生成更多 valid rubric。

**动态博弈**：
- Reasoner 越来越强 → 本来难满足的 rubric 变得容易满足 → std(rubric) → 0 → rubric 失效
- 失效的 rubric 不再贡献 reward → rubricator 被迫 propose 更难的 rubric
- 更难的 rubric → reasoner 需要更好的 reasoning 才能满足 → reasoner 继续变强

这就是 self-play 的 curriculum learning，难度自动升级。

Figure 6 直接验证了这一点：RLCER 的 rubric-answer correlation 持续上升（rubric 越来越 capture 真正的 reasoning quality），CoT reward 持续下降（rubric 越来越难满足）。对比 ablation（去掉 evolving reward）correlation flat、CoT reward 上升（rubric 被 reasoner 轻松 hack）。

---

## 为什么这个 setup 能 work

最 striking 的实验是 Figure 5：**把 outcome reward 完全去掉，只用 rubric reward 训练**，model 性能依然持续提升。用 random reward 替代 rubric reward 则性能下降。

这说明：通过 correlation gate 筛选出来的 rubric，本身就是个 sufficient 的 supervision signal。你不需要知道答案对不对，只要知道"满足这些 rubric 的 reasoning 更可能对"就够了。

这背后的 intuition 是：rubric 把 outcome reward 这个 sparse signal（只有最终对/错）变成了 dense signal（CoT 每个维度都有 reward）。虽然每个 rubric 单独看是 binary 的，但多个 rubric 加起来就提供了丰富的 gradient。

---

## 实际效果

8B Qwen3 上的主实验（Table 1）：

- AIME2024：34.79 → 37.50（+2.71）
- AIME2025：32.50 → 33.33（+0.83）
- AMC2023：84.53 → 86.41（+1.88）
- GPQA-Diamond：46.56 → 48.77（+2.21）—— 注意这是 graduate level 通用知识问答，但只用 math 数据训练，说明 rubric 学到的是 general reasoning skill

4B 模型上 gain 很小，几乎没动。说明 rubric generation 需要一定 model capacity——小模型自己都分不清什么是好 reasoning，自然提不出好 rubric。

---

## Case Study 看看 rubricator 到底在干啥

Appendix C 的例子：问 "2013 之后 10000 之前有多少个连续四位数字年份"，ground truth = 149，model 答 = 131（错了）。

Rubricator 生成的 rubric 包括：
- "Employs ad-hoc manual listing for small digit sets instead of systematic permutation formula application, increasing error risk" —— 直接定位了错误根源：model 在 d=0, d=1 这两个小 case 用了手动枚举而不是公式，所以算错
- "Avoids redundant recounting by categorizing sets with minimum permutation > 2013 upfront" —— 给出改进 hint：应该先按 minimum permutation 分类，避免重复计数

这比 PRM 强太多了。PRM 只能告诉你"step 3 有 65% 概率是错的"，RLCER 的 rubricator 能告诉你"step 3 错在哪、应该怎么改"。这是 actionable feedback。

---

## 工程上要注意的坑

1. **Cold-start 必要性**：Qwen3-8B-Base 直接当 rubricator 完全不行，根本不会按 format 输出。作者从 Doubao-Seed-1.6-thinking reject sample 了 40k 数据（20k math + 20k rubric）做 SFT 才稳定。

2. **Prompt trick**：rubricator prompt 显式要求"当前 response 的 score 应该低于中间值"，即强迫 rubricator 生成那些当前 response **没满足**的 rubric。不然 rubricator 会偷懒生成一堆 trivially satisfied 的 rubric 骗自己。

3. **Compute 开销**：每个 question 要跑 N=8 次 reasoner rollout + N 次 rubricator rollout + N×K 次 verifier call。比 vanilla RLVR 至少贵 3-4 倍。

4. **GRPO 不能用**：rubricator 的 context（Q + CoT）每条都不一样，无法 group 到同一 context 下做 GRPO 的 relative advantage。只能回到 PPO + critic，多了 critic 训练成本。

5. **Context length**：reasoner 的 output 是 rubricator 的 input，所以 prompt max length (16384) > response max length (12288)，两者加起来不能超过 32k。

---

## 我觉得几个值得讨论的问题

**1. Correlation gate 的统计可靠性**：N=8 算 correlation 真的够吗？Pearson correlation 在 N=8 下方差很大，真实 corr=0 的情况下观察到 |corr|>0.2 的概率不低。会不会放进不少 false positive rubric？这是 reward noise 的来源。paper 没有做这个 sensitivity analysis。

**2. Correlation ≠ Causation**：一个 rubric 和答案正确性相关，可能只是因为两者都被"model 整体能力"共同驱动，而非该 rubric 描述的 reasoning property 本身导致正确。比如"response 超过 500 字"这种 rubric 可能也和正确性正相关（更强的 model 写得更长），但 reward 这种 rubric 等于鼓励注水。这个 confounding 问题 paper 没讨论。

**3. Verifier 是 frozen 的**：如果 verifier 本身对某些 reasoning style 有 bias，所有 rubric 的 satisfaction judgment 都会受污染。paper 没做 verifier robustness 的 ablation。verifier 也参与 self-evolving 可能更好，但会引入 GAN-like 的训练不稳定。

**4. Non-verifiable domain 的 extension 是个悖论**：paper 说未来要扩展到 non-verifiable domain，但 validity gate 本身依赖 $\mathbf{z}$（答案正确性向量）。non-verifiable domain 算不出 $\mathbf{z}$，整个 self-evolving 机制就崩了。要么找到替代的 $\mathbf{z}$ signal（比如 human preference），要么 rubric evolution 就失去了方向。

**5. 为什么 4B 上没效果**：这个其实挺重要的。如果只有 8B+ 才能受益，说明 rubric generation 本身是个 capability threshold task。那对真正的小模型部署场景，这个方法基本没用。需要研究怎么让小模型也能用上 rubric——可能需要蒸馏大模型的 rubricator 能力。

---

## 我的 takeaway

RLCER 最有意思的地方不是 performance gain（+2 个点其实在 noise 范围），而是它 hint 了一个新的 paradigm：

**reasoning quality 的定义可以是 learned 的，而非 human-defined 的**

传统做法是 expert 写规则告诉 model 什么是好推理。RLCER 表明，model 可以通过"什么与正确性相关"这一弱信号，自主 discover 出 reasoning quality 的 criteria，而且这些 criteria 还能随着能力提升自动升级。

这个 idea 的延伸空间很大：
- 训练完的 rubric 集合可以当 "discovered reasoning principles" 来分析，看 model 到底学到了什么 strategy
- 能否做 hierarchical rubric（high-level principle → low-level checklist）
- 能否把 math 上 evolve 出的 rubric transfer 到 code / science reasoning
- 能否让 verifier 也参与 self-evolving（虽然会不稳定）

总的来说，这篇 paper 把 "self-play" 这个 idea 从"对答案"扩展到了"对推理过程"，是个很有审美的工作。虽然工程上还比较粗糙，compute 开销大、限制多，但 concept 上指向了一个值得探索的方向。

---

Reference:
- Paper: https://alphalab-ustc.github.io/rlcer-alphalab/
- DAPO: https://arxiv.org/abs/2503.14476
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- PRM800k: https://arxiv.org/abs/2305.20050
- Self-Rewarding LM: https://arxiv.org/abs/2401.10020
- Absolute Zero: https://arxiv.org/abs/2505.03335

---

# RLCER: Self-Evolving Rubrics for CoT Supervision 深度解析

## 1. Motivation 与问题定位

这篇 paper 来自 ByteDance Seed / NUS / USTC，发表于 2026 年 2 月。核心要解决的是当前 RLVR (Reinforcement Learning with Verifiable Rewards) 范式中的一个根本性 underconstrained 问题。

**当前 RLVR 的痛点**：DeepSeek-R1 / DAPO 这类工作主要奖励 final answer 的 correctness (公式 2 中的 $r = \psi(\mathbb{I}(\mathcal{A}, \hat{\mathcal{A}}))$)，CoT 本身几乎不获得 explicit supervision。这意味着：

- **Equivalence class 问题**：对于同一个 final answer，无数条不同的 reasoning trajectory 拿到完全相同的 reward，policy gradient 对这些 trajectory 无法区分。
- **Shortcut drift**：优化器倾向于找到"brittle strategies"——那些恰好撞对答案但 reasoning 脆弱的路径，因为 outcome reward 无法惩罚这类策略。
- **Static RM 的局限**：Lightman et al. 的 PRM800k 路线需要大量 fine-grained human annotation，且 RM 训练好后是 static 的，但 policy 的 CoT distribution 在 training 过程中持续 shift，导致 non-stationary supervision 和 reward hacking。

RLCER 的核心 idea：让 policy model $\pi_\theta$ 自己 propose rubrics（自然语言形式的 CoT 评估标准），并让这些 rubrics 在 training 过程中 self-evolve，evolving 的方向由 "rubric satisfaction 与 final answer correctness 的 correlation" 来 shape。

Project page: https://alphalab-ustc.github.io/rlcer-alphalab/

---

## 2. Methodology 架构解析

### 2.1 双角色单 policy 设计

RLCER 的关键 design choice 是让同一个 $\pi_\theta$ 通过不同 prompt 扮演两个角色，这沿袭了 SPICE / Absolute Zero / Self-Rewarding LM 这条 self-evolving 研究线。

**Reasoner** $\pi_\theta^{Rea}(\cdot) = \pi_\theta(\cdot \mid \mathcal{P}^{Rea})$：

$$\hat{\mathcal{C}}, \hat{\mathcal{A}} \sim \pi_\theta^{Rea}(\cdot \mid \mathcal{Q}) \tag{4}$$

- $\hat{\mathcal{C}}$：generated chain-of-thought
- $\hat{\mathcal{A}}$：generated final answer
- $\mathcal{Q}$：input question

**Rubricator** $\pi_\theta^{Rub}(\cdot) = \pi_\theta(\cdot \mid \mathcal{P}^{Rub})$：

$$\hat{\mathcal{R}} \sim \pi_\theta^{Rub}(\cdot \mid \mathcal{Q}, \hat{\mathcal{C}}), \quad \hat{\mathcal{R}} \triangleq \{\hat{\tau}_k\}_{k=1}^{K}, \quad \hat{\tau}_k \triangleq (\hat{c}_k, \hat{s}_k) \tag{5}$$

变量含义：
- $\hat{\mathcal{R}}$：rubricator 输出的 rubric 集合
- $K$：每次生成的 rubric 数量
- $\hat{\tau}_k$：第 k 个 rubric，是一个 (criterion, score) tuple
- $\hat{c}_k$：自然语言 criterion，例如 "avoids tangential explorations post-interval validation (3-∞) by focusing on count aggregation"
- $\hat{s}_k \in \mathbb{R}$：该 rubric 的重要性分数，**可正可负** —— 这个设计很关键，意味着 rubric 可以是"应该满足"也可以是"应该避免"

**Verifier** $\pi_\phi$：一个 frozen 的 fine-tuned 模型（用 Qwen3-4B-Base 蒸馏自 Doubao-Seed-1.6-thinking），输入 $(\hat{c}_k, \hat{\mathcal{C}})$，输出 binary judgment $\{0, 1\}$，判断 CoT 是否满足该 rubric。Verifier 不参与 gradient 更新，相当于一个 "rubric satisfaction oracle"。

这个三角色架构（reasoner / rubricator / verifier）让人联想到 GAN 的 generator-discriminator 结构，但有本质区别：这里 verifier 是 frozen 的，rubricator 才是被 evolve 的 "adversarial" component，且 rubricator 与 reasoner 共享参数。

### 2.2 CoT Reward 计算

公式 (6) 是核心：

$$r_{cot}^{Rea} = \mathrm{norm}\left(\sum_{\hat{\tau}_k \in \hat{\mathcal{R}}_{valid}} \pi_\phi(\hat{c}_k, \hat{\mathcal{C}}) \cdot \hat{s}_k\right) \tag{6}$$

逐项拆解：
- $\hat{\mathcal{R}}_{valid}$：valid rubric 子集（validity 定义见下一节）
- $\pi_\phi(\hat{c}_k, \hat{\mathcal{C}}) \in \{0, 1\}$：verifier 判断当前 CoT 是否满足第 k 个 rubric
- $\hat{s}_k$：rubric 的 score（带符号）
- $\mathrm{norm}(\cdot)$：min-max normalization，$\mathrm{norm}(x) = (x - \text{MinValue})/(\text{MaxValue} - \text{MinValue})$

这里 MinValue 和 MaxValue 是在当前 valid rubric 集下能达到的最小/最大 aggregated score。直觉上：如果所有 rubric 都被满足且 score 全正，norm 后 = 1；如果都没被满足或 score 全负，norm 后 = 0。这把 reward 压到 [0, 1] 区间，与 outcome reward $\{-1, +1\}$ 的尺度匹配。

### 2.3 Validity 判断 —— Self-evolving 的关键

这是整篇 paper 最有 insight 的部分。一个 rubric $\hat{\tau}_k$ 被认为是 "valid"（即作为 reward signal 是 informative 的），需要同时满足两个条件：

**条件 (i) Positive correlation**：
$$\mathrm{corr}(\mathbf{v}_k, \mathbf{z}) > \alpha \tag{7}$$

**条件 (ii) Discriminativeness**：
$$\mathrm{std}(\mathbf{v}_k) > 0 \tag{7}$$

其中，对同一个问题 $\mathcal{Q}$ 采样 N 个 rollout：

$$\mathbf{v}_k \triangleq [\pi_\phi(\hat{c}_k, \hat{\mathcal{C}}_0), \ldots, \pi_\phi(\hat{c}_k, \hat{\mathcal{C}}_N)]$$

$$\mathbf{z} \triangleq [\mathbb{I}(\mathcal{A}, \hat{\mathcal{A}}_0), \ldots, \mathbb{I}(\mathcal{A}, \hat{\mathcal{A}}_N)]$$

变量：
- $\mathbf{v}_k$：长度为 N 的 binary 向量，第 i 个元素表示第 i 个 rollout 的 CoT 是否满足 rubric k
- $\mathbf{z}$：长度为 N 的 binary 向量，第 i 个元素表示第 i 个 rollout 的答案是否正确
- $\mathrm{corr}(\cdot, \cdot)$：相关系数（应该是 Pearson 或 point-biserial，因为两个都是 binary）
- $\alpha = 0.2$：correlation threshold

**Intuition**：这个设计本质上是把 "rubric 是否有信息量" 转化为一个可计算的统计量。如果一个 rubric 满足与否和答案正确与否强相关，那么 reward 这个 rubric 的 satisfaction 等价于间接 reward 答案正确性，但提供了更细粒度的 gradient signal——因为 rubric 是关于 CoT process 的，而非 final outcome。这避免了 outcome reward 的 sparse 问题，同时通过 correlation gate 防止 rubricator 生成无关的、甚至有害的 rubric。

条件 (ii) 防止 "trivially satisfied rubric"：如果一个 rubric 在所有 rollout 上都满足（std=0），那它没有区分度，无法提供 gradient signal。

### 2.4 Self-evolving Reward

公式 (10) 是 rubricator 的 evolving reward：

$$r_{evolving}^{Rub} = \frac{K_{\mathrm{valid}}}{K} \tag{10}$$

变量：
- $K$：rubricator 生成的 rubric 总数
- $K_{\mathrm{valid}} \triangleq |\{\hat{\tau}_k^{valid}\}|$：其中 valid rubric 的数量

这个 reward 鼓励 rubricator 生成更多 valid rubric。配合 format reward (公式 11, 12)：

$$r^{Rub} = r_{evolving}^{Rub} + r_{format}^{Rub} \tag{12}$$

**为什么这个 design 能实现 self-evolving**：rubricator 的 reward 直接与 "其生成的 rubric 是否与答案正确性相关" 挂钩。随着 training 进行，rubricator 会学到生成那些真正 capture "good reasoning property" 的 rubric，而 rubric 的难度也会自然提升——因为 reasoner 在变强，原本 discriminative 的 rubric 会逐渐被所有 rollout 满足（std → 0），失去 validity，迫使 rubricator propose 更 challenging 的 rubric。这就是 Figure 6a 中 correlation 上升、Figure 6b 中 $r_{cot}^{Rea}$ 下降的机制——rubric 越来越难满足。

### 2.5 完整 Reasoner Reward

公式 (8) + (9)：

$$r_{outcome}^{Rea} = \begin{cases} 1, & \mathrm{is\_equiv}(\mathcal{A}, \hat{\mathcal{A}}) \\ -1, & \text{Otherwise} \end{cases} \tag{8}$$

$$r^{Rea} = r_{outcome}^{Rea} + r_{cot}^{Rea} \tag{9}$$

这里 outcome reward 用 $\{-1, +1\}$ 而非 $\{0, 1\}$，这是 DAPO 的设计——负 reward 对错误答案施加更强的惩罚梯度。CoT reward 是 [0, 1] 的 auxiliary signal。两者相加后 reasoner 的 total reward 范围是 $[-1, 2]$。

### 2.6 Optimization Objective

公式 (13) 是 PPO clip 风格的双角色联合优化：

$$\mathcal{I}(\theta) = \mathbb{E}_{(Q,A) \sim \mathcal{D}^{Rea}, o \sim \pi_{\theta_{old}}^{Rea}(\cdot | Q)}[\min(\rho_t(\theta)\hat{A}_t^{Rea}, \mathrm{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t^{Rea})]$$
$$+ \mathbb{E}_{(Q,\hat{c}) \sim \mathcal{D}^{Rub}, o \sim \pi_{\theta_{old}}^{Rub}(\cdot | Q, \hat{c})}[\min(\rho_t(\theta)\hat{A}_t^{Rub}, \mathrm{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t^{Rub})] \tag{13}$$

变量：
- $\rho_t(\theta) = \pi_\theta(o) / \pi_{\theta_{old}}(o)$：importance sampling ratio，新旧 policy 在 token o 上的概率比
- $\hat{A}_t^{Rea}, \hat{A}_t^{Rub}$：分别从 $r^{Rea}$ 和 $r^{Rub}$ 计算的 advantage（用 GAE 估计）
- $\epsilon = 0.2$：clip ratio
- $\mathcal{D}^{Rea}, \mathcal{D}^{Rub}$：两个角色的 data distribution

两个 expectation 项分别对应 reasoner 和 rubricator 的 PPO loss，**梯度累加**到同一组参数 $\theta$ 上。这与 multi-agent RL 中 shared backbone 的设计类似。

**重要 note**：paper 明确说 GRPO 不适用，因为 rubricator 在不同 context 下操作，无法 group 到同一 context 下做 baseline 减除（GRPO 依赖同 context 多 rollout 的 relative advantage）。这迫使作者回到 PPO + critic 的 setup，增加了 critic 的训练成本（learning rate for critic = 1e-5，比 actor 的 1e-6 高 10 倍）。

---

## 3. 实验数据深度分析

### 3.1 主实验结果 (Table 1)

8B 模型上 RLCER vs RLVR 的对比：

| Dataset | RLVR | RLCER | Gain |
|---------|------|-------|------|
| AIME2024 | 34.79 | 37.50 | +2.71 |
| AIME2025 | 32.50 | 33.33 | +0.83 |
| AMC2023 | 84.53 | 86.41 | +1.88 |
| GPQA-Diamond | 46.56 | 48.77 | +2.21 |
| SuperGPQA-Eng | 42.94 | 45.00 | +2.06 |
| SuperGPQA-Med | 38.31 | 36.50 | -1.81 |
| SuperGPQA-Sci | 48.81 | 50.25 | +1.44 |

观察：
1. 数学任务上 RLCER 一致优于 RLVR，AIME2024 提升 2.71 个点最显著
2. **Generalization**：仅用 DAPO-Math-17k 训练，GPQA-Diamond（graduate-level Q&A）也提升 2.21，说明 rubric supervision 学到的是 general reasoning skill，而非 math-specific trick
3. **Scale effect**：4B 模型上 gain 很小（AIME2024 持平，AIME2025 仅 +0.42），8B 上 gain 明显。这暗示 rubric generation 需要一定 model capacity 才能产生 informative criteria——小模型可能 generate 的 rubric 太 trivial 或太 noisy
4. SuperGPQA-Med 下降 1.81 是唯一负面结果，可能是 medical domain 的 reasoning pattern 与 math rubric 偏差大

### 3.2 RQ1: Rubric-only Training (Figure 5)

这是最 striking 的 ablation：**完全去掉 outcome reward，只用 $r^{Rea} = r_{cot}^{Rea}$ 训练**，模型性能依然持续提升。对照实验用 random reward（[0,1] 均匀分布）则性能下降或停滞。

这个结果的深层含义：self-proposed rubrics 通过 correlation gate 后，本身就是一个 sufficient supervision signal。这挑战了 "verifiable reward 是 RLVR 必要条件" 的假设，hinting at 一条通往 non-verifiable domain 的路径。

### 3.3 RQ3: Self-evolving 机制分析 (Figure 6)

Figure 6a 显示 $\mathrm{corr}(\mathbf{v}_k, \mathbf{z})$ 在 RLCER training 中持续上升，而 ablation（无 $r_{evolving}^{Rub}$）保持 flat。Figure 6b 显示 RLCER 的 $r_{cot}^{Rea}$ 持续下降（rubric 越来越难满足），ablation 则上升（rubric 被 reasoner 轻松满足，饱和）。

这组实验直接验证了 self-evolving 的设计意图：**rubricator 在 reasoner 进步的同时被迫 propose 更难的 rubric，形成 curriculum**。这与 AlphaGo 的 self-play 机制有结构相似性——对手变强迫使自己变强。

### 3.4 RQ4: Rubric as In-prompt Hint (Figure 7)

将训练好的 rubricator 生成的 rubric 作为 hint 注入 reasoner 的 prompt，inference 性能进一步提升。AIME 上 BoN=16 的提升尤为明显。这说明 rubric 不仅作为 training signal 有效，还 capture 了某种 "reasoning strategy" 的 explicit knowledge，可以作为 test-time reasoning guide。

---

## 4. 训练细节与工程考量

### 4.1 Cold-start Strategy

由于 Qwen3-8B-Base 直接作为 rubricator 无法稳定 follow format，作者从 Doubao-Seed-1.6-thinking reject sampling 了 40k 数据（20k math trajectory + 20k rubricator trajectory）做 SFT。这是一个关键的工程细节——**rubricator role 需要 cold-start 才能稳定生成 parseable rubric**。

一个有意思的 trick（Appendix B）：rubricator prompt 显式要求 "current response score falls below the middle value"，即强迫 rubricator 生成那些当前 response 没满足的 rubric。这避免了 rubricator 生成 trivially-satisfied rubric 的 degenerate solution。

### 4.2 RL Hyperparameters

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO |
| Max prompt length | 16384 |
| Max response length | 12288 |
| Overlong buffer | 4096 |
| Rollout N | 8 |
| Train batch size | 32 |
| Mini batch size | 64 |
| Actor LR | 1e-6 |
| Critic LR | 1e-5 |
| Clip ratio | 0.2 |
| Max steps | 1500 |

注意 prompt max length (16384) > response max length (12288)，因为 reasoner 的 output 会作为 rubricator 的 input prompt，两者之和不能超过 32k token。这意味着 rubricator 的 context 实际上很长（question + full CoT），compute 开销显著。

### 4.3 Rollout 与 Reward 计算流程

Algorithm 1 的核心循环：
1. 对 question $\mathcal{Q}$，reasoner 采样 N=8 个 rollout $\{(\hat{\mathcal{C}}_n, \hat{\mathcal{A}}_n)\}$
2. 对每个 rollout，rubricator 采样一组 rubric $\hat{\mathcal{R}}_n$
3. 计算 outcome reward（公式 8）
4. 对每个 rubric，计算 $\mathbf{v}_k$ 和 $\mathbf{z}$ 的 correlation，筛选 valid rubric
5. 计算 $r_{cot}^{Rea}$（公式 6）和 $r_{evolving}^{Rub}$（公式 10）
6. 分别计算 advantage，联合更新 $\theta$

这里 N=8 是 validity 判断的最小 sample size。$\mathrm{corr}$ 在 N=8 上的统计可靠性值得商榷——Pearson correlation 在小样本下方差很大，$\alpha=0.2$ 的 threshold 可能不够严格。这是潜在的 reward noise 来源。

---

## 5. Intuition Building 与 Critical Analysis

### 5.1 为什么 Self-evolving Rubrics 比 Static PRM 更好

传统 PRM (Process Reward Model) 的失败模式：
1. **Distribution shift**：PRM 在 human-labeled data 上训练，但 policy 的 CoT distribution 在 RL 中持续 shift，PRM 的 judgment 逐渐失准
2. **Reward hacking**：policy 学会 exploit PRM 的 bias（例如生成长但空洞的 step 来骗高分）
3. **Saturation**：PRM 对所有 rollout 给出相似分数，失去区分度

RLCER 的解法：
1. Rubricator 与 reasoner **共享参数**，自动跟随 distribution shift
2. Validity gate（correlation + std）**自动淘汰** 被 hack 的 rubric——如果 policy 学会 trivially satisfy 某 rubric，std→0，该 rubric 失效
3. Self-evolving reward 迫使 rubricator 持续 propose 新的、更难的 rubric

这本质上是一个 **closed-loop self-play**：reasoner 和 rubricator 在同一个 $\theta$ 下互相 drive 进步，类似 GAN 但共享 generator/discriminator 参数。

### 5.2 与 Constitutional AI / Self-Rewarding LM 的区别

- **Constitutional AI** (Anthropic)：用 predefined principles 让 model self-critique，principles 是 human-written 且 static 的
- **Self-Rewarding LM** (Yuan et al.)：model 作为 judge 评估 response 质量，但 judge prompt 是 general 的
- **RLCER**：rubrics 是 model **自主生成**的，且通过 correlation gate **自主筛选**，无需 human 指定评估维度

RLCER 更接近 "meta-learning the evaluation criteria"——model 不仅学习解决问题，还学习"什么样的 reasoning 是好的"。

### 5.3 潜在问题与 Limitations

1. **Verifier bottleneck**：verifier $\pi_\phi$ 是 frozen 的，如果 verifier 本身有 bias（例如对某些 reasoning style 系统性误判），所有 rubric 的 satisfaction judgment 都会受污染。Paper 没有做 verifier robustness 的 ablation。

2. **Correlation ≠ Causation**：validity gate 用 correlation 筛选 rubric，但相关不等于因果。一个 rubric 可能与正确性相关只是因为两者都由 model capability 共同驱动（confounder），而非该 rubric 描述的 property 本身导致正确。这可能导致 rubric reward 变成 outcome reward 的 noisy proxy，而非真正的 process supervision。

3. **Compute overhead**：每个 question 需要 N=8 个 reasoner rollout + 8 组 rubricator rollout + 8×K 次 verifier call。相比 vanilla RLVR（只需 N 个 rollout + outcome check），compute 至少翻 3-4 倍。Paper 在 limitations 中承认了这一点。

4. **N=8 的 statistical power**：在 N=8 上计算 correlation，即使真实 correlation=0，观察到 |corr|>0.2 的概率不低（约 30%+）。这意味着 validity gate 可能放过相当多的 false positive rubric。

5. **Domain limitation**：仅在 math (verifiable) 上验证。Paper 声称这是通往 non-verifiable domain 的路径，但 validity gate 本身依赖 $\mathbf{z}$（答案正确性），在 non-verifiable domain 上 $\mathbf{z}$ 无法计算，整个 self-evolving 机制失效。这是核心矛盾——想用 verifiable domain 训练出的 rubricator 泛化到 non-verifiable domain，但 evolving 信号本身依赖 verifiability。

### 5.4 与近期工作的关联

- **Rubrics as Rewards** (Gunjal et al., 2025, https://arxiv.org/abs/2507.17746)：predefined rubric for non-verifiable domain，RLCER 借鉴了 rubric-based reward 的 idea 但让 rubric self-evolve
- **Absolute Zero** (Zhao et al., 2025, https://arxiv.org/abs/2505.03335)：single model self-play reasoning with zero data，RLCER 的 multi-role single-policy 架构与其一脉相承
- **SPICE** (Liu et al., 2025, https://arxiv.org/abs/2510.24684)：self-play in corpus environments，类似的双角色 self-evolving 思路
- **PRM (Let's Verify Step by Step)** (Lightman et al., 2024, https://arxiv.org/abs/2305.20050)：RLCER 想替代的 static PRM 路线
- **DeepSeek-R1** (https://arxiv.org/abs/2501.12948)：outcome-centric RLVR 的代表，RLCER 的 baseline
- **DAPO** (Yu et al., 2025, https://arxiv.org/abs/2503.14476)：paper 使用的 RL 框架和 dataset 来源

---

## 6. Case Study 解读 (Appendix C)

Paper 给了一个具体例子（Figure 11-12）：问题是"2013 年之后、10000 年之前还有多少个由四个连续数字组成的年份"，ground truth = 149，reasoner 预测 = 131（错误）。

Rubricator 生成的 rubric 包括：
- "Employs ad-hoc manual listing for small digit sets instead of systematic permutation formula application, increasing error risk" —— 直接指出了 reasoner 在 d=0 和 d=1 subset 上用 manual listing 导致错误的 root cause
- "Avoids redundant recounting by categorizing sets with minimum permutation > 2013 upfront, reducing per-set analysis effort" —— 给出了更高效的解题 hint

这个 case 展示了 rubricator 的"debugging"能力：它不仅能判断 CoT 好坏，还能**定位具体错误并 propose 改进方向**。这是 static PRM 做不到的——PRM 只给 step-level scalar score，无法生成 actionable feedback。

---

## 7. 对未来的启示

RLCER hint 了一个有趣的方向：**reasoning quality 的定义本身可以是 learned 的**。传统观点认为"好 reasoning"需要 human expert 定义（如 PRM800k 的标注），RLCER 表明 model 可以通过 "什么与正确性相关" 这一信号自主 discover reasoning quality 的 criteria。

这引出几个值得探索的方向：

1. **Rubric interpretability**：训练后的 rubric 集合可以作为一种 "discovered reasoning principles" 的 dataset，用于分析 model 学到了什么样的 reasoning strategy
2. **Cross-domain rubric transfer**：math 上 evolve 出的 rubric 能否 transfer 到 code / science reasoning？这测试 rubric 是否 capture 了 domain-agnostic 的 reasoning property
3. **Rubric hierarchy**：当前 rubric 是 flat 的，能否引入 hierarchical rubric（high-level principle → low-level checklist）？
4. **Online verifier updating**：如果 verifier 也参与 self-evolving（而非 frozen），能否进一步提升？这会引入 GAN-like 的训练不稳定性，但可能解决 verifier bottleneck

---

## 8. 总结

RLCER 的核心贡献是把 CoT supervision 从 "human-defined + static" 转变为 "self-proposed + self-evolving"，通过 correlation-based validity gate 和 evolving reward 实现了 closed-loop self-improvement。技术上的精巧之处在于：

- **Validity gate** 把 "rubric 是否 informative" 转化为可计算的统计量，无需 human judgment
- **Shared parameter** 让 rubricator 自动跟随 reasoner 的 distribution shift
- **Saturation-driven curriculum** 通过 std→0 机制自然淘汰被满足的 rubric，驱动 rubric 难度上升

代价是 compute overhead、对小模型效果有限、以及 validity gate 对 verifiable reward 的依赖（限制了向 non-verifiable domain 的直接扩展）。但作为一个 "autonomous CoT supervision" 的 proof of concept，它开辟了一条值得深入研究的路径。

Reference links:
- Paper project page: https://alphalab-ustc.github.io/rlcer-alphalab/
- DAPO (RL framework): https://arxiv.org/abs/2503.14476
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Let's Verify Step by Step (PRM): https://arxiv.org/abs/2305.20050
- Self-Rewarding LM: https://arxiv.org/abs/2401.10020
- Absolute Zero: https://arxiv.org/abs/2505.03335
- Rubrics as Rewards: https://arxiv.org/abs/2507.17746
- SPICE: https://arxiv.org/abs/2510.24684
- verl (RL framework): https://github.com/volcengine/verl
- Qwen3: https://arxiv.org/abs/2505.09388
