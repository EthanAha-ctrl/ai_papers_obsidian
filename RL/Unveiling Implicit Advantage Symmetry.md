---
source_pdf: Unveiling Implicit Advantage Symmetry.pdf
paper_sha256: ac09ae48cc33e518d3da7f09cf19ba0066518fe56ba9e2c3b309bd5c9398291d
processed_at: '2026-08-12T20:28:10-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话总结

GRPO 训练 LLM 推理，本质上就是在做 SFT，只是给每个 token 的 cross-entropy loss 乘了一个 advantage 权重。这个权重有一个数学上的"对称性"毛病，导致两个问题：模型不去探索新解法，模型也学不会难题。这篇 paper 用一个几乎零成本的小 trick 把这两个毛病修了。

---

## 先从"GRPO 到底在干嘛"说起

你给模型一个数学题，让它采样 8 条回答（一个 group）。有 3 条答对了，5 条答错了。GRPO 给答对的每条一个正 advantage，答错的每条一个负 advantage，然后用这些 advantage 当权重去放大/压低对应 token 的梯度。就这么简单。

剥掉 PPO 的 clip、KL、importance sampling 那些工程细节，GRPO 的梯度更新长这样：

> **每条 rollout 的每个 token，都做一个 SFT 的 cross-entropy 更新，只是乘了一个标量权重 = advantage × importance ratio。**

importance ratio 被 clip 钉住了，所以真正起作用的那个权重就是 advantage。advantage 是正的，就相当于"使劲学这条 trajectory"；advantage 是负的，就相当于"使劲躲开这条 trajectory"。

这就是论文 Eq. 4 想说的全部。理解了这一点，后面的分析就都是纯算术了。

---

## 毛病一：没采样到的解法，梯度永远是零

GRPO 的 advantage 是这样算的：把 8 条 rollout 的 reward（0 或 1）减去均值，除以标准差。这个操作有一个数学后果——**所有 advantage 加起来严格等于零**。

把答对的 advantage 加起来，和答错的 advantage 加起来，绝对值完全相等。正的推力 = 负的拉力，完美对称。

这个对称性看起来很优雅，实际上很要命。论文 Theorem 1 在一个抽象的 "behavior space" 里推导了任意一条潜在 trajectory 的 logits 更新公式，结果分三种情况：

- 采样到的、答对的 → logits 往上推
- 采样到的、答错的 → logits 往下压
- **没采样到的 → logits 完全不动**

为什么不动？因为 advantage 之和等于零，公式里那个"未采样路径的更新项"被乘以零消掉了。

这就解释了业界最近吵得很凶的一个现象：GRPO 训完之后，Pass@1 涨了不少，但 Pass@256 居然比 base model 还低。意思是模型在"把已知正确答案的概率推高"这件事上很厉害，但它**完全没有学到新的正确解法**。它只是在 base model 的采样支持域内做概率重新分配，没扩展边界。论文在 AIME2025 上实测验证了这一点：GRPO Pass@256 = 40.8，base model 46.7，训练完反而退步了。

---

## 毛病二：中等难度的题吃掉了所有训练信号

再看 sample level。一道题，采样 8 次，成功 $p$ 次。论文 Theorem 2 算了一下这个 group 内所有 advantage 绝对值之和，结果是：

> $2G\sqrt{p(1-p)}$

这是一个关于 $p=0.5$ 对称的钟形曲线。$p=0.5$（中等难度）的时候最大，$p=0.25$（难题）和 $p=0.75$（简单题）的时候一样大。

也就是说，GRPO 自动给中等难度的题分配最多的梯度。难题和简单题，只要离 $p=0.5$ 一样远，拿到的训练信号就一样多。

但训练是一个动态过程。早期模型弱，batch 里大量简单题（$p$ 高）；后期模型强了，难题变少了但更值得学。GRPO 的这个钟形权重是静态的，不管你在哪个训练阶段，都按同一个对称分配信号。结果就是：早期在难题上浪费信号（模型还不会做，学不动），后期在简单题上浪费信号（模型已经会了，不用学）。

---

## 怎么破？两组对照实验

论文最精彩的部分是做了两组"故意打破对称性"的对照实验，直接看因果。

**第一组：打破 group 级对称**

把答对 trajectory 的 advantage 放大 10 倍，或者缩小 10 倍，看会怎样。

- 放大正确 advantage（Positive-Dominant）：模型很快 entropy collapse，过度锐化已知正确路径，Pass@k 在大 k 时严重掉队。
- **缩小正确 advantage（Negative-Dominant）：Pass@k 全面超越 GRPO，AIME 上 Pass@256 从 40.8 拉到 60.0，超过 base model。** entropy 持续上升，说明模型在真的探索新解法。

但有个坑：Negative-Dominant 跑 10 次有 3 次在训练后期突然 collapse。原因是持续压制正确 advantage，会让一些"过度自信的错误解法"在后期反超正确解法，整个模型崩掉。

直觉很清楚：**抑制正确路径能换来探索，但一直抑制会出事。需要动态地、在后期慢慢把抑制关掉。**

**第二组：打破 sample 级对称**

把 advantage 按 $\sqrt{p}$ 或 $\sqrt{1-p}$ 重新缩放，让难题或简单题拿更多信号。

- Hard-Focused 在最难的 AIME 上最好，Easy-Focused 在简单的 AMC23/MATH 上略好。
- 没有全局赢家。
- 但看训练动态（batch 内正确回答数量随训练步数的变化）：**Easy-Focused 早期收敛最快**，**Hard-Focused 后期才能拉开差距**。

这就是经典的 curriculum learning 直觉：**先学简单的，再学难的。** 早期模型连格式都不会，给它难题是浪费；后期模型基本会了，给它简单题也是浪费。

---

## A-GRAE：一个极简的修正

论文提出的 A-GRAE 做了两件事，都只需要一个超参 $\alpha$。

**第一件：动态衰减正确 advantage（ASS）**

用一个 batch 内的平均 reward $\omega_s$ 当训练阶段指示器。模型弱的时候 $\omega_s$ 低，把正 advantage 乘一个小于 1 的系数（温和版的 Negative-Dominant，鼓励探索）；模型强了 $\omega_s$ 接近 1，系数自动回到 1（退化为标准 GRPO，锁定学到的解）。

负 advantage 永远不衰减，保证错误解法始终被压住。这就解决了 Negative-Dominant 的 collapse 风险——10 次重复实验，ASS 是 0 次 collapse，Negative-Dominant 是 3 次。

**第二件：动态切换难度焦点（DDAS）**

还是用 $\omega_s$。早期 $\omega_s$ 小，主要走 Easy-Focused 分支；后期 $\omega_s$ 大，主要走 Hard-Focused 分支。两支的权重是 $\omega_s/2$ 和 $(1-\omega_s)/2$，自动平滑过渡。

这两个加在一起，就是一个由 reward 信号驱动的隐式 curriculum：早期探索 + 学简单的，后期 exploit + 学难的。整个过程不需要外挂 scheduler，不需要额外标注，不需要改模型结构，就是改一下 advantage 的计算方式。

---

## 结果怎么样

七个 benchmark，文本数学（MATH、AMC23、AIME2025）加多模态（几何、医学影像），套在 GRPO、DAPO、Dr.GRPO 上，**一致提升**。

最漂亮的数字在 AIME2025（最难的数据集）：
- GRPO Pass@256 = 40.8（低于 base 46.7，典型 capability boundary shrinkage）
- GRPO + A-GRAE Pass@256 = 56.7（超过 base，说明真的学到了新解法）
- Dr.GRPO + A-GRAE Pass@256 = 56.7，DAPO + A-GRAE = 60.0

Ablation 也干净：sample-level 修正主要涨 Pass@1，group-level 修正主要涨 Pass@k，两个正交，加一起最强。

---

## 这篇 paper 真正的贡献是什么

表面上看是一个 trick，但底下是一个**重新理解 GRPO 的框架**：把 GRPO 看成 reweighted SFT，然后发现 advantage 的归一化带来两个对称性瓶颈。这个框架还能解释一堆现有工作——W-REINFORCE 只破 group 级对称（所以 Pass@k 强 Pass@1 弱），GRPO-LEAD 只破 sample 级对称（所以 Pass@1 强 Pass@k 弱），都没同时解决两个问题。

A-GRAE 是第一个显式同时打破两层对称性的方法，而且几乎零成本、一个超参。这种"用一个小手术治一个大毛病"的工作，比堆架构堆数据的工作更值得读——它告诉你病根在哪，而不是给你一颗止痛药。

---

# Unveiling Implicit Advantage Symmetry: GRPO 的对称性"陷阱"与 A-GRAE 的破局

非常有趣的 paper, 核心洞察是把 GRPO 的 advantage 估计(GRAE)重新写成一个**带权 SFT** 的形式, 然后从这个视角挖掘出两层"对称性"瓶颈——group level 的对称性导致**未采样路径零梯度**(exploration 缺失), sample level 的对称性导致**中等难度样本主导更新**(difficulty adaptation 缺失)。A-GRAE 用一个轻量的、几乎免超参的修正同时打破这两层对称性。让我一层一层 build up the intuition。

---

## 1. 核心重写: GRPO = Reweighted SFT

整篇 paper 的起点是 Eq. (4), 这是后面所有分析的基石:

$$\nabla_\theta \mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{Q}, \{o_i\} \sim \pi_{\theta_{\mathrm{old}}}} \frac{1}{G}\sum_{i=1}^{G}\Big[\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\underbrace{\rho_{i,t} A_{i,t}}_{\text{Weight}}\underbrace{\nabla_\theta \log \pi_\theta(o_{i,t}\mid q, o_{i,<t})}_{\text{SFT gradient}}\Big]$$

变量解释:
- $q$: 从数据集 $\mathcal{Q}$ 采样的 question
- $o_i$: group 中第 $i$ 条 rollout, $|o_i|$ 是它的 token 数
- $G$: group size (paper 中默认 $G=8$)
- $\rho_{i,t} = \frac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q, o_{i,<t})}$: importance sampling ratio
- $A_{i,t}$: token-level advantage; GRPO 默认整序列共享一个 advantage, 即 $A_{i,1}=\cdots=A_{i,|o_i|}=A_i$
- $A_i$ 由 GRAE 计算: $A_i = \frac{r_i - \mathrm{mean}(\{r_1,\dots,r_G\})}{\mathrm{std}(\{r_1,\dots,r_G\})}$ (Eq. 3)

**直觉**: GRPO 不是神秘黑魔法, 它就是给 SFT 的 token-level gradient 乘了一个标量 weight $\rho_{i,t} A_{i,t}$。由于 clip 让 $\rho_{i,t}$ 稳定在 $[1-\epsilon, 1+\epsilon]$ 内, dominant 的 reweighting 信号就是 $A_{i,t}$。当 $A_i \to 1$ 且 $\rho \to 1$ 时, 整个式子退化成 teacher-forcing SFT。

这条 reformulation 把"policy gradient"翻译成了"reweighted cross-entropy", 让后面所有分析都可以直接从 weight 的视角入手。

---

## 2. Group Level Symmetry: 未采样路径的零梯度问题

### 2.1 对称性的代数根源

第一条对称性来自 GRAE 的 zero-sum 性质 (Eq. 5):

$$\sum_{i \in \mathcal{G}_{pos}} |A_i| = \sum_{i \in \mathcal{G}_{neg}} |A_i|$$

证明很直接: 由于 $A_i = (r_i - \mu)/\sigma$, 把所有 $A_i$ 加起来:

$$\sum_{i=1}^{G} A_i = \frac{1}{\sigma}\Big(\sum r_i - G\mu\Big) = 0$$

把 group 拆成正负两个不相交的子集 $\mathcal{G}_{pos} \cup \mathcal{G}_{neg} = \mathcal{G}$, 再利用 $A_i>0$ for positive, $A_i<0$ for negative, 就得到 $|\sum_{pos} A_i| = |\sum_{neg} A_i|$。

直觉: 给正确 rollout 的总"推动力" 与给错误 rollout 的总"抑制力" 严格相等。这是 normalization 带来的几何对称性, 同时也是一个非常强的约束。

### 2.2 Theorem 1: Behavior Space 里的 Logits 动力学

paper 把所有可能的 trajectory 集合记作 $\mathcal{B} = \{b_1,\dots,b_N\}$, 当前采样到的 group $\mathcal{G} \subset \mathcal{B}$, 没采样到的 $\mathcal{U} = \mathcal{B}\backslash\mathcal{G}$。定义 $C = \sum_{o_i \in \mathcal{G}} A_i$。Theorem 1 给出任意 trajectory $b_i$ 的 logits 更新:

$$\nabla_\theta J = \eta\big[\mathbb{I}(b_i \in \mathcal{G}) A_i - C \pi_{b_i}\big] \tag{Eq. 6}$$

变量解释:
- $b_i$: behavior space 中某条具体 trajectory
- $\mathbb{I}(b_i \in \mathcal{G})$: 指示函数, $b_i$ 在当前 group 中取 1
- $C$: 整个 group 的 advantage 之和
- $\pi_{b_i}$: 当前 policy 给 $b_i$ 的概率 (softmax over behavior space)
- $\eta$: 学习率

证明的关键是 $\partial \ln \pi_{b_k} / \partial h_{b_i} = \delta_{ik} - \pi_{b_i}$ (log-softmax 的标准导数), 代入目标 $J(h) = \sum_{b_k \in \mathcal{G}} \hat{A}_{b_k} \ln \pi_{b_k}$, 第一项坍缩成指示函数项, 第二项变成 $C \pi_{b_i}$。

**关键直觉**: 当 GRAE 把 advantage 归一化为 zero-sum 时, $C = 0$, 于是三种 case 截然不同:

| Case | Trajectory 类型 | Logit Update | 含义 |
|------|--------------|--------------|------|
| A | $b_i \in \mathcal{G}_{pos}$ (采样且正确) | $\Delta h = \eta A_{pos} > 0$ | 概率上升 |
| B | $b_i \in \mathcal{G}_{neg}$ (采样且错误) | $\Delta h = \eta A_{neg} < 0$ | 概率下降 |
| C | $b_i \in \mathcal{U}$ (未采样) | $\Delta h = 0$ | **完全不动!** |

这就是 GRPO exploration 不足的**精确数学根因**: behavior space 里所有未被采样的 trajectory, 即使是低概率的正确解, 它的 logits 在标准 GRPO 下**永远不动**, 只能等被随机采样到才有机会被强化。模型被困在自己已经采样到的 support 之内, 这就是 Yue et al. 2025 等观察到的 "Pass@k 在大 k 时反而低于 base model" 的本质机理。

参考:
- DeepSeekMath (GRPO 原版): https://arxiv.org/abs/2402.03300
- Yue et al. 2025 "Does RL Really Incentivize Reasoning Capacity": https://arxiv.org/abs/2504.13837
- He et al. 2025 "Rewarding the Unlikely": https://arxiv.org/abs/2504.13837 (近似关联)

---

## 3. Sample Level Symmetry: 中等难度样本的统治

### 3.1 Theorem 2: 总优势幅值与难度的关系

第二条对称性在 sample level。设 group 内的成功率 $p = \sum |r_i| / |G|$, 把 binary reward 下 $\mu = p$, $\sigma = \sqrt{p(1-p)}$ 代入, 对绝对 advantage 求和:

$$\sum_{i \in G} |A_i| = 2|G|\sqrt{p(1-p)} \tag{Eq. 10}$$

证明 (Eq. 32 in appendix B.4):
- 成功样本 ($r_i = 1$, 共 $Gp$ 个): $|A_i| = (1-p)/\sqrt{p(1-p)}$
- 失败样本 ($r_i = 0$, 共 $G(1-p)$ 个): $|A_i| = p/\sqrt{p(1-p)}$
- 总和: $\frac{Gp(1-p) + G(1-p)p}{\sqrt{p(1-p)}} = 2G\sqrt{p(1-p)}$

**直觉**: 这是一个关于 $p=0.5$ 对称的钟形曲线。$p=0.5$ 时总优势幅值最大, 也就是说中等难度样本贡献最多 gradient; $p=0.25$ 和 $p=0.75$ 贡献**完全相同**(对称性!), 即使前者是难样本、后者是简单样本。

### 3.2 这为什么是 sub-optimal

paper 在 Figure 6 里画出这条曲线, 然后给出一个动态视角: 训练过程中模型能力变化, 简单样本比例上升, 难样本比例下降, 整个 batch 的 $p$ 分布往右移。但 GRAE 的对称权重 $\sqrt{p(1-p)}$ 静态地把所有 phase 同等对待, 结果:
- 早期模型弱, 大量简单样本浪费了训练信号;
- 后期模型强, 难样本应该被关注时却仍然处于 $p$ 低端被对称地"等量"对待, 但难样本本来就稀少, 实际 contribution 远低于简单样本。

这是 difficulty-agnostic 的本质: 不是 GRPO 没考虑难度, 而是它**自动**用一个对称钟形把难度和样本数耦合在一起, 无法动态调整。

参考类似 difficulty-aware 的工作:
- GRPO-LEAD (Zhang & Zuo 2025): https://aclanthology.org/2025.emnlp-main.287/
- Curriculum RL (Deng et al. 2025a): https://arxiv.org/abs/2503.07065
- Pikus et al. 2025 "Hard Examples Are All You Need": https://arxiv.org/abs/2508.14094

---

## 4. Control Experiments: 用人工打破对称性来诊断

paper 最漂亮的部分是用两组 control experiment **因果性地**检验对称性的影响, 而不是只做事后解释。

### 4.1 Control Experiment I — Group Level

引入 $\beta = 10$ 扰动 zero-sum:

| Variant | $A_{pos}^*$ | $\sum_i A_i^*$ |
|---------|-------------|----------------|
| GRPO (control) | $A_{pos}$ | $0$ |
| Positive-Dominant | $\beta \cdot A_{pos}$ | $> 0$ |
| Negative-Dominant | $A_{pos}/\beta$ | $< 0$ |

注意, 当 $\sum A_i > 0$ 时, Theorem 1 的 Case C 变成 $\Delta h_{b_i} = -\eta C \pi_{b_i} < 0$ (未采样 path 概率被压); 当 $\sum A_i < 0$ 时, Case C 变成 $\Delta h_{b_i} = -\eta C \pi_{b_i} > 0$ (**未采样 path 概率被推上去**, 给 exploration 创造机会)。

实验结果 (Figure 2 + Figure 3, Qwen2.5-Math-7B 上):
- **GRPO**: Pass@1 大幅超越 base, 但 Pass@k 在 k=256 时跌到 base 以下 (MATH、AMC23), 验证了 capability boundary shrinkage。
- **Positive-Dominant**: 大 k 下严重 underperform, entropy collapse 最快——过度锐化已采样的正确路径, 多样性塌缩。
- **Negative-Dominant**: Pass@k **一致优于** GRPO, AIME2025 在 k=256 时甚至接近 base 上限; entropy **单调上升** (Figure 3)。但 appendix D.3 (Figure 9) 显示, 多次重复实验下 Negative-Dominant 在 10 次中有 3 次出现 step ~78 之后的 catastrophic collapse——overconfident 的错误 trajectory 在后期占据正确 trajectory 的位置。

这是这篇 paper 最 key 的 take-away 之一:
> **抑制正确 advantage 确实能换取 exploration, 但静态地、持续地抑制会引发后期 instability**——必须动态调节。

appendix B.5 给出 Theorem 1 在 $C<0$ 下的展开:
- Case A (sampled positive): $\Delta h = \eta(A_{pos} - C\pi_{b_i}) > 0$, 仍然上升 (双正)
- Case B (sampled negative): $\Delta h = \eta(A_{neg} - C\pi_{b_i})$, 当 $-C\pi_{b_i} > |A_{neg}|$ 时 **错误 trajectory 的 logit 反而上升**! 这正是 collapse 的机制。
- Case C (unsampled): $\Delta h = -\eta C \pi_{b_i} > 0$, 全体未采样 path 概率上升 → 这是 exploration 的来源。

### 4.2 Control Experiment II — Sample Level

用 $\gamma = 0.5$ 保持理论最大值一致, 重新 scale advantage:

| Variant | $A_i^*$ | $\sum |A_i^*|$ |
|---------|----------|----------------|
| GRPO | $A_i$ | $2G\sqrt{p(1-p)}$ |
| Hard-Focused | $\gamma A_i / \sqrt{p}$ | $G\sqrt{1-p}$ (单调, 难样本贡献大) |
| Easy-Focused | $\gamma A_i / \sqrt{1-p}$ | $G\sqrt{p}$ (单调, 简单样本贡献大) |

结果 (Figure 4 + Figure 5):
- 没有全局最优: Hard-Focused 在 AIME2025 最好, Easy-Focused 在 AMC23/MATH Pass@1 略好。
- Figure 5 的 within-batch correct count 显示: **Easy-Focused 早期收敛最快**(学到基础格式和推理模板), **Hard-Focused 后期才能拉开优势**(突破 ceiling)。

直觉: 这是经典 curriculum 的体现——**先简后难**。早期把信号分配给难样本是浪费(模型还不会做); 后期把信号分配给简单样本是浪费(模型已经会做)。GRAE 的对称钟形正是把这个 phase dependency 抹掉了。

---

## 5. A-GRAE: 一个几乎免参的修正

paper 的解法非常克制, 只引入一个超参 $\alpha$。

### 5.1 Sample Level — Dynamic Difficulty Attention Shift (DDAS, Eq. 12)

定义 batch-wise mean reward 作为训练状态 indicator:

$$\omega_s = \frac{1}{B}\sum_{i=1}^{B} r_i \tag{Eq. 11}$$

变量:
- $B$: batch 内 trajectory 总数
- $\omega_s$: 当前 batch 的平均 reward, 值越大代表模型当前能力越强

然后动态混合 hard-focused 和 easy-focused 两支:

$$A_i = \underbrace{\frac{\omega_s}{2} \cdot \frac{r_i - \mathrm{mean}}{\mathrm{std}\cdot\sqrt{p}}}_{\text{hard-focused weight}} + \underbrace{\frac{1-\omega_s}{2} \cdot \frac{r_i - \mathrm{mean}}{\mathrm{std}\cdot\sqrt{1-p}}}_{\text{easy-focused weight}} \tag{Eq. 12}$$

变量:
- $p$: 当前 query 的 group 内采样成功率
- 第一项系数 $\omega_s/2$: 模型越强占比越大, 把 focus 推向 hard
- 第二项系数 $(1-\omega_s)/2$: 模型越弱占比越大, 把 focus 留在 easy

直觉: $\omega_s$ 是一个 self-adaptive curriculum 信号, 不需要额外的 scheduler, 完全从 reward 信号里读出训练阶段。$p \in \{0, 1\}$ 时不做 rescale (此时 GRPO advantage 本来就是 0, 避免 0/0)。

### 5.2 Group Level — Attenuation Suppression Strategy (ASS, Eq. 13)

$$A_i^* = \begin{cases} A_i \cdot \min(1, \omega_s/\alpha), & A_i > 0 \\ A_i, & A_i \leq 0 \end{cases} \tag{Eq. 13}$$

变量:
- $\alpha \leq 1$: scaling parameter; Math 数据集 $\alpha=1$, 多模态 $\alpha=0.5$(因为多模态更易 collapse)
- $\omega_s$: 同上, batch mean reward

机制:
- 训练早期 $\omega_s$ 小, $\omega_s/\alpha < 1$, **正 advantage 被衰减**, 等价于温和版的 Negative-Dominant, 鼓励 exploration
- 训练后期 $\omega_s \to 1$, $\omega_s/\alpha > 1$, 取 1, **退化为标准 GRPO**, 锁定已学到的正确解, 避免 collapse
- 负 advantage 永远不被衰减, 错误 trajectory 始终被稳定抑制

这是相对 Negative-Dominant 的关键改进: 它**自适应地把抑制效应关闭**, 而不是持续抑制到底。appendix D.5 Table 6 + 10 次重复实验验证: ASS 和 Negative-Dominant 在 Pass@k 上几乎相同, 但 ASS **0/10 次出现 collapse**, Negative-Dominant 3/10 次 collapse。

直觉: ASS 不是一个新算法, 它是 Negative-Dominant 的"安全版本", 用 batch reward 当作阀门, 自动把 $C$ 从 $\sum A_i < 0$ 慢慢拉回 $\sum A_i = 0$。从 Theorem 1 的视角看, 这相当于把 Case C 的 exploration 项 $-C\pi_{b_i}$ 在后期自动归零, 同时让 Case B 的负 advantage 始终主导, 防止 overconfident 错误解反超正确解。

### 5.3 整体直觉图

把两层合起来:
1. **早期** ($\omega_s$ 小): ASS 强烈抑制正 advantage → exploration 充足; DDAS 主要走 easy-focused → 快速学到格式与基本推理模板。
2. **中期** ($\omega_s$ 中): ASS 渐渐释放正 advantage → exploitation 上升; DDAS 开始往 hard 偏移。
3. **后期** ($\omega_s \to 1$): ASS 退化为标准 GRPO → 稳定收敛; DDAS 主要走 hard-focused → 突破 ceiling。

这是一个由 reward 信号驱动的 implicit curriculum, 既在 trajectory level 上动态打破对称(exploration), 又在 sample level 上动态打破对称(difficulty adaptation)。

参考实现: https://github.com/Yu7-code/A-GRAE

---

## 6. 实验数据的关键读法

### 6.1 Table 2 — Qwen2.5-Math-7B 文本数学

挑几组最有信息量的对比 (Pass@k 全谱):

**AIME 2025 (最难)**:
| Method | P@1 | P@16 | P@64 | P@256 |
|--------|-----|------|------|-------|
| Base | 6.1 | 24.4 | 33.4 | 46.7 |
| GRPO | 10.3 | 27.5 | 36.1 | 40.8 |
| W-REINFORCE | 10.6 | 29.7 | 40.5 | 56.7 |
| GRPO-LEAD | 11.0 | 27.8 | 36.5 | 47.3 |
| **GRPO+A-GRAE** | 11.3 | 28.6 | 39.2 | **56.7** |
| **Dr.GRPO+A-GRAE** | 11.8 | 29.3 | 37.9 | **56.7** |

读法:
- GRPO 的 Pass@256 = 40.8 **低于 base 46.7**, 这正是 capability boundary shrinkage 的直接证据——GRPO 把概率质量过度集中到已采样支持域上, 在大 k 采样下反而不如原模型能采到多样化的正确解。
- A-GRAE 把 Pass@256 拉回 56.7, 不只是恢复, 而是**超越** base, 说明 group-level ASS 真的让模型学到了原本采样不到的正确解。
- DAPO+A-GRAE 在 AIME 上 Pass@256 = 60.0, 进一步说明 A-GRAE 和现有 GRPO 变体正交可叠加。

**MATH Pass@256**: GRPO 95.0 → Dr.GRPO+A-GRAE 96.9, 已经接近 base 上限 96.3, 几乎封顶。

### 6.2 Table 3 — 多模态 (Qwen2.5-VL-3B-Instruct)

Medical Xray300:
- Base 42.0 → GRPO 63.2 → GRPO+A-GRAE **71.3** (+8.1)
- Dr.GRPO 69.5 → Dr.GRPO+A-GRAE 72.0 (+2.5)

Xray 这一项是 single-task 跨数据集提升最大的, 体现 A-GRAE 在小模型 + 专业领域同样有效, 说明对称性问题不只是数学 benchmark 上的 artifact。

### 6.3 Table 5 — Ablation

- **Sample level only** 主要提升 Pass@1 (MATH 76.5 → 77.8): 对应 difficulty alignment 改善精度。
- **Group level only** 主要提升 Pass@k (MATH P@256 95.0 → 97.0; AIME P@256 46.7 → 60.0): 对应 exploration 改善多样性。
- **Full** 在两条轴上同时受益。

这个 ablation 验证了 paper 的核心 claim: 两层对称性是**正交的两个瓶颈**, 一个影响 exploitation accuracy, 一个影响 exploration diversity, 必须同时打破。

### 6.4 Figure 10 — Training Dynamics

- 训练集 entropy: GRPO 单调下降; A-GRAE 先快速下降再 plateau (avoid entropy collapse)。
- 测试集 entropy: A-GRAE 先升后降——前期 exploration, 后期 exploitation, 验证 implicit curriculum 的存在。
- Greedy accuracy: A-GRAE 在中后段显著超越 GRPO, 说明不是靠牺牲 Pass@1 换 Pass@k, 是真实能力提升。

---

## 7. 与相关工作的定位 — Advantage Symmetry 作为统一解释框架

Section 4.4 提供了一个很清晰的统一视角, 把一系列看似不同的 method 都归为"打破某一层对称性":

- **Dr.GRPO** (Liu et al. 2025b, https://arxiv.org/abs/2503.20783): 移除 length 和 std normalization, 但**没有打破对称性**, $C$ 仍然为 0, Pass@k 仍受限。
- **DAPO** (Yu et al. 2025, https://arxiv.org/abs/2503.14476): token-level 平衡长序列梯度, 同样没碰对称性。
- **W-REINFORCE** (Zhu et al. 2025, https://arxiv.org/abs/2411.14451): 只用 negative reward 学习, **隐式打破了 group-level 对称性**(完全去掉正 advantage), 类似 extreme Negative-Dominant。它在 Pass@k 上很强 (AIME P@256 56.7) 但 Pass@1 不强, 这正是只破一层对称的代价。
- **GRPO-LEAD** (Zhang & Zuo 2025, https://aclanthology.org/2025.emnlp-main.287/): difficulty-aware, **隐式打破 sample-level 对称性**, Pass@1 提升明显但 Pass@k 几乎不动。
- **High-entropy token 强调** 类工作 (Cheng et al. 2025, Hao et al. 2025, Zhang et al. 2025c): 也是隐式 group-level 干预。
- **Curriculum RL** (Deng et al. 2025a, Li et al. 2025a): 隐式 sample-level 干预。

A-GRAE 是**第一个显式同时打破两层对称性**的工作, 在 Table 2 里同时拿到了 Pass@1 和 Pass@k 的优势——而 W-REINFORCE/GRPO-LEAD 各自只在一边强, 这正验证了对称性框架的解释力。

paper 没有显式比较的另一条线是 ProRL (Liu et al. 2025a, https://arxiv.org/abs/2505.24864), 它通过 prolonged training 来探索, 是另一个方向; 还有 XRPO (Bamba et al. 2025, https://arxiv.org/abs/2510.06672) 用 targeted exploration。这些都是间接逼近同一问题, A-GRAE 的优势是几乎零开销、几乎免参。

---

## 8. 一些 Critical 思考点与潜在联想

1. **Importance sampling 偏置**: Theorem 1 假设 "negligible importance sampling bias", 但 Eq. 4 里 $\rho_{i,t}$ 在 on-policy 训练中通常接近 1, 这个假设在 GRPO 多 step PPO update 时会偏离, 后期 $\rho$ 漂移可能让 $C\neq 0$ 的实际效果与理论分析有 gap。A-GRAE 没有显式处理这点, 实证上靠 KL 项兜底。

2. **$\omega_s$ 作为 training stage 代理的脆弱性**: $\omega_s$ 是 batch mean reward, 在 reward hacking、format reward 混入 (多模态实验中用了 0.1·acc + 0.9·format) 时可能不能真实反映能力。如果一个 batch 全是简单样本, $\omega_s$ 高, DDAS 会过早切到 hard-focused, 早期可能 sub-optimal。可以考虑用 EMA 或者 dataset-stratified $\omega_s$。

3. **Group size $G$ 对称性**: Theorem 2 的 $\sqrt{p(1-p)}$ 依赖 $G$ 足够大才能让 $p$ 离散值近似连续。当 $G=8$ 时, $p \in \{0, 1/8, 2/8, \dots, 1\}$ 只有 9 个取值, $\sqrt{p(1-p)}$ 的对称钟形其实非常离散。如果 $G$ 变大 (DAPO 等用 dynamic sampling 让 $G$ 可变), 对称性结论需要重新审视。这是一个未被讨论的边界条件。

4. **Pass@k 作为 exploration proxy 的局限**: Pass@k 用 unbiased estimator $\binom{n-c}{k}/\binom{n}{k}$ (Eq. 37), 但 $n=16$ 时 Pass@128/256 的估计方差非常大, AIME2025 上 P@256 = 56.7 实际只是 16 次 run 中能不能采到至少 1 个正确解的二值化, 这跟 "exploration 能力" 的语义其实有距离。更严格的 exploration 度量应该看 unique correct solutions 的数量或 embedding diversity。

5. **KL 项的作用**: paper 在 Eq. 4 的推导中**完全去掉了 KL 项**, 但 KL 实际上对 entropy collapse 有抑制作用。Positive-Dominant 的 entropy collapse 在实际训练中可能被 KL 部分缓解, paper Figure 3 显示的 collapse 速度可能比"纯 policy gradient" 慢, 这一点需要谨慎解读。

6. **与 OpenAI o1 / R1 的 connection**: DeepSeek-R1 (https://arxiv.org/abs/2501.12948) 报告了 GRPO 在大规模 long-CoT 上的成功, 但也提到 R1-Zero 的 self-evolution 现象。从 A-GRAE 的视角看, R1 的 "aha moment" 可能正是 group-level 对称性在 long training 下偶然被打破(通过 reward variance 极端化)的结果——这是非常 speculative, 但值得 build intuition 上去 connect。

7. **Connection to offline RL / importance weighting**: Eq. 4 的 reweighted-SFT 视角其实和 offline RL 的 advantage-weighted regression (AWR, Nair et al. 2020; AWAC, Kostrikov et al. 2021) 高度类似。A-GRAE 可以看作一个**自适应温度的 advantage-weighted SFT**, $\omega_s$ 充当温度调度器。这条线可以连到:
   - AWR: https://arxiv.org/abs/2006.09359
   - AWAC: https://arxiv.org/abs/2006.09359 (近似)

8. **Diffusion model 里的对称性联想**: 这个 "未采样路径零梯度" 的现象在 diffusion sampling 里也有类似结构——DDPM 的 ancestral sampling 只走一条路径, score matching 对未走过的状态没有梯度, 这正是 classifier-free guidance 与 consistency model 的动机之一。A-GRAE 的"主动推未采样 path"思路, 类比到 diffusion 就是用 negative guidance 强行推开已采样 mode 来探索新模式。

---

## 9. 总结: 这篇 paper 给我的三条直觉

1. **GRPO 是带权 SFT**, advantage 是权重。从这点出发, 对称性瓶颈是 normalization 的几何后果, 不是 RL 神秘性质。
2. **未采样 trajectory 的零梯度** = exploration 缺失的精确根因; 通过让 $C<0$ 推未采样 path 可以破解, 但需要动态关闭来避免 collapse。
3. **$\sqrt{p(1-p)}$ 钟形** 让中等难度样本统治更新, 简单/难样本对称地被低估; $\omega_s$ 作为 reward-driven curriculum signal 可以打破这个对称。

A-GRAE 的工程价值在于: **只用一个超参 $\alpha$, 几乎零额外计算**, 可以 plug-in 到 GRPO/DAPO/Dr.GRPO 之上, 在文本和多模态、Pass@1 和 Pass@k 上都拿到一致提升。理论价值在于: 它把一堆零散的 "negative learning"、"hard sample focusing"、"entropy intervention" 工作统一到 "advantage symmetry" 一个框架下, 让"破对称"成为 advantage design 的一阶原理。

Code: https://github.com/Yu7-code/A-GRAE  
Contact: zhiqiyu777@connect.hku.hk

---

如果你想往下深挖, 我建议三个方向: (a) 在更大 base (DeepSeek-R1-7B 上他们 Table 4 已经做了, 收益 pattern 一致但更平缓, 因为 R1-7B 已经接近饱和); (b) 把 $\omega_s$ 换成 per-difficulty bucket 的 statistic, 看是否能进一步 push AIME Pass@k; (c) 在 Theorem 1 的 $C \neq 0$ 情况下重做一次完整的 convergence 分析, 把 KL 项 explicit 纳入, 这会给出一个更紧的 exploration-stability tradeoff 边界。
