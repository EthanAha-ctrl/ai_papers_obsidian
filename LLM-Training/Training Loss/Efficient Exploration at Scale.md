---
source_pdf: Efficient Exploration at Scale.pdf
paper_sha256: b1de3254893d464d2e99f102b3613a2c3a756c97853efd85130b920b5adb032c
processed_at: '2026-08-04T01:42:18-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话总结

Google DeepMind 这帮人搞了个 RLHF 训练新方法，**20K 条人类偏好数据就顶得上传统方法 200K 条**，而且 extrapolation 显示 1M 条能顶 1B 条——**1000 倍的数据效率**。

这数字听着像 PPT marketing，但仔细看 paper 我信了。原因下面讲。

---

## 传统 RLHF 为啥这么拉

先回忆下 standard RLHF 怎么干：

1. 拿一个 SFT model（比如 Gemma 9B）
2. 让它对一堆 prompt 各生成两个 response，请人类标注员选哪个好
3. 用这些 choice data 训一个 reward model（RM）
4. 再用 RM 当"假人类"，给 policy 提供 reward signal，让 policy 通过 RL 往高 reward 的方向挪

整个 pipeline 是**两阶段离线**的——先采一波固定数据训 RM，再在这个 RM 上训 policy，数据从头到尾不变。

听起来 OK，实际上有个致命问题：**你采数据用的是 SFT policy，但训完 policy 后它早就不在那个分布上了**。

打个比方。你让一个初中生（SFT model）写一堆作文，请老师批改打分。然后你用这批"老师打分"训一个"作文评分器"（RM）。问题来了——这个评分器只在**初中生作文**这个分布上学过。等你再用它指导一个高中生写作文（policy 更新后），它对高中生写的那些"超出初中生水平"的部分完全没概念，可能给特别新颖的内容打低分，反而把高中生往初中生水平拉。

这就是 Hou et al. (2024) 那篇说"RLHF 不 scale"的根本原因：**多给数据没用，因为多的数据还是在初中生分布上采的，覆盖不了你想去的新区域**。这是 distribution shift 问题，不是 data 量问题。

参考：Hou et al. 2024 https://arxiv.org/abs/2410.01268

---

## 第一个 fix：Online，让数据跟着 policy 走

最直接的 fix——**别离线了，边训边采**。

每个 batch：
1. 用当前 policy 采 responses
2. 问人类选哪个
3. 更新 RM
4. 用新 RM 更新 policy
5. 回到 1

这样 sampling distribution 永远跟 policy 同步，RM 永远在 policy 真正关心的区域上被训练，distribution shift 消解。

这个 idea 本身不新——Anthropic 2022（Bai et al. https://arxiv.org/abs/2204.05862）就搞过 iterative RLHF，DeepMind 这帮人也是沿着这条路。问题是**他们发现直接这么搞会 tank**。

---

## 第二个 fix：Affirmative nudge，防 tanking

### Tanking 是啥

Online RLHF 跑着跑着，policy 性能会突然崩——paper 里叫 "tanking"。Figure 4 right 那个曲线，红色线一路往上然后啪一下掉下来。

为啥崩？这是个**正反馈死亡螺旋**：

1. Policy 越训越 narrow，越来越 confident 在某个小区域
2. RM 在这个小区域上训练，越来越 confident 在错误方向
3. Policy 听 RM 的话继续往那个错误方向走
4. 新采的数据还是在小区域，RM 继续 reinforce 自己的错误
5. 整个系统 collapse 到一个 RM 自以为好但其实烂的 mode

经典 RL 里防这个用 entropy bonus——给 token 分布加个熵正则，强制保持多样性。但 token-level entropy bonus 在长序列 LLM 上经常失效，因为序列 entropy 是 token entropy 累乘，搞不好把生成质量也搞烂。

### Nudge 是啥

他们加了个简单得离谱的 trick。原 policy gradient 信号是：

$$\text{reward signal} = P(Y \succeq Y') - 0.5$$

这里 $P$ 是 RM 给的"Y 比 Y' 好的概率"。$P > 0.5$ 表示 Y 应该被强化，$P < 0.5$ 表示应该被弱化。$P = 0.5$ 表示没信息，不动。

他们改成：

$$\text{reward signal} = P(Y \succeq Y') - 0.5 + \epsilon$$

加一个**很小的正数** $\epsilon$。

效果是：**即使 RM 完全 unsure（$P = 0.5$），policy 还是被微弱推 on-policy response**。这相当于在 RL signal 上叠加了一个小小的 SFT signal——"你自己已经生成的东西，至少弱弱地肯定一下"。

### 为啥这个能防 tanking

关键在于这个 always-positive bias 让 policy **不可能完全 collapse**。

想象 RM 全错、信号反向的极端情况——传统 RLHF 里 policy 会被推到一个烂 mode，越走越窄，最后死掉。加了 $\epsilon$ 之后，即使 RM 说"烂的更好"，policy 也始终在 on-policy 自己的 generation 上有一点点 SFT-like pull，不至于完全失语。多样性保留住，新数据继续多样化，RM 持续在多样化数据上学习，正反馈死循环被打破。

这个 trick 极简但极有效，Figure 4 right 那张图对比很明显——没 nudge 一定 tank，有 nudge 一路往上不回头。

---

## 第三个 fix：Information-directed exploration，让每个 query 都"值"

### On-policy 也有浪费

Online 解决了 distribution shift，nudge 解决了 collapse，但还有个问题：**随机采的 response pair 经常是 RM 已经很 confident 的**。

RM 已经很 confident 的 pair 是啥意思？就是"问人类这个 pair 是浪费时间"——RM 已经知道答案，问完人类也不会更新啥。

举个例子：prompt 是"今天天气怎样"，policy 采出两个 response，一个是"今天天气很好"，一个是"今天天气非常非常好"。RM 一看就知道第一个更好（或一样好），你问人类，人类说第一个，RM 说"对，我就这么想的"，没学到任何东西。

真正 informative 的 pair 是 RM 自己拿不准的——比如一个简短一个详尽，一个有理一个有据，RM 不知道人类会偏好哪种。

### ENN: 用一个网络算 uncertainty

怎么知道 RM 拿不准？经典的几个方法：
- MC dropout：粗糙，理论不太对
- Deep ensemble：训 100 个独立 RM，看分歧——理论对但贵爆
- Bayesian NN：理论对但实际不好 scale

他们用了 ENN（Epistemic Neural Network，Osband 2023 https://arxiv.org/abs/2107.08924）——比 deep ensemble 便宜、比 dropout principled 的中庸方案。

具体架构：
- 一个 transformer backbone（共享，9B Gemma）
- 一个 point estimate head（普通 MLP，给"最可能 reward"）
- **100 个 prior networks**：小 MLP（2×256），random init，**永不训练**
- **100 个 differential networks**：大 MLP（2×1024），训练

第 $i$ 个 ensemble member 的 reward = 第 $i$ 个 prior network + 第 $i$ 个 differential network。这是 **Randomized Prior Functions**（Osband 2018, https://papers.nips.cc/paper/2018/hash/5eda7e4d25a1f6c5d25b25f26c20ab4c-Abstract.html）的经典套路。

### Randomized Prior 为啥 work

直觉是这样。Bayesian linear regression 里，posterior sample 可以写成：

$$\theta_i = \hat\mu + L z_i$$

- $\hat\mu$：posterior mean（数据告诉你的"最可能"答案）
- $Lz_i$：posterior 的"随机扰动"，方差由数据量决定——数据多方差小，数据少方差大

ENN 把这个搬到神经网络：
- $\hat\mu$ 由 differential network 学（trainable）
- $Lz_i$ 由 prior network 直接固定 random init 模拟

数据少的地方，differential 还没学到位，prior 主导，每个 ensemble member 给出不同的随机 guess → variance 高 → "我不确定"。
数据多的地方，differential 学到位，prior 被淹没，每个 ensemble member 都收敛到 $\hat\mu$ → variance 低 → "我确定了"。

这就给出了 principled 的 epistemic uncertainty estimate，比 dropout 那种 ad-hoc 方法理论根基扎实得多。

### 怎么用这个选 query

每个 prompt：
1. 用当前 policy 采 **16** 个 responses
2. 对所有 $\binom{16}{2} = 120$ 个 pair 算 choice probability
3. 每个 pair 用 100 个 ensemble member 各算一次，得到 100 个 choice probability
4. 算这 100 个值的 variance
5. 选 variance 最大的 pair 问人类

variance 大 = ensemble 分歧大 = RM 不确定 = 信息增益高。问完人类这个 pair，分歧会被消除，RM 学到最多。

这是 **Information-Directed Sampling**（Russo & Van Roy 的 bandit 经典理论）的简化版——IDS 原版是 minimize regret / information ratio，他们简化成 maximize information gain，因为 LLM alignment 里 regret 不好估。

参考：IDS 原始理论 https://arxiv.org/abs/1603.02282

---

## 三个 fix 为什么是乘性而非加性

这是 paper 最漂亮的地方。三个 trick 不是简单叠加，是**乘性协同**：

| Trick | 解决的问题 | 不加会怎样 |
|---|---|---|
| Online | Distribution shift | RM 学错区域，policy 走偏 |
| Nudge | Mode collapse | Policy tanking，正反馈死循环 |
| IDE | Query 浪费 | 大部分 query 问 RM 已知的，白费 label |

三个一起：每个 label 都在 policy 关心的区域（online）+ policy 保持多样（nudge）+ 每个 query 都最大化信息增益（IDE）。

少任何一个，10× gain 就没了。paper 里 Figure 8 那条曲线，offline / periodic / online / IDE 一档一档往上走，每一档对应一个 trick 加进来。

---

## 几个我特别 appreciate 的细节

### 用比 train model 大的模型当"假人类"

他们用 **Gemini 1.5 Pro** 当 human feedback simulator——比被训练的 9B Gemma 大得多。这样 simulator 表现出的 preference 比 9B policy 复杂，模拟"真实人类可能比 LLM 复杂"这一情况。Scaling 实验结果更可能 carry over 到真实人类标注。

Bradley-Terry 转换：simulator 给 prompt + 两个 responses，输出两个 reward $(R_1, R_2)$，然后：

$$P = \frac{e^{R_1}}{e^{R_1} + e^{R_2}}$$

人类 choice $C \sim \text{Bernoulli}(P)$。这是 RLHF simulator 标准做法，没什么花头，但用大模型当 simulator 这点很有意思。

### Anchor 用 EMA 而非 SFT

Policy 更新时 KL regularize 到一个 anchor $\bar\theta_t$。anchor 是当前 policy 参数的 EMA：

$$\bar\theta_{t+1} = \eta \bar\theta_t + (1-\eta)\theta_{t+1}$$

而不是固定的 SFT $\theta_0$。这意味着 anchor 跟 policy 走，但慢半拍——"别离你最近的自己太远"。比 PPO 里 anchor 到 SFT 灵活得多，policy 可以慢慢 drift 到新区域，但每一步都被局部约束。

### Win rate 而非 RM score 评估

性能评估用 **top-1 sampling** 跑 baseline vs new policy，让 simulator 当裁判算 win rate。不直接看 RM 分数，避免 RM over-optimization 的 cheating。这是 RLHF 实验正确做法。

---

## 实验数据看这张表

| Labels | Offline RLHF | Periodic | Online RLHF | IDE |
|---|---|---|---|---|
| 1K | ~0.52 | ~0.55 | ~0.55 | ~0.58 |
| 5K | ~0.56 | ~0.62 | ~0.62 | ~0.66 |
| 20K | ~0.60 | ~0.66 | ~0.68 | ~0.72 |
| 200K | ~0.66 | ~0.70 | ~0.72 | ~0.78 |

**关键看点**：IDE 在 20K labels 处达到 offline 200K 的水平——10× 提速。

Extrapolation 用 power-law 拟合 $w(n) = 1 - 0.5(n/a)^{-b}$。Offline 的 $b$ 小、IDE 的 $b$ 大。Log-log 图上 IDE 曲线斜率更陡，1M labels 处 IDE 预期能打 offline 1B labels 的水平——**1000× gain**。

---

## 例子看直觉

Paper 5.5 节给了两个非常好的例子解释 IDE 选的 query 长啥样。

**Infomin pair**（variance 最小，RM 已经确定）：
- Response 1: "Positive."
- Response 2: "Positive sentiment."

这俩 RM 一眼就知道等价，问人类纯属浪费。

**Infomax pair**（variance 最大，RM 不确定）：
- Response 1: "positive"
- Response 2: "Neutral."

一个判 positive，一个判 neutral——RM 不知道人类会不会觉得"中性"也算可接受答案。问人类这个能学到关于"边界 sentiment 判定"的信息。

这例子非常清楚地说明 IDE 在干啥：**找 RM 自己最纠结的那对，问人类到底哪个对**。

---

## 局限性

我说说我自己看完的几个保留：

1. **Simulator 是 Gemini 1.5 Pro**——它可能有 LLM-specific 偏好（比如偏好结构化、礼貌、详尽的回答），跟真实人类有 gap。真实人类可能有 noise、inconsistency、context-dependent preference。不过 simulator 比被训练模型大这点至少保证 preference 比 9B 复杂。

2. **ENN 计算 overhead**——100 ensemble members。inference 时 point estimate 可以共享 backbone，但 100 个 head 还是要算。训练时 differential 训练 backbone frozen 反而便宜。整体 OK 但 scale 到 100B+ model 可能瓶颈。

3. **Variance ≠ 完美的 information gain**——variance 是 information gain 的代理。当 RM systematic biased（而非 uncertain）时，variance 可能低但实际错。ENN 理论上应该捕获 aleatoric vs epistemic 区分，实际可能不完美。更严格的 IDS 见 Russo & Van Roy 原版。

4. **每个 prompt 采 16 responses**——inference 成本不低。如果 prompt response 很长（比如代码生成），16 倍开销不小。

5. **未做 prompt selection**——paper 6 节提到可以扩展到选 prompt，但没做。Active learning 完整版应该 prompt + response 都选。

---

## 这篇 paper 对 alignment 的实际意义

### 1. RLHF 是会 scale 的

之前 Hou et al. 2024 说 RLHF 不 scale 让人很悲观。这篇证明：**不是 RLHF 不 scale，是 offline RLHF 不 scale**。Online + exploration 之后，scaling law 斜率完全不同。

这对 alignment 路线图很重要——意味着只要方法对，**更多人类标注就能换更多性能**，不是 hitting ceiling。这对未来 alignment infra 投入有直接含义。

### 2. Exploration 是 alignment 的一等公民

之前 RLHF 主流方法（PPO、DPO）都假设数据是 fixed 给定的，研究重点放在"如何用得更好"。这篇把"如何采数据"提到核心位置。Epistemic uncertainty 不再是 nice-to-have，是 sample efficiency 的核心 multiplier。

### 3. ENN 这种"架构内嵌 uncertainty"思路值得推广

传统做法是训完一个 model 再加 dropout / ensemble 估 uncertainty。ENN 把 uncertainty 直接 baked into 架构——一个 forward pass（指定 $Z$）就给出一个 posterior sample。这种设计哲学可以推广到 RLHF 之外的其他领域：safety、interpretability、robustness。

### 4. 给 scaling law 实验立了规范

Figure 1 那张 log-log 图。横轴 log scale 一拉，offline / online / IDE 三条曲线斜率天差地别，qualitative 差异立马显现。之前一堆 RLHF paper 用 linear 横轴，看着都很平，掩盖了关键模式。**未来 alignment 实验报告 scaling 应该默认用 log scale**。

---

## Reading path 如果想深入

强烈推荐按这个顺序读：

1. **Osband et al. 2018 Randomized Prior Functions**（https://papers.nips.cc/paper/2018/hash/5eda7e4d25a1f6c5d25b25f26c20ab4c-Abstract.html）——理解 RPF 的 Bayesian foundation
2. **Osband et al. 2023 Epistemic Neural Networks**（https://arxiv.org/abs/2107.08924）——ENN 的统一框架
3. **Dwaracherla et al. 2024 Efficient Exploration for LLMs**（https://proceedings.mlr.press/v235/dwaracherla24a.html）——前作，只更新 RM 不更新 LM
4. **本文**——RM + LM 端到端，加 nudge、加 IDE query selection

读完这条 path，整个方法论的演化逻辑非常清晰——从 Bayesian RL 理论 → ENN 架构 → RM-only LLM 实验 → 端到端 RLHF pipeline，每一步都跟前一步严丝合缝。

---

## 最后总结成一句话

**Offline RLHF 不 scale，是因为你给它的数据不在它想去的区域；Online RLHF 会 tank，是因为 policy-RM 系统正反馈死循环；Random exploration 浪费 query，是因为问的都是 RM 已知的——三个问题分别由 online、affirmative nudge、ENN-based IDE 解决，三者协同给出 10× 到 1000× 的 sample efficiency gain。**

这是 RLHF 工程化以来我认为最重要的 methodological 进展之一。值得所有做 alignment 的人仔细读。

---

# Efficient Exploration at Scale 深入解读

## 论文核心定位

这是 Google DeepMind Efficient Agent Team 的工作（Asghari, Chute, Dwaracherla, Lu, Jafarnia, Minden, Wen, Van Roy），针对 **RLHF 的 sample efficiency** 做了一个相当硬核的实证研究。核心 claim：通过 online learning + uncertainty-guided exploration，在 Gemma 9B 上把 RLHF 的 label efficiency 从 offline baseline 提升至少 **10×**（20K labels 匹配 200K labels），并且 log-log scale 上 extrapolation 显示 **1M labels 可匹配 1B labels**，对应 **1000×**。

这个数字本身就值得停下来想一想。Hou et al. (2024) 之前刚说过 RLHF 不 scale（多给数据也没用），这篇直接打脸，并解释清楚为什么 offline RLHF 不 scale、online + exploration 怎么把它救回来。

参考链接：
- arXiv 上的论文：https://arxiv.org/abs/2507.07885 （Efficient Agent Team）
- ICML 2024 相关前作 Dwaracherla et al.: https://proceedings.mlr.press/v235/dwaracherla24a.html
- ENN 论文 Osband et al. 2023: https://arxiv.org/abs/2107.08924
- Randomized Prior Functions Osband 2018: https://papers.nips.cc/paper/2018/hash/5eda7e4d25a1f6c5d25b25f26c20ab4c-Abstract.html

---

## 1. 整体管线

### 1.1 Model 设置

- Baseline policy：Gemma 9B（Team et al., 2024, https://arxiv.org/abs/2408.00118），经过 pretraining + SFT，记作 $\pi_{\theta_0}$。$\theta_0$ 是参数。Top-1 decoding 即 baseline，确定性的。
- Experimentation policy：top-$k$ sampling，$k=5$。Top-$k$ 是 stochastic 的，使得同一个 prompt 产生多样 responses，从而 human choice 是 informative 的。

### 1.2 Human Feedback Simulator

为了做大样本 scaling law 实验，他们用一个**远大于**被训练模型的模型当"伪人类"：

- **Gemini 1.5 Pro**（Gemini Team, 2024, https://arxiv.org/abs/2403.05530）当作 ground-truth reward function $R^*$。
- 给定 prompt $X$ 与两个 responses $(Y_1, Y_2)$，计算 $R_1, R_2$。
- 通过 **Bradley-Terry model**（Bradley & Terry, 1952）转成 preference probability：

$$P = \frac{\exp(R_1)}{\exp(R_1) + \exp(R_2)}$$

- 再 Bernoulli 采样得到模拟人类 choice $C \sim \text{Bernoulli}(P)$。

**直觉**：因为 simulator 模型远大于 9B Gemma，它表现出比被训练模型复杂得多的 preference 模式。这样做出来的 scaling 结果更可能 carry over 到真实人类标注（真实人类也可能比 LLM 复杂）。

### 1.3 Prompts

202K prompts 涵盖 writing / coding / summarization / reading comprehension / math / science / ideation 等。200K 训练、1K 测试（hyperparameter selection）、1K out-of-sample eval。

### 1.4 Batch 与 evaluation

- 每个 batch 64 prompts，每个 prompt 生成 2 个 responses，得到一个 choice。
- $\theta_t$ 表示看完 $t$ 个 batch 后的 policy 参数。
- Win rate $\bar P$：用 top-1 解码在 1K out-of-sample prompts 上比较 $\pi_{\theta_t}$ vs $\pi_{\theta_0}$，由 simulator 给出 win rate。$\bar P = 1$ 表示永远赢，$\bar P = 0.5$ 表示五五开。

---

## 2. 四个算法的层次结构

论文构造了四个递进的 baseline，逻辑非常清晰：

| 算法 | RM 更新方式 | Policy 更新方式 | 是否用 ENN | 是否用 nudge |
|---|---|---|---|---|
| Offline RLHF | 在 $\pi_{\theta_0}$ 上采的固定 $T$ 批数据上 fit | 在固定数据上更新 | 否 | 否 |
| Periodic RLHF | 每 $\tau$ 批重训 RM（从 $\phi_0$ 开始） | 每 $\tau$ 批用当前数据重训 policy（从 $\theta_0$ 开始） | 否 | 否 |
| Online RLHF | 增量更新 RM | 增量更新 policy | 否 | 是 |
| Information-directed exploration | 增量更新 RM + ENN heads | 增量更新 policy | 是 | 是 |

每一步的"加法"都对应一个可量化的 efficiency gain。

### 2.1 Offline RLHF 的本质缺陷

Offline 用 $\pi_{\theta_0}$ 一次性采 $T$ 批数据。这个 sampling distribution 是**固定的**。一旦 policy 想偏离 $\theta_0$，新区域的 responses 根本没出现在数据集里。RM 是在 $\pi_{\theta_0}$ 的 support 上学的，policy 想去的地方可能是 RM 没见过的——这就是经典的 **distribution shift / covariate shift** 问题。

更深一层：RM 学的是 $r(Y|X)$，但训练时见到的 $Y$ 都来自 $\pi_{\theta_0}$。RM 在 $\pi_{\theta_0}$ 高密度区域可能很准，但只要 policy 一偏离，reward 估计就不可信。Policy gradient 用错误的 reward 推 policy，再把 policy 推到 RM 更不可信的地方——正反馈崩塌。

Hou et al. (2024) 说 RLHF "不 scale" 的根本原因就是这个：增加 data 只是把 $\pi_{\theta_0}$ 的 support 上 RM 拟合得更准，但 policy 想去的好区域可能不在 $\pi_{\theta_0}$ 的 support 里。**这无关 data 数量，而是 sampling distribution 的覆盖问题**。

### 2.2 Periodic RLHF 的中间方案

Periodic RLHF 每 $\tau$ 批（论文里 $\tau = 400$）就重新采数据：用当前 $\pi_{\theta_{k\tau}}$ 采新一批，然后把累计数据合并训练。这是 Bai et al. (2022, Anthropic, https://arxiv.org/abs/2204.05862) 这种 iterative RLHF 的思路。

$\tau$ 越小越好，但每次都从 $\phi_0$、$\theta_0$ 重训成本爆炸。

### 2.3 Online RLHF 的核心改动

直接 incremental 更新 RM 和 policy。每来一个 batch：
1. 用 $\pi_{\theta_{t-1}}$ 采 responses
2. 拿到 choice
3. 更新 RM 参数 $\phi_t \leftarrow \phi_{t-1}$
4. 用新 RM 算 reward 信号，更新 policy $\theta_t \leftarrow \theta_{t-1}$

这样 sampling distribution 始终跟 policy 同步，RM 也始终在 on-policy distribution 上 fine-tuned。Distribution shift 问题被消解。

### 2.4 Information-directed exploration 的最后一步

Online 已经解决 distribution shift，但还有一个问题：**on-policy 采的 responses 不一定是 RM 最不确定的**。如果每次都从 $\pi_{\theta_t}$ 随机抽两个 responses，可能 RM 已经很确定了，问人类拿到的信息量很低。

解决方案：每个 prompt 采 **16 个** responses，然后用 ENN 算每个 pair 的 variance，挑 variance 最大的 pair 问人类。

---

## 3. 关键公式逐个拆解

### 3.1 Reward model 的预测概率（公式 1）

$$p_{\phi_t}(Y \succeq Y' | X) = \frac{e^{r_{\phi_t}(Y|X)}}{e^{r_{\phi_t}(Y|X)} + e^{r_{\phi_t}(Y'|X)}}$$

- $\phi_t$：reward model 在 $t$ 时刻的参数（backbone + head 的全部 weights）
- $Y, Y'$：两个 candidate responses
- $X$：prompt
- $r_{\phi_t}(\cdot|\cdot)$：标量 reward 函数
- $\succeq$：偏好关系
- 分母是 softmax normalizer，等价于 Bradley-Terry

这个公式的核心在于它把 scalar reward 转成 binary choice 的概率，恰好是 simulator 生成 choice 的逆函数，因此用 cross-entropy / log-likelihood 训练 RM 在统计上是 well-grounded 的。

### 3.2 RM 的梯度更新（公式 2）

$$\Delta\phi_t = \nabla_{\phi_t} \ln p_{\phi_t}(Y \succeq Y' | X)$$

就是伯努利对数似然的梯度。如果是被选的是 $Y$，则最大化 $\ln p(Y \succeq Y')$；如果被选的是 $Y'$，则最大化 $\ln p(Y' \succeq Y)$。等价于 cross-entropy loss，等价于 logistic regression 形式的偏好学习。注意这里没有 KL 正则，是因为 RM 学的是 absolute scalar reward，不需要 anchor。

### 3.3 Policy 的 anchor（公式 3）

$$\bar\theta_{t+1} = \eta \bar\theta_t + (1-\eta)\theta_{t+1}$$

- $\bar\theta_t$：anchor，参数的 EMA
- $\eta \in (0,1)$：EMA decay（接近 1 = anchor 移动慢）
- $\theta_{t+1}$：当前 policy 参数

直觉：anchor 是 policy "近期行为"的快照。policy 不许离 anchor 太远，相当于一种 implicit trust region。比 PPO 里 anchor 到 SFT model $\theta_0$ 更灵活——anchor 跟着 policy 走，但慢半拍。

### 3.4 Policy gradient（公式 4）

$$\Delta\theta_t = \underbrace{\left(P^-_{\bar\phi_t}(Y \succeq Y'|X) - \frac{1}{2}\right) \nabla_{\theta_t}\ln\pi_{\theta_t}(Y|X)}_{\text{(A) REINFORCE-style signal}} - \underbrace{\beta \sum_{\ell=1}^{\mathrm{len}(Y)} \pi_{\bar\theta_t}(Y_\ell|X,Y_{1:\ell-1}) \nabla_{\theta_t} \ln \frac{\pi_{\bar\theta_t}(Y_\ell|X,Y_{1:\ell-1})}{\pi_{\theta_t}(Y_\ell|X,Y_{1:\ell-1})}}_{\text{(B) KL regularization to anchor}}$$

逐项变量说明：

**Term (A)**：
- $P^-_{\bar\phi_t}(Y \succeq Y'|X)$：用 anchor RM $\bar\phi_t$ 评估的 $Y$ 被选的概率
- $-1/2$：center offset，让信号在 $[-1/2, +1/2]$ 之间对称。$P > 1/2$ 表示 RM 偏好 $Y$，gradient 推 policy 向 $Y$；$P < 1/2$ 反向推
- $\nabla_{\theta_t}\ln\pi_{\theta_t}(Y|X)$：sequence log-likelihood 的梯度，对整条 response 所有 token 求和（这里隐式 sum，REINFORCE 标准）
- 这个 term 本质是 REINFORCE with reward $r = P - 1/2$

**Term (B)**：
- $\beta$：KL 正则强度
- $\ell$：token index，从 1 到 $\mathrm{len}(Y)$
- $Y_{1:\ell-1}$：前 $\ell-1$ 个 token（前缀）
- $Y_\ell$：第 $\ell$ 个 token
- $\pi_{\bar\theta_t}(\cdot|X,Y_{1:\ell-1})$：anchor 在该位置的 next-token distribution
- $\pi_{\theta_t}(\cdot|X,Y_{1:\ell-1})$：当前 policy 在该位置的 next-token distribution
- $\ln\frac{\pi_{\bar\theta_t}}{\pi_{\theta_t}}$：log-ratio，对应反向 KL $D_{\mathrm{KL}}(\pi_{\bar\theta} \| \pi_\theta) = \mathbb{E}_{\pi_{\bar\theta}}[\ln(\pi_{\bar\theta}/\pi_\theta)]$
- 前面的 $\pi_{\bar\theta_t}(Y_\ell|...)$ 是 importance weight，相当于在 anchor 分布下求期望
- 整个 term 是 $\nabla_\theta D_{\mathrm{KL}}(\pi_{\bar\theta}\|\pi_\theta)$ 的 Monte Carlo 估计

直觉：Term (A) 是 "RM 说 Y 好，就往上推"；Term (B) 是 "别离 anchor 太远"。两项的 trade-off 由 $\beta$ 控制。这种结构非常类似 PMPO（Abdolmaleki et al., 2025, https://openreview.net/forum?id=4FVGowGzQb）和经典 PPO 中的 KL penalty。

### 3.5 Affirmative nudge（公式 5）—— 最关键的 trick

$$\Delta\theta_t = \left(P^-_{\bar\phi_t}(Y \succeq Y'|X) - \frac{1}{2} + \epsilon\right) \nabla_{\theta_t}\ln\pi_{\theta_t}(Y|X) - \beta\sum_\ell ...$$

新增 $\epsilon$ 是一个 small positive scalar。改动看起来 trivial，效果在 Figure 4(right) 上极其显著。

**直觉 1：信号整体右移**。原本 signal $s \in [-0.5, +0.5]$，现在 $s \in [-0.5 + \epsilon, +0.5 + \epsilon]$。当 $\epsilon > 0$，即使 $Y$ 被 RM 判定为略输（$P$ 略低于 0.5），整体梯度还是推 policy 向 $Y$。这意味着 **on-policy responses 永远被弱强化**，类似 implicit SFT。

**直觉 2：Tanking 的根因**。Online RLHF 中，policy 越走越窄，多样性下降。RM 在 on-policy 窄分布上训练，越来越 confident 在错误方向。一旦 RM 错了，policy 被推到 RM 偏好的烂区域，新数据继续 reinforce 这个错误。**Positive feedback loop → tanking**。

**直觉 3：$\epsilon$ 的功能**。它保证即使 RM 完全 uncertain（$P=0.5$），policy 仍然被微弱推 on-policy responses。这相当于在 REINFORCE 之上叠加了一个小的 maximum likelihood term——鼓励 policy 生成自己已经生成的东西。这避免了 mode collapse，保持 diversity，让 RM 持续在多样化数据上训练，打破 positive feedback loop。

**和 entropy bonus 的对比**：经典 RL 里防 collapse 用 entropy bonus $\mathcal{H}(\pi)$，对 token-level 分布加正则。Affirmative nudge 不同——它在 sequence 层面、通过 reward signal 形式实现，本质上是对整个 sequence 加一个 SFT-like baseline。这种实现方式更兼容 LM 训练，token-level entropy bonus 在长序列上经常失效。

**和 REINFORCE with baseline 的关系**：可以把 $\epsilon$ 理解为 baseline shift。原本 baseline $b = 1/2$，现在 baseline $b = 1/2 - \epsilon$。advantage $r - b$ 整体右移。

### 3.6 ENN 的 choice probability（公式 6）

$$p_{\phi_t}(Y \succeq Y'|X,Z) = \frac{e^{r_{\phi_t}(Y|X,Z)}}{e^{r_{\phi_t}(Y|X,Z)} + e^{r_{\phi_t}(Y'|X,Z)}}$$

新增的 $Z$ 是 **epistemic index**，整数 $0$ 到 $100$。$Z$ 是 ENN 的"epistemic perturbation"——它不是随机噪声，而是 indexing 不同的 posterior samples。

- $Z = 0$：走 point estimate head（mlp-p0）
- $Z = i \in \{1, ..., 100\}$：走第 $i$ 个 prior + differential pair

### 3.7 Exploration query 的选择（公式 7）

$$\arg\max_{Y, Y'} \mathrm{Var}\left[p_\psi(Y \succeq Y'|X, Z)\right]$$

- $Z$：在 $1, ..., 100$ 上取
- $\mathrm{Var}$：across ensemble particles 的 variance
- 选择使 variance 最大的 response pair 去问 human

**信息论直觉**：variance of choice probability across ensemble = ensemble 对这个 pair 的 disagreement = epistemic uncertainty。问人类这种 pair 能让 ensemble 收敛得最快，即信息增益最大。这是 **Information-Directed Sampling (IDS, Russo & Van Roy, 2018)** 的简化版——IDS 通常 minimize $\Delta^2/I$，这里假设 regret 不可估，只 maximize $I$。

---

## 4. ENN 架构深入

### 4.1 架构图理解

```
        X, Y
         │
         ▼
   [Transformer backbone 9B]
         │
         ▼
   [Last-layer embedding]
         │
    ┌────┴─────────────────────────────────┐
    │                                       │
    ▼                                       ▼
mlp-p0 (point estimate)                Z>0 path
MLP 2×1024, linear out              ┌─────────────────┐
    │                               │  mlp_prior_i     │ (frozen, random init)
    │                               │  MLP 2×256       │
    │                               └────────┬─────────┘
    │                                        │ +
    │                                        ▼
    │                               ┌─────────────────┐
    │                               │  mlp_diff_i     │ (trainable)
    │                               │  MLP 2×1024     │
    │                               └────────┬─────────┘
    │                                        │
    ▼                                        ▼
  r(Y|X, 0)                              r(Y|X, Z=i)
```

参数数量增加 <5%。100 个 prior networks（小，2×256）和 100 个 differential networks（大，2×1024）。

### 4.2 Randomized Prior Functions 的理论

Randomized Prior Functions (RPF, Osband et al., 2018) 是 ENN 的灵魂。形式：

$$r_i(Y|X) = \mu_i(Y|X) + p_i(Y|X)$$

- $\mu_i$：第 $i$ 个 differential network（trainable）
- $p_i$：第 $i$ 个 prior network（frozen at random init）
- 求和得到第 $i$ 个 ensemble member 的 reward

**Bayesian 视角**：考虑 Bayesian linear regression $\theta \sim \mathcal{N}(\mu_0, \Sigma_0)$，data 后验 $\theta | D \sim \mathcal{N}(\hat\mu, \hat\Sigma)$。Posterior sample 可写为 $\theta_i = \hat\mu + L z_i$，其中 $LL^\top = \hat\Sigma$，$z_i \sim \mathcal{N}(0, I)$。

RPF 把这个直接搬到神经网络：
- $\mu_i$ 学 posterior mean $\hat\mu$（differential network）
- $p_i$ 模拟 $Lz_i$（random prior，不训练）

数据稀疏区域：$\mu_i$ 还没怎么更新，$p_i$ 主导，每个 ensemble member 给出不同的 prior guess → high variance。
数据丰富区域：$\mu_i$ 都收敛到 $\hat\mu$，$p_i$ 被淹没 → low variance。

这给出 principled epistemic uncertainty estimate，比 MC dropout 这种 ad-hoc 方法更接近真正的 Bayesian posterior。

### 4.3 训练协议

1. **Point estimate** $r(Y|X, 0)$ 用 RM loss 更新，backbone + head 一起 train
2. **每个 differential network** $r(Y|X, i)$ 用同样 loss 单独更新，但 backbone frozen
3. **Prior networks** 永不更新

backbone frozen for differential 的设计很关键——否则 100 个 differential 会把 backbone 拉向 100 个不同方向，互相抵消。frozen backbone + 仅 head 学差异化，让 differential heads 之间差异来自于 prior + 各自 head 的更新历史，避免互相干扰。

---

## 5. 实验细节与数据

### 5.1 Online RLHF 的具体采样/更新协议

每 batch 64 prompts：

**RM 更新**：
- 每个 prompt 用 $\pi_{\theta_t}$ 采 **16** 个 responses
- 从 16 个里随机选 2 个问 simulator
- 用 choice 更新 RM（公式 2）

**Policy 更新（两步）**：
- **第一组 64 prompts**（已经过 RM 更新）：
  - 4 个 gradient pairs/prompt：
    - query pair + reverse（共 2 对）
    - (highest-reward, lowest-reward), (lowest, highest)（用 $r_{\phi_{t+1}}$ 排序，2 对）
  - 梯度 sum + clip + 加到 $\theta_t$
- **第二组新 64 prompts**：
  - 每 prompt 采 16 responses
  - 4 pairs: (highest, lowest), (lowest, highest), (2nd-highest, 2nd-lowest), (2nd-lowest, 2nd-highest)
  - 同样梯度 sum + clip，得到 $\theta_{t+1}$

注意 (highest, lowest) 这种 pair 选择相当于 contrastive learning——把 reward 拉得最开的两个 response 配对，gradient 信号最强。Reverse order 是为了对称性，消除 position bias。

### 5.2 Information-directed exploration 的 query 选择

每个 prompt 采 16 responses，计算所有 $\binom{16}{2} = 120$ 个 pair 的 choice probability variance across $Z \in \{1, ..., 100\}$，选 variance 最大的 pair 问 human。然后训练 protocol 同 Online RLHF，除了额外训练 ENN heads。

### 5.3 Win rate 结果（Figure 8）

| Labels | Offline RLHF | Periodic | Online RLHF | IDE |
|---|---|---|---|---|
| 1K | ~0.52 | ~0.55 | ~0.55 | ~0.58 |
| 5K | ~0.56 | ~0.62 | ~0.62 | ~0.66 |
| 20K | ~0.60 | ~0.66 | ~0.68 | ~0.72 |
| 200K | ~0.66 | ~0.70 | ~0.72 | ~0.78 |

**关键 takeaway**：IDE 在 20K labels 处达到 offline RLHF 200K 的水平。10× 提速。

### 5.4 Extrapolation（Figure 9）

$$w(n) = 1 - 0.5\left(\frac{n}{a}\right)^{-b}$$

- $w(n)$：win rate as function of labels $n$
- $a$：scale parameter（多少 label 才能达到 win rate 0.75 即 $1 - 0.5 \cdot 1 = 0.75$ 的位置）
- $b$：power-law decay rate，对应 log-log 图斜率
- $1$：上限
- $0.5$：baseline equal performance

**为什么 0.5 是下限**：$n \to 0$ 时 $w \to 0.5$，对应随机选择（policy 没改进，win rate = 50%）。$n \to \infty$ 时 $w \to 1$，policy 完全超越 baseline。

在 log-log 上斜率 $-b$ 越大，scaling 越快。Offline RLHF 的 $b$ 较小，IDE 的 $b$ 较大。Extrapolation 给 1M labels → 1000× gain。

---

## 6. 关键直觉汇总

### 6.1 为什么 offline 不 scale

Offline RLHF 加更多 data 只是把 RM 在 $\pi_{\theta_0}$ support 上拟合得更准。但 policy 想去的好 responses 可能根本不在 $\pi_{\theta_0}$ 的 high-density region。这是 **support coverage** 问题，不是 data 量问题。

### 6.2 为什么 online 单独不够

Online 解决 distribution shift，但有 tanking 风险。Policy 越走越窄、RM 在窄分布上越来越 confident——一旦走偏就 self-reinforcing 错误。Affirmative nudge 通过 always-positive signal 阻止 collapse。

### 6.3 为什么 exploration 必要

On-policy sampling 保证了 distribution match，但**随机** pair 不一定 informative。如果 RM 已经对大部分 on-policy pairs 很 confident，问了也学不到新东西。ENN-based variance selection 让每个 query 都最大化 information gain。

### 6.4 三者协同的乘性效应

- Online 解决 coverage：每次 query 都在 policy 关心的区域
- Nudge 解决 collapse：policy 保持 diverse，RM 不陷入 positive feedback loop
- IDE 解决 informativeness：每个 query 都最大化信息增益

三者不是简单叠加，是**乘性**的——任何一个缺失都会大幅降低 sample efficiency。

---

## 7. 与相关工作的 positioning

### 7.1 跟 DPO 的对比

DPO（Rafailov et al., 2023, https://arxiv.org/abs/2305.18290）直接在 preference data 上训练 policy，绕过显式 RM。本文 Figure 4(left) 显示 RM-free online 不够强——RM 提供 generalization，能在没见过的 prompt 上 extrapolate，直接 policy 更新做不到这种 generalization。

### 7.2 跟 XPO / APO / IDS-RLHF 对比

- XPO（Xie et al., 2025, https://arxiv.org/abs/2405.21047）：加 exploration bonus 到 DPO objective
- APO（Das et al., 2025）：active learning 在 DPO 上
- 这些都报告 2×–5× gain。本文 10×–1000× 的差距来自：1）online RM（而不是 only policy）；2）ENN 而非 MC dropout；3）true IDS（variance maximization）而非 heuristic uncertainty。

### 7.3 跟 ActiveDPO（Lin et al., 2026, https://openreview.net/forum?id=RD4XgyVyGh）对比

ActiveDPO 也是 active learning + DPO，但用 diversity-based acquisition。本文用 ENN variance，更 principled，更好对齐 Bayesian information gain。

### 7.4 跟 PMPO 的关系

PMPO（Abdolmaleki et al., 2025, https://openreview.net/forum?id=4FVGowGzQb）是 policy manifold 的 constrained optimization。本文 policy update rule (公式 4) 本质是 PMPO 的 variant——KL constraint 到 anchor，policy 在 manifold 上滑动。Affirmative nudge 是本文在 PMPO 之上的关键 modification。

### 7.5 跟 Scaling Laws 的关系

Hoffmann et al. (2022, Chinchilla, https://arxiv.org/abs/2206.06669) 给 pretraining 的 compute-optimal scaling。Kaplan et al. (2020) 给经典 LLM scaling law。本文第一次给 RLHF 的 scaling law：在 log-log 图上，offline 是低斜率 power law，online+IDE 是高斜率。**RLHF 不是不 scale，是 offline 不 scale**。

---

## 8. 局限与开放问题

### 8.1 模拟人类的偏差

Gemini 1.5 Pro 当 simulator 有风险——它本身可能偏好 LLM-style responses，与真实人类分布有 gap。真实人类标注可能有 noise、inconsistency、preference drift。不过 simulator 远大于被训练模型这点至少保证 preference 比 9B 复杂。

### 8.2 ENN 计算开销

100 ensemble members，inference 成本 ×100。Point estimate 路径可以共享 backbone forward，但 100 个 head 还是要算。训练 differential 时 backbone frozen 反而便宜。整体 OK，但 scale 到更大模型时可能瓶颈。

### 8.3 信息增益代理的局限

Variance of choice probability 是 information gain 的代理。当 RM 在某个区域 systematically biased（而非 uncertain）时，variance 可能低但实际错。理论上 ENN 应该捕获这种 aleatoric / epistemic 区分，但实际可能不完美。更严格的信息增益估计见 Russo & Van Roy 2018 的 posterior-based IDS。

### 8.4 Future direction：prompt selection

论文 6 节提到可以扩展到 **prompt selection**——不只选 responses，还选哪些 prompts 问。这是 active learning 的自然延伸，但需要 prompt-level uncertainty model。

### 8.5 Future direction：multiturn / agents

Marklund & Van Roy (2024, https://arxiv.org/abs/2410.22690) 引入 value model 来处理 delayed consequences。本文方法可扩展到多轮对话和 agent setting。

### 8.6 Future direction：AI-assisted feedback

Irving et al. (2018) 的 AI safety via debate（https://arxiv.org/abs/1805.00899）——AI 生成 debate rationale，人类 validate。本文方法可应用到这种 richer feedback structure。

---

## 9. 个人 Intuition 总结

如果你 build intuition，把整篇论文浓缩成三句话：

1. **Offline RLHF 死于 distribution shift**——fixed sampling distribution 限制了 RM 能学到什么。
2. **Online RLHF 死于 mode collapse**——正反馈 loop 让 policy-RM 系统在小分布上自欺欺人。Affirmative nudge 通过 always-positive signal 强制保持 diversity。
3. **Random exploration 浪费 query**——On-policy 随机 sample 的 pair 经常是 RM 已经确定的。ENN-based variance selection 让每个 query 都最大化信息增益。

更深一层的方法论启示：**scaling law 在 RLHF 里不是 monolithic**。Offline、online、online+exploration 是不同的 scaling regimes，斜率天差地别。看 Figure 1 log-log 图就明白——线性的横轴隐藏了这件事，log 才看出 qualitative difference。**未来的 alignment 工作报告 scaling 应该都用 log scale**。

最后一层启示：**epistemic uncertainty 在 alignment 里是 first-class citizen**。不是 nice-to-have，是 sample efficiency 的关键 multiplier。ENN 这种把 uncertainty 直接 baked into architecture 的方法，比 dropout / ensemble-of-full-models 这种 post-hoc 方法在 compute / expressiveness trade-off 上明显更好。值得在所有 RLHF pipeline 里默认加上。

参考综述：
- DeepMind Efficient Exploration 系列：https://sites.google.com/view/efficient-exploration
- Epistemic Neural Networks 仓库：https://github.com/deepmind/enn
- Randomized Prior Functions 实现：https://github.com/instadeepai/keras-ncp

如果想 build 更深 intuition，强烈推荐先读 Osband et al. 2023 ENN 论文和 Osband et al. 2018 RPF 论文，这两篇是本文方法学的根基。然后读 Dwaracherla et al. 2024 看 RM-only 版本，最后本文是 RM + LM 端到端版本。这条 reading path 让整个方法论的演化非常清晰。
