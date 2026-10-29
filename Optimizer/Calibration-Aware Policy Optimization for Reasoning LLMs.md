---
source_pdf: Calibration-Aware Policy Optimization for Reasoning LLMs.pdf
paper_sha256: ef48e3589551bdfb9a9a5f29dab99828a0084b2e630bc942b00d3f51a83365b6
processed_at: '2026-08-03T14:39:27-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话总结

**GRPO这类RLVR算法把model教成了"自信的傻瓜"——错答案说得比对答案还溜，而且作者从数学上证明了为什么会这样，然后给出了一个fix。**

---

## 背景知识：什么是calibration，为什么care

假设你问一个model一道数学题，它给你一个答案。你怎么知道该不该信它？

理想情况下：model说"我90%确定"的时候，90%是对的；说"我50%确定"的时候，一半对一半。这就叫**well-calibrated**。

问题是我们没法直接问model"你多确定"——它会乱说。所以我们用**perplexity (PPL)** 当proxy：PPL低说明model对自己生成的内容很confident，PPL高说明它uncertain。

well-calibrated的意思就是：**对答案的PPL应该低于错答案的PPL**。

Anthropic在2022年那篇"Language Models (Mostly) Know What They Know" (https://arxiv.org/abs/2207.05271) 发现：base model天生就挺well-calibrated的。

---

## 核心问题：GRPO把calibration搞坏了

DeepSeek-R1, o1这些reasoning model都是用GRPO训练的。GRPO的核心思路：

```
给model一道题，让它生成一堆答案（比如16个），
对的对，错的错，
对的答案给它positive advantage，错的给negative advantage，
梯度下降。
```

听起来很合理对吧？但实验发现一个诡异现象：

> **训练得越久，model越overconfident——错答案的PPL反而比对答案还低了。**

Figure 1(b)里的曲线非常直观：accuracy在涨，AUC（衡量calibration的指标）在稳步跌。

---

## 为什么会这样？理论分析

这是paper最精彩的部分。作者做了个"等价变换"，把GRPO的gradient写成另一种形式，问题就暴露了。

### Step 1: GRPO到底在优化什么

GRPO的advantage长这样：
$$\hat{A}_{i,t} = \frac{R_i - \mathrm{mean}(\{R_i\}_{i=1}^G)}{\mathrm{std}(\{R_i\}_{i=1}^G)}$$

其中：
- $R_i \in \{0,1\}$ 是第$i$个response的reward（对就是1，错就是0）
- $G$是group size
- 就是把reward做组内normalize

它的gradient经过一串推导，本质上等价于：

$$\mathbb{E}\left[\sum_{i<j} (\nabla_\theta \mathrm{lpm}(o_i) - \nabla_\theta \mathrm{lpm}(o_j))(R_i - R_j)\right]$$

其中 $\mathrm{lpm}(o) = \frac{1}{|o|}\sum_t \log\pi(o_t|o_{<t})$ 是log perplexity。

翻译成人话：**对每一对答案，如果$i$对$j$错，就把$i$的PPL推低、$j$的PPL推高。**

### Step 2: 这其实是AUC optimization的一种

AUC的定义：随机抽一对(correct, wrong)，correct的confidence比wrong高的概率。

AUC optimization的标准做法是：找一个surrogate loss替代不可微的indicator function。常见的surrogate有logistic loss, hinge loss, exponential loss等。

GRPO的gradient对应的surrogate是 $\phi(t) = -t$——**最朴素的那种linear loss**。

### Step 3: $\phi(t) = -t$ 是broken的

Theorem 3的证明特别干净：

**AUC是scale-invariant的**：你把所有confidence score乘以100倍，排序不变，AUC不变。

**但$\phi(t) = -t$的surrogate risk是scale-sensitive的**：
$$\mathcal{L}_{-t}(\alpha f) = \alpha \mathcal{L}_{-t}(f)$$

构造一个序列：$f_m = \alpha_m f$，让$\alpha_m \to \infty$。

- AUC纹丝不动（scale-invariant）
- Surrogate risk $\mathcal{L}_{-t}(f_m) \to -\infty$（一直在降）

**所以优化surrogate可以一直"进步"，但AUC根本没改善。** 这就是inconsistency。

### Intuition: 为什么会overconfident

GRPO的gradient只看reward，不看model自己的confidence。所以它的行为是：

```
对每个correct response: 无脑push down PPL
对每个wrong response: 无脑push up PPL
```

问题在于：**如果某个correct response的PPL已经很低了（已经很confident了），GRPO还是会继续push**。如果某个wrong response碰巧PPL也挺低的，GRPO会使劲push它变高。

结果就是所有PPL都被往一个方向压，correct和wrong之间的**gap反而被压平了**。

打个比方：老师批作业，对的题使劲夸，错的题使劲骂，完全不管学生本来对哪题有把握、哪题没把握。结果学生学会了"所有题都装作很确定"——自信心爆棚，但完全不可信。

---

## CAPO: Calibration-Aware Policy Optimization

### 核心idea

换个surrogate loss，让gradient**自动忽略已经well-ranked的pair，focus在misranked的pair上**。

具体用logistic loss：
$$\phi_\tau(t) = \log(1 + \exp(-t/\tau))$$

其中：
- $t = (\mathrm{lpm}(o_i) - \mathrm{lpm}(o_j))(R_i - R_j)$ 是margin
- $\tau$是temperature，1.5B用0.6, 7B用0.5

为什么这个work？看它的derivative $\phi'(t) = -\sigma(-t/\tau)/\tau$：

| 情况 | margin $t$ | $\phi'(t)$ | gradient magnitude |
|------|-------------|-------------|-------------------|
| Correct已经比wrong confident很多 | $t \gg 0$ | $\approx 0$ | **小**（忽略） |
| Correct和wrong confidence差不多 | $t \approx 0$ | $\approx -1/2\tau$ | **中等**（关注） |
| Correct反而比wrong less confident（misranked） | $t \ll 0$ | $\approx -1/\tau$ | **大**（重点纠正） |

这就是Figure 2的曲线含义：**misaligned的pair gradient最大，well-aligned的pair gradient几乎为零**。

人话：**与其对所有题都使劲，不如把精力放在"学生以为自己对其实错了"和"学生以为自己错其实对了"的题上**。

### 新的advantage function

推导后的形式：
$$\tilde{A}_i = \begin{cases} 
-\sum_{j: R_j=0} \phi'(\mathrm{lpm}(o_i) - \mathrm{lpm}(o_j)), & R_i = 1 \\
\sum_{j: R_j=1} \phi'(\mathrm{lpm}(o_j) - \mathrm{lpm}(o_i)), & R_i = 0
\end{cases}$$

展开说：
- **如果你是个correct response**，你的advantage = sum over所有wrong responses的margin的sigmoid
  - 如果你已经比所有wrong都confident了 → advantage ≈ 0 → 基本不更新
  - 如果某些wrong比你还confident → advantage大 → 重点push down PPL
- **如果你是个wrong response**，对称的
  - 如果所有correct都比你confident → advantage ≈ 0 → 基本不更新
  - 如果你比某些correct还confident → advantage大 → 重点push up PPL

### Reference-Model-Based Noise Masking

还有个实际工程问题：binary reward会引入noise。

**场景1 - Lucky guess**: 推理乱七八糟，最后蒙了个对答案。reward=1但reasoning是垃圾。
**场景2 - Near-miss**: 推理99%对，最后一步小数点错了。reward=0但reasoning质量高。

这些样本会注入noisy gradient。作者的fix：

用**reference model (base model)的PPL**来判断reasoning质量：
- 高PPL_ref → reasoning大概有syntactic/logical flaw → 如果reward=1，可能是lucky guess，mask掉
- 低PPL_ref → reasoning大概coherent → 如果reward=0，可能是near-miss，mask掉

$$m(o) = \begin{cases}
\mathbb{I}[\mathrm{PPL}_{\mathrm{ref}}(o) \le \text{ref-high}], & R(o) = 1 \\
\mathbb{I}[\mathrm{PPL}_{\mathrm{ref}}(o) \ge \text{ref-low}], & R(o) = 0
\end{cases}$$

阈值：ref-high=2.5, ref-low=1.05（base model PPL分布的上下quartile）。

为什么这个work？因为base model是well-calibrated的（Anthropic那篇paper的发现），它的PPL是有意义的signal。

零额外训练开销，只在data preprocessing阶段算一次reference model的PPL。

### 最终objective

把GRPO的advantage换成CAPO的，保留PPO的clipping：
$$J_{\mathrm{CAPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t \min(r_{i,t}\hat{A}_i^{\mathrm{CAPO}}, \mathrm{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i^{\mathrm{CAPO}})\right]$$

其中 $\hat{A}_i^{\mathrm{CAPO}} = m(o_i) \tilde{A}_i$。

---

## 实验数据

### 主结果

6个benchmark，Qwen2.5-Math-1.5B和7B。AIME 2025上的数字最impressive：

| Model | GRPO AUC | CAPO AUC | 提升幅度 |
|-------|----------|----------|---------|
| 1.5B | 0.63 | 0.78 | **+15%** |
| 7B | 0.54 | 0.79 | **+25%** |

同时accuracy保持或提升。Figure 3显示CAPO在所有6个benchmark上都dominate GRPO。

### 和其他calibration方法对比

| 方法 | Calibration提升 | Accuracy影响 |
|------|----------------|-------------|
| CoDaPO | 有限 | 略有影响 |
| CDE | 有限 | 略有影响 |
| SimKO | 有提升 | **严重下降**（AMC -12%, AIME -7.7%）|
| **CAPO** | **显著** | **保持/提升** |

### Inference-Time Scaling效果

用Perplexity-Consistency算法（N=16 samples，aggregate confidence）：

| Model | CAPO | GRPO | 提升 |
|-------|------|------|------|
| 1.5B | 25.33 | 20.33 | **+5%** |
| 7B | 38.33 | 33.33 | **+5%** |

更好的calibration → confidence signal更可信 → inference-time aggregation更准确 → accuracy提升。

### Hallucination Mitigation

Precision-Coverage曲线：让model在confidence低时abstain（拒答），varying threshold画曲线。

CAPO达到Pareto-optimal：在所有coverage水平上precision都≥其他方法。

### Ablation关键发现

1. **去掉masking**：model entropy逐渐升高，accuracy停滞
2. **只用masking不用logistic surrogate**：AUC不改善
3. **超参数insensitive**：$\tau \in \{0.4, 0.6, 1.0\}$都work，masking interval也insensitive

---

## 为什么这篇paper重要

### 1. 揭示了一个systematic的问题

不只是GRPO，**任何reward-only的RLVR算法都有这个缺陷**。GSPO在Figure 1(c)也显示同样的calibration degradation。DAPO应该也一样。

这个proof是general的：只要advantage estimator只依赖reward不依赖uncertainty，对应的surrogate都是linear的，都scale-sensitive，都inconsistent。

### 2. 理论driven的解决方案

之前的工作（CoDaPO, CDE, SimKO）都是heuristic的——基于"我觉得应该这样"的设计。CAPO是从AUC consistency理论推导出来的，有regret bound guarantee。

### 3. 对工业界的启示

DeepSeek-R1, o1这些model都是用GRPO训练的。按这篇paper的分析，**它们很可能都overconfident**。这可能解释了为什么这些model还是会confidently hallucinate。

CAPO的overhead几乎为零（只是data preprocessing时多算一次reference PPL），可能成为下一代reasoning model training的standard component。

### 4. 更深层的intuition

Binary reward的信息量是1 bit。Model的log-probability是continuous signal。GRPO丢掉了后者。

CAPO通过logistic surrogate把model自己的uncertainty重新注入advantage estimation。这和self-distillation, Born-Again Networks的思想相通——**用model自己的预测来inform训练**。

从information-theoretic视角：GRPO的gradient只利用了reward的1 bit信息，CAPO额外利用了log-prob的continuous信息。

---

## 我的延伸思考

### 可能的extension

1. **DPO也有类似问题**：DPO的likelihood displacement (https://arxiv.org/abs/2306.02231) 本质也是calibration issue。CAPO的框架可能能fix
2. **RLHF也有类似问题**：只要reward是scalar的，advantage就是reward-only的，都有inconsistency
3. **Process Reward Model可以替代reference PPL**：更强的reasoning quality assessment，但trade-off是计算开销
4. **Extension到非math reasoning**：code generation, tool use, logical puzzles——理论是general的

### 一个speculation

GRPO的inconsistency可能和RLVR的scaling瓶颈有关。当model越来越强，binary reward的信息量相对越来越不够。CAPO通过引入uncertainty signal，可能是scaling RLVR的一个direction。

---

## 参考链接

- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- GSPO: https://arxiv.org/abs/2507.18071
- DAPO: https://arxiv.org/abs/2503.14476
- CoDaPO: https://arxiv.org/abs/2507.08089
- CDE: https://arxiv.org/abs/2509.09675
- SimKO: https://arxiv.org/abs/2510.14807
- Kadavath et al. (base model calibration): https://arxiv.org/abs/2207.05271
- AUC consistency theory (Gao & Zhou): https://arxiv.org/abs/1208.0645
- Bereket & Leskovec (GRPO overconfidence): https://arxiv.org/abs/2508.11800
- C²GSPG: https://arxiv.org/abs/2509.23129
- Xiao et al. (calibration-aware fine-tuning): https://arxiv.org/abs/2505.01997
- Qwen2.5-Math: https://arxiv.org/abs/2409.12122
- Let's Verify Step by Step (PRM): https://arxiv.org/abs/2305.20050
- DPO: https://arxiv.org/abs/2305.18290
- Likelihood displacement in DPO: https://arxiv.org/abs/2306.02231
- DeepScaler dataset: https://github.com/agentica-project/deepscaler
- Inference-time scaling (Perplexity Consistency): https://arxiv.org/abs/2502.00511
- Uncertainty as control signal: https://arxiv.org/abs/2509.02401
- Minimum Bayes Risk: https://arxiv.org/abs/2502.04964

---

# Calibration-Aware Policy Optimization for Reasoning LLMs 深度讲解

## 1. 这篇paper要解决的核心问题

GRPO (Group Relative Policy Optimization) 是当前LLM reasoning训练的主流算法，DeepSeek-R1, OpenAI o1系列的训练pipeline都依赖GRPO-style的RLVR (Reinforcement Learning from Verifiable Rewards)。这篇paper发现并从理论上证明了一个非常fundamental的问题：**GRPO在提升accuracy的同时会systematically破坏model calibration**。

具体表现为：训练后的模型会产生**overconfident wrong answers**——错误response的perplexity反而比正确response更低。这意味着model的内部confidence signal完全失效，对于hallucination mitigation, inference-time scaling, multi-agent routing等下游应用是毁灭性的。

这一点和Kadavath et al. 2022 (Anthropic的"Language Models (Mostly) Know What They Know" https://arxiv.org/abs/2207.05271) 中发现的base model well-calibrated现象形成鲜明对比——RL post-training把base model原本良好的calibration给破坏掉了。

---

## 2. Calibration的两种定义

### 2.1 Absolute calibration (绝对校准)
形式化定义为：
$$P(R=1 \mid f(q,o)=k) = k$$

其中：
- $R \in \{0,1\}$ 表示response的ground truth correctness
- $f(q,o)$ 表示confidence scoring function
- $k$ 表示某个confidence level

意思是：当model说自己有$k$的confidence时，实际上correct的概率就是$k$。这是Expected Calibration Error (ECE) 衡量的目标。

### 2.2 Relative calibration (相对校准)
本文focus的是relative calibration，用AUC衡量：
$$\mathrm{AUC}(\pi, q, f) = \mathbb{E}_{o_i, o_j \sim \mathcal{D}}[\mathbb{I}((R_i - R_j)(f(o_i) - f(o_j)) > 0)]$$

变量含义：
- $o_i, o_j$ 表示同一问题$q$随机采样的两个response
- $R_i, R_j$ 表示它们的correctness label
- $f(\cdot)$ 表示confidence scoring function (本文用perplexity的负值)
- $\mathbb{I}(\cdot)$ 表示indicator function

AUC的intuition：随机抽一对(correct, wrong) response，correct的confidence高于wrong的概率。

作者选择relative calibration而非absolute calibration的两个理由很关键：
1. PPL是free-form generation的uncertainty proxy，它和correctness probability不可直接比较，所以absolute calibration在数学上ill-defined
2. Absolute calibration是aggregate statistical property，不能保证instance-level discrimination。极端例子：给所有response都赋0.5的confidence，绝对满足absolute calibration但完全没用

---

## 3. 为什么GRPO会破坏Calibration——理论分析

这是这篇paper最有价值的部分。作者把GRPO的gradient重新推导成了一个AUC optimization的形式，揭示了它的根本缺陷。

### 3.1 GRPO Gradient的等价形式

GRPO原始的advantage计算：
$$\hat{A}_{i,t} = \frac{R_i - \mathrm{mean}(\{R_i\}_{i=1}^G)}{\mathrm{std}(\{R_i\}_{i=1}^G)}$$

变量含义：
- $R_i \in \{0, 1\}$ 表示第$i$个response的binary reward
- $G$ 表示group size
- 分子是group-relative reward，分母是group reward的标准差

作者通过一系列推导证明，GRPO的gradient等价于（一阶近似下）：
$$\mathbb{E}_{o_{1:G} \sim \mathcal{D}}\left[\frac{1}{G}\sum_{i=1}^G \hat{A}_i \nabla_\theta \mathrm{lpm}_\theta(o_i)\right]$$

其中：
$$\mathrm{lpm}_\theta(o_i) = \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \log \pi_\theta(o_{i,t} \mid o_{i,<t})$$

这是log perplexity (negative log perplexity, lpm)。lpm越高表示PPL越低表示confidence越高。

### 3.2 Pairwise rewriting via U-statistics

下一步关键推导利用了**U-statistics**的无偏性。把上面的sum重写为pairwise form：

$$\mathbb{E}_{o_{1:G} \sim \mathcal{D}}\left[\sum_{1 \le i < j \le G} (\nabla_\theta \mathrm{lpm}_\theta(o_i) - \nabla_\theta \mathrm{lpm}_\theta(o_j))(R_i - R_j)\right]$$

由于U-statistics的无偏性，这等价于对一个random pair的expectation：
$$\mathbb{E}_{o_1, o_2 \sim \mathcal{D}}\left[(\nabla_\theta \mathrm{lpm}_\theta(o_1) - \nabla_\theta \mathrm{lpm}_\theta(o_2))(R_1 - R_2)\right]$$

这恰好就是AUC optimization with surrogate loss $\phi(t) = -t$ 的gradient形式！

### 3.3 Inconsistency证明

Theorem 3的核心：$\phi(t) = -t$ 是inconsistent的AUC surrogate。

证明思路非常elegant，基于scale invariance：
- AUC是scale-invariant的：$\mathrm{AUC}(\alpha f) = \mathrm{AUC}(f)$ for any $\alpha > 0$
- 但$\phi(t) = -t$的surrogate risk是scale-sensitive的：
  $$\mathcal{L}_{-t}(\alpha f) = \alpha \mathcal{L}_{-t}(f)$$

构造反例序列：$f_m = \alpha_m f$ with $\alpha_m \to \infty$
- AUC不变（因为scale-invariant）
- Surrogate risk $\mathcal{L}_{-t}(f_m) \to -\infty$

所以最小化surrogate risk可以一直下降，但AUC纹丝不动。这就是inconsistency——优化目标diverge from真实目标。

Intuition：GRPO的gradient只考虑reward的相对差异，完全没考虑model自己的uncertainty。它会一直推高所有correct response的log-prob，压低所有wrong response的log-prob，导致**所有response的PPL都一起下降**——这就是为什么训练曲线图1(b)显示AUC稳步恶化，因为correct和wrong的PPL gap没有拉开，反而被一起压平了。

这个分析的普适性很重要——任何只用reward的advantage estimator（包括GSPO, DAPO）都有这个缺陷。GSPO虽然在sequence-level优化，但advantage仍是reward-based的，所以Figure 1(c)显示GSPO同样degrade calibration。

---

## 4. CAPO方法

### 4.1 Consistent Logistic Surrogate

替换为logistic surrogate loss：
$$\phi_\tau(t) = \log(1 + \exp(-t/\tau))$$

变量含义：
- $t$ 表示margin (correct和wrong的confidence gap)
- $\tau > 0$ 是temperature parameter，控制smoothness
- $\tau$ 小时接近hinge loss，$\tau$ 大时更smooth

根据Theorem 1 (Gao and Zhou 2012, https://arxiv.org/abs/1208.0645)，任何convex, differentiable, non-increasing且$\phi'(0) < 0$的surrogate都是AUC-consistent的。logistic loss满足这些条件。

Theorem 2给出regret bound：
$$L(f) - L^* \le \frac{1}{\ln 2}(L_\phi(f) - L_\phi^*)$$

意思是：surrogate risk的gap会被$1/\ln 2 \approx 1.44$这个常数factor bound住AUC risk的gap。这给了理论保证——优化surrogate一定会improve AUC。

### 4.2 优势函数推导

Policy optimization objective：
$$J_{\mathrm{logistic}}(\theta) = -\mathbb{E}_{o_1, o_2 \sim \mathcal{D}}[\log(1 + \exp(-t/\tau))]$$

其中：
$$t = (\mathrm{lpm}_\theta(o_1) - \mathrm{lpm}_\theta(o_2))(R_1 - R_2)$$

对$\theta$求gradient后，重新写回group sum form：
$$\nabla_\theta J_{\mathrm{logistic}}(\theta) = \mathbb{E}_{o_{1:G} \sim \mathcal{D}}\left[\frac{1}{G}\sum_{i=1}^G \tilde{A}_i \nabla_\theta \mathrm{lpm}_\theta(o_i)\right]$$

新的advantage：
$$\tilde{A}_i = \begin{cases} 
-\sum_{j: R_j=0} \phi'(\mathrm{lpm}_\theta(o_i) - \mathrm{lpm}_\theta(o_j)), & R_i = 1 \\
\sum_{j: R_j=1} \phi'(\mathrm{lpm}_\theta(o_j) - \mathrm{lpm}_\theta(o_i)), & R_i = 0
\end{cases}$$

其中$\phi'(t) = -\sigma(-t)$，$\sigma(\cdot)$是sigmoid。

Intuition展开：
- 对于correct response $o_i$ ($R_i=1$)：它的advantage是所有wrong response的margin的sigmoid负值之和。如果$o_i$的PPL已经比所有wrong都低（confidence已经很高），$\phi'$接近0，advantage接近0——不浪费gradient
- 对于wrong response $o_i$ ($R_i=0$)：它的advantage是所有correct response与它的margin的sigmoid负值之和。如果某个wrong response的PPL比correct还低（calibration错误），$\phi'$会接近1，advantage绝对值大——重点纠正

### 4.3 Gradient shape分析

这是paper里最informative的部分之一。看Figure 2——$\phi'(t) = -\sigma(-t)$的形状：
- 当margin $t$很大时（confidence gap正确），$\phi'(t) \to 0$，gradient contribution小
- 当margin $t$很小或负（misaligned confidence），$|\phi'(t)|$大，gradient contribution大

这就是CAPO的核心mechanism：**自动prioritize那些confidence misaligned的samples**。具体说：
- Correct samples with relatively high PPL (underconfident correct) → 需要push up
- Wrong samples with relatively low PPL (overconfident wrong) → 需要push down

这和GRPO对所有correct/wrong一视同仁形成鲜明对比。

### 4.4 Reference-Model-Based Noise Masking

这个设计是为了解决binary reward的inherent noise问题：
- **Lucky guess**: 推理错误但碰巧答对 → reward=1但reasoning质量差
- **Near-miss**: 推理接近正确但有小错误 → reward=0但reasoning质量好

这些样本会注入noisy gradient破坏训练。作者利用**reference model (base model)本身的calibration**来filter：
- Base model well-calibrated是已知现象（Kadavath et al. 2022）
- 高PPL_ref通常表示syntactic或logical flaw
- 低PPL_ref通常表示coherent reasoning

Masking策略：
$$m(o) = \begin{cases}
\mathbb{I}[\mathrm{PPL}_{\mathrm{ref}}(o) \le \text{ref-high}], & R(o) = 1 \\
\mathbb{I}[\mathrm{PPL}_{\mathrm{ref}}(o) \ge \text{ref-low}], & R(o) = 0
\end{cases}$$

阈值：ref-high=2.5, ref-low=1.05，对应reference model PPL分布的上下quartile。

这个设计的巧妙之处：**零额外训练开销**，只是inference时算一下reference model的PPL。

### 4.5 Final Objective

最终CAPO objective：
$$J_{\mathrm{CAPO}}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\mathrm{old}}}}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min(r_{i,t}(\theta)\hat{A}_i^{\mathrm{CAPO}}, \mathrm{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i^{\mathrm{CAPO}})\right]$$

其中：
$$\hat{A}_i^{\mathrm{CAPO}} = m(o_i) \tilde{A}_i$$

保留PPO-style的clipping mechanism保证trust region更新，但advantage替换为calibration-aware的版本。

---

## 5. 实验数据详解

### 5.1 主实验结果

实验在Qwen2.5-Math-1.5B和7B上进行，6个benchmark：AIME 2024, AIME 2025, MATH 500, AMC 2023, Minerva, Olympiad-Bench。

关键数字（AIME 2025上）：
- **1.5B模型**: AUC从0.63 (GRPO) 提升到0.78 (CAPO)，**+15%**
- **7B模型**: AUC从0.54 (GRPO) 提升到0.79 (CAPO)，**+25%**

同时accuracy保持在GRPO水平甚至更好。这是非常impressive的——基本上是"免费的"calibration提升。

### 5.2 Baselines对比

对比的其他calibration方法：
- **CoDaPO** (Zhou et al. 2025a, https://arxiv.org/abs/2507.08089): confidence and difficulty-adaptive policy optimization
- **CDE** (Dai et al. 2025, https://arxiv.org/abs/2509.09675): curiosity-driven exploration
- **SimKO** (Peng et al. 2025, https://arxiv.org/abs/2510.14807): simple pass@k policy optimization

这些方法的问题：
- CoDaPO, CDE：calibration提升有限
- SimKO：calibration有提升但accuracy严重下降（AMC掉12%, AIME 2024掉7.7%）

### 5.3 Inference-Time Scaling效果

用Zhou et al. 2025b的Perplexity-Consistency算法（Algorithm 1），N=16：

| Model | CAPO | GRPO | GSPO | CoDaPO | CDE | SimKO |
|-------|------|------|------|--------|-----|-------|
| 1.5B | 25.33 | 20.33 | 20.00 | 21.67 | 16.67 | 11.67 |
| 7B | 38.33 | 33.33 | 32.21 | 31.66 | 31.66 | 23.33 |

CAPO在1.5B上比GRPO高**5%**，在7B上同样高**5%**。这证明calibration对inference-time scaling至关重要——更好的calibration意味着confidence signal更可信，aggregation更准确。

### 5.4 Hallucination Mitigation

Precision-Coverage曲线显示CAPO达到Pareto-optimal trade-off：
- SimKO在low coverage时precision高（因为ranking ability好），但coverage上升时accuracy低导致precision暴跌
- GRPO accuracy高但calibration差，precision始终低
- CAPO两者兼得

### 5.5 Ablation Studies

1. **超参数敏感性**: $\tau \in \{0.4, 0.6, 1.0\}$，masking interval $[1.05, 2.5]$ vs $[1.25, 2.1]$，性能变化marginal
2. **Masking mechanism**: 去掉masking会导致model entropy逐渐升高，accuracy停滞甚至下降
3. **关键ablation**: 只对GRPO加masking不加logistic surrogate，AUC不改善——证明consistent surrogate才是核心，masking只是辅助

---

## 6. 与相关工作的关系

### 6.1 AUC Optimization理论
这篇paper站在Gao and Zhou 2012 (https://arxiv.org/abs/1208.0645) 的AUC consistency理论上。其他相关工作：
- Kotlowski et al. 2011 (http://proceedings.mlr.press/v15/kotlowski11a.html): bipartite ranking
- Zhao et al. 2011: online AUC maximization
- Yuan et al. 2021 (https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_ICCV_2021_paper): deep AUC maximization

### 6.2 RLVR Calibration问题
- Bereket and Leskovec 2025 (https://arxiv.org/abs/2508.11800): "Uncalibrated reasoning: GRPO induces overconfidence for stochastic outcomes"——类似发现
- Liu et al. 2025a (https://arxiv.org/abs/2509.23129): C²GSPG, confidence-calibrated GSPO
- Xiao et al. 2025 (https://arxiv.org/abs/2505.01997): calibration-aware fine-tuning

### 6.3 Inference-Time Scaling
- Stoisser et al. 2025 (https://arxiv.org/abs/2509.02401): uncertainty as control signal
- Vashurin et al. 2025 (https://arxiv.org/abs/2502.04964): minimum Bayes risk
- Zhou et al. 2025b (https://arxiv.org/abs/2502.00511): Perplexity Consistency

---

## 7. 我的Intuition与延伸思考

### 7.1 这个发现的更深层意义

GRPO的inconsistency问题本质上是**reward-only signal的信息不充分**。Binary reward只告诉你"对/错"，不告诉你"model自己有多confident"。这类似于supervised learning里只用0/1 label而不用logit——丢掉了大量信息。

CAPO的解决方案相当于**把model自己的uncertainty作为auxiliary signal注入advantage estimation**。这和self-distillation, Born-Again Networks的思想有相通之处——用model自己的预测来inform训练。

### 7.2 与DPO/IPO的关系

这个工作让我想到DPO的derivation——DPO也是从reward optimization推导出一个closed-form的policy loss。CAPO类似地从AUC optimization推导出advantage的形式。两者都是把一个non-differentiable的目标转化为可优化的形式。

DPO的失效模式也包括calibration问题（Gao et al. 2023的likelihood displacement问题 https://arxiv.org/abs/2306.02231），CAPO的思路可能也适用于DPO-style算法。

### 7.3 Limitations讨论

作者承认只在math reasoning上验证。但理论分析是general的——任何binary reward的RLVR算法都有这个问题。Potential extension：
- Code generation (verifiable via tests)
- Tool use (verifiable via tool output)
- Multi-step reasoning with intermediate verification

### 7.4 与Process Reward Model的关系

CAPO用reference model的PPL作为masking signal。如果用更强的Process Reward Model (PRM, Lightman et al. 2023, https://arxiv.org/abs/2305.20050) 来assess reasoning quality，masking可能更精确。但trade-off是计算开销。

### 7.5 与Constitutional AI / RLAIF的对比

OpenAI/Anthropic的RLHF/RLAIF pipeline是否也有类似calibration degradation？CAPO的分析框架可以扩展——只要advantage是reward-only的，都有inconsistency问题。这意味着所有RLHF后的model都可能overconfident，这可能解释了为什么Claude/GPT-4的hallucination问题持续存在。

### 7.6 Information-Theoretic视角

可以从信息论角度理解：binary reward $R \in \{0,1\}$ 最多提供1 bit的信息，而model的log-probability提供的是continuous information。GRPO丢掉了后者。CAPO通过logistic surrogate重新引入这部分信息。

### 7.7 与Curriculum Learning的联系

CAPO的gradient自然prioritize "borderline" samples——那些confidence misaligned的样本。这其实是一种implicit curriculum learning。Feng et al. 2025 (https://aclanthology.org/2025.acl-srw.23/) 的self-adaptive curriculum思想类似。

### 7.8 工程实现考量

从Table 2的超参数看，CAPO的实现overhead很小：
- 1.5B模型训练600 steps, 24小时
- 7B模型训练400 steps, 48小时
- 8×A100 GPU
- 关键超参数$\tau$对1.5B用0.6，7B用0.5

Masking的阈值ref-high=2.5, ref-low=1.05是reference model PPL分布的quartile——这是个非常robust的设置，不需要tuning。

### 7.9 对o1/R1-style训练的启示

DeepSeek-R1, OpenAI o1, Gemini 2.0 Flash Thinking都使用GRPO-style训练。CAPO的发现意味着这些model可能都有calibration问题。CAPO提供了一个principled的fix，可能成为下一代reasoning model training的标准组件。

### 7.10 Negative Result的深刻意义

GRPO的成功让我们误以为binary reward就够了。CAPO告诉我们：**reward signal是不充分的**，需要uncertainty-aware的advantage。这可能是RLVR scaling遇到瓶颈的深层原因之一——当model越来越强，binary reward的信息量越来越不够。

---

## 8. 总结

这篇paper的核心贡献：

1. **理论发现**：证明GRPO的gradient等价于inconsistent AUC surrogate优化，这解释了calibration degradation的根本原因
2. **方法创新**：用AUC-consistent的logistic surrogate重新推导advantage，加上reference model masking
3. **实验验证**：在6个benchmark上实现calibration +15~25%同时保持或提升accuracy
4. **应用价值**：Pareto-optimal precision-coverage trade-off，inference-time scaling +5%

这是一篇非常solid的工作——理论清晰，方法principled，实验comprehensive。它揭示的问题可能比paper本身更重要：**所有reward-only的RLVR算法都systematically破坏model calibration**，这是当前reasoning model training的一个fundamental issue。

参考链接：
- Paper: https://arxiv.org/abs/2509.24122 (假设)
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- GSPO: https://arxiv.org/abs/2507.18071
- DAPO: https://arxiv.org/abs/2503.14476
- CoDaPO: https://arxiv.org/abs/2507.08089
- Kadavath et al. (calibration of base models): https://arxiv.org/abs/2207.05271
- AUC consistency theory (Gao & Zhou): https://arxiv.org/abs/1208.0645
- DeepScaler dataset: https://github.com/agentica-project/deepscaler
- Qwen2.5-Math: https://arxiv.org/abs/2409.12122
- Let's Verify Step by Step (PRM): https://arxiv.org/abs/2305.20050
- DPO: https://arxiv.org/abs/2305.18290
- Likelihood displacement in DPO: https://arxiv.org/abs/2306.02231
