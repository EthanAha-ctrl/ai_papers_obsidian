---
source_pdf: Maximum Likelihood Reinforcement Learning.pdf
paper_sha256: dee96240062fc1d395335e774d7f83035594d598acfc5e0bbecd58f0b6129108
processed_at: '2026-08-05T16:59:48-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 MaxRL

## 一句话版本

**RL 训练 LLM 推理时,大家以为在优化"答对概率",其实只优化了一阶近似,把难题的信号丢了;MaxRL 把它修回来,代码改一行,效果翻几倍。**

---

## 故事从一个直觉开始

想象一个学生刷数学题。有两种给反馈的方式:

- **SFT 方式**:老师直接告诉正确答案,学生照着抄一遍。简单,但需要 ground truth。
- **RL 方式**:学生自己写解法,做对了奖励一下,做错了无视。不需要标准答案,只要能判断对错。

RL 听起来很美,但有个隐藏问题:学生刷 100 道题,90 道已经会了,10 道完全蒙不对。RL 给这 100 道题的"学习信号"权重是一样的——可是那 90 道已经会的题,再答对一次也没啥可学的;那 10 道完全不会的题,答错了 signal 是 0,答对一次(如果运气好)signal 也是 1,完全没被"放大"。

结果:学生变得越来越会在已经会的题上刷分,难题永远学不会。这就是 RLVR 里大家观察到的 **pass@k collapse / mode sharpening** 现象 (https://arxiv.org/abs/2504.13837)。

---

## 那理想的训练目标应该是什么?

如果我们能直接 access "模型对题 $x$ 答对的概率" $p_\theta(x)$,最自然的目标就是 **maximum likelihood**:最大化 $\log p_\theta(x)$,跟 SFT 最大化 ground truth 的 log prob 是一回事。

为什么 ML 比 RL 好?看梯度:

$$\nabla_\theta J_{RL} = \mathbb{E}_x[\nabla_\theta p_\theta(x)]$$

$$\nabla_\theta J_{ML} = \mathbb{E}_x\left[\frac{1}{p_\theta(x)} \nabla_\theta p_\theta(x)\right]$$

ML 多了一个 $1/p_\theta(x)$。代入具体数字感受一下:

| 题目难度 | $p_\theta(x)$ | RL 权重 | ML 权重 |
|---|---|---|---|
| 简单题 | 0.9 | 1 | 1.1 |
| 中等题 | 0.5 | 1 | 2 |
| 难题 | 0.1 | 1 | 10 |
| 超难题 | 0.01 | 1 | 100 |

**ML 自动给难题放大 100 倍的 gradient**——这就是 cross-entropy 在 supervised learning 里为什么 work 的核心原因。RL 失去了这个 reweighting,所以卡在简单题上出不来。

问题是:ML 直接优化不了,因为中间隔着一个 non-differentiable 的采样(模型先采样 chain-of-thought,再 deterministic extract 答案),没法直接 backprop。

---

## Paper 的核心数学发现

作者用一个 Maclaurin 展开,把 ML gradient 写成了 pass@k 的调和级数:

$$\nabla_\theta J_{ML}(x) = \sum_{k=1}^\infty \frac{1}{k} \nabla_\theta \text{pass@}k(x)$$

含义:
- $\text{pass@}k(x)$ = 模型在题 $x$ 上采 $k$ 个 sample 至少一个对的概率
- $\nabla_\theta \text{pass@}k$ = 这个概率对参数的梯度
- $\frac{1}{k}$ = 调和级数权重,大 $k$ 贡献小但加起来无限

而 classical RL 只优化 pass@1:

$$\nabla_\theta J_{RL}(x) = \nabla_\theta \text{pass@}1(x)$$

所以 **RL 就是 ML 的 first-order approximation,只保留 $k=1$ 那一项**。难题信号被砍掉了——因为难题的 pass@1 几乎是 0,但 pass@10、pass@100 可能还有信号,RL 完全用不上。

---

## MaxRL 的解法:截断到 T 项

$$\nabla_\theta J_{\text{MaxRL}}^{(T)}(x) = \sum_{k=1}^T \frac{1}{k} \nabla_\theta \text{pass@}k(x)$$

- $T=1$: 退化为 REINFORCE
- $T \to \infty$: 收敛到 ML
- $T$ 越大越接近 ML,但需要更多 rollout 来估

**Compute 真正买到了更好的 objective**——这是 MaxRL 和 RL 的根本区别。RL 加更多 rollout 只降 estimator variance,目标还是 pass@1;MaxRL 加 rollout 是把 objective 本身向 ML 推近。

---

## 估计器:简单到离谱

Theorem 1 证明 ML gradient 可以写成 **条件期望**:

$$\nabla_\theta J_{ML}(x) = \mathbb{E}\left[\nabla_\theta \log m_\theta(z|x) \mid f(z) = y^*(x)\right]$$

人话:**ML gradient = 只在成功 trajectory 上对 score function 求平均**。这跟 SFT 几乎一样——SFT 也是只在 ground truth trajectory 上算 log-likelihood gradient。

具体做法:
1. 采 $N$ 个 rollout $z_1, \dots, z_N$
2. 算每个的 reward $r_i \in \{0,1\}$ 和 score $S_i = \nabla_\theta \log m_\theta(z_i|x)$
3. 记 $K = \sum r_i$ 为成功数

**REINFORCE**: $\frac{1}{N}\sum r_i S_i$  
**MaxRL**: $\frac{1}{K}\sum r_i S_i$

就差一个归一化常数 $N$ vs $K$。

### 一个具体例子感受差距

采样 16 个 rollout,3 个对,13 个错。$p \approx 3/16 \approx 0.19$。

- REINFORCE: 把 3 个对的 score 加起来 / 16
- MaxRL: 把 3 个对的 score 加起来 / 3

同样 3 个对的 signal,MaxRL 把它放大了 $16/3 \approx 5.3$ 倍。恰好对应 $1/p \approx 5.3$ 的 ML reweighting!

**Theorem 2 证明**:这个简单 estimator 对 $\nabla_\theta J_{\text{MaxRL}}^{(N)}$ 是 unbiased 的。也就是说采 N 个 rollout,自然估出截断到 N 阶的 ML gradient。

---

## 为什么加 rollout 对两个方法意义完全不同

| | RL (REINFORCE) | MaxRL |
|---|---|---|
| N=16 时估的目标 | pass@1 | truncated ML (T=16) |
| N=32 时估的目标 | pass@1 | truncated ML (T=32) |
| 加 rollout 的效果 | 只降 variance | objective 本身更接近 ML |

这是 paper 最深刻的 insight:**compute 在 RL 里是"估得更准",在 MaxRL 里是"目标变得更好"**。

---

## GRPO 的问题

GRPO 用 std 归一化,对应 Bernoulli 的 std $\sqrt{p(1-p)}$,weight 是 $1/\sqrt{p(1-p)}$:

- $p \to 0$: weight $\sim 1/\sqrt{p}$,比 RL 强,但比 ML 的 $1/p$ 弱得多
- $p \to 1$: weight $\sim 1/\sqrt{1-p}$ **blow up** —— 给太简单的题加权,反直觉

所以 GRPO 介于 RL 和 ML 之间,但在简单题上有病态行为。Dr.GRPO (https://arxiv.org/abs/2503.20783) 也在试图修这个问题。

---

## 实验故事

### 1. ImageNet 控制实验(可微 setting,可精确算 ML)

ResNet-50 从头训,random init 的 pass rate ≈ 0.001。

- **REINFORCE 完全卡死**:因为 pass@1 信号太弱,即使每图 16384 rollouts 也学不动
- **Cross-entropy (exact ML)**:稳定收敛
- **MaxRL**:rollout ≥ 1024 时几乎重合 cross-entropy 曲线

这是 Theorem 2 的实证:MaxRL 在 compute 充足时真的复刻 ML。

### 2. Maze 导航(3M 小模型,infinite data)

1M 个程序生成的 17×17 maze。

- **MaxRL 用 4 rollouts > GRPO 用 128 rollouts**
- Pass@1: MaxRL 84.4 vs GRPO 43.6 vs RLOO 25.2
- Pass@256: MaxRL 94.3 vs GRPO 49.6

Compute efficiency 差距巨大。

### 3. GSM8K(SmolLM2-360M,data-scarce,50 epochs)

- **GRPO/RLOO** 10 epoch 达峰后 pass@1 开始降,pass@k 大幅 collapse
- **MaxRL** 30 epoch 才超越,且 pass@k 几乎不退,比 base model 还高
- Pass@1024: MaxRL 83.4 vs GRPO 48.8

MaxRL **更抗 overfitting**,diversity 保持得更好。直觉:ML-like weighting 让模型一直在 hard prompt 上探索,不会在已会题上 sharpen。

### 4. Qwen3-1.7B / 4B 大模型数学推理

POLARIS-53K 数据集,16 rollouts/prompt,1000 steps。评测 AIME 2025 / BeyondAIME / MATH-500 / Minerva。

- **MaxRL 在 4 个 benchmark 上全面 Pareto dominate GRPO**
- Pass@k 相对 base model **提升**(不是不退,是提升)——直接反驳"RL 必然 collapse diversity"的说法
- **Test-time scaling 效率 20×**:用 perfect verifier filter 错误答案时,MaxRL 模型达到 GRPO 同性能只需 1/20 的 inference samples
- Majority voting 也更好:AIME 2025 majority@4096 MaxRL 26.7 vs GRPO 23.3

### 5. 优化动力学差异

Scatter plot gradient norm vs pass rate:
- MaxRL 在 pass rate ≈ 0 的 prompt 上 gradient norm 最大,跟 cross-entropy 一模一样
- GRPO 集中在 pass rate ≈ 0.5 的中等题上
- RLOO 几乎全在简单题上

训练中"至少有一个 rollout 对"的 prompt 比例:MaxRL 始终高于 GRPO,说明 MaxRL 能从更多 prompt 上 extract signal。

---

## 全部 takeaways

1. **RL 是 ML 的一阶近似**——这个 Maclaurin 展开 identity 是 paper 的核心
2. **MaxRL estimator 极简**:把 REINFORCE 的 $1/N$ 改成 $1/K$,一行代码
3. **Compute 买到更好的 objective**,不只是降 variance
4. **Weight function 谱系**:RL ($w=1$) → GRPO ($1/\sqrt{p(1-p)}$) → MaxRL ($[1-(1-p)^T]/p$) → ML ($1/p$)
5. **实证**:从 ImageNet 到 4B 数学推理全面 Pareto dominate,pass@k 不 collapse,test-time 效率 20×

---

## 我的直觉总结

这篇 paper 给我的最大启发是:**很多 RL 训练 LLM 的毛病,根源不在算法实现,而在 objective 本身**。大家觉得 RL 训出来的模型 pass@k collapse、mode sharpening、不探索,搞各种 entropy bonus、exploration bonus、adaptive sampling 去修补。但 paper 指出这些问题来自一个更根本的事实——RL 的 $w(p)=1$ 让 hard prompt 信号被淹掉了。

MaxRL 的修法极其简单,但它重新确立了"ML 是原则,RL 是 workaround"这个 hierarchy。未来 RLVR 算法 design 应该向 ML 靠拢,而不是在 RL 基础上加 hack。

一个自然的问题:能不能把这个 idea 推到 continuous reward、multi-turn、off-policy?Paper 把这些留给 future work。但光是 binary reward + on-policy 这个 setting,已经覆盖了当前 RLVR 数学推理的主流场景。

参考:
- Paper 主页 / arXiv (如上线): https://arxiv.org/abs/2509.13885
- Davis & Recht 同期理论分析: https://arxiv.org/abs/2510.13651
- Xiong et al. Reinforce-Ada (adaptive sampling 路线): https://arxiv.org/abs/2510.04996
- PKPO (直接优化 pass@k): https://arxiv.org/abs/2505.15201
- Yue et al. RLVR diversity collapse 研究: https://arxiv.org/abs/2504.13837
- Wu et al. "Invisible leash" RLVR 分析: https://arxiv.org/abs/2507.14843
- Dr.GRPO 修 GRPO bias: https://arxiv.org/abs/2503.20783

---

# Maximum Likelihood Reinforcement Learning (MaxRL) 深度解析

## 1. 论文核心动机:RL 在学什么?

这篇 paper 要回答一个根本问题:**在 correctness-based RL (RLVR) 训练 LLM 推理模型时,我们到底在优化什么目标?**

考虑数学推理任务:给定 prompt $x$,模型 $m_\theta(z|x)$ 自回归采样 latent trajectory $z$ (chain-of-thought tokens),经 deterministic decoder $f(z)$ 提取出最终答案 $\boxed{\cdot}$。如果 $f(z) = y^*(x)$ 就算成功。这里隐式定义了一个 **pass rate**:

$$p_\theta^{\text{pass}}(x) := \sum_{z \in \mathcal{Z}} m_\theta(z|x) \mathbb{I}\{f(z) = y^*(x)\}$$

从 end-to-end 角度看,模型对每个 input $x$ 实际上诱导了一个"正确概率" $p_\theta^{\text{pass}}(x)$。那么原则上最自然的训练目标应该是 **maximum likelihood (ML)**——最大化正确事件的 log 概率,即类似 cross-entropy。但因为有 non-differentiable 的采样过程挡在中间,直接优化 ML 不可行,大家才退而求其次用 RL。

**论文的关键 insight**:RL (REINFORCE/GRPO) 其实只是 ML 目标的一阶近似,把"hard examples"的信号给丢了。MaxRL 就是来修补这个 gap 的。

参考链接:
- 原文 arXiv (目前版本): https://arxiv.org/abs/2509.13885 (注:作者中部分来自 CMU,论文 PDF 我引用时基于附件)
- Davis & Recht "What is the objective of reasoning with reinforcement learning?" https://arxiv.org/abs/2510.13651 (同期相关工作,推导出 log-like 渐近权重)
- Xiong et al. "Reinforce-Ada" https://arxiv.org/abs/2510.04996 (类似思路但侧重 adaptive sampling)

---

## 2. RL vs ML 的梯度对比(核心公式)

### 2.1 Population-level gradients

记 $p := p_\theta^{\text{pass}}(x)$ 简化符号。两种目标的梯度:

$$\nabla_\theta J_{RL} = \mathbb{E}_x [\nabla_\theta p_\theta(x)]$$

$$\nabla_\theta J_{ML} = \mathbb{E}_x [\nabla_\theta \log p_\theta(x)] = \mathbb{E}_x \left[ \frac{1}{p_\theta(x)} \nabla_\theta p_\theta(x) \right]$$

**变量含义**:
- $\theta$:模型参数(下标,向量)
- $p_\theta(x)$:模型对输入 $x$ 给出正确答案的概率(标量,取值 (0,1])
- $\nabla_\theta$:对参数求梯度
- $\mathbb{E}_x$:对 input 分布 $\rho$ 求期望

**关键 intuition**:ML 多了一个 $1/p_\theta(x)$ 的 **inverse-probability reweighting**。当一个 prompt 对模型来说特别难 ($p \to 0$),ML 会把它的梯度放大 $1/p$ 倍;而当 prompt 很简单 ($p \to 1$),权重趋近 1。这恰好就是 cross-entropy 的行为——专攻 hard examples。

RL 没有这个 reweighting,所有 prompt 权重都是 1,导致 hard prompt 的梯度信号被淹没,模型倾向于在已经会的题上继续 sharpen,产生 paper Section 7 提到的 **distribution sharpening / pass@k collapse** 现象(参见 Yue et al. https://arxiv.org/abs/2504.13837, Wu et al. https://arxiv.org/abs/2507.14843)。

### 2.2 Maclaurin 展开:ML = pass@k 的调和级数

这是 paper 最漂亮的一步。利用 Taylor 公式 $\log p = -\log(1-(1-p)) = -\sum_{k=1}^\infty \frac{(1-p)^k}{k}$ 对 $|1-p|<1$ 成立,得到:

$$\boxed{J_{ML}(x) = \log p = -\sum_{k=1}^\infty \frac{(1-p)^k}{k} = -\sum_{k=1}^\infty \frac{\text{fail@}k(x)}{k}} \tag{4}$$

其中 $\text{fail@}k(x) = 1 - \text{pass@}k(x) = (1-p)^k$ 是 $k$ 次采样全部失败的概率。

两边对 $\theta$ 求梯度,并用 $\nabla_\theta \text{pass@}k = k(1-p)^{k-1} \nabla_\theta p$ (链式法则),得到:

$$\boxed{\nabla_\theta J_{ML}(x) = \sum_{k=1}^\infty \frac{1}{k} \nabla_\theta \text{pass@}k(x)} \tag{5}$$

**这是 paper 的核心 identity**:ML 梯度是 pass@1, pass@2, pass@3, ... 梯度的调和级数加权和 $\sum \frac{1}{k}$。

而 classical RL 只优化 pass@1:

$$\nabla_\theta J_{RL}(x) = \nabla_\theta \text{pass@}1(x)$$

也就是说 **RL 是 ML 的 first-order approximation**,只保留 (5) 式的 $k=1$ 项。当 pass rate $p$ 接近 0 时,$\sum_{k=1}^\infty \frac{1}{k}$ 是发散的,意味着 ML 在 hard prompt 上几乎"无限"重视;而 RL 完全没有这种放大。

---

## 3. MaxRL 目标:截断的 Maclaurin 级数

直接用 (5) 的无穷级数在有限采样下不可估。Paper 提出 **truncated** 目标,在阶数 $T$ 处截断:

$$J_{\text{MaxRL}}^{(T)}(x) := -\sum_{k=1}^T \frac{(1-p)^k}{k} \tag{6}$$

$$\nabla_\theta J_{\text{MaxRL}}^{(T)}(x) = \sum_{k=1}^T \frac{1}{k} \nabla_\theta \text{pass@}k(x) \tag{7}$$

这是一个 **compute-indexed family of objectives**:
- $T=1$: 退化为 REINFORCE/RL
- $T \to \infty$: 收敛到 ML
- 中间的 $T$ 在两者之间 interpolate

**关键性质**:更多的 sampling compute (即更大的 $T$) 直接提升 **objective 本身的 fidelity**——这是 MaxRL 和 RL 的根本区别。RL 加更多 rollout 只是降低 estimator variance,objective 还是 pass@1;MaxRL 加 rollout 是逼近 ML。

---

## 4. Estimator 的魔法:把 ML gradient 写成条件期望

### 4.1 Theorem 1: 条件期望表示

$$\boxed{\nabla_\theta J_{ML}(x) = \mathbb{E}\left[\nabla_\theta \log m_\theta(z|x) \mid f(z) = y^*(x)\right]} \tag{8}$$

**证明 sketch**(Appendix B):
1. REINFORCE identity: $\nabla_\theta p = \mathbb{E}_{z \sim m_\theta(\cdot|x)}[\mathbb{I}\{f(z)=y^*\} \nabla_\theta \log m_\theta(z|x)]$
2. $\nabla_\theta \log p = \frac{\nabla_\theta p}{p} = \frac{\mathbb{E}[\mathbb{I}_A \cdot \text{score}]}{\mathbb{P}(A)}$
3. 由条件期望定义 $\mathbb{E}[X|A] = \frac{\mathbb{E}[X \mathbb{I}_A]}{\mathbb{P}(A)}$,取 $X = \nabla_\theta \log m_\theta(z|x)$, $A = \{z: f(z)=y^*\}$ 即得。

**Intuition**: ML 梯度就是 **只在成功轨迹上对 score function 求平均**。这与 supervised learning 极其相似——SFT 也是只对 ground-truth trajectory 算 log-likelihood 梯度。这给出一个非常自然的估计器:采 N 个 rollout,只对成功的做平均。

### 4.2 Empirical estimator (Theorem 2)

记号:
- 采样 $z_1, \dots, z_N \sim m_\theta(\cdot|x)$
- $r_i := \mathbb{I}\{f(z_i) = y^*(x)\}$ (binary reward)
- $S_i := \nabla_\theta \log m_\theta(z_i|x)$ (score function)
- $K := \sum_{i=1}^N r_i$ (成功采样数,是随机变量)

估计器:

$$\hat{g}_N(x) := \begin{cases} \frac{1}{K}\sum_{i=1}^N r_i S_i, & K \geq 1 \\ 0, & K = 0 \end{cases} \tag{9}$$

注意这和 REINFORCE 的区别只在 **归一化常数**:REINFORCE 用 $\frac{1}{N}$,MaxRL 用 $\frac{1}{K}$。但 unbiased 的目标完全不同:

| 估计器形式 | REINFORCE | MaxRL |
|---|---|---|
| 公式 | $\frac{1}{N}\sum r_i S_i$ | $\frac{1}{K}\sum r_i S_i$ |
| Unbiased for | $\nabla_\theta \text{pass@}1$ | $\sum_{k=1}^N \frac{1}{k} \nabla_\theta \text{pass@}k = \nabla_\theta J_{\text{MaxRL}}^{(N)}$ |

**Theorem 2 证明** (Appendix B):
- 条件在 $K \geq 1$ 上,$K$ 个成功样本 i.i.d. 来自 success-conditioned distribution,所以 $\mathbb{E}[\hat{g}_N | K \geq 1] = \nabla_\theta \log p$ (由 Theorem 1)
- $\mathbb{E}[\hat{g}_N] = \nabla_\theta \log p \cdot \mathbb{P}(K \geq 1) = \frac{\nabla_\theta p}{p} \cdot \text{pass@}N$
- 用 $\text{pass@}N = 1 - (1-p)^N$ 和 $\nabla_\theta \text{pass@}k = k(1-p)^{k-1} \nabla_\theta p$ 化简:
$$\frac{\nabla_\theta p}{p}(1-(1-p)^N) = \nabla_\theta p \sum_{k=1}^N (1-p)^{k-1} = \sum_{k=1}^N \frac{1}{k} \nabla_\theta \text{pass@}k$$

证毕。

### 4.3 Variance reduction via control variate

减去一个零均值的 baseline:

$$\tilde{g}_N(x) = \frac{1}{K}\sum_{i=1}^N r_i S_i - \frac{1}{N}\sum_{i=1}^N S_i = \sum_{i=1}^N \left(\frac{r_i}{K} - \frac{1}{N}\right) S_i \tag{10}$$

第二项 $\frac{1}{N}\sum S_i$ 是 unconditional average score,期望为 0(因为 $\mathbb{E}_{z\sim m_\theta}[\nabla_\theta \log m_\theta(z|x)] = 0$ 是 score function 的标准性质)。这一项与 $K$ 不相关地减少 variance,但不改变期望。

---

## 5. Algorithm 1: On-policy 实现

```
Algorithm 1: On-Policy MaxRL
Input: batch B, rollout 数 N, policy m_θ(·|·)
1: for x ∈ B do
2:   sample z_1,...,z_N ~ m_θ(·|x)
3:   for j=1..N: r_j ← I{f(z_j)=y*(x)}, S_j ← ∇_θ log m_θ(z_j|x)
4:   r̂(x) ← (1/N) Σ r_j
5:   ĝ(x) ← (1/(N·r̂(x))) Σ (r_j - r̂(x)) S_j  if r̂(x)>0 else 0
6: end for
7: return (1/|B|) Σ ĝ(x)
```

**与 GRPO/RLOO 的唯一差别就在 advantage 的归一化**:

| Method | Advantage 归一化 | 等价 population 目标 |
|---|---|---|
| RLOO | leave-one-out mean,不归一化 | pass@1 (期望) |
| GRPO | mean + std,Z-normalization | $1/\sqrt{p(1-p)}$ weight |
| **MaxRL** | **mean (per task)**,即按成功比例归一化 | $[1-(1-p)^N]/p$ weight |

GRPO 用 std 归一化,对应 Bernoulli 的 std $\sqrt{p(1-p)}$。这导致它在 $p \to 0$ 时 upweight 是 $1/\sqrt{p}$,介于 RL ($w=1$) 和 ML ($w=1/p$) 之间。但加再多 rollout 也不会逼近 ML,因为它的 population objective 本身就不是 ML——它本质是固定目标,rollout 增多只降 variance。

GRPO 在 $p \to 1$ 时 $w(p)$ 反而上升(因为 $1-p$ 趋零导致 $1/\sqrt{p(1-p)} \to \infty$),意味着 **GRPO 会给过于简单的 prompt 加权**,这与 likelihood-based 方法的逻辑相悖。

---

## 6. 统一视角:Weight function $w(p)$

把所有方法的 population gradient 写成统一形式:

$$\nabla_\theta J = \mathbb{E}_{x \sim \rho}[w(p_\theta(x)) \nabla_\theta p_\theta(x)] \tag{11}$$

| Method | $w(p)$ | $p \to 0$ 行为 |
|---|---|---|
| RL (REINFORCE) | $1$ | 不放大 |
| GRPO | $\frac{1}{\sqrt{p(1-p)}}$ | $\sim 1/\sqrt{p}$ |
| MaxRL (T) | $\frac{1-(1-p)^T}{p}$ | $\to T$ 当 $p\to 0$,$\to 1/p$ 当 $T\to\infty$ |
| ML | $\frac{1}{p}$ | $1/p$ |

**推导 MaxRL 的 $w_T(p)$** (Appendix C):
$$\nabla_\theta J_{\text{MaxRL}}^{(T)} = \sum_{k=1}^T \frac{1}{k} \cdot k(1-p)^{k-1} \nabla_\theta p = \left(\sum_{k=1}^T (1-p)^{k-1}\right) \nabla_\theta p$$

等比级数求和 $\sum_{k=1}^T (1-p)^{k-1} = \frac{1-(1-p)^T}{p}$。

**Intuition**: $T$ 越大,weight 在 $p\to 0$ 处越接近 $1/p$ (ML)。所以 MaxRL 是真正意义上"用 compute 换取更接近 ML 的 objective"的方法。

Figure 1 的曲线很直观:RL 是 $w=1$ 的水平线,ML 是 $1/p$ 的双曲线,GRPO 在中间但有 $p \to 1$ 处的反转 bump,MaxRL 随 $T$ 增大从 $w=1$ 平滑过渡到 $1/p$。

---

## 7. 实验

### 7.1 ImageNet 控制实验(Sec 6.1)

这是最关键的概念验证——**完全可微的 setting 下,可以精确算 ML (cross-entropy)**,然后看 MaxRL 是否真的收敛到 ML。

- Model: ResNet-50
- Reward: 1 if predicted class = ground truth, 0 otherwise
- Pass rate $p_\theta(x) = \pi_\theta(y^*(x)|x)$ 可以解析得到

**结果**(Figure 2):
- REINFORCE 完全卡住——因为 random init 的 ResNet 在 1000 类上的 pass rate ≈ 0.001,$\nabla_\theta \text{pass@}1$ 信号极弱。即使每图采样 16384 rollouts 也学不动。
- Cross-entropy (exact ML) 稳定收敛。
- **MaxRL 在 rollout 数 ≥ 1024 时几乎重合 cross-entropy 曲线**——这正是 Theorem 2 的实证验证。

**Gradient norm 分析**(Figure 8,Appendix D.5):用 131072 rollouts 估 population gradient。
- Cross-entropy 和 MaxRL 的 scatter plot 几乎一致:pass rate 接近 0 的图有最高 gradient norm,接近 1 的图 gradient norm 接近 0。
- GRPO 的 gradient norm 在 pass rate ≈ 0.5 处最大,hard example 几乎没 gradient。
- REINFORCE 全程 gradient norm 极小,印证了它卡住的原因。

### 7.2 Maze 导航 (Sec 6.2, infinite data)

- 1M 个 procedural 生成的 17×17 maze 用于训练,256 个 holdout 用于测试
- 3M 参数 transformer
- 比较不同 rollout 数 (4~128) 下的 scaling

**结果**(Figure 3, Table 3):
- MaxRL 在所有 pass@k 指标上 Pareto dominate GRPO 和 RLOO
- **MaxRL 用 4 rollouts 的效果 > GRPO 用 128 rollouts** —— efficiency 巨大
- Pass@1: MaxRL 84.4 vs GRPO 43.6 vs RLOO 25.2
- Pass@256: MaxRL 94.3 vs GRPO 49.6

也对比了 PKPO (Walder & Karkhanis, https://arxiv.org/abs/2505.15201) 和 Differential Smoothing (Gai et al., https://arxiv.org/abs/2511.19942) 等 baseline,MaxRL 都显著更好。

### 7.3 GSM8K (Sec 6.3, data-scarce)

- SmolLM2-360M-Instruct,固定 GSM8K 训练集,跑 50 epochs
- 这个 regime 下 overfitting 是主要风险

**结果**(Figure 4, Table 4):
- GRPO/RLOO 在 ~10 epoch 达到 pass@1 峰值后开始 degrade,pass@k 大幅 collapse
- **MaxRL 起步慢,但 ~30 epoch 后超越,且 pass@k 几乎不 degrade**——比 base model 还高
- Pass@1024: MaxRL 83.4 vs GRPO 48.8 vs RLOO 48.5

这指向 MaxRL **更抗 overfitting / 更保 diversity** 的性质。直觉是:ML-like 目标给 hard prompt 更大权重,使模型持续在未见过的解法上探索,而不是在已会的题上 sharpen。

### 7.4 大模型数学推理 (Sec 6.4)

- Qwen3-1.7B-Base 和 Qwen3-4B-Base
- POLARIS-53K 数据集 (An et al., https://arxiv.org/abs/2505.14970)
- 256 prompts/batch, 16 rollouts/prompt, 1000 RL steps
- 评测:AIME 2025, BeyondAIME, MATH-500, Minerva

**结果**(Figure 5):
- MaxRL 在 4 个 benchmark 上全面 Pareto dominate GRPO
- 关键的是 **pass@k 不但没退化,还相对 base 提升**——和 RL 普遍观察到的 pass@k collapse 完全相反
- **Test-time scaling 效率提升高达 20×**:即用 perfect verifier filter 时,MaxRL 训练的模型达到 GRPO 同等性能只需 1/20 的 inference samples
- Majority voting 也优于 GRPO (Appendix J, Table 7):AIME 2025 majority@4096: MaxRL 26.7 vs GRPO 23.3

### 7.5 优化动力学差异 (Sec 6.5)

Figure 6: Qwen2.5-1.5B 在 MATH-500 上的 gradient norm vs pass rate scatter plot:
- MaxRL 在 pass rate ≈ 0 的 prompt 上 gradient norm 最大,与 cross-entropy 行为一致
- GRPO gradient norm 集中在中等难度 prompt
- RLOO 几乎全在简单 prompt 上

Figure 7: 训练中"至少有一个 rollout 成功"的 prompt 比例。MaxRL 始终比 GRPO 高,且差距随训练保持——说明 MaxRL 能从更多 prompt 上 extract 学习信号,这正是 $1/p$ reweighting 的效果。

---

## 8. 与相关工作的关系

### 8.1 RL with verifiable rewards (RLVR)

主流 RLVR 方法 (DeepSeek-R1 https://arxiv.org/abs/2501.12948, DAPO https://arxiv.org/abs/2503.14476, Dr.GRPO https://arxiv.org/abs/2503.20783, GSPO https://arxiv.org/abs/2507.18071) 都在优化 expected reward 或 pass rate,只是 advantage 计算或 off-policy 更新细节不同。MaxRL 提出的是**根本不同的 objective**。

### 8.2 Pass@k optimization

PKPO (Walder & Karkhanis https://arxiv.org/abs/2505.15201) 和 Tang et al. 直接优化 pass@k。但 paper Section 3.1 指出:pass@k 只是 ML 的 Maclaurin 级数的一项,MaxRL 的 $\sum_{k=1}^T \frac{1}{k} \nabla \text{pass@}k$ 是它们的调和级数加权和——所以 pass@k 方法可视为 MaxRL 的特例(只取一项)。

### 8.3 Adaptive sampling

Xiong et al. Reinforce-Ada (https://arxiv.org/abs/2510.04996) 也观察到 RL underweight hard prompt,但用 adaptive rollout budget 来应对。MaxRL 不用 adaptive sampling,而是用统一的 estimator 自然实现 reweighting,且每一步的 finite-sample objective 是 explicit 的 (Theorem 2)。

### 8.4 Davis & Recht (https://arxiv.org/abs/2510.13651)

同期工作,从 asymptotic 角度证明某些 RL 算法诱导 log-like 权重。MaxRL 是 complementary 的:给出 **finite-sample 的 exact estimator-objective equivalence**,并实验验证 scaling 行为。

### 8.5 Diversity collapse 问题

Yue et al. (https://arxiv.org/abs/2504.13837) 和 Wu et al. (https://arxiv.org/abs/2507.14843) 发现 RLVR 降低 pass@k。MaxRL 论文确认了这一现象,并把它归因于 RL objective 本身:$w(p)=1$ 让 hard prompt 的 gradient 被淹没,模型 sharpen 已会的能力。MaxRL 通过 ML-like weighting 解决了这个根本问题。

---

## 9. 关键 Takeaways

1. **RL 是 ML 的一阶近似**:Maclaurin 展开给出 ML = $\sum_{k=1}^\infty \frac{1}{k} \nabla \text{pass@}k$,RL 只用第一项。
2. **MaxRL 的 estimator 极简**:只比 REINFORCE 多把归一化从 $1/N$ 改成 $1/K$,但 unbiased 的目标变成 truncated ML。
3. **Compute 真正买到了更好的 objective**:加 rollout 不只是降 variance,而是增加 Maclaurin 级数的阶数,逼近 ML。
4. **统一 weight function 视角**:RL→GRPO→MaxRL(T)→ML 形成谱系,权重从 $1$ 到 $1/p$,hard prompt 越来越被重视。
5. **实证**:从 ImageNet 控制实验到 4B 数学推理,MaxRL Pareto dominate 现有方法,pass@k 不 collapse,test-time scaling 效率提升 20×。

---

## 10. 个人的几点直觉思考

- **为什么 ML 比 RL 好?** Cross-entropy 在 supervised learning 之所以 work,核心是它对低概率正确类给大 gradient,自动 balance hard/easy。RL 失去这个 reweighting 后变成"在已经会的题上 sharpen",所以 mode collapse。
- **为什么 MaxRL estimator 这么简单却能逼近 ML?** 关键是 Theorem 1 的条件期望表示,把 ML gradient 重写成"成功 trajectory 上的平均 score"。这个视角让 ML 和 SFT 在概念上握手——SFT 也是只在 ground truth 上做 MLE,MaxRL 则只在 success rollout 上做。
- **为什么 GRPO 会给简单 prompt 加权?** Bernoulli variance 在 $p\to 1$ 处也趋零,所以 $1/\sqrt{p(1-p)}$ 在 $p\to 1$ 处 blow up。这是 std-normalization 的副作用,与 ML 的逻辑相反。Dr.GRPO (Liu et al.) 也在试图修正这个问题。
- **潜在局限**:Paper 假设 binary reward,未扩展到 continuous reward / multi-turn / off-policy。把 MaxRL 拓展到 PPO-style off-policy 是一个明显的 future direction。

参考论文 PDF:CMU 和 Tsinghua 等合作,主要作者 Fahim Tajwar, Andrea Zanette 等,附件已给出。arXiv 链接(若已上线)可查 https://arxiv.org/abs/2509.13885 或作者主页 https://www.cs.cmu.edu/~ftajwar/。
