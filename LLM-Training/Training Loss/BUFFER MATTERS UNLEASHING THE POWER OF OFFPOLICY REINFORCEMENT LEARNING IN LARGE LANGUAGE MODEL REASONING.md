---
source_pdf: BUFFER MATTERS UNLEASHING THE POWER OF OFFPOLICY REINFORCEMENT LEARNING
  IN LARGE LANGUAGE MODEL REASONING.pdf
paper_sha256: a89b51e671a921316b493897ef27e217c2bc27131d176b3589a453c954841389
processed_at: '2026-08-03T14:36:31-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 BAPO

---

## 一句话总结

GRPO 训练 LLM 的时候，大部分计算都被浪费了——模型做不出来的题，rollout 出来全是错的，advantage 信号归零，梯度等于没动。BAPO 的做法很简单：把过去做不出来的题存起来，等模型变强了再回头试一遍。

---

## 1. 现在的 RLVR 到底哪里浪费了

想象你在教一个小孩做数学题。你给他 100 道题，每道题让他做 8 遍（这就是 group size $G=8$）。

**情况 A**：8 遍全对。你跟他说"不错"，但你也说不出来该改进什么——因为他对了，你不知道他哪一步是蒙的、哪一步是真懂。没有信号。

**情况 B**：8 遍全错。你跟他说"错了"，但他也不知道正确答案是什么。他只能 suppress 自己所有的推理路径，但这并不等于学会新的正确路径。也没有信号。

**情况 C**：8 遍里对了 4 遍。这是黄金时刻。你拿着对的那 4 遍跟错的那 4 遍对比，告诉他"对的做法长这样，错的做法长那样"。这才是真正能学到东西的 feedback。

GRPO 的 advantage 公式：

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{r_\ell\})}{\sqrt{\text{std}^2(\{r_\ell\}) + \varepsilon}}$$

分子是"这条 response 比 group 平均好多少"，分母是"group 内部的 reward 波动幅度"。如果 8 条全对（mean=1, std=0）或全错（mean=0, std=0），分子分母都是 0，整个 advantage 就是 $\frac{0}{\varepsilon}$，约等于 0，梯度等于没传。

**所以 GRPO 实际上只在学"8 遍里对了 1-7 遍"的题**。太简单的题没信号，太难的题也没信号。

paper 的 Figure 1 把这件事画得很直观——GRPO 训练完之后，0/8 accuracy 的红色柱子几乎没降。

更糟糕的是 on-policy 的"use-once-and-discard"原则：每个 prompt 的 8 条 rollout 用完一次就扔了。难题永远只被 sample 一次，永远得到 0/8，永远没有第二次机会。

参考 GRPO 原文：https://arxiv.org/abs/2402.03300

---

## 2. DAPO 的解法：暴力加量

DAPO 的思路是 brute force——既然很多 rollout 是 0/8 没信号，那就多 sample 几遍，总有些 prompt 会恰好落在 1/8 到 7/8 之间。

代价是 rollout 数量翻 2.5 到 4 倍（paper Table 1：GRPO 677k rollouts，DAPO 1921k rollouts）。

这就像你想提高淘金效率，不是改进筛子，而是多挖 4 倍的矿。能用，但贵。

参考 DAPO：https://arxiv.org/abs/2503.14476

---

## 3. BAPO 的核心 idea：三桶分拣

BAPO 的做法是不要扔任何数据，把每个 training step 的 batch 分成三个桶：

### 桶 $\mathcal{X}_1$：当前的新鲜好样本

从当前 rollout 里，只保留 group accuracy 在 $\frac{1}{G}$ 到 $\frac{G-1}{G}$ 之间的 prompt。也就是说，8 条里至少对 1 条、至少错 1 条。这些是有 advantage 信号的样本。

全对和全错的样本直接过滤掉。

### 桶 $\mathcal{X}_2$：历史难题重新尝试

这是 BAPO 最关键的创新。

把过去被判定为"全错"的 prompt 存进一个 buffer $\mathcal{B}_{\text{bad}}$。每隔 $m$ 步（比如 $m=5$），用**当前最新的 policy** $\pi_{\theta_t}$ 重新对这些 prompt 做 rollout。

如果某个 prompt 这次 rollout 出现了部分正确（比如 2/8），说明模型进步了，现在能从这道题上学到东西了。把这个 prompt 加进训练 batch。

**直觉**：这就像错题本。你把做错的题抄下来，过一阵子你水平提高了，回头再做一遍。如果这次能做对一部分，你就从对比中学到了新东西。如果还是全错，那就再放回错题本，等下次再试。

### 桶 $\mathcal{X}_3$：历史好样本填充

光靠 $\mathcal{X}_1$ 和 $\mathcal{X}_2$ 可能不够填满 batch（不是每步都有难题恰好被解锁），所以用历史高质量 rollout 来补位。

为了防止"过时数据"污染当前 policy，只取最近 3 个 step 的 buffer（FIFO 机制），容量限制为 batch size。

**直觉**：这就像你做对了的题，隔两天再复习一遍，巩固记忆。但不能翻太久的旧账本，因为你的解题风格已经变了，一年前的做法可能现在不适用了。

---

## 4. 为什么三个桶缺一不可

paper Table 1(b) 的 ablation 很清楚：

| 配置 | CD-34 | CD-4 | Avg |
|------|-------|------|-----|
| 完整 BAPO | 73.00 | 47.50 | 60.25 |
| 去掉 $\mathcal{X}_2$（不回炉难题） | 60.31 | 35.31 | 47.81 |
| 去掉 $\mathcal{X}_3$（不补位） | 64.43 | 38.75 | 51.59 |

**去掉 $\mathcal{X}_2$ 掉 12.44 个点**，证明"回头试难题"是性能提升的最大来源。

**去掉 $\mathcal{X}_3$ 掉 8.66 个点**，证明"batch 饱和"也很重要。GPU 空转比 stale data 更浪费。

---

## 5. 理论保证：为什么不会训崩

off-policy 最怕的是历史数据跟当前 policy 差太远，importance ratio 爆炸，训练不稳定。

BAPO 的 Theorem 3.2 给了一个 lower bound：

$$\mathbb{E}[I(x)(J(\pi_\theta) - J(\pi_{\theta_t}))] \geq \sum_{i=1}^{3} \mathcal{L}_i(\pi_\theta, \alpha_i)$$

每一项 $\mathcal{L}_i$ 长这样：

$$\mathcal{L}_i = \underbrace{L_{\alpha_i}(\pi_\theta)}_{\text{advantage 推力}} - \underbrace{2K_i \cdot TV(\pi_\theta, \alpha_i)}_{\text{偏离 rollout 的惩罚}} - \underbrace{2 \cdot TV(\pi_{\theta_t}, \alpha_i)}_{\text{偏离当前 policy 的惩罚}}$$

**人话翻译**：policy 每步改进有一个保底值，这个保底值 = advantage 推力 - 两项偏离惩罚。

关键在于 $K_i$ 这个常数：

$$K_i = \frac{1 - \sigma_{\alpha_i,r,\varepsilon}(x)}{\sigma_{\alpha_i,r,\varepsilon}(x)}$$

$\sigma$ 是 reward 的标准差。当 $\sigma \to 0$（homogeneous reward），$K_i \to \infty$，偏离惩罚无穷大，policy 被锁死，一步都动不了。

当 $\sigma = 0.5$（一半对一半错），$K_i \to 1$，惩罚最小，policy 有最大改进空间。

**BAPO 的三个桶设计，本质上是主动控制 $\sigma$ 不让它塌到 0**：

- $\mathcal{X}_1$ 的 filter 保证 group accuracy 在 $[\frac{1}{G}, \frac{G-1}{G}]$，最低 variance 是 $\frac{G-1}{G^2}$
- $\mathcal{X}_2$ 的 re-evaluation 保证解锁的题至少有 $c_1$ 的 accuracy
- $\mathcal{X}_3$ 的 band $[c_2, c_3]$ 保证 replay 的题也有最低 variance

参考 TRPO trust region 思想：https://arxiv.org/abs/1502.05477

---

## 6. 最妙的数学细节：50% accuracy 是黄金点

Proposition A.3 证明了一个很漂亮的结论。

对 binary reward（0 或 1），group 的 reward 服从 Bernoulli 分布，方差是：

$$\sigma^2(\mu) = \mu(1-\mu)$$

求导令其为 0：

$$\frac{d}{d\mu}(\mu - \mu^2) = 1 - 2\mu = 0 \implies \mu = 0.5$$

当 group accuracy = 50%（4/8），variance 最大 = 0.25，$K_i$ 最小，trust region 最宽松，policy 改进空间最大。

**直觉**：模型对这道题"最犹豫"的时候，学到的信息量最大。这跟 active learning 里的 uncertainty sampling、信息论里的 maximum entropy principle 都是同一个思想。

paper 的 Mini-test 实验直接把 $\mathcal{X}_3$ 固定选 $\mu = 0.5$ 的历史样本（不调任何 threshold），结果依然大幅超过 GRPO。这证明"结构"比"超参"重要。

参考 uncertainty sampling：https://dl.acm.org/doi/10.1145/130385.130394

---

## 7. 实际效果：用更少的算力做更多的事

Table 1(a) 数学任务：

| Method | Avg Accuracy | Rollouts | 训练时间 |
|--------|-------------|----------|---------|
| GRPO | 51.58 | 677k | 1d16h |
| DAPO | 54.20 | 1921k | 2d15h |
| BAPO | **56.01** | 733k | 1d22h |

BAPO 比 GRPO 多用 8% 的 rollout（733k vs 677k），换来 4.43 个点的提升。
DAPO 用 2.6 倍的 rollout，只换来 2.62 个点提升，还慢一天。

**性价比差异巨大**。

Figure 7 显示 BAPO 解锁了 31% 的初始 0/8 难题，GRPO 只解锁了 19%。这就是 $\mathcal{X}_2$ 的直接价值。

---

## 8. 一个反直觉的风险：replay 好样本可能有害

paper Section 2.2 提到一个微妙问题：**简单 replay 高质量历史样本会让模型过度依赖已有的成功路径，丧失探索能力，premature convergence**。

想象一个学生，做对的题反复刷，做错的题不管。他会变成一个只会刷简单题的机器，遇到新题型就崩。

BAPO 用两个设计缓解这个问题：
1. **FIFO 限制最近 3 步**——好样本不会离当前 policy 太远
2. **$\mathcal{X}_2$ 强制注入难题**——保证 batch 里始终有"挑战性内容"

这跟 ReMix（https://arxiv.org/abs/2507.06892）的简单混合策略形成对比。

---

## 9. 整体 workflow 用人话走一遍

每个 training step $t$：

**Rollout 阶段**：
1. 每 $v=5$ 步同步一次 rollout policy（让 rollout policy 落后于 trainer policy 5 步）
2. 用 rollout policy 对当前 batch 的 prompt 生成 $G=8$ 条 response
3. 算 reward，算 logprob
4. 全错的 prompt 存进 $\mathcal{B}_{\text{bad}}$
5. accuracy 在 $[c_2, c_3]$ 之间的 prompt 存进 $\mathcal{B}_{\text{high}}$

**Batch 构造阶段**：
6. $\mathcal{X}_1$：从当前 rollout 选 accuracy 在 $[\frac{1}{8}, \frac{7}{8}]$ 的 prompt
7. 每 $m=5$ 步触发一次：用当前 policy 重新对 $\mathcal{B}_{\text{bad}}$ 做 rollout，选有进展的 prompt 作为 $\mathcal{X}_2$
8. $\mathcal{X}_3$：从 $\mathcal{B}_{\text{high}}$ 随机采样补位，填到 batch size 上限

**训练阶段**：
9. 用 $\mathcal{X}_1 \cup \mathcal{X}_2 \cup \mathcal{X}_3$ 算 advantage、算 importance ratio、更新 policy

---

## 10. 最核心的 intuition

GRPO 训练 LLM 就像用漏水的筛子淘金——大部分算力浪费在"没信号的样本"上。传统做法是换更大的筛子（DAPO 多 sample），BAPO 的做法是**把漏掉的矿砂收集起来，等工艺改进了再筛一遍**。

三个桶对应三种数据策略：
- $\mathcal{X}_1$：当前能学的——保证基础梯度
- $\mathcal{X}_2$：过去不会但现在能学的——解锁能力边界
- $\mathcal{X}_3$：过去学得好的——巩固已有能力

这是一个从"无脑扩张 rollout"到"精细管理数据流"的范式转变。不需要更多 GPU，只需要更聪明地用已有的数据。

---

## 参考链接

- BAPO 论文原文（你提供的）
- GRPO: https://arxiv.org/abs/2402.03300
- DAPO: https://arxiv.org/abs/2503.14476
- GSPO (sequence-level ratio): https://arxiv.org/abs/2507.18071
- DeepSeek R1: https://arxiv.org/abs/2501.12948
- AReaL (异步 RL): https://arxiv.org/abs/2505.24298
- RePO: https://arxiv.org/abs/2506.09340
- ReMix: https://arxiv.org/abs/2507.06892
- LUFFY (外部 policy 指导): https://arxiv.org/abs/2504.14945
- Kimi K1.5 (partial trajectory): https://arxiv.org/abs/2501.12599
- MoPPS (难度预测): https://arxiv.org/abs/2507.04632
- Entropy collapse 分析: https://arxiv.org/abs/2505.22617
- Verl/HybridFlow framework: https://arxiv.org/abs/2409.19256
- TRPO trust region: https://arxiv.org/abs/1502.05477
- DQN experience replay: https://www.nature.com/articles/nature14236
- Self-paced curriculum learning: https://arxiv.org/abs/1908.02200
- Uncertainty sampling: https://dl.acm.org/doi/10.1145/130385.130394
- Kantorovich-Rubenstein duality: https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures

---

# BAPO 论文深度技术讲解

Andrej，这篇 paper 切中了一个 RLVR post-training 里非常 fundamental 但被严重忽视的问题：**batch composition 直接决定 gradient signal 的质量**。我先把核心直觉给你，然后逐层拆解。

## 1. 核心问题诊断：GRPO 的"信号枯竭"困境

### 1.1 Advantage 公式的数学脆弱性

GRPO 的 advantage 估计核心公式 (Equation 2)：

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{r_\ell\})}{\sqrt{\text{std}^2(\{r_\ell\}) + \varepsilon}}$$

**变量逐项解释**：
- $r_i$：第 $i$ 条 response 的 binary reward（0 或 1）
- $\{r_\ell\}$：group $\mathcal{G}$ 里所有 $G$ 条 responses 的 reward 集合
- $\text{mean}(\cdot)$：group 内 reward 均值，即 $\mu_{\alpha,r}(x)$ 的 empirical 估计
- $\text{std}^2(\cdot)$：group 内 reward 方差
- $\varepsilon$：smoothing 防除零，通常极小（如 $10^{-8}$）

**关键直觉**：当 batch 里所有 response 全对（$r_\ell=1\ \forall \ell$）或全错（$r_\ell=0\ \forall \ell$），分子分母同时坍缩到 0。$\hat{A}_{i,t}$ 退化成无意义的 $\frac{0}{\varepsilon}$，gradient 信号消失。这就是 paper Figure 1 显示的 GRPO 后训练后 0/8 accuracy 样本数量基本不降的根本原因——**模型从未在它们身上获得有效梯度**。

### 1.2 两个病症的因果链

| 病症 | 数学根源 | 后果 |
|------|---------|------|
| Homogeneous rewards | intra-group reward std $\to 0$ | advantage 消亡，lower bound 退化 |
| Experience waste | on-policy use-once-and-discard | 难题永远没有第二次机会被解锁 |

paper Figure 1 的柱状图揭示一个残酷事实：post-training 后，0/8 accuracy 的"红色"样本几乎不动，而 4/8~7/8 的中等难度样本被显著推向 8/8。这说明 **GRPO 实际上只在学"差点就对了"的样本**，真正的难题被系统性遗弃。

相关参考链接：
- GRPO 原文：https://arxiv.org/abs/2402.03300
- DAPO（dynamic sampling 思路）：https://arxiv.org/abs/2503.14476
- 2025 关于 homogeneous reward 的分析：https://arxiv.org/abs/2505.22617

---

## 2. BAPO 的设计哲学：三分天下

### 2.1 Filter Function 的形式化拆解

Definition 3.1 给出的 indicator function 把 batch 拆成三个互不重叠的子集：

$$I(x) = \underbrace{1_{\{\frac{1}{G} \leq \mu_{\alpha,r}(x) \leq \frac{G-1}{G}\}}}_{\mathcal{X}_1} + \underbrace{1_{\{\mu_{\alpha_B,r}(x) \leq c_1 \wedge \mu_{\pi_{\theta_t},r}(x) > c_1\}}}_{\mathcal{X}_2} + \underbrace{1_{\{c_2 \leq \mu_{\alpha_B,r}(x) \leq c_3\}}}_{\mathcal{X}_3}$$

**变量与逻辑**：
- $\mu_{\alpha,r}(x)$：当前 rollout policy $\alpha$ 在 prompt $x$ 上的 expected reward
- $\mu_{\alpha_B,r}(x)$：buffer policy $\alpha_B$ 历史记录的 reward
- $\mu_{\pi_{\theta_t},r}(x)$：用**当前最新** policy $\pi_{\theta_t}$ 重新采样后的 reward（关键点）
- $c_1, c_2, c_3$：难度阈值

三个子集对应三种"训练数据经济学"：

| 子集 | 数据来源 | 角色 | 核心价值 |
|------|---------|------|---------|
| $\mathcal{X}_1$ | 当前 rollout（fresh） | 保住基础梯度信号 | 过滤 zero-advantage 样本 |
| $\mathcal{X}_2$ | 历史难题 + 当前 policy 重新评估 | "解锁"机制 | 把过去不会的现在能学的找出来 |
| $\mathcal{X}_3$ | 历史高质量样本（FIFO 近三步） | 填充 batch | 保证 batch 饱和、复用 expensive rollouts |

### 2.2 为什么是三个子集而不是两个？

**直觉**：纯 on-policy 的 $\mathcal{X}_1$ 永远受制于当下模型能力边界。模型不会做的题，rollout 永远是 0/8，永远进不了训练集。需要一个 mechanism 让难题**等到模型成长后再回来**——这就是 $\mathcal{X}_2$ 的 re-evaluation。

但单独靠 $\mathcal{X}_1 + \mathcal{X}_2$ 又会出现 batch 不饱和（难题不是每步都有进展），导致 GPU 利用率下降，于是 $\mathcal{X}_3$ 用历史高质量样本填充。这里 paper 用了 FIFO 限制只取最近 3 步的 buffer（$|B_{\text{high}}| \leq B$），这暗合传统 RL 里 experience replay 的 **recency constraint** 思想，避免 stale data 污染 policy。

### 2.3 Re-evaluation 的频率权衡

Algorithm 1 第 13-14 行：

```
if t mod m == 0:
    Re-evaluate B_bad with π_θt to get X_2
```

**$m$ 的物理含义**：每 $m$ 个 training step 用当前 policy 重新 sample 一遍难 buffer。$m$ 太小则频繁 inference 浪费 GPU，$m$ 太大则 policy 已经走得太远，难 buffer 里大部分还是 0/8。

paper 在 hyperparameter robustness 实验里发现 $m \in [3, 7]$ 都稳定（Figure 6 Column 1），但 $m > 15$ 时 trust region 约束失效，这是 Theorem 3.2 里 TV distance 约束 $TV(\pi_{\theta_t}, \alpha_B) \leq \delta_3$ 被破坏的体现。

参考异步 RL 架构 AReaL：https://arxiv.org/abs/2505.24298

---

## 3. 理论部分：Trust Region 视角下的 Lower Bound

### 3.1 Theorem 3.2 的核心结构

policy improvement 的 lower bound 形式：

$$\mathbb{E}_{x \sim \rho_\mathcal{X}}[I(x)(J(\pi_\theta) - J(\pi_{\theta_t}))] \geq \sum_{i=1}^{3} \mathcal{L}_i(\pi_\theta, \alpha_i)$$

其中每一项：

$$\mathcal{L}_i(\pi_\theta, \alpha_i) = \mathbb{E}_{x \in \mathcal{X}_i}\left[L_{\alpha_i}(\pi_\theta) - 2K_i \cdot TV(\pi_\theta, \alpha_i) - 2 \cdot TV(\pi_{\theta_t}, \alpha_i)\right]$$

**变量含义**：
- $J(\pi_\theta(\cdot|x)) = \mathbb{E}_{y \sim \pi_\theta}[r(x,y)]$：policy 在 $x$ 上的 expected reward
- $L_{\alpha_i}(\pi_\theta) = \frac{1}{\sigma_{\alpha_i,r,\varepsilon}(x)}(J(\pi_\theta) - J(\alpha_i))$：标准化后的 advantage-style 项
- $TV(\cdot, \cdot)$：total variation distance，等价于 $\frac{1}{2}||p - q||_1$
- $K_i$：依赖 $\sigma_{\alpha_i,r,\varepsilon}$ 的稳定性常数

**直觉解读**：每一项 lower bound 由三部分组成：
1. $L_{\alpha_i}$：正向的 advantage 推动力
2. $-2K_i \cdot TV(\pi_\theta, \alpha_i)$：新 policy 偏离 rollout policy 的惩罚
3. $-2 \cdot TV(\pi_{\theta_t}, \alpha_i)$：当前 policy 与参考 policy 的偏离惩罚

这正是 **TRPO trust region 思想**的精炼：单步改进有保证，但偏离不能太大。

### 3.2 $K_i$ 的物理意义

$$K_i = \frac{1 - \sigma_{\alpha_i,r,\varepsilon}(x)}{\sigma_{\alpha_i,r,\varepsilon}(x)}$$

**这是论文最妙的数学细节**。$K_i$ 衡量"reward variance 不足时 trust region 被收紧的程度"。

当 reward 方差 $\sigma \to 0$（homogeneous reward），$K_i \to \infty$，trust region 收紧到 0，任何 policy 偏离都会被惩罚致死——**梯度被困死**。

当 reward 方差 $\sigma \to 0.5$（Bernoulli 最大方差），$K_i \to 1$，trust region 最宽松，policy 改进空间最大。

三个子集对应三种 $K_i$：
- $K_1 = \frac{1 - \sqrt{\frac{G-1}{G^2} + \varepsilon}}{\sqrt{\frac{G-1}{G^2} + \varepsilon}}$：来自 $\mathcal{X}_1$ 的 range filter 保证最低 variance
- $K_2 = \frac{1 - \sqrt{c_1(1-c_1) + \varepsilon}}{\sqrt{c_1(1-c_1) + \varepsilon}}$：来自 $\mathcal{X}_2$ 的难 threshold $c_1$
- $K_3 = \frac{1 - \sqrt{\min(c_2(1-c_2), c_3(1-c_3)) + \varepsilon}}{\sqrt{\min(c_2(1-c_2), c_3(1-c_3)) + \varepsilon}}$：来自 $\mathcal{X}_3$ 的 quality band

### 3.3 Proposition A.3：为什么 50% accuracy 是金标准

对 Bernoulli 分布 $\sigma^2(\mu) = \mu(1-\mu)$ 求导：

$$\frac{d}{d\mu}(\mu - \mu^2) = 1 - 2\mu = 0 \implies \mu^* = 0.5$$

此时 $\sigma^2 = 0.25$，$\sigma = 0.5$，$K \to 1$ 取最小值。

**直觉**：$\mu = 0.5$ 意味着模型对 prompt "完全不确定"，每条 rollout 都是一半正确一半错误，group reward variance 最大，advantage 信号最强。这跟 active learning 里的 **uncertainty sampling** 一脉相承，也呼应了 information gain maximization 的思想。

paper 的 Mini-test 实验直接用 $\mu = 0.5$ 作为 $\mathcal{X}_3$ 的固定 filter（不需要调 $c_2, c_3$），依然显著超过 GRPO。这是把 Proposition A.3 的理论直接当 inductive bias 用。

参考 uncertainty sampling 经典文献：https://dl.acm.org/doi/10.1145/130385.130394

---

## 4. 与 PPO/TRPO 数学血缘的对照

让我把 lower bound 推导的关键步骤拆出来——Andrej 你应该会觉得跟 PPO 的推导很熟：

**Step 1**: Importance weighted advantage 展开

$$L_{\alpha_i}(\pi_\theta) = \mathbb{E}_{y \sim \alpha_i}\left[\frac{\pi_\theta(y|x)}{\alpha_i(y|x)} \cdot \frac{r(x,y) - \mu_{\alpha_i,r}(x)}{\sigma_{\alpha_i,r,\varepsilon}(x)}\right]$$

注意 ratio $\frac{\pi_\theta}{\alpha_i}$ 是 **sequence-level** importance sampling weight，而非 PPO 传统的 token-level。这一步跟 GSPO（https://arxiv.org/abs/2507.18071）的设计哲学类似。

**Step 2**: 代数恒等式

$$L_{\alpha_i}(\pi_\theta) - (J(\pi_\theta) - J(\pi_{\theta_t})) = \frac{1 - \sigma}{\sigma}(J(\pi_\theta) - J(\alpha_i)) + (J(\pi_{\theta_t}) - J(\alpha_i))$$

**变量含义**：
- 左边 $L_{\alpha_i}(\pi_\theta)$：off-policy corrected advantage
- $J(\pi_\theta) - J(\pi_{\theta_t})$：真正的 policy improvement（目标）
- 右边把差距拆成两部分，分别跟新旧 policy 与 $\alpha_i$ 的偏差挂钩

**Step 3**: Kantorovich-Rubenstein duality（Lemma A.1）

$$|J(p) - J(q)| \leq 2 \cdot TV(p, q)$$

把 expected reward 差异 bound 到 TV distance 上，再套入 Step 2 的恒等式，就得到 trust region 形式的 lower bound。

**关键 takeaway**：GRPO 的优势是 group-relative 标准化，但标准化引入 $1/\sigma$ 的脆弱性。BAPO 通过 batch 构造主动把 $\sigma$ 从 0 推开，从源头解决问题。DAPO 的 dynamic sampling 是 brute force 加 rollout 数量稀释 zero-advantage 比例，所以代价是 2.5×~4× 的 rollouts。

---

## 5. 实验数据的关键表读

### 5.1 Table 1(a) Mathematics Benchmarks

| Method | AIME24 | AMC | MATH500 | Minerva | Olympiad | Avg | Rollouts |
|--------|--------|-----|---------|---------|----------|-----|----------|
| Base R1 Distill 1.5B | 28.80 | 62.90 | 82.80 | 26.50 | 44.42 | 48.90 | - |
| +GRPO | 30.73 | 67.47 | 85.40 | 28.95 | 45.33 | 51.58 | 677k |
| +DAPO | 35.73 | 70.08 | 86.05 | 30.70 | 48.48 | 54.20 | **1921k** |
| +BAPO | **38.54** | **72.74** | **89.18** | 29.55 | **50.06** | **56.01** | 733k |

**关键观察**：
- BAPO vs GRPO：+4.43 平均准确率提升，rollout 几乎一样（733k vs 677k）
- BAPO vs DAPO：+1.81 平均准确率提升，但 DAPO 用了 2.6× 的 rollouts
- AIME24 这种最难的 benchmark 提升最大（+7.81 vs GRPO）——这正是 $\mathcal{X}_2$ 的解锁效应

### 5.2 Table 1(b) Planning 任务 ablation

| Method | CD-34 | CD-4 | Avg |
|--------|-------|------|-----|
| Base Qwen2.5 Math 1.5B | 1.12 | 0.37 | 0.75 |
| +GRPO | 62.94 | 35.88 | 49.41 |
| +BAPO w/o $\mathcal{X}_2$ | 60.31 | 35.31 | 47.81 |
| +BAPO w/o $\mathcal{X}_3$ | 64.43 | 38.75 | 51.59 |
| +BAPO | **73.00** | **47.50** | **60.25** |

**ablation 直觉**：
- 去掉 $\mathcal{X}_2$（难题 re-evaluation）：性能掉 ~13%，证明"解锁难题"是核心
- 去掉 $\mathcal{X}_3$（高质量 replay）：性能掉 ~9%，证明"batch 饱和"也重要
- 两个组件协同，缺一不可

### 5.3 Figure 7 难题解锁可视化

3 epochs 后，BAPO 把 0/8 accuracy 的样本中 **31%** 推到更高 bin，GRPO 只推了 19%。这个数据很关键，因为它直接量化了 $\mathcal{X}_2$ 的"解锁能力"。

### 5.4 Figure 8 动态 batch 构成

BAPO 实际 backward batch size 经常低于 GRPO 配置的上限（红色线）。这意味着：
- 实际 gradient step 计算量更少
- 多出来的 compute 预算给了 $\mathcal{X}_2$ 的 re-evaluation
- 总训练时间反而跟 GRPO 相当（Table 2: Math 任务 GRPO 1d16h vs BAPO 1d22h）

这是非常 elegant 的"compute reallocation"——你不需要更多 GPU，只需要把 rollouts 用得更聪明。

---

## 6. 关键的 Failure Mode 警示

### 6.1 Figure 10: Uniform Filter 的灾难性崩塌

uniform random filter（保留 60% 不分难度）在 150 步后 grad norm 爆炸、性能归零。原因：

- uniform 把 $\mu = 0$ 的全错样本大量保留
- advantage = $\frac{0 - 0}{\sqrt{0 + \varepsilon}} = 0$，但 ratio $\rho_{i,t}$ 仍要抑制错误 token
- 没有 positive signal 平衡，policy 被推到 suppress 一切 → entropy collapse

这跟 DAPO paper 里讨论的 "negative gradient domination" 是同一个现象。

参考 DAPO entropy collapse 分析：https://arxiv.org/abs/2503.14476

### 6.2 ReMix 类方法的潜在陷阱

paper Section 2.2 提到一个反直觉风险：**简单 replay 高质量历史样本会抑制 exploration**。模型过度 focus 高 advantage 的旧 reasoning path，premature convergence 到 suboptimal solution。BAPO 通过 **FIFO + 近 3 步限制** 缓解这个问题，让 buffer 始终"贴近"当前 policy。

参考 ReMix 论文：https://arxiv.org/abs/2507.06892

---

## 7. 与现有 RL 经典思想的血缘关系

### 7.1 DQN Experience Replay 的演化

DQN 的 replay buffer（https://www.nature.com/articles/nature14236）解决 sample correlation 问题，BAPO 解决的是 RLVR 里的 **reward sparsity** 问题。两者都是把"过去见过但没学好"的数据反复利用，但 BAPO 多了：

1. **难度感知分类**（$\mathcal{X}_2$ vs $\mathcal{X}_3$）
2. **Recency constraint**（FIFO 3 步）
3. **Policy consistency 保存**（存 $\alpha_B(y|x)$ 用于 importance ratio）

### 7.2 Curriculum Learning + RL

$\mathcal{X}_2$ 的"难题等到能学再学"本质上是 **self-paced curriculum learning**（Kumar et al. 2010）。BAPO 的 adaptive threshold（Equation 8）：

$$c_i = r_{\text{tot}} \cdot (c_i^{\text{high}} - c_i^{\text{low}}) + c_i^{\text{low}}, \quad i \in \{2, 3\}$$

**变量含义**：
- $r_{\text{tot}}$：global average reward
- $c_i^{\text{low}}, c_i^{\text{high}}$：阈值上下界

随着 $r_{\text{tot}}$ 上升，threshold 被推高，模型主动选择更难的样本。这是把 SPCL（self-paced curriculum learning）的思想直接 inject 进 RLVR。

参考 SPCL：https://arxiv.org/abs/1908.02200

### 7.3 DeepSeek R1 的"aha moment"

R1 paper（https://arxiv.org/abs/2501.12948）描述 RL 训练中模型自发学会 "wait, let me reconsider" 的 reflection 行为。BAPO 的 $\mathcal{X}_2$ 可能为这种 emergent behavior 提供更肥沃的土壤——难题被反复 re-evaluate，模型有机会在多次尝试中"顿悟"。

---

## 8. 我对这篇 paper 的批评性思考

### 8.1 强项

1. **Theoretical grounding 扎实**：Theorem 3.2 的 TV distance bound 给出清晰的 stability 解释
2. **Mini-test 设计巧妙**：剥离超参调优嫌疑，证明结构本身有效
3. **Generalization 验证**：BA-PPO（附录 A.9）证明 framework 不只对 GRPO 有效

### 8.2 待解决的疑虑

1. **FIFO 3 步的硬编码**：为什么是 3 而不是 5？这跟 trust region 大小耦合，但缺乏理论指导
2. **MoE 架构的适配**：paper conclusion 里承认这点没做。MoE 的 expert routing 在 off-policy 设置下可能极不稳定（GSPO 已揭示类似问题）
3. **Agentic RL 扩展**：multi-turn agentic 场景里 trajectory 长度变化大，buffer 存什么粒度（full vs partial trajectory，类似 Kimi K1.5 https://arxiv.org/abs/2501.12599）是开放问题
4. **Reward hacking 风险**：$\mathcal{X}_3$ 反复 replay 高 reward 样本，若 reward function 有 leakage，模型可能 overfit spurious pattern

---

## 9. 给实践者的快速实施 checklist

如果你想在 verl/HybridFlow framework 里复现 BAPO（https://arxiv.org/abs/2504.13349 是 paper 用的 infra），核心改动：

1. **Rollout 端**：让 rollout policy $\alpha$ 每 $v$ 步同步一次（Algorithm 1 第 3-5 行）
2. **Buffer 维护**：两个独立 FIFO——$\mathcal{B}_{\text{bad}}$（存 $\mu \leq c_1$ 的 prompt）和 $\mathcal{B}_{\text{high}}$（存 $c_2 \leq \mu \leq c_3$）
3. **Logprob 存储必备**：buffer 每条样本必须存 $\alpha_B(y|x)$（per-token 或 per-sequence logprob），否则 importance ratio 算不出来
4. **训练 batch 拼装**：每 step 按 $\mathcal{X}_1 \cup \mathcal{X}_2 \cup \mathcal{X}_3$ 构造，注意 batch size 可能动态变化
5. **Hyperparameter 起点**（Table 3）：
   - $v = 5, m = 5$
   - $c_1 = 1/8, c_2 = 1/8, c_3 = 5/8$（G=8 时）
   - $|B| = 256$
   - max re-evaluate prompts = 128（限制 inference overhead）

---

## 10. 最终直觉总结

BAPO 把 RLVR 的训练数据流当成一个**精炼炉**：

- $\mathcal{X}_1$ 是主燃料——新鲜 rollout 里"恰好够 informative"的样本
- $\mathcal{X}_2$ 是回炉重炼——过去的废料等当前工艺进步后重新冶炼
- $\mathcal{X}_3$ 是催化剂——让炉子始终满负荷运转

GRPO 把所有出炉样本用一次就扔，DAPO 通过多挖矿弥补浪费，BAPO 通过**精炼策略**让每吨矿都榨干价值。这是 RLVR post-training 从"无脑扩张"走向"精细化管理"的标志性工作。

---

## 关键参考链接汇总

- BAPO 论文（你提供的这篇）
- GRPO: https://arxiv.org/abs/2402.03300
- DAPO: https://arxiv.org/abs/2503.14476
- GSPO: https://arxiv.org/abs/2507.18071
- DeepSeek R1: https://arxiv.org/abs/2501.12948
- AReaL (asynchronous RL): https://arxiv.org/abs/2505.24298
- RePO: https://arxiv.org/abs/2506.09340
- ReMix: https://arxiv.org/abs/2507.06892
- LUFFY (off-policy with external policy): https://arxiv.org/abs/2504.14945
- Kimi K1.5: https://arxiv.org/abs/2501.12599
- MoPPS: https://arxiv.org/abs/2507.04632
- Homogeneous reward entropy analysis: https://arxiv.org/abs/2505.22617
- Verl/HybridFlow: https://arxiv.org/abs/2409.19256
- TRPO 原文: https://arxiv.org/abs/1502.05477
- Kantorovich-Rubenstein duality: https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
- DQN experience replay: https://www.nature.com/articles/nature14236
- Self-paced curriculum learning: https://arxiv.org/abs/1908.02200
- DeepScaleR: https://arxiv.org/abs/2504.14936

希望这能 build 起你对 RLVR 数据效率问题的完整 intuition。如果你想 dive deeper 到某个部分（比如 Theorem 证明的某个 step、或者 BA-PPO 在 actor-critic 上的扩展细节），告诉我，我可以继续展开。
