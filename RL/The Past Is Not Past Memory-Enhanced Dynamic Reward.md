---
source_pdf: The Past Is Not Past Memory-Enhanced Dynamic Reward.pdf
paper_sha256: 4faaaaf5cbb50c095ecd1c6857e1ef75b09d80348c40ded3d8ae01c3df46bf6c
processed_at: '2026-08-12T14:45:14-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 MEDS

## 一句话版本

模型在RL训练中会反复犯**同一个错**，传统方法只是让它"更随机一点"，但这没用——因为同一个错可以用一千种不同措辞犯出来。MEDS的做法是：**记住模型犯过的错，下次再犯就扣分**。

---

## 为什么这个问题难？

想象你教一个学生做数学题。他每次都犯同一个conceptual error——比如看到二次方程就忘记判别式。你跟他说"你做题要更多样化"——他会换不同的措辞、不同的符号，但**脑子里的错误回路还是那一条**。

这就是LLM RL训练的现状。entropy regularization相当于跟模型说"你token采样要更随机"——模型确实会写出不同的话，但reasoning trajectory是同一条broken path。

Figure 1说得特别清楚：
- (a) policy从一个"啥都试"的状态慢慢坍缩到一个"只会这一招（而且是错招）"的basin
- (b) 表面上不同rollout看起来文字不同，其实failure mode是同一个
- (c) DAPO的diversity随着训练一路下滑

**核心insight**：diversity不能在token层面看，要在reasoning structure层面看。

---

## 怎么知道两个response是不是"同一个错"？

最naive的办法是看final answer——但两个response可能都答1342却用了完全不同的（错误）思路。也有可能答1342和1349的response其实思路一模一样，只是最后一步算岔了。

下一个naive的办法是string matching或者embedding similarity——但这对"表面措辞"太敏感。

MEDS的trick：**偷窥模型内部的logits**。

具体来说，看Transformer每一层在final answer第一个token位置的logit值。为什么？因为logits反映了模型内部哪些"计算回路"被激活了。如果你和我走的是同一条reasoning path，我们激活的circuits就一样，logits轨迹就长得像；如果思路不同，logits会显著diverge。

Figure 3那个toy example很直观：三个小模型，前两个激活的circuits几乎一样，所以三层logits几乎相同；第三个模型circuit结构不同，logits就跑偏了。

**额外好处**：这些logitsforward pass本来就算了，只是被扔掉。MEDS等于免费捡回来用——zero overhead。

---

## 具体怎么做？

三步：

**第一步**：每个response算一个"指纹向量"——取Transformer后一半层的logits，concat起来。比如28层的模型就取后14层，得到一个14维向量。

**第二步**：每个prompt维护一个"错题本"。每次有response答错了，就把它的指纹存起来。然后用HDBSCAN（一种密度聚类算法）把相似指纹聚成一类。

为什么选HDBSCAN？因为你不知道有多少种错误模式，不能预先指定K。HDBSCAN能自动发现cluster数量，还能把outlier标为noise。

**第三步**：reward shaping。当一个新response被分到某个cluster $C_k$，它的reward会被减掉：

$$\text{penalty} = \min(\alpha \log(|C_k| + 1), \beta)$$

直觉：
- cluster越大，说明这个错犯得越多，penalty越重
- log是为了让penalty增长慢一点（不要爆炸）
- $\beta$是上限，防止penalty把reward信号整个吃掉

举个例子：一个错误cluster有3个response，penalty = $\alpha \log(4)$；有10个response，penalty = $\alpha \log(11)$。但永远不会超过$\beta$。

---

## 为什么这个方法数学上成立？

Theorem 1用Chebyshev不等式证明了：**penalize repeated errors不会让overall performance变差**。

直觉是：penalty相当于把probability mass从"高频错误"区域重新分配到"低频错误"或"正确答案"区域。因为高频错误通常reward更低（因为是错的），而penalty weight $w(\mathbf{y}) = \exp(-\eta \lambda c(\mathbf{y}))$ 对高频错误更小——这两个量关于"重复次数$c$"都单调递减，所以covariance非负，期望reward必然非递减。

公式核心是：

$$J_{\mathbf{x}}(q_2) - J_{\mathbf{x}}(q_1) = \frac{\text{Cov}_{q_1}(r(\mathbf{x}, \mathbf{y}), w(\mathbf{y}))}{\mathbb{E}_{q_1}[w(\mathbf{y})]} \geq 0$$

就是说：penalized policy的expected reward = 原policy的reward + 一个非负的covariance项。这个covariance为正是因为"reward低的东西恰好是被penalize的东西"。

---

## 实验结果有多好？

**Qwen3-8B** 上的亮点：
- Average pass@1: 63.11 (DAPO) → **66.72** (MEDS)
- OlympiadBench pass@128: 70.81 → **82.67**，**相对提升17%**

这个17%的pass@128提升特别重要——说明MEDS不只是让模型答对更多题，而是让模型**探索到更多不同的正确解法**。

训练开销：
- DAPO: 8.00分钟/50步
- MEDS: 8.46分钟/50步
- 只多了6%，几乎免费

**Qwen3-1.7B和Qwen2.5-Math-7B** 也有consistent提升，但smaller model的提升不如8B大——说明base model越强，MEDS的收益越大（因为更强的模型能explore更多paths，MEDS帮助它记住哪些是死胡同）。

---

## 怎么验证"logits真的反映reasoning structure"？

这是paper最严谨的地方——如果不证明这一点，整个方法就站不住脚。

**验证1：Case Study (Figure 6)**

找了一个具体问题，手动标注5个response的reasoning structure：
- A1, A2用quadratic formula方法
- B1, B2, B3用divisor enumeration方法

logit heatmap显示：
- A1和A2的logit轨迹很相似
- B1, B2, B3的logit轨迹很相似
- A和B的logit明显不同

注意：A1, B1, B2都答1342（同一个错答案），但因为reasoning structure不同，被正确分到不同cluster。这证明了"看final answer没用，看logits才有用"。

**验证2：与LLM-as-Judge的一致性**

用Claude-Haiku-4.5对800个错误response做semantic分类（11个error category），然后看logit clustering和LLM分类的一致率：

- 单cluster baseline: 45.16%（没信息量的下界）
- 用全部28层: 53.87%
- 用后14层: **61.2%**

61%的correlation说明：logits是reasoning structure的一个**reasonable proxy**，但不是perfect representation。用后14层比用全部28层好——因为early layers主要编码surface semantics（比如"这个问题在讲数论"），对reasoning structure不敏感。

**验证3：Clustering Quality与Performance的correlation**

这是最causal的证据。paper试了5种不同的feature construction策略，它们的clustering quality排名和downstream performance排名**完全一致**：

| Strategy | LLM Correlation | Average pass@1 |
|----------|----------------|----------------|
| single cluster | 45.16% | 55.14 |
| 28-diff | 52.52% | 55.78 |
| 14-diff | 54.71% | 55.52 |
| 28 | 53.87% | 56.17 |
| **14** | **61.2%** | **56.47** |

single cluster比DAPO还差——证明naive uniform penalty反而有害，**必须真的能区分错误模式才有用**。

---

## 用人话总结整个故事

1. **问题**：RL训练中模型反复犯同一个错，但传统方法（entropy regularization）只看token层面，看不到"reasoning结构层面的重复"。

2. **Observation**：模型内部logits是reasoning structure的fingerprint——相同的reasoning path会产生相似的layer-wise logit轨迹。

3. **Method**：维护一个per-prompt的错题本，用HDBSCAN聚类logit指纹，新response落到大cluster里就扣reward。

4. **理论**：penalize repeated errors在数学上保证不会降低overall performance（Chebyshev不等式）。

5. **实验**：跨3个模型5个benchmark一致提升，最大17% relative gain，开销只多6%。

6. **验证**：logit clustering与LLM semantic judgment有61%一致率，且clustering quality与performance强相关。

---

## 我觉得这个工作最clever的地方

它把mechanistic interpretability的工具（logit analysis、circuit-level reasoning）**从post-hoc analysis变成了training signal**。

以前我们用logits来**理解**模型，现在用logits来**训练**模型。这是一个视角的转换——不是"模型训练完我们去看它怎么想的"，而是"模型训练时我们就盯着它怎么想，发现它在同一个坑里反复跳就打它手"。

类比人类学习：这就像老师不只看学生答对答错，还盯着学生的解题过程，发现学生老是用同一个错误的方法，就专门针对那个方法给反馈。这比泛泛地说"你要多尝试不同方法"有效得多。

---

## 一些联想和Open Questions

- **Penalty的时序性**：当前所有历史error一视同仁，但intuitively，最近的error应该比远古的error更重要。可以加个exponential decay。

- **Cross-prompt generalization**：当前错题本是per-prompt的，但相似题目可能share error patterns。可以想象一个hierarchical memory——local per-prompt + global cross-prompt。

- **与Process Reward Model的结合**：MEDS是outcome-level的reward shaping。如果结合step-level reward（如[PRM800K](https://openreview.net/forum?id=v8L0pN6EOi)），可能在更细粒度上识别repeated errors。

- **Logit stability**：随着policy $\theta$更新，同一reasoning pattern产生的logits会变化。historical features是用旧$\theta$算的，当前rollout是用新$\theta$算的——混合在一起clustering是否stable？paper没讨论这个。

- **Extension到其他modality**：code generation、multimodal reasoning的logits是否也能反映reasoning structure？特别是VLM的visual tokens可能需要不同的feature extraction。

- **与经典RL exploration的联系**：MEDS的"penalize visited states"思想类似count-based exploration和curiosity-driven exploration，但classic RL的state是discrete的，LLM的"state"是high-dimensional continuous reasoning trajectory。MEDS的clustering相当于在reasoning space做discretization。

---

Related links:
- [MEDS GitHub](https://github.com/Linxi000/MEDS)
- [DAPO](https://doi.org/10.48550/arXiv.2503.14476)
- [DeepSeekMath](https://doi.org/10.48550/arXiv.2402.03300)
- [HDBSCAN](https://doi.org/10.21105/joss.00205)
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [Let's Verify Step by Step](https://openreview.net/forum?id=v8L0pN6EOi)

---

# The Past Is Not Past: Memory-Enhanced Dynamic Reward Shaping (MEDS) 深度解析

Karpathy 你好，这篇paper来自Fudan University的Xipeng Qiu组（第一作者Yang Liu），发表于2026年。它解决了一个非常具体但深刻的问题：RLVR训练中的**error collapse**现象。我会从intuition、理论、实现细节、实验数据多个层面来剖析。

---

## 1. Problem Motivation: Error Collapse 不是 Entropy 能解决的

### 1.1 现象描述

在on-policy RL（如GRPO/DAPO）中，policy会逐步坍缩到狭窄的错误basin中。Figure 1用三个panel展示了这个过程：
- (a) policy从一个high-entropy分布演化成narrow、stable的error basin
- (b) 不同errors共享相同failure mode，在训练中被progressively reinforced
- (c) DAPO vs MEDS的output diversity对比

**关键insight**：传统entropy regularization（如[SAC](https://proceedings.mlr.press/v80/haarnoja18b.html), [A3C](http://proceedings.mlr.press/v48/mniha16.html)）只在**current policy distribution**层面注入随机性——它鼓励token-level的stochasticity，但是无法显式detect跨rollouts的**recurrent error patterns**。

举个例子：模型可能在不同的rollout中用slightly different reasoning expressions犯**同一个conceptual error**（比如都把quadratic equation的判别式算错）。从token distribution看，entropy可能很高，但从reasoning structure看，这些rollouts是高度redundant的。

### 1.2 与现有工作的区别

| Method | 层次 | 历史信息 | 计算开销 |
|--------|------|---------|---------|
| KL regularization [InstructGPT](https://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html) | policy shift | 无 | 低 |
| Entropy regularization | token distribution | 无 | 低 |
| Cluster-based advantage [Hu et al.](https://doi.org/10.48550/arXiv.2601.08763) | current batch | 弱 | 中 |
| UCB exploration [Song et al.](https://doi.org/10.48550/arXiv.2509.06941) | outcome frequency | 粗粒度 | 低 |
| External reward model [Eureka](https://openreview.net/forum?id=IEduRUO55F) | adaptive | 有 | 高 |
| **MEDS** | **logit-level patterns** | **per-prompt memory** | **低** |

MEDS的独特之处在于：**reuse layer-wise logits作为reasoning trajectory的compact proxy**，通过clustering追踪跨step的error patterns。

---

## 2. Core Insight: Logits as Reasoning Fingerprint

这是整篇paper最clever的部分。让我们深入理解为什么logits能反映reasoning structure。

### 2.1 为什么选择logits？

考虑Transformer的第$n$层在position $t$产生的logits $l_t^{(n)} \in \mathbb{R}^V$（V是vocabulary size）。对于final answer的第一个token $y^* = y_{L-t}$，每层都有一个对应的logit值 $l^{*(n)} \in \mathbb{R}$（取该token位置的scalar）。

**Mechanistic intuition**（受[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)和[ROME](https://papers.nips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html)启发）：

- 早期layers编码shallow semantic information（[Conditional Memory](https://doi.org/10.48550/arXiv.2601.07372)）
- 后期layers承载reasoning computation circuits的输出
- 相同的reasoning trajectory会激活相同的circuits → 产生相似的layer-wise logit trajectories
- 不同的reasoning会activate不同的circuits → logits显著diverge

Figure 3用3-layer toy model说明了这一点：左、中两个模型activate nearly identical circuits，所以3-layer logit vectors几乎相同；右边模型circuit structure不同，logits diverge。

### 2.2 Feature Extraction公式

$$f(\mathbf{y}) = \text{concat}\left(l^{*(n)} \mid n = \frac{N}{2}, \ldots, N\right) \in \mathbb{R}^{\frac{N}{2}}$$

变量解释：
- $N$: Transformer总层数
- $n$: layer index
- $l^{*(n)}$: 第$n$层在final answer第一个token位置的logit scalar
- $f(\mathbf{y})$: response $\mathbf{y}$的feature representation，维度为$N/2$

**设计选择的依据**：
1. 只用后$N/2$层：早期层主要编码semantic info，对reasoning structure不敏感
2. 直接concat而非diff：作者在Section 4.6对比了多种策略（详见下面Table 2分析）
3. **零额外forward pass开销**：logits已经在forward时计算好了，只是被discard，这里reuse即可

---

## 3. Theoretical Analysis: Why Penalty Helps

Theorem 1的statement很简洁：

> 设有两个reward signals $\mu_1 = r(\mathbf{x}, \mathbf{y})$ 和 $\mu_2 = r(\mathbf{x}, \mathbf{y}) - \lambda c(\mathbf{y})$，对应优化得到的policies为 $q_1, q_2$。则 $J(q_2) \geq J(q_1)$，其中 $J(q) = \mathbb{E}_{\mathbf{x}, \mathbf{y} \sim q}[r(\mathbf{x}, \mathbf{y})]$。

### 3.1 证明的关键步骤

设KL-regularized one-step update的closed-form（[DPO推导](https://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)）：

$$q_\mu(\mathbf{y} \mid \mathbf{x}) = \frac{p_{\text{ref}}(\mathbf{y} \mid \mathbf{x}) \exp(\eta \mu(\mathbf{x}, \mathbf{y}))}{Z_\mu(\mathbf{x})}$$

变量：
- $\eta$: KL regularization coefficient的倒数
- $Z_\mu(\mathbf{x})$: partition function（normalization）
- $p_{\text{ref}}$: reference policy

定义 $w(\mathbf{y}) := \exp(-\eta \lambda c(\mathbf{y})) \in (0, 1]$（penalty weight），可以推出：

$$q_2(\mathbf{y} \mid \mathbf{x}) = \frac{q_1(\mathbf{y} \mid \mathbf{x}) w(\mathbf{y})}{\mathbb{E}_{\mathbf{y} \sim q_1(\cdot \mid \mathbf{x})}[w(\mathbf{y})]}$$

这是importance sampling的形式——$q_2$是$q_1$经过$w$重新加权后的分布。

### 3.2 Chebyshev关联不等式

核心数学步骤：

$$J_{\mathbf{x}}(q_2) - J_{\mathbf{x}}(q_1) = \frac{\text{Cov}_{q_1}(r(\mathbf{x}, \mathbf{y}), w(\mathbf{y}))}{\mathbb{E}_{q_1}[w(\mathbf{y})]}$$

由于：
- $w(\mathbf{y}) = \exp(-\eta \lambda c(\mathbf{y}))$ 关于$c$ non-increasing
- 设计假设：$r(\mathbf{x}, \mathbf{y})$ 关于$c$ non-increasing（即重复犯的错误不会得分更高）

由Chebyshev's rearrangement inequality：$\text{Cov}_{q_1}(r, w) \geq 0$，因此 $J_{\mathbf{x}}(q_2) \geq J_{\mathbf{x}}(q_1)$。

**Intuition**：penalty相当于一个"smart reweighting"——它把probability mass从low-reward + high-repetition的region转移到high-reward + low-repetition的region。因为$r$和$w$都关于$c$单调递减，它们的covariance非负，所以期望reward必然non-decreasing。

### 3.3 重要假设的微妙之处

公式(9)的假设 $c(\mathbf{y}_1) \geq c(\mathbf{y}_2) \implies r(\mathbf{x}, \mathbf{y}_1) \leq r(\mathbf{x}, \mathbf{y}_2)$ 在RLVR中通常成立——因为correct answers的reward高于incorrect answers，而重复errors的$c$值更高。但这个假设在reward shaping后可能被违反，所以MEDS实际实现时限制了penalty的上界（min with $\beta$）。

---

## 4. MEDS Architecture: Three-Stage Pipeline

Figure 2展示了完整pipeline：

### Stage A: Logic Feature Extraction

```
Input x → Sample response ỹ → Extract f(ỹ) from layer-wise logits
                                    ↓
                         Use last N/2 layers
                         Position: first token of final answer
```

### Stage B: Memory-based Clustering

Per-prompt维护error memory：

$$G_x = \{f(\mathbf{y}_1^1), f(\mathbf{y}_2^1), \ldots, f(\mathbf{y}_b^t)\}$$

变量：
- $b$: 每个prompt每次采样的response数（paper中=16）
- $t$: training epoch
- $G_x$: prompt $x$对应的historical features集合

使用[HDBSCAN](https://doi.org/10.21105/joss.00205)聚类：

$$G_x = \{C_1, C_2, \ldots, C_K, C_{\text{noise}}\}$$

**为什么选HDBSCAN而不是K-Means？**
- HDBSCAN不需要预先指定cluster数
- 能处理variable density clusters
- 自动识别noise points（不属于任何cluster的outliers）
- 适合online更新的dynamic memory

Hyperparameters：
- `min_cluster_size = 2`
- `min_samples = 1`
- metric: Euclidean distance

### Stage C: Reward Shaping

$$\widetilde{r}(\mathbf{x}, \widetilde{\mathbf{y}}) = r(\mathbf{x}, \widetilde{\mathbf{y}}) - \min\big(\alpha \log(|C_k| + 1), \beta\big), \quad \text{s.t. } f(\widetilde{\mathbf{y}}) \in C_k$$

变量：
- $|C_k|$: cluster $C_k$ 的大小
- $\alpha$: penalty strength coefficient
- $\beta$: penalty upper bound
- $\log(|C_k| + 1)$: 单调递增变换，保留Theorem 1的ordering要求

**Hyperparameter选择**：
- Qwen3-1.7B, Qwen2.5-Math-7B: $\alpha = 0.1, \beta = 0.2$
- Qwen3-8B: $\alpha = 0.02, \beta = 0.04$（更小，因为base model更强，error pattern更subtle）

**$\beta$ cap的作用**：防止某个cluster的penalty无界增长导致reward signal被dominate。这也对应了Section 3.3提到的理论限制。

---

## 5. Experimental Results: 量化分析

### 5.1 Main Results (Table 1)

让我提取Qwen3-8B的关键数据：

| Method | AIME24 pass@1 | AMC23 pass@1 | MATH500 pass@1 | Minerva pass@1 | OlympiadBench pass@1 | Average pass@1 |
|--------|--------------|--------------|-----------------|----------------|---------------------|----------------|
| Base | 34.51 | 65.72 | 83.62 | 40.90 | 44.69 | 53.89 |
| GRPO | 22.45 | 58.79 | 82.38 | 44.07 | 44.22 | 50.38 |
| DAPO | 45.42 | 81.37 | 89.18 | 46.82 | 52.77 | 63.11 |
| GRPO+Entropy | 27.32 | 65.49 | 85.89 | 45.32 | 49.92 | 54.79 |
| **MEDS** | **45.78** | **82.62** | **92.51** | **51.58** | **61.12** | **66.72** |

**值得注意的observations**：
1. GRPO相对base model出现**performance degradation**（50.38 vs 53.89）——这是经典的RLVR over-optimization issue，[Zhu et al.](https://doi.org/10.48550/arXiv.2506.01347)讨论了negative reinforcement的surprising effectiveness
2. DAPO显著优于GRPO，主要因为decoupled clipping和去除KL penalty
3. **MEDS在Qwen3-8B上的OlympiadBench pass@128从70.81提升到82.67，相对gain 17%**——这是single largest improvement
4. Entropy regularization在这里效果不好（54.79 vs DAPO的63.11）——说明token-level entropy不是correct mechanism

### 5.2 Training Efficiency

| Method | 50 steps | 100 steps |
|--------|----------|-----------|
| DAPO | 8.00 min | 8.95 min |
| MEDS | 8.46 min | 9.73 min |

Overhead仅约6-9%，主要来自HDBSCAN clustering。

### 5.3 Pass@k Curves (Figure 4)

paper画了 $k \in \{1, 2, 4, 8, 16, 32, 64, 128\}$ 的曲线。MEDS在所有$k$值都dominate或match baseline。这表明：
- pass@1的提升不是靠narrow correct answer，而是靠**更diverse的correct exploration**
- pass@128的提升说明**exploration space被expanded**，能找到更多correct solutions

---

## 6. Key Analytical Findings

### 6.1 Diversity Metrics (Figure 5 Left)

paper定义了**Top-1 Eigen Ratio**：

$$\text{Top-1 Eigen Ratio} = \frac{\lambda_1}{\sum_{j=1}^d \lambda_j}$$

变量：
- $\Sigma \in \mathbb{R}^{d \times d}$: 存储logits的covariance matrix
- $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_d \geq 0$: 按降序排列的eigenvalues
- $d$: feature维度（= $N/2$）

**Intuition**：这个ratio衡量logit space的"集中度"。如果所有rollouts都坍缩到同一个error direction，$\lambda_1$会dominate，ratio接近1；如果exploration分散在多个directions，ratio会小。

结果显示MEDS的Top-1 Eigen Ratio持续低于DAPO——证实了exploration diversity的提升。

### 6.2 Within-Step vs Across-Step Diversity

paper用Claude-Haiku-4.5做LLM-as-judge evaluation：

- **Within-Step Diversity**: 同一step内160个rollouts的diversity
- **Across-Step Diversity**: later step相对earlier step是否产生new reasoning patterns

MEDS在两个metric上都显著优于DAPO。具体protocol在Appendix D，使用1-5或1-3的scoring scale。

### 6.3 Logit-LLM Correlation (Figure 5 Right)

这是paper的**validity核心实验**：logit-based clustering是否真的capture了reasoning structure？

| Aggregation Strategy | Correlation with LLM (%) |
|---------------------|--------------------------|
| single cluster | 45.16 |
| 28-diff | 52.52 |
| 14-diff | 54.71 |
| 28 (all layers) | 53.87 |
| **14 (last 14 layers)** | **61.2** |

**关键发现**：
1. 单一cluster（无判别力）correlation最低45.16%——这是baseline
2. 用全部28层不如只用后14层——印证了"early layers encode shallow semantics"
3. Diff策略（相邻层logit差）不如direct aggregation——可能因为diff放大了noise
4. **61.2%的correlation是substantial但不是perfect**——logits是reasoning的proxy而非complete representation

### 6.4 Clustering Quality vs Performance (Table 2)

Qwen2.5-Math-7B上的对照实验：

| Method | Average pass@1 | Average pass@128 |
|--------|----------------|-------------------|
| DAPO | 55.61 | 82.85 |
| MEDS-single cluster | 55.14 | 82.60 |
| MEDS-28-diff | 55.78 | 82.28 |
| MEDS-14-diff | 55.52 | 83.38 |
| MEDS-28 | 56.17 | 82.17 |
| **MEDS-14** | **56.47** | **84.00** |

**关键insight**：
1. MEDS-single cluster性能**低于DAPO**——证明naive uniform penalty反而有害
2. Clustering quality的ranking（14 > 28 > 14-diff > 28-diff > single）与performance ranking**完全一致**
3. 这建立了 **"better clustering → better downstream performance"** 的causal link

---

## 7. Case Study Deep Dive (Figure 6 & Appendix C)

这个case study非常informative。问题是找"smallest good number"——三个distinct divisors和为2022。

### 7.1 Error Pattern Analysis

paper标注了5个responses：
- **Cluster A**: A1, A2
- **Cluster B**: B1, B2, B3

有趣的是：
- A1, B1, B2 都输出 **1342**（错误的相同答案）
- 但它们的**reasoning structures不同**：A系列用quadratic formula approach，B系列用divisor enumeration approach
- 所以被正确分到不同clusters

而B3输出1349（不同答案），但与B1/B2的approach相似（都是divisor enumeration），所以归为同cluster。

### 7.2 Logit Heatmap解读

Figure 7的heatmap显示：
- 前20层：logits变化小，主要编码semantic information
- 后8层（14层中的后8层）：logits显著diverge
  - Cluster A内部logits轨迹相似
  - Cluster B内部logits轨迹相似
  - A vs B的logits pattern明显不同

具体数值（取几个关键层）：

| Layer | A1 | A2 | B1 | B2 | B3 |
|-------|------|------|------|------|------|
| 23 | -17.50 | -17.00 | -19.38 | -19.38 | -19.50 |
| 24 | -17.12 | -16.50 | -18.88 | -18.88 | -19.12 |
| 25 | -18.50 | -18.38 | -20.38 | -20.38 | -20.38 |
| 27 | -24.62 | -24.25 | -25.50 | -25.25 | -24.88 |
| 28 | -28.25 | -28.38 | -27.50 | -27.00 | -27.62 |

可以看到A和B在深层有systematic的差异——A系列的logits更负（约-17到-28），B系列更负但pattern不同（约-19到-27.5）。

---

## 8. Method的Limitations & Future Directions

paper在Section 6诚实承认：logit aggregation方法相对简单。但还有更深层的问题：

### 8.1 Per-prompt Memory Scalability

$G_x$存储了per-prompt所有历史rollouts的features。如果训练corpus有$P$个prompts，每个prompt存储$b \times T$个features（$T$ = total epochs），总memory = $P \cdot b \cdot T \cdot (N/2) \cdot 4$ bytes。

对于paper的设置（512 prompts × 16 rollouts × 100 steps × 14 dims × 4 bytes）≈ 4.6MB，可控。但scale到millions of prompts时需要approximate nearest neighbor search。

### 8.2 Penalty的Temporal Dynamics

当前设计对所有historical errors一视同仁。但intuitively，**recent errors应该比ancient errors更重要**。可以想象一个exponential decay：

$$c(\mathbf{y}, t) = \sum_{\tau=1}^{t} \gamma^{t-\tau} \mathbb{I}[\mathbf{y} \in C_k^{(\tau)}]$$

其中$\gamma \in (0, 1)$是decay factor。这类似于[experience replay](https://doi.org/10.1038/nature14236)中的prioritization。

### 8.3 Cross-prompt Generalization

当前$G_x$是per-prompt独立的。但相似prompts可能share error patterns。可以想象一个hierarchical memory：
- Local: per-prompt memory（current MEDS）
- Global: cross-prompt pattern memory

这类似[RAG](https://doi.org/10.48550/arXiv.2005.11401)的思路，但用于error pattern而非knowledge。

### 8.4 Logit Selection的Optimization

paper只用了final answer first token的logits。更sophisticated的approach可以：
- 使用整段CoT的logit trajectories
- 使用[Sparse Autoencoders](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)提取interpretable features
- 使用[attention pattern](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)作为补充signal

### 8.5 与Process Reward Models的结合

MEDS是outcome-level reward shaping。结合[PRM800K](https://openreview.net/forum?id=v8L0pN6EOi)的step-level reward可能更有power。可以想象：

$$\widetilde{r}(\mathbf{x}, \mathbf{y}) = r_{\text{outcome}}(\mathbf{x}, \mathbf{y}) + \sum_t r_{\text{process}}^{(t)} - \min(\alpha \log(|C_k| + 1), \beta)$$

---

## 9. 与Recent Work的联系

### 9.1 与EEPO的关系

[EEPO](https://doi.org/10.48550/arXiv.2510.05837)（Chen et al. 2025）也关注RL exploration，但用"sample-then-forget"策略。MEDS的反向思路是"sample-then-remember"——记住errors以避免重复。

### 9.2 与Inverse Probability Scaling

[Sinha et al.](https://doi.org/10.48550/arXiv.2601.21669)指出expected return导致outcome-level mode collapse，提出inverse probability scaling。MEDS的penalty可以看作是**error-pattern-level**的inverse frequency weighting，比outcome-level更fine-grained。

### 9.3 与DSDR

[DSDR](https://arxiv.org/abs/2602.19895)用dual-scale diversity regularization。它operates在token-level entropy + response-level diversity。MEDS的logit clustering可以看作是**representation-level diversity**——更深入到model internals。

### 9.4 与Mechanistic Interpretability

paper的方法与mechanistic interpretability community的工具高度相关：
- [Llama Scope](https://doi.org/10.48550/arXiv.2410.20526)的SAE features
- [Low-rank Sparse Decomposition](https://doi.org/10.48550/arXiv.2504.20938)的attention分解
- [Verifying CoT via Computational Graph](https://doi.org/10.48550/arXiv.2510.09312)

MEDS可以看作是把这些分析工具**用于training signal**而非post-hoc analysis。

---

## 10. Implementation Notes (从代码角度)

虽然paper没详细说，但从[GitHub repo](https://github.com/Linxi000/MEDS)可以推断：

```python
# Pseudocode
class MEDSTrainer:
    def __init__(self, model, ...):
        self.error_memory = defaultdict(list)  # prompt -> list of features
        self.clusterer = HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean')
    
    def extract_feature(self, response, model_outputs):
        # Get logits at final answer first token position
        final_answer_token_idx = find_final_answer_start(response)
        # Take last N/2 layers
        layer_logits = [model_outputs.logits[layer][final_answer_token_idx] 
                        for layer in range(N//2, N)]
        # L2 normalize
        feature = torch.cat(layer_logits)
        return F.normalize(feature, dim=-1)
    
    def compute_penalty(self, prompt, response_feature):
        if len(self.error_memory[prompt]) < 2:
            return 0.0
        
        # Cluster all historical features
        all_features = torch.stack(self.error_memory[prompt] + [response_feature])
        labels = self.clusterer.fit_predict(all_features.numpy())
        
        # Find cluster of current response
        current_label = labels[-1]
        if current_label == -1:  # noise
            return 0.0
        
        cluster_size = (labels == current_label).sum()
        penalty = min(self.alpha * np.log(cluster_size + 1), self.beta)
        return penalty
    
    def update_memory(self, prompt, response_feature, is_correct):
        if not is_correct:
            self.error_memory[prompt].append(response_feature)
```

---

## 11. 总结：MEDS的核心贡献

1. **Conceptual**: 首次显式将**historical error patterns**纳入reward design，超越entropy regularization的distribution-level randomness

2. **Theoretical**: 严格的mathematical proof说明memory-based penalty能improve expected return（Theorem 1的Chebyshev inequality argument）

3. **Technical**: Reuse layer-wise logits作为zero-overhead reasoning proxy，配合HDBSCAN实现dynamic clustering

4. **Empirical**: 跨3个models、5个benchmarks的consistent improvement，最大17% relative gain

5. **Analytical**: 建立了clustering quality与downstream performance的causal link，并用LLM-as-judge验证了logit clustering的semantic meaningfulness

---

## 12. 我的Critique与Open Questions

作为friendly critique，我会问几个问题：

### Q1: Logits的Stability across Training

随着policy $\theta$更新，同一reasoning pattern产生的logits会变化。$G_x$中存储的historical features是用**旧$\theta$**计算的，但当前rollout的feature是用**新$\theta$**计算的。HDBSCAN在混合不同$\theta$的features上clustering是否stable？

可能的解决：存储$(\theta_t, f_t)$pair，并用$\theta$距离加权clustering。

### Q2: Penalty与Advantage Normalization的Interaction

GRPO/DAPO用group-relative advantage $A_i = (R_i - \mu) / \sigma$。MEDS的penalty改变了$R_i$，但是否影响$\mu, \sigma$的计算？如果penalty已经反映在advantage中，是否double-counting？

### Q3: Cold Start Problem

第一个epoch"主要用于memory formation"，此时没有penalty。如果模型在前几个step就collapse到某个error basin，memory还没建立起来。可以考虑用**base model的pre-training statistics**做warm start。

### Q4: Why HDBSCAN优于DBSCAN?

HDBSCAN的hierarchical structure允许variable density clusters，但在logit space中density是否真的variable？简单的DBSCAN可能就够了。

### Q5: Multi-modal Extensions

paper只验证了math reasoning。对于code generation（[StepCoder](https://doi.org/10.48550/arXiv.2402.01391), [CodeRL](https://openreview.net/forum?id=0XBuaxqEcG)）、multimodal reasoning（[Thinking with Videos](https://doi.org/10.48550/arXiv.2508.04416)），logit representation是否依然有效？特别是vision-language models的visual tokens的logits可能语义不同。

### Q6: Connection to Exploration in Classic RL

MEDS的"penalize visited states"思想类似classic RL中的[count-based exploration](https://doi.org/10.1038/nature14236)和[UMA](https://doi.org/10.48550/arXiv.1706.05225)。但classic RL的state space是discrete且small，LLM的"state"是high-dimensional continuous reasoning trajectory。MEDS的clustering相当于在reasoning space做discretization——这是否能借鉴[Curiosity-driven Exploration](https://doi.org/10.48550/arXiv.1705.05363)的prediction-error formulation？

---

## 13. Reference Links

- [MEDS GitHub](https://github.com/Linxi000/MEDS)
- [DAPO paper (arxiv 2503.14476)](https://doi.org/10.48550/arXiv.2503.14476)
- [DeepSeekMath (arxiv 2402.03300)](https://doi.org/10.48550/arXiv.2402.03300)
- [HDBSCAN (JOSS)](https://doi.org/10.21105/joss.00205)
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [DPO paper](https://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)
- [InstructGPT](https://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html)
- [Let's Verify Step by Step (PRM)](https://openreview.net/forum?id=v8L0pN6EOi)
- [Llama Scope SAE](https://doi.org/10.48550/arXiv.2410.20526)
- [Eureka](https://openreview.net/forum?id=IEduRUO55F)
- [MATH dataset](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html)
- [OlympiadBench](https://doi.org/10.18653/v1/2024.acl-long.211)
- [verl framework](https://doi.org/10.1145/3689031.3696075)
- [Negative Reinforcement in LLM Reasoning](https://doi.org/10.48550/arXiv.2506.01347)
- [Rewarding the Rare](https://doi.org/10.48550/arXiv.2601.08763)
- [Outcome-based Exploration](https://doi.org/10.48550/arXiv.2509.06941)

---

希望这个分析能帮你build intuition about why MEDS works。核心insight是：**reasoning structure lives in logits, and history matters for exploration**。这是一个非常clean的idea——把mechanistic interpretability的工具用作training signal，而不是post-hoc analysis tool。我觉得这个方向很有潜力扩展到更broad的RL settings。

如果你对某个具体部分（比如Theorem 1的证明细节、HDBSCAN的algorithm、或者case study的logit pattern）想深入讨论，我可以继续展开。
