---
source_pdf: Olmix A Framework for Data Mixing.pdf
paper_sha256: 6f2f997eb61ec7ccc82b64b3cb780d218349a0aa3514472f000ed1a956630a29
processed_at: '2026-08-05T23:04:16-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Olmix 用人话讲

兄弟，我把刚才那篇 Olmix 用大白话再过一遍，跳过数学细节，只讲 intuition 和为什么这事重要。

参考链接：
- 论文 PDF (arXiv 镜像推断): https://arxiv.org/abs/2504.13161
- Olmo 3 报告: https://arxiv.org/abs/2512.13961
- RegMix: https://arxiv.org/abs/2407.01492
- Data Mixing Laws: https://arxiv.org/abs/2403.16952
- DoReMi: https://arxiv.org/abs/2305.10429
- BiMix: https://arxiv.org/abs/2405.14908
- AutoScale: https://arxiv.org/abs/2407.20177
- Chameleon: https://arxiv.org/abs/2505.24844
- Muennighof data-constrained scaling: https://arxiv.org/abs/2305.16264

---

## 这篇 paper 在干什么

训练大语言模型这件事里，最被低估的 first-order 决定就是：**你的数据里有多少比例来自哪个 domain**。比如 web text、code、math paper、Wikipedia、PDF 文档，这些 domain 各占百分之几。

这件事重要到什么程度？同一个 model 架构、同一个 training budget，mix 调好和调坏，downstream 性能能差 10% 以上。Llama 3、Qwen 2.5、Olmo 3 的技术报告都把 data mixing 当成核心章节来写。

但工业界实际上怎么调 mix？大部分人靠直觉手动设权重，或者跑一堆 grid search 烧 GPU。学术界有 RegMix、DoReMi、Data Mixing Laws 这些方法想自动化这件事，但每个 paper 都说自己的方法最好，互相矛盾，也没人在真实大模型开发里完整跑过。

Olmix 就是 AI2 团队在训 Olmo 3 的时候，把这套流程系统化做了一遍。他们干了两件事：

**第一件**：把现有方法的所有 design choice 拿出来，一个一个做 ablation，告诉你到底应该怎么配。给出了一个开箱即用的 recipe 叫 OlmixBase。

**第二件**：处理一个之前所有 paper 都假装不存在的问题——**你的 domain set 在开发过程中一直在变**。今天加个 Stack-Edu，明天删个 AlgebraicStack，后天把 olmOCR 的 PDF 重新 format 一遍。每次变了，mix 都得重算。他们提出 mixture reuse，让你复用之前算过的信息，省 70%+ 的 compute 还几乎不损失性能。

---

## 为什么 mixing 这么难

直觉上的 naive 做法：按每个 domain 的 token 数量比例分配。DCLM 有 4T token，Wikipedia 有 10B token，那 DCLM 占 99.7%，Wikipedia 占 0.3%。

这个做法有个问题：**token 数量不等于 token 价值**。Wikipedia 那 0.3% 可能对 QA task 的贡献比 DCLM 那 99.7% 还大。Stack-Edu 的 Python 子集可能只有 18B token，但对你家 model 的 code 能力是 game-changer。

所以你得给高价值 domain 更高的权重。但高到多少？给 Stack-Edu:Python 5% 还是 15%？给 FineMath-3+ 是 2% 还是 8%？这跟每个 domain 的 utility、数据量、跟其他 domain 的重叠都有关系。

学术界搞出一套叫 "offline mixing schema" 的流程：

1. **Swarm**：训一堆 small proxy model，每个用不同的 mix 比例
2. **Regression**：学一个函数，输入 mix 比例，输出预测的 downstream 性能
3. **Optimization**：在这个函数上找最优 mix

这流程谁都会说，但每一步都有十几个 design choice，没人系统比较过。Olmix 第一部分就是干这个。

---

## P1: 七个 design choice 都该怎么选

### Proxy model 多大才够

你不想训 1B 的 proxy 来试 mix（那跟直接训 target 没区别），但 1M 的 proxy 又太弱，学不到"哪个 mix 对哪个 task 好"这个 ordering。

他们训了一堆 proxy-target pair，用相同 mix，看两者的 performance ranking 相关性。结论：**15M 参数起，ranking 就稳定了**。他们选 30M，跟 1B target 的 Spearman 相关 0.89。

RegMix 原来用 1M，这论文直接说不够。可能 RegMix 的 setting 简单所以凑合能用，但 24 个 domain 的复杂 setting 下 1M 不行。

### Swarm 要跑多少个

这是最 actionable 的发现。之前有人用 20 个，有人用 500 个，没人说清楚跟 domain 数 $m$ 的关系。

他们做实验发现：**用 log-linear regression 的话，$K = 3(m+1)$ 就够了**。24 个 domain 大概 75 个 proxy run，64 个 domain 大概 195 个。

之前 DML 和 BiMix 的 paper 假设 sample complexity 是 $\mathcal{O}(m^2)$，这篇证明是 $\mathcal{O}(m)$。差一个平方，实际预算差几十倍。

为什么是 $m+1$：log-linear model $\hat{f}(p) = c + \exp(A^\top p)$ 有 1 个常数 + $m$ 个系数 = $m+1$ 个参数，至少要 $m+1$ 个样本才能 uniquely solve。乘以 3 是为了让 fit 稳定。

### Swarm 怎么采样

两个发现：

1. **Sparse vs Dense**：如果你混的是 topic 级别（比如 DCLM 分成 24 个 topic，其中有些是 adult content 这种低价值 topic），用 sparse swarm（允许某些 domain 权重为 0，让模型学到"这些 domain 该排除"）。如果你混的是 source 级别（每个 source 都是人工 curated 的好数据），用 dense swarm（每个 run 都覆盖所有 source）。

2. **Dirichlet prior 用 natural distribution**：按 token 比例做 prior 就行。用强先验（已经知道好 mix 在哪）略好，但 natural distribution 是合理 fallback。用烂先验（故意往差 mix 方向采）会显著变差。

### Regression model 用什么

这是论文里最有趣的部分。他们比了 6 个 model：Search、LightGBM、Gaussian Process、BiMix、AutoScale、Log-linear。

**关键发现**：不同 model 在不同 swarm size 下表现不一样。BiMix 在小 swarm（25 个）最好，LightGBM 要 118+ 个才 fit 好，log-linear 在 75+ 个（即 $3(m+1)$）后持续最好。

**这就解释了为什么文献矛盾**——每个 paper 都说自己的 model 最好，因为他们在不同的 swarm size 下做实验。BiMix 参数少，小 swarm 够 fit；LightGBM 高方差，要数据喂满；log-linear 恰好在中间。

推荐：**log-linear model**。它参数恰好够、convex 可以用 exact solver、可解释、sample complexity 线性。

为什么是 $\exp(A^\top p)$ 而不是线性 $A^\top p$？因为 LM loss 在 domain 上的响应是 power law 形式——domain 权重翻倍不会线性降 loss，是指数级。这是 Data Mixing Laws 的核心 insight。

### Regression 颗粒度

给每个 task 单独 fit 一个 model，比把所有 task 平均后 fit 一个 model 好。因为 math task 跟 math domain 强相关、code task 跟 code domain 强相关，混在一起 fit 会 blur 掉这些信号。

### 数据约束怎么处理

这是文献第一次认真处理的问题。现有方法都假设你有无限数据。实际上，如果 mix 说"给 code 40%"，但你 code 只占 5% 数据，那 code 要被重复 8 次。Muennighof 的 scaling law 说重复超过 4 epoch 性能开始掉。

他们加约束 $p_j \le k N_j / R$，其中 $k=4$ 是最大重复次数。关键问题是：约束加在哪？

三个选择：
- Swarm 里所有 mix 满足约束 + optimization 不加约束 → **失败**，optimization 会走到 swarm 没采样过的违约区域
- Swarm 任意采 + optimization 加约束 → **最优**
- 两边都加约束 → 满足约束但性能差，因为 swarm 覆盖窄了

推荐：**optimization 里加约束，swarm 不加**。

这个约束不只是"防止坏结果"——它主动 shape 了 mix。Figure 7 显示，紧约束下 mix 会被推向 natural distribution；松约束下 mix 会给 high-utility domain 极高权重。你选 $k$ 是在做真实 trade-off。

### Optimization solver

log-linear model 是 convex 的，可以用 CVXPY 直接解。但直接解不是最优——因为你的 regression model 是 imperfect surrogate，过度优化会 overfit 到 surrogate 的偏差。

加一个 KL regularization $\lambda D_{KL}(p \| p_0)$ 往 natural distribution 拉，$\lambda = 0.05$ 最优。这跟 RL 里的 trust region 思想一样——你不完全信你的 model，加个 prior 防止走太远。

---

## P2: Domain set 一直在变怎么办

这才是这篇 paper 最有实际价值的部分。

在真实开发里，你的 domain set 一直在动。Olmo 1→2→3 的轨迹：今天发布新数据集（Add），明天发现某数据集质量差删掉（Remove），后天把一个粗 domain 拆成细 subdomain（Partition），大后天把 PDF 重新 format（Revise）。

每次 domain set 变了，理论上整个 mix 都得重算。如果你有 64 个 domain，每次 full recomputation 要 195 个 proxy run。5 次 update 就是 1000 个 run。这很贵。

Olmix 的 insight：**update 通常只影响一小部分 domain，剩下的大部分 domain 之间的相对比例不该变太多**。

所以做法是：
1. 把不受 update 影响的 domain 冻结它们的**相对**比例
2. 把它们 collapse 成一个 virtual domain
3. 只重算这个 virtual domain 的总权重 + 受影响 domain 的权重

举个例子。原来 3 个 domain，mix 是 $[0.25, 0.25, 0.5]$。加 1 个新 domain。不 reuse 要学 4 维 mix。Reuse 把前 3 个 collapse 成一个 virtual domain（内部相对比例 $[0.33, 0.33, 0.67]$），只学 2 维 mix $[0.4, 0.6]$（virtual domain 占 0.4，新 domain 占 0.6），expand 回去就是 $[0.1, 0.1, 0.2, 0.6]$。

**收益**：如果你有 64 个 domain 加 1 个新的，proxy run 从 195 降到 6。

### 什么时候 reuse 会翻车

论文给了理论：performance gap 由两个东西控制。

**第一个：reuse gap**。你 reuse 的比例和 update 后真实最优比例的差距。如果 update 影响小（加的 domain utility 低，或者数据少被 constraint cap 住），这个 gap 小。

**第二个：coupling term**。unaffected domain 和 affected domain 在 task 上的"重叠"。如果加的是 code domain，你 unaffected 里有 software_development 这个 topic 也影响 code task，那它们高度耦合——加 code 会改变 software_development 的最优权重，reuse 会出问题。

### PartialMixtureReuse

当 coupling 高的时候，FullMixtureReuse 会差。这时候 PartialMixtureReuse 把那些高耦合的 unaffected domain 也一起 recompute，不只 reuse。

直觉：加 Stack-Edu (code) 时，DCLM:software_development 也影响 code task，应该一起 recompute。Figure 17 直接验证：recompute software_development 时 coupling term 下降最多。

论文承认这里需要 domain expertise 手动选哪些 recompute，没给自动化方法。这是 future work。

---

## 实验结果有多好

模拟 Olmo 3 的 5 次 update（从 24 domain 到 64 domain）：

- **FullMixtureReuse**：比 natural distribution 好 11.6%，用 216 个 proxy run（full recomputation 要 832 个），省 74% compute，capture 95% 的 full recomp gain
- **PartialMixtureReuse**：好 12.0%，用 272 个 run，省 67%，capture 98%
- **3.05x data efficiency**：用最好的 mix，1B model 在 20K step 达到 natural distribution 在 61K step 的最终性能。同一架构、同 token 数、同 optimizer，只换了"哪些 token 先看"，快 3 倍

这个 3.05x 不是玄学——mix 自动把高质量 token 放前面，model 先学核心能力，再慢慢填 general knowledge。等于 mixture optimization 自动发现了一个 curriculum。

---

## 对 Karpathy 来说最值得关注的点

1. **$K = 3(m+1)$ 这个公式**：如果你做 mixing，这就是你的预算公式。64 个 domain 大概 195 个 proxy run，不要多不要少。

2. **Mixture reuse 的 collapse/expand trick**：这是一个 elegant 的 reparametrization。你保留所有 OlmixBase 的 machinery，只是维度从 $m'$ 降到 $1 + |\mathcal{D}_{\text{comp}}|$。代码改动很小，收益巨大。

3. **理论 + 实验闭环**：coupling term $\kappa$ 和 reuse gap 都是可计算的量，实验验证它们真的预测 performance gap。这种 "theory informed design" 在 systems paper 里少见。

4. **Log-linear model 的胜利**：不是因为它最 fancy，是因为它参数恰好、convex、可解释、sample complexity 线性。这跟简单胜过复杂的 general lesson 一致。

5. **Repetition constraint as first-class citizen**：把它放进 optimization 而不只是 swarm 里，这是一个小但重要的工程 insight。

6. **3.05x data efficiency**：暗示 mixing 决策可能是 curriculum learning 的 principled 自动化形式。不是手设计 curriculum，是让优化自动发现。

---

## 我觉得还有哪些没解决

1. **Proxy-target gap**：只验证到 1B target。32B 上 30M proxy 的 ranking 可能掉。Heineman 2025 说 BPB 跨 scale 稳，但 mixing 这种敏感决策可能不行。

2. **Task weights**：macro-average 52 个 task，math/code/QA 等权。想 emphasize 某 capability 要改 objective，论文没讨论。

3. **Domain granularity**：WebOrganizer 给的 topic 是一个特定 partition。换更细的 sub-topic 会改变最优 mix。这是 mixture 的 "resolution" 问题。

4. **Multi-stage training**：Olmo 3 实际有 continue pretraining + midtraining。不同 stage 的 mix 应该不同。论文只看第一阶段。

5. **Online mixing**：论文的 mix 是 static 的，整个训练用同一比例。训练中动态调比例可能更好，论文 Section 6 也提到这是 future work。

6. **PartialMixtureReuse 自动化**：选哪些 unaffected domain 一起 recompute 需要手动。理论上可以基于 coupling term $\kappa$ 自动选，论文没做。

7. **Non-log-linear model 的理论**：理论分析只在 log-linear 假设下。LightGBM、GP 这种 non-parametric model 难 extend。

---

## 一句话总结

Olmix 把 "手动调 mix ratio" 这件工程实践，变成有理论指导的状态化优化流程。它告诉你该用多大 proxy、跑几个 run、用什么 regression model、怎么处理数据约束；它告诉你 domain 变了怎么廉价复用旧 mix；它在真实 Olmo 3 开发里跑通，3 倍 data efficiency，省 70% compute。

这是 systems paper 里少见的 "theory + engineering + real deployment" 三合一。

---

# Olmix: A Framework for Data Mixing Throughout LM Development 详细解读

这篇 paper 是 AI2 (Allen Institute for AI) 和 Stanford 的合作工作，由 Mayee Chen 在 AI2 实习期间主导，为 Olmo 3 (7B 和 32B) 的实际预训练开发的一套 data mixing 框架。它解决的不是某个抽象的理论问题，而是在真实 LM 开发循环中遇到的两个 first-order 难题。下面我尽可能详细地拆解。

参考链接：
- 论文 (arXiv 镜像版本，可推断): https://arxiv.org/abs/2504.13161 的引用格式可以看出是 2025 年的工作
- Olmo 3 报告: https://arxiv.org/abs/2512.13961 (论文中引用 Olmo et al., 2025)
- DCLM-Baseline: https://arxiv.org/abs/2406.11794 (Li et al., 2024)
- RegMix: https://arxiv.org/abs/2407.01492 (Liu et al., 2025a)
- Data Mixing Laws: https://arxiv.org/abs/2403.16952 (Ye et al., 2025)
- DoReMi: https://arxiv.org/abs/2305.10429 (Xie et al., 2023)
- DoGE: https://arxiv.org/abs/2310.15393 (Fan et al., 2024)
- BiMix: https://arxiv.org/abs/2405.14908 (Ge et al., 2025b)
- AutoScale: https://arxiv.org/abs/2407.20177 (Kang et al., 2025)
- CLIMB: https://arxiv.org/abs/2504.13161 (Diao et al., 2025)
- ADMIRE-BayesOpt: https://arxiv.org/abs/2508.11551 (Chen et al., 2025b)
- WebOrganizer: https://arxiv.org/abs/2502.10341 (Wettig et al., 2025)
- UniMax: https://arxiv.org/abs/2304.09151
- Muennighof scaling data-constrained: https://arxiv.org/abs/2305.16264
- Aioli (同作者 Mayee Chen 前序工作): https://arxiv.org/abs/2411.05735
- Chameleon: https://arxiv.org/abs/2505.24844

---

## 1. 为什么这篇 paper 重要 (Big Picture Intuition)

现在训 LM 不是一次性的工作，而是一个迭代过程。在 Olmo 1→2→3 的开发轨迹里，data 团队不停在做四件事：add 新数据集、remove 不好的数据集、partition 把一个粗 domain 拆成细 subdomain、revise 修改已有数据集的内容 (reformat、rewrite)。每次 domain set 一变，理论上整个 mixture ratio 都得重算。这件事历史上是 ad-hoc 的——大家凭直觉手动调权重，或者跑一堆 proxy model 做 grid search。

Olmix 把这个流程系统化，回答两个问题：
- **P1 (configuration problem)**: 当你坐下开始训第一个模型，你的 domain set 是固定的，怎么把 offline mixing schema 配置好？现有 paper (RegMix, DML, BiMix, AutoScale, ADMIRE, CLIMB) 给出的 design choice 互相矛盾、缺 justification，并且都假设无限数据。Olmix 给出 7 个 RQ 的系统实验。
- **P2 (evolving domain problem)**: 一旦 domain set 在开发中变化，全量重算代价巨大。Olmix 提出 mixture reuse：复用之前 mix 中不受 update 影响那部分的相对比例，只重算受影响部分。

直觉上，这篇文章的核心 insight 是：**mixing 不是一次性优化问题，它是一个状态化的、随开发流程演进的优化问题**，并且过去的最优解里有大量可以廉价复用的信息。

---

## 2. Offline Mixing Schema 的三步形式化

许多方法 (RegMix, DML, ADMIRE-BayesOpt, CLIMB 等) 都遵循同一个 schema，论文称之为 "offline mixing schema"，分三步：

### Step 1: Swarm Construction

给定 domain set $\mathcal{D} = \{D_1, ..., D_m\}$，每个 domain $D_i$ 有 $N_i$ 个 token。从某个分布 $\mathcal{P}$ 采样 $K$ 个 mixture $p^1, p^2, ..., p^K \in \Delta^{m-1}$ (概率单纯形)，每个 mixture 训一个 small proxy model (参数量 $S_{\text{small}}$，训 $R_{\text{small}}$ 个 token)。每个 proxy model 在 $n$ 个 downstream task 上评估，得到 $y_{ij} := f_i(\text{LM}(S_{\text{small}}, R_{\text{small}}, p^j))$，其中 $f_i$ 是 task $i$ 的 BPB (bits-per-byte)。

这里 **BPB = NLL of gold answer normalized by UTF-8 byte length**。它比 raw loss 更可比，Heineman et al. 2025 显示 BPB 在小模型上就能做决策；Huang et al. 2024 ("Compression represents intelligence linearly") 显示 BPB 与 downstream 在跨模型家族下都线性相关。这是这篇 paper 选择 BPB 而非 accuracy 作为目标的关键理由——accuracy 在 30M 模型上太 noisy。

### Step 2: Regression Model

学习 $\hat{f}(p) \approx \frac{1}{n}\sum_{i=1}^n f_i(\text{LM}(S_{\text{small}}, R_{\text{small}}, p))$，从 mixture weight 直接预测平均 BPB。

### Step 3: Mixture Optimization

求解 $\min_{p \in \mathcal{S}} \hat{f}(p)$，$\mathcal{S} \subseteq \Delta^m$ 是可行集 (可能带约束)。

整个 schema 看起来很 clean，但论文 Table 1 列出 6 个现有方法对 7 个 design choice 的处理，发现：很多 cell 是空白的 (没解释)；有解释的 cell 互相矛盾；data constraint 这个真实问题没人处理。这就是 P1 要解决的事。

---

## 3. P1: 七个 RQ 的实证研究 (Section 3)

### 3.1 实验设置 (Appendix D.1)

- **Data**: DCLM 用 WebOrganizer 分成 24 个 topic domain (见 Table 8)；同时也在 source level 做实验 (DCLM, Stack-Edu, ArXiv, FineMath-3+, olmOCR PDFs, Wikipedia, peS2o)
- **Target model**: 1B Olmo 2 decoder-only transformer，n_layers=16, n_heads=16, d_model=2048, head_dim=128，训 100B tokens (5x Chinchilla optimal)，batch=512，seq=4096，lr=0.0018 with cosine+linear decay
- **Proxy model**: 默认 30M (4 layers, 8 heads, d_model=256, head_dim=32)，训 3B tokens (5x Chinchilla)
- **Evaluation**: 52 个 task 的 BPB，跨 math/code/commonsense QA (见 Table 9)，subtask 当独立 task 处理后做 macro-average

### 3.2 RQ1: Proxy model size

实验做的是 proxy-target correlation：训多个 proxy-target pair 用相同 mixture，算 Spearman rank correlation。

**关键发现 (Figure 3)**: 1M 参数的 proxy 不够 (ρ=0.73)，15M 起 ρ > 0.89。最终选 30M, ρ = 89.6 与 1B target。

直觉解读：proxy model 需要至少学到 "哪个 mixture 哪个 task 上更好" 这个 ordering，不需要学到绝对 performance。1M 模型容量太小，连 ordering 都学不稳。这与 RegMix 用 1M 的选择直接矛盾——他们的 setting 可能更简单。

### 3.3 RQ2: Swarm size 与 domain 数 m 的关系

这是这篇 paper 最 actionable 的发现之一。用 log-linear regression model (后面 RQ4 推荐)，做 swarm size sweep：$K = c(m+1)$，$c \in \{1,2,3,4,5\}$，对 $m \in \{6, 12, 18, 24\}$ 做实验。

**关键发现 (Figure 4)**: 当把 error 对 c (而不是对 K) 画时，不同 m 的曲线 collapse 在一起。这意味着 **sample complexity 是 $\mathcal{O}(m)$，不是 $\mathcal{O}(m^2)$**。这与 Ye et al. 2025 (DML) 和 Ge et al. 2025b (BiMix) 假设的二次 scaling 矛盾。

为什么是 $m+1$ 而不是 $m$：因为 log-linear model $\hat{f}_i(p) = c_i + \exp(A_i^\top p)$ 需要至少 $m+1$ 个样本才能 uniquely solve (一个常数项 + m 个线性系数)。

**结论**: 推荐 $K \ge 3(m+1)$，因为 $c=3,4,5$ 的 error 接近 0。

这个发现非常 actionable：如果你有 24 个 domain，你只需要 $\approx 75$ 个 proxy run；64 个 domain 只需要 $\approx 195$。这给 practitioners 一个明确的预算公式。

### 3.4 RQ3: Swarm distribution

研究两个轴：

**(a) Sparse vs Dense**：dense 是每个 proxy run 都覆盖所有 domain；sparse 是允许某些 domain weight = 0。

**关键发现 (Figure 5)**: topic level 上 sparse 好，source level 上 dense 好。论文的解释：DCLM topic 里有些低信号 domain (如 adult content)，sparse swarm 能学到 "这些 domain 应该被排除"；source 是手动 curated 的，每个都有用，dense swarm 保证都覆盖。

**(b) Dirichlet prior 的选择**: natural prior (按 token 比例)、strong prior (用 natural swarm 学出的最优 mix)、weak prior (反着来，maximize BPB)。

**关键发现 (Table 2)**: natural 和 strong 差不多 (BPB 0.765 vs 0.763)，weak 显著差 (0.797)。结论：用 strong prior 最好，但如果没有先验知识，natural 是合理 fallback。

直觉：Dirichlet prior 把 swarm "锚" 在一个 promising region。weak prior 把 swarm 推到 mixture space 的烂区，浪费样本。natural prior 在 data 多的 domain 上集中——通常是合理的初始猜测。

### 3.5 RQ4: Regression model family (核心 RQ)

比较 6 个 regression model：
1. Search (baseline)：直接选 swarm 中最好的
2. LightGBM (RegMix 用)
3. Gaussian Process with RBF kernel (ADMIRE-BayesOpt 用)
4. BiMix: $\hat{f}_i(p) = \sum_{j=1}^m A_{ij} p_j^{-\alpha_{ij}}$，其中 $A_{ij}, \alpha_{ij} \in \mathbb{R}^+$。变量解读：$A_{ij}$ 是 domain $j$ 对 task $i$ 的"基础贡献系数"，$\alpha_{ij}$ 是 power law 指数，$p_j^{-\alpha_{ij}}$ 表示 domain $j$ 占比越小，loss 越大 (power law 形式)
5. AutoScale: $\hat{f}_i(p) = c_i + \sum_{j=1}^m (R(A_{ij} + p_j))^{-\alpha_{ij}}$，其中 $c_i \in \mathbb{R}^+$ 是 task-specific 常数，$A_{ij} \in [0,1]$ 是"effective token from other domains"，$\alpha_{ij} \in \mathbb{R}^+$ 是 power law 指数，$R$ 是 requested token 总数。这里 $R(A_{ij} + p_j)$ 可以理解为 domain $j$ 的 effective token count
6. Log-linear (DML 启发): $\hat{f}_i(p) = c_i + \exp(A_i^\top p)$，其中 $c_i \in \mathbb{R}^+$ 是常数项，$A_i \in \mathbb{R}^m$ 是 per-task, per-domain 的系数向量。$A_{ij}$ 直觉上是 "domain $j$ 对 task $i$ 的 log-scale 影响系数"。

**关键发现 (Figure 6)**:
- 左图 (regression fit vs swarm size): swarm size 是关键 confounding factor。BiMix 在 K=25 最好，LightGBM 需要 K > 118 才 fit 好，log-linear 在 K ≥ 75 (即 K ≥ 3(m+1) for m=24) 持续最好，最终 ρ = 0.80
- 右图 (downstream BPB @ K=128): log-linear 最优

**重要 insight**: 之前文献的"矛盾" (每个 paper 都说自己的 model 最好) 主要来自 swarm size 不同。BiMix 适合小 swarm (因为参数少、好 fit)；LightGBM 适合大 swarm (因为它高 var、低 bias，需要数据填满树)；log-linear 是 sweet spot——参数恰好够 ($m+1$ 个参数)，又在大 swarm 下保持 convex 和可解释。

**推荐**: log-linear model + $K \ge 3(m+1)$。

### 3.6 RQ5: Regression granularity

三个选择：
- Per-task: 对每个 task $i$ 拟合 $\hat{f}_i(p)$，最后 $\min \frac{1}{n}\sum_i \hat{f}_i(p)$
- Per-family: 把 task 分成 math/code/QA 三族，每族一个 model
- Aggregated: 一个 model 直接预测平均 BPB

**关键发现 (Table 3)**: per-task 最优 (BPB 0.765, fit 0.983)，per-family 中等，aggregated 最差 (fit 0.866)。

直觉：per-task 拟合时每个 task 有自己的 mixture-response 关系 (e.g., math task 与 math domain 强相关，code task 与 code domain 强相关)，aggregating 会 blur 这些信号。代价是每个 task 单独 fit，sample 数不变但 fit 数变多——所以 fit 上还是更好。

### 3.7 RQ6: Data repetition constraints (核心 RQ，文献第一次系统处理)

这是 paper 最实用的 finding 之一。现有方法都假设无限数据，但实际中某些 mixture 会要求 sample 某个 domain 的 token 超过它的存量。比如 mixture 要 40% code，但 code 只占 5% 数据，那就重复 8 次。Muennighof et al. 2025 显示 >4 epoch 重复会显著降性能。

**形式化约束**: $p_j \le \frac{k N_j}{R}$ for all $j$。变量：$p_j$ 是 domain $j$ 的 mixture 比例，$N_j$ 是 domain $j$ 的可用 token，$R$ 是要训的总 token，$k$ 是允许的最大 epoch 数 (重复因子)。

实验比较三种 enforce 策略 (Table 4)：
1. Constrained swarm + unconstrained opt: swarm 里所有 mix 满足约束，但 optimization 不加约束 → 失败 (实际重复 5 次，超过 k=4)
2. Unconstrained swarm + constrained opt: swarm 任意采，但 optimization 加约束 → 成功，BPB 最优 (0.764718)
3. Constrained swarm + constrained opt: 都加约束 → 满足约束但 BPB 差 (0.785517)

**为什么 1 失败**: 即使 swarm 满足约束，regression model 学到的是"满足约束区域"的 loss surface。optimization 在这个 surface 上走，可能走到 swarm 没采样过的、不满足约束的 promising region。

**为什么 3 差**: swarm 限定在约束域内，对 mixture space 的覆盖变窄，regression fit 变差。

**为什么 2 好**: swarm 覆盖全空间 (好的 fit)，optimization 显式加约束 (feasibility 保证)。

**Figure 7**: 变 $k \in \{2,3,4,5,\infty\}$ 看 mixture 怎么变。Software development 这种高 utility domain 在 $k$ 放松时单调上升，literature 这种低 utility 但数据大的 domain 下降 (它们在紧约束下主要起到"补满"作用)。这说明约束**显著塑造** proposed mix。

直觉：tight constraint 把 mixture 往 natural distribution 推；loose constraint 让 optimization 找到真正的 optimum。Practitioner 应该意识到自己选 $k$ 是在做一个真实的 trade-off。

### 3.8 RQ7: Optimization solver

三个选择：
- Exact solver: CVXPY 直接解 (因为 log-linear 下 $\hat{f}$ 凸)
- Search: 从 Dirichlet 采样 candidate，取最好的 (RegMix 用)
- Exact + KL regularization: $\min_p \hat{f}(p) + \lambda D_{KL}(p \| p_0)$，$\lambda > 0$，$p_0$ 是 natural distribution

**关键发现 (Figure 8)**:
- Left (downstream BPB): Exact + KL(λ=0.05) 最优
- Right (predicted BPB @ 30M): Exact 最低 (这个是数学上必然的，因为它直接最小化 $\hat{f}$)

**重要 insight**: exact solver 数学上最优，但 regression 模型是 imperfect surrogate，且 proxy-to-target transfer 有噪声。在 surrogate objective 上过度优化会 overfit 到 surrogate 的偏差。KL regularization 给 natural distribution 一个"安全网"，防止 optimization 跑到 surrogate 的"假高 promise"区域。

**推荐**: Exact solver with KL penalty λ=0.05。

直觉：这就像 RL 中的 trust region 思想——你不完全相信你的 (imperfect) model，所以加一个 prior 防止走太远。这个 prior 强度需要 calibrate，λ=0.05 在他们的 setting 下刚好。

### 3.9 OlmixBase 算法 (Algorithm 1)

汇总上面所有发现：

```
Input: domains D = {D_1, ..., D_m}, sizes {N_i}, K = O(m), k, R, λ=0.05, p_0 ∝ {N_i}

1. Sample K mixes p^1, ..., p^K from Dirichlet (sparse for topic, dense for source)
2. Train S_small ≥ 15M proxy on each mix; evaluate → {(p^j, {y_ij})}
3. For each task i: fit log-linear f̂_i(p) = c_i + exp(A_i^T p)
4. Solve: min_p (1/n) Σ f̂_i(p) + λ D_KL(p || p_0)
   s.t. p_j ≤ k N_j / R for all j
5. Return p*
```

蓝色高亮 = 论文实证研究得到的 design choice。

---

## 4. P2: Evolving Domain Problem (Section 4)

### 4.1 Domain update operators

论文从 Olmo 1-3 和 SmolLM 1-3 的开发经验中提炼出 4 个 update operator：
- **Add**: $\mathcal{D}_2 = \emptyset$, $\mathcal{D}_2'$ 是新 domain (e.g., 新发布的 dataset)
- **Remove**: $\mathcal{D}_2$ 是要删的 domain, $\mathcal{D}_2' = \emptyset$
- **Partition**: $\mathcal{D}_2 = \{D\}$ 一个 domain, $\mathcal{D}_2' = \{D_1', ..., D_\ell'\}$ 是 subdomain，$D = \bigcup D_i'$
- **Revise**: $\mathcal{D}_2 = \{D\}$, $\mathcal{D}_2' = \{D'\}$ 同维度但内容改了 (e.g., reformat)

**Formal goal (公式 1)**: 给定旧 mix $\tilde{p} \in \Delta^{m-1}$，求新 $q^* \in \Delta^{m'-1}$ 使
$$\min_{q \in \Delta^{m'-1}} \frac{1}{n} \sum_{i=1}^n f_i(\text{LM}(S, R, q))$$
$$\text{s.t. } q_j \le \frac{k N_j'}{R} \quad \forall j \in [m']$$

**Baseline: Full Recomputation** = 直接在 $\mathcal{D}'$ 上跑 OlmixBase，需要 $\mathcal{O}(m')$ 个 proxy run。Update 多次后代价爆炸。

### 4.2 Mixture Reuse 的核心 trick

**核心 insight**: 如果 update 只影响部分 domain，那剩下 $\mathcal{D}_{\text{fix}}$ (unaffected domain) 的**相对**比例应该不变 (在 update 的方向上最优解不变太多)。所以冻结它们的相对比例，把它们 collapse 成一个 virtual domain，只重算这个 virtual domain 的总权重 + 受影响 domain 的权重。

**形式化**: partition $q = [\rho q_{\mathcal{D}_{\text{fix}}}, (1-\rho) q_{\mathcal{D}_{\text{comp}}}]$，其中 $\rho \in [0,1]$ 是 unaffected domain 的总权重，$q_{\mathcal{D}_{\text{fix}}} \in \Delta^{|\mathcal{D}_{\text{fix}}|-1}$ 是 unaffected 内部相对比例，$q_{\mathcal{D}_{\text{comp}}} \in \Delta^{|\mathcal{D}_{\text{comp}}|-1}$ 是 affected 内部相对比例。

加约束 $q_{\mathcal{D}_{\text{fix}}} = \tilde{p}_{\mathcal{D}_{\text{fix}}}$ (复用旧 mix 在 unaffected 部分的归一化比例)。

### 4.3 Change of variables (Algorithm 2 的数学基础)

定义 collapsed mixture $r = [\rho, (1-\rho) q_{\mathcal{D}_{\text{comp}}}] \in \Delta^{|\mathcal{D}_{\text{comp}}|}$，index 集为 $\{v\} \cup \mathcal{D}_{\text{comp}}$，其中 $r_v = \rho$ 是 virtual domain 权重。

定义 expansion function $\Phi_{\tilde{p}_{\mathcal{D}_{\text{fix}}}}(r)$: $q_j = r_v \cdot \tilde{p}_j$ for $j \in \mathcal{D}_{\text{fix}}$，$q_j = r_j$ for $j \in \mathcal{D}_{\text{comp}}$。

**Lemma 1**: 这个变量替换把 mixture reuse problem 变成在 $|\mathcal{D}_{\text{comp}}|$ 维空间 (而非 $m'$ 维) 上的 standard mixing problem。Repetition constraint $\rho \tilde{p}_j \le \frac{k N_j'}{R}$ 变成 $\rho \le \min_{j \in \mathcal{D}_{\text{fix}}} \frac{k N_j'}{R \tilde{p}_j}$ (virtual domain 上一个 constraint)。

**例子 (论文给的)**: $\tilde{p} = [0.25, 0.25, 0.5]$ (3 domain)，加 1 个新 domain。不 reuse 要学 4 维 mixture。Reuse 把前 3 个 collapse 成 virtual domain ($\tilde{p}_{\mathcal{D}_{\text{fix}}} = [0.33, 0.33, 0.67]$ 归一化)，学 2 维 mixture $r = [0.4, 0.6]$，expand 回去 $q = [0.1, 0.1, 0.2, 0.6]$。

**关键收益**: swarm size 从 $\mathcal{O}(m')$ 降到 $\mathcal{O}(|\mathcal{D}_{\text{comp}}|)$。如果 add 1 个 domain 到 64 个里，从 65×3=195 个 proxy run 降到 2×3=6 个。

### 4.4 理论分析 (Section 4.3)

这部分是 paper 最数学化的内容，目的是回答"什么时候 FullMixtureReuse 和 Full Recomputation 性能一样"。

**Assumption 1**: log-linear model 真的 hold: $f_i(\text{LM}(S,R,q)) = c_i + \exp(A_i^\top q)$。这个 assumption 强但 allow 闭式分析。

**Definition**: $F(q) := \frac{1}{n}\sum_i f_i(\text{LM}(S,R,q))$。$q^*$ 是公式 1 的解 (full recomputation)，$q^*(\tilde{p}_{\mathcal{D}_{\text{fix}}})$ 是公式 2 的解 (mixture reuse)。$q_{\mathcal{D}_{\text{fix}}}^*$ 是 $q^*$ 在 unaffected 上的归一化。

**Coupling term (定义在 Appendix C.1 公式 10)**:
$$\kappa(\alpha_{\text{fix}}, \alpha_{\text{comp}}) := \|(1 + \alpha_{\text{fix}} + \alpha_{\text{comp}}) \odot \alpha_{\text{fix}}\|$$
其中 $\alpha_{\text{fix}} \in \mathbb{R}^n$, $\alpha_{i,\text{fix}} = \|A_{i,\text{fix}}\|$ (task $i$ 在 unaffected domain 上的系数向量的 norm)，类似 $\alpha_{\text{comp}}$；$\odot$ 是 Hadamard 积；$1$ 是全 1 向量。

直觉：$\alpha_{i,\text{fix}}$ 大 = unaffected domain 对 task $i$ 影响大；$\alpha_{i,\text{comp}}$ 大 = affected domain 对 task $i$ 影响大。两者同时大 = 同一个 task 受两类 domain 共同影响 = "耦合"高，reuse 风险大。

#### Theorem 1 (Performance gap bound)

$$F(q^*(\tilde{p}_{\mathcal{D}_{\text{fix}}})) - F(q^*) \le C_1 \kappa(\mathcal{D}_{\text{fix}}, \mathcal{D}_{\text{comp}}) \|\tilde{p}_{\mathcal{D}_{\text{fix}}} - q_{\mathcal{D}_{\text{fix}}}^*\|$$

变量：$C_1 > 0$ 是常数 (论文 Appendix C.2.3 给出 explicit 形式)，$\|\tilde{p}_{\mathcal{D}_{\text{fix}}} - q_{\mathcal{D}_{\text{fix}}}^*\|$ 是 **reuse gap** (你 reuse 的比例和更新后真实最优比例的距离)。

**两个控制项**:
1. **Reuse gap**: 你 reuse 的 mix 和 update 后最优 mix 的距离——直觉上 update 影响小则 reuse gap 小
2. **Coupling term**: unaffected 和 affected 在 task 上的"重叠"程度——直觉上两类 domain 影响不同 task 则耦合低

证明思路 (Appendix C.2): 
- Lemma 2: 当 $\tilde{p}_{\mathcal{D}_{\text{fix}}} = q_{\mathcal{D}_{\text{fix}}}^*$ (reuse 完美) 时，performance gap = 0
- Lemma 3 (FOC inequality): 两个最优解的 first-order condition 给出一个 inner product 不等式
- Lemma 4: 沿着 $p_1^t = t p_1' + (1-t) p_1$ 的 gradient 用 mean value theorem bound
- Lemma 5: $\|\Delta r\| \le \frac{M}{\mu} \|\Delta\|$ — strong convexity 给出 argmin 的 Lipschitz 性质
- Lemma 6: 把 performance gap 分解为两个 term
- Lemma 7-9: 三个 gradient norm 的具体 bound (Cauchy-Schwarz 反复用)
- 最后合起来

证明很 technical 但 intuition 清晰: log-linear model 的指数结构让 gradient norm 由系数 $A$ 控制，而 coupling term 自然地从 cross-terms 中冒出来。

#### Theorem 2 (Reuse gap bound for Add update)

假设 $\tilde{p}$ 在 $\mathcal{D}$ 上最优，update 是 Add。则
$$\|\tilde{p}_{\mathcal{D}_{\text{fix}}} - q_{\mathcal{D}_{\text{fix}}}^*\| \le C_2 \kappa(\mathcal{D}_{\text{fix}}, \mathcal{D}_{\text{comp}}) (1 - \rho^*)$$

变量：$C_2 > 0$ 常数，$\rho^*$ 是 $q^*$ 在 unaffected domain 上的总权重 (公式见 Appendix C.3.1 Theorem 4)。

**两个控制项**:
1. **Coupling term** (同 Theorem 1)
2. **$1 - \rho^*$**: 这是 update "搬动" optimum 的程度。$1 - \rho^*$ 小意味着 affected domain 在新最优里占比小。

**$1 - \rho^*$ 什么时候小**:
- Added domain 低 utility (即使加了，最优还是把它们权重压低)
- Added domain 数据少 (repetition constraint 把它们 cap 住)

证明思路 (Appendix C.3): 用 Lemma 10 (argmin stability with mutual feasibility) — 两个 strongly convex 优化问题的 argmin 距离被 gradient 差的 norm 控制。Lemma 11 说当 $\rho^* = 1$ 时 reuse gap = 0 (因为加的 domain 没用，optimum 不动)。Lemma 12 把 $\|p_1 - p_1'\|$ 跟 $1 - \rho^*$ 联系起来：用 mean value theorem 沿 $r^t = tr_0 + (1-t) r^*$，其中 $r_0 = [1, 0]$ 是 update 前的状态。

**联立 Theorem 1 + Theorem 2**: Performance gap $\le C_1 C_2 \kappa^2 (1 - \rho^*)$。两个条件 (低耦合 + 小 update 影响) 都满足时，FullMixtureReuse 几乎无损。

### 4.5 PartialMixtureReuse (Section 4.4)

Theorem 1 说当 coupling $\kappa$ 大时 FullMixtureReuse 不好。PartialMixtureReuse 把一部分 unaffected domain 也 recompute (不只 reuse)，从而降低 coupling。

形式化：选 $\mathcal{D}_{\text{partial}} \subset \mathcal{D}_1$，重新定义 $\mathcal{D}_{\text{fix}} := \mathcal{D}_{\text{partial}}$，$\mathcal{D}_{\text{comp}} := (\mathcal{D}_1 \setminus \mathcal{D}_{\text{partial}}) \cup \mathcal{D}_2'$，然后 apply FullMixtureReuse (Algorithm 2)。

新维度 = $1 + |\mathcal{D}_2'| + |\mathcal{D}_1| - |\mathcal{D}_{\text{partial}}|$，在 FullMixtureReuse 的 $1 + |\mathcal{D}_2'|$ 和 Full Recomputation 的 $m'$ 之间。

**怎么选 $\mathcal{D}_{\text{partial}}$**: 论文用 intuitive 启发式——选那些和 affected domain 影响相同 task 的 domain。e.g., 加 Stack-Edu (code) 时，DCLM:software_development 这个 topic 也影响 code task，应该一起 recompute。Figure 17 验证：recompute software_development 时 $\kappa$ 下降最多。

**Limitation** (Section 6): 论文没给自动选 $\mathcal{D}_{\text{partial}}$ 的方法，需要 domain expertise。这是 future work。

---

## 5. 实验结果 (Section 5)

### 5.1 Real-world LM development scenario (Section 5.1)

**Setup**: 模拟 Olmo 3 开发流程，5 个 update (Table 5):
1. Initial: DCLM 24 topic domain
2. Add: Stack-Edu 15 个编程语言 → 39 domain
3. Add: ArXiv, FineMath-3+, olmOCR PDFs, Wikipedia, AlgebraicStack, peS2o → 45 domain
4. Revise: olmOCR PDFs reformat → 45 domain
5. Remove: AlgebraicStack → 44 domain
6. Partition: olmOCR PDFs 拆成 21 topic → 64 domain

**Method 比较**:
- Natural (按 token 比例)
- Full recomputation (OlmixBase 每次重算)
- Swarm reuse (复用所有旧 swarm run，映射到新 domain set，Algorithm 3)
- FullMixtureReuse
- PartialMixtureReuse

**主要结果 (Figure 10, Table 12-13)**:
- FullMixtureReuse: +11.6% over natural，**用 216 个 proxy run** (vs full recomp c=3 用 832 个)，**省 74% compute**，capture 95% 的 full recomp gain
- PartialMixtureReuse: +12.0%，用 272 个 run，省 67%，capture 98%
- Swarm reuse: +11.4%，268 个 run，比 mixture reuse 略差且用更多 run

**为什么 swarm reuse 差**: 
1. 旧 swarm 在新 domain set 上 represent 时 over-explore biased subspace (旧 swarm 的 affected domain weight 都是 0，相当于强行把 affected domain 推到 0)
2. Remove 和 Revise 操作下旧 swarm 的 run 直接废了 (旧 mix 在新 domain set 上没法 represent)

**Data efficiency (Figure 11)**: 用 PartialMixtureReuse 的最优 mix，1B target 在 20K 步就达到 natural distribution 的最终 BPB，natural 需要 61K 步——**3.05x data efficiency**。这个数字非常 impressive，因为 mixture 本身没有改变 model 架构、训练 token 数、optimizer，只改变了哪些 token 先看。

**Mix 相似度 (Figure 12, Table 14)**: PartialMixtureReuse 与 full recomp 的 total variation distance = 0.067，与 natural distance = 0.127。Top-level 看：PartialMixtureReuse 把更多 weight 给 finemath-3+ (0.136)、stack-edu:Python (0.053)、stack-edu:Markdown (0.085)、DCLM:software_dev (0.025) 等高质量 source。Natural 把大量 weight 浪费在 DCLM:politics (0.098) 这种 token 多但 utility 不高的 domain 上。

### 5.2 Theory validation (Section 5.2)

这部分很重要——它不只验证 method 工作，还验证理论 **predict** 得对。

**Reuse gap vs performance gap (Figure 13)**: 构造不同 $\tilde{p}_{\mathcal{D}_{\text{fix}}}$ (weak mix, intermediate mix, optimal mix)，量 reuse gap，跑 FullMixtureReuse 看性能 gap。结果：reuse gap 单调预测 performance gap，Theorem 1 成立。

**$1 - \rho^*$ vs reuse gap vs performance gap (Figure 14)**: 通过变 $R$ (1T vs 6T) 变 repetition 约束松紧，改变 $\rho^*$。结果：$1 - \rho^*$ 单调预测 reuse gap 和 performance gap，Theorem 2 成立。R=1T 时 $1-\rho^*$ 大 (放松约束，让新 domain 占比大)，R=6T 时小 (紧约束，optimum 几乎不动)。

**PartialMixtureReuse 减少 coupling (Figure 15-17)**: Figure 15 显示 DCLM:software_development 是唯一在 $\tilde{p}_{\mathcal{D}_{\text{fix}}}$ 和 $q_{\mathcal{D}_{\text{fix}}}^*$ 之间显著不同的 domain，暗示它和 Stack-Edu 高耦合。Figure 16 验证：recompute software_development 时 reuse gap 和 performance gap 都降，即使 $1-\rho^*$ 略升 (这违反 Theorem 2 单调预测，但因为 coupling 下降 dominate)。Figure 17 直接算 $\kappa$，证实 software_development 降 coupling 最多。

这种 "理论预测 + 实验验证" 的闭环在 systems paper 里少见，是这篇 paper 的亮点。

### 5.3 R=6T 结果 (Figure 24, Appendix D.3)

在更 data-constrained 的 R=6T 设定下，FullMixtureReuse 已经 capture 99% 的 full recomp gain (+6.94% vs +6.97%)。PartialMixtureReuse 在这个 setting 下没必要。直觉：R 紧 → constraint 把 affected domain weight cap 住 → $1-\rho^*$ 小 → Theorem 2 说 reuse gap 小 → Theorem 1 说 performance gap 小。这进一步验证理论。

### 5.4 Low budget regime (Figure 25)

Budget 76-272 run 的低预算 setting：
- Full recomp c=1 最少 267 run (因为每次都要 $m+1$)
- Mixture reuse/swarm reuse 可以低到 76 run
- 76 run 时 mixture reuse 仍有 +9.6% gain over natural (R=1T)

这对小团队意义重大：你不需要几百个 GPU 跑 800 个 proxy model 也能做 mixing。

### 5.5 Per-operator ablation (Figure 26)

对每个 update operator 单独看 FullMixtureReuse vs natural：
- Add/Remove/Partition 上 FullMixtureReuse 都显著接近 full recomp
- Revise 上 TV distance 只 0.21% (混和几乎一致)，论文没测 performance 因为差异会被噪声盖过

---

## 6. 关键 Insight 和 Karpathy 可能关心的点

### 6.1 为什么 log-linear model 胜出

log-linear 形式 $\hat{f}_i(p) = c_i + \exp(A_i^\top p)$ 的几个优势：
1. **参数恰好**: $m+1$ 个参数/task，对应 $m+1$ 个 swarm run 就 uniquely solvable。BiMix 参数 $2m$，AutoScale 更复杂，需要更多 run
2. **Convex**: $\exp$ 是凸的，$A_i^\top p$ 是线性的，所以 $\exp(A_i^\top p)$ 凸。Convexity 让 exact solver (CVXPY) 可用
3. **可解释**: $A_{ij}$ 直接是 "domain $j$ 对 task $i$ 的 log-scale 影响系数"，符号和大小都有物理意义
4. **Sample complexity**: $\mathcal{O}(m)$，因为参数 $\mathcal{O}(m)$

为什么 log-linear 而非 linear? 因为 LM loss 在 domain 上的 response 通常是 exponential (power law) 形式——domain weight 翻倍不会线性降 loss。这是 Data Mixing Laws (Ye et al. 2025) 的核心观察。

### 6.2 Mixture reuse 的更深 intuition

考虑 update 前的 optimality condition: $\nabla F(\tilde{p}) = 0$ (在 simplex 上是 KKT condition)。Update 后 $\nabla F'(q^*) = 0$。如果 update 只改了 affected domain 的 loss function (在 log-linear 下只改了 $A_{i,\text{comp}}$)，那么 $\tilde{p}_{\mathcal{D}_{\text{fix}}}$ 在 unaffected 部分的 gradient component 不变 (因为 $A_{i,\text{fix}}$ 不变)。所以 $\tilde{p}_{\mathcal{D}_{\text{fix}}}$ 在新 problem 下还是 unaffected 部分的 stationary point，只是 $\rho$ 需要重新平衡。这就是 mixture reuse 工作的本质——log-linear 结构让 unaffected 部分的 internal optimality 不随 update 改变。

Coupling term $\kappa$ 衡量的是：affected domain 改了之后，通过共享 task，会"间接"改变 unaffected domain 的最优比例。如果 task 不重叠 ($A_{i,\text{fix}}$ 和 $A_{i,\text{comp}}$ 不同时非零)，则 $\kappa$ 小，update 完全 decoupled。

### 6.3 数据效率 3.05x 的来源

Figure 11 的 3.05x 不是简单的"better mix 让你学得快"。而是：mixture 把高质量 token 放前面，model 先学到核心 capability (math, code)，再慢慢填 general knowledge。Natural distribution 把大量 weight 给 politics、entertainment 这些 token 多但 task 上不太有用的 domain，相当于浪费早期 compute。

这跟 "curriculum learning" 直觉一致，但不是手设计 curriculum——而是 mixture optimization 自动发现这个 curriculum。

### 6.4 Repetition constraint 的实际意义

$k=4$ 的选择基于 Muennighof 2025 的 scaling law for repeated data。这个 constraint 把 mixing problem 从"假设计算无限数据"拉回现实。Figure 7 显示，这个 constraint 不只是"防止坏结果"——它**主动 shape** 了 proposed mix。不加 constraint 时，proposal 会给 high utility domain 极高 weight (即使要重复 10 次也没关系，因为模型不知道)；加了之后，proposal 学会"用 low utility but abundant domain 来补 token"。

这是 ML system paper 里少见的——把一个"工程约束"变成优化问题的 first-class 部分，并展示它改变解的形状。

### 6.5 与 Chameleon (Xie et al. 2025) 的对比

Chameleon 也想解决 domain 演化问题，但思路不同：
- Chameleon: 用 domain embedding 算 kernel ridge leverage score，直接给新 domain weight，不需要 retraining
- Olmix mixture reuse: 保留 offline schema，复用 ratio 而非重新算 weight

Chameleon 的优势：超便宜 (zero proxy run for new domain)
Chameleon 的劣势：
1. 只针对 Add 操作，不 handle Remove/Partition/Revise
2. 没有 swarm-based regression 的 task-aware 信号——embedding-based 是 unsupervised
3. 没理论保证 performance gap

Olmix 的 mixture reuse 是更 general 的框架，trade off 是需要少量 proxy run。

### 6.6 与 DoReMi/DoGE 的对比

DoReMi 用 importance sampling 思想动态调权重，DoGE 类似。这些是 online 方法，不需要 swarm。优势是 compute 少 (1-2 个 proxy run)；劣势是 dynamic update rule 是 hand-crafted 启发式，论文引 Chen et al. 2025a (Aioli, 同作者前序工作) 指出其 suboptimality。

Olmix 是 offline schema，更显式、更可解释、更可复用，但需要更多 proxy compute。

### 6.7 Limitation 和 future work

论文自己列了三点：
1. 只研究 offline schema，online 方法 (DoReMi 类) 可能要不同 design
2. 理论分析只针对 log-linear model；non-parametric model (LightGBM, GP) 难 extend
3. PartialMixtureReuse 选 $\mathcal{D}_{\text{partial}}$ 需要手动，没自动化

我会再加几个我看到的：
4. **Proxy-target gap**: 论文只验证到 1B target。32B 上 proxy 30M 的 rank correlation 可能下降 (虽然 Heineman 2025 说 BPB 跨 scale 稳，但 mixing 这种 sensitive 决策可能不行)。论文 Section 6 自己提到要 validate at larger scale
5. **Task weights**: 论文 macro-average 52 个 task，把 math, code, QA 等权。如果 practitioner 想 emphasize 某 capability，objective 要改，论文没讨论
6. **Domain granularity**: WebOrganizer 给的 topic 是一个特定 partition。换 partition (e.g., 更细的 sub-topic) 会改变 optimal mix。这是 mixture 的"resolution"问题
7. **Multi-stage training**: Olmo 3 实际有 multi-stage (continue pretraining + midtraining)。Mixture 在不同 stage 应该不同。这篇只看 pretraining 第一阶段
8. **Online drift**: 论文的 mixture 是 static 的 (整个训练用同一比例)。Online mixing 在训练中调比例 (Albalak et al. 2023)，可能比 static 好，论文 Section 6 也提到这是 future work

---

## 7. 总结

这篇 paper 的核心贡献可以总结为三句话：

1. **OlmixBase = 7 个 design choice 的实证最优组合**：30M proxy, $K=3(m+1)$ swarm, Dirichlet (sparse for topic / dense for source), log-linear regression per-task, exact solver with KL λ=0.05, repetition constraint in optimization。这套配置在 Olmo 3 开发中实际用。

2. **Mixture reuse = 冻结 unaffected domain 的相对比例 + collapse 成 virtual domain**：把 recomputation 维度从 $m'$ 降到 $1 + |\mathcal{D}_{\text{comp}}|$，理论上 performance gap 由 coupling $\kappa$ 和 reuse gap 控制，实验验证理论。

3. **3.05x data efficiency**: PartialMixtureReuse 给出的 mix 在 Olmo 3 development 流程中让 1B model 用 1/3 的 step 达到 natural distribution 的最终 BPB，capture 98% of full recomputation's gain 用 67% less compute。

工程意义上，这是把"手工调 mixture ratio"变成"有理论指导的状态化优化流程"的工作。学术意义上，它把 offline mixing schema 的 design space 系统化，并第一次把 domain evolution 作为 first-class 问题来研究。

对 Karpathy 这样的 audience 来说，最值得关注的可能是：
- Log-linear model 的 $\mathcal{O}(m)$ sample complexity 这个 actionable 数字
- Mixture reuse 的 collapse/expand trick——这是一个 elegant 的 reparametrization 让你保留所有原有 machinery (OlmixBase) 但维度降低
- 理论 + 实验闭环的 coupling/reuse gap 分析——这是 systems paper 里罕见的"theory informed design"案例
- 3.05x data efficiency 暗示 mixing 决策可能是 curriculum learning 的一种 principled 自动化形式
