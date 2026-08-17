---
source_pdf: Continual Learning viaSparseMemoryFinetuning.pdf
paper_sha256: 3761cddbe948768737e13c91bbb78ffa4478469ce4037e3a08578a6714b57518
processed_at: '2026-08-03T17:13:46-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话总结

你想让 LLM 持续学习新东西又不忘记旧东西？别动所有参数，只动几百个跟新知识"强相关"的参数就行。

---

## 问题在哪

LLM 预训练完就定死了。你想让它学新东西，只能 finetune。但 finetune 一学新知识，旧能力就崩——这就是 **catastrophic forgetting**。

为啥会这样？因为**所有 task 共享同一套参数**。你学"Michelle Smith 被禁赛 4 年"这个新 fact，梯度会去改 model 里所有参数。但其中绝大部分参数还负责"语法"、"通用世界知识"、"数学推理"，你一动它们，这些能力就退化。

数字很触目惊心：在 TriviaQA 上 finetune 1000 条 facts 后，NaturalQuestions F1 掉 **89%**。LoRA 也掉 71%。

---

## 核心直觉

作者抓住一个根本 insight：forgetting 的根源是 **dense parameter sharing**。

你想——大脑不是这么工作的。你记一个新朋友的电话号码，不会把你脑子里所有神经元都重新调一遍。你只在某个局部区域写一小段改动。其他记忆完全不受影响。

那 LLM 为什么做不到？因为它没有"局部化存储"这个 architectural prior。每个 fact 都 distributed 在几十亿参数里，你想改一条就得动一大堆。

**解决思路**：给模型加一个"超大但稀疏"的 memory 模块。每条新知识只激活极少 slot，update 也只 update 那几个 slot，其他全部 freeze。

---

## 怎么做：Memory Layer

借鉴 Berges et al. 2024 的工作，把 Transformer 中间某一层 FFN 换成一个 memory lookup。

想象一个有 **100 万个 memory slot** 的大池子，每个 slot 是一对 key-value。每个 token 进来，用 query 去检索，只取 **top-32** 最相似的 key，加权求和对应的 value。

这就是"稀疏激活"——每个 token 只碰 100 万分之 32 个 slot，比例约 0.003%。但整个 memory pool 容量巨大，可以装下海量知识。

这种 architecture 天然适合 continual learning：每条新知识只激活很少的 slot，update 也只动那几个 slot，理论上其他 slot 完全不受干扰。

---

## 但 naive finetune 还是会 forgetting

如果你直接 finetune 所有"当前 batch 访问到的"slot，结果还是会 forgetting。为什么？

因为某个 batch 访问的几百个 slot 里，**有些是 general-purpose slot**——比如负责"swimmer"、"banned"、"year" 这种通用语义的 slot，它们在 pretraining 时被各种任务高频访问。你 update 这些 slot，就把别的任务依赖的知识污染了。

类比：你想在硬盘里存一条新 fact，这条 fact 引用了一些公共词汇。如果你不仅写新 fact，还去改公共词汇词典的解释，那其他所有引用这些词汇的 fact 都会受影响。

**你需要精准识别"哪些 slot 是这条 fact 独占的，哪些是 general-purpose 的"**。

---

## TF-IDF Ranking

作者借用了信息检索里最古老的 trick：TF-IDF。

对每个 memory slot 算一个分：
- **TF（term frequency）**：这个 slot 在当前 batch 上被访问多少次
- **IDF（inverse document frequency）**：这个 slot 在背景 corpus 上有多罕见

只有"当前 batch 高频 + 背景上罕见"的 slot 才会得高分。general-purpose slot 在背景上到处出现，IDF 接近 0，被自动剔除。

具体怎么算背景？用 1000 个 random DCLM batch（一个公开 pretraining corpus），统计每个 slot 在这些 batch 上被访问的频率。这个统计是一次性的，存进 checkpoint，finetune 时直接查。

然后每个 forward pass：
1. 算当前 batch 访问了哪些 slot、各访问多少次
2. 给每个 slot 算 TF-IDF 分
3. 选 top-$t$ 高分的 slot（实验中 $t$=500 或 10000）作为可训练 slot
4. 其他 slot 的梯度全部截断

---

## 怎么实现"动态 trainable mask"

难点：每个 batch 选的 top-$t$ slot 不一样，怎么在 PyTorch 里实现？

作者用了一个巧妙的 trick：

```python
mem = mem * trainable_mask + mem.detach() - (mem * trainable_mask).detach()
```

拆开看：
- 前向值：可训练位置走 `mem`（带梯度），其他位置走 `mem.detach()`（detach 掉梯度）
- 反向：只有可训练位置有梯度

这样前向输出完全不变，反向梯度只流向 top-$t$ slot。

---

## 实验结果

### 1. Fact Learning（小数据场景）

让模型按顺序学 1000 条 TriviaQA facts。每条 fact paraphrase 成 64 个版本填 batch。

结果：
- Full finetuning：NQ F1 掉 89%
- LoRA：NQ F1 掉 71%
- Sparse memory finetuning：NQ F1 只掉 **11%**
- 同时 TQA 上学到的新 fact 数量，sparse memory 和 baseline 持平甚至更好

### 2. Document QA

让模型按顺序读 1824 个 Wikipedia chunks（来自 100 个 SimpleQA 问题的引用文档）。

结果：full finetuning 和 LoRA 在 target 上能达到同样性能，但 held-out benchmark 上仍然 forgetting 明显。Sparse memory finetuning 在高 lr 和低 lr 下都 Pareto dominate。

### 3. Pareto Frontier

作者 sweep 了各方法的主要超参，画了 target performance vs held-out performance 的散点图。

观察：
- Full finetuning 和 LoRA 存在清晰的 tradeoff：学得越多，忘得越多
- Sparse memory finetuning 的点明显在 frontier 外推——同样的学习量下 forgetting 小得多

**点的 size 表示 trainable params per batch**：sparse memory 的点 size 很小（几百 K params），但 learning capacity 超过 LoRA（数十 M params）和 full finetuning（1.3B params）。这是真正的"小而精"。

---

## 一些有意思的 ablation

### TF-IDF vs TF-only

如果只用 TF（只看当前 batch 频率，不除以背景频率）会怎样？

- 当 $t$ 大时（如 500+），TF-only 和 TF-IDF 差不多——因为两者都接近"update 所有 accessed slot"
- 当 $t$ 小时（如 50），TF-IDF 显著更好——IDF 部分对识别"critical minimal set"至关重要

### Background corpus 选择

用 DCLM、TriviaQA 训练集、NaturalQuestions 三种 background 对比：
- DCLM：learning 好，forgetting 少
- TriviaQA 训练集：forgetting 显著 worse（没下权重 general slots）
- NaturalQuestions：与 DCLM 类似

Insight：background 不需要精确，只要能近似 pretraining 的 general index 分布就行。

### SGD vs AdamW

意外发现：sparse memory finetuning 用 SGD 比 AdamW forgetting 更少。但 baseline 方法用 SGD 反而不见改善。

推测原因：Adam 的 per-parameter adaptive learning rate 对 sparse updates 不友好——每个 slot 的二阶动量历史不一致，给出 inconsistent 的 effective step size。SGD 给统一步长更"诚实"。

---

## Memory 访问模式分析（最有趣的部分）

作者做了一个 qualitative 分析。定义 **core set** = 所有 paraphrase 和 question 共享的 slot（即"承载这条 fact 语义"的 slot）。

发现：
- Core set 典型大小 100-500 个 slot——一条 fact 不是存在单个 slot 里，而是 distributed 在几百个 slot 上
- 但实际只需 finetune 25-100 个 slot 就能让模型答对——远小于 core set
- TF-IDF 选出的 trainable slot 与 core set 高度重合
- 这些 slot 往往 **align with entity boundaries**——比如 "Michelle Smith-de Bruin"、"Walter Hagen"、"Vienna" 这些 entity 名词附近的 token

这个发现很 hint：**entity boundary tokens 是 critical memory reads/writes 发生的位置**。模型在遇到 entity 名词时，会激活那些承载这个 entity 相关知识的 slot。

---

## 这条路的 broader intuition

### Sparsity 是 continual learning 的第一性原理

forgetting 不是优化问题，是 parameter sharing 的几何问题。EWC 用 Fisher information 做 soft regularization，LoRA 用 low-rank subspace 限制 update 方向，效果都有限。**Hard sparse selection** 才能真正消除 interference——你不可能 forget 你没 update 的参数。

### 与 model editing 的关联

ROME、MEMIT、WISE 这些 model editing 工作也追求"局部化 update"。区别是它们做 one-shot edit，sparse memory finetuning 做 continual stream learning。Memory layer 提供了天然的"surgical instrument"。

### Parametric RAG

作者自己在 conclusion 里提到：fact learning 这个任务 RAG 现在就能解决。但 continual learning 真正想解决的是 reasoning、coding 这些 retrieval 困难的场景——你没法 retrieve 一段"如何 debug 这个 bug 的经验"。

Sparse memory finetuning 是一种 **parametric RAG**：把 experience 蒸馏进参数，但 surgical 到不会破坏其他能力。

### 与 biological memory 的类比

人脑有 sparse coding principle——每个概念由一小群神经元编码，不同概念之间几乎不重叠。Memory layer 在某种意义上 mimic 这种 prior：1M 个 slot 就像海马体的庞大但稀疏的细胞池，每条新记忆只动一小撮细胞。

---

## 局限与未解决问题

1. **只在 fact learning 上验证**——reasoning、coding 这些 distributed 的能力上效果未知
2. **需要从预训练就 swap FFN**——不能 retrofit 到现有 Llama、GPT
3. **Background corpus 选择需要先验**——DCLM 不一定是 post-training 后的好 proxy
4. **TF-IDF 是 static ranking**——不能 adapt 到 model 的 evolving state
5. **Top-$t$ 怎么随 task 复杂度调整**——目前手动调，1000 fact 用 500，document QA 用 10000
6. **Adam 与 sparsity 的 interaction**——作者发现 SGD 更好但没深挖，这里有大量研究空间
7. **Scale**——只测了 1.3B，10B+ 上是否能保持 Pareto 优势未知

---

## 我的 takeaway

这篇工作打动我的地方在于：它把 continual learning 这个老问题重新归约为一个 architectural 问题。Forget 不是因为算法不行，是因为我们的 architecture 强制 dense parameter sharing。一旦 architecture 提供 ultra-sparsity，catastrophic forgetting 几乎自动消失。

TF-IDF 这个 50 年前的检索 trick 在这里居然焕发第二春——它就是"识别局部信号"的最简单工具。

如果这条路 scale 上去（更大 memory pool、更多 memory layers、更聪明的 ranking），LLM 真有可能持续积累经验而越来越聪明，而不像现在这样 pretrain 完就冻结。

参考链接：
- Paper: https://arxiv.org/abs/2412.09764 (memory layers at scale)
- LoRA learns and forgets less: https://openreview.net/forum?id=aloEru2qCG
- EWC: https://www.pnas.org/doi/10.1073/pnas.1611835114
- WISE (lifelong model editing): https://arxiv.org/abs/2405.14768
- Product key memory: https://arxiv.org/abs/1907.05242
- Mixture of a million experts: https://arxiv.org/abs/2407.04153

---

# Continual Learning via Sparse Memory Finetuning 详细讲解

## 1. 核心动机与问题定义

Continual learning 的根本障碍是 **catastrophic forgetting**：更新参数学习新知识时，旧能力被覆盖。传统观点认为 replay-based methods (Robins, 1995; Scialom et al., 2022; Chen et al., 2025) 是黄金标准，但 replay 是 data-inefficient 且随着 experience 增长不可扩展。

作者抓住一个 fundamental insight：**catastrophic forgetting 的根源在于 trainable parameters 跨所有 task 共享**。如果每次 update 都动到所有参数，新 task 的 gradient 必然干扰旧 task 所依赖的参数。解决思路是从 parameter sharing 的对立面——**sparsity** 入手。

参考链接：
- Catastrophic forgetting 经典文献: https://www.sciencedirect.com/science/article/abs/pii/S0079742308604368 (McCloskey & Cohen 1989)
- Replay survey: https://arxiv.org/abs/2103.14877
- LoRA learns less and forgets less: https://openreview.net/forum?id=aloEru2qCG

---

## 2. Memory Layer 架构回顾

方法依赖 Berges et al. (2024) 和 He (2024) 提出的 memory layer。架构上，把 Transformer 中间某一层 FFN（实验中是 22 层模型中的第 12 层）替换成 memory lookup。

**前向计算公式**：

$$
\mathbb{I} = \mathrm{TopKIndices}(K q(x), k) \quad \text{\# retrieve top-k indices}
$$

$$
s = \mathrm{softmax}(K_{\mathbb{I}} q(x)) \quad \text{\# compute scores}
$$

$$
y = s V_{\mathbb{I}} \quad \text{\# compute weighted output}
$$

$$
\text{output} = (y \odot \mathrm{silu}(x^{\mathsf{T}} W_1))^{\mathsf{T}} W_2
$$

**变量逐项解释**：
- $x \in \mathbb{R}^n$：上一层输出（$n$ 是 model dim，实验中 n=2048）
- $q : \mathbb{R}^n \to \mathbb{R}^d$：query projection，把 $x$ 投影到 memory 的检索空间（$d$=1024 是 memory dim，比 model dim 小，这里 d ≠ n 是设计选择）
- $K \in \mathbb{R}^{N \times d}$：所有 keys，$N$ 是 memory pool 大小（1M）
- $V \in \mathbb{R}^{N \times d}$：所有 values，与 keys 维度一致
- $\mathbb{I}$：top-k indices 集合，$k=32$ per memory head
- $K_{\mathbb{I}} \in \mathbb{R}^{k \times d}$：被选中的 top-k keys 子集
- $s \in \mathbb{R}^k$：softmax 后的 attention scores
- $y \in \mathbb{R}^d$：values 加权和
- $W_1 \in \mathbb{R}^{n \times d}, W_2 \in \mathbb{R}^{d \times n}$：可选的 learned projection matrices
- $\mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x)$：SiLU 激活
- $\odot$：element-wise product；$y \odot \mathrm{silu}(\cdot)$ 是 input-dependent gating，让 input 决定 memory 贡献多少

**Product keys trick**（Lample et al., 2019）：把 $K$ 的每个 key 分成两半 $K^{(1)} \in \mathbb{R}^{N/2 \times d/2}, K^{(2)} \in \mathbb{R}^{N/2 \times d/2}$，top-k 可以用笛卡尔积高效检索 $N = (N/2)^2$ 个 keys，把 $O(N)$ 的检索降到 $O(\sqrt{N})$。这是 1M+ memory pool 在推理时可行的关键。

**与 MoE 的对比**：memory layer 可以看作"百万级 tiny experts"——每个 memory slot 是一个 expert，但只有 ~32 个被激活。而 MoE 通常 10-100 个 experts，每次激活几个大的 expert。memory layer 每个 token 激活参数仅占 ~0.03%~0.0002% of total memory params（32 × 1024 / 1M×1024 = 3.2e-5 = 0.0032% per head，4 heads 约 0.013%），而 MoE 一般每个 token 激活 1-2% of total experts。这种 ultra-sparsity 给 continual learning 提供了天然的"局部化"基础。

**实验配置的 active params 计算**：
- Memory layer: $k \cdot d = 32 \times 1024 = 32{,}768$ active params
- 原始 FFN: $n \cdot (4n) + 4n \cdot n = 2048 \times 8192 + 8192 \times 2048 \approx 50M$ params
- 即 memory layer 用 ~1/1500 的 active params 替换了原 FFN，但 total params 反而增加（1M × 1024 × 2 ≈ 2B，因为 keys + values）

参考链接：
- Memory layers at scale: https://arxiv.org/abs/2412.09764
- Mixture of a million experts: https://arxiv.org/abs/2407.04153
- Large memory layers with product keys: https://arxiv.org/abs/1907.05242

---

## 3. Sparse Memory Finetuning 方法

### 3.1 为什么 naive memory finetuning 还是会 forgetting？

直接 finetune 所有 accessed indices 仍然 catastrophic forgetting（见 Section 6 ablation）。原因：在某个 batch 上被 access 的 indices 中，**有些是 "general-purpose" indices**——比如负责 syntax、general world knowledge 的 slot，它们在 pretraining 时也被高频访问。直接 update 这些 slot 会污染其他 task 所依赖的参数。

Intuition 类比：你训练模型学"Michelle Smith-de Bruin 被禁赛 4 年"，这条事实激活了 ~1000 indices，但其中可能 800 个是负责"swimmer"、"banned"、"year" 这种 general semantics 的 slot，只有 ~100-200 个 slot 是真正"承载这条 fact 语义"的。如果你更新了那 800 个 general slots，下次模型遇到别的 swimmer 问题时表现就崩了。

### 3.2 TF-IDF Ranking 公式

借鉴信息检索里 TF-IDF 的思想：识别"在这个 batch 上高频、但在背景 corpus 上低频"的 indices——这些是 batch-specific 的、最适合承载新知识的 slot。

对每个 memory slot $i \in M$（$M$ 是所有 memory slots，大小 1M），TF-IDF 分数为：

$$
\mathrm{TF\text{-}IDF}(i) = \frac{c(i)}{\sum_{j \in M} c(j)} \cdot \log \frac{|B| + 1}{\sum_{b \in B} \mathbf{1}_{c_b(i) > 0} + 1}
$$

**变量解释**：
- $i$：某个 memory slot index，$i \in M$
- $c(i)$：slot $i$ 在当前 batch 上被 access 的次数（每个 token access $k \times \text{num\_heads}$ 个 slot，乘以 batch size × seqlen 总 token 数，所以总 access 数很大）
- $\sum_{j \in M} c(j)$：当前 batch 上所有 slot 的总 access 次数（归一化 TF 部分）
- $b$：某个 background batch，$b \in B$
- $B$：background batch 集合（实验中用 1000 个 DCLM random batches，Li et al., 2024）
- $c_b(i)$：slot $i$ 在 background batch $b$ 上的 access count
- $\mathbf{1}_{c_b(i) > 0}$：指示函数，slot $i$ 在 $b$ 上是否被 access（IDF 部分）
- $|B|$：background batch 总数（1000）
- "+1"：smoothing 防止 log(0) 或除零

**直觉化拆解**：
- **TF 部分** $\frac{c(i)}{\sum_j c(j)}$：slot $i$ 在当前 batch 上的"局部重要性"。被这个 batch 高频访问的 slot 分数高。
- **IDF 部分** $\log \frac{|B|+1}{\sum_b \mathbf{1}_{c_b(i)>0} + 1}$：slot $i$ 在 background corpus 上出现的"稀有度"。在 background 上从未出现过的 slot 分数高；在所有 background batch 上都出现（general slot）则接近 0。
- 两者相乘：在当前 batch 上高频且在 background 上稀有的 slot——正是"这条新 fact 独占的 slot"。

### 3.3 实现细节：动态 trainable mask

每个 forward pass 上 top-$t$ indices 是动态变化的（不同 batch 选不同 indices），所以不能简单 freeze 参数。作者用了一个 trick：

```python
# trainable_mask: shape (memory_size, 1), 1 if trainable else 0
# mem: shape (memory_size, value_dim)

# Forward value is unchanged, but gradient only flows through trainable positions
mem = mem * trainable_mask + mem.detach() - (mem * trainable_mask).detach()
```

**拆解这段代码**（这是 PyTorch 技巧）：
- 前向值：`mem * trainable_mask` (trainable 部分) + `mem.detach()` (全部 detach 版本) - `(mem * trainable_mask).detach()` (detach 后 trainable 部分)
  = `mem * trainable_mask` + `mem.detach() - mem.detach() * trainable_mask`
  = `mem * trainable_mask + mem.detach() * (1 - trainable_mask)`
  即 trainable 位置走 `mem`（带 grad），其他位置走 `mem.detach()`（无 grad）
- 反向梯度：对 trainable 位置 $\frac{\partial}{\partial \text{mem}} = \text{trainable\_mask}$，对其他位置为 0

这样既保持前向输出完全一致，又只让 top-$t$ slots 接收梯度。

### 3.4 训练设置

实验配置表：

| 配置项 | 值 |
|---|---|
| Base model | 1.3B（统一预训练） |
| Memory pool size | 1M |
| $k$ (top-k per head per token) | 32 |
| Memory heads | 4 |
| Value dimension | 1024 |
| Active params in memory layer | 32,768 |
| Memory layer position | layer 12 of 22 |
| Top-$t$ trainable slots | 500 (TriviaQA) / 10000 (SimpleQA) |
| Background corpus | 1000 random DCLM batches |
| Batch size | 64 |
| Seq len | 64 (TQA) / 512 (SimpleQA) |
| Optimizer (sparse) | SGD |
| Optimizer (baselines) | AdamW, $\lambda=0.1$ |

**优化器的关键选择**：作者发现 AdamW 与 sparsity 有 "unexpected interactions"——adaptive per-parameter step sizes 会给稀疏更新的参数施加不一致的步长（每个 slot 的二阶动量历史不同），weight decay 和 momentum 也会污染稀疏梯度信号。改用 SGD 之后 forgetting 进一步降低（虽然 baselines 反而不见改善）。这与 Hsu et al. (2019) 的观察一致——Adam 对 continual learning 不友好。

参考链接：
- DCLM benchmark: https://arxiv.org/abs/2406.11794
- Re-evaluating continual learning scenarios: https://arxiv.org/abs/1810.12488
- Elastic Weight Consolidation: https://www.pnas.org/doi/10.1073/pnas.1611835114

---

## 4. 实验结果与技术解读

### 4.1 Fact Learning (小数据 regime)

设置：模型按顺序学 1000 条 TriviaQA facts。每条 fact 重写为 statement，再用 paraphrase 把 batch 填到 size 64。这种小数据 + narrow domain 正是 catastrophic forgetting 最严重的场景。

**关键数字**：
- **NaturalQuestions F1 drop**：
  - Full finetuning: **89% drop**
  - LoRA: **71% drop**
  - Sparse memory finetuning: **11% drop**（同样学习水平的 TQA F1）
- 同时 TQA target F1 三者持平或 sparse memory 略高

**为什么 sparse memory 在小数据 regime 优势更明显**？小数据下：
1. Gradient 噪声大，full finetuning 更容易把 noise 写进所有参数
2. LoRA 虽然 rank 受限，但仍然影响所有 attention 和 FFN matrices（low-rank 但 globally distributed），而 narrow domain 下 general-purpose params 比 capacity 更宝贵
3. Sparse memory 把更新严格局限在 batch-specific 的几百个 slot，general slots 完全不动，所以即使 1000 步后 general capabilities 几乎无损

### 4.2 Document QA

设置：100 个 SimpleQA 问题对应的 Wikipedia 文档，切成 1824 个 chunks。每个 chunk 用 Active Reading (Lin et al., 2025) 生成 N 个 synthetic augmentations 填 batch。Sequential 暴露 chunks（不是 iid shuffle）。Top-$t$ 设为 10000（因为每 batch 信息量更高）。

**结果**：full finetuning 和 LoRA 在 target task 上比 fact learning setting 表现更好（数据更 diverse 接近 iid pretraining），但 held-out 仍然 forgetting。Sparse memory finetuning 在高低 lr 下都 Pareto dominate。

### 4.3 Pareto Frontier 分析（Figure 5）

作者 sweep 了每个方法的主要 hyperparameters：
- Full finetuning: lr $\in \{2e{-}6, 5e{-}6, 2e{-}5, 5e{-}5\}$
- LoRA: rank $\in \{32, 128, 256\}$, alpha $\in \{1/2, 1, 2, 4\} \times$ rank, lr $\in \{2e{-}4, 5e{-}5, 5e{-}6\}$
- Sparse memory: $t \in \{25, 50, 100, 200, 500, 1000\}$, lr $\in \{0.1, 2\}$

观察到的趋势：
1. 各方法都存在 "learning capacity ↑ → forgetting ↑" 的连续趋势
2. 当 learning capacity 过高（如 full ft lr > 5e-6），模型 "break"——learning 和 forgetting 同时退化
3. **Sparse memory finetuning 在 Pareto frontier 上明显外推**：高 learning capacity 时 forgetting 仍然 minimal

**点的 size 表示 trainable params per batch**：sparse memory 的点 size 很小（500 slots × 1024 dim ≈ 0.5M params/batch），但 learning capacity 超过 LoRA（rank 256, alpha 1024, 跨所有 linear ≈ 数十 M params/batch）和 full finetuning（1.3B params/batch）。

### 4.4 Naive Memory Finetuning Ablation（Figure 6）

| 方法 | Target F1 | Held-out performance |
|---|---|---|
| Finetune all accessed indices | 与 TF-IDF 相近 | 显著 worse |
| TF-only ranking (top-$t$) | 与 TF-IDF 相近 (t>500) | 比 TF-IDF worse |
| TF-IDF ranking (top-$t$) | best | best |
| Full finetuning memory model | — | 最差 (GSM8K NLL=3.87) |

**关键观察**：
- 当 $t$ 较大时（如 500+），TF-only 和 TF-IDF 差距变小——因为两者都收敛到"几乎 finetune 所有 accessed indices"
- 当 $t$ 小时（如 50），TF-IDF 显著优于 TF-only——IDF 部分对识别 "critical minimal set" 至关重要

### 4.5 Background Corpus 选择（Figure 7）

比较三种 background：
1. DCLM (general pretraining proxy) → learning 最好，forgetting 较少
2. TriviaQA 训练集 → learning 持平，forgetting 显著 worse（因为没下权重 general slots）
3. NaturalQuestions (held-out set) → 与 DCLM 类似，因为 NQ indices 倾向于跨多 NQ 问题共享，等价于识别"domain-shared" slots

**Insight**：background corpus 不需要精确——只要能近似 pretraining 的 general index 分布即可。如果你想保护特定 held-out domain，用该 domain 的 indices 做 background 也行（但效果与 DCLM 类似）。

### 4.6 Memory Access 分析（Table 1）

定义 **core set** = 所有 paraphrases 与 question 共享的 indices（即"承载这条 fact 语义"的 indices）。

发现：
- Core set 大小典型 100-500 indices（不是单一 slot 存一条 fact，而是 distributed representation）
- 但实际只需 finetune $t=25\sim100$ indices 就能让模型答对——远小于 core set
- TF-IDF 选出的 trainable indices 与 core set 高度重合
- 这些 indices 往往 **align with entity boundaries**（如 "Michelle Smith-de Bruin"、"Walter Hagen"、"Vienna"）——暗示 entity boundary tokens 是 critical parametric memory reads/writes 的位置

---

## 5. 联想与 broader intuition

### 5.1 Sparsity 作为 continual learning 的第一性原理

这篇工作实证了一个直觉：**catastrophic forgetting 不是优化算法问题，而是 parameter sharing 的几何问题**。EWC (Kirkpatrick et al., 2017) 用 Fisher information matrix 做 soft regularization——保留"重要"参数；LoRA 用 low-rank subspace 限制 update 方向；sparse memory finetuning 直接做 hard sparse selection。

三者的 spectrum：
- EWC: dense update + soft penalty → 难以彻底避免 interference
- LoRA: dense-ish update (low-rank 但仍 globally distributed) → 减小 update 范围但仍然 touching 所有 layers
- Sparse memory: hard sparse update on minimal indices → 几乎完全消除 interference

**为什么 hard sparsity 比 soft regularization 强**？因为 catastrophic forgetting 不是 "权重动太多" 的问题，而是"动到了不该动的具体位置"的问题。Soft penalty 仍然会让所有参数都有小幅 update，累积起来仍然会破坏 delicate learned representations。

### 5.2 与 model editing 工作的关联

WISE (Wang et al., 2024)、MEMIT、ROME 等模型编辑工作也追求"局部化" update。区别：
- Model editing 是 one-shot edit（修改特定 fact）
- Sparse memory finetuning 是 continual stream learning（持续吸收新知识）

Memory layer 提供了一个天然的"surgical instrument"——它的 sparsity 不是人为施加的（如 mask gradient），而是 architecture-inherent。这种 architecture-level sparsity 比 algorithmic sparsity 更稳定。

参考链接：
- WISE: https://arxiv.org/abs/2405.14768
- ROME: https://arxiv.org/abs/2202.05262
- MEMIT: https://arxiv.org/abs/2210.07229

### 5.3 与 RAG 的对比（作者自己提到）

作者在 conclusion 里指出：fact learning 这种任务 RAG 是 present-day solution。但 continual learning 的真正目标在 reasoning、coding 这种 retrieval 困难的领域——模型需要把 experience **distill** 进参数，而不只是 retrieve。

这个 framing 很重要：sparse memory finetuning 提供了一种 "**parametric RAG**" 的中间路线——参数化的、surgical 的、保留原有能力的 knowledge injection。

### 5.4 为什么 SGD 比 AdamW 对 sparse updates 更好？

深入推测：
1. AdamW 的二阶动量 $v_t$ 是 per-parameter 的 EMA of squared gradients。对于 sparse updates，被选中的 slot 在被选中的 batch 上有梯度，但其他 batch 上没梯度。Adam 会累积"稀疏的"二阶动量历史，对每个 slot 给出不一致的 effective learning rate。
2. AdamW 的 weight decay 施加在所有 params 上（包括 frozen slots 在 backward 时其实不接收，但 optimizer state 上仍有），引入了与 sparse gradient 不相关的 drift
3. SGD 给所有更新 params 统一的步长 $\eta$，更"诚实"地反映 gradient 信号

这与 sparse training literature 的发现一致：rigL (Evci et al., 2022) 等也发现 Adam 在 sparse training 上需要特殊处理。

参考链接：
- RigL: https://arxiv.org/abs/1911.11134
- Sparse training survey: https://arxiv.org/abs/2202.00599

### 5.5 可能的扩展方向

1. **Input-dependent $t$**：现在的 $t$ 是固定的，可以根据 batch 复杂度动态调整
2. **Sequence-level ranking**：现在按 batch 排序，可以按 sequence 排
3. **More sophisticated scoring**：TF-IDF 是简单选择，可以用 learned ranker
4. **Scale**：1.3B → 10B+ 的验证
5. **Beyond fact learning**：reasoning、coding、agent learning 场景
6. **Multiple memory layers**：现在只 swap 一个 FFN，可以更多
7. **Continual learning across modalities**：vision-language 的 continual learning

### 5.6 与 Progressive Neural Networks、Adapter 的对比

历史脉络：
- Progressive Neural Networks (Rusu et al., 2016)：每新 task 加一列全新 network，零 interference 但参数线性增长
- Adapter (Houlsby et al., 2019)：每 task 加 small adapter，参数量受控但仍 dense
- LoRA (Hu et al., 2021)：low-rank adapter，更参数高效但 Biderman et al. 2024 实证 learns less and forgets less
- MoE 扩展 (Gritsch et al., 2024)：每 task 加 expert，但 expert 数量受限
- **Sparse memory finetuning**：用 1M+ 大 memory pool 作为"持续可扩展"的参数储备，每次只更新 minimal subset，既不线性增长参数，又避免 interference

这是参数效率与遗忘 avoidance 的 Pareto 改进。

参考链接：
- Progressive Neural Networks: https://arxiv.org/abs/1606.04671
- Adapter: https://arxiv.org/abs/1902.00751
- LoRA: https://arxiv.org/abs/2106.09685
- Nexus (MoE continual): https://arxiv.org/abs/2408.15901

---

## 6. Method 局限性与批判性思考

1. **仅测 fact learning**：在 reasoning、in-context learning、long-context tasks 上的效果未验证。Fact learning 是相对"localized"的知识，可能 sparse updates 足够；但 reasoning ability 可能更 distributed，sparse updates 可能 learn 不够
2. **Background corpus 选择需要先验知识**：用 DCLM 作 background 需要知道"什么是 pretraining-like data"。如果 model 已经经过很多轮 post-training，DCLM 不再是好的 background proxy
3. **TF-IDF 是 static ranking**：pretraining 时一次性计算 IDF，不能 adapt 到 model 的 evolving state
4. **Memory layer 是 architecture modification**：需要在 pretraining 时就 swap FFN，不能 retrofit 到现有 LLM（如 Llama、GPT）。这限制了 adoption
5. **Active params 比 dense FFN 少很多**（32K vs 50M），可能影响原 model 的某些能力——但作者没报告 swap FFN 本身对 base capabilities 的影响
6. **Top-$t$ 与 task 复杂度的关系**：t=500 对 single fact，t=10000 对 document QA。如何对 unseen task 自动选 t 是 open problem
7. **SGD 比 AdamW 好**这一点暗示 optimizer 与 sparsity 的 interaction 还需要大量研究

---

## 7. 总结直觉

这篇工作用一句话总结：**catastrophic forgetting 的解药是 architectural sparsity + surgical parameter updates**。

更深层的直觉：continual learning 之所以难，是因为我们默认了 "dense parameter sharing" 这个 architectural prior。一旦 architecture 本身就把 knowledge 局部化到 sparse slots（每个 fact 只激活极少 slot），forgetting 就从根本问题上消失了——你不可能 forget 你没 update 的参数。

Memory layer 提供了这种 prior：1M+ slots 作为"知识容器"，每个 token 只激活 32 个。TF-IDF ranking 进一步告诉我们：在这 32 个 activated slot 里，可能只有 5-10 个是"这条 fact 独占的"，剩下的是 general-purpose。精准定位那 5-10 个 slot，做 surgical update，就能在不破坏其他能力的前提下吸收新知识。

这条路径如果 scale 上去（更大 memory pool、更多 memory layers、更智能的 ranking），可能真的能让 LLM 像 biological memory systems 一样持续积累 experience 而不遗忘。

参考链接（汇总）：
- Paper: https://arxiv.org/abs/2412.09764 (memory layers at scale, Berges et al. 2024)
- Active Reading: https://arxiv.org/abs/2508.09494
- SimpleQA: https://arxiv.org/abs/2411.04368
- TriviaQA: https://arxiv.org/abs/1705.03547
- NaturalQuestions: https://aclanthology.org/D19-5731/
- HellaSwag: https://aclanthology.org/P19-1039/
- EWC: https://www.pnas.org/doi/10.1073/pnas.1611835114
- LoRA learns and forgets less: https://openreview.net/forum?id=aloEru2qCG
- Catastrophic forgetting: https://www.sciencedirect.com/science/article/abs/pii/S0079742308604368
