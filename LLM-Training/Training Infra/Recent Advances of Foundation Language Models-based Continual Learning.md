---
source_pdf: Recent Advances of Foundation Language Models-based Continual Learning.pdf
paper_sha256: 93d1fd318888cf1d76864b434837f3cee694fc3ad52cc3700069e265f81e12bb
processed_at: '2026-08-11T21:44:39-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Survey

## 这篇 Paper 在干嘛

这群人把现在所有关于 "大模型怎么持续学习" 的研究都翻了一遍，做了个分类整理。核心问题就一句话：**模型学新东西会把旧东西忘掉**，这叫 catastrophic forgetting，怎么破。

## 为什么 Foundation LMs 时代的 CL 和以前不一样

以前的 CL 研究 都是拿小模型 (ResNet, LSTM) 从头训练，关注的是 "学 task B 别把 task A 忘了"。但到了 foundation LMs 时代，情况变了：

你手头是一个已经 pre-train 好的大家伙 (BERT, GPT, LLaMA, CLIP)，它有 zero-shot 能力，有 general world knowledge，有 instruction following 能力。现在你要 fine-tune 它做新 task，问题变成了三层：

1. 新 task 要学好
2. 旧 task 别忘
3. **预训练时的 general capability 也别丢**

第三层是以前没有的。你 fine-tune LLaMA 做了 medical QA，结果它不会写代码了，不会做数学了，safety alignment 也崩了——这在实际部署中是大问题。

所以传统 CL 那套 EWC、replay 的思路直接搬过来不够用，得重新想。

## 三个不同的 CL 难度等级

这个分类很关键，理解了就知道为什么有些方法只在特定场景 work：

### Domain-Incremental (DIL) - 最简单
你有个 sentiment analysis 模型，先在 restaurant reviews 上训，再在 movie reviews 上训，再在 laptop reviews 上训。Label 空间都是 positive/negative/neutral，只是 input distribution 变了。模型不需要知道现在在哪个 domain，推理时直接出结果就行。

### Task-Incremental (TIL) - 中等
每个 task 是不同的事情，比如先是 text classification，然后是 QA，然后是 translation。但好处是推理时你知道当前是哪个 task，可以路由到对应的模块。

### Class-Incremental (CIL) - 最难
先是识别 cat/dog，然后识别 bird/fish，然后识别 car/plane。推理时给你一张图，你得能分到所有学过的 class 里，但不知道这张图属于哪个 task。这就很难了，因为新 class 会和旧 class 搞混，而且 classifier head 要不断扩展。

## 四大类方法，对应不同的思路

### Traditional Methods - 老办法硬套
Replay (存点旧数据回放)、Regularization (限制重要参数别动)、Parameter-isolation (给每个 task 分配独立参数)。这些从传统 CL 直接搬过来，LAMOL 就是典型的 generative replay——用模型自己生成旧 task 的伪样本来回放。

### Continual Pre-training - 继续无监督预训练
不是 fine-tune 下游 task，而是在新 domain 的 raw corpus 上继续做 MLM 或 next token prediction。ELLE, DAS, EcomGPT-CT 都是这路子。挑战是 pre-train 时怎么不把 old knowledge 冲刷掉。

### Parameter-Efficient Tuning - 只动一点点参数
这在大模型时代最流行。核心想法：freeze backbone，只 train 一小部分参数，天然就防止 forgetting。LoRA, Adapter, Prompt tuning 都是这路子。O-LoRA 更进一步，让每个 task 的 LoRA 子空间正交，互不干扰。

### Instruction Tuning - 把 task 变成指令
这是 LLM 时代独有的。把所有 task 统一成 instruction format，比如 "Classify the sentiment of this review: ..."。好处是不同 task 之间可以 forward transfer，因为 instruction representation 有共享。Progressive Prompts (PP), ConTinTin 是代表。

## 几个有意思的架构设计

### PlugLM - 外挂 key-value memory
想法很直接：把知识存到 external key-value store 里，而不是塞进 model weights。新 domain 来了就往 store 里加新的 key-value pair，旧的不动。推理时 query 和 key 做 attention 检索相关 value。这个思路其实是 differentiable version of retrieval。

### Lifelong-MoE - 不断加 expert
Mixture-of-Experts 架构，每来一个新 task 就新增一个 expert，旧的 expert 全部 freeze。Gating network 也冻结，只 train 新 expert 的 gating。capacity 可以无限扩展，代价是 model size 线性增长。

### S-Prompts - 每个 domain 一个 prompt
CLIP backbone 完全 freeze，给每个 domain 学一个 prompt。推理时用 k-NN 在所有 domain prompt 里找最匹配的，再拼到 input 里。简单粗暴但 work，因为 CLIP 的 zero-shot 能力太强，只要给个 domain-specific hint 就行。

### TRIPLET - 多模态 prompt 解耦
VLM 的 CL 难在 vision 和 language 两个 modality 都要照顾。TRIPLET 给 vision, question, fusion 三种 modality 各自一套 prompt，而且分 General Prompt (跨 task 共享) 和 Expert Prompt (task-specific)。这样 task-invariant 知识和 task-specific 知识就分开了。

## 评估指标的核心 logic

CL 的评估比普通 ML 复杂，因为要同时看 "学新" 和 "保旧"。

想象一个 $N \times N$ 的矩阵 $R$，$R_{i,j}$ 表示训练完 task $i$ 后在 task $j$ 上的 test accuracy。

- **Last**: 看最后一行，训练完所有 task 后每个 task 还剩多少。这是最直接的 "最终能力" 指标。
- **BWT (Backward Transfer)**: 比较对角线 $R_{i,i}$ (刚学完时的 peak) 和最后一行 $R_{N,i}$ (最终)。差值为负就是 forgetting，正数说明学新 task 居然帮了旧 task (很少见)。
- **FWT (Forward Transfer)**: 看上三角 $R_{j,i}$ ($j < i$)，即学完旧 task 后在新 task 上的 zero-shot 表现，减去完全没训练时的 baseline。衡量 prior knowledge 帮了多少。
- **FM (Forgetting Measure)**: 对每个 task，找历史上最高点和最终点的最大 gap，然后平均。比 BWT 更严格，因为 BWT 只看对角线。

关键 insight：**多数方法的 BWT 都是负的**，真正 backward transfer (学新提升旧) 几乎没人做到。这说明 CL 的圣杯还是 "防 forgetting"，forward transfer 相对容易，backward transfer 基本是 holy grail。

## 我的几个 Take-away

**1. Foundation LMs 的 CL 本质是 capacity allocation 问题**

大模型参数多，redundancy 高，其实可以容纳很多 task 的知识。问题是怎么把这些知识 modulize，不让它们互相干扰。LoRA 系列方法 work 就是因为每个 task 有独立的低秩子空间，backbone 保持不动。这比 replay 和 regularization 更适合大模型。

**2. Zero-shot capability 是新维度的 "forgetting"**

传统 CL 不关心这个，但 foundation LMs 的核心价值就是 zero-shot。ZSCL, SND 这些工作专门研究 CL 后 zero-shot 退化问题。实践证明，即使 task-specific accuracy 看起来没掉，zero-shot generalization 可能已经崩了。这是评估时容易漏的点。

**3. Instruction tuning 改变了 CL 的 game**

传统 CL 每个 task 是独立的，cross-task transfer 很难。但 instruction format 让所有 task 在同一个 representation space 里，相似 task 自然就 transfer 了。而且 few-shot in-context learning 可以部分替代 parameter update，从根上避免 forgetting。这条路在 LLM 时代才走通。

**4. VLM 的 CL 比纯 NLP 难**

两个 modality 的 alignment 本身就脆弱，fine-tune 一个 modality 会破坏 alignment。而且 CLIP 的 zero-shot 是核心卖点，不能丢。所以 VLM CL 几乎都是 freeze backbone + light adapter/prompt 的范式，没人敢全量 fine-tune。

**5. 真实世界的 CL 还很远**

现在所有方法都假设 task boundary 清晰 (offline CL) 或至少有 task identity (online HTB)。真实世界是数据流不断来，distribution 慢慢漂移，没有明确的 task 切分。Blurry Task Boundary 设置更接近现实但研究很少。Autonomous CL (模型自己 detect distribution shift 并适应) 是未来大方向。

## 一句话总结

Foundation LMs 时代的 CL 从 "防止参数漂移" 变成了 "capacity modularization + knowledge routing"——你不再想办法让一组 weights 同时记住所有 task，而是想办法让不同 task 的知识住在不同的子空间里，推理时按需调用。LoRA variants + instruction tuning + retrieval 这个组合是当前最有希望的方向。

---

# Foundation Language Models-based Continual Learning 综述深度讲解

## 1. Paper 整体定位与动机

这篇 survey 来自 East China Normal University 的 Liang He 团队 (arXiv 链接 https://arxiv.org/abs/2503.06263 或 ACM 链接)，系统梳理了 Continual Learning (CL) 与 Foundation Language Models (Foundation LMs) 交叉领域的进展。核心 motivation 在于：传统 CL 研究 focused on 小规模 neural networks (ResNet、LSTM 等)，而 foundation LMs 具备几个根本性差异——巨大的参数量、strong zero-shot transfer、instruction following 能力，因此需要重新设计 CL 方法论。

从图 1 (Figure 1) 可以看到，传统 CL 和 Foundation LMs-based CL 的本质差异：

- **传统 CL**: 小模型，从头训练，主要解决 classification
- **Foundation LMs-based CL**: 巨型 pre-trained models，需要同时保留 zero-shot 能力、history task 能力，同时学新 skills

这种转变带来的关键 insight 是：CL 不再仅仅是 "防 forgetting"，而需要 maintain pre-trained model 的 general capability，这给 regularization-based 方法提出了新挑战——over-regularize 会损失 zero-shot 能力，under-regularize 会导致 catastrophic forgetting。

---

## 2. Taxonomy 详解 (Section 4)

paper 提出了一个 2D 的分类体系：

### 2.1 Learning Mode 维度 (Offline vs Online)

**Offline CL** 的三个 sub-setting (Figure 3)：

| Setting | 数据分布 | Label 空间 | Task ID (train/test) | 代表方法 |
|---------|---------|-----------|---------------------|---------|
| Domain-Incremental (DIL) | $p(X_t) \neq p(X_{t'})$ | 相同 | 不需要 | LAMOL, S-Prompt |
| Task-Incremental (TIL) | 任意 | 可重可异 | train 和 test 均提供 | PP, ConPET |
| Class-Incremental (CIL) | 任意 | $C_t \neq C_{t'}$ | 仅 train 时有 | ExtendNER, EPI |

这里关键 insight 是 DIL 最容易（label space 一致，只需适应 input distribution shift），CIL 最难（需要区分新旧 class 但没有 task ID 来路由）。

**Online CL** 的两个 sub-setting (Figure 4)：

- **Hard Task Boundary (HTB)**: 任务边界清晰，前一个任务数据完全消化完才进入下一个
- **Blurry Task Boundary (BTB)**: 任务数据混合，无法明确划分边界，更接近真实世界

### 2.2 Method 维度 (4 大类)

| 方法类别 | 核心思想 | 代表方法 | 适用场景 |
|---------|---------|---------|---------|
| Traditional | Replay / Regularization / Parameter-isolation | LAMOL, EWC, DEMIX | 直接迁移传统 CL 技术 |
| Continual Pre-training | 顺序 pre-train 在新 domain corpus | DAS, ELLE, EcomGPT-CT | Domain adaptation |
| Parameter-Efficient Tuning | Adapter / Prompt / LoRA 等微调少量参数 | AdapterCL, O-LoRA, MoE-Adapters4CL | 大模型场景 |
| Instruction Tuning | 任务转为 instruction 形式 | PP, ConTinTin, DYNAINST | 利用 instruction-following |

---

## 3. 基础公式与 Notation (Section 4.1)

CL 的基本 setup 定义如下：

给定 task sequence $T = \{1, 2, ..., N\}$，每个 task $t$ 对应 dataset：

$$X_t = \{(x_i^{(t)}, y_i^{(t)})\}_{i=1}^{|X_t|}$$

其中：
- $x_i^{(t)}$：第 $t$ 个 task 的第 $i$ 个 training example
- $y_i^{(t)}$：对应的 label
- $|X_t|$：task $t$ 的样本总数
- 上标 $(t)$：表示 task index
- 下标 $i$：表示样本 index

关键约束是：对于任意 $t \neq t'$，$p(X_t) \neq p(X_{t'})$，即不同 task 的 data distribution 不同。这是 catastrophic forgetting 的根源。

---

## 4. 代表性方法架构解析

### 4.1 PlugLM (PLM-based DIL) - Figure 5a

PlugLM 的核心创新是 **Differentiable Plug-in Memory (DPM)**，将知识存储与 model parameters 解耦。DPM 是一个 triplet $(D, K, V)$：

- $D$：gating function 计算 attention distribution
- $K = [k_1, k_2, ..., k_n]$：key vectors
- $V$：对应的 value vectors

公式：
$$v = \sum_k \alpha_k v_k, \quad \alpha_k = \text{softmax}(\text{KnowEncoder}(x) \cdot k_k)$$

其中 $\alpha_k$ 是 key-query matching weight。这种设计让新 domain 的知识可以增量插入而不干扰旧 domain 的 weights，类似于 external key-value store 但 differentiable。

### 4.2 Lifelong-MoE (LLM-based DIL) - Figure 5b

基于 Mixture-of-Experts (MoE) 架构：

$$\text{output} = \sum_{i=1}^{E} g_i(x) \cdot E_i(x)$$

其中 $g_i(x)$ 是 gating weight，$E_i(x)$ 是第 $i$ 个 expert 的输出。Lifelong-MoE 的关键：
- 新 task 时新增 expert $E_{new}$
- 冻结 prior experts $E_1, ..., E_{n-1}$ 和它们的 gating
- 保留 shared dense layers $\theta_d$ 用于 cross-task transfer

这种设计让 capacity 可扩展，但 model size 也会线性增长。

### 4.3 S-Prompts (VLM-based DIL) - Figure 5c

基于预训练 CLIP 的 prompt tuning：
- 为每个 domain 独立学习 prompt $P_d$
- 推理时用 k-NN 在 prompt pool 中检索最匹配的 domain prompt
- Backbone CLIP 完全冻结

核心 loss：
$$\mathcal{L} = -\sum_{(x, y)} \log p(y | x, P_d; \theta_{\text{CLIP}})$$

只有 $P_d$ 是 trainable，$\theta_{\text{CLIP}}$ frozen。

### 4.4 HMI (PLM-based TIL) - Figure 6a

Hippocampal Memory Indexing inspired 方法，包含：
- **Memory Module**: 存储压缩的 prior training instances
- **Generation Module**: 基于存储的 instances 生成 pseudo-samples
- **Main Model**: 在新 task 训练时 mixing real samples 和 generated pseudo-samples

这种 generative replay 思路解决了 experience replay 需要 store raw data 的隐私和存储问题。

### 4.5 DynaMind (LLM-based TIL) - Figure 6b

三个核心模块：
1. **Memory Module**: 存储 learned knowledge
2. **Modular Operator**: 处理 incoming data
3. **CL Module**: 动态调整 LLM 参数

架构图显示它结合了 retrieval 和 parametric update，类似 RAG + fine-tuning 的混合范式。

### 4.6 TRIPLET (VLM-based TIL) - Figure 6c

**Decoupled Prompts** 设计：
$$P^{(m)} = \{G^{(m)}\} \cup \{E^{(m)}_{t,k}\}$$

其中：
- $m \in \{v, q, f\}$：vision, question, fusion 三种 modality
- $G^{(m)} \in \mathbb{R}^{L_G \times D}$：General Prompt (layer-wise shared)
- $E^{(m)}_{t,k} \in \mathbb{R}^{L_E \times D}$：Expert Prompt (task $t$, layer $k$ specific)
- $L_G, L_E$：prompt length
- $D$：hidden dimension

公式 (4) 描述 prompt 结构：
$$T([P; x]) = (L_K \circ L_{K-1} \cdots L_0)([P; x])$$

Transformer $T$ 含 $K$ 层，每层处理 concatenation $[P; x]$。这种设计将 task-invariant 知识 (G-Prompt) 与 task-specific 知识 (E-Prompt) 分离，减少 interference。

### 4.7 ExtendNER (PLM-based CIL) - Figure 7a

针对 Named Entity Recognition 的 CIL：
- Teacher NER model 教 Student model
- Student 学新 entity types 同时保留旧 type 知识
- Loss：
$$\mathcal{L}_{\text{ExCE}} = \text{CE}(y, p^{i+1}_{E_{\text{new}}})$$
$$\mathcal{L} = \alpha \mathcal{L}_{\text{KL}} + \beta \mathcal{L}_{\text{CE}}$$

其中 $\alpha, \beta$ 是平衡系数，$\mathcal{L}_{\text{KL}}$ 是 teacher-student 间的 KL divergence，$\mathcal{L}_{\text{CE}}$ 是 student 在新 class 上的 cross-entropy。

### 4.8 Adaptation-CLIP (VLM-based CIL) - Figure 7c

三种策略组合：
1. **Linear Adapter**: 在 image encoder 后加线性层
2. **Self-attention Adapter**: 加自注意力机制
3. **Prompt Tuning**: 文本端加 prompt

公式：
$$Q = W_q I_i, \quad K = W_k I_i, \quad V = W_v I_i$$
$$A_i = \alpha V, \quad \alpha = \text{softmax}(Q K^T / \sqrt{d})$$

其中 $I_i$ 是 CLIP 第 $i$ 层输出，$W_q, W_k, W_v$ 是可学习投影矩阵，$\alpha$ 是 attention weight。Backbone frozen，只 train adapter。

---

## 5. 评估指标深度解析 (Section 8)

CL 评估有三大类指标，这里详细推导：

### 5.1 Overall Performance

**Last** (公式 1)：模型训练完所有 $N$ 个 task 后在所有 task 上的平均 accuracy：

$$\text{Last} = \frac{1}{N} \sum_{i=1}^{N} R_{N,i}$$

其中 $R \in \mathbb{R}^{N \times N}$ 是 performance matrix，$R_{i,j}$ 表示训练完 task $i$ 后在 task $j$ 上的 test accuracy。$R_{N,i}$ 即最后一行第 $i$ 列，表示最终模型在第 $i$ 个 task 上的表现。

**Avg** (公式 2)：考虑所有时间点的整体平均：

$$\text{Avg} = \frac{1}{N} \sum_{i=1}^{N} \left(\frac{1}{N} \sum_{j=1}^{N} R_{i,j}\right)$$

这个指标对训练过程中的波动更敏感。

**AIA (Average Incremental Accuracy)** (公式 3)：只取下三角部分（已学过的 task）：

$$\text{AIA} = \frac{1}{N} \sum_{i=1}^{N} \left(\frac{1}{i} \sum_{j=1}^{i} R_{i,j}\right)$$

这个指标评估 "如果我在 task $i$ 训练完后立即部署" 的平均表现。

**Transfer** (公式 4)：评估 zero-shot transfer 的保持：

$$\text{Transfer} = \frac{1}{N-1} \sum_{i=2}^{N} \left(\frac{1}{i-1} \sum_{j=1}^{i-1} R_{j,i}\right)$$

$R_{j,i}$ 表示训练完 task $j$ 后在 task $i$ ($i > j$) 上的 accuracy，即 forward transfer 能力。除以 $(i-1)$ 是因为只考虑 $j < i$ 的项。

### 5.2 Memory Stability

**BWT (Backward Transfer)** (公式 5)：

$$\text{BWT} = \frac{1}{N-1} \sum_{i=1}^{N-1} (R_{N,i} - R_{i,i})$$

其中 $R_{i,i}$ 是刚学完 task $i$ 时在该 task 上的 accuracy (最佳点)，$R_{N,i}$ 是最终模型在 task $i$ 上的 accuracy。差值为负表示 forgetting，正数表示 backward transfer (后续学习反而提升了旧 task)。除以 $(N-1)$ 是因为 task $N$ 本身不需要评估 backward。

**FM (Forgetting Measure)** (公式 9-10)：

对 task $j$ 的 forgetting：
$$f_j = \max_{l \in \{1,...,N-1\}} (R_{l,j} - R_{N,j}), \quad \forall j < N$$

取所有时间点最高 performance $R_{l,j}$ 与最终 $R_{N,j}$ 的最大差距。然后平均：

$$\text{FM} = \frac{1}{N-1} \sum_{j=1}^{N-1} f_j$$

FM 越低越好。

### 5.3 Learning Plasticity

**FWT (Forward Transfer)** (公式 6)：

$$\text{FWT} = \frac{1}{N-1} \sum_{i=2}^{N} (R_{i-1,i} - R_{0,i})$$

其中 $R_{0,i}$ 是在 task $i$ 上没有任何相关训练时的 baseline performance (例如 random initialization 的 zero-shot)。$R_{i-1,i}$ 是训练完 task $i-1$ 后在 task $i$ 上的表现。正值表示 prior learning 帮助了后续 task。

**IM (Intransigence Measure)** (公式 14)：

$$\text{IM} = R_N^* - R_{N,N}$$

$R_N^*$ 是 task $N$ 在 joint training (oracle) 下的 accuracy，$R_{N,N}$ 是 sequential training 下 task $N$ 的最终 accuracy。差值衡量 sequential learning 损失了多少 task $N$ 的学习能力。

### 5.4 Continual Pre-training 专用

**FUAR** (公式 15-17)：

$$\text{Eq}_1 = \sum_{i=0}^{N-1} \max(0, \text{Gap}(T_i^F, D_i, D_N)) \cdot \mathbb{1}_{\{T_i^F \neq n.d.\}}$$

$$\text{Eq}_2 = \sum_{i=0}^{N-1} \max(0, \text{Gap}(T_B^U, D_N, D_i) + \text{Gap}(T_N^A, D_N, D_i)) \cdot \mathbb{1}_{\{T_i^F \neq n.d.\}}$$

$$\text{FUAR} = \frac{\text{Eq}_1}{\text{Eq}_2}$$

其中：
- $T_i^F$：在 $D_i$ 上训练后被 $D_N$ "忘记" 的 knowledge
- $T_N^U, T_N^A$：在 $D_N$ 上 update 和 acquire 的 knowledge
- $\text{Gap}(T, D_a, D_b) = \text{Score}(T, LM_a) - \text{Score}(T, LM_b)$
- FUAR = 1.0 表示 forget 一份旧知识换一份新知识，效率均衡

### 5.5 Online CL 专用

**NFA (Near-future Accuracy)**：

$$a_t = \mathbb{1}\{f_{\theta_t}(x_{t+1+S}) = y_{t+1+S}\}$$

$$A_t^{RA} = \frac{1}{t}(A_{t-1}^{RA} \cdot (t-1) + a_t)$$

其中 $S$ 是 minimal shift，避免 label correlation。$f_{\theta_t}$ 是用前 $t$ 个样本训练后的模型，在 $t+1+S$ 位置评估。

---

## 6. 实验数据对比分析 (Table 2, Table 3)

### 6.1 方法分类对比 (Table 2)

Table 2 展示了 4 类 CL 技术在 3 种 setting (DIL, TIL, CIL) 下的分布。关键观察：

1. **Traditional methods (Replay / Reg / Para)** 仍是主力，特别是 Replay (LAMOL, MBPA++, PAGeR, COPF 等)
2. **Parameter-Efficient** 在 LLMs-based 方法中占主导 (O-LoRA, MoE-Adapters4CL, ConPET, ELM)
3. **Instruction Tuning** 是 LLMs 时代的新兴类别 (ConTinTin, PP, DYNAINST, Continual-T0)
4. **Continual Pre-training** 主要用于 DIL (ELLE, DAS, EcomGPT-CT, CPT)

### 6.2 性能对比 (Table 3)

在 foundational text classification benchmark (AGNews, Amazon, DBpedia, Yahoo, Yelp) 上：

| Method | Backbone | Last Acc |
|--------|----------|---------|
| LFPT5 | T5 | 52.71 |
| SLM(T5) | T5 | 73.10 |
| PP(T5) | T5 | 75.10 |
| MBPA++ | T5 | 70.60 |
| OML-ER | T5 | 75.70 |
| O-LoRA | LLaMA/Alpaca | 75.80 |
| LAMOL | GPT-2 | 76.50 |
| PP(BERT) | BERT | 77.90 |
| Meta-MBPA++ | T5 | 77.30 |
| SLM(BERT) | BERT | 79.10 |

Insight：
- BERT-based 方法 (PP, SLM) 略优于 T5/GPT-2，可能因为 classification 任务上 encoder 更高效
- LLM (O-LoRA on LLaMA) 能与 PLM 方法持平，但参数量差距巨大，说明 LoRA 等 PEFT 方法仍有改进空间
- Meta-MBPA++ (replay-based) 和 PP (prompt-based) 是最强 baseline

在 FewRel (relation extraction) 上：

| Method | F1 |
|--------|-----|
| OML-ER | 69.5 |
| DynaMind (FewRel) | 88.62 |

DynaMind 基于 LLaMA + retrieval，显著超越 PLM-based 方法，说明 LLM 的 prior knowledge 对 CL 有巨大加成。

---

## 7. 我的 Intuition 与关键 Insight

### 7.1 Foundation LMs 改变了 CL 的本质

传统 CL 的核心 trade-off 是 **stability-plasticity dilemma**：新 task 学习 (plasticity) vs 旧 task 保留 (stability)。但在 foundation LMs 时代，多了第三个维度——**zero-shot capability preservation**。这从 Figure 1 的对比可见。

具体而言，预训练 model 的 general capability 本身就是 "被 forget 的旧知识" 的一部分。Fine-tune 一个 task 后，model 在其他未见 task 上的 zero-shot 性能会下降。这从 ZSCL, SND 等工作专门研究 zero-shot transfer degradation 可见一斑。

### 7.2 Replay vs Regularization vs Parameter-isolation 的权衡

从 Table 2 可以看出：
- **Replay** (LAMOL, MBPA++): 最 robust，但需要 memory buffer，且有隐私问题
- **Regularization** (EWC, CLASSIC): 轻量但容易 over-constrain，限制新 task 学习
- **Parameter-isolation** (AdapterCL, O-LoRA): 在大模型时代最流行，因为可以冻结 backbone

对 LLMs 而言，Parameter-isolation (尤其是 LoRA 系列) 是当前主流，因为：
1. Backbone 不动，天然避免 catastrophic forgetting
2. 每个 task 独立 adapter，可灵活组合
3. 推理时可以动态选择 adapter

### 7.3 Continual Pre-training 与 Continual Fine-tuning 的区别

这两个 setting 在 paper 中没有完全区分清楚，但其实差别巨大：

- **Continual Pre-training**: 在 unlabeled corpus 上继续 MLM / next token prediction，目标是注入 domain knowledge
- **Continual Fine-tuning**: 在 labeled downstream task 上持续学习

Continual pre-training 的 forgetting 主要表现为 general capability degradation，而 continual fine-tuning 的 forgetting 主要是 task-specific performance drop。这两者的解决方法 should differ。

### 7.4 Instruction Tuning 是 Foundation LMs 时代的 game-changer

从 PP, ConTinTin, Continual-T0, DYNAINST 等工作可见，将 task 转为 instruction 形式有几个独特优势：

1. **统一 task 表征**: 不同 task (classification, QA, generation) 用同一种 instruction format，便于跨 task transfer
2. **Leverage in-context learning**: 新 task 可以通过 few-shot examples 而非 parameter update 适配
3. **Forward transfer via instruction similarity**: 相似 instruction 的 task 可以共享 prompt representation

这是 LLMs 时代独有的 CL 范式，传统 CL 不存在这个选项。

### 7.5 VLMs 的 CL 有独特挑战

VLMs (CLIP 等) 的 CL 比纯 NLP 困难，因为：
1. **Cross-modal alignment**: 不仅要保留 language 知识，还要保留 vision-language alignment
2. **Zero-shot capability 是核心**: CLIP 的价值就在 zero-shot，CL 不能破坏它
3. **Modality imbalance**: vision encoder 和 language encoder 的更新节奏不同 (见 SIT 工作)

S-Prompt, Adaptation-CLIP, MoE-Adapters4CL, TRIPLET 等方法的共同特点是 **冻结 backbone + 轻量级 adapter/prompt**，这几乎是 VLMs CL 的标准范式。

### 7.6 Open Problems 与未来方向

paper 在 Section 10 列出了几个值得深入的挑战：

1. **Autonomous CL**: 不假设 task boundary 已知，model 需自主 detect distribution shift
2. **Cognitive Science 桥接**: rehearsal, memory consolidation, adaptive forgetting 等人脑机制
3. **Conversational learning**: 从对话中增量学习 (LINC 等工作)
4. **Multi-modal CL**: 跨 vision, language, audio 的统一 CL framework
5. **Privacy-preserving CL**: 差分隐私、联邦学习与 CL 结合
6. **Continual Alignment**: RLHF 的持续版本 (CPPO, COPF)，动态 human preference

我个人补充几个 insight：

- **Evaluation crisis**: 当前 benchmark 太分散 (Table 1 显示 100+ dataset)，没有统一的 ImageNet-equivalent
- **Backward transfer is rare**: 多数方法 BWT < 0，真正的 backward transfer (学新 task 提升旧 task) 几乎没有方法能实现
- **Foundation model scale matters**: 小 PLM 上有效的 CL 方法 (如 EWC) 在 LLM 上失效，因为参数 redundancy 模式不同
- **Catastrophic forgetting 在 LLMs 上更隐蔽**: 不是 accuracy 归零，而是 subtle 的 capability degradation (reasoning 能力下降，safety 失效等)

---

## 8. 关键参考文献与 Web Links

### 8.1 综述类
- 本 survey: https://arxiv.org/abs/2503.06263
- Continual Learning comprehensive survey (Wang et al. TPAMI 2024): https://arxiv.org/abs/2302.00487
- LLMs CL survey (Shi et al.): https://arxiv.org/abs/2404.16789
- Continual Learning for LLMs (Wu et al.): https://arxiv.org/abs/2402.01364

### 8.2 PLM-based 经典方法
- LAMOL (ICLR 2020): https://arxiv.org/abs/1909.03386
- AdapterCL: https://arxiv.org/abs/2012.15504
- CLASSIC (EMNLP 2021): https://aclanthology.org/2021.emnlp-main.223/
- LFPT5 (ICLR 2022): https://arxiv.org/abs/2110.07231
- Progressive Prompts (ICLR 2023): https://arxiv.org/abs/2204.05740
- ExtendNER (AAAI 2021): https://arxiv.org/abs/2012.15276

### 8.3 LLM-based 方法
- Lifelong-MoE (ICML 2023): https://arxiv.org/abs/2305.12281
- O-LoRA (EMNLP 2023): https://arxiv.org/abs/2310.10631
- ConPET: https://arxiv.org/abs/2309.14763
- ELM (ICML 2023): https://arxiv.org/abs/2305.18365
- TRACE benchmark: https://arxiv.org/abs/2406.06108
- COPF: https://arxiv.org/abs/2310.15694
- CPPO (ICLR 2024): https://arxiv.org/abs/2310.14574
- DynaMind: https://arxiv.org/abs/2310.14248
- Continual-T0 (EMNLP 2022): https://arxiv.org/abs/2208.05928

### 8.4 VLM-based 方法
- S-Prompts (NeurIPS 2022): https://arxiv.org/abs/2207.12819
- MoE-Adapters4CL (CVPR 2024): https://arxiv.org/abs/2403.11549
- ZSCL (ICCV 2023): https://arxiv.org/abs/2303.09633
- Adaptation-CLIP: https://arxiv.org/abs/2310.20348
- TRIPLET (ICCV 2023): https://arxiv.org/abs/2308.08547
- AdaptLLM: https://arxiv.org/abs/2309.09530
- DIKI: https://arxiv.org/abs/2407.05342

### 8.5 Online CL
- MBPA++ (NeurIPS 2019): https://arxiv.org/abs/1906.01095
- Meta-MBPA++ (EMNLP 2020): https://arxiv.org/abs/2008.01081
- OML-ER: https://arxiv.org/abs/2004.08468
- G-NoCL: https://arxiv.org/abs/2403.10853

### 8.6 评估与 benchmark
- TRACE: https://arxiv.org/abs/2406.06108
- Continual-T0 benchmark: https://huggingface.co/datasets/bigscience/evaluation-results
- TemporalWiki (EMNLP 2022): https://arxiv.org/abs/2203.10414
- Continual Knowledge Learning (ICLR 2022): https://arxiv.org/abs/2108.09728
- CITB: https://arxiv.org/abs/2310.14510

### 8.7 相关 background
- EWC (PNAS 2017): https://arxiv.org/abs/1612.00796
- PackNet: https://arxiv.org/abs/1711.01789
- LoRA: https://arxiv.org/abs/2106.09685
- Prompt Tuning (P-tuning): https://arxiv.org/abs/2103.10385
- Adapter (Houlsby et al.): https://arxiv.org/abs/1902.00751
- Catastrophic forgetting (Kirkpatrick): 已经典
- GEM (Gradient Episodic Memory): https://arxiv.org/abs/1706.08840
- A-GEM: https://arxiv.org/abs/1812.00420

---

## 9. 总结

这篇 survey 的核心贡献在于：

1. **Taxonomy 系统化**: 将 Foundation LMs-based CL 按 setting (offline/online × DIL/TIL/CIL × HTB/BTB) 和 method (4 类) 两个维度组织清晰
2. **覆盖三个 model 类型**: PLMs, LLMs, VLMs 分别讨论，体现了从 PLM 到 LLM 时代方法论的迁移
3. **Datasets & Metrics 全面梳理**: Table 1 给出 30+ dataset 的统计信息，Section 8 详细推导评估公式

不过，paper 也存在几个潜在局限：
- 没有统一的实验对比 (各方法在不同 setup 下用不同 backbone, dataset)
- 对 reinforcement learning 与 CL 的结合讨论较少 (除 CPPO, COPF)
- 对 LLM safety/alignment 在 CL 下的 degradation 分析不够深入
- 缺少对 "retrieval-augmented CL" 这一新范式的系统讨论 (如 DynaMind 的 retrieval 部分)

对构建 intuition 而言，关键 take-away 是：**Foundation LMs 时代的 CL 已从 "防止参数漂移" 转向 "capacity allocation + knowledge modularization"**。Parameter-isolation (特别是 LoRA-variants) + instruction tuning + retrieval 是当前最有前途的组合方向，而传统 regularization-based 方法 (EWC 等) 在大模型上的有效性值得重新审视。
