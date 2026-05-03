基于第一性原理，这段文字所表达的 Continued Pretraining (CPT) 的核心本质是：**在一个已经完成 initial pretraining 的 base model 上，利用全新的、分布外的 raw text 数据，继续优化其参数，从而将其内部表征空间“导航”至全新的知识域 OR 语言空间。**

为了 build your intuition，我们可以将 LLM 视为一个已经学习了世界通用知识分布 $P_{general}$ 的概率模型。Initial pretraining 赋予了它广度，但面对特定的 domain（如 law, medicine）OR 特定的 language（如低资源语言），模型的分布 $P_{model}$ 与目标分布 $P_{domain}$ 之间存在巨大的 KL 散度。CPT 的目的就是最小化这个散度，使得 $P_{model} \rightarrow P_{domain}$。

下面从底层架构、数学公式、优化策略三个维度进行更细节的技术讲解：

### 1. 架构解析：为什么 CPT 需要训练 `embed_tokens` AND `lm_head`？

在 Transformer 架构中，input text 首先经过 tokenization 被映射为 integer indices，然后通过 `embed_tokens` 层（即输入嵌入矩阵 $W_e$）转化为连续向量，经过多层 Transformer blocks 计算后，最终通过 `lm_head` 层（即输出分类矩阵 $W_{out}$）映射回 vocabulary space 的 logits。

*   **First Principle Insight**: 当 base model（e.g., Llama-3 8b）没有见过某种语言 OR domain 的特定 token 时，这些 token 在 $W_e$ 中的初始化是随机 OR 稀疏的。模型无法在 semantic space 中为这些 new tokens 找到合理的坐标。
*   **CPT 的作用**: 通过将 `embed_tokens` AND `lm_head` 加入 `target_modules`，我们允许模型在反向传播时调整首尾两层的权重，从而为 new tokens 构建全新的语义流形。

**数学表达**：
假设 input sequence 为 $X = [x_1, x_2, ..., x_T]$，CPT 的优化目标是标准的自回归交叉熵：
$$ \mathcal{L}_{CPT}(\theta) = -\sum_{t=1}^{T} \log P_{\theta}(x_t | x_{<t}; D_{cpt}) $$
其中：
*   $T$: 序列长度 (sequence length)
*   $x_t$: 当前时刻的 token index
*   $x_{<t}$: 当前时刻之前的所有 token indices
*   $\theta$: 模型参数，重点包含了 $W_e$ (embed_tokens) AND $W_{out}$ (lm_head)
*   $D_{cpt}$: CPT 阶段的 domain-specific raw text dataset

在 LoRA (Low-Rank Adaptation) 设定下，全量参数 $\theta$ 被冻结，只有增量 $\Delta \theta$ 被更新。对于 `embed_tokens`，Unsloth 允许对其进行 LoRA 分解：
$$ W_e^{new} = W_e^{old} + \Delta W_e = W_e^{old} + B_e A_e $$
其中 $B_e \in \mathbb{R}^{d_{model} \times r}$, $A_e \in \mathbb{R}^{r \times |V|}$, $r$ 是 LoRA rank (e.g., 16), $d_{model}$ 是隐藏层维度，$|V|$ 是 vocabulary size。

### 2. 优化策略：为什么 `embedding_learning_rate` 需要 2-10x 更小？

Unsloth 代码中指出了一个关键细节：对 `lm_head` OR `embed_tokens` 使用比主网络小 2-10 倍的学习率 (e.g., `learning_rate = 5e-5` vs `embedding_learning_rate = 5e-6`)。

**First Principle Insight**: 模型的中间层 ($\theta_{transformer}$) 承载了逻辑推理和语法转换的能力，而首尾层 ($W_e, W_{out}$) 承载的是“符号到概念”的映射字典。如果对 $W_e$ AND $W_{out}$ 使用过大的 learning rate $\eta_{emb}$，会导致：
1.  **Manifold Collapse（流形坍缩）**: 新来的 domain tokens 会以极大的梯度撕裂原有的 semantic space，导致原本已经对齐的通用 tokens 发生位移，引发 Catastrophic Forgetting（灾难性遗忘）。
2.  **Representation Misalignment**: Input embedding 的剧烈变化会导致输入给 Transformer blocks 的向量偏离 pretraining 阶段的分布，导致中间层的 Attention 机制失效。

**梯度更新公式**：
$$ W_e^{(t+1)} = W_e^{(t)} - \eta_{emb} \nabla_{W_e} \mathcal{L}_{CPT} $$
$$ \theta_{transformer}^{(t+1)} = \theta_{transformer}^{(t)} - \eta_{base} \nabla_{\theta} \mathcal{L}_{CPT} $$
约束条件：$\eta_{emb} \in [\frac{\eta_{base}}{10}, \frac{\eta_{base}}{2}]$
较小的 $\eta_{emb}$ 确保了 new tokens 的表征是平滑地“嵌入”到现有流形中的空白区域，AND 不会剧烈扰动 known tokens 的相对位置。

### 3. 状态重置：加载 LoRA adapter 时 Optimizer State 的 Reset 问题

Unsloth 提到："The optimizer state will be reset as well." 

**First Principle Insight**: 现代 LLM 训练使用 AdamW OR AdamW 的变体（如 AdamW 8-bit）。Adam 优化器为每个参数 $\theta_i$ 维护两个动态统计量：
*   一阶动量（Momentum）$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
*   二阶动量$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$

当加载已有的 LoRA adapter 继续 CPT 时，如果 optimizer state 被 reset，意味着 $m_0 = 0, v_0 = 0$。这会导致最初的几步训练中，自适应学习率出现巨大波动（因为 $v_t$ 极小，导致参数更新的分母极小，步长骤增）。这也是为什么在 CPT 的初期，loss 可能会出现短暂 spike 的原因。

### 4. 扩展联想与实验数据参考

在 CPT 领域，还有几个关键的 First Principle 维度值得深挖：

*   **Data Mixing Ratio**: 为了防止 CPT 过程中的 Catastrophic Forgetting，通常不会只用 target domain 的 $D_{cpt}$，而是混入一定比例（通常 5%-15%）的 general corpus（如 Wikipedia）。实验数据表明，完全没有 general data 的 CPT 会导致 model 在通用 benchmark（如 MMLU, ARC）上性能下降 10%-30%。
*   **Learning Rate Schedule**: CPT 通常不采用 pretraining 阶段常见的 cosine decay 到 0 的策略，而是采用 constant with warmup OR cosine with a floor（比如衰减到最大 LR 的 10%），以保持对新知识的吸收能力。

**参考链接与延伸阅读：**
1.  **Unsloth Official Blog on CPT**: [Continual Pretraining with Unsloth](https://unsloth.ai/blog/continual-pretraining)
2.  **LoRA Paper (Hu et al.)**: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) (理解 target_modules 的数学基础)
3.  **Catastrophic Forgetting in LLMs**: [https://arxiv.org/abs/2308.08747](https://arxiv.org/abs/2308.08747) (理解为什么需要差异化学习率和数据混合)
4.  **Llama-3 Base Model Details**: [https://ai.meta.com/blog/meta-llama-3/](https://ai.meta.com/blog/meta-llama-3/) (理解 15 Trillion tokens 的 pretrain 规模如何影响 CPT 的起步)