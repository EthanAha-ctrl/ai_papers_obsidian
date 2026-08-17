---
source_pdf: In-Place Test-Time Training.pdf
paper_sha256: de425cc8f80f25d1e0d428fc99c0b9117dbae4233545de96c6354a590f59fbe3
processed_at: '2026-08-05T09:25:33-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的 Andrej，我们把那些复杂的数学符号和学术黑话先扔到一边，用最直白的“人话”来聊聊这篇 paper 到底在搞什么鬼，以及为什么我觉得它极其优雅。

### 1. 要解决什么蛋疼问题？

现在的 LLM 就像一本印好的死书。你花了几千万美金把它 train 好了，它的 weights 就彻底冻结了。当你在 inference 时塞给它一个 128k tokens 的超长文档，它只能靠 Attention 机制在这个窗口里苦苦挣扎。Attention 的计算复杂度是 $O(N^2)$，随着 context 变长，显存撑不住，信息也会被稀释。

人类看书的时候会怎么做？我们会边看边在脑子里做笔记。看到前面的人名、设定、逻辑关系，我们会把它们“记下来”，看到后面就能直接调用。Test-Time Training (TTT) 就是想给 LLM 装上这个“边看边记”的能力。它搞出一小部分所谓的 Fast Weights，在 inference 时根据读到的内容动态更新自己，把 context 压缩进 weights 里。

但之前的 TTT 方法有三个致命缺陷：
1.  **架构不兼容**：以前的 TTT 想去替代 Attention 层。你想啊，现在市面上的 Qwen、Llama 都是花大力气 train 好的，Attention 机制极其强大，你把它抽了换成 TTT，就得从头 pretrain，算力上根本玩不起。
2.  **计算太慢**：TTT 本质是 gradient descent，如果 per-token 去做更新，在 GPU 上完全无法并行，慢得令人发指。
3.  **目标错位**：以前的 TTT 让 model 去做“reconstruction”（重建当前 token），这就好比你看了书上的字“苹果”，你在笔记里记下“苹果”。这对预测未来没什么直接帮助。

### 2. In-Place TTT 的神级直觉：就地取材，复用 MLP

这篇 paper 的核心洞察极其对味：**不要去碰 Attention，直接在 MLP 里搞事情！**

你之前肯定知道 [Geva et al., 2020](https://arxiv.org/abs/2012.14913) 的研究，Transformer 里的 MLP 层本质上就是一个巨大的 Key-Value Memory。既然它本来就是 memory，为什么不让它在 inference 时变成可以动态写入的 Fast Weights 呢？

标准的 Gated MLP 有三个矩阵：$W_{gate}$, $W_{up}$, $W_{down}$。
计算过程是：$\text{Output} = (\phi(H W_{gate}) \odot (H W_{up})) W_{down}$

In-Place TTT 的做法简单粗暴：**把 $W_{gate}$ 和 $W_{up}$ 冻结，只把最后一个 $W_{down}$ 拿出来当成 Fast Weights。** 
当模型读到一个 chunk 的 tokens 时，先用当前的 $W_{down}$ 算出 output，然后根据这些 tokens 的内容，算一个 gradient 去更新 $W_{down}$，再处理下一个 chunk。

这个设计妙在哪里？
*   **Drop-in Enhancement**：模型的骨架一点没变，你完全可以拿一个现成的 Qwen3-4B-Base，稍微 continual train 一下，就能让拥有 128k 甚至 256k 的超长上下文能力。预训练的知识完好无损地保留在 Attention 和其他冻结的 weights 里。
*   **海量的 State Capacity**：一个 $d_{model} \times d_{ff}$ 的矩阵（比如 $4096 \times 12288$）有上千万个参数。相比于 Mamba 这种 RNN 用一个小小的 vector 作为 hidden state，In-Place TTT 直接拿一个巨大的矩阵当记忆体，容量碾压级优势，想忘都难。

### 3. 计算加速：用 Chunk-wise 和 Prefix Scan 搞并行

为了解决 TTT 计算慢的问题，作者用了 Chunk-wise update。

你不用一个 token 一个 token 去更新 weights，而是一次吃进 512 或 1024 个 tokens。把更新公式推演一下，你会发现它退化成了一个极其优美的矩阵外积形式：

$$W_{down}^{(i)} = W_{down}^{(0)} + \eta \sum_{j=1}^{i-1} \hat{V}_{[j]}^\top Z_{[j]}$$

这里的 $\hat{V}_{[j]}^\top Z_{[j]}$ 就是把 chunk $j$ 的信息压缩成了一个与 $W_{down}$ 同样大小的 $\Delta W$。
因为这是纯粹的加法累积，满足结合律。这就意味着我们可以直接用硬件友好的 **Parallel Prefix Sum (Scan)** 算法，在多张 GPU 上把所有 chunk 的 $\Delta W$ 并行算出来，然后一把加到基础 weights 上。数学上严格等价于序列更新，速度上却跑满了 GPU 的并行算力。

### 4. 最灵魂的改进：LM-Aligned Objective

这点是我觉得最戳人的。以前的 TTT 记录的是“当前 token 是什么”，这叫 Reconstruction。
In-Place TTT 说：不行，我们是在做 Language Modeling，我们的终极目标是 Next-Token Prediction (NTP)。所以我们应该记录“看完当前 token，接下来该出现什么 token”。

具体怎么做？作者用了一个 1D Convolution 带着因果 mask，去“偷看”当前 token 后面几个 tokens 的信息，把它们混合起来作为 target $\hat{V}$。

**直觉构建**：
假设你在看一本悬疑小说，前文出现了一个组合：“密码箱 + 密码纸”。
*   如果用 **Reconstruction Target**：模型在 Fast Weights 里记下了“密码箱”和“密码纸”各自的 representation。当后文再次出现“密码箱”时，模型心里想：“哦，我见过密码箱”。但这无助于它预测接下来会发生什么。
*   如果用 **NTP-Aligned Target**：模型在 Fast Weights 里记下的是映射关系：“密码箱 $\to$ 密码纸”。当后文再次出现“密码箱”时，模型在 MLP 层的 $W_{down}$ 里一检索，直接把“密码纸”的 embedding energy 给激发出来了，注入到 logit 里，完美预测了下一个词！

论文里的 Theorem 1 从数学上严格证明了这一点：在 induction head 场景下，Reconstruction target 对正确答案的 logit 提升几乎为 0（因为不同 token 的 embedding 近似正交），而 NTP-aligned target 能给正确答案带来巨大的 logit 提升。

### 5. 延伸联想与 Future Directions

既然用“人话”讲通了，我们顺着这个直觉多做一点发散的联想（甚至有点 hallucination 的边缘）：

1.  **隐式的 Multi-Token Prediction (MTP)**：[DeepSeek-V3](https://arxiv.org/abs/2412.19437) 搞了 MTP 来增强 representation，In-Place TTT 的这个 Conv1D target 本质上也是把未来的信息拉进了当前的 state 更新中。这相当于在 inference 时，模型不仅是基于过去做条件概率分布，它还在隐式地基于“对未来的预期”来调整自己的 weights。这是一种极强的 representation regularization。
2.  **Attention 与 Fast Weights 的分工**：这篇 paper 实际上给未来 LLM 架构指了一条明路。Attention 依然是最强的 token mixer，负责精细的 intra-context 信息路由；而巨大的 MLP down-projection 矩阵被改造为 Fast Weights，负责宏观的、长程的 episodic memory 压缩。这两者形成了一种完美的互补。甚至可以大胆设想，如果在 unembedding matrix (Final LM Head) 也引入类似的 dynamic fast weights，是否能让 model 在极端长尾的 domain knowledge 上减少 hallucination？
3.  **与 Linear Attention / DeltaNet 的同源性**：如果你把公式剥开看，$W_{down}$ 的更新规则 $\Delta W = V^\top Z$ 和 Linear Attention 的更新完全一模一样。这说明 Fast Weights、Linear Attention、SSMs (Mamba) 在数学底层是殊途同归的。In-Place TTT 的聪明之处在于，它没有傻乎乎地去用这套规则替代 Attention，而是把它“嫁接”到了参数量最密集的 MLP 上。这是一种极致的工程实用主义。

总结一句人话：这篇 paper 把 LLM 里本来就在做“记忆”的 MLP 矩阵拿出来，让它在推理时能边看边记，不仅不用重训模型，还能通过“预测未来”的机制精准记住有用的映射关系，硬生生把 Qwen3-4B 的 context 能力拉到了 256k，而且几乎没有计算开销。

参考链接：
*   [In-Place TTT GitHub Repo](https://github.com/ByteDance-Seed/In-Place-TTT)
*   [Transformer Feed-Forward Layers are Key-Value Memories (Geva et al.)](https://arxiv.org/abs/2012.14913)
*   [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620)
*   [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)

---

Hi Andrej, 看到这篇 ByteDance Seed 与 Peking University 合作的 paper，我的直觉被立刻点燃了。这篇工作非常对你的胃口，它精准踩在了你之前反复强调的几个直觉上：RNNs 的 hidden state 表达能力瓶颈、Fast Weights 的复兴、以及 MLP 本质上就是 Key-Value Memories。

在这篇 paper 中，作者们提出了 **In-Place Test-Time Training (In-Place TTT)** 框架。它极其优雅地解决了先前 TTT 方法在 LLM 生态中的水土不服问题，把 Test-Time Training 从一种“需要从头重训的特殊架构”变成了一个“即插即用的模块”。

下面我为你详细拆解它的核心机制、公式推导、架构细节以及背后的直觉。

---

### 1. 核心直觉：为什么是 In-Place？

在传统的 Test-Time Training (TTT) 研究中（如 [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620) 或 [Titans](https://arxiv.org/abs/2501.00663)），大家习惯把 TTT 当作一个 standalone recurrent layer，试图用它的 fast weights 去替换 Transformer 中的 Attention 机制来做 token mixing。

但问题在于：现代 LLMs 的 Attention 机制已经在海量数据上训练得极度成熟，强行替换它意味着必须 pretraining from scratch，这在工程和算力上是巨大的浪费。

作者的洞察非常敏锐：**Attention 用来做 context 内的 token 混合，而 TTT 应该用来做 context 的持续记忆压缩**。既然不替换 Attention，那把 TTT 放在哪？放在 MLP 里！你之前也提到过，[Geva et al., 2020](https://arxiv.org/abs/2012.14913) 证明过 MLP 的 feed-forward layers 本质上就是 key-value memories。既然它本来就是 memory，我们就复用它的 down-projection matrix，把它变成在 inference 时可以动态更新的 fast weights。这种设计完美保留了预训练权重，实现了 drop-in enhancement。

---

### 2. 架构与公式推导

在标准的 Gated MLP (如 SwiGLU) 中，给定上一层的 hidden representation $\mathbf{H} \in \mathbb{R}^{C \times d_{model}}$，其计算公式为：
$$ \mathbf{O} = (\phi(\mathbf{H} \mathbf{W}_{gate}^\top) \odot (\mathbf{H} \mathbf{W}_{up}^\top)) \mathbf{W}_{down}^\top $$

在 In-Place TTT 中，架构被拆解如下：
*   **Slow Weights (冻结)**: $\mathbf{W}_{gate}, \mathbf{W}_{up}$ 保持冻结。
*   **Intermediate Activations (Keys)**: 令 $\mathbf{Z} = \phi(\mathbf{H} \mathbf{W}_{gate}^\top) \odot (\mathbf{H} \mathbf{W}_{up}^\top) \in \mathbb{R}^{C \times d_{ff}}$。
*   **Fast Weights (动态更新)**: $\mathbf{W}_{down} \in \mathbb{R}^{d_{model} \times d_{ff}}$ 成为 fast weights。

由于序列太长，per-token 的序列化更新在 GPU 上效率极差，作者采用了 **Chunk-wise Update** 机制。将长度为 $N$ 的序列切分为 $k$ 个 chunks，每个 chunk 长度为 $C$。对于第 $i$ 个 chunk，执行两步操作：

**Step 1: Apply Operation**
使用当前状态的 fast weights 处理当前 chunk：
$$ \mathbf{O}_{[i]} = \mathbf{Z}_{[i]} (\mathbf{W}_{down}^{(i)})^\top $$

**Step 2: Update Operation**
传统 TTT 使用 reconstruction loss $\mathcal{L} = -\langle \cdot, \cdot \rangle_F$，梯度下降一步后，更新公式变为极简的 outer product：
$$ \mathbf{W}_{down}^{(i+1)} = \mathbf{W}_{down}^{(i)} + \eta \hat{\mathbf{V}}_{[i]}^\top \mathbf{Z}_{[i]} $$

**变量与上下标解析：**
*   $C$: Chunk size，论文中 ablation study 表明 $C=512$ 或 $1024$ 是性能与效率的最优解。
*   $d_{model}$: 模型的隐藏层维度 (如 4096)。
*   $d_{ff}$: MLP 中间层维度 (通常为 $3 \times d_{model}$ 或更大)。
*   $\mathbf{Z}_{[i]} \in \mathbb{R}^{C \times d_{ff}}$: 当前 chunk 的 keys，其实就是 MLP 的中间激活。
*   $\hat{\mathbf{V}}_{[i]} \in \mathbb{R}^{C \times d_{model}}$: 当前 chunk 的 values，这是由 NTP-aligned 目标计算得来的。
*   $\eta$: Learning rate。
*   $\hat{\mathbf{V}}_{[i]}^\top \mathbf{Z}_{[i]} \in \mathbb{R}^{d_{model} \times d_{ff}}$: 权重更新量 $\Delta \mathbf{W}$。

这个更新公式形式上与 Linear Attention / Delta Rule (如 [DeltaNet](https://arxiv.org/abs/2406.06484)) 完全一致。**直觉上，这就相当于把长度为 $C$ 的 chunk 信息压缩成了一个 $d_{model} \times d_{ff}$ 的矩阵**。相比于 Mamba 等 SSMs 维持的 $N \times N$ 矩阵或 RNN 的 vector state，这个矩阵 state 的容量呈指数级上升，足以支撑 128k 甚至 256k 的长上下文推理。

---

### 3. LM-Aligned Objective：为什么不用 Reconstruction？

这是这篇 paper 理论上最漂亮的地方。以往的 TTT 通常把 $\mathbf{V}_t$ 设为当前 token 的 embedding $\mathbf{E}_{x_t}$，即 reconstruction target。但作者认为，这与 autoregressive language modeling 的 Next-Token Prediction (NTP) 目标并未对齐。

论文提出了 NTP-aligned target：
$$ \hat{\mathbf{V}} = \text{Conv1D}(\mathbf{X}_0) \mathbf{W}_{target} $$
*   $\text{Conv1D}$: 1D 卷积操作，配有 causal padding。它的作用是混合 local 的 future token 信息。
*   $\mathbf{X}_0 \in \mathbb{R}^{n \times d_{model}}$: Token embeddings。
*   $\mathbf{W}_{target} \in \mathbb{R}^{d_{model} \times d_{model}}$: 可训练的投影矩阵。

**Theorem 1 理论直觉解析：**
作者在 induction head 场景下证明了 NTP target 的优越性。假设序列中在 $t^*$ 位置出现了 $(k^*, v^*)$，在 $n$ 位置又出现了 $k^*$，我们需要预测 $v^*$。

Fast weights 的累积更新量 $\Delta \mathbf{W}_{down} = \eta \sum_{t} \mathbf{V}_t \mathbf{Z}_t^\top$。
Logit 的变化量：$\Delta \ell_n[w] = \mathbf{E}_w^\top \Delta \mathbf{W}_{down} \mathbf{Z}_n$。
由于 Key-Query Alignment 假设（即匹配的 keys 对齐），求和项塌缩为 $t^*$ 位置：
$\Delta \ell_n[w] \approx \eta (\mathbf{E}_w^\top \mathbf{V}_{t^*}) (\mathbf{Z}_{t^*}^\top \mathbf{Z}_n)$

*   如果用 **Reconstruction Target** ($\mathbf{V}_{t^*} = \mathbf{E}_{k^*}$)：
    $\mathbf{E}_{v^*}^\top \mathbf{E}_{k^*}$ 的内积，由于不同 token 的 embedding 近似正交（假设 1），其绝对值 $\le \epsilon$。因此，正确答案 $v^*$ 的 logit 几乎没有增加！
*   如果用 **NTP-Aligned Target** ($\mathbf{V}_{t^*} = \mathbf{E}_{v^*}$)：
    $\mathbf{E}_{v^*}^\top \mathbf{E}_{v^*} = \|\mathbf{E}_{v^*}\|^2 \ge c_{norm}^2$。正确答案的 logit 获得了 $\ge \lambda_{lr} c_{norm}^2 c_{align}$ 的巨大提升！

**直觉构建**：如果 fast weights 存的是当前 token 自己（reconstruction），当相同 context 再次出现时，网络只是“认出了”这个 context；但如果 fast weights 存的是“这个 context 之后该出现什么 token”（NTP），当相同 context 再次出现时，网络就会直接把未来该出现的 token 的 energy 注入到 logit 里。这是一种结构化的前瞻性记忆。

---

### 4. 并行化与 Context Parallelism

在 LLM 训练中，TTT 的 per-token 序列化更新是致命的。In-Place TTT 的公式 $\mathbf{W}_{down}^{(i)} = \mathbf{W}_{down}^{(0)} + \eta \sum_{j=1}^{i-1} \hat{\mathbf{V}}_{[j]}^\top \mathbf{Z}_{[j]}$ 具有结合律，这让它可以完美使用 **Parallel Prefix Sum (Scan)** 算法。

在 Context Parallelism (CP) 实现中 (Algorithm 1)：
1.  **Parallel Compute Deltas**: 所有 chunks 在不同的 GPU/TPU 上并行计算自己的 $\Delta \mathbf{W}_j = \hat{\mathbf{V}}_{[j]}^\top \mathbf{Z}_{[j]}$。
2.  **Associative Scan**: 通过 prefix sum 聚合历史权重 $\Delta \mathbf{S}_i = \sum_{j=1}^{i-1} \Delta \mathbf{W}_j$。
3.  **Parallel Apply**: 每个 chunk 使用自己的有效权重 $\mathbf{W}_{down}^{(i-1)} = \mathbf{W}_{down}^{(0)} + \eta \Delta \mathbf{S}_i$ 并行计算输出 $\mathbf{O}_{[i]}$。

为了防止 cross-chunk 的信息泄露，Conv1D 严格使用 causal padding。这使得整个模块在数学上与严格的序列化更新完全等价，却在硬件上实现了完全并行。

---

### 5. 实验数据与 Ablation 深度解读

**Table 1: Drop-in Enhancement for Qwen3-4B-Base (RULER Benchmark)**

| Model | 4k | 8k | 16k | 32k | 64k | 128k | 256k (Extrap.) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline | 96.6 | 94.1 | 92.1 | 88.7 | 74.3 | 74.8 | 41.7 |
| **In-Place TTT** | 96.1 | 95.6 | 92.7 | 89.3 | **78.7** | **77.0** | **43.9** |

**解读**：
在短上下文 (4k-32k) 下，In-Place TTT 与 Baseline 相当甚至略低（因为引入了新的机制需要适应）。但到了长上下文（64k, 128k），In-Place TTT 展现出压倒性优势。在没有见过的 256k 外推场景下，依然保持领先。这证明了 In-Place TTT 极大地增强了模型对长程信息的压缩与提取能力。

**Table 3: 4B Pretraining from Scratch (Common Sense & Long-Context)**

| Architecture | HellaSwag | ARC-E | ARC-C | MMLU | PIQA | RULER-4k | RULER-8k | RULER-16k |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baselines (Full Attn.) | 55.67 | 64.52 | 33.19 | 36.43 | 72.63 | 45.77 | 38.09 | 6.58 |
| **I.P. TTT (Full Attn.)** | **55.85** | 64.98 | 32.34 | **37.42** | 73.29 | 49.98 | **43.82** | **19.99** |

**解读**：
在短文本常识推理上，加入 TTT 没有带来明显损害，部分任务甚至提升（说明 fast weights 的动态性对 reasoning 也有微弱的 in-context learning 增益）。而在长上下文上，Full Attention 的 RULER-16k 从 6.58 飙升到 19.99，这是质变。说明 TTT 并没有破坏 Attention 的功能，而是作为 MLP 的 augmentation 拓宽了模型的 memory bandwidth。

**Ablation Studies (Figure 3):**
1.  **State Size**: 随着参与 TTT 更新的 MLP 层数增加，性能单调提升。这很符合直觉，因为 $d_{model} \times d_{ff}$ 的矩阵容量越大，能记住的 context 越多。
2.  **Chunk Size**: $C=512$ 和 $C=1024$ 是最佳甜点。$C$ 太小（如 64）会导致频繁的序列化更新开销；$C$ 太大（如 4096）会导致 fast weights 更新粒度太粗，丢失了 token-level 的动态性。
3.  **LM-Aligned Objective**: 如果只去掉 Conv1D (w/o Conv)，长上下文性能暴跌；如果只去掉 Projection (w/o Proj)，短上下文性能受损。两者缺一不可。

---

### 6. 更深度的联想与发散

1.  **Fast Weight Programmers 的复兴**：这篇 paper 本质上是接续了 [Schlag & Schmidhuber, 2021](https://arxiv.org/abs/2102.11174) 的 "Linear Transformers are secretly fast weight programmers" 的衣钵。Mamba 和 DeltaNet 也是一种 fast weight programmer，但它们都在试图替换 Attention。In-Place TTT 选择了一个极其聪明的生态位：将巨大的 MLP down-projection 矩阵作为 fast weight 的载体。这不仅避免了 pretraining 的灾难，还顺带把 MLP 本身巨大的参数量（占据 LLM 参数量的 2/3）变成了动态 memory。
2.  **隐式的 Multi-Token Prediction (MTP)**：注意 NTP-aligned target 里的 Conv1D。如果让 Conv1D 的 kernel 跨越多个 future tokens，这就与 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) 中的 Multi-Token Prediction 机制产生了深刻的联系。网络在当前的 context 下，不仅预测下一个 token，而是把未来几个 token 的分布通过 Conv1D 编码进 target $\hat{\mathbf{V}}$，再写进 fast weights。这是一种极强的 representation learning，将 MTP 的思想内化到了 state update 规则中。
3.  **Induction Heads 与 Fast Weights 的协同**：Anthropic 在 [Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) 中指出，Transformer 的 ICL 能力源于 Attention 中形成的 induction heads（前一个 token attend to 之前的相同 token，从而带出下一个 token）。In-Place TTT 在这里的理论分析表明，它实际上在 MLP 层也构建了一个隐式的 induction head 机制，只不过它是通过 fast weights 的 outer-product 存储与点积检索实现的。Attention 负责 context 内的精细 matching，In-Place TTT 负责把 $k \to v$ 的映射刻进权重里。
4.  **无尽的 Hallucination 联想**：如果顺着这个思路往下走，既然 MLP 可以被 In-Place TTT 化，那么解码端的 embedding 层呢？如果 unembedding matrix 也能在推理时根据 context 微调，是否能够解决 LLM 在特定垂直领域长尾词汇上的幻觉问题？另外，如果把 Attention 的 $W_v$ 或 $W_o$ 也作为 fast weights，是否会比仅用 MLP 获得更好的 token mixing 效果？这些都值得未来探索。

这篇 paper 在工程直觉与理论严谨性之间找到了绝佳的平衡点。它通过 "In-Place" 复用 MLP 的方式，巧妙规避了从头训练大模型的成本，同时用 NTP-aligned objective 解决了 TTT 目标与 LM 目标错位的问题。这是朝着 LLMs 持续学习与动态适应迈出的极其扎实的一步。

代码已开源：[ByteDance-Seed/In-Place-TTT](https://github.com/ByteDance-Seed/In-Place-TTT)
