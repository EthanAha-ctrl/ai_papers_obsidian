---
source_pdf: memo.pdf
paper_sha256: ea145bc5c3eb19990361c75167c9d6a9d8f1d1e302e9e2316134d1ab885aecca
processed_at: '2026-08-05T17:27:58-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，咱们用最纯粹的大白话来剥开这篇 paper 的内核。你搞过 LL.M 和 RL，肯定一眼就能看穿这层窗户纸。

## 1. 核心痛点：Transformer 在 Embodied RL 里的“记仇”与“健忘”

想象你让一个 Fetch 机器人在一个 3D 房子里连续找 20 个不同的物体。它走了一万步。如果用标准的 Transformer 做 policy，会发生什么？

它会把过去一万步的每一帧 RGB 画面、每一个 Odometry 位置全塞进 Context 里。这就像一个人在房间里走着，脑子里把看过的每一块地砖、每一面白墙的像素都高清重放。结果是：
1. **算力爆炸**：Attention 是 $O(N^2)$ 的，上下文一长，GPU 显存直接 OOM。
2. **抓不住重点**：Reward 极其 sparse（只有找到物体才给分），面对上万帧的视觉噪音，Attention Weights 很容易把梯度分散到无关紧要的画面上，导致 Credit Assignment 极度困难。
3. **一截断就失忆**：Inference 时候为了省显存，把前面的 Context 截断了，模型立刻变成傻子，忘了刚才探索过哪里。

## 2. Memo 的解法：强制做“精华笔记”

人类的脑子怎么干的？我们不会记住地砖的颜色，我们只记住“厨房在左边，里面有香蕉”。Memo 就是让 Transformer 学会做笔记。

**机制极其简单粗暴：**
把长长的 Context 切成一个个 Segment（比如每 256 步切一刀）。在每段结束时，模型不直接输出 Action，而是输出 32 个特殊的 "Summary Tokens"。
当下一个 Segment 开始时，模型能看到的 Context 只有：
- 之前所有的 "Summary Tokens"（你的笔记集）
- 当前 256 步的 Raw Observations（你眼前正在看的东西）

这就形成了一个巨大的 Information Bottleneck。过去的成千上万帧画面被强制抹除了，模型要想在后面拿到 Reward，就必须在这 32 个 Tokens 里把对任务有用的信息（比如物体相对位置、房间拓扑结构）压缩进去。

## 3. 为什么这在 RL 里能 Work？两个致命细节

如果把 NLP 里的 Context Compression（比如 Autocompressors, RMT）直接生搬硬套过来，在 RL 里会死得很惨。Memo 解决了两个 RL 独有的致命问题：

### 3.1 梯度高速公路
**RMT 的死穴**：RMT（Recurrent Memory Transformer）只保留最新的一组 Summary，前一段的 Summary 必须被融合进下一段。如果在第 1 步看到了钥匙，第 10000 步要用，梯度必须穿过 10000 次 Sequential Memory Update。这在 RL 的 Sparse Reward 下，梯度早就 Vanishing 消失了，模型根本学不到“第 1 步的观察对第 10000 步的 Reward 有贡献”。

**Memo 的解法**：Summary 是**累积**的。第 10000 步的 Attention 可以直接 Softmax 到第 1 步生成的 Summary Tokens 上。这相当于在时间轴上建了一条 Skip-Connection 的高速公路。早期关键信息的梯度可以直接回传，不被中间的几千步稀释。

### 3.2 端到端的长程梯度传播
**Autocompressors 的死穴**：在 LLM 里，通常是预训练好短 Context 的模型，然后再微调做 Compression，并且为了省显存会把梯度截断（TBTT）。
**Memo 的解法**：从零开始训练，并且**不截断梯度**。为什么？因为 RL 没有 supervised target，模型唯一的学习信号就是 Trial 最后的 Reward。如果你把梯度截断了，模型根本不知道在第 1 段 Summary 里该压缩什么信息才能对第 10 段的 Reward 有帮助。必须让梯度穿透整个 Trial 的所有 Summary，模型才能学会“为了未来的成功，我现在该记住什么”。

## 4. 数学与架构直觉解析

### Attention Masking 的精妙之处
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V $$
在 Memo 中，当前 Segment 的 Query $Q_{current}$ 只能和当前 Segment 的 Key $K_{current}$ 以及所有历史 Summary 的 Key $K_{summary\_history}$ 做内积。
历史 Segment 的 Raw Observation 的 Keys 被直接 Mask 掉了。这就强迫模型必须把信息搬运到 $K_{summary}$ 里，否则信息就彻底丢失了。

### Positional Encoding 的重置
为了防止 Position Index 爆炸导致 RoPE 泛化失效，Memo 在生成 Summary Tokens 时做了一次“重置”。
假设当前是第 $n$ 个 Segment，当前时间步 $t = n \cdot l_{seg} + i$。
当前输入序列的 Position Index 不是从 $0$ 到 $t$，而是：
- 前 $n \cdot l_{sum}$ 个位置分给历史 Summary Tokens（Index $0$ 到 $n \cdot l_{sum} - 1$）。
- 接着是当前 Segment 的 Raw Observations，Index 从 $n \cdot l_{sum}$ 开始。
这种设计让模型在处理 Summary 时，仿佛在处理一个较短的新序列，大大提升了 Length Extrapolation 的能力。

### SPL (Success weighted by Path Length) 公式拆解
$$ SPL = \frac{1}{N} \sum_{i=1}^{N} S_i \cdot \frac{L_i^*}{\max(L_i, L_i^*)} $$
- $N$: 总 episode 数。
- $S_i$: 二值变量，成功找到目标为 1，否则为 0。
- $L_i$: Agent 实际走过的路径长度。
- $L_i^*$: 算法算出来的最短路径。
**Intuition**: 如果 $S_i=1$ 且 $L_i = L_i^*$（没走弯路），SPL 就是 1。如果瞎逛了半天才找到，分母变大，SPL 剧降。Memo 在 SPL 上击败了 Full Context Transformer，说明它不仅找到了物体，而且脑海里有一张更清晰的“Cognitive Map”，走了更高效的捷径。

## 5. 疯狂的实验现象与 Brain Storming

### 5.1 为什么“丢掉”信息反而性能更好？
实验中，Memo 用了 **8倍** 少的 Tokens，Success Rate 却比 Full Context Transformer **高了 7.5%**。
这极其反直觉。我的解释是：Full Context Transformer 在海量视觉帧中陷入了 Attention Distraction。没有强归纳偏置时，它把算力浪费在了计算“第 3000 帧的墙角和第 8000 帧的桌子有什么关系”上。Memo 的强制 Bottleneck 就是一种 Implicit Regularization，逼迫模型只去关注 Task-Relevant 的语义信息。

### 5.2 Streaming 评估：无限长推理的终极奥义
在 Inference 时，如果限制只能保留最近的 6000 步 KV Cache：
- **Transformer 直接崩盘**：把早期的探索轨迹一删，它在 Unseen 房间里立刻变成无头苍蝇。
- **Memo 稳如老狗，甚至性能还在涨**：为什么？因为早期的 Summary Tokens 保留了。丢掉的只是最近几步的 Raw Frames。这就像你忘记了五分钟前走过的走廊细节，但你手里攥着“整个房子的地图笔记”。这说明模型真正学会了把 Short-Term Visual Buffer 和 Long-Term Semantic Memory 解耦。

### 5.3 Summary Length 的 sweet spot
实验发现 $l_{sum} = 32$ 最好，16 太少（信息丢光了），64 太多（模型变懒了，直接把 Raw Features 拷贝进 Summary 里，导致信噪比下降，且 Overfit 到 Training Context Length）。这说明 Compression Ratio 必须卡在一个让模型“感到疼”但又能存下核心信息的点上。

## 6. 我的思想发散与过度联想

既然要 build intuition，我就放开联想一下：

1. **对比 LLM 的 StreamingLLM (Attention Sinks)**：StreamingLLM 是通过保留最开头的几个 Token 来锚定 Attention 分布，防止 Softmax 崩溃。Memo 在 RL 里的 Summary Tokens 其实就是高级版的 Attention Sinks，只不过它们承载了真实的 Semantic Memory。如果把 Memo 的思路用到 LLM Agent 上（比如让 LLM 玩《我的世界》连续玩几天），效果绝对爆炸。目前 LLM Agent 的 Context 爆炸问题，完全可以用 Memo 的 Segment Summarization 来解决。

2. **与世界模型 (World Models / JEPA) 的结合**：Memo 目前是完全依赖 RL 的 Sparse Reward 来学习怎么压缩的。这太苦了。如果我们在 Summary Tokens 上加一个 Auxiliary Loss，比如让它们去 Predict 未来几步的 Latent State（类似 DreamerV3 或 JEPA），模型就能在早期就学到有意义的 Compression，RL 的 Sample Efficiency 肯定能再翻一倍。

3. **Memory Consolidation 机制缺失**：Memo 是线性累积 Summary 的，如果 Trial 跑到 100 万步，Summary Tokens 也会爆炸。人类大脑有 Memory Consolidation（睡眠时的记忆固化），把短期记忆压缩成长期记忆。完全可以在 Memo 上层再加一个更高级别的 Compressor，每 10 个 Summary 再压缩成 1 个 Meta-Summary，形成分层的记忆树。这其实就是 Hierarchical RMT 的思路，但还没人结合 RL 做好。

4. **与 Test-Time Compute 的关联**：现在 LLM 圈很火的概念是 Scaling Test-Time Compute（比如 OpenAI 的 o1）。Memo 在做决策时，通过 Attention over Summaries，其实就是在做 Test-Time Reasoning over past experience。如果让模型在 Action 输出前，先在 Summary Tokens 上做几轮自回归的 CoT (Chain of Thought)，这算不算是 Embodied Agent 的 System 2 Thinking？

## 7. 总结成一句话

Memo 就是给 Transformer 装了一个“海马体”，强迫它定期把视觉流水线压缩成高密度的语义笔记，然后用一条梯度高速公路把这些笔记串起来，最终让模型在省下 10 倍显存的同时，反而因为过滤了噪音而做出了更聪明的决策。

**Reference Links:**
- Paper Github (Early Release): [https://github.com/gunshi/memo](https://github.com/gunshi/memo)
- ReLIC (On-policy baseline): [https://arxiv.org/abs/2410.02751](https://arxiv.org/abs/2410.02751)
- AMAGO (Off-policy baseline): [https://openreview.net/forum?id=M6XWoEdmwf](https://openreview.net/forum?id=M6XWoEdmwf)
- Autocompressors (Inspiration): [https://aclanthology.org/2023.emnlp-main.232](https://aclanthology.org/2023.emnlp-main.232)
- Recurrent Memory Transformer (RMT): [https://arxiv.org/abs/2304.11062](https://arxiv.org/abs/2304.11062)
- StreamingLLM (Attention Sinks concept): [https://arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)
- VC-1 (Visual Encoder used): [https://arxiv.org/abs/2306.11345](https://arxiv.org/abs/2306.11345)

---

你好 Andrej！这篇《Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning》非常契合你近期对 Memory, Long-context 以及 Embodied AI 的关注。这篇 paper 的核心 contribution 在于将 Language Model 领域中的 context compression 技术（如 Autocompressors 和 RMT）成功地迁移到 Reinforcement Learning (RL) 中，并解决了 RL 特有的 long-horizon credit assignment 与 instability 问题。

下面我为你进行深度的技术拆解，希望能 build your intuition。

## 1. Motivation & Problem Setting

在 Embodied AI 任务中，agent 需要在长周期的 trial 中不断探索并利用过去的经验。现有的 Transformer-based policy 面临几个核心痛点：
1. **Quadratic Attention Complexity**：随着 context length $t$ 的增加，self-attention 的计算和存储开销呈平方级增长。
2. **KV Cache 爆炸**：Inference 时需要存储庞大的 Key-Value cache，导致 GPU memory 紧张。
3. **Length Extrapolation 泛化困难**：如果在 training 时 context length 是 $L_{train}$，inference 时如果超过这个长度，positional encoding (如 RoPE) 往往无法泛化，导致性能崩溃。
4. **RL 独有的 Credit Assignment**：RL 的 reward 往往是 sparse 且 delayed 的，如果将历史信息压缩成单一 memory vector（如 RNN 或 RMT），梯度需要穿过大量 sequential memory updates，极易引发 vanishing gradients，导致 agent 无法学习到早期观测对后期 reward 的贡献。

**POMDP 形式化：**
Paper 将问题定义为 In-Context RL。在 timestep $t$，模型接收观测序列 $X_t = \{o_1, \ldots, o_t\}$，通过 Sequence Encoder 映射为 hidden representation $h_t = \text{SeqEnc}(X_t)$。随后 actor head 和 critic head 分别输出 policy 和 value：
$$ \pi(a_t | h_t), V(h_t) $$
其中 $a_t \in \mathcal{A}$ 是当前采取的 action，$\mathcal{A}$ 是 action space。Trial 由多个 episode 组成，episode 结束会 reset，但 agent 的 memory 跨 episode 保留。

## 2. Memo Architecture 深度解析

Memo 的核心思想是：**不要 attend 所有的历史 observations，只 attend 压缩后的 summary tokens。**

### 2.1 Segment 分割与 Summary Tokens 生成
Memo 将长的 input sequence 切分为长度为 $l_{seg}$ 的 segments。在每段 segment 结束时，模型生成 $l_{sum}$ 个 summary tokens。
- **Intuition**：这相当于强制模型做一次 "笔记"。由于大部分 visual input（如墙壁颜色、走过的空走廊）对当前任务无关，模型必须学会丢弃冗余信息，只保留 task-relevant 的特征（如目标物体的相对位置、未探索的区域）。
- **信息流瓶颈**：下一个 segment 只能 attend 之前所有的 summary tokens 和当前 segment 的 observations。这形成了一个严格的 information bottleneck，类似于 KL 散度约束下的 Variational Autoencoder，强迫 latent representation 承载高密度的语义信息。

### 2.2 Attention Masking 与 Positional Encoding
为了实现这种 bottleneck，Memo 设计了特殊的 Causal Masking：
- Current segment 的 observation 只能 attend 它之前的 summary tokens 和当前 segment 内的 observations。
- 过去 segment 的 raw observations 被 mask 掉，无法被直接 attend。

**Positional Encoding (PE) 细节：**
假设当前 timestep 为 $t$，已经经历了 $n$ 个 segments，其中 $n = \lfloor t / l_{seg} \rfloor$。
- 前 $n \times l_{sum}$ 个 positions 分配给 summary tokens（index 从 $0$ 到 $n \cdot l_{sum} - 1$）。
- 当前 segment 的 observations 从 $n \cdot l_{sum}$ 开始，到 $t - n \cdot l_{seg} + n \cdot l_{sum}$。
- **Key Insight**：这种 PE 分配方式避免了 summary tokens 的 position index 无限累积。在 Appendix A.8 的 ablation 中，如果直接从 hidden states 提取 KV cache 而不进行 re-encoding，会导致 summary 的 PE 不断累积，引发 distribution shift，模型性能迅速崩塌。

### 2.3 Segment Length Randomization
Training 时，$l_{seg}$ 在 $[0.8, 1.2] \times 256$ 范围内 uniform 随机采样；Evaluation 时固定为 $256$。
- **Intuition**：这起到了 data augmentation 和 curriculum learning 的作用。如果总是固定在 256 步做 summary，模型可能会 overfit 到这个 specific temporal boundary，变成单纯的 "步数计数器"。随机化迫使模型学习 semantic-level 的压缩，同时偶尔遇到更短的 segment（更简单的压缩任务），有利于训练初期的稳定。

### 2.4 KV Cache 与 On-Policy RL 的结合
在 On-policy RL（如 ReLIC, 基于 DD-PPO）中，模型在 rollout 期间收集数据，然后更新权重。权重更新后，旧权重生成的 KV cache 就 "stale" 了。ReLIC 的做法是用新权重重新 encode 历史 context 来 refresh KV cache。Memo 顺势而为，在 refresh KV cache 的同时，也重新计算所有的 summary tokens，确保 memory 与 current policy 保持一致。

## 3. 两大核心机制与 Baselines 对比

Memo 相比于之前的探索，关键在于以下两点设计，这也是它能 beat FCT (Full Context Transformer) 的根本原因。

### 3.1 Summary Accumulation (对比 RMT)
Recurrent Memory Transformer (RMT) 只保留固定大小的 memory（即最新的 summary tokens），前一个 segment 的 summary 必须被 overwrite 或者融合到下一个 summary 中。
- **痛点**：如果关键信息出现在 trial 早期，它的 gradient 必须穿过成百上千次 sequential memory updates 才能到达 loss function，这极易发生 gradient vanishing。
- **Memo 的解法**：Memo 累积所有的 summary tokens。在 timestep $t$，之前所有的 $S_1, S_2, \ldots, S_n$ 都作为直接的 input tokens 参与 attention。这意味着 early summary 可以通过 attention mechanism 直接与 current timestep 交互，梯度可以直接回传，形成了类似 ResNet 中 skip-connection 的效果。

### 3.2 Long-Horizon Gradient Propagation (对比 Autocompressors)
Autocompressors (AC) 在 NLP 领域用于压缩 context，但通常采用 Truncated Backpropagation Through Time (TBTT)，只允许梯度穿过相邻的 2 个 segments，且通常是基于预训练模型微调。
- **痛点**：在 RL 中，reward 极度稀疏。如果 gradient 被截断，模型根本不知道早期的 summary 应该压缩什么信息才能对后期的 reward 有益。
- **Memo 的解法**：端到端训练，从 scratch 开始，允许梯度穿过 trial 内的所有 segments（即 Algorithm 2 中的 `SeqEnc` 过程）。
- **实验佐证**：在 Figure 3b 中，AC (all segments) 在早期甚至略好于 Memo（因为预训练权重的加持），但随着训练进行，性能严重退化并崩塌到 AC (TBTT) 的水平。这证明了预训练短 context 模型可能会学到一种 "懒惰" 的 attention pattern，无法适应长 context 下的 compression 需求。

## 4. 实验数据与深度分析

### 4.1 EXTOBJNAV (Habitat 3D 导航)
**任务设定**：Agent 在 HSSD 场景中连续寻找不同的 object goals。Trial 长达 32k steps。
**Metrics**：
- Success Rate (SR)
- SPL (Success weighted by Path Length)：
  $$ SPL = \frac{1}{N} \sum_{i=1}^N S_i \cdot \frac{L_i^*}{\max(L_i, L_i^*)} $$
  其中 $N$ 是 episodes 数量，$S_i$ 是二值成功标志，$L_i$ 是 agent 实际走过的路径长度，$L_i^*$ 是最短路径长度。SPL 惩罚了瞎逛即使最终找到了目标。

**核心结果**：
- Memo 用了 **8x 更少的 tokens**，却取得了比 FCT 高 **7.5% SR** 和 **2.5% SPL** 的成绩。
- **Compute 效率**：GPU Memory 51.8 MB vs FCT 的 546.5 MB (10x 降低)；Model FLOPs 17.61 MFLOPs vs 74.49 MFLOPs (4.2x 降低)；Latency 5.3 ms vs 10.1 ms (2x 降低)。
- **Intuition 构建**：为什么压缩了信息反而性能更好？因为 FCT 会被海量的 irrelevant visual frames 干扰。Attention 机制虽然强大，但在没有强 inductive bias 的情况下，它会被 noise 分散注意力。Memo 的 bottleneck 强制模型丢弃 noise，这类似于一种 implicit regularization。

### 4.2 Streaming Evaluation (关键泛化能力)
在 Inference 时，如果 context 无限增长，终究会 OOM。Streaming 设定下，只保留最近 $T$ 个 KV cache elements。
- FCT 在 $T=6k$ 时开始 Streaming，性能急剧下降。因为它丢失了早期的探索信息，在 unseen 场景中立刻 "失忆"。
- **Memo 的惊艳表现**：Memo 在 $T=1024$（对应 6k env steps）时开始 Streaming，不仅没有性能下降，甚至 ICL 趋势还在上升！
- **深度推测**：Memo 的 summary tokens 扮演了类似 StreamingLLM 中 "Attention Sinks" 的角色。但比 Attention Sinks 更高级的是，Summary tokens 携带了高度浓缩的 semantic memory。当 recent observations 被丢弃时，Summary tokens 依然锚定了 agent 对整个环境的结构化认知。这也说明模型学会了将 "长期记忆" 和 "短期工作记忆" 解耦。

### 4.3 T-Maze (Adversarially Long-Context Task)
T-Maze 是一个极端测试：timestep 0 给一个 clue (左或右)，然后走过 2000 步的空走廊，最后根据 clue 做决策。
- RMT 需要 10x 的训练步数才能偶尔达到 reward 1.0，且极不稳定。
- Memo 和 FCT 都能较好解决。
- **Intuition**：这证明了 Summary Accumulation 的 skip-connection 效应。RMT 在这种任务上几乎是不可能收敛的，因为 2000 步的 BPTT 对 RNN 式的 memory 更新是毁灭性的。

### 4.4 Ablation: Summary Length $l_{sum}$
测试了 16, 32, 64。
- 32 最好，64 最差。
- **推测**：64 造成了 compression ratio 过低（$256/64 = 4\times$），模型倾向于在 summary tokens 中直接 copy raw features，而不是进行 high-level abstraction。这导致 summary tokens 之间出现大量 redundant information，降低了 signal-to-noise ratio，并使得模型更容易 overfit 到 training context length。这说明 "过度压缩"（16）虽然会丢失信息，但 "轻度压缩"（64）反而会破坏 implicit regularization 的效果。

## 5. 联想与未来方向

基于这篇 paper，我有以下几个维度的联想：

1. **Memory Consolidation 睡眠机制**：Memo 目前是线性累积 summary tokens 的，如果 trial 极长（比如百万步），summary tokens 本身也会爆炸。Paper 在 Limitations 中提到了这一点。未来可以引入 hierarchical 的 memory consolidation，比如每 10 个 summary tokens 再压缩成 1 个 "meta-summary"。这类似于人类大脑在睡眠期间将海马体的短期记忆转移到大脑皮层变成长期记忆的过程。

2. **与 Transformer-XL 的关联**：Transformer-XL 也是为了处理长序列，它的机制是在 inference 时保留前一段的 hidden states 作为 extended context，但不做压缩。Memo 实际上就是 learnable compression 版本的 Transformer-XL。Transformer-XL 在 RL 中也曾被尝试过（Tr-XL），但表现不佳。Memo 的成功说明，在 RL 中，"压缩" 这个动作本身比 "单纯缓存" 更有价值。

3. **LLM 领域的类似发展**：近期 LLM 领域有 Infini-attention (Google) 和 Activation Beacon，都在做类似的事情。Memo 的独特之处在于它强调了 RL 任务中 gradient propagation 的重要性。如果把 Memo 的思想用到 LLM Agent（如 Voyager 或 DEPS）上，处理极其漫长的交互历史，应该会有显著效果。

4. **Credit Assignment 的本质**：Memo 的成功本质上是因为它重塑了 credit assignment 的路径图。在 FCT 中，t=1 的 observation 要影响 t=10000 的 action，信息需要穿过 10000 个 residual blocks。而在 Memo 中，信息被压缩进 summary，然后通过 attention 直接跳跃传播，这改变了 gradient flow 的拓扑结构，使得 long-term dependency 的学习变得 tractable。

5. **Predictive Coding 的视角**：Memo 学到的 summary tokens 到底是什么？我们可以用 probing 实验去分析。我猜测它学到了类似于 "Cognitive Map" 的东西——不是具体的像素，而是物体间的 topological relationship 和 agent 的探索边界。如果能加上一个 auxiliary loss，强制 summary tokens 预测未来的 visual embedding（类似 JEPA 或者 DreamerV3 的 world model），可能会进一步提升 sample efficiency。

## 6. 总结

Memo 是一个非常 elegant 的工作。它没有发明复杂的新的 neural network module，而是巧妙地结合了 segment tokenization、attention masking 和 end-to-end RL training。它证明了在 Embodied RL 中，**"遗忘" 是 "学习" 的必要条件**。通过强制压缩，模型被迫提取高维语义，这不仅解决了工程上的 memory 瓶颈，更在算法层面上带来了更好的 generalization 和 ICL 能力。这种将 NLP 中 context processing 的成熟 idea 适配到 RL 并解决其 unique challenges 的思路，非常值得借鉴。

**Reference Links:**
- Paper Github (Early Release): [https://github.com/gunshi/memo](https://github.com/gunshi/memo) 或 [https://github.com/Memory-icrl/memo](https://github.com/Memory-icrl/memo)
- ReLIC (On-policy baseline): [https://arxiv.org/abs/2410.02751](https://arxiv.org/abs/2410.02751)
- AMAGO (Off-policy baseline): [https://openreview.net/forum?id=M6XWoEdmwf](https://openreview.net/forum?id=M6XWoEdmwf)
- Autocompressors (Inspiration): [https://aclanthology.org/2023.emnlp-main.232](https://aclanthology.org/2023.emnlp-main.232)
- Recurrent Memory Transformer (RMT): [https://arxiv.org/abs/2304.11062](https://arxiv.org/abs/2304.11062)
- Algorithm Distillation (Dark-Key-To-Door origin): [https://arxiv.org/abs/2210.14215](https://arxiv.org/abs/2210.14215)
- StreamingLLM (Attention Sinks concept): [https://arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)
- HSSD Dataset (Habitat Synthetic Scenes): [https://huggingface.co/datasets/hssd/hssd-data](https://huggingface.co/datasets/hssd/hssd-data)
- VC-1 (Visual Encoder): [https://arxiv.org/abs/2306.11345](https://arxiv.org/abs/2306.11345)
- DD-PPO (Distributed PPO): [https://arxiv.org/abs/1911.00357](https://arxiv.org/abs/1911.00357)
