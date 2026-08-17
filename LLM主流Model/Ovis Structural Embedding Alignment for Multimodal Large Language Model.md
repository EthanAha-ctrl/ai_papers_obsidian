---
source_pdf: Ovis Structural Embedding Alignment for Multimodal Large Language Model.pdf
paper_sha256: 3b099a451a95a615834d9f52811feaf333fb58d7dc026df6768e6374e2bab61c
processed_at: '2026-08-06T01:47:14-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Ovis 用人话版

## 一、先讲一个最直白的比喻

想象 LLM 是一个**只会读词卡的盲人学者**。它的大脑里有一本词典，你给它一个 word，它翻到那一页读出对应的解释向量，然后脑子里推理。

现在你想让它也能"看图"。问题来了：图不是词卡，图是连续的画。你怎么喂给它？

**LLaVA 这种做法**：把画塞进一个 MLP 黑盒，黑盒吐出一堆向量，硬塞到盲人学者面前说"你就当这些是词卡吧"。学者能凑合用，但总觉得这些"词卡"长得不像它熟悉的那种，得花大力气重新适应。

**Ovis 的做法**：给盲人学者**再发一本专用的"视觉词典"**，有 13 万页（K=2^17）。每幅小图先去查这本视觉词典，找到它和词典里每个"视觉词条"的相似度分布（softmax 出来的概率），然后把对应那几个词条的解释向量按相似度加权求和，得到一个"合成的词卡"。这个合成的词卡长得跟文字词卡一模一样的格式，学者用起来毫无违和感。

核心 insight 就这一句：**让 visual embedding 走和 textual embedding 完全同构的 "查词典" 流程，只是从 hard one-hot 查询变成 soft 概率查询**。

参考：LLaVA https://arxiv.org/abs/2304.08485

---

## 二、用一张表把"原来 vs 现在"摆出来

你看这个表就明白了：

| 步骤 | Textual 侧 | Visual 侧（旧：connector） | Visual 侧（新：Ovis） |
|------|-----------|------------------------|---------------------|
| 1. 输入形式 | "cat" 这个 word | patch 的 ViT 输出 $r_i$ | 同左 |
| 2. tokenize | 查 BPE 词表得到 one-hot $\mathbf{t}_i$ | 什么都不做，直接拿 $r_i$ | 用 $\mathbf{W}$ 算相似度，softmax 成 $\mathbf{v}_i \in \Delta^K$ |
| 3. lookup table | textual embedding table $\mathbf{E}_{\text{txt}} \in \mathbb{R}^{V \times d'}$ | 没有，用 MLP 投影 | visual embedding table $\mathbf{E}_{\text{vis}} \in \mathbb{R}^{K \times d'}$ |
| 4. embedding | $T_i = \mathbf{t}_i^\top \mathbf{E}_{\text{txt}}$（一行） | $V_i = \text{MLP}(r_i)$（黑盒） | $V_i = \mathbf{v}_i^\top \mathbf{E}_{\text{vis}}$（加权行） |
| 5. 词典训练 | 预训练时就在学 | MLP 是随机初始化现学的 | 端到端用 LLM loss 学 |

Ovis 这一列从第 2 行到第 4 行，**形式上和 textual 侧完全对称**，只是把 one-hot 换成 soft distribution。这就是 paper 标题 "Structural Embedding Alignment" 的字面意思。

---

## 三、为什么这个想法"对"——build intuition 的三个角度

### 角度 1：inductive bias 的对称性

LLM 内部所有 attention 计算其实都在做一件事：**给定一个 query 向量，去一堆 key 里找相关的，加权 value**。textual token 是这套机制的"原生公民"，因为它本身就是从 lookup table 里捞出来的。

MLP connector 出来的 visual embedding 没有这种"原生 lookup 血统"，它是 arbitrary continuous vector，attention 机制能用它，但它对 LLM 来说像个"外来物种"。

Ovis 强制 visual embedding 也是从 lookup table 加权得到的，这就让 visual embedding 在 representation space 里的**几何结构**与 textual embedding 一致 —— 它们都落在 embedding table 行向量张成的凸包里。这种"几何同构"让 LLM 的 attention / FFN 这些已经预训练好的模块不需要做大的 adaptation 就能吃 visual embedding。

### 角度 2：和 self-attention 是同一个运算

你可以把 Ovis 的 visual embedding 那一步**看作一次特殊的 attention**：

$$
\mathbf{V}_i = \text{softmax}(\mathbf{W}\mathbf{r}_i)^\top \mathbf{E}_{\text{vis}}
$$

- Query：$\mathbf{r}_i$（patch $i$ 的 ViT 表示，$\mathbb{R}^d$）
- Keys：$\mathbf{W}$ 的每一行（K 个 visual words 的 prototypes，$\mathbb{R}^d$）
- Values：$\mathbf{E}_{\text{vis}}$ 的每一行（K 个 visual words 的 LLM-space embeddings，$\mathbb{R}^{d'}$）
- 输出：$\mathbf{V}_i \in \mathbb{R}^{d'}$

这里 $\mathbf{W} \in \mathbb{R}^{K \times d}$，$\mathbf{E}_{\text{vis}} \in \mathbb{R}^{K \times d'}$。**K 是 visual vocabulary size，paper 取 $K = 2^{17} = 131{,}072$**，d 是 ViT 的 hidden dim（CLIP-ViT-L/14 是 1024），$d'$ 是 LLM 的 hidden dim（Qwen1.5 / Llama3 都是 4096）。

这就意味着 Ovis 的"visual embedding 生成"其实是 **一次 patch-local 的、与 LLM 内部 attention 同构的运算**。LLM 看到这种 visual embedding 就像看到自己人一样亲切。

参考：Vaswani attention 原文 https://arxiv.org/abs/1706.03762

### 角度 3：polysemy 的保留

一个 patch 里可能同时有"红色 + 圆形 + 苹果 logo"。如果用 hard argmax 查 visual 词典，只能选一个 visual word，丢掉其他语义。Ovis 用 soft distribution $\mathbf{v}_i$，所有相关 visual words 都参与加权，最终 embedding 是它们的 convex combination。

写得更直观：

$$
\mathbf{V}_i = \mathbb{E}_{k \sim \mathbf{v}_i}[\mathbf{e}_k]
$$

这里 $\mathbf{v}_i$ 是个概率分布（在 K 维 simplex 上），$k$ 是按这个分布抽样的 visual word 索引，$\mathbf{e}_k$ 是对应 embedding。这个 expectation 形式说明：**visual embedding 可以理解为"从一个离散 visual 词典里按分布采样得到 embedding 的期望"**。

textual token 是这个分布的极限情形（temperature → 0，退化成 one-hot），所以 textual 是 Ovis visual 的特例。**Ovis 把 textual embedding 推广到 soft 版本**，这种数学上的包含关系非常优雅。

---

## 四、训练：分三步慢慢解锁

直接全开训会崩 —— visual 词典是随机初始化的，LLM 也会被乱梯度毁掉。所以 paper 用了三阶段：

**Stage 1：只训 visual 词典末梢**
- 冻结 LLM 全部，冻结 ViT 除最后一个 block
- 训：$\mathbf{W}$、$\{\mathbf{e}_k\}$、ViT last block
- 数据：COYO-10M 的 caption（"图：xxx"格式）
- 目的：让 visual 词典先有个靠谱初值，能"认得"基本视觉概念
- batch size 8192（因为 caption 任务样本简单）

**Stage 2：解锁整个 ViT**
- 还冻结 LLM
- 训：$\mathbf{W}$、$\{\mathbf{e}_k\}$、整个 ViT
- 数据：ShareGPT4V-Pretrain + in-house description（多轮对话式描述）
- 目的：让 ViT 调整自己的输出，适配"被查 visual 词典"这个新机制（CLIP 原本输出是为 contrastive loss 优化的，现在要服务 generation）

**Stage 3：解锁 LLM**
- 全开训
- 数据：LLaVA-Finetune 多模态 instruction
- learning rate 降到 1e-5 ~ 2e-5（保护 LLM 预训练权重）
- 目的：让 LLM 学会消费这种新格式 visual embedding，完成最终多模态对齐

直觉就是：**先把 visual 这一头摆好，再把 visual 和 textual 接到一起，最后整体 fine-tune**。这个顺序跟 LLaVA 系列的 "pretrain connector → instruction tuning" 是同源的，只是 Ovis 多了一个"先让 visual vocab 收敛"的预热阶段。

参考：ShareGPT4V https://arxiv.org/abs/2311.12793 ; LLaVA-1.5 https://arxiv.org/abs/2310.03744

---

## 五、实验数字最关键的几个点

### 5.1 同条件对照实验（Table 3，最重要）

为了排除"是不是数据多 / backbone 好造成的提升"，作者做了一个非常干净的对照：
- 同一个 LLM：Qwen1.5-7B-Chat
- 同一个 ViT：CLIP-ViT-L/14@336px
- 同一份训练数据
- **connector baseline 的 MLP hidden size 设成 = Ovis 的 K = 131,072，保证参数量严格相等**

结果：

| 指标 | Connector | Ovis | 提升 |
|------|----------|------|------|
| MMStar | 41.1 | 44.3 | +7.8% |
| MMBench-EN | 71.0 | 75.1 | +5.8% |
| MMBench-CN | 65.2 | 70.2 | +7.7% |
| MMMU-V | 34.8 | 39.7 | **+14.1%** |
| MMMU-T | 33.8 | 37.7 | +11.5% |
| MathVista | 36.3 | 41.4 | **+14.0%** |
| MME | 1757 | 1882 | +7.1% |
| HallusionBench | 54.0 | 56.4 | +4.4% |
| RealWorldQA | 56.1 | 60.0 | +7.0% |
| **平均** | — | — | **+8.8%** |

**唯一变量就是"visual embedding 怎么生成"，平均涨 8.8%**。这相当于免费升级，参数和数据都一样。

特别注意 MMMU（大学水平多学科推理）和 MathVista（视觉数学推理）涨得最多，+14%。这两个任务都需要**精确读图 + 知识联动**，最能体现 visual representation 质量。结构化 visual embedding 让 LLM 能更"精确地"使用视觉信息，不是泛泛地"看到大概"，而是"看到结构化的概念组合"。

### 5.2 跟闭源模型对比（Table 1 & 2）

Ovis-14B（开源！）在多个 benchmark 上**超过 Qwen-VL-Plus**（阿里自己的闭源商用模型），RealWorldQA 上 62.7 甚至**超过 GPT4V 的 61.4 和 Qwen-VL-Max 的 61.3**。

这件事挺震撼的：RealWorldQA 是 1080P 高分辨率真实世界图像 benchmark，Ovis 只用 336px ViT，没有用 LLaVA-Next 的 dynamic high resolution，也没有用 Mini-Gemini-HD 的双 encoder，结果还是 SOTA。这说明**结构化 visual representation 的质量提升能在某种程度上补偿分辨率不足**。

### 5.3 词典稀疏性（Appendix E）

用 10K 张 ImageNet 图统计 visual token $\mathbf{v}_i$ 的稀疏性：
- 只有 **0.22%** 的概率值 > 1e-4
- 也就是 K=131,072 个 visual words 里，每个 patch 实际只激活几十个

这件事有两个意思：
1. **学出来的 visual 词典是稀疏激活的**，跟 textual 词典一样（一句话里每个位置只激活一个 word，Ovis 是 soft 但稀疏）
2. **推理时可以走 sparse implementation 大幅加速**（paper 没做，但这是个明显的工程优化方向）

这跟 MoE 的稀疏 routing、跟 sparse retrieval (SPLADE) 的稀疏激活是同一类现象，都说明"大容量 + 稀疏激活"是 efficient representation 的普遍模式。

参考：Mixture of Experts https://arxiv.org/abs/1701.06538 ; SPLADE https://arxiv.org/abs/2107.05720

---

## 六、为什么不用 VQ-VAE 那一套

很多人第一反应："这不就是 VQ-VAE 吗？" 其实差很远。

**VQ-VAE 路线**：
- 用 argmax 硬量化 continuous latent 到 discrete code
- 需要 commitment loss（让 encoder 输出靠近 codebook entries）
- 需要 reconstruction loss + decoder（重建图像）
- 有 codebook collapse 问题（很多 codes 死掉），要 EMA 更新或 reset
- straight-through estimator 只是近似梯度，丢失信息

放到 MLLM 里有两个大麻烦：
1. **需要额外 image reconstruction decoder 和 loss**，与 LLM 的 textual next-token prediction 目标不一致，任务耦合
2. **hard quantization 丢梯度信息**，在 visual-language alignment 阶段尤其伤

**Ovis 路线**：
- 完全不做 quantization，也不做 reconstruction
- $\mathbf{v}_i$ 是 soft distribution，梯度直接从 LLM cross-entropy 反传到 $\mathbf{W}$、到 ViT、到 visual embedding table
- "Visual words" 的语义完全由 **LLM downstream 任务驱动 emerge** —— 这是一种 task-driven, language-grounded visual vocabulary
- 没有 codebook collapse 问题，因为所有 codes 都通过 softmax 接收梯度，只要某个 code 对某个 patch 有非零概率就会更新

简短地说：**VQ-VAE 是"为了重建而量化"，Ovis 是"为了 LLM 消费而 soft 索引"**，目标不同，机制也不同。

参考：VQ-VAE https://arxiv.org/abs/1711.00937 ; VQGAN https://arxiv.org/abs/2012.09841 ; BEIT https://arxiv.org/abs/2106.08254

---

## 七、和 VW-LMM-PIF 的微妙差异

VW-LMM-PIF (arXiv 2403.07720) 也用了 "linear head 把 visual token 映射到词典" 的想法，看起来很像 Ovis，但有两个关键不同：

1. **它直接复用 LLM 的 textual embedding table 当 visual 词典**。这意味着 visual 和 textual 强行共享同一个词义空间，但视觉概念（颜色、纹理、形状）和语言概念（"苹果"、"自由"）本质不在一个 space 里，强行共享可能有干扰。
   
   Ovis 用**独立的 visual embedding table**（K 维独立于 V），可以学视觉特有的 primitives。

2. **它的 head 只在 vision data 上 distill 训练**，没有 LLM 的 gradient signal。
   
   Ovis 的 $\mathbf{W}$ 和 $\mathbf{E}_{\text{vis}}$ **通过 LLM 的 generation loss 端到端训练**，意味着 visual vocabulary 是被"什么对 LLM 有用"反向塑造的，不是被"什么能重建图像"塑造的。

这两点合起来：**Ovis 的 visual vocabulary 是 LLM-grounded 的，VW-LMM-PIF 的 visual vocabulary 是 vision-intrinsic 的**。前者更适配 MLLM 任务。

参考：VW-LMM-PIF https://arxiv.org/abs/2403.07720

---

## 八、把这件事放进更大的图景

### 8.1 这是 "tokenizer 对齐" 思潮的一部分

NLP 里 BPE / SentencePiece 解决了"怎么把文本切成离散单元"的问题。多模态里 visual tokenizer 一直没有定论：
- VQ-VAE 路线：硬量化，服务 generation
- BEIT 路线：masked prediction，service 自监督
- Connector 路线（LLaVA）：完全不 tokenize，连续投影
- **Ovis 路线：soft tokenize，服务 LLM consumption**

Ovis 提供了一种"既离散又可微"的中间方案，在 tokenizer 设计谱系上占了一个新位置。

### 8.2 跟 retrieval-augmented generation 的隐秘关联

你可以把 Ovis 看成 **implicit retrieval**：每个 patch 是一个 query，visual vocab 是 document store，retrieved embedding 的加权和是 result。LLM 后续 attention 是在这个 retrieved representation 上做 reasoning。这跟 RAG 的 "retrieve-then-read" 是同一个范式，只是 retrieval 步骤内化进 model 里，可微分，端到端训练。

### 8.3 跟 Product Quantization / Sparse Retrieval 的关联

PQ 把向量切成多个 sub-quantizer 的组合，SPLADE 把 query/doc 表示成 sparse vocabulary 上的权重。Ovis 是 **learned, soft, single-codebook retrieval**。这种 framing 让 MLLM 的 visual processing 可以借鉴 IR 领域几十年积累的 sparse / hierarchical / adaptive 技术栈。

### 8.4 跟 Mixture of Experts 的同构

每个 patch 通过 softmax 在 K 个 visual words 里稀疏激活几十个，这跟 MoE 的 "router 选 top-k experts" 几乎是同一个 pattern：
- MoE：router 把 token 分给几个 expert FFN
- Ovis：softmax 把 patch 分给几个 visual word embeddings

差异只在 MoE 的 experts 是 FFN，Ovis 的 experts 是 embedding 行向量。两者的"稀疏激活 + 大容量"思想完全同源。这也暗示 Ovis 可以直接借用 MoE 的 top-k 加速、load balancing loss、expert routing 分析工具。

参考：GShard https://arxiv.org/abs/2006.16668 ; Switch Transformer https://arxiv.org/abs/2101.03961

---

## 九、可能延伸的方向（hallucination 区）

Paper 自己承认的限制：没做高分辨率、没做多图、visual vocab 是 static 的。我顺着往下联想几个明显方向：

1. **Hierarchical visual vocabulary**：两层 codebook，粗粒度（物体类别级，K=1K）+ 细粒度（纹理细节级，K=130K），类似 hierarchical softmax 的思路。粗 codebook 解决"是什么"，细 codebook 解决"长什么样"。

2. **Dynamic / MoE-style visual vocab**：根据 image resolution / task 复杂度动态激活一部分 vocab。低分辨率图只用小 vocab，高分辨率图用大 vocab，实现自适应计算。

3. **Cross-modal vocabulary sharing**：让一部分 visual words 与 textual words 在 embedding space 上做 soft alignment（比如用 contrastive auxiliary loss），实现 concept-level grounding。比如 visual word #1234 和 textual word "red" 共享 embedding 子空间。

4. **Disentangled visual words**：加 auxiliary loss 鼓励 visual vocabulary 分解成 color / shape / texture / object 等独立 axis，提升可解释性。类似 β-VAE 的 disentanglement 思路。

5. **Generative direction**：把 visual embedding table 同时用作 image generation 的 decoder codebook，让 MLLM 变 any-to-any 模型。类似 SEED-X 的思路，但用 Ovis 的 soft vocab 替代 SEED 的 hard discrete tokens，可能训练更稳。

6. **Visual vocab 的 continual learning**：训练后 vocab 固定，但实际部署中新视觉概念不断出现（新产品、新风格）。能否设计 online update 机制让 vocab 增量扩展，类似 lifelong learning？

7. **Visual vocab 的可解释性分析**：每个 visual word $\mathbf{e}_k$ 对应什么？可以用 image retrieval 找出激活它的 top patches，类似 analyzing MoE experts。这能给 MLLM 的视觉理解打开"黑盒"。

参考：SEED-X https://arxiv.org/abs/2404.14396 ; β-VAE https://arxiv.org/abs/1804.03599

---

## 十、最后用一句话总结

**Ovis 把 MLLM 的 visual embedding 从 "MLP 黑盒投影" 改成 "softmax 加权查 visual 词典"，让 visual 和 textual 走完全同构的 "lookup-table-based structured embedding" 路径，用 LLM 的 generation loss 端到端训出一个 task-grounded 的视觉词典 —— 在参数量、数据、backbone 严格相同的对照下平均提升 8.8%，在 MMMU/MathVista 这种需要精细视觉推理的任务上提升超 14%**。

直觉上为什么 work：**让 LLM 不需要"学习"怎么处理异质的 visual embedding，因为它收到的 visual embedding 和它熟悉的 textual embedding 在几何结构上同构，attention / FFN 这些预训练模块直接复用即可**。这是用结构先验替代数据驱动的 adaptation，本质上是在 MLLM 这个层面复现了 "好的 inductive bias 比更多参数更值钱" 这条老道理。

---

参考链接汇总：
- Ovis GitHub: https://github.com/AIDC-AI/Ovis
- Ovis dataset: https://huggingface.co/datasets/AIDC-AI/Ovis-dataset
- LLaVA: https://arxiv.org/abs/2304.08485
- LLaVA-1.5: https://arxiv.org/abs/2310.03744
- CLIP: https://arxiv.org/abs/2103.00020
- Attention is All You Need: https://arxiv.org/abs/1706.03762
- VQ-VAE: https://arxiv.org/abs/1711.00937
- VQGAN: https://arxiv.org/abs/2012.09841
- BEIT: https://arxiv.org/abs/2106.08254
- VW-LMM-PIF: https://arxiv.org/abs/2403.07720
- ShareGPT4V: https://arxiv.org/abs/2311.12793
- Qwen-VL: https://arxiv.org/abs/2308.12966
- MMStar: https://arxiv.org/abs/2403.20330
- MMMU: https://arxiv.org/abs/2311.16502
- MathVista: https://arxiv.org/abs/2310.02244
- RealWorldQA (Grok-1.5V blog): https://x.ai/blog/grok-1.5v
- Switch Transformer: https://arxiv.org/abs/2101.03961
- GShard: https://arxiv.org/abs/2006.16668
- SPLADE: https://arxiv.org/abs/2107.05720
- SEED-X: https://arxiv.org/abs/2404.14396
- β-VAE: https://arxiv.org/abs/1804.03599

---

# Ovis: Structural Embedding Alignment for MLLM 深度解读

## 一、核心 Insight：Embedding 策略的不对称性

Karpathy 你看，这篇 paper 的切入点其实是非常微妙、非常 "Karpathy-flavored" 的一个观察。当前 MLLM 里其实存在一个被大家习以为常但本质上是 "不对称" 的设计：

**Textual 侧**：每个 word token $t_i$ 通过 one-hot 形式从 LLM 的 textual embedding look-up table $\mathbf{E}_{\text{txt}} \in \mathbb{R}^{V \times d'}$ 中**索引**出一行 $T_i = \mathbf{E}_{\text{txt}}[t_i]$。这是一种**结构化、离散、symbolic** 的 embedding 过程，词汇表大小 $V$ 通常是 32K~128K 量级。

**Visual 侧**：vision encoder (CLIP-ViT) 直接输出一个连续向量 $r_i \in \mathbb{R}^d$，这个向量是经过 attention layers 反复混合得到的、没有任何"离散锚点"的**连续、holistic** 表示。然后用一个 connector（MLP/Linear）硬投影到 $d'$ 维塞进 LLM。

这种 misalignment 在 LLaVA 这一代架构里通过 MLP connector 被"掩盖"住了，paper 的核心提问是：

> 如果把 visual embedding 也变成 "lookup-table-based + 结构化" 的形式，能不能让两边的 representation 在 inductive bias 层面对齐，从而让 LLM 更"舒服"地处理 visual token？

这其实呼应了你早期关于 "soft attention 是 differentiable lookup" 的说法 —— Ovis 就是把这件事从 "implicit via MLP" 显式化成 "explicit via embedding table + softmax weighting"。

参考：LLaVA https://arxiv.org/abs/2304.08485 ; CLIP https://arxiv.org/abs/2103.00020

---

## 二、架构解析：从 visual patch 到 structural embedding

### 2.1 整体 pipeline（对应 Figure 3）

```
Image T ∈ R^{C×W×H}
   → split into patches {P_i} , i=1..n , n = ⌈W/w⌉⌈H/h⌉
   → ViT backbone g_θ  →  {r_i ∈ R^d}_{i=1..n}      (连续 visual token)
   → Linear head W ∈ R^{K×d}  + softmax  →  v_i ∈ Δ^K  (probabilistic token, K simplex)
   → visual embedding table {e_k ∈ R^{d'}}_{k=1..K}
   → V_i = Σ_k v_{i,k} · e_k = E_{k~v_i}[e_k]      (结构化 visual embedding)
   → concat with textual embeddings → LLM f_φ  →  output tokens {o_j}
```

### 2.2 公式逐项拆解

**公式 (2)：Probabilistic Visual Token**

$$
\mathbf{v}_i = \text{softmax}(\mathbf{W}\mathbf{r}_i), \quad \mathbf{W} \in \mathbb{R}^{K \times d}
$$

变量含义：
- $\mathbf{r}_i \in \mathbb{R}^d$：第 $i$ 个 visual patch 经过 ViT 后的连续表示，$d$ 是 ViT 的 hidden dim（CLIP-ViT-L/14 是 1024）
- $\mathbf{W} \in \mathbb{R}^{K \times d}$：linear head 矩阵，**K 是 visual vocabulary size**，paper 取 $K = 2^{17} = 131{,}072$（与 LLM 的 textual vocab 量级对齐，这是一个很有 signal 的设计 —— 让 visual "词典" 在容量上不输 textual 词典）
- $\mathbf{v}_i \in \Delta^K$：落在 $K-1$ 维 probability simplex 上，是 visual token 在 visual vocabulary 上的一个分布
- softmax 内的 $\mathbf{W}\mathbf{r}_i$ 其实就是 $\mathbf{r}_i$ 与 visual vocabulary 中每个 prototype 的 inner product（normalized similarity）

这一步的 intuition 很关键：**它把 visual patch 变成了一个 "soft pointer"，指向 visual vocabulary 中所有可能的 "visual words"**。一个 patch 可能同时含有"边缘"、"红色"、"圆形"等多重语义，hard argmax 会丢掉这种 polysemy，soft distribution 保留了。

**公式 (3)：Visual Embedding Lookup**

$$
\mathbf{V}_i = \sum_{k=1}^{K} v_{i,k} \, \mathbf{e}_k \in \mathbb{R}^{d'}, \quad \text{等价地} \quad \mathbf{V}_i = \mathbb{E}_{k \sim \mathbf{v}_i}[\mathbf{e}_k]
$$

变量含义：
- $v_{i,k}$：$\mathbf{v}_i$ 的第 $k$ 个分量（patch $i$ 与 visual word $k$ 的相似度/概率）
- $\mathbf{e}_k \in \mathbb{R}^{d'}$：visual embedding table 的第 $k$ 行，$d'$ 设成与 textual embedding table 一致（Qwen1.5 是 4096，Llama3 是 4096）
- $\mathbf{V}_i$：最终塞进 LLM 的 visual embedding

把它写成 expectation $\mathbb{E}_{k \sim \mathbf{v}_i}[\mathbf{e}_k]$ 这种形式非常优雅 —— 这意味着把 visual embedding 视为**从一个 discrete visual vocabulary 中按分布 $\mathbf{v}_i$ 采样得到的 embedding 的期望**。这与 textual token 从 one-hot 分布中"采样"（其实是确定索引）得到 embedding 是同构的：one-hot 是 $\mathbf{v}_i$ 的极限情形（temperature → 0）。

### 2.3 为什么这个对称性 "美" 且 "对"

你可以在脑海中把两边并排画出来：

| 模态 | Token 形式 | Lookup Table | 最终 Embedding |
|------|----------|-------------|---------------|
| Textual | one-hot $\mathbf{t}_i \in \{0,1\}^V$ | $\mathbf{E}_{\text{txt}} \in \mathbb{R}^{V \times d'}$ | $T_i = \mathbf{t}_i^\top \mathbf{E}_{\text{txt}}$ |
| Visual (Ovis) | probabilistic $\mathbf{v}_i \in \Delta^K$ | $\mathbf{E}_{\text{vis}} \in \mathbb{R}^{K \times d'}$ | $V_i = \mathbf{v}_i^\top \mathbf{E}_{\text{vis}}$ |

两边都是 **"lookup table 行的加权组合"**，只是权重从 hard one-hot 变成 soft softmax distribution。LLM 后续的 attention 机制对这两种 embedding 的"形状"是完全一致的，不存在 modality-specific 的特殊处理。这种结构对齐让 LLM 不需要"学会"怎么处理一个异质来源的 embedding。

参考：VQ-VAE https://arxiv.org/abs/1711.00937 ; BEIT https://arxiv.org/abs/2106.08254

---

## 三、与 VQ-VAE / BEIT / VW-LMM-PIF 的关键差异

这一点 paper 讲得不够透，但其实是最值得 build intuition 的地方。

### 3.1 VQ-VAE 路线（hard discretization）

VQ-VAE 用 $\arg\max$ 把 continuous latent 硬量化成 discrete code，需要：
1. **commitment loss** 让 encoder 输出靠近 codebook entries
2. **reconstruction loss** 让 decoder 能从 discrete codes 重建图像
3. **codebook collapse 问题**（很多 codes 死掉），需要 EMA 更新或 reset

这套机制在 generation 任务里很成功，但放到 MLLM 里有两个麻烦：
- 需要额外的 image reconstruction decoder 与 loss，与 LLM 的 textual next-token prediction 目标不一致
- hard quantization 在 visual-language alignment 阶段会丢梯度信息（straight-through estimator 只是近似）

### 3.2 Ovis 路线（soft probabilistic indexing）

Ovis 完全不做 quantization，也不做 reconstruction：
- $\mathbf{v}_i$ 是 soft distribution，梯度可以直接从 LLM 的 cross-entropy loss 反传到 $\mathbf{W}$、到 ViT、再到 visual embedding table $\{\mathbf{e}_k\}$
- $\mathbf{W}$ 和 $\{\mathbf{e}_k\}$ 是通过 **LLM 的 textual generation loss** 端到端训练的，没有任何视觉 reconstruction supervision
- "Visual words" 的语义完全由 LLM downstream 任务驱动 emerge 出来 —— 这是一种 **task-driven, language-grounded visual vocabulary**

这非常像把 visual encoder 和 LLM 之间的接口从 "MLP black box" 换成 "可解释的、结构化的、可微的 vocabulary retrieval"。

### 3.3 与 VW-LMM-PIF [62] 的差异

VW-LMM-PIF 也用了一个 linear head 把 visual token 映射到 textual vocab，但有两个关键不同：
- 它**直接复用 LLM 的 textual embedding table** 做 visual lookup，导致 visual 和 textual "词义" 强行共享，可能存在干扰
- 它的 head 仅在 vision data 上 distill 训练，没有 LLM 的 gradient signal

Ovis 用 **独立的 visual embedding table**（$K$ 维独立于 $V$），并且通过 LLM 的 generation loss 训练 head 和 table。这意味着 visual vocabulary 可以学到与 textual 不同的、视觉特有的 "primitive concepts"（如颜色、纹理、形状、空间布局），同时仍然能被 LLM 的 attention 机制无缝消费。

参考：VW-LMM-PIF https://arxiv.org/abs/2403.07720

---

## 四、训练策略：三阶段 curriculum

### 4.1 阶段拆解

| Stage | Trainable | Frozen | 数据 | 目标 |
|-------|----------|--------|------|------|
| Stage 1 | $\mathbf{W}$、$\{\mathbf{e}_k\}$、ViT last block | LLM、ViT 其余 | COYO-10M caption | 建立 visual vocabulary 的初值 |
| Stage 2 | $\mathbf{W}$、$\{\mathbf{e}_k\}$、ViT 全部 | LLM | ShareGPT4V-Pretrain + in-house description | 让 visual encoder 适配 Ovis-style tokenization |
| Stage 3 | 全部参数 | 无 | LLaVA-Finetune multimodal instruction | 真正的多模态 instruction tuning |

这个 curriculum 的设计直觉：
- **Stage 1** 只训 visual 侧的最末端，避免破坏 LLM 的语言能力；用 caption 这种"低带宽对齐"任务先让 visual table 收敛到一个合理的初始分布
- **Stage 2** 解锁 ViT 让 visual representation 也适配 Ovis 的 lookup 机制（注意：CLIP-ViT 原本的输出是为 contrastive loss 优化的，要重新调整以服务 generation）
- **Stage 3** 解锁 LLM 做最终的多模态对齐，相当于让 LLM 学会"消费"这种新格式 visual embedding

### 4.2 训练超参（Table 4 关键信息）

- batch size: Stage 1 是 8192（很大！caption 任务 sample 简单），Stage 2/3 是 1024
- learning rate: 1e-4 → 1e-4 → 2e-5 (Qwen) / 1e-5 (Llama3)，Stage 3 把 LLM 解锁后 lr 显著减小，避免破坏预训练权重
- DeepSpeed: 7B 用 zero2→zero3，14B 全程 zero3
- 硬件成本：7B/8B 全程 128 H100 上 15 小时；14B 是 37 小时

参考：DeepSpeed https://arxiv.org/abs/1911.02134 ; ShareGPT4V https://arxiv.org/abs/2311.12793

---

## 五、实验结果深度分析

### 5.1 主表 (Table 1 & 2) 关键观察

**7B tier 的 Ovis-Llama3-8B**：
- MMStar: 49.5 (vs LLaVA-Llama3-8B 46.1, LLaVA-Next-Mistral-7B 38.4)
- MMBench-EN: 77.4 (vs DeepSeek-VL-7B 73.8)
- MMMU-V: 44.7 (这一项 7B tier 里非常突出，比 LLaVA-Next 高 7+ 个点)

**14B tier 的 Ovis-Qwen1.5-14B**：
- MMStar: 48.5, MMBench-EN: 78.4, MMMU-V: 46.7
- **整体超过 Qwen-VL-Plus**（专有模型！），RealWorldQA 上 62.7 vs Qwen-VL-Max 61.3

特别有意思的是 **RealWorldQA** —— 这是一个 1080P 高分辨率图像的 benchmark，Ovis 只用 336px ViT，没有 dynamic high resolution（不像 LLaVA-Next）也没有 dual encoder（不像 Mini-Gemini-HD），却仍然 SOTA。这说明 structural alignment 带来的 representation 质量提升，能在某种程度上 compensate 分辨率的劣势。

### 5.2 Ablation（Table 3）—— 最有说服力的一组数据

为了隔离架构本身的贡献，作者把 Ovis-7B 和一个 connector-based baseline 做 controlled comparison：
- 同样的 Qwen1.5-7B-Chat
- 同样的 CLIP-ViT-L/14@336px
- 同样的训练数据
- connector 是 2-layer MLP，hidden size = Ovis 的 visual vocab size（即 $2^{17}$），**保证参数量严格相等**

结果：

| 指标 | Connector | Ovis | 提升 |
|------|----------|------|------|
| MMStar | 41.1 | 44.3 | **+7.8%** |
| MMBench (EN/CN) | 71.0 / 65.2 | 75.1 / 70.2 | +5.8% / +7.7% |
| MMMU (V/T) | 34.8 / 33.8 | 39.7 / 37.7 | **+14.1% / +11.5%** |
| MathVista | 36.3 | 41.4 | **+14.0%** |
| MME | 1757 | 1882 | +7.1% |
| HallusionBench | 54.0 | 56.4 | +4.4% |
| RealWorldQA | 56.1 | 60.0 | +7.0% |
| **平均** | — | — | **+8.8%** |

这是一个非常干净的对照：参数量相同、数据相同、backbone 相同，唯一变量是"visual embedding 怎么生成"。在 MMMU 和 MathVista 上 +14% 是非常大的 margin，说明结构化 visual representation 对**需要精细视觉推理的任务**帮助最大 —— 这些任务需要模型精确"读懂"视觉细节并和知识联动，structural alignment 让 visual embedding 更"可被 LLM reasoning"。

### 5.3 Sparsity 实验（Appendix E）

用 10K ImageNet 图像统计 $\mathbf{v}_i$ 的分布稀疏性：
- 只有 **0.22%** 的概率值超过 $10^{-4}$
- 也就是 $K=131{,}072$ 个 visual words 里，每个 patch 实际只激活几十个

这背后有一个非常 deep 的 intuition：**虽然 vocabulary 很大，但每个 patch 实际使用的 visual words 是 sparse 的**。这跟 textual token 的稀疏性（一个位置只有一个 word）异曲同工，但保留了 multi-modal polysemy 的 soft 表达。这也意味着 Ovis 在推理时可以走 sparse implementation 大幅加速（paper 没明说，但这是一个明显的工程优化方向，类比 MoE 的 sparse routing）。

参考：Mixture-of-Experts 思路 https://arxiv.org/abs/1701.06538

---

## 六、Intuition Building：为什么这个架构 work

我尝试从几个 angle 帮你 build 一个连贯的 mental model：

### 6.1 Information Bottleneck 视角

MLP connector 是一个**自由形式的信息瓶颈**，它没有 strong inductive bias，可以学成任何 mapping —— 这意味着它要靠海量数据自己"发现"什么 visual feature 对 LLM 重要。Ovis 强加了一个 "vocabulary retrieval" 的结构先验：**visual 信息必须先分解成 K 个 primitives 的组合**。这个先验把搜索空间从 "任意 continuous mapping" 缩小到 "K 个离散 semantic anchors 的加权和"，相当于一个 structured bottleneck，data efficiency 更高，generalization 也更好（这与 MMMU 上 +14% 的一致 —— 知识泛化任务最受益于 strong inductive bias）。

### 6.2 Gradient Flow 视角

在 connector-based 架构里，LLM 的 loss 通过 MLP 反传到 ViT，ViT 需要同时：(a) 提取视觉特征，(b) 适配 MLP 的 nonlinear mapping。这两个目标耦合在一起。

Ovis 把 (b) 拆成了 **"vocabulary assignment + embedding combination"** 两步，每一步都有清晰语义：
- $\mathbf{W}$ 学的是 "patch → visual word similarity"
- $\{\mathbf{e}_k\}$ 学的是 "visual word → LLM-compatible embedding"

梯度信号更"分工明确"，每个参数都有清晰的"职责"，类似 modular design 带来更好的 optimization landscape。

### 6.3 与 Soft Attention 的同构性

你可以把 Ovis 的 visual embedding step 看作一个**单次、patch-local 的 cross-attention**：
- Query：$\mathbf{r}_i$（patch 的 ViT 输出）
- Keys：$\mathbf{W}$ 的行（visual vocabulary 的 prototypes）
- Values：$\{\mathbf{e}_k\}$（visual embedding table 的行）
- $\mathbf{V}_i = \text{softmax}(\mathbf{W}\mathbf{r}_i)^\top \mathbf{E}_{\text{vis}}$

这与 transformer 内部的 attention 是**同一类操作**，只是 keys/values 是学出来的 visual vocabulary 而非 sequence tokens。这种"操作同构"意味着 LLM 的 attention layers 处理 Ovis 的 visual embedding 时，用的是和它内部 self-attention 同样的"语言"，从而降低了跨模态 integration 的难度。

### 6.4 与 Product Quantization / Sparse Retrieval 的关联

Ovis 的 visual tokenizer 让我想到 retrieval 里的 product quantization 和 sparse retrieval：
- PQ 把 vector 量化成多个 sub-quantizers 的组合
- Sparse retrieval (如 SPLADE) 把 query/doc 表示成 sparse vocabulary 上的权重

Ovis 类似一个 **learned, soft, single-codebook retrieval**，visual patch 是 query，visual vocabulary 是 document store，retrieved embedding 的加权和是 result。这种 framing 让 MLLM 的 visual processing 变成了 "retrieve-then-read" 的 implicit 形式。

参考：SPLADE https://arxiv.org/abs/2107.05720

---

## 七、Limitations 与未来方向

Paper 自己承认：
1. **没有 high-resolution 处理**（仅 336px），RealWorldQA 的好成绩其实暗示如果能加 high-res 会更强
2. **单图像训练**，multi-image 场景未优化
3. **Visual vocabulary 是 static 的**，整个 table 在训练后固定

我（hallucination 部分）觉得几个明显可拓展的方向：
- **Hierarchical visual vocabulary**：粗粒度 + 细粒度两层 codebook，类似 hierarchical softmax 的思路
- **Dynamic / MoE-style visual vocab**：根据 image resolution / task 动态激活一部分 vocab
- **Cross-modal vocabulary sharing**：让一部分 visual words 与 textual words 在 embedding space 上对齐（类似 concept-level grounding）
- **Disentangled visual words**：用 auxiliary loss 鼓励 visual vocabulary 分解成 color / shape / texture / object 等独立 axis，提升可解释性
- **Generative direction**：把这个 visual embedding table 用作 generation 的 decoder codebook，把 MLLM 变成 any-to-any 模型（类似 SEED-X https://arxiv.org/abs/2404.14396）

---

## 八、One-liner 总结

**Ovis 把 MLLM 的 visual embedding 从 "MLP 黑盒投影" 改成 "softmax-weighted vocabulary retrieval"，让 visual 和 textual 走同一种 "lookup-table-based structured embedding" 路径，用 LLM 的 generation loss 端到端训练出一个 task-grounded visual vocabulary —— 在严格 controlled comparison 下平均提升 8.8%，在 MMMU/MathVista 上提升超 14%。**

参考链接汇总：
- Paper GitHub: https://github.com/AIDC-AI/Ovis
- Dataset: https://huggingface.co/datasets/AIDC-AI/Ovis-dataset
- LLaVA: https://arxiv.org/abs/2304.08485
- VQ-VAE: https://arxiv.org/abs/1711.00937
- BEIT: https://arxiv.org/abs/2106.08254
- CLIP: https://arxiv.org/abs/2103.00020
- Qwen-VL: https://arxiv.org/abs/2308.12966
- MMStar: https://arxiv.org/abs/2403.20330
- MMMU: https://arxiv.org/abs/2311.16502
- MathVista: https://arxiv.org/abs/2310.02244
- VW-LMM-PIF: https://arxiv.org/abs/2403.07720
- ShareGPT4V: https://arxiv.org/abs/2311.12793
- HallusionBench: https://arxiv.org/abs/2310.07704
- DeepSpeed: https://arxiv.org/abs/1911.02134
