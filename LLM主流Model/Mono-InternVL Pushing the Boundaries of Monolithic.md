---
source_pdf: Mono-InternVL Pushing the Boundaries of Monolithic.pdf
paper_sha256: 2eeabeb7486fbc7f8f39a5bb74309ef8836367707d1ca416a696f2d12c687abf
processed_at: '2026-08-05T20:20:43-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Mono-InternVL

Andrej，好，咱换种方式聊。我尽量把这 paper 拆得 intuitive 一点，但公式和数据还得留——因为光靠嘴说不清 why it works。

---

## 一句话版

现有 MLLM 基本都是"ViT + LLM"的外挂组合（modular），这帮人想搞一个**单一 LLM 直接吃 image 和 text**的版本（monolithic），别人做这件事都翻车了——要么训练崩、要么 language 能力被 visual 训练毁掉。他们用了一个很朴素的 trick：**在 LLM 的每一层 FFN 旁边塞一个 visual expert，freeze 原始 LLM，只训这个新 expert**。结果 1.8B 参数打平 modular 的 InternVL-1.5，first token latency 还砍掉 67%。

就这么个事。

---

## 为什么 monolithic 这么难

先说 background。MLLM 两大流派：

**Modular**（LLaVA、InternVL、Qwen-VL 这类）：一个 pre-trained CLIP-ViT 当眼睛，一个 LLM 当脑子，中间用一个 projector 连起来。
- 好处：好训，performance 强
- 坏处：deployment 复杂，ViT 的 visual representation 有上限，两个模型 capacity 怎么平衡是个玄学

**Monolithic**（Fuyu、EVE、Emu3、Chameleon 这类）：把视觉功能直接塞进 LLM，一个 transformer 搞定一切。
- 好处：简单、deployment 友好、没有 encoder bottleneck
- 坏处：**训练特别难搞**

Monolithic 的两种 train 法都有坑：

**Native pre-training**（Emu3、Chameleon 的路子）——从零开始训 mixed-modality data。
- 问题：要 trillions of tokens，算力爆炸，optimization 不稳定

**Continuous pre-training**（EVE 的路子）——拿一个已经训好的 LLM，继续喂 visual data。
- 问题：**catastrophic forgetting**。你一训 visual，原来的 language knowledge 就被冲掉了。EVE-7B 的 MATH 从 13.9 掉到 0.7，这就是典型 case

Root cause 是什么？**vision 和 language 共享同一组 FFN 参数**。你给 FFN 喂 visual gradient，它就忘了 language；喂 language，就学不动 visual。共享参数本身就是病根。

---

## 他们的 trick：MMoE + Delta Tuning

Idea 非常朴素。既然共享参数是问题，那就**不共享**。每一层 LLM 的 FFN 旁边，加一个并行的 visual FFN。

### 架构公式

第 $l$ 层的 forward（paper 里 Eq.4 和 Eq.5）：

$$x_m^{l'} = x_m^{l-1} + \text{MHA}(\text{RMSNorm}(x_m^{l-1}))$$

$$x_m^l = x_m^{l'} + \text{MMoE}(\text{RMSNorm}(x_m^{l'}))$$

其中 MMoE 是一个 routing 逻辑：

$$\text{MMoE}(x) = \begin{cases} \text{FFN}_v(x) & \text{if } x \in x_v \text{（visual token）} \\ \text{FFN}_t(x) & \text{if } x \in x_t \text{（text token）} \end{cases}$$

变量含义：
- $x_m^{l-1}$：第 $l-1$ 层输出的 multimodal hidden state
- $\text{MHA}$：multi-head attention（**vision 和 language 共享**，负责 cross-modal alignment）
- $\text{RMSNorm}$：LayerNorm 的一种，用 RMS 做归一化
- $\text{FFN}_v$：新增的 visual expert
- $\text{FFN}_t$：原 LLM 的 textual expert（**冻结**）
- $x_v, x_t$：visual / text token 的子集

**关键设计选择**：
1. **Static routing**——visual token 走 visual expert，text token 走 text expert，不学 router。避免 MoE 训练初期 router 乱跳导致的 instability
2. **Shared MHA, separated FFN**——attention 做跨模态对齐，FFN 做模态特异知识。这个思路和 VLMo、BEiT-3 一脉相承
3. **Initialization**：$\text{FFN}_v$ 从 $\text{FFN}_t$ **复制**初始化，不是随机初始化。这样 visual expert 起点就有 language knowledge 的 warm start，再逐步 specialize。这个细节很重要，随机初始化会让 visual expert 从零学起，浪费 pre-trained LLM 的 knowledge

这个 visual expert 一共 1.2B params，base LLM 是 InternLM2-1.8B。

---

## 为什么这个 idea 有用（intuition）

这里我展开讲讲，因为这是 paper 的核心 insight。

### 1. Catastrophic forgetting 的本质

Forgetting 不是"visual 训练太猛把 language 覆盖了"。Forgetting 是**参数空间被两个 task 抢**。同一个 FFN weight 既要编码 language 的 syntax、semantics，又要编码 visual 的 edge、texture、object——这两组 representation 在 weight space 里互相挤压，你训一个，另一个的 representation 就被扰动。

LoRA、adapter 这些 PEFT 方法为什么 work？因为它们**新增了一个 parameter subspace**，让新 task 在新 subspace 里学，不动 original weights。Mono-InternVL 做的就是这个事，只不过用 MoE 结构自然落地——visual expert 就是那个"新 subspace"。

### 2. 为什么 freeze 整个 LLM 还能学视觉

你 freeze 了 $\text{FFN}_t$ 和大部分参数，只训 $\text{FFN}_v$ 和 patch embedding，model 还能学视觉吗？能。因为：

- **MHA 是共享的**——attention 能学到 vision-language alignment（paper 在 S1.3 alignment learning 阶段会 unfreeze MHA，Table 9 证明这很重要）
- **$\text{FFN}_v$ 从 $\text{FFN}_t$ 复制初始化**——visual expert 起点就有 language representation 的结构，相当于"站在 LLM 肩膀上"学视觉
- **Patch embedding 是 trainable 的**——visual input 的 projection 能学

所以整个 pipeline 是：visual input → patch embedding（学）→ MHA（部分学）→ FFN_v（学）/ FFN_t（冻结）→ output。视觉知识进 $\text{FFN}_v$ 和 patch embedding，language knowledge 留在 $\text{FFN}_t$，井水不犯河水。

### 3. 推理成本为什么几乎不增

MoE 的 sparse activation 特性：每个 token 只激活一个 expert，所以 FLOPs 增量极小。增加的只是参数量（storage），不是计算量。这就是为什么 Table 6 里 Mono-InternVL 比 InternVL-1.5 快那么多——没有 ViT 这个额外的 encoder，visual token 直接进 LLM，省了 ViT 的 forward。

---

## EViP：三阶段 progressive pre-training

这个也值得细说。EViP（Endogenous Visual Pre-training）的本质是一个 **coarse-to-fine curriculum**。

| Stage | 数据量 | 数据源 | Max patches | 可训参数 | 目标 |
|-------|--------|--------|-------------|---------|------|
| **S1.1 Concept** | 922M | Laion-2B + COYO（noisy） | 1,280 | PatchEmb + $\text{FFN}_v$ | 学基本 object、shape |
| **S1.2 Semantic** | 258M | Laion + COYO + SAM，caption 用 InternVL2-8B 合成 | 1,792 | PatchEmb + $\text{FFN}_v$ | high-level semantics、world knowledge |
| **S1.3 Alignment** | 143M | InternVL-1.5 pre-training data（caption 53.9% / detection 5.2% / OCR 40.9%） | 3,328 | PatchEmb + $\text{FFN}_v$ + **MHA** | 对齐 downstream 任务 |
| **S2 Instruction** | 5M | InternVL instruction data | 6,400 | **全模型** | 指令跟随 |

### 为什么这么分阶段

- **S1.1 用 noisy data**：noisy data 量大便宜，先让 model 学到基本的 visual concepts（"这是一只猫"、"这是天空"）。这种 coarse grained 的东西 noisy data 足够。这个阶段很快 saturate（Figure 4 里 concept learning 曲线很快平了）

- **S1.2 用合成 caption**：noisy caption 质量太低，学不动 high-level semantics（关系、world knowledge）。用更强的 VLM（InternVL2-8B）给 258M 张图重新生成 caption，这些 caption 更干净、信息密度更高。这个阶段 performance 继续涨

- **S1.3 用 task-specific data**：caption / detection / OCR，对应 downstream 任务的 distribution。这里**开始 unfreeze MHA**，让 vision-language alignment 真正发生。Table 9 证明 unfreeze MHA 让 DocVQA 从 39.5 涨到 49.3，InfoVQA 从 19.7 涨到 22.7

- **S2 全模型 instruction tuning**：unfreeze 所有参数，学下游任务格式

这个 curriculum 的核心 insight：**noisy data 学 coarse，clean data 学 fine，task data 做 alignment**。和 human learning 的"先认字、后阅读、后做阅读理解"一个道理。

### 优化目标

$$\arg\min_{\Delta\theta} \mathcal{L}(\mathcal{F}_{\text{llm}}(x_m; \theta, \theta_v), \hat{y})$$

- $\theta$：pre-trained LLM 参数（frozen）
- $\theta_v$：visual expert + patch embedding 参数（trainable）
- $\Delta\theta$：可训参数集合（S1.1/S1.2 是 $\theta_v$，S1.3 还加 MHA）
- $\hat{y}$：ground truth
- $\mathcal{L}$：auto-regressive next-token loss

本质就是 delta tuning——只优化一个 parameter subset，其余冻结。

---

## 数据说话

### 主表（Table 2）核心对比

| Model | #A-Param | $\text{Avg}_{\text{MM}}$ | $\text{Avg}_{\text{QA}}$ | OCRBench | MathVista |
|-------|----------|--------------------------|---------------------------|-----------|-----------|
| InternVL-1.5-2B（modular） | 2.2B | 54.4 | 71.7 | 654 | 41.1 |
| Qwen2VL-2B（modular） | 2.1B | - | 73.5 | 809 | 43.0 |
| Chameleon-7B（mono） | 7B | 16.1 | 17.9 | 7 | 22.3 |
| EVE-7B HD（mono） | 7B | 38.9 | 54.6 | 398 | 34.2 |
| Emu3（mono） | 8B | - | 67.6 | 687 | - |
| **Mono-InternVL-2B** | **1.8B** | **55.2** | **70.1** | **767** | **45.7** |

**1.8B 打赢 8B 的 Emu3**，OCRBench 直接 +80 points。VQA avg 比 EVE-7B HD 高 15.4%。和 modular 的 InternVL-1.5 基本持平，MathVista 和 OCRBench 甚至更强。

### NLP 能力保留（Table 4）——这是关键

| Model | MMLU | CMMLU | AGIEval | MATH |
|-------|------|-------|---------|------|
| InternLM2-Chat（原始） | 47.1 | 46.1 | 38.8 | 13.9 |
| EVE-7B | 43.9 | 33.4 | 22.6 | **0.7** |
| Chameleon-7B | 52.1 | - | - | 11.5 |
| **Mono-InternVL** | **45.1** | **44.0** | **40.9** | **12.3** |

EVE 的 MATH 从 13.9 掉到 0.7——catastrophic forgetting 的教科书级翻车。Mono-InternVL 基本保住了所有 NLP 能力，MMLU 只掉了 2 个点。**这就是 freeze + delta tuning 的价值**。

### 推理效率（Table 6）

| Input tokens | InternVL-1.5 TTFT | Mono-InternVL TTFT | 降幅 |
|--------------|-------------------|---------------------|------|
| 1024 | 0.24s | 0.09s | **-63%** |
| 2048 | 0.45s | 0.15s | **-67%** |
| 4096 | 1.93s | 0.79s | **-59%** |

TTFT（Time To First Token）砍掉一大半，因为省掉了 ViT encoder 的 forward。throughput 还涨了 31%。这是 monolithic 架构的 intrinsic 优势——没有 encoder 这个额外开销。

---

## 我的 take 和延伸思考

### 1. 这个 idea 其实不新，但 execution 精准

MoE 做 modality separation，VLMo、BEiT-3、VL-MoE 都做过。Delta tuning 也是老话题。但把这两者**在 monolithic MLLM 这个场景下结合起来**，并且用 progressive curriculum + 合成 caption 把 visual expert 训出来，execution 很扎实。

### 2. Visual expert 从 language FFN 复制初始化是个 double-edged sword

好处是 warm start，坏处是 visual representation 的上限可能被 language FFN 的结构限制。Language FFN 学的是 syntax、semantics 这种 abstract representation，visual 需要 edge、texture、spatial 这种 low-level 的东西，两者最优 representation space 可能不一样。未来可能需要随机初始化 + 更长训练，或者某种 hybrid 初始化。

### 3. 浅层 locality 的发现很有意思

Figure 5 的 attention map 显示，transformer 第 1 层的 visual token attention 呈现局部 pattern，很像 CNN 的感受野。这暗示**monolithic MLLM 的浅层可能需要 CNN-like 的归纳偏置**。未来设计上，浅层用 conv、深层用 attention 的 hybrid 架构可能更优。这也是为什么 Mono-InternVL 在 high-res benchmark（InfoVQA）上还比不过 modular——浅层 visual encoding 能力受限于 transformer 的 depth。

### 4. 对 LLM 未来的 hint

这 paper 给了一个 broader 的 insight：**扩展 LLM 到新 modality，不应该 fine-tune 原参数，应该新增 parameter subspace**。这个思路对 audio、video、3D 等其他 modality 的 extension 也适用。MoE-style 的 modality expert + frozen base 可能成为 multimodal LLM 的标准范式。

### 5. 和 Emu3 路线的对比

Emu3 走的是 native pre-training（next-token prediction is all you need），从零训 trillions of tokens。Mono-InternVL 走的是 continuous pre-training + delta tuning。前者更纯粹但贵，后者更便宜但要处理 forgetting。短期内 delta tuning 路线更 practical，长期看 native 路线可能上限更高。两条路线的对比很像 BERT-style pre-training vs. GPT-style scaling 的争论。

---

## 相关 link

- Paper: https://arxiv.org/abs/2412.06739
- Code: https://github.com/OpenGVLab/Mono-InternVL
- InternVL-1.5（modular baseline）: https://arxiv.org/abs/2404.16821
- Emu3（native pre-training 路线）: https://arxiv.org/abs/2409.18869
- EVE-7B（continuous pre-training 路线）: https://arxiv.org/abs/2406.11832
- Chameleon（mixed-modal early fusion）: https://arxiv.org/abs/2405.09818
- VLMo（modality MoE 的先驱）: https://arxiv.org/abs/2111.02358
- BEiT-3: https://arxiv.org/abs/2208.10442
- Delta tuning 综述: https://arxiv.org/abs/2203.06904
- MoE 经典综述: https://arxiv.org/abs/2209.01667
- Catastrophic forgetting in MLLM: https://arxiv.org/abs/2309.10313

---

总之这 paper 的 takeaway 就一句：**要给 LLM 加新 modality，别动老参数，加个新 expert 冻结老的训新的**。简单粗暴但 work。希望这个讲解帮你 build 起 intuition，Andrej。

---

# Mono-InternVL 深度讲解

Andrej，这是一篇很有意思的工作。来自Shanghai AI Lab和清华的团队，针对monolithic MLLM（即将visual encoding和language decoding融合进单一LLM的架构）的核心痛点——**catastrophic forgetting**和**unstable optimization**——提出了一个相当优雅的solution。下面我从intuition出发，把architecture、pre-training strategy、实验结果都拆给你看。

---

## 1. 核心Motivation：为什么Monolithic MLLM这么难做

现有MLLM分两类：

| 类型 | 代表工作 | 结构 | 问题 |
|------|---------|------|------|
| **Modular MLLM** | LLaVA, InternVL, Qwen-VL | CLIP-ViT + LLM，外挂encoder | 性能强，但deployment复杂、visual encoder有上限 |
| **Monolithic MLLM** | Fuyu-8B, EVE, Emu3, Chameleon | 单一LLM直接处理visual+text | 简单高效，但pre-training不稳定，forgetting严重 |

Monolithic MLLM的两种pre-training路线都有问题：

1. **Native pre-training**（如Chameleon, Emu3）：从头训练mixed-modality data。**问题**：成本高（trillions of tokens）、optimization不稳定。
2. **Continuous pre-training**（如EVE）：在pre-trained LLM上继续做visual pre-training。**问题**：catastrophic forgetting——visual learning会破坏language knowledge。

**Mono-InternVL的insight**：问题的root cause是**shared architecture for joint vision-language modeling**——vision和language共享同一组FFN参数，导致视觉优化必然冲击语言能力。

**Solution原则**：引入一个**independent visual parameter set**（visual experts），freeze掉原始LLM，只训练新增的visual部分。这本质上是一个**delta tuning**思想，但用MoE结构自然落地。

---

## 2. 架构详解：Multimodal Mixture-of-Experts (MMoE)

### 2.1 Visual & Textual Embeddings

**Visual embedding**（Eq.1）：
$$x_v = \text{MLP}(\text{PatchEmbed}(I) + \text{PE})$$

- $I \in \mathbb{R}^{H \times W \times 3}$：输入image
- $\text{PatchEmbed}(\cdot)$：stride=28的patch embedding，每个visual token对应一个$28 \times 28$的image patch
- $\text{PE} \in \mathbb{R}^{(h \times w) \times d}$：learnable positional embedding（和InternVL-1.5一致）
- $\text{MLP}(\cdot)$：将patch投影到LLM的$d$-维embedding space
- 额外添加一个**thumbnail**提供全局信息

这个简单tokenizer可以处理**up to 8 million pixels（10,240 patches）**的高分辨率image。

**Textual embedding**（Eq.2）：
$$x_t = \text{Tokenizer}(T)$$
- $T \in \mathbb{Z}^n$：input text tokens
- 和原LLM tokenizer完全一致

### 2.2 MMoE结构（核心创新）

**关键公式**（Eq.4 + Eq.5）：

第$l$层LLM layer：
$$x_m^{l'} = x_m^{l-1} + \text{MHA}(\text{RMSNorm}(x_m^{l-1}))$$
$$x_m^l = x_m^{l'} + \text{MMoE}(\text{RMSNorm}(x_m^{l'}))$$

其中MMoE定义为（Eq.5）：
$$\text{MMoE}(x) = \begin{cases} \text{FFN}_v(x) & \text{if } x \in x_v \\ \text{FFN}_t(x) & \text{if } x \in x_t \end{cases}$$

- $\text{FFN}_v$：**visual expert**（新增）
- $\text{FFN}_t$：**textual expert**（即原LLM的FFN）
- **Static routing**：visual tokens走visual expert，text tokens走textual expert，不使用learned router（避免training instability）

**Initialization的关键细节**：$\text{FFN}_v$从$\text{FFN}_t$复制初始化，共1.2B parameters。这样visual expert继承了language knowledge作为起点，再逐步specialize到visual modality。

**为什么这个架构漂亮**：
1. **Shared MHA, separated FFN**——attention负责cross-modal alignment，FFN负责modality-specific knowledge。这和VLMo、BEiT-3的设计哲学一致。
2. **Frozen $\text{FFN}_t$ 保留language能力**，visual expert独立优化。
3. **推理成本几乎为零**：MoE的sparse activation使得只有对应expert被激活，增加的只是参数量，FLOPs增量极小。

---

## 3. Endogenous Visual Pre-training (EViP)

### 3.1 优化目标（Eq.6）

$$\arg\min_{\Delta\theta} \mathcal{L}(\mathcal{F}_{\text{llm}}(x_m; \theta, \theta_v), \hat{y})$$

- $\theta$：pre-trained LLM参数（frozen）
- $\theta_v$：patch embedding + visual experts（trainable）
- $\Delta\theta$：可训练参数子集
- 在alignment learning阶段，$\Delta\theta$还包含MHA层

### 3.2 三阶段Progressive Learning

| Stage | 数据量 | 数据源 | Max Patches | 可训练参数 | 目标 |
|-------|--------|--------|-------------|-----------|------|
| **S1.1 Concept Learning** | 922M | Laion-2B, COYO-700M (noisy) | 1,280 | PatchEmbed + $\text{FFN}_v$ | 学基本概念（object category, shape） |
| **S1.2 Semantic Learning** | 258M | Laion + COYO + SAM, caption由InternVL2-8B合成 | 1,792 | PatchEmbed + $\text{FFN}_v$ | 高级语义、world knowledge |
| **S1.3 Alignment Learning** | 143M | InternVL-1.5 pre-training data (caption 53.9% / detection 5.2% / OCR 40.9%) | 3,328 | PatchEmbed + $\text{FFN}_v$ + **MHA** | 对齐downstream任务 |
| **S2 Instruction Tuning** | 5M | InternVL instruction data | 6,400 | **全模型** | 多任务指令跟随 |

**Curriculum design的intuition**：
- S1.1：noisy data学**coarse visual concepts**（scale大、quality低）
- S1.2：用更强的VLM（InternVL2-8B）生成**clean captions**，学到更high-level的relationship、world knowledge
- S1.3：用**task-specific data**做alignment，引入MHA训练实现vision-language对齐
- S2：unfreeze整个模型做instruction following

### 3.3 Ablation验证EViP设计合理性

Table 5的关键对比：

| Model | Trainable Params | Strategy | MME-P | SQA-I | AI2D |
|-------|-----------------|----------|-------|-------|------|
| InternLM2 (baseline) | 1.8B | Full tuning | 753 | 36.7 | 27.7 |
| + V-Expert | 3.0B | Full tuning | 948 | 37.7 | 26.6 |
| + V-Expert | 1.2B | **Delta tuning** | **995** | **56.5** | **42.7** |

- Full tuning加visual expert收益有限（SQA-I只+1%）
- **Delta tuning巨大提升**：SQA-I +18.8%, AI2D +16.1%
- 证明：freeze原LLM + 只训练visual expert 是关键

Figure 4展示data scalability：
- Concept learning很快达到upper bound
- Semantic + Alignment learning后，performance随data size**monotonically提升**
- 证明progressive learning + coarse-to-fine data的有效性

Table 9：alignment learning阶段是否unfreeze MHA：

| Method | DocVQA | InfoVQA | SQA-I |
|--------|--------|---------|-------|
| Freeze attention | 39.5 | 19.7 | 56.5 |
| Unfreeze attention | **49.3** | **22.7** | **61.8** |

→ 在S1.3阶段unfreeze MHA对vision-language alignment至关重要。

---

## 4. 实验结果

### 4.1 主要Benchmark对比（Table 2）

| Model | #A-Param | $\text{Avg}_{\text{MM}}$ | $\text{Avg}_{\text{QA}}$ | OCRBench | MathVista |
|-------|----------|--------------------------|---------------------------|-----------|-----------|
| **Modular:** | | | | | |
| InternVL-1.5-2B | 2.2B | 54.4 | 71.7 | 654 | 41.1 |
| Qwen2VL-2B | 2.1B | - | 73.5 | 809 | 43.0 |
| **Monolithic:** | | | | | |
| Chameleon-7B | 7B | 16.1 | 17.9 | 7 | 22.3 |
| EVE-7B (HD) | 7B | 38.9 | 54.6 | 398 | 34.2 |
| Emu3 | 8B | - | 67.6 | 687 | - |
| **Mono-InternVL-2B** | **1.8B** | **55.2** | **70.1** | **767** | **45.7** |

**亮点**：
- **1.8B参数超过8B的Emu3**（OCR +80 points）
- 比EVE-7B (HD) VQA平均+15.4%
- 和modular InternVL-1.5-2B平均performance相当
- 在MathVista和OCRBench上尤其强（seamless text recognition + reasoning）

### 4.2 NLP能力保留（Table 4）

| Model | MMLU | CMMLU | AGIEval | MATH |
|-------|------|-------|---------|------|
| InternLM2-Chat (原始) | 47.1 | 46.1 | 38.8 | 13.9 |
| EVE-7B | 43.9 | 33.4 | 22.6 | 0.7 |
| Chameleon-7B | 52.1 | - | - | 11.5 |
| **Mono-InternVL** | **45.1** | **44.0** | **40.9** | **12.3** |

→ Mono-InternVL基本保留了InternLM2的NLP能力，而EVE严重退化（catastrophic forgetting）。这是delta tuning策略的直接验证。

### 4.3 Pre-training中间结果（Table 3）

| Model | #A-Param | COCO Caps (CIDEr) | Flickr30k | NoCaps | VQAv2 |
|-------|----------|-------------------|-----------|--------|-------|
| Flamingo-3B | 3B | 73.0 | - | - | 49.2 |
| MM1-3.5B | 3.5B | 73.5 | - | 55.6 | 46.2 |
| Chameleon-34B | 34B | 120.2 | 74.7 | - | 66.0 |
| Mono-InternVL-S1.2 | 1.8B | 87.3 | 72.7 | 54.1 | - |
| Mono-InternVL-S1.3 | 1.8B | 135.6 | 77.3 | 116.5 | 71.1 |

→ 经过concept+semantic learning的Mono-InternVL-S1.2已超过MM1（+13.8 CIDEr），1.8B完胜34B Chameleon。

### 4.4 推理效率（Table 6）

| Model | Input Tokens | TTFT (s) | TPS (tok/s) |
|-------|--------------|----------|-------------|
| InternVL-1.5-2B | 1024 | 0.24 | 382 |
| **Mono-InternVL-2B** | 1024 | **0.09 (-63%)** | **436 (+14%)** |
| InternVL-1.5-2B | 2048 | 0.45 | 183 |
| **Mono-InternVL-2B** | 2048 | **0.15 (-67%)** | **232 (+27%)** |
| InternVL-1.5-2B | 4096 | 1.93 | 52 |
| **Mono-InternVL-2B** | 4096 | **0.79 (-59%)** | **68 (+31%)** |

→ Eliminating visual encoder使得**first token latency最多降67%**，throughput提升31%。这是monolithic架构的intrinsic优势。

---

## 5. Qualitative Insights：Attention Map可视化

Figure 5展示不同层的attention pattern：

1. **浅层locality存在**：第1层visual tokens只与附近token交互，pattern类似CNN的局部感受野
2. **浅层modalities barely interactive**：第1层visual-text attention weights极小
3. **深层逐步融合**：随着layer加深，cross-modal attention增强

**这个发现非常重要**：即使transformer架构上是global的，浅层仍呈现类似CNN的归纳偏置。这暗示monolithic MLLM的设计可能需要考虑shallow visual encoding的特殊性。

---

## 6. 总结：为什么这个工作重要

**核心贡献**：

1. **First monolithic MLLM matching modular SOTA**：1.8B参数达到InternVL-1.5水平
2. **优雅解决catastrophic forgetting**：MMoE + delta tuning，简单但有效
3. **Endogenous pre-training strategy**：progressive coarse-to-fine curriculum，data scalable
4. **67% latency reduction**：deployment-friendly

**Limitations**：
- High-resolution benchmarks（如InfoVQA）仍弱于modular baseline（-12.4%），作者归因于model depth较浅限制了visual encoding能力
- Visual expert从language FFN初始化，可能限制了visual representation的上限

**对未来的hint**：
- Monolithic MLLM是design simplicity + deployment efficiency的合理方向
- MoE-style modality separation比shared FFN更优
- 浅层CNN-like locality暗示可能需要hybrid CNN-Transformer设计

Paper link: https://arxiv.org/abs/2412.06739
Code: https://github.com/OpenGVLab/Mono-InternVL

---

希望这个讲解帮你build起对monolithic MLLM design space的intuition，Andrej。这个工作的核心启示是：**modality-specific parameters + frozen base model** 是同时实现modality扩展和knowledge preservation的关键。这和LoRA、adapter等方法一脉相承，但用MoE架构更自然地落地了。
