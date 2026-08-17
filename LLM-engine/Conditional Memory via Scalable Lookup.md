---
source_pdf: Conditional Memory via Scalable Lookup.pdf
paper_sha256: 90bf37b36cf5b507705c53d54f6e8142870fba23955f82a6a3b0aa1a983193a3
processed_at: '2026-08-03T16:56:05-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

---

## 一句话说清楚

现在的 LLM 有个毛病：明明该用查表解决的事，它非要用计算硬扛。这篇 paper 说，**给它装个查表模块，模型反而变聪明了，而且算力一分没多花**。

---

## 现在的 LLM 有什么问题

你给模型一句话 "Diana, Princess of Wales"，模型要花 6 层 attention + FFN 才把这个实体的"意思"拼出来。

这件事很荒谬——"Diana, Princess of Wales" 是个固定短语，地球上几十亿人都知道它是谁。模型不该一层层去"推理"它，应该直接查一下就知道了。

**类比**：你要算 $7 \times 8$，不应该列竖式推导，应该背九九乘法表。但现在的 LLM 天天在"列竖式算乘法"，因为没人给它乘法表。

---

## 他们干了什么

给模型装了一个叫 **Engram** 的模块，本质就是一张超大的"乘法表"——**n-gram 查表**。

输入 "Princess of Wales" 这三个 token，直接 hash 一下，从一张几十亿参数的表里 O(1) 捞出一个 embedding，塞回模型。

**关键**：这个查表操作不算 FLOPs。表可以无限大，算力不变。

---

## 为什么不是简单查表就行

查表有个问题：hash 会撞，"apple" 可能查到一个垃圾。而且同一个词在不同上下文意思不同（polysemy）。

所以他们加了个 **gate**：模型当前 hidden state 当 Query，查回来的 embedding 当 Key/Value，算个 attention score 决定"这个查表结果我信不信"。

- 查回来的是垃圾 → gate 关 → 不影响模型
- 查回来的是有用先验 → gate 开 → 注入模型

这个 gate 是整个设计的灵魂。没有它，直接 average（像 OverEncoding 那样），效果差很多。

---

## 为什么这个事以前没人做好

其实 n-gram embedding 不是新东西，FastText 2017 年就有了。但有两个坎一直没跨过去：

**第一，没人在严格公平条件下验证过它到底值不值。** 以前的工作要么加在 input layer 破坏算力公平，要么带额外模块增加 FLOPs。这篇 paper 第一次做了 iso-parameter + iso-FLOPs 的严格对照：把 MoE 的 expert 砍掉 17 个，参数挪到 Engram 表里，总参数一样、每 token 算力一样，然后比谁强。

**第二，没人在系统层面把它做"便宜"了。** Engram 的查表 index 只依赖输入 token 序列，forward 还没开始就能算出来。这意味着你可以**提前**把要查的 embedding 从 CPU 内存 prefetch 过来，和 GPU 上的计算 overlap。实测 100B 参数的表全 offload 到 host memory，吞吐只掉 2.8%。

这两件事加起来，让 n-gram 从"玩具"变成"可以认真 scale 的东西"。

---

## 最反直觉的发现

装了查表模块，你预期知识任务涨（MMLU 确实涨了 3 分）。但**涨得最多的是推理任务**：

- BBH（通用推理）+5.0
- ARC-Challenge +3.7
- HumanEval（代码）+3.0
- MATH +2.4

知识任务才涨 3 分，推理任务涨 5 分。为什么？

**因为模型以前在用前几层"重建查表"**，这件事吃掉了 depth budget。现在查表外置了，前几层解放了，等于**白送了几层深度给推理用**。

他们用 CKA 验证了这件事：Engram 第 5 层的 representation，对应纯 MoE 第 12 层的 representation。Engram 浅层就达到了 MoE 要到深层才达到的"成熟度"。

用一句话说：**你不是给模型加了记忆，你是给模型减了负担**。

---

## U 形曲线讲了个什么道理

他们扫了一个分配比例 $\rho$：稀疏参数预算里，多少给 MoE expert，多少给 Engram 表。

结果是个 U 形：
- 全给 MoE（$\rho=1$）：差，因为没有查表，模型被迫算乘法
- 全给 Engram（$\rho=0$）：也差，因为丢了 conditional computation，动态推理没人做
- 最优点在 $\rho \approx 75\%-80\%$：大部分预算给 MoE 做计算，留 20%-25% 给 Engram 做查表

这个 U 形在两个规模上都稳定出现，说明不是偶然。

**道理**：不同任务需要不同 primitive。硬把所有事塞进一个 primitive 是浪费。就像 CPU 里 ALU 和 L2 cache 不能互相替代，MoE 和 Engram 也不能互相替代。

---

## 长上下文为什么也涨了

这也是个意外惊喜。Multi-Query NIAH（多针大海捞针）从 84.2 涨到 97.0。

原因：attention 以前要同时干两件事——记局部依赖（"Alexander the Great" 是个整体）+ 处理全局上下文（32k token 里某个变量绑定）。现在局部依赖被 Engram 卸走了，attention 可以全力做全局。

**类比**：你读书时如果不用花脑力认字，就能把全部注意力放在理解情节上。Engram 干的就是"认字"这件事。

---

## 系统层面为什么这事重要

MoE 的 routing 依赖 runtime hidden state，必须等前一层算完才知道去哪个 expert 取参数，通信很难 overlap。

Engram 的查表 index 只看输入 token，forward 开始前就知道要查哪些 row。所以可以提前从 host memory 拉数据，和前面几层的计算重叠。

n-gram 服从 Zipf 分布——少数高频 pattern 占绝大多数访问。这意味着可以分级 cache：热数据进 GPU HBM，温数据进 host DRAM，冷数据进 NVMe。

**合起来**：你可以把模型容量扩到 100B+ 参数，inference 几乎不慢。这绕过了 GPU 显存这个最大的 bottleneck。

---

## 我觉得这篇 paper 真正的贡献

不是"n-gram 有用"——这个大家都知道。是三件事：

1. **第一次在严格公平条件下证明 n-gram 值得作为 first-class primitive**，不是锦上添花的外挂
2. **揭示了"memory 释放 depth 给 reasoning"这个机制**，用 CKA 和 LogitLens 给出了 clean 证据
3. **系统 co-design 让 100B 参数表 offload 几乎免费**，打开了"靠扩 embedding 而非扩 expert"的 scaling 路线

---

## 一句总结

**LLM 不该用 GPU 去模拟 hash table。给它一个真的 hash table，它会把省下来的算力用在更值得的地方。**

---

# Conditional Memory via Scalable Lookup: 一篇把 n-gram 重新拉回 LLM 主舞台的工作

Andrej，我读了好几遍这篇 paper，它最打动我的地方在于**重新审视了一个被 neural 时代遗弃的经典工具**，并且用一个非常严格的 iso-FLOPs/iso-parameter 对照实验，证明 n-gram embedding 不是怀旧，而是和 MoE 互补的"第二根稀疏轴"。

---

## 1. 核心直觉：语言信号的双重性

语言建模里其实混着两种性质完全不同的子任务：

- **Compositional reasoning**：长程依赖、上下文相关、动态。需要 deep compute。
- **Knowledge retrieval**：命名实体、固定短语、习语、公式化表达。局部、静态、高度模式化。

Standard Transformer 没有原生 lookup primitive，它只能用 attention + FFN 在前几层里**模拟重建一个静态查找表**。Table 3 那个 Diana, Princess of Wales 的例子特别直观：

| Layer | Latent State Translation |
|---|---|
| 1-2 | Country in the United Kingdom → "Wales" |
| 3 | Country in Europe → "Wales" |
| 4 | Title held by female sovereigns → "Princess of Wales (unspecific)" |
| 5 | Wife of the Prince of Wales → "Princess of Wales (unspecific)" |
| 6 | Diana, Princess of Wales (1961-1997) → "Diana, Princess of Wales" |

要走 6 层，才把一个 4-token 的实体拼出来。这本质上是 runtime 重建一个 lookup table，浪费了宝贵的 sequential depth。Ghandeharioun et al. (2024) 的 PatchScope 工作 [1] 和 Jin et al. (2025) 的 concept depth 工作 [2] 都揭示了这一点。

**Engram 的关键观察**：如果把这堆静态模式直接外置到一个 O(1) 可查的 embedding 表里，backbone 的早期层就解放了，可以专心做 reasoning。

---

## 2. 架构拆解

### 2.1 Sparse Retrieval via Hashed n-grams

第一步是把局部 context 映射到静态 embedding。

**Tokenizer Compression**：标准 subword tokenizer（如 BPE）优先 lossless reconstruction，会给语义等价的 token 分配不同 ID（`Apple` vs `␣apple` vs `apple,`）。Engram 用一个 surjective function 𝒫: V → V' 把这些 ID 折叠成 canonical ID，用 NFKC normalization + lowercasing。实测在 128k vocab 上压缩 23%。

公式上看，对一个 token $x_t$，先映射 $x'_t = \mathcal{P}(x_t)$，然后取后缀 n-gram：

$$g_{t,n} = (x'_{t-n+1}, \dots, x'_t)$$

这里 $t$ 是时间位置，$n$ 是 n-gram 阶数。

**Multi-Head Hashing**：n-gram 的组合空间太大没法直接参数化。借鉴 Tito Svenstrup et al. (2017) [3] 的 hash embedding 思路，每个 n-gram 阶 $n$ 用 $K$ 个独立 hash head，每个 head $k$ 把压缩后的 context 哈希到一个 size 为 $M_{n,k}$（取素数以减少 collision pattern）的 embedding table $\mathbf{E}_{n,k}$ 里：

$$z_{t,n,k} \triangleq \varphi_{n,k}(g_{t,n}), \quad \mathbf{e}_{t,n,k} = \mathbf{E}_{n,k}[z_{t,n,k}] \tag{1}$$

- $z_{t,n,k}$：hash 索引
- $\varphi_{n,k}$：multiplicative-XOR hash
- $\mathbf{E}_{n,k} \in \mathbb{R}^{M_{n,k} \times d_{\text{mem}}/K}$：可学习 embedding table

最终 memory vector 是所有 head、所有阶拼接：

$$\mathbf{e}_t \triangleq \bigsqcup_{n=2}^{N} \big\Vert_{k=1}^{K} \mathbf{e}_{t,n,k} \tag{2}$$

直觉上：multi-head 既是 collision insurance（多个 head 同时冲突的概率极低），又提供了多视角的 retrieval，类似 multi-head attention 的精神。

这部分其实是 FastText [4] 的现代化版本，但用 hashing 而非 enumeration，让 table size 可以 scale 到数十亿而不爆显存。

### 2.2 Context-aware Gating

Hash 检索回来的 $\mathbf{e}_t$ 是 context-independent prior，会受 polysemy 和 hash collision 污染。需要 gating 来做 disambiguation。

借鉴 attention 的 Q/K/V 形式：用当前 hidden state $\mathbf{h}_t$ 作 Query（已经通过前面 attention 聚合了 global context），用检索回的 $\mathbf{e}_t$ 同时作 Key 和 Value 的 source：

$$\mathbf{k}_t = \mathbf{W}_K \mathbf{e}_t, \quad \mathbf{v}_t = \mathbf{W}_V \mathbf{e}_t \tag{3}$$

scalar gate $\alpha_t \in (0,1)$：

$$\alpha_t = \sigma\left(\frac{\text{RMSNorm}(\mathbf{h}_t)^\top \text{RMSNorm}(\mathbf{k}_t)}{\sqrt{d}}\right) \tag{4}$$

- $\sigma$：sigmoid，保证 gate 在 (0,1)
- $d$：hidden dim
- RMSNorm 是为了 gradient stability（Dehghani et al., 2023 [5]），并且让 dot-product 不被 magnitude 主导

gated output $\tilde{\mathbf{v}}_t = \alpha_t \cdot \mathbf{v}_t$。如果 $\mathbf{e}_t$ 和当前 context 不一致，$\alpha_t \to 0$，自动 suppress 噪声。

接着用 short depthwise causal conv 扩大 receptive field 并增加非线性（kernel size $L=4$，dilation $D$ = max n-gram order = 3）：

$$\mathbf{Y} = \text{SiLU}(\text{Conv1D}(\text{RMSNorm}(\tilde{\mathbf{V}}))) + \tilde{\mathbf{V}} \tag{5}$$

- $\tilde{\mathbf{V}} \in \mathbb{R}^{T \times d}$：所有位置的 gated value 拼起来
- SiLU + residual：让初始 conv 权重 init 为 0 时是 identity mapping，训练初期不破坏 backbone

最终通过 residual connection 注入 backbone：$\mathbf{H}^{(\ell)} \gets \mathbf{H}^{(\ell)} + \mathbf{Y}$，然后接标准 Attention 和 MoE。

### 2.3 Multi-branch Integration

DeepSeek 这一代 backbone 用的是 mHC (Xie et al., 2025) [6] / HyperConnections (Zhu et al., 2025) [7]，把 residual stream 展成 $M$ 个并行 branch，每个 branch 有自己的 learnable connection weight。

Engram 这里做了一个很聪明的 parameter-sharing：
- **共享**：一张 sparse embedding table + 一个 $\mathbf{W}_V$（Value projection）
- **独立**：$M$ 个 branch-specific $\mathbf{W}_K^{(m)}$（Key projection）

branch $m$ 的 gate：

$$\alpha_t^{(m)} = \sigma\left(\frac{\text{RMSNorm}(\mathbf{h}_t^{(m)})^\top \text{RMSNorm}(\mathbf{W}_K^{(m)} \mathbf{e}_t)}{\sqrt{d}}\right) \tag{6}$$

- $\mathbf{h}_t^{(m)}$：branch $m$ 的 hidden state
- $m$：branch index

直觉：retrieved memory 是共享先验，但每个 branch 用自己的视角判断"这个 memory 对我的语义分支有没有用"，从而让 dense FP8 matmul 把 $\mathbf{W}_V$ 和所有 $\mathbf{W}_K^{(m)}$ fuse 在一起，GPU 利用率拉满。

这个设计让我想到 Transformer 里 multi-head attention 的精神：Q 是 branch-specific 的"问题"，K/V 是共享的"知识库"。

### 2.4 System: Decoupling Compute and Memory

这一节是我觉得最 engineering-valuable 的部分。

**MoE 的问题**：routing 依赖 runtime hidden state，必须等前一层算完才知道去哪个 expert 取参数。这意味着 expert parallel 的 All-to-All 通信很难完全 overlap。

**Engram 的优势**：检索索引**只**依赖 input token sequence，**forward pass 开始之前就能算出来**。这打开了 prefetch + overlap 的大门：

- **Training**：embedding table shard 到所有 GPU，All-to-All 拉取 active rows（forward）/ dispatch gradients（backward）
- **Inference**：把巨大的 embedding table offload 到 host DRAM，PCIe 异步 prefetch，让前几层 dense block 的计算做 buffer 来 hide PCIe latency

Engram 的 layer placement 因此是一个 **hardware-algorithm co-design**：
- 越浅放 → 模型表达上越早卸下"local pattern reconstruction"负担（Section 6.2 ablation 显示 Layer 2 最优）
- 越深放 → 越多前置计算可以 hide prefetch latency

paper 里最终选了 [Layer 2, Layer 15] 双插入，既覆盖早期干预，又利用后期成熟的 context 做 gating，还提供了多级 cache hierarchy 的利用空间。

另一个关键 insight：n-gram 服从 Zipf distribution [8]，少量 pattern 占绝大多数访问。这天然支持 **Multi-Level Cache Hierarchy**：hot rows 进 HBM，warm rows 进 host DRAM，cold rows 进 NVMe。

Table 4 的实测数据：把 100B 参数的 Engram table 完全 offload 到 host memory，8B backbone 上 throughput penalty 只有 2.8%。这是非常震撼的结果——意味着你可以"几乎免费"地把模型容量扩到 100B+。

---

## 3. Sparsity Allocation: U-shaped Scaling Law

这是 paper 的理论核心。

### 三个参数量定义

- $P_{\text{tot}}$：总 trainable parameters（不含 vocab embedding 和 LM head）
- $P_{\text{act}}$：每 token 激活参数（决定 FLOPs）
- $P_{\text{sparse}} \triangleq P_{\text{tot}} - P_{\text{act}}$：稀疏参数预算（unselected experts + unretrieved embeddings）

### Allocation Ratio

$$P_{\text{MoE}}^{(\text{sparse})} = \rho P_{\text{sparse}}, \qquad P_{\text{Engram}} = (1 - \rho) P_{\text{sparse}} \tag{7}$$

- $\rho = 1$：pure MoE
- $\rho < 1$：把一部分 expert 容量换成 Engram slot

### U-shape 实验结果

两个 compute regime（$C=2\times10^{20}$ FLOPs / $C=6\times10^{20}$ FLOPs），固定 sparsity ratio $P_{\text{tot}}/P_{\text{act}} \approx 10$。

Figure 3 (left) 显示 validation loss vs $\rho$ 是一个清晰的 **U-shape**：

- $\rho = 100\%$（pure MoE）suboptimal
- 最优点在 $\rho \approx 75\%-80\%$，validation loss 改善 $\Delta = 0.0139$（10B regime）
- $\rho$ 太小（pure memory）也 suboptimal，因为丢了 conditional computation

这个 U-shape 在两个 regime 里位置稳定，说明这是一个 robust 的结构性偏好。

**直觉**：
- MoE-heavy：模型缺乏 dedicated memory，被迫用 depth 模拟 retrieval
- Engram-heavy：丢了 conditional computation，处理 dynamic reasoning 会变弱

这是这篇 paper 最 elegant 的 contribution——它把"什么时候该用 lookup，什么时候该用 compute"用一条曲线量化了。

### Infinite Memory Regime

把 MoE backbone 固定，sweep embedding slot 数 $M$ 从 $2.58\times10^5$ 到 $1.0\times10^7$（加 ≈13B 参数）。

Figure 3 (right)：validation loss 在 log-space 上是严格线性——power law。这意味着 Engram 是一个**可预测的 scaling knob**，加大 memory 持续受益且**不增加 FLOPs**。

对比 OverEncoding [9] 的 averaging 方式（直接把 n-gram embedding 和 vocab embedding 平均），Engram 的 scaling 潜力显著更大。这印证了 context-aware gating + multi-branch integration 的价值，而不是简单 average。

---

## 4. 27B Scale Pre-training Results

Table 1 是主结果，四个模型 iso-activated-params (3.8B)、iso-tokens (262B)：

| Model | Total Params | Engram Params | Val Loss | MMLU | BBH | HumanEval | MATH |
|---|---|---|---|---|---|---|---|
| Dense-4B | 4.1B | - | 1.768 | 48.6 | 42.8 | 26.8 | 15.2 |
| MoE-27B | 26.7B | 0 | 1.634 | 57.4 | 50.9 | 37.8 | 28.3 |
| Engram-27B | 26.7B | 5.7B | 1.622 | 60.4 (+3.0) | 55.9 (+5.0) | 40.8 (+3.0) | 30.7 (+2.4) |
| Engram-40B | 39.5B | 18.5B | 1.610 | 60.6 | 57.5 | 38.4 | 30.6 |

Engram-27B vs MoE-27B 是严格 iso-parameter、iso-FLOPs，对比如下：

- Knowledge 任务：MMLU +3.0，MMLU-Pro +1.8，CMMLU +4.0，C-Eval +4.7，AGIEval +3.2
- **Reasoning 任务**：BBH +5.0，ARC-Challenge +3.7，DROP +3.3
- **Code/Math**：HumanEval +3.0，MBPP +1.6，GSM8K +2.2，MATH +2.4

最 striking 的是：reasoning 和 code/math 任务的提升**比** knowledge 任务更大。这跟最初的直觉（"memory 帮知识检索"）相反——memory 真正的价值在于**释放 backbone depth 给复杂推理**。

---

## 5. Long Context: 注意力被解放了

Section 5 的实验设计很严谨：他们控制了 iso-loss 和 iso-FLOPs 两个 setting，避免"Engram 强是因为 base 更强"的混淆。

Table 2 摘录：

| Setting | LongPPL Avg | MQ-NIAH | VT | CWE | FWE |
|---|---|---|---|---|---|
| MoE-27B (50k, loss=1.63) | - | 84.2 | 77.0 | 4.5 | 73.0 |
| Engram-27B (46k, loss=1.63, iso-loss) | - | 97.0 (+12.8) | 87.2 (+10.2) | 4.3 | 98.6 (+25.6) |
| Engram-27B (50k, iso-FLOPs) | 更低 | 97.0 | 89.0 | 5.9 | 99.3 |

**Iso-loss setting** 是最有说服力的：两个模型 base capability 完全对齐，长上下文能力差异完全来自架构。Multi-Query NIAH 84.2 → 97.0，Variable Tracking 77.0 → 87.2。

**直觉**：local dependencies 被卸到 lookup 之后，attention 的容量被解放给 global context。这跟"短链路 vs 长链路"的资源分配问题完全一致——你不需要 attention 去记忆 "Alexander the Great" 是个固定实体，attention 就有更多 capacity 去 track 跨 32k token 的变量绑定。

---

## 6. Mechanistic Analysis: Engram 等价于"加深网络"

### 6.1 LogitLens

把每层 hidden state 用 final LM Head 投影，算和最终 output 分布的 KL divergence。Figure 4(a) 显示 Engram 早期层的 KL divergence 显著更低——预测收敛更快。

直觉：因为外部 lookup 直接把"Wales"的 representation 注入了，模型不需要从 layer 1 开始一层层 compose。

### 6.2 CKA (Centered Kernel Alignment)

CKA 公式：

$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K)\text{HSIC}(L, L)}} \tag{8}$$

- $K = XX^\top$，$L = YY^\top$：Gram matrices (linear kernel)
- HSIC：Hilbert-Schmidt Independence Criterion [10]
- 评估数据集：Few-NERD [11]，取 entity 最后一个 token 的 hidden state

为量化 "Engram layer j 对应 MoE 的第几层"，定义 soft alignment index：

$$a_j = \frac{\sum_{i \in \mathcal{I}_j} S_{i,j} \cdot i}{\sum_{i \in \mathcal{I}_j} S_{i,j}}, \quad \mathcal{I}_j = \arg\text{top}_k(S_{i,j}) \tag{9}$$

- $S_{i,j}$：MoE layer $i$ 和 Engram layer $j$ 的 CKA similarity
- $\mathcal{I}_j$：Engram layer $j$ 最相似的 top-$k=5$ 个 MoE layer
- $a_j$：weighted centroid，表示 Engram layer $j$ 的"effective MoE depth"

Figure 4(b-c) 的 heatmap 显示对角线**明显向上偏移**：$a_j > j$ 对很多 layer 都成立。具体例子：Engram layer 5 的 representation 和 MoE layer 12 最 align。

这是非常 clean 的证据：**Engram 在浅层就达到了 MoE 要到深层才达到的 representation depth**。换句话说，Engram = 模型 + free depth。

### 6.3 Ablation: 哪些设计重要？

Figure 5 的 ablation 在 3B MoE backbone 上做：

- **Layer sensitivity**：单层 Engram，sweep 插入位置 1-12，Layer 2 最优 (val loss 1.770)。深层插入效果递减——因为 backbone 已经把 local pattern compose 完了，再 lookup 是冗余。
- **双层 vs 单层**：把同样 1.6B memory 拆成两半，放 [2, 6]，val loss 1.768，比单 layer 2 略好。这平衡了"早期 offload"和"晚期 rich gating"。
- **组件贡献**（移除后 val loss 退步幅度）：
  - Multi-branch fusion：最大退步
  - Context-aware gating：次大退步  
  - Tokenizer compression：第三
  - Depthwise conv：marginally important
  - 4-gram：在 1.6B budget 下略 suboptimal（稀释 2/3-gram 容量），但在更大 memory scale 下可能反转

### 6.4 Sensitivity: Memory Ablation

Figure 6 是个非常 clean 的 stress test：推理时把 Engram output 完全置零，看 backbone 还能撑多少。

- **Factual knowledge**：崩盘，保留 29-44%（TriviaQA 只剩 29%）→ Engram 是 parametric knowledge 的主要载体
- **Reading comprehension**：稳健，保留 81-93%（C3 保留 93%）→ context-grounded 任务主要靠 attention

这个 dichotomy 完美印证了 paper 的核心 thesis：把静态知识移出 backbone，backbone 专心做 attention-based reasoning。

### 6.5 Gating 可视化

Figure 7 把 $\alpha_t$ 画出来，gate 在 **multi-token named entity 末尾** 和 **formulaic phrase 末尾** 强激活：

- 英文："Alexander the Great", "the Milky Way", "By the way", "Princess of Wales"
- 中文：四大发明、张仲景、成语

这正是 paper 期待的——Engram 识别并处理 stereotyped 依赖，把 backbone 从记忆这些静态关联中解放。

---

## 7. 把它放进更大的图景里

### 7.1 与 Memory Network 谱系的关系

- **PKM** [12] (Lample 2019)：product key memory，sparse key-value store 嵌入 layer 内部。是 parametric memory 的开创性工作，但 key 是 learnable 的、运行时动态检索，不像 Engram 是 deterministic addressing。
- **PEER** [13] (He 2024)：mixture of a million experts，每个 expert 极小。本质上是 fine-grained MoE，仍属 conditional computation，不属于 conditional memory。
- **Ultra-Mem / UltraMem-v2** [14, 15] (Huang 2025)：ultra-sparse memory network，scaling 到 120B。和 Engram 同属 parametric memory，但 addressing mechanism 和 system co-design 不同。
- **Memory+** [16] (Berges 2025)：memory layers at scale，ICML 2025。
- **RETRO** [17] (Borgeaud 2022)：non-parametric，外部 corpus retrieval。
- **REALM** [18] (Guu 2020)：retrieval-augmented pretraining。
- **PlugLM** [19] (Cheng 2023a)：decoupled knowledge from parameters。

Engram 的差异点：**deterministic hash addressing + algorithm-system co-design + iso-FLOPs 严格对照**。

### 7.2 与 n-gram embedding 复兴的关系

- **FastText** [4] (Bojanowski 2017)：subword n-gram + averaging，经典。
- **OverEncoding** [9] (Huang 2025a)：hash n-gram embedding 直接和 vocab embedding average。论文里直接对比，Engram 在 sparse MoE backbone 上 scaling 效率显著更高。
- **BLT** [20] (Pagnoni 2025)：byte-level n-gram embedding，Meta 的工作。
- **SCONE** [21] (Yu 2025)：f-gram auxiliary model，但 inference-focused 且增加 training FLOPs，破坏了 iso-compute 公平对比。
- **SuperBPE** [22] (Liu 2025)：把 multi-word expression 合成 superword token，走 tokenizer 路线。
- **Infini-gram** [23] (Liu 2024b)：trillion-token unbounded n-gram，纯统计。
- **DeepEmbed** [24] (RWKV Team 2025)：RWKV-V8 的 embedding scaling。

Engram 在这条线上的位置：**first-class modeling primitive + 公平 iso-compute 对照 + system-level prefetch**。

### 7.3 与 MoE 谱系的关系

- **Shazeer 2017** [25]：sparse MoE 起源。
- **GShard** [26]、**Switch Transformer** [27]、**GLaM** [28]：scale MoE。
- **DeepSeekMoE** [29] (Dai 2024)：fine-grained expert + shared expert，Engram 用的 backbone。
- **DeepSeek-V3** [30]：MLA + MoE + mHC，Engram 实际用的架构。
- **Kimi-k2 / Kimi Linear** [31]：另一个 frontier MoE。

### 7.4 与 Knowledge Mechanism 谱系的关系

- **Geva et al. 2021** [32]：FFN as key-value memory。
- **Dai et al. 2022** [33]：knowledge neurons。
- **ROME** [34] / **MEMIT** [35] (Meng et al.)：causal tracing + model editing。
- **PatchScope** [36] (Ghandeharioun 2024)：Table 3 那个例子的来源。
- **Concept Depth** [37] (Jin 2025)：concept 在不同 layer 被 acquire。
- **Echoes of BERT** [38] (Li & Subramani 2025)：modern LM 重新发现 classical NLP pipeline。

Engram 的视角：既然 FFN 在做 key-value memory，那就**显式地**做 key-value memory，并把静态部分外置。

---

## 8. 我对这篇 paper 的几点直觉

### 8.1 为什么是 n-gram，不是 retrieval

n-gram 的优势是 **addressing 是 deterministic 的**——给定 token sequence，hash index 在 forward 之前就能算出来。这意味着：
- 没有 dynamic routing 的 latency 不确定性
- 可以做 prefetch + overlap
- 可以做 multi-level cache（Zipf 分布）

这恰好是 RETRO / REALM 这类外部 retrieval 系统做不到的——它们的 retrieval 依赖 query embedding，必须等 query 算完才能 retrieve。

### 8.2 为什么 reasoning 提升比 knowledge 提升更大

这个反直觉结果是 paper 最深刻的发现。我的理解：

- Backbone 的"layer budget"是固定的
- 早期层被 local pattern reconstruction 占用，是 **dead depth**
- Engram 把这部分 dead depth 卸掉，等于**给 reasoning 增加了 effective depth**
- Knowledge 任务本来就被 backbone handle 得不错，提升空间小
- Reasoning 任务被 depth bottleneck，提升空间大

这跟 CKA analysis 的"Engram layer 5 ≈ MoE layer 12"完全自洽。

### 8.3 U-shape 给了我们什么 modeling 哲学

U-shape 本质上在说："不同 sub-task 需要不同的 primitive，把它们硬塞进一个 primitive 是 suboptimal 的"。这其实是 modularity principle 的一个 quantitative 证明。

类比：CPU 里有 L1 / L2 / L3 / DRAM / SSD。Engram 在做 L2/L3 cache，MoE 在做 ALU compute。强行让 ALU 模拟 cache 是浪费 silicon。

### 8.4 系统层面的 implication

100B 参数 offload 到 host memory，throughput penalty 2.8%。这个数字很关键：

- 它说 Engram 可以**绕过 GPU memory bottleneck**，做"无脑扩参"
- 它说未来的 LLM 容量 scaling 可能不在 expert 数量上，而在 embedding table 大小上
- 它说 inference infrastructure 的"参数受限"假设可能要重写

跟 DeepSeek-V3 / Kimi-K2 的"MoE 上百 B"路线相比，Engram 提供了"embedding table 上百 B"的另一条路，且 inference 时延几乎不变。

### 8.5 一个未充分讨论的问题

paper 没太讨论 Engram 的 **update / editing** 特性。既然静态知识外置在 hash table 里，理论上做 factual update 比 ROME/MEMIT 干净得多——直接改对应 slot 的 embedding 即可。

这跟 PlugLM [19] 的 motivation 一致，但 Engram 没展开。这可能是下一篇 paper 的方向。

### 8.6 跟 biological sparsity 的呼应

paper 开篇引用 Lennie 2003 [39] 和 Olshausen & Field 1997 [40] 的 sparse coding。Engram 的 conditional memory + conditional computation 双轴，其实很接近 biological neural circuit 里"memory consolidation (hippocampus) vs cortical computation (neocortex)"的分工。n-gram lookup ≈ hippocampus 的 pattern completion，MoE compute ≈ neocortex 的 compositional reasoning。这是个很优美的 conceptual mapping。

---

## 9. 我会想做的 follow-up

读完这篇 paper，我会有几个 instinct：

1. **Adaptive n-gram order**：现在是固定 {2,3}-gram，能不能让模型自己学 n-gram 阶数？类似 adaptive computation time 的思路。
2. **Cross-layer Engram sharing**：现在 layer 2 和 layer 15 各自一个 table，能不能 share 一张大 table，不同 layer 用不同 projection？类似 ALBERT 的 parameter sharing 哲学。
3. **Engram for multimodal**：image patch / audio frame 的 local pattern 也高度 stereotyped，Engram 应该可以直接迁移。
4. **Knowledge editing via Engram**：上面提的，直接验证 Engram 做 factual update 的 editability / generalization。
5. **Larger n-gram, larger memory**：paper 说 4-gram 在 1.6B budget 下 suboptimal，但 18.5B (Engram-40B) 下呢？更高阶 n-gram + 更大 table 是否打开新的 scaling mode？
6. **Engram × Test-time compute**：lookup 是 O(1) 的，能不能在 test-time scaling 里用 Engram 做"cheap retrieval chain-of-thought"？

---

## 10. 总结

这篇 paper 做的事很简洁：

- **Concept**：把 n-gram embedding 作为"conditional memory"primitive 引入 LLM，和 MoE 的"conditional computation"形成互补稀疏轴
- **Method**：tokenizer compression + multi-head hash + context-aware gating + multi-branch integration + system co-design
- **Theory**：Sparsity Allocation 的 U-shaped law，指导 MoE 和 Engram 的最优分配
- **Scale**：27B 实证，iso-parameter / iso-FLOPs 严格对照，全面超 MoE baseline
- **Insight**：memory 真正的价值是**释放 backbone depth 给 reasoning**，知识任务提升反而不是最大的
- **System**：deterministic addressing → 100B offload, <3% overhead

如果用一句话总结这篇 paper 的哲学：**LLM 不该用 GPU cycles 去模拟 hash table**。

---

### References

[1] PatchScope (Ghandeharioun et al., 2024): https://arxiv.org/abs/2401.06102  
[2] Concept Depth (Jin et al., 2025): https://aclanthology.org/2025.coling-main.37/  
[3] Hash Embeddings (Tito Svenstrup et al., 2017): https://papers.nips.cc/paper/2017  
[4] FastText (Bojanowski et al., 2017): https://aclanthology.org/Q17-1010/  
[5] Scaling ViT (Dehghani et al., 2023): https://proceedings.mlr.press/v202/dehghani23a.html  
[6] mHC (Xie et al., 2025): https://arxiv.org/abs/2512.24880  
[7] HyperConnections (Zhu et al., 2025): https://openreview.net/forum?id=9FqARW7dwB  
[8] Zipf's law (Piantadosi, 2014): https://link.springer.com/article/10.3758/s13423-014-0585-6  
[9] OverEncoding (Huang et al., 2025a): https://openreview.net/forum?id=gbeZKej40m  
[10] HSIC (Gretton et al., 2005): https://dl.acm.org/doi/10.5555/1046975  
[11] Few-NERD (Ding et al., 2021): https://aclanthology.org/2021.acl-long.371/  
[12] PKM (Lample et al., 2019): https://papers.nips.cc/paper/2019  
[13] PEER (He, 2024): https://arxiv.org/abs/2407.04153  
[14] Ultra-Mem (Huang et al., 2025c): https://openreview.net/forum?id=zjeHLSiNv1  
[15] UltraMem-v2 (Huang et al., 2025b): https://arxiv.org/abs/2508.18756  
[16] Memory+ (Berges et al., 2025): https://openreview.net/forum?id=ATqGm1WyDj  
[17] RETRO (Borgeaud et al., 2022): https://arxiv.org/abs/2110.15853  
[18] REALM (Guu et al., 2020): https://arxiv.org/abs/2002.08909  
[19] PlugLM (Cheng et al., 2023a): https://aclanthology.org/2023.findings-acl.901/  
[20] BLT (Pagnoni et al., 2025): https://arxiv.org/abs/2502.05779  
[21] SCONE (Yu et al., 2025): https://arxiv.org/abs/2502.01637  
[22] SuperBPE (Liu et al., 2025): https://openreview.net/forum?id=lcDRvffeNP  
[23] Infini-gram (Liu et al., 2024b): https://openreview.net/forum?id=u2vAyMeLMm  
[24] RWKV DeepEmbed: https://wiki.rwkv.com/basic/architecture.html  
[25] Sparsely-Gated MoE (Shazeer et al., 2017): https://arxiv.org/abs/1701.06538  
[26] GShard (Lepikhin et al., 2020): https://arxiv.org/abs/2006.16668  
[27] Switch Transformer (Fedus et al., 2022): https://arxiv.org/abs/2101.03961  
[28] GLaM (Du et al., 2022): https://arxiv.org/abs/2112.06905  
[29] DeepSeekMoE (Dai et al., 2024): https://arxiv.org/abs/2401.06066  
[30] DeepSeek-V3 (Liu et al., 2024a): https://arxiv.org/abs/2412.19437  
[31] Kimi Linear (Team et al., 2025): https://arxiv.org/abs/2510.26692  
[32] FFN as Key-Value Memory (Geva et al., 2021): https://aclanthology.org/2021.emnlp-main.446/  
[33] Knowledge Neurons (Dai et al., 2022): https://aclanthology.org/2022.acl-long.232/  
[34] ROME (Meng et al., 2022): https://arxiv.org/abs/2202.05262  
[35] MEMIT (Meng et al., 2023): https://openreview.net/forum?id=MkbcAHIYgyS  
[36] PatchScope: https://arxiv.org/abs/2401.06102  
[37] Concept Depth: https://aclanthology.org/2025.coling-main.37/  
[38] Echoes of BERT: https://arxiv.org/abs/2506.02132  
[39] Lennie 2003: https://www.cell.com/current-biology/fulltext/S0960-9822(03)00189-3  
[40] Olshausen & Field 1997: https://www.sciencedirect.com/science/article/pii/S004269899700169X  
[41] Engram GitHub: https://github.com/deepseek-ai/Engram
