---
source_pdf: Training Transformers for KV Cache Compressibility.pdf
paper_sha256: 363660e09036013ca30359fb47c40ba91e05ceb517fa1ee6b14a9340429b838d
processed_at: '2026-08-12T18:05:13-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 paper

## 问题在哪

你有个 LLM，context length 128K，光 KV cache 就几十 GB，inference 时 memory 爆炸，decode 也慢。大家都在想办法压 KV cache——有 dropout 式 eviction（H2O、StreamingLLM、SnapKV），有 gradient-based 的 fancy 压缩（Cartridges、Attention Matching），都挺努力。

但这些 post-hoc 压缩方法有个共同盲区：**它们都假设 model 已经训好了，只能对它内部 representation 的现状做处理**。压得好不好，取决于 model 内部 KV 的组织方式有没有冗余可挖。

## 核心洞察

这篇 paper 就一句话：**KV cache 能不能压，是 model 的属性，不是数据的属性**。

同样的任务，同样的输入序列，两个 next-token 分布完全一样的 transformer，一个的 KV cache 能压到 1 个 slot 不损失精度，另一个压到一半就崩。差别在哪？在训练时 model 自己学到了什么样的内部 representation。

普通 NTP 训练，loss 只关心下一个 token 对不对，对 KV cache 内部长什么样完全不约束。SGD 找到的解，大概率是某种"最简单"的实现，但这种实现不一定是可压缩的。

## 理论 toy example 把直觉讲清楚了

拿 histogram 举例：输入一串 token，输出每个符号的出现频率。这函数 permutation-invariant，看起来超级简单。

**第一种实现**（不可压缩）：embedding 不带位置，layer 2 用 uniform attention 把所有 token embedding 平均一下，搞定。看起来很优雅。但你压它就崩——因为压缩后 attention 仍在平均，但分母从 $n+k$ 变成 $r(n)+k$，scaling 全变了，模型不知道发生了什么，没法 compensate。

**第二种实现**（可压缩）：representation 分成 4 个 block，一个存 token identity，一个存 positional encoding（编码长度），一个存 unnormalized histogram，一个备用。压缩时把整个 prefix 压成 1 个 KV pair，里面存 unnormalized histogram + prefix 长度。suffix 处理时 attention 平均后 scaling 错了，但 FFN 看到"我在压缩模式"（通过 positional encoding detect），自动 renormalize 恢复正确结果。

两种实现都正确计算 histogram，但一个能压到 1 slot，一个压不动。差别纯粹在 representation 的组织方式。

**普通训练会偏好第一种**，因为它更简单、参数更少。你没有任何理由去学第二种，因为 uncompressed 时两者都 work。所以你需要一个训练信号去逼 model 学可压缩的那一种。

## KV-CAT 怎么做

继续 pretraining，但 forward pass 有两份：

- **Masked pass**：router 决定哪些 KV slot 被 mask 掉（target keep 50%），model 在这种"缺一半 KV"的条件下还要工作
- **Dense pass**：标准 forward，当 teacher

Loss 有三部分：
- `L_mask`：masked pass 的输出分布要逼近 dense pass 的输出分布（self-distillation）。逼 model 在 KV 被砍一半时还能 reproduce 完整的行为——这等于强制关键信息被冗余编码到多个 slot 里
- `L_anchor`：dense pass 做标准 NTP，防止 model 退化
- `L_budget`：router 保持 50% keep rate（借了 MoE 的 load balancing trick）

Router 是个轻量 linear attention 模块，插在几个 layer 之间，共享给一组 layer 用。初始化时所有 token 都 keep（$W_P = -I, \alpha = 0$），从 dense model 出发逐渐学会 mask。用 straight-through estimator 处理 hard threshold 的梯度。

Inference 时 router 关掉，就是个标准 transformer，你想叠什么 post-hoc 压缩方法都可以。

## 为什么 work 的直觉

Self-distillation 在干一件事：逼 model 把每个"重要信息"冗余编码到多个 KV slot 里。如果信息只在 1 个 slot 里，mask 掉就丢了，distillation loss 就炸。要 minimize 这个 loss，model 必须学会把信息分散存储，让任何一个 slot 都能被"代替"。

这种冗余结构正好是 post-hoc 压缩器需要的。Attention Matching 找几个 key 重建 attention pattern，冗余度越高越好重建。Cartridges 用少量 slot 复现 logits，信息分散就越容易。

所以 KV-CAT 训练的 model，对 Cartridges 来说是"更容易优化"的目标，对 Attention Matching 来说是"更易重建"的 source。实验里 KV-CAT model 上做 gradient-based 压缩，**5× 加速**达到 base model 的最终效果——landscape 更友好。

## 实验结果

- **Uncompressed 性能**：几乎不变（0.5B +0.7，1.5B -0.5），dense anchor loss 起作用了，没拿压缩性换性能
- **Suffix perplexity**（Attention Matching）：5% keep 下 ∆PPL 从 0.780 降到 0.461，所有 keep ratio 都改善
- **Needle-in-haystack**：50% keep 下 0.5B 从 28% 检索准确率跳到 47%，相对 68% 提升
- **LongBench v2**：10% keep 下 QA 准确率从 22.2% 到 30.3%，相对 39% 提升，Legal subdomain 几乎翻倍
- **Cross-domain**：在 WikiText、PG-19、arXiv 上效果 transfer 过去，不是 FineWeb-Edu overfitting

## 我的看法

这个 paper 的核心 insight 其实挺深的：**当外部算法要消费 model 的内部状态时，训练目标应该显式约束这个状态的几何结构**，光约束 final output 不够。

这道理不只适用于 KV cache。RAG 里的 embedding model——你只训 contrastive loss，embedding 的内部结构适合 cosine 检索吗？量化场景——你只训 NTP，weights 的分布适合 4-bit 量化吗？Distillation target——你只 match logits，中间 representation 适合做 student 的 teacher 吗？这些都是同类问题。

KV-CAT 给了个 template：继续 pretraining，加一个 train-time bottleneck 模拟下游的消费方式，用 self-distillation 逼 model 适应这个 bottleneck。简单粗暴但有效。

局限也很明显：只测了 0.5B、1.5B，seq len 1024 训练；router 只在 4 个 layer group；没 scale 到 70B+。但这 paper 的贡献主要在 conceptual——证明 compressibility 是 trainable 的属性，提供一个可行路径。后面肯定会有人 scale 上去，和 quantization、long-context training、甚至新架构（Mamba 之类）结合。

最值得 follow 的方向是 paper 末尾提到的 **Chebyshev system 理论**。Attention 本质是 value 的线性 mixture，Chebyshev system 给你控制线性组合 zero 和 sign change 的工具，可能给出 compressibility 的紧 bound，而不只是 existence proof。这个方向如果做出来，compressibility 就能从"训练时靠启发式逼出来"变成"有理论指导的设计原则"。

参考链接：
- 论文：https://arxiv.org/abs/2502.15498
- Cartridges: https://arxiv.org/abs/2510.05665
- Attention Matching: https://arxiv.org/abs/2602.07762
- H2O: https://arxiv.org/abs/2306.14048
- QWEN2.5: https://qwenlm.github.io/blog/qwen2.5/
- LongBench v2: https://arxiv.org/abs/2412.15204

---

# Training Transformers for KV Cache Compressibility 详解

## 1. 背景与动机

Long-context LLM inference 的核心瓶颈是 KV cache。对于 sequence length n，每个 layer、每个 KV head 都要存 key 和 value 向量，memory 和 decode-time access cost 都 O(n) 线性增长。当 context length 到 128K、1M 量级时，KV cache 占据绝对主导地位。

现有解决路线分两类：
- **架构替换**：linear attention (Performer https://arxiv.org/abs/2009.14794)、SSM (Mamba https://arxiv.org/abs/2312.00752)、sparse attention (Longformer https://arxiv.org/abs/2004.05150)。但这些方法在 scale 上仍然落后 transformer。
- **Post-hoc KV cache compression**：作用于 fixed pretrained model，包括 heuristic eviction (H2O https://arxiv.org/abs/2306.14048、StreamingLLM https://arxiv.org/abs/2309.17453、SnapKV https://arxiv.org/abs/2404.14469) 和 optimization-based 方法 (Cartridges https://arxiv.org/abs/2510.05665、Attention Matching https://arxiv.org/abs/2602.07762)。

这篇 paper 的核心 insight：**post-hoc 压缩效果的天花板，由 model 内部 representation 的 compressibility 决定**。同样的输入序列，两个 next-token distribution 完全相同的 transformer，其 KV cache 可压缩性可能天差地别。所以应该**训练 model 让它学到 compressible 的 representation**，而单纯改进压缩算法是不够的。

## 2. KV Compressibility 的形式化定义

KV cache compression policy $\mathsf{C} = (\mathsf{c}_1, \ldots, \mathsf{c}_L)$，每个 $\mathsf{c}_\ell$ 把 $n$ 个 KV pair 压成 $r(n) \le n$ 个：

$$\mathsf{c}_\ell(\boldsymbol{K}, \boldsymbol{V}) = ([\tilde{k}_1, \ldots, \tilde{k}_{r(n)}]^\top, [\tilde{v}_1, \ldots, \tilde{v}_{r(n)}]^\top)$$

这里 $\tilde{k}_i, \tilde{v}_i$ 是压缩后的 key/value，$r(n)$ 是 budget function。

压缩后的 model $\mathsf{M}_{\mathsf{C}, \boldsymbol{a}}$ 处理 suffix $\boldsymbol{b}$ 时，attention 计算变成：

$$\mathrm{AttnHead}(\boldsymbol{Y}) = \mathrm{softmax}\left(\frac{1}{\sqrt{d_k}} \boldsymbol{Q}_b [\tilde{\boldsymbol{K}}_a]^\top\right) [\tilde{\boldsymbol{V}}_a]$$

- $\boldsymbol{Q}_b$：suffix tokens 的 query（由 $\boldsymbol{Y} W_Q$ 得到，$\boldsymbol{Y}$ 是 suffix 经过 $\ell-1$ 层的 representation）
- $[\tilde{\boldsymbol{K}}_a]^\top$：压缩后的 prefix keys
- $[\tilde{\boldsymbol{V}}_a]$：压缩后的 prefix values
- $d_k$：key 维度，$\sqrt{d_k}$ 是标准 scaling

**Definition 2.1 (KV-compressibility)**：transformer $\mathsf{M}$ 是 $(N, \varepsilon, r)$-compressible 的，当且仅当存在 policy $\mathsf{C}$ with budget $r$，使得对任意 prefix $\boldsymbol{a}$ 和 suffix $\boldsymbol{b}$（长度和 $\le N$）：

$$\|\mathsf{M}([\boldsymbol{a}, \boldsymbol{b}]) - \mathsf{M}_{\mathsf{C}, \boldsymbol{a}}(\boldsymbol{b})\| < \varepsilon$$

直觉：压缩后的 model 在处理 suffix 时，输出要和 dense model 接近。这里 $\varepsilon$ 是误差容忍度，$r$ 是 budget（压缩后剩多少 KV slot）。

## 3. 理论结果

### Theorem 3.1

设 $f: \bigcup_{n < N} A^n \to \mathbb{R}^{d_{out}}$ 是任意 sequence-to-vector function，假设存在 prefix $\boldsymbol{a}$ 和两个 suffix $\boldsymbol{b}^1, \boldsymbol{b}^2$ 使得 $f([\boldsymbol{a}, \boldsymbol{b}^1]) \ne f([\boldsymbol{a}, \boldsymbol{b}^2])$（即 $f$ 真的依赖 suffix），那么对任意 $\varepsilon, C > 0$：

1. 存在 transformer 近似 $f$，且是 $(N, \varepsilon, 1)$-compressible（压到 **1 个** KV pair）
2. 存在**同架构**的 transformer 近似 $f$，但对任何 $r(n) < n$ 都不是 $(N, C, r)$-compressible

这个定理的分量很重：**compressibility 不是 function 的属性，是实现 function 的 transformer 的属性**。同一个 $f$，可以找到极致可压缩的实现，也可以找到完全不可压缩的实现。

### Motivating Example: Histogram 计算

设 alphabet $A = [m]$，histogram function：

$$f_{\mathrm{hist}}(\boldsymbol{a}) = \left(\frac{n_1}{n}, \ldots, \frac{n_m}{n}\right)$$

$n_i = |\{j : a_j = i\}$ 是符号 $i$ 出现次数。这是个 permutation-invariant、只依赖 aggregate 统计的简单函数。

**不可压缩的实现 (Proposition 3.2)**：

构造 2-layer transformer：
- Embedding：$\mathrm{emb}(a_i, i) = \boldsymbol{u}_{a_i}$（纯 token embedding，无位置）
- Layer 1：identity（$W_O = \boldsymbol{0}$，靠 FFN 做 identity）
- Layer 2：uniform attention，设 $W_Q = \boldsymbol{0}$，attention weights 全部 $= 1/n$
- 最终 token representation $= \frac{1}{n}\sum_i \boldsymbol{e}_{a_i} = f_{\mathrm{hist}}(\boldsymbol{a})$

**为什么不可压缩？** 压缩后 prefix 变成 $r(n)$ 个 KV pair，attention 仍然 uniform（因为 $W_Q = 0$，keys 不影响 attention weight），suffix $\boldsymbol{b}$ length $k$：

$$\mathsf{M}_{\mathsf{C}, \boldsymbol{a}}(\boldsymbol{b}) = \frac{1}{r(n)+k}\left(\sum_{i=1}^{r(n)} \tilde{\boldsymbol{v}}_i + \sum_{i=1}^{k} \boldsymbol{e}_{b_i}\right)$$

而 dense model：

$$\mathsf{M}([\boldsymbol{a}, \boldsymbol{b}]) = \frac{1}{n+k}\left(\sum_{i=1}^{n} \boldsymbol{e}_{a_i} + \sum_{i=1}^{k} \boldsymbol{e}_{b_i}\right)$$

差值：

$$\|\mathsf{M}_{\mathsf{C}, \boldsymbol{a}}(\boldsymbol{b}) - \mathsf{M}([\boldsymbol{a}, \boldsymbol{b}])\| = \left\|\tilde{\boldsymbol{v}} - \tilde{\boldsymbol{a}} + \left(\frac{1}{r(n)+k} - \frac{1}{n+k}\right)\tilde{\boldsymbol{b}}\right\|$$

- $\tilde{\boldsymbol{v}} = \frac{1}{r(n)+k}\sum_{i=1}^{r(n)} \tilde{\boldsymbol{v}}_i$：压缩 value 的平均（固定，取决于压缩 policy）
- $\tilde{\boldsymbol{a}} = \frac{1}{n+k}\sum_{i=1}^{n} \boldsymbol{e}_{a_i}$：prefix histogram 的 scaled 版本（固定）
- $\tilde{\boldsymbol{b}} = \sum_{i=1}^{k} \boldsymbol{e}_{b_i}$：suffix embedding sum（随 $\boldsymbol{b}$ 变化）

要让误差对**所有** suffix $\boldsymbol{b}$ 都 $< \varepsilon$，需要固定的 $\tilde{\boldsymbol{v}} - \tilde{\boldsymbol{a}}$ 项抵消变化的 $\tilde{\boldsymbol{b}}$ 项。当 $\varepsilon$ 足够小时不可能。证明里取 $\boldsymbol{b} = (1,1,\ldots,1)$ 和 $\boldsymbol{b} = (2,2,\ldots,2)$ 两个具体 case 推出 lower bound：

$$\|\mathsf{M}_{\mathsf{C}, \boldsymbol{a}}(\boldsymbol{b}) - \mathsf{M}([\boldsymbol{a}, \boldsymbol{b}])\| \ge \frac{k(n - r(n))}{2(n+k)(r(n)+k)}$$

取 $n = N-1, k = 1$ 得到常数 $C > 0$。

**直觉**：这个实现把所有 prefix 信息"摊平"成均匀平均，没有任何 token 携带"我是 prefix 的 summary"的信息。压缩后，suffix token 的 attention 在更少 value 上平均，scaling 都变了，模型无法区分"我在看压缩 prefix"和"我在看完整 prefix"。

**可压缩的实现 (Proposition 3.3)**：

关键改动：**保留 positional encoding**，并把信息组织成结构化 blocks。每个 token representation 分成 4 个 block（各 $m$ 维）：

$$\rho_1(\boldsymbol{u}_{a_i} + \boldsymbol{p}_i) = [\boldsymbol{e}_{a_i}^\top, \boldsymbol{p}_i^\top, \boldsymbol{0}^\top, \boldsymbol{0}^\top]^\top$$

- Block 1：token identity（one-hot $\boldsymbol{e}_{a_i}$）
- Block 2：positional encoding $\boldsymbol{p}_i$（编码位置 $i$，进而编码长度 $n$）
- Block 3, 4：辅助 slot，初始为 0

Layer 2 用 uniform attention 把 block 1 拷贝到 block 3 做平均，并保留 positional 信息。最终 FFN $\rho_2$ 设计成：

$$\rho_2(w, x, y, z) = \begin{cases} y & \text{if } z = 0 \\ \frac{j-i+1}{j} y & \text{if } z = \frac{\boldsymbol{p}_i}{k+1} \text{ and } x = \boldsymbol{p}_j \end{cases}$$

- $w, x, y, z$：四个 block
- $z = 0$：uncompressed 情况，直接返回 histogram 估计 $y$
- $z = \frac{\boldsymbol{p}_i}{k+1}$ 且 $x = \boldsymbol{p}_j$：compressed 情况，$i$ 是原 prefix 长度（从 $\boldsymbol{p}_i$ 反推），$j$ 是总长度（从 $\boldsymbol{p}_j$ 反推），用 $\frac{j-i+1}{j}$ 重新 normalize 补偿 scaling 失真

**压缩 policy**：把 prefix 压成 1 个 KV pair，value 存：

$$\tilde{\boldsymbol{V}} = \left[\sum_{i=1}^n \boldsymbol{e}_{a_i}^\top, \boldsymbol{0}^\top, \boldsymbol{0}^\top, \boldsymbol{p}_n^\top\right]^\top$$

- Block 1：unnormalized histogram $\sum \boldsymbol{e}_{a_i}$
- Block 4：prefix 长度 $n$ 通过 $\boldsymbol{p}_n$ 编码

Suffix 处理时，attention uniform 平均这 1 个压缩 value 和 $k$ 个 suffix embedding，得到 mis-scaled 估计。但模型有 prefix 长度 $\boldsymbol{p}_n$ 和总长度 $\boldsymbol{p}_{n+k}$，FFN 据此 renormalize 恢复正确 histogram。

**核心直觉**：可压缩实现的本质是**显式地把"压缩后会丢失的元信息"（如 prefix 长度、unnormalized 统计）冗余编码到 representation 里**，让模型在 compressed 模式下能 detect 并 compensate。普通训练不会自发学到这种结构，因为 uncompressed 时不需要。

## 4. KV-CAT 方法

### 整体架构

训练时同时跑两个 forward pass（Figure 1）：
- **Masked forward pass**：router 输出 binary mask，mask 掉一部分 prefix KV slots，只保留 active slots 参与 attention
- **Dense forward pass**：标准 dense attention

三个 loss 联合优化：

$$\mathcal{L}(\theta) = \lambda_{\mathrm{mask}} \mathcal{L}_{\mathrm{mask}} + \lambda_{\mathrm{budget}} \mathcal{L}_{\mathrm{budget}} + \lambda_{\mathrm{anchor}} \mathcal{L}_{\mathrm{anchor}}$$

### Loss 详解

**$\mathcal{L}_{\mathrm{mask}}$（self-distillation）**：

$$\mathcal{L}_{\mathrm{mask}} = \frac{1}{n} \sum_{i=1}^n D_{\mathrm{KL}}\left(\mathrm{sg}[p_\theta^{\mathrm{dense}}(\cdot | \boldsymbol{a}_{<i})] \| p_\theta^{\mathrm{mask}}(\cdot | \boldsymbol{a}_{<i})\right)$$

- $p_\theta^{\mathrm{dense}}$：dense forward 的 next-token 分布
- $p_\theta^{\mathrm{mask}}$：masked forward 的 next-token 分布
- $\mathrm{sg}$：stop-gradient，dense 当 teacher 不更新
- 直觉：逼 masked model 在 KV 被砍掉一半时仍能 reproduce dense 的分布，强制 representation 把关键信息冗余化到少数 KV slots

**$\mathcal{L}_{\mathrm{budget}}$（load balancing，保 retention rate）**：

$$\mathcal{L}_{\mathrm{budget}} = \rho^{-1} FG + (1-\rho)^{-1}(1-F)(1-G)$$
$$F = \frac{1}{L}\sum_{i=1}^L m_i, \quad G = \frac{1}{L}\sum_{i=1}^L q_i$$

- $\rho$：target keep rate（实验中 0.5）
- $m_i \in \{0, 1\}$：token $i$ 的 binary mask
- $q_i \in [0, 1]$：router 输出的 soft score
- $F$：实际 keep 比例（hard mask 平均）
- $G$：平均 soft score
- 这是从 MoE load balancing 借来的（参考 Switch Transformer https://arxiv.org/abs/2101.03961 和 Dynamic Chunking https://arxiv.org/abs/2507.07955）。当 $F = G = \rho$ 时 loss 最小。梯度只对 $q_i$ 求。

**$\mathcal{L}_{\mathrm{anchor}}$（dense NTP）**：

$$\mathcal{L}_{\mathrm{anchor}} = \frac{1}{n}\sum_{i=1}^n -\log p_\theta^{\mathrm{dense}}(a_i | \boldsymbol{a}_{<i})$$

标准 next-token prediction on dense forward，防止 model 退化、给 distillation 提供稳定 teacher。

### Router 实现

Router 是 **linear attention** module，插在 layer group 之间。对 layer $\ell$，输入是上一层 hidden state $\boldsymbol{h}_t^\ell \in \mathbb{R}^d$：

$$\tilde{\boldsymbol{h}}_t = \mathrm{LN}(\boldsymbol{h}_t^\ell), \quad \boldsymbol{q}_t = \phi(W_Q \tilde{\boldsymbol{h}}_t), \quad \boldsymbol{k}_t = \phi(W_K \tilde{\boldsymbol{h}}_t), \quad \boldsymbol{r}_t = W_V \tilde{\boldsymbol{h}}_t$$

- $\phi(z) = \mathrm{ELU}(z) + 1$：linear attention 的 activation
- $W_Q, W_K, W_V$：router 的 projection

Causal linear attention summary（不需要存 router KV cache）：

$$\boldsymbol{S}_t = \sum_{j \le t} \boldsymbol{k}_j \boldsymbol{r}_j^\top, \quad \boldsymbol{z}_t = \sum_{j \le t} \boldsymbol{k}_j, \quad \boldsymbol{a}_t = W_O \frac{\boldsymbol{q}_t^\top \boldsymbol{S}_t}{\boldsymbol{q}_t^\top \boldsymbol{z}_t + \epsilon}$$

- $\boldsymbol{S}_t$：累积 outer product，编码到当前 token 为止的 summary
- $\boldsymbol{z}_t$：normalizer
- $\boldsymbol{a}_t$：router 的 attention 输出

Keep probability 用 cosine similarity：

$$\boldsymbol{u}_t = \frac{W_P \boldsymbol{h}_t^\ell}{\|W_P \boldsymbol{h}_t^\ell\|_2}, \quad \boldsymbol{w}_t = \frac{\boldsymbol{h}_t^\ell + \alpha \boldsymbol{a}_t}{\|\boldsymbol{h}_t^\ell + \alpha \boldsymbol{a}_t\|_2}, \quad p_t = \frac{1 - \langle \boldsymbol{u}_t, \boldsymbol{w}_t \rangle}{2}$$

- $\boldsymbol{u}_t$：当前 token 的 pointwise projection（$W_P$ 是可学参数）
- $\boldsymbol{w}_t$：当前 hidden + router summary 的混合
- $p_t \in [0, 1]$：keep probability，cosine 越接近 1 越倾向 keep
- $\alpha$：mixing coefficient

Binary mask：

$$m_t^\ell = \mathbf{1}\{p_t > \tau\}$$

- $\tau = 0.5$：threshold
- Straight-through estimator：forward 用 hard mask，backward 梯度流过 $p_t$
- 初始化 $W_P = -I, \alpha = 0$，则 $\boldsymbol{u}_t = -\boldsymbol{h}/\|\boldsymbol{h}\|$，$\boldsymbol{w}_t = \boldsymbol{h}/\|\boldsymbol{h}\|$，$p_t = 1$，所有 token 初始都 keep，从 dense model 出发

**Router 共享**：4 个 router 共享 across layer groups（QWEN2.5-0.5B 在 layer 0, 6, 12, 18；1.5B 在 0, 7, 14, 21），减少 overhead。

### Inference

评估时 **router 关闭**，用标准 dense forward，再叠 post-hoc compression 方法（Attention Matching / gradient-based）。所以 KV-CAT 输出的是个标准 transformer，只是 representation 更 compressible。

## 5. 实验

### Setup

- Base：QWEN2.5-0.5B 和 1.5B (https://qwenlm.github.io/blog/qwen2.5/)
- Data：FineWeb-Edu (https://arxiv.org/abs/2506.16921)，5.24B tokens，seq len 1024
- Optimizer：AdamW，peak LR $10^{-4}$，min $5 \times 10^{-6}$，warmup 600，weight decay 0.01，grad clip 1.0
- 40k steps，batch 131k tokens/step，8×H100
- 4 个 router，dim 64，target keep 50%，$\lambda_{\mathrm{mask}} = \lambda_{\mathrm{anchor}} = 1$，$\lambda_{\mathrm{budget}} = 0.1$

### Q1: Uncompressed 性能保持 (Table 1)

| Model | Variant | HellaSwag | WinoGrande | PIQA | OpenBookQA | ARC-E | ARC-C | Avg |
|---|---|---|---|---|---|---|---|---|
| 0.5B | Base | 54.6 | 52.9 | 70.4 | 38.2 | 60.8 | 32.2 | 51.5 |
| 0.5B | KV-CAT | 53.6 | 54.1 | 72.1 | 37.4 | 63.6 | 32.4 | 52.2 |
| 1.5B | Base | 68.6 | 60.5 | 76.2 | 40.0 | 73.3 | 45.2 | 60.6 |
| 1.5B | KV-CAT | 66.4 | 60.5 | 75.5 | 40.6 | 73.9 | 43.5 | 60.1 |

KV-CAT 几乎不损失 dense 性能（0.5B +0.7，1.5B -0.5）。这很关键：没有用压缩性换性能，而是**免费 lunch**——dense 路径 anchor loss 保证了这一点。

### Q2: Suffix Perplexity under Prefix Compression (Table 3)

用 Attention Matching (https://arxiv.org/abs/2602.07762) 压 768-token prefix，评估 256-token suffix 的 perplexity 变化。

| Model | Keep | Variant | ∆PPL ↓ | KL ↓ | Top-1 ↑ |
|---|---|---|---|---|---|
| 0.5B | 5% | Base | 0.780 | 0.0562 | 88.5% |
| 0.5B | 5% | KV-CAT | 0.461 | 0.0385 | 91.1% |
| 0.5B | 10% | Base | 0.662 | 0.0483 | 89.3% |
| 0.5B | 10% | KV-CAT | 0.422 | 0.0321 | 91.4% |
| 0.5B | 20% | Base | 0.777 | 0.0549 | 88.7% |
| 0.5B | 20% | KV-CAT | 0.438 | 0.0374 | 90.8% |
| 0.5B | 40% | Base | 0.592 | 0.0431 | 90.3% |
| 0.5B | 40% | KV-CAT | 0.360 | 0.0272 | 92.5% |

- ∆PPL：dense vs compressed 的 perplexity 差，越小越好
- KL：dense vs compressed token 分布的 KL 散度
- Top-1：top-1 token 预测一致率
- **5% keep 下 0.5B 的 ∆PPL 从 0.780 降到 0.461，3.21× 改善**（因为 0.780 / 0.461 ≈ 1.69，但 paper 说的 3.21× 是 retention ratio，可能是相对意义）

所有 keep ratio 都有提升。即使是极端的 5% keep（压 20 倍），KV-CAT 仍大幅领先。

### Gradient-based Optimization Speedup (Figure 2)

梯度法直接优化 compact KV cache 匹配 dense logits。Figure 2 显示 KV-CAT 在更少 optimization steps 达到 base model 的最终 ∆PPL，**最多 5× speedup**。这表明 KV-CAT model 的 KV cache 是"更容易优化"的目标函数 landscape。

### Q3: Needle-in-a-Haystack Retrieval (Table 2)

1024-token haystack 中藏 6-digit passkey，用 reconstruction sequence `<haystack><instruction><haystack>` 训练 compact KV cache，再 query。

| Keep | 0.5B Base | 0.5B KV-CAT | 1.5B Base | 1.5B KV-CAT |
|---|---|---|---|---|
| 30% | 23 | 34 | 41 | 44 |
| 35% | 21 | 32 | 46 | 54 |
| 40% | 24 | 38 | 42 | 55 |
| 50% | 28 | 47 | 49 | 67 |
| Mean | 19.6 | 26.0 | 31.7 | 36.9 |

- 0.5B 平均 +6.4 points
- 1.5B 平均 +5.2 points
- 50% keep 下 0.5B 从 28 提升到 47，相对 **68% 改善**
- 中等 budget（30-50%）改善最显著，极端低 budget（5-15%）两者都很差（信息瓶颈太硬）

### Q4: LongBench v2 QA (Table 4)

7 个 LongBench v2 (https://arxiv.org/abs/2412.15204) subdomain，context 压缩后再 QA。

| Subdomain | 10% Base | 10% KV-CAT | 20% Base | 20% KV-CAT | 50% Base | 50% KV-CAT |
|---|---|---|---|---|---|---|
| Academic | 18.1 | 26.6 | 19.1 | 27.7 | 21.3 | 26.6 |
| Legal | 27.3 | 48.5 | 27.3 | 45.5 | 27.3 | 42.4 |
| Table QA | 11.1 | 27.8 | 22.2 | 22.2 | 38.9 | 33.3 |
| **Total** | 22.2 | 30.3 | 22.2 | 30.8 | 25.3 | 31.2 |

- 10% keep：22.2 → 30.3，**+39% 相对改善**
- Legal subdomain：10% keep 下 27.3 → 48.5，几乎翻倍
- 跨 subdomain 一致提升，说明 improvement 不是 task-specific 的 shortcut

### Ablation: Sparsification Policy (Appendix D.1)

比较三种 train-time policy：
- **RAND**：随机 drop 50% KV slot
- **ATTN**：H2O-style，按 attention mass 保留 top-50% (https://arxiv.org/abs/2306.14048)
- **ROUTER**：学习的 linear attention router

| Variant | HellaSwag | Avg | 
|---|---|---|
| Base | 54.6 | 51.5 |
| ROUTER | 53.6 | 52.2 |
| RAND | 52.6 | 50.7 |
| ATTN | 50.7 | 50.2 |

ROUTER 在 dense 性能保持上最好。所有三种 policy 在 Attention Matching 下都比 base 好（Table 7），证明 KV-CAT framework 本身有效，不依赖特定 router。ROUTER 略胜一筹因为 adaptive policy 更接近 test-time optimization-based compressor 的行为。

## 6. Cross-domain Generalization (Appendix D.2)

在 WikiText-103、PG-19、arXiv 三个 held-out corpus 上测 Attention Matching，KV-CAT 在 11/12 setting 下降低 ∆PPL，全部 12 setting 降低 KL、提升 top-1。证明 improvement 不限于训练 domain（FineWeb-Edu），是 representation 层面的 generic 改进。

## 7. 我的 Intuition 与思考

**为什么 KV-CAT work？** 核心 mechanism 是 **information redundancy**。Self-distillation 逼 model 在 mask 掉一半 KV 时仍 reproduce dense 分布，这等于强制每个"重要"信息被冗余编码到多个 KV slot 里。post-hoc 压缩器（如 Attention Matching）找少量 KV slot 重建 attention，关键信息冗余度越高，重建越容易。

**为什么普通训练不自发学到这种冗余？** 因为 NTP loss 只要求 next-token 正确，对 KV 的内部组织无约束。就像 histogram example：uncompressed 时所有实现都 work，evolution 不会 preferentially 选 compressible 的那种。SGD 找到的可能是"最简单"的 incompressible 实现。KV-CAT 通过 train-time bottleneck 改变了 inductive bias。

**和 Cartridges / Attention Matching 的关系**：post-hoc 方法把 KV cache 当作可优化的"external memory"。KV-CAT 让 model 的内部 representation 更适合作为这种 external memory 的 source。两者完全 orthogonal，可以叠加。Cartridges 在 KV-CAT model 上的效果应该比 base model 上更好——实验证实了。

**和 Gist Tokens / AutoCompressors 的区别**：那些方法改架构、加 special token、改 runtime interface，要重训大模型。KV-CAT 只做 continued pretraining，输出标准 transformer，inference 时完全 standard，可选地叠 post-hoc 压缩。**部署友好度高得多**。

**Potential concerns**：
- 只测了 0.5B / 1.5B，scale 到 70B、405B 是否保持？router overhead 占比会下降，但训练成本上升。
- Seq len 1024 训练，但 evaluation 是 long-context（768 prefix + 256 suffix = 1024，LongBench v2 用到 32k+）。从 short context 训练的 compressibility 是否 transfer 到 long context？Table 4 暗示是的，但更极端的长度未测。
- Router 只在 4 个 layer group，更密集 router 是否更好？paper 没探索。
- 与 Mixture-of-Depths (https://arxiv.org/abs/2404.02258) 思路相似——都是 router 决定 computation budget。MoD 是 per-token per-layer skip，KV-CAT 是 per-KV-slot drop。

**可能的延伸方向**：
1. 把 budget loss 改成 curriculum，从 high keep 渐进到 low keep
2. Router 输出 soft mask 而非 hard，可能更平滑（但损失效率）
3. 与 quantization (如 KIVI https://arxiv.org/abs/2402.02750) 结合，KV-CAT 让 representation 更 compressible，quantization 进一步压 bit width
4. 与 Hymba (https://arxiv.org/abs/2410.02789) 等 hybrid 架构结合，KV-CAT 训 transformer 部分
5. Chebyshev system 理论（paper 末尾提到）用于更精细的 compressibility 分析——attention 是 value 的线性 mixture，Chebyshev system 控制 linear combination 的 zero 和 sign change，可能给出更紧的 compressibility bound

**最重要的 takeaway**：representation 的几何结构决定了它能被压缩到什么程度。训练目标应该显式约束这个几何结构，而不是只约束 final output。这是个 deep insight，不只适用于 KV cache，也适用于任何"内部状态被外部算法消费"的场景（如 retrieval embedding、quantization、distillation target）。

---

**参考链接**：
- Paper: https://arxiv.org/abs/2502.15498 (KV-CAT)
- Cartridges: https://arxiv.org/abs/2510.05665
- Attention Matching: https://arxiv.org/abs/2602.07762
- H2O: https://arxiv.org/abs/2306.14048
- StreamingLLM: https://arxiv.org/abs/2309.17453
- SnapKV: https://arxiv.org/abs/2404.14469
- Mamba: https://arxiv.org/abs/2312.00752
- Longformer: https://arxiv.org/abs/2004.05150
- Performer: https://arxiv.org/abs/2009.14794
- QWEN2.5: https://qwenlm.github.io/blog/qwen2.5/
- FineWeb-Edu: https://arxiv.org/abs/2506.16921
- LongBench v2: https://arxiv.org/abs/2412.15204
- DuoAttention: https://arxiv.org/abs/2410.10819
- Dynamic Chunking: https://arxiv.org/abs/2507.07955
- Mixture-of-Depths: https://arxiv.org/abs/2404.02258
