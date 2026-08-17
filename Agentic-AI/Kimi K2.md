---
source_pdf: Kimi K2.pdf
paper_sha256: 035062b5e1ab865ffad9f90f81d6c8332f8af805063fac54cc6f2cca1fe608b5
processed_at: '2026-08-05T11:09:25-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Kimi K2 人话版

好，我把刚才那堆公式和术语翻译成人话。

---

## 一句话总结

Moonshot 做了个 1万亿参数的 MoE 模型，每次只激活 32B，主打 agentic 能力（写代码、调工具），在开源模型里 SOTA，逼近 Claude 4。

---

## 他们到底干了啥

### 1. MuonClip：训练不炸的秘密

**背景问题**：

训练大模型有个老毛病——训练到一半 loss 突然 spike（暴涨），有时候直接炸飞，前面几十亿 token 白训了。

Moonshot 选了 Muon 这个 optimizer，因为它 token efficiency 好——同样算力下学得更多。但 Muon 有个 bug：scale up 之后 attention 的 logits 会疯涨，涨到 1000+，然后炸。

为啥 Muon 比 Adam 更容易炸？这里有个很妙的洞察：

Adam 的 gradient update 是 "低秩" 的——少数几个方向主导，大部分方向没动。Muon 不一样，它把 gradient 正交化后，所有方向均匀更新，是 "满秩" 的。

满秩更新听起来更全面，但有个副作用：weight matrix 的 singular values 会 **加性增长**（因为 update 方向和已有 weight 方向对齐的概率更高）。而 attention 的 logit 是 $Q \cdot K$，涉及 $W_q$ 和 $W_k$ 两个矩阵的乘积，spectral norm 会被 **平方放大**。所以 Muon 训练的权重 singular values 涨一点，logits 就涨很多。

**他们的解法：QK-Clip**

思路特别直接：每步 forward 时反正已经算了 attention logits，顺手看一下最大的那个 logit 是多少。如果超过阈值（他们设 100），就把 $W_q$ 和 $W_k$ 缩小一点，让 logits 降回来。

关键细节：
- 不改当前 step 的 forward/backward，只改权重本身
- 只对炸掉的那个 head clip，不波及其他 head
- 训练初期 ~12.7% 的 head 会触发 clip，训到 30% 进度后所有 head 自己就稳定了，clip 自动失效

就像是给训练装了个 "安全阀"：前期压力大的时候帮你泄一下，等模型自己找到 stable region 就不管了。

结果：15.5T tokens 训完，**零 loss spike**。Figure 3 的 loss curve 平滑得像画出来的。

---

### 2. 架构选择：更 sparse，更省

K2 的架构说白了就是 DeepSeek-V3 的变体，但做了几个 "更极端" 的选择：

**专家更多，激活更少**：384 个 expert 选 8 个（sparsity 48:1），vs DeepSeek-V3 的 256 选 8（32:1）。

他们的 scaling law 实验发现：固定 FLOPs（固定激活参数），专家越多效果越好。因为更多 expert = 更多 "知识格子"，router 可以更精准地选相关的格子，比把所有知识塞进少数几个 expert 里强。

sparsity 48 达到同样 loss，比 sparsity 8 省 1.69× FLOPs。

**Attention heads 砍半**：128 → 64。

原因是 agentic 场景要处理长上下文（128k），heads 翻倍会让推理 FLOPs 涨 83%。实验显示 heads 翻倍只带来 0.5-1.2% 的 loss 改善，ROI 太低。反正 384 个 expert 已经提供了大量 capacity，attention heads 的边际收益递减。

---

### 3. 数据：把每个 token 榨干

高质量人类数据快用完了。重复训练会 overfit。怎么办？

**改写**：用 LLM 把同一段文本换个风格、换个视角重写一遍。10 次改写 + 训 1 遍 > 原文训 10 遍。

道理很简单：同一句话看 10 遍，模型只是死记硬背；同一件事用 10 种方式表达，模型学到的是 "概念" 而不是 "字面"。

数学数据也类似，把文档改写成 "学习笔记" 风格，加上跨语言翻译增加多样性。

---

### 4. Post-training：教模型用工具

这是 K2 的重头戏。要搞 agentic intelligence，模型得会用工具、会写代码、会多步推理。但这类数据自然界稀缺。

**合成数据 pipeline**：

1. 收集 3000+ 真实 MCP 工具 + 合成 20000+ 假工具
2. 给每个工具组合造一个 agent，再造配套任务
3. 模拟 agent 执行任务的过程，包括用户交互、工具调用、状态变化
4. 用 LLM judge 过滤，只留高质量 trajectory

对 coding 这种需要真实反馈的场景，用真 sandbox（Kubernetes 跑 10000+ 并发实例，unit test pass rate 当 ground truth）。

**RL 框架**：

两部分：
- **Verifiable rewards**：数学题有标准答案，代码能跑过测试，instruction following 有规则可查
- **Self-critique rubric reward**：创意写作这种没标准答案的，让模型自己当 judge

self-critique 的巧妙之处在于闭环：用 verifiable tasks 的信号来校准 critic 的判断力，再让 critic 去指导 non-verifiable tasks 的训练。相当于把 "客观信号" 通过 critic 这个中介传导到 "主观领域"。

RL 还有三个实用 trick：
- **Budget control**：限制 response 长度，防止模型话痨（RL 的 known issue）
- **PTX loss**：加一点高质量 SFT 数据防止遗忘
- **Temperature decay**：早期高温探索，后期低温收敛

---

### 5. 工程：1T 模型的 RL 训练怎么搞

RL 训练要交替跑 inference（生成数据）和 training（更新参数），两个 engine 共用 GPU。

1T 模型的参数在 train 和 inference 之间切换是个噩梦——sharding 方式不同，网络传输巨大。

他们搞了个 **checkpoint engine**：每个 train node 上放一个，负责收集本地参数 → broadcast 全集群 → inference engine 按需取 shard。流水线化操作，整个参数更新 < 30 秒。

还搞了 partial rollout：长 trajectory 跑不完就暂停，下一轮接着跑，不阻塞整个 batch。

---

## 结果怎么样

非 thinking mode 下：

| 能力 | K2 | 对标 |
|------|-----|------|
| SWE-bench Verified | 65.8% | Claude Sonnet 4: 72.7% |
| τ2-bench (工具调用) | 66.1 | 开源最强 |
| LiveCodeBench v6 | 53.7% | 开源最强 |
| AIME 2025 | 49.5% | 开源最强 |
| GPQA-Diamond | 75.1% | 开源最强 |

LMSYS Arena 排名：开源第 1，总榜第 5。

差距：和 Claude 4 Opus/Sonnet 在 agentic 任务上还差 5-10 个点，但已经是开源里最接近的。

---

## 我的直觉总结

K2 这篇 paper 的核心 takeaway：

**MuonClip 是真正的技术贡献**。它不只是 "又一个 trick"，而是揭示了 optimizer 的谱特性和 attention 结构之间的深层关系。这个 insight 对整个社区都有价值——选 optimizer 不能只看 loss 下降速度，还要看它怎么塑造 weight matrix 的 spectral structure。

**Sparsity 48 是激进但合理的选择**。在 expert 数和 infra 复杂度之间找了个平衡点。继续往上推（比如 64:1、128:1）的瓶颈不在模型设计，而在通信和路由。

**Agentic data synthesis 是工程实力体现**。没有花哨的理论，就是系统性地造数据、过滤、训练。但这种 "boring engineering" 往往是拉开差距的关键。

**Non-thinking 路线是有意识的 trade-off**。K2 把 reasoning 能力 bake 进权重，而不是靠 test-time 展开。inference 更便宜、更可控，但上限可能不如 thinking models（R1, o1 那种）。这个赌注对不对，还得看后续。

一句话：K2 是 Moonshot 用工程能力把已知 idea 推到 extreme 的产物——Muon 推到 1T scale，MoE sparsity 推到 48，agentic data 推到 20000+ tools。没有单一 silver bullet，但每一步都 solid。

---

# Kimi K2 技术深度解析

这篇 paper 是 Moonshot AI 发布的 Kimi K2 技术报告，核心定位是 **open agentic intelligence**。1.04T 参数的 MoE 模型，32B 激活参数，在 agentic tasks（SWE-bench, τ2-bench, ACEBench）上达到 open-source SOTA，close the gap with Claude 4 Opus/Sonnet。我下面按照技术维度逐一拆解，重点 build intuition。

---

## 1. 核心创新概览

Kimi K2 的三个主要技术贡献：

1. **MuonClip optimizer**：把 Muon optimizer 和 QK-Clip 结合，解决 Muon 在大规模训练时的 attention logits explosion 问题，实现 15.5T tokens 零 loss spike 训练
2. **Agentic data synthesis pipeline**：大规模合成 tool-use 数据，3000+ real MCP tools + 20000+ synthetic tools
3. **General RL framework**：结合 verifiable rewards 和 self-critique rubric reward

参考链接：
- Paper: https://arxiv.org/abs/2507.18841
- Muon 原始 blog: https://kellerjordan.github.io/posts/muon/
- Moonlight (Muon scaling): https://arxiv.org/abs/2502.16982
- DeepSeek-V3: https://arxiv.org/abs/2412.19437

---

## 2. MuonClip：这篇 paper 的灵魂

### 2.1 背景：为什么用 Muon

Muon 是 Keller Jordan 提出的 optimizer（https://kellerjordan.github.io/posts/muon/），核心 idea 是对 hidden layers 的 weight gradient 做 Newton-Schulz 迭代来近似矩阵的 sign function（即 orthogonalization），产生类似 "steepest descent on Stiefel manifold" 的更新。

Moonshot 自己的 Moonlight 工作（https://arxiv.org/abs/2502.16982）已经证明：在相同 compute budget 和 model size 下，Muon substantially outperforms AdamW。这意味着 **token efficiency** 更高——每个 token 带来的 learning signal 更多。在高质量人类数据日益稀缺的今天，token efficiency 成为 scaling 的 critical coefficient。

Muon 的核心 update rule（Algorithm 1）：

```
M_t = μ·M_{t-1} + G_t                    # momentum accumulation
O_t = NewtonSchulz(M_t) · √max(n,m) · 0.2  # orthogonalize + RMS match to Adam
W_t = W_{t-1} - η·(O_t + λ·W_{t-1})       # update with weight decay
```

这里：
- $M_t$ 是 momentum buffer，$\mu$ 是 momentum coefficient
- $G_t$ 是当前 step 的 gradient
- $NewtonSchulz(\cdot)$ 是 Newton-Schulz 迭代，近似 $M_t$ 的 matrix sign function（即 $M_t(M_t^TM_t)^{-1/2}$）
- $\sqrt{\max(n,m)}$ 和 $0.2$ 是为了 match Adam 的 update RMS（Adam 的典型 update RMS 约 0.1-0.3）
- $\lambda$ 是 weight decay
- $\eta$ 是 learning rate

### 2.2 问题：Muon 在 scale up 时会 logit explosion

实验发现（Figure 2 Left）：在 9B activated / 53B total 的 MoE 上用 vanilla Muon 训练，max attention logits 很快超过 1000。这种级别的 logit 通常导致 loss spike 甚至 divergence。

为什么已有的方法不够用：
- **Logit soft-cap**（Gemma 2, https://arxiv.org/abs/2408.00118）：直接 clip attention logits，但 $Q \cdot K$ dot product 在 cap 之前已经长得过大，gradient 信号不对
- **QK-Norm**（https://arxiv.org/abs/2309.14322）：对 Q 和 K 做 LayerNorm/RMSNorm。但对 MLA（Multi-head Latent Attention）不适用，因为 MLA 的 Key matrices 在 inference 时不 fully materialize

### 2.3 QK-Clip：核心机制

Intuition 很直接：与其在 forward 时 clip logits（改变当前 step 的计算），不如 **post-update rescale 权重**，让权重本身不会产生过大的 logits。这不会 alter 当前 step 的 forward/backward，只是用 max logit 作为 signal 来控制权重增长。

数学细节：

对于 attention head $h$，input $X$：

$$\mathbf{Q}^h = \mathbf{X}\mathbf{W}_q^h, \quad \mathbf{K}^h = \mathbf{X}\mathbf{W}_k^h, \quad \mathbf{V}^h = \mathbf{X}\mathbf{W}_\nu^h$$

- $X$：layer input representation，shape $[T, d_{model}]$，$T$ 是 sequence length
- $\mathbf{W}_q^h, \mathbf{W}_k^h, \mathbf{W}_\nu^h$：head $h$ 的 query/key/value projection weights
- 上标 $h$：head index

Attention output：

$$\mathbf{O}^h = \text{softmax}\left(\frac{1}{\sqrt{d}}\mathbf{Q}^h \mathbf{K}^{h^\top}\right)\mathbf{V}^h$$

- $d$：head dimension
- $\frac{1}{\sqrt{d}}$：standard scaling factor，防止 dot product 随 $d$ 增长而方差爆炸

**Max logit** 定义（per-head scalar）：

$$S_{max}^h = \frac{1}{\sqrt{d}}\max_{\mathbf{X} \in B}\max_{i,j}\mathbf{Q}_i^h \mathbf{K}_j^{h^\top}$$

- $B$：当前 batch
- $i, j$：batch 内不同 sample 的 token 索引
- 这个值在 forward 时已经计算了，所以 QK-Clip 几乎没有额外计算开销

**Clip 操作**（naive 版本，所有 heads 同时 clip）：

$$\mathbf{W}_q^h \leftarrow \gamma^\alpha \mathbf{W}_q^h$$
$$\mathbf{W}_k^h \leftarrow \gamma^{1-\alpha} \mathbf{W}_k^h$$

- $\gamma = \min(1, \tau / S_{max})$，$S_{max} = \max_h S_{max}^h$
- $\tau$：target threshold（K2 用 $\tau = 100$）
- $\alpha$：balancing parameter，通常 0.5，对 Q 和 K 对称缩放

为什么是 $\gamma^\alpha$ 和 $\gamma^{1-\alpha}$ 而不是直接 $\gamma$？因为 logit 是 $Q \cdot K = (XW_q) \cdot (XW_k)$，是 bilinear form，两边各缩放 $\sqrt{\gamma}$ 等价于整体缩放 $\gamma$。$\alpha = 0.5$ 时各缩放 $\gamma^{0.5} = \sqrt{\gamma}$。

**Per-head clip**（实际使用的版本）：

$$\gamma_h = \min(1, \tau / S_{max}^h)$$

只对 explode 的 head clip，最小化对其他 head 的干预。

**对 MLA 的特殊处理**：

MLA 把 key 分成 shared 和 head-specific 部分。K2 只 clip unshared components：
- $\mathbf{q}^C$ 和 $\mathbf{k}^C$（head-specific content components）：各 scaled by $\sqrt{\gamma_h}$
- $\mathbf{q}^R$（head-specific rotary）：scaled by $\gamma_h$（因为 rotary 是通过角度旋转，幅度直接缩放）
- $\mathbf{k}^R$（shared rotary）：**不动**，避免 cross-head 影响

这个设计很精妙：rotary embedding 的 shared 部分如果被 clip，会同时影响所有 heads，可能 over-regularize。

### 2.4 QK-Clip 的 self-deactivation 特性

一个非常重要的 empirical finding（Figure 2 Right）：在 K2 的整个训练中，max logits 在前 ~30% steps 被 capped at 100，然后 **逐渐 decay 到一个 stable range**，QK-Clip 自动 deactivate。

Appendix D 的数据：
- 前 70000 steps：12.7% 的 attention heads 触发过 QK-Clip
- 70000 steps 后：所有 heads 的 $S_{max}$ 都降到 100 以下，QK-Clip 完全 inactive

这说明 QK-Clip 是一个 **transient stabilizer**：在训练初期权重还远离 stable manifold 时拉一把，等模型自己找到好的 region 后就放手。

### 2.5 QK-Clip 不损害模型质量

Appendix D 的 ablation：0.5B activated / 3B total MoE，对比 vanilla Muon 和 MuonClip（$\tau = 30$，很激进的 threshold）。Figure 12 显示 loss curve 几乎重合，downstream task 也没有 statistically significant degradation。

Intuition：QK-Clip 只在 logits 超过阈值时才介入，而且只 rescale 权重的 magnitude 而不改变 direction（因为 $\gamma$ 是正标量）。权重的 spectral structure 被 preserve，只是 spectral norm 被 bounded。

---

## 3. 为什么 Muon 比 Adam 更容易 Logit Explosion

这是 Appendix E 的理论分析，我觉得是整篇 paper 最有洞察力的部分。

### 3.1 结构性差异

Muon 的 update 来自 msign 操作，所有奇异值相等 → effective rank 是满的。
Adam 的 update 谱是倾斜的，少数大奇异值主导 → effective rank 低。

这个 low-rank 假设对 Adam 不是新东西，muP（https://arxiv.org/abs/2203.03466）也用了类似假设。

Moonlight 16B 实验证实：Muon 训练的 weights 比 Adam 训练的 weights 有更高的 singular-value entropy（higher effective rank）。

### 3.2 SVD 分析

设 step $t-1$ 的参数矩阵 SVD：

$$\mathbf{W}_{t-1} = \sum_i \sigma_i u_i \nu_i^\top$$

- $\sigma_i$：singular values
- $u_i, \nu_i$：左右 singular vectors
- 求和 over rank

Update matrix：

$$\Delta\mathbf{W}_t = \sum_j \bar{\sigma}\bar{u}_j\bar{\nu}_j^\top$$

- $\bar{\sigma}$：Muon 下所有 singular values 相等

下一步参数：

$$\mathbf{W}_t \leftarrow \sum_i \sigma_i u_i \nu_i^\top + \sum_j \bar{\sigma}\bar{u}_j\bar{\nu}_j^\top$$

**Key insight**：因为 Muon 的 weights 和 updates 都有 higher effective rank，singular-vector pair $u_i\nu_i^\top$ 与 $\bar{u}_j\bar{\nu}_j^\top$ **对齐的概率更高**。一旦对齐，对应的 singular value 会 **加性增长** $\sigma_i \to \sigma_i + \bar{\sigma}$。

Adam 因为 update 是 low-rank，对齐的方向少，singular value 增长更分散。

### 3.3 Attention 的放大效应

Attention logit 是 bilinear form：

$$q_i \cdot k_j = (x_i\mathbf{W}_q) \cdot (x_j\mathbf{W}_k)$$

等价于涉及 $\mathbf{W}_q\mathbf{W}_k^\top$。这个乘积会 **平方 spectral norm**：

$$\|\mathbf{W}_q\mathbf{W}_k^\top\|_2 \leq \|\mathbf{W}_q\|_2 \cdot \|\mathbf{W}_k\|_2$$

所以 $\mathbf{W}_q$ 或 $\mathbf{W}_k$ 任何一个的 spectral norm 增长，都会在 logit 中被 compound。Muon 倾向于增大 singular values 的特性，直接 translate 成更高的 logit explosion 风险。

这给了我们一个很深的 intuition：**优化器的谱特性会通过 attention 的 bilinear structure 被放大**。选择 optimizer 时不能只看 loss 下降速度，还要考虑它对 weight matrix spectral structure 的长期影响。

---

## 4. 模型架构：Sparsity Scaling Law

### 4.1 架构总览（Table 2）

| 维度 | DeepSeek-V3 | Kimi K2 | 变化 |
|------|-------------|---------|------|
| Layers | 61 | 61 | = |
| Total Params | 671B | 1.04T | ↑54% |
| Activated Params | 37B | 32.6B | ↓13% |
| Experts (total) | 256 | 384 | ↑50% |
| Experts Active per Token | 8 | 8 | = |
| Shared Experts | 1 | 1 | = |
| Attention Heads | 128 | 64 | ↓50% |
| Dense Layers | 3 | 1 | ↓67% |
| Expert Grouping | Yes | No | — |

关键变化：
- **更 sparse**：384/8 = 48 vs 256/8 = 32
- **更少激活参数**：32.6B vs 37B，相同 FLOPs 下总参数更多
- **更少 attention heads**：64 vs 128，为了长上下文推理效率
- **更少 dense layers**：1 vs 3，几乎全部 MoE
- **去掉 expert grouping**：更灵活的路由

### 4.2 Sparsity Scaling Law（Figure 5）

这是 K2 架构选择的核心依据。实验设计：
- 固定 activated experts = 8
- 固定 shared experts = 1
- 变化 total experts 数 → 不同 sparsity levels
- 固定 activated parameters（即固定 FLOPs）

发现：**sparsity 越高，training loss 和 validation loss 都持续下降**。

具体数据点：达到 val loss = 1.5 时，sparsity 48 相比：
- sparsity 8：FLOPs 减少 1.69×
- sparsity 16：FLOPs 减少 1.39×
- sparsity 32：FLOPs 减少 1.15×

Intuition：在固定 FLOPs 下，增加 total parameters（通过增加 experts）给模型更多 "capacity slots" 可以存储不同的 knowledge/skills，router 可以更精准地选择 relevant experts。这比把所有 capacity 塞进少数 experts 更 efficient。

但 sparsity 增加带来 infrastructure complexity（更多 experts 意味着更多 all-to-all 通信、更多 memory碎片）。K2 选择 sparsity 48 作为平衡点。

这个 finding 和 Switch Transformer（https://arxiv.org/abs/2101.03961）、ST-MoE（https://arxiv.org/abs/2202.08906）早期的 sparsity 探索一致，但 K2 把它推到了更 extreme 的 48:1 ratio。

### 4.3 Attention Heads 数量：64 vs 128

DeepSeek-V3 用 ~2× layers 的 heads（128 for 61 layers）来 better utilize memory bandwidth。但 K2 选择 64 heads。

**推理开销分析**：sequence length 128k 时，heads 64→128，inference FLOPs 增加 **83%**（保持 expert count 固定）。这对 agentic 应用（需要长上下文）是 deal-breaker。

**训练收益实验**（Figure 6）：iso-token 训练下，heads 翻倍只带来 0.5%-1.2% 的 val loss 改善。ROI 太低。

Intuition：attention heads 的作用是让模型 attend to 不同 representation subspaces。但已经有 384 个 experts 提供 massive capacity，attention heads 的 marginal value 下降了。把省下的 compute 投到更多 experts 更划算。

---

## 5. Pre-training Data：Rephrasing 提升 Token Utility

### 5.1 核心问题

高质量人类数据有限。单 epoch 不够吸收，多 epoch 重复会 overfit。如何 squeeze 更多 learning signal 出每个 token？

### 5.2 Knowledge Data Rephrasing

三阶段 pipeline（Figure 4）：

1. **Style- and perspective-diverse prompting**：受 WRAP（https://arxiv.org/abs/2401.16380）启发，用多种 prompt 让 LLM 改写原文，保持 factual integrity 但变化 style 和 perspective
2. **Chunk-wise autoregressive generation**：长文档切分成 chunks，带 context 逐个改写，再拼接。避免 LLM 输出长度限制导致的信息丢失
3. **Fidelity verification**：语义对齐检查，过滤不忠实的改写

**实验结果**（Table 1，SimpleQA accuracy）：

| # Rephrasings | # Epochs | Accuracy |
|---------------|----------|----------|
| 0 (raw) | 10 | 23.76 |
| 1 | 10 | 27.39 |
| 10 | 1 | 28.94 |

关键 insight：**10 次改写 + 1 epoch > 1 次改写 + 10 epoch > 原文 10 epoch**。这说明改写带来的 diversity 比 naive repetition 更有效，每个 token 贡献的 learning signal 更高。

这和 "data quality > data quantity" 的共识一致，但 K2 把它 operationalize 成了一个 scalable pipeline。

### 5.3 Math Data Rephrasing

受 SwallowMath（https://arxiv.org/abs/2505.02881）启发，把数学文档改写成 "learning-note" style。加上跨语言翻译（其他语言 → English）增加 diversity。

### 5.4 Overall Corpus

15.5T tokens，四个 domain：Web Text, Code, Mathematics, Knowledge。沿用 K1.5（https://arxiv.org/abs/2501.12599）的 pipeline。

---

## 6. Training Infrastructure

### 6.1 集群配置

- NVIDIA H800 GPUs
- 每节点 2TB RAM, 8 GPUs, NVLink/NVSwitch
- 跨节点 8×400 Gbps RoCE

### 6.2 Parallelism 策略

灵活设计：可以训练在任何 32 倍数的节点上。

- **16-way Pipeline Parallelism (PP)** with virtual stages
- **16-way Expert Parallelism (EP)**
- **ZeRO-1 Data Parallelism**

Memory 预算（BF16 params + FP32 gradient buffer）：~6TB，分布在 256 GPUs 的 model-parallel group。每 GPU ~30GB 用于 states，其余用于 activations。

**为什么不用 DualPipe**（DeepSeek-V3 的设计）：DualPipe 会 double 参数和梯度的 memory，需要增加 parallelism 来 compensate。增加 PP 带来更多 bubble，增加 EP 带来更多 overhead。对 1T+ 模型成本太高。

**EP communication overlap**：增加 warm-up micro-batches，在 interleaved 1F1B schedule 下 overlap EP all-to-all。同时 decouple weight-gradient computation，与 PP communication 并行执行。

**Smaller EP size**：K2 的 64 heads（vs DeepSeek-V3 的 128）减少了 attention 计算时间，为了 full overlap，需要最小化 EP 操作时间。选择 EP=16。更小的 EP group 还 relax 了 expert-balance constraints。

### 6.3 Activation Reduction

三层策略：

1. **Selective recomputation**：LayerNorm, SwiGLU, MLA up-projections, MoE down-projections 重计算。这些都是 memory 高但 compute 低的操作
2. **FP8 storage**（不 FP8 computation）：MoE up-projections 和 SwiGLU 的 inputs 用 FP8-E4M3 (1×128 tiles with FP32 scales) 压缩。小规模实验显示无 loss 增加
3. **Activation CPU offload**：剩余 activations offload 到 CPU RAM，copy engine 流式传输，overlap with compute/communication

Figure 7 展示了 PP 各阶段的 compute/communication/offload overlap 模式。

### 6.4 Training Recipe

- Context window: 4096 tokens
- Optimizer: MuonClip
- LR schedule: WSD（https://arxiv.org/abs/2404.06395）
  - 500-step warmup
  - 前 10T tokens: constant LR 2e-4
  - 后 5.5T tokens: cosine decay 2e-4 → 2e-5
- Weight decay: 0.1
- Global batch size: 67M tokens
- **Annealing + long-context**:
  - 400B tokens @ 4k seq, LR 2e-5 → 7e-6
  - 60B tokens @ 32k seq
  - YaRN（https://arxiv.org/abs/2309.00071）扩展到 128k

Figure 3 的 loss curve 极其平滑，**零 spike**。这是 MuonClip 的直接证据。

---

## 7. Post-Training

### 7.1 SFT

用 Muon optimizer 做 fine-tuning（pre-train 用什么，fine-tune 用什么——这是 Moonlight 的结论）。

数据构建原则：maximize prompt diversity + ensure high response quality。用 K1.5 和 in-house expert models 生成候选 response，LLM/human judge 过滤。

### 7.2 Agentic Data Synthesis Pipeline

这是 K2 的另一个核心贡献。三阶段（Figure 8）：

**Stage 1: Tool spec generation**
- 3000+ real MCP tools 从 GitHub 抓取
- 20000+ synthetic tools 通过 hierarchical domain evolution 生成
- t-SNE 可视化（Figure 9）显示 real 和 synthetic tools 覆盖互补的 tool space 区域

**Stage 2: Agent and task generation**
- 合成各种 system prompts + 不同 tool 组合 → diverse agent population
- 每个 agent 配套 rubric-based tasks（从简单到复杂）
- Rubric 明确 success criteria, expected tool-use patterns, evaluation checkpoints

**Stage 3: Trajectory generation**
- User simulation: LLM-generated personas with distinct communication styles
- Tool execution environment: sophisticated simulator（功能上等价于 world model），维护 state，引入 controlled stochasticity（成功/部分失败/edge case）
- Quality evaluation: LLM judge 按 rubric 评估，只保留达标的 trajectory

**Hybrid approach**：对 authenticity 要求高的场景（coding, software engineering），用 real execution sandboxes（Kubernetes, 10000+ concurrent instances, unit test pass rate 作为 ground-truth）。

这个 pipeline 本质上是在做 **large-scale rejection sampling**——生成大量 trajectory，过滤出高质量的用于 SFT。

参考：
- ACEBench: https://arxiv.org/abs/2501.10951
- ToolLLM: https://arxiv.org/abs/2307.16789
- AgentInstruct: https://arxiv.org/abs/2407.03502
- MCP: https://modelcontextprotocol.io/

### 7.3 RL Framework

#### 7.3.1 Verifiable Rewards Gym

覆盖多个 domain：

**Math/STEM/Logical**：
- Diverse coverage: expert annotations + internal QA extraction + open datasets（NuminaMath: https://arxiv.org/abs/2404.14928, OpenMathReasoning: https://arxiv.org/abs/2504.16891）
- Moderate difficulty: 用 SFT model 的 pass@k 评估，只选中等难度（太易/太难都低 signal）

**Complex Instruction Following**：
- Hybrid rule verification: deterministic code interpreter + LLM-as-judge
- Hack-check layer: 检测 "声称满足但实际未满足" 的 adversarial behavior
- Multi-source generation: expert-crafted + AutoIF-style augmentation（https://arxiv.org/abs/2406.13542）+ fine-tuned model 生成 edge case

**Faithfulness**：
- 受 FACTS Grounding（https://arxiv.org/abs/2501.03200）启发
- 训练 sentence-level faithfulness judge model，检测无 evidence 的 factual claim

**Coding & Software Engineering**：
- 竞赛编程：open datasets + synthetic + human-written unit tests
- SWE: GitHub PRs/issues 构建 executable unit test 环境，Kubernetes sandbox

**Safety**：
- Seed prompts → automated evolution pipeline
- Attack model + Target model + Judge model 的 red-teaming 循环

#### 7.3.2 Self-Critique Rubric Reward

这是为了把 RL 扩展到 **non-verifiable tasks**（creative writing, open-ended QA）。

机制：
1. K2 actor 生成 responses
2. K2 critic 做 pairwise evaluation，按 rubrics 排序
3. Rubrics 包括：
   - **Core rubrics**（Appendix F.1）: Clarity/Relevance, Conversational Fluency, Objective/Grounded Interaction
   - **Prescriptive rubrics**（Appendix F.2）: 禁止 initial praise, 禁止 explicit justification
   - **Human-annotated rubrics**: 特定 instruction context

**Closed-loop refinement**：critic 用 verifiable-reward prompts 的 on-policy rollouts 持续更新，把 objective performance signal 蒸馏到 subjective judgment 中。这让 critic 的主观判断 grounded 在 verifiable data 上。

这个设计很 clever：**用 verifiable tasks 的 RL gains 来 calibrate critic，再让 critic 指导 non-verifiable tasks 的 RL**。相当于把 verifiable reward 的信号 "bridging" 到 subjective domain。

#### 7.3.3 RL Algorithm

基于 K1.5 的 policy optimization（variant of RLOO/REINFORCE with baseline）：

$$L_{RL}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\left[\frac{1}{K}\sum_{i=1}^K\left[\left(r(x,y_i) - \bar{r}(x) - \tau\log\frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)}\right)^2\right]\right]$$

变量解释：
- $x$：problem/prompt
- $\mathcal{D}$：prompt distribution
- $\{y_1, \ldots, y_K\}$：从 old policy $\pi_{old}$ 采样的 K 个 responses
- $\theta$：当前 policy 参数
- $r(x, y_i)$：reward function
- $\bar{r}(x) = \frac{1}{K}\sum_{i=1}^K r(x, y_i)$：mean reward（作为 baseline）
- $\tau > 0$：正则化参数，控制 KL penalty 强度
- $\log\frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)}$：log importance ratio

这个 objective 是 **squared deviation from baseline-adjusted advantage**，而不是 standard policy gradient。Intuition：最小化 $(advantage - KL\_penalty)^2$，让 policy 在 reward improvement 和 KL 约束之间找平衡。$\tau$ 越大，越保守（stay close to old policy）。

注意这里用 Muon optimizer 来 minimize 这个 objective，延续 pre-train 的选择。

#### 7.3.4 RL 的三个 additions

**Budget Control**：
- Per-sample maximum token budget，按 task type 设定
- 超出 budget 的 response 被 truncate 并 penalty
- 防止 RL 导致 response length 无限膨胀（DeepSeek-R1 的 known issue: https://arxiv.org/abs/2501.12948）
- 鼓励 concise yet effective solutions

**PTX Loss**：
- Auxiliary loss，用 hand-selected high-quality samples
- 防止 joint RL training 中 forgetting 高质量数据
- 类似 InstructGPT（https://arxiv.org/abs/2203.02155）的 PTX loss 设计

**Temperature Decay**：
- 训练初期高 temperature → exploration（diverse responses, discover strategies）
- 训练后期低 temperature → exploitation（stable, high-quality outputs）
- 从 exploration 到 exploitation 的 annealing

### 7.4 RL Infrastructure

#### 7.4.1 Colocated Architecture

Training 和 inference engine 在同一 workers 上，交替激活。Centralized controller 协调：
1. Inference engine 生成新数据
2. Training engine 训练
3. 更新参数传回 inference engine

#### 7.4.2 Efficient Engine Switching

挑战：1T 模型，training 和 inference 的 sharding paradigm 不同。用 network filesystem reshard 不现实（需要 PB/s 级 bandwidth）。

**Solution: Distributed checkpoint engine**（Figure 10）：
- 每个 training node 上 co-locate 一个 checkpoint engine worker
- Checkpoint engine worker 从 training engine 获取 local parameter copy
- Broadcast full parameter set across all checkpoint engine workers
- Inference engine 只取自己需要的 shard

**Pipeline 设计**（Appendix G, Figure 13）：
- 三个 buffer：H2D buffer + 两个 IPC buffer（shared 给 inference engine）
- 理论三阶段 pipeline：H2D → Broadcast → Reload
- 实际 H800 上 PCIe 饱和 → 退化为两阶段：synchronous H2D + parallel Broadcast/Reload
- 大规模设备上 model shard 小，一次 H2D 就能装下，overhead 消失

**性能**：full parameter update < 30 秒，对典型 RL iteration 可忽略。

代码开源：https://github.com/MoonshotAI/checkpoint-engine

#### 7.4.3 Agentic Rollout 优化

- 环境交互可能阻塞（VM, code interpreter）→ GPU idle
- 策略 1: heavy environments 部署为 dedicated services, 独立 scale
- 策略 2: 大量 concurrent rollouts amortize latency
- **Partial rollout**（K1.5 技术）: long-tail unfinished tasks 暂停，下一 iteration resume
- 统一 Gym-like interface（https://arxiv.org/abs/1606.01540）

---

## 8. Evaluations

### 8.1 Post-training Results（Table 3）

关键数字（non-thinking mode）：

**Agentic Coding**：
- SWE-bench Verified: 65.8%（single attempt agentic）, 71.6%（multi-attempt）
- SWE-bench Multilingual: 47.3%
- SWE-Lancer: 39.1%
- LiveCodeBench v6: 53.7%
- OJBench: 27.1%

对比 Claude Sonnet 4: SWE-bench Verified 72.7%, SWE-bench Multilingual 51.0%。K2 把 gap 压缩到 ~7-10 个点。

**Tool Use**：
- τ2-bench: 66.1（micro Pass@1, avg of retail/airline/telecom）
- ACEBench: 76.5

**Math/STEM**：
- AIME 2024: 69.6%, AIME 2025: 49.5%
- GPQA-Diamond: 75.1%
- MATH-500: 97.4%

**General**：
- MMLU: 89.5%, MMLU-Redux: 92.7%
- IFEval: 89.8%, Multi-Challenge: 54.1%
- SimpleQA: 31.0%

**LMSYS Arena**（July 17, 2025）: top-1 open-source, 5th overall（3000+ votes）。

### 8.2 Pre-training Base Results（Table 4）

K2-Base 在大多数 benchmark 上达到 open-source SOTA：
- MMLU: 87.79%
- LiveCodeBench v6: 26.29%
- MATH: 70.22%
- C-Eval: 92.50%

### 8.3 Safety（Table 6）

用 Promptfoo（https://www.promptfoo.dev/）做 red-teaming，覆盖 5 个 plugin 类别 × 5 个 strategy。

K2 在 Base64 strategy 上接近 100% passing rate（编码变换对模型 robustness 影响小）。Crescendo strategy 最有效（passing rate 普遍下降）。Iterative Jailbreak 对 Criminal 类别是最大挑战（K2: 57.57%）。

---

## 9. Tool Calling Token Template（Appendix B）

K2 用 TypeScript 表达 tool declaration（比 JSON 简洁，类型系统 comprehensive）：

```typescript
namespace functions {
  type get_weather = (_: {
    location: string,
    date?: string
  }) => any;
}
```

Tool call 格式：
```
<|tool_call_section_begin|>
<|tool_call_begin|>
functions.get_weather:0
<|tool_arguments_begin|>
{"location": "Beijing"}
<|tool_call_end|>
<|tool_call_section_end|>
```

支持 parallel tool calling（多个 `<|tool_call_begin|>` in one section）。call_id 格式 `functions.{tool-name}:{counter}`。

**Constrained decoding**：`<|tool_call_section_begin|>` 后用 enforcer 模块确保后续 tokens 符合 template 和 JSON schema。受 lm-format-enforcer 启发。

---

## 10. 我的思考和联想

### 10.1 MuonClip 的更深层意义

QK-Clip 本质上是一个 **spectral norm regularizer**，但它是 **adaptive** 和 **transient** 的。对比其他 spectral norm 接近方法：
- **Spectral norm regularization**（https://arxiv.org/abs/1705.10941）：每步都惩罚，过度保守
- **Power iteration**（https://arxiv.org/abs/1802.05957）：计算开销大
- QK-Clip：只在 logits 超阈值时介入，零额外 forward cost，自动 deactivate

这个 design pattern 可以推广：**用 output 的极端行为作为 signal 来 regularize weight 的 spectral properties**。比如可以想象类似的 "gradient norm clip"、"activation variance clip" 等。

### 10.2 和 MuP 的关系

Appendix E 提到 Adam 的 low-rank update 假设和 muP 一致。Muon 的 full-rank update 实际上打破了 muP 的假设，这可能解释了为什么 Muon 在 scale 时行为不同。

一个开放问题：**能否设计一个 hybrid optimizer，在 training 早期 full-rank（快速探索），后期 low-rank（精细收敛）**？QK-Clip 的 self-deactivation 特性暗示这种 staged approach 可能 work。

### 10.3 Sparsity 极限

K2 的 sparsity 48 已经很 extreme。继续增加 sparsity 的瓶颈：
- All-to-all 通信开销随 expert 数线性增长
- Router 的决策难度增加（384 选 8 比 256 选 8 更难）
- Expert specialization 的冷启动问题

可能的 next step：
- **Hierarchical routing**（先选 expert group，再选具体 expert）
- **Dynamic expert count**（不同 layer 不同 sparsity）
- **Expert merging/pruning**（训练中动态调整 expert 数）

### 10.4 Rephrasing 和 Synthetic Data 的哲学

K2 的 rephrasing 和 agentic data synthesis 都指向同一个哲学：**人类数据是 seed，模型自己 generate 的 synthetic data 是 amplifier**。

这和 "Welcome to the era of experience"（Silver & Sutton, https://arxiv.org/abs/2503.14422）的愿景一致。但 K2 的 approach 更 pragmatic：不是完全靠 self-play，而是用 human data 作为 grounding，用 synthetic data 作为 coverage 和 diversity 的 amplifier。

### 10.5 RL 的 Self-Critique 闭环

K2 的 self-critique rubric reward 是一个 **bootstrapping loop**：
1. Verifiable tasks 提供 objective signal
2. Critic 从 objective signal 学习如何 judge
3. Critic judge subjective tasks，提供 reward signal
4. Actor 从 subjective reward 学习
5. Actor 的 on-policy outputs 反过来更新 critic

这类似于 AlphaGo 的 policy-value network 共训，但用 natural language 而不是 game outcome 作为 signal。open question：这个 loop 会不会 reward hack？K2 用 prescriptive rubrics（禁止 initial praise 等）来 mitigate，但长期 stability 仍需观察。

### 10.6 和 DeepSeek 路线的对比

| 维度 | DeepSeek-V3/R1 | Kimi K2 |
|------|----------------|---------|
| Optimizer | AdamW | MuonClip |
| MoE sparsity | 32 (256/8) | 48 (384/8) |
| Attention | MLA, 128 heads | MLA, 64 heads |
| Thinking mode | R1 有 extended thinking | Explicitly non-thinking |
| RL approach | GRPO + verifiable rewards | RLOO-style + verifiable + self-critique |
| Data scaling | 14.8T tokens | 15.5T tokens + rephrasing |

K2 明确选择 **non-thinking** 路线，把 reasoning 能力 bake 进 base model 而不是靠 test-time compute。这和 Claude 3.5 Sonnet 的哲学一致。trade-off：inference 更便宜，但上限可能不如 thinking models。

### 10.7 Limitations 和未来方向

Paper 承认的 limitations：
- Hard reasoning 时可能 generate excessive tokens → truncation
- Tool definition 不清晰时输出 incomplete tool calls
- 不必要时启用 tool use 会降 performance
- One-shot prompting 不如 agentic framework

这些 limitations 揭示了一个 tension：**agentic capability 和 controlled output 之间的 trade-off**。模型学会了 use tools，但也学会了 over-use tools。Budget control 是一个 partial mitigation，但根本解决可能需要更好的 tool-use discretion。

---

## 总结

Kimi K2 的核心贡献是 **MuonClip**——一个让 Muon optimizer 在 1T scale 稳定训练的 clever hack。背后的理论洞察（Muon 的 full-rank update 导致 singular value 加性增长，被 attention 的 bilinear form 放大）对整个 LLM training 社区都有价值。

架构上，K2 把 MoE sparsity 推到 48:1，配合 64 heads 的精简 attention，在 agentic 长上下文场景下有很好的 inference efficiency。

Post-training 的 agentic data synthesis 和 self-critique RL 是实用的工程贡献，把 verifiable reward 的信号 bridge 到 subjective domain。

整体来看，K2 代表了 open-source LLM 在 agentic intelligence 方向的一个重要 milestone，close the gap with Claude 4。但 non-thinking 路线的天花板、tool over-use 的控制、以及 self-critique loop 的长期稳定性，都是未来需要解决的问题。

**主要参考链接**：
- Kimi K2 paper: https://arxiv.org/abs/2507.18841
- Muon: https://kellerjordan.github.io/posts/muon/
- Moonlight: https://arxiv.org/abs/2502.16982
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- Kimi K1.5: https://arxiv.org/abs/2501.12599
- Checkpoint engine code: https://github.com/MoonshotAI/checkpoint-engine
- Model weights: https://huggingface.co/moonshotai
