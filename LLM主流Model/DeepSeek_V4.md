---
source_pdf: DeepSeek_V4.pdf
paper_sha256: f4cbe4fcbd2888b25b2890a98cc6ef4ce0489df7c93e140b6f853c451d3f5c52
processed_at: '2026-08-03T18:47:19-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DeepSeek-V4 人话版

好, Andrej, 我把那些公式和工程细节都扒掉, 用大白话重新给你讲一遍 DeepSeek-V4 到底在干嘛, 为什么这么干。

---

## 这论文到底在解决什么问题?

一句话: **reasoning model (像 o1, R1) 要发挥威力, 得 generate 几千甚至几万 token 的 thinking chain。但如果 context window 本身就很大 (比如 1M tokens), 再叠加长 thinking, attention 的计算量直接爆炸**。

你想想, vanilla attention 是 $O(n^2)$ 的。1M tokens 的 prefill, 加上几千 token 的 generation, 每一步都要 attend 到前面所有 KV cache。KV cache 本身就几十 GB, 每次还要做巨量 dot product。这就卡死了 test-time scaling 的天花板。

DeepSeek-V4 的目标就是: **让 1M token context 在工程上变得可行, 而且推理 FLOPs 和 KV cache 都大幅下降**。

结果就是开头那张表: 1M context 下, V4-Pro 的单 token FLOPs 只有 V3.2 的 27%, KV cache 只有 10%。V4-Flash 更夸张, FLOPs 10%, KV cache 7%。

而且 V4-Pro 的 activated params (49B) 比 V3.2 (37B) 还多。参数变多了, 计算反而变少了——这全靠 architecture 层面的根本性改造。

参考: [DeepSeek-V3.2 Report](https://arxiv.org/abs/2512.02556)

---

## Architecture 的三个核心创新

### 1. Hybrid Attention: CSA + HCA 交错

这是整篇 paper 的灵魂。V4 设计了两种高效 attention, 在不同的 layer 之间交错使用。

#### CSA (Compressed Sparse Attention)

CSA 干两件事:

**第一步, 压缩**。把每 $m$ 个 KV entry 压成 1 个。具体怎么压? 每个 token 会产生一个 "compression weight" (通过一个小 projection 学出来的), 然后对 $m$ 个 token 的 KV 做 weighted sum, 得到一个 compressed KV entry。这就把 sequence length 缩小了 $m$ 倍。

这里有个巧妙的 overlap 设计: 压缩第 $i$ 个 block 时, 不光用当前 $m$ 个 token, 还用前一个 block 的 $m$ 个 token。相邻 compressed entries 共享部分 input, 避免硬切边界丢信息。

**第二步, sparse selection**。压缩完之后, 不能直接做 dense attention (因为压缩后还是可能很长)。所以搞了个 "Lightning Indexer"——一个轻量的 scoring network, 对每个 query token 算它跟每个 compressed block 的 relevance score, 然后只选 top-k 个最相关的 block 做 attention。

Pro 选 1024 个 compressed block, 每个对应 4 个原始 token, 相当于 attend 到 4096 个原始 token 的信息 (通过压缩表示)。Flash 选 512 个。

**Core attention 用 MQA** (Multi-Query Attention) 的极端形式: 所有 query head 共享同一套 KV。compressed KV entry 同时当 key 和 value。这让 KV cache 极小, 表达能力完全依赖 compression 质量。

最后还有个 **grouped output projection** 来省计算量: 因为 query head 很多 (Pro 128 个), head dim 也大 (512), 直接投影回 hidden dim 计算量爆炸。所以把 output 分组, 每组先降维再合并投影。本质是 low-rank decomposition。

**Intuition**: CSA 是 "适度压缩 + 精细选择"。适合需要 fine-grained token-level attention 的场景。

参考: [DeepSeek Sparse Attention](https://arxiv.org/abs/2502.11089), [MQA Paper](https://arxiv.org/abs/1911.02150)

#### HCA (Heavily Compressed Attention)

HCA 的思路完全不同: **极端压缩, 但保持 dense attention**。

压缩率 $m' = 128$ (CSA 是 4)。也就是说, 128 个 token 的 KV 压成 1 个。1M context 压完只剩 7812 个 KV entry, dense attention 完全可行。

没有 overlapped compression, 没有 sparse selection。就是粗暴地把 sequence length 缩小 128 倍, 然后做正常 attention。

**Intuition**: HCA 是 "极端压缩 + 全局 attention"。适合需要 broad context awareness 的场景——不需要精细到哪个 token, 但需要知道 "远处大概有什么信息"。

#### 为什么要 hybrid?

CSA 和 HCA 代表了两种不同的 efficiency-expressiveness trade-off。交错使用它们, 让模型同时获得:
- CSA 的 fine-grained selection (适合 "我要找到具体那个 token")
- HCA 的 coarse-grained overview (适合 "我要知道整体 context 在说什么")

这比单一 attention 架构灵活得多。

#### 还有一堆 trick

- **Sliding Window Attention 分支**: 每个 query 除了 attend compressed KV, 还额外保留最近 128 个 token 的 uncompressed KV。解决两个问题: 严格因果性导致同一 compression block 内 token 互相看不到; 以及 recent tokens 通常最相关。
- **Partial RoPE**: 只对最后 64 维加 RoPE。因为 shared KV 同时当 key 和 value, attention output 会带 absolute position。Countermeasure 是对 output 也加一个 position $-t$ 的 RoPE, 转成 relative position。这个 trick 解决了 MQA + RoPE 的兼容性问题。
- **Attention Sink**: 加一个 learnable 的 "sink logit" 到 softmax 分母, 让 attention scores 总和可以小于 1。相当于允许模型 "不 attend 到任何 KV"。来自 [StreamingLLM](https://arxiv.org/abs/2309.17453) 的发现。

---

### 2. mHC (Manifold-Constrained Hyper-Connections)

这个是对残差连接的升级。

#### Standard Hyper-Connections (HC)

标准残差连接是 $x_{l+1} = x_l + F_l(x_l)$。HC 把它扩展成:

$$X_{l+1} = B_l X_l + C_l F_l(A_l X_l)$$

这里 $X_l$ 不再是一个 vector, 而是一个 $n_{hc} \times d$ 的矩阵 (expansion factor $n_{hc} = 4$)。$A_l, B_l, C_l$ 是三个 learnable 的 mapping。

Intuition: HC 提供了一个额外的 scaling axis。$n_{hc}$ 远小于 $d$, 计算开销小, 但表达能力更强。你可以理解为 "残差通道" 从 1 条变成 4 条, 信号可以在 4 条通道之间混合。

**问题**: 标准 HC 在 stacking 多层后频繁 numerical instability。

#### mHC 的核心 idea

把 $B_l$ (residual transformation matrix) 约束到 **Birkhoff polytope**——所有 doubly stochastic matrices 的集合 (每行每列和都是 1, 且元素非负)。

为什么这能解决 stability?
1. **谱范数 ≤ 1**: doubly stochastic matrix 的 spectral norm bounded by 1, 所以 residual transformation 是 "non-expansive" mapping。信号不会 explosion。
2. **乘法封闭**: 两个 doubly stochastic matrices 相乘还是 doubly stochastic, 所以 deep stack 也稳定。
3. $A_l$ 和 $C_l$ 用 Sigmoid 约束为 non-negative 且 bounded, 避免 signal cancellation。

**怎么投影到 Birkhoff polytope?** 用 Sinkhorn-Knopp 算法: 先对 $B_l$ 取 $\exp$ 保证正数, 然后反复交替做行归一化和列归一化, 收敛到 doubly stochastic matrix。V4 用 20 次迭代。

参数是 dynamic (input-dependent) + static (input-independent) 的组合。Dynamic 部分由 input 经过 RMSNorm + linear projection 生成, static 部分是 learnable bias。

**Intuition**: 标准 HC 的 $B_l$ 是自由矩阵, 可以任意放大/缩小信号。多层 stacking 后, 谱范数指数级增长 (forward) 或衰减 (backward), 导致 instability。约束到 Birkhoff polytope 后, 谱范数 ≤ 1, 信号传播稳定。代价是表达能力略降, 但换来的是可扩展性。

参考: [mHC Paper](https://arxiv.org/abs/2512.24880), [Sinkhorn-Knopp](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem)

---

### 3. Muon Optimizer

Muon 是一个比较新的 optimizer, [Jordan et al., 2024](https://github.com/KellerJordan/Muon) 提出的, [Liu et al., 2025](https://arxiv.org/abs/2502.16982) 证明它 scalable for LLM training。

#### 核心 idea

传统 optimizer (Adam, AdamW) 是 element-wise 的: 每个参数有自己的 learning rate scaling。Muon 不一样, 它是 **matrix-level** 的: 对整个 gradient matrix 做近似 orthogonalization, 然后更新。

为什么 orthogonalize? 想象 gradient matrix 的 SVD: $G = U \Sigma V^T$。传统 optimizer 会被大的 singular value 主导 (outlier 方向)。Orthogonalize 后变成 $UV^T$ (正交部分), 所有方向被 "平等" 对待, spectral norm = 1。这比 Adam 的 per-element scaling 更适合 matrix parameters。

#### Algorithm

```
for each weight W:
    G = gradient(W)
    M = momentum * M_prev + G          # 累积 momentum
    O = Newton-Schulz(M)                # 近似正交化
    O = O * sqrt(max(n,m)) * gamma      # Rescale RMS, 复用 AdamW 学习率
    W = W * (1 - lr * wd) - lr * O      # Weight decay + 更新
```

Newton-Schulz 迭代是 polynomial approximation of sign function, 用来近似正交化。V4 用 hybrid 策略: 前 8 步用快速收敛系数, 后 2 步用精确稳定系数。

#### V4 的具体配置

- **AdamW**: embedding, prediction head, RMSNorm, mHC 的 static bias 和 gating factor
- **Muon**: 其他所有 modules
- Momentum: 0.95, Weight decay: 0.1, RMS rescale: 0.18

V4 的 attention 允许直接对 query/KV 做 RMSNorm, 防止 logit explosion, 所以不需要 QK-Clip。

---

## Infrastructure: 工程才是真正的难点

### MoE 的通信-计算 overlap

MoE 的 Expert Parallelism 需要复杂的跨节点通信。V4 的核心 insight: **通信延迟可以被计算完全隐藏, 只要 computation-communication ratio 足够高**。

对 V4-Pro, 每个 token-expert pair: 6h FLOPs 计算, 3h bytes 通信。所以每 GBps 带宽可以隐藏 6.1 TFLOP/s 计算。带宽只要过这个阈值, 就不是 bottleneck。

实现方式: **Wave-based scheduling**。把 experts 分成小波次, 每波完成通信立即开始计算, 同时下一波传输和已完成 experts 的结果发送并发进行。

性能: 1.5-1.7× speedup (general), 1.96× (RL rollout)。开源为 [MegaMoE](https://github.com/deepseek-ai/DeepGEMM)。

还提了几个给硬件厂商的建议: computation-communication ratio 比 bandwidth 更重要; power budget 要够 (kernel fusion 让 compute/memory/network 同时高负载); pull-based 比 push 好 (低 notification latency); 用 element-wise activation 替代 SwiGLU 省带宽。

参考: [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [Comet Paper](https://arxiv.org/abs/2502.19811)

### TileLang: Kernel 开发的 DSL

[TileLang](https://arxiv.org/abs/2502.19811) 平衡开发效率和运行时性能。亮点:

1. **Host Codegen**: 把 host-side validation 从 Python 移到 generated code, CPU overhead 从几十微秒降到 <1 微秒。
2. **Z3 SMT Solver**: 对 tensor index 算术做 formal analysis, 支持 vectorization, barrier insertion 等优化。
3. **Bitwise Reproducibility**: 默认关 fast-math, 提供 IEEE-compliant intrinsics, 支持 bit-identical outputs。

### Batch-Invariant 和 Deterministic Kernels

**Batch Invariance**: 任何 token 的 output 与它在 batch 中的位置无关 (bitwise identical)。这对调试和 post-training 一致性很重要。

主要挑战: split-KV / split-k 这些优化破坏 batch invariance (因为浮点加法不满足结合律, 不同 SM 的累加顺序不同)。V4 的解法:
- Attention: dual-kernel (单 SM 整 sequence + 多 SM 单 sequence), 精心对齐 accumulation order
- Matrix multiplication: 用 DeepGEMM 替代 cuBLAS, 避免 split-k
- MoE backward: token order pre-processing + buffer isolation

**Determinism**: 避免 atomicAdd 的 non-determinism, 用独立 accumulation buffer + global deterministic sum。

### FP4 Quantization-Aware Training

FP4 (MXFP4) 量化用于 MoE expert weights 和 CSA indexer 的 QK path。

亮点: **lossless FP4-to-FP8 dequantization**。FP8 (E4M3) 比 FP4 (E2M1) 多 2 个 exponent bits, dynamic range 更大。只要 FP4 sub-block 的 scale ratio 不超过阈值, FP4 的 fine-grained scale 可以被 FP8 的 dynamic range 完全吸收。这样整个 QAT pipeline 可以直接复用 FP8 训练框架, 无需修改。

Backward pass 用 STE (Straight-Through Estimator): 直接对 FP8 weights 算梯度, 传播回 FP32 master weights。

---

## Pre-Training 的关键点

### 32T tokens

数据包含 math, code, web, long documents, multilingual。特别强调 long-document curation (科学论文、技术报告)。

**Sample-level attention masking**: 不同 V3, V4 在 packing documents 时用 sample-level masking, 避免 cross-document attention。比 V3 的 padding 更高效。

### 训练不稳定性的两个实用 trick

#### Anticipatory Routing

观察: MoE outliers 与 routing mechanism 形成恶性循环。Routing 决策导致某些 expert 被过度激活, 这些 expert 的 activation 变成 outlier, 反过来又影响 routing。

解法: **Step $t$ 用当前参数 $\theta_t$ 计算 features, 但用历史参数 $\theta_{t-\Delta t}$ 计算 routing indices**。

实现: 提前一步 fetch 数据, pre-compute routing indices 并 cache。Pipeline 和 EP 通信 overlap, 额外 wall-time <20%。

**Auto-trigger**: Loss spike 时自动 rollback + 启用, 稳定后恢复。Dynamic 应用, 总开销可忽略。

**Intuition**: 这是一种 "lagged routing"。Routing 决策滞后于 backbone 更新, 打破了同步更新的 feedback loop。某种意义上类似 implicit regularization。

#### SwiGLU Clamping

- Linear component: clamp 到 $[-10, 10]$
- Gate component: 上限 10

**Intuition**: SwiGLU 的 unbounded exponential 在 outlier token 上产生极端激活值, 通过 residual 传播放大, 形成 loss spike。Clamping 是 explicit saturation, 牺牲少量 expressivity 换稳定性。来自 [Gemma 2](https://arxiv.org/abs/2408.00118) 和 [gpt-oss](https://arxiv.org/abs/2508.10925) 的经验。

### Base Model 评估

V4-Flash-Base (13B activated, 284B total) 在大部分 benchmark 上超越 V3.2-Base (37B activated, 671B total)。更少参数, 更好性能——architecture + data + training optimization 的综合效果。

V4-Pro-Base 进一步确立新 SOTA。特别值得注意的是 knowledge benchmarks 的巨大提升:
- FACTS Parametric: V3.2=27.1 → Pro=62.6
- SimpleQA-Verified: V3.2=28.3 → Pro=55.2

**Intuition**: knowledge 存在 parameters 里, 更多 params = 更多 knowledge。但 reasoning task 差距小 (HMMT: Flash=94.8, Pro=95.2), 因为 reasoning 更多依赖 architecture 和 training strategy。

---

## Post-Training: Specialist Training + On-Policy Distillation

### Pipeline

V4 用 **On-Policy Distillation (OPD)** 替代了 V3.2 的 mixed RL:

1. **Specialist Training**: 每个 domain (math, code, agent, instruction following) 独立训练 expert
   - SFT on domain-specific data
   - RL with GRPO + domain-specific reward
   
2. **On-Policy Distillation**: 10+ 个 teacher experts → 1 个 student model

### OPD 的核心

公式:
$$\mathcal{L}_{\mathrm{OPD}}(\theta) = \sum_{i=1}^N w_i \cdot D_{\mathrm{KL}}(\pi_\theta \| \pi_{E_i})$$

Student $\pi_\theta$ 在自己生成的 trajectory 上, 对每个 teacher $\pi_{E_i}$ 做 reverse KL。$w_i$ 是 teacher 的权重。

关键: **Full-vocabulary logit distillation**, 不是 token-level KL estimate。这避免了 gradient estimate 的高方差, 但计算量大 (vocabulary 100K+)。工程上用 cached hidden states + on-the-fly logit reconstruction 解决。

### Reasoning Effort 三模式

| Mode | 特征 | Context Window |
|------|------|----------------|
| Non-think | 快速直觉响应 | 8K |
| Think High | 逻辑分析 | 128K |
| Think Max | 推理极致 | 384K |

用特殊 token `illé` 和 `δML` 分隔。Think Max 还额外 prepend 一个 system prompt 引导深度推理。

### 其他 post-training 创新

- **Generative Reward Model (GRM)**: 不训练 scalar reward model, 直接让 actor 当 judge。RL 同时优化 generative 和 evaluative 能力。
- **Tool-call schema**: 用 XML 格式 + `δML` special token, 减少 escaping failure。
- **Interleaved Thinking**: Tool-call 场景下保留所有 reasoning trace 跨 user turn; 普通对话场景下每个 user turn 清空 reasoning。
- **Quick Instruction**: 用 special tokens 复用 KV cache 执行辅助任务 (web search 判断, query 生成, domain 识别), 避免额外 small model 的 redundant prefilling。

参考: [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation), [GRPO (DeepSeek-R1)](https://arxiv.org/abs/2501.12948)

---

## 评估结果亮点

### Standard Benchmark

**Knowledge**: V4-Pro-Max 在 open-source 中 SOTA。SimpleQA-Verified 比 open-source baseline 高 20 个百分点。但仍落后 Gemini-3.1-Pro。

**Reasoning**: V4-Pro-Max 超越所有 open model, 接近 frontier closed model。Codeforces rating 3206, 排名人类选手第 23。首次有 open model 匹配 closed model 的 code competition 水平。

**Formal Math**: 在 Lean 4 上, agentic setup 达到 SOTA, 超越 [Seed Prover](https://arxiv.org/abs/2512.17260)。Compute-intensive pipeline 下超越 [Aristotle](https://arxiv.org/abs/2510.01346), Putnam 2025 达到 120/120。

**1M Context**: MRCR 上超越 Gemini-3.1-Pro, 仍落后 Claude Opus 4.6。128K 内性能高度稳定, 1M 处仍有较强检索能力。

### Real-World Tasks

**Chinese Writing**: vs Gemini-3.1-Pro, functional writing 62.7% win rate, creative writing 60% (instruction following) / 77.5% (writing quality)。但最难场景 vs Claude Opus 4.5 仍落后 (45.9% vs 52.0%)。

**Search**: Agentic search vs RAG, 61.7% win rate, 成本只略高。

**White-Collar Task**: 30 个跨 13 个行业的高级中文专业任务, vs Opus-4.6-Max 非 loss rate 63%, 在 Task Completion 和 Content Quality 上有优势。

**Code Agent**: 内部 R&D coding benchmark, V4-Pro-Max 67% pass rate, 超越 Sonnet 4.5 (47%), 接近 Opus 4.5 (70%)。52% 的 DeepSeek 开发者愿意把 V4-Pro 当默认 coding model。

---

## 我的整体 intuition

这篇 paper 的核心 narrative 是: **要让 million-token context 真正可用, 不能只靠 hardware scaling, 必须在 architecture 层面做根本性改造**。

三个 architectural innovation 各自解决一个核心问题:
- **Hybrid CSA + HCA**: 解决 attention 的 $O(n^2)$ bottleneck, 让 1M context 在 FLOPs 和 KV cache 上都可控
- **mHC**: 解决深层 Transformer 的 signal propagation 不稳定, 让深层 + 宽层 training 可扩展
- **Muon**: 解决 optimizer 的 convergence speed 和 stability, 让 trillion-param MoE training 更快更稳

Infrastructure 部分则是把 architecture 的理论 efficiency 优势真正落地: EP overlap, batch invariance, FP4 QAT, TileLang kernel development, heterogeneous KV cache management, on-disk KV cache——每一块都是工程上的硬仗。

最后, post-training 用 OPD 替代 mixed RL, 用 full-vocabulary KL 替代 token-level estimate, 这也是当前 RL 路线遇到瓶颈后的一个重要 pivot。

整体感觉, V4 是一个非常 "工程驱动" 的 release: 很多设计选择 (Anticipatory Routing, SwiGLU Clamping, dual-kernel for batch invariance, lossless FP4-to-FP8) 都是 empirical 发现, 理论解释还不够。但它们 work, 而且在 1.6T param + 1M context 的 scale 上 work, 这本身就是巨大的工程成就。

参考链接汇总:
- [DeepSeek-V4 Collection](https://huggingface.co/collections/deepseek-ai/deepseek-v4)
- [mHC Paper](https://arxiv.org/abs/2512.24880)
- [Muon Optimizer](https://github.com/KellerJordan/Muon)
- [Muon Scalable for LLM](https://arxiv.org/abs/2502.16982)
- [TileLang](https://arxiv.org/abs/2502.19811)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-V3.2 Report](https://arxiv.org/abs/2512.02556)
- [DeepSeek Sparse Attention](https://arxiv.org/abs/2502.11089)
- [StreamingLLM](https://arxiv.org/abs/2309.17453)
- [MXFP4 Format](https://arxiv.org/abs/2310.10580)
- [DeepSeek-R1 (GRPO)](https://arxiv.org/abs/2501.12948)
- [Gemma 2 (Clamping)](https://arxiv.org/abs/2408.00118)
- [gpt-oss Model Card](https://arxiv.org/abs/2508.10925)
- [Seed Prover](https://arxiv.org/abs/2512.17260)
- [Aristotle](https://arxiv.org/abs/2510.01346)

---

希望这个版本更对你的胃口, Andrej。核心是 build intuition, 公式和工程细节都在之前那个版本里, 这个版本是 "为什么这么做" 的角度。如果你想深挖某一块 (比如 CSA 的 overlapped compression 为什么这么设计, 或者 Anticipatory Routing 的 theoretical mechanism), 随时说。

---

# DeepSeek-V4: 深度技术解析

嘿 Andrej, 这篇论文我仔细读了。我来给你做一个尽量深入的技术讲解, 重点关注那些能真正 build intuition 的设计决策和背后的数学/工程权衡。

---

## 1. 论文的核心定位: Million-Token Context 的 Efficiency Barrier

DeepSeek-V4 系列的核心命题在于: **reasoning models (如 o1, R1) 的 test-time scaling 范式被 vanilla attention 的 quadratic complexity 锁死了**. 当 context 达到 1M tokens, attention 的 FLOPs 和 KV cache 成了 dominant bottleneck, 这使得 long-horizon agentic workflows 和大规模 cross-document analysis 几乎不可行。

V4 的两个变体:
- **DeepSeek-V4-Pro**: 1.6T total params, 49B activated
- **DeepSeek-V4-Flash**: 284B total params, 13B activated

关键效率数据 (1M context, 对比 V3.2 baseline):
| 指标 | V4-Pro | V4-Flash | V3.2 |
|------|--------|----------|------|
| Single-token FLOPs | 27% | 10% | 100% |
| KV cache size | 10% | 7% | 100% |

这是非常激进的效率提升——Pro 比 V3.2 激活参数更多 (49B vs 37B), 但 1M context 下 FLOPs 只有 27%。这种"参数变多但 FLOPs 变少"的反转, 全靠 architecture 层面的根本性改造。

参考: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), [DeepSeek-V3.2 Report](https://arxiv.org/abs/2512.02556)

---

## 2. Architecture: 三个核心创新

### 2.1 Hybrid Attention: CSA + HCA 交错配置

这是整个 architecture 的灵魂。V4 设计了两种不同的高效 attention, 并在 layer 间交错使用。

#### 2.1.1 CSA (Compressed Sparse Attention)

CSA 的核心 idea 是**两阶段降复杂度**:
1. **Compression**: 把每 m 个 KV entries 压缩成 1 个
2. **Sparse Selection**: 在压缩后的 KV 上做 top-k sparse attention

让我解析 compression 公式 (11, 12):

$$[S_{mi:m(i+1)-1}^a; S_{m(i-1):mi-1}^b] = \mathrm{Softmax_{row}}([Z_{mi:m(i+1)-1}^a + B^a; Z_{m(i-1):mi-1}^b + B^b])$$

$$C_i^{\mathrm{Comp}} = \sum_{j=mi}^{m(i+1)-1} S_j^a \odot C_j^a + \sum_{j=m(i-1)}^{mi-1} S_j^b \odot C_j^b$$

变量解释:
- $m$: compression rate (Flash=4, Pro=4)
- $C^a, C^b \in \mathbb{R}^{n \times c}$: 两套 KV entries, $c$ 是 head dimension (=512)
- $Z^a, Z^b$: 对应的 compression weights, 由 hidden state 经 $W^{aZ}, W^{bZ}$ 投影得到
- $B^a, B^b \in \mathbb{R}^{m \times c}$: 可学习的 positional biases
- $S_j^a, S_j^b$: softmax 后的归一化权重
- $\odot$: Hadamard product (逐元素乘)

Intuition: 这里有个巧妙的 **overlapped compression** 设计。每个 $C_i^{\mathrm{Comp}}$ 来自 $2m$ 个原始 KV entries (当前 block 的 $m$ 个 + 前一个 block 的 $m$ 个), 但相邻 compressed entries 共享前/后 block。这相当于用 sliding window 的方式做 compression, 避免了硬切边界导致的信息丢失。

当 $i=0$ 时, $Z_{m(i-1):mi-1}^b$ 用 $-\infty$ padding, $C_{m(i-1):mi-1}^b$ 用 0 padding——这是处理 sequence 开头的因果性边界条件。

**Lightning Indexer** 的 sparse selection 公式 (13-17):

$$\mathbf{c}_t^Q = \mathbf{h}_t \cdot W^{DQ}$$
$$[P_{t,1}^I; P_{t,2}^I; ...; P_{t,n_h^I}^I] = P_t^I = \mathbf{c}_t^Q \cdot W^{IUQ}$$

$$I_{t,s} = \sum_{h=1}^{n_h^I} w_{t,h}^I \cdot \mathrm{ReLU}(\mathbf{q}_{t,h}^I \cdot K_s^{\mathrm{IComp}})$$

变量解释:
- $\mathbf{h}_t \in \mathbb{R}^d$: query token $t$ 的 hidden state
- $\mathbf{c}_t^Q \in \mathbb{R}^{d_c}$: 压缩后的 query latent vector, $d_c$ 是 query compression dimension (Flash=1024, Pro=1536)
- $n_h^I$: indexer query head 数量 (=64)
- $W^{DQ} \in \mathbb{R}^{d \times d_c}$, $W^{IUQ} \in \mathbb{R}^{d_c \times c^I n_h^I}$: 下投影和上投影矩阵
- $c^I$: indexer head dimension (=128)
- $K_s^{\mathrm{IComp}} \in \mathbb{R}^{n/m \times c^I}$: 压缩后的 indexer keys
- $w_{t,h}^I \in \mathbb{R}$: 第 $h$ 个 indexer head 的 weight, 由 $\mathbf{h}_t \cdot W^w$ 得到

注意 $\mathrm{ReLU}(\cdot)$ 的用法——这是把 dot product score 变成 non-negative, 然后用 learnable head weights $w_{t,h}^I$ 做 linear combination。最终 index score $I_{t,s}$ 越大, 表示 query $t$ 应该 attend 到 compressed block $s$。

Top-k selector (公式 17):
$$C_t^{\mathrm{SprsComp}} = \{C_s^{\mathrm{Comp}} : I_{t,s} \in \mathrm{Top\text{-}k}(I_{t,:})\}$$

Flash 选 512 个, Pro 选 1024 个。注意这里选的是 **compressed** blocks, 每个对应 $m$ 个原始 tokens。所以 Pro 实际上 attend 到 $1024 \times 4 = 4096$ 个原始 tokens 的信息 (通过压缩表示)。

**Shared Key-Value MQA** (公式 18-19):
$$[\mathbf{q}_{t,1}; \mathbf{q}_{t,2}; ...; \mathbf{q}_{t,n_h}] = \mathbf{q}_t = \mathbf{c}_t^Q \cdot W^{UQ}$$
$$\mathbf{o}_{t,i} = \mathrm{CoreAttn}(\text{query}=\mathbf{q}_{t,i}, \text{key}=C_t^{\mathrm{SprsComp}}, \text{value}=C_t^{\mathrm{SprsComp}})$$

这里非常激进: **compressed KV entry 同时充当 key 和 value**。这是 Multi-Query Attention (MQA) 的极端形式, 一个 shared KV head 服务所有 $n_h$ 个 query heads。$n_h$ 在 Flash=64, Pro=128。这种设计的 KV cache 极小, 但 expressive power 依赖于 compression 的质量。

**Grouped Output Projection**:
由于 $c \cdot n_h$ 很大 (Pro: $512 \times 128 = 65536$), 直接投影回 $d$ 维 hidden state 计算量巨大。解决方案: 把 $n_h$ 个 output 分成 $g$ 组 (Pro=16), 每组先投影到 $d_g$ 维 intermediate (Pro=1024), 再合并投影到 $d$ 维。这本质上是 low-rank decomposition。

参考: [DeepSeek Sparse Attention](https://arxiv.org/abs/2502.11089), [MQA Paper](https://arxiv.org/abs/1911.02150)

#### 2.1.2 HCA (Heavily Compressed Attention)

HCA 的 idea: **更激进的 compression, 但保持 dense attention**。

公式 (22, 23):
$$S_{m'i:m'(i+1)-1} = \mathrm{Softmax_{row}}(Z_{m'i:m'(i+1)-1} + B)$$
$$C_i^{\mathrm{Comp}} = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j$$

对比 CSA 的关键差异:
- $m' \gg m$: Flash 和 Pro 都是 $m'=128$, 比 CSA 的 $m=4$ 大 32 倍
- 没有 overlapped compression (只用当前 block, 不借用前一个 block)
- 没有 sparse selection (dense attention over all compressed entries)
- 1M context 下, HCA 只有 $10^6 / 128 = 7812$ 个 compressed KV entries, dense attention 仍然可行

**Intuition**: CSA 和 HCA 代表了两种不同的 efficiency-complexity 权衡:
- CSA: 适度 compression + sparsity → 更精细的 token-level attention, 适合需要 fine-grained selection 的场景
- HCA: 极端 compression + density → 全局信息聚合, 适合需要 broad context awareness 的场景

交错使用两种 attention, 让模型同时获得 fine-grained 和 coarse-grained 的 attention 能力。这种 hybrid 设计在 V3 的 NSA (Native Sparse Attention) 基础上进一步发展。

参考: [DeepSeek-V3 NSA](https://arxiv.org/abs/2502.11089)

#### 2.1.3 其他细节

**Partial RoPE**: 只对 query/KV/attention output 的最后 64 维应用 RoPE。由于 KV 同时充当 key 和 value, naive core attention output 会携带 absolute position embeddings。Countermeasure: 对 attention output 的最后 64 维也应用 RoPE (position $-t$), 使得 output 携带 relative position embeddings。这是个非常巧妙的 trick, 解决了 shared KV MQA 与 RoPE 的兼容性问题。

**Sliding Window Attention 分支**: 每个 query 额外保留 $n_{\mathrm{win}}$ (=128) 个 uncompressed KV entries (最近 tokens), 与 compressed KV entries 一起参与 core attention。这解决了两个问题:
1. 严格因果性导致 query 无法访问同一 compression block 内的 tokens
2. Recent tokens 通常与 query 最相关

**Attention Sink** (公式 27):
$$s_{h,i,j} = \frac{\mathrm{Exp}(z_{h,i,j})}{\sum_k \mathrm{Exp}(z_{h,i,k}) + \mathrm{Exp}(z_h')}$$

变量解释:
- $z_{h,i,j}$: 第 $h$ 个 head, 第 $i$ 个 query token, 第 $j$ 个 KV entry 的 logit
- $z_h'$: 第 $h$ 个 head 的 learnable sink logit

这个 trick 让 attention scores 的总和可以小于 1 (甚至接近 0), 相当于允许模型 "不 attend 到任何 KV"。这避免了 StreamingLLM 中发现的 attention 分配过度集中问题。

参考: [StreamingLLM](https://arxiv.org/abs/2309.17453)

#### 2.1.4 效率讨论

KV cache 的混合存储: BF16 (RoPE 维度) + FP8 (其他维度)。相比纯 BF16 减半。

Lightning indexer 用 FP4 计算, 加速 extreme long context 下的 attention score 计算。Index scores $I_{:, :}$ 从 FP32 降到 BF16, top-k selector 速度 2×, recall 99.7%。

以 BF16 GQA8 (head_dim=128) 为 baseline, V4 的 KV cache 在 1M context 下只有 baseline 的 ~2%。

---

### 2.2 Manifold-Constrained Hyper-Connections (mHC)

这是对 standard residual connection 的升级。先看 standard HC (Hyper-Connections)。

#### 2.2.1 Standard HC

公式 (1):
$$X_{l+1} = B_l X_l + C_l \mathcal{F}_l(A_l X_l)$$

变量解释:
- $X_l = [\mathbf{x}_{l,1}; ...; \mathbf{x}_{l,n_{\mathrm{hc}}}]^T \in \mathbb{R}^{n_{\mathrm{hc}} \times d}$: 第 $l$ 层之前的 residual state
- $n_{\mathrm{hc}}$: expansion factor (=4)
- $d$: hidden size (Flash=4096, Pro=7168)
- $A_l \in \mathbb{R}^{1 \times n_{\mathrm{hc}}}$: input mapping
- $B_l \in \mathbb{R}^{n_{\mathrm{hc}} \times n_{\mathrm{hc}}}$: residual transformation
- $C_l \in \mathbb{R}^{n_{\mathrm{hc}} \times 1}$: output mapping
- $\mathcal{F}_l$: 第 $l$ 层的实际计算 (MoE 等), 输入输出都是 $\mathbb{R}^d$

Intuition: HC 把 residual stream 从 $\mathbb{R}^d$ 扩展到 $\mathbb{R}^{n_{\mathrm{hc}} \times d}$, 提供了一个额外的 scaling axis。$n_{\mathrm{hc}}$ 远小于 $d$, 计算开销很小, 但提供了更大的表达能力。$A_l X_l \in \mathbb{R}^d$ 是实际 layer input, 所以内部 layer 设计不受影响。

**问题**: 标准 HC 在 stacking 多层后频繁出现 numerical instability。

#### 2.2.2 mHC 的核心创新: 流形约束

公式 (2):
$$B_l \in \mathcal{M} := \{M \in \mathbb{R}^{n \times n} | M \mathbf{1}_n = \mathbf{1}_n, \mathbf{1}_n^T M = \mathbf{1}_n^T, M \geqslant 0\}$$

这是 **Birkhoff polytope**——所有 doubly stochastic matrices 的集合。

为什么这个约束重要?
1. **谱范数有界**: 双随机矩阵的谱范数 $\|B_l\|_2 \leq 1$, 所以 residual transformation 是 **non-expansive** mapping。信号在多层传播时不会 explosion。
2. **乘法封闭**: 集合 $\mathcal{M}$ 在矩阵乘法下封闭, 保证 deep stack 的稳定性。
3. **避免 signal cancellation**: $A_l$ 和 $C_l$ 用 Sigmoid 约束为 non-negative 且 bounded。

**Intuition**: 标准 HC 的 $B_l$ 是自由矩阵, 可以任意放大或缩小信号。多层 stacking 后, 谱范数可能指数级增长 (forward) 或衰减 (backward), 导致 instability。约束到 Birkhoff polytope 后, 谱范数 bounded by 1, 信号传播稳定。

#### 2.2.3 Dynamic Parameterization

公式 (3-5): 三个 mapping 的参数由 dynamic (input-dependent) + static (input-independent) 组成:
$$\tilde{A}_l = \alpha_l^{\mathrm{pre}} \cdot (\hat{X}_l W_l^{\mathrm{pre}}) + S_l^{\mathrm{pre}}$$
$$\tilde{B}_l = \alpha_l^{\mathrm{res}} \cdot \mathrm{Mat}(\hat{X}_l W_l^{\mathrm{res}}) + S_l^{\mathrm{res}}$$
$$\tilde{C}_l = \alpha_l^{\mathrm{post}} \cdot (\hat{X}_l W_l^{\mathrm{post}})^T + S_l^{\mathrm{post}}$$

变量解释:
- $\hat{X}_l = \mathrm{RMSNorm}(\mathrm{vec}(X_l)) \in \mathbb{R}^{1 \times n_{\mathrm{hc}} d}$: flatten + 归一化后的输入
- $W_l^{\mathrm{pre}}, W_l^{\mathrm{post}} \in \mathbb{R}^{n_{\mathrm{hc}} d \times n_{\mathrm{hc}}}$, $W_l^{\mathrm{res}} \in \mathbb{R}^{n_{\mathrm{hc}} d \times n_{\mathrm{hc}}^2}$: learnable 参数生成 dynamic components
- $\mathrm{Mat}(\cdot)$: 把 $1 \times n_{\mathrm{hc}}^2$ 向量 reshape 成 $n_{\mathrm{hc}} \times n_{\mathrm{hc}}$ 矩阵
- $S_l^{\mathrm{pre}}, S_l^{\mathrm{post}}, S_l^{\mathrm{res}}$: learnable static biases
- $\alpha_l^{\mathrm{pre}}, \alpha_l^{\mathrm{res}}, \alpha_l^{\mathrm{post}} \in \mathbb{R}$: learnable gating factors, 初始化为小值

#### 2.2.4 流形投影: Sinkhorn-Knopp 算法

公式 (6-8):
$$A_l = \sigma(\tilde{A}_l)$$
$$C_l = 2\sigma(\tilde{C}_l)$$
$$M^{(0)} = \exp(\tilde{B}_l)$$
$$M^{(t)} = \mathcal{T}_r(\mathcal{T}_c(M^{(t-1)}))$$

最终 $B_l = M^{(t_{\max})}$, $t_{\max} = 20$。

Sinkhorn-Knopp 算法: 反复交替做 row normalization 和 column normalization, 收敛到 doubly stochastic matrix。这个算法在 optimal transport 等领域很经典。

Intuition: $\exp(\tilde{B}_l)$ 保证 positivity, 然后迭代归一化让每行每列和为 1。这是一个 projection 操作, 把任意矩阵投影到 Birkhoff polytope 上。

参考: [mHC Paper](https://arxiv.org/abs/2512.24880), [Sinkhorn-Knopp Algorithm](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem)

---

### 2.3 Muon Optimizer

Muon 是 [Jordan et al., 2024](https://github.com/KellerJordan/Muon) 提出的, [Liu et al., 2025](https://arxiv.org/abs/2502.16982) 证明 scalable for LLM training。核心 idea: **对 gradient matrix 做 orthogonalization, 再更新参数**。

#### 2.3.1 Algorithm 1 解析

```
for each weight W ∈ R^{n×m}:
    G_t = ∇_W L_t(W_{t-1})                    # 计算梯度
    M_t = μ M_{t-1} + G_t                       # 累积 momentum
    O_t' = HybridNewtonSchulz(μ M_t + G_t)     # Nesterov + 正交化
    O_t = O_t' · √max(n,m) · γ                  # Rescale RMS
    W_t = W_{t-1} · (1 - ηλ) - η O_t            # Weight decay + 更新
```

为什么 orthogonalize? Muon 的核心 insight 是: gradient 的方向很重要, 但 magnitude 可能被 outlier 扭曲。对 gradient matrix 做近似 orthogonalization (变成正交矩阵) 后, 每个参数方向的 update "公平"——spectral norm = 1, 所有方向被 equally 优化。这比 Adam 的 per-element scaling 更适合 matrix parameters。

#### 2.3.2 Hybrid Newton-Schulz Iterations

公式 (28):
$$M_k = a M_{k-1} + b (M_{k-1} M_{k-1}^T) M_{k-1} + c (M_{k-1} M_{k-1}^T)^2 M_{k-1}$$

SVD: $M = U \Sigma V^T$, 目标是近似 $UV^T$ (正交部分)。

V4 的 hybrid 策略:
- **前 8 步**: $(a, b, c) = (3.4445, -4.7750, 2.0315)$ → 快速收敛, singular values 接近 1
- **后 2 步**: $(a, b, c) = (2, -1.5, 0.5)$ → 精确稳定, singular values 严格为 1

Intuition: Newton-Schulz 迭代本质是 polynomial approximation of $\mathrm{sign}(\sigma)$ 函数。不同系数对应不同的多项式, 收敛速度和稳定性有 trade-off。V4 的两阶段策略兼顾 speed 和 precision。

#### 2.3.3 V4 的具体配置

- AdamW: embedding, prediction head, RMSNorm, mHC 的 static biases 和 gating factors
- Muon: 其他所有 modules

Hyper-parameters:
- Muon momentum: 0.95
- Weight decay: 0.1
- Update RMS rescale: 0.18 (reuse AdamW learning rate)
- 学习率: Flash $2.7 \times 10^{-4}$, Pro $2.0 \times 10^{-4}$

**避免 attention logit explosion**: V4 的 attention 允许直接对 query/KV 做 RMSNorm, 防止 logit explosion, 所以不需要 QK-Clip。

---

## 3. Infrastructure: Engineering 是真正的难点

### 3.1 Fine-Grained Communication-Computation Overlap in EP

MoE 的 Expert Parallelism (EP) 需要 complex inter-node communication。V4 的方案:

**核心 insight**: Communication latency 可以被 computation 完全隐藏, 只要 computation-communication ratio 足够高。

对 V4-Pro, 每个 token-expert pair:
- Computation: $6h$ FLOPs (SwiGLU gate, up, down projections)
- Communication: $3h$ bytes (FP8 Dispatch + BF16 Combine)

所以 $C/B \leq 2d = 6144$ FLOPs/Byte, 即每 GBps 带宽可以隐藏 6.1 TFLOP/s 的计算。一旦带宽满足这个阈值, 就不再是 bottleneck。

**Wave-based scheduling**: 把 experts 分成小波次, 每波内的 experts 完成通信后立即开始计算, 同时下一波的 token 传输和已完成 experts 的结果发送并发进行。形成 fine-grained pipeline。

性能: 1.50-1.73× speedup (general inference), 1.96× (RL rollouts/agent serving)。

开源实现: [MegaMoE (part of DeepGEMM)](https://github.com/deepseek-ai/DeepGEMM)

**硬件建议** (给硬件厂商的):
1. Computation-Communication Ratio 比 bandwidth 本身更重要
2. Power Budget: 极致 kernel fusion 让 compute/memory/network 同时高负载, power throttling 成为 limiter
3. Communication Primitives: Pull-based 避免 push 的高 notification latency
4. Activation Function: 用 element-wise activation 替代 SwiGLU, 移除 gate projection 扩大 intermediate dim

---

### 3.2 TileLang: DSL for Kernel Development

[TileLang](https://arxiv.org/abs/2502.19811) 平衡开发效率和运行时性能。三个核心特性:

1. **Host Codegen**: 把 host-side logic (runtime contract checks 等) 从 Python 移到 generated code。CPU-side validation overhead 从几十/几百 microseconds 降到 <1 microsecond。

2. **SMT-Solver-Assisted Formal Integer Analysis**: 集成 Z3 SMT solver, 对 tensor index 算术做 formal analysis。支持 quantifier-free non-linear integer arithmetic (QF_NIA)。Vectorization, barrier insertion, code simplification 都受益。

3. **Numerical Precision and Bitwise Reproducibility**: 默认 disable fast-math, 提供 IEEE-compliant intrinsics (T.ieee_fsqrt, T.ieee_fdiv, T.ieee_add)。Layout annotations 允许 pin down lowering decisions, 实现 bit-identical outputs。

---

### 3.3 Batch-Invariant and Deterministic Kernels

**Batch Invariance**: 任何 token 的 output 与其在 batch 中的位置无关, bitwise identical。

挑战和解法:
- **Attention**: 不能用 split-KV (会跨 SM 分布计算, 破坏 batch invariance)。Dual-kernel strategy: Kernel 1 (单 SM 处理整 sequence, 高吞吐) + Kernel 2 (多 SM 处理单 sequence, 缓解 wave-quantization)。两者 accumulation order 精心对齐。
- **Matrix Multiplication**: 用 DeepGEMM 替代 cuBLAS。避免 split-k (破坏 batch invariance), 用其他优化弥补性能。
- **mHC Matrix Multiplication**: output dim 只有 24, 小 batch size 需 split-k。输出各 split 部分, 后续 deterministic reduction。

**Determinism**: 避免 atomicAdd 的 non-determinism。
- Attention backward: 每个 SM 独立 accumulation buffer, 然后 global deterministic sum。
- MoE backward: token order pre-processing + buffer isolation across ranks。
- mHC: 用 split-k 时输出各部分 + 后续 deterministic reduction。

参考: [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [Stream-K Paper](https://doi.org/10.1145/3588568.3590139)

---

### 3.4 FP4 Quantization-Aware Training

FP4 (MXFP4) 量化用于:
1. **MoE expert weights**: GPU memory 占用大头
2. **CSA indexer 的 QK path**: 长上下文下 attention score 计算加速

**Lossless FP4-to-FP8 dequantization**: FP8 (E4M3) 比 FP4 (E2M1) 多 2 个 exponent bits, dynamic range 更大。只要 FP4 sub-blocks (1×32 tiles) 在 FP8 quantization block (128×128 tiles) 内的 scale factor 比值不超过阈值, FP4 的 fine-grained scale 信息可以完全被 FP8 的 dynamic range 吸收。

**STE (Straight-Through Estimator)**: Backward pass 直接对 FP8 weights 计算梯度, 传播回 FP32 master weights。避免 re-quantize transposed weights。

**Index scores 量化**: $I_{:, :}$ 从 FP32 降到 BF16, top-k selector 速度 2×, recall 99.7%。

参考: [MXFP4 Paper](https://arxiv.org/abs/2310.10580), [QAT Paper](https://arxiv.org/abs/1712.05877)

---

### 3.5 Training Framework 的关键优化

#### 3.5.1 Muon 与 ZeRO 的兼容

Muon 需要完整 gradient matrix, 与 ZeRO 的 element-wise partition 冲突。V4 的 hybrid 策略:

- **Dense params**: knapsack algorithm 分配 parameter matrices 到 ranks, 每 rank 管理 ~5 个 matrices, padding <10% memory overhead。
- **MoE params**: 每个 expert 独立优化。Flatten 所有 experts 的 down/up/gate projections, padding 后均匀分布到所有 ranks, 不分割任何 logically independent matrix。
- **BF16 gradient sync**: Newton-Schulz 在 BF16 下稳定, gradient 量化到 BF16 减半通信。用 all-to-all + local FP32 sum 替代 tree/ring reduce-scatter, 避免 low-precision adder 的 accumulation error。

#### 3.5.2 mHC 的高效实现

mHC 增加 activation memory 和 pipeline 通信。优化:
1. **Fused kernels**: 训练和推理都优化
2. **Selective recomputation**: 重计算大部分 hidden states, 避免 compute-intensive operations 的重计算
3. **DualPipe 1F1B 调整**: 适应增加的 pipeline 通信, 并发执行 mHC 内的部分操作

Wall-time overhead 只有 overlapped 1F1B pipeline stage 的 6.7%。

#### 3.5.3 Contextual Parallelism for Long-Context Attention

传统 CP 按 sequence dim 分割, 每 rank 维护 $s$ 个 contiguous tokens。CSA/HCA 的挑战:
1. Packed sequences, 每个 sequence 独立压缩, 尾部 tokens < $m$ 被丢弃, compressed KV lengths 跨 rank 不同
2. Compression 需要 $m$ 个连续 KV entries, 可能跨 CP rank 边界

V4 的两阶段通信:
1. 每 rank 发送最后 $m$ 个 uncompressed KV entries 到 rank+1, rank+1 压缩部分 received + local entries, 产生 $s/m + 1$ compressed entries
2. All-gather 跨所有 CP ranks 收集 locally compressed entries, fused select-and-pad operator 重组

#### 3.5.4 Extended Automatic Differentiation for Flexible Checkpointing

传统 activation checkpointing 是 module 粒度, 太粗。V4 实现 tensor 级别:
- 开发者只需标注 individual tensors for checkpointing
- TorchFX 追踪 computation graph, 对每个 annotated tensor 找到 minimal recomputation subgraph
- 直接 free annotated tensor 的 GPU memory, reusing storage pointer (无 memory copy)
- 自动 deduplication: 共享 storage 的 tensors (如 reshape 的 input/output) 不会重复重计算

---

### 3.6 Inference Framework

#### 3.6.1 Heterogeneous KV Cache

V4 的 hybrid attention 引入多种 KV entries:
- CSA compressed KV (compression ratio $m=4$)
- HCA compressed KV (compression ratio $m'=128$)
- Lightning indexer KV (不同 embedding size)
- SWA KV (不同 cache hit/eviction 策略)
- Uncompressed tail tokens (等待 compression 的 buffer)

PagedAttention 假设所有 layers 相同 KV cache size, 不适用 V4。

V4 方案:
1. **State Cache for SWA + Uncompressed Tail**: 固定大小 pool, 动态分配给每个 sequence。SWA 和 uncompressed tail 被视为 state-space model 的 sequence-specific state。
2. **Sparse Attention Kernel Co-Design**: KV cache layout 与 sparse attention kernel 共同设计。Block size 可以是 $\mathrm{lcm}(m, m')$ 的任意倍数, 允许不同 layers 使用不同 block 大小而无性能损失。

#### 3.6.2 On-Disk KV Cache

对于 shared-prefix requests, 避免重复 prefilling。

CSA/HCA 的 compressed KV 直接存 disk, hitting prefix 时读取重用 (到最后一个完整 compression block)。

SWA KV 体积大 (~8× compressed CSA/HCA), 三种策略:
1. **Full SWA Caching**: 存所有 SWA KV, 零计算冗余。但 SSD write-intensive access pattern 效率低。
2. **Periodic Checkpointing**: 每 $T$ tokens checkpoint 最近 $n_{\mathrm{win}}$ 个 SWA KV。Hitting 时加载最近 checkpoint, 重算 tail。
3. **Zero SWA Caching**: 不存 SWA KV, 利用 cached CSA/HCA KV, 重算最后 $n_{\mathrm{win}} \cdot L$ tokens (L = layer 数) 恢复最后 $n_{\mathrm{win}}$ 个 SWA KV。

---

## 4. Pre-Training

### 4.1 Data

32T+ tokens, 包含 math, code, web, long documents, multilingual。

**Sample-level attention masking**: 不同 V3, V4 在 packing documents 时用 sample-level attention masking, 避免 cross-document attention。这比 V3 的 padding 浪费更少。

**Agentic data mid-training**: mid-training 阶段引入 agentic data 增强 coding 能力。

### 4.2 Model Setups

| 参数 | V4-Flash | V4-Pro |
|------|----------|--------|
| Layers | 43 | 61 |
| Hidden dim $d$ | 4096 | 7168 |
| First layers attention | SWA | HCA |
| Subsequent attention | CSA + HCA interleaved | CSA + HCA interleaved |
| CSA compression $m$ | 4 | 4 |
| HCA compression $m'$ | 128 | 128 |
| CSA top-k | 512 | 1024 |
| Indexer heads $n_h^I$ | 64 | 64 |
| Indexer head dim $c^I$ | 128 | 128 |
| Query heads $n_h$ | 64 | 128 |
| Head dim $c$ | 512 | 512 |
| Query compress dim $d_c$ | 1024 | 1536 |
| Output projection groups $g$ | 8 | 16 |
| Intermediate output dim $d_g$ | 1024 | 1024 |
| SWA window $n_{\mathrm{win}}$ | 128 | 128 |
| MoE experts | 256 routed + 1 shared | 384 routed + 1 shared |
| Activated experts | 6 | 6 |
| Expert intermediate dim | 2048 | 3072 |
| MTP depth | 1 | 1 |
| mHC expansion $n_{\mathrm{hc}}$ | 4 | 4 |
| Sinkhorn iterations | 20 | 20 |
| Total params | 284B | 1.6T |
| Activated params | 13B | 49B |

**MoE 调整**:
- Affinity score function: `Sigmoid(·)` → `Sqrt(Softplus(·))`
- 移除 routing target 数量限制
- 前 3 个 MoE layers 用 **Hash routing** (基于 token ID 的 hash 函数决定 routing)

**Intuition on Hash routing for early layers**: 早期 layers 的 token 还没有被 attention 充分 contextualize, routing 决策应该基于 token identity 而非 contextualized representation。Hash routing 简单高效, 避免 early layer routing 的训练不稳定。

### 4.3 训练不稳定性的解决

训练 trillion-param MoE 的两个实用技巧:

#### 4.3.1 Anticipatory Routing

Observation: MoE outliers 与 routing mechanism 形成恶性循环。

Solution: Step $t$ 用 $\theta_t$ 计算 features, 但用 $\theta_{t-\Delta t}$ 计算 routing indices。

Implementation: Step $t - \Delta t$ 时提前 fetch step $t$ 的数据, 计算 cache routing indices 供 step $t$ 使用。

Optimization: Pipeline execution + EP communication overlap, 额外 wall-time <20%。

**Auto-trigger**: Loss spike 时自动 rollback + 启用 Anticipatory Routing, 稳定后恢复标准训练。Dynamic 应用, 总体额外开销可忽略。

**Intuition**: 这是一种 "lagged routing" 策略。Routing 决策相对 backbone 参数滞后, 避免了 routing network 与 backbone 同步更新导致的 feedback loop。类似 stochastic depth 或 dropout 的 "lagged" 效应——某种意义上的 implicit regularization。

#### 4.3.2 SwiGLU Clamping

- Linear component: clamp 到 $[-10, 10]$
- Gate component: 上限 10

**Intuition**: SwiGLU 的 unbounded exponential/growth 在 outlier tokens 上会产生极端激活值, 这些 outliers 通过 residual connection 传播放大, 形成 loss spike。Clamping 是 explicit 的 saturation, 牺牲少量 expressivity 换取稳定性。

参考: [Gemma 2 Paper (Riviere et al., 2024)](https://arxiv.org/abs/2408.00118), [gpt-oss Model Card](https://arxiv.org/abs/2508.10925)

### 4.4 Base Model 评估

V4-Flash-Base 在大部分 benchmarks 上超越 V3.2-Base, 尽管 activated/total params 都更少。V4-Pro-Base 进一步确立新 SOTA。

特别值得注意:
- **FACTS Parametric**: V3.2=27.1, Flash=33.9, Pro=62.6 (巨大提升)
- **SimpleQA-Verified**: V3.2=28.3, Flash=30.1, Pro=55.2
- **LongBench-V2**: V3.2=40.2, Flash=44.7, Pro=51.5

**Intuition**: Pro vs Flash 的巨大差距在 knowledge benchmarks 上 (SimpleQA 25+ point gap), 这符合 scaling laws——knowledge 存储在 parameters 中, 更多 params = 更多 knowledge。但 reasoning tasks 差距较小 (HMMT: Flash=94.8, Pro=95.2), 因为 reasoning 更多依赖于 architecture 和 training strategy。

---

## 5. Post-Training: Specialist Training + On-Policy Distillation

### 5.1 Pipeline 概览

V4 的 post-training 完全用 **On-Policy Distillation (OPD)** 替代了 V3.2 的 mixed RL stage。Pipeline:

1. **Specialist Training**: 对每个 domain (math, code, agent, instruction following) 独立训练 expert
   - SFT on domain-specific data
   - RL with GRPO + domain-specific reward models
   
2. **On-Policy Distillation**: 多个 teacher experts → 单一 student model

### 5.2 Reasoning Effort 三模式

| Mode | 特征 | Use Case |
|------|------|----------|
| Non-think | 快速直觉响应 | 日常低风险决策 |
| Think High | 有意识逻辑分析 | 复杂问题求解, 中等风险决策 |
| Think Max | 推理极致 | 探索 reasoning 边界 |

每种 mode 用不同 RL 配置 (length penalty, context window) 训练。Response format 用 `
