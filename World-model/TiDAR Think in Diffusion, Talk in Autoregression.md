---
source_pdf: TiDAR Think in Diffusion, Talk in Autoregression.pdf
paper_sha256: 988eb15cf3db3d1389a8635572da7ad6a6f0f174e1febd9fbde0517252990e31
processed_at: '2026-08-12T16:00:46-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# TiDAR 人话版

Andrej, 咱们坐下来喝杯咖啡讲讲这玩意儿。把学术腔调全扔掉, 用最直白的话重新捋一遍。

---

## 一句话概括

**现在的 LLM 推理太浪费 GPU 了。你 batch size=1 跑 AR decoding, 90% 的时间 GPU 在那儿干等 HBM 把 weights 吐出来。TiDAR 说: 既然都等了, 顺手在同一个 forward 里把后面几个 token 也预测了呗。**

就这么个事。但做得特别精巧, 精巧到可以同时拿到 diffusion 的并行速度 + AR 的质量保证。

---

## 为什么要这么搞? 背后的物理事实

你想想你在 llm.c 里跑 GPT 推理。batch size=1, 模型 weights 几十 GB, 每个 token forward 一次, 都得把 weights 从 HBM 搬到 SRAM。这个搬运时间几乎不变, 不管你 forward 输出 1 个 token 还是 8 个 token, 搬运成本一样。

这个就叫做 **memory-bound**。GPU 算力在那儿空转, 90% 时间在等带宽。

论文 Figure 1 里那个 profiling 是这意思: Qwen3-32B 在 H100 上, 你给它塞几个 extra token slot 进 forward, latency 几乎不动。一直塞到某个 threshold 才开始涨 — 那个点就是从 memory-bound 跨进 compute-bound 了。

所以 TiDAR 的物理 motivation 就一句话: **GPU 在等内存的时候, 你让它干点活儿, 别闲着**。

这个 idea 其实 Medusa、EAGLE、DeepSeek MTP 都想到了, 都是 speculative decoding 那一脉。问题在于它们还是 "draft 然后 verify" 的 sequential 流程 — 用一个小 draft model 先猜, 再用大 model 验证。两阶段, 再快也得两次 forward。

TiDAR 干的是: **一个 forward 同时 draft + verify**。这才是真正的"白嫖"。

---

## 为什么不用纯 diffusion LLM?

Diffusion LM 听起来挺美好 — 一次能 decode k 个 token 并行出来。Dream、LLaDA 都在做。但问题在哪?

你看 diffusion 的数学:

$$p_{\text{Diff}}(\cdot; \theta) = \mathbb{E}_{\tilde{\mathbf{x}} \sim q(\cdot|\mathbf{x})} \prod_i p_\theta^i(x_i | \tilde{\mathbf{x}})$$

人话翻译: 给定一个加了噪(部分 mask 掉)的序列 $\tilde{\mathbf{x}}$, 每个被 mask 的位置 $i$ 各自独立地预测自己应该是什么 token, 然后 $\prod_i$ 乘起来。

这里的 "独立" 是关键。你一次预测 4 个 token, 这 4 个位置**互相不知道对方会变成什么**, 各自从自己的 marginal distribution 里采样。

这跟 AR 完全不一样:

$$p_{\text{AR}}(\cdot; \boldsymbol{\theta}) = \prod_i p_\theta^i(x_i | \mathbf{x}_{<i}; \boldsymbol{\theta})$$

AR 里第 $i$ 个 token 条件化于前面所有已经确定的 token $\mathbf{x}_{<i}$, chain rule 一路下来, 严格匹配语言的因果性。

Diffusion LM 的并行越多, token 之间越独立, 序列级的 coherence 就崩。APD 那篇 paper 实测: Dream-7B 在 GSM8K 上, 1 token/step 到 2 token/step, 准确率掉 10%。再往上加更惨。

所以 diffusion LM 一直在质量跟速度之间撕扯。你想要质量, 就得 1 token/step, 那跟 AR 一样慢; 你想要速度, 质量就崩。

---

## TiDAR 怎么缝合的? Attention Mask 是灵魂

核心 trick 在 attention mask 上。一次 forward 里, 序列被切成三段:

### 段 1: Prefix (已经确认的 token)
attention 模式: **纯 causal**(像普通 AR)
KV cache: 永久保留, 复用

### 段 2: 上一步 draft 的 token (待 verify)
attention 模式: **causal with prefix**
KV cache: 接受就保留, 拒绝就丢掉

### 段 3: 下一步的 pre-draft mask token
attention 模式: **block-bidirectional**(块内双向 + 看前面 prefix)
KV cache: 用完就扔, 下一步不会再用

这三段在同一个 forward 里同时计算。attention mask 是精心设计的 — prefix 走 causal, 最后那个 decoding block 走 bidirectional。

跟 Block Diffusion 的区别在哪? Block Diffusion 是**每个 block 都是 intra-block bidirectional + inter-block causal**, 所以它的 prefix 部分也是分 block bidirectional 的。TiDAR 把 prefix 全改成纯 causal, 只留最后一个 block 是 bidirectional。

这一改有两个大好处:
1. prefix 是纯 causal, 可以算标准 NTP loss, 没有 label leakage
2. inference 时 prefix 走 causal, 可以直接接 AR 的 rejection sampling 逻辑

---

## 训练时候长啥样?

训练数据长这样: $[x_1, x_2, \dots, x_S, m_1, m_2, \dots, m_S]$

前一半是原 sequence, 后一半是 mask token。序列长度翻倍。

Loss 是两部分的加权和:

$$\mathcal{L}_{\text{TiDAR}}(\theta) = \frac{1}{1+\alpha} \bigg( \sum_{i=1}^{S-1} \frac{\alpha}{S-1} \mathcal{L}_{AR}(x_i, x_{i+1}; \theta) + \sum_{i=1}^{S-1} \frac{1}{S-1} \mathcal{L}_{Diff}([mask], x_i; \theta) \bigg)$$

变量解释:
- $\alpha$ 是 balancing factor, 默认 $\alpha=1$, 等权
- $S$ 是 sequence 长度(所以总 token 数是 $2S$)
- $\mathcal{L}_{AR}(x_i, x_{i+1}; \theta)$: prefix 位置 $i$ 的 logit 预测 $x_{i+1}$, 标准 NTP, label shift 1
- $\mathcal{L}_{Diff}([mask], x_i; \theta)$: mask 位置 $i$ 的 logit 预测 $x_i$, 无 label shift, 位置对齐

**为什么 diffusion 段无 shift?** Bidirectional attention 下, mask 位置 $i$ 直接预测原 sequence 位置 $i$ 就行, 不用 shift。这点跟 AR 段形成有意思的对比 — 同一个 transformer 在不同 mask 段同时承担两种 label alignment。

### Full Mask 这个小 trick 很关键

传统 diffusion LM 训练时随机 mask 一部分 token (比如 50%), loss 只在 masked 位置算。TiDAR 直接**全部 mask 掉**整个 diffusion 段。

三个好处:
1. **Loss 信号密集**: 每个 mask 位置都贡献一个 loss term, 不再稀疏
2. **Loss balancing 变 trivial**: AR loss 项数 = $S-1$, Diffusion loss 项数 = $S-1$, 直接 $\alpha$ 加权, 不用考虑每次 mask 比例不同
3. **Train-test consistency**: inference 时反正也只做 one-step diffusion (没有 iterative denoising), 全是 mask, 跟训练 100% 对齐

Table 5 ablation: 这个改动让 HumanEval Avg 从 32.62% → 38.42% (draft=4), GSM8k 也涨 ~1pt, T/NFE 略升。**零成本提升**, 因为本来就要送 mask token 进 forward。

---

## Inference 流程: 一次 forward 干三件事

每个 decoding step:

**第一件**: Verify 上一步 draft 出来的 token
- 通过 causal attention 在 prefix 段算出每个位置的 AR distribution $p_{AR}(\cdot|x_{<i})$
- 用 rejection sampling 跟 draft token 对比: 如果 draft token 跟 AR 采样的一致, 接受; 不一致, 用 AR 采样的 token 替换, 后面全 reject

**第二件**: Pre-draft 下一步的 token
- mask 段通过 block-bidirectional attention 算 $p_{Diff}$ 的 marginal
- 直接采样出下一批 draft token

**第三件** (这是巧思): 基于**所有可能 verify outcome** 的 pre-draft
- 上一步 draft 了 $[d_1, d_2, d_3]$, rejection sampling 可能 accept 0/1/2/3 个
- 下一步 pre-draft 要针对这 4 种 prefix 状态都准备一份
- 在 mask section 里, 因为各 mask 位置独立, 自然能分别对不同 prefix 状态出 pre-draft

这一切都在**同一个 forward pass** 完成。drafting 和 verify 是真正并行的, 不是 sequential 两阶段。

---

## Trust AR 还是 Trust Diff?

由于 diffusion 段无 label shift, AR 段有 label shift, **位置 $i$ 在两个段都会预测同一个 token**。所以可以 mix logits:

$$\text{logits}_i^{\text{mixed}} = \beta \cdot \text{logits}_i^{ar} + (1-\beta) \cdot \text{logits}_i^{diff}$$

$$\text{sampled} = \arg\max_{i \in |V|} \{ \text{logits}_i^{\text{mixed}} \}$$

变量: $\beta \in [0,1]$ 是 trust factor, $|V|$ 是 vocab 大小。$\beta=1$ 完全信 AR, $\beta=0$ 完全信 diffusion。

Figure 6 ablation 显示 $\beta$ 在 $[0,1]$ 扫, 质量几乎不变 (1.5B 上) — 说明 model 训练得到两个 distribution 已经高度对齐。但 8B 上 Trust Diff 在 GSM8k 反而略高 (80.44% vs 79.83% Trust AR), 暗示大模型上 diffusion 的 marginal 反而更全局一致。

直觉上: 这说明 rejection sampling 才是质量保证的真正来源, 而非 AR logit 本身。draft quality 决定 acceptance rate, 但 sampling correctness 由 rejection sampling 的数学保证决定。

---

## 实验数据

### 速度

- TiDAR 1.5B: **4.71× vs Qwen2.5-1.5B base**
- TiDAR 8B: **5.91× vs Qwen3-8B base**

8B 模型平均一次 forward 出 **8.25 个 token**, AR 是 1 个。差不多 6× 的吞吐。

对比 EAGLE-3 公开权重, TiDAR 首次让 diffusion-based 方法在效率上**超过** speculative decoding。原因: TiDAR 的 raw acceptance rate (T/NFE) 更高, 且 conversion rate (T/NFE → T/s) 更高 — 因为 drafting 和 verify 在同一 forward, 没有 sequential 两阶段开销。

### 质量

8B 规模 (Table 2):

| 任务 | Qwen3-8B (AR base) | Dream-7B | TiDAR-8B Trust AR | TiDAR-8B Trust Diff |
|---|---|---|---|---|
| HumanEval | 64.63% | 54.88% | 55.49% (7.46 T/NFE) | 57.93% (7.30 T/NFE) |
| GSM8k | 81.80% | 77.18% | 79.83% (6.90) | 80.44% (7.07) |
| Avg | 68.09% | 58.74% | 63.90% (8.23) | 65.31% (8.25) |

比 Dream 高 6.5 pt, 比 Block Diff 高 5 pt, 离 Qwen3-8B base 还差 2.8 pt。差的那 2-3 pt 论文承认是 continual pretraining 50B-150B tokens 不够 — TiDAR 训练时一半梯度去学 diffusion mode 了, 知识容量被分摊。更多数据应该能闭合。

### Pareto Frontier (Figure 5)

同 50B 训练 token 下, 1.5B 规模:
- AR base: 1 T/NFE, avg ~50%
- Fine-tuned AR: 1 T/NFE, avg ~55%
- Block Diff (threshold 0.8): ~3 T/NFE, avg ~48%
- TiDAR (block 4/8/16): 3-7 T/NFE, avg ~52-53%

TiDAR 在 7× T/NFE 下接近 fine-tuned AR 质量, 这是 Pareto frontier 的大跃迁。

### 最让我惊讶的 ablation (Table 4)

1.5B 规模, full mask 模型, 不同解码策略对比:

| 策略 | T/NFE | HumanEval Avg | GSM8k |
|---|---|---|---|
| Confidence Max (1 tok) | 1.00 | 36.28% | 53.37% |
| Confidence Max (2 tok) | 2.00 | 21.95% | 41.32% |
| Confidence > 0.9 | 2.63 | 32.01% | 51.40% |
| Confidence > 0.6 | 3.81 | 22.56% | 37.60% |
| **TiDAR (4 drafts)** | **3.47** | **38.42%** | **55.87%** |
| **TiDAR (16 drafts)** | **6.97** | **41.16%** | **53.90%** |

注意看这个趋势: 所有 confidence-based 方法在 T/NFE 升高时质量急速崩盘 (Confidence Max 从 36% 掉到 21%)。**TiDAR 是唯一一个 T/NFE 越高、质量不降反升的方法** (38.42% → 41.16%)。

这跟所有现有 dLM 的 trade-off 完全反过来。原因是更长的 draft length 让 rejection sampling 有更多 verify 机会, 接受的部分越多, 越接近 AR distribution 的质量。换句话说, TiDAR 在效率上限的方向上质量还在涨, 这是个反直觉的现象。

---

## 跟其他方法横着比

Table 1 给的对比一目了然:

| 框架 | Drafter 共享 base? | Draft capacity | Parallel drafting? | Draft 与 verify 并行? |
|---|---|---|---|---|
| Classic Spec Dec | ✗ (小 draft) | Low | ✗ | ✗ |
| APD | ✗ | High | ✓ | ✗ |
| EAGLE-3 | ✗ (额外 head) | Mid | ✗ | ✗ |
| DeepSeek-V3 MTP | ✗ (额外 layer) | Mid | ✗ | ✓ |
| Apple MTP | ✓ | High | ✗ | ✓ |
| **TiDAR** | **✓** | **High** | **✓** | **✓** |

TiDAR 是唯一同时打勾四个 box 的方法。EAGLE / DeepSeek MTP 的 drafter 是 AR + 单 last token input, 无法利用 mask token 的并行计算。TiDAR drafting 直接在 mask section 上 block-bidirectional, $k$ 个 mask 一次性算完。

---

## KV Cache 这块的处理

Exact KV cache 在 diffusion LM 一直是个痛点 — bidirectional attention 让 cache 失去意义。Block Diffusion 通过 inter-block causal 部分解决。TiDAR 进一步:

- Prefix 段 KV cache: 永久保留
- 上一步 draft 段 KV cache: 接受则保留进 prefix, 拒绝则 evict
- 当前 pre-draft 段 KV cache: 用完即弃

**关键: 不做任何 recompute**。Block Diffusion 因为 inter-block 是 causal, 前 block 接受后 block 还要重新走 attention, 有 recompute 开销。Fast-dLLM 用 approximate cache 有质量损失。d-KV Cache 延迟 cache 也有问题。TiDAR 因为 prefix 段全程 causal, 完美支持 exact cache。

这点对生产 serving 重要 — 不用重算 KV cache, O(1) 的 cache 操作, 性能可以保证。

---

## 我觉得可疑或者可以挑刺的地方

### 1. Sequence length 翻倍

训练时序列变 $2S$, 显存翻倍。1.5B 用 4096 max len, 实际等于 8192 的 attention computation。8B 还开了 gradient checkpointing。这个对训练成本不友好。

Section 5 提到 long context extension 没做, 正是这个 doubling 的副作用。一个可能改进: **packing + RoPE 位置 ID 不翻倍**, 只让 attention mask 翻倍结构, 可能省一半显存。

### 2. Batch size > 1 的故事没讲清

论文说 "we focus on batch size = 1 efficiency benchmarking", 承认大 batch 下 free token slots 收益缩小 (进入 compute-bound 区)。这点对生产 serving 重要 — 实际 serving 都是 large batch。TiDAR 在 large batch 下的 Pareto 位置需要后续工作。

### 3. Rejection sampling 的 math 没展开

Paper 里反复提 "rejection sampling guided by $p_{AR}$" 但没给完整的 accept/reject 算法。Speculative decoding 经典版是 Leviathan 2022 的:

$$P(\text{accept } x) = \min(1, p_{AR}(x)/p_{draft}(x))$$

TiDAR 的版本应该类似, 但 $p_{draft}$ 是 diffusion marginal。这个 ratio 在 well-trained model 上接近 1 (paper 在 4.4.3 的 ablation 印证), 所以 acceptance rate 高。完整推导值得单独写一篇。

### 4. Trust Diff 在大模型上更好, 为什么?

Table 2 显示 TiDAR 8B Trust Diff > Trust AR 在多数任务上 (尤其 GSM8k 80.44 vs 79.83)。这暗示在大模型上 diffusion 的 marginal distribution 反而比 AR 的 chain conditional 更鲁棒。可能因为 AR 在 math 长链上容易累积误差, diffusion 的全局视野更好。这是个有趣的 hypothesis, 值得单独 study。

### 5. Training data 配比

50B token 对 1.5B, 150B 对 8B, 这个量级对 continual pretraining 算 moderate。Qwen2.5 / Qwen3 base 的原训练 token 数远超这个。TiDAR 的 quality gap 多大程度来自架构, 多大程度来自数据量, 从 Figure 5 看仍有 2-3 pt gap on fine-tuned AR。Paper 说 "需要更多数据", 这个论点如果能上 1T token 训练验证会更有说服力。

---

## 我的整体判断

这篇 paper 的核心贡献是 **attention mask 工程上的精确切割**, 把 diffusion 的 parallelism 和 AR 的 chain factorization 在一次 forward 里同时算出来。novelty 在 mask 设计, 不在新算法。"full mask training" 和 "trust AR/Diff" 这两个小 trick 加起来让方法 surprisingly 工程友好。

最让我印象深刻的 ablation 是 Table 4: **TiDAR 是唯一一个 T/NFE 越高、质量不降反升的方法**。这跟所有现有 dLM 的 trade-off 完全反过来 — 这是 rejection sampling 给的质量 floor, 加上并行 drafting 给的 throughput ceiling。

但 paper 的 limitation 也诚实: 大 batch、long context、训练数据规模, 都是后续工作。Block Diffusion 已经在 large batch / KV cache 做了较多探索, TiDAR 需要补这一课。

---

## 如果你想在 llm.c / nanoGPT 上 prototype

最低成本路径:
1. 在 AR transformer 上加一个 attention mask 变体 (training 时前半 causal + 后半 block-bidirectional)
2. Loss = NTP loss + diffusion mask loss, 等权
3. Inference 时 draft 一批 mask token, 跟 AR logit 做 reject sample
4. 用 FlexAttention 写 mask, 不要自己写 kernel

工程上完全可以一天 prototype, 这点很友好。

---

## References (Web Links)

- **LLaDA**: https://arxiv.org/abs/2502.09992
- **Dream**: https://arxiv.org/abs/2508.15487
- **Block Diffusion**: https://arxiv.org/abs/2503.09573
- **EAGLE-3**: https://arxiv.org/abs/2503.01840
- **EAGLE**: https://arxiv.org/abs/2401.15077
- **Medusa**: https://arxiv.org/abs/2401.10774
- **APD**: https://arxiv.org/abs/2506.00413
- **Fast-dLLM**: https://arxiv.org/abs/2506.00413
- **DeepSeek-V3 (MTP)**: https://arxiv.org/abs/2412.19437
- **Speculative Decoding (Leviathan)**: https://arxiv.org/abs/2302.01318
- **Set Block Decoding**: https://arxiv.org/abs/2509.04185
- **MDLM**: https://arxiv.org/abs/2406.07524
- **FlashAttention-2**: https://arxiv.org/abs/2307.08691
- **FlexAttention**: https://arxiv.org/abs/2412.05496
- **Qwen2.5**: https://arxiv.org/abs/2412.15115
- **Qwen3**: https://arxiv.org/abs/2505.09388
- **Apple MTP**: https://arxiv.org/abs/2502.01423
- **Theoretical analysis of dLM**: https://arxiv.org/abs/2502.09622

如果你想我再展开某一块 — 比如 rejection sampling 的严格推导、PyTorch 伪代码实现、或者跟 EAGLE 的 acceptance rate 详比, 随时叫。

---

# TiDAR: Think in Diffusion, Talk in Autoregression — 深度解析

Andrej, 这篇 paper 直击一个你可能在 nanoGPT / llm.c 训练时反复纠结的核心痛点:**AR decoding 在 batch size=1 时是 memory-bound 的,GPU 算力浪费严重;而 diffusion LLM 虽然能并行 decode,但 parallel decoding 越多质量越烂**。NVIDIA 这帮人提出 TiDAR,把两边的优势在**一个 forward pass** 里缝合起来,同时拿到 5.91× 的吞吐和接近 AR 的质量。我下面按 build-intuition 的方式拆给你看。

---

## 1. 核心矛盾:为什么会有这条 paper 存在

### 1.1 AR 是 memory-bound 的物理事实

图 1 给的 profiling 是整篇文章的物理基础。Qwen3-32B 在 H100 上 batch=1, FlashAttention 2 下,**forward 的 latency 在 token 数量增大到某个阈值之前几乎不变**。原因是这个区间 latency 由 HBM 加载 weights + KV cache 的带宽决定,而非 FLOPs。

论文把这个写成下面这个简单命题:

$$x_{t+1} := F([x_{<t-1}; x_t]) \quad \text{(AR: 1 token / step)}$$

$$x_{t+1}, \dots, x_{t+k+1} := F([x_{<t}; m_{t+1}, \dots, m_{t+k+1}]) \quad \text{(Diffusion: k tokens / step)}$$

只要 $k$ 足够小,两个 $F(\cdot)$ 的 wall-clock 几乎相同。中间的额外 token slot $(m_{t+2}, \dots, m_{t+k+1})$ 就是 paper 反复强调的 **"free token slots"**——把它们塞进 forward 几乎零延迟开销,在进入 compute-bound 区之前都是"白送"的。

> 直觉:AR 推理时 90% 的等待是 GPU 在发呆等 HBM。任何能在这段空档里"白嫖"算力的方法,本质上都是 throughput 的免费午餐。Medusa、EAGLE、MTP 都是这个 motivation,只是它们仍然是 sequential drafting-verification,TiDAR 把它推到极致——parallel 且单 pass。

### 1.2 Diffusion LLM 的"并行惩罚"

Diffusion 那边的建模写成:

$$p_{\text{Diff}}(\cdot; \theta) = \mathbb{E}_{\tilde{\mathbf{x}} \sim q(\cdot|\mathbf{x})} \prod_i p_\theta^i(x_i | \tilde{\mathbf{x}})$$

变量解释:
- $\tilde{\mathbf{x}}$ 是从 corruption distribution $q(\cdot|\mathbf{x})$ 采样得到的"加噪"序列(训练时随机 mask 一部分位置)
- $p_\theta^i(x_i|\tilde{\mathbf{x}})$ 是 transformer 在位置 $i$ 给出的 clean token 分布
- $\prod_i$ 表示各位置独立预测,这本身就是 diffusion 的 marginal 形式

一旦决定一次 decode $K$ 个 token,分布退化成"独立 marginal 乘积":

$$p_{\text{Diff\_Independent\_K}}(\cdot; \theta) = \mathbb{E}_{\tilde{\mathbf{x}} \sim q(\cdot|\mathbf{x})} \prod_{\substack{s.t. \tilde{\mathbf{x}}_{i \in K} = [m]}} p_\theta^i(x_t^i | \tilde{\mathbf{x}})$$

这里的下标 $i \in K$ 表示只对要并行 decode 的 $K$ 个 mask 位置取 marginal。问题就在这:这 $K$ 个 token 互相**不知道对方会 decode 成什么**,各自独立从 marginal 里采样,序列级 coherence 崩了。APD [17] 报告 Dream-7B 在 GSM8K 上,从 1 token/step 增到 2 token/step,准确率掉 10%。

### 1.3 AR 的 chain factorization 是质量上限

$$p_{\text{AR}}(\cdot; \boldsymbol{\theta}) = \prod_i p_\theta^i(x_i | \mathbf{x}_{<i}; \boldsymbol{\theta})$$

每个 token 都条件化于**所有之前已经确定的 token**,这就是 chain rule,严格对应语言建模的因果关系,所以质量上限最高。

TiDAR 的核心赌注:**用 $p_{\text{Diff}}$ 的并行 marginal 做 drafting(便宜),用 $p_{\text{AR}}$ 的 chain factorization 做 verification/sampling(保质量)**,且在同一个 forward pass 完成。

---

## 2. Architecture: 一张 attention mask 讲完一切

### 2.1 序列分段与 mask 设计

Figure 2 / Figure 3 是这篇 paper 的灵魂。每一步 forward 时,序列被切成三段:

| 段 | 内容 | Attention 模式 | KV cache |
|---|---|---|---|
| Section A | Prefix (已确认) | Causal | 持久复用 |
| Section B | 上一步的 draft (待 verify) | Causal with A | 接受则保留,拒绝则 evict |
| Section C | 下一步的 pre-draft mask | Block-bidirectional with A+B | 用完即弃 |

训练时,序列形式是 $[x_1, \dots, x_S; m_1, \dots, m_S]$ — 把原 sequence 复制一份,后一半全置 mask token。前半 causal self-attention,后半 block-bidirectional + 与前半 causal cross-attend。

跟 Block Diffusion [12] 对比的关键差别:**Block Diffusion 是 intra-block bidirectional + inter-block causal** — 每个 block 内部都双向;TiDAR 只保留**最后一个 block(解码块)是 bidirectional**,前面所有 prefix 都是纯 causal。

这个改动带来两个能力:
1. prefix 段可以计算标准 NTP loss(label shift by 1),因为不存在 intra-prefix 的双向信息泄漏
2. inference 时 prefix 走严格 causal,可以直接接 AR 的 chain factorization 做 rejection sampling

### 2.2 训练目标

$$\mathcal{L}_{\text{TiDAR}}(\theta) = \frac{1}{1+\alpha} \bigg( \sum_{i=1}^{S-1} \frac{\alpha}{S-1} \mathcal{L}_{AR}(x_i, x_{i+1}; \theta) + \sum_{i=1}^{S-1} \frac{1}{S-1} \mathcal{L}_{Diff}([mask], x_i; \theta) \bigg)$$

变量含义:
- $\alpha \in [0,1]$ 是 loss balancing factor,论文默认 $\alpha=1$ 即等权
- $S$ 是 input sequence 长度(注意:diffusion 段长度也是 $S$,所以总 token 数是 $2S$)
- $\mathcal{L}_{AR}(x_i, x_{i+1};\theta)$ 是位置 $i$ 的 logit 对 $x_{i+1}$ 计算 cross-entropy(label shift 1)
- $\mathcal{L}_{Diff}([mask], x_i;\theta)$ 是 mask 位置的 logit 对 $x_i$ 计算 cross-entropy(无 shift,位置对齐)

**为什么 diffusion 段无 shift?** 因为 bidirectional attention 下,mask 位置 $i$ 的 logit 就是预测原序列位置 $i$ 的 token,不需要 shift。这点跟 AR 部分形成有意思的对比——同一个 transformer 在不同 mask 段同时承担两种 label alignment。

### 2.3 "Full Mask" 策略 — 这是个小但关键的 trick

传统 diffusion LM 训练时随机 mask 一部分 token(比如 ~50%),loss 只在 masked 位置算。TiDAR 直接把整个 diffusion section 全部置 mask:

三个好处:
1. **Loss 信号密集**:每个 mask 位置都贡献一个 loss term,不再是稀疏的
2. **Loss balancing trivial**:AR loss 项数 = $S-1$,Diffusion loss 项数 = $S-1$,直接 $\alpha$ 加权即可,不用考虑每次 sample 的 mask 比例不同
3. **Train-test consistency**:inference 时反正也是 one-step diffusion(没有 iterative denoising),全是 mask,跟训练 100% 对齐

Table 5 的 ablation 显示这个改动让 HumanEval Avg 从 32.62% → 38.42% (draft=4)、33.85% → 39.94% (draft=8),GSM8k 也涨 ~1pt,同时 T/NFE 略升。**这是个零成本的提升,因为本来就要送 mask token 进 forward**。

---

## 3. Inference: Self-Speculative Generation 的精巧之处

### 3.1 一次 forward 同时做三件事

每个 decoding step,单次 forward 同时计算:

1. **Verify 上一步的 drafts** — 通过 causal attention 在 section B 算出每个位置的 AR distribution $p_{AR}(\cdot|x_{<i})$,然后用 rejection sampling 跟 draft token 比对
2. **Pre-draft 下一步** — section C 的 mask token 通过 block-bidirectional attention 算 $p_{Diff}$ 的 marginal,直接采样
3. **基于所有 verify outcomes 的 pre-draft** — 这是关键:由于 rejection sampling 的 accept length 不确定,需要在 mask section 对**每个可能的 accept prefix** 都准备好对应的 pre-draft

最后这个"基于所有 outcomes"的设计,灵感来自 Apple MTP [20]。简单说,假设上一步 draft 了 $[d_1, d_2, d_3]$,rejection sampling 可能 accept 0/1/2/3 个,那么下一步 pre-draft 要针对这 4 种 prefix 状态都准备一份。在 paper 的实现里,这些 pre-draft 都在同一个 block-bidirectional section 内,因为 mask 位置彼此独立,自然分开。

### 3.2 Trust AR vs Trust Diff

由于 diffusion 段无 label shift,但 AR 段有 label shift,**位置 $i$ 在两个段都会预测同一个 token**:
- Causal section 位置 $i$ 预测 $x_{i+1}$ (label shifted)
- Mask section 位置 $i$ 预测 $x_i$ (aligned)

这俩其实预测的是同一个位置(只是 mask 段的位置编号偏移了),所以可以 mix logits:

$$\text{logits}_i^{\text{mixed}} = \beta \cdot \text{logits}_i^{ar} + (1-\beta) \cdot \text{logits}_i^{diff}$$

$$\text{sampled} = \arg\max_{i \in |V|} \{ \text{logits}_i^{\text{mixed}} \}$$

变量: $\beta \in [0,1]$ 是 trust factor,$|V|$ 是 vocab 大小。$\beta=1$ 完全信 AR,$\beta=0$ 完全信 diffusion。

Figure 6 的 ablation 显示 $\beta$ 在 $[0, 1]$ 区间扫,质量几乎不变(在 1.5B 上)— 说明 model 训练得到两个 distribution 已经高度对齐。但 8B 上 Trust Diff 在 GSM8k 反而略高(80.44% vs 79.83% Trust AR,Table 2),暗示大模型上 diffusion 的 marginal 反而更全局一致。

> 直觉:这个 ablation 说明 rejection sampling 才是质量保证的真正来源,而非 AR logit 本身。draft quality 决定 acceptance rate,但 sampling correctness 由 rejection sampling 的数学保证决定。

### 3.3 KV cache 管理

Exact KV cache 在 diffusion LM 一直是个痛点 — bidirectional attention 让 cache 失去意义。Block Diffusion 通过 inter-block causal 部分解决。TiDAR 进一步:
- Section A 的 KV cache:永久保留
- Section B 的 KV cache:接受则保留进 A,拒绝则 evict
- Section C 的 KV cache:用完即弃(下一步不会再作为 prefix)

**关键:不做任何 recompute**。Block Diffusion 因为 inter-block 是 causal,前 block 接受后 block 还要重新走 attention,有 recompute 开销;Fast-dLLM [13] 用 approximate cache 有质量损失;d-KV Cache [27] 延迟 cache 也有问题。TiDAR 因为 prefix 段全程 causal,完美支持 exact cache。

### 3.4 Attention mask 的复用 trick

Figure 3 右侧 + Appendix Figure 7:inference 时每个 sample 的 prompt 长度不同,但 attention mask 结构相同。做法是 initialization 时预分配一个 $(\text{max\_seq\_len} + \text{block\_size}) \times (\text{max\_seq\_len} + \text{block\_size})$ 的大 mask,每个 sample 通过**重排序列顺序 + slice 这个大 mask** 来复用,避免每个 sample 重新构造 mask。这是 FlexAttention [35] 友好的写法。

---

## 4. 跟其他范式的横纵对比

### 4.1 跟 Speculative Decoding 的对比 (Table 1)

| 框架 | Draft model 与 base 共享? | Draft capacity | Parallel drafting? | Drafting 与 verify 并行? |
|---|---|---|---|---|
| Classic Spec Dec [16] | ✗ (小 draft) | Low | ✗ | ✗ |
| APD [17] | ✗ | High | ✓ | ✗ |
| EAGLE-3 [18] | ✗ (额外 head) | Mid | ✗ | ✗ |
| DeepSeek-V3 MTP [19] | ✗ (额外 layer) | Mid | ✗ | ✓ |
| Apple MTP [20] | ✓ | High | ✗ | ✓ |
| **TiDAR** | **✓** | **High** | **✓** | **✓** |

TiDAR 是唯一同时打勾四个 box 的方法。EAGLE / DeepSeek MTP 的 drafter 是 AR + 单 last token input,无法利用 mask token 的并行计算 — 它们的 drafting 是 sequential 的,只是 cheap。TiDAR drafting 直接在 mask section 上 block-bidirectional,$k$ 个 mask 一次性算完。

### 4.2 跟 Block Diffusion 的对比

Block Diffusion:inter-block causal, intra-block bidirectional。每个 block 内部仍要做 multi-step denoising,且接受前 block 后下一 block 要 recompute。质量问题:within-block 并行 decode 时仍有 independence assumption(虽然 block 内有 bidirectional context,但并行 decode 仍是 marginal)。

TiDAR 把 bidirectional 限制到只最后一个 block,其余 prefix 全 causal,因此:
- prefix 上能算 NTP loss (Block Diffusion 不行,label leakage)
- prefix 上能做 exact KV cache
- 通过 rejection sampling 把 quality 控制权交给 AR distribution,而非依赖 diffusion 的 marginal correctness

### 4.3 跟 Diffusion LLM (Dream, LLaDA) 的对比

LLaDA / Dream 的 likelihood evaluation 需要 Monte Carlo 128 steps 估计,慢且不可比。TiDAR 因为有 AR mode,likelihood 跟 AR 模型完全一样算(Table 3 显示 TiDAR 8B 在 MMLU 76.57%,接近 Qwen3 8B 76.93%,远超 Dream 67.00% / LLaDA 65.86%)。

---

## 5. 实验数据深度解读

### 5.1 生成质量 (Table 2)

8B 规模 TiDAR (Trust AR / Trust Diff):

| 任务 | Qwen3-8B (AR base) | Dream-7B | Block Diff-4B | TiDAR-8B Trust AR | TiDAR-8B Trust Diff |
|---|---|---|---|---|---|
| HumanEval | 64.63% | 54.88% | 56.10% | 55.49% (7.46 T/NFE) | 57.93% (7.30 T/NFE) |
| GSM8k | 81.80% | 77.18% | 82.87% | 79.83% (6.90) | 80.44% (7.07) |
| Avg | 68.09% | 58.74% | 60.27% | 63.90% (8.23) | 65.31% (8.25) |

TiDAR 平均一次 forward 输出 8.25 个 token,vs AR 的 1 token/forward。质量上比 Dream 高 6.5 pt,比 Block Diff 高 5 pt,离 Qwen3-8B base 还差 2.8 pt(主要在 HumanEval,数学任务基本持平甚至略胜)。

> 直觉:TiDAR 在 8B 上相比 AR base 仍有 2-3 pt 的质量 gap,论文承认这是 continual pretraining 50B-150B tokens 不够 — TiDAR 训练时一半梯度去学 diffusion mode 了,知识容量被分摊。更多数据应该能闭合。

### 5.2 吞吐 (Figure 4)

- TiDAR 1.5B: **4.71× vs Qwen2.5-1.5B**
- TiDAR 8B: **5.91× vs Qwen3-8B**
- 对比 EAGLE-3 公开权重:TiDAR 首次让 diffusion-based 方法在效率上**超过** speculative decoding。原因:TiDAR 的 raw acceptance rate (T/NFE) 更高,且 conversion rate (T/NFE → T/s) 更高 — 因为 drafting 和 verify 在同一 forward,没有 sequential 的两阶段开销。

### 5.3 Pareto Frontier (Figure 5)

同 50B 训练 token 下,1.5B 规模:
- AR (Qwen2.5-1.5B base): 1 T/NFE, avg ~50% (task mix)
- Fine-tuned AR: 1 T/NFE, avg ~55%
- Block Diff (threshold 0.8): ~3 T/NFE, avg ~48%
- Block Diff (threshold max, 1 token/step): 1 T/NFE, ~50%
- TiDAR (block 4/8/16): 3-7 T/NFE, avg ~52-53%

TiDAR 在 7× T/NFE 下接近 fine-tuned AR 的质量,这是 Pareto frontier 的大跃迁。

### 5.4 解码策略对比 (Table 4)

1.5B 规模,full mask 模型:

| 策略 | T/NFE | HumanEval Avg | GSM8k |
|---|---|---|---|
| Confidence Max (1 tok) | 1.00 | 36.28% | 53.37% |
| Confidence Max (2 tok) | 2.00 | 21.95% | 41.32% |
| Left-to-right (AR, 1 tok) | 1.00 | 36.28% | 53.37% |
| Left-to-right (AR, 2 tok) | 2.00 | 21.95% | 41.32% |
| Confidence > 0.9 | 2.63 | 32.01% | 51.40% |
| Confidence > 0.6 | 3.81 | 22.56% | 37.60% |
| **TiDAR (4 drafts)** | **3.47** | **38.42%** | **55.87%** |
| **TiDAR (8 drafts)** | **5.49** | **39.94%** | **54.74%** |
| **TiDAR (16 drafts)** | **6.97** | **41.16%** | **53.90%** |

注意这里 TiDAR 在 draft=16 时 T/NFE=6.97 而 HumanEval 不降反升(38.42% → 41.16%),而 confidence-based 方法在 T/NFE 升高时质量急速崩盘。**TiDAR 是唯一在 T/NFE 增加时质量不退反进的方法**。原因是更长的 draft length 让 rejection sampling 有更多 verify 机会,接受的部分越多,越接近 AR distribution 的质量。

---

## 6. 一些值得深挖的细节 / 我觉得可疑或可改进的地方

### 6.1 Sequence length 翻倍的开销

训练时序列变 $2S$,显存翻倍。1.5B 用 4096 max len,实际等于 8192 的 attention computation。8B 还开了 gradient checkpointing。这个对训练成本不友好。Section 5 提到 long context extension 没做,正是这个 doubling 的副作用。一个可能改进:**packing + RoPE 位置 ID 不翻倍**,只让 attention mask 翻倍结构,可能省一半。

### 6.2 Batch size > 1 的故事没讲清

论文说 "we focus on batch size = 1 efficiency benchmarking",承认大 batch 下 free token slots 收益缩小(进入 compute-bound 区)。这点对生产 serving 重要 — 实际 serving 都是 large batch。TiDAR 在 large batch 下的 Pareto 位置需要后续工作。

### 6.3 Rejection sampling 的 math 没展开

paper 里反复提 "rejection sampling guided by $p_{AR}$" 但没给完整的 accept/reject 算法。Speculative decoding 经典版是 Leviathan 2022 [11] 的:

$$P(\text{accept } x) = \min(1, p_{AR}(x)/p_{draft}(x))$$

TiDAR 的版本应该类似,但 $p_{draft}$ 是 diffusion marginal。这个 ratio 在 well-trained model 上接近 1(paper 在 4.4.3 的 ablation 印证),所以 acceptance rate 高。完整推导值得单独写。

### 6.4 Training data 配比

50B token 对 1.5B、150B 对 8B,这个量级对 continual pretraining 算 moderate。Qwen2.5 / Qwen3 base 的原训练 token 数远超这个。TiDAR 的 quality gap 多大程度来自架构,多大程度来自数据量,从 Figure 5 看仍有 2-3 pt gap on fine-tuned AR,paper 说 "需要更多数据"。这个论点如果能上 1T token 训练验证会更有说服力。

### 6.5 Trust Diff 在大模型上更好,为什么?

Table 2 显示 TiDAR 8B Trust Diff > Trust AR 在多数任务上(尤其 GSM8k 80.44 vs 79.83)。这暗示在大模型上 diffusion 的 marginal distribution 反而比 AR 的 chain conditional 更鲁棒(可能因为 AR 在 math 长链上容易累积误差,diffusion 的全局视野更好)。这是个有趣的 hypothesis,值得单独 study。

---

## 7. 我的整体评价

这篇 paper 的核心贡献是 **attention mask 工程上的精确切割**,把 diffusion 的 parallelism 和 AR 的 chain factorization 在一次 forward 里同时算出来。novelty 在 mask 设计,不在新算法。"full mask training" 和 "trust AR/Diff" 这两个小 trick 加起来让方法 surprisingly 工程友好。

最让我印象深刻的 ablation 是 Table 4:**TiDAR 是唯一一个 T/NFE 越高、质量不降反升的方法**。这跟所有现有 dLM 的 trade-off 完全反过来 — 这是 rejection sampling 给的质量 floor,加上并行 drafting 给的 throughput ceiling。

但 paper 的 limitation 也诚实:大 batch、long context、训练数据规模,都是后续工作。Block Diffusion 已经在 large batch / KV cache 做了较多探索,TiDAR 需要补这一课。

如果你在 llm.c 或者 nanoGPT 上想试这个 idea,最低成本的复现路径是:
1. 在 AR transformer 上加一个 attention mask 变体 (training 时前半 causal + 后半 block-bidirectional)
2. Loss = NTP loss + diffusion mask loss,等权
3. Inference 时 draft 一批 mask token,跟 AR logit 做 reject sample
4. 用 FlexAttention 写 mask,不要自己写 kernel

工程上完全可以一天 prototype,这点很友好。

---

## References (Web Links)

相关核心论文的 arXiv 链接:

- **LLaDA (Diffusion LLM baseline)**: https://arxiv.org/abs/2502.09992
- **Dream (Diffusion LLM SOTA)**: https://arxiv.org/abs/2508.15487
- **Block Diffusion (TiDAR 的直接前身)**: https://arxiv.org/abs/2503.09573
- **EAGLE-3 (Speculative decoding SOTA)**: https://arxiv.org/abs/2503.01840
- **EAGLE (原始版)**: https://arxiv.org/abs/2401.15077
- **EAGLE-2**: https://arxiv.org/abs/2406.16858
- **Medusa (multi-head drafting)**: https://arxiv.org/abs/2401.10774
- **APD (Adaptive Parallel Decoding)**: https://arxiv.org/abs/2506.00413
- **Fast-dLLM (Diffusion KV cache)**: https://arxiv.org/abs/2506.00413
- **DeepSeek-V3 (MTP)**: https://arxiv.org/abs/2412.19437
- **Speculative Decoding (Leviathan)**: https://arxiv.org/abs/2302.01318
- **Set Block Decoding (SBD)**: https://arxiv.org/abs/2509.04185
- **MDLM (masked diffusion LM)**: https://arxiv.org/abs/2406.07524
- **FlashAttention-2**: https://arxiv.org/abs/2307.08691
- **FlexAttention**: https://arxiv.org/abs/2412.05496
- **Qwen2.5**: https://arxiv.org/abs/2412.15115
- **Qwen3**: https://arxiv.org/abs/2505.09388
- **Apple MTP**: https://arxiv.org/abs/2502.01423
- **Theoretical benefit/limitation of dLM**: https://arxiv.org/abs/2502.09622
- **d-KV Cache**: https://arxiv.org/abs/2505.0
- **TiDAR 官方项目页**(若 NVIDIA 放出):https://research.nvidia.com/labs/
- **Block Diffusion GitHub**: https://github.com/its-kind-of-a-fun-story/block-diffusion
- **LLaDA GitHub**: https://github.com/ML-GSAI/LLaDA
- **EAGLE GitHub**: https://github.com/SafeAILab/EAGLE
- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness

TiDAR 论文本身截至 2026/08/12 在 arXiv 上似乎还没放出公开版,但从作者信息 (Jingyu Liu @ UChicago + Ce Zhang, Xin Dong @ NVIDIA) 和 references 风格判断,大概率会在近期挂 arXiv。NVIDIA research blog 通常会有 highlight: https://developer.nvidia.com/blog/

如果你想我进一步深入某一块,比如 rejection sampling 的严格推导、attention mask 的 PyTorch 实现伪代码、或者跟 EAGLE 的 acceptance rate 详比,我可以继续展开。
