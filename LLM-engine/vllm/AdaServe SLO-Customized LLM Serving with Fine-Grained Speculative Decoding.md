---
source_pdf: AdaServe SLO-Customized LLM Serving with Fine-Grained Speculative Decoding.pdf
paper_sha256: 0464ca958a82860093b2e30e18713e7616f0c9c87d349e9d00050eff15ee8337
processed_at: '2026-07-18T02:10:58-07:00'
target_folder: LLM-engine/vllm
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AdaServe: SLO-Customized LLM Serving 深入解析

Karpathy 你好, 这篇 AdaServe 论文非常有意思, 让我从底层 hardware 一直到 system design 给你拆解一遍。我先讲整个 motivation 和核心 idea, 然后深入到 algorithm 的数学细节, 再讲 system 实现和 evaluation 里的关键 numbers。

## 1. 核心问题: 为什么 continuous batching 撞到 SLO customization 就失效

现有的 LLM serving 系统 (vLLM [https://vllm.ai/], Sarathi-Serve [https://arxiv.org/abs/2403.02310], Orca [https://arxiv.org/abs/2108.12409], DistServe [https://arxiv.org/abs/2401.09670], Splitwise [https://arxiv.org/abs/2311.18677]) 基本都依赖 **continuous batching**: 每个 decoding iteration 开始时, 把已完成请求移出, 把新请求加入 batch。这种 iteration-level batching 让 batch 内所有请求共享相同的 per-iteration latency, 于是每个请求的 TPOT (time per output token) 也都相同。

问题来了, 不同 application 的 SLO 截然不同:
- **Code completion / LLM agent**: 必须 < 12.5 ms/token, 用户在 typing pause 期间要看到结果
- **Search integration**: 30 ms/token (对应 100 token / 3s, 因为 ~50% 用户会在 3s 后 abandon, 参考 Google's consumer insights)
- **Chatbot**: 100 ms/token (人类阅读速度 ~10 token/s, 参考 Brysbaert 2019 [https://www.sciencedirect.com/science/article/pii/S0749596X19300786])

Continuous batching 在同一个 iteration 把所有请求的 decoding 捆在一起, 你没法在同一 iteration 内给 r_0 加速、给 r_1 减速。一种 naive 的 workaround 是 **SepPipe** (separate pipeline), 给每个 SLO category 单独一个 model replica。但这样 model utilization 会大幅下降, 论文 ablation 里 SepPipe 是最差的 baseline 之一。

AdaServe 的核心 insight: **Speculative decoding 天然就是一个 fine-grained budget allocator**。每个请求可以在 draft token tree 上分到不同数量的 node, 用 budget 分配来满足 SLO。这是一种把 hardware resource 在 iteration 内做 sub-batch 切分的方法, 同时还保留了 continuous batching 的高 throughput。

## 2. Hardware Budget 的精确定义

论文给了一个很关键的公式, 把 GPU 的 compute 和 bandwidth 拉到同一个量纲上:

$$B = \frac{\text{BF16 Compute (TFLOPS)}}{\text{HBM Bandwidth (TB/s)}}$$

变量解释:
- B = hardware budget, 单次 forward pass 的理想 token 数
- BF16 Compute = GPU 在 BF16 精度下的 peak FLOPS
- HBM Bandwidth = GPU HBM 内存带宽

为什么这个公式成立? Transformer decoding 是 memory-bound 的: 每个 token 的 decoding 都要 load 所有 parameter。FP16 下, 每个 parameter 需要 2 FLOPs (一次乘 + 一次加), 同时从 VRAM 加载 2 bytes (FP16)。所以 arithmetic intensity = 2 FLOPs / 2 bytes = 1 FLOP/byte。GPU 的 ridge point (从 memory-bound 跨入 compute-bound) 大约就是 BF16 FLOPS / HBM Bandwidth。

| Hardware | BF16 Compute | HBM Bandwidth | Budget B |
|----------|-------------|--------------|---------|
| A100     | 312 TFLOPS  | 2.0 TB/s     | 156     |
| H100     | 1979 TFLOPS | 3.4 TB/s     | 582     |

这个 B 是 AdaServe 整个调度的基础。超过 B 之后 forward pass latency 会快速上升, 因为从 memory-bound 切到 compute-bound, throughput 反而下降。

直觉: **Budget 就是 "GPU 在 memory-bound regime 下能并行处理多少 token 的上限"**。Continuous batching + speculative decoding 都是把多个 token 的 computation fuse 到一个 forward pass 里, 提高 arithmetic intensity, 让 GPU 接近 ridge point。

## 3. 问题形式化: 带约束的期望最大化

给定 n 个 request {r_1, ..., r_n} 和 budget B, 构造 n 个 token tree {T_1, ..., T_n}:

**Objective (期望接受 token 数最大化):**
$$E\Big[\sum_{i=1}^{n} \text{acc}(T_i)\Big] = \sum_{i=1}^{n} E\big[\text{acc}(T_i)\big]$$

变量解释:
- acc(T_i) = random variable, LLM 验证 T_i 时实际接受的 token 数
- E[acc(T_i)] = 期望值, 因为 speculation 时刻不知道哪些会被接受

**Constraint 1 (Budget):**
$$\sum_{i=1}^{n} |T_i| \leq B \tag{1}$$

|T_i| = T_i 的 node 数 (包括 root)。

**Constraint 2 (TPOT):**
$$\frac{l_i + t^{\text{spec}}}{o_i + \text{acc}(T_i)} \leq t_i^{\text{TPOT}}, \quad \forall i = 1, \dots, n \tag{2}$$

变量含义:
- l_i = r_i 从第一个 decoding step 起累计的 latency (ms)
- o_i = r_i 已 decoded 的 token 数
- t^spec = 当前 decoding iteration 的 latency (small model speculation + LLM verification)
- t_i^TPOT = r_i 的 TPOT SLO

Constraint 2 的 intuition: 当前 iteration 结束后, r_i 的平均 per-token latency = 总 latency / 总 token 数, 必须不超过 SLO。

简化为:
$$\text{acc}(T_i) \geq A(r_i), \quad A(r_i) = \frac{l_i + t^{\text{spec}}}{t_i^{\text{TPOT}}} - o_i \tag{3}$$

A(r_i) 就是 "当前 iteration 至少要接受多少 token 才能满足 SLO"。这个数会随 l_i 增长而增长 — 如果一个 request 已经 lag 了, 它需要的 acceptance 就更紧迫。

## 4. Theorem 3.1 — 这是整个 algorithm 的 backbone

$$E[\text{acc}(T)] = \sum_{v \in T} f(v) \tag{4}$$

变量解释:
- v = token tree 里的某个 node
- f(v) = node v 的 path probability, 即 "LLM 接受从 root 到 v 这条路径" 的条件概率
- 求和遍历 T 里所有 node

这个 decomposition 让期望接受数变成 **node-level 求和**, 这是 algorithm 1 能用贪心策略的关键。证明在 Leviathan 2022 [https://arxiv.org/abs/2211.17192] 和 Sun 2024 [https://arxiv.org/abs/2402.06648] 类似的 SD 文献里。

注意, 这隐含一个重要性质: 一棵 tree 接受的 token 数等于它所有 node 的 f(v) 之和。这让约束 (3) 变成:

$$\sum_{v \in T_i} f(v) \geq A(r_i), \quad \forall i = 1, \dots, n \tag{5}$$

而 objective 变成:

$$\sum_{i=1}^{n} E[\text{acc}(T_i)] = \sum_{v \in \bigcup_{i=1}^n T_i} f(v) \tag{6}$$

## 5. Algorithm 1: 全局最优的贪心

这个 algorithm 看起来简单, 但有几个深意:

**Step 1 (SLO 阶段, line 7-15):**
```
for i = 1...n:
    while n_acc[i] < A(r_i):
        v = GetTop(T_inf(r_i) - S_added)  # 在 r_i 自己的 tree 里取未选过的最大 f(v)
        T_i.Add(v)
        n_acc[i] += f(v)
        S_added.Add(v)
        B -= 1
```

每轮选 r_i 自身 token tree 里 f(v) 最高的未选 node, 累加到满足 A(r_i)。

**Step 2 (Throughput 阶段, line 16-21):**
```
while B >= 0:
    v = GetTop(∪ T_inf(r_i) - S_added)  # 跨所有 request, 取全局最大 f(v)
    i = GetReqIdx(v)
    T_i.Add(v)
    S_added.Add(v)
    B -= 1
```

剩余 budget 全局贪心, 把所有 candidate node 当一个池子按 f(v) 大小选。

**为什么最优?**

论文 Appendix A.2 给出形式化证明, 核心是两个 lemma:

**Lemma A.1 (Minimality in Threshold Attainment):** 给定 threshold τ, 贪心选到刚好满足 τ 所需 node 数 n 是最少的。反证: 若有更小的 n' < n 集合满足 τ, 那 n' 个 node 的 f(v) 之和 >= τ, 但贪心选前 n-1 个时是按 f 降序的, 前 n-1 个都不够, 任何 n' 个更不可能。

**Lemma A.2 (Maximality under Fixed Budget):** 给定 budget b, 贪心选 b 个 node 最大化 f(v) 之和。反证: 若 V' 有更大 sum, 则 V \ V' 里的 node 一定比 V' \ V 里的 f 大, swap 之后 V 的 sum 至少不变, 矛盾。

合起来: Step 1 保证 SLO 约束下每 request 用最少 node (省下最多 budget 给 Step 2), Step 2 在剩余 budget 上全局贪心, 所以全局最优。

**Connectivity 保证 (Appendix A.1):** 关键性质是 f(parent(v)) > f(v), 因为 path probability 是 conditional 概率的乘积, 每个条件概率 < 1。所以贪心选 v 时它的 parent 一定已经选过了, 整个选集一定 connectivity valid。

## 6. 两个核心 challenge

### Challenge 1: f(v) 未知

Algorithm 1 假设每个 node 的 path probability 已知, 但实际 SD 中 f(v) 要等 LLM 验证才能算出来。

**Solution: 用 draft model logits 近似**

$$\prod_{u \in \text{Path}(v)} M_q\big(u \mid X, \text{Path}(u.\text{parent})\big) \approx f(v) \tag{7}$$

变量解释:
- M_q = draft model (small model), 输入 token sequence 输出 vocabulary 上的 probability distribution
- X = 当前 context (prompt + 已生成 token)
- Path(v) = root 到 v 的 token 序列
- Path(u.parent) = root 到 u 的 parent 的 token 序列
- M_q(u | X, Path(u.parent)) = 给定 context 和 prefix, draft model 输出 token u 的概率

直觉: 如果 draft model 和 LLM 训练数据 / procedure 相似, 它们的 conditional distribution 也接近, 那 draft model 的 path probability 就是 LLM acceptance probability 的 good proxy。Distillation 训练的 draft model (如 EAGLE [https://arxiv.org/abs/2401.15077], EAGLE-2 [https://arxiv.org/abs/2406.16858], DistillSpec [https://arxiv.org/abs/2310.08461]) 让 logits 直接 align, 这个近似更准确。

### Challenge 2: Speculation overhead 太高

如果直接按 Algorithm 1 做: 每次 GetTop 后要做一次 draft model decode, 因为要 expand 新 child。B - n 个非 root node 要 decode, 共 B - n 次 decode。A100 B=156, 如果 n=4, 就是 152 次 small model forward — 这远比一次 large model forward 还慢。

**Solution: 解耦 speculation 和 selection**

**Theorem 4.1 (Bounding Box):** 设 Algorithm 1 最优 tree T_opt 的最深节点 depth = D_opt, 那么 T_opt 一定被一个 D_opt-step beam search (beam width B) 构造的 candidate tree T_cand 包含。

证明思路: beam search 在每步保留 top-B 个 node (按累积 path probability), 而 greedy 也是按 f(v) 降序选, 所以 greedy 选的 node 都会在 beam search 的 top-B 内, 一直延伸到深度 D_opt。

这个 theorem 的 implication 巨大: 只要 D_opt 次 small model decoding 就能 cover 最优 tree, 而不是 B-n 次。论文指出实证上 D_opt << B - n。

进一步 bound: 假设最深的 request 是 r_j, D_opt = D(T_opt(r_j)) ≤ |T_opt(r_j)| - 1 ≤ Σ|T_opt(r_i)| - n = B - n, 等号成立当且仅当所有 tree 退化为 chain, 实际中不会发生。

## 7. Algorithm 2: 实用的四阶段 pipeline

这是 AdaServe 真正用到的算法, Algorithm 1 的 practical adaptation。

### Stage 1: Speculation (beam search)

每个 request r_i 用 beam search 构造 candidate tree T_cand(r_i):
- depth d, beam width w
- 第一步: draft model 处理 root (即 last generated token), 输出 |V| 个 candidate, 选 top-w
- 第 2 到 d 步: draft model 并行处理前一步的 n*w 个 token, 每个输出 |V| 个, 每个 request 从 w*|V| 中选 top-w 加入 candidate tree

最终每个 request 的 T_cand 有 1 + w*(d-1) 个 node (root + 每层 w 个)。

### Stage 2: SLO-customized token selection

定义:
$$A_{\text{cap}}(r_i) = \min(A(r_i), d+1)$$

因为一次 iteration 最多接受 d+1 个 token (tree 深度 d + root), 如果 A(r_i) > d+1, 即使所有 draft token 都接受也满足不了, 只能 best effort。

算法:
1. 把 requests 按 A(r_i) 降序排 (越紧急的越先服务)
2. 对每个 r_i, 从 T_cand(r_i) 里按近似 f(v) 降序加 node 到 T_i, 直到 Σ f(v) ≥ A_cap(r_i) 或达到 n_max 限制
3. n_max 防止单个 request 把 budget 耗尽在低 f(v) node 上

n_max 是一个非常重要的工程细节 — 论文没详细说怎么设, 但直觉上应该和 B/n 同量级。

### Stage 3: Throughput-optimized token selection

剩余 budget 全局贪心: 从所有 ∪ T_cand(r_i) 里取未选过的最大近似 f(v) node, 加入对应 T_i, 直到 budget 用完。

注意: Stage 2 保证 SLO, Stage 3 把剩余 budget 填满 (最大化 Σ f(v) = 最大化 expected accepted tokens = 最大化 throughput)。两个 stage 加起来等价于 Algorithm 1 的最优解, 只要 candidate tree 足够大覆盖 T_opt。

### Stage 4: Verification

构造好的 token tree 交给 LLM 做 tree-based verification (Medusa-style [https://arxiv.org/abs/2401.10774] 或 SpecInfer-style [https://arxiv.org/abs/2305.09781])。LLM 在一次 forward pass 里 verify 所有 candidate node, 然后按 acceptance 决定接受多少 token。

## 8. System Design 细节

### 8.1 架构 (Figure 3)

- **Request manager**: 维护 active request pool + SLO-customized scheduler
- **Execution engine**: 在 GPU 上跑 small + large model

每个 iteration 流程:
1. Scheduler 取出所有 active requests
2. 调 engine 跑 small model d 次 (speculation)
3. 跑两阶段 selection (CPU 上, 几乎不耗时)
4. 把 draft token tree 给 LLM verification
5. LLM 返回 logits, scheduler 处理得到 accepted token, 写回 request pool

### 8.2 Dynamic d 和 w

$$d = \max\big(1, \min(D_{\max}, \lceil B/n \rceil)\big) \tag{8}$$

$$w = \max\big(1, \min(W_{\max}, \lfloor B/n \rfloor)\big) \tag{9}$$

变量:
- D_max, W_max = preset upper bound for d, w
- n = 当前 active request 数
- B = hardware budget

Intuition:
- n 大时 (高 load), B/n 小, d 和 w 都小, 避免 candidate tree 浪费计算
- n 小时 (低 load), B/n 大, d 和 w 可以放大, 充分利用 budget 提高 acceptance

注意 w * n < B 这个 invariant (只要 n < B), 保证 small model decoding 始终在 memory-bound regime, 不会进入 compute-bound。

### 8.3 CUDAGraph 优化

CUDAGraph [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs] 可以 capture 一系列 CUDA kernel 的执行图, 避免每次 kernel launch 的 overhead。

AdaServe 利用了两个 consistency:
1. **同一 iteration 内**: 第 2 到 d 次 decoding 的形状完全一样 (每 request decode w 个 token, 共 n 个 request)
2. **跨 iteration**: 只要 n 不变, decode 操作完全一致

所以 AdaServe 按 (n, w) 为 key 缓存 CUDAGraph, 多次复用。这对 small model 尤其重要, 因为 small model kernel launch overhead 占比大。

## 9. Evaluation 关键数据

### 9.1 Setup

- Hardware: 4x NVIDIA A100 80GB, NVLink
- Models: Llama-2-7b/13b/70b-chat, draft: Felladrin/Llama-160M-Chat-v1 [https://huggingface.co/Felladrin/Llama-160M-Chat-v1]
- 精度: FP16
- 基线: vLLM, Sarathi-Serve, SpecInfer (batched), IncrDec (incremental decoding)
- 实现: 基于 FlexFlow Serve [https://github.com/flexflow/FlexFlow], 集成 FlashInfer [https://github.com/flashinfer-ai/flashinfer] 的 batched prefill kernel + TensorRT-LLM [https://github.com/NVIDIA/TensorRT-LLM] 的 AllReduce

### 9.2 Workload

三类 request (Table 2):
| Model | Cat 1 (urgent) | Cat 2 (search) | Cat 3 (chatbot) |
|-------|---------------|---------------|-----------------|
| 7b    | baseline (12.5ms) | 30ms | 100ms |
| 13b   | baseline       | 30ms | 100ms |
| 70b   | baseline       | 60ms | 100ms |

混合 60% Cat 1 + 20% Cat 2 + 20% Cat 3 (peak urgent load scenario)。

数据集: Alpaca [https://github.com/tatsu-lab/stanford_alpaca] + ArXiv Summarization [https://arxiv.org/abs/1804.05285], 用 SplitWise [https://arxiv.org/abs/2311.18677] 的 trace 来模拟 arrival。

### 9.3 主要结果 (Figure 4, 5)

**SLO attainment 提升:**
- Llama-2-7b-chat: 最高 1.63x
- Llama-2-13b-chat: 最高 1.66x
- Llama-2-70b-chat: 最高 1.73x (对应论文标题的 73%)

**Goodput 提升:**
- 7b: 1.51x
- 13b: 1.73x
- 70b: 1.74x (对应论文标题的 74%)

Baseline 表现分析:
- **vLLM, IncrDec**: 新 prefill 会 preempt 当前 decoding, 紧迫 SLO 满足率随 RPS 上升快速下降
- **Sarathi-Serve**: chunked prefill 避免 preemption, 但每个 forward pass 略慢
- **SpecInfer**: 固定 tree 结构 (⟨1,1,3,1,1,1,1,1⟩), 低 load 时表现好, 高 load 时浪费 compute — 因为它没有 budget awareness

### 9.4 Stringent request 比例变化 (Figure 6)

固定 RPS, 提高 Cat 1 比例:
- AdaServe 几乎 100% SLO attainment 不变
- Baseline 随 Cat 1 比例上升下降明显
- 因为 baseline 只能靠小 batch 满足 tight SLO, 但 RPS 固定时 batch size 大致不变, urgent request 增多就崩了

### 9.5 Ablation: 三个 baseline 调度策略 (Figure 7)

- **FCFS** (first-come-first-serve, 带 preemption)
- **STTA** (shortest-time-to-attain, 优先接近 SLO violation 的 request)
- **SepPipe** (每个 category 一个 pipeline)

AdaServe 一致胜出。SepPipe 最差, 因为 hardware 没共享。

### 9.6 Ablation: tree construction (Figure 8)

- **EqualGreedy**: 均分 budget, 每 request 内部贪心 (类似 EAGLE-2 [https://arxiv.org/abs/2406.16858] 但 budget-aware)
- **GlobalGreedy**: AdaServe 去掉 SLO-customized selection, 直接全局贪心

AdaServe 仍然最好。GlobalGreedy 比 EqualGreedy 略好, 但都比 AdaServe 差, 因为没有 per-request SLO 优先级。

### 9.7 SLO stricter than baseline latency (Figure 10)

这组实验很有意思: 全 urgent request, 固定 RPS=1.0 (70b), 逐步降低 TPOT SLO 到 baseline latency 的 0.6x:
- 0.8x baseline: 95% SLO attainment
- 0.6x baseline: ~60% SLO attainment

这说明 AdaServe 通过 speculative decoding 能突破 continuous batching 的下限 — 因为 continuous batching 的最低 TPOT 就是单 iteration 串行 forward 的 latency, 而 SD 让一个 iteration 产出多个 token, 实际 TPOT = iteration latency / acceptance count, 可以低于 baseline。

## 10. 与 Related Work 的对比

| 系统 | 关键 idea | vs AdaServe |
|------|---------|-------------|
| **Orca** [https://arxiv.org/abs/2108.12409] | Continuous batching | 无 SLO customization, 同 batch 同 TPOT |
| **vLLM** [https://arxiv.org/abs/2305.09781] | PagedAttention 减少 fragmentation | Complementary (memory mgmt) |
| **Sarathi-Serve** [https://arxiv.org/abs/2403.02310] | Chunked prefill | 优化 prefill/decode 融合, 无 SLO |
| **DistServe** [https://arxiv.org/abs/2401.09670] | Prefill/decode 分离到不同 node | 互补, 也可叠加 SD |
| **Splitwise** [https://arxiv.org/abs/2311.18677] | Phase splitting | 互补 |
| **SpecInfer** [https://arxiv.org/abs/2305.09781] | Tree-based SD + verification | 固定 tree, 无 SLO/budget awareness |
| **Medusa** [https://arxiv.org/abs/2401.10774] | Multi-head SD | 固定 tree, 无 batch 感知 |
| **EAGLE / EAGLE-2** [https://arxiv.org/abs/2406.16858] | Dynamic tree based on context, draft model logits | 有 context-aware tree, 无 SLO/budget |
| **Sequoia** [https://arxiv.org/abs/2402.12374] | Hardware-aware tree via DP | 有 hardware awareness, 无 SLO |
| **SmartSpec** [https://arxiv.org/abs/2408.07678] | Adaptive draft length by batch size | Goodput 优化, 无 SLO customization |
| **MagicDec** [https://arxiv.org/abs/2408.11049] | Long-context SD + sparse KV | Long-context 专项, 互补 |
| **Lookahead Decoding** [https://arxiv.org/abs/2402.01857] | Jacobi iteration SD | 无 draft model, 无 SLO |

AdaServe 是第一个把 **SLO-customization** 和 **batched SD** 联系起来的系统。它本质上是把 SD 的 "draft token tree" 当成一种 **per-request budget allocation 机制**, 这个视角之前没人提出过。

## 11. Build Intuition: 为什么这个工作成立

我想强调几个让 AdaServe 成立的 **结构化条件**:

**条件 1: SD 的 f(v) 是 additive (Theorem 3.1)**
这让 per-request 约束 (acc(T_i) ≥ A(r_i)) 变成 node-level sum, 而 sum 满足 "可选 node 的贪心" 这种结构 (Lemma A.1/A.2)。如果 acc(T) 是非线性的 (比如 max), 贪心就不行。

**条件 2: Draft model logits 是 f(v) 的 good proxy (Challenge 1)**
这归功于 distillation。EAGLE 已经证明这一点, AdaServe 复用了。

**条件 3: T_opt 有 depth bound (Theorem 4.1)**
这让 speculation overhead 从 O(B-n) 降到 O(D_opt), D_opt << B-n。否则 small model decode 时间会 dominate。

**条件 4: GPU 的 arithmetic intensity profile 是 piecewise 的**
Budget B 就是 ridge point。在 B 内, forward pass latency 几乎不随 batch size 变化; 超过 B, latency 快速上升。这让 "budget allocation" 是有意义的 — 你想让 total nodes 尽量接近但不超 B。

这四个条件叠加, 才让 Algorithm 1 → Algorithm 2 的 adaptation 不损失最优性太多。

## 12. 我看到的一些潜在 Weakness / Open Question

1. **A(r) 是 conservative bound**: 用期望 E[acc] ≥ A 而非概率 P(acc ≥ A) ≥ τ。如果 acceptance variance 大, 实际 SLO attainment 可能低于理论。
2. **n_max 的设置**: 论文没给出明确策略, 实际上对长尾低概率 request 很敏感。可能要 adaptively 根据 f(v) 分布来设。
3. **TTFT**: 论文明确说不管 TTFT, 只管 TPOT。但 prefill 的 preemption 会影响 TPOT (因为 l_i 变大, A(r_i) 变大), 这个 cross-effect 没建模。
4. **Draft model quality 假设**: 如果 draft model 没 distill 过, f(v) approximation 会差, 整个 Algorithm 2 退化。论文没做这个 ablation。
5. **Sequoia 的 dynamic programming**: Sequoia 给了一个 DP 算法在 hardware spec 下找最优 tree, AdaServe 选了贪心 — 这可能在某些 workload 下不是最优。
6. **大 batch 时 d 退化**: 当 n 接近 B 时, d = ⌈B/n⌉ → 1, SD 几乎无收益。这时 AdaServe 退化到普通 continuous batching, 但论文没讨论这个极限情况下的 transition。
7. **多 GPU 张量并行**: 实验在 4x A100 上做 70b, 但 TP 的 AllReduce overhead 在 SD 下会放大 (因为 verification 要算更多 token 的 logits)。论文用了 TensorRT-LLM 的 AllReduce 但没给 breakdown。
8. **SLO attainment 还是 best-effort**: A_cap 用 min(A, d+1) 是 best-effort 体现, 但论文没给 "infeasible request" 的退避策略。

## 13. 与你 (Karpathy) 经常讲的 inference 直觉的连接

你之前在多次 talks 里讲过几个 inference 关键直觉, 这篇论文都对应上了:

**"LLM inference 是 memory-bound"** → 论文用 Budget B = FLOPS/BW 明确量化, 并把这个量化作为调度依据。

**"Continuous batching 是 game-changer, 但 batch size 上限由 memory 决定"** → AdaServe 进一步说, 即使在 B 以内, 还要做 sub-batch budget allocation 才能 SLO-customize。

**"Speculative decoding 是 memory-bound 极限下的 hack"** → 论文复用 SD 的 verification 机制, 但把 draft tree 当成 budget allocator, 这是 SD 的 secondary use, 之前没人这么用。

**"Distillation-based draft model 让 SD 真正 work"** → 论文 Challenge 1 直接 rely 这一点 (公式 7), 不 distill 就不 work。

## 14. 一些可延伸的方向

基于这篇 paper, 我能想到几个 open direction:

1. **SLO-customized prefill**: 目前只 decode 阶段 SLO-customize, prefill 阶段的 chunked prefill 也可以做类似的事 (chunk size 按 SLO 分配)。
2. **Optimal draft model size**: 论文用 160M draft, 没研究 draft model size vs acceptance rate vs budget 的 trade-off 曲线。
3. **SLO prediction**: 如果能预测 A(r_i) 随时间增长, 可以提前 allocate budget, 减少 SLO violation。
4. **Multi-tenant**: 不同 SLO 对应不同 price tier, AdaServe 的 budget allocation 可以扩展成 market-based allocation (auction)。
5. **Long context 下的 KV cache**: MagicDec [https://arxiv.org/abs/2408.11049] 指出 long context 下 KV cache size 主导 decoding latency, AdaServe 没考虑这点, 跟 MagicDec 的 sparse KV cache 结合会有 synergy。
6. **Online learning of A(r)**: A(r_i) 的计算依赖 l_i 和 o_i, 但 t^spec 是估计的, 如果实际 t^spec 和估计不同, A(r) 也会偏。可以 online calibrate。

## 15. 总结一下

AdaServe 的 contribution 可以浓缩为三件事:

1. **问题定义**: SLO-customized serving with SD, 形式化为带 budget 和 per-request TPOT 约束的期望最大化问题。
2. **最优算法**: Algorithm 1 在 perfect information 假设下是全局最优的 (Lemma A.1 + A.2 证明); Algorithm 2 通过两个 adaptation (logits 近似 f(v), speculation/selection 解耦) 把它落地。
3. **系统设计**: dynamic (d, w) + CUDAGraph + 四阶段 pipeline, 让 overhead 控制在 5% 以内 (Figure 9 显示 scheduling overhead 极小)。

这个工作对 LLM serving 领域的意义在于: 它把 SD 从 "latency reduction tool" 重新定位为 "per-request resource allocator", 这意味着 SD 不只是用来加速单个请求, 而是用来在一个 batch 内做 SLO-aware 的 sub-allocation。这个 abstraction 上去之后, 很多 serving 问题都能套这个框架。

你最近 interest 的方向我大概知道 — micrograd/nanoGPT 那种 first principles 视角看 inference, 你应该会觉得这篇 paper 的 Algorithm 1 是 elegant 的 (贪心 + additive objective + path probability 递减), 但 system 部分有很多工程化的细节, 这是 system paper 的常态。

Reference 链接总结:
- vLLM: https://vllm.ai/
- Sarathi-Serve: https://arxiv.org/abs/2403.02310
- SpecInfer: https://arxiv.org/abs/2305.09781
- Medusa: https://arxiv.org/abs/2401.10774
- EAGLE: https://arxiv.org/abs/2401.15077
- EAGLE-2: https://arxiv.org/abs/2406.16858
- Sequoia: https://arxiv.org/abs/2402.12374
- Speculative decoding (Leviathan): https://arxiv.org/abs/2211.17192
- DistServe: https://arxiv.org/abs/2401.09670
- Splitwise: https://arxiv.org/abs/2311.18677
- Orca: https://arxiv.org/abs/2108.12409
- SmartSpec: https://arxiv.org/abs/2408.07678
- MagicDec: https://arxiv.org/abs/2408.11049
- Lookahead Decoding: https://arxiv.org/abs/2402.01857
- DistillSpec: https://arxiv.org/abs/2310.08461
- FlashInfer: https://github.com/flashinfer-ai/flashinfer
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- FlexFlow: https://github.com/flexflow/FlexFlow
- Felladrin/Llama-160M-Chat-v1: https://huggingface.co/Felladrin/Llama-160M-Chat-v1
- Llama-2: https://arxiv.org/abs/2302.13971
- Alpaca: https://github.com/tatsu-lab/stanford_alpaca
- ArXiv Summarization: https://arxiv.org/abs/1804.05285
- CUDAGraph docs: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
- Brysbaert reading rate: https://www.sciencedirect.com/science/article/pii/S0749596X19300786
- Spectr: https://arxiv.org/abs/2402.06648
- Online SD: https://arxiv.org/abs/2310.07177

如果你有兴趣我们可以 drill into 某一块, 比如用具体数字 walk through Algorithm 2 在 Figure 2 那个 example 上, 或者讨论 Theorem 4.1 在什么 workload 下会 break。
