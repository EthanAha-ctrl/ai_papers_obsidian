---
source_pdf: AVO Agentic Variation Operators for.pdf
paper_sha256: 1536be7066825b3c03d8a6c26df983dafb259db0cd40bf40a95a86cc4a704727
processed_at: '2026-07-18T12:56:15-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AVO: Agentic Variation Operators for Autonomous Evolutionary Search 深度解读

Andrej 你好，这篇 paper 来自 NVIDIA 团队（作者包括 Humphrey Shi, Luis Ceze, Ming-Yu Liu 等），核心 idea 是把 LLM-augmented evolutionary search 中的 LLM 从 "candidate generator" 升级为 "variation operator 本身"，用一个 self-directed coding agent 替代传统的 `Sample → Generate` 流水线。他们在 NVIDIA Blackwell B200 GPU 上做 attention kernel 的 7 天连续进化，跑出比 cuDNN 快 3.5%、比 FlashAttention-4 快 10.5% 的 MHA kernel，并且只花 30 分钟就把优化迁移到 GQA。下面我尽量展开技术细节来 build your intuition。

---

## 1. Core Idea：从 `Vary = Generate(Sample(P))` 到 `Vary = Agent(P, K, f)`

### 1.1 经典 LLM-augmented evolutionary search 的形式

FunSearch [Romera-Paredes et al., Nature 2024](https://www.nature.com/articles/s41586-023-06908-8) 和 AlphaEvolve [Novikov et al., 2025](https://arxiv.org/abs/2506.13131) 把 evolution 写成：

$$
\mathcal{P}_{t+1} = \text{Update}\big(\mathcal{P}_t, \beta(x_{t+1}, \mathbf{f}(x_{t+1}))\big), \quad x_{t+1} = \text{Vary}(\mathcal{P}_t)
$$

- $\mathcal{P}_t = \{(x_i, \mathbf{f}(x_i))\}_{i=1}^{t}$：population，由 solution-score pairs 组成
- $x_{t+1}$：第 $t+1$ 代 candidate
- $\mathbf{f}$：scoring function（这里 $f$ 是 vector-valued，每个分量 $f_j$ 对应一个 test configuration 的 throughput）
- $\beta$：替换/淘汰策略，比如 MAP-Elites archive [Mouret & Clune, 2015](https://arxiv.org/abs/1504.04909)
- $\text{Vary}$：**variation operator**，经典实现拆成两阶段：

$$
\text{Vary}(\mathcal{P}_t) = \text{Generate}\big(\text{Sample}(\mathcal{P}_t)\big)
$$

- $\text{Sample}$：从 $\mathcal{P}_t$ 选 parent(s)，由 heuristic 控制（fitness-based + diversity-based，例如 AlphaEvolve 用 island-based MAP-Elites，LoongFlow [Wan et al., 2025](https://arxiv.org/abs/2512.24077) 用 Boltzmann selection）
- $\text{Generate}$：LLM 被 prompt 后输出 **一个** candidate，单 turn 完成

这种 pipeline 的关键约束在于：LLM 只在 Generate 步骤介入，每次调用产生一个 candidate，没有能力主动去查文档、跑 profiling、解读 nsight compute output、修正自己的实现策略。当目标是已经"榨干"的 kernel（如 attention），进一步优化需要深度迭代工程，单 turn 输出就力不从心。

### 1.2 AVO 的形式

AVO 把整个 `Vary` 替换成一个 self-directed agent loop：

$$
\boxed{\text{Vary}(\mathcal{P}_t) = \text{Agent}(\mathcal{P}_t, \mathcal{K}, \mathbf{f})}
$$

- $\mathcal{P}_t$：full lineage（不只是 sampled parents，而是所有 prior solutions + scores）
- $\mathcal{K}$：domain knowledge base，包含 CUDA programming guide、[PTX ISA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/)、Blackwell architecture spec、FlashAttention-4 source code
- $\mathbf{f}$：scoring function（correctness + TFLOPS vector）
- $\text{Agent}$：通用 coding agent，有 planning、tool use、persistent memory，单次 variation step 内可执行无数次 internal actions

intuition 上：经典的 EVO 把 LLM 当作一个 **stateless function**（输入 parents → 输出 child），AVO 把 LLM 当作一个 **stateful process**，agent 自己决定"什么时候去看 cuDNN assembly 反汇编、什么时候去读 PTX manual、什么时候 commit、什么时候 abandon"。

### 1.3 与 TTT-Discover 的对比

TTT-Discover [Yuksekgonul et al., 2026](https://arxiv.org/abs/2601.16175) 进一步在 test time 用 gradient updates 让 Generate policy 自身 evolve，但 Sample 仍是 PUCT-based 固定规则，buffer 用预定义 update rules。AVO 则把 Sample + Generate + evaluation 都吸收到 agent loop 里，agent 拥有 full agency。注意 AVO 是 **orthogonal to population structure**：可以套在 single-lineage、archive、island 上。这篇 paper 选择 single-lineage 来 isolate operator 的 effect。

---

## 2. Attention Kernel on Blackwell：背景硬件上下文

### 2.1 FlashAttention 的算法骨架

给定 $Q, K, V \in \mathbb{R}^{N \times d}$，attention 计算：

$$
O = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

- $N$：sequence length
- $d$：head dimension（实验中 $d=128$）
- $S = QK^\top \in \mathbb{R}^{N \times N}$：score matrix，naive 实现会 materialize 它，导致 memory-bound
- $P = \text{softmax}(S) \in \mathbb{R}^{N \times N}$：attention weights
- FlashAttention [Dao et al., 2022](https://arxiv.org/abs/2205.14135) 通过 tiling 把 K 分成 block，逐块处理，维护 running row-maximum $m$ 和 row-sum $\ell$：

$$
m_{\text{new}} = \max(m_{\text{old}}, m_{\text{block}}), \quad m_{\text{block}} = \text{rowmax}(S_{\text{block}})
$$

$$
\ell_{\text{new}} = \ell_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \text{rowsum}\left(e^{S_{\text{block}} - m_{\text{new}}}\right)
$$

$$
O_{\text{new}} = O_{\text{old}} \cdot \frac{\ell_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}}}{\ell_{\text{new}}} + \frac{P_{\text{block}} \cdot V_{\text{block}}}{\ell_{\text{new}}}
$$

这里 $O_{\text{old}}$ 的 rescale factor $\frac{\ell_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}}}{\ell_{\text{new}}}$ 就是后面 Section 5.1 Branchless Accumulator Rescaling 优化的对象。

### 2.2 Blackwell 上 FA4 的 warp specialization 架构

[FlashAttention-4 (Zadouri et al., 2026)](https://arxiv.org/abs/2603.05451) 在 Blackwell 上采用 warp specialization + dual Q-stage pipeline。一个 thread block 内部按 warp group 分工：

| Warp Group | 数量 | 职责 |
|---|---|---|
| MMA warps | (主) | 执行 QK GEMM 产生 $S$，执行 PV GEMM 累积 $O$，通过 Blackwell tensor core instructions |
| Softmax warps | 8 (在 AVO v32 中) | 从 $S$ 计算 $P = \text{softmax}(S)$，online softmax with running row-max |
| Correction warps | 4 (在 AVO v32 中) | 当 running maximum 改变时 rescale $O$ accumulator |
| Load / epilogue warps | 4 (剩余) | 通过 TMA (Tensor Memory Accelerator) 处理 data movement |

dual Q-stage：两个 Q-tile 并发跑，barrier-based signaling 协调 handoff。Causal attention 还要处理 fully-masked / fully-unmasked K-block iterations，执行路径不同。

直觉上：FA4 已经把 warp 间做成了 producer-consumer pipeline，要把这个 kernel 再推一步，需要在 **sub-warp-cycle 级别**做精细的 scheduling、register allocation、memory ordering 调优，这远超单 turn LLM 输出能覆盖的范围。

---

## 3. AVO Anatomy：一次 variation step 内部发生什么

一个 variation step 产生 $x_{t+1}$，是一个 **autonomous agent loop**：

1. **Context retrieval**：agent 检视 lineage $\mathcal{P}_t$ 中多个 prior versions（不一定只看 best，会对比它们的 profiling 特征）
2. **Knowledge consultation**：agent 在 $\mathcal{K}$ 中查阅 PTX ISA / Blackwell spec / FA4 source
3. **Edit**：实施一个 candidate optimization
4. **Invoke $\mathbf{f}$**：跑 correctness + throughput benchmark
5. **Diagnose**：如果 fail correctness 或 regression，agent 解读 compiler output / nsight compute profile
6. **Revise**：edit-evaluate-diagnose 循环直到 commit

只有当 candidate 通过 correctness + 至少 match best score 才被 commit 为新的 $x_{t+1}$，记录为 git commit；unsuccessful 中间 attempts 留在 agent internal trajectory 但不入 lineage。

### 3.1 Continuous Evolution 的 self-supervision

长跑 autonomous 有两个 failure mode：
- **Stall**：当前 exploration direction 耗尽
- **Unproductive cycle**：反复 edit 但 score 不动

AVO 加了一个 self-supervision mechanism：检测到上述场景时，review 整个 evolutionary trajectory，steer search 朝几个 candidate 方向。这等价于在 agent 层面引入 "meta-level intervention"，避免 local optimum 卡死。

直觉：这种 supervisor 不是规则化的 restart，而是让 agent 主动对自己的 history 做 reflection，类似 ReAct / Reflexion 的 self-critique 思路，但作用在 evolution 这个时间尺度上。

---

## 4. 实验结果

### 4.1 Setup

- **Agent**：NVIDIA 内部通用 coding agent，frontier LLM 驱动，可做 code editing / shell exec / file nav / doc retrieval，没有针对 kernel 优化做 task-specific modification
- **Hardware**：NVIDIA B200 GPU, CUDA 13.1, PyTorch 2.10.0
- **Baselines**：cuDNN 9.19.1（Blackwell 优化版）和 FlashAttention-4 (commit 71bf77c)
- **Benchmark configs**：forward prefilling，$d=128$，BF16，sequence lengths $\{4096, 8192, 16384, 32768\}$，total tokens 固定 32768（e.g. seq=4096 时 batch=8）。每个配置跑 10 次取 mean/std
- **MHA**：16 heads, causal & non-causal
- **GQA**：Qwen3 配置 — group=8 (32 Q heads / 4 KV heads, Qwen3-30B-A3B) 和 group=4 (32/8, Qwen3-8B)

### 4.2 MHA 主结果

| 比较 | Causal | Non-causal |
|---|---|---|
| AVO vs cuDNN | +0.4% ~ +3.5% 全配置领先 | +1.8% ~ +2.4% 在 seq≥16384；短序列在 noise 内 |
| AVO vs FA4 | +5.0% ~ +10.5% 全配置领先 | 同上 |

AVO 在 causal MHA 上最高达到 **1668 TFLOPS BF16**。

### 4.3 GQA Transfer

agent 被 prompt 让它把 evolved MHA kernel adapt 到 GQA，**autonomously 完成，30 分钟**，无需 human guidance。结果：

| 比较 | Causal GQA | Non-causal GQA |
|---|---|---|
| AVO vs cuDNN | up to +7.0% | up to +6.0% |
| AVO vs FA4 | up to +9.3% | up to +4.5% |

这个 transfer 结果是最让我印象深刻的部分：说明 agent 在 MHA 上发现的优化是 **跨 attention 变体通用的硬件级 reasoning**，而非 MHA-specific 的 hardcoded trick。直觉上，warp specialization pipeline 的 register rebalancing、pipeline overlap 这些优化在 GQA 的不同 compute/memory access pattern 下依然有效，agent 只需做少量 param/code 调整。

### 4.4 Evolution Trajectory 的统计特征

7 天，**40 个 committed versions**，但内部探索了 **500+ candidate optimization directions**（包括 fail correctness、regress throughput、被 profile 后 abandon 的尝试）。

轨迹观察到的三个 pattern：

1. **Scale of exploration**：500+ directions，每个都要读文档、实现、compile、test、profile，远超人类工程师同等时间能覆盖
2. **Discrete jumps**：throughput 不是连续上升，而是阶梯式跳变 + plateau，五个最大跳变对应 architectural inflection points：
   - v8: QK-PV interleaving + bitmask causal masking
   - v13: restructured single-pass softmax
   - v20: branchless accumulator rescaling + lighter fence
   - v30: correction/MMA pipeline overlap
   - v33: register rebalancing across warp groups
3. **Diminishing returns**：v1-v20 抓 coarse-grained 大头，v21-v40 通过 cycle-level scheduling + refined resource allocation 做 compounding small wins

---

## 5. Agent 发现的三个代表性优化（技术深挖）

这是 paper 最有价值的部分，因为它展示了 agent 做的是 **真正的 hardware reasoning**，而非表面 code transformation。

### 5.1 Branchless Accumulator Rescaling (v19 → v20)

**Bottleneck**：在 online softmax 算法中，每个 K-block 处理后 $m$ 可能更新，需要把 output accumulator $O$ rescale。v19 的实现：

```cuda
if (__any(rescale_needed)) {
    O *= rescale_factor;
    __syncthreads();  // or heavy fence
}
```

问题有两个：
- warp synchronization overhead 在每次 K-block iteration 都付
- conditional control flow 阻止用更轻的 memory fence

**AVO 的 fix (v20)**：

```cuda
// always compute rescale factor (predicated select)
float r = rescale_needed ? factor : 1.0f;
O *= r;
// non-blocking fence — only enforces ordering, no stall
```

为什么这个 safe？因为 branchless path 保证 warp 内所有 thread 走相同 control flow，reconvergence 在下一个 sync point 之前已经发生，所以只需要 ordering fence（acquire/release 语义）而不需要 wait-on-write fence。

**Measured impact**：

| 比较 | Non-causal | Causal |
|---|---|---|
| v19 → v20 | +8.1% | +1.6% |

非对称的直觉：branchless 路径只对 fully unmasked K-block iteration 生效。Non-causal 全部 K-block 都是 unmasked，所以全受益；causal 的 masked block 仍走原 branched logic，所以只部分受益。

这个 optimization 的反直觉之处：去掉 branch 后**无条件**做了一次 multiply-by-one，看起来浪费 FLOPs，但省掉了 warp divergence + heavy fence，整体是 +8.1% 大涨。这恰好是 GPU 性能中 "compute 便宜、sync 昂贵" 的经典现象。

### 5.2 Correction/MMA Pipeline Overlap (v29 → v30)

**Bottleneck**：dual Q-stage 设计中，stage 1 和 stage 2 各自跑 PV GEMM，correction warp 必须等**两个 stage 的 PV GEMM 都完成**才能开始 normalize 任何一个 stage 的 output。结果：correction warp 在第二个 PV GEMM 期间 idle。

```
v29 timeline:
Stage1 PV GEMM |─────────|
Stage2 PV GEMM            |─────────|
Correction                :wait:     |normalize both|
                       (idle)
```

**AVO 的 fix (v30)**：让 correction warp 在 stage 1 的 PV GEMM 完成后立刻开始 normalize stage 1 的输出，与 stage 2 的 PV GEMM 并行：

```
v30 timeline:
Stage1 PV GEMM |─────────|
Stage2 PV GEMM            |─────────|
Correction      :       |norm1|norm2|
```

把 sequential dependency 转成 pipelined execution，降低 correction warp idle 时间。

**Measured impact**：

| 比较 | Non-causal | Causal |
|---|---|---|
| v29 → v30 | +1.1% | +0.4% |

### 5.3 Register Rebalancing Across Warp Groups (v32 → v33)

**Blackwell 寄存器预算**：2048 warp-registers per SM，partition 到 warp groups。

v32 allocation（沿用 FA4 pattern）：

| Warp Group | # warps | registers/warp | total | 状态 |
|---|---|---|---|---|
| Softmax | 8 | 192 | 1536 | headroom 充足 |
| Correction | 4 | 80 | 320 | **spill 到 local memory** |
| Load/epilogue | 4 | 48 | 192 | — |
| **Total** | 16 | | **2048** | |

验证：$192 \times 8 + 80 \times 4 + 48 \times 4 = 1536 + 320 + 192 = 2048$ ✓

profiling 显示 correction warp 在 80 registers 预算下 spill 到 slower local memory，softmax 组却有大头 headroom。

**AVO 的 fix (v33)**：从 softmax 组 per-warp 减 8 registers，给其他两组各加 8 registers/warp：

| Warp Group | # warps | registers/warp | total |
|---|---|---|---|
| Softmax | 8 | 184 | 1472 |
| Correction | 4 | 88 | 352 |
| Load/epilogue | 4 | 56 | 224 |
| **Total** | 16 | | **2048** ✓ |

为什么 softmax 能减寄存器？因为 AVO kernel 的 softmax 实现用 fragment 粒度的 packed arithmetic，peak register usage 低，184 仍有 ample headroom。

为什么 correction 现在是 critical path？因为 v30 的 pipeline overlap 让 correction warp 与第二个 PV GEMM 并行跑，进入 critical path。88 vs 80 registers 让更少 output value spill 到 local memory，减少 stall。

**Measured impact**：

| 比较 | Non-causal | Causal |
|---|---|---|
| v32 → v33 | +2.1% | ~0% |

直觉：这个 optimization 是 v30 的 "follow-on"。如果没有 v30 把 correction 推上 critical path，给 correction 加 register 的收益会被掩盖。这是 agent 能做 **joint reasoning across multiple subsystems** 的证据——它需要同时理解 synchronization / memory ordering (v20)、pipeline scheduling (v30)、register allocation (v33) 三层才能找出这个组合。

---

## 6. 一些值得 build intuition 的观察

### 6.1 为什么 single-lineage 也 work

paper 刻意选 single-lineage 来 isolate operator effect。但直觉上 40 个 committed versions + 500+ internal attempts 也能 work，说明 **agent 自己在内部就做了 diversity management**：通过 persistent memory 看到完整 lineage，agent 自己判断何时 revisit 老思路、何时 abandon、何时 pivot strategy。这等价于把 MAP-Elites 的 archive management 内化到 agent 的 reasoning 里。

### 6.2 Compute-vs-sync asymmetry 的反复出现

三个 optimization 都印证一个 hardware-level truth：
- v20：multiply-by-one 比 warp sync 便宜 → 去掉 branch
- v30：让 correction warp 干活比让它 idle 等 sync 便宜
- v33：寄存器 spill 到 local memory（要走 L1→L2→HBM hierarchy）比在 SM register file 内访问贵得多

agent 没有显式被教过这些，它是通过 read PTX manual + profile + iterate 学到的。

### 6.3 Causal vs Non-causal 的 asymmetry

很多 optimization 在 non-causal 上收益更大（v20: +8.1% vs +1.6%, v33: +2.1% vs ~0%），因为 non-causal 全 K-block unmasked，新路径覆盖率高。Causal 在 masked iteration 仍走旧路径。这暗示下一步优化方向是 **为 causal masked path 也设计轻量路径**，例如用 bitmask-based 跳过 masked K-block + 在 partial-mask block 用更细粒度的 branchless scheme。

### 6.4 与 VibeTensor 的关系

paper 引用了同作者群的 [VibeTensor (Xu et al., 2026)](https://arxiv.org/abs/2602.07701)，"System Software for Deep Learning, Fully Generated by AI Agents"。AVO 是更专注于 **per-kernel extreme optimization** 的特化场景，而 VibeTensor 是更宏大的 system-level 蓝图。结合起来看，NVIDIA 在尝试把 "AI agent 做 system software" 这个范式从单点 kernel 推广到整个 DL software stack。

### 6.5 与 AlphaEvolve 的本质区别

AlphaEvolve 的 Generate 还是 LLM single-turn，innovation 在 LLM 算法（DB-driven island + diversity sampling）。AVO 的 Generate 是 multi-day agent loop，innovation 在 **agent 取代 operator 本身**。这意味着 AVO 的搜索 cost 远高于 AlphaEvolve 单次 Generate（7 天 500+ attempts），但每次 attempt 的 "信息量" 也远高——agent 真的在读 PTX manual、跑 nsight、改 PTX inline assembly。对 attention 这种 "high-skill、low-diversity" 的优化问题，AVO 的 trade-off 更合适；对 FunSearch 解决的 cap set problem 那类 "low-skill、high-diversity" 问题，AlphaEvolve 可能仍更合适。

---

## 7. 局限和开放问题

paper 自己指出 single-lineage 是当前 instantiation，未来可扩展到 archive-based / island-based。还有一些 paper 没明说但值得思考的：

1. **Reproducibility**：7 天运行 500+ attempts，agent non-determinism（temperature sampling、tool call latency）会让复现困难。建议作者 release trajectory log
2. **Cost**：500+ attempts × 每个 attempt 多轮 LLM call × frontier LLM API cost，单次 7 天 run 的 $ 估计不菲。是否可以用更小 LLM 做 first-pass filter + 大 LLM 做 fine-tune？类似 AlphaEvolve 的双 LLM setup
3. **Generalization beyond attention**：paper 自承 "beyond attention kernels" 是 future work。但 attention 有非常成熟的 benchmark suite (FA4 提供)，其他 kernel (conv, GEMM, layernorm) 是否同样 agent-friendly？
4. **Memory scaling**：lineage $\mathcal{P}_t$ 到 v40 时 context 已经很大，agent 的 persistent memory 如何 manage context budget？是 full history 还是 summarized？
5. **Supervisor 的触发条件**：Section 3.3 提到 self-supervision 在 stagnation 触发，但触发阈值是 heuristic 还是 learned？

---

## 8. Reference links

- AVO paper 本身（NVIDIA 2026，可能还没上 arXiv 公开版，留意作者主页）
- [FunSearch, Nature 2024](https://www.nature.com/articles/s41586-023-06908-8)
- [AlphaEvolve, arXiv 2506.13131](https://arxiv.org/abs/2506.13131)
- [LoongFlow, arXiv 2512.24077](https://arxiv.org/abs/2512.24077)
- [TTT-Discover, arXiv 2601.16175](https://arxiv.org/abs/2601.16175)
- [FlashAttention original, arXiv 2205.14135](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2, arXiv 2307.08691](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3, arXiv 2407.08608](https://arxiv.org/abs/2407.08608)
- [FlashAttention-4, arXiv 2603.05451](https://arxiv.org/abs/2603.05451)
- [cuDNN original paper, arXiv 1410.0759](https://arxiv.org/abs/1410.0759)
- [SWE-bench, arXiv 2310.06770](https://arxiv.org/abs/2310.06770)
- [SWE-agent, arXiv 2405.15793](https://arxiv.org/abs/2405.15793)
- [OpenHands, arXiv 2407.16741](https://arxiv.org/abs/2407.16741)
- [Claude 3.7 Sonnet & Claude Code](https://www.anthropic.com/news/claude-3-7-sonnet)
- [OpenAI Codex](https://openai.com/index/introducing-codex/)
- [MAP-Elites, arXiv 1504.04909](https://arxiv.org/abs/1504.04909)
- [VibeTensor, arXiv 2602.07701](https://arxiv.org/abs/2602.07701)
- [NVIDIA PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA Blackwell architecture whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture)
- [Qwen3 technical report, arXiv 2505.09388](https://arxiv.org/abs/2505.09388)
- [ReEvo, arXiv 2402.01145](https://arxiv.org/abs/2402.01145)
- [Evolution through Large Models, arXiv 2206.08896](https://arxiv.org/abs/2206.08896)

---

## 9. 我的 take-away intuition

读完最强烈的感受：这篇 paper 标志着 **LLM-for-systems 的范式从 "generation" 转向 "evolution-as-agentic-process"**。FunSearch/AlphaEvolve 让我们看到 LLM 可以做 evolutionary search 中的 mutation operator；AVO 让我们看到 **当 mutation operator 本身变成 agent，可以解决需要月级 expert engineering 的问题**。

关键的 meta-observation：FA4 是 Tri Dao + NVIDIA 团队月级工程产物，AVO 7 天自主跑超过它 10.5%。这个绝对数字不大，但相对意义惊人——意味着 **expert-level GPU kernel engineering 的 marginal cost 正在被 agent 显著压低**。如果这个 scaling 继续下去（更长 run、更强 agent、更多 docs in K），下一次 attention kernel 的 SOTA 可能不是 Tri Dao 手写，而是 agent 自主 evolve 出来的。

而 GQA 30 分钟迁移的结果暗示一个更深的现象：agent 发现的优化是 **以硬件 reasoning 为媒介的**，所以能跨 attention variant 复用。这跟人类 expert 写 kernel 时积累的 "hardware intuition" 是同构的——只是 agent 的 "intuition" 是从 lineage 的 success/failure pattern 中 infer 出来的，而人类 expert 的是从多年踩坑经验中积累的。

我会很期待看到：同样的 AVO 套用到 Hopper / AMD MI300 / TPU 上，看 agent 能否做 cross-vendor hardware reasoning；以及把 lineage $\mathcal{P}_t$ 显式组织成 graph（不只是 linear），看 archive-based AVO 是否能解决 agent 在长跑后期的 exploration exhaustion 问题。
