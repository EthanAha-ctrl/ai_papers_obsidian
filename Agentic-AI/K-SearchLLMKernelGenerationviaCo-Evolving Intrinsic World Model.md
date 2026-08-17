---
source_pdf: K-SearchLLMKernelGenerationviaCo-Evolving Intrinsic World Model.pdf
paper_sha256: bb7c52042664173039bf8c453f105c19ae3a1ce6116d7307bfaed2e806926cef
processed_at: '2026-08-05T11:00:44-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# K-Search 的人话版

让我用白板前聊天的方式重新讲一遍。

---

## 一句话概括

让 LLM 别再当一个"喂代码吐代码"的黑箱函数了，让它当回自己——一个**会规划、会反思、会改主意的工程师**，在 kernel 优化的 search tree 上边走边想。

---

## 问题本身有多恶心

GPU kernel 优化这件事，用人类工程师的话讲就是："你知道终点在哪，但路径上全是岔路，每条岔路走错一步就要重写一千行 CUDA。"

三个让人头秃的原因：

1. **Design space 太大**：tiling 怎么切、memory layout 怎么排、shared memory 用多少、register 怎么分配、要不要用 TMA、要不要用 WMMA、block size 多大、warp 怎么分工……这些 decisions 互相耦合，动一个影响一堆。参考 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 看一眼指令集就懂。

2. **硬件一代换一代**：Hopper 上写好的 kernel，到 Blackwell 上因为新增了 TMA、FP8 tensor core 这些东西，trade-off 全变了，之前的 optimal 变 sub-optimal。[FlashAttention-3](https://arxiv.org/abs/2407.08608) 就是为 Hopper 重写的例子。

3. **试一次成本高**：编译 + profiling 一个 kernel 要秒级，你不可能试几万次。[Ansor](https://arxiv.org/abs/2006.06762) 当年就为这个引入了 cost model 来减少真实 evaluation 次数。

---

## 现有方法为什么拉胯

OpenEvolve、ShinkaEvolve、FunSearch、AlphaEvolve 这帮方法，本质都是同一套思路：**LLM 当变异算子，evolution algorithm 当外层循环**。

工作流程长这样：
```
loop:
    从 archive 里挑几个 parent program（用 MAP-Elites 之类的启发式）
    把它们的源码 + profiling feedback 塞进 prompt
    LLM 生成一个新 code
    编译、跑、打分
    好的就塞回 archive
```

这跟 GA（genetic algorithm）的区别只是 mutation operator 从"随机翻转 bit"变成"LLM 生成 code"。

问题在哪？四个：

### 1. 策略和实现绑死了

我想做"用 padding 解决 bank conflict"这个 high-level 决定，LLM 写完代码有个 typo，compile 失败。Evolution 看到的是"这个 candidate 分数 0，丢掉"。它根本不知道 typo 背后的 strategy 是好的。**好主意被代码里的笔误连坐了**。

### 2. 优化路径经常是先退后进

高性能 kernel 的典型路径：
- 先 refactor memory layout（性能可能暂时下降）
- 再做 vectorization（开始回升）
- 再做 split-K（飞起来）

Evolution 在第一步看到性能下降就丢掉了，永远到不了第三步。它处理不了 **non-monotonic** 的优化轨迹。

### 3. LLM 的 prior 被浪费了

LLM 其实"知道" split-K 什么时候有用、什么时候没用。但 baseline 的用法里，这个 knowledge 每次调用都从 prior 重新 sample，搜了几十轮之后，LLM 对这个具体 kernel 的理解没有任何积累。**它的 domain knowledge 是沉睡的**。

### 4. Sample efficiency 烂

Paper 里实测：ShinkaEvolve 在 GQA 上的 logs 里，**绝大多数 generations 得 0 分**。budget 大量花在扩展无效 candidate 上。OpenEvolve 在 MoE 上平均 final score 只有 3.09，基本卡在"连正确都做不到"的 regime 出不来。

---

## K-Search 的核心 insight

一句话：**把 LLM 拆成两个角色，一个负责想（规划），一个负责写（实现），让它们各自做擅长的事**。

具体讲：

| 角色 | 做什么 | 类比 |
|---|---|---|
| World Model | 维护 search tree、估 priority、决定 Insert/Update/Prune | 资深工程师的白板 |
| Code Policy $\pi_{\text{code}}$ | 给定 high-level intent，反复 sample 具体代码 | 实习生敲键盘 |

这个拆分带来一个直接好处：**一个好主意即便第一次实现失败，也还有机会被再试 K 次**。Stagnation counter $K=7$ 意味着同一个 intent 可以被 LLM 尝试 7 次不同 implementation，只有 7 次都不 improve 才放弃。这就解决了"typo 连坐"问题。

---

## 形式化（但用大白话讲变量）

### Objective

$$J(x) = s \cdot \frac{p_{\text{ref}}}{p} \cdot 100$$

- $x$：candidate kernel 程序
- $s \in \{0,1\}$：correctness flag，过了 numerical check 才是 1
- $p_{\text{ref}}$：baseline (FlashInfer) 的 latency
- $p$：candidate 的 latency
- 乘 100 是把 speedup 缩放到 0-100 量级，方便 LLM 当数字读

$s$ 当乘法门用：不正确直接归零，强制 correctness first。这个设计很关键——否则 LLM 可能找到"跑得飞快但结果错"的 kernel。

### Search State $S_t$

$S_t$ 是 LLM 维护的一棵 search tree，包含：
- 已经探索过的 action 和它们的 outcome（Closed nodes，蓝色）
- 待探索的 frontier actions（Open nodes，橙色）
- 每个 frontier action 的 priority score $V \in [0,1]$

### Action 结构

$$a_t = (x_{\text{parent}}, \delta)$$

- $x_{\text{parent}}$：从哪个已有 program 出发
- $\delta$：natural language intent，比如 `"fuse heads"`、`"register-resident rescaling"`、`"chunk32 prescale vectorized"`

这一步是关键抽象：**action space 是 intent 空间，不是 code 空间**。intent 空间小、结构化、可推理；code 空间大、稀疏、充满噪声。

### World Model 的 transition

$$S_{t+1} \sim P_{\text{model}}(S \mid S_t, a_t; x_t, o_t)$$

- $S_t$：当前 search state
- $a_t$：刚执行的 action
- $x_t, o_t$：刚跑出来的 program 和 observation
- $P_{\text{model}}$：由 LLM 通过 in-context learning 实现的分布

这里没有 fine-tuning，纯 prompt + history。LLM 读进 history of (intent, score, feedback)，输出对 search tree 的三类 edit operation。

---

## 三个阶段一个 round

每个 round（消耗一次 evaluator 调用，一单位 budget）：

### Phase 1: Action Selection

$$a_t = \arg\max_{a \in \mathcal{A}(S_t)} V(a \mid S_t)$$

LLM 在 frontier 里挑 $V$ 最高的 action。简单粗暴的贪心。

可以联想到 UCB / Thompson sampling 那套 explore-exploit 框架——当前是纯 exploit。加个 $V(1-V)$ 的 exploration bonus 可能更好，类似 [RAP](https://arxiv.org/abs/2305.14992) 里的做法。

### Phase 2: Local Refinement

```
n = 0
while budget > 0 and n < K:
    x = LLM_sample(a_t)      # 给定 intent, 让 LLM 写代码
    o = E(x)                 # 编译 + 跑 + profile
    if J(x) > J(x_best):
        x_best = x; n = 0    # improve 了, reset counter
    else:
        n += 1               # 没进步, counter +1
```

$K=7$（FlashInfer 任务）或 $K=5$（Triton 任务）。

这段 loop 是整个系统的**抗噪层**。它让 high-level strategy 对 low-level coding noise 鲁棒。

### Phase 3: World Model Update

LLM 看 $(a_t, x_{\text{best}}, o_{\text{best}})$ 的 trajectory，对 search tree 做三类操作：

- **Insert**：propose 新 child action，赋初始 $V$
- **Update**：根据新证据调整已有 frontier action 的 $V$（figure 里 $u_{11}$ 从 $0.9 \to 0.6$）
- **Prune**：永久删掉死胡同分支（figure 里 $u_{10}$）

这三个操作加在一起，就是"world model co-evolution"——LLM 对 kernel 优化的 understanding 随搜索推进而 sharpen。

---

## MLA Case Study 是理解整篇 paper 的钥匙

Figure 2 的 trace 值得反复看。我从 round 1 到 102 讲故事：

### Round 1: 初始 frontier

LLM 注入三个 high-level candidate：
- `fused_multi_head`：把共享 CKV 的 head 合并处理，减少 global memory traffic 16×
- `split_k_decoding`：序列切分到多 block
- `independent_heads`：每个 head 独立处理

LLM 推理：fused 能砍 16× memory traffic，给它 $V$ 最高。

### Round 1-14: 第一个分支展开

`fused_multi_head` 被 instantiate，local refinement 跑完，得 $J=34$。

### Round 14-34: Belief sharpening

LLM 看到 fused 有效，做三件事：
- **Insert** `register_resident_rescaling`、`occupancy_tuned_chunk32`——深化成功分支
- **Update** `independent_heads` 的 $V$ 下调——既然 fusion proven 有效，独立处理就 less promising
- **Prune** `independent_heads`——round 34，evidence 够了，永久删除

这是 world model 在**修正自己的 prior**。Evolution baseline 做不到这个，因为它没有 explicit 的 belief state。

### Round 42: 结构性 insight（最精彩的部分）

LLM 做了一个**非平凡的结构重排**：把根层的 `split_k` 删掉，但在 `register_resident` 分支深处 re-Insert 一个 `low_overhead_split_k`。

这个动作的 reasoning：split-K 单独作为 baseline 没用（之前试过），但作为 strong fusion kernel 之上的 composable optimization 非常有用。

这需要 LLM 理解 **composability**——某个 optimization 的效果依赖于它作用在什么样的 base 上。Evolution baseline 看不到这层，因为它搜的是 raw code space，无法表达"同一 strategy 在不同 context 下不同效果"。

### Round 42-102: 收敛到全局最优

`chunk32_vectorized` 成功后，LLM 推断可以在 load Q 时立刻 apply `sm_scale`（而不是在后面 softmax 时再 apply），propose `chunk32_prescale_vectorized`。Round 102 yield 全局最优。

这个 refinement 是典型的"细节决定成败"——一个很小的数值 placement 优化，只有在前面所有 structural decision 都对了之后才有意义。Evolution baseline 在前面 structural step 就卡死了，根本到不了这里。

---

## 实验数字讲人话

### FlashInfer 四个 kernel（120 iterations）

| Kernel | K-Search | OpenEvolve | ShinkaEvolve | 倍数 |
|---|---|---|---|---|
| GQA decode | 76.0 | 44.2 | 27.7 | 1.72× / 2.74× |
| MLA decode | 47.1 | 39.9 | 34.7 | 1.18× / 1.36× |
| MLA prefill | 57.4 | 19.5 | 11.3 | 2.95× / 5.10× |
| **MoE (FP8)** | **44.1** | **3.09** | **27.9** | **14.3× / 1.58×** |
| **Overall** | **56.13** | **26.68** | **25.37** | **2.10× / 2.21×** |

MoE 上 14.3× 的 gain 最亮眼。MoE 涉及 irregular routing + load balancing + FP8 packing，是典型的 multi-step coordinated transformation 场景。OpenEvolve 卡在 3.09 出不来——它连正确性都搞不定，更别提性能。

### GPUMODE TriMul（Table 3）

| 提交 | 语言 | Model | Iter | Latency (µs) |
|---|---|---|---|---|
| shiyegao（人类）| CUDA | — | — | 1074 |
| Zeyu Shen（人类）| Triton | — | — | 1140 |
| TTT-Discover | Triton | GPT-OSS-20B + RL | 25600 | 1161 |
| **K-Search** | **Triton** | **GPT-5.2 → Gemini-3-Pro** | **300** | **1030** |

300 iterations 干翻 25600 iterations 的 RL 方法，还干翻人类。TriMul 是 AlphaFold3 里的 Triangle Multiplicative Update，涉及 LayerNorm + 5 个 gated projection + $\mathcal{O}(N^3)$ pairwise contraction + gated output——多 stage 结构，刚好是 K-Search 强项：可以对每个 stage 单独规划 intent。

参考 [TTT-Discover](https://arxiv.org/abs/2601.16175) 和 [GPUMODE leaderboard](https://www.gpumode.com/benchmarks)。

---

## 生成的 kernel 技术细节

### FP8 MoE（Blackwell）

**K-Search 的 routing**：每 token 一个 block（256 threads），warp 内 `__shfl_down_sync` 做 reduction 找 top-8 expert。Warp 级并行，避免 serialization。

**OpenEvolve**：persistent kernel + `atomicAdd` 在 while loop 里取 tile index → atomic 在 hot loop 里的 overhead 是灾难。

**ShinkaEvolve**：每 thread 串行 load 256 scores + for-loop 找 top → 没用 GPU 并行性。

**K-Search 的 FFN**：
- Pipeline：routing → sort-scatter（按 expert 重排 token 到 contiguous memory）→ gate+up → SiLU → down
- **WMMA tensor core** 在 $16 \times 16$ block 上
- **Double buffering**：load next block 与 compute current 重叠
- **Skip empty expert**：若某 expert 0 token 则跳过

### GQA Paged Decode（Hopper）

**K-Search**：split KV sequence 到多 block，每 block 处理一段，partial result 写 temporary buffer，最后 reduce。用 lightweight counter 检测哪个 block 是 last。

**Baseline**：单 block 处理整个 sequence → 长 sequence 时无法并行。

**Memory/compute overlap**：K-Search double-buffered chunks；ShinkaEvolve 单 buffer 串行。

### MLA Paged Decode（Hopper）

K-Search 三个 trick：
1. **Adaptive split**：短 sequence 单 block 直写 output（避免 reduce overhead）；长 sequence 才 split
2. **Register-resident Q**：每 token Q 是 $16 \times 576$，小但每 chunk reuse。K-Search 把 Q 放 register fragment，不 materialize 到 shared memory → 降低 SM pressure
3. **Deeper prefetch pipeline**：load **two chunks ahead**（chunk $i$ compute + chunk $i+1$ loading + chunk $i+2$ issuing load），比标准 double buffering 更满

### MLA Paged Prefill（Hopper）

**Variable-length batch on GPU**：16-row tile 可能跨 sequence boundary。K-Search 在 GPU 上用 prefix-sum array 动态 resolve split，每 block 处理完一个 segment 移到下一个。

**Baseline**：CPU 端 precompute tile-to-batch mapping → 多一次 CPU pass + 额外 memory。

---

## 直觉构建：为什么这套思路 work

让我把核心 mental model 讲清楚。

### Evolution baseline 的视角

LLM 是一个黑箱函数 $f: \text{prompt} \to \text{code}$。外层 evolution loop 决定 prompt 内容。LLM 的"理解"每次调用都从 prior 重新 sample，没有积累。搜了 100 轮和搜了 1 轮，LLM 对这个具体 kernel 的理解深度一样。

### K-Search 的视角

LLM 同时扮演两个 role：
- **World Model**：维护 $S_t$，推理 $V$，决策 tree edit
- **Code Policy** $\pi_{\text{code}}$：给定 $a_t$ 写代码

关键在第一个 role：LLM 的 prior knowledge 被显式 project 到一个 structured planning space。它不再隐式知道 split-K 何时有效，而是显式输出 $V(\text{split-K} | S_t) \in [0,1]$ 这个标量。

随着搜索推进，$S_t$ 越来越 informative，LLM 的 $V$ 估计越来越准。这就是 co-evolution 的本质——**LLM 的 belief 和 kernel 的优化进度一起往前走**。

### Local refinement 的意义

它是一道**抗噪滤波器**。intent $\delta$ 是 signal，具体 code 是 signal + noise。同一个 $\delta$ sample 7 次，只要有一次 work 就保留 intent。Evolution baseline 把 signal 和 noise 绑在一起评估，signal 被 noise 淹没。

### Non-monotonic 路径的 navigability

MLA case round 42 那个"删根层 split-K、在深处 re-Insert" 的操作是核心证据。LLM 理解了 split-K 的效果 **conditional on base kernel quality**——在弱 base 上无用，在强 base 上有大用。这种 compositional reasoning 是 evolution baseline 做不到的，因为它的 archive 只存 raw code，不存 "为什么这个 code work" 的 reasoning。

---

## 我（Karpathy 视角）的延伸思考

### 1. 这跟 Model-Based RL 的关系

K-Search 本质是 **model-based planning with LLM as the dynamics model**。类比 Dyna / MuZero 那套：learned dynamics model + planning。区别是 K-Search 的 "dynamics model" 是 LLM 的 prior + in-context learning，没有 weight update。

参考 [Ha & Schmidhuber, World Models](https://arxiv.org/abs/1803.10122)、[MuZero](https://arxiv.org/abs/1911.08265)、[PlaNet](https://arxiv.org/abs/1811.04551)。

### 2. $V$ function 的 calibration 问题

当前 $V$ 是 LLM 输出的 scalar，没有 ground truth 验证它是否 calibrated。LLM 可能系统性 overestimate 某 class of intent。可以用 historical (intent, final_score) pair 训练一个 calibration head，或者用 [conformal prediction](https://arxiv.org/abs/2107.07511) 给 $V$ 加 confidence interval。

### 3. Search tree 的 transfer

同一架构的 kernel optimization knowledge 能否跨任务 transfer？比如搜完 GQA decode 后，search tree 里关于 "register-resident Q" 的 belief 能否帮 MLA decode 起步？这通向 [open-ended learning](https://arxiv.org/abs/2509.19349) 的方向。需要一种 "abstracted intent transfer" 机制——提取 tree 中泛化性强的 subtree，作为新任务的 initial $S_0$。

### 4. 跟编译器 DSL 的结合

K-Search 当前在 CUDA/Triton program space 搜。可以把它接到 [CuTe layout abstraction](https://github.com/NVIDIA/cutlass) 或 [TVM Ansor](https://arxiv.org/abs/2006.06762) 的 schedule space。action $\delta$ 直接对应 schedule primitive（如 `"split_loop"`, `"reorder"`, `"fuse"`），combining symbolic compiler 的可验证性与 LLM 的 planning。

这其实是 [TVM AutoTVM](https://arxiv.org/abs/1805.08166) + LLM planner 的 hybrid。TVM 的 schedule space 是离散的、可枚举的，LLM 在这个空间做 planning 比在 raw CUDA space 做规划更结构化。

### 5. Multi-agent world model

当前单 LLM 维护 $S_t$。可以考虑多 LLM 各维护子树（类似 [island model in evolution](https://en.wikipedia.org/wiki/Island_model)），用 ensemble $V$ 做 action selection，降低单 LLM prior 的 bias。这跟 [AlphaEvolve 的 distributed setup](https://arxiv.org/abs/2506.13131) 有点像，但 AlphaEvolve 是 distributed archive of programs，K-Search 可以是 distributed archive of intents。

### 6. Active learning 的角度

LLM 可以主动选"信息量最大"的 action，而不仅 max $V$。类似 UCB：$\arg\max_a V(a) + \beta \cdot \text{uncertainty}(a)$。或者 information gain：$\arg\max_a V(a)(1-V(a))$，挑那些"50/50 可能性"的 action，evidence 最 informative。这是 curiosity-driven exploration 的思路，参考 [ICM](https://arxiv.org/abs/1705.05363) 在 RL 里的用法。

### 7. 跟 AlphaEvolve / FunSearch 的本质区别

| 维度 | FunSearch / AlphaEvolve / OpenEvolve | K-Search |
|---|---|---|
| Search space | Program space (raw code) | Intent space (natural language) + Program space (for instantiation) |
| LLM role | Stochastic code generator | World model + Code policy |
| Belief state | Archive of programs（隐式） | Explicit search tree with $V$（显式） |
| Non-monotonic handling | 无（差就丢） | 有（stagnation counter + intent 保留） |
| Compositional reasoning | 无（每个 candidate 独立） | 有（tree 结构表达 dependency） |
| Prior accumulation | 无（每次从 prior sample） | 有（in-context learning of $S_t$） |

这个对比表是理解 K-Search 贡献的最清晰方式。

---

## 局限

### 1. Workload-agnostic action

GQA decode 上 K-Search 在 batch_size=1 的 workload 不如 baseline。因为 action 没有 workload regime tag——K-Search 倾向选 split-K（对大 batch 好），但 split-K 的 coordination overhead 对 batch_size=1 不可摊销。

修正：action 扩展为 $(x_{\text{parent}}, \delta, w)$，$w$ 是 workload regime（如 `"small_batch"`、`"large_batch"`）。或者维护多棵 search tree，per regime 一棵。

### 2. In-context only co-evolution

当前 world model evolution 纯靠 in-context learning。长 search 后 context window 可能饱和。可以考虑：
- Structured memory + retrieval（只 retrieve relevant past trials）
- 在 LLM 外部维护一个 learned $V$ function（small NN），LLM 输出 feature，NN 输出 calibrated $V$

### 3. API 成本

TriMul 用 300 iterations 调 GPT-5.2 + Gemini-3-Pro，成本不低。对比 [Kevin](https://arxiv.org/abs/2507.11948) 的 multi-turn RL fine-tune small model，每 iter 成本低很多。K-Search 的优势是不用 fine-tune 就能适应新硬件，代价是每 iter 贵。混合方案：用 K-Search 做 cold start（少量 iter 找到 good structure），再 fine-tune small model 在 good structure 附近做 local refinement。

### 4. 单 LLM prior 的 bias

LLM 可能对某类 optimization 有 systematic blind spot。比如如果训练数据里 FP8 相关 kernel 少，LLM 对 FP8 packing strategy 的 $V$ 估计可能不准。Ensemble 多个 LLM 或引入 external knowledge base 可以缓解。

---

## 最后的 takeaway

K-Search 的贡献可以浓缩成一句话：**在 LLM-based program synthesis 里，把 planning 和 implementation 解耦，让 LLM 的 domain knowledge 显式参与 search procedure 的 belief maintenance，而不是隐式地通过 code generation 间接发挥**。

这个 insight 不限于 GPU kernel。任何需要 multi-step structural reasoning + 噪声实现 的 optimization 问题都可能受益：
- Compiler optimization（[TVM schedule search](https://arxiv.org/abs/2006.06762)）
- Algorithm design（[FunSearch 的数学发现](https://www.nature.com/articles/s41586-023-06907-w)）
- System design（database query plan、network protocol）
- 甚至 codebase refactoring（[AlphaEvolve](https://arxiv.org/abs/2506.13131) 的扩展）

本质上，K-Search 是 **LLM as world model** 这个 paradigm 在 program synthesis 域的一次成功 instantiation。它验证了 LLM 的 planning capability 在 outer loop 里比在 inner loop 里更能发挥作用。

---

## Reference Links

- [K-Search repo](https://github.com/caoshiyi/K-Search)
- [OpenEvolve blog](https://algorithmicsuperintelligence.ai/blog/openevolve-overview/)
- [ShinkaEvolve](https://arxiv.org/abs/2509.19349)
- [AlphaEvolve](https://arxiv.org/abs/2506.13131)
- [FunSearch (Nature)](https://www.nature.com/articles/s41586-023-06907-w)
- [RAP: Reasoning via Planning](https://arxiv.org/abs/2305.14992)
- [World Models (Ha & Schmidhuber)](https://arxiv.org/abs/1803.10122)
- [MuZero](https://arxiv.org/abs/1911.08265)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [FlashInfer-Bench](https://arxiv.org/abs/2601.00227)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [TVM Ansor](https://arxiv.org/abs/2006.06762)
- [AutoTVM](https://arxiv.org/abs/1805.08166)
- [Triton](https://www.eecs.harvard.edu/~htk/publications/2019-tile-kung-cox.pdf)
- [CUTLASS / CuTe](https://github.com/NVIDIA/cutlass)
- [KernelBench](https://arxiv.org/abs/2502.10517)
- [Kevin (multi-turn RL for CUDA)](https://arxiv.org/abs/2507.11948)
- [AutoTriton (RL)](https://arxiv.org/abs/2507.05687)
- [CUDA-L1](https://arxiv.org/abs/2507.14111)
- [TTT-Discover](https://arxiv.org/abs/2601.16175)
- [GPUMODE](https://www.gpumode.com/benchmarks)
- [SGLang](https://arxiv.org/abs/2312.07104)
- [WebEvolver](https://arxiv.org/abs/2505.15716)
- [ICM (curiosity-driven RL)](https://arxiv.org/abs/1705.05363)
- [PlaNet](https://arxiv.org/abs/1811.04551)
- [Conformal Prediction](https://arxiv.org/abs/2107.07511)

---

# K-Search 深度解析：LLM 作为 Co-Evolving World Model 驱动 GPU Kernel 搜索

## 1. Motivation 与 Problem Framing

GPU kernel 优化是一个极度困难的 combinatorial optimization 问题。难度源于三个相互纠缠的 factors：

1. **Design space explosion**：tiling、memory layout、synchronization、architecture-specific primitives 的组合空间巨大（参考 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)）
2. **Hardware evolution**：Hopper → Blackwell 引入 TMA、WMMA、FP8 tensor cores 等新指令，过去的 optimal kernel 变 sub-optimal
3. **Evaluation cost**：每次 compile + profile 一个 kernel 需要秒级开销，搜索 budget 严格受限（参考 [Ansor OSDI'20](https://arxiv.org/abs/2006.06762) 的 cost model motivation）

K-Search 把这个问题形式化为一个 **fixed-budget black-box optimization**：

$$
o = (s, p, m) = \mathcal{E}(x)
$$

变量含义：
- $x \in \mathcal{X}$：一个 kernel 程序（CUDA/Triton 实现）
- $\mathcal{E}: \mathcal{X} \to \mathcal{O}$：evaluator 函数，编译 + 正确性校验 + 性能 profiling
- $o \in \mathcal{O}$：observation tuple
- $s \in \{0, 1\}$：correctness flag（是否通过 numerical 校验）
- $p \in \mathbb{R}^+$：performance metric（latency，单位 µs）
- $m$：metadata（compiler logs、profiler output、寄存器占用等）

Maximization objective 定义为相对 baseline 的 speedup：

$$
J(x) = s \cdot \frac{p_{\text{ref}}}{p} \cdot 100
$$

变量含义：
- $p_{\text{ref}}$：参考 SoTA baseline（FlashInfer）的 latency
- $p$：candidate 的 latency
- $s$ 作为乘法门：不正确则 $J = 0$，强制 correctness first
- $\cdot 100$：将 speedup 缩放到 0–100 量级，便于 LLM 数值比较

优化目标：

$$
x^\star = \arg\max_{x \in \mathcal{X}} J(x) \quad \text{s.t. budget } B
$$

K-Search 设置 $B = 120$ evaluations on FlashInfer，$B = 300$ on GPUMODE TriMul。

---

## 2. Baseline 的根本缺陷（Why Existing Evolution Fails）

OpenEvolve / ShinkaEvolve / FunSearch / AlphaEvolve 这一类方法都在 program space 直接做 evolution：

$$
x_{t+1} \sim \pi_{\text{LLM}}\left(x \mid \{(x_k, o_k)\}_{(x_k, o_k) \in C_t}\right)
$$

变量含义：
- $C_t \subseteq \mathcal{H}_t$：由 MAP-Elites / novelty search 等启发式选出的 context subset
- $\pi_{\text{LLM}}$：LLM 作为 stochastic code generator，直接 condition 在历史 raw text 上
- $o_k$ 序列化成 prompt 文本（compiler error、profiler log）

这个 formulation 的关键缺陷有四：

### 2.1 Intent 与 Implementation 强耦合
一个 high-level strategy（例如 "resolve bank conflicts via padding"）会被随机绑定到一个具体实现上。若该实现因 typo / 微小 syntax error compile 失败，整个 strategy 在 archive 中被丢弃。**Theoretical soundness 被实现噪声污染**。

### 2.2 缺乏 non-monotonic 优化路径处理
高性能 kernel 经常需要 multi-step structural transformation：先 refactoring memory layout，再 vectorization，再 split-K。中间步骤本身不 yield 立即 perf gain，反而可能 slowdown。Evolution heuristic 倾向于丢弃这些 intermediate state。

### 2.3 Context 压缩为 raw text 损失结构
把 $(x_k, o_k)$ 全部序列化进 prompt 后，LLM 无法 maintain 一个 explicit 的"哪些策略 promising"的结构化 belief。每次重新对 raw code 做 pattern matching。

### 2.4 Sample efficiency 差
实验中观察（Section 4.4 "Key observations"）：ShinkaEvolve 在 GQA logs 中**绝大多数 generations 得 0 分**（compile fail 或 incorrect）。OpenEvolve 在 MoE 上 mean final score 仅 ~3，挣扎在 low-accuracy regime。Search budget 大量浪费在扩展 invalid candidates。

---

## 3. K-Search 的核心思想：Decoupling Planning from Implementation

K-Search 的核心 insight：**LLM 的 prior knowledge 足以充当 intrinsic world model**，可以用来 maintain 一个 explicit 的 search tree，并在其中做 planning，而 code generation 只是 search tree 节点的 instantiation。

引用的工作链：
- [Ha & Schmidhuber, "World Models" (2018)](https://arxiv.org/abs/1803.10122)：M（latent dynamics model）+ RNN controller
- [Hao et al., "RAP: Reasoning via Planning" (EMNLP 2023)](https://arxiv.org/abs/2305.14992)：LLM as world model + MCTS-style reasoning
- [Guan et al., NeurIPS 2023](https://arxiv.org/abs/2305.14992)：从 language induce PDDL 做 classical planning
- [Fang et al., "WebEvolver" (EMNLP 2025)](https://arxiv.org/abs/2505.15716)：co-evolving world model for web agents

K-Search 把这套思路搬到 program synthesis 域。

### 3.1 Search State $S_t$ 形式化

$S_t$ 包含三部分：
- History of explored actions 与它们的 performance
- Frontier $\mathcal{A}(S_t)$：当前待探索的 Open actions 集合
- Priority function $V: \mathcal{A}(S_t) \to [0, 1]$：world model 对每个 frontier action 的 estimated potential

World model 表示为 state transition distribution：

$$
P_{\text{model}}(S_{t+1} \mid S_t, a_t)
$$

变量含义：
- $S_t$：当前 search state snapshot
- $a_t$：执行的动作（intent + parent program）
- $P_{\text{model}}$：由 LLM 通过 in-context learning 实现的分布

### 3.2 Action 的结构

一个 action 显式拆为：

$$
a_t = (x_{\text{parent}}, \delta)
$$

- $x_{\text{parent}}$：base program（已被 evaluator 验证的某个 ancestor）
- $\delta$：natural language optimization intent，例如 `"fuse head"`、`"register-resident rescaling"`、`"chunk32 vectorized"`

这层抽象把 "what to do"（high-level）与 "how to do it"（low-level CUDA 代码）分离。一个 $\delta$ 可以由 LLM 尝试多次 $\pi_{\text{code}}$ 采样实现，直到 stagnation。

### 3.3 三阶段迭代

K-Search 每个 round（即一次 evaluator 调用，消耗一单位 budget）：

#### Phase 1: Action Selection

$$
a_t = \arg\max_{a \in \mathcal{A}(S_t)} V(a \mid S_t)
$$

LLM 在 frontier 中选择 priority 最高的 action。$V$ 由 world model 推理时估计，是一个 scalar ∈ [0,1]。

#### Phase 2: Program Instantiation (Local Refinement)

$$
x_t \sim \pi_{\text{code}}(x \mid a_t), \quad o_t = \mathcal{E}(x_t)
$$

$\pi_{\text{code}}$ 是 LLM 作为 stochastic policy。重复采样直到 **stagnation**：连续 $K$ 次（K-Search 默认 $K=7$，TriMul 任务 $K=5$）无 improvement。在此过程中保留 $x_{\text{best}}$：

```python
n = 0
x_best, o_best = None, None
while B > 0 and n < K:
    x ~ pi_code(. | a_t)
    o = E(x); B -= 1
    if J(x) > J(x_best):
        x_best, o_best = x, o
        n = 0          # reset
    else:
        n += 1
```

这一步至关重要——**它实现了实现噪声的鲁棒性**：一个理论上 promising 的 strategy $\delta$，即便第一次 code sample 因 typo 失败，仍然会被反复采样直到 $K$ 次失败。这避免了 baseline 的"一败涂地"问题。

#### Phase 3: World Model Co-Evolution

$$
S_{t+1} \sim P_{\text{model}}(S \mid S_t, a_t; x_t, o_t)
$$

LLM 分析 trajectory $(a_t, x_{\text{best}}, o_{\text{best}})$ 后，对 search tree 执行三类 **Tree Edit Operations**：

- **Insert**：propose 新的 child nodes（新 optimization intent），并赋初始 $V$
- **Update**：根据新证据 re-evaluate 既有 frontier nodes 的 $V$（例如 figure 中 $u_{11}$ 从 $0.9 \to 0.6$）
- **Prune**：永久删除 infeasible / redundant 分支（例如 figure 中 $u_{10}$）

注意当前版本 K-Search 的 co-evolution 仅通过 **in-context learning** 实现，no fine-tuning。LLM 被输入 history of observations（结构化的 (intent, score, feedback) tuples），由其 reasoning capability 推断下一个 state。

---

## 4. 系统架构图解析（Figure 1）

Figure 1 中的 search tree：

- **蓝色实线节点**：Closed nodes，已经完成 local refinement，附着 $x_{12}$ 等具体 program
- **橙色虚线节点**：Open/Frontier nodes（如 $u_{13}$），等待被 instantiate
- 每条 edge 对应一个 natural-language intent $\delta$
- $V$ 值在 Open node 上标注，随 co-evolution 动态变化

Three-phase 循环：
1. Selection：在 frontier 中取 $\arg\max V$，如选 $u_{11}$
2. Local Refinement：重复采样 $\pi_{\text{code}}$，直到 stagnation
3. Tree Edit：根据 outcome Insert / Update / Prune

这个架构与 [AlphaEvolve](https://arxiv.org/abs/2506.13131) 的 program database 概念对比：AlphaEvolve 维护 archive of programs，由 evolution heuristic 选 parents；K-Search 维护 search tree of intents，由 LLM 维护的 $V$ function 选 actions。

---

## 5. Case Study：MLA Paged Decode 搜索轨迹（Figure 2）

这是理解 K-Search 行为最有价值的部分。Round 1 到 Round 102 的演化：

### Round 1：初始 frontier
$S_0$ 中 LLM 注入三个 high-level actions：
- `fused_multi_head`（V 最高）
- `split_k_decoding`
- `independent_heads`

World model 推理："processing shared CKV heads together will reduce global memory traffic by 16× compared to independent processing"，因此赋予 `fused_multi_head` 最高 $V$。

### Round 14–34：拓扑演化
执行 `fused_multi_head` 得 $J=34$。LLM 据此：
- **Insert** `register_resident_rescaling`、`occupancy_tuned_chunk32` 深化成功分支
- **Update** `independent_heads` 的 $V$（下调，因 head fusion proven 有效）
- **Prune** `independent_heads`（round 34）永久删除

这是一个关键的"belief sharpening"过程：world model 通过 observation 推断出 sibling action 在新 evidence 下已 suboptimal。

### Round 42–102：结构性 insight
Round 42 出现**非平凡的结构变化**：LLM **deletes** 根层的 `split_k` action，**re-Inserts** `low_overhead_split_k` 到 `register_resident` 分支深处。这反映一个 learned insight：split-K 单独作为 baseline 无效，但作为 strong fusion kernel 之上的 composable optimization 高度有效。

随后 `chunk32_vectorized` 成功，LLM 推断可以在 load Q 时立即 apply `sm_scale`，propose `chunk32_prescale_vectorized`——这个 refinement 在 round 102 yield **全局最优**（star marker）。

**Takeaway**：K-Search 不在 sparse raw code space 枚举；它从 high-level intent 起步，local refinement 过滤 coding noise，让 world model 的 understanding 与 kernel 进度 co-evolve。它能 prune dead ends 并 **dynamically reposition strategies**。

---

## 6. 实验结果深度分析

### 6.1 主结果（Figure 3a, Table 综合）

四个 kernel 上 120 iterations 平均 final score：

| Kernel | K-Search | OpenEvolve | ShinkaEvolve | K/OpenEvolve | K/ShinkaEvolve |
|---|---|---|---|---|---|
| GQA decode | 76.0 | 44.2 | 27.7 | 1.72× | 2.74× |
| MLA decode | 47.1 | 39.9 | 34.7 | 1.18× | 1.36× |
| MLA prefill | 57.4 | 19.5 | 11.3 | 2.95× | 5.10× |
| MoE (FP8) | 44.1 | 3.09 | 27.9 | **14.3×** | 1.58× |
| **Overall** | **56.13** | **26.68** | **25.37** | **2.10×** | **2.21×** |

MoE 上 14.3× improvement 极其引人注目。OpenEvolve 在 MoE 上 stuck 在 3.09，因为 MoE 涉及 irregular data-dependent routing、load balancing、FP8 packing，是 multi-step coordinated transformation 的典型场景——baseline 的 "evolution in raw program space" 完全无法 navigate。

### 6.2 Per-workload 分布（Figure 3b）

跨 152 个 workload traces，K-Search 在绝大多数 workload 上超过 baseline。**值得注意的 failure case**：在 GQA decode 的某些 small-batch workload（batch_size=1 的 16 个、batch_size=16 的 4 个），K-Search 不如 baseline。

原因分析（Section 4.4）：K-Search 的 kernel 采用 split-K parallelism，将 KV sequence 切分到 multiple thread blocks，需要 lightweight counter 做 reduce。**Coordination overhead** 对 small batch 不可摊销；baseline 用 single-block-per-batch 设计，对 batch_size=1 反而更优。

这是一个有趣的 trade-off：K-Search 的 LLM 在 search tree 中倾向于 split-K（对 large batch 更优），但搜索过程未充分 explore small-batch-specific strategy。这暗示 search frontier 中可能需要**workload-conditioned action**（让 action 不仅 $(x_{\text{parent}}, \delta)$，还包含 workload regime tag）。

### 6.3 GPUMODE TriMul SOTA（Table 3）

| Submission | Lang | Model | Iter. | Latency (µs) |
|---|---|---|---|---|
| shiyegao | CUDA | (human) | — | 1074 |
| Zeyu Shen | Triton | (human) | — | 1140 |
| TTT (TTT-Discover) | Triton | GPT-OSS-20B + RL | 25,600 | 1161 |
| **K-Search** | **Triton** | **GPT-5.2 + Gemini-3-Pro** | **300** | **1030** |

K-Search 用 300 iterations（远少于 TTT-Discover 的 25600）超过人类与 RL 方法。配置：K=5（Triton 简化），前 150 步用 GPT-5.2，后 150 步从 GPT-5.2 的 best 解出发继续用 Gemini-3-Pro。

TriMul 任务涉及 LayerNorm、5 个 gated linear projections、$\mathcal{O}(N^3)$ pairwise contraction、gated output projection。这是一个 multi-stage kernel，恰好是 K-Search 的强项：high-level intent 可以分别针对每个 stage 做规划。

参考 [GPUMODE leaderboard](https://www.gpumode.com/benchmarks) 与 [TTT-Discover](https://arxiv.org/abs/2601.16175)。

---

## 7. 生成的 Kernel 深度技术分析

### 7.1 FP8 MoE Kernel (Blackwell)

DeepSeek-V3 风格：top-8 routing，32 local experts，hidden size 7168，每个 token 在 256 candidate experts 中选 top-8。

**K-Search 的 routing**：
- 每 token 一个 thread block（256 threads）
- Warp 内用 `__shfl_down_sync` 做 warp-level reduction 找 top-8
- 保持 warp 并行，避免 serialization

**对比 OpenEvolve**：persistent kernel + atomicAdd 取 tile index → 在 while loop 中 atomic 严重 overhead

**对比 ShinkaEvolve**：每 thread 串行 load 256 scores + for-loop 找 top → 极慢

**K-Search 的 FFN**：
- Pipeline: routing → sort-scatter（按 expert reorder tokens，contiguous memory）→ gate+up → SiLU → down
- 使用 **WMMA tensor cores** 在 $16 \times 16$ blocks 上
- **Double buffering**：load next block 与 compute current 重叠
- **Skip empty experts**：若某 expert 0 tokens 则跳过

**对比**：OpenEvolve 用 persistent single kernel 但 shared memory 占用大、occupancy 低；ShinkaEvolve 不用 tensor cores，纯 dot product，性能差。

### 7.2 GQA Paged Decode (Hopper)

**K-Search 的并行性**：
- Split sequence across blocks：每 block 处理 KV chunk，写 partial result 到 temporary buffer
- 通过 lightweight counter 检测 last block → reduce 合并
- 长 sequence 时多 block 并行不同 segment

**对比 baseline**：单 block 处理整个 KV sequence，长 sequence 时无法 exploit parallelism。

**K-Search 的 memory/compute overlap**：
- Double-buffered chunks：current chunk compute 与 next chunk fetch 重叠

**对比 ShinkaEvolve**：单 buffer 串行 → 性能损失。

### 7.3 MLA Paged Decode (Hopper)

MLA：每 token query 是 576-dim（512 CKV + 64 KPE）。

**K-Search 的策略**：
- **Adaptive split**：短 sequence 单 block 直写 output，避免 reduce pass；长 sequence 多 split
- **Register-resident Q**：每 token Q 是 $16 \times 576$，小但每 chunk reuse。K-Search 把 Q 加载到 register fragments，**不在 shared memory materialize**。降低 per-block shared memory 压力
- **Deeper prefetch pipeline**：load **two chunks ahead**（chunk $i$ compute + chunk $i+1$ loading + chunk $i+2$ issuing load），比标准 double buffering fuller

**对比 baseline**：Q 在 SRAM 中 materialize；标准 double buffering；chunk size 固定 64 token。

### 7.4 MLA Paged Prefill (Hopper)

576-dim query（512 CKV + 64 ROPE），causal attention，16 heads。

**Variable-length batch 处理**：
- 16-row tile，但 tile 可能跨 sequence boundary
- K-Search：每 block 处理 contiguous 16-row tile，**on-GPU 动态 resolve split**（用 prefix-sum array of sequence boundaries 找各 segment）
- 对每 segment fetch 对应 KV-cache range + compute attention
- 移动到下一 segment 直到 tile 完成

**对比 baseline**：CPU 端 precompute tile-to-batch mapping → 额外 CPU pass + 额外 memory。

**Score computation**：
- K-Search：所有 thread block 内 threads 联合 compute score matrix + softmax per row
- OpenEvolve：仅单 warp 做 score-softmax，其余 idle → GPU 利用率低

---

## 8. 与 RL-based 方法的对比

paper 中提到 [Kevin (Baronio et al., 2025)](https://arxiv.org/abs/2507.11948)、[AutoTriton (Li et al., 2025b)](https://arxiv.org/abs/2507.05687)、[CUDA-L1 (Li et al., 2025c)](https://arxiv.org/abs/2507.14111) 等 RL 方法 fine-tune 模型做 one-shot generation 或 local refinement。

K-Search 的关键差异：**no fine-tuning**，纯 in-context learning 实现 world model co-evolution。优势：
- 适应新硬件无需 retrain
- 可灵活切换 frontier LLM（GPT-5.2 → Gemini-3-Pro）

但代价：每 round 的 LLM 推理成本远高于 fine-tuned small model 的 inference。TriMul 用 300 iterations 调用 GPT-5.2/Gemini-3-Pro 的 API 成本不可忽视。RL 方法（如 TTT-Discover 用 25600 iter）虽然 iteration 多但每 iter 用 GPT-OSS-20B 本地推理。

---

## 9. 直觉构建：为什么 Co-Evolving World Model 是关键？

让我把 intuition 压缩成几个对比：

### 9.1 Baseline 视角
LLM 是一个 **black-box function** $f: \text{prompt} \to \text{code}$。Evolution algorithm 在 outer loop 决定 prompt 内容。LLM 内部对 kernel 优化的"理解"是 **opaque** 的，每次调用从 prior 重新 sample。

### 9.2 K-Search 视角
LLM 被显式置于两个 roles：
1. **World model**：维护 $S_t$ 的 belief，推理 $V$，决策 Insert/Update/Prune
2. **Code policy** $\pi_{\text{code}}$：给定 $a_t$ instantiate 程序

LLM 的 prior knowledge 在 world model role 中被 **structured activation**：它不再 implicit 理解 split-K 何时有效，而是**显式产出** $V(\text{split-K} | S_t) \in [0,1]$ 这个标量。这把 LLM 的 implicit knowledge **project 到 explicit planning space**。

### 9.3 Co-evolution 的意义
$S_t$ 不是固定 database，而是 LLM 自己维护的可演化 belief。新 observation $o_t$ 进来后，LLM 重新审视所有 frontier actions 的 $V$，可能：
- 上调某 sibling（因为它在新 context 下变 promising）
- 下调某 active 节点（新证据说明它 suboptimal）
- 完全 prune 某分支（belief collapse）
- Insert 一个之前未考虑的 intent（belief expansion）

这模拟了人类 kernel 工程师在调试时反复修正 mental model 的过程。

### 9.4 非单调路径的 navigability
Local refinement 的 stagnation 机制是关键。假设某 intent $\delta$ 第一次实现 compile 失败，第二次无 perf gain，第三次成功 yield 2× speedup。Baseline 会丢前两次；K-Search 保留到 stagnation $K=7$，给 LLM 多次尝试**同一 high-level intent 的不同 implementation**。这使得**意图层面的探索**对**实现层面的噪声**鲁棒。

---

## 10. 局限与未来方向

### 10.1 显式局限
1. **Workload-agnostic action**：GQA decode 中 K-Search 在 batch_size=1 上 underperform，因为 action 没有 workload regime tag。可扩展 action 为 $(x_{\text{parent}}, \delta, w)$，$w$ 标记适用 workload
2. **In-context only co-evolution**：长 search 后 context window 可能饱和；可考虑 structured memory + retrieval
3. **No fine-tuning**：API 成本高；可与 [Kevin 的 multi-turn RL](https://arxiv.org/abs/2507.11948) 结合做 hybrid

### 10.2 延伸思考

**World model 的可验证性**：当前 $V$ 是 LLM 输出的 scalar，但**没有 ground truth 验证 $V$ 是否 calibrated**。可考虑用 historical (intent, final_score) pair 训练一个 calibration head，类似 [RAP 的 reward model](https://arxiv.org/abs/2305.14992)。

**Search tree 的 transfer**：同一架构（如 Hopper）的 kernel optimization knowledge 可否 transfer 到新 kernel？类似 AlphaEvolve 的 program database 但跨任务。这是通向 [Open-ended evolution](https://arxiv.org/abs/2509.19349) 的方向。

**Multi-agent world model**：当前单一 LLM 维护 $S_t$。可考虑多个 LLM 各自维护子树（类似 island model in OpenEvolve），用 ensemble $V$ 做 action selection，降低单 LLM prior 的 bias。

**与 DSL 编译器集成**：K-Search 当前在 CUDA/Triton 程序空间搜索。可把它接到 [CuTe layout abstractions](https://github.com/NVIDIA/cutlass) 或 [TVM Ansor](https://arxiv.org/abs/2006.06762) 的 schedule space，让 action $\delta$ 直接对应 schedule primitive，combining symbolic compiler 的可验证性与 LLM 的 planning。

**World model 作为 active learner**：LLM 可主动选择"信息量最大"的 action（不仅 max $V$，而是 max information gain $V \cdot (1 - V)$ 类似 UCB），实现 curiosity-driven exploration。

---

## 11. 总结直觉

K-Search 的关键贡献：**把 LLM 从 code generator 提升为 search procedure 的 intrinsic planner**。

公式本质：

$$
\max_{a_1, a_2, \dots, a_T} \mathbb{E}\left[J(x_T)\right] \quad \text{s.t. } \sum_t \text{cost}(a_t) \leq B
$$

其中 $a_t$ 在 K-Search 中是 (parent, intent) 元组，而不是 raw code。这个 action space 的**语义抽象**是性能突破的根因。Co-evolving world model 让 LLM 的 prior 在搜索中 **持续 sharpen**，避免 baseline 的"prior 沉睡在 prompt 中"浪费。

14.3× MoE gain、TriMul SOTA、平均 2.1× improvement——这些数字验证了一个 deep insight：**LLM 的 planning capability 是 underutilized resource**，在 evolution loop 中应被显式 instantiate，而非通过 stochastic code generation 间接发挥。

---

## Reference Links

- **K-Search repo**: https://github.com/caoshiyi/K-Search
- **OpenEvolve**: https://algorithmicsuperintelligence.ai/blog/openevolve-overview/
- **ShinkaEvolve**: https://arxiv.org/abs/2509.19349
- **AlphaEvolve**: https://arxiv.org/abs/2506.13131
- **FunSearch (Nature)**: https://www.nature.com/articles/s41586-023-06907-w
- **RAP: Reasoning via Planning**: https://arxiv.org/abs/2305.14992
- **World Models (Ha & Schmidhuber)**: https://arxiv.org/abs/1803.10122
- **FlashInfer**: https://github.com/flashinfer-ai/flashinfer
- **FlashInfer-Bench**: https://arxiv.org/abs/2601.00227
- **FlashAttention-3**: https://arxiv.org/abs/2407.08608
- **DeepSeek-V3**: https://arxiv.org/abs/2412.19437
- **TVM/Ansor**: https://arxiv.org/abs/2006.06762
- **Triton**: https://www.eecs.harvard.edu/~htk/publications/2019-tile-kung-cox.pdf
- **CuTe / CUTLASS**: https://github.com/NVIDIA/cutlass
- **KernelBench**: https://arxiv.org/abs/2502.10517
- **Kevin (Multi-turn RL for CUDA)**: https://arxiv.org/abs/2507.11948
- **AutoTriton (RL)**: https://arxiv.org/abs/2507.05687
- **CUDA-L1**: https://arxiv.org/abs/2507.14111
- **TTT-Discover**: https://arxiv.org/abs/2601.16175
- **GPUMODE**: https://www.gpumode.com/benchmarks
- **SGLang**: https://arxiv.org/abs/2312.07104
- **WebEvolver (co-evolving world model for web agents)**: https://arxiv.org/abs/2505.15716
- **Guan et al. (LLM → PDDL)**: https://arxiv.org/abs/2305.14992
