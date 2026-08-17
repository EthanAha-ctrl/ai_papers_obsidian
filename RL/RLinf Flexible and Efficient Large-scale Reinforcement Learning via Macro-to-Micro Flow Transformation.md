---
source_pdf: RLinf Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro
  Flow Transformation.pdf
paper_sha256: b4f26818a69edf2cb55c4fcb1cc4ab9d0d753205c83a1fc08c42e4b42c5a068a
processed_at: '2026-08-12T00:05:31-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RLinf 用人话讲

Andrej，用大白话再过一遍。我换个角度，从"这东西到底解决什么烦人问题"讲起。

## 一句话版本

RL training 这事，generation、inference、training 三步流水线，老的 system 要么全挤一组 GPU 上排队跑（collocated），要么拆开各占一组 GPU 流水跑（disaggregated）。现实里这两种都不够好。RLinf 干的事：你只管写"先 rollout 再 inference 再 training"这个 high-level 逻辑，它自动帮你决定谁跟谁挤 GPU、谁单独占 GPU、pipeline 每次 pass 多大数据。

## 这事为什么是个问题

你想跑个 7B 模型的 math reasoning RL。一个 batch 比如 512 个 prompt，rollout 阶段同时生成 512 条 response。问题来了——response 长度 dynamic，有的题 200 token 解完，有的题 3000 token 还在絮叨。Figure 2 的 CDF 显示，**95% 的 response 在前 80% 时间内就生成完了，剩下 5% 的"长尾"硬是把整个 rollout 阶段拖了 20% 时间，这期间大部分 GPU 干等**。

这就是 collocated 模式（veRL 那套）的痛点。generation 和 training 共享 GPU，generation 不结束 training 就上不来。

那 disaggregated 模式呢？generation 占 64 卡，training 占 64 卡，generation 完一部分 training 就能开始，流水起来。听起来不错，但新问题：generation 太慢时，training 那边 GPU 闲着等数据。而且 generation 用 GPU 跟 training 用 GPU 的方式完全不一样——generation 是 memory bandwidth bound（矩阵 × 向量，auto-regressive decode），training 是 compute bound（forward + backward）。把这两种 workload 放在不同 GPU 上各跑各的，硬件利用率很难平衡。

**两种 mode 都不是 universal optimal。**这是 paper 的核心 motivation。

## Embodied RL 让事情更糟

看 Figure 3 的 profile，ManiSkill 这种 embodied 环境：
- Simulator：跑的 step 数越多，CPU memory 用得越多，但 GPU compute 利用率 < 24%
- Generation：batch size 越大，时间线性增长，GPU util > 70%
- Training：memory 吃得最狠，但跑得比 generation 快 3 倍

这种异构性意味着：simulator 应该尽量多开 instance 占一组 GPU 跑；training 不需要那么多 GPU 但 memory 大；generation 又是另一回事。**没有一种 execution mode 能把这些都装下**。

veRL 那种 collocated 不行——simulator 和 generation 撑爆 GPU memory。Open-RLHF 那种 fully disaggregated 也不行——training 单独占 GPU 闲得慌。你需要 **hybrid**：simulator + generation 在一组 GPU 上 pipeline，跑完了 offload 到 CPU，training 接管这组 GPU。

但 hybrid 这种东西，你让用户手写代码切换 mode？噩梦。每种 mode 的 program structure 和 communication pattern 都不一样。collocated 是 phase-level 粗粒度，disaggregated pipelining 是 batch-level 细粒度，hybrid 还要管 context switch。**改 mode 就要重写代码，这就是现有 system 的死结**。

## M2Flow 怎么破这个结

类比一下：compiler 把 Python 代码编译成机器码。你写的是 high-level 逻辑（`for x in list: sum += x`），编译器自动把它 lowering 成 SIMD 指令、unroll loop、register allocation。你不需要为不同 CPU 架构重写 Python 代码。

M2Flow 是同一个思路，只不过作用域是 RL workflow：

- **Macro flow**：你用 imperative Python 写"rollout → inference → training"，类似 Figure 5 那个例子，不到 100 行
- **Micro flow**：RLinf 把它"编译"成具体执行计划——哪些 worker 放哪些 GPU、什么时候 context switch、pipeline 每次 pass 多少 batch

你写的 macro flow 是 **placement-agnostic** 的，system 帮你 explore 整个 scheduling space。

## 三个关键机制，用大白话

### 1. Worker Abstraction + Device Lock

每个 RL component 封装成 Worker，必须实现 `onload()` 和 `offload()` 两个函数——就是"上 GPU"和"下 GPU"。

有个叫 `device_lock` 的 distributed lock。Worker 要用 GPU 前 acquire lock，用完 release。Lock 的状态全局一致，atomic 改。

聪明的地方：这个 lock 知道 data dependency。Rollout worker 在 generate 还没 enqueue 数据前，training worker 不能抢 lock——避免空转等待。Rollout enqueue 一批数据 release lock 后，training 才能 acquire。

所以 context switch 变成自然语义：拿锁→onload→算→release→offload，而不是手动管 GPU memory。

### 2. Elastic Pipelining

普通 pipeline 是固定 batch size，stage 1 跑完 batch 才进 stage 2。Elastic pipelining 的观察：RL 里大部分 worker 是 SPMD pattern，多大 batch 都能跑。SGLang/vLLM 这种 serving engine，单 prompt 或 list of prompts 都能处理。

所以可以调 "dequeue granularity"：
- 细粒度：rollout 每生成 1 个 response 就 enqueue，inference 立刻 dequeue，pipeline bubble 几乎为零
- 粗粒度：攒满 32 个再 enqueue，通信 overhead 小但 bubble 大

Scheduler 根据 profiling 选最优 granularity。这是"弹性"的含义。

### 3. Scheduling Policy（Algorithm 1）

核心是 dynamic programming on DAG。把 workflow graph 用 s-t cut [Ford-Fulkerson 那个经典 max-flow min-cut 算法] 递归切成两半，每半评估两种 cost：

- **Temporal cost**（共享 GPU）：$T_s + T_t$，两个 worker 串行跑 + offload overhead
- **Spatial cost**（拆 GPU）：pipeline 时间

Spatial cost 的公式：
$$T_{spatial} = T_{critical} + (M/m - 1) \times T_{bottleneck}$$

变量讲清楚：
- $T_{critical}$：pipeline warm-up + cool-down 时间
- $M$：总 batch size
- $m$：pipeline 每步处理的 batch size
- $M/m$：pipeline 总 step 数
- $M/m - 1$：steady state step 数（减掉 warm-up 第一步）
- $T_{bottleneck}$：最慢 stage 的时间

这个就是经典 pipeline parallelism 公式，paper 把它从 layer-level 扩到 worker-level。选 cost 最小的 cut，递归切下去，直到每个 subgraph 是单 node。

Cycle（embodied RL 那种 rollout ↔ environment 互馈）怎么处理？预处理把 cycle collapse 成一个超级 node，内部均匀 partition 到所有 GPU。这是 trade-off——避免组合爆炸但可能 sub-optimal。

## Scheduling Space 三种代表模式

Figure 7 三张图说明：

**纯 Temporal**：所有 worker 共享所有 GPU，排队跑。适合"单 worker 必须用所有 GPU 才能跑起来"的场景，比如大模型 training。坏处是 long-tail 卡 GPU。

**纯 Spatial**：worker 分到不同 GPU，pipeline。适合 stage 时间能 align 的情况。坏处是 rollout 慢时 training GPU 闲置。

**Hybrid**：部分 pipeline + 部分 context switch。比如 simulator + generation 在一组 GPU 上 pipeline 跑完，offload 到 CPU，training 接管这组 GPU。这是 paper 的精髓。

## 实验数据人话版

### Reasoning RL（math reasoning）

**Qwen2.5 GRPO**（Figure 8）：
- 1.5B on 64 GPU：RLinf 比 veRL 快 1.10×
- 7B on 128 GPU：快 1.58×
- 32B on 256 GPU：快 1.40×

注意，这里 RLinf 用的是 **temporal mode**，跟 veRL 设计思路类似，但仍快这么多。原因：KV-cache 更大（memory 管理好）、inference stage 的 sync overhead 小。veRL 在 GPU 数量增多时 scaling 差，因为 inference 占比从 15% 升到 20%，它的 rollout engine 不够优化，KV-cache 逼着缩小，rollout 只能 sublinear 加速。

Spatial mode 在 GRPO 这里反而慢 44-69%，因为 GRPO 只有一个 model，rollout/inference/training 拆到不同 GPU 反而拖慢 rollout，长 sequence 让 training 等第一批 rollout 等太久。

**Qwen2.5 PPO**（Figure 10）：PPO 有 4 个 model（actor + reference + reward + critic），情况反过来：
- 1.5B on 16-64 GPU：RLinf-Spatial 比 veRL 快 35-70%
- 7B on 32-128 GPU：快 38.7-60.7%
- 14B：快 27.2-56.5%

PPO 偏 spatial mode，因为 4 个 model 可以 pipeline overlap。GRPO 偏 temporal，因为只有 1 个 model，spatial 浪费资源。**这正印证了 paper 的核心论点：没有 universal optimal mode。**

**Qwen3-30B-A3B MoE GRPO**（Figure 12）：对比 Slime。32 GPU 上 RLinf-Spatial 比 Slime-Colocate 快 31.2%。MoE 模型 rollout memory 大，spatial mode 让 rollout 能用更大 parallelism。

### Embodied RL

**ManiSkill**（OpenVLA，Figure 14a）：
- RLinf-Hybrid 比 Temporal 快 52-69%，比 Spatial 快 60-87%

这是 Hybrid mode 的胜利。Simulator 是 memory-bound compute-light，要 dedicate GPU 跑多 env；rollout+training 要 share GPU 做 context switch。

**LIBERO**（OpenVLA-OFT，Figure 14b）：
- RLinf-Temporal 比 SimpleVLA-RL 快 37.8-143.4%

这里 Temporal 反而赢 Hybrid，因为 LIBERO 是 CPU-intensive，把环境 restrict 到一组 GPU 反而浪费 CPU core。又一次证明"没有 universal mode"。

### Model Performance

**Math reasoning**（Table 1）：
- RLinf 1.5B：AIME24 48.44（base 28.33，+20 分），avg 40.84，1.5B 量级 SOTA
- RLinf 7B：avg 56.23，超 Skywork/Polaris/AceMath

**LIBERO**（Table 3）：
- OpenVLA-OFT baseline：avg 34.33%
- RLinf RL 后：avg 97.83%

最炸裂的是 Long task：从 9.68% 跳到 94.35%。Long horizon RL 一直是个老大难，这个 jump 说明 stable training infra 对 long-horizon RL 的 enabling 作用。

## 关键洞察

### Insight 1: Programmability 与 Efficiency 的 trade-off 被 M2Flow 破了

传统 system design 假设："你要 flexible 编程，就牺牲 efficiency；要 efficiency，就要 hardcoded optimization"。M2Flow 通过 decouple logical/physical flow，让用户写 clean 的 macro flow，system 在 micro flow 上 explore 优化空间。**这是 system design 范式转变**，跟 Alpa（OSDI'22）做 inter/intra-op parallelism 分离、PyTorch 2.0 做 eager/graph 分离是同一脉络。

### Insight 2: Hybrid mode 是"被遗忘的中间地带"

Collocated 和 disaggregated 之间有大量 hybrid 配置空间，现有 system 没法 explore。RLinf 通过 elastic pipelining + context switching 把这个空间打开了。这就像 CPU 调度的 evolution——从 batch processing（collocated）到 time-sharing（disaggregated）到 multi-level feedback queue（hybrid）。

### Insight 3: Profiler-guided search 是 practical 的

理论上 RL workflow scheduling 是 NP-hard（DAG partition + bin packing）。但 RL workflow 节点数 < 10，GPU 数 8-1024，dynamic programming 在 5 秒内能搜完。Figure 16b 显示 exponential growth 但仍在可接受范围。这印证了 MLSys 一个 recurring wisdom：**practical instance 比理论复杂度重要**。

## 局限性（说说缺点）

1. **Cycle handling 粗糙**：把 cycle collapse 成单 node，内部均匀 partition。Embodied RL 的 long-horizon rollout（80 steps environment interaction）内部 scheduling 没被 explore。比如是否可以在 rollout 中间插入 training update 做 async RL？paper 没做。

2. **One-shot profiling**：response length distribution 会随训练漂移（curriculum learning、model 变强）。Profile 一次后 E 函数会失效。paper 没讨论 online re-profiling 或 adaptive scheduling。

3. **Search 复杂度**：8-1024 GPU 下 < 5s，10000+ GPU 会怎样？Figure 16b 是 exponential。万卡集群可能需要 hierarchical search 或 ML-based policy。

4. **Single controller**：所有 function invocation 都经过 Controller dispatch。大规模下是 potential bottleneck。Ray actor model 本身是 decentralized，但 RLinf 在上面加了 central controller 层。

5. **MoE 优化不足**：Qwen3-30B-A3B 在 128 GPU 上 spatial mode 反而不如 temporal，paper 说 "rollout and training do not overlap well"。MoE 的 expert routing dynamic 让 pipeline balancing 更难，需要专门优化。

## 我的几个直觉联想

### 类比 1: M2Flow 像 LLVM IR

LLVM 把 high-level language（C++、Rust）编译成 IR，IR 再 lower 到不同 backend（x86、ARM、GPU）。M2Flow 把 high-level RL workflow 编译成 "scheduling IR"（worker + channel + lock），再 lower 到不同执行模式（temporal、spatial、hybrid）。Profiler 像 cost model，Scheduler 像 instruction scheduler。这个架构让"加新算法"和"加新硬件"正交解耦。

### 类比 2: Data Channel 像 OS 的 pipe

Unix pipe 把 producer 和 consumer 解耦，producer write 完就走，consumer 阻塞 read。Data channel 加了 GPU memory offloading 和 load-balancing，是"GPU-aware pipe"。这个 abstraction 让 elastic pipelining 成为可能——granularity 可以动态调整。

### 类比 3: Device Lock 像 database lock

Database 的 lock 有 shared/exclusive、有 deadlock detection、有 priority queue。Device lock 借鉴了这些，但加了 data dependency awareness——lock 不是按时间顺序，是按 data flow 顺序。这避免了 long-tail rollout 卡死 training 的 deadlock。

### 联想 4: 跟 RL algorithm 的关系

RLinf 只动 system，不动 algorithm。但 system 改变 algorithm 的可行性边界：
- Async RL（AReaL）在 disaggregated 系统上才能做，因为 sync barrier 弱
- Long-horizon RL（LIBERO Long task 9.68% → 94.35%）需要 stable infra
- Multi-turn agentic RL（Deep Research）需要 cyclic data flow 支持

RLinf 让这些算法的 system 实现更统一，未来 algorithm innovation 可以更专注 algorithm 本身。

### 联想 5: 跟 Ray 的关系

Ray 本身是通用 distributed framework，但 Ray 的 actor abstraction 太底层，不解决 RL workflow 的 specific 问题（GPU memory 共享、pipeline、context switch）。RLinf 在 Ray 上建了一层 RL-specific 的 scheduling layer。这跟 vLLM 在 PyTorch 上建 serving-specific layer 类似——domain-specific system 总比 general-purpose 在特定场景下 efficient。

### 联想 6: 未来方向

paper §7 说 "operating system for AI workloads"。我推测几个方向：

1. **Online re-profiling**：用 RL 学调度策略，类似 learned optimizer in database
2. **Multi-tenant**：多个 RL job 共享 cluster，类似 Kubernetes scheduler
3. **Hierarchical scheduling**：cluster-level + node-level + device-level M2Flow，类似 NUMA
4. **Speculative rollout**：rollout 时同时跑 inference 预测 logprob，response 被截断就 cancel
5. **Cross-modal scheduling**：CPU/GPU/NPU/robot arm 混合调度，paper 已经提 robot arm 作为 schedulable device
6. **Federated RL**：跨 cluster 的 RL training，M2Flow 扩展到 WAN

## 最终一句总结

**RLinf 让 RL training system 从"二选一"变成"自由组合"**。你写一遍 workflow 代码，system 帮你 explore 整个 scheduling space 找最优配置。1.07-2.43× speedup 是 surface，深层意义是 **programmability 和 efficiency 的 trade-off 被 decouple 解开了**，future RL algorithm 和 workload 不再被 system mode 限制。

对 Andrej 你这种关心 long-horizon agentic RL 的人，关键是：**RLinf 让 long-horizon RL training 变得 practically tractable**，LIBERO Long task 9.68% → 94.35% 就是证据。你之前担心 agentic training system 不够 flexible，这篇 paper 给了个像样的答案。

主要参考：
- RLinf repo: https://github.com/RLinf/RLinf
- veRL (主 baseline): https://github.com/volcengine/verl
- Slime (MoE baseline): https://github.com/THUDM/slime
- SGLang: https://github.com/sgl-project/sglang
- Ray: https://github.com/ray-project/ray
- OpenVLA: https://github.com/openvla/openvla
- SimpleVLA-RL: https://github.com/VoltronBot/SimpleVLA-RL
- ManiSkill: https://github.com/haosulab/ManiSkill
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- AReaL (async RL baseline): https://arxiv.org/abs/2505.24298
- DeepSeekMath GRPO: https://arxiv.org/abs/2402.03300
- Ford-Fulkerson algorithm: https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM

---

# RLinf 深度技术讲解

Andrej，这篇 paper 我读了三遍，从 system design 角度看，它确实代表了 2025 年 RL training system 的一个关键拐点：从 "rigid execution mode" 转向 "decoupled logical/physical flow"。下面我尽量 build your intuition。

## 1. Motivation: 为什么需要 M2Flow

### 1.1 RL Workflows 的异构性本质

paper 在 §2.1 把 RL workflow 拆成四类，但本质问题只有一个：**heterogeneous components with conflicting resource profiles**。

- **Generation (rollout)**：memory bandwidth bound（不是 compute bound），矩阵-向量乘，GPU utilization 经常 < 70%。响应长度 dynamic，导致 long-tail 问题。
- **Inference (logprob)**：prefill-only，相对 compute 密集但 batch 友好。
- **Training**：需要 gradients + optimizer states，memory 是 generation 的 3-4 倍（AdamW: $2\times$ params for moments + $1\times$ for grads + $1\times$ for params，FP16/BF16 + FP32 master copy 时更多）。
- **Simulator**（embodied）：CPU-bound physics simulation + GPU rendering pipeline，memory 随 environment 数量线性增长，但 GPU util < 24%。

paper Figure 3 给出了关键 profile：
- Simulator：execution time 随 batch size 几乎不变，但 memory 线性增长
- Generation：execution time 和 memory 都线性增长

这个 asymmetry 是 hybrid scheduling 的根本 motivation。

### 1.2 现有系统的二分法失效

veRL（HybridFlow, ASPLOS'25）走 collocated 路线，DeepSpeed-Chat 也类似；NeMo-Aligner / Open-RLHF / AReaL 走 disaggregated 路线。paper §2.2 的 Figure 2 给出了 7B GRPO 的 CDF：
- 95% 的 response 在 80% 时间内完成
- 剩余 5% 的 long-tail 拖累整个 generation phase

collocated 模式下，GPU 必须等最慢的 response 完成；disaggregated 模式下，rollout 和 training 各占 GPU，rollout 完不成就浪费 training GPU 的 idle time。**Neither is universally optimal**——这是 paper 的核心观察。

## 2. M2Flow 核心思想

### 2.1 Macro-to-Micro 范式

M2Flow 的本质是 **decouple program semantics from execution plans**。开发者用 imperative procedural programming 描述 logical flow：

```python
for step in range(N):
    rollout.generate(data, ch1)      # macro: 把结果送进 channel ch1
    inference.compute(ch1, ch2)      # 从 ch1 读，写到 ch2
    training.update(ch2)             # 从 ch2 读，更新参数
```

这是 macro level 的 control flow，但 execution 时 RLinf 把它 transform 成 micro flow：
- **Spatial dimension**：哪些 worker 在哪些 GPU 上
- **Temporal dimension**：什么时候 worker 拿到 device lock，什么时候 offload
- **Granularity dimension**：data channel 每次吐出多少 batch（elastic pipelining）

### 2.2 三种 Execution Mode（Figure 7）

paper §3.3 给出三种典型 scheduling：

**Pure Temporal**（左）：所有 worker 共享所有 GPU，sequential 执行 + context switch。适合"单组件必须用所有 GPU 才能跑起来"的场景，比如大模型 training。问题：long-tail 浪费。

**Pure Spatial**（中）：worker 分配到 disjoint GPU sets，pipelined execution。适合 rollout/inference/training 时间能 align 的情况。问题：rollout 慢时 training GPU 闲置。

**Hybrid**（右）：部分 worker pipelined，部分 worker 共享 GPU + context switch。这是 paper 的精髓——比如 ManiSkill 场景下，simulator + generation rollout 在一组 GPU 上 pipeline，然后 offload，training 接管这组 GPU。

## 3. 系统架构详解（Figure 4）

### 3.1 分层架构

RLinf 分三层：

**Programming Layer**：imperative Python API。Worker 类继承 base Worker，自带 `send/recv`、`onload/offload`、`device_lock`。Workflow runner 用 WorkerGroup abstraction 管理一组 SPMD worker processes，function call 异步返回 result handle，`.wait()` 提供 synchronization barrier。

**Scheduling Layer**：Controller + Scheduler + Profiler。Profiler 测每个 component 在不同 data parallel size 下的 execution time 和 memory。Scheduler 用 Algorithm 1 做 graph partition 找最优 placement。Controller 分配 worker 到 accelerator、管理 connection、dispatch function invocation。

**Communication Layer**：point-to-point primitives + data channel。Data channel 是关键创新，下面详述。

### 3.2 Worker Abstraction 关键点

每个 Worker 必须实现：
- `onload()`: 获取 GPU resource，load model weights/optimizer state 到 GPU memory
- `offload()`: 释放 GPU resource，swapping 到 CPU memory
- `device_lock`: distributed lock，atomic 全局状态，用于 sequential access 同一组 GPU

这个 abstraction 让 context switch 变成"acquire lock → onload → compute → release lock → offload"的自然语义，而不是手动 swap in/out。**这是 M2Flow temporal scheduling 的基石**。

## 4. Elastic Pipelining 与 Context Switching 深度解析

### 4.1 Elastic Pipelining 的"弹性"在哪

传统 pipelining 是 fixed batch size，pipeline stage 1 完整跑完 batch 才能进 stage 2。Elastic pipelining 的核心 insight 是：

> RL training 中大部分 worker 是 SPMD pattern，支持任意 batch size 执行。

所以 data channel 可以配置 "dequeue granularity"：
- 细粒度：rollout 每生成 1 个 response 就 enqueue，inference 立刻 dequeue → 极小 pipeline bubble
- 粗粒度：rollout 攒满 32 个 response 才 enqueue → 减少 communication overhead 但 pipeline bubble 变大

这个 granularity 是 **per-channel configurable** 的，scheduler 根据 profiling 结果选择最优 granularity。

但有一个 caveat：**training worker 有 internal semantics constraint**。Training 有 micro-batch（forward/backward unit）和 global-batch（model update unit）两个概念。Elastic pipelining 不能打破 global-batch 边界，否则 gradient 计算不一致。所以 training 的 upstream channel 必须等到攒满一个 global-batch 才能 dequeue。

### 4.2 Context Switching 实现

`device_lock` 不是普通 lock，它有 **data dependency-aware priority**：
1. 当 parent worker 还没 enqueue 数据时，child worker 不能 acquire lock（避免空转等待）
2. 当 parent release lock 后，依赖它的 child 才能竞争 lock
3. Controller 知道 worker placement，如果两个 worker 在不同 device，不需要 lock

这避免了 naive lock 实现中的 deadlock 和 contention。想象一下：rollout worker 持有 lock 在生成，training worker 等 lock。如果 rollout 卡住（long-tail），training 永远等不到。paper 的 design 让 data dependency 显式驱动 lock 释放，rollout 一旦 enqueue 完一个 batch 就可以 release，training 就能拿 lock 开始准备。

## 5. Scheduling Policy（Algorithm 1）详解

### 5.1 算法核心：s-t cut 递归

Algorithm 1 是个 dynamic programming on DAG：
- Input：workflow graph G、time estimation function E、device 总数 N
- 把 G 通过 s-t cut [Ford-Fulkerson, 1962] 分成 $G_s$ 和 $G_t$
- 对每个 cut，评估两种 cost：
  - **Temporal**：$T_s + T_t$（共享 GPU，sequential + offload overhead）
  - **Spatial**：pipelining cost（不同 GPU sets，pipeline overlap）
- 取最优，递归到 single node

关键公式：

$$T_{spatial} = T_{critical} + (M/m - 1) \times T_{bottleneck}$$

变量解释：
- $T_{critical}$：pipeline warm-up 和 cool-down 时间，等于所有 stage 的 startup time 之和
- $M$：total batch size（一个 iteration 总数据量）
- $m$：data processing granularity（一个 pipeline step 处理的 batch size）
- $T_{bottleneck}$：最慢 stage 的执行时间
- $M/m$：pipeline 总共的 step 数
- $M/m - 1$：steady state 的 step 数（去掉 warm-up 的第一个 step）

这个公式背后的 intuition 是经典的 pipeline parallelism 时间分解：warm-up 1 step + steady state $(M/m - 1)$ steps × bottleneck time。但 paper 这里把它扩展到 RL workflow 的 worker-to-worker pipeline，不只是 model layer pipeline。

### 5.2 Cycle Handling

§3.4 提到：embodied RL 和 Deep Research 有 cyclic data flow（rollout 接收 environment feedback 再生成下一个 action）。Algorithm 1 在 recursion 前用 `ConvertCircleToNode` 把 cycle collapse 成单 node，这个 node 内部均匀 partition 到所有 GPU。这是 trade-off：避免 exhaustive enumeration，可能 sub-optimal 但 practical。

### 5.3 Profiler 设计

Profiler 测两个东西：
1. **Execution time** vs **data parallel size**：对 model component，data parallel size = total GPU / model parallel size；对 simulator，data parallel size = instance 数量
2. **Memory usage** vs **data parallel size**

然后用 polynomial extrapolation 拟合，输出估计函数 E。这个 E 喂给 scheduler 做 cost estimation。

Figure 16a 显示 estimation accuracy：
- Temporal mode: < 2% error（因为简单加和）
- Spatial mode with pipelining: < 5% error（response length variability 导致 pipeline imbalance）

## 6. Adaptive Communication 深度

### 6.1 为什么 NCCL/Gloo/MPI 不够用

§3.5 列了三个 deficiency：
1. **Spatial dynamicity**：worker 在不同 device 间动态 collocate/distribute，NCCL 的 fixed rank 不友好
2. **Temporal dynamicity**：worker 任意时间 launch/terminate，传统 comm library 假设 static process group
3. **Data dynamicity**：payload 是复杂 Python 对象（tensor + metadata），不是 contiguous buffer

### 6.2 RLinf 的解决方案

**Connection Lifecycle**：global worker manager 注册 placement/IP/port。Connection lazy establish（第一次 send/recv 时才建连）。Worker terminate 时 connection manager 通知所有 connected worker teardown。

**Backend 自动选择**：
- GPU ↔ GPU（跨 node）：NCCL
- GPU ↔ GPU（同 node）：cudaIPC zero-copy
- CPU ↔ CPU：Gloo
- CPU ↔ GPU：自动桥接

**Structure-aware serialization**：从 Python 对象中 extract tensor buffer，直接 zero-copy 传输，metadata piggyback 在通信 header 里。这避免了 pickle 的 deep copy 开销。

### 6.3 Data Channel

Data channel 是个 FIFO queue，但有几个增强：

1. **CPU offloading**：GPU tensor 可以配置自动 offload 到 CPU，避免 GPU memory 撑爆
2. **Load balancing**：每个 item 有 weight，consumer 可以自定义 dequeue policy。这对 long-tail 很关键——快的 consumer 可以多 dequeue 几个 item，慢的少 dequeue
3. **Control/data flow decoupling**：producer enqueue 完就 return，不需要等 consumer；consumer dequeue 时如果 queue 空就 block。这是 elastic pipelining 的基础

## 7. 实验数据深度分析

### 7.1 Reasoning RL（Figure 8, 10, 12）

**Qwen2.5 GRPO**（Figure 8）：
- 1.5B on 64 GPU：RLinf temporal vs veRL，~1.10x
- 7B on 128 GPU：~1.58x
- 32B on 256 GPU：~1.40x

为什么 temporal mode 也能 beat veRL？paper 给两个原因：
1. KV-cache 更大（memory 管理更好）
2. Inference stage 的 sync overhead 减少

**Qwen2.5 PPO**（Figure 10）：PPO 有 4 个 model（actor、reference、reward、critic），场景更复杂：
- 1.5B on 16 GPU：RLinf spatial 比 veRL 快 69.6%
- 7B on 32 GPU：spatial 快 38.7-60.7%
- 14B：spatial 快 27.2-56.5%

PPO 偏向 spatial mode 因为 4 个 model 可以有效 pipeline overlap。GRPO 偏向 temporal 因为只有 1 个 model，spatial 反而浪费 GPU。

**Qwen3-30B-A3B MoE GRPO**（Figure 12）：对比 Slime（spatial without pipelining）和 Slime-Colocate。RLinf spatial (1:1 rollout:training) 比 Slime-Colocate 快 31.2%（32 GPU）和 7.2%（64 GPU）。MoE 场景下 rollout memory 大，spatial mode 让 rollout 能用更大 parallelism。

### 7.2 Embodied RL（Figure 14, 15）

**ManiSkill**（OpenVLA, 256 parallel envs）：
- RLinf-Hybrid 比 Temporal 快 52.2-69.1%，比 Spatial 快 60.7-87.2%
- 因为 simulator 是 memory-bound 但 compute-light，需要 dedicate GPU 跑多 env，但 rollout+training 要 share GPU 做 context switch

**LIBERO**（OpenVLA-OFT, 512 parallel envs）：
- RLinf-Temporal 比 SimpleVLA-RL 快 37.8-143.4%
- 为什么这里 Temporal 赢 Hybrid？因为 LIBERO 是 CPU-intensive，dedicate GPU 反而限制 CPU core 利用

这个对比揭示了 paper 的核心论点：**没有 universal optimal mode，必须 per-workload 调度**。

### 7.3 Model Performance（Table 1, 2, 3）

**Math reasoning**：
- RLinf 1.5B：AIME24 48.44（base 28.33，+20 点），AIME25 35.63，GPQA 38.46，avg 40.84（SOTA 在 1.5B 量级）
- RLinf 7B：avg 56.23，超过 Skywork/Polaris/AceMath，GPQA 48.18 是该 size 最强

**LIBERO**（Table 3）：
- OpenVLA-OFT one-traj baseline：avg 34.33%
- RLinf RL 后：avg 97.83%（Spatial 98.99%, Object 98.99%, Goal 98.99%, Long 94.35%）

这个 Long task 从 9.68% 跳到 94.35% 非常惊人，说明 RL training 在长 horizon task 上的效果，RLinf 的稳定训练让 long-horizon RL 成为可能。

## 8. 与 Related Work 的定位

paper §6 列了三类 related work：

**RL Frameworks**：
- Collocated：DeepSpeed-Chat、veRL
- Disaggregated：NeMo-Aligner、Open-RLHF、AReaL（异步 update）

RLinf 的位置是 "flexible placement between collocated and disaggregated"。

**Distributed Training**：Megatron-LM、DeepSpeed ZeRO、FSDP。这些是 model training 的 infra，RLinf 用它们做底层 training engine，不直接竞争。

**Dataflow Systems**：MapReduce、Dryad、Naiad、Spark。传统 dataflow 是 static graph + central scheduling，RL workflow 是 dynamic graph + async component，需要 Ray 这种 actor-based decentralized 控制。

paper 没有深对比的是 **AReaL**（[11]）。AReaL 在 disaggregated 上做 async update，而 RLinf 在 hybrid placement 上做 elastic pipelining。两者正交，理论上可以结合。

## 9. 我的 Intuition 和 Critique

### 9.1 M2Flow 的本质

M2Flow 让我想到 **compiler 的 IR lowering**：high-level Python code (macro) → optimized LLVM IR (micro)。RLinf 把 imperative workflow 编译成 GPU placement + pipelining schedule。这个类比下，Profiler 像 cost model，Scheduler 像 instruction scheduler，data channel 像 register allocation 中的 virtual register。

更深一层，M2Flow 接近 **MLSys 中的 "separation of concerns"** 范式：
- Alpa（OSDI'22）分离 inter-op 和 intra-op parallelism
- RLinf 分离 logical flow 和 physical execution

这种分离让 search space 可被系统自动探索，而不是用户手动 tune。

### 9.2 局限性

1. **Cycle handling 简化**：§3.4 把 cycle collapse 成单 node，对 embodied RL 的 long horizon 可能 sub-optimal。如果 rollout 有 80 步 environment interaction，每步都有 model inference，这个 cycle 的内部 scheduling 没被 explore。

2. **Profiler 一次性**：profiled 在 training 前做一次，但 RL 训练中 response length distribution 会漂移（curriculum learning、model 能力提升），E 函数会失效。paper 没讨论 online re-profiling。

3. **Search 复杂度**：§5.2 说 search time 在 8-1024 GPU 下 < 5s，但 Figure 16b 显示 exponential growth。10000+ GPU 规模会怎样？paper 没测。

4. **Single-controller bottleneck**：Controller 是单点，dispatch 所有 function invocation。大规模下可能成为 scheduling bottleneck。Ray 的 actor model 本身是 decentralized 的，但 RLinf 的 Controller 是 central，可能限制扩展性。

5. **MoE 优化有限**：Qwen3-30B-A3B 实验中，128 GPU 时 RLinf-Spatial 反而比 Temporal 慢，因为 rollout 和 training 没有良好 overlap。MoE 的 expert routing dynamic 让 pipeline balancing 更难，paper 没专门解决。

### 9.3 关联联想

- **veRL/HybridFlow** [Sheng et al., EuroSys'25]：RLinf 的主要 baseline，HybridFlow 也是 hybridflow 思路，但 veRL 的 hybrid 是"在两个 fixed mode 间选择"，RLinf 是"在 continuous scheduling space 中搜索"。参考：https://github.com/volcengine/verl

- **OpenRLHF** [Hu et al., 2024]：disaggregated 路线代表，论文 https://arxiv.org/abs/2405.11143

- **AReaL** [Fu et al., 2025]：异步 RL，论文 https://arxiv.org/abs/2505.24298

- **SGLang** [Zheng et al., 2024]：RLinf 的 rollout engine，https://arxiv.org/abs/2401.05553

- **Ray** [Moritz et al., OSDI'18]：RLinf 的 cluster management 基础，https://www.usenix.org/conference/osdi18/presentation/moritz

- **DeepSeekMath GRPO** [Shao et al., 2024]：核心算法，https://arxiv.org/abs/2402.03300

- **ManiSkill3** [Tao et al., RSS'25]：embodied benchmark，https://github.com/haosulab/ManiSkill

- **LIBERO** [Liu et al., 2023]：lifelong robot learning benchmark，https://arxiv.org/abs/2306.03310

- **OpenVLA** [Kim et al., 2024]：VLA model，https://arxiv.org/abs/2406.09246

- **SimpleVLA-RL** [Li et al., 2025]：RL for VLA training，RLinf 的 embodied baseline，https://github.com/VoltronBot/SimpleVLA-RL

- **Megatron-LM** [Shoeybi et al., 2019]：RLinf 的 training backend，https://arxiv.org/abs/1909.08053

- **Ford-Fulkerson max-flow**：Algorithm 1 用的 s-t cut 算法，经典图论，https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm

- **DAPO** [Yu et al., 2025]：RLinf 支持的另一个算法，https://arxiv.org/abs/2503.14476

- **REINFORCE++** [Hu, 2025]：简化版 RLHF，https://arxiv.org/abs/2501.03262

- **Deep Research RL** [Zheng et al., 2025]：https://arxiv.org/abs/2504.03160

### 9.4 未来方向推测

paper §7 说 "RLinf marks an early step toward the operating system for AI workloads"。这个 vision 我认同。接下来可能的发展：

1. **Online re-profiling**：用 RL 学习调度策略本身，比如 Thompson sampling 调度 mode
2. **Multi-tenant RL**：多个 RL job 共享 cluster，类似 Kubernetes scheduler
3. **Hardware heterogeneity**：CPU/GPU/NPU/robot arm 混合调度，paper 已经提到 robot arm 作为 schedulable device
4. **Speculative execution**：rollout 时同时跑 inference 预测 logprob，cancel 如果 response 被截断
5. **Hierarchical M2Flow**：cluster level M2Flow + node level M2Flow + device level M2Flow，类似 NUMA hierarchy

## 10. 总结

RLinf 的核心贡献是把 RL training system 从 "either collocated or disaggregated" 推进到 "any point in the scheduling continuum"。M2Flow 范式通过 worker abstraction + elastic pipelining + context switching + profiling-guided scheduler 实现这个 continuum。1.07-2.43x speedup 主要来自两点：
1. 不被 long-tail 卡死
2. heterogeneous component 各得其所

代码开源在 https://github.com/RLinf/RLinf，20K 行 Python，5K 核心 + 2K common workers + 13K algorithm support。典型 workflow runner < 100 行，这个 programmability 是 paper 强调的优势。

Andrej，如果你之前关注过 RLHF 系统的进展，这篇 paper 应该是你需要的下一代 RL infra reference。它不会改变 RL algorithm，但会让 RL training 变得 elastic，让 long-horizon agentic RL（你关心的 Deep Research、code agent 等）变得 tractable。

主要参考链接：
- Paper GitHub: https://github.com/RLinf/RLinf
- veRL baseline: https://github.com/volcengine/verl
- Slime baseline: https://github.com/THUDM/slime
- SGLang (rollout engine): https://github.com/sgl-project/sglang
- Ray: https://github.com/ray-project/ray
- ManiSkill: https://github.com/haosulab/ManiSkill
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- OpenVLA: https://github.com/openvla/openvla
- SimpleVLA-RL: https://github.com/VoltronBot/SimpleVLA-RL
