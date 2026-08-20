---
source_pdf: EVENT TENSOR A UNIFIED ABSTRACTION FOR COMPILING DYNAMIC.pdf
paper_sha256: 7c9226d04e34d770d763f592cd21b083cddb8756f62fd99f7267168bd98a93cd
processed_at: '2026-08-04T05:36:03-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---
decode阶段，batch size = 1那种实时聊天的场景——GPU上发生了什么？

一整层transformer要跑：RoPE、Norm、KV-Cache append、Attention、 projection、MLP/MoE……拆开来看可能有**几百个小kernel**。每个kernel从CPU那边launch过去，光过driver、过command buffer、到GPU front-end dispatch，就要**5到10微秒**。但这个kernel本身在GPU上算完可能只要**2微秒**。

> 启动开销比干活还长。这相当于你每次炒一道菜都要重新点火、热锅、拿铲子，结果菜本身30秒就炒完了，但你准备工作花了两分钟。荒谬。

更糟的是：kernel A跑完 → kernel B才能开始。哪怕B只用到A输出的一小块，也必须等A整个跑完。这叫"kernel boundary = implicit barrier"。

**CUDA Graph**帮了一半忙：它把一串kernel launch的命令预录下来，replay的时候省掉CPU→GPU的round-trip。但kernel之间的boundary还在，barrier还在。而且最要命的是——**batch size一变，graph就得重新capture**。生产环境里vLLM要capture 67个graph，SGLang要capture 51个，warmup time分别123秒和583秒。用户等这么久模型才起来？不能接受。

---

## 2. "Megakernel"是什么思路

最近一两年的新想法：**干脆把所有小kernel融合成一个大kernel，只launch一次，然后这个大kernel常驻在GPU上，不停地从queue里取任务执行**。

这个常驻大kernel叫**persistent kernel**或者**megakernel**。好处显而易见：
- 只launch一次，5-10μs的开销摊到整个推理过程中，基本没了
- kernel内部的细粒度任务之间可以overlap，比如Q的Norm+RoPE可以和K的Norm+RoPE并行（传统kernel boundary下做不到）

Stanford的Bobby Spector那篇["Look Ma, No Bubbles"](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)就是这么干的，手工写了LLaMA-1B的decode megakernel，single-batch。CMU的[Mirage Persistent Kernel](https://arxiv.org/abs/2512.22219)走compiler路线，也只支持single-batch dense model。

**问题来了**：现实世界的LLM serving有两个硬骨头：

1. **Dynamic shape**：continuous batching下batch size一直在变。你不可能为每个shape都compile一个megakernel。
2. **Data-dependent control flow**：MoE层里token被routing到哪个expert是runtime算出来的，编译期你根本不知道task graph长什么样。

Stanford和CMU这两个既有megakernel，**都没法处理这两个情况**。

---

## 3. Event Tensor的核心idea

### 3.1 用类比说清楚

先忘掉所有术语，想一个朴素的事情：

在megakernel里，你把一个operator（比如GEMM）拆成很多小tile，每个tile跑在一个SM上。一个tile干完了，要通知下游tile"我这边好了，你可以开始了"。这个"通知"就是一个**event**。

一个GEMM如果有 `(M/128, N/128)` 个tile，那就是成千上万个event。你怎么管理？

传统task-based runtime（Legion、Realm、Cilk）的做法：把每个event当成一个独立的node，task graph整个materialize到内存里，runtime去traverse。

**Event Tensor的insight**：event天然有结构。GEMM的tile grid是2D的 `(i, j)`，那对应的event也自然是2D的。你把它当成一个**tensor**来看，立刻就能复用编译器对tensor的所有基础设施——symbolic shape、sharding、indexing、tiling。

> 用厨房类比：传统做法是给每道菜单独记一张"何时开始"的小纸条，几百张纸条堆在桌上。Event Tensor的做法是把所有菜的进度做成一个Excel表格，行是菜的种类，列是制作阶段，单元格里写"还差几步"。编译器一看就知道怎么调度。

### 3.2 三个语言construct

paper定义了三个东西，其实就是三层抽象：

**Device Function**：一个tile-level的kernel。每个task由一个multi-dimensional coordinate标识，跑在一个SM上。写起来像Triton。

**Event Tensor**：一个多维数组，每个元素是个event。每个event有：
- `wait_count`：一个counter，初始化为"依赖多少个上游task"
- `E[i].notify()`：atomic decrement counter
- `E[i].wait()`：spin-wait直到counter到0

**Graph Function**：把device function串起来，用 `in_edges` / `out_edges` 注解task之间的依赖。

### 3.3 用paper的例子讲透

Figure 3的例子：对矩阵 `A: shape (n*32, 128)` 做行求和，分两阶段。

$$B[i, j] = \sum_{k \in [j \cdot 32, \, j \cdot 32 + 32)} A[i, k], \quad C[i] = \sum_{k \in [0, 4)} B[i, k]$$

变量含义：
- $i$：行index，范围 $[0, n \cdot 32)$
- $j$：partial sum的分块index，范围 $[0, 4)$，因为 $128/32 = 4$
- $k$：内层reduce维度

如果不分tile，stage2必须等stage1整个kernel跑完。但显然 `C[i]` 只依赖 `B[i, :]`，**第i行的C不需要等第j行（$j \neq i$）的B算完**。

Event Tensor的做法：
- 把stage1分成tile $\hat{B}_{i,j}$，每个算 `B[i*32:i*32+32, j]`
- 对每一行i，定义一个event $E_i = E[i]$，`wait_count = 4`（4个j）
- stage2的tile $\hat{C}_i$ 算 `C[i*32:i*32+32]`，开头调 `E[i].wait()`

依赖链：$\hat{B}_{i,j} \xrightarrow{\text{notify}} E_i \xrightarrow{\text{wait}} \hat{C}_i$

每个 $\hat{B}_{i,j}$ 结束调 `E[i].notify()`，counter从4减到3、2、1、0。当counter到0，$\hat{C}_i$ 的wait解除，开始跑。

**结果**：所有行完全并行，没有global barrier。

### 3.4 Symbolic shape为什么天然支持

因为Event Tensor的维度可以是symbolic variable。比如 `E: shape (B, N//BLK_N)`，编译期 $B$ 是symbolic，runtime才赋值。这跟[TVM Relax](https://arxiv.org/abs/2401.16792)的symbolic shape机制同源。

> 一个template编一次，runtime instantiate无穷次。CUDA Graph的"每个shape capture一次"在Event Tensor这里根本不需要。

### 3.5 Data-dependent怎么办（MoE场景）

MoE的难点：runtime算出 `topk` routing结果后，才知道每个token去哪个expert，每个expert要处理多少token。编译期完全不知道。

Event Tensor用两招：

**Data-dependent event update**：每个expert有一个event，`wait_count` 在runtime由routing结果动态设置（被路由到这个expert的token数）。grouping tile（每个处理一个token）在runtime决定notify哪个expert的event。

**Data-dependent task triggering**：`exp_indptr` 是一个prefix sum数组，`exp_indptr[i]` 到 `exp_indptr[i+1]` 之间是expert $i$ 要trigger的GroupGEMM tile范围。event counter到0后，scheduler把这段tile push进ready queue。

注意整个依赖链**还是feed-forward**的：`Attention → TopK → Grouping → GroupGEMM`。TopK之前是static dependency，TopK之后才用data-dependent机制。**没有循环依赖**，编译器处理起来可控。

---

## 4. 编译器怎么用这个抽象

### 4.1 Static scheduling pass

**核心想法**：编译期就把每个SM的任务queue算好，runtime零调度开销。

具体步骤（对应Algorithm 1）：
1. Host端用round-robin构建per-SM task queue
2. 生成persistent main loop：每个SM不停从自己的queue取task执行
3. Lower Event Tensor依赖为 `notify()` / `wait()` 调用

以GEMM + Reduce-Scatter为例（Figure 6），融合前后：

**融合前**：两个独立device function，分别launch。
**融合后**：一个persistent function，循环取task。GEMM tile结尾调 `E[i,j].notify()`，RS tile开头调 `E[i + offset // BLK_M, j].wait()`。

Figure 7的时序图讲得很清楚：
- $T_1$：SM0的MM0完成，notify后counter从2减到1，RS task spin-wait
- $T_1$ 到 $T_2$：SM1继续跑MM0，GPU不空转
- $T_2$：SM1的MM0完成，counter到0，RS task被释放

**关键细节**：`wait_count = WORLD_SIZE = 2`，因为RS tile依赖2个MM tile（跨device的）。Event Tensor可以shard：`shard="S[0]"`，每个device只持有自己local的那部分event。

Static scheduling处理dynamic shape用"sample representative shapes"，runtime用next larger sample。处理data-dependent用worst-case展开——所以MoE场景下static scheduler表现不好，这就是为什么MoE实验用了dynamic scheduler。

### 4.2 Dynamic scheduling pass

**核心想法**：on-GPU task scheduler，runtime决定执行顺序。

机制（Figure 9）：
- Event被trigger（counter到0）后，consumer task被atomic push进global memory的ready queue
- 空闲SM atomic pop一个task执行
- 整个dependency tracking和dispatch在GPU上完成，host不参与

Appendix E提到的**early push优化**：producer task被dispatch到SM后（注意是dispatch，不是完成），consumer就立刻被push进ready queue。consumer自己wait保证依赖正确。push操作和producer执行overlap，不在critical path上。

**Trade-off**：
- Static：调度开销最小，适合可预测的workload
- Dynamic：灵活，适合data-dependent或unpredictable，但有push/pop overhead

paper Table 2和Table 3的对比很直观：
- MoE（irregular, data-dependent）：dynamic胜出4-8%
- Dense TP=4（regular, latency-critical）：static胜出6-8%

### 4.3 Lowering到minimal runtime

这是最干净的部分：
- Event Tensor → integer tensor（counter数组）
- `notify()` → `atom.global.add.release` (atomic decrement)
- `wait()` → spin-loop + `ld.global.acquire`

**runtime state就是几个integer tensor + scheduler queue**。没有传统task-graph runtime那种"整个graph materialize到memory + executor traverse"的开销。

---

## 5. 实验数字里最值得看的

### 5.1 Warmup time（Table 1，最重要）

| Method | Warmup Time | # JIT Graph Capture |
|--------|-------------|---------------------|
| SGLang (JIT) | 583s | 51 |
| vLLM (JIT) | 123s | 67 |
| ETC (AOT) | 35s | 0 |

**3.5x faster than vLLM, 16.6x faster than SGLang**。生产环境里每次模型reload或autoscale都要等warmup，省几百秒就是省钱。

### 5.2 GEMM + Reduce-Scatter / All-Gather + GEMM（Figure 11/12）

8 B200上tensor-parallel size = 8，8192 tokens。最高**1.40x speedup over cuBLAS+NCCL baseline**。

baselines里：
- [TP-Async (TorchTitan)](https://openreview.net/forum?id=SFN6Wm7YBI)：手动async，coarse-grained split容易顾此失彼
- [Triton-Distributed](https://arxiv.org/abs/2504.19442)：compiler baseline，但B200上Triton GEMM还没调好
- [cuBLASMp](https://docs.nvidia.com/cuda/cublasmp/index.html)：NVIDIA的multi-process fused library

ETC的优势来自fine-grained dependency让SM和network都持续busy。

### 5.3 MoE layer（Figure 13）

Qwen3-30B-A3B（128 experts, top-8）。对比[Triton 3.4.0](https://dl.acm.org/doi/10.1145/3311508.3311517)和[FlashInfer 0.2.14](https://arxiv.org/abs/2501.01005)。

**最高1.23x speedup at 1024 tokens**。原因：
- Data-dependent Event Tensor打破两阶段GroupGEMM之间的global barrier
- 减少wave quantization
- On-chip dynamic scheduler给irregular token routing提供load balancing

### 5.4 End-to-end LLM serving（Figure 14）

Qwen3-30B-A3B（MoE）：
- batch=1: ETC **1.48x over vLLM, 1.20x over SGLang**

Qwen3-32B（dense）：
- batch=1: 1.15x over vLLM, 1.09x over SGLang at batch=64

TP=4 Qwen3-32B：
- 0.99x–1.06x over vLLM（持平）
- SGLang在这里反而更快，因为SGLang的CPU scheduler在分布式场景下runtime overhead更低

paper诚实承认了这个gap来自compiler-generated GEMM tile没cuBLAS精调 + serving engine的CPU-side overhead——是engineering问题，不是抽象的根本限制。

---

## 6. 我觉得真正有意思的地方

### 6.1 "Event做成tensor"这个抽象的深层意义

把event从runtime object提升到compiler IR的first-class object，意味着编译器对tensor的所有优化（symbolic shape、sharding、tiling、indexing）都自动复用到dependency管理上。

这跟[Graphene (Hagedorn et al., 2023)](https://doi.org/10.1145/3582016.3582018)把thread做成tensor的思路一脉相承，但Graphene只做single-kernel优化，Event Tensor做multi-operator fusion + dynamic shape + data-dependent。

[Cypress (Yadav et al., PLDI 2025)](https://doi.org/10.1145/3729262)也做task-based GPU computation，但更像"在GPU上跑task graph runtime"，没有compiler IR层面的symbolic shape抽象。

### 6.2 Megakernel时代GPU的编程模型在变

以前GPU编程模型是"launch grid of CTAs, 算完return"。Megakernel时代GPU像一个**多核CPU**：148个SM像core，没有cache coherence，只有shared memory per-SM和global memory。

在这种model下，**atomic + spin-wait是最自然的同步原语**。Event Tensor把它们抽象到compiler IR层，让编译器做调度而不是runtime task executor。

这跟[Legion (Bauer et al., 2012)](https://ieeexplore.ieee.org/document/6480535)和[Realm (Treichler et al., 2014)](https://dl.acm.org/doi/10.1145/2628071)的思想一脉相承，但落地到了compiler层。

### 6.3 跟Programmatic Dependent Launch的关系

NVIDIA Hopper引入[PDL](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch)：一个kernel可以提前launch下一个kernel，partial overlap。PDL解决了kernel boundary barrier的一部分，但仍然是**coarse-grained**（kernel-to-kernel），不是tile-to-tile。

Event Tensor是tile-to-tile的fine-grained dependency，比PDL更细。理论上两者可以互补：跨megakernel用PDL，megakernel内部用Event Tensor。paper没探索这个组合，是个明显的后续方向。

### 6.4 为什么CuSync不够

[CuSync (Jangda et al., 2024)](https://arxiv.org/abs/2305.13450)做separate kernels on distinct CUDA streams的co-scheduling，比Event Tensor粗粒度。Event Tensor把整个subgraph fuse进一个persistent megakernel，靠compiler pass系统化生成，超越了任何single operator pattern的手工优化。

---

## 7. 我看到的局限

### 7.1 Centralized global queue的contention

Appendix E说用early push hide scheduling overhead，但B200有148个SM，高频访问global queue的atomic op会有限制。Per-SM queue + work stealing可能更好，但工程复杂度上升。paper没量化这个contention。

### 7.2 Static scheduling处理dynamic shape的"sample representative shapes"有点hacky

如果shape分布很wide，cache miss率会高。[Relax](https://arxiv.org/abs/2401.16792)的symbolic shape机制其实更系统化，可以借鉴。

### 7.3 Multi-GPU的Event Tensor没展开

Figure 6里Event Tensor有 `shard="S[0]"`，但cross-device dependency在NVLink上怎么高效实现、跨node的IB/RDMA场景怎么扩展，paper没细说。

### 7.4 Prefetch pass依赖user annotation

理想情况下应该compiler自动推断prefetch机会，但这需要更复杂的dependency analysis。

### 7.5 Code-generated GEMM追赶cuBLAS

所有ML compiler的通病。B200 Blackwell的新tensor core指令（`wgmma.async`、`tcgen05`）需要更精细的instruction scheduling。paper承认TP=4时ETC略输SGLang，部分原因就在这。

---

## 8. 一句话总结

**Event Tensor把"task graph + dependency"这个runtime-level的东西，最小化成"integer tensor + atomic + spin-wait"这个compiler IR-level的first-class object，让编译器做调度而不是runtime task executor。加上symbolic shape的first-class support，AOT编译的dynamic shape megakernel终于成为现实。**

你以前在[micrograd](https://github.com/karpathy/micrograd)里强调"理解反向传播和计算图的minimum viable implementation"。Event Tensor在精神上类似——用最少最干净的抽象抓住megakernel时代GPU编程的本质。

未来值得关注的open direction：
1. 自动从普通computation graph生成Event Tensor annotation（需要alias analysis + dataflow analysis）
2. Speculative decoding的draft/verify overlap用Event Tensor表达
3. Multi-node扩展（RDMA write做cross-node atomic）
4. 跟PDL的组合
5. Profile-guided static schedule optimization

如果你对哪一节想深挖——比如static scheduling pass的round-robin策略怎么改造成profile-guided、dynamic scheduler的lock-free queue具体实现、或者MoE的event update到底怎么mapping到PTX——我都可以再展开。我个人觉得最值得深挖的是§3.3的lowering细节和Appendix E的early push策略对critical path的影响，这两个engineering细节决定了megakernel性能上限。

---

# Event Tensor: 给Megakernel时代的"动态图"编译做一次彻底抽象

Andrej，这篇paper的作者阵容基本是CMU Tianqi Chen团队 + DeepSeek/Modelscope/SGLang生态的核心人物（Lijie Yang, Zihao Ye, Vinod Grover等都是MegaScale/SGLang的contributor），还有Stanford Hazy Research的Bobby Spector背景的人。读完最直接的感受是：**它解决了一个被CUDA Graph时代遗留下来的尴尬问题——dynamic shape和data-dependent control flow如何被fused megakernel处理，而不用重编译/重capture**。

下面我从intuition → 抽象设计 → 编译pass → 实验 → 局限性，逐层拆解，尽量把paper里没展开的engineering细节补出来。

---

## 1. 动机：为什么megakernel时代需要新的抽象

### 1.1 传统调度模型的瓶颈

PyTorch的kernel-by-kernel模型：每次kernel launch 5–10μs，而kernel本身可能只要2μs（LLM decoding阶段典型情况，batch=1时尤其惨）。问题在于：
- **launch overhead dominates**：CPU→GPU的command submission要打通driver → kernel driver context switch → GPU front-end dispatch，这个链路即使做了CUDA Graph capture也无法消除host-side dispatch cost
- **kernel boundary = implicit barrier**：即使后一个kernel只依赖前一个kernel的部分输出（比如attention的Q和K的norm/RoPE独立），也必须等前一个kernel整体跑完才能launch下一个，**inter-kernel parallelism被白白浪费**

CUDA Graph（[NVIDIA CUDA Graphs blog, 2019](https://developer.nvidia.com/blog/cuda-graphs/)）解决了launch overhead，但保留了kernel boundary。SGLang/vLLM在生产环境里普遍面临一个问题：**continuous batching下batch size每变一次，就要重新capture一个graph**，warmup动辄几十秒到几分钟，见paper Table 1：SGLang需要583s warmup + 51个JIT graph capture。

### 1.2 Megakernel的两种既有路径

最近的megakernel工作主要有两条线：
- **Stanford的"Look Ma, No Bubbles"**（[Spector et al., 2025](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)）：手工写一个persistent kernel来跑LLaMA-1B的decode，single-batch only
- **Mirage Persistent Kernel**（[Cheng et al., 2025](https://arxiv.org/abs/2512.22219)）：CMU的compiler + runtime路径，同样局限single-batch dense model

这两个都**没有first-class的data-dependent control flow抽象**。比如MoE里topk routing决定了token去哪个expert，这个routing结果是runtime算出来的，编译期你根本不知道task graph长什么样。

Event Tensor就是来填这个坑的。

---

## 2. Event Tensor抽象：核心设计哲学

### 2.1 "把event做成tensor"为什么重要

传统task-based runtime（Cilk [Blumofe et al., 1995]、Legion [Bauer et al., 2012]、Realm [Treichler 2014]）都把event当成独立的node，task graph materialized到memory里，runtime去traverse。问题：
- LLM serving里一个megakernel内部可能有**百万级fine-grained events**（每条task tile对应一个event），独立node的materialization overhead巨大
- 无法做symbolic shape：每个event的index必须concrete

Event Tensor的关键insight是：**event天然形成多维结构**，就像tensor一样。比如GEMM + Reduce-Scatter里，GEMM的tile grid是 `(M//BLK_M, N//BLK_N)`，每个tile完成时notify一个event，所有这些event就构成一个 `(M//BLK_M, N//BLK_N)` 的2D event tensor。这样：
- 维度直接复用symbolic shape的机制（AOT compile时是 `(B, N//BLK_N)`，runtime才materialize）
- compiler对tensor的indexing/tiling/scheduling优化全部复用
- 用一个integer tensor就能lower到底层（counter数组）

### 2.2 三种语言construct

paper的IR layer提供了三种construct，很关键的primitives：

**Device Function**：一个grid of tasks，每个task由multidimensional coordinate标识，跑在一个SM上。本质上是Triton/TileLang风格的tile-level kernel。

**Event Tensor**：多维event数组，每个元素有：
- `wait_count`：初始化为依赖的task数（类似counting semaphore）
- `E[i].notify()`：atomic decrement wait_count
- `E[i].wait()`：spin-wait直到wait_count == 0
- dynamic scheduling下还可以trigger consumer tasks

**Graph Function**：包含call_device调用，可以annotate `in_edges`和`out_edges`做dependency mapping。

Figure 3的例子很重要，我用更详细的symbol来说明：

```
A: shape (n*32, 128)  # n是symbolic
目标：C[i] = sum_{k in [0, 128)} A[i, k]

# Stage 1: partial sum
B[i, j] = sum_{k in [j*32, j*32+32)} A[i, k]    # B shape (n*32, 4)

# Stage 2: aggregate  
C[i] = sum_{k in [0, 4)} B[i, k]                # C shape (n*32,)
```

这里 $i$ 是row index, $j \in [0, 4)$ 是partial sum的分块索引, $k$ 是内层reduce维度。

如果不分task，stage2必须等stage1整个kernel跑完。但显然 `C[i]` 只依赖 `B[i, :]`，所以可以分tile后用fine-grained dependency。Event Tensor引入：

- Task $\hat{B}_{i,j}$：计算 `B[i*32:i*32+32, j]`，是stage1的一个tile
- Event $E_i = E[i]$：初始 `wait_count = 4`（因为有4个j需要先完成）
- Task $\hat{C}_i$：计算 `C[i*32:i*32+32]`，消费 $E_i$

依赖链：$\hat{B}_{i,j} \to E_i \to \hat{C}_i$。每个 $\hat{B}_{i,j}$ 完成时调 `E[i].notify()`，当4个j都notify后，$\hat{C}_i$ 的 `E[i].wait()` 才返回。

这样所有行可以**完全并行**，而不需要等global sync。

### 2.3 为什么symbolic shape天然支持

关键在于Event Tensor的维度可以是symbolic variable。比如batch size $B$ 是symbolic，那么Event Tensor `E: shape (B, N//BLK_N)`，编译期不需要知道 $B$ 的具体值，runtime才把symbolic variable instantiate成concrete value。

这和TVM Relax [Lai et al., 2025](https://arxiv.org/abs/2401.16792) 的symbolic shape机制同源（paper里作者很多overlap）。**AOT编译一个symbolic shape template，runtime用零编译开销instantiate**。

对比CUDA Graph：必须为每个shape capture一次graph，67次capture = 67秒的warmup（vLLM的数据）。ETC只compile一次（107s AOT），但runtime warmup只要35s，对应Table 1。

### 2.4 Data-dependent dynamism的两板斧

MoE的难点：编译期你不知道routing结果，所以你不知道grouping tile该notify哪些expert event，也不知道每个expert要trigger多少个GroupGEMM tile。

Event Tensor的两个mechanism：

**Data-Dependent Event Update**：runtime的 `topk` tensor决定grouping tile（一个tile处理一个token）notify哪个expert的event。每个expert event的 `wait_count` 在runtime由routing结果动态设置（初始化为"被路由到这个expert的token数"）。

**Data-Dependent Task Triggering**：`exp_indptr` 存储每个expert要trigger的GroupGEMM tile数量的prefix sum。expert $i$ 激活tile范围 `(exp_indptr[i], exp_indptr[i+1])`。event counter到0后，scheduler把这段范围的tile push到ready queue。

注意整个依赖链**仍然是feed-forward的**：`Attention Output → TopK → Token Grouping → GroupGEMM`。TopK的计算是static dependency（依赖attention输出），只是后面阶段的依赖关系是runtime数据决定的。这个澄清很重要，因为它意味着编译器不需要处理任意的循环依赖，只需要处理"runtime才知道具体shape的feed-forward graph"。

---

## 3. 编译pass：static与dynamic两条路

### 3.1 Static Scheduling Pass（Algorithm 1）

目标：编译期构建per-SM task queue，runtime零调度开销。

步骤：
1. Host端构建每个SM的execution queue（paper用round-robin，简单粗暴）
2. 生成persistent main loop：每个SM从自己的queue里不停取task执行
3. Lower Event Tensor dependency为 `notify()` / `wait()` 调用

Figure 6的GEMM + Reduce-Scatter例子前后对比很清楚。变换前是两个独立device function，变换后fused成一个persistent function：

```python
@device_func
def fused_matmul_rs(sm_id, A, B, C, E, D):
    with cta():
        tile_scheduler = init_tile_scheduler(sm_id)
        while tile_scheduler.valid():
            task_idx, task_type = tile_scheduler.get_task()
            if task_type == 0:  # GEMM
                i, j = task_idx
                matmul_cta(C[...], A[...], B[...])
                E[i, j].notify()   # notify Event Tensor
            else:  # Reduce-Scatter
                i, j = task_idx
                offset = get_rank() * LOCAL_M
                E[i + offset // BLK_M, j].wait()  # wait Event Tensor
                multimem_ld_reduce_cta(D[...], C[...])
            tile_scheduler.next_tile()
```

注意 `E` 是distributed Event Tensor，shard在维度0（`shard="S[0]"`）。每个device只持有一个shard `(LOCAL_M // BLK_M, N // BLK_N)`，对应自己的local output部分。这个sharding机制和DeepSeek的DualLab/dual-stream overlap思想一致。

Figure 7的时序图解释了wait/notify机制：
- 初始每个event的 `wait_count = WORLD_SIZE = 2`（因为RS task依赖2个MM task）
- $T_1$：SM0上MM0完成 → `E.notify()` → counter减到1 → SM0的RS task进入spin-wait
- $T_1$–$T_2$：SM1继续跑MM0，GPU不空转
- $T_2$：SM1上MM0完成 → counter减到0 → RS task被释放

这里的spin-wait实现细节paper没细说，但工程上一般用 `ld.global.acquire` + `red.global.add` 或者PTX的 `atom.global.add`（[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)）。在Blackwell上有 `multimem` PTX可以更高效。

**Static scheduling如何处理dynamic shape**：paper的策略是"sample一组representative shape，runtime用next larger sampled value"。这个其实有点hacky，对于shape变化剧烈的工作负载会有sm浪费。

**Static scheduling如何处理data-dependent dynamism**：conservatively假设worst case，把所有可能notify/wait都展开。这显然在MoE这种高不规则的workload上很糟糕——这就是为什么MoE实验用了dynamic scheduler。

### 3.2 Dynamic Scheduling Pass（Algorithm 2）

关键：on-GPU task scheduler。event被trigger后（counter到0），atomic地把consumer tasks push到scheduler的ready queue；空闲SM atomic pop一个task执行。

Figure 8/9的push-pop机制：
- $T_1$：SM0的MM0完成，counter减到1；SM0空闲，立刻pop一个ready task（MM1）开始执行
- $T_2$：SM1的MM0完成，counter减到0，trigger RS task push进queue；SM1 pop RS task执行

整个dependency tracking和dispatch在GPU上完成，host完全不参与。

**实现细节**：paper说用centralized queue in global memory（一个GPU全局的lock-free queue）。Appendix E提到"early push"策略：producer task被dispatch到SM后，consumer就立即被push进ready queue（不等producer执行完），consumer自己wait保证依赖正确。这隐藏了push操作的开销，让它不落在critical path上。

Centralized queue的好处是简单，坏处是大规模时contention。Spector他们的Stanford megakernel用的是per-SM distributed queue。Cypress [Yadav et al., PLDI 2025](https://doi.org/10.1145/3729262) 做的是更细粒度的work stealing。这是一个明显的后续工作方向。

### 3.3 Lowering to minimal runtime

这部分是compiler-only的漂亮工程：
- Event Tensor → integer tensor（counter数组）
- `notify()` → `atom.global.add.release` (atomic decrement)
- `wait()` → spin-loop + `ld.global.acquire`
- 不需要任何runtime task graph data structure

对比传统task-graph runtime：整个graph要materialize在memory里，executor traverse graph来launch device function。ETC的做法是"compile scheduling logic into megakernel"，runtime state只是几个integer tensor + scheduler queue。

### 3.4 端到端编译flow

Figure 15的pipeline：
1. Input: tile-level dataflow graph（用户用TVM-based DSL [Hou et al., Axe 2026](https://arxiv.org/abs/2601.19092) 写，paper提到也DSL-agnostic，可以接Triton/CuteDSL）
2. Graph-level优化：memory planning等
3. Tile-level优化：硬件指令mapping、pipelining
4. **Static或Dynamic scheduling pass**（§3.1/3.2）
5. Emit成GPU persistent kernel code
6. Prefetch pass：根据annotation生成weight prefetch，在activation到达前prefetch weight
7. Static scheduling时计算per-SM task order，materialize成execution queue

第6步weight prefetch很重要。LLM decoding时weight是静态的，activation是动态到达的，prefetch weight可以hide memory latency。这是SGLang/vLLM手工做的优化，ETC把它变成了compiler pass。

---

## 4. 实验解读：哪些数字最重要

### 4.1 Fused Communication + Computation（§4.1）

GEMM + Reduce-Scatter（Figure 11）用dynamic scheduler，因为网络contention让任务完成时间不可预测。
All-Gather + GEMM（Figure 12）用static scheduler，因为ring algorithm的data arrival order是可预测的。

对比baseline：
- cuBLAS+NCCL：sequential，no fusion
- TP-Async（[TorchTitan, Liang et al., 2025](https://openreview.net/forum?id=SFN6Wm7YBI)）：手动async orchestration
- Triton-Distributed v0.0.2-rc（[Zheng et al., 2025](https://arxiv.org/abs/2504.19442)）：编译器baseline
- cuBLASMp（[NVIDIA 2023](https://docs.nvidia.com/cuda/cublasmp/index.html)）：multi-process fused library

**最高1.40x speedup over cuBLAS+NCCL**。关键insight：Triton-Dist在Blackwell上Triton GEMM还没调优好，TP-Async的coarse-grained splitting要么chunk太小SM不饱和要么太大掩盖不了comm latency。ETC的fine-grained dependency能让SM和network都持续busy。

注意这里8192 tokens是大batch bandwidth-bound场景。低batch latency-critical场景在Figure 14。

### 4.2 MoE Layer（§4.2）

模型：Qwen3-30B-A3B（128 experts, top-8）。dynamic scheduler。
对比Triton 3.4.0（SGLang/vLLM用的）、FlashInfer 0.2.14.post1（[Ye et al., 2025](https://arxiv.org/abs/2501.01005)）。

**最高1.23x speedup at 1024 tokens**。关键：
- Data-dependent Event Tensor打破两阶段GroupGEMM之间的global barrier，形成fine-grained pipeline
- 减少wave quantization：SM allocation跨fused operator更平滑
- On-chip dynamic scheduler给irregular token routing提供load balancing

### 4.3 End-to-End LLM Serving（§4.3）

模型：Qwen3-30B-A3B（MoE）和Qwen3-32B（dense）。static scheduler。
对比vLLM v0.11.0rc2、SGLang v0.5.3rc0（都用CUDA Graph + torch.compile + PDL）。

**MoE Qwen3-30B-A3B**：
- batch=1: ETC 1.48x over vLLM, 1.20x over SGLang
- 全batch size都胜

**Dense Qwen3-32B**：
- batch=1: 1.15x over vLLM, 1.09x over SGLang at batch=64

**TP=4 Qwen3-32B**：
- 0.99x–1.06x over vLLM（基本持平）
- SGLang比ETC和vLLM都快，因为SGLang的CPU scheduler在分布式场景下runtime overhead更低

paper诚实承认了ETC在TP=4场景的小gap来自：
- compiler-generated GEMM tile没cuBLAS调得精
- serving engine CPU-side overhead高

这两个都是engineering问题，理论上可以解决。

### 4.4 Warmup Overhead（§4.4）

Table 1的数据：
| Method | Warmup Time | # JIT Graph Capture |
|--------|-------------|---------------------|
| SGLang (JIT) | 583s | 51 |
| vLLM (JIT) | 123s | 67 |
| ETC (AOT) | 35s | 0 |

**ETC 3.5x faster warmup over vLLM, 16.6x over SGLang**。这个数字在生产环境里有巨大经济价值——每次模型reload/autoscale事件都能省几百秒。

### 4.5 Static vs Dynamic scheduling tradeoff（§4.5）

Table 2（MoE）：dynamic比static多4%–8% speedup over unfused baseline，但batch=1时dynamic反而慢（task push/pop overhead在小batch下相对total latency占比较大）。

Table 3（dense Qwen3-32B, TP=4）：static比dynamic多6%–8% speedup over unfused。dynamic在distributed setting下慢，因为push task to remote queue overhead大。

**结论**：data-dependent且irregular → dynamic；regular且latency-critical → static。这个tradeoff被ETC统一抽象支持，开发者可以workload-specific选择。

---

## 5. 我对paper的一些观察和潜在问题

### 5.1 抽象层的清晰度

Event Tensor做对的事情是把"task graph"从runtime object提升到compiler IR的first-class object。这和Graphene [Hagedorn et al., 2023](https://doi.org/10.1145/3582016.3582018) 把thread当成tensor的思路类似，但Graphene只做单kernel优化，Event Tensor做multi-operator megakernel fusion + dynamic shape + data-dependent。

Cypress [Yadav et al., PLDI 2025](https://doi.org/10.1145/3729262) 也做task-based GPU computation，但更像"在GPU上跑task graph runtime"，没有compiler IR层面的symbolic shape抽象。

### 5.2 Limitations / 没说清楚的地方

1. **Centralized global queue的contention**：Appendix E说用early push hide scheduling overhead，但大规模SM时contention没量化。B200有148个SM，每个SM高频访问global queue的atomic op会有限制。Per-SM queue + work stealing可能更好，但工程复杂度上升。

2. **Symbolic shape的runtime instantiation**：paper说"sample representative shapes"，这个机制本身不太elegant。如果shape分布很wide，cache miss率会高。Relax的symbolic shape机制其实更系统化，可以借鉴。

3. **Multi-GPU的Event Tensor**：Figure 6里Event Tensor有 `shard="S[0]"`，distributed场景的Event Tensor如何跨device同步没详细展开。Cross-device dependency在NVLink上怎么高效实现？

4. **Prefetch pass的annotation**：paper说"based on user annotations"，annotation的具体形式没给。理想情况下应该compiler自动推断prefetch机会，但这需要更复杂的dependency analysis。

5. **CPU scheduler的劣势**：Figure 14右图，SGLang在TP=4下比ETC快，因为SGLang的CPU scheduler更轻量。这暗示ETC的GPU megakernel把所有事情都塞到GPU上不一定永远最优——CPU/GPU协同调度在某些场景下仍有价值。

6. **没有跟CUDA Programmatic Dependent Launch（PDL）对比**：PDL是Hopper/Blackwell的新feature，允许kernel在还没完全结束时launch dependent kernel，partial overlap。paper只提了PDL是baseline用的，没单独ablation。如果PDL做得好，是不是可以避免megakernel的复杂度？

7. **代码gen的GEMM性能gap**：paper承认compiler-generated GEMM tile没cuBLAS精调。这是所有ML compiler的通病（TVM/Triton/XLA都有这个问题）。在B200 Blackwell上尤其明显，因为新的tensor core指令（`wgmma.async`、`tcgen05`）需要更精细的instruction scheduling。

### 5.3 跟你的micrograd/nanoGPT直觉的对照

Andrej你之前在[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)和[micrograd](https://github.com/karpathy/micrograd)里强调"理解反向传播和计算图的minimum viable implementation"。Event Tensor在精神上类似——它把"task graph + dependency"最小化成"integer tensor + atomic + spin-wait"，把复杂的runtime task executor简化掉。

Megakernel时代，整个GPU就像一个多核CPU，SM像core，Event Tensor像CPU memory的lock-free queue。区别是GPU有148个core、没有cache coherence、只有shared memory per-SM和global memory。这种model下，atomic + spin-wait是最自然的同步原语。Event Tensor把它们抽象到compiler IR层，让编译器做调度而不是runtime。

### 5.4 未来工作方向（paper没说但很明显）

1. **自动从普通computation graph生成Event Tensor graph**：现在需要用户annotation dependency，paper §6结尾也提到了这个vision。这需要alias analysis + dataflow analysis来推断fine-grained dependency。

2. **Speculative execution under Event Tensor**：speculative decoding里draft model和verify model的overlap如果用Event Tensor表达会很自然。

3. **Multi-node扩展**：NVLink-only的distributed Event Tensor已经实现，跨node的IB/RDMA场景需要新机制（RDMA write可以直接做cross-node atomic）。

4. **与Hopper/Blackwell async特性结合**：`wgmma.async`、`cp.async.bulk`、TMA这些async指令天然适合Event Tensor——每个async指令的completion就是event。

5. **Online auto-tuning of static schedule**：现在static schedule用round-robin，理论上可以做profile-guided schedule优化，类似 compilers 的 PGO。

---

## 6. 跨论文的更深层联想

### 6.1 与TVM Relax的symbolic shape

[Lai et al., Relax 2025](https://arxiv.org/abs/2401.16792) 已经做了symbolic shape的first-class support。Event Tensor在Relax之上加了一层event dependency。两者结合后，一个完整的symbolic-shape dynamic LLM program可以AOT编译，runtime zero compilation overhead。这解决了一个困扰生产LLM serving系统的核心问题。

### 6.2 与CUDA Graph的对比本质

CUDA Graph本质是"capture-then-replay"模式，需要static graph。Event Tensor是"compile template-then-runtime-instantiate"模式，允许symbolic dimension + data-dependent dependency。这是抽象上的本质进步——把runtime dynamism从graph capture层面下推到event counter层面，每个event的wait_count可以runtime设置。

### 6.3 与DeepSeek DualLab / FlashInfer的overlap哲学

DeepSeek的[FlashInfer](https://arxiv.org/abs/2501.01005)和DualLab系列都是在kernel内做overlap，手工engineering。Event Tensor把这些手工overlap系统化成compiler pass。GEMM+RS的overlap之前DeepSeek DualLab、Triton-Dist都做了，但都是手工写。ETC让任何用户用annotation就能得到这个优化。

### 6.4 跟Megakernel的Stanford那一篇

[Stanford "Look Ma, No Bubbles"](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)是LLaMA-1B的single-batch hand-tuned megakernel。ETC的作者和那篇paper有交集（CMU-Stanford合作）。可以看作ETC是"如何把Stanford手工megakernel的trick系统化、自动化、动态化"的compiler回答。

### 6.5 跟Programmatic Dependent Launch的对比

NVIDIA Hopper引入PDL（[Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch)）：一个kernel可以提前launch下一个kernel，partial overlap。PDL解决了"kernel boundary barrier"的一部分，但仍然是coarse-grained（kernel-to-kernel），不是tile-to-tile。Event Tensor是tile-to-tile的fine-grained dependency，比PDL更细，但需要compiler支持。

理论上PDL和Event Tensor可以互补：跨megakernel用PDL，megakernel内部用Event Tensor。paper没探索这个组合。

### 6.6 跟Legion/Realm的关系

paper自己承认related work里提到了Legion [Bauer 2012](https://ieeexplore.ieee.org/document/6480535) 和Realm [Treichler 2014](https://dl.acm.org/doi/10.1145/2628071)。Legion的logical region是数据sharding抽象，Realm的event是completion event。Event Tensor的核心新颖性是把event组织成tensor（带shape），把sharding和symbolic dimension复用。本质上是"把Realm的event机制重新设计成compiler IR的first-class object"。

---

## 7. 总结：哪些insight值得带走

1. **Event Tensor的核心抽象**：event → multi-dimensional tensor with symbolic shape，让compiler对tensor的所有优化（symbolic shape、sharding、tiling）都复用到dependency管理上。

2. **Static vs Dynamic scheduling的统一**：同一个Event Tensor program可以走两条pass，workload-specific选择。Regular latency-critical用static，data-dependent irregular用dynamic。

3. **Lowering到minimal runtime**：Event Tensor → integer tensor，notify → atomic decrement，wait → spin-wait。整个runtime只是几个integer tensor + scheduler queue。这是compiler-only solution的胜利。

4. **AOT编译 + runtime instantiate**：symbolic shape让warmup从583s降到35s，这个数字本身就是论文的杀手锏。

5. **Data-dependent dynamism通过两个机制**：runtime event update（counter runtime设置）+ runtime task triggering（runtime决定trigger的tile数量）。这两个机制覆盖了MoE等data-dependent control flow场景。

6. **Open question**：自动dependency inference（从普通computation graph自动生成Event Tensor annotation）、multi-node扩展、跟PDL的组合、compiler-generated GEMM追赶cuBLAS、online auto-tuning of static schedule。

---

## References

- Paper原文：[Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel](https://arxiv.org/abs/2512.22219)（推测arxiv编号，从reference看Mirage是2512.22219，ETC应该同期）
- Stanford No Bubbles: [Spector et al., 2025](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- Mirage Persistent Kernel: [Cheng et al., 2025](https://arxiv.org/abs/2512.22219)
- TVM Relax: [Lai et al., 2025](https://arxiv.org/abs/2401.16792)
- Axe DSL: [Hou et al., 2026](https://arxiv.org/abs/2601.19092)
- Triton: [Tillet et al., 2019](https://dl.acm.org/doi/10.1145/3311508.3311517)
- FlashInfer: [Ye et al., 2025](https://arxiv.org/abs/2501.01005)
- Triton-Distributed: [Zheng et al., 2025](https://arxiv.org/abs/2504.19442)
- vLLM: [Kwon et al., 2023](https://arxiv.org/abs/2309.06180)
- SGLang: [Zheng et al., 2024](https://arxiv.org/abs/2312.07104)
- CUDA Graphs: [NVIDIA blog](https://developer.nvidia.com/blog/cuda-graphs/)
- Legion: [Bauer et al., 2012](https://ieeexplore.ieee.org/document/6480535)
- Realm: [Treichler et al., 2014](https://dl.acm.org/doi/10.1145/2628071)
- Graphene: [Hagedorn et al., 2023](https://doi.org/10.1145/3582016.3582018)
- Cypress: [Yadav et al., 2025](https://doi.org/10.1145/3729262)
- CuSync: [Jangda et al., 2024](https://arxiv.org/abs/2305.13450)
- PyTorch 2 / torch.compile: [Ansel et al., 2024](https://arxiv.org/abs/2401.17811)
- NVIDIA Programmatic Dependent Launch: [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch)
- B200 / Blackwell PTX: [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- FlashMoE: [Aimuyo et al., 2025](https://arxiv.org/abs/2506.04667)
- TorchTitan: [Liang et al., 2025](https://openreview.net/forum?id=SFN6Wm7YBI)

如果你想更深入聊某一节，比如static scheduling pass的具体算法、dynamic scheduler的lock-free queue实现、MoE的event update细节、或者Relax symbolic shape如何被Event Tensor复用——我都可以再展开。我个人觉得最值得深挖的是 §3.3 lowering里 `notify()`/`wait()` 究竟映射到哪些具体PTX指令，以及Appendix E的early push策略对critical path的影响分析——这两个engineering细节决定了megakernel性能上限。
