---
source_pdf: PyTorch FSDP Experiences on Scaling Fully Sharded Data Parallel.pdf
paper_sha256: 7ee0f202b567627d668deae8b786e2a7a48920fbed22cc125a060f4c6da27cab
processed_at: '2026-08-06T07:16:59-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

你想要 train 那种几百亿甚至上千亿参数的大模型，单张 80GB A100 根本塞不下，连初始化都报 OOM。以前的 DDP 思路太愣了，要求每个 GPU 都完整复印一整本"武功秘籍"（model parameters, gradients, optimizer states），模型一小就挺好，模型一大直接把显卡撑爆。

FSDP 的核心 intuition 极其简单：**把书撕了，一人拿几张页，要用的时候大家拼出来看，看完立刻拆散还回去。**

下面我用工程师的人话，给你拆解 FSDP 到底在玩什么把戏。

### 1. 核心机制：FlatParameter 与 Sharding

FSDP 不是随便乱撕书。它会把一个 module 里的所有 params（比如 weight, bias）拉平、首尾相连变成一个 1D 的大向量，叫 **FlatParameter**。然后把这个大向量平均切成 $W$ 份（$W$ 是 GPU 数），每个 GPU 只兜里揣着 $\frac{1}{W}$ 的残页。

**峰值显存公式背后的直觉**：
$$ \text{Memory} \propto \sum_{i=1}^N \frac{\psi_i}{F} + \max_{i=1}^N \psi_i $$
左边那项 $\sum \frac{\psi_i}{F}$ 是所有残页加起来的体积，极小，固定不变。右边那项 $\max \psi_i$ 是当前正在 AllGather 拼出来计算的那一“大块”的体积。决定你能不能跑得动的，就是右边这个 $\max \psi_i$。

这就带来了一个极其经典的 **Trade-off**：
你把模型切得稀碎（$N$ 极大），拼出来的单块就极小，peak memory 爆降。但是通信次数 $O(N)$ 线性飙升，吞吐量拉胯。你如果把整个模型当成一块（$N=1$），通信最少，但拼出来的还是一整个模型，等于没省显存。所以怎么 wrap 模型，决定了你在“显存”和“速度”之间怎么走钢丝。

### 2. 通信优化：把时间藏起来

FSDP 最大的性能杀手就是 AllGather（拼书）和 ReduceScatter（拆梯度书）。如果这些操作串行在 GPU 计算后面，pipeline 里全是气泡。

**Overlap 的关键**：
FSDP 用独立的 CUDA stream 发射通信。GPU 在算当前层的同时，背后偷偷把下一层的参数 AllGather 过来。在 backward pass 里更狠，叫做 **Backward Prefetching**。因为 PyTorch 是 eager execution，它不知道下一步要算啥。FSDP 就把上一轮 forward 的执行顺序录下来，倒着放，当作这一轮 backward 顺序的预测。算当前层的 backward 时，提前把下一层的参数拼好。这招在 GPT-175B 上直接提速 18%。

### 3. 底层微操：Rate Limiter 与 Caching Allocator 斗智斗勇

这个细节非常精彩，属于你不读源码根本不知道的坑。

PyTorch 的 CUDA caching allocator 是在 CPU 线程上决定显存块复用的。如果 CPU 发指令太快，GPU 算得太慢，CPU 会疯狂往前冲，给 AllGather 的结果分配 destination tensor。因为 GPU 还在算，旧显存块还没释放，allocator 不敢复用，只能去申请新显存。最后显存碎片化严重，触发 `cudaMalloc` retry，这玩意极慢，性能直接崩盘。

FSDP 搞了个 **Rate Limiter**，强行限制：**最多只允许 2 个 inflight AllGather**。这相当于给 CPU 踩了脚刹车，逼它等 GPU 把旧显存吐出来再发新指令。在 T5-11B 上，这招直接带来 5x speedup。如果你发现训练死慢，去查 `torch.cuda.memory_stats()` 里的 `num_alloc_retries`，如果这个数字在涨，赶紧开 Rate Limiter。

### 4. 拓扑感知：Hybrid Sharding

数据中心网络通常是“机内 NVLink 飞快，机间网线拉胯”。Full Sharding 会让所有 GPU 跨机疯狂传参数，网络堵死。

FSDP 搞了 **Hybrid Sharding**。你不用在全球所有 GPU 间撕书，只在机器内部（比如 8 张卡之间）撕书，机器之间还是各自保留完整模型副本。
梯度同步分两步：先在机内 8 张卡做 ReduceScatter，再在机间做 AllReduce。跨机流量从 $3M\frac{W-1}{W}$ 暴跌到 $2M\frac{W-1}{GW}$（$G$ 是单机 GPU 数）。这就把通信压力限制在了高带宽的机内，完美匹配了物理拓扑。

### 5. 混合精度的意外红利

传统混合精度要同时存 FP32（给 optimizer）和 BF16（给计算）两份。FSDP 发现了一个 bug 级别的 feature：平时只存 shard 出来的小份 FP32，真正算的时候，AllGather 拼出来的那一大块直接转成 BF16。
峰值显存公式里那个可怕的大头 $\max \psi_i$，直接从 $K_{full}$（4 bytes）降到了 $K_{low}$（2 bytes），直接砍半。这意味着你能塞进更大的 batch size。

### 6. 踩坑警告：FSDP 不是银弹

**Mathematical Equivalence 陷阱**：
Optimizer step 是在 sharded parameters 上跑的。因为 FlatParameter 把参数边界打乱了，任何依赖单个参数完整形状的 optimizer（比如算 weight decay 算 norm，或者二阶优化器）算出来的结果和单卡跑完全不一样。这是系统设计和数学正确性的妥协。

**Shared Parameters 陷阱**：
如果你的模型有参数共享，必须保证这个共享参数被分在它们共同的最低祖先 FSDP unit 里。否则前一个 unit 算完把它 free 了，后一个 unit 调用时直接报错找不到数据。

### 总结

FSDP 就是一个极其精细的“拼图游戏”。它把显存压力转化成了通信压力，然后通过 stream overlap、prefetching、rate limiting 这些系统级骚操作，把通信时间藏到计算时间里。本质上它是对硬件拓扑、显存分配机制和计算图的极度压榨，让 PyTorch 能够以一种 drop-in 的方式，跑起几百亿的大模型。

**References:**
* PyTorch FSDP Official Docs: https://pytorch.org/docs/stable/fsdp.html
* DeepSpeed ZeRO (FSDP 的精神导师): https://arxiv.org/abs/1910.02054
* Paper PDF (PVLDB): https://vldb.org/pvldb/vol16/p3848-zhao.pdf

---

这篇 paper 详细介绍了 PyTorch FSDP (Fully Sharded Data Parallel) 的设计、实现与评估。FSDP 的核心 motivation 在于解决大模型训练中的 OOM (Out of Memory) 问题。传统的 DDP (DistributedDataParallel) 要求每个 GPU 都保留完整的 model parameters, gradients, 并且 optimizer states 也要在每张卡上存一份。当模型参数规模达到 Billions 级别时，单张 GPU 的显存完全无法承受。FSDP 通过 sharding parameters, gradients, optimizer states，打破了单卡显存限制，并且通过一系列系统级优化，实现了近乎线性的 scalability。

下面我将从技术细节出发，深入解析 FSDP 的架构设计和核心公式，帮你 build 出对于大规模分布式训练的 intuition。

### System Design 与直觉构建

#### 1. Model Initialization 的挑战与 Deferred Initialization
传统 PyTorch 强制要求在初始化时将整个 model instance 完全 materialize 到目标设备上。如果模型有 175B 参数，单张 80GB A100 根本装不下，连 `model = MyModel().to('cuda')` 这一步都会 OOM。

FSDP 引入了 **Deferred Initialization** 机制。具体做法是：模型首先在一个 "fake" device (通常对应 PyTorch 中的 meta device) 上实例化。在这个过程中，所有的 Tensor 都不会分配实际的 GPU memory，系统仅仅记录初始化时调用的 operations。然后，FSDP 将模型按 FSDP unit 切分，逐个 unit 地移至真实 GPU 上，并 replay 之前记录的初始化操作。当一个 unit 初始化完成后，立刻将其 parameters shard 掉，释放出空间给下一个 unit。这就像搭积木一样，边搭边压缩，保证显存占用始终在一个 unit 的量级。

#### 2. FlatParameter 与 Memory-Throughput Trade-off
FSDP 会将一个 FSDP unit 内的所有参数 flatten 并 concat 成一个 1D 的 Tensor，称为 **FlatParameter**。

**架构图解析** (参考 Figure 3):
假设有一个 `4x3` 的 `nn.Linear` layer，其 weight 是 `4x3=12` 个 elements，bias 是 `4` 个 elements。总共 16 个 elements。FSDP 将这 16 个 elements flatten 成一个 1D 的 FlatParameter。如果使用 16 GPUs (Global World Size $W = 16$) 进行 Full Sharding，FSDP 会将这个 FlatParameter 分成 16 份。因为 16 能被 16 整除，所以不需要 padding。但如果总 elements 是 15，FSDP 会在右侧 pad 补齐到 16，然后再切分。这种 flatten-concat-chunk 算法保证了 NCCL (NVIDIA Collective Communication Library) 的 AllGather 和 ReduceScatter 操作可以直接在连续的内存上执行，避免了额外的内存拷贝。

**Memory Formula 解析**:
假设模型总共有 $\Psi$ 个 elements，FSDP 将其划分为 $N$ 个 FlatParameters，每个 FlatParameter 的大小为 $\psi_1, \psi_2, ..., \psi_N$。Sharding factor 为 $F$ (Full Sharding 时 $F = W$)。FSDP 的 peak parameter memory contribution 为：
$$ O\left( \sum_{i=1}^N \frac{\psi_i}{F} + \max_{i=1}^N \psi_i \right) $$

变量解释：
*   $\psi_i$：第 $i$ 个 FlatParameter 包含的参数数量。
*   $F$：Sharding factor，即参数被分片到的 rank 数量。
*   $\sum_{i=1}^N \frac{\psi_i}{F}$：这部分是所有 FlatParameter 在 sharded 状态下占用的总显存。因为每个 rank 只保留 $\frac{1}{F}$ 的 shard，所以这部分显存占用非常小且固定。
*   $\max_{i=1}^N \psi_i$：这部分是在 forward 或 backward 过程中，AllGather 出来必须被 fully materialized 的那个最大的 FlatParameter 的显存占用。

**Intuition**: FSDP 显存占用由两部分决定：一小部分是所有模型参数的 sharded 副本（极小），另一部分是当前正在计算的那个最大的 FSDP unit 的全量大小。这就产生了一个经典的 **memory-throughput trade-off**。如果你把 FSDP unit 切得非常碎（$N$ 很大，$\max \psi_i$ 很小），那么 peak memory 就小，但是通信次数 $O(N)$ 就多，吞吐量下降。如果你把整个模型当成一个 FSDP unit（$N=1$），那么通信次数最少，但 peak memory 就是 $\psi_1 = \Psi$，与不 shard 没区别。因此，合理的 wrap 策略至关重要。

#### 3. Hybrid Sharding 与网络拓扑感知
在 datacenter 中，机内 (intra-node) 通常有极高的 NVLink 带宽，而机间 (inter-node) 是带宽较低的 RoCE 或 InfiniBand。Full Sharding 在大集群上会导致大量的跨机 AllGather 通信，拖慢速度。Hybrid Sharding 介于 Full Replication (类似 DDP) 和 Full Sharding 之间，其 sharding factor $1 < F < W$。

**Gradient Reduction 公式解析** (参考 Equation 1):
对于 Hybrid Sharding，梯度同步分两步：先在 sharded group $S_i$ 内做 ReduceScatter，然后在 replicated group $R_j$ 内做 AllReduce。其数学等价性证明如下：
$$ \sum_{r=1}^W g_r = \sum_{i=1}^{W/F} \sum_{r \in S_i} g_r $$
变量解释：
*   $W$: Global world size。
*   $F$: Sharding factor。
*   $g_r$: Rank $r$ 上的 gradient。
*   $S_i$: 第 $i$ 个 sharded group，包含 $F$ 个 ranks。
*   $W/F$: Replicated group 的数量，也就是有 $W/F$ 个 sharded groups。

这个公式表明，所有 rank 上的梯度之和，等于各个 sharded group 内部梯度之和的再累加。因此，可以先在 $S_i$ 内部做 ReduceScatter（将组内梯度相加并分片），再在互补的 $R_j$ 组内做 AllReduce（将不同组的同分片梯度相加并广播）。这样做极大地减少了跨机的网络流量。对于 $M$ 大小的模型，Full Sharding 的 cross-host traffic 为 $3M \frac{W-1}{W}$，而 Hybrid Sharding 仅为 $2M \frac{W-1}{GW}$ (其中 $G$ 为单机 GPU 数)。

#### 4. Communication Overlap 与 Backward Prefetching
在 eager execution 模式下，FSDP 无法预知下一步需要哪个 FlatParameter。为了 overlap 通信与计算，FSDP 使用了独立的 CUDA stream 来发射 AllGather 通信，避开 default stream 上的计算依赖。

在 backward pass 中，FSDP 需要执行 ReduceScatter (当前 unit 的梯度分片) 和 AllGather (下一个 unit 的参数恢复)。如果在一个 NCCL stream 中串行执行，当前 ReduceScatter 会阻塞下一个 AllGather，造成 bubble。**Backward Prefetching** 机制通过记录 forward pass 的逆序作为 backward 顺序的 proxy，在当前 unit 计算 backward 的同时，提前发出下一个 unit 的 AllGather 请求。实验数据表明 (参考 Figure 6b)，在 GPT-175B 模型上，开启 Backward Prefetching 带来了约 18% 的 speedup。

#### 5. Rate Limiter 与 CUDA Caching Allocator
这是一个非常底层且精彩的优化。PyTorch 的 CUDA caching allocator 是在 CPU 线程上决定内存块复用的。如果 CPU 线程跑得比 GPU 计算快，它会疯狂地发射 AllGather 请求并为其分配 destination tensor。由于不同 CUDA stream 之间没有明确的顺序保证，allocator 无法确定某块显存是否还被 consumer stream (默认计算流) 中的 kernel 使用，导致无法复用。这会迫使 allocator 去请求新的显存，直到触发 OOM 或导致严重的 defragmentation (通过 `cudaMalloc` retry 触发，极慢)。

FSDP 实现了一个 **Rate Limiter**，强制限制最多只有 2 个 inflight AllGathers。这保证了 CPU 不会跑得太快，使得之前 AllGather 用的显存已经被计算流消费完毕并释放，从而可以被 allocator 复用。实验表明 (参考 Figure 6c)，对于 T5-11B 模型，Rate Limiter 带来了高达 5x 的 speedup。但是 DeepViT 模型因为 computation 本身就慢，CPU 没有跑太快，Rate Limiter 反而拖慢了 5% 的速度。这就要求用户通过 `torch.cuda.memory_stats()` 中的 `num_alloc_retries` 来判断是否真的发生了 defragmentation，再决定是否开启限流。

#### 6. Native Mixed Precision 的内存优势
传统的 Mixed Precision 会同时保存 FP32 (optimizer states) 和 BF16/FP16 (computation) 两份参数。如果参数量为 $\Psi$，低精度字节数为 $K_{low}$，高精度字节数为 $K_{full}$，总内存为 $(K_{low} + K_{full})\Psi$。

FSDP 的优势在于它永远只在 GPU 上保留 sharded 的 FP32 副本，而在计算时 materialize 出来的 unsharded 参数只保留低精度。公式如下：
$$ \frac{K_{full}}{F} \sum_{i=1}^N \psi_i + K_{low} \max_{i=1}^N \psi_i $$
变量解释：
*   $K_{full}$：高精度 (如 FP32) 每个元素的字节数 (4 bytes)。
*   $K_{low}$：低精度 (如 BF16) 每个元素的字节数 (2 bytes)。
*   $F$：Sharding factor。

这里的关键点在于，原先巨大的 $\max_{i=1}^N \psi_i$ 项，从高精度内存占用 $K_{full} \max \psi_i$ 降低到了低精度内存占用 $K_{low} \max \psi_i$。由于 $\max \psi_i$ 是 peak memory 的决定性因素，这个优化直接砍掉了 peak memory 中很大一部分，使得训练更大的模型成为可能。

### Evaluation 数据解析
论文在 512 张 80GB A100 上进行了测试，测试模型包括 T5-11B, minGPT-175B 和 DHEN (768B sparse + 550M dense) 推荐模型。
*   **Model Scale**: 当模型小于 2.28B 时，FSDP 与 DDP 性能相当。超过 2.28B 后 DDP OOM，FSDP 依然坚挺，开启 BF16 后 TFLOPS 显著提升。
*   **175B 模型 Scalability**: 在 512 GPUs 上，batch size 设为 2 时，达到了 186 TFLOPS per GPU，这相当于 A100 理论峰值 (312 TFLOPS BF16) 的约 60%。
*   **DHEN 模型中的 RAF vs NRAF**: 
    *   RAF (reshard-after-forward): Forward 之后立刻释放 unsharded 参数，backward 之前再 AllGather 回来。省显存，但通信开销大。
    *   NRAF (no-reshard-after-forward): Forward 之后保留 unsharded 参数直到 backward 结束。费显存，但省通信。这也是 FSDP 提供给用户控制 trade-off 的手段之一。

### Limitations
FSDP 也存在一些局限性。首先是 **Mathematical Equivalence** 的问题。因为 optimizer step 是在 sharded parameters 上运行的，FlatParameter 的数据布局不尊重原始单个 parameter 的边界。因此，任何依赖于单个参数整体数值（比如 weight decay 或 vector norm 计算）或其二阶矩的 optimizer，在 FSDP 下可能会产生与本地训练不同的结果。其次是 **Shared Parameters** 的问题。如果模型中有共享参数，必须将其放在它们共同的最低祖先 FSDP unit 中，否则前一个 unit 释放了该参数，后一个 unit 使用时会报错。

### References
*   PyTorch FSDP Official Documentation: https://pytorch.org/docs/stable/fsdp.html
*   FSDP Source Code: https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/fully_sharded_data_parallel.py/
*   ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (DeepSpeed): https://arxiv.org/abs/1910.02054
*   Automatic Cross-Replica Sharding of Weight Update in Data-Parallel Training: https://arxiv.org/abs/2004.13336
*   MiCS: Near-linear Scaling for Training Gigantic Model on Public Cloud: https://arxiv.org/abs/2205.00119
