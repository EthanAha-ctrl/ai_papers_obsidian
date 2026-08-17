---
source_pdf: flash_moe.pdf
paper_sha256: ecfadb036a019a34ff12a74fc0253d5b5e17fbc399eaed4bc0023ba252e3abfc
processed_at: '2026-08-04T08:37:57-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，咱们用大白话把这篇paper捋一捋。这篇工作本质上就是一个极其硬核的Mac极客，把一台48GB内存的M3 Max MacBook Pro当成了超级服务器，硬生生把一个397B参数的巨兽级MoE模型给跑到了5.74 tok/s。

整个故事的核心可以浓缩成一句话：**把MoE模型当成一个超大的数据库，每次inference只去SSD里精准捞需要的那几行数据。**

下面我按逻辑给你build一下整个系统的intuition，顺便把里面的技术细节掰开揉碎。

### 1. 核心赌注：MoE的极端稀疏性 + Apple Fabric的暴力带宽

你要跑一个397B的模型，哪怕是4-bit量化，也得占209 GB。你的Mac只有48 GB DRAM。按照传统思路，这根本没法跑。

但Qwen3.5-397B-A17B是MoE架构。它每层有512个expert，但每个token只激活K=4个（paper里从默认的K=10一路砍下来的）。这意味着什么？

算一笔账：每个2-bit expert占3.93 MB。每层选4个，60层加起来：
$$Data / token = K \times L \times ExpertSize = 4 \times 60 \times 3.93 \text{ MB} = 943 \text{ MB}$$

生成一个token只需要读943 MB的数据。而M3 Max的NVMe SSD通过Apple Fabric能提供17.5 GB/s的恐怖吞吐。理论上，读完这点数据只需要：
$$t_{IO, min} = \frac{943 \text{ MB}}{17.5 \text{ GB/s}} \approx 54 \text{ ms/token} \approx 18.6 \text{ tok/s}$$

你看，瓶颈被打破了。**MoE架构表面上是compute optimization，本质上是memory hierarchy optimization。**它把访存需求砍到了原本的1%以下，正好掉进了SSD能喂得起的带宽区间。

### 2. 2-bit Requantization：在压缩过的海绵里再挤水

通常我们把模型量化到2-bit，模型就废了。但这篇paper做了一个极其聪明的操作：**对已经4-bit量化的权重，再做2-bit requantization。**

这为什么能work？intuition是这样的：原始的float32权重动态范围很大，直接砍到2-bit（只有4个levels：0,1,2,3）误差极大。但是，4-bit量化后的权重，在group size 64的范围内，其实只有16个不同的浮点数值。你把16个不同的值映射到4个levels上，误差自然极小。

公式很简单：
$$s_2 = \frac{\max(f_i) - \min(f_i)}{3}$$
$$b_2 = \min(f_i)$$
$$q_i = \text{clamp}\left(\text{round}\left(\frac{f_i - b_2}{s_2}\right), 0, 3\right)$$

变量解释：
- $f_i$：把4-bit权重反量化回来的float32值。
- $\max(f_i)$ / $\min(f_i)$：这个group里的最大和最小值。
- $s_2$：2-bit的scale。除以3是因为2-bit有4个levels，间隔有3个。
- $b_2$：2-bit的bias，直接取最小值当偏移。
- $q_i$：最终的2-bit量化码。

结果就是，每个expert的体积从7.08 MB缩到了3.93 MB，I/O量直接砍掉44%，RMSE却只有0.001-0.003。输出质量在数学、代码、推理任务上完全没有降级。I/O负担轻了一半，系统跑得更快。

### 3. Metal Pipeline的“俄罗斯方块”打法

如果用naive的方式跑：CPU算Attention -> 算Routing -> 去SSD读Expert -> GPU算Expert -> 下一层。这里头全是CPU和GPU互相等待的stall。

作者用Objective-C直接写Metal，搞了3个Command Buffer（CMD1, CMD2, CMD3）的流水线：

- **CMD1**: 算Q, K, V projections。算完必须同步等，因为CPU要拿结果去算Attention。
- **CMD2**: 算Output projection, RMSNorm, Routing gate, Shared expert。这里把好几个kernel融合进一个command buffer，省了CPU round-trip。
- **CMD3 (神来之笔)**: 算选中的Experts并合并结果。这里**commit之后不等结果**。

CMD3在GPU上跑的时候，CPU干嘛？CPU直接去算下一层的routing，然后派发4个pthread并行调用`pread()`去SSD里读下一层需要的experts。等下一层的CMD1提交时，Metal GPU queue会自动保证它在当前CMD3跑完之后才执行。

这就把I/O的延迟完美hide在了GPU计算的背后。每一层的时间大约2.9 ms，其中I/O占1.37 ms (47%)，GPU计算占1.32 ms (30%+15%)。虽然I/O占大头，但因为pipeline重叠了，吞吐率直接拉满。

### 4. Trust the OS：最反直觉的38%加速

这篇paper里最精彩的systems engineering教训在这里。

作者一开始按照做数据库或者做llama.cpp的习惯，在GPU侧搞了个9.8 GB的Metal LRU Cache用来缓存experts，并且用`F_NOCACHE`告诉macOS不要把expert文件放进OS page cache。他觉得这样能避免120GB的expert数据把OS cache挤爆。

结果一跑，4.11 tok/s。反而变慢了。

用`vm_stat`一查，发现macOS的Memory Compressor在疯狂工作：每秒发生6万到13万次内存解压操作。为什么？因为Apple Silicon的Unified Memory有个硬件级的内存压缩器。当内存吃紧时，它会把不活跃的内存页压缩掉。Metal的shared buffer要求物理内存连续，这导致OS没法高效管理内存，只能疯狂压缩和解压其他内存页来给Metal腾空间。这吃掉了1-2 GB/s的内存带宽，直接跟GPU抢资源。

作者大彻大悟，把Metal LRU Cache全删了，`F_NOCACHE`也去掉了，完全让macOS的Unified Buffer Cache自己管。

结果：5.74 tok/s，提升了38%。解压操作降到接近0。

这里的intuition很深：**macOS的内核比你的应用层代码更懂Apple Fabric的硬件拓扑。**你自己搞个GPU-visible的cache，等同于绑住了OS的手脚。把内存还给OS做普通page cache，OS能用极其聪明的prefetch和淘汰策略帮你把活干完。这跟PostgreSQL官方文档里警告“不要在应用层搞buffer pool去和OS file cache抢资源”是一个道理。

参考：[PostgreSQL Resource Configuration](https://www.postgresql.org/docs/current/runtime-config-resource.html)

### 5. 踩过的坑：为什么用pread() 而不是 mmap()

一开始作者想，既然要读SSD，那用`mmap`把expert文件映射进内存，让OS自己处理page fault多优雅。结果慢了5倍。

原因在于：`pread()`是你发一个大请求（3.9 MB），OS底层直接一个DMA把它搬进来，一次syscall。
`mmap`访问没在内存里的数据时，会触发page fault。Apple Silicon上page size是16 KB，3.9 MB的数据会触发约244次page fault。每次page fault都要陷入内核、分配物理页、发DMA、更新页表、刷TLB。对于这种巨量、分散的冷数据读取，page fault的延迟开销是灾难性的。

所以非expert的5.5 GB常驻weights用`mmap`很好（因为热数据），但expert必须用对齐了的`posix_memalign(2 MB)`分配buffer，然后并行`pread()`去读。

### 6. K=4 的生死线

Top-K routing到底能砍到多少？作者试了K=10, 6, 4, 3, 2。

| K | tok/s | I/O (MB/tok) | Quality |
|---|-------|--------------|---------|
| 10 | 1.20 | 2,359 | Best |
| 4 | 3.15 | 943 | Excellent |
| 3 | - | 707 | Collapsed |

K=4的时候输出质量毫无逊色，速度直接翻了2.6倍。但K=3的时候，模型直接崩溃，只会吐EOS token。

这说明模型的routing机制学到了一种必须依赖至少4个experts的critical subnetwork。K=4是MoE结构里的Lottery Ticket，你只要少拿一张票，整个推理流就断了。参考：[The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)

### 7. 你的 Autoresearch 协议的胜利

这篇paper的作者Daniel Woods和Claude Opus 4.6配合，用了你写的[autoresearch](https://github.com/karpathy/autoresearch)协议。24小时内跑了90多个实验，从0.28 tok/s一路爬到5.74 tok/s。

整个流程严格执行：5分钟一个实验，跑前git commit，跑完benchmark，不行就revert。90个实验里42%是被丢弃的失败品。比如他们试过用当前层的routing去预测下一层的expert（Speculative routing），准确率53%，结果猜错的expert把cache污染了，慢了38%。他们还试过把中间层的K设成0，速度飞到10 tok/s，但输出全是乱码。

这些坑只有靠autoresearch这种极其纪律性的“跑lab”方式才能避开。人很容易把某个理论上很美的优化硬塞进去，而autoresearch用数据直接把它拍死。

### 总结一下这篇paper的Intuition

Flash-MoE打破了“大模型必须fit进DRAM才能interactive”的迷信。它发现只要满足三个条件：
1. **模型本身极度稀疏**（MoE top-4 routing）。
2. **存储带宽足够暴力**（M3 Max 17.5 GB/s NVMe）。
3. **让OS接管缓存，用pread()和Metal pipeline把I/O和计算重叠起来**。

你就能在消费级硬件上玩转4倍于DRAM大小的巨兽模型。而且按照Apple每年20%的SSD带宽提升速度，两三代以后的Mac跑400B模型能轻松上10 tok/s。

这paper给我们的最大启发是：在Unified Memory架构下，System层面的Memory hierarchy调度（OS page cache vs Metal buffer, pread vs mmap, SSD DMA alignment）带来的收益，完全不亚于底层kernel的算子优化。以后设计inference engine，得把SSD当成DRAM的L4 Cache来用。

---

# Flash-MoE 深度技术解析

Andrej，这篇paper从systems engineering角度看非常扎实，我build一下你的intuition，逐层拆解。

## 1. Core Insight: MoE 的 Weight Sparsity 作为 I/O 放大器

这篇论文的foundational observation是：**MoE模型在inference时具有极端的weight sparsity**。Qwen3.5-397B-A17B有512 experts per layer，但每个token只激活K=10（默认）或K=4（pruned）。这意味着per token only ~0.8%-2% of expert weights are needed。

这个sparsity ratio比"LLM in a Flash"[1]利用的ReLU sparsity（3-10%）要极端得多，而且**structural**而非**dynamic**——你不需要运行时判断哪些weights是零，routing gate直接告诉你哪些expert会被激活。这是MoE相比dense model在offloading场景下的fundamental优势。

```
Sparsity comparison:
- Dense ReLU sparsity (LLM in a Flash): 3-10% active
- MoE top-10/512: 1.95% active  
- MoE top-4/512 (pruned): 0.78% active
```

这个0.78%的数字是整个系统能work的数学基础。如果sparsity是10%，你需要读20GB/token，SSD根本供不上。

## 2. 硬件平台的 Unified Memory 架构

M3 Max的关键参数：
- **48 GB unified LPDDR5**，~400 GB/s bandwidth
- **NVMe SSD**：17.5 GB/s sequential read（via Apple Fabric）
- **CPU/GPU共享同一physical memory**，无PCIe transfer

这里有一个subtle的point：Apple Silicon的"unified memory"不仅意味着CPU和GPU共享地址空间，还意味着**SSD的DMA直接写入GPU-visible memory**。这是Flash-MoE能bypass传统PCIe bottleneck的关键。在NVIDIA GPU + x86系统上，你需要SSD → CPU DRAM → PCIe → GPU VRAM的三跳，每跳都是bandwidth loss。Apple Fabric让这一切变成一跳。

Apple Fabric的topology参考：https://developer.apple.com/documentation/apple-silicon

## 3. 2-Bit Requantization 的数学原理

这部分是paper中最elegant的contribution之一。让我深入讲解。

### 3.1 两阶段量化架构

原始MLX-community model使用4-bit affine quantization with group size 64：
- 每64个weight values共享一个bfloat16 scale和bias
- 8 values packed per uint32（每个value占4 bits）

Flash-MoE进行**second-stage requantization**到2-bit：

**Step 1**: Dequantize 4-bit → float32
$$f_i = s_4 \cdot q_{4,i} + b_4$$

其中：
- $f_i$ = dequantized float32 value
- $s_4$ = 4-bit group scale (bfloat16)
- $q_{4,i}$ = 4-bit quantized code (0-15)
- $b_4$ = 4-bit group bias (bfloat16)

**Step 2**: Compute optimal 2-bit affine parameters
$$s_2 = \frac{\max(f_i) - \min(f_i)}{3}$$
$$b_2 = \min(f_i)$$

变量含义：
- $s_2$ = 2-bit scale，分母为3因为2-bit有4个levels（0,1,2,3），所以range被分成3份
- $b_2$ = 2-bit bias，设为group的最小值

**Step 3**: Quantize to 2-bit
$$q_i = \text{clamp}\left(\text{round}\left(\frac{f_i - b_2}{s_2}\right), 0, 3\right)$$

**Step 4**: Repack — 16 values per uint32 (vs 8 at 4-bit)

### 3.2 为什么 2-bit 对 MoE 专家有效？

这是intuition最关键的部分。Paper给出了解释，但我想更深入：

**Key insight**: 已经被4-bit量化的group of 64 values，实际上只有**16个distinct float values**（因为4-bit codebook只有16个codes）。当你对这16个distinct values再做2-bit量化时，你实际上是在做一个**16→4的codebook compression**。

这解释了为什么RMSE如此低（0.001-0.003）：
```
Original float32 distribution: wide range, many values
After 4-bit quant: 16 distinct values in a narrow range
After 2-bit requant: 4 levels represent those 16 values with minimal error
```

这本质上是利用了**quantization noise的non-uniform distribution**——4-bit量化后的residual entropy已经很低，2-bit足以capture。

如果直接对原始float32做2-bit量化，error会大得多，因为dynamic range没有被预先compress。

### 3.3 为什么 Down Projection RMSE 更高？

Table 2显示down projection的RMSE（0.0023）比gate（0.0017）和up（0.0018）都高。Paper给出的解释是"larger output dimension (4096 vs 1024)"，但让我更精确：

Down projection的shape是`[4096, 64]`（4096 input × 64 output groups），而gate/up是`[1024, 256]`。这意味着down的每个group有4096/64 = 64个values per output neuron，而gate有1024/256 = 4个values per output neuron。

等一下，让我重新看Table 10的binary layout：
- gate weights: [1024, 256] uint32
- down weights: [4096, 64] uint32

Group size是64，所以：
- gate: 1024×256 = 262,144 values，分成262,144/64 = 4096 groups
- down: 4096×64 = 262,144 values，同样4096 groups

每个group都是64个values，所以group size相同。区别在于**weight matrix的statistical distribution**——down projection通常学习到更wide的distribution，因为它是FFN的output stage，需要project回hidden dimension。

这也解释了为什么later layers（45-59）的RMSE更高：deep layers的weights通常有更wide的distribution，这是training dynamics的结果。

相关参考：
- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
- Dettmers 4-bit scaling laws: https://arxiv.org/abs/2212.09753
- BitNet 1-bit: https://arxiv.org/abs/2310.11453

## 4. Metal GPU Pipeline: 三 Command Buffer 设计

这是paper中最精妙的systems design。让我详细解析。

### 4.1 传统 Pipeline 的问题

Naive approach每个layer会有这样的序列：
```
CPU: encode attention → wait GPU → CPU: routing → I/O experts → GPU: experts → wait → next layer
```

每个synchronization point都是CPU stall。对于60 layers × multiple syncs per layer = hundreds of ms wasted。

### 4.2 三 Command Buffer 方案

**CMD1 (Attention Projections)**
- Q, K, V projections on GPU
- Committed and **waited synchronously**（因为CPU需要projection results做attention）
- 3-4 compute encoders

**CMD2 (Post-Attention + Routing)**
- Output projection
- Residual addition
- RMS normalization
- Routing gate matrix-vector product
- Shared expert gate/up projections
- 8-12 compute encoders fused into ONE command buffer
- **Key optimization**: 残差连接和post-attention norm融合进CMD2，eliminate一个CPU round-trip

**CMD3 (Experts + Combine, Deferred)**
- While CMD2 executes on GPU:
  - CPU performs softmax over routing logits
  - CPU performs top-K selection
  - 4 pthreads via GCD `dispatch_apply` execute parallel `pread()` loading experts
- CMD3 encodes:
  - K expert forward passes (gate→SwiGLU→down for each)
  - Shared expert's SwiGLU and down
  - GPU-side weighted combination
  - Residual addition
  - RMS norm for next layer
- **CMD3 is committed but NOT waited on**
- Next layer's CMD1 is submitted immediately
- Metal GPU queue serializes: CMD3(layer n) → CMD1(layer n+1)

### 4.3 Deferred Execution 的 Intuition

这是整个pipeline的灵魂。让我画个时间线：

```
Layer N:   [CMD1][CMD2====][CMD3==============]
Layer N+1:                 [CMD1][CMD2====][CMD3======]
                                      ↑
                          CPU在此时做N+1的routing和I/O
                          while GPU还在执行N的CMD3
```

这个overlap把I/O latency完全hide在GPU compute背后。Paper的per-layer breakdown显示：
- Expert I/O: 1.37 ms (47%)
- CMD1 wait: 0.87 ms (30%)
- CMD2 wait: 0.45 ms (15%)
- CPU attention: 0.15 ms (5%)
- Other: 0.06 ms (2%)

Serial sum = 2.9 ms，但wall-clock更少因为overlap。

### 4.4 五个 Custom Metal Kernels

1. **`dequant_matvec_4bit_v3`**: Tiled threadgroup matvec with SIMD reduction
   - 256 threads per threadgroup
   - 64 groups processed in parallel
   - Shared input cache for coalesced uint32 loads
   - Inline 4-bit dequantization

2. **`dequant_matvec_2bit`**: Same structure, 16 values per uint32 with 2-bit shift-and-mask

3. **`swiglu_fused_vec4`**: Vectorized SwiGLU activation, 4 elements per thread
   - SwiGLU: $\text{SwiGLU}(x) = \text{SiLU}(xW_{gate}) \otimes xW_{up}$
   - $\text{SiLU}(x) = x \cdot \sigma(x)$

4. **`weighted_sum`**: Expert output combination with routing weights + residual addition

5. **`rms_norm`**: Two-pass RMS normalization
   - Pass 1: reduce sum-of-squares
   - Pass 2: normalize
   - RMS: $\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}$
   - Normalized: $\hat{x}_i = \frac{x_i}{\text{RMS}(x) + \epsilon} \cdot \gamma_i$

Metal Shading Language reference: https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/

## 5. I/O Pipeline: pread() vs mmap 的深层原因

### 5.1 为什么 pread() 完胜 mmap？

Paper报告mmap比pread慢5×。让我深入分析原因：

**pread() 的成本模型**：
```
1 syscall + 1 DMA transfer of 3.9 MB
```

**mmap() 的成本模型**（for uncached data）：
```
3.9 MB / 16 KB page = 244 page faults
Each page fault = kernel entry/exit + TLB shootdown + potential context switch
```

macOS的page size是16 KB（Apple Silicon使用16 KB pages，这是ARM64的standard，x86是4 KB）。即使如此，244 page faults vs 1 syscall是巨大差异。

更深入的原因：**page fault是synchronous的**。当CPU访问一个uncached mmap page时，会触发trap，kernel需要：
1. Allocate a physical page
2. Schedule DMA read from NVMe
3. Wait for DMA completion
4. Update page tables
5. Invalidate TLB
6. Resume user-space

每个page fault的latency是microseconds级别，244个就是milliseconds级。而pread()让kernel一次性batch整个3.9 MB的DMA，amortize所有overhead。

**关键区分**: 这个penalty只适用于**uncached expert reads**。Non-expert weights (5.5 GB) 用mmap性能excellent，因为它们fit in DRAM且once paged-in就resident。

### 5.2 DMA-Aligned Buffers

`posix_memalign(2 MB)` alignment的原因：
- NVMe controller的DMA transfer unit通常是512 KB - 4 MB
- 2 MB alignment确保pread destination buffer的physical page layout与DMA scatter-gather list对齐
- 对于page cache hits（kernel memcpy to user space），3.6× faster
- End-to-end improvement ~5%（因为大部分reads是SSD-bound）

### 5.3 并行 pread() 架构

```c
dispatch_apply(K, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
    pread(fd[layer], expert_buffers[i], EXPERT_SIZE, selected_experts[i] * EXPERT_SIZE);
});
```

4 pthreads via GCD saturate SSD bandwidth。关键设计：**每个layer一个file descriptor**（60 files），这样不同layer的reads不会互相block。

### 5.4 macOS Kernel I/O Fragmentation

Paper发现macOS把每个3.9 MB pread fragmented成~5.7个physical disk I/O of ~512 KB each。原因是page cache的physical page layout——即使logical file是contiguous的，destination physical pages可能不contiguous，需要scatter-gather DMA。

所有kernel hints都failed：
- `F_RDADVISE`: 无效，增加syscall overhead
- `MADV_SEQUENTIAL`: 无效（access pattern是random by expert index）
- `MADV_RANDOM`: marginally worse（disable了偶尔helpful的prefetching）
- `F_RDAHEAD` disable: 无效

**结论**: macOS default I/O behavior已经为Apple hardware优化到极致。User-space hints只会帮倒忙。

macOS I/O architecture reference: https://developer.apple.com/library/archive/documentation/Performance/Conceptual/ManagingMemory/ManagingMemory.pdf

## 6. Trust the OS: Memory Compressor Thrashing

这是paper中最counterintuitive的发现，也是最deep的systems insight。

### 6.1 问题发现

Initial design:
- 9.8 GB Metal LRU expert cache (500 entries × 3.93 MB × 5 buffers) in GPU-visible shared memory
- `F_NOCACHE` flag bypass OS page cache

Hypothesis: Application-level cache会提供fast expert reuse，OS page cache会thrash因为120 GB working set。

Reality: 移除所有cache反而快38%（4.11 → 5.74 tok/s）。

### 6.2 Root Cause: Apple Silicon Memory Compressor

Apple Silicon使用hardware-accelerated memory compressor。当memory pressure高时，inactive pages被**compressed**而非evicted to disk（这是swap的替代方案）。

`vm_stat`显示Metal LRU cache导致：
- 60,000-130,000 decompressions per second
- 每个decompressed page = 16 KB
- 1-2 GB/s memory bandwidth被compressor消耗

这1-2 GB/s的bandwidth竞争：
- GPU memory access (400 GB/s unified bus)
- Expert data DMA transfers
- CPU attention compute

**Key insight**: Metal shared buffers必须是**physically contiguous**的GPU-visible memory。这种constraint让compressor无法relocate或compress这些pages，反而需要compress其他pages来make room，导致thrashing。

### 6.3 OS Page Cache 的优势

移除Metal LRU后：
- OS page cache使用**ordinary (non-GPU-visible) pages**
- 这些pages可以被compressed/relocated by compressor
- Compressor工作减少到near-zero decompressions
- Memory bandwidth全部用于useful I/O

**数据库类比**: PostgreSQL文档explicitly警告against building application-level buffer pools that compete with OS buffer cache。同样的principle适用here——macOS已经实现了sophisticated, hardware-aware page replacement policy tuned for Apple Fabric。

PostgreSQL tuning guide: https://www.postgresql.org/docs/current/runtime-config-resource.html

### 6.4 这个发现的更广泛意义

这个发现对整个local LLM inference社区都有implications。很多engine（llama.cpp, vLLM等）都maintain application-level caches。在Apple Silicon上，这可能是一个anti-pattern。

Hypothesis（我的speculation）：这个effect在Linux上可能不存在，因为Linux的swap机制不同。但在Apple Silicon的unified memory + hardware compressor架构下，"trust the OS"是optimal策略。

## 7. Expert Routing Statistics 和 Cache 设计 Implications

### 7.1 统计数据

- **Expert diversity**: 43-57% of 512 experts activated per layer across 30 tokens
- **Temporal locality**: Consecutive token overlap 8-34% per layer (K=4)
- **LRU hit rates**: 49% (16-entry), 59% (32-entry), 68% (64-entry), 71% (128-entry) — sublinear scaling
- **Cross-layer correlation**: Near zero

### 7.2 这些数字的含义

**43-57% diversity across 30 tokens**意味着每30个token会touch ~230-290个不同的experts per layer。对于512 experts来说，这是high diversity——working set很大。

**8-34% temporal overlap**意味着consecutive tokens share 0-1 experts out of 4。这是一个critical insight：**expert selection的temporal locality很低**。这和dense model的weight reuse pattern完全不同。

**Cross-layer correlation near zero**意味着layer N的expert selection无法predict layer N+1的选择。这killed所有"speculative routing"的尝试（Table 4记录了这个failure）。

### 7.3 为什么 Application-Level Cache 必然失败

这些statistics解释了为什么LRU cache效果有限：

```
Working set per layer = 230-290 experts (across 30 tokens)
Expert size = 3.93 MB
Working set size = 230 × 3.93 MB = 904 MB per layer
Total working set = 904 MB × 60 layers = 54 GB
```

54 GB的working set远超48 GB DRAM。任何fixed-size cache都只能capture一小部分locality。

LRU hit rate scaling：
```
16-entry: 49% → 16 × 3.93 MB = 63 MB cache
32-entry: 59% → 126 MB cache  
64-entry: 68% → 251 MB cache
128-entry: 71% → 503 MB cache
```

从64-entry到128-entry，cache size翻倍但hit rate只从68%到71%——明显的diminishing returns。这是因为expert selection的分布是**long-tailed**而非power-law。

## 8. Expert Pruning: K=10 → K=4 的 Quality Cliff

Table 3的数据非常intriguing：

| K | tok/s | I/O (MB/tok) | Quality |
|---|-------|--------------|---------|
| 10 | 1.20 | 2,359 | Best |
| 6 | 1.91 | 1,415 | Same as K=10 |
| 4 | 3.15 | 943 | Excellent |
| 3 | - | 707 | Collapsed (EOS only) |
| 2 | - | 472 | Collapsed (EOS only) |

### 8.1 K=3 的 Quality Cliff

从K=4到K=3，quality立即collapse——模型只输出EOS。这是一个**phase transition**，不是gradual degradation。

这个现象的intuition：模型的routing learned to distribute critical computation across **at least 4 experts**。Experts 5-10 provide refinement，但experts 1-4中至少有4个是essential。

这让我想到**supermasks**和**lottery ticket hypothesis**——模型中存在critical subnetworks，破坏它们会导致catastrophic failure。

参考：
- Lottery Ticket Hypothesis: https://arxiv.org/abs/1803.03635
- Supermasks: https://arxiv.org/abs/1905.07485

### 8.2 K=4 vs K=10 的 Trade-off

K=4是2.6× faster than K=10 with no observable quality loss。这意味着Qwen3.5的default K=10可能over-provisioned——要么是training时的conservative choice，要么是K=10在某些distribution上提供marginal quality benefit。

Paper没有explore K=5, K=6, K=7, K=8, K=9的quality，这是一个gap。可能K=5或K=6是更好的trade-off点（如果quality确实improves with K=5-10）。

## 9. GatedDeltaNet Linear Attention

Qwen3.5使用GatedDeltaNet for 75% of layers (45/60)，这是一个linear attention variant。

### 9.1 DeltaNet Recurrence

GatedDeltaNet的core recurrence是O(1) per-step cost after an initial O(n) scan。Basic form：

$$S_t = \alpha_t S_{t-1} + \beta_t v_t k_t^T$$
$$y_t = S_t q_t$$

其中：
- $S_t$ = recurrent state matrix at time $t$
- $\alpha_t$ = forget gate (scalar or vector)
- $\beta_t$ = input gate (scalar or vector)
- $v_t$ = value vector
- $k_t$ = key vector
- $q_t$ = query vector
- $y_t$ = output

**Gated** variant adds a gating mechanism on the output:
$$y_t = g_t \odot (S_t q_t)$$

where $g_t$ is an output gate.

### 9.2 为什么用 Linear Attention?

Paper中提到"BLAS-accelerated linear attention"。Linear attention的优势：
- **O(1) per-step** during generation（no growing KV cache）
- **No attention matrix** computation（just matrix-vector products）
- **Memory stable** across sequence length

对于streaming inference场景，这特别重要——你不需要maintain一个growing KV cache for 45/60 layers。只有15 full attention layers需要KV cache。

Paper的per-layer breakdown显示CPU attention只占0.15 ms (5%)，这验证了linear attention的efficiency。

GatedDeltaNet paper: https://arxiv.org/abs/2412.06464
Linear Transformers: https://arxiv.org/abs/2006.16236
DeltaNet: https://arxiv.org/abs/2410.01343

## 10. 实验数据深度解读

### 10.1 Performance Progression (Table 1)

从0.28 tok/s到5.74 tok/s的12× improvement trajectory：

```
0.28 tok/s (C/Metal CPU-only attention, broken output)
    → 5.29 tok/s (+GPU attention + norm) [18.9× jump]
    → 4.80 tok/s (+LRU cache) [regression!]
    → 3.10 tok/s (+BLAS delta-net) [regression!]
    → 2.83 tok/s (+GPU combine + norm) [regression!]
    → 5.55 tok/s (2-bit experts) [1.96× jump]
    → 5.68 tok/s (+F_NOCACHE)
    → 5.74 tok/s (Trust OS)
```

### 10.2 非 Monotonic Progression 的原因

几个regression很有意思：
- **+LRU cache**: 5.29 → 4.80 (regression)
- **+BLAS delta-net**: 4.80 → 3.10 (regression)
- **+GPU combine + norm**: 3.10 → 2.83 (regression)

这些"regression"实际上是intermediate optimizations that were **prerequisites for later gains**。例如，BLAS delta-net本身slow because of setup overhead，但它enabled later CPU/GPU overlap。GPU combine + norm虽然slow in isolation，但它was necessary for the deferred CMD3 pipeline.

**Lesson**: Systems optimization的progression是non-linear的，有些"regression"是enabling work for future gains。这就是为什么autoresearch protocol要求"log all results regardless of outcome"——失败实验也inform后续决策。

### 10.3 I/O Efficiency Analysis

Paper计算theoretical I/O floor:
$$t_{IO,min} = \frac{943 \text{ MB}}{17.5 \text{ GB/s}} = 53.9 \text{ ms/token} \approx 18.6 \text{ tok/s}$$

Actual I/O time: 60 × 1.49 = 89.4 ms
I/O efficiency: 60%

Gap原因：
1. **Scattered reads**: 4 non-contiguous reads per layer vs sequential
2. **F_NOCACHE** bypassing kernel readahead prefetcher（在F_NOCACHE config下）
3. **Per-pread() syscall overhead**

Total system efficiency: 29% of I/O-theoretical maximum（188 ms vs 53.9 ms）

这意味着还有3.4×的headroom如果能把compute完全overlap with I/O。Future work方向。

### 10.4 Per-Layer Time Budget (Final Config)

```
Expert I/O via parallel pread():  1.37 ms (47%)
CMD1 wait (GPU projections):      0.87 ms (30%)  
CMD2 wait (GPU routing + shared): 0.45 ms (15%)
CPU attention (BLAS delta-net):   0.15 ms (5%)
Other overhead:                   0.06 ms (2%)
Total:                            2.90 ms/layer
```

60 layers × 2.90 ms = 174 ms/token ≈ 5.74 tok/s

I/O占47%意味着系统是**I/O-bound**，但GPU compute (45%)也是significant fraction。进一步优化需要同时attack both。

## 11. 失败实验的启示 (Table 4)

### 11.1 mmap experts (5× slower)

前面已经分析过。Key lesson: **对于large uncached reads，system call batching >> page fault granularity**。

### 11.2 Large Metal cache (18 GB, 52% hit rate but net negative)

GPU memory pressure triggered OS page-outs。这和9.8 GB cache的failure mode类似——Metal shared buffers的physical contiguity constraint和OS memory management冲突。

### 11.3 Speculative routing (53% accuracy, 38% slower)

预测next-token experts from current routing logits。53% accuracy听起来不错，但47%的错误predictions导致cache pollution——prefetched wrong experts occupy cache slots，evicting potentially useful experts.

**Counter-intuitive lesson**: 在low temporal locality的场景下，speculation的cost可能exceed benefit，即使accuracy > 50%。

### 11.4 Two-pass overlapped (6.0 tok/s but repetitive output)

Run routing on all layers first, then I/O + compute。6.0 tok/s但output repetitive。

Failure原因：routing decisions depend on updated hidden states。如果layer N+1的input还没被layer N的experts compute出来，layer N+1的routing就是wrong。

**Deep lesson**: Autoregressive generation的sequential dependency是fundamental的。你不能简单parallelize across layers。

### 11.5 Variable K (K=0 middle layers, 10 tok/s but blank)

K=0 for middle 40 layers。10 tok/s但blank output。

这告诉我们**middle layers are critical**——不能skip它们的expert computation。即使input/output layers有experts，middle layers的experts处理high-level reasoning。

K=2或K=3 also collapsed，confirming K=4是hard floor。

## 12. Batch Prefill 优化

对于multi-token prompt processing:
- Process all tokens through transformer
- **Discard intermediate expert results** for all but last token
- `discard_deferred_experts()`: cancel CMD3 expert forward pass for prefill tokens

TTFT reduction: 5.6s → 2.6s for 16-token prompt (53% reduction)

这是一个clever optimization。Intuition：prefill时你只关心final hidden state（用于first generation token），中间tokens的expert outputs会被后续tokens overwrite，所以skip它们的expert I/O + compute。

这个优化对long prompt特别valuable——savings scale linearly with prompt length。

## 13. Session Caching for Multi-turn Conversation

Persist KV cache and linear attention state between requests。Second-turn TTFT = prefill ~14 new tokens instead of full history。

对于10-turn conversation with 2000 cumulative tokens：
- Avoid re-processing 98% of context on each turn
- Massive TTFT improvement for chat applications

这是production deployment的关键feature。Without it，multi-turn chat会越来越慢。

## 14. Human-AI Collaborative Optimization Methodology

Paper提到使用autoresearch protocol (your own work, Andrej!) [11]。24小时continuous collaboration between Daniel Woods and Claude Opus 4.6。

### 14.1 Protocol Details
- 5-minute wall-clock budget per experiment
- Git commit before execution
- Results logged to TSV regardless of outcome
- Benchmark before and after
- Immediate revert on regression

### 14.2 Human's Key Insights

1. **"Treat expert weights like a database"** → pread()-based I/O architecture inspired by database page management
2. **"Trust the OS cache"** → counterintuitive removal of application-level caching
3. **"What about the hardware cache hierarchy"** → vm_stat investigation revealing memory compressor thrashing

这三个insights都是database/systems engineering的wisdom transfer到ML inference场景。

### 14.3 AI's Contribution

- Complete Objective-C/Metal inference engine (~5,000 lines)
- All Metal compute shaders (~1,100 lines)
- 2-bit requantization pipeline
- Systematic exploration of 90+ experiments

### 14.4 42% Experiments Discarded

Most common failure: broken output (throughput improved but computation corrupted)。这underscores一个重要principle：**always verify output quality, not just speed**。

你的autoresearch repo: https://github.com/karpathy/autoresearch

## 15. Future Directions 分析

### 15.1 1-bit and Ternary Experts

Ternary ({-1, 0, +1})或binary quantization可能halve I/O again。挑战：post-training quantization below 2 bits通常需要mixed-precision。

BitNet b1.58（ternary）在training时就low-precision，但post-training到ternary很难。

BitNet: https://arxiv.org/abs/2310.11453
BitNet b1.58: https://arxiv.org/abs/2402.17764

**我的speculation**: 对于MoE experts，可能可以用**importance-aware mixed precision**——frequently activated experts用4-bit，rarely activated的用1-bit。Paper的routing statistics显示expert activation是long-tailed，所以这种策略可能work。

### 15.2 ANE Co-processing

Apple Neural Engine提供~16 TFLOPS FP16，currently unused。

Challenge: ANE的programming modelrestricted（requires CoreML/MIL format），不支持dynamic control flow (top-K routing, variable expert selection)。

可能的workaround：把**fixed-shape** operations（如attention projections）offload到ANE，把dynamic operations留在GPU。

maderix/ANE project: https://github.com/maderix/ANE

### 15.3 Speculative Decoding

用Qwen3.5-35B-A3B（fits in DRAM at 37 tok/s）作为draft model。Initial experiment：33% acceptance rate at K=4 draft tokens，net 4.5× slowdown。

Failure原因：draft model的routing decisions和397B model的diverge significantly。

**Better approach**: Self-speculative decoding using 397B with K=1（very fast, very low quality）as draft。这个idea很clever——同一个model的K=1和K=4应该有更高acceptance rate，因为routing logic相同。

Speculative decoding paper: https://arxiv.org/abs/2211.17192

### 15.4 Multi-device Inference via Thunderbolt

Two M3 Max laptops via Thunderbolt，each handle 30 layers。Apple Fabric extends across Thunderbolt-connected devices。

Challenge: inter-device transfer latency and bandwidth characteristics未characterized。

**我的concern**: Thunderbolt 4的bandwidth是40 Gbps ≈ 5 GB/s，远低于single-device的400 GB/s unified memory bandwidth。30层的boundary处需要transfer hidden state (4096 × bfloat16 = 8 KB per token)，这个transfer latency可能negate multi-device的I/O benefit。

Thunderbolt architecture: https://developer.apple.com/documentation/devicedriverkit/thunderbolt

### 15.5 Applicability to DeepSeek-V3

DeepSeek-V3: 671B total, 37B active。At 2-bit expert quantization, expert weights ~200 GB。

这在Flash-MoE的framework范围内，只需要modest SSD capacity。

DeepSeek-V3: https://arxiv.org/abs/2412.19437

## 16. Broader Implications 和 我的 Speculations

### 16.1 对 Local LLM 生态的影响

Flash-MoE证明MoE + SSD streaming是local inference的viable path。这和llama.cpp的"dense model in DRAM"哲学是complementary的。

未来可能看到**hybrid engines**：
- Dense layers (attention, shared expert) in DRAM
- Sparse expert weights streamed from SSD
- OS manages all caching

### 16.2 对 Model Architecture 的影响

如果SSD streaming成为主流，model architecture可能会evolve to exploit它：
- **More experts, smaller each** (higher sparsity ratio)
- **Expert specialization** (某些experts for certain domains, enabling smart prefetching)
- **Hierarchical MoE** (coarse routing → expert groups → fine routing)

### 16.3 对 Hardware Design 的影响

Paper的scaling projection显示SSD bandwidth是bottleneck。这可能drive：
- **Faster NVMe** on consumer hardware
- **Larger on-chip caches** for expert weights (类似GPU的SRAM)
- **Hardware compression/decompression** for expert weights (类似texture compression in GPUs)

### 16.4 对 Quantization Research 的影响

2-bit requantization的success（在already-4-bit weights上）suggests一个new research direction：**cascaded quantization**——multi-stage quantization where each stage exploits the reduced entropy of the previous stage。

这可能extend to：4-bit → 2-bit → 1-bit → ternary，每一步都exploit前一步的compressed dynamic range。

### 16.5 Memory Hierarchy 类比

Flash-MoE实际上建立了一个新的memory hierarchy for LLM inference：

```
Level 0: GPU registers (fastest, smallest)
Level 1: GPU threadgroup shared memory
Level 2: Unified DRAM (400 GB/s, 48 GB)
Level 3: NVMe SSD (17.5 GB/s, 1 TB)  ← NEW
Level 4: Network/cloud storage
```

MoE的sparsity让SSD作为"slow L3 cache"变得viable。这是对传统memory hierarchy的extension。

## 17. Critique 和 Open Questions

### 17.1 Quality Evaluation 不足

Paper的quality assessment是qualitative（"Excellent", "Same as K=10"）。没有perplexity numbers或benchmark scores（MMLU, HumanEval等）。

对于production deployment，需要quantitative quality metrics。特别是2-bit quantization对long-context reasoning的影响。

### 17.2 Single Prompt Benchmarking

所有测量用同一个prompt ("Explain the concept of probability to a five year old")。不同prompt的routing patterns可能不同，影响cache hit rates和I/O patterns。

### 17.3 K=4 Quality Cliff 的更深入分析

为什么是K=4而不是K=3或K=5？这需要分析expert routing的learned structure。可能是某个critical functional component需要4个experts represent，但paper没有investigate这个。

### 17.4 2-bit Quantization 的 Generalizability

Paper只测试了Qwen3.5-397B。2-bit requantization在其他MoE models（Mixtral, DeepSeek-V3）上是否同样effective？这需要更多experiments。

### 17.5 Memory Compressor Finding 的 Generalizability

"Trust the OS"的发现是Apple Silicon specific。在Linux/Windows上可能不适用。Paper没有discuss这个limitation。

### 17.6 Lack of Comparison with vLLM/TGI

Paper比较了FlexGen和LLM in a Flash，但没有比较modern serving systems如vLLM或TGI。这些系统在data-center GPUs上可能有更好的offloading策略。

vLLM: https://arxiv.org/abs/2309.06180
TGI: https://github.com/huggingface/text-generation-inference

## 18. 技术细节补充

### 18.1 Expert Binary Layout (Table 10)

每个2-bit expert占3,932,160 bytes，layout：

```
Offset 0:        gate weights [1024, 256] uint32    (1,048,576 bytes)
Offset 1,048,576:  gate scales [1024, 64] bf16       (131,072 bytes)
Offset 1,179,648:  gate biases [1024, 64] bf16       (131,072 bytes)
Offset 1,310,720:  up weights [1024, 256] uint32     (1,048,576 bytes)
Offset 2,359,296:  up scales [1024, 64] bf16         (131,072 bytes)
Offset 2,490,368:  up biases [1024, 64] bf16         (131,072 bytes)
Offset 2,621,440:  down weights [4096, 64] uint32    (1,048,576 bytes)
Offset 3,670,016:  down scales [4096, 16] bf16       (131,072 bytes)
Offset 3,801,088:  down biases [4096, 16] bf16       (131,072 bytes)
Total: 3,932,160 bytes
```

注意down weights的shape是[4096, 64]而非[4096, 256]。这是因为down projection是`[intermediate_dim, hidden_dim] = [1024, 4096]`，而group size是64，所以scales/biases的shape是[4096, 1024/64] = [4096, 16]。

等等，这里有些confusion。让我重新算：
- gate: [1024, 256] uint32。1024 × 256 = 262,144 values。每个uint32 pack 16 values (2-bit)，所以实际values = 262,144 × 16 = 4,194,304。但expert的gate projection应该是[hidden_dim, intermediate_dim] = [4096, 1024] = 4,194,304 values。✓
- down: [4096, 64] uint32。4096 × 64 = 262,144 values。× 16 = 4,194,304 values。Down projection是[intermediate_dim, hidden_dim] = [1024, 4096] = 4,194,304 values。✓

这个layout的设计目的是让**sequential byte offset = expert_index × EXPERT_SIZE**，实现O(1) random access by expert index。

### 18.2 Qwen3.5-397B-A17B Architecture (Table 9)

```
Total parameters: 397B
Active parameters per token: 17B (default K=10), ~6.8B (K=4)
Hidden dimension: 4096
Layers: 60
  - Linear attention (GatedDeltaNet): 45
  - Full attention (RoPE): 15 (every 4th layer)
Attention heads: 32
KV heads (GQA): 2
Head dimension: 256
Experts per layer: 512
Default active experts (K): 10
Expert intermediate dim: 1024
Shared expert intermediate dim: 1024
Vocabulary size: 248,320
Expert activation: SwiGLU
RoPE θ: 10^7
Partial rotary fraction: 0.25
Conv kernel (linear attn): 4
```

GQA with 2 KV heads和32 attention heads意味着GroupGQA ratio = 32/2 = 16。这大幅减少KV cache size。

Linear attention的conv kernel = 4是一个short 1D convolution在input上，这是GatedDeltaNet的标准设计。

### 18.3 DRAM Budget (Table 5)

```
Non-expert weights (mmap'd):                5.5 GB
Metal command buffers + scratch:            0.3 GB
Expert I/O buffers (K × expert_size):       0.06 GB (4 × 3.93 MB)
KV cache (per full-attn layer):             0.1 GB
Linear attention state (per layer):         0.5 GB
Total:                                      ~6.5 GB (13.5% of 48 GB)
```

Only 13.5% DRAM occupancy。剩余41.5 GB给OS和其他apps。**Zero risk of OOM** because memory usage is entirely static after initialization。

Linear attention state 0.5 GB for 60 layers = ~8.5 MB per layer。这比full attention的KV cache小得多——full attention需要cache所有历史tokens的K,V，而linear attention只需固定大小的state matrix。

### 18.4 Scaling Projections (Table 8)

```
M3 Max (current):    17.5 GB/s SSD → 5.74 tok/s
M4 Max (2025):       ~25 GB/s SSD → ~8 tok/s
M5/M6 (projected):   ~35 GB/s SSD → ~11 tok/s
PCIe 5.0 NVMe:       ~14 GB/s → ~4.4 tok/s
```

Apple的SSD bandwidth每年~20% improvement。PCIe 5.0 NVMe的14 GB/s actually slower than M3 Max的integrated SSD——这highlight了Apple Fabric的优势（integrated SSD controller，no PCIe overhead）。

## 19. 和 Related Work 的深度对比

### 19.1 vs LLM in a Flash [1]

| Aspect | LLM in a Flash | Flash-MoE |
|--------|----------------|-----------|
| Hardware | iPhone | M3 Max laptop |
| Model/DRAM ratio | 2× | 4.4× |
| Sparsity type | ReLU activation | MoE expert routing |
| Sparsity % | 3-10% | 0.78-1.95% |
| Compute | CPU | Metal GPU |
| SSD bandwidth | ~5 GB/s | 17.5 GB/s |
| Caching | Windowing + row-column bundling | Trust OS page cache |

LLM in a Flash的"windowing" technique是keep recently activated neurons in DRAM。Flash-MoE发现这个approach在Apple Silicon上不如trust OS page cache——这是一个重要的methodological difference。

LLM in a Flash: https://arxiv.org/abs/2312.11514

### 19.2 vs FlexGen [7]

FlexGen是CPU-GPU-SSD offloading for batch inference on data-center GPUs。它targets throughput over latency。

Flash-MoE targets **interactive latency** on consumer hardware。这是不同的design point：
- FlexGen: batch size > 1, optimize aggregate throughput
- Flash-MoE: batch size = 1, optimize per-token latency

FlexGen: https://arxiv.org/abs/2303.06865

### 19.3 vs DeepSpeed ZeRO-Infinity [10]

ZeRO-Infinity是for training, not inference。它offloads optimizer states, gradients, and parameters to NVMe。

Flash-MoE的innovation是exploiting MoE sparsity for inference——ZeRO-Infinity不exploit sparsity，它offloads all parameters uniformly。

ZeRO-Infinity: https://arxiv.org/abs/2104.07857

### 19.4 vs llama.cpp [9]

llama.cpp是CPU-based inference with GGML quantization。它假设model fits in DRAM。

Flash-MoE的contribution是showing that for MoE models，SSD streaming is viable even when model >> DRAM。这是a different design point that llama.cpp doesn't address.

llama.cpp: https://github.com/ggerganov/llama.cpp

### 19.5 vs MLX [8]

Flash-MoE initially used MLX but switched to custom Objective-C/Metal engine。原因：
- Python overhead
- Lack of fine-grained control over command buffer scheduling
- MLX's lazy evaluation model doesn't fit the streaming pipeline

MLX: https://github.com/ml-explore/mlx

## 20. 我的 Final Thoughts

### 20.1 这篇 Paper 的真正 Contribution

Beyond the specific optimizations, this paper makes several broader contributions:

1. **Demonstrates MoE sparsity enables SSD streaming** — 这opens up一个new design space for local LLM inference
2. **"Trust the OS" principle** — counterintuitive但important finding for Apple Silicon
3. **2-bit requantization viability** — extends the quantization frontier
4. **Human-AI collaborative optimization methodology** — 24 hours to publication-quality result

### 20.2 对 Andrej 你的 Autoresearch 的 Validation

Paper使用你的autoresearch protocol [11]并explicitly highlight it as "a contribution in its own right"。这是一个strong validation of the methodology——human + AI collaboration在24小时内produce了publication-quality systems optimization result。

Key success factors:
- 5-minute experiment budget forces rapid iteration
- Git commit before execution enables instant revert
- Log all results (including failures) enables pattern recognition
- Benchmark-before-and-after discipline catches silent regressions

### 20.3 Open Research Questions

1. **Expert routing structure**: 为什么K=4是cliff？能否redesign routing to be more cache-friendly?
2. **Cascaded quantization**: 4-bit → 2-bit → 1-bit的viability on other models?
3. **Cross-platform "trust the OS"**: Linux/Windows上的equivalent finding?
4. **Expert importance scoring**: 能否identify "critical" experts and keep them in DRAM?
5. **Adaptive K**: 不同layers/tokens使用不同K值（paper的variable K experiment failed，但更sophisticated的approach可能work）

### 20.4 最深层的 Intuition

Flash-MoE的success源于一个deep insight：**MoE架构的sparsity不是compute optimization，而是memory hierarchy optimization**。

传统理解：MoE让你train bigger models with same compute。
Flash-MoE的理解：MoE让你**access fewer weights per token**，这正好match SSD streaming的bandwidth constraints。

这个reframing可能inspire未来的model architecture设计——design models not just for compute efficiency, but for **memory access efficiency** in a world where models exceed DRAM capacity.

---

## References

- [1] LLM in a Flash: https://arxiv.org/abs/2312.11514
- [2] Qwen3 Technical Report: https://qwenlm.github.io/blog/qwen3/
- [3] Mixtral of Experts: https://arxiv.org/abs/2401.04088
- [4] GPTQ: https://arxiv.org/abs/2210.17323
- [5] AWQ: https://arxiv.org/abs/2306.00978
- [6] Dettmers 4-bit Scaling Laws: https://arxiv.org/abs/2212.09753
- [7] FlexGen: https://arxiv.org/abs/2303.06865
- [8] MLX: https://github.com/ml-explore/mlx
- [9] llama.cpp: https://github.com/ggerganov/llama.cpp
- [10] DeepSpeed ZeRO-Infinity: https://arxiv.org/abs/2104.07857
- [11] Karpathy autoresearch: https://github.com/karpathy/autoresearch
- BitNet: https://arxiv.org/abs/2310.11453
- BitNet b1.58: https://arxiv.org/abs/2402.17764
- GatedDeltaNet: https://arxiv.org/abs/2412.06464
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- Lottery Ticket Hypothesis: https://arxiv.org/abs/1803.03635
- vLLM/PagedAttention: https://arxiv.org/abs/2309.06180
- Speculative Decoding: https://arxiv.org/abs/2211.17192
- Linear Transformers: https://arxiv.org/abs/2006.16236
- DeltaNet: https://arxiv.org/abs/2410.01343
- Apple Silicon Memory Architecture: https://developer.apple.com/documentation/apple-silicon
- Metal Programming Guide: https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/
- PostgreSQL Tuning: https://www.postgresql.org/docs/current/runtime-config-resource.html
- macOS Memory Management: https://developer.apple.com/library/archive/documentation/Performance/Conceptual/ManagingMemory/ManagingMemory.pdf
