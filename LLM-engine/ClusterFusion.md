---
source_pdf: ClusterFusion.pdf
paper_sha256: 865c3654389e19ca348ce89a82444703c874797966954a62197869d2e1c29009
processed_at: '2026-08-03T16:08:53-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ClusterFusion 人话版

## 一句话概括

现在的 LLM inference 框架把 Transformer 的计算拆成好几个小 kernel，每个 kernel 算完要把中间结果存到 GPU 的"硬盘"（HBM 显存）里，下一个 kernel 再读回来。这种来回搬运特别慢。这篇 paper 利用 NVIDIA Hopper GPU 新加的一个硬件特性——**让同一组 SM 之间能直接共享 memory**——把原来拆开的三个 kernel（QKV Projection + Attention + Output Projection）缝成一个，中间数据完全在 on-chip 流转，不用再去显存里绕一圈。

## 问题出在哪

LLM decoding 的时候，每生成一个 token，整个 Transformer block 都要跑一遍。但是每算一个 token 的计算量其实很小，大头时间全花在**搬数据**上。

你可以想象成：GPU 里面有 100 多个工人（SM），每个工人有自己的工作台（shared memory），但工人之间没法直接递东西，要交换零件只能把东西放到仓库（HBM 显存）里，另一个人再去仓库取。仓库虽然大，但来回跑一趟很慢（470 cycles 延迟）。

现有框架比如 SGLang、vLLM，它们处理一个 attention head 的流程大概是这样：

```
kernel 1 (QKV Projection) → 结果写到 HBM
kernel 2 (Attention) → 读 HBM，算完再写 HBM  
kernel 3 (rescale) → 读 HBM，写 HBM
kernel 4 (Output Projection) → 读 HBM，写最终结果
```

每个 kernel 之间都有一次"写出去再读回来"的 round-trip。而且每个 kernel 启动本身还有 overhead（即使有 CUDA Graph 也消不掉）。

## Hopper 给了什么新玩具

NVIDIA H100 这代 GPU 加了个叫 **Thread Block Cluster** 的东西。你可以把几个 SM（最多 16 个）绑成一个"小组"，组内的 SM 之间有一条高速直连网络（叫 DSMEM），可以直接读写彼此的 shared memory。

paper 里测了这条路的延迟：

| 路径 | 延迟 |
|------|------|
| SM → HBM → SM | > 470 cycles |
| SM → DSMEM → SM（cluster size=2）| 190 cycles |

快了 2.5 倍。这就好比你跟同事之间终于可以直接递东西了，不用再绕仓库。

但问题来了：NVIDIA 只给了最底层的 PTX 指令，相当于只给了你"往隔壁发一个包裹"这种原始操作，没有更高级的"把所有人的数据加起来"或者"把所有人的数据收集到一起"这种集体通信抽象。开发者要自己手写这些 pattern，门槛很高。

## ClusterFusion 干了什么

### 1. 造了两个"集体通信"积木

参考 MPI 里常用的 collective operation，作者在 DSMEM 上实现了两个 primitive：

**ClusterReduce**：把所有 block 的数据 reduce（求和或取最大值）。
- 用蝴蝶形通信，log₂(N) 轮，每轮跟不同 partner 交换
- 每轮数据量不变
- 用于 Attention 输出的聚合、softmax 统计量的 reduce

**ClusterGather**：把每个 block 的本地数据收集到所有 block。
- 跟 ClusterReduce 类似的蝴蝶形，但每轮数据量翻倍
- 用于 QKV Projection 之后把各 block 算的 head 维度碎片拼成完整的 head

这两个 primitive 就像搭积木的基础件，后面所有融合设计都靠它们。

### 2. 用积木把三个 kernel 缝成一个

核心设计：**一个 attention head 对应一个 cluster**。

cluster 内的 N 个 block 分工：
- QKV Projection：每个 block 算 head 维度的一段
- Attention：每个 block 算 KV cache sequence 的一段（FlashDecoding 风格）
- Output Projection：每个 block 算 output 维度的一段

流程变成：

```
所有 block 同时进 QKV Projection
    ↓ 各自算出 Q/K/V 的碎片
    ↓ ClusterGather（通过 DSMEM 拼成完整 Q/K/V）
所有 block 同时做 Attention（FlashDecoding 风格）
    ↓ 各自算出 partial softmax 统计量和 partial attention 输出
    ↓ ClusterReduce（sum 和 max，两次）
    ↓ online softmax rescale
    ↓ ClusterReduce（把 partial attention 聚合）
所有 block 同时做 Output Projection
    ↓ atomicAdd 写到 HBM（只有这一步碰显存）
```

整个过程只 launch 一个 kernel，中间数据全在 cluster 内部的 shared memory 和 register 里流转，零次 HBM round-trip。

## 为什么能加速

paper 实测在 H100 上，对比 SGLang/vLLM/TensorRT-LLM 这些 SOTA 框架：

- **端到端延迟**：平均 1.4~1.5 倍加速（batch=1）
- **核心模块（QKV+Attention+Output）**：1.6~3.5 倍加速
- **kernel launch overhead**：减少近一个数量级

加速来源就两个：
1. **少了 HBM 来回**：中间 Q/K/V/Attention 都不用写显存了
2. **少了 kernel launch**：原来 4 个 kernel 变 1 个

## 为什么不干脆把 cluster 搞大点

paper 测了不同 cluster size 的效果，发现 cluster size 4 是甜点，8 和 16 反而变差。原因有仨：

1. **DSMEM 带宽随 cluster size 下降**：cluster size 16 时带宽降到 2.90 TB/s，跟 HBM 的 2.96 TB/s 差不多了，失去优势
2. **Active SM 数量减少**：cluster 太大导致同时能 launch 的 cluster 数变少，整体并行度下降
3. **NoC 争用**：crossbar 架构，节点多了争用严重

所以设计哲学是"小而多"——每个 cluster 只管一个 attention head 的内部协作，head 之间靠 cluster 级别并行。

## 几个有意思的细节

### DeepSeek MLA 也能用

DeepSeek V2/V3 的 MLA 比 MHA 多了 Up/Down Projection，paper 附录给了 fused MLA dataflow。需要 3 个 ClusterGather + 3 个 ClusterReduce，比 MHA 还多。但实测加速比 Llama2 略低——因为 MLA 本身已经针对硬件优化过，进一步优化的空间被压缩了。

### SplitHead 数据流是个反例

paper 还设计了一个把中间数据放 register（比 shared memory 更快）的 dataflow，但实测反而更慢。原因是它需要 reduce 的数据量跟 sequence length 成正比，而 DSMEM traffic 一爆炸就崩了。这说明光看"register 比 shared memory 快"不够，还要看 communication pattern 的 traffic 量级。

### Batch 越大加速越小

batch=16 时加速降到 1.1~1.3 倍。因为 batch 大了之后计算变密集，从 memory-bound 往 compute-bound 走，fusion 的边际收益递减。这跟 FlashAttention 的经验完全一致。

## 对比 FlashAttention

FlashAttention 把 attention 内部的 QK^T、softmax、AV 融合了，但停在 attention 边界——输出还是要写回 HBM。

ClusterFusion 的融合**跨越了 attention 边界**，把前后的 Projection 也拉进来。这是 scope 上的扩展。

## 这篇 paper 的启示

对我来说最大的启发是：**operator fusion 的边界由数据通信代价决定**。如果 block 间通信要 470 cycles + 走 HBM，fusion 就停在 kernel 边界；如果只要 190 cycles 且全 on-chip，fusion 就能跨 kernel。硬件特性直接定义了软件 fusion 的可达空间。

Hopper 给了 DSMEM 这条"高速公路"，但 NVIDIA 没给高层抽象，大家不知道怎么用。ClusterFusion 做的事就是补上这层抽象，然后展示"有了这层抽象，我们能融合到什么程度"。

## 一些联想

**Blackwell 会怎么演进**：如果下一代 GPU 把 cluster scope 从 16 扩到 64 或 256，fusion 边界能推到 FFN 甚至整个 Transformer block。这是很值得关注的硬件方向。

**跟 warp specialization 正交**：FlashAttention-3 用 warp specialization + TMA 隐藏 memory latency，ClusterFusion 用 DSMEM 做跨 block 通信，两者可以叠加。paper 没做这个叠加，是个 obvious next step。

**编译器集成**：paper 的 traffic 公式是解析的，可以直接嵌到 compiler 里做 dataflow 选择。未来 LLM compiler 应该有 DSMEM-aware 的 cost model，自动决定 fusion scope 和 cluster size。

**MoE 场景**：MoE 的 expert routing 有跨 block 数据依赖，expert 输出的聚合可能也能用 ClusterReduce。这是一个 paper 没覆盖但很自然的扩展。

参考链接：
- ClusterFusion GitHub: https://github.com/xinhao-luo/ClusterFusion
- NVIDIA Hopper Architecture: https://resources.nvidia.com/en-us-hopper-architecture
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashDecoding: https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- DeepSeek V2: https://arxiv.org/abs/2405.04434
- SGLang: https://github.com/sgl-project/sglang
- vLLM: https://github.com/vllm-project/vllm
- FlashInfer: https://github.com/flashinfer-ai/flashinfer

---

# ClusterFusion: 用 Hopper DSMEM 扩展 LLM Decoding 的 Operator Fusion 边界

## 1. 直觉建立：为什么需要这篇 paper

LLM 推理的核心痛点在 decoding 阶段。paper 里 Fig.2 给出实测数据：在 256 tokens 序列上，decoding 占总延迟 **95% 以上**。原因是 decoding 本质上是 memory-bound 的 auto-regressive 过程：每生成一个 token，都要把整个 Transformer block 跑一遍，而每个 token 的计算量很小，导致算术强度极低，性能完全被 memory traffic 主宰。

更糟的是，现有框架（SGLang、vLLM、TensorRT-LLM 等）把 Transformer block 拆成多个独立 kernel：QKV Projection、Attention、Output Projection、FFN……这些 kernel 之间的中间结果（Q、K、V、Attention 输出）必须写到 global memory，再被下一个 kernel 读回来。这带来两类开销：
1. **Off-chip memory round-trip**：HBM 带宽 ~3 TB/s，延迟 ~470 cycles
2. **Kernel launch overhead**：即使有 CUDA Graph，launch 本身仍有开销

传统 CUDA 编程模型里，thread block 是一个"孤岛"——block 之间要交换数据只能通过 global memory。这就像多个进程要协作，但只能通过磁盘传文件。NVIDIA Hopper 给了一条"高速公路"：**Thread Block Cluster** + **Distributed Shared Memory (DSMEM)**，让同一 cluster 内的 SM 通过 NoC 直接读写彼此的 shared memory，延迟只有 ~190 cycles。但是 NVIDIA 只暴露了低层 PTX 指令（如 `cp.async.bulk.shared::cluster`），开发者得手写 peer-to-peer 数据搬运，缺乏 MPI 那种 `Reduce`/`Gather`/`Allreduce` 高层抽象。

ClusterFusion 的核心贡献就是：**在 DSMEM 上构建 cluster 级的 collective primitive（ClusterReduce / ClusterGather），然后用这些 primitive 把 QKV Projection + Attention + Output Projection 融合成一个 kernel**，中间数据完全在 on-chip 流转。

参考链接：
- NVIDIA Hopper 白皮书: https://resources.nvidia.com/en-us-hopper-architecture
- Hopper DSMEM PTX 文档: https://docs.nvidia.com/cuda/parallel-thread-execution/
- FlashDecoding 博客: https://crfm.stanford.edu/2023/10/12/flashdecoding.html

---

## 2. Hopper DSMEM 的硬件特性与权衡

paper Sec.2.3 做了细致的 microbenchmark（Fig.5），是理解整篇 paper 设计选择的钥匙。

**SM-to-SM 延迟**（左图）：
- cluster size = 2：190 cycles
- cluster size = 4：约 200 cycles
- cluster size = 16：约 280 cycles
- 对比 global memory：> 470 cycles

**SM-to-SM 带宽**（中图）：
- 随 cluster size 增加而下降
- cluster size = 16 时降到 2.90 TB/s，几乎和 global memory 的 2.96 TB/s 持平
- 原因是 NoC 是 crossbar 架构，节点越多争用越严重

**Active SM 数量**（右图）：
- H100 有 132 个 SM，但 cluster 大小限制能同时 launch 的 cluster 数
- cluster size = 16 时 active SM 数显著减少，整体并行度被压缩

这个 trade-off 直接决定了后面 cluster size 的调参结论：**不是越大越好，sweet spot 在 2~4 之间**。

直觉：DSMEM 像是一个"小型局域网"，节点少的时候低延迟、高带宽，节点多了就退化成跟访问 HBM 差不多的水平。所以 ClusterFusion 的设计哲学是"小 cluster、多 cluster"，每个 attention head 占一个 cluster，cluster 内部做紧密协作，cluster 之间做 head 级并行。

---

## 3. ClusterReduce 与 ClusterGather 算法详解

两个 primitive 都基于 **binary tree（蝶形）通信模式**，log₂(N) 轮，每轮 stride 翻倍。这是经典 MPI Reduce / Allgather 的 intra-cluster 版本。

### 3.1 ClusterReduce（Alg.1）

输入：N = 2^k 个 thread block，每个 block rank b ∈ [0, N-1]，本地 buffer D_b，可结合算子 ⊕（sum 或 max）。

```
stride = 1
while stride < N:
    send_to    = (b + stride) mod N
    recv_from  = (b - stride + N) mod N
    Send D_b  -> B_send_to   (via DSMEM)
    Recv      <- B_b          (from recv_from)
    Wait
    D_b = D_b ⊕ B_b
    stride *= 2
return D_b
```

变量含义：
- `D_b`：block b 持有的本地数据，最终会被 reduce 成"全 cluster 的总和/最大值"
- `B_b`：临时 buffer，用于接收远端数据，避免读写冲突
- `stride`：每轮通信 partner 的 rank 偏移量，按 2 的幂指数增长
- `⊕`：reduction 算子，比如 sum（用于 Attention 输出聚合）或 max（用于 online softmax 的 m 统计量）

直觉：第 1 轮每个 block 跟"邻居"交换数据并 reduce；第 2 轮跟"隔一个的邻居"交换；第 3 轮跟"隔三个的邻居"交换……log₂(N) 轮后所有 block 都拥有全 cluster 的 reduce 结果。这跟 NCCL 的 ring allreduce 在精神上一致，但适配了 cluster 内的 fully-meshed NoC 拓扑。

### 3.2 ClusterGather（Alg.2）

跟 ClusterReduce 的差别在于 **message size 每轮翻倍**：

```
stride = 1
while stride < N:
    send_to    = (b + stride) mod N
    recv_from  = (b - stride + N) mod N
    Send D_b[0 : size*stride]       -> D_send_to[stride*size : 2*stride*size]
    Recv D_recv_from[0:size*stride] -> D_b[stride*size : 2*stride*size]
    stride *= 2
return D_b   # 此时 D_b 包含所有 N 个 block 的数据
```

变量含义：
- `D_b` 初始只有本地一段数据 `D_b[0:size]`
- 每轮 stride 翻倍后，buffer 的"已填充段"也翻倍
- 最终 `D_b` 是 `[block_0 数据, block_1 数据, ..., block_{N-1} 数据]` 的拼接

这个 primitive 用于 QKV Projection 之后：每个 block 只算出 head 维度的一个 tile（`h` 维），需要把整个 head 维 `H` 拼起来才能进 Attention。

### 3.3 Traffic 公式推导

paper 给的两个公式（公式 3）：

$$
\text{Traffic}_{\text{Reduce}}(\text{size}, N) = \text{size} \times \log_2 N \times N
$$

$$
\text{Traffic}_{\text{Gather}}(\text{size}, N) = \text{size} \times \left(2^{\log_2 \frac{N}{2} + 1} - 1\right) \times N
$$

变量解释：
- `size`：每个 block 的本地 buffer 大小（字节或元素数）
- `N`：cluster size（2/4/8/16）
- 第一个公式：N 个 block × log₂(N) 轮 × 每轮 size 大小 → 因为每轮 size 不变
- 第二个公式化简：`2^(log₂(N/2)+1) - 1 = 2·(N/2) - 1 = N - 1`，所以等价于 `size × (N-1) × N`。这就是等比数列 `1+2+4+...+N/2 = N-1` 的求和，对应每轮 size 翻倍

整个 fused dataflow 的 DSMEM traffic（公式 4）：

$$
\text{Traffic}_{\text{Total}} = \text{Traffic}_{\text{Reduce}}(3h, N) + \text{Traffic}_{\text{Gather}}(H, N)
$$

- `3h`：QKV 三段的 head 维度 tile 大小，需要 reduce
- `H`：完整 head 维度，需要 gather
- 这里没算 softmax statistics 的 traffic（只有两个 float，可忽略）

直觉：**traffic 主要被 head 维度（而不是 sequence length）决定**，这是后面 SplitToken vs SplitHead 数据流对比的关键。

---

## 4. Cluster-Centric Dataflow：把三个 kernel 融成一个

### 4.1 设计原则

paper Sec.3.2 的核心 insight：
- **数据依赖的维度**（head 维度、KV sequence 维度）放在 cluster 内部，用 ClusterReduce/ClusterGather 在 on-chip 解决
- **数据独立的维度**（不同的 attention head）分布在 cluster 之间

每个 attention head 对应一个 cluster，cluster 内 N 个 block 协作完成：
- QKV Projection：沿 head 维度切分，每个 block 算 `h/N` 维
- Attention：沿 KV cache sequence 切分，每个 block 算 `s/N` 段
- Output Projection：沿 output 维度切分，每个 block 算 `d/N` 维

### 4.2 算法 3（Fused QKV+Attention+Output）逐步解析

```
1: 分配 shared memory: Q_b, K_b, V_b ∈ R^{B×h}, S_sum, S_max
2: Q_b, K_b, V_b = H_b × W_b^{QKV}           # 本地 GEMM，分块
3: (Q_b, K_b, V_b) = ClusterGather(...)        # 拼出完整 H 维
4: FlashDecoding 风格 Attention:
    S_b = exp(Q_b × (K_b^cache, K_b)^T)
    本地算 S_sum, S_max，max 存到 Reg_max
    A_b = S_b × (V_b^cache, V_b)               # 复用 Q_b 的 shared mem
5: S_sum = ClusterReduce(S_sum, sum)
   S_max = ClusterReduce(S_max, max)
6: A_b = A_b × exp(Reg_max - S_max) / S_sum    # online softmax rescale
7: A_b = ClusterReduce(A_b, sum)               # 聚合 partial attention
8: O_b = A_b × W_b^O  → atomicAdd 到 global
```

变量含义：
- `B`：batch size
- `D`：input hidden dimension（Llama2-7B 是 4096）
- `H`：total head dimension（Llama2 是 128 × 32 heads = 4096）
- `h, s, d`：每个 thread block 切到的 head/sequence/output 大小
- `H_b ∈ R^{B×D}`：每个 block 都读完整 input hidden（因为 input 是 batch × D，D 很大但只有一行要算）
- `W_b^{QKV} ∈ R^{D×3h}`：QKV 权重的本地分片
- `K_b^cache, V_b^cache ∈ R^{s×H}`：KV cache 的本地分段
- `Reg_max`：保存在 register 里，避免被 ClusterReduce 覆盖
- `atomicAdd`：Output Projection 跨 head 的输出要累加到同一个 `O` 向量

关键技巧：
- **Shared memory 复用**：`A_b` 直接复用 `Q_b` 的 shared memory 空间，因为 Q 用完后不再需要
- **Online softmax**：FlashAttention/FlashDecoding 的经典做法，先把 partial max 和 sum reduce 出来，再 rescale partial attention 输出，最后再 reduce 一次

### 4.3 整体架构图（Fig.7）解析

Fig.7 展示了 cluster-centric 的数据流，对比 Fig.3 的传统 dataflow：

传统 dataflow（Fig.3）：
```
QKVProj kernel → 写 HBM → Attention kernel (多 block) → 写 HBM → rescale kernel → 写 HBM → OutputProj kernel
```
有 4 个 kernel 边界，3 次 HBM round-trip。

ClusterFusion dataflow（Fig.7）：
```
单 kernel 内部:
  [QKVProj local] → ClusterGather (DSMEM) → [Attention local] → ClusterReduce×2 (DSMEM) → [OutputProj local] → atomicAdd HBM
```
只有 1 个 kernel，0 次 HBM 中间 round-trip（只有最终 output 一次 atomicAdd）。

---

## 5. DeepSeek MLA 的特殊处理（附录 B.1）

DeepSeek V2/V3 的 MLA（Multi-head Latent Attention）跟 MHA 不一样，引入了 Up/Down Projection 来压缩 KV cache。paper 附录给了 fused MLA dataflow（Alg.4）。

**Weight absorption 优化后的 MLA 计算**（公式 3-5）：

$$
Q = \text{Hidden} \times W_Q \times W_{\text{Up}}, \quad K = \text{Hidden} \times W_K, \quad V = K[:\text{kv\_lora\_rank}]
$$

$$
Z = \text{Concat}\left(\text{Softmax}\left(\frac{Q_i K^T}{\sqrt{d_k}}\right) V, \ldots\right)
$$

$$
\text{Output} = Z \times W_{\text{Down}}
$$

变量：
- `kv_lora_rank`：DeepSeek-V2-Lite 是 512，远大于 Llama2 head dim 128
- `W_Up`：把压缩的 KV 升维回多头
- `W_Down`：把 attention 输出降维
- MLA 用 MQA：所有 Q head 共享一份 KV

Fused MLA dataflow 需要 **3 个 ClusterGather + 3 个 ClusterReduce**（Q、KV、Attention 输出都要聚合），traffic 公式：

$$
\text{Traffic}_{\text{Gather}} = \text{Traffic}_{\text{Gather}}(h, N) + 2 \times \text{Traffic}_{\text{Gather}}(l, N)
$$

$$
\text{Traffic}_{\text{Reduce}} = \text{Traffic}_{\text{Reduce}}(l, N) + \text{Traffic}_{\text{Reduce}}(H, N)
$$

变量：
- `h`：head 维度 tile
- `l`：kv_lora_rank tile（512 / N）
- `H`：total head dim

MLA 因为多了 Up/Down 两个 projection，fusion 空间比 MHA 更大，但 paper 实测显示 MLA 的 speedup（1.34×~2.39×）反而比 Llama2 略低，原因 paper 解释为"MLA 本身已经针对硬件做了优化，进一步 fusion 的空间相对受限"。这个观察挺有意思——架构越专门化，留给系统优化的余量越小。

---

## 6. SplitToken vs SplitHead 数据流对比（附录 B.2）

paper 还设计了 SplitHead dataflow（Alg.5），跟主 paper 的 SplitToken dataflow 形成对比。

**SplitToken**（主 paper）：Attention 沿 KV cache sequence 维度切分
- 中间数据 Q/K/V 放 shared memory
- DSMEM traffic = Traffic_Reduce(H, N) + Traffic_Gather(3h, N)
- 主要依赖 H 和 h，都很小

**SplitHead**（附录）：Attention 沿 head 维度切分
- 中间数据 Q/K/V 可以放 register（更快）
- 但 Q×K^T 的结果 shape 是 `Sequence_Length × Batch_Size`，需要 reduce
- DSMEM traffic = Traffic_Reduce(S, N) + Traffic_Reduce(D, N)
- 主要依赖 S（sequence length）和 D（hidden dim），S 可能到 16K，巨大

Fig.20 实测：短序列时两者差不多；序列变长后 SplitHead 显著变慢，因为 DSMEM traffic 爆炸。这是 paper 一个很好的设计反例——**光看"register 比 shared memory 快"是不够的，还要看 communication pattern 的 traffic 量级**。

直觉：register 适合存"小而频繁访问"的数据，shared memory 适合存"需要在 block 间共享"的数据。SplitHead 把 attention 中间结果放进 register，但代价是 reduce 量跟 sequence length 线性增长，DSMEM 撑不住。

---

## 7. 实验数据深度解读

### 7.1 End-to-end TPOT（Fig.8/Fig.17）

Llama2-7B（batch=1，cluster size=4）平均 speedup：
- vs SGLang：1.41×
- vs vLLM：1.39×
- vs TensorRT-LLM：1.43×
- vs MLC-LLM：2.03×

DeepSeek-V2-Lite（batch=1）平均 speedup：
- vs SGLang：1.34×
- vs vLLM：1.37×
- vs TensorRT-LLM：1.51×
- vs MLC-LLM：2.39×

MLC-LLM 被甩得最远，因为它对 decoding 的优化比较基础，没有 CUDA Graph 全家桶。SGLang/vLLM/TRT-LLM 之间差距小，因为都用 FlashAttention/FlashInfer 这些顶尖 kernel。

### 7.2 Core module 延迟（Fig.9/Fig.18）

只看 QKV+Attention+Output Projection 这三块（不含 FFN）：
- Llama2-7B：1.85× / 1.73× / 1.61× / 3.19×
- DeepSeek-V2-Lite：1.66× / 1.64× / 1.35× / 3.5×

Core module 的 speedup 比 end-to-end 更高，因为 FFN 没被 fusion，依然是瓶颈。这暗示后续工作可以继续把 FFN 也拉进来 fusion（FFN 有两个 GEMM + 一个 element-wise，中间结果也可以 on-chip）。

### 7.3 Cluster size 调参（Fig.11）

| Heads | Optimal cluster size |
|-------|---------------------|
| 32    | 4                   |
| 64    | 4                   |
| 128   | 2                   |

Heads 越多，单个 head 能分配的 SM 越少，cluster size 要相应缩小。cluster size 8/16 普遍变差，验证了 Sec.2.3 的硬件 trade-off。

### 7.4 Ablation：DSMEM on/off（Fig.13、Tbl.1）

Tbl.1 微基准测试：
| Operation | Data Size | Off-chip (µs) | On-chip (µs) | Speedup |
|-----------|-----------|---------------|--------------|---------|
| ClusterReduce | 256 KB | 22.44 | 9.17 | 2.44× |
| ClusterGather | 256 KB | 6.61 | 4.15 | 1.59× |

数据越大，ClusterReduce 的优势越明显（256KB 时 2.44×），因为 off-chip traffic 跟数据量线性增长，而 DSMEM 的延迟相对恒定。ClusterGather 的 speedup 比较稳定在 1.5× 左右，因为它的 traffic 本身就比 Reduce 大（N-1 vs log N 轮）。

Fig.13 关闭 DSMEM 后 TPOT 增加 **最多 33%**，证明 DSMEM 是 speedup 的核心来源。

### 7.5 Multi-batch (batch=16)（附录 C.1）

batch=16 时 speedup 显著下降（Llama2-7B：1.11×~1.32×），原因 paper 给了三个：
1. KV cache 和 weight 占主导，中间 data traffic 占比小
2. 计算强度上升，从 memory-bound 往 compute-bound 移
3. atomicAdd 在 multi-batch 时争用加剧

这跟 FlashAttention 的经验一致：batch 大了之后 fusion 的边际收益递减，因为瓶颈从 memory 转向 compute。

### 7.6 Global memory traffic & launch overhead（Fig.12/Fig.19）

- Global memory transfer 减少：因为中间 Q/K/V/Attention 不再走 HBM
- Kernel launch overhead：减少近一个数量级（即使跟 CUDA Graph 比）

Kernel launch 减少是 secondary benefit，但对小 batch decoding 很关键——传统框架即使有 CUDA Graph，graph 内部还是多个 kernel node，每个 node 有 launch 抖动。

---

## 8. 跟相关工作对比

### 8.1 跟 FlashAttention/FlashDecoding 的关系

FlashAttention 融合了 attention 内部的 QK^T、softmax、AV 三个操作，但停在 attention 边界——attention 输出还得写回 HBM 给 Output Projection。

FlashDecoding 把 attention 沿 KV sequence 维度并行化，但同样停在 attention 边界，partial 结果需要单独的 rescale kernel。

ClusterFusion 的融合 scope **跨越了 attention 边界**：把 QKV Projection（attention 之前）和 Output Projection（attention 之后）都拉进来。这是关键区别——之前的工作是"kernel 内部 fusion"，ClusterFusion 是"跨 kernel fusion"。

### 8.2 跟 MonoNN、TVM、Triton 的关系

MonoNN（OSDI'24）做 global 的 monolithic 优化，但依然假设 block 间通过 global memory 通信。TVM/Triton 提供了 fusion 的 compiler 支持，但缺乏 cluster 级 collective primitive。ClusterFusion 的 primitive 可以看作是这些 compiler 框架缺失的一块"硬件原语"。

### 8.3 跟 WaferLLM、T10、Graphcore IPU、Cerebras WSE 的关系

这些系统跑在 fully inter-core connected 的硬件上（每个 core 都能直接访问别的 core），所以 collective primitive 是"原生"的。但 Hopper 是"局部 fully connected"（只有 cluster 内的 16 个 SM），所以 ClusterFusion 的设计要面对一个"边界"问题——超出 cluster scope 就得 fallback 到 HBM。Sec.5 的 discussion 明确提到这是未来架构改进的方向：topology-aware 的更广 intra-chip collective。

参考链接：
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashDecoding: https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- MonoNN: https://www.usenix.org/conference/osdi24/presentation/zhuang
- WaferLLM: https://arxiv.org/abs/2502.04563
- DeepSeek V2: https://arxiv.org/abs/2405.04434

---

## 9. 我的直觉总结与延伸联想

### 9.1 核心直觉

ClusterFusion 的本质是把 **"block 是孤立执行单元"** 这个 CUDA 编程模型的隐含假设打破了。它把 cluster 当成一个"小型 distributed system"，然后把 MPI 那套 collective primitive（reduce/gather）搬到 on-chip。这让我想起 NCCL 的 ring allreduce——只不过 NCCL 在多 GPU 间跑，ClusterFusion 在单 GPU 的 16 个 SM 间跑，latency 差了 3-4 个数量级（190 cycles vs 几十 µs）。

更深层的洞察是：**operator fusion 的边界本质上由"数据通信的代价"决定**。如果 block 间通信要 470 cycles + 走 HBM，那 fusion 就停在 kernel 边界；如果只要 190 cycles 且全 on-chip，fusion 就能跨 kernel。硬件特性直接决定了软件 fusion 的可达空间。

### 9.2 延伸联想

**1. 跟 Hopper TMA + Warp Specialization 的关系**：FlashAttention-3 用 warp specialization + TMA 做异步 memory load，跟 ClusterFusion 的 DSMEM 是正交的优化方向。理论上 ClusterFusion 也可以引入 warp specialization，让 producer warp 做 ClusterGather，consumer warp 做 Attention，进一步隐藏通信延迟。paper 没做这个，是个明显的 future work。

**2. 跟 Blackwell 的关系**：Blackwell B200 有没有扩展 cluster scope？如果 Blackwell 把 cluster size 从 16 提到 64 或 256，ClusterFusion 的 fusion scope 可以进一步扩展到 FFN，甚至整个 Transformer block。这是值得关注的下一代硬件演进方向。

**3. 跟 spec decoding / MoE 的关系**：speculative decoding 会同时跑多个 draft token，每个 token 都要过 attention，融合空间更大。MoE 的 expert routing 也有跨 block 数据依赖，可能也能用 ClusterReduce 做 expert 输出的聚合。

**4. 跟 continuous batching 的兼容性**：vLLM/SGLang 的 continuous batching 把不同 sequence 拼成一个 batch，ClusterFusion 的 cluster 分配（每个 head 一个 cluster）跟 batch 维度是正交的，理论兼容。但 paper 只测了 batch=1 和 batch=16，没测 continuous batching 下的 mixed sequence length 场景，这是个实际部署的 concern。

**5. KV cache 量化场景**：如果 KV cache 是 INT8/FP8 量化存储，ClusterGather 的 traffic 会减小（每个 element 1 字节 vs 2 字节 FP16），但 gather 之后还要 dequantize，可能影响 fusion 的 register pressure。

**6. 跟 CUDA Graph 的协同**：ClusterFusion 减少 kernel launch overhead 1 个数量级，那 CUDA Graph 还有必要吗？答案是仍然必要——CUDA Graph 解决的是 graph 级别的 launch 消除，ClusterFusion 解决的是单 kernel 内部的 fusion scope，两者正交。但 ClusterFusion 可能让某些场景下 CUDA Graph 变得冗余（比如 pure decode 单 stream）。

**7. Cost model 的可推广性**：paper 的 traffic 公式（公式 3、4）是解析的，可以直接嵌到编译器里做 dataflow 选择。这跟 TileFlow（MICRO'23）、Welder（OSDI'23）的 tile-graph 建模一脉相承。未来 LLM compiler 应该有"DSMEM-aware"的 cost model，自动决定 fusion scope 和 cluster size。

**8. Programmatic DSMEM 的语义抽象**：ClusterReduce/ClusterGather 实际上是把 distributed system 的 BSP（bulk synchronous parallel）模型搬到了 intra-GPU。这跟 recent 的"cuda async"语义（async barrier, mbarrier）结合，可能演化出更精细的 intra-cluster programming model。NVIDIA 自己的 cuCLUSTER / CUTLASS CuTe 已经在朝这个方向走。

参考链接：
- FlashAttention-3: https://arxiv.org/abs/2407.08608
- CUTLASS CuTe: https://github.com/NVIDIA/cutlass
- TileFlow: https:// Mengjia-Yan / paper 链接 https://dl.acm.org/doi/10.1145/3613424.3614265
- Welder: https://www.usenix.org/conference/osdi23/presentation/shi

---

## 10. 局限性与开放问题

paper Sec.5 自己承认的局限：cluster scope 最大 16，超出就得 fallback 到 HBM。我补充几个：

1. **Software complexity**：ClusterReduce/ClusterGather 需要手写 PTX/CUDA，paper 没提供更高层的 DSL/Triton backend，开发者复用门槛高
2. **No multi-GPU story**：DSMEM 只在单 GPU 内，跨 GPU 仍靠 NCCL + HBM，跨 node fusion 没解决
3. **Limited model coverage**：只测了 Llama2-7B 和 DeepSeek-V2-Lite，没测大模型（70B+）在 TP 切分下的行为
4. **No quantization integration**：FP16 only，FP8/INT4 量化下 attention 的 reduce 语义会变（max statistics 可能溢出），需要额外处理
5. **Ablation on long context**：32K+ 序列下，单 head 的 KV cache 可能超过 cluster 的 shared memory 容量，paper 没讨论这个边界
6. **Power/energy 没分析**：DSMEM NoC 的能耗 vs HBM access 的能耗对比没给

参考链接（paper repo）：
- ClusterFusion GitHub: https://github.com/xinhao-luo/ClusterFusion
- SGLang: https://github.com/sgl-project/sglang
- vLLM: https://github.com/vllm-project/vllm
- FlashInfer: https://github.com/flashinfer-ai/flashinfer
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM

---

## 总结一句话

ClusterFusion 把"GPU 是一堆孤立 SM"的编程模型升级成"GPU 是一组小分布式系统"，用 MPI 风格的 collective primitive 在 Hopper 的 DSMEM 上把 LLM decoding 的三个核心 kernel 融合到一起，拿回了被 global memory round-trip 浪费的延迟。这是 hardware-software co-design 的一个干净案例——新硬件特性（DSMEM）需要新软件抽象（cluster primitive）才能 unlock 真正的 fusion 空间。
