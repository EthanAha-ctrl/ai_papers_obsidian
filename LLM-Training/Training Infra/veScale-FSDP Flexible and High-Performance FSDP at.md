---
source_pdf: veScale-FSDP Flexible and High-Performance FSDP at.pdf
paper_sha256: 5ee193e5201d1c72713968e123437e20981787ac8a34db883fcfb28244653081
processed_at: '2026-08-13T00:21:42-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# veScale-FSDP 讲人话版

## 一句话概括

FSDP 训练大模型时，参数会被"切碎"分到各个 GPU 上。但这个"切"的方式太粗暴了，导致现在最前沿的训练技巧（FP8 量化、Muon 优化器）根本没法用。veScale-FSDP 重新设计了"切"的方式，让它既能匹配这些新技巧的结构需求，又不牺牲性能。

---

## 先从痛点说起

想象你在做一块 pizza，要分给 64 个人吃。

**DeepSpeed ZeRO 的做法**：把整块 pizza 扔进搅拌机打成泥，然后每人分一勺泥。公平是公平了，但你再也没法说"我要这块 pizza 上的 pepperoni"了 — pepperoni 早就被打散了。

**FSDP2 的做法**：把 pizza 切成 64 条等宽的扇形，每人一条。好一点，至少每条还是个 2D 的三角形。但如果 pepperoni 是按 4×4 的方阵排列的，你的扇形切线很可能把一颗 pepperoni 劈成两半，一半在你盘子里，一半在隔壁老王盘子里。

现在问题来了。DeepSeek-V3 训练用 FP8 量化，要求把参数切成 128×128 的方块来算 scaling factor。这就像说"每颗 pepperoni 必须完整地在一个人盘子里"。FSDP2 的等宽切法完全没法保证这一点。你得手动改模型代码、加 padding、或者手写复杂的跨卡通信来凑齐被劈开的 pepperoni。

Muon 优化器更极端。它要对整个 2D 矩阵做 Newton-Schulz 迭代（一种矩阵分解），需要完整的矩阵在一张卡上。FSDP2 把矩阵按行均分到 64 张卡，每卡只有 1/64 的行，根本没法做矩阵运算。

所以现状是：**模型的"结构需求"和 FSDP 的"切分方式"之间有一道裂缝**。模型越复杂、训练规模越大，这道裂缝越痛。

---

## RaggedShard：让"切"变得灵活

核心 idea 其实很朴素：**不要强迫每个人分到一样多的 pizza**。

RaggedShard 允许你定义一个"原子块"（atomic block），比如 128×128 的 pepperoni 方阵。然后每个 GPU 可以拿到**不同数量的块**。GPU 0 可能拿 1 个块，GPU 1 可能拿 3 个块，无所谓 — 只要每个块是完整的、不被劈开的就行。

这 generalize 了所有现有方式：

| 现有方式 | 对应的 RaggedShard 设置 |
|---|---|
| DeepSpeed element-wise | block size = 1 个 element |
| FSDP2 row-wise even | block size = 1 row，且每人块数相同 |
| DeepSeek FP8 量化 | block size = 128×128 |
| Muon optimizer | block size = 整个矩阵（1 个块，放一张卡上）|

所以 RaggedShard 不是又一个"优化 trick"，它是一个**表达能力的扩展** — 之前根本没法描述的东西，现在能用一行代码描述了。

代码上就是一个 PyTorch DTensor 的新 placement，用起来跟 `Shard(0)` 一样自然。

---

## 难点：怎么把块塞进通信 buffer

RaggedShard 解决了"怎么切"，但立刻引来一个新问题：FSDP 每次前向传播要把所有参数 `AllGather` 回来（每张卡都拿到完整参数），反向传播后要把梯度 `ReduceScatter` 切回去。这些操作需要把多个 tensor 塞进一个连续的通信 buffer 里，然后一次性通信（bucketing），这样网络利用率才高。

但塞的时候有三个坑（对应 paper Figure 6a）：

### 坑 1：Sharded Block

你把 tensor A 和 tensor B 首尾相接塞进 buffer。buffer 在每个 device 上是等长的 $S$。如果 device 边界 $kS$ 正好落在 tensor A 的某个 128×128 block 中间，这个 block 就被劈成两半了 — 你的 RaggedShard 抽象白搞了。

### 坑 2：Non-contiguous Memory

为了让每个 device 分到等长的 buffer（NCCL 要求），你可能要 padding。如果 padding 插在 tensor 内部（而不是 tensor 之间），那 tensor 在内存里就不连续了。AllGather 出来后你得把碎片 copy 到一起才能做矩阵乘法。FSDP2 就是这个毛病 — paper Table 1 实测，AllGather 花 43.7ms，之后的 copy-out 还要花 5.2ms，占了 12%。ReduceScatter 更夸张，copy-in 占 13%。

### 坑 3：Imbalanced Load

不同 tensor 大小不同、padding 不同，导致各 device 分到的有效数据量不均。Ring AllReduce 算法里，最慢的 device 决定整体速度，所以不均衡就是浪费。

---

## Planning Algorithm：NP-hard 但不用怕

这三个坑加在一起，构成了一个优化问题：**怎么排列 tensor、加多少 padding、让 buffer 尽量小，同时满足三个约束**。

### 形式化

设你有 $n$ 个 tensor $\{T_1, T_2, ..., T_n\}$，分到 $m$ 个 device 上。

每个 tensor $t$ 有：
- $g_t$：block size（原子块大小，比如 128×128 = 16384 elements）
- $e_t$：总 element 数
- $u_t = e_t / g_t$：block 个数

决策变量：
- $S$：每个 device 的 buffer 大小（要最小化的目标）
- $[\ell_t, r_t)$：每个 tensor 在 global buffer 中的起止位置

目标：
$$\min S$$

三个约束：

**约束 1 — Tensor 占满自己的区间且不超 buffer**：
$$r_t - \ell_t = e_t \quad \text{且} \quad r_t \leq mS, \quad \forall t$$

这里 $\ell_t$ 是 tensor $t$ 在 global buffer 中的起始 offset，$r_t$ 是终止 offset。$r_t - \ell_t = e_t$ 表示 tensor 恰好占满 $e_t$ 个 element。$r_t \leq mS$ 表示 tensor 不能超出 global buffer（$mS$ 是 buffer 总长，$m$ 个 device 每个 $S$）。

**约束 2 — Tensor 之间不重叠**：
$$r_t \leq \ell_{t'} \quad \vee \quad r_{t'} \leq \ell_t, \quad \forall t \neq t'$$

任意两个 tensor 要么 $t$ 在 $t'$ 左边（$r_t \leq \ell_{t'}$），要么反过来。不能交叠。

**约束 3 — Device 边界不劈开 block**：
$$kS \leq \ell_t \quad \vee \quad kS \geq r_t \quad \vee \quad (kS - \ell_t) \equiv 0 \pmod{g_t}, \quad \forall t, \forall k=1,...,m$$

对每个 device 边界 $kS$（$k=1$ 是第一个和第二个 device 的分界线，$k=2$ 是第二个和第三个...），它要么在 tensor $t$ 左边（$kS \leq \ell_t$），要么在右边（$kS \geq r_t$），要么如果落在 tensor 内部，则从 tensor 起点到边界的距离 $kS - \ell_t$ 必须是 block size $g_t$ 的整数倍。这就保证不会劈开任何 block。

### 为什么 NP-hard

这个问题可以 reduce 到经典 Partition problem：给你一堆数，能不能分成两堆和相等。这里的"分"变成了带 alignment 约束的 bin packing，更难。所以是 NP-hard。

### 解法：利用 Transformer 的规律性

ILP solver 理论上能解，但实测要几十分钟，在 10K GPU 部署时不可接受。veScale 的 insight 是：**Transformer 参数高度规律**。

Transformer 里绝大部分参数是 linear layer 的 weight matrix，形状就那么几种（hidden×hidden, hidden×ffn, ffn×hidden），而且 layer 之间重复。所以不需要搜索所有 tensor 排列，只需要试三种顺序：
1. model 定义顺序（default）
2. 按 block size 排序
3. 按 tensor shape 排序

实测 default order 已经最优或接近最优。

给定顺序后，用 dynamic programming 填 buffer。核心 trick 是把每个 tensor 分三类：
- **Case 1**：整个 tensor 在一个 device 的 shard 内
- **Case 2**：横跨两个相邻 device，但不包含完整的 shard
- **Case 3**：包含至少一个完整 shard

如果所有 tensor 都是 Case 1-2，feasibility 是 monotone 的（$S$ 可行 → $S + \Delta$ 也可行，多余空间当 padding）。如果有 Case 3 tensor，可行 $S$ 必须是这些 tensor block size 的 LCM 的倍数。

所以算法是：排序 tensor → 枚举 granularity prefix → 对每个 prefix 算 LCM → 二分搜索最小可行 $S$ → DP 检查 feasibility。

时间复杂度 $O(|T|^2 \cdot m \cdot \log E \cdot \log(|T| \cdot m))$，实测 < 0.3 秒搞定。

---

## DBuffer：干掉 copy overhead

Planning 算出了最优 layout，但执行时还需要一个高效的 buffer 管理器。这就是 DBuffer。

### FSDP2 的毛病

FSDP2 用 per-parameter sharding，每个参数独立 AllGather。AllGather 的输出 buffer 是按参数 interleaved 的 — 参数 A 的一片、参数 B 的一片、参数 A 的一片、参数 B 的一片... 你没法直接拿来做矩阵乘法，必须先 copy 到连续内存。paper Figure 2 画得很清楚。

Table 1 的实测数据（GPT-OSS-120B, 64 GPU）：

| 操作 | 通信时间 | 额外 copy 时间 | copy 占比 |
|---|---|---|---|
| AllGather | 43.71 ms | 5.22 ms (copy-out) | 12% |
| ReduceScatter | 94.24 ms | 12.37 ms (copy-in) | 13% |

如果用 Shard(1) 模式，copy overhead 更高：copy-out 13.72ms, copy-in 23.14ms。

这些 copy 是纯浪费 — 数据从一块显存搬到另一块，GPU SM 在等。

### DBuffer 的做法

DBuffer 是一个 global 的、persistent 的 buffer。Planning 算出每个 tensor 在 buffer 中的位置后，DBuffer 把 tensor 的 data pointer 直接指向 buffer 中的对应 slice。之后：

1. **Zero-copy**：AllGather 直接写进 DBuffer 的对应 slice，不需要 copy-out，因为 tensor 的 pointer 已经指向那里了，matmul 直接读。ReduceScatter 反过来，直接从 DBuffer 读、直接写回 shard。

2. **Group-level kernel fusion**：传统 FSDP 对每个 tensor 单独 launch kernel（zero、scale、add 等），DBuffer 把同质 kernel 跨 tensor fuse 成一次 launch。在 Hopper/Blackwell 上 kernel launch overhead 占比越来越高，这个优化很实打实。

3. **Batched allocation**：DBuffer 一次性分配一大块显存，避免 PyTorch caching allocator 的碎片化。FSDP2 的 per-parameter eager allocation 会导致碎片化，peak reserved memory 膨胀 20%（paper §6.1 实测）。

4. **In-place compute**：DBuffer 支持 in-place 操作，不产生中间 tensor。

Figure 7 展示了一个 2D DBuffer：dim 0 做 AllGather（FSDP 维度），dim 1 做 TP/EP 维度。梯度 reduction 时，从 (Partial, Partial) redistribute 到 (Replicate, Shard)，等价于先 ReduceScatter 再 AllReduce。

---

## 实验数据怎么看

### End-to-End 性能（Figure 8）

三个模型：LLaMA-3-70B（dense）、GPT-OSS-120B（dense）、内部 MoE model。1024 GPU。

**MoE model 上 veScale-FSDP 比所有 baseline 快 11%~66%**。这个差距很大。原因是 MoE model 的 sparse expert 计算量小但 FSDP 通信量大，所以通信效率的差距被放大了。

LLaMA-3-70B 上只快 5%，因为 dense model 的 compute/comm ratio 高，通信优化空间小。

**Memory 节省 16-30%**。主要来自 DBuffer 的 batched allocation 和确定性 deallocation。DeepSpeed 和 FSDP1 有个隐患：PyTorch 的 `record_stream` 机制导致 deallocation 时机不确定，caching allocator 没法复用 buffer，peak reserved memory 膨胀。FSDP2 的 per-parameter allocation 也有类似问题。

FSDP2 在 GPT-OSS-120B @ 256 GPU 直接 OOM。原因是 128 个 expert 分到 256 GPU，per-parameter Shard(0) 要求 even split，需要 padding 把 buffer 翻倍。这就是 even sharding 的硬伤。

### Scalability（Figure 9）

- **Weak scaling**：800B MoE，1K→8K GPU，每 GPU batch 固定，近线性。FSDP 的通信量只取决于 model size 和 per-GPU batch，跟 GPU 数无关，所以 weak scaling 天然好。
- **Strong scaling**：16M token global batch，1K→8K GPU，3.4× speedup（不是 8× 因为 per-GPU token 太少，通信 dominate）。128M token global batch 可以线性到 10K GPU。
- **Model scaling**：400B→2.4T @ 1K GPU，MFU 略升（compute intensity 变高，GPU 利用率更好）。

### Padding Overhead（Figure 11）

DeepSeek-V3-671B 和 GPT-OSS-120B，sweep row granularity：

- **1× 和 16×**：padding < 3%，非常好
- **128×**（DeepSeek FP8 的 128×128 tiling）：DeepSeek-V3 仍 < 3%，但 GPT-OSS 偶尔 spike 到 18%

为什么 GPT-OSS 更糟？因为 GPT-OSS 把所有 expert fuse 成一个大 tensor，padding 必须满足全局约束。DeepSeek-V3 每个 expert 是独立 tensor，padding 可以 per-expert 算，约束更松。

这给 model design 一个 engineering hint：**如果要用 block-wise 量化，尽量让每个 expert 保持独立 tensor，别 fuse**。

### Ablation（Table 2）

GPT-OSS + 8-bit Adam @ 32 GPU：

| 配置 | 归一化吞吐 |
|---|---|
| Full system | 100% |
| 去掉 DBuffer | 92.8%（-7.2%）|
| 去掉 Planning | 65.4%（-34.6%）|
| 去掉 RaggedShard | N/A（跑不了）|

Planning algorithm 贡献最大（34.6%）。没有 planning，quantization block 不保证在 device shard 内，系统要 fallback 到 DTensor redistribution 来凑齐 block，产生大量额外通信。

RaggedShard 不是优化而是 enabler — 没有它，你必须改模型代码让 block 边界跟 shard 边界对齐，或者手写跨卡通信交换 quantization metadata。

---

## Case Study：Muon 和 8-bit Adam

### 8-bit Adam

8-bit Adam 把 gradient statistics（m, v）用 INT8 block-wise 量化，省 optimizer state 内存。block 大小 32×32。

用 RaggedShard：设 matrix 参数的 granularity 为 32 row，每 device 的 shard 天然按 32 行对齐。每个 device 独立量化自己的 shard，不需要任何跨卡通信交换 quantization metadata。

用现有 FSDP：shard 边界随机，32×32 block 经常被切断。你要么改模型代码加 padding，要么手写 NCCL 通信来交换被切断 block 的 metadata。

### Distributed Muon

Muon 是 2024 年 Keller Jordan 提出的 optimizer，核心 idea 是对 hidden layer 的 2D weight matrix 做 Newton-Schulz 迭代近似矩阵 sign，作为 preconditioner。需要完整 2D 矩阵在一张卡上。

Algorithm 2 的伪代码：

```
for w in 2D parameter tensors:
    g ← grad(w)                    # 取梯度
    u ← MomentumUpdate(g, m)       # 动量更新
    r ← SelectRoot()               # 选一张卡做 root
    o ← Redistribute(u, RaggedShard(r))   # unshard 到 root
    o ← NewtonSchulz(o)            # 只在 root 上跑 Newton-Schulz
    o ← Redistribute(o, placement(u))    # shard 回去
    w ← w − η · o                  # 更新权重
```

关键：`Redistribute` 是标准 DTensor API，加 `RaggedShard(r)` placement。用户不需要写一句 NCCL 代码，SPMD 语义自动处理。非 root rank 的 NewtonSchulz 是 no-op（没有完整矩阵）。通信可以通过 async redistribute 与计算 overlap。

实测 256 GPU 上 47.3% MFU，用 torch.compile 进一步提升 compute density。Loss curve 显示 Muon 在 ~80B tokens 后比 AdamW 低约 0.01，与 Wen et al. 2025 (https://arxiv.org/abs/2509.02046) 的 benchmark 一致。

这也解释了为什么 ByteDance 做 veScale-FSDP — 他们在追 Muon 路线，需要一个能 natively 支持 matrix optimizer 的 FSDP 系统。

---

## 跟其他系统的关系

### vs JAX GSPMD

JAX 的 `pjit` + `PartitionSpec` 2021 年就支持任意 sharding annotation，包括 block-level。但 JAX 用 tracing + XLA 编译，planning 可以很慢很复杂（编译一次跑多次）。PyTorch eager mode 要求 planning 快速（毫秒级），所以 veScale 用 LCM + DP heuristic 而不是完整 ILP solver。这是 eager vs compiled 的 trade-off。

RaggedShard 已被列入 PyTorch 2026 H1 roadmap（https://dev-discuss.pytorch.org/t/meta-pytorch-team-2026-h1-roadmaps），说明社区认可这个方向。

### vs Megatron-LM

Megatron-LM 把 parallelism 优化和 model code 紧耦合 — 改 model 要动 parallelism config。veScale-FSDP 用 `fully_shard` API 保持 PyTorch-native 接口，model 开发者不需要懂 system。这是 veScale 的 design philosophy（参考 veScale: https://arxiv.org/abs/2509.07003）。

### vs Moonshot/Kimi K2

Kimi K2 (Moonshot) 用 Muon 训练 ~1T 参数 MoE，distributed Muon 实现就是这类 RaggedShard-based approach。这解释了 Muon 在 production-scale 的可行性。

---

## 三个 Takeaway

**1. Abstraction 先于性能**

RaggedShard 本身不让你变快，它让你**能做到之前做不到的事**。性能来自 DBuffer + Planning，但灵活性来自 RaggedShard。没有 RaggedShard，block-wise 量化和 Muon 要侵入式改模型代码；有了 RaggedShard，一行 placement 搞定。

**2. NP-hard 不怕，domain regularity 救你**

Transformer 参数高度规律（linear weight 主导、layer 间重复），所以 default order 就接近最优。完整 ILP solver 反而不实用。这跟 compiler 里的经验一致 — 利用 domain-specific regularity 比 general-purpose solver 更 effective。

**3. 站在 ecosystem 上**

RaggedShard 作为 DTensor 的 optional placement，自动复用 distributed checkpoint、TP/EP composition、eager-mode SPMD。这是 force multiplier — 7.6K 行代码就重造了 FSDP2 的 backend。

---

## 对 Karpathy 的 mental model

如果你要在 nanoGPT 上加 FSDP，会立刻遇到这个 shard/block misalignment 问题。一层里 $W_q, W_k, W_v, W_o, W_1, W_2$ 六个 matrix，shape 各不相同。FSDP2 的 per-parameter even shard 会让每个 matrix 独立 AllGather，通信碎片化。RaggedShard 的思路是把它们 group 起来、按 block 对齐、一次性通信 — 同时保留每个 matrix 的 2D shape 给未来可能的 Muon/FP8 使用。

开源代码在 https://github.com/volcengine/veScale ，可以直接看 RaggedShard 和 DBuffer 的实现。

---

# veScale-FSDP 深度解析 — 为 Karpathy 构建 Intuition

非常 fascinating 的 paper，因为 veScale-FSDP 正面解决了 production-scale 训练中一个长期被忽视的痛点：**FSDP 的 shard 边界与 model/optimizer 的结构边界对不齐**。这个看似简单的 misalignment 在 LLM 训练进到 10K+ GPU、trillion 参数、并采用 block-wise FP8 quantization 和 matrix-based optimizers (Muon/Shampoo) 的时代，会演化成严重的工程债和性能损耗。下面我尽量把技术细节摊开讲，帮你建立 mental model。

---

## 1. 问题本质：为什么现有 FSDP 会 "crack"

让我从 intuition 出发。FSDP/ZeRO 的核心 idea 是：把一层参数 $W \in \mathbb{R}^{M \times N}$ 在 forward 前 `AllGather`，backward 后 `ReduceScatter` gradient。早期 ZeRO 把 layer 内所有 tensor 拼接后 element-wise 切，导致：

1. **Element-wise shard (DeepSpeed / FSDP1)**：$W$ 被切成 $m$ 份，每份形状不规则（$m$ = device 数）。tensor 的 shape/stride 信息丢失。比如 $W$ 是 $(4096, 4096)$，64 个 GPU，每 GPU 拿到 262144 个元素，但 reshape 成什么？没法知道。这种 shard 完全破坏了 2D 结构 — matrix optimizer 没法做 Newton-Schulz 迭代（需要 full 2D matrix 在 local），FP8 block quantization 的 128×128 block 会被切断到两个 device 上。

2. **Row-wise even shard (FSDP2)**：每 device 拿 $W[0:64, :]$、$W[64:128, :]$ ... 这样 2D shape 保留，但 row 数被强行 even 化。问题：DeepSeek-V3 的 FP8 quantization 要求 128×128 的 atomic block，如果 $M = 4096$ 不被 128 整除（实际 4096/128 = 32 整除，但 MoE expert 的 intermediate dim 经常是 6144、7168 这种），FSDP2 必须手动 padding 或 reshape model code。

3. **Megatron-FSDP 的 padding 通胀**：为了能让 concat 后的 shard 仍保持 `Shard(0)` DTensor 语义（方便 distributed checkpoint），它在 concat 中插入 padding 让 row 边界对齐 device 边界。Paper §6.1 实测 MoE model 上 padding 膨胀 33%，直接拖慢 collectives。

**核心 insight**：FSDP 的 shard granularity 应当由 *model/optimizer 的 block structure* 决定，而不是由 device 数硬性均分。这是 veScale-FSDP 设计 RaggedShard 的根本动机。

参考：
- PyTorch FSDP2 RFC: https://github.com/pytorch/pytorch/issues/114299
- NCCL alignment caveat: https://github.com/NVIDIA/nccl/issues/413
- DeepSpeed slow issue: https://github.com/deepspeedai/DeepSpeed/issues/5047

---

## 2. RaggedShard：核心 Abstraction

### 2.1 Definition

RaggedShard 是 DTensor 的一种新 placement，允许：
- **Arbitrary sharding granularity**：定义一个 atomic block shape $g_t$（可以是 1 element、1 row、$32 \times 32$ block、$128 \times 128$ block），block 内不可再分。
- **Arbitrary distribution**：每个 device 可以持有不同数量的 blocks。例如 device 0 拿 1 个 block，device 1 拿 2 个 blocks。

这 generalize 了所有现有 format：
- $g_t = 1$ element → Element-wise shard (DeepSpeed/FSDP1)
- $g_t = 1$ row → Row-wise even shard (FSDP2，当 distribution 均匀时)
- $g_t = (128, 128)$ → Block-wise shard for FP8 quantization

### 2.2 为什么这个 abstraction 妙

我类比一下：在 JAX 的 GSPMD/XLA SPMD 里，sharding annotation 是 user-specified 的 partition spec，比如 `P("data", "model")`。但 PyTorch DTensor 历史上只有 `Shard(dim)`, `Replicate`, `Partial` 三种 placement，没法表达 "block-level atomic"。RaggedShard 实际上是把 GSPMD 的 partition spec 思想引入 PyTorch DTensor，但保留了 PyTorch eager-mode 的执行模型。

Paper §4 提到一个 subtle 的点：与现有 `Shard(0)` 组合时，引入 `StridedRaggedShard` 来携带 reorder/stride metadata。这是因为 PyTorch DTensor 的 placement list 是逆序应用的：`(RaggedShard, Shard(0))` 意味着先 Shard(0) 再 RaggedShard。对于 EP（Expert Parallel），EP 先 Shard(0) 沿 expert dim，再 FSDP RaggedShard — 需要 stride 信息来 materialize 时正确 reorder。

### 2.3 Granularity 与 TP/EP 的 LCM 协调

当 TP 用 `Shard(1)` 沿 row 切时，RaggedShard 的 granularity 必须**不切入** dim=1。Paper 用 LCM 公式：

$$g_{\text{ragged}} = \text{LCM}(\text{stride}_{\text{dim}}, g_{\text{user}})$$

其中 $\text{stride}_{\text{dim}}$ 是 TP 沿 dim 的 stride，$g_{\text{user}}$ 是 user-defined granularity。这保证 RaggedShard 切的时候 TP shard 不会被进一步切碎。这是工程上一个很 pragmatic 的妥协 — 完整的 N-D block-level sharding 在 GSPMD 里需要更复杂的 constraint solver，这里用 LCM 退化处理。

---

## 3. Planning Algorithm：把 NP-hard 拆成可解

### 3.1 优化问题形式化

Paper §5 给了 ILP formulation。设 tensor 集合 $\mathcal{T} = \{T_1, ..., T_n\}$，sharded across $m$ devices。每个 tensor $t$：
- $g_t$：block size (atomic block 大小，单位 element)
- $e_t$：total tensor size
- $u_t = e_t / g_t$：number of blocks

决策变量：global buffer size $S$，每个 tensor 的 interval $[\ell_t, r_t)$。

目标：
$$\min_{S, \{\ell_t, r_t\}} S$$

约束（对应 Figure 6(b) 的三个条件）：

1. **Tensor interval 等于自身 size 且不超 buffer**：
$$r_t - \ell_t = e_t \quad \wedge \quad r_t \leq mS, \quad \forall t \in \mathcal{T}$$
   解释：每个 tensor 占据连续 interval，最右端不超过 global buffer 总长 $mS$。

2. **Tensor 之间不重叠**：
$$r_t \leq \ell_{t'} \quad \vee \quad r_{t'} \leq \ell_t, \quad \forall t \neq t'$$
   解释：任意两个 tensor interval 要么 $t$ 在 $t'$ 左边，要么反之。

3. **Shard 边界不切入 block 内部**：
$$kS \leq \ell_t \quad \vee \quad kS \geq r_t \quad \vee \quad (kS - \ell_t) \equiv 0 \pmod{g_t}, \quad \forall t, \forall k = 1, ..., m$$
   解释：对每个 device 边界 $kS$，它要么在 tensor $t$ 左边，要么右边，要么正好落在 $t$ 内部某个 block 的边界上（即从 $\ell_t$ 起的偏移是 $g_t$ 的倍数）。这就保证了 **sharded block** 不会发生 — 量化 block 不会被切断。

这个问题是 NP-hard，reduce 自经典 Partition problem（Garey & Johnson 1975）。

### 3.2 Heuristic：DP + 二分搜索

Algorithm 1 的核心 insight 是 **case analysis**：对每个 tensor 在 buffer 中的位置分三类：
- **Case 1**：完全在某个 local shard 内（$\ell_t, r_t \in [(k-1)S, kS]$）
- **Case 2**：横跨两个相邻 shard，但不包含完整 shard
- **Case 3**：包含至少一个完整 shard

如果所有 tensor 都属于 Case 1-2，则 feasibility 对 $S$ 是 monotone 的：若 $S$ 可行，则 $S + \Delta$ 也可行（$\Delta$ 是 base alignment quantum，通常 $g_{\text{coll}}$ = NCCL preferred unit）。因为多余空间可以吸收为 padding。

若有 tensor 属 Case 3，则可行的 $S$ 必须是 $L = \text{LCM}\{g_t \mid t \text{ in Case 3}\}$ 的倍数。在该 regime 下 feasibility 在倍数上 monotone：$kL$ 可行 → $(k+1)L$ 可行。

**Algorithm 步骤**（结合 Line 17-21）：
1. 对 $G = \{g_t\}$ 排序，累积计算 LCM $g$
2. 对每个 granularity prefix，计算 candidate $S' = \min\{k \cdot g : \text{CheckValidShard}(k \cdot g)\}$
3. CheckValidShard 内部用 DP：$dp(t, i)$ = 存储 tensor $t$ 前 $i$ 个 unit 所需最小 device 数
4. 利用 $dp(t, i)$ 的 monotonicity（在 tensor 内至多 $m$ 个不同值），skip 中间 index

时间复杂度 $O(|T|^2 m \log E \log(|T| m))$。这里 $|T|$ 是 tensor 数，$m$ 是 device 数，$E$ 是 total elements。Paper 实测规划时间 < 0.3s，对 2.4T 模型一次性成本可忽略。

### 3.3 Permutation heuristic

Paper 探索三种 tensor 顺序：
- (i) default order（model 定义顺序）
- (ii) 按 $g_t$ 排序
- (iii) 按 tensor shape 排序

实证发现 transformer model 的参数高度结构化（linear weight 主导，layer 间 block size 一致），default order 已经接近最优。这是关键 engineering insight — 利用 domain regularity 避免 search 指数爆炸。

参考 Partition problem NP-hardness: https://en.wikipedia.org/wiki/Partition_problem

---

## 4. DBuffer：让 RaggedShard 真正 zero-copy

RaggedShard 解决了 abstraction，planning 解决了 layout，但底层 execution 仍需高效。DBuffer 是第三个组件：

### 4.1 设计要点

1. **Global buffer semantics over N-D device topology**：类似 DTensor，DBuffer 在 2D device mesh 上提供 sharding spec。Figure 7 显示一个 2D DBuffer：dim 0 是 FSDP group（AllGather），dim 1 是 TP/EP group。

2. **Group-level operator fusion**：传统 FSDP 对每个 tensor 单独 launch CUDA kernel（add/scale/zero/copy），导致 kernel fragmentation。DBuffer 把同质 kernel 跨 tensor fuse 在一次 launch，减少 CPU launch overhead 和 GPU sync stall。这点很关键 — 在 Hopper/Blackwell 上 kernel launch overhead 占比越来越显眼。

3. **Persistent address mapping → zero-copy**：planning algorithm 计算出每个 tensor 在 global buffer 中的 $[\ell_t, r_t)$ 后，DBuffer 持久化地映射 tensor 的 data pointer 到 buffer slice。通信前后不需要 copy-in/copy-out，直接 in-place。

4. **In-place communication and computation**：AllGather 直接写 DBuffer，ReduceScatter 直接从 DBuffer 读。

### 4.2 FSDP2 的 copy overhead 反面教材

Paper Table 1 给了 GPT-OSS-120B 64 GPU 实测：
- AllGather: 43.71 ms
- Copy-Out (after AllGather): 5.22 ms (12% of AllGather!)
- ReduceScatter: 94.24 ms  
- Copy-In (before ReduceScatter): 12.37 ms (13% of ReduceScatter!)

Shard(1) 模式更糟：copy-out 13.72ms, copy-in 23.14ms。这些 copy 完全来自 FSDP2 的 per-parameter Shard(0) 设计 — AllGather 出来的 buffer 是 interleaved 内存地址，必须 copy 到连续 tensor 才能做 matmul。DBuffer 通过 planning 一次性保证 buffer 内 tensor 连续，消掉这些 copy。

---

## 5. 实验数据深度解读

### 5.1 End-to-End (Figure 8)

LLaMA-3-70B / GPT-OSS-120B / 内部 MoE model，1024 GPUs：

| Model | vs DeepSpeed | vs FSDP1 | vs FSDP2 | vs Megatron-FSDP | Memory 节省 |
|---|---|---|---|---|---|
| LLaMA-3-70B | +5% | +5% | +5% | 微胜 | 16-30% |
| MoE | +11~66% | +11~66% | +11~66% | +11~66% | 显著 |

Memory 16-30% 节省来自：
- 确定性 batched allocation（避免 PyTorch caching allocator 的 record_stream non-determinism）
- 显式 stream dependency management
- DBuffer 一次性分配，无碎片

FSDP2 在 GPT-OSS-120B @ 256 GPU OOM，原因是 128 experts over 256 GPU 需要 padding 让 even split，AllGather buffer 翻倍。这印证了 RaggedShard 的必要性。

### 5.2 Scalability (Figure 9)

- **Weak scaling**: 800B MoE，1K → 8K GPU，输入固定 2K-16K tokens/GPU，近线性。这符合 FSDP communication cost 只依赖 model size 和 per-GPU batch，不依赖 GPU 数。
- **Strong scaling**: 16M-128M token global batch，1K → 10K GPU，128M batch 下线性，16M batch 1K→8K 仍 3.4× speedup。当 GPU 太多、per-GPU tokens 太少时，FSDP 通信 dominate — 通过 cross-node EP 缓解但引入 token exchange 开销。
- **Model scaling**: 400B → 2.4T @ 1K GPU，MFU 反而略升，因为 compute intensity 提升。

### 5.3 Padding Overhead (Figure 11)

DeepSeek-V3-671B 和 GPT-OSS-120B，sweep row granularity 1×/16×/128×：

- 1× 和 16×：padding < 3% across all FSDP sizes
- 128×：DeepSeek-V3 仍 < 3%（per-expert padding），GPT-OSS 偶尔 spike 到 18%

GPT-OSS spike 原因：把所有 expert fuse 成单 tensor，padding 必须满足全局约束。DeepSeek-V3 每个 expert 独立 tensor，per-expert padding 放松约束。这给 model 设计一个 hint：**结构粒度与 FSDP shard group 互质时尽量保持 per-expert tensor 独立**。

### 5.4 Ablation (Table 2)

GPT-OSS + 8-bit Adam @ 32 GPU：
- Full system: 100%
- -DBuffer: 92.8% (-7.2%)
- -Planning: 65.4% (-34.6%)
- -RaggedShard: N/A（必须改 model/optimizer 或手写 collectives）

Planning algorithm 贡献最大，因为它保证 quantization block 不被切断，否则要 fallback 到 DTensor redistribution 重组 optimizer states。这显示 **abstraction 的真正价值是避免 fallback path 的 expensive 通信**。

---

## 6. Case Studies：8-bit Adam 和 Muon

### 6.1 8-bit Adam (Figure 10a)

实现：32×32 quantization block，matrix 参数用 32-row block granularity。RaggedShard 保证 device 间 block 边界完美对齐，**无需任何 cross-device metadata exchange**。Loss curve 与 DDP 8-bit Adam 接近，差异来自 layer-wise ReduceScatter vs bucketed AllReduce 的 reduction schedule。

### 6.2 Distributed Muon (Algorithm 2)

Muon 是 hidden layer optimizer，用 Newton-Schulz 迭代计算 matrix sign preconditioner，需要 full 2D parameter matrix 本地。Algorithm 2 步骤：

```
for w in 2D parameter tensors:
    g ← grad(w)
    u ← MomentumUpdate(g, m)
    r ← SelectRoot()           # load balancing
    o ← Redistribute(u, RaggedShard(r))   # unshard to root
    o ← NewtonSchulz(o)        # only on root
    o ← Redistribute(o, placement(u))    # shard back
    w ← w − η o
```

关键：`Redistribute` 是标准 DTensor API + RaggedShard placement。**用户不需要写 collective 代码**，SPMD 语义自动处理。非 root rank 的 NewtonSchulz 是 no-op。通信可与计算 overlap 通过 async redistribute。优化后 Muon 在 256 GPU 上达到 47.3% MFU，用 torch.compile 进一步提升 compute density。

这跟 Moonlight/Kimi K2 训练用的 Muon 路线呼应 — paper 提到 Muon 在 80B tokens 后稳定低于 AdamW ~0.01 loss，与 Wen et al. 2025 一致。

参考：
- Muon: https://kellerjordan.github.io/posts/muon/
- 8-bit Adam: https://arxiv.org/abs/2110.02861
- Optimizer benchmark: https://arxiv.org/abs/2509.02046
- DeepSeek-V3: https://arxiv.org/abs/2412.19437

---

## 7. Lessons 中的 Engineering Wisdom

### Lesson 1：小规模预测大规模

FSDP 的 per-layer compute time + per-layer FSDP communication time 在 GPU 数增加时基本不变（per-GPU model size 不变）。所以 64 GPU profile 可外推到 10K GPU。前提是 network topology 相似、collective algorithm 一致、workload 足够大 saturate bandwidth。Practice 中用 HSDP/EP cap collective group size 防 latency variance。

这是 production 系统 scaling 预测的金科玉律 — 在 NVIDIA NeMo/MBH 等系统里也是这样做的。

### Lesson 2：站在 DTensor 肩膀上

RaggedShard 作为 DTensor 的 optional placement，自动复用：
- Distributed checkpoint (https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- TP/EP composition
- Eager-mode SPMD (veScale: https://arxiv.org/abs/2509.07003)

RaggedShard 已被列入 PyTorch 2026 H1 roadmap (https://dev-discuss.pytorch.org/t/meta-pytorch-team-2026-h1-roadmaps) — 说明这个 abstraction 是 community 共识。

### Lesson 3：解耦 model 和 system

Megatron-LM 把 system-level parallelization 和 model code 紧耦合，model 改一行要动 parallelism config。veScale-FSDP 用 `fully_shard` API 保持 PyTorch-native 接口，model 开发者不需要懂 system。

---

## 8. 我的批判性思考与联想

### 8.1 与 GSPMD/JAX 的对比

JAX 的 `pjit` + ` PartitionSpec` 在 2021 年就已经支持 N-D mesh 上的 arbitrary sharding，包括 named sharding with manual mesh。veScale-FSDP 的 RaggedShard 某种程度上是把 JAX 的 GSPMD 思想 retro-fit 到 PyTorch eager-mode，但保留了 PyTorch 的 imperative programming model — 这是 trade-off：JAX 的 tracing 一次编译多次执行，planning 可以更激进；PyTorch eager 必须快速 planning，所以 veScale 用 LCM + DP heuristic 而非完整 ILP solver。

### 8.2 与 torch.compile 的潜在结合

Paper §6.3 提到 Muon 用 torch.compile 提高 compute density。RaggedShard DTensor 的 placement 信息理论上可以喂给 torch.compile 的 Inductor 后端做 kernel fusion — 比如把 AllGather + matmul fuse 成一个 grouped GEMM with sharded input。这是未来 Hopper/Blackwell 上 further optimization 的方向。

### 8.3 类似 work

- **GSPMD (XLA)**: https://arxiv.org/abs/2105.04663
- **GShard**: https://arxiv.org/abs/2006.16668 — EP 的开山之作
- **Megatron-LM**: https://arxiv.org/abs/1909.08053 — TP 经典
- **MegaScale (字节)**: https://www.usenix.org/conference/nsdi24/presentation/jiang — 10K GPU 训练
- **veScale**: https://arxiv.org/abs/2509.07003 — veScale-FSDP 的 base framework
- **VeOmni**: https://arxiv.org/abs/2508.02317 — 字节的 modality-agnostic recipe zoo
- **PyTorch FSDP 经验**: https://arxiv.org/abs/2304.11277
- **ZeRO**: https://arxiv.org/abs/1910.02054 (实际 SC20)

### 8.4 RaggedShard 的数学本质

更深一层看，RaggedShard 实际上是把 tensor 的 sharding 表示为一个 **quotient space**：tensor space $\mathbb{R}^{e_t}$ 被 block size $g_t$ 划分为 $u_t$ 个等价类（blocks），sharding 是这些 blocks 在 device 上的分配。planning algorithm 解决的是把这些 blocks 装入 $m$ 个 device 的 "bin packing with alignment constraints"，最小化 max bin size $S$。这是一个 generalized bin packing problem，所以 NP-hard 不意外。

### 8.5 对 Karpathy 的 nanoGPT 启示

如果你在 nanoGPT 上加 FSDP，会立刻遇到这个 misalignment 问题 — 即使是单 layer 也涉及 $W_q, W_k, W_v, W_o, W_1, W_2$ 多个 tensor，每个 shape 不同。RaggedShard 的 mental model 可以让 nanoGPT 的 parallelism 写得更干净。开源 code: https://github.com/volcengine/veScale

### 8.6 关于 Moonshot/Kimi K2 的 Muon

Paper 提到 Muon 用于 Gemini/Kimi K2 级别 model。实际上 Kimi K2 (Moonshot) 公开报告 (https://arxiv.org/abs/2509.02046 附近的 ref) 用 Muon 训练 1T 参数 MoE，需要的 distributed Muon 实现正是 veScale-FSDP 这种 RaggedShard-based approach。这解释了为什么 ByteDance 自己也投入做 veScale-FSDP — 字节也在追 Muon 路线。

### 8.7 与 FP8 训练的下一阶段

DeepSeek-V3 的 FP8 训练 (https://arxiv.org/abs/2412.19437) 已经把 128×128 block quantization 推上 production。下一代 Hopper/Blackwell 的 FP8 tensor core 要求 block-aligned access。RaggedShard 的 block-wise granularity 让 FP8 quantization 的 block 与 sharding block 天然对齐，避免 cross-device 的 quantization metadata 交换。这是 FP8-at-scale 训练的 enabling technology。

---

## 9. 总结：Three Takeaways for Building Intuition

1. **Abstraction precedes performance**：RaggedShard 不是一个 "优化"，而是一个让 block-wise quantization 和 matrix optimizer **可能** 的 abstraction。Performance 来自 DBuffer + Planning，但 flexibility 来自 RaggedShard。
2. **NP-hard 问题在 domain-specific 场景下用 regularity heuristic 就够**：Transformer 参数高度结构化，default order 已经接近最优。完整 ILP solver 反而不实用（tens of minutes planning time）。
3. **站在 ecosystem 上**：基于 DTensor 而非另起炉灶，让 RaggedShard 自动继承 checkpointing、TP/EP composition、eager-mode SPMD 等成熟基础设施。这是工程上的 force multiplier。

对 Karpathy 而言，这个 paper 是研究 production-scale training system 的好案例 — 它展示了从一个具体的工程痛点（block 与 shard 边界 misalignment）出发，如何设计 abstraction（RaggedShard）+ algorithm（planning）+ runtime（DBuffer）三层 stack，最终在 10K GPU / 2.4T 参数上验证。这种 "abstraction-driven systems design" 路径正是 PyTorch distributed evolution 的下一步。

开源代码：https://github.com/volcengine/veScale
