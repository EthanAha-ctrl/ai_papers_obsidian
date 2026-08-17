---
source_pdf: Scalable Training of Mixture-of-Experts Models with Megatron Core.pdf
paper_sha256: 649b8a1bcd79dbc13f63827defec3b5de371d121c6d613961c105c24a7953ca3
processed_at: '2026-08-12T03:18:02-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

好,我把技术细节都拆掉了,用最直觉的方式讲一遍。目标就是让你读完能 build 一个 mental model,知道 MoE training 在 system 层面到底在跟什么搏斗。

---

## 1. MoE 训练的根本矛盾:参数跟计算脱钩了

先从最根本的地方说起。Dense transformer 有一个特别美好的特性,我管它叫"良性耦合":模型参数 $N$ 变大 → 每 token 计算量 $\approx 6N$ 变大 → 你需要更多 GPU 分摊 → 每个 GPU 分到的计算也变多 → 通信占比反而下降 → GPU 利用率稳定。这是一个 virtuous cycle,你越 scale 越 happy。

MoE 直接把这个循环打破了。你看 DeepSeek-V3 的数字:
- Total params $N_{\text{total}} = 685B$,随 expert 数 $E$ 增长
- Active params per token $N_{\text{active}} = 37B$,随 top-$K$ 增长
- $K \ll E$,所以 ratio 是 $18\times$

每 token FLOPs 大致是 $6 N_{\text{active}}$ 而非 $6 N_{\text{total}}$。这意味着你为了 memory 不得不把模型分到很多 GPU 上,但每个 token 实际要算的 FLOPs 没有同步增长,communication 占比就上来了。

Paper 里图 3 把这个画得很直观。Dense model 沿着 $\approx 2N$ 的 reference line 爬,MoE model 远远掉在 line 下面。这个 gap 就是 system 问题的根源。

打个比方:Dense model 像"人越多活越多,分到更多人手里每人还是够忙";MoE model 像"人很多但每时刻只有少数人干活,人越多闲人越多,互相通信成本反而暴露"。

---

## 2. 三堵墙:MoE 的三大 system 瓶颈

Paper 把 MoE 的 system 挑战归纳成 Three Walls。我用比喻讲:

### Memory Wall:仓库装不下

所有 $E$ 个 expert 的 parameters, gradients, optimizer states 都得驻留在 memory,即便每个 token 只激活 $K$ 个。Paper Table 3 给 DeepSeek-V3 的 per-GPU 分解:

| Component | Memory per GPU |
|---|---|
| Weights & Gradients | 36.4 GB |
| Main Weights & Optimizer States | 32.1 GB |
| Activations | 131.0 GB |
| Total | 199.5 GB |

H100 才 80 GB,直接装不下。反直觉的是 activations 比 weights + optimizer 加起来还大,因为 top-$K$ routing 会复制 token,activation 随 $K$、hidden dim $h$、batch、sequence length 一起膨胀,还有 dynamic routing 带来的 load imbalance spike。

### Communication Wall:货物运不过来

EP (Expert Parallelism) 把不同 expert 放到不同 GPU,每个 token 要 dispatch 到它的 expert 所在 GPU,算完再 combine 回来。这是 all-to-all 通信,每层 MoE 两次。per-GPU send volume 大致:

$$T \cdot K \cdot h \cdot \frac{EP - 1}{EP}$$

其中 $T$ 是 local token count,$K$ 是 top-$K$,$h$ 是 hidden dimension,$EP$ 是 expert parallelism degree。

DeepSeek-V3 有 58 层 MoE,forward 就 116 次 all-to-all,backward 翻倍。当 EP 跨 node 时(H100 上 NVL8,EP=64 要跨 8 个 node),inter-node 带宽比 NVLink 低 5–10×,paper 实测 unoptimized all-to-all 能吃掉 60% 训练时间。

### Compute Efficiency Wall:机器空转

Fine-grained MoE(DeepSeek-V3 有 256 个 expert)产生大量 small GEMM。Paper 实测:
- Llama-3 405B (dense):GEMM 占 ~70% 执行时间
- DeepSeek-V3 (MoE):GEMM 占 <50%,剩下都是 routing, permutation, kernel launch overhead 这些跟 tensor 数量(而非 FLOP 数量)scale 的操作

还有 host overhead:MoE 因为 sparsity + routing,同样的 FLOP 要 launch 更多 kernel,每个 kernel launch 有几微秒 CPU-side cost,GPU 在 kernel 之间就 idle 了。这叫 host-boundedness。

---

## 3. 三堵墙是耦合的——这是 paper 最 important 的 insight

这是这篇 paper 反复强调的点,也是我最想 build 给你的 intuition。

想象一个真实的优化 trajectory:
1. 你训 1000B MoE,先撞 Memory Wall,activation 超出 GPU 容量
2. 你开 activation recomputation,把 memory 压下去
3. Profiling 发现 all-to-all communication 现在成 dominant 了——compute wall 之前把 communication wall 藏住了
4. 你开 communication overlap,把 all-to-all 藏到 compute 后面
5. 但 fine-grained expert 算太快,没足够 compute 给你 overlap
6. 你开 FP8 降 memory,换更大 batch,有更多 compute 可以藏 communication
7. FP8 的 quantization kernel 增加 CPU overhead,系统变 host-bound,GPU idle
8. 你开 CUDA Graphs 减 launch overhead
9. 但 CUDA Graphs 要 static shape,跟 dropless routing 的 dynamic token count 冲突
10. 你又得搞 sync-free kernel + ECHO + Paged Stashing 才能让 CUDA Graphs 覆盖 dynamic 部分

每一步解决一个问题,又把压力 shift 到另一堵墙。所以 isolated 优化必然 suboptimal,必须 co-design 整个 stack。

---

## 4. Parallel Folding:paper 最 elegant 的 contribution

这是我最喜欢的部分。先说问题本质。

一个 transformer block 里同时有两种 computation pattern:
- **Attention(dense)**:每个 token 跟所有其他 token attend,需要大 QKV matrix,benefits from high TP,benefits from high CP for long sequence
- **MoE(sparse)**:每个 token route 到 $K$ 个 expert,expert 的 hidden dim 很小,high TP 会切碎小 shard;CP 跟它完全无关

传统 framework 把 EP 当作 DP 的子维度:

$$\text{World Size} = TP \times CP \times PP \times DP, \quad EP \subseteq DP$$

这导致三个问题:
1. **GPU 需求乘性膨胀**:想要 EP=8 + CP=8,传统下最少要 $1 \times 8 \times 1 \times 8 = 64$ GPU,即便 attention 和 MoE 理论上能共用 8 个 GPU
2. **Forced suboptimal**:attention 和 MoE 共享 TP,要么选 high TP 切碎 expert,要么选 low TP 让 attention under-parallelized
3. **Cross-node communication**:EP 被锁在 DP 里,高 EP 必然跨 node

**Parallel Folding 的解法极简:不要强迫 attention 和 MoE 共享 parallelism 配置。**

- Attention layers form groups over $TP \times CP \times DP \times PP$,优化 sequence-level dense computation
- MoE layers form groups over $ETP \times EP \times EDP \times PP$,其中 $ETP$ (Expert Tensor Parallelism) 和 $EDP$ (Expert Data Parallelism) 是 MoE 专属维度
- 唯一约束:$PP$ 必须一致,确保 gradient flow 正确

Figure 5 画得很清楚:传统约束下 EP 被困在 DP=8 的盒子里;Parallel Folding 让 EP 可以 "fold across" $TP \times CP$ group,从 EP=8 跳到 EP=64,而 attention 仍保持 TP=4, CP=2 的最优配置。

打个比方:传统方法像让篮球队和足球队穿同一套制服,篮球队嫌布料太多影响弹跳,足球队嫌布料太少保护不够。Parallel Folding 让它们各穿各的,但共享同一片场地(PP 一致)。

**好处巨大**:
1. 打破 $EP \le DP$ 约束,EP 可以超 DP
2. 降低最小 GPU 需求,CP 和 EP 共用同一组 GPU
3. Attention 用 high TP,MoE 用 ETP=1 + high EP,各自最优
4. **NVLink domain 保持**:CP 和 EP 的 all-to-all 都留在 NVLink group 内,避免跨 node

第 4 点对后面 case study 是决定性的——GB200 NVL72 让 EP=64 全在 NVLink domain,H100 NVL8 让 EP=64 必然跨 8 个 node,这差异直接决定两个平台用完全不同的优化栈。

---

## 5. Memory 优化的层次:从零开销到 trade-off

Paper 把 199.5 GB per-GPU 降到能跑,用了四个层次策略,按 overhead 从低到高排:

### 5.1 Zero-Overhead:Memory-Efficient Permutation

这是 paper 里最 elegant 的 trick,纯代数变换零开销省 26.3 GB。

Standard formulation:

$$\mathbf{y} = \sum_{i \in \mathcal{T}(\mathbf{x})} p_i \cdot \mathbf{W}_2^{(i)} \phi(\mathbf{W}_1^{(i)} \mathbf{x})$$

其中 $\mathcal{T}(\mathbf{x})$ 是 token $\mathbf{x}$ 选中的 expert 集合,$p_i$ 是 routing weight,$\mathbf{W}_1^{(i)}, \mathbf{W}_2^{(i)}$ 是 expert $i$ 的两层 MLP weight,$\phi$ 是 SwiGLU activation。

Memory-Efficient Permutation 把 $p_i$ 吸收进 activation:

$$\mathbf{y} = \sum_{i \in \mathcal{T}(\mathbf{x})} \mathbf{W}_2^{(i)} \left( p_i \cdot \phi(\mathbf{W}_1^{(i)} \mathbf{x}) \right)$$

为什么数学等价?Expert 没 bias term 时 $\mathbf{W}_2^{(i)}$ 是纯 linear map,scalar 乘法可交换:$p_i \cdot \mathbf{W}_2^{(i)} \mathbf{h} = \mathbf{W}_2^{(i)} (p_i \cdot \mathbf{h})$。

省 memory 的原理:standard 形式下,要算 $\partial \mathcal{L}/\partial p_i$,backward 必须保留每个 expert 的 output $E_i(\mathbf{x})$。Memory-efficient 形式下,$p_i$ 直接乘在 $\phi(\mathbf{z}_i)$ 上(其中 $\mathbf{z}_i = \mathbf{W}_1^{(i)} \mathbf{x}$),$\partial \mathcal{L}/\partial p_i$ 只依赖 $\phi(\mathbf{z}_i)$,而 $\phi(\mathbf{z}_i)$ 可以由 fused backward kernel 从 $\mathbf{z}_i$ 实时 recompute。$\mathbf{z}_i$ 反正要为 SwiGLU backward 保留,不引入额外 buffer。

这种 "algebraic rearrangement 省 storage" 的思路非常 system 设计——找到 computation 跟 memory 的解耦点。

### 5.2 Precision Trade-off:FP8/FP4 Activation

Linear layer 的 input 必须保留用于 backward 算 weight gradient。存 FP8/FP4 而非 BF16:
- FP8:省 50%
- FP4:省 75%

DeepSeek-V3 上 FP8 省约 16 GB activation memory per GPU,占 131 GB activation budget 的 12%。只针对 linear layer input;attention score, normalization intermediate, routing tensor 这些数值敏感的不存 FP8。

### 5.3 Compute Trade-off:Fine-Grained Recomputation

全 layer recomputation 对 MoE 是灾难:recompute expert 计算会 re-trigger EP all-to-all。Megatron-Core 用 granular recompute,用户精确指定哪个 op recompute。

Table 4 给 DeepSeek-V3 的具体数字:

| Recomputation Target | Memory Saved per GPU |
|---|---|
| MLA Up-Projection | 30.4 GB |
| SwiGLU Activation | 3.8 GB |
| LayerNorm | 8.2 GB |
| Total | 42.4 GB |

MLA Up-Projection 一个就省 30.4 GB,因为输出维度大但 recompute 便宜。这种 "memory-to-compute ratio 极高" 的 op 是 recompute 首选。

还有 **Output-discarding recomputation**:常规 checkpointing 把 checkpointed module 的 output 传给下游 layer 并存起来,但这个 output 反正 backward 要 recompute,存它冗余。Megatron-Core 在 downstream 消费完后立即 release。

### 5.4 Bandwidth Trade-off:Fine-Grained Activation Offloading

当 recompute + FP8 还不够,offload 到 CPU memory,trade PCIe 带宽换 GPU memory。

**核心 trick:stream overlap。** GPU 的 Copy Engine 和 Compute Engine 独立。当一个 module 的 compute time > activation transfer time 时,D2H copy 可以跟下一个 module 的 compute 并发跑,zero cost。

Forward:input activation 在 module 计算完立刻 offload,跟下一个 module compute 并发。例外:最后一层不 offload,因为 backward 立刻要用,没 compute 可藏。

Backward 用 **Layer-Staggered Reload**:算当前 layer gradient 时,从 CPU reload 下一 layer activation。任何时刻每 module type 只有一个 activation 驻留 GPU,避免 double storage。

**Peak memory 优势 vs full recompute**:
- Full recompute:peak = $L \times \text{layer\_input} + 1 \times \text{layer\_intermediate}$
- Offloading:peak = $1 \times \text{layer\_input} + 1 \times \text{layer\_intermediate}$,跟 model depth 解耦

对 60+ layer 的 DeepSeek-V3 这是 fundamental 优势。Table 5:
- DeepSeek-V3 full:169 → 151 GB(–10.7%),throughput 945 → 930 TF/s(–1.6%)
- Qwen3-235B(TP2→TP1 + EP16→EP64):172 → 175 GB(+1.7%),800 → 920 TF/s(+15.0%)

Qwen3 的例子特别妙:offload 省下的 memory headroom 让你能降 TP degree、提 EP degree,反而提 throughput 15%。

### 5.5 Optimizer 优化

**Precision-Aware Optimizer**:Adam 的 first/second moment 传统存 FP32(8 bytes/param)。Insight:optimizer state 对 storage precision 容忍度高,只要 update computation 用高精度。把 moment 存 BF16(2 bytes),storage 从 8 → 4 bytes/param。Update 时在 `FusedAdam` kernel 内部动态 cast 到 FP32 计算。

**State Offloading**:forward/backward 期间 optimizer state 不用,offload 到 CPU;optimizer step 前 load 回。GB200 上 NVLink-C2C 带宽高,async transfer 跟 compute overlap。DeepSeek-V3 省 15–20 GB,overhead 仅 0.1–0.2 s/iter。

---

## 6. Communication 优化:从加速到 overlap

### 6.1 为什么 standard NCCL all-to-all 不够

Standard all-to-all 之前要 permutation,把每个 token 复制 top-$K$ 次,产生冗余 traffic。

### 6.2 DeepEP & HybridEP:Token-Based Dispatch

**Token-based dispatch**(DeepEP 首创,HybridEP 跟进)消除 permutation,不发送 redundant token,降 volume 提 effective bandwidth。

**HybridEP dispatch kernel**(Figure 14):
- 从 global memory 读 token 到 shared memory,按 routing info 分发
- 通过 FIFO queue 写到 destination
- **Inter-node 优化**:先用 RDMA warp group 在同 local index 的 GPU 间跨 node 交换,再在每个 node 内 forward。减 cross-node traffic,inter-node 和 intra-node transfer 还能 overlap

**Combine kernel**(Figure 15):把 reduction fuse 进 communication kernel,从 FIFO 读数据,做 reduction,直接写 target。Inter-node 时先跨 node reduce,再 node 内 reduce。

Table 7 的 benchmark(hidden 7168, seq 4096, 256 experts):
- GB200 EP=8 dispatch:HybridEP 391 μs vs all-to-all 735 μs(1.9×)
- H100 EP=64 dispatch:HybridEP 4626 μs vs all-to-all 9164 μs(2×)
- H100 EP=16 combine:HybridEP 1485 μs vs all-to-all 5774 μs(3.9×!)

Inter-node scenario 上 HybridEP 优势更大。

### 6.3 EP Communication Overlap:1F1B FWD-BWD Merge

光提速不够,all-to-all 还在 critical path。Paper 用 DualPipe-like bidirectional schedule 建在 standard 1F1B 之上。

两种 merge pattern:
1. **Merged FWD-FWD / BWD-BWD**:两个 microbatch 同向 pass merge。代价 2× peak activation memory,forward compute 只占 backward 的一半,overlap 机会少
2. **Merged FWD-BWD**(preferred):一个 microbatch forward 跟另一个 backward merge。零额外 memory(forward activation 复用给 backward),跟 DualPipe 等价但避免复杂调度。限制:第一个 FWD 和最后一个 BWD 还在 critical path

为最大化 overlap:
- **Stream separation**:Compute Stream 跑 attention/expert MLP,Comm Stream 跑 all-to-all,两 stream 交替
- **W/D Split**:backward dispatch(B/dispatch)依赖 backward MLP(B/mlp),这阻塞 overlap。把 B/mlp 拆成 W/mlp(weight gradient,不依赖 B/dispatch)和 D/mlp(data gradient,feed 给 B/dispatch)。W/mlp 可以跟 F/mlp overlap 藏 B/dispatch

Figure 18 显示这个组合把 EP communication 占比从 30–40% 降到 <5%,overlap ratio 93%。

**诚实的 trade-off**:DeepEP 在 DeepSeek-V3 上用 20 SMs/GPU,引入约 20% GEMM efficiency overhead。Reserved 给通信的 SM 抢走了 GEMM 的 SM,overlap 不是 free lunch。

---

## 7. CUDA Graphs 跟 dynamic shape 的战争

这是 paper 工程 depth 的极致。先说为什么需要 CUDA Graphs。

### 7.1 Host Overhead 是 MoE 的隐形杀手

MoE 因为 sparsity + routing,同样 FLOP 要 launch 更多 kernel。每个 kernel launch 有几微秒 CPU-side cost。Python interpreter overhead, framework overhead, kernel launch overhead 三层叠加。GPU 计算越快,越没有时间 overlap CPU 工作,host overhead 越暴露。Figure 22 显示 traditional execution 的 GPU timeline 有大量 bubble,CPU launch 不够快。

CUDA Graphs 把 kernel 序列 capture 成可 replay 的 graph,后续 iteration 只 launch graph,绕过 per-op overhead。但要求 **static shape**,跟 dropless MoE 的 dynamic token count 冲突。

### 7.2 Partial CUDA Graphs:简单方案

Dropless MoE layer 里,static component 跟 dynamic component 分开:

**Static(can graph)**:
- Attention layer
- Router 计算
- EP preprocessing(permutation metadata)
- Shared expert
- Dense MLP

**Dynamic(不能 graph)**:
- Token dispatch
- Expert GEMM
- Token combine

Partial CUDA Graphs 把 static 部分 capture 成 per-layer graph,dynamic 部分正常执行。一个 layer 的 "attn+moe_router+moe_preprocess" scope 一图 capture 所有 static 部分。

**Pipeline Parallelism 的复杂度**:有 PP 时每个 microbatch 要独立 graph。原因:如果 microbatch share graph,mb+1 的 forward 会覆盖 mb 的 saved context(bwd 还要用),memory corruption。所以要 $L \times M \times 2$ 个 graph。无 PP 时 microbatch 可 share graph,只用 $L \times 2$ 个,通过 `is_first_microbatch` GPU flag 控制 microbatch-specific 行为。

实测:DeepSeek-V3 GB200 上 10% end-to-end speedup,约 7 GB 额外 memory。

### 7.3 Full CUDA Graphs for Dropless MoE:三大技术组合

这是 paper 最 hard 的工程。

#### 7.3.1 Device-Initiated Kernels(sync-free)

传统 kernel host-initiated:host 要先从 device query per-expert token count,才能决定 GEMM shape 和 launch config,产生 host-device sync barrier。

Device-initiated 三个要求:
- kernel 从 GPU memory 读 shape info,自己决定 compute
- kernel 把"实际工作量"跟"static launch config"解耦
- kernel 跳过 padding 数据的无效计算

两个实现:
- **Device-Initiated Grouped GEMM**:cuBLASLt Grouped GEMM(CUDA 13.1+)支持把 matrix shape 作 device array 传入;cuteDSL 实现把 SwiGLU 和 FP8 quantization fuse 进 epilogue
- **Sync-Free Dispatch with HybridEP**:给 HybridEP 一个 upper bound,dispatcher 预分配 output buffer,消除所有 sync,代价是额外 GPU memory

#### 7.3.2 ECHO:Elastic Cloning for Hot Experts

Load imbalance 是 MoE inherent 问题:hot expert 收到远超平均的 token。两个后果:
- Hot expert 所在 EP rank 是 compute bottleneck
- Worst-case buffer provisioning 浪费严重

ECHO 动态 clone hot expert 到 underutilized rank 的 spare slot。打个比方:餐厅高峰期把热门菜谱复制几份放到空闲灶台,平衡负载。

Workflow(Figure 27):
- Forward:ECHO planner 生成 hot expert map(哪些 expert clone 到哪些 spare slot)和 updated routing map(overflow token 重定向到 clone)。Expert Dispatch 把 hot expert weight copy 到 spare slot;Token Dispatch 路由 token 到 home + cloned expert;expert 计算在所有 expert 上进行
- Backward:Expert Gradient Dispatch 从 clone 收集 gradient reduce 回 home expert,保证一致性。Clone 计算完即 discard

Planner 用 bin-packing:算每个 expert 的 spillover(超过 EP rank 平均 load 的部分)+ 每个 rank 的 spare capacity,匹配 spillover 到 spare,用最少 clone 数达成 load balance。

两个 key benefit:减 memory fragmentation for CUDA Graphs(worst-case buffer 缩小),改善 compute efficiency(平衡负载减 straggler)。

#### 7.3.3 Paged Stashing:Fine-Grained Memory Management

即便 ECHO 降 load variance,worst-case buffer 还是大。Baseline CUDA Graph 给每层独立分配 worst-case buffer,总 memory = $\mathcal{O}(\text{layers} \times \text{worst\_case})$,严重 fragmentation。

Paged Stashing 观察:actual 需要的 activation memory 跟 worst-case 之间往往有一个数量级以上的 gap。打个比方:酒店最坏情况预订整个酒店太浪费,实际只住几个房间,用 paging 按需分配。

解法:decouple 两个 buffer:
- 一个 **tmp buffer**,按 worst-case size 分配,**所有 layer 共享**,用于当层的 computation
- 一个 **paged stashing buffer**,按 page 组织(默认 64 tokens/page),只存每层实际用的 activation

Forward 完一层后,activation 从 tmp buffer copy 到 stashing buffer 的 free page(只存 actual token count)。tmp buffer 立即给下一层复用。总 memory 从 $\mathcal{O}(\text{layers} \times \text{worst\_case})$ 降到 $\mathcal{O}(\text{worst\_case} + \text{actual\_total})$。

Paging 用 circular buffer 实现 free list。Stash 和 reload kernel 都是 device-initiated。stream overlap:stash 跟下一层 compute 并发,reload 在 bwd 当前层算完前 prefetch 下一层 activation,藏 reload latency。

**三者组合**:Device-initiated kernels 消除 host-device sync,ECHO 减 load variance 让 worst-case buffer 缩小,Paged Stashing 消除 layer 间 fragmentation。三者一起让 full CUDA Graphs 覆盖 dropless MoE。

---

## 8. FP8/FP4:跨 Three Walls 的 cross-cutting 优化

Section 5 把 reduced-precision 当 cross-cutting optimization 单独讲,因为它同时打三堵墙:

| Wall | Benefit |
|---|---|
| Memory | 50%(FP8）/75%（FP4）activation reduction + 消除 BF16 weight copy |
| Communication | 50% parameter AllGather |
| Compute | Faster Tensor Core GEMM（但有 quantization kernel overhead） |

### 8.1 Selective Precision 策略

MoE 放大了低精度的 benefit 和 risk:
- **Benefit 放大**:expert 数多,activation memory 同步放大,FP8/FP4 给更大绝对节省;expert GEMM 占 MoE compute 大头,FP8/FP4 加速明显
- **Risk 放大**:Router 的 token-expert assignment 极度依赖精确 score,quantization noise 可能 destabilize expert selection,引发 expert collapse

策略:**precision where it matters, efficiency everywhere else**。
1. Router 保持 FP32
2. Embedding, output layer, main gradients, master weights, optimizer states 保持原精度
3. Expert GEMM（占 computation 大头）用 reduced precision

### 8.2 FP8/FP4 Recipes 演进

| Recipe | 平台 | Granularity | Format |
|---|---|---|---|
| Per-Tensor FP8 | Hopper/Blackwell | 1 scale per tensor | E4M3(fwd) + E5M2(bwd) |
| Blockwise FP8 | Hopper | 1×128 activation, 128×128 weight | E4M3 |
| MXFP8 | Blackwell | 1×32, E8M0 scale | E4M3 |
| NVFP4 | Blackwell | 16 elements/block, two-level scale | E2M1 |

**NVFP4 最复杂**:FP4 E2M1 + 两级 microscaling:
- Per-tensor FP32 scale:把 tensor 分布 remap 到 block scaling 兼容 range
- Per-block 8-bit E4M3 scale:把每个 block(16 elements)map 到 FP4 range

三个 algorithmic trick 保稳定:
- **Random Hadamard Transform (RHT)**:用在 weight gradient 计算,减 outlier 影响
- **2D scaling**:16×16 weight block scaling(保留 FP32 tensor scale),让 weight 的 fwd/bwd quantization 更一致
- **Stochastic rounding**:用在 gradient 的 FP4 conversion,减 rounding bias

### 8.3 FP8/FP4 Primary Weights

传统 reduced-precision 训练维护三层 param hierarchy:FP32 master + BF16 model + FP8/FP4 compute。BF16 这层冗余。

Native FP8/FP4 直接从 FP32 master cast 到 FP8/FP4,绕过 BF16,省 memory + 加速 parameter AllGather(通信量减半)。

Quantization 流程(Per-tensor current scaling):
1. 算 master weight 的 local abs-max(无 sharded part 置 0)
2. AllReduce 算 global abs-max
3. 用 global abs-max + master weight 做 partial cast

### 8.4 MoE-Specific 量化挑战

**Padding Fusion**:FP8/FP4 GEMM 要求维度对齐(16 或 32)。Forward 时 hidden dim $K$ 已对齐,但 weight-gradient GEMM 的 dot-product dim 是 token dim $M$,动态变化,要 zero-padding。优化:padding routing map 而非 received token,或 fusing padding into permutation。

**Grouped Quantization**:Naive 一个 expert 一个 quantization kernel,CPU 开销大。Grouped quantization 把多个 expert 的 quantization fuse 进一个 kernel,降 CPU 开销 + CUDA Graph 兼容。

**NVFP4 Quantization Fusion**:NVFP4 量化 kernel 要吸收 RHT + 2D scaling + stochastic rounding。RHT fusion 最 latency-sensitive,单独 kernel 要在 global memory 多一次 BF16 read/write。Fuse 后 Hadamard + FP4 量化在单 kernel 内。Blackwell NVFP4 Tensor Core 是 TN-oriented,Wgrad 用 transposed activation,所以 Wgrad 路径的 RHT fusion kernel 还要吸收 transpose。

Fused kernel 多输出:标准 FP4 quantization(forward GEMM)+ Transpose + RHT + FP4 quantization(backward Wgrad)。Forward 时 launch 一次产生两份 FP4 copy,一份立即给 forward,一份存给 backward。原始高精度 input 直接 discard,避免存 BF16 activation。

---

## 9. Long Context:重心从 MoE 转到 Attention

Section 6 讨论序列长度从 4K/8K 拉到 16K/64K+ 时 optimization landscape 的根本 shift。

### 9.1 计算重心迁移

关键 insight:MoE 的 MLP component 随 sequence length 线性 scale $\mathcal{O}(s)$,但 attention 的 SDPA 是 $\mathcal{O}(s^2)$。64K tokens 时 SDPA 吃 69% FLOPs,短序列只占 10–15%。

Attention 变 dominant,但 FlashAttention/cuDNN 高度优化(Table 9:DeepSeek-V3 SDPA 在 Blackwell 上 16K seq fwd 1698 TF),所以 SDPA 本身不成为 bottleneck。优化焦点转到 memory 和 communication。

### 9.2 Memory Wall 加剧

**CP + TP**:让 sub-sequence length(每 CP/TP shard)近似 constant(4096 或 8192)。Scale $CP \times TP$ 跟 sequence length 同步,让 per-device memory 近 baseline。这把长序列 workload 变得像短序列,除 SDPA 和 TP/CP 专属通信,大部分短序列优化依然适用。

**Optimizer CPU Offloading**:省数十 GB。DeepSeek-V3 在 256 H100 上 50% MFU + ≥16K seq,最坏 overhead ~2%。

**Selective Recomputation**:SDPA 占 compute 大头时 recompute SDPA 太贵。DeepSeek-V3 64K 时 SDPA 占 72% total compute;recompute 它增加 18% compute overhead,16% 性能损失,只省 9 GB memory。Recompute 非 SDPA component 反而省 89.8 GB,性能影响相当或更低。Paper 推荐 disable core attention recomputation,priority recompute 其他 module。

### 9.3 CP vs TP 的选择

- **P2P CP** 常用,通信跟 compute 自然 overlap
- **TP** 通常 node 内首选(通信快 + 省 param memory + 改 SDPA kernel 效率)
- 跨 node 时 **P2P CP** 更好(TP 通信开销增长,P2P overlap 仍有效)
- **All-to-all CP** 介于两者之间,通常跟 TP 组合用于 node 内

Megatron-Core 支持 **hierarchical CP**:node 内 all-to-all CP + TP,跨 node P2P CP。

### 9.4 Packed Sequences + Dynamic CP

变长序列场景(RL、SFT)传统要 pad 到 batch 最大长度,浪费严重。4K/8K/32K 混合要全 pad 到 32K,浪费 ~60%。

**Packed Sequence Support**:用 THD(Total tokens × Heads × Dimension)tensor 格式替代 SBHD。THD 把所有 sequence 沿 token dim concat,通过 cumulative sequence length tracking 标记每个 sequence 起止。Attention 用 cumulative seqlen 保证跨 sequence 不互相 attend。RL 场景省 40–60% memory + 1.5–2× throughput。

**Dynamic Context Parallelism (Dynamic-CP)**:即便 packed 总 token 数相等,attention workload 因 $\mathcal{O}(s^2)$ 仍 vary。Static CP 必须为最长 packed sample 选 CP size,短 sample 被迫用同一 CP size 浪费 cross-device bandwidth。

Dynamic-CP per-microbatch 选 effective CP size,跟 packing plan 联合优化。CP resize 很轻量:只改 token slice 怎么 partition + attention 用哪个 CP communication group,无需 parameter redistribution。

**关键 trick**:framework 初始化时预构造多个 CP group per rank,候选 cp_size 从 1 到 $dp \times cp$(power of two),运行时 scheduler 选 effective cp_size + 对应预建 cp_group,无需动态创建 communication group。

**Loss 计算**:packed THD 下不同 sample 贡献不同 valid token 数,loss per-token 算避免 padding bias:

$$\mathcal{L} = \frac{\sum_{t \in \mathcal{V}} \ell_t}{|\mathcal{V}|}$$

$\mathcal{V}$ 是 packed representation 中 valid(非 padding)token 集合,$\ell_t$ 是 token $t$ 的 loss。

**Solver 设计**:joint 决定 packing + per-microbatch CP size。Attention compute $\approx \mathcal{O}(S^2)$,activation memory $\approx \mathcal{O}(S)$,难以同时平衡。Dynamic-CP 在 workload-oriented 和 memory-oriented 决策间交替。

实测:Dynamic-CP 在多模态等序列长度高度不均场景给 35–60% end-to-end 性能提升。

---

## 10. Production Features 速览

- **Load Balancing**:Auxiliary loss、expert choice routing、auxiliary-loss-free balancing(learnable expert bias)
- **Token Dropping**:Dropless(默认,最大 expressiveness)vs Droppable(capacity limit,可预测 memory,支持 pad-to-max 解锁 CUDA Graphs)
- **Shared Experts**:处理所有 token,`--moe-shared-expert-overlap` 跟 all-to-all 并发藏 latency
- **Latent MoE**:插入 down-projection $W_\downarrow \in \mathbb{R}^{\ell \times d}$ 和 up-projection $W_\uparrow \in \mathbb{R}^{d \times \ell}$,压缩比 $\alpha = d/\ell$ 降 all-to-all volume 和 expert weight size。NVIDIA Nemotron-3 用 $\ell\text{-MoE}_{\text{acc}}$ 同 cost 更高 accuracy
- **Distributed Checkpoint**:`ShardedTensor` descriptor 编码 global shape/offset/sharding pattern,enable any-to-any parallelism reconfiguration
- **Flexible Asymmetric VPP**:允许 per-virtual-stage 不同数量和类型的 layer,DeepSeek-V3 PP=16 VPP=2 第一个 PP rank 把 embedding + 3 个 dense decoder 组合匹配 2 个 MoE layer 成本
- **Upcycling**:Dense → MoE,softmax-then-topK routing 保证 init 时 MoE output 等于 dense
- **Multi-Token Prediction (MTP)**:每位置预测多个连续未来 token,densify supervision signal
- **Muon Optimizer**:对整个 weight matrix 做 orthogonalization 改 conditioning,集成 split QKV layout + distributed optimizer + CPU offloading。MuonClip 在 cuDNN 硬件加速解决 attention explosion

---

## 11. DeepSeek-V3 Case Study:同模型不同硬件完全不同优化栈

Section 9.2 是 paper 最实用的部分。Table 17:

| Config | GB200 | H100 |
|---|---|---|
| Hardware | 256×GB200 | 1024×H100 |
| Parallelism (TP/PP/EP) | 1/4/64 | 2/8/64 |
| VPP | 4 | 4 |
| Precision | MXFP8 | FP8-Blockwise |
| Dispatcher | HybridEP | DeepEP |
| Recompute | mlp | mlp, mla_up-proj, moe_act, layernorm |
| CUDA Graphs | Enabled | — |
| EP all-to-all Overlap | — | Enabled |
| Performance | 1048 | 368 |

**为什么差异这么大?**

**Memory 容量差**:GB200 192 GB/GPU vs H100 80 GB。GB200 能用 TP1/PP4(PP 减半),降 pipeline bubble + 简化 workload balance。H100 必须 TP2/PP8。

**Memory wall 策略差异**:
- H100:FP8 释放的 memory budget 关键,因为它 enable EP communication overlap(需要额外 buffer)。Fine-grained recompute 全开
- GB200:NVL72 让 EP local,communication overlap 不关键,这预算 free 出来。NVLink-C2C 带宽高让 optimizer state offloading 高效,offload 释放大量 GPU memory 给 activation,所以 fine-grained recompute 只要 mlp

**Communication wall 策略差异**:
- H100 (NVL8):EP64 跨 8 node,cross-node all-to-all 没优化会吃 50% step time。DeepEP 降 overhead,但还不够,必须开 EP communication overlap 藏剩余 cross-node latency。Overlap 可行是因为 FP8 释放了足够 memory 给额外 buffer
- GB200 (NVL72):EP64 全在 NVLink domain。HybridEP 满带宽利用 1.8 TB/s 双向带宽,无需 communication overlap。Hardware topology 单独解决 communication wall,bottleneck 转到 compute efficiency

**Compute efficiency wall 策略差异**:
- 共同:FP8 加速 GEMM,但加速 GPU 暴露 CPU overhead
- GB200:NVL72 已消除 communication bottleneck,CPU overhead 变 dominant。Partial CUDA Graphs capture attention + router + MoE preprocessing 到 static graph。Kernel fusions 减 launch 数。CPU/NUMA binding 减 host-side memory access latency
- H100:kernel fusions,但 CUDA Graphs 没开(EP overlap 跟 CUDA Graphs 组合在 H100 上更复杂)

**Four insights generalize**:
1. **Platform characteristics drive strategy**:GB200 大 memory + 高 C2C → 更激进选择;H100 NVL8 → 必须 EP communication overlap
2. **Parallel Folding unlocks flexibility**:decouple attention TP 跟 expert EP,跟 flexible VPP 组合做 fine-grained workload balancing
3. **FP8 shifts bottlenecks**:加速 GEMM + 降 memory,但放大 CPU overhead 作 dominant bottleneck。CUDA Graphs + kernel fusions + CPU/NUMA binding 变 essential
4. **Iterative optimization**:cyclical。Memory 优化 free GPU memory → enable communication overlap → 暴露 compute efficiency bottleneck。某些优化 cross-cutting:FP8 同时降 memory 和 compute time 但增加 CPU overhead;CUDA Graphs 降 CPU overhead 但 consume 额外 memory。每次 change 后要 re-profile,identifying diminishing returns 时转到下一个 bottleneck

---

## 12. RL 后训练的独特挑战

Section 10 讲 RL post-training:

1. **Variable-length sequences**:RL 产生高度变长序列,最大 128K 甚至 1M,mean 是最大值的 1/2 到 1/4。长尾分布难以平衡 compute efficiency vs peak memory
2. **Memory offloading**:RL framework 通常 co-locate training + inference engine 在同 GPU,offload 一个 engine state 当另一个 active。要求两个 engine 都快速完整 release + restore memory footprint
3. **Online weight export**:inference engine 每 training step 后要更新参数,要求 training engine 快速 export weight 到 inference engine 能 load 的格式
4. **Training stability**:标准 RL 假设 sampled response 来自 current policy distribution。但 inference 和 training engine 用不同优化 kernel,即便参数相同 token probability 略有差异,引入 off-policy bias。MoE 上更严重:同序列 token 可能 route 到不同 expert,放大差异

**Megatron-Bridge** 解决 HF ↔ Megatron checkpoint 转换。veRL, Slime, NeMo RL 都用这个 pattern。

**RL-Specific 优化**:
- **Packed Sequence Support**:packed 去 padding + packing-aware dynamic batch size。还有 load-balancing strategy 显式考虑 transformer block 异构计算成本:attention $\mathcal{O}(L^2)$ vs FFN $\mathcal{O}(L)$,token count 平衡的 batch wall-clock 仍可能高度不均。算每 sample seq len² 的 mini-batch sum,按 "attention cost" metric 排序 microbatch,用 small-to-large-to-small serpentine 调度
- **Dynamic CP**:per-micro-batch 选 CP degree
- **CPU Optimizer Offloading**:forward/backward 期间 offload 到 host DRAM,只 optimizer step 时 load 回
- **FP16 Training Support**:某些 RL 训练 hyper-param 下 FP16 数值稳定性更好,提供 full FP16 path(loss scaling + mixed-precision optimizer kernel)
- **Router Replay**:capture inference 时 router 产生的 routing decision,后续 training 阶段 replay 改善 convergence consistency。Decouple routing variability 跟 weight update,稳定 optimization trajectory

---

## 13. 我的大局观

读完这篇 paper,我对 MoE training system 的 mental model:

**Insight 1:MoE 的 sparsity 是 system 设计的"原罪"**。所有 Three Walls 都从"总参数随 $E$ 增长但 per-token compute 只随 $K$ 增长"这个 mismatch 衍生。这导致 parallelism 不能再用 dense 的 virtuous cycle。MoE 必须有 EP 这个第五维度,必须有 Parallel Folding 把 attention 和 MoE 的 parallelism 解耦,必须有 communication overlap 把 all-to-all 藏到 compute 后面,必须有 Grouped GEMM 把 small GEMM batch 起来,必须有 CUDA Graphs + sync-free 把 host overhead 干掉。

**Insight 2:Three Walls 是耦合的,isolated 优化 suboptimal**。Paper 反复强调这点。DeepSeek-V3 在 GB200 vs H100 上的差异是最好例子:同一模型,GB200 用 HybridEP + CUDA Graphs + optimizer offload,H100 用 DeepEP + EP overlap + 激进 fine-grained recompute。硬件拓扑决定瓶颈,瓶颈决定优化栈,优化栈又有 cross-cutting effect 需要 iterative refine。

**Insight 3:algebraic rearrangement 是最 elegant 的优化**。Memory-Efficient Permutation 通过把 $p_i$ 移到 activation 内部,用纯代数变换零开销省 26 GB memory。这种 "computation 跟 memory 解耦" 的思路在 system 设计里 value 极高。

**Insight 4:dynamic shape 是 MoE 的 fundamental 难点**。Dropless MoE 的 dynamic token count 跟 CUDA Graphs 的 static shape 要求冲突,跟 host-device sync 冲突,跟 buffer allocation 冲突。ECHO + Paged Stashing + device-initiated kernel 三个技术组合才把 full CUDA Graphs 在 dropless MoE 上跑通。这是 paper 工程 depth 的极致体现。

**Insight 5:precision 是 selective 的,不是 uniform 的**。Router FP32,embedding/output/optimizer state 原精度,expert GEMM aggressive FP8/FP4。NVFP4 还要 RHT + 2D scaling + stochastic rounding 三个 trick 保稳定。Reduced precision 是精细 recipe 设计。

**Insight 6:topology 决定一切**。NVLink domain 大小(NVL8 vs NVL72)决定 EP 是否跨 node,跨 node 决定 communication 是否 dominant,communication 是否 dominant 决定要不要 overlap,要不要 overlap 决定 memory budget 分配,memory budget 分配决定 recompute 激进度,recompute 激进度决定 host overhead 暴露程度,host overhead 暴露程度决定要不要 CUDA Graphs。这是一条从 hardware topology 到 software stack 的因果链。

---

## Reference Web Links

- Megatron-LM 官方仓库:https://github.com/NVIDIA/Megatron-LM
- TransformerEngine:https://github.com/NVIDIA/TransformerEngine
- DeepEP:https://github.com/deepseek-ai/DeepEP
- DualPipe:https://github.com/deepseek-ai/DualPipe-with-HybridFlow
- Megatron-Bridge:https://github.com/NVIDIA/Megatron-Bridge
- NVIDIA NVFP4 pretraining paper:https://arxiv.org/abs/2507.12145
- DeepSeek-V3 technical report:https://github.com/deepseek-ai/DeepSeek-V3
- Qwen3 technical report:https://github.com/QwenLM/Qwen3
- Slime RL framework:https://github.com/THUDM/slime
- veRL:https://github.com/volcengine/verl
- ByteScale (long context training):https://arxiv.org/abs/2502.21231
- Tutel (MoE framework 对比):https://github.com/microsoft/tutel
- DeepSpeed-MoE:https://github.com/microsoft/DeepSpeed
- MegaBlocks (Grouped GEMM MoE):https://arxiv.org/abs/2211.15841
- PyTorch FSDP:https://pytorch.org/docs/stable/fsdp.html
- NVIDIA Developer Blog on Dynamic-CP:https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/
- Megatron-Core Memory Estimator Blog:https://developer.nvidia.cn/blog/explore-using-the-megatron-core-training-framework-to-improve-gpu-memory-efficiency-in-large-model-training/

希望这个"人话"版本能帮你 build 出 MoE training system 的 mental model。核心记住一点:**MoE 训练是个 hardware topology → bottleneck → optimization stack 的因果链,每一步优化都会 shift 压力到另一堵墙,所以必须 co-design 整个 stack**。如果某个技术细节想再深挖(比如 HybridEP kernel 内部结构、NVFP4 RHT 数学推导、1F1B W/D split 的具体依赖图、ECHO planner 的 bin-packing 算法),我可以再展开。

---

# Megatron-Core MoE: 系统级解读

这篇 NVIDIA 的技术报告让我非常兴奋,因为它把 MoE 训练这件事讲透到了 system co-design 的本质层面。我会从第一性原理出发,把这篇 paper 的核心 insight 和技术细节都拆开讲,目的就是 build 你对 MoE training system 的 intuition。

---

## 1. 核心矛盾:为什么 MoE 训练是个 systems nightmare

Dense transformer 的 scaling law 有一个"良性耦合":参数 $N$ 增加 → FLOPs/token 增加 ($\approx 6N$) → 需要更多 GPU 分摊 → 每 GPU 的 computation 也增加 → communication 占比下降 → MFU 保持稳定。这是一个 virtuous cycle。

MoE 把这个耦合直接打断了。Paper 里用 DeepSeek-V3 的数字说明:
- Total params: $N_{\text{total}} = 685B$ (随 expert 数 $E$ 增长)
- Active params per token: $N_{\text{active}} = 37B$ (随 top-$K$ 增长, $K \ll E$)
- Ratio: $18\times$ mismatch

公式上,per-token FLOPs 大致是 $6 N_{\text{active}}$ 而非 $6 N_{\text{total}}$。这意味着你为了 memory 不得不把模型分到很多 GPU 上,但每个 token 的计算量没有同步增长,communication 占比就上来了。Paper 里图 3 画得很清楚:dense 模型沿着 $\approx 2N$ 的参考线增长,MoE 模型远远掉到线下方。

这个 single root cause 衍生出 paper 里所谓的 **Three Walls**:

### 1.1 Memory Wall
所有 $E$ 个 expert 的 parameters, gradients, optimizer states 都得驻留在 memory 里,即便每个 token 只激活 $K$ 个。DeepSeek-V3 在 BF16 + Adam 下,Table 3 给出的 per-GPU 内存分解:
- Weights & Gradients: 36.4 GB
- Main Weights & Optimizer States: 32.1 GB
- Activations: 131.0 GB ← dominant
- Total: 199.5 GB

Activations 比 weights + optimizer states 加起来还大,这是 MoE 的一个反直觉点。原因是 top-$K$ routing 会复制 token,activation 随 $K$、hidden dim $h$、batch、sequence length 一起膨胀,还有 dynamic routing 带来的 load imbalance spike。

### 1.2 Communication Wall
EP (Expert Parallelism) 的 all-to-all dispatch/combine 是 MoE 独有的通信 pattern。per-GPU send volume 大致是:

$$T \cdot K \cdot h \cdot \frac{EP - 1}{EP}$$

其中 $T$ 是 local token count, $K$ 是 top-$K$, $h$ 是 hidden dimension, $EP$ 是 expert parallelism degree。注意这个 volume 随 $K$ 线性增长,而且 full dispatch+combine cycle 还要 ×2。

在 DeepSeek-V3 这种架构下,58 层 MoE,每层 2 个 all-to-all,forward 就 116 次,backward 再翻倍。当 EP 跨 node 时(比如 H100 上 NVL8, EP=64 要跨 8 个 node),inter-node 带宽比 NVLink 低 5–10×,paper 实测 unoptimized all-to-all 能吃掉 60% 训练时间。

### 1.3 Compute Efficiency Wall
Fine-grained MoE (DeepSeek-V3 有 256 个 expert) 产生大量 small GEMM。Paper 实测:
- Llama-3 405B (dense): GEMM 占 ~70% 执行时间
- DeepSeek-V3 (MoE): GEMM 占 <50%,剩下都是 routing, permutation, launch overhead 这些跟 tensor 数量(而非 FLOP 数量)scale 的操作

还有 host overhead:MoE 因为 sparsity + routing,同样的 FLOP 要 launch 更多 kernel,每个 kernel launch 有几微秒 CPU-side cost,GPU 在 kernel 之间就 idle 了。这叫 host-boundedness。

**关键 insight:Three Walls 是 tightly coupled 的。** Paper 里举了个 trajectory:你开 activation recomputation 降 memory → communication 暴露 → 开 communication overlap → 但 fine-grained expert 太快没东西 overlap → 开 FP8 降 memory 换更大 batch → FP8 quantization kernel 增加 CPU overhead → host-bound → 开 CUDA Graphs → 但 CUDA Graphs 要 static shape,跟 dropless routing 的 dynamic shape 冲突 → 要 sync-free kernel + ECHO + Paged Stashing 才能解决。这是一个连锁反应系统,isolated 优化必然 suboptimal。

---

## 2. MoE Layer 架构:Route, Dispatch, Compute, Combine

Paper Section 2 把 MoE layer 拆成三个 modular component,对应 forward 的四个 stage。这个 abstraction 很关键,因为后面所有优化都是针对某一个 component 替换 backend 而不破坏其他部分。

### 2.1 Router: Token-to-Expert Assignment

Router 是一个 learned linear projection:

$$\mathbf{p}(\mathbf{x}) = \text{Softmax}(\mathbf{W}_r \mathbf{x})$$

其中 $\mathbf{W}_r \in \mathbb{R}^{h \times E}$ 是 router weight, $\mathbf{x} \in \mathbb{R}^h$ 是 token hidden state, $\mathbf{p} \in \mathbb{R}^E$ 是每个 expert 的 probability。Top-$K$ selection 选出 $K$ 个最高分的 expert。MoE layer 输出:

$$\text{MoE}(\mathbf{x}) = \sum_{i \in \text{TopK}(\mathbf{p}(\mathbf{x}))} p_i(\mathbf{x}) \cdot E_i(\mathbf{x})$$

$p_i$ 是 routing weight, $E_i$ 是第 $i$ 个 expert 的 MLP。Paper 在这里点了一个 numerical stability 的小细节:当 $E$ 很大时,router 可以在 FP32 下跑 (`--moe-router-dtype fp32`),避免 softmax 在大量 expert 下数值爆炸。这跟后面 FP8 training "selective precision" 的策略呼应:router 永远是高精度。

### 2.2 Token Dispatcher:通信抽象

Dispatcher 是 6-phase pipeline:
- Forward: `dispatch_preprocess → token_dispatch → dispatch_postprocess`
- Backward: `combine_preprocess → token_combine → combine_postprocess`

三种 backend:
1. **AllGather**: 每个 GPU gather 所有 token,filter 本地 expert。简单但 memory-intensive,适合小 EP。
2. **All-to-all**: 标准 NCCL point-to-point,每个 GPU 只发目标 GPU 需要的 token。scales 好但有 sync overhead。
3. **Flex**: 统一设计,支持 DeepEP 和 HybridEP 两个 optimized backend。这是 paper 的重头戏之一,Section 4.2.2 讲。

### 2.3 Experts: Grouped GEMM

每个 expert 是一个两层的 MLP,可选 SwiGLU/GeGLU gating。关键 optimization 是 **Grouped GEMM**:把所有 local expert 的计算 batch 成一个 kernel call (`TEGroupedMLP`),避免 256 个 expert 串行 launch。SequentialMLP 只用于 debug。

---

## 3. Parallel Folding:这篇 paper 最 elegant 的 contribution

这是我最喜欢的部分。Section 3 把 dense-sparse mismatch 这个根本矛盾讲透了。

### 3.1 问题本质

一个 transformer block 里同时有两种 computation pattern:
- **Attention (dense)**: 每个 token 跟所有其他 token attend,需要大 QKV matrix,benefits from high TP,benefits from high CP for long sequence。
- **MoE (sparse)**: 每个 token route 到 $K$ 个 expert,expert 的 hidden dim 很小,high TP 会把已经小的 shard 切得更碎;CP 对它完全 irrelevant。

Table 2 很直观:
| Aspect | Attention | MoE |
|---|---|---|
| TP | Large QKV matrices benefit | Small per-expert dim, high TP 反而 counterproductive |
| CP | Long sequence benefit | No sequence dependency |
| EP | N/A | Essential |

传统 framework 把 EP 当作 DP 的子维度:

$$\text{World Size} = TP \times CP \times PP \times DP, \quad EP \subseteq DP$$

这导致三个 challenge:
1. **GPU 需求乘性膨胀**:想要 EP=8 + CP=8,传统下需要 $1 \times 8 \times 1 \times 8 = 64$ GPU,即便 attention 和 MoE 理论上能共用 8 个 GPU。
2. **Forced suboptimal**:attention 和 MoE 共享 TP,要么选 high TP 切碎 expert,要么选 low TP 让 attention under-parallelized。
3. **Cross-node communication**:EP 被锁在 DP 里,高 EP 必然跨 node,带宽掉一个数量级。

### 3.2 Parallel Folding 的解法

核心 idea 极简:**不要强迫 attention 和 MoE 共享 parallelism 配置**。

- Attention layers form groups over $TP \times CP \times DP \times PP$,优化 sequence-level dense computation。
- MoE layers form groups over $ETP \times EP \times EDP \times PP$,其中 $ETP$ (Expert Tensor Parallelism) 和 $EDP$ (Expert Data Parallelism) 是 MoE 专属维度。
- 唯一约束:$PP$ 必须在两个 layout 间一致,确保 gradient flow 正确。

Figure 5 画得很清楚:传统约束下 EP 被困在 DP=8 的盒子里;Parallel Folding 让 EP 可以 "fold across" $TP \times CP$ group,从 EP=8 跳到 EP=64,而 attention 仍保持 TP=4, CP=2 的最优配置。

**好处:**
1. 打破 $EP \le DP$ 约束:EP 可以超过 DP。
2. 降低最小 GPU 需求:CP=8 和 EP=8 可以 share 同一组 8 GPU。
3. 独立优化:attention 用 high TP,MoE 用 ETP=1 + high EP。
4. **NVLink domain 保持**:CP 和 EP 的 all-to-all 都能留在 NVLink-connected GPU group 内,避免跨 node 通信。这一点对 H100 vs GB200 的策略差异(后面 case study 会讲)是决定性的。

### 3.3 Process Group 的实现细节

Paper Section 2.2.1 给了一个 `ProcessGroupCollection` 的伪代码:

```
ProcessGroupCollection
├── Attention Layer Groups: tp, cp, dp, pp
└── Expert Layer Groups: ep, expt_tp, expt_dp, pp
```

每个 MoE component 用哪个 group 是有讲究的(Table 1):
- Router: tp, cp, tp_cp (weight 在 EP rank 间 duplicated)
- Token Dispatcher: ep, tp_ep (all-to-all across expert ranks)
- Experts: ep, expt_tp, expt_dp (sharded across EP, gradients 在 EDP 里 reduce)
- Shared Experts: tp (跟 dense MLP 一样)

Optimizer 处理也特殊:用 `Chained-Optimizer` 包两套 optimizer,dense 和 expert 分开。三个 key design:
1. Expert param 标记 `allreduce=False`,跟 dense param 的 standard DP gradient reduce 区分。
2. Dense layer 在 `dp_cp_group` reduce,expert 在 `expt_dp_group` reduce。
3. Expert gradient scale by `edp_size / dp_size`,补偿 expert 看到的 effective batch size 跟 dense layer 不同。

这个 gradient scaling 的细节很容易被忽略,但它是正确性的关键:expert 处理的是 routing-dependent 的 token subset,不是完整 batch,所以 gradient 的"平均"基准跟 dense layer 不一样。

### 3.4 FSDP for MoE

Parallel Folding 的基础上,Paper 还集成了 Megatron-FSDP(定制版 FSDP)。核心设计是 **dual DeviceMesh**:
- Primary DeviceMesh: 管 $DP_{\text{Shard}} \times DP_{\text{Outer}} \times TP \times CP$,给 dense module 用。
- Auxiliary Expert DeviceMesh: 管 EP module,FSDP sharding scope 限定在 $EDP$ 维度,而不是全局 DP。

这样 expert param 的 `AllGather` / `ReduceScatter` 只在小 EDP group 里发生,collective volume 跟 EDP size scale,而不是 total DP size。

两个 zero-copy 优化:
1. **Non-uniform sharding**:标准 FSDP2 按 parameter 独立 shard,产生均匀 per-parameter shard;Megatron-FSDP 把 module 内所有 param flatten + concatenate,然后 non-uniform shard,shard boundary 跟 communication buffer layout 对齐,collective 直接从 flat storage 读,无 redundant copy。Llama3 405B 上减 ~10% communication overhead。
2. **Persistent double buffer + NCCL User Buffer Registration (UBR)**:预分配两个 persistent buffer 轮流用,避免 alloc churn;再 register 到 NCCL,NCCL 直接读写 pre-registered memory,无 intermediate copy。SM 占用从 8–32 SMs 降到 1–4 SMs;在 SHARP-enabled InfiniBand 上甚至完全 free GPU SM。

---

## 4. Breaking the Memory Wall

Paper Section 4.1 把 199.5 GB per-GPU 的 DeepSeek-V3 footprint 降到能跑的配置,用了四个互补策略。

### 4.1 Memory-Efficient Permutation: Zero-Overhead

这是 paper 里最 elegant 的 trick。Standard formulation:

$$\mathbf{y} = \sum_{i \in \mathcal{T}(\mathbf{x})} p_i \cdot \mathbf{W}_2^{(i)} \phi(\mathbf{W}_1^{(i)} \mathbf{x}) \tag{1}$$

Memory-Efficient Permutation 把 $p_i$ 吸收进 activation,移到第二层 linear 之前:

$$\mathbf{y} = \sum_{i \in \mathcal{T}(\mathbf{x})} \mathbf{W}_2^{(i)} \left( p_i \cdot \phi(\mathbf{W}_1^{(i)} \mathbf{x}) \right) \tag{2}$$

为什么数学等价?当 expert 没有 bias term 时,$\mathbf{W}_2^{(i)}$ 是纯 linear map,scalar multiplication commutes:$p_i \cdot \mathbf{W}_2^{(i)} \mathbf{h} = \mathbf{W}_2^{(i)} (p_i \cdot \mathbf{h})$。

省 memory 的原理:在 standard 形式下,要算 $\partial \mathcal{L}/\partial p_i$,backward pass 必须保留每个 expert 的 output $E_i(\mathbf{x})$。而在 memory-efficient 形式下,$p_i$ 直接乘在 $\phi(\mathbf{z}_i)$ 上(其中 $\mathbf{z}_i = \mathbf{W}_1^{(i)} \mathbf{x}$),$\partial \mathcal{L}/\partial p_i$ 只依赖 $\phi(\mathbf{z}_i)$,而 $\phi(\mathbf{z}_i)$ 可以由 fused backward kernel 从 $\mathbf{z}_i$ 实时 recompute。$\mathbf{z}_i$ 反正要为 SwiGLU 的 backward 保留,所以不引入额外 buffer。

DeepSeek-V3 上省 ~26.3 GB activation memory per GPU,**零计算开销**。这种 "algebraic rearrangement 省 storage" 的思路非常 Karpathy 风格——找到 computation 跟 memory 的解耦点。

### 4.2 FP8/FP4 Activation

Linear layer 的 input 必须保留用于 backward 算 weight gradient。把这些 input tensor 存 FP8/FP4 而非 BF16:
- FP8: 省 50%
- FP4: 省 75%

DeepSeek-V3 上 FP8 省约 16 GB activation memory per GPU,占 131 GB activation budget 的 12%。这部分只针对 linear layer input;attention score, normalization intermediate, routing tensor 这些数值敏感的不存 FP8。

### 4.3 Fine-Grained Recomputation

全 layer recomputation 对 MoE 是灾难:recompute expert 计算会 re-trigger EP all-to-all communication。Megatron-Core 用 granular recompute:用户精确指定哪个 op recompute,比如只 recompute expert MLP 的 activation function,LayerNorm,MLA 的 up-projection。

Table 4 给 DeepSeek-V3 的具体数字:
| Recomputation Target | Memory Saved per GPU |
|---|---|
| MLA Up-Projection | 30.4 GB |
| SwiGLU Activation | 3.8 GB |
| LayerNorm | 8.2 GB |
| Total | 42.4 GB |

注意 MLA Up-Projection 一个就省 30.4 GB,因为 MLA 的 up-proj 输出维度很大,但 recompute 这个 op 很便宜(就一个 GEMM)。这种 "memory-to-compute ratio 极高" 的 op 是 recompute 的首选。

还有 **Output-discarding recomputation**:常规 activation checkpointing 把 checkpointed module 的 output 传给下游 layer 并存起来,但这个 output 反正 backward 要 recompute,存它是冗余。Megatron-Core 在 downstream layer 消费完后立即 release,bwd 时从 recompute 结果恢复。

### 4.4 Fine-Grained Activation Offloading

当 recompute + FP8 还不够,offload 到 CPU memory,trade PCIe 带宽换 GPU memory。挑战是把 transfer latency 藏起来。

**核心 trick:stream overlap。** GPU 有独立的 Copy Engine 和 Compute Engine。当一个 module 的 compute time > activation transfer time 时,D2H copy 可以跟下一个 module 的 compute 并发跑,zero cost。

Forward pass:input activation 在 module 计算完后立刻 offload 到 CPU,跟下一个 module 的 compute 并发。例外:最后一层不 offload,因为 bwd 立刻要用,没有 compute 可以 hide。

Backward pass 用 **Layer-Staggered Reload**:算当前 layer 的 gradient 时,从 CPU reload 下一 layer 的 activation。任何时刻每个 module type 只有一个 activation 驻留 GPU,避免 double storage。这一点对单个 module activation 极大的情况至关重要,否则 2× memory 会引发意外 peak。

PP/VPP 场景:`ChunkOffloadHandler` 管理 (microbatch, VPP stage) 组合,deque 用 VPP stage 反序 (FILO) + microbatch 正序 (FIFO),backward pop 时自动匹配 VPP chunk 执行顺序。

**Peak memory 优势 vs full recompute:**
- Full recompute:存每层 input,peak memory = $L \times \text{layer\_input} + 1 \times \text{layer\_intermediate}$
- Offloading:layer input 移到 CPU,backward 前刚 reload 完即 release,peak memory = $1 \times \text{layer\_input} + 1 \times \text{layer\_intermediate}$,跟 model depth 解耦

对于 60+ layer 的 DeepSeek-V3,这是 fundamental 优势。Table 5 数据:
- DeepSeek-V3 full:169 → 151 GB (–10.7%),throughput 945 → 930 TF/s (–1.6%)
- Qwen3-235B (TP2→TP1 + EP16→EP64):172 → 175 GB (+1.7%),800 → 920 TF/s (+15.0%)

Qwen3 那个例子特别有意思:offload 省下的 memory headroom 让你能降低 TP degree、提高 EP degree,反而提升 throughput 15%。

### 4.5 Precision-Aware Optimizer + State Offloading

Adam 的 first/second moment 传统存 FP32,8 bytes/param。Insight:optimizer state 对 storage precision 容忍度高,只要 update computation 用高精度即可。把 moment 存 BF16 (2 bytes) 或 FP8 (1 byte),storage 从 8 → 4 或 2 bytes/param。Update 时在 `FusedAdam` kernel 内部动态 cast 到 FP32 计算。

DeepSeek-V3 + distributed optimizer,memory per param per DP rank 从 $6 + 12/d$ bytes 降到 $6 + 8/d$ bytes。省约 10–12 GB(从 32.1 GB optimizer 预算)。

State offloading:forward/backward 期间 optimizer state 不用,offload 到 CPU;optimizer step 前再 load 回 GPU。GB200 上 NVLink-C2C 带宽高,async transfer 跟 compute overlap,pinned memory 满带宽。DeepSeek-V3 省 15–20 GB,overhead 仅 0.1–0.2 s/iter。

### 4.6 FSDP for MoE(前面 3.4 已讲)

---

## 5. Breaking the Communication Wall

### 5.1 为什么 standard NCCL all-to-all 不够

Figure 13 的 expert parallelism pattern:每个 MoE 层需要 2 个 collective(dispatch + combine)。Volume 是 $\mathcal{O}(TKh)$ per rank,跟 EP degree 没关系,但更大 EP 把通信推到 inter-node,带宽掉一个数量级。

### 5.2 DeepEP & HybridEP:Token-Based Dispatch

Standard all-to-all 之前需要一个 permutation 阶段,把每个 token 复制 top-$K$ 次,产生冗余 traffic。**Token-based dispatch**(DeepEP 首创,HybridEP 跟进)消除 permutation 步骤,不发送 redundant token,降 volume 提 effective bandwidth。

**HybridEP** 的 dispatch kernel 设计(Figure 14):
- 从 global memory 读 token 到 shared memory,按 routing info 分发
- 通过 FIFO queue 写到 destination
- **Inter-node 优化**:不直接通过 NIC 发冗余 payload,而是先用 RDMA warp group 在同 local index 的 GPU 间跨 node 交换,再在每个 node 内 forward。这样减 cross-node traffic,inter-node 和 intra-node transfer 还能 overlap

**Combine kernel**(Figure 15):standard all-to-all dispatch 只做通信,需要单独的 unpermute 阶段。HybridEP 把 reduction fuse 进 communication kernel:从 FIFO 读数据,做 reduction,直接写 target。Inter-node 时先跨 node reduce,再 node 内 reduce 一次完成。

Table 7 的 benchmark:DeepSeek-V3 配置(hidden 7168, seq 4096, 256 experts)。
- GB200 EP=8 dispatch:HybridEP 391 μs vs all-to-all 735 μs (1.9× 加速)
- GB200 EP=64 dispatch:HybridEP 675 μs vs all-to-all 930 μs (1.4×)
- H100 EP=64 dispatch:HybridEP 4626 μs vs all-to-all 9164 μs (2×)
- H100 EP=16 combine:HybridEP 1485 μs vs all-to-all 5774 μs (3.9×!)

Inter-node scenario 上 HybridEP 优势更大,因为它把 RDMA 跨 node 跟 node 内 forward 解耦并 overlap。

### 5.3 EP Communication Overlap:1F1B FWD-BWD Merge

光提速不够,all-to-all 还在 critical path 上。Paper 用一个 **DualPipe-like 的 bidirectional schedule** 建在 standard 1F1B 之上。

两种 merge pattern:
1. **Merged FWD-FWD / BWD-BWD**:两个 microbatch 的同向 pass merge。代价是 2× peak activation memory,而且 forward compute 只有 backward 的一半,overlap 机会少。
2. **Merged FWD-BWD**(preferred):一个 microbatch 的 forward 跟另一个的 backward merge。零额外 memory(forward activation 复用给 backward),跟 DualPipe 设计等价但避免复杂调度。限制:第一个 FWD 和最后一个 BWD 还在 critical path 上无法 hide。

为了最大化 overlap,两个优化:
- **Stream separation**:Compute Stream 跑 attention/expert MLP,Comm Stream 跑 all-to-all。两个 stream 交替,通信跟计算并行。
- **W/D Split**(Weight-Gradient / Data-Gradient Split):backward dispatch (B/dispatch) 依赖 backward MLP (B/mlp) 的输出,这阻塞了 overlap。把 B/mlp 拆成:
  - **W/mlp**:weight gradient 计算,不依赖 B/dispatch,可以跟 F/mlp overlap 藏 B/dispatch
  - **D/mlp**:data gradient 计算,feed 给 B/dispatch

Figure 18 显示这个组合把 EP communication 占比从 30–40% 降到 <5%,overlap ratio 93%。

**Interleaved PP 扩展**:VPP 把 model 切成多个 virtual stage,扩大 overlap 机会。一个 trick:1F1B 阶段相邻 FWD-BWD pair 如果属于同一 microbatch 会有 data dependency,所以 warmup 阶段额外跑一个 microbatch,保证 1F1B 阶段相邻 pair 是 dependency-free。

### 5.4 SM Carve-out 的代价

Paper 诚实地讲了一个 trade-off:DeepEP 在 DeepSeek-V3 上用 20 SMs/GPU,引入约 20% GEMM efficiency overhead。这意味着 reserved 给通信的 SM 抢走了 GEMM 的 SM,所以 overlap 的 speedup 不是免费的。

---

## 6. Breaking the Compute Efficiency Wall

### 6.1 Grouped GEMM

四种实现:
1. **Multi-stream cuBLASLt GEMMs**:多个 GEMM launch 到不同 CUDA stream,overlap wave tail。支持 BF16, per-tensor FP8, blockwise FP8, MXFP8, NVFP4。
2. **CUTLASS Grouped GEMM**:fuse 进单 kernel,expert 数多时更好,但精度/平台需要单独开发。当前 TE 里只支持 Hopper BF16。
3. **cuBLASLt Grouped GEMM via `CUBLASLT_BATCH_MODE_GROUPED`** (CUDA 13.1+):shape 信息放 device array,单 kernel 完成,built-in heuristics 选 kernel config,覆盖所有精度,unblock CUDA Graphs。
4. **cuteDSL Grouped GEMM with fusions**(Blackwell):fuse activation, quantization, scaling factor swizzling 进 GEMM epilogue,专门优化 MXFP8/NVFP4 的 fprop FC1 和 dgrad FC2。

### 6.2 Permutation Fusion

Grouped GEMM 要求同一 expert 的 token 在 memory 中 contiguous,需要 permutation。Native PyTorch 实现要 launch 很多小 kernel + CPU overhead。

Permute fusion 三阶段(Figure 20):
- **Preprocessing**:生成 offset map (Row ID map),只调一次。
- **Permute**:按 offset map 从 input buffer copy 到 output buffer。Memory-efficient permute 下还要 permute probability,直接进 expert activation。
- **Unpermute**:permute 的逆操作。同一 token 被 copy 多次到不同 expert,combine 时要 sum 回来。Memory-efficient 下直接 add;否则用 probability 作 weight。所有 accumulation 在 FP32。

### 6.3 Router & Aux-Loss Fusion

Router 里有 GEMM 和通信不好 fuse,剩下 op 拆成三个 fused kernel(Figure 21):
1. Score computation (top-$K$ + softmax/sigmoid,支持 group top-$K$、sigmoid/softmax 组合、scaling)
2. Score computation for auxiliary loss
3. Auxiliary loss computation

### 6.4 CUDA Graphs:消除 Host Overhead

CUDA Graphs 把 kernel 序列 capture 成可 replay 的 graph,后续 iteration 只 launch graph,绕过 per-op Python/framework overhead 和 per-kernel launch overhead。

但 CUDA Graphs 要求 **static shape**,跟 dropless MoE 的 dynamic token count 冲突。Paper 给两套方案:

#### 6.4.1 Partial CUDA Graphs(简单方案)

Dropless MoE layer 里,static component 跟 dynamic component 分开:
- **Static(can graph)**:Attention layer、Router 计算、EP preprocessing(permutation metadata)、Shared expert、dense MLP
- **Dynamic(不能 graph)**:Token dispatch、Expert GEMM、Token combine

Partial CUDA Graphs 把 static 部分 capture 成 per-layer graph,dynamic 部分正常执行。Figure 24 的 "attn+moe_router+moe_preprocess" scope 一图 capture 一个 layer 的所有 static 部分。

**Pipeline Parallelism 的复杂度**:有 PP 时每个 microbatch 要独立 graph。原因:如果 microbatch share graph,mb+1 的 forward 会覆盖 mb 的 saved context(bwd 还要用),memory corruption。所以要 $L \times M \times 2$ 个 graph($L$ = layers/GPU, $M$ = microbatch, ×2 for fwd/bwd)。无 PP 时 microbatch 可 share graph,只用 $L \times 2$ 个 graph,通过 `is_first_microbatch` GPU flag 控制 microbatch-specific 行为(如只在第一个 microbatch 跑 quantization)。

**Memory optimizations**:
- 减少 graph 数量(如上)
- Pool sharing:graph 和非 graph op 用独立 memory pool,但所有 graph 按执行顺序 capture 时可 share 一个 pool(`make_graphed_callables()` 的 `_order` 参数)
- Buffer reuse:static input/output buffer 按 PP 执行顺序复用,只在 bwd output buffer 要保留到下一 PP stage 时不能复用

实测:DeepSeek-V3 GB200 上 10% end-to-end speedup,约 7 GB 额外 memory。

#### 6.4.2 Full CUDA Graphs for Dropless MoE

这是 paper 最 hard 的工程。三大技术组合:

**1. Device-Initiated Kernels(sync-free)**

传统 kernel host-initiated:host 要先从 device query per-expert token count,才能决定 GEMM shape 和 launch config,产生 host-device sync barrier。

Device-initiated 三个要求:
- kernel 从 GPU memory 读 shape info,自己决定 compute
- kernel 把 "实际工作量"(运行时才知道)和 "static launch config" 解耦
- kernel 跳过 padding 数据的无效计算

两个具体实现:
- **Device-Initiated Grouped GEMM**:cuBLASLt Grouped GEMM(CUDA 13.1+)支持把 matrix shape 作 device array 传入;cuteDSL 实现把 SwiGLU 和 FP8 quantization fuse 进 epilogue。
- **Sync-Free Dispatch with HybridEP**:给 HybridEP 一个 upper bound,dispatcher 预分配 output buffer,消除所有 sync,代价是额外 GPU memory。

**2. ECHO (Elastic Cloning for Hot Experts)**

Load imbalance 是 MoE inherent 问题:hot expert 收到远超平均的 token。两个后果:
- Hot expert 所在 EP rank 是 compute bottleneck
- Worst-case buffer provisioning 浪费严重(如果所有 token 都到一个 expert)

ECHO 动态 clone hot expert 到 underutilized rank 的 spare slot。Workflow(Figure 27):
- Forward:ECHO planner 生成 hot expert map(哪些 expert clone 到哪些 spare slot)和 updated routing map(overflow token 重定向到 clone)。Expert Dispatch 把 hot expert weight copy 到 spare slot;Token Dispatch 路由 token 到 home + cloned expert;expert 计算在所有 expert 上进行
- Backward:Expert Gradient Dispatch 从 clone 收集 gradient reduce 回 home expert,保证一致性。Clone 计算完即 discard 省 memory

Planner 用 bin-packing:算每个 expert 的 spillover(超过 EP rank 平均 load 的部分)+ 每个 rank 的 spare capacity,匹配 spillover 到 spare,用最少 clone 数达成 load balance。

**3. Paged Stashing**

即便 ECHO 降 load variance,worst-case buffer 还是大。Baseline CUDA Graph 给每层独立分配 worst-case buffer,总 memory = $\mathcal{O}(\text{layers} \times \text{worst\_case})$,严重 fragmentation。

Paged Stashing 观察:actual 需要的 activation memory 跟 worst-case 之间往往有一个数量级以上的 gap。解法:decouple 两个 buffer:
- 一个 **tmp buffer**,按 worst-case size 分配,**所有 layer 共享**,用于当层的 computation
- 一个 **paged stashing buffer**,按 page 组织(默认 64 tokens/page),只存每层实际用的 activation

Forward 完一层后,activation 从 tmp buffer copy 到 stashing buffer 的 free page(只存 actual token count)。tmp buffer 立即给下一层复用。总 memory 从 $\mathcal{O}(\text{layers} \times \text{worst\_case})$ 降到 $\mathcal{O}(\text{worst\_case} + \text{actual\_total})$。

Paging 用 circular buffer 实现 free list。Stash 和 reload kernel 都是 device-initiated。stream overlap:stash 用 dedicated Pack stream,跟下一层 compute 并发;reload 用 Unpack stream,在 bwd 当前层算完前 prefetch 下一层 activation,藏 reload latency。double buffer 有轻微 memory overhead。

**三者组合**:Device-initiated kernels 消除 host-device sync,ECHO 减 load variance 让 worst-case buffer 缩小,Paged Stashing 消除 layer 间 fragmentation。三者一起让 full CUDA Graphs 覆盖 dropless MoE。

---

## 7. Reduced-Precision Training: FP8/FP4 跨 Three Walls

Section 5 把 reduced-precision 当 cross-cutting optimization 单独讲,因为它同时打三堵墙。

### 7.1 Selective Precision 策略

MoE 放大了低精度的 benefit 和 risk:
- **Benefit 放大**:expert 数多,activation memory 同步放大,FP8/FP4 给更大绝对节省;expert GEMM 占 MoE compute 大头,FP8/FP4 加速明显
- **Risk 放大**:Router 的 token-expert assignment 极度依赖精确 score,quantization noise 可能 destabilize expert selection,引发 expert collapse

策略:**precision where it matters, efficiency everywhere else**。
1. Router 保持 FP32
2. Embedding, output layer (LM head), main gradients, master weights, optimizer states 保持原精度
3. Expert GEMM(占 computation 大头)用 reduced precision

### 7.2 FP8 Recipes

Paper 提供 3 个 FP8 recipe + 1 个 FP4 recipe,演进反映了硬件代际更迭(Figure 30):

| Recipe | 平台 | Granularity | Format |
|---|---|---|---|
| Per-Tensor FP8 | Hopper/Blackwell | 1 scale per tensor | E4M3 (fwd) + E5M2 (bwd) |
| Blockwise FP8 | Hopper | 1×128 activation, 128×128 weight | E4M3 |
| MXFP8 | Blackwell | 1×32, E8M0 scale | E4M3 |
| NVFP4 | Blackwell | 16 elements/block, two-level scale | E2M1 |

**Per-Tensor FP8**:有两种 scaling:
- Delayed scaling:用历史窗口的 amax,断数据依赖,性能最好但精度差
- Current (live) scaling:JIT 算 amax,精度好。Paper 推荐 current scaling。

Hopper 上 FP8 GEMM 只支持 TN layout,必须存 transposed FP8 activation;bwd 用。Blackwell 上 FP8 GEMM 全 layout 支持,无需 transposed version,进一步省 memory。

**Blockwise FP8 (Hopper 推荐)**:activation 1×128 tile quantize,weight 128×128 block quantize,全 E4M3。DeepSeek-V3, Minimax-M2, Ant Ling-2.0 等大模型生产验证过。

**MXFP8 (Blackwell 默认)**:1×32 granularity,E8M0 scale factor。原生第五代 Tensor Core 支持,理论更精确(finer scaling)+ 更快(硬件 native)。

**NVFP4 (Blackwell)**:FP4 E2M1 + 两级 microscaling:
- Per-tensor FP32 scale:把 tensor 分布 remap 到 block scaling 兼容 range
- Per-block 8-bit E4M3 scale:把每个 block(16 elements)map 到 FP4 range

三个 algorithmic trick 保稳定:
- **Random Hadamard Transform (RHT)**:用在 weight gradient 计算,减 outlier 影响
- **2D scaling**:16×16 weight block scaling(保留 FP32 tensor scale),让 weight 的 fwd/bwd quantization 更一致
- **Stochastic rounding**:用在 gradient 的 FP4 conversion,减 rounding bias

### 7.3 FP8/FP4 Primary Weights

传统 reduced-precision 训练维护三层 param hierarchy:FP32 master + BF16 model + FP8/FP4 compute。BF16 这层是冗余的。

Native FP8/FP4 直接从 FP32 master cast 到 FP8/FP4,绕过 BF16 中间层,省 memory + 加速 parameter AllGather。

Quantization 流程(Per-tensor current scaling,Figure 33):
1. 算 master weight 的 local abs-max(无 sharded part 则置 0)
2. AllReduce 算 global abs-max
3. 用 global abs-max + master weight 做 partial cast

Blockwise recipe 因为 abs-max 在 2D block 上算,需要专门 kernel 感知 weight 2D layout 和 master-weight 到 weight 的对应关系(Figure 32)。

Parameter AllGather 在 FP8/FP4 下通信量减半(1 byte vs 2 bytes per param)。NVFP4 需要 gather row-wise + column-wise FP4 weight,所以也减半(2 byte vs 1 byte per param in BF16)。MXFP8 例外:需要先 copy FP32 master 到 BF16 temp buffer,以 BF16 通信(因为 MXFP8 fwd/bwd 量化方向不同,通信 MXFP8 weight 需要 row-wise + column-wise 两个版本,跟 BF16 通信量等同)。

### 7.4 MoE-Specific 量化挑战

#### 7.4.1 Padding Fusion

FP8/FP4 GEMM 要求维度对齐:per-tensor/blockwise 16,MXFP8/NVFP4 32(TMA 16-byte 对齐 + block scaling granularity)。Forward 时 hidden dim $K$ 已对齐,但 weight-gradient GEMM 的 dot-product dim 是 token dim $M$,动态变化,经常不对齐,需要 zero-padding。

为了 grouped quantization kernel 低 CPU 开销 + CUDA Graph 兼容,per-expert token padding 还要提高到 128(为了 NVFP4 grouped quantization kernel)。最终 token 维 padding 由所有参与 kernel 的要求联合决定。

两个优化:
- **Routing map padding**:padding routing map 而非 received token,只多发少量 token,避免 per-tensor padding 开销
- **Fusing padding into permutation**:避免一次 global memory read/write,默认选项

#### 7.4.2 Grouped Quantization

Naive 一个 expert 一个 quantization kernel,CPU 开销大。Grouped quantization kernel 把多个 expert 的 quantization fuse 进一个 kernel,降 CPU 开销 + CUDA Graph 兼容。

#### 7.4.3 NVFP4 Quantization Fusion(NVFP4 最复杂)

NVFP4 量化 kernel 不是简单 "scale + cast",要吸收 RHT + 2D scaling + stochastic rounding。

**RHT fusion 最 latency-sensitive**:单独 kernel 的话,Hadamard transform 要在 global memory 多一次 BF16 read/write,带宽代价大。Fuse 后 Hadamard + FP4 量化在单 kernel 内,避免 BF16 traffic。

**Blackwell NVFP4 Tensor Core 是 TN-oriented**,而 Wgrad 用 transposed activation 和 gradient。所以 Wgrad 路径的 RHT fusion kernel 还要吸收 transpose,不能依赖单独的 BF16 transpose kernel。

Fused kernel 多输出:
1. 标准 FP4 quantization(forward GEMM)
2. Transpose + RHT + FP4 quantization(backward Wgrad 路径)

Forward 时 launch 一次产生两份 FP4 copy,一份立即给 forward GEMM,一份存给 backward。原始高精度 input 直接 discard,避免存 BF16 activation。

**Per-tensor FP32 scale 问题**:NVFP4 recipe 里 tensor-wide amax 在 Hadamard transform 之后算,所以需要专门的 Hadamard-amax kernel 只算 amax 不 materialize BF16 output。Hadamard 实际算两次:Hadamard-amax kernel 一次 + fused quantization kernel 一次。但比写出完整 transformed BF16 tensor 还是更快(避免高带宽 BF16 read/write)。

**Grouped NVFP4 for MoE**:全 iteration CUDA Graph 要求下,MoE 量化路径必须 CUDA Graph safe:host 不能依赖 CPU 上的 dynamic expert token count,只能依赖 device 上的 tokens-per-expert tensor。所以 input activation 不能 split 单独量化,必须 grouped quantization 整个 NVFP4 pipeline,输出分配为一个 flat buffer,shape 不在中间 expose。

三个关键约束:
1. **128-token alignment per expert**:quantization thread block 不能跨 expert,引入 control-flow 开销,所以 per-expert token 数 zero-pad 到 thread block shape 在 token dim 的整数倍(通常 128)
2. **Per-expert transpose**:NVFP4 transpose 必须 per-expert 做(每个 expert 的 packed activation 独立 transpose,再 concat),不等价于 transpose 整个 grouped buffer
3. **Scale-factor swizzling**:NVFP4 GEMM 要求 scale factor pad 到 128×4 对齐 shape,swizzle 到 32×16 layout。per-expert GEMM 应用,意味着确定 padding size 隐含需要 CPU 看到 tokens-per-expert,非 CUDA Graph safe。所以 enforce 128-token per-expert alignment by construction

Per-tensor FP32 二级 scale 在 MoE 里意味着 per-expert 二级 scale(不共享 amax),这些 amax 必须在线从 routed token 算,每次 iteration 都要产生 distinct per-expert amax。利用 128-aligned tokens-per-expert guarantee,把 dense 实现改成 CUDA-Graph-safe grouped Hadamard-amax kernel。

---

## 8. Long-Context MoE Training

Section 6 讨论序列长度从 4K/8K 拉到 16K/64K+ 时 optimization landscape 的根本 shift。

### 8.1 计算重心的迁移

关键 insight:MoE 的 MLP component 随 sequence length 线性 scale ($\mathcal{O}(s)$),但 attention 的 SDPA 是 $\mathcal{O}(s^2)$(Figure 34)。64K tokens 时 SDPA 吃 69% FLOPs,而短序列场景只占 10–15%。

Attention 变 dominant,但 FlashAttention/cuDNN 高度优化(Table 9 显示 DeepSeek-V3 SDPA 在 Blackwell 上 16K seq fwd 1698 TF,bwd 1298 TF),所以 SDPA 本身不成为 bottleneck。优化焦点转到 memory 和 communication。

### 8.2 Memory Wall 加剧

Activation memory 随 sequence length 增长,要组合多种技术:

**CP + TP**:让 sub-sequence length(每 CP/TP shard 的 sequence length)近似 constant(4096 或 8192)。Scale $CP \times TP$ 跟 sequence length 同步,让 per-device memory 近 baseline。这把长序列 workload 变得像短序列,除 SDPA 和 TP/CP 专属通信,大部分短序列优化依然适用。

**Optimizer CPU Offloading**:省数十 GB,代价是 transfer + host-side optimizer overhead。DeepSeek-V3 在 256 H100 上 50% MFU + ≥16K seq,最坏 overhead ~2%。

**Selective Recomputation**:关键——SDPA 占 compute 大头时 recompute SDPA 太贵。DeepSeek-V3 64K 时 SDPA 占 72% total compute;recompute 它增加 18% compute overhead,16% 性能损失,只省 9 GB memory。Recompute 非 SDPA component 反而省 89.8 GB,性能影响相当或更低。Paper 推荐 **disable core attention recomputation,priority recompute 其他 module**。

### 8.3 CP vs TP 的选择

CP 和 TP 都降 activation memory,但 communication pattern 不同(Figure 35):
- **CP**:partition activation 沿 sequence dim,activation memory 降 $CP$ 倍。两种 SDPA 模式:P2P(ring-style 交换 KV cache,跟 SDPA compute overlap)和 all-to-all(transform tensor 从 sequence-sharded 到 head-sharded)
- **TP**:额外 shard linear weight,降 param memory,但 linear layer 引入额外 collective

实践:
- **P2P CP** 常用,通信跟 compute 自然 overlap
- **TP** 通常 node 内首选(通信快 + 省 param memory + 改 SDPA kernel 效率)
- 跨 node 时 **P2P CP** 更好(TP 通信开销增长,P2P overlap 仍有效)
- **All-to-all CP** 介于两者之间,通常跟 TP 组合用于 node 内

Megatron-Core 支持 **hierarchical CP**,practical 起始配置:node 内 all-to-all CP + TP,跨 node P2P CP。

### 8.4 Packed Sequences + Dynamic CP

变长序列场景(RL、SFT)传统要 pad 到 batch 最大长度,浪费严重(4K/8K/32K 混合要全 pad 到 32K,浪费 ~60%)。

**Packed Sequence Support**:用 THD(Total tokens × Heads × Dimension)tensor 格式替代 SBHD(Sequence × Batch × Heads × Dimension)。THD 把所有 sequence 沿 token dim concat,通过 cumulative sequence length tracking 标记每个 sequence 起止。Attention 用 cumulative seqlen 保证跨 sequence 不互相 attend。RL 场景省 40–60% memory + 1.5–2× throughput。

**Dynamic Context Parallelism (Dynamic-CP)**:即便 packed 总 token 数相等,attention workload 因 $\mathcal{O}(s^2)$ 仍 vary(Figure 37)。Static CP 必须为最长 packed sample 选 CP size,短 sample 被迫用同一 CP size 浪费 cross-device bandwidth。

Dynamic-CP per-microbatch 选 effective CP size,跟 packing plan 联合优化。CP resize 很轻量:只改 token slice 怎么 partition + attention 用哪个 CP communication group,无需 parameter redistribution 或 optimizer state migration。

**关键 trick**:framework 初始化时预构造多个 CP group per rank,候选 cp_size 从 1 到 $dp \times cp$(power of two),运行时 scheduler 选 effective cp_size + 对应预建 cp_group,无需动态创建 communication group。

**Loss 计算**:packed THD 下不同 sample 贡献不同 valid token 数,loss per-token 算避免 padding bias:

$$\mathcal{L} = \frac{\sum_{t \in \mathcal{V}} \ell_t}{|\mathcal{V}|}$$

$\mathcal{V}$ 是 packed representation 中 valid(非 padding) token 集合。

**Solver 设计**:joint 决定 packing + per-microbatch CP size,在 GPU memory 约束下。Attention compute $\approx \mathcal{O}(S^2)$,activation memory $\approx \mathcal{O}(S)$,难以同时平衡 compute 和 memory。Dynamic-CP 在 workload-oriented 和 memory-oriented 决策间交替:workload 超 target 的 microbatch 给更大 cp_size 降 per-rank compute,然后 memory 变 dominant constraint,用 compute 较轻的 sample 填剩余 capacity 同时保持 feasible。

实测:Dynamic-CP 在多模态等序列长度高度不均场景给 35–60% end-to-end 性能提升。

---

## 9. Production Features

### 9.1 Load Balancing + Token Dropping

**Load balancing**:
- Auxiliary loss:可微 penalty,阻止 token 全 route 到小 subset expert
- Expert choice routing:balanced routing 作 optimal transport 问题
- Auxiliary-loss-free balancing:learnable expert bias term,基于历史 load 动态调整 routing 决策

**Token dropping** 两种模式:
- **Dropless**(默认):所有 routed token 处理,无 capacity 约束,最大 expressiveness 但 per-expert workload 变化
- **Droppable**:explicit expert capacity limit,超额 token drop + bypass 到 residual。可预测 memory bound,router 初始化差时有用。还支持 `pad-to-max`:把所有 expert input pad 到同一 capacity,转 dynamic token count 为 static shape,解锁 CUDA Graphs。

### 9.2 Shared Experts

DeepSeek-V2/V3, Qwen 等架构有 shared expert 处理所有 token。`--moe-shared-expert-overlap` 让 shared expert compute 跟 all-to-all 通信 + routed expert compute 并发,藏 latency。

### 9.3 Latent MoE

标准 MoE 的 all-to-all 在 full hidden dim $d$ 上 dispatch,expert weight 在 $\mathbb{R}^{m \times d}$ 和 $\mathbb{R}^{\ell \times m}$。LatentMoE 插入 shared down-projection $W_\downarrow \in \mathbb{R}^{\ell \times d}$ 在 dispatch 前,up-projection $W_\uparrow \in \mathbb{R}^{d \times \ell}$ 在 combine 后($\ell < d$):

$$\text{output}(\mathbf{x}) = W_\uparrow \cdot \left(\sum_{i \in \mathcal{T}_{K,E}} p_i E_i(W_\downarrow \cdot \mathbf{x}; \ell)\right) + \sum_j E_j^{\text{shared}}(\mathbf{x}; d)$$

压缩比 $\alpha = d/\ell$:all-to-all volume 降 $\alpha$ 倍,per-expert weight size 降 $\alpha$ 倍(expert matrix 从 $\mathbb{R}^{m \times d}$ 缩到 $\mathbb{R}^{m \times \ell}$)。

两种利用方式:
- $\ell\text{-MoE}_{\text{eff}}$:$E$ 缩 $\alpha$ 倍,$K$ 不变,保 baseline accuracy 降 inference cost
- $\ell\text{-MoE}_{\text{acc}}$(推荐):$E$ 和 $K$ 都缩 $\alpha$ 倍,inference cost 恢复到 standard MoE,但 expert selection 组合空间指数膨胀 $\binom{\alpha E}{\alpha K} \geq \binom{E}{K}^\alpha$,同 cost 下更高 accuracy。NVIDIA Nemotron-3 Super/Ultra 用这个。

### 9.4 Distributed Checkpoint

`ShardedTensor` descriptor 编码 global shape, offset, sharding pattern。Saving:每 rank 独立写 shard(Fully Parallel Saving),无 coordinator bottleneck。Loading:每 rank 根据新 sharding spec 决定需要 global tensor 的哪部分,只读那部分。Enable any-to-any parallelism reconfiguration:TP=2/EP=4 存的 checkpoint 可以 TP=4/EP=8 加载,无离线转换。

### 9.5 Flexible Asymmetric VPP

传统 VPP 要求均匀 layer 分布(24 层 model, PP=4, VPP=2 → [6,6,6,6])。MoE workload 异构:MoE layer、dense layer、embedding、loss、MTP 计算成本差很大。

Flexible Asymmetric VPP 允许 per-virtual-stage 不同数量和类型的 layer。DeepSeek-V3 PP=16, VPP=2(Table 10):第一个 PP rank 把轻量 embedding 跟 3 个 dense decoder 组合(匹配 2 个 MoE layer 成本);大部分 rank 每 stage 2 个 MoE decoder;最后 rank 把重 MTP layer 和轻 loss layer 战略性放置。

### 9.6 Upcycling

把 pre-trained dense model 转 sparse MoE,扩 capacity 不从头训。Virtual group initialization + expert weight scaling。Softmax-then-topK routing 保证 init 时 MoE output 跟 dense model 一致(Figure 42):把 dense MLP weight 在 intermediate dim shard 然后 duplicate,router weight 初始化一半然后 duplicate,保证 top-2 总选一个 shard,MoE output 等于 dense。

### 9.7 Multi-Token Prediction (MTP)

每位置预测多个连续未来 token,densify supervision signal。跟 single-token prediction 不同,MTP 通过 hidden state transition 维持 prediction 间 causal dependency。Inference 时 revert 到 single-token prediction 保部署兼容。

### 9.8 Muon Optimizer

跟 AdamW 的 element-wise update 不同,Muon 对整个 weight matrix 做 orthogonalization,改 optimization trajectory 的 conditioning。集成要点:
1. 全支持 split QKV weight layout,attention projection matrix 分开存也能正确 orthogonalize
2. 跟 distributed optimizer 无缝集成,optimizer state shard 跨 DP rank 保 orthogonalization 语义
3. CPU offloading for orthogonalization buffer

**MuonClip**:trillion-param 训练稳定性挑战,query-key dot product 可能无界增长引发 attention explosion。MuonClip 在 cuDNN/cudnn-frontend/TransformerEngine 硬件加速实现。

---

## 10. Performance Evaluation

Table 11 是核心 benchmark:

| Model | System | GPUs | SeqLen | Dtype | Per-GPU TF | Tokens/s/GPU |
|---|---|---|---|---|---|---|
| DeepSeek-V3 | GB300 | 256 | 4K | MXFP8 | 1233 | 4730 |
| DeepSeek-V3 | GB200 | 256 | 4K | MXFP8 | 1048 | 4020 |
| DeepSeek-V3 | GB200 | 256 | 4K | BF16 | 857 | 3298 |
| DeepSeek-V3 | H100 | 1024 | 4K | FP8-BLK | 368 | 1412 |
| Qwen3-235B | GB300 | 256 | 4K | MXFP8 | 974 | 6583 |
| Qwen3-235B | GB200 | 256 | 4K | MXFP8 | 919 | 6212 |
| Qwen3-235B | GB200 | 256 | 4K | BF16 | 750 | 5100 |
| Qwen3-235B | H100 | 256 | 4K | BF16 | 320 | 2132 |
| Qwen3-235B | GB300 | 128 | 131K | MXFP8 | 1150 | 1556 |

几个 take:
- GB200/GB300 vs H100:约 3× token throughput 提升,来自 memory 带宽 + 算力 + 原生 MXFP8 Tensor Core
- FP8 vs BF16(GB200 DeepSeek-V3):1048 vs 857 TF/s,约 22% 加速
- Qwen3-235B GB300 128K seq:1150 TF/s,接近短序列性能,验证长序列优化栈有效

---

## 11. Best Practices:DeepSeek-V3 Case Study

Section 9 给了三阶段 systematic workflow,然后 DeepSeek-V3 在 GB200 和 H100 上的对比 case study 是 paper 最实用的部分。

### 11.1 三阶段 workflow

**Phase 1: Establish Memory-Feasible Parallelism**
- 用 `--fake-init-process-group` 在单 GPU 模拟分布式,快速 iterate parallelism 配置
- Interactive Memory Estimator web GUI

Table 12 给各 parallelism strategy 对 memory 和 communication 的影响:
| Strategy | Peak Activation | Weight Memory | Optimizer States | Comm (Per-Layer) |
|---|---|---|---|---|
| TP | 1/d (with SP) | 1/d | 1/d | High |
| EP | ~1 (load-dependent) | 1/d (MoE only) | 1/d | Medium |
| PP | 1 (>1 with VPP) | 1/d | 1/d | Medium |
| CP | 1/d | 1 | 1/d† | Medium |
| DP | 1 | 1 | 1/d† | Low |

**Phase 2: Select Optimal Parallelism Strategy**

五条 guideline:
1. 最小化 model parallelism,最大化 DP(用 distributed optimizer shard optimizer state,腾内存给更大 DP)
2. EP×TP 保持 NVLink domain 内(通常 8 GPU/单 node,除非 MNNVL)。scale 出 NVLink domain 时 prefer PP 而非扩 TP/EP 跨 node
3. PP 用于 multi-node scaling;开 VPP 减 pipeline bubble(PP ≥ 2 时)
4. **Expert layer prefer EP over TP**:GEMM 效率更好,通信更少,computation graph 更简单,EP=num_experts 时 local token permutation 消除
5. CP for long sequence(≥8K),<4K 时 CP overhead 超 benefit

**Phase 3: Profile and Optimize Bottlenecks**

根据 profiling 结果识别是哪个 wall,对症下药。Paper 给了 4 张表(Table 13–16)列出 memory/communication/CPU overhead/computation bottleneck 各自的优化开关。

### 11.2 DeepSeek-V3 GB200 vs H100 的不同 stack

Table 17:
| Config | GB200 | H100 |
|---|---|---|
| Hardware | 256×GB200 | 1024×H100 |
| Parallelism (TP/PP/EP)† | 1/4/64 | 2/8/64 |
| VPP | 4 | 4 |
| Precision | MXFP8 | FP8-Blockwise |
| Dispatcher | HybridEP | DeepEP |
| Recompute | mlp | mlp, mla_up-proj, moe_act, layernorm |
| CUDA Graphs | Enabled | — |
| EP all-to-all Overlap | — | Enabled |
| Performance | 1048 | 368 |

**为什么差异这么大?**

**Memory 容量差**:GB200 192 GB/GPU vs H100 80 GB。GB200 能用 TP1/PP4(PP 减半),降 pipeline bubble + 简化 workload balance。H100 必须 TP2/PP8。

**Memory wall 策略差异**:
- H100:FP8 释放的 memory budget 关键,因为它 enable EP communication overlap(需要额外 buffer)。Fine-grained recompute 砍剩 memory 压力(mlp, mla_up_proj, layernorm, moe_act 全开)
- GB200:NVL72 让 EP local,communication overlap 不关键,这预算 free 出来。NVLink-C2C 带宽高让 optimizer state offloading 高效,offload 释放大量 GPU memory 给 activation,所以 fine-grained recompute 只要 mlp,远没 H100 激进

**Communication wall 策略差异**:
- H100 (NVL8):EP64 跨 8 node,cross-node all-to-all 没优化会吃 50% step time。DeepEP 降 overhead,但还不够,必须开 EP communication overlap 藏剩余 cross-node latency。overlap 可行是因为 FP8 释放了足够 memory 给额外 buffer
- GB200 (NVL72):EP64 全在 NVLink domain 内。HybridEP 满带宽利用 1.8 TB/s 双向带宽,无需 communication overlap。Hardware topology 单独解决 communication wall,bottleneck 转到 compute efficiency

**Compute efficiency wall 策略差异**:
- 共同:FP8 加速 GEMM(blockwise on H100, MXFP8 on GB200),但加速 GPU 暴露 CPU overhead
- GB200:NVL72 已消除 communication bottleneck,CPU overhead 变 dominant。Partial CUDA Graphs capture attention + router + MoE preprocessing 到 static graph。Kernel fusions 减 launch 数。CPU/NUMA binding 减 host-side memory access latency
- H100:kernel fusions,但 CUDA Graphs 没开(可能是 EP overlap 跟 CUDA Graphs 的组合在 H100 上更复杂)

**Four insights generalize**:
1. **Platform characteristics drive strategy**:GB200 大 memory + 高 C2C → 更激进选择;H100 NVL8 → 必须 EP communication overlap
2. **Parallel Folding unlocks flexibility**:decouple attention TP 跟 expert EP,跟 flexible VPP 组合做 fine-grained workload balancing
3. **FP8 shifts bottlenecks**:加速 GEMM + 降 memory,但放大 CPU overhead 作 dominant bottleneck。CUDA Graphs + kernel fusions + CPU/NUMA binding 变 essential
4. **Iterative optimization**:cyclical。Memory 优化 free GPU memory → enable communication overlap → 暴露 compute efficiency bottleneck。某些优化 cross-cutting:FP8 同时降 memory 和 compute time 但增加 CPU overhead;CUDA Graphs 降 CPU overhead 但 consume 额外 memory。每次 change 后要 re-profile,identifying diminishing returns 时转到下一个 bottleneck

---

## 12. RL 后训练支持

Section 10 讲 RL post-training 的独特挑战。

### 12.1 RL vs Pre-training 差异

1. **Variable-length sequences**:RL 产生高度变长序列,最大 128K 甚至 1M,mean 是最大值的 1/2 到 1/4。长尾分布难以平衡 compute efficiency vs peak memory
2. **Memory offloading**:RL framework 通常 co-locate training + inference engine 在同 GPU,offload 一个 engine state 当另一个 active。要求两个 engine 都快速完整 release + restore memory footprint
3. **Online weight export**:inference engine 每 training step 后要更新参数,要求 training engine 快速 export weight 到 inference engine 能 load 的格式
4. **Training stability**:标准 RL 假设 sampled response 来自 current policy distribution。但 inference 和 training engine 用不同优化 kernel,即便参数相同 token probability 略有差异,引入 off-policy bias。MoE 上更严重:同序列 token 可能 route 到不同 expert,放大差异

### 12.2 Megatron-Bridge

解决 HF ↔ Megatron checkpoint 转换。典型 RL workflow:从 pretrained HF 模型出发,在 RL framework 内 fine-tune。Scale 上 RL framework 通常在 Ray worker 内跑 Megatron-Core 作 distributed training backend,rollout 用期望 HF format checkpoint 的 inference stack。两个 integration need:HF model definition → Megatron-Core module mapping,checkpoint 转换。Megatron-Bridge 给 fast HF-to-Megatron conversion for init, training, export。veRL, Slime, NeMo RL 都用这个 pattern。

### 12.3 RL-Specific 优化

**Packed Sequence Support**:packed 去 padding + packing-aware dynamic batch size 保每 batch 相近 effective token。还有 load-balancing strategy 显式考虑 transformer block 异构计算成本:attention $\mathcal{O}(L^2)$ vs FFN $\mathcal{O}(L)$,token count 平衡的 batch wall-clock 仍可能高度不均。算每 sample seq len² 的 mini-batch sum,按这个 "attention cost" metric 排序 microbatch,用 small-to-large-to-small serpentine 调度。两个 benefit:DP/PP/EP 减 sync bubble(连续 microbatch attention workload 相近),PP warmup/cooldown 阶段 idle 时间缩(轻 microbatch 早到晚到)。

**Dynamic CP**:per-micro-batch 选 CP degree。长 sequence 给高 CP degree 保 memory-safe,短 sequence 用低 CP degree 最大化 arithmetic intensity + 降 communication。

**CPU Optimizer Offloading**:forward/backward 期间 optimizer state offload 到 host DRAM,只 optimizer step 时 load 回 GPU。释放数十 GB 高带宽 GPU memory 给 activation cache 或更长 sequence。

**FP16 Training Support**:BF16 是 pre-training 主流,但某些 RL 训练 hyper-param 下 FP16 数值稳定性更好。Megatron-Core MoE 提供 full FP16 path(loss scaling + mixed-precision optimizer kernel),让 practitioner 按 stability 需求选 precision mode 不牺牲 throughput。

**Router Replay**:近期工作显示 capture inference 时 router 产生的 routing decision,后续 training 阶段 replay 可改善 convergence consistency。Megatron-Core MoE 现支持:inference engine log 每 token 的 expert assignment,training stack ingest 并 enforce 同 routing pattern。这 decouple routing variability 跟 weight update,稳定 optimization trajectory,RL 场景 on-policy data distribution shift 频繁时尤其重要。

---

## 13. 我的 Intuition 总结

读完这篇 paper,我对 MoE training system 的 mental model 大致这样:

**核心 insight 1:MoE 的 sparsity 是 system 设计的"原罪"**。所有 Three Walls 都从 "总参数随 $E$ 增长但 per-token compute 只随 $K$ 增长" 这个 mismatch 衍生。这导致 parallelism 不能再用 dense 的"GPU 越多 computation 越多所以 communication 占比降"的 virtuous cycle。MoE 必须有 EP 这个第五维度,必须有 Parallel Folding 把 attention 和 MoE 的 parallelism 解耦,必须有 communication overlap 把 all-to-all 藏到 compute 后面,必须有 Grouped GEMM 把 small GEMM batch 起来,必须有 CUDA Graphs + sync-free 把 host overhead 干掉。

**核心 insight 2:Three Walls 是耦合的,isolated 优化 suboptimal**。Paper 反复强调这点,Section 9.2 的 DeepSeek-V3 case study 是最好的例子:同一模型,GB200 用 HybridEP + CUDA Graphs + optimizer offload,H100 用 DeepEP + EP overlap + 激进 fine-grained recompute。硬件拓扑决定瓶颈在哪,瓶颈决定优化栈,优化栈又有 cross-cutting effect 需要 iterative refine。

**核心 insight 3:algebraic rearrangement 是最 elegant 的优化**。Memory-Efficient Permutation 通过把 $p_i$ 移到 activation 内部,用纯代数变换零开销省 26 GB memory。这种 "computation 跟 memory 解耦" 的思路在 system 设计里 value 极高。

**核心 insight 4:dynamic shape 是 MoE 的 fundamental 难点**。Dropless MoE 的 dynamic token count 跟 CUDA Graphs 的 static shape 要求冲突,跟 host-device sync 冲突,跟 buffer allocation 冲突。ECHO + Paged Stashing + device-initiated kernel 三个技术组合才把 full CUDA Graphs 在 dropless MoE 上跑通。这是 paper 工程 depth 的极致体现。

**核心 insight 5:precision 是 selective 的,不是 uniform 的**。Router FP32,embedding/output/optimizer state 原精度,expert GEMM aggressive FP8/FP4。NVFP4 还要 RHT + 2D scaling + stochastic rounding 三个 trick 保稳定。Reduced precision 不是 free lunch,是精细 recipe 设计。

---

## Reference Web Links

- Megatron-LM (官方仓库): https://github.com/NVIDIA/Megatron-LM
- TransformerEngine: https://github.com/NVIDIA/TransformerEngine
- DeepEP: https://github.com/deepseek-ai/DeepEP
- DualPipe: https://github.com/deepseek-ai/DualPipe-with-HybridFlow
- Megatron-Bridge: https://github.com/NVIDIA/Megatron-Bridge
- NVIDIA NVFP4 pretraining paper: https://arxiv.org/abs/2507.12145
- DeepSeek-V3 technical report: https://github.com/deepseek-ai/DeepSeek-V3
- Qwen3 technical report: https://github.com/QwenLM/Qwen3
- Slime RL framework: https://github.com/THUDM/slime
- veRL: https://github.com/volcengine/verl
- ByteScale (long context training): https://arxiv.org/abs/2502.21231
- Tutel (MoE framework 对比): https://github.com/microsoft/tutel
- DeepSpeed-MoE: https://github.com/microsoft/DeepSpeed
- MegaBlocks (Grouped GEMM MoE): https://arxiv.org/abs/2211.15841
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- NVIDIA Developer Blog on Dynamic-CP: https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/

希望这个深度解读能 build 你对 MoE training system 的 intuition。如果某个技术细节你想再深挖(比如 HybridEP kernel 内部结构、NVFP4 RHT 数学推导、1F1B W/D split 的具体依赖图),我可以再展开。
