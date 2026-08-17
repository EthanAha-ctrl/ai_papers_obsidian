---
source_pdf: How Fast Can I Run My VLA？ Demystifying VLA Inference Performance with
  VLA-Perf.pdf
paper_sha256: b7e67a7f05fcd6e34e3f0c71c4d2b834d1715fe02673874c77efb608c8ba7c95
processed_at: '2026-08-05T00:02:26-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VLA-Perf 用人话说

## 这 paper 到底在干啥

NVIDIA 的人发现一个尴尬的事:**VLA 模型火得一塌糊涂，但没人知道这玩意儿到底能跑多快**。

你说 π0 厉害，OpenVLA 开源，GR00T 能控制人形机器人——好，那我拿个 Jetson 板子能跑吗？放云端跑网络扛得住吗？模型再做大 10 倍还能 real-time 吗？

没人答得上来。因为要回答这些问题，你得把 N 种 model 架构 × M 种 hardware × K 种 network 组合全跑一遍——组合爆炸，跑不起。

所以 NVIDIA 的做法很聪明：**别跑了，我用数学算**。

---

## 怎么算

核心就一句话：**每个 operator 的 latency = max(算力瓶颈时间, 显存带宽瓶颈时间)**。

这就是经典的 roofline model——一个 operator 要么卡在 compute 上，要么卡在 memory 上，谁慢就是谁。

然后 end-to-end latency 就是把所有 component (vision encoder、VLM、action expert) 的 latency 加起来，再加上 network 传数据的延迟。

这么算出来的 latency 是个**乐观估计**(理论 upper bound)，跟实际 optimized 系统差大概 20-30%——但够用了，因为我们要的是 order-of-magnitude 的 insight，不是精确到毫秒。

---

## 核心发现，用人话讲

### 发现 1：VLA inference 跟 LLM inference 是一个模子刻出来的

| VLA component | 类比 LLM | 瓶颈 |
|---|---|---|
| Vision encoder + VLM backbone | Prefill (处理大量 input tokens) | **Compute-bound** (算力不够) |
| Diffusion action expert | Decode (生成少量 output tokens) | **Memory-bound** (带宽不够) |

这个 insight 太关键了。Vision 和 VLM 在处理 768-800 个 input token，token 多，matrix 乘法能 batch 起来，GPU 算力能打满。但 action expert 只生成 ~50 个 action token，token 太少，大部分时间花在**反复 load model weights 上**，GPU 算力闲置。

**推论**：想加速 action expert？别买更强的 GPU，要么减小 model，要么做 quantization (INT8 直接砍掉一半 memory access)，要么加 memory bandwidth。

### 发现 2：Action chunk size 几乎免费，denoising steps 很贵

这是最 actionable 的发现。

Diffusion action expert 每做一次 denoising step，就要 reload 一遍 weights 和 KV cache。所以 10 steps → 50 steps，latency 直接 5×。

但 action chunk size 从 50 → 250 呢？只慢了 11%。因为 weights 只 load 一次，多生成几个 action token 在 memory-bound regime 下几乎是"搭便车"。

**人话**：**denoising step 能少就少，action chunk size 能大就大**。π0 用 10 steps 可能已经过多了，4-5 步可能够。Chunk size 不影响 latency 但影响你多久才 re-observe 一次环境——这是 control stability 的 trade-off，不是 performance 问题。

### 发现 3：Diffusion 完爆 Autoregressive

Vanilla autoregressive VLA 生成 14 DoF × 50 chunk = 700 个 token，要**串行 700 次 forward pass**。每次都 reload weights。B100 上跑 327 ms。

Diffusion VLA 做 10 次 denoising step，每次同时 refine 所有 700 个 action dimension。B100 上跑 3.2 ms。

**差 102 倍**。

这就是为什么 π0、GR00T、SmolVLA、TinyVLA 全都用 **separate small diffusion action expert** 这个 design。OpenVLA 那种 autoregressive 路线在 real-time 场景下 architectural 上吃亏，除非上 parallel decoding (类似 Medusa/EAGLE)。

### 发现 4：Edge GPU 的瓶颈不是算力，是带宽

Jetson Thor 的 BF16 算力是 400 TFLOP/s，比 RTX 4090 的 165 TFLOP/s 还高。但跑 π0 反而慢 1.7 倍。

原因：Jetson Thor 用 LPDDR memory，bandwidth 只有 270 GB/s。RTX 4090 用 GDDR，1008 GB/s。B100 用 HBM，8000 GB/s。

VLA workload 大量是 memory-bound，所以 Jetson Thor 上**几乎所有东西都卡在 memory**——连 compute-bound 的 vision encoder 和 VLM 都变成 memory-bound 了。

**推论**：Edge GPU 的 roadmap 关键不在加 TFLOP/s，在加 memory bandwidth。下一代 edge chip 如果还是 LPDDR，VLA 就别想 real-time 跑大 model。

### 发现 5：81B VLA 在 B100 上能跑 10 Hz

这个数字惊到我了。π0 是 2.7B，放大 30 倍到 81B (Llama2-70B 做 VLM backbone)，B100 上还能跑 9.6 Hz。

**推论**：Model size 在 server-side 部署**不是 bottleneck**。瓶颈在 training data 和 task generality。下一代 B200/Rubin 出来，GPT-4 scale 的 VLA 完全 viable。

### 发现 6：Long context 是个真问题

想让 VLA 记住过去 1000 个 timestep 的视觉历史？KV cache 要 13.2 GB。B100 (192 GB) 能 fit，但 latency 从 3.2 ms 涨到 85 ms——因为 attention 计算量随 sequence length linearly 增长 (即使有 FlashAttention)。

10K timesteps？131.8 GB KV cache，B100 能 fit 但只跑 1.2 Hz。Jetson Thor 和 RTX 4090 直接 OOM。

**推论**：Long-context VLA (ContextVLA、MemoryVLA) 需要 KV cache compression 或 hierarchical memory，光靠堆硬件不行。

### 发现 7：Server-side 比 on-device 强，除非网络极差

RTX 4090 + WiFi 都比 Jetson Thor on-device 快——因为 GPU 算力差距远大于网络延迟。

但 4G 下就不一定了。4G 的 base latency 25 ms，加上传 image 的 bandwidth 延迟，可能比 on-device 还慢。

**推论**：固定位置机器人用 edge server + Ethernet 是最优。移动机器人 (人形、轮式) 在 WiFi 7 / 5G 下 edge server 也能用，4G 以下老老实实 on-device。

### 发现 8：Device-server 协作基本是坑

想法很美好：VLM 跑 server，action expert 跑 device 上。但实际上要把 VLM 的 KV cache (117 MB) 从 server 传到 device。10G Ethernet 要 12 ms，WiFi 7 要 44 ms，5G 要 258 ms。

而 server 自己跑完整 inference 只要 3.2 ms。

**所以协作比纯 server 慢，还经常比纯 device 慢**。除非有 KV cache 大幅压缩技术，否则别走这条路。

### 发现 9：Async inference 是 wireless 部署的救星

Synchronous 模式下，network 传输和 GPU compute 串行，5G 下 sync 只有 36 Hz。

Async 模式下，robot 用 stale observation 跑 inference，network 传输和 GPU compute pipeline 起来，5G 下 async 能到 215 Hz。**6× 提升**。

Slow cloud 更夸张：sync 3.7 Hz → async 50.5 Hz，**13.8×**。

代价是 staleness——observation 到 action 之间有 delay，可能影响 control stability。但这是 deployment 必须做的 trade-off。

### 发现 10：Dual-system (System 1 + System 2) 只在好网络下才值

VLM 降频到 10 Hz，action expert 高频跑——听起来很美。但实测：

- B100 + 10G Ethernet: 2.24× speedup ✅
- B100 + 5G: 1.05× speedup ❌ (网络是瓶颈，降 VLM 频率没用)
- Jetson Thor on-device: 1.30× speedup (一般)

Figure Helix 用 on-device GPU 跑 dual-system 是合理的(网络稳定)。但如果你部署在 5G 环境下，dual-system 基本白搭。

---

## 一句话总结

这篇 paper 给了 VLA 领域第一份**性能地图**。核心 message 是：

> VLA inference 的瓶颈分布跟 LLM 一样：vision/VLM 卡算力，action 卡带宽。Diffusion action expert 是对的 architecture choice。Edge GPU 的真正瓶颈是 memory bandwidth 不是 TFLOP/s。Server-side 部署多数情况赢 on-device，但 async inference 是 wireless 场景的必备。Model size 在 server-side 不是 bottleneck，81B 都能 real-time——真正的 bottleneck 是 training data 和 long-context memory。

代码在 https://github.com/NVlabs/vla-perf ，可以直接拿来算你自己的 model-system 组合。

---

# VLA-Perf: Demystifying VLA Inference Performance 深度解读

这篇 NVIDIA Research 的工作堪称 VLA (Vision-Language-Action) systems 性能分析领域的开山之作。它解决了一个被业界忽视但极其关键的问题:**在 real-time 约束下,如何系统性地理解 VLA model 与 inference system 组合空间的性能特征**。下面我从 motivation、methodology、key findings 三个层面展开,重点 build 你的 intuition。

---

## 1. 为什么需要这篇 paper

当前 VLA 领域存在一个 fundamental 的 gap: 算法研究者关注 accuracy,部署工程师关注 latency,但**两者之间的 combinatorial space 几乎完全未被系统探索**。

这个 space 由两个正交维度构成:
- **Model dimension**: vision encoder 选择 (SigLIP-So400m vs SigLIP-Giant)、VLM backbone (Gemma-2B vs Llama2-7B/13B/70B)、action expert 架构 (autoregressive vs diffusion)、denoising steps、action chunk size、context length 等
- **System dimension**: inference location (on-device / edge-server / cloud-server)、accelerator (Jetson Thor / RTX 4090 / A100 / H100 / B100)、network (Ethernet / WiFi / 4G / 5G)、synchronous vs asynchronous execution

作者明确指出:同一个 VLA model 在不同 inference system 上跑,性能可以差**多个数量级**(orders of magnitude)。但既有工作要么针对特定 model-system pair 调优,要么用 empirical benchmarking 评估少数组合——组合爆炸让 exhaustive evaluation 在 cost 和 time 上都不可行。

这就引出了 VLA-Perf 的核心设计哲学:**用 analytical performance modeling 替代 empirical measurement**,在精度损失可接受 (~80%) 的前提下换取 exploration 的覆盖度。这与 LLM serving 领域的 Vidur [33]、Liminal [31] 思路一脉相承,是 system performance analysis 的标准做法。

---

## 2. VLA-Perf 的核心方法: Roofline-based Analytical Model

### 2.1 Architecture Abstraction

VLA-Perf 把 VLA inference 抽象为**model components 与 data transfers 交错**的 pipeline (Figure 3)。每个 model component (vision encoder, VLM backbone, action expert) 进一步抽象为一串 operators (fully-connected layers, linear projections, attention blocks)。

关键假设:
- 单个 model component 假设在单一 accelerator 上执行(因为现代 edge GPU 如 Jetson Thor 已有 128GB memory,足以 host 完整 VLA model)
- 不同 model components 可以 co-located 也可以 distributed 跨多个 accelerators

### 2.2 核心公式解析

**End-to-end latency (Eq. 1)**:
$$T_{total} = \sum_{m \in \mathcal{M}} T_m + \sum_{d \in \mathcal{D}} T_d$$

- $T_{total}$: end-to-end inference latency (从 robot 感知到收到 action prediction 的 elapsed time)
- $\mathcal{M}$: model inference components 的集合 (vision encoder, VLM, action expert)
- $T_m$: 单个 model component 的 inference latency
- $\mathcal{D}$: data movement stages 的集合
- $T_d$: 单次 data movement 的 latency

这里隐含一个 critical assumption: **所有 model components 和 data movements 是串行执行的**(没有 overlap)。这在 synchronous inference 下成立,在 asynchronous inference 下需要修正(后面会讲)。

**Per-component latency (Eq. 2)**:
$$T_m = \sum_{o \in \mathcal{O}_m} T_o$$

- $\mathcal{O}_m$: model $m$ 内部 operators 的序列
- $T_o$: 单个 operator $o$ 的 execution latency

这个公式直接相加所有 operators 的 latency,隐含**所有 operators 也是串行执行**。这其实是个乐观估计,因为实际系统里 GPU 可以做 operator-level overlap (CUDA streams),但作为 upper bound 估计是合理的。

**Roofline per-operator (Eq. 3)**:
$$T_o = \max\left(\frac{\mathrm{FLOPs}_o}{\mathrm{FLOP/s}_h}, \frac{\mathrm{Bytes}_o}{\mathrm{MemBW}_h}\right)$$

- $\mathrm{FLOPs}_o$: operator $o$ 执行的总 floating-point operations
- $\mathrm{FLOP/s}_h$: hardware $h$ 的 peak compute throughput (这里指 BF16 throughput)
- $\mathrm{Bytes}_o$: operator $o$ 访问的 memory bytes (weight + activations + KV cache)
- $\mathrm{MemBW}_h$: hardware $h$ 的 peak memory bandwidth

**Intuition**: 这是经典 roofline model。max 表示 operator 要么 compute-bound 要么 memory-bound,不能两者兼顾。比值 $\frac{\mathrm{FLOPs}_o}{\mathrm{Bytes}_o}$ 就是 operator intensity (OI),与 hardware balance point $\frac{\mathrm{FLOP/s}_h}{\mathrm{MemBW}_h}$ 比较决定 bound 类型。

**Network transfer latency (Eq. 4)**:
$$T_d^{net} = \mathrm{NetLat} + \frac{\mathrm{Bytes}_d}{\mathrm{NetBW}}$$

- $\mathrm{Bytes}_d$: transferred data size (raw images, vision tokens, KV cache, or actions)
- $\mathrm{NetBW}$: single-directional network bandwidth
- $\mathrm{NetLat}$: network base latency (RTT 的一半或 propagation delay)

这里有个简化:**没考虑 TCP 拥塞控制、packet loss、jitter**——纯 bandwidth-limited model。对 LAN (Ethernet) 合理,对 cellular network (4G/5G) 偏乐观。

### 2.3 Validation: 80% 的 fidelity 够用吗?

Table 1 显示 VLA-Perf 与 Ma et al. [29] 的 Triton-tuned π0 实现在 RTX 4090 上对比:

| Cameras | Roofline (VLA-Perf) | Real Perf (Triton) | Fidelity |
|---|---|---|---|
| 1 | 14.7 ms | 20.0 ms | 73.3% |
| 2 | 22.5 ms | 27.3 ms | 82.3% |
| 3 | 30.4 ms | 36.8 ms | 82.6% |

工作负载越大,fidelity 越高——这是 roofline model 的典型行为,因为 fixed overheads (kernel launch, OS interference) 在大 workload 下被摊薄。

作者引用 [29] 的研究:optimized system 能达到 roofline predicted 性能的 **68-75%**。这与这里 73-82% 的数据一致,说明 VLA-Perf 是个偏乐观但 order-of-magnitude 准确的工具——对于 exploration 阶段够用了。

---

## 3. Hardware Landscape 与 Roofline Balance Points

### 3.1 硬件规格全貌

Table 10 给出了详细硬件参数,这里提取关键比例:

| Hardware | BF16 TFLOP/s | Memory GB | MemBW GB/s | Balance OI (FLOP/Byte) |
|---|---|---|---|---|
| Jetson Thor | 400 | 128 | 270 | **1481.5** |
| RTX 4090 | 165 | 24 | 1008 | 163.7 |
| A100 | 312 | 80 | 2039 | 153.0 |
| H100 | 989 | 80 | 3350 | 295.2 |
| B100 | 1750 | 192 | 8000 | 218.8 |

**关键 insight**: Jetson Thor 的 balance OI 高达 1481.5,意味着需要每 byte memory access 完成 1481.5 FLOP 才能打满 compute——这是因为 LPDDR 的 bandwidth 太低 (270 GB/s vs B100 的 8 TB/s)。这导致 Jetson Thor 上**几乎所有 workload 都是 memory-bound**。

### 3.2 VLA Workload 的 OI 特征

Table 4 揭示了 VLA 各 component 的 operator intensity:

- **Vision Encoder** (SigLIP-So400m): OI = 321.4 — compute-bound on most GPUs except Jetson Thor
- **VLM Backbone** (Gemma-2B): OI = 542.8 — compute-bound on most GPUs except Jetson Thor
- **Action Expert** (Diffusion): OI = 54.0 — memory-bound on all GPUs

**Intuition**: 这个现象与 LLM inference 的 prefill vs decode 二分法完全一致 [37]:
- Vision encoder 和 VLM backbone 处理 768-800 个 input tokens,batching computation across tokens,类似 LLM **prefill** —— compute-bound
- Diffusion action expert 处理 ~50 个 action tokens (action chunk size),token 数量少,主要时间花在 loading weights 和 KV cache,类似 LLM **decode** —— memory-bound

这个 insight 非常 actionable: **加速 action expert 的关键不是更强的 compute,而是更高的 memory bandwidth 或更小的 model**。所以对 action expert 做 quantization (INT8/FP8) 收益巨大,因为它把 memory-bound 转化为更轻量的 memory access。

---

## 4. 15 个 Takeaways 的深度解读

我把它们按 model-side 和 system-side 重新组织,并补充技术 intuition。

### 4.1 Baseline 性能 (Takeaways 1-2)

**Takeaway 1**: Datacenter GPU 已能匹配 camera frame rate (24-60 Hz) 跑 π0;edge GPU 仍然受限。

具体数据 (Table 3):
- B100: 3.18 ms / 314.4 Hz
- H100: 6.15 ms / 162.5 Hz
- A100: 16.20 ms / 61.7 Hz
- RTX 4090: 31.06 ms / 32.2 Hz
- Jetson Thor: 52.57 ms / 19.0 Hz

**Intuition**: 注意 RTX 4090 的 BF16 throughput (165 TFLOP/s) 比 Jetson Thor (400 TFLOP/s) 还低,但 inference latency 反而快 1.7×。原因就是 memory bandwidth 差距 (1008 vs 270 GB/s),而 VLA workload 严重依赖 memory bandwidth。

**Takeaway 2**: Action prediction 在所有硬件上 memory-bound;vision 和 VLM 在 Jetson Thor 之外都 compute-bound。这条 takeaway 上面已分析。

### 4.2 Model Scaling (Takeaways 3-4)

作者构造了 π0-L (9.1B)、π0-XL (16.7B)、π0-XXL (81.3B) 系列。

**Takeaway 3**: VLA 各 component 的 latency 大致与 model size 线性相关 (Figure 5 log-log 图)。

这在 log-log plot 上表现为斜率 ~1。直觉上,transformer 的 FLOPs 与参数量 linear 相关 (per-token FLOPs ≈ 2 × params),memory access bytes 也与参数量 linear,所以 roofline 下 latency 与 params 正比。

**Takeaway 4**: Edge 和 consumer GPU 在 large model 上挣扎,datacenter GPU 可支持比 π0 大一个数量级的 VLA real-time inference。

具体 (Table 5):
- π0-XXL (81B) on B100: **9.6 Hz** (仍 > 10 Hz 阈值附近)
- π0-XL (16.7B) on RTX 4090: N/A (OOM)
- π0-XL (16.7B) on Jetson Thor: 2.1 Hz (不可用)

**Critical insight**: 81B 参数的 VLA 在 B100 上仍能跑到 9.6 Hz——这意味着**模型 scaling 的真正瓶颈不在 inference latency,而在 memory capacity 和 training data**。这指向一个研究机会: 类似 GPT-4 scale 的 VLA 在下一代 datacenter GPU (B200, Rubin) 上完全可行。

### 4.3 Long-Context VLA (Takeaway 5)

Long-context VLA 让模型处理过去 1K-10K timesteps 的视觉历史 (Table 6)。

| Timesteps | KV Cache Size | B100 Hz |
|---|---|---|
| 1 | 0.01 GB | 314.4 |
| 10 | 0.13 GB | 254.6 |
| 100 | 1.3 GB | 88.4 |
| 1000 | 13.2 GB | **11.7** |
| 10000 | 131.8 GB | 1.2 |

**Intuition**: KV cache linearly 增长,每 timestep 增加 768 vision tokens × KV per token。在 1K timesteps 时 KV cache 达 13.2 GB,占总 memory 显著比例。B100 有 192 GB memory,所以 1K timesteps 仍可 fit;但 10K timesteps 需要 131.8 GB,虽然 fit 但 attention 计算成本随 sequence length quadratic 增长 (即使有 FlashAttention 也是 linear 在 sequence 上)。

**Important caveat**: Table 6 显示 attention 计算成本主导了 long-context latency。从 1K 到 10K timesteps,latency 从 85.2 ms 暴涨到 823.7 ms,**约 10× 增长**,正好对应 sequence length 增长 10× (因为 attention 的 sequence dimension 是 linear 的 with FlashAttention,不是 quadratic)。

**研究启示**: Long-context VLA 是个**真问题**。ContextVLA [39]、MemoryVLA [40]、KARMA [41] 这类工作需要严肃考虑 KV cache compression 或 hierarchical memory。

### 4.4 Denoising Steps vs Action Chunk Size (Takeaway 6)

这是 diffusion-based VLA 最 actionable 的发现之一。

**Takeaway 6**: Denoising steps 对 latency 影响显著,action chunk size 影响微乎其微。

具体 (Figure 6, B100):
- Denoising steps 从 10 → 50 (5×): action latency 5× 增加,VLA latency 2.15× 增加
- Action chunk size 从 50 → 250 (5×): action latency 仅 40% 增加,VLA latency 仅 11% 增加

**Intuition**: 这是 memory-bound workload 的典型特征。Diffusion action expert 每次 denoising step 都要 reload 整个 model weights 和 KV cache (因为没有 batching across steps),所以 step 数 linearly 增加 latency。但增加 chunk size 只增加 output 维度,weights 和 KV cache 只 load 一次,额外的 compute (matrix-vector with larger output dim) 在 memory-bound regime 下几乎免费。

**Actionable insight**: 
1. 减少 denoising steps (10 → 5 或更少) 是最直接的 latency 优化手段。π0 论文也提到 4 steps 可能足够。
2. 增加 action chunk size 几乎免费——如果任务允许 (execution horizon 可以拉长),不妨把 chunk size 设大些。这与 ACT [22]、π0-FAST [14] 的 design choice 一致。

### 4.5 Diffusion vs Autoregressive (Takeaways 7-8)

这是 paper 里最有 architectural significance 的对比。

**Takeaway 7**: 有 action chunking 时,diffusion-based VLA 比 vanilla autoregressive VLA 快 1-2 个数量级。

具体 (Figure 7a, B100, chunk size 50):
- Diffusion (standard, 300M action expert): 3.2 ms
- Diffusion-Large (action expert = VLM size): 仍然很快
- Autoregressive (classic, 700 sequential steps): **327.6 ms** (102.4× slower)
- Autoregressive-Parallel (parallel decoding): 中间值

**Intuition**: Autoregressive 生成 14 DoF × 50 chunk = 700 个 token,需要 700 次串行 forward pass,每次都 reload weights——这是 worst case for memory-bound execution。Diffusion 用 10 次 denoising step,每次同时 refine 所有 700 个 action dimension (用 batched matrix multiplication),所以 GPU 利用率高。

**Takeaway 8**: Autoregressive VLA 只在以下情况有竞争力:
1. Action token 数量少 (e.g., 7 DoF 无 chunking, 见 Figure 7b)
2. Parallel decoding (类似 Medusa/EAGLE) 在 chunk size ≤ 10 时快过 diffusion

**Critical nuance** (Figure 7a 趋势): Parallel decoding 在 chunk size 增大时从 memory-bound 转为 compute-bound。B100 的 balance OI = 218.8,parallel decoding 的 OI 从 chunk=10 时的 135.9 升到 chunk=50 时的 477.7——**跨过了 balance point**,变成 compute-bound,所以 latency 暴涨。Diffusion 因为 OI 一直保持 ~54.0 (远低于 balance point),始终 memory-bound,latency 稳定。

**研究启示**: 未来 VLA 架构设计的 sweet spot 是 **separate, smaller diffusion action expert**——这是 π0、GR00T [6]、SmolVLA [8]、TinyVLA [7] 都采用的 design。OpenVLA [10] 这种 autoregressive 路线在 real-time 场景下确实有 architectural disadvantage,除非有 parallel decoding 加持。

### 4.6 Deployment Location (Takeaways 9-10)

**Takeaway 9**: Server-side inference,即使只有 consumer GPU (RTX 4090 + WiFi),多数情况下也胜过 on-device inference。只有网络极差时 on-device 才胜出。

**Intuition**: 服务器侧的优势在于 hardware capability (RTX 4090 比 Jetson Thor 强很多),劣势在于 network latency。Trade-off 的临界点大致是:
- RTX 4090 + WiFi 6: 网络加 3.5 ms latency,但 GPU 省下 ~20 ms —— 净赚
- RTX 4090 + 4G: 网络 25 ms,可能亏
- B100 + 5G: 网络 10 ms,B100 速度优势巨大 —— 净赚

**Takeaway 10**: Device-server collaboration 通常比 device-only 慢,总是比 server-only 慢。

作者测试的 collaboration 方案: VLM (vision encoder + VLM backbone) 在 B100 server,action expert 在 Jetson Thor device。问题在于**KV cache 从 server 下载到 device**:
- Ethernet 10G: 12.4 ms
- WiFi 7: 43.7 ms
- 5G: 257.7 ms

**Intuition**: KV cache 大小 = sequence length × num layers × hidden dim × 2 (K and V) × 2 (bytes per BF16)。π0 的 VLM KV cache 大约 = 800 × 18 × 2048 × 2 × 2 ≈ 117 MB。在 1 Gbps Ethernet 上 transfer 117 MB 需要 ~1 s,所以即使 10G Ethernet 也要 100 ms——而 server-side 跑完整 inference 只要 3.2 ms。Collaboration 根本不划算,除非有 model compression 大幅缩减 KV cache 大小。

**Implication for mobile robots**: 人形机器人配 Jetson Thor 做 on-device inference 是合理的(因为网络不稳定),但 hybrid 方案基本行不通。要么全 on-device,要么全 server-side。

### 4.7 Asynchronous Inference (Takeaways 11-12)

**Takeaway 11**: Robot execution 和 inference 之间的 asynchrony 在慢网络下能带来 2.6-6× throughput 提升 (Table 8)。

具体:
- B100 + 5G: sync 35.9 Hz → async 215.3 Hz (5.99×)
- B100 + 4G: sync 13.7 Hz → async 50.5 Hz (3.68×)
- B100 + Slow Cloud: sync 3.7 Hz → async 50.5 Hz (**13.79×**)

**Intuition**: Asynchronous 模式下,network transmission 和 GPU compute 可以 pipeline。Throughput 受限于 max(network time, GPU time)。在 5G 下,network upload ~26 ms,GPU 3.2 ms,所以 throughput 受 network 限制,但比 sync 模式好因为 sync 模式下完全不能 overlap。

**Important caveat**: Async 不减少 latency,只增加 throughput。staleness (从 observation 到 action 的时间差) 可能影响 control stability。这是 OpenVLA-OFT [25]、VLAsh [17] 等工作的研究方向。

**Takeaway 12**: System 1 + System 2 dual-system paradigm 的 speedup 高度依赖硬件和网络。

具体 (Table 9):
- B100 + Ethernet 10G + S2 cap=10Hz: 2.24× speedup
- B100 + 5G + S2 cap=10Hz: 仅 1.05× speedup
- Jetson Thor + on-device + S2 cap=5Hz: 1.46× speedup

**Intuition**: Dual-system 的核心 idea 是把 VLM (System 2) 从 301.4 Hz 降到 10 Hz,把节省的 compute 给 System 1 (action expert)。但如果 network 已经是瓶颈,降 VLM frequency 也没用——这正是 5G 下 speedup 微弱的原因。

**研究启示**: Figure Helix [18]、HIRT [19]、Hume [20] 这类 dual-system 设计**只在 fast network + powerful GPU 组合下才有显著收益**。Helix 用 on-device GPU 但设计成 System 1 高频,JRT [19] 用 cloud GPU 配 fast network。设计选择必须与 deployment scenario 匹配。

### 4.8 10 Hz 与 100 Hz 目标 (Takeaways 13-15)

**Takeaway 13**: On-device inference 下,Jetson Thor 跑 π0 可达 19 Hz (超 10 Hz 目标),但 100 Hz 需 model-level 优化 (减 diffusion steps、quantization、小 model)。

**Takeaway 14**: Edge-server inference 下,10 Hz 用 RTX 4090 + 4G 即可;100 Hz 需要 B100 + fast network (Ethernet 或 WiFi 7)。

**Takeaway 15**: Cloud-server inference 下,10 Hz 在好网络下可行;100 Hz 必须 async inference。Slow cloud 下 sync 只有 3.7 Hz,async 可恢复到 50.5 Hz。

**Intuition for cloud 100Hz**: Cloud inference 的 RTT 通常 >10 ms,单次 upload+download 就 20+ ms,所以 sync 模式天花板就是 50 Hz。Async 把 network 当 pipeline stage,只要 GPU compute (3.2 ms) 快于 network (~10 ms),throughput 由 network 决定——但仍可达 100 Hz。

---

## 5. Paper 没说但很重要的 Extension 想法

### 5.1 Quantization 与 Sparsity 未充分建模

VLA-Perf 只考虑 BF16,但实际部署大量使用 INT8 (BitVLA [11])、FP8 (H100/B100 原生支持)。对 memory-bound 的 action expert,INT8 能直接 2× 加速(memory bytes 减半),这对结论会有显著影响。Paper 也未考虑 sparsity (MoE, structured pruning),这些都能改变 OI 分布。

### 5.2 Vision Encoder 是 on-device 的天然候选

Vision encoder (SigLIP-So400m) 是 411M 参数,相对小,且 output (vision tokens) 只有 768 × hidden_dim ≈ 1.5 MB——远小于 raw images (3 × 224 × 224 × 3 = 1.4 MB)。**所以 vision encoder 放 device 上跑,只 upload vision tokens 给 server,能显著降低 network 压力**。这是 Figure 9 没探索的 alternative collaboration 方案。

### 5.3 Action Chunking 与 Control Stability

Paper 假设 action chunk size 越大越好(因为 latency 几乎免费)。但实际 robot control 中,大 chunk size 意味着对环境变化的反应延迟。π0-FAST [14]、ACT [22] 都讨论过 execution horizon 与 closed-loop stability 的权衡。这个 dimension 不在 VLA-Perf 模型内,但 deployment 时必须考虑。

### 5.4 Multi-robot Sharing

VLA-Perf 假设 batch size = 1 (单 robot)。但 server-side 部署天然支持 batching 跨多个 robots。这时 GPU utilization 会大幅提升,VLM prefill 的 batching efficiency 接近线性。Paper 完全没探索这个 dimension,但实际 fleet 部署 (e.g., 仓库机器人) 这是最重要的 cost factor。

### 5.5 推理之外:Camera 与 Actuator Latency

Paper 在 Section 5 明确承认没考虑 sensor latency (camera exposure time ~10-30 ms) 和 actuator latency (motor control loop ~1-10 ms)。这些 latency 加起来可能 50 ms,占 end-to-end latency 大头。完整的 robot system 性能分析必须包括 sensing-perception-action 全链路。

---

## 6. 与相关工作 Positioning

### 6.1 与 LLM Serving 性能分析的关系

VLA-Perf 直接借鉴 LLM serving 领域的 roofline modeling 传统 [31-36],特别是 Vidur [33]、Liminal [31]。但有几个独特挑战:
1. VLA 是 multi-component pipeline (vision + VLM + action expert),而 LLM 通常单 model
2. VLA 涉及 robot-server 通信,LLM serving 通常是 datacenter-internal
3. VLA 的 action expert 是 diffusion-based,与 LLM 的 autoregressive decode 性质不同

### 6.2 与 Real-time VLA Inference Optimizations

Ma et al. [29] (https://arxiv.org/abs/2510.26742) 是 VLA-Perf 的 validation 基准,使用 Triton 优化 π0 在 RTX 4090 上跑 20-36.8 ms。Dadu-Corki [30] (https://arxiv.org/abs/2504.04586) 用 algorithm-architecture co-design。VLA-Perf 提供了**评估这些优化工作 upper bound 的工具**——未来如果某个 VLA inference engine 想知道还有多少优化空间,跑 VLA-Perf 看理论 roofline,再量实际 latency,差距就是优化空间。

### 6.3 与 VLA Model 设计的 feedback loop

Paper 的 15 个 takeaways 应该成为 VLA model designer 的 checklist:
- 想用 long context?→ 确认 deployment hardware,1K timesteps 是 B100 的上限
- 想用 autoregressive action expert?→ 必须 parallel decoding,否则 real-time 不可行
- 想用 dual-system?→ 必须配 fast network,5G 下基本无效
- 想要 100 Hz?→ on-device 必须做 model-level optimization,server-side 必须 async

---

## 7. 我的总体评价

### 7.1 强项

1. **First-mover advantage**: 这是 VLA inference 性能分析的开山之作,占了一个重要生态位
2. **Methodology soundness**: Roofline modeling 在 LLM 领域已验证,直接迁移合理
3. **Breadth of exploration**: 15 个 takeaways 涵盖 model、system、deployment 多个维度
4. **Open-source**: VLA-Perf 代码在 https://github.com/NVlabs/vla-perf 开放,可复现可扩展

### 7.2 局限

1. **Validation scope 窄**: 只用 π0 + RTX 4090 一个组合 validate,其他 model 架构 (OpenVLA、GR00T) 和硬件 (H100, Jetson) 的 fidelity 未知
2. **忽略 batching**: batch=1 假设对 fleet deployment 严重低估 server-side 效率
3. **未考虑 quantization**: 实际部署基本都用 INT8/FP8,纯 BF16 估计高估 latency 2-4×
4. **未考虑 kernel-level optimization**: CUDA graph、operator fusion 在 [29] 报告 5× 加速,VLA-Perf 没建模这部分

### 7.3 对 Andrej 的针对性思考

如果你在思考下一个 VLA 项目,VLA-Perf 的结论可以这样用:

- **不要 over-index on small models**: B100 能跑 81B VLA 在 9.6 Hz,所以 model size 在 server-side 部署不是 bottleneck。瓶颈在 training data 和 task generality
- **Action expert 设计是关键**: 用 separate, small (300M-1B) diffusion action expert,OI 保持低 (~50) 以确保 memory-bound 但 model 小,latency 自然低
- **Long-context 是真问题**: 1K timesteps 是 B100 上限,要更长需要 KV cache compression 或 amortized context (ContextVLA [39])
- **Async 是 deployment 必备**: 任何 server-side 部署都应考虑 async inference,特别是 wireless 场景

---

## 8. 参考资源

- VLA-Perf GitHub: https://github.com/NVlabs/vla-perf
- π0 paper: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- GR00T N1: https://arxiv.org/abs/2503.14734
- OpenVLA: https://arxiv.org/abs/2406.09246
- Octo: https://arxiv.org/abs/2405.12213
- RT-2: https://arxiv.org/abs/2307.15818
- Figure Helix: https://www.figure.ai/news/helix
- SmolVLA: https://arxiv.org/abs/2506.01844
- TinyVLA: https://arxiv.org/abs/2409.12514
- BitVLA (1-bit quantization): https://arxiv.org/abs/2506.07530
- Real-time VLA execution (Ma et al.): https://arxiv.org/abs/2510.26742
- Dadu-Corki: https://arxiv.org/abs/2504.04586
- Vidur (LLM serving simulator): https://arxiv.org/abs/2405.05797
- Splitwise (prefill/decode split): https://arxiv.org/abs/2311.18677
- π0-FAST (real-time chunking): https://arxiv.org/abs/2506.07339
- ContextVLA: https://arxiv.org/abs/2510.04246
- MemoryVLA: https://arxiv.org/abs/2508.19236
- OpenVLA-OFT (fine-tuning speed): https://arxiv.org/abs/2502.19645
- VLAsh (async inference): https://arxiv.org/abs/2512.01031
- Gemini Robotics: https://arxiv.org/abs/2503.20020

这篇 paper 是 VLA systems 研究的奠基性工作,后续会有大量 follow-up 在 quantization、batching、kernel fusion、KV cache compression 等 dimension 上细化。它建立了 baseline framework,后续工作可以在这个框架内填空。
