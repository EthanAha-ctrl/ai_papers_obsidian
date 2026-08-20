---
source_pdf: DualPath Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference.pdf
paper_sha256: 6d135c6563a7b6db1c5f2b27de50dee65f1bbdb4507bc6ddf871b88e3e0d6aa2
processed_at: '2026-08-04T00:34:58-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---
## 一、场景：一个 agent 在干活的某一天

想象你在跑一个 coding agent，让它修一个 bug。它不是一问一答就完事，它会循环很多轮：

```
Round 1: 读文件 → 想想 → 跑测试
Round 2: 看报错 → 改代码 → 再跑测试
Round 3: 又报错 → 再改 → ...
...
Round 157: 终于修好了
```

paper 里的真实 trace 就是这种：平均 157 轮，每轮只新增 429 个 token，但 context 累积到 32k。到 round 100 的时候，model 已经"记得"前面 30k token 的对话历史。

"记得"在工程上是什么意思？就是这 30k token 的 **KV-Cache** 得存在某地方。下一轮推理时，model 要先把这些 KV-Cache 从某处加载回来，才能接着算那新增的 429 个 token。

**关键 ratio**：每算 1 PFLOP 的 attention，需要读回 13~267 GB 的 KV-Cache（取决于 model 架构，见 paper Table 1）。而一个节点的 storage NIC 才 400 Gbps ≈ 50 GB/s。于是 GPU 大部分时间在等 storage 把 KV-Cache 喂进来——**GPU 闲死，disk 忙死**。

这就是 paper 第一段想说的：agentic 时代，inference 已经从 compute-bound 变成了 **I/O-bound**，而且是 storage I/O bound。

参考: [DeepSeek-V3 tech report](https://arxiv.org/abs/2412.19437), [Mooncake (FAST'25)](https://www.usenix.org/conference/fast25/presentation/qin)

---

## 二、现状架构里藏着一个"半边闲半边死"的尴尬

现代大厂 inference 基本都用 **PD disaggregation**：prefill 和 decode 拆到不同 GPU 上跑。原因很简单——prefill 是 compute-heavy 的批处理，decode 是 latency-sensitive 的逐 token 生成，硬塞一起会互相打架。DistServe、Splitwise 这些 paper 早就讲清楚了。

架构长这样：

```
            ┌─────────────────────┐
Storage ───►│  PE (prefill engine) │─── KV via RDMA ───►  DE (decode engine)
            └─────────────────────┘                       │
                  SNIC 忙死                              SNIC 闲死
```

PE 要从 storage 把 hit 的 KV-Cache 全读进来（因为 prefill 要算 attention，必须有完整 KV-Cache），所以 PE 节点的 SNIC 被打爆。

DE 呢？decode 阶段它只需要把新生成的 KV-Cache 往 storage 里 write back（每生成 64 token 存一次），读很少。于是 DE 的 SNIC 大部分时间在睡大觉。

**一半带宽在排队等，一半带宽在打呼噜**。这就是 paper Figure 1 画的那张图想说的。

为什么以前没人在乎？因为 chatbot 时代一轮就完事，KV-Cache 不大，这个不对称不明显。agentic 一上场，几十上百轮、context 飙到几十 k，问题直接爆出来。

参考: [DistServe (OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin), [Splitwise (ISCA'24)](https://dl.acm.org/doi/10.1145/3695054.3695065)

---

## 三、Aha moment：让闲着的 DE 帮忙读

paper 的核心 insight 一句话能讲完：

> KV-Cache 不一定要直接读到 PE，可以先读到 DE，再通过 compute network 用 RDMA 顺手捎给 PE。

为什么这是个好主意？两个观察凑一起：

**观察 1：CNIC 带宽远大于 SNIC。** 一个 Hopper DGX 节点有 8 个 400Gbps CNIC（east-west，给 GPU 间通信用），只有 1 个 400Gbps SNIC（south-north，给 storage 用）。CNIC 总带宽是 SNIC 的 8 倍。

**观察 2：CNIC 上的 traffic 是 bursty 的。** Model 执行时，MoE 的 AllToAll、TP 的 ReduceScatter 这些 collective 是亚毫秒级的尖峰，中间有大段空闲。KV-Cache 传输可以"捡漏"用这些空闲。

把这两个加起来：让 DE 的 SNIC 帮忙读 KV-Cache（利用闲置的 storage 带宽），然后用 DE 的 CNIC 通过 compute network 顺手送到 PE（利用 bursty 间隙的空闲带宽）。**等于白嫖了一倍的 storage 带宽**。

这就是 DualPath 名字的由来——两条 path：

```
Path 1 (老路):    Storage ──► PE buffer ──► PE HBM
Path 2 (新路):    Storage ──► DE buffer ──► (CNIC RDMA) ──► PE HBM
```

Scheduler 根据两边 queue 长度动态决定每个 request 走哪条。

参考: [DeepEP - MoE expert parallel comm](https://github.com/deepseek-ai/DeepEP), [3FS - DeepSeek 的分布式 FS](https://github.com/deepseek-ai/3FS)

---

## 四、为什么这事 tricky：三个工程硬骨头

idea 听起来简单，做起来 paper 自己列了三个 challenge：

### Challenge 1: KV-Cache 被切成了细碎的 per-layer block

现在主流 prefill engine 都用 **layerwise prefill**（LayerKV、PrefillOnly 这些 work）。原因：长 context prefill 时，如果整批 KV-Cache 全装进 HBM，HBM 撑不住，batch size 上不去。layerwise 的做法是一层一层来——每层只装当前层的 KV-Cache，算完释放，再装下一层。这样 effective batch size 放大 ~N_layers 倍。

副作用：原本"一次大块 I/O"变成"N_layers 次小块 I/O"。比如 60 层 model，block 数量 ×60，每个 block 大小 ÷60。RDMA 提交延迟对小块极敏感——每次 submit 都有固定 overhead。

paper 的解法：用 **CNIC-assisted copy** 代替 CUDA copy engine。理由是 CUDA copy engine 单次 `cudaMemcpyAsync` 要 5-7 μs（driver 黑盒），而 RDMA Write work request 只需 ~1 μs（user space 直接 mmio 写 NIC 寄存器），还能用 doorbell batching 摊销。所以表面看绕了一圈（storage→DRAM→CNIC→GPU），实际更快，因为提交开销低一个数量级。

参考: [LayerKV](https://arxiv.org/abs/2410.00428), [PrefillOnly (SOSP'25)](https://dl.acm.org/doi/10.1145/3694715.3695963), [Doorbell batching paper (USENIX ATC'16)](https://www.usenix.org/conference/atc16/conference-program/presentation/kalia)

### Challenge 2: KV-Cache traffic 不能影响 model execution

MoE 的 AllToAll 这些 collective 是 latency-critical 的，亚毫秒级 burst。如果 KV-Cache traffic 跟它们抢 PCIe 带宽，inference 延迟直接飙。GPU 又不支持 PCIe QoS（paper 引用 [39] 说的），软件 traffic shaping 来不及反应（collective 太快了）。

paper 的招：**CNIC-centric**——所有 GPU 的进出 traffic 都强制走 paired CNIC，利用 InfiniBand 的 Virtual Lane 做硬件级 QoS。

具体配置（paper §A.1）：
- 4 个 VL，VL0/1/3 给高优先级（model execution），VL2 给低优先级（KV-Cache）
- 高优先级占 ~99% 带宽，低优先级 ~1%（防 starvation）
- 用 Weighted Round Robin arbiter

这样 model traffic 几乎不受影响，KV-Cache traffic 捡漏用那 1% + 空闲时段的剩余带宽。

代价：所有 GPU 显存读写都得绕 CNIC，包括本地的 H2D/D2H。看起来低效，但 paper 实测在大量小块场景下比 CUDA copy engine 还快——因为 RDMA submit 开销低。

参考: [InfiniBand spec](https://www.infinibandta.org/), [PCIe QoS 问题 (DSD'16)](https://ieeexplore.ieee.org/document/7784694)

### Challenge 3: 实时调度要平衡三个维度

每个 request 来了，要决定：
1. 给哪个 PE
2. 给哪个 DE
3. KV-Cache 从 PE 读还是从 DE 读

同时要平衡：GPU compute 负载、SNIC queue、CNIC traffic。

paper 用两阶段调度，核心是用 **token count 当 proxy**——因为 GPU/SNIC/CNIC 负载都跟 token 数强相关。

**PE 调度**（Algorithm 1）把 PE 分三类：
- $C_1$: $tok_e > \beta$（过载，不分配）
- $C_2$: disk queue 短且 $tok_e \leq \beta$（首选，SNIC 闲）
- $C_3$: disk queue 长但 $tok_e \leq \beta$（次选）

$\alpha$ 和 $\beta$ 是两个阈值常量，paper 实测设成"3 秒能读的 token 数"和"5 秒 GPU 能算的 token 数"。

**DE 调度**两阶段：先 across group（选总 token 最少的 group），再 within group（在 HBM 够的 DE 里选 token 最少的，留 5% buffer）。

直觉：用 token 当 proxy 是个简单但有效的近似——所有资源维度都跟 token 数正相关，平衡了 token 就大致平衡了所有资源。

---

## 五、那个漂亮的 Bottleneck-Free 证明

paper §4.2 有段数学，看起来吓人，其实讲的事情很简单：**dual-path 在合理的 P/D ratio 下不会引入新 bottleneck**。

让我把变量翻译成人话：

| 变量 | 意思 |
|------|------|
| $P$ | prefill 节点数 |
| $D$ | decode 节点数 |
| $g$ | 每节点 GPU 数（Hopper DGX = 8） |
| $s$ | 每节点 storage NIC 数（一般 1） |
| $B$ | 单个 CNIC 带宽（50 GB/s） |
| $M$ | 每节点 DRAM 带宽（~500 GB/s） |
| $T_p$ | PE read path 每 PE-DE pair 流量 = $Bs/(Dg^2)$ |
| $T_c$ | DE read path 每 pair 流量 = $Bs/(Pg^2)$ |

为什么 $T_p$ 分母有 $Dg^2$？因为 storage 带宽 $Bs$ 被均摊到 $D \cdot g$ 个 DE 上（每个 DE 都可能帮任意 PE 读），每 pair 又共享，所以再除一个 $g$。下标 $p$ 指 prefill path，$c$ 指 cross（DE 协助）path。

核心 4 个不等式：

**PE CNIC read**：$2Bs/g \leq B$，化简因为 $s \leq g$ 总成立，所以永远 OK。

**PE CNIC write**：$\frac{Bs}{g}(1 + D/P) \leq B$ → $P/D \geq s/(g-s) = 1/7$

**DE CNIC read**：$\frac{s}{g}(P/D + 2)B \leq B$ → $P/D \leq (g-2s)/s = 6$

**DE CNIC write**：类似 → $P/D \leq (g-s)/(2s) = 3.5$

**DRAM**：$(3 + 2P/D)Bs \leq M$ → $P/D \leq (M/Bs - 3)/2 = (500/50 - 3)/2 = 3.5$

合起来：

$$\frac{1}{7} \leq \frac{P}{D} \leq 3.5$$

**人话**：只要 prefill 节点数和 decode 节点数的比例在 1:7 到 3.5:1 之间，dual-path 不会让任何单一资源爆掉。这覆盖了几乎所有实际配置——1P1D、2P4D、4P8D 都在范围内。

**为什么这么宽松？** 因为根本 asymmetry：CNIC 总带宽是 SNIC 的 $g$ 倍（8 倍）。"绕路"通过 CNIC 几乎免费，所以无论怎么分配，CNIC 都撑得住。bottleneck 始终在 SNIC，而 dual-path 恰好把 SNIC 负载摊到所有节点上。

这个证明是 paper 的理论 backbone——它告诉你"dual-path 不是经验主义 hack，在很大配置范围内都成立"。

---

## 六、实验讲了个什么故事

### 6.1 离线推理（rollout 场景）

RL rollout 就是同时跑几千个 agent，谁先跑完谁先回去更新参数。测的是 **JCT（job completion time）**。

paper Figure 7 的核心数字：

| Model | DualPath vs Basic | DualPath vs Oracle |
|-------|-------------------|--------------------|
| DS 660B | 最高 **1.87×** | 接近 Oracle，I/O 几乎消除 |
| DS 27B | 最高 1.78× | 仍慢 1.09-1.85×（1P1D storage 带宽小） |
| Qwen 32B | 类似 27B | - |

**Ablation**（Figure 12 右）很关键，逐项加：
- + Layerwise prefill: JCT 降 17.21%
- + Dual-path: 累计降 38.19%（**主要 gain 在这**）
- + Scheduling: 累计降 45.62%

直觉：layerwise 是 enabler（让 overlap 可能），dual-path 是 main course（真正聚合带宽），scheduling 是 polish（智能选 path）。

### 6.2 P/D ratio 扫描（Figure 8）——最有说服力的对照

DS 27B 上跑了 1P1D、2P1D、1P2D：

```
Basic 1P1D ≈ Basic 1P2D         (storage 带宽都是 1 节点)
DualPath 1P1D ≈ Basic 2P1D      (都是 2 节点 storage 带宽)
DualPath 2P1D ≈ DualPath 1P2D   (都是 3 节点 storage 带宽)
```

**这表说明**：性能几乎只取决于"能用上几个节点的 SNIC"，跟 P/D 怎么分没关系。这反向证实了 paper 的核心论点——**storage bandwidth 是 dominant bottleneck**，GPU 数量反而是次要的。

### 6.3 Online serving（Figure 10-12）

SLO: TTFT ≤ 4s, TPOT ≤ 50ms。

- DS 660B: DualPath **2.25×** APS 容量
- DS 27B: DualPath 1.67× APS 容量
- TPOT/TTST 跟 Basic 相当（没引入额外 decode 开销）

Figure 12 左的 TTFT breakdown 最直观：DualPath 的 queue time 随 APS 几乎不涨，Basic 的 queue time 随 APS 线性飙——因为 Basic 的 storage 带宽不够，请求堆在队列里等。

### 6.4 大规模（Table 3, Figure 15）

48P96D（1152 GPU）跑 48K context 的离线推理，JCT 3,201s，跟 2P4D 跑 2K agent 的 3,167s 几乎一样。**24× 规模，JCT 几乎不变**——近线性扩展。

Online: 44P88D 能扛 8.8 APS，比 2P4D 的 0.4 APS 高 22×，latency 几乎不变。Scheduler CPU < 10 核，不是瓶颈。

---

## 七、几个值得琢磨的细节

### 7.1 Block layout 的解耦

paper §A.5 设计了两种 block：
- **Layer Block**: shape `[1, block_size, cache_bytes]`，单层
- **Full Block**: shape `[n_layer, block_size, cache_bytes]`，所有层

storage 用 Full Block（大块 I/O，throughput 友好），transfer/compute 时用 Layer Block（小块，能 overlap）。两者用 `concat` 互转，**避免手动 memory layout 转换**。

直觉：storage 和 transfer 的 optimal granularity 不一样，layout 解耦让两边各取所需。这种"用数据结构 hiding complexity"的工程思路很经典。

### 7.2 Working set 分析（§8.2）

paper 给了个 working set 估算公式：

$$\text{Working Set} = \lambda \bar{T} \times total\_len_{avg} / 2$$

- $\lambda$: arrival rate（agents/秒）
- $\bar{T}$: 平均 JCT
- $total\_len_{avg}$: 平均 trajectory 总 token
- 除 2: trajectory 渐进增长，平均长度约最终一半

DS 660B serving 实测：APS 0.1 时 working set 69 GB，APS 0.45 时 681 GB。

paper 还提醒：真实场景下 agent 间有 inter-arrival gap 和 tool call latency，JCT 会膨胀 $r$ 倍，APS 容量也涨 $r$ 倍，working set 涨 $r^2$ 倍。所以真实 working set 可能远超实验测的。这意味着 Mooncake 那种纯 DRAM pool 在长 agent 场景下可能不够用，SSD-based storage 是必要的。

参考: [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin), [TokenLake](https://arxiv.org/abs/2508.17219)

### 7.3 跟 Mooncake 的关系

paper 明确说：DualPath 可以跟 DRAM cache pool 结合，但"performance gain is marginal"。理由：working set 太大时 DRAM 装不下，hit rate 下降；小 working set 时 storage 已经够快，DRAM cache 帮助有限。

DualPath 的定位是**直接 attack storage I/O 的不对称**，跟 Mooncake 解决的问题（减少 storage 访问）正交。理论上可以叠，但 paper 没深做。

### 7.4 没采用 GPU Direct RDMA 的原因

paper §4.1 提到：理论上 DE buffer 可以用 GPU Direct RDMA bypass，省一次 H2D。但 agentic generation 通常很短，TTFT 占比大，DE buffer 能减少 GPU memory 占用，trade-off 划算。

这是个反直觉的设计选择——绕路反而更优，因为省 HBM 比 省 H2D 延迟更重要。

---

## 八、Intuition 总结

### 8.1 paper 的真正贡献

不是"dual path"这个 idea 本身多新颖，而是它**把一个被忽视的 asymmetry 暴露出来并系统化解决**。这个 asymmetry 在 chatbot 时代不显眼，agentic 时代变成头号瓶颈。

### 8.2 为什么 dual-path 成立的根本

两个 asymmetry 叠加：
1. **CNIC >> SNIC**（8:1 带宽比）→ 绕路几乎免费
2. **PE SNIC 满载, DE SNIC 闲置** → 有闲置资源可借

只要这两个 asymmetry 存在，dual-path 就能 work。未来硬件如果 SNIC 带宽追上 CNIC，或者 workload 特性变化让 DE SNIC 也忙起来，收益会减小。

### 8.3 工程上的精彩之处

1. **CNIC-centric**: 强制所有 traffic 走 CNIC，利用硬件 QoS 做 isolation。看似绕路，实际是当前唯一能严格隔离的方式。
2. **Layer Block + Full Block**: 用数据结构解耦 storage 和 transfer 的 granularity 需求。
3. **Token-count proxy**: 用一个简单指标平衡三个资源维度，避免复杂 multi-objective scheduling。
4. **Bottleneck-free 证明**: 给出 P/D ratio 的安全范围 [1/7, 3.5]，告诉 practitioner 什么时候该用 dual-path。

### 8.4 对未来的启示

paper §10 最后的 implication 比较深远：**disaggregated 架构里，不同 engine 类型的资源利用率不对称是普遍问题**。DualPath 是 prefill/decode 不对称的一个 instance。同样的 pattern 出现在：
- RL rollout：actor inference vs reward model
- 多模态：vision encoder vs LLM decoder
- Speculative decoding：draft model vs verify model

未来 systems 工作可以从"symmetry breaking in disaggregation"这个角度切入，DualPath 是一个具体且优雅的实例。

参考汇总：
- [DualPath 原文](https://arxiv.org/abs/2601.22705) (CONCUR 那篇也提到 agentic batch inference)
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [Strata (hierarchical caching)](https://arxiv.org/abs/2508.18572)
- [SGLang](https://docs.sglang.io/)
- [Sarathi-Serve (chunked prefill)](https://www.usenix.org/conference/osdi24/presentation/agrawal)
- [DeepInsights into DeepSeek-V3 hardware (ISCA'25)](https://dl.acm.org/doi/10.1145/3695054.3695059)
- [PagedAttention (vLLM)](https://dl.acm.org/doi/10.1145/3600006.3613165)

---

人话总结一句：**agent 时代 LLM 推理是被 storage I/O 卡死的，而现有架构一半 storage 带宽在睡大觉；DualPath 让闲着的 decode 节点帮忙读 KV-Cache，再用 GPU 间的高速网络顺手捎给 prefill 节点，等于白嫖一倍带宽，还顺带证明了在绝大多数配置下都不会引入新瓶颈**。

---

# DualPath: 深度技术解析

## 1. Problem Statement — 为什么这件事值得做

这篇paper直击一个在 agentic LLM inference 中被严重忽视的 bottleneck: KV-Cache 的 storage I/O。

传统 LLM inference 是 compute-bound 的, 但 agentic workload 完全改变了这个特性。让我们用数据说话:

### 1.1 Agentic Workload 的特征

paper Table 2 给出了 trace 数据:
- **MaxLen=64K**: 平均 157 turns, 每轮 append 仅 429 tokens, generation 176 tokens, 但 context 平均累积到 32,721 tokens
- KV-Cache hit rate: **98.7%**

这意味着什么? 每个 turn GPU 只需要对 ~429 个新 tokens 做 prefill computation, 但必须从 storage 加载 ~32k tokens 的 KV-Cache。compute 很少, I/O 巨大。

### 1.2 Cache-Compute Ratio

这是 paper 最关键的 insight 之一, Table 1:

```
Cache-compute ratio (GB per PFLOP):
Qwen2.5-32B (FP16): 117-267 GB/PFLOP
GPT-OSS-120B:      47-95 GB/PFLOP  
Qwen3-235B-A22B:   39-60 GB/PFLOP
DeepSeek-V3.2:     13-36 GB/PFLOP
DeepSeek-V3:       4.8-5.8 GB/PFLOP
```

即使像 DeepSeek-V3.2 这样用 MLA + sparse attention 已经高度优化 KV-Cache size 的模型, cache-compute ratio 仍是 13-36 GB/PFLOP。对 Qwen2.5-32B (FP16) 这种 GQA dense model, 高达 117-267 GB/PFLOP。

直觉: **GPU 算 1 PFLOP 的同时, 需要从 storage 读出 13~267 GB 的 KV-Cache**。而单节点的 SNIC (Storage NIC) 带宽是 400 Gbps ≈ 50 GB/s。这个 ratio 就决定了 system 严重 I/O-bound。

### 1.3 硬件演化趋势的不利因素

Figure 3 (左) 显示: 从 NVIDIA Ampere 到 Blackwell, I/O-compute ratio 下降了 **14.4×**。也就是 GPU FLOPS 增长远快于 NIC bandwidth 和 HBM capacity。

直觉: hardware roadmap 本质上是在加剧这个问题, 而不是缓解。未来的 GPU 会更"饿", 但 storage 带宽增长跟不上。

### 1.4 PD-Disaggregated 架构的不对称性

这是 paper 识别的核心 low-hanging fruit:

```
Traditional architecture:
  PE (prefill engine) ←── storage read (saturated SNIC) ── Storage
  DE (decode engine)  ←── idle SNIC ───────────────────── Storage
```

PE 必须加载大量 KV-Cache (因为 prefill 阶段需要 hit 的 cache), DE 的 SNIC 几乎闲置 (decode 阶段 KV-Cache 比较小且只需要 write-back)。

直觉: **一半的 storage 带宽在睡大觉, 另一半被打爆**。

---

## 2. DualPath 的核心 idea

### 2.1 Insight

KV-Cache loading **不一定要 prefill-centric**。可以让 KV-Cache 先进 DE, 再通过 compute network (CNIC) 用 RDMA transfer 到 PE。

```
DualPath architecture:
  Path 1 (traditional): Storage → PE buffer → PE HBM
  Path 2 (novel):       Storage → DE buffer → (RDMA over CNIC) → PE HBM
```

这利用了两个观察:
1. DE 的 SNIC 带宽是闲置的
2. CNIC 的 compute network 带宽远大于 SNIC (因为 model execution 的 collective communication 是 bursty 的, 间歇性空闲)

### 2.2 为什么这个 idea tricky — 三大 challenges

paper §4.3 列了三个 challenges, 这是真正工程上的硬骨头:

1. **Fine-grained data transfer**: layerwise prefill (LayerKV, PrefillOnly) 把 KV-Cache 切成 per-layer 的 fine-grained block。原来 N_layers 个 layer 的 KV-Cache 在一次 forward 中处理, 现在变成 N_layers 个独立的 transfer。block 数量 ×N_layers, 每个 block 大小 ÷N_layers。对 RDMA 提交延迟极其敏感。

2. **Traffic isolation**: dual-path 引入的 KV-Cache traffic 会跟 model execution 的 collective communication (比如 MoE 的 AllToAll, TP/CP 的 ReduceScatter/AllGather) 抢 PCIe 带宽。collective 是亚毫秒级 burst, software-based traffic shaping 根本来不及反应。

3. **Dynamic load balancing**: 在线决策每个 request 走哪条 path, 同时平衡 GPU compute、storage NIC、compute NIC 三个维度的负载。

---

## 3. System Architecture 详解

### 3.1 整体组件

```
DualPath Components:
├── Inference Engines
│   ├── Prefill Engines (PEs)
│   └── Decode Engines (DEs)
├── Traffic Manager (per engine)
│   ├── H2D / D2H copies
│   ├── PE ↔ DE KV-Cache transfers (RDMA)
│   └── Storage reads/writes (via SNIC)
└── Request Scheduler (central)
    ├── Inter-engine scheduling (request → (PE, DE) + path)
    └── Intra-engine scheduling (request → batch)
```

### 3.2 Dual-Path Loading 的数据流 (§4.1, Figure 4)

这是 paper 的核心机制。需要分别看 PE read path 和 DE read path。

#### PE Read Path (Figure 4a)

```
Step 1-2: Storage → PE Buffer (via SNIC)
Step 3-4: PE Buffer → PE HBM (compute attention for miss tokens)
Step 5-7: PE HBM → DE Buffer (via CNIC RDMA, layerwise streaming)
重复 n_layer 次
Step 8-9: DE Buffer → DE HBM (H2D, after prefill done)
```

关键: **3-7 这四步在每个 layer 上重复**, transfer 和 computation overlap。这是 layerwise prefill 的精髓 —— HBM 同时只装一个 layer 的 KV-Cache, 把 effective batch size 放大 ~N_layers 倍。

#### DE Read Path (Figure 4b)

```
Step 1-2: Storage → DE Buffer (via DE's SNIC, 关键!)
Step 3-5: DE Buffer → PE HBM (via CNIC RDMA, layerwise streaming, overlap with compute)
重复 n_layer 次
Step 6-7: miss tokens 的 KV-Cache from PE → DE Buffer (合并)
         DE Buffer → DE HBM (H2D)
```

注意 DE read path 的一个微妙之处: 它把 KV-Cache 直接从 storage 读到 DE, **然后通过 CNIC 反向送到 PE**。这相当于让 DE 的 SNIC 帮 PE 干活, 用 CNIC 兜底。

#### Decode Phase 共同部分

```
DE HBM ← DE Buffer (H2D)
Release CPU memory
Decode (generate tokens)
When 64 tokens accumulated → persist to disk
```

paper 特别提到一个 trade-off: 理论上可以用 GPU Direct RDMA bypass DE buffer, 但 generation 通常很短, TTFT 占比大, 引入 DE buffer 能减少 GPU memory 占用, 值得。

#### Block Layouts (§A.5)

paper 设计了两种 block:
- **Layer Block**: shape `[1, block_size, cache_bytes]`, 单层 KV-Cache
- **Full Block**: shape `[n_layer, block_size, cache_bytes]`, 所有层

设计动机: layerwise prefill 把 KV-Cache block 切成 1/n_layer 大小, 但 block 数量 ×n_layer。storage 层使用 Full Block, transfer 和 compute 时使用 Layer Block。两者之间通过简单 `concat` n_layer 个 Layer Block 得到 Full Block, **避免手动 KV-Cache memory layout 转换**。

直觉: storage 适合大块 I/O (throughput), transfer 适合小块 I/O (overlap with compute)。layout 解耦让两层各取所需。

---

## 4. Bottleneck-Free Analysis (§4.2) — 这是 paper 最 math-heavy 的部分

paper 给了一个理论分析, 证明在 "合理" 的 P/D ratio 下, dual-path 不会引入新的 bottleneck。

### 4.1 Notation

- $P, D$: prefill 和 decode 节点数
- $g$: 每个节点的 GPU 数 (Hopper DGX 是 8)
- $s$: 每个节点的 storage NIC 数 (一般 1)
- $B$: 每个 CNIC 的带宽 (400 Gbps ≈ 50 GB/s)
- $M$: 每机器的 DRAM 带宽

### 4.2 Traffic per PE-DE pair

假设 load-balanced, storage NIC 带宽均分。

**PE read path 流量 per pair**:
$$T_p = \frac{B \cdot s}{D \cdot g^2}$$

变量含义: $B$ 是 CNIC 带宽, $s$ 是 storage 带宽除以 CNIC 带宽的比例系数。下标 $p$ 表示 prefill path。$D \cdot g^2$ 是因为 $D \cdot g$ 个 DE 和 $P \cdot g$ 个 PE, 每个 PE 跟每个 DE 都有 pair, 但流量均摊。

**DE read path 流量 per pair**:
$$T_c = \frac{B \cdot s}{P \cdot g^2}$$

下标 $c$ 表示 "cross" path (DE 协助)。

### 4.3 PE CNIC Bandwidth 分析

#### Read 方向 (PE 路径 3 和 5)

总流量 across all pairs:
$$2 \times T_p \times D \cdot g = \frac{2 B s}{g} \leq B \quad \text{(1)}$$

变量: $T_p$ 是 per-pair 流量, 乘以 $D \cdot g$ (DE 总数) 得到总流量, 乘以 2 是因为路径 3 和路径 5 都用 PE CNIC。

化简: 因为实际中 $s \leq g$ (storage NIC 数 ≤ GPU 数), 所以 read 方向永远 bottleneck-free。

#### Write 方向 (PE 路径 4 和 DE 路径 5)

$$\left(T_p + T_c\right) \times D \cdot g = \frac{B s}{g} \left(1 + \frac{D}{P}\right) \leq B \quad \text{(2)}$$

化简得到:
$$\frac{P}{D} \geq \frac{s}{g - s} \quad \text{(3)}$$

直觉: $s/(g-s)$ 是 lower bound。当 $s=1, g=8$ 时, $P/D \geq 1/7$, 几乎总是满足。

### 4.4 DE CNIC Bandwidth 分析

#### Read 方向 (PE 路径 8, DE 路径 3/6)

$$\left(T_p + T_c \times 2\right) \times P \cdot g = \frac{s}{g} \left(\frac{P}{D} + 2\right) B \leq B \quad \text{(4)}$$

化简:
$$\frac{P}{D} \leq \frac{g - 2s}{s} \quad \text{(5)}$$

#### Write 方向 (PE 路径 7/9, DE 路径 7)

$$\left(2 T_p + T_c\right) \times P \cdot g \leq B \quad \text{(6)}$$

化简:
$$\frac{P}{D} \leq \frac{g - s}{2s} \quad \text{(7)}$$

### 4.5 DRAM Pressure

PE MEM 压力 $2Bs$, 一般不会爆。

DE MEM 压力 $(3 + 2P/D) Bs$, 要求 ≤ $M$:
$$\frac{P}{D} \leq \frac{M/(Bs) - 3}{2} \quad \text{(8)}$$

### 4.6 综合结论

$$\frac{s}{g-s} \leq \frac{P}{D} \leq \min\left\{\frac{g-2s}{s}, \frac{g-s}{2s}, \frac{M/(Bs)-3}{2}\right\} \quad \text{(9)}$$

对 $(g=8, s=1)$, $M \approx 500$ GB/s, $Bs \approx 50$ GB/s:
$$\frac{1}{7} \leq \frac{P}{D} \leq \frac{7}{2}$$

**直觉**: P/D ratio 在 [1/7, 3.5] 之间时, dual-path 不会引入新 bottleneck。这覆盖了几乎所有 practical 配置 (1P1D, 2P4D, 4P8D 等都在范围内)。

---

## 5. CNIC-Centric Traffic Manager (§5) — 工程细节

### 5.1 为什么不用 GPUDirect Storage / CUDA Copy Engine

paper §5.2 解释了为什么不用现成的数据传输技术:

- **GPUDirect Storage**: 直接从 storage 读到 GPU HBM。问题: 走独立 path, 不共享 compute network 的 QoS, 会跟 collective communication 抢 PCIe 带宽。GPU 不支持 PCIe QoS (引用 [39] 的 paper), 无法隔离。

- **CUDA Copy Engine**: 直接 host DRAM → GPU HBM。问题: 单次 cudaMemcpyAsync 延迟 $5-7 \mu s$ (paper 实测)。

而 **CNIC-assisted H2D/D2H** (用 RDMA Write 到 GPU 的 paired CNIC 做本地拷贝):
- 单次 RDMA Write work request 只需 ~$1 \mu s$ (只需 user-space mmio 写 NIC 寄存器)
- Doorbell batching 可进一步摊销提交开销 ([25])

直觉: 看起来绕了一圈 (storage → DRAM → CNIC → GPU), 但这是当前唯一能实现严格 traffic isolation 的方法。

### 5.2 Traffic Isolation 机制

**InfiniBand 配置** (§A.1):
```
qos_max_vls 4                        # 4 个 virtual lanes
qos_high_limit 240                   # high-priority arbiter 阈值
qos_vlarb_high 0:192,1:192,2:0,3:192 # high-priority WRR weights
qos_vlarb_low 0:192,1:192,2:64,3:192 # low-priority WRR weights
```

VL0,1,3 给高优先级 (model inference), VL2 给低优先级 (KV-Cache)。高优先级占 ~99% 带宽, 低优先级 ~1% (防 starvation)。

**RoCE 等价配置**: 用 DSCP + TC (Traffic Class) + PFC (Priority Flow Control) 实现相同效果。

直觉: 利用 InfiniBand/RoCE 硬件级 QoS, 让 model execution 的 collective communication 几乎不受影响, 同时让 KV-Cache traffic 捡漏用空闲带宽。

---

## 6. Adaptive Request Scheduler (§6)

### 6.1 Inter-Engine PE Scheduling (Algorithm 1)

每个 PE engine $e$ 上报:
- $seq_e$: 未完成 request 数
- $tok_e$: 这些 request 的总 token 数
- $q_{n(e)}$: 所属节点 $n(e)$ 的 disk read queue 长度

两个常量:
- $\alpha$: short reading queue 阈值 (3 秒能读的 token 数)
- $\beta$: unfinished token 上限 (5 秒 GPU 能算的 token 数)

**三类 PE**:
- $C_1$: $tok_e > \beta$ (overloaded, 不分配)
- $C_2$: $q_{n(e)} \leq \alpha$ 且 $tok_e \leq \beta$ (短 disk queue, 优先)
- $C_3$: $q_{n(e)} > \alpha$ 且 $tok_e \leq \beta$ (长 disk queue, 次优)

直觉: 既看 GPU 负载 ($tok_e$), 又看 disk queue 长度 ($q_{n(e)}$)。前者防止 GPU 过载, 后者确保 SNIC 不闲置。

### 6.2 DE Scheduling (两阶段)

**Phase 1 (across groups)**: 把 request 分给 $tok$ 总和最小的 group (balance NIC + GPU load)

**Phase 2 (within group)**:
- 计算 group 内剩余 HBM 总和
- 高 token 阈值: $Z = 1.05 \times (\sum_{r \in R} len_r + \sum_{e \in E} tok_e) / |E|$
  - $R$: 可调度 request 集合
  - $E$: group 内 engines
  - $|E|$: engine 数
  - 1.05 是 5% buffer

两类 DE:
- $tok_e + len(r) > Z$: 高 token DE (避免)
- 其他: 优先 (其中选 $tok_e$ 最小的)

直觉: 在 group 内做负载均衡, 5% buffer 留出容差, 避免 fragmentation。

### 6.3 KV-Cache Read Task Scheduling

简单策略: 选 PE 和 DE 中 reading queue 较短的一边读。

paper 承认可以更优 (split request 到两条 path 并行读), 留作 future work。

### 6.4 Intra-Engine PE Scheduling (§6.2)

目标: 在 attention 层做 data parallelism 时, 让所有 GPU 的 attention execution time 接近, 减少同步 bubble。

**Layer Time Estimation**: 每个 request 描述为 $(length, miss)$:
- $length$: 有 KV-Cache 的 token 数
- $miss$: 需要 compute 的 token 数

通过预先 profile 得到 attention 层执行时间与 (length, miss) 的关系, 用 FIFO packing 决定 batch 大小, 不超过 "compute quota" (300ms)。

binary search 找 chunked prefill 的 $miss'$ 来填满剩余 quota。

直觉: 类似 bin packing, 但目标是 execution time 而非简单 token 数。Figure 6 右图展示了效果: 应用前 GPU 间 attention 时间差异大, 应用后 Max/Avg ratio 维持在 1.06。

---

## 7. Experimental Results 详解

### 7.1 Setup

- **Hardware**: 8× NVIDIA Hopper GPU/node, 8× 400Gbps RDMA NIC + 1× storage NIC, InfiniBand
- **Storage**: 3FS (DeepSeek 的分布式 FS, 无 DRAM cache, 能 saturated 400Gbps)
- **Models**: 
  - DeepSeek-V3.2 660B (MLA + sparse attention + MoE)
  - DS 27B (660B 的缩小版, 同架构)
  - Qwen2.5-32B (dense + GQA)
- **Workload**: 3 个 agent trace 数据集 (32K/48K/64K MaxLen), 500 trajectories each

### 7.2 Offline Inference (§7.3, Figure 7)

**DS 660B (Figure 7 middle)**:
- DualPath vs Basic: **最高 1.87× speedup**
- DualPath 接近 Oracle (zero I/O overhead), 说明 KV-Cache I/O 几乎被消除

**DS 27B (Figure 7 top)**:
- DualPath vs Basic: 最高 1.78× speedup
- 但 DualPath 仍比 Oracle 慢 1.09-1.85× (1P1D 配置 storage 带宽有限)

**Qwen 32B (Figure 7 bottom)**:
- 趋势类似 DS 27B

### 7.3 Append/Generation Length 扫描 (Figure 9)

**Append length scaling (左图)**:
- 随着 append length 增加, Basic 性能逐渐接近 DualPath 和 Oracle
- DualPath vs Basic: **1.82-1.99× speedup** (各 scale)

直觉: append 越长 → GPU compute pressure 越大 → I/O 占比下降 → DualPath 优势减小。但即使在 compute-bound 区域, DualPath 仍保持 ~1.8× speedup, 因为还是有一些 I/O。

**Generation length scaling (右图)**:
- 趋势类似

直觉: generation 越长 → prefill gap 时间越长 → KV-Cache loading 压力减小 (有更多时间 overlap) → DualPath 优势减小。

### 7.4 P/D Ratio 扫描 (Figure 8, DS 27B)

| 配置 | 等效 storage 带宽 | 性能关系 |
|------|------------------|----------|
| Basic 1P1D | 1 节点 SNIC | ≈ Basic 1P2D |
| DualPath 1P1D | 2 节点 SNIC | ≈ Basic 2P1D |
| DualPath 2P1D | 3 节点 SNIC | ≈ DualPath 1P2D |

平均 speedup **1.64×** (最高 2.46×)。

直觉: 这张表验证了 storage bandwidth 才是 dominant bottleneck, 不是 GPU 数量。

### 7.5 Online Serving (§7.4, Figure 10-12)

**SLO**: TTFT ≤ 4s, TPOT ≤ 50ms

**APS (agents per second) 容量**:
- DS 27B: DualPath **1.67×** Basic
- DS 660B: DualPath **2.25×** Basic

**TTST**: DualPath ≈ Basic (无 decode 开销)

**TPOT**: DualPath ≈ Basic (无额外 decode 开销)

但 DS 27B 上 DualPath 和 Basic 的 TPOT 都比 Oracle 高, 说明小模型的 P-D transfer overhead 仍可观, paper 留作 future work。

**TTFT Breakdown (Figure 12 左)**:
- DualPath 的 TTFT 组成稳定, 不随 APS 增长
- Basic 的 queue time 随 APS 急剧增长 (storage 带宽不足导致排队)

### 7.6 Ablation Study (§7.5, Figure 12 右)

逐步添加技术, DS 660B, 64K context:

| 阶段 | JCT 减少 (vs Basic) |
|------|---------------------|
| + Layerwise prefill | -17.21% |
| + Dual-path loading | -38.19% (累计) |
| + Scheduling | -45.62% (累计) |

直觉:
- Layerwise prefill 缓解 HBM 瓶颈, hide transfer overhead
- Dual-path loading 是主要 gain, 因为真正聚合了所有 SNIC 带宽
- Scheduling 进一步优化, 通过智能选 path

### 7.7 Load Balance (Figure 13, 14)

**Storage NIC traffic balance** (Max/Avg ratio):
- Round robin: 1.53
- DualPath scheduling: **1.18** (接近 1.0 perfect)

**Attention execution time balance** (Max/Avg, 前 5% task):
- DualPath: **1.06** (基本完美)

### 7.8 Large-Scale Scalability (§7.6, Table 3, Figure 15)

**Offline**: 2P4D (2K agents) → 48P96D (48K agents)
- JCT: 3,167s → 3,201s (近线性扩展, **24× 规模几乎不增 JCT**)

**Online**: 2P4D (0.4 APS) → 44P88D (8.8 APS)
- TTFT: 1.739s → 1.847s
- TPOT: 0.039s → 0.036s
- **22× 吞吐量提升**, latency 几乎不变

Scheduler CPU usage < 10 cores, 不是瓶颈。

---

## 8. Working Set Analysis (§8.2)

paper 给了一个有趣的 working set 估算:

$$\text{Working Set} = \lambda \bar{T} \times total\_len_{avg} / 2$$

变量:
- $\lambda$: arrival rate (agents/second)
- $\bar{T}$: 平均 JCT
- $total\_len_{avg}$: 平均 trajectory 总 token 数
- 除以 2: 因为 trajectory 是渐进增长的, 平均长度约为最终长度一半

DS 660B serving:
- APS 0.1: working set 69 GB
- APS 0.45: working set **681 GB**

直觉: 这已经远超单节点 DRAM (1.5TB DGX H100), 但跟 cluster 总 DRAM 比仍小。paper 指出实际场景中 working set 会更大 (因为 inter-arrival 和 tool call latency 让 JCT 增长 r 倍, working set 增长 $r^2$ 倍)。

---

## 9. 与 Related Work 的对比

| System | KV-Cache 存储 | Path | DRAM usage |
|--------|---------------|------|------------|
| Mooncake [38] | 分布式 DRAM pool | storage→prefill | 高 |
| TokenLake [46] | unified segment pool | storage→prefill | 中 |
| Strata [49] | hierarchical storage | 单 path + GPU-assist | 中 |
| KVPR [23] | recomputation overlap | 单 path | 低 |
| TailorKV [53] | layer-granular hybrid quant | 单 path | 低 |
| **DualPath** | **storage (3FS) + 可选 DRAM** | **dual-path (PE+DE)** | **低** |

直觉: DualPath 的核心差异化是"利用现有闲置带宽", 而非"减少要传的数据"。其他工作大多在单 path 上做优化, 没有触及 PE/DE 间的不对称性。

参考链接:
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)
- [Mooncake paper](https://www.usenix.org/conference/fast25/presentation/qin)
- [DistServe (PD disaggregation)](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [SGLang](https://docs.sglang.io/)
- [3FS - DeepSeek 的分布式 FS](https://github.com/deepseek-ai/3FS)
- [LayerKV](https://arxiv.org/abs/2410.00428)
- [PrefillOnly (SOSP'25)](https://dl.acm.org/doi/10.1145/3694715.3695963)
- [DeepEP - expert parallel communication](https://github.com/deepseek-ai/DeepEP)
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [Strata: hierarchical context caching](https://arxiv.org/abs/2508.18572)
- [DeepSeek-V3.2 release](https://arxiv.org/abs/2512.02556)
- [Sarathi-Serve (chunked prefill)](https://www.usenix.org/conference/osdi24/presentation/agrawal)
- [DeepInsights into DeepSeek-V3 hardware (ISCA'25)](https://dl.acm.org/doi/10.1145/3695054.3695059)
- [Splitwise (PD disaggregation, ISCA'24)](https://dl.acm.org/doi/10.1145/3695054.3695065)

---

## 10. 我的 Intuition 总结

### 10.1 这篇 paper 的 "aha moment"

paper 真正的贡献是把一个被忽视的 asymmetry 暴露出来: **PE 的 SNIC 被打爆, DE 的 SNIC 在睡大觉**。这个 asymmetry 在 chatbot 时代不明显 (单 turn, KV-Cache 不大), 但 agentic 时代 (多 turn, 长 context, 高 hit rate) 变成 dominant bottleneck。

### 10.2 为什么 dual-path 真的 work

关键不在"dual path"本身, 而在于两个被 paper 隐式利用的特性:

1. **CNIC 带宽 >> SNIC 带宽**: 一个 DGX 节点有 8 个 400Gbps CNIC (3200Gbps 总 east-west), 但只有 1 个 400Gbps SNIC。DE → PE 的 KV-Cache transfer 用 CNIC, 几乎不构成 bottleneck。

2. **collective communication 是 bursty 的**: MoE 的 AllToAll、TP 的 ReduceScatter 都是亚毫秒级 burst, 中间有大段空闲。dual-path 的 KV-Cache traffic 捡漏用这些空闲带宽, 通过 VL/TC 的硬件 QoS 严格隔离, 不影响 model execution。

### 10.3 Layerwise prefill 是 enabler, 不是主角

paper ablation 显示 layerwise prefill 只贡献 17.21% JCT 减少, dual-path loading 贡献了 38.19%。但 layerwise prefill 是 dual-path 的 enabler —— 它把 KV-Cache 切成 per-layer block, 才能做 storage/HBM/transfer 的 pipeline overlap。没有 layerwise prefill, dual-path 的 overlap 难以实现。

### 10.4 Bottleneck-free analysis 的妙处

paper §4.2 的理论分析揭示了一个有趣的现象: **dual-path 不需要复杂的 coordination 就能 self-balance**。原因是 storage 带宽是瓶颈, 但它被均摊到所有节点的 SNIC 上, 而 CNIC 的 bandwidth 比 SNIC 大 ~g 倍 (8 倍), 所以"绕路"通过 CNIC 几乎免费。这个 asymmetry (CNIC >> SNIC) 是 dual-path 成立的根本。

### 10.5 限制和 future work

paper 自己也指出几个未解问题:
1. **大集群下的 P/D ratio tuning**: 48P96D 没有比 multiple small units 多 gain, 但有 fragmentation 和 bursty 处理优势。
2. **Working set 超出 DRAM 时 Mooncake 风格的 DRAM pool 是否有用**: paper 说 "performance gain is marginal", 但在 $r > 1$ 的真实场景下结论可能不同。
3. **小模型的 P-D transfer overhead**: DS 27B 上 DualPath 和 Basic 的 TPOT 都比 Oracle 高, 说明 P-D KV-Cache transfer 在小模型上占比大。这可能需要 GPU Direct RDMA 绕过 DE buffer (paper 提到但没采用)。
4. **Split request 到 dual path 并行读**: 当前是选一条 path, 没充分利用并行性。

### 10.6 更广的 implication

DualPath 揭示了一个 pattern: **在 disaggregated 架构中, 不同 engine 类型的资源利用率不对称是普遍问题**。这个 insight 可以推广到:
- **RL rollout 阶段**: actor 模型 inference 和 reward model 的资源不对称
- **多模态 inference**: vision encoder 和 LLM decoder 的不对称
- **Speculative decoding**: draft model 和 verify model 的不对称

未来 systems 工作可以从 "disaggregation symmetry breaking" 这个角度切入, dual-path 是一个具体 instance。

---

## 11. 公式变量速查表

为方便你 build intuition, 整理所有公式变量:

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $P$ | Prefill 节点数 | 2 |
| $D$ | Decode 节点数 | 4 |
| $g$ | 每节点 GPU 数 | 8 (Hopper DGX) |
| $s$ | 每节点 storage NIC 数 | 1 |
| $B$ | 单个 CNIC 带宽 | 400 Gbps ≈ 50 GB/s |
| $M$ | 每节点 DRAM 带宽 | ~500 GB/s |
| $T_p$ | PE read path per-pair 流量 | $Bs/(Dg^2)$ |
| $T_c$ | DE read path per-pair 流量 | $Bs/(Pg^2)$ |
| $\alpha$ | short reading queue 阈值 | 3秒可读 token 数 |
| $\beta$ | unfinished token 上限 | 5秒 GPU 可算 token 数 |
| $Z$ | group 内高 token 阈值 | $1.05 \times \text{avg}$ |
| $seq_e$ | engine $e$ 未完成请求数 | - |
| $tok_e$ | engine $e$ 未完成 token 数 | - |
| $q_{n(e)}$ | 节点 $n(e)$ disk read queue 长度 | - |
| $n_{layer}$ | 模型层数 | DS-V3 是 61 |
| $length$ | request 中有 KV-Cache 的 token 数 | - |
| $miss$ | request 中需 compute 的 token 数 | - |

---

希望这个深度解析能让你 build 起对 dual-path 设计、bottleneck-free 分析、scheduler 逻辑、以及 experiment 结果的完整 intuition。如果某个具体细节还想展开 (比如 InfiniBand VL arbiter 的具体行为、doorbell batching 的实现、或 3FS 的存储布局), 随时告诉我。
