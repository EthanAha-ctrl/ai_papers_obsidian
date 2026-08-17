---
source_pdf: Prefill-as-a-Service KVCache of Next-Generation.pdf
paper_sha256: 153dcaa3f6349319372df95746a0c9fb143ca93085b1703d90a1c32106e3b222
processed_at: '2026-08-06T05:46:42-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，让我们抛开学术黑话，用最直白的人话把这篇 paper 扒开揉碎了讲。

核心直觉非常漂亮：LLM 推理分为 Prefill（读 prompt，疯狂算力消耗）和 Decode（吐答案，疯狂访存消耗）。以前大家必须把 Prefill 和 Decode 捆在同一个机房里用 RDMA 豪华网络连着，因为 Prefill 产生的 KVCache 太庞大了，跨机房搬运根本搬不起。现在 hybrid attention 模型把 KVCache 压缩了一个数量级，我们终于可以“外包”了——把长文本的 Prefill 扔给远端算力超强但网络一般的机房，算完把轻量级的 KVCache 通过普通 Ethernet 传回来做 Decode。

为了 build your intuition，我们分几个层次把技术细节抠干净。

### 1. 为什么以前不能跨机房？The Bandwidth Wall

想象 Prefill 就像读书做笔记。Dense attention（比如 GQA）就是一个话痨，每读一个 token 就要写一大段笔记。Decode 就是看着笔记写作文。

如果我们在远端建一个超强算力的“外包书房”专门做 Prefill，算完后必须把笔记快递回本地。问题在于，这台 GPU 产生笔记的速度太快了。我们用 KV throughput 来衡量这个“水流速度”：

$$\Phi_{\mathrm{kv}}(l) = \frac{S_{\mathrm{kv}}(l)}{T_{\mathrm{prefill}}(l)}$$

变量解释：
- $\Phi_{\mathrm{kv}}(l)$: 长度为 $l$ 的请求，其 Prefill 阶段产生 KVCache 的速度（单位 Gbps）。
- $S_{\mathrm{kv}}(l)$: 该请求产生的 KVCache 总大小。
- $T_{\mathrm{prefill}}(l)$: Prefill 计算花费的时间。
- $l$: 请求的 input length。

对于 dense model 如 MiniMax-M2.5，在 32K tokens 时，$\Phi_{\mathrm{kv}} \approx 60$ Gbps。单台机器的跨机房网线通常也就 100 Gbps，几台机器一起发包就把跨机房带宽塞爆了。所以传统架构必须把 Prefill 和 Decode 关在同一个数据中心里，用 800 Gbps 的 RDMA 网络连着。

### 2. Hybrid Attention 怎么救场的？笔记变小了

Hybrid model 的设计就像一个学霸。它大部分层用 linear attention 或 SWA，这些层不存逐 token 的长篇大论，只在脑子里维护一个固定大小的 recurrent state（可以理解为一句高度压缩的总结）。只有少数的 full attention 层才存详细的 block-level 笔记。

看 Table 1 里的配置，Ring-2.5-1T 用了 7:1 的比例（7层 linear 配 1层 full）。这意味着 8 层里只有 1 层产生随长度增长的 KVCache，其余 7 层的 state 大小是固定的。加上 MLA 压缩，总体积缩小了 36 倍。

再看公式 (1) 的动态变化：随着 $l$ 增加，$T_{\mathrm{prefill}}(l)$ 是近二次方增长的，而 $S_{\mathrm{kv}}(l)$ 是线性增长的。所以 $\Phi_{\mathrm{kv}}$ 随着长度增加反而**下降**。这在 Table 5 中得到了验证：1K tokens 时 $\Phi_{\mathrm{kv}} = 3.61$ Gbps，128K tokens 时降到了 2.62 Gbps。

**关键直觉**：越长的请求，Prefill 计算时间越久，但因为 hybrid 架构，KVCache 没涨多少。这导致长请求产生 KVCache 的“速率”很低。跨机房网线最怕高并发的高速水流，长请求恰好是涓涓细流。

### 3. PrfaaS 调度：不能全外包，只能选长的

既然笔记变小了，能不能把所有活儿都扔给远端 PrfaaS 机房？不行。

短请求的 Prefill 算得太快了，瞬间产生一小坨 KVCache，$\Phi_{\mathrm{kv}}$ 极高。如果把成千上万个短请求也外包，网线一样爆掉，而且远端高算力 GPU 处理短请求时由于 arithmetical intensity 太低，根本跑不满，纯属浪费。

所以 PrfaaS 的核心系统设计是：**Selective Offloading（选择性外包）**。

设一个 routing threshold $t$。增量 prefill 长度 $l > t$ 的请求，扔给远端 PrfaaS；$l \le t$ 的请求留在本地 PD cluster 处理。

我们看系统 throughput 模型（公式 3）：

$$\Theta_{\mathrm{prfaaS}} = \min\left(\frac{N_{\mathrm{prfaaS}}}{T_{\mathrm{prefill}}(l_{\mathrm{long}})}, \frac{B_{\mathrm{out}}}{S_{\mathrm{kv}}(l_{\mathrm{long}})}\right)$$

变量解释：
- $\Theta_{\mathrm{prfaaS}}$: 远端 PrfaaS 集群的产出速率。
- $N_{\mathrm{prfaaS}}$: 远端 GPU 实例数。
- $l_{\mathrm{long}}$: 被扔给远端的那些请求的平均长度。
- $B_{\mathrm{out}}$: 跨机房网线带宽。
- $S_{\mathrm{kv}}(l_{\mathrm{long}})$: 这些长请求的 KVCache 大小。

前半部分是 GPU 的算力上限，后半部分是网线的带宽上限。系统吞吐被这两个里较小的那个卡死。调度的目标是让两边都刚好跑满。

在 Case Study 中，系统找出的最优解是 $t = 19.4K$。大约 50% 的请求被扔给了 PrfaaS，这些被扔过去的请求平均长度是 44K。最终测下来，远端 egress 带宽只用了 13 Gbps！100 Gbps 的网线还有巨大余量。

### 4. Hybrid Prefix Cache Pool：两种笔记的收纳术

这里有个极好的系统设计细节。Hybrid 模型有两种 KVCache，传统系统处理不了。

- **Linear/SWA state**: request-level。它的大小和文本长度无关，是一个固定大小的 vector。只有当另一个请求的 prefix 和它**完全一致**时才能复用。
- **Full attention KVCache**: block-level。随长度线性增长，支持 partial prefix matching（前面对齐了就能借去用）。

PrfaaS 基于 [vLLM hybrid manager](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/) 设计了一个统一的 block pool。两类 cache 虽然逻辑不同，但都从同一个池子分配显存。

同时，为了跨机房传输，他们把 block 分成两类：
- **Prefix-cache blocks**: 留在本地供多个请求复用的缓存。
- **Transfer-cache blocks**: 专门用来接 PrfaaS 算完传回来的 KVCache。一旦传给本地 decode 节点用完，直接丢弃，因为远端不负责 decode，存着没用。

### 5. Dual-Timescale Scheduling：长短期的两手准备

跨机房网络是波动的，请求也是突发的。PrfaaS 设计了双时间尺度的调度。

**Short-term (短期，秒级)**: 监控 egress 带宽和队列深度。如果发现网络要堵了，动态提高 threshold $t$，让更多的请求留在本地，减少跨机房流量。对于有 prefix cache 命中的请求，计算 $l_{\mathrm{total}} - l_{\mathrm{pd}}$，如果本地缓存能覆盖大部分，就不扔给远端了。如果带宽充裕，甚至可以把远端的 prefix cache 传回本地来省计算。

**Long-term (长期，分钟/小时级)**: 监控三个角色（PrfaaS, PD-P 本地 prefill, PD-D 本地 decode）的 queue depth。如果发现 decode 积压了，就在本地 PD cluster 内部把一些 prefill 节点动态转换成 decode 节点，调整 $N_p / N_d$ 比例，重新满足公式 (8) 的均衡条件 $\Theta_{\mathrm{prfaaS}} + \Theta_{\mathrm{pd-p}} = \Theta_{\mathrm{pd-d}}$。

### 6. 实验 Case Study 结果深度解读

他们用 1T 参数的内部 hybrid 模型（Kimi Linear 架构，KDA:MLA = 3:1）做实验。
配置：远端 32 个 H200（算力强），本地 64 个 H20（显存带宽好，便宜）。基线是 96 个 H20 同构集群。

看 Table 6 的数据：
- PrfaaS 配置：$N_{\mathrm{prfaaS}}/N_p/N_d = 4/3/5$。吞吐量 3.24 req/s。
- Homogeneous 基线：$N_p/N_d = 9/3$。吞吐量 2.11 req/s。
- Naive Heterogeneous（全扔远端无调度）：$N_{\mathrm{prfaaS}}/N_d = 4/18$。吞吐量 2.45 req/s。

**直觉解读**：
1. PrfaaS 把长请求扔给 H200 算，本地 H20 只需 3 个 prefill 实例就够了，腾出大量机器做 decode，吞吐量提升 54%。
2. Naive 方案不筛长短请求，全扔给远端 H200。结果远端算得飞快（2.45 req/s），把本地 H20 的 decode 喂撑了，本地 decode 成为巨大瓶颈（需要 18 个 decode 实例），吞吐量只提升了 16%，白白浪费了 H200 的算力。
3. TTFT (Time To First Token) 大幅下降。长请求在基线下要和短请求抢本地 prefill 资源，排队很长。现在直接去远端 H200 走 VIP 通道，加上网络传输时间，整体 Mean TTFT 从 4.44s 降到 2.22s，P90 TTFT 从 9.73s 降到 3.51s。

### 7. 我的延伸联想与 Future Implications

顺着这个逻辑往下推演，这不仅仅是一个 serving 系统的优化，这可能重塑 AI infra 的物理拓扑。

1. **Hardware Decoupling 终于可行了**：NVIDIA 的 [Rubin CPX](https://developer.nvidia.com/blog/nvidia-rubin-cpx-accelerates-inference-performance-and-efficiency-for-1m-token-context-workloads/) 专攻 prefill 算力，[Groq LPU](https://groq.com/blog/the-groq-lpu-explained) 专攻 decode 带宽。以前要把这两种奇葩芯片塞进同一个机架里搞 RDMA，工程难度极大。现在可以建“Rubin CPX 算力中心”在冰岛，建“Groq decode 中心”在纽约，中间拉根普通光缆连着就行了。
2. **Agentic Workloads 的天然契合**：Agent 的工作流往往是多轮对话，context 越来越长，但每一轮只增加几个 token。在 PrfaaS 架构下，庞大的 prefix 已经在本地 decode cluster 有缓存，或者在远端有缓存。只需要把那一小段 incremental prefill 扔过去算，传回来的增量 KVCache 极小。这简直就是为 Agent 量身定制的。
3. **KVCache Compression 的协同效应**：如果把 PrfaaS 和 [KIVI](https://arxiv.org/abs/2402.02750) 2-bit 量化或者 [CacheGen](https://arxiv.org/abs/2310.07240) 压缩结合，跨机房带宽需求还能再砍一半。那我们甚至可以用普通的互联网宽带把 1M context 的 KVCache 传回家里。
4. **经济学意义的改变**：算力贵，带宽也贵，但两者的曲线不同。Hybrid 模型把 KVCache 压缩到 13 Gbps 这个量级，意味着跨机房传输的成本只占极小一部分。云厂商可以构建巨大的中央 prefill 算力池，然后在各个 region 部署轻量级的 decode 集群，实现算力的规模化经济。

### 参考链接与延伸阅读

*   **PrfaaS 基础架构依赖**:
    *   [Mooncake: KVCache-centric Disaggregated Architecture](https://arxiv.org/abs/2407.00079)
    *   [vLLM Hybrid Models Support](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/)
*   **Hybrid Attention 模型架构**:
    *   [Kimi Linear (KDA)](https://arxiv.org/abs/2510.26692)
    *   [DeepSeek-V2 (MLA)](https://arxiv.org/abs/2405.04434)
    *   [Gated Delta Networks](https://openreview.net/forum?id=22PQY4rkJE)
    *   [Longformer (SWA)](https://arxiv.org/abs/2004.05150)
*   **硬件演进趋势**:
    *   [NVIDIA Rubin CPX](https://developer.nvidia.com/blog/nvidia-rubin-cpx-accelerates-inference-performance-and-efficiency-for-1m-token-context-workloads/)
    *   [Groq LPU Architecture](https://groq.com/blog/the-groq-lpu-explained)
*   **KVCache 压缩与重用**:
    *   [KIVI: 2-bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)
    *   [CacheBlend: RAG KV Fusion](https://arxiv.org/abs/2405.16444)
    *   [CacheGen: KV Cache Compression](https://arxiv.org/abs/2310.07240)

---

# Prefill-as-a-Service: 跨数据中心 KVCache 架构深度解析

## 1. 核心问题与动机

这篇 paper 来自 Moonshot AI 和 Tsinghua University，核心要解决的问题是：**hybrid-attention 模型架构如何改变 PD disaggregation 的部署边界**。

### 1.1 传统 PD Disaggregation 的 Bandwidth Wall

在传统 dense-attention 模型中，prefill 阶段产生的 KVCache 非常庞大，导致 prefill 和 decode 必须在同一个 high-bandwidth network domain（通常是单个数据中心的 RDMA fabric）内紧密耦合。这个约束的量化度量是 **KV throughput**：

$$\Phi_{\mathrm{kv}}(l) = \frac{S_{\mathrm{kv}}(l)}{T_{\mathrm{prefill}}(l)}$$

其中：
- $\Phi_{\mathrm{kv}}(l)$: 对于长度为 $l$ 的请求，prefill 阶段每单位时间产生的 KVCache 量（单位 Gbps）
- $S_{\mathrm{kv}}(l)$: 长度为 $l$ 的请求对应的 KVCache 大小
- $T_{\mathrm{prefill}}(l)$: 对应的 prefill 延迟（秒）
- $l$: 请求的输入长度

**直觉构建**：$\Phi_{\mathrm{kv}}$ 越大，说明每单位 prefill 计算产生的 KVCache 越多，网络传输压力越大。对于 dense models 如 MiniMax-M2.5（GQA），在 32K tokens 时 $\Phi_{\mathrm{kv}} \approx 60$ Gbps，这远超单台机器的跨数据中心 Ethernet 容量，所以必须用 RDMA。这就是 "bandwidth wall"。

### 1.2 Hybrid Attention 架构的变革

新兴 hybrid 模型交错排列少量 full-attention layers 和大量 linear-complexity layers：

| Model | Attention Type A | Attention Type B | A:B Ratio | Params |
|-------|-----------------|-----------------|-----------|--------|
| Kimi Linear | KDA | MLA | 3:1 | 48B |
| MiMo-V2-Flash | SWA | GQA | 5:1 | 309B |
| Qwen3.5-397B | GDN | GQA | 3:1 | 397B |
| Ring-2.5-1T | Lightning | MLA | 7:1 | 1T |

**关键洞察**：只有 full-attention layers 产生随序列长度增长的 KVCache，linear-complexity layers 维护固定大小的 recurrent state，其 footprint 在长上下文场景下可忽略。这使得 $\Phi_{\mathrm{kv}}$ 下降一个数量级。

从 Table 3 的 benchmark 看：
- 32K tokens 时，MiMo-V2-Flash 的 $\Phi_{\mathrm{kv}} = 4.66$ Gbps vs MiniMax-M2.5 的 59.93 Gbps → **13× 减少**
- Ring-2.5-1T 中 MLA 贡献 ~4.5× 压缩，7:1 hybrid ratio 贡献 ~8× 减少，总体 ~36× KV 内存节省

参考 hybrid attention 架构：
- [Kimi Linear (KDA)](https://arxiv.org/abs/2510.26692)
- [Gated Delta Networks (GDN)](https://openreview.net/forum?id=22PQY4rkJE)
- [Lightning Attention-2](https://arxiv.org/abs/2401.04658)
- [Sliding Window Attention (Longformer)](https://arxiv.org/abs/2004.05150)

### 1.3 但 Hybrid 不够，还需要系统设计

**核心洞察**：KVCache 变小只是 "plausible"，要达到 "practical" 还需要系统设计应对：
- **Bursty traffic**: 突发流量造成 congestion
- **Skewed request lengths**: 请求长度高度倾斜
- **Uneven prefix-cache distribution**: prefix cache 分布不均
- **Fluctuating inter-cluster bandwidth**: 跨集群带宽波动

如果 naive 地将所有 prefill 都外移到远端，仍然会受 congestion、unstable queueing、poor utilization 困扰。

---

## 2. PrfaaS 架构设计

### 2.1 整体拓扑

架构由三个子系统组成：

**Compute Subsystem**:
- **Local PD clusters**: 执行完整 PD-disaggregated serving，可端到端完成推理
- **PrfaaS clusters**: 专用的 compute-dense prefill 集群，使用低成本高吞吐量加速器，处理增量 uncached 长度超过 routing threshold 的请求

**Network Subsystem**（两层）:
- **Intra-cluster**: RDMA，用于延迟敏感的 collective communication 和 PD KVCache transfer
- **Inter-cluster**: VPC peering 或 dedicated lines，用于跨数据中心 KVCache transfer

**Storage Subsystem**:
- 每个 cluster 内构建分布式 hybrid prefix cache pool
- Global KVCache manager 维护跨所有 cluster 的 KVCache metadata
- Global scheduler 基于 request 特征、网络条件、cache 分布路由请求

### 2.2 Hybrid Prefix Cache Pool

传统 prefix cache pools 假设单一 KVCache 类型，在 token 或 block 级别匹配/驱逐。Hybrid models 打破这个假设：

- **Linear attention / SWA recurrent states**: request-level，大小与输入长度无关，只在 cached 长度**完全匹配**时才能重用
- **Full-attention KVCache**: block-level，随输入长度线性增长，支持 **partial prefix matching**

基于 [vLLM 的 hybrid KVCache manager](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/) 设计：
- Linear states 和 full-attention KVCache 由**独立 KVCache groups** 管理，block size 对齐
- 所有 groups 从**共享 KVCache pool** 分配/释放 blocks
- Cache blocks 分为两类：
  - **Prefix-cache blocks**: 必须完全填充后才能跨请求重用（intra-cluster only, block-aligned）
  - **Transfer-cache blocks**: 保存 prefill 请求尾部产生的 KVCache，传输完成后丢弃（cross-cluster）

### 2.3 PrfaaS-PD Disaggregation

**Routing Policy**:
- 设 $l$ 为请求的增量 prefill 长度（排除 cached prefix）
- 设 $t$ 为 routing threshold
- 当 $l > t$: 请求路由到 PrfaaS cluster，完成 prefill 后 KVCache transfer 到 decode node
- 当 $l \leq t$: 请求由 PD cluster 内的 prefill node 处理

**实现技术栈**:
1. **Layer-wise prefill pipelining**: 重叠 KVCache 生成和传输
2. **Multi-connection TCP transport**: 充分利用可用带宽
3. **Congestion monitoring**: 早期检测丢失和重传信号，防止 congestion 累积

---

## 3. 建模与调度

### 3.1 Throughput Model

三个角色的 steady-state throughput 建模：

**PrfaaS throughput**（公式 3）:
$$\Theta_{\mathrm{prfaaS}} = \min\left(\frac{N_{\mathrm{prfaaS}}}{T_{\mathrm{prefill}}(l_{\mathrm{long}})}, \frac{B_{\mathrm{out}}}{S_{\mathrm{kv}}(l_{\mathrm{long}})}\right)$$

变量：
- $\Theta_{\mathrm{prfaaS}}$: PrfaaS cluster 的吞吐量
- $N_{\mathrm{prfaaS}}$: PrfaaS prefill instances 数
- $T_{\mathrm{prefill}}(l_{\mathrm{long}})$: 平均 PrfaaS 请求长度的 prefill 时间
- $B_{\mathrm{out}}$: PrfaaS egress 带宽
- $S_{\mathrm{kv}}(l_{\mathrm{long}})$: 平均 PrfaaS 请求的 KVCache 大小
- $l_{\mathrm{long}} = \mathbb{E}[L | L > t]$: 路由到 PrfaaS 的请求的平均长度

**PD-P throughput**（公式 4，intra-cluster RDMA 非瓶颈）:
$$\Theta_{\mathrm{pd-p}} = \frac{N_p}{T_{\mathrm{prefill}}(l_{\mathrm{short}})}$$

**PD-D throughput**（公式 5）:
$$\Theta_{\mathrm{pd-d}} = \frac{N_d \cdot BS_{\mathrm{max}}}{T_{\mathrm{decode}} \cdot L_{\mathrm{out}}}$$

变量：
- $N_p, N_d$: PD-P / PD-D instances
- $BS_{\mathrm{max}}$: 最大 decode batch size
- $T_{\mathrm{decode}}$: 每步 decode 时间
- $L_{\mathrm{out}}$: 平均输出长度

**端到端系统吞吐量**（公式 6，汇聚 pipeline）:
$$\Lambda_{\max} = \min\left(\frac{\Theta_{\mathrm{prfaaS}}}{p}, \frac{\Theta_{\mathrm{pd-p}}}{1-p}, \Theta_{\mathrm{pd-d}}\right)$$

其中 $p = P(L > t)$ 是路由到 PrfaaS 的请求比例。

**直觉构建**：这是一个三段汇聚 pipeline，PrfaaS 和 PD-P 作为上游 producers（各处理 $p$ 和 $1-p$ 比例的请求），PD-D 作为唯一下游 consumer。系统吞吐量受限于最慢的 stage，除以对应的 routing fraction。

### 3.2 Throughput-Optimal Configuration

两个决策变量：
1. **Routing threshold** $t$ → 决定 $p, l_{\mathrm{long}}, l_{\mathrm{short}}$
2. **PD-cluster prefill/decode ratio** $N_p/N_d$

**Threshold t 的 trade-off**:
- 增大 $t$: 限制 PrfaaS 处理更长的请求，$T_{\mathrm{prefill}}(l)$ 近二次增长而 $S_{\mathrm{kv}}(l)$ 线性增长，降低 per-instance KV throughput，缓解带宽压力
- 减小 $t$: PrfaaS 被短请求淹没，高 KV throughput 容易触发带宽瓶颈

最优 $t$ 使 PrfaaS 和 PD-P 吞吐量同时饱和（公式 7）:
$$\frac{\Theta_{\mathrm{prfaaS}}}{p} = \frac{\Theta_{\mathrm{pd-p}}}{1-p}$$

**Prefill/decode ratio** 平衡 producer 和 consumer 吞吐量（公式 8）:
$$\Theta_{\mathrm{prfaaS}} + \Theta_{\mathrm{pd-p}} = \Theta_{\mathrm{pd-d}}$$

由于 $\Theta_{\mathrm{prfaaS}}/p$ 随 $p$ 单调递减，$\Theta_{\mathrm{pd-p}}/(1-p)$ 随 $p$ 单调递增，可用 **grid search** 高效找到最优操作点。

### 3.3 Dual-Timescale Scheduling

#### Short-term: bandwidth- and cache-aware routing

PrfaaS cluster 有带宽强制的吞吐量上限 $B_{\mathrm{out}} / S_{\mathrm{kv}}(l_{\mathrm{long}})$。当 utilization 接近阈值或 queuing 突增时，触发短期路由调整。

对于有 prefix cache hits 的请求，设：
- $l_{\mathrm{total}}$: 请求总输入长度
- $l_{\mathrm{prfaaS}}$: PrfaaS cluster 中的 cached prefix 长度
- $l_{\mathrm{pd}}$: PD cluster 中的 cached prefix 长度

**当带宽稀缺**（independent evaluation）:
- 如果 $l_{\mathrm{total}} - l_{\mathrm{pd}} \leq t$: 在 PD-P 本地 prefill
- 否则: offload 到 PrfaaS

**当带宽充足**（compute 成为稀缺资源，cross-cluster cache transfer 可减少冗余计算）:
- 令 $l_{\mathrm{prefix}} = \max(l_{\mathrm{prfaaS}}, l_{\mathrm{pd}})$
- 如果 $l_{\mathrm{total}} - l_{\mathrm{prefix}} \leq t$: 在 PD-P prefill
- 否则去 PrfaaS，若 cache 在另一个 cluster 则执行跨 cluster cache transfer

#### Long-term: traffic-driven allocation re-optimization

- 当 $\Theta_{\mathrm{prfaaS}} + \Theta_{\mathrm{pd-p}} \ll \Theta_{\mathrm{pd-d}}$: prefill 是瓶颈
- 当 $\Theta_{\mathrm{prfaaS}} + \Theta_{\mathrm{pd-p}} \gg \Theta_{\mathrm{pd-d}}$: decode 是瓶颈
- 调度器监控各 stage queue depth 和 utilization，定期在 PD cluster 内转换 prefill/decode 角色，调整 $N_p, N_d$ 恢复公式 (7)(8) 的最优性

---

## 4. Case Study：1T 参数 Hybrid 模型

### 4.1 Setup

- **PrfaaS cluster**: 32 个 H200 GPUs，高计算吞吐量
- **Local PD cluster**: 64 个 H20 GPUs，每节点 800 Gbps RDMA
- **Cross-cluster bandwidth**: ~100 Gbps VPC
- **Baseline**: 96 个 H20 GPUs 的 homogeneous PD cluster
- **Model**: 内部 1T 参数 hybrid 模型，Kimi Linear 架构（KDA:MLA = 3:1），8 GPUs/instance
- **Workload**: 截断对数正态分布 $(\mu=9.90, \sigma=1.00$，截断到 [128, 128K])，平均 ~27K tokens，输出 1024 tokens，SLO 40 tokens/s

Table 5 是该模型的 profiling 数据：

| Seq Len | KVCache Size | Prefill Latency | KV Throughput |
|---------|--------------|-----------------|---------------|
| 1K | 190.8 MiB | 0.44s | 3.61 Gbps |
| 8K | 308.9 MiB | 0.72s | 3.59 Gbps |
| 32K | 701.3 MiB | 1.84s | 3.19 Gbps |
| 128K | 2316.3 MiB | 7.40s | 2.62 Gbps |

**直觉构建**：注意 $\Phi_{\mathrm{kv}}$ 随长度增加而**下降**（3.61 → 2.62），因为 prefill latency 近二次增长而 KVCache 大小线性增长。这正是长请求更适合 offload 到 PrfaaS 的原因——per-request 带宽需求反而更低。

### 4.2 最优配置求解

通过 2D grid search：
- **最优 threshold**: $t = 19.4K$（约 50% 请求 offload 到 PrfaaS）
- **最优 PD 内部分配**: $N_p = 3, N_d = 5$（PrfaaS:PD-P:PD-D = 4:3:5）
- PrfaaS 卸载子集平均长度 $\mathbb{E}[L | L > t] \approx 44K$ tokens

### 4.3 结果对比

Table 6 的完整对比：

| Metric | PrfaaS-PD | Homogeneous PD | Naive Hetero PD |
|--------|-----------|----------------|-----------------|
| Threshold t | 19.4K | — | — |
| $N_{\mathrm{prfaaS}}/N_p/N_d$ | 4/3/5 | —/9/3 | 4/—/18 |
| Mean/P90 TTFT (s) | 2.22/3.51 | 4.44/9.73 | 1.74/3.51 |
| $\Theta$ (req/s) | 1.61/1.64/3.91 | —/2.11/2.35 | 2.45/—/6.25 |
| $\Lambda_{\max}$ (req/s) | 3.24 | 2.11 | 2.45 |
| Ratio | 1.54× | 1.00× | 1.16× |

**关键发现**：

1. **带宽利用**: PrfaaS egress 仅 ~13 Gbps，消耗 100 Gbps Ethernet link 的 13%，远低于 RDMA 需求
2. **吞吐量提升**: 相比 homogeneous PD 提升 **54%**，相比 naive heterogeneous PD 提升 **32%**
3. **TTFT 改善**: Mean/P90 TTFT 相比 homogeneous 基线分别降低 **50%/64%**
4. **Naive heterogeneous PD 的教训**: 不调度时仅 1.16× 提升，比 PrfaaS-PD 低 25%。原因：严重不平衡 + 将异构 prefill 视为通用路径

**直觉构建**：PrfaaS-PD 的优势来源是"选择性"——只 offload 真正受益于快速 compute 的长请求，短请求保留在本地避免带宽浪费。Naive 方案把所有 prefill 都外移，既造成 egress 带宽压力，又让 PD-D 成为瓶颈（$\Theta_{\mathrm{pd-d}} = 6.25$ 但只有 2.45 被利用）。

---

## 5. 相关联想与延伸思考

### 5.1 与 Mooncake 的传承关系

[Mooncake](https://arxiv.org/abs/2407.00079) 是 Moonshot AI 之前的 KVCache-centric disaggregated architecture，将 KVCache 视为一等系统资源。PrfaaS 可以看作是 Mooncake 思想的**跨数据中心延伸**：从单一 RDMA domain 内的 KVCache pool，扩展到跨 loosely coupled clusters 的 KVCache transfer。Mooncake 与 [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [Dynamo](https://github.com/ai-dynamo/dynamo) 的合作推动了 PD disaggregation 在生态中的传播。

### 5.2 Hardware Co-design 趋势

论文提到几个 phase-specialized hardware roadmap：
- **[NVIDIA Rubin CPX](https://developer.nvidia.com/blog/nvidia-rubin-cpx-accelerates-inference-performance-and-efficiency-for-1m-token-context-workloads/)**: 针对高吞吐量长上下文 prefill
- **[Groq LPU](https://groq.com/blog/the-groq-lpu-explained)**: 极端内存带宽用于 decode
- **[Taalas HC1](https://taalas.com/products)**: 高内存带宽快速 decode

PrfaaS 架构天然适配这个趋势：无需将异构 chips 放在同一 RDMA island，可独立扩展 prefill 和 decode 容量。

### 5.3 KVCache Compression 的协同效应

论文提到的互补技术：
- **[H2O](https://arxiv.org/abs/2306.14048)**: Heavy-Hitter Oracle，选择性驱逐
- **[KIVI](https://arxiv.org/abs/2402.02750)**: 2-bit 非对称量化
- **[CacheGen](https://arxiv.org/abs/2310.07240)**: KVCache 压缩和流式传输
- **[CacheBlend](https://arxiv.org/abs/2405.16444)**: 非精确 prefix KVCache fusion
- **[KVQuant](https://arxiv.org/abs/2401.18079)**: 10M context length 的 KV 量化

这些技术可进一步减少 KVCache 传输量，扩大 PrfaaS 的可行部署范围。

### 5.4 PD Disaggregation 相关工作谱系

- **[Splitwise](https://arxiv.org/abs/2311.18677)**: 从 cost/power 角度
- **[DistServe](https://arxiv.org/abs/2401.09670)**: 从 goodput 角度
- **[Helix](https://arxiv.org/abs/2503.02457)**: 异构 GPUs/networks 的 max-flow 优化
- **[Hetis](https://arxiv.org/abs/2503.04940)**: fine-grained 动态并行
- **[LLM-PQ](https://arxiv.org/abs/2403.01114)**: phase-aware partition + adaptive quantization
- **[DynamoLLM](https://arxiv.org/abs/2410.00515)**: energy/cost/碳效率调度
- **[FREESH](https://arxiv.org/abs/2511.00807)**: 异构 GPUs 公平调度

### 5.5 对 Long-context Serving 的启示

论文的核心论点可归纳为：**模型架构创新改变了系统设计的可能性边界，但系统设计必须主动 exploit 这个机会**。

具体来说：
- Dense models: $\Phi_{\mathrm{kv}} \sim 60$ Gbps → 必须 RDMA → 无法跨数据中心
- Hybrid models: $\Phi_{\mathrm{kv}} \sim 4$ Gbps → commodity Ethernet 可行 → 但需 selective offloading + bandwidth-aware scheduling + cache-aware placement

这个论点对下一代 LLM serving 系统设计有重要启示：**模型-系统协同设计**是趋势。未来模型 co-design 不仅优化 FLOPs，还要优化 KVCache transfer volume。

### 5.6 局限性与开放问题

1. **实验规模有限**: Case study 用 32 H200 + 64 H20，生产规模（10K+ GPUs）的验证仍需更多数据
2. **Prefix cache 跨集群传输**: 当 cache 在另一 cluster 时需要 transfer，论文提到但未深入量化成本
3. **网络波动建模**: 短期调度应对 fluctuating bandwidth，但具体检测机制和响应延迟未详述
4. **Agentic workloads**: 论文提到 incremental prefills 占多数，但 multi-turn agentic 的复杂 prefix 关系未深入
5. **Failure recovery**: PrfaaS cluster 故障时的 fallback 策略未讨论

### 5.7 延伸：为什么这个方向重要

从 Karpathy 的视角看，这篇 paper 触及几个重要的 scaling 趋势：
1. **Context length 指数增长**: 1M+ token context 已在 roadmap，KVCache 管理将成为主导成本
2. **Hardware specialization**: compute-bound (prefill) vs memory-bound (decode) 的分化是必然
3. **Geographic distribution**: AI 训练和推理必然走向跨数据中心，KVCache 的可移植性是关键使能
4. **Cost efficiency**: 异构部署而非同构堆砌，是降低 inference 成本的正确方向

PrfaaS 的贡献在于认识到**模型架构演进已经改变了系统约束的硬度**，但单纯依赖这个改变还不够，必须配以相应的系统设计才能将 "plausible" 变为 "practical"。这是典型的"机会识别 + 主动 exploit"的研究范式。

---

## 6. 公式索引与变量速查

| 符号 | 含义 | 出现公式 |
|------|------|---------|
| $\Phi_{\mathrm{kv}}(l)$ | KV throughput (Gbps) | (1) |
| $S_{\mathrm{kv}}(l)$ | 长度 $l$ 的 KVCache 大小 | (1), (3) |
| $T_{\mathrm{prefill}}(l)$ | 长度 $l$ 的 prefill 延迟 | (1), (3), (4) |
| $B_{\mathrm{out}}$ | PrfaaS egress 带宽 | (2), (3) |
| $N$ | GPU 总数 | (2) |
| $P$ | 并行度 (GPUs/instance) | (2) |
| $L_{\mathrm{avg}}$ | 平均 uncached 输入长度 | (2) |
| $l_{\mathrm{long}} = \mathbb{E}[L \mid L>t]$ | PrfaaS 平均长度 | (3) |
| $l_{\mathrm{short}} = \mathbb{E}[L \mid L \leq t]$ | PD-P 平均长度 | (4) |
| $p = P(L>t)$ | PrfaaS 比例 | (6), (7) |
| $L_{\mathrm{out}}$ | 平均输出长度 | (5) |
| $N_{\mathrm{prfaaS}}$ | PrfaaS instances | (3) |
| $N_p, N_d$ | PD-P / PD-D instances | (4), (5), (8) |
| $BS_{\mathrm{max}}$ | 最大 decode batch size | (5) |
| $T_{\mathrm{decode}}$ | 每步 decode 时间 | (5) |
| $\Theta_{\mathrm{prfaaS}}, \Theta_{\mathrm{pd-p}}, \Theta_{\mathrm{pd-d}}$ | 各角色吞吐量 | (3)-(8) |
| $\Lambda_{\max}$ | 系统端到端吞吐量 | (6) |
| $t$ | routing threshold | (7) |

---

## 参考链接汇总

**核心论文与系统**:
- [Mooncake (KVCache-centric)](https://arxiv.org/abs/2407.00079)
- [vLLM](https://github.com/vllm-project/vllm) | [SGLang](https://github.com/sgl-project/sglang) | [Dynamo](https://github.com/ai-dynamo/dynamo)

**Hybrid Attention 架构**:
- [Kimi Linear (KDA)](https://arxiv.org/abs/2510.26692) | [DeepSeek-V2 (MLA)](https://arxiv.org/abs/2405.04434) | [GQA](https://arxiv.org/abs/2305.13245) | [Gated Delta Networks](https://openreview.net/forum?id=22PQY4rkJE) | [Lightning Attention-2](https://arxiv.org/abs/2401.04658) | [Longformer (SWA)](https://arxiv.org/abs/2004.05150)

**PD Disaggregation**:
- [Splitwise](https://arxiv.org/abs/2311.18677) | [DistServe](https://arxiv.org/abs/2401.09670) | [Helix](https://arxiv.org/abs/2503.02457) | [DynamoLLM](https://arxiv.org/abs/2410.00515)

**KVCache 压缩与重用**:
- [H2O](https://arxiv.org/abs/2306.14048) | [KIVI](https://arxiv.org/abs/2402.02750) | [CacheGen](https://arxiv.org/abs/2310.07240) | [CacheBlend](https://arxiv.org/abs/2405.16444) | [KVQuant](https://arxiv.org/abs/2401.18079)

**Hardware**:
- [NVIDIA Rubin CPX](https://developer.nvidia.com/blog/nvidia-rubin-cpx-accelerates-inference-performance-and-efficiency-for-1m-token-context-workloads/) | [Groq LPU](https://groq.com/blog/the-groq-lpu-explained) | [Taalas HC1](https://taalas.com/products)

**vLLM Hybrid Models**:
- [Hybrid models as first-class citizens in vLLM](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/)
