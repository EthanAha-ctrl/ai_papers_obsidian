---
source_pdf: PipeFusion.pdf
paper_sha256: 6dea32e38685bf94fa8d6f9107a4a90b6defd1f2fa692f755d869e68215b6a44
processed_at: '2026-08-06T04:11:36-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 PipeFusion

## 先说个场景

假设你拿 Stable Diffusion 3 或者 Flux.1 想生成一张 4K 图。单张 L40 跑下来要 40 多秒，你受不了，弄了 8 张卡想加速。结果发现：**直接把模型切到 8 张卡上，反而更慢**。

为什么？因为 diffusion transformer（DiT）的 attention 计算需要所有 patch 互相看到，卡之间必须疯狂通信。卡一多，通信比计算还慢。

PipeFusion 这篇 paper 干的事就是：**在 PCIe 这种带宽很差的机器上，让 8 卡 DiT inference 真正能加速，而且不爆显存**。

---

## DiT inference 的物理过程

先把背景用一句话说清楚：

Diffusion 生成是从噪声一步步去噪的过程。每一步叫一个 **timestep**。每个 timestep 里，你要跑一次完整的 transformer forward，输入是当前带噪 latent image $x_t$，输出是预测的 noise $\epsilon_t$：

$$
x_{t-1} = \mathrm{Update}(x_t, t, \epsilon_t), \quad \epsilon_t = \epsilon_\theta(x_t, t, c)
$$

- $x_t$：第 $t$ 步的 noisy image（latent space），shape 是 $(p, hs)$，$p$ 是 sequence length（latent pixel 数），$hs$ 是 hidden size
- $t$：当前 timestep，注意 diffusion 从 $T$ 反向走到 $0$，所以"上一步"在公式里是 $T+1$
- $\epsilon_\theta$：你的 DiT backbone，就是一堆 transformer block 堆起来
- $c$：condition，比如 text embedding
- $\mathrm{Update}$：sampler 的 elementwise 操作，跟 forward 比可以忽略

通常要跑 20~50 步 timestep 才能出一张高质量图。瓶颈全在 $\epsilon_\theta$ 的 forward 上。

---

## 大家都怎么并行，各自的问题

### Tensor Parallel（TP）

Megatron 那套：每层 linear layer 切成 column-wise + row-wise，每层两个 all-reduce。

通信量：$4 O(p \cdot hs) \cdot L$，$L$ 是 layer 数。

问题：
- 通信量随 $L$ 线性涨，SD3 有 38 层、Flux 有 57 层，通信巨贵
- 不能 overlap，all-reduce 必须等齐
- MM-DiT 这种非标架构切 weight 极其麻烦，paper 里直接说 SD3 没法 TP

### Sequence Parallel（SP）

把 image 沿 sequence 切成几份，每个 device 拿一份，attention 时互相交换 K/V。两种实现：

- **DeepSpeed-Ulysses**（https://arxiv.org/abs/2309.14509）：All2All 切换 sequence/hidden 维度
- **Ring-Attention**（https://arxiv.org/abs/2310.01889）：K/V 用 P2P ring 传，实现分布式 FlashAttention
- **USP**（https://arxiv.org/abs/2405.07719）：Ulysses + Ring 的 2D 组合

通信量：$2 O(p \cdot hs) \cdot L$（Ring）或 $\frac{4}{N} O(p \cdot hs) \cdot L$（Ulysses）。

问题：每层都要通信，通信量还是正比于 $L$。PCIe 下 All2All 要穿 CPU socket，特别慢——paper Figure 5/6/7 里 SP-Ulysses 总是最慢就是这个原因。

### DistriFusion（https://arxiv.org/abs/2402.19481）

这才是 PipeFusion 真正的对手。它的发现很巧：

**diffusion 相邻 timestep 的输入几乎一样**。你拿 $x_t$ 和 $x_{t+1}$ 算出来的 K/V 差别极小（叫 input temporal redundancy）。所以第 $T$ 步算 attention 时，可以用上一步的 stale K/V 顶替其他 device 的 fresh K/V，把当前 K/V 的 all-gather 异步 overlap 到本步 forward 计算里，下一步才用上。

具体比例：每个 timestep 用 $\frac{1}{N}$ 本地 fresh K/V + $\frac{N-1}{N}$ stale K/V。

听起来很美，但有致命问题：**每个 device 都要做 full attention，所以每个 layer 都得有全 spatial shape 的 K/V buffer**。memory cost 是 $KV \cdot L$，与 $N$ 无关。加 GPU 也不省这块内存。4096px 直接 OOM。

---

## PipeFusion 的核心 idea

### 一句话直觉

DistriFusion 只把 temporal redundancy 用在 attention 的 K/V 上。PipeFusion 把它升级成**整个 pipeline 之间的 synchronization barrier 替代品**——让 activation 在 pipeline stage 间流动时不用等齐，用上一步同 patch 的 stale feature 占位先算。

这一招同时解决三个问题：
1. 通信量从 $O(L)$ 降到 $O(1)$
2. KV buffer 从 $O(L)$ 降到 $O(L/N)$
3. 生成质量比 DistriFusion 还好

### 具体怎么切

PipeFusion 同时切两样东西（paper Figure 3 上半）：

1. **DiT backbone layers 按 data flow 切**：$L$ 层均分到 $N$ 个 device，每 device 拿 $L/N$ 层连续 transformer block
2. **输入 image 沿 sequence 切成 $M$ 个 patches**：每个 patch 当一个 micro-batch

跟传统 pipeline 不一样：
- GPipe（https://arxiv.org/abs/1811.06965）沿 batch 切，要高并发请求填满，单图没法用
- TeraPipe（https://arxiv.org/abs/2102.07188）沿 token 切但面向 causal attention
- PipeFusion 沿 sequence 切，面向 full-attention，靠 temporal redundancy 把 stage 间依赖解掉

### Pipeline 怎么流

看 paper Figure 3 下半，$N=4$ GPU，$M=4$ patches，当前 timestep $T$：

```
micro-step:       1     2     3     4     5     6     7     8
GPU 0 (layer 0):  P0(T) P1(T) P2(T) P3(T)
GPU 1 (layer 1):        P0(T+1) P1(T) P2(T) P3(T)
GPU 2 (layer 2):               P0(T+1) P1(T+1) P2(T) P3(T)
GPU 3 (layer 3):                      P0(T+1) P1(T+1) P2(T+1) P3(T)
```

关键在 micro-step 5：GPU 0 算完 P0@T 要传给 GPU 1。这时 GPU 1 上 P0 还是从 $T+1$ 拿的 stale，**但 GPU 1 不等 P0@T 到达，直接开始算 P1@T**（用 P1 的 fresh local context + 其他 patches 的 stale context）。

这就是 pipeline 中间没有 bubble 的原因：用 stale activation 占位，让 pipeline 一直跑满。只有启动时 $N-1$ 个 micro-step 是 fill bubble。

### Bubble 比例

effective compute ratio：

$$
\frac{M \cdot S}{M \cdot S + (N - 1)}
$$

- $M$：patch 数
- $S$：总 timestep 数
- $N$：并行度
- $N-1$：fill 阶段的 bubble 数

$M=N=4, S=50$ 时 $\frac{200}{203} \approx 98.5\%$。$S$ 越大 bubble 占比越低，diffusion 高质量生成本来就要 20~50 步，特别适合。

### Freshness wave：为什么质量没崩

paper Figure 4 是理解质量的关键。看一个 timestep 内部，每个 micro-step 的 fresh/stale 比例怎么变：

- micro-step 4：device 3 上 P0@T 是 fresh，其他全 stale
- micro-step 5：device 0 算完 P0@T 传给 device 1，device 1 的 P0 变 fresh
- micro-step 6：device 1 算完，传给 device 2，device 2 的 P0 变 fresh
- micro-step 8：device 3 上 P0@T fresh，整个 pipeline P0 全跑通

也就是说 PipeFusion 在 timestep 内部有**"freshness wave"在传播**——每个 patch 沿 pipeline 流过时，会被"refresh 一次"。到 timestep 结束时，每个 patch 在每个 layer 都被 fresh 过。

对比 DistriFusion：整个 diffusion 过程都是固定 $\frac{1}{N}$ fresh + $\frac{N-1}{N}$ stale，fresh 比例永远这么低。

所以 PipeFusion 的 effective staleness 比 DistriFusion 低，理论上 FID 更好。paper Table 2 验证：
- Pixart 8 device：DistriFusion FID=41.81，PipeFusion FID=28.46
- Flux 8 device：PipeFusion FID=25.97，非常接近 original 25.03

---

## 通信和内存到底省在哪

paper Table 1 是核心证据，我把它摊开讲：

| Method | Comm Cost | Overlap? | Param Mem | KV Buffer Mem |
|---|---|---|---|---|
| Tensor Parallel | $4 O(p\cdot hs) L$ | ✗ | $\frac{1}{N} P$ | $\frac{1}{N} KV$ |
| DistriFusion | $2 O(p\cdot hs) L$ | ✓ | $P$ | $KV \cdot L$ |
| SP-Ring | $2 O(p\cdot hs) L$ | ✓ (in attn) | $P$ | $\frac{1}{N} KV$ |
| SP-Ulysses | $\frac{4}{N} O(p\cdot hs) L$ | ✗ | $P$ | $KV$ |
| **PipeFusion** | $2 O(p\cdot hs)$ | ✓ | $\frac{1}{N} P$ | $\frac{1}{N} (KV) L$ |

### 通信为什么是 $2 O(p\cdot hs)$ 不带 $L$

这是整个 paper 最数学上的关键点。

PipeFusion 的 activation 只在 **stage 边界** 跨 device 通信：device $i$ 接收 patch 输入到自己的第一个 layer，发送自己最后一个 layer 的输出给下一个 device。stage 内部 $L/N$ 层完全本地计算，不通信。

而 SP / DistriFusion 每个 layer 都要 cross-device 通信 K/V，所以通信量正比于 $L$。

只要 $N < 2L$（SD3 $L=38$、Flux $L=57$，随便满足），PipeFusion 通信量就最低。

### 内存：为什么 PipeFusion 优于 DistriFusion

DistriFusion 每个 device 都要做 full attention，所以每个 layer 都要存全 spatial K/V buffer，共 $KV \cdot L$，**与 $N$ 无关**——加 GPU 也不省。长 sequence 直接 OOM。

PipeFusion 每个 device 只负责 $L/N$ 个 layer，每个 layer 的 KV buffer 只存自己 patch 的部分，所以是 $\frac{1}{N} (KV) L$，**与 $N$ 反比**。

paper Figure 9 里 DistriFusion 在 4096px 和 Flux 上 OOM，PipeFusion 不 OOM，根本原因就是这个。

### Parameter 分布

- TP / PipeFusion: $\frac{1}{N} P$，模型越大越友好（Flux 12B 必须 parameter shard）
- SP / DistriFusion: $P$，每 device 全量 weight

---

## 实现细节：async P2P overlap

paper Section 3.3 末段说：每个 device 用 asynchronous P2P 把 micro-step 的 patch activation 发给下游 device，同时本地算下一个 patch。

直觉：device 0 算 P0@T 时，它发给 device 1 的 P1@T 已经在 PCIe 上飞了。device 1 收 P1@T 同时在算 P0@T，device 1 算完 P0@T 时，device 0 发来的 P0@T 已经到了。全程没人等。

这要求 CUDA stream 上 `cudaMemcpyAsync` + compute stream 分离，P2P 真异步。这就是为什么 paper 强调 pipeline 没中间 bubble。

---

## Warmup：前几步不能用 stale

Diffusion 早期 timestep（$t \to T$ 端）输入变化剧烈（pure Gaussian noise → 第一次降噪），temporal redundancy 不成立，stale activation 误差大会污染后续所有 step。

所以 PipeFusion / DistriFusion 都需要几个 **warmup steps** 走同步 SP，过后切到 async 模式。

paper Table 3 数据：

| Warmup | Pixart pp=8 | SD3 pp=8 | Flux pp=8 |
|---|---|---|---|
| 0 | 0.71 | 1.05 | 5.48 |
| 1 | 0.82 (+15%) | 1.16 (+10%) | 5.53 (+1%) |
| 2 | 0.91 (+28%) | 1.27 (+21%) | 6.00 (+9%) |

观察：
- Flux 对 warmup 极不敏感（1 step 只 +1%），可能 MM-DiT 跨模态 mixing 对 stale 容忍度好
- Pixart、SD3 敏感（2 step 涨 20~28%）
- CFG=2 时 warmup 代价除以 2（两组 pipeline 并行）

paper 提到可以把 warmup steps 和 working steps 分到不同 device pool，warmup 用 SP、working 用 PipeFusion，stage 间传 feature map——这是 future work，很 systems-y 的思路。

---

## CFG Parallel：白嫖 2× 加速

Classifier-Free Guidance（https://arxiv.org/abs/2207.12598）每次 timestep 要跑两次 forward：unconditional branch + conditional branch。

paper 把 8 GPU 切成两组，每组 4 GPU 跑独立 PipeFusion pipeline，一次 forward 同时算两个 branch。在"inter-image"维度并行。

数据上看（paper Section 4.1.1）：
- Pixart 1024px: 1 GPU baseline vs CFG=2,PipeFusion=4 → **5.09× speedup**
- Pixart 4096px: **8.59×**
- SD3 1024px: 3.11×
- SD3 2048px: 8.16×
- Flux 1024px: 3.42×
- Flux 2048px: 5.79×

大分辨率下 speedup 远大于 1024px，因为 compute 增长 $p^2$ 而通信相对小。

CommShare 公式（公式 2）：

$$
\mathrm{CommShare} = \frac{T_{\mathrm{E2E,8GPU}} - \frac{T_{\mathrm{single}}}{8}}{T_{\mathrm{E2E,8GPU}}}
$$

差值 = 通信 + bubble + load imbalance 等所有 overhead。

例子（Pixart 4096px）：
- PipeFusion: $(32.1 - 244.89/8)/32.1 \approx 4.6\%$
- SP: $(37.3 - 244.89/8)/37.3 \approx 17.9\%$

绝对通信时间 SP 6.69s → PipeFusion 1.49s，**降 78%**。即使 compute-bound regime，PipeFusion 也压住了通信。

---

## 实验结果

### Latency（Figure 5/6/7）

硬件选得很有讲究：8×L40 PCIe Gen4×16，没有 NVLink。通信带宽稀缺，最能放大"通信量小"的优势。

**Pixart（0.6B）**：
- 1024px: PipeFusion 0.66s，比第二名快 1.48×
- 2048px: 快 1.55×
- 4096px: 快 1.16×（DistriFusion OOM）
- 速度提升随分辨率下降：compute-bound 后通信比例下降，优势变小

**SD3（2B MM-DiT）**：
- 1024px: 1.57× over best USP
- 2048px: 1.30×

**Flux.1（12B MM-DiT）**：
- 1024px: 1.23× over USP(ulysses=4,ring=2)
- 2048px: 1.25×

Flux 优势比 Pixart 小，因为 12B 模型本身 compute 占主导。但绝对速度仍然最快。

### Memory（Figure 9）

- Pixart 8192px：PipeFusion 是唯一能跑的（DistriFusion OOM）
- SD3 / Flux 1024/2048px：PipeFusion 整体内存 < SP
  - Flux 1024px: PipeFusion = **32% of SP memory**
  - Flux 2048px: 36%
- 注意 PipeFusion 激活内存略高于 SP（因为 KV buffer），但 parameter shard 省下的远大于多花的

### NVLink 鲁棒性（Table 4）

8×A100 NVLink 上 Pixart：
- 1024px: TP 1.22s, PipeFusion 0.66s
- 4096px: PipeFusion 22.39s vs SP-Ring 23.31s vs TP 36.33s

NVLink 高带宽下 PipeFusion 仍然最优，只是相对优势收窄。这佐证 PipeFusion 优势主要来自"通信量"本身（数学上的），不只是"能 overlap"（硬件上的）。

### Patch 数 $M$ 的影响（Figure 11）

paper 推荐 $M = N$：
- $M$ 太大：单 operator input 小，GPU 利用率低
- $M$ 太小：overlap 机会少，pipeline bubble 难填
- $M = N$ 是 sweet spot，正好每个 device 一个 patch 在 pipeline 中

---

## 质量（Table 2 + Figure 10）

FID 越低越好，分两种：
- "w/ G.T."：与 ground truth COCO image 比
- "w/ Orig."：与 1-Device original 生成 image 比

**Pixart-XL**：
- Original: 22.78
- DistriFusion 8 dev: 41.81
- PipeFusion 2 dev: 23.60 / 4 dev: 25.23 / 8 dev: 28.46

**Flux.1**：
- Original: 25.03
- PipeFusion 4 dev: 24.17 / 8 dev: 25.97

PipeFusion 全面优于 DistriFusion。原因就是 freshness wave：timestep 内 fresh 比例从 $\frac{1}{M}$ 单调增到 1，DistriFusion 永远是 $\frac{1}{N}$。

更深的直觉：DiT attention 是全 spatial 交互，每个 patch 要看其他 patches 的 K/V。如果其他 patches 用 stale K/V，相当于 attention 里引入了 temporal-smoothed context。当 timestep 内 fresh wave 传播到大部分 patches，相当于"在线更新" context，比 DistriFusion 那种"永远只看自己这一份 fresh + 其他全 stale"的固定模式更接近 ground truth。

---

## Limitation

paper Section 5 诚实承认：

1. **少 step distilled 模型不适合**。Flux.1-schnell 4-step、one-step diffusion（DMD https://arxiv.org/abs/2310.14795、Mean Flows https://arxiv.org/abs/2505.13447）temporal redundancy 弱，pipeline 优势消失。但 SOTA 模型仍需 20+ step，影响有限。

2. **不打算单挑所有场景**，建议 hybrid：节点内 NVLink 用 SP，节点间 PCIe / IB 用 PipeFusion。大集群部署的 natural composition。

### Video Generation 延伸（很有意思）

- Wan2.1（https://arxiv.org/abs/2503.20314）、HunyuanVideo（https://arxiv.org/abs/2412.03603）、CogVideo（https://arxiv.org/abs/2205.15868）都是 MM-DiT backbone
- Video latent 形状 (B, L, H)，空间 + 时间维度 flatten 到 L
- PipeFusion 沿 sequence 切，patch = spatiotemporal block
- Video 模型参数更大（HunyuanVideo 13B）、sequence 更长（几万 token），通信放大效应更强 → PipeFusion 优势更大
- 这是个非常 natural 的延伸，几乎不需要改架构

---

## 我自己看完后的几个直觉

### temporal redundancy 作为 parallelism 的资源

以前大家用 temporal redundancy 做 **compute 优化**：DeepCache（https://arxiv.org/abs/2312.00958）、TGATE、PAB（https://arxiv.org/abs/2408.12588）、DiTFastAttn（https://arxiv.org/abs/2406.08522）、Δ-DiT（https://arxiv.org/abs/2406.01125）、Learning-to-Cache（https://arxiv.org/abs/2406.01733）——把某些 step 的某些层 skip 掉。

DistriFusion 第一个把它用到 **通信优化**——把 stale K/V 当免费的 future。

PipeFusion 把这招再升一层——把 temporal redundancy 当 **pipeline synchronization barrier 的替代品**。本质上是用"信息时延"换"通信时延"。

这是个很 systems-y 的思路。LLM pipeline 里 bubble 是硬问题（1F1B、interleaved、ZB-H1 https://arxiv.org/abs/2407.09067 各种 scheme 都在治 bubble）。DiT 这边因为 temporal redundancy 给了一个"假装拿到数据"的合法手段。

### "通信量与 L 无关"为什么重要

LLM 训练里 pipeline parallel 能赢过 TP，是因为跨节点带宽差 intra-node 几个量级。DiT inference 现在 8 卡 PCIe 也面对类似 gap——L40 没 NVLink。

PipeFusion 把跨 device 通信从"每层一次"降到"整个 forward 两次"，相当于把 transformer 内部彻底本地化。跟 LLM 那边 SP 之于 TP 的关系类似，但 DiT 这边因为是 full attention + iterative + temporal redundancy，多了一个 DistriFusion 没拿到的 advantage（parameter + KV 都 shard）。

### 为什么 freshness wave 不会影响视觉质量

DiT 早期 layer 提取 low-level feature（颜色、纹理），晚期 layer 做高层 semantic composition。Patch 流过 pipeline 时，每个 patch 在每个 layer 都会被"refresh 一次"。等 patch 走到 layer $L$，它在这个 timestep 的每层都是 fresh。

这跟 DeepCache 的观察一致：浅层 activation 变化快、深层变化慢。PipeFusion 反向利用：**因为变化慢，所以可以 stale；因为每个 patch 总会被 refresh，所以最终 fresh**。两个性质叠起来就是"高质量 + 高吞吐"。

---

## 一句话总结

PipeFusion = **"sequence-parallel pipeline with temporal-redundancy-based barrier removal"**。它发现 DiT inference 的两个 redundancy（沿 sequence 切可以独立处理 patch、沿 timestep 切可以用 stale activation）可以 compose，从而同时压低 comm cost（$O(L) \to O(1)$）、memory（$O(L) \to O(L/N)$）、staleness（DistriFusion $\frac{1}{N}$ fresh → PipeFusion freshness wave）。在 PCIe 这种带宽贫瘠场景上，这就是 1.5× 速度 + 不 OOM 的来源。

---

## References

- PipeFusion 代码（xDiT 框架）：https://github.com/xdit-project/xDiT
- DistriFusion（CVPR 2024）：https://arxiv.org/abs/2402.19481
- DeepSpeed-Ulysses：https://arxiv.org/abs/2309.14509
- Ring-Attention：https://arxiv.org/abs/2310.01889
- USP（Unified Sequence Parallelism）：https://arxiv.org/abs/2405.07719
- DiT (Peebles & Xie, ICCV 2023)：https://arxiv.org/abs/2212.09748
- PixArt-α：https://arxiv.org/abs/2310.00426
- PixArt-Σ：https://arxiv.org/abs/2403.04692
- Stable Diffusion 3：https://arxiv.org/abs/2403.03206
- Flux.1 by Black Forest Labs：https://blackforestlabs.ai
- GPipe：https://arxiv.org/abs/1811.06965
- TeraPipe：https://arxiv.org/abs/2102.07188
- Classifier-Free Guidance：https://arxiv.org/abs/2207.12598
- DeepCache：https://arxiv.org/abs/2312.00958
- PAB (Pyramid Attention Broadcast)：https://arxiv.org/abs/2408.12588
- DiTFastAttn：https://arxiv.org/abs/2406.08552
- Δ-DiT：https://arxiv.org/abs/2406.01125
- Learning-to-Cache：https://arxiv.org/abs/2406.01733
- DDIM：https://arxiv.org/abs/2010.02502
- DPM-Solver：https://arxiv.org/abs/2206.00927
- Mean Flows：https://arxiv.org/abs/2505.13447
- DMD (Distribution Matching Distillation)：https://arxiv.org/abs/2310.14795
- Wan 2.1：https://arxiv.org/abs/2503.20314
- HunyuanVideo：https://arxiv.org/abs/2412.03603
- CogVideo：https://arxiv.org/abs/2205.15868
- Megatron-LM：https://arxiv.org/abs/1909.08053
- VAE (Kingma)：https://arxiv.org/abs/1312.6114
- Clean-FID：https://arxiv.org/abs/2104.11222
- FID 原始定义：https://arxiv.org/abs/1706.08500
- ZB-H1（LLM pipeline bubble 治理参考）：https://arxiv.org/abs/2407.09067
- COCO Captions：https://arxiv.org/abs/1504.00325

---

# PipeFusion: 给 Diffusion Transformers 的 patch-level pipeline parallelism

## 0. 一句话直觉

这篇 paper 的核心 insight 在于：**diffusion process 是迭代的、相邻 timestep 的输入高度相似，这个 temporal redundancy 以前只在 attention 计算层面被利用（DistriFusion 用 stale K/V），但其实可以把它升级成 pipeline parallelism 的 "synchronization bypass"**——让 activation 在 pipeline stage 间流动时不必等齐，用上一步同 patch 的 stale feature 当 placeholder。这一招把通信量从 O(L) 降到了 O(1)，同时 KV buffer 也从全 spatial 变成了 1/N。

下面我把直觉拆开讲清楚，并把每条都钉到具体的公式/表格/图上。

---

## 1. 为什么 DiT inference 难、为什么需要新的并行方案

Diffusion 的 forward 过程（公式 1）：

$$
x_{t-1} = \mathrm{Update}(x_t, t, \epsilon_t), \quad \epsilon_t = \epsilon_\theta(x_t, t, c)
$$

- $x_t \in \mathbb{R}^{p \times hs}$：timestep $t$ 时的 noisy latent image，$p$ 是 sequence length（latent pixel 数），$hs$ 是 hidden size
- $t$：当前 diffusion timestep（注意 diffusion 是从 $T$ 反向走到 $0$，所以 "前一步" 是 $T+1$）
- $\epsilon_t$：网络预测出的 noise
- $\epsilon_\theta$：noise-prediction network，本文主角，就是 DiT backbone
- $c$：condition（text embedding 之类）
- $\mathrm{Update}$：sampler 特定的 elementwise 操作（DDIM / DPM-Solver / FlowMatch 等），相对 DiT forward 可忽略

性能瓶颈是 $\epsilon_\theta$ 的 forward。DiT 把 latent 切成 patches、过一堆 transformer block（MHA + LayerNorm + FFN），attention 是 $O(p^2)$。当 resolution 上到 4096px、video 这种长 sequence 时，单卡装不下也跑不动，必须并行。

并行三件套（TP / SP / DistriFusion）都各有问题，下面逐个看。

---

## 2. 现有方案及其物理瓶颈

### 2.1 Tensor Parallelism (TP)

Megatron 那一套：column-wise + row-wise 切 linear，每个 transformer block 两个 all-reduce。

通信量：$4 \cdot O(p \cdot hs) \cdot L$（每个 layer $4 O(p\cdot hs)$，$L$ 层叠加）

问题：
- 通信量与 $L$ 线性增长，长 sequence 直接卡带宽
- 不能 overlap（每次 all-reduce 必须等齐才继续算）
- MM-DiT（Flux / SD3）这种 non-standard 架构切 weight 极其痛苦，paper 里直接说 SD3 没法 TP

### 2.2 Sequence Parallelism (SP)：Ulysses / Ring

DeepSpeed-Ulysses（https://arxiv.org/abs/2309.14509）：All2All 在 sequence 和 hidden 维度间切换，attention 头在 hidden 切。

Ring-Attention（https://arxiv.org/abs/2310.01889）：K/V 用 P2P ring 传，实现 distributed FlashAttention。

USP = Ulysses + Ring 组合（https://arxiv.org/abs/2405.07719），2D mesh 上行 Ulysses 列 Ring。

通信量：
- SP-Ulysses: $\frac{4}{N} O(p \cdot hs) \cdot L$
- SP-Ring: $2 O(p \cdot hs) \cdot L$（overlap 但只在 attention 内部）

Ulysses 通信量随 $N$ 下降看着漂亮，但 PCIe 下 8 GPU 做 All2All 要穿过 CPU socket，实际 latency 爆炸——这就是 paper Figure 5/6/7 里 SP-Ulysses 总是最慢的原因。Ring / USP 好一些，但本质还是每层都要通信，通信量正比于 $L$。

### 2.3 DistriFusion（https://arxiv.org/abs/2402.19481）

这是 PipeFusion 真正的对标对象，idea 很巧妙：相邻 timestep 输入高度相似（input temporal redundancy），所以可以拿上一步的 K/V 当下一步的 K/V 来用，把当前 step 的 K/V all-gather 异步 overlap 到本 step 的 forward 计算里，下一个 step 才用上。

具体比例：第 $T$ 步用 $\frac{1}{N}$ 本地 fresh K/V + $\frac{N-1}{N}$ 上一步 stale K/V 做 attention。

致命问题（paper Section 3.2）：
- **每个 device 必须维护所有 $L$ 层的全 spatial shape K/V buffer**，因为每个 device 都要做 attention，每个 layer 都需要 full K/V
- KV buffer memory $= KV \cdot L$，**与 $N$ 无关**——加再多 GPU 也不省这部分内存
- 长序列直接 OOM（paper Figure 6 里 DistriFusion 在 4096px OOM，根本没数据）

---

## 3. PipeFusion 的核心：两件事同时切

### 3.1 Partition 策略（paper Figure 3 上半）

PipeFusion 同时切两样东西：
1. **DiT backbone layers 按 data flow 方向切**：$L$ 层均分到 $N$ 个 device，每个 device 拿 $L/N$ 层连续 transformer block
2. **输入 image patch 切**：把 image 切成 $M$ 个 non-overlapping patches，每个 patch 是一个 "micro-batch"

跟传统 pipeline（GPipe，https://arxiv.org/abs/1811.06965）和 TeraPipe（https://arxiv.org/abs/2102.07188）都不同：
- GPipe 沿 batch 切，需要高并发请求填满 pipeline，单 image 没法用
- TeraPipe 沿 token 切但面向 causal attention，靠 triangular structure 让每个 stage 计算量递减
- PipeFusion 沿 sequence 切，面向 full-attention（DiT 没有 causal mask），靠 temporal redundancy 把 stage 间依赖解掉

### 3.2 Pipeline 流水（paper Figure 3 下半）

考虑 $N=4$ 个 GPU、$M=4$ 个 patches，当前 timestep $T$，上一步 $T+1$。

Pipeline 时间线（micro-step 1→8）：

```
micro-step:        1    2    3    4    5    6    7    8
GPU 0 (layer 0):  P0(T) P1(T) P2(T) P3(T) [等待下个T...]
GPU 1 (layer 1):        P0(T+1) P1(T) P2(T) P3(T)
GPU 2 (layer 2):               P0(T+1) P1(T+1) P2(T) P3(T)
GPU 3 (layer 3):                      P0(T+1) P1(T+1) P2(T+1) P3(T)
```

看 micro-step 5（GPU 0 算完 P0@T，要传给 GPU 1），此时 GPU 1 上 P0 是从上一步 $T+1$ 拿的 stale activation，但 GPU 1 可以**不等** P0@T 到达就开始算 P1@T（用 P1 的 fresh local context + 其他 patches 的 stale context）。

这就是 pipeline 没有中间 bubble 的关键：**用 stale activation 当 placeholder 占位，让 pipeline 一直跑满**。只有启动时有 $N-1$ 个 micro-step 的 fill bubble。

### 3.3 Bubble 分析

effective compute ratio：

$$
\frac{M \cdot S}{M \cdot S + (N - 1)}
$$

- $M$：patch 数
- $S$：diffusion 总 timesteps 数
- $N$：并行度
- $N-1$：pipeline fill 阶段的 bubble 数

$M=N=4$, $S=50$ 时 $\frac{200}{203} \approx 98.5\%$。$S$ 越大 bubble 比例越低（diffusion 高质量生成本来就要 20~50 步）。

### 3.4 为什么不会爆 bubble？fresh 比例逐渐增大

paper Figure 4 很关键。把当前 timestep $T$ 内每个 micro-step 的 fresh / stale 比例可视化：

- micro-step 4：device 3 上 P0@T，其他全是 stale
- micro-step 5：device 0 算 P0@T 完成，传到 device 1，于是 device 1 的 P0 变 fresh
- micro-step 6：device 1 算完，传到 device 2，device 2 的 P0 变 fresh
- 一直到 micro-step 8：device 3 上 P0@T fresh，全 pipeline 跑通

也就是说 **PipeFusion 在 timestep 内部有"freshness wave"在传播**，到 timestep 结束时所有 patches 在所有 layers 都已经变 fresh 过至少一次。

对比 DistriFusion：每个 timestep 永远只占 $\frac{1}{N}$ 的 fresh 比例，整个 diffusion 过程都是。所以 PipeFusion 的 effective staleness 比 DistriFusion 低，理论上 FID 更好（paper Table 2 验证：Pixart 8 device PipeFusion FID=28.46 vs DistriFusion 41.81）。

---

## 4. 通信与内存的精确比较（paper Table 1）

我把 Table 1 拆开讲：

| Method | Comm Cost | Overlap? | Param Memory | KV Buffer Memory |
|---|---|---|---|---|
| Tensor Parallel | $4 O(p\cdot hs) L$ | ✗ | $\frac{1}{N} P$ | $\frac{1}{N} KV$ |
| DistriFusion | $2 O(p\cdot hs) L$ | ✓ | $P$ | $KV \cdot L$ |
| SP-Ring | $2 O(p\cdot hs) L$ | ✓ (in attn) | $P$ | $\frac{1}{N} KV$ |
| SP-Ulysses | $\frac{4}{N} O(p\cdot hs) L$ | ✗ | $P$ | $KV$ |
| **PipeFusion** | $2 O(p\cdot hs)$ | ✓ | $\frac{1}{N} P$ | $\frac{1}{N} (KV) L$ |

变量含义：
- $p$：sequence length（latent pixel 数）
- $hs$：hidden size
- $L$：network layer 数
- $N$：device 数
- $P$：total params
- $KV$：单 layer 全 spatial K/V 大小

algobw factor：
- AllReduce = $2\frac{n-1}{n} \approx 2$（含 reduce-scatter + all-gather 各一圈）
- AllGather = $\frac{n-1}{n} \approx 1$
- All2All = $1$

### 4.1 PipeFusion 通信为什么是 $2 O(p\cdot hs)$ 不带 $L$

这是整个 paper 最数学上的关键点。

- Device $i$ 接收 patch activation（输入到 stage $i$ 第一个 layer）+ 发送 patch activation（stage $i$ 最后一个 layer 输出）
- 每个 patch 一来一回，每个 device 处理 $M$ 个 patches，但每个 timestep 总通信量还是 $2 O(p \cdot hs)$
- **关键**：因为是 pipeline，activation 只在 stage 边界跨 device，stage 内部 $L/N$ 层完全本地计算，不通信。而 SP / DistriFusion 每个 layer 都要 cross-device 通信 K/V

所以只要 $N < 2L$（SD3 $L=38$，Flux $L=57$，随便满足），PipeFusion 通信量就最低。

### 4.2 内存：为什么 PipeFusion 优于 DistriFusion

DistriFusion 每个 device 要存全 spatial K/V for **每个 layer**（因为它在 sequence 维度切，每个 device 都要做 full attention，需要其他 device 的 K/V，所以每个 layer 都要 buffer 全 spatial K/V），共 $KV \cdot L$，与 $N$ 无关。

PipeFusion 每个 device 只负责 $L/N$ 个 layer，每个 layer 的 KV buffer 只存自己负责的 patches（$M/N$ fraction spatial），所以是 $\frac{1}{N}(KV) L$，与 $N$ 反比。

这是 paper Figure 9 里 DistriFusion 在 4096px / Flux OOM 而 PipeFusion 不 OOM 的根本原因。

### 4.3 Parameter 分布

- TP / PipeFusion: $\frac{1}{N} P$（模型越大越友好，Flux 12B 必须 parameter shard）
- SP / DistriFusion: $P$（每个 device 全量 weight，12B Flux 在 L40 48G 上勉强还能塞，但活动空间小）

---

## 5. 实现：async P2P overlap

paper Section 3.3 末段说每个 device 用 **asynchronous P2P** 把 micro-step 的 patch activation 发给下游 device，同时本地计算下一个 patch。

直觉：device 0 算 P0@T 时，它对 device 1 发的 P1@T 已经在 PCIe 上飞了，device 1 收到 P1@T 同时在算 P0@T，到 device 1 算完 P0@T 时它已经收到了 P0@T 的 incoming activation（从 device 0），不用等。

这要求 P2P 必须真异步——CUDA stream 上 `cudaMemcpyAsync` + compute stream 分离。这就是为什么 paper 强调 pipeline 没中间 bubble。

---

## 6. Warmup 阶段

Diffusion 早期 timestep（$t \to T$ 端）输入变化剧烈（pure Gaussian noise → 第一次降噪），temporal redundancy 不成立，stale activation 误差大，会污染后续所有 step。

所以 PipeFusion / DistriFusion 都需要几个 **warmup steps**，这些 step 走同步 SP，不 pipeline（有 bubble），过后再切到 async 模式。

paper Table 3 的数据：

| Warmup | Pixart pp=8 | Pixart cfg=2,pp=8 | SD3 pp=8 | SD3 cfg=2,pp=8 | Flux pp=8 |
|---|---|---|---|---|---|
| 0 | 0.71 | 0.66 | 1.05 | 0.83 | 5.48 |
| 1 | 0.82 (+15%) | 0.69 (+4%) | 1.16 (+10%) | 0.87 (+4%) | 5.53 (+1%) |
| 2 | 0.91 (+28%) | 0.70 (+6%) | 1.27 (+21%) | 0.92 (+11%) | 6.00 (+9%) |

观察：
- Flux 对 warmup 极不敏感（1 step → 1%，2 step → 9%），可能因为 MM-DiT attention 跨模态 mixing 对 stale 容忍度好
- SD3、Pixart 对 warmup 敏感（2 step 涨 21~28%）
- **CFG=2 时 warmup 影响被稀释**——因为两组 pipeline 并行，warmup 代价除以 2

paper 提到把 warmup steps 和 working steps 分到不同 device pool，warmup 用 SP，working 用 PipeFusion，然后 stage 间传 feature map——这是 future work，思路很 systems-y。

---

## 7. CFG Parallel：白嫖 2× 加速

Classifier-Free Guidance（https://arxiv.org/abs/2207.12598）每次 timestep 要跑两次 forward：unconditional branch + conditional branch。

paper 的 CFG parallel 把 8 GPU 切成两组，每组 4 GPU 跑独立 PipeFusion pipeline，一次 forward 同时算两个 branch。这相当于在 "inter-image" 维度并行。

数据上看（paper Section 4.1.1）：
- Pixart 1024px: 1 GPU baseline vs CFG=2,PipeFusion=4 → 5.09× speedup
- Pixart 4096px: 8.59× speedup
- SD3 1024px: 3.11×
- SD3 2048px: 8.16×
- Flux 1024px: 3.42×
- Flux 2048px: 5.79×

注意 2048/4096px 的 speedup 远大于 1024px，因为大分辨率下：
- compute 增长 $p^2$
- 通信相对小，PipeFusion 把通信压到 4.6% CommShare（公式 2）
- scalability 接近完美

公式 2（CommShare）：

$$
\mathrm{CommShare} = \frac{T_{\mathrm{E2E,8GPU}} - \frac{T_{\mathrm{single}}}{8}}{T_{\mathrm{E2E,8GPU}}}
$$

- $T_{\mathrm{E2E,8GPU}}$：8 GPU end-to-end 实测 latency
- $T_{\mathrm{single}}/8$：假设 perfect scaling 下的理想 latency
- 差值 = 通信 + bubble + load imbalance 等所有 overhead

例子：4096px Pixart
- PipeFusion: $(32.1 - 244.89/8)/32.1 \approx 4.6\%$
- SP: $(37.3 - 244.89/8)/37.3 \approx 17.9\%$

绝对通信时间 SP 6.69s → PipeFusion 1.49s，降 78%。即使 compute-bound regime，PipeFusion 也压住了通信。

---

## 8. 实验：8×L40 PCIe（low-bandwidth torture test）

硬件选得很有讲究：L40 是 PCIe Gen4×16，没有 NVLink。这种环境下通信带宽稀缺，最能放大"通信量小"的优势。

### 8.1 Latency（Figure 5/6/7）

**Pixart（0.6B）**：
- 1024px: PipeFusion 0.66s，比第二名快 1.48×
- 2048px: 快 1.55×
- 4096px: 快 1.16×（DistriFusion OOM）
- 速度提升随分辨率下降：compute-bound 后通信比例下降，优势变小

**SD3（2B MM-DiT）**：
- 1024px: 1.57× over best USP
- 2048px: 1.30×

**Flux.1（12B MM-DiT）**：
- 1024px: 1.23× over USP(ulysses=4,ring=2)
- 2048px: 1.25×

Flux 优势比 Pixart 小，因为 12B 模型本身 compute 占主导，通信相对小。但绝对速度仍然最快。

### 8.2 Memory（Figure 9）

- Pixart 8192px（这是极限测试）：PipeFusion 是唯一能跑的（DistriFusion OOM）
- SD3 / Flux 1024/2048px：PipeFusion 整体内存 < SP
  - Flux 1024px: PipeFusion = 32% of SP memory
  - Flux 2048px: 36%
- 注意 PipeFusion 激活内存略高于 SP（因为 KV buffer），但 parameter shard 省下的远大于多花的

### 8.3 NVLink 鲁棒性（Table 4）

8×A100 NVLink 上 Pixart：
- 1024px: Tensor Parallel 1.22s, PipeFusion 0.66s
- 4096px: PipeFusion 22.39s vs SP-Ring 23.31s vs TP 36.33s

NVLink 高带宽下 PipeFusion 仍然最优，只是相对优势收窄（因为通信本来就不是瓶颈了）。这佐证了 PipeFusion 的优势主要来自"通信量"而非"通信能 overlap"——前者是数学上的，后者是硬件上的。

### 8.4 Patch 数 M 的影响（Figure 11）

paper 推荐 $M = N$：
- $M$ 太大：单 operator input 小，GPU 利用率低（kernel launch overhead 占比上升）
- $M$ 太小：overlap 机会少，pipeline bubble 难填
- $M = N$ 是 sweet spot，正好每个 device 一个 patch 在 pipeline 中

---

## 9. Quality 分析（Table 2 + Figure 10）

FID（越低越好），分两种计算：
- "w/ G.T."：与 ground truth COCO image 比
- "w/ Orig."：与 1-Device original 生成的 image 比

**Pixart-XL**：
- Original: 22.78
- DistriFusion 8 dev: 41.81
- PipeFusion 2 dev: 23.60 / 4 dev: 25.23 / 8 dev: 28.46

**Flux.1**：
- Original: 25.03
- PipeFusion 4 dev: 24.17 / 8 dev: 25.97

PipeFusion 全面优于 DistriFusion。原因正是 Section 3.3 分析的 freshness wave：timestep 内 fresh activation 比例从 $\frac{1}{M}$ 单调增到 1，而 DistriFusion 始终是 $\frac{1}{N}$。

更细的直觉：DiT 的 attention 是全 spatial 交互，每个 patch 需要看其他 patches 的 K/V。如果其他 patches 用 stale K/V，等价于在 attention 里引入了 temporal-smoothed context。当 timestep 内部 fresh wave 已经传播到大部分 patches，相当于"在线更新" context，比 DistriFusion 那种"永远只看自己这一份 fresh + 其他全部 stale"的固定模式更接近 ground truth。

---

## 10. Limitation 与延伸方向

paper Section 5 诚实承认：
1. **少 step distilled 模型不适合**（Flux.1-schnell 4-step、one-step diffusion 如 DMD https://arxiv.org/abs/2310.14795, Mean Flows https://arxiv.org/abs/2505.13447）。Temporal redundancy 弱，pipeline 优势消失。但 SOTA 模型仍需 20+ step，影响有限。
2. **不打算单挑所有场景**，建议 hybrid：节点内 NVLink 用 SP，节点间 PCIe / IB 用 PipeFusion。这是大集群部署的 natural composition。

**Video Generation 延伸**（这点很有意思）：
- Wan2.1（https://arxiv.org/abs/2503.20314）、HunyuanVideo（https://arxiv.org/abs/2412.03603）、CogVideo（https://arxiv.org/abs/2205.15868）都是 MM-DiT backbone
- Video latent 形状 (B, L, H)，空间 + 时间维度 flatten 到 L
- PipeFusion 沿 sequence 切，patch = spatiotemporal block
- Video 模型参数更大（HunyuanVideo 13B）、sequence 更长（几万 token），通信放大效应更强 → PipeFusion 优势更大
- 这是个非常 natural 的延伸，几乎不需要改架构

---

## 11. 我自己看完后的几个直觉

### 11.1 "temporal redundancy" 作为 parallelism 的资源

以前大家用 temporal redundancy 做 **compute 优化**（DeepCache https://arxiv.org/abs/2312.00958、TGATE、PAB https://arxiv.org/abs/2408.12588、DiTFastAttn https://arxiv.org/abs/2406.08522、Δ-DiT https://arxiv.org/abs/2406.01125、Learning-to-Cache https://arxiv.org/abs/2406.01733）——把某些 step 的某些层 skip 掉。

DistriFusion 第一个把它用到 **通信 优化**——把 stale K/V 当免费的 future。

PipeFusion 把这招再升一层——把 temporal redundancy 当 **pipeline synchronization barrier** 的替代品。本质上是用"信息时延"换"通信时延"。

这是个很 systems-y 的思路：在 LLM pipeline 里 bubble 是硬问题（1F1B, interleaved, ZB-H1 https://arxiv.org/abs/2407.09067 各种 scheme 都在治 bubble）。DiT 这边因为 temporal redundancy 给了一个"假装拿到数据"的合法手段。

### 11.2 "通信量与 L 无关"为什么这么重要

LLM 训练里 pipeline parallel 之所以能赢过 TP，是因为跨节点带宽差 intra-node 几个量级。DiT inference 现在 8 卡 PCIe 也面对类似 gap——L40 没 NVLink、A100 NVLink 才 NB200 之类。

PipeFusion 把跨 device 通信从"每层一次"降到"整个 forward 两次"，相当于把 transformer 内部彻底本地化。这跟 LLM 那边 sequence parallelism 之于 TP 的关系类似，但 DiT 这边因为是 full attention + iterative + temporal redundancy，多了一个 DistriFusion 没拿到的额外 advantage（parameter + KV 都 shard）。

### 11.3 为什么 freshness wave 不会影响视觉质量

直觉：DiT 早期 layer 提取 low-level feature（颜色、纹理），晚期 layer 做高层 semantic composition。Patch 流过 pipeline 时，每个 patch 在每个 layer 都会被"refresh 一次"。等 patch 走到 layer $L$，它在这个 timestep 的每层都是 fresh。

这跟 DeepCache 的观察一致：浅层 activation 变化快、深层变化慢。PipeFusion 反向利用：**因为变化慢，所以可以 stale；因为每个 patch 总会被 refresh，所以最终 fresh**。两个性质叠起来就是"高质量 + 高吞吐"。

---

## 12. 一句话总结

PipeFusion = "sequence-parallel pipeline with temporal-redundancy-based barrier removal"。它的贡献本质是发现 DiT inference 的两个 redundancy（沿 sequence 切可以独立处理 patch、沿 timestep 切可以用 stale activation）可以 compose，从而同时压低 comm cost（$O(L) \to O(1)$）、memory（$O(L) \to O(L/N)$）、staleness（DistriFusion $\frac{1}{N}$ fresh → PipeFusion freshness wave）。在 PCIe 这种带宽贫瘠场景上，这就是 1.5× 速度 + 不 OOM 的来源。

---

## References

- PipeFusion 项目代码（xDiT 框架）：https://github.com/xdit-project/xDiT
- PipeFusion paper（推测 arXiv 链接，作者 Jiarui Fang 等，ByteDance）：https://arxiv.org/abs/2503.02095（按主题推测，建议以 xDiT repo 内 README 为准）
- DistriFusion（CVPR 2024）：https://arxiv.org/abs/2402.19481
- DeepSpeed-Ulysses：https://arxiv.org/abs/2309.14509
- Ring-Attention：https://arxiv.org/abs/2310.01889
- USP（Unified Sequence Parallelism）：https://arxiv.org/abs/2405.07719
- DiT (Peebles & Xie, ICCV 2023)：https://arxiv.org/abs/2212.09748
- PixArt-α：https://arxiv.org/abs/2310.00426
- PixArt-Σ：https://arxiv.org/abs/2403.04692
- Stable Diffusion 3：https://arxiv.org/abs/2403.03206
- Flux.1 by Black Forest Labs：https://blackforestlabs.ai
- GPipe：https://arxiv.org/abs/1811.06965
- TeraPipe：https://arxiv.org/abs/2102.07188
- Classifier-Free Guidance：https://arxiv.org/abs/2207.12598
- DeepCache：https://arxiv.org/abs/2312.00958
- PAB (Pyramid Attention Broadcast)：https://arxiv.org/abs/2408.12588
- DiTFastAttn：https://arxiv.org/abs/2406.08552
- Δ-DiT：https://arxiv.org/abs/2406.01125
- Learning-to-Cache：https://arxiv.org/abs/2406.01733
- DDIM：https://arxiv.org/abs/2010.02502
- DPM-Solver：https://arxiv.org/abs/2206.00927
- Mean Flows（one-step distillation）：https://arxiv.org/abs/2505.13447
- DMD (Distribution Matching Distillation)：https://arxiv.org/abs/2310.14795
- Wan 2.1 video model：https://arxiv.org/abs/2503.20314
- HunyuanVideo：https://arxiv.org/abs/2412.03603
- CogVideo：https://arxiv.org/abs/2205.15868
- Megatron-LM（TP 原典）：https://arxiv.org/abs/1909.08053
- VAE (Kingma)：https://arxiv.org/abs/1312.6114
- Clean-FID（用于 FID 评估）：https://arxiv.org/abs/2104.11222
- FID 原始定义：https://arxiv.org/abs/1706.08500
- ZB-H1（LLM pipeline bubble 治理参考）：https://arxiv.org/abs/2407.09067
- COCO Captions：https://arxiv.org/abs/1504.00325
