---
source_pdf: Forcing-KV Hybrid KV Cache Compression for.pdf
paper_sha256: 6939dde675dc09883ee4e227231220a296b727b2b014ba11047c831b58ee0a15
processed_at: '2026-08-18T22:37:36-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Forcing-KV

## 先说背景：AR video diffusion 为啥火

最近 video generation 这块有个明显趋势，从 bidirectional 的一次性生成全部帧，转向 autoregressive 的"一段一段生成"模式。代表就是 Self Forcing（https://arxiv.org/abs/2506.24139）和 LongLive（https://arxiv.org/abs/2506.21526）这类工作。为啥要 AR 呢？因为 bidirectional 模型生成 30 秒视频要一次性 denoise 所有帧，计算量爆表；AR 可以 chunk-by-chunk 流式吐帧，首帧延迟低，还能无限长。

但 AR 有个天然的麻烦：**KV cache 越积越多**。

打个比方：bidirectional 模型像拍全景照片，一次性把所有像素拍完；AR 模型像拍视频，每拍一段新内容，之前的所有片段都得留在内存里供参考。生成 30 秒 1080P 视频时，KV cache 单卡就吃掉 60+ GB GPU 内存，attention 计算量随时间线性涨，最后 self-attention 占掉整个 transformer block 89% 的时间（Appendix K, Figure 13）。

这就尴尬了：AR 本来是为了 streaming 实时性设计的，结果长视频一跑就卡在 KV cache 上。

## 已有方法的毛病

这块有几路人马在搞：

**Sparse attention 一派**（Sparse VideoGen, MonarchRT, Light Forcing）：让 attention 少算点，但 cache 还在内存里躺着。

**Feature caching 一派**（Flow Caching, Timestep Embedding Caching）：cache 的是 intermediate feature，完全不碰 KV cache。

**StreamingLLM**（https://arxiv.org/abs/2309.17453）：老牌方法，只保留开头的 sink frame + 最近的 window。简单粗暴，所有 head 一视同仁。

**Dummy Forcing**（https://arxiv.org/abs/2601.20499）：AR video 专属，观察到有些 head 只看 local chunk，就丢掉这些 head 的历史 cache。但问题来了——丢太狠了，chunk 边界出现 flickering，视频一段一段接不上，dynamic degree 也暴跌（从 42.54 掉到 26.56，Table 1）。

Dummy Forcing 的毛病在于：它知道"有些 head 不怎么用历史"，但没搞清楚"为什么不用、什么时候必须用"。

## Forcing-KV 的核心发现：head 有分工

这篇 paper 做的第一件事，不是急着提方法，而是耐心观察 AR video diffusion 模型的 attention head 到底在干嘛。

他们画了 Wan2.1、SkyReels-V2、Self Forcing、LongLive 这些模型的 attention map，发现 head 可以明确分成两类：

### Static head（静态头）

attention map 长这样：一大块亮区在当前 chunk 上，然后历史 cache 里只有**最近那一帧**（paper 叫 transition anchor frame）亮，其他历史帧全暗。

功能上，static head 干的是"把当前 chunk 内部的像素画好"和"和上一个 chunk 的最后一帧接上"这两件事。也就是说，它管的是 intra-frame fidelity 和 chunk 间视觉连续性。

为啥是最近一帧而不是更早的帧？因为 AR 生成是 Markovian chain，新 chunk 要从前一个 chunk 平滑过渡，最有用的信息就是紧挨着的那一帧。这跟 bidirectional 模型不一样，bidirectional 可以双向看，transition 信息分散在 local temporal range 内。

### Dynamic head（动态头）

attention map 长这样：一条条等间隔的对角线 stripe。为啥是对角线？因为每帧 token 数固定（$F$ 个），每 chunk 帧数固定（$C$ 个），所以同一空间位置在不同帧的 key 上有固定 stride $F$。dynamic head 就是让当前 chunk 的每个空间位置去对应找历史帧里"同一个空间位置"的信息。

功能上，dynamic head 管的是 subject consistency（主体一致性）和 motion dynamics（运动动态）。比如一个人挥手，dynamic head 负责确保手在前后帧里是同一只手，且动作连贯。

### 为啥这个分工很关键

这个分工在 100 个 prompts × 4 个 denoising step 上都稳定（Figure 3e, PCA 可视化）。intra-head divergence 0.16，inter-head divergence 0.83，head 类型聚类非常紧。意味着 head 类型可以 offline 一次性 profiling 出来，不需要每次推理都重算。

更关键的：**static head 根本不在乎远距离历史帧**。你把它的历史 cache 全 mask 掉，dynamic degree 和 consistency 几乎不变；但你要是 mask 掉它的 transition anchor frame，chunk discontinuity 立刻从 2.4 飙到 4.1（Table 3, w/o static-head cache）。

这就是 Dummy Forcing 没注意到的地方：它一刀切掉历史 cache，把 transition anchor 也一起切了，导致 chunk 接缝处崩溃。

## 方法：根据 head 分工对症下药

既然 static head 和 dynamic head 用 cache 的方式完全不同，那就别一锅炖了，分别处理：

### 对 static head：structural pruning（公式 2）

$$
\mathbf{O}_i^{static} = \mathrm{Attention}\Big( Q_{iCF:(i+1)CF}, [K_{sink}, K_{(iC-1)F:iCF}, K_{iCF:(i+1)CF}], [V_{sink}, V_{(iC-1)F:iCF}, V_{iCF:(i+1)CF}] \Big)
$$

翻译成人话：static head 只保留三类东西的 K, V：
- $K_{sink}, V_{sink}$：sink frame（开头那几帧，某些模型训练时有特殊处理，必须留）
- $K_{(iC-1)F:iCF}$：transition anchor frame（前一个 chunk 的最后一帧，$(iC-1)$ 是帧索引，$F$ 是每帧 token 数）
- $K_{iCF:(i+1)CF}$：当前 chunk 自己的 K, V

其他所有历史帧的 cache 直接丢掉。为啥敢这么狠？因为 observation 2 已经证明了 static head 对远距离历史不敏感。

### 对 dynamic head：segment-wise similarity pruning（公式 3）

$$
s_t^{(j)} = \mathrm{Cosine}(K_{t,j}, K_{t+1,j}), \quad \mathcal{I}_t^{keep} = \mathrm{BottomK}\Big( \{s_t^{(j)}\}_{j=1}^{n}, \lfloor rn \rfloor \Big)
$$

变量解释：
- $K_{t,j}$：第 $t$ 帧的第 $j$ 个 segment 的 key（每帧切成 $n=6$ 个 segment）
- $s_t^{(j)}$：第 $t$ 帧第 $j$ segment 和第 $t+1$ 帧同 segment 的 cosine similarity
- $\mathrm{BottomK}$：取 similarity 最低的 $\lfloor rn \rfloor$ 个 segment 保留（$r=0.3$ 默认，意味着保留 70%）
- $\mathcal{I}_t^{keep}$：保留的 segment 索引集

翻译成人话：把每帧切成 6 个 segment，算相邻帧对应 segment 的相似度，相似度高的（说明几乎没变化，比如背景）丢掉，相似度低的（说明变化大，比如运动物体）保留。

**为啥是 segment-wise 不是 token-wise**？因为视频里相邻帧的相似性是区域性的——一块背景区域整体都不变，token-wise 随机切容易切断连续性。segment-wise 更对齐视频内容的连续性（Figure 7 ablation 验证）。

**关键工程 trick**：similarity 计算只在 DiT 的第一个 block 做，用它的 key 作为 proxy。因为不同 block 的 attention pattern 相似，没必要每个 block 都算一遍 similarity。这极大降低 profiling 开销。

### Offline head profiling（公式 1）

$$
\mathrm{HeadType} = \begin{cases} \mathrm{Static}, & \text{if } \frac{\mathcal{A}_{Generate} + \mathcal{A}_{Transition}}{\mathcal{A}_{Total} - \mathcal{A}_{Sink}} > \alpha \\ \mathrm{Dynamic}, & \text{otherwise} \end{cases}
$$

变量：
- $\mathcal{A}_{Total}$：某 head 在整个 attention window 上的 attention mass 总和
- $\mathcal{A}_{Generate}$：落在当前 generated chunk 上的 attention mass
- $\mathcal{A}_{Transition}$：落在 transition anchor frame 上的 attention mass
- $\mathcal{A}_{Sink}$：落在 sink frame 上的 attention mass（排除掉，因为某些模型训练 recipe 对 sink 有特殊处理）
- $\alpha = 0.8$：阈值，从 0.5 到 0.8 不敏感（Table 3 显示 dynamic degree 仅从 42.87 变 43.65）

翻译：一个 head 把多少注意力放在"当前 chunk + 最近一帧"上，超过阈值就是 static，否则是 dynamic。单 prompt 几分钟搞定。

## 实验结果：快了还更好

### LongLive 30s（Table 1）

| 方法 | Speedup | Chunk Disc.↓ | Dynamic Degree↑ | Total Score↑ |
|---|---|---|---|---|
| Full KV | 1.00× | 2.4 | 42.54 | 80.38 |
| StreamingLLM | 1.06× | 2.4 | 40.93 | 80.38 |
| Dummy Forcing (L=1) | 1.30× | 3.0 | 26.56 | 79.37 |
| Dummy Forcing (L=2) | 1.24× | 2.3 | 33.90 | 79.61 |
| **Forcing-KV** | **1.27×** | **2.4** | **43.65** | **80.65** |

重点看：
- vs Dummy Forcing (L=1)：速度一样，但 Dynamic Degree 43.65 vs 26.56，直接救回来。Total Score 80.65 vs 79.37。
- vs Full KV：**质量反而略升**（80.65 > 80.38）。作者解释：base model 训练时让 static head 也 attend 远距离 context，但模型没学好这个能力，远距离 cache 反而是噪声，压缩掉反而受益。

### Self Forcing 30s

| 方法 | Speedup | Chunk Disc.↓ | Dynamic Degree↑ | Total Score↑ |
|---|---|---|---|---|
| Full KV | 1.00× | 3.4 | 46.86 | 79.72 |
| Dummy Forcing (L=1) | 1.56× | 3.5 | 46.55 | 79.95 |
| **Forcing-KV** | **1.50×** | **2.7** | **52.23** | **80.07** |

Dummy Forcing 速度更快（1.56× vs 1.50×），但 Chunk Disc. 3.5 是最差的，Forcing-KV 拿到 2.7 同时 Dynamic Degree 52.23。

### Scaling law（Figure 6）

分辨率从 480P 涨到 1080P，Forcing-KV 的 speedup 从 1.40× 涨到 2.82×。因为 attention 是 $\mathcal{O}(n^2)$，cache 越大压缩收益越大。这是 paper 强论点：**越长的 context、越大的分辨率，收益越显著**。

### User study（Figure 5）

12 人 × 15 组 × 3 视频 = 540 评估。vs Dummy Forcing：
- Visual Quality: 45.0% vs 5.0%（碾压）
- Dynamic Degree: 52.8% vs 42.2%
- Consistency: 47.2% vs 42.8%

benchmark 上的提升在 human perception 上对应得很清楚。

### 正交可叠加

FP8 quantization（SageAttention2, https://arxiv.org/abs/2410.02325）叠加 Forcing-KV：LongLive 30s 28.05 FPS（1.37×），几乎无损（Table 6）。

Krea-Realtime-14B 上也验证有效（Table 4, 1.26× speedup, Total Score 略升）。

## 一个有趣的 bonus 发现

Appendix J（Figure 12）：dynamic head 约占 60%，且集中在 middle layers（layer 13, 15, 17）。推测浅层/深层关注 structural/local info，中间层更利用 context 做 feature refinement。这个 head 分布的可解释性是个有意思的研究方向。

## 我的直觉和联想

**为啥这个工作有意义**

它把"AR video diffusion 的 KV cache 利用有 inductive bias"这件事用三个 observation 讲清楚了，并基于此给出对应设计。这不是单纯工程 trick，而是关于 AR video diffusion 模型功能分化的经验性研究 + 对应 compression 设计。

**transition anchor frame 这个概念是关键 insight**

AR video diffusion 的 KV cache 不是随意的 cache，而是有"近邻 anchor + 长程对应"的双层结构。这跟 video super-resolution 的 reference frame、DDIM inversion anchor 有异曲同工之妙。这种结构在 video LLM、video understanding transformer 里可能也存在。

**几个可以延伸的方向**

1. **Adaptive compression ratio r**：现在固定 0.3，但高动态视频和低动态视频最优 r 应该不同。可以基于相邻帧相似度分布动态决定 prune 多少。

2. **Multi-anchor**：现在只保留最近一帧作为 anchor，但 L=2 在 Dummy Forcing 上显示更好（chunk disc 2.9 vs 3.6）。Forcing-KV 是否可以支持 "anchor + 1 frame backoff"？

3. **训练阶段 cache 压缩**：paper Limitations 明确说没碰训练。Self Forcing 训练时 self-rollout 也累积 KV cache，长视频训练更贵。Causal Forcing（https://arxiv.org/abs/2602.02214）这类 distillation 工作和 Forcing-KV 结合可能很有意思。

4. **Retrieval for dynamic head**：dynamic head 现在 full history 都保留（只是 segment-wise 压缩），memory 还大。retrieval-based 方法（如 Reformer 的 LSH）只在 query time 检索相关历史 segments，可能进一步压缩。但 retrieve 引入额外 latency 要权衡。

5. **理论分析**：为啥 dynamic head 学到 stripe pattern？为啥 static head 关注 transition anchor？这是 Self Forcing training 的 inductive bias 还是 architecture property？理论上可以分析 Self Forcing 训练目标下的最优 attention pattern。

6. **与 Linear Attention 的 unify**：Sana-Video（ICLR 2026）的 block linear attention 思路和 segment-wise similarity pruning 在 spirit 上有点像（都基于局部相似性减少 attention 计算量）。能不能 unify？

7. **Head 分布的可解释性**：Appendix J 显示 dynamic head 集中在中间层。这让我联想到 LLM 里的 induction head 分布、Circle Attention 里发现的某些 layer 特化现象。AR video diffusion 的中间层特化做 feature refinement，跟 LLM 中间层做 reasoning 的 hypothesis 有呼应。

**一句话总结 intuition**

AR video diffusion 的 attention head 有明确分工：static head 管"当前帧画好 + 和前一帧接上"，dynamic head 管"主体一致性 + 运动连贯"。对 static head 保留 sink + 最近一帧 + 当前 chunk 就够，其他历史全丢；对 dynamic head 基于相邻帧相似度丢弃冗余 segment。这套方法 training-free、随分辨率/window size scaling 收益递增、与 quantization 正交可叠加。

Project page: https://zju-jiyicheng.github.io/Forcing-KV-Page

希望能帮你 build 起直觉。如果想深入某个具体点（比如 segment-wise similarity 的设计 choice、transition anchor 的更深 ablation、或与 DuoAttention 的类比），可以继续聊。

---

# Forcing-KV: Hybrid KV Cache Compression for AR Video Diffusion — 深入讲解

非常好的 paper，by Yicheng Ji et al. from ZJU / Video Rebirth / HKUST。这篇工作做的事，是在 AR video diffusion model（如 Self Forcing、LongLive、SkyReels-V2、Krea-Realtime-14B 等）的 KV cache 上做 training-free 的 hybrid 压缩。我尝试把 intuition 和技术细节都讲清楚。

Project page: https://zju-jiyicheng.github.io/Forcing-KV-Page

---

## 1. 为什么这篇 paper 重要 — 问题的根源

AR video diffusion model 的核心范式来自 SelfForcing（NeurIPS 2025, https://arxiv.org/abs/2506.24139）和它的后继者 LongLive（ICLR 2026）、Self-Forcing++、Rolling Forcing、Reward Forcing 等。这类模型生成 video 的方式是 chunk-by-chunk：每生成一个 chunk $C$ 帧，就把这部分 token 的 $K, V$ 写入 cache，下一个 chunk 的 $Q$ 去 attend 全部历史 cache。

这就导致两个 scaling 病：

1. **Attention complexity**: $Q$ 是当前 chunk 的长度 $CF$（每帧 $F$ 个 token），$K, V$ 是累积历史 $iCF$，attention 复杂度 $\mathcal{O}(CF \cdot iCF)$，随 AR step $i$ 线性增长。随 sequence 长度增加，self-attention 在 DiT block 中的占比从 24% 涨到 89%（Appendix K, Figure 13）。
2. **Memory overhead**: 1080P 下 30 秒视频的 KV cache 单卡就 60+ GB，无法在 memory-constrained 场景部署。

**已有加速方法的不足**：
- Sparse attention（Sparse VideoGen, MonarchRT, Light Forcing）：减少 attention FLOPs，但不减少 cache size。
- Feature caching（Flow Caching, Timestep Caching）：作用在 intermediate feature 上，不动 KV cache。
- StreamingLLM（sink + recent window）：naive 滑窗，不利用 head 功能分化。
- Dummy Forcing（https://arxiv.org/abs/2601.20499）：观察到某些 head 只看 local chunk，丢掉历史 cache，但缺深入分析、导致 chunk discontinuity 严重（dynamic degree 从 43.56 掉到 26.02）。

**核心 motivation**: 想要找出 AR video diffusion model 的 KV cache 利用上有什么 inductive bias 是可被利用的，从而做语义化的压缩而不是均匀压缩。

---

## 2. 三个核心观察

### Observation 1: Static Head vs Dynamic Head 模式

在 Wan2.1, SkyReels-V2, Self Forcing, LongLive 上跑了 VBench prompts（Figure 2, Figure 9），把 attention map 画出来：

**Static head** 的 pattern：
- attention mass 集中在 current chunk（对角线附近一整块亮区）
- 同时对 **transition anchor frame**（即历史 cache 中最近的那一帧）有显著 attention
- 用途：维持 intra-frame fidelity + chunk 间视觉连续性（结构骨架）

**Dynamic head** 的 pattern：
- attention map 上呈现等间隔 diagonal stripe
- 物理解释：每帧 token 数固定（$F$），每 chunk 帧数固定（$C$），所以同一 spatial region 在不同帧的 key 上有固定 stride $F$。dynamic head 就是让每个生成 region 去找它在历史帧中"同一 spatial 位置"的 region。
- 用途：subject consistency + motion dynamics（长程时序对应）

这里有个很重要的 insight：**transition anchor frame** 这个概念是 AR 模型特有的。bidirectional 模型（如 Wan2.1）的 transition 通过 bidirectional local attention 实现，而 AR 模型因为 Markovian chain，transition 信息主要落在"前一帧"这一 anchor 上，而不是整个历史。这为后面 static head 的 structural pruning 埋下伏笔。

### Observation 2: 功能验证

用 LongLive 在 MovieGen 128 prompts 上做 progressive masking ablation（Figure 3 a-c）：

- 对 **dynamic head** 逐步 mask 历史帧 → dynamic degree 和 consistency 渐进下降
- 对 **static head** mask 大部分历史 → 几乎没影响
- 但对 **static head** mask 掉 transition anchor frame → **chunk discontinuity 飙升**（4.1 vs 2.4，Table 3）

这说明：static head 关心的不是"远"而是"近"，缺近的 anchor 就会出现 chunk 边界的 flickering/broken transition。Dummy Forcing 的失败正是因为丢掉了这个 anchor。

同时 Figure 3(d) 显示相邻帧的 key 状态 cosine similarity 普遍较高，但不同 segment 差异大——这是 dynamic head 做 segment-wise similarity pruning 的依据。

### Observation 3: Head 性质稳定

100 个 VBench prompts × 4 denoising step，对 random subset 的 head 提取 frame-wise attention feature，PCA 可视化（Figure 3e）：
- intra-head divergence: 0.16
- inter-head divergence: 0.83
- 每个 head 形成紧密 cluster

意义：head 类型可以 offline 一次性 profiling 出来，不需要每次推理都算。

---

## 3. 方法：Forcing-KV

整体流程（Figure 4）：
1. Offline head profiling（一次性）
2. Inference 时对 static head 用 structural pruning
3. Inference 时对 dynamic head 用 segment-wise similarity pruning

### 3.1 Offline Head Profiling

公式 (1)：

$$
\mathrm{HeadType} = \begin{cases} \mathrm{Static}, & \text{if } \frac{\mathcal{A}_{Generate} + \mathcal{A}_{Transition}}{\mathcal{A}_{Total} - \mathcal{A}_{Sink}} > \alpha \\ \mathrm{Dynamic}, & \text{otherwise} \end{cases}
$$

变量解释：
- $\mathcal{A}_{Total}$：某 head 在整个 attention window 上的 attention mass 总和（通常归一化后是 1）
- $\mathcal{A}_{Generate}$：落在当前 generated chunk 上的 attention mass
- $\mathcal{A}_{Transition}$：落在 transition anchor frame 上的 attention mass
- $\mathcal{A}_{Sink}$：落在 sink frame 上的 attention mass（被排除，因为某些模型训练 recipe 对 sink frame 有特殊处理，如 LongLive、Reward Forcing、Infinite-Rope）
- $\alpha$：模型特定超参，默认 $\alpha = 0.8$；$\alpha$ 从 0.8 降到 0.5 仅使 dynamic degree 略降（43.65 → 42.87），对超参不敏感
- 单 prompt 几分钟内完成 profiling

这个 criterion 本质上和 H2O (https://arxiv.org/abs/2306.14048) 的"低 accumulated attention mass 是 better eviction candidate"一致。

### 3.2 Static Structural Pruning for Static Heads

公式 (2)，对第 $i$ 个 AR step：

$$
\mathbf{O}_i^{static} = \mathrm{Attention}\Big( Q_{iCF:(i+1)CF}, [K_{sink}, K_{(iC-1)F:iCF}, K_{iCF:(i+1)CF}], [V_{sink}, V_{(iC-1)F:iCF}, V_{iCF:(i+1)CF}] \Big)
$$

变量：
- $C$：每个 chunk 的帧数
- $F$：每帧的 token 数
- $Q_{iCF:(i+1)CF}$：第 $i$ 个 chunk 的所有 query（shape: $[CF, d]$）
- $K_{sink}, V_{sink}$：sink frame 的 K, V
- $K_{(iC-1)F:iCF}$：transition anchor frame（第 $iC-1$ 帧，即前一个 chunk 的最后一帧）的 K
- $K_{iCF:(i+1)CF}$：当前 chunk 自己的 K

也就是 static head 只保留：sink + transition anchor + current chunk，其他历史帧全部丢弃。这里 transition anchor 的下标是 $(iC-1)$，这是为什么 $i \geq 2$ 才能开始压缩（i=1 时没历史）——所以第一个 AR step 用 full KV cache。

### 3.3 Dynamic Similarity Pruning for Dynamic Heads

公式 (3)：

$$
s_t^{(j)} = \mathrm{Cosine}(K_{t,j}, K_{t+1,j}), \qquad \mathcal{I}_t^{keep} = \mathrm{BottomK}\Big( \{s_t^{(j)}\}_{j=1}^{n}, \lfloor rn \rfloor \Big)
$$

变量：
- $K_{t,j}$：第 $t$ 个历史帧的第 $j$ 个 segment 的 key
- $n$：每帧划分的 segment 数，默认 $n = 6$（contiguous 切分）
- $s_t^{(j)}$：第 $t$ 帧的第 $j$ segment 与第 $t+1$ 帧同 segment 的 cosine similarity
- $r$：压缩比，默认 $r = 0.3$（即丢掉 30% 最高相似度的 segment，保留 70%）
- $\mathrm{BottomK}$：取 similarity 最小的 $\lfloor rn \rfloor$ 个 segment 索引，即保留变化大的部分（不冗余的部分）
- $\mathcal{I}_t^{keep}$：保留的 segment 索引集

**关键实现 trick**: 计算 segment-wise similarity 只用 **第一个 DiT block** 的 key 作为 proxy，避免在所有 block 上算。这极大降低 profiling 开销。这是因为 attention pattern 在不同 block 间相似（前面 Observation 也支持）。

压缩后的 K, V：

$$
\widetilde{K}_{\mathcal{H}} = \bigcup_{t \in \mathcal{H}} \{ K_{t,j} \mid j \in \mathcal{I}_t^{keep} \}, \quad \widetilde{V}_{\mathcal{H}} = \bigcup_{t \in \mathcal{H}} \{ V_{t,j} \mid j \in \mathcal{I}_t^{keep} \}
$$

公式 (4)：

$$
\mathbf{O}_i^{dynamic} = \mathrm{Attention}\Big( Q_{iCF:(i+1)CF}, [K_{sink}, \widetilde{K}_{\mathcal{H}}, K_{iCF:(i+1)CF}], [V_{sink}, \widetilde{V}_{\mathcal{H}}, V_{iCF:(i+1)CF}] \Big)
$$

直觉上：dynamic head 的 cache 保留"变化大"的 segment，丢掉"几乎不变"的 segment（背景、静态物体）。这和人类视觉冗余度的直觉一致：相邻帧大片相同区域的信息量低。

**为什么 segment-wise 比 token-wise 好**（Figure 7 ablation）：相邻帧往往局部区域整体相似，token-wise random/uniform pruning 容易打断连续性，segment-wise 保留内容连续性。这也呼应 Figure 2 中 dynamic head 的 stripe "有一定宽度"——意味着对应的 region 是有 spatial extent 的。

---

## 4. 评估指标：Chunk Discontinuity

paper 提了一个新 metric（Appendix A），用 RAFT optical flow：

$$
\delta_t = \Delta \mathrm{RAFT}(I_t, I_{t+1}), \quad t = 1, \ldots, F-1
$$

$$
\mathrm{Chunk\ Disc.} = \frac{\mathrm{Sum}(\mathrm{Top}_{K-1}(\{\delta_t\}_{t=1}^{F-1})) / (K-1)}{\mathrm{Sum}(\{\delta_t\}_{t=1}^{F-1}) / (F-1)}
$$

变量：
- $I_t$：第 $t$ 帧
- $\Delta \mathrm{RAFT}$：RAFT 网络算出的相邻帧 optical flow 差异（衡量像素级运动变化）
- $F$：总帧数
- $K$：chunk 数（即 video 由 $K$ 个 AR chunk 拼成）
- 分子：Top-$K-1$ 个最大 $\delta_t$ 的均值（即 chunk 边界处的跳变期望）
- 分母：所有 $\delta_t$ 的均值（即整体平均跳变）

直觉：如果 chunk 边界处 optical flow 突然跳变大，分子高，metric 高 = 不连续。Figure 8 给的 case study 显示在 fixed-interval chunk boundary 处 optical flow 差异出现规律的局部最大值——这恰是 Dummy Forcing 出问题的根源。

---

## 5. 实验数据深度解读

### 5.1 主结果（Table 1, Table 2）

**LongLive 30s（VBench-Long）**：

| Method | FPS↑ | Speedup↑ | Chunk Disc.↓ | Dynamic Degree↑ | Total Score↑ |
|---|---|---|---|---|---|
| Full KV | 21.10 | 1.00× | 2.4 | 42.54 | 80.38 |
| StreamingLLM | 22.34 | 1.06× | 2.4 | 40.93 | 80.38 |
| Dummy Forcing (L=1) | 27.45 | 1.30× | 3.0 | 26.56 | 79.37 |
| Dummy Forcing (L=2) | 26.21 | 1.24× | 2.3 | 33.90 | 79.61 |
| **Forcing-KV** | 26.77 | 1.27× | 2.4 | **43.65** | **80.65** |

关键 takeaways：
- vs StreamingLLM：速度快不少（1.27× vs 1.06×），质量持平甚至略升
- vs Dummy Forcing：速度相当，但 Dynamic Degree 从 26.56 救回到 43.65（+64%），chunk discontinuity 从 3.0 降到 2.4
- 对比 Full KV：Total Score 80.65 > 80.38，**轻微提升**。作者推测是 base model 训练时把 static head 强行 attend 到远距离 context，但模型其实没学好这个能力，反而引入噪声；压缩掉反而受益。这点在 StreamingLLM 上也看得到（80.38 vs 80.38 持平），说明远距离 cache 对 static head 是冗余的。

**Self Forcing 30s**：

| Method | Speedup↑ | Chunk Disc.↓ | Dynamic Degree↑ | Total Score↑ |
|---|---|---|---|---|
| Full KV | 1.00× | 3.4 | 46.86 | 79.72 |
| StreamingLLM | 1.23× | 2.8 | 54.50 | 80.06 |
| Dummy Forcing (L=1) | 1.56× | 3.5 | 46.55 | 79.95 |
| Dummy Forcing (L=6) | 1.24× | 3.4 | 50.47 | 79.78 |
| **Forcing-KV** | 1.50× | **2.7** | **52.23** | **80.07** |

注意 Self Forcing 的 Dummy Forcing (L=1) 在 dynamic degree 上甚至超过 Full KV，但 chunk discontinuity 3.5 是最差的——这是 Dummy Forcing 的 trade-off：local 完整但 transition 烂。Forcing-KV 同时拿到最好的 chunk disc 和最高的 dynamic degree。

**Self Forcing 5s（Table 2）**: Dynamic Degree 69.17 vs Full KV 66.39（+2.78）。短视频上 dynamic head 的 cache 仍冗余。

### 5.2 Scaling Law（Figure 6）

随分辨率从 480P → 1080P，attention window 增长，KV cache 二次增长，Forcing-KV 的 speedup 从 1.40× 增到 2.82×。这是 paper 的强论点：**越长的 context、越大的分辨率，收益越显著**。因为压缩比固定时，attention 在总时间中占比越大，节省越多。

附录 K（Figure 13）补了一个证据：随 sequence length 增长，self-attention 在 DiT block 的耗时占比从 24% → 61% → 89%。这意味着 self-attention 是真正的瓶颈，KV cache 压缩是正确的"刀刃"。

### 5.3 Ablation（Table 3, LongLive 30s）

**Head Profiling ablation**：

| Method | Chunk Disc.↓ | Dynamic Degree↑ | Total Score↑ |
|---|---|---|---|
| Forcing-KV (α=0.8) | 2.4 | 43.65 | 80.65 |
| Forcing-KV (α=0.5) | 2.4 | 42.87 | 80.63 |
| Random Profiling | 2.5 | 40.72 | 80.44 |
| Human Profiling | 2.3 | 44.44 | 80.71 |

- 自动 profiling 距离 human profiling 仅差 0.06
- Random profiling 主要伤 dynamic degree（dynamic head 误判为 static 导致丢失历史）

**Hybrid compression ablation**：

| Method | Chunk Disc.↓ | Dynamic Degree↑ | Total Score↑ |
|---|---|---|---|
| Forcing-KV | 2.4 | 43.65 | 80.65 |
| w/o static-head cache | **4.1** | 42.58 | 79.57 |
| w/o dynamic-head cache | 2.6 | **40.78** | 80.25 |

完美验证 Observation 2 的功能对应：
- 去 static cache → chunk discontinuity 飙升（2.4 → 4.1）
- 去 dynamic cache → dynamic degree 下降（43.65 → 40.78）

### 5.4 Dynamic Pruning 策略 ablation（Figure 7）

random token / uniform token / similarity segment 三种 pruning，segment-wise similarity 始终 dynamic degree 最高。原因是相邻帧相似性高（Figure 3d 验证），segment-wise 与视频内容连续性 align。

### 5.5 FP8 集成（Table 6）

Forcing-KV + FP8 (SageAttention2, https://arxiv.org/abs/2410.02325)：
- LongLive 30s: 28.05 FPS (1.37×)，dynamic degree 43.27（几乎无损）
- Self Forcing 30s: 27.55 FPS (1.55×)，dynamic degree 52.15

证明与 quantization 正交，可叠加。

### 5.6 14B 模型（Table 4, Krea-Realtime-14B）

| Method | FPS↑ | Speedup↑ | Total Score↑ |
|---|---|---|---|
| Full KV | 4.13 | 1.00× | 84.33 |
| Forcing-KV | 5.22 | 1.26× | 84.59 |

14B 模型上 speedup 1.26×，质量略升，证明方法对模型规模 generalize。

### 5.7 交互式长视频（Table 5, 60s）

LongLive interactive prompts，6 段每段 10s：
- LongLive: 23.07 FPS, Total 78.63
- Dummy Forcing: 25.74 FPS, Total 78.04
- Forcing-KV: 26.35 FPS, Total **79.52**

Dynamic Degree: Forcing-KV 36.11 vs LongLive 26.39（+37%）。这种连续 interactive 长视频正是 Forcing-KV 的甜区。

### 5.8 User Study（Figure 5, 12 人 × 15 组 × 3 视频 = 540 评估）

vs Dummy Forcing：
- Visual Quality: 45.0% vs 5.0%（碾压）
- Dynamic Degree: 52.8% vs 42.2%
- Consistency: 47.2% vs 42.8%

benchmark 上的 dynamic degree 提升在 human perception 上对应得很清楚。

---

## 6. Head 分布的 bonus 发现（Appendix J, Figure 12）

dynamic head ≈ 60%，且集中在 middle layers (13, 15, 17)。作者推测：
- 浅层/深层关注 structural/local info（保 local quality）
- 中间层更利用 context 做 feature refinement（保 detail consistency & temporal dynamics）

这是个挺有意思的研究方向——可解释性 / interpretability of AR video diffusion 的中间层。

---

## 7. 与其他相关工作的位置

| 方法 | 作用层面 | 是否减 cache | 是否分 head | 是否保 transition anchor |
|---|---|---|---|---|
| StreamingLLM (https://arxiv.org/abs/2309.17453) | KV cache | Yes | No | No（窗口） |
| H2O (https://arxiv.org/abs/2306.14048) | KV cache | Yes | No | No |
| DuoAttention (https://arxiv.org/abs/2410.10819) | KV cache | Yes | Yes（retrieval/streaming） | No |
| Sparse VideoGen (https://arxiv.org/abs/2502.09892) | attention | No | Yes（spatial/temporal） | N/A |
| Dummy Forcing (https://arxiv.org/abs/2601.20499) | KV cache | Yes | Yes（local/neighbor） | **No** |
| **Forcing-KV** | KV cache | Yes | Yes（static/dynamic） | **Yes** |

DuoAttention 是 LLM 上的"retrieval vs streaming head"二分法，思想最接近，但 LLM 的 head 不需要 transition anchor 这种 AR video 特有概念。Forcing-KV 是把这个思想 extend 到 AR video diffusion 并加入了 transition anchor 和 segment-wise similarity 这两个 video-specific 设计。

---

## 8. 我的几个 critique 与联想

**Strengths**:
1. Observation 部分扎实：跨 4 个模型（bidirectional + AR, many-step + few-step）验证 pattern 的普遍性。
2. 方法的两个组件（structural pruning for static, similarity pruning for dynamic）都有 observation 直接支撑，不是凭空设计。
3. Chunk discontinuity metric 设计巧妙，填补了 VBench 评测上的空白——传统 metric 对 chunk 边界 flickering 不敏感。
4. 训练 free，与 FP8 quantization 正交可叠加，工程友好。
5. Scaling law 实验（Figure 6）证明了在高分辨率长视频上的 growing benefit。

**可能的 limitation / 联想到的问题**:

1. **Compression ratio r 是固定的 0.3**：不同视频内容（高动态 vs 低动态）最优 r 应不同。一个 video-specific 的 adaptive r 可能更优——尤其在 dynamic head 上，可以基于相邻帧相似度的分布动态决定 prune 多少。例如 background-only 视频可以 prune 70%，快速运动视频只能 prune 10%。

2. **n = 6 segments** 是 uniform contiguous 划分。如果用 spatial pooling 或 learned segmentation（基于 patch 的 semantic group），segment 划分可能更对齐 content boundary。Sana 那篇 linear attention 思路可以借鉴。

3. **Transition anchor frame** 是"最近一帧"，但 Observation 没探索 multiple anchor frames（如最近 2-3 帧）会不会更好。对 L=2 vs L=1 的 ablation 在 Dummy Forcing 上确实显示 L=2 更好（chunk disc 2.9 vs 3.6）。Forcing-KV 是否可以支持 "anchor + 1 frame backoff" 进一步降 discontinuity？

4. **第一个 AR step 不压缩**——这是边界效应。对长视频无所谓，对短视频（5s = 1-2 chunk）影响大。Table 2 上 5s 的 speedup 确实比 30s 略低（1.35 vs 1.27 LongLive 上反过来，但 Self Forcing 5s 1.44 vs 30s 1.50）。可以考虑对第一个 chunk 也做 intra-chunk pruning（基于 chunk 内的 segment similarity）。

5. **Profiling 用 single prompt**——文中说"几分钟完成"。但不同 prompt 的 attention pattern 真的稳定到单 prompt 够吗？Observation 3 的 PCA 显示 intra-head divergence 0.16 vs inter-head 0.83，量级差 5×，但 0.16 也不是 0。一个 robustness 检查是用 multi-prompt profiling 的 head classification 一致性（不同 prompt 是否分到同一类）。这对一些 edge case prompt 可能不稳。

6. **训练阶段的 KV cache 压缩**: paper 在 Limitations 里明确说没碰训练。Self Forcing 训练时 self-rollout 也累积 KV cache，长视频训练更贵。把这套方法 extend 到 training 阶段（distill 一个"压缩感知"的学生模型）是显然的 follow-up。Causal Forcing（https://arxiv.org/abs/2602.02214）这类 distillation 工作和 Forcing-KV 结合可能很有意思。

7. **理论分析缺失**: 为什么 dynamic head 学到 stripe pattern？为什么 static head 关注 transition anchor？这是 inductive bias from Self Forcing training 还是 architecture property？理论上可以分析 Self Forcing 训练目标下的最优 attention pattern——这可能像 Zipf's law 在 LLM 中的 emergent pattern 一样有 principled 解释。

8. **Retrieval for dynamic head**: paper 提到 "A potential retrieval strategy may further reduce memory usage"。确实，dynamic head 现在 full history 都保留（只是 segment-wise 压缩），memory 还是大。一个 retrieval-based 方法（如 Reformer / RT 的 locality sensitive hashing）只在 query time 检索相关的历史 segments，可能进一步压缩。但 retrieve 引入额外 latency，要权衡。

9. **与 Linear Attention 的对比**: Sana-Video（https://arxiv.org/abs/2604.x）的 block linear attention 思路和 segment-wise similarity pruning 在 spirit 上有点像（都基于局部相似性减少 attention 计算量）。能不能 unify？

10. **AR video 的本质 insight**: transition anchor 这个概念让我联想到 diffusion model 中的 "DDIM inversion anchor" 或者 video super-resolution 的 reference frame。这说明 AR video diffusion 的 KV cache 不是随意的 cache，而是有"近邻 anchor + 长程对应"的双层结构。这种结构在 video diffusion 之外（如 video LLM、video understanding transformer）可能也存在。

---

## 9. 工程实践 tips

如果你想复现/落地：

1. **何时开始压缩**：从第二个 AR step 开始（第一个 chunk 没"历史"），代码里 $i \geq 2$ 才触发 compression。第一个 chunk 用 full cache。
2. **Sink frame 保留**：和 Dummy Forcing 一致保留 sink frame。LongLive、Reward Forcing 等模型训练时对 sink 有特殊处理，必须保留。
3. **Segment 相似度只在第一个 block 算**：用 DiT 第一个 block 的 key 作为 proxy，是性价比最高的工程 trick。$n=6$ segments 的 uniform contiguous 划分就够。
4. **α = 0.8 是好默认**：从 0.5 到 0.8 不太敏感，但太低会让太多 head 被判为 static，丢历史导致 dynamic degree 降。
5. **r = 0.3 的 default**：实验显示 0.3 是 sweet spot。可以基于 dynamic degree 监控自适应调。
6. **量化叠加**：和 FP8（SageAttention2）正交，可叠加再 +5-7% speedup（Table 6）。

---

## 10. 一句话总结

**Forcing-KV 把 AR video diffusion 的 attention head 分成"看近的 static" 和 "看远的 dynamic" 两类，static head 用 structural pruning 保留 sink + transition anchor + current chunk，dynamic head 用 segment-wise cosine similarity pruning 保留变化大的 segment，整体 training-free、与 quantization 正交、随分辨率/window size scaling 收益递增。**

这个工作的核心贡献是把"AR video diffusion 的 KV cache 利用有 inductive bias"这件事用三个 observation 讲清楚了，并基于此给出了对应的 hybrid compression。它不只是一个工程 trick，而是一个关于 AR video diffusion 模型功能分化的经验性研究 + 对应的 compression 设计。

非常值得 follow 的方向：训练阶段 cache 压缩、adaptive r、retrieval-based dynamic head compression、与 sparse attention 进一步整合、以及 head functional specialization 的理论分析。

希望这些细节能 build 你的 intuition！如果你想深入某一块（比如 segment-wise similarity 的设计 choice，或 transition anchor 的更深 ablation，或与 DuoAttention 的类比），可以再展开。
