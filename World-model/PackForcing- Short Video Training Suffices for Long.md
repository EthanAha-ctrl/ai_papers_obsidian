---
source_pdf: PackForcing- Short Video Training Suffices for Long.pdf
paper_sha256: 303cdcdb3ec43c0269f7f174e864ca1f1a5d5c9f7d9160355e5e4e257f80bd98
processed_at: '2026-08-06T01:48:26-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，咱们抛开那些学术黑话，用最直白的大白话来聊聊这篇 paper 到底在搞什么名堂。

### 一句话总结
这篇 paper 做的事就是：**教一个只会拍 5 秒短片的 AI 导演，怎么用不到 4GB 的内存，连贯地拍出 2 分钟的长电影，而且画面还不崩。**

### 做视频就像写故事：记忆管理的艺术
你想想，现在的 Video Diffusion Models 为什么做不长？因为它们是“全量注意力”，要把所有画面铺在桌上一块看。如果拍 2 分钟的视频，桌子（显存）根本放不下。

为了解决这个问题，大家搞出了 Autoregressive（一块一块拍）的方法。拍新的一块时，看一眼之前拍的旧块。但这有个大麻烦：旧块越攒越多，内存照样爆掉。如果直接把旧块扔了，故事就接不上了，主角的脸可能下一秒就变了。

PackForcing 的核心思路是：**不要一刀切地保留或丢弃历史，要给历史分层。** 它把 KV Cache 分成了三个区，就像人脑的记忆系统：

1.  **Sink Tokens（长期核心记忆）**：死死记住电影开头的前两块画面，绝对不删，也不压缩。
    *   **Intuition**: 就像拍动作片，主角长啥样、穿啥衣服、在哪个场景，第一秒就定死了。后面不管拍多久，都要回头看这两眼，保证主角不会逐渐变成怪兽。Ablation 证明，只要删了这几帧，Subject Consistency 立刻从 93 掉到 74，视频就“精神错乱”了。
2.  **Recent & Current Tokens（短期工作记忆）**：保留刚拍完的最后一两块画面，原汁原味，连像素细节都不放过。
    *   **Intuition**: 拍下一帧动作时，必须看清楚上一帧的手抬到哪了，不然动作就卡顿或者瞬移了。
3.  **Compressed Mid Tokens（中期模糊记忆）**：这是最神的一步。中间那一大堆历史画面，既不删，也不原样保留，而是暴力压缩成“缩略图”，并且每次拍新画面时，只在缩略图里挑最相关的几张看。

### 暴力压缩的艺术：Dual-Branch Network
中间的历史画面太多了，比如一个 2 分钟的视频，中间有几百个 blocks，每个 block 6240 个 tokens。PackForcing 要把它们压到每个 block 只有 182 个 tokens（32 倍压缩）。

怎么压？用一条路压会丢信息。比如你只做空间下采样，主角衣服上的花纹就糊了；你只做语义池化，主角可能就偏出画外了。所以它搞了个双分支：

*   **HR Branch（搞结构的）**：直接在 Latent 上跑 3D CNN。就像拿个尺子把画面网格化，记住“哪里有东西，轮廓是啥”。
*   **LR Branch（搞背景的）**：先把 Latent 解码回像素，拿 VAE 池化缩小，再编码回 Latent。就像把图画缩小成邮票大小再放大，细节没了，但“整体是个啥场景”还在。
*   最后两者一加：$\tilde{\mathbf{h}} = \mathbf{h}_{\mathrm{HR}} + \mathbf{h}_{\mathrm{LR}}$。这样既保住了骨架，又保住了灵魂。

公式里 $N_c = \lfloor B_f/2 \rfloor \times \lfloor h/4 \rfloor \times \lfloor w/4 \rfloor$。原本 $B_f=4, h=30, w=52$，现在变成 $2 \times 7 \times 13 = 182$。这一步直接把 138GB 的显存需求砸到了 4GB。

### 省钱的绝招：Dynamic Context Selection
虽然压成了缩略图，但如果历史太长，缩略图也会积少成多。PackForcing 又搞了个“按需调阅”机制。

在去噪的第一步，它算一下当前的 query 和所有历史 mid blocks 的亲和力（Attention Score），只挑 Top-K（比如 16 个）最相关的 mid blocks 放进 attention 里算。剩下的先存着，说不定下一秒就用到了。

这个亲和力打分公式长这样：
$$ s_m = \sum_{j=1}^{L_k} \sum_{i \in \mathcal{S}_q} \left( \frac{1}{B \cdot N_{opt}} \sum_{b=1}^B \sum_{h=1}^{N_{opt}} \frac{Q_{b,h,i} K_{m,b,h,j}^\top}{\sqrt{d_h}} \right) $$
*   $s_m$ 是第 $m$ 个 block 的分数。
*   $Q$ 是当前画的 query，$K$ 是历史的 key。
*   $\mathcal{S}_q$ 是采样了一部分 query，不用全算，省时间。
*   $N_{opt}$ 是只用一半的 attention head 算，因为各个 head 的打分其实差不多。

这步操作开销极小（小于 1%），但避免了无脑丢数据（FIFO）。实验证明，它比无脑 FIFO 的 Subject Consistency 高了 0.8 个点。

### 位置的魔法：Incremental RoPE Adjustment
这个 trick 非常妙。假设中间的压缩区满了，必须要扔掉最老的一个 mid block。这时候，前面的 Sink tokens 和剩下的 mid tokens 之间就空出来了一个位置缺口。

模型用的是 3D RoPE，位置信息是乘在 key 上的：
$$ \mathbf{k}_{\mathrm{cached}}^{(p)} = \mathbf{k}_{\mathrm{raw}} \odot e^{i\pmb{\theta}_p} $$
如果 Sink 的位置还是 0，但最近的 mid 位置变成了 8+4=12，中间 4-11 的位置没了，Transformer 就会懵逼。

PackForcing 的做法是：因为 RoPE 旋转是可以乘法叠加的，干脆给 Sink tokens 再乘一个时间偏移量的旋转矩阵，把它的位置整体平移过去：
$$ \mathbf{k}_{\mathrm{sink}}' = \mathbf{k}_{\mathrm{sink}} \odot e^{i\pmb{\theta}_t(\delta)} \mathbf{1}_h \mathbf{1}_w $$
*   $e^{i\pmb{\theta}_t(\delta)}$ 是只在时间轴上旋转 $\delta$ 的距离。
*   $\mathbf{1}_h, \mathbf{1}_w$ 是空间轴上啥也不做（乘 1）。

这样 Sink 就无缝滑到了 mid 的前面，位置完全连续。这个操作开销极小，但把长程语义漂移（CLIP score 下降幅度）降低了 62%。

### Build My Intuition
你可以这么理解：PackForcing 就像一个带了“记忆眼镜”的画家。
*   眼镜中心是清晰的：Recent Tokens。
*   眼镜边缘有个固定的锚点：Sink Tokens。
*   其余全是极度压缩、但能随时按需调取的模糊胶卷：Compressed Mid Tokens。
*   当胶卷盒满了，它不会扔掉最老的开头（因为那是定海神针），而是把开头的时间戳偷偷改一下，让它连在剩下的胶卷前面。

这就是为什么只用 5 秒的短视频训练，它也能生成长达 120 秒的视频——因为不管视频多长，它每次画画时“看到”的记忆结构大小和形状永远是一样的（约 27,872 个 tokens），不存在 Out-of-Distribution 的问题。

### References
*   [PackForcing GitHub Page](https://github.com/ShandaAI/PackForcing)
*   [Original Paper PDF](https://arxiv.org/abs/2503.20314)
*   [Self-Forcing Paper](https://arxiv.org/abs/2506.08009)
*   [StreamingLLM Paper](https://arxiv.org/abs/2309.17453)

---

Andrej，这篇 paper 的核心思想非常优雅，它本质上是将 Large Language Model (LLM) 领域中处理 Long Context 的 KV Cache 管理技术（如 StreamingLLM, H2O, SnapKV）与 Video Diffusion Transformer (DiT) 的 Spatiotemporal 冗余特性深度结合，解决了 Autoregressive Video Generation 中的 Memory Explosion 与 Error Accumulation 困境。其最 Radical 的声明在于：仅用 5-second clips 训练，即可实现 24x 的 Temporal Extrapolation（生成 120s 视频），且 KV Cache 严格 Bound 在 4GB 左右。

下面为你进行深度的技术拆解，旨在建立你对此架构的 Intuition。

### 1. 核心动机：Autoregressive Video Diffusion 的困境

现有的 Video Diffusion Models（如 Sora, Wan, CogVideoX）通常基于 Bidirectional DiT，需要将所有 frames 放在一个 Spatiotemporal volume 中进行全量 Attention，这导致计算复杂度呈 $O((THW)^2)$ 增长。Autoregressive 范式（如 CausVid, Self-Forcing）通过 Block-by-Block 生成并缓存历史 KV Cache 解决了这一问题，但引入了两个致命问题：

1.  **Error Accumulation**: 模型在自身生成的 Noisy History 上 Rollout，预测误差会 Compound。Self-Forcing 在 60s 后 CLIP score 会从 33.89 暴跌至 27.12。
2.  **Unbounded Memory Growth**: 对于 2-minute, 832x480, 16FPS 的视频，Token 数量达到惊人的 749K。若 30 层 Transformer 每层都存全量 KV，需要约 138GB 显存，单张 H200 根本无法支撑。

PackForcing 的破局之法是：**History 不应被简单 Truncate 或保留，而应被 Hierarchical Compress 与 Route。**

### 2. 架构解析：Three-Partition KV Cache

PackForcing 将 Monotonically growing 的 KV Cache 解耦为三个功能截然不同的 Partition，总 Context 大小严格 Bounded 在约 27,872 tokens。

#### A. Sink Tokens (Global Semantics Anchor)
受 LLM 中 Attention Sink 现象启发，模型对最初几帧的 Attention 极高，这些帧锁定了 Scene Layout 和 Subject Identity。
*   **定义**: 保留最初 $N_{sink}$ 个 frames 的 Full-resolution KV Cache，永不 Evict 或 Compress。公式中 $N_{sink}=8$（即 2 个 blocks）。
*   **公式**:
    $$ \mathcal{C}_{\mathrm{sink}}^l = \{(\mathbf{K}_j^l, \mathbf{V}_j^l)\}_{j=1}^{N_{\mathrm{sink}}/B_f} $$
    这里 $l$ 是 Layer index，$j$ 是 Block index，$B_f$ 是每个 Block 的 frame 数（=4）。$\mathbf{K}, \mathbf{V}$ 保持原始精度。消耗不到总 Token Budget 的 2%。
*   **Intuition**: 类似于 LLM 中的 System Prompt，提供了一个稳定的全局参考点，防止 Long-horizon 生成中的 Semantic Drift。

#### B. Compressed Mid Tokens (Massive Spatiotemporal Compression)
这是论文最核心的创新。介于 Sink 和 Recent 之间的庞大 History 被极度压缩。
*   **定义**: 使用 Dual-Branch Network 将每个 Block 从 6,240 tokens 压缩到 $N_c = 182$ tokens（32x 压缩率）。同时，不盲目 Attend 所有 Mid tokens，而是通过 Dynamic Context Selection 挑选 Top-K 最相关的 Blocks 形成 Active Set $\mathcal{S}_{mid}$（限制 $N_{mid}$）。
*   **公式**:
    $$ \mathcal{C}_{\mathrm{mid}}^l = \{(\tilde{\mathbf{K}}_j^l, \tilde{\mathbf{V}}_j^l)\}_{j \in \mathcal{S}_{\mathrm{mid}}} $$
    其中 $\tilde{\mathbf{K}}, \tilde{\mathbf{V}}$ 表示压缩后的 KV。$N_c = \lfloor B_f/2 \rfloor \times \lfloor h/4 \rfloor \times \lfloor w/4 \rfloor$。
*   **Intuition**: Video 具有极高的 Temporal Redundancy。Mid History 不需要 Full-resolution 的 Texture，只需要保留 Structural 和 Semantic Skeleton 供 Query 检索。

#### C. Recent & Current Tokens (Local Coherence)
*   **定义**: 最近生成的 $N_{recent}$ 个 frames 保持 Full Resolution，确保局部时间平滑性。
*   **公式**:
    $$ \mathcal{C}_{\mathrm{rc}}^l = \{(\mathbf{K}_j^l, \mathbf{V}_j^l)\}_{j=i-N_{\mathrm{recent}}/B_f}^i $$
*   **Intuition**: 类似 Sliding Window 机制，保证帧间连续性不因为压缩而出现 Artifacts。

### 3. 技术核心 1：Dual-Branch HR Compression

为了实现 32x 的 Token Reduction 且不丢失关键信息，PackForcing 设计了双分支压缩模块。这有点类似 U-Net 的 Skip Connection 思想，但作用在 Token Compression 层面。

给定的 Latent $\mathbf{z} \in \mathbb{R}^{B \times C \times T \times H \times W}$，双分支并行处理：

1.  **HR Branch (High-Resolution)**:
    直接在 Latent Space 操作。使用 Progressive 3D CNN（Stride: $2\times$ Temporal, $8\times$ Spatial）加上 SiLU 激活，最后 $1\times1$ 投影到 Hidden dim $d=1536$。
    *Intuition*: 保留精细的 Local Texture 和 Spatial 结构。

2.  **LR Branch (Low-Resolution)**:
    走 Pixel Space 通路。Latent $\mathbf{z}$ -> VAE Decode -> 3D Average Pooling ($2\times$ Temp, $4\times$ Spatial) -> VAE Encode -> Patch Embedding。
    *Intuition*: 保留 Global Perceptual Layout。直接在 Latent Pooling 会破坏 VAE 的分布，所以绕道 Pixel Space。

**Fusion**:
$$ \tilde{\mathbf{h}} = \mathbf{h}_{\mathrm{HR}} + \mathbf{h}_{\mathrm{LR}} \in \mathbb{R}^{B \times N_c \times d} $$
两者维度对齐后 Element-wise 相加。这种设计在极端压缩下保住了 Attention Pattern 的连贯性。

### 4. 技术核心 2：Dynamic Context Selection

如何决定哪些 Mid Blocks 进入 Active Set？受 H2O (Heavy-Hitter Oracle) 启发，PackForcing 计算 Query-Key Affinity，但做了极致的工程优化以避免 Overhead。

*   **Affinity Score 公式**:
    $$ s_m = \sum_{j=1}^{L_k} \sum_{i \in \mathcal{S}_q} \left( \frac{1}{B \cdot N_{opt}} \sum_{b=1}^B \sum_{h=1}^{N_{opt}} \frac{Q_{b,h,i} K_{m,b,h,j}^\top}{\sqrt{d_h}} \right) $$
    其中 $s_m$ 是候选 Block $m$ 的重要性分数。$Q$ 来自 Recent/Current blocks，$K_m$ 来自候选 Mid blocks。$\mathcal{S}_q$ 是 Subsampled queries，$N_{opt} = N_h/2$ 是只用一半 Attention Heads。

*   **工程优化**:
    1.  **Deterministic Query Subsampling**: 均匀采样 Query tokens，减少计算量。
    2.  **Half-Head Evaluation**: Importance 在 Heads 间高度相关，减半计算。
    3.  **Step-wise Caching**: 只在每个 Block 生成的第一个 Denoising Step 计算 Affinity，后续 Step 复用 Indices。开销 <1%。

*   **Intuition**: Attention 的重要信息是 Sparse 且 Dynamic 的（如图3所示，Jaccard Distance 高达 0.75）。Soft-Routing 机制允许模型在 Scene 变化时动态拉回之前被 Archive 的 Mid Tokens，这比 FIFO 优出 +0.12 CLIP score。

### 5. 技术核心 3：Incremental RoPE Adjustment

这是非常 Brilliant 的数学 trick。当 Mid Buffer 满了，需要 Evict 最老的 Mid Block 时，Sink Tokens 和剩下的 Mid Tokens 之间会出现 Position Gap。

Backbone 使用 3D RoPE，频率分为 $\pmb{\theta} = [\pmb{\theta}_t, \pmb{\theta}_h, \pmb{\theta}_w]$。
缓存时 Key 已经被绝对位置 $p$ 旋转过：
$$ \mathbf{k}_{\mathrm{cached}}^{(p)} = \mathbf{k}_{\mathrm{raw}} \odot e^{i\pmb{\theta}_p} $$

Evict $\Delta$ blocks（$\delta = \Delta B_f$ frames）后，Sink 的位置还是 $0, 1, ...$，但最早的 Mid Key 位置变成了 $N_{sink}+\delta$。位置断裂导致 Attention 崩溃。

由于 RoPE 满足乘法结合律 $e^{i\pmb{\theta}_p} \cdot e^{i\pmb{\theta}_\delta} = e^{i\pmb{\theta}_{p+\delta}}$，且 Eviction 只影响 Temporal 轴，PackForcing 直接对 Sink Keys 施加一个纯 Temporal 旋转：
$$ \mathbf{k}_{\mathrm{sink}}' = \mathbf{k}_{\mathrm{sink}} \odot e^{i\pmb{\theta}_t(\delta)} \mathbf{1}_h \mathbf{1}_w $$
这里 $\mathbf{1}_h, \mathbf{1}_w$ 是 Identity rotation（幅值为1），保持空间坐标不变。
*Intuition*: 相当于把 Sink Tokens 在时间轴上“平移”到紧挨着当前 Mid Buffer 的前面，填补了 Gap。开销极小（<0.1% FLOPs），但将长程语义漂移降低了 62%。

### 6. 实验数据与直觉验证

*   **24x Temporal Extrapolation**: 模型只在 20 latent frames (5s) 上训练，却能生成 120s。这是因为通过 Compression，训练和推理时 Transformer 看到的 Context Size 是恒定的（~27,872 tokens），没有 Out-of-Distribution 的 Length Shift。这是典型的 "Train Short, Test Long" 范式，类似于 LLM 中的 YaRN 或 NTK-aware Interpolation。
*   **Motion Richness vs Consistency Trade-off**: 表1显示，PackForcing 的 Dynamic Degree 达到 56.25（SOTA），远超 LongLive (44.53)。LongLive 依靠 Freezing Motion 来维持 Consistency，而 PackForcing 靠 Compressed History 提供可靠的 Spatiotemporal Grounding，敢于生成大动态。
*   **Ablation on Sink Size (Table 3)**: 如果 $N_{sink}=0$，Subject Consistency 暴跌至 74.72；如果 $N_{sink}=16$，Consistency 最高但 Dynamic Degree 暴跌至 35.16（模型过度依赖静态早期帧，不敢动）。$N_{sink}=8$ 是 Sweet spot。
*   **Attention Pattern Analysis (Fig 3)**: 重要性分数曲线后期极其平坦，意味着 FIFO Eviction 会丢失同等重要的早期信息。High Jaccard Distance (0.75) 证明了 Top-K Routing 相比 Static Window 的巨大优势。

### 7. 联想与延伸思考

1.  **与 LLM KV Cache 技术的同源性**: PackForcing 本质上是 StreamingLLM (Sink) + H2O (Top-K Eviction) + GQA/MQA (Compression) 在 Video 领域的集大成者。区别在于，Video 的 KV 不能直接 Drop，因为 Spatiotemporal 信息密度远大于 Text Token，必须 Learn 一个 Compressor。
2.  **World Models 的 Context 瓶颈**: 像 Genie 或 Sora 这样的 World Models，长期受限于 Context Window 无法维持长程物理一致性。PackForcing 的 3-Partition 设计直接提供了一种工程上可行的 Long-horizon World Simulation 架构。如果将 Sink Tokens 视为 "Immutable World Rules"，Mid Tokens 视为 "Long-term Memory"，Recent Tokens 视为 "Working Memory"，这与人脑的记忆分层机制惊人一致。
3.  **Visual Token 压缩 vs Token Pruning**: 目前 LLM 领域流行 Token Merging (如 ToMe)，Video 领域也有 OAttn 等。PackForcing 走了更极端的 VAE Re-encoding 路线。这种 Dual-Branch (Latent CNN + Pixel VAE) 的融合方式，可能启发未来多模态大模型（如 Video-LLaMA）中 Visual Token 的压缩方式。
4.  **流式生成**: Algorithm 1 支持 Streaming VAE Decode，每个 Block 增量解码，这极大地降低了 Time-to-First-Frame。对于实时交互式 Video Generation（类似 ChatGPT 的逐字输出）具有商业级落地意义。

**References for Further Reading:**
*   [StreamingLLM (Attention Sinks)](https://arxiv.org/abs/2309.17453)
*   [H2O (Heavy-Hitter Oracle)](https://arxiv.org/abs/2306.14048)
*   [Self-Forcing (Train-Test Gap in AR Video)](https://arxiv.org/abs/2506.08009)
*   [DeepForcing (Deep Sink & Participative Compression)](https://arxiv.org/abs/2512.05081)
*   [Wan2.1 (Base Video Model)](https://arxiv.org/abs/2503.20314)
*   [PackForcing Project Page](https://github.com/ShandaAI/PackForcing)
