---
source_pdf: FastVLM.pdf
paper_sha256: e262ec23249d76175a0292657d359ce7cf3dd4d7de2565dc41ac40ca39bb0e05
processed_at: '2026-08-04T07:54:06-07:00'
target_folder: Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FastVLM 用人话讲讲

Andrej, 好嘞, 我换一种方式跟你聊这篇 paper, 就像我们在咖啡厅白板上画图那种感觉。

---

## 这篇 paper 到底在干啥

一句话总结: **Apple 团队做了一个新的 vision encoder, 让 VLM 在手机/笔记本上跑得飞快, 同时性能不掉**。

你想想现在的 VLM, 比如 LLaVA-OneVision, 想处理一张 1152×1152 的图, 光 vision encoder 就要跑 2.7 秒, 加上 LLM prefilling 一共 **14 秒** 才吐出第一个 token。用户等 14 秒? 这没法用。

FastVLM 同样的图, 同样的 0.5B LLM, **166 毫秒**。快了 85 倍。性能还差不多, 有些 benchmark 还更好。

这就是这篇 paper 的全部故事。

---

## 为什么现在 VLM 慢

我们拆一下 TTFT (time-to-first-token) 这个东西:

$$\text{TTFT} = T_{\text{vision}} + T_{\text{prefill}}$$

两部分, vision encoder 算图的时间, 加上 LLM 处理 visual tokens 的时间。

问题出在哪? 出在 ViT 这个架构本身。

ViT 的做法是: 把图切成 patch, 每个 patch 14×14 像素, 一个 patch 变一个 token。图越大, token 越多, **平方关系**。

336×336 的图: $(336/14)^2 = 576$ tokens
1024×1024 的图: $(1024/14)^2 \approx 5300$ tokens
1152×1152 的图: ~6700 tokens

而 self-attention 是 $\mathcal{O}(N^2)$ 的, token 涨 10 倍, attention 计算量涨 100 倍。所以高分辨率下 ViT 就废了, 又慢 token 又多, LLM prefilling 也跟着爆炸。

**这就是 paper 要解决的核心痛点**。

---

## FastVLM 的核心 trick: 多下采样一次

ViT 是 isotropic 的, 从头到尾 token 数不变。

传统 hybrid 架构 (比如 FastViT, ConvNeXT) 是 4 个 stage, 每个 stage 之间下采样 2 倍, 最后下采样 16 倍。1024 输入 → 64×64 = 4096 tokens, 还是太多。

FastViTHD 的想法特别简单粗暴: **我多加一个 stage, 再下采样一次**。

5 个 stage, 最后下采样 64 倍。1024 输入 → 16×16 = **256 tokens**。

token 数从 4096 降到 256, **减少 16 倍**。attention 计算量减少 256 倍。

就这么一个 trick。

你可能会问: 多下采样不会丢信息吗? 奇妙的地方在于, **并没有丢多少**。因为 self-attention 在最后两个 stage 才用, 前面 3 个 stage 是 conv, conv 本身就是 local feature extractor, 下采样对 conv 影响不大。真正需要全局信息的 self-attention, 放在 32× 和 64× 下采样之后做, token 少, 算得快, 信号够用。

这就是 Figure 2 那张架构图的全部意思。

---

## 架构长啥样

我用文字画一下:

```
Input Image (1024×1024)
    ↓
[Stem: 4× downsample]  →  256×256, 96 channels
    ↓
[Stage 1: 2× RepMixer]  →  128×128, 96 channels
    ↓ [downsample 2×]
[Stage 2: 12× RepMixer] →  64×64, 192 channels
    ↓ [downsample 2×]
[Stage 3: 24× RepMixer] →  32×32, 384 channels
    ↓ [downsample 2×]
[Stage 4: 4× MHSA]      →  16×16, 768 channels   ← self-attention 开始
    ↓ [downsample 2×]
[Stage 5: 2× MHSA]      →  8×8 = 64 tokens, 1536 channels
    ↓
[Projector] → LLM
```

注意 stage 4 和 5 才是 self-attention, 在 16×16 和 8×8 的 feature map 上做, token 数 256 和 64, 算起来飞快。

总参数 125M, 比 ViT-L/14 的 304M 小 2.4 倍。

---

## RepMixer 是啥

RepMixer 来自 FastViT 那篇 paper, 核心思想是 **training 时多分支, inference 时融合成单分支**。

训练时:
$$y = \text{Branch}_{\text{main}}(x) + \text{Branch}_{\text{skip}}(x)$$

推理时通过数学等价变换, 把两个分支的 conv kernel 加起来, 变成一个 conv:
$$y = \text{Conv}_{\text{fused}}(x)$$

**零延迟开销**, training 时有多分支的表达能力, inference 时是单分支的速度。

这个设计在 Apple Neural Engine 上特别友好, 因为 NE 喜欢 simple conv, 不喜欢复杂的多分支图。

---

## 为什么 naive scaling 不行

Apple 团队还试了一种 naive 做法: 不加 stage, 就把原来 FastViT 的 stage 3, 4 的 self-attention 层数加多, embedding dim 加大。

结果 Figure 3 显示, **naive scaling 比 ConvNeXT-L 还慢**。

为什么? 因为它在 16× 下采样的 tensor 上做 self-attention, token 数 $(R/16)^2$。R=1024 时就是 4096 tokens, attention 计算量爆炸。

FastViTHD 的 insight 是: **self-attention 应该推到下采样更多的 stage 做**, conv 负责前面的下采样和 low-level feature。

这个 insight 其实挺朴素: attention 是 global operator, 不需要 spatial resolution, 应该在 token 少的地方做; conv 是 local operator, 适合在 high resolution 上提取 low-level feature。

---

## Pareto 前沿这个分析很关键

Figure 4 是 paper 最有价值的部分。

他们扫了 3 个 LLM size (0.5B, 1.5B, 7B) × 多个 resolution (256, 512, 768, 1024), 对每个点训一个 VLM, 画在 accuracy vs TTFT 的图上。

发现两件事:

**第一, 小 LLM 配高分辨率是 suboptimal**。你给 0.5B LLM 喂 1024 分辨率的 256 个 token, LLM 吃不消, 性能没涨多少, 但 vision encoder 慢了一大截。

**第二, FastViTHD 的 Pareto 前沿整体比 FastViT 好一大截**。给定同样的时间预算, FastViTHD 能多拿 2.5 分 (Avg-5)。给定同样的性能目标, 快 3 倍。

这个分析方式其实挺值得学习的, 不要只比一个点, 要扫整个 (resolution, LLM size) 的网格, 找 Pareto 前沿。

---

## 跟 token pruning 方法比

现在有一堆 paper 做 token pruning, 比如把 ViT 的 576 tokens 砍到 64 个。Table 5 直接对比:

- ViT-L/14 + pruning → 64 tokens, Avg-5 大约 57-60
- FastViTHD @ 512 → 64 tokens, Avg-5 = 60.4

**hierarchical 原生减少 token, 比 ViT 后端 pruning 更好**。

道理很简单: ViT pruning 是先算完 576 tokens 的 attention, 再扔掉一些, 计算量没省; FastViTHD 是架构层面就只有 64 tokens, encoder 内部计算也省了。

---

## 跟 SOTA 比

Table 6 是主战场, 我挑几个关键的:

**vs LLaVA-OneVision (0.5B)**:
- LLaVA-OV: 7290 tokens, TTFT 14.1 秒
- FastVLM: 256 tokens, TTFT 0.166 秒
- 性能差不多, 快 85 倍

**vs MM1 (7B)**:
- MM1: ViT-H 632M encoder, 720 tokens
- FastVLM: FastViTHD 125M encoder, 256 tokens
- DocVQA: FastVLM 82.7 vs MM1 76.8, 高 6 分

**vs Cambrian-1 (8B, 4 个 encoder)**:
- Cambrian-1: 4 个 encoder 加起来 1.88B 参数, TTFT 5 秒
- FastVLM: 1 个 encoder 125M, TTFT 0.64 秒
- 性能相当, 快 8 倍

**FastVLM 用一个又小又快的 encoder, 打赢了用一堆 encoder 堆出来的模型**。这个结果挺震撼的。

---

## 他们怎么 benchmark 的

这点 paper 做得很严谨。

没用 NVIDIA GPU 估, 用的是 **MacBook Pro M1 Max**:
- Vision encoder → Core ML → Neural Engine
- LLM → MLX framework → GPU
- 真实测延迟, 不是 FLOPs 估算

这才是端侧部署的真实场景。Apple Silicon 上 Neural Engine 和 GPU 是分开的, vision encoder 跑 NE, LLM 跑 GPU, unified memory 共享 weights。这种 benchmark 才有意义。

---

## 训练细节

他们用 3 种规模的数据训:

| Scale | Pretraining | Instruction Tuning | 备注 |
|-------|-------------|---------------------|------|
| Small | 558K (LLaVA-1.5) | 665K (LLaVA-1.5) | baseline |
| Medium | 558K + 15M (CC3M/12M recap) | 1.1M / 6.5M / 11.9M / 12.5M | scale up |
| Large | 同上 | + 10.6M (MammothVL, Stage 3) | chain-of-thought |

Stage 1.5 是关键, 用 15M 的 densely captioned 数据让 encoder 适应高分辨率。因为 FastViTHD 是 224 分辨率 CLIP pretrain 的, 直接上 1024 会崩, 需要一个中间适应阶段。

8× H100, Stage 1.5 训 77 小时, Stage 2 训 8 小时。

---

## 失败案例分析

Section E 这部分很诚实, 我喜欢。

他们发现 3 种失败模式:

1. **Text 太小看不清** → 提高 resolution 就行 (Table 15)
2. **需要 background knowledge** → 换大 LLM (Table 14, 比如心脏 SA node 那个例子)
3. **既需要看清楚又需要推理** → 大 LLM + 高 resolution (Table 16)

这种 decomposition 挺有用的, 能指导实践: 不是所有问题都靠 scale resolution 解决, 有时候是 LLM 能力不够。

---

## 对 VLM 设计的 intuition 更新

读完这篇, 我的 mental model 有几个更新:

**1. Vision encoder 架构不是 secondary 的**

以前大家觉得 VLM 就是 LLM + 一个 CLIP-pretrained ViT, encoder 选啥不重要。FastVLM 证明 encoder 架构对 latency-accuracy trade-off 有 first-order 影响。

**2. Token count 不等于 information**

256 个 well-designed tokens 可以 encode 1024² image 的 sufficient information。7290 个 ViT tokens 不一定 information-richer。visual token 的 information density 比 token count 更重要。

**3. Hierarchical 在 high resolution 上天然占优**

ViT 的 isotropic design 在 336 分辨率上挺好, 到 1024 就废了。hierarchical 通过 multi-stage downsample 自然缓解 token 爆炸, 是 high resolution VLM 的正确 path。

**4. 端侧部署改变设计哲学**

如果在 NVIDIA GPU 上跑, 你可能不在乎 encoder 是多分支还是单分支。但在 Apple Neural Engine 上, reparameterization 的单分支 conv 才是快的。hardware-aware design 很重要。

**5. Multi-encoder ensemble 可能不是正道**

Cambrian-1 用 4 个 encoder, 被 FastVLM 一个 encoder 打平。这暗示 efficient single encoder 可能是更好 path, ensemble 的 marginal 收益在递减。

---

## 局限性

paper 也有一些没回答的问题:

1. **Pretraining data**: 只用了 DataCompDR-1B, 没试更大的 dataset, 不知道 scaling 趋势
2. **Video**: 只做了 single image, video VLM 的 temporal dimension 怎么处理没讲
3. **Encoder-decoder 最优 ratio**: 给定 fixed FLOPs budget, encoder 和 decoder 怎么分配没量化
4. **3D / multi-view**: robotics 场景的 multi-view input 没涉及

这些 open questions 挺值得 follow-up 的。

---

## 后续方向

如果要我 follow up, 我会想做:

1. **FastViTHD + Mamba**: 把 stage 4, 5 的 MHSA 换成 Mamba, $\mathcal{O}(N)$ complexity, 可能进一步提速
2. **Learned downsample**: 64× 是 hand-crafted, 能不能 learnable
3. **Video extension**: temporal redundancy 怎么利用
4. **Encoder-decoder co-design**: end-to-end jointly optimize

---

## 总结

FastVLM 就一个核心 insight: **在 hybrid 架构里多加一个 downsample stage, 把 self-attention 推到 token 少的地方做**。简单, 但有效。

结果就是 85× 更快, 性能相当, 在 Apple 端侧真实 benchmark。这对 VLM 在手机/笔记本上落地意义重大。

paper 写得很 solid, Pareto 分析 + 真实硬件 benchmark 是 VLM efficiency 研究的好榜样。

Reference:
- [FastVLM GitHub](https://github.com/apple/ml-fastvlm)
- [FastViT (ICCV 2023)](https://arxiv.org/abs/2303.14189)
- [MobileCLIP (CVPR 2024)](https://arxiv.org/abs/2311.17049)
- [MLX framework](https://github.com/ml-explore/mlx)
- [LLaVA-1.5](https://arxiv.org/abs/2310.03744)
- [ConvLLaVA](https://arxiv.org/abs/2405.04319)
- [Cambrian-1](https://arxiv.org/abs/2406.16860)
- [ViTamin](https://arxiv.org/abs/2404.02100)
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- [Mamba](https://arxiv.org/abs/2312.00752)

---

# FastVLM: Efficient Vision Encoding for Vision Language Models 深度解析

Andrej, 这是一篇非常 solid 的 Apple 工作, 由 Pavan Kumar Anasosalu Vasu 和 Hadi Pouransari 带队完成。核心 insight 是**通过 hybrid convolution-transformer 架构在 VLM 的 resolution-latency-accuracy 三元 trade-off 上取得 Pareto 前沿突破**。让我为你系统拆解。

---

## 1. 核心问题: TTFT 的 bottleneck 结构

VLM 的 TTFT (Time-To-First-Token) 可以形式化分解为:

$$\text{TTFT} = \underbrace{T_{\text{vision}}(R, \theta_{\text{enc}})}_{\text{vision encoder latency}} + \underbrace{T_{\text{prefill}}(N_{\text{tokens}}, \theta_{\text{LLM}})}_{\text{LLM prefilling}}$$

其中:
- $R$ 表示 input image resolution (e.g., 1024×1024)
- $\theta_{\text{enc}}$ 表示 vision encoder 的参数
- $N_{\text{tokens}} = f(R, \text{patch\_size}, \text{downsample\_factor})$ 表示 visual token 数量
- $\theta_{\text{LLM}}$ 表示 LLM decoder 参数

关键 insight 在于 Figure 5 揭示的 **latency regime crossover**: 当 $R$ 增大到某阈值后, vision encoder latency 开始 dominate 整个 TTFT。这个 crossover 点决定了架构优化的优先级。

以 FastViTHD + Qwen2-1.5B 为例, 当 resolution 从 256 → 2048 增长时, vision encoder latency 从 ~10ms 指数增长到 ~600ms, 而 LLM prefilling 从 ~50ms 增长到 ~340ms。**这是论文设计的根本动机**: 想要 push high resolution, 必须先优化 vision encoder 的 latency 曲线斜率。

Reference: [FastVLM paper](https://github.com/apple/ml-fastvlm)

---

## 2. FastViTHD 架构解析

### 2.1 从 FastViT 到 FastViTHD 的演进逻辑

FastViTHD 的设计核心在于**多增加一个 downsampling stage**, 使得 self-attention 在 downsample factor 32 (而非传统 16) 的 tensor 上计算。

**为什么这个设计有效?** 让我从 computational complexity 角度推演。

ViT 的 self-attention 计算复杂度:
$$\mathcal{O}(N^2 \cdot d)$$

其中 $N$ 是 token 数, $d$ 是 embedding dimension。对于 ViT-L/14 @ 336:
$$N = (336/14)^2 = 576 \text{ tokens}$$

对于传统 4-stage hybrid 架构 (downsample 16×) @ 1024 input:
$$N = (1024/16)^2 = 4096 \text{ tokens}$$

而 FastViTHD 的 5-stage (downsample 64×) @ 1024:
$$N = (1024/64)^2 = 256 \text{ tokens}$$

token 数量减少 16×, attention 计算量减少 256×! 这是一个**几何级数的 reduction**。

### 2.2 Stage-wise 架构详细配置

FastViTHD 由 5 个 stages 组成 (Figure 2):

| Stage | Blocks | Embedding Dim | Block Type | Downsample Factor |
|-------|--------|---------------|------------|-------------------|
| 1 | 2 | 96 | RepMixer | 4× |
| 2 | 12 | 192 | RepMixer | 8× |
| 3 | 24 | 384 | RepMixer | 16× |
| 4 | 4 | 768 | MHSA | 32× |
| 5 | 2 | 1536 | MHSA | 64× |

总参数量: **125.1M**, 比 FastViT (MobileCLIP) 大 3.5×, 但仍然小于 ViT-L/14 (304M) 和 ViTamin-L (333M)。

### 2.3 RepMixer Block 的 structural reparameterization

RepMixer (来自 FastViT [Vasu et al., ICCV 2023]) 的核心是 train-time multi-branch / inference-time single-branch reparameterization:

**Training 时:**
$$y = \text{CMix}_{\text{train}}(x) + \text{CMix}_{\text{skip}}(x)$$

**Inference 时:**
$$y = \text{CMix}_{\text{fused}}(x)$$

其中 CMix = Conv1×1 → DWConv3×3 → Conv1×1, fusion 通过将 skip branch 的 conv kernel 加到主 branch 实现, **零推理延迟增加**。

这个设计在 mobile/edge 部署场景下尤其重要, 也是 FastVLM 能在 M1 Macbook Pro 上 benchmark 出 3.2× speedup 的关键。

Reference: [FastViT paper (ICCV 2023)](https://arxiv.org/abs/2303.14189)

### 2.4 Patch Embedding 的细节

Patch embedding layers 包含:
- 7×7 depthwise convolution (stride=2, train-time overparameterization following MobileOne style)
- 1×1 pointwise convolution
- **放弃了 squeeze-excite**: 论文发现 SE layer 在高分辨率时对 inference latency 有负面影响

Stem 部分下采样 4×, 每个 patch embedding layer 下采样 2×, 最终达到 64× overall downsample。

### 2.5 Multi-Scale Feature Aggregation

$$F_{\text{multi}} = \text{Concat}[\text{DWConv}(F_3), \text{DWConv}(F_4), F_5]$$

其中 $F_i$ 表示 stage $i$ 的输出 feature map, DWConv 是 2D depthwise convolution 用于 spatial pooling。

Table 2 的 ablation 显示, multi-scale + DWConv 相比单 scale 提升 +0.3 Avg-5 (62.6 → 62.9), 提升虽然小但 consistent, 尤其在 DocVQA (+0.3) 这种 text-rich 任务上更有意义。

---

## 3. Naive Scaling 为什么不行 (Section B.1)

这部分是论文的**关键 ablation, 也是 FastViTHD 设计 motivation 的核心**。

Naive scaling 的做法: 把 FastViT 的 stage 3, 4 的 self-attention 层数增加, embedding dim 设为 [128, 256, 512, 1024], 层数 [2, 12, 16, 6]。

Figure 3 的结果显示: **naive scaling 在任何 resolution 下都比 ConvNeXT-L 慢**, 尤其是高分辨率时差距拉大。

为什么? 因为 naive scaling 在 downsample 16× 的 tensor 上做 self-attention, token 数量 $N = (R/16)^2$。当 $R = 1024$, $N = 4096$, attention 计算量 $\mathcal{O}(4096^2 \cdot 1024) \approx 1.7 \times 10^{10}$ FLOPs per layer。

FastViTHD 通过额外的 downsample stage 把 attention 推到 downsample 32× 和 64× 上:
- Stage 4 (32×): $N = 1024$, attention FLOPs $\approx 6.7 \times 10^8$
- Stage 5 (64×): $N = 256$, attention FLOPs $\approx 6.7 \times 10^7$

**这是 architectural insight**: self-attention 层应该尽量集中在高 downsample factor 的 stage, convolution 层负责 low-level feature extraction + spatial downsampling。

---

## 4. Pareto-Optimal Analysis: Vision Encoder × LLM Interplay

Section 3.2.1 是论文最有 insight 的部分, Figure 4 给出了 Pareto curve。

### 4.1 优化空间的形式化

VLM 性能 $P$ 是三维函数:
$$P = g(R, N_{\text{tokens}}, \theta_{\text{LLM}})$$

Latency $L$ 也是三维函数:
$$L = h(R, N_{\text{tokens}}, \theta_{\text{LLM}}) = T_{\text{vision}}(R) + T_{\text{prefill}}(N_{\text{tokens}}, \theta_{\text{LLM}})$$

Pareto-optimal 前沿:
$$\mathcal{P} = \{(R^*, N_{\text{tokens}}^*, \theta_{\text{LLM}}^*) : \nexists (R, N, \theta) \text{ s.t. } L \leq L^* \text{ and } P > P^*\}$$

### 4.2 关键发现

论文对每个 vision encoder (FastViT, FastViTHD) 在 3 个 LLM sizes (Qwen2-0.5B/1.5B/7B) × 多个 resolutions 上做了 exhaustive sweep, 发现:

1. **小 LLM + 高 resolution 是 suboptimal**: 小 LLM 无法有效利用大量 visual tokens, 但 vision encoder latency 仍然爆炸增长
2. **FastViTHD 的 Pareto 前沿显著优于 FastViT**: 给定 runtime budget, Avg-5 提升 2.5+ points; 给定 target performance, 速度快 3×

这个 analysis 直接挑战了 LLaVA-NeXT / OneVision 的 "anyres + ViT" 路线。**单纯 scale resolution 在 ViT 架构上不是 free lunch**, 必须同时优化 vision encoder 架构。

### 4.3 Static vs Dynamic Resolution (AnyRes)

Section 3.2.2 的 ablation 发现:

- **Static resolution 直接设置 input 大小通常更好**, 除了极高 resolution (1536+) 时 dynamic 才有优势
- Dynamic (AnyRes) 使用 2×2 tile grid 比 3×3 或更多 tile 更好

为什么? 因为 tiling 会产生 semantic breaks (tile 边界割裂 object), 2×2 tiles 的最小单元是 1024×1024, 边界割裂概率低; 而 3×3 tiles 的 768×768 单元更容易割裂 object。

FastVLM 在 2048×2048 dynamic mode 下 (4 tiles × 1024), 比 InternVL2 在 2688×2688 用 ~36 tiles 高效得多。

---

## 5. Token Pruning vs Hierarchical Architecture

Table 5 是论文的**关键对比实验**, 直接对比 FastViTHD 与 token pruning 方法:

| Method | Tokens | Avg-5 | Notes |
|--------|--------|-------|-------|
| ViT-L/14 MQT [Matryoshka] | 16 | 57.6 | Token pruning on ViT |
| **FastViTHD @ 256** | **16** | **55.5 → 60.4 (w/ multi-scale)** | Hierarchical native |
| ViT-L/14 PruMerge | 40 | ~68 (SQA) | Token pruning |
| **FastViTHD @ 512** | **64** | **60.4** | Hierarchical native |
| DynamicLLaVA | 115 | ~62 | Token pruning |
| **FastViTHD @ 768** | **144** | **62.8** | Hierarchical native |
| **FastViTHD @ 1024** | **256** | **63.9** | Hierarchical native |

**关键 insight**: hierarchical backbone (像 FastViTHD, ConvNeXT) 在 token count 优化上**本质上优于 isotropic ViT + token pruning**。因为 token pruning 在 ViT 后端做减少, vision encoder 内部仍然计算了 $N^2$ attention; 而 hierarchical 从 architecture 层面减少 token, encoder 内部计算量也同步减少。

---

## 6. CLIP Pretraining 的细节

FastViTHD 使用 DataCompDR-1B dataset 进行 CLIP pretraining, 遵循 MobileCLIP [Vasu et al., CVPR 2024] 的 setup。

Table 3 显示 FastViTHD 在 CLIP benchmark 上:
- **2.4× smaller + 6.9× faster than ViT-L/14**, zero-shot ImageNet 78.3 vs 79.2
- **2.7× smaller + 5.6× faster than ViTamin-L**, avg retrieval 67.7 vs 60.3

Reference: [MobileCLIP paper (CVPR 2024)](https://arxiv.org/abs/2311.17049)

这个 pretraining 的 quality 是 FastVLM 能在下游 VLM 任务表现出色的基础。CLIP pretraining 不仅是 visual feature quality, 还涉及 vision-language alignment 的隐式学习。

---

## 7. Training Pipeline 深度解析

论文有 2-stage 和 3-stage (实际是 4-stage 含 Stage 3) 两种 setup。

### 7.1 2-Stage Setup (LLaVA-1.5 style)

**Stage 1**: Projector-only training
- Data: LLaVA-1.5 558K (image-text alignment)
- LR: 1e-3, batch 256, 1 epoch
- Input resolution: backbone pretraining resolution (256 for FastViT, 224 for FastViTHD)

**Stage 2**: Full model finetuning
- Data: LLaVA-1.5 665K (visual instruction tuning)
- LR: 2e-5, batch 128, 1 epoch
- Input resolution: target resolution (768/1024)
- 所有 modules trainable: vision encoder + projector + LLM

### 7.2 4-Stage Setup (Scale-up)

**Stage 1.5** 是关键插入点, 用于 resolution scaling adaptation:
- Data: Recap-CC3M + Recap-CC12M (15M densely captioned pairs)
- 这一步让模型适应高分辨率输入, 因为 CLIP pretraining 时 FastViTHD 是 224 resolution
- 8× H100-80GB 上, Stage 1.5 (15M samples) 耗时 77 hours, Stage 2 (1.1M samples) 8 hours

**Stage 3**: 高质量 instruction tuning
- Data: MammothVL (10.6M, filtered single-image)
- 用于 chain-of-thought reasoning 提升
- R5, R13, R42 checkpoints 是这个 stage 的产物

### 7.3 Dataset Scaling 路径

| Dataset | Size | Composition |
|---------|------|-------------|
| 665K | 0.665M | LLaVA-1.5 original |
| 1.1M | 1.1M | 665K + AI2D, ScienceQA, ChartQA, COCO, DocVQA, DVQA, GeoQA+, OCRVQA, SAM, SynthDoG-EN, TextVQA, VG |
| 6.5M | 6.5M | 1.1M + filtered Cambrian-7M (5.4M) |
| 11.9M | 11.9M | 6.5M + LLaVA-OneVision single-image |
| 12.5M | 12.5M | 11.9M + DocMatix (0.6M) |
| 10.6M | 10.6M | MammothVL filtered (Stage 3) |

Table 6 的结果显示 dataset scaling 仍然有显著收益: R20 (1.1M) Avg-5 ~64 → R21 (15M PT + 1.1M IT) Avg-5 ~65 → R41 (15M + 12.5M) Avg-5 ~75。

---

## 8. Benchmarking Methodology 的严谨性

这部分论文做得非常 rigor:

### 8.1 Hardware Setup
- **MacBook Pro M1 Max, 32GB RAM**
- Vision encoder: Core ML package (.mlpackage), Neural Engine, XCode 15.4
- LLM: MLX framework, GPU, FP16
- Prefilling latency: `mlx_lm.cache_prompt` tool

### 8.2 TTFT 计算
$$\text{TTFT} = \text{Vision Enc Latency}(\text{CoreML, Neural Engine}) + \text{LLM Prefilling}(\text{MLX, GPU})$$

这种 setup 的意义在于: **Apple Silicon 上 Neural Engine + GPU 的协同 benchmark**, 而不是 NVIDIA GPU 上的估算。这反映了 Apple 端侧部署的真实场景。

### 8.3 MLX framework
MLX 是 Apple 开源的 ML framework, 类似 PyTorch 但优化 Apple Silicon unified memory。

Reference: [MLX GitHub](https://github.com/ml-explore/mlx)

---

## 9. 关键实验结果深度对比

### 9.1 vs LLaVA-OneVision (Table 6, R2 vs R3)

| Metric | LLaVA-OV (R2) | FastVLM (R3) | Δ |
|--------|---------------|--------------|---|
| Vision Enc Size | 430M (SigLIP-SO400M) | 125M (FastViTHD) | **3.4× smaller** |
| #Visual Tokens | 7290 (1152²) | 256 (1024²) | **28.5× fewer** |
| TTFT | 14124 ms | 166 ms | **85× faster** |
| SeedBench | 70.0 | 70.4 | +0.4 |
| MMMU | 31.4 | 30.9 | -0.5 |
| DocVQA | - | 66.7 | - |

**核心 insight**: 同样 0.5B LLM (Qwen2-0.5B), FastViTHD 用 85× 更快的 TTFT 达到 comparable performance。这彻底颠覆了 "scale resolution + scale tokens = better VLM" 的简单思路。

### 9.2 vs MM1 (Table 6, R38 vs R41)

| Metric | MM1 (R38) | FastVLM (R41) | Δ |
|--------|-----------|---------------|---|
| Vision Enc | ViT-H (632M) | FastViTHD (125M) | 5.1× smaller |
| #Visual Tokens | 720 (1344²) | 256 (1024²) | 2.8× fewer |
| GQA | 72.6 | 65.2 | -7.4 |
| TextVQA | 72.8 | 73.4 | +0.6 |
| DocVQA | 76.8 | 82.7 | **+5.9** |
| SeedBench | 82.8 | 81.6 | -1.2 |

注意 MM1 用 3000M pretraining data + 1.5M SFT, FastVLM 用 15M + 12.5M。**FastVLM 在 text-rich 任务上甚至超越 MM1**, 即使 vision encoder 小 5×。

### 9.3 vs Cambrian-1 (Table 6, R44 vs R41)

Cambrian-1 用 **4 个 vision encoders**: ConvNeXt-XXL (846M) + DINOv2-ViT-L/14 (304M) + ViT-L/14 (304M) + ViT-SO400M (430M) = ~1.88B 参数, TTFT ~5085ms。

FastVLM R41: 125M FastViTHD, TTFT 641ms, **7.9× faster**, 性能相当甚至更优 (DocVQA: FastVLM 82.7 vs Cambrian-1 77.8)。

**这表明 multi-encoder ensemble 的收益正在被 single efficient encoder 蚕食**。

---

## 10. Qualitative Failure Analysis (Section E)

论文做了非常诚实的 failure analysis:

### 10.1 Text-rich benchmarks 失败模式
- Text 太小 → 提高 resolution 可以缓解 (Table 15)
- 需要精确 alignment (e.g., reading tables) → 仍然困难
- 需要广 general knowledge → 需要 larger LLM (Table 14)
- 需要高 resolution + reasoning → 需要 larger LLM + higher resolution (Table 16)

### 10.2 三类失败对应的解决策略

$$\text{Failure Type} \rightarrow \text{Required Intervention}$$

1. **Resolution-limited** (e.g., small text): scale up $R$ 即可
2. **Knowledge-limited** (e.g., SA node anatomy): scale up $\theta_{\text{LLM}}$ 即可
3. **Joint-limited** (e.g., ChartQA 复杂推理): 需要同时 scale $R$ 和 $\theta_{\text{LLM}}$

这种 decomposition 对 VLM scaling law 研究非常有启发。

---

## 11. 与我 (Karpathy) 的工作的关联思考

### 11.1 与 LLaVA 系列的对比

LLaVA 的 core idea 是 "visual instruction tuning", 用 GPT-4 生成 multimodal instruction data。FastVLM 沿用 LLaVA-1.5 的 training setup, 但**核心创新在 vision encoder 架构**, 而非 data 或 training recipe。

这指向一个重要 trend: **VLM 的架构优化空间仍然巨大**, 不仅仅是 LLM scaling + data scaling。

### 11.2 与 nanoGPT / minimalism 的张力

FastVLM 的 5-stage hybrid 架构相对复杂, 包含 RepMixer + MHSA + DWConv + multi-scale aggregation。这与 nanoGPT 的极简主义哲学存在张力。

但仔细思考: **复杂度的来源是 hardware-aware design**。RepMixer 的 reparameterization 是为了 inference latency, multi-stage downsample 是为了 token reduction。这些不是 gratuitous complexity, 而是 **latency-budget-driven architectural choices**。

### 11.3 Token 数量 vs Sequence Length 的反思

传统 LLM wisdom: sequence length 越长, capacity 越大。但 VLM 中 visual tokens 是 "compressed representation", **更多 tokens 不等于更多信息**。

FastVLM 证明了 256 个 visual tokens (1024² input) 可以匹配甚至超越 7290 个 visual tokens (LLaVA-OV @ 1152²) 的表现。这暗示了 **visual token 的 information density 比 token count 更重要**。

类似 insight 在 Matryoshka Representation Learning [Kusupati et al., NeurIPS 2022] 中也有体现: nested representation 可以在 coarse granularity 提供足够 signal。

Reference: [Matryoshka RL paper](https://arxiv.org/abs/2205.13147)

---

## 12. 与 Efficient VLM 生态的关联

### 12.1 Token Reduction 方法谱系

1. **Post-hoc pruning** (LLaVA-PruMerge, FastV, SparseVLM, VisionZip, DynamicLLaVA): ViT 后端 token selection
2. **Perceiver resampler** (Honeybee, Flamingo): cross-attention learnable query tokens
3. **Matryoshka-style** (M³, MQT): nested token budget
4. **Hierarchical native** (ConvLLaVA, FastVLM): 架构层面减少 token

FastVLM 的 ablation (Table 5) 证明 **hierarchical native > post-hoc pruning**, 这是因为:
- Post-hoc pruning 仍然 compute vision encoder 的 $N^2$ attention
- Hierarchical native 在 encoder 内部就 reduce computation

### 12.2 Mobile/Edge VLM 部署

FastVLM 在 M1 Macbook Pro 上 benchmark, 暗示了 **on-device VLM** 的 vision:
- Neural Engine 跑 vision encoder
- GPU 跑 LLM (via MLX)
- Unified memory 共享 weights

这与 Apple Intelligence 的方向一致, 也是 mobileVLM, Llama-Vision 等工作的共同目标。

Reference: [Apple MLX](https://ml-explore.github.io/mlx/)

### 12.3 Hybrid Architecture 的复兴

纯 ViT 在 classification 上一统天下, 但在 dense prediction (detection, segmentation) 和 VLM 中, **hierarchical architecture 卷土重来**:
- ConvNeXt (ConvLLaVA)
- FastViT (FastVLM)
- ViTamin (hybrid transformer)
- Mamba-based vision encoders (emerging)

这个 trend 的 root cause 是: **ViT 的 isotropic design 在 high-resolution input 上 token count 爆炸**, 而 hierarchical 通过 multi-stage downsample 自然缓解。

---

## 13. 局限性 / Open Questions

虽然论文结果 impressive, 仍有几个 potential limitations:

### 13.1 Pretraining Data Dependence
FastViTHD 用 DataCompDR-1B 做 CLIP pretraining。如果用更大 dataset (e.g., LAION-5B, DataCompDR-12B) 是否会进一步拉开与 ViT 的 gap? 论文未探讨。

### 13.2 Video Extension
论文仅 address single image VLM。Video VLM (e.g., LLaVA-NeXT-Video, Video-LLaVA) 中 temporal dimension 增加 token count, FastViTHD 的 5-stage downsample 是否对 video token reduction 同样有效? Open question。

### 13.3 3D / Multi-view Fusion
Robotics VLM (e.g., RT-2, OpenVLA) 经常处理 multi-view camera input。FastViTHD 是否能 efficiently encode multi-view input? Architecture 没有显式设计。

### 13.4 Encoder-Decoder Capacity Mismatch
论文 Section 3.2.1 提到 "small LLM + high resolution 是 suboptimal", 但没有深入量化 **encoder capacity vs decoder capacity 的最优 ratio**。给定 fixed total FLOPs budget, 如何在 encoder 和 decoder 间分配? 这是一个重要的 open problem。

类似问题在 Kevin Li et al. "Inference optimal VLMs need only one visual token but larger models" [arXiv 2024] 中有初步探讨。

Reference: [Inference optimal VLMs paper](https://arxiv.org/abs/2405.10770)

### 13.5 Decoder-Only Vision Tokenization
Fuyu, EVE, Chameleon 等 decoder-only 架构跳过 vision encoder, 直接 tokenize raw image。论文 Section 2 提到 "performance lags behind"。但 long term, 这种 unified architecture 是否会更优雅? FastViTHD 的 hybrid insight 能否融入 decoder-only design?

---

## 14. 公式总结与 Intuition Building

### 14.1 Core Trade-off 公式

VLM efficiency 可以用 unified metric 描述:

$$\text{Efficiency Score} = \frac{\text{Avg-5}}{\log(\text{TTFT}) \cdot \log(\text{Vision Enc Size})}$$

FastVLM 的 score 显著高于 LLaVA-OV / MM1 / Cambrian-1, 体现了 Pareto 前沿优势。

### 14.2 Token Count 公式

对于 hierarchical backbone with downsample factor $D$ and input resolution $R$:
$$N_{\text{tokens}} = \left(\frac{R}{D}\right)^2$$

FastViTHD ($D=64$) vs ViT-L/14 ($D=14$) at same $R$:
$$\frac{N_{\text{ViT}}}{N_{\text{FastViTHD}}} = \left(\frac{64}{14}\right)^2 \approx 20.9\times$$

这意味着 FastViTHD 在相同 resolution 下 token 数减少 ~21×, LLM prefilling 时间相应减少。

### 14.3 Latency Decomposition Intuition

Vision encoder latency 主要来自:
$$T_{\text{vision}} \approx \alpha \cdot R^2 \cdot D_{\text{model}} + \beta \cdot N_{\text{tokens}}^2 \cdot d_{\text{attention}}$$

第一项是 conv layers (与 $R^2$ 成正比), 第二项是 self-attention (与 $N^2$ 成正比)。

FastViTHD 通过把 self-attention 推到 downsample 64× 的 stage, 使第二项的 $N^2$ 减少 4096×, 这是 speedup 的核心来源。

---

## 15. 对我 (Karpathy) 的 Intuition 影响

读完这篇 paper, 我对 VLM 的 mental model 有几点更新:

1. **VLM ≠ LLM + ViT 这个简单 picture**: vision encoder 架构选择对 latency-accuracy trade-off 有 first-order 影响, 不只是 "better visual features"。

2. **Resolution scaling 不是 free lunch**: ViT 的 isotropic architecture 在 high resolution 上 token 爆炸, 需要 architectural intervention (hierarchical downsample, token pruning, perceiver resampler) 才能可行。

3. **Edge deployment 改变 game**: 如果目标 platform 是 Apple Silicon / mobile, vision encoder 必须考虑 Neural Engine 友好 (reparameterization, depthwise conv), 这与 NVIDIA GPU 上的设计哲学不同。

4. **Multi-encoder ensemble 不一定最优**: Cambrian-1 用 4 个 encoder 但被 FastVLM 单 encoder 超越, 暗示 efficient single encoder 可能是更好 path。

5. **Token count 与 information density 的分离**: 256 个 well-designed visual tokens 可以 encode 1024² image 的 sufficient information, 7290 个 ViT tokens 不一定 information-richer。这是 VLM scaling law 的 subtle dimension。

6. **Hardware benchmarking 的重要性**: 论文在 M1 Macbook Pro 上真实 benchmark, 而非 estimate, 这是 VLM research 应该 adopt 的 rigor standard。

---

## 16. 后续可探索方向

基于这篇 paper, 我看到几个 promising follow-up directions:

### 16.1 FastViTHD + Mamba
Mamba [Gu & Dao, 2023] 在 sequence modeling 上有 $\mathcal{O}(N)$ complexity, 替代 stage 4, 5 的 MHSA 可能进一步减少 latency, 尤其在 high resolution 时。

Reference: [Mamba paper](https://arxiv.org/abs/2312.00752)

### 16.2 Learned Downsample Factor
FastViTHD 的 64× downsample 是 hand-crafted。能否 learnable downsample, 根据 input complexity 动态调整? 类似 VisionZip 的 dynamic token selection 但在 encoder 内部。

### 16.3 Cross-Resolution Knowledge Distillation
能否用 high-resolution FastViTHD teacher distill low-resolution student, 进一步 push low-resolution regime 的 Pareto?

### 16.4 VideoVLM with FastViTHD
将 FastViTHD 扩展到 video, 利用 temporal redundancy (相邻 frame 共享 visual features) 实现 video token reduction。

### 16.5 Encoder-Decoder Co-design
论文将 encoder 和 decoder 独立 optimize。能否 jointly optimize end-to-end, 学习 encoder architecture 和 decoder allocation 的最优组合?

---

## 总结

FastVLM 是一篇 solid 的 VLM efficiency paper, core contribution 是 FastViTHD 这个 hybrid vision encoder, 通过 5-stage 64× downsample 设计, 在 Apple Silicon 端侧 benchmark 上实现 3.2-85× TTFT speedup, 同时保持甚至超越 SOTA 性能。

核心 insight 在于 **architectural-level token reduction 优于 post-hoc token pruning**, 且 **self-attention 应该集中在高 downsample factor 的 stage**。这些 insight 对未来 VLM 设计, 尤其是端侧部署, 有重要指导意义。

paper 的严谨之处在于 **exhaustive Pareto analysis** (3 LLM sizes × multiple resolutions) 和 **real hardware benchmarking** (M1 Macbook Pro, Neural Engine + MLX GPU)。这种 rigor 是 VLM efficiency 研究应该 adopt 的 standard。

对于 VLM 领域的 long-term direction, 我倾向于认为 hierarchical hybrid architecture + hardware-aware design 是 mobile/edge VLM 的正确 path, 而 ViT + scale resolution + post-hoc pruning 的 LLaVA-NeXT / OneVision 路线可能在 latency-constrained 场景下被 fast-vision-encoder 架构蚕食。

Reference:
- [FastVLM GitHub](https://github.com/apple/ml-fastvlm)
- [FastViT paper (ICCV 2023)](https://arxiv.org/abs/2303.14189)
- [MobileCLIP paper (CVPR 2024)](https://arxiv.org/abs/2311.17049)
- [MLX framework](https://github.com/ml-explore/mlx)
- [LLaVA-1.5](https://arxiv.org/abs/2310.03744)
- [ConvLLaVA](https://arxiv.org/abs/2405.04319)
- [ViTamin (CVPR 2024)](https://arxiv.org/abs/2404.02100)
- [Cambrian-1](https://arxiv.org/abs/2406.16860)
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- [Mamba](https://arxiv.org/abs/2312.00752)
- [Inference optimal VLMs](https://arxiv.org/abs/2405.10770)
- [DataComp](https://arxiv.org/abs/2304.14108)
- [Apple Intelligence](https://machinelearning.apple.com/research/introducing-apple-foundation-models)
