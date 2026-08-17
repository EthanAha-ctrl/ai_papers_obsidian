---
source_pdf: COMPRESSOR-VLA.pdf
paper_sha256: 4d40f953d7ff260dd8681d1d4b7438d7fc4d8133204240269477507dcf470afb
processed_at: '2026-08-03T16:52:32-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 Compressor-VLA

---

## 痛点是什么

你想想 OpenVLA 这种 VLA model 在干啥：给它一张 camera image + 一句 language instruction，它得吐出 robot 该怎么动。

问题在于那张 image 被 ViT 切成了 **512 个 visual tokens**，全塞给 LLaMA-2-7B 处理。你算一下 512 tokens 进 self-attention，还得跑 32 层 transformer，FLOPs 直接飙到 3.95T，real-time control 基本没戏。

更要命的是——这 512 个 token 里面**大半是垃圾**。指令说"把 eggplant 放进 bucket"，背景里的 butter、cream cheese、台面纹理、光照斑点对当前任务毫无意义，但它们都作为 equal citizen 进了 LLM 的 attention。这相当于让你做一道菜，但厨房把所有食材、调料、锅碗瓢盆全倒你桌上让你自己挑——信息过载反而拉胯 performance。

---

## 之前别人怎么解决

主流是 **token pruning**——算个 importance score 把低分 token 砍掉。FastV (https://arxiv.org/abs/2403.06764)、SparseVLM (https://arxiv.org/abs/2410.04417)、VLA-Cache (https://arxiv.org/abs/2502.02175) 都是这个套路。

问题是这些方法 **task-agnostic**——它根据 attention score 砍 token，不管你指令在说啥。结果就是可能砍掉了一个对当前任务要命的小细节（比如 mug 的 handle），就因为那个 token 整体 attention 不高。

类比一下：就像你想找钥匙，一个朋友帮你"按重要性"整理房间，把看起来"不重要"的小东西全扔了——结果钥匙也被扔了，因为钥匙体积小不显眼。

---

## Compressor-VLA 的核心思路

它换了个思路：**不砍，而是重建**。

具体讲就是：别一个一个 token 决定留不留，直接用 **16 个 learnable query** 去 cross-attend 那 512 个 visual token，让这 16 个 query 自己学会"我要从这堆视觉信息里吸出什么"。这套路在 BLIP-2 的 Q-Former (https://arxiv.org/abs/2301.12597) 和 Perceiver (https://arxiv.org/abs/2107.14795) 里早就验证过。

但 Compressor-VLA 加了一个关键东西：**用 language instruction 调制这 16 个 query**。

那把钥匙的例子讲：你的朋友不是盲目整理，而是先问你"找啥"，你说"钥匙"，他就专门去检查小物件收纳盒、口袋、门边这些"可能藏钥匙的地方"，不去动沙发垫、台面这种"钥匙不太可能在"的地方。**指令把整理行为本身 task-conditioned 了**。

---

## 两条路并行

但单靠 16 个 query 压缩 512 token 有个隐患——**空间细节丢了**。比如抓 mug 你得知道 handle 在哪，光知道"图里有 mug"不够，得知道 mug 把手朝向哪边。全局 summary 撑不住这种精细需求。

所以作者搞了**两条并行路径**，最后 concat 起来：

### Path 1: STC (Semantic Task Compressor) — "大局观"

这一路负责 **what to do + where to go**。16 个 learnable query，用 FiLM (https://arxiv.org/abs/1709.07871) 被 instruction 调制一下，然后 cross-attend 全部 512 visual token，蒸馏出 16 个 "concept summary"。

公式上其实就三步：
1. 把 language instruction 过 MLP 得到 task embedding $E_L$
2. $E_L$ 再过一个小 MLP 生成 scale $\gamma$ 和 shift $\beta$（这是 FiLM 标准动作）
3. 用 $\gamma \odot Q + \beta$ 把原始 learnable query 给 "tilt" 一下，让它们偏向当前任务关心的 concept
4. 调制后的 query cross-attend visual tokens，输出 16 个 summary token

你可以把 16 个 learnable query 想成 16 个"概念探测器"，平时它们是泛化的；指令来了，FiLM 给每个探测器加一个 "task-specific 偏置"，让它们瞬间变成"针对当前任务的探测器"。指令说"抓 mug"，第 3 个 query 就变成 "mug detector"；指令说"开抽屉"，它又变成 "handle detector"。

### Path 2: SRC (Spatial Refinement Compressor) — "细节控"

这一路负责 **how to act**——保住空间细节。

它不在 token sequence 上操作，而是把 visual feature reshape 成 2D grid（$H \times W$），切成 $w \times w$ 的小窗口（$w=2$，每窗 4 个 token），每个窗口用一个 query 把 4 个 token 压成 1 个。

这里 instruction 是 **直接相加**进 query 的，不用 FiLM。为什么？作者做 ablation 发现 SRC 用 FiLM 反而掉 0.3%——因为 SRC 的任务是保住 spatial fidelity，FiLM 的 scale-shift 太"aggressive"会扭曲 representation；additive 只是一个 gentle hint，保持 query 接近原始 visual content。

这其实是个很 elegant 的 finding：**全局路径需要 aggressive 调制（FiLM），局部路径需要 gentle 调制（additive）**。两者 conditioning 强度匹配各自的目标。

### 最终

STC 输出 16 tokens（全局语义），SRC 输出 144 tokens（局部空间），concat 起来 **160 tokens** 进 LLM。比原来 512 压缩了 3.2x，FLOPs 从 3.95T 降到 1.62T。

---

## 为什么这个思路 work

### 1. Reconstruction 比 Pruning 更稳

Pruning 是 hard decision——一旦砍了就回不来。Reconstruction 是 soft bottleneck——16 个 query 学的是"该提取什么"，而不是"该删什么"，梯度友好，也不容易误删关键信息。

### 2. Instruction guidance 把噪声过滤前置了

原本 512 token 全进 LLM，task-irrelevant 噪声也全进了，LLM 得自己学忽略它们。Compressor-VLA 在 ViT 输出后立刻用 instruction 做 filter，**噪声压根没进 LLM**。Table 1 显示 SR 反而涨 0.2%（97.3 vs 97.1），说明这些"噪声"对 baseline 确实是负贡献。

### 3. Hybrid 架构匹配机器人认知的双层需求

机器人 control 本来就是两层：high-level planning（"我要去抓那个 mug"）+ low-level control（"手指该怎么张、力度多大"）。STC 对应前者，SRC 对应后者。单路径做不出这种分工。Q-Former 那种纯 query-based aggregation 只能顾 high-level，spatial 细节保不住。

---

## 几个有意思的 ablation 发现

**Table 2 的 ablation** 信息量很大：

- **STC-Only 32 tokens 还能跑 95.9%**：说明 instruction-guided bottleneck 极强，16 个 query 就能 cover 大部分 LIBERO 语义。但 Spatial suite 从 98.8 掉到 96.0，证明空间细节确实要靠 SRC。
- **SRC-Only 95.5%**：比 STC-Only 还低，说明 global semantic 不可缺，单靠 local window 撑不起 long-horizon reasoning。
- **No Guidance 96.3% vs Full 97.3%**：instruction guidance 贡献 1% 绝对增益，主要在 LIBERO-Goal 上（93.8 → 96.4），因为 Goal 任务要根据目标序列动态切 focus，task-agnostic 压缩根本处理不了。
- **STC+SRC-FiLM 97.0% < STC+SRC 97.3%**：再次印证 SRC 不能用太强的 conditioning。

**Table 3 超参 sensitivity**：
- $k=16$ 是 sweet spot，$k=32$ 没涨点反而 FLOPs 多——16 个 concept detector 够用了
- $w=2$ 最佳，$w=4$ 掉 2.2%，$w=8$ 掉 3.4%——spatial fidelity 是 cliff effect，超过阈值就崩

---

## Real-World 的反直觉结果

Mobile ALOHA (https://arxiv.org/abs/2401.02117) 双臂真机实验：

| Task | OpenVLA-OFT | Compressor-VLA |
|---|---|---|
| Spatial Awareness | 91.7% (22/24) | **100% (24/24)** |
| Semantic Understanding | 76.7% (23/30) | **83.3% (25/30)** |

**Compressor-VLA 在真机上反而比 baseline 高**。这个反直觉结果其实合理——真机 noise 比 sim 多得多（光照变化、遮挡、纹理干扰），baseline 把全 512 token 喂进去等于把 noise 也全喂了；Compressor 用 instruction 做 prior filter，等价于 task-conditioned denoising，noise 进不来，performance 反而稳。

不过 trial 数太少（54 个 total）统计意义有限，这点作者该多跑几个 task。

---

## 我对这篇 paper 的整体判断

**Strength**:
- Reconstruction + instruction conditioning 的组合很 clean，借鉴了 Q-Former 但加了 task modulation
- Hybrid 双路架构符合机器人控制 "planning vs control" 的天然分层
- Same-source conditioning（用 LLM 自己的 embedding 而非外部 CLIP）省参数又避免 distribution mismatch
- Real-robot deployment 真做出来了，sim-to-real transfer 这关过了

**Weakness**:
- 真机实验太少，54 trial 的统计意义弱
- 没有 wall-clock latency 数据，只有 FLOPs——FLOPs ≠ real-time
- $k=16$ 固定 bottleneck 在 object 数量 > 16 的复杂任务上可能不够
- SRC 的 $w=2$ local window 没考虑 non-local interaction，抓长物体可能出问题
- Mean pool instruction 会丢顺序信息，对 "先 A 后 B" 的多步指令不够友好

**Open questions**:
- 能不能让 $k$ 动态——比如 task 复杂时自动多分 query？
- 这个 instruction-guided compression 能不能搬到 LLaVA、Qwen-VL 上做 VQA token reduction？
- Action token 也该压缩，OpenVLA 把 action discretize 成 7 个 token，π0 用 flow matching，这块冗余在哪没人研究
- 3D 场景扩展——point cloud 输入下 SRC 该怎么设计？VoxPoser (https://arxiv.org/abs/2307.05973) 那种 voxel 表示下还有"window"概念吗

---

## 一句话总结

**Compressor-VLA 干的事就是把"视觉 token 压缩"从"按分数砍"翻到"按任务重建"**——用 16 个被指令调制的 learnable query 提取全局语义，用 144 个被指令 hint 过的 windowed query 保住空间细节，160 token 进 LLM，省 59% FLOPs 还不掉点。

直觉上讲，这论文的核心信念是：**在 task-irrelevant 噪声主导的 VLA 视觉输入里，正确的压缩就是 denoising**。这个 insight 其实挺 deep——它说明 VLA 当前 bottleneck 不只是算力不够，更是"信息没被 task-aware 过滤"。

对 VLA 这领域的意义类似 Q-Former 对 BLIP-2 的意义：把高维感知输入"翻译"成 LLM 消化得了的 compact representation，只不过这个 bottleneck 现在被任务指令动态调制了。下一波 efficient VLA 工作估计都会沿这个方向走。

---

# Compressor-VLA 深度讲解

Karpathy 你好，这篇 paper 我读了三遍，下面把我对它的工程直觉和数学细节都摊开讲。

---

## 1. 问题定位：VLA 的"视觉 token 墙"

VLA 的标准三段式架构：**ViT (视觉感知) → LLM (推理) → Action Head (动作解码)**。瓶颈在 ViT 把单帧图像切成了 $H/16 \times W/16$ 的 patch grid，每个 patch 投影成一个 $D$ 维 token，OpenVLA-OFT 上是 **512 个 visual tokens** 进入 LLaMA-2-7B 的 KV cache。

关键算术：LLM self-attention 的复杂度对序列长度 $N$ 是 $O(N^2)$，pre-fill 阶段 512 tokens × 4096 hidden dim × 32 layers × token dim 的 FLOPs 累计到 3.95T（Table 1），推理延迟对 real-time control（典型要求 5-15Hz）就是死路。

更阴险的问题：**task-agnostic 的冗余**。比如指令是"把 purple eggplant 放进 bucket"，背景里的 cream cheese、butter 对这个任务就是噪声。Token pruning 类方法（FastV、SparseVLM、VLA-Cache）按 attention score 砍 token，但 score 是模型自评的，并不指向 task semantics。

**Compressor-VLA 的核心立场**：把语言指令作为 "query prior" 注入到 token 压缩过程，并且**重建**（reconstruct）一个 compact set，而不是**剪枝**（prune）——这是一个 reconstructive bottleneck 的设计哲学，和 Perceiver、Q-Former 一脉相承，但多了 task-conditioning。

参考：
- Perceiver IO: https://arxiv.org/abs/2107.14795
- Q-Former (BLIP-2): https://arxiv.org/abs/2301.12597
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645

---

## 2. 整体架构 (Figure 2 解析)

数据流分两条并行路径，最后 concat：

```
Camera image
    │
    ▼
[DINOv2 + SigLIP] (frozen + LoRA) → X ∈ ℝ^(N×D), N=256 (each), merged to 512
    │
    ├──► [STC] ──► Z_G ∈ ℝ^(k×D), k=16  (global semantic summary)
    │
    └──► [SRC] ──► Z_L ∈ ℝ^(N'×D), N'=144 (local spatial detail)
    │
    ▼
Concat → Z ∈ ℝ^(160×D) ──► LLaMA-2-7B (LoRA) ──► action tokens ──► Action Head
                ▲
                │
Language instruction → LLM embeds → mean pool → L_pooled (conditioning signal)
```

**总 token 从 512 压到 160**（3.2x 压缩比），FLOPs 从 3.95T 降到 1.62T（59% reduction）。

这里有一个**重要设计选择**：conditioning signal $L_{pooled}$ 不从外部 CLIP text encoder 取，而是从 VLA 自己的 LLM embedding 取 mean pool。论文给的理由是 (i) 参数省 (ii) LLM 内部 embedding 本身语义足够丰富。我觉得这里还有第三个隐含好处：**避免了 distribution mismatch**——CLIP text encoder 和 LLaMA-2 的 embedding 空间不共享，做 cross-attention 会有表示对齐的负担，用同源 embedding 等于"在自己的语言空间里做 modulation"。

---

## 3. Semantic Task Compressor (STC) — 全局语义路径

STC 的角色：**指令调制的 Q-Former**。它把整张图压缩成 k=16 个 "concept queries"。

### 公式逐个拆解

**Eq. (1):** 
$$E_L = \text{MLP}_{\text{STC}}(L_{pooled})$$

- $L_{pooled} \in \mathbb{R}^{D}$: 指令 embedding 经过 LLM 后 mean pool 得到的向量（$D$ 是 LLM hidden dim，OpenVLA 里 $D=4096$）
- $\text{MLP}_{\text{STC}}$: 一个独立的 small MLP，把 $L_{pooled}$ 投影到 STC 工作空间 $E_L \in \mathbb{R}^{D'}$（$D'$ 通常等于 query embedding dim）
- $E_L$: 称作 "semantic task representation"——指令的语义压缩形态

**Eq. (2):** 
$$\gamma, \beta = \text{MLP}_{\text{FiLM}}(E_L)$$

- $\text{MLP}_{\text{FiLM}}$: FiLM 生成器，输出两个向量
- $\gamma \in \mathbb{R}^{k \times D'}$: **per-query scale** (每个 learnable query 有自己的 scale)
- $\beta \in \mathbb{R}^{k \times D'}$: **per-query shift** (每个 learnable query 有自己的 shift)

这是 FiLM 的标准配方（Perez et al. 2018, https://arxiv.org/abs/1709.07871），核心思想：conditioning signal 不直接 concat 到 query，而是生成 affine 变换参数去 **modulate** query。等价于让每个 query 知道："在这个指令下，我应该激活什么 mode"。

**Eq. (3):** 
$$Q_{\text{con}} = \gamma \odot Q + \beta$$

- $Q \in \mathbb{R}^{k \times D'}$: **learnable queries**，k=16，类似 Perceiver 的 latent array 或 DETR 的 object queries
- $\odot$: Hadamard product (element-wise)
- $Q_{\text{con}} \in \mathbb{R}^{k \times D'}$: **conditioned queries**——指令调制后的查询

物理直觉：$Q$ 是 "concept detectors"，$\gamma, \beta$ 让这些 detectors "tilt" 向当前指令关心的方向。比如指令是"抓 mug"，$\gamma, \beta$ 让第 3 个 query 变成 "mug-detector"，第 7 个变成 "rim-detector"。

**Eq. (4):** 
$$Z_G = \text{Attention}(Q=Q_{\text{con}}, K=X, V=X)$$

- 标准 cross-attention
- $Q = Q_{\text{con}} \in \mathbb{R}^{k \times D'}$ (k=16 query)
- $K = V = X \in \mathbb{R}^{N \times D}$ (N=512 visual tokens)
- 输出 $Z_G \in \mathbb{R}^{k \times D'} = \mathbb{R}^{16 \times D'}$——只 16 个 token！

这是 information bottleneck 的极致体现：512 → 16，模型被迫只保留指令最关心的全局语义。

**STC 的注意力可视化** (Figure 4) 特别有意思：同一个 scene，指令 "alphabet soup + tomato sauce" 时 attention 落在 soup can；指令变成 "cream cheese + butter" 时 attention 动态迁到 cream cheese box。而且模型还表现出了 **temporal anticipation**——只 focus 在即将要抓的下一个物体，而非所有提到的物体。这是 VLA pretraining 留下的 inductive bias，被压缩器继承了。

---

## 4. Spatial Refinement Compressor (SRC) — 局部空间路径

STC 把 512 压到 16，理论上信息 bottleneck 太窄，精细动作（抓把手、对齐 rim）会丢。SRC 的任务：**保住 spatial precision**。

### 公式拆解

SRC 不在 token 序列上操作，而在 reshape 后的 2D feature map 上做 **windowed attention**。

输入 reshape：$X' \in \mathbb{R}^{H \times W \times D}$，对 N=512 个 token，假设输入 224×224 → patch 14×14，所以 $H=W=14$。

对每个 $w \times w$ 局部窗口（论文取 $w=2$，所以一个窗口含 4 个 token）：

**Eq. (5):** 
$$q_{raw} = \text{Downsample}(X_w)$$

- $X_w \in \mathbb{R}^{w \times w \times D}$: 局部窗口 token
- $\text{Downsample}(\cdot)$: 通常用 strided conv 或 adaptive avg pool，把 $w \times w$ 压到 1×1 然后 flatten
- $q_{raw} \in \mathbb{R}^{D}$: 该窗口的 "平均查询"——一个 spatially coarse summary

**Eq. (6):** 
$$q_w = q_{raw} + E_L'$$

- $E_L' = \text{MLP}_{\text{SRC}}(L_{pooled})$: 指令在 SRC 路径的 projection（注意：与 STC 的 $\text{MLP}_{\text{STC}}$ 独立，不共享权重）
- 这里是 **直接相加**（additive injection），没有用 FiLM

为什么 STC 用 FiLM 但 SRC 用 additive？论文 4.2.1 的 ablation 给出答案：**STC+SRC-FiLM 变体比 STC+SRC 低 0.3%**（97.0% vs 97.3%，Table 2）。SRC 的任务是 preserve spatial fidelity，FiLM 的 nonlinear scale-shift 会过度 "distort" representation；additive 只是一个 "gentle hint"，保持 query 接近 raw visual content。

这是一个很 elegant 的 ablation finding，说明 **不同路径需要不同强度的 conditioning**：
- STC 是 "what to look for" → 需要 aggressive semantic modulation → FiLM
- SRC 是 "where exactly" → 需要 gentle bias → additive

**Eq. (7):** 
$$z_w = \text{Attention}(Q=q_w, K=X_w, V=X_w)$$

- $Q = q_w \in \mathbb{R}^{D}$ (单 query)
- $K = V = X_w \in \mathbb{R}^{w^2 \times D}$ (窗口内 4 个 token)
- $z_w \in \mathbb{R}^{D}$: 该窗口压缩后的 1 个 token

最终 $Z_L = \text{Concat}([z_1, z_2, \ldots, z_{N/w^2}]) \in \mathbb{R}^{N' \times D}$

$N' = N / w^2 = 512 / 4 = 128$？但 Table 1 说 compressed token 是 160——因为 $Z = Z_G \oplus Z_L = 16 + 144 = 160$。所以 $N'_{\text{actual}} = 144$，意味着实际 feature map $H \times W = 144 = 12 \times 12$，可能 patch size 是 16 而非 14（输入 192×192）或者有其他边界处理。

**关键 intuition**：SRC 是一个 **windowed self-attention 压缩器**，每个窗口内做"小 cross-attention"，token 数从 $w^2 \to 1$。相比 STC 的"全局 16 query bottleneck"，SRC 是 **spatially-local** 的、保拓扑的、信息损失极小的。它扮演的是"低分辨率 semantic 概图 + 高分辨率 spatial 详情"两路融合里的后者。

---

## 5. 实验：LIBERO 数据深度解读

LIBERO 四个 task suite (Liu et al. 2023, https://arxiv.org/abs/2306.03310)：

| Suite | 挑战重点 |
|---|---|
| LIBERO-Spatial | 复杂空间关系 ("把白色 mug 放在左侧 plate 上") |
| LIBERO-Object | 不同 object 的泛化 |
| LIBERO-Goal | 不同目标序列 |
| LIBERO-Long | Long-horizon 多步任务 |

Table 1 的关键对比：

| Model | Avg. SR | FLOPs (T) | Tokens |
|---|---|---|---|
| OpenVLA-OFT (baseline) | 97.1 | 3.95 | 512 |
| CogACT | 93.6 | — | 512 |
| π0 | 94.2 | — | 512 |
| FastV | 86.8 | 3.18 | 256 |
| SparseVLM | 95.6 | 3.04 | 256 |
| SpecPrune-VLA | 96.6 | 1.70 | 197 |
| VLA-Cache | 97.0 | 3.28 | 212 (min) |
| **Compressor-VLA** | **97.3** | **1.62** | **160** |

观察：
1. **比 baseline 高 0.2% 而非降**——这点最 striking。说明 token 压缩不是单纯"少即损失"，task-conditioned 压缩反而是 **正则化**，去掉了 task-irrelevant 噪声。这和 ConvNeXt-pruning、Llm-pruning 的某些观察一致：当 redundancy 是 noise 时，压缩 ≈ denoising。

2. **vs SpecPrune-VLA (97.3 vs 96.6, FLOPs 1.62 vs 1.70)**：两者 FLOPs 接近，但 Compressor-VLA 在 SR 上 +0.7%。SpecPrune 用 self-speculative decoding 做 token selection（https://arxiv.org/abs/2509.05614），仍是 pruning 范式；Compressor 是 reconstructive，更有优势。

3. **vs VLA-Cache (97.3 vs 97.0, FLOPs 1.62 vs 3.28)**：VLA-Cache (https://arxiv.org/abs/2502.02175) 复用上一步 KV，**只省 inference latency 不省 FLOPs**。Compressor-VLA 真省 FLOPs，且 SR 更高。

4. **vs π0 (97.3 vs 94.2)**：π0 (https://arxiv.org/abs/2410.24164) 是 flow-matching VLA，参数量和 FLOPs 大得多，Compressor 用更小代价超过它——这是个 fair-ish comparison，π0 是 generalist 多任务，但 LIBERO 单测上 Compressor 的 task-conditioned 优势体现出来了。

---

## 6. Ablation 深度解析

### 6.1 组件 ablation (Table 2)

| Config | Avg SR | FLOPs | Tokens |
|---|---|---|---|
| STC+SRC (full) | 97.3 | 1.62 | 160 |
| STC+SRC-FiLM (SRC 也用 FiLM) | 97.0 | 1.62 | 160 |
| No Guidance (去 instruction) | 96.3 | 1.43 | 160 |
| STC-Only | 95.9 | 0.76 | 32 |
| SRC-Only | 95.5 | 1.20 | 128 |

**核心 takeaways**：
- **STC-Only 仅 32 token 仍达 95.9%**——证明 task-conditioned bottleneck 极强，纯 query-based aggregation 就能撑住 LIBERO 大部分语义。但 **Spatial 任务从 98.8 掉到 96.0**，说明 spatial detail 是 SRC 不可替代的价值。
- **SRC-Only 95.5%**——单 SRC 比 STC-Only 还低，说明 global semantic context 对 long-horizon reasoning 不可缺。
- **No Guidance 96.3 vs Full 97.3**——instruction guidance 贡献 1.0% 绝对增益，主要来自 LIBERO-Goal (93.8→96.4, +2.6%)。Goal 任务需要根据目标序列动态切换 focus，task-agnostic 压缩处理不了。
- **STC+SRC-FiLM vs STC+SRC**：SRC 用 FiLM 反而降 0.3%，再次印证 SRC 需要 "gentle" modulation。

### 6.2 超参 sensitivity (Table 3, LIBERO-Long)

**Global queries k 的扫描**：
- k=8: 94.2%, 1.51T, 144 tokens
- k=16: 94.8%, 1.62T, 160 tokens ← best
- k=32: 94.8%, 1.83T, 192 tokens

性能曲线在 k=16 处 saturate，更多 query 只是冗余计算。这说明 **16 个 "concept detector" 足以表达 LIBERO 任务的全局语义**——很 compact。

**Local window w 的扫描**：
- w=2: 94.8%, 1.62T, 160 tokens ← best
- w=4: 92.6%, 0.86T, 64 tokens
- w=8: 91.4%, 0.70T, 40 tokens

w 越大压缩越狠但掉点越多。w=8 时单窗口含 64 个 token 压成 1 个，spatial detail 损失太重。**这条曲线的陡峭性**说明 SRC 的 spatial fidelity 是 "cliff effect"——超过阈值就崩。$w=2$ 几乎是 sweet spot，4 token 压 1 个，spatial granularity 大约减半但保留了 2D 拓扑。

### 6.3 Real-World (Mobile ALOHA 双臂)

https://arxiv.org/abs/2401.02117 (Mobile ALOHA) 平台，cobot magic dual-arm + Piper 7-DoF × 2 = 14 DoF。

| Task | OpenVLA-OFT | Compressor-VLA |
|---|---|---|
| Spatial Awareness (put X into bucket) | 91.7% (22/24) | **100% (24/24)** |
| Semantic Understanding (Tower of Hanoi) | 76.7% (23/30) | **83.3% (25/30)** |

**Compressor-VLA 在真机上反而比 baseline 高**，这个反直觉的结果很有信息量。可能解释：
- 真机 noise 比 sim 多（光照、遮挡、纹理），baseline 处理全 512 token 时 noise 也全进了 LLM；Compressor 用 instruction 做 prior filter，等价于 **task-conditioned denoising**。
- 这点和 EfficientNet 时代发现的 "适当容量限制改善 generalization" 类似。

---

## 7. 与相关方法的谱系关系

### 7.1 Token Compression 家族

| 方法 | 范式 | Task-aware? | 代表作 |
|---|---|---|---|
| Token Pruning | hard discard by score | ✗ | FastV, SparseVLM, SP-VLA |
| Token Merging | soft merge by similarity | ✗ | ToMe (Bolya et al. https://arxiv.org/abs/2210.09461) |
| Query-based Aggregation | learnable query bottleneck | ✗ (Q-Former) / ✓ (本文) | Flamingo, BLIP-2, Compressor-VLA |
| KV Cache reuse | temporal reuse | ✗ | VLA-Cache |
| Speculative Pruning | action-conditioned score | partial | SpecPrune-VLA |

Compressor-VLA 是 **"Query-based Aggregation + Instruction Conditioning"** 的组合，定位很清楚。

### 7.2 Instruction Conditioning 家族

- **FiLM** (Perez et al. 2018, https://arxiv.org/abs/1709.07871): VQA 时代提出的 conditioning 层，本文 STC 用
- **AdaIN** (Huang & Belongie 2017, https://arxiv.org/abs/1703.06868): style transfer 里用，但本质类似 FiLM
- **Q-Former conditioning**: BLIP-2 里 Q-Former 不带 task conditioning，本文相当于给 Q-Former 加了 instruction FiLM
- **Cross-attention as conditioning**: DETR object queries 本身也算一种，本文更动态

### 7.3 VLA 架构家族

- **OpenVLA** (https://arxiv.org/abs/2406.09246): 单 frame + DINOv2+SigLIP fusion + LLaMA-2-7B
- **OpenVLA-OFT** (https://arxiv.org/abs/2502.19645): fine-tuning 改进，本文的 baseline
- **π0** (https://arxiv.org/abs/2410.24164): flow matching + PaLI-Gemma 2B
- **CogACT** (https://arxiv.org/abs/2411.19650): action tokenizer 改进
- **RoboFlamingo** (https://arxiv.org/abs/2310.02193): 早期 VLM-policy 设计

---

## 8. 我对这篇 paper 的批判性思考

### 强项

1. **Reconstruction vs Pruning 哲学**：剪枝是 hard decision 不可逆，reconstructive bottleneck 是 soft 信息蒸馏。后者梯度更友好，且避免 "重要 token 误杀"。这是从 VLM token compression 学到的 lesson (Q-Former 早就证明了)，本文第一次系统带到 VLA。

2. **Hybrid architecture**：global/local 双路分工非常符合 robot manipulation 的认知结构。规划层要 "what"，执行层要 "where"。Q-Former 单路径做不到这种分工。

3. **Same-source conditioning**：从 LLM 自己的 embedding 取 instruction representation 而非外部 CLIP，是工程上很聪明的简化。

### 弱项 / 可以质疑

1. **k=16 的固定 bottleneck 在 long-horizon 任务上的极限**：LIBERO-Long 表现 94.8% 还可以，但如果任务 object 数量 > 16，STC 的 16 query 可能不够。理论上应该 dynamic k 或者 hierarchical query。

2. **SRC 的 w=2 window 没考虑 non-local interaction**：抓取有时需要 object 的全局 shape（如长棍子），w=2 完全 local 可能不够。Swin 风格的 shifted window 或 dilated window 可能更好。

3. **真机实验只有 2 个任务、54 个 trial**：n 太小，100% vs 91.7% 差异在 24 trial 上统计意义不强。需要更多 task、更多 trial 才能确认 sim-to-real 优势。

4. **没有 latency 测量**：FLOPs 降了 59%，但 wall-clock 时间还要看 KV cache 实现。论文没给 inference latency 数据，对 real-time 部署是关键缺失。

5. **conditioning signal 是 mean-pooled**：变长指令 mean pool 会损失顺序信息，对 "先 A 后 B" 的多步指令可能不够。attention pool 或者用 instruction 的 last hidden state 可能更合理。

### Open Questions / 联想方向

1. **能否把 STC 的 16 query 改成 learnable但 task-specific**？比如用 task embedding 做 retrieval，给不同任务 family 分配不同 query set。
2. **3D 场景的扩展**：本文是单帧 2D feature map。如果输入是 point cloud 或 multi-view，SRC 的 windowed attention 是否还能保拓扑？VoxPoser (https://arxiv.org/abs/2307.05973) 那种 3D voxel 表示下，SRC 该怎么设计？
3. **VLM 通用化**：这个 instruction-guided compression 思路能否直接搬到 LLaVA、Qwen-VL 上做 VQA token reduction？值得做。其实 LLaVA-PruMerge (https://arxiv.org/abs/2403.03793) 已经在探索类似方向。
4. **Action token 也压缩**：本文只压 visual token。OpenVLA 把 action discretize 成 7 个 token，π0 用 flow matching。Action representation 的冗余在哪？这是 orthogonal 的方向。
5. **Closed-loop 与 KV cache reuse 的结合**：Compressor 已经把 token 降到 160，能否进一步在时序上 reuse？类似 VLA-Cache 但在 compressed space 做。

---

## 9. 关键链接汇总

- **Paper**: Compressor-VLA (this work)
- **OpenVLA**: https://arxiv.org/abs/2406.09246
- **OpenVLA-OFT**: https://arxiv.org/abs/2502.19645
- **LIBERO benchmark**: https://arxiv.org/abs/2306.03310
- **Mobile ALOHA**: https://arxiv.org/abs/2401.02117
- **FiLM**: https://arxiv.org/abs/1709.07871
- **Perceiver IO**: https://arxiv.org/abs/2107.14795
- **BLIP-2 / Q-Former**: https://arxiv.org/abs/2301.12597
- **FastV**: https://arxiv.org/abs/2403.06764
- **SparseVLM**: https://arxiv.org/abs/2410.04417
- **VLA-Cache**: https://arxiv.org/abs/2502.02175
- **SpecPrune-VLA**: https://arxiv.org/abs/2509.05614
- **Token Merging (ToMe)**: https://arxiv.org/abs/2210.09461
- **π0**: https://arxiv.org/abs/2410.24164
- **CogACT**: https://arxiv.org/abs/2411.19650
- **Open X-Embodiment**: https://arxiv.org/abs/2310.08864
- **Octo**: https://arxiv.org/abs/2405.12213
- **EfficientVLA**: https://arxiv.org/abs/2506.10100
- **SP-VLA**: https://arxiv.org/abs/2506.12723
- **RT-1**: https://arxiv.org/abs/2212.06817
- **RT-2**: https://arxiv.org/abs/2307.15818
- **OTTER**: https://arxiv.org/abs/2503.03734

---

## 10. 一句话总结直觉

Compressor-VLA 把"视觉 token 压缩"从 **pruning-based score selection** 翻转到 **instruction-conditioned reconstructive bottleneck**，并通过 STC (global FiLM-modulated queries) + SRC (local windowed attention + additive injection) 的 hybrid 架构，同时撑住 high-level semantic reasoning 和 low-level spatial precision。LIBERO 上 3.2x token 压缩、59% FLOPs 削减，SR 反升 0.2 个点的结果说明：在 task-irrelevant 噪声主导的 VLA 视觉输入里，**正确的压缩本身就是 denoising**。

如果要类比，这个工作对 VLA 的意义类似 Q-Former 对 BLIP-2 的意义：用一个 learnable bottleneck 把高维感知输入"翻译"成 LLM 消化得了的 compact representation，只不过这个 bottleneck 现在被任务指令动态调制了。
