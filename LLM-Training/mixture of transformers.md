---
source_pdf: mixture of transformers.pdf
paper_sha256: 79eb51c94c39c824257c210b3d1975c4997e011b9cb3f1ee11aedd498c2f7af6
processed_at: '2026-08-05T18:53:41-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MoT 用人话讲

## 一句话版本

**多模态模型里, 文本 token 和图像 token 天生就是不一样的东西, 硬塞进同一套参数里训练又慢又别扭 — MoT 给每个模态配一套独立的参数, 但 attention 还让它们互相看, 训练直接快一倍。**

---

## 这事儿怎么来的

你训过一个多模态模型就知道了, 拿 Chameleon 来说, 要 9.2T tokens 才能追上 LLaMA2 用 2T tokens 达到的文本水平。算力翻 4 倍多就为了加个图像, 这账怎么算都不划算。

作者做了件很聪明的事: 把 Chameleon+Speech 7B dense model 的中间层 activations 拉出来做 PCA, 结果一看 — 文本、语音、图像 token 自动分成了三团 (Figure 2, Figure 23)。模型自己都没被告知"这是文本那是图像", 它自己在内部偷偷把不同模态推到了特征空间的不同区域。

这就像你家猫, 你没教它区分猫粮和玩具, 但它自己知道哪个能吃哪个不能。**模型在偷偷做的事, 我们干脆显式地帮它做** — 给每个模态一套独立参数, 省得它费劲去学这个分离。

---

## 架构到底改了啥

### Dense Transformer (原来的样子)

每个 token 都过同一套 attention 参数、同一套 FFN 参数。文本 token 和图像 token 共享一切。就像不管你来的是中文书还是英文书, 都用同一个翻译官翻, 这翻译官得同时精通两种语言, 训练起来当然慢。

### MoE (Mixture of Experts)

加几个 expert FFN, 学一个 router 来决定每个 token 该去哪个 expert。问题是 router 要自己学, 训练早期 router 和 expert 都没训好, 互相拖累。还要搞 load balancing 不然有的 expert 闲死有的累死。DeepSeek-V3 用了 256 个 expert, 参数量爆炸。

### MoT (这篇 paper)

**核心 insight: 模态标签是免费的、确定的、不会错的**。

你用 VQGAN 把图像 tokenize 成离散 token, 这些 token 的来源你早就知道是图像。用 speech tokenizer 编出来的 token 你也知道是语音。这个信号不需要学, 不需要 router, 不会 OOD, 不会 load imbalance。

所以 MoT 干脆这么搞:

```
文本 token → 用文本那套 W_Q, W_K, W_V, W_O, FFN, LayerNorm
图像 token → 用图像那套 W_Q, W_K, W_V, W_O, FFN, LayerNorm
语音 token → 用语音那套 W_Q, W_K, W_V, W_O, FFN, LayerNorm

但是! attention 是全局的 — 文本 token 可以看图像 token, 图像 token 也可以看文本 token
```

用公式说就是 (公式 2 和 3):

```
Q_i = x_i · W_Q^{m_i}     # m_i 是 token i 的模态, 用对应模态的 W_Q
K_i = x_i · W_K^{m_i}
V_i = x_i · W_V^{m_i}

A = softmax(Q · K^T / √d_k) · V    # 这步是全局的, 跨所有模态

output_i = A_i · W_O^{m_i}          # 输出投影也按模态走
output_i = output_i + LayerNorm^{m_i}(...)
output_i = FFN^{m_i}(output_i) + ...
```

变量含义:
- `x_i ∈ R^d`: 第 i 个 token 的 hidden state, d 是 hidden dimension
- `m_i ∈ {text, image, speech}`: 这个 token 属于哪个模态 (训练前就知道)
- `W_Q^{m_i} ∈ R^{d × d_k}`: 模态 m_i 专属的 query 投影矩阵, d_k 是 per-head 的 key/query 维度
- `W_K^{m_i}, W_V^{m_i}`: 同理, 模态专属的 key/value 投影
- `W_O^{m_i} ∈ R^{d × d}`: 模态专属的 output 投影
- `LayerNorm^{m_i}`: 模态专属的 LayerNorm (后面 ablation 证明这个几乎没用)
- `FFN^{m_i}`: 模态专属的 FFN, 这是参数大头
- `√d_k`: 标准 attention 的 scaling factor

**关键 trick**: Q/K/V 投影是模态专属的, 但 attention score 是全局算的。意思是文本 token 的 query 会去和图像 token 的 key 算点积, 得到 cross-modal attention — 这就是融合的来源。然后输出再投影回各自模态的表示空间。

这就像你开一个国际会议, 每个国家代表用自己的语言写议题 (模态专属投影), 但议题放进同一个会议室让所有人讨论 (全局 attention), 讨论完各自用自己的语言记录回去 (模态专属输出投影)。代表之间能互相听懂 (cross-modal attention), 但各自保持自己的专业领域 (专属参数)。

---

## FLOP 这事儿容易搞混

很多人第一反应: "MoT 参数变多了, 那 FLOP 也变多了吧?"

**没有**。FLOP 和 dense model 一模一样。因为:
- 每个 token 还是被处理一次, 只是用了对应模态的那套参数
- 全局 attention 还是 n×n 的
- FFN 还是过一次

参数确实变多了 (K-1)×7D² per layer (K 是模态数, D 是 hidden dim), 但每个 token 只激活其中一套, 所以是 sparse activation, isoFLOP。

那 MoT 凭啥更快? **凭 optimization 效率高**。当模型不用费劲在共享参数里同时记住"文本该怎么处理"和"图像该怎么处理", 它学得更快。就像两个人分别学法语和日语, 各自专精比一个人同时学两门要快。

---

## 实验结果到底有多牛

### Chameleon 设定 (文本+图像, 都 autoregressive)

7B 模型 (Figure 5):
- 整体 loss: MoT 用 45.5% 的训练步数就达到 dense model 的最终 loss
- 图像 loss: 只需 34.8% 步数 — 这是 MoT 最闪亮的地方
- 文本 loss: MoT 和 MoE-4x 都比 dense 快, MoT 略好
- 验证 loss: MoT 在 55.8% 训练步数时, 验证 loss 已经达到或超过 dense model 训练完的最终验证 loss

**翻译成人话: 用一半多一点的算力, 达到同样效果。**

跨尺度 37M 到 7B 都做了 (Figure 6), 图像模态加速在所有尺度都稳定。MoE-4x 在 7B 时图像加速效果消失 — learned router 在大规模时反而拖后腿, 这是一个非常有意思的 negative result。

### 加上语音 (Chameleon+Speech)

语音数据来自 SpiRit-LM, 用 DinoSR tokenizer (vocab 500, 25Hz, 每个 token 40ms 音频)。

7B 结果 (Figure 8):
- 语音模态: MoT 只需 22.9% 步数匹配 dense baseline — 37.2% FLOPs
- 图像和文本的加速效果没因为加了语音而退化

更 striking 的是 MoE-4x 在语音验证集上 underperform dense (Figure 9)。原因: 语音训练数据分布和验证集 (LL60K, PPL30K) 差异大, learned router 在 OOD 时不稳定。MoT 用确定性的模态路由, 没这个问题。

**这是 MoT 对 MoE 最致命的一击: 不只是快, 还更稳定, 更鲁棒。**

### Transfusion (文本 autoregressive + 图像 diffusion)

这个 setting 是最有趣的, 因为 objective 本身已经 decoupled 了 — 文本用 next-token prediction, 图像用 diffusion loss。

公式 (公式 4):
```
L_Transfusion = L_LM + λ · L_DDP_M
```
- `L_LM`: 语言建模 loss, 每个 token 算一次
- `L_DDP_M`: diffusion loss, 每张图算一次
- `λ = 5`: 平衡系数

Diffusion 的 forward 过程 (Appendix A.1):
```
x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε
```
- `x_0`: 原始图像 (在 VAE latent space 里, 256×256 图 → 256 个 continuous tokens, 每个 8 维)
- `ᾱ_t = ∏_{s=1}^t α_s`: noise schedule 的累积乘积, cosine scheduler 设定
- `ε ~ N(0, I)`: 标准 Gaussian 噪声
- `t`: timestep, 从 0 到 T

7B 结果 (Figure 10):
- 图像 training loss: MoT 用 30% 步数匹配 dense
- CLIP score (文本-图像对齐, 越高越好): MoT 显著更高
- FID (图像质量, 越低越好): MoT 8.14 vs 外部 dense 9.22 (guidance 1.6)
- CIDEr (图像 captioning): MoT 显著更高

最 striking 的比较 (Figure 11): **760M MoT 用 1.4B dense 的一半 FLOPs, 全面超过 1.4B dense**:
- CLIP: 0.214 vs 0.206 (MoT 赢)
- FID: 21.145 vs 24.688 (MoT 赢, 低更好)
- CIDEr: 0.320 vs 0.286 (MoT 赢)

**一半算力的小模型, 打败两倍大的模型, 全方位打败。**

文本模态在 Transfusion 里加速不明显。作者解释: objective decoupling 本身已经给了 dense model 类似 MoT 的好处, 所以 MoT 的边际收益小。这也说明 MoT 主要帮的是"模态在同一套参数里互相干扰"的情况, 如果 objective 已经分开了, 干扰本来就小。

---

## Ablation: 到底哪些部件要分开

Section 3.5 做了 4 个变体 (Figure 14):

1. Dense baseline
2. 只 FFN 分开 (类似 MoMoA)
3. FFN + Q/K/V 分开 (不分 LayerNorm)
4. Full MoT (FFN + Q/K/V + LayerNorm 都分)

**结果**:
- FFN 分开单独就显著提升, 特别是图像模态 — FFN 是 transformer 的 "memory bank", 模态专属 memory 让每个模态学自己的知识
- 加 Q/K/V 分开再提升一截 (Obelisc 上图像省 33.3% FLOPs, 文本省 10%)
- 加 LayerNorm 分开几乎无影响 (Figure 14 两条曲线几乎重合)

直觉: LayerNorm 就是个 scale + shift, 信息容量小, 分开没啥用。FFN 是知识存储, 必须分。Q/K/V 是表示变换的核心, 分开让每个模态投影到适合自己的 attention 子空间。

**实践建议: 工程上可以跳过 LayerNorm untying, 省参数省实现复杂度, 性能几乎不损失。**

---

## LOO 实验: 证明"分开就是好"

Section 4 的 Leave-One-Out 实验是 paper 的 conceptual highlight。

设计 5 个架构 (Figure 15), 都 isoFLOP:
1. Full MoT: 3 个独立 tower (文本/图像/语音)
2. LOO-image: 文本+语音共享 1 个 tower, 图像独立
3. LOO-text: 图像+语音共享, 文本独立
4. LOO-speech: 文本+图像共享, 语音独立
5. Dense: 全部一个 tower

**结果 (Figure 15f-n)**:
- LOO-text 给出最低文本 loss — 文本独立 tower 文本学得最好
- LOO-image 给出最低图像 loss
- LOO-speech 给出最低语音 loss
- 任何两个模态合并到一个 tower, 都会拖累这两个模态

非对称观察: 合并的伤害对语音最大, 图像中等, 文本较小。这暗示"non-reciprocal modality competition" — 不同模态抢参数时的竞争强度不一样。

**这个实验直接证明了: 参数隔离本身就是 MoT 的核心价值, 不需要别的什么 fancy 机制。**

---

## Wall-Clock 时间: 真实世界的胜利

Figure 19 是最 practical 的结果。AWS p4de.24xlarge + A100, 256 GPU:

- 图像 training loss: MoT 用 47.2% wall-clock 时间达到 dense 最终 loss
- 文本 training loss: MoT 用 75.6% wall-clock 时间
- **MoE-4x: 文本没加速, 图像反而慢 1.7x!**

MoE 慢的原因:
1. Top-K selection 的串行操作
2. Token indexing + scattering + gathering expert outputs
3. Load imbalance 导致 GPU underutilization

MoT 的 overhead 主要在 CPU-GPU sync (grouping by modality), 但这个不在 critical path 上, 因为模态标签一次确定后可以 cache。

**翻译成人话: 不只是 FLOP 少, 真实跑起来也快, 因为 MoT 的 sparse pattern 更 GPU-friendly。**

---

## Horizontal Scaling: 越大越爽

Figure 18, 443M 模型, GPU 从 16 到 256:

- 图像验证 loss: GPU=16 时 MoT 需 42.1% 步数匹配 dense; GPU=256 时只需 21.6%
- 文本验证 loss: 75.7% → 50.9%

**Super-linear scaling!** GPU 越多 MoT 优势越大。

原因 (Section 6.1): MoT 的 Parameter-to-FLOPs (PpF) ratio 比 MoE 低得多。分布式训练通信开销取决于 PpF — 参数要 all-gather, gradient 要 all-reduce, 参数越多通信越重。

数学上, 每层新增参数:
- MoE: `Δ_MoE = 3(E-1)D² + ED ≈ 3(E-1)D²` (E 是 expert 数)
- MoT: `Δ_MoT = 7(K-1)D²` (K 是模态数, 通常 2-3)

E 可以几十到几百 (DeepSeek-V3 用 256, Mixtral 用 8), K 通常就 2-3。所以 MoE 的 PpF 随 E 线性增长, MoT 几乎不变。

**在 256 GPU 规模, MoT 比 dense 快一倍多; 在更大规模, 优势会更明显。**

---

## Hybrid: MoT + MoE 两全其美

Section 5 是 proof-of-concept。既然 MoT 图像强, MoE 文本还 OK, 那:
- 文本 tower 的 FFN 换成 MoE-4x
- 图像 tower 保持 MoT 原样

结果 (Figure 16, 17):
- 文本训练 loss: "MoT + Text MoE-4x" 比 MoT 单独还快
- 图像训练 loss: 保持 MoT 速度优势
- 验证: 文本上 hybrid 最好, 图像上保持 MoT 水平

**这验证了一个设计原则: 不同模态可以用不同的 sparse strategy。文本 dense 分布适合 MoE learned routing; 图像/语音 sparse 长序列适合 rule-based 模态分流。**

---

## 为什么 MoT work? 深层直觉

我觉得 MoT 揭示了一个 deep 的事实: **多模态模型的"通用性"被 over-estimate 了**。

当你把文本和图像塞进同一个 dense transformer, 模型其实在内部偷偷把它们分开 — Figure 23 的 PCA 可视化就是证据。既然模型自己想分开, 那 explicit 地分开参数反而更 efficient。

这和 MoE 的 expert specialization 是同一个 phenomenon 的不同侧面 — Mixtral 论文发现不同 expert 自动学了不同的 syntactic function; MoT 里发现模态自动分离。**Learning 一直在找 low-rank / block-diagonal structure**, 给它这个 structure 作为 inductive bias 反而加速学习。

Rule-based routing > learned routing 不是因为它简单, 而是因为模态标签是免费的、确定性的、没有 OOD 问题的强信号。Learned router 试图重新发明这个信号反而引入不稳定。就像你明明有 GPS 还非要凭太阳位置猜方向, 何必呢。

---

## 工程 Takeaways

如果我要 deploy MoT:

1. **Cache modality indices**: 每次迭代前算一次 token 模态索引, 不要每次 layer 都重新算。Algorithm 1 line 3-5 这步 O(n), 算一次就够了。

2. **Group GEMM**: 用 NVIDIA cublas 的 grouped GEMM API 或 MegaBlocks 的 block-sparse GEMM, 一次 GEMM 处理所有模态的不等长 token batch, 消除"sequential processing modality"的 underutilization。

3. **Padding strategy**: 输入主要一个模态时, 可以 pad minor 模态到 major 模态长度, 用一次 batched matmul 换 slightly wasted compute 换 lower latency。

4. **Inference dynamic batching**: 不同 request 模态 mix 不同, 聚合相似模态的 request 一起跑。

5. **FSDP 全分片**: MoT 参数比 dense 多 (K-1)×7D² per layer, 但 activation 不变。FSDP 全分片正合适, PpF 低意味着通信轻。

6. **跳过 LayerNorm untying**: Ablation 证明几乎无影响, 省参数省实现复杂度。

7. **PyTorch 2 compile**: 模态 grouping 是动态 control flow, 需要 dynamic shape 支持, 工程上可能需要 static shape per modality 配合。

---

## 开放问题

1. **模态标签的扩展性**: 未来有 video, 3D, audio music, code, table 等 10+ 模态, MoT 的 (K-1)×7D² 参数增长会显著。是否需要模态 hierarchy (文本→语言, 图像/视频→视觉)?

2. **Continuous modality token**: Transfusion 的图像 token 是 continuous 的, 用 BOI/EOI 分隔给明确标签。更 general 的混合 representation 怎么办?

3. **模态 imbalance**: 99:1 的文本:图像 token ratio 怎么办? 图像 tower 会严重 under-trained 吗? Section 6.1 提到 MoT 的 GPU underutilization 在 imbalance 时更严重, 但实验上没充分 explore。

4. **Fine-tune 时新增模态**: 加一个新模态需要新加一个 tower, 能否 share 已有 tower 的 weights 来 warm-start?

5. **Cross-modal knowledge transfer**: paper 强调了 decoupling 的好处, 但没量化 cross-modal transfer 受益多少。完全隔离可能损失某些 transfer — 文本 tower 学的 "red car" 概念对图像 tower 有帮助吗? LOO 实验显示合并会 hurt, 但完全隔离的 transfer loss 没量化。

---

## 一句话总结

**MoT 证明了一个朴素但深刻的事: 当数据天然有类别结构, 让模型在参数空间显式尊重这个结构, 比强迫学一个"统一表示"更高效。Dense model 内部本来就在偷偷做这个事, MoT 只是把它变成了 inductive bias。** Rule-based routing > learned routing 不是因为它简单, 是因为模态标签是免费的、确定的、没有 OOD 问题的强信号。

在多模态 foundation model 这个方向, MoT 是一个 strong 的 architectural baseline — isoFLOP, drop-in 替换 dense transformer, 节省 40-60% 训练成本, 所有尺度验证, wall-clock 优势, 还能和 MoE hybrid — 这是一个很漂亮的工程 + 科学工作。

---

**参考链接**:
- MoT paper: https://openreview.net/forum?id=Nu6N69i8SB
- Chameleon: https://arxiv.org/abs/2405.09818
- Transfusion: https://arxiv.org/abs/2408.11039
- SpiRit-LM: https://arxiv.org/abs/2402.05755
- Expert Choice MoE: https://arxiv.org/abs/2202.09368
- MoMoA: https://arxiv.org/abs/2407.21770
- CogVLM: https://arxiv.org/abs/2311.03079
- Playground v3: https://arxiv.org/abs/2409.10695
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- Mixtral: https://arxiv.org/abs/2401.04088
- OLMoE: https://arxiv.org/abs/2409.02060
- MegaBlocks: https://arxiv.org/abs/2211.15841
- PyTorch FSDP: https://arxiv.org/abs/2306.09733
- Stable Diffusion 3: https://arxiv.org/abs/2403.03206
- Latent Diffusion: https://arxiv.org/abs/2112.10752
- VQGAN (Make-A-Scene): https://arxiv.org/abs/2203.13131
- DinoSR: https://arxiv.org/abs/2305.10005
- Branch-Train-Mix: https://arxiv.org/abs/2403.07816
- Lory (differentiable MoE): https://openreview.net/forum?id=LKEJPySnlt

---

# Mixture-of-Transformers (MoT) 深度讲解

## 1. 核心动机与直觉

这篇 paper 想解决一个非常实际的问题: **multi-modal foundation model 的 pretraining 计算成本太高**。Chameleon 训了 9.2T tokens (包括 image tokens) 才能匹配 LLaMA2 用 2T text tokens 达到的 text 性能 — compute 翻了 4 倍多。这种 scaling 显然不可持续。

作者从两个观察出发建立 intuition:

**观察 1: Modality 的 training dynamics 会冲突。** Figure 15 显示 dense transformer 里, 不同 modality 的 loss 曲线会"打架", 同一个 layer 的 weights 要同时服务三种非常不同的统计分布。

**观察 2: 即使没有任何 modality-specific prior, 模型内部 features 自然按 modality 聚类。** Figure 2 和 Appendix Figure 23 是非常 striking 的可视化 — 在 Chameleon+Speech 7B dense model 的不同 layer (1, 5, 17, 32) 和不同 training checkpoint (4%, 24%, 50%, 100%) 做 PCA, text / speech / image token 的 activations 始终占据 feature space 的不同 region。这是 model **自己学到的** clustering, 而 paper 把这个现象当作 "modality 本质上需要不同 processing" 的证据。

这两个观察合起来给了一个非常自然的 hypothesis: 既然 dense model 自己都把不同 modality 推到不同的 feature subspace, 那 explicit 地给每个 modality 分配一套独立的 non-embedding parameters, 应该能加速学习 — 这就是 MoT。

参考链接:
- Paper OpenReview: https://openreview.net/forum?id=Nu6N69i8SB
- Chameleon: https://arxiv.org/abs/2405.09818
- Transfusion: https://arxiv.org/abs/2408.11039

---

## 2. MoT 架构详解

### 2.1 与 dense / MoE 的对比

Dense transformer (公式 1) 每层所有 token 共享一套 parameters θ_attn, θ_ffn:

```
a = Attn(x, θ_attn)
h = x + LayerNorm_attn(a)
output = h + LayerNorm_ffn(FFN(h, θ_ffn))
```

MoE 在 FFN 上引入 E 个 expert + learned router, 每个 token 激活 top-k experts, 训练时 expert 和 router 都 under-trained 容易不稳定, 还需要 load balancing loss。

MoT 的设计哲学: **routing 用 modality label (一个确定的、免费的信号), 而不是 learned router**。这样避免了 MoE 的 bi-level optimization 不稳定性, 同时 sparse activation 的所有好处都保留 — 每个 token 只激活自己 modality 那一套 parameters。

### 2.2 公式 2 和算法 1 的逐行拆解

MoT layer 的核心 (公式 2):

```
a = GlobalAttn(x, {θ_attn^m}_{m ∈ {text, image, speech}})
h_i = x_i + LayerNorm_attn^{m_i}(a_i)
output_i = h_i + LayerNorm_ffn^{m_i}(FFN(h_i, θ_ffn^{m_i}))
```

变量解释:
- `x = (x_1, ..., x_n)`: 输入 token 序列, 每个 x_i ∈ R^d (d 是 hidden dimension)
- `m_i ∈ {text, image, speech}`: token i 的 modality label (由 tokenizer 决定, 训练前已知)
- `θ_attn^m` 和 `θ_ffn^m`: modality m 独立的 attention 和 FFN 参数
- `LayerNorm_attn^{m_i}` 和 `LayerNorm_ffn^{m_i}`: modality-specific LayerNorm
- `a_i`: token i 在 global attention 之后的输出
- `h_i`: residual + modality-specific LayerNorm 之后

关键 trick: **global self-attention 跨 modality 共享**。这一步是跨 modality 的信息流 — text token 可以 attend 到 image token, 反之亦然。这保留了 cross-modal interaction, 同时所有 modality-specific 的"非线性变换" (Q/K/V projection, output projection, FFN) 都是 modality-private 的。

公式 3 给出 GlobalAttn 的具体形式:

```
GlobalAttn(x, {θ_attn^m}) = softmax(QK^T / sqrt(d_k)) V  W_O^{m_i}
Q_i = x_i W_Q^{m_i},  K_i = x_i W_K^{m_i},  V_i = x_i W_V^{m_i}
```

变量解释:
- `W_Q^{m_i}, W_K^{m_i}, W_V^{m_i} ∈ R^{d × d_k}`: token i 的 modality-specific 投影矩阵
- `W_O^{m_i} ∈ R^{d × d}`: token i 的 modality-specific output projection
- `d_k`: key/query 的 per-head dimension
- `Q, K, V`: 把所有 modality 的 Q_m / K_m / V_m 重新 concat 回原 sequence 顺序后得到的完整矩阵
- `softmax(QK^T / sqrt(d_k))`: 标准 scaled dot-product attention, 在**所有 token 上**做 (这就是 "global" 的含义)

注意一个非常微妙的设计: Q, K, V projection 是 modality-specific 的, 但 attention score 计算时跨 modality。这意味着 text token 的 query 可以和 image token 的 key 算 dot product, 得到 cross-modal attention weight — 这是 cross-modal fusion 的关键。然后用 modality-specific 的 W_O 投影回 modality-private 的 representation space。

### 2.3 算法 1 的流程图化

```
Input: x_1, ..., x_n  with modality labels m_1, ..., m_n

# Step 1: Group tokens by modality (line 3-7)
for m in {text, image, speech}:
    I_m = {i : m_i = m}             # 该 modality 的 token 索引集合
    X_m = {x_i : i ∈ I_m}           # 取出这些 tokens
    Q_m = X_m @ W_Q^m               # modality-specific 投影 (只对该 modality token 做)
    K_m = X_m @ W_K^m
    V_m = X_m @ W_V^m

# Step 2: Restore sequence order (line 8)
Q = concat(Q_m for m in M) [按原顺序放回]
K = concat(K_m for m in M)
V = concat(V_m for m in M)

# Step 3: Global self-attention (line 9)
A = softmax(Q @ K^T / sqrt(d_k)) @ V    # 所有 token 互相 attend

# Step 4: Modality-specific output + FFN (line 10-15)
for m in M:
    O_m = A[I_m] @ W_O^m                # 取出该 modality 的 attention 输出, 做输出投影
    H_m = X_m + LayerNorm_attn^m(O_m)   # residual + modality-specific LN
    F_m = FFN_m(H_m)                    # modality-specific FFN
    Y_m = H_m + LayerNorm_ffn^m(F_m)    # residual + modality-specific LN
```

这里有一个关键 insight: 虽然 parameters 是 sparse 的 (每个 token 只激活一套), 但 FLOPs 和 dense model 完全一样 — 因为总 token 数没变, 每个 token 都被处理一次。所以 MoT 是 **isoFLOP** 的 sparse architecture, 它的效率优势来自 sparse 参数让 optimization 更高效, 而不是单纯减少计算量。

---

## 3. FLOP 控制与参数分析

### 3.1 为什么是 isoFLOP 而不是更少 FLOP

这一点容易误解。MoT 的 FLOPs 和 dense baseline 一样, 因为:
- 每个 token 仍然做一次 Q/K/V projection (只是用 modality-specific 的 W)
- Global attention 还是 n × n 的
- 每个 token 还过一次 FFN (只是 modality-specific 的 FFN)

但是 MoT 的 **总参数量** 比 dense 多了 (K-1) × (|ATTN| + |FFN|), 因为有 K 套独立的 attention 和 FFN。每个 token 只激活其中一套。

对比 MoE-4x: 总参数多了 (E-1) × |FFN| + |ROUTER|, 每个 token 也只激活 top-1 expert。

### 3.2 ML Systems 角度: PpF ratio

Section 6.1 给了一个非常重要的 systems 分析。在分布式训练里, Parameter-to-FLOPs (PpF) ratio 决定通信开销 — PpF 越高, 越多时间花在 all-gather / all-reduce 上。

假设 transformer hidden dim = embedding dim = D, 使用 SwiGLU FFN:
- `|FFN| = 3D²` (SwiGLU 有三个矩阵: gate, up, down)
- `|ATTN| = 4D²` (Q, K, V, O)

**MoE 每层新增参数**:
```
Δ_MoE = (E-1) × |FFN| + |ROUTER|
      = 3(E-1)D² + ED
      ≈ 3(E-1)D²
```

**MoT 每层新增参数** (K 个 modality):
```
Δ_MoT = (K-1) × (|ATTN| + |FFN|)
      = (K-1) × 7D²
      = 7(K-1)D²
```

关键观察: E (expert 数) 通常从几十到几百 (DeepSeek-V3 用了 256 个 expert, OLMoE 64, Mixtral 8), 而 K (modality 数) 通常就是 2-3 个。所以 PpF_MoT 通常比 PpF_MoE 小得多, 在大规模分布式训练里通信开销更小, throughput 更高。

数值例子: K=3, E=4
- Δ_MoT = 7 × 2 × D² = 14 D²
- Δ_MoE = 3 × 3 × D² + 4D ≈ 9 D²

K=3, E=8 (DeepSeek 风格)
- Δ_MoT = 14 D²
- Δ_MoE = 21 D²

K=3, E=64 (OLMoE 风格)
- Δ_MoT = 14 D²
- Δ_MoE = 189 D²

所以 MoE 在 E 大的时候 PpF 飙升, 而 MoT 几乎线性保持低 PpF。

参考: DeepSeek-V3 https://arxiv.org/abs/2412.19437, OLMoE https://arxiv.org/abs/2409.02060, Mixtral https://arxiv.org/abs/2401.04088

---

## 4. 三个实验 Setting 详解

### 4.1 Chameleon Setting: 纯 autoregressive

- Text 和 image 都用 autoregressive next-token prediction
- Image 用预训练的 VQGAN tokenizer (Make-A-Scene 风格) 编码成 1,024 个 discrete tokens
- 单一 loss function, 单一 objective
- 对比 dense baseline 和 MoE-4x

**7B 结果 (Figure 5)**:
- 总 training loss: MoT 用 45.5% 的 steps 就达到 dense model 最终 loss (Figure 5b)
- Image modality: 只需 34.8% steps — 这是 MoT 最 shines 的地方
- Text modality: MoT 和 MoE-4x 都比 dense 快, MoT 略好
- Validation loss (Obelisc, COCO, Flickr, SSTK): MoT 在 55.8% training step 时就能达到或超过 dense model 最终的 validation loss

跨 scale (Figure 6): 37M, 94M, 443M, 1.5B, 7B 都做了一遍, image modality 加速效果在所有 scale 都稳定。MoE-4x 在 7B 时 image 加速效果消失 — 这是一个非常关键的 negative result, 说明 learned routing 在大 scale 时反而不如 rule-based routing。

Table 1 给了 model spec:
| Model | Hidden | Layers | Heads | Seq | Steps | Tokens |
|-------|--------|--------|-------|-----|-------|--------|
| 37M   | 256    | 4      | 8     | 4096| 160k  | 0.252T |
| 94M   | 512    | 8      | 8     | 4096| 160k  | 0.168T |
| 443M  | 1024   | 24     | 16    | 4096| 160k  | 0.252T |
| 1.5B  | 2048   | 24     | 16    | 4096| 120k  | 0.252T |
| 7B    | 4096   | 32     | 32    | 4096| 120k  | 0.377T |

### 4.2 Chameleon+Speech: 加入第三 modality

- 用 SpiRit-LM 的 speech data (Table 2): People's Speech, Voxpopuli, LibriLight, MLS, Spotify
- Speech tokenizer 是 DinoSR variant, vocab size 500, 25Hz (每 token = 40ms audio)
- Speech data 和 Chameleon data 1:6 混合
- 三个 modality 都用 autoregressive

**7B 结果 (Figure 8)**:
- Speech modality: MoT 只需 22.9% steps 匹配 dense baseline, 即只用 37.2% FLOPs
- Image 和 text modality 的加速效果保持 (没有因为加了 speech 而退化)
- Validation loss 在 LL60K 和 PPL30K 上 MoT 也优于 dense

跨 scale (Figure 9): 443M, 880M, 1.5B 都做, speech 加速比 15.1%-33.6% steps。MoE-4x 在 speech validation 上 underperform dense, 这是非常关键的 negative result — MoE 的 learned router 在 speech 这种 OOD 多的 setting 下不稳定。

### 4.3 Transfusion: 多 objective training

这是最 interesting 的 setting, 因为 objective 本身已经 decoupled:
- Text: autoregressive next-token prediction (L_LM)
- Image: diffusion loss (L_DDP_M)

Diffusion loss (Appendix A.1):
```
q(x_t | x_{t-1}) = N(x_t; sqrt(α_t) x_{t-1}, (1-α_t) I)
x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε
```
- `α_t ∈ (0,1)`: timestep t 的 noise schedule
- `ᾱ_t = ∏_{s=1}^t α_s`: cumulative product
- `ε ~ N(0, I)`: 标准 Gaussian noise
- Cosine scheduler 设定 α_t

反向 denoising:
```
x_{t-1} = (1/sqrt(α_t)) × (x_t - (1-α_t)/sqrt(1-ᾱ_t) × ε_θ(x_t, t, c)) + σ_t z
```
- `ε_θ`: 神经网络预测的 noise
- `c`: 条件 (text prompt)
- `σ_t`: reverse step noise std
- `z ~ N(0, I)`: 辅助随机性

Transfusion 总 loss (公式 4):
```
L_Transfusion = L_LM + λ × L_DDP_M
```
λ = 5。

Image 用 VAE encode 成 latent, 每 8×8 patch 一个 8-d vector, 一张 256×256 图变成 256 个 continuous tokens。

**7B 结果 (Figure 10)**:
- Image training loss: MoT 只用 30% steps 匹配 dense
- Image validation loss: MoT 在 ~1/3 FLOPs 时达到 dense baseline 水平
- CLIP score: MoT 显著高于 dense
- FID: MoT 显著低于 dense (8.14 vs 9.22 at guidance 1.6 vs 外部 dense model)
- CIDEr (image captioning): MoT 显著高于 dense

**760M vs 1.4B (Figure 11)** — 这是 paper 里最 striking 的比较:
- 760M MoT 用 1.4B dense 的一半 FLOPs
- CLIP: 0.214 vs 0.206 (MoT 更高, 越高越好)
- FID: 21.145 vs 24.688 (MoT 更低, 越低越好)
- CIDEr: 0.320 vs 0.286 (MoT 更高)
- Image training loss: MoT 更低

也就是说, **一半 FLOPs 的小 MoT 全面超过两倍大的 dense model**。

**Text modality 在 Transfusion 里加速不明显** — 作者解释这是因为 objective decoupling 本身已经给了 dense model 类似 MoT 的好处, 所以 MoT 的边际收益小。

---

## 5. Ablation: 哪些 component 该 untying

Section 3.5 + Figure 14 是一个非常重要的 ablation, 探究 modality-specific decoupling 在哪些 component 最有效:

1. Dense baseline (无 untying)
2. FFN-only untying (类似 MoMo, Lin et al. 2024)
3. FFN + Q/K/V untying (无 LayerNorm untying)
4. Full MoT (FFN + Q/K/V + LayerNorm 都 untying)

**Result**:
- FFN untying 单独就显著提升, 特别是 image modality (FFN 占 transformer 大部分 FLOPs, 也是 memory component)
- 加 Q/K/V untying 再提升一截 (Obelisc 上 image 省 33.3% FLOPs, text 省 10%)
- 加 LayerNorm untying 几乎无影响 (Figure 14 中两条曲线几乎重合)

直觉: LayerNorm 只是一个 affine 变换 (scale + shift), 它的能力有限, modality-specific 的 scale/shift 带来的信息增益很小。而 FFN 是 transformer 的 "memory bank", modality-specific 的 memory 让每个 modality 学自己的知识。Q/K/V 是 representation 变换的核心, modality-specific 让每个 modality 投影到适合自己 attention 的子空间。

参考: MoMoA / Lin et al. https://arxiv.org/abs/2407.21770

---

## 6. Leave-One-Out (LOO) 分析

Section 4 是 paper 的一个 conceptual highlight。问题: **modality-specific tower 之间的好处, 究竟来自 parameter 隔离, 还是来自 global attention 的 cross-modal fusion?**

设计 5 个 architecture (Figure 15), 都是 isoFLOP:

1. Full MoT: 3 个 tower (text, image, speech)
2. LOO-image: text+speech 共享 1 个 tower, image 独立
3. LOO-text: image+speech 共享, text 独立
4. LOO-speech: text+image 共享, speech 独立
5. Dense: 全部一个 tower

**Result (Figure 15f-n)**:
- LOO-text 给出最低的 text loss — 隔离 text tower 让 text 学得最好
- LOO-image 给出最低的 image loss — 同理
- LOO-speech 给出最低的 speech loss
- 任何两个 modality 合并到一个 tower, 都会 degrade 这两个 modality 的性能
- 一个非对称观察: 合并的伤害对 speech 最大, 对 image 中等, 对 text 较小 — 这暗示了"non-reciprocal modality competition"

这个 ablation 直接证明了: **modality-specific parameter allocation 本身就是 MoT 的核心价值来源**, global attention 只是 cross-modal fusion 的机制, 不需要 parameters 共享。

---

## 7. Hybrid: MoT + MoE-4x (Best of Both Worlds)

Section 5 是一个 proof-of-concept。Idea: 既然 MoT 在 image modality 最强, 而 MoE 在 text modality 表现 OK, 那就 hybrid:
- Text transformer tower 内部的 FFN 替换成 MoE-4x
- Image transformer tower 保持 MoT 原样

**Chameleon 373M (Figure 16)**:
- Text training loss: "MoT + Text MoE-4x" 比 MoT 单独还快
- Image training loss: 保持 MoT 的速度优势
- Validation: text 上"MoT + Text MoE-4x" 在多个 dataset 上都是最好

**Transfusion 760M (Figure 17)**: 同样的 pattern。

这个实验验证了一个非常有用的设计原则: **不同 modality 可以用不同的 sparse strategy**。Text 是 dense 分布的 token, 适合 MoE 的 learned routing; image/speech 是 sparse 出现的长 sequence, 适合 rule-based 的 modality 分流。

---

## 8. Wall-Clock Time 和 Horizontal Scaling

### 8.1 Wall-clock (Figure 19)

这是 paper 里最 practical 的 section。在 AWS p4de.24xlarge + NVIDIA A100 上, 256 GPU 训练:

- Image training loss: MoT 用 47.2% wall-clock 时间达到 dense baseline 的最终 loss
- Text training loss: MoT 用 75.6% wall-clock 时间
- MoE-4x: text 没有 speed advantage, image 反而慢 1.7x!

MoE-4x 在 wall-clock 上变慢的原因是它的 overhead:
1. Top-K selection 的 sequential operations
2. Token indexing + scattering + adding expert outputs
3. Load imbalance 导致 GPU underutilization

MoT 的 overhead 主要在 CPU-GPU sync (grouping by modality), 但作者发现这个不在 critical path 上, 因为 modality label 一次确定后可以 cache sequence indices per iteration。

### 8.2 Horizontal scaling (Figure 18)

443M model, GPU 数从 16 到 256 (global batch 和总 token 数等比例增加):

- Image validation loss: GPU=16 时 MoT 需要 42.1% steps 匹配 dense; GPU=256 时只需 21.6% steps
- Text validation loss: 75.7% → 50.9% steps

这是一个 **super-linear** scaling — GPU 越多, MoT 优势越大。原因是 MoT 的 PpF 低, 通信开销小, 所以在大规模分布式训练里 throughput 优势更明显。这非常符合 Karpathy 你之前提到过的 "scale 才是 LLM 的真正 bottleneck" 的直觉。

参考: PyTorch FSDP https://arxiv.org/abs/2306.09733, MegaBlocks https://arxiv.org/abs/2211.15841

---

## 9. 关于 MoE-4x 的讨论和局限

Paper 里有一个诚实的 caveat (Section 3.2.1): MoE-4x baseline 用的是 Expert Choice (EC) routing (Zhou et al. 2022), 它在 training 时 expert 选择 top-k tokens, 不违反 causality... 等等, 实际上 EC **违反** causality 因为它看到整个 batch 的所有 token 包括未来的。所以 inference 时不能直接用 EC。Paper 选择在 validation perplexity 评估时还是用 EC, 这可能:
1. Overestimate MoE-4x 的 performance (因为 future token 信息泄露)
2. Underestimate MoE-4x 在 OOD evaluation data 上的 performance (因为 router 见过的分布和 eval 分布不同)

这个 caveat 让 MoT 对 MoE 的胜利看起来更 striking — 因为 MoT 用的是 deterministic rule-based routing, 没有 information leakage, 没有 OOD 问题。

参考: Expert Choice https://arxiv.org/abs/2202.09368, Lory (fully differentiable MoE) https://openreview.net/forum?id=LKEJPySnlt

---

## 10. 直觉总结与开放问题

### 10.1 为什么 MoT work? 我 (Karpathy 视角) 的直觉

我觉得 MoT 的成功本质上揭示了一个 deep 的事实: **multi-modal model 的"通用性"是被 over-estimated 的**。当你把 text 和 image 强行塞进同一个 dense transformer, 模型其实在内部偷偷把它们分开 — Figure 23 的 PCA 可视化就是证据。既然模型自己想分开, 那 explicit 地分开 parameters 反而是更 efficient 的 inductive bias。

这和 "MoE 的 expert specialization 现象" 是同一个 phenomenon 的不同侧 — Mixtral 论文里发现不同 expert 自动学了不同的 syntactic function; MoT 里发现 modality 自动分离。**Learning 一直在找 low-rank / block-diagonal structure**, 给它这个 structure (作为 inductive bias) 反而加速学习。

### 10.2 一些 paper 没完全解决的开放问题

1. **Modality label 的扩展性**: 如果未来有 video, 3D, audio music, code, table 等 10+ 个 modality, MoT 的 (K-1)×7D² 参数增长会变得显著。是否需要 modality hierarchy (text→language, image/video→visual)?

2. **Continuous modality token (Transfusion) 下的 modality label**: paper 假设每个 token 的 modality 是 well-defined。但是混合 representation (比如 image patch + text token 在同一个 latent space) 怎么办? Transfusion 的 image tokens 是 continuous 的, 是用 special token (BOI/EOI) 分隔, 这给了明确的 modality label, 但是更 general 的 setting 呢?

3. **Modality imbalance**: 1:1 的 text:image token ratio 在实验里是平衡的。如果 99:1 (text-heavy 预训练再加一点 image) 怎么办? image tower 会严重 under-trained 吗? Section 6.1 提到 MoT 的 GPU underutilization 在 imbalance 时更严重, 但实验上没充分 explore。

4. **Pre-training 之后 fine-tuning 时 modality 的新增**: 如果 fine-tune 时加一个新 modality (比如 video), 需要新加一个 tower。能否 share 一部分已有 tower 的 weights 来 warm-start?

5. **Modality 之间的 knowledge transfer**: paper 强调了 decoupling 的好处, 但没量化 cross-modal knowledge transfer 受益多少。比如 image tower 学到的 visual concept 是否对 text tower 理解 "red car" 有帮助? LOO 实验显示合并会 hurt, 但完全隔离可能也损失了某些 transfer。

### 10.3 与其他工作的关系

- **vs. CogVLM (Wang et al. 2023)**: CogVLM 只在 attention 上 visual expert, text 用预训练 LLM frozen。MoT 全部 non-embedding 都 decouple, 而且 from scratch 训练。
- **vs. Playground v3 (Liu et al. 2024)**: PGv3 用 Llama3-8B frozen 当 text backbone + DiT-style image transformer, global self-attention 跨 modality。架构上和 MoT 很像, 但 PGv3 是"利用 frozen LLM" 路线, MoT 是"from scratch sparse pre-training"路线。
- **vs. MoMoA (Lin et al. 2024)**: MoMoA 只在 FFN 上做 modality-aware expert, MoT 推广到 attention + LayerNorm。
- **vs. Branch-Train-Mix (Sukhbaatar et al. 2024)**: BTM 训独立 branch 再 MoE 合并, 是 post-hoc 的; MoT 是 pretraining 时的 end-to-end sparse architecture。

参考链接:
- CogVLM: https://arxiv.org/abs/2311.03079
- Playground v3: https://arxiv.org/abs/2409.10695
- Branch-Train-Mix: https://arxiv.org/abs/2403.07816

---

## 11. Engineering Takeaways

如果我要 reproduce 或 deploy MoT, 关键工程点:

1. **Modality grouping**: 每次迭代前 cache token indices per modality, 不要每次都重新 mask。Algorithm 1 line 3-5 这个操作其实只需要 O(n) 一次。

2. **Group GEMM**: 用 NVIDIA 的 grouped GEMM API 或 MegaBlocks 的 block-sparse GEMM, 一次 GEMM 处理所有 modality 的不等长 token batch。这能消除"sequential processing modality"的 underutilization。

3. **Padding strategy**: 当 input 主要一个 modality 时, 可以 pad minor modality 到 major modality 的长度, 用一次 batched matmul 换 slightly wasted compute 换 lower latency。

4. **Inference 时的 dynamic batching**: 不同 request 的 modality mix 不同, 聚合相似 modality 的 requests 一起跑, 提高 GPU utilization。

5. **FSDP sharding**: MoT 的参数比 dense 多 (K-1)×7D² per layer, 但 activation 不变。FSDP 全分片正好合适, 因为通信开销主要在 parameter shard, 而 PpF 低意味着通信更轻。

6. **PyTorch 2 Compiler**: paper 提到用 torch.compile 优化, 对于 MoT 这种动态 control flow (modality grouping) 需要 dynamic shape 支持, 实际工程上可能需要 static shape per modality 配合。

---

## 12. 一句话直觉

**MoT 揭示了一个朴素但深刻的事实: 当你的数据天然有类别结构 (modality), 让模型在参数空间显式地尊重这个结构, 比强迫它学一个"统一表示"更高效 — dense model 内部本来就在偷偷做这个事, MoT 只是把它变成了 inductive bias。** Rule-based routing > learned routing 不是因为它简单, 而是因为 modality label 是免费的、确定性的、没有 OOD 问题的强信号 — learned router 试图重新发明这个信号反而引入不稳定。

这是一个 "in-place replace dense transformer for multi-modal" 的工作, 直接 drop-in 替换, isoFLOP, 节省 40-60% 训练成本, 在所有 scale 验证, 还有 wall-clock 优势, 还有和 MoE 互补的 hybrid 形态 — 在 multi-modal foundation model 这个方向, 这是一个非常 strong 的 architectural baseline。

---

**主要参考链接**:
- Paper OpenReview: https://openreview.net/forum?id=Nu6N69i8SB
- Chameleon: https://arxiv.org/abs/2405.09818
- Transfusion: https://arxiv.org/abs/2408.11039
- SpiRit-LM: https://arxiv.org/abs/2402.05755
- Expert Choice MoE: https://arxiv.org/abs/2202.09368
- MoMoA: https://arxiv.org/abs/2407.21770
- CogVLM: https://arxiv.org/abs/2311.03079
- Playground v3: https://arxiv.org/abs/2409.10695
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- Mixtral: https://arxiv.org/abs/2401.04088
- OLMoE: https://arxiv.org/abs/2409.02060
- MegaBlocks: https://arxiv.org/abs/2211.15841
- PyTorch FSDP: https://arxiv.org/abs/2306.09733
- Lory (differentiable MoE): https://openreview.net/forum?id=LKEJPySnlt
- Branch-Train-Mix: https://arxiv.org/abs/2403.07816
- Stable Diffusion 3 (cosine scheduler, rectified flow): https://arxiv.org/abs/2403.03206
- Latent Diffusion Models: https://arxiv.org/abs/2112.10752
- VQGAN (Make-A-Scene): https://arxiv.org/abs/2203.13131
- DinoSR (speech tokenizer): https://arxiv.org/abs/2305.10005
