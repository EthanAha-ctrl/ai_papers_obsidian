---
source_pdf: MiniMax-01.pdf
paper_sha256: e5fe73cf990aa81f8e2fa43f2a5b650418308b0a07e0c5e1a994287afa77c0f2
processed_at: '2026-08-05T18:42:02-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MiniMax-01 用人话讲

好，我换个方式，像跟同事白板前聊天那样讲。

---

## 这篇paper到底干了啥

一句话：**他们把linear attention第一次真正搞到了production scale**，456B参数的MoE，context window能到4M tokens，性能跟GPT-4o打平。

这件事为什么重要？因为过去几年大家都觉得"linear attention是个好idea但用不了"。从小模型实验到大模型部署，中间有巨大的engineering gap。这篇paper本质是把这个gap填上了，顺便证明了hybrid架构（linear + 少量softmax）在retrieval上能超越纯softmax——这个结论挺反直觉的。

链接：https://github.com/MiniMax-AI

---

## 为什么softmax attention走到头了

softmax attention的复杂度是$O(n^2 d)$，$n$是sequence length，$d$是head dimension。当$n$从8K涨到1M，计算量涨1万倍。硬件跟不上。

FlashAttention（https://arxiv.org/abs/2205.14135）把memory读写优化了，但计算量本身没变。所以现在主流模型的context window基本卡在128K-256K（GPT-4o 128K，Claude 200K，Gemini 1.5 Pro 2M但性能下降明显）。

要突破这个瓶颈，得换attention机制。

---

## Linear attention的核心idea

原始attention：
$$O = \text{Softmax}(QK^T)V$$

这里的$QK^T$是$n \times n$矩阵，这是quadratic的根源。

Linear attention的观察：如果去掉softmax，就能用结合律换乘法顺序：
$$Q(K^T V) \text{ instead of } (QK^T)V$$

$K^T V$是$d \times d$矩阵，跟$n$无关。一旦算出这个矩阵，每个新query只需要做$Q \times KV$，复杂度$O(d^2)$，跟sequence length无关。

变量解释：
- $Q, K, V$：query, key, value矩阵，shape都是$\mathbb{R}^{n \times d}$
- $n$：sequence长度
- $d$：feature dimension（每个head的维度）
- $K^T V \in \mathbb{R}^{d \times d}$：累积的"memory"，注意它跟$n$无关

这个idea 2016年就有人提了（de Brébisson和Vincent，https://arxiv.org/abs/1609.05866），但9年没人用在大模型上。原因：**causal masking**。

---

## Causal masking的麻烦

language model是autoregressive的，token $t$只能看到token $1$到$t$。这在softmax attention下就是加个下三角mask $M$：
$$O = [(QK^T) \odot M]V$$

在linear attention下，这个mask变成cumsum（累积和）：
$$kv_t = kv_{t-1} + k_t v_t^T, \quad o_t = q_t^T kv_t$$

变量：
- $kv_t \in \mathbb{R}^{d \times d}$：到time step $t$为止累积的KV state
- $k_t, v_t \in \mathbb{R}^d$：第$t$个token的key和value
- $o_t$：第$t$个token的output

问题：$kv_t$依赖$kv_{t-1}$，**有递推关系，没法并行**。GPU最怕串行。

这就是为啥9年没人用。小模型实验可以忍，大模型训练几千张卡，串行就是灾难。

---

## Lightning Attention怎么解决并行问题

核心trick：**tiling**，把sequence切成block，block内用left product（可以并行），block间用right product（累积状态）。

设block size $B$，把$Q, K, V$切成$T = n/B$个block。对第$t$个block：

**Intra-block**（block内，用left product，标准attention计算）:
$$O_{\text{intra}} = [(Q_t K_t^T) \odot M] V_t$$

这里$Q_t, K_t, V_t \in \mathbb{R}^{B \times d}$，$M$是$B \times B$的causal mask。这个计算block内所有token互相attend，$B$很小（比如256），所以即使$O(B^2 d)$也OK，而且block内完全可以并行。

**Inter-block**（block间，用right product，累积状态）:
$$O_{\text{inter}} = Q_t \cdot KV_{\text{prev}}$$

这里$KV_{\text{prev}} \in \mathbb{R}^{d \times d}$是之前所有block累积的state。

**State更新**:
$$KV_{\text{new}} = KV_{\text{prev}} + K_t^T V_t$$

变量：
- $KV_{\text{prev}}$：到第$t-1$个block为止的累积KV state
- $K_t^T V_t$：第$t$个block贡献的新KV
- $Q_t \cdot KV_{\text{prev}}$：当前block的query对历史memory的attend

关键insight：intra-block是$O(B^2 d)$，inter-block是$O(Bd^2)$。总复杂度$O(nd^2 + nBd)$，当$d$和$B$固定时，跟$n$线性。而softmax attention是$O(n^2 d)$。

Algorithm 1的伪代码（IO-aware版，类似FlashAttention的tiling）：

```
Initialize KV = 0 ∈ R^{d×d}  (在HBM里)
for t = 1 to T:
    Load Q_t, K_t, V_t from HBM to SRAM  (一次IO)
    O_intra = [(Q_t K_t^T) ⊙ M] V_t      (on-chip计算)
    O_inter = Q_t @ KV                    (on-chip计算)
    KV = KV + K_t^T @ V_t                (on-chip更新)
    Write O_t = O_intra + O_inter to HBM
```

SRAM里计算的都不涉及跨block依赖，block内可以并行。跨block的只有$KV$的更新，这是$O(d^2)$的addition，很便宜。

这就是lightning attention（Qin et al.，https://arxiv.org/abs/2401.04658）的核心。原paper是作者团队成员写的。

---

## 为什么不能纯用linear attention

paper里做了实验（Figure 7，Table 2.2.2.3）：纯lightning attention在大多数downstream task上跟softmax打平，**除了NIAH（Needle in A Haystack）**——retrieval任务上纯linear attention明显弱。

这很合理。linear attention的memory是$d \times d$矩阵，固定大小。往里塞太多信息会"忘"掉早期的内容。softmax attention每次都重新attend所有历史token，理论上不会忘。

所以作者搞了hybrid：**7层lightning + 1层softmax，循环80层**。

这个比例怎么选的？Table 3的ablation：
- Hybrid-cosformer2: NIAH 43.6
- Hybrid-hgrn2: NIAH 91.8
- Hybrid-lightning: NIAH 95.7（最好）
- Hybrid-window (window=1024): NIAH 53.9

Lightning attention在NIAH上碾压sliding window attention，且speed相当。所以选了lightning而不是其他linear变体。

---

## 最反直觉的结论：hybrid比纯softmax在retrieval上更强

这是Section 2.2.4的精华。

作者把softmax attention改写成linear RNN form来对比"capacity"：

Softmax attention的RNN form：
$$s_t^j = s_t^{j-1} + \exp(q_t k_j^T / \sqrt{d})$$
$$o_t^j = (s_t^{j-1}/s_t^j) o_t^{j-1} + (1 - s_t^{j-1}/s_t^j) v_j$$

变量：
- $s_t^j$：到第$j$个token为止的累积分母（用于softmax归一化）
- $o_t^j$：到第$j$个token为止的累积output
- $q_t, k_j, v_j$：第$t$个query，第$j$个key和value
- $o_t = o_t^t$：最终output是累积到$t$为止

注意：softmax的RNN state是$O(d)$大小（标量$s_t$加向量$o_t$）。

Lightning attention的RNN form：
$$kv_j = kv_{j-1} + k_j v_j^T, \quad o_j = q_j^T kv_j$$

变量：
- $kv_j \in \mathbb{R}^{d \times d}$：累积KV state
- $k_j v_j^T$：第$j$个token的outer product贡献
- $q_j^T kv_j$：query对累积memory的attend

Lightning的RNN state是$O(d^2/h)$大小（$d \times d$矩阵，除以$h$个head）。

**Capacity对比**：
- Softmax: $O(d)$
- Lightning: $O(d^2/h)$

当$d > h$（head dim 128 > head数 64对总维度而言，实际是$d \cdot h$ vs $d^2$），lightning有更大memory capacity。

但这里有个subtle的点作者没完全讲透：softmax的"Going Through a Book"机制（每次query都从头recompute）虽然precise，但它的state capacity受限。Hybrid让大部分层用大capacity的lightning memory，少数层用precise的softmax recompute，这种分工可能比全用softmax更好。

这个结论挑战了"softmax attention是retrieval金标准"的默认假设。

---

## MoE部分

这部分相对standard。456B总参数，45.9B activation，32 experts，top-2 routing。

公式（MoE output）:
$$h_t = \sum_{i=1}^{E} \text{Softmax}_i(\text{TopK}(x_t \cdot W_g)) \cdot \text{FFN}_i(x_t)$$

变量：
- $E = 32$：expert总数
- $W_g$：gate权重矩阵
- $\text{TopK}(\cdot)$：保留top-2分数，其余设$-\infty$
- $\text{FFN}_i$：第$i$个expert的FFN

Auxiliary loss（GShard style，https://arxiv.org/abs/2006.16668）:
$$L_{\text{aux}} = \alpha_{\text{aux}} \cdot \frac{1}{E} \sum_{i=1}^{E} f_i \cdot m_i$$

变量：
- $\alpha_{\text{aux}} = 0.01$：loss系数
- $f_i$：分配到expert $i$的token fraction
- $m_i$：expert $i$的平均routing probability

这个loss鼓励token均匀分布到所有expert。

**Global Router**是他们的创新点：跨EP group做allgather同步每个expert的token count，然后global dispatch。这解决了micro batch size小导致单EP group内token分布fluctuation大的问题。

Ablation里还有个有意思的发现：**PostNorm比PreNorm好**（Table 5）。大多数LLM用PreNorm因为训练稳定，但PostNorm保留effective depth对80层深模型重要。他们用DeepNorm（https://arxiv.org/abs/2203.04655）来稳定PostNorm训练，scaling factors $\alpha = (2N)^{0.25}$, $\beta = (8N)^{-0.25}$，$N$是layer数。

---

## Scaling law的发现

Table 2的scaling law拟合（基于Chinchilla方法，https://arxiv.org/abs/2203.15542）：

Hybrid-lightning的loss formula: $L = 3.4797 C^{-0.0763}$
- $L$：training loss
- $C$：compute budget (FLOPs)

对比：
- Softmax: $3.7087 C^{-0.0798}$（$\beta$更高，同样compute loss更高）
- Lightning: $3.5391 C^{-0.0768}$
- Hybrid: $3.4797 C^{-0.0763}$（最低$\beta$，最steep $\alpha$）

Intuition：给定相同compute budget，hybrid会用更多参数+更多tokens，但loss最低。这推翻了"linear attention scaling efficiency不如softmax"的假设。

Optimal model size $N_{opt} \propto C^{0.6670}$ (hybrid) vs $C^{0.7118}$ (softmax)。Hybrid对参数量scaling更慢，意味着compute增加时不需要堆那么多参数。

---

## 工程优化：这才是paper的隐藏价值

### EP-ETP Overlap

MoE训练的bottleneck是all-to-all (a2a) communication。传统方案：
- 用TP分expert参数 → compute intensity太低
- 不用TP → 参数太大，需要大PP，但PP不省activation memory

他们引入两个新ProcessGroup：
- **ETP** (Expert Tensor Parallel): 管expert weight partition
- **EDP** (Expert Data Parallel): 管identical expert的data parallel

约束：
$$\text{world\_size} = \text{size}_{PP} \times \text{size}_{DP} \times \text{size}_{CP} \times \text{size}_{TP}$$
$$\text{world\_size} = \text{size}_{PP} \times \text{size}_{EDP} \times \text{size}_{ETP} \times \text{size}_{EP}$$

这样MoE parallel策略完全从non-MoE解耦。然后做token分组overlap：一组compute时另一组做a2a。

结果：MoE pure communication overhead减50%。

### LASP+（Linear Attention Sequence Parallelism+）

原LASP（https://arxiv.org/abs/2404.02882）的问题：所有CP rank必须serial做send-recv交换KV block。

LASP+的改进：
1. 每个CP rank独立算local prefix sum $KV_L$
2. AllGather同步所有$KV_L$
3. 每个rank基于计算顺序选对应$KV_L$做global prefix sum

从serial变parallel，速度达到原LASP的$1/N_{pcn}$（$N_{pcn}$是parallel node数）。

### Varlen Ring Attention

原ring attention（https://openreview.net/forum?id=WsRHpHH4s0）要求每个sequence长度是$2 \times \text{size}_{CP}$的整数倍，data-packing下padding浪费大。

Varlen版直接对packed sequence应用ring attention，区分每个sequence在ring计算中的attention mask offset。把causal改成varlen causal，non-causal改成varlen non-causal。

### Lightning Attention Inference优化

四个trick让H20上MFU > 75%：

1. **Kernel fusion**: prefill阶段fuse Q/K/V处理，decoding阶段fuse KV计算和prefix cache更新
2. **Separated prefill/decoding**: length-1 token和length>1 token用不同kernel + 不同CUDA stream
3. **Multi-level padding**: block size从固定256变成32/64/128/256动态选择
4. **StridedBatchedMatmul**: 用cuBLAS `cublasGemmStridedBatchedEx`，256×256 GEMM用WGMMA指令+TMA异步

关键数据：1M sequence length下，softmax attention占95% latency，lightning attention只占<12%。这证明了lightning attention的inference效率优势。

---

## 长context训练recipe

Table 6的三阶段：

| Stage | Length | RoPE freq | Tokens | Short | Medium | Long |
|---|---|---|---|---|---|---|
| 1 | 128K | 5M | 300B | 30% | 70% | 0% |
| 2 | 512K | 10M | 32B | 35% | 35% | 30% |
| 3 | 1M | 10M | 26B | 30% | 30% | 40% |

RoPE base frequency从10K（短context默认）升到10M，这是length extrapolation的关键。短/中/长数据比例逐步调整，长数据占比增加。

**重要发现**: NIAH在128K training steps内就saturation了，不足以monitor训练进度。需要RULER（https://arxiv.org/abs/2404.06654）、LongBench-V2（https://arxiv.org/abs/2412.15204）这种更难的任务才能看到持续improvement。

---

## Post-training五阶段

Table 7的短-长-短-长-短交替：

1. SFT 8K tokens（baseline能力）
2. SFT 1M tokens（长context adaptation，50%长context数据）
3. DPO 8K tokens（短context preference calibration）
4. DPO 1M tokens（长context preference reinforcement）
5. Online RL 8K tokens（short context final polish）

RoPE base固定10M。

Online RL用modified GRPO（https://arxiv.org/abs/2402.03300），三个改进：

1. **Importance sampling weight clipping**: 双侧clip，避免大policy ratio + 负advantage的gradient不稳定
2. **KL divergence with stop-gradient**: 
$$D_{KL}(\theta) = \mathbb{E}_t[SG(\pi_\theta(a_t|s_t) - \pi_{\text{ref}}(a_t|s_t)) \log \pi_\theta(a_t|s_t)]$$
   $SG(\cdot)$是stop-gradient，减少gradient variance
3. **Balanced advantage estimation**: 调节正负样本reward贡献

---

## 长context benchmark结果

Table 9 (RULER) 在1M tokens：
- GPT-4o: 不支持（128K上限）
- Claude-3.5-Sonnet: 不支持（200K上限）
- Gemini-1.5-Pro: 0.850
- **MiniMax-Text-01: 0.910**

Table 10 (LongBench-V2, w/ CoT):
- GPT-4o: 51.4
- Claude-3.5-Sonnet: 46.7
- **MiniMax-Text-01: 56.5**（最好）

MR-NIAH（多轮needle，2000轮对话历史中retrieve）：MiniMax在English和Chinese上都显著优于GPT/Claude/Gemini，且长context下degradation最小。

MTOB（从一本语法书学Kalamang语）的$\Delta$ half book: **45.7**（最高），证明long-context in-context learning能力。

---

## VLM部分

ViT-MLP-LLM架构：
- ViT-L/14: 303M params，从头训练，ImageNet-1K zero-shot 80.55%
- 2-layer MLP projector: 随机初始化
- MiniMax-Text-01作为LLM

Dynamic resolution: 336×336到2016×2016 grid，每个patch 336×336，保留336×336 thumbnail。不用pooling，直接用raw features靠LLM long-context能力处理。

四阶段训练：
1. Modality alignment（80B tokens image description）
2. Vision understanding instruction tuning（420B multimodal + 21B text，20:1）
3. User experience enhancement（44.8B multimodal tokens）
4. DPO preference optimization（40K image-text pairs）

结果（Table 13）：
- MMMU: 68.5（接近GPT-4o 63.5，Claude 72.0）
- ChartQA: 91.7（接近SOTA）
- DocVQA: 96.4（接近SOTA）
- OCRBench: 865（SOTA）

---

## 我的核心takeaway

1. **Linear attention在production scale可行**。这打开了长context的新路径，不一定要靠softmax + FlashAttention的暴力优化。

2. **Hybrid架构反直觉地优于纯softmax**。Capacity分析（$O(d^2/h)$ vs $O(d)$）是核心insight。大部分层用cheap大capacity的linear memory，少数层用expensive精确的softmax recompute，这种分工可能更优。

3. **Engineering是关键**。LASP+把serial变parallel，EP-ETP overlap解决MoE communication，varlen ring attention避免padding浪费。这些engineering contribution跟architecture创新同等重要。

4. **NIAH saturation问题**。长context训练的monitoring需要更难的任务，RULER/LongBench-V2比NIAH更有区分度。

5. **短-长交替训练recipe**。不能只训长context，会忘短context能力。交替训练保持两种能力。

6. **RoPE base frequency调整**。从10K到10M是length extrapolation的关键lever。

7. **Prefilling latency优势**。由于linear复杂度，长context下prefilling latency远低于quadratic架构（Figure 2）。这对real-world deployment很重要。

参考资源：
- Paper repo: https://github.com/MiniMax-AI
- Hailuo AI体验: https://www.hailuo.ai/
- Lightning Attention原paper: https://arxiv.org/abs/2401.04658
- TransNormer: https://arxiv.org/abs/2205.12724（Qin et al., 2022a）
- LASP原paper: https://arxiv.org/abs/2404.02882
- Ring Attention: https://openreview.net/forum?id=WsRHpHH4s0
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- GShard: https://arxiv.org/abs/2006.16668
- DeepNorm: https://arxiv.org/abs/2203.04655
- Chinchilla: https://arxiv.org/abs/2203.15542
- DPO: https://arxiv.org/abs/2305.18290
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- RULER: https://arxiv.org/abs/2404.06654
- LongBench-V2: https://arxiv.org/abs/2412.15204
- MTOB: https://openreview.net/forum?id=tbVWug9f2h
- CoCa (ViT训练): https://arxiv.org/abs/2205.01917

---

# MiniMax-01 深度讲解

## 1. 核心贡献与动机

这篇paper的核心目标是**把context window推到百万级别甚至4M tokens**，同时在standard benchmarks上match GPT-4o和Claude-3.5-Sonnet。作者认为，单纯靠softmax attention + FlashAttention的优化路径已经遇到瓶颈，因为quadratic complexity在百万级context下硬件能力跟不上计算需求增长。他们选择了**linear attention的hybrid架构**这条路，这是首次在商业scale (456B params) 上成功部署linear attention。

paper链接: https://arxiv.org/abs/2501.08085 (实际paper URL需要查证，github: https://github.com/MiniMax-AI)

---

## 2. 架构设计: Hybrid Lightning Attention + MoE

### 2.1 整体架构

核心架构是**7个transnormer block (lightning attention) + 1个softmax transformer block**的循环pattern，共80层。这个设计的intuition是：
- Lightning attention提供long-context的constant-time inference能力
- 每8层插一个softmax attention层，解决retrieval能力不足的问题

具体hyperparameters：
- hidden size: 6144
- attention heads: 64, head dim: 128
- softmax attention用GQA, group size 8
- RoPE applied to half of head dimension, base frequency 10000
- 每层MoE: 32 experts, top-2 routing, FFN hidden dim 9216
- Total: 456B params, 45.9B activated per token

这个参数量的选择有约束: 单机8 GPU + 640GB memory + 8-bit quantization能跑1M tokens。

### 2.2 为什么是7:1 hybrid (而不是别的比例)

paper里做了ablation (Table 3, 4)：
- Hybrid-cosformer2: TGS 23.3K, NIAH 43.6
- Hybrid-hgrn2: TGS 29.5K, NIAH 91.8
- Hybrid-lightning: TGS 33.4K, NIAH 95.7 (best)
- Hybrid-window (window 1024): TGS 33.6K, NIAH 53.9

Lightning attention在NIAH (Needle in A Haystack) 上碾压sliding window，且speed相当。这就是为什么选lightning。

---

## 3. Lightning Attention 数学原理 (这是核心)

### 3.1 从softmax attention到linear attention

原始softmax attention (causal):
$$O = \text{Softmax}(QK^T / \sqrt{d}) V$$

Linear attention的"right product kernel trick":
$$O = \text{Norm}(QK^T V) = \text{Norm}(Q(K^T V))$$

变量含义：
- $Q, K, V \in \mathbb{R}^{n \times d}$: query, key, value matrices
- $n$: sequence length
- $d$: feature dimension
- 注意: 这里**没有softmax**，所以可以交换结合律

但causal masking下，需要计算cumsum，破坏并行性。这就是为什么linear attention提了9年没人用在大模型上。

### 3.2 Lightning Attention的tiling trick

这是paper的关键创新。把attention计算split成两部分：

**Intra-block** (block内，用left product):
$$O_{\text{intra}} = [(Q_t K_t^T) \odot M] V_t$$

**Inter-block** (block间，用right product):
$$O_{\text{inter}} = Q_t \cdot KV$$

其中$KV$是累积状态:
$$KV = KV_{prev} + K_t^T V_t$$

详细推导 (公式4-9):

设$X = [X_1; X_2]$, $X_1 \in \mathbb{R}^{m \times d}$, $X_2 \in \mathbb{R}^{(n-m) \times d}$

递推形式:
$$kv_s = kv_0 + \sum_{j=1}^{s} k_j v_j^T, \quad s = 1, ..., m$$
$$o_s^T = q_s^T kv_s = q_s^T kv_0 + q_s^T \sum_{j=1}^{s} k_j v_j^T$$

变量解释：
- $kv_s \in \mathbb{R}^{d \times d}$: 累积的key-value state (这是"memory")
- $q_s, k_s, v_s$: 第$s$个token的query, key, value
- $o_s$: 第$s$个token的output

block形式 (公式7):
$$O_1 = Q_1 kv_0 + [(Q_1 K_1^T) \odot M] V_1 \triangleq Q_1 KV_0 + [(Q_1 K_1^T) \odot M] V_1$$

第二block (公式8):
$$O_2 = Q_2 kv_m + [(Q_2 K_2^T) \odot M] V_2 \triangleq Q_2 KV_1 + [(Q_2 K_2^T) \odot M] V_2$$

状态更新 (公式9):
$$KV_1 = KV_0 + K_1^T V_1$$

**复杂度**: $O(nd^2 + nBd)$, 其中$B$是block size。当$B$选得合适，这就是线性的。

### 3.3 Algorithm 1 (IO-aware实现)

```
Input: Q, K, V ∈ R^{n×d}, block sizes B
Divide X into T = n/B blocks
Initialize KV = 0 ∈ R^{d×d}
for t = 1, ..., T:
    Load Q_t, K_t, V_t from HBM to SRAM
    O_intra = [(Q_t K_t^T) ⊙ M] V_t  (on-chip)
    O_inter = Q_t (KV)               (on-chip)
    KV = KV + K_t^T V_t              (on-chip update)
    Write O_t = O_intra + O_inter to HBM
```

这是类似FlashAttention的tiling思想，但用在linear attention上。

### 3.4 为什么hybrid比纯softmax在retrieval上更强？

这是paper里最interesting的insight (Section 2.2.4)。

把softmax attention改写成linear RNN form (公式11):
$$s_t^0 = 0, \quad s_t^j = s_t^{j-1} + \exp(q_t k_j^T / \sqrt{d})$$
$$o_t^j = (s_t^{j-1}/s_t^j) o_t^{j-1} + (1 - s_t^{j-1}/s_t^j) v_j$$
$$o_t = o_t^t, \quad j = 1, ..., t$$

注意这里softmax的RNN形式：**每个query token $t$都要从$t_0 = 1$重新recompute hidden state**。这叫"Going Through a Book"。

而lightning attention的RNN form (公式12):
$$kv_0 = 0, \quad kv_j = kv_{j-1} + k_j v_j^T, \quad o_j = kv_j^T q_j$$

**Capacity对比**:
- Softmax attention capacity: $O(d)$ (因为state是标量累加，归一化后)
- Lightning attention capacity: $O(d^2/h)$ (因为state是$d \times d$矩阵)
- 因为$d > h$ (head dim通常128, head数通常64)，lightning attention有更大capacity

这就是为什么hybrid-lightning在NIAH上比纯softmax强——它的"memory"更大。这是一个反直觉的结论。

---

## 4. Scaling Law分析

paper做了70M到7B的scaling experiments (Table 1, 2, Figure 6)。

### 4.1 FLOPs计算

| Architecture | Params | FLOPs |
|---|---|---|
| Softmax | $12ld^2$ | $72bnld^2(1 + \frac{n}{6d} + \frac{5}{18d})$ |
| Lightning | $12ld^2 + 2ld^2/h$ | $72bnld^2(1 + \frac{1}{2h} + \frac{5}{18d})$ |
| Hybrid | $12ld^2 + 7ld^2/(4h)$ | $72bnld^2(1 + \frac{n}{48d} + \frac{7}{16h} + \frac{5}{18d})$ |

变量：
- $l$: layer数
- $d$: model dimension
- $h$: attention head数
- $b$: batch size
- $n$: sequence length

关键观察: Lightning的FLOPs不依赖$n$ (sequence length)，这就是constant-time inference的根源。Hybrid的$n/48d$项远小于pure softmax的$n/6d$项。

### 4.2 Scaling law拟合

| Arch | L(C) | N_opt(C) | D_opt(C) |
|---|---|---|---|
| Softmax | $3.7087 C^{-0.0798}$ | $(1.82 \times 10^8) C^{0.7118}$ | $(2.56 \times 10^{10}) C^{0.5102}$ |
| Lightning | $3.5391 C^{-0.0768}$ | $(2.74 \times 10^8) C^{0.6470}$ | $(4.43 \times 10^{10}) C^{0.4684}$ |
| Hybrid | $3.4797 C^{-0.0763}$ | $(2.57 \times 10^8) C^{0.6670}$ | $(3.70 \times 10^{10}) C^{0.4707}$ |

变量：
- $L$: loss
- $C$: compute budget (FLOPs)
- $N_{opt}$: optimal model size
- $D_{opt}$: optimal dataset size

Hybrid的loss最低 ($\beta = 3.4797 < 3.5391 < 3.7087$)，且指数$\alpha = 0.0763$最steep，意味着scaling效率最高。**给定相同compute budget，hybrid会用更多参数+更多tokens，但loss更低**。

---

## 5. MoE设计

### 5.1 公式 (公式1)

$$h_t = \sum_{i=1}^{E} \text{Softmax}_i(\text{TopK}(x_t \cdot W_g)) \cdot \text{FFN}_i(x_t)$$

变量：
- $E$: expert总数 (这里32)
- $W_g$: gate weight
- $\text{FFN}_i$: 第$i$个expert
- $\text{TopK}(\cdot)$: 保留top K scores (K=2)，其余设为$-\infty$

### 5.2 Auxiliary Loss

$$L_{\text{aux}} = \alpha_{\text{aux}} \cdot \frac{1}{E} \sum_{i=1}^{E} f_i \cdot m_i$$

变量：
- $\alpha_{\text{aux}}$: auxiliary loss系数 (这里0.01)
- $f_i$: 分配到第$i$个expert的token fraction
- $m_i$: expert $i$的平均routing probability

这是GShard-style的load balancing loss。

### 5.3 Global Router

关键创新: 跨EP group的token同步。问题在于micro batch size受GPU memory限制，单个EP group内token分布fluctuation大。Global router加一个allgather来synchronize每个expert的token count，跨EP group做dispatch。

### 5.4 PreNorm vs PostNorm ablation

Table 5显示PostNorm (with DeepNorm) 全面优于PreNorm。DeepNorm scaling: $\alpha = (2N)^{0.25}$, $\beta = (8N)^{-0.25}$，其中$N$是layer数。

intuition: PostNorm保留模型effective depth，对deep model (80层) 重要。PreNorm让gradient直接通过residual bypass sub-layer，降低effective depth。

---

## 6. 计算优化 (Section 3)

### 6.1 EP-ETP Overlap

这是分布式训练的核心trick。

**问题**: MoE的all-to-all (a2a) communication是bottleneck。如果用TP分expert参数，compute intensity太低；如果不用TP，参数太大需要大PP，但PP不省activation memory。

**解决方案**: 引入两个新ProcessGroup:
- ETP (Expert Tensor Parallel): 管expert weight partition
- EDP (Expert Data Parallel): 管identical expert的data parallel

约束 (公式15, 16):
$$\text{world\_size} = \text{size}_{PP} \times \text{size}_{DP} \times \text{size}_{CP} \times \text{size}_{TP}$$
$$\text{world\_size} = \text{size}_{PP} \times \text{size}_{EDP} \times \text{size}_{ETP} \times \text{size}_{EP}$$

这样MoE的parallel策略完全从non-MoE解耦。

**Overlap策略**: token分组，一组做compute时另一组做a2a communication。Figure 10展示三种配置: (a) low compute, (b) high compute, (c) fewer groups。High compute时overlap更好。

结果: MoE pure communication overhead减少50%。

### 6.2 Varlen Ring Attention

**问题**: 传统ring attention要求每个sequence长度是$2 \times \text{size}_{CP}$的整数倍，data-packing下需要大量padding。

**解决**: 直接对packed sequence应用ring attention，通过区分每个sequence在ring计算中的attention mask offset。把causal computation改成varlen causal，non-causal改成varlen non-causal (Figure 11)。

### 6.3 LASP+ (改进的Linear Attention Sequence Parallelism)

**原LASP问题** (Figure 12a): 所有CP rank必须serial做send-recv交换KV block，效率极差。

**LASP+改进** (Figure 12b):
1. **Local Prefix Sum**: 每个CP rank独立计算local prefix sum $KV_L$
2. **AllGather**: 全局同步所有rank的$KV_L$
3. **Global Prefix Sum**: 每个rank基于计算顺序选择对应CP rank的$KV_L$做prefix sum

结果: 计算速度达到原LASP的$1/N_{pcn}$ ($N_{pcn}$是parallel node数)，因为从serial变parallel。AllGather的额外communication cost minimal。

### 6.4 Lightning Attention Inference优化

四个trick:

**Batched Kernel Fusion**: prefill阶段fuse Q/K/V处理的多个memory-bound kernel；decoding阶段fuse KV计算和prefix KV cache更新。decoding latency降低10%。

**Separated Prefill and Decoding**: 把length-1 token和length>1 token用不同kernel + 不同CUDA stream并行调度。例: batch 20里2个length-50 + 18个length-1，latency从100ms降到50ms。

**Multi-level Padding**: 原本block size固定256，但prefix cache后token长度通常<256。引入32/64/128选项，动态选最小padding overhead的scale。

**StridedBatchedMatmul Extension**: 用cuBLAS的`cublasGemmStridedBatchedEx`。256×256 GEMM用WGMMA指令，配合TMA异步操作，CUDA Cores做pre/post processing。

结果: H20上end-to-end inference MFU > 75%。在1M sequence length下，softmax attention占95% latency，lightning attention只占<12%。

---

## 7. Pre-training

### 7.1 数据

- Tokenizer: byte-level BPE, 200K vocab, multilingual up-sampling
- 数据质量: 用前代MoE model (5B activation, 60B total) 做reward labeler，评估knowledge depth, helpfulness, categorical distribution
- Repetition-aware实验: 发现低质量数据>2 epoch性能下降，高质量数据可训4 epoch
- 关键metric: $\log \text{acc}_{\text{norm}^2}$, 用byte-normalized概率排除tokenizer影响

### 7.2 训练schedule

- AdamW: $\beta_1 = 0.9, \beta_2 = 0.95$, weight decay 0.1
- Sequence length 8192开始
- Batch size: 16M → 32M (69B tokens) → 64M (790B tokens) → 128M (4.7T tokens)
- LR: warmup到$2 \times 10^{-4}$, 7.2T tokens constant, 然后降到$1.3 \times 10^{-4}$ (因为gradient norm异常), fast decay到$3 \times 10^{-5}$
- MoE auxiliary loss系数: 0.01

### 7.3 三阶段Long-Context Extension

| Length | RoPE freq | Tokens | Short% | Medium% | Long% |
|---|---|---|---|---|---|
| 128K | 5M | 300B | 30 | 70 | 0 |
| 512K | 10M | 32B | 35 | 35 | 30 |
| 1M | 10M | 26B | 30 | 30 | 40 |

关键观察: NIAH在128K training steps内就saturation了，不足以monitor训练进度。需要更难的任务 (RULER, LongBench-V2) 才能看到持续improvement。

RoPE base frequency从10K (短context) 升到10M (长context)，这是为了length extrapolation。

---

## 8. Post-training

### 8.1 五阶段训练 (Table 7)

| Stage | Length | Epoch | Batch | Max LR | Min LR | LR Decay |
|---|---|---|---|---|---|---|
| I (SFT short) | 8192 | 2 | 128 | 1e-5 | 1e-6 | Cosine |
| II (SFT long) | 1032192 | 2 | 80 | 3e-6 | 3e-6 | Constant |
| III (DPO short) | 8192 | 1 | 64 | 5e-7 | 5e-8 | Cosine |
| IV (DPO long) | 1032192 | 1 | 64 | 5e-7 | 5e-7 | Constant |
| V (Online RL short) | 8192 | 1 | 512 | 1e-6 | 1e-7 | Cosine |

intuition: 短-长-短-长-短交替，让模型既保持短context能力又获得长context能力。RoPE base固定10M。

### 8.2 Online RL改进 (GRPO变体)

三个trick:

**1. Importance Sampling Weight Clipping**: 传统PPO/GRPO用one-sided clipping，policy ratio大且advantage为负时gradient不稳定。这里直接abandon这种情况。

**2. KL Divergence Optimization**:
$$D_{KL}(\theta) = \mathbb{E}_t[SG(\pi_\theta(a_t|s_t) - \pi_{\text{ref}}(a_t|s_t)) \log \pi_\theta(a_t|s_t)]$$

变量：
- $\pi_\theta$: current policy
- $\pi_{\text{ref}}$: reference policy
- $SG(\cdot)$: stop-gradient operator
- $a_t$: action at step $t$, $s_t$: state at step $t$

用stop-gradient处理$\pi_\theta - \pi_{\text{ref}}$项，减少gradient variance。

**3. Balanced Advantage Estimation**: 调节正负样本reward贡献，应对skewed distribution。

### 8.3 Search Tool集成

约30-40% user query触发search。通过special tokens直接invoke tool，避免multi-step planning或CoT打断对话flow。Search decision boundary对齐model knowledge boundary，丢弃model已master的query。性能从58% → 71.5%。

---

## 9. Vision-Language Model

### 9.1 架构

"ViT-MLP-LLM" paradigm:
- ViT-L/14: 303M params, 从头训练
- 2-layer MLP projector: 随机初始化
- MiniMax-Text-01作为LLM

**Dynamic resolution**: 336×336到2016×2016的grid配置，每个patch 336×336。同时保留一个336×336 thumbnail。所有patch独立encode后concatenate。

**不用pooling**: 直接用raw high-dim features，靠LLM的long-context能力处理。

### 9.2 四阶段训练

| Stage | 目标 | 数据量 |
|---|---|---|
| I | Modality alignment (update ViT + adapter) | 80B tokens (image description) |
| II | Vision understanding instruction tuning (update all) | 420B multimodal + 21B text (20:1) |
| III | User experience enhancement | 44.8B multimodal tokens |
| IV | DPO preference optimization | 40K image-text pairs |

ViT-L/14训练: 先224×224训37B image-caption pairs，再336×336 finetune 1.2B pairs。ImageNet-1K zero-shot 80.55%。

---

## 10. 实验结果

### 10.1 Core benchmarks (Table 8)

MiniMax-Text-01 vs GPT-4o:
- MMLU: 88.5 vs 85.7 (胜)
- C-SimpleQA: 67.4 vs 64.6 (胜)
- IFEval: 89.1 vs 84.1 (胜)
- Arena-Hard: 89.1 vs 92.4 (略逊)
- MATH: 77.4 vs 76.6 (胜)
- HumanEval: 86.9 vs 90.2 (略逊)

### 10.2 Long-context (Table 9, RULER)

在1M tokens时：
- GPT-4o: 不支持 (128K上限)
- Claude-3.5-Sonnet: 不支持 (200K上限)
- Gemini-1.5-Pro: 0.850
- Gemini-2.0-Flash: 不支持 (1M但性能崩溃)
- **MiniMax-Text-01: 0.910** (碾压)

### 10.3 MR-NIAH (多轮needle)

在2000轮对话历史中retrieve特定response，MiniMax-Text-01在English和Chinese上都显著优于GPT/Claude/Gemini，且长context下degradation最小。

### 10.4 MTOB (从一本语法书学习新语言)

eng → kalam (ChrF):
- No context: 6.0 (最低，因为pre-training数据少Kalamang)
- Half book: 51.74
- Full book: 51.60
- Δ half book: **45.7** (最高，意味着从context学到的最多)

这证明long-context in-context learning能力。

---

## 11. Prefilling Latency优势 (Figure 2)

由于lightning attention的线性复杂度，MiniMax-Text-01在长context下prefilling latency远低于Llama3-70B等。这是架构的天然优势——quadratic vs linear。

---

## 12. 局限性

1. 仍保留1/8的softmax attention，未来希望完全eliminate
2. Coding能力不足 (HumanEval 86.9 vs GPT-4o 90.2)
3. Long-context evaluation dataset仍偏artificial

---

## 13. 我的intuition总结

这篇paper的深层意义在于证明了**linear attention在production scale可行**。之前的linear attention工作都卡在小scale，因为causal masking下的cumsum无法parallelize。Lightning attention的tiling trick把问题拆成intra-block (left product, small, 可以并行) + inter-block (right product, 累积状态)，这是algebraic的精妙。

最counter-intuitive的结论是hybrid比pure softmax在retrieval上更强。作者的解释是capacity: softmax attention作为RNN capacity是$O(d)$，而lightning attention是$O(d^2/h)$。当$d > h$时lightning更大。但这只解释了部分——还有一个未明说的点: softmax attention的"Going Through a Book"机制虽然精确但expensive，hybrid让大部分层用cheap的lightning attention，少数层用precise的softmax，这种分工可能更优。

engineering上，LASP+把serial变parallel的trick很elegant。EP-ETP overlap解决MoE communication也是实用的engineering contribution。H20上75% MFU对于这种hybrid架构很impressive。

参考链接:
- GitHub: https://github.com/MiniMax-AI
- Hailuo AI: https://www.hailuo.ai/
- API: https://intl.minimaxi.com
- Lightning Attention原paper: https://arxiv.org/abs/2401.04658
- TransNormer: Qin et al., 2022a
- LASP原paper: https://arxiv.org/abs/2404.02882
- Ring Attention: https://openreview.net/forum?id=WsRHpHH4s0
- FlashAttention-2: https://openreview.net/forum?id=mZn2Xyh9Ec
- GShard: https://openreview.net/forum?id=qrwe7XHTmYb
- DeepNorm: Wang et al., 2024a
- Chinchilla scaling laws: Hoffmann et al., 2022
- DPO: https://arxiv.org/abs/2305.18290
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
