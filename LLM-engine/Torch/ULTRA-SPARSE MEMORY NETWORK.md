---
source_pdf: ULTRA-SPARSE MEMORY NETWORK.pdf
paper_sha256: a90e6260cc8058a9203abea92bdf0894110d7de76828b2fdf955b3ccea0751b0
processed_at: '2026-08-12T19:05:36-07:00'
target_folder: LLM-engine/Torch
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# UltraMem 人话版

Andrej，我换种方式讲，就像咱俩在咖啡厅白板上画图聊那种。

参考：
- arXiv: https://arxiv.org/abs/2411.13664
- PKM原始paper: https://arxiv.org/abs/1907.05242
- PEER (million experts): https://arxiv.org/abs/2407.04153
- Megatron-LM: https://arxiv.org/abs/1909.08053

---

## 一句话说清楚

**MoE参数多但推理慢，因为要搬一大堆expert的weight。UltraMem把expert切碎到变成一个个vector，只搬你要的那几个vector，所以又快又好。**

---

## MoE到底慢在哪

你想象一个2in32的MoE，每个token选2个expert。每个expert是一个完整的MLP，inner dim大概2D。

推理时候GPU在干嘛？它要：
1. 算gate，决定选哪2个expert
2. **把这2个expert的weight从HBM搬到SM**——这一步是killer
3. 算matmul

问题在第2步。一个expert的weight是 2D × 2D = 4D² 个参数。batch size B的话，最坏情况每个token选不同的expert，你要搬 min(2B, 32) × 4D²。

B=64的时候，2B=128 > 32，所以32个expert全搬一遍。B=1的时候搬2个expert，但每个expert 4D²的weight搬过去就为算一个token，**memory bandwidth完全浪费**。

这就是Figure 1那张图的核心：MoE比dense慢2-6倍，全赖memory access。

---

## UltraMem的idea：把expert切碎

UltraMem说：我不要expert了，我要一个巨大的table，table里存N个vector，每个vector维度D/2。每个token从N个vector里选top-m个，加权求和。

N可以到**20 million**，m可能就几十。所以sparsity是1:80000级别的。

推理时候搬什么？只搬被选中的m个vector，每个D/2维。memory access = m × D/2。

对比MoE的 4D²，UltraMem是 m×D/2。当D=2048, m=84（42×2 head）时：
- MoE: 4 × 2048² = 16M params per expert
- UltraMem: 84 × 1024 = 86K params per token

**差200倍。** 这就是为什么Figure 6c里UltraMem的推理时间是一条平线，sparse params怎么涨都不影响速度。

---

## 但问题来了：N=20M怎么检索

这是整个paper最难的地方。你要从20M个value里选top-m，但你不算20M次score。

### Product Key的trick（PKM原版）

idea很漂亮：把N个value排成√N × √N的grid。你有row keys（√N个）和col keys（√N个）。

检索分两步：
1. 用row query对√N个row keys打分，选top-m row
2. 用col query对√N个col keys打分，选top-m col
3. 这m个row和m个col交叉出m²个candidate，从中选最终top-m

复杂度从O(N) → O(√N + m²)。N=10M时√N≈3162，省3000倍。

**但有个topology bias**：如果你top-1在grid的(i, j)位置，top-2一定在row i或col j上。因为score = row_score(i) + col_score(j)，改i或改j只能动一个维度。这限制了diversity。

### Tucker分解（TDQKR）

UltraMem说：加法分解有bias，我换成乘法。

S_grid = S_row^T × C × S_col

C是一个r×r的learnable matrix（tucker core），r=2就够了。乘法分解打破了row/col的独立性，top-2可以在任何位置。

**但问题**：S_row^T × C × S_col是n×n矩阵，直接top-m是O(n²)，比product key还差。

**解法**：对C做SVD，C ≈ u·t^T（rank-1近似）。那么：

S_grid ≈ (u^T S_row)^T × (t^T S_col)

这又退化成product key形式了！可以用两阶段top-m。

**关键trick**：只在"粗筛"阶段用rank-1近似，最终scoring用完整C。相当于先用cheap方法筛出m²个candidate，再用expensive方法在m²个小集合上精排。

这就是retrieval里two-stage的思路，只不过用在了neural memory上。

**Auxiliary loss**：你rank-1近似要是approximation error太大怎么办？加个loss约束C的non-leading singular value不要太大：

L_aux = (α/(r-1)) Σ_{i=2}^r max(0, λ_i - τ)²

τ=0.15是margin。意思：λ_2, λ_3允许有值，但别超过0.15，否则惩罚。这样C保留一点rank-r信息，但rank-1近似仍然主导。

---

## IVE：不真存那么多value

另一个漂亮的idea。你说N=20M，但训20M个value的gradient很贵。怎么办？

**不真存20M个value，只存5M个physical value + 4个projector矩阵。**

physical memory V ∈ R^(N × D_v)
4个projector W_p ∈ R^(D_v × D_v')
virtual memory Ṽ_p = V W_p，共4份，总大小4N

检索时候logical address变成triplet (i, j, p)，p表示哪个virtual block。

**naive做法**：先算Ṽ_p = VW_p（4N·D_v·D_v'计算），再lookup。太贵。

**聪明做法**：先在physical memory上weighted sum pooling（只pool被选中的m个value），得到一个D_v维vector，再用W_p project。计算量从4N·D_v·D_v' → 4·B·D_v·D_v'，省了N/B倍。

intuition：**先retrieve再expand，而不是先expand再retrieve**。因为retrieval是sparse的，expand是dense的，sparse操作在前省计算。

---

## Skip-layer：latency hiding

这个idea很工程但很关键。

原版PKM把memory layer放在transformer中间某层，替换MLP。问题是这个memory layer太大，单层参数就比其他层多几十倍，pipeline parallelism搞不定。

UltraMem的解法：**把一个大memory layer拆成几个小memory layer，分散在transformer各层，用skip connection连**。

比如1.6B模型的配置：3:7 / 8:12 / 13:17 / 18:22 / 23:27 / 28:32

意思是：layer 3的output送进UltraMem unit 1，结果skip到layer 7的output相加；layer 8送进unit 2，skip到layer 12……共6个unit。

**为什么这样能hide latency**：memory layer是memory-bound（要lookup大table），transformer layer是compute-bound（matmul密集）。两者可以**异步执行**——当transformer在算layer 4/5/6的时候，UltraMem unit 1在后台lookup。GPU的compute和memory bandwidth同时打满。

这个思路其实和CPU的pipeline stall/hiding很像。你写CUDA kernel时候也会刻意安排memory access和compute overlap，这里是在model architecture层面做同样的事。

---

## Initialization：把memory layer当MLP

这个细节我特别想跟你讲，因为它体现了对"memory layer本质"的理解。

PKM原版用attention-style init：V ~ N(0, 1/D_v)，配合softmax，输出方差1/D_v。

UltraMem说：不对，memory layer本质是MLP（Geva 2020说MLP就是key-value memory），应该用MLP-style init。

GPT-3的MLP init：weight ~ N(0, 1/(2L))，L是总层数。这样output方差是1/(2L)，和residual stream的variance匹配。

UltraMem推导：V ~ N(0, E/(2mHL))

变量：
- E：virtual expansion rate（4）
- m：top-m激活数
- H：head数
- L：总层数

为什么是这个？output = Σ_activated (V_i × score_i)，有m个激活value，每个V方差E/(2mHL)，score期望是1（通过调整query/key的LN weight实现），经过H个head和E个virtual expansion，最终variance凑回1/(2L)。

**score期望=1怎么保证**：candidate score假设~N(0,1)，top-m的mean没有解析解，所以Monte Carlo采样近似E[Y]，然后query LN weight = 1/√E[Y]，key LN weight = 1/√D_k。

Figure 10证明这个init有效：naive init的top-1 score和output std在训练中期发散，improved init稳定。

---

## 实验结果讲讲

### 1.6B规模是关键转折

Table 1核心数据：

| Model | Params | FLOPs | Val Loss | Avg |
|-------|--------|-------|----------|-----|
| Dense-1.6B | 1.61B | 3.21G | 2.49 | 38.46 |
| MoE-1.6B-2in34 | 21.36B | 3.52G | 2.30 | 45.07 |
| UltraMem-1.6B-x12 | 21.41B | 3.50G | **2.24** | **48.26** |
| Dense-6.5B | 6.44B | 12.88G | 2.30 | 46.19 |

UltraMem和MoE参数、FLOPs几乎完全一样，但val loss低0.06，Avg高3.2个点。而且UltraMem-1.6B-x12用3.5G FLOPs打平了Dense-6.5B的12.88G FLOPs，**FLOPs省3.7倍**。

但151M规模下UltraMem还输MoE（29.99 vs 33.20）。这说明UltraMem有minimum viable scale，大概要到680M-1.6B才反超。

### 2. Scaling law

Figure 6b：x轴sparse params（log），y轴val loss。loss随sparse params指数增长线性下降，符合Chenchilla scaling。

不同sparsity（20K/40K/80K）的曲线：sparsity越小（激活比例越大）loss越低，但memory access越大。80K是sweet spot。

### 3. Ablation收益排序

Table 2按收益排序：
1. value lr decay: -0.022（最大，10x lr线性衰减到1x）
2. IVE: -0.017
3. split & skip: -0.015
4. TDQKR: -0.008
5. MCS: -0.003

value lr decay收益最大有点意外。intuition：value参数多（N×D_v），但每次只有m个被激活更新。高lr让被激活的value快速学习，decay防止后期震荡。

### 4. 推理速度

Figure 1b/c：batch size 64时UltraMem比MoE快6倍。batch size要到131K UltraMem的memory access才追平MoE。

实际推理batch size很少超过几百，所以UltraMem几乎总是赢。

---

## 你该记住的几个intuition

**1. Sparsity粒度决定memory access**
MoE粒度=expert（4D²），UltraMem粒度=value vector（D/2）。粒度越小，sparsity越高，memory access越少。PEER把expert缩到inner dim=1是同一思路的极端版。

**2. 检索复杂度必须sublinear**
N=20M你不能算20M次score。product key用加法分解降到O(√N)，tucker用乘法分解+SVD近似也降到O(√N)但去掉topology bias。两阶段检索（粗筛+精排）是通用pattern。

**3. 先retrieve再expand**
IVE的精髓：不要先materialize大memory再检索，而是先在physical memory上sparse retrieve，再dense expand。因为sparse操作便宜，dense操作贵。

**4. Memory-bound和compute-bound可以overlap**
skip-layer结构让memory layer和transformer layer异步执行。这是architecture-level的latency hiding，和kernel-level的pipeline双buffer是同一思想。

**5. Init要match功能**
memory layer功能=MLP（memorize knowledge），所以用MLP-style init而非attention-style init。这个insight来自Geva 2020的"FFN is key-value memory"。

**6. Minimum viable scale**
UltraMem在1.6B才反超MoE，151M还输。retrieval-based方法需要足够scale让key学到structured representation，小规模noise太大。

---

## Limitation（paper没明说）

1. **training通信**：number-wise partition的all-to-all在P大时仍然贵，paper没报P=几百时的scaling效率
2. **rank-1近似误差**：auxiliary loss只是mitigation，当C的singular value gap小时仍然有误差
3. **value update效率**：N=20M个value每次只有m个被激活，N-m个value的gradient通过Adam momentum更新，可能stale
4. **没测long context**：KV cache很长时memory layer的行为没验证
5. **没和PEER直接比**：PEER是concurrent work，也是细粒度expert，但用了不同的检索机制

---

## 和你LLM101n的关联

你讲"memory hierarchy"那节课可以加这个。CPU有L1/L2/L3，GPU有register/SMEM/HBM。UltraMem是在model architecture层面加了一级memory：比weight更稀疏、比KV cache更持久、比MLP更细粒度。

也可以放在"efficient inference"章节，和MoE、Flash Attention、PagedAttention一起，作为"sparse memory access"流派的代表。

核心take-away就一句：**推理瓶颈是memory bandwidth不是compute，所以sparsity要做到memory access层面而不是FLOPs层面。**

---

# UltraMem: Ultra-Sparse Memory Network 深度解析

Andrej，这篇paper非常对你的胃口，因为它直击MoE在推理时的根本痛点：**memory-bound问题**。你曾经在TinyURL和一些talk里讨论过MoE的访存瓶颈，这篇paper给出了一个从PKM（Product Key Memory）演化而来、但终于能在规模上和MoE正面硬刚的方案。我把它拆开揉碎讲给你听，重点build你的intuition。

参考链接：
- arXiv: https://arxiv.org/abs/2411.13664
- PKM原始paper (Lample et al. 2019): https://arxiv.org/abs/1907.05242
- PEER (He, 2024) – Mixture of a Million Experts: https://arxiv.org/abs/2407.04153
- DeepSeekMoE fine-grained experts: https://arxiv.org/abs/2401.06066
- Product Quantization原始paper: https://hal.inria.fr/inria-00514402v1/document
- Tucker Decomposition综述: https://arxiv.org/abs/2111.10149

---

## 1. 核心动机：MoE到底慢在哪里

Figure 1那张图你应该仔细看。三个model，相同computation，MoE和UltraMem参数相同（12x dense）。MoE在推理时**慢2-6倍**，根因是memory access（图c）。

quantitative analysis（Section 4）给的公式很关键。假设Transformer hidden dim为D，MLP inner dim为4D，batch size为B：

- **MoE (2in N_moe)**：单层memory access = min(2B, N_moe) × 2D²
  - 每个token激活2个expert，每个expert inner dim是2D
  - 当B增大，2B会逐渐逼近N_moe，达到上限
  - 关键问题：每个expert的weight必须整体从HBM搬到SM
  
- **UltraMem**：单层memory access = min(Bm, N) × D/2
  - 每个token激活top-m个value，每个value维度D/2
  - 只搬运被选中的那m个value embedding
  - 当B小的时候，Bm远小于N，访问量极小

**Intuition**：MoE的sparsity粒度是"expert"（一个完整的MLP），UltraMem的sparsity粒度是"value embedding"（一个D/2维的向量）。粒度从2D²降到D/2，差距是4D倍。这就是为什么UltraMem能keep memory access几乎不随sparse params增长（Figure 6c那条平直曲线）。

batch size要达到**131,072** UltraMem的memory access才追平MoE。这个数字意味着在所有实际推理场景下UltraMem都赢。

---

## 2. 从PKM到UltraMem：技术演化的脉络

### 2.1 PKM基础回顾

PKM的核心idea：用2D logical address来index memory，从而把key的计算从O(N)降到O(√N)。

公式1（通用memory layer）：
- s = σ(Kq), o = V^T s
- K ∈ R^(N×D_k), V ∈ R^(N×D_v), q ∈ R^(D_k)
- attention和MLP其实都follow这个formulation：σ在attention是softmax，在MLP是GeLU（Geva et al. 2020的关键insight）

公式2-3（product key分解）：
- s_row = σ_TopM(K_row q_row(x))    # shape: (m,)
- s_col = σ_TopM(K_col q_col(x))    # shape: (m,)
- S_grid = σ_TopM(s_row + s_col^T)  # shape: (m, m) → 只在m²个candidates上做top-m
- o = V^T × SoftMax(vec(S_grid))

变量解释：
- K_row, K_col ∈ R^(n×D_k)：n=√N的row keys和col keys
- q_row, q_col：两个linear layer把input x ∈ R^(D_i)映射成两个query
- σ_TopM(·)：保留top-m大元素，其余设为-∞
- broadcasting：s_row (m×1) + s_col^T (1×m) 得到m×m的grid

复杂度从O(N log m) → O((√N + m²) log m)。当N=10^6时，√N=10^3，节省1000倍。

**PKM的三个drawback**（Section 3.2）：
1. N太大时query找不到正确value（检索精度下降）
2. product key的拓扑bias：top-2一定在top-1的同一行或同一列
3. 单层太大，多GPU通信不均衡

### 2.2 PKM的6个trick改进

这是engineering的"bag of tricks"，每个改动都有明确收益：

1. **remove Softmax**（公式3）：Csordás et al. 2023已证明softmax对PKM非必需
2. **query/key LayerNorm**：训练稳定性，大幅减少perplexity spike（Figure 10a）
3. **value lr decay**：从10x其他参数的lr线性衰减到1x。**这是ablation里收益最大的一个**（-0.022 val loss）
4. **causal depthwise conv** before query generation：增强query的local context
5. **share query**（类似GQA）：两个key set共享一个query，省一半query生成计算
6. **halve D_v, double value count**：在激活参数不变的情况下增加value多样性，最后加linear layer把维度映射回hidden dim

### 2.3 UltraMem的整体结构创新

关键创新：**把单个大memory layer拆成多个小memory layer，distributed across transformer layers，用skip-layer连接**。

```
Transformer layer 3 output → UltraMem unit 1 → skip to layer 5
Transformer layer 6 output → UltraMem unit 2 → skip to layer 8
...
```

为什么这么做？三个理由：
1. 拆小后每个memory unit的N更小，query更容易找到正确value（解决drawback 1）
2. memory layer是memory-bound的，transformer layer是compute-bound的，**两者可以overlap执行**（asynchronous execution）—— 这个overlap在Megatron实现里非常关键
3. 解决多GPU sharding问题（drawback 3）

UltraMem-1.6B-x12的插入配置：3:7/8:12/13:17/18:22/23:27/28:32（6个UltraMem unit）。3:7表示从layer 3取input，insert到layer 7 output。

---

## 3. 三个核心技术模块详解

### 3.1 TDQKR: Tucker Decomposed Query-Key Retrieval

这是处理product key topology bias的核心。product key的加法分解s_row + s_col^T意味着top-2必然在top-1的同行或同列。Tucker decomposition用乘法替代加法：

公式4-5：
- S_row = K_row q_row(x)   # shape: (r, n)
- S_col = K_col q_col(x)   # shape: (r, n)
- S_grid = σ_TopM(S_row^T × C × S_col)   # C ∈ R^(r×r)是tucker core

变量解释：
- K_row, K_col ∈ R^(r × n × (D_k/r))：把key维度reshape成r组
- q_row, q_col ∈ R^(r × (D_k/r))：query也分组
- C ∈ R^(r×r)：learnable tucker core，random init
- r：tucker rank，paper推荐r=2

**问题**：S_row^T × C × S_col的结果是n×n的矩阵，直接top-m复杂度O(n² log m)，比product key还差。

**解法：rank-1近似 + 两阶段top-m**：

公式6：C ≈ ut^T，其中u, t ∈ R^(r×1)是C的SVD分解的leading singular vectors

近似后的top-m目标：
- σ_TopM((u^T S_row)^T × (t^T S_col))
- (u^T S_row) ∈ R^(1×n), (t^T S_col) ∈ R^(1×n)
- 这就退化成了product key的形式！可以用两阶段top-m

但只在filtering阶段用近似，最终scoring用完整C：

公式8-10：
- S̃_row = I_TopM(u^T S_row) ⊙ S_row   # 先用rank-1近似筛出top-m的row
- S̃_col = I_TopM(t^T S_col) ⊙ S_col   # 同理筛col
- S_grid = σ_TopM(S̃_row^T × C × S̃_col)  # 在m²个candidate上用完整C精确scoring

I_TopM(·)是binary indicator function，top-m设1其余设0。⊙是element-wise乘法。

**Intuition**：用rank-1近似做"粗筛"（cheap），用完整tucker core做"精排"（expensive但只在m²个小集合上）。这其实是你熟悉的retrieval two-stage idea的精致版。

**Auxiliary loss**（公式11-12）：约束C不要退化成rank-1
- C = UΛT^T（SVD）
- L_aux = (α/(r-1)) Σ_{i=2}^r (max(0, λ_i - τ))²
- τ=0.15是margin，α=0.001是weight
- 直觉：如果λ_2, ..., λ_r都小于τ，loss为0；如果有非leading singular value超过τ，就惩罚。保证C保留rank-r信息，但允许rank-1近似主导

### 3.2 IVE: Implicit Value Expansion

这是把memory table "虚化"的trick，核心idea：**不真正存储E倍大的memory，而是用E个linear projector把physical memory V重参数化成E个virtual memory block**。

公式13：Ṽ_p = V W_p，其中W_p ∈ R^(D_v × D_v')

变量：
- V ∈ R^(N × D_v)：physical memory
- W_p：第p个reparameterization矩阵，p ∈ [1, E]
- Ṽ_p：第p个virtual memory block
- E：expansion rate，paper推荐E=4
- D_v'：virtual value维度，可以≠D_v

总virtual memory Ṽ = [Ṽ_0^T, Ṽ_1^T, ..., Ṽ_E^T]^T，大小变成E×N。

**naive实现的问题**：要先算出Ṽ_p = VW_p（EN·D_v·D_v'计算量），还要E倍GPU memory access。

**聪明实现**（公式14-15）：
- logical address扩展成triplet (i, j, p)
- 先对每个virtual block做weighted sum pooling：V^T × ŝ_p（shape: D_v）
- 再用W_p变换：W_p^T (V^T × ŝ_p)
- 总output：o = Σ_p W_p^T (V^T × ŝ_p)

额外计算从E·N·D_v·D_v' → E·B·D_v·D_v'，省了N/B倍。B通常是几百到几千，N是几百万，省1000倍以上。

**Random shuffle**：virtual memory block如果直接concatenate，同一个physical value的E个expansion会在同一列，容易被同时选中。shuffle后打散这个topology prior。

### 3.3 MCS: Multi-Core Scoring

公式16-17：
- C = Σ_{i=1}^h C^(i)，把tucker core拆成h个component
- S_tucker^(i) = S_row^T C^(i) S_col
- S_tucker = Σ_i S_tucker^(i)（aggregate后做top-m）
- 但对value的pooling用individual score：o = [ŝ^(1)T V^(1), ..., ŝ^(h)T V^(h)]^T
- V被vertically split成h份：V = [V^(1), ..., V^(h)]，V^(i) ∈ R^(Ñ × (D̃_v/h))

**Intuition**：原本一个value在D_v维度上共享一个score，现在每一段维度有自己的score。类似multi-head attention里每个head有自己的attention pattern。h=2是最佳，h=4/8开始overfit（Table 3）。

---

## 4. Improved Initialization: 一个被低估的细节

这个我重点讲，因为它体现了对"memory layer本质上是MLP"的深刻理解。

**PKM的init**：V ~ N(0, 1/D_v)，配合softmax(scores)，输出方差1/D_v。这是attention-style的init。

**UltraMem的init**：V ~ N(0, E/(2mHL))，目标是输出方差1/(2L)，这是MLP-style的init（GPT-3的init）。

变量：
- E：value expansion rate
- m：top-m激活数
- H：head数
- L：总layer数

**推导**（Appendix A）：
1. 假设candidate score ~ N(0, 1)
2. Y = mean(top-m(X_1, ..., X_n))，n个标准正态分布的top-m均值
3. E(Y)没有解析解，所以**Monte Carlo采样近似**
4. query LN weight = 1/√E(Y)
5. key LN weight = 1/√D_k
6. 这样确保candidate score期望=1

输出方差计算：
- 每个被激活value贡献V^(i) × score^(i)
- V方差E/(2mHL)，score期望1
- 经过softmax-free的weighted sum，m个激活value求和
- 经过H个head聚合
- 经过E个virtual expansion
- 最终方差 = E × (E/(2mHL)) × m × H × ... = 1/(2L)

Figure 10b/c证明这个init有效：top-1 score和output std在训练中后期稳定，naive init会发散。

---

## 5. 实验数据深度解读

### 5.1 主实验（Table 1 & Table 7）

关键对比 **UltraMem-1.6B-x12 vs MoE-1.6B-2in34**：
- Parameters: 21.41B vs 21.36B（几乎相同）
- FLOPs: 3.50G vs 3.52G（几乎相同）
- Val loss: **2.24 vs 2.30**（UltraMem低0.06，显著）
- TriviaQA: 66.38 vs 59.56（UltraMem大幅领先，knowledge retrieval任务）
- BBH-cot: 30.63 vs 29.46
- HellaSwag: 71.52 vs 67.34
- Avg: 48.26 vs 45.07

更惊人的对比：**UltraMem-1.6B-x12 (21.41B params, 3.50G FLOPs) ≈ Dense-6.5B (6.44B params, 12.88G FLOPs)**，Avg 48.26 vs 46.19。用1/3.7的FLOPs、3.3x的params（但sparse）打平dense 6.5B。

但151M规模下UltraMem还打不过MoE（Table 1: 29.99 vs 33.20），这说明**UltraMem需要足够规模才显现优势**，类似MoE也有minimum viable scale。

### 5.2 Scaling law（Figure 6a/b）

Figure 6b是核心scaling figure：
- x轴：sparse params（log scale）
- y轴：validation loss
- 不同曲线：不同sparsity（20K/40K/80K表示1/20000到1/80000的激活比例）

关键观察：
1. **loss随sparse params指数增长线性下降**（log-linear关系），和Chinchilla scaling一致
2. sparsity越小（激活比例越大），loss越低
3. 但sparsity小的代价是memory access大，所以选80K作为default trade-off

Figure 6c：UltraMem的推理时间几乎不随sparse params增长（平直），MoE线性增长。这是UltraMem的核心selling point。

### 5.3 Ablation（Table 2 & 3）

Table 2的ablation按贡献排序：
1. **+value lr decay**: -0.022 val loss（最大）
2. **+IVE**: -0.017 val loss
3. **+split big mem&skip**: -0.015 val loss
4. **+TDQKR**: -0.008 val loss
5. **+MCS**: -0.003 val loss
6. **+half vdim+proj**: -0.022 val loss（但参数略增）
7. **+rm softmax**: -0.006 val loss

Table 3的config sweep：
- IVE: E=4是sweet spot，E=16收益递减但FLOPs+26.4%
- TDQKR: r=2足够，r=3/4无显著提升
- MCS: h=2最佳，h=8反而变差（overfitting）

---

## 6. 训练优化：Megatron适配

Section 4+C+D讲的是工程，但对理解 scalability 很重要。

**3D parallelism的扩展问题**：pipeline parallelism无法处理单层参数超过单GPU memory的情况；tensor parallelism通常只在小GPU group内。所以memory table需要**shard到data parallel + tensor parallel的组合group**。

**两种partition策略**（Figure 8）：

1. **Number-wise partitioning**：按value数量N切分
   - all-to-all indices到对应device
   - lookup后all-to-all embeddings回原device
   - communication: sizeof(int) × bs × topm × (P-1)/P + sizeof(bf16) × bs × topm × v_dim × (P-1)/P
   
2. **Dimension-wise partitioning**：按value维度D_v切分
   - all-gather indices和scores
   - lookup后reduce-scatter结果
   - communication: sizeof(int) × bs × topm × (P-1) + sizeof(bf16) × bs × topm × (P-1) + sizeof(bfloat16) × bs × v_dim × (P-1)/P

Figure 9给出选择策略：v_dim和P的函数关系图，shaded area表示number-wise更优的区域。

**两个额外优化**：
1. **Fused Lookup-Reduce Operator**：把lookup和weighted sum pooling融合成一个kernel
2. **Asynchronous Execution**：memory layer和dense layer overlap执行，因为memory layer是memory-bound，dense layer是compute-bound，资源互补

---

## 7. 给Karpathy的intuition总结

我帮你把这篇paper的"mental model"提炼一下：

1. **UltraMem本质是"细粒度MoE"的极致**：MoE的expert是完整MLP（参数量2D²），PEER把expert缩小到inner dim=1，UltraMem把expert缩小到"一个D/2维向量"。粒度越细，sparsity越高，memory access越少。

2. **检索的核心矛盾**：sparsity越大，需要检索的candidate越多，但top-m计算必须cheap。product key用加法分解（O(√N)），TDQKR用乘法分解+SVD近似（O(√N)但无topology bias）。两者都把检索复杂度从O(N)降到O(√N)。

3. **IVE是参数化的"内存压缩"**：不真存EN个value，只存N个value + E个projector。retrieval时先在物理memory上pool，再project。这是"compressed retrieval"的优雅实现。

4. **Skip-layer是latency hiding的关键**：memory-bound的memory layer和compute-bound的transformer layer交替执行，让GPU的compute和memory bandwidth都打满。这个思路其实在hardware design里很常见（latency vs throughput trade-off）。

5. **Initialization的insight**：把memory layer当MLP来init，而不是当attention来init。因为memory layer memorize world knowledge，和MLP功能相同（Geva et al. 2020）。这个对init的深刻理解是paper的隐藏亮点。

6. **为什么UltraMem需要规模才显现优势**：小规模时retrieval noise大，N个value里选m个，正确value被选中的概率不高。规模大了之后，key学到的representation更structured，retrieval精度上升。这也是为什么151M规模MoE还赢，1.6B规模UltraMem反超。

7. **潜在的limitation**（paper没明说但你应该意识到）：
   - training时的all-to-all通信在P很大时仍然是瓶颈
   - top-m的近似（rank-1 SVD）在C的singular value gap小时误差大，auxiliary loss只是mitigation
   - memory table的update效率：所有N个value都要经过gradient，但只有m个被激活，剩下N-m个的update通过Momentum/Adam state，可能导致stale
   - 没有测in-context learning和long context场景下的表现

---

## 8. 与你LLM101n课程的相关性

你LLM101n里讲的"memory hierarchy"概念和这篇paper高度相关。CPU有L1/L2/L3 cache，GPU有SMEM/register/HBM。UltraMem本质上是在model architecture层面引入了一个新的memory hierarchy level：**比parameter更稀疏、比KV cache更持久、比MLP weight更细粒度**。

如果你要在课程里讲这篇paper，我建议把它放在"efficient inference"那一章，和MoE、Flash Attention、PagedAttention一起讲，作为"sparse memory access"流派的代表。

希望这个讲解帮你build了intuition。如果你想深入某个具体模块（比如TDQKR的SVD近似误差分析，或者IVE的Monte Carlo推导），我可以再展开。
