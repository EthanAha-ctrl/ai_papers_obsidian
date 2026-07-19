---
source_pdf: AUnified Sequence Parallelism Approach for Long Context Generative AI.pdf
paper_sha256: abd818e65f7da3718041044106aefb1f482f4471baaf7fdc84e93ba227d039fd
processed_at: '2026-07-18T11:32:24-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# USP: A Unified Sequence Parallelism Approach for Long Context Generative AI 深度解析

你好 Andrej！这篇 paper 来自 Tencent 的 Jiarui Fang 和 Shangchun Zhao，核心 idea 是把 DeepSpeed-Ulysses 和 Ring-Attention 这两个看似互斥的 sequence parallelism (SP) 方法统一成一个 2D mesh 拓扑，并给出 4D hybrid parallelism 的最佳实践。我会从底层机制一路 build up 到系统级直觉。

GitHub repo: https://github.com/feifeibear/long-context-attention

参考论文:
- DeepSpeed-Ulysses: https://arxiv.org/abs/2309.14509
- Ring-Attention: https://arxiv.org/abs/2310.01889
- Megatron-LM Sequence Parallelism: https://arxiv.org/abs/2205.05198
- FlashAttention: https://arxiv.org/abs/2205.14135
- Striped Attention: https://arxiv.org/abs/2311.09431

---

## 1. 问题动机：为什么需要新的 SP

long context 已是 generative AI 的硬需求。截至 2024 中：
- Claude 100K tokens
- GPT-4 128K tokens
- Gemini 1.5 Pro 10M tokens
- Sora 至少 1M visual tokens

挑战在于 self-attention 的 $QK^T$ 中 sequence 维度 $L$ 同时是矩阵乘法的公共维度，naive partition 之后做 softmax 没办法直接 reduce，所以 SP 比 DP/TP/ZeRO 难做。

### 1.1 标准 Transformer 计算回顾

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

变量说明：
- $Q, K, V \in \mathbb{R}^{L \times d}$
- $L$：sequence length
- $d$：hidden dimension
- $d = h_c \times h_s$，其中 $h_c$ 是 head count，$h_s$ 是 head size
- $N$：device 数
- $bs$：batch size

FFN：
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

关键观察：FFN 是 position-wise 的，所以对 $L$ 维切分天然友好；attention 的 softmax 依赖全局 $K$，这是 SP 的本质难点。

### 1.2 三种 SP 的演进谱系

#### (a) Megatron-LM Sequence Parallelism (TP-sp)

它本质上是 TP 的一个 activation memory 优化，并不是独立 SP。原理：把 TP 中 forward 后的 AllReduce 拆成 AllGather + ReduceScatter，让 activation 沿 $L$ 维 partition。

AllReduce = AllGather + ReduceScatter，所以通信量没变，但 activation 内存降到 $1/N$。缺点：必须依赖 TP 才能工作，独立用没意义。

#### (b) SP-Ulysses

核心 trick：对 Q, K, V 做 All2All，让 partition 维度从 $L$ 切到 $h_c$（head count 维度）。All2All 之后每个 GPU 持有完整的某些 head，可以直接用 FlashAttention 算 $\text{softmax}(QK^T)V$。

致命限制：
1. **并行度 $\le h_c$**：因为 All2All 之后是按 head 切分的，head 数不够就开不了并行度。
2. **对 GQA/MQA 几乎失效**：Llama3-8B 用 GQA，KV head 数只有 8，所以 SP-Ulysses 最大 8 度；MQA 时 KV head=1，直接挂掉。
3. **与 TP 冲突**：TP 也要切 $h_c$ 维度，两者抢同一个维度。

#### (c) SP-Ring (Ring-Attention)

可以看作 distributed FlashAttention。两重循环：外层 ring send/recv K, V 块，内层算 $Q \cdot K^T$ block。通信可以和计算 overlap。

问题：
1. **计算效率低**：softmax $(QK^T)V$ 被切成小 block，fused kernel 效率下降。即使通信完全 overlap，总时间仍可能慢于 Ulysses。
2. **causal mask 负载不均**：lower triangular matrix，均匀切分时 GPU0 几乎闲着，GPU3 满载。paper 里量化：4 GPU 时 GPU3 计算量 ≈ 7× GPU0。
3. 没有 $h_c$ 限制。

paper 在 Sec.3 中的关键 insight：**这两个方法不是对立的，可以组合**。（虽然 Ring-Attention 作者在 ICLR rebuttal 里明确说它们 mutually exclusive，paper 的反例证明这是错的。）

---

## 2. Unified SP (USP) 的核心机制

### 2.1 2D mesh 拓扑

把 SP process group 看成 2D mesh：
- 每一行跑 SP-Ulysses（All2All）
- 每一列跑 SP-Ring（P2P ring）

例子：8 个 process，$2 \times 4$ mesh，ulysses_pg 大小 2，ring_pg 大小 4。完全类比 DP × TP 的 2D 划分。

约束：ulysses_degree × ring_degree = SP_degree

边界情况：
- ulysses_degree = N → 退化成 SP-Ulysses
- ring_degree = N → 退化成 SP-Ring

所以 USP 严格包含两种方法的能力。

### 2.2 Algorithm 1 详解

```
function USP-ATTN(ulysses_pg, ring_pg, Q, K, V, scatter_idx, gather_idx):
    Q ← AllToAll4D(Q, scatter_idx, gather_idx, group=ulysses_pg)
    K ← AllToAll4D(K, scatter_idx, gather_idx, group=ulysses_pg)
    V ← AllToAll4D(V, scatter_idx, gather_idx, group=ulysses_pg)
    O ← LoadBalance-RingAttention(Q, K, V, group=ring_pg)
    O ← AllToAll4D(O, gather_idx, scatter_idx, group=ulysses_pg)
    return O
```

#### 张量 shape 演化

输入（每个 GPU 上）：
$$Q, K, V \in \mathbb{R}^{bs \times (L/N) \times h_c \times h_s}$$

forward 时 scatter_idx=1（切 $L$ 维），gather_idx=2（合并 $h_c$ 维）。

经过 AllToAll4D：
$$\mathbb{R}^{bs \times (L/N) \times h_c \times h_s} \longrightarrow \mathbb{R}^{(h_c/N_{ulysses}) \times bs \times L_{ulysses} \times d}$$

其中 $L_{ulysses} = L/N_{ulysses}$ 是经过 ulysses_pg 重排后每个 GPU 看到的"局部 sequence length"。

直觉：All2All 把"每张卡拿到一段 sequence 的所有 head"变换成"每张卡拿到所有 sequence 的一段 head"。这样 ring-attention 在 head 已经分配好的情况下，沿着 ring_pg 跨 GPU 轮转 K, V 就可以做完整的 attention。

backward 时 scatter_idx 和 gather_idx 互换，做逆变换。

#### 通信代价分解

总通信 = $N_{ulysses}$ 个 GPU 上的 All2All（在 ulysses_pg 内） + ring 上的 P2P（在 ring_pg 内）。

All2All 通信量：每个 GPU 通信 $O(bs \cdot L \cdot d)$ 量级（每张卡送出一份拿进一份）。
P2P ring 通信量：每个 GPU $N_{ring}$ 次 send/recv，每次 $O(bs \cdot (L/N_{ulysses}) \cdot d)$，总计 $O(bs \cdot L \cdot d \cdot N_{ring}/N_{ulysses})$。

对 GQA（group size = $G$）：K, V 的 head 数变 $h_c/G$，所以 K, V 通信降到 $1/G$。

### 2.3 Load Balancing for Ring（核心创新点之一）

#### 问题可视化

对 4 GPU、16 token 的 causal attention，naive 划分：

| GPU | token id | 计算的 attention block |
|-----|----------|----------------------|
| 0 | 0-3 | 仅 4×4 下三角 |
| 1 | 4-7 | 8×8（包括对前 4 个的 attention）|
| 2 | 8-11 | 12×12 |
| 3 | 12-15 | 16×16（最大）|

GPU3 几乎是 GPU0 的 7 倍计算量。

#### Solution：reorder sequence

把 16 个 token 按 zig-zag 重新分配：
- GPU0 处理 tokens [0, 1, 14, 15]：包含开头的 cheap block 和结尾的 expensive block
- GPU3 处理 tokens [5, 6, 11, 12]：中间的均匀负载

公式化：把 sequence 切成 $2 \times \text{ring\_degree}$ 个 chunk，第 $r$ 个 rank 取 chunk[$r$] 和 chunk[$2 \cdot rd - r - 1$]。

#### Algorithm 2 解读

```
function LOCALBALANCELOCALSEQ(seq, ring_pg, ulysses_pg):
    ring_degree ← ring_pg.get_world_size()
    ring_rank ← ring_pg.get_rank()
    ulysses_rank ← ulysses_pg.get_rank()
    seq_chunks ← seq.chunk(2 × ring_degree)
    reorder_seq ← concat([seq_chunks[ring_rank], 
                          seq_chunks[2*ring_degree - ring_rank - 1]])
    local_seq ← reorder_seq.chunk(ulysses_degree)[ulysses_rank]
    return local_seq
```

关键点：
1. **对 RoPE 也要做同样的 reorder**：因为 RoPE 是 element-wise 的旋转位置编码，如果 token 顺序变了，position embedding 的顺序必须跟着变，否则 attention 的几何意义破坏。paper 在 Sec.5.5 通过 loss 曲线和 DP 完全重合验证了这一点。
2. **overhead 可忽略**：reorder 在 integer token id 上做，长度 $bs \cdot L$，相对 attention 计算可忽略。
3. vs Striped Attention (https://arxiv.org/abs/2311.09431)：USP 的方法更简单，不需要改变 attention kernel 内部的 indexing。

### 2.4 异构网络适应性

paper Figure 4 展示了 USP 在异构网络下的优势：
- 高带宽段（如 NVLink within node）跑 All2All
- 低带宽段（如 Ethernet 跨节点）跑 P2P ring，可以异步 overlap

具体策略：把 ulysses_pg 放在同一节点内（NVLink），ring_pg 跨节点（Ethernet/RDMA）。这是 paper Tip 1 的实证基础。

---

## 3. 4D Parallelism 系统级分析

### 3.1 Table 2 详解

paper 把 GPT-2 风格 transformer block 的通信和内存 cost 列成一张大表。这里拆解关键变量。

#### 通信 cost 计算

Cost = 通信元素数 × algobw factor

algobw factor（n 是 group size）：
- AllReduce: $2 \cdot \frac{n-1}{n}$
- AllGather: $\frac{n-1}{n}$
- ReduceScatter: $\frac{n-1}{n}$
- AllToAll: 1

AllReduce = AllGather + ReduceScatter，所以 factor 是 2×，这就是表里 SP-Ulysses 的 all2all 标 8 (4个张量 × 2 for fwd+bwd) 而 allreduce 标 12 的原因。

Param 通信量基线：GPT-2 有 4 个 attention linear + 2 个 FFN linear，共 12 个 $O(d^2)$ 参数，所以 $12 \cdot O(d^2)$。Llama 因为 FFN intermediate size 不同，是 $9.37 \cdot O(d^2)$。

Activation 通信量基线：单个 hidden state tensor = $bs \cdot L \cdot d$。

#### 各方法对比

| 方法 | Param 通信 | Act 通信 | Split Dim | P/G | OS | Act 内存 |
|------|----------|---------|-----------|-----|-----|---------|
| SP-Ulysses | allreduce, $12O(d^2)$ | $8 \cdot \text{all2all}$, $\frac{8}{N}O(bs \cdot L \cdot d)$ | $h_c / L$ | $P+G$ | $6P$ | $A/N$ |
| SP-Ring | allreduce, $12O(d^2)$ | P2P, $4 \cdot O(bs \cdot L \cdot d)$ | $L / L$ | $P+G$ | $6P$ | $A/N$ |
| DP | allreduce, $12O(d^2)$ | 0 | $bs / bs$ | $P+G$ | $6P$ | $A/N$ |
| ZeRO1 | allgather + reducescatter, $12O(d^2)$ | 0 | $h_c / L$ | $P+G$ | $6P/N$ | $A/N$ |
| TP | 0 | $4 \cdot$ allreduce, $8 \cdot O(bs \cdot L \cdot d)$ | $h_c / d$ | $(P+G)/N$ | $6P/N$ | $\alpha A$ |
| TP-sp | 0 | $6 \cdot$ allgather + $4 \cdot$ reducescatter, $10 \cdot O(bs \cdot L \cdot d)$ | $h_c / d$ | $(P+G)/N$ | $6P/N$ | $A/N$ |

#### 关键洞察

**SP vs DP**：
- 都做 gradient allreduce，cost 都是 $12O(d^2)$
- SP 额外要做 attention 通信（all2all 或 P2P）
- DP 对 activation 不通信
- 内存上两者等价：activation 都降到 $A/N$

直觉：DP 是免费的（除了 gradient sync），SP 是有额外 cost 的。所以 **bs 够用就先用 DP**，bs 太小（比如 long context 一次只能塞 bs=1）才用 SP。这是 Tip 2 的本质。

**SP vs TP-sp**：
- TP-sp 的 act 通信 $10 \cdot O(bs \cdot L \cdot d)$ 是 constant（不随 N 减小）
- SP-Ulysses 的 act 通信 $\frac{8}{N} O(bs \cdot L \cdot d)$ 随 N 减小
- 大规模下 SP 更优
- GQA 进一步放大 SP 优势：K, V 通信降到 $1/G$，total $\frac{4}{N}O(bs \cdot L \cdot d) + \frac{4}{N}O(bs \cdot L \cdot d / G)$

**SP vs TP-sp 内存**：
- TP-sp 已经把 P/G/OS 都分到 1/N
- SP+ZeRO1/2 只分 OS 和 act，P/G 没分
- SP+ZeRO3 才能匹配 TP-sp 内存水平
- 这是为什么 DS-Ulysses 原论文用 SP+ZeRO3

### 3.2 7 个 Tips 的逻辑链条

1. **Tip 1**：用 USP 替代单独 Ring 或 Ulysses（包含两者能力 + 异构网络适应性）
2. **Tip 2**：优先 DP，bs 不够再用 SP（DP act 通信为 0，SP 有额外 cost）
3. **Tip 3**：SP 必须配 ZeRO-1/2，可选 ZeRO-3 + Offload（内存匹配）
4. **Tip 4**：SP 通信优于 TP-sp，GQA 进一步降低 SP 通信
5. **Tip 5**：从 TP-sp 切到 SP 不会增加 seq length；SP+ZeRO3 才能匹配 TP-sp
6. **Tip 6**：高 SP 度（大 ring degree）训练长序列，TP-sp 受限于 $h_c$
7. **Tip 7**：4D 顺序 TP < SP-Ulysses < SP-Ring < ZeRO-DP < PP

#### Tip 7 的直觉

process group 维度从低到高的顺序对应通信频率/敏感度从高到低：
- TP 通信最频繁（每个 transformer block 内多次 allreduce）→ 放最内层（同 NVLink domain）
- SP-Ulysses 用 All2All → 也需要高带宽，但比 TP 频率低
- SP-Ring 用 P2P，可异步 overlap → 容忍稍低带宽
- ZeRO-DP：gradient sync，可异步 → 跨节点 OK
- PP：层间通信最少，间隔大 → 最外层

---

## 4. 实验数据深度解读

### 4.1 Table 3: 8xL20 PCIe cluster

L20 是 PCIe 连接（无 NVLink），是异构网络的代理。固定 ring × ulysses = 8。

| seqlen | ulysses_deg | basic-ring (iters/s) | lb-ring (iters/s) | 提升 |
|--------|-------------|---------------------|-------------------|------|
| 8K | 8 | 57.35 | 57.10 | ~0% |
| 8K | 2 | 415.5 | 454.93 | +9.5% |
| 32K | 4 | 28.58 | 32.82 | +14.8% |
| 32K | 2 | 44.35 | 62.75 | +41.5% |
| 128K | 4 | 3.22 | 4.24 | +31.7% |
| 128K | 2 | 3.40 | 5.48 | +61.2% |

观察：
1. **短 seq (8K) 时 lb-ring 没优势**：causal mask 不均衡问题在短 seq 时不显著
2. **长 seq (128K) 时 lb-ring 大幅领先**：完美验证 load balance 的重要性
3. **ulysses_degree=4 通常是 PCIe 上的最佳**：因为 All2All 走 PCIe 带宽有限，全部 8 度 All2All 太重；混合 4+2 让 All2All 限定在 4 卡内，P2P ring 走剩下的 2 度

### 4.2 Table 4: 8xA100-SXM4 NVLink

NVLink 高带宽场景：

| seqlen | ulysses_deg | basic-ring | lb-ring |
|--------|-------------|-----------|---------|
| 32K | 8 | 135.57 | 136.38 |
| 32K | 4 | 103.53 | 132.98 |
| 32K | 2 | 91.37 | 132.98 |
| 32K | 1 | 81.99 | 113.79 |
| 128K | 8 | 2.78 | 2.79 |
| 128K | 4 | 2.02 | 2.77 |
| 128K | 2 | 1.73 | 2.89 |
| 128K | 1 | 1.63 | 2.91 |

观察：
1. **NVLink 下 ulysses=8 最优**：纯 SP-Ulysses 胜出，因为 All2All 在 NVLink 上几乎是 free 的，而 Ring 的分块计算效率损失无法用通信 overlap 补偿
2. **lb-ring 在 ulysses=1 时仍能接近 ulysses=8 的性能**：128K 时 lb-ring(ulysses=1)=2.91 vs lb-ring(ulysses=8)=2.79，差距很小。这说明 Ring 的计算效率损失被 load balance 弥补了大半。

### 4.3 Table 5: LLAMA2-7B 单节点 8xA800

| seqlen | global-bs | tp | ulysses | ring | FLOPS/GPU | MFU |
|--------|-----------|----|---------| -----|-----------|-----|
| 64K | 1 | 4 | 2 | 1 | 154.49 | 0.50 |
| 64K | 1 | 8 | 1 | 1 | 141.85 | 0.45 |
| 30K | 16 | 1 | 8 | 1 | 163.42 | 0.52 |
| 30K | 16 | 8 | 1 | 1 | 129.12 | 0.41 |

关键发现：
1. **64K 极限长度下 SP-only OOM**：必须有 TP 提供 parameter sharding 才能塞下，这是 Tip 5 的直接证据
2. **tp=4, ulysses=2 比 tp=8, ulysses=1 高 10%**：混合优于纯 TP
3. **30K 时 SP-Ulysses (ulysses=8) 比 TP-sp-only 高 26%**：通信量随 N 下降的优势在 large batch + medium seq 体现最明显

### 4.4 Table 6 & 7: LLAMA3-8B 两节点 16xA800

LLAMA3-8B 只有 8 个 KV head (GQA)，所以 ulysses × tp ≤ 8。

| seqlen | tp | ulysses | ring | FLOPS/GPU | MFU |
|--------|----|---------| ----|-----------|-----|
| 64K | 1 | 4 | 4 | 137.48 | 0.44 |
| 64K | 1 | 8 | 2 | 136.31 | 0.44 |
| 64K | 1 | 2 | 8 | 129.44 | 0.41 |
| 64K | 1 | 1 | 16 | 121.83 | 0.39 |
| 80K | 1 | 4 | 4 | 148.90 | 0.48 |
| 80K | 1 | 8 | 2 | 147.46 | 0.47 |
| 120K | 4 | 2 | 2 | 152.51 | 0.49 |
| 120K | 4 | 1 | 4 | 150.96 | 0.48 |
| 160K | 4 | 2 | 2 | 158.64 | 0.51 |
| 160K | 4 | 1 | 4 | 159.37 | 0.51 |
| 208K | 8 | 1 | 2 | 147.26 | **0.47** |

关键发现：
1. **64K/80K 最优是 ulysses=4, ring=4**：跨节点 RDMA 时 ring 跨节点，ulysses 节点内 NVLink，完美匹配异构拓扑（Tip 1 验证）
2. **Unified SP 比 SP-Ring 快 12-13%**：64K 时 USP=137 vs Ring-only(ulysses=1,ring=16)=122，差 13%；80K 类似
3. **120K 时必须引入 TP**：SP-only 会 OOM，需要 TP 分 parameter
4. **208K 是当前 setup 的极限**：tp=8, ulysses=1, ring=2，MFU 0.47（paper abstract 强调的 47% MFU）

### 4.5 收敛性验证 (Figure 6)

4 GPU 上跑 10K iterations，USP 和 DP 的 loss 曲线完全重合。这证明：
1. RoPE 的 reorder 处理正确
2. Ring attention 的 load balance reorder 数学等价
3. 没有引入数值误差

---

## 5. 直觉性总结

### 5.1 为什么 USP 有效（intuition 层面）

可以这么想 SP 的本质困难：attention 需要全局信息聚合，但你想局部化存储和计算。

- **Ulysses 的 trick**：换一个"全局化"的维度——head 维度。All2All 让每个 GPU 拿到完整 sequence 的部分 head，head 内部可以独立算 attention。这要求 head 数 ≥ GPU 数。
- **Ring 的 trick**：保留 sequence 切分，但通过 ring 传递 K, V，把全局聚合变成 streaming 计算。代价是 attention kernel 被切成小块，效率下降。
- **USP 的 trick**：把两个 trick 复合。先用 Ulysses 把 head 维度切一部分（绕开 $h_c$ 限制），剩下的 sequence 维度用 Ring 处理（绕开计算效率问题）。两者各自的短板被对方补上。

### 5.2 异构网络的对应关系

现代 GPU 集群典型拓扑：
- node 内：NVLink (800GB/s) 或 PCIe (64GB/s)
- node 间：RDMA (200-400GB/s) 或 Ethernet (100GB/s)

USP 的映射策略：
- ulysses_pg ↔ 高带宽域（同节点）
- ring_pg ↔ 低带宽域（跨节点），但 P2P 可异步

这与 NCCL 的 topology awareness 天然契合。可以认为是 SP-Ulysses 和 SP-Ring 的拓扑敏感组合。

### 5.3 4D parallelism 的设计哲学

可以从"通信代价比"理解 Tip 7 的顺序：
- TP：act 通信最高频（每 layer 多次 allreduce，cost $8 \cdot O(bs \cdot L \cdot d)$）→ 必须最快网络
- SP-Ulysses：act 通信 all2all，cost $\frac{8}{N} \cdot O(bs \cdot L \cdot d)$，频率比 TP 低
- SP-Ring：act 通信 P2P，可 overlap，对延迟敏感但对带宽要求低
- ZeRO-DP：gradient 通信，可异步 overlap，频率最低（每 step 一次）
- PP：micro-batch 间通信，频率介于 SP 和 DP 之间，但通信量小

直觉：通信频率/敏感性 从高到低 = 维度 从低到高。

### 5.4 Llama3-8B 案例的启示

Llama3-8B 用 GQA，KV head=8，是 SP-Ulysses 的 worst case。但 paper 用 USP 仍然跑到了 208K 序列长度，47% MFU。这说明：
1. **USP 把 GQA/MQA 的限制转化为 ring degree 上的额外并行**：ulysses=1 时退化成纯 Ring，但仍可用
2. **结合 TP 解决内存瓶颈**：长 seq 时 parameter 内存也是瓶颈，TP 提供干净的 parameter sharding
3. **load balance ring 让 causal attention 在长 seq 下不再浪费 50% 算力**

---

## 6. 未来工作与开放问题

paper Sec.6 提了三个方向：

### 6.1 大规模集群 SP

10K+ GPU 训练（MegaScale, Llama3）目前没用 SP，因为这些项目启动于 2023.11 之前（SP 方法未成熟）。

SP 的潜在优势：
- MegaScale 用大 global batch size 增加 DP 度，但大 batch 影响收敛。改用增加 SP 度可以保持小 batch。
- SP-Ring 度可任意增加，TP 度受限于 head 数。理论上 SP 可以扩展到超大集群训练超长 context。

### 6.2 SP + ZeRO-3

Megatron-LM 官方不支持 ZeRO-3，因为 TP-sp 已经分了 P/G。但 SP 没有 TP 时 ZeRO-3 必要。这是工程上的 next step。

### 6.3 SP + MoE

Attention 上的 SP 与 FFN 上的 expert parallelism 解耦。关键：Attention 和 FFN 之间的 All2All 通信要重新设计，确保 SP shard 和 EP shard 不冲突。

相关参考：
- DeepSpeed-MoE: https://arxiv.org/abs/2101.03961
- GShard: https://arxiv.org/abs/2006.16668

---

## 7. 批判性思考与潜在问题

为了 build your intuition，我也指出一些 paper 没充分讨论的点：

1. **All2All 的实际性能**：paper 假设 All2All 在 NVLink 上接近 free。但 All2All 在实际硬件上是 $\log N$ 步骤的 hierarchical 实现，对 latency 敏感。在 batch size 极小（如 long context 训练 bs=1）时，All2All 的 latency 可能主导。

2. **Ring overlap 的实际效果**：paper 假设 P2P 可以完全 overlap 计算。但 blockwise FlashAttention 的计算时间随 block size 变化，overlap 比例不是 100%。这是为什么 NVLink 上 Ulysses(8) 仍优于混合配置——overlap 不完美。

3. **load balance reorder 的边界**：reorder 在 token id 层面做，对 RoPE OK，但对 ALiBi 或 absolute positional encoding 是否 OK 需要单独验证。paper 没讨论。

4. **SP-Ring 的 backward**：causal mask 在 backward 中也有同样的不对称问题，paper 的 load balance 是否对 backward 也平衡没明确说。

5. **MoE 兼容性的细节**：Sec.6 提到 All2All 要重新设计，但具体如何避免 SP shard 和 EP shard 冲突（两者都用 All2All，是否会重叠通信）是个开放问题。

参考相关工作的深入阅读：
- Megatron-LM: https://arxiv.org/abs/1909.08053
- ZeRO: https://arxiv.org/abs/1910.02054
- GQA: https://arxiv.org/abs/2305.13245
- MQA: https://arxiv.org/abs/1911.02150
- RoPE: https://arxiv.org/abs/2104.09864
- MegaScale: https://arxiv.org/abs/2402.15627
- Llama 3: https://ai.meta.com/blog/meta-llama-3/

---

## 8. 关键 Takeaways（一句话总结）

1. **USP = Ulysses × Ring 的 2D mesh**，两个方法不是互斥而是互补
2. **Load balance reorder** 是 Ring 在 causal attention 下能 scale 的关键 trick
3. **异构网络友好**：All2All 走高带宽域，P2P ring 走低带宽域
4. **GQA/MQA 友好**：通过 ring degree 绕开 $h_c$ 限制
5. **4D 顺序**：TP < SP-Ulysses < SP-Ring < ZeRO-DP < PP，按通信敏感性递减排列
6. **Llama3-8B 在 16xA800 上跑 208K seq len，47% MFU**，是 long context 训练的 SOTA 工程实践

这篇 paper 的最大贡献不在算法创新，而在 **systematic 的工程整合**：把两个独立工作的优势组合，给出 4D 并行的完整决策树，并通过实验验证每个 Tip。这正是 long context LLM training 在工程上从 "work" 到 "work well" 的关键一跳。
